import Mathlib

namespace NUMINAMATH_CALUDE_number_forms_and_products_l3625_362573

theorem number_forms_and_products (n m : ℕ) :
  -- Part 1: Any number not divisible by 2 or 3 is of the form 6n + 1 or 6n + 5
  (∀ k : ℤ, (¬(2 ∣ k) ∧ ¬(3 ∣ k)) → (∃ n : ℕ, k = 6*n + 1 ∨ k = 6*n + 5)) ∧
  
  -- Part 2: Product of two numbers of form 6n + 1 or 6n + 5 is of form 6m + 1
  ((6*n + 1) * (6*m + 1) ≡ 1 [MOD 6]) ∧
  ((6*n + 5) * (6*m + 5) ≡ 1 [MOD 6]) ∧
  
  -- Part 3: Product of 6n + 1 and 6n + 5 is of form 6m + 5
  ((6*n + 1) * (6*m + 5) ≡ 5 [MOD 6]) :=
by sorry


end NUMINAMATH_CALUDE_number_forms_and_products_l3625_362573


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l3625_362504

theorem consecutive_integers_median (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) :
  let median := S / n
  median = 729 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l3625_362504


namespace NUMINAMATH_CALUDE_rachel_plant_arrangement_count_l3625_362596

/-- Represents the types of plants Rachel has -/
inductive Plant
| Basil
| Aloe
| Cactus

/-- Represents the colors of lamps Rachel has -/
inductive LampColor
| White
| Red
| Blue

/-- A configuration of plants under lamps -/
structure Configuration where
  plantUnderLamp : Plant → LampColor

/-- Checks if a configuration is valid according to the given conditions -/
def isValidConfiguration (config : Configuration) : Prop :=
  -- Each plant is under exactly one lamp
  (∀ p : Plant, ∃! l : LampColor, config.plantUnderLamp p = l) ∧
  -- No lamp is used for just one plant unless it's the red one
  (∀ l : LampColor, l ≠ LampColor.Red → (∃ p₁ p₂ : Plant, p₁ ≠ p₂ ∧ config.plantUnderLamp p₁ = l ∧ config.plantUnderLamp p₂ = l))

/-- The number of valid configurations -/
def validConfigurationsCount : ℕ := sorry

theorem rachel_plant_arrangement_count :
  validConfigurationsCount = 4 := by sorry

end NUMINAMATH_CALUDE_rachel_plant_arrangement_count_l3625_362596


namespace NUMINAMATH_CALUDE_sum_in_base_9_l3625_362591

/-- Converts a base-9 number to base-10 --/
def base9To10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 9^i) 0

/-- Converts a base-10 number to base-9 --/
def base10To9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- The sum of 263₉, 504₉, and 72₉ in base 9 is 850₉ --/
theorem sum_in_base_9 :
  base10To9 (base9To10 [3, 6, 2] + base9To10 [4, 0, 5] + base9To10 [2, 7]) = [0, 5, 8] :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base_9_l3625_362591


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3625_362527

theorem gcd_of_powers_of_two : Nat.gcd (2^1010 - 1) (2^1000 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3625_362527


namespace NUMINAMATH_CALUDE_tablet_savings_l3625_362592

/-- Proves that buying a tablet in cash saves $70 compared to an installment plan -/
theorem tablet_savings : 
  let cash_price : ℕ := 450
  let down_payment : ℕ := 100
  let first_four_months : ℕ := 4 * 40
  let next_four_months : ℕ := 4 * 35
  let last_four_months : ℕ := 4 * 30
  let total_installment : ℕ := down_payment + first_four_months + next_four_months + last_four_months
  total_installment - cash_price = 70 := by
  sorry

end NUMINAMATH_CALUDE_tablet_savings_l3625_362592


namespace NUMINAMATH_CALUDE_express_x_in_terms_of_y_l3625_362567

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  x = (4 * y + 8) / 3 := by
  sorry

end NUMINAMATH_CALUDE_express_x_in_terms_of_y_l3625_362567


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l3625_362541

theorem prime_factorization_sum (a b c : ℕ+) : 
  2^(a : ℕ) * 3^(b : ℕ) * 5^(c : ℕ) = 36000 → 3*(a : ℕ) + 4*(b : ℕ) + 6*(c : ℕ) = 41 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l3625_362541


namespace NUMINAMATH_CALUDE_equation_solutions_l3625_362554

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop :=
  x^4 = 2*x^2 + (floor x)

/-- The set of solutions to the equation -/
def solution_set : Set ℝ :=
  {0, Real.sqrt (1 + Real.sqrt 2), -1}

/-- Theorem stating that the solution set contains exactly the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3625_362554


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3625_362538

theorem complex_equation_solution (z : ℂ) : 
  (Complex.I * z = 4 + 3 * Complex.I) → (z = 3 - 4 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3625_362538


namespace NUMINAMATH_CALUDE_a_lt_2_necessary_not_sufficient_for_a_sq_lt_4_l3625_362582

theorem a_lt_2_necessary_not_sufficient_for_a_sq_lt_4 :
  (∀ a : ℝ, a^2 < 4 → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_a_lt_2_necessary_not_sufficient_for_a_sq_lt_4_l3625_362582


namespace NUMINAMATH_CALUDE_add_45_minutes_to_10_20_l3625_362540

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, sorry⟩

theorem add_45_minutes_to_10_20 :
  addMinutes ⟨10, 20, sorry⟩ 45 = ⟨11, 5, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_add_45_minutes_to_10_20_l3625_362540


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l3625_362570

theorem quadratic_roots_existence (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    3 * x₁^2 - 2*(a+b+c)*x₁ + a*b + b*c + a*c = 0 ∧
    3 * x₂^2 - 2*(a+b+c)*x₂ + a*b + b*c + a*c = 0 ∧
    a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l3625_362570


namespace NUMINAMATH_CALUDE_vector_properties_l3625_362588

/-- Prove properties of vectors a, b, and c in a plane -/
theorem vector_properties (a b c : ℝ × ℝ) (θ : ℝ) : 
  a = (1, -2) →
  ‖c‖ = 2 * Real.sqrt 5 →
  ∃ (k : ℝ), c = k • a →
  ‖b‖ = 1 →
  (a + b) • (a - 2 • b) = 0 →
  (c = (-2, 4) ∨ c = (2, -4)) ∧ 
  Real.cos θ = (3 * Real.sqrt 5) / 5 :=
by sorry


end NUMINAMATH_CALUDE_vector_properties_l3625_362588


namespace NUMINAMATH_CALUDE_line_through_intersection_parallel_to_given_l3625_362562

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 4 * x + y - 4 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

-- Define the resulting line
def result_line (x y : ℝ) : Prop := 4 * x + y - 10 = 0

-- Theorem statement
theorem line_through_intersection_parallel_to_given :
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧
  ∀ (x y : ℝ), (∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ 
                parallel_line (x₀ + 1) (y₀ + k)) ↔
               result_line x y := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_parallel_to_given_l3625_362562


namespace NUMINAMATH_CALUDE_mom_tshirt_packages_l3625_362509

/-- The number of packages mom will have when buying t-shirts -/
def packages_bought (shirts_per_package : ℕ) (total_shirts : ℕ) : ℕ :=
  total_shirts / shirts_per_package

/-- Theorem: Mom will have 3 packages when buying 39 t-shirts sold in packages of 13 -/
theorem mom_tshirt_packages :
  packages_bought 13 39 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_packages_l3625_362509


namespace NUMINAMATH_CALUDE_translated_parabola_vertex_l3625_362569

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The translation amount to the right -/
def translation : ℝ := 2

/-- The new parabola function after translation -/
def f_translated (x : ℝ) : ℝ := f (x - translation)

/-- Theorem stating the coordinates of the vertex of the translated parabola -/
theorem translated_parabola_vertex :
  ∃ (x y : ℝ), x = 4 ∧ y = -1 ∧
  ∀ (t : ℝ), f_translated t ≥ f_translated x :=
sorry

end NUMINAMATH_CALUDE_translated_parabola_vertex_l3625_362569


namespace NUMINAMATH_CALUDE_wednesday_pages_proof_l3625_362517

def total_pages : ℕ := 158
def monday_pages : ℕ := 23
def tuesday_pages : ℕ := 38
def thursday_pages : ℕ := 12

def friday_pages : ℕ := 2 * thursday_pages

theorem wednesday_pages_proof :
  total_pages - (monday_pages + tuesday_pages + thursday_pages + friday_pages) = 61 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_pages_proof_l3625_362517


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l3625_362520

theorem min_shift_for_symmetry (m : ℝ) : 
  m > 0 ∧ 
  (∀ x : ℝ, 2 * Real.sin (x + m + π/3) = 2 * Real.sin (-x + m + π/3)) →
  m ≥ π/6 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l3625_362520


namespace NUMINAMATH_CALUDE_quadratic_ratio_l3625_362586

/-- Given a quadratic polynomial of the form x^2 + 1800x + 2700,
    prove that when written as (x + b)^2 + c, the ratio c/b equals -897 -/
theorem quadratic_ratio (x : ℝ) :
  let f := fun x => x^2 + 1800*x + 2700
  ∃ b c : ℝ, (∀ x, f x = (x + b)^2 + c) ∧ c / b = -897 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l3625_362586


namespace NUMINAMATH_CALUDE_fence_height_l3625_362556

theorem fence_height (wall_length : ℕ) (num_walls : ℕ) (wall_depth : ℕ) (total_bricks : ℕ) : 
  wall_length = 20 →
  num_walls = 4 →
  wall_depth = 2 →
  total_bricks = 800 →
  (total_bricks / (wall_length * num_walls * wall_depth) : ℕ) = 5 := by
sorry

end NUMINAMATH_CALUDE_fence_height_l3625_362556


namespace NUMINAMATH_CALUDE_ada_paul_scores_l3625_362559

/-- Ada and Paul's test scores problem -/
theorem ada_paul_scores (A1 A2 A3 P1 P2 P3 : ℤ) 
  (h1 : A1 > P1)
  (h2 : A2 = P2 + 4)
  (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)
  (h4 : P3 = A3 + 26) :
  A1 - P1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ada_paul_scores_l3625_362559


namespace NUMINAMATH_CALUDE_cube_surface_area_l3625_362587

/-- The surface area of a cube with edge length 3a is 54a² -/
theorem cube_surface_area (a : ℝ) : 
  6 * (3 * a)^2 = 54 * a^2 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3625_362587


namespace NUMINAMATH_CALUDE_derivative_x_minus_sin_l3625_362530

/-- The derivative of x - sin(x) is 1 - cos(x) -/
theorem derivative_x_minus_sin (x : ℝ) : 
  deriv (fun x => x - Real.sin x) x = 1 - Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_minus_sin_l3625_362530


namespace NUMINAMATH_CALUDE_conversion_theorem_l3625_362534

-- Define conversion rates
def meters_per_km : ℝ := 1000
def minutes_per_hour : ℝ := 60

-- Problem 1: Convert 70 kilometers and 50 meters to kilometers
def problem1 (km : ℝ) (m : ℝ) : Prop :=
  km + m / meters_per_km = 70.05

-- Problem 2: Convert 3.6 hours to hours and minutes
def problem2 (h : ℝ) : Prop :=
  ∃ (whole_hours : ℕ) (minutes : ℕ),
    h = whole_hours + (minutes : ℝ) / minutes_per_hour ∧
    whole_hours = 3 ∧
    minutes = 36

theorem conversion_theorem :
  problem1 70 50 ∧ problem2 3.6 := by sorry

end NUMINAMATH_CALUDE_conversion_theorem_l3625_362534


namespace NUMINAMATH_CALUDE_area_outside_inscribed_angle_l3625_362515

theorem area_outside_inscribed_angle (R : ℝ) (h : R = 12) :
  let θ : ℝ := 120 * π / 180
  let sector_area := θ / (2 * π) * π * R^2
  let triangle_area := 1/2 * R^2 * Real.sin θ
  sector_area - triangle_area = 48 * π - 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_inscribed_angle_l3625_362515


namespace NUMINAMATH_CALUDE_probability_no_dessert_l3625_362590

def probability_dessert : ℝ := 0.60
def probability_dessert_no_coffee : ℝ := 0.20

theorem probability_no_dessert :
  1 - probability_dessert = 0.40 :=
sorry

end NUMINAMATH_CALUDE_probability_no_dessert_l3625_362590


namespace NUMINAMATH_CALUDE_prob_one_student_two_books_is_eight_ninths_l3625_362597

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of books --/
def num_books : ℕ := 4

/-- The probability of exactly one student receiving two different books --/
def prob_one_student_two_books : ℚ := 8/9

/-- Theorem stating that the probability of exactly one student receiving two different books
    when four distinct books are randomly gifted to three students is equal to 8/9 --/
theorem prob_one_student_two_books_is_eight_ninths :
  prob_one_student_two_books = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_one_student_two_books_is_eight_ninths_l3625_362597


namespace NUMINAMATH_CALUDE_infinitely_many_n_with_large_prime_divisor_l3625_362523

theorem infinitely_many_n_with_large_prime_divisor :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ p : ℕ, Nat.Prime p ∧ p > 2*n + Real.sqrt (2*n) ∧ p ∣ (n^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_with_large_prime_divisor_l3625_362523


namespace NUMINAMATH_CALUDE_z_is_in_fourth_quadrant_l3625_362577

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, z * (1 + i)}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem z_is_in_fourth_quadrant (z : ℂ) :
  M z ∪ N = {1, 2, 3, 4} → z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_z_is_in_fourth_quadrant_l3625_362577


namespace NUMINAMATH_CALUDE_cucumber_weight_problem_l3625_362578

/-- Given cucumbers that are initially 99% water by weight, then 96% water by weight after
    evaporation with a new weight of 25 pounds, prove that the initial weight was 100 pounds. -/
theorem cucumber_weight_problem (initial_water_percent : ℝ) (final_water_percent : ℝ) (final_weight : ℝ) :
  initial_water_percent = 0.99 →
  final_water_percent = 0.96 →
  final_weight = 25 →
  ∃ (initial_weight : ℝ), initial_weight = 100 ∧
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * final_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_problem_l3625_362578


namespace NUMINAMATH_CALUDE_min_value_P_l3625_362524

theorem min_value_P (a b : ℝ) (h : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ t : ℝ, a * t^3 - t^2 + b * t - 1 = 0 ↔ t = x ∨ t = y ∨ t = z)) :
  ∀ P : ℝ, P = (5 * a^2 - 3 * a * b + 2) / (a^2 * (b - a)) → P ≥ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_P_l3625_362524


namespace NUMINAMATH_CALUDE_tangent_line_at_one_unique_solution_l3625_362574

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a - 1/x - Real.log x

-- Part 1: Tangent line when a = 2
theorem tangent_line_at_one (x : ℝ) :
  let a : ℝ := 2
  let f' : ℝ → ℝ := λ x => (1 - x) / (x^2)
  (f' 1 = 0) ∧ (f a 1 = 1) → (λ y => y = 1) = (λ y => y = f' 1 * (x - 1) + f a 1) :=
sorry

-- Part 2: Unique solution when a = 1
theorem unique_solution :
  (∃! x : ℝ, f 1 x = 0) ∧ (∀ a : ℝ, a ≠ 1 → ¬(∃! x : ℝ, f a x = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_unique_solution_l3625_362574


namespace NUMINAMATH_CALUDE_shirley_cases_needed_l3625_362510

-- Define the number of boxes sold
def boxes_sold : ℕ := 54

-- Define the number of boxes per case
def boxes_per_case : ℕ := 6

-- Define the number of cases needed
def cases_needed : ℕ := boxes_sold / boxes_per_case

-- Theorem statement
theorem shirley_cases_needed : cases_needed = 9 := by
  sorry

end NUMINAMATH_CALUDE_shirley_cases_needed_l3625_362510


namespace NUMINAMATH_CALUDE_pill_bottle_duration_l3625_362594

/-- Proves that a bottle of 60 pills will last 8 months if 1/4 of a pill is consumed daily -/
theorem pill_bottle_duration (pills_per_bottle : ℕ) (daily_consumption : ℚ) (days_per_month : ℕ) :
  pills_per_bottle = 60 →
  daily_consumption = 1/4 →
  days_per_month = 30 →
  (pills_per_bottle : ℚ) / daily_consumption / days_per_month = 8 := by
  sorry

end NUMINAMATH_CALUDE_pill_bottle_duration_l3625_362594


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3625_362551

theorem negation_of_universal_statement :
  (¬ ∀ x ∈ Set.Icc 0 2, x^2 - 2*x ≤ 0) ↔ (∃ x ∈ Set.Icc 0 2, x^2 - 2*x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3625_362551


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l3625_362565

theorem sum_of_two_equals_third (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l3625_362565


namespace NUMINAMATH_CALUDE_self_inverse_fourth_power_congruence_l3625_362571

theorem self_inverse_fourth_power_congruence (n : ℕ+) (a : ℤ) 
  (h : a * a ≡ 1 [ZMOD n]) : 
  a^4 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_fourth_power_congruence_l3625_362571


namespace NUMINAMATH_CALUDE_selling_price_theorem_l3625_362564

/-- Calculates the selling price per tire given production costs and profit -/
def selling_price_per_tire (cost_per_batch : ℝ) (cost_per_tire : ℝ) (batch_size : ℕ) (profit_per_tire : ℝ) : ℝ :=
  cost_per_tire + profit_per_tire

/-- Theorem: The selling price per tire is the sum of cost per tire and profit per tire -/
theorem selling_price_theorem (cost_per_batch : ℝ) (cost_per_tire : ℝ) (batch_size : ℕ) (profit_per_tire : ℝ) :
  selling_price_per_tire cost_per_batch cost_per_tire batch_size profit_per_tire = cost_per_tire + profit_per_tire :=
by sorry

#eval selling_price_per_tire 22500 8 15000 10.5

end NUMINAMATH_CALUDE_selling_price_theorem_l3625_362564


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3625_362581

theorem polynomial_division_theorem (x : ℝ) : 
  (x^4 + 13) = (x - 1) * (x^3 + x^2 + x + 1) + 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3625_362581


namespace NUMINAMATH_CALUDE_prime_factors_count_four_equals_two_squared_seven_is_prime_eleven_is_prime_l3625_362516

/-- The total number of prime factors in the expression (4)^11 × (7)^5 × (11)^2 -/
def totalPrimeFactors : ℕ := 29

/-- The exponent of 4 in the expression -/
def exponent4 : ℕ := 11

/-- The exponent of 7 in the expression -/
def exponent7 : ℕ := 5

/-- The exponent of 11 in the expression -/
def exponent11 : ℕ := 2

theorem prime_factors_count :
  totalPrimeFactors = 2 * exponent4 + exponent7 + exponent11 := by
  sorry

/-- 4 is equal to 2^2 -/
theorem four_equals_two_squared : (4 : ℕ) = 2^2 := by
  sorry

/-- 7 is a prime number -/
theorem seven_is_prime : Nat.Prime 7 := by
  sorry

/-- 11 is a prime number -/
theorem eleven_is_prime : Nat.Prime 11 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_four_equals_two_squared_seven_is_prime_eleven_is_prime_l3625_362516


namespace NUMINAMATH_CALUDE_min_value_expression_l3625_362518

theorem min_value_expression (a b : ℝ) (hb : b ≠ 0) :
  a^2 + b^2 + a/b + 1/b^2 ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3625_362518


namespace NUMINAMATH_CALUDE_cricket_average_l3625_362546

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (initial_average : ℕ) : 
  innings = 20 →
  next_runs = 200 →
  increase = 8 →
  (innings * initial_average + next_runs) / (innings + 1) = initial_average + increase →
  initial_average = 32 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l3625_362546


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l3625_362563

/-- The circle with equation x^2 + y^2 = 20 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 20}

/-- The point M (2, -4) -/
def M : ℝ × ℝ := (2, -4)

/-- The proposed tangent line with equation x - 2y - 10 = 0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 - 10 = 0}

theorem tangent_line_is_correct :
  ∀ p ∈ TangentLine,
    (p ∈ Circle → p = M) ∧
    (∀ q ∈ Circle, q ≠ M → (p.1 - M.1) * (q.1 - M.1) + (p.2 - M.2) * (q.2 - M.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l3625_362563


namespace NUMINAMATH_CALUDE_dance_to_electropop_ratio_l3625_362593

def total_requests : ℕ := 30
def electropop_requests : ℕ := total_requests / 2
def rock_requests : ℕ := 5
def oldies_requests : ℕ := rock_requests - 3
def dj_choice_requests : ℕ := oldies_requests / 2
def rap_requests : ℕ := 2

def non_electropop_requests : ℕ := rock_requests + oldies_requests + dj_choice_requests + rap_requests

def dance_music_requests : ℕ := total_requests - non_electropop_requests

theorem dance_to_electropop_ratio :
  dance_music_requests = electropop_requests :=
sorry

end NUMINAMATH_CALUDE_dance_to_electropop_ratio_l3625_362593


namespace NUMINAMATH_CALUDE_james_balloons_l3625_362598

-- Define the number of balloons Amy has
def amy_balloons : ℕ := 513

-- Define the difference in balloons between James and Amy
def difference : ℕ := 208

-- Theorem statement
theorem james_balloons : amy_balloons + difference = 721 := by
  sorry

end NUMINAMATH_CALUDE_james_balloons_l3625_362598


namespace NUMINAMATH_CALUDE_inscribed_equilateral_triangle_side_length_l3625_362528

theorem inscribed_equilateral_triangle_side_length 
  (diameter : ℝ) (side_length : ℝ) 
  (h1 : diameter = 2000) 
  (h2 : side_length = 1732 + 1/20) : 
  side_length = diameter / 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_equilateral_triangle_side_length_l3625_362528


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3625_362580

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ := num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3625_362580


namespace NUMINAMATH_CALUDE_solution_of_system_l3625_362576

theorem solution_of_system (x y : ℝ) : 
  (1 / Real.sqrt (1 + 2 * x^2) + 1 / Real.sqrt (1 + 2 * y^2) = 2 / Real.sqrt (1 + 2 * x * y)) ∧
  (Real.sqrt (x * (1 - 2 * x)) + Real.sqrt (y * (1 - 2 * y)) = 2 / 9) →
  (x = y) ∧ 
  ((x = 1 / 4 + Real.sqrt 73 / 36) ∨ (x = 1 / 4 - Real.sqrt 73 / 36)) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l3625_362576


namespace NUMINAMATH_CALUDE_carnation_tulip_difference_l3625_362568

theorem carnation_tulip_difference :
  let carnations : ℕ := 13
  let tulips : ℕ := 7
  carnations - tulips = 6 :=
by sorry

end NUMINAMATH_CALUDE_carnation_tulip_difference_l3625_362568


namespace NUMINAMATH_CALUDE_A_equals_set_l3625_362539

def A : Set ℤ := {x | -1 < |x - 1| ∧ |x - 1| < 2}

theorem A_equals_set : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_equals_set_l3625_362539


namespace NUMINAMATH_CALUDE_original_price_with_loss_l3625_362543

/-- Proves that given an article sold for 300 with a 50% loss, the original price was 600 -/
theorem original_price_with_loss (selling_price : ℝ) (loss_percent : ℝ) : 
  selling_price = 300 → loss_percent = 50 → 
  ∃ original_price : ℝ, 
    original_price = 600 ∧ 
    selling_price = original_price * (1 - loss_percent / 100) := by
  sorry

end NUMINAMATH_CALUDE_original_price_with_loss_l3625_362543


namespace NUMINAMATH_CALUDE_rickeys_race_time_l3625_362557

/-- Given that Prejean's speed is three-quarters of Rickey's speed and their combined race time is 70 minutes, prove that Rickey's race time is 30 minutes. -/
theorem rickeys_race_time (prejean_speed rickey_speed : ℝ) (total_time : ℝ) : 
  prejean_speed = (3/4) * rickey_speed →
  total_time = 70 →
  (rickey_speed * (1/rickey_speed)) + (rickey_speed * (1/prejean_speed)) = total_time →
  rickey_speed * (1/rickey_speed) = 30 := by
sorry

end NUMINAMATH_CALUDE_rickeys_race_time_l3625_362557


namespace NUMINAMATH_CALUDE_cows_died_last_year_l3625_362542

/-- Represents the number of cows in a herd and its changes over two years -/
structure CowHerd where
  initial : ℕ
  sold_last_year : ℕ
  increased_this_year : ℕ
  bought_this_year : ℕ
  gifted_this_year : ℕ
  current : ℕ

/-- Calculates the number of cows that died last year -/
def cows_died (herd : CowHerd) : ℕ :=
  herd.initial - herd.sold_last_year + herd.increased_this_year + herd.bought_this_year + herd.gifted_this_year - herd.current

/-- Theorem: The number of cows that died last year is 31 -/
theorem cows_died_last_year (herd : CowHerd) 
  (h_initial : herd.initial = 39)
  (h_sold : herd.sold_last_year = 6)
  (h_increased : herd.increased_this_year = 24)
  (h_bought : herd.bought_this_year = 43)
  (h_gifted : herd.gifted_this_year = 8)
  (h_current : herd.current = 83) :
  cows_died herd = 31 := by
  sorry

#eval cows_died { initial := 39, sold_last_year := 6, increased_this_year := 24, bought_this_year := 43, gifted_this_year := 8, current := 83 }

end NUMINAMATH_CALUDE_cows_died_last_year_l3625_362542


namespace NUMINAMATH_CALUDE_books_gotten_rid_of_correct_l3625_362531

/-- Calculates the number of coloring books gotten rid of in a sale -/
def books_gotten_rid_of (initial_stock : ℕ) (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  initial_stock - (num_shelves * books_per_shelf)

/-- Proves that the number of coloring books gotten rid of is correct -/
theorem books_gotten_rid_of_correct (initial_stock : ℕ) (num_shelves : ℕ) (books_per_shelf : ℕ) :
  books_gotten_rid_of initial_stock num_shelves books_per_shelf =
  initial_stock - (num_shelves * books_per_shelf) :=
by sorry

#eval books_gotten_rid_of 40 5 4

end NUMINAMATH_CALUDE_books_gotten_rid_of_correct_l3625_362531


namespace NUMINAMATH_CALUDE_student_comprehensive_score_l3625_362508

/-- Represents the scores and weights for a science and technology innovation competition. -/
structure CompetitionScores where
  theoretical_knowledge : ℝ
  innovative_design : ℝ
  on_site_presentation : ℝ
  theoretical_weight : ℝ
  innovative_weight : ℝ
  on_site_weight : ℝ

/-- Calculates the comprehensive score for a given set of competition scores. -/
def comprehensive_score (scores : CompetitionScores) : ℝ :=
  scores.theoretical_knowledge * scores.theoretical_weight +
  scores.innovative_design * scores.innovative_weight +
  scores.on_site_presentation * scores.on_site_weight

/-- Theorem stating that the student's comprehensive score is 90 points. -/
theorem student_comprehensive_score :
  let scores : CompetitionScores := {
    theoretical_knowledge := 95,
    innovative_design := 88,
    on_site_presentation := 90,
    theoretical_weight := 0.2,
    innovative_weight := 0.5,
    on_site_weight := 0.3
  }
  comprehensive_score scores = 90 := by
  sorry


end NUMINAMATH_CALUDE_student_comprehensive_score_l3625_362508


namespace NUMINAMATH_CALUDE_service_provider_selection_l3625_362547

theorem service_provider_selection (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 :=
by sorry

end NUMINAMATH_CALUDE_service_provider_selection_l3625_362547


namespace NUMINAMATH_CALUDE_larger_triangle_perimeter_l3625_362549

/-- Two similar triangles where one has side lengths 12, 12, and 15, and the other has longest side 30 -/
structure SimilarTriangles where
  /-- Side lengths of the smaller triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Longest side of the larger triangle -/
  longest_side : ℝ
  /-- The smaller triangle is isosceles -/
  h_isosceles : a = b
  /-- The side lengths of the smaller triangle -/
  h_sides : a = 12 ∧ c = 15
  /-- The longest side of the larger triangle -/
  h_longest : longest_side = 30

/-- The perimeter of the larger triangle is 78 -/
theorem larger_triangle_perimeter (t : SimilarTriangles) : 
  (t.longest_side / t.c) * (t.a + t.b + t.c) = 78 := by
  sorry

end NUMINAMATH_CALUDE_larger_triangle_perimeter_l3625_362549


namespace NUMINAMATH_CALUDE_stop_duration_l3625_362575

/-- Calculates the duration of a stop given the total distance, speed, and total travel time. -/
theorem stop_duration (distance : ℝ) (speed : ℝ) (total_time : ℝ) 
  (h1 : distance = 360) 
  (h2 : speed = 60) 
  (h3 : total_time = 7) :
  total_time - distance / speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_stop_duration_l3625_362575


namespace NUMINAMATH_CALUDE_simplify_fraction_l3625_362548

theorem simplify_fraction : (144 : ℚ) / 12672 = 1 / 88 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3625_362548


namespace NUMINAMATH_CALUDE_divisibility_by_133_l3625_362536

theorem divisibility_by_133 (n : ℕ) : ∃ k : ℤ, 11^(n+2) + 12^(2*n+1) = 133 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_133_l3625_362536


namespace NUMINAMATH_CALUDE_expression_value_l3625_362533

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3625_362533


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3625_362560

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x - 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y - 2 * y + 12 = 0 → y = x) ↔ 
  (k = 10 ∨ k = -14) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3625_362560


namespace NUMINAMATH_CALUDE_evaluate_expression_l3625_362525

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -6) :
  x^2 * y^3 * z^2 = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3625_362525


namespace NUMINAMATH_CALUDE_gcd_459_357_l3625_362585

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3625_362585


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3625_362555

theorem root_sum_theorem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → 
  b^2 - 5*b + 6 = 0 → 
  a^3 + a^4*b^2 + a^2*b^4 + b^3 + a*b^3 + b*a^3 = 683 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3625_362555


namespace NUMINAMATH_CALUDE_difference_of_squares_l3625_362537

theorem difference_of_squares (a : ℝ) : a * a - (a - 1) * (a + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3625_362537


namespace NUMINAMATH_CALUDE_tan_negative_five_pi_sixths_l3625_362545

theorem tan_negative_five_pi_sixths : 
  Real.tan (-5 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_five_pi_sixths_l3625_362545


namespace NUMINAMATH_CALUDE_sector_area_l3625_362526

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 6) (h2 : central_angle = 2) :
  let radius := arc_length / central_angle
  (1/2) * radius^2 * central_angle = 9 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l3625_362526


namespace NUMINAMATH_CALUDE_pet_store_problem_l3625_362519

/-- The number of ways to choose pets for Emily, John, and Lucy -/
def pet_store_combinations (num_puppies num_kittens num_rabbits : ℕ) : ℕ :=
  num_puppies * num_kittens * num_rabbits * 6

/-- Theorem: Given 20 puppies, 10 kittens, and 12 rabbits, there are 14400 ways for
    Emily, John, and Lucy to buy pets, ensuring they all get different types of pets. -/
theorem pet_store_problem : pet_store_combinations 20 10 12 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_problem_l3625_362519


namespace NUMINAMATH_CALUDE_min_value_z3_l3625_362561

open Complex

theorem min_value_z3 (z₁ z₂ z₃ : ℂ) 
  (h_im : (z₁ / z₂).im ≠ 0 ∧ (z₁ / z₂).re = 0)
  (h_mag_z1 : abs z₁ = 1)
  (h_mag_z2 : abs z₂ = 1)
  (h_sum : abs (z₁ + z₂ + z₃) = 1) :
  abs z₃ ≥ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_z3_l3625_362561


namespace NUMINAMATH_CALUDE_square_eq_sixteen_l3625_362502

theorem square_eq_sixteen (x : ℝ) : (x - 3)^2 = 16 ↔ x = 7 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_sixteen_l3625_362502


namespace NUMINAMATH_CALUDE_prob_sum_less_than_15_l3625_362552

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ 3

/-- The number of outcomes where the sum is less than 15 -/
def favorableOutcomes : ℕ := totalOutcomes - 26

/-- The probability of rolling three fair six-sided dice and getting a sum less than 15 -/
theorem prob_sum_less_than_15 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 95 / 108 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_15_l3625_362552


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3625_362584

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (D : ℝ), D = Real.sqrt (12 / 17) ∧
  (∀ (x y : ℝ), x^2 + 2*y^2 + 3 ≥ D*(3*x + 4*y)) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 + 3 ≥ D'*(3*x + 4*y)) → D' ≤ D) :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3625_362584


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l3625_362532

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l3625_362532


namespace NUMINAMATH_CALUDE_retail_price_calculation_l3625_362501

/-- Proves that the retail price of a machine is $120 given the specified conditions -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  wholesale_price = 90 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ),
    retail_price = 120 ∧
    wholesale_price * (1 + profit_rate) = retail_price * (1 - discount_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l3625_362501


namespace NUMINAMATH_CALUDE_extreme_values_and_bounds_l3625_362558

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- State the theorem
theorem extreme_values_and_bounds (a b : ℝ) :
  (∃ (x : ℝ), x = 1 ∧ (∀ (h : ℝ), f a b x ≥ f a b h ∨ f a b x ≤ f a b h)) ∧
  (∃ (y : ℝ), y = -2/3 ∧ (∀ (h : ℝ), f a b y ≥ f a b h ∨ f a b y ≤ f a b h)) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x ≥ -5/2) ∧
  (∃ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x = 2) ∧
  (∃ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x = -5/2) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_bounds_l3625_362558


namespace NUMINAMATH_CALUDE_football_practice_missed_days_l3625_362514

/-- The number of days a football team missed practice due to rain -/
def days_missed (daily_practice_hours : ℕ) (total_practice_hours : ℕ) (days_in_week : ℕ) : ℕ :=
  days_in_week - (total_practice_hours / daily_practice_hours)

/-- Theorem: The football team missed 1 day of practice due to rain -/
theorem football_practice_missed_days :
  days_missed 6 36 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_football_practice_missed_days_l3625_362514


namespace NUMINAMATH_CALUDE_pattern_equality_l3625_362553

theorem pattern_equality (n : ℕ) (h : n > 1) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_pattern_equality_l3625_362553


namespace NUMINAMATH_CALUDE_a_positive_necessary_not_sufficient_l3625_362566

theorem a_positive_necessary_not_sufficient :
  (∀ a : ℝ, a^2 < a → a > 0) ∧
  (∃ a : ℝ, a > 0 ∧ a^2 ≥ a) :=
by sorry

end NUMINAMATH_CALUDE_a_positive_necessary_not_sufficient_l3625_362566


namespace NUMINAMATH_CALUDE_pauls_rate_l3625_362500

/-- The number of cars Paul and Jack can service in a day -/
def total_cars (paul_rate : ℝ) : ℝ := 8 * (paul_rate + 3)

/-- Theorem stating Paul's rate of changing oil in cars per hour -/
theorem pauls_rate : ∃ (paul_rate : ℝ), total_cars paul_rate = 40 ∧ paul_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_rate_l3625_362500


namespace NUMINAMATH_CALUDE_red_paint_percentage_l3625_362512

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : ℝ
  red : ℝ
  white : ℝ
  total : ℝ
  blue_percentage : ℝ
  red_percentage : ℝ
  white_percentage : ℝ

/-- Given a paint mixture with 70% blue paint, 140 ounces of blue paint, 
    and 20 ounces of white paint, prove that 20% of the mixture is red paint -/
theorem red_paint_percentage 
  (mixture : PaintMixture) 
  (h1 : mixture.blue_percentage = 0.7) 
  (h2 : mixture.blue = 140) 
  (h3 : mixture.white = 20) 
  (h4 : mixture.total = mixture.blue + mixture.red + mixture.white) 
  (h5 : mixture.blue_percentage + mixture.red_percentage + mixture.white_percentage = 1) :
  mixture.red_percentage = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_red_paint_percentage_l3625_362512


namespace NUMINAMATH_CALUDE_wendys_pastries_l3625_362583

/-- Wendy's pastry problem -/
theorem wendys_pastries (cupcakes cookies sold : ℕ) 
  (h1 : cupcakes = 4)
  (h2 : cookies = 29)
  (h3 : sold = 9) :
  cupcakes + cookies - sold = 24 := by
  sorry

end NUMINAMATH_CALUDE_wendys_pastries_l3625_362583


namespace NUMINAMATH_CALUDE_venus_meal_cost_calculation_l3625_362544

/-- The cost per meal at Venus Hall -/
def venus_meal_cost : ℝ := 35

/-- The room rental cost for Caesar's -/
def caesars_room_cost : ℝ := 800

/-- The meal cost for Caesar's -/
def caesars_meal_cost : ℝ := 30

/-- The room rental cost for Venus Hall -/
def venus_room_cost : ℝ := 500

/-- The number of guests at which the costs are equal -/
def num_guests : ℝ := 60

theorem venus_meal_cost_calculation :
  caesars_room_cost + caesars_meal_cost * num_guests =
  venus_room_cost + venus_meal_cost * num_guests :=
by sorry

end NUMINAMATH_CALUDE_venus_meal_cost_calculation_l3625_362544


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l3625_362550

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l3625_362550


namespace NUMINAMATH_CALUDE_probability_continuous_stripe_l3625_362589

/-- Represents a single cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Bool

/-- Represents a tower of three cubes -/
structure CubeTower where
  top : StripedCube
  middle : StripedCube
  bottom : StripedCube

/-- Checks if there's a continuous stripe through a vertical face pair -/
def has_continuous_stripe (face_pair : Fin 4) (tower : CubeTower) : Bool :=
  sorry

/-- Counts the number of cube towers with a continuous vertical stripe -/
def count_continuous_stripes (towers : List CubeTower) : Nat :=
  sorry

/-- The total number of possible cube tower configurations -/
def total_configurations : Nat := 2^18

/-- The number of cube tower configurations with a continuous vertical stripe -/
def favorable_configurations : Nat := 64

theorem probability_continuous_stripe :
  (favorable_configurations : ℚ) / total_configurations = 1 / 4096 :=
sorry

end NUMINAMATH_CALUDE_probability_continuous_stripe_l3625_362589


namespace NUMINAMATH_CALUDE_second_division_percentage_l3625_362507

theorem second_division_percentage 
  (total_students : ℕ) 
  (first_division_percentage : ℚ) 
  (just_passed : ℕ) 
  (h1 : total_students = 300)
  (h2 : first_division_percentage = 28/100)
  (h3 : just_passed = 54)
  : (↑(total_students - (total_students * first_division_percentage).floor - just_passed) / total_students : ℚ) = 54/100 := by
  sorry

end NUMINAMATH_CALUDE_second_division_percentage_l3625_362507


namespace NUMINAMATH_CALUDE_calculation_proof_l3625_362595

theorem calculation_proof : 
  (168 / 100 * ((1265^2) / 21)) / (6 - (3^2)) = -42646.26666666667 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3625_362595


namespace NUMINAMATH_CALUDE_max_tennis_court_area_l3625_362599

/-- Represents the dimensions of a rectangular tennis court --/
structure CourtDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular tennis court --/
def area (d : CourtDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular tennis court --/
def perimeter (d : CourtDimensions) : ℝ := 2 * (d.length + d.width)

/-- Checks if the court dimensions meet the minimum requirements --/
def meetsMinimumRequirements (d : CourtDimensions) : Prop :=
  d.length ≥ 85 ∧ d.width ≥ 45

/-- Theorem stating the maximum area of the tennis court --/
theorem max_tennis_court_area :
  ∃ (d : CourtDimensions),
    perimeter d = 320 ∧
    meetsMinimumRequirements d ∧
    area d = 6375 ∧
    ∀ (d' : CourtDimensions),
      perimeter d' = 320 ∧ meetsMinimumRequirements d' → area d' ≤ area d :=
by sorry

end NUMINAMATH_CALUDE_max_tennis_court_area_l3625_362599


namespace NUMINAMATH_CALUDE_judy_hits_percentage_l3625_362521

theorem judy_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 8)
  (h5 : total_hits ≥ home_runs + triples + doubles) :
  (((total_hits - (home_runs + triples + doubles)) : ℚ) / total_hits) * 100 = 74 := by
sorry

end NUMINAMATH_CALUDE_judy_hits_percentage_l3625_362521


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3625_362505

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - x^3)^3 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3625_362505


namespace NUMINAMATH_CALUDE_infinitely_many_rational_solutions_l3625_362511

theorem infinitely_many_rational_solutions :
  ∃ f : ℕ → ℚ × ℚ,
    (∀ n : ℕ, (f n).1^3 + (f n).2^3 = 9) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_rational_solutions_l3625_362511


namespace NUMINAMATH_CALUDE_sum_of_squares_l3625_362513

theorem sum_of_squares (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 10)
  (h2 : (a * b * c)^(1/3 : ℝ) = 6)
  (h3 : 3 / (1/a + 1/b + 1/c) = 4) :
  a^2 + b^2 + c^2 = 576 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3625_362513


namespace NUMINAMATH_CALUDE_shaded_area_in_circumscribed_square_l3625_362579

/-- Given a square with side length 20 cm circumscribed around a circle,
    where two of its diagonals form an isosceles triangle with the circle's center,
    the sum of the areas of the two small shaded regions is 100π - 200 square centimeters. -/
theorem shaded_area_in_circumscribed_square (π : ℝ) :
  let square_side : ℝ := 20
  let circle_radius : ℝ := square_side * Real.sqrt 2 / 2
  let sector_area : ℝ := π * circle_radius^2 / 4
  let triangle_area : ℝ := circle_radius^2 / 2
  let shaded_area : ℝ := 2 * (sector_area - triangle_area)
  shaded_area = 100 * π - 200 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_circumscribed_square_l3625_362579


namespace NUMINAMATH_CALUDE_john_finishes_at_305_l3625_362535

/-- Represents time in minutes since midnight -/
def Time := ℕ

/-- Converts hours and minutes to Time -/
def toTime (hours minutes : ℕ) : Time :=
  hours * 60 + minutes

/-- The time John starts working -/
def startTime : Time := toTime 9 0

/-- The time John finishes the fourth task -/
def fourthTaskEndTime : Time := toTime 13 0

/-- The number of tasks John completes -/
def totalTasks : ℕ := 6

/-- The number of tasks completed before the first break -/
def tasksBeforeBreak : ℕ := 1

/-- The duration of each break in minutes -/
def breakDuration : ℕ := 10

/-- Calculates the time John finishes all tasks -/
noncomputable def calculateEndTime : Time := sorry

theorem john_finishes_at_305 :
  calculateEndTime = toTime 15 5 := by sorry

end NUMINAMATH_CALUDE_john_finishes_at_305_l3625_362535


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_ratio_l3625_362522

/-- A geometric sequence with first term less than zero and increasing terms has a common ratio between 0 and 1. -/
theorem geometric_sequence_increasing_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (h1 : a 1 < 0)  -- First term is negative
  (h2 : ∀ n : ℕ, a n < a (n + 1))  -- Sequence is strictly increasing
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q)  -- Definition of geometric sequence
  : 0 < q ∧ q < 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_ratio_l3625_362522


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3625_362529

theorem coin_flip_probability (n : ℕ) : n = 7 → (n.choose 2 : ℚ) / 2^n = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3625_362529


namespace NUMINAMATH_CALUDE_sin_squared_alpha_minus_pi_fourth_l3625_362506

theorem sin_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2/3) :
  Real.sin (α - Real.pi/4)^2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_alpha_minus_pi_fourth_l3625_362506


namespace NUMINAMATH_CALUDE_sector_min_perimeter_l3625_362503

theorem sector_min_perimeter (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (1/2 * l * r = 4) → (l + 2*r ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_sector_min_perimeter_l3625_362503


namespace NUMINAMATH_CALUDE_triangle_side_length_l3625_362572

/-- Given a triangle ABC with the condition that cos(∠A - ∠B) + sin(∠A + ∠B) = 2 and AB = 4,
    prove that BC = 2√2 -/
theorem triangle_side_length (A B C : ℝ) (h1 : 0 < A ∧ A < π)
    (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
    (h5 : Real.cos (A - B) + Real.sin (A + B) = 2) (h6 : ∃ AB : ℝ, AB = 4) :
    ∃ BC : ℝ, BC = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3625_362572
