import Mathlib

namespace NUMINAMATH_CALUDE_black_tshirt_cost_black_tshirt_cost_is_30_l1214_121438

/-- The cost of black t-shirts given the sale conditions -/
theorem black_tshirt_cost (total_tshirts : ℕ) (sale_duration : ℕ) 
  (white_tshirt_cost : ℕ) (revenue_per_minute : ℕ) : ℕ :=
  let total_revenue := sale_duration * revenue_per_minute
  let num_black_tshirts := total_tshirts / 2
  let num_white_tshirts := total_tshirts / 2
  let white_tshirt_revenue := num_white_tshirts * white_tshirt_cost
  let black_tshirt_revenue := total_revenue - white_tshirt_revenue
  black_tshirt_revenue / num_black_tshirts

/-- The cost of black t-shirts is $30 given the specific sale conditions -/
theorem black_tshirt_cost_is_30 : 
  black_tshirt_cost 200 25 25 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_black_tshirt_cost_black_tshirt_cost_is_30_l1214_121438


namespace NUMINAMATH_CALUDE_shaded_area_is_48_l1214_121483

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  area : ℝ
  small_triangle_count : ℕ
  small_triangle_area : ℝ

/-- The shaded area in an isosceles right triangle -/
def shaded_area (t : IsoscelesRightTriangle) (shaded_count : ℕ) : ℝ :=
  shaded_count * t.small_triangle_area

/-- Theorem: The shaded area of 12 small triangles in the given isosceles right triangle is 48 -/
theorem shaded_area_is_48 (t : IsoscelesRightTriangle) 
    (h1 : t.leg_length = 12)
    (h2 : t.area = 1/2 * t.leg_length * t.leg_length)
    (h3 : t.small_triangle_count = 18)
    (h4 : t.small_triangle_area = t.area / t.small_triangle_count) :
  shaded_area t 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_48_l1214_121483


namespace NUMINAMATH_CALUDE_page_number_added_twice_l1214_121430

theorem page_number_added_twice (m : ℕ) (p : ℕ) : 
  m = 71 → 
  1 ≤ p → 
  p ≤ m → 
  (m * (m + 1)) / 2 + p = 2550 → 
  p = 6 := by
sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l1214_121430


namespace NUMINAMATH_CALUDE_double_square_root_simplification_l1214_121431

theorem double_square_root_simplification (a b m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 2 * Real.sqrt b > 0) 
  (hm : m > 0) (hn : n > 0)
  (h1 : Real.sqrt m ^ 2 + Real.sqrt n ^ 2 = a)
  (h2 : Real.sqrt m * Real.sqrt n = Real.sqrt b) :
  Real.sqrt (a + 2 * Real.sqrt b) = |Real.sqrt m + Real.sqrt n| ∧
  Real.sqrt (a - 2 * Real.sqrt b) = |Real.sqrt m - Real.sqrt n| :=
by sorry

end NUMINAMATH_CALUDE_double_square_root_simplification_l1214_121431


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1214_121462

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1214_121462


namespace NUMINAMATH_CALUDE_salary_average_increase_l1214_121406

theorem salary_average_increase 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 24 → 
  avg_salary = 2400 → 
  manager_salary = 4900 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_average_increase_l1214_121406


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1214_121485

theorem quadratic_equation_roots (k : ℝ) (h : k ≠ 0) :
  (∃ x₁ x₂ : ℝ, k * x₁^2 + (k + 3) * x₁ + 3 = 0 ∧ k * x₂^2 + (k + 3) * x₂ + 3 = 0) ∧
  (∀ x : ℤ, k * x^2 + (k + 3) * x + 3 = 0 → k = 1 ∨ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1214_121485


namespace NUMINAMATH_CALUDE_reflect_M_x_axis_l1214_121495

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point M -/
def M : ℝ × ℝ := (3, -4)

/-- Theorem stating that reflecting M across the x-axis results in (3, 4) -/
theorem reflect_M_x_axis : reflect_x M = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_reflect_M_x_axis_l1214_121495


namespace NUMINAMATH_CALUDE_shorter_stick_length_l1214_121460

theorem shorter_stick_length (longer shorter : ℝ) 
  (h1 : longer - shorter = 12)
  (h2 : (2/3) * longer = shorter) : 
  shorter = 24 := by
  sorry

end NUMINAMATH_CALUDE_shorter_stick_length_l1214_121460


namespace NUMINAMATH_CALUDE_special_vector_exists_l1214_121477

/-- Define a new operation * for 2D vectors -/
def vec_mult (m n : Fin 2 → ℝ) : Fin 2 → ℝ := 
  λ i => if i = 0 then m 0 * n 0 + m 1 * n 1 else m 0 * n 1 + m 1 * n 0

/-- Theorem: If m * p = m for all m, then p = (1, 0) -/
theorem special_vector_exists :
  ∃ p : Fin 2 → ℝ, (∀ m : Fin 2 → ℝ, vec_mult m p = m) → p 0 = 1 ∧ p 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_special_vector_exists_l1214_121477


namespace NUMINAMATH_CALUDE_price_reduction_proof_l1214_121476

/-- The original selling price in yuan -/
def original_price : ℝ := 40

/-- The cost price in yuan -/
def cost_price : ℝ := 30

/-- The initial daily sales volume -/
def initial_sales : ℕ := 48

/-- The price after two consecutive reductions in yuan -/
def reduced_price : ℝ := 32.4

/-- The increase in daily sales for every 0.5 yuan reduction in price -/
def sales_increase_rate : ℝ := 8

/-- The desired daily profit in yuan -/
def desired_profit : ℝ := 504

/-- The percentage reduction that results in the reduced price after two consecutive reductions -/
def percentage_reduction : ℝ := 0.1

/-- The price reduction that achieves the desired daily profit -/
def price_reduction : ℝ := 3

theorem price_reduction_proof :
  (∃ x : ℝ, original_price * (1 - x)^2 = reduced_price ∧ 0 < x ∧ x < 1 ∧ x = percentage_reduction) ∧
  (∃ y : ℝ, (original_price - cost_price - y) * (initial_sales + sales_increase_rate * y) = desired_profit ∧ y = price_reduction) :=
sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l1214_121476


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1214_121433

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a^2 - 4 + (a^2 - 3*a + 2)*I
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1214_121433


namespace NUMINAMATH_CALUDE_max_value_of_f_sum_of_powers_gt_one_l1214_121408

-- Part 1
theorem max_value_of_f (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ M : ℝ, M = 1 ∧ ∀ x > -1, (1 + x)^a - a * x ≤ M := by sorry

-- Part 2
theorem sum_of_powers_gt_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^b + b^a > 1 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_sum_of_powers_gt_one_l1214_121408


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l1214_121429

/-- Given a square carpet with the following properties:
  * Side length of the carpet is 12 feet
  * Contains one large shaded square and twelve smaller congruent shaded squares
  * S is the side length of the large shaded square
  * T is the side length of each smaller shaded square
  * The ratio 12:S is 4
  * The ratio S:T is 4
  Prove that the total shaded area is 15.75 square feet -/
theorem shaded_area_of_carpet (S T : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 4)
  : S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l1214_121429


namespace NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l1214_121487

theorem max_y_value (x y : ℤ) (h : x * y + 5 * x + 4 * y = -5) : y ≤ 10 := by
  sorry

theorem max_y_value_achievable : ∃ x y : ℤ, x * y + 5 * x + 4 * y = -5 ∧ y = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l1214_121487


namespace NUMINAMATH_CALUDE_xyz_product_l1214_121491

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l1214_121491


namespace NUMINAMATH_CALUDE_decreasing_iff_a_in_range_l1214_121496

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a / x else (2 - 3 * a) * x + 1

/-- The property that f is a decreasing function on ℝ -/
def is_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → f a x > f a y

/-- The main theorem stating the equivalence between f being decreasing and a being in (2/3, 3/4] -/
theorem decreasing_iff_a_in_range (a : ℝ) :
  is_decreasing a ↔ 2/3 < a ∧ a ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_decreasing_iff_a_in_range_l1214_121496


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l1214_121479

theorem min_value_of_sum_of_fractions (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 ∧ 
  ((a / b + b / c + c / d + d / a) = 4 ↔ a = b ∧ b = c ∧ c = d) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l1214_121479


namespace NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l1214_121461

theorem cubic_and_quadratic_equations :
  (∃ x : ℝ, x^3 + 64 = 0 ↔ x = -4) ∧
  (∃ x : ℝ, (x - 2)^2 = 81 ↔ x = 11 ∨ x = -7) := by
  sorry

end NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l1214_121461


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l1214_121450

theorem logarithm_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) 
  (h_eq : (Real.log x) / (3 * Real.log b) + (Real.log b) / (3 * Real.log x) = 1) :
  x = b ^ ((3 + Real.sqrt 5) / 2) ∨ x = b ^ ((3 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l1214_121450


namespace NUMINAMATH_CALUDE_first_term_of_ap_l1214_121428

/-- 
Given an arithmetic progression where:
- The 10th term is 26
- The common difference is 2

Prove that the first term is 8
-/
theorem first_term_of_ap (a : ℝ) : 
  (∃ (d : ℝ), d = 2 ∧ a + 9 * d = 26) → a = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_ap_l1214_121428


namespace NUMINAMATH_CALUDE_team_ages_mode_l1214_121439

def team_ages : List Nat := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem team_ages_mode :
  mode team_ages = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_ages_mode_l1214_121439


namespace NUMINAMATH_CALUDE_power_of_128_equals_32_l1214_121457

theorem power_of_128_equals_32 : (128 : ℝ) ^ (5/7 : ℝ) = 32 := by
  have h : 128 = 2^7 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_128_equals_32_l1214_121457


namespace NUMINAMATH_CALUDE_percent_of_a_l1214_121423

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10/3) * a := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_l1214_121423


namespace NUMINAMATH_CALUDE_largest_B_term_l1214_121494

def B (k : ℕ) : ℝ := (Nat.choose 2000 k) * (0.1 ^ k)

theorem largest_B_term : 
  ∀ j ∈ Finset.range 2001, B 181 ≥ B j :=
sorry

end NUMINAMATH_CALUDE_largest_B_term_l1214_121494


namespace NUMINAMATH_CALUDE_fraction_exceeding_by_20_l1214_121463

theorem fraction_exceeding_by_20 (N : ℚ) (F : ℚ) : 
  N = 32 → N = F * N + 20 → F = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exceeding_by_20_l1214_121463


namespace NUMINAMATH_CALUDE_no_perfect_square_E_l1214_121475

-- Define E(x) as the integer closest to x on the number line
noncomputable def E (x : ℝ) : ℤ :=
  round x

-- Theorem statement
theorem no_perfect_square_E (n : ℕ+) : ¬∃ (k : ℕ), E (n + Real.sqrt n) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_E_l1214_121475


namespace NUMINAMATH_CALUDE_total_fish_l1214_121486

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 8) : 
  lilly_fish + rosy_fish = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l1214_121486


namespace NUMINAMATH_CALUDE_johns_tour_program_l1214_121409

theorem johns_tour_program (total_budget : ℕ) (budget_reduction : ℕ) (extra_days : ℕ) :
  total_budget = 360 ∧ budget_reduction = 3 ∧ extra_days = 4 →
  ∃ (days : ℕ) (daily_expense : ℕ),
    total_budget = days * daily_expense ∧
    total_budget = (days + extra_days) * (daily_expense - budget_reduction) ∧
    days = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_tour_program_l1214_121409


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l1214_121474

theorem square_root_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l1214_121474


namespace NUMINAMATH_CALUDE_xyz_and_fourth_power_sum_l1214_121470

theorem xyz_and_fourth_power_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 1)
  (sum_sq_eq : x^2 + y^2 + z^2 = 2)
  (sum_cube_eq : x^3 + y^3 + z^3 = 3) :
  x * y * z = 1/6 ∧ x^4 + y^4 + z^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_and_fourth_power_sum_l1214_121470


namespace NUMINAMATH_CALUDE_triangle_construction_l1214_121426

/-- Given a point A, a plane S, and distances ρ, ρₐ, and b-c,
    we can construct a triangle ABC with specific properties. -/
theorem triangle_construction (A : ℝ × ℝ) (S : ℝ × ℝ) (ρ ρₐ : ℝ) (b_minus_c : ℝ) 
  (s a b c : ℝ) :
  -- Side a lies in plane S (represented by the condition that a is real)
  -- One vertex is A (implicit in the construction)
  -- ρ is the inradius
  -- ρₐ is the exradius opposite to side a
  (s = (a + b + c) / 2) →  -- Definition of semiperimeter
  (ρ > 0) →  -- Inradius is positive
  (ρₐ > 0) →  -- Exradius is positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- Triangle sides are positive
  (b - c = b_minus_c) →  -- Given difference of sides
  -- Then the following relationships hold:
  ((s - b) * (s - c) = ρ * ρₐ) ∧
  ((s - c) - (s - b) = b - c) ∧
  (Real.sqrt ((s - b) * (s - c)) = Real.sqrt (ρ * ρₐ)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_l1214_121426


namespace NUMINAMATH_CALUDE_least_pennies_count_l1214_121411

theorem least_pennies_count (a : ℕ) : 
  (a > 0) → 
  (a % 7 = 1) → 
  (a % 3 = 0) → 
  (∀ b : ℕ, b > 0 → b % 7 = 1 → b % 3 = 0 → a ≤ b) → 
  a = 15 := by
sorry

end NUMINAMATH_CALUDE_least_pennies_count_l1214_121411


namespace NUMINAMATH_CALUDE_units_digit_factorial_product_squared_l1214_121472

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The main theorem -/
theorem units_digit_factorial_product_squared :
  unitsDigit ((factorial 1 * factorial 2 * factorial 3 * factorial 4) ^ 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_product_squared_l1214_121472


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1214_121455

theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) :
  r = 3.5 →
  A = 56 →
  A = r * (p / 2) →
  p = 32 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1214_121455


namespace NUMINAMATH_CALUDE_parabola_directrix_l1214_121442

/-- The directrix of a parabola with equation y = -1/8 * x^2 is y = 2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/8 * x^2) → (∃ (k : ℝ), k = 2 ∧ k = y + 1/(4 * (1/8))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1214_121442


namespace NUMINAMATH_CALUDE_luisas_books_l1214_121425

theorem luisas_books (maddie_books amy_books : ℕ) (h1 : maddie_books = 15) (h2 : amy_books = 6)
  (h3 : ∃ luisa_books : ℕ, amy_books + luisa_books = maddie_books + 9) :
  ∃ luisa_books : ℕ, luisa_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_luisas_books_l1214_121425


namespace NUMINAMATH_CALUDE_eq1_roots_eq2_roots_l1214_121416

-- Define the quadratic equations
def eq1 (x : ℝ) : Prop := x^2 + 10*x + 16 = 0
def eq2 (x : ℝ) : Prop := x*(x+4) = 8*x + 12

-- Theorem for the first equation
theorem eq1_roots : 
  (∃ x : ℝ, eq1 x) ↔ (eq1 (-2) ∧ eq1 (-8)) :=
sorry

-- Theorem for the second equation
theorem eq2_roots : 
  (∃ x : ℝ, eq2 x) ↔ (eq2 (-2) ∧ eq2 6) :=
sorry

end NUMINAMATH_CALUDE_eq1_roots_eq2_roots_l1214_121416


namespace NUMINAMATH_CALUDE_pollys_age_equals_sum_of_children_ages_l1214_121467

/-- Represents Polly's age when it equals the sum of her three children's ages -/
def pollys_age : ℕ := 33

/-- Represents the age of Polly's first child -/
def first_child_age (x : ℕ) : ℕ := x - 20

/-- Represents the age of Polly's second child -/
def second_child_age (x : ℕ) : ℕ := x - 22

/-- Represents the age of Polly's third child -/
def third_child_age (x : ℕ) : ℕ := x - 24

/-- Theorem stating that Polly's age equals the sum of her three children's ages -/
theorem pollys_age_equals_sum_of_children_ages :
  pollys_age = first_child_age pollys_age + second_child_age pollys_age + third_child_age pollys_age :=
by sorry

end NUMINAMATH_CALUDE_pollys_age_equals_sum_of_children_ages_l1214_121467


namespace NUMINAMATH_CALUDE_max_marble_diff_is_six_l1214_121418

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : Nat
  color2 : String
  count2 : Nat

/-- Calculates the absolute difference between two numbers -/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Theorem: The maximum difference between marble counts in any basket is 6 -/
theorem max_marble_diff_is_six (basketA basketB basketC : Basket)
  (hA : basketA = { color1 := "red", count1 := 4, color2 := "yellow", count2 := 2 })
  (hB : basketB = { color1 := "green", count1 := 6, color2 := "yellow", count2 := 1 })
  (hC : basketC = { color1 := "white", count1 := 3, color2 := "yellow", count2 := 9 }) :
  max (absDiff basketA.count1 basketA.count2)
      (max (absDiff basketB.count1 basketB.count2)
           (absDiff basketC.count1 basketC.count2)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_max_marble_diff_is_six_l1214_121418


namespace NUMINAMATH_CALUDE_total_songs_bought_l1214_121404

theorem total_songs_bought (country_albums pop_albums rock_albums : ℕ)
  (country_songs_per_album pop_songs_per_album rock_songs_per_album : ℕ) :
  country_albums = 2 ∧
  pop_albums = 8 ∧
  rock_albums = 5 ∧
  country_songs_per_album = 7 ∧
  pop_songs_per_album = 10 ∧
  rock_songs_per_album = 12 →
  country_albums * country_songs_per_album +
  pop_albums * pop_songs_per_album +
  rock_albums * rock_songs_per_album = 154 :=
by sorry

end NUMINAMATH_CALUDE_total_songs_bought_l1214_121404


namespace NUMINAMATH_CALUDE_area_between_curves_l1214_121413

-- Define the functions
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := 2 - x^2

-- Define the intersection points
def x₁ : ℝ := -2
def x₂ : ℝ := 1

-- State the theorem
theorem area_between_curves : 
  (∫ (x : ℝ) in x₁..x₂, g x - f x) = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l1214_121413


namespace NUMINAMATH_CALUDE_fifth_decimal_place_of_1_0025_pow_10_l1214_121493

theorem fifth_decimal_place_of_1_0025_pow_10 :
  ∃ (n : ℕ) (r : ℚ), 
    (1 + 1/400)^10 = n + r ∧ 
    n < (1 + 1/400)^10 ∧
    (1 + 1/400)^10 < n + 1 ∧
    (r * 100000).floor = 8 :=
by sorry

end NUMINAMATH_CALUDE_fifth_decimal_place_of_1_0025_pow_10_l1214_121493


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l1214_121444

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l1214_121444


namespace NUMINAMATH_CALUDE_fraction_sum_cube_l1214_121481

theorem fraction_sum_cube (a b : ℝ) (h : (a + b) / (a - b) + (a - b) / (a + b) = 4) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_cube_l1214_121481


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_one_l1214_121465

theorem sum_of_roots_equals_negative_one :
  ∀ x y : ℝ, (x - 4) * (x + 5) = 33 ∧ (y - 4) * (y + 5) = 33 → x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_one_l1214_121465


namespace NUMINAMATH_CALUDE_quadratic_sequence_existence_l1214_121435

theorem quadratic_sequence_existence (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧
  ∀ i : ℕ, i ≤ n → i ≠ 0 → |a i - a (i - 1)| = i^2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_existence_l1214_121435


namespace NUMINAMATH_CALUDE_complex_square_l1214_121489

theorem complex_square (z : ℂ) : z = 2 + 3*I → z^2 = -5 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1214_121489


namespace NUMINAMATH_CALUDE_maria_earnings_l1214_121400

def brush_cost_1 : ℕ := 20
def brush_cost_2 : ℕ := 25
def brush_cost_3 : ℕ := 30
def acrylic_paint_cost : ℕ := 8
def oil_paint_cost : ℕ := 12
def acrylic_paint_amount : ℕ := 5
def oil_paint_amount : ℕ := 3
def selling_price : ℕ := 200

def total_brush_cost : ℕ := brush_cost_1 + brush_cost_2 + brush_cost_3

def canvas_cost_1 : ℕ := 3 * total_brush_cost
def canvas_cost_2 : ℕ := 2 * total_brush_cost

def total_paint_cost : ℕ := acrylic_paint_cost * acrylic_paint_amount + oil_paint_cost * oil_paint_amount

def total_cost : ℕ := total_brush_cost + canvas_cost_1 + canvas_cost_2 + total_paint_cost

theorem maria_earnings : (selling_price : ℤ) - total_cost = -326 := by sorry

end NUMINAMATH_CALUDE_maria_earnings_l1214_121400


namespace NUMINAMATH_CALUDE_min_value_theorem_l1214_121454

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (2 / x) + (3 / y) ≥ 1 ∧ ((2 / x) + (3 / y) = 1 ↔ x = 12 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1214_121454


namespace NUMINAMATH_CALUDE_sum_of_even_integers_2_to_2022_l1214_121458

theorem sum_of_even_integers_2_to_2022 : 
  (Finset.range 1011).sum (fun i => 2 * (i + 1)) = 1023112 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_2_to_2022_l1214_121458


namespace NUMINAMATH_CALUDE_total_triangles_in_4_layer_grid_l1214_121401

/-- Represents a triangular grid with a given number of layers -/
def TriangularGrid (layers : ℕ) : Type := Unit

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid 4) : ℕ := 10

/-- Counts the number of medium triangles (made of 4 small triangles) in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid 4) : ℕ := 6

/-- Counts the number of large triangles (made of 9 small triangles) in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid 4) : ℕ := 1

/-- The total number of triangles in a 4-layer triangular grid is 17 -/
theorem total_triangles_in_4_layer_grid (grid : TriangularGrid 4) :
  countSmallTriangles grid + countMediumTriangles grid + countLargeTriangles grid = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_in_4_layer_grid_l1214_121401


namespace NUMINAMATH_CALUDE_water_layer_thickness_l1214_121447

/-- Thickness of water layer after removing a sphere from a cylindrical vessel -/
theorem water_layer_thickness (R r : ℝ) (h_R : R = 4) (h_r : r = 3) :
  let V := π * R^2 * (2 * r)
  let V_sphere := (4/3) * π * r^3
  let V_water := V - V_sphere
  V_water / (π * R^2) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_water_layer_thickness_l1214_121447


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1214_121452

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 5 →
  downstream_distance = 34.47 →
  downstream_time = 44 / 60 →
  ∃ (boat_speed : ℝ), abs (boat_speed - 42.01) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1214_121452


namespace NUMINAMATH_CALUDE_max_daily_sales_amount_l1214_121459

def f (t : ℕ) : ℝ := -t + 30

def g (t : ℕ) : ℝ :=
  if t ≤ 10 then 2 * t + 40 else 15

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales_amount (t : ℕ) (h1 : 1 ≤ t) (h2 : t ≤ 20) :
  ∃ (max_t : ℕ) (max_value : ℝ), 
    (∀ t', 1 ≤ t' → t' ≤ 20 → S t' ≤ S max_t) ∧ 
    S max_t = max_value ∧ 
    max_t = 5 ∧ 
    max_value = 1250 :=
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_amount_l1214_121459


namespace NUMINAMATH_CALUDE_solutions_of_quartic_equation_l1214_121497

theorem solutions_of_quartic_equation :
  ∀ x : ℂ, x^4 - 16 = 0 ↔ x ∈ ({2, -2, 2*I, -2*I} : Set ℂ) :=
by sorry

end NUMINAMATH_CALUDE_solutions_of_quartic_equation_l1214_121497


namespace NUMINAMATH_CALUDE_balloon_arrangements_eq_36_l1214_121471

/-- The number of distinguishable arrangements of letters in "BALLOON" with vowels first -/
def balloon_arrangements : ℕ :=
  let vowels := ['A', 'O', 'O']
  let consonants := ['B', 'L', 'L', 'N']
  let vowel_arrangements := Nat.factorial 3 / Nat.factorial 2
  let consonant_arrangements := Nat.factorial 4 / Nat.factorial 2
  vowel_arrangements * consonant_arrangements

/-- Theorem stating that the number of distinguishable arrangements of "BALLOON" with vowels first is 36 -/
theorem balloon_arrangements_eq_36 : balloon_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_eq_36_l1214_121471


namespace NUMINAMATH_CALUDE_vector_problem_l1214_121402

/-- Given vectors and triangle properties, prove vector n coordinates and magnitude range of n + p -/
theorem vector_problem (m n p q : ℝ × ℝ) (A B C : ℝ) : 
  m = (1, 1) →
  q = (1, 0) →
  (m.1 * n.1 + m.2 * n.2) = -1 →
  ∃ (k : ℝ), n = k • q →
  p = (2 * Real.cos (C / 2) ^ 2, Real.cos A) →
  B = π / 3 →
  A + B + C = π →
  (n = (-1, 0) ∧ Real.sqrt 2 / 2 ≤ Real.sqrt ((n.1 + p.1)^2 + (n.2 + p.2)^2) ∧ 
   Real.sqrt ((n.1 + p.1)^2 + (n.2 + p.2)^2) < Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l1214_121402


namespace NUMINAMATH_CALUDE_periodic_function_value_l1214_121432

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2015 = 5 → f 2016 = 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1214_121432


namespace NUMINAMATH_CALUDE_candy_problem_l1214_121448

theorem candy_problem (initial_candy : ℕ) (num_bowls : ℕ) (removed_per_bowl : ℕ) (remaining_in_bowl : ℕ)
  (h1 : initial_candy = 100)
  (h2 : num_bowls = 4)
  (h3 : removed_per_bowl = 3)
  (h4 : remaining_in_bowl = 20) :
  initial_candy - (num_bowls * (remaining_in_bowl + removed_per_bowl)) = 8 :=
sorry

end NUMINAMATH_CALUDE_candy_problem_l1214_121448


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1214_121424

theorem binomial_coefficient_equation_solution (x : ℕ) : 
  Nat.choose 11 x = Nat.choose 11 (2*x - 4) ↔ x = 4 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1214_121424


namespace NUMINAMATH_CALUDE_louis_age_l1214_121480

/-- Given that Carla will be 30 years old in 6 years and the sum of Carla and Louis's current ages is 55, prove that Louis is currently 31 years old. -/
theorem louis_age (carla_future_age : ℕ) (years_until_future : ℕ) (sum_of_ages : ℕ) :
  carla_future_age = 30 →
  years_until_future = 6 →
  sum_of_ages = 55 →
  sum_of_ages - (carla_future_age - years_until_future) = 31 := by
  sorry

end NUMINAMATH_CALUDE_louis_age_l1214_121480


namespace NUMINAMATH_CALUDE_fraction_simplification_l1214_121473

theorem fraction_simplification (a b c d : ℝ) 
  (ha : a = Real.sqrt 125)
  (hb : b = 3 * Real.sqrt 45)
  (hc : c = 4 * Real.sqrt 20)
  (hd : d = Real.sqrt 75) :
  5 / (a + b + c + d) = Real.sqrt 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1214_121473


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1214_121410

theorem complex_modulus_equality (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1214_121410


namespace NUMINAMATH_CALUDE_marble_difference_l1214_121498

/-- The number of bags Mara has -/
def mara_bags : ℕ := 12

/-- The number of marbles in each of Mara's bags -/
def mara_marbles_per_bag : ℕ := 2

/-- The number of bags Markus has -/
def markus_bags : ℕ := 2

/-- The number of marbles in each of Markus's bags -/
def markus_marbles_per_bag : ℕ := 13

/-- The difference in the total number of marbles between Markus and Mara -/
theorem marble_difference : 
  markus_bags * markus_marbles_per_bag - mara_bags * mara_marbles_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l1214_121498


namespace NUMINAMATH_CALUDE_max_sequence_sum_l1214_121420

def arithmetic_sequence (n : ℕ) : ℚ := 5 - (5/7) * (n - 1)

def sequence_sum (n : ℕ) : ℚ := n * (2 * 5 + (n - 1) * (-5/7)) / 2

theorem max_sequence_sum :
  (∃ n : ℕ, sequence_sum n = 20) ∧
  (∀ m : ℕ, sequence_sum m ≤ 20) ∧
  (∀ n : ℕ, sequence_sum n = 20 → (n = 7 ∨ n = 8)) :=
sorry

end NUMINAMATH_CALUDE_max_sequence_sum_l1214_121420


namespace NUMINAMATH_CALUDE_juggler_count_l1214_121443

theorem juggler_count (balls_per_juggler : ℕ) (total_balls : ℕ) (h1 : balls_per_juggler = 6) (h2 : total_balls = 2268) :
  total_balls / balls_per_juggler = 378 := by
  sorry

end NUMINAMATH_CALUDE_juggler_count_l1214_121443


namespace NUMINAMATH_CALUDE_sandys_puppies_l1214_121449

/-- Given that Sandy initially had 8 puppies and gave away 4 puppies,
    prove that she now has 4 puppies remaining. -/
theorem sandys_puppies (initial_puppies : ℕ) (puppies_given_away : ℕ)
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = 4) :
  initial_puppies - puppies_given_away = 4 := by
sorry

end NUMINAMATH_CALUDE_sandys_puppies_l1214_121449


namespace NUMINAMATH_CALUDE_trig_expression_equality_l1214_121499

theorem trig_expression_equality : 
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  2 * (Real.sin (50 * π / 180) - 1) / Real.sin (40 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l1214_121499


namespace NUMINAMATH_CALUDE_circle_sum_l1214_121446

theorem circle_sum (square circle : ℚ) 
  (eq1 : 2 * square + 3 * circle = 26)
  (eq2 : 3 * square + 2 * circle = 23) :
  4 * circle = 128 / 5 := by
sorry

end NUMINAMATH_CALUDE_circle_sum_l1214_121446


namespace NUMINAMATH_CALUDE_vector_combination_l1214_121490

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (e₁ e₂ : V)

-- Define vectors a and b
def a (e₁ e₂ : V) : V := e₁ + 2 • e₂
def b (e₁ e₂ : V) : V := 3 • e₁ - e₂

-- State the theorem
theorem vector_combination (e₁ e₂ : V) :
  3 • (a e₁ e₂) - 2 • (b e₁ e₂) = -3 • e₁ + 8 • e₂ := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_l1214_121490


namespace NUMINAMATH_CALUDE_raisin_count_l1214_121436

theorem raisin_count (total_raisins : ℕ) (total_boxes : ℕ) (second_box : ℕ) (other_boxes : ℕ) (other_box_count : ℕ) :
  total_raisins = 437 →
  total_boxes = 5 →
  second_box = 74 →
  other_boxes = 97 →
  other_box_count = 3 →
  ∃ (first_box : ℕ), first_box = total_raisins - (second_box + other_box_count * other_boxes) ∧ first_box = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_raisin_count_l1214_121436


namespace NUMINAMATH_CALUDE_total_players_on_ground_l1214_121468

theorem total_players_on_ground (cricket hockey football softball basketball volleyball netball rugby : ℕ) 
  (h1 : cricket = 35)
  (h2 : hockey = 28)
  (h3 : football = 33)
  (h4 : softball = 35)
  (h5 : basketball = 29)
  (h6 : volleyball = 32)
  (h7 : netball = 34)
  (h8 : rugby = 37) :
  cricket + hockey + football + softball + basketball + volleyball + netball + rugby = 263 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l1214_121468


namespace NUMINAMATH_CALUDE_solve_system_l1214_121412

theorem solve_system (u v : ℝ) 
  (eq1 : 3 * u - 7 * v = 29)
  (eq2 : 5 * u + 3 * v = -9) :
  u + v = -3.363 := by sorry

end NUMINAMATH_CALUDE_solve_system_l1214_121412


namespace NUMINAMATH_CALUDE_johns_yearly_oil_change_cost_l1214_121482

/-- Calculates the yearly cost of oil changes for a driver. -/
def yearly_oil_change_cost (miles_per_month : ℕ) (miles_per_oil_change : ℕ) (free_changes_per_year : ℕ) (cost_per_change : ℕ) : ℕ :=
  let changes_per_year := 12 * miles_per_month / miles_per_oil_change
  let paid_changes := changes_per_year - free_changes_per_year
  paid_changes * cost_per_change

/-- Theorem stating that John's yearly oil change cost is $150. -/
theorem johns_yearly_oil_change_cost :
  yearly_oil_change_cost 1000 3000 1 50 = 150 := by
  sorry

#eval yearly_oil_change_cost 1000 3000 1 50

end NUMINAMATH_CALUDE_johns_yearly_oil_change_cost_l1214_121482


namespace NUMINAMATH_CALUDE_equal_candy_sharing_l1214_121407

/-- Represents the number of candies each person has initially -/
structure CandyDistribution :=
  (mark : ℕ)
  (peter : ℕ)
  (john : ℕ)

/-- Calculates the total number of candies -/
def totalCandies (d : CandyDistribution) : ℕ :=
  d.mark + d.peter + d.john

/-- Calculates the number of candies each person gets after equal sharing -/
def sharedCandies (d : CandyDistribution) : ℕ :=
  totalCandies d / 3

/-- Proves that when Mark (30 candies), Peter (25 candies), and John (35 candies)
    combine their candies and share equally, each person will have 30 candies -/
theorem equal_candy_sharing :
  let d : CandyDistribution := { mark := 30, peter := 25, john := 35 }
  sharedCandies d = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_candy_sharing_l1214_121407


namespace NUMINAMATH_CALUDE_cow_count_l1214_121456

/-- Represents the number of cows in a field with given conditions -/
def total_cows (male_cows female_cows : ℕ) : Prop :=
  female_cows = 2 * male_cows ∧
  female_cows / 2 = male_cows / 2 + 50

/-- Proves that the total number of cows is 300 given the conditions -/
theorem cow_count : ∃ (male_cows female_cows : ℕ), 
  total_cows male_cows female_cows ∧ 
  male_cows + female_cows = 300 :=
sorry

end NUMINAMATH_CALUDE_cow_count_l1214_121456


namespace NUMINAMATH_CALUDE_german_french_fraction_l1214_121445

/-- Conference language distribution -/
structure ConferenceLanguages where
  total : ℝ
  english : ℝ
  french : ℝ
  german : ℝ
  english_french : ℝ
  english_german : ℝ
  french_german : ℝ
  all_three : ℝ

/-- Language distribution satisfies the given conditions -/
def ValidDistribution (c : ConferenceLanguages) : Prop :=
  c.english_french = (1/5) * c.english ∧
  c.english_german = (1/3) * c.english ∧
  c.english_french = (1/8) * c.french ∧
  c.french_german = (1/2) * c.french ∧
  c.english_german = (1/6) * c.german

/-- The fraction of German speakers who also speak French is 2/5 -/
theorem german_french_fraction (c : ConferenceLanguages) 
  (h : ValidDistribution c) : 
  c.french_german / c.german = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_german_french_fraction_l1214_121445


namespace NUMINAMATH_CALUDE_buses_needed_for_trip_l1214_121434

/-- Calculates the number of buses needed for a school trip -/
theorem buses_needed_for_trip (total_students : ℕ) (van_students : ℕ) (bus_capacity : ℕ) 
  (h1 : total_students = 500)
  (h2 : van_students = 56)
  (h3 : bus_capacity = 45) :
  Nat.ceil ((total_students - van_students) / bus_capacity) = 10 := by
  sorry

#check buses_needed_for_trip

end NUMINAMATH_CALUDE_buses_needed_for_trip_l1214_121434


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1214_121469

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (current_weight weight_lost : ℕ) 
  (h1 : current_weight = 27)
  (h2 : weight_lost = 101) :
  current_weight + weight_lost = 128 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1214_121469


namespace NUMINAMATH_CALUDE_lowest_cost_option_c_l1214_121419

/-- Represents a shipping option with a flat fee and per-pound rate -/
structure ShippingOption where
  flatFee : ℝ
  perPoundRate : ℝ

/-- Calculates the total cost for a given shipping option and weight -/
def totalCost (option : ShippingOption) (weight : ℝ) : ℝ :=
  option.flatFee + option.perPoundRate * weight

/-- The three shipping options available -/
def optionA : ShippingOption := ⟨5.00, 0.80⟩
def optionB : ShippingOption := ⟨4.50, 0.85⟩
def optionC : ShippingOption := ⟨3.00, 0.95⟩

/-- The weight of the package in pounds -/
def packageWeight : ℝ := 5

theorem lowest_cost_option_c :
  let costA := totalCost optionA packageWeight
  let costB := totalCost optionB packageWeight
  let costC := totalCost optionC packageWeight
  (costC < costA ∧ costC < costB) ∧ costC = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_lowest_cost_option_c_l1214_121419


namespace NUMINAMATH_CALUDE_circle_with_n_integer_points_l1214_121422

/-- A circle in the Euclidean plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of integer points on a circle -/
def num_integer_points (c : Circle) : ℕ :=
  sorry

/-- For any natural number n, there exists a circle with exactly n integer points -/
theorem circle_with_n_integer_points :
  ∀ n : ℕ, ∃ c : Circle, num_integer_points c = n :=
sorry

end NUMINAMATH_CALUDE_circle_with_n_integer_points_l1214_121422


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1214_121417

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) = 3 ↔ y = x * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1214_121417


namespace NUMINAMATH_CALUDE_square_area_ratio_l1214_121451

theorem square_area_ratio (big_side : ℝ) (small_side : ℝ) 
  (h1 : big_side = 12)
  (h2 : small_side = 6) : 
  (small_side ^ 2) / (big_side ^ 2 - small_side ^ 2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1214_121451


namespace NUMINAMATH_CALUDE_b_is_negative_l1214_121405

def is_two_positive_two_negative (a b : ℝ) : Prop :=
  (((a + b > 0) ∧ (a - b > 0)) ∨ ((a + b > 0) ∧ (a * b > 0)) ∨ ((a + b > 0) ∧ (a / b > 0)) ∨
   ((a - b > 0) ∧ (a * b > 0)) ∨ ((a - b > 0) ∧ (a / b > 0)) ∨ ((a * b > 0) ∧ (a / b > 0))) ∧
  (((a + b < 0) ∧ (a - b < 0)) ∨ ((a + b < 0) ∧ (a * b < 0)) ∨ ((a + b < 0) ∧ (a / b < 0)) ∨
   ((a - b < 0) ∧ (a * b < 0)) ∨ ((a - b < 0) ∧ (a / b < 0)) ∨ ((a * b < 0) ∧ (a / b < 0)))

theorem b_is_negative (a b : ℝ) (h : b ≠ 0) (condition : is_two_positive_two_negative a b) : b < 0 := by
  sorry

end NUMINAMATH_CALUDE_b_is_negative_l1214_121405


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1214_121466

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 2 - 2
  ((x + 3) / (x^2 - 1) - 2 / (x - 1)) / ((x + 2) / (x^2 + x)) = Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1214_121466


namespace NUMINAMATH_CALUDE_integral_f_cos_nonnegative_l1214_121421

open MeasureTheory Interval RealInnerProductSpace Set

theorem integral_f_cos_nonnegative 
  (f : ℝ → ℝ) 
  (hf_continuous : ContinuousOn f (Icc 0 (2 * Real.pi)))
  (hf'_continuous : ContinuousOn (deriv f) (Icc 0 (2 * Real.pi)))
  (hf''_continuous : ContinuousOn (deriv^[2] f) (Icc 0 (2 * Real.pi)))
  (hf''_nonneg : ∀ x ∈ Icc 0 (2 * Real.pi), deriv^[2] f x ≥ 0) :
  ∫ x in Icc 0 (2 * Real.pi), f x * Real.cos x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_f_cos_nonnegative_l1214_121421


namespace NUMINAMATH_CALUDE_tip_amount_is_36_dollars_l1214_121437

/-- The cost of a woman's haircut in dollars -/
def womens_haircut_cost : ℝ := 48

/-- The cost of a child's haircut in dollars -/
def childs_haircut_cost : ℝ := 36

/-- The cost of a teenager's haircut in dollars -/
def teens_haircut_cost : ℝ := 40

/-- The cost of Tayzia's hair treatment in dollars -/
def hair_treatment_cost : ℝ := 20

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.20

/-- The total cost of haircuts and treatment before tip -/
def total_cost : ℝ :=
  womens_haircut_cost + 2 * childs_haircut_cost + teens_haircut_cost + hair_treatment_cost

/-- The theorem stating that the 20% tip is $36 -/
theorem tip_amount_is_36_dollars : tip_percentage * total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_tip_amount_is_36_dollars_l1214_121437


namespace NUMINAMATH_CALUDE_animal_population_l1214_121414

theorem animal_population (lions leopards elephants : ℕ) : 
  lions = 200 →
  lions = 2 * leopards →
  elephants = (lions + leopards) / 2 →
  lions + leopards + elephants = 450 := by
sorry

end NUMINAMATH_CALUDE_animal_population_l1214_121414


namespace NUMINAMATH_CALUDE_right_triangles_in_18gon_l1214_121464

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- A right-angled triangle formed by three vertices of a regular polygon --/
structure RightTriangle (p : RegularPolygon n) where
  vertices : Fin 3 → Fin n
  is_right_angled : sorry

/-- The number of right-angled triangles in a regular polygon --/
def num_right_triangles (p : RegularPolygon n) : ℕ :=
  sorry

theorem right_triangles_in_18gon :
  ∀ (p : RegularPolygon 18), num_right_triangles p = 144 :=
sorry

end NUMINAMATH_CALUDE_right_triangles_in_18gon_l1214_121464


namespace NUMINAMATH_CALUDE_bike_distance_l1214_121478

/-- Proves that the distance covered by a bike is 88 miles given the conditions -/
theorem bike_distance (time : ℝ) (truck_distance : ℝ) (speed_difference : ℝ) : 
  time = 8 → 
  truck_distance = 112 → 
  speed_difference = 3 → 
  (truck_distance / time - speed_difference) * time = 88 := by
  sorry

#check bike_distance

end NUMINAMATH_CALUDE_bike_distance_l1214_121478


namespace NUMINAMATH_CALUDE_unanswered_test_completion_ways_l1214_121415

/-- Represents a multiple-choice test -/
structure MultipleChoiceTest where
  num_questions : ℕ
  choices_per_question : ℕ

/-- Calculates the number of ways to complete the test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : ℕ := 1

/-- Theorem: For a 4-question test with 5 choices per question, there is only one way to complete it with all questions unanswered -/
theorem unanswered_test_completion_ways 
  (test : MultipleChoiceTest) 
  (h1 : test.num_questions = 4) 
  (h2 : test.choices_per_question = 5) : 
  ways_to_complete_unanswered test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_completion_ways_l1214_121415


namespace NUMINAMATH_CALUDE_min_value_theorem_l1214_121440

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  6 * x + 1 / x^6 ≥ 7 ∧ ∃ y > 0, 6 * y + 1 / y^6 = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1214_121440


namespace NUMINAMATH_CALUDE_division_inequality_l1214_121492

theorem division_inequality (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_inequality_l1214_121492


namespace NUMINAMATH_CALUDE_sisters_ages_l1214_121484

theorem sisters_ages (s g : ℕ) : 
  (s > 0) → 
  (g > 0) → 
  (1000 ≤ g * 100 + s) → 
  (g * 100 + s < 10000) → 
  (∃ a : ℕ, g * 100 + s = a * a) →
  (∃ b : ℕ, (g + 13) * 100 + (s + 13) = b * b) →
  s + g = 55 := by
sorry

end NUMINAMATH_CALUDE_sisters_ages_l1214_121484


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1214_121427

theorem rectangle_area_perimeter_relation :
  ∀ (a b : ℕ), 
    a ≠ b →                  -- non-square condition
    a > 0 →                  -- positive dimension
    b > 0 →                  -- positive dimension
    a * b = 2 * (2 * a + 2 * b) →  -- area equals twice perimeter
    2 * (a + b) = 36 :=      -- perimeter is 36
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1214_121427


namespace NUMINAMATH_CALUDE_parabola_specific_point_l1214_121488

def parabola_point (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = (y + 2)^2

theorem parabola_specific_point :
  let x : ℝ := Real.sqrt 704
  let y : ℝ := 88
  parabola_point x y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  Real.sqrt (x^2 + (y - 2)^2) = 90 := by sorry

end NUMINAMATH_CALUDE_parabola_specific_point_l1214_121488


namespace NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l1214_121453

/-- The number of walnut trees planted in a park -/
def trees_planted (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the number of trees planted is the difference between final and initial counts -/
theorem walnut_trees_planted (initial : ℕ) (final : ℕ) (h : initial ≤ final) :
  trees_planted initial final = final - initial :=
by sorry

/-- The specific problem instance -/
theorem park_walnut_trees :
  trees_planted 22 55 = 33 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l1214_121453


namespace NUMINAMATH_CALUDE_simplify_fraction_l1214_121441

theorem simplify_fraction : 48 / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1214_121441


namespace NUMINAMATH_CALUDE_juan_running_time_l1214_121403

theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 250) (h2 : speed = 8) :
  distance / speed = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_time_l1214_121403
