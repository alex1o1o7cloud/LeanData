import Mathlib

namespace divisibility_of_fraction_l2456_245697

theorem divisibility_of_fraction (a b n : ℕ) (h1 : a ≠ b) (h2 : n ∣ (a^n - b^n)) :
  n ∣ ((a^n - b^n) / (a - b)) :=
sorry

end divisibility_of_fraction_l2456_245697


namespace knights_on_red_chairs_l2456_245651

structure Room where
  total_chairs : Nat
  knights : Nat
  liars : Nat
  knights_on_red : Nat
  liars_on_blue : Nat

/-- The room satisfies the initial conditions -/
def initial_condition (r : Room) : Prop :=
  r.total_chairs = 20 ∧ 
  r.knights + r.liars = r.total_chairs

/-- The room satisfies the conditions after switching seats -/
def after_switch_condition (r : Room) : Prop :=
  r.knights_on_red + (r.knights - r.knights_on_red) = r.total_chairs / 2 ∧
  (r.liars - r.liars_on_blue) + r.liars_on_blue = r.total_chairs / 2 ∧
  r.knights_on_red = r.liars_on_blue

theorem knights_on_red_chairs (r : Room) 
  (h1 : initial_condition r) 
  (h2 : after_switch_condition r) : 
  r.knights_on_red = 5 := by
  sorry

end knights_on_red_chairs_l2456_245651


namespace remainder_2519_div_3_l2456_245695

theorem remainder_2519_div_3 : 2519 % 3 = 2 := by
  sorry

end remainder_2519_div_3_l2456_245695


namespace square_garden_area_perimeter_difference_l2456_245668

theorem square_garden_area_perimeter_difference :
  ∀ (s : ℝ), 
    s > 0 →
    4 * s = 28 →
    s^2 - 4 * s = 21 :=
by
  sorry

end square_garden_area_perimeter_difference_l2456_245668


namespace max_sum_fourth_powers_l2456_245655

theorem max_sum_fourth_powers (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 16) :
  ∃ (M : ℝ), M = 64 ∧ a^4 + b^4 + c^4 + d^4 ≤ M ∧ 
  ∃ (a' b' c' d' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 = 16 ∧ a'^4 + b'^4 + c'^4 + d'^4 = M :=
by sorry

end max_sum_fourth_powers_l2456_245655


namespace jacks_paycheck_l2456_245630

theorem jacks_paycheck (paycheck : ℝ) : 
  (0.2 * (0.8 * paycheck) = 20) → paycheck = 125 := by
  sorry

end jacks_paycheck_l2456_245630


namespace gcd_lcm_sum_8_12_l2456_245615

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l2456_245615


namespace largest_x_satisfying_equation_l2456_245673

theorem largest_x_satisfying_equation : 
  ∃ x : ℚ, x = 3/25 ∧ 
  (∀ y : ℚ, y ≥ 0 → Real.sqrt (3 * y) = 5 * y → y ≤ x) ∧
  Real.sqrt (3 * x) = 5 * x :=
by sorry

end largest_x_satisfying_equation_l2456_245673


namespace ducks_killed_per_year_is_correct_l2456_245648

/-- The number of ducks killed every year -/
def ducks_killed_per_year : ℕ := 20

/-- The original flock size -/
def original_flock_size : ℕ := 100

/-- The number of ducks born every year -/
def ducks_born_per_year : ℕ := 30

/-- The number of years before joining with another flock -/
def years_before_joining : ℕ := 5

/-- The size of the other flock -/
def other_flock_size : ℕ := 150

/-- The combined flock size after joining -/
def combined_flock_size : ℕ := 300

theorem ducks_killed_per_year_is_correct :
  original_flock_size + years_before_joining * (ducks_born_per_year - ducks_killed_per_year) + other_flock_size = combined_flock_size :=
by sorry

end ducks_killed_per_year_is_correct_l2456_245648


namespace point_movement_on_number_line_l2456_245616

theorem point_movement_on_number_line (A B : ℝ) : 
  A = 7 → B - A = 3 → B = 10 := by sorry

end point_movement_on_number_line_l2456_245616


namespace production_and_salary_optimization_l2456_245609

-- Define the variables and constants
def workday_minutes : ℕ := 8 * 60
def base_salary : ℕ := 100
def type_b_wage : ℚ := 2.5
def total_products : ℕ := 28

-- Define the production time equations
def production_equation_1 (x y : ℚ) : Prop := 6 * x + 4 * y = 170
def production_equation_2 (x y : ℚ) : Prop := 10 * x + 10 * y = 350

-- Define the time constraint
def time_constraint (x y : ℚ) (m : ℕ) : Prop :=
  x * m + y * (total_products - m) ≤ workday_minutes

-- Define the salary function
def salary (a : ℚ) (m : ℕ) : ℚ :=
  a * m + type_b_wage * (total_products - m) + base_salary

-- Theorem statement
theorem production_and_salary_optimization
  (x y : ℚ) (a : ℚ) (h_a : 2 < a ∧ a < 3) :
  (production_equation_1 x y ∧ production_equation_2 x y) →
  (x = 15 ∧ y = 20) ∧
  (∀ m : ℕ, m ≤ total_products →
    time_constraint x y m →
    (2 < a ∧ a < 2.5 → salary a 16 ≥ salary a m) ∧
    (a = 2.5 → salary a m = salary a 16) ∧
    (2.5 < a ∧ a < 3 → salary a 28 ≥ salary a m)) :=
sorry

end production_and_salary_optimization_l2456_245609


namespace total_annual_interest_l2456_245619

def total_investment : ℕ := 3200
def first_part : ℕ := 800
def first_rate : ℚ := 3 / 100
def second_rate : ℚ := 5 / 100

def second_part : ℕ := total_investment - first_part

def interest_first : ℚ := (first_part : ℚ) * first_rate
def interest_second : ℚ := (second_part : ℚ) * second_rate

theorem total_annual_interest :
  interest_first + interest_second = 144 := by sorry

end total_annual_interest_l2456_245619


namespace expression_evaluation_l2456_245669

theorem expression_evaluation (x y z k : ℤ) 
  (hx : x = 25) (hy : y = 12) (hz : z = 3) (hk : k = 4) :
  (x - (y - z)) - ((x - y) - (z + k)) = 10 := by
  sorry

end expression_evaluation_l2456_245669


namespace complex_fraction_equality_l2456_245607

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^3 + a*b + b^3 = 0) : 
  (a^10 + b^10) / (a + b)^10 = 1/18 := by
  sorry

end complex_fraction_equality_l2456_245607


namespace hyperbola_equation_l2456_245680

/-- The standard equation of a hyperbola with given eccentricity and focus -/
theorem hyperbola_equation (e : ℝ) (f : ℝ × ℝ) :
  e = 5/3 →
  f = (0, 5) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), y^2/a^2 - x^2/b^2 = 1 ↔ y^2/9 - x^2/16 = 1) :=
by sorry

end hyperbola_equation_l2456_245680


namespace book_pages_proof_l2456_245602

/-- Calculates the total number of pages in a book given the reading schedule --/
def total_pages (pages_per_day_first_four : ℕ) (pages_per_day_next_two : ℕ) (pages_last_day : ℕ) : ℕ :=
  4 * pages_per_day_first_four + 2 * pages_per_day_next_two + pages_last_day

/-- Proves that the total number of pages in the book is 264 --/
theorem book_pages_proof : total_pages 42 38 20 = 264 := by
  sorry

#eval total_pages 42 38 20

end book_pages_proof_l2456_245602


namespace gcf_72_108_l2456_245658

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l2456_245658


namespace julia_pet_food_cost_l2456_245683

/-- The total amount Julia spent on food for her animals -/
def total_spent (weekly_total : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) (rabbit_cost : ℕ) : ℕ :=
  let parrot_cost := weekly_total - rabbit_cost
  rabbit_weeks * rabbit_cost + parrot_weeks * parrot_cost

/-- Theorem stating the total amount Julia spent on food for her animals -/
theorem julia_pet_food_cost :
  total_spent 30 5 3 12 = 114 := by
  sorry

end julia_pet_food_cost_l2456_245683


namespace isosceles_right_triangle_rotation_volume_l2456_245620

theorem isosceles_right_triangle_rotation_volume :
  ∀ (r h : ℝ), r = 1 → h = 1 →
  (1 / 3 : ℝ) * Real.pi * r^2 * h = Real.pi / 3 := by
sorry

end isosceles_right_triangle_rotation_volume_l2456_245620


namespace fraction_of_half_is_one_seventh_l2456_245692

theorem fraction_of_half_is_one_seventh : (1 : ℚ) / 7 / ((1 : ℚ) / 2) = 2 / 7 := by
  sorry

end fraction_of_half_is_one_seventh_l2456_245692


namespace cone_volume_l2456_245684

/-- Given a cone with slant height 13 cm and height 12 cm, its volume is 100π cubic centimeters -/
theorem cone_volume (s h r : ℝ) (hs : s = 13) (hh : h = 12) 
  (hpythag : s^2 = h^2 + r^2) : (1/3 : ℝ) * π * r^2 * h = 100 * π := by
  sorry

end cone_volume_l2456_245684


namespace negation_of_absolute_value_plus_square_nonnegative_l2456_245640

theorem negation_of_absolute_value_plus_square_nonnegative :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by sorry

end negation_of_absolute_value_plus_square_nonnegative_l2456_245640


namespace quadratic_function_k_value_l2456_245676

theorem quadratic_function_k_value (a b c k : ℤ) :
  let g := λ (x : ℤ) => a * x^2 + b * x + c
  (g 2 = 0) →
  (110 < g 9) →
  (g 9 < 120) →
  (130 < g 10) →
  (g 10 < 140) →
  (6000 * k < g 100) →
  (g 100 < 6000 * (k + 1)) →
  k = 1 := by sorry

end quadratic_function_k_value_l2456_245676


namespace exponent_simplification_l2456_245661

theorem exponent_simplification :
  (10 ^ 0.7) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ (-0.1)) * (10 ^ 0.5) = 100 := by
  sorry

end exponent_simplification_l2456_245661


namespace function_and_inequality_properties_l2456_245690

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |2*x - 1|

-- Define the theorem
theorem function_and_inequality_properties :
  ∀ (a b : ℝ), a ≠ 0 →
  (∀ (x m : ℝ), |b + 2*a| - |2*b - a| ≥ |a| * (|x + 1| + |x - m|)) →
  (∀ (x : ℝ), f x > -5 ↔ x ∈ Set.Ioo (-2) 8) ∧
  (∀ (m : ℝ), m ∈ Set.Icc (-7/2) (3/2)) :=
by sorry

end function_and_inequality_properties_l2456_245690


namespace two_distinct_arrangements_l2456_245638

/-- Represents a face of a cube -/
inductive Face : Type
| F | E | A | B | H | J

/-- Represents an arrangement of numbers on a cube -/
def Arrangement := Face → Fin 6

/-- Two faces are adjacent if they share an edge -/
def adjacent : Face → Face → Prop :=
  sorry

/-- Two numbers are consecutive if they differ by 1 or are 1 and 6 -/
def consecutive (a b : Fin 6) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 5 ∧ b = 0) ∨ (a = 0 ∧ b = 5)

/-- An arrangement is valid if consecutive numbers are on adjacent faces -/
def valid_arrangement (arr : Arrangement) : Prop :=
  ∀ f1 f2 : Face, adjacent f1 f2 → consecutive (arr f1) (arr f2)

/-- Two arrangements are equivalent if they can be transformed into each other
    by cube symmetry or cyclic permutation of numbers -/
def equivalent_arrangements (arr1 arr2 : Arrangement) : Prop :=
  sorry

/-- The number of distinct valid arrangements -/
def num_distinct_arrangements : ℕ :=
  sorry

theorem two_distinct_arrangements :
  num_distinct_arrangements = 2 :=
sorry

end two_distinct_arrangements_l2456_245638


namespace min_value_theorem_l2456_245660

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  (1 / x + 1 / (3 * y)) ≥ 4 ∧ 
  ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ 
    Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 2 ∧
    1 / x₀ + 1 / (3 * y₀) = 4 :=
by sorry

end min_value_theorem_l2456_245660


namespace average_towel_price_l2456_245613

def towel_price_1 : ℕ := 100
def towel_price_2 : ℕ := 150
def towel_price_3 : ℕ := 650

def towel_count_1 : ℕ := 3
def towel_count_2 : ℕ := 5
def towel_count_3 : ℕ := 2

def total_cost : ℕ := towel_price_1 * towel_count_1 + towel_price_2 * towel_count_2 + towel_price_3 * towel_count_3
def total_towels : ℕ := towel_count_1 + towel_count_2 + towel_count_3

theorem average_towel_price :
  total_cost / total_towels = 235 := by sorry

end average_towel_price_l2456_245613


namespace system_solution_l2456_245685

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 4 ∧ 2 * x - y = 2

-- State the theorem
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 2 ∧ y = 2 := by
  sorry

end system_solution_l2456_245685


namespace janet_gained_lives_l2456_245656

/-- Calculates the number of lives Janet gained in a video game level -/
def lives_gained (initial : ℕ) (lost : ℕ) (final : ℕ) : ℕ :=
  final - (initial - lost)

/-- Proves that Janet gained 32 lives in the next level -/
theorem janet_gained_lives : lives_gained 38 16 54 = 32 := by
  sorry

end janet_gained_lives_l2456_245656


namespace factorization_x2_minus_4x_minus_12_minimum_value_4x2_plus_4x_minus_1_l2456_245601

-- Problem 1
theorem factorization_x2_minus_4x_minus_12 :
  ∀ x : ℝ, x^2 - 4*x - 12 = (x - 6) * (x + 2) := by sorry

-- Problem 2
theorem minimum_value_4x2_plus_4x_minus_1 :
  ∀ x : ℝ, 4*x^2 + 4*x - 1 ≥ -2 ∧
  ∃ x : ℝ, 4*x^2 + 4*x - 1 = -2 ∧ x = -1/2 := by sorry

end factorization_x2_minus_4x_minus_12_minimum_value_4x2_plus_4x_minus_1_l2456_245601


namespace complex_simplification_l2456_245614

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Proof that 3(4-2i) + 2i(3+2i) = 8 -/
theorem complex_simplification : 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := by
  sorry

end complex_simplification_l2456_245614


namespace mk97_equality_check_l2456_245612

/-- The MK-97 microcalculator operations -/
class Calculator where
  /-- Check if two numbers are equal -/
  equal : ℝ → ℝ → Prop
  /-- Add two numbers -/
  add : ℝ → ℝ → ℝ
  /-- Find roots of a quadratic equation -/
  quadratic_roots : ℝ → ℝ → Option (ℝ × ℝ)

/-- The theorem to be proved -/
theorem mk97_equality_check (x : ℝ) :
  x = 1 ↔ x ≠ 0 ∧ (4 * (x^2 - x) = 0) :=
sorry

end mk97_equality_check_l2456_245612


namespace problem_solution_l2456_245645

/-- Calculates the total number of new cans that can be made from a given number of cans,
    considering that newly made cans can also be recycled. -/
def totalNewCans (initialCans : ℕ) (damagedCans : ℕ) (requiredForNewCan : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions of the problem,
    the total number of new cans that can be made is 95. -/
theorem problem_solution :
  totalNewCans 500 20 6 = 95 := by
  sorry

end problem_solution_l2456_245645


namespace overlapping_circles_common_chord_l2456_245611

theorem overlapping_circles_common_chord 
  (r : ℝ) 
  (h : r = 12) : 
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 12 * Real.sqrt 3 := by
  sorry

end overlapping_circles_common_chord_l2456_245611


namespace q_must_be_false_l2456_245688

theorem q_must_be_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_must_be_false_l2456_245688


namespace function_characterization_l2456_245642

theorem function_characterization (a : ℝ) (ha : a > 0) :
  ∀ (f : ℕ → ℝ),
    (∀ (k m : ℕ), k > 0 ∧ m > 0 ∧ a * m ≤ k ∧ k < (a + 1) * m → f (k + m) = f k + f m) ↔
    ∃ (b : ℝ), ∀ (n : ℕ), f n = b * n :=
by sorry

end function_characterization_l2456_245642


namespace complex_power_eight_l2456_245631

theorem complex_power_eight : (2 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^8 = -128 - 128 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_eight_l2456_245631


namespace age_difference_l2456_245698

/-- Represents the ages of four people: Patrick, Michael, Monica, and Nathan. -/
structure Ages where
  patrick : ℝ
  michael : ℝ
  monica : ℝ
  nathan : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.patrick / ages.michael = 3 / 5 ∧
  ages.michael / ages.monica = 3 / 5 ∧
  ages.monica / ages.nathan = 4 / 7 ∧
  ages.patrick + ages.michael + ages.monica + ages.nathan = 142

/-- The theorem stating the difference between Patrick's and Nathan's ages -/
theorem age_difference (ages : Ages) (h : satisfies_conditions ages) :
  ∃ ε > 0, |ages.patrick - ages.nathan - 1.46| < ε :=
sorry

end age_difference_l2456_245698


namespace smallest_undefined_value_l2456_245624

theorem smallest_undefined_value (y : ℝ) :
  let f := fun y : ℝ => (y - 3) / (9 * y^2 - 56 * y + 7)
  let roots := {y : ℝ | 9 * y^2 - 56 * y + 7 = 0}
  ∃ (smallest : ℝ), smallest ∈ roots ∧ 
    (∀ y ∈ roots, y ≥ smallest) ∧
    (∀ z < smallest, f z ≠ 0⁻¹) :=
by sorry

end smallest_undefined_value_l2456_245624


namespace a_3_value_l2456_245682

def S (n : ℕ+) : ℕ := 5 * n.val ^ 2 + 10 * n.val

theorem a_3_value : ∃ (a : ℕ+ → ℕ), a 3 = 35 :=
  sorry

end a_3_value_l2456_245682


namespace probability_a_squared_geq_4b_l2456_245699

-- Define the set of numbers
def S : Set Nat := {1, 2, 3, 4}

-- Define the condition
def condition (a b : Nat) : Prop := a^2 ≥ 4*b

-- Define the total number of ways to select two numbers
def total_selections : Nat := 12

-- Define the number of favorable selections
def favorable_selections : Nat := 6

-- State the theorem
theorem probability_a_squared_geq_4b :
  (favorable_selections : ℚ) / total_selections = 1 / 2 := by sorry

end probability_a_squared_geq_4b_l2456_245699


namespace least_number_divisible_l2456_245662

theorem least_number_divisible (n : ℕ) : n = 858 ↔ 
  (∀ m : ℕ, m < n → 
    ¬((m + 6) % 24 = 0 ∧ 
      (m + 6) % 32 = 0 ∧ 
      (m + 6) % 36 = 0 ∧ 
      (m + 6) % 54 = 0)) ∧
  ((n + 6) % 24 = 0 ∧ 
   (n + 6) % 32 = 0 ∧ 
   (n + 6) % 36 = 0 ∧ 
   (n + 6) % 54 = 0) :=
by sorry

end least_number_divisible_l2456_245662


namespace chicken_egg_production_roberto_chicken_problem_l2456_245634

/-- Represents the problem of determining the number of eggs each chicken needs to produce per week -/
theorem chicken_egg_production (num_chickens : ℕ) (chicken_cost : ℚ) (weekly_feed_cost : ℚ) 
  (eggs_per_dozen : ℕ) (dozen_cost : ℚ) (weeks : ℕ) : ℚ :=
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weeks * weekly_feed_cost
  let total_chicken_expenses := total_chicken_cost + total_feed_cost
  let total_egg_expenses := weeks * dozen_cost
  let eggs_per_week := eggs_per_dozen
  (eggs_per_week / num_chickens : ℚ)

/-- Proves that each chicken needs to produce 3 eggs per week to be cheaper than buying eggs after 81 weeks -/
theorem roberto_chicken_problem : 
  chicken_egg_production 4 20 1 12 2 81 = 3 := by
  sorry

end chicken_egg_production_roberto_chicken_problem_l2456_245634


namespace inequality_proof_l2456_245677

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a :=
by sorry

end inequality_proof_l2456_245677


namespace unique_solution_exponential_system_l2456_245696

theorem unique_solution_exponential_system :
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 →
  (x^y = z ∧ y^z = x ∧ z^x = y) →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_solution_exponential_system_l2456_245696


namespace angle_trisection_l2456_245635

theorem angle_trisection (n : ℕ) (h : ¬ 3 ∣ n) :
  ∃ (a b : ℤ), 3 * a + n * b = 1 :=
by sorry

end angle_trisection_l2456_245635


namespace student_grades_l2456_245646

theorem student_grades (grade1 grade2 grade3 : ℚ) : 
  grade1 = 60 → 
  grade3 = 85 → 
  (grade1 + grade2 + grade3) / 3 = 75 → 
  grade2 = 80 := by
sorry

end student_grades_l2456_245646


namespace unique_solution_of_equation_l2456_245663

theorem unique_solution_of_equation :
  ∃! x : ℝ, x ≠ 3 ∧ x + 36 / (x - 3) = -9 :=
by
  -- The proof goes here
  sorry

end unique_solution_of_equation_l2456_245663


namespace speeding_ticket_percentage_l2456_245657

/-- The percentage of motorists who exceed the speed limit -/
def exceed_speed_limit : ℝ := 12.5

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 20

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket_percentage : ℝ := 10

/-- Theorem stating that the percentage of motorists receiving speeding tickets is 10% -/
theorem speeding_ticket_percentage :
  receive_ticket_percentage = exceed_speed_limit * (100 - no_ticket_percentage) / 100 := by
  sorry

end speeding_ticket_percentage_l2456_245657


namespace levi_has_five_lemons_l2456_245628

/-- The number of lemons each person has -/
structure LemonCounts where
  levi : ℕ
  jayden : ℕ
  eli : ℕ
  ian : ℕ

/-- The conditions of the lemon problem -/
def LemonProblem (counts : LemonCounts) : Prop :=
  counts.jayden = counts.levi + 6 ∧
  counts.jayden * 3 = counts.eli ∧
  counts.eli * 2 = counts.ian ∧
  counts.levi + counts.jayden + counts.eli + counts.ian = 115

/-- Theorem stating that under the given conditions, Levi has 5 lemons -/
theorem levi_has_five_lemons :
  ∃ (counts : LemonCounts), LemonProblem counts ∧ counts.levi = 5 := by
  sorry

end levi_has_five_lemons_l2456_245628


namespace least_n_factorial_divisible_by_840_l2456_245654

theorem least_n_factorial_divisible_by_840 :
  ∀ n : ℕ, n > 0 → (n.factorial % 840 = 0) → n ≥ 7 :=
by sorry

end least_n_factorial_divisible_by_840_l2456_245654


namespace clock_angle_at_8_l2456_245681

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees per hour on a clock face -/
def degrees_per_hour : ℕ := 360 / clock_hours

/-- The position of the minute hand at 8:00 in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 8:00 in degrees -/
def hour_hand_position : ℕ := 8 * degrees_per_hour

/-- The smaller angle between the hour and minute hands at 8:00 -/
def smaller_angle : ℕ := min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position))

theorem clock_angle_at_8 : smaller_angle = 120 := by
  sorry

end clock_angle_at_8_l2456_245681


namespace z_in_fourth_quadrant_implies_a_in_open_interval_l2456_245603

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 + a * Complex.I) * (1 - Complex.I)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant_implies_a_in_open_interval :
  ∀ a : ℝ, in_fourth_quadrant (z a) → -1 < a ∧ a < 1 := by
  sorry

end z_in_fourth_quadrant_implies_a_in_open_interval_l2456_245603


namespace parallel_vectors_dot_product_l2456_245691

/-- Given two vectors a and b in ℝ², if a is parallel to b and a = (1,3) and b = (-3,x), 
    then their dot product is -30 -/
theorem parallel_vectors_dot_product (x : ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (-3, x)
  (∃ (k : ℝ), b = k • a) → a.1 * b.1 + a.2 * b.2 = -30 := by
  sorry

end parallel_vectors_dot_product_l2456_245691


namespace y1_greater_than_y2_l2456_245621

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 1

-- Define the theorem
theorem y1_greater_than_y2 (y1 y2 : ℝ) 
  (h1 : line_equation (-3) y1) 
  (h2 : line_equation (-5) y2) : 
  y1 > y2 := by
  sorry

end y1_greater_than_y2_l2456_245621


namespace train_passing_time_train_passes_jogger_in_40_seconds_l2456_245671

/-- Calculates the time for a train to pass a jogger given their initial speeds,
    distances, and speed reduction due to incline. -/
theorem train_passing_time (jogger_speed train_speed : ℝ)
                           (initial_distance train_length : ℝ)
                           (incline_reduction : ℝ) : ℝ :=
  let jogger_effective_speed := jogger_speed * (1 - incline_reduction)
  let train_effective_speed := train_speed * (1 - incline_reduction)
  let relative_speed := train_effective_speed - jogger_effective_speed
  let total_distance := initial_distance + train_length
  total_distance / relative_speed * (3600 / 1000)

/-- The time for the train to pass the jogger is 40 seconds. -/
theorem train_passes_jogger_in_40_seconds :
  train_passing_time 9 45 240 120 0.1 = 40 := by
  sorry


end train_passing_time_train_passes_jogger_in_40_seconds_l2456_245671


namespace borel_sets_closed_under_countable_operations_l2456_245629

-- Define the σ-algebra of Borel sets
def BorelSets : Set (Set ℝ) := sorry

-- Define the property of being generated by open sets
def GeneratedByOpenSets (S : Set (Set ℝ)) : Prop := sorry

-- Define closure under countable union
def ClosedUnderCountableUnion (S : Set (Set ℝ)) : Prop := sorry

-- Define closure under countable intersection
def ClosedUnderCountableIntersection (S : Set (Set ℝ)) : Prop := sorry

-- Theorem statement
theorem borel_sets_closed_under_countable_operations :
  GeneratedByOpenSets BorelSets →
  ClosedUnderCountableUnion BorelSets ∧ ClosedUnderCountableIntersection BorelSets := by
  sorry

end borel_sets_closed_under_countable_operations_l2456_245629


namespace same_prime_factors_implies_power_of_two_l2456_245632

theorem same_prime_factors_implies_power_of_two (b m n : ℕ) 
  (hb : b ≠ 1) (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k := by
sorry

end same_prime_factors_implies_power_of_two_l2456_245632


namespace fixed_point_on_line_l2456_245623

/-- The line (m-1)x+(2m-1)y=m-5 always passes through the point (9, -4) for all real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end fixed_point_on_line_l2456_245623


namespace correct_point_satisfies_conditions_l2456_245604

def point_satisfies_conditions (x y : ℝ) : Prop :=
  -2 < x ∧ x < 0 ∧ 2 < y ∧ y < 4

theorem correct_point_satisfies_conditions :
  point_satisfies_conditions (-1) 3 ∧
  ¬ point_satisfies_conditions 1 3 ∧
  ¬ point_satisfies_conditions 1 (-3) ∧
  ¬ point_satisfies_conditions (-3) 1 ∧
  ¬ point_satisfies_conditions 3 (-1) :=
by sorry

end correct_point_satisfies_conditions_l2456_245604


namespace range_of_a_l2456_245600

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ |x - 5| + |x - 3|) → a ≤ 2 := by
  sorry

end range_of_a_l2456_245600


namespace triangle_problem_l2456_245637

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 2) :
  Real.cos t.B = 15/17 ∧ t.b = 2 := by
  sorry


end triangle_problem_l2456_245637


namespace sequence_formula_l2456_245647

def S₁ (n : ℕ) : ℕ := n^2

def S₂ (n : ℕ) : ℕ := n^2 + n + 1

def a₁ (n : ℕ) : ℕ := 2*n - 1

def a₂ (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2*n

theorem sequence_formula (n : ℕ) (h : n ≥ 1) :
  (∀ k, S₁ k - S₁ (k-1) = a₁ k) ∧
  (∀ k, S₂ k - S₂ (k-1) = a₂ k) :=
sorry

end sequence_formula_l2456_245647


namespace detergent_quarts_in_altered_solution_l2456_245675

/-- Represents the ratio of bleach : detergent : water in a cleaning solution -/
structure CleaningSolution :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

/-- Calculates the amount of detergent in quarts given the conditions of the problem -/
def calculate_detergent_quarts (original : CleaningSolution) (water_gallons : ℚ) : ℚ :=
  let new_ratio := CleaningSolution.mk 
    (original.bleach * 3) 
    original.detergent
    (original.water / 2)
  let total_parts := new_ratio.bleach + new_ratio.detergent + new_ratio.water
  let detergent_gallons := (new_ratio.detergent / new_ratio.water) * water_gallons
  detergent_gallons * 4

/-- Theorem stating that the altered solution will contain 160 quarts of detergent -/
theorem detergent_quarts_in_altered_solution :
  let original := CleaningSolution.mk 2 25 100
  calculate_detergent_quarts original 80 = 160 := by
  sorry


end detergent_quarts_in_altered_solution_l2456_245675


namespace student_distribution_theorem_l2456_245627

/-- The number of ways to distribute n students to k towns, ensuring each town receives at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to choose r items from n items. -/
def binomial_coefficient (n : ℕ) (r : ℕ) : ℕ :=
  sorry

/-- The number of permutations of n items. -/
def permutations (n : ℕ) : ℕ :=
  sorry

theorem student_distribution_theorem :
  distribute_students 4 3 = 36 :=
by sorry

end student_distribution_theorem_l2456_245627


namespace f_times_g_equals_one_l2456_245626

/-- The formal power series f(x) defined as an infinite geometric series -/
noncomputable def f (x : ℝ) : ℝ := ∑' n, x^n

/-- The function g(x) defined as 1 - x -/
def g (x : ℝ) : ℝ := 1 - x

/-- Theorem stating that f(x)g(x) = 1 -/
theorem f_times_g_equals_one (x : ℝ) (hx : |x| < 1) : f x * g x = 1 := by
  sorry

end f_times_g_equals_one_l2456_245626


namespace correct_number_of_men_l2456_245606

/-- The number of men in the first group that completes a job in 15 days,
    given that 25 men can finish the same job in 18 days. -/
def number_of_men : ℕ := 30

/-- The number of days taken by the first group to complete the job. -/
def days_first_group : ℕ := 15

/-- The number of men in the second group. -/
def men_second_group : ℕ := 25

/-- The number of days taken by the second group to complete the job. -/
def days_second_group : ℕ := 18

/-- Theorem stating that the number of men in the first group is correct. -/
theorem correct_number_of_men :
  number_of_men * days_first_group = men_second_group * days_second_group :=
sorry

end correct_number_of_men_l2456_245606


namespace line_parameterization_l2456_245689

/-- Given a line y = 2x - 30 parameterized by (x,y) = (g(t), 12t - 10), 
    prove that g(t) = 6t + 10 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t : ℝ, 12 * t - 10 = 2 * g t - 30) → 
  (∀ t : ℝ, g t = 6 * t + 10) := by
sorry

end line_parameterization_l2456_245689


namespace brush_chess_prices_l2456_245639

theorem brush_chess_prices (brush_price chess_price : ℚ) : 
  (5 * brush_price + 12 * chess_price = 315) →
  (8 * brush_price + 6 * chess_price = 240) →
  (brush_price = 15 ∧ chess_price = 20) := by
sorry

end brush_chess_prices_l2456_245639


namespace inequality_for_positive_numbers_l2456_245672

theorem inequality_for_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (a + 1)⁻¹ + (b + 1)⁻¹ ≥ 4/3 := by sorry

end inequality_for_positive_numbers_l2456_245672


namespace jason_has_21_toys_l2456_245643

/-- The number of toys Rachel has -/
def rachel_toys : ℕ := 1

/-- The number of toys John has -/
def john_toys : ℕ := rachel_toys + 6

/-- The number of toys Jason has -/
def jason_toys : ℕ := 3 * john_toys

/-- Theorem: Jason has 21 toys -/
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end jason_has_21_toys_l2456_245643


namespace cow_count_is_seven_l2456_245687

/-- Represents the number of animals in the group -/
structure AnimalCount where
  cows : ℕ
  chickens : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ := 4 * ac.cows + 2 * ac.chickens

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ := ac.cows + ac.chickens

/-- The main theorem stating that if the number of legs is 14 more than twice the number of heads,
    then the number of cows is 7 -/
theorem cow_count_is_seven (ac : AnimalCount) :
  totalLegs ac = 2 * totalHeads ac + 14 → ac.cows = 7 := by
  sorry


end cow_count_is_seven_l2456_245687


namespace sin_cos_difference_identity_l2456_245644

theorem sin_cos_difference_identity :
  Real.sin (47 * π / 180) * Real.cos (17 * π / 180) - 
  Real.cos (47 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_difference_identity_l2456_245644


namespace solve_equation_for_m_l2456_245641

theorem solve_equation_for_m : ∃ m : ℤ, 
  62519 * 9999^2 / 314 * (314 - m) = 547864 ∧ m = -547550 := by
  sorry

end solve_equation_for_m_l2456_245641


namespace least_positive_integer_with_remainders_l2456_245664

theorem least_positive_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 4) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 → m ≥ n) ∧
  n = 59 := by
  sorry

end least_positive_integer_with_remainders_l2456_245664


namespace hexagon_puzzle_solution_l2456_245665

/-- Represents the positions in the hexagon puzzle --/
inductive Position
| A | B | C | D | E | F

/-- Represents a valid assignment of digits to positions --/
def Assignment := Position → Fin 6

/-- Checks if an assignment is valid (uses each digit exactly once) --/
def isValidAssignment (a : Assignment) : Prop :=
  ∀ (i : Fin 6), ∃! (p : Position), a p = i

/-- Checks if an assignment satisfies the sum condition for all lines --/
def satisfiesSumCondition (a : Assignment) : Prop :=
  (a Position.A + a Position.C + 9 = 15) ∧
  (a Position.A + 8 + a Position.F = 15) ∧
  (7 + a Position.C + a Position.E = 15) ∧
  (7 + a Position.D + a Position.F = 15) ∧
  (9 + a Position.B + a Position.D = 15) ∧
  (a Position.A + a Position.D + a Position.E = 15)

/-- The main theorem stating the existence and uniqueness of a valid solution --/
theorem hexagon_puzzle_solution :
  ∃! (a : Assignment), isValidAssignment a ∧ satisfiesSumCondition a :=
sorry

end hexagon_puzzle_solution_l2456_245665


namespace product_of_reciprocal_minus_one_bound_l2456_245659

theorem product_of_reciprocal_minus_one_bound 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end product_of_reciprocal_minus_one_bound_l2456_245659


namespace probability_two_acceptable_cans_l2456_245605

theorem probability_two_acceptable_cans (total_cans : Nat) (acceptable_cans : Nat) 
  (h1 : total_cans = 6)
  (h2 : acceptable_cans = 4) : 
  (Nat.choose acceptable_cans 2 : ℚ) / (Nat.choose total_cans 2 : ℚ) = 2/5 := by
  sorry

end probability_two_acceptable_cans_l2456_245605


namespace new_stereo_price_l2456_245666

theorem new_stereo_price 
  (old_cost : ℝ) 
  (trade_in_percentage : ℝ) 
  (new_discount_percentage : ℝ) 
  (out_of_pocket : ℝ) 
  (h1 : old_cost = 250)
  (h2 : trade_in_percentage = 0.8)
  (h3 : new_discount_percentage = 0.25)
  (h4 : out_of_pocket = 250) :
  let trade_in_value := old_cost * trade_in_percentage
  let total_spent := trade_in_value + out_of_pocket
  let original_price := total_spent / (1 - new_discount_percentage)
  original_price = 600 := by sorry

end new_stereo_price_l2456_245666


namespace count_multiples_of_seven_between_squares_l2456_245694

theorem count_multiples_of_seven_between_squares : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, n % 7 = 0 ∧ (17 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 18) ∧
    (∀ n : ℕ, n % 7 = 0 ∧ (17 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 18 → n ∈ s) ∧
    Finset.card s = 5 := by
  sorry

end count_multiples_of_seven_between_squares_l2456_245694


namespace square_ends_in_six_tens_digit_odd_l2456_245652

theorem square_ends_in_six_tens_digit_odd (n : ℤ) : 
  n^2 % 100 = 6 → (n^2 / 10) % 2 = 1 := by
  sorry

end square_ends_in_six_tens_digit_odd_l2456_245652


namespace parallel_vectors_x_value_l2456_245649

def vec_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-4, 2)  -- Derived from a - (1/2)b = (3,1)
  let c : ℝ × ℝ := (x, 3)
  vec_parallel (2 * a + b) c → x = -1 :=
by sorry

end parallel_vectors_x_value_l2456_245649


namespace distance_BD_l2456_245653

/-- Given three points B, C, and D in a 2D plane, prove that the distance between B and D is 13. -/
theorem distance_BD (B C D : ℝ × ℝ) : 
  B = (3, 9) → C = (3, -3) → D = (-2, -3) → 
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 13 := by
  sorry

end distance_BD_l2456_245653


namespace min_even_integers_l2456_245618

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 →
  a + b + c + d = 50 →
  a + b + c + d + e + f = 70 →
  Even e →
  Even f →
  (∃ (count : ℕ), count ≥ 2 ∧ 
    count = (if Even a then 1 else 0) +
            (if Even b then 1 else 0) +
            (if Even c then 1 else 0) +
            (if Even d then 1 else 0) + 2) :=
by sorry

end min_even_integers_l2456_245618


namespace work_speed_l2456_245650

/-- Proves that given a round trip of 2 hours, 72 minutes to work, and 90 km/h return speed, the speed to work is 60 km/h -/
theorem work_speed (total_time : Real) (time_to_work : Real) (return_speed : Real) :
  total_time = 2 ∧ 
  time_to_work = 72 / 60 ∧ 
  return_speed = 90 →
  (2 * return_speed * time_to_work) / (total_time + time_to_work) = 60 := by
  sorry

end work_speed_l2456_245650


namespace river_depth_problem_l2456_245670

theorem river_depth_problem (d k : ℝ) : 
  (d + 0.5 * d + k = 1.5 * (d + 0.5 * d)) →  -- Depth in mid-July is 1.5 times the depth at the end of May
  (1.5 * (d + 0.5 * d) = 45) →               -- Final depth in mid-July is 45 feet
  (d = 15 ∧ k = 11.25) :=                    -- Initial depth is 15 feet and depth increase in June is 11.25 feet
by sorry

end river_depth_problem_l2456_245670


namespace arithmetic_sequence_common_difference_l2456_245610

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (a 1 + a n) / 2
  sum_13 : sum 13 = 104
  a_6 : a 6 = 5

/-- The common difference of the arithmetic sequence is 3 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) : seq.d = 3 := by
  sorry

end arithmetic_sequence_common_difference_l2456_245610


namespace base_n_1001_not_prime_l2456_245622

/-- For a positive integer n ≥ 2, 1001_n represents n^3 + 1 in base 10 -/
def base_n_1001 (n : ℕ) : ℕ := n^3 + 1

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (m : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < m ∧ m % k = 0

theorem base_n_1001_not_prime : 
  ∀ n : ℕ, n ≥ 2 → is_composite (base_n_1001 n) := by
  sorry

end base_n_1001_not_prime_l2456_245622


namespace binomial_coefficient_problem_l2456_245636

theorem binomial_coefficient_problem (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x, (2 + x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625 := by
sorry

end binomial_coefficient_problem_l2456_245636


namespace smallest_n_congruence_l2456_245678

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 7 → (7^m : ℤ) % 5 ≠ m^7 % 5) ∧ 
  (7^7 : ℤ) % 5 = 7^7 % 5 := by
sorry

end smallest_n_congruence_l2456_245678


namespace positive_sum_from_absolute_difference_l2456_245608

theorem positive_sum_from_absolute_difference (a b : ℝ) : 
  b - |a| > 0 → a + b > 0 := by
  sorry

end positive_sum_from_absolute_difference_l2456_245608


namespace largest_three_digit_multiple_of_8_with_digit_sum_24_l2456_245686

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 24 → n ≤ 888 :=
by sorry

end largest_three_digit_multiple_of_8_with_digit_sum_24_l2456_245686


namespace function_analysis_l2456_245617

def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x + a - 1| - a^2 - 2*a

theorem function_analysis (a : ℝ) :
  (a = 3 → {x : ℝ | f 3 x ≥ -10} = {x : ℝ | x ≥ 3 ∨ x ≤ -1}) ∧
  ({a : ℝ | ∀ x, f a x ≥ 0} = {a : ℝ | -2 ≤ a ∧ a ≤ 0}) :=
by sorry

end function_analysis_l2456_245617


namespace range_of_a_l2456_245679

-- Define the propositions p and q
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the sets A and B
def A : Set ℝ := {x | p x}
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(p x) ∧ q x a)) →
  (∃ S : Set ℝ, S = {a : ℝ | 0 ≤ a ∧ a ≤ 1/2} ∧ 
  (∀ a : ℝ, a ∈ S ↔ 
    (∀ x : ℝ, x ∈ A → x ∈ B a) ∧ 
    (∃ x : ℝ, x ∉ A ∧ x ∈ B a))) :=
sorry

end range_of_a_l2456_245679


namespace parker_dumbbell_weight_l2456_245674

/-- Given an initial setup of dumbbells and additional dumbbells added, 
    calculate the total weight Parker is using for his exercises. -/
theorem parker_dumbbell_weight 
  (initial_count : ℕ) 
  (additional_count : ℕ) 
  (weight_per_dumbbell : ℕ) : 
  initial_count = 4 → 
  additional_count = 2 → 
  weight_per_dumbbell = 20 → 
  (initial_count + additional_count) * weight_per_dumbbell = 120 := by
  sorry

end parker_dumbbell_weight_l2456_245674


namespace perfect_square_trinomial_m_value_l2456_245693

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number k such that
    ax^2 + bx + c = (x + k)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + k)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 1 m 9 → m = 6 ∨ m = -6 :=
by sorry

end perfect_square_trinomial_m_value_l2456_245693


namespace trigonometric_equation_system_solution_l2456_245625

theorem trigonometric_equation_system_solution :
  ∃ (x y : ℝ),
    3 * Real.cos x + 4 * Real.sin x = -1.4 ∧
    13 * Real.cos x - 41 * Real.cos y = -45 ∧
    13 * Real.sin x + 41 * Real.sin y = 3 :=
by sorry

end trigonometric_equation_system_solution_l2456_245625


namespace books_sold_in_three_days_l2456_245633

/-- The total number of books sold over three days -/
def total_books_sold (tuesday_sales wednesday_sales thursday_sales : ℕ) : ℕ :=
  tuesday_sales + wednesday_sales + thursday_sales

/-- Theorem stating the total number of books sold over three days -/
theorem books_sold_in_three_days :
  ∃ (tuesday_sales wednesday_sales thursday_sales : ℕ),
    tuesday_sales = 7 ∧
    wednesday_sales = 3 * tuesday_sales ∧
    thursday_sales = 3 * wednesday_sales ∧
    total_books_sold tuesday_sales wednesday_sales thursday_sales = 91 := by
  sorry

end books_sold_in_three_days_l2456_245633


namespace systematic_sampling_smallest_number_l2456_245667

theorem systematic_sampling_smallest_number
  (n : ℕ) -- Total number of products
  (k : ℕ) -- Sample size
  (x : ℕ) -- A number in the sample
  (h1 : n = 80) -- Total number of products is 80
  (h2 : k = 5) -- Sample size is 5
  (h3 : x = 42) -- The number 42 is in the sample
  (h4 : x < n) -- The number in the sample is less than the total number of products
  : ∃ (interval : ℕ) (smallest : ℕ),
    interval = n / k ∧
    x = interval * 2 + smallest ∧
    smallest = 10 :=
by sorry

end systematic_sampling_smallest_number_l2456_245667
