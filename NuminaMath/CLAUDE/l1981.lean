import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1981_198182

theorem quadratic_inequality_solution_set (m : ℝ) :
  m > 2 → ∀ x : ℝ, x^2 - 2*x + m > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1981_198182


namespace NUMINAMATH_CALUDE_employee_pay_l1981_198188

theorem employee_pay (total_pay m_pay n_pay : ℝ) : 
  total_pay = 550 →
  m_pay = 1.2 * n_pay →
  m_pay + n_pay = total_pay →
  n_pay = 250 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l1981_198188


namespace NUMINAMATH_CALUDE_fraction_simplification_l1981_198172

theorem fraction_simplification (b x : ℝ) (h : b^2 + x^4 ≠ 0) :
  (Real.sqrt (b^2 + x^4) - (x^4 - b^2) / (2 * Real.sqrt (b^2 + x^4))) / (b^2 + x^4) =
  (3 * b^2 + x^4) / (2 * (b^2 + x^4)^(3/2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1981_198172


namespace NUMINAMATH_CALUDE_new_shoes_duration_l1981_198103

/-- Given information about shoe costs and durability, prove the duration of new shoes. -/
theorem new_shoes_duration (used_repair_cost : ℝ) (used_duration : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  used_repair_cost = 13.50 →
  used_duration = 1 →
  new_cost = 32.00 →
  cost_increase_percentage = 0.1852 →
  let new_duration := new_cost / (used_repair_cost * (1 + cost_increase_percentage))
  new_duration = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_duration_l1981_198103


namespace NUMINAMATH_CALUDE_meaningful_expression_l1981_198135

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1981_198135


namespace NUMINAMATH_CALUDE_f_plus_one_is_odd_l1981_198130

-- Define the property of the function f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

-- Theorem statement
theorem f_plus_one_is_odd (f : ℝ → ℝ) (h : satisfies_property f) :
  is_odd (fun x => f x + 1) :=
sorry

end NUMINAMATH_CALUDE_f_plus_one_is_odd_l1981_198130


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1981_198126

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ → Prop :=
  λ b => a + b = 0

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 (-2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1981_198126


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1981_198104

theorem invalid_external_diagonals : ¬ ∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (8 = Real.sqrt (a^2 + b^2) ∨ 8 = Real.sqrt (b^2 + c^2) ∨ 8 = Real.sqrt (a^2 + c^2)) ∧
  (15 = Real.sqrt (a^2 + b^2) ∨ 15 = Real.sqrt (b^2 + c^2) ∨ 15 = Real.sqrt (a^2 + c^2)) ∧
  (18 = Real.sqrt (a^2 + b^2) ∨ 18 = Real.sqrt (b^2 + c^2) ∨ 18 = Real.sqrt (a^2 + c^2)) :=
by sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1981_198104


namespace NUMINAMATH_CALUDE_ninas_school_students_l1981_198171

theorem ninas_school_students : ∀ (n m : ℕ),
  n = 5 * m →
  n + m = 4800 →
  (n - 200) + (m + 200) = 2 * (m + 200) →
  n = 4000 := by
sorry

end NUMINAMATH_CALUDE_ninas_school_students_l1981_198171


namespace NUMINAMATH_CALUDE_xyz_value_l1981_198117

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x*y + x*z + y*z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = 11) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1981_198117


namespace NUMINAMATH_CALUDE_inequality_proof_l1981_198154

/-- The function f(x) defined as |x-a| + |x-3| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

/-- Theorem: Given the conditions, prove that m + 2n ≥ 2 -/
theorem inequality_proof (a m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : Set.Icc 1 3 = {x | f a x ≤ 1 + |x - 3|})
  (h_a : 1/m + 1/(2*n) = a) : m + 2*n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1981_198154


namespace NUMINAMATH_CALUDE_difference_of_squares_l1981_198159

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1981_198159


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l1981_198157

theorem sum_abcd_equals_negative_twenty_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) : 
  a + b + c + d = -20/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l1981_198157


namespace NUMINAMATH_CALUDE_basement_pump_time_l1981_198146

-- Define the constants
def basement_length : ℝ := 30
def basement_width : ℝ := 40
def water_depth : ℝ := 2
def initial_pumps : ℕ := 4
def pump_capacity : ℝ := 10
def breakdown_time : ℝ := 120
def cubic_foot_to_gallon : ℝ := 7.5

-- Define the theorem
theorem basement_pump_time :
  let initial_volume : ℝ := basement_length * basement_width * water_depth * cubic_foot_to_gallon
  let initial_pump_rate : ℝ := initial_pumps * pump_capacity
  let volume_pumped_before_breakdown : ℝ := initial_pump_rate * breakdown_time
  let remaining_volume : ℝ := initial_volume - volume_pumped_before_breakdown
  let remaining_pumps : ℕ := initial_pumps - 1
  let remaining_pump_rate : ℝ := remaining_pumps * pump_capacity
  let remaining_time : ℝ := remaining_volume / remaining_pump_rate
  breakdown_time + remaining_time = 560 := by
  sorry

end NUMINAMATH_CALUDE_basement_pump_time_l1981_198146


namespace NUMINAMATH_CALUDE_complex_product_sum_l1981_198120

theorem complex_product_sum (i : ℂ) : i * i = -1 →
  let z := (1 + i) * (1 - i)
  let p := z.re
  let q := z.im
  p + q = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_product_sum_l1981_198120


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l1981_198125

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (max : ℕ), 
    (∀ (n : ℕ), n > 0 → Nat.gcd (13*n + 4) (8*n + 3) ≤ max) ∧ 
    (∃ (n : ℕ), n > 0 ∧ Nat.gcd (13*n + 4) (8*n + 3) = max)) ∧
  (∀ (m : ℕ), 
    (∀ (n : ℕ), n > 0 → Nat.gcd (13*n + 4) (8*n + 3) ≤ m) →
    (∃ (n : ℕ), n > 0 ∧ Nat.gcd (13*n + 4) (8*n + 3) = m) →
    m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l1981_198125


namespace NUMINAMATH_CALUDE_expand_expression_l1981_198118

theorem expand_expression (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4*x + 4) = x^4 + 4*x^3 - 16*x - 16 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1981_198118


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1981_198167

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1981_198167


namespace NUMINAMATH_CALUDE_function_analysis_l1981_198164

/-- Given a real number a and a function f(x) = x²(x-a), this theorem proves:
    (I) If f'(1) = 3, then a = 0 and the equation of the tangent line at (1, f(1)) is 3x - y - 2 = 0
    (II) The maximum value of f(x) in the interval [0, 2] is max{8 - 4a, 0} for a < 3 and 0 for a ≥ 3 -/
theorem function_analysis (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 * (x - a)) :
  (deriv f 1 = 3 → a = 0 ∧ ∀ x y, 3*x - y - 2 = 0 ↔ y = f x ∧ x = 1) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ max (8 - 4*a) 0 ∧ (a ≥ 3 → f x ≤ 0)) :=
sorry

end NUMINAMATH_CALUDE_function_analysis_l1981_198164


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1981_198161

theorem fraction_equality_implies_numerator_equality 
  {x y m : ℝ} (h1 : m ≠ 0) (h2 : x / m = y / m) : x = y :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1981_198161


namespace NUMINAMATH_CALUDE_oplus_example_l1981_198122

-- Define the ⊕ operation
def oplus (a b : ℕ) : ℕ := a + b + a * b

-- Statement to prove
theorem oplus_example : oplus (oplus 2 3) 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_oplus_example_l1981_198122


namespace NUMINAMATH_CALUDE_line_intercept_sum_l1981_198180

/-- A line passing through (5, 3) with slope 3 has x-intercept + y-intercept = -8 -/
theorem line_intercept_sum : ∀ (f : ℝ → ℝ),
  (f 5 = 3) →                        -- The line passes through (5, 3)
  (∀ x y, f y - f x = 3 * (y - x)) → -- The slope is 3
  (∃ a, f a = 0) →                   -- x-intercept exists
  (∃ b, f 0 = b) →                   -- y-intercept exists
  (∃ a b, f a = 0 ∧ f 0 = b ∧ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l1981_198180


namespace NUMINAMATH_CALUDE_min_phi_value_l1981_198142

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * (x + φ))

theorem min_phi_value (φ : ℝ) : 
  (φ > 0) →
  (∀ x, g x φ = g ((2 * π / 3) - x) φ) →
  (∀ ψ, ψ > 0 → (∀ x, g x ψ = g ((2 * π / 3) - x) ψ) → φ ≤ ψ) →
  φ = 5 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_min_phi_value_l1981_198142


namespace NUMINAMATH_CALUDE_distance_to_point_l1981_198160

/-- The distance from the origin to the point (12, 5) on the line y = 5/12 x is 13 -/
theorem distance_to_point : 
  let point : ℝ × ℝ := (12, 5)
  let line (x : ℝ) : ℝ := (5/12) * x
  (point.2 = line point.1) →
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_point_l1981_198160


namespace NUMINAMATH_CALUDE_range_of_a_l1981_198147

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ y : ℝ, y ≥ a ∧ |y - 1| ≥ 1) → 
  a ∈ Set.Iic 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1981_198147


namespace NUMINAMATH_CALUDE_specific_factory_production_l1981_198175

/-- A factory produces toys with the following parameters:
  * weekly_production: The total number of toys produced in a week
  * work_days: The number of days worked in a week
  * constant_daily_production: Whether the daily production is constant throughout the week
-/
structure ToyFactory where
  weekly_production : ℕ
  work_days : ℕ
  constant_daily_production : Prop

/-- Calculate the daily toy production for a given factory -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.work_days

/-- Theorem stating that for a factory producing 6500 toys per week,
    working 5 days a week, with constant daily production,
    the daily production is 1300 toys -/
theorem specific_factory_production :
  ∀ (factory : ToyFactory),
    factory.weekly_production = 6500 ∧
    factory.work_days = 5 ∧
    factory.constant_daily_production →
    daily_production factory = 1300 := by
  sorry

end NUMINAMATH_CALUDE_specific_factory_production_l1981_198175


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1981_198138

theorem sum_of_cubes (p q r : ℝ) 
  (sum_eq : p + q + r = 4)
  (sum_prod_eq : p * q + p * r + q * r = 3)
  (prod_eq : p * q * r = -6) :
  p^3 + q^3 + r^3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1981_198138


namespace NUMINAMATH_CALUDE_digit_3000_is_1_l1981_198131

/-- Represents the decimal expansion of integers from 1 to 1001 concatenated -/
def x : ℝ :=
  sorry

/-- Returns the nth digit after the decimal point in the given real number -/
def nthDigit (n : ℕ) (r : ℝ) : ℕ :=
  sorry

/-- The 3000th digit after the decimal point in x is 1 -/
theorem digit_3000_is_1 : nthDigit 3000 x = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_3000_is_1_l1981_198131


namespace NUMINAMATH_CALUDE_factor_expression_l1981_198191

theorem factor_expression (x : ℝ) : 9*x^2 + 3*x = 3*x*(3*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1981_198191


namespace NUMINAMATH_CALUDE_max_arrangements_at_six_min_winning_probability_at_six_l1981_198152

/-- The number of cities and days in the championship --/
def n : ℕ := 8

/-- Calculate the number of possible arrangements for k rounds --/
def arrangements (k : ℕ) : ℕ :=
  (Nat.factorial n * Nat.factorial n) / (Nat.factorial (n - k) * Nat.factorial (n - k) * Nat.factorial k)

/-- Theorem stating that 6 rounds maximizes the number of arrangements --/
theorem max_arrangements_at_six :
  ∀ k : ℕ, k ≤ n → arrangements 6 ≥ arrangements k :=
by sorry

/-- Corollary: The probability of winning the grand prize is minimized when there are 6 rounds --/
theorem min_winning_probability_at_six :
  ∀ k : ℕ, k ≤ n → (1 : ℚ) / arrangements 6 ≤ (1 : ℚ) / arrangements k :=
by sorry

end NUMINAMATH_CALUDE_max_arrangements_at_six_min_winning_probability_at_six_l1981_198152


namespace NUMINAMATH_CALUDE_sugar_amount_theorem_l1981_198189

def sugar_amount (sugar flour baking_soda chocolate_chips : ℚ) : Prop :=
  -- Ratio of sugar to flour is 5:4
  sugar / flour = 5 / 4 ∧
  -- Ratio of flour to baking soda is 10:1
  flour / baking_soda = 10 / 1 ∧
  -- Ratio of baking soda to chocolate chips is 3:2
  baking_soda / chocolate_chips = 3 / 2 ∧
  -- New ratio after adding 120 pounds of baking soda and 50 pounds of chocolate chips
  flour / (baking_soda + 120) = 16 / 3 ∧
  flour / (chocolate_chips + 50) = 16 / 2 ∧
  -- The amount of sugar is 1714 pounds
  sugar = 1714

theorem sugar_amount_theorem :
  ∃ sugar flour baking_soda chocolate_chips : ℚ,
    sugar_amount sugar flour baking_soda chocolate_chips :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_theorem_l1981_198189


namespace NUMINAMATH_CALUDE_inequality_condition_l1981_198128

theorem inequality_condition (a b : ℝ) : 
  (a > b → ((a + b) / 2)^2 > a * b) ∧ 
  (∃ a b : ℝ, ((a + b) / 2)^2 > a * b ∧ a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1981_198128


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l1981_198105

/-- The time taken for a monkey to climb a tree -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let effective_climb := hop_distance - slip_distance
  let full_climbs := (tree_height - 1) / effective_climb
  let remaining_distance := (tree_height - 1) % effective_climb
  full_climbs + if remaining_distance > 0 then 1 else 0

/-- Theorem: A monkey climbing a 17 ft tree, hopping 3 ft and slipping 2 ft each hour, takes 17 hours to reach the top -/
theorem monkey_climb_theorem :
  monkey_climb_time 17 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l1981_198105


namespace NUMINAMATH_CALUDE_angle_with_double_supplement_l1981_198194

theorem angle_with_double_supplement (α : ℝ) :
  (180 - α = 2 * α) → α = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_double_supplement_l1981_198194


namespace NUMINAMATH_CALUDE_symmetry_of_sum_and_product_l1981_198195

-- Define a property for function symmetry about a point
def SymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) + f (a - x) = 2 * b

-- Theorem statement
theorem symmetry_of_sum_and_product 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : SymmetricAbout f a b) (hg : SymmetricAbout g a b) :
  (SymmetricAbout (fun x ↦ f x + g x) a (2 * b)) ∧
  (∃ f g : ℝ → ℝ, SymmetricAbout f 0 0 ∧ SymmetricAbout g 0 0 ∧
    ¬∃ c d : ℝ, SymmetricAbout (fun x ↦ f x * g x) c d) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_sum_and_product_l1981_198195


namespace NUMINAMATH_CALUDE_equation_solution_l1981_198170

theorem equation_solution : 
  ∀ x : ℝ, (x + 4)^2 = 5*(x + 4) ↔ x = -4 ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1981_198170


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l1981_198181

theorem fraction_sum_proof (fractions : Finset ℚ) 
  (h1 : fractions.card = 9)
  (h2 : ∀ f ∈ fractions, ∃ n : ℕ+, f = 1 / n)
  (h3 : (fractions.sum id) = 1)
  (h4 : (1 / 3) ∈ fractions ∧ (1 / 7) ∈ fractions ∧ (1 / 9) ∈ fractions ∧ 
        (1 / 11) ∈ fractions ∧ (1 / 33) ∈ fractions)
  (h5 : ∃ f1 f2 f3 f4 : ℚ, f1 ∈ fractions ∧ f2 ∈ fractions ∧ f3 ∈ fractions ∧ f4 ∈ fractions ∧
        ∃ n1 n2 n3 n4 : ℕ, f1 = 1 / n1 ∧ f2 = 1 / n2 ∧ f3 = 1 / n3 ∧ f4 = 1 / n4 ∧
        n1 % 10 = 5 ∧ n2 % 10 = 5 ∧ n3 % 10 = 5 ∧ n4 % 10 = 5) :
  ∃ f1 f2 f3 f4 : ℚ, f1 ∈ fractions ∧ f2 ∈ fractions ∧ f3 ∈ fractions ∧ f4 ∈ fractions ∧
  f1 = 1 / 5 ∧ f2 = 1 / 15 ∧ f3 = 1 / 45 ∧ f4 = 1 / 385 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l1981_198181


namespace NUMINAMATH_CALUDE_solve_for_y_l1981_198101

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 10) (h2 : x = 2) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1981_198101


namespace NUMINAMATH_CALUDE_unique_valid_n_l1981_198132

def is_valid_number (n : ℕ) (x : ℕ) : Prop :=
  (x.digits 10).length = n ∧
  (x.digits 10).count 7 = 1 ∧
  (x.digits 10).count 1 = n - 1

def all_numbers_prime (n : ℕ) : Prop :=
  ∀ x : ℕ, is_valid_number n x → Nat.Prime x

theorem unique_valid_n : 
  ∀ n : ℕ, (n > 0 ∧ all_numbers_prime n) ↔ (n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_valid_n_l1981_198132


namespace NUMINAMATH_CALUDE_geometric_sum_10_terms_l1981_198196

theorem geometric_sum_10_terms : 
  let a : ℚ := 3/4
  let r : ℚ := 3/4
  let n : ℕ := 10
  let S : ℚ := (a * (1 - r^n)) / (1 - r)
  S = 2971581/1048576 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_10_terms_l1981_198196


namespace NUMINAMATH_CALUDE_smallest_irreducible_n_l1981_198177

def is_irreducible (n k : ℕ) : Prop :=
  Nat.gcd k (n + k + 2) = 1

def all_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 68 ≤ k → k ≤ 133 → is_irreducible n k

theorem smallest_irreducible_n :
  (all_irreducible 65 ∧
   all_irreducible 135 ∧
   (∀ n : ℕ, n < 65 → ¬all_irreducible n) ∧
   (∀ n : ℕ, 65 < n → n < 135 → ¬all_irreducible n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_irreducible_n_l1981_198177


namespace NUMINAMATH_CALUDE_proposition_implication_l1981_198199

theorem proposition_implication (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬¬p) : 
  ¬q := by
sorry

end NUMINAMATH_CALUDE_proposition_implication_l1981_198199


namespace NUMINAMATH_CALUDE_junior_trip_fraction_l1981_198183

theorem junior_trip_fraction (S J : ℚ) 
  (h1 : J = 2/3 * S) 
  (h2 : 2/3 * S + x * J = 1/2 * (S + J)) 
  (h3 : S > 0) 
  (h4 : J > 0) : 
  x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_junior_trip_fraction_l1981_198183


namespace NUMINAMATH_CALUDE_vehicles_with_high_speed_l1981_198112

theorem vehicles_with_high_speed (vehicles_80_to_89 vehicles_90_to_99 vehicles_100_to_109 : ℕ) :
  vehicles_80_to_89 = 15 →
  vehicles_90_to_99 = 30 →
  vehicles_100_to_109 = 5 →
  vehicles_80_to_89 + vehicles_90_to_99 + vehicles_100_to_109 = 50 :=
by sorry

end NUMINAMATH_CALUDE_vehicles_with_high_speed_l1981_198112


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1981_198193

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon : ℕ := 5

theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1981_198193


namespace NUMINAMATH_CALUDE_domain_range_equal_iff_l1981_198165

/-- The function f(x) = √(ax² + bx) where b > 0 -/
noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

/-- The domain of f -/
def domain (a b : ℝ) : Set ℝ :=
  if a > 0 then {x | x ≤ -b/a ∨ x ≥ 0}
  else if a < 0 then {x | 0 ≤ x ∧ x ≤ -b/a}
  else {x | x ≥ 0}

/-- The range of f -/
def range (a b : ℝ) : Set ℝ :=
  if a ≥ 0 then {y | y ≥ 0}
  else {y | 0 ≤ y ∧ y ≤ b / (2 * Real.sqrt (-a))}

theorem domain_range_equal_iff (b : ℝ) (hb : b > 0) :
  ∀ a : ℝ, domain a b = range a b ↔ a = -4 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_domain_range_equal_iff_l1981_198165


namespace NUMINAMATH_CALUDE_one_fourth_in_five_eighths_l1981_198129

theorem one_fourth_in_five_eighths : (5 / 8 : ℚ) / (1 / 4 : ℚ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_in_five_eighths_l1981_198129


namespace NUMINAMATH_CALUDE_winnie_fell_behind_l1981_198137

/-- The number of repetitions Winnie fell behind --/
def repetitions_fell_behind (yesterday_reps today_reps : ℕ) : ℕ :=
  yesterday_reps - today_reps

/-- Proof that Winnie fell behind by 13 repetitions --/
theorem winnie_fell_behind :
  repetitions_fell_behind 86 73 = 13 := by
  sorry

end NUMINAMATH_CALUDE_winnie_fell_behind_l1981_198137


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l1981_198110

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l1981_198110


namespace NUMINAMATH_CALUDE_log_expression_equality_l1981_198114

-- Define lg as base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_expression_equality :
  (Real.log (Real.sqrt 27) / Real.log 3) + lg 25 + lg 4 + 7^(Real.log 2 / Real.log 7) + (-9.8)^0 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1981_198114


namespace NUMINAMATH_CALUDE_triangle_formation_conditions_l1981_198115

theorem triangle_formation_conditions 
  (E F G H : ℝ × ℝ)  -- Points in 2D plane
  (a b c : ℝ)        -- Lengths
  (θ φ : ℝ)          -- Angles
  (h_distinct : E ≠ F ∧ F ≠ G ∧ G ≠ H)  -- Distinct points
  (h_collinear : ∃ (m k : ℝ), F.2 = m * F.1 + k ∧ G.2 = m * G.1 + k ∧ H.2 = m * H.1 + k)  -- Collinearity
  (h_order : E.1 < F.1 ∧ F.1 < G.1 ∧ G.1 < H.1)  -- Order on line
  (h_lengths : dist E F = a ∧ dist E G = b ∧ dist E H = c)  -- Segment lengths
  (h_rotation : ∃ (E' : ℝ × ℝ), 
    dist F E' = a ∧ 
    dist G H = c - b ∧
    E' = H)  -- Rotation result
  (h_triangle : ∃ (F' G' : ℝ × ℝ), 
    dist E' F' = a ∧ 
    dist F' G' > 0 ∧ 
    dist G' E' = c - b ∧
    (F'.1 - E'.1) * (G'.2 - E'.2) ≠ (G'.1 - E'.1) * (F'.2 - E'.2))  -- Non-degenerate triangle formed
  : a < c / 2 ∧ b < a + c * Real.cos φ ∧ b * Real.cos θ < c / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_conditions_l1981_198115


namespace NUMINAMATH_CALUDE_circle_equation_part1_circle_equation_part2_l1981_198158

-- Part 1
theorem circle_equation_part1 (A B : ℝ × ℝ) (center_line : ℝ → ℝ) :
  A = (5, 2) →
  B = (3, 2) →
  (∀ x y, center_line x = 2*x - y - 3) →
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 2)) ∧
    center_line h = k) ∧
  h = 4 ∧ k = 5 ∧ r^2 = 10 :=
sorry

-- Part 2
theorem circle_equation_part2 (A : ℝ × ℝ) (sym_line chord_line : ℝ → ℝ) :
  A = (2, 3) →
  (∀ x y, sym_line x = -x - 2*y) →
  (∀ x y, chord_line x = x - y + 1) →
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x = 2 ∧ y = 3) ∨ 
     (∃ x' y', sym_line x' = y' ∧ (x' - h)^2 + (y' - k)^2 = r^2)) ∧
    (∃ x1 y1 x2 y2, chord_line x1 = y1 ∧ chord_line x2 = y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 8)) ∧
  ((h = 6 ∧ k = -3 ∧ r^2 = 52) ∨ (h = 14 ∧ k = -7 ∧ r^2 = 244)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_part1_circle_equation_part2_l1981_198158


namespace NUMINAMATH_CALUDE_number_of_teams_is_twelve_l1981_198106

/-- The number of teams in the baseball league --/
def n : ℕ := sorry

/-- The number of games each team plays against every other team --/
def games_per_pair : ℕ := 6

/-- The total number of games played in the league --/
def total_games : ℕ := 396

/-- Theorem stating that the number of teams in the league is 12 --/
theorem number_of_teams_is_twelve :
  (n * (n - 1) / 2) * games_per_pair = total_games ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_number_of_teams_is_twelve_l1981_198106


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l1981_198133

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Statement 1
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Statement 2
theorem A_subset_C_implies_a_geq_7 (a : ℝ) (h : A ⊆ C a) : a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l1981_198133


namespace NUMINAMATH_CALUDE_minimum_ladder_rungs_l1981_198111

theorem minimum_ladder_rungs (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let n := a + b - Nat.gcd a b
  ∀ m : ℕ, m < n → ¬ (∃ (x y : ℤ), x ≥ 0 ∧ y ≥ 0 ∧ a * x - b * y = m) ∧
  ∃ (x y : ℤ), x ≥ 0 ∧ y ≥ 0 ∧ a * x - b * y = n :=
by sorry

end NUMINAMATH_CALUDE_minimum_ladder_rungs_l1981_198111


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1981_198151

theorem complex_equation_solution :
  ∃ (x : ℂ), (5 : ℂ) + 2 * Complex.I * x = (3 : ℂ) - 4 * Complex.I * x ∧ x = Complex.I / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1981_198151


namespace NUMINAMATH_CALUDE_job_completion_time_l1981_198186

/-- Given workers A, B, and C who can complete a job individually in 18, 30, and 45 days respectively,
    prove that they can complete the job together in 9 days. -/
theorem job_completion_time (a b c : ℝ) (ha : a = 18) (hb : b = 30) (hc : c = 45) :
  (1 / a + 1 / b + 1 / c)⁻¹ = 9 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1981_198186


namespace NUMINAMATH_CALUDE_lemon_juice_for_dozen_cupcakes_l1981_198184

/-- The number of tablespoons of lemon juice provided by one lemon -/
def tablespoons_per_lemon : ℕ := 4

/-- The number of lemons needed for 3 dozen cupcakes -/
def lemons_for_three_dozen : ℕ := 9

/-- The number of tablespoons of lemon juice needed for a dozen cupcakes -/
def tablespoons_for_dozen : ℕ := 12

/-- Proves that the number of tablespoons of lemon juice needed for a dozen cupcakes is 12 -/
theorem lemon_juice_for_dozen_cupcakes : 
  tablespoons_for_dozen = (lemons_for_three_dozen * tablespoons_per_lemon) / 3 :=
by sorry

end NUMINAMATH_CALUDE_lemon_juice_for_dozen_cupcakes_l1981_198184


namespace NUMINAMATH_CALUDE_mike_afternoon_seeds_l1981_198136

/-- Represents the number of tomato seeds planted by Mike and Ted -/
structure TomatoSeeds where
  mike_morning : ℕ
  ted_morning : ℕ
  mike_afternoon : ℕ
  ted_afternoon : ℕ

/-- The conditions of the tomato planting problem -/
def tomato_planting_conditions (s : TomatoSeeds) : Prop :=
  s.mike_morning = 50 ∧
  s.ted_morning = 2 * s.mike_morning ∧
  s.ted_afternoon = s.mike_afternoon - 20 ∧
  s.mike_morning + s.ted_morning + s.mike_afternoon + s.ted_afternoon = 250

/-- Theorem stating that under the given conditions, Mike planted 60 tomato seeds in the afternoon -/
theorem mike_afternoon_seeds (s : TomatoSeeds) 
  (h : tomato_planting_conditions s) : s.mike_afternoon = 60 := by
  sorry

end NUMINAMATH_CALUDE_mike_afternoon_seeds_l1981_198136


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1981_198174

theorem intersection_implies_a_value (A B : Set ℝ) (a : ℝ) :
  A = {-1, 1, 3} →
  B = {a + 2, a^2 + 4} →
  A ∩ B = {3} →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1981_198174


namespace NUMINAMATH_CALUDE_fourth_graders_pizza_problem_l1981_198156

theorem fourth_graders_pizza_problem :
  ∀ (n : ℕ),
  (∀ (student : ℕ), student ≤ n → 20 * 6 * student = 1200) →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_graders_pizza_problem_l1981_198156


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l1981_198113

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l1981_198113


namespace NUMINAMATH_CALUDE_rhombus_region_area_l1981_198121

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem rhombus_region_area (r : Rhombus) 
  (h1 : r.side_length = 3)
  (h2 : r.angle_B = π / 2) :
  area (region_R r) = 9 * π / 16 :=
sorry

end NUMINAMATH_CALUDE_rhombus_region_area_l1981_198121


namespace NUMINAMATH_CALUDE_all_expressions_zero_l1981_198149

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define points in the vector space
variable (A B C D O N M P Q : V)

-- Define the expressions
def expr1 (A B C : V) : V := (B - A) + (C - B) + (A - C)
def expr2 (A B C D : V) : V := (B - A) - (C - A) + (D - B) - (D - C)
def expr3 (O A D : V) : V := (A - O) - (D - O) + (D - A)
def expr4 (N Q P M : V) : V := (Q - N) + (P - Q) + (N - M) - (P - M)

-- Theorem stating that all expressions result in the zero vector
theorem all_expressions_zero (A B C D O N M P Q : V) : 
  expr1 A B C = 0 ∧ 
  expr2 A B C D = 0 ∧ 
  expr3 O A D = 0 ∧ 
  expr4 N Q P M = 0 :=
sorry

end NUMINAMATH_CALUDE_all_expressions_zero_l1981_198149


namespace NUMINAMATH_CALUDE_intersection_k_range_l1981_198119

-- Define the lines l₁ and l₂
def l₁ (x y k : ℝ) : Prop := y = 2 * x - 5 * k + 7
def l₂ (x y : ℝ) : Prop := y = -1/2 * x + 2

-- Define the intersection point
def intersection (x y k : ℝ) : Prop := l₁ x y k ∧ l₂ x y

-- Define the first quadrant condition
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem intersection_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, intersection x y k ∧ first_quadrant x y) ↔ (1 < k ∧ k < 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_k_range_l1981_198119


namespace NUMINAMATH_CALUDE_wall_width_l1981_198198

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 4 * w →
  l = 3 * h →
  volume = w * h * l →
  volume = 10368 →
  w = 6 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l1981_198198


namespace NUMINAMATH_CALUDE_license_plate_count_l1981_198190

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 14

/-- The length of the license plate -/
def plate_length : ℕ := 6

/-- The number of possible first letters (B or C) -/
def first_letter_choices : ℕ := 2

/-- The number of possible last letters (N) -/
def last_letter_choices : ℕ := 1

/-- The number of letters that cannot be used in the middle (B, C, M, N) -/
def excluded_middle_letters : ℕ := 4

theorem license_plate_count :
  (first_letter_choices * (alphabet_size - excluded_middle_letters) *
   (alphabet_size - excluded_middle_letters - 1) *
   (alphabet_size - excluded_middle_letters - 2) *
   (alphabet_size - excluded_middle_letters - 3) *
   last_letter_choices) = 15840 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1981_198190


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1981_198176

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + y * z = 30) (h2 : y * z + z * x = 36) (h3 : z * x + x * y = 42) :
  x + y + z = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1981_198176


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1981_198109

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^3 + 3 * x₁^2 - 11 * x₁ + 6 = 0) →
  (2 * x₂^3 + 3 * x₂^2 - 11 * x₂ + 6 = 0) →
  (2 * x₃^3 + 3 * x₃^2 - 11 * x₃ + 6 = 0) →
  (x₁ + x₂ + x₃ = -3/2) →
  (x₁*x₂ + x₂*x₃ + x₃*x₁ = -11/2) →
  (x₁*x₂*x₃ = -3) →
  x₁^3 + x₂^3 + x₃^3 = -99/8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1981_198109


namespace NUMINAMATH_CALUDE_incident_ray_equation_l1981_198141

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (5, 7)

-- Define the reflection of B across the x-axis
def B_reflected : ℝ × ℝ := (B.1, -B.2)

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  10 * x + 7 * y - 1 = 0

-- Theorem statement
theorem incident_ray_equation :
  line_equation A.1 A.2 ∧
  line_equation B_reflected.1 B_reflected.2 :=
sorry

end NUMINAMATH_CALUDE_incident_ray_equation_l1981_198141


namespace NUMINAMATH_CALUDE_smallest_base_sum_l1981_198185

theorem smallest_base_sum : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 1 ∧ 
  b > 1 ∧
  5 * a + 2 = 2 * b + 5 ∧ 
  (∀ (a' b' : ℕ), a' ≠ b' → a' > 1 → b' > 1 → 5 * a' + 2 = 2 * b' + 5 → a + b ≤ a' + b') ∧
  a + b = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_sum_l1981_198185


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1981_198163

/-- A parabola y = ax² + bx + 7 is tangent to the line y = 2x + 3 if and only if b = 2 ± 4√a -/
theorem parabola_tangent_to_line (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x^2 + b * x + 7 ∧ y = 2 * x + 3 ∧
    ∀ x' : ℝ, a * x'^2 + b * x' + 7 ≥ 2 * x' + 3) ↔
  (b = 2 + 4 * Real.sqrt a ∨ b = 2 - 4 * Real.sqrt a) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1981_198163


namespace NUMINAMATH_CALUDE_two_lines_intersecting_at_distance_5_l1981_198179

/-- Given a line and a point, find two lines passing through the point and intersecting the given line at a distance of 5 from the given point. -/
theorem two_lines_intersecting_at_distance_5 :
  ∃ (l₂₁ l₂₂ : ℝ → ℝ → Prop),
    (∀ x y, l₂₁ x y ↔ x = 1) ∧
    (∀ x y, l₂₂ x y ↔ 3 * x + 4 * y + 1 = 0) ∧
    (∀ x y, l₂₁ x y → l₂₁ 1 (-1)) ∧
    (∀ x y, l₂₂ x y → l₂₂ 1 (-1)) ∧
    (∃ x₁ y₁, l₂₁ x₁ y₁ ∧ 2 * x₁ + y₁ - 6 = 0 ∧ (x₁ - 1)^2 + (y₁ + 1)^2 = 5^2) ∧
    (∃ x₂ y₂, l₂₂ x₂ y₂ ∧ 2 * x₂ + y₂ - 6 = 0 ∧ (x₂ - 1)^2 + (y₂ + 1)^2 = 5^2) :=
by sorry


end NUMINAMATH_CALUDE_two_lines_intersecting_at_distance_5_l1981_198179


namespace NUMINAMATH_CALUDE_french_toast_slices_per_loaf_l1981_198192

/-- The number of slices in each loaf of bread for Suzanne's french toast -/
def slices_per_loaf (days_per_week : ℕ) (slices_per_day : ℕ) (weeks : ℕ) (total_loaves : ℕ) : ℕ :=
  (days_per_week * slices_per_day * weeks) / total_loaves

/-- Proof that the number of slices in each loaf is 6 -/
theorem french_toast_slices_per_loaf :
  slices_per_loaf 2 3 52 26 = 6 := by
  sorry

end NUMINAMATH_CALUDE_french_toast_slices_per_loaf_l1981_198192


namespace NUMINAMATH_CALUDE_problem_solution_l1981_198123

theorem problem_solution (x : ℝ) : (0.50 * x = 0.05 * 500 - 20) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1981_198123


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1981_198178

def team_size : ℕ := 12
def quadruplets_size : ℕ := 4
def starters_size : ℕ := 5
def max_quadruplets_in_lineup : ℕ := 2

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem basketball_lineup_combinations :
  (choose (team_size - quadruplets_size) starters_size) +
  (choose quadruplets_size 1 * choose (team_size - quadruplets_size) (starters_size - 1)) +
  (choose quadruplets_size 2 * choose (team_size - quadruplets_size) (starters_size - 2)) = 672 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1981_198178


namespace NUMINAMATH_CALUDE_seventh_power_sum_l1981_198168

theorem seventh_power_sum (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  α^7 + β^7 + γ^7 = 65.38 := by
  sorry

end NUMINAMATH_CALUDE_seventh_power_sum_l1981_198168


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1981_198162

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x ∈ Set.Icc (-1) 1, f (-x) + f x = 0)
  (h2 : ∀ m n, m ∈ Set.Icc 0 1 → n ∈ Set.Icc 0 1 → m ≠ n → (f m - f n) / (m - n) < 0) :
  {x : ℝ | f (1 - 3*x) ≤ f (x - 1)} = Set.Icc 0 (1/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1981_198162


namespace NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l1981_198144

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line (a^2 - 2a)x + y = 0 -/
def slope1 (a : ℝ) : ℝ := -(a^2 - 2*a)

/-- The slope of the line 3x + y + 1 = 0 -/
def slope2 : ℝ := -3

/-- The lines (a^2 - 2a)x + y = 0 and 3x + y + 1 = 0 are parallel -/
def lines_are_parallel (a : ℝ) : Prop := are_parallel (slope1 a) slope2

theorem a_equals_3_necessary_not_sufficient :
  (∀ a : ℝ, lines_are_parallel a → a = 3) ∧
  ¬(∀ a : ℝ, a = 3 → lines_are_parallel a) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l1981_198144


namespace NUMINAMATH_CALUDE_complex_power_difference_l1981_198143

theorem complex_power_difference (i : ℂ) : i^2 = -1 → i^123 - i^45 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1981_198143


namespace NUMINAMATH_CALUDE_x_axis_conditions_l1981_198169

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a line is the x-axis -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- Theorem stating the conditions for a line to be the x-axis -/
theorem x_axis_conditions (l : Line) : 
  is_x_axis l ↔ l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 := by sorry

end NUMINAMATH_CALUDE_x_axis_conditions_l1981_198169


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt10_implies_a_plusminus2_l1981_198100

theorem complex_modulus_sqrt10_implies_a_plusminus2 (a : ℝ) : 
  Complex.abs ((a + Complex.I) * (1 - Complex.I)) = Real.sqrt 10 → 
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt10_implies_a_plusminus2_l1981_198100


namespace NUMINAMATH_CALUDE_wheel_revolutions_l1981_198124

/-- The number of revolutions of a wheel with diameter 10 feet to travel half a mile -/
theorem wheel_revolutions (π : ℝ) (h : π > 0) : 
  let diameter : ℝ := 10
  let circumference : ℝ := π * diameter
  let half_mile_in_feet : ℝ := 5280 / 2
  half_mile_in_feet / circumference = 264 / π := by
  sorry

end NUMINAMATH_CALUDE_wheel_revolutions_l1981_198124


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1981_198145

theorem sum_of_squares_and_products (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 52)
  (sum_of_products : a*b + b*c + c*a = 24) : 
  a + b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1981_198145


namespace NUMINAMATH_CALUDE_ln_101_100_gt_2_201_l1981_198150

theorem ln_101_100_gt_2_201 : Real.log (101/100) > 2/201 := by
  sorry

end NUMINAMATH_CALUDE_ln_101_100_gt_2_201_l1981_198150


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1981_198166

def M : ℕ := 36 * 36 * 85 * 128

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 4094 = sum_even_divisors M :=
sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1981_198166


namespace NUMINAMATH_CALUDE_train_passing_time_l1981_198173

/-- The time it takes for two trains to pass each other -/
theorem train_passing_time (v1 l1 v2 l2 : ℝ) : 
  v1 > 0 → l1 > 0 → v2 > 0 → l2 > 0 →
  (l1 / v1 = 5) →
  (v1 = 2 * v2) →
  (l1 = 3 * l2) →
  (l1 + l2) / (v1 + v2) = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1981_198173


namespace NUMINAMATH_CALUDE_union_subset_iff_m_range_no_m_for_equality_l1981_198187

-- Define the sets P and S
def P : Set ℝ := {x : ℝ | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ m}

-- Theorem 1: (P ∪ S) ⊆ P if and only if m ∈ (-∞, 3]
theorem union_subset_iff_m_range (m : ℝ) : 
  (P ∪ S m) ⊆ P ↔ m ≤ 3 :=
sorry

-- Theorem 2: There does not exist an m such that P = S
theorem no_m_for_equality : 
  ¬∃ m : ℝ, P = S m :=
sorry

end NUMINAMATH_CALUDE_union_subset_iff_m_range_no_m_for_equality_l1981_198187


namespace NUMINAMATH_CALUDE_circle_equation_l1981_198127

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x + 3)^2 + (y + 3)^2 = 18

-- Define the point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the circle we want to prove
def target_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_equation :
  (∀ x y : ℝ, target_circle x y ↔ 
    (((x, y) = point_A ∨ (x, y) = origin) ∧ 
     (∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
       ((x' - x)^2 + (y' - y)^2 < δ^2 → 
        (target_circle x' y' ↔ ¬given_circle x' y'))))) := 
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1981_198127


namespace NUMINAMATH_CALUDE_cost_price_percentage_l1981_198134

/-- Proves that given a discount of 12% and a gain percent of 37.5%, 
    the cost price is approximately 64% of the marked price. -/
theorem cost_price_percentage (marked_price : ℝ) (cost_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : (marked_price - cost_price) / cost_price = 0.375) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |cost_price / marked_price - 0.64| < ε :=
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l1981_198134


namespace NUMINAMATH_CALUDE_birthday_check_value_l1981_198102

theorem birthday_check_value (initial_balance : ℝ) (check_value : ℝ) : 
  initial_balance = 150 →
  check_value = (1/4) * (initial_balance + check_value) →
  check_value = 50 := by
sorry

end NUMINAMATH_CALUDE_birthday_check_value_l1981_198102


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1981_198148

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 3*x = 2) : 3*x^2 - 9*x - 7 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1981_198148


namespace NUMINAMATH_CALUDE_intersection_and_subset_l1981_198108

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m)*(x - m - 9) < 0}

theorem intersection_and_subset :
  (A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6}) ∧
  (∀ m : ℝ, A ⊆ C m ↔ -3 ≤ m ∧ m ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_subset_l1981_198108


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1981_198153

/-- Given a journey with the following parameters:
  * total_distance: The total distance of the journey in km
  * total_time: The total time of the journey in hours
  * first_half_speed: The speed for the first half of the journey in km/hr
  
  This theorem proves that the speed for the second half of the journey is equal to the second_half_speed. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ)
  (h1 : total_distance = 672)
  (h2 : total_time = 30)
  (h3 : first_half_speed = 21)
  : ∃ second_half_speed : ℝ, second_half_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1981_198153


namespace NUMINAMATH_CALUDE_yolandas_walking_rate_l1981_198155

/-- Proves that Yolanda's walking rate is 3 miles per hour given the problem conditions -/
theorem yolandas_walking_rate 
  (total_distance : ℝ) 
  (bobs_delay : ℝ) 
  (bobs_rate : ℝ) 
  (bobs_distance : ℝ) 
  (h1 : total_distance = 24)
  (h2 : bobs_delay = 1)
  (h3 : bobs_rate = 4)
  (h4 : bobs_distance = 12) : 
  (total_distance - bobs_distance) / (bobs_distance / bobs_rate + bobs_delay) = 3 := by
  sorry

#check yolandas_walking_rate

end NUMINAMATH_CALUDE_yolandas_walking_rate_l1981_198155


namespace NUMINAMATH_CALUDE_alpha_value_l1981_198116

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 2*β)).re > 0)
  (h3 : β = 3 + 2*Complex.I) :
  α = 6 - 2*Complex.I := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l1981_198116


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1981_198140

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), 
    (x > 0) ∧ 
    (x < 1) ∧ 
    (initial_price * (1 - x)^2 = final_price) ∧
    (x = 1 - (4/5)) := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1981_198140


namespace NUMINAMATH_CALUDE_bridget_profit_is_42_l1981_198139

/-- Calculates Bridget's profit from bread sales --/
def bridget_profit (total_loaves : ℕ) (morning_price afternoon_price late_afternoon_price cost_per_loaf : ℚ) : ℚ :=
  let morning_sold := total_loaves / 3
  let morning_revenue := morning_sold * morning_price
  
  let afternoon_remaining := total_loaves - morning_sold
  let afternoon_sold := afternoon_remaining / 2
  let afternoon_revenue := afternoon_sold * afternoon_price
  
  let late_afternoon_remaining := afternoon_remaining - afternoon_sold
  let late_afternoon_sold := late_afternoon_remaining / 4
  let late_afternoon_revenue := late_afternoon_sold * late_afternoon_price
  
  let evening_remaining := late_afternoon_remaining - late_afternoon_sold
  let evening_price := late_afternoon_price / 2
  let evening_revenue := evening_remaining * evening_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue + evening_revenue
  let total_cost := total_loaves * cost_per_loaf
  
  total_revenue - total_cost

/-- Theorem stating Bridget's profit is $42 --/
theorem bridget_profit_is_42 :
  bridget_profit 60 3 (3/2) 1 1 = 42 := by
  sorry


end NUMINAMATH_CALUDE_bridget_profit_is_42_l1981_198139


namespace NUMINAMATH_CALUDE_rotated_line_equation_l1981_198197

/-- Given a line with equation x - y + 1 = 0 and a point P(3, 4) on this line,
    rotating the line 90° counterclockwise around P results in a line with equation x + y - 7 = 0 -/
theorem rotated_line_equation (x y : ℝ) : 
  (x - y + 1 = 0 ∧ 3 - 4 + 1 = 0) → 
  (∃ (m : ℝ), m * (x - 3) + (y - 4) = 0 ∧ m = 1) →
  x + y - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l1981_198197


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1981_198107

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1981_198107
