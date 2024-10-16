import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2893_289383

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 : ℝ) / (b - c) > (1 : ℝ) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2893_289383


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2893_289349

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line: x + my + 6 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

/-- The second line: 3x + (m - 2)y + 2m = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + (m - 2) * y + 2 * m = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, (are_parallel 1 m 3 (m - 2)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2893_289349


namespace NUMINAMATH_CALUDE_people_after_yoongi_l2893_289364

theorem people_after_yoongi (total : ℕ) (before : ℕ) (h1 : total = 20) (h2 : before = 11) :
  total - (before + 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_people_after_yoongi_l2893_289364


namespace NUMINAMATH_CALUDE_petes_diner_cost_theorem_l2893_289375

/-- Represents the cost calculation at Pete's Diner -/
def PetesDinerCost (burgerPrice juicePrice discountAmount : ℕ) 
                   (discountThreshold : ℕ) 
                   (burgerCount juiceCount : ℕ) : ℕ :=
  let totalItems := burgerCount + juiceCount
  let subtotal := burgerCount * burgerPrice + juiceCount * juicePrice
  if totalItems > discountThreshold then subtotal - discountAmount else subtotal

/-- Proves that the total cost of 7 burgers and 5 juices at Pete's Diner is 38 dollars -/
theorem petes_diner_cost_theorem : 
  PetesDinerCost 4 3 5 10 7 5 = 38 := by
  sorry

#eval PetesDinerCost 4 3 5 10 7 5

end NUMINAMATH_CALUDE_petes_diner_cost_theorem_l2893_289375


namespace NUMINAMATH_CALUDE_three_number_sum_l2893_289363

theorem three_number_sum (a b c : ℝ) : 
  a + b = 35 ∧ b + c = 54 ∧ c + a = 58 → a + b + c = 73.5 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l2893_289363


namespace NUMINAMATH_CALUDE_solution_set_f_geq_zero_range_m_three_zero_points_l2893_289321

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 2| - |x - 2| + m

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := f m x - x

-- Theorem 1: Solution set of f(x) ≥ 0 when m = 1
theorem solution_set_f_geq_zero (x : ℝ) :
  f 1 x ≥ 0 ↔ x ≥ -1/2 :=
sorry

-- Theorem 2: Range of m when g(x) has three zero points
theorem range_m_three_zero_points :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m x = 0 ∧ g m y = 0 ∧ g m z = 0) →
  -2 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_zero_range_m_three_zero_points_l2893_289321


namespace NUMINAMATH_CALUDE_student_selection_problem_l2893_289304

theorem student_selection_problem (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) (n_competitions : ℕ) :
  n_male = 5 →
  n_female = 4 →
  n_select = 3 →
  n_competitions = 2 →
  (Nat.choose (n_male + n_female) n_select - Nat.choose n_male n_select - Nat.choose n_female n_select) *
  (Nat.factorial n_select) = 420 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_problem_l2893_289304


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equal_area_l2893_289393

theorem rectangle_perimeter_equal_area (x y : ℕ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  2 * (x + y) = x * y →
  ((x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equal_area_l2893_289393


namespace NUMINAMATH_CALUDE_max_quarters_is_twelve_l2893_289336

/-- Represents the number of each type of coin -/
structure CoinCount where
  count : ℕ

/-- Represents the total value of coins in cents -/
def total_value (c : CoinCount) : ℕ :=
  25 * c.count + 5 * c.count + 10 * c.count

/-- The maximum number of quarters possible given the conditions -/
def max_quarters : Prop :=
  ∃ (c : CoinCount), total_value c = 480 ∧ 
    ∀ (c' : CoinCount), total_value c' = 480 → c'.count ≤ c.count

theorem max_quarters_is_twelve : 
  max_quarters ∧ ∃ (c : CoinCount), c.count = 12 ∧ total_value c = 480 :=
sorry


end NUMINAMATH_CALUDE_max_quarters_is_twelve_l2893_289336


namespace NUMINAMATH_CALUDE_set_operations_l2893_289319

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {1, 2, 5}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l2893_289319


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l2893_289376

theorem inequality_holds_for_all_reals (x : ℝ) : 4 * x / (x^2 + 4) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l2893_289376


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2893_289328

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a₂ + a₇ + a₁₅ = 12, prove that a₈ = 4 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 7 + a 15 = 12) : 
  a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2893_289328


namespace NUMINAMATH_CALUDE_triangle_sum_theorem_l2893_289371

theorem triangle_sum_theorem (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : c^2 = a^2 + b^2 - a*b) :
  a / (b + c) + b / (c + a) = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_theorem_l2893_289371


namespace NUMINAMATH_CALUDE_vessel_capacity_l2893_289381

/-- The capacity of the vessel in litres -/
def C : ℝ := 60.01

/-- The amount of liquid removed and replaced with water each time, in litres -/
def removed : ℝ := 9

/-- The amount of pure milk in the final solution, in litres -/
def final_milk : ℝ := 43.35

/-- Theorem stating that the capacity of the vessel is 60.01 litres -/
theorem vessel_capacity :
  (C - removed) * (C - removed) / C = final_milk :=
sorry

end NUMINAMATH_CALUDE_vessel_capacity_l2893_289381


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l2893_289340

theorem rectangle_area_reduction (initial_length initial_width : ℝ)
  (reduced_length reduced_width : ℝ) :
  initial_length = 5 →
  initial_width = 7 →
  reduced_length = initial_length - 2 →
  reduced_width = initial_width - 1 →
  reduced_length * initial_width = 21 →
  reduced_length * reduced_width = 18 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l2893_289340


namespace NUMINAMATH_CALUDE_patients_per_doctor_l2893_289339

/-- Given a hospital with 400 patients and 16 doctors, prove that each doctor takes care of 25 patients. -/
theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) :
  total_patients = 400 → total_doctors = 16 →
  total_patients / total_doctors = 25 := by
  sorry

end NUMINAMATH_CALUDE_patients_per_doctor_l2893_289339


namespace NUMINAMATH_CALUDE_tan_pi_fourth_plus_alpha_l2893_289309

theorem tan_pi_fourth_plus_alpha (α : Real) (h : Real.tan α = 2) : 
  Real.tan (π/4 + α) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_fourth_plus_alpha_l2893_289309


namespace NUMINAMATH_CALUDE_odd_function_period_range_l2893_289366

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, HasPeriod f q → q > 0 → p ≤ q

theorem odd_function_period_range (f : ℝ → ℝ) (m : ℝ) :
  IsOdd f →
  SmallestPositivePeriod f 3 →
  f 2015 > 1 →
  f 1 = (2 * m + 3) / (m - 1) →
  -2/3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_period_range_l2893_289366


namespace NUMINAMATH_CALUDE_polynomial_coefficients_sum_l2893_289307

theorem polynomial_coefficients_sum 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h : ∀ x, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) : 
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1) ∧ (a₀ + a₂ + a₄ + a₆ = 365) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_sum_l2893_289307


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2893_289395

theorem interest_rate_calculation (total_sum second_part : ℝ)
  (h1 : total_sum = 2691)
  (h2 : second_part = 1656)
  (h3 : total_sum > second_part) :
  let first_part := total_sum - second_part
  let interest_rate_first := 0.03
  let time_first := 8
  let time_second := 3
  let interest_rate_second := (first_part * interest_rate_first * time_first) / (second_part * time_second)
  ∃ ε > 0, |interest_rate_second - 0.05| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2893_289395


namespace NUMINAMATH_CALUDE_order_of_differences_l2893_289325

theorem order_of_differences (a b c : ℝ) : 
  a = Real.sqrt 3 - Real.sqrt 2 →
  b = Real.sqrt 6 - Real.sqrt 5 →
  c = Real.sqrt 7 - Real.sqrt 6 →
  a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_order_of_differences_l2893_289325


namespace NUMINAMATH_CALUDE_work_completion_time_l2893_289354

-- Define the work rates and time worked by Y
def x_rate : ℚ := 1 / 24
def y_rate : ℚ := 1 / 16
def y_days_worked : ℕ := 10

-- Define the theorem
theorem work_completion_time :
  let total_work : ℚ := 1
  let work_done_by_y : ℚ := y_rate * y_days_worked
  let remaining_work : ℚ := total_work - work_done_by_y
  let days_needed_by_x : ℚ := remaining_work / x_rate
  days_needed_by_x = 9 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2893_289354


namespace NUMINAMATH_CALUDE_carol_fraction_l2893_289386

/-- Represents the money each person has -/
structure Money where
  alice : ℚ
  bob : ℚ
  carol : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.carol = 0 ∧ 
  m.alice > 0 ∧ 
  m.bob > 0 ∧ 
  m.alice / 6 = m.bob / 3 ∧
  m.alice / 6 > 0

/-- The final state after Alice and Bob give money to Carol -/
def final_state (m : Money) : Money :=
  { alice := m.alice * (5/6),
    bob := m.bob * (2/3),
    carol := m.alice / 6 + m.bob / 3 }

/-- The theorem to be proved -/
theorem carol_fraction (m : Money) 
  (h : problem_conditions m) : 
  (final_state m).carol / ((final_state m).alice + (final_state m).bob + (final_state m).carol) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_carol_fraction_l2893_289386


namespace NUMINAMATH_CALUDE_no_solution_l2893_289388

/-- Sequence definition -/
def u : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 6 * u (n + 1) + 7 * u n

/-- Main theorem -/
theorem no_solution :
  ¬ ∃ (a b c n : ℕ), a * b * (a + b) * (a^2 + a*b + b^2) = c^2022 + 42 ∧ c^2022 + 42 = u n :=
by sorry

end NUMINAMATH_CALUDE_no_solution_l2893_289388


namespace NUMINAMATH_CALUDE_missing_number_proof_l2893_289348

theorem missing_number_proof (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) 
  (h3 : a^3 = x * 25 * 15 * b) : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2893_289348


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2893_289342

theorem inequality_equivalence (x : ℝ) :
  (2 * (5 ^ (2 * x)) * Real.sin (2 * x) - 3 ^ x ≥ 5 ^ (2 * x) - 2 * (3 ^ x) * Real.sin (2 * x)) ↔
  (∃ k : ℤ, π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2893_289342


namespace NUMINAMATH_CALUDE_fundamental_inequality_variant_l2893_289372

theorem fundamental_inequality_variant (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  1 / (2 * a) + 1 / b ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_fundamental_inequality_variant_l2893_289372


namespace NUMINAMATH_CALUDE_quadratic_theorem_l2893_289310

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The maximum value of f occurs at x = 2 -/
def has_max_at_2 (a b c : ℝ) : Prop :=
  ∀ x, f a b c x ≤ f a b c 2

/-- The maximum value of f is 7 -/
def max_value_is_7 (a b c : ℝ) : Prop :=
  f a b c 2 = 7

/-- f passes through the point (0, -7) -/
def passes_through_0_neg7 (a b c : ℝ) : Prop :=
  f a b c 0 = -7

theorem quadratic_theorem (a b c : ℝ) 
  (h1 : has_max_at_2 a b c)
  (h2 : max_value_is_7 a b c)
  (h3 : passes_through_0_neg7 a b c) :
  f a b c 5 = -24.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_theorem_l2893_289310


namespace NUMINAMATH_CALUDE_product_equality_l2893_289373

theorem product_equality (p r j : ℝ) : 
  (6 * p^2 - 4 * p + r) * (2 * p^2 + j * p - 7) = 12 * p^4 - 34 * p^3 - 19 * p^2 + 28 * p - 21 →
  r + j = 3 := by
sorry

end NUMINAMATH_CALUDE_product_equality_l2893_289373


namespace NUMINAMATH_CALUDE_pregnant_cows_l2893_289359

theorem pregnant_cows (total_cows : ℕ) (female_ratio : ℚ) (pregnant_ratio : ℚ) : 
  total_cows = 44 →
  female_ratio = 1/2 →
  pregnant_ratio = 1/2 →
  (↑total_cows * female_ratio * pregnant_ratio : ℚ) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_pregnant_cows_l2893_289359


namespace NUMINAMATH_CALUDE_sin_graph_shift_l2893_289326

/-- Shifting the graph of y = sin(1/2x - π/6) to the left by π/3 units results in the graph of y = sin(1/2x) -/
theorem sin_graph_shift (x : ℝ) : 
  Real.sin (1/2 * (x + π/3) - π/6) = Real.sin (1/2 * x) := by sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l2893_289326


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2893_289322

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  a 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2893_289322


namespace NUMINAMATH_CALUDE_spatial_relations_l2893_289346

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (linePerpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Plane → Prop)
variable (lineIn : Line → Plane → Prop)

-- Axioms for the properties of these relations
axiom perpendicular_symmetric {a b : Plane} : perpendicular a b → perpendicular b a
axiom parallel_symmetric {a b : Plane} : parallel a b → parallel b a
axiom parallel_transitive {a b c : Plane} : parallel a b → parallel b c → parallel a c

-- The theorem to be proved
theorem spatial_relations 
  (α β γ : Plane) (l m n : Line) 
  (h_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (¬ ∀ (α β : Plane) (l : Line), perpendicular α β → linePerpendicular l β → lineParallel l α) ∧
  (∀ (α β : Plane) (l : Line), linePerpendicular l α → linePerpendicular l β → parallel α β) ∧
  (∀ (α β γ : Plane), perpendicular α γ → parallel β γ → perpendicular α β) ∧
  (¬ ∀ (α β : Plane) (m n : Line), lineIn m α → lineIn n α → lineParallel m β → lineParallel n β → parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_spatial_relations_l2893_289346


namespace NUMINAMATH_CALUDE_polynomial_always_positive_l2893_289344

theorem polynomial_always_positive (m : ℝ) : m^6 - m^5 + m^4 + m^2 - m + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_always_positive_l2893_289344


namespace NUMINAMATH_CALUDE_integer_solutions_system_l2893_289390

theorem integer_solutions_system (x y z t : ℤ) : 
  (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
  ((x, y, z, t) = (1, 0, 3, 1) ∨ 
   (x, y, z, t) = (-1, 0, -3, -1) ∨ 
   (x, y, z, t) = (3, 1, 1, 0) ∨ 
   (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l2893_289390


namespace NUMINAMATH_CALUDE_sale_price_calculation_l2893_289361

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  let sellingPrice := costPrice * (1 + profitRate)
  sellingPrice * (1 + taxRate)

/-- Theorem stating that the sale price with tax is approximately 677.61 -/
theorem sale_price_calculation :
  let costPrice := 526.50
  let profitRate := 0.17
  let taxRate := 0.10
  abs (salePriceWithTax costPrice profitRate taxRate - 677.61) < 0.01 := by
  sorry

#eval salePriceWithTax 526.50 0.17 0.10

end NUMINAMATH_CALUDE_sale_price_calculation_l2893_289361


namespace NUMINAMATH_CALUDE_original_price_calculation_l2893_289312

/-- Proves that if an article is sold for $120 with a 20% gain, its original price was $100. -/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 120 ∧ gain_percent = 20 → 
  selling_price = (100 : ℝ) * (1 + gain_percent / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2893_289312


namespace NUMINAMATH_CALUDE_no_factors_of_polynomial_l2893_289355

theorem no_factors_of_polynomial (x : ℝ) : 
  let p (x : ℝ) := x^4 - 4*x^2 + 16
  let f1 (x : ℝ) := x^2 + 4
  let f2 (x : ℝ) := x^2 - 1
  let f3 (x : ℝ) := x^2 + 1
  let f4 (x : ℝ) := x^2 + 3*x + 2
  (∃ (y : ℝ), p x = f1 x * y) = False ∧
  (∃ (y : ℝ), p x = f2 x * y) = False ∧
  (∃ (y : ℝ), p x = f3 x * y) = False ∧
  (∃ (y : ℝ), p x = f4 x * y) = False :=
by sorry

end NUMINAMATH_CALUDE_no_factors_of_polynomial_l2893_289355


namespace NUMINAMATH_CALUDE_f_range_implies_a_range_l2893_289300

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + a else -a * (x - 2)^2 + 1

/-- The range of f(x) is (-∞, +∞) -/
def has_full_range (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- The range of a is (0, 2] -/
def a_range : Set ℝ := Set.Ioo 0 2 ∪ {2}

theorem f_range_implies_a_range :
  ∀ a : ℝ, has_full_range a → a ∈ a_range :=
sorry

end NUMINAMATH_CALUDE_f_range_implies_a_range_l2893_289300


namespace NUMINAMATH_CALUDE_no_double_composition_f_l2893_289333

def q : ℕ+ → ℕ+ :=
  fun n => match n with
  | 1 => 3
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => n

theorem no_double_composition_f (f : ℕ+ → ℕ+) :
  ¬(∀ n : ℕ+, f (f n) = q n + 2) :=
sorry

end NUMINAMATH_CALUDE_no_double_composition_f_l2893_289333


namespace NUMINAMATH_CALUDE_total_bee_legs_l2893_289327

/-- The number of legs a single bee has -/
def legs_per_bee : ℕ := 6

/-- The number of bees -/
def num_bees : ℕ := 8

/-- Theorem: The total number of legs for 8 bees is 48 -/
theorem total_bee_legs : legs_per_bee * num_bees = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_bee_legs_l2893_289327


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_ferry_tourists_sum_proof_l2893_289324

theorem ferry_tourists_sum : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun n a d s =>
    n = 10 ∧ a = 120 ∧ d = 2 →
    s = n * (2 * a - (n - 1) * d) / 2 →
    s = 1110

-- The proof is omitted
theorem ferry_tourists_sum_proof : ferry_tourists_sum 10 120 2 1110 := by sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_ferry_tourists_sum_proof_l2893_289324


namespace NUMINAMATH_CALUDE_house_height_from_shadows_l2893_289358

/-- Given a tree and a house casting shadows, calculate the height of the house -/
theorem house_height_from_shadows 
  (tree_height : ℝ) 
  (tree_shadow : ℝ) 
  (house_shadow : ℝ) 
  (h_tree_height : tree_height = 15)
  (h_tree_shadow : tree_shadow = 18)
  (h_house_shadow : house_shadow = 72)
  (h_similar_triangles : tree_height / tree_shadow = house_height / house_shadow) :
  house_height = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_house_height_from_shadows_l2893_289358


namespace NUMINAMATH_CALUDE_marble_distribution_l2893_289392

theorem marble_distribution (y : ℕ) : 
  let first_friend := 2 * y + 2
  let second_friend := y
  let third_friend := 3 * y - 1
  first_friend + second_friend + third_friend = 6 * y + 1 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l2893_289392


namespace NUMINAMATH_CALUDE_find_a_l2893_289311

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then a^x else -a^(-x)

-- State the theorem
theorem find_a : 
  ∀ a : ℝ, 
  (a > 0) → 
  (a ≠ 1) → 
  (∀ x : ℝ, f a x = -(f a (-x))) → 
  (f a (Real.log 4 / Real.log (1/2)) = -3) → 
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l2893_289311


namespace NUMINAMATH_CALUDE_ian_roses_problem_l2893_289314

theorem ian_roses_problem (initial_roses : ℕ) : 
  initial_roses = 6 + 9 + 4 + 1 → initial_roses = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ian_roses_problem_l2893_289314


namespace NUMINAMATH_CALUDE_group_collection_theorem_l2893_289378

/-- Calculates the total collection amount in rupees for a group where each member
    contributes as many paise as there are members in the group. -/
def total_collection (group_size : ℕ) : ℚ :=
  (group_size * group_size : ℚ) / 100

/-- Proves that for a group of 99 members, where each member contributes as many
    paise as there are members, the total collection amount is 98.01 rupees. -/
theorem group_collection_theorem :
  total_collection 99 = 98.01 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_theorem_l2893_289378


namespace NUMINAMATH_CALUDE_association_membership_l2893_289367

theorem association_membership (M : ℕ) : 
  (525 : ℕ) ≤ M ∧ 
  (315 : ℕ) = (525 * 60 : ℕ) / 100 ∧ 
  (315 : ℝ) = (M : ℝ) * 19.6875 / 100 →
  M = 1600 := by
sorry

end NUMINAMATH_CALUDE_association_membership_l2893_289367


namespace NUMINAMATH_CALUDE_parallelogram_properties_l2893_289320

-- Define a parallelogram
structure Parallelogram where
  is_quadrilateral : Bool
  opposite_sides_parallel : Bool

-- Define the properties
def has_equal_sides (p : Parallelogram) : Prop := sorry
def is_square (p : Parallelogram) : Prop := sorry

-- Theorem statement
theorem parallelogram_properties (p : Parallelogram) :
  (p.is_quadrilateral ∧ p.opposite_sides_parallel) →
  (∃ p1 : Parallelogram, has_equal_sides p1 ∧ ¬is_square p1) ∧
  (∀ p2 : Parallelogram, is_square p2 → has_equal_sides p2) ∧
  (∃ p3 : Parallelogram, has_equal_sides p3 ∧ is_square p3) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l2893_289320


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2893_289330

theorem divisibility_by_five (a b : ℕ) : 
  (5 ∣ a ∨ 5 ∣ b) → (5 ∣ a * b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2893_289330


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2893_289398

-- Define the hyperbola equation
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 4 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop :=
  y = 2 * x

-- Define the real axis length
def real_axis_length (a : ℝ) : ℝ :=
  2 * a

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∃ a : ℝ, (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  real_axis_length a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2893_289398


namespace NUMINAMATH_CALUDE_function_inequality_l2893_289353

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x ∈ (Set.Ioo 0 (π/2)), HasDerivAt f (f' x) x) :
  (∀ x ∈ (Set.Ioo 0 (π/2)), f' x * sin x - cos x * f x > 0) →
  Real.sqrt 3 * f (π/6) < f (π/3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2893_289353


namespace NUMINAMATH_CALUDE_permutation_combination_relation_l2893_289323

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def permutations (n r : ℕ) : ℕ := factorial n / factorial (n - r)

def combinations (n r : ℕ) : ℕ := factorial n / (factorial r * factorial (n - r))

theorem permutation_combination_relation :
  ∃ k : ℕ, permutations 32 6 = k * combinations 32 6 ∧ k = 720 := by
sorry

end NUMINAMATH_CALUDE_permutation_combination_relation_l2893_289323


namespace NUMINAMATH_CALUDE_max_rational_products_l2893_289365

/-- Represents a number that can be either rational or irrational -/
inductive Number
| Rational : ℚ → Number
| Irrational : ℝ → Number

/-- Definition of the table structure -/
structure Table :=
  (top : Fin 50 → Number)
  (left : Fin 50 → Number)

/-- Counts the number of rational and irrational numbers in a list -/
def countNumbers (numbers : List Number) : Nat × Nat :=
  numbers.foldl (fun (ratCount, irratCount) n =>
    match n with
    | Number.Rational _ => (ratCount + 1, irratCount)
    | Number.Irrational _ => (ratCount, irratCount + 1)
  ) (0, 0)

/-- Checks if the product of two Numbers is rational -/
def isRationalProduct (a b : Number) : Bool :=
  match a, b with
  | Number.Rational _, Number.Rational _ => true
  | Number.Rational 0, _ => true
  | _, Number.Rational 0 => true
  | _, _ => false

/-- Counts the number of rational products in the table -/
def countRationalProducts (t : Table) : Nat :=
  (List.range 50).foldl (fun count i =>
    (List.range 50).foldl (fun count j =>
      if isRationalProduct (t.top i) (t.left j) then count + 1 else count
    ) count
  ) 0

/-- The main theorem -/
theorem max_rational_products (t : Table) :
  (countNumbers (List.ofFn t.top) = (25, 25) ∧
   countNumbers (List.ofFn t.left) = (25, 25) ∧
   (∀ i j : Fin 50, t.top i ≠ t.left j) ∧
   (∃ i : Fin 50, t.top i = Number.Rational 0)) →
  countRationalProducts t ≤ 1275 :=
sorry

end NUMINAMATH_CALUDE_max_rational_products_l2893_289365


namespace NUMINAMATH_CALUDE_money_sharing_l2893_289389

theorem money_sharing (total : ℝ) (maggie_share : ℝ) : 
  maggie_share = 0.75 * total ∧ maggie_share = 4500 → total = 6000 :=
by sorry

end NUMINAMATH_CALUDE_money_sharing_l2893_289389


namespace NUMINAMATH_CALUDE_five_digit_multiples_of_five_l2893_289374

theorem five_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 90000)).card = 18000 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_multiples_of_five_l2893_289374


namespace NUMINAMATH_CALUDE_negative_sum_l2893_289305

theorem negative_sum (x w : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hw : -2 < w ∧ w < -1) : 
  x + w < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l2893_289305


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2893_289369

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem smallest_n_square_and_cube : 
  (∀ m : ℕ, m > 0 ∧ m < 100 → ¬(is_perfect_square (4*m) ∧ is_perfect_cube (5*m))) ∧ 
  (is_perfect_square (4*100) ∧ is_perfect_cube (5*100)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2893_289369


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2893_289308

theorem largest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≤ Real.sqrt n / 2) ∧
  (∀ (m : ℕ), m > n → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m > Real.sqrt m / 2) ∧
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2893_289308


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2893_289362

theorem complex_equation_solution (z : ℂ) : (Complex.I * (z + 1) = -3 + 2 * Complex.I) → z = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2893_289362


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2893_289338

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2893_289338


namespace NUMINAMATH_CALUDE_missing_number_proof_l2893_289360

theorem missing_number_proof (numbers : List ℕ) (missing : ℕ) : 
  numbers = [744, 745, 747, 748, 749, 752, 752, 753, 755] →
  (numbers.sum + missing) / 10 = 750 →
  missing = 805 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2893_289360


namespace NUMINAMATH_CALUDE_student_distribution_problem_l2893_289356

/-- The number of ways to distribute n students among k schools,
    where each school must have at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  (n - 1).choose (k - 1) * k.factorial

/-- The specific problem statement -/
theorem student_distribution_problem :
  distribute_students 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_problem_l2893_289356


namespace NUMINAMATH_CALUDE_overlapping_circles_area_l2893_289382

/-- The area of a figure consisting of two overlapping circles -/
theorem overlapping_circles_area (r1 r2 : ℝ) (overlap_area : ℝ) :
  r1 = 4 →
  r2 = 6 →
  overlap_area = 2 * Real.pi →
  (Real.pi * r1^2) + (Real.pi * r2^2) - overlap_area = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_overlapping_circles_area_l2893_289382


namespace NUMINAMATH_CALUDE_expected_adjacent_black_pairs_l2893_289337

theorem expected_adjacent_black_pairs (total_cards : ℕ) (black_cards : ℕ) (red_cards : ℕ)
  (h1 : total_cards = 60)
  (h2 : black_cards = 36)
  (h3 : red_cards = 24)
  (h4 : total_cards = black_cards + red_cards) :
  (black_cards : ℚ) * (black_cards - 1 : ℚ) / (total_cards - 1 : ℚ) = 1260 / 59 := by
sorry

end NUMINAMATH_CALUDE_expected_adjacent_black_pairs_l2893_289337


namespace NUMINAMATH_CALUDE_line_circle_intersection_condition_l2893_289357

/-- The line equation: ax + y + a + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + a + 1 = 0

/-- The circle equation: x^2 + y^2 - 2x - 2y + b = 0 -/
def circle_equation (b x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + b = 0

/-- The line intersects the circle for all real a -/
def line_intersects_circle (b : ℝ) : Prop :=
  ∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation b x y

theorem line_circle_intersection_condition :
  ∀ b : ℝ, line_intersects_circle b ↔ b < -6 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_condition_l2893_289357


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l2893_289347

theorem quadratic_inequality_implies_m_range (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≥ 0) → -1 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l2893_289347


namespace NUMINAMATH_CALUDE_machines_needed_for_faster_job_additional_machines_needed_l2893_289313

theorem machines_needed_for_faster_job (initial_machines : ℕ) (initial_days : ℕ) : ℕ :=
  let total_machine_days := initial_machines * initial_days
  let new_days := initial_days * 3 / 4
  let new_machines := total_machine_days / new_days
  new_machines - initial_machines

theorem additional_machines_needed :
  machines_needed_for_faster_job 12 40 = 4 := by
  sorry

end NUMINAMATH_CALUDE_machines_needed_for_faster_job_additional_machines_needed_l2893_289313


namespace NUMINAMATH_CALUDE_linear_function_increasing_l2893_289331

theorem linear_function_increasing (k : ℝ) (y₁ y₂ : ℝ) :
  let f := fun x => (k^2 + 1) * x - 5
  f (-3) = y₁ ∧ f 4 = y₂ → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l2893_289331


namespace NUMINAMATH_CALUDE_nina_calculation_l2893_289391

theorem nina_calculation (y : ℚ) : (y + 25) * 5 = 200 → (y - 25) / 5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_nina_calculation_l2893_289391


namespace NUMINAMATH_CALUDE_cone_volume_l2893_289335

/-- 
Given a cone with surface area 4π and whose unfolded side view is a sector 
with a central angle of 2π/3, prove that its volume is 2√2π/3.
-/
theorem cone_volume (r l h : ℝ) : 
  (π * r * l + π * r^2 = 4 * π) →  -- Surface area condition
  ((2 * π / 3) * l = 2 * π * r) →  -- Sector condition
  (h^2 + r^2 = l^2) →              -- Pythagorean theorem
  ((1/3) * π * r^2 * h = (2 * Real.sqrt 2 * π) / 3) := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l2893_289335


namespace NUMINAMATH_CALUDE_laundry_bill_calculation_l2893_289368

/-- Given a laundry bill with trousers and shirts, calculate the charge per pair of trousers. -/
theorem laundry_bill_calculation 
  (total_bill : ℝ) 
  (shirt_charge : ℝ) 
  (num_trousers : ℕ) 
  (num_shirts : ℕ) 
  (h1 : total_bill = 140)
  (h2 : shirt_charge = 5)
  (h3 : num_trousers = 10)
  (h4 : num_shirts = 10) :
  (total_bill - shirt_charge * num_shirts) / num_trousers = 9 := by
sorry

end NUMINAMATH_CALUDE_laundry_bill_calculation_l2893_289368


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2893_289317

theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 - 2*p - 2 = 0) → 
  (q^3 - 2*q - 2 = 0) → 
  (r^3 - 2*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -24 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2893_289317


namespace NUMINAMATH_CALUDE_kafelnikov_served_first_l2893_289379

/-- Represents a tennis player -/
inductive Player : Type
| Kafelnikov : Player
| Becker : Player

/-- Represents the result of a tennis match -/
structure MatchResult :=
  (winner : Player)
  (winner_games : Nat)
  (loser_games : Nat)

/-- Represents the serving pattern in a tennis match -/
structure ServingPattern :=
  (server_wins : Nat)
  (receiver_wins : Nat)

/-- Determines who served first in a tennis match -/
def first_server (result : MatchResult) (serving : ServingPattern) : Player :=
  sorry

/-- Theorem stating that Kafelnikov served first -/
theorem kafelnikov_served_first 
  (result : MatchResult) 
  (serving : ServingPattern) :
  result.winner = Player.Kafelnikov ∧
  result.winner_games = 6 ∧
  result.loser_games = 3 ∧
  serving.server_wins = 5 ∧
  serving.receiver_wins = 4 →
  first_server result serving = Player.Kafelnikov :=
by sorry

end NUMINAMATH_CALUDE_kafelnikov_served_first_l2893_289379


namespace NUMINAMATH_CALUDE_alternating_seating_theorem_l2893_289399

/-- The number of ways to arrange n girls and n boys alternately around a round table with 2n seats -/
def alternating_seating_arrangements (n : ℕ) : ℕ :=
  2 * (n.factorial)^2

/-- Theorem stating that the number of alternating seating arrangements
    for n girls and n boys around a round table with 2n seats is 2(n!)^2 -/
theorem alternating_seating_theorem (n : ℕ) :
  alternating_seating_arrangements n = 2 * (n.factorial)^2 := by
  sorry

end NUMINAMATH_CALUDE_alternating_seating_theorem_l2893_289399


namespace NUMINAMATH_CALUDE_complex_solution_l2893_289350

theorem complex_solution (z : ℂ) (h : (2 + Complex.I) * z = 3 + 4 * Complex.I) :
  z = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_solution_l2893_289350


namespace NUMINAMATH_CALUDE_problem_solution_l2893_289306

theorem problem_solution (a b : ℝ) (h1 : a - b = 7) (h2 : a * b = 18) :
  a^2 + b^2 = 85 ∧ (a + b)^2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2893_289306


namespace NUMINAMATH_CALUDE_smaller_circle_with_integer_points_l2893_289303

/-- Given a circle centered at the origin with radius R, there exists a circle
    with radius R/√2 that contains at least as many points with integer coordinates. -/
theorem smaller_circle_with_integer_points (R : ℝ) (R_pos : R > 0) :
  ∃ (R' : ℝ), R' = R / Real.sqrt 2 ∧
  (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 →
    ∃ (x' y' : ℤ), x'^2 + y'^2 ≤ R'^2) :=
by sorry

end NUMINAMATH_CALUDE_smaller_circle_with_integer_points_l2893_289303


namespace NUMINAMATH_CALUDE_f_definition_l2893_289385

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 - 4

-- State the theorem
theorem f_definition : 
  (∀ x : ℝ, f (x - 2) = x^2 - 4*x) → 
  (∀ x : ℝ, f x = x^2 - 4) := by sorry

end NUMINAMATH_CALUDE_f_definition_l2893_289385


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l2893_289394

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

theorem arithmetic_sequence_second_term 
  (y : ℝ) 
  (h1 : y > 0) 
  (h2 : is_arithmetic_sequence (5^2) (y^2) (13^2)) : 
  y = Real.sqrt 97 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l2893_289394


namespace NUMINAMATH_CALUDE_unique_ten_digit_square_match_l2893_289380

theorem unique_ten_digit_square_match : 
  ∃! (N : ℕ), 
    (10^9 ≤ N) ∧ (N < 10^10) ∧ 
    (∃ (K : ℕ), N^2 = 10^10 * K + N) ∧
    N = 10^9 := by
  sorry

end NUMINAMATH_CALUDE_unique_ten_digit_square_match_l2893_289380


namespace NUMINAMATH_CALUDE_quadratic_completion_l2893_289370

theorem quadratic_completion (y : ℝ) : ∃ a : ℝ, y^2 + 14*y + 60 = (y + a)^2 + 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2893_289370


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2893_289315

theorem trigonometric_identity : 
  2 * Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2893_289315


namespace NUMINAMATH_CALUDE_paige_to_remainder_ratio_l2893_289318

/-- Represents the number of pieces in a chocolate bar -/
def total_pieces : ℕ := 60

/-- Represents the number of pieces Michael takes -/
def michael_pieces : ℕ := total_pieces / 2

/-- Represents the number of pieces Mandy gets -/
def mandy_pieces : ℕ := 15

/-- Represents the number of pieces Paige takes -/
def paige_pieces : ℕ := total_pieces - michael_pieces - mandy_pieces

/-- Theorem stating the ratio of Paige's pieces to pieces left after Michael's share -/
theorem paige_to_remainder_ratio :
  (paige_pieces : ℚ) / (total_pieces - michael_pieces : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_paige_to_remainder_ratio_l2893_289318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2893_289377

/-- An arithmetic sequence and its partial sums -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := seq.a 2 - seq.a 1

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 5 < seq.S 6)
    (h2 : seq.S 6 = seq.S 7)
    (h3 : seq.S 7 > seq.S 8) :
  (common_difference seq < 0) ∧ 
  (seq.a 7 = 0) ∧ 
  (seq.S 9 ≤ seq.S 5) ∧
  (∀ n : ℕ+, seq.S n ≤ seq.S 6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2893_289377


namespace NUMINAMATH_CALUDE_sum_convergence_l2893_289329

/-- Given two sequences of real numbers (a_n) and (b_n) satisfying the condition
    (3 - 2i)^n = a_n + b_ni for all integers n ≥ 0, where i = √(-1),
    prove that the sum ∑(n=0 to ∞) (a_n * b_n) / 8^n converges to 4/5. -/
theorem sum_convergence (a b : ℕ → ℝ) 
    (h : ∀ n : ℕ, Complex.I ^ 2 = -1 ∧ (3 - 2 * Complex.I) ^ n = a n + b n * Complex.I) :
    HasSum (λ n => (a n * b n) / 8^n) (4/5) :=
by sorry

end NUMINAMATH_CALUDE_sum_convergence_l2893_289329


namespace NUMINAMATH_CALUDE_horner_operation_count_l2893_289345

/-- Polynomial coefficients in descending order of degree -/
def polynomial : List ℝ := [3, 4, 5, 6, 7, 8, 1]

/-- Number of multiplication operations in Horner's method -/
def horner_mult_ops (p : List ℝ) : ℕ := p.length - 1

/-- Number of addition operations in Horner's method -/
def horner_add_ops (p : List ℝ) : ℕ := p.length - 1

theorem horner_operation_count :
  horner_mult_ops polynomial = 6 ∧ horner_add_ops polynomial = 6 := by
  sorry

end NUMINAMATH_CALUDE_horner_operation_count_l2893_289345


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2893_289302

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 2.44

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 0.87

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem total_cost_calculation :
  total_cost = 8.36 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2893_289302


namespace NUMINAMATH_CALUDE_determinant_calculation_l2893_289397

variable (a₁ b₁ b₂ c₁ c₂ c₃ d₁ d₂ d₃ d₄ : ℝ)

def matrix : Matrix (Fin 4) (Fin 4) ℝ := λ i j =>
  match i, j with
  | 0, 0 => a₁
  | 0, 1 => b₁
  | 0, 2 => c₁
  | 0, 3 => d₁
  | 1, 0 => a₁
  | 1, 1 => b₂
  | 1, 2 => c₂
  | 1, 3 => d₂
  | 2, 0 => a₁
  | 2, 1 => b₂
  | 2, 2 => c₃
  | 2, 3 => d₃
  | 3, 0 => a₁
  | 3, 1 => b₂
  | 3, 2 => c₃
  | 3, 3 => d₄
  | _, _ => 0

theorem determinant_calculation :
  Matrix.det (matrix a₁ b₁ b₂ c₁ c₂ c₃ d₁ d₂ d₃ d₄) = a₁ * (b₂ - b₁) * (c₃ - c₂) * (d₄ - d₃) := by
  sorry

end NUMINAMATH_CALUDE_determinant_calculation_l2893_289397


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2893_289316

theorem inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 1 2 = {x | x^2 - a*x + b < 0}) :
  {x : ℝ | 1/x < b/a} = Set.union (Set.Iio 0) (Set.Ioi (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2893_289316


namespace NUMINAMATH_CALUDE_luke_gave_five_stickers_l2893_289352

/-- Calculates the number of stickers Luke gave to his sister -/
def stickers_given_to_sister (initial : ℕ) (bought : ℕ) (birthday : ℕ) (used : ℕ) (left : ℕ) : ℕ :=
  initial + bought + birthday - used - left

/-- Proves that Luke gave 5 stickers to his sister -/
theorem luke_gave_five_stickers :
  stickers_given_to_sister 20 12 20 8 39 = 5 := by
  sorry

#eval stickers_given_to_sister 20 12 20 8 39

end NUMINAMATH_CALUDE_luke_gave_five_stickers_l2893_289352


namespace NUMINAMATH_CALUDE_sum_s_r_equals_negative_62_l2893_289396

def r (x : ℝ) : ℝ := abs x + 1

def s (x : ℝ) : ℝ := -2 * abs x

def xValues : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_s_r_equals_negative_62 :
  (xValues.map (fun x => s (r x))).sum = -62 := by
  sorry

end NUMINAMATH_CALUDE_sum_s_r_equals_negative_62_l2893_289396


namespace NUMINAMATH_CALUDE_gcd_2873_1233_l2893_289332

theorem gcd_2873_1233 : Nat.gcd 2873 1233 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2873_1233_l2893_289332


namespace NUMINAMATH_CALUDE_f_period_and_range_l2893_289384

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + Real.sqrt 3 / 2

theorem f_period_and_range :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-Real.sqrt 3 / 2) 1 ↔
    ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_f_period_and_range_l2893_289384


namespace NUMINAMATH_CALUDE_simplify_expression_l2893_289387

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2893_289387


namespace NUMINAMATH_CALUDE_additive_inverse_of_negative_2023_l2893_289341

theorem additive_inverse_of_negative_2023 :
  ∃! x : ℝ, -2023 + x = 0 ∧ x = 2023 := by sorry

end NUMINAMATH_CALUDE_additive_inverse_of_negative_2023_l2893_289341


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l2893_289351

def solution_set (x : ℝ) := -1 < x ∧ x < 0

theorem abs_inequality_solution_set :
  {x : ℝ | |2*x + 1| < 1} = {x : ℝ | solution_set x} := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l2893_289351


namespace NUMINAMATH_CALUDE_overall_average_l2893_289301

theorem overall_average (n : ℕ) (avg_first : ℝ) (avg_last : ℝ) (middle : ℝ) :
  n = 25 →
  avg_first = 14 →
  avg_last = 17 →
  middle = 78 →
  (avg_first * 12 + middle + avg_last * 12) / n = 18 := by
sorry

end NUMINAMATH_CALUDE_overall_average_l2893_289301


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2893_289334

theorem trigonometric_equation_solution (x : ℝ) :
  2 * Real.cos (13 * x) + 3 * Real.cos (3 * x) + 3 * Real.cos (5 * x) - 8 * Real.cos x * (Real.cos (4 * x))^3 = 0 →
  ∃ k : ℤ, x = π * k / 12 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2893_289334


namespace NUMINAMATH_CALUDE_count_nine_digit_integers_l2893_289343

/-- The number of different 9-digit positive integers -/
def nine_digit_integers : ℕ := 9 * (10 ^ 8)

theorem count_nine_digit_integers : nine_digit_integers = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_count_nine_digit_integers_l2893_289343
