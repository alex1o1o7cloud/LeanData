import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_equation_l604_60489

/-- The slope of the line perpendicular to 2x - 6y + 1 = 0 -/
def perpendicular_slope : ℝ := -3

/-- The equation of the curve -/
def curve (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

/-- The derivative of the curve -/
def curve_derivative (x : ℝ) : ℝ := 3*x^2 + 6*x

theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ),
    curve x₀ = y₀ ∧
    curve_derivative x₀ = perpendicular_slope ∧
    ∀ (x y : ℝ), y - y₀ = perpendicular_slope * (x - x₀) ↔ 3*x + y + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l604_60489


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l604_60466

theorem fractional_equation_solution : 
  ∃ x : ℝ, (x * (x - 2) ≠ 0) ∧ (5 / (x - 2) = 3 / x) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l604_60466


namespace NUMINAMATH_CALUDE_jason_stored_23_bales_l604_60470

/-- The number of bales Jason stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem: Jason stored 23 bales in the barn -/
theorem jason_stored_23_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73) 
  (h2 : final_bales = 96) : 
  bales_stored initial_bales final_bales = 23 := by
  sorry

end NUMINAMATH_CALUDE_jason_stored_23_bales_l604_60470


namespace NUMINAMATH_CALUDE_sum_vertices_is_nine_l604_60406

/-- The number of vertices in a rectangle -/
def rectangle_vertices : ℕ := 4

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- The sum of vertices of a rectangle and a pentagon -/
def sum_vertices : ℕ := rectangle_vertices + pentagon_vertices

theorem sum_vertices_is_nine : sum_vertices = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_vertices_is_nine_l604_60406


namespace NUMINAMATH_CALUDE_inequality_problem_l604_60453

theorem inequality_problem (s x y : ℝ) (h1 : s > 0) (h2 : x^2 + y^2 ≠ 0) (h3 : x * s^2 < y * s^2) :
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l604_60453


namespace NUMINAMATH_CALUDE_max_m_value_l604_60424

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m * a * b / (3 * a + b) ≤ a + 3 * b) ↔ m ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l604_60424


namespace NUMINAMATH_CALUDE_inequality_proof_l604_60455

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l604_60455


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l604_60412

theorem proportion_fourth_term (x y : ℝ) : 
  (0.75 : ℝ) / 0.6 = 10 / y → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l604_60412


namespace NUMINAMATH_CALUDE_jezebel_bouquet_cost_l604_60405

/-- The cost of a bouquet of flowers -/
def bouquet_cost (red_roses_per_dozen : ℕ) (red_rose_cost : ℚ) (sunflowers : ℕ) (sunflower_cost : ℚ) : ℚ :=
  (red_roses_per_dozen * 12 * red_rose_cost) + (sunflowers * sunflower_cost)

/-- Theorem: The cost of Jezebel's bouquet is $45 -/
theorem jezebel_bouquet_cost :
  bouquet_cost 2 (3/2) 3 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jezebel_bouquet_cost_l604_60405


namespace NUMINAMATH_CALUDE_invisible_dots_count_l604_60496

-- Define the number of dice
def num_dice : ℕ := 4

-- Define the numbers on a single die
def die_numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the visible numbers
def visible_numbers : List ℕ := [2, 2, 3, 4, 4, 5, 6, 6]

-- Theorem to prove
theorem invisible_dots_count :
  (num_dice * (die_numbers.sum)) - (visible_numbers.sum) = 52 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l604_60496


namespace NUMINAMATH_CALUDE_marble_sculpture_second_week_cut_l604_60451

/-- Proves that the percentage of marble cut away in the second week is 20% --/
theorem marble_sculpture_second_week_cut (
  original_weight : ℝ)
  (first_week_cut_percent : ℝ)
  (third_week_cut_percent : ℝ)
  (final_weight : ℝ)
  (h1 : original_weight = 250)
  (h2 : first_week_cut_percent = 30)
  (h3 : third_week_cut_percent = 25)
  (h4 : final_weight = 105)
  : ∃ (second_week_cut_percent : ℝ),
    second_week_cut_percent = 20 ∧
    final_weight = original_weight *
      (1 - first_week_cut_percent / 100) *
      (1 - second_week_cut_percent / 100) *
      (1 - third_week_cut_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_marble_sculpture_second_week_cut_l604_60451


namespace NUMINAMATH_CALUDE_distance_between_points_l604_60409

theorem distance_between_points : 
  let point1 : ℝ × ℝ := (0, 6)
  let point2 : ℝ × ℝ := (4, 0)
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l604_60409


namespace NUMINAMATH_CALUDE_difference_of_squares_l604_60428

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l604_60428


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l604_60443

theorem max_value_sqrt_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 18) :
  Real.sqrt (35 - x) + Real.sqrt x + Real.sqrt (18 - x) ≤ Real.sqrt 35 + Real.sqrt 18 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l604_60443


namespace NUMINAMATH_CALUDE_pool_capacity_l604_60498

theorem pool_capacity (C : ℝ) 
  (h1 : 0.8 * C = 0.5 * C + 300) 
  (h2 : 300 = 0.3 * C) : 
  C = 1000 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_l604_60498


namespace NUMINAMATH_CALUDE_exists_linear_bound_l604_60465

def Color := Bool

def is_valid_coloring (coloring : ℕ+ → Color) : Prop :=
  ∀ n : ℕ+, coloring n = true ∨ coloring n = false

structure ColoredIntegerFunction where
  f : ℕ+ → ℕ+
  coloring : ℕ+ → Color
  is_valid_coloring : is_valid_coloring coloring
  monotone : ∀ x y : ℕ+, x ≤ y → f x ≤ f y
  color_additive : ∀ x y z : ℕ+, 
    coloring x = coloring y ∧ coloring y = coloring z → 
    x + y = z → f x + f y = f z

theorem exists_linear_bound (cf : ColoredIntegerFunction) : 
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℕ+, (cf.f x : ℝ) ≤ a * x :=
sorry

end NUMINAMATH_CALUDE_exists_linear_bound_l604_60465


namespace NUMINAMATH_CALUDE_quadratic_ratio_l604_60425

theorem quadratic_ratio (x : ℝ) : 
  ∃ (d e : ℝ), 
    (∀ x, x^2 + 900*x + 1800 = (x + d)^2 + e) ∧ 
    (e / d = -446) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l604_60425


namespace NUMINAMATH_CALUDE_intersection_with_complement_l604_60427

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_with_complement : M ∩ (I \ N) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l604_60427


namespace NUMINAMATH_CALUDE_function_value_at_ten_l604_60432

/-- Given a function f satisfying the recursive relation
    f(x+1) = f(x) / (1 + f(x)) for all x, and f(1) = 1,
    prove that f(10) = 1/10 -/
theorem function_value_at_ten
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 1) = f x / (1 + f x))
  (h2 : f 1 = 1) :
  f 10 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_ten_l604_60432


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l604_60437

theorem smallest_factorization_coefficient (b : ℕ+) (p q : ℤ) : 
  (∀ x, (x^2 : ℤ) + b * x + 1760 = (x + p) * (x + q)) →
  (∀ b' : ℕ+, b' < b → 
    ¬∃ p' q' : ℤ, ∀ x, (x^2 : ℤ) + b' * x + 1760 = (x + p') * (x + q')) →
  b = 108 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l604_60437


namespace NUMINAMATH_CALUDE_ham_bread_percentage_l604_60476

def bread_cost : ℕ := 50
def ham_cost : ℕ := 150
def cake_cost : ℕ := 200

def total_cost : ℕ := bread_cost + ham_cost + cake_cost
def ham_bread_cost : ℕ := bread_cost + ham_cost

theorem ham_bread_percentage :
  (ham_bread_cost : ℚ) / (total_cost : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ham_bread_percentage_l604_60476


namespace NUMINAMATH_CALUDE_unique_root_quadratic_root_l604_60484

/-- A quadratic polynomial with exactly one root -/
structure UniqueRootQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  has_unique_root : (b ^ 2 - 4 * a * c) = 0

/-- The theorem stating that the root of the quadratic polynomial is -11 -/
theorem unique_root_quadratic_root (f : UniqueRootQuadratic) 
  (h : ∃ g : UniqueRootQuadratic, 
    g.a = -f.a ∧ 
    g.b = (f.b - 30 * f.a) ∧ 
    g.c = (17 * f.a - 7 * f.b + f.c)) :
  (f.a ≠ 0) → (-f.b / (2 * f.a)) = -11 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_root_l604_60484


namespace NUMINAMATH_CALUDE_points_order_l604_60416

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = 3 * x - b

-- Define the points
def point1 (y₁ b : ℝ) : Prop := line_equation (-3) y₁ b
def point2 (y₂ b : ℝ) : Prop := line_equation 1 y₂ b
def point3 (y₃ b : ℝ) : Prop := line_equation (-1) y₃ b

theorem points_order (y₁ y₂ y₃ b : ℝ) 
  (h1 : point1 y₁ b) (h2 : point2 y₂ b) (h3 : point3 y₃ b) : 
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_order_l604_60416


namespace NUMINAMATH_CALUDE_division_problem_l604_60417

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 181)
  (h2 : divisor = 20)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l604_60417


namespace NUMINAMATH_CALUDE_triangle_area_l604_60403

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 - t.b * t.c ∧
  t.b * t.c * Real.cos t.A = -4

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : satisfies_conditions t) : 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l604_60403


namespace NUMINAMATH_CALUDE_percentage_problem_l604_60471

theorem percentage_problem (p : ℝ) : 
  (p / 100) * 180 - (1 / 3) * ((p / 100) * 180) = 18 ↔ p = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l604_60471


namespace NUMINAMATH_CALUDE_monica_wednesday_study_time_l604_60481

/-- Represents the study schedule of Monica over five days -/
structure StudySchedule where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  weekend : ℝ
  total : ℝ

/-- The study schedule satisfies the given conditions -/
def validSchedule (s : StudySchedule) : Prop :=
  s.thursday = 3 * s.wednesday ∧
  s.friday = 1.5 * s.wednesday ∧
  s.weekend = 5.5 * s.wednesday ∧
  s.total = 22 ∧
  s.total = s.wednesday + s.thursday + s.friday + s.weekend

/-- Theorem stating that Monica studied 2 hours on Wednesday -/
theorem monica_wednesday_study_time (s : StudySchedule) 
  (h : validSchedule s) : s.wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_monica_wednesday_study_time_l604_60481


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l604_60431

theorem subset_implies_m_equals_one (m : ℝ) : 
  let A : Set ℝ := {3, m^2}
  let B : Set ℝ := {-1, 3, 2*m - 1}
  A ⊆ B → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l604_60431


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_88_l604_60448

theorem triangle_perimeter_not_88 (a b x : ℝ) (h1 : a = 18) (h2 : b = 25) (h3 : a + b > x) (h4 : a + x > b) (h5 : b + x > a) : a + b + x ≠ 88 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_88_l604_60448


namespace NUMINAMATH_CALUDE_proposition_equivalence_l604_60490

theorem proposition_equivalence (a : ℝ) : 
  (∀ x : ℝ, ((x < a ∨ x > a + 1) → (x ≤ 1/2 ∨ x ≥ 1)) ∧ 
   ∃ x : ℝ, (x ≤ 1/2 ∨ x ≥ 1) ∧ ¬(x < a ∨ x > a + 1)) ↔ 
  (0 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l604_60490


namespace NUMINAMATH_CALUDE_quadratic_properties_l604_60407

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a > 0)
  (hc : c > 0)
  (hf : f a b c c = 0)
  (h_positive : ∀ x, 0 < x → x < c → f a b c x > 0)
  (h_distinct : ∃ x, x ≠ c ∧ f a b c x = 0) :
  -- 1. The other x-intercept is at x = 1/a
  (∃ x, x ≠ c ∧ f a b c x = 0 ∧ x = 1/a) ∧
  -- 2. f(x) < 0 for x ∈ (c, 1/a)
  (∀ x, c < x → x < 1/a → f a b c x < 0) ∧
  -- 3. If the area of the triangle is 8, then 0 < a ≤ 1/8
  (((1/a - c) * c / 2 = 8) → (0 < a ∧ a ≤ 1/8)) ∧
  -- 4. If m^2 - 2km + 1 + b + ac ≥ 0 for all k ∈ [-1, 1], then m ≤ -2 or m = 0 or m ≥ 2
  ((∀ k, -1 ≤ k → k ≤ 1 → ∀ m, m^2 - 2*k*m + 1 + b + a*c ≥ 0) →
   ∀ m, m ≤ -2 ∨ m = 0 ∨ m ≥ 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l604_60407


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l604_60473

theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c > 0 ↔ x < -1 ∨ x > 2) → 
  b + c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l604_60473


namespace NUMINAMATH_CALUDE_sin_angle_RPT_l604_60478

theorem sin_angle_RPT (RPQ : Real) (h : Real.sin RPQ = 3/5) : 
  Real.sin (2 * Real.pi - RPQ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_angle_RPT_l604_60478


namespace NUMINAMATH_CALUDE_dawn_savings_l604_60468

/-- Dawn's financial situation --/
def dawn_finances : Prop :=
  let annual_income : ℝ := 48000
  let monthly_income : ℝ := annual_income / 12
  let tax_rate : ℝ := 0.20
  let variable_expense_rate : ℝ := 0.30
  let stock_investment_rate : ℝ := 0.05
  let retirement_contribution_rate : ℝ := 0.15
  let savings_rate : ℝ := 0.10
  let after_tax_income : ℝ := monthly_income * (1 - tax_rate)
  let variable_expenses : ℝ := after_tax_income * variable_expense_rate
  let stock_investment : ℝ := after_tax_income * stock_investment_rate
  let retirement_contribution : ℝ := after_tax_income * retirement_contribution_rate
  let total_deductions : ℝ := variable_expenses + stock_investment + retirement_contribution
  let remaining_income : ℝ := after_tax_income - total_deductions
  let monthly_savings : ℝ := remaining_income * savings_rate
  monthly_savings = 160

theorem dawn_savings : dawn_finances := by
  sorry

end NUMINAMATH_CALUDE_dawn_savings_l604_60468


namespace NUMINAMATH_CALUDE_shopping_remaining_amount_l604_60447

def initial_amount : ℚ := 74
def sweater_cost : ℚ := 9
def tshirt_cost : ℚ := 11
def shoes_cost : ℚ := 30
def refund_percentage : ℚ := 90 / 100

theorem shopping_remaining_amount :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost * (1 - refund_percentage)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remaining_amount_l604_60447


namespace NUMINAMATH_CALUDE_product_of_largest_and_smallest_l604_60426

/-- The set of digits allowed to form the numbers -/
def allowed_digits : Finset Nat := {0, 2, 4, 6}

/-- Predicate to check if a number is a valid three-digit number using allowed digits -/
def is_valid_number (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ 
  (∃ a b c, n = 100 * a + 10 * b + c ∧ 
            a ∈ allowed_digits ∧ b ∈ allowed_digits ∧ c ∈ allowed_digits ∧
            a ≠ b ∧ b ≠ c ∧ a ≠ c)

/-- The largest valid number -/
def largest_number : Nat := 642

/-- The smallest valid number -/
def smallest_number : Nat := 204

theorem product_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : Nat, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : Nat, is_valid_number n → n ≥ smallest_number) ∧
  largest_number * smallest_number = 130968 := by
  sorry

#check product_of_largest_and_smallest

end NUMINAMATH_CALUDE_product_of_largest_and_smallest_l604_60426


namespace NUMINAMATH_CALUDE_tv_diagonal_problem_l604_60446

theorem tv_diagonal_problem (larger_diagonal smaller_diagonal : ℝ) :
  larger_diagonal = 24 →
  larger_diagonal ^ 2 / 2 - smaller_diagonal ^ 2 / 2 = 143.5 →
  smaller_diagonal = 17 := by
sorry

end NUMINAMATH_CALUDE_tv_diagonal_problem_l604_60446


namespace NUMINAMATH_CALUDE_log_cutting_theorem_l604_60479

/-- The number of cuts required to divide a log into a given number of pieces -/
def num_cuts (num_pieces : ℕ) : ℕ :=
  num_pieces - 1

/-- Theorem: For a log cut into 12 pieces (including fixed ends), 11 cuts are required -/
theorem log_cutting_theorem :
  let total_pieces := 12
  num_cuts total_pieces = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_cutting_theorem_l604_60479


namespace NUMINAMATH_CALUDE_problem_statement_l604_60491

theorem problem_statement (a b c d e : ℕ+) : 
  a * b * c * d * e = 362880 →
  a * b + a + b = 728 →
  b * c + b + c = 342 →
  c * d + c + d = 464 →
  d * e + d + e = 780 →
  (a : ℤ) - (e : ℤ) = 172 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l604_60491


namespace NUMINAMATH_CALUDE_negative_expression_l604_60400

theorem negative_expression : 
  -(-2) > 0 ∧ (-1)^2023 < 0 ∧ |-1^2| > 0 ∧ (-5)^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expression_l604_60400


namespace NUMINAMATH_CALUDE_not_in_A_negative_eleven_in_A_three_k_squared_minus_one_in_A_negative_thirty_four_l604_60480

-- Define the set A
def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

-- Theorem 1: -11 is not an element of A
theorem not_in_A_negative_eleven : -11 ∉ A := by sorry

-- Theorem 2: For any integer k, 3k² - 1 is an element of A
theorem in_A_three_k_squared_minus_one (k : ℤ) : 3 * k^2 - 1 ∈ A := by sorry

-- Theorem 3: -34 is an element of A
theorem in_A_negative_thirty_four : -34 ∈ A := by sorry

end NUMINAMATH_CALUDE_not_in_A_negative_eleven_in_A_three_k_squared_minus_one_in_A_negative_thirty_four_l604_60480


namespace NUMINAMATH_CALUDE_middle_number_is_four_l604_60419

def is_valid_triple (a b c : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ a + b + c = 13

def multiple_possibilities_for_bc (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ is_valid_triple a b₁ c₁ ∧ is_valid_triple a b₂ c₂

def multiple_possibilities_for_ab (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ is_valid_triple a₁ b₁ c ∧ is_valid_triple a₂ b₂ c

def multiple_possibilities_for_ac (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, a₁ ≠ a₂ ∧ is_valid_triple a₁ b c₁ ∧ is_valid_triple a₂ b c₂

theorem middle_number_is_four (a b c : ℕ) :
  is_valid_triple a b c →
  multiple_possibilities_for_bc a →
  multiple_possibilities_for_ab c →
  multiple_possibilities_for_ac b →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_four_l604_60419


namespace NUMINAMATH_CALUDE_trig_expression_equality_l604_60464

theorem trig_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 
  Real.sin (5 * π / 180) / Real.sin (8 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l604_60464


namespace NUMINAMATH_CALUDE_system_solution_l604_60411

theorem system_solution (x y m n : ℝ) 
  (eq1 : x + y = m)
  (eq2 : x - y = n + 1)
  (sol_x : x = 3)
  (sol_y : y = 2) :
  m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l604_60411


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l604_60414

theorem missing_digit_divisible_by_six (n : ℕ) (h1 : n ≥ 100 ∧ n < 1000) 
  (h2 : ∃ d : ℕ, d < 10 ∧ n = 500 + 10 * d + 2) (h3 : n % 6 = 0) : 
  ∃ d : ℕ, d = 2 ∧ n = 500 + 10 * d + 2 := by
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l604_60414


namespace NUMINAMATH_CALUDE_factor_expression_l604_60457

theorem factor_expression (a : ℝ) : 37 * a^2 + 111 * a = 37 * a * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l604_60457


namespace NUMINAMATH_CALUDE_unique_solution_condition_l604_60430

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 3 + a = b * x + 2) ↔ b ≠ 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l604_60430


namespace NUMINAMATH_CALUDE_john_pushups_l604_60450

def zachary_pushups : ℕ := 51
def david_pushups_difference : ℕ := 22
def john_pushups_difference : ℕ := 4

theorem john_pushups : 
  zachary_pushups + david_pushups_difference - john_pushups_difference = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_john_pushups_l604_60450


namespace NUMINAMATH_CALUDE_g_53_l604_60495

/-- A function satisfying g(xy) = yg(x) for all real x and y, with g(1) = 15 -/
def g : ℝ → ℝ :=
  sorry

/-- The functional equation for g -/
axiom g_eq (x y : ℝ) : g (x * y) = y * g x

/-- The value of g at 1 -/
axiom g_one : g 1 = 15

/-- The theorem to be proved -/
theorem g_53 : g 53 = 795 :=
  sorry

end NUMINAMATH_CALUDE_g_53_l604_60495


namespace NUMINAMATH_CALUDE_alyssa_future_games_l604_60454

/-- The number of soccer games Alyssa attended this year -/
def games_this_year : ℕ := 11

/-- The number of soccer games Alyssa missed this year -/
def games_missed_this_year : ℕ := 12

/-- The number of soccer games Alyssa attended last year -/
def games_last_year : ℕ := 13

/-- The total number of soccer games Alyssa will attend over three years -/
def total_games : ℕ := 39

/-- The number of games Alyssa plans to attend next year -/
def games_next_year : ℕ := total_games - (games_this_year + games_last_year)

theorem alyssa_future_games : games_next_year = 15 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_future_games_l604_60454


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l604_60402

theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = 5)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) = 3 / 10 ∧ 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) ≠ 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l604_60402


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l604_60492

/-- Calculates overtime hours given regular pay rate, regular hours, and total pay -/
def calculate_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours
  let overtime_rate := 2 * regular_rate
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Theorem stating that given the problem conditions, the overtime hours are 11 -/
theorem overtime_hours_calculation :
  let regular_rate : ℚ := 3
  let regular_hours : ℚ := 40
  let total_pay : ℚ := 186
  calculate_overtime_hours regular_rate regular_hours total_pay = 11 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l604_60492


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_l604_60456

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_n (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 1 = 1/3 →
  a 2 + a 5 = 4 →
  ∃ n : ℕ, a n = 33 →
  ∃ n : ℕ, a n = 33 ∧ n = 50 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_l604_60456


namespace NUMINAMATH_CALUDE_fuji_ratio_l604_60463

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchardConditions (o : Orchard) : Prop :=
  o.crossPollinated = o.totalTrees / 10 ∧
  o.pureFuji + o.crossPollinated = 204 ∧
  o.pureGala = 36 ∧
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated

/-- The theorem stating the ratio of pure Fuji trees to all trees -/
theorem fuji_ratio (o : Orchard) (h : orchardConditions o) :
  3 * o.totalTrees = 4 * o.pureFuji := by
  sorry


end NUMINAMATH_CALUDE_fuji_ratio_l604_60463


namespace NUMINAMATH_CALUDE_area_outside_parallel_chords_l604_60422

/-- Given a circle with radius 10 inches and two equal parallel chords 10 inches apart,
    the area of the region outside these chords but inside the circle is (200π/3 - 25√3) square inches. -/
theorem area_outside_parallel_chords (r : ℝ) (d : ℝ) : 
  r = 10 → d = 10 → 
  (2 * π * r^2 / 3 - 5 * r * Real.sqrt 3) = (200 * π / 3 - 25 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_area_outside_parallel_chords_l604_60422


namespace NUMINAMATH_CALUDE_divide_by_four_theorem_l604_60435

theorem divide_by_four_theorem (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_four_theorem_l604_60435


namespace NUMINAMATH_CALUDE_quadratic_sum_l604_60462

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (-6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 261) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l604_60462


namespace NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l604_60434

/-- Represents the step length of the hare -/
def hare_step : ℝ := 1

/-- Represents the step length of the wolf -/
def wolf_step : ℝ := 2 * hare_step

/-- Represents the number of steps the hare takes in a time unit -/
def hare_frequency : ℕ := 3

/-- Represents the number of steps the wolf takes in a time unit -/
def wolf_frequency : ℕ := 1

/-- Theorem stating that the wolf will not catch the hare -/
theorem wolf_does_not_catch_hare : 
  (hare_step * hare_frequency) > (wolf_step * wolf_frequency) := by
  sorry


end NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l604_60434


namespace NUMINAMATH_CALUDE_line_points_k_value_l604_60441

/-- A line contains the points (4, 10), (-4, k), and (-12, 6). Prove that k = 8. -/
theorem line_points_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (10 = m * 4 + b) ∧ 
    (k = m * (-4) + b) ∧ 
    (6 = m * (-12) + b)) → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l604_60441


namespace NUMINAMATH_CALUDE_tony_puzzle_time_l604_60442

/-- The total time Tony spent solving puzzles -/
def total_puzzle_time (warm_up_time : ℝ) : ℝ :=
  let challenging_puzzle_time := 3 * warm_up_time
  let set_puzzle1_time := 0.5 * warm_up_time
  let set_puzzle2_time := 2 * set_puzzle1_time
  let set_puzzle3_time := set_puzzle1_time + set_puzzle2_time + 2
  let set_puzzle4_time := 1.5 * set_puzzle3_time
  warm_up_time + 2 * challenging_puzzle_time + set_puzzle1_time + set_puzzle2_time + set_puzzle3_time + set_puzzle4_time

/-- Theorem stating that Tony spent 127.5 minutes solving puzzles -/
theorem tony_puzzle_time : total_puzzle_time 10 = 127.5 := by
  sorry

end NUMINAMATH_CALUDE_tony_puzzle_time_l604_60442


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_odd_sides_l604_60439

theorem right_triangle_consecutive_odd_sides (k : ℤ) :
  let a : ℤ := 2 * k + 1
  let c : ℤ := 2 * k + 3
  let b : ℤ := (c^2 - a^2).sqrt
  (a^2 + b^2 = c^2) → (b^2 = 8 * k + 8) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_odd_sides_l604_60439


namespace NUMINAMATH_CALUDE_parabola_coefficients_l604_60483

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_coefficients :
  ∃ (a b c : ℝ),
    (∀ x, parabola a b c x = parabola a b c (4 - x)) ∧  -- Vertical axis of symmetry at x = 2
    (parabola a b c 2 = 3) ∧                            -- Vertex at (2, 3)
    (parabola a b c 0 = 1) ∧                            -- Passes through (0, 1)
    (a = -1/2 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l604_60483


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_one_fifth_l604_60445

/-- Given that the terminal side of angle α passes through the point (3a, -4a) where a < 0,
    prove that sin α + cos α = 1/5 -/
theorem sin_plus_cos_equals_one_fifth 
  (α : Real) (a : Real) (h1 : a < 0) 
  (h2 : ∃ (t : Real), t > 0 ∧ Real.cos α = 3 * a / t ∧ Real.sin α = -4 * a / t) : 
  Real.sin α + Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_one_fifth_l604_60445


namespace NUMINAMATH_CALUDE_handbag_profit_optimization_handbag_profit_constraint_l604_60475

/-- Represents the daily sales quantity as a function of price -/
def daily_sales (x : ℝ) : ℝ := -x + 80

/-- Represents the daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - 50) * (daily_sales x)

/-- The cost price of the handbag -/
def cost_price : ℝ := 50

/-- The lower bound of the selling price -/
def price_lower_bound : ℝ := 50

/-- The upper bound of the selling price -/
def price_upper_bound : ℝ := 80

theorem handbag_profit_optimization :
  ∃ (max_price max_profit : ℝ),
    (∀ x, price_lower_bound < x ∧ x < price_upper_bound → daily_profit x ≤ max_profit) ∧
    daily_profit max_price = max_profit ∧
    max_price = 65 ∧
    max_profit = 225 :=
sorry

theorem handbag_profit_constraint (target_profit : ℝ) (price_limit : ℝ) :
  target_profit = 200 →
  price_limit = 68 →
  ∃ (optimal_price : ℝ),
    optimal_price ≤ price_limit ∧
    daily_profit optimal_price = target_profit ∧
    optimal_price = 60 :=
sorry

end NUMINAMATH_CALUDE_handbag_profit_optimization_handbag_profit_constraint_l604_60475


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l604_60440

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l604_60440


namespace NUMINAMATH_CALUDE_ice_problem_solution_l604_60485

def ice_problem (tray_a_initial tray_a_added : ℕ) : ℕ :=
  let tray_a := tray_a_initial + tray_a_added
  let tray_b := tray_a / 3
  let tray_c := 2 * tray_a
  tray_a + tray_b + tray_c

theorem ice_problem_solution :
  ice_problem 2 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ice_problem_solution_l604_60485


namespace NUMINAMATH_CALUDE_range_of_a_l604_60423

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (0 < a ∧ a ≤ 1/3) ∨ a > 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l604_60423


namespace NUMINAMATH_CALUDE_tangent_line_equation_l604_60458

/-- The equation of the tangent line to y = (3x - 2x^3) / 3 at x = 1 is y = -x + 4/3 -/
theorem tangent_line_equation (x y : ℝ) :
  let f : ℝ → ℝ := λ x => (3*x - 2*x^3) / 3
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let f' : ℝ → ℝ := λ x => 1 - 2*x^2
  y = -x + 4/3 ↔ y - y₀ = f' x₀ * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l604_60458


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l604_60401

theorem quadratic_coefficient (b : ℝ) (p : ℝ) : 
  (∀ x, x^2 + b*x + 64 = (x + p)^2 + 16) → b = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l604_60401


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l604_60472

theorem roots_sum_of_powers (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^4 + p^3*q^2 + p^2*q^3 + q^4 = 241 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l604_60472


namespace NUMINAMATH_CALUDE_budget_circle_graph_l604_60499

theorem budget_circle_graph (transportation research_development utilities equipment supplies : ℝ)
  (h1 : transportation = 15)
  (h2 : research_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : transportation + research_development + utilities + equipment + supplies < 100) :
  let salaries := 100 - (transportation + research_development + utilities + equipment + supplies)
  (salaries / 100) * 360 = 234 := by
sorry

end NUMINAMATH_CALUDE_budget_circle_graph_l604_60499


namespace NUMINAMATH_CALUDE_robotics_team_combinations_l604_60459

def girls : ℕ := 4
def boys : ℕ := 7
def team_size : ℕ := 5
def min_girls : ℕ := 2

theorem robotics_team_combinations : 
  (Finset.sum (Finset.range (girls - min_girls + 1))
    (λ k => Nat.choose girls (k + min_girls) * Nat.choose boys (team_size - (k + min_girls)))) = 301 := by
  sorry

end NUMINAMATH_CALUDE_robotics_team_combinations_l604_60459


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l604_60487

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l604_60487


namespace NUMINAMATH_CALUDE_relay_race_distance_per_member_l604_60488

theorem relay_race_distance_per_member 
  (total_distance : ℝ) 
  (team_members : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_members = 5) : 
  total_distance / team_members = 30 := by
sorry

end NUMINAMATH_CALUDE_relay_race_distance_per_member_l604_60488


namespace NUMINAMATH_CALUDE_odd_number_bound_l604_60460

/-- Sum of digits in base 2 -/
def S₂ (n : ℕ) : ℕ := sorry

theorem odd_number_bound (K a b l m : ℕ) (hK_odd : Odd K) (hS₂K : S₂ K = 2)
  (hK_factor : K = a * b) (ha_pos : a > 1) (hb_pos : b > 1)
  (hl_pos : l > 2) (hm_pos : m > 2)
  (hS₂a : S₂ a < l) (hS₂b : S₂ b < m) : K ≤ 2^(l*m - 6) + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_bound_l604_60460


namespace NUMINAMATH_CALUDE_quadratic_root_range_l604_60494

theorem quadratic_root_range (m : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - m*x + 1 = 0) ∧ 
  (α^2 - m*α + 1 = 0) ∧ 
  (β^2 - m*β + 1 = 0) ∧ 
  (0 < α) ∧ (α < 1) ∧ 
  (1 < β) ∧ (β < 2) →
  (2 < m) ∧ (m < 5/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l604_60494


namespace NUMINAMATH_CALUDE_num_perfect_square_factors_of_2940_l604_60438

/-- The number of positive integer factors of 2940 that are perfect squares -/
def num_perfect_square_factors : ℕ := 4

/-- The prime factorization of 2940 -/
def prime_factorization_2940 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1), (7, 1)]

/-- A function to check if a list represents a valid prime factorization -/
def is_valid_prime_factorization (l : List (ℕ × ℕ)) : Prop :=
  l.all (fun (p, e) => Nat.Prime p ∧ e > 0)

/-- A function to compute the product of a prime factorization -/
def product_of_factorization (l : List (ℕ × ℕ)) : ℕ :=
  l.foldl (fun acc (p, e) => acc * p^e) 1

theorem num_perfect_square_factors_of_2940 :
  is_valid_prime_factorization prime_factorization_2940 ∧
  product_of_factorization prime_factorization_2940 = 2940 →
  num_perfect_square_factors = (List.filter (fun (_, e) => e % 2 = 0) prime_factorization_2940).length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_num_perfect_square_factors_of_2940_l604_60438


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l604_60486

/-- A line passing through the point (3, -4) with equal intercepts on the coordinate axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (3, -4)
  point_condition : -4 = m * 3 + b
  -- The line has equal intercepts on both axes
  equal_intercepts : m ≠ -1 → b / (1 + m) = -b / m

/-- The equation of an EqualInterceptLine is either 4x + 3y = 0 or x + y + 1 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (4 * l.m + 3 = 0 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = -1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l604_60486


namespace NUMINAMATH_CALUDE_hyperbola_equation_l604_60474

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_asymptote : b / a = Real.sqrt 5 / 2
  h_shared_focus : ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c^2 = 3

/-- The equation of the hyperbola is x^2/4 - y^2/5 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : C.a^2 = 4 ∧ C.b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l604_60474


namespace NUMINAMATH_CALUDE_mean_home_runs_l604_60415

def home_runs_data : List (Nat × Nat) := [(5, 6), (6, 8), (4, 10)]

theorem mean_home_runs :
  let total_home_runs := (home_runs_data.map (λ (players, hrs) => players * hrs)).sum
  let total_players := (home_runs_data.map (λ (players, _) => players)).sum
  (total_home_runs : ℚ) / total_players = 118 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l604_60415


namespace NUMINAMATH_CALUDE_copy_pages_proof_l604_60493

/-- Given a cost per page in cents and a budget in dollars, 
    calculates the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Proves that with a cost of 3 cents per page and a budget of $15,
    the maximum number of pages that can be copied is 500. -/
theorem copy_pages_proof :
  max_pages_copied 3 15 = 500 := by
  sorry

#eval max_pages_copied 3 15

end NUMINAMATH_CALUDE_copy_pages_proof_l604_60493


namespace NUMINAMATH_CALUDE_two_A_minus_four_B_y_value_when_independent_of_x_l604_60461

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y - 2 * x
def B (x y : ℝ) : ℝ := x^2 - x * y + 1

-- Theorem 1: 2A - 4B = 10xy - 4x - 4
theorem two_A_minus_four_B (x y : ℝ) :
  2 * A x y - 4 * B x y = 10 * x * y - 4 * x - 4 := by sorry

-- Theorem 2: When 2A - 4B is independent of x, y = 2/5
theorem y_value_when_independent_of_x (y : ℝ) :
  (∀ x : ℝ, 2 * A x y - 4 * B x y = 10 * x * y - 4 * x - 4) →
  y = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_two_A_minus_four_B_y_value_when_independent_of_x_l604_60461


namespace NUMINAMATH_CALUDE_car_license_combinations_l604_60449

def letter_choices : ℕ := 2
def digit_choices : ℕ := 10
def num_digits : ℕ := 6

def total_license_combinations : ℕ := letter_choices * digit_choices ^ num_digits

theorem car_license_combinations :
  total_license_combinations = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_car_license_combinations_l604_60449


namespace NUMINAMATH_CALUDE_right_triangle_sinC_l604_60482

theorem right_triangle_sinC (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sinC_l604_60482


namespace NUMINAMATH_CALUDE_function_range_theorem_l604_60477

-- Define the function types
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def OddFunction (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define the main theorem
theorem function_range_theorem (f g : ℝ → ℝ) (a : ℝ) :
  EvenFunction f →
  OddFunction g →
  (∀ x, f x + g x = 2^(x + 1)) →
  (∀ x, a * f (2*x) + g x ≤ 25/8 + a * f (2*0) + g 0) →
  (∀ x, a * f (2*x) + g x ≥ a * f (2*0) + g 0 - 25/8) →
  -2 ≤ a ∧ a ≤ 13/18 :=
by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l604_60477


namespace NUMINAMATH_CALUDE_days_in_year_l604_60408

/-- The number of days in a year, given the number of hours in a year and hours in a day -/
theorem days_in_year (hours_in_year : ℕ) (hours_in_day : ℕ) 
  (h1 : hours_in_year = 8760) (h2 : hours_in_day = 24) : 
  hours_in_year / hours_in_day = 365 := by
  sorry

end NUMINAMATH_CALUDE_days_in_year_l604_60408


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l604_60429

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Theorem stating the center and radius of the circle
theorem circle_center_and_radius :
  (∃ (x₀ y₀ r : ℝ), (∀ x y : ℝ, C x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧ x₀ = 2 ∧ y₀ = -1 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l604_60429


namespace NUMINAMATH_CALUDE_f_properties_when_a_is_1_f_minimum_on_interval_l604_60497

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part 1
theorem f_properties_when_a_is_1 :
  let a := 1
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f a x > f a y ∧
  ∀ z : ℝ, f a z ≥ 1 ∧ ∃ w : ℝ, f a w = 1 := by sorry

-- Part 2
theorem f_minimum_on_interval (a : ℝ) (h : a ≥ -1) :
  let min_value := if a < 1 then -a^2 + 2 else 3 - 2*a
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≥ min_value ∧
  ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ f a y = min_value := by sorry

end NUMINAMATH_CALUDE_f_properties_when_a_is_1_f_minimum_on_interval_l604_60497


namespace NUMINAMATH_CALUDE_james_two_semester_cost_l604_60420

/-- The cost of James's two semesters at community college -/
def two_semester_cost (units_per_semester : ℕ) (cost_per_unit : ℕ) : ℕ :=
  2 * units_per_semester * cost_per_unit

/-- Proof that James pays $2000 for two semesters -/
theorem james_two_semester_cost :
  two_semester_cost 20 50 = 2000 := by
  sorry

#eval two_semester_cost 20 50

end NUMINAMATH_CALUDE_james_two_semester_cost_l604_60420


namespace NUMINAMATH_CALUDE_right_triangle_special_property_l604_60444

theorem right_triangle_special_property :
  ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive sides
  (a^2 + b^2 = c^2) →        -- right triangle (Pythagorean theorem)
  ((1/2) * a * b = 24) →     -- area is 24
  (a^2 + b^2 = 2 * 24) →     -- sum of squares of legs equals twice the area
  (a = 2 * Real.sqrt 6 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_special_property_l604_60444


namespace NUMINAMATH_CALUDE_even_weeks_count_l604_60452

/-- Represents a day in a month --/
structure Day where
  number : ℕ
  month : ℕ
  deriving Repr

/-- Represents a week in a calendar --/
structure Week where
  days : List Day
  deriving Repr

/-- Determines if a week is even based on the sum of its day numbers --/
def isEvenWeek (w : Week) : Bool :=
  (w.days.map (λ d => d.number)).sum % 2 == 0

/-- Generates the 52 weeks starting from the first Monday of January --/
def generateWeeks : List Week :=
  sorry

/-- Counts the number of even weeks in a list of weeks --/
def countEvenWeeks (weeks : List Week) : ℕ :=
  (weeks.filter isEvenWeek).length

/-- Theorem stating that the number of even weeks in the 52-week period is 30 --/
theorem even_weeks_count :
  countEvenWeeks generateWeeks = 30 := by
  sorry

end NUMINAMATH_CALUDE_even_weeks_count_l604_60452


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sixth_root_64_l604_60467

theorem cube_root_125_times_fourth_root_256_times_sixth_root_64 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sixth_root_64_l604_60467


namespace NUMINAMATH_CALUDE_function_properties_l604_60436

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x + b

theorem function_properties (a b : ℝ) :
  (∀ x y : ℝ, x = 1 → y = f a b x → (3 * x - y - 3 = 0) → (a = -2 ∧ b = -1/2)) ∧
  ((∀ x : ℝ, x ≠ 0 → (deriv (f a b) x = 0 ↔ x = 1)) → a = 1) ∧
  ((-2 ≤ a ∧ a < 0) →
    (∃ m : ℝ, m = 12 ∧
      (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ ≤ 2 ∧ 0 < x₂ ∧ x₂ ≤ 2 →
        |f a b x₁ - f a b x₂| ≤ m * |1/x₁ - 1/x₂|) ∧
      (∀ m' : ℝ, m' < m →
        ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ ≤ 2 ∧ 0 < x₂ ∧ x₂ ≤ 2 ∧
          |f a b x₁ - f a b x₂| > m' * |1/x₁ - 1/x₂|))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l604_60436


namespace NUMINAMATH_CALUDE_five_person_circle_greetings_l604_60410

/-- Represents a circular arrangement of people --/
structure CircularArrangement (n : ℕ) where
  people : Fin n

/-- Number of greetings in a circular arrangement --/
def greetings (c : CircularArrangement 5) : ℕ := sorry

theorem five_person_circle_greetings :
  ∀ c : CircularArrangement 5, greetings c = 5 := by sorry

end NUMINAMATH_CALUDE_five_person_circle_greetings_l604_60410


namespace NUMINAMATH_CALUDE_point_transformation_l604_60418

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The theorem stating that if P is in the second quadrant, then P' is in the third quadrant -/
theorem point_transformation (m n : ℝ) :
  let P : Point := ⟨m, n⟩
  let P' : Point := ⟨-m^2, -n⟩
  SecondQuadrant P → ThirdQuadrant P' := by
  sorry


end NUMINAMATH_CALUDE_point_transformation_l604_60418


namespace NUMINAMATH_CALUDE_jake_bitcoin_factor_l604_60413

theorem jake_bitcoin_factor (initial_fortune : ℕ) (first_donation : ℕ) (second_donation : ℕ) (final_amount : ℕ) :
  initial_fortune = 80 ∧
  first_donation = 20 ∧
  second_donation = 10 ∧
  final_amount = 80 →
  ∃ (factor : ℚ), 
    factor = 3 ∧
    final_amount = (((initial_fortune - first_donation) / 2) * factor).floor - second_donation :=
by sorry

end NUMINAMATH_CALUDE_jake_bitcoin_factor_l604_60413


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l604_60404

theorem a_is_perfect_square (a b : ℤ) (h : a = a^2 + b^2 - 8*b - 2*a*b + 16) :
  ∃ k : ℤ, a = k^2 := by sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l604_60404


namespace NUMINAMATH_CALUDE_inverse_f_negative_three_l604_60421

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_negative_three :
  (Function.invFun f) (-3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_negative_three_l604_60421


namespace NUMINAMATH_CALUDE_list_price_is_40_l604_60469

/-- The list price of the item. -/
def list_price : ℝ := 40

/-- Alice's selling price. -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price. -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate. -/
def alice_rate : ℝ := 0.15

/-- Bob's commission rate. -/
def bob_rate : ℝ := 0.25

/-- Alice's commission. -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission. -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem list_price_is_40 :
  alice_commission list_price = bob_commission list_price ∧
  list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_list_price_is_40_l604_60469


namespace NUMINAMATH_CALUDE_bowls_lost_l604_60433

/-- Proves that the number of lost bowls is 26 given the problem conditions --/
theorem bowls_lost (total_bowls : ℕ) (fee : ℕ) (safe_payment : ℕ) (penalty : ℕ) 
  (broken_bowls : ℕ) (total_payment : ℕ) :
  total_bowls = 638 →
  fee = 100 →
  safe_payment = 3 →
  penalty = 4 →
  broken_bowls = 15 →
  total_payment = 1825 →
  ∃ (lost_bowls : ℕ), 
    fee + safe_payment * (total_bowls - lost_bowls - broken_bowls) - 
    penalty * (lost_bowls + broken_bowls) = total_payment ∧
    lost_bowls = 26 :=
by sorry

end NUMINAMATH_CALUDE_bowls_lost_l604_60433
