import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l867_86709

theorem equation_solution : ∀ x : ℝ, x + 36 / (x - 3) = -9 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l867_86709


namespace NUMINAMATH_CALUDE_rectangle_y_value_l867_86728

/-- Given a rectangle with vertices at (-1, y), (7, y), (-1, 3), and (7, 3),
    where y is positive and the area is 72 square units, y must equal 12. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : 8 * (y - 3) = 72) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l867_86728


namespace NUMINAMATH_CALUDE_linear_function_proof_l867_86790

def f (x : ℝ) := -3 * x + 5

theorem linear_function_proof :
  (∀ x y : ℝ, f y - f x = -3 * (y - x)) ∧ 
  (∃ y : ℝ, f 0 = 3 * 0 + 5 ∧ f 0 = y) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l867_86790


namespace NUMINAMATH_CALUDE_quadrilateral_diagonals_l867_86765

-- Define a convex quadrilateral
structure ConvexQuadrilateral :=
  (perimeter : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)
  (is_convex : perimeter > 0 ∧ diagonal1 > 0 ∧ diagonal2 > 0)

-- Theorem statement
theorem quadrilateral_diagonals 
  (q : ConvexQuadrilateral) 
  (h1 : q.perimeter = 2004) 
  (h2 : q.diagonal1 = 1001) : 
  (q.diagonal2 ≠ 1) ∧ 
  (∃ q' : ConvexQuadrilateral, q'.perimeter = 2004 ∧ q'.diagonal1 = 1001 ∧ q'.diagonal2 = 2) ∧
  (∃ q'' : ConvexQuadrilateral, q''.perimeter = 2004 ∧ q''.diagonal1 = 1001 ∧ q''.diagonal2 = 1001) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonals_l867_86765


namespace NUMINAMATH_CALUDE_madeline_leisure_hours_l867_86772

def total_hours_in_week : ℕ := 24 * 7

def class_hours : ℕ := 18
def homework_hours : ℕ := 4 * 7
def extracurricular_hours : ℕ := 3 * 3
def tutoring_hours : ℕ := 1 * 2
def work_hours : ℕ := 5 + 4 + 4 + 7
def sleep_hours : ℕ := 8 * 7

def total_scheduled_hours : ℕ := 
  class_hours + homework_hours + extracurricular_hours + tutoring_hours + work_hours + sleep_hours

theorem madeline_leisure_hours : 
  total_hours_in_week - total_scheduled_hours = 35 := by sorry

end NUMINAMATH_CALUDE_madeline_leisure_hours_l867_86772


namespace NUMINAMATH_CALUDE_james_pays_37_50_l867_86700

/-- Calculates the amount James pays for singing lessons given the specified conditions. -/
def james_payment (total_lessons : ℕ) (lesson_cost : ℚ) (free_lessons : ℕ) (fully_paid_lessons : ℕ) (uncle_contribution : ℚ) : ℚ :=
  let remaining_lessons := total_lessons - free_lessons - fully_paid_lessons
  let partially_paid_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := fully_paid_lessons + partially_paid_lessons
  let total_cost := total_paid_lessons * lesson_cost
  total_cost * (1 - uncle_contribution)

/-- Theorem stating that James pays $37.50 for his singing lessons. -/
theorem james_pays_37_50 :
  james_payment 20 5 1 10 (1/2) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_james_pays_37_50_l867_86700


namespace NUMINAMATH_CALUDE_inequality_minimum_l867_86784

theorem inequality_minimum (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_minimum_l867_86784


namespace NUMINAMATH_CALUDE_bobby_shoes_count_l867_86774

/-- Given information about the number of shoes owned by Bonny, Becky, and Bobby, 
    prove that Bobby has 27 pairs of shoes. -/
theorem bobby_shoes_count : 
  ∀ (becky_shoes : ℕ), 
  (13 = 2 * becky_shoes - 5) →  -- Bonny's shoes are 5 less than twice Becky's
  (27 = 3 * becky_shoes) -- Bobby has 3 times as many shoes as Becky
  := by sorry

end NUMINAMATH_CALUDE_bobby_shoes_count_l867_86774


namespace NUMINAMATH_CALUDE_usable_area_formula_l867_86701

/-- The usable area of a rectangular field with flooded region -/
def usableArea (x : ℝ) : ℝ :=
  (x + 9) * (x + 7) - (2 * x - 2) * (x - 1)

/-- Theorem stating the usable area of the field -/
theorem usable_area_formula (x : ℝ) : 
  usableArea x = -x^2 + 20*x + 61 := by
  sorry

end NUMINAMATH_CALUDE_usable_area_formula_l867_86701


namespace NUMINAMATH_CALUDE_return_flight_is_98_minutes_l867_86739

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  outbound_time : ℝ
  total_time : ℝ
  still_air_difference : ℝ

/-- Calculates the return flight time given a flight scenario --/
def return_flight_time (scenario : FlightScenario) : ℝ :=
  scenario.total_time - scenario.outbound_time

/-- Theorem stating that the return flight time is 98 minutes --/
theorem return_flight_is_98_minutes (scenario : FlightScenario) 
  (h1 : scenario.outbound_time = 120)
  (h2 : scenario.total_time = 222)
  (h3 : scenario.still_air_difference = 6) :
  return_flight_time scenario = 98 := by
  sorry

#eval return_flight_time { outbound_time := 120, total_time := 222, still_air_difference := 6 }

end NUMINAMATH_CALUDE_return_flight_is_98_minutes_l867_86739


namespace NUMINAMATH_CALUDE_range_of_m_l867_86766

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2*x - 3 > 0) → (x < m - 1 ∨ x > m + 1)) ∧ 
  (∃ x : ℝ, (x < m - 1 ∨ x > m + 1) ∧ ¬(x^2 - 2*x - 3 > 0)) →
  0 ≤ m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l867_86766


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l867_86736

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 - Complex.I)
  Complex.re z = Complex.im z → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l867_86736


namespace NUMINAMATH_CALUDE_tangent_line_at_half_l867_86782

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x - 1)

theorem tangent_line_at_half (x y : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → 
    |(f (1/2 + h) - f (1/2)) / h - 2| < ε) →
  (y = 2 * x ↔ y - f (1/2) = 2 * (x - 1/2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_half_l867_86782


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l867_86715

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : x ≠ y
  /-- The perimeter is 20 -/
  perimeter : x + x + y = 20

/-- Properties of the isosceles triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) :
  (t.y = -2 * t.x + 20) ∧ (5 < t.x ∧ t.x < 10) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l867_86715


namespace NUMINAMATH_CALUDE_five_candies_three_kids_l867_86763

/-- The number of ways to distribute n candies among k kids, with each kid getting at least one candy -/
def distribute_candies (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 6 ways to distribute 5 candies among 3 kids, with each kid getting at least one candy -/
theorem five_candies_three_kids : distribute_candies 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_candies_three_kids_l867_86763


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l867_86775

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x ≤ y → f y ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l867_86775


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l867_86796

/-- The first line bounding the triangle -/
def line1 (x y : ℝ) : Prop := y - 2*x = 4

/-- The second line bounding the triangle -/
def line2 (x y : ℝ) : Prop := 2*y - x = 6

/-- The x-axis -/
def x_axis (y : ℝ) : Prop := y = 0

/-- A point is in the triangle if it satisfies the equations of both lines and is above or on the x-axis -/
def in_triangle (x y : ℝ) : Prop :=
  line1 x y ∧ line2 x y ∧ y ≥ 0

/-- The area of the triangle -/
noncomputable def triangle_area : ℝ := 2

theorem triangle_area_is_two :
  triangle_area = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l867_86796


namespace NUMINAMATH_CALUDE_permutation_remainder_l867_86793

/-- The number of characters in the string -/
def string_length : ℕ := 16

/-- The number of A's in the string -/
def count_A : ℕ := 4

/-- The number of B's in the string -/
def count_B : ℕ := 5

/-- The number of C's in the string -/
def count_C : ℕ := 5

/-- The number of D's in the string -/
def count_D : ℕ := 2

/-- The length of the first segment -/
def first_segment : ℕ := 5

/-- The length of the second segment -/
def second_segment : ℕ := 5

/-- The length of the third segment -/
def third_segment : ℕ := 6

/-- The function to calculate the number of valid permutations -/
def count_permutations : ℕ := sorry

theorem permutation_remainder :
  count_permutations % 1000 = 540 := by sorry

end NUMINAMATH_CALUDE_permutation_remainder_l867_86793


namespace NUMINAMATH_CALUDE_sum_of_cubes_l867_86768

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l867_86768


namespace NUMINAMATH_CALUDE_power_of_512_l867_86780

theorem power_of_512 : (512 : ℝ) ^ (4/3) = 4096 := by sorry

end NUMINAMATH_CALUDE_power_of_512_l867_86780


namespace NUMINAMATH_CALUDE_factor_expression_l867_86786

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) := by sorry

end NUMINAMATH_CALUDE_factor_expression_l867_86786


namespace NUMINAMATH_CALUDE_input_is_input_statement_l867_86788

-- Define the type for programming language statements
inductive Statement
  | Print
  | Input
  | If
  | Let

-- Define properties for different types of statements
def isPrintStatement (s : Statement) : Prop :=
  s = Statement.Print

def isInputStatement (s : Statement) : Prop :=
  s = Statement.Input

def isConditionalStatement (s : Statement) : Prop :=
  s = Statement.If

theorem input_is_input_statement :
  isPrintStatement Statement.Print →
  isInputStatement Statement.Input →
  isConditionalStatement Statement.If →
  isInputStatement Statement.Input :=
by
  sorry

end NUMINAMATH_CALUDE_input_is_input_statement_l867_86788


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l867_86714

theorem smallest_addition_for_divisibility : 
  ∃! x : ℕ, x < 169 ∧ (2714 + x) % 169 = 0 ∧ ∀ y : ℕ, y < x → (2714 + y) % 169 ≠ 0 :=
by
  use 119
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l867_86714


namespace NUMINAMATH_CALUDE_all_props_true_l867_86724

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 - 3*x + 2 ≠ 0 → x ≠ 1 ∧ x ≠ 2

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2 → x^2 - 3*x + 2 ≠ 0

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0

-- Theorem stating that all propositions are true
theorem all_props_true : 
  (∀ x : ℝ, original_prop x) ∧ 
  (∀ x : ℝ, converse_prop x) ∧ 
  (∀ x : ℝ, inverse_prop x) ∧ 
  (∀ x : ℝ, contrapositive_prop x) :=
sorry

end NUMINAMATH_CALUDE_all_props_true_l867_86724


namespace NUMINAMATH_CALUDE_min_value_of_f_max_value_of_sum_squares_max_value_is_tight_l867_86730

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 3|

-- Theorem for the minimum value of f
theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 5/2 := by sorry

-- Theorem for the maximum value of a² + 2b² + 3c²
theorem max_value_of_sum_squares (a b c : ℝ) (h : a^4 + b^4 + c^4 = 5/2) :
  a^2 + 2*b^2 + 3*c^2 ≤ Real.sqrt (35/2) := by sorry

-- Theorem to show that the upper bound is tight
theorem max_value_is_tight :
  ∃ (a b c : ℝ), a^4 + b^4 + c^4 = 5/2 ∧ a^2 + 2*b^2 + 3*c^2 = Real.sqrt (35/2) := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_max_value_of_sum_squares_max_value_is_tight_l867_86730


namespace NUMINAMATH_CALUDE_min_a_value_l867_86795

theorem min_a_value (a b : ℝ) : 
  (∀ x : ℝ, 3 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) ≤ 3) →
  (∀ c d : ℝ, (∀ x : ℝ, 3 * c * (Real.sin x + Real.cos x) + 2 * d * Real.sin (2 * x) ≤ 3) → 
    a + b ≤ c + d) →
  a = -4/5 := by sorry

end NUMINAMATH_CALUDE_min_a_value_l867_86795


namespace NUMINAMATH_CALUDE_divisibility_theorem_l867_86746

def C (s : ℕ) : ℕ := s * (s + 1)

def product_C (m k n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (C (m + i + 1) - C k)) 1

def product_C_seq (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * C (i + 1)) 1

theorem divisibility_theorem (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0)
  (h4 : Nat.Prime (m + k + 1)) (h5 : m + k + 1 > n + 1) :
  ∃ z : ℤ, (product_C m k n : ℤ) = z * (product_C_seq n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l867_86746


namespace NUMINAMATH_CALUDE_power_of_64_l867_86755

theorem power_of_64 : (64 : ℝ) ^ (5/6 : ℝ) = 32 :=
by
  have h : (64 : ℝ) = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l867_86755


namespace NUMINAMATH_CALUDE_boot_shoe_price_difference_l867_86760

-- Define the price of shoes and boots as real numbers
variable (S B : ℝ)

-- Monday's sales equation
axiom monday_sales : 22 * S + 16 * B = 460

-- Tuesday's sales equation
axiom tuesday_sales : 8 * S + 32 * B = 560

-- Theorem stating the price difference between boots and shoes
theorem boot_shoe_price_difference : B - S = 5 := by sorry

end NUMINAMATH_CALUDE_boot_shoe_price_difference_l867_86760


namespace NUMINAMATH_CALUDE_savings_calculation_l867_86787

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) 
  (h1 : income = 20000)
  (h2 : ratio_income = 4)
  (h3 : ratio_expenditure = 3) :
  income - (income * ratio_expenditure / ratio_income) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l867_86787


namespace NUMINAMATH_CALUDE_friends_decks_count_l867_86720

/-- The number of decks Victor's friend bought -/
def friends_decks : ℕ := 2

/-- The cost of each deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Victor bought -/
def victors_decks : ℕ := 6

/-- The total amount spent by both Victor and his friend in dollars -/
def total_spent : ℕ := 64

theorem friends_decks_count : 
  deck_cost * (victors_decks + friends_decks) = total_spent := by sorry

end NUMINAMATH_CALUDE_friends_decks_count_l867_86720


namespace NUMINAMATH_CALUDE_bug_path_tiles_l867_86794

-- Define the rectangle's dimensions
def width : ℕ := 12
def length : ℕ := 20

-- Define the function to calculate the number of tiles visited
def tilesVisited (w l : ℕ) : ℕ := w + l - Nat.gcd w l

-- Theorem statement
theorem bug_path_tiles : tilesVisited width length = 28 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l867_86794


namespace NUMINAMATH_CALUDE_sin_cos_sum_18_12_l867_86732

theorem sin_cos_sum_18_12 : 
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_18_12_l867_86732


namespace NUMINAMATH_CALUDE_hoseok_persimmons_l867_86703

theorem hoseok_persimmons :
  ∀ (jungkook_persimmons hoseok_persimmons : ℕ),
    jungkook_persimmons = 25 →
    jungkook_persimmons = 3 * hoseok_persimmons + 4 →
    hoseok_persimmons = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_persimmons_l867_86703


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l867_86734

theorem divisibility_in_sequence (x : Fin 2020 → ℤ) :
  ∃ i j : Fin 2020, i ≠ j ∧ (x j - x i) % 2019 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l867_86734


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l867_86773

theorem cookie_jar_problem :
  ∃ (n c : ℕ),
    12 ≤ n ∧ n ≤ 36 ∧
    (n - 1) * c + (c + 1) = 1000 ∧
    n + (c + 1) = 65 :=
by sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l867_86773


namespace NUMINAMATH_CALUDE_triangle_sin_c_equals_one_l867_86731

theorem triangle_sin_c_equals_one 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : a = 1) 
  (h2 : b = Real.sqrt 3) 
  (h3 : A + C = 2 * B) 
  (h4 : 0 < A ∧ A < π) 
  (h5 : 0 < B ∧ B < π) 
  (h6 : 0 < C ∧ C < π) 
  (h7 : A + B + C = π) 
  (h8 : a / Real.sin A = b / Real.sin B) 
  (h9 : b / Real.sin B = c / Real.sin C) 
  : Real.sin C = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_c_equals_one_l867_86731


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l867_86717

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x < 0} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l867_86717


namespace NUMINAMATH_CALUDE_difference_of_squares_l867_86785

theorem difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l867_86785


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l867_86764

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.50) :
  let price_increase_ratio := new_price / initial_price
  let consumption_reduction_percentage := (1 - 1 / price_increase_ratio) * 100
  consumption_reduction_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l867_86764


namespace NUMINAMATH_CALUDE_inequality_proof_l867_86791

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l867_86791


namespace NUMINAMATH_CALUDE_assembly_line_theorem_l867_86744

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of tasks that can be freely arranged -/
def free_tasks : ℕ := num_tasks - 1

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := factorial free_tasks

theorem assembly_line_theorem : assembly_line_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_theorem_l867_86744


namespace NUMINAMATH_CALUDE_order_of_expressions_l867_86737

theorem order_of_expressions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  let a := Real.sqrt ((x^2 + y^2) / 2) - (x + y) / 2
  let b := (x + y) / 2 - Real.sqrt (x * y)
  let c := Real.sqrt (x * y) - 2 / (1 / x + 1 / y)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l867_86737


namespace NUMINAMATH_CALUDE_drill_bits_purchase_cost_l867_86711

/-- The total cost of a purchase with tax -/
def total_cost (num_sets : ℕ) (cost_per_set : ℝ) (tax_rate : ℝ) : ℝ :=
  let pre_tax_cost := num_sets * cost_per_set
  let tax_amount := pre_tax_cost * tax_rate
  pre_tax_cost + tax_amount

/-- Theorem stating the total cost for the specific purchase -/
theorem drill_bits_purchase_cost :
  total_cost 5 6 0.1 = 33 := by
  sorry

end NUMINAMATH_CALUDE_drill_bits_purchase_cost_l867_86711


namespace NUMINAMATH_CALUDE_f_value_at_neg_five_halves_l867_86722

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_neg_five_halves :
  (∀ x, f x = f (-x)) →                     -- f is even
  (∀ x, f (x + 2) = f x) →                  -- f has period 2
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2*x*(1 - x)) → -- f definition for 0 ≤ x ≤ 1
  f (-5/2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_neg_five_halves_l867_86722


namespace NUMINAMATH_CALUDE_lost_ship_depth_l867_86754

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def depth_of_lost_ship (descent_rate : ℝ) (time_taken : ℝ) : ℝ :=
  descent_rate * time_taken

/-- Theorem: The depth of the lost ship is 2400 feet below sea level. -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 30  -- feet per minute
  let time_taken : ℝ := 80    -- minutes
  depth_of_lost_ship descent_rate time_taken = 2400 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l867_86754


namespace NUMINAMATH_CALUDE_c_range_theorem_l867_86710

-- Define the rectangular prism
def rectangular_prism (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the condition a + b - c = 1
def sum_condition (a b c : ℝ) : Prop :=
  a + b - c = 1

-- Define the condition that the length of the diagonal is 1
def diagonal_condition (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 1

-- Define the condition a ≠ b
def not_equal_condition (a b : ℝ) : Prop :=
  a ≠ b

-- Theorem statement
theorem c_range_theorem (a b c : ℝ) :
  rectangular_prism a b c →
  sum_condition a b c →
  diagonal_condition a b c →
  not_equal_condition a b →
  0 < c ∧ c < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_c_range_theorem_l867_86710


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l867_86706

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l867_86706


namespace NUMINAMATH_CALUDE_constant_width_max_length_l867_86762

/-- A convex curve in a 2D plane. -/
structure ConvexCurve where
  -- Add necessary fields and conditions to define a convex curve
  is_convex : Bool
  diameter : ℝ
  length : ℝ

/-- A curve of constant width. -/
structure ConstantWidthCurve extends ConvexCurve where
  constant_width : ℝ
  is_constant_width : Bool

/-- The theorem stating that curves of constant width 1 have the greatest length among all convex curves of diameter 1. -/
theorem constant_width_max_length :
  ∀ (K : ConvexCurve),
    K.diameter = 1 →
    ∀ (C : ConstantWidthCurve),
      C.diameter = 1 →
      C.constant_width = 1 →
      C.is_constant_width →
      K.length ≤ C.length :=
sorry


end NUMINAMATH_CALUDE_constant_width_max_length_l867_86762


namespace NUMINAMATH_CALUDE_parabola_sine_no_intersection_l867_86797

theorem parabola_sine_no_intersection :
  ∀ x : ℝ, x^2 - x + 5.35 > 2 * Real.sin x + 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sine_no_intersection_l867_86797


namespace NUMINAMATH_CALUDE_binomial_zero_binomial_312_0_l867_86752

theorem binomial_zero (n : ℕ) : Nat.choose n 0 = 1 := by sorry

theorem binomial_312_0 : Nat.choose 312 0 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_zero_binomial_312_0_l867_86752


namespace NUMINAMATH_CALUDE_relative_error_comparison_l867_86740

theorem relative_error_comparison :
  let line1_length : ℚ := 15
  let line1_error : ℚ := 3 / 100
  let line2_length : ℚ := 125
  let line2_error : ℚ := 1 / 4
  let relative_error1 : ℚ := line1_error / line1_length
  let relative_error2 : ℚ := line2_error / line2_length
  relative_error1 = relative_error2 :=
by sorry

end NUMINAMATH_CALUDE_relative_error_comparison_l867_86740


namespace NUMINAMATH_CALUDE_parabola_c_value_l867_86704

/-- A parabola in the xy-plane with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = 4 →  -- vertex at (4, 1)
  p.x_coord 3 = -2 →  -- passes through (-2, 3)
  p.c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l867_86704


namespace NUMINAMATH_CALUDE_second_red_ball_most_likely_l867_86738

/-- The total number of balls in the urn -/
def total_balls : ℕ := 101

/-- The number of red balls in the urn -/
def red_balls : ℕ := 3

/-- The probability of drawing the second red ball on the kth draw -/
def prob_second_red (k : ℕ) : ℚ :=
  if 1 < k ∧ k < total_balls
  then (k - 1 : ℚ) * (total_balls - k : ℚ) / (total_balls.choose red_balls : ℚ)
  else 0

/-- The draw number that maximizes the probability of drawing the second red ball -/
def max_prob_draw : ℕ := 51

theorem second_red_ball_most_likely :
  ∀ k, prob_second_red max_prob_draw ≥ prob_second_red k :=
sorry

end NUMINAMATH_CALUDE_second_red_ball_most_likely_l867_86738


namespace NUMINAMATH_CALUDE_rational_function_sum_l867_86729

/-- Given rational functions r and s, prove r(x) + s(x) = -x^3 + 3x under specific conditions -/
theorem rational_function_sum (r s : ℝ → ℝ) : 
  (∃ (a b : ℝ), s x = a * (x - 2) * (x + 2) * x) →  -- s(x) is cubic with roots at 2, -2, and 0
  (∃ (b : ℝ), r x = b * x) →  -- r(x) is linear with a root at 0
  r (-1) = 1 →  -- condition on r
  s 1 = 3 →  -- condition on s
  ∀ x, r x + s x = -x^3 + 3*x := by
  sorry

end NUMINAMATH_CALUDE_rational_function_sum_l867_86729


namespace NUMINAMATH_CALUDE_inequality_proof_l867_86745

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a + b + c + d = 1) :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ 1 / 27 + 176 / 27 * a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l867_86745


namespace NUMINAMATH_CALUDE_fraction_sum_bound_l867_86789

theorem fraction_sum_bound (a b c : ℕ+) (h : (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ < 1) :
  (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ ≤ 41 / 42 ∧
  ∃ (x y z : ℕ+), (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ = 41 / 42 :=
sorry

#check fraction_sum_bound

end NUMINAMATH_CALUDE_fraction_sum_bound_l867_86789


namespace NUMINAMATH_CALUDE_total_profit_is_390_4_l867_86707

/-- Represents the partnership of A, B, and C -/
structure Partnership where
  a_share : Rat
  b_share : Rat
  c_share : Rat
  a_withdrawal_time : Nat
  a_withdrawal_fraction : Rat
  profit_distribution_time : Nat
  b_profit_share : Rat

/-- Calculates the total profit given the partnership conditions -/
def calculate_total_profit (p : Partnership) : Rat :=
  sorry

/-- Theorem stating that the total profit is 390.4 given the specified conditions -/
theorem total_profit_is_390_4 (p : Partnership) 
  (h1 : p.a_share = 1/2)
  (h2 : p.b_share = 1/3)
  (h3 : p.c_share = 1/4)
  (h4 : p.a_withdrawal_time = 2)
  (h5 : p.a_withdrawal_fraction = 1/2)
  (h6 : p.profit_distribution_time = 10)
  (h7 : p.b_profit_share = 144) :
  calculate_total_profit p = 390.4 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_390_4_l867_86707


namespace NUMINAMATH_CALUDE_min_value_on_circle_l867_86713

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 20 = 0

/-- A point on the circle -/
structure PointOnCircle where
  a : ℝ
  b : ℝ
  on_circle : circle_equation a b

/-- The theorem stating the minimum value of a^2 + b^2 for points on the circle -/
theorem min_value_on_circle :
  ∀ P : PointOnCircle, ∃ m : ℝ, 
    (∀ Q : PointOnCircle, m ≤ Q.a^2 + Q.b^2) ∧
    m = 30 - 10 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l867_86713


namespace NUMINAMATH_CALUDE_deck_restoration_l867_86798

/-- Represents a cut operation on a deck of cards -/
def cut (n : ℕ) (deck : List ℕ) : List ℕ := sorry

/-- Represents the composition of multiple cuts -/
def compose_cuts (cuts : List ℕ) (deck : List ℕ) : List ℕ := sorry

theorem deck_restoration (x : ℕ) :
  let deck := List.range 52
  let cuts := [28, 31, 2, x, 21]
  compose_cuts cuts deck = deck →
  x = 22 := by sorry

end NUMINAMATH_CALUDE_deck_restoration_l867_86798


namespace NUMINAMATH_CALUDE_negative_sqrt_13_less_than_negative_3_l867_86742

theorem negative_sqrt_13_less_than_negative_3 : -Real.sqrt 13 < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_13_less_than_negative_3_l867_86742


namespace NUMINAMATH_CALUDE_c_share_l867_86779

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The theorem stating C's share given the conditions -/
theorem c_share (s : Share) 
  (h1 : s.a / s.b = 5 / 3)
  (h2 : s.b / s.c = 3 / 2)
  (h3 : s.c / s.d = 2 / 3)
  (h4 : s.a = s.b + 1000) :
  s.c = 1000 := by
sorry

end NUMINAMATH_CALUDE_c_share_l867_86779


namespace NUMINAMATH_CALUDE_det_equals_xy_l867_86758

/-- The determinant of the matrix
    [1, x, y]
    [1, x+y, y]
    [1, x, x+y]
    is equal to xy -/
theorem det_equals_xy (x y : ℝ) : 
  Matrix.det !![1, x, y; 1, x+y, y; 1, x, x+y] = x * y := by
  sorry

end NUMINAMATH_CALUDE_det_equals_xy_l867_86758


namespace NUMINAMATH_CALUDE_function_divisibility_property_l867_86778

theorem function_divisibility_property (f : ℤ → ℤ) : 
  (∀ m n : ℤ, (Int.gcd m n : ℤ) ∣ (f m + f n)) → 
  ∃ k : ℤ, ∀ n : ℤ, f n = k * n :=
by sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l867_86778


namespace NUMINAMATH_CALUDE_special_right_triangle_median_property_l867_86753

/-- A right triangle with a special median property -/
structure SpecialRightTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- B is the right angle
  right_angle : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  -- BM is the median from B to AC
  M : ℝ × ℝ
  is_median : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  -- The special property BM² = AB·BC
  special_property : 
    ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 
    (((A.1 - B.1)^2 + (A.2 - B.2)^2) * ((C.1 - B.1)^2 + (C.2 - B.2)^2))^(1/2)

/-- Theorem: In a SpecialRightTriangle, BM = 1/2 AC -/
theorem special_right_triangle_median_property (t : SpecialRightTriangle) :
  ((t.M.1 - t.B.1)^2 + (t.M.2 - t.B.2)^2) = 
  (1/4) * ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_median_property_l867_86753


namespace NUMINAMATH_CALUDE_intersection_and_complement_l867_86761

def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x | x + 3 ≥ 0}

theorem intersection_and_complement :
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (Set.compl (A ∩ B) = {x | x < -3 ∨ x > -2}) := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_l867_86761


namespace NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l867_86735

/-- A cubic polynomial Q with specific values at 0, 1, and -1 -/
structure CubicPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c d : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + d
  value_at_zero : Q 0 = m
  value_at_one : Q 1 = 3 * m
  value_at_neg_one : Q (-1) = 4 * m

/-- The sum of the polynomial values at 2 and -2 is 22m -/
theorem sum_at_two_and_neg_two (m : ℝ) (Q : CubicPolynomial m) :
  Q.Q 2 + Q.Q (-2) = 22 * m := by
  sorry

end NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l867_86735


namespace NUMINAMATH_CALUDE_sequence_decreases_eventually_l867_86751

def a (n : ℕ) : ℚ := (100 : ℚ) ^ n / n.factorial

theorem sequence_decreases_eventually :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 1) ≤ a n := by sorry

end NUMINAMATH_CALUDE_sequence_decreases_eventually_l867_86751


namespace NUMINAMATH_CALUDE_inequality_proof_l867_86769

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a) / (a^2 + b * c) + (2 * b) / (b^2 + c * a) + (2 * c) / (c^2 + a * b) ≤ 
  a / (b * c) + b / (c * a) + c / (a * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l867_86769


namespace NUMINAMATH_CALUDE_total_cakes_is_fifteen_l867_86771

/-- The number of cakes served during lunch -/
def lunch_cakes : ℕ := 6

/-- The number of cakes served during dinner -/
def dinner_cakes : ℕ := 9

/-- The total number of cakes served today -/
def total_cakes : ℕ := lunch_cakes + dinner_cakes

/-- Proof that the total number of cakes served today is 15 -/
theorem total_cakes_is_fifteen : total_cakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_is_fifteen_l867_86771


namespace NUMINAMATH_CALUDE_hilary_jar_big_toenails_l867_86770

/-- Represents the capacity and contents of a toenail jar. -/
structure ToenailJar where
  regularCapacity : ℕ  -- Total capacity in terms of regular toenails
  bigSize : ℕ  -- Size of a big toenail relative to a regular toenail
  regularCount : ℕ  -- Number of regular toenails already in the jar
  remainingRegularSpace : ℕ  -- Number of additional regular toenails that can fit

/-- Calculates the number of big toenails in the jar. -/
def bigToenailCount (jar : ToenailJar) : ℕ :=
  (jar.regularCapacity - (jar.regularCount + jar.remainingRegularSpace)) / jar.bigSize

/-- Theorem stating the number of big toenails in Hilary's jar. -/
theorem hilary_jar_big_toenails :
  let jar : ToenailJar := {
    regularCapacity := 100,
    bigSize := 2,
    regularCount := 40,
    remainingRegularSpace := 20
  }
  bigToenailCount jar = 10 := by
  sorry


end NUMINAMATH_CALUDE_hilary_jar_big_toenails_l867_86770


namespace NUMINAMATH_CALUDE_happy_formations_correct_l867_86716

def happy_formations (n : ℕ) : ℕ :=
  if n % 3 = 1 then 0
  else if n % 3 = 0 then
    (Nat.choose (n-1) (2*n/3) - Nat.choose (n-1) ((2*n+6)/3))^3 +
    (Nat.choose (n-1) ((2*n-3)/3) - Nat.choose (n-1) ((2*n+3)/3))^3
  else
    (Nat.choose (n-1) ((2*n-1)/3) - Nat.choose (n-1) ((2*n+1)/3))^3 +
    (Nat.choose (n-1) ((2*n-4)/3) - Nat.choose (n-1) ((2*n-2)/3))^3

theorem happy_formations_correct (n : ℕ) :
  happy_formations n =
    if n % 3 = 1 then 0
    else if n % 3 = 0 then
      (Nat.choose (n-1) (2*n/3) - Nat.choose (n-1) ((2*n+6)/3))^3 +
      (Nat.choose (n-1) ((2*n-3)/3) - Nat.choose (n-1) ((2*n+3)/3))^3
    else
      (Nat.choose (n-1) ((2*n-1)/3) - Nat.choose (n-1) ((2*n+1)/3))^3 +
      (Nat.choose (n-1) ((2*n-4)/3) - Nat.choose (n-1) ((2*n-2)/3))^3 :=
by sorry

end NUMINAMATH_CALUDE_happy_formations_correct_l867_86716


namespace NUMINAMATH_CALUDE_ripe_fruits_weight_l867_86725

/-- Given the following conditions:
    - Total fruits: 14 apples, 10 pears, 5 lemons
    - Average weights of ripe fruits: apples 150g, pears 200g, lemons 100g
    - Average weights of unripe fruits: apples 120g, pears 180g, lemons 80g
    - Unripe fruits: 6 apples, 4 pears, 2 lemons
    Prove that the total weight of ripe fruits is 2700 grams -/
theorem ripe_fruits_weight (
  total_apples : ℕ) (total_pears : ℕ) (total_lemons : ℕ)
  (ripe_apple_weight : ℕ) (ripe_pear_weight : ℕ) (ripe_lemon_weight : ℕ)
  (unripe_apple_weight : ℕ) (unripe_pear_weight : ℕ) (unripe_lemon_weight : ℕ)
  (unripe_apples : ℕ) (unripe_pears : ℕ) (unripe_lemons : ℕ)
  (h1 : total_apples = 14)
  (h2 : total_pears = 10)
  (h3 : total_lemons = 5)
  (h4 : ripe_apple_weight = 150)
  (h5 : ripe_pear_weight = 200)
  (h6 : ripe_lemon_weight = 100)
  (h7 : unripe_apple_weight = 120)
  (h8 : unripe_pear_weight = 180)
  (h9 : unripe_lemon_weight = 80)
  (h10 : unripe_apples = 6)
  (h11 : unripe_pears = 4)
  (h12 : unripe_lemons = 2) :
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ripe_fruits_weight_l867_86725


namespace NUMINAMATH_CALUDE_inequality_solution_l867_86718

theorem inequality_solution (x : ℝ) : 
  1 / (x^2 + 4) > 5 / x + 21 / 10 ↔ -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l867_86718


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l867_86748

/-- Represents the daily sales and profit scenario for a product -/
structure SalesScenario where
  initialSales : ℕ
  initialProfit : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ

/-- Calculates the daily profit for a given sales scenario -/
def dailyProfit (s : SalesScenario) : ℕ :=
  (s.initialSales + s.salesIncrease * s.priceReduction) * (s.initialProfit - s.priceReduction)

/-- Theorem: A price reduction of 25 yuan results in a daily profit of 2000 yuan -/
theorem optimal_price_reduction (s : SalesScenario) 
  (h1 : s.initialSales = 30)
  (h2 : s.initialProfit = 50)
  (h3 : s.salesIncrease = 2)
  (h4 : s.priceReduction = 25) :
  dailyProfit s = 2000 := by
  sorry

#eval dailyProfit { initialSales := 30, initialProfit := 50, salesIncrease := 2, priceReduction := 25 }

end NUMINAMATH_CALUDE_optimal_price_reduction_l867_86748


namespace NUMINAMATH_CALUDE_point_on_y_axis_l867_86702

/-- If point M (a+3, 2a-2) is on the y-axis, then its coordinates are (0, -8) -/
theorem point_on_y_axis (a : ℝ) : 
  (a + 3 = 0) → ((a + 3, 2*a - 2) : ℝ × ℝ) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l867_86702


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l867_86750

theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_time = 480 →
  let train_speed := train_length / signal_post_time
  let total_distance := train_speed * bridge_time
  let bridge_length := total_distance - train_length
  bridge_length = 6600 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l867_86750


namespace NUMINAMATH_CALUDE_cyclist_speed_l867_86759

/-- The cyclist's problem -/
theorem cyclist_speed (initial_time : ℝ) (faster_time : ℝ) (faster_speed : ℝ) :
  initial_time = 6 →
  faster_time = 3 →
  faster_speed = 14 →
  ∃ (distance : ℝ) (initial_speed : ℝ),
    distance = initial_speed * initial_time ∧
    distance = faster_speed * faster_time ∧
    initial_speed = 7 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_l867_86759


namespace NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l867_86723

theorem smallest_n_with_partial_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - 2*n + 1) % k = 0) ∧ 
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - 2*n + 1) % k ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - 2*m + 1) % k = 0) ∨ 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - 2*m + 1) % k ≠ 0)) ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l867_86723


namespace NUMINAMATH_CALUDE_parallelogram_properties_l867_86741

-- Define the parallelogram vertices as complex numbers
def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

-- Define the parallelogram
def parallelogram (A B C : ℂ) : Prop :=
  ∃ D : ℂ, (C - B) = (D - A) ∧ (D - C) = (B - A)

-- Theorem statement
theorem parallelogram_properties (h : parallelogram A B C) :
  ∃ D : ℂ,
    D = 4 + 3 * Complex.I ∧
    Complex.abs (C - A) = Real.sqrt 17 ∧
    Complex.abs (D - B) = Real.sqrt 18 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l867_86741


namespace NUMINAMATH_CALUDE_marys_next_birthday_age_l867_86721

theorem marys_next_birthday_age 
  (d : ℝ) -- Danielle's age
  (j : ℝ) -- John's age
  (s : ℝ) -- Sally's age
  (m : ℝ) -- Mary's age
  (h1 : j = 1.15 * d) -- John is 15% older than Danielle
  (h2 : s = 1.30 * d) -- Sally is 30% older than Danielle
  (h3 : m = 1.25 * s) -- Mary is 25% older than Sally
  (h4 : j + d + s + m = 80) -- Sum of ages is 80
  : Int.floor m + 1 = 26 := by
  sorry

#check marys_next_birthday_age

end NUMINAMATH_CALUDE_marys_next_birthday_age_l867_86721


namespace NUMINAMATH_CALUDE_becky_lollipops_l867_86712

theorem becky_lollipops (total_lollipops : ℕ) (num_friends : ℕ) (lemon : ℕ) (peppermint : ℕ) (watermelon : ℕ) (marshmallow : ℕ) :
  total_lollipops = lemon + peppermint + watermelon + marshmallow →
  total_lollipops = 795 →
  num_friends = 13 →
  total_lollipops % num_friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_becky_lollipops_l867_86712


namespace NUMINAMATH_CALUDE_heather_remaining_blocks_l867_86767

/-- The number of blocks Heather starts with -/
def initial_blocks : ℕ := 86

/-- The number of blocks Heather shares with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ends with -/
def remaining_blocks : ℕ := initial_blocks - shared_blocks

theorem heather_remaining_blocks : remaining_blocks = 45 := by
  sorry

end NUMINAMATH_CALUDE_heather_remaining_blocks_l867_86767


namespace NUMINAMATH_CALUDE_equation_proof_l867_86799

theorem equation_proof : 3889 + 12.808 - 47.806 = 3854.002 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l867_86799


namespace NUMINAMATH_CALUDE_platform_length_calculation_l867_86743

/-- Calculates the length of a platform given train and crossing information -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 470 →
  train_speed_kmh = 55 →
  crossing_time = 64.79481641468682 →
  ∃ (platform_length : ℝ), abs (platform_length - 520) < 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l867_86743


namespace NUMINAMATH_CALUDE_ab_inequality_l867_86727

theorem ab_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_inequality_l867_86727


namespace NUMINAMATH_CALUDE_total_chinese_hours_l867_86792

/-- The number of hours Ryan spends learning Chinese daily -/
def chinese_hours_per_day : ℕ := 4

/-- The number of days Ryan learns -/
def learning_days : ℕ := 6

/-- Theorem: The total hours Ryan spends learning Chinese over 6 days is 24 hours -/
theorem total_chinese_hours : chinese_hours_per_day * learning_days = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_chinese_hours_l867_86792


namespace NUMINAMATH_CALUDE_inequality_solution_set_l867_86708

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (1/3)*(m*x - 1) > 2 - m ↔ x < -4) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l867_86708


namespace NUMINAMATH_CALUDE_no_negative_roots_l867_86783

/-- Given f(x) = a^x + (x-2)/(x+1) where a > 1, prove that f(x) ≠ 0 for all x < 0 -/
theorem no_negative_roots (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, x < 0 → a^x + (x - 2) / (x + 1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l867_86783


namespace NUMINAMATH_CALUDE_existence_of_three_quadratic_polynomials_l867_86776

/-- A quadratic polynomial of the form ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b ^ 2 - 4 * p.a * p.c

/-- A quadratic polynomial has two distinct real roots if its discriminant is positive -/
def has_two_distinct_real_roots (p : QuadraticPolynomial) : Prop :=
  discriminant p > 0

/-- The sum of two quadratic polynomials -/
def sum_polynomials (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  { a := p.a + q.a, b := p.b + q.b, c := p.c + q.c }

/-- A quadratic polynomial has no real roots if its discriminant is negative -/
def has_no_real_roots (p : QuadraticPolynomial) : Prop :=
  discriminant p < 0

theorem existence_of_three_quadratic_polynomials :
  ∃ (p₁ p₂ p₃ : QuadraticPolynomial),
    (has_two_distinct_real_roots p₁) ∧
    (has_two_distinct_real_roots p₂) ∧
    (has_two_distinct_real_roots p₃) ∧
    (has_no_real_roots (sum_polynomials p₁ p₂)) ∧
    (has_no_real_roots (sum_polynomials p₁ p₃)) ∧
    (has_no_real_roots (sum_polynomials p₂ p₃)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_three_quadratic_polynomials_l867_86776


namespace NUMINAMATH_CALUDE_consecutive_points_segment_length_l867_86749

/-- Given 5 consecutive points on a straight line, if certain segment lengths are known,
    prove that the length of ac is 11. -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Representing points as real numbers on a line
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points
  (h_bc_cd : c - b = 2 * (d - c)) -- bc = 2cd
  (h_de : e - d = 8) -- de = 8
  (h_ab : b - a = 5) -- ab = 5
  (h_ae : e - a = 22) -- ae = 22
  : c - a = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_segment_length_l867_86749


namespace NUMINAMATH_CALUDE_opposite_sides_range_l867_86777

/-- Given two points A and B on opposite sides of a line, prove the range of y₀ -/
theorem opposite_sides_range (y₀ : ℝ) : 
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (1, y₀)
  let line (x y : ℝ) : ℝ := x - 2*y + 5
  (line A.1 A.2) * (line B.1 B.2) < 0 → y₀ > 3 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l867_86777


namespace NUMINAMATH_CALUDE_collinear_points_sum_l867_86719

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Collinearity of three points in 3D space -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t s : ℝ, t ≠ s ∧ 
    q.x = (1 - t) * p.x + t * r.x ∧
    q.y = (1 - t) * p.y + t * r.y ∧
    q.z = (1 - t) * p.z + t * r.z ∧
    q.x = (1 - s) * p.x + s * r.x ∧
    q.y = (1 - s) * p.y + s * r.y ∧
    q.z = (1 - s) * p.z + s * r.z

theorem collinear_points_sum (x y z : ℝ) :
  collinear (Point3D.mk x 1 z) (Point3D.mk 2 y z) (Point3D.mk x y 3) →
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l867_86719


namespace NUMINAMATH_CALUDE_min_baskets_needed_l867_86756

/-- Represents the number of points earned for scoring a basket -/
def points_per_basket : ℤ := 3

/-- Represents the number of points deducted for missing a basket -/
def points_per_miss : ℤ := 1

/-- Represents the total number of shots taken -/
def total_shots : ℕ := 12

/-- Represents the minimum score Xiao Li wants to achieve -/
def min_score : ℤ := 28

/-- Calculates the score based on the number of baskets made -/
def score (baskets_made : ℕ) : ℤ :=
  points_per_basket * baskets_made - points_per_miss * (total_shots - baskets_made)

/-- Proves that Xiao Li needs to make at least 10 baskets to score at least 28 points -/
theorem min_baskets_needed :
  ∀ baskets_made : ℕ, baskets_made ≤ total_shots →
    (∀ n : ℕ, n < baskets_made → score n < min_score) →
    score baskets_made ≥ min_score →
    baskets_made ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_min_baskets_needed_l867_86756


namespace NUMINAMATH_CALUDE_exchange_problem_l867_86781

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Represents the exchange scenario -/
def exchangeScenario (d : ℕ) : Prop :=
  (8 : ℚ) / 5 * d - 80 = d

theorem exchange_problem :
  ∃ d : ℕ, exchangeScenario d ∧ sumOfDigits d = 9 := by sorry

end NUMINAMATH_CALUDE_exchange_problem_l867_86781


namespace NUMINAMATH_CALUDE_coincident_foci_and_vertices_m_range_l867_86726

-- Define the ellipse equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (9 - m) + y^2 / (2 * m) = 1 ∧ 9 - m > 2 * m ∧ 2 * m > 0

-- Define the hyperbola equation and its eccentricity condition
def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  ∃ (e : ℝ), e > Real.sqrt 6 / 2 ∧ e < Real.sqrt 2 ∧
  e^2 = (5 + m) / 5

-- Theorem for part (I)
theorem coincident_foci_and_vertices (m : ℝ) 
  (h1 : is_ellipse m) (h2 : hyperbola_eccentricity_condition m) :
  (∃ (x : ℝ), x^2 / (9 - m) + 0^2 / (2 * m) = 1 ∧ 
              x^2 / 5 - 0^2 / m = 1) → m = 4 / 3 :=
sorry

-- Theorem for part (II)
theorem m_range (m : ℝ) 
  (h1 : is_ellipse m) (h2 : hyperbola_eccentricity_condition m) :
  5 / 2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_coincident_foci_and_vertices_m_range_l867_86726


namespace NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l867_86705

theorem no_perfect_square_in_sequence : ¬∃ (k n : ℕ), 3 * k - 1 = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l867_86705


namespace NUMINAMATH_CALUDE_intersection_x_coordinates_equal_l867_86747

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

/-- Definition of the right focus -/
def rightFocus (a b : ℝ) : Prop :=
  ellipse a b 1 (Real.sqrt 3 / 2)

/-- Definition of the perpendicular chord -/
def perpendicularChord (a b : ℝ) : Prop :=
  ∃ y₁ y₂, y₁ ≠ y₂ ∧ ellipse a b 0 y₁ ∧ ellipse a b 0 y₂ ∧ y₂ - y₁ = 1

/-- Definition of a point on the ellipse -/
def pointOnEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  ellipse a b x y

/-- Theorem statement -/
theorem intersection_x_coordinates_equal
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hrf : rightFocus a b)
  (hpc : perpendicularChord a b)
  (x₁ y₁ x₂ y₂ : ℝ)
  (hm : pointOnEllipse a b x₁ y₁)
  (hn : pointOnEllipse a b x₂ y₂)
  (hl : ∃ m, y₁ = m * (x₁ - 1) ∧ y₂ = m * (x₂ - 1)) :
  ∃ x_int, (∃ y_am, y_am = (y₁ / (x₁ + 1)) * (x_int + 1)) ∧
           (∃ y_bn, y_bn = (y₂ / (x₂ - 1)) * (x_int - 1)) :=
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinates_equal_l867_86747


namespace NUMINAMATH_CALUDE_simplify_expression_l867_86757

theorem simplify_expression (a b : ℝ) :
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l867_86757


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l867_86733

-- Define the hyperbola equation
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

-- Define the distance between foci
def foci_distance (d : ℝ) : Prop := d = 4

-- Define the range of n
def n_range (n : ℝ) : Prop := -1 < n ∧ n < 3

-- Theorem statement
theorem hyperbola_n_range (m n : ℝ) :
  is_hyperbola m n → foci_distance 4 → n_range n :=
sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l867_86733
