import Mathlib

namespace NUMINAMATH_CALUDE_pipe_length_difference_l3407_340768

theorem pipe_length_difference (total_length shorter_length : ℕ) : 
  total_length = 68 → 
  shorter_length = 28 → 
  shorter_length < total_length - shorter_length →
  total_length - shorter_length - shorter_length = 12 :=
by sorry

end NUMINAMATH_CALUDE_pipe_length_difference_l3407_340768


namespace NUMINAMATH_CALUDE_tangent_implies_one_point_one_point_not_always_tangent_l3407_340776

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a parabola in 2D space
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of a line being tangent to a parabola
def is_tangent (l : Line) (p : Parabola) : Prop := sorry

-- Define the property of a line and a parabola having exactly one common point
def has_one_common_point (l : Line) (p : Parabola) : Prop := sorry

-- Theorem stating the relationship between tangency and having one common point
theorem tangent_implies_one_point (l : Line) (p : Parabola) :
  is_tangent l p → has_one_common_point l p :=
sorry

-- Theorem stating that having one common point doesn't always imply tangency
theorem one_point_not_always_tangent :
  ∃ l : Line, ∃ p : Parabola, has_one_common_point l p ∧ ¬is_tangent l p :=
sorry

end NUMINAMATH_CALUDE_tangent_implies_one_point_one_point_not_always_tangent_l3407_340776


namespace NUMINAMATH_CALUDE_exponential_inequality_l3407_340710

theorem exponential_inequality (m n : ℝ) (h1 : m > n) (h2 : n > 0) : (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3407_340710


namespace NUMINAMATH_CALUDE_cos_four_theta_value_l3407_340758

theorem cos_four_theta_value (θ : ℝ) (h : ∑' n, (Real.cos θ)^(2*n) = 8) :
  Real.cos (4 * θ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_theta_value_l3407_340758


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_nine_and_n_plus_two_l3407_340720

theorem gcd_n_cube_plus_nine_and_n_plus_two (n : ℕ) (h : n > 2^3) :
  Nat.gcd (n^3 + 3^2) (n + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_nine_and_n_plus_two_l3407_340720


namespace NUMINAMATH_CALUDE_complex_sixth_power_l3407_340769

theorem complex_sixth_power (z : ℂ) : z = (-Real.sqrt 3 - I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_power_l3407_340769


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3407_340794

theorem rectangle_area_theorem (L W : ℝ) (h : 2 * L * (3 * W) = 1800) : L * W = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3407_340794


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3407_340735

theorem boys_to_girls_ratio 
  (T : ℕ) -- Total number of students
  (B : ℕ) -- Number of boys
  (G : ℕ) -- Number of girls
  (h1 : T = B + G) -- Total is sum of boys and girls
  (h2 : 2 * G = 3 * (T / 4)) -- 2/3 of girls = 1/4 of total
  : B * 3 = G * 5 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3407_340735


namespace NUMINAMATH_CALUDE_mikes_books_l3407_340712

theorem mikes_books (tim_books : ℕ) (total_books : ℕ) (h1 : tim_books = 22) (h2 : total_books = 42) :
  total_books - tim_books = 20 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l3407_340712


namespace NUMINAMATH_CALUDE_circle_symmetry_l3407_340732

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetric point
def symmetric_point (x y x' y' : ℝ) : Prop := x' = -y ∧ y' = -x

-- Theorem statement
theorem circle_symmetry (x y : ℝ) :
  (∃ (x' y' : ℝ), symmetric_point x y x' y' ∧ 
   symmetry_line x y ∧ 
   original_circle x' y') →
  x^2 + (y + 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3407_340732


namespace NUMINAMATH_CALUDE_part1_part2_l3407_340714

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

/-- Theorem for part 1 -/
theorem part1 :
  ∃ (k : ℝ), k = -1/2 ∧ collinear ((k * a.1 - b.1, k * a.2 - b.2)) (a.1 + 2 * b.1, a.2 + 2 * b.2) :=
sorry

/-- Theorem for part 2 -/
theorem part2 :
  ∃ (m : ℝ), m = 3/2 ∧
  (∃ (t : ℝ), (2 * a.1 + 3 * b.1, 2 * a.2 + 3 * b.2) = (t * (a.1 + m * b.1), t * (a.2 + m * b.2))) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3407_340714


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l3407_340783

/-- Given a group of 10 persons, prove that if replacing one person with a new person
    weighing 110 kg increases the average weight by 4 kg, then the weight of the
    replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (initial_avg : ℝ)
  (h1 : initial_avg > 0)
  (h2 : (10 * (initial_avg + 4) - 10 * initial_avg) = (110 - 70)) :
  70 = 110 - (10 * (initial_avg + 4) - 10 * initial_avg) :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l3407_340783


namespace NUMINAMATH_CALUDE_cubic_function_parallel_tangents_l3407_340722

/-- Given a cubic function f(x) = x³ + ax + b where a ≠ b, and the tangent lines
    to the graph of f at x=a and x=b are parallel, prove that f(1) = 1. -/
theorem cubic_function_parallel_tangents (a b : ℝ) (h : a ≠ b) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x + b
  (∃ k : ℝ, (3*a^2 + a = k) ∧ (3*b^2 + a = k)) → f 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_parallel_tangents_l3407_340722


namespace NUMINAMATH_CALUDE_bookkeeper_arrangements_l3407_340752

/-- The number of distinct arrangements of letters in a word with the given letter distribution -/
def distinctArrangements (totalLetters : Nat) (repeatedLetters : Nat) (repeatCount : Nat) : Nat :=
  Nat.factorial totalLetters / (Nat.factorial repeatCount ^ repeatedLetters)

/-- Theorem stating the number of distinct arrangements for the specific word structure -/
theorem bookkeeper_arrangements :
  distinctArrangements 10 4 2 = 226800 := by
  sorry

end NUMINAMATH_CALUDE_bookkeeper_arrangements_l3407_340752


namespace NUMINAMATH_CALUDE_three_thousandths_decimal_l3407_340782

theorem three_thousandths_decimal : (3 : ℚ) / 1000 = (0.003 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_three_thousandths_decimal_l3407_340782


namespace NUMINAMATH_CALUDE_tan_product_eighths_of_pi_l3407_340744

theorem tan_product_eighths_of_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) * Real.tan (7 * π / 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_of_pi_l3407_340744


namespace NUMINAMATH_CALUDE_train_meeting_time_l3407_340771

/-- Calculates the time for two trains to meet given their speeds, lengths, and the platform length --/
theorem train_meeting_time (length_A length_B platform_length : ℝ)
                           (speed_A speed_B : ℝ)
                           (h1 : length_A = 120)
                           (h2 : length_B = 150)
                           (h3 : platform_length = 180)
                           (h4 : speed_A = 90 * 1000 / 3600)
                           (h5 : speed_B = 72 * 1000 / 3600) :
  (length_A + length_B + platform_length) / (speed_A + speed_B) = 10 := by
  sorry

#check train_meeting_time

end NUMINAMATH_CALUDE_train_meeting_time_l3407_340771


namespace NUMINAMATH_CALUDE_michael_anna_ratio_is_500_251_l3407_340766

/-- Sum of odd integers from 1 to n -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

/-- Sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The ratio of Michael's sum to Anna's sum -/
def michaelAnnaRatio : ℚ :=
  (sumOddIntegers 500 : ℚ) / (sumIntegers 500 : ℚ)

theorem michael_anna_ratio_is_500_251 :
  michaelAnnaRatio = 500 / 251 := by
  sorry

end NUMINAMATH_CALUDE_michael_anna_ratio_is_500_251_l3407_340766


namespace NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l3407_340778

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_three_digit_non_divisor : 
  ∀ n : ℕ, is_three_digit n → (n - 1) ∣ factorial n → n ≥ 1004 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l3407_340778


namespace NUMINAMATH_CALUDE_defective_product_probability_l3407_340764

theorem defective_product_probability
  (p_first : ℝ)
  (p_second : ℝ)
  (h1 : p_first = 0.65)
  (h2 : p_second = 0.3)
  (h3 : p_first + p_second + p_defective = 1)
  (h4 : p_first ≥ 0 ∧ p_second ≥ 0 ∧ p_defective ≥ 0)
  : p_defective = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_defective_product_probability_l3407_340764


namespace NUMINAMATH_CALUDE_sparkling_juice_bottles_l3407_340779

def total_guests : ℕ := 120
def champagne_percentage : ℚ := 60 / 100
def wine_percentage : ℚ := 30 / 100
def juice_percentage : ℚ := 10 / 100

def champagne_glasses_per_guest : ℕ := 2
def wine_glasses_per_guest : ℕ := 1
def juice_glasses_per_guest : ℕ := 1

def champagne_servings_per_bottle : ℕ := 6
def wine_servings_per_bottle : ℕ := 5
def juice_servings_per_bottle : ℕ := 4

theorem sparkling_juice_bottles (
  total_guests : ℕ)
  (juice_percentage : ℚ)
  (juice_glasses_per_guest : ℕ)
  (juice_servings_per_bottle : ℕ)
  (h1 : total_guests = 120)
  (h2 : juice_percentage = 10 / 100)
  (h3 : juice_glasses_per_guest = 1)
  (h4 : juice_servings_per_bottle = 4)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_sparkling_juice_bottles_l3407_340779


namespace NUMINAMATH_CALUDE_correct_regression_equation_l3407_340743

-- Define the sample means
def x_mean : ℝ := 2
def y_mean : ℝ := 1.5

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := -2 * x + 5.5

-- State the theorem
theorem correct_regression_equation :
  -- Condition: x and y are negatively correlated
  (∃ k : ℝ, k < 0 ∧ ∀ x y : ℝ, y = k * x + linear_regression x_mean - k * x_mean) →
  -- The linear regression equation passes through the point (x_mean, y_mean)
  linear_regression x_mean = y_mean := by
  sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l3407_340743


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l3407_340774

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n)) ∧ 
  (∀ m : ℕ, m > 6 → Nat.choose 10 4 + Nat.choose 10 5 ≠ Nat.choose 11 m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l3407_340774


namespace NUMINAMATH_CALUDE_perpendicular_and_tangent_l3407_340706

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^3 + 3 * x^2 - 1

-- Define the perpendicular line (our answer)
def perp_line (x y : ℝ) : Prop := 3 * x + y + 2 = 0

-- State the theorem
theorem perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    curve x₀ y₀ ∧
    -- The point (x₀, y₀) is on the perpendicular line
    perp_line x₀ y₀ ∧
    -- The perpendicular line is indeed perpendicular to the given line
    (3 : ℝ) * (1 / 3 : ℝ) = -1 ∧
    -- The slope of the curve at (x₀, y₀) equals the slope of the perpendicular line
    (3 * x₀^2 + 6 * x₀) = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_and_tangent_l3407_340706


namespace NUMINAMATH_CALUDE_ratio_RN_NS_l3407_340789

/-- Square ABCD with side length 10, F is on DC 3 units from D, N is midpoint of AF,
    perpendicular bisector of AF intersects AD at R and BC at S -/
structure SquareConfiguration where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  F : ℝ × ℝ
  N : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_square : A = (0, 10) ∧ B = (10, 10) ∧ C = (10, 0) ∧ D = (0, 0)
  h_F : F = (3, 0)
  h_N : N = (3/2, 5)
  h_R : R.1 = 57/3 ∧ R.2 = 10
  h_S : S.1 = -43/3 ∧ S.2 = 0

/-- The ratio of RN to NS is 1:1 -/
theorem ratio_RN_NS (cfg : SquareConfiguration) : 
  dist cfg.R cfg.N = dist cfg.N cfg.S :=
by sorry


end NUMINAMATH_CALUDE_ratio_RN_NS_l3407_340789


namespace NUMINAMATH_CALUDE_frequency_below_70kg_l3407_340786

def total_students : ℕ := 50

def weight_groups : List (ℝ × ℝ) := [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]

def frequencies : List ℕ := [6, 8, 15, 18, 3]

def students_below_70kg : ℕ := 29

theorem frequency_below_70kg :
  (students_below_70kg : ℝ) / total_students = 0.58 :=
sorry

end NUMINAMATH_CALUDE_frequency_below_70kg_l3407_340786


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3407_340775

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 = x}
def N : Set ℝ := {x : ℝ | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3407_340775


namespace NUMINAMATH_CALUDE_books_count_l3407_340713

def total_books (beatrix alannah queen kingston : ℕ) : ℕ :=
  beatrix + alannah + queen + kingston

theorem books_count :
  ∀ (beatrix alannah queen kingston : ℕ),
    beatrix = 30 →
    alannah = beatrix + 20 →
    queen = alannah + alannah / 5 →
    kingston = 2 * (beatrix + queen) →
    total_books beatrix alannah queen kingston = 320 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l3407_340713


namespace NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_95_minus_107_l3407_340751

-- Define the number
def n : ℕ := 95

-- Define the function to calculate the number
def f (n : ℕ) : ℤ := 10^n - 107

-- Define the function to calculate the sum of digits
def sum_of_digits (z : ℤ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_10_pow_95_minus_107 :
  sum_of_digits (f n) = 849 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_95_minus_107_l3407_340751


namespace NUMINAMATH_CALUDE_inequality_proof_l3407_340734

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3407_340734


namespace NUMINAMATH_CALUDE_simplify_expression_l3407_340717

theorem simplify_expression (a : ℝ) : 5*a + 2*a + 3*a - 2*a = 8*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3407_340717


namespace NUMINAMATH_CALUDE_proposition_and_variants_l3407_340747

theorem proposition_and_variants (x y : ℝ) :
  (∀ x y, xy = 0 → x = 0 ∨ y = 0) ∧
  (∀ x y, x = 0 ∨ y = 0 → xy = 0) ∧
  (∀ x y, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → xy ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variants_l3407_340747


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3407_340736

theorem unique_triple_solution : 
  ∃! (a b c : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
    a^2 + b^2 + c^2 = 3 ∧ 
    (a + b + c) * (a^2*b + b^2*c + c^2*a) = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3407_340736


namespace NUMINAMATH_CALUDE_exprC_is_factorization_left_to_right_l3407_340705

/-- Represents a polynomial expression -/
structure PolynomialExpression where
  left : ℝ → ℝ → ℝ
  right : ℝ → ℝ → ℝ

/-- Checks if an expression is in product form -/
def isProductForm (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ (f g : ℝ → ℝ → ℝ), ∀ x y, expr x y = f x y * g x y

/-- Defines factorization from left to right -/
def isFactorizationLeftToRight (expr : PolynomialExpression) : Prop :=
  ¬(isProductForm expr.left) ∧ (isProductForm expr.right)

/-- The specific expression we're examining -/
def exprC : PolynomialExpression :=
  { left := λ a b => a^2 - 4*a*b + 4*b^2,
    right := λ a b => (a - 2*b)^2 }

/-- Theorem stating that exprC represents factorization from left to right -/
theorem exprC_is_factorization_left_to_right :
  isFactorizationLeftToRight exprC :=
sorry

end NUMINAMATH_CALUDE_exprC_is_factorization_left_to_right_l3407_340705


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3407_340715

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3407_340715


namespace NUMINAMATH_CALUDE_sticker_pages_calculation_l3407_340788

/-- Given a total number of stickers and the number of stickers per page,
    calculate the number of pages. -/
def calculate_pages (total_stickers : ℕ) (stickers_per_page : ℕ) : ℕ :=
  total_stickers / stickers_per_page

/-- Theorem stating that with 220 total stickers and 10 stickers per page,
    the number of pages is 22. -/
theorem sticker_pages_calculation :
  calculate_pages 220 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_sticker_pages_calculation_l3407_340788


namespace NUMINAMATH_CALUDE_inequality_proof_root_mean_square_arithmetic_mean_l3407_340723

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

theorem root_mean_square_arithmetic_mean (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_root_mean_square_arithmetic_mean_l3407_340723


namespace NUMINAMATH_CALUDE_power_sum_problem_l3407_340703

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l3407_340703


namespace NUMINAMATH_CALUDE_monotonicity_depends_on_a_l3407_340745

/-- The function f(x) = x³ + ax² + 1 where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

/-- Theorem stating that the monotonicity of f depends on the value of a -/
theorem monotonicity_depends_on_a :
  ∀ a : ℝ, ∃ x y : ℝ, x < y ∧
    ((f_derivative a x > 0 ∧ f_derivative a y < 0) ∨
     (f_derivative a x < 0 ∧ f_derivative a y > 0) ∨
     (∀ z : ℝ, f_derivative a z ≥ 0) ∨
     (∀ z : ℝ, f_derivative a z ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_depends_on_a_l3407_340745


namespace NUMINAMATH_CALUDE_triple_tangent_identity_l3407_340725

theorem triple_tangent_identity (x y z : ℝ) 
  (hx : |x| ≠ 1 / Real.sqrt 3) 
  (hy : |y| ≠ 1 / Real.sqrt 3) 
  (hz : |z| ≠ 1 / Real.sqrt 3) 
  (h_sum : x + y + z = x * y * z) : 
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2) = 
  (3 * x - x^3) / (1 - 3 * x^2) * (3 * y - y^3) / (1 - 3 * y^2) * (3 * z - z^3) / (1 - 3 * z^2) := by
  sorry

end NUMINAMATH_CALUDE_triple_tangent_identity_l3407_340725


namespace NUMINAMATH_CALUDE_zoey_reading_schedule_l3407_340733

def days_to_read (n : ℕ) : ℕ := n

def total_days (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem zoey_reading_schedule (num_books : ℕ) (start_day : ℕ) 
  (h1 : num_books = 20)
  (h2 : start_day = 5) -- Friday is represented as 5 (0 is Sunday)
  : day_of_week start_day (total_days num_books) = start_day := by
  sorry

#check zoey_reading_schedule

end NUMINAMATH_CALUDE_zoey_reading_schedule_l3407_340733


namespace NUMINAMATH_CALUDE_zachary_needs_money_l3407_340730

/-- The additional amount of money Zachary needs to buy football equipment -/
def additional_money_needed (football_price : ℝ) (shorts_price : ℝ) (shoes_price : ℝ) 
  (socks_price : ℝ) (water_bottle_price : ℝ) (eur_to_usd : ℝ) (gbp_to_usd : ℝ) 
  (jpy_to_usd : ℝ) (krw_to_usd : ℝ) (discount_rate : ℝ) (current_money : ℝ) : ℝ :=
  let total_cost := football_price * eur_to_usd + 2 * shorts_price * gbp_to_usd + 
    shoes_price + 4 * socks_price * jpy_to_usd + water_bottle_price * krw_to_usd
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost - current_money

/-- Theorem stating the additional amount Zachary needs -/
theorem zachary_needs_money : 
  additional_money_needed 3.756 2.498 11.856 135.29 7834 1.19 1.38 0.0088 0.00085 0.1 24.042 = 7.127214 := by
  sorry

end NUMINAMATH_CALUDE_zachary_needs_money_l3407_340730


namespace NUMINAMATH_CALUDE_divisible_by_five_l3407_340726

theorem divisible_by_five (n : ℕ) : 
  (∃ B : ℕ, B < 10 ∧ n = 5270 + B) → (n % 5 = 0 ↔ B = 0 ∨ B = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l3407_340726


namespace NUMINAMATH_CALUDE_worker_completion_time_l3407_340741

/-- Given workers A and B, where A can complete a job in 15 days, works for 5 days,
    and B finishes the remaining work in 18 days, prove that B can complete the
    entire job alone in 27 days. -/
theorem worker_completion_time
  (total_days_A : ℕ)
  (work_days_A : ℕ)
  (remaining_days_B : ℕ)
  (h1 : total_days_A = 15)
  (h2 : work_days_A = 5)
  (h3 : remaining_days_B = 18) :
  (total_days_A * remaining_days_B) / (total_days_A - work_days_A) = 27 := by
  sorry

end NUMINAMATH_CALUDE_worker_completion_time_l3407_340741


namespace NUMINAMATH_CALUDE_largest_six_digit_divisible_by_five_l3407_340746

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem largest_six_digit_divisible_by_five :
  ∃ (n : ℕ), is_six_digit n ∧ n % 5 = 0 ∧ ∀ (m : ℕ), is_six_digit m ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_six_digit_divisible_by_five_l3407_340746


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3407_340704

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ ¬(a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3407_340704


namespace NUMINAMATH_CALUDE_smallest_sum_on_square_corners_l3407_340767

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_not_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b > 1

theorem smallest_sum_on_square_corners (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  is_relatively_prime A C →
  is_relatively_prime B D →
  is_not_relatively_prime A B →
  is_not_relatively_prime B C →
  is_not_relatively_prime C D →
  is_not_relatively_prime D A →
  A + B + C + D ≥ 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_on_square_corners_l3407_340767


namespace NUMINAMATH_CALUDE_bus_distance_l3407_340702

/-- Represents the distance traveled by each mode of transportation -/
structure TravelDistances where
  total : ℝ
  plane : ℝ
  train : ℝ
  bus : ℝ

/-- The conditions of the travel problem -/
def travel_conditions (d : TravelDistances) : Prop :=
  d.total = 900 ∧
  d.plane = d.total / 3 ∧
  d.train = 2 / 3 * d.bus ∧
  d.total = d.plane + d.train + d.bus

/-- The theorem stating that under the given conditions, the bus travel distance is 360 km -/
theorem bus_distance (d : TravelDistances) (h : travel_conditions d) : d.bus = 360 := by
  sorry

end NUMINAMATH_CALUDE_bus_distance_l3407_340702


namespace NUMINAMATH_CALUDE_max_value_of_function_l3407_340765

open Real

theorem max_value_of_function (x : ℝ) : 
  ∃ (M : ℝ), M = 2 - sqrt 3 ∧ ∀ y : ℝ, sin (2 * y) - 2 * sqrt 3 * sin y ^ 2 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3407_340765


namespace NUMINAMATH_CALUDE_root_equation_l3407_340700

theorem root_equation (p : ℝ) :
  (0 ≤ p ∧ p ≤ 4/3) →
  (∃! x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x ∧
             x = (4 - p) / Real.sqrt (8 * (2 - p))) ∧
  (p < 0 ∨ p > 4/3) →
  (∀ x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) ≠ x) := by
sorry

end NUMINAMATH_CALUDE_root_equation_l3407_340700


namespace NUMINAMATH_CALUDE_M_equals_P_l3407_340753

-- Define set M
def M : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}

-- Define set P
def P : Set ℝ := {y | ∃ b : ℝ, y = b^2 - 4*b + 5}

-- Theorem stating that M equals P
theorem M_equals_P : M = P := by sorry

end NUMINAMATH_CALUDE_M_equals_P_l3407_340753


namespace NUMINAMATH_CALUDE_probability_of_valid_pair_l3407_340780

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

def is_divisible_by_5 (n : ℤ) : Prop := n % 5 = 0

def valid_pair (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧
  1 ≤ b ∧ b ≤ 20 ∧
  a ≠ b ∧
  is_odd (a * b) ∧
  is_divisible_by_5 (a + b)

def total_pairs : ℕ := 190

def valid_pairs : ℕ := 18

theorem probability_of_valid_pair :
  (valid_pairs : ℚ) / total_pairs = 9 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_of_valid_pair_l3407_340780


namespace NUMINAMATH_CALUDE_lesser_fraction_l3407_340760

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 13 / 14)
  (prod_eq : x * y = 1 / 8) :
  min x y = (13 - Real.sqrt 113) / 28 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l3407_340760


namespace NUMINAMATH_CALUDE_problem_statement_l3407_340784

theorem problem_statement (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * (Real.log x / Real.log y) + 2 * (Real.log y / Real.log x) = 8) 
  (h4 : x * y = 256) : (x + y) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3407_340784


namespace NUMINAMATH_CALUDE_initial_student_count_l3407_340729

/-- Proves that the initial number of students is 29 given the conditions of the problem -/
theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 28 →
  new_avg = 27.5 →
  new_student_weight = 13 →
  ∃ n : ℕ, n = 29 ∧
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by
  sorry


end NUMINAMATH_CALUDE_initial_student_count_l3407_340729


namespace NUMINAMATH_CALUDE_transistors_in_1995_l3407_340772

/-- Moore's law states that the number of transistors doubles every 18 months -/
def moores_law_period : ℕ := 18

/-- Initial year when the count began -/
def initial_year : ℕ := 1985

/-- Initial number of transistors -/
def initial_transistors : ℕ := 500000

/-- Target year for calculation -/
def target_year : ℕ := 1995

/-- Calculate the number of transistors based on Moore's law -/
def calculate_transistors (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ (months / moores_law_period))

/-- Theorem stating that the number of transistors in 1995 is 32,000,000 -/
theorem transistors_in_1995 :
  calculate_transistors initial_transistors ((target_year - initial_year) * 12) = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_1995_l3407_340772


namespace NUMINAMATH_CALUDE_square_difference_l3407_340754

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3407_340754


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l3407_340761

theorem initial_amount_calculation (deposit : ℚ) (initial : ℚ) : 
  deposit = 750 → 
  deposit = initial * (20 / 100) * (25 / 100) * (30 / 100) → 
  initial = 50000 := by
sorry

end NUMINAMATH_CALUDE_initial_amount_calculation_l3407_340761


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3407_340707

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  units_digit (27 * 46) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3407_340707


namespace NUMINAMATH_CALUDE_smallest_common_factor_l3407_340759

theorem smallest_common_factor (m : ℕ) : m = 108 ↔ 
  (m > 0 ∧ 
   ∃ (k : ℕ), k > 1 ∧ k ∣ (11*m - 3) ∧ k ∣ (8*m + 5) ∧
   ∀ (n : ℕ), n < m → ¬(∃ (l : ℕ), l > 1 ∧ l ∣ (11*n - 3) ∧ l ∣ (8*n + 5))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l3407_340759


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3407_340742

theorem quadratic_equation_properties (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  let f : ℝ → ℝ := fun x ↦ x^2 + (a + b - 1 : ℝ) * x + (a * b - a - b : ℝ)
  -- The equation has two distinct real solutions
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧
  -- If one solution is an integer, then both are non-positive integers and b < 2a
  (∃ z : ℤ, f (z : ℝ) = 0 → ∃ r s : ℤ, r ≤ 0 ∧ s ≤ 0 ∧ f (r : ℝ) = 0 ∧ f (s : ℝ) = 0 ∧ b < 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3407_340742


namespace NUMINAMATH_CALUDE_vessel_mixture_theorem_main_vessel_theorem_l3407_340711

/-- Represents the contents of a vessel -/
structure VesselContents where
  milk : ℚ
  water : ℚ

/-- Represents a vessel with its volume and contents -/
structure Vessel where
  volume : ℚ
  contents : VesselContents

/-- Theorem stating the relationship between vessel contents and their mixture -/
theorem vessel_mixture_theorem 
  (v1 v2 : Vessel) 
  (h1 : v1.volume / v2.volume = 3 / 5)
  (h2 : v2.contents.milk / v2.contents.water = 3 / 2)
  (h3 : (v1.contents.milk + v2.contents.milk) / (v1.contents.water + v2.contents.water) = 1) :
  v1.contents.milk / v1.contents.water = 1 / 2 := by
  sorry

/-- Main theorem proving the equivalence of the vessel mixture problem -/
theorem main_vessel_theorem :
  ∀ (v1 v2 : Vessel),
  v1.volume / v2.volume = 3 / 5 →
  v2.contents.milk / v2.contents.water = 3 / 2 →
  (v1.contents.milk + v2.contents.milk) / (v1.contents.water + v2.contents.water) = 1 ↔
  v1.contents.milk / v1.contents.water = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vessel_mixture_theorem_main_vessel_theorem_l3407_340711


namespace NUMINAMATH_CALUDE_distance_from_point_to_line_l3407_340749

def point : ℝ × ℝ × ℝ := (2, 4, 5)

def line_point : ℝ × ℝ × ℝ := (4, 6, 8)
def line_direction : ℝ × ℝ × ℝ := (1, 1, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  distance_to_line point line_point line_direction = 2 * Real.sqrt 33 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_line_l3407_340749


namespace NUMINAMATH_CALUDE_integer_count_equality_l3407_340716

theorem integer_count_equality : 
  ∃! (count : ℕ), count = 39999 ∧ 
  (∀ n : ℤ, (2 + ⌊(200 * n : ℚ) / 201⌋ = ⌈(198 * n : ℚ) / 199⌉) ↔ 
    (∃ k : ℤ, 0 ≤ k ∧ k < count ∧ n ≡ k [ZMOD 39999])) :=
by sorry

end NUMINAMATH_CALUDE_integer_count_equality_l3407_340716


namespace NUMINAMATH_CALUDE_f_is_decreasing_l3407_340721

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 2

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being a decreasing function on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem f_is_decreasing (a b : ℝ) :
  is_even_function (f a b) ∧ (Set.Icc (1 + a) 2).Nonempty →
  is_decreasing_on (f a b) 1 2 := by
  sorry

end NUMINAMATH_CALUDE_f_is_decreasing_l3407_340721


namespace NUMINAMATH_CALUDE_christines_speed_l3407_340791

theorem christines_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 20 ∧ time = 5 ∧ speed = distance / time → speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_christines_speed_l3407_340791


namespace NUMINAMATH_CALUDE_smallest_y_l3407_340748

theorem smallest_y (x y : ℕ+) (h1 : x.val - y.val = 8) 
  (h2 : Nat.gcd ((x.val^3 + y.val^3) / (x.val + y.val)) (x.val * y.val) = 16) :
  ∀ z : ℕ+, z.val < y.val → 
    Nat.gcd ((z.val^3 + (z.val + 8)^3) / (z.val + (z.val + 8))) (z.val * (z.val + 8)) ≠ 16 :=
by sorry

#check smallest_y

end NUMINAMATH_CALUDE_smallest_y_l3407_340748


namespace NUMINAMATH_CALUDE_video_library_space_per_hour_l3407_340737

/-- Given a video library with the following properties:
  * Contains 15 days of videos
  * Each day consists of 18 hours of videos
  * The entire library takes up 45,000 megabytes of disk space
  This theorem proves that the disk space required for one hour of video,
  when rounded to the nearest whole number, is 167 megabytes. -/
theorem video_library_space_per_hour :
  ∀ (days hours_per_day total_space : ℕ),
  days = 15 →
  hours_per_day = 18 →
  total_space = 45000 →
  round ((total_space : ℝ) / (days * hours_per_day : ℝ)) = 167 := by
  sorry

end NUMINAMATH_CALUDE_video_library_space_per_hour_l3407_340737


namespace NUMINAMATH_CALUDE_max_value_on_interval_l3407_340795

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem max_value_on_interval :
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l3407_340795


namespace NUMINAMATH_CALUDE_max_value_vx_minus_yz_l3407_340757

def A : Set Int := {-3, -2, -1, 0, 1, 2, 3}

theorem max_value_vx_minus_yz :
  ∃ (v x y z : Int), v ∈ A ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A ∧
    v * x - y * z = 6 ∧
    ∀ (v' x' y' z' : Int), v' ∈ A → x' ∈ A → y' ∈ A → z' ∈ A →
      v' * x' - y' * z' ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_vx_minus_yz_l3407_340757


namespace NUMINAMATH_CALUDE_both_games_count_l3407_340724

/-- The number of people who play both kabadi and kho kho -/
def both_games : ℕ := sorry

/-- The total number of players -/
def total_players : ℕ := 45

/-- The number of people who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of people who play only kho kho -/
def only_kho_kho : ℕ := 35

theorem both_games_count : both_games = 10 := by sorry

end NUMINAMATH_CALUDE_both_games_count_l3407_340724


namespace NUMINAMATH_CALUDE_ellipse_properties_l3407_340763

/-- Properties of the ellipse y²/25 + x²/16 = 1 -/
theorem ellipse_properties :
  let ellipse := (fun (x y : ℝ) => y^2 / 25 + x^2 / 16 = 1)
  ∃ (a b c : ℝ),
    -- Major and minor axis lengths
    a = 5 ∧ b = 4 ∧
    -- Vertices
    ellipse (-4) 0 ∧ ellipse 4 0 ∧ ellipse 0 5 ∧ ellipse 0 (-5) ∧
    -- Foci
    c = 3 ∧
    -- Eccentricity
    c / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3407_340763


namespace NUMINAMATH_CALUDE_arithmetic_progression_solution_l3407_340701

theorem arithmetic_progression_solution (a : ℝ) : 
  (3 - 2*a = a - 6 - 3) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_solution_l3407_340701


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3407_340755

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3407_340755


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_zero_l3407_340790

/-- The system of equations has no solution if and only if n = 0 -/
theorem no_solution_iff_n_eq_zero (n : ℝ) :
  (∃ (x y z : ℝ), 2*n*x + y = 2 ∧ 3*n*y + z = 3 ∧ x + 2*n*z = 2) ↔ n ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_zero_l3407_340790


namespace NUMINAMATH_CALUDE_james_total_spent_l3407_340750

def milk_price : ℚ := 3
def bananas_price : ℚ := 2
def bread_price : ℚ := 3/2
def cereal_price : ℚ := 4

def milk_tax_rate : ℚ := 1/5
def bananas_tax_rate : ℚ := 3/20
def bread_tax_rate : ℚ := 1/10
def cereal_tax_rate : ℚ := 1/4

def total_spent : ℚ := milk_price * (1 + milk_tax_rate) + 
                       bananas_price * (1 + bananas_tax_rate) + 
                       bread_price * (1 + bread_tax_rate) + 
                       cereal_price * (1 + cereal_tax_rate)

theorem james_total_spent : total_spent = 251/20 := by
  sorry

end NUMINAMATH_CALUDE_james_total_spent_l3407_340750


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3407_340796

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) ∧ ¬(∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3407_340796


namespace NUMINAMATH_CALUDE_total_milk_count_l3407_340738

theorem total_milk_count (chocolate : ℕ) (strawberry : ℕ) (regular : ℕ)
  (h1 : chocolate = 2)
  (h2 : strawberry = 15)
  (h3 : regular = 3) :
  chocolate + strawberry + regular = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_milk_count_l3407_340738


namespace NUMINAMATH_CALUDE_smallest_sum_mn_smallest_sum_value_unique_smallest_sum_l3407_340797

theorem smallest_sum_mn (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : 
  ∀ (a b : ℕ), (3 * a^3 = 5 * b^2) → m + n ≤ a + b :=
by
  sorry

theorem smallest_sum_value : 
  ∃ (m n : ℕ), (3 * n^3 = 5 * m^2) ∧ (m + n = 60) :=
by
  sorry

theorem unique_smallest_sum : 
  ∀ (m n : ℕ), (3 * n^3 = 5 * m^2) → m + n ≥ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_mn_smallest_sum_value_unique_smallest_sum_l3407_340797


namespace NUMINAMATH_CALUDE_equation_equivalence_l3407_340770

theorem equation_equivalence (x : ℝ) : 1 - (x + 3) / 3 = x / 2 ↔ 6 - 2 * x - 6 = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3407_340770


namespace NUMINAMATH_CALUDE_teachers_pizza_fraction_l3407_340762

theorem teachers_pizza_fraction (teachers : ℕ) (staff : ℕ) (staff_pizza_fraction : ℚ) (non_pizza_eaters : ℕ) :
  teachers = 30 →
  staff = 45 →
  staff_pizza_fraction = 4/5 →
  non_pizza_eaters = 19 →
  (teachers : ℚ) * (2/3) + (staff : ℚ) * staff_pizza_fraction = (teachers + staff : ℚ) - non_pizza_eaters := by
  sorry

end NUMINAMATH_CALUDE_teachers_pizza_fraction_l3407_340762


namespace NUMINAMATH_CALUDE_largest_difference_l3407_340787

-- Define the type for our table
def Table := Fin 20 → Fin 20 → Fin 400

-- Define a property that checks if a table is valid
def is_valid_table (t : Table) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

-- Define the property we want to prove
theorem largest_difference (t : Table) (h : is_valid_table t) :
  (∃ i j k, (i = k ∨ j = k) ∧ 
    (t i j : ℕ).succ.pred - (t i k : ℕ).succ.pred ≥ 209) ∧
  ∀ m, m > 209 → 
    ∃ t', is_valid_table t' ∧ 
      ∀ i j k, (i = k ∨ j = k) → 
        (t' i j : ℕ).succ.pred - (t' i k : ℕ).succ.pred < m :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l3407_340787


namespace NUMINAMATH_CALUDE_problem_solution_l3407_340756

-- Define p₁
def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem problem_solution : (¬p₁) ∨ (¬p₂) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3407_340756


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3407_340785

/-- Given a line L1 with equation 3x - y = 7 and a point P (2, -3),
    this theorem proves that the line L2 with equation y = -1/3x - 7/3
    is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y = 7
  let L2 : ℝ → ℝ → Prop := λ x y => y = -1/3 * x - 7/3
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y → L2 x y → (3 * (-1/3) = -1)) ∧ 
  (L2 P.1 P.2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3407_340785


namespace NUMINAMATH_CALUDE_diplomats_conference_l3407_340793

theorem diplomats_conference (D : ℕ) : 
  (20 : ℕ) ≤ D ∧  -- Number of diplomats who spoke Japanese
  (32 : ℕ) ≤ D ∧  -- Number of diplomats who did not speak Russian
  (D - (20 + (D - 32) - (D / 10 : ℕ)) : ℤ) = (D / 5 : ℕ) ∧  -- 20% spoke neither Japanese nor Russian
  (D / 10 : ℕ) ≤ 20  -- 10% spoke both Japanese and Russian (this must be ≤ 20)
  → D = 40 := by
sorry

end NUMINAMATH_CALUDE_diplomats_conference_l3407_340793


namespace NUMINAMATH_CALUDE_jesse_banana_sharing_l3407_340708

theorem jesse_banana_sharing :
  ∀ (total_bananas : ℕ) (bananas_per_friend : ℕ) (num_friends : ℕ),
    total_bananas = 21 →
    bananas_per_friend = 7 →
    total_bananas = bananas_per_friend * num_friends →
    num_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_jesse_banana_sharing_l3407_340708


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l3407_340799

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 120 = longer_base
  area_ratio : (shorter_base + (shorter_base + 60)) / ((shorter_base + 60) + longer_base) = 3 / 4
  equal_areas : equal_area_segment > shorter_base ∧ equal_area_segment < longer_base

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  ⌊t.equal_area_segment ^ 2 / 120⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l3407_340799


namespace NUMINAMATH_CALUDE_center_is_eight_l3407_340773

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def isAdjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Check if a grid is valid according to the problem conditions --/
def isValidGrid (g : Grid) : Prop :=
  (∀ n : Fin 9, ∃! p : Fin 3 × Fin 3, g p.1 p.2 = n.val + 1) ∧
  (∀ n : Fin 8, ∃ p1 p2 : Fin 3 × Fin 3, 
    g p1.1 p1.2 = n.val + 1 ∧ 
    g p2.1 p2.2 = n.val + 2 ∧ 
    isAdjacent p1 p2) ∧
  (g 0 0 + g 0 2 + g 2 0 + g 2 2 = 24)

theorem center_is_eight (g : Grid) (h : isValidGrid g) : 
  g 1 1 = 8 := by sorry

end NUMINAMATH_CALUDE_center_is_eight_l3407_340773


namespace NUMINAMATH_CALUDE_log_equation_solution_l3407_340781

theorem log_equation_solution : 
  ∃! x : ℝ, (1 : ℝ) + Real.log x = Real.log (1 + x) :=
by
  use (1 : ℝ) / 9
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3407_340781


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3407_340727

theorem polynomial_expansion (x : ℝ) :
  (7 * x - 3) * (2 * x^3 + 5 * x^2 - 4) = 14 * x^4 + 29 * x^3 - 15 * x^2 - 28 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3407_340727


namespace NUMINAMATH_CALUDE_inequality_solution_l3407_340728

theorem inequality_solution :
  {n : ℕ+ | 25 - 5 * n.val < 15} = {n : ℕ+ | n.val > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3407_340728


namespace NUMINAMATH_CALUDE_sum_of_valid_a_l3407_340719

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ (3*x - a)/(x - 3) + (x + 1)/(3 - x) = 1) ∧
    (∀ y : ℝ, (y + 9 ≤ 2*(y + 2) ∧ (2*y - a)/3 > 1) ↔ y ≥ 5)) ∧
  (∀ a : ℤ, 
    ((∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ (3*x - a)/(x - 3) + (x + 1)/(3 - x) = 1) ∧
    (∀ y : ℝ, (y + 9 ≤ 2*(y + 2) ∧ (2*y - a)/3 > 1) ↔ y ≥ 5)) → a ∈ S) ∧
  Finset.sum S id = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_l3407_340719


namespace NUMINAMATH_CALUDE_pens_bought_l3407_340709

/-- Represents the cost of a single notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- Represents the number of pens Maria bought -/
def num_pens : ℝ := sorry

/-- Theorem stating the relationship between the number of pens, total cost, and notebook cost -/
theorem pens_bought (notebook_cost num_pens : ℝ) : 
  (10 * notebook_cost + 2 * num_pens = 30) → 
  (num_pens = (30 - 10 * notebook_cost) / 2) := by
  sorry

end NUMINAMATH_CALUDE_pens_bought_l3407_340709


namespace NUMINAMATH_CALUDE_parabola_focus_is_correct_l3407_340740

/-- The focus of a parabola given by y = ax^2 + bx + c -/
def parabola_focus (a b c : ℝ) : ℝ × ℝ := sorry

/-- The equation of the parabola -/
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 - 6 * x

theorem parabola_focus_is_correct :
  parabola_focus (-3) (-6) 0 = (-1, 35/12) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_is_correct_l3407_340740


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3407_340777

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 8
  let θ : ℝ := 5 * π / 4
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (-2 * Real.sqrt 2, -2 * Real.sqrt 2, 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3407_340777


namespace NUMINAMATH_CALUDE_grass_seed_cost_l3407_340718

/-- Represents the cost and weight of a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  cost : Float

/-- Represents a purchase of grass seed bags -/
structure Purchase where
  bags : List GrassSeedBag
  totalWeight : Nat
  totalCost : Float

def tenPoundBag : GrassSeedBag := { weight := 10, cost := 20.43 }
def twentyFivePoundBag : GrassSeedBag := { weight := 25, cost := 32.20 }

/-- The optimal purchase satisfying the given conditions -/
def optimalPurchase (fivePoundBagCost : Float) : Purchase :=
  { bags := [twentyFivePoundBag, twentyFivePoundBag, twentyFivePoundBag, 
             { weight := 5, cost := fivePoundBagCost }],
    totalWeight := 80,
    totalCost := 3 * 32.20 + fivePoundBagCost }

theorem grass_seed_cost 
  (h1 : ∀ p : Purchase, p.totalWeight ≥ 65 ∧ p.totalWeight ≤ 80 → p.totalCost ≥ 98.68)
  (h2 : (optimalPurchase 2.08).totalCost = 98.68) :
  ∃ fivePoundBagCost : Float, fivePoundBagCost = 2.08 ∧ 
    (optimalPurchase fivePoundBagCost).totalCost = 98.68 ∧
    (optimalPurchase fivePoundBagCost).totalWeight ≥ 65 ∧
    (optimalPurchase fivePoundBagCost).totalWeight ≤ 80 := by
  sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l3407_340718


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3407_340792

/-- Given a rectangle with length 40 cm, if reducing the length by 5 cm and
    increasing the width by 5 cm results in an area increase of 75 cm²,
    then the ratio of the original length to the original width is 2:1. -/
theorem rectangle_ratio (w : ℝ) : 
  (40 - 5) * (w + 5) = 40 * w + 75 → 40 / w = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3407_340792


namespace NUMINAMATH_CALUDE_weight_difference_is_one_black_dog_weight_conditions_are_consistent_l3407_340739

-- Define the weights of the dogs
def brown_weight : ℝ := 4
def black_weight : ℝ := 5  -- This is derived from the solution, but we'll prove it
def white_weight : ℝ := 2 * brown_weight
def grey_weight : ℝ := black_weight - 2

-- Define the average weight
def average_weight : ℝ := 5

-- Define the number of dogs
def num_dogs : ℕ := 4

-- Theorem to prove
theorem weight_difference_is_one :
  black_weight - brown_weight = 1 :=
by
  -- The proof would go here
  sorry

-- Additional theorem to prove the black dog's weight
theorem black_dog_weight :
  black_weight = 5 :=
by
  -- This proof would use the given conditions to show that black_weight must be 5
  sorry

-- Theorem to show that the conditions are consistent
theorem conditions_are_consistent :
  (brown_weight + black_weight + white_weight + grey_weight) / num_dogs = average_weight :=
by
  -- This proof would show that the given weights satisfy the average weight condition
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_one_black_dog_weight_conditions_are_consistent_l3407_340739


namespace NUMINAMATH_CALUDE_final_balance_is_correct_l3407_340731

/-- Represents a currency with its exchange rate to USD -/
structure Currency where
  name : String
  exchange_rate : Float

/-- Represents a transaction with amount, currency, and discount -/
structure Transaction where
  amount : Float
  currency : Currency
  discount : Float

def initial_balance : Float := 126.00

def gbp : Currency := { name := "GBP", exchange_rate := 1.39 }
def eur : Currency := { name := "EUR", exchange_rate := 1.18 }
def jpy : Currency := { name := "JPY", exchange_rate := 0.0091 }
def usd : Currency := { name := "USD", exchange_rate := 1.0 }

def uk_transaction : Transaction := { amount := 50.0, currency := gbp, discount := 0.1 }
def france_transaction : Transaction := { amount := 70.0, currency := eur, discount := 0.15 }
def japan_transaction : Transaction := { amount := 10000.0, currency := jpy, discount := 0.05 }
def us_gas_transaction : Transaction := { amount := 25.0, currency := gbp, discount := 0.0 }
def return_transaction : Transaction := { amount := 45.0, currency := usd, discount := 0.0 }

def monthly_interest_rate : Float := 0.015

def calculate_final_balance (initial_balance : Float) 
  (transactions : List Transaction) 
  (return_transaction : Transaction)
  (monthly_interest_rate : Float) : Float :=
  sorry

theorem final_balance_is_correct :
  calculate_final_balance initial_balance 
    [uk_transaction, france_transaction, japan_transaction, us_gas_transaction]
    return_transaction
    monthly_interest_rate = 340.00 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_is_correct_l3407_340731


namespace NUMINAMATH_CALUDE_library_shelf_count_l3407_340798

theorem library_shelf_count (notebooks : ℕ) (pen_difference : ℕ) (pencil_difference : ℕ) 
  (h1 : notebooks = 40)
  (h2 : pen_difference = 80)
  (h3 : pencil_difference = 45) :
  notebooks + (notebooks + pen_difference) + (notebooks + pencil_difference) = 245 :=
by
  sorry

end NUMINAMATH_CALUDE_library_shelf_count_l3407_340798
