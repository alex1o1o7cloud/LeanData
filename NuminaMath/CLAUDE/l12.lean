import Mathlib

namespace NUMINAMATH_CALUDE_range_of_g_minus_x_l12_1213

def g (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem range_of_g_minus_x :
  Set.range (fun x => g x - x) ∩ Set.Icc (-2 : ℝ) 2 = Set.Icc 0 16 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_minus_x_l12_1213


namespace NUMINAMATH_CALUDE_cookie_sale_loss_l12_1234

/-- Represents the cookie sale scenario --/
structure CookieSale where
  total_cookies : ℕ
  purchase_rate : ℚ  -- cookies per dollar
  selling_rate : ℚ   -- cookies per dollar

/-- Calculates the loss from a cookie sale --/
def calculate_loss (sale : CookieSale) : ℚ :=
  let cost := sale.total_cookies / sale.purchase_rate
  let revenue := sale.total_cookies / sale.selling_rate
  cost - revenue

/-- The main theorem stating the loss for the given scenario --/
theorem cookie_sale_loss : 
  let sale : CookieSale := {
    total_cookies := 800,
    purchase_rate := 4/3,  -- 4 cookies for $3
    selling_rate := 3/2    -- 3 cookies for $2
  }
  calculate_loss sale = 64 := by
  sorry


end NUMINAMATH_CALUDE_cookie_sale_loss_l12_1234


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l12_1212

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d = 64000) :
  (∃ k : ℕ, (7 : ℚ) / d = k / 10^(n + 1) ∧ k % 10 ≠ 0 ∧ k < 10^n) → n = 4 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l12_1212


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l12_1296

theorem min_value_x_plus_y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 9 / (x + 1) + 1 / (y + 1) = 1) : 
  x + y ≥ 14 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9 / (x₀ + 1) + 1 / (y₀ + 1) = 1 ∧ x₀ + y₀ = 14 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l12_1296


namespace NUMINAMATH_CALUDE_prime_ends_of_sequence_l12_1257

theorem prime_ends_of_sequence (k : ℕ) : 
  (∃ S : Finset ℕ, S.card = 7 ∧ 
    (∀ n ∈ S, n ∈ Finset.range 29 ∧ Nat.Prime (30 * k + n + 1)) ∧
    (∀ n ∈ Finset.range 29 \ S, ¬Nat.Prime (30 * k + n + 1))) →
  Nat.Prime (30 * k + 1) ∧ Nat.Prime (30 * k + 29) := by
sorry

end NUMINAMATH_CALUDE_prime_ends_of_sequence_l12_1257


namespace NUMINAMATH_CALUDE_ball_max_height_l12_1216

/-- The height function of the ball's trajectory -/
def f (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 161 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l12_1216


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l12_1274

/-- Given two lines AB and CD, where:
    - AB passes through points A(-2,m) and B(m,4)
    - CD passes through points C(m+1,1) and D(m,3)
    - AB is parallel to CD
    Prove that m = -8 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  let C : ℝ × ℝ := (m + 1, 1)
  let D : ℝ × ℝ := (m, 3)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_CD := (D.2 - C.2) / (D.1 - C.1)
  slope_AB = slope_CD →
  m = -8 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l12_1274


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l12_1215

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 16) :
  let r := d / 2
  let chord_length := r * Real.sqrt 2
  chord_length ^ 2 = 128 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l12_1215


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l12_1275

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_kayak_ratio : ℚ
  total_revenue : ℕ

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (info : RentalInfo) : ℕ :=
  let canoes := (info.total_revenue / (3 * info.canoe_cost + 2 * info.kayak_cost)) * 3
  let kayaks := (info.total_revenue / (3 * info.canoe_cost + 2 * info.kayak_cost)) * 2
  canoes - kayaks

/-- Theorem stating the difference between canoes and kayaks rented --/
theorem canoe_kayak_difference_is_five (info : RentalInfo)
  (h1 : info.canoe_cost = 15)
  (h2 : info.kayak_cost = 18)
  (h3 : info.canoe_kayak_ratio = 3/2)
  (h4 : info.total_revenue = 405) :
  canoe_kayak_difference info = 5 := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l12_1275


namespace NUMINAMATH_CALUDE_sum_of_roots_l12_1292

-- Define the quadratic equation
def quadratic (x h : ℝ) : ℝ := 6 * x^2 - 5 * h * x - 4 * h

-- Define the roots
def roots (h : ℝ) : Set ℝ := {x | quadratic x h = 0}

-- Theorem statement
theorem sum_of_roots (h : ℝ) (hne : ∃ (x₁ x₂ : ℝ), x₁ ∈ roots h ∧ x₂ ∈ roots h ∧ x₁ ≠ x₂) :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots h ∧ x₂ ∈ roots h ∧ x₁ + x₂ = 5 * h / 6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l12_1292


namespace NUMINAMATH_CALUDE_smallest_odd_with_24_divisors_l12_1262

/-- The number of divisors of a positive integer -/
def numDivisors (n : ℕ+) : ℕ := sorry

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem smallest_odd_with_24_divisors :
  ∃ (n : ℕ+),
    isOdd n.val ∧
    numDivisors n = 24 ∧
    (∀ (m : ℕ+), isOdd m.val ∧ numDivisors m = 24 → n ≤ m) ∧
    n = 3465 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_with_24_divisors_l12_1262


namespace NUMINAMATH_CALUDE_unique_a_value_l12_1271

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x : ℝ | x^2 - 4*x + 3 ≥ 0}

-- Define the proposition p and q
def p (a : ℝ) : Prop := ∃ x, x ∈ A a
def q : Prop := ∃ x, x ∈ B

-- Define the negation of q
def not_q : Prop := ∃ x, x ∉ B

-- Theorem statement
theorem unique_a_value : 
  ∃! a : ℝ, (∀ x : ℝ, not_q → p a) ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l12_1271


namespace NUMINAMATH_CALUDE_candy_sales_theorem_l12_1235

-- Define the candy sales for each week
structure CandySales :=
  (week1_initial : ℕ)
  (week1_monday : ℕ)
  (week1_tuesday : ℕ)
  (week1_wednesday_left : ℕ)
  (week2_initial : ℕ)
  (week2_monday : ℕ)
  (week2_tuesday : ℕ)
  (week2_wednesday : ℕ)
  (week2_thursday : ℕ)
  (week2_friday : ℕ)
  (week3_initial : ℕ)
  (week3_highest : ℕ)

-- Define the theorem
theorem candy_sales_theorem (sales : CandySales) 
  (h1 : sales.week1_initial = 80)
  (h2 : sales.week1_monday = 15)
  (h3 : sales.week1_tuesday = 2 * sales.week1_monday)
  (h4 : sales.week1_wednesday_left = 7)
  (h5 : sales.week2_initial = 100)
  (h6 : sales.week2_monday = 12)
  (h7 : sales.week2_tuesday = 18)
  (h8 : sales.week2_wednesday = 20)
  (h9 : sales.week2_thursday = 11)
  (h10 : sales.week2_friday = 25)
  (h11 : sales.week3_initial = 120)
  (h12 : sales.week3_highest = 40) :
  (sales.week1_initial - sales.week1_wednesday_left = 73) ∧
  (sales.week2_monday + sales.week2_tuesday + sales.week2_wednesday + sales.week2_thursday + sales.week2_friday = 86) ∧
  (sales.week3_highest = 40) := by
  sorry

end NUMINAMATH_CALUDE_candy_sales_theorem_l12_1235


namespace NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l12_1293

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - t|

-- State the theorem
theorem max_value_implies_t_equals_one :
  ∀ t : ℝ, (∀ x ∈ Set.Icc 0 3, f t x ≤ 2) ∧ (∃ x ∈ Set.Icc 0 3, f t x = 2) → t = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l12_1293


namespace NUMINAMATH_CALUDE_dancer_count_l12_1208

theorem dancer_count (n : ℕ) : 
  (200 ≤ n ∧ n ≤ 300) ∧
  (∃ k : ℕ, n + 5 = 12 * k) ∧
  (∃ m : ℕ, n + 5 = 10 * m) →
  n = 235 ∨ n = 295 := by
sorry

end NUMINAMATH_CALUDE_dancer_count_l12_1208


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l12_1255

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 100 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l12_1255


namespace NUMINAMATH_CALUDE_book_selling_price_l12_1276

theorem book_selling_price (cost_price selling_price : ℝ) : 
  cost_price = 200 →
  selling_price - cost_price = (340 - cost_price) + 0.05 * cost_price →
  selling_price = 350 := by
sorry

end NUMINAMATH_CALUDE_book_selling_price_l12_1276


namespace NUMINAMATH_CALUDE_sqrt_product_equals_150_sqrt_3_l12_1207

theorem sqrt_product_equals_150_sqrt_3 : 
  Real.sqrt 75 * Real.sqrt 45 * Real.sqrt 20 = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_150_sqrt_3_l12_1207


namespace NUMINAMATH_CALUDE_john_payment_is_8000_l12_1298

/-- Calculates John's payment for lawyer fees --/
def johnPayment (upfrontPayment : ℕ) (hourlyRate : ℕ) (courtTime : ℕ) : ℕ :=
  let totalTime := courtTime + 2 * courtTime
  let totalFee := upfrontPayment + hourlyRate * totalTime
  totalFee / 2

/-- Theorem: John's payment for lawyer fees is $8,000 --/
theorem john_payment_is_8000 :
  johnPayment 1000 100 50 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_is_8000_l12_1298


namespace NUMINAMATH_CALUDE_pyramid_inequality_l12_1225

/-- A triangular pyramid with vertex O and base ABC -/
structure TriangularPyramid (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  O : V
  A : V
  B : V
  C : V

/-- The area of a triangle -/
def triangleArea (A B C : V) [NormedAddCommGroup V] [InnerProductSpace ℝ V] : ℝ :=
  sorry

/-- Statement of the theorem -/
theorem pyramid_inequality (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (pyramid : TriangularPyramid V) (M : V) :
  let S_ABC := triangleArea pyramid.A pyramid.B pyramid.C
  let S_MBC := triangleArea M pyramid.B pyramid.C
  let S_MAC := triangleArea M pyramid.A pyramid.C
  let S_MAB := triangleArea M pyramid.A pyramid.B
  ‖pyramid.O - M‖ * S_ABC ≤ 
    ‖pyramid.O - pyramid.A‖ * S_MBC + 
    ‖pyramid.O - pyramid.B‖ * S_MAC + 
    ‖pyramid.O - pyramid.C‖ * S_MAB :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_inequality_l12_1225


namespace NUMINAMATH_CALUDE_point_M_coordinates_l12_1252

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem point_M_coordinates :
  ∃ (x y : ℝ), 
    curve y = curve x ∧ 
    curve_derivative x = -4 ∧ 
    x = -1 ∧ 
    y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l12_1252


namespace NUMINAMATH_CALUDE_q_polynomial_expression_l12_1204

theorem q_polynomial_expression (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 8 * x^2) = (5 * x^4 + 18 * x^3 + 20 * x^2 + 2)) →
  (∀ x, q x = -2 * x^6 + x^4 + 18 * x^3 + 12 * x^2 + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_expression_l12_1204


namespace NUMINAMATH_CALUDE_find_e_l12_1291

theorem find_e : ∃ e : ℕ, (1/5 : ℝ)^e * (1/4 : ℝ)^18 = 1 / (2 * 10^35) ∧ e = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_e_l12_1291


namespace NUMINAMATH_CALUDE_common_difference_is_two_l12_1209

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  fifth_term : a 5 = 6
  third_term : a 3 = 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) :
  commonDifference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l12_1209


namespace NUMINAMATH_CALUDE_total_dress_designs_l12_1239

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve types available -/
def num_sleeve_types : ℕ := 3

/-- Theorem stating the total number of possible dress designs -/
theorem total_dress_designs :
  num_colors * num_patterns * num_sleeve_types = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l12_1239


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l12_1297

theorem sum_of_coefficients (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l12_1297


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l12_1219

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : x + y = 4) : 
  x^2 * y + x * y^2 = -8 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l12_1219


namespace NUMINAMATH_CALUDE_unique_a_value_l12_1218

def A (a : ℝ) : Set ℝ := {1, a}
def B : Set ℝ := {1, 3}

theorem unique_a_value (a : ℝ) (h : A a ∪ B = {1, 2, 3}) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l12_1218


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l12_1267

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l12_1267


namespace NUMINAMATH_CALUDE_f_difference_l12_1201

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as the sum of all positive divisors of n divided by n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(640) - f(320) = 3/320 -/
theorem f_difference : f 640 - f 320 = 3 / 320 := by sorry

end NUMINAMATH_CALUDE_f_difference_l12_1201


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l12_1230

theorem regular_polygon_properties :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
  n > 2 →
  interior_angle - exterior_angle = 90 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = n * interior_angle →
  (n - 2) * 180 = 1080 ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l12_1230


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l12_1244

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

-- State the theorem
theorem arithmetic_sequence_formula 
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h1 : a 3 + a 4 = 4)
  (h2 : a 5 + a 7 = 6) :
  ∃ C : ℚ, ∀ n : ℕ, a n = (2 * n + C) / 5 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l12_1244


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l12_1237

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 - I) / I ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l12_1237


namespace NUMINAMATH_CALUDE_jelly_bean_count_l12_1210

/-- The number of jelly beans in jar X -/
def jarX (total : ℕ) (y : ℕ) : ℕ := 3 * y - 400

/-- The number of jelly beans in jar Y -/
def jarY (total : ℕ) (x : ℕ) : ℕ := total - x

theorem jelly_bean_count (total : ℕ) (h : total = 1200) :
  ∃ y : ℕ, jarX total y + jarY total (jarX total y) = total ∧ jarX total y = 800 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_count_l12_1210


namespace NUMINAMATH_CALUDE_original_denominator_proof_l12_1290

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →
  (2 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 20 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l12_1290


namespace NUMINAMATH_CALUDE_contest_awards_l12_1205

theorem contest_awards (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  (n.factorial / (n - k).factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_contest_awards_l12_1205


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l12_1231

/-- Given an arithmetic sequence {a_n} where a_5 + a_6 + a_7 = 15,
    prove that the sum (a_3 + a_4 + ... + a_9) is equal to 35. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 + a 7 = 15 →                            -- given condition
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=    -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l12_1231


namespace NUMINAMATH_CALUDE_h_has_two_roots_l12_1282

/-- The function f(x) = 2x -/
def f (x : ℝ) : ℝ := 2 * x

/-- The function g(x) = 3 - x^2 -/
def g (x : ℝ) : ℝ := 3 - x^2

/-- The function h(x) = f(x) - g(x) -/
def h (x : ℝ) : ℝ := f x - g x

/-- The theorem stating that h(x) has exactly two distinct real roots -/
theorem h_has_two_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 ∧ ∀ x, h x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_h_has_two_roots_l12_1282


namespace NUMINAMATH_CALUDE_union_equals_S_l12_1227

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem union_equals_S : S ∪ T = S := by sorry

end NUMINAMATH_CALUDE_union_equals_S_l12_1227


namespace NUMINAMATH_CALUDE_discount_sales_income_increase_l12_1265

/-- Proves that a 10% discount with 15% increase in sales volume results in 3.5% increase in gross income -/
theorem discount_sales_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : sales_increase_rate = 0.15) : 
  let new_price := original_price * (1 - discount_rate)
  let new_quantity := original_quantity * (1 + sales_increase_rate)
  let original_income := original_price * original_quantity
  let new_income := new_price * new_quantity
  (new_income - original_income) / original_income = 0.035 := by
sorry

end NUMINAMATH_CALUDE_discount_sales_income_increase_l12_1265


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l12_1270

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 220 (-5) n = 35 ∧ n = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l12_1270


namespace NUMINAMATH_CALUDE_trapezoid_point_distance_l12_1261

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Returns the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- Returns the line passing through two points -/
def lineThroughPoints (p1 p2 : Point) : Line :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Recursively defines points A_n and B_n -/
def definePoints (trap : Trapezoid) (E : Point) (n : ℕ) : Point × Point :=
  match n with
  | 0 => (trap.A, trap.B)
  | n+1 =>
    let (A_n, _) := definePoints trap E n
    let B_next := intersectionPoint (lineThroughPoints A_n trap.C) (lineThroughPoints trap.B trap.D)
    let A_next := intersectionPoint (lineThroughPoints E B_next) (lineThroughPoints trap.A trap.B)
    (A_next, B_next)

/-- The main theorem to be proved -/
theorem trapezoid_point_distance (trap : Trapezoid) (E : Point) (n : ℕ) :
  let (A_n, _) := definePoints trap E n
  distance A_n trap.B = distance trap.A trap.B / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_point_distance_l12_1261


namespace NUMINAMATH_CALUDE_kylie_coins_left_l12_1249

/-- Calculates the number of coins Kylie is left with after various transactions --/
def coins_left (piggy_bank : ℕ) (from_brother : ℕ) (from_father : ℕ) (given_to_friend : ℕ) : ℕ :=
  piggy_bank + from_brother + from_father - given_to_friend

/-- Theorem stating that Kylie is left with 15 coins --/
theorem kylie_coins_left : 
  coins_left 15 13 8 21 = 15 := by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_left_l12_1249


namespace NUMINAMATH_CALUDE_sector_max_area_l12_1220

/-- Given a circular sector with perimeter 40 units, prove that the area is maximized
    when the central angle is 2 radians and the maximum area is 100 square units. -/
theorem sector_max_area (R : ℝ) (α : ℝ) (h : R * α + 2 * R = 40) :
  (R * α * R / 2 ≤ 100) ∧
  (R * α * R / 2 = 100 ↔ α = 2 ∧ R = 10) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l12_1220


namespace NUMINAMATH_CALUDE_inequality_solution_set_l12_1222

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l12_1222


namespace NUMINAMATH_CALUDE_unit_vector_magnitude_is_one_l12_1229

variable {V : Type*} [NormedAddCommGroup V]

/-- The magnitude of a unit vector is equal to 1. -/
theorem unit_vector_magnitude_is_one (v : V) (h : ‖v‖ = 1) : ‖v‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_vector_magnitude_is_one_l12_1229


namespace NUMINAMATH_CALUDE_peggy_needs_825_stamps_l12_1281

/-- The number of stamps Peggy needs to add to have as many as Bert -/
def stamps_to_add (peggy_stamps : ℕ) : ℕ :=
  4 * (3 * peggy_stamps) - peggy_stamps

/-- Theorem stating that Peggy needs to add 825 stamps to have as many as Bert -/
theorem peggy_needs_825_stamps : stamps_to_add 75 = 825 := by
  sorry

end NUMINAMATH_CALUDE_peggy_needs_825_stamps_l12_1281


namespace NUMINAMATH_CALUDE_power_function_through_point_l12_1253

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the condition that f passes through (9, 3)
def passesThroughPoint (f : ℝ → ℝ) : Prop :=
  f 9 = 3

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) (h2 : passesThroughPoint f) : f 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l12_1253


namespace NUMINAMATH_CALUDE_max_triangles_hit_five_times_is_25_l12_1254

/-- Represents a triangular target divided into smaller equilateral triangles -/
structure Target where
  total_triangles : Nat
  mk_valid : total_triangles = 100

/-- Represents a shot by the sniper -/
structure Shot where
  aimed_triangle : Nat
  hit_triangle : Nat
  mk_valid : hit_triangle = aimed_triangle ∨ 
             hit_triangle = aimed_triangle - 1 ∨ 
             hit_triangle = aimed_triangle + 1

/-- Represents the result of multiple shots -/
def ShotResult := Nat → Nat

/-- The maximum number of triangles that can be hit exactly five times -/
def max_triangles_hit_five_times (t : Target) (shots : List Shot) : Nat :=
  sorry

/-- Theorem stating the maximum number of triangles hit exactly five times -/
theorem max_triangles_hit_five_times_is_25 (t : Target) :
  ∃ (shots : List Shot), max_triangles_hit_five_times t shots = 25 ∧
  ∀ (other_shots : List Shot), max_triangles_hit_five_times t other_shots ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_max_triangles_hit_five_times_is_25_l12_1254


namespace NUMINAMATH_CALUDE_investment_interest_rate_l12_1221

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (rate1 rate2 : ℝ) 
  (h1 : total_investment = 6000)
  (h2 : rate1 = 0.05)
  (h3 : rate2 = 0.07)
  (h4 : ∃ (part1 part2 : ℝ), 
    part1 + part2 = total_investment ∧ 
    part1 * rate1 = part2 * rate2) :
  (rate1 * (total_investment - (rate2 * total_investment) / (rate1 + rate2)) + 
   rate2 * ((rate1 * total_investment) / (rate1 + rate2))) / total_investment = 0.05833 :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l12_1221


namespace NUMINAMATH_CALUDE_min_distance_complex_circles_l12_1266

theorem min_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 5 - 6*I) = 4) :
  ∃ (m : ℝ), m = Real.sqrt 149 - 6 ∧ ∀ (z' w' : ℂ), 
    Complex.abs (z' + 2 + 4*I) = 2 → 
    Complex.abs (w' - 5 - 6*I) = 4 → 
    Complex.abs (z' - w') ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_circles_l12_1266


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l12_1268

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | |x - 1| < 2}

theorem M_intersect_N_equals_zero : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l12_1268


namespace NUMINAMATH_CALUDE_copper_in_mixture_l12_1283

/-- Given a mixture of zinc and copper in the ratio 9:11 with a total weight of 60 kg,
    the amount of copper in the mixture is 33 kg. -/
theorem copper_in_mixture (zinc_ratio : ℕ) (copper_ratio : ℕ) (total_weight : ℝ) :
  zinc_ratio = 9 →
  copper_ratio = 11 →
  total_weight = 60 →
  (copper_ratio : ℝ) / ((zinc_ratio : ℝ) + (copper_ratio : ℝ)) * total_weight = 33 :=
by sorry

end NUMINAMATH_CALUDE_copper_in_mixture_l12_1283


namespace NUMINAMATH_CALUDE_prove_c_value_l12_1295

-- Define the variables
variable (c k x y z : ℝ)

-- Define the conditions
axiom model : y = c * Real.exp (k * x)
axiom log_transform : z = Real.log y
axiom regression : z = 0.4 * x + 2

-- Theorem to prove
theorem prove_c_value : c = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_prove_c_value_l12_1295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l12_1284

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: The common difference of an arithmetic sequence is 1, 
    given S_3 = 6 and a_1 = 1 -/
theorem arithmetic_sequence_common_difference :
  ∀ d : ℚ, S 3 1 d = 6 → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l12_1284


namespace NUMINAMATH_CALUDE_triangle_side_length_l12_1242

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 - c^2 = 2*b →
  Real.sin B = 4 * Real.cos A * Real.sin C →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l12_1242


namespace NUMINAMATH_CALUDE_beavers_working_on_home_l12_1233

/-- The number of beavers initially working on their home -/
def initial_beavers : ℕ := 2

/-- The number of beavers that went for a swim -/
def swimming_beavers : ℕ := 1

/-- The number of beavers still working on their home -/
def remaining_beavers : ℕ := initial_beavers - swimming_beavers

theorem beavers_working_on_home : remaining_beavers = 1 := by
  sorry

end NUMINAMATH_CALUDE_beavers_working_on_home_l12_1233


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l12_1214

theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l12_1214


namespace NUMINAMATH_CALUDE_symmetric_point_example_l12_1288

/-- Given a point A and a point of symmetry, find the symmetric point -/
def symmetric_point (A : ℝ × ℝ × ℝ) (sym : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (2 * sym.1 - A.1, 2 * sym.2.1 - A.2.1, 2 * sym.2.2 - A.2.2)

/-- The theorem states that the symmetric point of A(3, -2, 4) with respect to (0, 1, -3) is (-3, 4, -10) -/
theorem symmetric_point_example : 
  symmetric_point (3, -2, 4) (0, 1, -3) = (-3, 4, -10) := by
  sorry

#eval symmetric_point (3, -2, 4) (0, 1, -3)

end NUMINAMATH_CALUDE_symmetric_point_example_l12_1288


namespace NUMINAMATH_CALUDE_type_of_2004_least_type_B_after_2004_l12_1286

/-- Represents the type of a number in the game -/
inductive NumberType
| A
| B

/-- Determines if a number is of type A or B in the game -/
def numberType (n : ℕ) : NumberType :=
  sorry

/-- Theorem stating that 2004 is of type A -/
theorem type_of_2004 : numberType 2004 = NumberType.A :=
  sorry

/-- Theorem stating that 2048 is the least number greater than 2004 of type B -/
theorem least_type_B_after_2004 :
  (numberType 2048 = NumberType.B) ∧
  (∀ m : ℕ, 2004 < m → m < 2048 → numberType m = NumberType.A) :=
  sorry

end NUMINAMATH_CALUDE_type_of_2004_least_type_B_after_2004_l12_1286


namespace NUMINAMATH_CALUDE_expand_and_simplify_l12_1211

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l12_1211


namespace NUMINAMATH_CALUDE_line_circle_intersection_l12_1226

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 2 ∧ (x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l12_1226


namespace NUMINAMATH_CALUDE_polynomial_factorization_l12_1243

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) = 
  (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l12_1243


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l12_1206

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l12_1206


namespace NUMINAMATH_CALUDE_constant_function_satisfies_inequality_l12_1264

theorem constant_function_satisfies_inequality :
  ∀ f : ℕ → ℝ,
  (∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 → f (a * c) + f (b * c) - f c * f (a * b) ≥ 1) →
  (∀ x : ℕ, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_constant_function_satisfies_inequality_l12_1264


namespace NUMINAMATH_CALUDE_perp_condition_relationship_l12_1260

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Predicate indicating if a line is perpendicular to a plane -/
def perp_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate indicating if a line is perpendicular to countless lines in a plane -/
def perp_to_countless_lines (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Theorem stating the relationship between the two conditions -/
theorem perp_condition_relationship :
  (∀ (l : Line3D) (α : Plane3D), perp_to_plane l α → perp_to_countless_lines l α) ∧
  (∃ (l : Line3D) (α : Plane3D), perp_to_countless_lines l α ∧ ¬perp_to_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perp_condition_relationship_l12_1260


namespace NUMINAMATH_CALUDE_rectangle_count_is_297_l12_1238

/-- Represents a grid with a hole in the middle -/
structure Grid :=
  (size : ℕ)
  (hole_x : ℕ)
  (hole_y : ℕ)

/-- Counts the number of non-degenerate rectangles in a grid with a hole -/
def count_rectangles (g : Grid) : ℕ :=
  sorry

/-- The specific 7x7 grid with a hole at (4,4) -/
def specific_grid : Grid :=
  { size := 7, hole_x := 4, hole_y := 4 }

/-- Theorem stating that the number of non-degenerate rectangles in the specific grid is 297 -/
theorem rectangle_count_is_297 : count_rectangles specific_grid = 297 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_count_is_297_l12_1238


namespace NUMINAMATH_CALUDE_factorial_ratio_100_98_l12_1256

/-- Factorial function -/
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The ratio of 100! to 98! is 9900 -/
theorem factorial_ratio_100_98 : factorial 100 / factorial 98 = 9900 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_100_98_l12_1256


namespace NUMINAMATH_CALUDE_autograph_distribution_theorem_l12_1273

/-- Represents a set of autographs from 11 players -/
def Autographs := Fin 11 → Bool

/-- The set of all residents -/
def Residents := Fin 1111

/-- Distribution of autographs to residents -/
def AutographDistribution := Residents → Autographs

theorem autograph_distribution_theorem (d : AutographDistribution) 
  (h : ∀ (i j : Residents), i ≠ j → d i ≠ d j) :
  ∃ (i j : Residents), i ≠ j ∧ 
    (∀ (k : Fin 11), (d i k = true ∧ d j k = false) ∨ (d i k = false ∧ d j k = true)) :=
sorry

end NUMINAMATH_CALUDE_autograph_distribution_theorem_l12_1273


namespace NUMINAMATH_CALUDE_room_area_difference_l12_1279

-- Define the dimensions of the rooms
def largest_room_width : ℕ := 45
def largest_room_length : ℕ := 30
def smallest_room_width : ℕ := 15
def smallest_room_length : ℕ := 8

-- Define the function to calculate the area of a rectangular room
def room_area (width : ℕ) (length : ℕ) : ℕ := width * length

-- Theorem statement
theorem room_area_difference :
  room_area largest_room_width largest_room_length - 
  room_area smallest_room_width smallest_room_length = 1230 := by
  sorry

end NUMINAMATH_CALUDE_room_area_difference_l12_1279


namespace NUMINAMATH_CALUDE_james_weight_vest_savings_l12_1294

/-- The savings James makes by buying a separate vest and plates instead of a discounted 200-pound weight vest -/
theorem james_weight_vest_savings 
  (separate_vest_cost : ℝ) 
  (plate_weight : ℝ) 
  (cost_per_pound : ℝ) 
  (full_vest_cost : ℝ) 
  (discount : ℝ)
  (h1 : separate_vest_cost = 250)
  (h2 : plate_weight = 200)
  (h3 : cost_per_pound = 1.2)
  (h4 : full_vest_cost = 700)
  (h5 : discount = 100) :
  full_vest_cost - discount - (separate_vest_cost + plate_weight * cost_per_pound) = 110 := by
  sorry

end NUMINAMATH_CALUDE_james_weight_vest_savings_l12_1294


namespace NUMINAMATH_CALUDE_range_of_a_l12_1223

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1/2 ≤ x ∧ x ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ∃ x, ¬p x ∧ q x a

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l12_1223


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l12_1246

theorem vegetable_planting_methods (n m : ℕ) (hn : n = 4) (hm : m = 3) :
  (n.choose m) * (m.factorial) = 24 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l12_1246


namespace NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_given_line_l12_1203

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y = 0

/-- The equation of the given line -/
def given_line (x y : ℝ) : Prop :=
  2*x - y = 0

/-- The equation of the line we need to prove -/
def target_line (x y : ℝ) : Prop :=
  2*x - y - 3 = 0

/-- Theorem stating that the line passing through the center of the circle
    and parallel to the given line has the equation 2x - y - 3 = 0 -/
theorem line_through_circle_center_parallel_to_given_line :
  ∃ (cx cy : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = cx^2 + cy^2) ∧
    (given_line cx cy → target_line cx cy) ∧
    (∀ (x y : ℝ), given_line x y → ∃ (k : ℝ), target_line (x + k) (y + 2*k)) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_given_line_l12_1203


namespace NUMINAMATH_CALUDE_circle_sum_zero_l12_1272

theorem circle_sum_zero (a : Fin 55 → ℤ) 
  (h : ∀ i : Fin 55, a i = a (i - 1) + a (i + 1)) : 
  ∀ i : Fin 55, a i = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_zero_l12_1272


namespace NUMINAMATH_CALUDE_constant_term_value_l12_1200

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of the first two binomial coefficients equals 10 -/
def sum_first_two_coefficients (n : ℕ) : Prop :=
  binomial n 0 + binomial n 1 = 10

/-- The constant term in the expansion -/
def constant_term (n : ℕ) : ℕ :=
  2^(n - 6) * binomial n 6

theorem constant_term_value (n : ℕ) :
  sum_first_two_coefficients n → constant_term n = 672 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l12_1200


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l12_1289

theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (π/6) (π/3), Monotone (fun x => (a - Real.sin x) / Real.cos x)) →
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l12_1289


namespace NUMINAMATH_CALUDE_cut_cube_volume_l12_1299

/-- A polyhedron formed by cutting off the eight corners of a cube -/
structure CutCube where
  /-- The polyhedron has 6 octagonal faces -/
  octagonal_faces : Nat
  /-- The polyhedron has 8 triangular faces -/
  triangular_faces : Nat
  /-- All edges of the polyhedron have length 2 -/
  edge_length : ℝ

/-- The volume of the CutCube -/
def volume (c : CutCube) : ℝ := sorry

/-- Theorem stating the volume of the CutCube -/
theorem cut_cube_volume (c : CutCube) 
  (h1 : c.octagonal_faces = 6)
  (h2 : c.triangular_faces = 8)
  (h3 : c.edge_length = 2) :
  volume c = 56 + 112 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_cut_cube_volume_l12_1299


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l12_1224

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x + 4 > 0) ↔ (x > -2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l12_1224


namespace NUMINAMATH_CALUDE_right_triangle_area_l12_1277

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters, its area is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) : 
  a = 13 → -- hypotenuse is 13 meters
  b = 5 → -- one side is 5 meters
  c^2 + b^2 = a^2 → -- Pythagorean theorem (right triangle condition)
  (1/2 : ℝ) * b * c = 30 := by -- area formula
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l12_1277


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l12_1263

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l12_1263


namespace NUMINAMATH_CALUDE_binomial_17_5_l12_1258

theorem binomial_17_5 (h1 : Nat.choose 15 3 = 455)
                      (h2 : Nat.choose 15 4 = 1365)
                      (h3 : Nat.choose 15 5 = 3003) :
  Nat.choose 17 5 = 6188 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_5_l12_1258


namespace NUMINAMATH_CALUDE_kelsey_watched_160_l12_1287

/-- The number of videos watched by three friends satisfies the given conditions -/
structure VideoWatching where
  total : ℕ
  kelsey_more_than_ekon : ℕ
  uma_more_than_ekon : ℕ
  h_total : total = 411
  h_kelsey_more : kelsey_more_than_ekon = 43
  h_uma_more : uma_more_than_ekon = 17

/-- Given the conditions, prove that Kelsey watched 160 videos -/
theorem kelsey_watched_160 (vw : VideoWatching) : 
  ∃ (ekon uma kelsey : ℕ), 
    ekon + uma + kelsey = vw.total ∧ 
    kelsey = ekon + vw.kelsey_more_than_ekon ∧ 
    uma = ekon + vw.uma_more_than_ekon ∧
    kelsey = 160 := by
  sorry

end NUMINAMATH_CALUDE_kelsey_watched_160_l12_1287


namespace NUMINAMATH_CALUDE_school_sections_theorem_l12_1236

/-- The number of sections formed when dividing students into equal groups -/
def number_of_sections (boys girls : Nat) : Nat :=
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls)

/-- Theorem stating that the total number of sections for 408 boys and 216 girls is 26 -/
theorem school_sections_theorem :
  number_of_sections 408 216 = 26 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_theorem_l12_1236


namespace NUMINAMATH_CALUDE_fraction_equality_l12_1217

theorem fraction_equality (x y : ℝ) (h : x / y = 5 / 3) :
  x / (y - x) = -3 / 2 ∧ x / (y - x) ≠ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l12_1217


namespace NUMINAMATH_CALUDE_evaluate_expression_l12_1278

theorem evaluate_expression (x z : ℝ) (hx : x = 2) (hz : z = 1) :
  z * (z - 4 * x) = -7 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l12_1278


namespace NUMINAMATH_CALUDE_linear_equation_solution_l12_1228

theorem linear_equation_solution (a b : ℝ) : 
  (a * (-2) - 3 * b * 3 = 5) → 
  (a * 4 - 3 * b * 1 = 5) → 
  a + b = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l12_1228


namespace NUMINAMATH_CALUDE_binomial_1300_2_l12_1280

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1300_2_l12_1280


namespace NUMINAMATH_CALUDE_travis_annual_cereal_cost_l12_1202

/-- Calculates the annual cereal cost for Travis --/
theorem travis_annual_cereal_cost :
  let box_a_cost : ℝ := 3.50
  let box_b_cost : ℝ := 4.00
  let box_c_cost : ℝ := 5.25
  let box_a_consumption : ℝ := 1
  let box_b_consumption : ℝ := 0.5
  let box_c_consumption : ℝ := 1/3
  let discount_rate : ℝ := 0.1
  let weeks_per_year : ℕ := 52

  let weekly_cost : ℝ := 
    box_a_cost * box_a_consumption + 
    box_b_cost * box_b_consumption + 
    box_c_cost * box_c_consumption

  let discounted_weekly_cost : ℝ := weekly_cost * (1 - discount_rate)

  let annual_cost : ℝ := discounted_weekly_cost * weeks_per_year

  annual_cost = 339.30 := by sorry

end NUMINAMATH_CALUDE_travis_annual_cereal_cost_l12_1202


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l12_1259

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧ ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l12_1259


namespace NUMINAMATH_CALUDE_tax_threshold_value_l12_1240

def tax_calculation (X : ℝ) (I : ℝ) : ℝ := 0.12 * X + 0.20 * (I - X)

theorem tax_threshold_value :
  ∃ (X : ℝ), 
    X = 40000 ∧
    tax_calculation X 56000 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_tax_threshold_value_l12_1240


namespace NUMINAMATH_CALUDE_houses_around_square_l12_1248

/-- The number of houses around the square. -/
def n : ℕ := 32

/-- Maria's count for a given house. -/
def M (k : ℕ) : ℕ := k % n

/-- João's count for a given house. -/
def J (k : ℕ) : ℕ := k % n

/-- Theorem stating the number of houses around the square. -/
theorem houses_around_square :
  (M 5 = J 12) ∧ (J 5 = M 30) → n = 32 := by
  sorry

end NUMINAMATH_CALUDE_houses_around_square_l12_1248


namespace NUMINAMATH_CALUDE_remainder_theorem_l12_1285

/-- The polynomial f(x) = 4x^5 - 9x^4 + 7x^2 - x - 35 -/
def f (x : ℝ) : ℝ := 4 * x^5 - 9 * x^4 + 7 * x^2 - x - 35

/-- The theorem stating that the remainder when f(x) is divided by (x - 2.5) is 45.3125 -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = λ x => (x - 2.5) * q x + 45.3125 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l12_1285


namespace NUMINAMATH_CALUDE_equation_system_solution_l12_1245

theorem equation_system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 1 / (x * y) = x / z + 1)
  (eq2 : 1 / (y * z) = y / x + 1)
  (eq3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l12_1245


namespace NUMINAMATH_CALUDE_max_value_is_16_l12_1247

/-- A function f(x) that is symmetric about x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- Symmetry condition: f(x) = f(-4-x) for all x -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b x = f a b (-4-x)

/-- The maximum value of f(x) is 16 -/
theorem max_value_is_16 (a b : ℝ) (h : is_symmetric a b) :
  ∃ M, M = 16 ∧ ∀ x, f a b x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_is_16_l12_1247


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l12_1269

theorem mean_of_remaining_numbers (numbers : Finset ℕ)
  (h1 : numbers = {1971, 2008, 2101, 2150, 2220, 2300, 2350})
  (subset : Finset ℕ)
  (h2 : subset ⊆ numbers)
  (h3 : Finset.card subset = 5)
  (h4 : (Finset.sum subset id) / 5 = 2164) :
  let remaining := numbers \ subset
  (Finset.sum remaining id) / 2 = 2140 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l12_1269


namespace NUMINAMATH_CALUDE_water_temperature_difference_l12_1250

theorem water_temperature_difference (n : ℕ) : 
  let T_h := (T_c : ℝ) + 64/3
  let T_n := T_h - (1/4)^n * (T_h - T_c)
  (T_h - T_n ≠ 1/2) ∧ (T_h - T_n ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_water_temperature_difference_l12_1250


namespace NUMINAMATH_CALUDE_inequality_proof_l12_1232

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : d > 0) :
  d / c < (d + 4) / (c + 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l12_1232


namespace NUMINAMATH_CALUDE_curve_C₁_and_constant_product_l12_1251

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the curve C₁
def C₁ (x y : ℝ) : Prop :=
  ∀ (x' y' : ℝ), C₂ x' y' → (x - x')^2 + (y - y')^2 ≥ (y + 2)^2

-- Define the line y = -4
def line_y_neg4 (x y : ℝ) : Prop := y = -4

-- Define the tangent lines from a point to C₂
def tangent_to_C₂ (x₀ y₀ x y : ℝ) : Prop :=
  ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ (x₀^2 - 9) * k^2 + 18 * x₀ * k + 72 = 0

-- Theorem statement
theorem curve_C₁_and_constant_product :
  (∀ x y : ℝ, C₁ x y ↔ x^2 = 20 * y) ∧
  (∀ x₀ : ℝ, x₀ ≠ 3 ∧ x₀ ≠ -3 →
    ∀ x₁ x₂ x₃ x₄ : ℝ,
    (∃ y₀, line_y_neg4 x₀ y₀) →
    (∃ y₁, C₁ x₁ y₁ ∧ tangent_to_C₂ x₀ (-4) x₁ y₁) →
    (∃ y₂, C₁ x₂ y₂ ∧ tangent_to_C₂ x₀ (-4) x₂ y₂) →
    (∃ y₃, C₁ x₃ y₃ ∧ tangent_to_C₂ x₀ (-4) x₃ y₃) →
    (∃ y₄, C₁ x₄ y₄ ∧ tangent_to_C₂ x₀ (-4) x₄ y₄) →
    x₁ * x₂ * x₃ * x₄ = 6400) :=
sorry

end NUMINAMATH_CALUDE_curve_C₁_and_constant_product_l12_1251


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l12_1241

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₅ = 2, then a₁ * a₉ = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l12_1241
