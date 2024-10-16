import Mathlib

namespace NUMINAMATH_CALUDE_natural_fraction_condition_l552_55271

theorem natural_fraction_condition (k n : ℕ) :
  (∃ m : ℕ, (7 * k + 15 * n - 1) = m * (3 * k + 4 * n)) ↔
  (∃ a : ℕ, k = 3 * a - 2 ∧ n = 2 * a - 1) :=
by sorry

end NUMINAMATH_CALUDE_natural_fraction_condition_l552_55271


namespace NUMINAMATH_CALUDE_parallel_lines_l552_55285

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle Γ
def circle_gamma (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 8

-- Define point M on Γ
def point_on_gamma (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  circle_gamma x y

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define circle ⊙F
def circle_F (M : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  let (x_m, y_m) := M
  let (x_p, y_p) := P
  (x_p - 1)^2 + y_p^2 = (x_m - 1)^2 + y_m^2

-- Define line l tangent to ⊙F at M
def line_l (M A B : ℝ × ℝ) : Prop :=
  ∃ (k b : ℝ), ∀ (x y : ℝ),
    ((x, y) = M ∨ (x, y) = A ∨ (x, y) = B) → y = k*x + b

-- Define lines l₁ and l₂
def line_l1_l2 (A B Q R : ℝ × ℝ) : Prop :=
  ∃ (k1 b1 k2 b2 : ℝ),
    (∀ (x y : ℝ), ((x, y) = A ∨ (x, y) = Q) → y = k1*x + b1) ∧
    (∀ (x y : ℝ), ((x, y) = B ∨ (x, y) = R) → y = k2*x + b2)

-- Main theorem
theorem parallel_lines
  (M A B Q R : ℝ × ℝ)
  (h_M : point_on_gamma M)
  (h_A : parabola A.1 A.2)
  (h_B : parabola B.1 B.2)
  (h_l : line_l M A B)
  (h_F : circle_F M Q)
  (h_F' : circle_F M R)
  (h_l1_l2 : line_l1_l2 A B Q R) :
  -- Conclusion: l₁ is parallel to l₂
  ∃ (k1 b1 k2 b2 : ℝ),
    (∀ (x y : ℝ), ((x, y) = A ∨ (x, y) = Q) → y = k1*x + b1) ∧
    (∀ (x y : ℝ), ((x, y) = B ∨ (x, y) = R) → y = k2*x + b2) ∧
    k1 = k2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_l552_55285


namespace NUMINAMATH_CALUDE_code_deciphering_probability_l552_55258

theorem code_deciphering_probability 
  (p_a p_b : ℝ) 
  (h_a : p_a = 0.3) 
  (h_b : p_b = 0.3) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p_a) * (1 - p_b) = 0.51 :=
sorry

end NUMINAMATH_CALUDE_code_deciphering_probability_l552_55258


namespace NUMINAMATH_CALUDE_sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x_l552_55272

theorem sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x :
  ∀ x : ℝ, x ≤ 0 → Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x_l552_55272


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l552_55279

theorem largest_angle_in_triangle (a b y : ℝ) : 
  a = 60 ∧ b = 70 ∧ a + b + y = 180 → 
  max a (max b y) = 70 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l552_55279


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l552_55209

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- The probability of not having two consecutive 1s in a sequence of length 15 -/
theorem probability_no_consecutive_ones : 
  (validSequences 15 : ℚ) / (totalSequences 15 : ℚ) = 1597 / 32768 := by
  sorry

#eval validSequences 15
#eval totalSequences 15

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l552_55209


namespace NUMINAMATH_CALUDE_reciprocal_operations_l552_55207

theorem reciprocal_operations : ∃! n : ℕ, n = 2 ∧ 
  (¬ (1 / 4 + 1 / 8 = 1 / 12)) ∧
  (¬ (1 / 8 - 1 / 3 = 1 / 5)) ∧
  ((1 / 3) * (1 / 9) = 1 / 27) ∧
  ((1 / 15) / (1 / 3) = 1 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_reciprocal_operations_l552_55207


namespace NUMINAMATH_CALUDE_students_just_passed_l552_55282

theorem students_just_passed (total_students : ℕ) 
  (first_division_percentage : ℚ) (second_division_percentage : ℚ) : 
  total_students = 300 →
  first_division_percentage = 30 / 100 →
  second_division_percentage = 54 / 100 →
  (total_students : ℚ) * (1 - first_division_percentage - second_division_percentage) = 48 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l552_55282


namespace NUMINAMATH_CALUDE_range_of_m_for_P_and_not_Q_l552_55269

/-- The function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m+6)*x + 1

/-- Predicate P: 2 ≤ m ≤ 8 -/
def P (m : ℝ) : Prop := 2 ≤ m ∧ m ≤ 8

/-- Predicate Q: f has both a maximum and a minimum value -/
def Q (m : ℝ) : Prop := ∃ (a b : ℝ), ∀ (x : ℝ), f m a ≤ f m x ∧ f m x ≤ f m b

/-- The range of m for which P ∩ ¬Q is true is [2, 6] -/
theorem range_of_m_for_P_and_not_Q :
  {m : ℝ | P m ∧ ¬Q m} = {m : ℝ | 2 ≤ m ∧ m ≤ 6} := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_P_and_not_Q_l552_55269


namespace NUMINAMATH_CALUDE_adjacent_complex_numbers_max_sum_squares_l552_55237

theorem adjacent_complex_numbers_max_sum_squares :
  ∀ (a b : ℝ),
  let z1 : ℂ := a + Complex.I * Real.sqrt 3
  let z2 : ℂ := 1 + Complex.I * b
  Complex.abs (z1 - z2) = 1 →
  ∃ (max : ℝ), max = 9 ∧ a^2 + b^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_adjacent_complex_numbers_max_sum_squares_l552_55237


namespace NUMINAMATH_CALUDE_factor_expression_l552_55262

theorem factor_expression (x : ℝ) : 5*x*(x-4) + 6*(x-4) = (x-4)*(5*x+6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l552_55262


namespace NUMINAMATH_CALUDE_smallest_y_squared_l552_55201

/-- An isosceles trapezoid with a tangent circle --/
structure IsoscelesTrapezoidWithCircle where
  -- The length of the longer base
  pq : ℝ
  -- The length of the shorter base
  rs : ℝ
  -- The length of the equal sides
  y : ℝ
  -- Assumption that pq > rs
  h_pq_gt_rs : pq > rs

/-- The theorem stating the smallest possible value of y^2 --/
theorem smallest_y_squared (t : IsoscelesTrapezoidWithCircle) 
  (h_pq : t.pq = 120) (h_rs : t.rs = 25) :
  ∃ (n : ℝ), n^2 = 4350 ∧ ∀ (y : ℝ), y ≥ n → 
  ∃ (c : IsoscelesTrapezoidWithCircle), c.pq = 120 ∧ c.rs = 25 ∧ c.y = y :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_squared_l552_55201


namespace NUMINAMATH_CALUDE_bicycle_course_remaining_distance_l552_55288

/-- Calculates the remaining distance of a bicycle course after Yoongi's travel -/
def remaining_distance (total_length : ℝ) (first_distance : ℝ) (second_distance : ℝ) : ℝ :=
  total_length * 1000 - (first_distance * 1000 + second_distance)

/-- Theorem: Given a bicycle course of 10.5 km, if Yoongi travels 1.5 km and then 3730 m, 
    the remaining distance is 5270 m -/
theorem bicycle_course_remaining_distance :
  remaining_distance 10.5 1.5 3730 = 5270 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_course_remaining_distance_l552_55288


namespace NUMINAMATH_CALUDE_area_eq_xy_l552_55235

/-- A right-angled triangle with an inscribed circle. -/
structure RightTriangleWithIncircle where
  /-- The length of one segment of the hypotenuse. -/
  x : ℝ
  /-- The length of the other segment of the hypotenuse. -/
  y : ℝ
  /-- The radius of the inscribed circle. -/
  r : ℝ
  /-- x and y are positive -/
  x_pos : 0 < x
  y_pos : 0 < y
  /-- r is positive -/
  r_pos : 0 < r

/-- The area of a right-angled triangle with an inscribed circle. -/
def area (t : RightTriangleWithIncircle) : ℝ :=
  t.x * t.y

/-- Theorem: The area of a right-angled triangle with an inscribed circle
    touching the hypotenuse at a point dividing it into segments of lengths x and y
    is equal to x * y. -/
theorem area_eq_xy (t : RightTriangleWithIncircle) : area t = t.x * t.y := by
  sorry

end NUMINAMATH_CALUDE_area_eq_xy_l552_55235


namespace NUMINAMATH_CALUDE_sequence_inequality_l552_55227

theorem sequence_inequality (n : ℕ) : n / (n + 2) < (n + 1) / (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l552_55227


namespace NUMINAMATH_CALUDE_number_ratio_l552_55203

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l552_55203


namespace NUMINAMATH_CALUDE_dennis_loose_coins_l552_55246

def loose_coins_problem (initial_amount : ℕ) (shirt_cost : ℕ) (bill_value : ℕ) (num_bills : ℕ) : Prop :=
  let total_change := initial_amount - shirt_cost
  let bills_amount := bill_value * num_bills
  let loose_coins := total_change - bills_amount
  loose_coins = 3

theorem dennis_loose_coins : 
  loose_coins_problem 50 27 10 2 := by
  sorry

end NUMINAMATH_CALUDE_dennis_loose_coins_l552_55246


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l552_55299

-- Define the angles
def angle_A : ℝ := 35
def angle_B : ℝ := 80
def angle_C : ℝ := 30

-- Define the hexagon
def is_hexagon (x y : ℝ) : Prop :=
  angle_A + angle_B + (360 - x) + 90 + 60 + y = 720

-- Theorem statement
theorem hexagon_angle_sum (x y : ℝ) (h : is_hexagon x y) : x + y = 95 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l552_55299


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l552_55267

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l552_55267


namespace NUMINAMATH_CALUDE_total_folded_sheets_l552_55211

theorem total_folded_sheets (initial_sheets : ℕ) (additional_sheets : ℕ) : 
  initial_sheets = 45 → additional_sheets = 18 → initial_sheets + additional_sheets = 63 := by
sorry

end NUMINAMATH_CALUDE_total_folded_sheets_l552_55211


namespace NUMINAMATH_CALUDE_xiao_ping_weighing_l552_55225

/-- Represents the weights of 8 items -/
structure EightWeights where
  weights : Fin 8 → ℕ
  distinct : ∀ i j, i ≠ j → weights i ≠ weights j
  bounded : ∀ i, 1 ≤ weights i ∧ weights i ≤ 15

/-- The weighing inequalities -/
def weighing_inequalities (w : EightWeights) : Prop :=
  w.weights 0 + w.weights 4 + w.weights 5 + w.weights 6 >
    w.weights 1 + w.weights 2 + w.weights 3 + w.weights 7 ∧
  w.weights 4 + w.weights 5 > w.weights 0 + w.weights 6 ∧
  w.weights 4 > w.weights 5

theorem xiao_ping_weighing (w : EightWeights) :
  weighing_inequalities w →
  (∀ i, i ≠ 4 → w.weights 4 ≤ w.weights i) →
  (∀ i, i ≠ 3 → w.weights i ≤ w.weights 3) →
  w.weights 4 = 11 ∧ w.weights 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ping_weighing_l552_55225


namespace NUMINAMATH_CALUDE_variance_scaling_l552_55218

def variance (data : List ℝ) : ℝ := sorry

theorem variance_scaling (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  let data₁ := [a₁, a₂, a₃, a₄, a₅, a₆]
  let data₂ := [2*a₁, 2*a₂, 2*a₃, 2*a₄, 2*a₅, 2*a₆]
  variance data₁ = 2 → variance data₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_variance_scaling_l552_55218


namespace NUMINAMATH_CALUDE_triangle_inequality_l552_55287

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l552_55287


namespace NUMINAMATH_CALUDE_angle_conversion_negative_1125_conversion_l552_55283

theorem angle_conversion (angle : ℝ) : ∃ (k : ℤ) (α : ℝ), 
  angle = k * 360 + α ∧ 0 ≤ α ∧ α < 360 :=
by
  -- Proof goes here
  sorry

theorem negative_1125_conversion : 
  ∃ (k : ℤ) (α : ℝ), -1125 = k * 360 + α ∧ 0 ≤ α ∧ α < 360 ∧ k = -4 ∧ α = 315 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_angle_conversion_negative_1125_conversion_l552_55283


namespace NUMINAMATH_CALUDE_jelly_mold_radius_l552_55291

theorem jelly_mold_radius :
  let original_radius : ℝ := 1.5
  let num_molds : ℕ := 64
  let hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
  let original_volume := hemisphere_volume original_radius
  let small_mold_radius := (3 : ℝ) / 8
  original_volume = num_molds * hemisphere_volume small_mold_radius :=
by sorry

end NUMINAMATH_CALUDE_jelly_mold_radius_l552_55291


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a13_l552_55248

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a13
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a13_l552_55248


namespace NUMINAMATH_CALUDE_mode_of_data_set_l552_55270

def data_set : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 3 := by sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l552_55270


namespace NUMINAMATH_CALUDE_rose_cost_l552_55221

/-- Proves that the cost of each rose is $5 given the wedding decoration costs --/
theorem rose_cost (num_tables : ℕ) (tablecloth_cost place_setting_cost lily_cost total_cost : ℚ)
  (place_settings_per_table roses_per_table lilies_per_table : ℕ) :
  num_tables = 20 →
  tablecloth_cost = 25 →
  place_setting_cost = 10 →
  place_settings_per_table = 4 →
  roses_per_table = 10 →
  lilies_per_table = 15 →
  lily_cost = 4 →
  total_cost = 3500 →
  (total_cost - 
   (num_tables * tablecloth_cost + 
    num_tables * place_settings_per_table * place_setting_cost + 
    num_tables * lilies_per_table * lily_cost)) / (num_tables * roses_per_table) = 5 := by
  sorry


end NUMINAMATH_CALUDE_rose_cost_l552_55221


namespace NUMINAMATH_CALUDE_base_of_first_term_l552_55212

/-- Given a positive integer h that is divisible by both 225 and 216,
    and can be expressed as h = x^a * 3^b * 5^c where x, a, b, and c are positive integers,
    and the least possible value of a + b + c is 8,
    prove that x must be 2. -/
theorem base_of_first_term (h : ℕ+) (x : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h)
  (h_div_216 : 216 ∣ h)
  (h_eq : h = x^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ))
  (abc_min : a + b + c = 8 ∧ ∀ a' b' c' : ℕ+, a' + b' + c' ≥ 8) : x = 2 :=
sorry

end NUMINAMATH_CALUDE_base_of_first_term_l552_55212


namespace NUMINAMATH_CALUDE_inscribed_square_triangle_l552_55239

/-- 
Given a triangle with base length a and height h, and an inscribed square with side length x 
where one side of the square lies on the base of the triangle, if the area of the square is 1/6 
of the area of the triangle, then x = (3 - √6)/6 * a and h = (5 - 2√6) * a.
-/
theorem inscribed_square_triangle (a h x : ℝ) (ha : a > 0) (hh : h > 0) (hx : x > 0) : 
  x^2 = 1/6 * (1/2 * a * h) → 
  x = (3 - Real.sqrt 6) / 6 * a ∧ h = (5 - 2 * Real.sqrt 6) * a :=
by sorry


end NUMINAMATH_CALUDE_inscribed_square_triangle_l552_55239


namespace NUMINAMATH_CALUDE_cookies_per_bag_l552_55255

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l552_55255


namespace NUMINAMATH_CALUDE_band_members_max_l552_55257

theorem band_members_max (m r x : ℕ) : 
  m < 100 →
  r * x + 3 = m →
  (r - 1) * (x + 2) = m →
  ∀ n : ℕ, n < 100 ∧ (∃ r' x' : ℕ, r' * x' + 3 = n ∧ (r' - 1) * (x' + 2) = n) → n ≤ 91 :=
by sorry

end NUMINAMATH_CALUDE_band_members_max_l552_55257


namespace NUMINAMATH_CALUDE_elenas_garden_tulips_l552_55234

/-- Represents Elena's garden with lilies and tulips. -/
structure Garden where
  lilies : ℕ
  tulips : ℕ
  lily_petals : ℕ
  tulip_petals : ℕ
  total_petals : ℕ

/-- Theorem stating the number of tulips in Elena's garden. -/
theorem elenas_garden_tulips (g : Garden)
  (h1 : g.lilies = 8)
  (h2 : g.lily_petals = 6)
  (h3 : g.tulip_petals = 3)
  (h4 : g.total_petals = 63)
  (h5 : g.total_petals = g.lilies * g.lily_petals + g.tulips * g.tulip_petals) :
  g.tulips = 5 := by
  sorry


end NUMINAMATH_CALUDE_elenas_garden_tulips_l552_55234


namespace NUMINAMATH_CALUDE_other_factor_l552_55222

def n : ℕ := 75

def expression (k : ℕ) : ℕ := k * (2^5) * (6^2) * (7^3)

theorem other_factor : 
  (∃ (m : ℕ), expression n = m * (3^3)) ∧ 
  (∀ (k : ℕ), k < n → ¬∃ (m : ℕ), expression k = m * (3^3)) →
  ∃ (p : ℕ), n = p * 25 ∧ p % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_other_factor_l552_55222


namespace NUMINAMATH_CALUDE_square_completion_and_max_value_l552_55217

theorem square_completion_and_max_value :
  -- Part 1: Completing the square
  ∀ a : ℝ, a^2 + 4*a + 4 = (a + 2)^2 ∧
  -- Part 2: Factorization using completion of squares
  ∀ a : ℝ, a^2 - 24*a + 143 = (a - 11)*(a - 13) ∧
  -- Part 3: Maximum value of quadratic function
  ∀ a : ℝ, -1/4*a^2 + 2*a - 1 ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_square_completion_and_max_value_l552_55217


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_i_plus_two_l552_55284

theorem imaginary_part_of_i_times_i_plus_two (i : ℂ) : 
  Complex.im (i * (i + 2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_i_plus_two_l552_55284


namespace NUMINAMATH_CALUDE_polynomial_equality_l552_55236

theorem polynomial_equality (a b c : ℝ) : 
  (∀ x : ℝ, x * (x + 1) = a + b * x + c * x^2) → a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l552_55236


namespace NUMINAMATH_CALUDE_representation_2015_l552_55231

theorem representation_2015 : ∃ (a b c : ℤ), 
  a + b + c = 2015 ∧ 
  Nat.Prime a.natAbs ∧ 
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m :=
sorry

end NUMINAMATH_CALUDE_representation_2015_l552_55231


namespace NUMINAMATH_CALUDE_strawberries_eaten_l552_55263

theorem strawberries_eaten (initial : Float) (remaining : Nat) (eaten : Nat) : 
  initial = 78.0 → remaining = 36 → eaten = 42 → initial - remaining.toFloat = eaten.toFloat := by
  sorry

end NUMINAMATH_CALUDE_strawberries_eaten_l552_55263


namespace NUMINAMATH_CALUDE_inequality_proof_l552_55208

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤ (1 / 4) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l552_55208


namespace NUMINAMATH_CALUDE_odd_functions_sum_sufficient_not_necessary_l552_55252

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_functions_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsOdd f ∧ IsOdd g → IsOdd (f + g)) ∧
  (∃ f g : ℝ → ℝ, IsOdd (f + g) ∧ ¬(IsOdd f ∧ IsOdd g)) :=
sorry

end NUMINAMATH_CALUDE_odd_functions_sum_sufficient_not_necessary_l552_55252


namespace NUMINAMATH_CALUDE_no_equilateral_grid_triangle_l552_55240

/-- A point with integer coordinates -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three grid points -/
structure GridTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- Check if a triangle is equilateral -/
def isEquilateral (t : GridTriangle) : Prop :=
  let d1 := (t.a.x - t.b.x)^2 + (t.a.y - t.b.y)^2
  let d2 := (t.b.x - t.c.x)^2 + (t.b.y - t.c.y)^2
  let d3 := (t.c.x - t.a.x)^2 + (t.c.y - t.a.y)^2
  d1 = d2 ∧ d2 = d3

/-- The main theorem: no equilateral triangle exists on the integer grid -/
theorem no_equilateral_grid_triangle :
  ¬ ∃ t : GridTriangle, isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_grid_triangle_l552_55240


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_six_l552_55220

theorem sum_of_roots_equals_six : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 9*x) / (x + 3)
  ∃ (a b : ℝ), (f a = 9 ∧ f b = 9 ∧ a ≠ b) ∧ a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_six_l552_55220


namespace NUMINAMATH_CALUDE_smallest_n_complex_equality_l552_55281

theorem smallest_n_complex_equality (a b : ℝ) (c : ℕ+) 
  (h_a : a > 0) (h_b : b > 0) 
  (h_smallest : ∀ k : ℕ+, k < 3 → (c * a + b * Complex.I) ^ k.val ≠ (c * a - b * Complex.I) ^ k.val) 
  (h_equal : (c * a + b * Complex.I) ^ 3 = (c * a - b * Complex.I) ^ 3) :
  b / (c * a) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_complex_equality_l552_55281


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l552_55219

theorem ceiling_floor_sum : ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l552_55219


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l552_55260

/-- Given a line segment with midpoint (3, 0) and one endpoint at (7, -4),
    prove that the other endpoint is at (-1, 4). -/
theorem other_endpoint_of_line_segment
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, 0))
  (h_endpoint1 : endpoint1 = (7, -4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-1, 4) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l552_55260


namespace NUMINAMATH_CALUDE_bread_inventory_l552_55266

/-- The number of loaves of bread sold during the day -/
def loaves_sold : ℕ := 629

/-- The number of loaves of bread delivered in the evening -/
def loaves_delivered : ℕ := 489

/-- The number of loaves of bread at the end of the day -/
def loaves_end : ℕ := 2215

/-- The number of loaves of bread at the start of the day -/
def loaves_start : ℕ := 2355

theorem bread_inventory : loaves_start - loaves_sold + loaves_delivered = loaves_end := by
  sorry

end NUMINAMATH_CALUDE_bread_inventory_l552_55266


namespace NUMINAMATH_CALUDE_children_savings_l552_55241

/-- The total savings of Josiah, Leah, and Megan -/
def total_savings (josiah_daily : ℚ) (josiah_days : ℕ) 
                  (leah_daily : ℚ) (leah_days : ℕ)
                  (megan_daily : ℚ) (megan_days : ℕ) : ℚ :=
  josiah_daily * josiah_days + leah_daily * leah_days + megan_daily * megan_days

/-- Theorem stating that the total savings of the three children is $28 -/
theorem children_savings : 
  total_savings 0.25 24 0.50 20 1.00 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_children_savings_l552_55241


namespace NUMINAMATH_CALUDE_cube_sum_digits_eq_square_self_l552_55290

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The set of solutions to the problem -/
def solutionSet : Set ℕ := {1, 27}

/-- The main theorem -/
theorem cube_sum_digits_eq_square_self :
  ∀ n : ℕ, n < 1000 → (sumOfDigits n)^3 = n^2 ↔ n ∈ solutionSet := by sorry

end NUMINAMATH_CALUDE_cube_sum_digits_eq_square_self_l552_55290


namespace NUMINAMATH_CALUDE_sibling_count_product_l552_55276

/-- Represents a family with a given number of boys and girls -/
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

/-- Represents a sibling in a family -/
structure Sibling :=
  (family : Family)
  (isBoy : Bool)

/-- Counts the number of sisters a sibling has -/
def sisterCount (s : Sibling) : ℕ :=
  s.family.girls - if s.isBoy then 0 else 1

/-- Counts the number of brothers a sibling has -/
def brotherCount (s : Sibling) : ℕ :=
  s.family.boys - if s.isBoy then 1 else 0

theorem sibling_count_product (f : Family) (h : Sibling) (henry : Sibling)
    (henry_sisters : sisterCount henry = 4)
    (henry_brothers : brotherCount henry = 7)
    (h_family : h.family = f)
    (henry_family : henry.family = f)
    (h_girl : h.isBoy = false)
    (henry_boy : henry.isBoy = true) :
    sisterCount h * brotherCount h = 28 := by
  sorry

end NUMINAMATH_CALUDE_sibling_count_product_l552_55276


namespace NUMINAMATH_CALUDE_largest_s_value_l552_55247

theorem largest_s_value (r s : ℕ) : 
  r ≥ s ∧ s ≥ 3 ∧ 
  (r - 2) * s * 5 = (s - 2) * r * 4 →
  s ≤ 130 ∧ ∃ (r' : ℕ), r' ≥ 130 ∧ (r' - 2) * 130 * 5 = (130 - 2) * r' * 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_s_value_l552_55247


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l552_55223

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x > -3, 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l552_55223


namespace NUMINAMATH_CALUDE_max_distance_complex_circle_l552_55274

theorem max_distance_complex_circle (z : ℂ) (z₀ : ℂ) :
  z₀ = 1 - 2*I →
  Complex.abs z = 3 →
  ∃ (max_dist : ℝ), max_dist = 3 + Real.sqrt 5 ∧
    ∀ (w : ℂ), Complex.abs w = 3 → Complex.abs (w - z₀) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_circle_l552_55274


namespace NUMINAMATH_CALUDE_machine_purchase_price_machine_purchase_price_is_14000_l552_55242

/-- Proves that the purchase price of a machine is 14000 given the specified conditions -/
theorem machine_purchase_price : ℝ → Prop :=
  fun purchase_price =>
    let repair_cost : ℝ := 5000
    let transport_cost : ℝ := 1000
    let profit_percentage : ℝ := 50
    let selling_price : ℝ := 30000
    let total_cost : ℝ := purchase_price + repair_cost + transport_cost
    let profit_multiplier : ℝ := (100 + profit_percentage) / 100
    selling_price = profit_multiplier * total_cost →
    purchase_price = 14000

/-- The purchase price of the machine is 14000 -/
theorem machine_purchase_price_is_14000 : machine_purchase_price 14000 := by
  sorry

end NUMINAMATH_CALUDE_machine_purchase_price_machine_purchase_price_is_14000_l552_55242


namespace NUMINAMATH_CALUDE_f_of_3_equals_29_l552_55245

/-- Given f(x) = x^2 + 4x + 8, prove that f(3) = 29 -/
theorem f_of_3_equals_29 :
  let f : ℝ → ℝ := λ x ↦ x^2 + 4*x + 8
  f 3 = 29 := by sorry

end NUMINAMATH_CALUDE_f_of_3_equals_29_l552_55245


namespace NUMINAMATH_CALUDE_max_perimeter_triangle_is_isosceles_l552_55297

/-- Given a fixed base length and a fixed angle at one vertex, 
    the triangle with maximum perimeter is isosceles. -/
theorem max_perimeter_triangle_is_isosceles 
  (b : ℝ) 
  (β : ℝ) 
  (h1 : b > 0) 
  (h2 : 0 < β ∧ β < π) : 
  ∃ (a c : ℝ), 
    a > 0 ∧ c > 0 ∧
    a = c ∧
    ∀ (a' c' : ℝ), 
      a' > 0 → c' > 0 → 
      a' + b + c' ≤ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_triangle_is_isosceles_l552_55297


namespace NUMINAMATH_CALUDE_b_bounds_l552_55216

theorem b_bounds (a : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) :
  let b := a^3 + 1 / (1 + a)
  (b ≥ 1 - a + a^2) ∧ (3/4 < b) ∧ (b ≤ 3/2) := by sorry

end NUMINAMATH_CALUDE_b_bounds_l552_55216


namespace NUMINAMATH_CALUDE_product_evaluation_l552_55277

theorem product_evaluation : (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7 - 1) * 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l552_55277


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l552_55296

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l552_55296


namespace NUMINAMATH_CALUDE_range_of_m_l552_55228

theorem range_of_m (x y m : ℝ) : 
  (2 * x + y = -4 * m + 5) →
  (x + 2 * y = m + 4) →
  (x - y > -6) →
  (x + y < 8) →
  (-5 < m ∧ m < 7/5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l552_55228


namespace NUMINAMATH_CALUDE_sequence_is_quadratic_l552_55295

/-- Checks if a sequence is consistent with a quadratic function --/
def is_quadratic_sequence (seq : List ℕ) : Prop :=
  let first_differences := List.zipWith (·-·) (seq.tail) seq
  let second_differences := List.zipWith (·-·) (first_differences.tail) first_differences
  second_differences.all (· = second_differences.head!)

/-- The given sequence of function values --/
def given_sequence : List ℕ := [1600, 1764, 1936, 2116, 2304, 2500, 2704, 2916]

theorem sequence_is_quadratic :
  is_quadratic_sequence given_sequence :=
sorry

end NUMINAMATH_CALUDE_sequence_is_quadratic_l552_55295


namespace NUMINAMATH_CALUDE_distinct_hexagon_colorings_l552_55275

-- Define the number of disks and colors
def num_disks : ℕ := 6
def num_blue : ℕ := 3
def num_red : ℕ := 2
def num_green : ℕ := 1

-- Define the symmetry group of a hexagon
def hexagon_symmetries : ℕ := 12

-- Define the function to calculate the number of distinct colorings
def distinct_colorings : ℕ :=
  let total_colorings := (num_disks.choose num_blue) * ((num_disks - num_blue).choose num_red)
  let fixed_points_identity := total_colorings
  let fixed_points_reflection := 3 * (3 * 2 * 1)  -- 3 reflections, each with 6 fixed points
  let fixed_points_rotation := 0  -- 120° and 240° rotations have no fixed points
  (fixed_points_identity + fixed_points_reflection + fixed_points_rotation) / hexagon_symmetries

-- Theorem statement
theorem distinct_hexagon_colorings :
  distinct_colorings = 13 :=
sorry

end NUMINAMATH_CALUDE_distinct_hexagon_colorings_l552_55275


namespace NUMINAMATH_CALUDE_minimize_expression_l552_55292

theorem minimize_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 50 / n ≥ 8.1667 ∧
  ((n : ℝ) / 3 + 50 / n = 8.1667 ↔ n = 12) :=
sorry

end NUMINAMATH_CALUDE_minimize_expression_l552_55292


namespace NUMINAMATH_CALUDE_pizza_order_l552_55232

theorem pizza_order (num_people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 10)
  (h2 : slices_per_person = 2)
  (h3 : slices_per_pizza = 4) :
  (num_people * slices_per_person + slices_per_pizza - 1) / slices_per_pizza = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l552_55232


namespace NUMINAMATH_CALUDE_possible_values_of_a_l552_55259

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l552_55259


namespace NUMINAMATH_CALUDE_abc_relationship_l552_55244

-- Define the constants a, b, and c
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

-- State the theorem
theorem abc_relationship : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_abc_relationship_l552_55244


namespace NUMINAMATH_CALUDE_expression_simplification_l552_55273

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^4 / ((a - b) * (a - c)) + (x + b)^4 / ((b - a) * (b - c)) + (x + c)^4 / ((c - a) * (c - b)) =
  a + b + c + 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l552_55273


namespace NUMINAMATH_CALUDE_root_sum_product_theorem_l552_55249

theorem root_sum_product_theorem (m : ℚ) :
  (∃ x y : ℚ, 
    (2*(x-1)*(x-3*m) = x*(m-4)) ∧ 
    (x + y = x * y) ∧
    (∀ z : ℚ, 2*z^2 + (5*m + 6)*z + 6*m = 0 ↔ (z = x ∨ z = y))) →
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_theorem_l552_55249


namespace NUMINAMATH_CALUDE_purchasing_power_decrease_l552_55204

def initial_amount : ℝ := 100
def monthly_price_index_increase : ℝ := 0.00465
def months : ℕ := 12

theorem purchasing_power_decrease :
  let final_value := initial_amount * (1 - monthly_price_index_increase) ^ months
  initial_amount - final_value = 4.55 := by
  sorry

end NUMINAMATH_CALUDE_purchasing_power_decrease_l552_55204


namespace NUMINAMATH_CALUDE_bart_earnings_l552_55256

/-- The amount Bart earns per question in dollars -/
def earnings_per_question : ℚ := 0.2

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday -/
def monday_surveys : ℕ := 3

/-- The number of surveys Bart completed on Tuesday -/
def tuesday_surveys : ℕ := 4

/-- Theorem stating Bart's total earnings over two days -/
theorem bart_earnings : 
  (earnings_per_question * questions_per_survey * (monday_surveys + tuesday_surveys) : ℚ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l552_55256


namespace NUMINAMATH_CALUDE_dvd_price_ratio_l552_55215

theorem dvd_price_ratio (mike_price steve_total_price : ℝ) 
  (h1 : mike_price = 5)
  (h2 : steve_total_price = 18)
  (h3 : ∃ (steve_online_price : ℝ), 
    steve_total_price = steve_online_price + 0.8 * steve_online_price) :
  ∃ (steve_online_price : ℝ), 
    steve_online_price / mike_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_dvd_price_ratio_l552_55215


namespace NUMINAMATH_CALUDE_hypotenuse_to_brush_ratio_l552_55250

/-- A right triangle with hypotenuse 2a and a brush of width w painting one-third of its area -/
structure PaintedTriangle (a : ℝ) where
  w : ℝ
  area_painted : (a ^ 2) / 3 = a * w

/-- The ratio of the hypotenuse to the brush width is 6 -/
theorem hypotenuse_to_brush_ratio (a : ℝ) (t : PaintedTriangle a) :
  (2 * a) / t.w = 6 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_to_brush_ratio_l552_55250


namespace NUMINAMATH_CALUDE_vector_colinearity_l552_55233

def vector_a : Fin 2 → ℝ := ![(-3 : ℝ), 4]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![m, 2]

def colinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = λ i => k * w i

theorem vector_colinearity (m : ℝ) :
  colinear (λ i => 2 * vector_a i - 3 * vector_b m i) (vector_b m) →
  m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_colinearity_l552_55233


namespace NUMINAMATH_CALUDE_election_votes_proof_l552_55294

theorem election_votes_proof (total_votes : ℕ) 
  (h1 : (total_votes : ℝ) * 0.8 * 0.55 + 2520 = (total_votes : ℝ) * 0.8) : 
  total_votes = 7000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_proof_l552_55294


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l552_55280

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, -a^2 + 9*a - 20 ≥ 0 → a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l552_55280


namespace NUMINAMATH_CALUDE_dessert_distribution_l552_55293

theorem dessert_distribution (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) :
  mini_cupcakes = 14 →
  donut_holes = 12 →
  students = 13 →
  (mini_cupcakes + donut_holes) % students = 0 →
  (mini_cupcakes + donut_holes) / students = 2 :=
by sorry

end NUMINAMATH_CALUDE_dessert_distribution_l552_55293


namespace NUMINAMATH_CALUDE_rope_length_satisfies_conditions_l552_55229

/-- The length of the rope in feet -/
def rope_length : ℝ := 10

/-- The length of the rope hanging down from the top of the pillar to the ground in feet -/
def hanging_length : ℝ := 4

/-- The distance from the base of the pillar to where the rope reaches the ground when pulled taut in feet -/
def ground_distance : ℝ := 8

/-- Theorem stating that the rope length satisfies the given conditions -/
theorem rope_length_satisfies_conditions :
  rope_length ^ 2 = (rope_length - hanging_length) ^ 2 + ground_distance ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_rope_length_satisfies_conditions_l552_55229


namespace NUMINAMATH_CALUDE_equation_solutions_count_l552_55238

theorem equation_solutions_count :
  let f : Real → Real := fun θ => (Real.sin θ ^ 2 - 1) * (2 * Real.sin θ ^ 2 - 1)
  ∃! (s : Finset Real), s.card = 6 ∧ 
    (∀ θ ∈ s, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0) ∧
    (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0 → θ ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l552_55238


namespace NUMINAMATH_CALUDE_range_of_product_l552_55230

theorem range_of_product (a b : ℝ) (ha : |a| ≤ 1) (hab : |a + b| ≤ 1) :
  -2 ≤ (a + 1) * (b + 1) ∧ (a + 1) * (b + 1) ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_product_l552_55230


namespace NUMINAMATH_CALUDE_work_completion_theorem_l552_55253

theorem work_completion_theorem (days_group1 days_group2 : ℕ) 
  (men_group2 : ℕ) (total_work : ℕ) :
  days_group1 = 18 →
  days_group2 = 8 →
  men_group2 = 81 →
  total_work = men_group2 * days_group2 →
  ∃ men_group1 : ℕ, men_group1 * days_group1 = total_work ∧ men_group1 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l552_55253


namespace NUMINAMATH_CALUDE_store_profit_maximization_l552_55210

/-- Represents the store's profit function --/
def profit_function (x : ℕ) : ℚ := -10 * x^2 + 800 * x + 20000

/-- Represents the constraint on the price increase --/
def valid_price_increase (x : ℕ) : Prop := x ≤ 100

theorem store_profit_maximization :
  ∃ (x : ℕ), valid_price_increase x ∧
    (∀ (y : ℕ), valid_price_increase y → profit_function y ≤ profit_function x) ∧
    x = 40 ∧ profit_function x = 36000 := by
  sorry

#check store_profit_maximization

end NUMINAMATH_CALUDE_store_profit_maximization_l552_55210


namespace NUMINAMATH_CALUDE_pattern_steps_l552_55224

/-- The number of sticks used in the kth step of the pattern -/
def sticks_in_step (k : ℕ) : ℕ := 2 * k + 1

/-- The total number of sticks used in a pattern with n steps -/
def total_sticks (n : ℕ) : ℕ := n^2

/-- Theorem: If a stair-like pattern is constructed where the kth step uses 2k + 1 sticks,
    and the total number of sticks used is 169, then the number of steps in the pattern is 13 -/
theorem pattern_steps :
  ∀ n : ℕ, (∀ k : ℕ, k ≤ n → sticks_in_step k = 2 * k + 1) →
  total_sticks n = 169 → n = 13 := by sorry

end NUMINAMATH_CALUDE_pattern_steps_l552_55224


namespace NUMINAMATH_CALUDE_medium_kite_area_l552_55261

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a kite on a 2D grid -/
structure Kite where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint
  v4 : GridPoint

/-- Calculates the area of a kite given its vertices on a grid with 2-inch spacing -/
def kiteArea (k : Kite) : Real :=
  sorry

/-- Theorem: The area of the specified kite is 288 square inches -/
theorem medium_kite_area : 
  let k : Kite := {
    v1 := { x := 0, y := 4 },
    v2 := { x := 4, y := 10 },
    v3 := { x := 12, y := 4 },
    v4 := { x := 4, y := 0 }
  }
  kiteArea k = 288 := by
  sorry

end NUMINAMATH_CALUDE_medium_kite_area_l552_55261


namespace NUMINAMATH_CALUDE_sum_of_squares_l552_55251

theorem sum_of_squares : (10 + 3)^2 + (7 - 5)^2 = 173 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l552_55251


namespace NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l552_55206

theorem ab_plus_cd_eq_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l552_55206


namespace NUMINAMATH_CALUDE_max_trig_product_bound_max_trig_product_achievable_l552_55298

theorem max_trig_product_bound (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 :=
sorry

theorem max_trig_product_achievable :
  ∃ x y z : ℝ,
    (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
    (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_trig_product_bound_max_trig_product_achievable_l552_55298


namespace NUMINAMATH_CALUDE_base3_sum_equality_l552_55226

/-- Converts a list of digits in base 3 to its decimal representation -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- Converts a decimal number to its base 3 representation -/
def toBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 3) ((m % 3) :: acc)
    go n []

/-- The sum of the given numbers in base 3 is equal to 112010 in base 3 -/
theorem base3_sum_equality : 
  let a := [2]
  let b := [1, 1]
  let c := [2, 0, 2]
  let d := [1, 0, 0, 2]
  let e := [2, 2, 1, 1, 1]
  let sum := [0, 1, 0, 2, 1, 1]
  toBase3 (toDecimal a + toDecimal b + toDecimal c + toDecimal d + toDecimal e) = sum := by
  sorry

end NUMINAMATH_CALUDE_base3_sum_equality_l552_55226


namespace NUMINAMATH_CALUDE_exists_non_polynomial_satisfying_inequality_l552_55243

-- Define a periodic function with period 2
def periodic_function (k : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, k (x + 2) = k x

-- Define a bounded function
def bounded_function (k : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ x : ℝ, |k x| ≤ M

-- Define a non-constant function
def non_constant_function (k : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, k x ≠ k y

-- Main theorem
theorem exists_non_polynomial_satisfying_inequality :
  ∃ f : ℝ → ℝ, 
    (∀ x : ℝ, (x - 1) * f (x + 1) - (x + 1) * f (x - 1) ≥ 4 * x * (x^2 - 1)) ∧
    (∃ k : ℝ → ℝ, 
      (∀ x : ℝ, f x = x^3 + x * k x) ∧
      periodic_function k ∧
      bounded_function k ∧
      non_constant_function k) :=
sorry

end NUMINAMATH_CALUDE_exists_non_polynomial_satisfying_inequality_l552_55243


namespace NUMINAMATH_CALUDE_stone_collision_distance_l552_55213

/-- The initial distance between two colliding stones -/
theorem stone_collision_distance (v₀ H g : ℝ) (h_v₀_pos : 0 < v₀) (h_H_pos : 0 < H) (h_g_pos : 0 < g) :
  let t := H / v₀
  let y₁ := H - (1/2) * g * t^2
  let y₂ := v₀ * t - (1/2) * g * t^2
  let x₁ := v₀ * t
  y₁ = y₂ →
  Real.sqrt (H^2 + x₁^2) = H * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_stone_collision_distance_l552_55213


namespace NUMINAMATH_CALUDE_difference_of_ones_and_zeros_l552_55202

-- Define the decimal number
def decimal_number : ℕ := 173

-- Define the binary representation as a list of bits
def binary_representation : List Bool := [true, false, true, false, true, true, false, true]

-- Define x as the number of zeros
def x : ℕ := binary_representation.filter (· = false) |>.length

-- Define y as the number of ones
def y : ℕ := binary_representation.filter (· = true) |>.length

-- Theorem to prove
theorem difference_of_ones_and_zeros : y - x = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_ones_and_zeros_l552_55202


namespace NUMINAMATH_CALUDE_r_equals_four_l552_55254

theorem r_equals_four (p c : ℚ) : 
  p * 4 = 360 → 6 * c * 4 = 15 → 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_r_equals_four_l552_55254


namespace NUMINAMATH_CALUDE_min_height_is_eleven_l552_55278

/-- Represents the dimensions of a rectangular box with square bases -/
structure BoxDimensions where
  base : ℝ
  height : ℝ

/-- Calculates the surface area of the box -/
def surfaceArea (d : BoxDimensions) : ℝ :=
  2 * d.base^2 + 4 * d.base * d.height

/-- Checks if the box dimensions satisfy the given conditions -/
def isValidBox (d : BoxDimensions) : Prop :=
  d.height = d.base + 6 ∧ surfaceArea d ≥ 150 ∧ d.base > 0

theorem min_height_is_eleven :
  ∀ d : BoxDimensions, isValidBox d → d.height ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_height_is_eleven_l552_55278


namespace NUMINAMATH_CALUDE_EDTA_Ca_complex_weight_l552_55286

-- Define the molecular weight of EDTA
def EDTA_weight : ℝ := 292.248

-- Define the atomic weight of calcium
def Ca_weight : ℝ := 40.08

-- Define the complex ratio
def complex_ratio : ℕ := 1

-- Theorem statement
theorem EDTA_Ca_complex_weight :
  complex_ratio * (EDTA_weight + Ca_weight) = 332.328 := by sorry

end NUMINAMATH_CALUDE_EDTA_Ca_complex_weight_l552_55286


namespace NUMINAMATH_CALUDE_calories_in_pound_of_fat_l552_55268

/-- Represents the number of calories in a pound of body fat -/
def calories_per_pound : ℝ := 3500

/-- Represents the number of calories burned per day through light exercise -/
def calories_burned_per_day : ℝ := 2500

/-- Represents the number of calories consumed per day -/
def calories_consumed_per_day : ℝ := 2000

/-- Represents the number of days it takes to lose the weight -/
def days_to_lose_weight : ℝ := 35

/-- Represents the number of pounds lost -/
def pounds_lost : ℝ := 5

theorem calories_in_pound_of_fat :
  calories_per_pound = 
    ((calories_burned_per_day - calories_consumed_per_day) * days_to_lose_weight) / pounds_lost :=
by sorry

end NUMINAMATH_CALUDE_calories_in_pound_of_fat_l552_55268


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l552_55200

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 0 → a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l552_55200


namespace NUMINAMATH_CALUDE_longest_tape_measure_l552_55265

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 315) 
  (hb : b = 458) 
  (hc : c = 1112) : 
  Nat.gcd a (Nat.gcd b c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l552_55265


namespace NUMINAMATH_CALUDE_winning_configurations_l552_55289

/-- Calculates the nim-value of a wall with given number of bricks -/
def nimValue (bricks : Nat) : Nat :=
  match bricks with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => 2
  | 6 => 1
  | 7 => 3
  | 8 => 1
  | 9 => 2
  | _ => sorry  -- For simplicity, we don't define beyond 9

/-- Calculates the nim-sum (XOR) of a list of natural numbers -/
def nimSum (list : List Nat) : Nat :=
  list.foldl Nat.xor 0

/-- Represents a configuration of walls -/
structure WallConfiguration where
  walls : List Nat

/-- Checks if a configuration is a winning position for the second player -/
def isWinningForSecondPlayer (config : WallConfiguration) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- Theorem: The given configurations are winning positions for the second player -/
theorem winning_configurations :
  (isWinningForSecondPlayer ⟨[8, 2, 3]⟩) ∧
  (isWinningForSecondPlayer ⟨[9, 3, 3]⟩) ∧
  (isWinningForSecondPlayer ⟨[9, 5, 2]⟩) :=
by sorry

end NUMINAMATH_CALUDE_winning_configurations_l552_55289


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_l552_55264

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_C_in_triangle (t : Triangle) :
  t.a = Real.sqrt 2 ∧ 
  t.b = Real.sqrt 3 ∧ 
  t.A = 45 * (π / 180) →
  t.C = 75 * (π / 180) ∨ t.C = 15 * (π / 180) := by
  sorry


end NUMINAMATH_CALUDE_angle_C_in_triangle_l552_55264


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l552_55214

-- Part 1
theorem part_one : 12 - (-11) - 1 = 22 := by sorry

-- Part 2
theorem part_two : -1^4 / (-3)^2 / (9/5) = -5/81 := by sorry

-- Part 3
theorem part_three : -8 * (1/2 - 3/4 + 5/8) = -3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l552_55214


namespace NUMINAMATH_CALUDE_current_speed_current_speed_calculation_l552_55205

/-- The speed of the current in a river given the man's rowing speed in still water,
    the time taken to cover a distance downstream, and the distance covered. -/
theorem current_speed (rowing_speed : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let downstream_speed := distance / (time / 3600)
  downstream_speed - rowing_speed

/-- Proof that the speed of the current is approximately 3.00048 kmph given the specified conditions. -/
theorem current_speed_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |current_speed 9 17.998560115190788 0.06 - 3.00048| < ε :=
sorry

end NUMINAMATH_CALUDE_current_speed_current_speed_calculation_l552_55205
