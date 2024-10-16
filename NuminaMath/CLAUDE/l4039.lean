import Mathlib

namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l4039_403926

def a : ℝ × ℝ × ℝ := (3, -2, 4)
def b : ℝ → ℝ → ℝ × ℝ × ℝ := λ x y ↦ (1, x, y)

theorem parallel_vectors_sum (x y : ℝ) :
  (∃ (k : ℝ), b x y = k • a) → x + y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l4039_403926


namespace NUMINAMATH_CALUDE_cos_2x_values_l4039_403902

theorem cos_2x_values (x : ℝ) (h : Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - 3 * Real.cos x ^ 2 = 0) :
  Real.cos (2 * x) = -4/5 ∨ Real.cos (2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_values_l4039_403902


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l4039_403943

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 36) 
  (h2 : a * c = 54) 
  (h3 : b * c = 72) : 
  a * b * c = 648 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l4039_403943


namespace NUMINAMATH_CALUDE_prime_factors_count_l4039_403998

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The main expression in the problem -/
def f (n : ℕ) : ℕ := (n^(2*n) + n^n + n + 1)^(2*n) + (n^(2*n) + n^n + n + 1)^n + 1

/-- The theorem statement -/
theorem prime_factors_count (n : ℕ) (h : ¬3 ∣ n) : 
  2 * d n ≤ (Nat.factors (f n)).card := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_l4039_403998


namespace NUMINAMATH_CALUDE_six_by_six_untileable_large_rectangle_tileable_six_by_eight_tileable_l4039_403984

/-- A domino is a 1x2 tile -/
structure Domino :=
  (length : Nat := 2)
  (width : Nat := 1)

/-- A rectangle with dimensions m and n -/
structure Rectangle (m n : Nat) where
  mk ::

/-- A tiling of a rectangle with dominoes -/
def Tiling (m n : Nat) := List Domino

/-- A seam is a straight line not cutting through any dominoes -/
def HasSeam (t : Tiling m n) : Prop := sorry

/-- Theorem: A 6x6 square cannot be tiled with dominoes without a seam -/
theorem six_by_six_untileable : 
  ∀ (t : Tiling 6 6), HasSeam t := sorry

/-- Theorem: Any m×n rectangle where m, n > 6 and mn is even can be tiled without a seam -/
theorem large_rectangle_tileable (m n : Nat) 
  (hm : m > 6) (hn : n > 6) (h_even : Even (m * n)) : 
  ∃ (t : Tiling m n), ¬HasSeam t := sorry

/-- Theorem: A 6x8 rectangle can be tiled without a seam -/
theorem six_by_eight_tileable : 
  ∃ (t : Tiling 6 8), ¬HasSeam t := sorry

end NUMINAMATH_CALUDE_six_by_six_untileable_large_rectangle_tileable_six_by_eight_tileable_l4039_403984


namespace NUMINAMATH_CALUDE_garden_area_l4039_403918

theorem garden_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l4039_403918


namespace NUMINAMATH_CALUDE_nth_root_approximation_l4039_403985

/-- Approximation of nth root of x₀ⁿ + Δx --/
theorem nth_root_approximation
  (n : ℕ) (x₀ Δx ε : ℝ) (h_x₀_pos : x₀ > 0) (h_Δx_small : |Δx| < x₀^n) :
  ∃ (ε : ℝ), ε > 0 ∧ 
  |((x₀^n + Δx)^(1/n : ℝ) : ℝ) - (x₀ + Δx / (n * x₀^(n-1)))| < ε :=
by sorry

end NUMINAMATH_CALUDE_nth_root_approximation_l4039_403985


namespace NUMINAMATH_CALUDE_classroom_ratio_l4039_403972

theorem classroom_ratio (num_boys num_girls : ℕ) 
  (h_positive : num_boys > 0 ∧ num_girls > 0) :
  let total := num_boys + num_girls
  let prob_boy := num_boys / total
  let prob_girl := num_girls / total
  prob_boy = (3/4 : ℚ) * prob_girl →
  (num_boys : ℚ) / total = 3/7 := by
sorry

end NUMINAMATH_CALUDE_classroom_ratio_l4039_403972


namespace NUMINAMATH_CALUDE_largest_number_with_digits_3_2_sum_11_l4039_403934

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digits_3_2_sum_11 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 11 → n ≤ 32222 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digits_3_2_sum_11_l4039_403934


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l4039_403979

theorem recreation_spending_comparison (W : ℝ) : 
  let last_week_recreation := 0.15 * W
  let this_week_wages := 0.8 * W
  let this_week_recreation := 0.5 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 267 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l4039_403979


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l4039_403928

theorem x_squared_minus_y_squared_equals_five 
  (x y : ℝ) 
  (h1 : 23 * x + 977 * y = 2023) 
  (h2 : 977 * x + 23 * y = 2977) : 
  x^2 - y^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l4039_403928


namespace NUMINAMATH_CALUDE_abc_inequality_l4039_403914

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a / (1 + b) + b / (1 + c) + c / (1 + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l4039_403914


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l4039_403977

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (-2 + a * Complex.I) / (1 + Complex.I)
  (z.re = 0) ↔ (a = 2) := by
sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l4039_403977


namespace NUMINAMATH_CALUDE_calculate_expression_l4039_403941

theorem calculate_expression : 
  50000 - ((37500 / 62.35)^2 + Real.sqrt 324) = -311752.222 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4039_403941


namespace NUMINAMATH_CALUDE_max_clock_digit_sum_l4039_403920

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10 + digit_sum (n / 10))

def clock_digit_sum (h m : ℕ) : ℕ := digit_sum h + digit_sum m

theorem max_clock_digit_sum : 
  ∀ h m, is_valid_hour h → is_valid_minute m → 
  clock_digit_sum h m ≤ 28 ∧ 
  ∃ h' m', is_valid_hour h' ∧ is_valid_minute m' ∧ clock_digit_sum h' m' = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_clock_digit_sum_l4039_403920


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l4039_403970

/-- Represents the ages of three brothers -/
structure BrotherAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrotherAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to be proved -/
theorem david_scott_age_difference (ages : BrotherAges) :
  problem_conditions ages → ages.david - ages.scott = 8 := by
  sorry


end NUMINAMATH_CALUDE_david_scott_age_difference_l4039_403970


namespace NUMINAMATH_CALUDE_triangle_perimeter_ratio_l4039_403953

theorem triangle_perimeter_ratio (X Y Z D J : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : 
  let XZ := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let YZ := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let XY := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  -- XYZ is a right triangle with hypotenuse XY
  (X.1 - Z.1) * (Y.1 - Z.1) + (X.2 - Z.2) * (Y.2 - Z.2) = 0 →
  -- XZ = 8, YZ = 15
  XZ = 8 →
  YZ = 15 →
  -- ZD is the altitude to XY
  (X.1 - Y.1) * (D.1 - Z.1) + (X.2 - Y.2) * (D.2 - Z.2) = 0 →
  -- ω is the circle with ZD as diameter
  ω = {P : ℝ × ℝ | (P.1 - Z.1)^2 + (P.2 - Z.2)^2 = (D.1 - Z.1)^2 + (D.2 - Z.2)^2} →
  -- J is outside XYZ
  (J.1 - X.1) * (Y.2 - X.2) - (J.2 - X.2) * (Y.1 - X.1) ≠ 0 →
  -- XJ and YJ are tangent to ω
  ∃ P ∈ ω, (J.1 - X.1) * (P.1 - X.1) + (J.2 - X.2) * (P.2 - X.2) = 0 →
  ∃ Q ∈ ω, (J.1 - Y.1) * (Q.1 - Y.1) + (J.2 - Y.2) * (Q.2 - Y.2) = 0 →
  -- The ratio of the perimeter of XYJ to XY is 30/17
  (Real.sqrt ((X.1 - J.1)^2 + (X.2 - J.2)^2) + 
   Real.sqrt ((Y.1 - J.1)^2 + (Y.2 - J.2)^2) + XY) / XY = 30/17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_ratio_l4039_403953


namespace NUMINAMATH_CALUDE_sally_grew_six_carrots_l4039_403925

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := total_carrots - fred_carrots

theorem sally_grew_six_carrots : sally_carrots = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_six_carrots_l4039_403925


namespace NUMINAMATH_CALUDE_cos_315_degrees_l4039_403910

theorem cos_315_degrees :
  Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l4039_403910


namespace NUMINAMATH_CALUDE_license_plate_increase_l4039_403903

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 6760 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l4039_403903


namespace NUMINAMATH_CALUDE_ellipse_properties_l4039_403921

/-- Given an ellipse with equation x²/m + y²/(m/(m+3)) = 1 where m > 0,
    and eccentricity e = √3/2, prove the following properties. -/
theorem ellipse_properties (m : ℝ) (h_m : m > 0) :
  let e := Real.sqrt 3 / 2
  let a := Real.sqrt m
  let b := Real.sqrt (m / (m + 3))
  let c := Real.sqrt ((m * (m + 2)) / (m + 3))
  (e = c / a) →
  (m = 1 ∧
   2 * a = 2 ∧ 2 * b = 1 ∧
   c = Real.sqrt 3 / 2 ∧
   a = 1 ∧ b = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4039_403921


namespace NUMINAMATH_CALUDE_green_paint_amount_l4039_403989

/-- Paint mixture ratios -/
structure PaintMixture where
  blue : ℚ
  green : ℚ
  white : ℚ
  red : ℚ

/-- Theorem: Given a paint mixture with ratio 5:3:4:2 for blue:green:white:red,
    if 10 quarts of blue paint are used, then 6 quarts of green paint should be used. -/
theorem green_paint_amount (mix : PaintMixture) 
  (ratio : mix.blue = 5 ∧ mix.green = 3 ∧ mix.white = 4 ∧ mix.red = 2) 
  (blue_amount : ℚ) (h : blue_amount = 10) : 
  (blue_amount * mix.green / mix.blue) = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_paint_amount_l4039_403989


namespace NUMINAMATH_CALUDE_xy_value_l4039_403965

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 + y^2 = 2) (h2 : x^4 + y^4 = 14/8) : x * y = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l4039_403965


namespace NUMINAMATH_CALUDE_intersection_of_sets_l4039_403924

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | |x| ≤ 4} → 
  B = {x : ℝ | 4 ≤ x ∧ x < 5} → 
  A ∩ B = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l4039_403924


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocals_l4039_403939

theorem roots_sum_of_squares_reciprocals (α : ℝ) :
  let f (x : ℝ) := x^2 + x * Real.sin α + 1
  let g (x : ℝ) := x^2 + x * Real.cos α - 1
  ∀ a b c d : ℝ,
    f a = 0 → f b = 0 → g c = 0 → g d = 0 →
    1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocals_l4039_403939


namespace NUMINAMATH_CALUDE_xiaohuas_stamp_buying_ways_l4039_403986

/-- Represents the number of ways to buy stamps given the total money and stamp prices -/
def waysToByStamps (totalMoney : ℕ) (stamp1Price : ℕ) (stamp2Price : ℕ) : ℕ := 
  let maxStamp1 := totalMoney / stamp1Price
  let maxStamp2 := totalMoney / stamp2Price
  (maxStamp1 + 1) * (maxStamp2 + 1) - 1

/-- The problem statement -/
theorem xiaohuas_stamp_buying_ways :
  waysToByStamps 7 2 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_xiaohuas_stamp_buying_ways_l4039_403986


namespace NUMINAMATH_CALUDE_range_of_a_l4039_403978

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_ineq : f (a + 1) ≤ f 4) :
  -5 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4039_403978


namespace NUMINAMATH_CALUDE_monomial_difference_l4039_403908

theorem monomial_difference (m n : ℤ) : 
  (∃ (a : ℝ) (p q : ℤ), ∀ (x y : ℝ), 9 * x^(m-2) * y^2 - (-3 * x^3 * y^(n+1)) = a * x^p * y^q) → 
  n - m = -4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_difference_l4039_403908


namespace NUMINAMATH_CALUDE_distinct_values_of_combination_sum_l4039_403958

theorem distinct_values_of_combination_sum :
  ∃ (S : Finset ℕ), 
    (∀ r : ℕ, r + 1 ≤ 10 ∧ 17 - r ≤ 10 → 
      (Nat.choose 10 (r + 1) + Nat.choose 10 (17 - r)) ∈ S) ∧
    Finset.card S = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_of_combination_sum_l4039_403958


namespace NUMINAMATH_CALUDE_rational_term_count_is_seventeen_l4039_403936

/-- The number of terms with rational coefficients in the expansion of (√3x + ∛2)^100 -/
def rationalTermCount : ℕ := 17

/-- The exponent in the binomial expansion -/
def exponent : ℕ := 100

/-- Predicate to check if a number is a multiple of 2 -/
def isMultipleOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Predicate to check if a number is a multiple of 3 -/
def isMultipleOfThree (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

/-- Theorem stating that the number of terms with rational coefficients in the expansion of (√3x + ∛2)^100 is 17 -/
theorem rational_term_count_is_seventeen :
  (∀ r : ℕ, r ≤ exponent →
    (isMultipleOfTwo (exponent - r) ∧ isMultipleOfThree r) ↔
    (∃ n : ℕ, r = 6 * n ∧ n ≤ 16)) ∧
  rationalTermCount = 17 := by sorry

end NUMINAMATH_CALUDE_rational_term_count_is_seventeen_l4039_403936


namespace NUMINAMATH_CALUDE_incircle_tangents_concurrent_l4039_403945

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

/-- Checks if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Returns the tangent line to a circle at a given point -/
def tangent_line (c : Circle) (p : Point) : Line := sorry

/-- Returns the line passing through two points -/
def line_through_points (p1 p2 : Point) : Line := sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Checks if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem incircle_tangents_concurrent 
  (t : Triangle) 
  (incircle : Circle) 
  (m n k l : Point) :
  point_on_circle m incircle →
  point_on_circle n incircle →
  point_on_circle k incircle →
  point_on_circle l incircle →
  point_on_line m (line_through_points t.a t.b) →
  point_on_line n (line_through_points t.b t.c) →
  point_on_line k (line_through_points t.c t.a) →
  point_on_line l (line_through_points t.a t.c) →
  are_concurrent 
    (line_through_points m n)
    (line_through_points k l)
    (tangent_line incircle t.a) :=
by
  sorry

end NUMINAMATH_CALUDE_incircle_tangents_concurrent_l4039_403945


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l4039_403990

/-- Given a cylinder whose front view is a rectangle with area 6,
    prove that its lateral area is 6π. -/
theorem cylinder_lateral_area (h : ℝ) (h_pos : h > 0) : 
  let diameter := 6 / h
  let lateral_area := π * diameter * h
  lateral_area = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l4039_403990


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_81_factorial_l4039_403955

/-- The largest power of 3 that divides n! -/
def largest_power_of_three_dividing_factorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27) + (n / 81)

/-- The ones digit of 3^n -/
def ones_digit_of_power_of_three (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem ones_digit_of_largest_power_of_three_dividing_81_factorial :
  ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 81) = 1 := by
  sorry

#eval ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 81)

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_81_factorial_l4039_403955


namespace NUMINAMATH_CALUDE_tan_double_angle_special_point_l4039_403971

/-- Given a point P(1, -2) in the plane, and an angle α whose terminal side passes through P,
    prove that tan(2α) = 4/3 -/
theorem tan_double_angle_special_point (α : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = -2 ∧ Real.tan α = P.2 / P.1) →
  Real.tan (2 * α) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_point_l4039_403971


namespace NUMINAMATH_CALUDE_childrens_book_weight_l4039_403931

-- Define the weight of a comic book
def comic_book_weight : ℝ := 0.8

-- Define the total weight of all books
def total_weight : ℝ := 10.98

-- Define the number of comic books
def num_comic_books : ℕ := 9

-- Define the number of children's books
def num_children_books : ℕ := 7

-- Theorem to prove
theorem childrens_book_weight :
  (total_weight - (num_comic_books : ℝ) * comic_book_weight) / num_children_books = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_childrens_book_weight_l4039_403931


namespace NUMINAMATH_CALUDE_max_temp_range_l4039_403988

/-- Given 5 temperatures with an average of 40 and a minimum of 30,
    the maximum possible range is 50. -/
theorem max_temp_range (temps : Fin 5 → ℝ) 
    (avg : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 40)
    (min : ∀ i, temps i ≥ 30) 
    (exists_min : ∃ i, temps i = 30) : 
    (∀ i j, temps i - temps j ≤ 50) ∧ 
    (∃ i j, temps i - temps j = 50) := by
  sorry

end NUMINAMATH_CALUDE_max_temp_range_l4039_403988


namespace NUMINAMATH_CALUDE_time_difference_per_question_l4039_403973

/-- Prove that the difference in time per question between the Math and English exams is 4 minutes -/
theorem time_difference_per_question (english_questions math_questions : ℕ) 
  (english_duration math_duration : ℚ) : 
  english_questions = 30 →
  math_questions = 15 →
  english_duration = 1 →
  math_duration = 3/2 →
  (math_duration * 60 / math_questions) - (english_duration * 60 / english_questions) = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_question_l4039_403973


namespace NUMINAMATH_CALUDE_largest_n_value_largest_n_achievable_l4039_403930

/-- Represents a digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents a digit in base 9 -/
def Base9Digit := Fin 9

/-- Converts a number from base 5 to base 10 -/
def fromBase5 (a b c : Base5Digit) : ℕ :=
  25 * a.val + 5 * b.val + c.val

/-- Converts a number from base 9 to base 10 -/
def fromBase9 (c b a : Base9Digit) : ℕ :=
  81 * c.val + 9 * b.val + a.val

theorem largest_n_value (n : ℕ) 
  (h1 : ∃ (a b c : Base5Digit), n = fromBase5 a b c)
  (h2 : ∃ (a b c : Base9Digit), n = fromBase9 c b a) :
  n ≤ 111 := by
  sorry

theorem largest_n_achievable : 
  ∃ (n : ℕ) (a b c : Base5Digit) (x y z : Base9Digit),
    n = fromBase5 a b c ∧ 
    n = fromBase9 z y x ∧ 
    n = 111 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_value_largest_n_achievable_l4039_403930


namespace NUMINAMATH_CALUDE_valley_of_five_lakes_streams_l4039_403991

structure Lake :=
  (name : String)

structure Valley :=
  (lakes : Finset Lake)
  (streams : Finset (Lake × Lake))
  (start : Lake)

def Valley.valid (v : Valley) : Prop :=
  v.lakes.card = 5 ∧
  ∃ S B : Lake,
    S ∈ v.lakes ∧
    B ∈ v.lakes ∧
    S ≠ B ∧
    (∀ fish : ℕ → Lake,
      fish 0 = v.start →
      (∀ i < 4, (fish i, fish (i + 1)) ∈ v.streams) →
      (fish 4 = S ∧ fish 4 = v.start) ∨ fish 4 = B)

theorem valley_of_five_lakes_streams (v : Valley) :
  v.valid → v.streams.card = 3 :=
sorry

end NUMINAMATH_CALUDE_valley_of_five_lakes_streams_l4039_403991


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l4039_403919

theorem lcm_of_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l4039_403919


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_l4039_403937

/-- Represents the grades of products -/
inductive Grade
| A
| B
| C
| D

/-- Represents a branch of the factory -/
structure Branch where
  name : String
  processingCost : ℝ
  gradeDistribution : Grade → ℝ

/-- Calculates the processing fee for a given grade -/
def processingFee (g : Grade) : ℝ :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Calculates the average profit per 100 products for a given branch -/
def averageProfit (b : Branch) : ℝ :=
  (processingFee Grade.A - b.processingCost) * b.gradeDistribution Grade.A +
  (processingFee Grade.B - b.processingCost) * b.gradeDistribution Grade.B +
  (processingFee Grade.C - b.processingCost) * b.gradeDistribution Grade.C +
  (processingFee Grade.D - b.processingCost) * b.gradeDistribution Grade.D

/-- Branch A of the factory -/
def branchA : Branch :=
  { name := "A"
    processingCost := 25
    gradeDistribution := fun g => match g with
      | Grade.A => 0.4
      | Grade.B => 0.2
      | Grade.C => 0.2
      | Grade.D => 0.2 }

/-- Branch B of the factory -/
def branchB : Branch :=
  { name := "B"
    processingCost := 20
    gradeDistribution := fun g => match g with
      | Grade.A => 0.28
      | Grade.B => 0.17
      | Grade.C => 0.34
      | Grade.D => 0.21 }

theorem branch_A_more_profitable :
  averageProfit branchA > averageProfit branchB :=
sorry

end NUMINAMATH_CALUDE_branch_A_more_profitable_l4039_403937


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4039_403911

theorem quadratic_inequality (x : ℝ) : x^2 - 7*x + 6 < 0 ↔ 1 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4039_403911


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l4039_403932

theorem smallest_number_satisfying_conditions : 
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬((m + 3) % 5 = 0 ∧ (m - 3) % 6 = 0)) ∧
    (n + 3) % 5 = 0 ∧ 
    (n - 3) % 6 = 0 ∧
    n = 27 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l4039_403932


namespace NUMINAMATH_CALUDE_diamond_two_three_l4039_403976

-- Define the diamond operation
def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_three_l4039_403976


namespace NUMINAMATH_CALUDE_souvenir_problem_l4039_403997

/-- Represents the number of ways to select souvenirs -/
def souvenir_selection_ways (total_types : ℕ) (expensive_types : ℕ) (cheap_types : ℕ) 
  (expensive_price : ℕ) (cheap_price : ℕ) (total_spent : ℕ) : ℕ :=
  (Nat.choose expensive_types 5) + 
  (Nat.choose expensive_types 4) * (Nat.choose cheap_types 2)

/-- The problem statement -/
theorem souvenir_problem : 
  souvenir_selection_ways 11 8 3 10 5 50 = 266 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_problem_l4039_403997


namespace NUMINAMATH_CALUDE_sixth_employee_salary_l4039_403913

/-- Given the salaries of 5 employees and the mean salary of all 6 employees,
    prove that the salary of the sixth employee is equal to the difference between
    the total salary of all 6 employees and the sum of the known 5 salaries. -/
theorem sixth_employee_salary
  (salary1 salary2 salary3 salary4 salary5 : ℝ)
  (mean_salary : ℝ)
  (h1 : salary1 = 1000)
  (h2 : salary2 = 2500)
  (h3 : salary3 = 3100)
  (h4 : salary4 = 1500)
  (h5 : salary5 = 2000)
  (h_mean : mean_salary = 2291.67)
  : ∃ (salary6 : ℝ),
    salary6 = 6 * mean_salary - (salary1 + salary2 + salary3 + salary4 + salary5) :=
by sorry

end NUMINAMATH_CALUDE_sixth_employee_salary_l4039_403913


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l4039_403929

/-- The angle between two 2D vectors is obtuse if and only if their dot product is negative -/
def is_obtuse_angle (a b : Fin 2 → ℝ) : Prop :=
  (a 0 * b 0 + a 1 * b 1) < 0

/-- The set of real numbers x for which the angle between (1, 3) and (x, -1) is obtuse -/
def obtuse_angle_set : Set ℝ :=
  {x : ℝ | is_obtuse_angle (![1, 3]) (![x, -1])}

theorem obtuse_angle_range :
  obtuse_angle_set = {x : ℝ | x < -1/3 ∨ (-1/3 < x ∧ x < 3)} := by sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l4039_403929


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4039_403950

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + 3*c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀ + 2*b₀ + 3*c₀ = 1 ∧ 1/a₀ + 2/b₀ + 3/c₀ = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4039_403950


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l4039_403944

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 20 and x - y = 4, then y = 24 when x = 4. -/
theorem inverse_proportion_problem (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k) 
    (h2 : x + y = 20) (h3 : x - y = 4) : 
    x = 4 → y = 24 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l4039_403944


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l4039_403904

theorem minimum_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  2 / x + 1 / y ≥ 9 / 2 ∧ (2 / x + 1 / y = 9 / 2 ↔ x = 2 / 3 ∧ y = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l4039_403904


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4039_403916

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 5 + a 12 - a 2 = 12 →
  a 7 + a 11 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4039_403916


namespace NUMINAMATH_CALUDE_only_third_proposition_true_l4039_403951

theorem only_third_proposition_true :
  ∃ (a b c d : ℝ),
    (∃ c, a > b ∧ c ≠ 0 ∧ ¬(a * c > b * c)) ∧
    (∃ c, a > b ∧ ¬(a * c^2 > b * c^2)) ∧
    (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
    (∃ a b, a > b ∧ ¬(1 / a < 1 / b)) ∧
    (∃ a b c d, a > b ∧ b > 0 ∧ c > d ∧ ¬(a * c > b * d)) :=
by sorry

end NUMINAMATH_CALUDE_only_third_proposition_true_l4039_403951


namespace NUMINAMATH_CALUDE_systematic_sample_property_l4039_403969

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Finset ℕ
  h_class_size : class_size > 0
  h_sample_size : sample_size > 0
  h_sample_size_le : sample_size ≤ class_size
  h_known_seats : known_seats.card < sample_size

/-- The seat number of the missing student in the systematic sample -/
def missing_seat (s : SystematicSample) : ℕ := sorry

/-- Theorem stating the property of the systematic sample -/
theorem systematic_sample_property (s : SystematicSample) 
  (h_seats : s.known_seats = {3, 15, 39, 51}) 
  (h_class_size : s.class_size = 60) 
  (h_sample_size : s.sample_size = 5) : 
  missing_seat s = 27 := by sorry

end NUMINAMATH_CALUDE_systematic_sample_property_l4039_403969


namespace NUMINAMATH_CALUDE_negation_of_universal_quantification_l4039_403912

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantification_l4039_403912


namespace NUMINAMATH_CALUDE_sequence_properties_l4039_403906

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, n > 1 → a (n - 1) + a (n + 1) > 2 * a n

theorem sequence_properties (a : ℕ+ → ℝ) (h : sequence_property a) :
  (a 2 > a 1 → ∀ n : ℕ+, n > 1 → a n > a (n - 1)) ∧
  (∃ d : ℝ, ∀ n : ℕ+, a n > a 1 + (n - 1) * d) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l4039_403906


namespace NUMINAMATH_CALUDE_S_periodic_l4039_403949

def S (x y z : ℤ) : ℤ × ℤ × ℤ := (x*y - x*z, y*z - y*x, z*x - z*y)

def S_power (n : ℕ) (a b c : ℤ) : ℤ × ℤ × ℤ :=
  match n with
  | 0 => (a, b, c)
  | n + 1 => S (S_power n a b c).1 (S_power n a b c).2.1 (S_power n a b c).2.2

def congruent_triple (u v : ℤ × ℤ × ℤ) (m : ℤ) : Prop :=
  u.1 % m = v.1 % m ∧ u.2.1 % m = v.2.1 % m ∧ u.2.2 % m = v.2.2 % m

theorem S_periodic (a b c : ℤ) (h : a * b * c > 1) :
  ∃ (n₀ k : ℕ), 0 < k ∧ k ≤ a * b * c ∧
  ∀ n ≥ n₀, congruent_triple (S_power (n + k) a b c) (S_power n a b c) (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_S_periodic_l4039_403949


namespace NUMINAMATH_CALUDE_difference_is_perfect_square_l4039_403993

theorem difference_is_perfect_square (m n : ℕ+) 
  (h : 2001 * m^2 + m = 2002 * n^2 + n) : 
  ∃ k : ℕ, (m : ℤ) - (n : ℤ) = k^2 :=
sorry

end NUMINAMATH_CALUDE_difference_is_perfect_square_l4039_403993


namespace NUMINAMATH_CALUDE_quadratic_sum_powers_divisibility_l4039_403974

/-- Represents a quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  p : ℤ
  q : ℤ

/-- Condition that the polynomial has a positive discriminant -/
def has_positive_discriminant (f : QuadraticPolynomial) : Prop :=
  f.p * f.p - 4 * f.q > 0

/-- Sum of the hundredth powers of the roots of a quadratic polynomial -/
noncomputable def sum_of_hundredth_powers (f : QuadraticPolynomial) : ℝ :=
  let α := (-f.p + Real.sqrt (f.p * f.p - 4 * f.q)) / 2
  let β := (-f.p - Real.sqrt (f.p * f.p - 4 * f.q)) / 2
  α^100 + β^100

/-- Main theorem statement -/
theorem quadratic_sum_powers_divisibility 
  (f : QuadraticPolynomial) 
  (h_disc : has_positive_discriminant f) 
  (h_p : f.p % 5 = 0) 
  (h_q : f.q % 5 = 0) : 
  ∃ (k : ℤ), sum_of_hundredth_powers f = k * (5^50 : ℝ) ∧ 
  ∀ (n : ℕ), n > 50 → ¬∃ (m : ℤ), sum_of_hundredth_powers f = m * (5^n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_powers_divisibility_l4039_403974


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l4039_403968

/-- If the terminal side of angle α passes through point (-1, 2), 
    then tan(α + π/4) = -1/3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) :
  (∃ (t : ℝ), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) →
  Real.tan (α + π/4) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l4039_403968


namespace NUMINAMATH_CALUDE_function_inequality_l4039_403956

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4039_403956


namespace NUMINAMATH_CALUDE_intersection_at_one_point_l4039_403900

theorem intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 2 * x + 3 = 3 * x + 4) ↔ b = -1/4 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_one_point_l4039_403900


namespace NUMINAMATH_CALUDE_bales_in_barn_l4039_403915

/-- The number of bales in the barn after Tim's addition -/
def total_bales (initial_bales added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- Theorem stating that the total number of bales after Tim's addition is 54 -/
theorem bales_in_barn (initial_bales added_bales : ℕ) 
  (h1 : initial_bales = 28)
  (h2 : added_bales = 26) :
  total_bales initial_bales added_bales = 54 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l4039_403915


namespace NUMINAMATH_CALUDE_part_one_part_two_l4039_403923

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 2| - |x + 2|

-- Part 1
theorem part_one : 
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} :=
sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) → (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4039_403923


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4039_403957

theorem sqrt_equation_solution (x : ℝ) (h : x > 6) :
  Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3 ↔ x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4039_403957


namespace NUMINAMATH_CALUDE_dc_length_l4039_403975

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def conditions (q : Quadrilateral) : Prop :=
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  let sinAngle := λ p₁ p₂ p₃ : ℝ × ℝ => 
    let v1 := (p₂.1 - p₁.1, p₂.2 - p₁.2)
    let v2 := (p₃.1 - p₁.1, p₃.2 - p₁.2)
    (v1.1 * v2.2 - v1.2 * v2.1) / (dist p₁ p₂ * dist p₁ p₃)
  dist q.A q.B = 30 ∧
  (q.A.1 - q.D.1) * (q.B.1 - q.D.1) + (q.A.2 - q.D.2) * (q.B.2 - q.D.2) = 0 ∧
  sinAngle q.B q.A q.D = 4/5 ∧
  sinAngle q.B q.C q.D = 1/5

-- State the theorem
theorem dc_length (q : Quadrilateral) (h : conditions q) : 
  dist q.D q.C = 48 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_dc_length_l4039_403975


namespace NUMINAMATH_CALUDE_sophie_total_spent_l4039_403983

def cupcake_quantity : ℕ := 5
def cupcake_price : ℚ := 2

def doughnut_quantity : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_slice_quantity : ℕ := 4
def apple_pie_slice_price : ℚ := 2

def cookie_quantity : ℕ := 15
def cookie_price : ℚ := 0.60

def total_spent : ℚ := cupcake_quantity * cupcake_price + 
                        doughnut_quantity * doughnut_price + 
                        apple_pie_slice_quantity * apple_pie_slice_price + 
                        cookie_quantity * cookie_price

theorem sophie_total_spent : total_spent = 33 := by
  sorry

end NUMINAMATH_CALUDE_sophie_total_spent_l4039_403983


namespace NUMINAMATH_CALUDE_non_adjacent_book_selection_l4039_403995

/-- The number of books on the shelf -/
def total_books : ℕ := 12

/-- The number of books to be chosen -/
def books_to_choose : ℕ := 5

/-- The theorem stating that the number of ways to choose 5 books out of 12
    such that no two chosen books are adjacent is equal to C(8,5) -/
theorem non_adjacent_book_selection :
  (Nat.choose (total_books - books_to_choose + 1) books_to_choose) =
  (Nat.choose 8 5) := by sorry

end NUMINAMATH_CALUDE_non_adjacent_book_selection_l4039_403995


namespace NUMINAMATH_CALUDE_triangle_properties_l4039_403996

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b - (1/2) * t.c = t.a * Real.cos t.C)
  (h2 : 4 * (t.b + t.c) = 3 * t.b * t.c)
  (h3 : t.a = 2 * Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4039_403996


namespace NUMINAMATH_CALUDE_vector_magnitude_l4039_403947

/-- Given plane vectors a, b, and c, if (a + b) is parallel to c, then the magnitude of c is 2√17. -/
theorem vector_magnitude (a b c : ℝ × ℝ) (h : ∃ (t : ℝ), a + b = t • c) : 
  a = (-1, 1) → b = (2, 3) → c.1 = -2 → ‖c‖ = 2 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l4039_403947


namespace NUMINAMATH_CALUDE_work_completion_time_l4039_403980

/-- Proves the time taken to complete a work when two people work together -/
theorem work_completion_time (rahul_rate meena_rate : ℚ) 
  (hrahul : rahul_rate = 1 / 5)
  (hmeena : meena_rate = 1 / 10) :
  1 / (rahul_rate + meena_rate) = 10 / 3 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l4039_403980


namespace NUMINAMATH_CALUDE_cistern_length_l4039_403907

/-- The length of a cistern with given dimensions and wet surface area -/
theorem cistern_length (width : ℝ) (depth : ℝ) (wet_surface_area : ℝ) 
  (h1 : width = 2)
  (h2 : depth = 1.25)
  (h3 : wet_surface_area = 23) :
  ∃ length : ℝ, 
    wet_surface_area = length * width + 2 * length * depth + 2 * width * depth ∧ 
    length = 4 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_l4039_403907


namespace NUMINAMATH_CALUDE_octagon_area_l4039_403963

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
theorem octagon_area (r : ℝ) (h : r = 3) : 
  let s := 2 * r * Real.sqrt ((1 - 1 / Real.sqrt 2) / 2)
  let area_triangle := 1 / 2 * s^2 * (1 / Real.sqrt 2)
  8 * area_triangle = 48 * (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l4039_403963


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l4039_403952

theorem circle_radius_from_area (r : ℝ) : r > 0 → π * r^2 = 9 * π → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l4039_403952


namespace NUMINAMATH_CALUDE_fourth_root_of_256000000_l4039_403964

theorem fourth_root_of_256000000 : (256000000 : ℝ) ^ (1/4 : ℝ) = 40 * (10 : ℝ).sqrt := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_256000000_l4039_403964


namespace NUMINAMATH_CALUDE_aunt_gemma_dog_food_l4039_403917

/-- Calculates the amount of food each dog consumes per meal given the number of dogs, 
    feedings per day, number of sacks, weight per sack, and days of food supply. -/
def food_per_dog_per_meal (num_dogs : ℕ) (feedings_per_day : ℕ) 
                           (num_sacks : ℕ) (weight_per_sack : ℕ) 
                           (days_of_supply : ℕ) : ℕ :=
  (num_sacks * weight_per_sack * 1000) / (num_dogs * feedings_per_day * days_of_supply)

/-- Theorem stating that given the specific conditions, each dog consumes 250 grams per meal. -/
theorem aunt_gemma_dog_food : 
  food_per_dog_per_meal 4 2 2 50 50 = 250 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gemma_dog_food_l4039_403917


namespace NUMINAMATH_CALUDE_wall_length_calculation_l4039_403948

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) : 
  mirror_side = 21 →
  wall_width = 28 →
  (mirror_side ^ 2) * 2 = wall_width * (31.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l4039_403948


namespace NUMINAMATH_CALUDE_acute_angle_solution_l4039_403940

theorem acute_angle_solution (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) :
  Real.cos α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1 →
  α = 40 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_solution_l4039_403940


namespace NUMINAMATH_CALUDE_sqrt_expressions_l4039_403966

theorem sqrt_expressions :
  (2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2) ∧
  (Real.sqrt ((-3)^2) ≠ -3) ∧
  (Real.sqrt 24 / Real.sqrt 6 ≠ 4) ∧
  (Real.sqrt 3 + Real.sqrt 2 ≠ Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l4039_403966


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4039_403994

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = x^2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4039_403994


namespace NUMINAMATH_CALUDE_tenth_difference_optimal_number_l4039_403961

/-- A positive integer that can be expressed as the difference of squares of two positive integers m and n, where m - n > 1 -/
def DifferenceOptimalNumber (k : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ k = m^2 - n^2

/-- The sequence of difference optimal numbers in ascending order -/
def DifferenceOptimalSequence : ℕ → ℕ :=
  sorry

theorem tenth_difference_optimal_number :
  DifferenceOptimalNumber (DifferenceOptimalSequence 10) ∧ 
  DifferenceOptimalSequence 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_tenth_difference_optimal_number_l4039_403961


namespace NUMINAMATH_CALUDE_two_sqrt_two_gt_sqrt_seven_l4039_403992

theorem two_sqrt_two_gt_sqrt_seven : 2 * Real.sqrt 2 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_two_gt_sqrt_seven_l4039_403992


namespace NUMINAMATH_CALUDE_divisibility_condition_l4039_403938

theorem divisibility_condition (a b : ℕ+) :
  (a.val * b.val^2 + b.val + 7) ∣ (a.val^2 * b.val + a.val + b.val) →
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l4039_403938


namespace NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_for_subset_condition_l4039_403959

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x - 1|

theorem solution_set_for_negative_one :
  {x : ℝ | f (-1) x ≤ 2} = {x : ℝ | x = 1/2 ∨ x = -1/2} := by sorry

theorem range_of_a_for_subset_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1/2) 1, f a x ≤ |2*x + 1|) → a ∈ Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_for_subset_condition_l4039_403959


namespace NUMINAMATH_CALUDE_min_f_1998_l4039_403960

/-- A function from positive integers to positive integers satisfying the given property -/
def SpecialFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ (s t : ℕ+), f (t^2 * f s) = s * (f t)^2

/-- The theorem stating the minimum value of f(1998) -/
theorem min_f_1998 (f : ℕ+ → ℕ+) (h : SpecialFunction f) :
  ∃ (m : ℕ+), f 1998 = m ∧ ∀ (g : ℕ+ → ℕ+), SpecialFunction g → m ≤ g 1998 :=
sorry

end NUMINAMATH_CALUDE_min_f_1998_l4039_403960


namespace NUMINAMATH_CALUDE_divisors_of_8_factorial_l4039_403933

theorem divisors_of_8_factorial : Nat.card (Nat.divisors (Nat.factorial 8)) = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8_factorial_l4039_403933


namespace NUMINAMATH_CALUDE_train_speed_proof_l4039_403905

/-- Proves that a train crossing a 320-meter platform in 34 seconds and passing a stationary man in 18 seconds has a speed of 72 km/h -/
theorem train_speed_proof (platform_length : ℝ) (platform_crossing_time : ℝ) (man_passing_time : ℝ) :
  platform_length = 320 →
  platform_crossing_time = 34 →
  man_passing_time = 18 →
  ∃ (train_speed : ℝ),
    train_speed * man_passing_time = train_speed * platform_crossing_time - platform_length ∧
    train_speed * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_proof_l4039_403905


namespace NUMINAMATH_CALUDE_multiply_binomial_l4039_403987

theorem multiply_binomial (x : ℝ) : (-2*x)*(x - 3) = -2*x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_multiply_binomial_l4039_403987


namespace NUMINAMATH_CALUDE_equation_solution_l4039_403909

theorem equation_solution :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4039_403909


namespace NUMINAMATH_CALUDE_tower_surface_area_l4039_403981

/-- Represents a cube in the tower --/
structure Cube where
  volume : ℕ
  sideLength : ℕ
  deriving Repr

/-- Represents the tower of cubes --/
def Tower : List Cube := [
  { volume := 343, sideLength := 7 },
  { volume := 125, sideLength := 5 },
  { volume := 27,  sideLength := 3 },
  { volume := 64,  sideLength := 4 },
  { volume := 1,   sideLength := 1 }
]

/-- Calculates the visible surface area of a cube in the tower --/
def visibleSurfaceArea (cube : Cube) (aboveCube : Option Cube) : ℕ := sorry

/-- Calculates the total visible surface area of the tower --/
def totalVisibleSurfaceArea (tower : List Cube) : ℕ := sorry

/-- Theorem stating that the total visible surface area of the tower is 400 square units --/
theorem tower_surface_area : totalVisibleSurfaceArea Tower = 400 := by sorry

end NUMINAMATH_CALUDE_tower_surface_area_l4039_403981


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l4039_403935

-- Define the common ratios and terms of the geometric sequences
variables {k p r : ℝ} {a₂ a₃ b₂ b₃ : ℝ}

-- Define the geometric sequences
def is_geometric_sequence (k p a₂ a₃ : ℝ) : Prop :=
  a₂ = k * p ∧ a₃ = k * p^2

-- State the theorem
theorem sum_of_common_ratios
  (h₁ : is_geometric_sequence k p a₂ a₃)
  (h₂ : is_geometric_sequence k r b₂ b₃)
  (h₃ : p ≠ r)
  (h₄ : k ≠ 0)
  (h₅ : a₃ - b₃ = 4 * (a₂ - b₂)) :
  p + r = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l4039_403935


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l4039_403999

theorem dvd_pack_cost (total_amount : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) :
  total_amount = 104 →
  num_packs = 4 →
  cost_per_pack = total_amount / num_packs →
  cost_per_pack = 26 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l4039_403999


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l4039_403901

structure Orangeade where
  orange_juice : ℝ
  water : ℝ
  price_per_glass : ℝ
  glasses_sold : ℝ

def revenue (o : Orangeade) : ℝ := o.price_per_glass * o.glasses_sold

theorem orangeade_price_day2 (day1 day2 : Orangeade) :
  day1.orange_juice > 0 →
  day1.orange_juice = day1.water →
  day2.orange_juice = day1.orange_juice →
  day2.water = 2 * day1.water →
  day1.price_per_glass = 0.9 →
  revenue day1 = revenue day2 →
  day2.glasses_sold = (3/2) * day1.glasses_sold →
  day2.price_per_glass = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l4039_403901


namespace NUMINAMATH_CALUDE_dinner_payment_difference_l4039_403954

/-- The problem of calculating the difference in payment between John and Jane --/
theorem dinner_payment_difference :
  let original_price : ℝ := 36.000000000000036
  let discount_rate : ℝ := 0.10
  let tip_rate : ℝ := 0.15
  let discounted_price := original_price * (1 - discount_rate)
  let john_tip := original_price * tip_rate
  let jane_tip := discounted_price * tip_rate
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.5400000000000023 :=
by sorry

end NUMINAMATH_CALUDE_dinner_payment_difference_l4039_403954


namespace NUMINAMATH_CALUDE_pyramid_volume_scaling_l4039_403982

theorem pyramid_volume_scaling (V₀ : ℝ) (l w h : ℝ) : 
  V₀ = (1/3) * l * w * h → 
  V₀ = 60 → 
  (1/3) * (3*l) * (4*w) * (2*h) = 1440 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_scaling_l4039_403982


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_l4039_403962

theorem rectangle_square_overlap (s w h : ℝ) 
  (h1 : 0.4 * s^2 = 0.25 * w * h) 
  (h2 : w = 4 * h) : 
  w / h = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_l4039_403962


namespace NUMINAMATH_CALUDE_mona_group_size_l4039_403942

/-- The number of groups Mona joined --/
def num_groups : ℕ := 9

/-- The number of unique players Mona grouped with --/
def unique_players : ℕ := 33

/-- The number of non-unique player slots --/
def non_unique_slots : ℕ := 3

/-- The number of players in each group, including Mona --/
def players_per_group : ℕ := 5

theorem mona_group_size :
  (num_groups * (players_per_group - 1)) - non_unique_slots = unique_players :=
by sorry

end NUMINAMATH_CALUDE_mona_group_size_l4039_403942


namespace NUMINAMATH_CALUDE_problem_l4039_403967

theorem problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1/a + 1/b) :
  (a + b ≥ 2) ∧ ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_l4039_403967


namespace NUMINAMATH_CALUDE_rebecca_bought_four_tent_stakes_l4039_403922

/-- The number of tent stakes bought by Rebecca. -/
def tent_stakes : ℕ := sorry

/-- The number of packets of drink mix bought by Rebecca. -/
def drink_mix : ℕ := sorry

/-- The number of bottles of water bought by Rebecca. -/
def water_bottles : ℕ := sorry

/-- The total number of items bought by Rebecca. -/
def total_items : ℕ := 22

/-- Theorem stating that Rebecca bought 4 tent stakes. -/
theorem rebecca_bought_four_tent_stakes :
  (drink_mix = 3 * tent_stakes) ∧
  (water_bottles = tent_stakes + 2) ∧
  (tent_stakes + drink_mix + water_bottles = total_items) →
  tent_stakes = 4 := by
sorry

end NUMINAMATH_CALUDE_rebecca_bought_four_tent_stakes_l4039_403922


namespace NUMINAMATH_CALUDE_average_age_problem_l4039_403946

theorem average_age_problem (a c : ℝ) : 
  (a + c) / 2 = 29 →
  ((a + c) + 26) / 3 = 28 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l4039_403946


namespace NUMINAMATH_CALUDE_lukes_father_twenty_bills_l4039_403927

def mother_fifty : ℕ := 1
def mother_twenty : ℕ := 2
def mother_ten : ℕ := 3

def father_fifty : ℕ := 4
def father_ten : ℕ := 1

def school_fee : ℕ := 350

theorem lukes_father_twenty_bills :
  ∃ (father_twenty : ℕ),
    50 * mother_fifty + 20 * mother_twenty + 10 * mother_ten +
    50 * father_fifty + 20 * father_twenty + 10 * father_ten = school_fee ∧
    father_twenty = 1 :=
by sorry

end NUMINAMATH_CALUDE_lukes_father_twenty_bills_l4039_403927
