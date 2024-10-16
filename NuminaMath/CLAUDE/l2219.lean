import Mathlib

namespace NUMINAMATH_CALUDE_triangle_sine_ratio_l2219_221946

theorem triangle_sine_ratio (A B C : ℝ) (h1 : 0 < A ∧ A < π)
                                       (h2 : 0 < B ∧ B < π)
                                       (h3 : 0 < C ∧ C < π)
                                       (h4 : A + B + C = π)
                                       (h5 : Real.sin A / Real.sin B = 6/5)
                                       (h6 : Real.sin B / Real.sin C = 5/4) :
  Real.sin B = 5 * Real.sqrt 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_ratio_l2219_221946


namespace NUMINAMATH_CALUDE_excircle_lengths_sum_gt_semiperimeter_l2219_221985

/-- Given a triangle with sides a, b, and c, and semi-perimeter p,
    BB' and CC' are specific lengths related to the excircles of the triangle. -/
def triangle_excircle_lengths (a b c : ℝ) (p : ℝ) (BB' CC' : ℝ) : Prop :=
  p = (a + b + c) / 2 ∧ BB' = p - a ∧ CC' = p - b

/-- The sum of BB' and CC' is greater than the semi-perimeter p for any triangle. -/
theorem excircle_lengths_sum_gt_semiperimeter 
  {a b c p BB' CC' : ℝ} 
  (h : triangle_excircle_lengths a b c p BB' CC') :
  BB' + CC' > p :=
sorry

end NUMINAMATH_CALUDE_excircle_lengths_sum_gt_semiperimeter_l2219_221985


namespace NUMINAMATH_CALUDE_energy_drink_cost_l2219_221972

/-- The cost of an energy drink bottle given the sales and purchases of a basketball team. -/
theorem energy_drink_cost (cupcakes : ℕ) (cupcake_price : ℚ) 
  (cookies : ℕ) (cookie_price : ℚ)
  (basketballs : ℕ) (basketball_price : ℚ)
  (energy_drinks : ℕ) :
  cupcakes = 50 →
  cupcake_price = 2 →
  cookies = 40 →
  cookie_price = 1/2 →
  basketballs = 2 →
  basketball_price = 40 →
  energy_drinks = 20 →
  (cupcakes : ℚ) * cupcake_price + (cookies : ℚ) * cookie_price 
    - (basketballs : ℚ) * basketball_price = (energy_drinks : ℚ) * 2 :=
by sorry

end NUMINAMATH_CALUDE_energy_drink_cost_l2219_221972


namespace NUMINAMATH_CALUDE_shoes_theorem_l2219_221996

def shoes_problem (bonny becky bobby cherry diane : ℚ) : Prop :=
  -- Conditions
  bonny = 13 ∧
  bonny = 2 * becky - 5 ∧
  bobby = 3.5 * becky ∧
  cherry = bonny + becky + 4.5 ∧
  diane = 3 * cherry - 2 - 3 ∧
  -- Conclusion
  ⌊bonny + becky + bobby + cherry + diane⌋ = 154

theorem shoes_theorem : ∃ bonny becky bobby cherry diane : ℚ, 
  shoes_problem bonny becky bobby cherry diane := by
  sorry

end NUMINAMATH_CALUDE_shoes_theorem_l2219_221996


namespace NUMINAMATH_CALUDE_square_area_ratio_l2219_221900

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2219_221900


namespace NUMINAMATH_CALUDE_cookies_distribution_l2219_221942

theorem cookies_distribution (cookies_per_person : ℝ) (total_cookies : ℕ) (h1 : cookies_per_person = 24.0) (h2 : total_cookies = 144) :
  (total_cookies : ℝ) / cookies_per_person = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_distribution_l2219_221942


namespace NUMINAMATH_CALUDE_sector_area_l2219_221909

/-- Given a circular sector with circumference 6 and central angle 1 radian, its area is 2 -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) :
  circumference = 6 →
  central_angle = 1 →
  area = 2 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l2219_221909


namespace NUMINAMATH_CALUDE_remainder_theorem_l2219_221901

theorem remainder_theorem (n m : ℤ) (q2 : ℤ) 
  (h1 : n % 11 = 1) 
  (h2 : m % 17 = 3) 
  (h3 : m = 17 * q2 + 3) : 
  (5 * n + 3 * m) % 11 = (3 + 7 * q2) % 11 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2219_221901


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2219_221906

theorem mean_equality_implies_z_value : ∃ z : ℚ, 
  (8 + 15 + 27) / 3 = (18 + z) / 2 → z = 46 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2219_221906


namespace NUMINAMATH_CALUDE_girls_in_class_l2219_221975

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 35) (h2 : ratio_girls = 3) (h3 : ratio_boys = 4) : 
  (total * ratio_girls) / (ratio_girls + ratio_boys) = 15 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_class_l2219_221975


namespace NUMINAMATH_CALUDE_min_years_plan_b_exceeds_plan_a_l2219_221981

-- Define the investment amount for Plan A
def plan_a_investment : ℕ := 1000000

-- Define the initial investment and yearly increase for Plan B
def plan_b_initial : ℕ := 100000
def plan_b_increase : ℕ := 100000

-- Function to calculate the total investment of Plan B after n years
def plan_b_total (n : ℕ) : ℕ :=
  n * (2 * plan_b_initial + (n - 1) * plan_b_increase) / 2

-- Theorem stating the minimum number of years for Plan B to match or exceed Plan A
theorem min_years_plan_b_exceeds_plan_a :
  ∃ n : ℕ, (∀ k : ℕ, k < n → plan_b_total k < plan_a_investment) ∧
           plan_b_total n ≥ plan_a_investment ∧
           n = 5 :=
sorry

end NUMINAMATH_CALUDE_min_years_plan_b_exceeds_plan_a_l2219_221981


namespace NUMINAMATH_CALUDE_equal_area_divide_sum_of_squares_l2219_221970

-- Define the region S as a set of points in the plane
def S : Set (ℝ × ℝ) := sorry

-- Define the line m with slope 4
def m : Set (ℝ × ℝ) := {(x, y) | 4 * x = y + c} where c : ℝ := sorry

-- Define the property of m dividing S into two equal areas
def divides_equally (l : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop := sorry

-- Define the equation of line m in the form ax = by + c
def line_equation (a b c : ℕ) : Set (ℝ × ℝ) := {(x, y) | a * x = b * y + c}

-- Main theorem
theorem equal_area_divide_sum_of_squares :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.gcd a (Nat.gcd b c) = 1 ∧
    divides_equally (line_equation a b c) S ∧
    m = line_equation a b c ∧
    a^2 + b^2 + c^2 = 65 := by sorry

end NUMINAMATH_CALUDE_equal_area_divide_sum_of_squares_l2219_221970


namespace NUMINAMATH_CALUDE_modulo_evaluation_l2219_221930

theorem modulo_evaluation : (203 * 19 - 22 * 8 + 6) % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_modulo_evaluation_l2219_221930


namespace NUMINAMATH_CALUDE_unit_square_folding_l2219_221935

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (P Q R : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    P.x = Q.x + t * (R.x - Q.x) ∧
    P.y = Q.y + t * (R.y - Q.y)

/-- Checks if two line segments intersect -/
def segmentsIntersect (P Q R S : Point) : Prop :=
  ∃ I : Point, isOnSegment I P Q ∧ isOnSegment I R S

theorem unit_square_folding (ABCD : UnitSquare) 
  (E : Point) (F : Point) 
  (hE : isOnSegment E ABCD.A ABCD.B) 
  (hF : isOnSegment F ABCD.C ABCD.B) 
  (hF_mid : F.x = 1 ∧ F.y = 1/2) 
  (hFold : segmentsIntersect ABCD.A ABCD.D E F ∧ 
           segmentsIntersect ABCD.C ABCD.D E F) : 
  E.x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unit_square_folding_l2219_221935


namespace NUMINAMATH_CALUDE_problem_solution_l2219_221964

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (a + b > 1/2) ∧ 
  (a + b < 1) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 → 2/x + 1/y ≥ 8) ∧
  (a * b ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2219_221964


namespace NUMINAMATH_CALUDE_bird_cost_problem_l2219_221994

/-- Calculates the cost per bird given the total money and number of birds -/
def cost_per_bird (total_money : ℚ) (num_birds : ℕ) : ℚ :=
  total_money / num_birds

/-- The problem statement -/
theorem bird_cost_problem :
  let total_money : ℚ := 4 * 50
  let total_wings : ℕ := 20
  let wings_per_bird : ℕ := 2
  let num_birds : ℕ := total_wings / wings_per_bird
  cost_per_bird total_money num_birds = 20 := by
  sorry

end NUMINAMATH_CALUDE_bird_cost_problem_l2219_221994


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2219_221907

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_1_2 : a 1 + a 2 = 1)
  (h_sum_3_4 : a 3 + a 4 = 5) :
  a 5 = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2219_221907


namespace NUMINAMATH_CALUDE_odd_numbers_product_equality_l2219_221927

theorem odd_numbers_product_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_odd_numbers_product_equality_l2219_221927


namespace NUMINAMATH_CALUDE_not_all_numbers_representable_l2219_221938

theorem not_all_numbers_representable :
  ∃ k : ℕ, k % 6 = 0 ∧ k > 1000 ∧
  ∀ m n : ℕ, k ≠ n * (n + 1) * (n + 2) * (n + 3) * (n + 4) - m * (m + 1) * (m + 2) :=
by sorry

end NUMINAMATH_CALUDE_not_all_numbers_representable_l2219_221938


namespace NUMINAMATH_CALUDE_alien_head_volume_l2219_221999

theorem alien_head_volume (surface_area : ℝ) (h : surface_area = 150) :
  let side_length := Real.sqrt (surface_area / 6)
  let volume := side_length ^ 3
  volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_alien_head_volume_l2219_221999


namespace NUMINAMATH_CALUDE_binary_to_hexadecimal_conversion_l2219_221932

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  sorry

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hexadecimal (decimal : ℕ) : List ℕ :=
  sorry

theorem binary_to_hexadecimal_conversion :
  let binary : List Bool := [true, false, true, true, false, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let hexadecimal : List ℕ := decimal_to_hexadecimal decimal
  hexadecimal = [2, 2, 5] := by sorry

end NUMINAMATH_CALUDE_binary_to_hexadecimal_conversion_l2219_221932


namespace NUMINAMATH_CALUDE_annika_hiking_rate_l2219_221952

/-- Annika's hiking problem -/
theorem annika_hiking_rate (initial_distance : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  initial_distance = 2.75 →
  total_distance = 3.5 →
  total_time = 51 →
  (total_time / (2 * (total_distance - initial_distance))) = 34 :=
by
  sorry

#check annika_hiking_rate

end NUMINAMATH_CALUDE_annika_hiking_rate_l2219_221952


namespace NUMINAMATH_CALUDE_complex_subtraction_l2219_221980

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 4*I) :
  a - 3*b = -1 - 15*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2219_221980


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l2219_221917

theorem cuboid_edge_length (x : ℝ) : 
  x > 0 → 2 * x * 3 = 30 → x = 5 := by sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l2219_221917


namespace NUMINAMATH_CALUDE_unique_prime_sum_difference_l2219_221957

theorem unique_prime_sum_difference : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ x y z w : ℕ, Prime x ∧ Prime y ∧ Prime z ∧ Prime w ∧ 
    p = x + y ∧ p = z - w) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_difference_l2219_221957


namespace NUMINAMATH_CALUDE_billy_hike_distance_l2219_221955

theorem billy_hike_distance :
  let east_distance : ℝ := 7
  let north_distance : ℝ := 3 * Real.sqrt 3
  let total_distance : ℝ := Real.sqrt (east_distance^2 + north_distance^2)
  total_distance = 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_billy_hike_distance_l2219_221955


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2219_221943

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 4 →
  e = 2 * (-a - c) →
  Complex.mk (a + c + e) (b + d + f) = Complex.I * 6 →
  d + f = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2219_221943


namespace NUMINAMATH_CALUDE_mikes_earnings_l2219_221967

def working_game_prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]

theorem mikes_earnings : List.sum working_game_prices = 75 := by
  sorry

end NUMINAMATH_CALUDE_mikes_earnings_l2219_221967


namespace NUMINAMATH_CALUDE_right_triangle_construction_impossibility_l2219_221916

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a point being inside a circle
def IsInside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), c = Circle center radius ∧
    (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2

-- Define a circle with diameter AB
def CircleWithDiameter (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  Circle ((A.1 + B.1)/2, (A.2 + B.2)/2) (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2)

-- Define intersection of two sets
def Intersects (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ s1 ∧ p ∈ s2

-- Main theorem
theorem right_triangle_construction_impossibility
  (C : Set (ℝ × ℝ)) (A B : ℝ × ℝ)
  (h_circle : ∃ center radius, C = Circle center radius)
  (h_A_inside : IsInside A C)
  (h_B_inside : IsInside B C) :
  (¬ ∃ P Q R : ℝ × ℝ,
    P ∈ C ∧ Q ∈ C ∧ R ∈ C ∧
    (A.1 - P.1) * (Q.1 - P.1) + (A.2 - P.2) * (Q.2 - P.2) = 0 ∧
    (B.1 - P.1) * (R.1 - P.1) + (B.2 - P.2) * (R.2 - P.2) = 0 ∧
    (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  ↔
  ¬ Intersects (CircleWithDiameter A B) C :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_construction_impossibility_l2219_221916


namespace NUMINAMATH_CALUDE_horner_method_evaluation_l2219_221947

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem horner_method_evaluation : f 5 = 4881 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_evaluation_l2219_221947


namespace NUMINAMATH_CALUDE_angle_complement_half_supplement_is_zero_l2219_221921

theorem angle_complement_half_supplement_is_zero (x : ℝ) :
  (90 - x) = (1/2) * (180 - x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_half_supplement_is_zero_l2219_221921


namespace NUMINAMATH_CALUDE_quadratic_not_through_point_l2219_221969

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_not_through_point (p q : ℝ) :
  f p q 1 = 1 → f p q 3 = 1 → f p q 4 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_not_through_point_l2219_221969


namespace NUMINAMATH_CALUDE_expression_evaluation_l2219_221974

theorem expression_evaluation : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = 
  (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2219_221974


namespace NUMINAMATH_CALUDE_remainder_of_12345678_div_9_l2219_221914

theorem remainder_of_12345678_div_9 : 12345678 % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_12345678_div_9_l2219_221914


namespace NUMINAMATH_CALUDE_range_of_a_correct_l2219_221913

/-- Proposition p: For all x ∈ ℝ, ax^2 + ax + 1 > 0 always holds -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The function f(x) = 4x^2 - ax is monotonically increasing on [1, +∞) -/
def q (a : ℝ) : Prop := ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := {a : ℝ | a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)}

theorem range_of_a_correct (a : ℝ) : (p a ∨ q a) ∧ ¬(p a) → a ∈ range_of_a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_correct_l2219_221913


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l2219_221941

/-- Given a triangle ABC with sides BC = 5 and AC = 4, and cos(A - B) = 7/8, prove that cos C = -1/4 -/
theorem triangle_cosine_theorem (A B C : ℝ) (h1 : BC = 5) (h2 : AC = 4) (h3 : Real.cos (A - B) = 7/8) :
  Real.cos C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l2219_221941


namespace NUMINAMATH_CALUDE_vacuum_cleaner_cost_l2219_221939

theorem vacuum_cleaner_cost (dishwasher_cost coupon_value total_spent : ℕ) 
  (h1 : dishwasher_cost = 450)
  (h2 : coupon_value = 75)
  (h3 : total_spent = 625) :
  ∃ (vacuum_cost : ℕ), vacuum_cost = 250 ∧ vacuum_cost + dishwasher_cost - coupon_value = total_spent :=
by sorry

end NUMINAMATH_CALUDE_vacuum_cleaner_cost_l2219_221939


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l2219_221978

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l2219_221978


namespace NUMINAMATH_CALUDE_decimal_power_equivalence_l2219_221950

theorem decimal_power_equivalence : (1 / 10 : ℝ) ^ 2 = 0.010000000000000002 := by
  sorry

end NUMINAMATH_CALUDE_decimal_power_equivalence_l2219_221950


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2219_221918

/-- Given a sample with 16 units of model A, and the ratio of quantities of 
    models A, B, and C being 2:3:5, the total sample size n is 80. -/
theorem sample_size_calculation (model_a_count : ℕ) (ratio_a ratio_b ratio_c : ℕ) :
  model_a_count = 16 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 5 →
  (ratio_a : ℚ) / (ratio_a + ratio_b + ratio_c : ℚ) * (model_a_count * (ratio_a + ratio_b + ratio_c) / ratio_a) = 80 :=
by sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2219_221918


namespace NUMINAMATH_CALUDE_triangle_f_sign_l2219_221931

/-- Triangle ABC with sides a ≤ b ≤ c, circumradius R, and inradius r -/
structure Triangle where
  a : Real
  b : Real
  c : Real
  R : Real
  r : Real
  h_sides : a ≤ b ∧ b ≤ c
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0

/-- The function f defined for the triangle -/
def f (t : Triangle) : Real := t.a + t.b - 2 * t.R - 2 * t.r

/-- Angle C of the triangle -/
noncomputable def angle_C (t : Triangle) : Real := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

theorem triangle_f_sign (t : Triangle) :
  (f t > 0 ↔ angle_C t < Real.pi / 2) ∧
  (f t = 0 ↔ angle_C t = Real.pi / 2) ∧
  (f t < 0 ↔ angle_C t > Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_triangle_f_sign_l2219_221931


namespace NUMINAMATH_CALUDE_teacher_estimate_difference_l2219_221995

/-- The difference between the teacher's estimated increase and the actual increase in exam scores -/
theorem teacher_estimate_difference (expected_increase actual_increase : ℕ) 
  (h1 : expected_increase = 2152)
  (h2 : actual_increase = 1264) : 
  expected_increase - actual_increase = 888 := by
  sorry

end NUMINAMATH_CALUDE_teacher_estimate_difference_l2219_221995


namespace NUMINAMATH_CALUDE_class_size_l2219_221912

theorem class_size (debate_only : ℕ) (singing_only : ℕ) (both : ℕ)
  (h1 : debate_only = 10)
  (h2 : singing_only = 18)
  (h3 : both = 17) :
  debate_only + singing_only + both - both = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l2219_221912


namespace NUMINAMATH_CALUDE_system_solutions_l2219_221968

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x + y + z = 17 ∧ x*y + y*z + z*x = 94 ∧ x*y*z = 168

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 4, -12), (1, -12, 4), (4, 1, -12), (4, -12, 1), (-12, 1, 4), (-12, 4, 1)}

-- Theorem statement
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l2219_221968


namespace NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l2219_221910

/-- Given a wire cut into two pieces of lengths a and b, where a forms a square and b forms a regular octagon with equal perimeters, prove that a/b = 1 -/
theorem wire_cut_square_octagon_ratio (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 4 * (a / 4) = 8 * (b / 8)) : a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l2219_221910


namespace NUMINAMATH_CALUDE_circle_inside_polygon_l2219_221987

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : Bool  -- We assume this is true for a convex polygon

/-- The area of a convex polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- The perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- The distance from a point to a line segment -/
def distance_to_side (point : ℝ × ℝ) (side : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: In any convex polygon, there exists a point that is at least A/P distance away from all sides -/
theorem circle_inside_polygon (p : ConvexPolygon) :
  ∃ (center : ℝ × ℝ), 
    (∀ (side : (ℝ × ℝ) × (ℝ × ℝ)), 
      side.1 ∈ p.vertices ∧ side.2 ∈ p.vertices →
      distance_to_side center side ≥ area p / perimeter p) :=
sorry

end NUMINAMATH_CALUDE_circle_inside_polygon_l2219_221987


namespace NUMINAMATH_CALUDE_f_deriv_negative_one_eq_negative_two_l2219_221991

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- State the theorem
theorem f_deriv_negative_one_eq_negative_two 
  (a b c : ℝ) (h : f_deriv a b 1 = 2) : f_deriv a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_deriv_negative_one_eq_negative_two_l2219_221991


namespace NUMINAMATH_CALUDE_line_circle_intersection_equilateral_l2219_221973

/-- Given a line and a circle in a Cartesian coordinate system, 
    if they intersect to form an equilateral triangle with the circle's center,
    then the parameter 'a' in the line equation must be 0. -/
theorem line_circle_intersection_equilateral (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + A.2 - 2 = 0) ∧ 
    (a * B.1 + B.2 - 2 = 0) ∧
    ((A.1 - 1)^2 + (A.2 - a)^2 = 16/3) ∧
    ((B.1 - 1)^2 + (B.2 - a)^2 = 16/3) ∧
    (let C : ℝ × ℝ := (1, a);
     (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
     (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
     (B.1 - C.1)^2 + (B.2 - C.2)^2 = 
     (C.1 - A.1)^2 + (C.2 - A.2)^2)) →
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_equilateral_l2219_221973


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2219_221937

theorem units_digit_of_7_power_2023 : (7^2023) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2219_221937


namespace NUMINAMATH_CALUDE_shooting_range_problem_l2219_221993

theorem shooting_range_problem :
  ∀ (total_targets : ℕ) 
    (red_targets green_targets : ℕ) 
    (red_score green_score : ℚ)
    (hit_red_targets : ℕ),
  total_targets = 100 →
  total_targets = red_targets + green_targets →
  red_targets < green_targets / 3 →
  red_score = 10 →
  green_score = 8.5 →
  (green_score * green_targets + red_score * hit_red_targets : ℚ) = 
    (green_score * green_targets + red_score * red_targets : ℚ) →
  red_targets = 20 := by
sorry

end NUMINAMATH_CALUDE_shooting_range_problem_l2219_221993


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2219_221904

theorem triangle_angle_calculation (A B C a b c : Real) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  a * Real.cos B - b * Real.cos A = c →
  -- Given angle C
  C = π / 5 →
  -- Conclusion
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2219_221904


namespace NUMINAMATH_CALUDE_karls_savings_l2219_221908

/-- The problem of calculating Karl's savings --/
theorem karls_savings :
  let folder_price : ℚ := 5/2
  let pen_price : ℚ := 1
  let folder_count : ℕ := 7
  let pen_count : ℕ := 10
  let folder_discount : ℚ := 3/10
  let pen_discount : ℚ := 15/100
  
  let folder_savings := folder_count * (folder_price * folder_discount)
  let pen_savings := pen_count * (pen_price * pen_discount)
  
  folder_savings + pen_savings = 27/4 := by
  sorry

end NUMINAMATH_CALUDE_karls_savings_l2219_221908


namespace NUMINAMATH_CALUDE_special_function_is_zero_l2219_221983

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z

/-- Theorem stating that any function satisfying the special property must be the constant zero function -/
theorem special_function_is_zero (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_special_function_is_zero_l2219_221983


namespace NUMINAMATH_CALUDE_roberto_outfits_l2219_221956

def trousers : ℕ := 5
def shirts : ℕ := 6
def jackets : ℕ := 3
def ties : ℕ := 2

theorem roberto_outfits : trousers * shirts * jackets * ties = 180 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l2219_221956


namespace NUMINAMATH_CALUDE_mariela_cards_l2219_221998

/-- The total number of get well cards Mariela received -/
def total_cards (hospital_cards : ℕ) (home_cards : ℕ) : ℕ :=
  hospital_cards + home_cards

/-- Theorem stating the total number of cards Mariela received -/
theorem mariela_cards : 
  total_cards 403 287 = 690 := by
  sorry

end NUMINAMATH_CALUDE_mariela_cards_l2219_221998


namespace NUMINAMATH_CALUDE_dans_initial_green_marbles_l2219_221936

/-- Represents the number of marbles Dan has -/
structure DanMarbles where
  initial_green : ℕ
  violet : ℕ
  taken_green : ℕ
  remaining_green : ℕ

/-- Theorem stating that Dan's initial number of green marbles is 32 -/
theorem dans_initial_green_marbles 
  (dan : DanMarbles)
  (h1 : dan.taken_green = 23)
  (h2 : dan.remaining_green = 9)
  (h3 : dan.initial_green = dan.taken_green + dan.remaining_green) :
  dan.initial_green = 32 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_green_marbles_l2219_221936


namespace NUMINAMATH_CALUDE_son_age_l2219_221982

/-- Represents the age of the father when the son was born -/
def N : ℕ := sorry

/-- Represents the current age of the son -/
def k : ℕ := sorry

/-- The father's current age is no more than 75 -/
axiom father_age_bound : N + k ≤ 75

/-- The son is exactly half the age of the father -/
axiom son_half_father_age : 2 * k = N + k

/-- There are exactly 8 distinct values of k where N is divisible by k -/
axiom eight_divisors : ∃ (S : Finset ℕ), S.card = 8 ∧ ∀ x ∈ S, N % x = 0

/-- The son's age is either 24 or 30 -/
theorem son_age : k = 24 ∨ k = 30 := by sorry

end NUMINAMATH_CALUDE_son_age_l2219_221982


namespace NUMINAMATH_CALUDE_price_increase_l2219_221919

theorem price_increase (original_price : ℝ) (increase_percentage : ℝ) : 
  increase_percentage > 0 →
  (1 + increase_percentage) * (1 + increase_percentage) = 1 + 0.44 →
  increase_percentage = 0.2 := by
sorry

end NUMINAMATH_CALUDE_price_increase_l2219_221919


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2219_221965

theorem min_value_quadratic :
  (∃ (x : ℝ), x^2 + 12*x = -36) ∧ (∀ (x : ℝ), x^2 + 12*x ≥ -36) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2219_221965


namespace NUMINAMATH_CALUDE_complex_equality_condition_l2219_221915

theorem complex_equality_condition (a b c d : ℝ) : 
  let z1 : ℂ := Complex.mk a b
  let z2 : ℂ := Complex.mk c d
  (z1 = z2 → a = c) ∧ 
  ∃ a b c d : ℝ, a = c ∧ Complex.mk a b ≠ Complex.mk c d :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_condition_l2219_221915


namespace NUMINAMATH_CALUDE_certain_number_is_three_l2219_221925

theorem certain_number_is_three (x : ℝ) (n : ℝ) : 
  (4 / (1 + n / x) = 1) → (x = 1) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l2219_221925


namespace NUMINAMATH_CALUDE_bank_cash_increase_l2219_221902

/-- Represents a bank transaction --/
inductive Transaction
  | Deposit (amount : ℕ)
  | Withdrawal (amount : ℕ)

/-- Calculates the net change in cash after a series of transactions --/
def netChange (transactions : List Transaction) : ℤ :=
  transactions.foldl
    (fun acc t => match t with
      | Transaction.Deposit a => acc + a
      | Transaction.Withdrawal a => acc - a)
    0

/-- The list of transactions for the day --/
def dayTransactions : List Transaction := [
  Transaction.Withdrawal 960000,
  Transaction.Deposit 500000,
  Transaction.Withdrawal 700000,
  Transaction.Deposit 1200000,
  Transaction.Deposit 2200000,
  Transaction.Withdrawal 1025000,
  Transaction.Withdrawal 240000
]

theorem bank_cash_increase :
  netChange dayTransactions = 975000 := by
  sorry

end NUMINAMATH_CALUDE_bank_cash_increase_l2219_221902


namespace NUMINAMATH_CALUDE_maxwells_speed_l2219_221954

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwells_speed (total_distance : ℝ) (brads_speed : ℝ) (maxwell_time : ℝ) 
  (brad_delay : ℝ) (h1 : total_distance = 74) (h2 : brads_speed = 6) 
  (h3 : maxwell_time = 8) (h4 : brad_delay = 1) : 
  ∃ (maxwell_speed : ℝ), maxwell_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_maxwells_speed_l2219_221954


namespace NUMINAMATH_CALUDE_route_choice_and_expected_value_l2219_221923

-- Define the data types
structure RouteData where
  good : ℕ
  average : ℕ

structure GenderRouteData where
  male : ℕ
  female : ℕ

-- Define the constants
def total_tourists : ℕ := 300
def route_a : RouteData := { good := 50, average := 75 }
def route_b : RouteData := { good := 75, average := 100 }
def gender_data : GenderRouteData := { male := 120, female := 180 }

-- Define the K^2 formula
def k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for K^2 at 0.001 significance level
def k_critical : ℚ := 10.828

-- Define the expected value calculation
def expected_value (good_prob : ℚ) : ℚ :=
  let good_score := 5
  let avg_score := 2
  (1 - good_prob)^3 * (3 * avg_score) +
  3 * good_prob * (1 - good_prob)^2 * (2 * avg_score + good_score) +
  3 * good_prob^2 * (1 - good_prob) * (avg_score + 2 * good_score) +
  good_prob^3 * (3 * good_score)

-- Theorem statement
theorem route_choice_and_expected_value :
  let k_value := k_squared gender_data.male (gender_data.female - gender_data.male)
                            (total_tourists - gender_data.male - gender_data.female) gender_data.female
  let prob_a := (route_a.good : ℚ) / (route_a.good + route_a.average)
  let prob_b := (route_b.good : ℚ) / (route_b.good + route_b.average)
  k_value > k_critical ∧ expected_value prob_a > expected_value prob_b := by
  sorry

end NUMINAMATH_CALUDE_route_choice_and_expected_value_l2219_221923


namespace NUMINAMATH_CALUDE_tan_sum_quarter_pi_l2219_221922

theorem tan_sum_quarter_pi (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  Real.tan (α + π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_quarter_pi_l2219_221922


namespace NUMINAMATH_CALUDE_square_circle_union_area_l2219_221903

/-- The area of the union of a square and a circle, where the square has side length 8 and the circle has radius 12 and is centered at one of the square's vertices. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 64 + 108 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l2219_221903


namespace NUMINAMATH_CALUDE_mikas_height_mikas_height_is_70_l2219_221979

/-- Proves that Mika's current height is 70 inches given the problem conditions -/
theorem mikas_height (original_height : ℝ) (sheas_growth_rate : ℝ) (mikas_growth_ratio : ℝ) 
  (sheas_current_height : ℝ) : ℝ :=
  let sheas_growth := sheas_current_height - original_height
  let mikas_growth := mikas_growth_ratio * sheas_growth
  original_height + mikas_growth
where
  -- Shea and Mika were originally the same height
  original_height_positive : 0 < original_height := by sorry
  -- Shea has grown by 25%
  sheas_growth_rate_def : sheas_growth_rate = 0.25 := by sorry
  -- Mika has grown two-thirds as many inches as Shea
  mikas_growth_ratio_def : mikas_growth_ratio = 2/3 := by sorry
  -- Shea is now 75 inches tall
  sheas_current_height_def : sheas_current_height = 75 := by sorry
  -- Shea's current height is 25% more than the original height
  sheas_growth_equation : sheas_current_height = original_height * (1 + sheas_growth_rate) := by sorry

theorem mikas_height_is_70 : mikas_height 60 0.25 (2/3) 75 = 70 := by sorry

end NUMINAMATH_CALUDE_mikas_height_mikas_height_is_70_l2219_221979


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l2219_221934

theorem gcd_powers_of_two : Nat.gcd (2^2024 - 1) (2^2007 - 1) = 2^17 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l2219_221934


namespace NUMINAMATH_CALUDE_cos_570_deg_l2219_221962

theorem cos_570_deg : Real.cos (570 * π / 180) = - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_570_deg_l2219_221962


namespace NUMINAMATH_CALUDE_min_faces_two_dice_l2219_221966

theorem min_faces_two_dice (a b : ℕ) : 
  a ≥ 8 → b ≥ 8 →  -- Both dice have at least 8 faces
  (∀ i j, 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b) →  -- Each face has a distinct integer from 1 to the number of faces
  (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 9} : ℚ) / (a * b : ℚ) = 
    (2/3) * ((Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 11} : ℚ) / (a * b : ℚ)) →  -- Probability condition for sum of 9 and 11
  (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ a ∧ 1 ≤ j ∧ j ≤ b ∧ i + j = 14} : ℚ) / (a * b : ℚ) = 1/9 →  -- Probability condition for sum of 14
  a + b ≥ 22 ∧ ∀ c d, c ≥ 8 → d ≥ 8 → 
    (∀ i j, 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d) →
    (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 9} : ℚ) / (c * d : ℚ) = 
      (2/3) * ((Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 11} : ℚ) / (c * d : ℚ)) →
    (Nat.card {(i, j) | 1 ≤ i ∧ i ≤ c ∧ 1 ≤ j ∧ j ≤ d ∧ i + j = 14} : ℚ) / (c * d : ℚ) = 1/9 →
    c + d ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_min_faces_two_dice_l2219_221966


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2219_221963

theorem min_fraction_sum (c d : ℕ) (hc : c > 0) (hd : d > 0) (hcd : c > d) 
  (hodd : Odd (c + d)) :
  (c + d : ℚ) / (c - d) + (c - d : ℚ) / (c + d) ≥ 10 / 3 ∧
  ∃ (c' d' : ℕ), c' > 0 ∧ d' > 0 ∧ c' > d' ∧ Odd (c' + d') ∧
    (c' + d' : ℚ) / (c' - d') + (c' - d' : ℚ) / (c' + d') = 10 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2219_221963


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2219_221988

theorem arithmetic_sequence_sum (a₁ aₙ : ℤ) (n : ℕ) (h : n > 0) :
  let S := n * (a₁ + aₙ) / 2
  a₁ = -3 ∧ aₙ = 48 ∧ n = 12 → S = 270 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2219_221988


namespace NUMINAMATH_CALUDE_complement_intersection_equals_five_l2219_221959

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_intersection_equals_five :
  (U \ M) ∩ (U \ N) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_five_l2219_221959


namespace NUMINAMATH_CALUDE_sector_triangle_area_equality_l2219_221948

/-- Given a circle with center C and radius r, and an angle φ where 0 < φ < π/2,
    prove that the area of the circular sector formed by φ is equal to 
    the area of the triangle formed by the tangent line and the radius 
    if and only if tan φ = φ. -/
theorem sector_triangle_area_equality (φ : Real) (h1 : 0 < φ) (h2 : φ < π/2) :
  let r : Real := 1  -- Assuming unit circle for simplicity
  let sector_area : Real := (φ * r^2) / 2
  let triangle_area : Real := (r^2 * Real.tan φ) / 2
  sector_area = triangle_area ↔ Real.tan φ = φ := by
  sorry

end NUMINAMATH_CALUDE_sector_triangle_area_equality_l2219_221948


namespace NUMINAMATH_CALUDE_thirty_percent_of_eighty_l2219_221924

theorem thirty_percent_of_eighty : ∃ x : ℝ, (30 / 100) * x = 24 ∧ x = 80 := by sorry

end NUMINAMATH_CALUDE_thirty_percent_of_eighty_l2219_221924


namespace NUMINAMATH_CALUDE_test_composition_l2219_221905

theorem test_composition (total_points total_questions : ℕ) 
  (h1 : total_points = 100) 
  (h2 : total_questions = 40) : 
  ∃ (two_point_questions four_point_questions : ℕ),
    two_point_questions + four_point_questions = total_questions ∧
    2 * two_point_questions + 4 * four_point_questions = total_points ∧
    two_point_questions = 30 := by
  sorry

end NUMINAMATH_CALUDE_test_composition_l2219_221905


namespace NUMINAMATH_CALUDE_special_polygon_perimeter_l2219_221911

/-- A polygon with specific properties -/
structure SpecialPolygon where
  AB : ℝ
  AE : ℝ
  BD : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  angle_DBC : ℝ
  angle_BCD : ℝ
  angle_CDB : ℝ
  h_AB_eq_AE : AB = AE
  h_AB_val : AB = 120
  h_DE_val : DE = 226
  h_BD_val : BD = 115
  h_BD_eq_BC : BD = BC
  h_angle_DBC_eq_BCD : angle_DBC = angle_BCD
  h_triangle_BCD_equilateral : angle_DBC = 60 ∧ angle_BCD = 60 ∧ angle_CDB = 60
  h_CD_eq_BD : CD = BD

/-- The perimeter of the special polygon is 696 -/
theorem special_polygon_perimeter (p : SpecialPolygon) : 
  p.AB + p.AE + p.BD + p.BC + p.CD + p.DE = 696 := by
  sorry


end NUMINAMATH_CALUDE_special_polygon_perimeter_l2219_221911


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2219_221953

theorem sqrt_sum_fractions : Real.sqrt (1/8 + 1/25) = Real.sqrt 33 / (10 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2219_221953


namespace NUMINAMATH_CALUDE_farmer_randy_planting_rate_l2219_221940

/-- Represents the cotton planting problem for Farmer Randy -/
structure CottonPlanting where
  total_acres : ℕ
  total_days : ℕ
  first_crew_tractors : ℕ
  first_crew_days : ℕ
  second_crew_tractors : ℕ
  second_crew_days : ℕ

/-- Calculates the acres per tractor per day needed to meet the planting deadline -/
def acres_per_tractor_per_day (cp : CottonPlanting) : ℚ :=
  cp.total_acres / (cp.first_crew_tractors * cp.first_crew_days + cp.second_crew_tractors * cp.second_crew_days)

/-- Theorem stating that for Farmer Randy's specific situation, each tractor needs to plant 68 acres per day -/
theorem farmer_randy_planting_rate :
  let cp : CottonPlanting := {
    total_acres := 1700,
    total_days := 5,
    first_crew_tractors := 2,
    first_crew_days := 2,
    second_crew_tractors := 7,
    second_crew_days := 3
  }
  acres_per_tractor_per_day cp = 68 := by
  sorry

end NUMINAMATH_CALUDE_farmer_randy_planting_rate_l2219_221940


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l2219_221944

theorem opposite_sides_line_range (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l2219_221944


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2219_221986

def geometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometricSequence a → a 5 = 2 → a 1 * a 2 * a 3 * a 7 * a 8 * a 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2219_221986


namespace NUMINAMATH_CALUDE_caroline_lassis_l2219_221920

/-- Given that Caroline can make 11 lassis with 2 mangoes, 
    prove that she can make 55 lassis with 10 mangoes. -/
theorem caroline_lassis (lassis_per_two_mangoes : ℕ) (mangoes : ℕ) :
  lassis_per_two_mangoes = 11 ∧ mangoes = 10 →
  (lassis_per_two_mangoes : ℚ) / 2 * mangoes = 55 := by
  sorry

end NUMINAMATH_CALUDE_caroline_lassis_l2219_221920


namespace NUMINAMATH_CALUDE_alpha_value_l2219_221958

theorem alpha_value (α β γ : Real) 
  (h1 : 0 < α ∧ α < π)
  (h2 : α + β + γ = π)
  (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
  α = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2219_221958


namespace NUMINAMATH_CALUDE_consecutive_product_sum_l2219_221961

theorem consecutive_product_sum : ∃ (a b c d e : ℤ),
  (b = a + 1) ∧
  (d = c + 1) ∧
  (e = d + 1) ∧
  (a * b = 990) ∧
  (c * d * e = 990) ∧
  (a + b + c + d + e = 90) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_sum_l2219_221961


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l2219_221997

theorem reciprocal_of_2023 : 
  (∀ x : ℝ, x ≠ 0 → (1 / x) = x⁻¹) → 2023⁻¹ = (1 : ℝ) / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l2219_221997


namespace NUMINAMATH_CALUDE_division_and_addition_l2219_221960

theorem division_and_addition : (12 / (1/4)) + 5 = 53 := by sorry

end NUMINAMATH_CALUDE_division_and_addition_l2219_221960


namespace NUMINAMATH_CALUDE_cheryl_mm_theorem_l2219_221992

def cheryl_mm_problem (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) (remaining : ℕ) : Prop :=
  initial - eaten_lunch - eaten_dinner - remaining = 18

theorem cheryl_mm_theorem :
  cheryl_mm_problem 40 7 5 10 := by sorry

end NUMINAMATH_CALUDE_cheryl_mm_theorem_l2219_221992


namespace NUMINAMATH_CALUDE_nine_times_reverse_unique_l2219_221929

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem nine_times_reverse_unique : 
  ∀ n : ℕ, is_four_digit n → (n = 9 * reverse_number n) → n = 9801 :=
by sorry

end NUMINAMATH_CALUDE_nine_times_reverse_unique_l2219_221929


namespace NUMINAMATH_CALUDE_taco_truck_profit_l2219_221951

/-- Calculates the profit for a taco truck given the specified conditions -/
theorem taco_truck_profit
  (total_beef : ℝ)
  (beef_per_taco : ℝ)
  (selling_price : ℝ)
  (cost_per_taco : ℝ)
  (h1 : total_beef = 100)
  (h2 : beef_per_taco = 0.25)
  (h3 : selling_price = 2)
  (h4 : cost_per_taco = 1.5) :
  (total_beef / beef_per_taco) * (selling_price - cost_per_taco) = 200 :=
by sorry

end NUMINAMATH_CALUDE_taco_truck_profit_l2219_221951


namespace NUMINAMATH_CALUDE_fraction_simplification_l2219_221977

theorem fraction_simplification : (3/8 + 5/6) / (5/12 + 1/4) = 29/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2219_221977


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2219_221976

theorem ceiling_floor_difference : 
  ⌈(12 : ℚ) / 7 * (-29 : ℚ) / 3⌉ - ⌊(12 : ℚ) / 7 * ⌊(-29 : ℚ) / 3⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2219_221976


namespace NUMINAMATH_CALUDE_iron_bar_height_l2219_221990

/-- Proves that the height of an iron bar is 6 cm given specific conditions --/
theorem iron_bar_height : 
  ∀ (length width height : ℝ) (num_bars num_balls ball_volume : ℕ),
  length = 12 →
  width = 8 →
  num_bars = 10 →
  num_balls = 720 →
  ball_volume = 8 →
  (num_bars : ℝ) * length * width * height = (num_balls : ℝ) * (ball_volume : ℝ) →
  height = 6 := by
sorry

end NUMINAMATH_CALUDE_iron_bar_height_l2219_221990


namespace NUMINAMATH_CALUDE_student_take_home_pay_l2219_221971

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takeHomePay (baseSalary bonus taxRate : ℚ) : ℚ :=
  let totalEarnings := baseSalary + bonus
  let taxAmount := totalEarnings * taxRate
  totalEarnings - taxAmount

/-- Theorem stating that the take-home pay for a well-performing student is 26100 rubles --/
theorem student_take_home_pay :
  takeHomePay 25000 5000 (13/100) = 26100 := by
  sorry

#eval takeHomePay 25000 5000 (13/100)

end NUMINAMATH_CALUDE_student_take_home_pay_l2219_221971


namespace NUMINAMATH_CALUDE_office_officers_count_l2219_221989

/-- Represents the salary and employee data for an office --/
structure OfficeSalaryData where
  avgSalaryAll : ℚ
  avgSalaryOfficers : ℚ
  avgSalaryNonOfficers : ℚ
  numNonOfficers : ℕ

/-- Calculates the number of officers given the office salary data --/
def calculateOfficers (data : OfficeSalaryData) : ℕ :=
  sorry

/-- Theorem stating that the number of officers is 15 given the specific salary data --/
theorem office_officers_count (data : OfficeSalaryData) 
  (h1 : data.avgSalaryAll = 120)
  (h2 : data.avgSalaryOfficers = 450)
  (h3 : data.avgSalaryNonOfficers = 110)
  (h4 : data.numNonOfficers = 495) :
  calculateOfficers data = 15 := by
  sorry

end NUMINAMATH_CALUDE_office_officers_count_l2219_221989


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2219_221984

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 9) : 
  let s := d / Real.sqrt 3
  s^3 = 81 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2219_221984


namespace NUMINAMATH_CALUDE_sons_age_l2219_221933

theorem sons_age (son_age man_age : ℕ) : 
  man_age = 3 * son_age →
  man_age + 12 = 2 * (son_age + 12) →
  son_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l2219_221933


namespace NUMINAMATH_CALUDE_marbles_given_to_mary_l2219_221928

def initial_marbles : ℕ := 64
def remaining_marbles : ℕ := 50

theorem marbles_given_to_mary :
  initial_marbles - remaining_marbles = 14 :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_to_mary_l2219_221928


namespace NUMINAMATH_CALUDE_right_triangle_product_divisible_by_60_l2219_221926

theorem right_triangle_product_divisible_by_60 
  (a b c : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  60 ∣ (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_product_divisible_by_60_l2219_221926


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l2219_221949

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l2219_221949


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_7_l2219_221945

/-- A bag containing cards numbered from 0 to 5 -/
def Bag : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- The sample space of drawing one card from each bag -/
def SampleSpace : Finset (ℕ × ℕ) := Bag.product Bag

/-- The event where the sum of two drawn cards is greater than 7 -/
def EventSumGreaterThan7 : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun p => p.1 + p.2 > 7)

/-- The probability of the event -/
def ProbabilityEventSumGreaterThan7 : ℚ :=
  EventSumGreaterThan7.card / SampleSpace.card

theorem probability_sum_greater_than_7 :
  ProbabilityEventSumGreaterThan7 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_7_l2219_221945
