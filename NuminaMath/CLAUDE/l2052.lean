import Mathlib

namespace fraction_inequality_l2052_205277

theorem fraction_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end fraction_inequality_l2052_205277


namespace total_damage_cost_l2052_205254

/-- The cost of damages caused by Jack --/
def cost_of_damages (tire_cost : ℕ) (num_tires : ℕ) (window_cost : ℕ) : ℕ :=
  tire_cost * num_tires + window_cost

/-- Theorem stating the total cost of damages --/
theorem total_damage_cost :
  cost_of_damages 250 3 700 = 1450 := by
  sorry

end total_damage_cost_l2052_205254


namespace solve_2a_plus_b_l2052_205201

theorem solve_2a_plus_b (a b : ℝ) 
  (h1 : 4 * a^2 - b^2 = 12) 
  (h2 : 2 * a - b = 4) : 
  2 * a + b = 3 := by
sorry

end solve_2a_plus_b_l2052_205201


namespace special_function_range_l2052_205252

open Set Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  condition1 : ∀ x, f (-x) / f x = exp (2 * x)
  condition2 : ∀ x, x < 0 → f x + deriv f x > 0

/-- The theorem statement -/
theorem special_function_range (sf : SpecialFunction) :
  {a : ℝ | exp a * sf.f (2 * a + 1) ≥ sf.f (a + 1)} = Icc (-2/3) 0 :=
sorry

end special_function_range_l2052_205252


namespace bird_migration_difference_l2052_205295

/-- The number of bird families that flew away for the winter -/
def flew_away : ℕ := 86

/-- The number of bird families initially living near the mountain -/
def initial_families : ℕ := 45

/-- The difference between the number of bird families that flew away and those that stayed behind -/
def difference : ℕ := flew_away - initial_families

theorem bird_migration_difference :
  difference = 41 :=
sorry

end bird_migration_difference_l2052_205295


namespace sum_is_zero_l2052_205251

def circular_sequence (n : ℕ) := Fin n → ℤ

def neighbor_sum_property (s : circular_sequence 14) : Prop :=
  ∀ i : Fin 14, s i = s (i - 1) + s (i + 1)

theorem sum_is_zero (s : circular_sequence 14) 
  (h : neighbor_sum_property s) : 
  (Finset.univ.sum s) = 0 := by
  sorry

end sum_is_zero_l2052_205251


namespace product_198_202_l2052_205212

theorem product_198_202 : 198 * 202 = 39996 := by
  sorry

end product_198_202_l2052_205212


namespace percent_calculation_l2052_205232

theorem percent_calculation (x : ℝ) (h : 0.20 * x = 200) : 1.20 * x = 1200 := by
  sorry

end percent_calculation_l2052_205232


namespace min_odd_sided_polygon_divisible_into_parallelograms_l2052_205246

/-- A polygon is a closed shape with straight sides. -/
structure Polygon where
  sides : ℕ
  is_closed : Bool

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  is_quadrilateral : Bool
  opposite_sides_parallel : Bool

/-- A function that checks if a polygon can be divided into parallelograms. -/
def can_be_divided_into_parallelograms (p : Polygon) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ ∃ (parallelograms : Fin n → Parallelogram), True

/-- Theorem stating the minimum number of sides for an odd-sided polygon
    that can be divided into parallelograms is 7. -/
theorem min_odd_sided_polygon_divisible_into_parallelograms :
  ∀ (p : Polygon),
    p.sides % 2 = 1 →
    can_be_divided_into_parallelograms p →
    p.sides ≥ 7 ∧
    ∃ (q : Polygon), q.sides = 7 ∧ can_be_divided_into_parallelograms q :=
sorry

end min_odd_sided_polygon_divisible_into_parallelograms_l2052_205246


namespace double_average_l2052_205204

theorem double_average (n : Nat) (original_avg : Nat) (h1 : n = 12) (h2 : original_avg = 36) :
  let total := n * original_avg
  let doubled_total := 2 * total
  let new_avg := doubled_total / n
  new_avg = 72 := by
sorry

end double_average_l2052_205204


namespace sqrt_equation_solution_l2052_205237

theorem sqrt_equation_solution :
  ∀ z : ℝ, (Real.sqrt (3 + z) = 12) ↔ (z = 141) :=
by sorry

end sqrt_equation_solution_l2052_205237


namespace intersection_equality_l2052_205292

theorem intersection_equality (m : ℝ) : 
  let A : Set ℝ := {2, 5, m^2 - m}
  let B : Set ℝ := {2, m + 3}
  A ∩ B = B → m = 3 := by
sorry

end intersection_equality_l2052_205292


namespace down_payment_calculation_l2052_205263

theorem down_payment_calculation (purchase_price : ℝ) 
  (monthly_payment : ℝ) (num_payments : ℕ) (interest_rate : ℝ) :
  purchase_price = 118 →
  monthly_payment = 10 →
  num_payments = 12 →
  interest_rate = 0.15254237288135593 →
  ∃ (down_payment : ℝ),
    down_payment + (monthly_payment * num_payments) = 
      purchase_price * (1 + interest_rate) ∧
    down_payment = 16 :=
by sorry

end down_payment_calculation_l2052_205263


namespace complex_expression_equality_l2052_205238

theorem complex_expression_equality (y : ℂ) (h : y = Complex.exp (2 * π * I / 9)) :
  (3 * y + y^3) * (3 * y^3 + y^9) * (3 * y^6 + y^18) = 121 + 48 * (y + y^6) := by
  sorry

end complex_expression_equality_l2052_205238


namespace line_l_equation_l2052_205255

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the given conditions
def point_on_l : Point := (2, 3)
def L1 : Line := λ x y => 2*x - 5*y + 9
def L2 : Line := λ x y => 2*x - 5*y - 7
def midpoint_line : Line := λ x y => x - 4*y - 1

-- Define the line l
def l : Line := λ x y => 4*x - 5*y + 7

-- Theorem statement
theorem line_l_equation : 
  ∃ (A B : Point),
    (L1 A.1 A.2 = 0 ∧ L2 B.1 B.2 = 0) ∧ 
    (midpoint_line ((A.1 + B.1)/2) ((A.2 + B.2)/2) = 0) ∧
    (l point_on_l.1 point_on_l.2 = 0) ∧
    (∀ (x y : ℝ), l x y = 0 ↔ 4*x - 5*y + 7 = 0) :=
by sorry

end line_l_equation_l2052_205255


namespace furniture_shop_cost_price_l2052_205227

/-- Proves that the cost price of an item is 6672 when the selling price is 8340
    and the markup is 25%. -/
theorem furniture_shop_cost_price : 
  ∀ (cost_price selling_price : ℝ),
  selling_price = 8340 →
  selling_price = cost_price * (1 + 0.25) →
  cost_price = 6672 := by sorry

end furniture_shop_cost_price_l2052_205227


namespace intersection_point_is_solution_l2052_205296

-- Define the two lines
def line1 (x y : ℚ) : Prop := 2 * x - 3 * y = 3
def line2 (x y : ℚ) : Prop := 4 * x + 2 * y = 2

-- Define the intersection point
def intersection_point : ℚ × ℚ := (3/4, -1/2)

-- Theorem statement
theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → (x', y') = (x, y) := by
  sorry

end intersection_point_is_solution_l2052_205296


namespace truncated_cone_radius_l2052_205236

/-- Represents a cone with its base radius -/
structure Cone :=
  (baseRadius : ℝ)

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone :=
  (smallerBaseRadius : ℝ)

/-- Checks if three cones are touching each other -/
def areTouching (c1 c2 c3 : Cone) : Prop :=
  -- This is a simplification. In reality, we'd need to check the geometric conditions.
  true

/-- Checks if a truncated cone has a common generatrix with other cones -/
def hasCommonGeneratrix (tc : TruncatedCone) (c1 c2 c3 : Cone) : Prop :=
  -- This is a simplification. In reality, we'd need to check the geometric conditions.
  true

/-- The main theorem -/
theorem truncated_cone_radius 
  (c1 c2 c3 : Cone) 
  (tc : TruncatedCone) 
  (h1 : c1.baseRadius = 6) 
  (h2 : c2.baseRadius = 24) 
  (h3 : c3.baseRadius = 24) 
  (h4 : areTouching c1 c2 c3) 
  (h5 : hasCommonGeneratrix tc c1 c2 c3) : 
  tc.smallerBaseRadius = 2 := by
  sorry

end truncated_cone_radius_l2052_205236


namespace triangle_isosceles_if_c_eq_2a_cos_B_l2052_205279

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- State the theorem
theorem triangle_isosceles_if_c_eq_2a_cos_B (t : Triangle) 
  (h : t.c = 2 * t.a * Real.cos t.B) : isIsosceles t :=
sorry

end triangle_isosceles_if_c_eq_2a_cos_B_l2052_205279


namespace largest_even_number_under_300_l2052_205247

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 2 = 0 ∧ n ≤ 300

theorem largest_even_number_under_300 :
  ∀ n : ℕ, is_valid_number n → n ≤ 298 :=
by
  sorry

#check largest_even_number_under_300

end largest_even_number_under_300_l2052_205247


namespace fraction_simplification_l2052_205221

theorem fraction_simplification (x : ℝ) (h : 2 * x - 3 ≠ 0) :
  (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) / (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) = (2 * x + 3) / (2 * x - 3) := by
  sorry

end fraction_simplification_l2052_205221


namespace field_ratio_l2052_205249

/-- Proves that a rectangular field with perimeter 336 meters and width 70 meters has a length-to-width ratio of 7:5 -/
theorem field_ratio (perimeter width : ℝ) (h1 : perimeter = 336) (h2 : width = 70) :
  (perimeter / 2 - width) / width = 7 / 5 := by
  sorry

end field_ratio_l2052_205249


namespace polynomial_factorization_l2052_205244

theorem polynomial_factorization (x : ℝ) : 4*x^3 - 4*x^2 + x = x*(2*x - 1)^2 := by
  sorry

end polynomial_factorization_l2052_205244


namespace quadratic_inequality_properties_l2052_205272

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c m n : ℝ) : Prop :=
  ∀ x, f a b c x > 0 ↔ m < x ∧ x < n

-- State the theorem
theorem quadratic_inequality_properties
  (a b c m n : ℝ)
  (h_sol : solution_set a b c m n)
  (h_m_pos : m > 0)
  (h_n_gt_m : n > m) :
  a < 0 ∧
  b > 0 ∧
  (∀ x, f c b a x > 0 ↔ 1/n < x ∧ x < 1/m) :=
sorry

end quadratic_inequality_properties_l2052_205272


namespace tangent_line_equation_l2052_205205

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem tangent_line_equation (P : ℝ × ℝ) (h₁ : P = (-2, -2)) :
  ∃ (m b : ℝ), (∀ x, (m * x + b = 9 * x + 16) ∨ (m * x + b = -2)) ∧
  (∃ x₀, f x₀ = m * x₀ + b ∧ 
         ∀ x, f x ≥ m * x + b ∧ 
         (f x = m * x + b ↔ x = x₀)) ∧
  (m * P.1 + b = P.2) := by
sorry

end tangent_line_equation_l2052_205205


namespace sphere_roll_coplanar_l2052_205270

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a rectangular box -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the transformation of a point on a sphere's surface after rolling -/
def sphereRoll (s : Sphere) (b : RectangularBox) (p : Point3D) : Point3D :=
  sorry

/-- States that four points lie in the same plane -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem sphere_roll_coplanar (s : Sphere) (b : RectangularBox) (X : Point3D) :
  let X₁ := sphereRoll s b X
  let X₂ := sphereRoll s b X₁
  let X₃ := sphereRoll s b X₂
  coplanar X X₁ X₂ X₃ :=
sorry

end sphere_roll_coplanar_l2052_205270


namespace m_range_l2052_205216

theorem m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m * x + 1 > 0) → 
  -2 < m ∧ m < 0 := by sorry

end m_range_l2052_205216


namespace point_outside_circle_l2052_205283

theorem point_outside_circle 
  (a b : ℝ) 
  (line_intersects_circle : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    a * x₁ + b * y₁ = 1 ∧ a * x₂ + b * y₂ = 1 ∧
    x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1) :
  a^2 + b^2 > 1 := by
  sorry

end point_outside_circle_l2052_205283


namespace acid_solution_dilution_l2052_205290

theorem acid_solution_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 15) / 100 * (m + x)) → x = 15 * m / (m - 15) := by
  sorry

end acid_solution_dilution_l2052_205290


namespace largest_smallest_divisible_by_165_l2052_205262

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧  -- 7-digit number
  (n % 165 = 0) ∧  -- divisible by 165
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6] →
    (∃! i : ℕ, i < 7 ∧ (n / 10^i) % 10 = d)  -- each digit appears exactly once

theorem largest_smallest_divisible_by_165 :
  (∀ n : ℕ, is_valid_number n → n ≤ 6431205) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ 1042635) ∧
  is_valid_number 6431205 ∧
  is_valid_number 1042635 :=
sorry

end largest_smallest_divisible_by_165_l2052_205262


namespace young_photographer_club_l2052_205284

theorem young_photographer_club (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size :=
by sorry


end young_photographer_club_l2052_205284


namespace cycle_reappearance_l2052_205269

theorem cycle_reappearance (letter_cycle_length digit_cycle_length : ℕ) 
  (h1 : letter_cycle_length = 7)
  (h2 : digit_cycle_length = 4) :
  Nat.lcm letter_cycle_length digit_cycle_length = 28 := by
  sorry

end cycle_reappearance_l2052_205269


namespace rational_number_statements_l2052_205224

theorem rational_number_statements (a b : ℚ) : 
  (∃! n : ℕ, n = 2 ∧ 
    (((a + b > 0 ∧ (a > 0 ↔ b > 0)) → (a > 0 ∧ b > 0)) = true) ∧
    ((a + b < 0 → ¬(a > 0 ↔ b > 0)) = false) ∧
    (((abs a > abs b ∧ ¬(a > 0 ↔ b > 0)) → a + b > 0) = false) ∧
    ((abs a < b → a + b > 0) = true)) :=
sorry

end rational_number_statements_l2052_205224


namespace at_least_two_consecutive_successes_l2052_205256

def probability_success : ℚ := 2 / 5

def probability_failure : ℚ := 1 - probability_success

def number_of_attempts : ℕ := 4

theorem at_least_two_consecutive_successes :
  let p_success := probability_success
  let p_failure := probability_failure
  let n := number_of_attempts
  (1 : ℚ) - (p_failure^n + n * p_success * p_failure^(n-1) + 3 * p_success^2 * p_failure^2) = 44 / 125 := by
  sorry

end at_least_two_consecutive_successes_l2052_205256


namespace total_fruits_bought_l2052_205259

/-- The total number of fruits bought given the cost and quantity constraints -/
theorem total_fruits_bought
  (total_cost : ℕ)
  (plum_cost peach_cost : ℕ)
  (plum_quantity : ℕ)
  (h1 : total_cost = 52)
  (h2 : plum_cost = 2)
  (h3 : peach_cost = 1)
  (h4 : plum_quantity = 20)
  (h5 : plum_cost * plum_quantity + peach_cost * (total_cost - plum_cost * plum_quantity) = total_cost) :
  plum_quantity + (total_cost - plum_cost * plum_quantity) = 32 := by
  sorry

end total_fruits_bought_l2052_205259


namespace final_price_is_20_70_l2052_205299

/-- The price of one kilogram of cucumbers in dollars -/
def cucumber_price : ℝ := 5

/-- The price of one kilogram of tomatoes in dollars -/
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

/-- The number of kilograms of tomatoes bought -/
def tomato_kg : ℝ := 2

/-- The number of kilograms of cucumbers bought -/
def cucumber_kg : ℝ := 3

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.1

/-- The final price paid for the items after discount -/
def final_price : ℝ := (tomato_price * tomato_kg + cucumber_price * cucumber_kg) * (1 - discount_rate)

theorem final_price_is_20_70 : final_price = 20.70 := by
  sorry

end final_price_is_20_70_l2052_205299


namespace proposition_false_range_l2052_205211

open Set

theorem proposition_false_range (a : ℝ) : 
  (¬∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ Iio (-3) ∪ Ioi 1 :=
sorry

end proposition_false_range_l2052_205211


namespace five_balls_four_boxes_l2052_205214

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to place 5 distinguishable balls into 4 distinguishable boxes is 4^5 -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by
  sorry

end five_balls_four_boxes_l2052_205214


namespace problem_1_problem_2_problem_3_l2052_205202

-- Problem 1
theorem problem_1 : 
  |(-3)| + (-1)^2021 * (Real.pi - 3.14)^0 - (-1/2)⁻¹ = 4 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : 
  (x + 3)^2 - (x + 2) * (x - 2) = 6 * x + 13 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) : 
  (2*x - y + 3) * (2*x + y - 3) = 4*x^2 - y^2 + 6*y - 9 := by sorry

end problem_1_problem_2_problem_3_l2052_205202


namespace cab_ride_cost_total_cost_is_6720_l2052_205242

/-- Calculate the total cost of cab rides for a one-week event with carpooling --/
theorem cab_ride_cost (off_peak_rate : ℚ) (peak_rate : ℚ) (distance : ℚ) 
  (days : ℕ) (participants : ℕ) (discount : ℚ) : ℚ :=
  let daily_cost := off_peak_rate * distance + peak_rate * distance
  let total_cost := daily_cost * days
  let discounted_cost := total_cost * (1 - discount)
  discounted_cost

/-- Prove that the total cost for all participants is $6720 --/
theorem total_cost_is_6720 : 
  cab_ride_cost (5/2) (7/2) 200 7 4 (1/5) = 6720 := by
  sorry

end cab_ride_cost_total_cost_is_6720_l2052_205242


namespace some_number_value_l2052_205291

theorem some_number_value : ∃ (x : ℚ), 
  (1 / 2 : ℚ) + ((2 / 3 : ℚ) * (3 / 8 : ℚ) + x) - (8 / 16 : ℚ) = (17 / 4 : ℚ) ∧ x = 4 := by
  sorry

end some_number_value_l2052_205291


namespace sin_bounds_l2052_205258

theorem sin_bounds :
  (∀ x : ℝ, -5 ≤ 2 * Real.sin x - 3 ∧ 2 * Real.sin x - 3 ≤ -1) ∧
  (∃ x y : ℝ, 2 * Real.sin x - 3 = -5 ∧ 2 * Real.sin y - 3 = -1) := by sorry

end sin_bounds_l2052_205258


namespace last_digit_of_one_over_three_to_fifteen_l2052_205200

/-- The last digit of the decimal expansion of 1/3^15 is 0 -/
theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (∃ (k : ℕ), (1 : ℚ) / 3^n = k * (1 / 10^n) + (1 / 10^n)) :=
by sorry

end last_digit_of_one_over_three_to_fifteen_l2052_205200


namespace common_tangent_length_l2052_205220

/-- The length of the common tangent of two externally tangent circles -/
theorem common_tangent_length (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  let d := R + r  -- distance between centers
  2 * Real.sqrt (r * R) = Real.sqrt (d^2 - (R - r)^2) :=
by sorry

end common_tangent_length_l2052_205220


namespace existence_of_n_l2052_205266

theorem existence_of_n (p a k : ℕ) (h_prime : Nat.Prime p) (h_pos_a : a > 0) (h_pos_k : k > 0)
  (h_lower : p^a < k) (h_upper : k < 2*p^a) :
  ∃ n : ℕ, n < p^(2*a) ∧ (Nat.choose n k : ZMod (p^a)) = n ∧ (n : ZMod (p^a)) = k := by
  sorry

end existence_of_n_l2052_205266


namespace tangent_addition_formula_l2052_205203

theorem tangent_addition_formula : 
  (Real.tan (12 * π / 180) + Real.tan (18 * π / 180)) / 
  (1 - Real.tan (12 * π / 180) * Real.tan (18 * π / 180)) = Real.sqrt 3 / 3 :=
by sorry

end tangent_addition_formula_l2052_205203


namespace tank_capacity_l2052_205225

/-- Proves that a tank's full capacity is 270/7 gallons, given initial and final fill levels -/
theorem tank_capacity (initial_fill : Rat) (final_fill : Rat) (used_gallons : Rat) 
  (h1 : initial_fill = 4/5)
  (h2 : final_fill = 1/3)
  (h3 : used_gallons = 18)
  (h4 : initial_fill * full_capacity - final_fill * full_capacity = used_gallons) :
  full_capacity = 270/7 :=
by
  sorry

#check tank_capacity

end tank_capacity_l2052_205225


namespace gum_cost_proof_l2052_205219

/-- The cost of gum in dollars -/
def cost_in_dollars (pieces : ℕ) (cents_per_piece : ℕ) : ℚ :=
  (pieces * cents_per_piece : ℚ) / 100

/-- Proof that 500 pieces of gum at 2 cents each costs 10 dollars -/
theorem gum_cost_proof : cost_in_dollars 500 2 = 10 := by
  sorry

end gum_cost_proof_l2052_205219


namespace goose_egg_count_l2052_205222

/-- The number of goose eggs laid at a certain pond -/
def total_eggs : ℕ := 1000

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 1/4

/-- The fraction of hatched geese that survived the first month -/
def first_month_survival_rate : ℚ := 4/5

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_mortality_rate : ℚ := 2/5

/-- The number of geese that survived the first year -/
def survivors : ℕ := 120

theorem goose_egg_count :
  total_eggs * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = survivors := by
  sorry

end goose_egg_count_l2052_205222


namespace divisibility_condition_l2052_205264

theorem divisibility_condition (x y : ℕ+) :
  (x * y^2 + 2 * y) ∣ (2 * x^2 * y + x * y^2 + 8 * x) ↔
  (∃ a : ℕ+, x = a ∧ y = 2 * a) ∨ (x = 3 ∧ y = 1) ∨ (x = 8 ∧ y = 1) :=
by sorry

end divisibility_condition_l2052_205264


namespace harvard_acceptance_rate_l2052_205245

/-- Proves that the percentage of accepted students is 5% given the conditions -/
theorem harvard_acceptance_rate 
  (total_applicants : ℕ) 
  (attendance_rate : ℚ) 
  (attending_students : ℕ) 
  (h1 : total_applicants = 20000)
  (h2 : attendance_rate = 9/10)
  (h3 : attending_students = 900) :
  (attending_students / attendance_rate) / total_applicants = 1/20 := by
  sorry

#check harvard_acceptance_rate

end harvard_acceptance_rate_l2052_205245


namespace sum_of_roots_l2052_205287

theorem sum_of_roots (k c x₁ x₂ : ℝ) (h_distinct : x₁ ≠ x₂) 
  (h₁ : 2 * x₁^2 - k * x₁ = 2 * c) (h₂ : 2 * x₂^2 - k * x₂ = 2 * c) : 
  x₁ + x₂ = k / 2 := by
sorry

end sum_of_roots_l2052_205287


namespace pen_cost_l2052_205215

theorem pen_cost (pen_cost ink_cost : ℝ) 
  (total_cost : pen_cost + ink_cost = 1.10)
  (price_difference : pen_cost = ink_cost + 1) : 
  pen_cost = 1.05 := by
sorry

end pen_cost_l2052_205215


namespace even_function_property_l2052_205257

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property 
  (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_increasing : ∀ x y, x < y → x < 0 → y < 0 → f x < f y)
  (x₁ x₂ : ℝ)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : x₂ > 0)
  (h_abs : abs x₁ < abs x₂) :
  f (-x₁) > f (-x₂) := by
sorry

end even_function_property_l2052_205257


namespace circle_diameter_when_area_circumference_ratio_is_5_l2052_205235

-- Define the circle properties
def circle_area (M : ℝ) := M
def circle_circumference (N : ℝ) := N

-- Theorem statement
theorem circle_diameter_when_area_circumference_ratio_is_5 
  (M N : ℝ) 
  (h1 : M > 0) 
  (h2 : N > 0) 
  (h3 : circle_area M / circle_circumference N = 5) : 
  2 * (circle_circumference N / (2 * Real.pi)) = 20 := by
  sorry

#check circle_diameter_when_area_circumference_ratio_is_5

end circle_diameter_when_area_circumference_ratio_is_5_l2052_205235


namespace fraction_equality_l2052_205286

theorem fraction_equality (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hab : a - b * (1 / a) ≠ 0) : 
  (a^2 - 1/b^2) / (b^2 - 1/a^2) = a^2 / b^2 := by
  sorry

end fraction_equality_l2052_205286


namespace shifted_parabola_passes_through_point_l2052_205207

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

/-- Evaluates a parabola at a given x-coordinate -/
def eval_parabola (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem shifted_parabola_passes_through_point :
  let original := Parabola.mk (-1) (-2) 3
  let shifted := shift_parabola original 1 (-2)
  eval_parabola shifted (-1) = 1 := by sorry

end shifted_parabola_passes_through_point_l2052_205207


namespace cubes_not_touching_foil_l2052_205210

/-- Represents a rectangular prism with inner and outer dimensions -/
structure RectangularPrism where
  inner_length : ℕ
  inner_width : ℕ
  inner_height : ℕ
  outer_width : ℕ

/-- Creates a RectangularPrism with the given constraints -/
def create_prism (outer_width : ℕ) : RectangularPrism :=
  { inner_length := (outer_width - 2) / 2,
    inner_width := outer_width - 2,
    inner_height := (outer_width - 2) / 2,
    outer_width := outer_width }

/-- Calculates the number of cubes not touching tin foil -/
def inner_cubes (prism : RectangularPrism) : ℕ :=
  prism.inner_length * prism.inner_width * prism.inner_height

/-- Theorem stating the number of cubes not touching tin foil -/
theorem cubes_not_touching_foil :
  inner_cubes (create_prism 10) = 128 := by
  sorry

#eval inner_cubes (create_prism 10)

end cubes_not_touching_foil_l2052_205210


namespace sequence_with_special_sums_l2052_205260

theorem sequence_with_special_sums : ∃ (seq : Fin 20 → ℝ),
  (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧
  (Finset.sum Finset.univ seq < 0) := by
  sorry

end sequence_with_special_sums_l2052_205260


namespace tomato_theorem_l2052_205217

def tomato_problem (initial_tomatoes : ℕ) : ℕ :=
  let after_first_birds := initial_tomatoes - initial_tomatoes / 3
  let after_second_birds := after_first_birds - after_first_birds / 2
  let final_tomatoes := after_second_birds + (after_second_birds + 1) / 2
  final_tomatoes

theorem tomato_theorem : tomato_problem 21 = 11 := by
  sorry

end tomato_theorem_l2052_205217


namespace line_contains_point_l2052_205239

/-- The value of k that makes the line 3 - ky = -4x contain the point (2, -1) -/
def k : ℝ := -11

/-- The equation of the line -/
def line_equation (x y : ℝ) (k : ℝ) : Prop :=
  3 - k * y = -4 * x

/-- The point that should lie on the line -/
def point : ℝ × ℝ := (2, -1)

/-- Theorem stating that k makes the line contain the given point -/
theorem line_contains_point : line_equation point.1 point.2 k := by sorry

end line_contains_point_l2052_205239


namespace solution_set_inequality1_solution_set_inequality2_l2052_205282

-- Problem 1
def inequality1 (x : ℝ) : Prop := abs (x - 2) + abs (2 * x - 3) < 4

theorem solution_set_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | 1/3 < x ∧ x < 3} :=
by sorry

-- Problem 2
def inequality2 (x : ℝ) : Prop := (x^2 - 3*x) / (x^2 - x - 2) ≤ x

theorem solution_set_inequality2 :
  {x : ℝ | inequality2 x} = 
    {x : ℝ | -1 < x ∧ x ≤ 0} ∪ {1} ∪ {x : ℝ | x > 2} :=
by sorry

end solution_set_inequality1_solution_set_inequality2_l2052_205282


namespace max_side_squared_acute_triangle_l2052_205208

theorem max_side_squared_acute_triangle (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 + 4 * c^2 = 8 →
  Real.sin B + 2 * Real.sin C = 6 * b * Real.sin A * Real.sin C →
  a^2 ≤ (15 - 8 * Real.sqrt 2) / 3 :=
by sorry

end max_side_squared_acute_triangle_l2052_205208


namespace electronic_components_probability_l2052_205206

theorem electronic_components_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : 1 - (1 - p)^3 = 0.999) : 
  p = 0.9 := by
sorry

end electronic_components_probability_l2052_205206


namespace total_colored_pencils_l2052_205234

theorem total_colored_pencils (madeline_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : ∃ cheryl_pencils : ℕ, cheryl_pencils = 2 * madeline_pencils)
  (h3 : ∃ cyrus_pencils : ℕ, 3 * cyrus_pencils = cheryl_pencils) :
  ∃ total_pencils : ℕ, total_pencils = madeline_pencils + cheryl_pencils + cyrus_pencils ∧ total_pencils = 231 :=
by
  sorry


end total_colored_pencils_l2052_205234


namespace perfect_square_trinomial_l2052_205265

theorem perfect_square_trinomial : 120^2 - 40 * 120 + 20^2 = 10000 := by
  sorry

end perfect_square_trinomial_l2052_205265


namespace lion_meeting_day_l2052_205289

/-- Represents days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day after a given day -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns true if the lion lies on the given day according to his pattern -/
def lionLies (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Friday ∨ d = Day.Saturday

/-- The day Alice met the lion -/
def meetingDay : Day := Day.Monday

theorem lion_meeting_day :
  (lionLies (nextDay (nextDay meetingDay)) ∧
   lionLies (nextDay (nextDay (nextDay meetingDay))) ∧
   ¬lionLies (nextDay meetingDay)) ∧
  ¬(lionLies (nextDay (nextDay (nextDay meetingDay))) ∧
    lionLies (nextDay (nextDay (nextDay (nextDay meetingDay)))) ∧
    ¬lionLies (nextDay (nextDay (nextDay (nextDay (nextDay meetingDay))))) ∧
    meetingDay ≠ Day.Monday) :=
by sorry


end lion_meeting_day_l2052_205289


namespace red_surface_area_fraction_is_three_fourths_l2052_205293

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  red_cube_count : ℕ
  blue_cube_count : ℕ
  blue_corners_per_face : ℕ

/-- The fraction of the surface area of the large cube that is red -/
def red_surface_area_fraction (c : LargeCube) : ℚ :=
  sorry

/-- The given large cube constructed from smaller cubes -/
def given_cube : LargeCube :=
  { edge_length := 4
  , small_cube_count := 64
  , red_cube_count := 32
  , blue_cube_count := 32
  , blue_corners_per_face := 4 }

theorem red_surface_area_fraction_is_three_fourths :
  red_surface_area_fraction given_cube = 3/4 := by
  sorry

end red_surface_area_fraction_is_three_fourths_l2052_205293


namespace twelfth_term_value_l2052_205248

-- Define the sequence
def a (n : ℕ) : ℚ := n / (n^2 + 1) * (-1)^(n+1)

-- State the theorem
theorem twelfth_term_value : a 12 = -12 / 145 := by
  sorry

end twelfth_term_value_l2052_205248


namespace geometric_sequence_problem_l2052_205271

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ 30 * r = a ∧ a * r = 9/4) : 
  a = 15 * Real.sqrt 3 / 2 := by
  sorry

end geometric_sequence_problem_l2052_205271


namespace apple_seedling_survival_probability_l2052_205226

/-- Survival rate data for apple seedlings -/
def survival_data : List (ℕ × ℝ) := [
  (100, 0.81),
  (200, 0.78),
  (500, 0.79),
  (1000, 0.8),
  (2000, 0.8)
]

/-- The estimated probability of survival for apple seedlings after transplantation -/
def estimated_survival_probability : ℝ := 0.8

/-- Theorem stating that the estimated probability of survival is 0.8 -/
theorem apple_seedling_survival_probability :
  estimated_survival_probability = 0.8 :=
sorry

end apple_seedling_survival_probability_l2052_205226


namespace diet_soda_bottles_l2052_205243

theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (diet : ℕ) : 
  total = 30 → regular = 28 → diet = total - regular → diet = 2 := by
  sorry

end diet_soda_bottles_l2052_205243


namespace min_even_integers_l2052_205233

theorem min_even_integers (a b c d e f g : ℤ) : 
  a + b + c = 30 →
  a + b + c + d + e = 48 →
  a + b + c + d + e + f + g = 60 →
  ∃ (a' b' c' d' e' f' g' : ℤ), 
    a' + b' + c' = 30 ∧
    a' + b' + c' + d' + e' = 48 ∧
    a' + b' + c' + d' + e' + f' + g' = 60 ∧
    Even a' ∧ Even b' ∧ Even c' ∧ Even d' ∧ Even e' ∧ Even f' ∧ Even g' :=
by sorry

end min_even_integers_l2052_205233


namespace digit_sum_problem_l2052_205230

theorem digit_sum_problem (x y z u : ℕ) : 
  x < 10 → y < 10 → z < 10 → u < 10 →
  x ≠ y → x ≠ z → x ≠ u → y ≠ z → y ≠ u → z ≠ u →
  10 * x + y + 10 * z + x = 10 * u + x - (10 * z + x) →
  x + y + z + u = 18 := by
sorry

end digit_sum_problem_l2052_205230


namespace symmetry_implies_k_and_b_l2052_205228

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are symmetric with respect to the vertical line x = a -/
def symmetric_lines (l1 l2 : Line) (a : ℝ) : Prop :=
  l1.slope = -l2.slope ∧
  l1.intercept + l2.intercept = 2 * (l1.slope * a + l1.intercept)

/-- The main theorem stating the conditions for symmetry and the resulting values of k and b -/
theorem symmetry_implies_k_and_b (k b : ℝ) :
  symmetric_lines (Line.mk k 3) (Line.mk 2 b) 1 →
  k = -2 ∧ b = -1 := by
  sorry

end symmetry_implies_k_and_b_l2052_205228


namespace point_inside_circle_l2052_205278

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside_circle (center_distance radius : ℝ) : Prop :=
  center_distance < radius

/-- Given a circle with radius 5 and a point A at distance 4 from the center,
    prove that point A is inside the circle -/
theorem point_inside_circle (center_distance radius : ℝ)
  (h1 : center_distance = 4)
  (h2 : radius = 5) :
  is_inside_circle center_distance radius := by
  sorry

end point_inside_circle_l2052_205278


namespace quadrilateral_AD_length_l2052_205285

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Real × Real)

-- Define the conditions of the problem
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_BAC_eq_BDA (q : Quadrilateral) : Prop := sorry

def angle_BAD_eq_60 (q : Quadrilateral) : Prop := sorry

def angle_ADC_eq_60 (q : Quadrilateral) : Prop := sorry

def length_AB_eq_14 (q : Quadrilateral) : Real := sorry

def length_CD_eq_6 (q : Quadrilateral) : Real := sorry

def length_AD (q : Quadrilateral) : Real := sorry

-- Theorem statement
theorem quadrilateral_AD_length 
  (q : Quadrilateral) 
  (h1 : is_convex q)
  (h2 : angle_BAC_eq_BDA q)
  (h3 : angle_BAD_eq_60 q)
  (h4 : angle_ADC_eq_60 q)
  (h5 : length_AB_eq_14 q = 14)
  (h6 : length_CD_eq_6 q = 6) :
  length_AD q = 20 := by sorry

end quadrilateral_AD_length_l2052_205285


namespace cube_edge_from_volume_l2052_205280

theorem cube_edge_from_volume (volume : ℝ) (edge : ℝ) :
  volume = 3375 ∧ volume = edge ^ 3 → edge = 15 := by
  sorry

end cube_edge_from_volume_l2052_205280


namespace apts_on_fewer_floors_eq_30_total_apts_on_fewer_floors_l2052_205241

/-- Represents a block of flats with given specifications -/
structure BlockOfFlats where
  total_floors : ℕ
  floors_with_more_apts : ℕ
  apts_on_more_floors : ℕ
  max_residents_per_apt : ℕ
  max_total_residents : ℕ

/-- The number of apartments on floors with fewer apartments -/
def apts_on_fewer_floors (b : BlockOfFlats) : ℕ :=
  (b.max_total_residents - b.max_residents_per_apt * b.floors_with_more_apts * b.apts_on_more_floors) /
  (b.max_residents_per_apt * (b.total_floors - b.floors_with_more_apts))

/-- Theorem stating the number of apartments on floors with fewer apartments -/
theorem apts_on_fewer_floors_eq_30 (b : BlockOfFlats) 
  (h1 : b.total_floors = 12)
  (h2 : b.floors_with_more_apts = 6)
  (h3 : b.apts_on_more_floors = 6)
  (h4 : b.max_residents_per_apt = 4)
  (h5 : b.max_total_residents = 264) :
  apts_on_fewer_floors b = 5 := by
  sorry

/-- Corollary for the total number of apartments on floors with fewer apartments -/
theorem total_apts_on_fewer_floors (b : BlockOfFlats) 
  (h1 : b.total_floors = 12)
  (h2 : b.floors_with_more_apts = 6)
  (h3 : b.apts_on_more_floors = 6)
  (h4 : b.max_residents_per_apt = 4)
  (h5 : b.max_total_residents = 264) :
  (b.total_floors - b.floors_with_more_apts) * apts_on_fewer_floors b = 30 := by
  sorry

end apts_on_fewer_floors_eq_30_total_apts_on_fewer_floors_l2052_205241


namespace product_calculation_l2052_205275

theorem product_calculation : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end product_calculation_l2052_205275


namespace final_amount_is_301_l2052_205213

def initial_quarters : ℕ := 7
def initial_dimes : ℕ := 3
def initial_nickels : ℕ := 5
def initial_pennies : ℕ := 12
def initial_half_dollars : ℕ := 3

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.1
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def half_dollar_value : ℚ := 0.5

def lose_one_of_each (q d n p h : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  (q - 1, d - 1, n - 1, p - 1, h - 1)

def exchange_nickels_for_dimes (n d : ℕ) : ℕ × ℕ :=
  (n - 3, d + 2)

def exchange_half_dollar (h q d : ℕ) : ℕ × ℕ × ℕ :=
  (h - 1, q + 1, d + 2)

def calculate_total (q d n p h : ℕ) : ℚ :=
  q * quarter_value + d * dime_value + n * nickel_value + 
  p * penny_value + h * half_dollar_value

theorem final_amount_is_301 :
  let (q1, d1, n1, p1, h1) := lose_one_of_each initial_quarters initial_dimes initial_nickels initial_pennies initial_half_dollars
  let (n2, d2) := exchange_nickels_for_dimes n1 d1
  let (h2, q2, d3) := exchange_half_dollar h1 q1 d2
  calculate_total q2 d3 n2 p1 h2 = 3.01 := by sorry

end final_amount_is_301_l2052_205213


namespace club_members_count_l2052_205261

theorem club_members_count :
  ∃! n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 ∧ n = 226 := by
  sorry

end club_members_count_l2052_205261


namespace day_crew_fraction_is_eight_elevenths_l2052_205223

/-- Represents the fraction of boxes loaded by the day crew given the relative productivity and size of the night crew -/
def day_crew_fraction (night_crew_productivity : ℚ) (night_crew_size : ℚ) : ℚ :=
  1 / (1 + night_crew_productivity * night_crew_size)

theorem day_crew_fraction_is_eight_elevenths :
  day_crew_fraction (3/4) (1/2) = 8/11 := by
  sorry

end day_crew_fraction_is_eight_elevenths_l2052_205223


namespace andrews_friends_pizza_l2052_205250

theorem andrews_friends_pizza (total_slices : ℕ) (slices_per_friend : ℕ) (num_friends : ℕ) :
  total_slices = 16 →
  slices_per_friend = 4 →
  total_slices = num_friends * slices_per_friend →
  num_friends = 4 := by
sorry

end andrews_friends_pizza_l2052_205250


namespace power_sum_l2052_205267

theorem power_sum (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end power_sum_l2052_205267


namespace roger_earnings_l2052_205218

theorem roger_earnings : ∀ (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ),
  rate = 9 →
  total_lawns = 14 →
  forgotten_lawns = 8 →
  (total_lawns - forgotten_lawns) * rate = 54 :=
by
  sorry

end roger_earnings_l2052_205218


namespace lunch_cost_theorem_l2052_205209

/-- The cost of a Taco Grande Plate -/
def taco_grande_cost : ℕ := 8

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℕ := 2 + 4 + 2

/-- Mike's total bill -/
def mike_bill : ℕ := taco_grande_cost + mike_additional_cost

/-- John's total bill -/
def john_bill : ℕ := taco_grande_cost

/-- The combined total cost of Mike and John's lunch -/
def combined_total_cost : ℕ := mike_bill + john_bill

theorem lunch_cost_theorem :
  (mike_bill = 2 * john_bill) →
  (combined_total_cost = 24) :=
by sorry

end lunch_cost_theorem_l2052_205209


namespace multiplication_formula_98_102_l2052_205229

theorem multiplication_formula_98_102 : 98 * 102 = 9996 := by
  sorry

end multiplication_formula_98_102_l2052_205229


namespace n_solution_approx_l2052_205288

def n_equation (n : ℝ) : Prop :=
  (n + 2 * 1.5) ^ 5 = (1 + 3 * 1.5) ^ 4

theorem n_solution_approx : ∃ n : ℝ, n_equation n ∧ abs (n - 0.72) < 0.01 := by
  sorry

end n_solution_approx_l2052_205288


namespace smallest_proportional_part_l2052_205276

theorem smallest_proportional_part (total : ℕ) (parts : List ℕ) : 
  total = 360 → 
  parts = [5, 7, 4, 8] → 
  List.length parts = 4 → 
  (List.sum parts) ∣ total → 
  (List.minimum parts).isSome → 
  (total / (List.sum parts)) * (List.minimum parts).get! = 60 :=
sorry

end smallest_proportional_part_l2052_205276


namespace student_average_greater_than_true_average_l2052_205253

theorem student_average_greater_than_true_average
  (x y z w : ℝ) (h : x < y ∧ y < z ∧ z < w) :
  ((((x + y) / 2 + z) / 2) + w) / 2 > (x + y + z + w) / 4 :=
by sorry

end student_average_greater_than_true_average_l2052_205253


namespace min_value_sqrt_sum_l2052_205240

theorem min_value_sqrt_sum (x : ℝ) :
  Real.sqrt (x^2 + 3*x + 3) + Real.sqrt (x^2 - 3*x + 3) ≥ 2 * Real.sqrt 3 := by
  sorry

end min_value_sqrt_sum_l2052_205240


namespace digit_sum_theorem_l2052_205274

theorem digit_sum_theorem (f o g : ℕ) : 
  f < 10 → o < 10 → g < 10 →
  4 * (100 * f + 10 * o + g) = 1464 →
  f + o + g = 15 := by
sorry

end digit_sum_theorem_l2052_205274


namespace product_of_symmetric_complex_numbers_l2052_205294

theorem product_of_symmetric_complex_numbers :
  ∀ (z₁ z₂ : ℂ),
  (z₁.im = -z₂.im) →  -- Symmetry with respect to real axis
  (z₁.re = z₂.re) →   -- Symmetry with respect to real axis
  (z₁ = 1 + I) →      -- Given condition
  z₁ * z₂ = 2 :=
by
  sorry

end product_of_symmetric_complex_numbers_l2052_205294


namespace snowboard_discount_proof_l2052_205298

theorem snowboard_discount_proof (original_price : ℝ) (friday_discount : ℝ) (monday_discount : ℝ) :
  original_price = 120 →
  friday_discount = 0.4 →
  monday_discount = 0.2 →
  let friday_price := original_price * (1 - friday_discount)
  let final_price := friday_price * (1 - monday_discount)
  final_price = 57.6 := by
sorry

end snowboard_discount_proof_l2052_205298


namespace print_time_rounded_l2052_205231

/-- Represents a printer with fast and normal modes -/
structure Printer :=
  (fast_speed : ℕ)
  (normal_speed : ℕ)

/-- Calculates the total printing time in minutes -/
def total_print_time (p : Printer) (fast_pages normal_pages : ℕ) : ℚ :=
  (fast_pages : ℚ) / p.fast_speed + (normal_pages : ℚ) / p.normal_speed

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem print_time_rounded (p : Printer) (h1 : p.fast_speed = 23) (h2 : p.normal_speed = 15) :
  round_to_nearest (total_print_time p 150 130) = 15 := by
  sorry

end print_time_rounded_l2052_205231


namespace prob_first_ace_is_one_eighth_l2052_205297

/-- Represents a deck of cards -/
structure Deck :=
  (size : ℕ)
  (num_aces : ℕ)

/-- Represents the card game setup -/
structure CardGame :=
  (deck : Deck)
  (num_players : ℕ)

/-- The probability of a player getting the first Ace -/
def prob_first_ace (game : CardGame) (player : ℕ) : ℚ :=
  1 / game.num_players

/-- Theorem stating that the probability of each player getting the first Ace is 1/8 -/
theorem prob_first_ace_is_one_eighth (game : CardGame) :
  game.deck.size = 32 ∧ game.deck.num_aces = 4 ∧ game.num_players = 4 →
  ∀ player, player > 0 ∧ player ≤ game.num_players →
    prob_first_ace game player = 1 / 8 :=
sorry

end prob_first_ace_is_one_eighth_l2052_205297


namespace seating_theorem_l2052_205273

/-- The number of seats in the row -/
def total_seats : ℕ := 8

/-- The number of people to be seated -/
def people_to_seat : ℕ := 3

/-- A function that calculates the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  -- The actual implementation is not provided in the problem
  sorry

/-- Theorem stating that the number of seating arrangements is 24 -/
theorem seating_theorem : seating_arrangements total_seats people_to_seat = 24 := by
  sorry

end seating_theorem_l2052_205273


namespace ellipse_eccentricity_l2052_205281

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The distance between the foci -/
  focal_distance : ℝ
  /-- The distance from the center to the line connecting a focus and the endpoint of the minor axis -/
  center_to_focus_minor_line : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem: The eccentricity of the ellipse with given properties is √5/3 -/
theorem ellipse_eccentricity (e : Ellipse) 
    (h1 : e.focal_distance = 3) 
    (h2 : e.center_to_focus_minor_line = 1) : 
  eccentricity e = Real.sqrt 5 / 3 := by
  sorry

end ellipse_eccentricity_l2052_205281


namespace stadium_length_feet_l2052_205268

/-- Converts yards to feet -/
def yards_to_feet (yards : ℕ) : ℕ := yards * 3

/-- The length of the sports stadium in yards -/
def stadium_length_yards : ℕ := 80

theorem stadium_length_feet :
  yards_to_feet stadium_length_yards = 240 := by
  sorry

end stadium_length_feet_l2052_205268
