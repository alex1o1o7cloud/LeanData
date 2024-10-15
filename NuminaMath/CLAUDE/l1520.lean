import Mathlib

namespace NUMINAMATH_CALUDE_sin_squared_sum_6_to_174_l1520_152021

theorem sin_squared_sum_6_to_174 : 
  (Finset.range 29).sum (fun k => Real.sin ((6 * k + 6 : ℕ) * π / 180) ^ 2) = 31 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_6_to_174_l1520_152021


namespace NUMINAMATH_CALUDE_sine_cosine_sum_simplification_l1520_152099

theorem sine_cosine_sum_simplification (x y : ℝ) : 
  Real.sin (x - 2*y) * Real.cos (3*y) + Real.cos (x - 2*y) * Real.sin (3*y) = Real.sin (x + y) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_simplification_l1520_152099


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_144_l1520_152057

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_144_l1520_152057


namespace NUMINAMATH_CALUDE_seed_germination_rate_l1520_152089

theorem seed_germination_rate (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot2 overall_germination_rate : ℝ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot2 = 35 →
  overall_germination_rate = 28.999999999999996 →
  (((overall_germination_rate / 100) * (seeds_plot1 + seeds_plot2) - 
    (germination_rate_plot2 / 100) * seeds_plot2) / seeds_plot1) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_rate_l1520_152089


namespace NUMINAMATH_CALUDE_circle_area_through_points_l1520_152046

/-- The area of a circle with center R(1, 2) passing through S(-7, 6) is 80π -/
theorem circle_area_through_points : 
  let R : ℝ × ℝ := (1, 2)
  let S : ℝ × ℝ := (-7, 6)
  let radius := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  π * radius^2 = 80 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l1520_152046


namespace NUMINAMATH_CALUDE_domain_of_f_l1520_152019

noncomputable def f (x : ℝ) := Real.log (2 * (Real.cos x)^2 - 1)

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem domain_of_f : domain f = {x : ℝ | ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l1520_152019


namespace NUMINAMATH_CALUDE_f_positive_iff_x_range_l1520_152094

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_positive_iff_x_range :
  (∀ x : ℝ, (∀ a ∈ Set.Icc (-1 : ℝ) 1, f x a > 0)) ↔
  (∀ x : ℝ, x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_x_range_l1520_152094


namespace NUMINAMATH_CALUDE_sum_equals_fraction_l1520_152064

/-- Given a real number k > 2 such that the infinite sum of (6n-2)/k^n from n=1 to infinity
    equals 31/9, prove that k = 147/62. -/
theorem sum_equals_fraction (k : ℝ) 
  (h1 : k > 2)
  (h2 : ∑' n, (6 * n - 2) / k^n = 31/9) : 
  k = 147/62 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_fraction_l1520_152064


namespace NUMINAMATH_CALUDE_gumballs_to_todd_l1520_152024

/-- Represents the distribution of gumballs among friends --/
structure GumballDistribution where
  total : ℕ
  remaining : ℕ
  todd : ℕ
  alisha : ℕ
  bobby : ℕ

/-- Checks if a gumball distribution satisfies the given conditions --/
def isValidDistribution (d : GumballDistribution) : Prop :=
  d.total = 45 ∧
  d.remaining = 6 ∧
  d.alisha = 2 * d.todd ∧
  d.bobby = 4 * d.alisha - 5 ∧
  d.total = d.todd + d.alisha + d.bobby + d.remaining

theorem gumballs_to_todd (d : GumballDistribution) :
  isValidDistribution d → d.todd = 4 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_to_todd_l1520_152024


namespace NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l1520_152020

-- Define the schools and their teacher compositions
def school_A : Nat := 3
def school_A_males : Nat := 2
def school_A_females : Nat := 1

def school_B : Nat := 3
def school_B_males : Nat := 1
def school_B_females : Nat := 2

def total_teachers : Nat := school_A + school_B

-- Theorem for the first question
theorem same_gender_probability :
  (school_A_males * school_B_males + school_A_females * school_B_females) /
  (school_A * school_B) = 4 / 9 :=
by sorry

-- Theorem for the second question
theorem same_school_probability :
  (school_A * (school_A - 1) / 2 + school_B * (school_B - 1) / 2) /
  (total_teachers * (total_teachers - 1) / 2) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l1520_152020


namespace NUMINAMATH_CALUDE_total_maggots_served_l1520_152055

def feeding_1 : ℕ := 10
def feeding_2 : ℕ := 15
def feeding_3 : ℕ := 2 * feeding_2
def feeding_4 : ℕ := feeding_3 - 5

theorem total_maggots_served :
  feeding_1 + feeding_2 + feeding_3 + feeding_4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_maggots_served_l1520_152055


namespace NUMINAMATH_CALUDE_pyramid_volume_theorem_l1520_152050

/-- Represents a pyramid with a square base ABCD and vertex E -/
structure Pyramid where
  baseArea : ℝ
  triangleABEArea : ℝ
  triangleCDEArea : ℝ

/-- Calculates the volume of the pyramid -/
def pyramidVolume (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the pyramid with given conditions -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.triangleABEArea = 128)
  (h3 : p.triangleCDEArea = 96) :
  pyramidVolume p = 1194 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_theorem_l1520_152050


namespace NUMINAMATH_CALUDE_perimeter_increase_is_237_point_5_percent_l1520_152069

/-- Represents the side length ratio between consecutive triangles -/
def ratio : ℝ := 1.5

/-- Calculates the percent increase in perimeter from the first to the fourth triangle -/
def perimeter_increase : ℝ :=
  (ratio^3 - 1) * 100

/-- Theorem stating that the percent increase in perimeter is 237.5% -/
theorem perimeter_increase_is_237_point_5_percent :
  ∃ ε > 0, |perimeter_increase - 237.5| < ε :=
sorry

end NUMINAMATH_CALUDE_perimeter_increase_is_237_point_5_percent_l1520_152069


namespace NUMINAMATH_CALUDE_factor_expression_l1520_152044

theorem factor_expression (z : ℝ) : 75 * z^23 + 225 * z^46 = 75 * z^23 * (1 + 3 * z^23) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1520_152044


namespace NUMINAMATH_CALUDE_pencil_profit_proof_l1520_152060

/-- Proves that selling 1500 pencils results in a profit of exactly $150.00 -/
theorem pencil_profit_proof (total_pencils : ℕ) (buy_price sell_price : ℚ) (profit_target : ℚ) 
  (h1 : total_pencils = 2000)
  (h2 : buy_price = 15/100)
  (h3 : sell_price = 30/100)
  (h4 : profit_target = 150) :
  (1500 : ℚ) * sell_price - (total_pencils : ℚ) * buy_price = profit_target := by
  sorry

end NUMINAMATH_CALUDE_pencil_profit_proof_l1520_152060


namespace NUMINAMATH_CALUDE_remaining_balance_proof_l1520_152098

def gift_card_balance (initial_balance : ℚ) (latte_price : ℚ) (croissant_price : ℚ) 
  (days : ℕ) (cookie_price : ℚ) (num_cookies : ℕ) : ℚ :=
  initial_balance - (latte_price + croissant_price) * days - cookie_price * num_cookies

theorem remaining_balance_proof :
  gift_card_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_proof_l1520_152098


namespace NUMINAMATH_CALUDE_area_after_shortening_l1520_152038

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := ⟨5, 7⟩

/-- Shortens either the length or the width of a rectangle by 2 --/
def shorten (r : Rectangle) (shortenLength : Bool) : Rectangle :=
  if shortenLength then ⟨r.length - 2, r.width⟩ else ⟨r.length, r.width - 2⟩

theorem area_after_shortening :
  (area (shorten original true) = 21 ∧ area (shorten original false) = 25) ∨
  (area (shorten original true) = 25 ∧ area (shorten original false) = 21) :=
by sorry

end NUMINAMATH_CALUDE_area_after_shortening_l1520_152038


namespace NUMINAMATH_CALUDE_mary_total_spending_l1520_152035

-- Define the amounts spent on each item
def shirt_cost : ℚ := 13.04
def jacket_cost : ℚ := 12.27

-- Define the total cost
def total_cost : ℚ := shirt_cost + jacket_cost

-- Theorem to prove
theorem mary_total_spending : total_cost = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_mary_total_spending_l1520_152035


namespace NUMINAMATH_CALUDE_number_less_than_l1520_152093

theorem number_less_than : (0.86 : ℝ) - 0.82 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_number_less_than_l1520_152093


namespace NUMINAMATH_CALUDE_factorial_vs_power_l1520_152081

theorem factorial_vs_power : 100^200 > Nat.factorial 200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_vs_power_l1520_152081


namespace NUMINAMATH_CALUDE_common_roots_product_l1520_152048

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10√[3]{2} -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ),
    (u^3 + C*u + 20 = 0) ∧ 
    (v^3 + C*v + 20 = 0) ∧ 
    (w^3 + C*w + 20 = 0) ∧
    (u^3 + D*u^2 + 100 = 0) ∧ 
    (v^3 + D*v^2 + 100 = 0) ∧ 
    (t^3 + D*t^2 + 100 = 0) ∧
    (u ≠ v) ∧ 
    (u * v = 10 * Real.rpow 2 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l1520_152048


namespace NUMINAMATH_CALUDE_max_profit_plan_l1520_152076

-- Define the appliance types
inductive Appliance
| TV
| Refrigerator
| WashingMachine

-- Define the cost and selling prices
def cost_price (a : Appliance) : ℕ :=
  match a with
  | Appliance.TV => 2000
  | Appliance.Refrigerator => 1600
  | Appliance.WashingMachine => 1000

def selling_price (a : Appliance) : ℕ :=
  match a with
  | Appliance.TV => 2200
  | Appliance.Refrigerator => 1800
  | Appliance.WashingMachine => 1100

-- Define the purchasing plan
structure PurchasingPlan where
  tv_count : ℕ
  refrigerator_count : ℕ
  washing_machine_count : ℕ

-- Define the constraints
def is_valid_plan (p : PurchasingPlan) : Prop :=
  p.tv_count + p.refrigerator_count + p.washing_machine_count = 100 ∧
  p.tv_count = p.refrigerator_count ∧
  p.washing_machine_count ≤ p.tv_count ∧
  p.tv_count * cost_price Appliance.TV +
  p.refrigerator_count * cost_price Appliance.Refrigerator +
  p.washing_machine_count * cost_price Appliance.WashingMachine ≤ 160000

-- Define the profit calculation
def profit (p : PurchasingPlan) : ℕ :=
  p.tv_count * (selling_price Appliance.TV - cost_price Appliance.TV) +
  p.refrigerator_count * (selling_price Appliance.Refrigerator - cost_price Appliance.Refrigerator) +
  p.washing_machine_count * (selling_price Appliance.WashingMachine - cost_price Appliance.WashingMachine)

-- Theorem statement
theorem max_profit_plan :
  ∃ (p : PurchasingPlan),
    is_valid_plan p ∧
    profit p = 17400 ∧
    ∀ (q : PurchasingPlan), is_valid_plan q → profit q ≤ profit p :=
sorry

end NUMINAMATH_CALUDE_max_profit_plan_l1520_152076


namespace NUMINAMATH_CALUDE_ken_released_three_fish_l1520_152066

/-- The number of fish Ken released -/
def fish_released (ken_caught : ℕ) (kendra_caught : ℕ) (brought_home : ℕ) : ℕ :=
  ken_caught + kendra_caught - brought_home

theorem ken_released_three_fish :
  ∀ (ken_caught kendra_caught brought_home : ℕ),
  ken_caught = 2 * kendra_caught →
  kendra_caught = 30 →
  brought_home = 87 →
  fish_released ken_caught kendra_caught brought_home = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ken_released_three_fish_l1520_152066


namespace NUMINAMATH_CALUDE_complementary_angle_adjustment_l1520_152006

theorem complementary_angle_adjustment (a b : ℝ) (h1 : a + b = 90) (h2 : a / b = 1 / 2) :
  let a' := a * 1.2
  let b' := 90 - a'
  (b - b') / b = 0.1 := by sorry

end NUMINAMATH_CALUDE_complementary_angle_adjustment_l1520_152006


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1520_152082

theorem decimal_to_fraction : 
  ∃ (n d : ℤ), d ≠ 0 ∧ 3.75 = (n : ℚ) / (d : ℚ) ∧ n = 15 ∧ d = 4 :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1520_152082


namespace NUMINAMATH_CALUDE_inequality_proof_l1520_152010

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Define the set M
def M : Set ℝ := {x | f x < 4}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  2 * |a + b| < |4 + a * b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1520_152010


namespace NUMINAMATH_CALUDE_product_ab_equals_ten_l1520_152016

theorem product_ab_equals_ten (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + b + c = 21) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_ab_equals_ten_l1520_152016


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1520_152079

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 + 2*x < 3

-- Define the solution set
def solution_set : Set ℝ := {x | -3 < x ∧ x < 1}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1520_152079


namespace NUMINAMATH_CALUDE_trig_identity_l1520_152049

theorem trig_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sqrt 3 * Real.sin (10 * π / 180) * Real.tan (70 * π / 180) - 
  2 * Real.sin (50 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1520_152049


namespace NUMINAMATH_CALUDE_det_A_eq_58_l1520_152042

def A : Matrix (Fin 2) (Fin 2) ℝ := !![10, 4; -2, 5]

theorem det_A_eq_58 : Matrix.det A = 58 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_58_l1520_152042


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l1520_152030

def A (a : ℝ) : Set ℝ := {2, 3, a^2 + 4*a + 2}

def B (a : ℝ) : Set ℝ := {0, 7, 2 - a, a^2 + 4*a - 2}

theorem set_intersection_and_union (a : ℝ) :
  A a ∩ B a = {3, 7} → a = 1 ∧ A a ∪ B a = {0, 1, 2, 3, 7} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l1520_152030


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l1520_152078

/-- Represents the angles of a quadrilateral in arithmetic sequence -/
structure QuadrilateralAngles where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- Conditions for the quadrilateral angles -/
def quadrilateral_conditions (q : QuadrilateralAngles) : Prop :=
  q.a > 0 ∧
  q.d > 0 ∧
  q.a + (q.a + q.d) + (q.a + 2 * q.d) + (q.a + 3 * q.d) = 360 ∧
  q.a + (q.a + 2 * q.d) = 160

theorem smallest_angle_measure (q : QuadrilateralAngles) 
  (h : quadrilateral_conditions q) : q.a = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l1520_152078


namespace NUMINAMATH_CALUDE_max_product_arithmetic_mean_l1520_152045

theorem max_product_arithmetic_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : 2 = (2 * a + b) / 2) : 
  a * b ≤ 2 ∧ (a * b = 2 ↔ b = 2 ∧ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_max_product_arithmetic_mean_l1520_152045


namespace NUMINAMATH_CALUDE_james_lego_collection_l1520_152028

/-- Represents the number of Legos in James' collection -/
def initial_collection : ℕ := sorry

/-- Represents the number of Legos James uses for his castle -/
def used_legos : ℕ := sorry

/-- Represents the number of Legos put back in the box -/
def legos_in_box : ℕ := 245

/-- Represents the number of missing Legos -/
def missing_legos : ℕ := 5

theorem james_lego_collection :
  (initial_collection = 500) ∧
  (used_legos = initial_collection / 2) ∧
  (legos_in_box + missing_legos = initial_collection - used_legos) :=
sorry

end NUMINAMATH_CALUDE_james_lego_collection_l1520_152028


namespace NUMINAMATH_CALUDE_min_framing_for_specific_picture_l1520_152097

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height border_width : ℕ) : ℕ :=
  let enlarged_width := 2 * original_width
  let enlarged_height := 2 * original_height
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  (perimeter_inches + 11) / 12

/-- Theorem stating that for a 5-inch by 7-inch picture, enlarged and bordered as described, 
    the minimum framing needed is 6 feet. -/
theorem min_framing_for_specific_picture : 
  min_framing_feet 5 7 3 = 6 := by
  sorry

#eval min_framing_feet 5 7 3

end NUMINAMATH_CALUDE_min_framing_for_specific_picture_l1520_152097


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_300_l1520_152002

theorem distinct_prime_factors_of_300 : Nat.card (Nat.factors 300).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_300_l1520_152002


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1520_152043

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (2 + 1 / (3 + 1 / 4))) = 30 / 43 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1520_152043


namespace NUMINAMATH_CALUDE_scout_troop_profit_l1520_152052

/-- The profit calculation for a scout troop selling candy bars -/
theorem scout_troop_profit : 
  let total_bars : ℕ := 1500
  let cost_price : ℚ := 1 / 3  -- price per bar when buying more than 800
  let selling_price : ℚ := 1 / 2  -- price per bar when selling
  let total_cost : ℚ := total_bars * cost_price
  let total_revenue : ℚ := total_bars * selling_price
  let profit : ℚ := total_revenue - total_cost
  profit = 250 := by sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l1520_152052


namespace NUMINAMATH_CALUDE_clothing_prices_l1520_152051

-- Define the original prices
def original_sweater_price : ℝ := 43.11
def original_shirt_price : ℝ := original_sweater_price - 7.43
def original_pants_price : ℝ := 2 * original_shirt_price

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the total cost after discount
def total_cost : ℝ := 143.67

-- Theorem statement
theorem clothing_prices :
  (original_shirt_price = 35.68) ∧
  (original_sweater_price = 43.11) ∧
  (original_pants_price = 71.36) ∧
  (original_shirt_price + (1 - discount_rate) * original_sweater_price + original_pants_price = total_cost) := by
  sorry

end NUMINAMATH_CALUDE_clothing_prices_l1520_152051


namespace NUMINAMATH_CALUDE_min_width_for_rectangular_area_l1520_152053

theorem min_width_for_rectangular_area :
  ∀ w : ℝ,
  w > 0 →
  w * (w + 18) ≥ 150 →
  (∀ x : ℝ, x > 0 ∧ x * (x + 18) ≥ 150 → x ≥ w) →
  w = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_width_for_rectangular_area_l1520_152053


namespace NUMINAMATH_CALUDE_julie_rowing_distance_l1520_152075

theorem julie_rowing_distance (downstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) 
  (h1 : downstream_distance = 72)
  (h2 : time = 4)
  (h3 : stream_speed = 0.5) :
  ∃ (upstream_distance : ℝ), 
    upstream_distance = 68 ∧ 
    time = upstream_distance / (downstream_distance / (2 * time) - stream_speed) ∧
    time = downstream_distance / (downstream_distance / (2 * time) + stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_julie_rowing_distance_l1520_152075


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_l1520_152070

/-- For a parabola y = x^2 + 2x + m - 1 to intersect with the x-axis, m must be less than or equal to 2 -/
theorem parabola_intersects_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m - 1 = 0) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_l1520_152070


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1520_152040

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = -12) : 
  x^2 + y^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1520_152040


namespace NUMINAMATH_CALUDE_unique_solution_system_l1520_152011

theorem unique_solution_system (a b : ℕ+) 
  (h1 : a^(b:ℕ) + 3 = b^(a:ℕ)) 
  (h2 : 3 * a^(b:ℕ) = b^(a:ℕ) + 13) : 
  a = 2 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1520_152011


namespace NUMINAMATH_CALUDE_faye_age_l1520_152007

/-- Represents the ages of the individuals in the problem --/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ
  george : ℕ

/-- The conditions of the problem --/
def valid_ages (a : Ages) : Prop :=
  a.diana + 2 = a.eduardo ∧
  a.eduardo = a.chad + 6 ∧
  a.faye = a.chad + 4 ∧
  a.george + 5 = a.chad ∧
  a.diana = 16

/-- The theorem to prove --/
theorem faye_age (a : Ages) (h : valid_ages a) : a.faye = 16 := by
  sorry

#check faye_age

end NUMINAMATH_CALUDE_faye_age_l1520_152007


namespace NUMINAMATH_CALUDE_distance_Y_to_GH_l1520_152036

-- Define the square
def Square (t : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ t ∧ 0 ≤ p.2 ∧ p.2 ≤ t}

-- Define the half-circle centered at E
def ArcE (t : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = (t/2)^2 ∧ 0 ≤ p.1 ∧ 0 ≤ p.2}

-- Define the half-circle centered at F
def ArcF (t : ℝ) := {p : ℝ × ℝ | (p.1 - t)^2 + p.2^2 = (3*t/2)^2 ∧ p.1 ≤ t ∧ 0 ≤ p.2}

-- Define the intersection point Y
def Y (t : ℝ) := {p : ℝ × ℝ | p ∈ ArcE t ∧ p ∈ ArcF t ∧ p ∈ Square t}

-- Theorem statement
theorem distance_Y_to_GH (t : ℝ) (h : t > 0) :
  ∀ y ∈ Y t, t - y.2 = t :=
sorry

end NUMINAMATH_CALUDE_distance_Y_to_GH_l1520_152036


namespace NUMINAMATH_CALUDE_three_correct_deliveries_l1520_152084

def num_houses : ℕ := 5
def num_packages : ℕ := 5

def probability_three_correct : ℚ := 1 / 6

theorem three_correct_deliveries :
  let total_arrangements := num_houses.factorial
  let correct_three_ways := num_houses.choose 3
  let incorrect_two_ways := 1  -- derangement of 2
  let prob_three_correct := correct_three_ways * incorrect_two_ways / total_arrangements
  prob_three_correct = probability_three_correct := by sorry

end NUMINAMATH_CALUDE_three_correct_deliveries_l1520_152084


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l1520_152059

/-- Proves the distance traveled downstream by a boat -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : travel_time = 5) : 
  boat_speed + stream_speed * travel_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l1520_152059


namespace NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l1520_152004

/-- Proof of the original ratio of boarders to day students -/
theorem original_ratio_of_boarders_to_day_students :
  let initial_boarders : ℕ := 120
  let new_boarders : ℕ := 30
  let total_boarders : ℕ := initial_boarders + new_boarders
  let day_students : ℕ := 2 * total_boarders
  (initial_boarders : ℚ) / day_students = 1 / (5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l1520_152004


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1520_152090

theorem arithmetic_expression_evaluation : 7 + 15 / 3 - 5 * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1520_152090


namespace NUMINAMATH_CALUDE_max_value_condition_l1520_152005

theorem max_value_condition (x y : ℝ) : 
  (2 * x^2 - y^2 + 3/2 ≤ 1 ∧ y^4 + 4*x + 2 ≤ 1) ↔ 
  ((x = -1/2 ∧ y = 1) ∨ (x = -1/2 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_condition_l1520_152005


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1520_152001

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 4) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = -1) ∧ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1520_152001


namespace NUMINAMATH_CALUDE_equation_solution_l1520_152025

theorem equation_solution : ∃ x : ℝ, 
  x = 160 + 64 * Real.sqrt 6 ∧ 
  Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1520_152025


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1520_152088

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1520_152088


namespace NUMINAMATH_CALUDE_asian_games_survey_l1520_152071

theorem asian_games_survey (total students : ℕ) 
  (table_tennis badminton not_interested : ℕ) : 
  total = 50 → 
  table_tennis = 35 → 
  badminton = 30 → 
  not_interested = 5 → 
  table_tennis + badminton - (total - not_interested) = 20 := by
  sorry

end NUMINAMATH_CALUDE_asian_games_survey_l1520_152071


namespace NUMINAMATH_CALUDE_min_abs_z_on_locus_l1520_152012

theorem min_abs_z_on_locus (z : ℂ) (h : Complex.abs (z - (0 : ℂ) + 4*I) + Complex.abs (z - 5) = 7) : 
  ∃ (w : ℂ), Complex.abs (w - (0 : ℂ) + 4*I) + Complex.abs (w - 5) = 7 ∧ 
  (∀ (v : ℂ), Complex.abs (v - (0 : ℂ) + 4*I) + Complex.abs (v - 5) = 7 → Complex.abs w ≤ Complex.abs v) ∧
  Complex.abs w = 20 / 7 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_on_locus_l1520_152012


namespace NUMINAMATH_CALUDE_a_bounds_l1520_152026

/-- Given a linear equation y = ax + 1/3 where x and y are bounded,
    prove that a is bounded between -1/3 and 2/3. -/
theorem a_bounds (a : ℝ) : 
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ y = a * x + 1/3) →
  -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry

#check a_bounds

end NUMINAMATH_CALUDE_a_bounds_l1520_152026


namespace NUMINAMATH_CALUDE_classmate_heights_most_suitable_l1520_152087

/-- Represents a survey option -/
inductive SurveyOption
  | LightBulbs
  | RiverWater
  | TVViewership
  | ClassmateHeights

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  accessibility : Bool
  non_destructive : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (s : SurveyCharacteristics) : Prop :=
  s.population_size < 1000 ∧ s.accessibility ∧ s.non_destructive

/-- Assigns characteristics to each survey option -/
def survey_properties : SurveyOption → SurveyCharacteristics
  | SurveyOption.LightBulbs => ⟨100, true, false⟩
  | SurveyOption.RiverWater => ⟨10000, false, true⟩
  | SurveyOption.TVViewership => ⟨1000000, false, true⟩
  | SurveyOption.ClassmateHeights => ⟨30, true, true⟩

/-- Theorem stating that surveying classmate heights is the most suitable for a comprehensive survey -/
theorem classmate_heights_most_suitable :
  ∀ (s : SurveyOption), s ≠ SurveyOption.ClassmateHeights →
  ¬(is_comprehensive (survey_properties s)) ∧
  (is_comprehensive (survey_properties SurveyOption.ClassmateHeights)) :=
sorry


end NUMINAMATH_CALUDE_classmate_heights_most_suitable_l1520_152087


namespace NUMINAMATH_CALUDE_P_identity_l1520_152009

def P (n : ℕ) : ℕ := (n + 1).factorial / n.factorial

def oddProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

theorem P_identity (n : ℕ) : P n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_P_identity_l1520_152009


namespace NUMINAMATH_CALUDE_sundae_price_l1520_152096

/-- Given a caterer's order of ice-cream bars and sundaes, calculate the price of each sundae. -/
theorem sundae_price
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℚ)
  (ice_cream_bar_price : ℚ)
  (h1 : num_ice_cream_bars = 225)
  (h2 : num_sundaes = 125)
  (h3 : total_price = 200)
  (h4 : ice_cream_bar_price = 0.60) :
  (total_price - (↑num_ice_cream_bars * ice_cream_bar_price)) / ↑num_sundaes = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_sundae_price_l1520_152096


namespace NUMINAMATH_CALUDE_smallest_unachievable_score_l1520_152058

def dart_scores : Set ℕ := {0, 1, 3, 8, 12}

def is_achievable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ dart_scores ∧ b ∈ dart_scores ∧ c ∈ dart_scores ∧ a + b + c = n

theorem smallest_unachievable_score :
  (∀ m < 22, is_achievable m) ∧ ¬is_achievable 22 :=
sorry

end NUMINAMATH_CALUDE_smallest_unachievable_score_l1520_152058


namespace NUMINAMATH_CALUDE_right_trapezoid_with_inscribed_circle_sides_l1520_152068

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  R : ℝ
  shorter_base : ℝ
  longer_base : ℝ
  longer_leg : ℝ
  shorter_base_eq : shorter_base = 4/3 * R

/-- Theorem: In a right trapezoid with an inscribed circle of radius R and shorter base 4/3 R, 
    the longer base is 4R and the longer leg is 10/3 R -/
theorem right_trapezoid_with_inscribed_circle_sides 
  (t : RightTrapezoidWithInscribedCircle) : 
  t.longer_base = 4 * t.R ∧ t.longer_leg = 10/3 * t.R := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_with_inscribed_circle_sides_l1520_152068


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1520_152033

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (l w : ℝ),
    r = 7 →
    l / w = 3 →
    w = 2 * r →
    l * w = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1520_152033


namespace NUMINAMATH_CALUDE_subsets_and_proper_subsets_of_S_l1520_152003

def S : Set ℕ := {0, 1, 2}

theorem subsets_and_proper_subsets_of_S :
  (Finset.powerset {0, 1, 2} = {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}}) ∧
  (Finset.powerset {0, 1, 2} \ {{0, 1, 2}} = {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}}) := by
  sorry

end NUMINAMATH_CALUDE_subsets_and_proper_subsets_of_S_l1520_152003


namespace NUMINAMATH_CALUDE_b_17_value_l1520_152034

/-- A sequence where consecutive terms are roots of a quadratic equation -/
def special_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, (a n)^2 - n * (a n) + (b n) = 0 ∧
       (a (n + 1))^2 - n * (a (n + 1)) + (b n) = 0

theorem b_17_value (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h : special_sequence a b) (h10 : a 10 = 7) : b 17 = 66 := by
  sorry

end NUMINAMATH_CALUDE_b_17_value_l1520_152034


namespace NUMINAMATH_CALUDE_new_crew_member_weight_l1520_152014

/-- Given a crew of 10 oarsmen, prove that replacing a 53 kg member with a new member
    that increases the average weight by 1.8 kg results in the new member weighing 71 kg. -/
theorem new_crew_member_weight (crew_size : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  crew_size = 10 →
  weight_increase = 1.8 →
  replaced_weight = 53 →
  (crew_size : ℝ) * weight_increase + replaced_weight = 71 := by
  sorry

end NUMINAMATH_CALUDE_new_crew_member_weight_l1520_152014


namespace NUMINAMATH_CALUDE_log_product_theorem_l1520_152085

-- Define the exponent rule
axiom exponent_rule {a : ℝ} (m n : ℝ) : a^m * a^n = a^(m + n)

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_product_theorem (b x y : ℝ) (hb : b > 0) (hb1 : b ≠ 1) (hx : x > 0) (hy : y > 0) :
  log b (x * y) = log b x + log b y :=
sorry

end NUMINAMATH_CALUDE_log_product_theorem_l1520_152085


namespace NUMINAMATH_CALUDE_reflection_direction_vector_l1520_152013

/-- Given a particle moving in a plane along direction u = (1,2) and reflecting off a line l
    to move in direction v = (-2,1) according to the optical principle,
    one possible direction vector of line l is ω = (1,-3). -/
theorem reflection_direction_vector :
  let u : ℝ × ℝ := (1, 2)
  let v : ℝ × ℝ := (-2, 1)
  ∃ ω : ℝ × ℝ, ω = (1, -3) ∧
    (∀ k : ℝ, (k - 2) / (1 + 2*k) = (-1/2 - k) / (1 - 1/2*k) → k = -3) ∧
    (∀ θ₁ θ₂ : ℝ, θ₁ = θ₂ → 
      (u.2 / u.1 - ω.2 / ω.1) / (1 + (u.2 / u.1) * (ω.2 / ω.1)) =
      (v.2 / v.1 - ω.2 / ω.1) / (1 + (v.2 / v.1) * (ω.2 / ω.1))) :=
by sorry

end NUMINAMATH_CALUDE_reflection_direction_vector_l1520_152013


namespace NUMINAMATH_CALUDE_ed_lost_seven_marbles_l1520_152037

/-- Represents the number of marbles each person has -/
structure MarbleCount where
  doug : ℕ
  ed : ℕ
  tim : ℕ

/-- The initial state of marble distribution -/
def initial_state (d : ℕ) : MarbleCount :=
  { doug := d
  , ed := d + 19
  , tim := d - 10 }

/-- The final state of marble distribution after transactions -/
def final_state (d : ℕ) (l : ℕ) : MarbleCount :=
  { doug := d
  , ed := d + 8
  , tim := d }

/-- Theorem stating that Ed lost 7 marbles -/
theorem ed_lost_seven_marbles (d : ℕ) :
  ∃ l : ℕ, 
    (initial_state d).ed - l - 4 = (final_state d l).ed ∧
    (initial_state d).tim + 4 + 3 = (final_state d l).tim ∧
    l = 7 := by
  sorry

#check ed_lost_seven_marbles

end NUMINAMATH_CALUDE_ed_lost_seven_marbles_l1520_152037


namespace NUMINAMATH_CALUDE_inequality_proof_l1520_152077

theorem inequality_proof (x : ℝ) 
  (h : (abs x ≤ 1) ∨ (abs x ≥ 2)) : 
  Real.cos (2*x^3 - x^2 - 5*x - 2) + 
  Real.cos (2*x^3 + 3*x^2 - 3*x - 2) - 
  Real.cos ((2*x + 1) * Real.sqrt (x^4 - 5*x^2 + 4)) < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1520_152077


namespace NUMINAMATH_CALUDE_triangle_area_coefficient_product_l1520_152054

/-- Given a triangle in the first quadrant bounded by the coordinate axes and a line,
    prove that if the area is 9, then the product of the coefficients is 4/3. -/
theorem triangle_area_coefficient_product (a b : ℝ) : 
  a > 0 → b > 0 → (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 2*a*x + 3*b*y ≤ 12) → 
  (1/2 * (12/(2*a)) * (12/(3*b)) = 9) → a * b = 4/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_coefficient_product_l1520_152054


namespace NUMINAMATH_CALUDE_sphere_intersection_circles_area_sum_l1520_152092

/-- Given a sphere of radius R and a point inside it at distance d from the center,
    the sum of the areas of three circles formed by the intersection of three
    mutually perpendicular planes passing through the point is equal to π(3R² - d²). -/
theorem sphere_intersection_circles_area_sum
  (R d : ℝ) (h_R : R > 0) (h_d : 0 ≤ d ∧ d < R) :
  ∃ (A : ℝ), A = π * (3 * R^2 - d^2) ∧
  ∀ (x y z : ℝ),
    x^2 + y^2 + z^2 = d^2 →
    A = π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_intersection_circles_area_sum_l1520_152092


namespace NUMINAMATH_CALUDE_max_value_implies_m_equals_four_l1520_152008

theorem max_value_implies_m_equals_four (x y m : ℝ) : 
  x > 1 →
  y ≥ x →
  y ≤ 2 * x →
  x + y ≤ 1 →
  (∀ x' y' : ℝ, y' ≥ x' → y' ≤ 2 * x' → x' + y' ≤ 1 → x' + m * y' ≤ x + m * y) →
  x + m * y = 3 →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_equals_four_l1520_152008


namespace NUMINAMATH_CALUDE_milkshake_production_l1520_152000

/-- Augustus's milkshake production rate per hour -/
def augustus_rate : ℕ := 3

/-- Luna's milkshake production rate per hour -/
def luna_rate : ℕ := 7

/-- The number of hours Augustus and Luna have been making milkshakes -/
def hours_worked : ℕ := 8

/-- The total number of milkshakes made by Augustus and Luna -/
def total_milkshakes : ℕ := (augustus_rate + luna_rate) * hours_worked

theorem milkshake_production :
  total_milkshakes = 80 := by sorry

end NUMINAMATH_CALUDE_milkshake_production_l1520_152000


namespace NUMINAMATH_CALUDE_parallel_vectors_difference_l1520_152041

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_difference (x : ℝ) :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (x - 2, -2)
  are_parallel a b → a - b = (-2, 1) := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_difference_l1520_152041


namespace NUMINAMATH_CALUDE_line_intersection_symmetry_l1520_152018

/-- Given a line y = -x + m intersecting the x-axis at A, prove that when moved 6 units left
    to intersect the x-axis at A', if A' is symmetric to A about the origin, then m = 3. -/
theorem line_intersection_symmetry (m : ℝ) : 
  let A : ℝ × ℝ := (m, 0)
  let A' : ℝ × ℝ := (m - 6, 0)
  (A'.1 = -A.1 ∧ A'.2 = -A.2) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_symmetry_l1520_152018


namespace NUMINAMATH_CALUDE_car_speed_problem_l1520_152031

/-- Proves that if a car traveling at 600 km/h takes 2 seconds longer to cover 1 km 
    than it would at speed v km/h, then v = 900 km/h. -/
theorem car_speed_problem (v : ℝ) : 
  (1 / (600 / 3600) - 1 / (v / 3600) = 2) → v = 900 := by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l1520_152031


namespace NUMINAMATH_CALUDE_linear_decreasing_iff_k_lt_neg_half_l1520_152072

/-- A function f: ℝ → ℝ is decreasing if for all x₁ < x₂, f(x₁) > f(x₂) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The linear function y = (2k+1)x + b -/
def f (k b : ℝ) (x : ℝ) : ℝ := (2*k + 1)*x + b

theorem linear_decreasing_iff_k_lt_neg_half (k b : ℝ) :
  IsDecreasing (f k b) ↔ k < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_iff_k_lt_neg_half_l1520_152072


namespace NUMINAMATH_CALUDE_sixth_term_value_l1520_152027

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  a_1_eq_4 : a 1 = 4
  a_3_eq_prod : a 3 = a 2 * a 4

/-- The sixth term of the geometric sequence is either 1/8 or -1/8 -/
theorem sixth_term_value (seq : GeometricSequence) : 
  seq.a 6 = 1/8 ∨ seq.a 6 = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l1520_152027


namespace NUMINAMATH_CALUDE_breakfast_cost_l1520_152067

def toast_price : ℕ := 1
def egg_price : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

theorem breakfast_cost : 
  (dale_toast * toast_price + dale_eggs * egg_price) +
  (andrew_toast * toast_price + andrew_eggs * egg_price) = 15 := by
sorry

end NUMINAMATH_CALUDE_breakfast_cost_l1520_152067


namespace NUMINAMATH_CALUDE_library_wage_calculation_l1520_152032

/-- Represents the weekly work schedule and earnings of a student with two part-time jobs -/
structure WorkSchedule where
  library_hours : ℝ
  construction_hours : ℝ
  library_wage : ℝ
  construction_wage : ℝ
  total_earnings : ℝ

/-- Theorem stating the library wage given the problem conditions -/
theorem library_wage_calculation (w : WorkSchedule) :
  w.library_hours = 10 ∧
  w.construction_hours = 15 ∧
  w.construction_wage = 15 ∧
  w.library_hours + w.construction_hours = 25 ∧
  w.total_earnings ≥ 300 ∧
  w.total_earnings = w.library_hours * w.library_wage + w.construction_hours * w.construction_wage →
  w.library_wage = 7.5 := by
  sorry

#check library_wage_calculation

end NUMINAMATH_CALUDE_library_wage_calculation_l1520_152032


namespace NUMINAMATH_CALUDE_escalator_speed_calculation_l1520_152056

/-- The speed of the escalator in feet per second. -/
def escalator_speed : ℝ := 12

/-- The length of the escalator in feet. -/
def escalator_length : ℝ := 160

/-- The walking speed of the person in feet per second. -/
def walking_speed : ℝ := 8

/-- The time taken to cover the entire length of the escalator in seconds. -/
def time_taken : ℝ := 8

theorem escalator_speed_calculation :
  (walking_speed + escalator_speed) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_calculation_l1520_152056


namespace NUMINAMATH_CALUDE_ball_throw_height_difference_l1520_152083

/-- A proof of the height difference between Janice's final throw and Christine's first throw -/
theorem ball_throw_height_difference :
  let christine_first : ℕ := 20
  let janice_first : ℕ := christine_first - 4
  let christine_second : ℕ := christine_first + 10
  let janice_second : ℕ := janice_first * 2
  let christine_third : ℕ := christine_second + 4
  let highest_throw : ℕ := 37
  let janice_third : ℕ := highest_throw
  janice_third - christine_first = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_throw_height_difference_l1520_152083


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1520_152061

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + 7 * y = 14
def line2 (k x y : ℝ) : Prop := k * x - y = k + 1

-- Define the intersection point
def intersection (k : ℝ) : Prop :=
  ∃ x y : ℝ, line1 x y ∧ line2 k x y

-- Define the first quadrant condition
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem intersection_in_first_quadrant :
  ∀ k : ℝ, (∃ x y : ℝ, intersection k ∧ first_quadrant x y) → k > 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1520_152061


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_theorem_l1520_152065

def numbers : List Nat := [15, 20, 30]

theorem gcf_lcm_sum_theorem :
  (Nat.gcd (Nat.gcd 15 20) 30) + (Nat.lcm (Nat.lcm 15 20) 30) = 65 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_theorem_l1520_152065


namespace NUMINAMATH_CALUDE_gianna_savings_l1520_152029

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℕ) * d)

/-- Gianna's savings problem -/
theorem gianna_savings :
  arithmetic_sum 365 39 2 = 147095 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_l1520_152029


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1520_152073

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  EF : ℝ
  angleEGF : ℝ
  angleFHE : ℝ
  height : ℝ
  EF_length : EF = 60
  angleEGF_value : angleEGF = 45 * π / 180
  angleFHE_value : angleFHE = 45 * π / 180
  height_value : height = 30 * Real.sqrt 2

/-- The perimeter of the trapezoid EFGH is 180 + 60√2 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  ∃ (perimeter : ℝ), perimeter = 180 + 60 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1520_152073


namespace NUMINAMATH_CALUDE_cathys_money_proof_l1520_152023

/-- Calculates the total amount of money Cathy has after receiving contributions from her parents. -/
def cathys_total_money (initial : ℕ) (dads_contribution : ℕ) : ℕ :=
  initial + dads_contribution + 2 * dads_contribution

/-- Proves that Cathy's total money is 87 given the initial conditions. -/
theorem cathys_money_proof :
  cathys_total_money 12 25 = 87 := by
  sorry

#eval cathys_total_money 12 25

end NUMINAMATH_CALUDE_cathys_money_proof_l1520_152023


namespace NUMINAMATH_CALUDE_megans_hourly_wage_l1520_152086

/-- Megan's hourly wage problem -/
theorem megans_hourly_wage (hours_per_day : ℕ) (days_per_month : ℕ) (earnings_two_months : ℕ) 
  (h1 : hours_per_day = 8)
  (h2 : days_per_month = 20)
  (h3 : earnings_two_months = 2400) :
  (earnings_two_months : ℚ) / (2 * days_per_month * hours_per_day) = 15/2 := by
  sorry

#eval (2400 : ℚ) / (2 * 20 * 8) -- This should evaluate to 7.5

end NUMINAMATH_CALUDE_megans_hourly_wage_l1520_152086


namespace NUMINAMATH_CALUDE_correct_number_of_small_boxes_l1520_152091

-- Define the number of chocolate bars in each small box
def chocolates_per_small_box : ℕ := 26

-- Define the total number of chocolate bars in the large box
def total_chocolates : ℕ := 442

-- Define the number of small boxes in the large box
def num_small_boxes : ℕ := total_chocolates / chocolates_per_small_box

-- Theorem statement
theorem correct_number_of_small_boxes : num_small_boxes = 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_small_boxes_l1520_152091


namespace NUMINAMATH_CALUDE_equation_solutions_l1520_152080

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (1/2 * (x₁ - 3)^2 = 18 ∧ x₁ = 9) ∧
                (1/2 * (x₂ - 3)^2 = 18 ∧ x₂ = -3)) ∧
  (∃ y₁ y₂ : ℝ, (y₁^2 + 6*y₁ = 5 ∧ y₁ = -3 + Real.sqrt 14) ∧
                (y₂^2 + 6*y₂ = 5 ∧ y₂ = -3 - Real.sqrt 14)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1520_152080


namespace NUMINAMATH_CALUDE_problem_solution_l1520_152062

theorem problem_solution (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 4 * a^2 - 8 * a - 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1520_152062


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1520_152015

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 + x + b

-- Define the solution set of the original inequality
def solution_set (a b : ℝ) : Set ℝ := {x | x < -2 ∨ x > 1}

-- Define the new quadratic function
def g (c x : ℝ) := x^2 - (c - 2) * x - 2 * c

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ a b : ℝ, (∀ x : ℝ, f a b x > 0 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = -2) ∧
  (∀ c : ℝ, 
    (c = -2 → {x : ℝ | g c x < 0} = ∅) ∧
    (c > -2 → {x : ℝ | g c x < 0} = Set.Ioo (-2) c) ∧
    (c < -2 → {x : ℝ | g c x < 0} = Set.Ioo c (-2))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1520_152015


namespace NUMINAMATH_CALUDE_sisyphus_stones_l1520_152022

/-- The minimum number of operations to move n stones to the rightmost square -/
def minOperations (n : ℕ) : ℕ :=
  (Finset.range n).sum fun k => (n + k) / (k + 1)

/-- The problem statement -/
theorem sisyphus_stones (n : ℕ) (h : n > 0) :
  ∀ (ops : ℕ), 
    (∃ (final_state : Fin (n + 1) → ℕ), 
      (final_state (Fin.last n) = n) ∧ 
      (∀ i < n, final_state i = 0) ∧
      (∃ (initial_state : Fin (n + 1) → ℕ),
        (initial_state 0 = n) ∧
        (∀ i > 0, initial_state i = 0) ∧
        (∃ (moves : Fin ops → Fin (n + 1) × Fin (n + 1)),
          (∀ m, (moves m).1 < (moves m).2) ∧
          (∀ m, (moves m).2.val - (moves m).1.val ≤ initial_state (moves m).1)))) →
    ops ≥ minOperations n :=
by sorry

end NUMINAMATH_CALUDE_sisyphus_stones_l1520_152022


namespace NUMINAMATH_CALUDE_sum_of_digits_1729_base_8_l1520_152047

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits in a list of natural numbers -/
def sumDigits (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

/-- The sum of digits of 1729 in base 8 is equal to 7 -/
theorem sum_of_digits_1729_base_8 :
  sumDigits (toBase8 1729) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1729_base_8_l1520_152047


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1520_152017

/-- Given single-digit integers a and b satisfying certain conditions, prove their sum is 7 --/
theorem digit_sum_problem (a b : ℕ) : 
  a < 10 → b < 10 → (4 * a) % 10 = 6 → 3 * b * 10 + 4 * a = 116 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1520_152017


namespace NUMINAMATH_CALUDE_investment_ratio_l1520_152095

def investment_x : ℤ := 5000
def investment_y : ℤ := 15000

theorem investment_ratio :
  (investment_x : ℚ) / investment_y = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_investment_ratio_l1520_152095


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l1520_152074

def cubic_equation (x : Int) : Int :=
  x^3 - 4*x^2 - 11*x + 24

def is_root (x : Int) : Prop :=
  cubic_equation x = 0

theorem integer_roots_of_cubic :
  ∀ x : Int, is_root x ↔ x = -4 ∨ x = 3 ∨ x = 8 := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l1520_152074


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l1520_152063

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (1 + Complex.I) = b * Complex.I) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l1520_152063


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l1520_152039

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) : 
  r - p = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l1520_152039
