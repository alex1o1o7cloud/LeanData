import Mathlib

namespace right_triangle_from_sine_condition_l2305_230516

theorem right_triangle_from_sine_condition (A B C : Real) (h1 : 0 < A) (h2 : A < π/2) 
  (h3 : 0 < B) (h4 : B < π/2) (h5 : A + B + C = π) 
  (h6 : Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin (A + B)) : 
  C = π/2 := by
sorry

end right_triangle_from_sine_condition_l2305_230516


namespace rice_quantity_calculation_rice_quantity_proof_l2305_230563

/-- Calculates the final quantity of rice that can be bought given initial conditions and price changes -/
theorem rice_quantity_calculation (initial_quantity : ℝ) 
  (first_price_reduction : ℝ) (second_price_reduction : ℝ) 
  (kg_to_pound_ratio : ℝ) (currency_exchange_rate : ℝ) : ℝ :=
  let after_first_reduction := initial_quantity * (1 / (1 - first_price_reduction))
  let after_second_reduction := after_first_reduction * (1 / (1 - second_price_reduction))
  let in_pounds := after_second_reduction * kg_to_pound_ratio
  let after_exchange_rate := in_pounds * (1 + currency_exchange_rate)
  let final_quantity := after_exchange_rate / kg_to_pound_ratio
  final_quantity

/-- The final quantity of rice that can be bought is approximately 29.17 kg -/
theorem rice_quantity_proof :
  ∃ ε > 0, |rice_quantity_calculation 20 0.2 0.1 2.2 0.05 - 29.17| < ε :=
by
  sorry

end rice_quantity_calculation_rice_quantity_proof_l2305_230563


namespace right_triangle_area_and_height_l2305_230543

theorem right_triangle_area_and_height :
  let a : ℝ := 9
  let b : ℝ := 40
  let c : ℝ := 41
  -- Condition: it's a right triangle
  a ^ 2 + b ^ 2 = c ^ 2 →
  -- Prove the area
  (1 / 2 : ℝ) * a * b = 180 ∧
  -- Prove the height
  (2 * ((1 / 2 : ℝ) * a * b)) / c = 360 / 41 := by
sorry

end right_triangle_area_and_height_l2305_230543


namespace distance_between_blue_lights_l2305_230593

/-- Represents the pattern of lights -/
inductive LightColor
| Blue
| Yellow

/-- Represents the recurring pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Blue, LightColor.Blue, LightColor.Blue,
   LightColor.Yellow, LightColor.Yellow, LightColor.Yellow, LightColor.Yellow]

/-- The spacing between lights in inches -/
def lightSpacing : ℕ := 7

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- Calculates the position of the nth blue light in the sequence -/
def bluePosition (n : ℕ) : ℕ :=
  ((n - 1) / 3) * lightPattern.length + ((n - 1) % 3) + 1

/-- The main theorem to prove -/
theorem distance_between_blue_lights :
  (bluePosition 25 - bluePosition 4) * lightSpacing / inchesPerFoot = 28 := by
  sorry

end distance_between_blue_lights_l2305_230593


namespace base_prime_representation_of_540_l2305_230511

/-- Base prime representation of a natural number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list represents a valid base prime representation -/
def IsValidBasePrimeRepresentation (l : List ℕ) : Prop :=
  sorry

theorem base_prime_representation_of_540 :
  let representation := [1, 3, 1]
  540 = 2^1 * 3^3 * 5^1 →
  IsValidBasePrimeRepresentation representation ∧
  BasePrimeRepresentation 540 = representation :=
by sorry

end base_prime_representation_of_540_l2305_230511


namespace v_2004_equals_1_l2305_230594

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- Default case for completeness

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n + 1)

-- Theorem statement
theorem v_2004_equals_1 : v 2004 = 1 := by
  sorry

end v_2004_equals_1_l2305_230594


namespace clock_strike_time_l2305_230502

/-- If a clock strikes 12 in 33 seconds, it will strike 6 in 15 seconds -/
theorem clock_strike_time (strike_12_time : ℕ) (strike_6_time : ℕ) : 
  strike_12_time = 33 → strike_6_time = 15 := by
  sorry

#check clock_strike_time

end clock_strike_time_l2305_230502


namespace abs_increasing_on_unit_interval_l2305_230510

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_increasing_on_unit_interval : 
  ∀ x y : ℝ, 0 < x → x < y → y < 1 → f x < f y := by
  sorry

end abs_increasing_on_unit_interval_l2305_230510


namespace min_distance_midpoint_to_origin_l2305_230561

/-- Given two parallel lines in a 2D plane, this theorem states that 
    the minimum distance from the midpoint of any line segment 
    connecting points on these lines to the origin is 3√2. -/
theorem min_distance_midpoint_to_origin 
  (l₁ l₂ : Set (ℝ × ℝ)) 
  (h₁ : l₁ = {(x, y) | x + y = 7})
  (h₂ : l₂ = {(x, y) | x + y = 5})
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A ∈ l₁) (hB : B ∈ l₂) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧ 
    ∀ (A' : ℝ × ℝ) (B' : ℝ × ℝ), A' ∈ l₁ → B' ∈ l₂ → 
      let M' := ((A'.1 + B'.1) / 2, (A'.2 + B'.2) / 2)
      d ≤ Real.sqrt (M'.1^2 + M'.2^2) :=
by sorry

end min_distance_midpoint_to_origin_l2305_230561


namespace log_properties_l2305_230584

-- Define the logarithm function
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_properties (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (f b b = 1) ∧
  (f b 1 = 0) ∧
  (∀ x, 0 < x → x < b → f b x < 1) ∧
  (∀ x, x > b → f b x > 1) :=
by sorry

end log_properties_l2305_230584


namespace unique_divisible_triple_l2305_230569

theorem unique_divisible_triple :
  ∃! (x y z : ℕ), 
    0 < x ∧ x < y ∧ y < z ∧
    Nat.gcd x (Nat.gcd y z) = 1 ∧
    (x + y) % z = 0 ∧
    (y + z) % x = 0 ∧
    (z + x) % y = 0 ∧
    x = 1 ∧ y = 2 ∧ z = 3 := by
  sorry

end unique_divisible_triple_l2305_230569


namespace girls_in_class_l2305_230549

theorem girls_in_class (total_students : ℕ) (girls : ℕ) (boys : ℕ) :
  total_students = 250 →
  girls + boys = total_students →
  girls = 2 * (total_students - (girls + boys - girls)) →
  girls = 100 :=
by
  sorry

end girls_in_class_l2305_230549


namespace water_tank_capacity_l2305_230579

theorem water_tank_capacity (c : ℚ) : 
  (1 / 5 : ℚ) * c + 5 = (2 / 7 : ℚ) * c → c = 35 / 3 := by
  sorry

end water_tank_capacity_l2305_230579


namespace sin_1320_degrees_l2305_230578

theorem sin_1320_degrees : Real.sin (1320 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_1320_degrees_l2305_230578


namespace lawsuit_probability_comparison_l2305_230534

def probability_lawsuit1_win : ℝ := 0.3
def probability_lawsuit2_win : ℝ := 0.5
def probability_lawsuit3_win : ℝ := 0.4

def probability_lawsuit1_lose : ℝ := 1 - probability_lawsuit1_win
def probability_lawsuit2_lose : ℝ := 1 - probability_lawsuit2_win
def probability_lawsuit3_lose : ℝ := 1 - probability_lawsuit3_win

def probability_win_all : ℝ := probability_lawsuit1_win * probability_lawsuit2_win * probability_lawsuit3_win
def probability_lose_all : ℝ := probability_lawsuit1_lose * probability_lawsuit2_lose * probability_lawsuit3_lose

theorem lawsuit_probability_comparison :
  (probability_lose_all - probability_win_all) / probability_win_all * 100 = 250 := by
sorry

end lawsuit_probability_comparison_l2305_230534


namespace logarithm_expression_equality_l2305_230554

theorem logarithm_expression_equality : 
  (Real.log 160 / Real.log 4) / (Real.log 4 / Real.log 80) - 
  (Real.log 40 / Real.log 4) / (Real.log 4 / Real.log 10) = 
  4.25 + (3/2) * (Real.log 5 / Real.log 4) := by sorry

end logarithm_expression_equality_l2305_230554


namespace binomial_consecutive_ratio_l2305_230539

theorem binomial_consecutive_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 :=
by sorry

end binomial_consecutive_ratio_l2305_230539


namespace quadratic_function_properties_g_zero_for_negative_m_g_max_abs_value_case1_g_max_abs_value_case2_l2305_230545

def f (x : ℝ) := (x + 1)^2 - 4

def g (m : ℝ) (x : ℝ) := m * f x + 1

theorem quadratic_function_properties :
  (∀ x, f x ≥ -4) ∧ f (-2) = -3 ∧ f 0 = -3 := by sorry

theorem g_zero_for_negative_m (m : ℝ) (hm : m < 0) :
  ∃! x, x ≤ 1 ∧ g m x = 0 := by sorry

theorem g_max_abs_value_case1 (m : ℝ) (hm : 0 < m ∧ m ≤ 8/7) :
  ∀ x ∈ [-3, 3/2], |g m x| ≤ 9/4 * m + 1 := by sorry

theorem g_max_abs_value_case2 (m : ℝ) (hm : m > 8/7) :
  ∀ x ∈ [-3, 3/2], |g m x| ≤ 4 * m - 1 := by sorry

end quadratic_function_properties_g_zero_for_negative_m_g_max_abs_value_case1_g_max_abs_value_case2_l2305_230545


namespace line_tangent_to_circle_l2305_230521

/-- The line 5x + 12y + a = 0 is tangent to the circle (x-1)^2 + y^2 = 1 if and only if a = 8 or a = -18 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (5 * x + 12 * y + a = 0) → ((x - 1)^2 + y^2 = 1)) ↔ (a = 8 ∨ a = -18) := by
sorry

end line_tangent_to_circle_l2305_230521


namespace cat_ratio_l2305_230503

theorem cat_ratio (melanie_cats jacob_cats : ℕ) 
  (melanie_twice_annie : melanie_cats = 2 * (melanie_cats / 2))
  (jacob_has_90 : jacob_cats = 90)
  (melanie_has_60 : melanie_cats = 60) :
  (melanie_cats / 2) / jacob_cats = 1 / 3 := by
  sorry

end cat_ratio_l2305_230503


namespace kennel_total_is_45_l2305_230546

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure KennelDogs where
  long_fur : ℕ
  brown : ℕ
  neither : ℕ
  long_fur_and_brown : ℕ

/-- Calculates the total number of dogs in the kennel. -/
def total_dogs (k : KennelDogs) : ℕ :=
  k.long_fur + k.brown - k.long_fur_and_brown + k.neither

/-- Theorem stating the total number of dogs in the kennel is 45. -/
theorem kennel_total_is_45 (k : KennelDogs) 
    (h1 : k.long_fur = 29)
    (h2 : k.brown = 17)
    (h3 : k.neither = 8)
    (h4 : k.long_fur_and_brown = 9) :
  total_dogs k = 45 := by
  sorry

end kennel_total_is_45_l2305_230546


namespace mistaken_calculation_l2305_230575

theorem mistaken_calculation (x : ℕ) : 
  (x / 16 = 8) → (x % 16 = 4) → (x * 16 + 8 = 2120) :=
by
  sorry

end mistaken_calculation_l2305_230575


namespace alcohol_mixture_proof_l2305_230523

/-- Proves that adding 750 mL of 30% alcohol solution to 250 mL of 10% alcohol solution
    results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 250
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 750
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  
  total_alcohol / total_volume = target_concentration := by sorry

end alcohol_mixture_proof_l2305_230523


namespace delacroix_band_size_l2305_230573

theorem delacroix_band_size (n : ℕ) : 
  (∃ k : ℕ, 30 * n = 28 * k + 6) →
  30 * n < 1200 →
  (∀ m : ℕ, (∃ j : ℕ, 30 * m = 28 * j + 6) → 30 * m < 1200 → 30 * m ≤ 30 * n) →
  30 * n = 930 :=
by sorry

end delacroix_band_size_l2305_230573


namespace pie_chart_central_angle_l2305_230536

theorem pie_chart_central_angle 
  (total_data : ℕ) 
  (group_frequency : ℕ) 
  (h1 : total_data = 60) 
  (h2 : group_frequency = 15) : 
  (group_frequency : ℝ) / (total_data : ℝ) * 360 = 90 := by
sorry

end pie_chart_central_angle_l2305_230536


namespace tangent_line_intersection_l2305_230556

/-- Given two circles with centers at (0, 0) and (17, 0) and radii 3 and 8 respectively,
    the x-coordinate of the point where a line tangent to both circles intersects the x-axis
    (to the right of the origin) is equal to 51/11. -/
theorem tangent_line_intersection (x : ℝ) : x > 0 →
  (x^2 = 3^2 + x^2) ∧ ((17 - x)^2 = 8^2 + x^2) → x = 51 / 11 := by
  sorry

end tangent_line_intersection_l2305_230556


namespace loan_split_l2305_230598

/-- Given a total sum of 2691 Rs. split into two parts, if the interest on the first part
    for 8 years at 3% per annum is equal to the interest on the second part for 3 years
    at 5% per annum, then the second part of the sum is 1656 Rs. -/
theorem loan_split (x : ℚ) : 
  (x ≥ 0) →
  (2691 - x ≥ 0) →
  (x * 3 * 8 / 100 = (2691 - x) * 5 * 3 / 100) →
  (2691 - x = 1656) :=
by sorry

end loan_split_l2305_230598


namespace abs_inequality_solution_l2305_230520

-- Define the solution set for |x+1| < 5
def solution_set : Set ℝ := {x : ℝ | |x + 1| < 5}

-- Define the open interval (-6, 4)
def open_interval : Set ℝ := Set.Ioo (-6) 4

-- Theorem stating that the solution set is equal to the open interval
theorem abs_inequality_solution : solution_set = open_interval := by sorry

end abs_inequality_solution_l2305_230520


namespace largest_circle_tangent_to_line_l2305_230509

/-- The largest circle with center (0,2) that is tangent to the line mx - y - 3m - 1 = 0 -/
theorem largest_circle_tangent_to_line (m : ℝ) :
  ∃! (r : ℝ), r > 0 ∧
    (∀ (x y : ℝ), x^2 + (y - 2)^2 = r^2 →
      ∃ (x₀ y₀ : ℝ), x₀^2 + (y₀ - 2)^2 = r^2 ∧
        m * x₀ - y₀ - 3 * m - 1 = 0) ∧
    (∀ (r' : ℝ), r' > r →
      ¬∃ (x y : ℝ), x^2 + (y - 2)^2 = r'^2 ∧
        m * x - y - 3 * m - 1 = 0) ∧
    r^2 = 18 :=
by sorry

end largest_circle_tangent_to_line_l2305_230509


namespace quadratic_single_solution_l2305_230519

theorem quadratic_single_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, 3 * x^2 + b * x + 10 = 0) →
  (∃ x, 3 * x^2 + b * x + 10 = 0 ∧ x = -Real.sqrt 30 / 3) :=
by sorry

end quadratic_single_solution_l2305_230519


namespace franks_chips_purchase_franks_chips_purchase_correct_l2305_230591

theorem franks_chips_purchase (chocolate_bars : ℕ) (chocolate_price : ℕ) 
  (chip_price : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  let total_spent := paid - change
  let chocolate_cost := chocolate_bars * chocolate_price
  let chips_cost := total_spent - chocolate_cost
  chips_cost / chip_price

#check franks_chips_purchase 5 2 3 20 4 = 2

theorem franks_chips_purchase_correct : franks_chips_purchase 5 2 3 20 4 = 2 := by
  sorry

end franks_chips_purchase_franks_chips_purchase_correct_l2305_230591


namespace count_valid_voucher_codes_l2305_230507

/-- Represents a voucher code -/
structure VoucherCode where
  first : Char
  second : Nat
  third : Nat
  fourth : Nat

/-- Checks if a character is a valid first character -/
def isValidFirstChar (c : Char) : Bool :=
  c = 'V' || c = 'X' || c = 'P'

/-- Checks if a voucher code is valid -/
def isValidVoucherCode (code : VoucherCode) : Bool :=
  isValidFirstChar code.first &&
  code.second < 10 &&
  code.third < 10 &&
  code.second ≠ code.third &&
  code.fourth = (code.second + code.third) % 10

/-- The set of all valid voucher codes -/
def validVoucherCodes : Finset VoucherCode :=
  sorry

/-- The number of valid voucher codes is 270 -/
theorem count_valid_voucher_codes :
  Finset.card validVoucherCodes = 270 :=
sorry

end count_valid_voucher_codes_l2305_230507


namespace correct_city_determination_l2305_230548

/-- Represents the two cities on Mars -/
inductive City
| MarsPolis
| MarsCity

/-- Represents the possible answers to a question -/
inductive Answer
| Yes
| No

/-- A Martian's response to the question "Do you live here?" -/
def martianResponse (city : City) (martianOrigin : City) : Answer :=
  match city, martianOrigin with
  | City.MarsPolis, _ => Answer.Yes
  | City.MarsCity, _ => Answer.No

/-- Determines the city based on the Martian's response -/
def determineCity (response : Answer) : City :=
  match response with
  | Answer.Yes => City.MarsPolis
  | Answer.No => City.MarsCity

/-- Theorem stating that asking "Do you live here?" always determines the correct city -/
theorem correct_city_determination (actualCity : City) (martianOrigin : City) :
  determineCity (martianResponse actualCity martianOrigin) = actualCity :=
by sorry

end correct_city_determination_l2305_230548


namespace angle_relation_l2305_230541

-- Define the structure for a point in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the structure for a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define the structure for a quadrilateral
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

-- Define a function to calculate the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Define a function to check if two triangles are anti-similar
def antiSimilar (t1 t2 : Triangle) : Prop := sorry

-- Define a function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a function to find the intersection of perpendicular bisectors
def perpendicularBisectorIntersection (A B C D : Point) : Point := sorry

-- Main theorem
theorem angle_relation 
  (A B C D X : Point)
  (Y : Point := perpendicularBisectorIntersection A B C D)
  (h1 : antiSimilar ⟨B, X, C⟩ ⟨A, X, D⟩)
  (h2 : isConvex ⟨A, B, C, D⟩)
  (h3 : angle A D X = angle B C X)
  (h4 : angle D A X = angle C B X)
  (h5 : angle A D X < π/2)
  (h6 : angle D A X < π/2)
  (h7 : angle B C X < π/2)
  (h8 : angle C B X < π/2) :
  angle A Y B = 2 * angle A D X := by
  sorry

end angle_relation_l2305_230541


namespace shop_c_tv_sets_l2305_230583

theorem shop_c_tv_sets (a b c d e : ℕ) : 
  a = 20 ∧ b = 30 ∧ d = 80 ∧ e = 50 ∧ 
  (a + b + c + d + e) / 5 = 48 →
  c = 60 := by
sorry

end shop_c_tv_sets_l2305_230583


namespace geometric_sequence_sum_l2305_230524

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  (a 1 + a 2 + a 3 = 8) →                    -- first condition
  (a 4 + a 5 + a 6 = -4) →                   -- second condition
  (a 7 + a 8 + a 9 = 2) :=                   -- conclusion to prove
by sorry

end geometric_sequence_sum_l2305_230524


namespace hot_dog_purchase_l2305_230500

theorem hot_dog_purchase (cost_per_hot_dog : ℕ) (total_paid : ℕ) (h1 : cost_per_hot_dog = 50) (h2 : total_paid = 300) :
  total_paid / cost_per_hot_dog = 6 := by
  sorry

end hot_dog_purchase_l2305_230500


namespace sum_product_inequality_l2305_230537

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + a * c ≤ 0 := by
  sorry

end sum_product_inequality_l2305_230537


namespace no_solution_condition_l2305_230565

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (m * x) / (x - 3) ≠ 3 / (x - 3)) ↔ (m = 1 ∨ m = 0) := by
  sorry

end no_solution_condition_l2305_230565


namespace solve_sandwich_cost_l2305_230574

def sandwich_cost_problem (total_cost soda_cost : ℚ) : Prop :=
  let num_sandwiches : ℕ := 2
  let num_sodas : ℕ := 4
  let sandwich_cost : ℚ := (total_cost - num_sodas * soda_cost) / num_sandwiches
  total_cost = 838/100 ∧ soda_cost = 87/100 → sandwich_cost = 245/100

theorem solve_sandwich_cost : 
  sandwich_cost_problem (838/100) (87/100) := by
  sorry

end solve_sandwich_cost_l2305_230574


namespace polar_to_cartesian_conversion_l2305_230550

/-- The polar to Cartesian conversion theorem for a specific curve -/
theorem polar_to_cartesian_conversion (ρ θ x y : ℝ) :
  (ρ * (Real.cos θ)^2 = 2 * Real.sin θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  x^2 = 2*y := by
  sorry

end polar_to_cartesian_conversion_l2305_230550


namespace sqrt_2700_minus_37_cube_l2305_230590

theorem sqrt_2700_minus_37_cube (a b : ℕ+) :
  (Real.sqrt 2700 - 37 : ℝ) = (Real.sqrt a.val - b.val)^3 →
  a.val + b.val = 13 := by
  sorry

end sqrt_2700_minus_37_cube_l2305_230590


namespace gcd_of_sums_of_squares_l2305_230553

theorem gcd_of_sums_of_squares : Nat.gcd (130^2 + 240^2 + 350^2) (131^2 + 241^2 + 351^2) = 3 := by
  sorry

end gcd_of_sums_of_squares_l2305_230553


namespace system_sample_fourth_number_l2305_230547

/-- Represents a system sampling of employees -/
structure SystemSample where
  total : Nat
  sample_size : Nat
  sample : Finset Nat

/-- Checks if a given set of numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (s : Finset Nat) : Prop :=
  ∃ (a d : Nat), ∀ (x : Nat), x ∈ s → ∃ (k : Nat), x = a + k * d

/-- The main theorem about the system sampling -/
theorem system_sample_fourth_number
  (s : SystemSample)
  (h_total : s.total = 52)
  (h_size : s.sample_size = 4)
  (h_contains : {6, 32, 45} ⊆ s.sample)
  (h_arithmetic : is_arithmetic_sequence s.sample) :
  19 ∈ s.sample :=
sorry

end system_sample_fourth_number_l2305_230547


namespace sequence_sum_l2305_230570

-- Define the sequence type
def Sequence := Fin 10 → ℝ

-- Define the property of consecutive terms summing to 20
def ConsecutiveSum (s : Sequence) : Prop :=
  ∀ i : Fin 8, s i + s (i + 1) + s (i + 2) = 20

-- Define the theorem
theorem sequence_sum (s : Sequence) 
  (h1 : ConsecutiveSum s) 
  (h2 : s 4 = 8) : 
  s 0 + s 9 = 8 := by
  sorry


end sequence_sum_l2305_230570


namespace x_intercept_of_line_l2305_230504

/-- The x-intercept of the line 4x + 6y = 24 is (6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 6 * y = 24 → y = 0 → x = 6 := by
  sorry

end x_intercept_of_line_l2305_230504


namespace monkey_fruit_ratio_l2305_230577

theorem monkey_fruit_ratio (a b x y z : ℝ) : 
  a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
  x = 1.4 * a →
  y + 0.25 * y = 1.25 * y →
  b = 2 * z →
  a + b = x + y →
  a + b = z + 1.4 * a →
  a / b = 1 / 2 := by
sorry


end monkey_fruit_ratio_l2305_230577


namespace trapezoid_parallel_line_length_l2305_230571

/-- Represents a trapezoid with bases of lengths a and b -/
structure Trapezoid (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- 
Given a trapezoid with bases of lengths a and b, 
if a line parallel to the bases divides the trapezoid into two equal-area trapezoids,
then the length of the segment of this line between the non-parallel sides 
is sqrt((a^2 + b^2)/2).
-/
theorem trapezoid_parallel_line_length 
  (a b : ℝ) (trap : Trapezoid a b) : 
  ∃ (x : ℝ), x > 0 ∧ x = Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end trapezoid_parallel_line_length_l2305_230571


namespace complement_A_B_when_a_is_one_A_intersection_B_equals_A_l2305_230512

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem for part (1)
theorem complement_A_B_when_a_is_one :
  (Set.univ \ A 1) ∩ B = {x | -1 < x ∧ x ≤ -1/2} ∪ {2} := by sorry

-- Theorem for part (2)
theorem A_intersection_B_equals_A (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 ∨ a ≥ 2 := by sorry

end complement_A_B_when_a_is_one_A_intersection_B_equals_A_l2305_230512


namespace vanya_number_theorem_l2305_230506

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the property that a number satisfies the condition
def satisfiesCondition (n : ℕ) : Prop := n + sumOfDigits n = 2021

-- Theorem statement
theorem vanya_number_theorem : 
  (∀ n : ℕ, satisfiesCondition n ↔ (n = 2014 ∨ n = 1996)) := by sorry

end vanya_number_theorem_l2305_230506


namespace sum_of_matching_indices_l2305_230586

def sequence_length : ℕ := 1011

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_odds (n : ℕ) : ℕ := ((n + 1) / 2) ^ 2

theorem sum_of_matching_indices :
  sum_of_odds sequence_length = 256036 :=
sorry

end sum_of_matching_indices_l2305_230586


namespace pen_pencil_ratio_proof_l2305_230582

def number_of_pencils : ℕ := 48
def pencil_pen_difference : ℕ := 8

def number_of_pens : ℕ := number_of_pencils - pencil_pen_difference

def pen_pencil_ratio : ℚ × ℚ := (5, 6)

theorem pen_pencil_ratio_proof :
  (number_of_pens : ℚ) / (number_of_pencils : ℚ) = pen_pencil_ratio.1 / pen_pencil_ratio.2 :=
by sorry

end pen_pencil_ratio_proof_l2305_230582


namespace number_problem_l2305_230540

theorem number_problem (x : ℝ) : x^2 + 100 = (x - 20)^2 → x = 7.5 := by
  sorry

end number_problem_l2305_230540


namespace total_arrangements_l2305_230557

def news_reports : ℕ := 5
def interviews : ℕ := 4
def total_programs : ℕ := 5
def min_news : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

def arrangements (news interviews total min_news : ℕ) : ℕ :=
  (choose news min_news * choose interviews (total - min_news) * permute total total) +
  (choose news (min_news + 1) * choose interviews (total - min_news - 1) * permute total total) +
  (choose news total * permute total total)

theorem total_arrangements :
  arrangements news_reports interviews total_programs min_news = 9720 := by
  sorry

end total_arrangements_l2305_230557


namespace bicycle_sale_price_l2305_230522

/-- Given a cost price and two consecutive percentage markups, 
    calculate the final selling price. -/
def final_price (cost_price : ℚ) (markup_percent : ℚ) : ℚ :=
  let first_sale := cost_price * (1 + markup_percent / 100)
  first_sale * (1 + markup_percent / 100)

/-- Theorem: The final selling price of a bicycle with an initial cost of 144,
    after two consecutive 25% markups, is 225. -/
theorem bicycle_sale_price : final_price 144 25 = 225 := by
  sorry

#eval final_price 144 25

end bicycle_sale_price_l2305_230522


namespace diagonal_length_in_special_quadrilateral_l2305_230531

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral ABCD with diagonals intersecting at E -/
structure Quadrilateral :=
  (A B C D E : Point)

/-- The length of a line segment between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Main theorem -/
theorem diagonal_length_in_special_quadrilateral 
  (ABCD : Quadrilateral) 
  (h1 : distance ABCD.A ABCD.B = 10)
  (h2 : distance ABCD.C ABCD.D = 15)
  (h3 : distance ABCD.A ABCD.C = 18)
  (h4 : triangleArea ABCD.A ABCD.E ABCD.D = triangleArea ABCD.B ABCD.E ABCD.C) :
  distance ABCD.A ABCD.E = 7.2 := by sorry

end diagonal_length_in_special_quadrilateral_l2305_230531


namespace total_amount_distributed_l2305_230518

/-- The total amount distributed when an amount of money is equally divided among a group of people. -/
def total_amount (num_people : ℕ) (amount_per_person : ℕ) : ℕ :=
  num_people * amount_per_person

/-- Theorem stating that the total amount distributed is 42900 when 22 people each receive 1950. -/
theorem total_amount_distributed : total_amount 22 1950 = 42900 := by
  sorry

end total_amount_distributed_l2305_230518


namespace line_through_point_parallel_to_line_l2305_230528

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def lies_on (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The problem statement -/
theorem line_through_point_parallel_to_line :
  ∃ (l : Line),
    lies_on (-1) 2 l ∧
    parallel l { a := 2, b := -3, c := 4 } ∧
    l = { a := 2, b := -3, c := 8 } := by
  sorry

end line_through_point_parallel_to_line_l2305_230528


namespace sum_of_five_variables_l2305_230558

theorem sum_of_five_variables (a b c d e : ℝ) 
  (eq1 : a + b = 16)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3)
  (eq4 : d + e = 5)
  (eq5 : e + a = 7) :
  a + b + c + d + e = 20 := by
  sorry

end sum_of_five_variables_l2305_230558


namespace point_on_segment_l2305_230551

-- Define the space we're working in (Euclidean plane)
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

-- Define points A, B, C, and M
variable (A B C M : E)

-- Define the condition that for any M, either MA ≤ MB or MA ≤ MC
def condition (A B C : E) : Prop :=
  ∀ M : E, ‖M - A‖ ≤ ‖M - B‖ ∨ ‖M - A‖ ≤ ‖M - C‖

-- Define what it means for A to lie on the segment BC
def lies_on_segment (A B C : E) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (1 - t) • B + t • C

-- State the theorem
theorem point_on_segment (A B C : E) :
  condition A B C → lies_on_segment A B C :=
by
  sorry

end point_on_segment_l2305_230551


namespace min_guesses_theorem_l2305_230572

/-- The minimum number of guesses required to determine the leader's binary string -/
def minGuesses (n k : ℕ+) : ℕ :=
  if n = 2 * k then 2 else 1

/-- Theorem stating the minimum number of guesses required -/
theorem min_guesses_theorem (n k : ℕ+) (h : n > k) :
  minGuesses n k = 2 ↔ n = 2 * k :=
sorry

end min_guesses_theorem_l2305_230572


namespace intersection_of_M_and_N_l2305_230526

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N_l2305_230526


namespace units_digit_of_n_squared_plus_two_to_n_l2305_230527

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : 
  n = 2018^2 + 2^2018 → (n^2 + 2^n) % 10 = 5 := by
  sorry

end units_digit_of_n_squared_plus_two_to_n_l2305_230527


namespace prime_product_digital_sum_difference_l2305_230592

def digital_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digital_sum (n / 10)

theorem prime_product_digital_sum_difference 
  (p q r : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hpqr : p * q * r = 18 * 962) 
  (hdiff : p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ∃ (result : ℕ), digital_sum p + digital_sum q + digital_sum r - digital_sum (p * q * r) = result :=
sorry

end prime_product_digital_sum_difference_l2305_230592


namespace tournament_games_theorem_l2305_230525

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  single_elimination : Bool
  no_ties : Bool

/-- Calculates the number of games needed to determine a winner in a tournament. -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- The theorem stating that a single-elimination tournament with 23 teams and no ties requires 22 games to determine a winner. -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.single_elimination = true) 
  (h3 : t.no_ties = true) : 
  games_to_winner t = 22 := by
  sorry


end tournament_games_theorem_l2305_230525


namespace regular_hexagon_area_l2305_230501

/-- The area of a regular hexagon with vertices A at (0,0) and C at (8,2) is 34√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (8, 2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * AC^2
  let hexagon_area : ℝ := 2 * triangle_area
  hexagon_area = 34 * Real.sqrt 3 := by sorry

end regular_hexagon_area_l2305_230501


namespace sam_pages_sam_read_100_pages_l2305_230513

def minimum_assigned : ℕ := 25

def harrison_extra : ℕ := 10

def pam_extra : ℕ := 15

def sam_multiplier : ℕ := 2

theorem sam_pages : ℕ :=
  let harrison_pages := minimum_assigned + harrison_extra
  let pam_pages := harrison_pages + pam_extra
  sam_multiplier * pam_pages

theorem sam_read_100_pages : sam_pages = 100 := by
  sorry

end sam_pages_sam_read_100_pages_l2305_230513


namespace number_greater_than_three_l2305_230562

theorem number_greater_than_three (x : ℝ) : 7 * x - 15 > 2 * x → x > 3 := by
  sorry

end number_greater_than_three_l2305_230562


namespace scientific_notation_of_60000_l2305_230552

theorem scientific_notation_of_60000 : 60000 = 6 * (10 ^ 4) := by
  sorry

end scientific_notation_of_60000_l2305_230552


namespace coin_ratio_is_one_one_one_l2305_230532

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- Represents the value of a coin in rupees -/
def coinValue : CoinType → Rat
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

/-- Represents the number of coins of each type -/
def numCoins : CoinType → Nat
  | _ => 40

/-- The total value of all coins in the bag -/
def totalValue : Rat := 70

/-- Theorem stating that the ratio of coin counts is 1:1:1 -/
theorem coin_ratio_is_one_one_one :
  numCoins CoinType.OneRupee = numCoins CoinType.FiftyPaise ∧
  numCoins CoinType.OneRupee = numCoins CoinType.TwentyFivePaise ∧
  (numCoins CoinType.OneRupee : Rat) * coinValue CoinType.OneRupee +
  (numCoins CoinType.FiftyPaise : Rat) * coinValue CoinType.FiftyPaise +
  (numCoins CoinType.TwentyFivePaise : Rat) * coinValue CoinType.TwentyFivePaise = totalValue :=
by sorry


end coin_ratio_is_one_one_one_l2305_230532


namespace arithmetic_sequence_common_difference_l2305_230595

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence. -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a2 : a 2 = 14) 
  (h_a5 : a 5 = 5) : 
  common_difference a = -3 :=
sorry

end arithmetic_sequence_common_difference_l2305_230595


namespace unique_expression_value_l2305_230538

theorem unique_expression_value (m n : ℤ) : 
  (∃! z : ℤ, m * n + 13 * m + 13 * n - m^2 - n^2 = z ∧ 
   (∀ k l : ℤ, k * l + 13 * k + 13 * l - k^2 - l^2 = z → k = m ∧ l = n)) →
  m * n + 13 * m + 13 * n - m^2 - n^2 = 169 :=
by sorry

end unique_expression_value_l2305_230538


namespace cubic_equation_solution_l2305_230568

theorem cubic_equation_solution (a b c : ℂ) : 
  (∃ (x₁ x₂ x₃ : ℂ), x₁ = 1 ∧ x₂ = 1 - Complex.I ∧ x₃ = 1 + Complex.I ∧
    (∀ (x : ℂ), x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a + b - c = 3 := by
  sorry

end cubic_equation_solution_l2305_230568


namespace complement_intersection_A_B_l2305_230576

open Set Real

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem complement_intersection_A_B :
  (Aᶜ ∪ Bᶜ) = {x : ℝ | x ≠ 0} := by sorry

end complement_intersection_A_B_l2305_230576


namespace integer_count_in_sequence_l2305_230530

def arithmeticSequence (n : ℕ) : ℚ :=
  8505 / (5 ^ n)

def isInteger (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem integer_count_in_sequence :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → isInteger (arithmeticSequence n)) ∧
    ¬isInteger (arithmeticSequence k)) →
  (∃! (k : ℕ), k = 3 ∧
    (∀ (n : ℕ), n < k → isInteger (arithmeticSequence n)) ∧
    ¬isInteger (arithmeticSequence k)) :=
by sorry

end integer_count_in_sequence_l2305_230530


namespace largest_rational_l2305_230515

theorem largest_rational (a b c d : ℚ) : 
  a = -1 → b = 0 → c = -3 → d = (8 : ℚ) / 100 → 
  max a (max b (max c d)) = d := by
  sorry

end largest_rational_l2305_230515


namespace floor_identity_l2305_230588

theorem floor_identity (x : ℝ) : 
  ⌊(3+x)/6⌋ - ⌊(4+x)/6⌋ + ⌊(5+x)/6⌋ = ⌊(1+x)/2⌋ - ⌊(1+x)/3⌋ := by
  sorry

end floor_identity_l2305_230588


namespace income_ratio_l2305_230564

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  p1.income = 3500 ∧
  p1.savings = 1400 ∧
  p2.savings = 1400 ∧
  p1.expenditure * 2 = p2.expenditure * 3 ∧
  p1.income = p1.expenditure + p1.savings ∧
  p2.income = p2.expenditure + p2.savings

/-- The theorem to prove -/
theorem income_ratio (p1 p2 : Person) 
  (h : financialProblem p1 p2) : 
  p1.income * 4 = p2.income * 5 := by
  sorry


end income_ratio_l2305_230564


namespace ellipse_chord_slope_ellipse_chord_slope_at_4_2_l2305_230529

/-- The slope of a chord in an ellipse given its midpoint -/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 4) →         -- Midpoint x-coordinate is 4
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  (y₂ - y₁) / (x₂ - x₁) = -1/2  -- Slope of the chord
:= by sorry

/-- The main theorem stating the slope of the chord with midpoint (4, 2) -/
theorem ellipse_chord_slope_at_4_2 : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1) ∧ 
    ((x₁ + x₂) / 2 = 4) ∧ 
    ((y₁ + y₂) / 2 = 2) ∧ 
    (y₂ - y₁) / (x₂ - x₁) = -1/2
:= by sorry

end ellipse_chord_slope_ellipse_chord_slope_at_4_2_l2305_230529


namespace counterexample_exists_l2305_230597

theorem counterexample_exists : ∃ n : ℕ+, 
  ¬(Nat.Prime n.val) ∧ Nat.Prime (n.val - 2) ∧ n.val = 33 := by
  sorry

end counterexample_exists_l2305_230597


namespace min_representatives_per_table_l2305_230596

/-- Represents the number of representatives for each country -/
structure Representatives where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- The condition that country ratios are satisfied -/
def satisfies_ratios (r : Representatives) : Prop :=
  r.A = 2 * r.B ∧ r.A = 3 * r.C ∧ r.A = 4 * r.D

/-- The condition that each country is outnumbered by others at a table -/
def is_outnumbered (r : Representatives) (total : ℕ) : Prop :=
  r.A < r.B + r.C + r.D ∧
  r.B < r.A + r.C + r.D ∧
  r.C < r.A + r.B + r.D ∧
  r.D < r.A + r.B + r.C

/-- The main theorem stating the minimum number of representatives per table -/
theorem min_representatives_per_table (r : Representatives) 
  (h_ratios : satisfies_ratios r) : 
  (∃ (n : ℕ), n > 0 ∧ is_outnumbered r n ∧ 
    ∀ (m : ℕ), m > 0 ∧ is_outnumbered r m → n ≤ m) → 
  (∃ (n : ℕ), n > 0 ∧ is_outnumbered r n ∧ 
    ∀ (m : ℕ), m > 0 ∧ is_outnumbered r m → n ≤ m) ∧ n = 25 :=
sorry

end min_representatives_per_table_l2305_230596


namespace sum_parity_when_sum_of_squares_odd_l2305_230566

theorem sum_parity_when_sum_of_squares_odd (n m : ℤ) (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end sum_parity_when_sum_of_squares_odd_l2305_230566


namespace f_extrema_max_k_bound_l2305_230533

noncomputable section

def f (x : ℝ) : ℝ := x + x * Real.log x

theorem f_extrema :
  (∃ (x_min : ℝ), x_min = Real.exp (-2) ∧
    (∀ x > 0, f x ≥ f x_min) ∧
    f x_min = -Real.exp (-2)) ∧
  (∀ M : ℝ, ∃ x > 0, f x > M) :=
sorry

theorem max_k_bound :
  (∀ k : ℤ, (∀ x > 1, f x > k * (x - 1)) → k ≤ 3) ∧
  (∃ x > 1, f x > 3 * (x - 1)) :=
sorry

end f_extrema_max_k_bound_l2305_230533


namespace opposite_reciprocal_absolute_value_l2305_230580

theorem opposite_reciprocal_absolute_value (a b c d m : ℝ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (m = 3 ∨ m = -3) →  -- |m| = 3
  ((a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4) := by
sorry

end opposite_reciprocal_absolute_value_l2305_230580


namespace locus_of_vertex_C_l2305_230544

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def IsEquilateral (t : EquilateralTriangle) : Prop :=
  let d_AB := ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)^(1/2)
  let d_BC := ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)^(1/2)
  let d_CA := ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2)
  d_AB = d_BC ∧ d_BC = d_CA

-- Define the theorem
theorem locus_of_vertex_C (c : Circle) (t : EquilateralTriangle) :
  IsEquilateral t →
  PointOnCircle c t.A →
  PointOnCircle c t.B →
  ∃ c1 c2 : Circle,
    c1.center = c.center ∧
    c2.center = c.center ∧
    c1.radius = c.radius ∧
    c2.radius = c.radius ∧
    PointOnCircle c1 t.C ∨ PointOnCircle c2 t.C :=
by sorry

end locus_of_vertex_C_l2305_230544


namespace complex_subtraction_l2305_230585

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 2 + 3*I) (h₂ : z₂ = 3 + I) : 
  z₁ - z₂ = -1 + 2*I := by
  sorry

end complex_subtraction_l2305_230585


namespace polynomial_roots_in_arithmetic_progression_l2305_230555

theorem polynomial_roots_in_arithmetic_progression (j k : ℝ) : 
  (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    (∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r) ∧
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 900 = (x - a)*(x - b)*(x - c)*(x - d))) →
  j = -900 := by
sorry

end polynomial_roots_in_arithmetic_progression_l2305_230555


namespace polynomial_divisibility_l2305_230560

theorem polynomial_divisibility : ∀ (x : ℂ),
  (x^5 + x^4 + x^3 + x^2 + x + 1 = 0) →
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0) := by
sorry

end polynomial_divisibility_l2305_230560


namespace class_visual_conditions_most_suitable_l2305_230599

/-- Represents a survey method --/
inductive SurveyMethod
  | EnergyLamps
  | ClassVisualConditions
  | ProvinceInternetUsage
  | CanalFishTypes

/-- Defines what constitutes a comprehensive investigation --/
def isComprehensive (method : SurveyMethod) : Prop :=
  match method with
  | .ClassVisualConditions => true
  | _ => false

/-- Theorem stating that understanding the visual conditions of Class 803 
    is the most suitable method for a comprehensive investigation --/
theorem class_visual_conditions_most_suitable :
  ∀ (method : SurveyMethod), 
    isComprehensive method → method = SurveyMethod.ClassVisualConditions :=
by sorry

end class_visual_conditions_most_suitable_l2305_230599


namespace sparrow_percentage_among_non_eagles_l2305_230508

theorem sparrow_percentage_among_non_eagles (total percentage : ℝ)
  (robins eagles falcons sparrows : ℝ)
  (h1 : total = 100)
  (h2 : robins = 20)
  (h3 : eagles = 30)
  (h4 : falcons = 15)
  (h5 : sparrows = total - (robins + eagles + falcons))
  (h6 : percentage = (sparrows / (total - eagles)) * 100) :
  percentage = 50 := by
sorry

end sparrow_percentage_among_non_eagles_l2305_230508


namespace hannah_mugs_theorem_l2305_230559

def hannah_mugs (total_mugs : ℕ) (total_colors : ℕ) (yellow_mugs : ℕ) : Prop :=
  ∃ (red_mugs blue_mugs other_mugs : ℕ),
    total_mugs = red_mugs + blue_mugs + yellow_mugs + other_mugs ∧
    blue_mugs = 3 * red_mugs ∧
    red_mugs = yellow_mugs / 2 ∧
    other_mugs = 4

theorem hannah_mugs_theorem :
  hannah_mugs 40 4 12 :=
by sorry

end hannah_mugs_theorem_l2305_230559


namespace marks_percentage_raise_l2305_230542

/-- Calculates the percentage raise Mark received at his job -/
theorem marks_percentage_raise
  (original_hourly_rate : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (old_weekly_bills : ℚ)
  (new_weekly_expense : ℚ)
  (new_leftover_amount : ℚ)
  (h1 : original_hourly_rate = 40)
  (h2 : hours_per_day = 8)
  (h3 : days_per_week = 5)
  (h4 : old_weekly_bills = 600)
  (h5 : new_weekly_expense = 100)
  (h6 : new_leftover_amount = 980) :
  (new_leftover_amount + old_weekly_bills + new_weekly_expense - 
   (original_hourly_rate * hours_per_day * days_per_week)) / 
  (original_hourly_rate * hours_per_day * days_per_week) = 1/20 :=
by sorry

end marks_percentage_raise_l2305_230542


namespace calculate_interest_rate_l2305_230589

/-- Calculates the simple interest rate given loan amounts and repayment details -/
theorem calculate_interest_rate 
  (initial_loan : ℝ) 
  (additional_loan : ℝ) 
  (total_repayment : ℝ) 
  (initial_period : ℝ) 
  (total_period : ℝ)
  (h1 : initial_loan = 10000)
  (h2 : additional_loan = 12000)
  (h3 : total_repayment = 27160)
  (h4 : initial_period = 2)
  (h5 : total_period = 5) :
  ∃ r : ℝ, r = 6 ∧ 
    initial_loan * (1 + r / 100 * initial_period) + 
    (initial_loan + additional_loan) * (1 + r / 100 * (total_period - initial_period)) = 
    total_repayment :=
by sorry

end calculate_interest_rate_l2305_230589


namespace triangle_longest_side_l2305_230505

/-- Given a triangle with sides of lengths 7, x+4, and 2x+1, and a perimeter of 36,
    prove that the length of the longest side is 17. -/
theorem triangle_longest_side (x : ℝ) : 
  (7 : ℝ) + (x + 4) + (2*x + 1) = 36 → 
  max 7 (max (x + 4) (2*x + 1)) = 17 :=
by sorry

end triangle_longest_side_l2305_230505


namespace pastor_prayer_theorem_l2305_230514

/-- Represents the number of times Pastor Paul prays per day (except on Sundays) -/
def paul_prayers : ℕ := sorry

/-- Represents the number of times Pastor Bruce prays per day (except on Sundays) -/
def bruce_prayers : ℕ := sorry

/-- The total number of times Pastor Paul prays in a week -/
def paul_weekly_prayers : ℕ := 6 * paul_prayers + 2 * paul_prayers

/-- The total number of times Pastor Bruce prays in a week -/
def bruce_weekly_prayers : ℕ := 6 * (paul_prayers / 2) + 4 * paul_prayers

theorem pastor_prayer_theorem :
  paul_prayers = 20 ∧
  bruce_prayers = paul_prayers / 2 ∧
  paul_weekly_prayers = bruce_weekly_prayers + 20 := by
sorry

end pastor_prayer_theorem_l2305_230514


namespace new_person_age_l2305_230517

/-- Given a group of 10 persons where replacing a 45-year-old person with a new person
    decreases the average age by 3 years, the age of the new person is 15 years. -/
theorem new_person_age (initial_avg : ℝ) : 
  (10 * initial_avg - 45 + 15) / 10 = initial_avg - 3 := by
  sorry

#check new_person_age

end new_person_age_l2305_230517


namespace fraction_sum_equality_l2305_230567

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := by
  sorry

end fraction_sum_equality_l2305_230567


namespace f_max_min_l2305_230581

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- State the theorem
theorem f_max_min :
  (∃ (x : ℝ), f x = 5 ∧ ∀ (y : ℝ), f y ≤ 5) ∧
  (∃ (x : ℝ), f x = -27 ∧ ∀ (y : ℝ), f y ≥ -27) := by
  sorry

end f_max_min_l2305_230581


namespace correct_prices_l2305_230535

/-- Prices of items in a shopping scenario -/
def shopping_prices (total belt pants shirt shoes : ℝ) : Prop :=
  -- Total cost condition
  total = belt + pants + shirt + shoes ∧
  -- Pants price condition
  pants = belt - 2.93 ∧
  -- Shirt price condition
  shirt = 1.5 * pants ∧
  -- Shoes price condition
  shoes = 3 * shirt

/-- Theorem stating the correct prices for the shopping scenario -/
theorem correct_prices : 
  ∃ (belt pants shirt shoes : ℝ),
    shopping_prices 205.93 belt pants shirt shoes ∧ 
    belt = 28.305 ∧ 
    pants = 25.375 ∧ 
    shirt = 38.0625 ∧ 
    shoes = 114.1875 :=
by
  sorry

end correct_prices_l2305_230535


namespace inverse_variation_sqrt_l2305_230587

/-- Given that y varies inversely as √x, prove that when y = 2 for x = 4, then x = 1/4 when y = 8 -/
theorem inverse_variation_sqrt (k : ℝ) (h1 : k > 0) : 
  (∀ x y, x > 0 → y = k / Real.sqrt x) → 
  (2 = k / Real.sqrt 4) → 
  (8 = k / Real.sqrt (1/4)) := by
sorry

end inverse_variation_sqrt_l2305_230587
