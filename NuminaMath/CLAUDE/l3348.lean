import Mathlib

namespace NUMINAMATH_CALUDE_vector_projection_l3348_334858

/-- Given vectors a and b in ℝ², prove that the projection of a onto 2√3b is √65/5 -/
theorem vector_projection (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-4, 7)) :
  let proj := (a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2) * Real.sqrt 3 * 2
  proj = Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l3348_334858


namespace NUMINAMATH_CALUDE_log_sum_equals_four_l3348_334808

theorem log_sum_equals_four : Real.log 64 / Real.log 8 + Real.log 81 / Real.log 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_four_l3348_334808


namespace NUMINAMATH_CALUDE_buffy_stolen_apples_l3348_334824

theorem buffy_stolen_apples (initial_apples : ℕ) (fallen_apples : ℕ) (remaining_apples : ℕ) 
  (h1 : initial_apples = 79)
  (h2 : fallen_apples = 26)
  (h3 : remaining_apples = 8) :
  initial_apples - fallen_apples - remaining_apples = 45 :=
by sorry

end NUMINAMATH_CALUDE_buffy_stolen_apples_l3348_334824


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l3348_334895

theorem unique_root_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l3348_334895


namespace NUMINAMATH_CALUDE_system_solution_l3348_334804

theorem system_solution : ∃! (x y : ℝ), 
  (x + Real.sqrt (x + 2*y) - 2*y = 7/2) ∧ 
  (x^2 + x + 2*y - 4*y^2 = 27/2) ∧
  (x = 19/4) ∧ (y = 17/8) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3348_334804


namespace NUMINAMATH_CALUDE_subset_condition_l3348_334883

def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l3348_334883


namespace NUMINAMATH_CALUDE_complex_number_proof_l3348_334887

theorem complex_number_proof (z : ℂ) :
  (∃ (z₁ : ℝ), z₁ = (z / (1 + z^2)).re ∧ (z / (1 + z^2)).im = 0) ∧
  (∃ (z₂ : ℝ), z₂ = (z^2 / (1 + z)).re ∧ (z^2 / (1 + z)).im = 0) →
  z = -1/2 + (Complex.I * Real.sqrt 3) / 2 ∨ z = -1/2 - (Complex.I * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_proof_l3348_334887


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3348_334850

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 14

-- Define the two fixed points
def point1 : ℝ × ℝ := (0, 2)
def point2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ) (h : 0 < a ∧ 0 < b),
    ∀ (x y : ℝ), conic_equation x y ↔
      (x - (point1.1 + point2.1)/2)^2/a^2 + (y - (point1.2 + point2.2)/2)^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3348_334850


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l3348_334813

-- Define the ratio of central angles
def angle_ratio : ℚ := 3 / 4

-- Define a function to calculate the volume ratio given the angle ratio
def volume_ratio (r : ℚ) : ℚ := r^2

-- Theorem statement
theorem cone_volume_ratio :
  volume_ratio angle_ratio = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_l3348_334813


namespace NUMINAMATH_CALUDE_vendor_profit_calculation_l3348_334876

/-- Calculates the profit for a vendor selling apples and oranges --/
def vendor_profit (apple_buy_price : ℚ) (apple_sell_price : ℚ) 
                  (orange_buy_price : ℚ) (orange_sell_price : ℚ) 
                  (apples_sold : ℕ) (oranges_sold : ℕ) : ℚ :=
  let apple_profit := apple_sell_price - apple_buy_price
  let orange_profit := orange_sell_price - orange_buy_price
  apple_profit * apples_sold + orange_profit * oranges_sold

theorem vendor_profit_calculation :
  let apple_buy_price : ℚ := 3 / 2  -- $3 for 2 apples
  let apple_sell_price : ℚ := 2     -- $10 for 5 apples, so $2 each
  let orange_buy_price : ℚ := 9 / 10  -- $2.70 for 3 oranges
  let orange_sell_price : ℚ := 1    -- $1 each
  let apples_sold : ℕ := 5
  let oranges_sold : ℕ := 5
  vendor_profit apple_buy_price apple_sell_price orange_buy_price orange_sell_price apples_sold oranges_sold = 3 := by
  sorry


end NUMINAMATH_CALUDE_vendor_profit_calculation_l3348_334876


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l3348_334819

def total_crayons : ℕ := 15
def red_crayons : ℕ := 3
def selected_crayons : ℕ := 5

theorem crayon_selection_theorem :
  (Nat.choose total_crayons selected_crayons - 
   Nat.choose (total_crayons - red_crayons) selected_crayons) = 2211 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l3348_334819


namespace NUMINAMATH_CALUDE_circle_radius_constant_l3348_334874

theorem circle_radius_constant (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 5^2) → 
  c = 42 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_constant_l3348_334874


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_half_sum_of_other_squares_l3348_334838

theorem sum_of_squares_equals_half_sum_of_other_squares (a b : ℝ) :
  a^2 + b^2 = ((a + b)^2 + (a - b)^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_half_sum_of_other_squares_l3348_334838


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3348_334837

theorem rationalize_denominator :
  (Real.sqrt 18 + Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 8) = 5 * Real.sqrt 6 - 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3348_334837


namespace NUMINAMATH_CALUDE_fourth_root_of_y_squared_times_sqrt_y_l3348_334831

theorem fourth_root_of_y_squared_times_sqrt_y (y : ℝ) (h : y > 0) :
  (y^2 * y^(1/2))^(1/4) = y^(5/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_y_squared_times_sqrt_y_l3348_334831


namespace NUMINAMATH_CALUDE_difference_of_squares_l3348_334896

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3348_334896


namespace NUMINAMATH_CALUDE_at_least_one_angle_le_30_deg_l3348_334857

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a point P
variable (P : Point)

-- Define that P is inside the triangle
def isInside (P : Point) (t : Triangle) : Prop := sorry

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem at_least_one_angle_le_30_deg (t : Triangle) (P : Point) 
  (h : isInside P t) : 
  (angle P t.A t.B ≤ 30) ∨ (angle P t.B t.C ≤ 30) ∨ (angle P t.C t.A ≤ 30) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_angle_le_30_deg_l3348_334857


namespace NUMINAMATH_CALUDE_cookies_with_seven_cups_l3348_334894

/-- The number of cookies Lee can make with a given number of cups of flour -/
def cookies_made (cups : ℕ) : ℝ :=
  if cups ≤ 4 then 36
  else cookies_made (cups - 1) * 1.5

/-- The theorem stating the number of cookies Lee can make with 7 cups of flour -/
theorem cookies_with_seven_cups :
  cookies_made 7 = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_with_seven_cups_l3348_334894


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l3348_334825

/-- Two cyclists meet on a course -/
theorem cyclists_meeting_time
  (course_length : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : course_length = 45)
  (h2 : speed1 = 14)
  (h3 : speed2 = 16) :
  ∃ t : ℝ, t * (speed1 + speed2) = course_length ∧ t = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l3348_334825


namespace NUMINAMATH_CALUDE_bushes_for_sixty_zucchinis_l3348_334877

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 12

/-- The number of containers of blueberries that can be traded for 3 zucchinis -/
def containers_per_trade : ℕ := 8

/-- The number of zucchinis received in one trade -/
def zucchinis_per_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- The function to calculate the number of bushes needed for a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) : ℕ :=
  (zucchinis * containers_per_trade + containers_per_bush * zucchinis_per_trade - 1) / 
  (containers_per_bush * zucchinis_per_trade)

theorem bushes_for_sixty_zucchinis :
  bushes_needed target_zucchinis = 14 := by
  sorry

end NUMINAMATH_CALUDE_bushes_for_sixty_zucchinis_l3348_334877


namespace NUMINAMATH_CALUDE_jaymee_shara_age_difference_l3348_334855

theorem jaymee_shara_age_difference (shara_age jaymee_age : ℕ) 
  (h1 : shara_age = 10) 
  (h2 : jaymee_age = 22) : 
  jaymee_age - 2 * shara_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_jaymee_shara_age_difference_l3348_334855


namespace NUMINAMATH_CALUDE_prime_sum_product_l3348_334815

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_sum_product :
  ∃ p q : ℕ,
    is_prime p ∧
    is_prime q ∧
    p + q = 102 ∧
    (p > 30 ∨ q > 30) ∧
    p * q = 2201 :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3348_334815


namespace NUMINAMATH_CALUDE_mike_reaches_sarah_time_l3348_334880

-- Define the initial conditions
def initial_distance : ℝ := 24
def sarah_speed_ratio : ℝ := 4
def distance_decrease_rate : ℝ := 2
def sarah_stop_time : ℝ := 6

-- Define the theorem
theorem mike_reaches_sarah_time :
  let total_speed := distance_decrease_rate * 60 -- Convert km/min to km/h
  let mike_speed := total_speed / (sarah_speed_ratio + 1)
  let sarah_speed := mike_speed * sarah_speed_ratio
  let distance_after_sarah_stops := initial_distance - distance_decrease_rate * sarah_stop_time
  let mike_remaining_time := distance_after_sarah_stops / mike_speed
  sarah_stop_time + mike_remaining_time * 60 = 36 := by
  sorry


end NUMINAMATH_CALUDE_mike_reaches_sarah_time_l3348_334880


namespace NUMINAMATH_CALUDE_service_cost_is_correct_l3348_334849

/-- Represents the service cost per vehicle at a fuel station. -/
def service_cost_per_vehicle : ℝ := 2.30

/-- Represents the cost of fuel per liter. -/
def fuel_cost_per_liter : ℝ := 0.70

/-- Represents the number of mini-vans. -/
def num_mini_vans : ℕ := 4

/-- Represents the number of trucks. -/
def num_trucks : ℕ := 2

/-- Represents the total cost for all vehicles. -/
def total_cost : ℝ := 396

/-- Represents the capacity of a mini-van's fuel tank in liters. -/
def mini_van_tank_capacity : ℝ := 65

/-- Represents the percentage by which a truck's tank is larger than a mini-van's tank. -/
def truck_tank_percentage : ℝ := 120

/-- Theorem stating that the service cost per vehicle is $2.30 given the problem conditions. -/
theorem service_cost_is_correct :
  let truck_tank_capacity := mini_van_tank_capacity * (1 + truck_tank_percentage / 100)
  let total_fuel_volume := num_mini_vans * mini_van_tank_capacity + num_trucks * truck_tank_capacity
  let total_fuel_cost := total_fuel_volume * fuel_cost_per_liter
  let total_service_cost := total_cost - total_fuel_cost
  service_cost_per_vehicle = total_service_cost / (num_mini_vans + num_trucks) := by
  sorry


end NUMINAMATH_CALUDE_service_cost_is_correct_l3348_334849


namespace NUMINAMATH_CALUDE_not_divisible_by_three_times_sum_of_products_l3348_334844

theorem not_divisible_by_three_times_sum_of_products (x y z : ℕ+) :
  ¬ (3 * (x * y + y * z + z * x) ∣ x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_times_sum_of_products_l3348_334844


namespace NUMINAMATH_CALUDE_darryl_earnings_l3348_334865

/-- Calculates the total earnings from selling melons --/
def melon_earnings (
  cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (remaining_cantaloupes : ℕ)
  (remaining_honeydews : ℕ) : ℕ :=
  let sold_cantaloupes := initial_cantaloupes - dropped_cantaloupes - remaining_cantaloupes
  let sold_honeydews := initial_honeydews - rotten_honeydews - remaining_honeydews
  cantaloupe_price * sold_cantaloupes + honeydew_price * sold_honeydews

/-- Theorem stating that Darryl's earnings are $85 --/
theorem darryl_earnings : 
  melon_earnings 2 3 30 27 2 3 8 9 = 85 := by
  sorry

end NUMINAMATH_CALUDE_darryl_earnings_l3348_334865


namespace NUMINAMATH_CALUDE_optimal_weight_combination_l3348_334840

/-- Represents a combination of weights -/
structure WeightCombination where
  weight3 : ℕ
  weight5 : ℕ
  weight7 : ℕ

/-- Calculates the total weight of a combination -/
def totalWeight (c : WeightCombination) : ℕ :=
  3 * c.weight3 + 5 * c.weight5 + 7 * c.weight7

/-- Calculates the total number of weights in a combination -/
def totalWeights (c : WeightCombination) : ℕ :=
  c.weight3 + c.weight5 + c.weight7

/-- Checks if a combination is valid (totals 130 grams) -/
def isValid (c : WeightCombination) : Prop :=
  totalWeight c = 130

/-- The optimal combination of weights -/
def optimalCombination : WeightCombination :=
  { weight3 := 2, weight5 := 1, weight7 := 17 }

theorem optimal_weight_combination :
  isValid optimalCombination ∧
  (∀ c : WeightCombination, isValid c → totalWeights optimalCombination ≤ totalWeights c) :=
by sorry

end NUMINAMATH_CALUDE_optimal_weight_combination_l3348_334840


namespace NUMINAMATH_CALUDE_or_implies_at_least_one_true_l3348_334888

theorem or_implies_at_least_one_true (p q : Prop) : 
  (p ∨ q) → (p ∨ q) :=
by
  sorry

end NUMINAMATH_CALUDE_or_implies_at_least_one_true_l3348_334888


namespace NUMINAMATH_CALUDE_range_of_t_l3348_334814

theorem range_of_t (a b c t : ℝ) 
  (eq1 : 6 * a = 2 * b - 6)
  (eq2 : 6 * a = 3 * c)
  (cond1 : b ≥ 0)
  (cond2 : c ≤ 2)
  (def_t : t = 2 * a + b - c) :
  0 ≤ t ∧ t ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l3348_334814


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l3348_334892

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l3348_334892


namespace NUMINAMATH_CALUDE_cycle_price_proof_l3348_334867

/-- Proves that given a cycle sold for 1350 with a 50% gain, the original price was 900 -/
theorem cycle_price_proof (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1350)
  (h2 : gain_percentage = 50) : 
  selling_price / (1 + gain_percentage / 100) = 900 := by
  sorry

#check cycle_price_proof

end NUMINAMATH_CALUDE_cycle_price_proof_l3348_334867


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3348_334803

/-- The area of a triangular region bounded by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

/-- The x-intercept of the line -/
def xIntercept : ℝ := 4

/-- The y-intercept of the line -/
def yIntercept : ℝ := 6

theorem triangle_area_proof :
  lineEquation xIntercept 0 ∧
  lineEquation 0 yIntercept ∧
  triangleArea = (1 / 2) * xIntercept * yIntercept :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3348_334803


namespace NUMINAMATH_CALUDE_power_of_product_cubes_l3348_334836

theorem power_of_product_cubes : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cubes_l3348_334836


namespace NUMINAMATH_CALUDE_systematic_sampling_class_l3348_334843

/-- Systematic sampling function that returns the nth selected item given the start and interval -/
def systematicSample (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * interval

/-- Theorem for systematic sampling in a class -/
theorem systematic_sampling_class (classSize : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) 
  (h1 : classSize = 48) 
  (h2 : sampleSize = 4) 
  (h3 : firstSelected = 7) :
  let interval := classSize / sampleSize
  (systematicSample firstSelected interval 2 = 19) ∧
  (systematicSample firstSelected interval 3 = 31) ∧
  (systematicSample firstSelected interval 4 = 43) := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_class_l3348_334843


namespace NUMINAMATH_CALUDE_age_calculation_l3348_334812

/-- Given Luke's current age and Mr. Bernard's future age relative to Luke's,
    calculate 10 years less than their average current age. -/
theorem age_calculation (luke_age : ℕ) (bernard_future_age_factor : ℕ) (years_in_future : ℕ) : 
  luke_age = 20 →
  years_in_future = 8 →
  bernard_future_age_factor = 3 →
  10 < luke_age →
  (luke_age + (bernard_future_age_factor * luke_age - years_in_future)) / 2 - 10 = 26 := by
sorry

end NUMINAMATH_CALUDE_age_calculation_l3348_334812


namespace NUMINAMATH_CALUDE_multiple_of_six_as_sum_of_four_cubes_l3348_334853

theorem multiple_of_six_as_sum_of_four_cubes (k : ℤ) :
  ∃ (a b c d : ℤ), 6 * k = a^3 + b^3 + c^3 + d^3 :=
sorry

end NUMINAMATH_CALUDE_multiple_of_six_as_sum_of_four_cubes_l3348_334853


namespace NUMINAMATH_CALUDE_find_S_l3348_334878

theorem find_S : ∃ S : ℚ, (1/4 : ℚ) * (1/6 : ℚ) * S = (1/5 : ℚ) * (1/8 : ℚ) * 160 ∧ S = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l3348_334878


namespace NUMINAMATH_CALUDE_claire_crafting_time_l3348_334860

/-- Represents the system of equations for Claire's time allocation --/
structure ClaireTimeSystem where
  x : ℝ
  y : ℝ
  z : ℝ
  crafting : ℝ
  tailoring : ℝ
  eq1 : (2 * y) + y + (y - 1) + crafting + crafting + 8 = 24
  eq2 : x = 2 * y
  eq3 : z = y - 1
  eq4 : crafting = tailoring
  eq5 : 2 * crafting = 9 - tailoring

/-- Theorem stating that in any valid ClaireTimeSystem, the crafting time is 3 hours --/
theorem claire_crafting_time (s : ClaireTimeSystem) : s.crafting = 3 := by
  sorry

end NUMINAMATH_CALUDE_claire_crafting_time_l3348_334860


namespace NUMINAMATH_CALUDE_tan_negative_435_degrees_l3348_334842

theorem tan_negative_435_degrees :
  Real.tan ((-435 : ℝ) * π / 180) = -((Real.sqrt 6 + Real.sqrt 2) / (Real.sqrt 6 - Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_435_degrees_l3348_334842


namespace NUMINAMATH_CALUDE_initial_number_proof_l3348_334841

theorem initial_number_proof (x : ℝ) : (x - 1/4) / (1/2) = 4.5 → x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3348_334841


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3348_334818

theorem inequality_solution_set : 
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | -7 < x ∧ x ≤ -1 ∨ 5 ≤ x ∧ x < 11} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3348_334818


namespace NUMINAMATH_CALUDE_number_problem_l3348_334835

theorem number_problem : 
  ∃ x : ℝ, (1345 - (x / 20.04) = 1295) ∧ (x = 1002) := by sorry

end NUMINAMATH_CALUDE_number_problem_l3348_334835


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l3348_334822

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) : 
  total_boys = 850 →
  hindu_percentage = 32/100 →
  sikh_percentage = 10/100 →
  other_boys = 119 →
  (total_boys - (hindu_percentage * total_boys).num - (sikh_percentage * total_boys).num - other_boys) / total_boys = 44/100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l3348_334822


namespace NUMINAMATH_CALUDE_product_equals_72_17_l3348_334830

/-- Represents the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal 0.456̄ and 9 -/
def product : ℚ := 9 * repeating_decimal

theorem product_equals_72_17 : product = 72 / 17 := by sorry

end NUMINAMATH_CALUDE_product_equals_72_17_l3348_334830


namespace NUMINAMATH_CALUDE_trapezoid_area_l3348_334846

/-- Given an outer equilateral triangle with area 36, an inner equilateral triangle
    with area 4, and three congruent trapezoids between them, the area of one
    trapezoid is 32/3. -/
theorem trapezoid_area (outer_triangle : Real) (inner_triangle : Real) (num_trapezoids : Nat) :
  outer_triangle = 36 →
  inner_triangle = 4 →
  num_trapezoids = 3 →
  (outer_triangle - inner_triangle) / num_trapezoids = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3348_334846


namespace NUMINAMATH_CALUDE_first_group_size_l3348_334854

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The daily wage in rupees -/
def daily_wage : ℚ := sorry

theorem first_group_size :
  (M * 10 * daily_wage = 1200) ∧
  (9 * 6 * daily_wage = 1620) →
  M = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l3348_334854


namespace NUMINAMATH_CALUDE_data_analytics_course_hours_l3348_334823

/-- Calculates the total hours spent on a course given the course duration and weekly time commitments. -/
def total_course_hours (weeks : ℕ) (class_hours_1 : ℕ) (class_hours_2 : ℕ) (class_hours_3 : ℕ) (homework_hours : ℕ) : ℕ :=
  weeks * (class_hours_1 + class_hours_2 + class_hours_3 + homework_hours)

/-- Theorem stating that the total hours spent on the described course is 336. -/
theorem data_analytics_course_hours : 
  total_course_hours 24 3 3 4 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_data_analytics_course_hours_l3348_334823


namespace NUMINAMATH_CALUDE_machine_b_time_for_150_copies_l3348_334832

/-- Given two machines A and B with the following properties:
    1. Machine A makes 100 copies in 20 minutes
    2. Machines A and B working simultaneously for 30 minutes produce 600 copies
    This theorem proves that it takes 10 minutes for Machine B to make 150 copies -/
theorem machine_b_time_for_150_copies 
  (rate_a : ℚ) -- rate of machine A in copies per minute
  (rate_b : ℚ) -- rate of machine B in copies per minute
  (h1 : rate_a = 100 / 20) -- condition 1
  (h2 : 30 * (rate_a + rate_b) = 600) -- condition 2
  : 150 / rate_b = 10 := by sorry

end NUMINAMATH_CALUDE_machine_b_time_for_150_copies_l3348_334832


namespace NUMINAMATH_CALUDE_hexagon_reachability_l3348_334863

def Hexagon := Fin 6 → ℤ

def initial_hexagon : Hexagon := ![12, 1, 10, 6, 8, 3]

def is_valid_move (h1 h2 : Hexagon) : Prop :=
  ∃ i : Fin 6, 
    (h2 i = h1 i + 1 ∧ h2 ((i + 1) % 6) = h1 ((i + 1) % 6) + 1) ∨
    (h2 i = h1 i - 1 ∧ h2 ((i + 1) % 6) = h1 ((i + 1) % 6) - 1) ∧
    ∀ j : Fin 6, j ≠ i ∧ j ≠ (i + 1) % 6 → h2 j = h1 j

def is_reachable (start goal : Hexagon) : Prop :=
  ∃ (n : ℕ) (sequence : Fin (n + 1) → Hexagon),
    sequence 0 = start ∧
    sequence n = goal ∧
    ∀ i : Fin n, is_valid_move (sequence i) (sequence (i + 1))

theorem hexagon_reachability :
  (is_reachable initial_hexagon ![14, 6, 13, 4, 5, 2]) ∧
  ¬(is_reachable initial_hexagon ![6, 17, 14, 3, 15, 2]) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_reachability_l3348_334863


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_square_difference_150_l3348_334800

theorem no_integer_solutions_for_square_difference_150 :
  ∀ m n : ℕ+, m ≥ n → m^2 - n^2 ≠ 150 := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_square_difference_150_l3348_334800


namespace NUMINAMATH_CALUDE_table_size_lower_bound_l3348_334839

/-- Represents a table with 10 columns and n rows, where each cell contains a digit. -/
structure DigitTable (n : ℕ) :=
  (rows : Fin n → Fin 10 → Fin 10)

/-- 
Given a table with 10 columns and n rows, where each cell contains a digit, 
and for any row A and any two columns, there exists a row that differs from A 
in exactly these two columns, prove that n ≥ 512.
-/
theorem table_size_lower_bound {n : ℕ} (t : DigitTable n) 
  (h : ∀ (A : Fin n) (i j : Fin 10), i ≠ j → 
    ∃ (B : Fin n), (∀ k : Fin 10, k ≠ i ∧ k ≠ j → t.rows A k = t.rows B k) ∧
                   t.rows A i ≠ t.rows B i ∧ 
                   t.rows A j ≠ t.rows B j) : 
  n ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_table_size_lower_bound_l3348_334839


namespace NUMINAMATH_CALUDE_ab_squared_nonpositive_l3348_334859

theorem ab_squared_nonpositive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_squared_nonpositive_l3348_334859


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l3348_334806

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; -1, 0]
  A * B = !![4, 2; -3, 0] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l3348_334806


namespace NUMINAMATH_CALUDE_smallest_even_cube_ending_392_l3348_334816

theorem smallest_even_cube_ending_392 :
  ∀ n : ℕ, n > 0 → Even n → n^3 ≡ 392 [ZMOD 1000] → n ≥ 892 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_cube_ending_392_l3348_334816


namespace NUMINAMATH_CALUDE_min_value_theorem_inequality_theorem_l3348_334811

variable (a b c : ℝ)

-- Define the conditions
def sum_condition (a b c : ℝ) : Prop := a + 2 * b + 3 * c = 6

-- Define the non-zero condition
def non_zero (x : ℝ) : Prop := x ≠ 0

-- Theorem for the first part
theorem min_value_theorem (ha : non_zero a) (hb : non_zero b) (hc : non_zero c) 
  (h_sum : sum_condition a b c) : 
  a^2 + 2 * b^2 + 3 * c^2 ≥ 6 := by sorry

-- Theorem for the second part
theorem inequality_theorem (ha : non_zero a) (hb : non_zero b) (hc : non_zero c) 
  (h_sum : sum_condition a b c) : 
  a^2 / (1 + a) + 2 * b^2 / (3 + b) + 3 * c^2 / (5 + c) ≥ 9/7 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_inequality_theorem_l3348_334811


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3348_334809

/-- Given two parallel vectors a and b, prove that y = 4 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) :
  a = (2, 6) →
  b = (1, -1 + y) →
  ∃ (k : ℝ), a = k • b →
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3348_334809


namespace NUMINAMATH_CALUDE_fraction_of_boys_l3348_334820

theorem fraction_of_boys (total_students : ℕ) (girls_no_pets : ℕ) 
  (dog_owners_percent : ℚ) (cat_owners_percent : ℚ) :
  total_students = 30 →
  girls_no_pets = 8 →
  dog_owners_percent = 40 / 100 →
  cat_owners_percent = 20 / 100 →
  (17 : ℚ) / 30 = (total_students - (girls_no_pets / ((1 : ℚ) - dog_owners_percent - cat_owners_percent))) / total_students :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_boys_l3348_334820


namespace NUMINAMATH_CALUDE_wrapping_paper_distribution_l3348_334833

theorem wrapping_paper_distribution (total : ℚ) (decoration : ℚ) (num_presents : ℕ) :
  total = 5/8 ∧ decoration = 1/24 ∧ num_presents = 4 →
  (total - decoration) / (num_presents - 1) = 7/36 := by
sorry

end NUMINAMATH_CALUDE_wrapping_paper_distribution_l3348_334833


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l3348_334885

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (hn : n > 0) (h_div : 72 ∣ n^2) :
  ∃ m : ℕ, m = 12 ∧ m ∣ n ∧ ∀ k : ℕ, k ∣ n → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l3348_334885


namespace NUMINAMATH_CALUDE_selina_pants_sold_l3348_334845

/-- Represents the number of pants Selina sold -/
def pants_sold : ℕ := sorry

/-- The price of each pair of pants -/
def pants_price : ℕ := 5

/-- The price of each pair of shorts -/
def shorts_price : ℕ := 3

/-- The price of each shirt -/
def shirt_price : ℕ := 4

/-- The number of shorts Selina sold -/
def shorts_sold : ℕ := 5

/-- The number of shirts Selina sold -/
def shirts_sold : ℕ := 5

/-- The price of each new shirt Selina bought -/
def new_shirt_price : ℕ := 10

/-- The number of new shirts Selina bought -/
def new_shirts_bought : ℕ := 2

/-- The amount of money Selina left the store with -/
def money_left : ℕ := 30

theorem selina_pants_sold : 
  pants_sold * pants_price + 
  shorts_sold * shorts_price + 
  shirts_sold * shirt_price = 
  money_left + new_shirts_bought * new_shirt_price ∧ 
  pants_sold = 3 := by sorry

end NUMINAMATH_CALUDE_selina_pants_sold_l3348_334845


namespace NUMINAMATH_CALUDE_fish_tagging_problem_l3348_334847

/-- The number of fish initially tagged in a pond -/
def initially_tagged (total_fish : ℕ) (later_catch : ℕ) (tagged_in_catch : ℕ) : ℕ :=
  (tagged_in_catch * total_fish) / later_catch

theorem fish_tagging_problem (total_fish : ℕ) (later_catch : ℕ) (tagged_in_catch : ℕ)
  (h1 : total_fish = 1800)
  (h2 : later_catch = 60)
  (h3 : tagged_in_catch = 2)
  (h4 : initially_tagged total_fish later_catch tagged_in_catch = (tagged_in_catch * total_fish) / later_catch) :
  initially_tagged total_fish later_catch tagged_in_catch = 60 :=
by sorry

end NUMINAMATH_CALUDE_fish_tagging_problem_l3348_334847


namespace NUMINAMATH_CALUDE_soda_cost_per_ounce_l3348_334870

/-- The cost of soda per ounce, given initial money, remaining money, and amount bought. -/
def cost_per_ounce (initial_money remaining_money amount_bought : ℚ) : ℚ :=
  (initial_money - remaining_money) / amount_bought

/-- Theorem stating that the cost per ounce is $0.25 under given conditions. -/
theorem soda_cost_per_ounce :
  cost_per_ounce 2 0.5 6 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_per_ounce_l3348_334870


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3348_334817

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3348_334817


namespace NUMINAMATH_CALUDE_sequence_properties_and_sum_l3348_334828

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

def geometric_sequence (b₁ : ℕ) (q : ℕ) : ℕ → ℕ
  | n => b₁ * q^(n - 1)

def merge_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  sorry

theorem sequence_properties_and_sum :
  ∀ (a b : ℕ → ℕ),
    (a 1 = 1) →
    (∀ n, a (b n) = 2^(n+1) - 1) →
    (∀ n, a n = 2*n - 1) →
    (∀ n, b n = 2^n) →
    merge_and_sum a b 100 = 8903 :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_and_sum_l3348_334828


namespace NUMINAMATH_CALUDE_circle_intersection_radius_range_l3348_334882

/-- The range of r values for which there are two points P satisfying the given conditions -/
theorem circle_intersection_radius_range :
  ∀ (r : ℝ),
  (∃ (P₁ P₂ : ℝ × ℝ),
    P₁ ≠ P₂ ∧
    ((P₁.1 - 3)^2 + (P₁.2 - 4)^2 = r^2) ∧
    ((P₂.1 - 3)^2 + (P₂.2 - 4)^2 = r^2) ∧
    ((P₁.1 + 2)^2 + P₁.2^2 + (P₁.1 - 2)^2 + P₁.2^2 = 40) ∧
    ((P₂.1 + 2)^2 + P₂.2^2 + (P₂.1 - 2)^2 + P₂.2^2 = 40)) ↔
  (1 < r ∧ r < 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_radius_range_l3348_334882


namespace NUMINAMATH_CALUDE_meteorologist_more_reliable_l3348_334856

/-- Probability of a clear day -/
def p_clear : ℝ := 0.74

/-- Accuracy of a senator's forecast -/
def p_senator_accuracy : ℝ := sorry

/-- Accuracy of the meteorologist's forecast -/
def p_meteorologist_accuracy : ℝ := 1.5 * p_senator_accuracy

/-- Event that the day is clear -/
def G : Prop := sorry

/-- Event that the first senator predicts a clear day -/
def M₁ : Prop := sorry

/-- Event that the second senator predicts a clear day -/
def M₂ : Prop := sorry

/-- Event that the meteorologist predicts a rainy day -/
def S : Prop := sorry

/-- Probability of an event -/
noncomputable def P : Prop → ℝ := sorry

/-- Conditional probability -/
noncomputable def P_cond (A B : Prop) : ℝ := P (A ∧ B) / P B

theorem meteorologist_more_reliable :
  P_cond (¬G) (S ∧ M₁ ∧ M₂) > P_cond G (S ∧ M₁ ∧ M₂) :=
sorry

end NUMINAMATH_CALUDE_meteorologist_more_reliable_l3348_334856


namespace NUMINAMATH_CALUDE_max_b_for_integer_solution_l3348_334861

theorem max_b_for_integer_solution : ∃ (b : ℤ), b = 9599 ∧
  (∀ (b' : ℤ), (∃ (x : ℤ), x^2 + b'*x - 9600 = 0 ∧ 10 ∣ x ∧ 12 ∣ x) → b' ≤ b) ∧
  (∃ (x : ℤ), x^2 + b*x - 9600 = 0 ∧ 10 ∣ x ∧ 12 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_max_b_for_integer_solution_l3348_334861


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l3348_334827

theorem arithmetic_geometric_mean_equation (α β : ℝ) :
  (α + β) / 2 = 8 →
  Real.sqrt (α * β) = 15 →
  (∀ x : ℝ, x^2 - 16*x + 225 = 0 ↔ x = α ∨ x = β) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l3348_334827


namespace NUMINAMATH_CALUDE_sector_angle_measure_l3348_334899

/-- Given a circular sector with arc length and area both equal to 5,
    prove that the radian measure of its central angle is 5/2 -/
theorem sector_angle_measure (r : ℝ) (α : ℝ) 
    (h1 : α * r = 5)  -- arc length formula
    (h2 : (1/2) * α * r^2 = 5)  -- sector area formula
    : α = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l3348_334899


namespace NUMINAMATH_CALUDE_min_value_inequality_l3348_334834

theorem min_value_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) : 
  (1 / (x + y)) + ((x + y) / z) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3348_334834


namespace NUMINAMATH_CALUDE_lesser_number_proof_l3348_334889

theorem lesser_number_proof (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : min a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_proof_l3348_334889


namespace NUMINAMATH_CALUDE_prob_A_plus_B_complement_l3348_334898

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event A
def A : Finset Nat := {2, 4}

-- Define event B
def B : Finset Nat := {1, 2, 3, 4, 5}

-- Define the complement of B
def B_complement : Finset Nat := Ω \ B

-- Define the probability measure
def P (E : Finset Nat) : Rat := (E.card : Rat) / (Ω.card : Rat)

-- State the theorem
theorem prob_A_plus_B_complement : P (A ∪ B_complement) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_plus_B_complement_l3348_334898


namespace NUMINAMATH_CALUDE_amount_distribution_l3348_334826

theorem amount_distribution (A : ℕ) : 
  (A / 14 = A / 18 + 80) → A = 5040 :=
by
  sorry

end NUMINAMATH_CALUDE_amount_distribution_l3348_334826


namespace NUMINAMATH_CALUDE_abs_equation_one_l3348_334829

theorem abs_equation_one (x : ℝ) : |3*x - 5| + 4 = 8 ↔ x = 3 ∨ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_one_l3348_334829


namespace NUMINAMATH_CALUDE_cost_per_chicken_problem_l3348_334871

/-- Given a total number of birds, a fraction of ducks, and the total cost to feed chickens,
    calculate the cost per chicken. -/
def cost_per_chicken (total_birds : ℕ) (duck_fraction : ℚ) (total_cost : ℚ) : ℚ :=
  let chicken_fraction : ℚ := 1 - duck_fraction
  let num_chickens : ℚ := chicken_fraction * total_birds
  total_cost / num_chickens

theorem cost_per_chicken_problem :
  cost_per_chicken 15 (1/3) 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_chicken_problem_l3348_334871


namespace NUMINAMATH_CALUDE_captain_smollett_problem_l3348_334852

theorem captain_smollett_problem :
  ∃! (a c l : ℕ), 
    0 < a ∧ a < 100 ∧
    c > 3 ∧
    l > 0 ∧
    a * c * l = 32118 ∧
    a = 53 ∧ c = 6 ∧ l = 101 := by
  sorry

end NUMINAMATH_CALUDE_captain_smollett_problem_l3348_334852


namespace NUMINAMATH_CALUDE_cube_inequality_l3348_334893

theorem cube_inequality (a b : ℝ) (h : a < 0 ∧ 0 < b) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l3348_334893


namespace NUMINAMATH_CALUDE_circular_tank_properties_l3348_334805

theorem circular_tank_properties (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) :
  let r := (AB / 2)^2 + DC^2
  (π * r = 244 * π) ∧ (2 * π * Real.sqrt r = 2 * π * Real.sqrt 244) := by
  sorry

end NUMINAMATH_CALUDE_circular_tank_properties_l3348_334805


namespace NUMINAMATH_CALUDE_entire_group_is_population_l3348_334848

/-- Represents a group of students who took a test -/
structure StudentGroup where
  size : ℕ
  scores : Finset ℝ
  h_size : scores.card = size

/-- Represents a sample extracted from a larger group -/
structure Sample (group : StudentGroup) where
  size : ℕ
  scores : Finset ℝ
  h_size : scores.card = size
  h_subset : scores ⊆ group.scores

/-- Definition of a population in statistical terms -/
def isPopulation (group : StudentGroup) : Prop :=
  ∀ (sample : Sample group), sample.scores ⊆ group.scores

/-- The theorem to be proved -/
theorem entire_group_is_population 
  (entireGroup : StudentGroup) 
  (sample : Sample entireGroup) 
  (h_entire_size : entireGroup.size = 5000) 
  (h_sample_size : sample.size = 200) : 
  isPopulation entireGroup := by
  sorry

end NUMINAMATH_CALUDE_entire_group_is_population_l3348_334848


namespace NUMINAMATH_CALUDE_monday_loaves_l3348_334884

/-- Represents the number of loaves baked on a given day -/
def loaves : Fin 6 → ℕ
  | 0 => 5  -- Wednesday
  | 1 => 7  -- Thursday
  | 2 => 10 -- Friday
  | 3 => 14 -- Saturday
  | 4 => 19 -- Sunday
  | 5 => 25 -- Monday (to be proven)

/-- The pattern of increase in loaves from one day to the next -/
def increase (n : Fin 5) : ℕ := loaves (n + 1) - loaves n

/-- The theorem stating that the number of loaves baked on Monday is 25 -/
theorem monday_loaves :
  (∀ n : Fin 4, increase (n + 1) = increase n + 1) →
  loaves 5 = 25 := by
  sorry


end NUMINAMATH_CALUDE_monday_loaves_l3348_334884


namespace NUMINAMATH_CALUDE_sold_to_production_ratio_l3348_334864

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def phones_left : ℕ := 7500

def sold_phones : ℕ := this_year_production - phones_left

theorem sold_to_production_ratio : 
  (sold_phones : ℚ) / this_year_production = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sold_to_production_ratio_l3348_334864


namespace NUMINAMATH_CALUDE_vacation_cost_l3348_334802

theorem vacation_cost (C : ℝ) : 
  (C / 5 - C / 8 = 60) → C = 800 := by sorry

end NUMINAMATH_CALUDE_vacation_cost_l3348_334802


namespace NUMINAMATH_CALUDE_composition_equation_solution_l3348_334875

theorem composition_equation_solution (p q : ℝ → ℝ) (c : ℝ) :
  (∀ x, p x = 3 * x - 8) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 14 →
  c = 23 / 3 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l3348_334875


namespace NUMINAMATH_CALUDE_geometric_sequence_characterization_l3348_334801

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_characterization (a : ℕ → ℚ) :
  is_geometric_sequence a ↔ 
  (∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_characterization_l3348_334801


namespace NUMINAMATH_CALUDE_student_sampling_interval_l3348_334810

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 40 is 25 -/
theorem student_sampling_interval :
  systematicSamplingInterval 1000 40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_student_sampling_interval_l3348_334810


namespace NUMINAMATH_CALUDE_expected_sales_at_2_degrees_l3348_334890

/-- Represents the linear regression model for hot drink sales based on temperature -/
def hot_drink_sales (x : ℝ) : ℝ := -2.35 * x + 147.7

/-- Theorem stating that when the temperature is 2°C, the expected number of hot drinks sold is 143 -/
theorem expected_sales_at_2_degrees :
  Int.floor (hot_drink_sales 2) = 143 := by
  sorry

end NUMINAMATH_CALUDE_expected_sales_at_2_degrees_l3348_334890


namespace NUMINAMATH_CALUDE_circle_center_apollonius_l3348_334866

/-- The center of the circle formed by points P where OP:PQ = 5:4, given O(0,0) and Q(1,2) -/
theorem circle_center_apollonius (P : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (1, 2)
  let OP := Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)
  let PQ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (4 * OP = 5 * PQ) → (25/9, 50/9) = (
    (P.1^2 + P.2^2 - 25 * ((P.1 - 1)^2 + (P.2 - 2)^2) / 16) / (2 * P.1),
    (P.1^2 + P.2^2 - 25 * ((P.1 - 1)^2 + (P.2 - 2)^2) / 16) / (2 * P.2)
  ) := by
sorry


end NUMINAMATH_CALUDE_circle_center_apollonius_l3348_334866


namespace NUMINAMATH_CALUDE_sun_division_l3348_334872

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
theorem sun_division (s : ShareDistribution) :
  s.x = 1 →                -- For each rupee x gets
  s.y = 0.45 →             -- y gets 45 paisa (0.45 rupees)
  s.z = 0.5 →              -- z gets 50 paisa (0.5 rupees)
  s.y * (1 / 0.45) = 45 →  -- The share of y is Rs. 45
  s.x * (1 / 0.45) + s.y * (1 / 0.45) + s.z * (1 / 0.45) = 195 := by
  sorry

#check sun_division

end NUMINAMATH_CALUDE_sun_division_l3348_334872


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3348_334881

theorem trigonometric_inequality : 
  let a := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let b := (2 * Real.tan (76 * π / 180)) / (1 + Real.tan (76 * π / 180) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3348_334881


namespace NUMINAMATH_CALUDE_jason_oranges_l3348_334862

theorem jason_oranges (mary_oranges total_oranges : ℕ)
  (h1 : mary_oranges = 14)
  (h2 : total_oranges = 55) :
  total_oranges - mary_oranges = 41 := by
  sorry

end NUMINAMATH_CALUDE_jason_oranges_l3348_334862


namespace NUMINAMATH_CALUDE_equation_solution_l3348_334851

theorem equation_solution :
  let f (x : ℝ) := (7 * x + 3) / (3 * x^2 + 7 * x - 6)
  let g (x : ℝ) := (3 * x) / (3 * x - 2)
  ∀ x : ℝ, f x = g x ↔ x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3348_334851


namespace NUMINAMATH_CALUDE_project_work_time_difference_l3348_334886

/-- Given three people working on a project for a total of 140 hours,
    with their working times in the ratio of 3:5:6,
    prove that the difference between the longest and shortest working times is 30 hours. -/
theorem project_work_time_difference (x : ℝ) 
  (h1 : 3 * x + 5 * x + 6 * x = 140) : 6 * x - 3 * x = 30 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_difference_l3348_334886


namespace NUMINAMATH_CALUDE_impossibility_of_circular_arrangement_l3348_334869

theorem impossibility_of_circular_arrangement : ¬ ∃ (arrangement : Fin 1995 → ℕ), 
  (∀ i j : Fin 1995, i ≠ j → arrangement i ≠ arrangement j) ∧ 
  (∀ i : Fin 1995, Nat.Prime ((max (arrangement i) (arrangement (i + 1))) / 
                               (min (arrangement i) (arrangement (i + 1))))) :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_circular_arrangement_l3348_334869


namespace NUMINAMATH_CALUDE_root_sum_quotient_l3348_334821

theorem root_sum_quotient (m₁ m₂ : ℝ) : 
  m₁^2 - 21*m₁ + 4 = 0 → 
  m₂^2 - 21*m₂ + 4 = 0 → 
  m₁ / m₂ + m₂ / m₁ = 108.25 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l3348_334821


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3348_334868

theorem hemisphere_surface_area :
  let sphere_surface_area (r : ℝ) := 4 * Real.pi * r^2
  let base_area := 3
  let hemisphere_surface_area (r : ℝ) := 2 * Real.pi * r^2 + base_area
  ∃ r : ℝ, base_area = Real.pi * r^2 ∧ hemisphere_surface_area r = 9 :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3348_334868


namespace NUMINAMATH_CALUDE_alien_martian_limb_difference_l3348_334807

/-- Number of arms an Alien has -/
def alien_arms : ℕ := 3

/-- Number of legs an Alien has -/
def alien_legs : ℕ := 8

/-- Number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- Number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- Total number of limbs for an Alien -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- Total number of limbs for a Martian -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- Number of Aliens and Martians in the comparison -/
def group_size : ℕ := 5

theorem alien_martian_limb_difference :
  group_size * alien_limbs - group_size * martian_limbs = 5 := by
  sorry

end NUMINAMATH_CALUDE_alien_martian_limb_difference_l3348_334807


namespace NUMINAMATH_CALUDE_hcf_problem_l3348_334879

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 45276) (h2 : Nat.lcm a b = 2058) :
  Nat.gcd a b = 22 := by sorry

end NUMINAMATH_CALUDE_hcf_problem_l3348_334879


namespace NUMINAMATH_CALUDE_boxer_win_ratio_is_one_l3348_334897

/-- Represents a boxer's career statistics -/
structure BoxerStats where
  wins_before_first_loss : ℕ
  total_losses : ℕ
  win_loss_difference : ℕ

/-- Calculates the ratio of wins after first loss to wins before first loss -/
def win_ratio (stats : BoxerStats) : ℚ :=
  let wins_after_first_loss := stats.win_loss_difference + stats.total_losses - stats.wins_before_first_loss
  wins_after_first_loss / stats.wins_before_first_loss

/-- Theorem stating that for a boxer with given statistics, the win ratio is 1 -/
theorem boxer_win_ratio_is_one (stats : BoxerStats)
  (h1 : stats.wins_before_first_loss = 15)
  (h2 : stats.total_losses = 2)
  (h3 : stats.win_loss_difference = 28) :
  win_ratio stats = 1 := by
  sorry

end NUMINAMATH_CALUDE_boxer_win_ratio_is_one_l3348_334897


namespace NUMINAMATH_CALUDE_negative_numbers_l3348_334873

theorem negative_numbers (x y z : ℝ) 
  (h1 : 2 * x - y < 0) 
  (h2 : 3 * y - 2 * z < 0) 
  (h3 : 4 * z - 3 * x < 0) : 
  x < 0 ∧ y < 0 ∧ z < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_numbers_l3348_334873


namespace NUMINAMATH_CALUDE_ellipse_condition_l3348_334891

/-- Represents an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a ^ 2) ∧ n = 1 / (b ^ 2)

/-- The main theorem stating that m > n > 0 is necessary and sufficient for mx^2 + ny^2 = 1 
    to represent an ellipse with foci on the y-axis -/
theorem ellipse_condition (m n : ℝ) : 
  (m > n ∧ n > 0) ↔ is_ellipse_y_axis m n := by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3348_334891
