import Mathlib

namespace NUMINAMATH_CALUDE_sin_two_phi_l2464_246400

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l2464_246400


namespace NUMINAMATH_CALUDE_vector_collinearity_l2464_246499

/-- Given vectors a and b, prove that k makes k*a + b collinear with a - 3*b -/
theorem vector_collinearity (a b : ℝ × ℝ) (k : ℝ) : 
  a = (1, 2) →
  b = (-3, 2) →
  k = -1/3 →
  ∃ (t : ℝ), t • (k • a + b) = a - 3 • b := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2464_246499


namespace NUMINAMATH_CALUDE_next_perfect_cube_l2464_246436

/-- Given a perfect cube x, the next larger perfect cube is x + 3(∛x)² + 3∛x + 1 -/
theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^3) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m^3) ∧ y = x + 3 * (x^(1/3))^2 + 3 * x^(1/3) + 1 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_cube_l2464_246436


namespace NUMINAMATH_CALUDE_garden_fence_length_l2464_246428

/-- The length of a fence surrounding a square garden -/
def fence_length (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The length of the fence surrounding a square garden with side length 28 meters is 112 meters -/
theorem garden_fence_length :
  fence_length 28 = 112 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_length_l2464_246428


namespace NUMINAMATH_CALUDE_table_tennis_tournament_l2464_246440

theorem table_tennis_tournament (n : ℕ) (total_matches : ℕ) (withdrawn_players : ℕ) 
  (matches_per_withdrawn : ℕ) (h1 : n = 13) (h2 : total_matches = 50) 
  (h3 : withdrawn_players = 3) (h4 : matches_per_withdrawn = 2) : 
  (n.choose 2) - ((n - withdrawn_players).choose 2) - 
  (withdrawn_players * matches_per_withdrawn) = 1 := by
sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l2464_246440


namespace NUMINAMATH_CALUDE_cuboid_volume_l2464_246443

/-- Given a cuboid with perimeters of opposite faces A, B, and C, prove its volume is 240 cubic centimeters -/
theorem cuboid_volume (A B C : ℝ) (hA : A = 20) (hB : B = 32) (hC : C = 28) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  2 * (x + y) = A ∧
  2 * (y + z) = B ∧
  2 * (x + z) = C ∧
  x * y * z = 240 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l2464_246443


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l2464_246466

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  (x^5 + 5*x^3 + 2*x) = 384 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l2464_246466


namespace NUMINAMATH_CALUDE_tickets_sold_is_525_l2464_246467

/-- Represents the total number of tickets sold given ticket prices, total money collected, and number of general admission tickets. -/
def total_tickets_sold (student_price general_price total_collected general_tickets : ℕ) : ℕ :=
  let student_tickets := (total_collected - general_price * general_tickets) / student_price
  student_tickets + general_tickets

/-- Theorem stating that given the specific conditions, the total number of tickets sold is 525. -/
theorem tickets_sold_is_525 :
  total_tickets_sold 4 6 2876 388 = 525 := by
  sorry

end NUMINAMATH_CALUDE_tickets_sold_is_525_l2464_246467


namespace NUMINAMATH_CALUDE_friendly_function_fixed_point_l2464_246437

def FriendlyFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

theorem friendly_function_fixed_point
  (f : ℝ → ℝ)
  (h_friendly : FriendlyFunction f)
  (x₀ : ℝ)
  (h_x₀_in_range : x₀ ∈ Set.Icc 0 1)
  (h_fx₀_in_range : f x₀ ∈ Set.Icc 0 1)
  (h_ffx₀_eq_x₀ : f (f x₀) = x₀) :
  f x₀ = x₀ :=
by sorry

end NUMINAMATH_CALUDE_friendly_function_fixed_point_l2464_246437


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l2464_246461

theorem largest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 2 * c →
  5 * a = 2 * d →
  d = 480 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l2464_246461


namespace NUMINAMATH_CALUDE_gcd_660_924_l2464_246483

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end NUMINAMATH_CALUDE_gcd_660_924_l2464_246483


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l2464_246433

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions -/
theorem field_length_width_ratio :
  ∀ (w : ℝ),
  w > 0 →
  ∃ (k : ℕ), k > 0 ∧ 20 = k * w →
  25 = (1/8) * (20 * w) →
  (20 : ℝ) / w = 2 := by
sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l2464_246433


namespace NUMINAMATH_CALUDE_simplify_expression_l2464_246481

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : x/y + y/x - 3/(x*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2464_246481


namespace NUMINAMATH_CALUDE_inequality_solution_l2464_246442

theorem inequality_solution (x : ℝ) : 
  (x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0 ↔ 
  x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ x > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2464_246442


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_350_l2464_246408

theorem closest_perfect_square_to_350 :
  ∀ n : ℕ, n ≠ 19 → (n ^ 2 : ℝ) ≠ 361 → |350 - (19 ^ 2 : ℝ)| < |350 - (n ^ 2 : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_350_l2464_246408


namespace NUMINAMATH_CALUDE_correct_division_l2464_246445

theorem correct_division (dividend : ℕ) (incorrect_divisor correct_divisor incorrect_quotient : ℕ) 
  (h1 : incorrect_divisor = 72)
  (h2 : correct_divisor = 36)
  (h3 : incorrect_quotient = 24)
  (h4 : dividend = incorrect_divisor * incorrect_quotient) :
  dividend / correct_divisor = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l2464_246445


namespace NUMINAMATH_CALUDE_kangaroo_jumps_l2464_246486

theorem kangaroo_jumps (time_for_4_jumps : ℝ) (jumps_to_calculate : ℕ) : 
  time_for_4_jumps = 6 → jumps_to_calculate = 30 → 
  (time_for_4_jumps / 4) * jumps_to_calculate = 45 := by
sorry

end NUMINAMATH_CALUDE_kangaroo_jumps_l2464_246486


namespace NUMINAMATH_CALUDE_min_value_zero_l2464_246488

/-- The quadratic form as a function of x, y, and k -/
def f (x y k : ℝ) : ℝ :=
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9

/-- The theorem stating that 3/2 is the value of k that makes the minimum of f zero -/
theorem min_value_zero (k : ℝ) : 
  (∀ x y : ℝ, f x y k ≥ 0) ∧ (∃ x y : ℝ, f x y k = 0) ↔ k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_zero_l2464_246488


namespace NUMINAMATH_CALUDE_cost_price_calculation_article_cost_price_l2464_246448

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem article_cost_price : 
  cost_price_calculation 15000 0.1 0.08 = 12500 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_article_cost_price_l2464_246448


namespace NUMINAMATH_CALUDE_difference_of_a_and_reciprocal_l2464_246492

theorem difference_of_a_and_reciprocal (a : ℝ) (h : a + 1/a = Real.sqrt 13) :
  a - 1/a = 3 ∨ a - 1/a = -3 := by
sorry

end NUMINAMATH_CALUDE_difference_of_a_and_reciprocal_l2464_246492


namespace NUMINAMATH_CALUDE_last_digit_of_max_value_l2464_246441

/-- Represents the operation of replacing two numbers with their product plus one -/
def combine (a b : ℕ) : ℕ := a * b + 1

/-- The maximum value after performing the combine operation 127 times on 128 ones -/
def max_final_value : ℕ := sorry

/-- The problem statement -/
theorem last_digit_of_max_value :
  (max_final_value % 10) = 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_max_value_l2464_246441


namespace NUMINAMATH_CALUDE_percentage_problem_l2464_246497

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2464_246497


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2464_246450

/-- Proves that given a simple interest of 4052.25, an annual interest rate of 9%,
    and a time period of 5 years, the principal sum is 9005. -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) 
    (h1 : interest = 4052.25)
    (h2 : rate = 9)
    (h3 : time = 5)
    (h4 : principal = interest / (rate * time / 100)) :
  principal = 9005 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2464_246450


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l2464_246462

theorem rationalize_sqrt_five_eighteenths : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l2464_246462


namespace NUMINAMATH_CALUDE_max_profit_at_180_l2464_246401

-- Define the selling price x and daily sales y
variable (x y : ℝ)

-- Define the cost price
def cost_price : ℝ := 80

-- Define the range of selling price
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the relationship between y and x
def sales_function (x : ℝ) : ℝ := -0.5 * x + 160

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (sales_function x)

-- Theorem statement
theorem max_profit_at_180 :
  ∀ x, selling_price_range x →
    profit_function x ≤ profit_function 180 ∧
    profit_function 180 = 7000 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_180_l2464_246401


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2464_246431

theorem smallest_three_digit_congruence :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m < n → (75 * m) % 345 ≠ 225) ∧
    n = 118 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2464_246431


namespace NUMINAMATH_CALUDE_sarahs_job_men_degree_percentage_l2464_246421

/-- Calculates the percentage of men with a college degree -/
def percentage_men_with_degree (total_employees : ℕ) (women_percentage : ℚ) 
  (num_women : ℕ) (men_without_degree : ℕ) : ℚ :=
  let num_men := total_employees - num_women
  let men_with_degree := num_men - men_without_degree
  (men_with_degree : ℚ) / (num_men : ℚ) * 100

/-- The percentage of men with a college degree at Sarah's job is 75% -/
theorem sarahs_job_men_degree_percentage :
  ∃ (total_employees : ℕ),
    (48 : ℚ) / (total_employees : ℚ) = (60 : ℚ) / 100 ∧
    percentage_men_with_degree total_employees ((60 : ℚ) / 100) 48 8 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_job_men_degree_percentage_l2464_246421


namespace NUMINAMATH_CALUDE_clock_cost_price_l2464_246493

/-- The cost price of each clock -/
def cost_price : ℝ := 125

/-- The number of clocks purchased -/
def total_clocks : ℕ := 150

/-- The number of clocks sold at 12% gain -/
def clocks_12_percent : ℕ := 60

/-- The number of clocks sold at 18% gain -/
def clocks_18_percent : ℕ := 90

/-- The gain percentage for the first group of clocks -/
def gain_12_percent : ℝ := 0.12

/-- The gain percentage for the second group of clocks -/
def gain_18_percent : ℝ := 0.18

/-- The uniform gain percentage -/
def uniform_gain : ℝ := 0.16

/-- The difference in total selling price -/
def price_difference : ℝ := 75

theorem clock_cost_price : 
  (clocks_12_percent : ℝ) * cost_price * (1 + gain_12_percent) + 
  (clocks_18_percent : ℝ) * cost_price * (1 + gain_18_percent) = 
  (total_clocks : ℝ) * cost_price * (1 + uniform_gain) + price_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l2464_246493


namespace NUMINAMATH_CALUDE_water_depth_relation_l2464_246407

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the water depth when the tank is upright -/
def uprightDepth (tank : WaterTank) (horizontalDepth : ℝ) : ℝ :=
  sorry

/-- Theorem stating the relation between horizontal and upright water depths -/
theorem water_depth_relation (tank : WaterTank) (horizontalDepth : ℝ) :
  tank.height = 20 →
  tank.baseDiameter = 5 →
  horizontalDepth = 4 →
  abs (uprightDepth tank horizontalDepth - 8.1) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_relation_l2464_246407


namespace NUMINAMATH_CALUDE_tablet_charge_time_proof_l2464_246454

/-- Time in minutes to fully charge a smartphone -/
def smartphone_charge_time : ℕ := 26

/-- Time in minutes to fully charge a tablet -/
def tablet_charge_time : ℕ := 53

/-- The total time taken to charge half a smartphone and a full tablet -/
def total_charge_time : ℕ := 66

/-- Theorem stating that the time to fully charge a tablet is 53 minutes -/
theorem tablet_charge_time_proof :
  tablet_charge_time = total_charge_time - (smartphone_charge_time / 2) :=
by sorry

end NUMINAMATH_CALUDE_tablet_charge_time_proof_l2464_246454


namespace NUMINAMATH_CALUDE_board_highest_point_l2464_246458

/-- Represents a rectangular board with length and height -/
structure Board where
  length : ℝ
  height : ℝ

/-- Calculates the distance from the ground to the highest point of an inclined board -/
def highestPoint (board : Board) (angle : ℝ) : ℝ :=
  sorry

theorem board_highest_point :
  let board := Board.mk 64 4
  let angle := 30 * π / 180
  ∃ (a b c : ℕ), 
    (highestPoint board angle = a + b * Real.sqrt c) ∧
    (a = 32) ∧ (b = 2) ∧ (c = 3) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c)) :=
  sorry

end NUMINAMATH_CALUDE_board_highest_point_l2464_246458


namespace NUMINAMATH_CALUDE_original_purchase_cups_l2464_246460

/-- The cost of a single paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of a single paper cup -/
def cup_cost : ℝ := sorry

/-- The number of paper cups in the original purchase -/
def num_cups : ℕ := sorry

/-- The total cost of 100 paper plates and some paper cups is $6.00 -/
axiom total_cost : 100 * plate_cost + num_cups * cup_cost = 6

/-- The total cost of 20 plates and 40 cups is $1.20 -/
axiom partial_cost : 20 * plate_cost + 40 * cup_cost = 1.2

theorem original_purchase_cups : num_cups = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_purchase_cups_l2464_246460


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l2464_246432

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l2464_246432


namespace NUMINAMATH_CALUDE_largest_squared_fraction_l2464_246415

theorem largest_squared_fraction : 
  let a := (8/9 : ℚ)^2
  let b := (2/3 : ℚ)^2
  let c := (3/4 : ℚ)^2
  let d := (5/8 : ℚ)^2
  let e := (7/12 : ℚ)^2
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end NUMINAMATH_CALUDE_largest_squared_fraction_l2464_246415


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2464_246491

/-- Represents a 2 × 2 × 2 cube -/
structure Cube where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- The sum of areas of all triangles with vertices on the cube -/
def sum_triangle_areas (c : Cube) : ℝ := sorry

/-- The sum can be expressed as m + √n + √p -/
def sum_representation (m n p : ℕ) (c : Cube) : Prop :=
  sum_triangle_areas c = m + Real.sqrt n + Real.sqrt p

theorem cube_triangle_areas_sum (c : Cube) :
  ∃ (m n p : ℕ), sum_representation m n p c ∧ m + n + p = 5424 := by sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2464_246491


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2464_246419

theorem count_negative_numbers : ∃ (S : Finset ℝ), 
  S = {8, 0, |(-2)|, -5, -2/3, (-1)^2} ∧ 
  (S.filter (λ x => x < 0)).card = 2 := by
sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2464_246419


namespace NUMINAMATH_CALUDE_a_range_l2464_246469

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {a + 2}

theorem a_range (a : ℝ) : A ∩ B a = ∅ → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2464_246469


namespace NUMINAMATH_CALUDE_decagon_angle_property_l2464_246444

theorem decagon_angle_property (n : ℕ) : 
  (n - 2) * 180 = 360 * 4 ↔ n = 10 := by sorry

end NUMINAMATH_CALUDE_decagon_angle_property_l2464_246444


namespace NUMINAMATH_CALUDE_max_sphere_surface_area_from_cube_l2464_246418

/-- Given a cube with side length 2, the maximum surface area of a sphere carved from this cube is 4π. -/
theorem max_sphere_surface_area_from_cube (cube_side_length : ℝ) (sphere_surface_area : ℝ → ℝ) :
  cube_side_length = 2 →
  (∀ r : ℝ, r ≤ 1 → sphere_surface_area r ≤ sphere_surface_area 1) →
  sphere_surface_area 1 = 4 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_max_sphere_surface_area_from_cube_l2464_246418


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l2464_246409

/-- Represents the average number of visitors on a given day type -/
structure VisitorAverage where
  sunday : ℕ
  other : ℕ

/-- Represents a month with visitor data -/
structure Month where
  days : ℕ
  startsWithSunday : Bool
  avgVisitorsPerDay : ℕ
  visitorAvg : VisitorAverage

/-- Calculates the number of Sundays in a month -/
def numSundays (m : Month) : ℕ :=
  if m.startsWithSunday then
    (m.days + 6) / 7
  else
    m.days / 7

/-- Theorem: Given the conditions, prove that the average number of visitors on non-Sunday days is 80 -/
theorem library_visitors_theorem (m : Month) 
    (h1 : m.days = 30)
    (h2 : m.startsWithSunday = true)
    (h3 : m.visitorAvg.sunday = 140)
    (h4 : m.avgVisitorsPerDay = 90) :
    m.visitorAvg.other = 80 := by
  sorry


end NUMINAMATH_CALUDE_library_visitors_theorem_l2464_246409


namespace NUMINAMATH_CALUDE_model_M_completion_time_l2464_246485

/-- The time (in minutes) taken by a model N computer to complete the task -/
def model_N_time : ℝ := 18

/-- The number of model M computers used -/
def num_model_M : ℝ := 12

/-- The time (in minutes) taken to complete the task when using both models -/
def total_time : ℝ := 1

/-- The time (in minutes) taken by a model M computer to complete the task -/
def model_M_time : ℝ := 36

theorem model_M_completion_time :
  (num_model_M / model_M_time + num_model_M / model_N_time) * total_time = num_model_M :=
by sorry

end NUMINAMATH_CALUDE_model_M_completion_time_l2464_246485


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2464_246417

/-- Proves that given a 200-gallon tank filled with two types of fuel,
    where one contains 12% ethanol and the other 16% ethanol,
    if the full tank contains 30 gallons of ethanol,
    then the volume of the first fuel added is 50 gallons. -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) 
    (total_ethanol : ℝ) (fuel_a : ℝ) :
  tank_capacity = 200 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  fuel_a * ethanol_a + (tank_capacity - fuel_a) * ethanol_b = total_ethanol →
  fuel_a = 50 := by
sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2464_246417


namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l2464_246463

def hours_before_midterms (
  chess_hours_per_week : ℕ)
  (drama_hours_per_week : ℕ)
  (glee_hours_odd_week : ℕ)
  (robotics_hours_even_week : ℕ)
  (soccer_hours_odd_week : ℕ)
  (soccer_hours_even_week : ℕ)
  (weeks_in_semester : ℕ)
  (sick_weeks : ℕ)
  (midterm_week : ℕ)
  (drama_cancel_week : ℕ)
  (holiday_week : ℕ)
  (holiday_soccer_hours : ℕ) : ℕ :=
  -- Function body
  sorry

theorem annie_extracurricular_hours :
  hours_before_midterms 2 8 3 4 1 2 12 2 8 5 7 1 = 81 :=
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l2464_246463


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_8_11_24_l2464_246477

theorem smallest_number_divisible_by_8_11_24 :
  ∃ (k : ℕ), 255 + k > 255 ∧ (255 + k) % 8 = 0 ∧ (255 + k) % 11 = 0 ∧ (255 + k) % 24 = 0 ∧
  ∀ (n : ℕ), n < 255 → ¬∃ (m : ℕ), m > 0 ∧ (n + m) % 8 = 0 ∧ (n + m) % 11 = 0 ∧ (n + m) % 24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_8_11_24_l2464_246477


namespace NUMINAMATH_CALUDE_yellow_second_draw_probability_l2464_246429

/-- Represents the total number of ping-pong balls -/
def total_balls : ℕ := 10

/-- Represents the number of yellow balls -/
def yellow_balls : ℕ := 6

/-- Represents the number of white balls -/
def white_balls : ℕ := 4

/-- Represents the number of draws -/
def num_draws : ℕ := 2

/-- Calculates the probability of drawing a yellow ball on the second draw -/
def prob_yellow_second_draw : ℚ :=
  (white_balls : ℚ) / total_balls * yellow_balls / (total_balls - 1)

theorem yellow_second_draw_probability :
  prob_yellow_second_draw = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_yellow_second_draw_probability_l2464_246429


namespace NUMINAMATH_CALUDE_village_population_equality_l2464_246484

/-- The initial population of Village X -/
def initial_population_X : ℕ := 72000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The initial population of Village Y -/
def initial_population_Y : ℕ := 42000

/-- The yearly increase in population of Village Y -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations are equal -/
def years : ℕ := 15

theorem village_population_equality :
  initial_population_X - (decrease_rate_X * years) =
  initial_population_Y + (increase_rate_Y * years) :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_l2464_246484


namespace NUMINAMATH_CALUDE_levi_additional_baskets_l2464_246427

/-- Calculates the number of additional baskets Levi needs to score to beat his brother by at least the given margin. -/
def additional_baskets_needed (levi_initial : ℕ) (brother_initial : ℕ) (brother_additional : ℕ) (margin : ℕ) : ℕ :=
  (brother_initial + brother_additional + margin) - levi_initial

/-- Proves that Levi needs to score 12 more times to beat his brother by at least 5 baskets. -/
theorem levi_additional_baskets : 
  additional_baskets_needed 8 12 3 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_levi_additional_baskets_l2464_246427


namespace NUMINAMATH_CALUDE_min_value_abc_l2464_246478

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 ∧ 
  (a + 3 * b + 9 * c = 27 ↔ a = 9 ∧ b = 3 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l2464_246478


namespace NUMINAMATH_CALUDE_number_problem_l2464_246456

theorem number_problem : ∃ n : ℝ, 8 * n - 4 = 17 ∧ n = 2.625 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2464_246456


namespace NUMINAMATH_CALUDE_students_needed_to_fill_buses_l2464_246459

theorem students_needed_to_fill_buses (total_students : ℕ) (bus_capacity : ℕ) : 
  total_students = 254 → bus_capacity = 30 → 
  (((total_students + 16) / bus_capacity : ℕ) * bus_capacity = total_students + 16) ∧
  (((total_students + 15) / bus_capacity : ℕ) * bus_capacity < total_students + 15) := by
  sorry


end NUMINAMATH_CALUDE_students_needed_to_fill_buses_l2464_246459


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l2464_246426

theorem angle_sum_around_point (x : ℝ) : 
  150 + 90 + x + 90 = 360 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l2464_246426


namespace NUMINAMATH_CALUDE_hare_hunt_probability_l2464_246457

theorem hare_hunt_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 3/5) 
  (h2 : p2 = 3/10) 
  (h3 : p3 = 1/10) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 0.748 := by
  sorry

end NUMINAMATH_CALUDE_hare_hunt_probability_l2464_246457


namespace NUMINAMATH_CALUDE_red_paint_cans_l2464_246434

theorem red_paint_cans (total_cans : ℕ) (red_ratio white_ratio : ℕ) 
  (h1 : total_cans = 35)
  (h2 : red_ratio = 4)
  (h3 : white_ratio = 3) : 
  (red_ratio : ℚ) / (red_ratio + white_ratio : ℚ) * total_cans = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_red_paint_cans_l2464_246434


namespace NUMINAMATH_CALUDE_expand_expression_l2464_246430

theorem expand_expression (a b : ℝ) : (a - 2) * (a - 2*b) = a^2 - 2*a + 4*b := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2464_246430


namespace NUMINAMATH_CALUDE_sqrt_ratio_equation_l2464_246468

theorem sqrt_ratio_equation (x : ℝ) :
  (Real.sqrt (2 * x + 7) / Real.sqrt (4 * x + 7) = Real.sqrt 7 / 2) →
  x = -21 / 20 := by
sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equation_l2464_246468


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_product_l2464_246470

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has unique digits -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The main theorem -/
theorem smallest_sum_of_digits_product :
  ∃ (x y : ℕ),
    is_two_digit x ∧
    is_two_digit y ∧
    has_unique_digits (x * 100 + y) ∧
    (x * y ≥ 1000) ∧
    (x * y < 10000) ∧
    sum_of_digits (x * y) = 12 ∧
    ∀ (a b : ℕ),
      is_two_digit a →
      is_two_digit b →
      has_unique_digits (a * 100 + b) →
      (a * b ≥ 1000) →
      (a * b < 10000) →
      sum_of_digits (a * b) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_product_l2464_246470


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2464_246474

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2464_246474


namespace NUMINAMATH_CALUDE_gcd_triple_existence_l2464_246446

theorem gcd_triple_existence (S : Set ℕ+) 
  (h_infinite : Set.Infinite S)
  (h_distinct_gcd : ∃ (a b c d : ℕ+), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Nat.gcd a b ≠ Nat.gcd c d) :
  ∃ (x y z : ℕ+), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x y = Nat.gcd y z ∧ Nat.gcd y z ≠ Nat.gcd z x :=
by
  sorry

end NUMINAMATH_CALUDE_gcd_triple_existence_l2464_246446


namespace NUMINAMATH_CALUDE_unique_solution_3_and_7_equation_l2464_246490

theorem unique_solution_3_and_7_equation :
  ∀ a y : ℕ, a ≥ 1 → y ≥ 1 →
  (3 ^ (2 * a - 1) + 3 ^ a + 1 = 7 ^ y) →
  (a = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_3_and_7_equation_l2464_246490


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2464_246405

theorem sqrt_equation_solutions (x : ℝ) :
  Real.sqrt ((2 + Real.sqrt 5) ^ x) + Real.sqrt ((2 - Real.sqrt 5) ^ x) = 6 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2464_246405


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2464_246420

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 4 * x^2 + (k - 1) * x + 9 = (a * x + b)^2) → 
  (k = 13 ∨ k = -11) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2464_246420


namespace NUMINAMATH_CALUDE_product_of_fractions_l2464_246479

theorem product_of_fractions : (2 : ℚ) / 3 * (3 : ℚ) / 8 = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2464_246479


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2464_246495

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) - 1
  f (-1) = 0 ∧ ∀ x : ℝ, f x = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2464_246495


namespace NUMINAMATH_CALUDE_slope_of_right_triangle_l2464_246416

/-- Given a right triangle ABC in the x-y plane where ∠B = 90°, AC = 100, and AB = 80,
    the slope of AC is 4/3 -/
theorem slope_of_right_triangle (A B C : ℝ × ℝ) : 
  (B.2 - A.2) ^ 2 + (B.1 - A.1) ^ 2 = 80 ^ 2 →
  (C.2 - A.2) ^ 2 + (C.1 - A.1) ^ 2 = 100 ^ 2 →
  (C.2 - B.2) ^ 2 + (C.1 - B.1) ^ 2 = (C.2 - A.2) ^ 2 + (C.1 - A.1) ^ 2 - (B.2 - A.2) ^ 2 - (B.1 - A.1) ^ 2 →
  (C.2 - A.2) / (C.1 - A.1) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_right_triangle_l2464_246416


namespace NUMINAMATH_CALUDE_circle_condition_l2464_246404

/-- The equation of a potential circle -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + x + 2*m*y + m^2 + m - 1 = 0

/-- Theorem stating the condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2) ↔
  m < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l2464_246404


namespace NUMINAMATH_CALUDE_athletes_total_yards_l2464_246455

-- Define the athletes and their performances
def malik_yards_per_game : ℕ := 18
def malik_games : ℕ := 5

def josiah_yards_per_game : ℕ := 22
def josiah_games : ℕ := 7

def darnell_yards_per_game : ℕ := 11
def darnell_games : ℕ := 4

def kade_yards_per_game : ℕ := 15
def kade_games : ℕ := 6

-- Define the total yards function
def total_yards : ℕ := 
  malik_yards_per_game * malik_games + 
  josiah_yards_per_game * josiah_games + 
  darnell_yards_per_game * darnell_games + 
  kade_yards_per_game * kade_games

-- Theorem statement
theorem athletes_total_yards : total_yards = 378 := by
  sorry

end NUMINAMATH_CALUDE_athletes_total_yards_l2464_246455


namespace NUMINAMATH_CALUDE_jar_marbles_l2464_246403

theorem jar_marbles (a b c : ℕ) : 
  b = a + 12 →
  c = 2 * b →
  a + b + c = 148 →
  a = 28 := by sorry

end NUMINAMATH_CALUDE_jar_marbles_l2464_246403


namespace NUMINAMATH_CALUDE_administrative_staff_sample_size_l2464_246413

/-- Represents the number of administrative staff to be drawn in a stratified sample -/
def administrative_staff_in_sample (total_population : ℕ) (sample_size : ℕ) (administrative_staff : ℕ) : ℕ :=
  (administrative_staff * sample_size) / total_population

/-- Theorem stating that the number of administrative staff to be drawn is 4 -/
theorem administrative_staff_sample_size :
  administrative_staff_in_sample 160 20 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_administrative_staff_sample_size_l2464_246413


namespace NUMINAMATH_CALUDE_odd_sum_of_cubes_not_both_even_l2464_246425

theorem odd_sum_of_cubes_not_both_even (n m : ℤ) 
  (h : Odd (n^3 + m^3)) : ¬(Even n ∧ Even m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_cubes_not_both_even_l2464_246425


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2464_246475

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A predicate that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem unique_three_digit_number : 
  ∃! n : ℕ, isThreeDigit n ∧ n = 12 * sumOfDigits n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2464_246475


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2464_246414

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (a b : Line) 
  (α β : Plane) 
  (h_diff_lines : a ≠ b) 
  (h_diff_planes : α ≠ β) 
  (h1 : perp_line a b) 
  (h2 : perp_line_plane a α) 
  (h3 : perp_line_plane b β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2464_246414


namespace NUMINAMATH_CALUDE_subtract_squares_l2464_246453

theorem subtract_squares (a : ℝ) : 3 * a^2 - a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_squares_l2464_246453


namespace NUMINAMATH_CALUDE_journey_feasibility_l2464_246476

/-- Proves that a journey can be completed in the given time at the given average speed -/
theorem journey_feasibility 
  (total_distance : ℝ) 
  (segment1 : ℝ) 
  (segment2 : ℝ) 
  (total_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = segment1 + segment2)
  (h2 : total_distance = 693)
  (h3 : segment1 = 420)
  (h4 : segment2 = 273)
  (h5 : total_time = 11)
  (h6 : average_speed = 63)
  : total_distance / average_speed = total_time :=
by sorry

#check journey_feasibility

end NUMINAMATH_CALUDE_journey_feasibility_l2464_246476


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2464_246439

theorem quadratic_inequality_range :
  (∀ x : ℝ, ∀ a : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔
  (a ∈ Set.Ioc (-2 : ℝ) 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2464_246439


namespace NUMINAMATH_CALUDE_green_shirt_cost_l2464_246422

-- Define the number of students in each grade
def kindergartners : ℕ := 101
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

-- Define the cost of shirts for each grade (in cents to avoid floating-point issues)
def orange_shirt_cost : ℕ := 580
def yellow_shirt_cost : ℕ := 500
def blue_shirt_cost : ℕ := 560

-- Define the total amount spent on all shirts (in cents)
def total_spent : ℕ := 231700

-- Theorem to prove
theorem green_shirt_cost :
  (total_spent - 
   (kindergartners * orange_shirt_cost + 
    first_graders * yellow_shirt_cost + 
    second_graders * blue_shirt_cost)) / third_graders = 525 := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_cost_l2464_246422


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l2464_246447

theorem permutation_combination_equality (n : ℕ) : 
  n * (n - 1) * (n - 2) = 6 * (n * (n - 1) * (n - 2) * (n - 3)) / 24 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l2464_246447


namespace NUMINAMATH_CALUDE_convex_polygon_20_sides_diagonals_l2464_246406

/-- A convex polygon is a polygon in which every interior angle is less than 180 degrees. -/
def ConvexPolygon (n : ℕ) : Prop := sorry

/-- A diagonal of a convex polygon is a line segment that connects two non-adjacent vertices. -/
def Diagonal (n : ℕ) : Prop := sorry

/-- The number of diagonals in a convex polygon with n sides. -/
def NumDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem convex_polygon_20_sides_diagonals :
  ∀ p : ConvexPolygon 20, NumDiagonals 20 = 170 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_20_sides_diagonals_l2464_246406


namespace NUMINAMATH_CALUDE_nicki_total_miles_l2464_246487

/-- Represents the number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- Represents the number of miles Nicki ran per week in the first half of the year -/
def first_half_miles_per_week : ℕ := 20

/-- Represents the number of miles Nicki ran per week in the second half of the year -/
def second_half_miles_per_week : ℕ := 30

/-- Calculates the total miles Nicki ran for the year -/
def total_miles_run : ℕ := 
  (first_half_miles_per_week * (weeks_in_year / 2)) + 
  (second_half_miles_per_week * (weeks_in_year / 2))

/-- Theorem stating that Nicki ran 1300 miles in total for the year -/
theorem nicki_total_miles : total_miles_run = 1300 := by
  sorry

end NUMINAMATH_CALUDE_nicki_total_miles_l2464_246487


namespace NUMINAMATH_CALUDE_paint_per_statue_l2464_246471

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) : 
  total_paint = 7/8 ∧ num_statues = 14 → 
  total_paint / num_statues = 7/112 := by
sorry

end NUMINAMATH_CALUDE_paint_per_statue_l2464_246471


namespace NUMINAMATH_CALUDE_staircase_steps_l2464_246412

theorem staircase_steps (x : ℤ) 
  (h1 : x % 2 = 1)
  (h2 : x % 3 = 2)
  (h3 : x % 4 = 3)
  (h4 : x % 5 = 4)
  (h5 : x % 6 = 5)
  (h6 : x % 7 = 0) :
  ∃ k : ℤ, x = 119 + 420 * k := by
sorry

end NUMINAMATH_CALUDE_staircase_steps_l2464_246412


namespace NUMINAMATH_CALUDE_power_of_product_square_l2464_246480

theorem power_of_product_square (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_square_l2464_246480


namespace NUMINAMATH_CALUDE_two_and_one_third_of_eighteen_is_fortytwo_l2464_246424

theorem two_and_one_third_of_eighteen_is_fortytwo : 
  (7 : ℚ) / 3 * 18 = 42 := by sorry

end NUMINAMATH_CALUDE_two_and_one_third_of_eighteen_is_fortytwo_l2464_246424


namespace NUMINAMATH_CALUDE_morning_campers_l2464_246482

theorem morning_campers (total : ℕ) (afternoon : ℕ) (morning : ℕ) : 
  total = 62 → afternoon = 27 → morning = total - afternoon → morning = 35 := by
  sorry

end NUMINAMATH_CALUDE_morning_campers_l2464_246482


namespace NUMINAMATH_CALUDE_simplify_expression_l2464_246472

theorem simplify_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + 2 * Real.sqrt (x * y) + y) / (Real.sqrt x + Real.sqrt y) - 
  (Real.sqrt (x * y) + Real.sqrt x) * Real.sqrt (1 / x) = Real.sqrt x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2464_246472


namespace NUMINAMATH_CALUDE_max_xy_value_l2464_246464

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 5 * y = 200) : x * y ≤ 285 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2464_246464


namespace NUMINAMATH_CALUDE_angle_measure_theorem_l2464_246438

theorem angle_measure_theorem (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_theorem_l2464_246438


namespace NUMINAMATH_CALUDE_hat_number_sum_l2464_246496

theorem hat_number_sum : ∀ (alice_num bob_num : ℕ),
  alice_num ∈ Finset.range 51 →
  bob_num ∈ Finset.range 51 →
  alice_num ≠ bob_num →
  (∃ (x : ℕ), alice_num < x ∧ x ≤ 50) →
  (∃ (y : ℕ), y < bob_num ∧ y ≤ 50) →
  bob_num % 3 = 0 →
  ∃ (k : ℕ), 2 * bob_num + alice_num = k^2 →
  alice_num + bob_num = 22 :=
by sorry

end NUMINAMATH_CALUDE_hat_number_sum_l2464_246496


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_four_l2464_246494

theorem sqrt_sum_equals_four : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_four_l2464_246494


namespace NUMINAMATH_CALUDE_park_maple_trees_l2464_246452

/-- The number of maple trees in the park after planting -/
def total_maple_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 11 maple trees after planting -/
theorem park_maple_trees :
  let initial_trees : ℕ := 2
  let trees_to_plant : ℕ := 9
  total_maple_trees initial_trees trees_to_plant = 11 := by
  sorry

end NUMINAMATH_CALUDE_park_maple_trees_l2464_246452


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l2464_246489

/-- Proves that the contractor was engaged for 20 days given the problem conditions --/
theorem contractor_engagement_days : 
  ∀ (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) (absent_days : ℕ),
    daily_wage = 25 →
    daily_fine = (15/2) →
    total_amount = 425 →
    absent_days = 10 →
    ∃ (engaged_days : ℕ), 
      engaged_days * daily_wage - absent_days * daily_fine = total_amount ∧
      engaged_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_days_l2464_246489


namespace NUMINAMATH_CALUDE_divisibility_inequality_l2464_246411

theorem divisibility_inequality (a b c d e f : ℕ) 
  (h_f_lt_a : f < a)
  (h_div_c : ∃ k : ℕ, a * b * d + 1 = k * c)
  (h_div_b : ∃ l : ℕ, a * c * e + 1 = l * b)
  (h_div_a : ∃ m : ℕ, b * c * f + 1 = m * a)
  (h_ineq : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by sorry

end NUMINAMATH_CALUDE_divisibility_inequality_l2464_246411


namespace NUMINAMATH_CALUDE_area_R_specific_rhombus_l2464_246498

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region R in the rhombus -/
def area_R (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of region R in the specific rhombus -/
theorem area_R_specific_rhombus :
  let r : Rhombus := { side_length := 3, angle_B := 150 * π / 180 }
  area_R r = 9 * (Real.sqrt 6 - Real.sqrt 2) / 8 := by
    sorry

end NUMINAMATH_CALUDE_area_R_specific_rhombus_l2464_246498


namespace NUMINAMATH_CALUDE_friend_game_l2464_246435

theorem friend_game (a b c d : ℕ) : 
  3^a * 7^b = 3 * 7 ∧ 3^c * 7^d = 3 * 7 → (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by sorry

end NUMINAMATH_CALUDE_friend_game_l2464_246435


namespace NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2464_246451

def three_digit_palindrome (a b : ℕ) : ℕ := 100 * a + 10 * b + a

theorem gcf_three_digit_palindromes :
  ∃ (gcf : ℕ), 
    gcf > 0 ∧
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → gcf ∣ three_digit_palindrome a b) ∧
    (∀ (d : ℕ), d > 0 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → d ∣ three_digit_palindrome a b) → d ≤ gcf) ∧
    gcf = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2464_246451


namespace NUMINAMATH_CALUDE_product_of_three_terms_l2464_246465

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem product_of_three_terms 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 5 = 4) : 
  a 4 * a 5 * a 6 = 64 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_terms_l2464_246465


namespace NUMINAMATH_CALUDE_square_not_prime_plus_square_l2464_246410

theorem square_not_prime_plus_square (n : ℕ) (h1 : n ≥ 5) (h2 : n % 3 = 2) :
  ¬ ∃ (p k : ℕ), Prime p ∧ n^2 = p + k^2 := by
sorry

end NUMINAMATH_CALUDE_square_not_prime_plus_square_l2464_246410


namespace NUMINAMATH_CALUDE_speed_to_achieve_average_l2464_246402

/-- Given a person driving at two different speeds over two time periods, 
    this theorem proves the required speed for the second period to achieve a specific average speed. -/
theorem speed_to_achieve_average 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (additional_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : initial_speed = 60) 
  (h2 : initial_time = 3) 
  (h3 : additional_time = 2) 
  (h4 : average_speed = 70) : 
  ∃ x : ℝ, 
    (initial_speed * initial_time + x * additional_time) / (initial_time + additional_time) = average_speed 
    ∧ x = 85 := by
  sorry

end NUMINAMATH_CALUDE_speed_to_achieve_average_l2464_246402


namespace NUMINAMATH_CALUDE_cricket_game_overs_l2464_246423

theorem cricket_game_overs (total_target : ℝ) (initial_rate : ℝ) (remaining_overs : ℝ) (required_rate : ℝ) 
  (h1 : total_target = 282)
  (h2 : initial_rate = 3.6)
  (h3 : remaining_overs = 40)
  (h4 : required_rate = 6.15) :
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * required_rate = total_target ∧ 
    initial_overs = 10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l2464_246423


namespace NUMINAMATH_CALUDE_synthetic_analytic_direct_l2464_246473

-- Define proof methods
structure ProofMethod where
  name : String
  direction : String
  isDirect : Bool

-- Define synthetic and analytic methods
def synthetic : ProofMethod := {
  name := "Synthetic",
  direction := "cause to effect",
  isDirect := true
}

def analytic : ProofMethod := {
  name := "Analytic",
  direction := "effect to cause",
  isDirect := true
}

-- Theorem statement
theorem synthetic_analytic_direct :
  synthetic.isDirect ∧ analytic.isDirect :=
sorry

end NUMINAMATH_CALUDE_synthetic_analytic_direct_l2464_246473


namespace NUMINAMATH_CALUDE_smallest_prime_between_squares_l2464_246449

theorem smallest_prime_between_squares : ∃ (p : ℕ), 
  Prime p ∧ 
  (∃ (n : ℕ), p = n^2 + 6) ∧ 
  (∃ (m : ℕ), p = (m+1)^2 - 9) ∧
  (∀ (q : ℕ), q < p → 
    (Prime q → ¬(∃ (k : ℕ), q = k^2 + 6 ∧ q = (k+1)^2 - 9))) ∧
  p = 127 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_between_squares_l2464_246449
