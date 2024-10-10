import Mathlib

namespace hyperbola_eccentricity_l3653_365355

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The ratio of y to x for the asymptotes -/
  asymptote_slope : ℝ
  /-- The hyperbola has foci on the x-axis -/
  foci_on_x_axis : Bool

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_slope : h.asymptote_slope = 2/3) 
  (h_foci : h.foci_on_x_axis = true) : 
  eccentricity h = Real.sqrt 13 / 3 := by sorry

end hyperbola_eccentricity_l3653_365355


namespace inequality_solution_l3653_365323

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x ≥ 3 }
  else if a < 0 then { x | x ≥ 3 ∨ x ≤ 2/a }
  else if 0 < a ∧ a < 2/3 then { x | 3 ≤ x ∧ x ≤ 2/a }
  else if a = 2/3 then { x | x = 3 }
  else { x | 2/a ≤ x ∧ x ≤ 3 }

theorem inequality_solution (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 - (3*a + 2) * x + 6 ≤ 0 :=
by sorry

end inequality_solution_l3653_365323


namespace slope_intercept_form_through_points_l3653_365301

/-- Slope-intercept form of a line passing through two points -/
theorem slope_intercept_form_through_points
  (x₁ y₁ x₂ y₂ : ℚ)
  (h₁ : x₁ = -3)
  (h₂ : y₁ = 7)
  (h₃ : x₂ = 4)
  (h₄ : y₂ = -2)
  : ∃ (m b : ℚ), m = -9/7 ∧ b = 22/7 ∧ ∀ x y, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
by sorry

end slope_intercept_form_through_points_l3653_365301


namespace eighteen_is_seventyfive_percent_of_twentyfour_l3653_365353

theorem eighteen_is_seventyfive_percent_of_twentyfour (x : ℝ) : 
  18 = 0.75 * x → x = 24 := by
  sorry

end eighteen_is_seventyfive_percent_of_twentyfour_l3653_365353


namespace total_distance_l3653_365317

def road_trip (tracy_miles michelle_miles katie_miles : ℕ) : Prop :=
  tracy_miles = 2 * michelle_miles + 20 ∧
  michelle_miles = 3 * katie_miles ∧
  michelle_miles = 294

theorem total_distance (tracy_miles michelle_miles katie_miles : ℕ) 
  (h : road_trip tracy_miles michelle_miles katie_miles) : 
  tracy_miles + michelle_miles + katie_miles = 1000 := by
  sorry

end total_distance_l3653_365317


namespace average_of_first_16_even_divisible_by_5_l3653_365378

def first_16_even_divisible_by_5 : List Nat :=
  List.range 16 |> List.map (fun n => 10 * (n + 1))

theorem average_of_first_16_even_divisible_by_5 :
  (List.sum first_16_even_divisible_by_5) / first_16_even_divisible_by_5.length = 85 := by
  sorry

end average_of_first_16_even_divisible_by_5_l3653_365378


namespace nancys_payment_is_384_l3653_365335

/-- Nancy's annual payment for her daughter's car insurance -/
def nancys_annual_payment (total_monthly_cost : ℝ) (nancy_share_percent : ℝ) : ℝ :=
  total_monthly_cost * nancy_share_percent * 12

/-- Proof that Nancy's annual payment is $384 -/
theorem nancys_payment_is_384 :
  nancys_annual_payment 80 0.4 = 384 := by
  sorry

end nancys_payment_is_384_l3653_365335


namespace bus_stop_walking_time_bus_stop_walking_time_proof_l3653_365347

/-- The time to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 7 minutes later than normal, is 28 minutes. -/
theorem bus_stop_walking_time : ℝ → Prop :=
  fun T : ℝ =>
    (4 / 5 * T + 7 = T) → T = 28

/-- Proof of the bus_stop_walking_time theorem -/
theorem bus_stop_walking_time_proof : ∃ T : ℝ, bus_stop_walking_time T :=
  sorry

end bus_stop_walking_time_bus_stop_walking_time_proof_l3653_365347


namespace volume_of_three_cubes_cuboid_l3653_365360

/-- The volume of a cuboid formed by attaching three identical cubes -/
def cuboid_volume (cube_side_length : ℝ) (num_cubes : ℕ) : ℝ :=
  (cube_side_length ^ 3) * num_cubes

/-- Theorem: The volume of a cuboid formed by three 6cm cubes is 648 cm³ -/
theorem volume_of_three_cubes_cuboid : 
  cuboid_volume 6 3 = 648 := by
  sorry

end volume_of_three_cubes_cuboid_l3653_365360


namespace log_expression_simplification_l3653_365398

theorem log_expression_simplification 
  (p q r s t z : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) : 
  Real.log (p / q) + Real.log (q / r) + 2 * Real.log (r / s) - Real.log (p * t / (s * z)) = 
  Real.log (r * z / (s * t)) := by
sorry

end log_expression_simplification_l3653_365398


namespace raffle_ticket_sales_difference_l3653_365339

/-- Proves that the difference between Saturday's and Sunday's raffle ticket sales is 284 -/
theorem raffle_ticket_sales_difference (friday_sales : ℕ) (sunday_sales : ℕ) 
  (h1 : friday_sales = 181)
  (h2 : sunday_sales = 78) : 
  2 * friday_sales - sunday_sales = 284 := by
  sorry

#check raffle_ticket_sales_difference

end raffle_ticket_sales_difference_l3653_365339


namespace investment_ratio_is_seven_to_five_l3653_365325

/-- Represents the investment and profit information for two partners -/
structure PartnerInvestment where
  profit_ratio : Rat
  p_investment_time : ℕ
  q_investment_time : ℕ

/-- Calculates the investment ratio given the profit ratio and investment times -/
def investment_ratio (info : PartnerInvestment) : Rat :=
  (info.profit_ratio * info.q_investment_time) / info.p_investment_time

/-- Theorem stating that given the specified conditions, the investment ratio is 7:5 -/
theorem investment_ratio_is_seven_to_five (info : PartnerInvestment) 
  (h1 : info.profit_ratio = 7 / 10)
  (h2 : info.p_investment_time = 2)
  (h3 : info.q_investment_time = 4) :
  investment_ratio info = 7 / 5 := by
  sorry

#eval investment_ratio { profit_ratio := 7 / 10, p_investment_time := 2, q_investment_time := 4 }

end investment_ratio_is_seven_to_five_l3653_365325


namespace arithmetic_sequence_formula_l3653_365385

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_formula :
  ∀ n : ℕ, arithmetic_sequence (-1) 4 n = 4 * n - 5 :=
by sorry

end arithmetic_sequence_formula_l3653_365385


namespace prime_count_200_to_220_l3653_365390

theorem prime_count_200_to_220 : ∃! p, Nat.Prime p ∧ 200 < p ∧ p < 220 := by
  sorry

end prime_count_200_to_220_l3653_365390


namespace cassie_water_refills_l3653_365305

/-- Represents the number of cups of water Cassie aims to drink daily -/
def daily_cups : ℕ := 12

/-- Represents the capacity of Cassie's water bottle in ounces -/
def bottle_capacity : ℕ := 16

/-- Represents the number of ounces in a cup -/
def ounces_per_cup : ℕ := 8

/-- Calculates the number of times Cassie needs to refill her water bottle -/
def refills_needed : ℕ := (daily_cups * ounces_per_cup) / bottle_capacity

theorem cassie_water_refills :
  refills_needed = 6 :=
sorry

end cassie_water_refills_l3653_365305


namespace math_contest_score_difference_l3653_365306

theorem math_contest_score_difference (score60 score75 score85 score95 : ℝ)
  (percent60 percent75 percent85 percent95 : ℝ)
  (h1 : score60 = 60)
  (h2 : score75 = 75)
  (h3 : score85 = 85)
  (h4 : score95 = 95)
  (h5 : percent60 = 0.2)
  (h6 : percent75 = 0.4)
  (h7 : percent85 = 0.25)
  (h8 : percent95 = 0.15)
  (h9 : percent60 + percent75 + percent85 + percent95 = 1) :
  let median := score75
  let mean := percent60 * score60 + percent75 * score75 + percent85 * score85 + percent95 * score95
  median - mean = -2.5 := by
  sorry

end math_contest_score_difference_l3653_365306


namespace smaller_sphere_radius_l3653_365362

-- Define the type for a sphere
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the function to check if two spheres are externally tangent
def are_externally_tangent (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (s1.radius + s2.radius)^2

-- Define the theorem
theorem smaller_sphere_radius 
  (s1 s2 s3 s4 : Sphere)
  (h1 : s1.radius = 2)
  (h2 : s2.radius = 2)
  (h3 : s3.radius = 3)
  (h4 : s4.radius = 3)
  (h5 : are_externally_tangent s1 s2)
  (h6 : are_externally_tangent s1 s3)
  (h7 : are_externally_tangent s1 s4)
  (h8 : are_externally_tangent s2 s3)
  (h9 : are_externally_tangent s2 s4)
  (h10 : are_externally_tangent s3 s4)
  (s5 : Sphere)
  (h11 : are_externally_tangent s1 s5)
  (h12 : are_externally_tangent s2 s5)
  (h13 : are_externally_tangent s3 s5)
  (h14 : are_externally_tangent s4 s5) :
  s5.radius = 6/11 :=
by sorry

end smaller_sphere_radius_l3653_365362


namespace solution_to_money_division_l3653_365302

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℝ  -- Amount x gets
  y : ℝ  -- Amount y gets
  z : ℝ  -- Amount z gets
  a : ℝ  -- Amount y gets for each rupee x gets

/-- The conditions of the problem -/
def problem_conditions (d : MoneyDivision) : Prop :=
  d.y = d.a * d.x ∧
  d.z = 0.5 * d.x ∧
  d.x + d.y + d.z = 78 ∧
  d.y = 18

/-- The theorem stating the solution to the problem -/
theorem solution_to_money_division :
  ∀ d : MoneyDivision, problem_conditions d → d.a = 0.45 := by
  sorry

end solution_to_money_division_l3653_365302


namespace seating_arrangements_count_l3653_365345

/-- The number of seats in a row -/
def total_seats : ℕ := 22

/-- The number of candidates to be seated -/
def num_candidates : ℕ := 4

/-- The minimum number of empty seats required between any two candidates -/
def min_empty_seats : ℕ := 5

/-- Calculate the number of ways to arrange the candidates -/
def seating_arrangements : ℕ := sorry

/-- Theorem stating that the number of seating arrangements is 840 -/
theorem seating_arrangements_count : seating_arrangements = 840 := by sorry

end seating_arrangements_count_l3653_365345


namespace logarithm_inequality_l3653_365303

theorem logarithm_inequality (x y z : ℝ) 
  (hx : x = Real.log π)
  (hy : y = Real.log π / Real.log (1/2))
  (hz : z = Real.exp (-1/2)) : 
  y < z ∧ z < x := by sorry

end logarithm_inequality_l3653_365303


namespace lawn_chair_price_calculation_l3653_365319

/-- Calculates the final price and overall percent decrease of a lawn chair after discounts and tax --/
theorem lawn_chair_price_calculation (original_price : ℝ) 
  (first_discount_rate second_discount_rate tax_rate : ℝ) :
  original_price = 72.95 ∧ 
  first_discount_rate = 0.10 ∧ 
  second_discount_rate = 0.15 ∧ 
  tax_rate = 0.07 →
  ∃ (final_price percent_decrease : ℝ),
    (abs (final_price - 59.71) < 0.01) ∧ 
    (abs (percent_decrease - 23.5) < 0.1) ∧
    final_price = (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) * (1 + tax_rate) ∧
    percent_decrease = (1 - (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) / original_price) * 100 := by
  sorry

end lawn_chair_price_calculation_l3653_365319


namespace fourth_month_sale_l3653_365351

/-- Calculates the missing sale amount given the other sales and desired average -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 desired_average : ℕ) : ℕ :=
  5 * desired_average - (sale1 + sale2 + sale3 + sale5)

/-- Theorem: Given the sales for 5 consecutive months, where 4 of the 5 sales are known,
    and the desired average sale, the sale in the fourth month must be 7720. -/
theorem fourth_month_sale 
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale5 : ℕ) (desired_average : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h3 : sale3 = 6200)
  (h4 : sale5 = 6500)
  (h5 : desired_average = 6300) :
  calculate_missing_sale sale1 sale2 sale3 sale5 desired_average = 7720 := by
  sorry

#eval calculate_missing_sale 5420 5660 6200 6500 6300

end fourth_month_sale_l3653_365351


namespace gambler_max_return_l3653_365366

/-- Represents the maximum amount a gambler can receive back after losing chips at a casino. -/
def max_amount_received (initial_value : ℕ) (chip_20_value : ℕ) (chip_100_value : ℕ) 
  (total_chips_lost : ℕ) : ℕ :=
  let chip_20_lost := (total_chips_lost + 2) / 2
  let chip_100_lost := total_chips_lost - chip_20_lost
  let value_lost := chip_20_lost * chip_20_value + chip_100_lost * chip_100_value
  initial_value - value_lost

/-- Theorem stating the maximum amount a gambler can receive back under specific conditions. -/
theorem gambler_max_return :
  max_amount_received 3000 20 100 16 = 2120 := by
  sorry

end gambler_max_return_l3653_365366


namespace inequality_proof_l3653_365377

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end inequality_proof_l3653_365377


namespace isosceles_with_120_exterior_is_equilateral_equal_exterior_angles_is_equilateral_l3653_365383

-- Define an isosceles triangle
structure IsoscelesTriangle :=
  (a b c : ℝ)
  (ab_eq_ac : a = c)

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (a b c : ℝ)
  (all_sides_equal : a = b ∧ b = c)

-- Define exterior angle
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle

theorem isosceles_with_120_exterior_is_equilateral 
  (t : IsoscelesTriangle) 
  (h : ∃ (angle : ℝ), exterior_angle angle = 120) : 
  EquilateralTriangle :=
sorry

theorem equal_exterior_angles_is_equilateral 
  (t : IsoscelesTriangle) 
  (h : ∃ (angle : ℝ), 
    exterior_angle angle = exterior_angle (exterior_angle angle) ∧ 
    exterior_angle angle = exterior_angle (exterior_angle (exterior_angle angle))) : 
  EquilateralTriangle :=
sorry

end isosceles_with_120_exterior_is_equilateral_equal_exterior_angles_is_equilateral_l3653_365383


namespace top_coat_drying_time_l3653_365316

/-- Given nail polish drying times, prove the top coat drying time -/
theorem top_coat_drying_time 
  (base_coat_time : ℕ) 
  (color_coat_time : ℕ) 
  (num_color_coats : ℕ) 
  (total_drying_time : ℕ) 
  (h1 : base_coat_time = 2)
  (h2 : color_coat_time = 3)
  (h3 : num_color_coats = 2)
  (h4 : total_drying_time = 13) :
  total_drying_time - (base_coat_time + num_color_coats * color_coat_time) = 5 := by
  sorry

end top_coat_drying_time_l3653_365316


namespace intersection_A_B_l3653_365374

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = Set.Ioo (-1 : ℝ) 2 := by sorry

end intersection_A_B_l3653_365374


namespace opposite_of_negative_four_l3653_365358

theorem opposite_of_negative_four :
  ∀ x : ℤ, x + (-4) = 0 → x = 4 := by
  sorry

end opposite_of_negative_four_l3653_365358


namespace min_students_is_fifteen_l3653_365354

/-- Represents the attendance for each day of the week -/
structure WeeklyAttendance where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the minimum number of students given weekly attendance -/
def minStudents (attendance : WeeklyAttendance) : Nat :=
  max (attendance.monday + attendance.wednesday + attendance.friday)
      (attendance.tuesday + attendance.thursday)

/-- Theorem: The minimum number of students who visited the library during the week is 15 -/
theorem min_students_is_fifteen (attendance : WeeklyAttendance)
  (h1 : attendance.monday = 5)
  (h2 : attendance.tuesday = 6)
  (h3 : attendance.wednesday = 4)
  (h4 : attendance.thursday = 8)
  (h5 : attendance.friday = 7) :
  minStudents attendance = 15 := by
  sorry

#eval minStudents ⟨5, 6, 4, 8, 7⟩

end min_students_is_fifteen_l3653_365354


namespace quadratic_extrema_l3653_365361

theorem quadratic_extrema :
  (∀ x : ℝ, 2 * x^2 - 1 ≥ -1) ∧
  (∀ x : ℝ, 2 * x^2 - 1 = -1 ↔ x = 0) ∧
  (∀ x : ℝ, -2 * (x + 1)^2 + 1 ≤ 1) ∧
  (∀ x : ℝ, -2 * (x + 1)^2 + 1 = 1 ↔ x = -1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 ≥ -1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 = -1 ↔ x = 1) :=
by sorry

end quadratic_extrema_l3653_365361


namespace intersection_M_N_l3653_365337

-- Define the sets M and N
def M : Set ℝ := {x | x / (x - 1) > 0}
def N : Set ℝ := {x | ∃ y, y * y = x}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 1} := by sorry

end intersection_M_N_l3653_365337


namespace total_earnings_l3653_365308

/-- Given that 5 men are equal to W women, W women are equal to B boys,
    and men's wages are 10, prove that the total amount earned by all groups is 150. -/
theorem total_earnings (W B : ℕ) (men_wage : ℕ) : 
  (5 = W) → (W = B) → (men_wage = 10) → 
  (5 * men_wage + W * men_wage + B * men_wage = 150) :=
by sorry

end total_earnings_l3653_365308


namespace moon_arrangements_l3653_365368

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "MOON" has 4 letters with one letter repeated twice -/
def moonWord : (ℕ × List ℕ) := (4, [2])

theorem moon_arrangements :
  distinctArrangements moonWord.fst moonWord.snd = 12 := by
  sorry

end moon_arrangements_l3653_365368


namespace elliot_book_pages_left_l3653_365309

def pages_left_after_week (total_pages : ℕ) (pages_read : ℕ) (pages_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pages - pages_read - (pages_per_day * days)

theorem elliot_book_pages_left : pages_left_after_week 381 149 20 7 = 92 := by
  sorry

end elliot_book_pages_left_l3653_365309


namespace brothers_initial_money_l3653_365336

theorem brothers_initial_money 
  (michael_initial : ℝ) 
  (brother_final : ℝ) 
  (candy_cost : ℝ) :
  michael_initial = 42 →
  brother_final = 35 →
  candy_cost = 3 →
  ∃ (brother_initial : ℝ),
    brother_initial + michael_initial / 2 - candy_cost = brother_final ∧
    brother_initial = 17 := by
  sorry

end brothers_initial_money_l3653_365336


namespace min_values_ab_and_a_plus_2b_l3653_365384

theorem min_values_ab_and_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 1/a + 2/b = 2) : 
  a * b ≥ 2 ∧ 
  a + 2*b ≥ 9/2 ∧ 
  (a + 2*b = 9/2 ↔ a = 3/2 ∧ b = 3/2) :=
by sorry

end min_values_ab_and_a_plus_2b_l3653_365384


namespace test_has_hundred_questions_l3653_365356

/-- Represents a test with a specific scoring system -/
structure Test where
  total_questions : ℕ
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  score_calculation : score = correct_responses - 2 * incorrect_responses
  total_questions_sum : total_questions = correct_responses + incorrect_responses

/-- Theorem stating that given the conditions, the test has 100 questions -/
theorem test_has_hundred_questions (t : Test) 
  (h1 : t.score = 79) 
  (h2 : t.correct_responses = 93) : 
  t.total_questions = 100 := by
  sorry


end test_has_hundred_questions_l3653_365356


namespace pillsbury_sugar_needed_l3653_365324

/-- Chef Pillsbury's recipe ratios -/
structure RecipeRatios where
  eggs_to_flour : ℚ
  milk_to_eggs : ℚ
  sugar_to_milk : ℚ

/-- Calculate the number of tablespoons of sugar needed for a given amount of flour -/
def sugar_needed (ratios : RecipeRatios) (flour_cups : ℚ) : ℚ :=
  let eggs := flour_cups * ratios.eggs_to_flour
  let milk := eggs * ratios.milk_to_eggs
  milk * ratios.sugar_to_milk

/-- Theorem: For 24 cups of flour, Chef Pillsbury needs 90 tablespoons of sugar -/
theorem pillsbury_sugar_needed :
  let ratios : RecipeRatios := {
    eggs_to_flour := 7 / 2,
    milk_to_eggs := 5 / 14,
    sugar_to_milk := 3 / 1
  }
  sugar_needed ratios 24 = 90 := by
  sorry

end pillsbury_sugar_needed_l3653_365324


namespace initial_number_of_persons_l3653_365330

theorem initial_number_of_persons (avg_weight_increase : ℝ) 
  (old_person_weight new_person_weight : ℝ) :
  avg_weight_increase = 2.5 →
  old_person_weight = 75 →
  new_person_weight = 95 →
  (new_person_weight - old_person_weight) / avg_weight_increase = 8 :=
by
  sorry

end initial_number_of_persons_l3653_365330


namespace sum_of_i_powers_l3653_365328

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^14 + i^19 + i^24 + i^29 + i^34 = -1 := by
  sorry

-- Define the property of i
axiom i_squared : i^2 = -1

end sum_of_i_powers_l3653_365328


namespace sqrt_product_quotient_equals_twelve_l3653_365367

theorem sqrt_product_quotient_equals_twelve :
  Real.sqrt 27 * Real.sqrt (8/3) / Real.sqrt (1/2) = 12 := by
  sorry

end sqrt_product_quotient_equals_twelve_l3653_365367


namespace equal_expressions_l3653_365320

theorem equal_expressions (x : ℝ) (hx : x > 0) :
  (x^(x+1) + x^(x+1) = 2*x^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ x^(2*x+2)) ∧
  (x^(x+1) + x^(x+1) ≠ (x+1)^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(x+1)) :=
by sorry

end equal_expressions_l3653_365320


namespace sum_equals_four_sqrt_860_l3653_365304

theorem sum_equals_four_sqrt_860 (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (products : a*c = 1008 ∧ b*d = 1008) :
  a + b + c + d = 4 * Real.sqrt 860 := by
  sorry

end sum_equals_four_sqrt_860_l3653_365304


namespace hyperbola_properties_l3653_365311

/-- Properties of a hyperbola -/
theorem hyperbola_properties (x y : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) →
  (∃ (imaginary_axis_length : ℝ) (asymptote_slope : ℝ) (focus_y : ℝ) (eccentricity : ℝ),
    imaginary_axis_length = 8 ∧
    asymptote_slope = 3/4 ∧
    focus_y = 5 ∧
    eccentricity = 5/3) :=
by sorry

end hyperbola_properties_l3653_365311


namespace min_reciprocal_sum_l3653_365387

theorem min_reciprocal_sum (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_sum : x + y + z = 2) :
  (1 / x + 1 / y + 1 / z) ≥ 4.5 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1 / x + 1 / y + 1 / z = 4.5 :=
sorry

end min_reciprocal_sum_l3653_365387


namespace candy_canes_count_l3653_365372

-- Define the problem parameters
def num_kids : ℕ := 3
def beanie_babies_per_stocking : ℕ := 2
def books_per_stocking : ℕ := 1
def total_stuffers : ℕ := 21

-- Define the function to calculate candy canes per stocking
def candy_canes_per_stocking : ℕ :=
  let non_candy_items_per_stocking := beanie_babies_per_stocking + books_per_stocking
  let total_non_candy_items := non_candy_items_per_stocking * num_kids
  let total_candy_canes := total_stuffers - total_non_candy_items
  total_candy_canes / num_kids

-- Theorem statement
theorem candy_canes_count : candy_canes_per_stocking = 4 := by
  sorry

end candy_canes_count_l3653_365372


namespace xy_min_max_l3653_365380

theorem xy_min_max (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) (h3 : -2 ≤ a ∧ a ≤ 2) :
  (∃ (x y : ℝ), x * y = -1 ∧ 
    ∀ (x' y' : ℝ), x' + y' = a → x'^2 + y'^2 = -a^2 + 2 → x' * y' ≥ -1) ∧
  (∃ (x y : ℝ), x * y = 1/3 ∧ 
    ∀ (x' y' : ℝ), x' + y' = a → x'^2 + y'^2 = -a^2 + 2 → x' * y' ≤ 1/3) :=
by sorry

end xy_min_max_l3653_365380


namespace range_of_m_l3653_365318

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → 
  -4 < m ∧ m < 2 := by
sorry

end range_of_m_l3653_365318


namespace parallel_planes_and_line_l3653_365369

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the "not contained in" relation for lines and planes
variable (line_not_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_planes_and_line 
  (α β : Plane) (a : Line) 
  (h1 : plane_parallel α β)
  (h2 : line_not_in_plane a α)
  (h3 : line_not_in_plane a β)
  (h4 : line_parallel_plane a α) :
  line_parallel_plane a β :=
sorry

end parallel_planes_and_line_l3653_365369


namespace hypergeometric_distribution_proof_l3653_365307

def hypergeometric_prob (N n m k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m

theorem hypergeometric_distribution_proof (N n m : ℕ) 
  (h1 : N = 10) (h2 : n = 8) (h3 : m = 2) : 
  (hypergeometric_prob N n m 0 = 1/45) ∧
  (hypergeometric_prob N n m 1 = 16/45) ∧
  (hypergeometric_prob N n m 2 = 28/45) := by
  sorry

end hypergeometric_distribution_proof_l3653_365307


namespace max_value_of_s_l3653_365313

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 := by
  sorry

end max_value_of_s_l3653_365313


namespace two_digit_reverse_sum_square_l3653_365391

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Predicate for a number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- The main theorem -/
theorem two_digit_reverse_sum_square :
  (∃ (S : Finset ℕ), S.card = 8 ∧
    ∀ n ∈ S, 10 ≤ n ∧ n < 100 ∧
    is_perfect_square (n + reverse_digits n)) ∧
  ¬∃ (S : Finset ℕ), S.card > 8 ∧
    ∀ n ∈ S, 10 ≤ n ∧ n < 100 ∧
    is_perfect_square (n + reverse_digits n) := by
  sorry


end two_digit_reverse_sum_square_l3653_365391


namespace combined_tax_rate_l3653_365349

/-- Represents the combined tax rate problem for Mork, Mindy, and Julie -/
theorem combined_tax_rate 
  (mork_rate : ℚ) 
  (mindy_rate : ℚ) 
  (julie_rate : ℚ) 
  (mork_income : ℚ) 
  (mindy_income : ℚ) 
  (julie_income : ℚ) :
  mork_rate = 45/100 →
  mindy_rate = 25/100 →
  julie_rate = 35/100 →
  mindy_income = 4 * mork_income →
  julie_income = 2 * mork_income →
  julie_income = (1/2) * mindy_income →
  (mork_rate * mork_income + mindy_rate * mindy_income + julie_rate * julie_income) / 
  (mork_income + mindy_income + julie_income) = 215/700 :=
by sorry

end combined_tax_rate_l3653_365349


namespace cube_root_simplification_l3653_365396

theorem cube_root_simplification :
  (72^3 + 108^3 + 144^3 : ℝ)^(1/3) = 36 * 99^(1/3) :=
by sorry

end cube_root_simplification_l3653_365396


namespace paper_crane_ratio_l3653_365364

/-- Represents the number of paper cranes Alice wants in total -/
def total_cranes : ℕ := 1000

/-- Represents the number of paper cranes Alice still needs to fold -/
def remaining_cranes : ℕ := 400

/-- Represents the ratio of cranes folded by Alice's friend to remaining cranes after Alice folded half -/
def friend_to_remaining_ratio : Rat := 1 / 5

theorem paper_crane_ratio :
  let alice_folded := total_cranes / 2
  let remaining_after_alice := total_cranes - alice_folded
  let friend_folded := remaining_after_alice - remaining_cranes
  friend_folded / remaining_after_alice = friend_to_remaining_ratio := by
sorry

end paper_crane_ratio_l3653_365364


namespace divisible_by_37_l3653_365392

theorem divisible_by_37 (n d : ℕ) (h : d ≤ 9) : 
  ∃ k : ℕ, (d * (10^(3*n) - 1) / 9) = 37 * k := by
  sorry

end divisible_by_37_l3653_365392


namespace fraction_sum_equals_percentage_l3653_365399

theorem fraction_sum_equals_percentage (y : ℝ) (h : y > 0) :
  (7 * y) / 20 + (3 * y) / 10 = 0.65 * y := by
  sorry

end fraction_sum_equals_percentage_l3653_365399


namespace chastity_gummy_packs_l3653_365393

/-- Given Chastity's candy purchase scenario, prove the number of gummy packs bought. -/
theorem chastity_gummy_packs :
  ∀ (initial_money : ℚ) 
    (remaining_money : ℚ) 
    (lollipop_count : ℕ) 
    (lollipop_price : ℚ) 
    (gummy_pack_price : ℚ),
  initial_money = 15 →
  remaining_money = 5 →
  lollipop_count = 4 →
  lollipop_price = 3/2 →
  gummy_pack_price = 2 →
  ∃ (gummy_pack_count : ℕ),
    gummy_pack_count = 2 ∧
    initial_money - remaining_money = 
      (lollipop_count : ℚ) * lollipop_price + (gummy_pack_count : ℚ) * gummy_pack_price :=
by sorry

end chastity_gummy_packs_l3653_365393


namespace quadratic_two_real_roots_l3653_365386

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ + k = 0 ∧ x₂^2 + 4*x₂ + k = 0) ↔ k ≤ 4 :=
sorry

end quadratic_two_real_roots_l3653_365386


namespace diamond_equation_solution_l3653_365310

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B * 2

-- Theorem statement
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 5 = 64 ∧ A = 8.5 := by
  sorry

end diamond_equation_solution_l3653_365310


namespace complex_reciprocal_l3653_365350

theorem complex_reciprocal (i : ℂ) : i * i = -1 → (1 : ℂ) / (1 - i) = (1 : ℂ) / 2 + i / 2 := by
  sorry

end complex_reciprocal_l3653_365350


namespace binomial_expansion_ratio_l3653_365340

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 := by
  sorry

end binomial_expansion_ratio_l3653_365340


namespace other_players_score_l3653_365389

theorem other_players_score (total_score : ℕ) (faye_score : ℕ) (total_players : ℕ) :
  total_score = 68 →
  faye_score = 28 →
  total_players = 5 →
  ∃ (other_player_score : ℕ),
    other_player_score * (total_players - 1) = total_score - faye_score ∧
    other_player_score = 10 := by
  sorry

end other_players_score_l3653_365389


namespace tim_sleep_total_l3653_365376

theorem tim_sleep_total (sleep_first_two_days sleep_next_two_days : ℕ) 
  (h1 : sleep_first_two_days = 6 * 2)
  (h2 : sleep_next_two_days = 10 * 2) :
  sleep_first_two_days + sleep_next_two_days = 32 := by
  sorry

end tim_sleep_total_l3653_365376


namespace paul_bought_two_pants_l3653_365370

def shirtPrice : ℝ := 15
def pantPrice : ℝ := 40
def suitPrice : ℝ := 150
def sweaterPrice : ℝ := 30
def storeDiscount : ℝ := 0.2
def couponDiscount : ℝ := 0.1
def finalSpent : ℝ := 252

def totalBeforeDiscount (numPants : ℝ) : ℝ :=
  4 * shirtPrice + numPants * pantPrice + suitPrice + 2 * sweaterPrice

def discountedTotal (numPants : ℝ) : ℝ :=
  (1 - storeDiscount) * totalBeforeDiscount numPants

def finalTotal (numPants : ℝ) : ℝ :=
  (1 - couponDiscount) * discountedTotal numPants

theorem paul_bought_two_pants :
  ∃ (numPants : ℝ), numPants = 2 ∧ finalTotal numPants = finalSpent :=
sorry

end paul_bought_two_pants_l3653_365370


namespace roger_trays_second_table_l3653_365348

/-- Represents the number of trays Roger can carry in one trip -/
def trays_per_trip : ℕ := 4

/-- Represents the number of trips Roger made -/
def num_trips : ℕ := 3

/-- Represents the number of trays Roger picked up from the first table -/
def trays_first_table : ℕ := 10

/-- Calculates the number of trays Roger picked up from the second table -/
def trays_second_table : ℕ := trays_per_trip * num_trips - trays_first_table

theorem roger_trays_second_table : trays_second_table = 2 := by
  sorry

end roger_trays_second_table_l3653_365348


namespace teacher_earnings_five_weeks_l3653_365314

/-- Calculates the teacher's earnings for piano lessons over a given number of weeks -/
def teacher_earnings (rate_per_half_hour : ℕ) (lesson_duration_hours : ℕ) (lessons_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  rate_per_half_hour * (lesson_duration_hours * 2) * lessons_per_week * num_weeks

/-- Proves that the teacher earns $100 in 5 weeks under the given conditions -/
theorem teacher_earnings_five_weeks :
  teacher_earnings 10 1 1 5 = 100 := by
  sorry

end teacher_earnings_five_weeks_l3653_365314


namespace fraction_simplification_l3653_365344

theorem fraction_simplification :
  ((3^12)^2 - (3^10)^2) / ((3^11)^2 - (3^9)^2) = 9 := by
  sorry

end fraction_simplification_l3653_365344


namespace brandy_excess_caffeine_l3653_365373

/-- Represents the caffeine consumption and tolerance of a person named Brandy --/
structure BrandyCaffeine where
  weight : ℝ
  baseLimit : ℝ
  additionalTolerance : ℝ
  coffeeConsumption : ℝ
  energyDrinkConsumption : ℝ
  medicationEffect : ℝ

/-- Calculates the excess caffeine consumed by Brandy --/
def excessCaffeineConsumed (b : BrandyCaffeine) : ℝ :=
  let maxSafe := b.weight * b.baseLimit + b.additionalTolerance - b.medicationEffect
  let consumed := b.coffeeConsumption + b.energyDrinkConsumption
  consumed - maxSafe

/-- Theorem stating that Brandy has consumed 495 mg more caffeine than her adjusted maximum safe amount --/
theorem brandy_excess_caffeine :
  let b : BrandyCaffeine := {
    weight := 60,
    baseLimit := 2.5,
    additionalTolerance := 50,
    coffeeConsumption := 2 * 95,
    energyDrinkConsumption := 4 * 120,
    medicationEffect := 25
  }
  excessCaffeineConsumed b = 495 := by sorry

end brandy_excess_caffeine_l3653_365373


namespace limit_equals_derivative_l3653_365341

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem limit_equals_derivative
  (h1 : Differentiable ℝ f)  -- f is differentiable
  (h2 : deriv f 1 = 1) :     -- f'(1) = 1
  Filter.Tendsto
    (fun Δx => (f (1 - Δx) - f 1) / (-Δx))
    (nhds 0)
    (nhds 1) :=
by
  sorry

end limit_equals_derivative_l3653_365341


namespace no_solution_exists_l3653_365346

theorem no_solution_exists : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (∃ (n : ℕ), 
    ((b + c - a) / a = n) ∧
    ((a + c - b) / b = n) ∧
    ((a + b - c) / c = n)) ∧
  ((a + b) * (b + c) * (a + c)) / (a * b * c) = 12 := by
  sorry

end no_solution_exists_l3653_365346


namespace even_painted_faces_count_l3653_365343

/-- Represents a rectangular block -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with even number of painted faces in a painted block -/
def countEvenPaintedFaces (b : Block) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem even_painted_faces_count (b : Block) :
  b.length = 6 → b.width = 4 → b.height = 2 →
  countEvenPaintedFaces b = 32 := by
  sorry

end even_painted_faces_count_l3653_365343


namespace ratio_equation_solution_product_l3653_365382

theorem ratio_equation_solution_product (x : ℝ) :
  (3 * x + 5) / (4 * x + 4) = (5 * x + 4) / (10 * x + 5) →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (3 * x₁ + 5) / (4 * x₁ + 4) = (5 * x₁ + 4) / (10 * x₁ + 5) ∧
    (3 * x₂ + 5) / (4 * x₂ + 4) = (5 * x₂ + 4) / (10 * x₂ + 5) ∧
    x₁ * x₂ = 9 / 10 :=
by sorry

end ratio_equation_solution_product_l3653_365382


namespace count_integer_lengths_problem_triangle_l3653_365394

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of distinct integer lengths of line segments 
    from a vertex to points on the hypotenuse -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { leg1 := 15, leg2 := 36 }

/-- The theorem stating that the number of distinct integer lengths 
    for the given triangle is 24 -/
theorem count_integer_lengths_problem_triangle : 
  countIntegerLengths problemTriangle = 24 :=
sorry

end count_integer_lengths_problem_triangle_l3653_365394


namespace uniform_cost_is_355_l3653_365331

/-- Calculates the total cost of uniforms for a student given the costs of individual items -/
def uniform_cost (pants_cost shirt_cost tie_cost socks_cost : ℚ) : ℚ :=
  5 * (pants_cost + shirt_cost + tie_cost + socks_cost)

/-- Proves that the total cost of uniforms for a student is $355 -/
theorem uniform_cost_is_355 :
  uniform_cost 20 40 8 3 = 355 := by
  sorry

#eval uniform_cost 20 40 8 3

end uniform_cost_is_355_l3653_365331


namespace midpoint_chain_l3653_365363

/-- Given a line segment AB with multiple midpoints, prove its length --/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 5) →        -- AG = 5
  (B - A = 160) :=     -- AB = 160
by sorry

end midpoint_chain_l3653_365363


namespace yellow_highlighters_l3653_365352

theorem yellow_highlighters (total : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : total = 15) 
  (h2 : pink = 3) 
  (h3 : blue = 5) : 
  total - pink - blue = 7 := by
  sorry

end yellow_highlighters_l3653_365352


namespace greatest_value_problem_l3653_365315

theorem greatest_value_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hy : 0 < y₁ ∧ y₁ < y₂) 
  (hsum : x₁ + x₂ = 1 ∧ y₁ + y₂ = 1) : 
  max (x₁*y₁ + x₂*y₂) (max (x₁*x₂ + y₁*y₂) (max (x₁*y₂ + x₂*y₁) (1/2))) = x₁*y₁ + x₂*y₂ := by
  sorry

end greatest_value_problem_l3653_365315


namespace max_sum_three_consecutive_l3653_365395

/-- A circular arrangement of numbers from 1 to 10 -/
def CircularArrangement := Fin 10 → Fin 10

/-- The sum of three consecutive numbers in a circular arrangement -/
def sumThreeConsecutive (arr : CircularArrangement) (i : Fin 10) : Nat :=
  arr i + arr ((i + 1) % 10) + arr ((i + 2) % 10)

/-- The theorem stating the maximum sum of three consecutive numbers -/
theorem max_sum_three_consecutive :
  (∀ arr : CircularArrangement, ∃ i : Fin 10, sumThreeConsecutive arr i ≥ 18) ∧
  ¬(∀ arr : CircularArrangement, ∃ i : Fin 10, sumThreeConsecutive arr i ≥ 19) :=
sorry

end max_sum_three_consecutive_l3653_365395


namespace melanie_cats_count_l3653_365332

/-- Given that Jacob has 90 cats, Annie has three times fewer cats than Jacob,
    and Melanie has twice as many cats as Annie, prove that Melanie has 60 cats. -/
theorem melanie_cats_count :
  ∀ (jacob_cats annie_cats melanie_cats : ℕ),
    jacob_cats = 90 →
    annie_cats * 3 = jacob_cats →
    melanie_cats = annie_cats * 2 →
    melanie_cats = 60 := by
  sorry

end melanie_cats_count_l3653_365332


namespace angle_measure_proof_l3653_365381

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 7) → x = 24.25 := by
  sorry

end angle_measure_proof_l3653_365381


namespace final_value_l3653_365321

/-- The value of A based on bundles --/
def A : ℕ := 6 * 1000 + 36 * 100

/-- The value of B based on jumping twice --/
def B : ℕ := 876 - 197 - 197

/-- Theorem stating the final result --/
theorem final_value : A - B = 9118 := by
  sorry

end final_value_l3653_365321


namespace boys_girls_difference_l3653_365342

theorem boys_girls_difference (x y : ℕ) (a b : ℚ) : 
  x > y → 
  x * a + y * b = x * b + y * a - 1 → 
  x = y + 1 := by
sorry

end boys_girls_difference_l3653_365342


namespace negate_negative_twenty_l3653_365327

theorem negate_negative_twenty : -(-20) = 20 := by
  sorry

end negate_negative_twenty_l3653_365327


namespace power_division_l3653_365357

theorem power_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end power_division_l3653_365357


namespace ampersand_example_l3653_365371

-- Define the & operation
def ampersand (a b : ℚ) : ℚ := (a + 1) / b

-- State the theorem
theorem ampersand_example : ampersand 2 (ampersand 3 4) = 3 := by
  sorry

end ampersand_example_l3653_365371


namespace polar_line_properties_l3653_365312

/-- A line in polar coordinates passing through (4,0) and perpendicular to the polar axis -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

theorem polar_line_properties (ρ θ : ℝ) :
  polar_line ρ θ →
  (ρ * Real.cos θ = 4 ∧ ρ * Real.sin θ = 0) ∧
  (∀ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ → x = 4) :=
sorry

end polar_line_properties_l3653_365312


namespace quadratic_roots_real_distinct_l3653_365379

theorem quadratic_roots_real_distinct (d : ℝ) : 
  let a : ℝ := 3
  let b : ℝ := -4 * Real.sqrt 3
  let c : ℝ := d
  let discriminant : ℝ := b^2 - 4*a*c
  discriminant = 12 →
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end quadratic_roots_real_distinct_l3653_365379


namespace subtraction_is_perfect_square_l3653_365322

def A : ℕ := (10^1001 - 1) / 9
def B : ℕ := (10^2002 - 1) / 9
def C : ℕ := 2 * A

theorem subtraction_is_perfect_square : B - C = (3 * A)^2 := by
  sorry

end subtraction_is_perfect_square_l3653_365322


namespace problem_statement_l3653_365375

theorem problem_statement (x y : ℝ) (h1 : x - y = 3) (h2 : x * y = 2) :
  3 * x - 5 * x * y - 3 * y = -1 := by
  sorry

end problem_statement_l3653_365375


namespace quadratic_roots_average_l3653_365334

theorem quadratic_roots_average (c : ℝ) 
  (h : ∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - 6 * x + c = 0 ∧ 2 * y^2 - 6 * y + c = 0) :
  ∃ x y : ℝ, x ≠ y ∧ 
    2 * x^2 - 6 * x + c = 0 ∧ 
    2 * y^2 - 6 * y + c = 0 ∧ 
    (x + y) / 2 = 3 / 2 :=
by sorry

end quadratic_roots_average_l3653_365334


namespace sector_forms_cylinder_l3653_365365

-- Define the sector
def sector_angle : ℝ := 300
def sector_radius : ℝ := 12

-- Define the cylinder
def cylinder_base_radius : ℝ := 10
def cylinder_height : ℝ := 12

-- Theorem statement
theorem sector_forms_cylinder :
  2 * Real.pi * cylinder_base_radius = (sector_angle / 360) * 2 * Real.pi * sector_radius ∧
  cylinder_height = sector_radius :=
by sorry

end sector_forms_cylinder_l3653_365365


namespace product_of_specific_difference_and_cube_difference_l3653_365397

theorem product_of_specific_difference_and_cube_difference
  (x y : ℝ) (h1 : x - y = 4) (h2 : x^3 - y^3 = 28) : x * y = -3 := by
  sorry

end product_of_specific_difference_and_cube_difference_l3653_365397


namespace car_dealership_hourly_wage_l3653_365300

/-- Calculates the hourly wage for employees in a car dealership --/
theorem car_dealership_hourly_wage :
  let fiona_weekly_hours : ℕ := 40
  let john_weekly_hours : ℕ := 30
  let jeremy_weekly_hours : ℕ := 25
  let weeks_per_month : ℕ := 4
  let total_monthly_pay : ℕ := 7600

  let total_monthly_hours : ℕ := 
    (fiona_weekly_hours + john_weekly_hours + jeremy_weekly_hours) * weeks_per_month

  (total_monthly_pay : ℚ) / total_monthly_hours = 20 := by
  sorry


end car_dealership_hourly_wage_l3653_365300


namespace expression_equals_25_l3653_365338

theorem expression_equals_25 : 
  (5^1010)^2 - (5^1008)^2 / (5^1009)^2 - (5^1007)^2 = 25 := by
  sorry

end expression_equals_25_l3653_365338


namespace ratio_not_always_constant_l3653_365326

theorem ratio_not_always_constant : ∃ (f g : ℝ → ℝ), ¬(∀ x : ℝ, ∃ c : ℝ, f x = c * g x) :=
sorry

end ratio_not_always_constant_l3653_365326


namespace magpie_porridge_l3653_365329

/-- Represents the amount of porridge each chick received -/
structure ChickPorridge where
  x1 : ℝ
  x2 : ℝ
  x3 : ℝ
  x4 : ℝ
  x5 : ℝ
  x6 : ℝ

/-- The conditions of porridge distribution -/
def porridge_conditions (p : ChickPorridge) : Prop :=
  p.x3 = p.x1 + p.x2 ∧
  p.x4 = p.x2 + p.x3 ∧
  p.x5 = p.x3 + p.x4 ∧
  p.x6 = p.x4 + p.x5 ∧
  p.x5 = 10

/-- The total amount of porridge cooked by the magpie -/
def total_porridge (p : ChickPorridge) : ℝ :=
  p.x1 + p.x2 + p.x3 + p.x4 + p.x5 + p.x6

/-- Theorem stating that the total amount of porridge is 40 grams -/
theorem magpie_porridge (p : ChickPorridge) :
  porridge_conditions p → total_porridge p = 40 := by
  sorry

end magpie_porridge_l3653_365329


namespace geometric_sequence_ratio_l3653_365388

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- positive terms
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  (a 2 - (1/2) * a 3 = (1/2) * a 3 - a 1) →  -- arithmetic sequence condition
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 :=
by sorry

end geometric_sequence_ratio_l3653_365388


namespace random_variable_iff_preimage_singleton_l3653_365359

variable {Ω : Type*} [MeasurableSpace Ω]
variable {E : Set ℝ} (hE : Countable E)
variable (ξ : Ω → ℝ) (hξ : ∀ ω, ξ ω ∈ E)

theorem random_variable_iff_preimage_singleton :
  Measurable ξ ↔ ∀ x ∈ E, MeasurableSet {ω | ξ ω = x} := by
  sorry

end random_variable_iff_preimage_singleton_l3653_365359


namespace john_weight_lifting_l3653_365333

/-- John's weight lifting problem -/
theorem john_weight_lifting 
  (weight_per_rep : ℕ) 
  (reps_per_set : ℕ) 
  (num_sets : ℕ) 
  (h1 : weight_per_rep = 15)
  (h2 : reps_per_set = 10)
  (h3 : num_sets = 3) :
  weight_per_rep * reps_per_set * num_sets = 450 := by
  sorry

#check john_weight_lifting

end john_weight_lifting_l3653_365333
