import Mathlib

namespace hyperbola_center_l3752_375268

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (5, 0) → f2 = (9, 4) → center = (7, 2) := by
  sorry

end hyperbola_center_l3752_375268


namespace chicken_wing_distribution_l3752_375260

theorem chicken_wing_distribution (total_wings : ℕ) (num_people : ℕ) 
  (h1 : total_wings = 35) (h2 : num_people = 12) :
  let wings_per_person := total_wings / num_people
  let leftover_wings := total_wings % num_people
  wings_per_person = 2 ∧ leftover_wings = 11 := by
  sorry

end chicken_wing_distribution_l3752_375260


namespace smallest_number_divisible_l3752_375252

theorem smallest_number_divisible (n : ℕ) : n = 1009 ↔ 
  (∀ m : ℕ, m < n → ¬(12 ∣ (m - 2) ∧ 16 ∣ (m - 2) ∧ 18 ∣ (m - 2) ∧ 21 ∣ (m - 2) ∧ 28 ∣ (m - 2))) ∧
  (12 ∣ (n - 2) ∧ 16 ∣ (n - 2) ∧ 18 ∣ (n - 2) ∧ 21 ∣ (n - 2) ∧ 28 ∣ (n - 2)) :=
by sorry

end smallest_number_divisible_l3752_375252


namespace charity_fundraising_l3752_375200

theorem charity_fundraising 
  (total_amount : ℝ) 
  (sponsor_contribution : ℝ) 
  (number_of_people : ℕ) :
  total_amount = 2400 →
  sponsor_contribution = 300 →
  number_of_people = 8 →
  (total_amount - sponsor_contribution) / number_of_people = 262.5 := by
sorry

end charity_fundraising_l3752_375200


namespace trees_not_replanted_l3752_375206

/-- 
Given a track with trees planted every 4 meters along its 48-meter length,
prove that when replanting trees every 6 meters, 5 trees do not need to be replanted.
-/
theorem trees_not_replanted (track_length : ℕ) (initial_spacing : ℕ) (new_spacing : ℕ) : 
  track_length = 48 → initial_spacing = 4 → new_spacing = 6 → 
  (track_length / Nat.lcm initial_spacing new_spacing) + 1 = 5 := by
  sorry

end trees_not_replanted_l3752_375206


namespace car_distance_theorem_l3752_375216

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours. -/
def total_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + (initial_speed + h * speed_increase)) 0

/-- Theorem stating that a car with initial speed 50 km/h, increasing by 2 km/h each hour, travels 732 km in 12 hours. -/
theorem car_distance_theorem : total_distance 50 2 12 = 732 := by
  sorry

end car_distance_theorem_l3752_375216


namespace arithmetic_sequence_length_l3752_375271

theorem arithmetic_sequence_length : 
  ∀ (a₁ : ℕ) (aₙ : ℕ) (d : ℕ),
    a₁ = 4 →
    aₙ = 130 →
    d = 2 →
    (aₙ - a₁) / d + 1 = 64 :=
by
  sorry

end arithmetic_sequence_length_l3752_375271


namespace sqrt_expression_equals_one_l3752_375269

theorem sqrt_expression_equals_one :
  1 + (Real.sqrt 2 - Real.sqrt 3) + |Real.sqrt 2 - Real.sqrt 3| = 1 := by
  sorry

end sqrt_expression_equals_one_l3752_375269


namespace square_garden_perimeter_l3752_375232

theorem square_garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) (h3 : q = p + 21) :
  p = 28 := by
  sorry

end square_garden_perimeter_l3752_375232


namespace f_of_2_eq_0_l3752_375259

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 1

-- State the theorem
theorem f_of_2_eq_0 : f 2 = 0 := by sorry

end f_of_2_eq_0_l3752_375259


namespace quadratic_inequality_l3752_375283

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 4 < 0 ↔ -1 < x ∧ x < 4 := by
  sorry

end quadratic_inequality_l3752_375283


namespace problem_solution_l3752_375257

theorem problem_solution : (2010^2 - 2010) / 2010 = 2009 := by
  sorry

end problem_solution_l3752_375257


namespace calculate_expression_l3752_375258

theorem calculate_expression : (1/2)⁻¹ + |3 - Real.sqrt 12| + (-1)^2 = 2 * Real.sqrt 3 := by
  sorry

end calculate_expression_l3752_375258


namespace two_digit_number_problem_l3752_375222

theorem two_digit_number_problem : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10) * 2 = (n % 10) * 3 ∧
  (n / 10) = (n % 10) + 3 ∧
  n = 63 := by
  sorry

end two_digit_number_problem_l3752_375222


namespace walts_investment_l3752_375296

/-- Proves that given the conditions of Walt's investment, the unknown interest rate is 8% -/
theorem walts_investment (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (unknown_investment : ℝ) :
  total_investment = 9000 →
  known_rate = 0.09 →
  total_interest = 770 →
  unknown_investment = 4000 →
  ∃ (unknown_rate : ℝ),
    unknown_rate * unknown_investment + known_rate * (total_investment - unknown_investment) = total_interest ∧
    unknown_rate = 0.08 := by
  sorry

end walts_investment_l3752_375296


namespace delta_zero_implies_c_sqrt_30_l3752_375213

def Δ (a b c : ℝ) : ℝ := c^2 - 3*a*b

theorem delta_zero_implies_c_sqrt_30 (a b c : ℝ) (h1 : Δ a b c = 0) (h2 : a = 2) (h3 : b = 5) :
  c = Real.sqrt 30 ∨ c = -Real.sqrt 30 := by sorry

end delta_zero_implies_c_sqrt_30_l3752_375213


namespace fifth_row_dots_l3752_375221

/-- Represents the number of green dots in a row -/
def greenDots : ℕ → ℕ
  | 0 => 3  -- First row (index 0) has 3 dots
  | n + 1 => greenDots n + 3  -- Each subsequent row increases by 3 dots

/-- The theorem stating that the fifth row has 15 green dots -/
theorem fifth_row_dots : greenDots 4 = 15 := by
  sorry

end fifth_row_dots_l3752_375221


namespace six_steps_position_l3752_375289

/-- Given a number line with equally spaced markings where 8 steps cover 48 units,
    prove that 6 steps from 0 reach position 36. -/
theorem six_steps_position (total_distance : ℕ) (total_steps : ℕ) (steps : ℕ) :
  total_distance = 48 →
  total_steps = 8 →
  steps = 6 →
  (total_distance / total_steps) * steps = 36 := by
  sorry

end six_steps_position_l3752_375289


namespace min_abs_z_plus_2i_l3752_375233

-- Define the complex number z
variable (z : ℂ)

-- Define the condition from the problem
def condition (z : ℂ) : Prop := Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*Complex.I))

-- State the theorem
theorem min_abs_z_plus_2i :
  (∀ z, condition z → Complex.abs (z + 2*Complex.I) ≥ 5/2) ∧
  (∃ z, condition z ∧ Complex.abs (z + 2*Complex.I) = 5/2) :=
sorry

end min_abs_z_plus_2i_l3752_375233


namespace cookie_theorem_l3752_375274

def cookie_problem (initial_cookies : ℕ) (given_to_friend : ℕ) (eaten : ℕ) : ℕ :=
  let remaining_after_friend := initial_cookies - given_to_friend
  let given_to_family := remaining_after_friend / 2
  let remaining_after_family := remaining_after_friend - given_to_family
  remaining_after_family - eaten

theorem cookie_theorem : cookie_problem 19 5 2 = 5 := by
  sorry

end cookie_theorem_l3752_375274


namespace person_A_silver_sheets_l3752_375287

-- Define the exchange rates
def red_to_gold_rate : ℚ := 5 / 2
def gold_to_red_and_silver_rate : ℚ := 1

-- Define the initial number of sheets
def initial_red_sheets : ℕ := 3
def initial_gold_sheets : ℕ := 3

-- Define the function to calculate the total silver sheets
def total_silver_sheets : ℕ :=
  let gold_to_silver := initial_gold_sheets
  let red_to_silver := (initial_red_sheets + initial_gold_sheets) / 3 * 2
  gold_to_silver + red_to_silver

-- Theorem statement
theorem person_A_silver_sheets :
  total_silver_sheets = 7 :=
sorry

end person_A_silver_sheets_l3752_375287


namespace parabola_focus_coordinates_l3752_375236

/-- Given a parabola y = (1/m)x^2 where m ≠ 0, its focus has coordinates (0, m/4) -/
theorem parabola_focus_coordinates (m : ℝ) (hm : m ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = (1/m) * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, m/4) := by
  sorry

end parabola_focus_coordinates_l3752_375236


namespace group_size_proof_l3752_375255

theorem group_size_proof (W : ℝ) (n : ℕ) : 
  (W + 15) / n = W / n + 2.5 → n = 6 := by
  sorry

end group_size_proof_l3752_375255


namespace f_has_two_roots_l3752_375256

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem f_has_two_roots : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry

end f_has_two_roots_l3752_375256


namespace equation_condition_l3752_375280

theorem equation_condition (a b c d : ℝ) :
  (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2) →
  (a = c ∨ a^2 + d + 2*b = 0) :=
by sorry

end equation_condition_l3752_375280


namespace tangent_circle_radius_l3752_375208

/-- The radius of a circle tangent to eight semicircles lining the inside of a square --/
theorem tangent_circle_radius (square_side : ℝ) (h : square_side = 4) :
  let semicircle_radius : ℝ := square_side / 4
  let diagonal : ℝ := Real.sqrt (square_side ^ 2 / 4 + (square_side / 4) ^ 2)
  diagonal - semicircle_radius = Real.sqrt 5 - 1 := by
  sorry

end tangent_circle_radius_l3752_375208


namespace max_value_expression_l3752_375277

theorem max_value_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) * (x - y + z) ≤ 2187/216 := by
  sorry

end max_value_expression_l3752_375277


namespace exactly_one_correct_probability_l3752_375297

theorem exactly_one_correct_probability
  (probA : ℝ) (probB : ℝ) (probC : ℝ)
  (hprobA : probA = 3/4)
  (hprobB : probB = 2/3)
  (hprobC : probC = 2/3)
  (hprobA_bounds : 0 ≤ probA ∧ probA ≤ 1)
  (hprobB_bounds : 0 ≤ probB ∧ probB ≤ 1)
  (hprobC_bounds : 0 ≤ probC ∧ probC ≤ 1) :
  probA * (1 - probB) * (1 - probC) +
  (1 - probA) * probB * (1 - probC) +
  (1 - probA) * (1 - probB) * probC = 7/36 :=
sorry

end exactly_one_correct_probability_l3752_375297


namespace eighth_term_is_22_n_equals_8_when_an_is_22_l3752_375267

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 8th term of the sequence is 22 -/
theorem eighth_term_is_22 : arithmeticSequence 8 = 22 := by
  sorry

/-- Theorem proving that if the nth term is 22, then n must be 8 -/
theorem n_equals_8_when_an_is_22 (n : ℕ) (h : arithmeticSequence n = 22) : n = 8 := by
  sorry

end eighth_term_is_22_n_equals_8_when_an_is_22_l3752_375267


namespace rain_both_days_no_snow_l3752_375238

theorem rain_both_days_no_snow (rain_sat rain_sun snow_sat : ℝ) 
  (h_rain_sat : rain_sat = 0.7)
  (h_rain_sun : rain_sun = 0.5)
  (h_snow_sat : snow_sat = 0.2)
  (h_independence : True) -- Assumption of independence
  : rain_sat * rain_sun * (1 - snow_sat) = 0.28 := by
  sorry

end rain_both_days_no_snow_l3752_375238


namespace multiplication_puzzle_l3752_375210

theorem multiplication_puzzle :
  ∃ (AB C : ℕ),
    10 ≤ AB ∧ AB < 100 ∧
    1 ≤ C ∧ C < 10 ∧
    100 ≤ AB * 8 ∧ AB * 8 < 1000 ∧
    1000 ≤ AB * 9 ∧
    AB * C = 1068 := by
  sorry

end multiplication_puzzle_l3752_375210


namespace battery_difference_proof_l3752_375223

/-- The number of batteries Tom used in flashlights -/
def flashlight_batteries : ℕ := 2

/-- The number of batteries Tom used in toys -/
def toy_batteries : ℕ := 15

/-- The difference between the number of batteries in toys and flashlights -/
def battery_difference : ℕ := toy_batteries - flashlight_batteries

theorem battery_difference_proof : battery_difference = 13 := by
  sorry

end battery_difference_proof_l3752_375223


namespace factorization_equality_l3752_375263

theorem factorization_equality (a b : ℝ) : 9 * a * b - a^3 * b = a * b * (3 + a) * (3 - a) := by
  sorry

end factorization_equality_l3752_375263


namespace club_officer_selection_l3752_375207

/-- The number of ways to select three distinct positions from a group of n people --/
def selectThreePositions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The number of club members --/
def clubMembers : ℕ := 12

theorem club_officer_selection :
  selectThreePositions clubMembers = 1320 := by
  sorry

end club_officer_selection_l3752_375207


namespace balance_weights_l3752_375295

def weights : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def target_weight : ℕ := 1998

theorem balance_weights :
  (∀ w ∈ weights, is_power_of_two w) →
  (∃ subset : List ℕ, subset.Subset weights ∧ subset.sum = target_weight ∧ subset.length = 8) ∧
  (∀ subset : List ℕ, subset.Subset weights → subset.sum = target_weight → subset.length ≥ 8) :=
by sorry

end balance_weights_l3752_375295


namespace waiter_customers_theorem_l3752_375212

/-- Calculates the final number of customers for a waiter --/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Proves that the final number of customers is correct --/
theorem waiter_customers_theorem (initial left new : ℕ) 
  (h1 : initial ≥ left) : 
  final_customers initial left new = initial - left + new :=
by
  sorry

end waiter_customers_theorem_l3752_375212


namespace albert_run_distance_l3752_375288

/-- Calculates the total distance run on a circular track -/
def totalDistance (trackLength : ℕ) (lapsRun : ℕ) (additionalLaps : ℕ) : ℕ :=
  trackLength * (lapsRun + additionalLaps)

/-- Proves that running 11 laps on a 9-meter track results in 99 meters total distance -/
theorem albert_run_distance :
  totalDistance 9 6 5 = 99 := by
  sorry

end albert_run_distance_l3752_375288


namespace triangle_area_combinations_l3752_375205

theorem triangle_area_combinations (a b c : ℝ) (A B C : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a = Real.sqrt 3 ∧ b = 2 ∧ (Real.sin B + Real.sin C) / Real.sin A = (a + c) / (b - c) →
    1/2 * a * c * Real.sin B = 3 * (Real.sqrt 7 - Real.sqrt 3) / 8) ∧
  (a = Real.sqrt 3 ∧ b = 2 ∧ Real.cos ((B - C) / 2)^2 - Real.sin B * Real.sin C = 1/4 →
    1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) :=
by sorry

end triangle_area_combinations_l3752_375205


namespace min_value_circle_l3752_375234

theorem min_value_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 1 = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ 
  m = x^2 + y^2 ∧ m = 7 - 4*Real.sqrt 3 :=
sorry

end min_value_circle_l3752_375234


namespace profit_percent_calculation_l3752_375235

/-- 
Proves that the profit percent is 32% when selling an article at a certain price, 
given that selling at 2/3 of that price results in a 12% loss.
-/
theorem profit_percent_calculation 
  (P : ℝ) -- The selling price
  (C : ℝ) -- The cost price
  (h : (2/3) * P = 0.88 * C) -- Condition: selling at 2/3 of P results in a 12% loss
  : (P - C) / C * 100 = 32 := by
  sorry

end profit_percent_calculation_l3752_375235


namespace harriet_ran_approximately_45_miles_l3752_375239

/-- The total distance run by six runners -/
def total_distance : ℝ := 378.5

/-- The distance run by Katarina -/
def katarina_distance : ℝ := 47.5

/-- The distance run by Adriana -/
def adriana_distance : ℝ := 83.25

/-- The distance run by Jeremy -/
def jeremy_distance : ℝ := 92.75

/-- The difference in distance between Tomas, Tyler, and Harriet -/
def difference : ℝ := 6.5

/-- Harriet's approximate distance -/
def harriet_distance : ℕ := 45

theorem harriet_ran_approximately_45_miles :
  ∃ (tomas_distance tyler_distance harriet_exact_distance : ℝ),
    tomas_distance ≠ tyler_distance ∧
    tyler_distance ≠ harriet_exact_distance ∧
    tomas_distance ≠ harriet_exact_distance ∧
    (tomas_distance = tyler_distance + difference ∨ tyler_distance = tomas_distance + difference) ∧
    (tyler_distance = harriet_exact_distance + difference ∨ harriet_exact_distance = tyler_distance + difference) ∧
    tomas_distance + tyler_distance + harriet_exact_distance + katarina_distance + adriana_distance + jeremy_distance = total_distance ∧
    harriet_distance = round harriet_exact_distance :=
by
  sorry

end harriet_ran_approximately_45_miles_l3752_375239


namespace fifteenth_term_of_sequence_l3752_375278

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 17) (h₃ : a₃ = 31) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 199 := by
  sorry

#check fifteenth_term_of_sequence

end fifteenth_term_of_sequence_l3752_375278


namespace total_apples_l3752_375211

def marin_apples : ℕ := 8

def david_apples : ℚ := (3/4) * marin_apples

def amanda_apples : ℚ := 1.5 * david_apples + 2

theorem total_apples : marin_apples + david_apples + amanda_apples = 25 := by
  sorry

end total_apples_l3752_375211


namespace complex_roots_to_real_pair_l3752_375249

theorem complex_roots_to_real_pair :
  ∀ (a b : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 3 * Complex.I) * (a + 3 * Complex.I) - (12 + 15 * Complex.I) * (a + 3 * Complex.I) + (50 + 29 * Complex.I) = 0 →
  (b + 6 * Complex.I) * (b + 6 * Complex.I) - (12 + 15 * Complex.I) * (b + 6 * Complex.I) + (50 + 29 * Complex.I) = 0 →
  a = 5 / 3 ∧ b = 31 / 3 := by
  sorry

end complex_roots_to_real_pair_l3752_375249


namespace community_center_tables_l3752_375204

/-- The number of chairs per table -/
def chairs_per_table : ℕ := 8

/-- The number of legs per chair -/
def legs_per_chair : ℕ := 4

/-- The number of legs per table -/
def legs_per_table : ℕ := 3

/-- The total number of legs from all chairs and tables -/
def total_legs : ℕ := 759

/-- The number of tables in the community center -/
def num_tables : ℕ := 22

theorem community_center_tables :
  chairs_per_table * num_tables * legs_per_chair + num_tables * legs_per_table = total_legs :=
sorry

end community_center_tables_l3752_375204


namespace log_343_property_l3752_375253

theorem log_343_property (x : ℝ) (h : Real.log (343 : ℝ) / Real.log (3 * x) = x) :
  (∃ (a b : ℤ), x = (a : ℝ) / (b : ℝ)) ∧ 
  (∀ (n : ℕ), n ≥ 2 → ¬∃ (m : ℤ), x = (m : ℝ) ^ (1 / n : ℝ)) ∧
  (¬∃ (n : ℤ), x = (n : ℝ)) := by
  sorry

end log_343_property_l3752_375253


namespace people_in_hall_l3752_375237

theorem people_in_hall (total_chairs : ℕ) (seated_people : ℕ) (empty_chairs : ℕ) :
  seated_people = (5 : ℕ) * total_chairs / 8 →
  empty_chairs = 8 →
  seated_people = total_chairs - empty_chairs →
  seated_people * 2 = 80 :=
by
  sorry

end people_in_hall_l3752_375237


namespace arithmetic_sequence_n_value_l3752_375247

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. --/
def isArithmeticSequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_n_value
  (a : ℕ → ℤ) (d : ℤ) (h : isArithmeticSequence a d)
  (h2 : a 2 = 12) (hn : a n = -20) (hd : d = -2) :
  n = 18 := by
  sorry

end arithmetic_sequence_n_value_l3752_375247


namespace money_distribution_l3752_375275

theorem money_distribution (total : ℝ) (p q r : ℝ) : 
  total = 4000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 1600 := by
sorry

end money_distribution_l3752_375275


namespace line_outside_circle_l3752_375266

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- A line with a given distance from a point -/
structure Line where
  distanceFromPoint : ℝ

/-- Relationship between a line and a circle -/
inductive Relationship
  | inside
  | tangent
  | outside

/-- Function to determine the relationship between a line and a circle -/
def relationshipBetweenLineAndCircle (c : Circle) (l : Line) : Relationship :=
  sorry

/-- Theorem stating that a line is outside a circle under given conditions -/
theorem line_outside_circle (c : Circle) (l : Line) 
  (h1 : c.diameter = 4)
  (h2 : l.distanceFromPoint = 3) :
  relationshipBetweenLineAndCircle c l = Relationship.outside :=
sorry

end line_outside_circle_l3752_375266


namespace circle_and_line_theorem_l3752_375203

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 0 ∨ y = -3/4 * x

-- Define the point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 1

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

theorem circle_and_line_theorem :
  -- Circle C passes through point A
  circle_C point_A.1 point_A.2 →
  -- Circle C is tangent to the line x+y=1
  ∃ (x y : ℝ), circle_C x y ∧ tangent_line x y →
  -- The center of the circle lies on the line y=-2x
  ∃ (x y : ℝ), circle_C x y ∧ center_line x y →
  -- Line l passes through the origin
  ∃ (x y : ℝ), line_l x y ∧ (x, y) = origin →
  -- The chord intercepted by circle C on line l has a length of 2
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 →
  -- Conclusion: The equations of circle C and line l are correct
  (∀ (x y : ℝ), circle_C x y ↔ (x - 1)^2 + (y + 2)^2 = 2) ∧
  (∀ (x y : ℝ), line_l x y ↔ (x = 0 ∨ y = -3/4 * x)) :=
by
  sorry


end circle_and_line_theorem_l3752_375203


namespace inequality_and_equality_proof_l3752_375214

theorem inequality_and_equality_proof :
  (∀ a b : ℝ, (a^2 + 1) * (b^2 + 1) + 50 ≥ 2 * (2*a + 1) * (3*b + 1)) ∧
  (∀ n p : ℕ+, (n^2 + 1) * (p^2 + 1) + 45 = 2 * (2*n + 1) * (3*p + 1) ↔ n = 2 ∧ p = 2) :=
by sorry

end inequality_and_equality_proof_l3752_375214


namespace target_walmart_tool_difference_l3752_375225

/-- Represents a multitool with its components -/
structure Multitool where
  screwdrivers : Nat
  knives : Nat
  files : Nat
  scissors : Nat
  other_tools : Nat

/-- The Walmart multitool -/
def walmart_multitool : Multitool :=
  { screwdrivers := 1
    knives := 3
    files := 0
    scissors := 0
    other_tools := 2 }

/-- The Target multitool -/
def target_multitool : Multitool :=
  { screwdrivers := 1
    knives := 2 * walmart_multitool.knives
    files := 3
    scissors := 1
    other_tools := 0 }

/-- Total number of tools in a multitool -/
def total_tools (m : Multitool) : Nat :=
  m.screwdrivers + m.knives + m.files + m.scissors + m.other_tools

/-- Theorem stating the difference in the number of tools between Target and Walmart multitools -/
theorem target_walmart_tool_difference :
  total_tools target_multitool - total_tools walmart_multitool = 5 := by
  sorry


end target_walmart_tool_difference_l3752_375225


namespace solution_count_l3752_375291

/-- The number of positive integer solutions to the equation 3x + 4y = 1024 -/
def num_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 1024 ∧ p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 1025) (Finset.range 1025))).card

theorem solution_count : num_solutions = 85 := by
  sorry

end solution_count_l3752_375291


namespace ceiling_of_negative_three_point_six_l3752_375218

theorem ceiling_of_negative_three_point_six :
  ⌈(-3.6 : ℝ)⌉ = -3 := by sorry

end ceiling_of_negative_three_point_six_l3752_375218


namespace max_nondegenerate_triangles_l3752_375243

/-- Represents a triangle with colored sides -/
structure ColoredTriangle where
  blue : ℝ
  red : ℝ
  white : ℝ
  is_nondegenerate : blue + red > white ∧ blue + white > red ∧ red + white > blue

/-- The number of triangles -/
def num_triangles : ℕ := 2009

/-- A collection of 2009 non-degenerated triangles with colored sides -/
def triangle_collection : Fin num_triangles → ColoredTriangle := sorry

/-- Sorted blue sides -/
def sorted_blue : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).blue

/-- Sorted red sides -/
def sorted_red : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).red

/-- Sorted white sides -/
def sorted_white : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).white

/-- Sides are sorted in non-decreasing order -/
axiom sides_sorted : 
  (∀ i j, i ≤ j → sorted_blue i ≤ sorted_blue j) ∧
  (∀ i j, i ≤ j → sorted_red i ≤ sorted_red j) ∧
  (∀ i j, i ≤ j → sorted_white i ≤ sorted_white j)

/-- The main theorem: The maximum number of indices for which we can form non-degenerated triangles is 2009 -/
theorem max_nondegenerate_triangles : 
  (∃ f : Fin num_triangles → Fin num_triangles, 
    Function.Injective f ∧
    ∀ i, (sorted_blue (f i) + sorted_red (f i) > sorted_white (f i)) ∧
         (sorted_blue (f i) + sorted_white (f i) > sorted_red (f i)) ∧
         (sorted_red (f i) + sorted_white (f i) > sorted_blue (f i))) ∧
  (∀ k > num_triangles, ¬∃ f : Fin k → Fin num_triangles, 
    Function.Injective f ∧
    ∀ i, (sorted_blue (f i) + sorted_red (f i) > sorted_white (f i)) ∧
         (sorted_blue (f i) + sorted_white (f i) > sorted_red (f i)) ∧
         (sorted_red (f i) + sorted_white (f i) > sorted_blue (f i))) :=
by sorry

end max_nondegenerate_triangles_l3752_375243


namespace range_of_a_l3752_375228

-- Define propositions p and q
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1

def q (x a : ℝ) : Prop := x^2 + (2*a + 1)*x + a*(a + 1) ≥ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- Theorem statement
theorem range_of_a :
  {a : ℝ | sufficient_not_necessary a} = {a | a ≤ -4 ∨ a ≥ -1} :=
sorry

end range_of_a_l3752_375228


namespace cube_root_problem_l3752_375262

theorem cube_root_problem (a : ℕ) (h : a^3 = 21 * 25 * 315 * 7) : a = 105 := by
  sorry

end cube_root_problem_l3752_375262


namespace average_of_three_numbers_l3752_375282

theorem average_of_three_numbers (a : ℝ) : 
  (3 + a + 10) / 3 = 5 → a = 2 := by
  sorry

end average_of_three_numbers_l3752_375282


namespace dodgeball_team_theorem_l3752_375250

/-- The number of players in the dodgeball league -/
def total_players : ℕ := 12

/-- The number of players on each team -/
def team_size : ℕ := 6

/-- The number of times two specific players are on the same team -/
def same_team_count : ℕ := 210

/-- The total number of possible team combinations -/
def total_combinations : ℕ := Nat.choose total_players team_size

theorem dodgeball_team_theorem :
  ∀ (player1 player2 : Fin total_players),
    player1 ≠ player2 →
    (Nat.choose (total_players - 2) (team_size - 2) : ℕ) = same_team_count :=
by sorry

end dodgeball_team_theorem_l3752_375250


namespace share_calculation_l3752_375240

/-- Proves that given Debby takes 25% of a total sum and Maggie takes the rest,
if Maggie's share is $4,500, then the total sum is $6,000. -/
theorem share_calculation (total : ℝ) (debby_share : ℝ) (maggie_share : ℝ) : 
  debby_share = 0.25 * total →
  maggie_share = total - debby_share →
  maggie_share = 4500 →
  total = 6000 := by
  sorry

end share_calculation_l3752_375240


namespace quadratic_symmetry_l3752_375209

/-- A quadratic function passing through the points (1, 8), (3, -1), and (5, 8) -/
def quadratic_function (x : ℝ) : ℝ := sorry

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry : ℝ := 3

theorem quadratic_symmetry :
  (quadratic_function 1 = 8) ∧
  (quadratic_function 3 = -1) ∧
  (quadratic_function 5 = 8) →
  axis_of_symmetry = 3 := by
  sorry

end quadratic_symmetry_l3752_375209


namespace gcf_three_digit_palindromes_l3752_375244

/-- A three-digit palindrome -/
def ThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 102 * a + 10 * b

/-- The greatest common factor of all three-digit palindromes is 1 -/
theorem gcf_three_digit_palindromes :
  ∃ (g : ℕ), g > 0 ∧ 
    (∀ n : ℕ, ThreeDigitPalindrome n → g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n : ℕ, ThreeDigitPalindrome n → d ∣ n) → d ≤ g) ∧
    g = 1 :=
sorry

end gcf_three_digit_palindromes_l3752_375244


namespace archie_marbles_problem_l3752_375202

/-- The number of marbles Archie started with. -/
def initial_marbles : ℕ := 100

/-- The fraction of marbles Archie keeps after losing some in the street. -/
def street_loss_fraction : ℚ := 2/5

/-- The fraction of remaining marbles Archie keeps after losing some in the sewer. -/
def sewer_loss_fraction : ℚ := 1/2

/-- The number of marbles Archie has left at the end. -/
def final_marbles : ℕ := 20

theorem archie_marbles_problem :
  (↑final_marbles : ℚ) = ↑initial_marbles * street_loss_fraction * sewer_loss_fraction :=
by sorry

end archie_marbles_problem_l3752_375202


namespace find_b_plus_c_l3752_375230

theorem find_b_plus_c (a b c d : ℚ)
  (eq1 : a * b + a * c + b * d + c * d = 40)
  (eq2 : a + d = 6)
  (eq3 : a * b + b * c + c * d + d * a = 28) :
  b + c = 17 / 3 := by
sorry

end find_b_plus_c_l3752_375230


namespace forgotten_poems_sally_forgotten_poems_l3752_375273

/-- Given the number of initially memorized poems and the number of poems that can be recited,
    prove that the number of forgotten poems is their difference. -/
theorem forgotten_poems (initially_memorized recitable : ℕ) :
  initially_memorized ≥ recitable →
  initially_memorized - recitable = initially_memorized - recitable :=
by
  sorry

/-- Application to Sally's specific case -/
theorem sally_forgotten_poems :
  let initially_memorized := 8
  let recitable := 3
  initially_memorized - recitable = 5 :=
by
  sorry

end forgotten_poems_sally_forgotten_poems_l3752_375273


namespace three_digit_number_relation_l3752_375229

theorem three_digit_number_relation (h t u : ℕ) : 
  h ≥ 1 ∧ h ≤ 9 ∧  -- h is a single digit
  t ≥ 0 ∧ t ≤ 9 ∧  -- t is a single digit
  u ≥ 0 ∧ u ≤ 9 ∧  -- u is a single digit
  h = t + 2 ∧      -- hundreds digit is 2 more than tens digit
  h + t + u = 27   -- sum of digits is 27
  → ∃ (r : ℕ → ℕ → Prop), r t u  -- there exists some relation r between t and u
:= by sorry

end three_digit_number_relation_l3752_375229


namespace robot_types_count_l3752_375286

theorem robot_types_count (shapes : ℕ) (colors : ℕ) (h1 : shapes = 3) (h2 : colors = 4) :
  shapes * colors = 12 := by
  sorry

end robot_types_count_l3752_375286


namespace min_perimeter_two_isosceles_triangles_l3752_375251

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.side : ℚ) ^ 2 - ((t.base : ℚ) / 2) ^ 2).sqrt) / 4

theorem min_perimeter_two_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
      perimeter t1 = 524 :=
by sorry

end min_perimeter_two_isosceles_triangles_l3752_375251


namespace carwash_solution_l3752_375276

/-- Represents the carwash problem --/
structure CarWash where
  car_price : ℕ
  truck_price : ℕ
  suv_price : ℕ
  total_raised : ℕ
  num_suvs : ℕ
  num_trucks : ℕ

/-- Calculates the number of cars washed --/
def cars_washed (cw : CarWash) : ℕ :=
  (cw.total_raised - (cw.suv_price * cw.num_suvs + cw.truck_price * cw.num_trucks)) / cw.car_price

/-- Theorem stating the solution to the carwash problem --/
theorem carwash_solution (cw : CarWash) 
  (h1 : cw.car_price = 5)
  (h2 : cw.truck_price = 6)
  (h3 : cw.suv_price = 7)
  (h4 : cw.total_raised = 100)
  (h5 : cw.num_suvs = 5)
  (h6 : cw.num_trucks = 5) :
  cars_washed cw = 7 := by
  sorry

#eval cars_washed ⟨5, 6, 7, 100, 5, 5⟩

end carwash_solution_l3752_375276


namespace pet_sitting_earnings_l3752_375294

def hourly_rate : ℕ := 5
def hours_week1 : ℕ := 20
def hours_week2 : ℕ := 30

theorem pet_sitting_earnings : 
  hourly_rate * (hours_week1 + hours_week2) = 250 := by
  sorry

end pet_sitting_earnings_l3752_375294


namespace average_math_score_l3752_375293

/-- Represents the total number of students -/
def total_students : ℕ := 500

/-- Represents the number of male students -/
def male_students : ℕ := 300

/-- Represents the number of female students -/
def female_students : ℕ := 200

/-- Represents the sample size -/
def sample_size : ℕ := 60

/-- Represents the average score of male students in the sample -/
def male_avg_score : ℝ := 110

/-- Represents the average score of female students in the sample -/
def female_avg_score : ℝ := 100

/-- Theorem stating that the average math score of first-year students is 106 points -/
theorem average_math_score : 
  (male_students : ℝ) / total_students * male_avg_score + 
  (female_students : ℝ) / total_students * female_avg_score = 106 := by
  sorry

end average_math_score_l3752_375293


namespace chess_piece_loss_l3752_375242

theorem chess_piece_loss (total_pieces : ℕ) (arianna_lost : ℕ) : 
  total_pieces = 20 →
  arianna_lost = 3 →
  32 - total_pieces = arianna_lost + 9 :=
by
  sorry

end chess_piece_loss_l3752_375242


namespace onions_sum_is_eighteen_l3752_375226

/-- The total number of onions grown by Sara, Sally, and Fred -/
def total_onions (sara_onions sally_onions fred_onions : ℕ) : ℕ :=
  sara_onions + sally_onions + fred_onions

/-- Theorem stating that the total number of onions grown is 18 -/
theorem onions_sum_is_eighteen :
  total_onions 4 5 9 = 18 := by
  sorry

end onions_sum_is_eighteen_l3752_375226


namespace complement_union_problem_l3752_375241

def U : Set ℝ := {-2, -8, 0, Real.pi, 6, 10}
def A : Set ℝ := {-2, Real.pi, 6}
def B : Set ℝ := {1}

theorem complement_union_problem : (U \ A) ∪ B = {0, 1, -8, 10} := by sorry

end complement_union_problem_l3752_375241


namespace more_likely_same_l3752_375245

/-- Represents the number of crows on each tree -/
structure CrowCounts where
  white_birch : ℕ
  black_birch : ℕ
  white_oak : ℕ
  black_oak : ℕ

/-- Conditions from the problem -/
def valid_crow_counts (c : CrowCounts) : Prop :=
  c.white_birch > 0 ∧
  c.white_birch + c.black_birch = 50 ∧
  c.white_oak + c.black_oak = 50 ∧
  c.black_birch ≥ c.white_birch ∧
  c.black_oak ≥ c.white_oak - 1

/-- Probability of number of white crows on birch remaining the same -/
def prob_same (c : CrowCounts) : ℚ :=
  (c.black_birch * (c.black_oak + 1) + c.white_birch * (c.white_oak + 1)) / 2550

/-- Probability of number of white crows on birch changing -/
def prob_change (c : CrowCounts) : ℚ :=
  (c.black_birch * c.white_oak + c.white_birch * c.black_oak) / 2550

/-- Theorem stating that it's more likely for the number of white crows to remain the same -/
theorem more_likely_same (c : CrowCounts) (h : valid_crow_counts c) :
  prob_same c > prob_change c :=
sorry

end more_likely_same_l3752_375245


namespace probability_same_number_l3752_375220

def emily_options : ℕ := 250 / 20
def eli_options : ℕ := 250 / 30
def common_options : ℕ := 250 / 60

theorem probability_same_number : 
  (emily_options : ℚ) * eli_options ≠ 0 →
  (common_options : ℚ) / (emily_options * eli_options) = 1 / 24 := by
  sorry

end probability_same_number_l3752_375220


namespace no_three_consecutive_digit_sum_squares_l3752_375215

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Theorem: There do not exist three consecutive integers such that 
    the sum of digits of each is a perfect square -/
theorem no_three_consecutive_digit_sum_squares :
  ¬ ∃ n : ℕ, (is_perfect_square (sum_of_digits n)) ∧ 
             (is_perfect_square (sum_of_digits (n + 1))) ∧ 
             (is_perfect_square (sum_of_digits (n + 2))) :=
sorry

end no_three_consecutive_digit_sum_squares_l3752_375215


namespace algorithm_swaps_values_l3752_375201

-- Define the algorithm steps
def algorithm (x y : ℝ) : ℝ × ℝ :=
  let z := x
  let x' := y
  let y' := z
  (x', y')

-- Theorem statement
theorem algorithm_swaps_values (x y : ℝ) :
  algorithm x y = (y, x) := by sorry

end algorithm_swaps_values_l3752_375201


namespace baseball_hits_percentage_l3752_375292

theorem baseball_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 10) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 70 := by
  sorry

end baseball_hits_percentage_l3752_375292


namespace equation_solution_l3752_375298

theorem equation_solution (x : ℝ) : 
  Real.sqrt (9 + Real.sqrt (27 + 3*x)) + Real.sqrt (3 + Real.sqrt (9 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 1 := by
sorry

end equation_solution_l3752_375298


namespace smallest_n_with_75_divisors_l3752_375265

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_n_with_75_divisors :
  ∃ n : ℕ, 
    is_multiple_of_75 n ∧ 
    count_divisors n = 75 ∧ 
    (∀ m : ℕ, m < n → ¬(is_multiple_of_75 m ∧ count_divisors m = 75)) ∧
    n / 75 = 432 :=
sorry

end smallest_n_with_75_divisors_l3752_375265


namespace basis_from_noncoplanar_vectors_l3752_375219

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem basis_from_noncoplanar_vectors (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) :
  LinearIndependent ℝ ![a + b, b - a, c] :=
sorry

end basis_from_noncoplanar_vectors_l3752_375219


namespace sum_of_max_min_g_l3752_375246

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8| + |x - 7|

-- Define the interval [3, 9]
def I : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}

-- State the theorem
theorem sum_of_max_min_g : 
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ I, g x ≤ max_val) ∧
    (∃ x ∈ I, g x = max_val) ∧
    (∀ x ∈ I, min_val ≤ g x) ∧
    (∃ x ∈ I, g x = min_val) ∧
    max_val + min_val = 14 :=
sorry

end sum_of_max_min_g_l3752_375246


namespace two_statements_correct_l3752_375227

-- Define a structure for a line in 2D plane
structure Line where
  slope : Option ℝ
  angle_of_inclination : ℝ

-- Define parallel and perpendicular relations
def parallel (l₁ l₂ : Line) : Prop := sorry

def perpendicular (l₁ l₂ : Line) : Prop := sorry

-- Define the four statements
def statement1 (l₁ l₂ : Line) : Prop :=
  (l₁.slope.isSome ∧ l₂.slope.isSome ∧ l₁.slope = l₂.slope) → parallel l₁ l₂

def statement2 (l₁ l₂ : Line) : Prop :=
  perpendicular l₁ l₂ →
    (l₁.slope.isSome ∧ l₂.slope.isSome ∧
     ∃ (s₁ s₂ : ℝ), l₁.slope = some s₁ ∧ l₂.slope = some s₂ ∧ s₁ * s₂ = -1)

def statement3 (l₁ l₂ : Line) : Prop :=
  l₁.angle_of_inclination = l₂.angle_of_inclination → parallel l₁ l₂

def statement4 : Prop :=
  ∀ (l₁ l₂ : Line), parallel l₁ l₂ → (l₁.slope.isSome ∧ l₂.slope.isSome ∧ l₁.slope = l₂.slope)

theorem two_statements_correct (l₁ l₂ : Line) (h : l₁ ≠ l₂) :
  (statement1 l₁ l₂ ∧ statement3 l₁ l₂ ∧ ¬statement2 l₁ l₂ ∧ ¬statement4) := by
  sorry

end two_statements_correct_l3752_375227


namespace recipe_total_cups_l3752_375299

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients used in a recipe given the ratio and cups of sugar -/
def total_cups (ratio : RecipeRatio) (sugar_cups : ℕ) : ℕ :=
  let part_size := sugar_cups / ratio.sugar
  part_size * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for a recipe with ratio 1:8:5 and 10 cups of sugar, the total cups used is 28 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨1, 8, 5⟩
  let sugar_cups : ℕ := 10
  total_cups ratio sugar_cups = 28 := by
  sorry

end recipe_total_cups_l3752_375299


namespace keith_card_spending_l3752_375272

/-- Represents the total amount spent on trading cards -/
def total_spent (digimon_packs pokemon_packs yugioh_packs magic_packs : ℕ) 
  (digimon_price pokemon_price yugioh_price magic_price baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + 
  pokemon_packs * pokemon_price + 
  yugioh_packs * yugioh_price + 
  magic_packs * magic_price + 
  baseball_price

/-- Theorem stating the total amount Keith spent on cards -/
theorem keith_card_spending :
  total_spent 4 3 6 2 4.45 5.25 3.99 6.75 6.06 = 77.05 := by
  sorry

end keith_card_spending_l3752_375272


namespace cake_division_l3752_375284

theorem cake_division (x y z : ℚ) :
  x + y + z = 1 →
  2 * z = x →
  z = (1/2) * (y + (2/3) * x) →
  (2/3) * x = 4/11 :=
by
  sorry

end cake_division_l3752_375284


namespace perfect_square_condition_l3752_375254

theorem perfect_square_condition (n : ℕ) (h : n > 0) :
  (∃ k : ℤ, 2 + 2 * Real.sqrt (1 + 12 * n^2) = k) →
  ∃ m : ℕ, n = m^2 :=
by sorry

end perfect_square_condition_l3752_375254


namespace sum_of_palindromic_primes_less_than_70_l3752_375217

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def isPalindromicPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 70 ∧ isPrime n ∧ isPrime (reverseDigits n) ∧ reverseDigits n < 70

def sumOfPalindromicPrimes : ℕ := sorry

theorem sum_of_palindromic_primes_less_than_70 :
  sumOfPalindromicPrimes = 92 := by sorry

end sum_of_palindromic_primes_less_than_70_l3752_375217


namespace mixed_fruit_cost_calculation_l3752_375279

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 34

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 22.666666666666668

/-- The cost per litre of mixed fruit juice -/
def mixed_fruit_cost : ℝ := 264.1764705882353

theorem mixed_fruit_cost_calculation :
  mixed_fruit_cost * mixed_fruit_volume + acai_cost * acai_volume = 
  cocktail_cost * (mixed_fruit_volume + acai_volume) := by sorry

end mixed_fruit_cost_calculation_l3752_375279


namespace consecutive_integers_square_sum_l3752_375285

theorem consecutive_integers_square_sum (n : ℕ) : 
  (n > 0) → (n^2 + (n+1)^2 = n*(n+1) + 91) → (n+1 = 10) := by
  sorry

end consecutive_integers_square_sum_l3752_375285


namespace triangle_max_area_l3752_375248

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  ∀ (area : ℝ), area = (1/2) * b * c * Real.sin A → area ≤ Real.sqrt 3 :=
by sorry

end triangle_max_area_l3752_375248


namespace hyperbola_parabola_intersection_l3752_375281

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  e : ℝ  -- eccentricity
  a : ℝ  -- semi-major axis
  f1 : Point  -- left focus
  f2 : Point  -- right focus

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point

/-- The main theorem -/
theorem hyperbola_parabola_intersection (h : Hyperbola) (p : Parabola) (P : Point) (a c : ℝ) :
  h.f1 = p.focus →
  h.f2 = p.vertex →
  (P.x - h.f2.x) ^ 2 + P.y ^ 2 = (h.e * h.a) ^ 2 →  -- P is on the right branch of the hyperbola
  P.y ^ 2 = 2 * h.a * (P.x - h.f2.x) →  -- P is on the parabola
  a * |P.x - h.f2.x| + c * |h.f1.x - P.x| = 8 * a ^ 2 →
  h.e = 8 := by
  sorry

end hyperbola_parabola_intersection_l3752_375281


namespace prob_three_primes_l3752_375270

def num_dice : ℕ := 6
def sides_per_die : ℕ := 12
def prob_prime : ℚ := 5/12

theorem prob_three_primes :
  let choose_three := Nat.choose num_dice 3
  let prob_three_prime := (prob_prime ^ 3 : ℚ)
  let prob_three_non_prime := ((1 - prob_prime) ^ 3 : ℚ)
  choose_three * prob_three_prime * prob_three_non_prime = 312500/248832 := by
sorry

end prob_three_primes_l3752_375270


namespace train_length_l3752_375261

/-- The length of a train given its relative speed and passing time -/
theorem train_length (relative_speed : ℝ) (passing_time : ℝ) : 
  relative_speed = 72 - 36 →
  passing_time = 12 →
  relative_speed * (1000 / 3600) * passing_time = 120 := by
  sorry

#check train_length

end train_length_l3752_375261


namespace adam_katie_miles_difference_l3752_375231

/-- Proves that Adam ran 25 miles more than Katie -/
theorem adam_katie_miles_difference :
  let adam_miles : ℕ := 35
  let katie_miles : ℕ := 10
  adam_miles - katie_miles = 25 := by
  sorry

end adam_katie_miles_difference_l3752_375231


namespace octagon_side_length_l3752_375264

/-- Given an octagon-shaped box with a perimeter of 72 cm, prove that each side length is 9 cm. -/
theorem octagon_side_length (perimeter : ℝ) (num_sides : ℕ) : 
  perimeter = 72 ∧ num_sides = 8 → perimeter / num_sides = 9 := by
  sorry

end octagon_side_length_l3752_375264


namespace infinite_primes_dividing_x_l3752_375290

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPoly := ℕ → ℕ

/-- Definition of x_n -/
def x (P Q : NonNegIntPoly) (n : ℕ) : ℕ := 2016^(P n) + Q n

/-- A number is squarefree if it's not divisible by any prime square -/
def IsSquarefree (m : ℕ) : Prop := ∀ p : ℕ, Nat.Prime p → (p^2 ∣ m) → False

theorem infinite_primes_dividing_x (P Q : NonNegIntPoly) 
  (hP : ¬ ∀ n : ℕ, P n = P 0) 
  (hQ : ¬ ∀ n : ℕ, Q n = Q 0) : 
  ∃ S : Set ℕ, (S.Infinite) ∧ 
  (∀ p ∈ S, Nat.Prime p ∧ ∃ m : ℕ, IsSquarefree m ∧ (p ∣ x P Q m)) := by
  sorry

end infinite_primes_dividing_x_l3752_375290


namespace imaginary_part_of_complex_number_l3752_375224

theorem imaginary_part_of_complex_number :
  (Complex.im ((2 : ℂ) - Complex.I * Complex.I)) = 2 := by sorry

end imaginary_part_of_complex_number_l3752_375224
