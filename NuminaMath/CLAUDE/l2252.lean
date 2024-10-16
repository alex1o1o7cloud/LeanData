import Mathlib

namespace NUMINAMATH_CALUDE_estate_area_calculation_l2252_225248

/-- Represents the scale of the map in miles per inch -/
def scale : ℚ := 300 / 2

/-- The length of the first side of the rectangle on the map in inches -/
def side1_map : ℚ := 10

/-- The length of the second side of the rectangle on the map in inches -/
def side2_map : ℚ := 6

/-- Converts a length on the map to the actual length in miles -/
def map_to_miles (map_length : ℚ) : ℚ := map_length * scale

/-- Calculates the area of a rectangle given its side lengths -/
def rectangle_area (length width : ℚ) : ℚ := length * width

theorem estate_area_calculation :
  rectangle_area (map_to_miles side1_map) (map_to_miles side2_map) = 1350000 := by
  sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l2252_225248


namespace NUMINAMATH_CALUDE_smallest_number_of_groups_result_is_five_l2252_225271

/-- Given a class of 30 students, prove that the smallest number of equal groups
    needed, with each group containing no more than 6 students, is 5. -/
theorem smallest_number_of_groups : ℕ :=
  let total_students : ℕ := 30
  let max_per_group : ℕ := 6
  let number_of_groups : ℕ := total_students / max_per_group
  
  have h1 : total_students = number_of_groups * max_per_group :=
    sorry
  
  have h2 : ∀ k : ℕ, k < number_of_groups → k * max_per_group < total_students :=
    sorry
  
  have h3 : number_of_groups ≤ max_per_group :=
    sorry
  
  number_of_groups

/-- Prove that the result is indeed 5 -/
theorem result_is_five : smallest_number_of_groups = 5 :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_groups_result_is_five_l2252_225271


namespace NUMINAMATH_CALUDE_josiah_cookie_spending_l2252_225260

/-- The number of days in March -/
def days_in_march : ℕ := 31

/-- The number of cookies Josiah buys each day -/
def cookies_per_day : ℕ := 2

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 16

/-- Josiah's total spending on cookies in March -/
def total_spending : ℕ := days_in_march * cookies_per_day * cost_per_cookie

/-- Theorem stating that Josiah's total spending on cookies in March is 992 dollars -/
theorem josiah_cookie_spending : total_spending = 992 := by
  sorry

end NUMINAMATH_CALUDE_josiah_cookie_spending_l2252_225260


namespace NUMINAMATH_CALUDE_fourth_side_length_l2252_225291

/-- A quadrilateral inscribed in a circle with three equal sides -/
structure InscribedQuadrilateral where
  -- The radius of the circumscribed circle
  r : ℝ
  -- The length of three equal sides
  s : ℝ
  -- Assumption that the radius is 150√2
  h1 : r = 150 * Real.sqrt 2
  -- Assumption that the three equal sides have length 150
  h2 : s = 150

/-- The length of the fourth side of the quadrilateral -/
def fourthSide (q : InscribedQuadrilateral) : ℝ := 375

/-- Theorem stating that the fourth side has length 375 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : 
  fourthSide q = 375 := by sorry

end NUMINAMATH_CALUDE_fourth_side_length_l2252_225291


namespace NUMINAMATH_CALUDE_journey_time_calculation_l2252_225284

/-- Calculates the time spent on the road given start time, end time, and total stop time. -/
def timeOnRoad (startTime endTime stopTime : ℕ) : ℕ :=
  (endTime - startTime) - stopTime

/-- Proves that for a journey from 7:00 AM to 8:00 PM with 60 minutes of stops, the time on the road is 12 hours. -/
theorem journey_time_calculation :
  let startTime : ℕ := 7  -- 7:00 AM
  let endTime : ℕ := 20   -- 8:00 PM (20:00 in 24-hour format)
  let stopTime : ℕ := 1   -- 60 minutes = 1 hour
  timeOnRoad startTime endTime stopTime = 12 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l2252_225284


namespace NUMINAMATH_CALUDE_wall_decorations_l2252_225250

theorem wall_decorations (total : ℕ) (nails thumbtacks sticky_strips : ℕ) : 
  (nails : ℚ) = 2/3 * total ∧
  (thumbtacks : ℚ) = 2/5 * (1/3 * total) ∧
  sticky_strips = 15 ∧
  total = nails + thumbtacks + sticky_strips →
  nails = 50 := by
sorry

end NUMINAMATH_CALUDE_wall_decorations_l2252_225250


namespace NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l2252_225295

theorem inverse_true_implies_negation_true (P : Prop) :
  (¬P → ¬P) → (¬P) :=
by sorry

end NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l2252_225295


namespace NUMINAMATH_CALUDE_find_special_number_l2252_225259

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k^2

/-- The problem statement -/
theorem find_special_number : 
  ∃ m : ℕ+, 
    is_perfect_square (m.val + 100) ∧ 
    is_perfect_square (m.val + 168) ∧ 
    m.val = 156 := by
  sorry

end NUMINAMATH_CALUDE_find_special_number_l2252_225259


namespace NUMINAMATH_CALUDE_students_suggesting_both_l2252_225256

/-- Given the total number of students suggesting bacon and the number of students suggesting only bacon,
    prove that the number of students suggesting both mashed potatoes and bacon
    is equal to the difference between these two values. -/
theorem students_suggesting_both (total_bacon : ℕ) (only_bacon : ℕ)
    (h : total_bacon = 569 ∧ only_bacon = 351) :
    total_bacon - only_bacon = 218 := by
  sorry

end NUMINAMATH_CALUDE_students_suggesting_both_l2252_225256


namespace NUMINAMATH_CALUDE_kids_at_reunion_l2252_225296

theorem kids_at_reunion (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h1 : adults = 123)
  (h2 : tables = 14)
  (h3 : people_per_table = 12) :
  tables * people_per_table - adults = 45 :=
by sorry

end NUMINAMATH_CALUDE_kids_at_reunion_l2252_225296


namespace NUMINAMATH_CALUDE_range_of_x_plus_3y_l2252_225224

theorem range_of_x_plus_3y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 4) 
  (h2 : 2 ≤ x - y ∧ x - y ≤ 3) : 
  -5 ≤ x + 3*y ∧ x + 3*y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_plus_3y_l2252_225224


namespace NUMINAMATH_CALUDE_committee_selection_l2252_225288

theorem committee_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 5) : 
  Nat.choose n m = 252 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2252_225288


namespace NUMINAMATH_CALUDE_minimum_k_value_l2252_225242

theorem minimum_k_value : ∃ (k : ℝ), 
  (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → 
    (∃ (a b : ℝ), (a = x ∧ b = y) ∨ (a = x ∧ b = z) ∨ (a = y ∧ b = z) ∧ 
      (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k))) ∧
  (∀ (k' : ℝ), 
    (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → 
      (∃ (a b : ℝ), (a = x ∧ b = y) ∨ (a = x ∧ b = z) ∨ (a = y ∧ b = z) ∧ 
        (|a - b| ≤ k' ∨ |1/a - 1/b| ≤ k'))) → 
    k ≤ k') ∧
  k = 3/2 := by
sorry

end NUMINAMATH_CALUDE_minimum_k_value_l2252_225242


namespace NUMINAMATH_CALUDE_interest_calculation_time_l2252_225210

-- Define the given values
def simple_interest : ℚ := 345/100
def principal : ℚ := 23
def rate_paise : ℚ := 5

-- Convert rate from paise to rupees
def rate : ℚ := rate_paise / 100

-- Define the simple interest formula
def calculate_time (si p r : ℚ) : ℚ := si / (p * r)

-- State the theorem
theorem interest_calculation_time :
  calculate_time simple_interest principal rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_time_l2252_225210


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2252_225213

-- Define the inequality
def inequality (x m : ℝ) : Prop := |x - 1| + |x - m| < 2 * m

-- Define the theorem
theorem empty_solution_set_range :
  (∀ m : ℝ, (0 < m ∧ m < 1/3) ↔ ∀ x : ℝ, ¬(inequality x m)) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2252_225213


namespace NUMINAMATH_CALUDE_exponential_inequality_l2252_225278

theorem exponential_inequality (a x : Real) (h1 : 0 < a) (h2 : a < 1) (h3 : x > 0) :
  a^(-x) > a^x := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2252_225278


namespace NUMINAMATH_CALUDE_original_number_proof_l2252_225222

theorem original_number_proof (n : ℕ) : 
  n - 7 = 62575 ∧ (62575 % 99 = 92) → n = 62582 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2252_225222


namespace NUMINAMATH_CALUDE_valid_m_values_l2252_225205

-- Define the set A
def A (m : ℝ) : Set ℝ := {1, m + 2, m^2 + 4}

-- State the theorem
theorem valid_m_values :
  ∀ m : ℝ, 5 ∈ A m → (m = 1 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_valid_m_values_l2252_225205


namespace NUMINAMATH_CALUDE_tournament_max_k_l2252_225276

def num_teams : ℕ := 20

-- Ice Hockey scoring system
def ice_hockey_max_k (n : ℕ) : ℕ := n - 2

-- Volleyball scoring system
def volleyball_max_k (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 5 else n - 4

theorem tournament_max_k :
  ice_hockey_max_k num_teams = 18 ∧
  volleyball_max_k num_teams = 15 := by
  sorry

#eval ice_hockey_max_k num_teams
#eval volleyball_max_k num_teams

end NUMINAMATH_CALUDE_tournament_max_k_l2252_225276


namespace NUMINAMATH_CALUDE_complex_real_condition_l2252_225203

/-- If z = m^2 - 1 + (m-1)i is a real number and m is real, then m = 1 -/
theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := m^2 - 1 + (m - 1) * Complex.I
  (z.im = 0) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2252_225203


namespace NUMINAMATH_CALUDE_annas_ebook_readers_l2252_225247

theorem annas_ebook_readers (anna_readers john_initial_readers john_final_readers total_readers : ℕ) 
  (h1 : john_initial_readers = anna_readers - 15)
  (h2 : john_final_readers = john_initial_readers - 3)
  (h3 : anna_readers + john_final_readers = total_readers)
  (h4 : total_readers = 82) : anna_readers = 50 := by
  sorry

end NUMINAMATH_CALUDE_annas_ebook_readers_l2252_225247


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l2252_225293

/-- The height difference between Janet's sandcastle and her sister's sandcastle -/
def height_difference : ℝ := 1.333333333333333

/-- Janet's sandcastle height in feet -/
def janet_height : ℝ := 3.6666666666666665

/-- Janet's sister's sandcastle height in feet -/
def sister_height : ℝ := 2.3333333333333335

/-- Theorem stating that the height difference between Janet's sandcastle and her sister's sandcastle
    is equal to Janet's sandcastle height minus her sister's sandcastle height -/
theorem sandcastle_height_difference :
  height_difference = janet_height - sister_height := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l2252_225293


namespace NUMINAMATH_CALUDE_set_A_characterization_l2252_225282

theorem set_A_characterization (A : Set ℕ) : 
  ({1} ∪ A = {1, 3, 5}) → (A = {1, 3, 5} ∨ A = {3, 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_A_characterization_l2252_225282


namespace NUMINAMATH_CALUDE_pirate_treasure_l2252_225221

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l2252_225221


namespace NUMINAMATH_CALUDE_virus_radius_scientific_notation_l2252_225266

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The radius of the virus in meters -/
def virus_radius : ℝ := 0.00000000495

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_radius_scientific_notation :
  to_scientific_notation virus_radius = ScientificNotation.mk 4.95 (-9) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_virus_radius_scientific_notation_l2252_225266


namespace NUMINAMATH_CALUDE_solve_equation_l2252_225267

theorem solve_equation : ∃ x : ℝ, (4 / 7) * (1 / 5) * x = 2 ∧ x = 35 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2252_225267


namespace NUMINAMATH_CALUDE_race_distances_l2252_225246

/-- In a 100 m race, if B beats C by 4 m and A beats C by 28 m, then A beats B by 24 m. -/
theorem race_distances (x : ℝ) : 
  (100 : ℝ) - x - 4 = 100 - 28 → x = 24 := by sorry

end NUMINAMATH_CALUDE_race_distances_l2252_225246


namespace NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l2252_225287

theorem base_10_to_base_7_conversion :
  ∃ (a b c d : ℕ),
    746 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l2252_225287


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2252_225274

theorem and_sufficient_not_necessary_for_or :
  (∃ p q : Prop, p ∧ q → p ∨ q) ∧
  (∃ p q : Prop, p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2252_225274


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2252_225208

/-- A type representing the scientists -/
def Scientist : Type := Fin 17

/-- A type representing the topics -/
inductive Topic
| A
| B
| C

/-- A function representing the correspondence between scientists on topics -/
def correspondence : Scientist → Scientist → Topic := sorry

/-- The main theorem stating that there exists a monochromatic triangle -/
theorem monochromatic_triangle_exists :
  ∃ (s1 s2 s3 : Scientist) (t : Topic),
    s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3 ∧
    correspondence s1 s2 = t ∧
    correspondence s1 s3 = t ∧
    correspondence s2 s3 = t :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2252_225208


namespace NUMINAMATH_CALUDE_car_uphill_speed_l2252_225261

/-- Given a car's travel information, prove that its uphill speed is 80 km/hour. -/
theorem car_uphill_speed
  (downhill_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time : ℝ)
  (h1 : downhill_speed = 50)
  (h2 : total_time = 15)
  (h3 : total_distance = 650)
  (h4 : downhill_time = 5)
  (h5 : uphill_time = 5)
  : ∃ (uphill_speed : ℝ), uphill_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_car_uphill_speed_l2252_225261


namespace NUMINAMATH_CALUDE_tank_capacity_l2252_225232

/-- Represents the flow rate in kiloliters per minute -/
def flow_rate (volume : ℚ) (time : ℚ) : ℚ := volume / time

/-- Calculates the net flow rate into the tank -/
def net_flow_rate (fill_rate drain_rate1 drain_rate2 : ℚ) : ℚ :=
  fill_rate - (drain_rate1 + drain_rate2)

/-- Calculates the amount of water added to the tank -/
def water_added (net_rate : ℚ) (time : ℚ) : ℚ := net_rate * time

/-- Converts kiloliters to liters -/
def kiloliters_to_liters (kl : ℚ) : ℚ := kl * 1000

theorem tank_capacity :
  let fill_rate := flow_rate 1 2
  let drain_rate1 := flow_rate 1 4
  let drain_rate2 := flow_rate 1 6
  let net_rate := net_flow_rate fill_rate drain_rate1 drain_rate2
  let added_water := water_added net_rate 36
  let full_capacity := 2 * added_water
  kiloliters_to_liters full_capacity = 6000 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2252_225232


namespace NUMINAMATH_CALUDE_cricket_bat_theorem_l2252_225281

def cricket_bat_problem (a_cost_price b_selling_price c_purchase_price : ℝ) 
  (a_profit_percentage : ℝ) : Prop :=
  let a_selling_price := a_cost_price * (1 + a_profit_percentage)
  let b_profit := c_purchase_price - a_selling_price
  let b_profit_percentage := b_profit / a_selling_price * 100
  a_cost_price = 156 ∧ 
  a_profit_percentage = 0.20 ∧ 
  c_purchase_price = 234 → 
  b_profit_percentage = 25

theorem cricket_bat_theorem : 
  ∃ (a_cost_price b_selling_price c_purchase_price : ℝ),
    cricket_bat_problem a_cost_price b_selling_price c_purchase_price 0.20 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_theorem_l2252_225281


namespace NUMINAMATH_CALUDE_max_value_of_x_l2252_225283

theorem max_value_of_x (x : ℝ) : 
  (((4 * x - 16) / (3 * x - 4)) ^ 2 + (4 * x - 16) / (3 * x - 4) = 18) →
  x ≤ (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_l2252_225283


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l2252_225298

theorem sum_reciprocal_inequality (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (h_sum_squares : a^2 + b^2 + c^2 = 12) : 
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l2252_225298


namespace NUMINAMATH_CALUDE_min_value_abc_l2252_225225

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1 / 2) :
  ∃ (min : ℝ), min = 18 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x * y * z = 1 / 2 →
    x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l2252_225225


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l2252_225249

theorem cubic_sum_problem (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l2252_225249


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l2252_225263

theorem three_digit_multiples_of_seven (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0) → 
  (∃ k, k = (Nat.floor (999 / 7) - Nat.ceil (100 / 7) + 1) ∧ k = 128) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l2252_225263


namespace NUMINAMATH_CALUDE_manager_chef_wage_difference_l2252_225235

/-- Represents the hourly wages at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- Defines the wage relationships at Joe's Steakhouse -/
def valid_steakhouse_wages (w : SteakhouseWages) : Prop :=
  w.manager = 8.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.22

/-- Theorem stating the wage difference between manager and chef -/
theorem manager_chef_wage_difference (w : SteakhouseWages) 
  (h : valid_steakhouse_wages w) : 
  w.manager - w.chef = 3.315 := by
  sorry

end NUMINAMATH_CALUDE_manager_chef_wage_difference_l2252_225235


namespace NUMINAMATH_CALUDE_polynomial_always_positive_l2252_225273

theorem polynomial_always_positive (x y : ℝ) : x^2 + y^2 - 2*x - 4*y + 16 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_always_positive_l2252_225273


namespace NUMINAMATH_CALUDE_smallest_m_with_divisibility_l2252_225219

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_m_with_divisibility : 
  (∀ M : ℕ, M > 0 ∧ M < 250 → 
    ¬(is_divisible M (5^3) ∧ is_divisible (M+1) (2^3) ∧ is_divisible (M+2) (3^2) ∨
      is_divisible M (5^3) ∧ is_divisible (M+1) (3^2) ∧ is_divisible (M+2) (2^3) ∨
      is_divisible M (2^3) ∧ is_divisible (M+1) (5^3) ∧ is_divisible (M+2) (3^2) ∨
      is_divisible M (2^3) ∧ is_divisible (M+1) (3^2) ∧ is_divisible (M+2) (5^3) ∨
      is_divisible M (3^2) ∧ is_divisible (M+1) (5^3) ∧ is_divisible (M+2) (2^3) ∨
      is_divisible M (3^2) ∧ is_divisible (M+1) (2^3) ∧ is_divisible (M+2) (5^3))) ∧
  (is_divisible 250 (5^3) ∧ is_divisible 252 (2^3) ∧ is_divisible 252 (3^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_with_divisibility_l2252_225219


namespace NUMINAMATH_CALUDE_interval_intersection_l2252_225299

theorem interval_intersection (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 4) ∧ (2 < 5 * x ∧ 5 * x < 4) ↔ 1/2 < x ∧ x < 4/5 := by
sorry

end NUMINAMATH_CALUDE_interval_intersection_l2252_225299


namespace NUMINAMATH_CALUDE_max_sum_of_coefficients_l2252_225243

/-- Given a temperature function T(t) = a * sin(t) + b * cos(t) where a and b are positive real
    numbers, and the maximum temperature difference is 10 degrees Celsius, 
    the maximum value of a + b is 5√2. -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ t : ℝ, t > 0 → ∃ T : ℝ, T = a * Real.sin t + b * Real.cos t) →
  (∃ t₁ t₂ : ℝ, t₁ > 0 ∧ t₂ > 0 ∧ 
    (a * Real.sin t₁ + b * Real.cos t₁) - (a * Real.sin t₂ + b * Real.cos t₂) = 10) →
  a + b ≤ 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_coefficients_l2252_225243


namespace NUMINAMATH_CALUDE_grape_juice_mixture_proof_l2252_225268

/-- Proves that adding 10 gallons of grape juice to 40 gallons of a mixture
    containing 20% grape juice results in a new mixture with 36% grape juice. -/
theorem grape_juice_mixture_proof :
  let initial_mixture : ℝ := 40
  let initial_concentration : ℝ := 0.20
  let added_juice : ℝ := 10
  let final_concentration : ℝ := 0.36
  let initial_juice : ℝ := initial_mixture * initial_concentration
  let final_mixture : ℝ := initial_mixture + added_juice
  let final_juice : ℝ := initial_juice + added_juice
  final_juice / final_mixture = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_proof_l2252_225268


namespace NUMINAMATH_CALUDE_jim_apples_count_l2252_225229

theorem jim_apples_count : ∀ (j : ℕ), 
  (j + 60 + 40) / 3 = 2 * j → j = 200 := by
  sorry

end NUMINAMATH_CALUDE_jim_apples_count_l2252_225229


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2252_225285

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → 
  Complex.im ((1 - i)^2 / (1 + i)) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2252_225285


namespace NUMINAMATH_CALUDE_complex_cube_roots_sum_of_powers_l2252_225228

theorem complex_cube_roots_sum_of_powers (ω ω' : ℂ) :
  ω^3 = 1 → ω'^3 = 1 → ω = (-1 + Complex.I * Real.sqrt 3) / 2 → ω' = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^12 + ω'^12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_roots_sum_of_powers_l2252_225228


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2252_225289

theorem imaginary_part_of_complex_number (b : ℝ) :
  let z : ℂ := 2 + b * Complex.I
  (Complex.abs z = 2 * Real.sqrt 2) → (b = 2 ∨ b = -2) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2252_225289


namespace NUMINAMATH_CALUDE_select_three_from_eight_l2252_225200

theorem select_three_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_eight_l2252_225200


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2252_225253

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2252_225253


namespace NUMINAMATH_CALUDE_matrix_inverse_l2252_225212

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem matrix_inverse :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 2/46, 4/46]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_l2252_225212


namespace NUMINAMATH_CALUDE_circle_distance_inequality_l2252_225214

theorem circle_distance_inequality (r s AB : ℝ) (h1 : r > s) (h2 : AB > 0) : ¬(r - s > AB) := by
  sorry

end NUMINAMATH_CALUDE_circle_distance_inequality_l2252_225214


namespace NUMINAMATH_CALUDE_distance_ratio_l2252_225270

def travel_scenario (x y w : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ w > 0 ∧ y / w = x / w + (x + y) / (5 * w)

theorem distance_ratio (x y w : ℝ) (h : travel_scenario x y w) : x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_l2252_225270


namespace NUMINAMATH_CALUDE_base_six_units_digit_l2252_225269

theorem base_six_units_digit : 
  (123 * 78 - 156) % 6 = 0 := by sorry

end NUMINAMATH_CALUDE_base_six_units_digit_l2252_225269


namespace NUMINAMATH_CALUDE_exponential_inequality_solution_set_l2252_225257

theorem exponential_inequality_solution_set :
  {x : ℝ | (4 : ℝ)^(8 - x) > (4 : ℝ)^(-2 * x)} = {x : ℝ | x > -8} := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_solution_set_l2252_225257


namespace NUMINAMATH_CALUDE_jisoo_drank_least_l2252_225215

-- Define the amount of juice each person drank
def jennie_juice : ℚ := 9/5

-- Define Jisoo's juice amount in terms of Jennie's
def jisoo_juice : ℚ := jennie_juice - 1/5

-- Define Rohee's juice amount in terms of Jisoo's
def rohee_juice : ℚ := jisoo_juice + 3/10

-- Theorem statement
theorem jisoo_drank_least : 
  jisoo_juice < jennie_juice ∧ jisoo_juice < rohee_juice := by
  sorry


end NUMINAMATH_CALUDE_jisoo_drank_least_l2252_225215


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2252_225244

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of the coordinates of the other endpoint is 12. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (6 + x) / 2 = 3 →
  (-2 + y) / 2 = 5 →
  x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2252_225244


namespace NUMINAMATH_CALUDE_percentage_repeated_approx_l2252_225223

/-- The count of five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The count of five-digit numbers without repeated digits -/
def unique_digit_numbers : ℕ := 9 * 9 * 8 * 7 * 6

/-- The count of five-digit numbers with at least one repeated digit -/
def repeated_digit_numbers : ℕ := total_five_digit_numbers - unique_digit_numbers

/-- The percentage of five-digit numbers with at least one repeated digit -/
def percentage_repeated : ℚ := (repeated_digit_numbers : ℚ) / (total_five_digit_numbers : ℚ) * 100

theorem percentage_repeated_approx :
  ∃ ε > 0, ε < 0.1 ∧ |percentage_repeated - 69.8| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_repeated_approx_l2252_225223


namespace NUMINAMATH_CALUDE_max_children_in_candy_game_l2252_225294

/-- Represents the candy distribution game. -/
structure CandyGame where
  n : ℕ  -- number of children
  k : ℕ  -- number of complete circles each child passes candies
  a : ℕ  -- number of candies each child has when the game is interrupted

/-- Checks if the game satisfies the conditions. -/
def is_valid_game (game : CandyGame) : Prop :=
  ∃ (i j : ℕ), i < game.n ∧ j < game.n ∧ i ≠ j ∧
  (game.a + 2 * game.n * game.k - 2 * i) / (game.a + 2 * game.n * game.k - 2 * j) = 13

/-- The theorem stating the maximum number of children in the game. -/
theorem max_children_in_candy_game :
  ∃ (game : CandyGame), is_valid_game game ∧
    (∀ (other_game : CandyGame), is_valid_game other_game → other_game.n ≤ game.n) ∧
    game.n = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_children_in_candy_game_l2252_225294


namespace NUMINAMATH_CALUDE_max_product_constraint_l2252_225230

theorem max_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : 6 * a + 8 * b = 72) :
  a * b ≤ 27 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 6 * a₀ + 8 * b₀ = 72 ∧ a₀ * b₀ = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l2252_225230


namespace NUMINAMATH_CALUDE_range_of_a_l2252_225231

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → 
  a ∈ Set.Iic 1 ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2252_225231


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_vector_l2252_225240

/-- A line in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point lies on a line given by its parametric equation -/
def lies_on_line (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧ p.2 = l.point.2 + t * l.direction.2

theorem line_through_point_parallel_to_vector 
  (P : ℝ × ℝ) (d : ℝ × ℝ) :
  let l : Line2D := ⟨P, d⟩
  (∀ x y : ℝ, (x - P.1) / d.1 = (y - P.2) / d.2 ↔ lies_on_line (x, y) l) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_vector_l2252_225240


namespace NUMINAMATH_CALUDE_exam_students_count_l2252_225280

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 0 →
    T / N = 80 →
    (T - 100) / (N - 5) = 90 →
    N = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l2252_225280


namespace NUMINAMATH_CALUDE_solve_for_y_l2252_225216

theorem solve_for_y (x y z : ℝ) (h1 : x = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) : y = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2252_225216


namespace NUMINAMATH_CALUDE_negation_equivalence_l2252_225255

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2252_225255


namespace NUMINAMATH_CALUDE_opposite_sides_iff_m_range_l2252_225258

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation 3x - y + m = 0 -/
def lineEquation (p : Point) (m : ℝ) : ℝ := 3 * p.x - p.y + m

/-- Two points are on opposite sides of the line if the product of their line equations is negative -/
def oppositeSides (p1 p2 : Point) (m : ℝ) : Prop :=
  lineEquation p1 m * lineEquation p2 m < 0

/-- The theorem stating the equivalence between the points being on opposite sides and the range of m -/
theorem opposite_sides_iff_m_range (m : ℝ) :
  oppositeSides (Point.mk 1 2) (Point.mk 1 1) m ↔ -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_iff_m_range_l2252_225258


namespace NUMINAMATH_CALUDE_mixed_number_difference_l2252_225277

def digit_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_valid_mixed_number (whole : ℕ) (numer : ℕ) (denom : ℕ) : Prop :=
  whole ∈ digit_set ∧ numer ∈ digit_set ∧ denom ∈ digit_set ∧
  whole ≠ numer ∧ whole ≠ denom ∧ numer ≠ denom ∧
  numer < denom

def mixed_number_to_rational (whole : ℕ) (numer : ℕ) (denom : ℕ) : ℚ :=
  (whole : ℚ) + (numer : ℚ) / (denom : ℚ)

def largest_mixed_number : ℚ :=
  mixed_number_to_rational 5 3 4

def smallest_mixed_number : ℚ :=
  mixed_number_to_rational 1 2 5

theorem mixed_number_difference :
  largest_mixed_number - smallest_mixed_number = 87 / 20 :=
sorry

end NUMINAMATH_CALUDE_mixed_number_difference_l2252_225277


namespace NUMINAMATH_CALUDE_exam_question_count_l2252_225297

/-- Represents the scoring system and results of an examination. -/
structure ExamResult where
  correct_score : ℕ  -- Score for each correct answer
  wrong_penalty : ℕ  -- Penalty for each wrong answer
  total_score : ℤ    -- Total score achieved
  correct_count : ℕ  -- Number of correctly answered questions
  total_count : ℕ    -- Total number of questions attempted

/-- Theorem stating the relationship between exam parameters and the total number of questions attempted. -/
theorem exam_question_count 
  (exam : ExamResult) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.wrong_penalty = 1)
  (h3 : exam.total_score = 130)
  (h4 : exam.correct_count = 42) :
  exam.total_count = 80 := by
  sorry

#check exam_question_count

end NUMINAMATH_CALUDE_exam_question_count_l2252_225297


namespace NUMINAMATH_CALUDE_min_pool_cost_l2252_225207

/-- Minimum cost for constructing a rectangular open-top water pool --/
theorem min_pool_cost (volume : ℝ) (depth : ℝ) (bottom_cost : ℝ) (wall_cost : ℝ) :
  volume = 8 →
  depth = 2 →
  bottom_cost = 120 →
  wall_cost = 80 →
  ∃ (cost : ℝ), cost = 1760 ∧ 
    ∀ (length width : ℝ),
      length > 0 →
      width > 0 →
      length * width * depth = volume →
      bottom_cost * length * width + wall_cost * (2 * length + 2 * width) * depth ≥ cost :=
by sorry

end NUMINAMATH_CALUDE_min_pool_cost_l2252_225207


namespace NUMINAMATH_CALUDE_math_club_election_l2252_225209

theorem math_club_election (total_candidates : ℕ) (past_officers : ℕ) (positions : ℕ) :
  total_candidates = 20 →
  past_officers = 9 →
  positions = 6 →
  (Nat.choose total_candidates positions - Nat.choose (total_candidates - past_officers) positions) = 38298 :=
by sorry

end NUMINAMATH_CALUDE_math_club_election_l2252_225209


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l2252_225226

theorem complex_subtraction_simplification :
  (5 - 3 * Complex.I) - (-2 + 7 * Complex.I) = 7 - 10 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l2252_225226


namespace NUMINAMATH_CALUDE_f_2x_l2252_225292

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem f_2x (x : ℝ) : f (2*x) = 4*x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_l2252_225292


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2252_225217

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 5 / (1 + 2 * I)
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2252_225217


namespace NUMINAMATH_CALUDE_shark_sightings_multiple_l2252_225218

/-- The number of shark sightings in Daytona Beach -/
def daytona_sightings : ℕ := 26

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 7

/-- The additional number of sightings in Daytona Beach beyond the multiple -/
def additional_sightings : ℕ := 5

/-- The theorem stating the multiple of shark sightings in Cape May compared to Daytona Beach -/
theorem shark_sightings_multiple :
  ∃ (x : ℚ), x * cape_may_sightings + additional_sightings = daytona_sightings ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_shark_sightings_multiple_l2252_225218


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_390_l2252_225202

theorem sin_n_eq_cos_390 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (390 * π / 180) →
  n = 60 ∨ n = 120 := by
sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_390_l2252_225202


namespace NUMINAMATH_CALUDE_cherry_pits_correct_l2252_225238

/-- The number of cherry pits Kim planted -/
def cherry_pits : ℕ := 80

/-- The fraction of cherry pits that sprout -/
def sprout_rate : ℚ := 1/4

/-- The number of saplings Kim sold -/
def saplings_sold : ℕ := 6

/-- The number of saplings left after selling -/
def saplings_left : ℕ := 14

/-- Theorem stating that the number of cherry pits is correct given the conditions -/
theorem cherry_pits_correct : 
  (↑cherry_pits * sprout_rate : ℚ) - saplings_sold = saplings_left := by sorry

end NUMINAMATH_CALUDE_cherry_pits_correct_l2252_225238


namespace NUMINAMATH_CALUDE_total_tax_percentage_l2252_225204

-- Define the spending percentages
def clothing_percent : ℝ := 0.40
def food_percent : ℝ := 0.30
def other_percent : ℝ := 0.30

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0
def other_tax_rate : ℝ := 0.08

-- Theorem statement
theorem total_tax_percentage (total_spent : ℝ) (total_spent_pos : total_spent > 0) :
  let clothing_spent := clothing_percent * total_spent
  let food_spent := food_percent * total_spent
  let other_spent := other_percent * total_spent
  let clothing_tax := clothing_tax_rate * clothing_spent
  let food_tax := food_tax_rate * food_spent
  let other_tax := other_tax_rate * other_spent
  let total_tax := clothing_tax + food_tax + other_tax
  (total_tax / total_spent) * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_total_tax_percentage_l2252_225204


namespace NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l2252_225239

theorem pentagon_triangle_side_ratio :
  ∀ (p t s : ℝ),
  p > 0 ∧ t > 0 ∧ s > 0 →
  5 * p = 3 * t →
  5 * p = 4 * s →
  p / t = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l2252_225239


namespace NUMINAMATH_CALUDE_binomial_coefficient_formula_l2252_225265

theorem binomial_coefficient_formula (n k : ℕ) (h : k ≤ n) : 
  Nat.choose n k = n.factorial / ((n - k).factorial * k.factorial) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_formula_l2252_225265


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l2252_225279

/-- Proves that reducing speed to 60 km/h increases travel time by a factor of 1.5 --/
theorem car_speed_time_relation (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ original_time = 6 ∧ new_speed = 60 →
  (distance / new_speed) / original_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_time_relation_l2252_225279


namespace NUMINAMATH_CALUDE_minibus_capacity_insufficient_l2252_225275

theorem minibus_capacity_insufficient (students : ℕ) (bus_capacity : ℕ) (num_buses : ℕ) : 
  students = 300 → 
  bus_capacity = 23 → 
  num_buses = 13 → 
  num_buses * bus_capacity < students := by
sorry

end NUMINAMATH_CALUDE_minibus_capacity_insufficient_l2252_225275


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2252_225211

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxes (carton : BoxDimensions) (soapBox : BoxDimensions) : ℕ :=
  (carton.length / soapBox.length) * (carton.width / soapBox.height) * (carton.height / soapBox.width)

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  let carton : BoxDimensions := ⟨48, 25, 60⟩
  let soapBox : BoxDimensions := ⟨8, 6, 5⟩
  maxBoxes carton soapBox = 300 := by
  sorry

#eval maxBoxes ⟨48, 25, 60⟩ ⟨8, 6, 5⟩

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2252_225211


namespace NUMINAMATH_CALUDE_transylvania_statements_l2252_225234

/-- Represents a statement that can be made by a resident of Transylvania -/
structure Statement :=
  (proposition : Prop)

/-- Defines what it means for one statement to be the converse of another -/
def is_converse (X Y : Statement) : Prop :=
  ∃ P Q : Prop, X.proposition = (P → Q) ∧ Y.proposition = (Q → P)

/-- Defines the property that asserting one statement implies the truth of another -/
def implies_truth (X Y : Statement) : Prop :=
  ∀ (resident : Prop), (resident → X.proposition) → Y.proposition

/-- The main theorem stating the existence of two statements satisfying the given conditions -/
theorem transylvania_statements : ∃ (X Y : Statement),
  is_converse X Y ∧
  (¬ (X.proposition → Y.proposition)) ∧
  (¬ (Y.proposition → X.proposition)) ∧
  implies_truth X Y ∧
  implies_truth Y X := by
  sorry

end NUMINAMATH_CALUDE_transylvania_statements_l2252_225234


namespace NUMINAMATH_CALUDE_class_average_calculation_l2252_225241

theorem class_average_calculation (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 20 → 
  excluded_students = 5 → 
  excluded_avg = 50 → 
  remaining_avg = 90 → 
  (total_students * (total_students * remaining_avg - excluded_students * remaining_avg + 
   excluded_students * excluded_avg)) / (total_students * total_students) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_calculation_l2252_225241


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l2252_225206

/-- The largest prime number with 2009 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2009 digits -/
axiom q_digits : 10^2008 ≤ q ∧ q < 10^2009

/-- q is the largest prime with 2009 digits -/
axiom q_largest : ∀ p, Nat.Prime p → 10^2008 ≤ p ∧ p < 10^2009 → p ≤ q

/-- The theorem to be proved -/
theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l2252_225206


namespace NUMINAMATH_CALUDE_diamond_properties_l2252_225236

def diamond (a b : ℤ) : ℤ := a^2 - 2*b

theorem diamond_properties :
  (diamond (-1) 2 = -3) ∧
  (∃ a b : ℤ, diamond a b ≠ diamond b a) := by sorry

end NUMINAMATH_CALUDE_diamond_properties_l2252_225236


namespace NUMINAMATH_CALUDE_function_and_monotonicity_l2252_225237

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_and_monotonicity 
  (a b : ℝ) 
  (h1 : f a b 1 = -3)  -- f(1) = -3
  (h2 : f' a b 1 = 0)  -- f'(1) = 0
  : 
  (∀ x, f a b x = 2 * x^3 - 6 * x + 1) ∧  -- Explicit formula
  (∀ x, x < -1 → (f' a b x > 0)) ∧       -- Monotonically increasing for x < -1
  (∀ x, x > 1 → (f' a b x > 0))          -- Monotonically increasing for x > 1
  := by sorry

end NUMINAMATH_CALUDE_function_and_monotonicity_l2252_225237


namespace NUMINAMATH_CALUDE_total_arrangements_l2252_225252

-- Define the number of people
def total_people : ℕ := 5

-- Define the number of positions for person A
def positions_for_A : ℕ := 2

-- Define the number of positions for person B
def positions_for_B : ℕ := 3

-- Define the number of remaining people
def remaining_people : ℕ := total_people - 2

-- Theorem statement
theorem total_arrangements :
  (positions_for_A * positions_for_B * (Nat.factorial remaining_people)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_l2252_225252


namespace NUMINAMATH_CALUDE_square_area_relation_l2252_225220

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := 2*a + 2*b
  let area_I := (diagonal_I / Real.sqrt 2)^2
  let area_II := 3 * area_I
  area_II = 6 * (a + b)^2 := by
sorry

end NUMINAMATH_CALUDE_square_area_relation_l2252_225220


namespace NUMINAMATH_CALUDE_sqrt_three_bounds_l2252_225254

theorem sqrt_three_bounds (n : ℕ+) : 
  (1 + 3 / (n + 1 : ℝ) < Real.sqrt 3) ∧ (Real.sqrt 3 < 1 + 3 / (n : ℝ)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_bounds_l2252_225254


namespace NUMINAMATH_CALUDE_liquid_volume_in_tin_l2252_225290

/-- The volume of liquid in a cylindrical tin with a conical cavity -/
theorem liquid_volume_in_tin (tin_diameter tin_height : ℝ) 
  (liquid_fill_ratio : ℝ) (cavity_height cavity_diameter : ℝ) : 
  tin_diameter = 10 →
  tin_height = 5 →
  liquid_fill_ratio = 2/3 →
  cavity_height = 2 →
  cavity_diameter = 4 →
  (liquid_fill_ratio * tin_height * π * (tin_diameter/2)^2 - 
   (1/3) * π * (cavity_diameter/2)^2 * cavity_height) = (242/3) * π := by
  sorry

#check liquid_volume_in_tin

end NUMINAMATH_CALUDE_liquid_volume_in_tin_l2252_225290


namespace NUMINAMATH_CALUDE_circle_op_inequality_solution_set_l2252_225227

-- Define the ⊙ operation
def circle_op (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem circle_op_inequality_solution_set :
  ∀ x : ℝ, circle_op x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_circle_op_inequality_solution_set_l2252_225227


namespace NUMINAMATH_CALUDE_alpha_minus_beta_eq_pi_fourth_l2252_225262

theorem alpha_minus_beta_eq_pi_fourth 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (π/4) (π/2)) 
  (h2 : β ∈ Set.Icc (π/4) (π/2)) 
  (h3 : Real.sin α + Real.cos α = Real.sqrt 2 * Real.cos β) : 
  α - β = π/4 := by
sorry

end NUMINAMATH_CALUDE_alpha_minus_beta_eq_pi_fourth_l2252_225262


namespace NUMINAMATH_CALUDE_sum_abcd_l2252_225272

theorem sum_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) : 
  a + b + c + d = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_abcd_l2252_225272


namespace NUMINAMATH_CALUDE_monomial_combination_l2252_225245

/-- 
Given two monomials that can be combined, this theorem proves 
the values of their exponents.
-/
theorem monomial_combination (m n : ℕ) : 
  (∃ (a b : ℝ), 3 * a^(m+1) * b = -b^(n-1) * a^3) → 
  (m = 2 ∧ n = 2) := by
  sorry

#check monomial_combination

end NUMINAMATH_CALUDE_monomial_combination_l2252_225245


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2252_225286

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) ≥ 0 ∧
  ((a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) = 0 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2252_225286


namespace NUMINAMATH_CALUDE_triangle_properties_l2252_225251

/-- Given a triangle ABC where 2b cos B = a cos C + c cos A, prove the measure of angle B and the range of sin A + sin C -/
theorem triangle_properties (a b c A B C : ℝ) 
  (h_triangle : 2 * b * Real.cos B = a * Real.cos C + c * Real.cos A)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_triangle_sum : A + B + C = π) : 
  B = π / 3 ∧ Set.Icc (Real.sqrt 3 / 2) (Real.sqrt 3) (Real.sin A + Real.sin C) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2252_225251


namespace NUMINAMATH_CALUDE_visibility_condition_l2252_225264

/-- The curve C: y = 2x^2 -/
def C (x : ℝ) : ℝ := 2 * x^2

/-- Point A -/
def A : ℝ × ℝ := (0, -2)

/-- Point B -/
def B (a : ℝ) : ℝ × ℝ := (3, a)

/-- A point (x, y) is above the curve C -/
def is_above_curve (x y : ℝ) : Prop := y > C x

/-- A point (x, y) is on or below the line passing through two points -/
def is_on_or_below_line (x1 y1 x2 y2 x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) ≤ (y2 - y1) * (x - x1)

/-- B is visible from A without being obstructed by C -/
def is_visible (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 →
    is_above_curve x ((a + 2) / 3 * x - 2)

theorem visibility_condition (a : ℝ) :
  is_visible a ↔ a < 10 := by sorry

end NUMINAMATH_CALUDE_visibility_condition_l2252_225264


namespace NUMINAMATH_CALUDE_income_remainder_relation_l2252_225201

/-- Represents a person's income distribution --/
structure IncomeDistribution where
  total : ℝ
  children : ℝ
  wife : ℝ
  bills : ℝ
  savings : ℝ
  remainder : ℝ

/-- Theorem stating the relationship between income and remainder --/
theorem income_remainder_relation (d : IncomeDistribution) :
  d.children = 0.18 * d.total ∧
  d.wife = 0.28 * d.total ∧
  d.bills = 0.12 * d.total ∧
  d.savings = 0.15 * d.total ∧
  d.remainder = 35000 →
  0.27 * d.total = 35000 := by
  sorry

end NUMINAMATH_CALUDE_income_remainder_relation_l2252_225201


namespace NUMINAMATH_CALUDE_babysitting_earnings_l2252_225233

/-- Calculates the earnings for a given hourly rate and number of minutes worked. -/
def calculate_earnings (hourly_rate : ℚ) (minutes_worked : ℚ) : ℚ :=
  hourly_rate * minutes_worked / 60

/-- Proves that given an hourly rate of $12 and 50 minutes of work, the earnings are equal to $10. -/
theorem babysitting_earnings :
  calculate_earnings 12 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l2252_225233
