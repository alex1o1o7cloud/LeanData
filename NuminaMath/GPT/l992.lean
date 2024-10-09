import Mathlib

namespace flood_monitoring_technology_l992_99287

def geographicInformationTechnologies : Type := String

def RemoteSensing : geographicInformationTechnologies := "Remote Sensing"
def GlobalPositioningSystem : geographicInformationTechnologies := "Global Positioning System"
def GeographicInformationSystem : geographicInformationTechnologies := "Geographic Information System"
def DigitalEarth : geographicInformationTechnologies := "Digital Earth"

def effectiveFloodMonitoring (tech1 tech2 : geographicInformationTechnologies) : Prop :=
  (tech1 = RemoteSensing ∧ tech2 = GeographicInformationSystem) ∨ 
  (tech1 = GeographicInformationSystem ∧ tech2 = RemoteSensing)

theorem flood_monitoring_technology :
  effectiveFloodMonitoring RemoteSensing GeographicInformationSystem :=
by
  sorry

end flood_monitoring_technology_l992_99287


namespace trigonometric_identity_l992_99293

theorem trigonometric_identity (m : ℝ) (h : m < 0) :
  2 * (3 / -5) + 4 / -5 = -2 / 5 :=
by
  sorry

end trigonometric_identity_l992_99293


namespace square_pyramid_properties_l992_99276

-- Definitions for the square pyramid with a square base
def square_pyramid_faces : Nat := 4 + 1
def square_pyramid_edges : Nat := 4 + 4
def square_pyramid_vertices : Nat := 4 + 1

-- Definition for the number of diagonals in a square
def diagonals_in_square_base (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem statement
theorem square_pyramid_properties :
  (square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18) ∧ (diagonals_in_square_base 4 = 2) :=
by
  sorry

end square_pyramid_properties_l992_99276


namespace maximum_sum_each_side_equals_22_l992_99291

theorem maximum_sum_each_side_equals_22 (A B C D : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ 10)
  → (∀ S, S = A ∨ S = B ∨ S = C ∨ S = D ∧ A + B + C + D = 33)
  → (A + B + C + D + 55) / 4 = 22 :=
by
  sorry

end maximum_sum_each_side_equals_22_l992_99291


namespace speed_of_train_l992_99215

-- Define the conditions
def length_of_train : ℕ := 240
def length_of_bridge : ℕ := 150
def time_to_cross : ℕ := 20

-- Compute the expected speed of the train
def expected_speed : ℝ := 19.5

-- The statement that needs to be proven
theorem speed_of_train : (length_of_train + length_of_bridge) / time_to_cross = expected_speed := by
  -- sorry is used to skip the actual proof
  sorry

end speed_of_train_l992_99215


namespace average_production_per_day_for_entire_month_l992_99235

-- Definitions based on the conditions
def average_first_25_days := 65
def average_last_5_days := 35
def number_of_days_in_first_period := 25
def number_of_days_in_last_period := 5
def total_days_in_month := 30

-- The goal is to prove that the average production per day for the entire month is 60 TVs/day.
theorem average_production_per_day_for_entire_month :
  (average_first_25_days * number_of_days_in_first_period + 
   average_last_5_days * number_of_days_in_last_period) / total_days_in_month = 60 := 
by
  sorry

end average_production_per_day_for_entire_month_l992_99235


namespace second_game_score_count_l992_99253

-- Define the conditions and problem
def total_points (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  A1 + A2 + A3 + B1 + B2 + B3 = 31

def valid_game_1 (A1 B1 : ℕ) : Prop :=
  A1 ≥ 11 ∧ A1 - B1 ≥ 2

def valid_game_2 (A2 B2 : ℕ) : Prop :=
  B2 ≥ 11 ∧ B2 - A2 ≥ 2

def valid_game_3 (A3 B3 : ℕ) : Prop :=
  A3 ≥ 11 ∧ A3 - B3 ≥ 2

def game_sequence (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  valid_game_1 A1 B1 ∧ valid_game_2 A2 B2 ∧ valid_game_3 A3 B3

noncomputable def second_game_score_possibilities : ℕ := 
  8 -- This is derived from calculating the valid scores where B wins the second game.

theorem second_game_score_count (A1 A2 A3 B1 B2 B3 : ℕ) (h_total : total_points A1 A2 A3 B1 B2 B3) (h_sequence : game_sequence A1 A2 A3 B1 B2 B3) :
  second_game_score_possibilities = 8 := sorry

end second_game_score_count_l992_99253


namespace complex_number_solution_l992_99290

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i^2 = -1) 
  (h : -i * z = (3 + 2 * i) * (1 - i)) : z = 1 + 5 * i :=
by
  sorry

end complex_number_solution_l992_99290


namespace contrapositive_example_l992_99221

theorem contrapositive_example (a b m : ℝ) :
  (a > b → a * (m^2 + 1) > b * (m^2 + 1)) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end contrapositive_example_l992_99221


namespace cakes_sold_l992_99234

/-- If a baker made 54 cakes and has 13 cakes left, then the number of cakes he sold is 41. -/
theorem cakes_sold (original_cakes : ℕ) (cakes_left : ℕ) 
  (h1 : original_cakes = 54) (h2 : cakes_left = 13) : 
  original_cakes - cakes_left = 41 := 
by 
  sorry

end cakes_sold_l992_99234


namespace smallest_solution_l992_99255

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end smallest_solution_l992_99255


namespace trisha_bought_amount_initially_l992_99233

-- Define the amounts spent on each item
def meat : ℕ := 17
def chicken : ℕ := 22
def veggies : ℕ := 43
def eggs : ℕ := 5
def dogs_food : ℕ := 45
def amount_left : ℕ := 35

-- Define the total amount spent
def total_spent : ℕ := meat + chicken + veggies + eggs + dogs_food

-- Define the amount brought at the beginning
def amount_brought_at_beginning : ℕ := total_spent + amount_left

-- Theorem stating the amount Trisha brought at the beginning is 167
theorem trisha_bought_amount_initially : amount_brought_at_beginning = 167 := by
  -- Formal proof would go here, we use sorry to skip the proof
  sorry

end trisha_bought_amount_initially_l992_99233


namespace pyramid_height_l992_99263

def height_of_pyramid (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem pyramid_height (n : ℕ) : height_of_pyramid n = 2 * (n - 1) :=
by
  -- The proof would typically go here
  sorry

end pyramid_height_l992_99263


namespace bruno_coconuts_per_trip_is_8_l992_99275

-- Definitions related to the problem conditions
def total_coconuts : ℕ := 144
def barbie_coconuts_per_trip : ℕ := 4
def trips : ℕ := 12
def bruno_coconuts_per_trip : ℕ := total_coconuts - (barbie_coconuts_per_trip * trips)

-- The main theorem stating the question and the answer
theorem bruno_coconuts_per_trip_is_8 : bruno_coconuts_per_trip / trips = 8 :=
by
  sorry

end bruno_coconuts_per_trip_is_8_l992_99275


namespace rita_swimming_months_l992_99289

theorem rita_swimming_months
    (total_required_hours : ℕ := 1500)
    (backstroke_hours : ℕ := 50)
    (breaststroke_hours : ℕ := 9)
    (butterfly_hours : ℕ := 121)
    (monthly_hours : ℕ := 220) :
    (total_required_hours - (backstroke_hours + breaststroke_hours + butterfly_hours)) / monthly_hours = 6 := 
by
    -- Proof is omitted
    sorry

end rita_swimming_months_l992_99289


namespace book_pages_l992_99260

theorem book_pages (x : ℕ) : 
  (x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15) - (1/3 * ((x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15)) + 18) = 62 →
  x = 240 :=
by
  -- This is where the proof would go, but it's omitted for this task.
  sorry

end book_pages_l992_99260


namespace pyramid_surface_area_l992_99261

-- Definitions based on conditions
def upper_base_edge_length : ℝ := 2
def lower_base_edge_length : ℝ := 4
def side_edge_length : ℝ := 2

-- Problem statement in Lean
theorem pyramid_surface_area :
  let slant_height := Real.sqrt ((side_edge_length ^ 2) - (1 ^ 2))
  let perimeter_base := (4 * upper_base_edge_length) + (4 * lower_base_edge_length)
  let lsa := (perimeter_base * slant_height) / 2
  let total_surface_area := lsa + (upper_base_edge_length ^ 2) + (lower_base_edge_length ^ 2)
  total_surface_area = 10 * Real.sqrt 3 + 20 := sorry

end pyramid_surface_area_l992_99261


namespace triangle_perimeter_l992_99245

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : a = 3) (h2 : b = 3) 
    (h3 : c^2 = a * Real.cos B + b * Real.cos A) : 
    a + b + c = 7 :=
by 
  sorry

end triangle_perimeter_l992_99245


namespace middle_number_is_10_l992_99207

theorem middle_number_is_10 (A B C : ℝ) (h1 : B - C = A - B) (h2 : A * B = 85) (h3 : B * C = 115) : B = 10 :=
by
  sorry

end middle_number_is_10_l992_99207


namespace exists_five_digit_number_with_property_l992_99295

theorem exists_five_digit_number_with_property :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (n^2 % 100000) = n := 
sorry

end exists_five_digit_number_with_property_l992_99295


namespace expression_value_l992_99256

theorem expression_value : (36 + 9) ^ 2 - (9 ^ 2 + 36 ^ 2) = -1894224 :=
by
  sorry

end expression_value_l992_99256


namespace henry_friend_fireworks_l992_99240

-- Definitions of variables and conditions
variable 
  (F : ℕ) -- Number of fireworks Henry's friend bought

-- Main theorem statement
theorem henry_friend_fireworks (h1 : 6 + 2 + F = 11) : F = 3 :=
by
  sorry

end henry_friend_fireworks_l992_99240


namespace trucks_on_lot_l992_99224

-- We'll state the conditions as hypotheses and then conclude the total number of trucks.
theorem trucks_on_lot (T : ℕ)
  (h₁ : ∀ N : ℕ, 50 ≤ N ∧ N ≤ 20 → N / 2 = 10)
  (h₂ : T ≥ 20 + 10): T = 30 :=
sorry

end trucks_on_lot_l992_99224


namespace jerry_age_is_13_l992_99219

variable (M J : ℕ)

theorem jerry_age_is_13 (h1 : M = 2 * J - 6) (h2 : M = 20) : J = 13 := by
  sorry

end jerry_age_is_13_l992_99219


namespace find_m_l992_99273

-- Define the arithmetic sequence and its properties
variable {α : Type*} [OrderedRing α]
variable (a : Nat → α) (S : Nat → α) (m : ℕ)

-- The conditions from the problem
variable (is_arithmetic_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
variable (sum_of_terms : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)
variable (m_gt_one : m > 1)
variable (condition1 : a (m - 1) + a (m + 1) - a m ^ 2 - 1 = 0)
variable (condition2 : S (2 * m - 1) = 39)

-- Prove that m = 20
theorem find_m : m = 20 :=
sorry

end find_m_l992_99273


namespace base4_sum_conversion_to_base10_l992_99278

theorem base4_sum_conversion_to_base10 :
  let n1 := 2213
  let n2 := 2703
  let n3 := 1531
  let base := 4
  let sum_base4 := n1 + n2 + n3 
  let sum_base10 :=
    (1 * base^4) + (0 * base^3) + (2 * base^2) + (5 * base^1) + (1 * base^0)
  sum_base10 = 309 :=
by
  sorry

end base4_sum_conversion_to_base10_l992_99278


namespace total_pounds_of_peppers_l992_99257

-- Definitions based on the conditions
def greenPeppers : ℝ := 0.3333333333333333
def redPeppers : ℝ := 0.3333333333333333

-- Goal statement expressing the problem
theorem total_pounds_of_peppers :
  greenPeppers + redPeppers = 0.6666666666666666 := 
by
  sorry

end total_pounds_of_peppers_l992_99257


namespace conic_sections_of_equation_l992_99201

theorem conic_sections_of_equation :
  ∀ x y : ℝ, y^4 - 9*x^6 = 3*y^2 - 1 →
  (∃ y, y^2 - 3*x^3 = 4 ∨ y^2 + 3*x^3 = 0) :=
by 
  sorry

end conic_sections_of_equation_l992_99201


namespace total_sample_any_candy_42_percent_l992_99205

-- Define percentages as rational numbers to avoid dealing with decimals directly
def percent_of_caught_A : ℚ := 12 / 100
def percent_of_not_caught_A : ℚ := 7 / 100
def percent_of_caught_B : ℚ := 5 / 100
def percent_of_not_caught_B : ℚ := 6 / 100
def percent_of_caught_C : ℚ := 9 / 100
def percent_of_not_caught_C : ℚ := 3 / 100

-- Sum up the percentages for those caught and not caught for each type of candy
def total_percent_A : ℚ := percent_of_caught_A + percent_of_not_caught_A
def total_percent_B : ℚ := percent_of_caught_B + percent_of_not_caught_B
def total_percent_C : ℚ := percent_of_caught_C + percent_of_not_caught_C

-- Sum of the total percentages for all types
def total_percent_sample_any_candy : ℚ := total_percent_A + total_percent_B + total_percent_C

theorem total_sample_any_candy_42_percent :
  total_percent_sample_any_candy = 42 / 100 :=
by
  sorry

end total_sample_any_candy_42_percent_l992_99205


namespace min_value_y_minus_one_over_x_l992_99296

variable {x y : ℝ}

-- Condition 1: x is the median of the dataset
def is_median (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 5

-- Condition 2: The average of the dataset is 1
def average_is_one (x y : ℝ) : Prop := 1 + 2 + x^2 - y = 4

-- The statement to be proved
theorem min_value_y_minus_one_over_x :
  ∀ (x y : ℝ), is_median x → average_is_one x y → y = x^2 - 1 → (y - 1/x) ≥ 23/3 :=
by 
  -- This is a placeholder for the actual proof
  sorry

end min_value_y_minus_one_over_x_l992_99296


namespace probability_of_selecting_female_l992_99218

theorem probability_of_selecting_female (total_students female_students male_students : ℕ)
  (h_total : total_students = female_students + male_students)
  (h_female : female_students = 3)
  (h_male : male_students = 1) :
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end probability_of_selecting_female_l992_99218


namespace calc_value_l992_99248

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem calc_value :
  ((diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2))) = -13 / 28 :=
by sorry

end calc_value_l992_99248


namespace FC_value_l992_99209

theorem FC_value (DC CB AD FC : ℝ) (h1 : DC = 10) (h2 : CB = 9)
  (h3 : AB = (1 / 3) * AD) (h4 : ED = (3 / 4) * AD) : FC = 13.875 := by
  sorry

end FC_value_l992_99209


namespace product_of_six_numbers_l992_99232

theorem product_of_six_numbers (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x^3 * y^2 = 108) : 
  x * y * (x * y) * (x^2 * y) * (x^3 * y^2) * (x^5 * y^3) = 136048896 := 
by
  sorry

end product_of_six_numbers_l992_99232


namespace quadratic_inequality_range_of_k_l992_99237

theorem quadratic_inequality (a b x : ℝ) (h1 : a = 1) (h2 : b > 1) :
  (a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
sorry

theorem range_of_k (x y k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1/x) + (2/y) = 1) (h4 : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 :=
sorry

end quadratic_inequality_range_of_k_l992_99237


namespace consecutive_arithmetic_sequence_l992_99242

theorem consecutive_arithmetic_sequence (a b c : ℝ) 
  (h : (2 * b - a)^2 + (2 * b - c)^2 = 2 * (2 * b^2 - a * c)) : 
  2 * b = a + c :=
by
  sorry

end consecutive_arithmetic_sequence_l992_99242


namespace area_of_rectangular_garden_l992_99203

-- Definitions based on conditions
def width : ℕ := 15
def length : ℕ := 3 * width
def area : ℕ := length * width

-- The theorem we want to prove
theorem area_of_rectangular_garden : area = 675 :=
by sorry

end area_of_rectangular_garden_l992_99203


namespace kevin_watermelons_l992_99222

theorem kevin_watermelons (w1 w2 w_total : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) (h_total : w_total = 14.02) : 
  w1 + w2 = w_total → 2 = 2 :=
by
  sorry

end kevin_watermelons_l992_99222


namespace necessary_but_not_sufficient_l992_99211

theorem necessary_but_not_sufficient (a : ℝ) : (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) := sorry

end necessary_but_not_sufficient_l992_99211


namespace inequality_3var_l992_99292

theorem inequality_3var (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x * y + y * z + z * x = 1) : 
    1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 :=
sorry

end inequality_3var_l992_99292


namespace roots_equation_1352_l992_99258

theorem roots_equation_1352 {c d : ℝ} (hc : c^2 - 6 * c + 8 = 0) (hd : d^2 - 6 * d + 8 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 1352 :=
by
  sorry

end roots_equation_1352_l992_99258


namespace percentage_relation_l992_99283

theorem percentage_relation (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end percentage_relation_l992_99283


namespace Robert_salary_loss_l992_99262

-- Define the conditions as hypotheses
variable (S : ℝ) (decrease_percent increase_percent : ℝ)
variable (decrease_percent_eq : decrease_percent = 0.6)
variable (increase_percent_eq : increase_percent = 0.6)

-- Define the problem statement to prove that Robert loses 36% of his salary.
theorem Robert_salary_loss (S : ℝ) (decrease_percent increase_percent : ℝ) 
  (decrease_percent_eq : decrease_percent = 0.6) 
  (increase_percent_eq : increase_percent = 0.6) :
  let new_salary := S * (1 - decrease_percent)
  let increased_salary := new_salary * (1 + increase_percent)
  let loss_percentage := (S - increased_salary) / S * 100 
  loss_percentage = 36 := 
by
  sorry

end Robert_salary_loss_l992_99262


namespace total_steps_l992_99227

def steps_on_feet (jason_steps : Nat) (nancy_ratio : Nat) : Nat :=
  jason_steps + (nancy_ratio * jason_steps)

theorem total_steps (jason_steps : Nat) (nancy_ratio : Nat) (h1 : jason_steps = 8) (h2 : nancy_ratio = 3) :
  steps_on_feet jason_steps nancy_ratio = 32 :=
by
  sorry

end total_steps_l992_99227


namespace age_of_son_l992_99281

theorem age_of_son (S M : ℕ) (h1 : M = S + 28) (h2 : M + 2 = 2 * (S + 2)) : S = 26 := by
  sorry

end age_of_son_l992_99281


namespace liam_total_time_l992_99252

noncomputable def total_time_7_laps : Nat :=
let time_first_200 := 200 / 5  -- Time in seconds for the first 200 meters
let time_next_300 := 300 / 6   -- Time in seconds for the next 300 meters
let time_per_lap := time_first_200 + time_next_300
let laps := 7
let total_time := laps * time_per_lap
total_time

theorem liam_total_time : total_time_7_laps = 630 := by
sorry

end liam_total_time_l992_99252


namespace isosceles_triangle_y_value_l992_99284

theorem isosceles_triangle_y_value :
  ∃ y : ℝ, (y = 1 + Real.sqrt 51 ∨ y = 1 - Real.sqrt 51) ∧ 
  (Real.sqrt ((y - 1)^2 + (4 - (-3))^2) = 10) :=
by sorry

end isosceles_triangle_y_value_l992_99284


namespace initial_amount_100000_l992_99271

noncomputable def compound_interest_amount (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value (P CI : ℝ) : ℝ :=
  P + CI

theorem initial_amount_100000
  (CI : ℝ) (P : ℝ) (r : ℝ) (n t : ℕ) 
  (h1 : CI = 8243.216)
  (h2 : r = 0.04)
  (h3 : n = 2)
  (h4 : t = 2)
  (h5 : future_value P CI = compound_interest_amount P r n t) :
  P = 100000 :=
by
  sorry

end initial_amount_100000_l992_99271


namespace mass_of_man_l992_99202

def boat_length : ℝ := 3 -- boat length in meters
def boat_breadth : ℝ := 2 -- boat breadth in meters
def boat_sink_depth : ℝ := 0.01 -- boat sink depth in meters
def water_density : ℝ := 1000 -- density of water in kg/m^3

/- Theorem: The mass of the man is equal to 60 kg given the parameters defined above. -/
theorem mass_of_man : (water_density * (boat_length * boat_breadth * boat_sink_depth)) = 60 :=
by
  simp [boat_length, boat_breadth, boat_sink_depth, water_density]
  sorry

end mass_of_man_l992_99202


namespace value_of_a_2015_l992_99247

def a : ℕ → Int
| 0 => 1
| 1 => 5
| n+2 => a (n+1) - a n

theorem value_of_a_2015 : a 2014 = -5 := by
  sorry

end value_of_a_2015_l992_99247


namespace andy_older_than_rahim_l992_99239

-- Define Rahim's current age
def Rahim_current_age : ℕ := 6

-- Define Andy's age in 5 years
def Andy_age_in_5_years : ℕ := 2 * Rahim_current_age

-- Define Andy's current age
def Andy_current_age : ℕ := Andy_age_in_5_years - 5

-- Define the difference in age between Andy and Rahim right now
def age_difference : ℕ := Andy_current_age - Rahim_current_age

-- Theorem stating the age difference between Andy and Rahim right now is 1 year
theorem andy_older_than_rahim : age_difference = 1 :=
by
  -- Proof is skipped
  sorry

end andy_older_than_rahim_l992_99239


namespace find_value_l992_99223

theorem find_value
  (x a y b z c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 :=
by 
  sorry

end find_value_l992_99223


namespace arccos_one_over_sqrt_two_l992_99286

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l992_99286


namespace proof_of_problem_l992_99251

noncomputable def problem_statement (a b c x y z : ℝ) : Prop :=
  23 * x + b * y + c * z = 0 ∧
  a * x + 33 * y + c * z = 0 ∧
  a * x + b * y + 52 * z = 0 ∧
  a ≠ 23 ∧
  x ≠ 0 →
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1

theorem proof_of_problem (a b c x y z : ℝ) (h : problem_statement a b c x y z) : 
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 :=
sorry

end proof_of_problem_l992_99251


namespace acute_angle_is_three_pi_over_eight_l992_99269

noncomputable def acute_angle_concentric_circles : Real :=
  let r₁ := 4
  let r₂ := 3
  let r₃ := 2
  let total_area := (r₁ * r₁ * Real.pi) + (r₂ * r₂ * Real.pi) + (r₃ * r₃ * Real.pi)
  let unshaded_area := 5 * (total_area / 8)
  let shaded_area := (3 / 5) * unshaded_area
  let theta := shaded_area / total_area * 2 * Real.pi
  theta

theorem acute_angle_is_three_pi_over_eight :
  acute_angle_concentric_circles = (3 * Real.pi / 8) :=
by
  sorry

end acute_angle_is_three_pi_over_eight_l992_99269


namespace first_laptop_cost_l992_99225

variable (x : ℝ)

def cost_first_laptop (x : ℝ) : ℝ := x
def cost_second_laptop (x : ℝ) : ℝ := 3 * x
def total_cost (x : ℝ) : ℝ := cost_first_laptop x + cost_second_laptop x
def budget : ℝ := 2000

theorem first_laptop_cost : total_cost x = budget → x = 500 :=
by
  intros h
  sorry

end first_laptop_cost_l992_99225


namespace games_played_l992_99226

def total_points : ℝ := 120.0
def points_per_game : ℝ := 12.0
def num_games : ℝ := 10.0

theorem games_played : (total_points / points_per_game) = num_games := 
by 
  sorry

end games_played_l992_99226


namespace number_of_good_games_l992_99206

def total_games : ℕ := 11
def bad_games : ℕ := 5
def good_games : ℕ := total_games - bad_games

theorem number_of_good_games : good_games = 6 := by
  sorry

end number_of_good_games_l992_99206


namespace range_of_n_l992_99297

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n : ℝ) : Set ℝ := {x | n-1 < x ∧ x < n+1}

-- Define the condition A ∩ B ≠ ∅
def A_inter_B_nonempty (n : ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B n

-- Prove the range of n for which A ∩ B ≠ ∅ is (-2, 2)
theorem range_of_n : ∀ n, A_inter_B_nonempty n ↔ (-2 < n ∧ n < 2) := by
  sorry

end range_of_n_l992_99297


namespace man_l992_99210

theorem man's_age_twice_son_in_2_years 
  (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = 38) (h3 : M = S + 20) : 
  ∃ X : ℕ, (M + X = 2 * (S + X)) ∧ X = 2 :=
by
  sorry

end man_l992_99210


namespace find_y_l992_99288

theorem find_y : 
  let mean1 := (7 + 9 + 14 + 23) / 4
  let mean2 := (18 + y) / 2
  mean1 = mean2 → y = 8.5 :=
by
  let y := 8.5
  sorry

end find_y_l992_99288


namespace square_side_length_l992_99230

theorem square_side_length (A : ℝ) (h : A = 625) : ∃ l : ℝ, l^2 = A ∧ l = 25 :=
by {
  sorry
}

end square_side_length_l992_99230


namespace check_error_difference_l992_99267

-- Let us define x and y as two-digit natural numbers
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem check_error_difference
    (x y : ℕ)
    (hx : isTwoDigit x)
    (hy : isTwoDigit y)
    (hxy : x > y)
    (h_difference : (100 * y + x) - (100 * x + y) = 2187)
    : x - y = 22 :=
by
  sorry

end check_error_difference_l992_99267


namespace trig_identity_l992_99216

theorem trig_identity : 
  Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l992_99216


namespace range_of_x_l992_99298

open Real

def p (x : ℝ) : Prop := log (x^2 - 2 * x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4
def not_p (x : ℝ) : Prop := -1 < x ∧ x < 3
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 4

theorem range_of_x (x : ℝ) :
  (¬ p x ∧ ¬ q x ∧ (p x ∨ q x)) →
  x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4 :=
sorry

end range_of_x_l992_99298


namespace relationship_between_a_and_b_l992_99285

variable {a b : ℝ} (n : ℕ)

theorem relationship_between_a_and_b (h₁ : a^n = a + 1) (h₂ : b^(2 * n) = b + 3 * a)
  (h₃ : 2 ≤ n) (h₄ : 1 < a) (h₅ : 1 < b) : a > b ∧ b > 1 :=
by
  sorry

end relationship_between_a_and_b_l992_99285


namespace repaired_shoes_last_time_l992_99214

theorem repaired_shoes_last_time :
  let cost_of_repair := 13.50
  let cost_of_new := 32.00
  let duration_of_new := 2.0
  let surcharge := 0.1852
  let avg_cost_new := cost_of_new / duration_of_new
  let avg_cost_repair (T : ℝ) := cost_of_repair / T
  (avg_cost_new = (1 + surcharge) * avg_cost_repair 1) ↔ T = 1 := 
by
  sorry

end repaired_shoes_last_time_l992_99214


namespace digit_150_in_fraction_l992_99231

-- Define the decimal expansion repeating sequence for the fraction 31/198
def repeat_seq : List Nat := [1, 5, 6, 5, 6, 5]

-- Define a function to get the nth digit of the repeating sequence
def nth_digit (n : Nat) : Nat :=
  repeat_seq.get! ((n - 1) % repeat_seq.length)

-- State the theorem to be proved
theorem digit_150_in_fraction : nth_digit 150 = 5 := 
sorry

end digit_150_in_fraction_l992_99231


namespace solve_system_eqns_l992_99241

theorem solve_system_eqns (x y : ℚ) 
    (h1 : (x - 30) / 3 = (2 * y + 7) / 4)
    (h2 : x - y = 10) :
  x = -81 / 2 ∧ y = -101 / 2 := 
sorry

end solve_system_eqns_l992_99241


namespace minimum_value_fraction_l992_99200

theorem minimum_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  ∃ (x : ℝ), (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 → x ≤ (a + b) / (a * b * c)) ∧ x = 16 / 9 := 
sorry

end minimum_value_fraction_l992_99200


namespace range_of_a_l992_99208

def p (x : ℝ) : Prop := (1/2 ≤ x ∧ x ≤ 1)

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬ p x) → 
  (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end range_of_a_l992_99208


namespace parabola_point_focus_distance_l992_99280

/-- 
  Given a point P on the parabola y^2 = 4x, and the distance from P to the line x = -2
  is 5 units, prove that the distance from P to the focus of the parabola is 4 units.
-/
theorem parabola_point_focus_distance {P : ℝ × ℝ} 
  (hP : P.2^2 = 4 * P.1) 
  (h_dist : (P.1 + 2)^2 + P.2^2 = 25) : 
  dist P (1, 0) = 4 :=
sorry

end parabola_point_focus_distance_l992_99280


namespace regular_pentagonal_pyramid_angle_l992_99282

noncomputable def angle_between_slant_height_and_non_intersecting_edge (base_edge_slant_height : ℝ) : ℝ :=
  -- Assuming the base edge and slant height are given as input and equal
  if base_edge_slant_height > 0 then 36 else 0

theorem regular_pentagonal_pyramid_angle
  (base_edge_slant_height : ℝ)
  (h : base_edge_slant_height > 0) :
  angle_between_slant_height_and_non_intersecting_edge base_edge_slant_height = 36 :=
by
  -- omitted proof steps
  sorry

end regular_pentagonal_pyramid_angle_l992_99282


namespace equilateral_triangle_l992_99259

variable (A B C A₀ B₀ C₀ : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]

variable (midpoint : ∀ (X₁ X₂ : Type), Type) 
variable (circumcircle : ∀ (X Y Z : Type), Type)

def medians_meet_circumcircle := ∀ (A A₁ B B₁ C C₁ : Type) 
  [AddGroup A] [AddGroup A₁] [AddGroup B] [AddGroup B₁] [AddGroup C] [AddGroup C₁], 
  Prop

def areas_equal := ∀ (ABC₀ AB₀C A₀BC : Type) 
  [AddGroup ABC₀] [AddGroup AB₀C] [AddGroup A₀BC], 
  Prop

theorem equilateral_triangle (A B C A₀ B₀ C₀ A₁ B₁ C₁ : Type)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]
  [AddGroup A₁] [AddGroup B₁] [AddGroup C₁] 
  (midpoint_cond : ∀ (X Y Z : Type), Z = midpoint X Y)
  (circumcircle_cond : ∀ (X Y Z : Type), Z = circumcircle X Y Z)
  (medians_meet_circumcircle : Prop)
  (areas_equal: Prop) :
    A = B ∧ B = C ∧ C = A :=
  sorry

end equilateral_triangle_l992_99259


namespace index_difference_l992_99213

noncomputable def index_females (n k1 k2 k3 : ℕ) : ℚ :=
  ((n - k1 + k2 : ℚ) / n) * (1 + k3 / 10)

noncomputable def index_males (n k1 l1 l2 : ℕ) : ℚ :=
  ((n - (n - k1) + l1 : ℚ) / n) * (1 + l2 / 10)

theorem index_difference (n k1 k2 k3 l1 l2 : ℕ)
  (h_n : n = 35) (h_k1 : k1 = 15) (h_k2 : k2 = 5) (h_k3 : k3 = 8)
  (h_l1 : l1 = 6) (h_l2 : l2 = 10) : 
  index_females n k1 k2 k3 - index_males n k1 l1 l2 = 3 / 35 :=
by
  sorry

end index_difference_l992_99213


namespace find_value_at_l992_99217

-- Defining the function f
variable (f : ℝ → ℝ)

-- Conditions
-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 4
def periodic_function (f : ℝ → ℝ) := ∀ x, f (x + 4) = f x

-- Condition 3: In the interval [0,1], f(x) = 3x
def definition_on_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- Statement to prove
theorem find_value_at (f : ℝ → ℝ) 
  (odd_f : odd_function f) 
  (periodic_f : periodic_function f) 
  (def_on_interval : definition_on_interval f) :
  f 11.5 = -1.5 := by 
  sorry

end find_value_at_l992_99217


namespace evaluate_expression_l992_99279

def star (A B : ℚ) : ℚ := (A + B) / 3

theorem evaluate_expression : star (star 7 15) 10 = 52 / 9 := by
  sorry

end evaluate_expression_l992_99279


namespace visible_sides_probability_l992_99266

theorem visible_sides_probability
  (r : ℝ)
  (side_length : ℝ := 4)
  (probability : ℝ := 3 / 4) :
  r = 8 * Real.sqrt 3 / 3 :=
sorry

end visible_sides_probability_l992_99266


namespace min_air_routes_l992_99254

theorem min_air_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) : 
  a + b + c ≥ 21 :=
sorry

end min_air_routes_l992_99254


namespace sum_of_roots_quadratic_eq_l992_99250

theorem sum_of_roots_quadratic_eq : ∀ P Q : ℝ, (3 * P^2 - 9 * P + 6 = 0) ∧ (3 * Q^2 - 9 * Q + 6 = 0) → P + Q = 3 :=
by
  sorry

end sum_of_roots_quadratic_eq_l992_99250


namespace Cannot_Halve_Triangles_With_Diagonals_l992_99238

structure Polygon where
  vertices : Nat
  edges : Nat

def is_convex (n : Nat) (P : Polygon) : Prop :=
  P.vertices = n ∧ P.edges = n

def non_intersecting_diagonals (P : Polygon) : Prop :=
  -- Assuming a placeholder for the actual non-intersecting diagonals condition
  true

def count_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  P.vertices - 2 -- This is the simplification used for counting triangles

def count_all_diagonals_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  -- Placeholder for function to count triangles formed exclusively by diagonals
  1000

theorem Cannot_Halve_Triangles_With_Diagonals (P : Polygon) (h : is_convex 2002 P) (d : non_intersecting_diagonals P) :
  count_triangles P d = 2000 → ¬ (count_all_diagonals_triangles P d = 1000) :=
by
  intro h1
  sorry

end Cannot_Halve_Triangles_With_Diagonals_l992_99238


namespace rectangle_perimeter_l992_99264

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b) = 4 * (2 * a + 2 * b) - 12) :
    (2 * (a + b) = 72) ∨ (2 * (a + b) = 100) := by
  sorry

end rectangle_perimeter_l992_99264


namespace age_problem_l992_99246

theorem age_problem (A B C D E : ℕ)
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : D = C / 2)
  (h4 : E = D - 3)
  (h5 : A + B + C + D + E = 52) : B = 16 :=
by
  sorry

end age_problem_l992_99246


namespace power_sum_l992_99243

theorem power_sum (h : (9 : ℕ) = 3^2) : (2^567 + (9^5 / 3^2) : ℕ) = 2^567 + 6561 := by
  sorry

end power_sum_l992_99243


namespace problem_statement_l992_99249

open Real

theorem problem_statement (α : ℝ) 
  (h1 : cos (α + π / 4) = (7 * sqrt 2) / 10)
  (h2 : cos (2 * α) = 7 / 25) :
  sin α + cos α = 1 / 5 :=
sorry

end problem_statement_l992_99249


namespace calculate_expression_l992_99204

theorem calculate_expression : 1453 - 250 * 2 + 130 / 5 = 979 := by
  sorry

end calculate_expression_l992_99204


namespace diagonal_of_rectangle_l992_99212

theorem diagonal_of_rectangle (l w d : ℝ) (h_length : l = 15) (h_area : l * w = 120) (h_diagonal : d^2 = l^2 + w^2) : d = 17 :=
by
  sorry

end diagonal_of_rectangle_l992_99212


namespace smallest_divisor_after_323_l992_99270

-- Let n be an even 4-digit number such that 323 is a divisor of n.
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisor (d n : ℕ) : Prop :=
  n % d = 0

theorem smallest_divisor_after_323 (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : is_divisor 323 n) : ∃ k, k > 323 ∧ is_divisor k n ∧ k = 340 :=
by
  sorry

end smallest_divisor_after_323_l992_99270


namespace volume_of_cube_for_tetrahedron_l992_99244

theorem volume_of_cube_for_tetrahedron (h : ℝ) (b1 b2 : ℝ) (V : ℝ) 
  (h_condition : h = 15) (b1_condition : b1 = 8) (b2_condition : b2 = 12)
  (V_condition : V = 3375) : 
  V = (max h (max b1 b2)) ^ 3 := by
  -- To illustrate the mathematical context and avoid concrete steps,
  -- sorry provides the completion of the logical binding to the correct answer
  sorry

end volume_of_cube_for_tetrahedron_l992_99244


namespace problem1_problem2_l992_99274

variable {a b : ℝ}

theorem problem1
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : (a + b) * (a^5 + b^5) ≥ 4 := sorry

theorem problem2
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : a + b ≤ 2 := sorry

end problem1_problem2_l992_99274


namespace discount_percentage_l992_99272

theorem discount_percentage (cp mp pm : ℤ) (x : ℤ) 
    (Hcp : cp = 160) 
    (Hmp : mp = 240) 
    (Hpm : pm = 20) 
    (Hcondition : mp * (100 - x) = cp * (100 + pm)) : 
  x = 20 := 
  sorry

end discount_percentage_l992_99272


namespace total_goals_by_other_players_l992_99236

theorem total_goals_by_other_players (total_players goals_season games_played : ℕ)
  (third_players_goals avg_goals_per_third : ℕ)
  (h1 : total_players = 24)
  (h2 : goals_season = 150)
  (h3 : games_played = 15)
  (h4 : third_players_goals = total_players / 3)
  (h5 : avg_goals_per_third = 1)
  : (goals_season - (third_players_goals * avg_goals_per_third * games_played)) = 30 :=
by
  sorry

end total_goals_by_other_players_l992_99236


namespace daragh_sisters_count_l992_99294

theorem daragh_sisters_count (initial_bears : ℕ) (favorite_bears : ℕ) (eden_initial_bears : ℕ) (eden_total_bears : ℕ) 
    (remaining_bears := initial_bears - favorite_bears)
    (eden_received_bears := eden_total_bears - eden_initial_bears)
    (bears_per_sister := eden_received_bears) :
    initial_bears = 20 → favorite_bears = 8 → eden_initial_bears = 10 → eden_total_bears = 14 → 
    remaining_bears / bears_per_sister = 3 := 
by
  sorry

end daragh_sisters_count_l992_99294


namespace days_to_shovel_l992_99299

-- Defining conditions as formal statements
def original_task_time := 10
def original_task_people := 10
def original_task_weight := 10000
def new_task_weight := 40000
def new_task_people := 5

-- Definition of rate in terms of weight, people and time
def rate_per_person (total_weight : ℕ) (total_people : ℕ) (total_time : ℕ) : ℕ :=
  total_weight / total_people / total_time

-- Theorem statement to prove
theorem days_to_shovel (t : ℕ) :
  (rate_per_person original_task_weight original_task_people original_task_time) * new_task_people * t = new_task_weight := sorry

end days_to_shovel_l992_99299


namespace propositionA_necessary_but_not_sufficient_for_propositionB_l992_99228

-- Definitions for propositions and conditions
def propositionA (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0
def propositionB (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement for the necessary but not sufficient condition
theorem propositionA_necessary_but_not_sufficient_for_propositionB (a : ℝ) :
  (propositionA a) → (¬ propositionB a) ∧ (propositionB a → propositionA a) :=
by
  sorry

end propositionA_necessary_but_not_sufficient_for_propositionB_l992_99228


namespace quarters_difference_nickels_eq_l992_99277

variable (q : ℕ)

def charles_quarters := 7 * q + 2
def richard_quarters := 3 * q + 7
def quarters_difference := charles_quarters q - richard_quarters q
def money_difference_in_nickels := 5 * quarters_difference q

theorem quarters_difference_nickels_eq :
  money_difference_in_nickels q = 20 * (q - 5/4) :=
by
  sorry

end quarters_difference_nickels_eq_l992_99277


namespace distinct_real_numbers_condition_l992_99220

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)

theorem distinct_real_numbers_condition (a b x1 x2 x3 : ℝ) :
  f a b x1 = x2 → f a b x2 = x3 → f a b x3 = x1 → x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → a = -b^2 :=
by
  sorry

end distinct_real_numbers_condition_l992_99220


namespace solve_inequality_l992_99268

theorem solve_inequality (a : ℝ) : 
  {x : ℝ | x^2 - (a + 2) * x + 2 * a > 0} = 
  (if a > 2 then {x | x < 2 ∨ x > a}
   else if a = 2 then {x | x ≠ 2}
   else {x | x < a ∨ x > 2}) :=
sorry

end solve_inequality_l992_99268


namespace find_k_l992_99229

theorem find_k (x y k : ℝ) (h_line : 2 - k * x = -4 * y) (h_point : x = 3 ∧ y = -2) : k = -2 :=
by
  -- Given the conditions that the point (3, -2) lies on the line 2 - kx = -4y, 
  -- we want to prove that k = -2
  sorry

end find_k_l992_99229


namespace determine_a_l992_99265

theorem determine_a (a : ℕ) : 
  (2 * 10^10 + a ) % 11 = 0 ∧ 0 ≤ a ∧ a < 11 → a = 9 :=
by
  sorry

end determine_a_l992_99265
