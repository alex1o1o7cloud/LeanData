import Mathlib

namespace frac_div_l81_81061

theorem frac_div : (3 / 7) / (4 / 5) = 15 / 28 := by
  sorry

end frac_div_l81_81061


namespace positive_difference_is_127_div_8_l81_81804

-- Defining the basic expressions
def eight_squared : ℕ := 8 ^ 2 -- 64

noncomputable def expr1 : ℝ := (eight_squared + eight_squared) / 8
noncomputable def expr2 : ℝ := (eight_squared / eight_squared) / 8

-- Problem statement
theorem positive_difference_is_127_div_8 :
  (expr1 - expr2) = 127 / 8 :=
by
  sorry

end positive_difference_is_127_div_8_l81_81804


namespace trash_cans_street_count_l81_81263

theorem trash_cans_street_count (S B : ℕ) (h1 : B = 2 * S) (h2 : S + B = 42) : S = 14 :=
by
  sorry

end trash_cans_street_count_l81_81263


namespace surveys_on_tuesday_l81_81025

theorem surveys_on_tuesday
  (num_surveys_monday: ℕ) -- number of surveys Bart completed on Monday
  (earnings_monday: ℕ) -- earning per survey on Monday
  (total_earnings: ℕ) -- total earnings over the two days
  (earnings_per_survey: ℕ) -- earnings Bart gets per survey
  (monday_earnings_eq : earnings_monday = num_surveys_monday * earnings_per_survey)
  (total_earnings_eq : total_earnings = earnings_monday + (8 : ℕ))
  (earnings_per_survey_eq : earnings_per_survey = 2)
  : ((8 : ℕ) / earnings_per_survey = 4) := sorry

end surveys_on_tuesday_l81_81025


namespace find_cost_price_l81_81101

theorem find_cost_price 
  (C : ℝ)
  (h1 : 1.10 * C + 110 = 1.15 * C)
  : C = 2200 :=
sorry

end find_cost_price_l81_81101


namespace system_equivalence_l81_81081

theorem system_equivalence (f g : ℝ → ℝ) (x : ℝ) (h1 : f x > 0) (h2 : g x > 0) : f x + g x > 0 :=
sorry

end system_equivalence_l81_81081


namespace arithmetic_sequence_num_terms_l81_81979

theorem arithmetic_sequence_num_terms (a d l : ℕ) (h1 : a = 15) (h2 : d = 4) (h3 : l = 159) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 37 :=
by {
  sorry
}

end arithmetic_sequence_num_terms_l81_81979


namespace octagon_mass_l81_81811

theorem octagon_mass :
  let side_length := 1 -- side length of the original square (meters)
  let thickness := 0.3 -- thickness of the sheet (cm)
  let density := 7.8 -- density of steel (g/cm^3)
  let x := 50 * (2 - Real.sqrt 2) -- side length of the triangles (cm)
  let octagon_area := 20000 * (Real.sqrt 2 - 1) -- area of the octagon (cm^2)
  let volume := octagon_area * thickness -- volume of the octagon (cm^3)
  let mass := volume * density / 1000 -- mass of the octagon (kg), converted from g to kg
  mass = 19 :=
by
  sorry

end octagon_mass_l81_81811


namespace frog_arrangement_l81_81631

def arrangementCount (total_frogs green_frogs red_frogs blue_frog : ℕ) : ℕ :=
  if (green_frogs + red_frogs + blue_frog = total_frogs ∧ 
      green_frogs = 3 ∧ red_frogs = 4 ∧ blue_frog = 1) then 40320 else 0

theorem frog_arrangement :
  arrangementCount 8 3 4 1 = 40320 :=
by {
  -- Proof omitted
  sorry
}

end frog_arrangement_l81_81631


namespace number_of_shoes_lost_l81_81595

-- Definitions for the problem conditions
def original_pairs : ℕ := 20
def pairs_left : ℕ := 15
def shoes_per_pair : ℕ := 2

-- Translating the conditions to individual shoe counts
def original_shoes : ℕ := original_pairs * shoes_per_pair
def remaining_shoes : ℕ := pairs_left * shoes_per_pair

-- Statement of the proof problem
theorem number_of_shoes_lost : original_shoes - remaining_shoes = 10 := by
  sorry

end number_of_shoes_lost_l81_81595


namespace find_y_l81_81834

theorem find_y (x y : ℝ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1 / 10 := by
  sorry

end find_y_l81_81834


namespace ratio_in_sequence_l81_81733

theorem ratio_in_sequence (a1 a2 b1 b2 b3 : ℝ)
  (h1 : ∃ d, a1 = 1 + d ∧ a2 = 1 + 2 * d ∧ 9 = 1 + 3 * d)
  (h2 : ∃ r, b1 = 1 * r ∧ b2 = 1 * r^2 ∧ b3 = 1 * r^3 ∧ 9 = 1 * r^4) :
  b2 / (a1 + a2) = 3 / 10 := by
  sorry

end ratio_in_sequence_l81_81733


namespace find_weight_of_a_l81_81104

theorem find_weight_of_a (A B C D E : ℕ) 
  (h1 : A + B + C = 3 * 84)
  (h2 : A + B + C + D = 4 * 80)
  (h3 : E = D + 3)
  (h4 : B + C + D + E = 4 * 79) : 
  A = 75 := by
  sorry

end find_weight_of_a_l81_81104


namespace price_of_kid_ticket_l81_81952

theorem price_of_kid_ticket (k a : ℤ) (hk : k = 6) (ha : a = 2)
  (price_kid price_adult : ℤ)
  (hprice_adult : price_adult = 2 * price_kid)
  (hcost_total : 6 * price_kid + 2 * price_adult = 50) :
  price_kid = 5 :=
by
  sorry

end price_of_kid_ticket_l81_81952


namespace number_of_sections_l81_81490

-- Definitions based on the conditions in a)
def num_reels : Nat := 3
def length_per_reel : Nat := 100
def section_length : Nat := 10

-- The math proof problem statement
theorem number_of_sections :
  (num_reels * length_per_reel) / section_length = 30 := by
  sorry

end number_of_sections_l81_81490


namespace containers_per_truck_l81_81372

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l81_81372


namespace solve_for_a_l81_81704

theorem solve_for_a (x a : ℝ) (h : x = 5) (h_eq : a * x - 8 = 10 + 4 * a) : a = 18 :=
by
  sorry

end solve_for_a_l81_81704


namespace shiny_pennies_probability_l81_81674

theorem shiny_pennies_probability :
  ∃ (a b : ℕ), gcd a b = 1 ∧ a / b = 5 / 11 ∧ a + b = 16 :=
sorry

end shiny_pennies_probability_l81_81674


namespace factor_expression_l81_81740

theorem factor_expression (x : ℤ) : 63 * x + 28 = 7 * (9 * x + 4) :=
by sorry

end factor_expression_l81_81740


namespace determinant_identity_l81_81319

variable (x y z w : ℝ)
variable (h1 : x * w - y * z = -3)

theorem determinant_identity :
  (x + z) * w - (y + w) * z = -3 :=
by sorry

end determinant_identity_l81_81319


namespace largest_even_among_consecutives_l81_81453

theorem largest_even_among_consecutives (x : ℤ) (h : (x + (x + 2) + (x + 4) = x + 18)) : x + 4 = 10 :=
by
  sorry

end largest_even_among_consecutives_l81_81453


namespace first_place_points_is_eleven_l81_81150

/-
Conditions:
1. Points are awarded as follows: first place = x points, second place = 7 points, third place = 5 points, fourth place = 2 points.
2. John participated 7 times in the competition.
3. John finished in each of the top four positions at least once.
4. The product of all the points John received was 38500.
Theorem: The first place winner receives 11 points.
-/

noncomputable def archery_first_place_points (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), -- number of times John finished first, second, third, fourth respectively
    a + b + c + d = 7 ∧ -- condition 2, John participated 7 times
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ -- condition 3, John finished each position at least once
    x ^ a * 7 ^ b * 5 ^ c * 2 ^ d = 38500 -- condition 4, product of all points John received

theorem first_place_points_is_eleven : archery_first_place_points 11 :=
  sorry

end first_place_points_is_eleven_l81_81150


namespace total_cost_correct_l81_81278

noncomputable def totalCost : ℝ :=
  let fuel_efficiences := [15, 12, 14, 10, 13, 15]
  let distances := [10, 6, 7, 5, 3, 9]
  let gas_prices := [3.5, 3.6, 3.4, 3.55, 3.55, 3.5]
  let gas_used := distances.zip fuel_efficiences |>.map (λ p => (p.1 : ℝ) / p.2)
  let costs := gas_prices.zip gas_used |>.map (λ p => p.1 * p.2)
  costs.sum

theorem total_cost_correct : abs (totalCost - 10.52884) < 0.01 := by
  sorry

end total_cost_correct_l81_81278


namespace partial_fraction_sum_l81_81067

theorem partial_fraction_sum :
  ∃ P Q R : ℚ, 
    P * ((-1 : ℚ) * (-2 : ℚ)) + Q * ((-3 : ℚ) * (-2 : ℚ)) + R * ((-3 : ℚ) * (1 : ℚ))
    = 14 ∧ 
    R * (1 : ℚ) * (3 : ℚ) + Q * ((-4 : ℚ) * (-3 : ℚ)) + P * ((3 : ℚ) * (1 : ℚ)) 
      = 12 ∧ 
    P + Q + R = 115 / 30 := by
  sorry

end partial_fraction_sum_l81_81067


namespace maximize_profit_at_six_l81_81227

-- Defining the functions (conditions)
def y1 (x : ℝ) : ℝ := 17 * x^2
def y2 (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := y1 x - y2 x

-- The condition x > 0
def x_pos (x : ℝ) : Prop := x > 0

-- Proving the maximum profit is achieved at x = 6 (question == answer)
theorem maximize_profit_at_six : ∀ x > 0, (∀ y > 0, y = profit x → x = 6) :=
by 
  intros x hx y hy
  sorry

end maximize_profit_at_six_l81_81227


namespace total_students_in_class_l81_81828

theorem total_students_in_class (front_pos back_pos : ℕ) (H_front : front_pos = 23) (H_back : back_pos = 23) : front_pos + back_pos - 1 = 45 :=
by
  -- No proof required as per instructions
  sorry

end total_students_in_class_l81_81828


namespace increasing_on_interval_of_m_l81_81415

def f (m x : ℝ) := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_on_interval_of_m (m : ℝ) :
  (∀ x : ℝ, 2 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0) → m ≤ 5 / 2 :=
sorry

end increasing_on_interval_of_m_l81_81415


namespace amelia_wins_probability_l81_81620

theorem amelia_wins_probability :
  let pA := 1 / 4
  let pB := 1 / 3
  let pC := 1 / 2
  let cycle_probability := (1 - pA) * (1 - pB) * (1 - pC)
  let infinite_series_sum := 1 / (1 - cycle_probability)
  let total_probability := pA * infinite_series_sum
  total_probability = 1 / 3 :=
by
  sorry

end amelia_wins_probability_l81_81620


namespace ron_spending_increase_l81_81072

variable (P Q : ℝ) -- initial price and quantity
variable (X : ℝ)   -- intended percentage increase in spending

theorem ron_spending_increase :
  (1 + X / 100) * P * Q = 1.25 * P * (0.92 * Q) →
  X = 15 := 
by
  sorry

end ron_spending_increase_l81_81072


namespace determine_x0_minus_y0_l81_81424

theorem determine_x0_minus_y0 
  (x0 y0 : ℝ)
  (data_points : List (ℝ × ℝ) := [(1, 2), (3, 5), (6, 8), (x0, y0)])
  (regression_eq : ∀ x, (x + 2) = (x + 2)) :
  x0 - y0 = -3 :=
by
  sorry

end determine_x0_minus_y0_l81_81424


namespace frac_mul_eq_l81_81583

theorem frac_mul_eq : (2/3) * (3/8) = 1/4 := 
by 
  sorry

end frac_mul_eq_l81_81583


namespace fully_loaded_truck_weight_l81_81409

def empty_truck : ℕ := 12000
def weight_soda_crate : ℕ := 50
def num_soda_crates : ℕ := 20
def weight_dryer : ℕ := 3000
def num_dryers : ℕ := 3
def weight_fresh_produce : ℕ := 2 * (weight_soda_crate * num_soda_crates)

def total_loaded_truck_weight : ℕ :=
  empty_truck + (weight_soda_crate * num_soda_crates) + weight_fresh_produce + (weight_dryer * num_dryers)

theorem fully_loaded_truck_weight : total_loaded_truck_weight = 24000 := by
  sorry

end fully_loaded_truck_weight_l81_81409


namespace integer_value_of_expression_l81_81428

theorem integer_value_of_expression (m n p : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ 9)
  (h3 : 2 ≤ n) (h4 : n ≤ 9) (h5 : 2 ≤ p) (h6 : p ≤ 9)
  (h7 : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  (m + n + p) / (m + n) = 1 :=
sorry

end integer_value_of_expression_l81_81428


namespace three_digit_numbers_l81_81734

theorem three_digit_numbers (a b c n : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
    (h5 : 0 ≤ c) (h6 : c ≤ 9) (h7 : n = 100 * a + 10 * b + c) (h8 : 10 * b + c = (100 * a + 10 * b + c) / 5) :
    n = 125 ∨ n = 250 ∨ n = 375 := 
by 
  sorry

end three_digit_numbers_l81_81734


namespace neg_p_implies_neg_q_sufficient_but_not_necessary_l81_81558

variables (x : ℝ) (p : Prop) (q : Prop)

def p_condition := (1 < x ∨ x < -3)
def q_condition := (5 * x - 6 > x ^ 2)

theorem neg_p_implies_neg_q_sufficient_but_not_necessary :
  p_condition x → q_condition x → ((¬ p_condition x) → (¬ q_condition x)) :=
by 
  intro h1 h2
  sorry

end neg_p_implies_neg_q_sufficient_but_not_necessary_l81_81558


namespace stationary_train_length_l81_81683

noncomputable def speed_train_kmh : ℝ := 144
noncomputable def speed_train_ms : ℝ := (speed_train_kmh * 1000) / 3600
noncomputable def time_to_pass_pole : ℝ := 8
noncomputable def time_to_pass_stationary : ℝ := 18
noncomputable def length_moving_train : ℝ := speed_train_ms * time_to_pass_pole
noncomputable def total_distance : ℝ := speed_train_ms * time_to_pass_stationary
noncomputable def length_stationary_train : ℝ := total_distance - length_moving_train

theorem stationary_train_length :
  length_stationary_train = 400 := by
  sorry

end stationary_train_length_l81_81683


namespace exponent_rule_example_l81_81872

theorem exponent_rule_example {a : ℝ} : (a^3)^4 = a^12 :=
by {
  sorry
}

end exponent_rule_example_l81_81872


namespace find_diameter_endpoint_l81_81754

def circle_center : ℝ × ℝ := (4, 1)
def diameter_endpoint_1 : ℝ × ℝ := (1, 5)

theorem find_diameter_endpoint :
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  (2 * h - x1, 2 * k - y1) = (7, -3) :=
by
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  sorry

end find_diameter_endpoint_l81_81754


namespace weight_of_new_person_l81_81629

-- Define the given conditions
variables (avg_increase : ℝ) (num_people : ℕ) (replaced_weight : ℝ)
variable (new_weight : ℝ)

-- These are the conditions directly from the problem
axiom avg_weight_increase : avg_increase = 4.5
axiom number_of_people : num_people = 6
axiom person_to_replace_weight : replaced_weight = 75

-- Mathematical equivalent of the proof problem
theorem weight_of_new_person :
  new_weight = replaced_weight + avg_increase * num_people := 
sorry

end weight_of_new_person_l81_81629


namespace truck_boxes_per_trip_l81_81327

theorem truck_boxes_per_trip (total_boxes trips : ℕ) (h1 : total_boxes = 871) (h2 : trips = 218) : total_boxes / trips = 4 := by
  sorry

end truck_boxes_per_trip_l81_81327


namespace value_is_6_l81_81493

-- We know the conditions that the least number which needs an increment is 858
def least_number : ℕ := 858

-- Define the numbers 24, 32, 36, and 54
def num1 : ℕ := 24
def num2 : ℕ := 32
def num3 : ℕ := 36
def num4 : ℕ := 54

-- Define the LCM function to compute the least common multiple
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- Define the LCM of the four numbers
def lcm_all : ℕ := lcm (lcm num1 num2) (lcm num3 num4)

-- Compute the value that needs to be added
def value_to_be_added : ℕ := lcm_all - least_number

-- Prove that this value equals to 6
theorem value_is_6 : value_to_be_added = 6 := by
  -- Proof would go here
  sorry

end value_is_6_l81_81493


namespace total_yarn_length_is_1252_l81_81882

/-- Defining the lengths of the yarns according to the conditions --/
def green_yarn : ℕ := 156
def red_yarn : ℕ := 3 * green_yarn + 8
def blue_yarn : ℕ := (green_yarn + red_yarn) / 2
def average_yarn_length : ℕ := (green_yarn + red_yarn + blue_yarn) / 3
def yellow_yarn : ℕ := average_yarn_length - 12

/-- Proving the total length of the four pieces of yarn is 1252 cm --/
theorem total_yarn_length_is_1252 :
  green_yarn + red_yarn + blue_yarn + yellow_yarn = 1252 := by
  sorry

end total_yarn_length_is_1252_l81_81882


namespace third_vs_second_plant_relationship_l81_81113

-- Define the constants based on the conditions
def first_plant_tomatoes := 24
def second_plant_tomatoes := 12 + 5  -- Half of 24 plus 5
def total_tomatoes := 60

-- Define the production of the third plant based on the total number of tomatoes
def third_plant_tomatoes := total_tomatoes - (first_plant_tomatoes + second_plant_tomatoes)

-- Define the relationship to be proved
theorem third_vs_second_plant_relationship : 
  third_plant_tomatoes = second_plant_tomatoes + 2 :=
by
  -- Proof not provided, adding sorry to skip
  sorry

end third_vs_second_plant_relationship_l81_81113


namespace initial_students_per_group_l81_81290

-- Define the conditions
variables {x : ℕ} (h : 3 * x - 2 = 22)

-- Lean 4 statement of the proof problem
theorem initial_students_per_group (x : ℕ) (h : 3 * x - 2 = 22) : x = 8 :=
sorry

end initial_students_per_group_l81_81290


namespace leak_drains_in_34_hours_l81_81580

-- Define the conditions
def pump_rate := 1 / 2 -- rate at which the pump fills the tank (tanks per hour)
def time_with_leak := 17 / 8 -- time to fill the tank with the pump and the leak (hours)

-- Define the combined rate of pump and leak
def combined_rate := 1 / time_with_leak -- tanks per hour

-- Define the leak rate
def leak_rate := pump_rate - combined_rate -- solve for leak rate

-- Define the proof statement
theorem leak_drains_in_34_hours : (1 / leak_rate) = 34 := by
    sorry

end leak_drains_in_34_hours_l81_81580


namespace a_eq_b_pow_n_l81_81211

theorem a_eq_b_pow_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (a - k^n) % (b - k) = 0) : a = b^n :=
sorry

end a_eq_b_pow_n_l81_81211


namespace volume_truncated_cone_l81_81673

-- Define the geometric constants
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height_truncated_cone : ℝ := 8

-- The statement to prove the volume of the truncated cone
theorem volume_truncated_cone :
  let V_large := (1/3) * Real.pi * (large_base_radius^2) * (height_truncated_cone + height_truncated_cone)
  let V_small := (1/3) * Real.pi * (small_base_radius^2) * height_truncated_cone
  V_large - V_small = (1400/3) * Real.pi :=
by
  sorry

end volume_truncated_cone_l81_81673


namespace subtraction_divisible_l81_81188

theorem subtraction_divisible (n m d : ℕ) (h1 : n = 13603) (h2 : m = 31) (h3 : d = 13572) : 
  (n - m) % d = 0 := by
  sorry

end subtraction_divisible_l81_81188


namespace vertex_x_coordinate_l81_81846

theorem vertex_x_coordinate (a b c : ℝ) :
  (∀ x, x = 0 ∨ x = 4 ∨ x = 7 →
    (0 ≤ x ∧ x ≤ 7 →
      (x = 0 → c = 1) ∧
      (x = 4 → 16 * a + 4 * b + c = 1) ∧
      (x = 7 → 49 * a + 7 * b + c = 5))) →
  (2 * x = 2 * 2 - b / a) ∧ (0 ≤ x ∧ x ≤ 7) :=
sorry

end vertex_x_coordinate_l81_81846


namespace train_distance_covered_l81_81854

-- Definitions based on the given conditions
def average_speed := 3   -- in meters per second
def total_time := 9      -- in seconds

-- Theorem statement: Given the average speed and total time, the total distance covered is 27 meters
theorem train_distance_covered : average_speed * total_time = 27 := 
by
  sorry

end train_distance_covered_l81_81854


namespace find_b_l81_81990

theorem find_b (a b : ℝ) (h1 : (1 : ℝ)^3 + a*(1)^2 + b*1 + a^2 = 10)
    (h2 : 3*(1 : ℝ)^2 + 2*a*(1) + b = 0) : b = -11 :=
sorry

end find_b_l81_81990


namespace laptop_final_price_l81_81692

theorem laptop_final_price (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  initial_price = 500 → first_discount = 10 → second_discount = 20 →
  (initial_price * (1 - first_discount / 100) * (1 - second_discount / 100)) = initial_price * 0.72 :=
by
  sorry

end laptop_final_price_l81_81692


namespace find_other_number_l81_81912

theorem find_other_number
  (a b : ℕ)
  (HCF : ℕ)
  (LCM : ℕ)
  (h1 : HCF = 12)
  (h2 : LCM = 396)
  (h3 : a = 36)
  (h4 : HCF * LCM = a * b) :
  b = 132 :=
by
  sorry

end find_other_number_l81_81912


namespace complex_numbers_are_real_l81_81365

theorem complex_numbers_are_real
  (a b c : ℂ)
  (h1 : (a + b) * (a + c) = b)
  (h2 : (b + c) * (b + a) = c)
  (h3 : (c + a) * (c + b) = a) : 
  a.im = 0 ∧ b.im = 0 ∧ c.im = 0 :=
sorry

end complex_numbers_are_real_l81_81365


namespace configuration_of_points_l81_81407

-- Define a type for points
structure Point :=
(x : ℝ)
(y : ℝ)

-- Assuming general position in the plane
def general_position (points : List Point) : Prop :=
  -- Add definition of general position, skipping exact implementation
  sorry

-- Define the congruence condition
def triangles_congruent (points : List Point) : Prop :=
  -- Add definition of the congruent triangles condition
  sorry

-- Define the vertices of two equilateral triangles inscribed in a circle
def two_equilateral_triangles (points : List Point) : Prop :=
  -- Add definition to check if points form two equilateral triangles in a circle
  sorry

theorem configuration_of_points (points : List Point) (h6 : points.length = 6) :
  general_position points →
  triangles_congruent points →
  two_equilateral_triangles points :=
by
  sorry

end configuration_of_points_l81_81407


namespace math_problem_l81_81562

theorem math_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 35) : L = 1631 := 
by
  sorry

end math_problem_l81_81562


namespace find_k_l81_81161

-- Define the set A using a condition on the quadratic equation
def A (k : ℝ) : Set ℝ := {x | k * x ^ 2 + 4 * x + 4 = 0}

-- Define the condition for the set A to have exactly one element
def has_exactly_one_element (k : ℝ) : Prop :=
  ∃ x : ℝ, A k = {x}

-- The problem statement is to find the value of k for which A has exactly one element
theorem find_k : ∃ k : ℝ, has_exactly_one_element k ∧ k = 1 :=
by
  simp [has_exactly_one_element, A]
  sorry

end find_k_l81_81161


namespace determine_f_3_2016_l81_81221

noncomputable def f : ℕ → ℕ → ℕ
| 0, y       => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem determine_f_3_2016 : f 3 2016 = 2 ^ 2019 - 3 := by
  sorry

end determine_f_3_2016_l81_81221


namespace students_in_high_school_l81_81887

-- Definitions from conditions
def H (L: ℝ) : ℝ := 4 * L
def middleSchoolStudents : ℝ := 300
def combinedStudents (H: ℝ) (L: ℝ) : ℝ := H + L
def combinedIsSevenTimesMiddle (H: ℝ) (L: ℝ) : Prop := combinedStudents H L = 7 * middleSchoolStudents

-- The main goal to prove
theorem students_in_high_school (L H: ℝ) (h1: H = 4 * L) (h2: combinedIsSevenTimesMiddle H L) : H = 1680 := by
  sorry

end students_in_high_school_l81_81887


namespace angle_measure_F_l81_81047

theorem angle_measure_F (D E F : ℝ) 
  (h1 : D = 75) 
  (h2 : E = 4 * F - 15) 
  (h3 : D + E + F = 180) : 
  F = 24 := 
sorry

end angle_measure_F_l81_81047


namespace average_total_goals_l81_81796

theorem average_total_goals (carter_avg shelby_avg judah_avg total_avg : ℕ) 
    (h1: carter_avg = 4) 
    (h2: shelby_avg = carter_avg / 2)
    (h3: judah_avg = 2 * shelby_avg - 3) 
    (h4: total_avg = carter_avg + shelby_avg + judah_avg) :
  total_avg = 7 :=
by
  sorry

end average_total_goals_l81_81796


namespace necessary_but_not_sufficient_condition_l81_81236

theorem necessary_but_not_sufficient_condition (x : ℝ) : |x - 1| < 2 → -3 < x ∧ x < 3 :=
by
  sorry

end necessary_but_not_sufficient_condition_l81_81236


namespace age_problem_l81_81420

theorem age_problem (M D : ℕ) (h1 : M = 40) (h2 : 2 * D + M = 70) : 2 * M + D = 95 := by
  sorry

end age_problem_l81_81420


namespace two_positive_numbers_inequality_three_positive_numbers_am_gm_l81_81333

theorem two_positive_numbers_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 ≥ x^2 * y + x * y^2 ∧ (x = y ↔ x^3 + y^3 = x^2 * y + x * y^2) := by
sorry

theorem three_positive_numbers_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≥ (a * b * c)^(1/3) ∧ (a = b ∧ b = c ↔ (a + b + c) / 3 = (a * b * c)^(1/3)) := by
sorry

end two_positive_numbers_inequality_three_positive_numbers_am_gm_l81_81333


namespace nina_money_l81_81039

theorem nina_money (W : ℝ) (h1 : W > 0) (h2 : 10 * W = 14 * (W - 1)) : 10 * W = 35 := by
  sorry

end nina_money_l81_81039


namespace greatest_value_inequality_l81_81398

theorem greatest_value_inequality (x : ℝ) :
  x^2 - 6 * x + 8 ≤ 0 → x ≤ 4 := 
sorry

end greatest_value_inequality_l81_81398


namespace cosine_of_arcsine_l81_81251

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end cosine_of_arcsine_l81_81251


namespace scientific_notation_of_169200000000_l81_81699

theorem scientific_notation_of_169200000000 : 169200000000 = 1.692 * 10^11 :=
by sorry

end scientific_notation_of_169200000000_l81_81699


namespace area_of_rectangle_with_diagonal_length_l81_81179

variable (x : ℝ)

def rectangle_area_given_diagonal_length (x : ℝ) : Prop :=
  ∃ (w l : ℝ), l = 3 * w ∧ w^2 + l^2 = x^2 ∧ (w * l = (3 / 10) * x^2)

theorem area_of_rectangle_with_diagonal_length (x : ℝ) :
  rectangle_area_given_diagonal_length x :=
sorry

end area_of_rectangle_with_diagonal_length_l81_81179


namespace polynomial_perfect_square_l81_81679

theorem polynomial_perfect_square (k : ℤ) : (∃ b : ℤ, (x + b)^2 = x^2 + 8 * x + k) -> k = 16 := by
  sorry

end polynomial_perfect_square_l81_81679


namespace swimming_speed_solution_l81_81451

-- Definition of the conditions
def speed_of_water : ℝ := 2
def distance_against_current : ℝ := 10
def time_against_current : ℝ := 5

-- Definition of the person's swimming speed in still water
def swimming_speed_in_still_water (v : ℝ) :=
  distance_against_current = (v - speed_of_water) * time_against_current

-- Main theorem we want to prove
theorem swimming_speed_solution : 
  ∃ v : ℝ, swimming_speed_in_still_water v ∧ v = 4 :=
by
  sorry

end swimming_speed_solution_l81_81451


namespace bill_sunday_miles_l81_81918

variables (B J M S : ℝ)

-- Conditions
def condition_1 := B + 4
def condition_2 := 2 * (B + 4)
def condition_3 := J = 0 ∧ M = 5 ∧ (M + 2 = 7)
def condition_4 := (B + 5) + (B + 4) + 2 * (B + 4) + 7 = 50

-- The main theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (h1 : S = B + 4) (h2 : ∀ B, J = 0 → M = 5 → S + 2 = 7 → (B + 5) + S + 2 * S + 7 = 50) : S = 10.5 :=
by {
  sorry
}

end bill_sunday_miles_l81_81918


namespace problem_solution_l81_81877

theorem problem_solution (k a b : ℝ) (h1 : k = a + Real.sqrt b) 
  (h2 : abs (Real.logb 5 k - Real.logb 5 (k^2 + 3)) = 0.6) : 
  a + b = 15 :=
sorry

end problem_solution_l81_81877


namespace Trisha_walked_total_distance_l81_81613

theorem Trisha_walked_total_distance 
  (d1 d2 d3 : ℝ) (h_d1 : d1 = 0.11) (h_d2 : d2 = 0.11) (h_d3 : d3 = 0.67) :
  d1 + d2 + d3 = 0.89 :=
by sorry

end Trisha_walked_total_distance_l81_81613


namespace arrangement_count_equivalent_problem_l81_81477

noncomputable def number_of_unique_arrangements : Nat :=
  let n : Nat := 6 -- Number of balls and boxes
  let match_3_boxes_ways := Nat.choose n 3 -- Choosing 3 boxes out of 6
  let permute_remaining_boxes := 2 -- Permutations of the remaining 3 boxes such that no numbers match
  match_3_boxes_ways * permute_remaining_boxes

theorem arrangement_count_equivalent_problem :
  number_of_unique_arrangements = 40 := by
  sorry

end arrangement_count_equivalent_problem_l81_81477


namespace coverage_is_20_l81_81857

noncomputable def cost_per_kg : ℝ := 60
noncomputable def total_cost : ℝ := 1800
noncomputable def side_length : ℝ := 10

-- Surface area of one side of the cube
noncomputable def area_side : ℝ := side_length * side_length

-- Total surface area of the cube
noncomputable def total_area : ℝ := 6 * area_side

-- Kilograms of paint used
noncomputable def kg_paint_used : ℝ := total_cost / cost_per_kg

-- Coverage per kilogram of paint
noncomputable def coverage_per_kg (total_area : ℝ) (kg_paint_used : ℝ) : ℝ := total_area / kg_paint_used

theorem coverage_is_20 : coverage_per_kg total_area kg_paint_used = 20 := by
  sorry

end coverage_is_20_l81_81857


namespace vertex_parabola_l81_81984

theorem vertex_parabola (h k : ℝ) : 
  (∀ x : ℝ, -((x - 2)^2) + 3 = k) → (h = 2 ∧ k = 3) :=
by 
  sorry

end vertex_parabola_l81_81984


namespace find_ab_l81_81998

noncomputable def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

theorem find_ab (a b : ℝ) :
  (f 1 a b = 10) ∧ ((3 * 1^2 - 2 * a * 1 - b = 0)) → (a, b) = (-4, 11) ∨ (a, b) = (3, -3) :=
by
  sorry

end find_ab_l81_81998


namespace line_forms_equivalence_l81_81612

noncomputable def points (P Q : ℝ × ℝ) : Prop := 
  ∃ m c, ∃ b d, P = (b, m * b + c) ∧ Q = (d, m * d + c)

theorem line_forms_equivalence :
  points (-2, 3) (4, -1) →
  (∀ x y : ℝ, (y + 1) / (3 + 1) = (x - 4) / (-2 - 4)) ∧
  (∀ x y : ℝ, y + 1 = - (2 / 3) * (x - 4)) ∧
  (∀ x y : ℝ, y = - (2 / 3) * x + 5 / 3) ∧
  (∀ x y : ℝ, x / (5 / 2) + y / (5 / 3) = 1) :=
  sorry

end line_forms_equivalence_l81_81612


namespace area_above_line_of_circle_l81_81655

-- Define the circle equation
def circle_eq (x y : ℝ) := (x - 10)^2 + (y - 5)^2 = 50

-- Define the line equation
def line_eq (x y : ℝ) := y = x - 6

-- The area to determine
def area_above_line (R : ℝ) := 25 * R

-- Proof statement
theorem area_above_line_of_circle : area_above_line Real.pi = 25 * Real.pi :=
by
  -- mark the proof as sorry to skip the proof
  sorry

end area_above_line_of_circle_l81_81655


namespace andrew_calculation_l81_81144

theorem andrew_calculation (x y : ℝ) (hx : x ≠ 0) :
  0.4 * 0.5 * x = 0.2 * 0.3 * y → y = (10 / 3) * x :=
by
  sorry

end andrew_calculation_l81_81144


namespace all_numbers_equal_l81_81265

theorem all_numbers_equal 
  (x : Fin 2007 → ℝ)
  (h : ∀ (I : Finset (Fin 2007)), I.card = 7 → ∃ (J : Finset (Fin 2007)), J.card = 11 ∧ 
  (1 / 7 : ℝ) * I.sum x = (1 / 11 : ℝ) * J.sum x) :
  ∃ c : ℝ, ∀ i : Fin 2007, x i = c :=
by sorry

end all_numbers_equal_l81_81265


namespace moles_of_C6H5CH3_formed_l81_81736

-- Stoichiometry of the reaction
def balanced_reaction (C6H6 CH4 C6H5CH3 H2 : ℝ) : Prop :=
  C6H6 + CH4 = C6H5CH3 + H2

-- Given conditions
def reaction_conditions (initial_CH4 : ℝ) (initial_C6H6 final_C6H5CH3 final_H2 : ℝ) : Prop :=
  balanced_reaction initial_C6H6 initial_CH4 final_C6H5CH3 final_H2 ∧ initial_CH4 = 3 ∧ final_H2 = 3

-- Theorem to prove
theorem moles_of_C6H5CH3_formed (initial_CH4 final_C6H5CH3 : ℝ) : reaction_conditions initial_CH4 3 final_C6H5CH3 3 → final_C6H5CH3 = 3 :=
by
  intros h
  sorry

end moles_of_C6H5CH3_formed_l81_81736


namespace weight_of_7_weights_l81_81959

theorem weight_of_7_weights :
  ∀ (w : ℝ), (16 * w + 0.6 = 17.88) → 7 * w = 7.56 :=
by
  intros w h
  sorry

end weight_of_7_weights_l81_81959


namespace incorrect_option_c_l81_81700

theorem incorrect_option_c (a b c d : ℝ)
  (h1 : a + b + c ≥ d)
  (h2 : a + b + d ≥ c)
  (h3 : a + c + d ≥ b)
  (h4 : b + c + d ≥ a) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) :=
by sorry

end incorrect_option_c_l81_81700


namespace trajectory_of_point_l81_81841

theorem trajectory_of_point (x y k : ℝ) (hx : x ≠ 0) (hk : k ≠ 0) (h : |y| / |x| = k) : y = k * x ∨ y = -k * x :=
by
  sorry

end trajectory_of_point_l81_81841


namespace sufficient_but_not_necessary_condition_l81_81192

theorem sufficient_but_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x > 0 ∧ y > 0 → (x / y + y / x ≥ 2)) ∧ ¬((x / y + y / x ≥ 2) → (x > 0 ∧ y > 0)) :=
sorry

end sufficient_but_not_necessary_condition_l81_81192


namespace sum_of_four_digit_integers_up_to_4999_l81_81401

theorem sum_of_four_digit_integers_up_to_4999 : 
  let a := 1000
  let l := 4999
  let n := l - a + 1
  let S := (n / 2) * (a + l)
  S = 11998000 := 
by
  sorry

end sum_of_four_digit_integers_up_to_4999_l81_81401


namespace august_8th_is_saturday_l81_81466

-- Defining the conditions
def august_has_31_days : Prop := true

def august_has_5_mondays : Prop := true

def august_has_4_tuesdays : Prop := true

-- Statement of the theorem
theorem august_8th_is_saturday (h1 : august_has_31_days) (h2 : august_has_5_mondays) (h3 : august_has_4_tuesdays) : ∃ d : ℕ, d = 6 :=
by
  -- Translate the correct answer "August 8th is a Saturday" into the equivalent proposition
  -- Saturday is represented by 6 if we assume 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
  sorry

end august_8th_is_saturday_l81_81466


namespace largest_square_test_plots_l81_81680

/-- 
  A fenced, rectangular field measures 30 meters by 45 meters. 
  An agricultural researcher has 1500 meters of fence that can be used for internal fencing to partition 
  the field into congruent, square test plots. 
  The entire field must be partitioned, and the sides of the squares must be parallel to the edges of the field. 
  What is the largest number of square test plots into which the field can be partitioned using all or some of the 1500 meters of fence?
 -/
theorem largest_square_test_plots
  (field_length : ℕ := 30)
  (field_width : ℕ := 45)
  (total_fence_length : ℕ := 1500):
  ∃ (n : ℕ), n = 576 := 
sorry

end largest_square_test_plots_l81_81680


namespace even_function_increasing_l81_81651

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * b * x + 1

theorem even_function_increasing (h_even : ∀ x : ℝ, f a b x = f a b (-x))
  (h_increasing : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → f a b x < f a b y) :
  f a b (a-2) < f a b (b+1) :=
sorry

end even_function_increasing_l81_81651


namespace squirrel_burrow_has_44_walnuts_l81_81091

def boy_squirrel_initial := 30
def boy_squirrel_gathered := 20
def boy_squirrel_dropped := 4
def boy_squirrel_hid := 8
-- "Forgets where he hid 3 of them" does not affect the main burrow

def girl_squirrel_brought := 15
def girl_squirrel_ate := 5
def girl_squirrel_gave := 4
def girl_squirrel_lost_playing := 3
def girl_squirrel_knocked := 2

def third_squirrel_gathered := 10
def third_squirrel_dropped := 1
def third_squirrel_hid := 3
def third_squirrel_returned := 6 -- Given directly instead of as a formula step; 9-3=6
def third_squirrel_gave := 1 -- Given directly as a friend

def final_walnuts := boy_squirrel_initial + boy_squirrel_gathered
                    - boy_squirrel_dropped - boy_squirrel_hid
                    + girl_squirrel_brought - girl_squirrel_ate
                    - girl_squirrel_gave - girl_squirrel_lost_playing
                    - girl_squirrel_knocked + third_squirrel_returned

theorem squirrel_burrow_has_44_walnuts :
  final_walnuts = 44 :=
by
  sorry

end squirrel_burrow_has_44_walnuts_l81_81091


namespace car_collision_frequency_l81_81764

theorem car_collision_frequency
  (x : ℝ)
  (h_collision : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * x)
  (h_big_crash : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * 20)
  (h_total_accidents : 240 / x + 240 / 20 = 36) :
  x = 10 :=
by
  sorry

end car_collision_frequency_l81_81764


namespace sin_alpha_value_l81_81455

-- Given conditions
variables (α : ℝ) (h1 : Real.tan α = -5 / 12) (h2 : π / 2 < α ∧ α < π)

-- Assertion to prove
theorem sin_alpha_value : Real.sin α = 5 / 13 :=
by
  -- Proof goes here
  sorry

end sin_alpha_value_l81_81455


namespace train_length_l81_81082

theorem train_length (L : ℝ) (h1 : 46 - 36 = 10) (h2 : 45 * (10 / 3600) = 1 / 8) : L = 62.5 :=
by
  sorry

end train_length_l81_81082


namespace vacuum_tube_pins_and_holes_l81_81402

theorem vacuum_tube_pins_and_holes :
  ∀ (pins holes : Finset ℕ), 
  pins = {1, 2, 3, 4, 5, 6, 7} →
  holes = {1, 2, 3, 4, 5, 6, 7} →
  (∃ (a : ℕ), ∀ k ∈ pins, ∃ b ∈ holes, (2 * k) % 7 = b) := by
  sorry

end vacuum_tube_pins_and_holes_l81_81402


namespace curvilinear_quadrilateral_area_l81_81202

-- Conditions: Define radius R, and plane angles of the tetrahedral angle.
noncomputable def radius (R : Real) : Prop :=
  R > 0

noncomputable def angle (theta : Real) : Prop :=
  theta = 60

-- Establishing the final goal based on the given conditions and solution's correct answer.
theorem curvilinear_quadrilateral_area
  (R : Real)     -- given radius of the sphere
  (hR : radius R) -- the radius of the sphere touching all edges
  (theta : Real)  -- given angle in degrees
  (hθ : angle theta) -- the plane angle of 60 degrees
  :
  ∃ A : Real, 
    A = π * R^2 * (16/3 * (Real.sqrt (2/3)) - 2) := 
  sorry

end curvilinear_quadrilateral_area_l81_81202


namespace root_range_m_l81_81209

theorem root_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * m * x + 4 = 0 → (x > 1 ∧ ∃ y : ℝ, y < 1 ∧ y^2 - 2 * m * y + 4 = 0)
  ∨ (x < 1 ∧ ∃ y : ℝ, y > 1 ∧ y^2 - 2 * m * y + 4 = 0))
  → m > 5 / 2 := 
sorry

end root_range_m_l81_81209


namespace square_and_product_l81_81616

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : (x = 42) ∧ ((x + 2) * (x - 2) = 1760) :=
by
  sorry

end square_and_product_l81_81616


namespace restore_triangle_Nagel_point_l81_81533

-- Define the variables and types involved
variables {Point : Type}

-- Assume a structure to capture the properties of a triangle
structure Triangle (Point : Type) :=
(A B C : Point)

-- Define the given conditions
variables (N B E : Point)

-- Statement of the main Lean theorem to reconstruct the triangle ABC
theorem restore_triangle_Nagel_point 
    (N B E : Point) :
    ∃ (ABC : Triangle Point), 
      (ABC).B = B ∧
      -- Additional properties of the triangle to be stated here
      sorry
    :=
sorry

end restore_triangle_Nagel_point_l81_81533


namespace three_students_with_A_l81_81041

-- Define the statements of the students
variables (Eliza Fiona George Harry : Prop)

-- Conditions based on the problem statement
axiom Fiona_implies_Eliza : Fiona → Eliza
axiom George_implies_Fiona : George → Fiona
axiom Harry_implies_George : Harry → George

-- There are exactly three students who scored an A
theorem three_students_with_A (hE : Bool) : 
  (Eliza = false) → (Fiona = true) → (George = true) → (Harry = true) :=
by
  sorry

end three_students_with_A_l81_81041


namespace solve_for_y_l81_81220

def solution (y : ℝ) : Prop :=
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = Real.pi / 4

theorem solve_for_y (y : ℝ) : solution y → y = 31 / 9 :=
by
  intro h
  sorry

end solve_for_y_l81_81220


namespace polynomial_coefficient_l81_81520

theorem polynomial_coefficient :
  ∀ d : ℝ, (2 * (2 : ℝ)^4 + 3 * (2 : ℝ)^3 + d * (2 : ℝ)^2 - 4 * (2 : ℝ) + 15 = 0) ↔ (d = -15.75) :=
by
  sorry

end polynomial_coefficient_l81_81520


namespace range_of_a_for_two_zeros_l81_81470

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ x : ℝ, (x + 1) * Real.exp x - a = 0 → -- There's no need to delete this part, see below note 
                                              -- The question of "exactly" is virtually ensured by other parts of the Lean theories
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                (x₁ + 1) * Real.exp x₁ - a = 0 ∧
                (x₂ + 1) * Real.exp x₂ - a = 0) → 
  (-1 / Real.exp 2 < a ∧ a < 0) :=
sorry

end range_of_a_for_two_zeros_l81_81470


namespace max_sqrt_sum_l81_81946

theorem max_sqrt_sum (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hxy : x + y = 8) :
  abs (Real.sqrt (x - 1 / y) + Real.sqrt (y - 1 / x)) ≤ Real.sqrt 15 :=
sorry

end max_sqrt_sum_l81_81946


namespace cos_alpha_plus_pi_six_l81_81607

theorem cos_alpha_plus_pi_six (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (α + Real.pi / 6) = - (4 / 5) := 
by 
  sorry

end cos_alpha_plus_pi_six_l81_81607


namespace ratio_of_45_and_9_l81_81752

theorem ratio_of_45_and_9 : (45 / 9) = 5 := 
by
  sorry

end ratio_of_45_and_9_l81_81752


namespace y_intercept_of_parallel_line_l81_81661

theorem y_intercept_of_parallel_line (m : ℝ) (c1 c2 : ℝ) (x1 y1 : ℝ) (H_parallel : m = -3) (H_passing : (x1, y1) = (1, -4)) : 
    c2 = -1 :=
  sorry

end y_intercept_of_parallel_line_l81_81661


namespace joes_bid_l81_81141

/--
Nelly tells her daughter she outbid her rival Joe by paying $2000 more than thrice his bid.
Nelly got the painting for $482,000. Prove that Joe's bid was $160,000.
-/
theorem joes_bid (J : ℝ) (h1 : 482000 = 3 * J + 2000) : J = 160000 :=
by
  sorry

end joes_bid_l81_81141


namespace TamekaBoxesRelation_l81_81556

theorem TamekaBoxesRelation 
  (S : ℤ)
  (h1 : 40 + S + S / 2 = 145) :
  S - 40 = 30 :=
by
  sorry

end TamekaBoxesRelation_l81_81556


namespace parallelogram_sides_l81_81590

theorem parallelogram_sides (x y : ℕ) 
  (h₁ : 2 * x + 3 = 9) 
  (h₂ : 8 * y - 1 = 7) : 
  x + y = 4 :=
by
  sorry

end parallelogram_sides_l81_81590


namespace set_representation_l81_81570

open Nat

def isInPositiveNaturals (x : ℕ) : Prop :=
  x ≠ 0

def isPositiveDivisor (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

theorem set_representation :
  {x | isInPositiveNaturals x ∧ isPositiveDivisor 6 (6 - x)} = {3, 4, 5} :=
by
  sorry

end set_representation_l81_81570


namespace carmela_gives_each_l81_81775

noncomputable def money_needed_to_give_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) : ℕ :=
  let total_cousins_money := cousins * cousins_count
  let total_money := carmela + total_cousins_money
  let people_count := 1 + cousins_count
  let equal_share := total_money / people_count
  let total_giveaway := carmela - equal_share
  total_giveaway / cousins_count

theorem carmela_gives_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) (h_carmela : carmela = 7) (h_cousins : cousins = 2) (h_cousins_count : cousins_count = 4) :
  money_needed_to_give_each carmela cousins cousins_count = 1 :=
by
  rw [h_carmela, h_cousins, h_cousins_count]
  sorry

end carmela_gives_each_l81_81775


namespace expression_evaluation_l81_81860

theorem expression_evaluation (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 :=
by
  sorry

end expression_evaluation_l81_81860


namespace quadratic_sum_constants_l81_81647

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 27 * x + 135

-- Define the representation of the quadratic in the form a(x + b)^2 + c
def quadratic_rewritten (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum_constants :
  ∃ a b c, (∀ x, quadratic x = quadratic_rewritten a b c x) ∧ a + b + c = 197.75 :=
by
  sorry

end quadratic_sum_constants_l81_81647


namespace soap_bubble_radius_l81_81654

/-- Given a spherical soap bubble that divides into two equal hemispheres, 
    each having a radius of 6 * (2 ^ (1 / 3)) cm, 
    show that the radius of the original bubble is also 6 * (2 ^ (1 / 3)) cm. -/
theorem soap_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h_r : r = 6 * (2 ^ (1 / 3)))
  (h_volume_eq : (4 / 3) * π * R^3 = (4 / 3) * π * r^3) : 
  R = 6 * (2 ^ (1 / 3)) :=
by
  sorry

end soap_bubble_radius_l81_81654


namespace simplify_trig_expression_l81_81920

theorem simplify_trig_expression (A : ℝ) :
  (2 - (Real.cos A / Real.sin A) + (1 / Real.sin A)) * (3 - (Real.sin A / Real.cos A) - (1 / Real.cos A)) = 
  7 * Real.sin A * Real.cos A - 2 * Real.cos A ^ 2 - 3 * Real.sin A ^ 2 - 3 * Real.cos A + Real.sin A + 1 :=
by
  sorry

end simplify_trig_expression_l81_81920


namespace constant_term_binomial_expansion_l81_81182

theorem constant_term_binomial_expansion (n : ℕ) (hn : n = 6) :
  (2 : ℤ) * (x : ℝ) - (1 : ℤ) / (2 : ℝ) / (x : ℝ) ^ n == -20 := by
  sorry

end constant_term_binomial_expansion_l81_81182


namespace total_amount_paid_after_discount_l81_81644

-- Define the given conditions
def marked_price_per_article : ℝ := 10
def discount_percentage : ℝ := 0.60
def number_of_articles : ℕ := 2

-- Proving the total amount paid
theorem total_amount_paid_after_discount : 
  (marked_price_per_article * number_of_articles) * (1 - discount_percentage) = 8 := by
  sorry

end total_amount_paid_after_discount_l81_81644


namespace product_polynomial_coeffs_l81_81345

theorem product_polynomial_coeffs
  (g h : ℚ)
  (h1 : 7 * d^2 - 3 * d + g * (3 * d^2 + h * d - 5) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) :
  g + h = -28/9 := 
  sorry

end product_polynomial_coeffs_l81_81345


namespace actual_diameter_of_tissue_l81_81863

theorem actual_diameter_of_tissue (magnification: ℝ) (magnified_diameter: ℝ) :
  magnification = 1000 ∧ magnified_diameter = 1 → magnified_diameter / magnification = 0.001 :=
by
  intro h
  sorry

end actual_diameter_of_tissue_l81_81863


namespace total_turtles_l81_81875

theorem total_turtles (G H L : ℕ) (h_G : G = 800) (h_H : H = 2 * G) (h_L : L = 3 * G) : G + H + L = 4800 :=
by
  sorry

end total_turtles_l81_81875


namespace regression_line_fits_l81_81418

variables {x y : ℝ}

def points := [(1, 2), (2, 5), (4, 7), (5, 10)]

def regression_line (x : ℝ) : ℝ := x + 3

theorem regression_line_fits :
  (∀ p ∈ points, regression_line p.1 = p.2) ∧ (regression_line 3 = 6) :=
by
  sorry

end regression_line_fits_l81_81418


namespace sum_of_numbers_in_50th_row_l81_81511

-- Defining the array and the row sum
def row_sum (n : ℕ) : ℕ :=
  2^n

-- Proposition stating that the 50th row sum is equal to 2^50
theorem sum_of_numbers_in_50th_row : row_sum 50 = 2^50 :=
by sorry

end sum_of_numbers_in_50th_row_l81_81511


namespace min_value_3x_4y_l81_81212

open Real

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y = 5 :=
by
  sorry

end min_value_3x_4y_l81_81212


namespace sector_area_l81_81446

theorem sector_area (theta : ℝ) (d : ℝ) (r : ℝ := d / 2) (circle_area : ℝ := π * r^2) 
    (sector_area : ℝ := (theta / 360) * circle_area) : 
  theta = 120 → d = 6 → sector_area = 3 * π :=
by
  intro htheta hd
  sorry

end sector_area_l81_81446


namespace symmetric_point_y_axis_l81_81717

theorem symmetric_point_y_axis (x y : ℝ) (hx : x = -3) (hy : y = 2) :
  (-x, y) = (3, 2) :=
by
  sorry

end symmetric_point_y_axis_l81_81717


namespace total_canvas_area_l81_81991

theorem total_canvas_area (rect_length rect_width tri1_base tri1_height tri2_base tri2_height : ℕ)
    (h1 : rect_length = 5) (h2 : rect_width = 8)
    (h3 : tri1_base = 3) (h4 : tri1_height = 4)
    (h5 : tri2_base = 4) (h6 : tri2_height = 6) :
    (rect_length * rect_width) + ((tri1_base * tri1_height) / 2) + ((tri2_base * tri2_height) / 2) = 58 := by
  sorry

end total_canvas_area_l81_81991


namespace min_value_of_expr_l81_81300

theorem min_value_of_expr (x : ℝ) (h : x > 2) : ∃ y, (y = x + 4 / (x - 2)) ∧ y ≥ 6 :=
by
  sorry

end min_value_of_expr_l81_81300


namespace second_person_percentage_of_Deshaun_l81_81218

variable (days : ℕ) (books_read_by_Deshaun : ℕ) (pages_per_book : ℕ) (pages_per_day_by_second_person : ℕ)

theorem second_person_percentage_of_Deshaun :
  days = 80 →
  books_read_by_Deshaun = 60 →
  pages_per_book = 320 →
  pages_per_day_by_second_person = 180 →
  ((pages_per_day_by_second_person * days) / (books_read_by_Deshaun * pages_per_book) * 100) = 75 := 
by
  intros days_eq books_eq pages_eq second_pages_eq
  rw [days_eq, books_eq, pages_eq, second_pages_eq]
  simp
  sorry

end second_person_percentage_of_Deshaun_l81_81218


namespace reflected_ray_eqn_l81_81205

theorem reflected_ray_eqn (P : ℝ × ℝ)
  (incident_ray : ∀ x : ℝ, P.2 = 2 * P.1 + 1)
  (reflection_line : P.2 = P.1) :
  P.1 - 2 * P.2 - 1 = 0 :=
sorry

end reflected_ray_eqn_l81_81205


namespace lcm_first_ten_numbers_l81_81330

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l81_81330


namespace kittens_count_l81_81244

def cats_taken_in : ℕ := 12
def cats_initial : ℕ := cats_taken_in / 2
def cats_post_adoption : ℕ := cats_taken_in + cats_initial - 3
def cats_now : ℕ := 19

theorem kittens_count :
  ∃ k : ℕ, cats_post_adoption + k - 1 = cats_now :=
by
  use 5
  sorry

end kittens_count_l81_81244


namespace matrix_pow_2020_l81_81802

-- Define the matrix type and basic multiplication rule
def M : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![3, 1]]

theorem matrix_pow_2020 :
  M ^ 2020 = ![![1, 0], ![6060, 1]] := by
  sorry

end matrix_pow_2020_l81_81802


namespace earnings_ratio_l81_81992

-- Definitions for conditions
def jerusha_earnings : ℕ := 68
def total_earnings : ℕ := 85
def lottie_earnings : ℕ := total_earnings - jerusha_earnings

-- Prove that the ratio of Jerusha's earnings to Lottie's earnings is 4:1
theorem earnings_ratio : 
  ∃ (k : ℕ), jerusha_earnings = k * lottie_earnings ∧ (jerusha_earnings + lottie_earnings = total_earnings) ∧ (jerusha_earnings = 68) ∧ (total_earnings = 85) →
  68 / (total_earnings - 68) = 4 := 
by
  sorry

end earnings_ratio_l81_81992


namespace people_came_later_l81_81576

theorem people_came_later (lollipop_ratio initial_people lollipops : ℕ) 
  (h1 : lollipop_ratio = 5) 
  (h2 : initial_people = 45) 
  (h3 : lollipops = 12) : 
  (lollipops * lollipop_ratio - initial_people) = 15 := by 
  sorry

end people_came_later_l81_81576


namespace integer_ratio_condition_l81_81972

variable (x y : ℝ)

theorem integer_ratio_condition 
  (h : 3 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 6)
  (h_int : ∃ t : ℤ, x = t * y) :
  ∃ t : ℤ, t = -2 :=
by
  sorry

end integer_ratio_condition_l81_81972


namespace octal_sum_l81_81491

open Nat

def octal_to_decimal (oct : ℕ) : ℕ :=
  match oct with
  | 0 => 0
  | n => let d3 := (n / 100) % 10
         let d2 := (n / 10) % 10
         let d1 := n % 10
         d3 * 8^2 + d2 * 8^1 + d1 * 8^0

def decimal_to_octal (dec : ℕ) : ℕ :=
  let rec aux (n : ℕ) (mul : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 8) (mul * 10) (acc + (n % 8) * mul)
  aux dec 1 0

theorem octal_sum :
  let a := 451
  let b := 167
  octal_to_decimal 451 + octal_to_decimal 167 = octal_to_decimal 640 := sorry

end octal_sum_l81_81491


namespace square_floor_tile_count_l81_81279

/-
A square floor is tiled with congruent square tiles.
The tiles on the two diagonals of the floor are black.
If there are 101 black tiles, then the total number of tiles is 2601.
-/
theorem square_floor_tile_count  
  (s : ℕ) 
  (hs_odd : s % 2 = 1)  -- s is odd
  (h_black_tile_count : 2 * s - 1 = 101) 
  : s^2 = 2601 := 
by 
  sorry

end square_floor_tile_count_l81_81279


namespace find_time_period_l81_81114

theorem find_time_period (P r CI : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (hP : P = 10000)
  (hr : r = 0.15)
  (hCI : CI = 3886.25)
  (hn : n = 1)
  (hA : A = P + CI)
  (h_formula : A = P * (1 + r / n) ^ (n * t)) : 
  t = 2 := 
  sorry

end find_time_period_l81_81114


namespace differential_equation_solution_l81_81908

theorem differential_equation_solution (x y : ℝ) (C : ℝ) :
  (∀ dx dy, 2 * x * y * dx + x^2 * dy = 0) → x^2 * y = C :=
sorry

end differential_equation_solution_l81_81908


namespace unit_prices_l81_81063

theorem unit_prices (x y : ℕ) (h1 : 5 * x + 4 * y = 139) (h2 : 4 * x + 5 * y = 140) :
  x = 15 ∧ y = 16 :=
by
  -- Proof will go here
  sorry

end unit_prices_l81_81063


namespace fraction_of_work_completed_l81_81755

-- Definitions
def work_rate_x : ℚ := 1 / 14
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 25

-- Given the combined work rate and time
def combined_work_rate : ℚ := work_rate_x + work_rate_y + work_rate_z
def time_worked : ℚ := 5

-- The fraction of work completed
def fraction_work_completed : ℚ := combined_work_rate * time_worked

-- Statement to prove
theorem fraction_of_work_completed : fraction_work_completed = 113 / 140 := by
  sorry

end fraction_of_work_completed_l81_81755


namespace max_xy_l81_81351

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions given in the problem
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom eq1 : x + 1/y = 3
axiom eq2 : y + 2/x = 3

theorem max_xy : ∃ (xy : ℝ), 
  xy = x * y ∧ xy = 3 + Real.sqrt 7 := sorry

end max_xy_l81_81351


namespace am_gm_inequality_l81_81656

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem am_gm_inequality : (a / b) + (b / c) + (c / a) ≥ 3 := by
  sorry

end am_gm_inequality_l81_81656


namespace function_characterization_l81_81077

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization (f : ℝ → ℝ) (k : ℝ) :
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) →
  (∀ x : ℝ, |f x - k * x| ≤ |x^2 - x|) →
  ∀ x : ℝ, f x = k * x :=
by
  sorry

end function_characterization_l81_81077


namespace base_8_subtraction_l81_81876

def subtract_in_base_8 (a b : ℕ) : ℕ := 
  -- Implementing the base 8 subtraction
  sorry

theorem base_8_subtraction : subtract_in_base_8 0o652 0o274 = 0o356 :=
by 
  -- Faking the proof to ensure it can compile.
  sorry

end base_8_subtraction_l81_81876


namespace natasha_average_speed_l81_81737

theorem natasha_average_speed :
  (4 * 2.625 * 2) / (4 + 2) = 3.5 := 
by
  sorry

end natasha_average_speed_l81_81737


namespace total_squares_after_removals_l81_81207

/-- 
Prove that the total number of squares of various sizes on a 5x5 grid,
after removing two 1x1 squares, is 55.
-/
theorem total_squares_after_removals (total_squares_in_5x5_grid: ℕ) (removed_squares: ℕ) : 
  (total_squares_in_5x5_grid = 25 + 16 + 9 + 4 + 1) →
  (removed_squares = 2) →
  (total_squares_in_5x5_grid - removed_squares = 55) :=
sorry

end total_squares_after_removals_l81_81207


namespace diapers_per_pack_l81_81414

def total_boxes := 30
def packs_per_box := 40
def price_per_diaper := 5
def total_revenue := 960000

def total_packs_per_week := total_boxes * packs_per_box
def total_diapers_sold := total_revenue / price_per_diaper

theorem diapers_per_pack :
  total_diapers_sold / total_packs_per_week = 160 :=
by
  -- Placeholder for the actual proof
  sorry

end diapers_per_pack_l81_81414


namespace only_valid_pairs_l81_81769

theorem only_valid_pairs (a b : ℕ) (h₁ : a ≥ 1) (h₂ : b ≥ 1) :
  a^b^2 = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) :=
by
  sorry

end only_valid_pairs_l81_81769


namespace speed_of_current_l81_81325

theorem speed_of_current (v_w v_c : ℝ) (h_downstream : 125 = (v_w + v_c) * 10)
                         (h_upstream : 60 = (v_w - v_c) * 10) :
  v_c = 3.25 :=
by {
  sorry
}

end speed_of_current_l81_81325


namespace vanessa_scored_27_points_l81_81934

variable (P : ℕ) (number_of_players : ℕ) (average_points_per_player : ℚ) (vanessa_points : ℕ)

axiom team_total_points : P = 48
axiom other_players : number_of_players = 6
axiom average_points_per_other_player : average_points_per_player = 3.5

theorem vanessa_scored_27_points 
  (h1 : P = 48)
  (h2 : number_of_players = 6)
  (h3 : average_points_per_player = 3.5)
: vanessa_points = 27 :=
sorry

end vanessa_scored_27_points_l81_81934


namespace geometric_sequence_value_l81_81909

theorem geometric_sequence_value 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_diff : d ≠ 0)
  (h_condition : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom_seq : ∀ n, b (n + 1) = b n * (b 1 / b 0))
  (h_b7_eq_a7 : b 7 = a 7) :
  b 6 * b 8 = 16 :=
sorry

end geometric_sequence_value_l81_81909


namespace same_function_C_l81_81777

theorem same_function_C (x : ℝ) (hx : x ≠ 0) : (x^0 = 1) ∧ ((1 / x^0) = 1) :=
by
  -- Definition for domain exclusion
  have h1 : x ^ 0 = 1 := by 
    sorry -- proof skipped
  have h2 : 1 / x ^ 0 = 1 := by 
    sorry -- proof skipped
  exact ⟨h1, h2⟩

end same_function_C_l81_81777


namespace water_tank_capacity_l81_81839

theorem water_tank_capacity (rate : ℝ) (time : ℝ) (fraction : ℝ) (capacity : ℝ) : 
(rate = 10) → (time = 300) → (fraction = 3/4) → 
(rate * time = fraction * capacity) → 
capacity = 4000 := 
by
  intros h_rate h_time h_fraction h_equation
  rw [h_rate, h_time, h_fraction] at h_equation
  linarith

end water_tank_capacity_l81_81839


namespace find_x_range_l81_81906

noncomputable def p (x : ℝ) := x^2 + 2*x - 3 > 0
noncomputable def q (x : ℝ) := 1/(3 - x) > 1

theorem find_x_range (x : ℝ) : (¬q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  intro h
  sorry

end find_x_range_l81_81906


namespace red_balloon_is_one_l81_81433

open Nat

theorem red_balloon_is_one (R B : Nat) (h1 : R + B = 85) (h2 : R ≥ 1) (h3 : ∀ i j, i < R → j < R → i ≠ j → (i < B ∨ j < B)) : R = 1 :=
by
  sorry

end red_balloon_is_one_l81_81433


namespace sum_of_n_values_l81_81726

theorem sum_of_n_values (n_values : List ℤ) 
  (h : ∀ n ∈ n_values, ∃ k : ℤ, 24 = k * (2 * n - 1)) : n_values.sum = 2 :=
by
  -- Proof to be provided.
  sorry

end sum_of_n_values_l81_81726


namespace handrail_length_nearest_tenth_l81_81124

noncomputable def handrail_length (rise : ℝ) (turn_degree : ℝ) (radius : ℝ) : ℝ :=
  let arc_length := (turn_degree / 360) * (2 * Real.pi * radius)
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_nearest_tenth
  (h_rise : rise = 12)
  (h_turn_degree : turn_degree = 180)
  (h_radius : radius = 3) : handrail_length rise turn_degree radius = 13.1 :=
  by
  sorry

end handrail_length_nearest_tenth_l81_81124


namespace range_of_m_l81_81794

variable {x m : ℝ}
variable (q: ℝ → Prop) (p: ℝ → Prop)

-- Definition of q
def q_cond : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

-- Definition of p
def p_cond : Prop := |1 - (x - 1) / 3| ≤ 2

-- Statement of the proof problem
theorem range_of_m (h1 : ∀ x, q x → p x) (h2 : ∃ x, ¬p x → q x) 
  (h3 : m > 0) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l81_81794


namespace sum_of_mapped_elements_is_ten_l81_81805

theorem sum_of_mapped_elements_is_ten (a b : ℝ) (h1 : a = 1) (h2 : b = 9) : a + b = 10 := by
  sorry

end sum_of_mapped_elements_is_ten_l81_81805


namespace find_b_l81_81013

noncomputable def angle_B : ℝ := 60
noncomputable def c : ℝ := 8
noncomputable def diff_b_a (b a : ℝ) : Prop := b - a = 4

theorem find_b (b a : ℝ) (h₁ : angle_B = 60) (h₂ : c = 8) (h₃ : diff_b_a b a) :
  b = 7 :=
sorry

end find_b_l81_81013


namespace ship_passengers_round_trip_tickets_l81_81889

theorem ship_passengers_round_trip_tickets (total_passengers : ℕ) (p1 : ℝ) (p2 : ℝ) :
  (p1 = 0.25 * total_passengers) ∧ (p2 = 0.6 * (p * total_passengers)) →
  (p * total_passengers = 62.5 / 100 * total_passengers) :=
by
  sorry

end ship_passengers_round_trip_tickets_l81_81889


namespace number_of_times_difference_fits_is_20_l81_81977

-- Definitions for Ralph's pictures
def ralph_wild_animals := 75
def ralph_landscapes := 36
def ralph_family_events := 45
def ralph_cars := 20
def ralph_total_pictures := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars

-- Definitions for Derrick's pictures
def derrick_wild_animals := 95
def derrick_landscapes := 42
def derrick_family_events := 55
def derrick_cars := 25
def derrick_airplanes := 10
def derrick_total_pictures := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Combined total number of pictures
def combined_total_pictures := ralph_total_pictures + derrick_total_pictures

-- Difference in wild animals pictures
def difference_wild_animals := derrick_wild_animals - ralph_wild_animals

-- Number of times the difference fits into the combined total (rounded down)
def times_difference_fits := combined_total_pictures / difference_wild_animals

-- Statement of the problem
theorem number_of_times_difference_fits_is_20 : times_difference_fits = 20 := by
  -- The proof will be written here
  sorry

end number_of_times_difference_fits_is_20_l81_81977


namespace unique_x_condition_l81_81245

theorem unique_x_condition (x : ℝ) : 
  (1 ≤ x ∧ x < 2) ∧ (∀ n : ℕ, 0 < n → (⌊2^n * x⌋ % 4 = 1 ∨ ⌊2^n * x⌋ % 4 = 2)) ↔ x = 4/3 := 
by 
  sorry

end unique_x_condition_l81_81245


namespace track_circumference_l81_81121

theorem track_circumference (A_speed B_speed : ℝ) (y : ℝ) (c : ℝ)
  (A_initial B_initial : ℝ := 0)
  (B_meeting_distance_A_first_meeting : ℝ := 150)
  (A_meeting_distance_B_second_meeting : ℝ := y - 150)
  (A_second_distance : ℝ := 2 * y - 90)
  (B_second_distance : ℝ := y + 90) 
  (first_meeting_eq : B_meeting_distance_A_first_meeting = 150)
  (second_meeting_eq : A_second_distance + 90 = 2 * y)
  (uniform_speed : A_speed / B_speed = (y + 90)/(2 * y - 90)) :
  c = 2 * y → c = 720 :=
by
  sorry

end track_circumference_l81_81121


namespace exponent_of_4_l81_81878

theorem exponent_of_4 (x : ℕ) (h₁ : (1 / 4 : ℚ) ^ 2 = 1 / 16) (h₂ : 16384 * (1 / 16 : ℚ) = 1024) :
  4 ^ x = 1024 → x = 5 :=
by
  sorry

end exponent_of_4_l81_81878


namespace baby_panda_daily_bamboo_intake_l81_81197

theorem baby_panda_daily_bamboo_intake :
  ∀ (adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week : ℕ),
    adult_bamboo_per_day = 138 →
    total_bamboo_per_week = 1316 →
    total_bamboo_per_week = 7 * adult_bamboo_per_day + 7 * baby_bamboo_per_day →
    baby_bamboo_per_day = 50 :=
by
  intros adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week h1 h2 h3
  sorry

end baby_panda_daily_bamboo_intake_l81_81197


namespace problem_statement_l81_81502

noncomputable def roots (a b : ℝ) (coef1 coef2 : ℝ) :=
  ∃ x : ℝ, (x = a ∨ x = b) ∧ x^2 + coef1 * x + coef2 = 0

theorem problem_statement
  (a b c d : ℝ)
  (h1 : a + b = -57)
  (h2 : a * b = 1)
  (h3 : c + d = 57)
  (h4 : c * d = 1) :
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := 
by
  sorry

end problem_statement_l81_81502


namespace acute_angle_89_l81_81960

def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

theorem acute_angle_89 :
  is_acute_angle 89 :=
by {
  -- proof details would go here, since only the statement is required
  sorry
}

end acute_angle_89_l81_81960


namespace calc_x_squared_y_squared_l81_81552

theorem calc_x_squared_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -9) : x^2 + y^2 = 22 := by
  sorry

end calc_x_squared_y_squared_l81_81552


namespace outfits_count_l81_81568

-- Definitions of the counts of each type of clothing item
def num_blue_shirts : Nat := 6
def num_green_shirts : Nat := 4
def num_pants : Nat := 7
def num_blue_hats : Nat := 9
def num_green_hats : Nat := 7

-- Statement of the problem to prove
theorem outfits_count :
  (num_blue_shirts * num_pants * num_green_hats) + (num_green_shirts * num_pants * num_blue_hats) = 546 :=
by
  sorry

end outfits_count_l81_81568


namespace sqrt2_minus1_mul_sqrt2_plus1_eq1_l81_81285

theorem sqrt2_minus1_mul_sqrt2_plus1_eq1 : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 :=
  sorry

end sqrt2_minus1_mul_sqrt2_plus1_eq1_l81_81285


namespace problem1_problem2_l81_81603

-- Define the conditions and the target proofs based on identified questions and answers

-- Problem 1
theorem problem1 (x : ℚ) : 
  9 * (x - 2)^2 ≤ 25 ↔ x = 11 / 3 ∨ x = 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (x y : ℚ) :
  (x + 1) / 3 = 2 * y ∧ 2 * (x + 1) - y = 11 ↔ x = 5 ∧ y = 1 :=
sorry

end problem1_problem2_l81_81603


namespace sum_reciprocal_squares_l81_81598

open Real

theorem sum_reciprocal_squares (a : ℝ) (A B C D E F : ℝ)
    (square_ABCD : A = 0 ∧ B = a ∧ D = a ∧ C = a)
    (line_intersects : A = 0 ∧ E ≥ 0 ∧ E ≤ a ∧ F ≥ 0 ∧ F ≤ a) 
    (phi : ℝ) : 
    (cos phi * (a/cos phi))^2 + (sin phi * (a/sin phi))^2 = (1/a^2) := 
sorry 

end sum_reciprocal_squares_l81_81598


namespace maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l81_81916

def pine_tree_height : ℚ := 37 / 4  -- 9 1/4 feet
def maple_tree_height : ℚ := 62 / 4  -- 15 1/2 feet (converted directly to common denominator)
def growth_rate : ℚ := 7 / 4  -- 1 3/4 feet per year

theorem maple_tree_taller_than_pine_tree : maple_tree_height - pine_tree_height = 25 / 4 := 
by sorry

theorem pine_tree_height_in_one_year : pine_tree_height + growth_rate = 44 / 4 := 
by sorry

end maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l81_81916


namespace max_wickets_bowler_can_take_l81_81474

noncomputable def max_wickets_per_over : ℕ := 3
noncomputable def overs_bowled : ℕ := 6
noncomputable def max_possible_wickets := max_wickets_per_over * overs_bowled

theorem max_wickets_bowler_can_take : max_possible_wickets = 18 → max_possible_wickets == 10 :=
by
  sorry

end max_wickets_bowler_can_take_l81_81474


namespace Marissa_sunflower_height_l81_81038

-- Define the necessary conditions
def sister_height_feet : ℕ := 4
def sister_height_inches : ℕ := 3
def extra_sunflower_height : ℕ := 21
def inches_per_foot : ℕ := 12

-- Calculate the total height of the sister in inches
def sister_total_height_inch : ℕ := (sister_height_feet * inches_per_foot) + sister_height_inches

-- Calculate the sunflower height in inches
def sunflower_height_inch : ℕ := sister_total_height_inch + extra_sunflower_height

-- Convert the sunflower height to feet
def sunflower_height_feet : ℕ := sunflower_height_inch / inches_per_foot

-- The theorem we want to prove
theorem Marissa_sunflower_height : sunflower_height_feet = 6 := by
  sorry

end Marissa_sunflower_height_l81_81038


namespace angle_between_generatrix_and_base_of_cone_l81_81228

theorem angle_between_generatrix_and_base_of_cone (r R H : ℝ) (α : ℝ)
  (h_cylinder_height : H = 2 * R)
  (h_total_surface_area : 2 * Real.pi * r * H + 2 * Real.pi * r^2 = Real.pi * R^2) :
  α = Real.arctan (2 * (4 + Real.sqrt 6) / 5) :=
sorry

end angle_between_generatrix_and_base_of_cone_l81_81228


namespace find_number_l81_81870

variable (x : ℕ)
variable (result : ℕ)

theorem find_number (h : x * 9999 = 4690640889) : x = 469131 :=
by
  sorry

end find_number_l81_81870


namespace incident_ray_slope_in_circle_problem_l81_81858

noncomputable def slope_of_incident_ray : ℚ := sorry

theorem incident_ray_slope_in_circle_problem :
  ∃ (P : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ),
  P = (-1, -3) ∧
  C = (2, -1) ∧
  (D = (C.1, -C.2)) ∧
  (D = (2, 1)) ∧
  ∀ (m : ℚ), (m = (D.2 - P.2) / (D.1 - P.1)) → m = 4 / 3 := 
sorry

end incident_ray_slope_in_circle_problem_l81_81858


namespace fraction_sum_is_integer_l81_81732

theorem fraction_sum_is_integer (n : ℤ) : 
  ∃ k : ℤ, (n / 3 + (n^2) / 2 + (n^3) / 6) = k := 
sorry

end fraction_sum_is_integer_l81_81732


namespace odd_function_at_zero_l81_81194

theorem odd_function_at_zero
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  f 0 = 0 :=
by
  sorry

end odd_function_at_zero_l81_81194


namespace problem_1_1_eval_l81_81298

noncomputable def E (a b c : ℝ) : ℝ :=
  let A := (1/a - 1/(b+c))/(1/a + 1/(b+c))
  let B := 1 + (b^2 + c^2 - a^2)/(2*b*c)
  let C := (a - b - c)/(a * b * c)
  (A * B) / C

theorem problem_1_1_eval :
  E 0.02 (-11.05) 1.07 = 0.1 :=
by
  -- Proof goes here
  sorry

end problem_1_1_eval_l81_81298


namespace intersection_complement_eq_singleton_l81_81763

def U : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) }
def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ (y - 3) / (x - 2) = 1 }
def N : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1 }
def complement_U (M : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := { p | p ∈ U ∧ p ∉ M }

theorem intersection_complement_eq_singleton :
  N ∩ complement_U M = {(2,3)} :=
by
  sorry

end intersection_complement_eq_singleton_l81_81763


namespace polar_to_cartesian_l81_81377

theorem polar_to_cartesian (ρ θ x y : ℝ) (h1 : ρ = 2 * Real.sin θ)
  (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end polar_to_cartesian_l81_81377


namespace eliot_account_balance_l81_81962

variable (A E F : ℝ)

theorem eliot_account_balance
  (h1 : A > E)
  (h2 : F > A)
  (h3 : A - E = (1 : ℝ) / 12 * (A + E))
  (h4 : F - A = (1 : ℝ) / 8 * (F + A))
  (h5 : 1.1 * A = 1.2 * E + 21)
  (h6 : 1.05 * F = 1.1 * A + 40) :
  E = 210 := 
sorry

end eliot_account_balance_l81_81962


namespace chess_tournament_proof_l81_81936

-- Define the conditions
variables (i g n I G : ℕ)
variables (VI VG VD : ℕ)

-- Condition 1: The number of GMs is ten times the number of IMs
def condition1 : Prop := g = 10 * i
  
-- Condition 2: The sum of the points of all GMs is 4.5 times the sum of the points of all IMs
def condition2 : Prop := G = 5 * I + I / 2

-- Condition 3: The total number of players is the sum of IMs and GMs
def condition3 : Prop := n = i + g

-- Condition 4: Each player played only once against all other opponents
def condition4 : Prop := n * (n - 1) = 2 * (VI + VG + VD)

-- Condition 5: The sum of the points of all games is 5.5 times the sum of the points of all IMs
def condition5 : Prop := I + G = 11 * I / 2

-- Condition 6: Total games played
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The questions to be proven given the conditions
theorem chess_tournament_proof:
  condition1 i g →
  condition2 I G →
  condition3 i g n →
  condition4 n VI VG VD →
  condition5 I G →
  i = 1 ∧ g = 10 ∧ total_games n = 55 :=
by
  -- The proof is left as an exercise
  sorry

end chess_tournament_proof_l81_81936


namespace park_area_l81_81225

-- Definitions for the conditions
def length (breadth : ℕ) : ℕ := 4 * breadth
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the proof problem
theorem park_area (breadth : ℕ) (h1 : perimeter (length breadth) breadth = 1600) : 
  let len := length breadth
  len * breadth = 102400 := 
by 
  sorry

end park_area_l81_81225


namespace greatest_positive_integer_difference_l81_81749

-- Define the conditions
def condition_x (x : ℝ) : Prop := 4 < x ∧ x < 6
def condition_y (y : ℝ) : Prop := 6 < y ∧ y < 10

-- Define the problem statement
theorem greatest_positive_integer_difference (x y : ℕ) (hx : condition_x x) (hy : condition_y y) : y - x = 4 :=
sorry

end greatest_positive_integer_difference_l81_81749


namespace min_value_m2n_mn_l81_81780

theorem min_value_m2n_mn (m n : ℝ) 
  (h1 : (x - m)^2 + (y - n)^2 = 9)
  (h2 : x + 2 * y + 2 = 0)
  (h3 : 0 < m)
  (h4 : 0 < n)
  (h5 : m + 2 * n + 2 = 5)
  (h6 : ∃ l : ℝ, l = 4 ): (m + 2 * n) / (m * n) = 8/3 :=
by
  sorry

end min_value_m2n_mn_l81_81780


namespace condition_holds_iff_b_eq_10_l81_81191

-- Define xn based on given conditions in the problem
def x_n (b : ℕ) (n : ℕ) : ℕ :=
  if b > 5 then
    b^(2*n) + b^n + 3*b - 5
  else
    0

-- State the main theorem to be proven in Lean
theorem condition_holds_iff_b_eq_10 :
  ∀ (b : ℕ), (b > 5) ↔ ∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2 := sorry

end condition_holds_iff_b_eq_10_l81_81191


namespace frames_per_page_l81_81069

theorem frames_per_page (total_frames : ℕ) (total_pages : ℝ) (h1 : total_frames = 1573) (h2 : total_pages = 11.0) : total_frames / total_pages = 143 := by
  sorry

end frames_per_page_l81_81069


namespace angles_congruence_mod_360_l81_81606

theorem angles_congruence_mod_360 (a b c d : ℤ) : 
  (a = 30) → (b = -30) → (c = 630) → (d = -630) →
  (b % 360 = 330 % 360) ∧ 
  (a % 360 ≠ 330 % 360) ∧ (c % 360 ≠ 330 % 360) ∧ (d % 360 ≠ 330 % 360) :=
by
  intros
  sorry

end angles_congruence_mod_360_l81_81606


namespace build_time_40_workers_l81_81935

theorem build_time_40_workers (r : ℝ) : 
  (60 * r) * 5 = 1 → (40 * r) * t = 1 → t = 7.5 :=
by
  intros h1 h2
  sorry

end build_time_40_workers_l81_81935


namespace fourth_square_state_l81_81810

inductive Shape
| Circle
| Triangle
| LineSegment
| Square

inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

structure SquareState where
  circle : Position
  triangle : Position
  line_segment_parallel_to : Bool -- True = Top & Bottom; False = Left & Right
  square : Position

def move_counterclockwise : Position → Position
| Position.TopLeft => Position.BottomLeft
| Position.BottomLeft => Position.BottomRight
| Position.BottomRight => Position.TopRight
| Position.TopRight => Position.TopLeft

def update_square_states (s1 s2 s3 : SquareState) : Prop :=
  move_counterclockwise s1.circle = s2.circle ∧
  move_counterclockwise s2.circle = s3.circle ∧
  move_counterclockwise s1.triangle = s2.triangle ∧
  move_counterclockwise s2.triangle = s3.triangle ∧
  s1.line_segment_parallel_to = !s2.line_segment_parallel_to ∧
  s2.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  move_counterclockwise s1.square = s2.square ∧
  move_counterclockwise s2.square = s3.square

theorem fourth_square_state (s1 s2 s3 s4 : SquareState) (h : update_square_states s1 s2 s3) :
  s4.circle = move_counterclockwise s3.circle ∧
  s4.triangle = move_counterclockwise s3.triangle ∧
  s4.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  s4.square = move_counterclockwise s3.square :=
sorry

end fourth_square_state_l81_81810


namespace find_length_of_first_train_l81_81820

noncomputable def length_of_first_train (speed_train1 speed_train2 : ℕ) (time_to_cross : ℕ) (length_train2 : ℚ) : ℚ :=
  let relative_speed := (speed_train1 + speed_train2) * 1000 / 3600
  let combined_length := relative_speed * time_to_cross
  combined_length - length_train2

theorem find_length_of_first_train :
  length_of_first_train 120 80 9 280.04 = 220 := sorry

end find_length_of_first_train_l81_81820


namespace arithmetic_sequence_l81_81198

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 2 + a 3 = 32) 
  (h2 : a 11 + a 12 + a 13 = 118) 
  (arith_seq : ∀ n, a (n + 1) = a n + d) : 
  a 4 + a 10 = 50 :=
by 
  sorry

end arithmetic_sequence_l81_81198


namespace standard_deviation_of_applicants_ages_l81_81807

noncomputable def average_age : ℝ := 30
noncomputable def max_different_ages : ℝ := 15

theorem standard_deviation_of_applicants_ages 
  (σ : ℝ)
  (h : max_different_ages = 2 * σ) 
  : σ = 7.5 :=
by
  sorry

end standard_deviation_of_applicants_ages_l81_81807


namespace cody_paid_amount_l81_81706

/-- Cody buys $40 worth of stuff,
    the tax rate is 5%,
    he receives an $8 discount after taxes,
    and he and his friend split the final price equally.
    Prove that Cody paid $17. -/
theorem cody_paid_amount
  (initial_cost : ℝ)
  (tax_rate : ℝ)
  (discount : ℝ)
  (final_split : ℝ)
  (H1 : initial_cost = 40)
  (H2 : tax_rate = 0.05)
  (H3 : discount = 8)
  (H4 : final_split = 2) :
  (initial_cost * (1 + tax_rate) - discount) / final_split = 17 :=
by
  sorry

end cody_paid_amount_l81_81706


namespace prob_all_meet_standard_prob_at_least_one_meets_standard_l81_81535

def P_meeting_standard_A := 0.8
def P_meeting_standard_B := 0.6
def P_meeting_standard_C := 0.5

theorem prob_all_meet_standard :
  (P_meeting_standard_A * P_meeting_standard_B * P_meeting_standard_C) = 0.24 :=
by
  sorry

theorem prob_at_least_one_meets_standard :
  (1 - ((1 - P_meeting_standard_A) * (1 - P_meeting_standard_B) * (1 - P_meeting_standard_C))) = 0.96 :=
by
  sorry

end prob_all_meet_standard_prob_at_least_one_meets_standard_l81_81535


namespace bisection_interval_l81_81711

def f(x : ℝ) := x^3 - 2 * x - 5

theorem bisection_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 > 0 →
  ∃ a b : ℝ, a = 2 ∧ b = 2.5 ∧ f a * f b ≤ 0 :=
by
  sorry

end bisection_interval_l81_81711


namespace hank_newspaper_reading_time_l81_81052

theorem hank_newspaper_reading_time
  (n_days_weekday : ℕ := 5)
  (novel_reading_time_weekday : ℕ := 60)
  (n_days_weekend : ℕ := 2)
  (total_weekly_reading_time : ℕ := 810)
  (x : ℕ)
  (h1 : n_days_weekday * x + n_days_weekday * novel_reading_time_weekday +
        n_days_weekend * 2 * x + n_days_weekend * 2 * novel_reading_time_weekday = total_weekly_reading_time) :
  x = 30 := 
by {
  sorry -- Proof would go here
}

end hank_newspaper_reading_time_l81_81052


namespace values_satisfying_ggx_eq_gx_l81_81302

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem values_satisfying_ggx_eq_gx (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 1 ∨ x = 3 ∨ x = 4 :=
by
  -- The proof is omitted
  sorry

end values_satisfying_ggx_eq_gx_l81_81302


namespace students_arrangement_count_l81_81140

theorem students_arrangement_count : 
  let total_permutations := Nat.factorial 5
  let a_first_permutations := Nat.factorial 4
  let b_last_permutations := Nat.factorial 4
  let both_permutations := Nat.factorial 3
  total_permutations - a_first_permutations - b_last_permutations + both_permutations = 78 :=
by
  sorry

end students_arrangement_count_l81_81140


namespace find_a_plus_b_eq_102_l81_81983

theorem find_a_plus_b_eq_102 :
  ∃ (a b : ℕ), (1600^(1 / 2) - 24 = (a^(1 / 2) - b)^2) ∧ (a + b = 102) :=
by {
  sorry
}

end find_a_plus_b_eq_102_l81_81983


namespace total_time_preparing_games_l81_81847

def time_A_game : ℕ := 15
def time_B_game : ℕ := 25
def time_C_game : ℕ := 30
def num_each_type : ℕ := 5

theorem total_time_preparing_games : 
  (num_each_type * time_A_game + num_each_type * time_B_game + num_each_type * time_C_game) = 350 := 
  by sorry

end total_time_preparing_games_l81_81847


namespace intersection_of_sets_l81_81256

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets :
  setA ∩ setB = { z : ℝ | z ∈ [-1, 1] } :=
sorry

end intersection_of_sets_l81_81256


namespace sphere_radius_five_times_surface_area_l81_81281

theorem sphere_radius_five_times_surface_area (R : ℝ) (h₁ : (4 * π * R^3 / 3) = 5 * (4 * π * R^2)) : R = 15 :=
sorry

end sphere_radius_five_times_surface_area_l81_81281


namespace division_sum_l81_81818

theorem division_sum (quotient divisor remainder : ℕ) (hquot : quotient = 65) (hdiv : divisor = 24) (hrem : remainder = 5) : 
  (divisor * quotient + remainder) = 1565 := by 
  sorry

end division_sum_l81_81818


namespace darts_game_score_l81_81354

variable (S1 S2 S3 : ℕ)
variable (n : ℕ)

theorem darts_game_score :
  n = 8 →
  S2 = 2 * S1 →
  S3 = (3 * S1) →
  S2 = 48 :=
by
  intros h1 h2 h3
  sorry

end darts_game_score_l81_81354


namespace shortest_player_height_l81_81499

-- let h_tall be the height of the tallest player
-- let h_short be the height of the shortest player
-- let diff be the height difference between the tallest and the shortest player

variable (h_tall h_short diff : ℝ)

-- conditions given in the problem
axiom tall_player_height : h_tall = 77.75
axiom height_difference : diff = 9.5
axiom height_relationship : h_tall = h_short + diff

-- the statement we need to prove
theorem shortest_player_height : h_short = 68.25 := by
  sorry

end shortest_player_height_l81_81499


namespace no_injective_function_l81_81393

theorem no_injective_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m * n) = f m + f n) : ¬ Function.Injective f := 
sorry

end no_injective_function_l81_81393


namespace smallest_discount_n_l81_81634

noncomputable def effective_discount_1 (x : ℝ) : ℝ := 0.64 * x
noncomputable def effective_discount_2 (x : ℝ) : ℝ := 0.614125 * x
noncomputable def effective_discount_3 (x : ℝ) : ℝ := 0.63 * x 

theorem smallest_discount_n (x : ℝ) (n : ℕ) (hx : x > 0) :
  (1 - n / 100 : ℝ) * x < effective_discount_1 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_2 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_3 x ↔ n = 39 := 
sorry

end smallest_discount_n_l81_81634


namespace area_of_triangle_LEF_l81_81884

noncomputable
def radius : ℝ := 10
def chord_length : ℝ := 10
def diameter_parallel_chord : Prop := True -- this condition ensures EF is parallel to LM
def LZ_length : ℝ := 20
def collinear_points : Prop := True -- this condition ensures L, M, O, Z are collinear

theorem area_of_triangle_LEF : 
  radius = 10 ∧
  chord_length = 10 ∧
  diameter_parallel_chord ∧
  LZ_length = 20 ∧ 
  collinear_points →
  (∃ area : ℝ, area = 50 * Real.sqrt 3) :=
by
  sorry

end area_of_triangle_LEF_l81_81884


namespace find_values_of_a2_b2_l81_81915

-- Define the conditions
variables {a b : ℝ}
variable (h1 : a > b)
variable (h2 : b > 0)
variable (hP : (-2, (Real.sqrt 14) / 2) ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 })
variable (hCircle : ∀ Q : ℝ × ℝ, (Q ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 2 }) → (∃ tA tB : ℝ × ℝ, (tA ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tB ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tA = - tB ∨ tB = - tA) ∧ ((tA.1 + tB.1)/2 = (-2 + tA.1)/2) ))

-- The theorem to be proven
theorem find_values_of_a2_b2 : a^2 + b^2 = 15 :=
sorry

end find_values_of_a2_b2_l81_81915


namespace brenda_initial_peaches_l81_81618

variable (P : ℕ)

def brenda_conditions (P : ℕ) : Prop :=
  let fresh_peaches := P - 15
  (P > 15) ∧ (fresh_peaches * 60 = 100 * 150)

theorem brenda_initial_peaches : ∃ (P : ℕ), brenda_conditions P ∧ P = 250 :=
by
  sorry

end brenda_initial_peaches_l81_81618


namespace employee_total_correct_l81_81137

variable (total_employees : ℝ)
variable (percentage_female : ℝ)
variable (percentage_male_literate : ℝ)
variable (percentage_total_literate : ℝ)
variable (number_female_literate : ℝ)
variable (percentage_male : ℝ := 1 - percentage_female)

variables (E : ℝ) (CF : ℝ) (M : ℝ) (total_literate : ℝ)

theorem employee_total_correct :
  percentage_female = 0.60 ∧
  percentage_male_literate = 0.50 ∧
  percentage_total_literate = 0.62 ∧
  number_female_literate = 546 ∧
  (total_employees = 1300) :=
by
  -- Change these variables according to the context or find a way to prove this
  let total_employees := 1300
  have Cf := number_female_literate / (percentage_female * total_employees)
  have total_male := percentage_male * total_employees
  have male_literate := percentage_male_literate * total_male
  have total_literate := percentage_total_literate * total_employees

  -- We replace "proof statements" with sorry here
  sorry

end employee_total_correct_l81_81137


namespace evaluate_f_2010_times_l81_81358

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x^2011)^(1/2011)

theorem evaluate_f_2010_times (x : ℝ) (h : x = 2011) :
  (f^[2010] x)^2011 = 2011^2011 :=
by
  rw [h]
  sorry

end evaluate_f_2010_times_l81_81358


namespace distance_between_points_l81_81927

/-- Given points P1 and P2 in the plane, prove that the distance between 
P1 and P2 is 5 units. -/
theorem distance_between_points : 
  let P1 : ℝ × ℝ := (-1, 1)
  let P2 : ℝ × ℝ := (2, 5)
  dist P1 P2 = 5 :=
by 
  sorry

end distance_between_points_l81_81927


namespace point_B_in_third_quadrant_l81_81609

theorem point_B_in_third_quadrant (x y : ℝ) (hx : x < 0) (hy : y < 1) :
    (y - 1 < 0) ∧ (x < 0) :=
by
  sorry  -- proof to be filled

end point_B_in_third_quadrant_l81_81609


namespace average_salary_techs_l81_81666

noncomputable def total_salary := 20000
noncomputable def average_salary_all := 750
noncomputable def num_technicians := 5
noncomputable def average_salary_non_tech := 700
noncomputable def total_workers := 20

theorem average_salary_techs :
  (20000 - (num_technicians + average_salary_non_tech * (total_workers - num_technicians))) / num_technicians = 900 := by
  sorry

end average_salary_techs_l81_81666


namespace max_rect_area_l81_81832

theorem max_rect_area (l w : ℤ) (h1 : 2 * l + 2 * w = 40) (h2 : 0 < l) (h3 : 0 < w) : 
  l * w ≤ 100 :=
by sorry

end max_rect_area_l81_81832


namespace smallest_integer_value_of_m_l81_81208

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem smallest_integer_value_of_m :
  ∀ m : ℤ, (x^2 + 4 * x - m = 0) ∧ has_two_distinct_real_roots 1 4 (-m : ℝ) → m ≥ -3 :=
by
  intro m h
  sorry

end smallest_integer_value_of_m_l81_81208


namespace B_can_complete_work_in_6_days_l81_81601

theorem B_can_complete_work_in_6_days (A B : ℝ) (h1 : (A + B) = 1 / 4) (h2 : A = 1 / 12) : B = 1 / 6 := 
by
  sorry

end B_can_complete_work_in_6_days_l81_81601


namespace original_number_of_players_l81_81848

theorem original_number_of_players 
    (n : ℕ) (W : ℕ)
    (h1 : W = n * 112)
    (h2 : W + 110 + 60 = (n + 2) * 106) : 
    n = 7 :=
by
  sorry

end original_number_of_players_l81_81848


namespace x_plus_2y_equals_2_l81_81956

theorem x_plus_2y_equals_2 (x y : ℝ) (h : |x + 3| + (2 * y - 5)^2 = 0) : x + 2 * y = 2 := 
sorry

end x_plus_2y_equals_2_l81_81956


namespace distance_along_stream_l81_81867
-- Define the problem in Lean 4

noncomputable def speed_boat_still : ℝ := 11   -- Speed of the boat in still water
noncomputable def distance_against_stream : ℝ := 9  -- Distance traveled against the stream in one hour

theorem distance_along_stream : 
  ∃ (v_s : ℝ), (speed_boat_still - v_s = distance_against_stream) ∧ (11 + v_s) * 1 = 13 := 
by
  use 2
  sorry

end distance_along_stream_l81_81867


namespace avg_marks_l81_81488

theorem avg_marks (P C M : ℕ) (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  -- Proof goes here
  sorry

end avg_marks_l81_81488


namespace quadrants_I_and_II_l81_81110

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y > 3 * x
def condition2 (x y : ℝ) : Prop := y > 6 - x^2

-- Prove that any point satisfying the conditions lies in Quadrant I or II
theorem quadrants_I_and_II (x y : ℝ) (h1 : y > 3 * x) (h2 : y > 6 - x^2) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- The proof steps are omitted
  sorry

end quadrants_I_and_II_l81_81110


namespace max_value_of_function_l81_81928

theorem max_value_of_function : ∀ x : ℝ, (0 < x ∧ x < 1) → x * (1 - x) ≤ 1 / 4 :=
sorry

end max_value_of_function_l81_81928


namespace min_abs_sum_of_diffs_l81_81504

theorem min_abs_sum_of_diffs (x : ℝ) (α β : ℝ)
  (h₁ : α * α - 6 * α + 5 = 0)
  (h₂ : β * β - 6 * β + 5 = 0)
  (h_ne : α ≠ β) :
  ∃ m, ∀ x, m = min (|x - α| + |x - β|) :=
by
  use (4)
  sorry

end min_abs_sum_of_diffs_l81_81504


namespace problem_a_l81_81933

theorem problem_a (k l m : ℝ) : 
  (k + l + m) ^ 2 >= 3 * (k * l + l * m + m * k) :=
by sorry

end problem_a_l81_81933


namespace fx_le_1_l81_81151

-- Statement
theorem fx_le_1 (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := 
sorry

end fx_le_1_l81_81151


namespace santino_fruit_total_l81_81406

-- Definitions of the conditions
def numPapayaTrees : ℕ := 2
def numMangoTrees : ℕ := 3
def papayasPerTree : ℕ := 10
def mangosPerTree : ℕ := 20
def totalFruits (pTrees : ℕ) (pPerTree : ℕ) (mTrees : ℕ) (mPerTree : ℕ) : ℕ :=
  (pTrees * pPerTree) + (mTrees * mPerTree)

-- Theorem that states the total number of fruits is 80 given the conditions
theorem santino_fruit_total : totalFruits numPapayaTrees papayasPerTree numMangoTrees mangosPerTree = 80 := 
  sorry

end santino_fruit_total_l81_81406


namespace time_for_A_alone_l81_81098

variable {W : ℝ}
variable {x : ℝ}

theorem time_for_A_alone (h1 : (W / x) + (W / 24) = W / 12) : x = 24 := 
sorry

end time_for_A_alone_l81_81098


namespace gas_cost_is_4_l81_81710

theorem gas_cost_is_4
    (mileage_rate : ℝ)
    (truck_efficiency : ℝ)
    (profit : ℝ)
    (trip_distance : ℝ)
    (trip_cost : ℝ)
    (gallons_used : ℝ)
    (cost_per_gallon : ℝ) :
  mileage_rate = 0.5 →
  truck_efficiency = 20 →
  profit = 180 →
  trip_distance = 600 →
  trip_cost = mileage_rate * trip_distance - profit →
  gallons_used = trip_distance / truck_efficiency →
  cost_per_gallon = trip_cost / gallons_used →
  cost_per_gallon = 4 :=
by
  sorry

end gas_cost_is_4_l81_81710


namespace converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l81_81370

-- Definitions
variables {α : Type} [LinearOrderedField α] {a b : α}
variables {A B C : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]

-- Proof Problem for Question 1
theorem converse_angle_bigger_side (A B C : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C]
  (angle_C angle_B : A) (side_AB side_AC : B) (h : angle_C > angle_B) : side_AB > side_AC :=
sorry

-- Proof Problem for Question 2
theorem negation_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

-- Proof Problem for Question 3
theorem contrapositive_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l81_81370


namespace cos_value_in_second_quadrant_l81_81004

variable (a : ℝ)
variables (h1 : π/2 < a ∧ a < π) (h2 : Real.sin a = 5/13)

theorem cos_value_in_second_quadrant : Real.cos a = -12/13 :=
  sorry

end cos_value_in_second_quadrant_l81_81004


namespace smallest_scalene_triangle_perimeter_is_prime_l81_81546

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧
  (p2 = p1 + 2) ∧ (p3 = p1 + 6)

noncomputable def smallest_prime_perimeter : ℕ :=
  5 + 7 + 11

theorem smallest_scalene_triangle_perimeter_is_prime :
  ∃ (p1 p2 p3 : ℕ), p1 < p2 ∧ p2 < p3 ∧ consecutive_primes p1 p2 p3 ∧ is_prime (p1 + p2 + p3) ∧ (p1 + p2 + p3 = smallest_prime_perimeter) :=
by 
  sorry

end smallest_scalene_triangle_perimeter_is_prime_l81_81546


namespace city_cleaning_total_l81_81903

variable (A B C D : ℕ)

theorem city_cleaning_total : 
  A = 54 →
  A = B + 17 →
  C = 2 * B →
  D = A / 3 →
  A + B + C + D = 183 := 
by 
  intros hA hAB hC hD
  sorry

end city_cleaning_total_l81_81903


namespace geometric_mean_unique_solution_l81_81850

-- Define the conditions
variable (k : ℕ) -- k is a natural number
variable (hk_pos : 0 < k) -- k is a positive natural number

-- The geometric mean condition translated to Lean
def geometric_mean_condition (k : ℕ) : Prop :=
  (2 * k)^2 = (k + 9) * (6 - k)

-- The main statement to prove
theorem geometric_mean_unique_solution (k : ℕ) (hk_pos : 0 < k) (h: geometric_mean_condition k) : k = 3 :=
sorry -- proof placeholder

end geometric_mean_unique_solution_l81_81850


namespace probability_of_matching_pair_l81_81891
-- Import the necessary library for probability and combinatorics

def probability_matching_pair (pairs : ℕ) (total_shoes : ℕ) : ℚ :=
  if total_shoes = 2 * pairs then
    (pairs : ℚ) / ((total_shoes * (total_shoes - 1) / 2) : ℚ)
  else 0

theorem probability_of_matching_pair (pairs := 6) (total_shoes := 12) : 
  probability_matching_pair pairs total_shoes = 1 / 11 := 
by
  sorry

end probability_of_matching_pair_l81_81891


namespace boxes_of_nerds_l81_81443

def totalCandies (kitKatBars hersheyKisses lollipops babyRuths reeseCups nerds : Nat) : Nat := 
  kitKatBars + hersheyKisses + lollipops + babyRuths + reeseCups + nerds

def adjustForGivenLollipops (total lollipopsGiven : Nat) : Nat :=
  total - lollipopsGiven

theorem boxes_of_nerds :
  ∀ (kitKatBars hersheyKisses lollipops babyRuths reeseCups lollipopsGiven totalAfterGiving nerds : Nat),
  kitKatBars = 5 →
  hersheyKisses = 3 * kitKatBars →
  lollipops = 11 →
  babyRuths = 10 →
  reeseCups = babyRuths / 2 →
  lollipopsGiven = 5 →
  totalAfterGiving = 49 →
  totalCandies kitKatBars hersheyKisses lollipops babyRuths reeseCups 0 - lollipopsGiven + nerds = totalAfterGiving →
  nerds = 8 :=
by
  intros
  sorry

end boxes_of_nerds_l81_81443


namespace find_x4_plus_y4_l81_81071

theorem find_x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^4 + y^4 = 135.5 :=
by
  sorry

end find_x4_plus_y4_l81_81071


namespace hindi_speaking_students_l81_81571

theorem hindi_speaking_students 
    (G M T A : ℕ)
    (Total : ℕ)
    (hG : G = 6)
    (hM : M = 6)
    (hT : T = 2)
    (hA : A = 1)
    (hTotal : Total = 22)
    : ∃ H, Total = G + H + M - (T - A) + A ∧ H = 10 := by
  sorry

end hindi_speaking_students_l81_81571


namespace total_gold_value_l81_81029

def legacy_bars : ℕ := 5
def aleena_bars : ℕ := legacy_bars - 2
def value_per_bar : ℕ := 2200
def total_bars : ℕ := legacy_bars + aleena_bars
def total_value : ℕ := total_bars * value_per_bar

theorem total_gold_value : total_value = 17600 :=
by
  -- Begin proof
  sorry

end total_gold_value_l81_81029


namespace chocolate_mixture_l81_81714

theorem chocolate_mixture (x : ℝ) (h_initial : 110 / 220 = 0.5)
  (h_equation : (110 + x) / (220 + x) = 0.75) : x = 220 := by
  sorry

end chocolate_mixture_l81_81714


namespace book_pages_total_l81_81652

theorem book_pages_total
  (days_in_week : ℕ)
  (daily_read_times : ℕ)
  (pages_per_time : ℕ)
  (additional_pages_per_day : ℕ)
  (num_days : days_in_week = 7)
  (times_per_day : daily_read_times = 3)
  (pages_each_time : pages_per_time = 6)
  (extra_pages : additional_pages_per_day = 2) :
  daily_read_times * pages_per_time + additional_pages_per_day * days_in_week = 140 := 
sorry

end book_pages_total_l81_81652


namespace ellipse_line_intersection_l81_81100

-- Definitions of the conditions in the Lean 4 language
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

def midpoint_eq (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2 = 1) ∧ (y1 + y2 = -2)

-- The problem statement
theorem ellipse_line_intersection :
  (∃ (l : ℝ → ℝ → Prop),
  (∀ x1 y1 x2 y2 : ℝ, ellipse_eq x1 y1 → ellipse_eq x2 y2 → midpoint_eq x1 y1 x2 y2 →
     l x1 y1 ∧ l x2 y2) ∧
  (∀ x y : ℝ, l x y → (x - 4 * y - 9 / 2 = 0))) :=
sorry

end ellipse_line_intersection_l81_81100


namespace conical_surface_radius_l81_81394

theorem conical_surface_radius (r : ℝ) :
  (2 * Real.pi * r = 5 * Real.pi) → r = 2.5 :=
by
  sorry

end conical_surface_radius_l81_81394


namespace product_sum_diff_l81_81385

variable (a b : ℝ) -- Real numbers

theorem product_sum_diff (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) :=
by
  sorry

end product_sum_diff_l81_81385


namespace invisible_trees_in_square_l81_81145

theorem invisible_trees_in_square (n : ℕ) : 
  ∃ (N M : ℕ), ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 
  Nat.gcd (N + i) (M + j) ≠ 1 :=
by
  sorry

end invisible_trees_in_square_l81_81145


namespace negation_of_p_l81_81940

theorem negation_of_p :
  ¬(∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_p_l81_81940


namespace railway_tunnel_construction_days_l81_81339

theorem railway_tunnel_construction_days
  (a b t : ℝ)
  (h1 : a = 1/3)
  (h2 : b = 20/100)
  (h3 : t = 4/5 ∨ t = 0.8)
  (total_days : ℝ)
  (h_total_days : total_days = 185)
  : total_days = 180 := 
sorry

end railway_tunnel_construction_days_l81_81339


namespace observe_three_cell_types_l81_81492

def biology_experiment
  (material : Type) (dissociation_fixative : material) (acetic_orcein_stain : material) (press_slide : Prop) : Prop :=
  ∃ (testes : material) (steps : material → Prop),
    steps testes ∧ press_slide ∧ (steps dissociation_fixative) ∧ (steps acetic_orcein_stain)

theorem observe_three_cell_types (material : Type)
  (dissociation_fixative acetic_orcein_stain : material)
  (press_slide : Prop)
  (steps : material → Prop) :
  biology_experiment material dissociation_fixative acetic_orcein_stain press_slide →
  ∃ (metaphase_of_mitosis metaphase_of_first_meiosis metaphase_of_second_meiosis : material), 
    steps metaphase_of_mitosis ∧ steps metaphase_of_first_meiosis ∧ steps metaphase_of_second_meiosis :=
sorry

end observe_three_cell_types_l81_81492


namespace min_value_inequality_l81_81010

theorem min_value_inequality (a b c : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h : a + b + c = 2) : 
  (1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a)) ≥ 27 / 8 :=
sorry

end min_value_inequality_l81_81010


namespace total_number_of_people_l81_81664

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l81_81664


namespace division_quotient_proof_l81_81660

theorem division_quotient_proof :
  (300324 / 29 = 10356) →
  (100007892 / 333 = 300324) :=
by
  intros h1
  sorry

end division_quotient_proof_l81_81660


namespace evaluate_expression_at_minus3_l81_81239

theorem evaluate_expression_at_minus3:
  (∀ x, x = -3 → (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2) :=
by
  sorry

end evaluate_expression_at_minus3_l81_81239


namespace fill_time_without_leak_l81_81303

theorem fill_time_without_leak (F L : ℝ)
  (h1 : (F - L) * 12 = 1)
  (h2 : L * 24 = 1) :
  1 / F = 8 := 
sorry

end fill_time_without_leak_l81_81303


namespace product_sequence_l81_81971

theorem product_sequence : 
  let seq := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/19683, 59049/1]
  ((seq[0] * seq[1]) * (seq[2] * seq[3]) * (seq[4] * seq[5]) * (seq[6] * seq[7]) * (seq[8] * seq[9])) = 243 :=
by
  sorry

end product_sequence_l81_81971


namespace number_of_zeros_of_f_l81_81577

noncomputable def f (x : ℝ) := (1 / 3) * x ^ 3 - x ^ 2 - 3 * x + 9

theorem number_of_zeros_of_f : ∃ (z : ℕ), z = 2 ∧ ∀ x : ℝ, (f x = 0 → x = -3 ∨ x = -2 / 3 ∨ x = 1 ∨ x = 3) := 
sorry

end number_of_zeros_of_f_l81_81577


namespace marbles_total_is_260_l81_81423

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end marbles_total_is_260_l81_81423


namespace find_constants_C_D_l81_81387

theorem find_constants_C_D
  (C : ℚ) (D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → (5 * x - 3) / (x^2 - 5 * x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32 / 9 ∧ D = 13 / 9 :=
by
  sorry

end find_constants_C_D_l81_81387


namespace function_is_monotonic_and_odd_l81_81309

   variable (a : ℝ) (x : ℝ)

   noncomputable def f : ℝ := (a^x - a^(-x))

   theorem function_is_monotonic_and_odd (h1 : a > 0) (h2 : a ≠ 1) : 
     (∀ x : ℝ, f (-x) = -f (x)) ∧ ((a > 1 → ∀ x y : ℝ, x < y → f x < f y) ∧ (0 < a ∧ a < 1 → ∀ x y : ℝ, x < y → f x > f y)) :=
   by
         sorry
   
end function_is_monotonic_and_odd_l81_81309


namespace car_production_l81_81615

theorem car_production (mp : ℕ) (h1 : 1800 = (mp + 50) * 12) : mp = 100 :=
by
  sorry

end car_production_l81_81615


namespace max_value_k_l81_81392

noncomputable def max_k (S : Finset ℕ) (A : ℕ → Finset ℕ) (k : ℕ) :=
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2)

theorem max_value_k : ∀ (S : Finset ℕ) (A : ℕ → Finset ℕ), 
  S = Finset.range 14 \{0} → 
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2) →
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) →
  ∃ k, max_k S A k ∧ k = 4 :=
sorry

end max_value_k_l81_81392


namespace factor_expression_l81_81677

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l81_81677


namespace sum_and_product_roots_l81_81758

structure quadratic_data where
  m : ℝ
  n : ℝ

def roots_sum_eq (qd : quadratic_data) : Prop :=
  qd.m / 3 = 9

def roots_product_eq (qd : quadratic_data) : Prop :=
  qd.n / 3 = 20

theorem sum_and_product_roots (qd : quadratic_data) :
  roots_sum_eq qd → roots_product_eq qd → qd.m + qd.n = 87 := by
  sorry

end sum_and_product_roots_l81_81758


namespace value_of_fraction_l81_81027

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = (-4 - (-1)) / (4 - 1)

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
  b2 * b2 = (-4) * (-1) ∧ b2 < 0

theorem value_of_fraction (a1 a2 b2 : ℝ)
  (h1 : arithmetic_sequence a1 a2)
  (h2 : geometric_sequence b2) :
  (a2 - a1) / b2 = 1 / 2 :=
by
  sorry

end value_of_fraction_l81_81027


namespace find_x_l81_81885

-- Definitions based on the conditions
def remaining_scores_after_removal (s: List ℕ) : List ℕ :=
  s.erase 87 |>.erase 94

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Converting the given problem into a Lean 4 theorem statement
theorem find_x (x : ℕ) (s : List ℕ) :
  s = [94, 87, 89, 88, 92, 90, x, 93, 92, 91] →
  average (remaining_scores_after_removal s) = 91 →
  x = 2 :=
by
  intros h1 h2
  sorry

end find_x_l81_81885


namespace identical_cubes_probability_l81_81090

/-- Statement of the problem -/
theorem identical_cubes_probability :
  let total_ways := 3^8 * 3^8  -- Total ways to paint two cubes
  let identical_ways := 3 + 72 + 252 + 504  -- Ways for identical appearance after rotation
  (identical_ways : ℝ) / total_ways = 1 / 51814 :=
by
  sorry

end identical_cubes_probability_l81_81090


namespace volume_of_triangular_prism_l81_81564

theorem volume_of_triangular_prism (S_side_face : ℝ) (distance : ℝ) :
  ∃ (Volume_prism : ℝ), Volume_prism = 1/2 * (S_side_face * distance) :=
by sorry

end volume_of_triangular_prism_l81_81564


namespace f_is_monotonic_decreasing_l81_81849

noncomputable def f (x : ℝ) : ℝ := Real.sin (1/2 * x + Real.pi / 6)

theorem f_is_monotonic_decreasing : ∀ x y : ℝ, (2 * Real.pi / 3 ≤ x ∧ x ≤ 8 * Real.pi / 3) → (2 * Real.pi / 3 ≤ y ∧ y ≤ 8 * Real.pi / 3) → x < y → f y ≤ f x :=
sorry

end f_is_monotonic_decreasing_l81_81849


namespace cost_price_of_article_l81_81266

theorem cost_price_of_article (SP CP : ℝ) (h1 : SP = 150) (h2 : SP = CP + (1 / 4) * CP) : CP = 120 :=
by
  sorry

end cost_price_of_article_l81_81266


namespace composite_sum_of_powers_l81_81268

theorem composite_sum_of_powers (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ a^2016 + b^2016 + c^2016 + d^2016 = x * y :=
by sorry

end composite_sum_of_powers_l81_81268


namespace problem_l81_81133

noncomputable def d : ℝ := -8.63

theorem problem :
  let floor_d := ⌊d⌋
  let frac_d := d - floor_d
  (3 * floor_d^2 + 20 * floor_d - 67 = 0) ∧
  (4 * frac_d^2 - 15 * frac_d + 5 = 0) → 
  d = -8.63 :=
by {
  sorry
}

end problem_l81_81133


namespace find_angle_l81_81094

theorem find_angle {x : ℝ} (h1 : ∀ i, 1 ≤ i ∧ i ≤ 9 → ∃ x, x > 0) (h2 : 9 * x = 900) : x = 100 :=
  sorry

end find_angle_l81_81094


namespace find_x_l81_81938

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, 2)
noncomputable def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def scalar_vec_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
noncomputable def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

theorem find_x (x : ℝ) :
  (vec_add a (b x)).1 * (vec_sub a (scalar_vec_mul 2 (b x))).2 =
  (vec_add a (b x)).2 * (vec_sub a (scalar_vec_mul 2 (b x))).1 →
  x = 4 :=
by sorry

end find_x_l81_81938


namespace minimum_odd_numbers_in_set_l81_81369

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l81_81369


namespace lg_eight_plus_three_lg_five_l81_81320

theorem lg_eight_plus_three_lg_five : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  sorry

end lg_eight_plus_three_lg_five_l81_81320


namespace fraction_inspected_by_Jane_l81_81509

theorem fraction_inspected_by_Jane (P : ℝ) (x y : ℝ) 
    (h1: 0.007 * x * P + 0.008 * y * P = 0.0075 * P) 
    (h2: x + y = 1) : y = 0.5 :=
by sorry

end fraction_inspected_by_Jane_l81_81509


namespace difference_of_fractions_l81_81567

theorem difference_of_fractions (h₁ : 1/10 * 8000 = 800) (h₂ : (1/20) / 100 * 8000 = 4) : 800 - 4 = 796 :=
by
  sorry

end difference_of_fractions_l81_81567


namespace solve_system_l81_81565

theorem solve_system :
  ∃ (x y z : ℝ), x + y + z = 9 ∧ (1/x + 1/y + 1/z = 1) ∧ (x * y + x * z + y * z = 27) ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end solve_system_l81_81565


namespace measles_cases_1995_l81_81985

-- Definitions based on the conditions
def initial_cases_1970 : ℕ := 300000
def final_cases_2000 : ℕ := 200
def cases_1990 : ℕ := 1000
def decrease_rate : ℕ := 14950 -- Annual linear decrease from 1970-1990
def a : ℤ := -8 -- Coefficient for the quadratic phase

-- Function modeling the number of cases in the quadratic phase (1990-2000)
def measles_cases (x : ℕ) : ℤ := a * (x - 1990)^2 + cases_1990

-- The statement we want to prove
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end measles_cases_1995_l81_81985


namespace rolling_dice_probability_l81_81574

-- Defining variables and conditions
def total_outcomes : Nat := 6^7

def favorable_outcomes : Nat :=
  Nat.choose 7 2 * 6 * (Nat.factorial 5) -- Calculation for exactly one pair of identical numbers

def probability : Rat :=
  favorable_outcomes / total_outcomes

-- The main theorem to prove the probability is 5/18
theorem rolling_dice_probability :
  probability = 5 / 18 := by
  sorry

end rolling_dice_probability_l81_81574


namespace find_m_n_l81_81111

theorem find_m_n (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_gcd : m.gcd n = 1) (h_div : (m^3 + n^3) ∣ (m^2 + 20 * m * n + n^2)) :
  (m, n) ∈ [(1, 2), (2, 1), (2, 3), (3, 2), (1, 5), (5, 1)] :=
by
  sorry

end find_m_n_l81_81111


namespace oil_truck_radius_l81_81015

theorem oil_truck_radius
  (r_stationary : ℝ) (h_stationary : ℝ) (h_drop : ℝ) 
  (h_truck : ℝ)
  (V_pumped : ℝ) (π : ℝ) (r_truck : ℝ) :
  r_stationary = 100 → h_stationary = 25 → h_drop = 0.064 → h_truck = 10 →
  V_pumped = π * r_stationary^2 * h_drop →
  V_pumped = π * r_truck^2 * h_truck →
  r_truck = 8 := 
by 
  intros r_stationary_eq h_stationary_eq h_drop_eq h_truck_eq V_pumped_eq1 V_pumped_eq2
  sorry

end oil_truck_radius_l81_81015


namespace kendra_total_earnings_l81_81623

-- Definitions of the conditions based on the problem statement
def kendra_earnings_2014 : ℕ := 30000 - 8000
def laurel_earnings_2014 : ℕ := 30000
def kendra_earnings_2015 : ℕ := laurel_earnings_2014 + (laurel_earnings_2014 / 5)

-- The statement to be proved
theorem kendra_total_earnings : kendra_earnings_2014 + kendra_earnings_2015 = 58000 :=
by
  -- Using Lean tactics for the proof
  sorry

end kendra_total_earnings_l81_81623


namespace farmer_apples_after_giving_l81_81134

-- Define the initial number of apples and the number of apples given to the neighbor
def initial_apples : ℕ := 127
def given_apples : ℕ := 88

-- Define the expected number of apples after giving some away
def remaining_apples : ℕ := 39

-- Formulate the proof problem
theorem farmer_apples_after_giving : initial_apples - given_apples = remaining_apples := by
  sorry

end farmer_apples_after_giving_l81_81134


namespace part1_part2_a_part2_b_part2_c_l81_81668

noncomputable def f (x a : ℝ) := Real.exp x - x - a

theorem part1 (x : ℝ) : f x 0 > x := 
by 
  -- here would be the proof
  sorry

theorem part2_a (a : ℝ) : a > 1 → ∃ z₁ z₂ : ℝ, f z₁ a = 0 ∧ f z₂ a = 0 ∧ z₁ ≠ z₂ := 
by 
  -- here would be the proof
  sorry

theorem part2_b (a : ℝ) : a < 1 → ¬ (∃ z : ℝ, f z a = 0) := 
by 
  -- here would be the proof
  sorry

theorem part2_c : f 0 1 = 0 := 
by 
  -- here would be the proof
  sorry

end part1_part2_a_part2_b_part2_c_l81_81668


namespace new_area_is_726_l81_81171

variable (l w : ℝ)
variable (h_area : l * w = 576)
variable (l' : ℝ := 1.20 * l)
variable (w' : ℝ := 1.05 * w)

theorem new_area_is_726 : l' * w' = 726 := by
  sorry

end new_area_is_726_l81_81171


namespace find_x_l81_81048

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Lean statement asserting that if a is parallel to b for some x, then x = 2
theorem find_x (x : ℝ) (h : parallel a (b x)) : x = 2 := 
by sorry

end find_x_l81_81048


namespace original_number_is_17_l81_81252

theorem original_number_is_17 (x : ℤ) (h : (x + 6) % 23 = 0) : x = 17 :=
sorry

end original_number_is_17_l81_81252


namespace tommys_family_members_l81_81944

-- Definitions
def ounces_per_member : ℕ := 16
def ounces_per_steak : ℕ := 20
def steaks_needed : ℕ := 4

-- Theorem statement
theorem tommys_family_members : (steaks_needed * ounces_per_steak) / ounces_per_member = 5 :=
by
  -- Proof goes here
  sorry

end tommys_family_members_l81_81944


namespace largest_x_floor_condition_l81_81507

theorem largest_x_floor_condition :
  ∃ x : ℝ, (⌊x⌋ : ℝ) / x = 8 / 9 ∧
      (∀ y : ℝ, (⌊y⌋ : ℝ) / y = 8 / 9 → y ≤ x) →
  x = 63 / 8 :=
by
  sorry

end largest_x_floor_condition_l81_81507


namespace smallest_number_is_28_l81_81537

theorem smallest_number_is_28 (a b c : ℕ) (h1 : (a + b + c) / 3 = 30) (h2 : b = 28) (h3 : b = c - 6) : a = 28 :=
by sorry

end smallest_number_is_28_l81_81537


namespace two_numbers_with_difference_less_than_half_l81_81735

theorem two_numbers_with_difference_less_than_half
  (x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x1) (h2 : x1 < 1)
  (h3 : 0 ≤ x2) (h4 : x2 < 1)
  (h5 : 0 ≤ x3) (h6 : x3 < 1) :
  ∃ a b, 
    (a = x1 ∨ a = x2 ∨ a = x3) ∧
    (b = x1 ∨ b = x2 ∨ b = x3) ∧
    a ≠ b ∧ 
    |b - a| < 1 / 2 :=
sorry

end two_numbers_with_difference_less_than_half_l81_81735


namespace no_solution_in_natural_numbers_l81_81614

theorem no_solution_in_natural_numbers (x y z : ℕ) : 
  (x / y : ℚ) + (y / z : ℚ) + (z / x : ℚ) ≠ 1 := 
by sorry

end no_solution_in_natural_numbers_l81_81614


namespace g_value_at_2002_l81_81189

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions given in the problem
axiom f_one : f 1 = 1
axiom f_inequality_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define the function g based on f
def g (x : ℝ) : ℝ := f x + 1 - x

-- The goal is to prove that g 2002 = 1
theorem g_value_at_2002 : g 2002 = 1 :=
sorry

end g_value_at_2002_l81_81189


namespace factorize_expression_l81_81147

theorem factorize_expression (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) :=
  sorry

end factorize_expression_l81_81147


namespace relationship_between_abc_l81_81827

noncomputable def a : ℝ := Real.exp 0.9 + 1
def b : ℝ := 2.9
noncomputable def c : ℝ := Real.log (0.9 * Real.exp 3)

theorem relationship_between_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_between_abc_l81_81827


namespace five_term_geometric_sequence_value_of_b_l81_81435

theorem five_term_geometric_sequence_value_of_b (a b c : ℝ) (h₁ : b ^ 2 = 81) (h₂ : a ^ 2 = b) (h₃ : 1 * a = a) (h₄ : c * c = c) :
  b = 9 :=
by 
  sorry

end five_term_geometric_sequence_value_of_b_l81_81435


namespace argument_friends_count_l81_81617

-- Define the conditions
def original_friends: ℕ := 20
def current_friends: ℕ := 19
def new_friend: ℕ := 1

-- Define the statement that needs to be proved
theorem argument_friends_count : 
  (original_friends + new_friend - current_friends = 1) :=
by
  -- Placeholder for the proof
  sorry

end argument_friends_count_l81_81617


namespace infinitely_many_solutions_l81_81953

def circ (x y : ℝ) : ℝ := 4 * x - 3 * y + x * y

theorem infinitely_many_solutions : ∀ y : ℝ, circ 3 y = 12 := by
  sorry

end infinitely_many_solutions_l81_81953


namespace percentage_2x_minus_y_of_x_l81_81500

noncomputable def x_perc_of_2x_minus_y (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) : ℤ :=
  (2 * x - y) * 100 / x

theorem percentage_2x_minus_y_of_x (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) :
  x_perc_of_2x_minus_y x y z h1 h2 h3 h4 = 175 :=
  sorry

end percentage_2x_minus_y_of_x_l81_81500


namespace altered_solution_water_amount_l81_81864

def initial_bleach_ratio := 2
def initial_detergent_ratio := 40
def initial_water_ratio := 100

def new_bleach_to_detergent_ratio := 3 * initial_bleach_ratio
def new_detergent_to_water_ratio := initial_detergent_ratio / 2

def detergent_amount := 60
def water_amount := 75

theorem altered_solution_water_amount :
  (initial_detergent_ratio / new_detergent_to_water_ratio) * detergent_amount / new_bleach_to_detergent_ratio = water_amount :=
by
  sorry

end altered_solution_water_amount_l81_81864


namespace remainder_sum_l81_81447

-- Define the conditions given in the problem.
def remainder_13_mod_5 : ℕ := 3
def remainder_12_mod_5 : ℕ := 2
def remainder_11_mod_5 : ℕ := 1

theorem remainder_sum :
  ((13 ^ 6 + 12 ^ 7 + 11 ^ 8) % 5) = 3 := by
  sorry

end remainder_sum_l81_81447


namespace range_of_m_for_log_function_domain_l81_81772

theorem range_of_m_for_log_function_domain (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + m > 0) → m > 8 :=
by
  sorry

end range_of_m_for_log_function_domain_l81_81772


namespace xyz_value_l81_81049

theorem xyz_value (x y z : ℝ) (h1 : y = x + 1) (h2 : x + y = 2 * z) (h3 : x = 3) : x * y * z = 42 :=
by
  -- proof here
  sorry

end xyz_value_l81_81049


namespace calculate_subtraction_l81_81292

theorem calculate_subtraction :
  ∀ (x : ℕ), (49 = 50 - 1) → (49^2 = 50^2 - 99)
  := by
  intros x h
  sorry

end calculate_subtraction_l81_81292


namespace harry_did_not_get_an_A_l81_81604

theorem harry_did_not_get_an_A
  (emily_Imp_frank : Prop)
  (frank_Imp_gina : Prop)
  (gina_Imp_harry : Prop)
  (exactly_one_did_not_get_an_A : ¬ (emily_Imp_frank ∧ frank_Imp_gina ∧ gina_Imp_harry)) :
  ¬ harry_Imp_gina :=
  sorry

end harry_did_not_get_an_A_l81_81604


namespace proof_problem_l81_81958

theorem proof_problem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 + 2 * |y| = 2 * x * y) :
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) :=
by
  sorry

end proof_problem_l81_81958


namespace unique_function_solution_l81_81193

theorem unique_function_solution (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, x ≥ 1 → f x ≥ 1)
  (h₂ : ∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1))
  (h₃ : ∀ x : ℝ, x ≥ 1 → f (x + 1) = (f x)^2/x - 1/x) :
  ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
by
  intro x hx
  sorry

end unique_function_solution_l81_81193


namespace three_divides_n_of_invertible_diff_l81_81129

theorem three_divides_n_of_invertible_diff
  (n : ℕ)
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A * A + B * B = A * B)
  (h2 : Invertible (B * A - A * B)) :
  3 ∣ n :=
sorry

end three_divides_n_of_invertible_diff_l81_81129


namespace new_angle_after_rotation_l81_81518

def initial_angle : ℝ := 25
def rotation_clockwise : ℝ := 350
def equivalent_rotation := rotation_clockwise - 360  -- equivalent to -10 degrees

theorem new_angle_after_rotation :
  initial_angle + equivalent_rotation = 15 := by
  sorry

end new_angle_after_rotation_l81_81518


namespace necessary_but_not_sufficient_l81_81653

variable (x : ℝ)

theorem necessary_but_not_sufficient (h : x > 2) : x > 1 ∧ ¬ (x > 1 → x > 2) :=
by
  sorry

end necessary_but_not_sufficient_l81_81653


namespace fifth_term_power_of_five_sequence_l81_81269

theorem fifth_term_power_of_five_sequence : 5^0 + 5^1 + 5^2 + 5^3 + 5^4 = 781 := 
by
sorry

end fifth_term_power_of_five_sequence_l81_81269


namespace mary_paid_amount_l81_81422

-- Definitions for the conditions:
def is_adult (person : String) : Prop := person = "Mary"
def children_count (n : ℕ) : Prop := n = 3
def ticket_cost_adult : ℕ := 2  -- $2 for adults
def ticket_cost_child : ℕ := 1  -- $1 for children
def change_received : ℕ := 15   -- $15 change

-- Mathematical proof to find the amount Mary paid given the conditions
theorem mary_paid_amount (person : String) (n : ℕ) 
  (h1 : is_adult person) (h2 : children_count n) :
  ticket_cost_adult + ticket_cost_child * n + change_received = 20 := 
by 
  -- Sorry as the proof is not required
  sorry

end mary_paid_amount_l81_81422


namespace common_tangent_lines_count_l81_81628

-- Define the first circle
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

-- Define the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y - 9 = 0

-- Definition for the number of common tangent lines between two circles
def number_of_common_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The theorem stating the number of common tangent lines between the given circles
theorem common_tangent_lines_count : number_of_common_tangent_lines C1 C2 = 2 := by
  sorry

end common_tangent_lines_count_l81_81628


namespace ratio_of_eggs_used_l81_81797

theorem ratio_of_eggs_used (total_eggs : ℕ) (eggs_left : ℕ) (eggs_broken : ℕ) (eggs_bought : ℕ) :
  total_eggs = 72 →
  eggs_left = 21 →
  eggs_broken = 15 →
  eggs_bought = total_eggs - (eggs_left + eggs_broken) →
  (eggs_bought / total_eggs) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_eggs_used_l81_81797


namespace ravi_work_alone_days_l81_81246

theorem ravi_work_alone_days (R : ℝ) (h1 : 1 / 75 + 1 / R = 1 / 30) : R = 50 :=
sorry

end ravi_work_alone_days_l81_81246


namespace tape_for_small_box_l81_81816

theorem tape_for_small_box (S : ℝ) :
  (2 * 4) + (8 * 2) + (5 * S) + (2 + 8 + 5) = 44 → S = 1 :=
by
  intro h
  sorry

end tape_for_small_box_l81_81816


namespace mrs_oaklyn_profit_is_correct_l81_81610

def cost_of_buying_rugs (n : ℕ) (cost_per_rug : ℕ) : ℕ :=
  n * cost_per_rug

def transportation_fee (n : ℕ) (fee_per_rug : ℕ) : ℕ :=
  n * fee_per_rug

def selling_price_before_tax (n : ℕ) (price_per_rug : ℕ) : ℕ :=
  n * price_per_rug

def total_tax (price_before_tax : ℕ) (tax_rate : ℕ) : ℕ :=
  price_before_tax * tax_rate / 100

def total_selling_price_after_tax (price_before_tax : ℕ) (tax_amount : ℕ) : ℕ :=
  price_before_tax + tax_amount

def profit (selling_price_after_tax : ℕ) (cost_of_buying : ℕ) (transport_fee : ℕ) : ℕ :=
  selling_price_after_tax - (cost_of_buying + transport_fee)

def rugs := 20
def cost_per_rug := 40
def transport_fee_per_rug := 5
def price_per_rug := 60
def tax_rate := 10

theorem mrs_oaklyn_profit_is_correct : 
  profit 
    (total_selling_price_after_tax 
      (selling_price_before_tax rugs price_per_rug) 
      (total_tax (selling_price_before_tax rugs price_per_rug) tax_rate)
    )
    (cost_of_buying_rugs rugs cost_per_rug) 
    (transportation_fee rugs transport_fee_per_rug) 
  = 420 :=
by sorry

end mrs_oaklyn_profit_is_correct_l81_81610


namespace annular_region_area_l81_81411

noncomputable def area_annulus (r1 r2 : ℝ) : ℝ :=
  (Real.pi * r2 ^ 2) - (Real.pi * r1 ^ 2)

theorem annular_region_area :
  area_annulus 4 7 = 33 * Real.pi :=
by 
  sorry

end annular_region_area_l81_81411


namespace evaluate_expression_l81_81508

variables (x : ℝ)

theorem evaluate_expression :
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 :=
by 
  sorry

end evaluate_expression_l81_81508


namespace bruce_paid_amount_l81_81076

def kg_of_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_of_mangoes : ℕ := 10
def rate_per_kg_mangoes : ℕ := 55

def total_amount_paid : ℕ := (kg_of_grapes * rate_per_kg_grapes) + (kg_of_mangoes * rate_per_kg_mangoes)

theorem bruce_paid_amount : total_amount_paid = 1110 :=
by sorry

end bruce_paid_amount_l81_81076


namespace songs_can_be_stored_l81_81792

def totalStorageGB : ℕ := 16
def usedStorageGB : ℕ := 4
def songSizeMB : ℕ := 30
def gbToMb : ℕ := 1000

def remainingStorageGB := totalStorageGB - usedStorageGB
def remainingStorageMB := remainingStorageGB * gbToMb
def numberOfSongs := remainingStorageMB / songSizeMB

theorem songs_can_be_stored : numberOfSongs = 400 :=
by
  sorry

end songs_can_be_stored_l81_81792


namespace least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l81_81471

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_primes_greater_than_five : List ℕ :=
  [7, 11, 13]

theorem least_positive_integer_divisible_by_three_smallest_primes_greater_than_five : 
  ∃ n : ℕ, n > 0 ∧ (∀ p ∈ smallest_primes_greater_than_five, p ∣ n) ∧ n = 1001 := by
  sorry

end least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l81_81471


namespace geo_arith_sequences_sum_first_2n_terms_l81_81434

variables (n : ℕ)

-- Given conditions in (a)
def common_ratio : ℕ := 3
def arithmetic_diff : ℕ := 2

-- The sequences provided in the solution (b)
def a_n (n : ℕ) : ℕ := common_ratio ^ n
def b_n (n : ℕ) : ℕ := 2 * n + 1

-- Sum formula for geometric series up to 2n terms
def S_2n (n : ℕ) : ℕ := (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n

theorem geo_arith_sequences :
  a_n n = common_ratio ^ n
  ∨ b_n n = 2 * n + 1 := sorry

theorem sum_first_2n_terms :
  S_2n n = (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n := sorry

end geo_arith_sequences_sum_first_2n_terms_l81_81434


namespace problem_statement_l81_81085

theorem problem_statement
  (a b c : ℝ)
  (h1 : a + 2 * b + 3 * c = 12)
  (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 := 
sorry

end problem_statement_l81_81085


namespace milk_left_is_correct_l81_81821

def total_morning_milk : ℕ := 365
def total_evening_milk : ℕ := 380
def milk_sold : ℕ := 612
def leftover_milk_from_yesterday : ℕ := 15

def total_milk_left : ℕ :=
  (total_morning_milk + total_evening_milk - milk_sold) + leftover_milk_from_yesterday

theorem milk_left_is_correct : total_milk_left = 148 := by
  sorry

end milk_left_is_correct_l81_81821


namespace sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l81_81503

theorem sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30 :
  (7^30 + 13^30) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l81_81503


namespace surveyed_parents_women_l81_81987

theorem surveyed_parents_women (W : ℝ) :
  (5/6 : ℝ) * W + (3/4 : ℝ) * (1 - W) = 0.8 → W = 0.6 :=
by
  intro h
  have hw : W * (1/6) + (1 - W) * (1/4) = 0.2 := sorry
  have : W = 0.6 := sorry
  exact this

end surveyed_parents_women_l81_81987


namespace carlotta_performance_time_l81_81261

theorem carlotta_performance_time :
  ∀ (s p t : ℕ),  -- s for singing, p for practicing, t for tantrums
  (∀ (n : ℕ), p = 3 * n ∧ t = 5 * n) →
  s = 6 →
  (s + p + t) = 54 :=
by 
  intros s p t h1 h2
  rcases h1 1 with ⟨h3, h4⟩
  sorry

end carlotta_performance_time_l81_81261


namespace remove_terms_to_get_two_thirds_l81_81813

noncomputable def sum_of_terms : ℚ := 
  (1/3) + (1/6) + (1/9) + (1/12) + (1/15) + (1/18)

noncomputable def sum_of_remaining_terms := 
  (1/3) + (1/6) + (1/9) + (1/18)

theorem remove_terms_to_get_two_thirds :
  sum_of_terms - (1/12 + 1/15) = (2/3) :=
by
  sorry

end remove_terms_to_get_two_thirds_l81_81813


namespace simplify_expression_l81_81808

theorem simplify_expression (x y : ℝ) : (5 - 4 * y) - (6 + 5 * y - 2 * x) = -1 - 9 * y + 2 * x := by
  sorry

end simplify_expression_l81_81808


namespace Brad_has_9_green_balloons_l81_81731

theorem Brad_has_9_green_balloons
  (total_balloons : ℕ)
  (red_balloons : ℕ)
  (green_balloons : ℕ)
  (h1 : total_balloons = 17)
  (h2 : red_balloons = 8)
  (h3 : total_balloons = red_balloons + green_balloons) :
  green_balloons = 9 := 
sorry

end Brad_has_9_green_balloons_l81_81731


namespace peanuts_added_l81_81636

theorem peanuts_added (initial final added : ℕ) (h1 : initial = 4) (h2 : final = 8) (h3 : final = initial + added) : added = 4 :=
by
  rw [h1] at h3
  rw [h2] at h3
  sorry

end peanuts_added_l81_81636


namespace union_of_sets_l81_81942

def A := { x : ℝ | -1 ≤ x ∧ x < 3 }
def B := { x : ℝ | 2 < x ∧ x ≤ 5 }

theorem union_of_sets : A ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } := 
by sorry

end union_of_sets_l81_81942


namespace part_a_part_b_l81_81575

noncomputable def triangle_exists (h1 h2 h3 : ℕ) : Prop :=
  ∃ a b c, 2 * a = h1 * (b + c) ∧ 2 * b = h2 * (a + c) ∧ 2 * c = h3 * (a + b)

theorem part_a : ¬ triangle_exists 2 3 6 :=
sorry

theorem part_b : triangle_exists 2 3 5 :=
sorry

end part_a_part_b_l81_81575


namespace fire_alarms_and_passengers_discrete_l81_81167

-- Definitions of the random variables
def xi₁ : ℕ := sorry  -- number of fire alarms in a city within one day
def xi₂ : ℝ := sorry  -- temperature in a city within one day
def xi₃ : ℕ := sorry  -- number of passengers at a train station in a city within a month

-- Defining the concept of discrete random variable
def is_discrete (X : Type) : Prop := 
  ∃ f : X → ℕ, ∀ x : X, ∃ n : ℕ, f x = n

-- Statement of the proof problem
theorem fire_alarms_and_passengers_discrete :
  is_discrete ℕ ∧ is_discrete ℕ ∧ ¬ is_discrete ℝ :=
by
  have xi₁_discrete : is_discrete ℕ := sorry
  have xi₃_discrete : is_discrete ℕ := sorry
  have xi₂_not_discrete : ¬ is_discrete ℝ := sorry
  exact ⟨xi₁_discrete, xi₃_discrete, xi₂_not_discrete⟩

end fire_alarms_and_passengers_discrete_l81_81167


namespace mixture_kerosene_l81_81195

theorem mixture_kerosene (x : ℝ) (h₁ : 0.25 * x + 1.2 = 0.27 * (x + 4)) : x = 6 :=
sorry

end mixture_kerosene_l81_81195


namespace age_sum_in_5_years_l81_81154

variable (MikeAge MomAge : ℕ)
variable (h1 : MikeAge = MomAge - 30)
variable (h2 : MikeAge + MomAge = 70)

theorem age_sum_in_5_years (h1 : MikeAge = MomAge - 30) (h2 : MikeAge + MomAge = 70) :
  (MikeAge + 5) + (MomAge + 5) = 80 := by
  sorry

end age_sum_in_5_years_l81_81154


namespace angle_QPR_l81_81139

theorem angle_QPR (PQ QR PR RS : Real) (angle_PQR angle_PRS : Real) 
  (h1 : PQ = QR) (h2 : PR = RS) (h3 : angle_PQR = 50) (h4 : angle_PRS = 100) : 
  ∃ angle_QPR : Real, angle_QPR = 25 :=
by
  -- We are proving that angle_QPR is 25 given the conditions.
  sorry

end angle_QPR_l81_81139


namespace find_original_price_l81_81925

-- Define the original price and conditions
def original_price (P : ℝ) : Prop :=
  ∃ discount final_price, discount = 0.55 ∧ final_price = 450000 ∧ ((1 - discount) * P = final_price)

-- The theorem to prove the original price before discount
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1000000 :=
by
  sorry

end find_original_price_l81_81925


namespace proof_problem_l81_81276

noncomputable def a : ℚ := 2 / 3
noncomputable def b : ℚ := - 3 / 2
noncomputable def n : ℕ := 2023

theorem proof_problem :
  (a ^ n) * (b ^ n) = -1 :=
by
  sorry

end proof_problem_l81_81276


namespace marble_arrangement_mod_l81_81196

def num_ways_arrange_marbles (m : ℕ) : ℕ := Nat.choose (m + 3) 3

theorem marble_arrangement_mod (N : ℕ) (m : ℕ) (h1: m = 11) (h2: N = num_ways_arrange_marbles m): 
  N % 1000 = 35 := by
  sorry

end marble_arrangement_mod_l81_81196


namespace sin_cos_product_l81_81973

theorem sin_cos_product (ϕ : ℝ) (h : Real.tan (ϕ + Real.pi / 4) = 5) : 
  1 / (Real.sin ϕ * Real.cos ϕ) = 13 / 6 :=
by
  sorry

end sin_cos_product_l81_81973


namespace expression_a_equals_half_expression_c_equals_half_l81_81748

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end expression_a_equals_half_expression_c_equals_half_l81_81748


namespace find_wsquared_l81_81412

theorem find_wsquared : 
  (2 * w + 10) ^ 2 = (5 * w + 15) * (w + 6) →
  w ^ 2 = (90 + 10 * Real.sqrt 65) / 4 := 
by 
  intro h₀
  sorry

end find_wsquared_l81_81412


namespace alpha_plus_beta_eq_l81_81321

variable {α β : ℝ}
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin (α - β) = 5 / 6)
variable (h4 : Real.tan α / Real.tan β = -1 / 4)

theorem alpha_plus_beta_eq : α + β = 7 * Real.pi / 6 := by
  sorry

end alpha_plus_beta_eq_l81_81321


namespace parallelogram_area_l81_81597

theorem parallelogram_area (b h : ℕ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l81_81597


namespace A_share_in_profit_l81_81464

/-
Given:
1. a_contribution (A's amount contributed in Rs. 5000) and duration (in months 8)
2. b_contribution (B's amount contributed in Rs. 6000) and duration (in months 5)
3. total_profit (Total profit in Rs. 8400)

Prove that A's share in the total profit is Rs. 4800.
-/

theorem A_share_in_profit 
  (a_contribution : ℝ) (a_months : ℝ) 
  (b_contribution : ℝ) (b_months : ℝ) 
  (total_profit : ℝ) :
  a_contribution = 5000 → 
  a_months = 8 → 
  b_contribution = 6000 → 
  b_months = 5 → 
  total_profit = 8400 → 
  (a_contribution * a_months / (a_contribution * a_months + b_contribution * b_months) * total_profit) = 4800 := 
by {
  sorry
}

end A_share_in_profit_l81_81464


namespace find_x_range_l81_81183

-- Define the condition for the expression to be meaningful
def meaningful_expr (x : ℝ) : Prop := x - 3 ≥ 0

-- The range of values for x is equivalent to x being at least 3
theorem find_x_range (x : ℝ) : meaningful_expr x ↔ x ≥ 3 := by
  sorry

end find_x_range_l81_81183


namespace min_sum_of_factors_of_144_is_neg_145_l81_81059

theorem min_sum_of_factors_of_144_is_neg_145 
  (a b : ℤ) 
  (h : a * b = 144) : 
  a + b ≥ -145 := 
sorry

end min_sum_of_factors_of_144_is_neg_145_l81_81059


namespace find_x_l81_81829

theorem find_x (p : ℕ) (hprime : Nat.Prime p) (hgt5 : p > 5) (x : ℕ) (hx : x ≠ 0) :
    (∀ n : ℕ, 0 < n → (5 * p + x) ∣ (5 * p ^ n + x ^ n)) ↔ x = p := by
  sorry

end find_x_l81_81829


namespace prove_x_plus_y_leq_zero_l81_81555

-- Definitions of the conditions
def valid_powers (a b : ℝ) (x y : ℝ) : Prop :=
  1 < a ∧ a < b ∧ a^x + b^y ≤ a^(-x) + b^(-y)

-- The theorem statement
theorem prove_x_plus_y_leq_zero (a b x y : ℝ) (h : valid_powers a b x y) : 
  x + y ≤ 0 :=
by
  sorry

end prove_x_plus_y_leq_zero_l81_81555


namespace electricity_consumption_scientific_notation_l81_81280

def electricity_consumption (x : Float) : String := 
  let scientific_notation := "3.64 × 10^4"
  scientific_notation

theorem electricity_consumption_scientific_notation :
  electricity_consumption 36400 = "3.64 × 10^4" :=
by 
  sorry

end electricity_consumption_scientific_notation_l81_81280


namespace rate_of_interest_l81_81619

theorem rate_of_interest (P T SI CI : ℝ) (hP : P = 4000) (hT : T = 2) (hSI : SI = 400) (hCI : CI = 410) :
  ∃ r : ℝ, SI = (P * r * T) / 100 ∧ CI = P * ((1 + r / 100) ^ T - 1) ∧ r = 5 :=
by
  sorry

end rate_of_interest_l81_81619


namespace expression_divisible_by_1968_l81_81904

theorem expression_divisible_by_1968 (n : ℕ) : 
  ( -1 ^ (2 * n) +  9 ^ (4 * n) - 6 ^ (8 * n) + 8 ^ (16 * n) ) % 1968 = 0 :=
by
  sorry

end expression_divisible_by_1968_l81_81904


namespace white_line_longer_l81_81605

-- Define the lengths of the white and blue lines
def white_line_length : ℝ := 7.678934
def blue_line_length : ℝ := 3.33457689

-- State the main theorem
theorem white_line_longer :
  white_line_length - blue_line_length = 4.34435711 :=
by
  sorry

end white_line_longer_l81_81605


namespace no_nat_fourfold_digit_move_l81_81581

theorem no_nat_fourfold_digit_move :
  ¬ ∃ (N : ℕ), ∃ (a : ℕ), ∃ (n : ℕ), ∃ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) ∧ 
    (N = a * 10^n + x) ∧ 
    (4 * N = 10 * x + a) :=
by
  sorry

end no_nat_fourfold_digit_move_l81_81581


namespace anton_stationary_escalator_steps_l81_81146

theorem anton_stationary_escalator_steps
  (N : ℕ)
  (H1 : N = 30)
  (H2 : 5 * N = 150) :
  (stationary_steps : ℕ) = 50 :=
by
  sorry

end anton_stationary_escalator_steps_l81_81146


namespace quadratic_root_eq_l81_81669

theorem quadratic_root_eq {b : ℝ} (h : (2 : ℝ)^2 + b * 2 - 6 = 0) : b = 1 :=
by
  sorry

end quadratic_root_eq_l81_81669


namespace calculate_fraction_l81_81486

theorem calculate_fraction :
  ( (12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484) )
  /
  ( (6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484) )
  = 181 := by
  sorry

end calculate_fraction_l81_81486


namespace beach_weather_condition_l81_81373

theorem beach_weather_condition
  (T : ℝ) -- Temperature in degrees Fahrenheit
  (sunny : Prop) -- Whether it is sunny
  (crowded : Prop) -- Whether the beach is crowded
  (H1 : ∀ (T : ℝ) (sunny : Prop), (T ≥ 80) ∧ sunny → crowded) -- Condition 1
  (H2 : ¬ crowded) -- Condition 2
  : T < 80 ∨ ¬ sunny := sorry

end beach_weather_condition_l81_81373


namespace shortest_chord_through_point_on_circle_l81_81231

theorem shortest_chord_through_point_on_circle :
  ∀ (M : ℝ × ℝ) (x y : ℝ),
    M = (3, 0) →
    x^2 + y^2 - 8 * x - 2 * y + 10 = 0 →
    ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ a = 1 ∧ b = 1 ∧ c = -3 :=
by
  sorry

end shortest_chord_through_point_on_circle_l81_81231


namespace carl_watermelons_left_l81_81822

-- Define the conditions
def price_per_watermelon : ℕ := 3
def profit : ℕ := 105
def starting_watermelons : ℕ := 53

-- Define the main proof statement
theorem carl_watermelons_left :
  (starting_watermelons - (profit / price_per_watermelon) = 18) :=
sorry

end carl_watermelons_left_l81_81822


namespace potatoes_fraction_l81_81376

theorem potatoes_fraction (w : ℝ) (x : ℝ) (h_weight : w = 36) (h_fraction : w / x = 36) : x = 1 :=
by
  sorry

end potatoes_fraction_l81_81376


namespace cyclic_inequality_l81_81525

theorem cyclic_inequality
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) :=
by
  sorry

end cyclic_inequality_l81_81525


namespace tg_arccos_le_cos_arctg_l81_81311

theorem tg_arccos_le_cos_arctg (x : ℝ) (h₀ : -1 ≤ x ∧ x ≤ 1) :
  (Real.tan (Real.arccos x) ≤ Real.cos (Real.arctan x)) → 
  (x ∈ Set.Icc (-1:ℝ) 0 ∨ x ∈ Set.Icc (Real.sqrt ((Real.sqrt 5 - 1) / 2)) 1) :=
by
  sorry

end tg_arccos_le_cos_arctg_l81_81311


namespace complement_A_in_U_l81_81824

/-- Problem conditions -/
def is_universal_set (x : ℕ) : Prop := (x - 6) * (x + 1) ≤ 0
def A : Set ℕ := {1, 2, 4}
def U : Set ℕ := { x | is_universal_set x }

/-- Proof statement -/
theorem complement_A_in_U : (U \ A) = {3, 5, 6} :=
by
  sorry  -- replacement for the proof

end complement_A_in_U_l81_81824


namespace value_of_f_at_2_and_neg_log2_3_l81_81547

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 else 2^(-x)

theorem value_of_f_at_2_and_neg_log2_3 :
  f 2 * f (-Real.log 3 / Real.log 2) = 3 := by
  sorry

end value_of_f_at_2_and_neg_log2_3_l81_81547


namespace PQ_length_l81_81910

theorem PQ_length (BC AD : ℝ) (angle_A angle_D : ℝ) (P Q : ℝ) 
  (H1 : BC = 700) (H2 : AD = 1400) (H3 : angle_A = 45) (H4 : angle_D = 45) 
  (mid_BC : P = BC / 2) (mid_AD : Q = AD / 2) :
  abs (Q - P) = 350 :=
by
  sorry

end PQ_length_l81_81910


namespace student_failed_by_l81_81084

-- Definitions based on the problem conditions
def total_marks : ℕ := 500
def passing_percentage : ℕ := 40
def marks_obtained : ℕ := 150
def passing_marks : ℕ := (passing_percentage * total_marks) / 100

-- The theorem statement
theorem student_failed_by :
  (passing_marks - marks_obtained) = 50 :=
by
  -- The proof is omitted
  sorry

end student_failed_by_l81_81084


namespace four_p_minus_three_is_square_l81_81963

theorem four_p_minus_three_is_square
  (n : ℕ) (p : ℕ)
  (hn_pos : n > 1)
  (hp_prime : Prime p)
  (h1 : n ∣ (p - 1))
  (h2 : p ∣ (n^3 - 1)) : ∃ k : ℕ, 4 * p - 3 = k^2 := sorry

end four_p_minus_three_is_square_l81_81963


namespace smallest_integer_for_perfect_square_l81_81120

theorem smallest_integer_for_perfect_square :
  let y := 2^5 * 3^5 * 4^5 * 5^5 * 6^4 * 7^3 * 8^3 * 9^2
  ∃ z : ℕ, z = 70 ∧ (∃ k : ℕ, y * z = k^2) :=
by
  sorry

end smallest_integer_for_perfect_square_l81_81120


namespace complement_intersection_eq_l81_81911

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ (M ∩ N)) = {1, 3, 4} := by
  sorry

end complement_intersection_eq_l81_81911


namespace problem1_problem2_problem3_l81_81831

-- Proof statement for Problem 1
theorem problem1 : 23 * (-5) - (-3) / (3 / 108) = -7 := 
by 
  sorry

-- Proof statement for Problem 2
theorem problem2 : (-7) * (-3) * (-0.5) + (-12) * (-2.6) = 20.7 := 
by 
  sorry

-- Proof statement for Problem 3
theorem problem3 : ((-1 / 2) - (1 / 12) + (3 / 4) - (1 / 6)) * (-48) = 0 := 
by 
  sorry

end problem1_problem2_problem3_l81_81831


namespace merchant_mixture_solution_l81_81449

variable (P C : ℝ)

def P_price : ℝ := 2.40
def C_price : ℝ := 6.00
def total_weight : ℝ := 60
def total_price_per_pound : ℝ := 3.00
def total_price : ℝ := total_price_per_pound * total_weight

theorem merchant_mixture_solution (h1 : P + C = total_weight)
                                  (h2 : P_price * P + C_price * C = total_price) :
  C = 10 := 
sorry

end merchant_mixture_solution_l81_81449


namespace average_price_l81_81541

theorem average_price (books1 books2 : ℕ) (price1 price2 : ℝ)
  (h1 : books1 = 65) (h2 : price1 = 1380)
  (h3 : books2 = 55) (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2) = 19 :=
by
  sorry

end average_price_l81_81541


namespace min_value_of_a_plus_b_l81_81825

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / (a + 1)) + (2 / (1 + b)) = 1) : 
  a + b ≥ 2 * Real.sqrt 2 + 1 :=
sorry

end min_value_of_a_plus_b_l81_81825


namespace abcde_sum_l81_81484

theorem abcde_sum : 
  ∀ (a b c d e : ℝ), 
  a + 1 = b + 2 → 
  b + 2 = c + 3 → 
  c + 3 = d + 4 → 
  d + 4 = e + 5 → 
  e + 5 = a + b + c + d + e + 10 → 
  a + b + c + d + e = -35 / 4 :=
sorry

end abcde_sum_l81_81484


namespace find_n_l81_81932

theorem find_n (n : ℤ) (hn_range : -150 < n ∧ n < 150) (h_tan : Real.tan (n * Real.pi / 180) = Real.tan (286 * Real.pi / 180)) : 
  n = -74 :=
sorry

end find_n_l81_81932


namespace apples_fallen_l81_81432

theorem apples_fallen (H1 : ∃ ground_apples : ℕ, ground_apples = 10 + 3)
                      (H2 : ∃ tree_apples : ℕ, tree_apples = 5)
                      (H3 : ∃ total_apples : ℕ, total_apples = ground_apples ∧ total_apples = 10 + 3 + 5)
                      : ∃ fallen_apples : ℕ, fallen_apples = 13 :=
by
  sorry

end apples_fallen_l81_81432


namespace loss_is_negative_one_point_twenty_seven_percent_l81_81408

noncomputable def book_price : ℝ := 600
noncomputable def gov_tax_rate : ℝ := 0.05
noncomputable def shipping_fee : ℝ := 20
noncomputable def seller_discount_rate : ℝ := 0.03
noncomputable def selling_price : ℝ := 624

noncomputable def gov_tax : ℝ := gov_tax_rate * book_price
noncomputable def seller_discount : ℝ := seller_discount_rate * book_price
noncomputable def total_cost : ℝ := book_price + gov_tax + shipping_fee - seller_discount
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def loss_percentage : ℝ := (profit / total_cost) * 100

theorem loss_is_negative_one_point_twenty_seven_percent :
  loss_percentage = -1.27 :=
by
  sorry

end loss_is_negative_one_point_twenty_seven_percent_l81_81408


namespace number_of_multiples_of_10_lt_200_l81_81074

theorem number_of_multiples_of_10_lt_200 : 
  ∃ n, (∀ k, (1 ≤ k) → (k < 20) → k * 10 < 200) ∧ n = 19 := 
by
  sorry

end number_of_multiples_of_10_lt_200_l81_81074


namespace find_w_squared_l81_81965

theorem find_w_squared (w : ℝ) (h : (2 * w + 19) ^ 2 = (4 * w + 9) * (3 * w + 13)) :
  w ^ 2 = ((6 + Real.sqrt 524) / 4) ^ 2 :=
sorry

end find_w_squared_l81_81965


namespace arithmetic_calculation_l81_81051

theorem arithmetic_calculation : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 :=
by
  sorry

end arithmetic_calculation_l81_81051


namespace total_sweaters_l81_81031

-- Define the conditions
def washes_per_load : ℕ := 9
def total_shirts : ℕ := 19
def total_loads : ℕ := 3

-- Define the total_sweaters theorem to prove Nancy had to wash 9 sweaters
theorem total_sweaters {n : ℕ} (h1 : washes_per_load = 9) (h2 : total_shirts = 19) (h3 : total_loads = 3) : n = 9 :=
by
  sorry

end total_sweaters_l81_81031


namespace remaining_employees_earn_rate_l81_81650

theorem remaining_employees_earn_rate
  (total_employees : ℕ)
  (employees_12_per_hour : ℕ)
  (employees_14_per_hour : ℕ)
  (total_cost : ℝ)
  (hourly_rate_12 : ℝ)
  (hourly_rate_14 : ℝ)
  (shift_hours : ℝ)
  (remaining_employees : ℕ)
  (remaining_hourly_rate : ℝ) :
  total_employees = 300 →
  employees_12_per_hour = 200 →
  employees_14_per_hour = 40 →
  total_cost = 31840 →
  hourly_rate_12 = 12 →
  hourly_rate_14 = 14 →
  shift_hours = 8 →
  remaining_employees = 60 →
  remaining_hourly_rate = 
    (total_cost - (employees_12_per_hour * hourly_rate_12 * shift_hours) - 
    (employees_14_per_hour * hourly_rate_14 * shift_hours)) / 
    (remaining_employees * shift_hours) →
  remaining_hourly_rate = 17 :=
by
  sorry

end remaining_employees_earn_rate_l81_81650


namespace expand_product_l81_81204

noncomputable def question_expression (x : ℝ) := -3 * (2 * x + 4) * (x - 7)
noncomputable def correct_answer (x : ℝ) := -6 * x^2 + 30 * x + 84

theorem expand_product (x : ℝ) : question_expression x = correct_answer x := 
by sorry

end expand_product_l81_81204


namespace reflect_parabola_x_axis_l81_81255

theorem reflect_parabola_x_axis (x : ℝ) (a b c : ℝ) :
  (∀ y : ℝ, y = x^2 + x - 2 → -y = x^2 + x - 2) →
  (∀ y : ℝ, -y = x^2 + x - 2 → y = -x^2 - x + 2) :=
by
  intros h₁ h₂
  intro y
  sorry

end reflect_parabola_x_axis_l81_81255


namespace payment_ratio_l81_81056

theorem payment_ratio (m p t : ℕ) (hm : m = 14) (hp : p = 84) (ht : t = m * 12) :
  (p : ℚ) / ((t : ℚ) - p) = 1 :=
by
  sorry

end payment_ratio_l81_81056


namespace heather_ends_up_with_45_blocks_l81_81318

-- Conditions
def initialBlocks (Heather : Type) : ℕ := 86
def sharedBlocks (Heather : Type) : ℕ := 41

-- The theorem to prove
theorem heather_ends_up_with_45_blocks (Heather : Type) :
  (initialBlocks Heather) - (sharedBlocks Heather) = 45 :=
by
  sorry

end heather_ends_up_with_45_blocks_l81_81318


namespace students_water_count_l81_81064

-- Define the given conditions
def pct_students_juice (total_students : ℕ) : ℕ := 70 * total_students / 100
def pct_students_water (total_students : ℕ) : ℕ := 30 * total_students / 100
def students_juice (total_students : ℕ) : Prop := pct_students_juice total_students = 140

-- Define the proposition that needs to be proven
theorem students_water_count (total_students : ℕ) (h1 : students_juice total_students) : 
  pct_students_water total_students = 60 := 
by
  sorry


end students_water_count_l81_81064


namespace johns_final_push_time_l81_81130

theorem johns_final_push_time :
  ∃ t : ℝ, t = 17 / 4.2 := 
by
  sorry

end johns_final_push_time_l81_81130


namespace smaller_root_of_quadratic_l81_81543

theorem smaller_root_of_quadratic :
  ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂) ∧ (x₁^2 - 14 * x₁ + 45 = 0) ∧ (x₂^2 - 14 * x₂ + 45 = 0) ∧ (min x₁ x₂ = 5) :=
sorry

end smaller_root_of_quadratic_l81_81543


namespace cos_C_value_l81_81800

theorem cos_C_value (a b c : ℝ) (A B C : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : 
  Real.cos C = 7 / 25 :=
  sorry

end cos_C_value_l81_81800


namespace inequality_correctness_l81_81426

theorem inequality_correctness (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_correctness_l81_81426


namespace total_yards_thrown_l81_81897

-- Definitions for the conditions
def distance_50_degrees : ℕ := 20
def distance_80_degrees : ℕ := distance_50_degrees * 2

def throws_on_saturday : ℕ := 20
def throws_on_sunday : ℕ := 30

def headwind_penalty : ℕ := 5
def tailwind_bonus : ℕ := 10

-- Theorem for the total yards thrown in two days
theorem total_yards_thrown :
  ((distance_50_degrees - headwind_penalty) * throws_on_saturday) + 
  ((distance_80_degrees + tailwind_bonus) * throws_on_sunday) = 1800 :=
by
  sorry

end total_yards_thrown_l81_81897


namespace unique_fraction_condition_l81_81584

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l81_81584


namespace Petya_wins_optimally_l81_81177

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end Petya_wins_optimally_l81_81177


namespace papaya_cost_is_one_l81_81324

theorem papaya_cost_is_one (lemons_cost : ℕ) (mangos_cost : ℕ) (total_fruits : ℕ) (total_cost_paid : ℕ) :
    (lemons_cost = 2) → (mangos_cost = 4) → (total_fruits = 12) → (total_cost_paid = 21) → 
    let discounts := total_fruits / 4
    let lemons_bought := 6
    let mangos_bought := 2
    let papayas_bought := 4
    let total_discount := discounts
    let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
    total_cost_before_discount - total_discount = total_cost_paid → 
    P = 1 := 
by 
  intros h1 h2 h3 h4 
  let discounts := total_fruits / 4
  let lemons_bought := 6
  let mangos_bought := 2
  let papayas_bought := 4
  let total_discount := discounts
  let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
  sorry

end papaya_cost_is_one_l81_81324


namespace penalty_kicks_l81_81871

-- Define the soccer team data
def total_players : ℕ := 16
def goalkeepers : ℕ := 2
def players_shooting : ℕ := total_players - goalkeepers -- 14

-- Function to calculate total penalty kicks
def total_penalty_kicks (total_players goalkeepers : ℕ) : ℕ :=
  let players_shooting := total_players - goalkeepers
  players_shooting * goalkeepers

-- Theorem stating the number of penalty kicks
theorem penalty_kicks : total_penalty_kicks total_players goalkeepers = 30 :=
by
  sorry

end penalty_kicks_l81_81871


namespace rosa_called_last_week_l81_81168

noncomputable def total_pages_called : ℝ := 18.8
noncomputable def pages_called_this_week : ℝ := 8.6
noncomputable def pages_called_last_week : ℝ := total_pages_called - pages_called_this_week

theorem rosa_called_last_week :
  pages_called_last_week = 10.2 :=
by
  sorry

end rosa_called_last_week_l81_81168


namespace jillian_distance_l81_81715

theorem jillian_distance : 
  ∀ (x y z : ℝ),
  (1 / 63) * x + (1 / 77) * y + (1 / 99) * z = 11 / 3 →
  (1 / 63) * z + (1 / 77) * y + (1 / 99) * x = 13 / 3 →
  x + y + z = 308 :=
by
  sorry

end jillian_distance_l81_81715


namespace don_eats_80_pizzas_l81_81999

variable (D Daria : ℝ)

-- Condition 1: Daria consumes 2.5 times the amount of pizza that Don does.
def condition1 : Prop := Daria = 2.5 * D

-- Condition 2: Together, they eat 280 pizzas.
def condition2 : Prop := D + Daria = 280

-- Conclusion: The number of pizzas Don eats is 80.
theorem don_eats_80_pizzas (h1 : condition1 D Daria) (h2 : condition2 D Daria) : D = 80 :=
by
  sorry

end don_eats_80_pizzas_l81_81999


namespace inequality_always_true_l81_81539

theorem inequality_always_true 
  (a b : ℝ) 
  (h1 : ab > 0) : 
  (b / a) + (a / b) ≥ 2 := 
by sorry

end inequality_always_true_l81_81539


namespace remaining_sand_fraction_l81_81348

theorem remaining_sand_fraction (total_weight : ℕ) (used_weight : ℕ) (h1 : total_weight = 50) (h2 : used_weight = 30) : 
  (total_weight - used_weight) / total_weight = 2 / 5 :=
by 
  sorry

end remaining_sand_fraction_l81_81348


namespace roots_of_quadratic_eq_l81_81855

theorem roots_of_quadratic_eq:
  (8 * γ^3 + 15 * δ^2 = 179) ↔ (γ^2 - 3 * γ + 1 = 0 ∧ δ^2 - 3 * δ + 1 = 0) :=
sorry

end roots_of_quadratic_eq_l81_81855


namespace arithmetic_sequence_m_value_l81_81506

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

noncomputable def find_m (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  (a (m + 1) + a (m - 1) - a m ^ 2 = 0) → (S (2 * m - 1) = 38) → m = 10

-- Problem Statement
theorem arithmetic_sequence_m_value :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ),
    arithmetic_sequence a → 
    sum_of_first_n_terms S a → 
    find_m a S m :=
by
  intros a S m ha hs h₁ h₂
  sorry

end arithmetic_sequence_m_value_l81_81506


namespace factor_expression_l81_81638

theorem factor_expression (x : ℝ) :
  (16 * x ^ 7 + 36 * x ^ 4 - 9) - (4 * x ^ 7 - 6 * x ^ 4 - 9) = 6 * x ^ 4 * (2 * x ^ 3 + 7) :=
by
  sorry

end factor_expression_l81_81638


namespace find_d_plus_q_l81_81823

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + d * (n * (n - 1) / 2)

noncomputable def sum_geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * b₁
  else b₁ * (q ^ n - 1) / (q - 1)

noncomputable def sum_combined_sequence (a₁ d b₁ q : ℝ) (n : ℕ) : ℝ :=
  sum_arithmetic_sequence a₁ d n + sum_geometric_sequence b₁ q n

theorem find_d_plus_q (a₁ d b₁ q : ℝ) (h_seq: ∀ n : ℕ, 0 < n → sum_combined_sequence a₁ d b₁ q n = n^2 - n + 2^n - 1) :
  d + q = 4 :=
  sorry

end find_d_plus_q_l81_81823


namespace fourth_vs_third_difference_l81_81009

def first_competitor_distance : ℕ := 22

def second_competitor_distance : ℕ := first_competitor_distance + 1

def third_competitor_distance : ℕ := second_competitor_distance - 2

def fourth_competitor_distance : ℕ := 24

theorem fourth_vs_third_difference : 
  fourth_competitor_distance - third_competitor_distance = 3 := by
  sorry

end fourth_vs_third_difference_l81_81009


namespace exists_a_not_divisible_l81_81259

theorem exists_a_not_divisible (p : ℕ) (hp_prime : Prime p) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ (p^2 ∣ (a^(p-1) - 1)) ∧ ¬ (p^2 ∣ ((a+1)^(p-1) - 1))) :=
  sorry

end exists_a_not_divisible_l81_81259


namespace expected_points_experts_over_100_games_probability_of_envelope_five_selected_l81_81404

-- Game conditions and probabilities
def game_conditions (experts_points audience_points : ℕ) : Prop :=
  experts_points = 6 ∨ audience_points = 6

noncomputable def equal_teams := (1 : ℝ) / 2

-- Expected score of Experts over 100 games
noncomputable def expected_points_experts (games : ℕ) := 465

-- Probability that envelope number 5 is chosen in the next game
noncomputable def probability_envelope_five := (12 : ℝ) / 13

theorem expected_points_experts_over_100_games : 
  expected_points_experts 100 = 465 := 
sorry

theorem probability_of_envelope_five_selected : 
  probability_envelope_five = 0.715 := 
sorry

end expected_points_experts_over_100_games_probability_of_envelope_five_selected_l81_81404


namespace isosceles_right_triangle_area_l81_81152

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l81_81152


namespace randy_blocks_left_l81_81001

theorem randy_blocks_left 
  (initial_blocks : ℕ := 78)
  (used_blocks : ℕ := 19)
  (given_blocks : ℕ := 25)
  (bought_blocks : ℕ := 36)
  (sets_from_sister : ℕ := 3)
  (blocks_per_set : ℕ := 12) :
  (initial_blocks - used_blocks - given_blocks + bought_blocks + (sets_from_sister * blocks_per_set)) / 2 = 53 := 
by
  sorry

end randy_blocks_left_l81_81001


namespace vector_sum_is_correct_l81_81799

-- Definitions for vectors a and b
def vector_a := (1, -2)
def vector_b (m : ℝ) := (2, m)

-- Condition for parallel vectors a and b
def parallel_vectors (m : ℝ) : Prop :=
  1 * m - (-2) * 2 = 0

-- Defining the target calculation for given m
def calculate_sum (m : ℝ) : ℝ × ℝ :=
  let a := vector_a
  let b := vector_b m
  (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2)

-- Statement of the theorem to be proved
theorem vector_sum_is_correct (m : ℝ) (h : parallel_vectors m) : calculate_sum m = (7, -14) :=
by sorry

end vector_sum_is_correct_l81_81799


namespace find_x_l81_81582

theorem find_x : ∃ x : ℤ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 ∧ x = 28 := 
by sorry

end find_x_l81_81582


namespace a_value_for_even_function_l81_81395

def f (x a : ℝ) := (x + 1) * (x + a)

theorem a_value_for_even_function (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = -1 :=
by
  sorry

end a_value_for_even_function_l81_81395


namespace possible_values_of_g_l81_81283

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem possible_values_of_g (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
by
  sorry

end possible_values_of_g_l81_81283


namespace square_side_length_l81_81837

theorem square_side_length :
  ∀ (w l : ℕ) (area : ℕ),
  w = 9 → l = 27 → area = w * l →
  ∃ s : ℝ, s^2 = area ∧ s = 9 * Real.sqrt 3 :=
by
  intros w l area hw hl harea
  sorry

end square_side_length_l81_81837


namespace find_missing_number_l81_81223

def average (l : List ℕ) : ℚ := l.sum / l.length

theorem find_missing_number : 
  ∃ x : ℕ, 
    average [744, 745, 747, 748, 749, 752, 752, 753, 755, x] = 750 :=
sorry

end find_missing_number_l81_81223


namespace triangle_area_l81_81738

theorem triangle_area (a b c : ℝ)
    (h1 : Polynomial.eval a (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h2 : Polynomial.eval b (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h3 : Polynomial.eval c (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (sum_roots : a + b + c = 4)
    (sum_prod_roots : a * b + a * c + b * c = 5)
    (prod_roots : a * b * c = 1):
    Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) = 1 :=
  sorry

end triangle_area_l81_81738


namespace flower_shop_options_l81_81326

theorem flower_shop_options:
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, 2 * p.1 + 3 * p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) ∧ S.card = 4 :=
sorry

end flower_shop_options_l81_81326


namespace partition_exists_l81_81544
open Set Real

theorem partition_exists (r : ℚ) (hr : r > 1) :
  ∃ (A B : ℕ → Prop), (∀ n, A n ∨ B n) ∧ (∀ n, ¬(A n ∧ B n)) ∧ 
  (∀ k l, A k → A l → (k : ℚ) / (l : ℚ) ≠ r) ∧ 
  (∀ k l, B k → B l → (k : ℚ) / (l : ℚ) ≠ r) :=
sorry

end partition_exists_l81_81544


namespace equal_real_roots_quadratic_l81_81687

theorem equal_real_roots_quadratic (k : ℝ) : (∀ x : ℝ, (x^2 + 2*x + k = 0)) → k = 1 :=
by
sorry

end equal_real_roots_quadratic_l81_81687


namespace perimeter_inequality_l81_81515

-- Define the problem parameters
variables {R S : ℝ}  -- radius and area of the inscribed polygon
variables (P : ℝ)    -- perimeter of the convex polygon formed by chosen points

-- Define the various conditions
def circle_with_polygon (r : ℝ) := r > 0 -- Circle with positive radius
def polygon_with_area (s : ℝ) := s > 0 -- Polygon with positive area

-- Main theorem to be proven
theorem perimeter_inequality (hR : circle_with_polygon R) (hS : polygon_with_area S) :
  P ≥ (2 * S / R) :=
sorry

end perimeter_inequality_l81_81515


namespace expr1_simplified_expr2_simplified_l81_81788

variable (a x : ℝ)

theorem expr1_simplified : (-a^3 + (-4 * a^2) * a) = -5 * a^3 := 
by
  sorry

theorem expr2_simplified : (-x^2 * (-x)^2 * (-x^2)^3 - 2 * x^10) = -x^10 := 
by
  sorry

end expr1_simplified_expr2_simplified_l81_81788


namespace greatest_value_y_l81_81450

theorem greatest_value_y (y : ℝ) (hy : 11 = y^2 + 1/y^2) : y + 1/y ≤ Real.sqrt 13 :=
sorry

end greatest_value_y_l81_81450


namespace simplify_complex_fraction_l81_81923

theorem simplify_complex_fraction : 
  ∀ (i : ℂ), 
  i^2 = -1 → 
  (2 - 2 * i) / (3 + 4 * i) = -(2 / 25 : ℝ) - (14 / 25) * i :=
by
  intros
  sorry

end simplify_complex_fraction_l81_81923


namespace arithmetic_progression_correct_l81_81340

noncomputable def nth_term_arithmetic_progression (n : ℕ) : ℝ :=
  4.2 * n + 9.3

def recursive_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  a 1 = 13.5 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n + 4.2

theorem arithmetic_progression_correct (n : ℕ) :
  (nth_term_arithmetic_progression n = 4.2 * n + 9.3) ∧
  ∀ (a : ℕ → ℝ), recursive_arithmetic_progression a → a n = 4.2 * n + 9.3 :=
by
  sorry

end arithmetic_progression_correct_l81_81340


namespace inequality_solution_set_l81_81522

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x) + 2

lemma f_increasing {x₁ x₂ : ℝ} (hx₁ : 1 ≤ x₁) (hx₂ : 1 ≤ x₂) (h : x₁ < x₂) : f x₁ < f x₂ := sorry

lemma solve_inequality (x : ℝ) (hx : 1 ≤ x) : (2 * x - 1 / 2 < x + 1007) → (f (2 * x - 1 / 2) < f (x + 1007)) := sorry

theorem inequality_solution_set {x : ℝ} : (1 ≤ x) → (2 * x - 1 / 2 < x + 1007) ↔ (3 / 4 ≤ x ∧ x < 2015 / 2) := sorry

end inequality_solution_set_l81_81522


namespace shaded_area_correct_l81_81545

def unit_triangle_area : ℕ := 10

def small_shaded_area : ℕ := unit_triangle_area

def medium_shaded_area : ℕ := 6 * unit_triangle_area

def large_shaded_area : ℕ := 7 * unit_triangle_area

def total_shaded_area : ℕ :=
  small_shaded_area + medium_shaded_area + large_shaded_area

theorem shaded_area_correct : total_shaded_area = 110 := 
  by
    sorry

end shaded_area_correct_l81_81545


namespace trajectory_and_min_area_l81_81079

theorem trajectory_and_min_area (C : ℝ → ℝ → Prop) (P : ℝ × ℝ → Prop)
  (l : ℝ → ℝ) (F : ℝ × ℝ) (M : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (k : ℝ) : 
  (∀ x y, P (x, y) ↔ x ^ 2 = 4 * y) → 
  P (0, 1) →
  (∀ y, l y = -1) →
  F = (0, 1) →
  (∀ x1 y1 x2 y2, x1 + x2 = 4 * k → x1 * x2 = -4 →
    M (x1, y1) (x2, y2) = (2 * k, -1)) →
  (min_area : ℝ) → 
  min_area = 4 :=
by
  intros
  sorry

end trajectory_and_min_area_l81_81079


namespace repeated_three_digit_divisible_l81_81337

theorem repeated_three_digit_divisible (μ : ℕ) (h : 100 ≤ μ ∧ μ < 1000) :
  ∃ k : ℕ, (1000 * μ + μ) = k * 7 * 11 * 13 := by
sorry

end repeated_three_digit_divisible_l81_81337


namespace seq_eleven_l81_81883

noncomputable def seq (n : ℕ) : ℤ := sorry

axiom seq_add (p q : ℕ) (hp : 0 < p) (hq : 0 < q) : seq (p + q) = seq p + seq q
axiom seq_two : seq 2 = -6

theorem seq_eleven : seq 11 = -33 := by
  sorry

end seq_eleven_l81_81883


namespace bob_equals_alice_l81_81951

-- Define conditions as constants
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25

-- Bob's total calculation
def bob_total : ℝ := (original_price * (1 + tax_rate)) * (1 - discount_rate)

-- Alice's total calculation
def alice_total : ℝ := (original_price * (1 - discount_rate)) * (1 + tax_rate)

-- Theorem statement to be proved
theorem bob_equals_alice : bob_total = alice_total := by sorry

end bob_equals_alice_l81_81951


namespace prime_condition_l81_81709

theorem prime_condition (p : ℕ) (hp : Nat.Prime p) (h2p : Nat.Prime (p + 2)) : p = 3 ∨ 6 ∣ (p + 1) := 
sorry

end prime_condition_l81_81709


namespace servings_of_popcorn_l81_81442

theorem servings_of_popcorn (popcorn_per_serving : ℕ) (jared_consumption : ℕ)
    (friend_consumption : ℕ) (num_friends : ℕ) :
    popcorn_per_serving = 30 →
    jared_consumption = 90 →
    friend_consumption = 60 →
    num_friends = 3 →
    (jared_consumption + num_friends * friend_consumption) / popcorn_per_serving = 9 := 
by
  intros h1 h2 h3 h4
  sorry

end servings_of_popcorn_l81_81442


namespace similar_triangle_perimeter_l81_81331

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (T : Triangle) : Prop :=
  T.a = T.b ∨ T.a = T.c ∨ T.b = T.c

def similar_triangles (T1 T2 : Triangle) : Prop :=
  T1.a / T2.a = T1.b / T2.b ∧ T1.b / T2.b = T1.c / T2.c ∧ T1.a / T2.a = T1.c / T2.c

noncomputable def perimeter (T : Triangle) : ℝ :=
  T.a + T.b + T.c

theorem similar_triangle_perimeter
  (T1 T2 : Triangle)
  (T1_isosceles : is_isosceles T1)
  (T1_sides : T1.a = 7 ∧ T1.b = 7 ∧ T1.c = 12)
  (T2_similar : similar_triangles T1 T2)
  (T2_longest_side : T2.c = 30) :
  perimeter T2 = 65 :=
by
  sorry

end similar_triangle_perimeter_l81_81331


namespace foci_distance_l81_81127

open Real

-- Defining parameters and conditions
variables (a : ℝ) (b : ℝ) (c : ℝ)
  (F1 F2 A B : ℝ × ℝ) -- Foci and points A, B
  (hyp_cavity : c ^ 2 = a ^ 2 + b ^ 2)
  (perimeters_eq : dist A B = 3 * a ∧ dist A F1 + dist B F1 = dist B F1 + dist B F2 + dist F1 F2)
  (distance_property : dist A F2 - dist A F1 = 2 * a)
  (c_value : c = 2 * a) -- Derived from hyperbolic definition
  
-- Main theorem to prove the distance between foci
theorem foci_distance : dist F1 F2 = 4 * a :=
  sorry

end foci_distance_l81_81127


namespace find_AX_l81_81861

theorem find_AX (AB AC BC : ℝ) (CX_bisects_ACB : Prop) (h1 : AB = 50) (h2 : AC = 28) (h3 : BC = 56) : AX = 50 / 3 :=
by
  -- Proof can be added here
  sorry

end find_AX_l81_81861


namespace first_player_winning_strategy_l81_81630

-- Definitions based on conditions
def initial_position (m n : ℕ) : ℕ × ℕ := (m - 1, n - 1)

-- Main theorem statement
theorem first_player_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (initial_position m n).fst ≠ (initial_position m n).snd ↔ m ≠ n :=
by
  sorry

end first_player_winning_strategy_l81_81630


namespace quadratic_has_real_roots_l81_81954

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 - 4 * x - 2 * k + 8 = 0) ->
  k ≥ 2 :=
by
  sorry

end quadratic_has_real_roots_l81_81954


namespace girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l81_81328

-- Definition of the primary condition
def girls := 3
def boys := 5

-- Statement for each part of the problem
theorem girls_together (A : ℕ → ℕ → ℕ) : 
  A (girls + boys - 1) girls * A girls girls = 4320 := 
sorry

theorem girls_separated (A : ℕ → ℕ → ℕ) : 
  A boys boys * A (girls + boys - 1) girls = 14400 := 
sorry

theorem girls_not_both_ends (A : ℕ → ℕ → ℕ) : 
  A boys 2 * A (girls + boys - 2) (girls + boys - 2) = 14400 := 
sorry

theorem girls_not_both_ends_simul (P : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ) : 
  P (girls + boys) (girls + boys) - A girls 2 * A (girls + boys - 2) (girls + boys - 2) = 36000 := 
sorry

end girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l81_81328


namespace edge_length_of_cube_l81_81523

/-- Define the total paint volume, remaining paint and cube paint volume -/
def total_paint_volume : ℕ := 25 * 40
def remaining_paint : ℕ := 271
def cube_paint_volume : ℕ := total_paint_volume - remaining_paint

/-- Define the volume of the cube and the statement for edge length of the cube -/
theorem edge_length_of_cube (s : ℕ) : s^3 = cube_paint_volume → s = 9 :=
by
  have h1 : cube_paint_volume = 729 := by rfl
  sorry

end edge_length_of_cube_l81_81523


namespace adjusted_smallest_part_proof_l81_81215

theorem adjusted_smallest_part_proof : 
  ∀ (x : ℝ), 14 * x = 100 → x + 12 = 19 + 1 / 7 := 
by
  sorry

end adjusted_smallest_part_proof_l81_81215


namespace tan_a_div_tan_b_l81_81725

variable {a b : ℝ}

-- Conditions
axiom sin_a_plus_b : Real.sin (a + b) = 1/2
axiom sin_a_minus_b : Real.sin (a - b) = 1/4

-- Proof statement (without the explicit proof)
theorem tan_a_div_tan_b : (Real.tan a) / (Real.tan b) = 3 := by
  sorry

end tan_a_div_tan_b_l81_81725


namespace sufficient_not_necessary_condition_l81_81329

theorem sufficient_not_necessary_condition (x : ℝ) : (|x - 1/2| < 1/2) → (x^3 < 1) ∧ ¬(x^3 < 1) → (|x - 1/2| < 1/2) :=
sorry

end sufficient_not_necessary_condition_l81_81329


namespace counter_represents_number_l81_81892

theorem counter_represents_number (a b : ℕ) : 10 * a + b = 10 * a + b := 
by 
  sorry

end counter_represents_number_l81_81892


namespace geometric_sequence_sum_l81_81021

theorem geometric_sequence_sum (S : ℕ → ℚ) (n : ℕ) 
  (hS_n : S n = 54) 
  (hS_2n : S (2 * n) = 60) 
  : S (3 * n) = 60 + 2 / 3 := 
sorry

end geometric_sequence_sum_l81_81021


namespace find_C_l81_81166

noncomputable def A : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a}
def isSolutionC (C : Set ℝ) : Prop := C = {2, 3}

theorem find_C : ∃ C : Set ℝ, isSolutionC C ∧ ∀ a, (A ∪ B a = A) ↔ a ∈ C :=
by
  sorry

end find_C_l81_81166


namespace no_digit_satisfies_equations_l81_81949

-- Define the conditions as predicates.
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x < 10

-- Formulate the proof problem based on the given problem conditions and conclusion
theorem no_digit_satisfies_equations : 
  ¬ (∃ x : ℤ, is_digit x ∧ (x - (10 * x + x) = 801 ∨ x - (10 * x + x) = 812)) :=
by
  sorry

end no_digit_satisfies_equations_l81_81949


namespace range_of_b_div_a_l81_81573

theorem range_of_b_div_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
(h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end range_of_b_div_a_l81_81573


namespace remainder_of_product_mod_17_l81_81250

theorem remainder_of_product_mod_17 :
  (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 :=
sorry

end remainder_of_product_mod_17_l81_81250


namespace part1_part2_l81_81759

theorem part1 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin α - Real.cos α = 7 / 5 := sorry

theorem part2 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin (2 * α + Real.pi / 3) = -12 / 25 - 7 * Real.sqrt 3 / 50 := sorry

end part1_part2_l81_81759


namespace plane_through_points_eq_l81_81501

-- Define the points M, N, P
def M := (1, 2, 0)
def N := (1, -1, 2)
def P := (0, 1, -1)

-- Define the target plane equation
def target_plane_eq (x y z : ℝ) := 5 * x - 2 * y + 3 * z - 1 = 0

-- Main theorem statement
theorem plane_through_points_eq :
  ∀ (x y z : ℝ),
    (∃ A B C : ℝ,
      A * (x - 1) + B * (y - 2) + C * z = 0 ∧
      A * (1 - 1) + B * (-1 - 2) + C * (2 - 0) = 0 ∧
      A * (0 - 1) + B * (1 - 2) + C * (-1 - 0) = 0) →
    target_plane_eq x y z :=
by
  sorry

end plane_through_points_eq_l81_81501


namespace sum_of_digits_l81_81907

theorem sum_of_digits (a b c d : ℕ) (h1 : a + c = 11) (h2 : b + c = 9) (h3 : a + d = 10) (h_d : d - c = 1) : 
  a + b + c + d = 21 :=
sorry

end sum_of_digits_l81_81907


namespace find_special_integers_l81_81226

theorem find_special_integers 
  : ∃ n : ℕ, 100 ≤ n ∧ n ≤ 1997 ∧ (2^n + 2) % n = 0 ∧ (n = 66 ∨ n = 198 ∨ n = 398 ∨ n = 798) :=
by
  sorry

end find_special_integers_l81_81226


namespace second_job_hourly_wage_l81_81786

-- Definitions based on conditions
def total_wages : ℕ := 160
def first_job_wages : ℕ := 52
def second_job_hours : ℕ := 12

-- Proof statement
theorem second_job_hourly_wage : 
  (total_wages - first_job_wages) / second_job_hours = 9 :=
by
  sorry

end second_job_hourly_wage_l81_81786


namespace students_in_class_l81_81844

theorem students_in_class (n : ℕ) 
  (h1 : 15 = 15)
  (h2 : ∃ m, n = m + 20 - 1)
  (h3 : ∃ x : ℕ, x = 3) :
  n = 38 :=
by
  sorry

end students_in_class_l81_81844


namespace sequence_an_correct_l81_81815

theorem sequence_an_correct (S_n : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S_n n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n, n ≥ 2 → a n = 2 * n - 1) :=
by
  -- We assume S_n is defined such that S_n = n^2 + 1
  -- From this, we have to show that:
  -- for n = 1, a_1 = 2,
  -- and for n ≥ 2, a_n = 2n - 1
  sorry

end sequence_an_correct_l81_81815


namespace total_games_correct_l81_81553

noncomputable def number_of_games_per_month : ℕ := 13
noncomputable def number_of_months_in_season : ℕ := 14
noncomputable def total_games_in_season : ℕ := number_of_games_per_month * number_of_months_in_season

theorem total_games_correct : total_games_in_season = 182 := by
  sorry

end total_games_correct_l81_81553


namespace divisible_by_77_l81_81217

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k :=
by
  sorry

end divisible_by_77_l81_81217


namespace only_nice_number_is_three_l81_81386

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def nice (n : ℕ) : Prop :=
  ∃ (xs ys : ℕ → ℕ), 
    xs 1 = 1 ∧ ys 1 = 3 ∧
    (∀ k, xs (k+1) = P (xs k) ∧ ys (k+1) = Q (ys k) ∨ xs (k+1) = Q (xs k) ∧ ys (k+1) = P (ys k)) ∧
    xs n = ys n

theorem only_nice_number_is_three (n : ℕ) : nice n ↔ n = 3 :=
by
  sorry

end only_nice_number_is_three_l81_81386


namespace transform_identity_l81_81536

theorem transform_identity (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 := 
sorry

end transform_identity_l81_81536


namespace min_red_hair_students_l81_81032

theorem min_red_hair_students (B N R : ℕ) 
  (h1 : B + N + R = 50)
  (h2 : N ≥ B - 1)
  (h3 : R ≥ N - 1) :
  R = 17 := sorry

end min_red_hair_students_l81_81032


namespace original_cards_l81_81865

-- Define the number of cards Jason gave away
def cards_given_away : ℕ := 9

-- Define the number of cards Jason now has
def cards_now : ℕ := 4

-- Prove the original number of Pokemon cards Jason had
theorem original_cards (x : ℕ) : x = cards_given_away + cards_now → x = 13 :=
by {
    sorry
}

end original_cards_l81_81865


namespace number_of_sequences_l81_81489

theorem number_of_sequences : 
  let n : ℕ := 7
  let ones : ℕ := 5
  let twos : ℕ := 2
  let comb := Nat.choose
  (ones + twos = n) ∧  
  comb (ones + 1) twos + comb (ones + 1) (twos - 1) = 21 := 
  by sorry

end number_of_sequences_l81_81489


namespace locus_of_point_M_l81_81695

open Real

def distance (x y: ℝ × ℝ): ℝ :=
  ((x.1 - y.1)^2 + (x.2 - y.2)^2)^(1/2)

theorem locus_of_point_M :
  (∀ (M : ℝ × ℝ), 
     distance M (2, 0) + 1 = abs (M.1 + 3)) 
  → ∀ (M : ℝ × ℝ), M.2^2 = 8 * M.1 :=
sorry

end locus_of_point_M_l81_81695


namespace domain_of_function_l81_81142

theorem domain_of_function :
  {x : ℝ | x < -1 ∨ 4 ≤ x} = {x : ℝ | (x^2 - 7*x + 12) / (x^2 - 2*x - 3) ≥ 0} \ {3} :=
by
  sorry

end domain_of_function_l81_81142


namespace awards_distribution_count_l81_81391

-- Define the problem conditions
def num_awards : Nat := 5
def num_students : Nat := 3

-- Verify each student gets at least one award
def each_student_gets_at_least_one (distributions : List (List Nat)) : Prop :=
  ∀ (dist : List Nat), dist ∈ distributions → (∀ (d : Nat), d > 0)

-- Define the main theorem to be proved
theorem awards_distribution_count :
  ∃ (distributions : List (List Nat)), each_student_gets_at_least_one distributions ∧ distributions.length = 150 :=
sorry

end awards_distribution_count_l81_81391


namespace simple_interest_sum_l81_81591

variable {P R : ℝ}

theorem simple_interest_sum :
  P * (R + 6) = P * R + 3000 → P = 500 :=
by
  intro h
  sorry

end simple_interest_sum_l81_81591


namespace total_chocolate_bars_in_large_box_l81_81579

def large_box_contains_18_small_boxes : ℕ := 18
def small_box_contains_28_chocolate_bars : ℕ := 28

theorem total_chocolate_bars_in_large_box :
  (large_box_contains_18_small_boxes * small_box_contains_28_chocolate_bars) = 504 := 
by
  sorry

end total_chocolate_bars_in_large_box_l81_81579


namespace gcd_1680_1683_l81_81866

theorem gcd_1680_1683 :
  ∀ (n : ℕ), n = 1683 →
  (∀ m, (m = 5 ∨ m = 67 ∨ m = 8) → n % m = 3) →
  (∃ d, d > 1 ∧ d ∣ 1683 ∧ d = Nat.gcd 1680 n ∧ Nat.gcd 1680 n = 3) :=
by
  sorry

end gcd_1680_1683_l81_81866


namespace largest_of_four_l81_81454

theorem largest_of_four : 
  let a := 1 
  let b := 0 
  let c := |(-2)| 
  let d := -3 
  max (max (max a b) c) d = c := by
  sorry

end largest_of_four_l81_81454


namespace bus_stops_per_hour_l81_81784

-- Define the speeds as constants
def speed_excluding_stoppages : ℝ := 60
def speed_including_stoppages : ℝ := 50

-- Formulate the main theorem
theorem bus_stops_per_hour :
  (1 - speed_including_stoppages / speed_excluding_stoppages) * 60 = 10 := 
by
  sorry

end bus_stops_per_hour_l81_81784


namespace square_inscription_l81_81463

theorem square_inscription (a b : ℝ) (s1 s2 : ℝ)
  (h_eq_side_smaller : s1 = 4)
  (h_eq_side_larger : s2 = 3 * Real.sqrt 2)
  (h_sum_segments : a + b = s2)
  (h_eq_sum_squares : a^2 + b^2 = (4 * Real.sqrt 2)^2) :
  a * b = -7 := 
by sorry

end square_inscription_l81_81463


namespace range_of_ab_l81_81469

-- Given two positive numbers a and b such that ab = a + b + 3, we need to prove ab ≥ 9.

theorem range_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = a + b + 3) : 9 ≤ a * b :=
by
  sorry

end range_of_ab_l81_81469


namespace swim_distance_l81_81131

theorem swim_distance 
  (v c d : ℝ)
  (h₁ : c = 2)
  (h₂ : (d / (v + c) = 5))
  (h₃ : (25 / (v - c) = 5)) :
  d = 45 :=
by
  sorry

end swim_distance_l81_81131


namespace determine_a_l81_81312

theorem determine_a : ∀ (a b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) → (a = 3 ∨ a = 7) :=
by
  sorry

end determine_a_l81_81312


namespace eccentricity_of_ellipse_l81_81335

open Real

noncomputable def ellipse_eccentricity : ℝ :=
  let a : ℝ := 4
  let b : ℝ := 2 * sqrt 3
  let c : ℝ := sqrt (a^2 - b^2)
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (ha : a = 4) (hb : b = 2 * sqrt 3) (h_eq : ∀ A B : ℝ, |A - B| = b^2 / 2 → |A - 2 * sqrt 3| + |B - 2 * sqrt 3| ≤ 10) :
  ellipse_eccentricity = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l81_81335


namespace alien_heads_l81_81505

theorem alien_heads (l o : ℕ) 
  (h1 : l + o = 60) 
  (h2 : 4 * l + o = 129) : 
  l + 2 * o = 97 := 
by 
  sorry

end alien_heads_l81_81505


namespace central_cell_value_l81_81014

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l81_81014


namespace find_x_of_series_eq_15_l81_81102

noncomputable def infinite_series (x : ℝ) : ℝ :=
  5 + (5 + x) / 3 + (5 + 2 * x) / 3^2 + (5 + 3 * x) / 3^3 + ∑' n, (5 + (n + 1) * x) / 3 ^ (n + 1)

theorem find_x_of_series_eq_15 (x : ℝ) (h : infinite_series x = 15) : x = 10 :=
sorry

end find_x_of_series_eq_15_l81_81102


namespace min_value_frac_l81_81390

open Real

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 1) :
  (1 / a + 2 / b) = 9 + 4 * sqrt 2 :=
sorry

end min_value_frac_l81_81390


namespace cost_of_bought_movie_l81_81890

theorem cost_of_bought_movie 
  (ticket_cost : ℝ)
  (ticket_count : ℕ)
  (rental_cost : ℝ)
  (total_spent : ℝ)
  (bought_movie_cost : ℝ) :
  ticket_cost = 10.62 →
  ticket_count = 2 →
  rental_cost = 1.59 →
  total_spent = 36.78 →
  bought_movie_cost = total_spent - (ticket_cost * ticket_count + rental_cost) →
  bought_movie_cost = 13.95 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_bought_movie_l81_81890


namespace super_rare_snake_cost_multiple_l81_81978

noncomputable def price_of_regular_snake : ℕ := 250
noncomputable def total_money_obtained : ℕ := 2250
noncomputable def number_of_snakes : ℕ := 3
noncomputable def eggs_per_snake : ℕ := 2

theorem super_rare_snake_cost_multiple :
  (total_money_obtained - (number_of_snakes * eggs_per_snake - 1) * price_of_regular_snake) / price_of_regular_snake = 4 :=
by
  sorry

end super_rare_snake_cost_multiple_l81_81978


namespace speed_in_still_water_l81_81374

-- Define the velocities (speeds)
def speed_downstream (V_w V_s : ℝ) : ℝ := V_w + V_s
def speed_upstream (V_w V_s : ℝ) : ℝ := V_w - V_s

-- Define the given conditions
def downstream_condition (V_w V_s : ℝ) : Prop := speed_downstream V_w V_s = 9
def upstream_condition (V_w V_s : ℝ) : Prop := speed_upstream V_w V_s = 1

-- The main theorem to prove
theorem speed_in_still_water (V_s V_w : ℝ) (h1 : downstream_condition V_w V_s) (h2 : upstream_condition V_w V_s) : V_w = 5 :=
  sorry

end speed_in_still_water_l81_81374


namespace min_value_of_y_min_value_achieved_l81_81840

noncomputable def y (x : ℝ) : ℝ := x + 1/x + 16*x / (x^2 + 1)

theorem min_value_of_y : ∀ x > 1, y x ≥ 8 :=
  sorry

theorem min_value_achieved : ∃ x, (x > 1) ∧ (y x = 8) :=
  sorry

end min_value_of_y_min_value_achieved_l81_81840


namespace evaluate_expression_l81_81353

theorem evaluate_expression (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + (a⁻¹ / 3)) / a = 10 / 27 :=
by 
  sorry

end evaluate_expression_l81_81353


namespace range_of_a_l81_81893

theorem range_of_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2)
    (hxy : x * y = 2) (h : ∀ x y, 2 - x ≥ a / (4 - y)) : a ≤ 0 :=
sorry

end range_of_a_l81_81893


namespace pq_false_l81_81306

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x > 3 ↔ x^2 > 9
def q (a b : ℝ) : Prop := a^2 > b^2 ↔ a > b

-- Theorem to prove that p ∨ q is false given the conditions
theorem pq_false (x a b : ℝ) (hp : ¬ p x) (hq : ¬ q a b) : ¬ (p x ∨ q a b) :=
by
  sorry

end pq_false_l81_81306


namespace triangle_area_ratios_l81_81530

theorem triangle_area_ratios (K : ℝ) 
  (hCD : ∃ AC, ∃ CD, CD = AC / 4) 
  (hAE : ∃ AB, ∃ AE, AE = AB / 5) 
  (hBF : ∃ BC, ∃ BF, BF = BC / 3) :
  ∃ area_N1N2N3, area_N1N2N3 = (8 / 15) * K :=
by
  sorry

end triangle_area_ratios_l81_81530


namespace negative_values_of_x_l81_81921

theorem negative_values_of_x : 
  let f (x : ℤ) := Int.sqrt (x + 196)
  ∃ (n : ℕ), (f (n ^ 2 - 196) > 0 ∧ f (n ^ 2 - 196) = n) ∧ ∃ k : ℕ, k = 13 :=
by
  sorry

end negative_values_of_x_l81_81921


namespace correct_model_is_pakistan_traditional_l81_81355

-- Given definitions
def hasPrimitiveModel (country : String) : Prop := country = "Nigeria"
def hasTraditionalModel (country : String) : Prop := country = "India" ∨ country = "Pakistan" ∨ country = "Nigeria"
def hasModernModel (country : String) : Prop := country = "China"

-- The proposition to prove
theorem correct_model_is_pakistan_traditional :
  (hasPrimitiveModel "Nigeria")
  ∧ (hasModernModel "China")
  ∧ (hasTraditionalModel "India")
  ∧ (hasTraditionalModel "Pakistan") →
  (hasTraditionalModel "Pakistan") := by
  intros h
  exact (h.right.right.right)

end correct_model_is_pakistan_traditional_l81_81355


namespace solve_abs_equation_l81_81913

theorem solve_abs_equation (x : ℝ) :
  (|2 * x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10 / 3) :=
by sorry

end solve_abs_equation_l81_81913


namespace tan_alpha_equals_one_l81_81184

theorem tan_alpha_equals_one (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β))
  : Real.tan α = 1 := 
by
  sorry

end tan_alpha_equals_one_l81_81184


namespace quadratic_minimum_value_l81_81173

theorem quadratic_minimum_value :
  ∀ (x : ℝ), (x - 1)^2 + 2 ≥ 2 :=
by
  sorry

end quadratic_minimum_value_l81_81173


namespace sum_of_angles_l81_81210

-- Definitions of acute, right, and obtuse angles
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_right (θ : ℝ) : Prop := θ = 90
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The main statement we want to prove
theorem sum_of_angles :
  (∀ (α β : ℝ), is_acute α ∧ is_acute β → is_acute (α + β) ∨ is_right (α + β) ∨ is_obtuse (α + β)) ∧
  (∀ (α β : ℝ), is_acute α ∧ is_right β → is_obtuse (α + β)) :=
by sorry

end sum_of_angles_l81_81210


namespace projectile_reaches_40_at_first_time_l81_81042

theorem projectile_reaches_40_at_first_time : ∃ t : ℝ, 0 < t ∧ (40 = -16 * t^2 + 64 * t) ∧ (∀ t' : ℝ, 0 < t' ∧ t' < t → ¬ (40 = -16 * t'^2 + 64 * t')) ∧ t = 0.8 :=
by
  sorry

end projectile_reaches_40_at_first_time_l81_81042


namespace price_of_first_oil_l81_81053

variable {x : ℝ}
variable {price1 volume1 price2 volume2 mix_price mix_volume : ℝ}

theorem price_of_first_oil:
  volume1 = 10 →
  price2 = 68 →
  volume2 = 5 →
  mix_volume = 15 →
  mix_price = 56 →
  (volume1 * x + volume2 * price2 = mix_volume * mix_price) →
  x = 50 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h1 : volume1 = 10 := h1
  have h2 : price2 = 68 := h2
  have h3 : volume2 = 5 := h3
  have h4 : mix_volume = 15 := h4
  have h5 : mix_price = 56 := h5
  have h6 : volume1 * x + volume2 * price2 = mix_volume * mix_price := h6
  sorry

end price_of_first_oil_l81_81053


namespace machine_A_time_to_produce_x_boxes_l81_81787

-- Definitions of the conditions
def machine_A_rate (T : ℕ) (x : ℕ) : ℚ := x / T
def machine_B_rate (x : ℕ) : ℚ := 2 * x / 5
def combined_rate (T : ℕ) (x : ℕ) : ℚ := (x / 2) 

-- The theorem statement
theorem machine_A_time_to_produce_x_boxes (x : ℕ) : 
  ∀ T : ℕ, 20 * (machine_A_rate T x + machine_B_rate x) = 10 * x → T = 10 :=
by
  intros T h
  sorry

end machine_A_time_to_produce_x_boxes_l81_81787


namespace total_animals_made_it_to_shore_l81_81238

def boat (total_sheep total_cows total_dogs sheep_drowned cows_drowned dogs_saved : Nat) : Prop :=
  cows_drowned = sheep_drowned * 2 ∧
  dogs_saved = total_dogs ∧
  total_sheep + total_cows + total_dogs - sheep_drowned - cows_drowned = 35

theorem total_animals_made_it_to_shore :
  boat 20 10 14 3 6 14 :=
by
  sorry

end total_animals_made_it_to_shore_l81_81238


namespace polygon_with_120_degree_interior_angle_has_6_sides_l81_81135

theorem polygon_with_120_degree_interior_angle_has_6_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (sum_interior_angles : ℕ) = (n-2) * 180 / n ∧ (each_angle : ℕ) = 120) : n = 6 :=
by
  sorry

end polygon_with_120_degree_interior_angle_has_6_sides_l81_81135


namespace supplement_of_complementary_angle_of_35_deg_l81_81919

theorem supplement_of_complementary_angle_of_35_deg :
  let A := 35
  let C := 90 - A
  let S := 180 - C
  S = 125 :=
by
  let A := 35
  let C := 90 - A
  let S := 180 - C
  -- we need to prove S = 125
  sorry

end supplement_of_complementary_angle_of_35_deg_l81_81919


namespace ninety_eight_times_ninety_eight_l81_81213

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 := 
by
  sorry

end ninety_eight_times_ninety_eight_l81_81213


namespace determineHairColors_l81_81782

structure Person where
  name : String
  hairColor : String

def Belokurov : Person := { name := "Belokurov", hairColor := "" }
def Chernov : Person := { name := "Chernov", hairColor := "" }
def Ryzhev : Person := { name := "Ryzhev", hairColor := "" }

-- Define the possible hair colors
def Blonde : String := "Blonde"
def Brunette : String := "Brunette"
def RedHaired : String := "Red-Haired"

-- Define the conditions based on the problem statement
axiom hairColorConditions :
  Belokurov.hairColor ≠ Blonde ∧
  Belokurov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ Blonde ∧
  ∀ p : Person, p.hairColor = Brunette → p.name ≠ "Belokurov"

-- Define the uniqueness condition that each person has a different hair color
axiom uniqueHairColors :
  Belokurov.hairColor ≠ Chernov.hairColor ∧
  Belokurov.hairColor ≠ Ryzhev.hairColor ∧
  Chernov.hairColor ≠ Ryzhev.hairColor

-- Define the proof problem
theorem determineHairColors :
  Belokurov.hairColor = RedHaired ∧
  Chernov.hairColor = Blonde ∧
  Ryzhev.hairColor = Brunette := by
  sorry

end determineHairColors_l81_81782


namespace construct_triangle_given_side_and_medians_l81_81304

theorem construct_triangle_given_side_and_medians
  (AB : ℝ) (m_a m_b : ℝ)
  (h1 : AB > 0) (h2 : m_a > 0) (h3 : m_b > 0) :
  ∃ (A B C : ℝ × ℝ),
    (∃ G : ℝ × ℝ, 
      dist A B = AB ∧ 
      dist A G = (2 / 3) * m_a ∧
      dist B G = (2 / 3) * m_b ∧ 
      dist G (midpoint ℝ A C) = m_b / 3 ∧ 
      dist G (midpoint ℝ B C) = m_a / 3) :=
sorry

end construct_triangle_given_side_and_medians_l81_81304


namespace average_ducks_l81_81008

theorem average_ducks (a e k : ℕ) 
  (h1 : a = 2 * e) 
  (h2 : e = k - 45) 
  (h3 : a = 30) :
  (a + e + k) / 3 = 35 :=
by
  sorry

end average_ducks_l81_81008


namespace yellow_crayons_count_l81_81083

def red_crayons := 14
def blue_crayons := red_crayons + 5
def yellow_crayons := 2 * blue_crayons - 6

theorem yellow_crayons_count : yellow_crayons = 32 := by
  sorry

end yellow_crayons_count_l81_81083


namespace coordinates_of_A_l81_81334

/-- The initial point A and the transformations applied to it -/
def initial_point : Prod ℤ ℤ := (-3, 2)

def translate_right (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1 + units, p.2)

def translate_down (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1, p.2 - units)

/-- Proof that the point A' has coordinates (1, -1) -/
theorem coordinates_of_A' : 
  translate_down (translate_right initial_point 4) 3 = (1, -1) :=
by
  sorry

end coordinates_of_A_l81_81334


namespace find_a_l81_81665

noncomputable def f (a x : ℝ) := a * x + 1 / Real.sqrt 2

theorem find_a (a : ℝ) (h_pos : 0 < a) (h : f a (f a (1 / Real.sqrt 2)) = f a 0) : a = 0 :=
by
  sorry

end find_a_l81_81665


namespace geometric_sequence_150th_term_l81_81681

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem geometric_sequence_150th_term :
  geometric_sequence 8 (-1 / 2) 150 = -8 * (1 / 2) ^ 149 :=
by
  -- This is the proof placeholder
  sorry

end geometric_sequence_150th_term_l81_81681


namespace whole_numbers_count_between_cubic_roots_l81_81496

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l81_81496


namespace cos_squared_sin_pi_over_2_plus_alpha_l81_81720

variable (α : ℝ)

-- Given conditions
def cond1 : Prop := (Real.pi / 2) < α * Real.pi
def cond2 : Prop := Real.cos α = -3 / 5

-- Proof goal
theorem cos_squared_sin_pi_over_2_plus_alpha :
  cond1 α → cond2 α →
  (Real.cos (Real.sin (Real.pi / 2 + α)))^2 = 8 / 25 :=
by
  intro h1 h2
  sorry

end cos_squared_sin_pi_over_2_plus_alpha_l81_81720


namespace copper_tin_ratio_l81_81308

theorem copper_tin_ratio 
    (w1 w2 w_new : ℝ) 
    (r1_copper r1_tin r2_copper r2_tin : ℝ) 
    (r_new_copper r_new_tin : ℝ)
    (pure_copper : ℝ)
    (h1 : w1 = 10)
    (h2 : w2 = 16)
    (h3 : r1_copper = 4 / 5 * w1)
    (h4 : r1_tin = 1 / 5 * w1)
    (h5 : r2_copper = 1 / 4 * w2)
    (h6 : r2_tin = 3 / 4 * w2)
    (h7 : r_new_copper = r1_copper + r2_copper + pure_copper)
    (h8 : r_new_tin = r1_tin + r2_tin)
    (h9 : w_new = 35)
    (h10 : r_new_copper + r_new_tin + pure_copper = w_new)
    (h11 : pure_copper = 9) :
    r_new_copper / r_new_tin = 3 / 2 :=
by
  sorry

end copper_tin_ratio_l81_81308


namespace branches_on_fourth_tree_l81_81640

theorem branches_on_fourth_tree :
  ∀ (height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot : ℕ),
    height_1 = 50 →
    branches_1 = 200 →
    height_2 = 40 →
    branches_2 = 180 →
    height_3 = 60 →
    branches_3 = 180 →
    height_4 = 34 →
    avg_branches_per_foot = 4 →
    (height_4 * avg_branches_per_foot = 136) :=
by
  intros height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot
  intros h1_eq_50 b1_eq_200 h2_eq_40 b2_eq_180 h3_eq_60 b3_eq_180 h4_eq_34 avg_eq_4
  -- We assume the conditions of the problem are correct, so add them to the context
  have height1 := h1_eq_50
  have branches1 := b1_eq_200
  have height2 := h2_eq_40
  have branches2 := b2_eq_180
  have height3 := h3_eq_60
  have branches3 := b3_eq_180
  have height4 := h4_eq_34
  have avg_branches := avg_eq_4
  -- Now prove the desired result
  sorry

end branches_on_fourth_tree_l81_81640


namespace max_k_for_ineq_l81_81801

theorem max_k_for_ineq (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 > (m + n)^2) :
  m^3 + n^3 ≥ (m + n)^2 + 10 :=
sorry

end max_k_for_ineq_l81_81801


namespace chess_group_players_l81_81375

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_group_players_l81_81375


namespace perpendicular_line_through_intersection_l81_81560

theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), (x + y - 2 = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (4 * x - 3 * y - 1 = 0) :=
sorry

end perpendicular_line_through_intersection_l81_81560


namespace sin_square_alpha_minus_pi_div_4_l81_81524

theorem sin_square_alpha_minus_pi_div_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α - Real.pi / 4) ^ 2 = 1 / 6 := 
sorry

end sin_square_alpha_minus_pi_div_4_l81_81524


namespace no_solutions_Y_l81_81277

theorem no_solutions_Y (Y : ℕ) : 2 * Y + Y + 3 * Y = 14 ↔ false :=
by 
  sorry

end no_solutions_Y_l81_81277


namespace solve_problem_l81_81468

variable (f : ℝ → ℝ)

axiom f_property : ∀ x : ℝ, f (x + 1) = x^2 - 2 * x

theorem solve_problem : f 2 = -1 :=
by
  sorry

end solve_problem_l81_81468


namespace tom_tickets_l81_81611

theorem tom_tickets :
  (45 + 38 + 52) - (12 + 23) = 100 := by
sorry

end tom_tickets_l81_81611


namespace avg_one_fourth_class_l81_81549

variable (N : ℕ) (A : ℕ)
variable (h1 : ((N : ℝ) * 80) = (N / 4) * A + (3 * N / 4) * 76)

theorem avg_one_fourth_class : A = 92 :=
by
  sorry

end avg_one_fourth_class_l81_81549


namespace cost_per_square_meter_of_mat_l81_81050

theorem cost_per_square_meter_of_mat {L W E : ℝ} : 
  L = 20 → W = 15 → E = 57000 → (E / (L * W)) = 190 :=
by
  intros hL hW hE
  rw [hL, hW, hE]
  sorry

end cost_per_square_meter_of_mat_l81_81050


namespace time_needed_n_l81_81065

variable (n : Nat)
variable (d : Nat := n - 1)
variable (s : ℚ := 2 / 3 * (d))
variable (time_third_mile : ℚ := 3)
noncomputable def time_needed (n : Nat) : ℚ := (3 * (n - 1)) / 2

theorem time_needed_n: 
  (∀ (n : Nat), n > 2 → time_needed n = (3 * (n - 1)) / 2) :=
by
  intros n hn
  sorry

end time_needed_n_l81_81065


namespace simplify_div_expression_l81_81914

theorem simplify_div_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2 * x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 :=
sorry

end simplify_div_expression_l81_81914


namespace problem_statement_l81_81862

open Set

def M : Set ℝ := {x | x^2 - 2008 * x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

theorem problem_statement (a b : ℝ) :
  (M ∪ N a b = univ) →
  (M ∩ N a b = {x | 2009 < x ∧ x ≤ 2010}) →
  (a = 2009 ∧ b = 2010) :=
by
  sorry

end problem_statement_l81_81862


namespace contrapositive_of_x_squared_gt_1_l81_81744

theorem contrapositive_of_x_squared_gt_1 (x : ℝ) (h : x ≤ 1) : x^2 ≤ 1 :=
sorry

end contrapositive_of_x_squared_gt_1_l81_81744


namespace james_gave_away_one_bag_l81_81012

theorem james_gave_away_one_bag (initial_marbles : ℕ) (bags : ℕ) (marbles_left : ℕ) (h1 : initial_marbles = 28) (h2 : bags = 4) (h3 : marbles_left = 21) : (initial_marbles / bags) = (initial_marbles - marbles_left) / (initial_marbles / bags) :=
by
  sorry

end james_gave_away_one_bag_l81_81012


namespace tanner_remaining_money_l81_81964
-- Import the entire Mathlib library

-- Define the conditions using constants
def s_Sep : ℕ := 17
def s_Oct : ℕ := 48
def s_Nov : ℕ := 25
def v_game : ℕ := 49

-- Define the total amount left and prove it equals 41
theorem tanner_remaining_money :
  (s_Sep + s_Oct + s_Nov - v_game) = 41 :=
by { sorry }

end tanner_remaining_money_l81_81964


namespace Smith_gave_Randy_l81_81430

theorem Smith_gave_Randy {original_money Randy_keeps gives_Sally Smith_gives : ℕ}
  (h1: original_money = 3000)
  (h2: Randy_keeps = 2000)
  (h3: gives_Sally = 1200)
  (h4: Randy_keeps + gives_Sally = original_money + Smith_gives) :
  Smith_gives = 200 :=
by
  sorry

end Smith_gave_Randy_l81_81430


namespace cos_sq_half_diff_eq_csquared_over_a2_b2_l81_81257

theorem cos_sq_half_diff_eq_csquared_over_a2_b2
  (a b c α β : ℝ)
  (h1 : a^2 + b^2 ≠ 0)
  (h2 : a * (Real.cos α) + b * (Real.sin α) = c)
  (h3 : a * (Real.cos β) + b * (Real.sin β) = c)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  Real.cos (α - β) / 2 = c^2 / (a^2 + b^2) :=
by
  sorry

end cos_sq_half_diff_eq_csquared_over_a2_b2_l81_81257


namespace problem_equivalence_l81_81448

variables (P Q : Prop)

theorem problem_equivalence :
  (P ↔ Q) ↔ ((P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q)) :=
by sorry

end problem_equivalence_l81_81448


namespace greatest_possible_sum_example_sum_case_l81_81258

/-- For integers x and y such that x^2 + y^2 = 50, the greatest possible value of x + y is 10. -/
theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 :=
sorry

-- Auxiliary theorem to state that 10 can be achieved
theorem example_sum_case : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 10 :=
sorry

end greatest_possible_sum_example_sum_case_l81_81258


namespace bank1_more_advantageous_l81_81378

-- Define the quarterly interest rate for Bank 1
def bank1_quarterly_rate : ℝ := 0.8

-- Define the annual interest rate for Bank 2
def bank2_annual_rate : ℝ := 9.0

-- Define the annual compounded interest rate for Bank 1
def bank1_annual_yield : ℝ :=
  (1 + bank1_quarterly_rate) ^ 4

-- Define the annual rate directly for Bank 2
def bank2_annual_yield : ℝ :=
  1 + bank2_annual_rate

-- The theorem stating that Bank 1 is more advantageous than Bank 2
theorem bank1_more_advantageous : bank1_annual_yield > bank2_annual_yield :=
  sorry

end bank1_more_advantageous_l81_81378


namespace round_24_6375_to_nearest_tenth_l81_81235

def round_to_nearest_tenth (n : ℚ) : ℚ :=
  let tenths := (n * 10).floor / 10
  let hundredths := (n * 100).floor % 10
  if hundredths < 5 then tenths else (tenths + 0.1)

theorem round_24_6375_to_nearest_tenth :
  round_to_nearest_tenth 24.6375 = 24.6 :=
by
  sorry

end round_24_6375_to_nearest_tenth_l81_81235


namespace smaller_number_of_product_l81_81389

theorem smaller_number_of_product :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5610 ∧ a = 34 :=
by
  -- Proof would go here
  sorry

end smaller_number_of_product_l81_81389


namespace pairs_count_l81_81157

theorem pairs_count (A B : Set ℕ) (h1 : A ∪ B = {1, 2, 3, 4, 5}) (h2 : 3 ∈ A ∩ B) : 
  Nat.card {p : Set ℕ × Set ℕ | p.1 ∪ p.2 = {1, 2, 3, 4, 5} ∧ 3 ∈ p.1 ∩ p.2} = 81 := by
  sorry

end pairs_count_l81_81157


namespace range_of_M_l81_81729

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
    ( (1 / a - 1) * (1 / b - 1) * (1 / c - 1) )  ≥ 8 := 
  sorry

end range_of_M_l81_81729


namespace BethsHighSchoolStudents_l81_81554

-- Define the variables
variables (B P : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := B = 4 * P
def condition2 : Prop := B + P = 5000

-- The theorem to be proved
theorem BethsHighSchoolStudents (h1 : condition1 B P) (h2 : condition2 B P) : B = 4000 :=
by
  -- Proof will be here
  sorry

end BethsHighSchoolStudents_l81_81554


namespace smallest_pos_integer_for_frac_reducible_l81_81322

theorem smallest_pos_integer_for_frac_reducible :
  ∃ n : ℕ, n > 0 ∧ ∃ d > 1, d ∣ (n - 17) ∧ d ∣ (6 * n + 8) ∧ n = 127 :=
by
  sorry

end smallest_pos_integer_for_frac_reducible_l81_81322


namespace fraction_halfway_l81_81078

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l81_81078


namespace geometric_seq_arith_seq_problem_l81_81349

theorem geometric_seq_arith_seq_problem (a : ℕ → ℝ) (q : ℝ)
  (h : ∀ n, a (n + 1) = q * a n)
  (h_q_pos : q > 0)
  (h_arith : 2 * (1/2 : ℝ) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 2014 - a 2015) / (a 2016 - a 2017) = 1 / 9 := 
sorry

end geometric_seq_arith_seq_problem_l81_81349


namespace problem_1_problem_2_l81_81323

theorem problem_1 (a b c: ℝ) (h1: a > 0) (h2: b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

theorem problem_2 (a b c: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_1_problem_2_l81_81323


namespace expression_equals_negative_two_l81_81495

def f (x : ℝ) : ℝ := x^3 - x - 1
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem expression_equals_negative_two : 
  f 2023 + f' 2023 + f (-2023) - f' (-2023) = -2 :=
by
  sorry

end expression_equals_negative_two_l81_81495


namespace general_formula_l81_81880

def sum_of_terms (a : ℕ → ℕ) (n : ℕ) : ℕ := 3 / 2 * a n - 3

def sequence_term (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 6 
  else a (n - 1) * 3

theorem general_formula (a : ℕ → ℕ) (n : ℕ) :
  (∀ n, sum_of_terms a n = 3 / 2 * a n - 3) →
  (∀ n, n = 0 → a n = 6) →
  (∀ n, n > 0 → a n = a (n - 1) * 3) →
  a n = 2 * 3^n := by
  sorry

end general_formula_l81_81880


namespace num_ways_to_designated_face_l81_81361

-- Define the structure of the dodecahedron
inductive Face
| Top
| Bottom
| TopRing (n : ℕ)   -- n ranges from 1 to 5
| BottomRing (n : ℕ)  -- n ranges from 1 to 5
deriving Repr, DecidableEq

-- Define adjacency relations on Faces (simplified)
def adjacent : Face → Face → Prop
| Face.Top, Face.TopRing n          => true
| Face.TopRing n, Face.TopRing m    => (m = (n % 5) + 1) ∨ (m = ((n + 3) % 5) + 1)
| Face.TopRing n, Face.BottomRing m => true
| Face.BottomRing n, Face.BottomRing m => true
| _, _ => false

-- Predicate for specific face on the bottom ring
def designated_bottom_face (f : Face) : Prop :=
  match f with
  | Face.BottomRing 1 => true
  | _ => false

-- Define the number of ways to move from top to the designated bottom face
noncomputable def num_ways : ℕ :=
  5 + 10

-- Lean statement that represents our equivalent proof problem
theorem num_ways_to_designated_face :
  num_ways = 15 := by
  sorry

end num_ways_to_designated_face_l81_81361


namespace find_valid_pairs_l81_81262

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m : ℕ, 2 ≤ m → m ≤ p / 2 → ¬(m ∣ p)

def valid_pair (n p : ℕ) : Prop :=
  is_prime p ∧ 0 < n ∧ n ≤ 2 * p ∧ n ^ (p - 1) ∣ (p - 1) ^ n + 1

theorem find_valid_pairs (n p : ℕ) : valid_pair n p ↔ (n = 1 ∧ is_prime p) ∨ (n, p) = (2, 2) ∨ (n, p) = (3, 3) := by
  sorry

end find_valid_pairs_l81_81262


namespace no_nontrivial_solutions_l81_81400

theorem no_nontrivial_solutions :
  ∀ (x y z t : ℤ), (¬(x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0)) → ¬(x^2 = 2 * y^2 ∧ x^4 + 3 * y^4 + 27 * z^4 = 9 * t^4) :=
by
  intros x y z t h_nontrivial h_eqs
  sorry

end no_nontrivial_solutions_l81_81400


namespace largest_divisor_of_462_and_231_l81_81708

def is_factor (a b : ℕ) : Prop := a ∣ b

def largest_common_divisor (a b c : ℕ) : Prop :=
  is_factor c a ∧ is_factor c b ∧ (∀ d, (is_factor d a ∧ is_factor d b) → d ≤ c)

theorem largest_divisor_of_462_and_231 :
  largest_common_divisor 462 231 231 :=
by
  sorry

end largest_divisor_of_462_and_231_l81_81708


namespace sum_of_coefficients_sum_even_odd_coefficients_l81_81662

noncomputable def P (x : ℝ) : ℝ := (2 * x^2 - 2 * x + 1)^17 * (3 * x^2 - 3 * x + 1)^17

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

theorem sum_even_odd_coefficients :
  (P 1 + P (-1)) / 2 = (1 + 35^17) / 2 ∧ (P 1 - P (-1)) / 2 = (1 - 35^17) / 2 := by
  sorry

end sum_of_coefficients_sum_even_odd_coefficients_l81_81662


namespace total_amount_of_money_l81_81743

theorem total_amount_of_money (N50 N500 : ℕ) (h1 : N50 = 97) (h2 : N50 + N500 = 108) : 
  50 * N50 + 500 * N500 = 10350 := by
  sorry

end total_amount_of_money_l81_81743


namespace total_amount_pqr_l81_81028

theorem total_amount_pqr (p q r : ℕ) (T : ℕ) 
  (hr : r = 2 / 3 * (T - r))
  (hr_value : r = 1600) : 
  T = 4000 :=
by
  sorry

end total_amount_pqr_l81_81028


namespace two_pow_ge_n_cubed_l81_81676

theorem two_pow_ge_n_cubed (n : ℕ) : 2^n ≥ n^3 ↔ n ≥ 10 := 
by sorry

end two_pow_ge_n_cubed_l81_81676


namespace find_m_for_local_minimum_l81_81534

noncomputable def f (x m : ℝ) := x * (x - m) ^ 2

theorem find_m_for_local_minimum :
  ∃ m : ℝ, (∀ x : ℝ, (x = 1 → deriv (λ x => f x m) x = 0) ∧ 
                  (x = 1 → deriv (deriv (λ x => f x m)) x > 0)) ∧ 
            m = 1 :=
by
  sorry

end find_m_for_local_minimum_l81_81534


namespace workshop_worker_allocation_l81_81886

theorem workshop_worker_allocation :
  ∃ (x y : ℕ), 
    x + y = 22 ∧
    6 * x = 5 * y ∧
    x = 10 ∧ y = 12 :=
by
  sorry

end workshop_worker_allocation_l81_81886


namespace functional_equation_l81_81753

theorem functional_equation 
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
  (h2 : f 0 ≠ 0) :
  f 2009 = 1 :=
sorry

end functional_equation_l81_81753


namespace calc_num_int_values_l81_81459

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l81_81459


namespace range_of_x_range_of_a_l81_81310

-- Part (1): 
theorem range_of_x (x : ℝ) : 
  (a = 1) → (x^2 - 6 * a * x + 8 * a^2 < 0) → (x^2 - 4 * x + 3 ≤ 0) → (2 < x ∧ x ≤ 3) := sorry

-- Part (2):
theorem range_of_a (a : ℝ) : 
  (a ≠ 0) → (∀ x, (x^2 - 4 * x + 3 ≤ 0) → (x^2 - 6 * a * x + 8 * a^2 < 0)) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 4) := sorry

end range_of_x_range_of_a_l81_81310


namespace union_A_B_l81_81307

open Set

def A := {x : ℝ | x * (x - 2) < 3}
def B := {x : ℝ | 5 / (x + 1) ≥ 1}
def U := {x : ℝ | -1 < x ∧ x ≤ 4}

theorem union_A_B : A ∪ B = U := 
sorry

end union_A_B_l81_81307


namespace distance_to_town_l81_81762

theorem distance_to_town (fuel_efficiency : ℝ) (fuel_used : ℝ) (distance : ℝ) : 
  fuel_efficiency = 70 / 10 → 
  fuel_used = 20 → 
  distance = fuel_efficiency * fuel_used → 
  distance = 140 :=
by
  intros
  sorry

end distance_to_town_l81_81762


namespace bucket_proof_l81_81881

variable (CA : ℚ) -- capacity of Bucket A
variable (CB : ℚ) -- capacity of Bucket B
variable (SA_init : ℚ) -- initial amount of sand in Bucket A
variable (SB_init : ℚ) -- initial amount of sand in Bucket B

def bucket_conditions : Prop := 
  CB = (1 / 2) * CA ∧
  SA_init = (1 / 4) * CA ∧
  SB_init = (3 / 8) * CB

theorem bucket_proof (h : bucket_conditions CA CB SA_init SB_init) : 
  (SA_init + SB_init) / CA = 7 / 16 := 
  by sorry

end bucket_proof_l81_81881


namespace n_squared_plus_n_plus_1_is_perfect_square_l81_81018

theorem n_squared_plus_n_plus_1_is_perfect_square (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 1 = k^2) ↔ n = 0 :=
by
  sorry

end n_squared_plus_n_plus_1_is_perfect_square_l81_81018


namespace volume_ratio_l81_81795

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) : 
  C / (A + B) = 23 / 12 :=
sorry

end volume_ratio_l81_81795


namespace financier_invariant_l81_81627

theorem financier_invariant (D A : ℤ) (hD : D = 1 ∨ D = 10 * (A - 1) + D ∨ D = D - 1 + 10 * A)
  (hA : A = 0 ∨ A = A + 10 * (1 - D) ∨ A = A - 1):
  (D - A) % 11 = 1 := 
sorry

end financier_invariant_l81_81627


namespace totalCerealInThreeBoxes_l81_81730

def firstBox := 14
def secondBox := firstBox / 2
def thirdBox := secondBox + 5
def totalCereal := firstBox + secondBox + thirdBox

theorem totalCerealInThreeBoxes : totalCereal = 33 := 
by {
  sorry
}

end totalCerealInThreeBoxes_l81_81730


namespace bathroom_area_is_eight_l81_81697

def bathroomArea (length width : ℕ) : ℕ :=
  length * width

theorem bathroom_area_is_eight : bathroomArea 4 2 = 8 := 
by
  -- Proof omitted.
  sorry

end bathroom_area_is_eight_l81_81697


namespace vertex_of_parabola_l81_81367

theorem vertex_of_parabola (x : ℝ) : 
  ∀ x y : ℝ, (y = x^2 - 6 * x + 1) → (∃ h k : ℝ, y = (x - h)^2 + k ∧ h = 3 ∧ k = -8) :=
by
  -- This is to state that given the parabola equation x^2 - 6x + 1, its vertex coordinates are (3, -8).
  sorry

end vertex_of_parabola_l81_81367


namespace shane_chewed_pieces_l81_81344

theorem shane_chewed_pieces :
  ∀ (Elyse Rick Shane: ℕ),
  Elyse = 100 →
  Rick = Elyse / 2 →
  Shane = Rick / 2 →
  Shane_left = 14 →
  (Shane - Shane_left) = 11 :=
by
  intros Elyse Rick Shane Elyse_def Rick_def Shane_def Shane_left_def
  sorry

end shane_chewed_pieces_l81_81344


namespace Yoongi_stack_taller_than_Taehyung_l81_81176

theorem Yoongi_stack_taller_than_Taehyung :
  let height_A := 3
  let height_B := 3.5
  let count_A := 16
  let count_B := 14
  let total_height_A := height_A * count_A
  let total_height_B := height_B * count_B
  total_height_B > total_height_A ∧ (total_height_B - total_height_A = 1) :=
by
  sorry

end Yoongi_stack_taller_than_Taehyung_l81_81176


namespace find_z_l81_81589

theorem find_z (z : ℂ) (h : (Complex.I * z = 4 + 3 * Complex.I)) : z = 3 - 4 * Complex.I :=
by
  sorry

end find_z_l81_81589


namespace common_point_graphs_l81_81201

theorem common_point_graphs 
  (a b c d : ℝ)
  (h1 : ∃ x : ℝ, 2*a + (1 / (x - b)) = 2*c + (1 / (x - d))) :
  ∃ x : ℝ, 2*b + (1 / (x - a)) = 2*d + (1 / (x - c)) :=
by
  sorry

end common_point_graphs_l81_81201


namespace distinct_real_roots_eq_one_l81_81986

theorem distinct_real_roots_eq_one : 
  (∃ x : ℝ, |x| - 4/x = (3 * |x|) / x) ∧ 
  ¬∃ x1 x2 : ℝ, 
    x1 ≠ x2 ∧ 
    (|x1| - 4/x1 = (3 * |x1|) / x1) ∧ 
    (|x2| - 4/x2 = (3 * |x2|) / x2) :=
sorry

end distinct_real_roots_eq_one_l81_81986


namespace probability_none_A_B_C_l81_81260

-- Define the probabilities as given conditions
def P_A : ℝ := 0.25
def P_B : ℝ := 0.40
def P_C : ℝ := 0.35
def P_AB : ℝ := 0.20
def P_AC : ℝ := 0.15
def P_BC : ℝ := 0.25
def P_ABC : ℝ := 0.10

-- Prove that the probability that none of the events A, B, C occur simultaneously is 0.50
theorem probability_none_A_B_C : 1 - (P_A + P_B + P_C - P_AB - P_AC - P_BC + P_ABC) = 0.50 :=
by
  sorry

end probability_none_A_B_C_l81_81260


namespace final_price_percentage_l81_81926

theorem final_price_percentage (original_price sale_price final_price : ℝ) (h1 : sale_price = 0.9 * original_price) 
(h2 : final_price = sale_price - 0.1 * sale_price) : final_price / original_price = 0.81 :=
by
  sorry

end final_price_percentage_l81_81926


namespace triangle_inequality_sides_l81_81006

theorem triangle_inequality_sides {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : b + c > a) (triangle_ineq3 : c + a > b) : 
  |(a / b) + (b / c) + (c / a) - (b / a) - (c / b) - (a / c)| < 1 :=
  sorry

end triangle_inequality_sides_l81_81006


namespace impossible_15_cents_l81_81852

theorem impossible_15_cents (a b c d : ℕ) (ha : a ≤ 4) (hb : b ≤ 4) (hc : c ≤ 4) (hd : d ≤ 4) (h : a + b + c + d = 4) : 
  1 * a + 5 * b + 10 * c + 25 * d ≠ 15 :=
by
  sorry

end impossible_15_cents_l81_81852


namespace negation_proposition_l81_81088

theorem negation_proposition :
  (∀ x : ℝ, 0 < x → x^2 + 1 ≥ 2 * x) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proposition_l81_81088


namespace max_value_of_symmetric_function_l81_81108

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l81_81108


namespace cars_at_2023_cars_less_than_15_l81_81521

def a_recurrence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 0.9 * a n + 8

def initial_condition (a : ℕ → ℝ) : Prop :=
a 1 = 300

theorem cars_at_2023 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a) :
  a 4 = 240 :=
sorry

def shifted_geom_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - 80 = 0.9 * (a n - 80)

theorem cars_less_than_15 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a)
  (h_geom_seq : shifted_geom_seq a) :
  ∃ n, n ≥ 12 ∧ a n < 15 :=
sorry

end cars_at_2023_cars_less_than_15_l81_81521


namespace find_a_l81_81240

-- Define the sets A and B and their union
variables (a : ℕ)
def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a^2}
def C : Set ℕ := {0, 1, 2, 3, 9}

-- Define the condition and prove that it implies a = 3
theorem find_a (h : A a ∪ B a = C) : a = 3 := 
by
  sorry

end find_a_l81_81240


namespace ned_weekly_sales_l81_81563

-- Define the conditions given in the problem
def normal_mouse_price : ℝ := 120
def normal_keyboard_price : ℝ := 80
def normal_scissor_price : ℝ := 30

def lt_hand_mouse_price := normal_mouse_price * 1.3
def lt_hand_keyboard_price := normal_keyboard_price * 1.2
def lt_hand_scissor_price := normal_scissor_price * 1.5

def lt_hand_mouse_daily_sales : ℝ := 25 * lt_hand_mouse_price
def lt_hand_keyboard_daily_sales : ℝ := 10 * lt_hand_keyboard_price
def lt_hand_scissor_daily_sales : ℝ := 15 * lt_hand_scissor_price

def total_daily_sales := lt_hand_mouse_daily_sales + lt_hand_keyboard_daily_sales + lt_hand_scissor_daily_sales
def days_open_per_week : ℝ := 4

def weekly_sales := total_daily_sales * days_open_per_week

-- The theorem to prove
theorem ned_weekly_sales : weekly_sales = 22140 := by
  -- The proof is omitted
  sorry

end ned_weekly_sales_l81_81563


namespace sum_three_numbers_l81_81452

theorem sum_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 23 := by
  sorry

end sum_three_numbers_l81_81452


namespace alex_plays_with_friends_l81_81232

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l81_81232


namespace math_expression_identity_l81_81817

theorem math_expression_identity :
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 :=
by
  sorry

end math_expression_identity_l81_81817


namespace sales_decrease_percentage_l81_81779

theorem sales_decrease_percentage 
  (P S : ℝ) 
  (P_new : ℝ := 1.30 * P) 
  (R : ℝ := P * S) 
  (R_new : ℝ := 1.04 * R) 
  (x : ℝ) 
  (S_new : ℝ := S * (1 - x/100)) 
  (h1 : 1.30 * P * S * (1 - x/100) = 1.04 * P * S) : 
  x = 20 :=
by
  sorry

end sales_decrease_percentage_l81_81779


namespace factorization_from_left_to_right_l81_81696

theorem factorization_from_left_to_right (a x y b : ℝ) :
  (a * (a + 1) = a^2 + a ∨
   a^2 + 3 * a - 1 = a * (a + 3) + 1 ∨
   x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) ∨
   (a - b)^3 = -(b - a)^3) →
  (x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)) := sorry

end factorization_from_left_to_right_l81_81696


namespace dan_bought_one_candy_bar_l81_81532

-- Define the conditions
def initial_money : ℕ := 4
def cost_per_candy_bar : ℕ := 3
def money_left : ℕ := 1

-- Define the number of candy bars Dan bought
def number_of_candy_bars_bought : ℕ := (initial_money - money_left) / cost_per_candy_bar

-- Prove the number of candy bars bought is equal to 1
theorem dan_bought_one_candy_bar : number_of_candy_bars_bought = 1 := by
  sorry

end dan_bought_one_candy_bar_l81_81532


namespace final_cost_correct_l81_81118

def dozen_cost : ℝ := 18
def num_dozen : ℝ := 2.5
def discount_rate : ℝ := 0.15

def cost_before_discount : ℝ := num_dozen * dozen_cost
def discount_amount : ℝ := discount_rate * cost_before_discount

def final_cost : ℝ := cost_before_discount - discount_amount

theorem final_cost_correct : final_cost = 38.25 := by
  -- The proof would go here, but we just provide the statement.
  sorry

end final_cost_correct_l81_81118


namespace quadratic_has_two_distinct_real_roots_l81_81155

theorem quadratic_has_two_distinct_real_roots : 
  ∃ α β : ℝ, (α ≠ β) ∧ (2 * α^2 - 3 * α + 1 = 0) ∧ (2 * β^2 - 3 * β + 1 = 0) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l81_81155


namespace min_value_fraction_l81_81694

theorem min_value_fraction : ∃ (x : ℝ), (∀ y : ℝ, (y^2 + 9) / (Real.sqrt (y^2 + 5)) ≥ (9 * Real.sqrt 5) / 5)
  := sorry

end min_value_fraction_l81_81694


namespace simplify_expression_l81_81181

theorem simplify_expression (x : ℝ) : (2 * x)^3 + (3 * x) * (x^2) = 11 * x^3 := 
  sorry

end simplify_expression_l81_81181


namespace total_fuel_usage_is_250_l81_81360

-- Define John's fuel consumption per km
def fuel_consumption_per_km : ℕ := 5

-- Define the distance of the first trip
def distance_trip1 : ℕ := 30

-- Define the distance of the second trip
def distance_trip2 : ℕ := 20

-- Define the fuel usage calculation
def fuel_usage_trip1 := distance_trip1 * fuel_consumption_per_km
def fuel_usage_trip2 := distance_trip2 * fuel_consumption_per_km
def total_fuel_usage := fuel_usage_trip1 + fuel_usage_trip2

-- Prove that the total fuel usage is 250 liters
theorem total_fuel_usage_is_250 : total_fuel_usage = 250 := by
  sorry

end total_fuel_usage_is_250_l81_81360


namespace celine_buys_two_laptops_l81_81115

variable (number_of_laptops : ℕ)
variable (laptop_cost : ℕ := 600)
variable (smartphone_cost : ℕ := 400)
variable (number_of_smartphones : ℕ := 4)
variable (total_money_spent : ℕ := 3000)
variable (change_back : ℕ := 200)

def total_spent : ℕ := total_money_spent - change_back

def cost_of_laptops (n : ℕ) : ℕ := n * laptop_cost

def cost_of_smartphones (n : ℕ) : ℕ := n * smartphone_cost

theorem celine_buys_two_laptops :
  cost_of_laptops number_of_laptops + cost_of_smartphones number_of_smartphones = total_spent →
  number_of_laptops = 2 := by
  sorry

end celine_buys_two_laptops_l81_81115


namespace only_n_1_has_integer_solution_l81_81645

theorem only_n_1_has_integer_solution :
  ∀ n : ℕ, (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 := 
by 
  sorry

end only_n_1_has_integer_solution_l81_81645


namespace d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l81_81117

variable (c d : ℕ)

-- Conditions: c is a multiple of 4 and d is a multiple of 8
def is_multiple_of_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k
def is_multiple_of_8 (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k

-- Statements to prove:

-- A. d is a multiple of 4
theorem d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 d :=
sorry

-- B. c - d is a multiple of 4
theorem c_minus_d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 (c - d) :=
sorry

-- D. c - d is a multiple of 2
theorem c_minus_d_is_multiple_of_2 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : ∃ k : ℕ, c - d = 2 * k :=
sorry

end d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l81_81117


namespace sin_alpha_sol_cos_2alpha_pi4_sol_l81_81106

open Real

-- Define the main problem conditions
def cond1 (α : ℝ) := sin (α + π / 3) + sin α = 9 * sqrt 7 / 14
def range (α : ℝ) := 0 < α ∧ α < π / 3

-- Define the statement for the first problem
theorem sin_alpha_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) : sin α = 2 * sqrt 7 / 7 := 
sorry

-- Define the statement for the second problem
theorem cos_2alpha_pi4_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) (h3 : sin α = 2 * sqrt 7 / 7) : 
  cos (2 * α - π / 4) = (4 * sqrt 6 - sqrt 2) / 14 := 
sorry

end sin_alpha_sol_cos_2alpha_pi4_sol_l81_81106


namespace complement_correct_l81_81585

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}
def complement_U (M: Set ℕ) (U: Set ℕ) := {x ∈ U | x ∉ M}

theorem complement_correct : complement_U M U = {3, 4, 6} :=
by 
  sorry

end complement_correct_l81_81585


namespace sufficient_condition_l81_81126

-- Definitions:
-- 1. Arithmetic sequence with first term a_1 and common difference d
-- 2. Define the sum of the first n terms of the arithmetic sequence

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + n * d

def sum_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Conditions given in the problem:
-- Let a_6 = a_1 + 5d
-- Let a_7 = a_1 + 6d
-- Condition p: a_6 + a_7 > 0

def p (a_1 d : ℤ) : Prop := a_1 + 5 * d + a_1 + 6 * d > 0

-- Sum of first 9 terms S_9 and first 3 terms S_3
-- Condition q: S_9 >= S_3

def q (a_1 d : ℤ) : Prop := sum_first_n_terms a_1 d 9 ≥ sum_first_n_terms a_1 d 3

-- The statement to prove:
theorem sufficient_condition (a_1 d : ℤ) : (p a_1 d) -> (q a_1 d) :=
sorry

end sufficient_condition_l81_81126


namespace doug_money_l81_81643

def money_problem (J D B: ℝ) : Prop :=
  J + D + B = 68 ∧
  J = 2 * B ∧
  J = (3 / 4) * D

theorem doug_money (J D B: ℝ) (h: money_problem J D B): D = 36.27 :=
by sorry

end doug_money_l81_81643


namespace min_abs_ab_perpendicular_lines_l81_81352

theorem min_abs_ab_perpendicular_lines (a b : ℝ) (h : a * b = a ^ 2 + 1) : |a * b| = 1 :=
by sorry

end min_abs_ab_perpendicular_lines_l81_81352


namespace eliot_votes_l81_81900

theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ)
                    (h1 : randy_votes = 16)
                    (h2 : shaun_votes = 5 * randy_votes)
                    (h3 : eliot_votes = 2 * shaun_votes) :
                    eliot_votes = 160 :=
by {
  -- Proof will be conducted here
  sorry
}

end eliot_votes_l81_81900


namespace max_area_of_triangle_on_parabola_l81_81368

noncomputable def area_of_triangle_ABC (p : ℝ) : ℝ :=
  (1 / 2) * abs (3 * p^2 - 14 * p + 15)

theorem max_area_of_triangle_on_parabola :
  ∃ p : ℝ, 1 ≤ p ∧ p ≤ 3 ∧ area_of_triangle_ABC p = 2 := sorry

end max_area_of_triangle_on_parabola_l81_81368


namespace solve_equation_l81_81722

theorem solve_equation (x : ℝ) :
  (x^2 + 2*x + 1 = abs (3*x - 2)) ↔ 
  (x = (-7 + Real.sqrt 37) / 2) ∨ 
  (x = (-7 - Real.sqrt 37) / 2) :=
by
  sorry

end solve_equation_l81_81722


namespace find_percentage_l81_81116

noncomputable def percentage (P : ℝ) : Prop :=
  (P / 100) * 1265 / 6 = 354.2

theorem find_percentage : ∃ (P : ℝ), percentage P ∧ P = 168 :=
by
  sorry

end find_percentage_l81_81116


namespace determine_b_l81_81093

theorem determine_b (b : ℚ) (x y : ℚ) (h1 : x = -3) (h2 : y = 4) (h3 : 2 * b * x + (b + 2) * y = b + 6) :
  b = 2 / 3 := 
sorry

end determine_b_l81_81093


namespace number_of_positive_integers_l81_81241

theorem number_of_positive_integers (n : ℕ) (hpos : 0 < n) (h : 24 - 6 * n ≥ 12) : n = 1 ∨ n = 2 :=
sorry

end number_of_positive_integers_l81_81241


namespace gcd_90_405_l81_81557

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l81_81557


namespace area_triangle_BFC_l81_81087

-- Definitions based on conditions
def Rectangle (A B C D : Type) (AB BC CD DA : ℝ) := AB = 5 ∧ BC = 12 ∧ CD = 5 ∧ DA = 12

def PointOnDiagonal (F A C : Type) := True  -- Simplified definition as being on the diagonal
def Perpendicular (B F A C : Type) := True  -- Simplified definition as being perpendicular

-- Main theorem statement
theorem area_triangle_BFC 
  (A B C D F : Type)
  (rectangle_ABCD : Rectangle A B C D 5 12 5 12)
  (F_on_AC : PointOnDiagonal F A C)
  (BF_perpendicular_AC : Perpendicular B F A C) :
  ∃ (area : ℝ), area = 30 :=
sorry

end area_triangle_BFC_l81_81087


namespace stock_index_approximation_l81_81200

noncomputable def stock_index_after_days (initial_index : ℝ) (daily_increase : ℝ) (days : ℕ) : ℝ :=
  initial_index * (1 + daily_increase / 100) ^ (days - 1)

theorem stock_index_approximation :
  let initial_index := 2
  let daily_increase := 0.02
  let days := 100
  abs (stock_index_after_days initial_index daily_increase days - 2.041) < 0.001 :=
by
  sorry

end stock_index_approximation_l81_81200


namespace minimum_workers_needed_l81_81174

-- Definitions
def job_completion_time : ℕ := 45
def days_worked : ℕ := 9
def portion_job_done : ℚ := 1 / 5
def team_size : ℕ := 10
def job_remaining : ℚ := (1 - portion_job_done)
def days_remaining : ℕ := job_completion_time - days_worked
def daily_completion_rate_by_team : ℚ := portion_job_done / days_worked
def daily_completion_rate_per_person : ℚ := daily_completion_rate_by_team / team_size
def required_daily_rate : ℚ := job_remaining / days_remaining

-- Statement to be proven
theorem minimum_workers_needed :
  (required_daily_rate / daily_completion_rate_per_person) = 10 :=
sorry

end minimum_workers_needed_l81_81174


namespace gate_paid_more_l81_81425

def pre_booked_economy_cost : Nat := 10 * 140
def pre_booked_business_cost : Nat := 10 * 170
def total_pre_booked_cost : Nat := pre_booked_economy_cost + pre_booked_business_cost

def gate_economy_cost : Nat := 8 * 190
def gate_business_cost : Nat := 12 * 210
def gate_first_class_cost : Nat := 10 * 300
def total_gate_cost : Nat := gate_economy_cost + gate_business_cost + gate_first_class_cost

theorem gate_paid_more {gate_paid_more_cost : Nat} :
  total_gate_cost - total_pre_booked_cost = 3940 :=
by
  sorry

end gate_paid_more_l81_81425


namespace audio_cassettes_in_first_set_l81_81159

theorem audio_cassettes_in_first_set (A V : ℝ) (num_audio_cassettes : ℝ) : 
  (V = 300) → (A * num_audio_cassettes + 3 * V = 1110) → (5 * A + 4 * V = 1350) → (A = 30) → (num_audio_cassettes = 7) := 
by
  intros hV hCond1 hCond2 hA
  sorry

end audio_cassettes_in_first_set_l81_81159


namespace janet_total_pockets_l81_81659

theorem janet_total_pockets
  (total_dresses : ℕ)
  (dresses_with_pockets : ℕ)
  (dresses_with_2_pockets : ℕ)
  (dresses_with_3_pockets : ℕ)
  (pockets_from_2 : ℕ)
  (pockets_from_3 : ℕ)
  (total_pockets : ℕ)
  (h1 : total_dresses = 24)
  (h2 : dresses_with_pockets = total_dresses / 2)
  (h3 : dresses_with_2_pockets = dresses_with_pockets / 3)
  (h4 : dresses_with_3_pockets = dresses_with_pockets - dresses_with_2_pockets)
  (h5 : pockets_from_2 = 2 * dresses_with_2_pockets)
  (h6 : pockets_from_3 = 3 * dresses_with_3_pockets)
  (h7 : total_pockets = pockets_from_2 + pockets_from_3)
  : total_pockets = 32 := 
by
  sorry

end janet_total_pockets_l81_81659


namespace sufficient_but_not_necessary_for_circle_l81_81497

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (m = 0 → ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0) ∧ ¬(∀m, ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0 → m = 0) :=
 by
  sorry

end sufficient_but_not_necessary_for_circle_l81_81497


namespace statement1_statement2_l81_81599

def is_pow_of_two (a : ℕ) : Prop := ∃ n : ℕ, a = 2^(n + 1)
def in_A (a : ℕ) : Prop := is_pow_of_two a
def not_in_A (a : ℕ) : Prop := ¬ in_A a ∧ a ≠ 1

theorem statement1 : 
  ∀ (a : ℕ), in_A a → ∀ (b : ℕ), b < 2 * a - 1 → ¬ (2 * a ∣ b * (b + 1)) := 
by {
  sorry
}

theorem statement2 :
  ∀ (a : ℕ), not_in_A a → ∃ (b : ℕ), b < 2 * a - 1 ∧ (2 * a ∣ b * (b + 1)) :=
by {
  sorry
}

end statement1_statement2_l81_81599


namespace original_salary_l81_81062

theorem original_salary (S : ℝ) (h : (1.12) * (0.93) * (1.09) * (0.94) * S = 1212) : 
  S = 1212 / ((1.12) * (0.93) * (1.09) * (0.94)) :=
by
  sorry

end original_salary_l81_81062


namespace sin_330_l81_81622

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l81_81622


namespace irrational_product_rational_l81_81138

-- Definitions of irrational and rational for clarity
def irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q
def rational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement of the problem in Lean 4
theorem irrational_product_rational (a b : ℕ) (ha : irrational (Real.sqrt a)) (hb : irrational (Real.sqrt b)) :
  rational ((Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b)) :=
by
  sorry

end irrational_product_rational_l81_81138


namespace distribute_tourists_l81_81838

theorem distribute_tourists (guides tourists : ℕ) (hguides : guides = 3) (htourists : tourists = 8) :
  ∃ k, k = 5796 := by
  sorry

end distribute_tourists_l81_81838


namespace lamp_height_difference_l81_81247

def old_lamp_height : ℝ := 1
def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := new_lamp_height - old_lamp_height

theorem lamp_height_difference :
  height_difference = 1.3333333333333335 := by
  sorry

end lamp_height_difference_l81_81247


namespace factorize_expression_l81_81180

theorem factorize_expression (R : Type*) [CommRing R] (m n : R) : 
  m^2 * n - n = n * (m + 1) * (m - 1) := 
sorry

end factorize_expression_l81_81180


namespace boys_and_girls_are_equal_l81_81814

theorem boys_and_girls_are_equal (B G : ℕ) (h1 : B + G = 30)
    (h2 : ∀ b₁ b₂, b₁ ≠ b₂ → (0 ≤ b₁) ∧ (b₁ ≤ G - 1) → (0 ≤ b₂) ∧ (b₂ ≤ G - 1) → b₁ ≠ b₂)
    (h3 : ∀ g₁ g₂, g₁ ≠ g₂ → (0 ≤ g₁) ∧ (g₁ ≤ B - 1) → (0 ≤ g₂) ∧ (g₂ ≤ B - 1) → g₁ ≠ g₂) : 
    B = 15 ∧ G = 15 := by
  sorry

end boys_and_girls_are_equal_l81_81814


namespace share_of_B_l81_81791

theorem share_of_B (x : ℕ) (A B C : ℕ) (h1 : A = 3 * B) (h2 : B = C + 25)
  (h3 : A + B + C = 645) : B = 134 :=
by
  sorry

end share_of_B_l81_81791


namespace solution_set_of_x_x_plus_2_lt_3_l81_81341

theorem solution_set_of_x_x_plus_2_lt_3 :
  {x : ℝ | x*(x + 2) < 3} = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_x_x_plus_2_lt_3_l81_81341


namespace nesbitt_inequality_l81_81975

theorem nesbitt_inequality (a b c : ℝ) (h_pos1 : 0 < a) (h_pos2 : 0 < b) (h_pos3 : 0 < c) (h_abc: a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
sorry

end nesbitt_inequality_l81_81975


namespace sequence_difference_l81_81593

theorem sequence_difference (a : ℕ → ℤ) (h_rec : ∀ n : ℕ, a (n + 1) + a n = n) (h_a1 : a 1 = 2) :
  a 4 - a 2 = 1 :=
sorry

end sequence_difference_l81_81593


namespace Danica_additional_cars_l81_81592

theorem Danica_additional_cars (num_cars : ℕ) (cars_per_row : ℕ) (current_cars : ℕ) 
  (h_cars_per_row : cars_per_row = 8) (h_current_cars : current_cars = 35) :
  ∃ n, num_cars = 5 ∧ n = 40 ∧ n - current_cars = num_cars := 
by
  sorry

end Danica_additional_cars_l81_81592


namespace sales_tax_difference_l81_81856

theorem sales_tax_difference :
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  difference = 0.375 :=
by
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  exact sorry

end sales_tax_difference_l81_81856


namespace determine_number_of_20_pound_boxes_l81_81020

variable (numBoxes : ℕ) (avgWeight : ℕ) (x : ℕ) (y : ℕ)

theorem determine_number_of_20_pound_boxes 
  (h1 : numBoxes = 30) 
  (h2 : avgWeight = 18) 
  (h3 : x + y = 30) 
  (h4 : 10 * x + 20 * y = 540) : 
  y = 24 :=
  by
  sorry

end determine_number_of_20_pound_boxes_l81_81020


namespace simplify_expression_l81_81483

variable {a : ℝ}

theorem simplify_expression (h₁ : a ≠ 0) (h₂ : a ≠ -1) (h₃ : a ≠ 1) :
  ( ( (a^2 + 1) / a - 2 ) / ( (a^2 - 1) / (a^2 + a) ) ) = a - 1 :=
sorry

end simplify_expression_l81_81483


namespace athlete_with_most_stable_performance_l81_81366

def variance_A : ℝ := 0.78
def variance_B : ℝ := 0.2
def variance_C : ℝ := 1.28

theorem athlete_with_most_stable_performance : variance_B < variance_A ∧ variance_B < variance_C :=
by {
  -- Variance comparisons:
  -- 0.2 < 0.78
  -- 0.2 < 1.28
  sorry
}

end athlete_with_most_stable_performance_l81_81366


namespace even_mult_expressions_divisible_by_8_l81_81315

theorem even_mult_expressions_divisible_by_8 {a : ℤ} (h : ∃ k : ℤ, a = 2 * k) :
  (8 ∣ a * (a^2 + 20)) ∧ (8 ∣ a * (a^2 - 20)) ∧ (8 ∣ a * (a^2 - 4)) := by
  sorry

end even_mult_expressions_divisible_by_8_l81_81315


namespace exists_abc_l81_81035

theorem exists_abc (n k : ℕ) (hn : n > 20) (hk : k > 1) (hdiv : k^2 ∣ n) : 
  ∃ (a b c : ℕ), n = a * b + b * c + c * a :=
by
  sorry

end exists_abc_l81_81035


namespace polynomial_remainder_l81_81624

theorem polynomial_remainder (z : ℂ) :
  let dividend := 4*z^3 - 5*z^2 - 17*z + 4
  let divisor := 4*z + 6
  let quotient := z^2 - 4*z + (1/4 : ℝ)
  let remainder := 5*z^2 + 6*z + (5/2 : ℝ)
  dividend = divisor * quotient + remainder := sorry

end polynomial_remainder_l81_81624


namespace amusement_park_ticket_price_l81_81600

-- Conditions as definitions in Lean
def weekday_adult_ticket_cost : ℕ := 22
def weekday_children_ticket_cost : ℕ := 7
def weekend_adult_ticket_cost : ℕ := 25
def weekend_children_ticket_cost : ℕ := 10
def adult_discount_rate : ℕ := 20
def sales_tax_rate : ℕ := 10
def num_of_adults : ℕ := 2
def num_of_children : ℕ := 2

-- Correct Answer to be proved equivalent:
def expected_total_price := 66

-- Statement translating the problem to Lean proof obligation
theorem amusement_park_ticket_price :
  let cost_before_discount := (num_of_adults * weekend_adult_ticket_cost) + (num_of_children * weekend_children_ticket_cost)
  let discount := (num_of_adults * weekend_adult_ticket_cost) * adult_discount_rate / 100
  let subtotal := cost_before_discount - discount
  let sales_tax := subtotal * sales_tax_rate / 100
  let total_cost := subtotal + sales_tax
  total_cost = expected_total_price :=
by
  sorry

end amusement_park_ticket_price_l81_81600


namespace area_of_shape_l81_81253

theorem area_of_shape (x y : ℝ) (α : ℝ) (P : ℝ × ℝ) :
  (x - 2 * Real.cos α)^2 + (y - 2 * Real.sin α)^2 = 16 →
  ∃ A : ℝ, A = 32 * Real.pi :=
by
  sorry

end area_of_shape_l81_81253


namespace solve_abcd_l81_81431

theorem solve_abcd : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 - d * x| ≤ 1) ∧ 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 + a * x^2 + b * x + c| ≤ 1) →
  d = 3 ∧ b = -3 ∧ a = 0 ∧ c = 0 :=
by
  sorry

end solve_abcd_l81_81431


namespace apples_on_tree_now_l81_81297

-- Definitions based on conditions
def initial_apples : ℕ := 11
def apples_picked : ℕ := 7
def new_apples : ℕ := 2

-- Theorem statement proving the final number of apples on the tree
theorem apples_on_tree_now : initial_apples - apples_picked + new_apples = 6 := 
by 
  sorry

end apples_on_tree_now_l81_81297


namespace least_number_condition_l81_81002

-- Define the set of divisors as a constant
def divisors : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 15}

-- Define the least number that satisfies the condition
def least_number : ℕ := 125

-- The theorem stating that the least number 125 leaves a remainder of 5 when divided by the given set of numbers
theorem least_number_condition : ∀ d ∈ divisors, least_number % d = 5 :=
by
  sorry

end least_number_condition_l81_81002


namespace minimum_jumps_l81_81693

theorem minimum_jumps (dist_cm : ℕ) (jump_mm : ℕ) (dist_mm : ℕ) (cm_to_mm_conversion : dist_mm = dist_cm * 10) (leap_condition : ∃ n : ℕ, jump_mm * n ≥ dist_mm) : ∃ n : ℕ, 19 * n = 18120 → n = 954 :=
by
  sorry

end minimum_jumps_l81_81693


namespace scalene_triangle_smallest_angle_sum_l81_81034

theorem scalene_triangle_smallest_angle_sum :
  ∀ (A B C : ℝ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A = 45 ∧ C = 135 → (∃ x y : ℝ, x = y ∧ x = 45 ∧ y = 45 ∧ x + y = 90) :=
by
  intros A B C h
  sorry

end scalene_triangle_smallest_angle_sum_l81_81034


namespace mutually_exclusive_not_opposite_l81_81948

-- Define the given conditions
def boys := 6
def girls := 5
def total_students := boys + girls
def selection := 3

-- Define the mutually exclusive and not opposite events
def event_at_least_2_boys := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (b ≥ 2) ∧ (g ≤ (selection - b))
def event_at_least_2_girls := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (g ≥ 2) ∧ (b ≤ (selection - g))

-- Statement that these events are mutually exclusive but not opposite
theorem mutually_exclusive_not_opposite :
  (event_at_least_2_boys ∧ event_at_least_2_girls) → 
  (¬ ((∃ (b: ℕ) (g: ℕ), b + g = selection ∧ b ≥ 2 ∧ g ≥ 2) ∧ ¬(event_at_least_2_boys))) :=
sorry

end mutually_exclusive_not_opposite_l81_81948


namespace kid_ticket_price_l81_81350

theorem kid_ticket_price (adult_price kid_tickets tickets total_profit : ℕ) 
  (h_adult_price : adult_price = 6) 
  (h_kid_tickets : kid_tickets = 75) 
  (h_tickets : tickets = 175) 
  (h_total_profit : total_profit = 750) : 
  (total_profit - (tickets - kid_tickets) * adult_price) / kid_tickets = 2 :=
by
  sorry

end kid_ticket_price_l81_81350


namespace set_B_can_form_right_angled_triangle_l81_81988

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l81_81988


namespace unique_solution_c_eq_one_l81_81060

theorem unique_solution_c_eq_one (b c : ℝ) (hb : b > 0) 
  (h_unique_solution : ∃ x : ℝ, x^2 + (b + 1/b) * x + c = 0 ∧ 
  ∀ y : ℝ, y^2 + (b + 1/b) * y + c = 0 → y = x) : c = 1 :=
by
  sorry

end unique_solution_c_eq_one_l81_81060


namespace children_ages_l81_81436

-- Define the ages of the four children
variable (a b c d : ℕ)

-- Define the conditions
axiom h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom h2 : a + b + c + d = 31
axiom h3 : (a - 4) + (b - 4) + (c - 4) + (d - 4) = 16
axiom h4 : (a - 7) + (b - 7) + (c - 7) + (d - 7) = 8
axiom h5 : (a - 11) + (b - 11) + (c - 11) + (d - 11) = 1
noncomputable def ages : ℕ × ℕ × ℕ × ℕ := (12, 10, 6, 3)

-- The theorem to prove
theorem children_ages (h1 : a = 12) (h2 : b = 10) (h3 : c = 6) (h4 : d = 3) : a = 12 ∧ b = 10 ∧ c = 6 ∧ d = 3 :=
by sorry

end children_ages_l81_81436


namespace items_per_baggie_l81_81248

def num_pretzels : ℕ := 64
def num_suckers : ℕ := 32
def num_kids : ℕ := 16
def num_goldfish : ℕ := 4 * num_pretzels
def total_items : ℕ := num_pretzels + num_goldfish + num_suckers

theorem items_per_baggie : total_items / num_kids = 22 :=
by
  -- Calculation proof
  sorry

end items_per_baggie_l81_81248


namespace four_m0_as_sum_of_primes_l81_81561

theorem four_m0_as_sum_of_primes (m0 : ℕ) (h1 : m0 > 1) 
  (h2 : ∀ n : ℕ, ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n) 
  (h3 : ∀ p1 p2 : ℕ, Prime p1 → Prime p2 → (2 * m0 ≠ p1 + p2)) : 
  ∃ p1 p2 p3 p4 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ (4 * m0 = p1 + p2 + p3 + p4) ∨ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 4 * m0 = p1 + p2 + p3) :=
by sorry

end four_m0_as_sum_of_primes_l81_81561


namespace difference_of_cubes_l81_81981

theorem difference_of_cubes (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) : m^3 - n^3 = 1387 :=
by
  sorry

end difference_of_cubes_l81_81981


namespace find_coordinates_of_M_l81_81896

-- Definitions of the points A, B, C
def A : (ℝ × ℝ) := (2, -4)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definitions of vectors CA and CB
def vector_CA : (ℝ × ℝ) := (A.1 - C.1, A.2 - C.2)
def vector_CB : (ℝ × ℝ) := (B.1 - C.1, B.2 - C.2)

-- Definition of the point M
def M : (ℝ × ℝ) := (-11, -15)

-- Definition of vector CM
def vector_CM : (ℝ × ℝ) := (M.1 - C.1, M.2 - C.2)

-- The condition to prove
theorem find_coordinates_of_M : vector_CM = (2 * vector_CA.1 + 3 * vector_CB.1, 2 * vector_CA.2 + 3 * vector_CB.2) :=
by
  sorry

end find_coordinates_of_M_l81_81896


namespace find_z_l81_81016

-- Define the given angles
def angle_ABC : ℝ := 95
def angle_BAC : ℝ := 65

-- Define the angle sum property for triangle ABC
def angle_sum_triangle_ABC (a b : ℝ) : ℝ := 180 - (a + b)

-- Define the angle DCE as equal to angle BCA
def angle_DCE : ℝ := angle_sum_triangle_ABC angle_ABC angle_BAC

-- Define the angle sum property for right triangle CDE
def z (dce : ℝ) : ℝ := 90 - dce

-- State the theorem to be proved
theorem find_z : z angle_DCE = 70 :=
by
  -- Statement for proof is provided
  sorry

end find_z_l81_81016


namespace sum_of_reciprocals_l81_81989

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 56) : (1/x) + (1/y) = 15/56 := 
by 
  sorry

end sum_of_reciprocals_l81_81989


namespace carton_height_is_60_l81_81663

-- Definitions
def carton_length : ℕ := 30
def carton_width : ℕ := 42
def soap_length : ℕ := 7
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 360

-- Theorem Statement
theorem carton_height_is_60 (h : ℕ) (H : ∀ (layers : ℕ), layers = max_soap_boxes / ((carton_length / soap_length) * (carton_width / soap_width)) → h = layers * soap_height) : h = 60 :=
  sorry

end carton_height_is_60_l81_81663


namespace range_of_m_l81_81336

open Real

def vector_a (m : ℝ) : ℝ × ℝ := (m, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (-2 * m, m)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def not_parallel (m : ℝ) : Prop :=
  m^2 + 2 * m ≠ 0

theorem range_of_m (m : ℝ) (h1 : dot_product (vector_a m) (vector_b m) < 0) (h2 : not_parallel m) :
  m < 0 ∨ (m > (1 / 2) ∧ m ≠ -2) :=
sorry

end range_of_m_l81_81336


namespace andrew_age_l81_81879

/-- 
Andrew and his five cousins are ages 4, 6, 8, 10, 12, and 14. 
One afternoon two of his cousins whose ages sum to 18 went to the movies. 
Two cousins younger than 12 but not including the 8-year-old went to play baseball. 
Andrew and the 6-year-old stayed home. How old is Andrew?
-/
theorem andrew_age (ages : Finset ℕ) (andrew_age: ℕ)
  (h_ages : ages = {4, 6, 8, 10, 12, 14})
  (movies : Finset ℕ) (baseball : Finset ℕ)
  (h_movies1 : movies.sum id = 18)
  (h_baseball1 : ∀ x ∈ baseball, x < 12 ∧ x ≠ 8)
  (home : Finset ℕ) (h_home : home = {6, andrew_age}) :
  andrew_age = 12 :=
sorry

end andrew_age_l81_81879


namespace fill_up_mini_vans_l81_81314

/--
In a fuel station, the service costs $2.20 per vehicle and every liter of fuel costs $0.70.
Assume that mini-vans have a tank size of 65 liters, and trucks have a tank size of 143 liters.
Given that 2 trucks were filled up and the total cost was $347.7,
prove the number of mini-vans filled up is 3.
-/
theorem fill_up_mini_vans (m : ℝ) (t : ℝ) 
    (service_cost_per_vehicle fuel_cost_per_liter : ℝ)
    (van_tank_size truck_tank_size total_cost : ℝ):
    service_cost_per_vehicle = 2.20 →
    fuel_cost_per_liter = 0.70 →
    van_tank_size = 65 →
    truck_tank_size = 143 →
    t = 2 →
    total_cost = 347.7 →
    (service_cost_per_vehicle * m + service_cost_per_vehicle * t) + (fuel_cost_per_liter * van_tank_size * m) + (fuel_cost_per_liter * truck_tank_size * t) = total_cost →
    m = 3 :=
by
  intros
  sorry

end fill_up_mini_vans_l81_81314


namespace find_y_l81_81671

variable (t : ℝ)
variable (x : ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := x = 3 - t
def condition2 : Prop := y = 2 * t + 11
def condition3 : Prop := x = 1

theorem find_y (h1 : condition1 x t) (h2 : condition2 t y) (h3 : condition3 x) : y = 15 := by
  sorry

end find_y_l81_81671


namespace complement_of_A_in_U_l81_81626

theorem complement_of_A_in_U :
    ∀ (U A : Set ℕ),
    U = {1, 2, 3, 4} →
    A = {1, 3} →
    (U \ A) = {2, 4} :=
by
  intros U A hU hA
  rw [hU, hA]
  sorry

end complement_of_A_in_U_l81_81626


namespace lcm_of_54_96_120_150_l81_81272

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end lcm_of_54_96_120_150_l81_81272


namespace polynomial_real_roots_l81_81531

theorem polynomial_real_roots :
  (∃ x : ℝ, x^4 - 3*x^3 - 2*x^2 + 6*x + 9 = 0) ↔ (x = 1 ∨ x = 3) := 
by
  sorry

end polynomial_real_roots_l81_81531


namespace neg_p_equiv_l81_81270

theorem neg_p_equiv (p : Prop) : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end neg_p_equiv_l81_81270


namespace total_weight_l81_81548

-- Define the weights of almonds and pecans.
def weight_almonds : ℝ := 0.14
def weight_pecans : ℝ := 0.38

-- Prove that the total weight of nuts is 0.52 kilograms.
theorem total_weight (almonds pecans : ℝ) (h_almonds : almonds = 0.14) (h_pecans : pecans = 0.38) :
  almonds + pecans = 0.52 :=
by
  sorry

end total_weight_l81_81548


namespace final_price_hat_final_price_tie_l81_81291

theorem final_price_hat (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (h_initial : initial_price = 20) 
    (h_first : first_discount = 0.25) 
    (h_second : second_discount = 0.20) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 12 := 
by 
  rw [h_initial, h_first, h_second]
  norm_num

theorem final_price_tie (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (t_initial : initial_price = 15) 
    (t_first : first_discount = 0.10) 
    (t_second : second_discount = 0.30) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 9.45 := 
by 
  rw [t_initial, t_first, t_second]
  norm_num

end final_price_hat_final_price_tie_l81_81291


namespace canned_boxes_equation_l81_81122

theorem canned_boxes_equation (x : ℕ) (h₁: x ≤ 300) :
  2 * 14 * x = 32 * (300 - x) :=
by
sorry

end canned_boxes_equation_l81_81122


namespace sum_and_product_of_roots_l81_81994

theorem sum_and_product_of_roots :
  let a := 1
  let b := -7
  let c := 12
  (∀ x: ℝ, x^2 - 7*x + 12 = 0 → (x = 3 ∨ x = 4)) →
  (-b/a = 7) ∧ (c/a = 12) := 
by
  sorry

end sum_and_product_of_roots_l81_81994


namespace second_rectangle_area_l81_81417

theorem second_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hbx : x < h):
  2 * b * x * (h - 3 * x) / h = (2 * b * x * (h - 3 * x))/h := 
sorry

end second_rectangle_area_l81_81417


namespace range_of_a_l81_81719

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

noncomputable def q (a : ℝ) : Prop :=
  a < 1 ∧ a ≠ 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0 :=
by sorry

end range_of_a_l81_81719


namespace perimeter_shaded_region_l81_81185

theorem perimeter_shaded_region (r: ℝ) (circumference: ℝ) (h1: circumference = 36) (h2: {x // x = 3 * (circumference / 6)}) : x = 18 :=
by
  sorry

end perimeter_shaded_region_l81_81185


namespace odd_function_values_l81_81267

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l81_81267


namespace annual_interest_rate_l81_81513

theorem annual_interest_rate (principal total_paid: ℝ) (h_principal : principal = 150) (h_total_paid : total_paid = 162) : 
  ((total_paid - principal) / principal) * 100 = 8 :=
by
  sorry

end annual_interest_rate_l81_81513


namespace min_value_of_quadratic_l81_81798

def quadratic_function (x : ℝ) : ℝ := x^2 + 6 * x + 13

theorem min_value_of_quadratic :
  (∃ x : ℝ, quadratic_function x = 4) ∧ (∀ y : ℝ, quadratic_function y ≥ 4) :=
sorry

end min_value_of_quadratic_l81_81798


namespace probability_picasso_consecutive_l81_81657

-- Given Conditions
def total_pieces : Nat := 12
def picasso_paintings : Nat := 4

-- Desired probability calculation
theorem probability_picasso_consecutive :
  (Nat.factorial (total_pieces - picasso_paintings + 1) * Nat.factorial picasso_paintings) / 
  Nat.factorial total_pieces = 1 / 55 :=
by
  sorry

end probability_picasso_consecutive_l81_81657


namespace solve_fraction_eq_zero_l81_81462

theorem solve_fraction_eq_zero (x : ℝ) (h : x - 2 ≠ 0) : (x + 1) / (x - 2) = 0 ↔ x = -1 :=
by
  sorry

end solve_fraction_eq_zero_l81_81462


namespace smallest_sum_of_consecutive_integers_gt_420_l81_81242

theorem smallest_sum_of_consecutive_integers_gt_420 : 
  ∃ n : ℕ, (n * (n + 1) > 420) ∧ (n + (n + 1) = 43) := sorry

end smallest_sum_of_consecutive_integers_gt_420_l81_81242


namespace arctan_tan_expression_l81_81190

noncomputable def tan (x : ℝ) : ℝ := sorry
noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem arctan_tan_expression :
  arctan (tan 65 - 2 * tan 40) = 25 := sorry

end arctan_tan_expression_l81_81190


namespace after_2_pow_2009_days_is_monday_l81_81216

-- Define the current day as Thursday
def today := "Thursday"

-- Define the modulo operation for calculating days of the week
def day_of_week_after (days : ℕ) : ℕ :=
  days % 7

-- Define the exponent in question
def exponent := 2009

-- Since today is Thursday, which we can represent as 4 (considering Sunday as 0, Monday as 1, ..., Saturday as 6)
def today_as_num := 4

-- Calculate the day after 2^2009 days
def future_day := (today_as_num + day_of_week_after (2 ^ exponent)) % 7

-- Prove that the future_day is 1 (Monday)
theorem after_2_pow_2009_days_is_monday : future_day = 1 := by
  sorry

end after_2_pow_2009_days_is_monday_l81_81216


namespace range_of_a_l81_81947

noncomputable def min_expr (x: ℝ) : ℝ := x + 2/(x - 2)

theorem range_of_a (a: ℝ) : 
  (∀ x > 2, a ≤ min_expr x) ↔ a ≤ 2 + 2 * Real.sqrt 2 := 
by
  sorry

end range_of_a_l81_81947


namespace prime_power_sum_l81_81707

theorem prime_power_sum (a b p : ℕ) (hp : p = a ^ b + b ^ a) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (hp_prime : Nat.Prime p) : 
  p = 17 := 
sorry

end prime_power_sum_l81_81707


namespace domain_of_fraction_is_all_real_l81_81961

theorem domain_of_fraction_is_all_real (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 3 * x + 4 * k ≠ 0) ↔ k < -9 / 112 :=
by sorry

end domain_of_fraction_is_all_real_l81_81961


namespace number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l81_81957

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of rooms
def total_rooms : ℕ := 2

-- For the first question, define: each room must receive exact three people
def room_receives_three_people (n m : ℕ) : Prop :=
  n = 3 ∧ m = 3

-- For the second question, define: each room must receive at least one person
def room_receives_at_least_one_person (n m : ℕ) : Prop :=
  n ≥ 1 ∧ m ≥ 1

theorem number_of_distributions_room_receives_three_people :
  ∃ (ways : ℕ), ways = 20 :=
by
  sorry

theorem number_of_distributions_room_receives_at_least_one_person :
  ∃ (ways : ℕ), ways = 62 :=
by
  sorry

end number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l81_81957


namespace simplify_fractions_l81_81705

theorem simplify_fractions :
  (240 / 20) * (6 / 180) * (10 / 4) = 1 :=
by sorry

end simplify_fractions_l81_81705


namespace planA_charge_for_8_minutes_eq_48_cents_l81_81073

theorem planA_charge_for_8_minutes_eq_48_cents
  (X : ℝ)
  (hA : ∀ t : ℝ, t ≤ 8 → X = X)
  (hB : ∀ t : ℝ, 6 * 0.08 = 0.48)
  (hEqual : 6 * 0.08 = X) :
  X = 0.48 := by
  sorry

end planA_charge_for_8_minutes_eq_48_cents_l81_81073


namespace num_baskets_l81_81439

axiom num_apples_each_basket : ℕ
axiom total_apples : ℕ

theorem num_baskets (h1 : num_apples_each_basket = 17) (h2 : total_apples = 629) : total_apples / num_apples_each_basket = 37 :=
  sorry

end num_baskets_l81_81439


namespace sampling_methods_correct_l81_81299

-- Definitions of the conditions:
def is_simple_random_sampling (method : String) : Prop := 
  method = "random selection of 24 students by the student council"

def is_systematic_sampling (method : String) : Prop := 
  method = "selection of students numbered from 001 to 240 whose student number ends in 3"

-- The equivalent math proof problem:
theorem sampling_methods_correct :
  is_simple_random_sampling "random selection of 24 students by the student council" ∧
  is_systematic_sampling "selection of students numbered from 001 to 240 whose student number ends in 3" :=
by
  sorry

end sampling_methods_correct_l81_81299


namespace usual_time_to_school_l81_81770

theorem usual_time_to_school (R : ℝ) (T : ℝ) (h : (17 / 13) * (T - 7) = T) : T = 29.75 :=
sorry

end usual_time_to_school_l81_81770


namespace simplify_expression_l81_81859

theorem simplify_expression (a b c d x y : ℝ) (h : cx ≠ -dy) :
  (cx * (b^2 * x^2 + 3 * b^2 * y^2 + a^2 * y^2) + dy * (b^2 * x^2 + 3 * a^2 * x^2 + a^2 * y^2)) / (cx + dy)
  = (b^2 + 3 * a^2) * x^2 + (a^2 + 3 * b^2) * y^2 := by
  sorry

end simplify_expression_l81_81859


namespace heather_bicycling_time_l81_81686

theorem heather_bicycling_time (distance speed : ℝ) (h_distance : distance = 40) (h_speed : speed = 8) : (distance / speed) = 5 := 
by
  rw [h_distance, h_speed]
  norm_num

end heather_bicycling_time_l81_81686


namespace no_n_in_range_l81_81528

theorem no_n_in_range :
  ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n % 7 = 10467 % 7 := by
  sorry

end no_n_in_range_l81_81528


namespace correct_equation_for_tournament_l81_81465

theorem correct_equation_for_tournament (x : ℕ) (h : x * (x - 1) / 2 = 28) : True :=
sorry

end correct_equation_for_tournament_l81_81465


namespace max_abc_l81_81776

def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

theorem max_abc (a b c n : ℕ) (hpos : n > 0) (h1 : 1 ≤ a ∧ a < 10) (h2 : 1 ≤ b ∧ b < 10) (h3 : 1 ≤ c ∧ c < 10) (h_eq : C_n c n - B_n b n = A_n a n ^ 2) :  a + b + c ≤ 18 :=
by sorry

end max_abc_l81_81776


namespace find_hours_spent_l81_81874

/-- Let 
  h : ℝ := hours Ed stayed in the hotel last night
  morning_hours : ℝ := 4 -- hours Ed stayed in the hotel this morning
  
  conditions:
  night_cost_per_hour : ℝ := 1.50 -- the cost per hour for staying at night
  morning_cost_per_hour : ℝ := 2 -- the cost per hour for staying in the morning
  initial_amount : ℝ := 80 -- initial amount Ed had
  remaining_amount : ℝ := 63 -- remaining amount after stay
  
  Then the total cost calculated by Ed is:
  total_cost : ℝ := (night_cost_per_hour * h) + (morning_cost_per_hour * morning_hours)
  spent_amount : ℝ := initial_amount - remaining_amount

  We need to prove that h = 6 given the above conditions.
-/
theorem find_hours_spent {h morning_hours night_cost_per_hour morning_cost_per_hour initial_amount remaining_amount total_cost spent_amount : ℝ}
  (hc1 : night_cost_per_hour = 1.50)
  (hc2 : morning_cost_per_hour = 2)
  (hc3 : initial_amount = 80)
  (hc4 : remaining_amount = 63)
  (hc5 : morning_hours = 4)
  (hc6 : spent_amount = initial_amount - remaining_amount)
  (hc7 : total_cost = night_cost_per_hour * h + morning_cost_per_hour * morning_hours)
  (hc8 : spent_amount = 17)
  (hc9 : total_cost = spent_amount) :
  h = 6 :=
by 
  sorry

end find_hours_spent_l81_81874


namespace shelves_fit_l81_81688

-- Define the total space of the room for the library
def totalSpace : ℕ := 400

-- Define the space each bookshelf takes up
def spacePerBookshelf : ℕ := 80

-- Define the reserved space for desk and walking area
def reservedSpace : ℕ := 160

-- Define the space available for bookshelves
def availableSpace : ℕ := totalSpace - reservedSpace

-- Define the number of bookshelves that can fit in the available space
def numberOfBookshelves : ℕ := availableSpace / spacePerBookshelf

-- The theorem stating the number of bookshelves Jonas can fit in the room
theorem shelves_fit : numberOfBookshelves = 3 := by
  -- We can defer the proof as we only need the statement for now
  sorry

end shelves_fit_l81_81688


namespace shaded_area_correct_l81_81460

noncomputable def side_length : ℝ := 24
noncomputable def radius : ℝ := side_length / 4
noncomputable def area_of_square : ℝ := side_length ^ 2
noncomputable def area_of_one_circle : ℝ := Real.pi * radius ^ 2
noncomputable def total_area_of_circles : ℝ := 5 * area_of_one_circle
noncomputable def shaded_area : ℝ := area_of_square - total_area_of_circles

theorem shaded_area_correct :
  shaded_area = 576 - 180 * Real.pi := by
  sorry

end shaded_area_correct_l81_81460


namespace problem1_problem2_problem3_problem4_l81_81036

theorem problem1 : (-4.7 : ℝ) + 0.9 = -3.8 := by
  sorry

theorem problem2 : (- (1 / 2) : ℝ) - (-(1 / 3)) = -(1 / 6) := by
  sorry

theorem problem3 : (- (10 / 9) : ℝ) * (- (6 / 10)) = (2 / 3) := by
  sorry

theorem problem4 : (0 : ℝ) * (-5) = 0 := by
  sorry

end problem1_problem2_problem3_problem4_l81_81036


namespace identify_false_statement_l81_81479

-- Definitions for the conditions
def isMultipleOf (n k : Nat) : Prop := ∃ m, n = k * m

def conditions : Prop :=
  isMultipleOf 12 2 ∧
  isMultipleOf 123 3 ∧
  isMultipleOf 1234 4 ∧
  isMultipleOf 12345 5 ∧
  isMultipleOf 123456 6

-- The statement which proves which condition is false
theorem identify_false_statement : conditions → ¬ (isMultipleOf 1234 4) :=
by
  intros h
  sorry

end identify_false_statement_l81_81479


namespace absolute_value_c_l81_81478

noncomputable def condition_polynomial (a b c : ℤ) : Prop :=
  a * (↑(Complex.ofReal 3) + Complex.I)^4 +
  b * (↑(Complex.ofReal 3) + Complex.I)^3 +
  c * (↑(Complex.ofReal 3) + Complex.I)^2 +
  b * (↑(Complex.ofReal 3) + Complex.I) +
  a = 0

noncomputable def coprime_integers (a b c : ℤ) : Prop :=
  Int.gcd (Int.gcd a b) c = 1

theorem absolute_value_c (a b c : ℤ) (h1 : condition_polynomial a b c) (h2 : coprime_integers a b c) :
  |c| = 97 :=
sorry

end absolute_value_c_l81_81478


namespace find_certain_number_l81_81672

theorem find_certain_number (a : ℤ) (certain_number : ℤ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * certain_number) : certain_number = 49 := 
sorry

end find_certain_number_l81_81672


namespace slope_of_line_l81_81642

theorem slope_of_line (x y : ℝ) (h : 4 * x - 7 * y = 28) : (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 7) :=
by
  -- Proof omitted
  sorry

end slope_of_line_l81_81642


namespace prime_sequence_constant_l81_81950

open Nat

-- Define a predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the recurrence relation
def recurrence_relation (p : ℕ → ℕ) (k : ℤ) : Prop :=
  ∀ n : ℕ, p (n + 2) = p (n + 1) + p n + k

-- Define the proof problem
theorem prime_sequence_constant (p : ℕ → ℕ) (k : ℤ) : 
  (∀ n, is_prime (p n)) →
  recurrence_relation p k →
  ∃ (q : ℕ), is_prime q ∧ (∀ n, p n = q) ∧ k = -q :=
by
  -- Sorry proof here
  sorry

end prime_sequence_constant_l81_81950


namespace polygon_area_l81_81388

theorem polygon_area (n : ℕ) (s : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : n = 24) 
  (h2 : n * s = perimeter) 
  (h3 : perimeter = 48) 
  (h4 : s = 2) 
  (h5 : area = n * s^2 / 2) : 
  area = 96 :=
by
  sorry

end polygon_area_l81_81388


namespace people_ratio_l81_81316

theorem people_ratio (pounds_coal : ℕ) (days1 : ℕ) (people1 : ℕ) (pounds_goal : ℕ) (days2 : ℕ) :
  pounds_coal = 10000 → days1 = 10 → people1 = 10 → pounds_goal = 40000 → days2 = 80 →
  (people1 * pounds_goal * days1) / (pounds_coal * days2) = 1 / 2 :=
by
  sorry

end people_ratio_l81_81316


namespace varies_fix_l81_81342

variable {x y z : ℝ}

theorem varies_fix {k j : ℝ} 
  (h1 : x = k * y^4)
  (h2 : y = j * z^(1/3)) : x = (k * j^4) * z^(4/3) := by
  sorry

end varies_fix_l81_81342


namespace minimum_jellybeans_l81_81254

theorem minimum_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n = 164 :=
by sorry

end minimum_jellybeans_l81_81254


namespace total_tiles_covering_floor_l81_81136

-- Let n be the width of the rectangle (in tiles)
-- The length would then be 2n (in tiles)
-- The total number of tiles that lie on both diagonals is given as 39

theorem total_tiles_covering_floor (n : ℕ) (H : 2 * n + 1 = 39) : 2 * n^2 = 722 :=
by sorry

end total_tiles_covering_floor_l81_81136


namespace system_sampling_arithmetic_sequence_l81_81888

theorem system_sampling_arithmetic_sequence :
  ∃ (seq : Fin 5 → ℕ), seq 0 = 8 ∧ seq 3 = 104 ∧ seq 1 = 40 ∧ seq 2 = 72 ∧ seq 4 = 136 ∧ 
    (∀ n m : Fin 5, 0 < n.val - m.val → seq n.val = seq m.val + 32 * (n.val - m.val)) :=
sorry

end system_sampling_arithmetic_sequence_l81_81888


namespace roger_coins_left_l81_81058

theorem roger_coins_left {pennies nickels dimes donated_coins initial_coins remaining_coins : ℕ} 
    (h1 : pennies = 42) 
    (h2 : nickels = 36) 
    (h3 : dimes = 15) 
    (h4 : donated_coins = 66) 
    (h5 : initial_coins = pennies + nickels + dimes) 
    (h6 : remaining_coins = initial_coins - donated_coins) : 
    remaining_coins = 27 := 
sorry

end roger_coins_left_l81_81058


namespace ratio_bones_child_to_adult_woman_l81_81572

noncomputable def num_skeletons : ℕ := 20
noncomputable def num_adult_women : ℕ := num_skeletons / 2
noncomputable def num_adult_men_and_children : ℕ := num_skeletons - num_adult_women
noncomputable def num_adult_men : ℕ := num_adult_men_and_children / 2
noncomputable def num_children : ℕ := num_adult_men_and_children / 2
noncomputable def bones_per_adult_woman : ℕ := 20
noncomputable def bones_per_adult_man : ℕ := bones_per_adult_woman + 5
noncomputable def total_bones : ℕ := 375
noncomputable def bones_per_child : ℕ := (total_bones - (num_adult_women * bones_per_adult_woman + num_adult_men * bones_per_adult_man)) / num_children

theorem ratio_bones_child_to_adult_woman : 
  (bones_per_child : ℚ) / (bones_per_adult_woman : ℚ) = 1 / 2 := by
sorry

end ratio_bones_child_to_adult_woman_l81_81572


namespace min_points_on_dodecahedron_min_points_on_icosahedron_l81_81559

-- Definitions for the dodecahedron problem
def dodecahedron_has_12_faces : Prop := true
def each_vertex_in_dodecahedron_belongs_to_3_faces : Prop := true

-- Proof statement for dodecahedron
theorem min_points_on_dodecahedron : dodecahedron_has_12_faces ∧ each_vertex_in_dodecahedron_belongs_to_3_faces → ∃ n, n = 4 :=
by
  sorry

-- Definitions for the icosahedron problem
def icosahedron_has_20_faces : Prop := true
def icosahedron_has_12_vertices : Prop := true
def each_vertex_in_icosahedron_belongs_to_5_faces : Prop := true
def vertices_of_icosahedron_grouped_into_6_pairs : Prop := true

-- Proof statement for icosahedron
theorem min_points_on_icosahedron : 
  icosahedron_has_20_faces ∧ icosahedron_has_12_vertices ∧ each_vertex_in_icosahedron_belongs_to_5_faces ∧ vertices_of_icosahedron_grouped_into_6_pairs → ∃ n, n = 6 :=
by
  sorry

end min_points_on_dodecahedron_min_points_on_icosahedron_l81_81559


namespace ratio_of_flour_to_eggs_l81_81332

theorem ratio_of_flour_to_eggs (F E : ℕ) (h1 : E = 60) (h2 : F + E = 90) : F / 30 = 1 ∧ E / 30 = 2 := by
  sorry

end ratio_of_flour_to_eggs_l81_81332


namespace find_extrema_l81_81494

theorem find_extrema (x y : ℝ) (h1 : x < 0) (h2 : -1 < y) (h3 : y < 0) : 
  max (max x (x*y)) (x*y^2) = x*y ∧ min (min x (x*y)) (x*y^2) = x :=
by sorry

end find_extrema_l81_81494


namespace conic_section_is_parabola_l81_81296

theorem conic_section_is_parabola (x y : ℝ) : y^4 - 16 * x^2 = 2 * y^2 - 64 → ((y^2 - 1)^2 = 16 * x^2 - 63) ∧ (∃ k : ℝ, y^2 = 4 * k * x + 1) :=
sorry

end conic_section_is_parabola_l81_81296


namespace a1_lt_a3_iff_an_lt_an1_l81_81945

-- Define arithmetic sequence and required properties
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℝ)

-- Define the necessary and sufficient condition theorem
theorem a1_lt_a3_iff_an_lt_an1 (h_arith : is_arithmetic_sequence a) :
  (a 1 < a 3) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end a1_lt_a3_iff_an_lt_an1_l81_81945


namespace Correct_Statement_l81_81356

theorem Correct_Statement : 
  (∀ x : ℝ, 7 * x = 4 * x - 3 → 7 * x - 4 * x = -3) ∧
  (∀ x : ℝ, (2 * x - 1) / 3 = 1 + (x - 3) / 2 → 2 * (2 * x - 1) = 6 + 3 * (x - 3)) ∧
  (∀ x : ℝ, 2 * (2 * x - 1) - 3 * (x - 3) = 1 → 4 * x - 2 - 3 * x + 9 = 1) ∧
  (∀ x : ℝ, 2 * (x + 1) = x + 7 → x = 5) :=
by
  sorry

end Correct_Statement_l81_81356


namespace cos_value_l81_81112

variable (α : ℝ)

theorem cos_value (h : Real.sin (Real.pi / 6 + α) = 1 / 3) : Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l81_81112


namespace reverse_difference_198_l81_81757

theorem reverse_difference_198 (a : ℤ) : 
  let N := 100 * (a - 1) + 10 * a + (a + 1)
  let M := 100 * (a + 1) + 10 * a + (a - 1)
  M - N = 198 := 
by
  sorry

end reverse_difference_198_l81_81757


namespace line_b_y_intercept_l81_81107

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l81_81107


namespace quad_to_square_l81_81007

theorem quad_to_square (a b z : ℝ)
  (h_dim : a = 9) 
  (h_dim2 : b = 16) 
  (h_area : a * b = z * z) :
  z = 12 :=
by
  -- Proof outline would go here, but let's skip the actual proof for this definition.
  sorry

end quad_to_square_l81_81007


namespace point_in_second_quadrant_l81_81716

theorem point_in_second_quadrant (a : ℝ) :
  ∃ q : ℕ, q = 2 ∧ (-3 : ℝ) < 0 ∧ (a^2 + 1) > 0 := 
by sorry

end point_in_second_quadrant_l81_81716


namespace A_investment_l81_81046

theorem A_investment (x : ℝ) (hx : 0 < x) :
  (∃ a b c d e : ℝ,
    a = x ∧ b = 12 ∧ c = 200 ∧ d = 6 ∧ e = 60 ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    ((a * b) / (a * b + c * d)) * 100 = e)
  → x = 150 :=
by
  sorry

end A_investment_l81_81046


namespace fraction_meaningful_iff_l81_81287

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l81_81287


namespace tangent_line_eq_l81_81109

noncomputable def equation_of_tangent_line (x y : ℝ) : Prop := 
  ∃ k : ℝ, (y = k * (x - 2) + 2) ∧ 2 * x + y - 6 = 0

theorem tangent_line_eq :
  ∀ (x y : ℝ), 
    (y = 2 / (x - 1)) ∧ (∃ (a b : ℝ), (a, b) = (1, 4)) ->
    equation_of_tangent_line x y :=
by
  sorry

end tangent_line_eq_l81_81109


namespace calculation_correct_l81_81728

def calculation : ℝ := 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123

theorem calculation_correct : calculation = 172.20 := by
  sorry

end calculation_correct_l81_81728


namespace finding_b_for_infinite_solutions_l81_81778

theorem finding_b_for_infinite_solutions :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 :=
by
  sorry

end finding_b_for_infinite_solutions_l81_81778


namespace quadratic_inequality_solution_set_l81_81103

theorem quadratic_inequality_solution_set :
  (∃ x : ℝ, 2 * x + 3 - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_set_l81_81103


namespace rectangle_area_l81_81514

theorem rectangle_area (w l : ℕ) (h1 : l = w + 8) (h2 : 2 * l + 2 * w = 176) :
  l * w = 1920 :=
by
  sorry

end rectangle_area_l81_81514


namespace ranking_sequences_l81_81230

theorem ranking_sequences
    (A D B E C : Type)
    (h_no_ties : ∀ (X Y : Type), X ≠ Y)
    (h_games : (W1 = A ∨ W1 = D) ∧ (W2 = B ∨ W2 = E) ∧ (W3 = W1 ∨ W3 = C)) :
  ∃! (n : ℕ), n = 48 := 
sorry

end ranking_sequences_l81_81230


namespace gcd_12345_6789_l81_81771

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l81_81771


namespace prism_faces_vertices_l81_81160

theorem prism_faces_vertices {L E F V : ℕ} (hE : E = 21) (hEdges : E = 3 * L) 
    (hF : F = L + 2) (hV : V = L) : F = 9 ∧ V = 7 :=
by
  sorry

end prism_faces_vertices_l81_81160


namespace find_monic_cubic_polynomial_with_root_l81_81075

-- Define the monic cubic polynomial
def Q (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 6

-- Define the root condition we need to prove
theorem find_monic_cubic_polynomial_with_root (a : ℝ) (ha : a = (5 : ℝ)^(1/3) + 1) : Q a = 0 :=
by
  -- Proof goes here (omitted)
  sorry

end find_monic_cubic_polynomial_with_root_l81_81075


namespace number_of_distinct_products_l81_81473

   -- We define the set S
   def S : Finset ℕ := {2, 3, 5, 11, 13}

   -- We define what it means to have a distinct product of two or more elements
   def distinctProducts (s : Finset ℕ) : Finset ℕ :=
     (s.powerset.filter (λ t => 2 ≤ t.card)).image (λ t => t.prod id)

   -- We state the theorem that there are exactly 26 distinct products
   theorem number_of_distinct_products : (distinctProducts S).card = 26 :=
   sorry
   
end number_of_distinct_products_l81_81473


namespace boys_in_class_l81_81043

theorem boys_in_class (total_students : ℕ) (fraction_girls : ℝ) (fraction_girls_eq : fraction_girls = 1 / 4) (total_students_eq : total_students = 160) :
  (total_students - fraction_girls * total_students = 120) :=
by
  rw [fraction_girls_eq, total_students_eq]
  -- Here, additional lines proving the steps would follow, but we use sorry for completeness.
  sorry

end boys_in_class_l81_81043


namespace remainder_of_product_l81_81658

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l81_81658


namespace max_value_abs_expression_l81_81970

noncomputable def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

theorem max_value_abs_expression (x y : ℝ) (h : circle_eq x y) : 
  ∃ t : ℝ, |3 * x + 4 * y - 3| = t ∧ t ≤ 8 :=
sorry

end max_value_abs_expression_l81_81970


namespace roberto_outfits_l81_81068

-- Define the conditions
def trousers := 5
def shirts := 8
def jackets := 4

-- Define the total number of outfits
def total_outfits : ℕ := trousers * shirts * jackets

-- The theorem stating the actual problem and answer
theorem roberto_outfits : total_outfits = 160 :=
by
  -- skip the proof for now
  sorry

end roberto_outfits_l81_81068


namespace measure_of_angle_BCD_l81_81096

-- Define angles and sides as given in the problem
variables (α β : ℝ)

-- Conditions: angles and side equalities
axiom angle_ABD_eq_BDC : α = β
axiom angle_DAB_eq_80 : α = 80
axiom side_AB_eq_AD : ∀ AB AD : ℝ, AB = AD
axiom side_DB_eq_DC : ∀ DB DC : ℝ, DB = DC

-- Prove that the measure of angle BCD is 65 degrees
theorem measure_of_angle_BCD : β = 65 :=
sorry

end measure_of_angle_BCD_l81_81096


namespace mode_and_median_of_survey_l81_81902

/-- A data structure representing the number of students corresponding to each sleep time. -/
structure SleepSurvey :=
  (time7 : ℕ)
  (time8 : ℕ)
  (time9 : ℕ)
  (time10 : ℕ)

def survey : SleepSurvey := { time7 := 6, time8 := 9, time9 := 11, time10 := 4 }

theorem mode_and_median_of_survey (s : SleepSurvey) :
  (mode=9 ∧ median = 8.5) :=
by
  -- proof would go here
  sorry

end mode_and_median_of_survey_l81_81902


namespace probability_of_region_C_l81_81621

theorem probability_of_region_C (P_A P_B P_C : ℚ) (hA : P_A = 1/3) (hB : P_B = 1/2) (hTotal : P_A + P_B + P_C = 1) : P_C = 1/6 := 
by
  sorry

end probability_of_region_C_l81_81621


namespace find_g_values_l81_81608

theorem find_g_values
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x * y) = x * g y)
  (h2 : g 1 = 30) :
  g 50 = 1500 ∧ g 0.5 = 15 :=
by
  sorry

end find_g_values_l81_81608


namespace total_books_in_classroom_l81_81980

-- Define the given conditions using Lean definitions
def num_children : ℕ := 15
def books_per_child : ℕ := 12
def additional_books : ℕ := 22

-- Define the hypothesis and the corresponding proof statement
theorem total_books_in_classroom : num_children * books_per_child + additional_books = 202 := 
by sorry

end total_books_in_classroom_l81_81980


namespace divide_talers_l81_81969

theorem divide_talers (loaves1 loaves2 : ℕ) (coins : ℕ) (loavesShared : ℕ) :
  loaves1 = 3 → loaves2 = 5 → coins = 8 → loavesShared = (loaves1 + loaves2) →
  (3 - loavesShared / 3) * coins / loavesShared = 1 ∧ (5 - loavesShared / 3) * coins / loavesShared = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end divide_talers_l81_81969


namespace find_y_of_equations_l81_81066

theorem find_y_of_equations (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 2 + 1 / x) : 
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by
  sorry

end find_y_of_equations_l81_81066


namespace find_d_l81_81003

theorem find_d (a b c d : ℝ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) (hdc : 0 < d)
  (oscillates : ∀ x, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 :=
sorry

end find_d_l81_81003


namespace find_f_neg_2017_l81_81186

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom log_function : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem find_f_neg_2017 : f (-2017) = 1 := by
  sorry

end find_f_neg_2017_l81_81186


namespace number_of_apps_needed_l81_81982

-- Definitions based on conditions
variable (cost_per_app : ℕ) (total_money : ℕ) (remaining_money : ℕ)

-- Assume the conditions given
axiom cost_app_eq : cost_per_app = 4
axiom total_money_eq : total_money = 66
axiom remaining_money_eq : remaining_money = 6

-- The goal is to determine the number of apps Lidia needs to buy
theorem number_of_apps_needed (n : ℕ) (h : total_money - remaining_money = cost_per_app * n) :
  n = 15 :=
by
  sorry

end number_of_apps_needed_l81_81982


namespace ratio_shortest_to_middle_tree_l81_81057

theorem ratio_shortest_to_middle_tree (height_tallest : ℕ) 
  (height_middle : ℕ) (height_shortest : ℕ)
  (h1 : height_tallest = 150) 
  (h2 : height_middle = (2 * height_tallest) / 3) 
  (h3 : height_shortest = 50) : 
  height_shortest / height_middle = 1 / 2 := by sorry

end ratio_shortest_to_middle_tree_l81_81057


namespace tan_ratio_l81_81427

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 7 / 3 :=
sorry

end tan_ratio_l81_81427


namespace expression_for_3_diamond_2_l81_81105

variable {a b : ℝ}

def diamond (a b : ℝ) : ℝ := 2 * a - 3 * b + a * b

theorem expression_for_3_diamond_2 (a : ℝ) :
  3 * diamond a 2 = 12 * a - 18 :=
by
  sorry

end expression_for_3_diamond_2_l81_81105


namespace proof_complement_U_A_l81_81724

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set A
def A : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := { x ∈ U | x ∉ A }

-- The theorem statement
theorem proof_complement_U_A :
  complement_U_A = {1, 5} :=
by
  -- Proof goes here
  sorry

end proof_complement_U_A_l81_81724


namespace minimize_PA_PB_l81_81701

theorem minimize_PA_PB 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (5, 1)) : 
  ∃ P : ℝ × ℝ, P = (4, 0) ∧ 
  ∀ P' : ℝ × ℝ, P'.snd = 0 → (dist P A + dist P B) ≤ (dist P' A + dist P' B) :=
sorry

end minimize_PA_PB_l81_81701


namespace percentage_difference_l81_81760

variables (P P' : ℝ)

theorem percentage_difference (h : P' = 1.25 * P) :
  ((P' - P) / P') * 100 = 20 :=
by sorry

end percentage_difference_l81_81760


namespace maximum_perimeter_triangle_area_l81_81836

-- Part 1: Maximum Perimeter
theorem maximum_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h_c : c = 2) 
  (h_C : C = Real.pi / 3) :
  (a + b + c) ≤ 6 :=
sorry

-- Part 2: Area under given trigonometric condition
theorem triangle_area (A B C a b c : ℝ) 
  (h_c : 2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C) :
  (1/2 * a * b * Real.sin C) = (2 * Real.sqrt 6) / 3 :=
sorry

end maximum_perimeter_triangle_area_l81_81836


namespace ratio_of_w_to_y_l81_81793

theorem ratio_of_w_to_y
  (w x y z : ℚ)
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 :=
by sorry

end ratio_of_w_to_y_l81_81793


namespace solution_sets_l81_81922

-- These are the hypotheses derived from the problem conditions.
structure Conditions (a b c d : ℕ) : Prop :=
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (positive_even : ∃ u v w x : ℕ, a = 2*u ∧ b = 2*v ∧ c = 2*w ∧ d = 2*x ∧ 
                   u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0)
  (sum_100 : a + b + c + d = 100)
  (third_fourth_single_digit : c < 20 ∧ d < 20)
  (sum_2000 : 12 * a + 30 * b + 52 * c = 2000)

-- The main theorem in Lean asserting that these are the only possible sets of numbers.
theorem solution_sets :
  ∃ (a b c d : ℕ), Conditions a b c d ∧
  ( 
    (a = 62 ∧ b = 14 ∧ c = 4 ∧ d = 1) ∨ 
    (a = 48 ∧ b = 22 ∧ c = 2 ∧ d = 3)
  ) :=
  sorry

end solution_sets_l81_81922


namespace escalator_walk_rate_l81_81997

theorem escalator_walk_rate (v : ℝ) : (v + 15) * 10 = 200 → v = 5 := by
  sorry

end escalator_walk_rate_l81_81997


namespace proof_height_difference_l81_81026

noncomputable def height_in_inches_between_ruby_and_xavier : Prop :=
  let janet_height_inches := 62.75
  let inch_to_cm := 2.54
  let janet_height_cm := janet_height_inches * inch_to_cm
  let charlene_height := 1.5 * janet_height_cm
  let pablo_height := charlene_height + 1.85 * 100
  let ruby_height := pablo_height - 0.5
  let xavier_height := charlene_height + 2.13 * 100 - 97.75
  let paul_height := ruby_height + 50
  let height_diff_cm := xavier_height - ruby_height
  let height_diff_inches := height_diff_cm / inch_to_cm
  height_diff_inches = -18.78

theorem proof_height_difference :
  height_in_inches_between_ruby_and_xavier :=
by
  sorry

end proof_height_difference_l81_81026


namespace sum_of_coordinates_of_B_l81_81080

theorem sum_of_coordinates_of_B (x : ℝ) (y : ℝ) 
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0,0)) 
  (hB : B = (x, 3))
  (hslope : (3 - 0) / (x - 0) = 4 / 5) :
  x + 3 = 6.75 := 
by
  sorry

end sum_of_coordinates_of_B_l81_81080


namespace calculate_expression_l81_81646

theorem calculate_expression : 
  -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5 / 2 :=
by
  sorry

end calculate_expression_l81_81646


namespace total_cost_proof_l81_81397

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l81_81397


namespace triangle_side_relation_l81_81456

-- Definitions for the conditions
variable {A B C a b c : ℝ}
variable (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variable (sides_rel : a = (B * (1 + 2 * C)).sin)
variable (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin))

-- The statement to be proven
theorem triangle_side_relation (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (sides_rel : a = (B * (1 + 2 * C)).sin)
  (trig_eq : (B.sin * (1 + 2 * C.cos)) = (2 * A.sin * C.cos + A.cos * C.sin)) :
  a = 2 * b := 
sorry

end triangle_side_relation_l81_81456


namespace probability_of_spinner_stopping_in_region_G_l81_81809

theorem probability_of_spinner_stopping_in_region_G :
  let pE := (1:ℝ) / 2
  let pF := (1:ℝ) / 4
  let y  := (1:ℝ) / 6
  let z  := (1:ℝ) / 12
  pE + pF + y + z = 1 → y = 2 * z → y = (1:ℝ) / 6 := by
  intros htotal hdouble
  sorry

end probability_of_spinner_stopping_in_region_G_l81_81809


namespace age_of_replaced_man_l81_81364

-- Definitions based on conditions
def avg_age_men (A : ℝ) := A
def age_man1 := 10
def avg_age_women := 23
def total_age_women := 2 * avg_age_women
def new_avg_age_men (A : ℝ) := A + 2

-- Proposition stating that given conditions yield the age of the other replaced man
theorem age_of_replaced_man (A M : ℝ) :
  8 * avg_age_men A - age_man1 - M + total_age_women = 8 * new_avg_age_men A + 16 →
  M = 20 :=
by
  sorry

end age_of_replaced_man_l81_81364


namespace fewest_students_possible_l81_81437

theorem fewest_students_possible : 
  ∃ n : ℕ, n % 3 = 1 ∧ n % 6 = 4 ∧ n % 8 = 5 ∧ ∀ m, m % 3 = 1 ∧ m % 6 = 4 ∧ m % 8 = 5 → n ≤ m := 
by
  sorry

end fewest_students_possible_l81_81437


namespace parenthesis_removal_correctness_l81_81438

theorem parenthesis_removal_correctness (x y z : ℝ) : 
  (x^2 - (x - y + 2 * z) ≠ x^2 - x + y - 2 * z) ∧
  (x - (-2 * x + 3 * y - 1) ≠ x + 2 * x - 3 * y + 1) ∧
  (3 * x + 2 * (x - 2 * y + 1) ≠ 3 * x + 2 * x - 4 * y + 2) ∧
  (-(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2 * x^2 - 4) :=
by
  sorry

end parenthesis_removal_correctness_l81_81438


namespace sandy_spent_money_l81_81790

theorem sandy_spent_money :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 :=
by
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  have total_spent : shorts + shirt + jacket = 33.56 := sorry
  exact total_spent

end sandy_spent_money_l81_81790


namespace calculate_area_bounded_figure_l81_81055

noncomputable def area_of_bounded_figure (R : ℝ) : ℝ :=
  (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi)

theorem calculate_area_bounded_figure (R : ℝ) :
  ∀ r, r = (R / 3) → area_of_bounded_figure R = (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi) :=
by
  intros r hr
  subst hr
  exact rfl

end calculate_area_bounded_figure_l81_81055


namespace s_mores_graham_crackers_l81_81853

def graham_crackers_per_smore (total_graham_crackers total_marshmallows : ℕ) : ℕ :=
total_graham_crackers / total_marshmallows

theorem s_mores_graham_crackers :
  let total_graham_crackers := 48
  let available_marshmallows := 6
  let additional_marshmallows := 18
  let total_marshmallows := available_marshmallows + additional_marshmallows
  graham_crackers_per_smore total_graham_crackers total_marshallows = 2 := sorry

end s_mores_graham_crackers_l81_81853


namespace coefficient_B_is_1_l81_81901

-- Definitions based on the conditions
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- Given conditions
def condition1 (A B C D : ℝ) := g A B C D (-2) = 0 
def condition2 (A B C D : ℝ) := g A B C D 0 = -1
def condition3 (A B C D : ℝ) := g A B C D 2 = 0

-- The main theorem to prove
theorem coefficient_B_is_1 (A B C D : ℝ) 
  (h1 : condition1 A B C D) 
  (h2 : condition2 A B C D) 
  (h3 : condition3 A B C D) : 
  B = 1 :=
sorry

end coefficient_B_is_1_l81_81901


namespace total_distance_hiked_l81_81542

def distance_car_to_stream : ℝ := 0.2
def distance_stream_to_meadow : ℝ := 0.4
def distance_meadow_to_campsite : ℝ := 0.1

theorem total_distance_hiked : 
  distance_car_to_stream + distance_stream_to_meadow + distance_meadow_to_campsite = 0.7 := by
  sorry

end total_distance_hiked_l81_81542


namespace part_one_part_two_l81_81313

noncomputable def f (x a : ℝ) : ℝ :=
  Real.log (1 + x) + a * Real.cos x

noncomputable def g (x : ℝ) : ℝ :=
  f x 2 - 1 / (1 + x)

theorem part_one (a : ℝ) : 
  (∀ x, f x a = Real.log (1 + x) + a * Real.cos x) ∧ 
  f 0 a = 2 ∧ 
  (∀ x, x + f (0:ℝ) a = x + 2) → 
  a = 2 := 
sorry

theorem part_two : 
  (∀ x, g x = Real.log (1 + x) + 2 * Real.cos x - 1 / (1 + x)) →
  (∃ y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) ∧ 
  (∀ x, -1 < x ∧ x < (Real.pi / 2) → g x ≠ 0) →
  (∃! y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) :=
sorry

end part_one_part_two_l81_81313


namespace part_1_part_2_l81_81044

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part_1 (a : ℝ) (h : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) : a = 2 :=
sorry

theorem part_2 (a : ℝ) (h : a = 2) : ∀ m, (∀ x, f (3 * x) a + f (x + 3) a ≥ m) ↔ m ≤ 5 / 3 :=
sorry

end part_1_part_2_l81_81044


namespace more_blue_count_l81_81633

-- Definitions based on the conditions given in the problem
def total_people : ℕ := 150
def more_green : ℕ := 95
def both_green_blue : ℕ := 35
def neither_green_blue : ℕ := 25

-- The Lean statement to prove the number of people who believe turquoise is "more blue"
theorem more_blue_count : 
  (total_people - neither_green_blue) - (more_green - both_green_blue) = 65 :=
by 
  sorry

end more_blue_count_l81_81633


namespace remainder_div_l81_81898

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 39 * k + 18) :
  N % 13 = 5 := 
by
  sorry

end remainder_div_l81_81898


namespace intersection_M_N_l81_81362

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def N : Set ℝ := { y | 0 < y }

theorem intersection_M_N : (M ∩ N) = { z | 0 < z ∧ z ≤ 2 } :=
by
  -- proof to be completed
  sorry

end intersection_M_N_l81_81362


namespace sequence_inequality_l81_81750

theorem sequence_inequality (a : ℕ → ℝ) 
  (h₀ : a 0 = 5) 
  (h₁ : ∀ n, a (n + 1) * a n - a n ^ 2 = 1) : 
  35 < a 600 ∧ a 600 < 35.1 :=
sorry

end sequence_inequality_l81_81750


namespace carter_baseball_cards_l81_81773

theorem carter_baseball_cards (M C : ℕ) (h1 : M = 210) (h2 : M = C + 58) : C = 152 :=
by
  sorry

end carter_baseball_cards_l81_81773


namespace sum_of_tetrahedron_properties_eq_14_l81_81768

-- Define the regular tetrahedron properties
def regular_tetrahedron_edges : ℕ := 6
def regular_tetrahedron_vertices : ℕ := 4
def regular_tetrahedron_faces : ℕ := 4

-- State the theorem that needs to be proven
theorem sum_of_tetrahedron_properties_eq_14 :
  regular_tetrahedron_edges + regular_tetrahedron_vertices + regular_tetrahedron_faces = 14 :=
by
  sorry

end sum_of_tetrahedron_properties_eq_14_l81_81768


namespace volume_of_cone_l81_81678

noncomputable def lateral_surface_area : ℝ := 8 * Real.pi

theorem volume_of_cone (l r h : ℝ)
  (h_lateral_surface : l * Real.pi = 2 * lateral_surface_area)
  (h_radius : l = 2 * r)
  (h_height : h = Real.sqrt (l^2 - r^2)) :
  (1/3) * Real.pi * r^2 * h = (8 * Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry

end volume_of_cone_l81_81678


namespace solution_system_of_inequalities_l81_81396

theorem solution_system_of_inequalities (x : ℝ) : 
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * (x^2) - x - 1 > 0 ↔ (-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6) :=
by {
  sorry
}

end solution_system_of_inequalities_l81_81396


namespace solve_system_l81_81022

open Classical

theorem solve_system : ∃ t : ℝ, ∀ (x y z : ℝ), 
  (x^2 - 9 * y^2 = 0 ∧ x + y + z = 0) ↔ 
  (x = 3 * t ∧ y = t ∧ z = -4 * t) 
  ∨ (x = -3 * t ∧ y = t ∧ z = 2 * t) := 
by 
  sorry

end solve_system_l81_81022


namespace no_C_makes_2C7_even_and_multiple_of_5_l81_81835

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

theorem no_C_makes_2C7_even_and_multiple_of_5 : ∀ C : ℕ, ¬(C < 10) ∨ ¬(is_even (2 * 100 + C * 10 + 7) ∧ is_multiple_of_5 (2 * 100 + C * 10 + 7)) :=
by
  intro C
  sorry

end no_C_makes_2C7_even_and_multiple_of_5_l81_81835


namespace digit_150_of_17_div_70_is_2_l81_81721

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l81_81721


namespace modulus_zero_l81_81169

/-- Given positive integers k and α such that 10k - α is also a positive integer, 
prove that the remainder when 8^(10k + α) + 6^(10k - α) - 7^(10k - α) - 2^(10k + α) is divided by 11 is 0. -/
theorem modulus_zero {k α : ℕ} (h₁ : 0 < k) (h₂ : 0 < α) (h₃ : 0 < 10 * k - α) :
  (8 ^ (10 * k + α) + 6 ^ (10 * k - α) - 7 ^ (10 * k - α) - 2 ^ (10 * k + α)) % 11 = 0 :=
by
  sorry

end modulus_zero_l81_81169


namespace automotive_test_l81_81273

theorem automotive_test (D T1 T2 T3 T_total : ℕ) (H1 : 3 * D = 180) 
  (H2 : T1 = D / 4) (H3 : T2 = D / 5) (H4 : T3 = D / 6)
  (H5 : T_total = T1 + T2 + T3) : T_total = 37 :=
  sorry

end automotive_test_l81_81273


namespace pyramid_volume_theorem_l81_81384

noncomputable def volume_of_regular_square_pyramid : ℝ := 
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * side_edge_length * Real.sqrt 3
  (1 / 3) * base_area * height

theorem pyramid_volume_theorem :
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * (side_edge_length * Real.sqrt 3)
  (1 / 3) * base_area * height = 6 := 
by
  sorry

end pyramid_volume_theorem_l81_81384


namespace find_number_eq_fifty_l81_81955

theorem find_number_eq_fifty (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 := by 
  sorry

end find_number_eq_fifty_l81_81955


namespace gina_good_tipper_l81_81781

noncomputable def calculate_tip_difference (bill_in_usd : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (low_tip_rate : ℝ) (high_tip_rate : ℝ) (conversion_rate : ℝ) : ℝ :=
  let discounted_bill := bill_in_usd * (1 - discount_rate)
  let taxed_bill := discounted_bill * (1 + tax_rate)
  let low_tip := taxed_bill * low_tip_rate
  let high_tip := taxed_bill * high_tip_rate
  let difference_in_usd := high_tip - low_tip
  let difference_in_eur := difference_in_usd * conversion_rate
  difference_in_eur * 100

theorem gina_good_tipper : calculate_tip_difference 26 0.08 0.07 0.05 0.20 0.85 = 326.33 := 
by
  sorry

end gina_good_tipper_l81_81781


namespace arithmetic_sequence_values_l81_81089

theorem arithmetic_sequence_values (a b c : ℤ) 
  (h1 : 2 * b = a + c)
  (h2 : 2 * a = b + 1)
  (h3 : 2 * c = b + 9) 
  (h4 : a + b + c = -15) :
  b = -5 ∧ a * c = 21 :=
by
  sorry

end arithmetic_sequence_values_l81_81089


namespace area_inequalities_l81_81347

noncomputable def f1 (x : ℝ) : ℝ := 1 - (1 / 2) * x
noncomputable def f2 (x : ℝ) : ℝ := 1 / (x + 1)
noncomputable def f3 (x : ℝ) : ℝ := 1 - (1 / 2) * x^2

noncomputable def S1 : ℝ := 1 - (1 / 4)
noncomputable def S2 : ℝ := Real.log 2
noncomputable def S3 : ℝ := (5 / 6)

theorem area_inequalities : S2 < S1 ∧ S1 < S3 := by
  sorry

end area_inequalities_l81_81347


namespace solve_for_a_l81_81765

theorem solve_for_a (a : ℝ) (h : a⁻¹ = (-1 : ℝ)^0) : a = 1 :=
sorry

end solve_for_a_l81_81765


namespace john_fan_usage_per_day_l81_81132

theorem john_fan_usage_per_day
  (power : ℕ := 75) -- fan's power in watts
  (energy_per_month_kwh : ℕ := 18) -- energy consumption per month in kWh
  (days_in_month : ℕ := 30) -- number of days in a month
  : (energy_per_month_kwh * 1000) / power / days_in_month = 8 := 
by
  sorry

end john_fan_usage_per_day_l81_81132


namespace max_profit_achieved_l81_81789

theorem max_profit_achieved :
  ∃ x : ℤ, 
    (x = 21) ∧ 
    (21 + 14 = 35) ∧ 
    (30 - 21 = 9) ∧ 
    (21 - 5 = 16) ∧
    (-x + 1965 = 1944) :=
by
  sorry

end max_profit_achieved_l81_81789


namespace proof_problem_l81_81162

open Real

theorem proof_problem :
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 4) →
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 16/4) →
  (∀ x : ℤ, abs x = 4 → abs (-4) = 4) →
  (∀ x : ℤ, x^2 = 16 → (-4)^2 = 16) →
  (- sqrt 16 = -4) := 
by 
  simp
  sorry

end proof_problem_l81_81162


namespace rectangle_clear_area_l81_81295

theorem rectangle_clear_area (EF FG : ℝ)
  (radius_E radius_F radius_G radius_H : ℝ) : 
  EF = 4 → FG = 6 → 
  radius_E = 2 → radius_F = 3 → radius_G = 1.5 → radius_H = 2.5 → 
  abs ((EF * FG) - (π * radius_E^2 / 4 + π * radius_F^2 / 4 + π * radius_G^2 / 4 + π * radius_H^2 / 4)) - 7.14 < 0.5 :=
by sorry

end rectangle_clear_area_l81_81295


namespace max_distance_proof_l81_81441

-- Definitions for fuel consumption rates per 100 km
def fuel_consumption_U : Nat := 20 -- liters per 100 km
def fuel_consumption_V : Nat := 25 -- liters per 100 km
def fuel_consumption_W : Nat := 5  -- liters per 100 km
def fuel_consumption_X : Nat := 10 -- liters per 100 km

-- Definitions for total available fuel
def total_fuel : Nat := 50 -- liters

-- Distance calculation
def distance (fuel_consumption : Nat) (fuel : Nat) : Nat :=
  (fuel * 100) / fuel_consumption

-- Distances
def distance_U := distance fuel_consumption_U total_fuel
def distance_V := distance fuel_consumption_V total_fuel
def distance_W := distance fuel_consumption_W total_fuel
def distance_X := distance fuel_consumption_X total_fuel

-- Maximum total distance calculation
def maximum_total_distance : Nat :=
  distance_U + distance_V + distance_W + distance_X

-- The statement to be proved
theorem max_distance_proof :
  maximum_total_distance = 1950 := by
  sorry

end max_distance_proof_l81_81441


namespace person_A_number_is_35_l81_81641

theorem person_A_number_is_35
    (A B : ℕ)
    (h1 : A + B = 8)
    (h2 : 10 * B + A - (10 * A + B) = 18) :
    10 * A + B = 35 :=
by
    sorry

end person_A_number_is_35_l81_81641


namespace master_li_speeding_l81_81274

theorem master_li_speeding (distance : ℝ) (time : ℝ) (speed_limit : ℝ) (average_speed : ℝ)
  (h_distance : distance = 165)
  (h_time : time = 2)
  (h_speed_limit : speed_limit = 80)
  (h_average_speed : average_speed = distance / time)
  (h_speeding : average_speed > speed_limit) :
  True :=
sorry

end master_li_speeding_l81_81274


namespace percentage_of_alcohol_in_new_mixture_l81_81538

def original_solution_volume : ℕ := 11
def added_water_volume : ℕ := 3
def alcohol_percentage_original : ℝ := 0.42

def total_volume : ℕ := original_solution_volume + added_water_volume
def amount_of_alcohol : ℝ := alcohol_percentage_original * original_solution_volume

theorem percentage_of_alcohol_in_new_mixture :
  (amount_of_alcohol / total_volume) * 100 = 33 := by
  sorry

end percentage_of_alcohol_in_new_mixture_l81_81538


namespace swim_team_more_people_l81_81472

theorem swim_team_more_people :
  let car1_people := 5
  let car2_people := 4
  let van1_people := 3
  let van2_people := 3
  let van3_people := 5
  let minibus_people := 10

  let car_max_capacity := 6
  let van_max_capacity := 8
  let minibus_max_capacity := 15

  let actual_people := car1_people + car2_people + van1_people + van2_people + van3_people + minibus_people
  let max_capacity := 2 * car_max_capacity + 3 * van_max_capacity + minibus_max_capacity
  (max_capacity - actual_people : ℕ) = 21 := 
  by
    sorry

end swim_team_more_people_l81_81472


namespace divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l81_81170

theorem divisible_by_6_implies_divisible_by_2 :
  ∀ (n : ℤ), (6 ∣ n) → (2 ∣ n) :=
by sorry

theorem not_divisible_by_2_implies_not_divisible_by_6 :
  ∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n) :=
by sorry

theorem equivalence_of_propositions :
  (∀ (n : ℤ), (6 ∣ n) → (2 ∣ n)) ↔ (∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n)) :=
by sorry


end divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l81_81170


namespace units_digit_of_fraction_l81_81941

theorem units_digit_of_fraction :
  let numer := 30 * 31 * 32 * 33 * 34 * 35
  let denom := 1000
  (numer / denom) % 10 = 6 :=
by
  sorry

end units_digit_of_fraction_l81_81941


namespace solve_y_l81_81594

theorem solve_y (y : ℝ) (h : (y ^ (7 / 8)) = 4) : y = 2 ^ (16 / 7) :=
sorry

end solve_y_l81_81594


namespace dragon_jewels_l81_81930

theorem dragon_jewels (x : ℕ) (h1 : (x / 3 = 6)) : x + 6 = 24 :=
sorry

end dragon_jewels_l81_81930


namespace nathan_paintable_area_l81_81343

def total_paintable_area (rooms : ℕ) (length width height : ℕ) (non_paintable_area : ℕ) : ℕ :=
  let wall_area := 2 * (length * height + width * height)
  rooms * (wall_area - non_paintable_area)

theorem nathan_paintable_area :
  total_paintable_area 4 15 12 9 75 = 1644 :=
by sorry

end nathan_paintable_area_l81_81343


namespace scoops_arrangement_count_l81_81751

theorem scoops_arrangement_count :
  (5 * 4 * 3 * 2 * 1 = 120) :=
by
  sorry

end scoops_arrangement_count_l81_81751


namespace jelly_bean_ratio_l81_81487

theorem jelly_bean_ratio 
  (Napoleon_jelly_beans : ℕ)
  (Sedrich_jelly_beans : ℕ)
  (Mikey_jelly_beans : ℕ)
  (h1 : Napoleon_jelly_beans = 17)
  (h2 : Sedrich_jelly_beans = Napoleon_jelly_beans + 4)
  (h3 : Mikey_jelly_beans = 19) :
  2 * (Napoleon_jelly_beans + Sedrich_jelly_beans) / Mikey_jelly_beans = 4 := 
sorry

end jelly_bean_ratio_l81_81487


namespace inverse_proportion_indeterminate_l81_81868

theorem inverse_proportion_indeterminate (k : ℝ) (x1 x2 y1 y2 : ℝ) (h1 : x1 < x2)
  (h2 : y1 = k / x1) (h3 : y2 = k / x2) : 
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0) ∨ (y1 * y2 < 0) → false :=
sorry

end inverse_proportion_indeterminate_l81_81868


namespace problem_21_sum_correct_l81_81741

theorem problem_21_sum_correct (A B C D E : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
    (h_eq : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 21 :=
sorry

end problem_21_sum_correct_l81_81741


namespace cylinder_intersection_in_sphere_l81_81178

theorem cylinder_intersection_in_sphere
  (a b c d e f : ℝ)
  (x y z : ℝ)
  (h1 : (x - a)^2 + (y - b)^2 < 1)
  (h2 : (y - c)^2 + (z - d)^2 < 1)
  (h3 : (z - e)^2 + (x - f)^2 < 1) :
  (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 := 
sorry

end cylinder_intersection_in_sphere_l81_81178


namespace factorization_correct_l81_81812

theorem factorization_correct (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 12 * x^4 + 3) = 12 * (x^6 + 4 * x^4 - 1) := by
  sorry

end factorization_correct_l81_81812


namespace find_y_l81_81461

-- Define the problem conditions
def avg_condition (y : ℝ) : Prop := (15 + 25 + y) / 3 = 23

-- Prove that the value of 'y' satisfying the condition is 29
theorem find_y (y : ℝ) (h : avg_condition y) : y = 29 :=
sorry

end find_y_l81_81461


namespace online_sale_discount_l81_81529

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end online_sale_discount_l81_81529


namespace median_and_mode_l81_81000

open Set

variable (data_set : List ℝ)
variable (mean : ℝ)

noncomputable def median (l : List ℝ) : ℝ := sorry -- Define medial function
noncomputable def mode (l : List ℝ) : ℝ := sorry -- Define mode function

theorem median_and_mode (x : ℝ) (mean_set : (3 + x + 4 + 5 + 8) / 5 = 5) :
  data_set = [3, 4, 5, 5, 8] ∧ median data_set = 5 ∧ mode data_set = 5 :=
by
  have hx : x = 5 := sorry
  have hdata_set : data_set = [3, 4, 5, 5, 8] := sorry
  have hmedian : median data_set = 5 := sorry
  have hmode : mode data_set = 5 := sorry
  exact ⟨hdata_set, hmedian, hmode⟩

end median_and_mode_l81_81000


namespace different_algorithms_for_same_problem_l81_81380

-- Define the basic concept of a problem
def Problem := Type

-- Define what it means for something to be an algorithm solving a problem
def Algorithm (P : Problem) := P -> Prop

-- Define the statement to be true: Different algorithms can solve the same problem
theorem different_algorithms_for_same_problem (P : Problem) (A1 A2 : Algorithm P) :
  P = P -> A1 ≠ A2 -> true :=
by
  sorry

end different_algorithms_for_same_problem_l81_81380


namespace speed_of_current_l81_81851

-- Definitions
def downstream_speed (m current : ℝ) := m + current
def upstream_speed (m current : ℝ) := m - current

-- Theorem
theorem speed_of_current 
  (m : ℝ) (current : ℝ) 
  (h1 : downstream_speed m current = 20) 
  (h2 : upstream_speed m current = 14) : 
  current = 3 :=
by
  -- proof goes here
  sorry

end speed_of_current_l81_81851


namespace necessary_but_not_sufficient_l81_81747

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 < x) → ((x^2 < x) ↔ (0 < x ∧ x < 1)) ∧ ((1/x > 2) ↔ (0 < x ∧ x < 1/2)) := 
by 
  sorry

end necessary_but_not_sufficient_l81_81747


namespace arithmetic_sequence_sum_six_terms_l81_81690

noncomputable def sum_of_first_six_terms (a : ℤ) (d : ℤ) : ℤ :=
  let a1 := a
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let a5 := a4 + d
  let a6 := a5 + d
  a1 + a2 + a3 + a4 + a5 + a6

theorem arithmetic_sequence_sum_six_terms
  (a3 a4 a5 : ℤ)
  (h3 : a3 = 8)
  (h4 : a4 = 13)
  (h5 : a5 = 18)
  (d : ℤ) (a : ℤ)
  (h_d : d = a4 - a3)
  (h_a : a + 2 * d = 8) :
  sum_of_first_six_terms a d = 63 :=
by
  sorry

end arithmetic_sequence_sum_six_terms_l81_81690


namespace inequality_proof_l81_81803

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0)
  (hxz : x * z = 1) 
  (h₁ : x * (1 + z) > 1) 
  (h₂ : y * (1 + x) > 1) 
  (h₃ : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 :=
sorry

end inequality_proof_l81_81803


namespace polynomial_proof_l81_81482

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

theorem polynomial_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) : 
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
by 
  sorry

end polynomial_proof_l81_81482


namespace bus_children_count_l81_81512

theorem bus_children_count
  (initial_count : ℕ)
  (first_stop_add : ℕ)
  (second_stop_add : ℕ)
  (second_stop_remove : ℕ)
  (third_stop_remove : ℕ)
  (third_stop_add : ℕ)
  (final_count : ℕ)
  (h1 : initial_count = 18)
  (h2 : first_stop_add = 5)
  (h3 : second_stop_remove = 4)
  (h4 : third_stop_remove = 3)
  (h5 : third_stop_add = 5)
  (h6 : final_count = 25)
  (h7 : initial_count + first_stop_add = 23)
  (h8 : 23 + second_stop_add - second_stop_remove - third_stop_remove + third_stop_add = final_count) :
  second_stop_add = 4 :=
by
  sorry

end bus_children_count_l81_81512


namespace valid_decomposition_2009_l81_81092

/-- A definition to determine whether a number can be decomposed
    into sums of distinct numbers with repeated digits representation. -/
def decomposable_2009 (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  a = 1111 ∧ b = 777 ∧ c = 66 ∧ d = 55 ∧ a + b + c + d = n

theorem valid_decomposition_2009 :
  decomposable_2009 2009 :=
sorry

end valid_decomposition_2009_l81_81092


namespace always_composite_for_x64_l81_81305

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem always_composite_for_x64 (n : ℕ) : is_composite (n^4 + 64) :=
by
  sorry

end always_composite_for_x64_l81_81305


namespace y_work_time_l81_81698

theorem y_work_time (x_days : ℕ) (x_work_time : ℕ) (y_work_time : ℕ) :
  x_days = 40 ∧ x_work_time = 8 ∧ y_work_time = 20 →
  let x_rate := 1 / 40
  let work_done_by_x := 8 * x_rate
  let remaining_work := 1 - work_done_by_x
  let y_rate := remaining_work / 20
  y_rate * 25 = 1 :=
by {
  sorry
}

end y_work_time_l81_81698


namespace compute_d_for_ellipse_l81_81929

theorem compute_d_for_ellipse
  (in_first_quadrant : true)
  (is_tangent_x_axis : true)
  (is_tangent_y_axis : true)
  (focus1 : (ℝ × ℝ) := (5, 4))
  (focus2 : (ℝ × ℝ) := (d, 4)) :
  d = 3.2 := by
  sorry

end compute_d_for_ellipse_l81_81929


namespace remainder_theorem_example_l81_81403

def polynomial (x : ℝ) : ℝ := x^15 + 3

theorem remainder_theorem_example :
  polynomial (-2) = -32765 :=
by
  -- Substitute x = -2 in the polynomial and show the remainder is -32765
  sorry

end remainder_theorem_example_l81_81403


namespace union_complement_A_B_l81_81519

-- Definitions based on conditions
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | x < 6}
def C_R (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- The proof problem statement
theorem union_complement_A_B :
  (C_R B ∪ A = {x | 0 ≤ x}) :=
by 
  sorry

end union_complement_A_B_l81_81519


namespace tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l81_81727

noncomputable def tangent_line (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_point_is_x_minus_y_plus_1_eq_0:
  tangent_line 0 = 1 →
  ∀ x y, y = tangent_line x → x - y + 1 = 0 → y = x * Real.exp x + 1 →
  x = 0 ∧ y = 1 → x - y + 1 = 0 :=
by
  intro h_point x y h_tangent h_eq h_coord
  sorry

end tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l81_81727


namespace connie_initial_marbles_l81_81294

theorem connie_initial_marbles (marbles_given : ℝ) (marbles_left : ℝ) : 
  marbles_given = 183 → marbles_left = 593 → marbles_given + marbles_left = 776 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end connie_initial_marbles_l81_81294


namespace bank_record_withdrawal_l81_81767

def deposit (x : ℤ) := x
def withdraw (x : ℤ) := -x

theorem bank_record_withdrawal : withdraw 500 = -500 :=
by
  sorry

end bank_record_withdrawal_l81_81767


namespace evaluation_of_expression_l81_81527

theorem evaluation_of_expression: 
  (3^10 + 3^7) / (3^10 - 3^7) = 14 / 13 := 
  sorry

end evaluation_of_expression_l81_81527


namespace painting_time_equation_l81_81746

theorem painting_time_equation
  (Hannah_rate : ℝ)
  (Sarah_rate : ℝ)
  (combined_rate : ℝ)
  (temperature_factor : ℝ)
  (break_time : ℝ)
  (t : ℝ)
  (condition1 : Hannah_rate = 1 / 6)
  (condition2 : Sarah_rate = 1 / 8)
  (condition3 : combined_rate = (Hannah_rate + Sarah_rate) * temperature_factor)
  (condition4 : temperature_factor = 0.9)
  (condition5 : break_time = 1.5) :
  (combined_rate * (t - break_time) = 1) ↔ (t = 1 + break_time + 1 / combined_rate) :=
by
  sorry

end painting_time_equation_l81_81746


namespace hannah_age_l81_81054

-- Define the constants and conditions
variables (E F G H : ℕ)
axiom h₁ : E = F - 4
axiom h₂ : F = G + 6
axiom h₃ : H = G + 2
axiom h₄ : E = 15

-- Prove that Hannah is 15 years old
theorem hannah_age : H = 15 :=
by sorry

end hannah_age_l81_81054


namespace cost_price_computer_table_l81_81125

variable (C : ℝ) -- Cost price of the computer table
variable (S : ℝ) -- Selling price of the computer table

-- Conditions based on the problem
axiom h1 : S = 1.10 * C
axiom h2 : S = 8800

-- The theorem to be proven
theorem cost_price_computer_table : C = 8000 :=
by
  -- Proof will go here
  sorry

end cost_price_computer_table_l81_81125


namespace winning_candidate_percentage_l81_81405

theorem winning_candidate_percentage
  (votes_candidate1 : ℕ) (votes_candidate2 : ℕ) (votes_candidate3 : ℕ)
  (total_votes : ℕ) (winning_votes : ℕ) (percentage : ℚ)
  (h1 : votes_candidate1 = 1000)
  (h2 : votes_candidate2 = 2000)
  (h3 : votes_candidate3 = 4000)
  (h4 : total_votes = votes_candidate1 + votes_candidate2 + votes_candidate3)
  (h5 : winning_votes = votes_candidate3)
  (h6 : percentage = (winning_votes : ℚ) / total_votes * 100) :
  percentage = 57.14 := 
sorry

end winning_candidate_percentage_l81_81405


namespace square_area_max_l81_81894

theorem square_area_max (perimeter : ℝ) (h_perimeter : perimeter = 32) : 
  ∃ (area : ℝ), area = 64 :=
by
  sorry

end square_area_max_l81_81894


namespace g_at_3_l81_81635

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 : (∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2 + 1) → 
  g 3 = 130 / 21 := 
by 
  sorry

end g_at_3_l81_81635


namespace original_number_l81_81019

variable (x : ℝ)

theorem original_number (h1 : x - x / 10 = 37.35) : x = 41.5 := by
  sorry

end original_number_l81_81019


namespace lisa_eggs_total_l81_81224

def children_mon_tue := 4 * 2 * 2
def husband_mon_tue := 3 * 2 
def lisa_mon_tue := 2 * 2
def total_mon_tue := children_mon_tue + husband_mon_tue + lisa_mon_tue

def children_wed := 4 * 3
def husband_wed := 4
def lisa_wed := 3
def total_wed := children_wed + husband_wed + lisa_wed

def children_thu := 4 * 1
def husband_thu := 2
def lisa_thu := 1
def total_thu := children_thu + husband_thu + lisa_thu

def children_fri := 4 * 2
def husband_fri := 3
def lisa_fri := 2
def total_fri := children_fri + husband_fri + lisa_fri

def total_week := total_mon_tue + total_wed + total_thu + total_fri

def weeks_per_year := 52
def yearly_eggs := total_week * weeks_per_year

def children_holidays := 4 * 2 * 8
def husband_holidays := 2 * 8
def lisa_holidays := 2 * 8
def total_holidays := children_holidays + husband_holidays + lisa_holidays

def total_annual_eggs := yearly_eggs + total_holidays

theorem lisa_eggs_total : total_annual_eggs = 3476 := by
  sorry

end lisa_eggs_total_l81_81224


namespace quadratic_inequality_solution_set_l81_81095

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3 * x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end quadratic_inequality_solution_set_l81_81095


namespace geometric_sequence_seventh_term_l81_81467

theorem geometric_sequence_seventh_term (a r : ℕ) (h₁ : a = 6) (h₂ : a * r^4 = 486) : a * r^6 = 4374 :=
by
  -- The proof is not required, hence we use sorry.
  sorry

end geometric_sequence_seventh_term_l81_81467


namespace find_k_l81_81229

-- Conditions
def t : ℕ := 6
def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

-- Given these conditions, we need to prove that k = 9
theorem find_k (k t : ℕ) (h1 : t = 6) (h2 : is_nonzero_digit k) (h3 : is_nonzero_digit t) :
    (8 * 10^2 + k * 10 + 8) + (k * 10^2 + 8 * 10 + 8) - 16 * t * 10^0 * 6 = (9 * 10 + 8) + (9 * 10^2 + 8 * 10 + 8) - (16 * 6 * 10^1 + 6) → k = 9 := 
sorry

end find_k_l81_81229


namespace volume_correct_l81_81457

-- Define the structure and conditions
structure Point where
  x : ℝ
  y : ℝ

def is_on_circle (C : Point) (P : Point) : Prop :=
  (P.x - C.x)^2 + (P.y - C.y)^2 = 25

def volume_of_solid_of_revolution (P A B : Point) : ℝ := sorry

noncomputable def main : ℝ :=
  volume_of_solid_of_revolution {x := 2, y := -8} {x := 4.58, y := -1.98} {x := -3.14, y := -3.91}

theorem volume_correct :
  main = 672.1 := by
  -- Proof skipped
  sorry

end volume_correct_l81_81457


namespace square_area_l81_81723

theorem square_area (y1 y2 y3 y4 : ℤ) 
  (h1 : y1 = 0) (h2 : y2 = 3) (h3 : y3 = 0) (h4 : y4 = -3) : 
  ∃ (area : ℤ), area = 36 :=
by
  sorry

end square_area_l81_81723


namespace real_roots_quadratic_iff_l81_81086

theorem real_roots_quadratic_iff (a : ℝ) : (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := 
sorry

end real_roots_quadratic_iff_l81_81086


namespace max_m_value_l81_81359

noncomputable def f (x m : ℝ) : ℝ := x * Real.log x + x^2 - m * x + Real.exp (2 - x)

theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) → m ≤ 3 :=
sorry

end max_m_value_l81_81359


namespace arithmetic_geometric_sequence_l81_81481

theorem arithmetic_geometric_sequence (a b : ℝ)
  (h1 : 2 * a = 1 + b)
  (h2 : b^2 = a)
  (h3 : a ≠ b) : a = 1 / 4 :=
by
  sorry

end arithmetic_geometric_sequence_l81_81481


namespace minji_total_water_intake_l81_81024

variable (morning_water : ℝ)
variable (afternoon_water : ℝ)

theorem minji_total_water_intake (h_morning : morning_water = 0.26) (h_afternoon : afternoon_water = 0.37):
  morning_water + afternoon_water = 0.63 :=
sorry

end minji_total_water_intake_l81_81024


namespace solve_for_y_l81_81712

theorem solve_for_y (x y : ℝ) (h1 : 3 * x^2 + 4 * x + 7 * y + 2 = 0) (h2 : 3 * x + 2 * y + 5 = 0) : 4 * y^2 + 33 * y + 11 = 0 :=
sorry

end solve_for_y_l81_81712


namespace simplify_expression_l81_81156

theorem simplify_expression (x y : ℝ) : 
    3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := 
by 
    sorry

end simplify_expression_l81_81156


namespace best_shooter_l81_81939

noncomputable def avg_A : ℝ := 9
noncomputable def avg_B : ℝ := 8
noncomputable def avg_C : ℝ := 9
noncomputable def avg_D : ℝ := 9

noncomputable def var_A : ℝ := 1.2
noncomputable def var_B : ℝ := 0.4
noncomputable def var_C : ℝ := 1.8
noncomputable def var_D : ℝ := 0.4

theorem best_shooter :
  (avg_A = 9 ∧ var_A = 1.2) →
  (avg_B = 8 ∧ var_B = 0.4) →
  (avg_C = 9 ∧ var_C = 1.8) →
  (avg_D = 9 ∧ var_D = 0.4) →
  avg_D = 9 ∧ var_D = 0.4 :=
by {
  sorry
}

end best_shooter_l81_81939


namespace f_2007_l81_81458

noncomputable def f : ℕ → ℝ :=
  sorry

axiom functional_eq (x y : ℕ) : f (x + y) = f x * f y

axiom f_one : f 1 = 2

theorem f_2007 : f 2007 = 2 ^ 2007 :=
by
  sorry

end f_2007_l81_81458


namespace good_carrots_l81_81845

-- Definitions
def vanessa_carrots : ℕ := 17
def mother_carrots : ℕ := 14
def bad_carrots : ℕ := 7

-- Proof statement
theorem good_carrots : (vanessa_carrots + mother_carrots) - bad_carrots = 24 := by
  sorry

end good_carrots_l81_81845


namespace remainder_43_pow_43_plus_43_mod_44_l81_81045

theorem remainder_43_pow_43_plus_43_mod_44 :
  let n := 43
  let m := 44
  (n^43 + n) % m = 42 :=
by 
  let n := 43
  let m := 44
  sorry

end remainder_43_pow_43_plus_43_mod_44_l81_81045


namespace max_value_of_f_on_interval_exists_x_eq_min_1_l81_81691

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f_on_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → f x ≤ 1 / 4 := sorry

theorem exists_x_eq_min_1 : 
  ∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f x = 1 / 4 := sorry

end max_value_of_f_on_interval_exists_x_eq_min_1_l81_81691


namespace basil_plants_yielded_l81_81602

def initial_investment (seed_cost soil_cost : ℕ) : ℕ :=
  seed_cost + soil_cost

def total_revenue (net_profit initial_investment : ℕ) : ℕ :=
  net_profit + initial_investment

def basil_plants (total_revenue price_per_plant : ℕ) : ℕ :=
  total_revenue / price_per_plant

theorem basil_plants_yielded
  (seed_cost soil_cost net_profit price_per_plant expected_plants : ℕ)
  (h_seed_cost : seed_cost = 2)
  (h_soil_cost : soil_cost = 8)
  (h_net_profit : net_profit = 90)
  (h_price_per_plant : price_per_plant = 5)
  (h_expected_plants : expected_plants = 20) :
  basil_plants (total_revenue net_profit (initial_investment seed_cost soil_cost)) price_per_plant = expected_plants :=
by
  -- Proof steps will be here
  sorry

end basil_plants_yielded_l81_81602


namespace hall_length_width_difference_l81_81444

theorem hall_length_width_difference (L W : ℝ) 
(h1 : W = 1 / 2 * L) 
(h2 : L * W = 200) : L - W = 10 := 
by 
  sorry

end hall_length_width_difference_l81_81444


namespace combined_height_of_cylinders_l81_81382

/-- Given three cylinders with perimeters 6 feet, 9 feet, and 11 feet respectively,
    and rolled out on a rectangular plate with a diagonal of 19 feet,
    the combined height of the cylinders is 26 feet. -/
theorem combined_height_of_cylinders
  (p1 p2 p3 : ℝ) (d : ℝ)
  (h_p1 : p1 = 6) (h_p2 : p2 = 9) (h_p3 : p3 = 11) (h_d : d = 19) :
  p1 + p2 + p3 = 26 :=
sorry

end combined_height_of_cylinders_l81_81382


namespace abc_sum_equals_9_l81_81842

theorem abc_sum_equals_9 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a * b + c = 57) (h5 : b * c + a = 57) (h6 : a * c + b = 57) :
  a + b + c = 9 := 
sorry

end abc_sum_equals_9_l81_81842


namespace A_minus_3B_A_minus_3B_independent_of_y_l81_81761

variables (x y : ℝ)
def A : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B : ℝ := x^2 - 2*x - y + x*y - 5

theorem A_minus_3B (x y : ℝ) : A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15 :=
by
  sorry

theorem A_minus_3B_independent_of_y (x : ℝ) (hyp : ∀ y : ℝ, A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15) :
  5 - 7*x = 0 → x = 5 / 7 :=
by
  sorry

end A_minus_3B_A_minus_3B_independent_of_y_l81_81761


namespace no_integer_right_triangle_side_x_l81_81158

theorem no_integer_right_triangle_side_x :
  ∀ (x : ℤ), (12 + 30 > x ∧ 12 + x > 30 ∧ 30 + x > 12) →
             (12^2 + 30^2 = x^2 ∨ 12^2 + x^2 = 30^2 ∨ 30^2 + x^2 = 12^2) →
             (¬ (∃ x : ℤ, 18 < x ∧ x < 42)) :=
by
  sorry

end no_integer_right_triangle_side_x_l81_81158


namespace symmetric_point_correct_l81_81739

-- Define the point and line
def point : ℝ × ℝ := (-1, 2)
def line (x : ℝ) : ℝ := x - 1

-- Define a function that provides the symmetric point with respect to the line
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ :=
  -- Since this function is a critical part of the problem, we won't define it explicitly. Using a placeholder.
  sorry

-- The proof problem
theorem symmetric_point_correct : symmetric_point point line = (3, -2) :=
  sorry

end symmetric_point_correct_l81_81739


namespace f_14_52_l81_81005

def f : ℕ × ℕ → ℕ := sorry

axiom f_xx (x : ℕ) : f (x, x) = x
axiom f_symm (x y : ℕ) : f (x, y) = f (y, x)
axiom f_eq (x y : ℕ) : (x + y) * f (x, y) = y * f (x, x + y)

theorem f_14_52 : f (14, 52) = 364 := sorry

end f_14_52_l81_81005


namespace angle_difference_l81_81783

theorem angle_difference (X Y Z Z1 Z2 : ℝ) (h1 : Y = 2 * X) (h2 : X = 30) (h3 : Z1 + Z2 = Z) (h4 : Z1 = 60) (h5 : Z2 = 30) : Z1 - Z2 = 30 := 
by 
  sorry

end angle_difference_l81_81783


namespace cylinder_height_l81_81569

theorem cylinder_height (r₁ r₂ : ℝ) (S : ℝ) (hR : r₁ = 3) (hL : r₂ = 4) (hS : S = 100 * Real.pi) : 
  (∃ h : ℝ, h = 7 ∨ h = 1) :=
by 
  sorry

end cylinder_height_l81_81569


namespace difference_between_two_numbers_l81_81163

theorem difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) 
  (h3 : x^2 - y^2 = 200) : 
  x - y = 10 :=
by 
  sorry

end difference_between_two_numbers_l81_81163


namespace bus_speed_l81_81826

def distance : ℝ := 350.028
def time : ℝ := 10
def speed_kmph : ℝ := 126.01

theorem bus_speed :
  (distance / time) * 3.6 = speed_kmph := 
sorry

end bus_speed_l81_81826


namespace calculate_a_mul_a_sub_3_l81_81566

variable (a : ℝ)

theorem calculate_a_mul_a_sub_3 : a * (a - 3) = a^2 - 3 * a := 
by
  sorry

end calculate_a_mul_a_sub_3_l81_81566


namespace a_received_share_l81_81625

variables (I_a I_b I_c b_share total_investment total_profit a_share : ℕ)
  (h1 : I_a = 11000)
  (h2 : I_b = 15000)
  (h3 : I_c = 23000)
  (h4 : b_share = 3315)
  (h5 : total_investment = I_a + I_b + I_c)
  (h6 : total_profit = b_share * total_investment / I_b)
  (h7 : a_share = I_a * total_profit / total_investment)

theorem a_received_share : a_share = 2662 := by
  sorry

end a_received_share_l81_81625


namespace algebra_expr_eval_l81_81703

theorem algebra_expr_eval {x y : ℝ} (h : x - 2 * y = 3) : 5 - 2 * x + 4 * y = -1 :=
by sorry

end algebra_expr_eval_l81_81703


namespace largest_divisor_of_product_of_five_consecutive_integers_l81_81429

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l81_81429


namespace zhang_shan_sales_prediction_l81_81275

theorem zhang_shan_sales_prediction (x : ℝ) (y : ℝ) (h : x = 34) (reg_eq : y = 2 * x + 60) : y = 128 :=
by
  sorry

end zhang_shan_sales_prediction_l81_81275


namespace cone_height_l81_81587

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l81_81587


namespace rent_expense_calculation_l81_81097

variable (S : ℝ)
variable (saved_amount : ℝ := 2160)
variable (milk_expense : ℝ := 1500)
variable (groceries_expense : ℝ := 4500)
variable (education_expense : ℝ := 2500)
variable (petrol_expense : ℝ := 2000)
variable (misc_expense : ℝ := 3940)
variable (salary_percent_saved : ℝ := 0.10)

theorem rent_expense_calculation 
  (h1 : salary_percent_saved * S = saved_amount) :
  S = 21600 → 
  0.90 * S - (milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense) = 5000 :=
by
  sorry

end rent_expense_calculation_l81_81097


namespace set_of_positive_reals_l81_81099

theorem set_of_positive_reals (S : Set ℝ) (h1 : ∀ x, x ∈ S → 0 < x)
  (h2 : ∀ a b, a ∈ S → b ∈ S → a + b ∈ S)
  (h3 : ∀ (a b : ℝ), 0 < a → a ≤ b → ∃ c d, a ≤ c ∧ c ≤ d ∧ d ≤ b ∧ ∀ x, c ≤ x ∧ x ≤ d → x ∈ S) :
  S = {x : ℝ | 0 < x} :=
sorry

end set_of_positive_reals_l81_81099


namespace cameron_books_ratio_l81_81899

theorem cameron_books_ratio (Boris_books : ℕ) (Cameron_books : ℕ)
  (Boris_after_donation : ℕ) (Cameron_after_donation : ℕ)
  (total_books_after_donation : ℕ) (ratio : ℚ) :
  Boris_books = 24 → 
  Cameron_books = 30 → 
  Boris_after_donation = Boris_books - (Boris_books / 4) →
  total_books_after_donation = 38 →
  Cameron_after_donation = total_books_after_donation - Boris_after_donation →
  ratio = (Cameron_books - Cameron_after_donation) / Cameron_books →
  ratio = 1 / 3 :=
by
  -- Proof goes here.
  sorry

end cameron_books_ratio_l81_81899


namespace average_of_combined_results_l81_81070

theorem average_of_combined_results {avg1 avg2 n1 n2 : ℝ} (h1 : avg1 = 28) (h2 : avg2 = 55) (h3 : n1 = 55) (h4 : n2 = 28) :
  ((n1 * avg1) + (n2 * avg2)) / (n1 + n2) = 37.11 :=
by sorry

end average_of_combined_results_l81_81070


namespace problem1_problem2_l81_81338

-- Problem 1: Prove the range of k for any real number x
theorem problem1 (k : ℝ) (x : ℝ) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  1 ≤ k ∧ k < 13 :=
sorry

-- Problem 2: Prove the range of k for any x in the interval (0, 1]
theorem problem2 (k : ℝ) (x : ℝ) (hx : 0 < x) (hx1 : x ≤ 1) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  k > -1/2 :=
sorry

end problem1_problem2_l81_81338


namespace hannah_total_savings_l81_81819

theorem hannah_total_savings :
  let a1 := 4
  let a2 := 2 * a1
  let a3 := 2 * a2
  let a4 := 2 * a3
  let a5 := 20
  a1 + a2 + a3 + a4 + a5 = 80 :=
by
  sorry

end hannah_total_savings_l81_81819


namespace arithmetic_sequence_sum_l81_81843

theorem arithmetic_sequence_sum (a_n : ℕ → ℝ) (h1 : a_n 1 + a_n 2 + a_n 3 + a_n 4 = 30) 
                               (h2 : a_n 1 + a_n 4 = a_n 2 + a_n 3) :
  a_n 2 + a_n 3 = 15 := 
by 
  sorry

end arithmetic_sequence_sum_l81_81843


namespace wrong_value_l81_81873

-- Definitions based on the conditions
def initial_mean : ℝ := 32
def corrected_mean : ℝ := 32.5
def num_observations : ℕ := 50
def correct_observation : ℝ := 48

-- We need to prove that the wrong value of the observation was 23
theorem wrong_value (sum_initial : ℝ) (sum_corrected : ℝ) : 
  sum_initial = num_observations * initial_mean ∧ 
  sum_corrected = num_observations * corrected_mean →
  48 - (sum_corrected - sum_initial) = 23 :=
by
  sorry

end wrong_value_l81_81873


namespace blood_flow_scientific_notation_l81_81596

theorem blood_flow_scientific_notation (blood_flow : ℝ) (h : blood_flow = 4900) : 
  4900 = 4.9 * (10 ^ 3) :=
by
  sorry

end blood_flow_scientific_notation_l81_81596


namespace time_equal_l81_81931

noncomputable def S : ℝ := sorry 
noncomputable def S_flat : ℝ := S
noncomputable def S_uphill : ℝ := (1 / 3) * S
noncomputable def S_downhill : ℝ := (2 / 3) * S
noncomputable def V_flat : ℝ := sorry 
noncomputable def V_uphill : ℝ := (1 / 2) * V_flat
noncomputable def V_downhill : ℝ := 2 * V_flat
noncomputable def t_flat: ℝ := S / V_flat
noncomputable def t_uphill: ℝ := S_uphill / V_uphill
noncomputable def t_downhill: ℝ := S_downhill / V_downhill
noncomputable def t_hill: ℝ := t_uphill + t_downhill

theorem time_equal: t_flat = t_hill := 
  by sorry

end time_equal_l81_81931


namespace jogging_track_circumference_l81_81588

theorem jogging_track_circumference
  (Deepak_speed : ℝ)
  (Wife_speed : ℝ)
  (meet_time_minutes : ℝ)
  (H_deepak_speed : Deepak_speed = 4.5)
  (H_wife_speed : Wife_speed = 3.75)
  (H_meet_time_minutes : meet_time_minutes = 3.84) :
  let meet_time_hours := meet_time_minutes / 60
  let distance_deepak := Deepak_speed * meet_time_hours
  let distance_wife := Wife_speed * meet_time_hours
  let total_distance := distance_deepak + distance_wife
  let circumference := 2 * total_distance
  circumference = 1.056 :=
by
  sorry

end jogging_track_circumference_l81_81588


namespace determine_M_l81_81637

theorem determine_M : ∃ M : ℕ, 36^2 * 75^2 = 30^2 * M^2 ∧ M = 90 := 
by
  sorry

end determine_M_l81_81637


namespace unique_real_y_l81_81974

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_real_y (y : ℝ) : (∃! y : ℝ, star 4 y = 10) :=
  by {
    sorry
  }

end unique_real_y_l81_81974


namespace tom_seashells_now_l81_81670

def original_seashells : ℕ := 5
def given_seashells : ℕ := 2

theorem tom_seashells_now : original_seashells - given_seashells = 3 :=
by
  sorry

end tom_seashells_now_l81_81670


namespace fraction_to_terminating_decimal_l81_81766

theorem fraction_to_terminating_decimal :
  (47 : ℚ) / (2^2 * 5^4) = 0.0188 :=
sorry

end fraction_to_terminating_decimal_l81_81766


namespace find_point_B_find_line_BC_l81_81869

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the equation of the median on side AB
def median_AB (x y : ℝ) : Prop := x + 3 * y = 6

-- Define the equation of the internal angle bisector of ∠ABC
def bisector_BC (x y : ℝ) : Prop := x - y = -1

-- Prove the coordinates of point B
theorem find_point_B :
  (a b : ℝ) →
  (median_AB ((a + 2) / 2) ((b - 1) / 2)) →
  (a - b = -1) →
  a = 5 / 2 ∧ b = 7 / 2 :=
sorry

-- Define the line equation BC
def line_BC (x y : ℝ) : Prop := x - 9 * y + 29 = 0

-- Prove the equation of the line containing side BC
theorem find_line_BC :
  (x0 y0 : ℝ) →
  bisector_BC x0 y0 →
  (x0, y0) = (-2, 3) →
  line_BC x0 y0 :=
sorry

end find_point_B_find_line_BC_l81_81869


namespace complement_union_l81_81123

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l81_81123


namespace Ruth_school_hours_l81_81143

theorem Ruth_school_hours (d : ℝ) :
  0.25 * 5 * d = 10 → d = 8 :=
by
  sorry

end Ruth_school_hours_l81_81143


namespace ab_bc_ca_abc_inequality_l81_81639

open Real

theorem ab_bc_ca_abc_inequality :
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 + a * b * c = 4 →
    0 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 2 :=
by
  intro a b c
  intro h
  sorry

end ab_bc_ca_abc_inequality_l81_81639


namespace problem_condition_l81_81249

noncomputable def f (x b : ℝ) := Real.exp x * (x - b)
noncomputable def f_prime (x b : ℝ) := Real.exp x * (x - b + 1)
noncomputable def g (x : ℝ) := (x^2 + 2*x) / (x + 1)

theorem problem_condition (b : ℝ) :
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x b + x * f_prime x b > 0) → b < 8 / 3 :=
by
  sorry

end problem_condition_l81_81249


namespace min_S_n_condition_l81_81586

noncomputable def a_n (n : ℕ) : ℤ := -28 + 4 * (n - 1)

noncomputable def S_n (n : ℕ) : ℤ := n * (a_n 1 + a_n n) / 2

theorem min_S_n_condition : S_n 7 = S_n 8 ∧ (∀ m < 7, S_n m > S_n 7) ∧ (∀ m < 8, S_n m > S_n 8) := 
by
  sorry

end min_S_n_condition_l81_81586


namespace negation_of_p_l81_81550

def p := ∀ x, x ≤ 0 → Real.exp x ≤ 1

theorem negation_of_p : ¬ p ↔ ∃ x, x ≤ 0 ∧ Real.exp x > 1 := 
by
  sorry

end negation_of_p_l81_81550


namespace find_15th_term_l81_81632

-- Define the initial terms and the sequence properties
def first_term := 4
def second_term := 13
def third_term := 22

-- Define the common difference
def common_difference := second_term - first_term

-- Define the nth term formula for arithmetic sequence
def nth_term (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- State the theorem
theorem find_15th_term : nth_term first_term common_difference 15 = 130 := by
  -- The proof will come here
  sorry

end find_15th_term_l81_81632


namespace determine_x_l81_81996

theorem determine_x (y : ℚ) (h : y = (36 + 249 / 999) / 100) :
  ∃ x : ℕ, y = x / 99900 ∧ x = 36189 :=
by
  sorry

end determine_x_l81_81996


namespace sandwich_bread_consumption_l81_81149

theorem sandwich_bread_consumption :
  ∀ (num_bread_per_sandwich : ℕ),
  (2 * num_bread_per_sandwich) + num_bread_per_sandwich = 6 →
  num_bread_per_sandwich = 2 := by
    intros num_bread_per_sandwich h
    sorry

end sandwich_bread_consumption_l81_81149


namespace emily_patches_difference_l81_81649

theorem emily_patches_difference (h p : ℕ) (h_eq : p = 3 * h) :
  (p * h) - ((p + 5) * (h - 3)) = (4 * h + 15) :=
by
  sorry

end emily_patches_difference_l81_81649


namespace percentage_of_products_by_m1_l81_81128

theorem percentage_of_products_by_m1
  (x : ℝ)
  (h1 : 30 / 100 > 0)
  (h2 : 3 / 100 > 0)
  (h3 : 1 / 100 > 0)
  (h4 : 7 / 100 > 0)
  (h_total_defective : 
    0.036 = 
      (0.03 * x / 100) + 
      (0.01 * 30 / 100) + 
      (0.07 * (100 - x - 30) / 100)) :
  x = 40 :=
by
  sorry

end percentage_of_products_by_m1_l81_81128


namespace value_increase_factor_l81_81023

theorem value_increase_factor (P S : ℝ) (frac F : ℝ) (hP : P = 200) (hS : S = 240) (hfrac : frac = 0.40) :
  frac * (P * F) = S -> F = 3 := by
  sorry

end value_increase_factor_l81_81023


namespace remainder_div_14_l81_81440

variables (x k : ℕ)

theorem remainder_div_14 (h : x = 142 * k + 110) : x % 14 = 12 := by 
  sorry

end remainder_div_14_l81_81440


namespace division_and_multiply_l81_81187

theorem division_and_multiply :
  (-128) / (-16) * 5 = 40 := 
by
  sorry

end division_and_multiply_l81_81187


namespace even_composite_sum_consecutive_odd_numbers_l81_81203

theorem even_composite_sum_consecutive_odd_numbers (a k : ℤ) : ∃ (n m : ℤ), n = 2 * k ∧ m = n * (2 * a + n) ∧ m % 4 = 0 :=
by
  sorry

end even_composite_sum_consecutive_odd_numbers_l81_81203


namespace comprehensive_score_correct_l81_81480

def comprehensive_score
  (study_score hygiene_score discipline_score participation_score : ℕ)
  (study_weight hygiene_weight discipline_weight participation_weight : ℚ) : ℚ :=
  study_score * study_weight +
  hygiene_score * hygiene_weight +
  discipline_score * discipline_weight +
  participation_score * participation_weight

theorem comprehensive_score_correct :
  let study_score := 80
  let hygiene_score := 90
  let discipline_score := 84
  let participation_score := 70
  let study_weight := 0.4
  let hygiene_weight := 0.25
  let discipline_weight := 0.25
  let participation_weight := 0.1
  comprehensive_score study_score hygiene_score discipline_score participation_score
                      study_weight hygiene_weight discipline_weight participation_weight
  = 82.5 :=
by 
  sorry

#eval comprehensive_score 80 90 84 70 0.4 0.25 0.25 0.1  -- output should be 82.5

end comprehensive_score_correct_l81_81480


namespace number_is_450064_l81_81286

theorem number_is_450064 : (45 * 10000 + 64) = 450064 :=
by
  sorry

end number_is_450064_l81_81286


namespace arithmetic_geometric_sequence_problem_l81_81413

variable {a_n : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 0 + n * d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = (n * (a_n 0 + a_n (n-1))) / 2

def forms_geometric_sequence (a1 a3 a4 : ℝ) :=
  a3^2 = a1 * a4

-- The main proof statement
theorem arithmetic_geometric_sequence_problem
        (h_arith : is_arithmetic_sequence a_n)
        (h_sum : sum_of_first_n_terms a_n S)
        (h_geom : forms_geometric_sequence (a_n 0) (a_n 2) (a_n 3)) :
        (S 3 - S 2) / (S 5 - S 3) = 2 ∨ (S 3 - S 2) / (S 5 - S 3) = 1 / 2 :=
  sorry

end arithmetic_geometric_sequence_problem_l81_81413


namespace power_function_value_at_3_l81_81233

theorem power_function_value_at_3
  (f : ℝ → ℝ)
  (h1 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)
  (h2 : f 2 = 1 / 4) :
  f 3 = 1 / 9 := 
sorry

end power_function_value_at_3_l81_81233


namespace probability_at_least_one_pen_l81_81967

noncomputable def PAs  := 3/5
noncomputable def PBs  := 2/3
noncomputable def PABs := PAs * PBs

theorem probability_at_least_one_pen : PAs + PBs - PABs = 13 / 15 := by
  sorry

end probability_at_least_one_pen_l81_81967


namespace solution_to_equation_l81_81357

theorem solution_to_equation (x : ℝ) (h : (5 - x / 2)^(1/3) = 2) : x = -6 :=
sorry

end solution_to_equation_l81_81357


namespace area_of_sector_l81_81410

def radius : ℝ := 5
def central_angle : ℝ := 2

theorem area_of_sector : (1 / 2) * radius^2 * central_angle = 25 := by
  sorry

end area_of_sector_l81_81410


namespace lawnmower_blade_cost_l81_81234

theorem lawnmower_blade_cost (x : ℕ) : 4 * x + 7 = 39 → x = 8 :=
by
  sorry

end lawnmower_blade_cost_l81_81234


namespace compute_diff_squares_l81_81667

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l81_81667


namespace height_of_trapezium_l81_81288

-- Define the lengths of the parallel sides
def length_side1 : ℝ := 10
def length_side2 : ℝ := 18

-- Define the given area of the trapezium
def area_trapezium : ℝ := 210

-- The distance between the parallel sides (height) we want to prove
def height_between_sides : ℝ := 15

-- State the problem as a theorem in Lean: prove that the height is correct
theorem height_of_trapezium :
  (1 / 2) * (length_side1 + length_side2) * height_between_sides = area_trapezium :=
by
  sorry

end height_of_trapezium_l81_81288


namespace decreasing_y_as_x_increases_l81_81445

theorem decreasing_y_as_x_increases :
  (∀ x1 x2, x1 < x2 → (-2 * x1 + 1) > (-2 * x2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (x1^2 + 1) > (x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (-x1^2 + 1) > (-x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (2 * x1 + 1) > (2 * x2 + 1)) :=
by
  sorry

end decreasing_y_as_x_increases_l81_81445


namespace valid_parametrizations_l81_81317

-- Define the line as a function
def line (x : ℝ) : ℝ := -2 * x + 7

-- Define vectors and their properties
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def on_line (v : Vector2D) : Prop :=
  v.y = line v.x

def direction_vector (v1 v2 : Vector2D) : Vector2D :=
  ⟨v2.x - v1.x, v2.y - v1.y⟩

def is_multiple (v1 v2 : Vector2D) : Prop :=
  ∃ k : ℝ, v2.x = k * v1.x ∧ v2.y = k * v1.y

-- Define the given parameterizations
def param_A (t : ℝ) : Vector2D := ⟨0 + t * 5, 7 + t * 10⟩
def param_B (t : ℝ) : Vector2D := ⟨2 + t * 1, 3 + t * -2⟩
def param_C (t : ℝ) : Vector2D := ⟨7 + t * 4, 0 + t * -8⟩
def param_D (t : ℝ) : Vector2D := ⟨-1 + t * 2, 9 + t * 4⟩
def param_E (t : ℝ) : Vector2D := ⟨3 + t * 2, 1 + t * 0⟩

-- Define the theorem
theorem valid_parametrizations :
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨0, 7⟩ (param_A t)) ∧ on_line (param_A t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨2, 3⟩ (param_B t)) ∧ on_line (param_B t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨7, 0⟩ (param_C t)) ∧ on_line (param_C t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨-1, 9⟩ (param_D t)) ∧ on_line (param_D t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨3, 1⟩ (param_E t)) ∧ on_line (param_E t) → False) :=
by
  sorry

end valid_parametrizations_l81_81317


namespace solution_set_of_inequality_l81_81510

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) * (x - 2) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l81_81510


namespace real_part_zero_implies_a_eq_one_l81_81895

open Complex

theorem real_part_zero_implies_a_eq_one (a : ℝ) : 
  (1 + (1 : ℂ) * I) * (1 + a * I) = 0 ↔ a = 1 := by
  sorry

end real_part_zero_implies_a_eq_one_l81_81895


namespace guo_can_pay_exactly_l81_81199

theorem guo_can_pay_exactly (
  x y z : ℕ
) (h : 10 * x + 20 * y + 50 * z = 20000) : ∃ a b c : ℕ, a + 2 * b + 5 * c = 1000 :=
sorry

end guo_can_pay_exactly_l81_81199


namespace last_three_digits_of_expression_l81_81785

theorem last_three_digits_of_expression : 
  let prod := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349
  (prod ^ 3) % 1000 = 976 :=
by
  sorry

end last_three_digits_of_expression_l81_81785


namespace line_through_points_l81_81264

theorem line_through_points 
  (A1 B1 A2 B2 : ℝ) 
  (h₁ : A1 * -7 + B1 * 9 = 1) 
  (h₂ : A2 * -7 + B2 * 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), (A1, B1) ≠ (A2, B2) → y = k * x + (B1 - k * A1) → -7 * x + 9 * y = 1 :=
sorry

end line_through_points_l81_81264


namespace ratio_vegan_gluten_free_cupcakes_l81_81516

theorem ratio_vegan_gluten_free_cupcakes :
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  (vegan_gluten_free_cupcakes / vegan_cupcakes) = 1 / 2 :=
by {
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  have h : vegan_gluten_free_cupcakes = 12 := by norm_num
  have r : 12 / 24 = 1 / 2 := by norm_num
  exact r
}

end ratio_vegan_gluten_free_cupcakes_l81_81516


namespace tom_needs_44000_pounds_salt_l81_81774

theorem tom_needs_44000_pounds_salt 
  (flour_needed : ℕ)
  (flour_bag_weight : ℕ)
  (flour_bag_cost : ℕ)
  (salt_cost_per_pound : ℝ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_revenue : ℕ) 
  (expected_salt_cost : ℝ) 
  (S : ℝ) : 
  flour_needed = 500 → 
  flour_bag_weight = 50 → 
  flour_bag_cost = 20 → 
  salt_cost_per_pound = 0.2 → 
  promotion_cost = 1000 → 
  ticket_price = 20 → 
  tickets_sold = 500 → 
  total_revenue = 8798 → 
  0.2 * S = (500 * 20) - (500 / 50) * 20 - 1000 →
  S = 44000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end tom_needs_44000_pounds_salt_l81_81774


namespace domain_of_f_l81_81037

noncomputable def f (x : ℝ) : ℝ := (5 * x - 2) / Real.sqrt (x^2 - 3 * x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ (f_x : ℝ), f x = f_x} = {x : ℝ | (x < -1) ∨ (x > 4)} :=
by
  sorry

end domain_of_f_l81_81037


namespace intersecting_lines_ratio_l81_81164

theorem intersecting_lines_ratio (k1 k2 a : ℝ) (h1 : k1 * a + 4 = 0) (h2 : k2 * a - 2 = 0) : k1 / k2 = -2 :=
by
    sorry

end intersecting_lines_ratio_l81_81164


namespace solution_of_system_l81_81219

variable (x y : ℝ) 

def equation1 (x y : ℝ) : Prop := 3 * |x| + 5 * y + 9 = 0
def equation2 (x y : ℝ) : Prop := 2 * x - |y| - 7 = 0

theorem solution_of_system : ∃ y : ℝ, equation1 0 y ∧ equation2 0 y := by
  sorry

end solution_of_system_l81_81219


namespace girls_in_art_class_l81_81383

theorem girls_in_art_class (g b : ℕ) (h_ratio : 4 * b = 3 * g) (h_total : g + b = 70) : g = 40 :=
by {
  sorry
}

end girls_in_art_class_l81_81383


namespace find_prime_numbers_of_form_p_p_plus_1_l81_81243

def has_at_most_19_digits (n : ℕ) : Prop := n < 10^19

theorem find_prime_numbers_of_form_p_p_plus_1 :
  {n : ℕ | ∃ p : ℕ, n = p^p + 1 ∧ has_at_most_19_digits n ∧ Nat.Prime n} = {2, 5, 257} :=
by
  sorry

end find_prime_numbers_of_form_p_p_plus_1_l81_81243


namespace number_of_factors_l81_81745

theorem number_of_factors (K : ℕ) (hK : K = 2^4 * 3^3 * 5^2 * 7^1) : 
  ∃ n : ℕ, (∀ d e f g : ℕ, (0 ≤ d ∧ d ≤ 4) → (0 ≤ e ∧ e ≤ 3) → (0 ≤ f ∧ f ≤ 2) → (0 ≤ g ∧ g ≤ 1) → n = 120) :=
sorry

end number_of_factors_l81_81745


namespace total_cost_for_tickets_l81_81976

-- Definitions given in conditions
def num_students : ℕ := 20
def num_teachers : ℕ := 3
def ticket_cost : ℕ := 5

-- Proof Statement 
theorem total_cost_for_tickets : num_students + num_teachers * ticket_cost = 115 := by
  sorry

end total_cost_for_tickets_l81_81976


namespace cosQ_is_0_point_4_QP_is_12_prove_QR_30_l81_81379

noncomputable def find_QR (Q : Real) (QP : Real) : Real :=
  let cosQ := 0.4
  let QR := QP / cosQ
  QR

theorem cosQ_is_0_point_4_QP_is_12_prove_QR_30 :
  find_QR 0.4 12 = 30 :=
by
  sorry

end cosQ_is_0_point_4_QP_is_12_prove_QR_30_l81_81379


namespace quadrilateral_EFGH_area_l81_81578

-- Definitions based on conditions
def quadrilateral_EFGH_right_angles (F H : ℝ) : Prop :=
  ∃ E G, E - F = 0 ∧ H - G = 0

def quadrilateral_length_hypotenuse (E G : ℝ) : Prop :=
  E - G = 5

def distinct_integer_lengths (EF FG EH HG : ℝ) : Prop :=
  EF ≠ FG ∧ EH ≠ HG ∧ ∃ a b : ℕ, EF = a ∧ FG = b ∧ EH = b ∧ HG = a ∧ a * a + b * b = 25

-- Proof statement
theorem quadrilateral_EFGH_area (F H : ℝ) 
  (EF FG EH HG E G : ℝ) 
  (h1 : quadrilateral_EFGH_right_angles F H) 
  (h2 : quadrilateral_length_hypotenuse E G)
  (h3 : distinct_integer_lengths EF FG EH HG) 
: 
  EF * FG / 2 + EH * HG / 2 = 12 := 
sorry

end quadrilateral_EFGH_area_l81_81578


namespace water_in_bowl_after_adding_4_cups_l81_81222

def total_capacity_bowl := 20 -- Capacity of the bowl in cups

def initially_half_full (C : ℕ) : Prop :=
C = total_capacity_bowl / 2

def after_adding_4_cups (initial : ℕ) : ℕ :=
initial + 4

def seventy_percent_full (C : ℕ) : ℕ :=
7 * C / 10

theorem water_in_bowl_after_adding_4_cups :
  ∀ (C initial after_adding) (h1 : initially_half_full initial)
  (h2 : after_adding = after_adding_4_cups initial)
  (h3 : after_adding = seventy_percent_full C),
  after_adding = 14 := 
by
  intros C initial after_adding h1 h2 h3
  -- Proof goes here
  sorry

end water_in_bowl_after_adding_4_cups_l81_81222


namespace max_cut_length_l81_81346

theorem max_cut_length (board_size : ℕ) (total_pieces : ℕ) 
  (area_each : ℕ) 
  (total_area : ℕ)
  (total_perimeter : ℕ)
  (initial_perimeter : ℕ)
  (max_possible_length : ℕ)
  (h1 : board_size = 30) 
  (h2 : total_pieces = 225)
  (h3 : area_each = 4)
  (h4 : total_area = board_size * board_size)
  (h5 : total_perimeter = total_pieces * 10)
  (h6 : initial_perimeter = 4 * board_size)
  (h7 : max_possible_length = (total_perimeter - initial_perimeter) / 2) :
  max_possible_length = 1065 :=
by 
  -- Here, we do not include the proof as per the instructions
  sorry

end max_cut_length_l81_81346


namespace negB_sufficient_for_A_l81_81943

variables {A B : Prop}

theorem negB_sufficient_for_A (h : ¬A → B) (hnotsuff : ¬(B → ¬A)) : ¬ B → A :=
by
  sorry

end negB_sufficient_for_A_l81_81943


namespace arithmetic_sequence_sum_l81_81381

/-- Let {a_n} be an arithmetic sequence and S_n the sum of its first n terms.
   Given a_1 - a_5 - a_10 - a_15 + a_19 = 2, prove that S_19 = -38. --/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
  S 19 = -38 := 
sorry

end arithmetic_sequence_sum_l81_81381


namespace orcs_per_squad_is_eight_l81_81830

-- Defining the conditions
def total_weight_of_swords := 1200
def weight_each_orc_can_carry := 15
def number_of_squads := 10

-- Proof statement to demonstrate the answer
theorem orcs_per_squad_is_eight :
  (total_weight_of_swords / weight_each_orc_can_carry) / number_of_squads = 8 := by
  sorry

end orcs_per_squad_is_eight_l81_81830


namespace time_to_fill_tank_l81_81421

-- Definitions for conditions
def pipe_a := 50
def pipe_b := 75
def pipe_c := 100

-- Definition for the combined rate and time to fill the tank
theorem time_to_fill_tank : 
  (1 / pipe_a + 1 / pipe_b + 1 / pipe_c) * (300 / 13) = 1 := 
by
  sorry

end time_to_fill_tank_l81_81421


namespace michael_truck_meet_once_l81_81416

noncomputable def meets_count (michael_speed : ℕ) (pail_distance : ℕ) (truck_speed : ℕ) (truck_stop_duration : ℕ) : ℕ :=
  if michael_speed = 4 ∧ pail_distance = 300 ∧ truck_speed = 8 ∧ truck_stop_duration = 45 then 1 else sorry

theorem michael_truck_meet_once :
  meets_count 4 300 8 45 = 1 :=
by simp [meets_count]

end michael_truck_meet_once_l81_81416


namespace brigade_harvest_time_l81_81517

theorem brigade_harvest_time (t : ℕ) :
  (t - 5 = (3 * t / 5) + ((t * (t - 8)) / (5 * (t - 4)))) → t = 20 := sorry

end brigade_harvest_time_l81_81517


namespace natalie_list_count_l81_81301

theorem natalie_list_count : ∀ n : ℕ, (15 ≤ n ∧ n ≤ 225) → ((225 - 15 + 1) = 211) :=
by
  intros n h
  sorry

end natalie_list_count_l81_81301


namespace sugar_cubes_left_l81_81905

theorem sugar_cubes_left (h w d : ℕ) (hd1 : w * d = 77) (hd2 : h * d = 55) :
  (h - 1) * w * (d - 1) = 300 ∨ (h - 1) * w * (d - 1) = 0 :=
by
  sorry

end sugar_cubes_left_l81_81905


namespace tangent_lines_to_circle_l81_81684

-- Conditions
def regions_not_enclosed := 68
def num_lines := 30 - 4

-- Theorem statement
theorem tangent_lines_to_circle (h: regions_not_enclosed = 68) : num_lines = 26 :=
by {
  sorry
}

end tangent_lines_to_circle_l81_81684


namespace fraction_sum_eq_l81_81713

variable {x : ℝ}

theorem fraction_sum_eq (h : x ≠ -1) : 
  (x / (x + 1) ^ 2) + (1 / (x + 1) ^ 2) = 1 / (x + 1) := 
by
  sorry

end fraction_sum_eq_l81_81713


namespace saturn_moon_approximation_l81_81289

theorem saturn_moon_approximation : (1.2 * 10^5) * 10 = 1.2 * 10^6 := 
by sorry

end saturn_moon_approximation_l81_81289


namespace largest_perfect_square_factor_of_3780_l81_81993

theorem largest_perfect_square_factor_of_3780 :
  ∃ m : ℕ, (∃ k : ℕ, 3780 = k * m * m) ∧ m * m = 36 :=
by
  sorry

end largest_perfect_square_factor_of_3780_l81_81993


namespace total_length_of_table_free_sides_l81_81293

theorem total_length_of_table_free_sides
  (L W : ℕ) -- Define lengths of the sides
  (h1 : L = 2 * W) -- The side opposite the wall is twice the length of each of the other two free sides
  (h2 : L * W = 128) -- The area of the rectangular table is 128 square feet
  : L + 2 * W = 32 -- Prove the total length of the table's free sides is 32 feet
  :=
sorry -- proof omitted

end total_length_of_table_free_sides_l81_81293


namespace smallest_positive_integer_x_l81_81685

theorem smallest_positive_integer_x (x : ℕ) (h : 725 * x ≡ 1165 * x [MOD 35]) : x = 7 :=
sorry

end smallest_positive_integer_x_l81_81685


namespace complex_magnitude_l81_81033

theorem complex_magnitude (z : ℂ) (i_unit : ℂ := Complex.I) 
  (h : (z - i_unit) * i_unit = 2 + i_unit) : Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_magnitude_l81_81033


namespace largest_value_of_a_l81_81551

noncomputable def largest_possible_value_of_a (a b c d : ℕ) 
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150) : Prop :=
  a = 8924

theorem largest_value_of_a (a b c d : ℕ)
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150)
  (h6 : largest_possible_value_of_a a b c d h1 h2 h3 h4 h5) : a = 8924 := h6

end largest_value_of_a_l81_81551


namespace profit_percent_l81_81995

variable (P C : ℝ)
variable (h₁ : (2/3) * P = 0.84 * C)

theorem profit_percent (P C : ℝ) (h₁ : (2/3) * P = 0.84 * C) : 
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_l81_81995


namespace power_of_two_divides_factorial_iff_l81_81675

theorem power_of_two_divides_factorial_iff (n : ℕ) (k : ℕ) : 2^(n - 1) ∣ n! ↔ n = 2^k := sorry

end power_of_two_divides_factorial_iff_l81_81675


namespace area_of_shaded_region_l81_81475

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end area_of_shaded_region_l81_81475


namespace range_of_a_same_side_of_line_l81_81011

theorem range_of_a_same_side_of_line 
  {P Q : ℝ × ℝ} 
  (hP : P = (3, -1)) 
  (hQ : Q = (-1, 2)) 
  (h_side : (3 * a - 3) * (-a + 3) > 0) : 
  a > 1 ∧ a < 3 := 
by 
  sorry

end range_of_a_same_side_of_line_l81_81011


namespace manager_salary_4200_l81_81718

theorem manager_salary_4200
    (avg_salary_employees : ℕ → ℕ → ℕ) 
    (total_salary_employees : ℕ → ℕ → ℕ)
    (new_avg_salary : ℕ → ℕ → ℕ)
    (total_salary_with_manager : ℕ → ℕ → ℕ) 
    (n_employees : ℕ)
    (employee_salary : ℕ) 
    (n_total : ℕ)
    (total_salary_before : ℕ)
    (avg_increase : ℕ)
    (new_employee_salary : ℕ) 
    (total_salary_after : ℕ) 
    (manager_salary : ℕ) :
    n_employees = 15 →
    employee_salary = 1800 →
    avg_increase = 150 →
    avg_salary_employees n_employees employee_salary = 1800 →
    total_salary_employees n_employees employee_salary = 27000 →
    new_avg_salary employee_salary avg_increase = 1950 →
    new_employee_salary = 1950 →
    total_salary_with_manager (n_employees + 1) new_employee_salary = 31200 →
    total_salary_before = 27000 →
    total_salary_after = 31200 →
    manager_salary = total_salary_after - total_salary_before →
    manager_salary = 4200 := 
by 
  intros 
  sorry

end manager_salary_4200_l81_81718


namespace total_cards_after_giveaway_l81_81689

def ben_basketball_boxes := 8
def cards_per_basketball_box := 20
def ben_baseball_boxes := 10
def cards_per_baseball_box := 15
def ben_football_boxes := 12
def cards_per_football_box := 12

def alex_hockey_boxes := 6
def cards_per_hockey_box := 15
def alex_soccer_boxes := 9
def cards_per_soccer_box := 18

def cards_given_away := 175

def total_cards_for_ben := 
  (ben_basketball_boxes * cards_per_basketball_box) + 
  (ben_baseball_boxes * cards_per_baseball_box) + 
  (ben_football_boxes * cards_per_football_box)

def total_cards_for_alex := 
  (alex_hockey_boxes * cards_per_hockey_box) + 
  (alex_soccer_boxes * cards_per_soccer_box)

def total_cards_before_exchange := total_cards_for_ben + total_cards_for_alex

def ben_gives_to_alex := 
  (ben_basketball_boxes * (cards_per_basketball_box / 2)) + 
  (ben_baseball_boxes * (cards_per_baseball_box / 2))

def total_cards_remaining := total_cards_before_exchange - cards_given_away

theorem total_cards_after_giveaway :
  total_cards_before_exchange - cards_given_away = 531 := by
  sorry

end total_cards_after_giveaway_l81_81689


namespace max_area_of_rectangle_l81_81271

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * (x + y) = 60) : x * y ≤ 225 :=
by sorry

end max_area_of_rectangle_l81_81271


namespace smallest_fraction_l81_81284

theorem smallest_fraction 
  (x y z t : ℝ) 
  (h1 : 1 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < t) : 
  (min (min (min (min ((x + y) / (z + t)) ((x + t) / (y + z))) ((y + z) / (x + t))) ((y + t) / (x + z))) ((z + t) / (x + y))) = (x + y) / (z + t) :=
by {
    sorry
}

end smallest_fraction_l81_81284


namespace slope_problem_l81_81806

theorem slope_problem (m : ℝ) (h₀ : m > 0) (h₁ : (3 - m) = m * (1 - m)) : m = Real.sqrt 3 := by
  sorry

end slope_problem_l81_81806


namespace fraction_expression_l81_81924

theorem fraction_expression :
  ((3 / 7) + (5 / 8)) / ((5 / 12) + (2 / 9)) = (531 / 322) :=
by
  sorry

end fraction_expression_l81_81924


namespace area_of_curve_l81_81968

noncomputable def polar_curve (φ : Real) : Real :=
  (1 / 2) + Real.sin φ

noncomputable def area_enclosed_by_polar_curve : Real :=
  2 * ((1 / 2) * ∫ (φ : Real) in (-Real.pi / 2)..(Real.pi / 2), (polar_curve φ) ^ 2)

theorem area_of_curve : area_enclosed_by_polar_curve = (3 * Real.pi) / 4 :=
by
  sorry

end area_of_curve_l81_81968


namespace black_area_after_six_transformations_l81_81833

noncomputable def remaining_fraction_after_transformations (initial_fraction : ℚ) (transforms : ℕ) (reduction_factor : ℚ) : ℚ :=
  reduction_factor ^ transforms * initial_fraction

theorem black_area_after_six_transformations :
  remaining_fraction_after_transformations 1 6 (2 / 3) = 64 / 729 := 
by
  sorry

end black_area_after_six_transformations_l81_81833


namespace sarahs_score_is_140_l81_81498

theorem sarahs_score_is_140 (g s : ℕ) (h1 : s = g + 60) 
  (h2 : (s + g) / 2 = 110) (h3 : s + g < 450) : s = 140 :=
by
  sorry

end sarahs_score_is_140_l81_81498


namespace sphere_surface_area_l81_81175

theorem sphere_surface_area 
  (a b c : ℝ) 
  (h1 : a = 1)
  (h2 : b = 2)
  (h3 : c = 2)
  (h_spherical_condition : ∃ R : ℝ, ∀ (x y z : ℝ), x^2 + y^2 + z^2 = (2 * R)^2) :
  4 * Real.pi * ((3 / 2)^2) = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_l81_81175


namespace greatest_value_of_a_plus_b_l81_81153

-- Definition of the problem conditions
def is_pos_int (n : ℕ) := n > 0

-- Lean statement to prove the greatest possible value of a + b
theorem greatest_value_of_a_plus_b :
  ∃ a b : ℕ, is_pos_int a ∧ is_pos_int b ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 9) ∧ a + b = 100 :=
sorry  -- Proof omitted

end greatest_value_of_a_plus_b_l81_81153


namespace percent_of_g_is_a_l81_81476

-- Definitions of the seven consecutive numbers
def consecutive_7_avg_9 (a b c d e f g : ℝ) : Prop :=
  a + b + c + d + e + f + g = 63

def is_median (d : ℝ) : Prop :=
  d = 9

def express_numbers (a b c d e f g : ℝ) : Prop :=
  a = d - 3 ∧ b = d - 2 ∧ c = d - 1 ∧ d = d ∧ e = d + 1 ∧ f = d + 2 ∧ g = d + 3

-- Main statement asserting the percentage relationship
theorem percent_of_g_is_a (a b c d e f g : ℝ) (h_avg : consecutive_7_avg_9 a b c d e f g)
  (h_median : is_median d) (h_express : express_numbers a b c d e f g) :
  (a / g) * 100 = 50 := by
  sorry

end percent_of_g_is_a_l81_81476


namespace incorrect_conclusions_l81_81648

variables (a b : ℝ)

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem incorrect_conclusions :
  a > 0 → b > 0 → a ≠ 1 → b ≠ 1 → log_base a b > 1 →
  (a < 1 ∧ b > a ∨ (¬ (b < 1 ∧ b < a) ∧ ¬ (a < 1 ∧ a < b))) :=
by intros ha hb ha_ne1 hb_ne1 hlog; sorry

end incorrect_conclusions_l81_81648


namespace smallest_m_l81_81165

noncomputable def f (x : ℝ) : ℝ := sorry

theorem smallest_m (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x) (hy : y ≤ 1) (h_eq : f 0 = f 1) 
(h_lt : forall x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → |f x - f y| < |x - y|): 
|f x - f y| < 1 / 2 := 
sorry

end smallest_m_l81_81165


namespace remainder_when_divided_by_8_l81_81742

theorem remainder_when_divided_by_8:
  ∀ (n : ℕ), (∃ (q : ℕ), n = 7 * q + 5) → n % 8 = 1 :=
by
  intro n h
  rcases h with ⟨q, hq⟩
  sorry

end remainder_when_divided_by_8_l81_81742


namespace quadratic_inequality_solution_l81_81237

theorem quadratic_inequality_solution:
  ∀ x : ℝ, (x^2 + 2 * x < 3) ↔ (-3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l81_81237


namespace max_value_2ab_sqrt2_plus_2ac_l81_81526

theorem max_value_2ab_sqrt2_plus_2ac (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 :=
sorry

end max_value_2ab_sqrt2_plus_2ac_l81_81526


namespace time_to_cross_platform_l81_81682

-- Definitions from conditions
def train_speed_kmph : ℕ := 72
def speed_conversion_factor : ℕ := 1000 / 3600
def train_speed_mps : ℤ := train_speed_kmph * speed_conversion_factor
def time_cross_man_sec : ℕ := 16
def platform_length_meters : ℕ := 280

-- Proving the total time to cross platform
theorem time_to_cross_platform : ∃ t : ℕ, t = (platform_length_meters + (train_speed_mps * time_cross_man_sec)) / train_speed_mps ∧ t = 30 := 
by
  -- Since the proof isn't required, we add "sorry" to act as a placeholder.
  sorry

end time_to_cross_platform_l81_81682


namespace max_right_angle_triangles_in_pyramid_l81_81937

noncomputable def pyramid_max_right_angle_triangles : Nat :=
  let pyramid : Type := { faces : Nat // faces = 4 }
  1

theorem max_right_angle_triangles_in_pyramid (p : pyramid) : pyramid_max_right_angle_triangles = 1 :=
  sorry

end max_right_angle_triangles_in_pyramid_l81_81937


namespace max_d_n_l81_81485

def sequence_a (n : ℕ) : ℤ := 100 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (sequence_a n) (sequence_a (n + 1))

theorem max_d_n : ∃ n, d_n n = 401 :=
by
  -- Placeholder for the actual proof
  sorry

end max_d_n_l81_81485


namespace initial_population_l81_81119

theorem initial_population (P : ℝ) (h : 0.78435 * P = 4500) : P = 5738 := 
by 
  sorry

end initial_population_l81_81119


namespace min_x_y_l81_81363

noncomputable def min_value (x y : ℝ) : ℝ := x + y

theorem min_x_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 16 * y = x * y) :
  min_value x y = 25 :=
sorry

end min_x_y_l81_81363


namespace tan_alpha_plus_pi_div_4_l81_81540

theorem tan_alpha_plus_pi_div_4 (α : ℝ) (hcos : Real.cos α = 3 / 5) (h0 : 0 < α) (hpi : α < Real.pi) :
  Real.tan (α + Real.pi / 4) = -7 :=
by
  sorry

end tan_alpha_plus_pi_div_4_l81_81540


namespace isosceles_triangle_perimeter_l81_81017

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6) (h2 : b = 13) 
  (triangle_inequality : b + b > a) : 
  (2 * b + a) = 32 := by
  sorry

end isosceles_triangle_perimeter_l81_81017


namespace total_apples_correct_l81_81148

def craig_initial := 20.5
def judy_initial := 11.25
def dwayne_initial := 17.85
def eugene_to_craig := 7.15
def craig_to_dwayne := 3.5 / 2
def judy_to_sally := judy_initial / 2

def craig_final := craig_initial + eugene_to_craig - craig_to_dwayne
def dwayne_final := dwayne_initial + craig_to_dwayne
def judy_final := judy_initial - judy_to_sally
def sally_final := judy_to_sally

def total_apples := craig_final + judy_final + dwayne_final + sally_final

theorem total_apples_correct : total_apples = 56.75 := by
  -- skipping proof
  sorry

end total_apples_correct_l81_81148


namespace inequality_always_true_l81_81702

variable (a b c : ℝ)

theorem inequality_always_true (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_always_true_l81_81702


namespace audrey_lost_pieces_l81_81206

theorem audrey_lost_pieces {total_pieces_on_board : ℕ} {thomas_lost : ℕ} {initial_pieces_each : ℕ} (h1 : total_pieces_on_board = 21) (h2 : thomas_lost = 5) (h3 : initial_pieces_each = 16) :
  (initial_pieces_each - (total_pieces_on_board - (initial_pieces_each - thomas_lost))) = 6 :=
by
  sorry

end audrey_lost_pieces_l81_81206


namespace cubic_polynomial_solution_l81_81966

noncomputable def q (x : ℝ) : ℝ := - (4 / 3) * x^3 + 6 * x^2 - (50 / 3) * x - (14 / 3)

theorem cubic_polynomial_solution :
  q 1 = -8 ∧ q 2 = -12 ∧ q 3 = -20 ∧ q 4 = -40 := by
  have h₁ : q 1 = -8 := by sorry
  have h₂ : q 2 = -12 := by sorry
  have h₃ : q 3 = -20 := by sorry
  have h₄ : q 4 = -40 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end cubic_polynomial_solution_l81_81966


namespace exists_special_number_divisible_by_1991_l81_81030

theorem exists_special_number_divisible_by_1991 :
  ∃ (N : ℤ) (n : ℕ), n > 2 ∧ (N % 1991 = 0) ∧ 
  (∃ a b x : ℕ, N = 10 ^ (n + 1) * a + 10 ^ n * x + 9 * 10 ^ (n - 1) + b) :=
sorry

end exists_special_number_divisible_by_1991_l81_81030


namespace percentage_games_won_l81_81917

def total_games_played : ℕ := 75
def win_rate_first_100_games : ℝ := 0.65

theorem percentage_games_won : 
  (win_rate_first_100_games * total_games_played / total_games_played * 100) = 65 := 
by
  sorry

end percentage_games_won_l81_81917


namespace correct_proposition_D_l81_81419

theorem correct_proposition_D (a b : ℝ) (h1 : a < 0) (h2 : b < 0) : 
  (b / a) + (a / b) ≥ 2 := 
sorry

end correct_proposition_D_l81_81419


namespace angle_of_inclination_l81_81399

theorem angle_of_inclination (θ : ℝ) : 
  (∀ x y : ℝ, x - y + 3 = 0 → ∃ θ : ℝ, Real.tan θ = 1 ∧ θ = Real.pi / 4) := by
  sorry

end angle_of_inclination_l81_81399


namespace andre_flowers_given_l81_81040

variable (initialFlowers totalFlowers flowersGiven : ℕ)

theorem andre_flowers_given (h1 : initialFlowers = 67) (h2 : totalFlowers = 90) :
  flowersGiven = totalFlowers - initialFlowers → flowersGiven = 23 :=
by
  intro h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end andre_flowers_given_l81_81040


namespace eighth_term_of_arithmetic_sequence_l81_81756

theorem eighth_term_of_arithmetic_sequence
  (a l : ℕ) (n : ℕ) (h₁ : a = 4) (h₂ : l = 88) (h₃ : n = 30) :
  (a + 7 * (l - a) / (n - 1) = (676 : ℚ) / 29) :=
by
  sorry

end eighth_term_of_arithmetic_sequence_l81_81756


namespace correct_calculation_is_c_l81_81214

theorem correct_calculation_is_c (a b : ℕ) :
  (2 * a ^ 2 * b) ^ 3 = 8 * a ^ 6 * b ^ 3 := 
sorry

end correct_calculation_is_c_l81_81214


namespace rectangular_solid_edges_sum_l81_81282

theorem rectangular_solid_edges_sum 
  (a b c : ℝ) 
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b^2 = a * c) : 
  4 * (a + b + c) = 32 := 
  sorry

end rectangular_solid_edges_sum_l81_81282


namespace slope_range_of_tangent_line_l81_81371

theorem slope_range_of_tangent_line (x : ℝ) (h : x ≠ 0) : (1 - 1/(x^2)) < 1 :=
by
  calc 
    1 - 1/(x^2) < 1 := sorry

end slope_range_of_tangent_line_l81_81371


namespace delta_gj_l81_81172

def vj := 120
def total := 770
def gj := total - vj

theorem delta_gj : gj - 5 * vj = 50 := by
  sorry

end delta_gj_l81_81172
