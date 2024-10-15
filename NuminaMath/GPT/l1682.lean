import Mathlib

namespace NUMINAMATH_GPT_train_passes_jogger_in_36_seconds_l1682_168298

/-- A jogger runs at 9 km/h, 240m ahead of a train moving at 45 km/h.
The train is 120m long. Prove the train passes the jogger in 36 seconds. -/
theorem train_passes_jogger_in_36_seconds
  (distance_ahead : ℝ)
  (jogger_speed_km_hr train_speed_km_hr train_length_m : ℝ)
  (jogger_speed_m_s train_speed_m_s relative_speed_m_s distance_to_cover time_to_pass : ℝ)
  (h1 : distance_ahead = 240)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_speed_km_hr = 45)
  (h4 : train_length_m = 120)
  (h5 : jogger_speed_m_s = jogger_speed_km_hr * 1000 / 3600)
  (h6 : train_speed_m_s = train_speed_km_hr * 1000 / 3600)
  (h7 : relative_speed_m_s = train_speed_m_s - jogger_speed_m_s)
  (h8 : distance_to_cover = distance_ahead + train_length_m)
  (h9 : time_to_pass = distance_to_cover / relative_speed_m_s) :
  time_to_pass = 36 := 
sorry

end NUMINAMATH_GPT_train_passes_jogger_in_36_seconds_l1682_168298


namespace NUMINAMATH_GPT_distance_between_trees_l1682_168297

def yard_length : ℝ := 1530
def number_of_trees : ℝ := 37
def number_of_gaps := number_of_trees - 1

theorem distance_between_trees :
  number_of_gaps ≠ 0 →
  (yard_length / number_of_gaps) = 42.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l1682_168297


namespace NUMINAMATH_GPT_bike_ride_energetic_time_l1682_168243

theorem bike_ride_energetic_time :
  ∃ x : ℚ, (22 * x + 15 * (7.5 - x) = 142) ∧ x = (59 / 14) :=
by
  sorry

end NUMINAMATH_GPT_bike_ride_energetic_time_l1682_168243


namespace NUMINAMATH_GPT_peter_hunts_3_times_more_than_mark_l1682_168200

theorem peter_hunts_3_times_more_than_mark : 
  ∀ (Sam Rob Mark Peter : ℕ),
  Sam = 6 →
  Rob = Sam / 2 →
  Mark = (Sam + Rob) / 3 →
  Sam + Rob + Mark + Peter = 21 →
  Peter = 3 * Mark :=
by
  intros Sam Rob Mark Peter h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_peter_hunts_3_times_more_than_mark_l1682_168200


namespace NUMINAMATH_GPT_total_sheets_of_paper_l1682_168240

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end NUMINAMATH_GPT_total_sheets_of_paper_l1682_168240


namespace NUMINAMATH_GPT_prop_for_real_l1682_168259

theorem prop_for_real (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end NUMINAMATH_GPT_prop_for_real_l1682_168259


namespace NUMINAMATH_GPT_combined_avg_score_l1682_168231

-- Define the average scores
def avg_score_u : ℕ := 65
def avg_score_b : ℕ := 80
def avg_score_c : ℕ := 77

-- Define the ratio of the number of students
def ratio_u : ℕ := 4
def ratio_b : ℕ := 6
def ratio_c : ℕ := 5

-- Prove the combined average score
theorem combined_avg_score : (ratio_u * avg_score_u + ratio_b * avg_score_b + ratio_c * avg_score_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end NUMINAMATH_GPT_combined_avg_score_l1682_168231


namespace NUMINAMATH_GPT_operation_multiplication_in_P_l1682_168263

-- Define the set P
def P : Set ℕ := {n | ∃ k : ℕ, n = k^2}

-- Define the operation "*" as multiplication within the set P
def operation (a b : ℕ) : ℕ := a * b

-- Define the property to be proved
theorem operation_multiplication_in_P (a b : ℕ)
  (ha : a ∈ P) (hb : b ∈ P) : operation a b ∈ P :=
sorry

end NUMINAMATH_GPT_operation_multiplication_in_P_l1682_168263


namespace NUMINAMATH_GPT_time_to_run_100_meters_no_wind_l1682_168252

-- Definitions based on the conditions
variables (v w : ℝ)
axiom speed_with_wind : v + w = 9
axiom speed_against_wind : v - w = 7

-- The theorem statement to prove
theorem time_to_run_100_meters_no_wind : (100 / v) = 12.5 :=
by 
  sorry

end NUMINAMATH_GPT_time_to_run_100_meters_no_wind_l1682_168252


namespace NUMINAMATH_GPT_barbata_interest_rate_l1682_168286

theorem barbata_interest_rate
  (initial_investment: ℝ)
  (additional_investment: ℝ)
  (additional_rate: ℝ)
  (total_income_rate: ℝ)
  (total_income: ℝ)
  (h_total_investment_eq: initial_investment + additional_investment = 4800)
  (h_total_income_eq: 0.06 * (initial_investment + additional_investment) = total_income):
  (initial_investment * (r : ℝ) + additional_investment * additional_rate = total_income) →
  r = 0.04 := sorry

end NUMINAMATH_GPT_barbata_interest_rate_l1682_168286


namespace NUMINAMATH_GPT_cone_volume_ratio_l1682_168206

noncomputable def ratio_of_volumes (r h : ℝ) : ℝ :=
  let S1 := r^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 12
  let S2 := r^2 * (10 * Real.pi + 3 * Real.sqrt 3) / 12
  S1 / S2

theorem cone_volume_ratio (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  ratio_of_volumes r h = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3) :=
  sorry

end NUMINAMATH_GPT_cone_volume_ratio_l1682_168206


namespace NUMINAMATH_GPT_cheezit_bag_weight_l1682_168260

-- Definitions based on the conditions of the problem
def cheezit_bags : ℕ := 3
def calories_per_ounce : ℕ := 150
def run_minutes : ℕ := 40
def calories_per_minute : ℕ := 12
def excess_calories : ℕ := 420

-- Main theorem stating the question with the solution
theorem cheezit_bag_weight (x : ℕ) : 
  (calories_per_ounce * cheezit_bags * x) - (run_minutes * calories_per_minute) = excess_calories → 
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_cheezit_bag_weight_l1682_168260


namespace NUMINAMATH_GPT_find_c_l1682_168258

theorem find_c (c : ℝ) (h : (-c / 4) + (-c / 7) = 22) : c = -56 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1682_168258


namespace NUMINAMATH_GPT_min_orange_chips_l1682_168219

theorem min_orange_chips (p g o : ℕ)
    (h1: g ≥ (1 / 3) * p)
    (h2: g ≤ (1 / 4) * o)
    (h3: p + g ≥ 75) : o = 76 :=
    sorry

end NUMINAMATH_GPT_min_orange_chips_l1682_168219


namespace NUMINAMATH_GPT_find_a4_l1682_168264

-- Given expression of x^5
def polynomial_expansion (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5

theorem find_a4 (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (h : polynomial_expansion x a_0 a_1 a_2 a_3 a_4 a_5) : a_4 = -5 :=
  sorry

end NUMINAMATH_GPT_find_a4_l1682_168264


namespace NUMINAMATH_GPT_alpha_cubic_expression_l1682_168299

theorem alpha_cubic_expression (α : ℝ) (hα : α^2 - 8 * α - 5 = 0) : α^3 - 7 * α^2 - 13 * α + 6 = 11 :=
sorry

end NUMINAMATH_GPT_alpha_cubic_expression_l1682_168299


namespace NUMINAMATH_GPT_prove_fraction_identity_l1682_168289

theorem prove_fraction_identity 
  (x y z : ℝ)
  (h1 : (x * z) / (x + y) + (y * z) / (y + z) + (x * y) / (z + x) = -18)
  (h2 : (z * y) / (x + y) + (z * x) / (y + z) + (y * x) / (z + x) = 20) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 20.5 := 
by
  sorry

end NUMINAMATH_GPT_prove_fraction_identity_l1682_168289


namespace NUMINAMATH_GPT_probability_at_least_one_trip_l1682_168245

theorem probability_at_least_one_trip (p_A_trip : ℚ) (p_B_trip : ℚ)
  (h1 : p_A_trip = 1/4) (h2 : p_B_trip = 1/5) :
  (1 - ((1 - p_A_trip) * (1 - p_B_trip))) = 2/5 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_trip_l1682_168245


namespace NUMINAMATH_GPT_find_k_l1682_168278

theorem find_k (x k : ℝ) :
  (∀ x, x ∈ Set.Ioo (-4 : ℝ) 3 ↔ x * (x^2 - 9) < k) → k = 0 :=
  by
  sorry

end NUMINAMATH_GPT_find_k_l1682_168278


namespace NUMINAMATH_GPT_intersectionAandB_l1682_168242

def setA (x : ℝ) : Prop := abs (x + 3) + abs (x - 4) ≤ 9
def setB (x : ℝ) : Prop := ∃ t : ℝ, 0 < t ∧ x = 4 * t + 1 / t - 6

theorem intersectionAandB : {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end NUMINAMATH_GPT_intersectionAandB_l1682_168242


namespace NUMINAMATH_GPT_maximum_value_of_3m_4n_l1682_168215

noncomputable def max_value (m n : ℕ) : ℕ :=
  3 * m + 4 * n

theorem maximum_value_of_3m_4n 
  (m n : ℕ) 
  (h_even : ∀ i, i < m → (2 * (i + 1)) > 0) 
  (h_odd : ∀ j, j < n → (2 * j + 1) > 0)
  (h_sum : m * (m + 1) + n^2 ≤ 1987) 
  (h_odd_n : n % 2 = 1) :
  max_value m n ≤ 221 := 
sorry

end NUMINAMATH_GPT_maximum_value_of_3m_4n_l1682_168215


namespace NUMINAMATH_GPT_correlation_statements_l1682_168211

def heavy_snow_predicts_harvest_year (heavy_snow benefits_wheat : Prop) : Prop := benefits_wheat → heavy_snow
def great_teachers_produce_students (great_teachers outstanding_students : Prop) : Prop := great_teachers → outstanding_students
def smoking_is_harmful (smoking harmful_to_health : Prop) : Prop := smoking → harmful_to_health
def magpies_call_signifies_joy (magpies_call joy_signified : Prop) : Prop := joy_signified → magpies_call

theorem correlation_statements (heavy_snow benefits_wheat great_teachers outstanding_students smoking harmful_to_health magpies_call joy_signified : Prop)
  (H1 : heavy_snow_predicts_harvest_year heavy_snow benefits_wheat)
  (H2 : great_teachers_produce_students great_teachers outstanding_students)
  (H3 : smoking_is_harmful smoking harmful_to_health) :
  ¬ magpies_call_signifies_joy magpies_call joy_signified := sorry

end NUMINAMATH_GPT_correlation_statements_l1682_168211


namespace NUMINAMATH_GPT_airplane_distance_difference_l1682_168229

variable (a : ℝ)

theorem airplane_distance_difference :
  let wind_speed := 20
  (4 * a) - (3 * (a - wind_speed)) = a + 60 := by
  sorry

end NUMINAMATH_GPT_airplane_distance_difference_l1682_168229


namespace NUMINAMATH_GPT_equivalent_proof_problem_l1682_168244

theorem equivalent_proof_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l1682_168244


namespace NUMINAMATH_GPT_no_common_points_lines_l1682_168214

theorem no_common_points_lines (m : ℝ) : 
    ¬∃ x y : ℝ, (x + m^2 * y + 6 = 0) ∧ ((m - 2) * x + 3 * m * y + 2 * m = 0) ↔ m = 0 ∨ m = -1 := 
by 
    sorry

end NUMINAMATH_GPT_no_common_points_lines_l1682_168214


namespace NUMINAMATH_GPT_highest_value_of_a_divisible_by_8_l1682_168253

theorem highest_value_of_a_divisible_by_8 :
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (8 ∣ (100 * a + 16)) ∧ 
  (∀ (b : ℕ), (0 ≤ b ∧ b ≤ 9) → 8 ∣ (100 * b + 16) → b ≤ a) :=
sorry

end NUMINAMATH_GPT_highest_value_of_a_divisible_by_8_l1682_168253


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l1682_168284

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l1682_168284


namespace NUMINAMATH_GPT_find_k_value_l1682_168246

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1682_168246


namespace NUMINAMATH_GPT_maximum_illuminated_surfaces_l1682_168248

noncomputable def optimal_position (r R d : ℝ) (h : d > r + R) : ℝ :=
  d / (1 + Real.sqrt (R^3 / r^3))

theorem maximum_illuminated_surfaces (r R d : ℝ) (h : d > r + R) (h1 : r ≤ optimal_position r R d h) (h2 : optimal_position r R d h ≤ d - R) :
  (optimal_position r R d h = d / (1 + Real.sqrt (R^3 / r^3))) ∨ (optimal_position r R d h = r) :=
sorry

end NUMINAMATH_GPT_maximum_illuminated_surfaces_l1682_168248


namespace NUMINAMATH_GPT_molecular_weight_calc_l1682_168255

namespace MolecularWeightProof

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def number_of_H : ℕ := 1
def number_of_Br : ℕ := 1
def number_of_O : ℕ := 3

theorem molecular_weight_calc :
  (number_of_H * atomic_weight_H + number_of_Br * atomic_weight_Br + number_of_O * atomic_weight_O) = 128.91 :=
by
  sorry

end MolecularWeightProof

end NUMINAMATH_GPT_molecular_weight_calc_l1682_168255


namespace NUMINAMATH_GPT_number_of_taxis_l1682_168268

-- Define the conditions explicitly
def number_of_cars : ℕ := 3
def people_per_car : ℕ := 4
def number_of_vans : ℕ := 2
def people_per_van : ℕ := 5
def people_per_taxi : ℕ := 6
def total_people : ℕ := 58

-- Define the number of people in cars and vans
def people_in_cars := number_of_cars * people_per_car
def people_in_vans := number_of_vans * people_per_van
def people_in_taxis := total_people - (people_in_cars + people_in_vans)

-- The theorem we need to prove
theorem number_of_taxis : people_in_taxis / people_per_taxi = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_taxis_l1682_168268


namespace NUMINAMATH_GPT_hockey_league_games_l1682_168269

theorem hockey_league_games (n t : ℕ) (h1 : n = 15) (h2 : t = 1050) :
  ∃ k, ∀ team1 team2 : ℕ, team1 ≠ team2 → k = 10 :=
by
  -- Declare k as the number of times each team faces the other teams
  let k := 10
  -- Verify the total number of teams and games
  have hn : n = 15 := h1
  have ht : t = 1050 := h2
  -- For any two distinct teams, they face each other k times
  use k
  intros team1 team2 hneq
  -- Show that k equals 10 under given conditions
  exact rfl

end NUMINAMATH_GPT_hockey_league_games_l1682_168269


namespace NUMINAMATH_GPT_total_num_novels_receiving_prizes_l1682_168225

-- Definitions based on conditions
def total_prize_money : ℕ := 800
def first_place_prize : ℕ := 200
def second_place_prize : ℕ := 150
def third_place_prize : ℕ := 120
def remaining_award_amount : ℕ := 22

-- Total number of novels receiving prizes
theorem total_num_novels_receiving_prizes : 
  (3 + (total_prize_money - (first_place_prize + second_place_prize + third_place_prize)) / remaining_award_amount) = 18 :=
by {
  -- We leave the proof as an exercise (denoted by sorry)
  sorry
}

end NUMINAMATH_GPT_total_num_novels_receiving_prizes_l1682_168225


namespace NUMINAMATH_GPT_ones_digit_of_22_to_22_11_11_l1682_168238

theorem ones_digit_of_22_to_22_11_11 : (22 ^ (22 * (11 ^ 11))) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_22_to_22_11_11_l1682_168238


namespace NUMINAMATH_GPT_solve_inequality_1_find_range_of_a_l1682_168295

def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem solve_inequality_1 :
  {x : ℝ | f x ≥ 5} = {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 2} :=
by
  sorry
  
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2 * a - 5) ↔ -2 < a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_1_find_range_of_a_l1682_168295


namespace NUMINAMATH_GPT_forest_area_relationship_l1682_168271

variable (a b c x : ℝ)

theorem forest_area_relationship
    (hb : b = a * (1 + x))
    (hc : c = a * (1 + x) ^ 2) :
    a * c = b ^ 2 := by
  sorry

end NUMINAMATH_GPT_forest_area_relationship_l1682_168271


namespace NUMINAMATH_GPT_inequality_proof_l1682_168256

theorem inequality_proof (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
    (x * y * z) / ((1 + 5 * x) * (4 * x + 3 * y) * (5 * y + 6 * z) * (z + 18)) ≤ (1 : ℝ) / 5120 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1682_168256


namespace NUMINAMATH_GPT_power_of_two_last_digit_product_divisible_by_6_l1682_168293

theorem power_of_two_last_digit_product_divisible_by_6 (n : Nat) (h : 3 < n) :
  ∃ d m : Nat, (2^n = 10 * m + d) ∧ (m * d) % 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_last_digit_product_divisible_by_6_l1682_168293


namespace NUMINAMATH_GPT_exponentiation_rule_l1682_168275

theorem exponentiation_rule (a : ℝ) : (a^4) * (a^4) = a^8 :=
by 
  sorry

end NUMINAMATH_GPT_exponentiation_rule_l1682_168275


namespace NUMINAMATH_GPT_triangle_obtuse_l1682_168294

theorem triangle_obtuse
  (A B : ℝ) 
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (h : Real.cos A > Real.sin B) : 
  π / 2 < π - (A + B) ∧ π - (A + B) < π :=
by
  sorry

end NUMINAMATH_GPT_triangle_obtuse_l1682_168294


namespace NUMINAMATH_GPT_negation_of_proposition_l1682_168212

theorem negation_of_proposition :
  (¬ (∀ a b : ℤ, a = 0 → a * b = 0)) ↔ (∃ a b : ℤ, a = 0 ∧ a * b ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1682_168212


namespace NUMINAMATH_GPT_clayton_total_points_l1682_168209

theorem clayton_total_points 
  (game1 game2 game3 : ℕ)
  (game1_points : game1 = 10)
  (game2_points : game2 = 14)
  (game3_points : game3 = 6)
  (game4 : ℕ)
  (game4_points : game4 = (game1 + game2 + game3) / 3) :
  game1 + game2 + game3 + game4 = 40 :=
sorry

end NUMINAMATH_GPT_clayton_total_points_l1682_168209


namespace NUMINAMATH_GPT_exp_sum_l1682_168207

theorem exp_sum (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x + 3 * y) = 108 :=
sorry

end NUMINAMATH_GPT_exp_sum_l1682_168207


namespace NUMINAMATH_GPT_jonathan_typing_time_l1682_168235

theorem jonathan_typing_time 
(J : ℕ) 
(h_combined_rate : (1 / (J : ℝ)) + (1 / 30) + (1 / 24) = 1 / 10) : 
  J = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_jonathan_typing_time_l1682_168235


namespace NUMINAMATH_GPT_smallest_x_for_multiple_of_625_l1682_168220

theorem smallest_x_for_multiple_of_625 (x : ℕ) (hx_pos : 0 < x) : (500 * x) % 625 = 0 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_of_625_l1682_168220


namespace NUMINAMATH_GPT_sum_of_cube_angles_l1682_168272

theorem sum_of_cube_angles (W X Y Z : Point) (cube : Cube)
  (angle_WXY angle_XYZ angle_YZW angle_ZWX : ℝ)
  (h₁ : angle_WXY = 90)
  (h₂ : angle_XYZ = 90)
  (h₃ : angle_YZW = 90)
  (h₄ : angle_ZWX = 60) :
  angle_WXY + angle_XYZ + angle_YZW + angle_ZWX = 330 := by
  sorry

end NUMINAMATH_GPT_sum_of_cube_angles_l1682_168272


namespace NUMINAMATH_GPT_car_dealership_l1682_168285

variable (sportsCars : ℕ) (sedans : ℕ) (trucks : ℕ)

theorem car_dealership (h1 : 3 * sedans = 5 * sportsCars) 
  (h2 : 3 * trucks = 3 * sportsCars) 
  (h3 : sportsCars = 45) : 
  sedans = 75 ∧ trucks = 45 := by
  sorry

end NUMINAMATH_GPT_car_dealership_l1682_168285


namespace NUMINAMATH_GPT_negation_of_proposition_l1682_168205

theorem negation_of_proposition (x : ℝ) :
  ¬ (∃ x > -1, x^2 + x - 2018 > 0) ↔ ∀ x > -1, x^2 + x - 2018 ≤ 0 := sorry

end NUMINAMATH_GPT_negation_of_proposition_l1682_168205


namespace NUMINAMATH_GPT_sum_of_properly_paintable_numbers_l1682_168287

-- Definitions based on conditions
def properly_paintable (a b c : ℕ) : Prop :=
  ∀ n : ℕ, (n % a = 0 ∧ n % b ≠ 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b = 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b ≠ 1 ∧ n % c = 3) → n < 100

-- Main theorem to prove
theorem sum_of_properly_paintable_numbers : 
  (properly_paintable 3 3 6) ∧ (properly_paintable 4 2 8) → 
  100 * 3 + 10 * 3 + 6 + 100 * 4 + 10 * 2 + 8 = 764 :=
by
  sorry  -- The proof goes here, but it's not required

-- Note: The actual condition checks in the definition of properly_paintable 
-- might need more detailed splits into depending on specific post visits and a 
-- more rigorous formalization to comply with the exact checking as done above. 
-- This definition is a simplified logical structure to represent the condition.


end NUMINAMATH_GPT_sum_of_properly_paintable_numbers_l1682_168287


namespace NUMINAMATH_GPT_minutkin_bedtime_l1682_168290

def time_minutkin_goes_to_bed 
    (morning_time : ℕ) 
    (morning_turns : ℕ) 
    (night_turns : ℕ) 
    (morning_hours : ℕ) 
    (morning_minutes : ℕ)
    (hours_per_turn : ℕ) 
    (minutes_per_turn : ℕ) : Nat := 
    ((morning_hours * 60 + morning_minutes) - (night_turns * hours_per_turn * 60 + night_turns * minutes_per_turn)) % 1440 

theorem minutkin_bedtime : 
    time_minutkin_goes_to_bed 9 9 11 8 30 1 12 = 1290 :=
    sorry

end NUMINAMATH_GPT_minutkin_bedtime_l1682_168290


namespace NUMINAMATH_GPT_johns_uncommon_cards_l1682_168296

def packs_bought : ℕ := 10
def cards_per_pack : ℕ := 20
def uncommon_fraction : ℚ := 1 / 4

theorem johns_uncommon_cards : packs_bought * (cards_per_pack * uncommon_fraction) = (50 : ℚ) := 
by 
  sorry

end NUMINAMATH_GPT_johns_uncommon_cards_l1682_168296


namespace NUMINAMATH_GPT_find_middle_number_l1682_168208

theorem find_middle_number (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 22) (h4 : x + z = 29) (h5 : y + z = 31) (h6 : x = 10) :
  y = 12 :=
sorry

end NUMINAMATH_GPT_find_middle_number_l1682_168208


namespace NUMINAMATH_GPT_find_other_number_l1682_168254

theorem find_other_number 
  {A B : ℕ} 
  (h_A : A = 24)
  (h_hcf : Nat.gcd A B = 14)
  (h_lcm : Nat.lcm A B = 312) :
  B = 182 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_find_other_number_l1682_168254


namespace NUMINAMATH_GPT_spider_travel_distance_l1682_168232

theorem spider_travel_distance (r : ℝ) (journey3 : ℝ) (diameter : ℝ) (leg2 : ℝ) :
    r = 75 → journey3 = 110 → diameter = 2 * r → 
    leg2 = Real.sqrt (diameter^2 - journey3^2) → 
    diameter + leg2 + journey3 = 362 :=
by
  sorry

end NUMINAMATH_GPT_spider_travel_distance_l1682_168232


namespace NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l1682_168216

theorem tan_theta_minus_pi_over_4 (θ : Real) (k : ℤ)
  (h1 : - (π / 2) + (2 * k * π) < θ)
  (h2 : θ < 2 * k * π)
  (h3 : Real.sin (θ + π / 4) = 3 / 5) :
  Real.tan (θ - π / 4) = -4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l1682_168216


namespace NUMINAMATH_GPT_irrational_sqrt3_l1682_168204

theorem irrational_sqrt3 : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a * a = 3 * b * b) :=
by
  sorry

end NUMINAMATH_GPT_irrational_sqrt3_l1682_168204


namespace NUMINAMATH_GPT_strategy_probabilities_l1682_168281

noncomputable def P1 : ℚ := 1 / 3
noncomputable def P2 : ℚ := 1 / 2
noncomputable def P3 : ℚ := 2 / 3

theorem strategy_probabilities :
  (P1 < P2) ∧
  (P1 < P3) ∧
  (2 * P1 = P3) := by
  sorry

end NUMINAMATH_GPT_strategy_probabilities_l1682_168281


namespace NUMINAMATH_GPT_elizabeth_wedding_gift_cost_l1682_168265

-- Defining the given conditions
def cost_steak_knife_set : ℝ := 80.00
def num_steak_knife_sets : ℝ := 2
def cost_dinnerware_set : ℝ := 200.00
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Calculating total expense
def total_cost (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set : ℝ) : ℝ :=
  (cost_steak_knife_set * num_steak_knife_sets) + cost_dinnerware_set

def discounted_price (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (total_cost * discount_rate)

def final_price (discounted_price sales_tax_rate : ℝ) : ℝ :=
  discounted_price + (discounted_price * sales_tax_rate)

def elizabeth_spends (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate : ℝ) : ℝ :=
  final_price (discounted_price (total_cost cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set) discount_rate) sales_tax_rate

theorem elizabeth_wedding_gift_cost
  (cost_steak_knife_set : ℝ)
  (num_steak_knife_sets : ℝ)
  (cost_dinnerware_set : ℝ)
  (discount_rate : ℝ)
  (sales_tax_rate : ℝ) :
  elizabeth_spends cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate = 340.20 := 
by
  sorry -- Proof is to be completed

end NUMINAMATH_GPT_elizabeth_wedding_gift_cost_l1682_168265


namespace NUMINAMATH_GPT_gratuities_charged_l1682_168234

-- Define the conditions in the problem
def total_bill : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def ny_striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Calculate the total cost before tax and gratuities
def subtotal : ℝ := ny_striploin_cost + wine_cost

-- Calculate the taxes paid
def tax : ℝ := subtotal * sales_tax_rate

-- Calculate the total bill before gratuities
def total_before_gratuities : ℝ := subtotal + tax

-- Goal: Prove that gratuities charged is 41
theorem gratuities_charged : (total_bill - total_before_gratuities) = 41 := by sorry

end NUMINAMATH_GPT_gratuities_charged_l1682_168234


namespace NUMINAMATH_GPT_min_m_value_l1682_168270

theorem min_m_value :
  ∃ (x y m : ℝ), x - y + 2 ≥ 0 ∧ x + y - 2 ≤ 0 ∧ 2 * y ≥ x + 2 ∧
  (m > 0) ∧ (x^2 / 4 + y^2 = m^2) ∧ m = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_min_m_value_l1682_168270


namespace NUMINAMATH_GPT_parabola_vertex_point_sum_l1682_168273

theorem parabola_vertex_point_sum (a b c : ℚ) 
  (h1 : ∃ (a b c : ℚ), ∀ x : ℚ, (y = a * x ^ 2 + b * x + c) = (y = - (1 / 3) * (x - 5) ^ 2 + 3)) 
  (h2 : ∀ x : ℚ, ((x = 2) ∧ (y = 0)) → (0 = a * 2 ^ 2 + b * 2 + c)) :
  a + b + c = -7 / 3 := 
sorry

end NUMINAMATH_GPT_parabola_vertex_point_sum_l1682_168273


namespace NUMINAMATH_GPT_f_positive_for_specific_a_l1682_168218

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x * Real.log x

theorem f_positive_for_specific_a (x : ℝ) (h : x > 0) :
  f x (Real.exp 3 / 4) > 0 := sorry

end NUMINAMATH_GPT_f_positive_for_specific_a_l1682_168218


namespace NUMINAMATH_GPT_sales_worth_l1682_168274

def old_scheme_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_scheme_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)
def remuneration_difference (S : ℝ) : ℝ := new_scheme_remuneration S - old_scheme_remuneration S

theorem sales_worth (S : ℝ) (h : remuneration_difference S = 600) : S = 24000 :=
by
  sorry

end NUMINAMATH_GPT_sales_worth_l1682_168274


namespace NUMINAMATH_GPT_compute_m_n_sum_l1682_168227

theorem compute_m_n_sum :
  let AB := 10
  let BC := 15
  let height := 30
  let volume_ratio := 9
  let smaller_base_AB := AB / 3
  let smaller_base_BC := BC / 3
  let diagonal_AC := Real.sqrt (AB^2 + BC^2)
  let smaller_diagonal_A'C' := Real.sqrt ((smaller_base_AB)^2 + (smaller_base_BC)^2)
  let y_length := 145 / 9   -- derived from geometric considerations
  let YU := 20 + y_length
  let m := 325
  let n := 9
  YU = m / n ∧ Nat.gcd m n = 1 ∧ m + n = 334 :=
  by
  sorry

end NUMINAMATH_GPT_compute_m_n_sum_l1682_168227


namespace NUMINAMATH_GPT_cinema_cost_comparison_l1682_168201

theorem cinema_cost_comparison (x : ℕ) (hx : x = 1000) :
  let cost_A := if x ≤ 100 then 30 * x else 24 * x + 600
  let cost_B := 27 * x
  cost_A < cost_B :=
by
  sorry

end NUMINAMATH_GPT_cinema_cost_comparison_l1682_168201


namespace NUMINAMATH_GPT_solution_l1682_168249

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * 3 * x + 4

def problem (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : Prop :=
  f a b (-Real.logb 3 3) = 3

theorem solution (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : problem a b m h1 h2 :=
sorry

end NUMINAMATH_GPT_solution_l1682_168249


namespace NUMINAMATH_GPT_joseph_total_cost_l1682_168262

variable (cost_refrigerator cost_water_heater cost_oven : ℝ)

-- Conditions
axiom h1 : cost_refrigerator = 3 * cost_water_heater
axiom h2 : cost_oven = 500
axiom h3 : cost_oven = 2 * cost_water_heater

-- Theorem
theorem joseph_total_cost : cost_refrigerator + cost_water_heater + cost_oven = 1500 := by
  sorry

end NUMINAMATH_GPT_joseph_total_cost_l1682_168262


namespace NUMINAMATH_GPT_point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l1682_168228

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 4 * y - m = 0

def inside_circle (x y m : ℝ) : Prop :=
  (x-1)^2 + (y+2)^2 < 5 + m

theorem point_A_inside_circle (m : ℝ) : -1 < m ∧ m < 4 ↔ inside_circle m (-2) m :=
sorry

def circle_equation_m_4 (x y : ℝ) : Prop :=
  circle_equation x y 4

def dist_square_to_point_H (x y : ℝ) : ℝ :=
  (x - 4)^2 + (y - 2)^2

theorem max_min_dist_square_on_circle (P : ℝ × ℝ) :
  circle_equation_m_4 P.1 P.2 →
  4 ≤ dist_square_to_point_H P.1 P.2 ∧ dist_square_to_point_H P.1 P.2 ≤ 64 :=
sorry

def line_equation (m x y : ℝ) : Prop :=
  y = x + m

theorem chord_through_origin (m : ℝ) :
  ∃ m : ℝ, line_equation m (1 : ℝ) (-2 : ℝ) ∧ 
  (m = -4 ∨ m = 1) :=
sorry

end NUMINAMATH_GPT_point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l1682_168228


namespace NUMINAMATH_GPT_orange_juice_fraction_in_mixture_l1682_168251

theorem orange_juice_fraction_in_mixture :
  let capacity1 := 800
  let capacity2 := 700
  let fraction1 := (1 : ℚ) / 4
  let fraction2 := (3 : ℚ) / 7
  let orange_juice1 := capacity1 * fraction1
  let orange_juice2 := capacity2 * fraction2
  let total_orange_juice := orange_juice1 + orange_juice2
  let total_volume := capacity1 + capacity2
  let fraction := total_orange_juice / total_volume
  fraction = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_GPT_orange_juice_fraction_in_mixture_l1682_168251


namespace NUMINAMATH_GPT_image_of_2_in_set_B_l1682_168237

theorem image_of_2_in_set_B (f : ℤ → ℤ) (h : ∀ x, f x = 2 * x + 1) : f 2 = 5 :=
by
  apply h

end NUMINAMATH_GPT_image_of_2_in_set_B_l1682_168237


namespace NUMINAMATH_GPT_sum_of_two_numbers_is_10_l1682_168283

variable (a b : ℝ)

theorem sum_of_two_numbers_is_10
  (h1 : a + b = 10)
  (h2 : a - b = 8)
  (h3 : a^2 - b^2 = 80) :
  a + b = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_is_10_l1682_168283


namespace NUMINAMATH_GPT_forgot_days_l1682_168230

def July_days : ℕ := 31
def days_took_capsules : ℕ := 27

theorem forgot_days : July_days - days_took_capsules = 4 :=
by
  sorry

end NUMINAMATH_GPT_forgot_days_l1682_168230


namespace NUMINAMATH_GPT_percentage_less_A_than_B_l1682_168224

theorem percentage_less_A_than_B :
  ∀ (full_marks A_marks D_marks C_marks B_marks : ℝ),
    full_marks = 500 →
    A_marks = 360 →
    D_marks = 0.80 * full_marks →
    C_marks = (1 - 0.20) * D_marks →
    B_marks = (1 + 0.25) * C_marks →
    ((B_marks - A_marks) / B_marks) * 100 = 10 :=
  by intros full_marks A_marks D_marks C_marks B_marks
     intros h_full h_A h_D h_C h_B
     sorry

end NUMINAMATH_GPT_percentage_less_A_than_B_l1682_168224


namespace NUMINAMATH_GPT_add_congruence_mul_congruence_l1682_168279

namespace ModularArithmetic

-- Define the congruence relation mod m
def is_congruent_mod (a b m : ℤ) : Prop := ∃ k : ℤ, a - b = k * m

-- Part (a): Proving a + c ≡ b + d (mod m)
theorem add_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a + c) (b + d) m :=
  sorry

-- Part (b): Proving a ⋅ c ≡ b ⋅ d (mod m)
theorem mul_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a * c) (b * d) m :=
  sorry

end ModularArithmetic

end NUMINAMATH_GPT_add_congruence_mul_congruence_l1682_168279


namespace NUMINAMATH_GPT_student_age_is_24_l1682_168276

-- Defining the conditions
variables (S M : ℕ)
axiom h1 : M = S + 26
axiom h2 : M + 2 = 2 * (S + 2)

-- The proof statement
theorem student_age_is_24 : S = 24 :=
by
  sorry

end NUMINAMATH_GPT_student_age_is_24_l1682_168276


namespace NUMINAMATH_GPT_log_stack_total_l1682_168239

theorem log_stack_total :
  let a := 5
  let l := 15
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 110 :=
sorry

end NUMINAMATH_GPT_log_stack_total_l1682_168239


namespace NUMINAMATH_GPT_general_term_is_correct_l1682_168266

variable (a : ℕ → ℤ)
variable (n : ℕ)

def is_arithmetic_sequence := ∃ d a₁, ∀ n, a n = a₁ + d * (n - 1)

axiom a_10_eq_30 : a 10 = 30
axiom a_20_eq_50 : a 20 = 50

noncomputable def general_term (n : ℕ) : ℤ := 2 * n + 10

theorem general_term_is_correct (a: ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 10 = 30)
  (h3 : a 20 = 50)
  : ∀ n, a n = general_term n :=
sorry

end NUMINAMATH_GPT_general_term_is_correct_l1682_168266


namespace NUMINAMATH_GPT_intersection_of_sets_l1682_168217

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1682_168217


namespace NUMINAMATH_GPT_dihedral_angles_pyramid_l1682_168280

noncomputable def dihedral_angles (a b : ℝ) : ℝ × ℝ :=
  let alpha := Real.arccos ((a * Real.sqrt 3) / Real.sqrt (4 * b ^ 2 - a ^ 2))
  let gamma := 2 * Real.arctan (b / Real.sqrt (4 * b ^ 2 - a ^ 2))
  (alpha, gamma)

theorem dihedral_angles_pyramid (a b alpha gamma : ℝ) (h1 : a > 0) (h2 : b > 0) :
  dihedral_angles a b = (alpha, gamma) :=
sorry

end NUMINAMATH_GPT_dihedral_angles_pyramid_l1682_168280


namespace NUMINAMATH_GPT_polygon_sides_l1682_168223

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 > 2970) :
  n = 19 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1682_168223


namespace NUMINAMATH_GPT_parabola_focus_directrix_eq_l1682_168250

open Real

def distance (p : ℝ × ℝ) (l : ℝ) : ℝ := abs (p.fst - l)

def parabola_eq (focus_x focus_y l : ℝ) : Prop :=
  ∀ x y, (distance (x, y) focus_x = distance (x, y) l) ↔ y^2 = 2 * x - 1

theorem parabola_focus_directrix_eq :
  parabola_eq 1 0 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_directrix_eq_l1682_168250


namespace NUMINAMATH_GPT_simplify_expression_l1682_168203

theorem simplify_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  ((a^3 - a^2 * b) / (a^2 * b) - (a^2 * b - b^3) / (a * b - b^2) - (a * b) / (a^2 - b^2)) = 
  (-3 * a) / (a^2 - b^2) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1682_168203


namespace NUMINAMATH_GPT_proof_problem_l1682_168282

variable {α : Type*} [LinearOrderedField α]

theorem proof_problem 
  (a b x y : α) 
  (h0 : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y)
  (h1 : a + b + x + y < 2)
  (h2 : a + b^2 = x + y^2)
  (h3 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1682_168282


namespace NUMINAMATH_GPT_samuel_remaining_distance_l1682_168261

noncomputable def remaining_distance
  (total_distance : ℕ)
  (segment1_speed : ℕ) (segment1_time : ℕ)
  (segment2_speed : ℕ) (segment2_time : ℕ)
  (segment3_speed : ℕ) (segment3_time : ℕ)
  (segment4_speed : ℕ) (segment4_time : ℕ) : ℕ :=
  total_distance -
  (segment1_speed * segment1_time +
   segment2_speed * segment2_time +
   segment3_speed * segment3_time +
   segment4_speed * segment4_time)

theorem samuel_remaining_distance :
  remaining_distance 1200 60 2 70 3 50 4 80 5 = 270 :=
by
  sorry

end NUMINAMATH_GPT_samuel_remaining_distance_l1682_168261


namespace NUMINAMATH_GPT_spent_on_new_tires_is_correct_l1682_168247

-- Conditions
def amount_spent_on_speakers : ℝ := 136.01
def amount_spent_on_cd_player : ℝ := 139.38
def total_amount_spent : ℝ := 387.85

-- Goal
def amount_spent_on_tires : ℝ := total_amount_spent - (amount_spent_on_speakers + amount_spent_on_cd_player)

theorem spent_on_new_tires_is_correct : 
  amount_spent_on_tires = 112.46 :=
by
  sorry

end NUMINAMATH_GPT_spent_on_new_tires_is_correct_l1682_168247


namespace NUMINAMATH_GPT_meaningful_sqrt_l1682_168241

theorem meaningful_sqrt (a : ℝ) (h : a - 4 ≥ 0) : a ≥ 4 :=
sorry

end NUMINAMATH_GPT_meaningful_sqrt_l1682_168241


namespace NUMINAMATH_GPT_smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l1682_168226

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

theorem smallest_positive_period_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem range_of_f : ∀ y : ℝ, y ∈ Set.range f ↔ -1 ≤ y ∧ y ≤ 1 := by
  sorry

theorem intervals_monotonically_increasing : 
  ∀ k : ℤ, 
  ∀ x : ℝ, 
  (2 * k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 6) → 
  (f (x + Real.pi / 6) - f x) ≥ 0 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l1682_168226


namespace NUMINAMATH_GPT_train_length_l1682_168233

/-- 
Given that a train can cross an electric pole in 200 seconds and its speed is 18 km/h,
prove that the length of the train is 1000 meters.
-/
theorem train_length
  (time_to_cross : ℕ)
  (speed_kmph : ℕ)
  (h_time : time_to_cross = 200)
  (h_speed : speed_kmph = 18)
  : (speed_kmph * 1000 / 3600 * time_to_cross = 1000) :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1682_168233


namespace NUMINAMATH_GPT_int_solve_ineq_l1682_168291

theorem int_solve_ineq (x : ℤ) : (x + 3)^3 ≤ 8 ↔ x ≤ -1 :=
by sorry

end NUMINAMATH_GPT_int_solve_ineq_l1682_168291


namespace NUMINAMATH_GPT_exponent_sum_l1682_168257

theorem exponent_sum : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end NUMINAMATH_GPT_exponent_sum_l1682_168257


namespace NUMINAMATH_GPT_rectangle_ratio_l1682_168221

noncomputable def ratio_of_sides (a b : ℝ) : ℝ := a / b

theorem rectangle_ratio (a b d : ℝ) (h1 : d = Real.sqrt (a^2 + b^2)) (h2 : (a/b)^2 = b/d) : 
  ratio_of_sides a b = (Real.sqrt 5 - 1) / 3 :=
by sorry

end NUMINAMATH_GPT_rectangle_ratio_l1682_168221


namespace NUMINAMATH_GPT_max_possible_cables_l1682_168222

theorem max_possible_cables (num_employees : ℕ) (num_brand_X : ℕ) (num_brand_Y : ℕ) 
  (max_connections : ℕ) (num_cables : ℕ) :
  num_employees = 40 →
  num_brand_X = 25 →
  num_brand_Y = 15 →
  max_connections = 3 →
  (∀ x : ℕ, x < max_connections → num_cables ≤ 3 * num_brand_Y) →
  num_cables = 45 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_max_possible_cables_l1682_168222


namespace NUMINAMATH_GPT_bowling_ball_weight_l1682_168210

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = 20 := by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l1682_168210


namespace NUMINAMATH_GPT_division_remainder_l1682_168292

theorem division_remainder : 
  ∀ (Dividend Divisor Quotient Remainder : ℕ), 
  Dividend = 760 → 
  Divisor = 36 → 
  Quotient = 21 → 
  Dividend = (Divisor * Quotient) + Remainder → 
  Remainder = 4 := 
by 
  intros Dividend Divisor Quotient Remainder h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  have h5 : 760 = 36 * 21 + Remainder := h4
  linarith

end NUMINAMATH_GPT_division_remainder_l1682_168292


namespace NUMINAMATH_GPT_factorization_correct_l1682_168267

def expression (x : ℝ) : ℝ := 16 * x^3 + 4 * x^2
def factored_expression (x : ℝ) : ℝ := 4 * x^2 * (4 * x + 1)

theorem factorization_correct (x : ℝ) : expression x = factored_expression x := 
by 
  sorry

end NUMINAMATH_GPT_factorization_correct_l1682_168267


namespace NUMINAMATH_GPT_printer_Y_time_l1682_168202

theorem printer_Y_time (T_y : ℝ) : 
    (12 * (1 / (1 / T_y + 1 / 20)) = 1.8) → T_y = 10 := 
by 
sorry

end NUMINAMATH_GPT_printer_Y_time_l1682_168202


namespace NUMINAMATH_GPT_solve_quadratic_l1682_168277

theorem solve_quadratic (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1682_168277


namespace NUMINAMATH_GPT_transformation_result_l1682_168236

noncomputable def initial_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x => f (x + a)

noncomputable def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x => f (k * x)

theorem transformation_result :
  (compress_horizontal (translate_left initial_function (Real.pi / 3)) 2) x = Real.sin (4 * x + (2 * Real.pi / 3)) :=
sorry

end NUMINAMATH_GPT_transformation_result_l1682_168236


namespace NUMINAMATH_GPT_average_age_of_students_l1682_168213

theorem average_age_of_students (A : ℝ) (h1 : ∀ n : ℝ, n = 20 → A + 1 = n) (h2 : ∀ k : ℝ, k = 40 → 19 * A + k = 20 * (A + 1)) : A = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_students_l1682_168213


namespace NUMINAMATH_GPT_fixed_point_l1682_168288

theorem fixed_point (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_l1682_168288
