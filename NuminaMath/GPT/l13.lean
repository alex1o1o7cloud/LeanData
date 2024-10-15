import Mathlib

namespace NUMINAMATH_GPT_find_a_l13_1357

def line1 (a : ℝ) (P : ℝ × ℝ) : Prop := 2 * P.1 - a * P.2 - 1 = 0

def line2 (P : ℝ × ℝ) : Prop := P.1 + 2 * P.2 = 0

theorem find_a (a : ℝ) :
  (∀ P : ℝ × ℝ, line1 a P ∧ line2 P) → a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l13_1357


namespace NUMINAMATH_GPT_range_of_m_l13_1390

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then 1 / (Real.exp x) + m * x^2
else Real.exp x + m * x^2

theorem range_of_m {m : ℝ} : (∀ m, ∃ x y, f x m = 0 ∧ f y m = 0 ∧ x ≠ y) ↔ m < -Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l13_1390


namespace NUMINAMATH_GPT_mixedGasTemperature_is_correct_l13_1391

noncomputable def mixedGasTemperature (V₁ V₂ p₁ p₂ T₁ T₂ : ℝ) : ℝ := 
  (p₁ * V₁ + p₂ * V₂) / ((p₁ * V₁) / T₁ + (p₂ * V₂) / T₂)

theorem mixedGasTemperature_is_correct :
  mixedGasTemperature 2 3 3 4 400 500 = 462 := by
    sorry

end NUMINAMATH_GPT_mixedGasTemperature_is_correct_l13_1391


namespace NUMINAMATH_GPT_sqrt_sum_l13_1377

theorem sqrt_sum : (Real.sqrt 50) + (Real.sqrt 32) = 9 * (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_l13_1377


namespace NUMINAMATH_GPT_angle_between_lines_at_most_l13_1373
-- Import the entire Mathlib library for general mathematical definitions

-- Define the problem statement in Lean 4
theorem angle_between_lines_at_most (n : ℕ) (h : n > 0) :
  ∃ (l1 l2 : ℝ), l1 ≠ l2 ∧ (n : ℝ) > 0 → ∃ θ, 0 ≤ θ ∧ θ ≤ 180 / n := by
  sorry

end NUMINAMATH_GPT_angle_between_lines_at_most_l13_1373


namespace NUMINAMATH_GPT_unique_ordered_triples_count_l13_1397

theorem unique_ordered_triples_count :
  ∃ (n : ℕ), n = 1 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  abc = 4 * (ab + bc + ca) ∧ a = c / 4 -> False :=
sorry

end NUMINAMATH_GPT_unique_ordered_triples_count_l13_1397


namespace NUMINAMATH_GPT_age_ratio_l13_1339

theorem age_ratio (B_age : ℕ) (H1 : B_age = 34) (A_age : ℕ) (H2 : A_age = B_age + 4) :
  (A_age + 10) / (B_age - 10) = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l13_1339


namespace NUMINAMATH_GPT_distance_to_school_is_correct_l13_1356

-- Define the necessary constants, variables, and conditions
def distance_to_market : ℝ := 2
def total_weekly_mileage : ℝ := 44
def school_trip_miles (x : ℝ) : ℝ := 16 * x
def market_trip_miles : ℝ := 2 * distance_to_market
def total_trip_miles (x : ℝ) : ℝ := school_trip_miles x + market_trip_miles

-- Prove that the distance from Philip's house to the children's school is 2.5 miles
theorem distance_to_school_is_correct (x : ℝ) (h : total_trip_miles x = total_weekly_mileage) :
  x = 2.5 :=
by
  -- Insert necessary proof steps starting with the provided hypothesis
  sorry

end NUMINAMATH_GPT_distance_to_school_is_correct_l13_1356


namespace NUMINAMATH_GPT_base_7_to_base_10_l13_1304

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end NUMINAMATH_GPT_base_7_to_base_10_l13_1304


namespace NUMINAMATH_GPT_points_in_quadrant_I_l13_1319

theorem points_in_quadrant_I (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → (x > 0) ∧ (y > 0) := by
  sorry

end NUMINAMATH_GPT_points_in_quadrant_I_l13_1319


namespace NUMINAMATH_GPT_average_revenue_per_hour_l13_1393

theorem average_revenue_per_hour 
    (sold_A_hour1 : ℕ) (sold_B_hour1 : ℕ) (sold_A_hour2 : ℕ) (sold_B_hour2 : ℕ)
    (price_A_hour1 : ℕ) (price_A_hour2 : ℕ) (price_B_constant : ℕ) : 
    (sold_A_hour1 = 10) ∧ (sold_B_hour1 = 5) ∧ (sold_A_hour2 = 2) ∧ (sold_B_hour2 = 3) ∧
    (price_A_hour1 = 3) ∧ (price_A_hour2 = 4) ∧ (price_B_constant = 2) →
    (54 / 2 = 27) :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_revenue_per_hour_l13_1393


namespace NUMINAMATH_GPT_integer_solutions_l13_1381

theorem integer_solutions (n : ℤ) : (n^2 + 1) ∣ (n^5 + 3) ↔ n = -3 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2 := 
sorry

end NUMINAMATH_GPT_integer_solutions_l13_1381


namespace NUMINAMATH_GPT_max_alpha_flights_achievable_l13_1383

def max_alpha_flights (n : ℕ) : ℕ :=
  let total_flights := n * (n - 1) / 2
  let max_beta_flights := n / 2
  total_flights - max_beta_flights

theorem max_alpha_flights_achievable (n : ℕ) : 
  ∃ k, k = n * (n - 1) / 2 - n / 2 ∧ k ≤ max_alpha_flights n :=
by
  sorry

end NUMINAMATH_GPT_max_alpha_flights_achievable_l13_1383


namespace NUMINAMATH_GPT_inheritance_amount_l13_1385

theorem inheritance_amount
  (x : ℝ)
  (H1 : 0.25 * x + 0.15 * (x - 0.25 * x) = 15000) : x = 41379 := 
sorry

end NUMINAMATH_GPT_inheritance_amount_l13_1385


namespace NUMINAMATH_GPT_garden_width_l13_1382

theorem garden_width :
  ∃ w l : ℝ, (2 * l + 2 * w = 60) ∧ (l * w = 200) ∧ (l = 2 * w) ∧ (w = 10) :=
by
  sorry

end NUMINAMATH_GPT_garden_width_l13_1382


namespace NUMINAMATH_GPT_race_head_start_l13_1341

variables {Va Vb L H : ℝ}

theorem race_head_start
  (h1 : Va = 20 / 14 * Vb)
  (h2 : L / Va = (L - H) / Vb) : 
  H = 3 / 10 * L :=
by
  sorry

end NUMINAMATH_GPT_race_head_start_l13_1341


namespace NUMINAMATH_GPT_sequence_2010_eq_4040099_l13_1300

def sequence_term (n : Nat) : Int :=
  if n % 2 = 0 then 
    (n^2 - 1 : Int) 
  else 
    -(n^2 - 1 : Int)

theorem sequence_2010_eq_4040099 : sequence_term 2010 = 4040099 := 
  by 
    sorry

end NUMINAMATH_GPT_sequence_2010_eq_4040099_l13_1300


namespace NUMINAMATH_GPT_third_row_number_of_trees_l13_1359

theorem third_row_number_of_trees (n : ℕ) 
  (divisible_by_7 : 84 % 7 = 0) 
  (divisible_by_6 : 84 % 6 = 0) 
  (divisible_by_n : 84 % n = 0) 
  (least_trees : 84 = 84): 
  n = 4 := 
sorry

end NUMINAMATH_GPT_third_row_number_of_trees_l13_1359


namespace NUMINAMATH_GPT_ratio_areas_ACEF_ADC_l13_1315

-- Define the basic geometric setup
variables (A B C D E F : Point) 
variables (BC CD DE : ℝ) 
variable (α : ℝ)
variables (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) 

-- Assuming the given conditions, we want to prove the ratio of areas
noncomputable def ratio_areas (α : ℝ) : ℝ := 4 * (1 - α)

theorem ratio_areas_ACEF_ADC (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) :
  ratio_areas α = 4 * (1 - α) :=
sorry

end NUMINAMATH_GPT_ratio_areas_ACEF_ADC_l13_1315


namespace NUMINAMATH_GPT_cost_of_article_l13_1314

theorem cost_of_article (C : ℝ) (G : ℝ)
    (h1 : G = 520 - C)
    (h2 : 1.08 * G = 580 - C) :
    C = 230 :=
by
    sorry

end NUMINAMATH_GPT_cost_of_article_l13_1314


namespace NUMINAMATH_GPT_find_k_in_expression_l13_1307

theorem find_k_in_expression :
  (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 :=
by
  sorry

end NUMINAMATH_GPT_find_k_in_expression_l13_1307


namespace NUMINAMATH_GPT_sum_of_two_smallest_l13_1318

variable (a b c d : ℕ)
variable (x : ℕ)

-- Four numbers a, b, c, d are in the ratio 3:5:7:9
def ratios := (a = 3 * x) ∧ (b = 5 * x) ∧ (c = 7 * x) ∧ (d = 9 * x)

-- The average of these numbers is 30
def average := (a + b + c + d) / 4 = 30

-- The theorem to prove the sum of the two smallest numbers (a and b) is 40
theorem sum_of_two_smallest (h1 : ratios a b c d x) (h2 : average a b c d) : a + b = 40 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_smallest_l13_1318


namespace NUMINAMATH_GPT_f_inv_f_inv_15_l13_1374

def f (x : ℝ) : ℝ := 3 * x + 6

noncomputable def f_inv (x : ℝ) : ℝ := (x - 6) / 3

theorem f_inv_f_inv_15 : f_inv (f_inv 15) = -1 :=
by
  sorry

end NUMINAMATH_GPT_f_inv_f_inv_15_l13_1374


namespace NUMINAMATH_GPT_x_intercept_is_2_l13_1328

noncomputable def x_intercept_of_line : ℝ :=
  by
  sorry -- This is where the proof would go

theorem x_intercept_is_2 :
  (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → y = 0 → x = 2) :=
  by
  intro x y H_eq H_y0
  rw [H_y0] at H_eq
  simp at H_eq
  sorry -- This is where the proof would go

end NUMINAMATH_GPT_x_intercept_is_2_l13_1328


namespace NUMINAMATH_GPT_vertices_of_parabolas_is_parabola_l13_1337

theorem vertices_of_parabolas_is_parabola 
  (a c k : ℝ) (ha : 0 < a) (hc : 0 < c) (hk : 0 < k) :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) ∧ 
  ∀ (pt : ℝ × ℝ), (∃ t : ℝ, pt = (-(k * t) / (2 * a), f t)) → 
  ∃ a' b' c', (∀ t : ℝ, pt.2 = a' * pt.1^2 + b' * pt.1 + c') ∧ (a < 0) :=
by sorry

end NUMINAMATH_GPT_vertices_of_parabolas_is_parabola_l13_1337


namespace NUMINAMATH_GPT_circle_passing_through_points_eq_l13_1302

theorem circle_passing_through_points_eq :
  let A := (-2, 1)
  let B := (9, 3)
  let C := (1, 7)
  let center := (7/2, 2)
  let radius_sq := 125 / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_sq ↔ 
    (∃ t : ℝ, (x - center.1)^2 + (y - center.2)^2 = t^2) ∧
    ∀ P : ℝ × ℝ, P = A ∨ P = B ∨ P = C → (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius_sq := by sorry

end NUMINAMATH_GPT_circle_passing_through_points_eq_l13_1302


namespace NUMINAMATH_GPT_mangoes_per_kg_l13_1312

theorem mangoes_per_kg (total_kg : ℕ) (sold_market_kg : ℕ) (sold_community_factor : ℚ) (remaining_mangoes : ℕ) (mangoes_per_kg : ℕ) :
  total_kg = 60 ∧ sold_market_kg = 20 ∧ sold_community_factor = 1/2 ∧ remaining_mangoes = 160 → mangoes_per_kg = 8 :=
  by
  sorry

end NUMINAMATH_GPT_mangoes_per_kg_l13_1312


namespace NUMINAMATH_GPT_bicycle_parking_income_l13_1342

theorem bicycle_parking_income (x : ℝ) (y : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 2000)
    (h2 : y = 0.5 * x + 0.8 * (2000 - x)) : 
    y = -0.3 * x + 1600 := by
  sorry

end NUMINAMATH_GPT_bicycle_parking_income_l13_1342


namespace NUMINAMATH_GPT_adam_chocolate_boxes_l13_1398

theorem adam_chocolate_boxes 
  (c : ℕ) -- number of chocolate boxes Adam bought
  (h1 : 4 * c + 4 * 5 = 28) : 
  c = 2 := 
by
  sorry

end NUMINAMATH_GPT_adam_chocolate_boxes_l13_1398


namespace NUMINAMATH_GPT_identify_roles_l13_1335

-- Define the number of liars and truth-tellers
def num_liars : Nat := 1000
def num_truth_tellers : Nat := 1000

-- Define the properties of the individuals
def first_person_is_liar := true
def second_person_is_truth_teller := true

-- The main statement equivalent to the problem
theorem identify_roles : first_person_is_liar = true ∧ second_person_is_truth_teller = true := by
  sorry

end NUMINAMATH_GPT_identify_roles_l13_1335


namespace NUMINAMATH_GPT_total_hunts_l13_1352

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end NUMINAMATH_GPT_total_hunts_l13_1352


namespace NUMINAMATH_GPT_fraction_capacity_noah_ali_l13_1389

def capacity_Ali_closet : ℕ := 200
def total_capacity_Noah_closet : ℕ := 100
def each_capacity_Noah_closet : ℕ := total_capacity_Noah_closet / 2

theorem fraction_capacity_noah_ali : (each_capacity_Noah_closet : ℚ) / capacity_Ali_closet = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_fraction_capacity_noah_ali_l13_1389


namespace NUMINAMATH_GPT_homework_duration_reduction_l13_1336

theorem homework_duration_reduction (x : ℝ) (initial_duration final_duration : ℝ) (h_initial : initial_duration = 90) (h_final : final_duration = 60) : 
  90 * (1 - x)^2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_homework_duration_reduction_l13_1336


namespace NUMINAMATH_GPT_both_teams_joint_renovation_team_renovation_split_l13_1332

-- Problem setup for part 1
def renovation_total_length : ℕ := 2400
def teamA_daily_progress : ℕ := 30
def teamB_daily_progress : ℕ := 50
def combined_days_to_complete_renovation : ℕ := 30

theorem both_teams_joint_renovation (x : ℕ) :
  (teamA_daily_progress + teamB_daily_progress) * x = renovation_total_length → 
  x = combined_days_to_complete_renovation :=
by
  sorry

-- Problem setup for part 2
def total_renovation_days : ℕ := 60
def length_renovated_by_teamA : ℕ := 900
def length_renovated_by_teamB : ℕ := 1500

theorem team_renovation_split (a b : ℕ) :
  a / teamA_daily_progress + b / teamB_daily_progress = total_renovation_days ∧ 
  a + b = renovation_total_length → 
  a = length_renovated_by_teamA ∧ b = length_renovated_by_teamB :=
by
  sorry

end NUMINAMATH_GPT_both_teams_joint_renovation_team_renovation_split_l13_1332


namespace NUMINAMATH_GPT_no_square_cube_l13_1334

theorem no_square_cube (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, k^2 = n * (n + 1) * (n + 2) * (n + 3)) ∧ ¬ (∃ l : ℕ, l^3 = n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end NUMINAMATH_GPT_no_square_cube_l13_1334


namespace NUMINAMATH_GPT_solve_system_of_equations_l13_1331

theorem solve_system_of_equations (x y : ℝ) :
    (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧ 5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
    (x = 2 ∧ y = 1) ∨ (x = 2 / 5 ∧ y = -(1 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l13_1331


namespace NUMINAMATH_GPT_hyperbola_equation_l13_1348

theorem hyperbola_equation 
  (x y : ℝ)
  (h_ellipse : x^2 / 10 + y^2 / 5 = 1)
  (h_asymptote : 3 * x + 4 * y = 0)
  (h_hyperbola : ∃ k ≠ 0, 9 * x^2 - 16 * y^2 = k) :
  ∃ k : ℝ, k = 45 ∧ (x^2 / 5 - 16 * y^2 / 45 = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l13_1348


namespace NUMINAMATH_GPT_max_pawns_19x19_l13_1353

def maxPawnsOnChessboard (n : ℕ) := 
  n * n

theorem max_pawns_19x19 :
  maxPawnsOnChessboard 19 = 361 := 
by
  sorry

end NUMINAMATH_GPT_max_pawns_19x19_l13_1353


namespace NUMINAMATH_GPT_evening_to_morning_ratio_l13_1316

-- Definitions based on conditions
def morning_miles : ℕ := 2
def total_miles : ℕ := 12
def evening_miles : ℕ := total_miles - morning_miles

-- Lean statement to prove the ratio
theorem evening_to_morning_ratio : evening_miles / morning_miles = 5 := by
  -- we simply state the final ratio we want to prove
  sorry

end NUMINAMATH_GPT_evening_to_morning_ratio_l13_1316


namespace NUMINAMATH_GPT_tim_original_vocab_l13_1399

theorem tim_original_vocab (days_in_year : ℕ) (years : ℕ) (learned_per_day : ℕ) (vocab_increase : ℝ) :
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  original_vocab = 14600 :=
by
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  show original_vocab = 14600
  sorry

end NUMINAMATH_GPT_tim_original_vocab_l13_1399


namespace NUMINAMATH_GPT_glove_pair_probability_l13_1346

/-- 
A box contains 6 pairs of black gloves (i.e., 12 black gloves) and 4 pairs of beige gloves (i.e., 8 beige gloves).
We need to prove that the probability of drawing a matching pair of gloves is 47/95.
-/
theorem glove_pair_probability : 
  let total_gloves := 20
  let black_gloves := 12
  let beige_gloves := 8
  let P1_black := (black_gloves / total_gloves) * ((black_gloves - 1) / (total_gloves - 1))
  let P2_beige := (beige_gloves / total_gloves) * ((beige_gloves - 1) / (total_gloves - 1))
  let total_probability := P1_black + P2_beige
  total_probability = 47 / 95 :=
sorry

end NUMINAMATH_GPT_glove_pair_probability_l13_1346


namespace NUMINAMATH_GPT_find_smaller_circle_radius_l13_1338

noncomputable def smaller_circle_radius (R : ℝ) : ℝ :=
  R / (Real.sqrt 2 - 1)

theorem find_smaller_circle_radius (R : ℝ) (x : ℝ) :
  (∀ (c1 c2 c3 c4 : ℝ),  c1 = c2 ∧ c2 = c3 ∧ c3 = c4 ∧ c4 = x
  ∧ c1 + c2 = 2 * c3 * Real.sqrt 2)
  → x = smaller_circle_radius R :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_find_smaller_circle_radius_l13_1338


namespace NUMINAMATH_GPT_gcd_8fact_11fact_9square_l13_1330

theorem gcd_8fact_11fact_9square : Nat.gcd (Nat.factorial 8) ((Nat.factorial 11) * 9^2) = 40320 := 
sorry

end NUMINAMATH_GPT_gcd_8fact_11fact_9square_l13_1330


namespace NUMINAMATH_GPT_cost_per_foot_l13_1308

theorem cost_per_foot (area : ℕ) (total_cost : ℕ) (side_length : ℕ) (perimeter : ℕ) (cost_per_foot : ℕ) :
  area = 289 → total_cost = 3944 → side_length = Nat.sqrt 289 → perimeter = 4 * 17 →
  cost_per_foot = total_cost / perimeter → cost_per_foot = 58 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_per_foot_l13_1308


namespace NUMINAMATH_GPT_value_of_bill_used_to_pay_l13_1347

-- Definitions of the conditions
def num_games : ℕ := 6
def cost_per_game : ℕ := 15
def num_change_bills : ℕ := 2
def change_per_bill : ℕ := 5
def total_cost : ℕ := num_games * cost_per_game
def total_change : ℕ := num_change_bills * change_per_bill

-- Proof statement: What was the value of the bill Jed used to pay
theorem value_of_bill_used_to_pay : 
  total_value = (total_cost + total_change) :=
by
  sorry

end NUMINAMATH_GPT_value_of_bill_used_to_pay_l13_1347


namespace NUMINAMATH_GPT_product_of_roots_l13_1327

theorem product_of_roots (p q r : ℝ) (hp : 3*p^3 - 9*p^2 + 5*p - 15 = 0) 
  (hq : 3*q^3 - 9*q^2 + 5*q - 15 = 0) (hr : 3*r^3 - 9*r^2 + 5*r - 15 = 0) :
  p * q * r = 5 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l13_1327


namespace NUMINAMATH_GPT_infinite_either_interval_exists_rational_infinite_elements_l13_1384

variable {ε : ℝ} (x : ℕ → ℝ) (hε : ε > 0) (hεlt : ε < 1/2)

-- Problem 1
theorem infinite_either_interval (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) :
  (∃ N : ℕ, ∀ n ≥ N, x n < 1/2) ∨ (∃ N : ℕ, ∀ n ≥ N, x n ≥ 1/2) :=
sorry

-- Problem 2
theorem exists_rational_infinite_elements (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) (hε : ε > 0) (hεlt : ε < 1/2) :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃ N : ℕ, ∀ n ≥ N, x n ∈ [α - ε, α + ε] :=
sorry

end NUMINAMATH_GPT_infinite_either_interval_exists_rational_infinite_elements_l13_1384


namespace NUMINAMATH_GPT_factor_expression_l13_1392

theorem factor_expression (x : ℚ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := sorry

end NUMINAMATH_GPT_factor_expression_l13_1392


namespace NUMINAMATH_GPT_total_distance_travelled_l13_1367

def speed_one_sail : ℕ := 25 -- knots
def speed_two_sails : ℕ := 50 -- knots
def conversion_factor : ℕ := 115 -- 1.15, in hundredths

def distance_in_nautical_miles : ℕ :=
  (2 * speed_one_sail) +      -- Two hours, one sail
  (3 * speed_two_sails) +     -- Three hours, two sails
  (1 * speed_one_sail) +      -- One hour, one sail, navigating around obstacles
  (2 * (speed_one_sail - speed_one_sail * 30 / 100)) -- Two hours, strong winds, 30% reduction in speed

def distance_in_land_miles : ℕ :=
  distance_in_nautical_miles * conversion_factor / 100 -- Convert to land miles

theorem total_distance_travelled : distance_in_land_miles = 299 := by
  sorry

end NUMINAMATH_GPT_total_distance_travelled_l13_1367


namespace NUMINAMATH_GPT_mike_spent_on_new_tires_l13_1345

-- Define the given amounts
def amount_spent_on_speakers : ℝ := 118.54
def total_amount_spent_on_car_parts : ℝ := 224.87

-- Define the amount spent on new tires
def amount_spent_on_new_tires : ℝ := total_amount_spent_on_car_parts - amount_spent_on_speakers

-- The theorem we want to prove
theorem mike_spent_on_new_tires : amount_spent_on_new_tires = 106.33 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_mike_spent_on_new_tires_l13_1345


namespace NUMINAMATH_GPT_new_number_is_100t_plus_10u_plus_3_l13_1379

theorem new_number_is_100t_plus_10u_plus_3 (t u : ℕ) (ht : t < 10) (hu : u < 10) :
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  new_number = 100 * t + 10 * u + 3 :=
by
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  show new_number = 100 * t + 10 * u + 3
  sorry

end NUMINAMATH_GPT_new_number_is_100t_plus_10u_plus_3_l13_1379


namespace NUMINAMATH_GPT_lucy_deposit_l13_1322

theorem lucy_deposit :
  ∃ D : ℝ, 
    let initial_balance := 65 
    let withdrawal := 4 
    let final_balance := 76 
    initial_balance + D - withdrawal = final_balance ∧ D = 15 :=
by
  -- sorry skips the proof
  sorry

end NUMINAMATH_GPT_lucy_deposit_l13_1322


namespace NUMINAMATH_GPT_three_layers_rug_area_l13_1378

theorem three_layers_rug_area :
  ∀ (A B C D E : ℝ),
    A + B + C = 212 →
    (A + B + C) - D - 2 * E = 140 →
    D = 24 →
    E = 24 :=
by
  intros A B C D E h1 h2 h3
  sorry

end NUMINAMATH_GPT_three_layers_rug_area_l13_1378


namespace NUMINAMATH_GPT_good_number_is_1008_l13_1311

-- Given conditions
def sum_1_to_2015 : ℕ := (2015 * (2015 + 1)) / 2
def sum_mod_2016 : ℕ := sum_1_to_2015 % 2016

-- The proof problem expressed in Lean
theorem good_number_is_1008 (x : ℕ) (h1 : sum_1_to_2015 = 2031120)
  (h2 : sum_mod_2016 = 1008) :
  x = 1008 ↔ (sum_1_to_2015 - x) % 2016 = 0 := by
  sorry

end NUMINAMATH_GPT_good_number_is_1008_l13_1311


namespace NUMINAMATH_GPT_triangle_inscribed_relation_l13_1376

noncomputable def herons_area (p a b c : ℝ) : ℝ := (p * (p - a) * (p - b) * (p - c)).sqrt

theorem triangle_inscribed_relation
  (S S' p p' : ℝ)
  (a b c a' b' c' r : ℝ)
  (h1 : r = S / p)
  (h2 : r = S' / p')
  (h3 : S = herons_area p a b c)
  (h4 : S' = herons_area p' a' b' c') :
  (p - a) * (p - b) * (p - c) / p = (p' - a') * (p' - b') * (p' - c') / p' :=
by sorry

end NUMINAMATH_GPT_triangle_inscribed_relation_l13_1376


namespace NUMINAMATH_GPT_mass_of_alcl3_formed_l13_1387

noncomputable def molarMass (atomicMasses : List (ℕ × ℕ)) : ℕ :=
atomicMasses.foldl (λ acc elem => acc + elem.1 * elem.2) 0

theorem mass_of_alcl3_formed :
  let atomic_mass_al := 26.98
  let atomic_mass_cl := 35.45
  let molar_mass_alcl3 := 2 * atomic_mass_al + 3 * atomic_mass_cl
  let moles_al2co3 := 10
  let moles_alcl3 := 2 * moles_al2co3
  let mass_alcl3 := moles_alcl3 * molar_mass_alcl3
  mass_alcl3 = 3206.2 := sorry

end NUMINAMATH_GPT_mass_of_alcl3_formed_l13_1387


namespace NUMINAMATH_GPT_total_cost_is_346_l13_1366

-- Definitions of the given conditions
def total_people : ℕ := 35 + 5 + 1
def total_lunches : ℕ := total_people + 3
def vegetarian_lunches : ℕ := 10
def gluten_free_lunches : ℕ := 5
def nut_free_lunches : ℕ := 3
def halal_lunches : ℕ := 4
def veg_and_gluten_free_lunches : ℕ := 2
def regular_cost : ℕ := 7
def special_cost : ℕ := 8
def veg_and_gluten_free_cost : ℕ := 9

-- Calculate regular lunches considering dietary overlaps
def regular_lunches : ℕ := 
  total_lunches - vegetarian_lunches - gluten_free_lunches - nut_free_lunches - halal_lunches + veg_and_gluten_free_lunches

-- Calculate costs per category of lunches
def total_regular_cost : ℕ := regular_lunches * regular_cost
def total_vegetarian_cost : ℕ := (vegetarian_lunches - veg_and_gluten_free_lunches) * special_cost
def total_gluten_free_cost : ℕ := gluten_free_lunches * special_cost
def total_nut_free_cost : ℕ := nut_free_lunches * special_cost
def total_halal_cost : ℕ := halal_lunches * special_cost
def total_veg_and_gluten_free_cost : ℕ := veg_and_gluten_free_lunches * veg_and_gluten_free_cost

-- Calculate total cost
def total_cost : ℕ :=
  total_regular_cost + total_vegetarian_cost + total_gluten_free_cost + total_nut_free_cost + total_halal_cost + total_veg_and_gluten_free_cost

-- Theorem stating the main question
theorem total_cost_is_346 : total_cost = 346 :=
  by
    -- This is where the proof would go
    sorry

end NUMINAMATH_GPT_total_cost_is_346_l13_1366


namespace NUMINAMATH_GPT_james_total_distance_l13_1365

-- Define the conditions
def speed_part1 : ℝ := 30  -- mph
def time_part1 : ℝ := 0.5  -- hours
def speed_part2 : ℝ := 2 * speed_part1  -- 2 * 30 mph
def time_part2 : ℝ := 2 * time_part1  -- 2 * 0.5 hours

-- Compute distances
def distance_part1 : ℝ := speed_part1 * time_part1
def distance_part2 : ℝ := speed_part2 * time_part2

-- Total distance
def total_distance : ℝ := distance_part1 + distance_part2

-- The theorem to prove
theorem james_total_distance :
  total_distance = 75 := 
sorry

end NUMINAMATH_GPT_james_total_distance_l13_1365


namespace NUMINAMATH_GPT_monthly_manufacturing_expenses_l13_1368

theorem monthly_manufacturing_expenses 
  (num_looms : ℕ) (total_sales_value : ℚ) 
  (monthly_establishment_charges : ℚ) 
  (decrease_in_profit : ℚ) 
  (sales_per_loom : ℚ) 
  (manufacturing_expenses_per_loom : ℚ) 
  (total_manufacturing_expenses : ℚ) : 
  num_looms = 80 → 
  total_sales_value = 500000 → 
  monthly_establishment_charges = 75000 → 
  decrease_in_profit = 4375 → 
  sales_per_loom = total_sales_value / num_looms → 
  manufacturing_expenses_per_loom = sales_per_loom - decrease_in_profit → 
  total_manufacturing_expenses = manufacturing_expenses_per_loom * num_looms →
  total_manufacturing_expenses = 150000 :=
by
  intros h_num_looms h_total_sales h_monthly_est_charges h_decrease_in_profit h_sales_per_loom h_manufacturing_expenses_per_loom h_total_manufacturing_expenses
  sorry

end NUMINAMATH_GPT_monthly_manufacturing_expenses_l13_1368


namespace NUMINAMATH_GPT_general_solution_of_differential_eq_l13_1333

noncomputable def y (x C : ℝ) : ℝ := x * (Real.exp (x ^ 2) + C)

theorem general_solution_of_differential_eq {x C : ℝ} (h : x ≠ 0) :
  let y' := (1 : ℝ) * (Real.exp (x ^ 2) + C) + x * (2 * x * Real.exp (x ^ 2))
  y' = (y x C / x) + 2 * x ^ 2 * Real.exp (x ^ 2) :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_general_solution_of_differential_eq_l13_1333


namespace NUMINAMATH_GPT_inequality_proof_l13_1326

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > y)
  (hy : y > 1)
  (hz : 1 > z)
  (hzpos : z > 0)
  (a : ℝ := (1 + x * z) / z)
  (b : ℝ := (1 + x * y) / x)
  (c : ℝ := (1 + y * z) / y) :
  a > b ∧ a > c :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l13_1326


namespace NUMINAMATH_GPT_simplify_and_evaluate_l13_1351

theorem simplify_and_evaluate (a b : ℤ) (h₁ : a = -1) (h₂ : b = 3) :
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l13_1351


namespace NUMINAMATH_GPT_initial_food_supplies_l13_1372

theorem initial_food_supplies (x : ℝ) 
  (h1 : (3 / 5) * x - (3 / 5) * ((3 / 5) * x) = 96) : x = 400 :=
by
  sorry

end NUMINAMATH_GPT_initial_food_supplies_l13_1372


namespace NUMINAMATH_GPT_quotient_of_division_l13_1340

theorem quotient_of_division (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 181) (h2 : divisor = 20) (h3 : remainder = 1) 
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 9 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_quotient_of_division_l13_1340


namespace NUMINAMATH_GPT_overall_average_score_l13_1388

structure Club where
  members : Nat
  average_score : Nat

def ClubA : Club := { members := 40, average_score := 90 }
def ClubB : Club := { members := 50, average_score := 81 }

theorem overall_average_score : 
  (ClubA.members * ClubA.average_score + ClubB.members * ClubB.average_score) / 
  (ClubA.members + ClubB.members) = 85 :=
by
  sorry

end NUMINAMATH_GPT_overall_average_score_l13_1388


namespace NUMINAMATH_GPT_find_two_numbers_l13_1323

noncomputable def x := 5 + 2 * Real.sqrt 5
noncomputable def y := 5 - 2 * Real.sqrt 5

theorem find_two_numbers :
  (x * y = 5) ∧ (x + y = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_two_numbers_l13_1323


namespace NUMINAMATH_GPT_point_on_graph_l13_1350

theorem point_on_graph (x y : ℝ) (h : y = 3 * x + 1) : (x, y) = (2, 7) :=
sorry

end NUMINAMATH_GPT_point_on_graph_l13_1350


namespace NUMINAMATH_GPT_triangle_perimeter_l13_1329

theorem triangle_perimeter (x : ℕ) :
  (x = 6 ∨ x = 3) →
  ∃ (a b c : ℕ), (a = x ∧ (b = x ∨ c = x)) ∧ 
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l13_1329


namespace NUMINAMATH_GPT_orange_balls_count_l13_1358

theorem orange_balls_count (P_black : ℚ) (O : ℕ) (total_balls : ℕ) 
  (condition1 : total_balls = O + 7 + 6) 
  (condition2 : P_black = 7 / total_balls) 
  (condition3 : P_black = 0.38095238095238093) :
  O = 5 := 
by
  sorry

end NUMINAMATH_GPT_orange_balls_count_l13_1358


namespace NUMINAMATH_GPT_Rachel_average_speed_l13_1317

noncomputable def total_distance : ℝ := 2 + 4 + 6

noncomputable def time_to_Alicia : ℝ := 2 / 3
noncomputable def time_to_Lisa : ℝ := 4 / 5
noncomputable def time_to_Nicholas : ℝ := 1 / 2

noncomputable def total_time : ℝ := (20 / 30) + (24 / 30) + (15 / 30)

noncomputable def average_speed : ℝ := total_distance / total_time

theorem Rachel_average_speed : average_speed = 360 / 59 :=
by
  sorry

end NUMINAMATH_GPT_Rachel_average_speed_l13_1317


namespace NUMINAMATH_GPT_bus_is_there_probability_l13_1386

noncomputable def probability_bus_present : ℚ :=
  let total_area := 90 * 90
  let triangle_area := (75 * 75) / 2
  let parallelogram_area := 75 * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem bus_is_there_probability :
  probability_bus_present = 7/16 :=
by
  sorry

end NUMINAMATH_GPT_bus_is_there_probability_l13_1386


namespace NUMINAMATH_GPT_students_in_both_clubs_l13_1321

theorem students_in_both_clubs:
  ∀ (U D S : Finset ℕ ), (U.card = 300) → (D.card = 100) → (S.card = 140) → (D ∪ S).card = 210 → (D ∩ S).card = 30 := 
sorry

end NUMINAMATH_GPT_students_in_both_clubs_l13_1321


namespace NUMINAMATH_GPT_find_second_derivative_at_1_l13_1306

-- Define the function f(x) and its second derivative
noncomputable def f (x : ℝ) := x * Real.exp x
noncomputable def f'' (x : ℝ) := (x + 2) * Real.exp x

-- State the theorem to be proved
theorem find_second_derivative_at_1 : f'' 1 = 2 * Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_find_second_derivative_at_1_l13_1306


namespace NUMINAMATH_GPT_compare_quadratics_maximize_rectangle_area_l13_1320

-- (Ⅰ) Problem statement for comparing quadratic expressions
theorem compare_quadratics (x : ℝ) : (x + 1) * (x - 3) > (x + 2) * (x - 4) := by
  sorry

-- (Ⅱ) Problem statement for maximizing rectangular area with given perimeter
theorem maximize_rectangle_area (x y : ℝ) (h : 2 * (x + y) = 36) : 
  x = 9 ∧ y = 9 ∧ x * y = 81 := by
  sorry

end NUMINAMATH_GPT_compare_quadratics_maximize_rectangle_area_l13_1320


namespace NUMINAMATH_GPT_number_of_girls_l13_1396

-- Define the problem conditions as constants
def total_saplings : ℕ := 44
def teacher_saplings : ℕ := 6
def boy_saplings : ℕ := 4
def girl_saplings : ℕ := 2
def total_students : ℕ := 12
def students_saplings : ℕ := total_saplings - teacher_saplings

-- The proof problem statement
theorem number_of_girls (x y : ℕ) (h1 : x + y = total_students)
  (h2 : boy_saplings * x + girl_saplings * y = students_saplings) :
  y = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l13_1396


namespace NUMINAMATH_GPT_leaf_distance_after_11_gusts_l13_1310

def distance_traveled (gusts : ℕ) (swirls : ℕ) (forward_per_gust : ℕ) (backward_per_swirl : ℕ) : ℕ :=
  (gusts * forward_per_gust) - (swirls * backward_per_swirl)

theorem leaf_distance_after_11_gusts :
  ∀ (forward_per_gust backward_per_swirl : ℕ),
  forward_per_gust = 5 →
  backward_per_swirl = 2 →
  distance_traveled 11 11 forward_per_gust backward_per_swirl = 33 :=
by
  intros forward_per_gust backward_per_swirl hfg hbs
  rw [hfg, hbs]
  unfold distance_traveled
  sorry

end NUMINAMATH_GPT_leaf_distance_after_11_gusts_l13_1310


namespace NUMINAMATH_GPT_gcd_lcm_product_l13_1305

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l13_1305


namespace NUMINAMATH_GPT_find_n_l13_1369

theorem find_n :
  ∃ n : ℕ, ∀ (a b c : ℕ), a + b + c = 200 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (n = a + b * c) ∧ (n = b + c * a) ∧ (n = c + a * b) → n = 199 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_n_l13_1369


namespace NUMINAMATH_GPT_distinct_integer_roots_l13_1395

-- Definitions of m and the polynomial equation.
def poly (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2 * (2 * m - 3) * x + 4 * m^2 - 14 * m + 8 = 0

-- Theorem stating that for m = 12 and m = 24, the polynomial has specific roots.
theorem distinct_integer_roots (m x : ℤ) (h1 : 4 < m) (h2 : m < 40) :
  (m = 12 ∨ m = 24) ∧ 
  ((m = 12 ∧ (x = 26 ∨ x = 16) ∧ poly m x) ∨
   (m = 24 ∧ (x = 52 ∨ x = 38) ∧ poly m x)) :=
by
  sorry

end NUMINAMATH_GPT_distinct_integer_roots_l13_1395


namespace NUMINAMATH_GPT_percentage_error_calculation_l13_1375

theorem percentage_error_calculation (x : ℝ) :
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  percentage_error = 64 := 
by
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  sorry

end NUMINAMATH_GPT_percentage_error_calculation_l13_1375


namespace NUMINAMATH_GPT_find_f_minus_3_l13_1301

def rational_function (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x / x) = 2 * x^2

theorem find_f_minus_3 (f : ℚ → ℚ) (h : rational_function f) : 
  f (-3) = 494 / 117 :=
by
  sorry

end NUMINAMATH_GPT_find_f_minus_3_l13_1301


namespace NUMINAMATH_GPT_min_possible_value_l13_1309

theorem min_possible_value (a b : ℤ) (h : a > b) :
  (∃ x : ℚ, x = (2 * a + 3 * b) / (a - 2 * b) ∧ (x + 1 / x = (2 : ℚ))) :=
sorry

end NUMINAMATH_GPT_min_possible_value_l13_1309


namespace NUMINAMATH_GPT_car_value_correct_l13_1361

-- Define the initial value and the annual decrease percentages
def initial_value : ℝ := 10000
def annual_decreases : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

-- Function to compute the value of the car after n years
def value_after_years (initial_value : ℝ) (annual_decreases : List ℝ) : ℝ :=
  annual_decreases.foldl (λ acc decrease => acc * (1 - decrease)) initial_value

-- The target value after 5 years
def target_value : ℝ := 5348.88

-- Theorem stating that the computed value matches the target value
theorem car_value_correct :
  value_after_years initial_value annual_decreases = target_value := 
sorry

end NUMINAMATH_GPT_car_value_correct_l13_1361


namespace NUMINAMATH_GPT_line_PQ_passes_through_fixed_point_l13_1364

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

-- Define the conditions for points P and Q on the hyperbola
def on_hyperbola (P Q : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2

-- Define the condition for perpendicular lines, given points A, P, and Q
def perpendicular (A P Q : ℝ × ℝ) : Prop :=
  ((P.2 - A.2) / (P.1 - A.1)) * ((Q.2 - A.2) / (Q.1 - A.1)) = -1

-- Define the main theorem to prove
theorem line_PQ_passes_through_fixed_point :
  ∀ (P Q : ℝ × ℝ), on_hyperbola P Q → perpendicular ⟨-1, 0⟩ P Q →
    ∃ (b : ℝ), ∀ (y : ℝ), (P.1 = y * P.2 + b ∨ Q.1 = y * Q.2 + b) → (b = 3) :=
by
  sorry

end NUMINAMATH_GPT_line_PQ_passes_through_fixed_point_l13_1364


namespace NUMINAMATH_GPT_linear_function_through_two_points_l13_1355

theorem linear_function_through_two_points :
  ∃ (k b : ℝ), (∀ x, y = k * x + b) ∧
  (k ≠ 0) ∧
  (3 = 2 * k + b) ∧
  (2 = 3 * k + b) ∧
  (∀ x, y = -x + 5) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_through_two_points_l13_1355


namespace NUMINAMATH_GPT_mean_of_remaining_students_l13_1360

theorem mean_of_remaining_students
  (n : ℕ) (h : n > 20)
  (mean_score_first_15 : ℝ)
  (mean_score_next_5 : ℝ)
  (overall_mean_score : ℝ) :
  mean_score_first_15 = 10 →
  mean_score_next_5 = 16 →
  overall_mean_score = 11 →
  ∀ a, a = (11 * n - 230) / (n - 20) := by
sorry

end NUMINAMATH_GPT_mean_of_remaining_students_l13_1360


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l13_1394

theorem geometric_sequence_third_term (r : ℕ) (h_r : 5 * r ^ 4 = 1620) : 5 * r ^ 2 = 180 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l13_1394


namespace NUMINAMATH_GPT_max_value_is_63_l13_1349

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^2 + 3*x*y + 4*y^2

theorem max_value_is_63 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (cond : x^2 - 3*x*y + 4*y^2 = 9) :
  max_value x y ≤ 63 :=
by
  sorry

end NUMINAMATH_GPT_max_value_is_63_l13_1349


namespace NUMINAMATH_GPT_average_weight_of_children_l13_1313

theorem average_weight_of_children
  (S_B S_G : ℕ)
  (avg_boys_weight : S_B = 8 * 160)
  (avg_girls_weight : S_G = 5 * 110) :
  (S_B + S_G) / 13 = 141 := 
by
  sorry

end NUMINAMATH_GPT_average_weight_of_children_l13_1313


namespace NUMINAMATH_GPT_consecutive_page_sum_l13_1303

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 479160) : n + (n + 1) + (n + 2) = 234 :=
sorry

end NUMINAMATH_GPT_consecutive_page_sum_l13_1303


namespace NUMINAMATH_GPT_possible_values_of_d_l13_1371

theorem possible_values_of_d :
  ∃ (e f d : ℤ), (e + 12) * (f + 12) = 1 ∧
  ∀ x, (x - d) * (x - 12) + 1 = (x + e) * (x + f) ↔ (d = 22 ∨ d = 26) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_d_l13_1371


namespace NUMINAMATH_GPT_problem_l13_1325

theorem problem (a b : ℝ) (h₁ : a = -a) (h₂ : b = 1 / b) : a + b = 1 ∨ a + b = -1 :=
  sorry

end NUMINAMATH_GPT_problem_l13_1325


namespace NUMINAMATH_GPT_largest_y_coordinate_l13_1362

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end NUMINAMATH_GPT_largest_y_coordinate_l13_1362


namespace NUMINAMATH_GPT_derivative_y_l13_1343

noncomputable def y (x : ℝ) : ℝ :=
  (1 / 4) * Real.log ((x - 1) / (x + 1)) - (1 / 2) * Real.arctan x

theorem derivative_y (x : ℝ) : deriv y x = 1 / (x^4 - 1) :=
  sorry

end NUMINAMATH_GPT_derivative_y_l13_1343


namespace NUMINAMATH_GPT_probability_of_number_between_21_and_30_l13_1344

-- Define the success condition of forming a two-digit number between 21 and 30.
def successful_number (d1 d2 : Nat) : Prop :=
  let n1 := 10 * d1 + d2
  let n2 := 10 * d2 + d1
  (21 ≤ n1 ∧ n1 ≤ 30) ∨ (21 ≤ n2 ∧ n2 ≤ 30)

-- Calculate the probability of a successful outcome.
def probability_success (favorable total : Nat) : Nat :=
  favorable / total

-- The main theorem claiming the probability that Melinda forms a number between 21 and 30.
theorem probability_of_number_between_21_and_30 :
  let successful_counts := 10
  let total_possible := 36
  probability_success successful_counts total_possible = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_number_between_21_and_30_l13_1344


namespace NUMINAMATH_GPT_Marias_score_l13_1380

def total_questions := 30
def points_per_correct_answer := 20
def points_deducted_per_incorrect_answer := 5
def total_answered := total_questions
def correct_answers := 19
def incorrect_answers := total_questions - correct_answers
def score := (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect_answer)

theorem Marias_score : score = 325 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_Marias_score_l13_1380


namespace NUMINAMATH_GPT_remainder_4x_div_9_l13_1324

theorem remainder_4x_div_9 (x : ℕ) (k : ℤ) (h : x = 9 * k + 5) : (4 * x) % 9 = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_4x_div_9_l13_1324


namespace NUMINAMATH_GPT_find_a_l13_1354

theorem find_a (a : ℝ) (h : ∫ x in -a..a, (2 * x - 1) = -8) : a = 4 :=
sorry

end NUMINAMATH_GPT_find_a_l13_1354


namespace NUMINAMATH_GPT_quadratic_equation_l13_1363

theorem quadratic_equation (p q : ℝ) 
  (h1 : p^2 + 9 * q^2 + 3 * p - p * q = 30)
  (h2 : p - 5 * q - 8 = 0) : 
  p^2 - p - 6 = 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_equation_l13_1363


namespace NUMINAMATH_GPT_calculate_x_value_l13_1370

theorem calculate_x_value : 
  529 + 2 * 23 * 3 + 9 = 676 := 
by
  sorry

end NUMINAMATH_GPT_calculate_x_value_l13_1370
