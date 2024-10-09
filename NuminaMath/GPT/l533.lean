import Mathlib

namespace rojo_speed_l533_53344

theorem rojo_speed (R : ℝ) 
  (H : 32 = (R + 3) * 4) : R = 5 :=
sorry

end rojo_speed_l533_53344


namespace compute_mod_expression_l533_53334

theorem compute_mod_expression :
  (3 * (1 / 7) + 9 * (1 / 13)) % 72 = 18 := sorry

end compute_mod_expression_l533_53334


namespace cricket_team_captain_age_l533_53307

theorem cricket_team_captain_age
    (C W : ℕ)
    (h1 : W = C + 3)
    (h2 : (23 * 11) = (22 * 9) + C + W)
    : C = 26 :=
by
    sorry

end cricket_team_captain_age_l533_53307


namespace sum_of_solutions_eq_3_l533_53360

theorem sum_of_solutions_eq_3 (x y : ℝ) (h1 : x * y = 1) (h2 : x + y = 3) :
  x + y = 3 := sorry

end sum_of_solutions_eq_3_l533_53360


namespace find_original_rabbits_l533_53355

theorem find_original_rabbits (R S : ℕ) (h1 : R + S = 50)
  (h2 : 4 * R + 8 * S = 2 * R + 16 * S) :
  R = 40 :=
sorry

end find_original_rabbits_l533_53355


namespace quadratic_completing_square_l533_53313

theorem quadratic_completing_square
  (a : ℤ) (b : ℤ) (c : ℤ)
  (h1 : a > 0)
  (h2 : 64 * a^2 * x^2 - 96 * x - 48 = 64 * x^2 - 96 * x - 48)
  (h3 : (a * x + b)^2 = c) :
  a + b + c = 86 :=
sorry

end quadratic_completing_square_l533_53313


namespace radius_of_circle_l533_53392

theorem radius_of_circle :
  ∃ r : ℝ, ∀ x : ℝ, (x^2 + r = x) ↔ (r = 1 / 4) :=
by
  sorry

end radius_of_circle_l533_53392


namespace interest_rate_calculation_l533_53350

-- Define the problem conditions and proof statement in Lean
theorem interest_rate_calculation 
  (P : ℝ) (r : ℝ) (T : ℝ) (CI SI diff : ℝ) 
  (principal_condition : P = 6000.000000000128)
  (time_condition : T = 2)
  (diff_condition : diff = 15)
  (CI_formula : CI = P * (1 + r)^T - P)
  (SI_formula : SI = P * r * T)
  (difference_condition : CI - SI = diff) : 
  r = 0.05 := 
by 
  sorry

end interest_rate_calculation_l533_53350


namespace equilateral_triangle_of_angle_and_side_sequences_l533_53383

variable {A B C a b c : ℝ}

theorem equilateral_triangle_of_angle_and_side_sequences
  (H_angles_arithmetic : 2 * B = A + C)
  (H_sum_angles : A + B + C = Real.pi)
  (H_sides_geometric : b^2 = a * c) :
  A = Real.pi / 3 ∧ B = Real.pi / 3 ∧ C = Real.pi / 3 ∧ a = b ∧ b = c :=
by
  sorry

end equilateral_triangle_of_angle_and_side_sequences_l533_53383


namespace amount_earned_from_each_family_l533_53356

theorem amount_earned_from_each_family
  (goal : ℕ) (earn_from_fifteen_families : ℕ) (additional_needed : ℕ) (three_families : ℕ) 
  (earn_from_three_families_total : ℕ) (per_family_earn : ℕ) :
  goal = 150 →
  earn_from_fifteen_families = 75 →
  additional_needed = 45 →
  three_families = 3 →
  earn_from_three_families_total = (goal - additional_needed) - earn_from_fifteen_families →
  per_family_earn = earn_from_three_families_total / three_families →
  per_family_earn = 10 :=
by
  sorry

end amount_earned_from_each_family_l533_53356


namespace find_b_l533_53386

theorem find_b (c b : ℤ) (h : ∃ k : ℤ, (x^2 - x - 1) * (c * x - 3) = c * x^3 + b * x^2 + 3) : b = -6 :=
by
  sorry

end find_b_l533_53386


namespace mode_of_scores_is_85_l533_53380

-- Define the scores based on the given stem-and-leaf plot
def scores : List ℕ := [50, 55, 55, 62, 62, 68, 70, 71, 75, 79, 81, 81, 83, 85, 85, 85, 92, 96, 96, 98, 100, 100]

-- Define a function to compute the mode
def mode (s : List ℕ) : ℕ :=
  s.foldl (λ acc x => if s.count x > s.count acc then x else acc) 0

-- The theorem to prove that the mode of the scores is 85
theorem mode_of_scores_is_85 : mode scores = 85 :=
by
  -- The proof is omitted
  sorry

end mode_of_scores_is_85_l533_53380


namespace container_volumes_l533_53324

variable (a : ℕ)

theorem container_volumes (h₁ : a = 18) :
  a^3 = 5832 ∧ (a - 4)^3 = 2744 ∧ (a - 6)^3 = 1728 :=
by {
  sorry
}

end container_volumes_l533_53324


namespace sector_area_l533_53394

/-- The area of a sector with a central angle of 72 degrees and a radius of 20 cm is 80π cm². -/
theorem sector_area (radius : ℝ) (angle : ℝ) (h_angle_deg : angle = 72) (h_radius : radius = 20) :
  (angle / 360) * π * radius^2 = 80 * π :=
by sorry

end sector_area_l533_53394


namespace probability_two_red_two_blue_one_green_l533_53358

def total_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_choose_red (r : ℕ) : ℕ := total_ways_to_choose 4 r
def ways_to_choose_blue (b : ℕ) : ℕ := total_ways_to_choose 3 b
def ways_to_choose_green (g : ℕ) : ℕ := total_ways_to_choose 2 g

def successful_outcomes (r b g : ℕ) : ℕ :=
  ways_to_choose_red r * ways_to_choose_blue b * ways_to_choose_green g

def total_outcomes : ℕ := total_ways_to_choose 9 5

def probability_of_selection (r b g : ℕ) : ℚ :=
  (successful_outcomes r b g : ℚ) / (total_outcomes : ℚ)

theorem probability_two_red_two_blue_one_green :
  probability_of_selection 2 2 1 = 2 / 7 := by
  sorry

end probability_two_red_two_blue_one_green_l533_53358


namespace length_of_train_is_correct_l533_53379

noncomputable def speed_kmh := 30 
noncomputable def time_s := 9 
noncomputable def speed_ms := (speed_kmh * 1000) / 3600 
noncomputable def length_of_train := speed_ms * time_s

theorem length_of_train_is_correct : length_of_train = 75 := 
by 
  sorry

end length_of_train_is_correct_l533_53379


namespace dryer_less_than_washing_machine_by_30_l533_53396

-- Definitions based on conditions
def washing_machine_price : ℝ := 100
def discount_rate : ℝ := 0.10
def total_paid_after_discount : ℝ := 153

-- The equation for price of the dryer
def original_dryer_price (D : ℝ) : Prop :=
  washing_machine_price + D - discount_rate * (washing_machine_price + D) = total_paid_after_discount

-- The statement we need to prove
theorem dryer_less_than_washing_machine_by_30 (D : ℝ) (h : original_dryer_price D) :
  washing_machine_price - D = 30 :=
by 
  sorry

end dryer_less_than_washing_machine_by_30_l533_53396


namespace geometric_progression_common_ratio_l533_53375

theorem geometric_progression_common_ratio (r : ℝ) (a : ℝ) (h_pos : 0 < a)
    (h_geom_prog : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) :
    r^3 + r^2 + r - 1 = 0 :=
by
  sorry

end geometric_progression_common_ratio_l533_53375


namespace average_income_family_l533_53374

theorem average_income_family (income1 income2 income3 income4 : ℕ) 
  (h1 : income1 = 8000) (h2 : income2 = 15000) (h3 : income3 = 6000) (h4 : income4 = 11000) :
  (income1 + income2 + income3 + income4) / 4 = 10000 := by
  sorry

end average_income_family_l533_53374


namespace number_of_trees_in_yard_l533_53366

theorem number_of_trees_in_yard :
  ∀ (yard_length tree_distance : ℕ), yard_length = 360 ∧ tree_distance = 12 → 
  (yard_length / tree_distance + 1 = 31) :=
by
  intros yard_length tree_distance h
  have h1 : yard_length = 360 := h.1
  have h2 : tree_distance = 12 := h.2
  sorry

end number_of_trees_in_yard_l533_53366


namespace value_of_a_plus_b_l533_53354

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem value_of_a_plus_b (a b : ℝ) (h1 : 3 * a + b = 4) (h2 : a + b + 1 = 3) : a + b = 2 :=
by
  sorry

end value_of_a_plus_b_l533_53354


namespace sequence_divisible_by_three_l533_53372

-- Define the conditions
variable (k : ℕ) (h_pos_k : k > 0)
variable (a : ℕ → ℤ)
variable (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n)

-- Define the proof goal
theorem sequence_divisible_by_three (k : ℕ) (h_pos_k : k > 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n) : (k - 2) % 3 = 0 :=
by
  sorry

end sequence_divisible_by_three_l533_53372


namespace ratio_of_ages_l533_53371

-- Define the conditions and the main proof goal
theorem ratio_of_ages (R J : ℕ) (Tim_age : ℕ) (h1 : Tim_age = 5) (h2 : J = R + 2) (h3 : J = Tim_age + 12) :
  R / Tim_age = 3 := 
by
  sorry

end ratio_of_ages_l533_53371


namespace jason_advertising_cost_l533_53312

def magazine_length : ℕ := 9
def magazine_width : ℕ := 12
def cost_per_square_inch : ℕ := 8
def half (x : ℕ) := x / 2
def area (L W : ℕ) := L * W
def total_cost (a c : ℕ) := a * c

theorem jason_advertising_cost :
  total_cost (half (area magazine_length magazine_width)) cost_per_square_inch = 432 := by
  sorry

end jason_advertising_cost_l533_53312


namespace dice_arithmetic_progression_l533_53335

theorem dice_arithmetic_progression :
  let valid_combinations := [
     (1, 1, 1), (1, 3, 2), (1, 5, 3), 
     (2, 4, 3), (2, 6, 4), (3, 3, 3),
     (3, 5, 4), (4, 6, 5), (5, 5, 5)
  ]
  (valid_combinations.length : ℚ) / (6^3 : ℚ) = 1 / 24 :=
  sorry

end dice_arithmetic_progression_l533_53335


namespace melanie_gave_mother_l533_53346

theorem melanie_gave_mother {initial_dimes dad_dimes final_dimes dimes_given : ℕ}
  (h₁ : initial_dimes = 7)
  (h₂ : dad_dimes = 8)
  (h₃ : final_dimes = 11)
  (h₄ : initial_dimes + dad_dimes - dimes_given = final_dimes) :
  dimes_given = 4 :=
by 
  sorry

end melanie_gave_mother_l533_53346


namespace john_vegetables_used_l533_53340

noncomputable def pounds_of_beef_bought : ℕ := 4
noncomputable def pounds_of_beef_used : ℕ := pounds_of_beef_bought - 1
noncomputable def pounds_of_vegetables_used : ℕ := 2 * pounds_of_beef_used

theorem john_vegetables_used : pounds_of_vegetables_used = 6 :=
by
  -- the proof can be provided here later
  sorry

end john_vegetables_used_l533_53340


namespace four_digit_number_difference_l533_53308

theorem four_digit_number_difference
    (digits : List ℕ) (h_digits : digits = [2, 0, 1, 3, 1, 2, 2, 1, 0, 8, 4, 0])
    (max_val : ℕ) (h_max_val : max_val = 3840)
    (min_val : ℕ) (h_min_val : min_val = 1040) :
    max_val - min_val = 2800 :=
by
    sorry

end four_digit_number_difference_l533_53308


namespace average_speed_of_car_l533_53377

theorem average_speed_of_car : 
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  total_distance / total_time = 55 := 
by
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  show total_distance / total_time = 55
  sorry

end average_speed_of_car_l533_53377


namespace seven_fifths_of_fraction_l533_53343

theorem seven_fifths_of_fraction :
  (7 / 5) * (-18 / 4) = -63 / 10 :=
by
  sorry

end seven_fifths_of_fraction_l533_53343


namespace age_ratio_l533_53384

theorem age_ratio (R D : ℕ) (hR : R + 4 = 32) (hD : D = 21) : R / D = 4 / 3 := 
by sorry

end age_ratio_l533_53384


namespace total_employees_with_advanced_degrees_l533_53369

theorem total_employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (num_females : ℕ) 
  (num_males_college_only : ℕ) 
  (num_females_advanced_degrees : ℕ)
  (h1 : total_employees = 180)
  (h2 : num_females = 110)
  (h3 : num_males_college_only = 35)
  (h4 : num_females_advanced_degrees = 55) :
  ∃ num_employees_advanced_degrees : ℕ, num_employees_advanced_degrees = 90 :=
by
  have num_males := total_employees - num_females
  have num_males_advanced_degrees := num_males - num_males_college_only
  have num_employees_advanced_degrees := num_males_advanced_degrees + num_females_advanced_degrees
  use num_employees_advanced_degrees
  sorry

end total_employees_with_advanced_degrees_l533_53369


namespace work_rate_solution_l533_53331

theorem work_rate_solution (y : ℕ) (hy : y > 0) : 
  ∃ z : ℕ, z = (y^2 + 3 * y) / (2 * y + 3) :=
by
  sorry

end work_rate_solution_l533_53331


namespace problem_statement_l533_53306

theorem problem_statement (x y : ℝ) (hx : x - y = 3) (hxy : x = 4 ∧ y = 1) : 2 * (x - y) = 6 * y :=
by
  rcases hxy with ⟨hx', hy'⟩
  rw [hx', hy']
  sorry

end problem_statement_l533_53306


namespace find_xyz_values_l533_53352

theorem find_xyz_values (x y z : ℝ) (h₁ : x + y + z = Real.pi) (h₂ : x ≥ 0) (h₃ : y ≥ 0) (h₄ : z ≥ 0) :
    (x = Real.pi ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = Real.pi ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = Real.pi) ∨
    (x = Real.pi / 6 ∧ y = Real.pi / 3 ∧ z = Real.pi / 2) :=
sorry

end find_xyz_values_l533_53352


namespace dan_initial_amount_l533_53305

variables (initial_amount spent_amount remaining_amount : ℝ)

theorem dan_initial_amount (h1 : spent_amount = 1) (h2 : remaining_amount = 2) : initial_amount = spent_amount + remaining_amount := by
  sorry

end dan_initial_amount_l533_53305


namespace slope_of_parallel_line_l533_53387

theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) : 
  (5 * x - 3 * y = 12) → m = 5 / 3 → (∃ b : ℝ, y = (5 / 3) * x + b) :=
by
  intro h_eqn h_slope
  use -4 / 3
  sorry

end slope_of_parallel_line_l533_53387


namespace percentage_w_less_x_l533_53320

theorem percentage_w_less_x 
    (z : ℝ) 
    (y : ℝ) 
    (x : ℝ) 
    (w : ℝ) 
    (hy : y = 1.20 * z)
    (hx : x = 1.20 * y)
    (hw : w = 1.152 * z) 
    : (x - w) / x * 100 = 20 :=
by
  sorry

end percentage_w_less_x_l533_53320


namespace no_nat_solutions_m2_eq_n2_plus_2014_l533_53329

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l533_53329


namespace radiator_initial_fluid_l533_53323

theorem radiator_initial_fluid (x : ℝ)
  (h1 : (0.10 * x - 0.10 * 2.2857 + 0.80 * 2.2857) = 0.50 * x) :
  x = 4 :=
sorry

end radiator_initial_fluid_l533_53323


namespace part1_part2_l533_53364

-- Define the universal set R
def R := ℝ

-- Define set A
def A (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0

-- Define set B parameterized by a
def B (x a : ℝ) : Prop := (x - (a + 5)) / (x - a) > 0

-- Prove (1): A ∩ B when a = -2
theorem part1 : { x : ℝ | A x } ∩ { x : ℝ | B x (-2) } = { x : ℝ | 3 < x ∧ x ≤ 4 } :=
by
  sorry

-- Prove (2): The range of a such that A ⊆ B
theorem part2 : { a : ℝ | ∀ x, A x → B x a } = { a : ℝ | a < -6 ∨ a > 4 } :=
by
  sorry

end part1_part2_l533_53364


namespace square_nonneg_l533_53391

theorem square_nonneg (x h k : ℝ) (h_eq: (x + h)^2 = k) : k ≥ 0 := 
by 
  sorry

end square_nonneg_l533_53391


namespace tan_sum_l533_53318

theorem tan_sum (θ : ℝ) (h : Real.sin (2 * θ) = 2 / 3) : Real.tan θ + 1 / Real.tan θ = 3 := sorry

end tan_sum_l533_53318


namespace pilot_weeks_l533_53368

-- Given conditions
def milesTuesday : ℕ := 1134
def milesThursday : ℕ := 1475
def totalMiles : ℕ := 7827

-- Calculate total miles flown in one week
def milesPerWeek : ℕ := milesTuesday + milesThursday

-- Define the proof problem statement
theorem pilot_weeks (w : ℕ) (h : w * milesPerWeek = totalMiles) : w = 3 :=
by
  -- Here we would provide the proof, but we leave it with a placeholder
  sorry

end pilot_weeks_l533_53368


namespace spencer_total_jumps_l533_53341

noncomputable def jumps_per_minute : ℕ := 4
noncomputable def minutes_per_session : ℕ := 10
noncomputable def sessions_per_day : ℕ := 2
noncomputable def days : ℕ := 5

theorem spencer_total_jumps : 
  (jumps_per_minute * minutes_per_session) * (sessions_per_day * days) = 400 :=
by
  sorry

end spencer_total_jumps_l533_53341


namespace sum_of_arithmetic_sequence_l533_53349

def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith : is_arithmetic_sequence a d)
  (h_nonzero : d ≠ 0)
  (h_sum_f : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by 
  sorry

end sum_of_arithmetic_sequence_l533_53349


namespace polynomial_simplification_l533_53327

theorem polynomial_simplification (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := 
by 
  sorry

end polynomial_simplification_l533_53327


namespace train_passes_bridge_in_20_seconds_l533_53315

def train_length : ℕ := 360
def bridge_length : ℕ := 140
def train_speed_kmh : ℕ := 90

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance : ℕ := train_length + bridge_length
noncomputable def travel_time : ℝ := total_distance / train_speed_ms

theorem train_passes_bridge_in_20_seconds :
  travel_time = 20 := by
  sorry

end train_passes_bridge_in_20_seconds_l533_53315


namespace probability_compare_l533_53353

-- Conditions
def v : ℝ := 0.1
def n : ℕ := 998

-- Binomial distribution formula
noncomputable def binom_prob (n k : ℕ) (v : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * (v ^ k) * ((1 - v) ^ (n - k))

-- Theorem to prove
theorem probability_compare :
  binom_prob n 99 v > binom_prob n 100 v :=
by
  sorry

end probability_compare_l533_53353


namespace Ma_Xiaohu_speed_l533_53385

theorem Ma_Xiaohu_speed
  (distance_home_school : ℕ := 1800)
  (distance_to_school : ℕ := 1600)
  (father_speed_factor : ℕ := 2)
  (time_difference : ℕ := 10)
  (x : ℕ)
  (hx : distance_home_school - distance_to_school = 200)
  (hspeed : father_speed_factor * x = 2 * x)
  :
  (distance_to_school / x) - (distance_to_school / (2 * x)) = time_difference ↔ x = 80 :=
by
  sorry

end Ma_Xiaohu_speed_l533_53385


namespace quadratic_inequality_solution_l533_53347

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 - 8 * x + c > 0) ↔ (0 < c ∧ c < 16) := 
sorry

end quadratic_inequality_solution_l533_53347


namespace garden_width_l533_53381

theorem garden_width (w l : ℝ) (h_length : l = 3 * w) (h_area : l * w = 675) : w = 15 :=
by
  sorry

end garden_width_l533_53381


namespace fourth_number_in_sequence_l533_53326

noncomputable def fifth_number_in_sequence : ℕ := 78
noncomputable def increment : ℕ := 11
noncomputable def final_number_in_sequence : ℕ := 89

theorem fourth_number_in_sequence : (fifth_number_in_sequence - increment) = 67 := by
  sorry

end fourth_number_in_sequence_l533_53326


namespace geometric_sequence_third_term_l533_53309

theorem geometric_sequence_third_term (a₁ a₄ : ℕ) (r : ℕ) (h₁ : a₁ = 4) (h₂ : a₄ = 256) (h₃ : a₄ = a₁ * r^3) : a₁ * r^2 = 64 := 
by
  sorry

end geometric_sequence_third_term_l533_53309


namespace polygon_diagonals_150_sides_l533_53348

-- Define the function to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The theorem to state what we want to prove
theorem polygon_diagonals_150_sides : num_diagonals 150 = 11025 :=
by sorry

end polygon_diagonals_150_sides_l533_53348


namespace divide_shape_into_equal_parts_l533_53333

-- Definitions and conditions
structure Shape where
  has_vertical_symmetry : Bool
  -- Other properties of the shape can be added as necessary

def vertical_line_divides_equally (s : Shape) : Prop :=
  s.has_vertical_symmetry

-- Theorem statement
theorem divide_shape_into_equal_parts (s : Shape) (h : s.has_vertical_symmetry = true) :
  vertical_line_divides_equally s :=
by
  -- Begin proof
  sorry

end divide_shape_into_equal_parts_l533_53333


namespace stephanie_running_time_l533_53373

theorem stephanie_running_time
  (Speed : ℝ) (Distance : ℝ) (Time : ℝ)
  (h1 : Speed = 5)
  (h2 : Distance = 15)
  (h3 : Time = Distance / Speed) :
  Time = 3 :=
sorry

end stephanie_running_time_l533_53373


namespace coin_flip_sequences_count_l533_53319

noncomputable def num_sequences_with_given_occurrences : ℕ :=
  sorry

theorem coin_flip_sequences_count : num_sequences_with_given_occurrences = 560 :=
  sorry

end coin_flip_sequences_count_l533_53319


namespace monotone_increasing_range_of_a_l533_53342

noncomputable def f (a x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_increasing_range_of_a :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Icc (-1 / 3 : ℝ) (1 / 3 : ℝ)) :=
sorry

end monotone_increasing_range_of_a_l533_53342


namespace white_ball_probability_l533_53362

theorem white_ball_probability :
  ∀ (n : ℕ), (2/(n+2) = 2/5) → (n = 3) → (n/(n+2) = 3/5) :=
by
  sorry

end white_ball_probability_l533_53362


namespace volume_tetrahedron_l533_53363

def A1 := 4^2
def A2 := 3^2
def h := 1

theorem volume_tetrahedron:
  (h / 3 * (A1 + A2 + Real.sqrt (A1 * A2))) = 37 / 3 := by
  sorry

end volume_tetrahedron_l533_53363


namespace four_x_plus_y_greater_than_four_z_l533_53365

theorem four_x_plus_y_greater_than_four_z
  (x y z : ℝ)
  (h1 : y > 2 * z)
  (h2 : 2 * z > 4 * x)
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) > 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z)
  : 4 * x + y > 4 * z := 
by
  sorry

end four_x_plus_y_greater_than_four_z_l533_53365


namespace infinite_solutions_a_value_l533_53300

theorem infinite_solutions_a_value (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) ↔ a = 5 := 
by 
  sorry

end infinite_solutions_a_value_l533_53300


namespace cos_double_angle_l533_53338

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) : Real.cos (2 * θ) = -1/3 := by
  sorry

end cos_double_angle_l533_53338


namespace part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l533_53325

def set_A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def set_B (m : ℝ) : Set ℝ := {x | x < m}

-- Problem 1
theorem part1_A_complement_B_intersection_eq (m : ℝ) (h : m = 3) :
  set_A ∩ {x | x >= 3} = {x | 3 <= x ∧ x < 4} :=
sorry

-- Problem 2
theorem part2_m_le_neg2 (m : ℝ) (h : set_A ∩ set_B m = ∅) :
  m <= -2 :=
sorry

-- Problem 3
theorem part3_m_ge_4 (m : ℝ) (h : set_A ∩ set_B m = set_A) :
  m >= 4 :=
sorry

end part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l533_53325


namespace vector_satisfies_condition_l533_53336

def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 5 + 2 * t)
def line_m (s : ℝ) : ℝ × ℝ := (1 + 2 * s, 3 + 2 * s)

variable (A B P : ℝ × ℝ)

def vector_BA (B A : ℝ × ℝ) : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vector_v : ℝ × ℝ := (1, -1)

theorem vector_satisfies_condition : 
  2 * vector_v.1 - vector_v.2 = 3 := by
  sorry

end vector_satisfies_condition_l533_53336


namespace smallest_value_is_nine_l533_53339

noncomputable def smallest_possible_value (a b c d : ℝ) : ℝ :=
  (⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ : ℝ)

theorem smallest_value_is_nine {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_possible_value a b c d = 9 :=
sorry

end smallest_value_is_nine_l533_53339


namespace linear_eq_solution_l533_53316

theorem linear_eq_solution (m : ℤ) (x : ℝ) (h1 : |m| = 1) (h2 : 1 - m ≠ 0) : x = -1/2 :=
by
  sorry

end linear_eq_solution_l533_53316


namespace max_x2_plus_2xy_plus_3y2_l533_53302

theorem max_x2_plus_2xy_plus_3y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 18 + 9 * Real.sqrt 3 :=
sorry

end max_x2_plus_2xy_plus_3y2_l533_53302


namespace tulips_sum_l533_53397

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum_l533_53397


namespace cos_sin_identity_l533_53359

theorem cos_sin_identity (x : ℝ) (h : Real.cos (x - Real.pi / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * Real.pi / 3) + Real.sin (Real.pi / 3 - x) ^ 2 = 5 / 3 :=
sorry

end cos_sin_identity_l533_53359


namespace num_real_a_satisfy_union_l533_53311

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end num_real_a_satisfy_union_l533_53311


namespace speed_of_stream_l533_53303

-- Define the problem conditions
def downstream_distance := 100 -- distance in km
def downstream_time := 8 -- time in hours
def upstream_distance := 75 -- distance in km
def upstream_time := 15 -- time in hours

-- Define the constants
def total_distance (B S : ℝ) := downstream_distance = (B + S) * downstream_time
def total_time (B S : ℝ) := upstream_distance = (B - S) * upstream_time

-- Stating the main theorem to be proved
theorem speed_of_stream (B S : ℝ) (h1 : total_distance B S) (h2 : total_time B S) : S = 3.75 := by
  sorry

end speed_of_stream_l533_53303


namespace regression_line_is_y_eq_x_plus_1_l533_53395

def Point : Type := ℝ × ℝ

def A : Point := (1, 2)
def B : Point := (2, 3)
def C : Point := (3, 4)
def D : Point := (4, 5)

def points : List Point := [A, B, C, D]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (lst.foldr (fun x acc => x + acc) 0) / lst.length

noncomputable def regression_line (pts : List Point) : ℝ → ℝ :=
  let xs := pts.map Prod.fst
  let ys := pts.map Prod.snd
  fun x : ℝ => x + 1

theorem regression_line_is_y_eq_x_plus_1 :
  regression_line points = fun x => x + 1 := sorry

end regression_line_is_y_eq_x_plus_1_l533_53395


namespace total_bill_is_95_l533_53376

noncomputable def total_bill := 28 + 8 + 10 + 6 + 14 + 11 + 12 + 6

theorem total_bill_is_95 : total_bill = 95 := by
  sorry

end total_bill_is_95_l533_53376


namespace A_rotated_l533_53370

-- Define initial coordinates of point A
def A_initial : ℝ × ℝ := (1, 2)

-- Define the transformation for a 180-degree clockwise rotation around the origin
def rotate_180_deg (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- The Lean statement to prove the coordinates after the rotation
theorem A_rotated : rotate_180_deg A_initial = (-1, -2) :=
by
  sorry

end A_rotated_l533_53370


namespace conic_not_parabola_l533_53389

def conic_equation (m x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

theorem conic_not_parabola (m : ℝ) :
  ¬ (∃ (x y : ℝ), conic_equation m x y ∧ ∃ (a b c d e f : ℝ), m * x^2 + (m + 1) * y^2 = a * x^2 + b * xy + c * y^2 + d * x + e * y + f ∧ (a = 0 ∨ c = 0) ∧ (b ≠ 0 ∨ a ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0)) :=  
sorry

end conic_not_parabola_l533_53389


namespace sequence_contains_30_l533_53330

theorem sequence_contains_30 :
  ∃ n : ℕ, n * (n + 1) = 30 :=
sorry

end sequence_contains_30_l533_53330


namespace purely_imaginary_z_eq_a2_iff_a2_l533_53393

theorem purely_imaginary_z_eq_a2_iff_a2 (a : Real) : 
(∃ (b : Real), a^2 - a - 2 = 0 ∧ a + 1 ≠ 0) → a = 2 :=
by
  sorry

end purely_imaginary_z_eq_a2_iff_a2_l533_53393


namespace gcd_gx_x_is_210_l533_53328

-- Define the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, y = k * x

-- The main proof problem
theorem gcd_gx_x_is_210 (x : ℕ) (hx : is_multiple_of 17280 x) :
  Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (4 * x + 5)) x = 210 :=
by
  sorry

end gcd_gx_x_is_210_l533_53328


namespace find_number_l533_53367

theorem find_number (x : ℝ) : ((x - 50) / 4) * 3 + 28 = 73 → x = 110 := 
  by 
  sorry

end find_number_l533_53367


namespace trigo_identity_l533_53310

variable (α : ℝ)

theorem trigo_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (Real.pi / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end trigo_identity_l533_53310


namespace age_ratio_l533_53321

theorem age_ratio 
    (a m s : ℕ) 
    (h1 : m = 60) 
    (h2 : m = 3 * a) 
    (h3 : s = 40) : 
    (m + a) / s = 2 :=
by
    sorry

end age_ratio_l533_53321


namespace brian_spent_on_kiwis_l533_53357

theorem brian_spent_on_kiwis :
  ∀ (cost_per_dozen_apples : ℝ)
    (cost_for_24_apples : ℝ)
    (initial_money : ℝ)
    (subway_fare_one_way : ℝ)
    (total_remaining : ℝ)
    (kiwis_spent : ℝ)
    (bananas_spent : ℝ),
  cost_per_dozen_apples = 14 →
  cost_for_24_apples = 2 * cost_per_dozen_apples →
  initial_money = 50 →
  subway_fare_one_way = 3.5 →
  total_remaining = initial_money - 2 * subway_fare_one_way - cost_for_24_apples →
  total_remaining = 15 →
  bananas_spent = kiwis_spent / 2 →
  kiwis_spent + bananas_spent = total_remaining →
  kiwis_spent = 10 :=
by
  -- Sorry means we are skipping the proof
  sorry

end brian_spent_on_kiwis_l533_53357


namespace quadratic_one_positive_root_l533_53322

theorem quadratic_one_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y ∈ {t | t^2 - a * t + a - 2 = 0} → y = x)) → a ≤ 2 :=
by
  sorry

end quadratic_one_positive_root_l533_53322


namespace find_a_l533_53337

noncomputable def f (a x : ℝ) : ℝ := 2^x / (2^x + a * x)

variables (a p q : ℝ)

theorem find_a
  (h1 : f a p = 6 / 5)
  (h2 : f a q = -1 / 5)
  (h3 : 2^(p + q) = 16 * p * q)
  (h4 : a > 0) :
  a = 4 :=
  sorry

end find_a_l533_53337


namespace slope_of_line_l533_53332

theorem slope_of_line (θ : ℝ) (h_cosθ : (Real.cos θ) = 4/5) : (Real.sin θ) / (Real.cos θ) = 3/4 :=
by
  sorry

end slope_of_line_l533_53332


namespace skating_rink_visitors_by_noon_l533_53304

-- Defining the initial conditions
def initial_visitors : ℕ := 264
def visitors_left : ℕ := 134
def visitors_arrived : ℕ := 150

-- Theorem to prove the number of people at the skating rink by noon
theorem skating_rink_visitors_by_noon : initial_visitors - visitors_left + visitors_arrived = 280 := 
by 
  sorry

end skating_rink_visitors_by_noon_l533_53304


namespace greatest_positive_integer_x_l533_53378

theorem greatest_positive_integer_x : ∃ (x : ℕ), (x > 0) ∧ (∀ y : ℕ, y > 0 → (y^3 < 20 * y → y ≤ 4)) ∧ (x^3 < 20 * x) ∧ ∀ z : ℕ, (z > 0) → (z^3 < 20 * z → x ≥ z)  :=
sorry

end greatest_positive_integer_x_l533_53378


namespace product_of_four_consecutive_is_perfect_square_l533_53317

theorem product_of_four_consecutive_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
by
  sorry

end product_of_four_consecutive_is_perfect_square_l533_53317


namespace max_product_decomposition_l533_53345

theorem max_product_decomposition : ∃ x y : ℝ, x + y = 100 ∧ x * y = 50 * 50 := by
  sorry

end max_product_decomposition_l533_53345


namespace chapters_per_day_l533_53314

theorem chapters_per_day (chapters : ℕ) (total_days : ℕ) : ℝ :=
  let chapters := 2
  let total_days := 664
  chapters / total_days

example : chapters_per_day 2 664 = 2 / 664 := by sorry

end chapters_per_day_l533_53314


namespace domain_of_f_l533_53361

noncomputable def f (x : ℝ) : ℝ := (4 * x - 2) / (Real.sqrt (x - 7))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = f x } = {x : ℝ | x > 7} :=
by
  sorry

end domain_of_f_l533_53361


namespace perimeter_of_equilateral_triangle_l533_53399

theorem perimeter_of_equilateral_triangle (s : ℝ) 
  (h1 : (s ^ 2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_equilateral_triangle_l533_53399


namespace min_children_l533_53390

theorem min_children (x : ℕ) : 
  (4 * x + 28 - 5 * (x - 1) < 5) ∧ (4 * x + 28 - 5 * (x - 1) ≥ 2) → (x = 29) :=
by
  sorry

end min_children_l533_53390


namespace percentage_error_l533_53382

theorem percentage_error (e : ℝ) : (1 + e / 100)^2 = 1.1025 → e = 5.125 := 
by sorry

end percentage_error_l533_53382


namespace tan_periodic_mod_l533_53388

theorem tan_periodic_mod (m : ℤ) (h1 : -180 < m) (h2 : m < 180) : 
  (m : ℤ) = 10 := by
  sorry

end tan_periodic_mod_l533_53388


namespace total_money_shared_l533_53301

theorem total_money_shared (ratio_jonah ratio_kira ratio_liam kira_share : ℕ)
  (h_ratio : ratio_jonah = 2) (h_ratio2 : ratio_kira = 3) (h_ratio3 : ratio_liam = 8)
  (h_kira : kira_share = 45) :
  (ratio_jonah * (kira_share / ratio_kira) + kira_share + ratio_liam * (kira_share / ratio_kira)) = 195 := 
by
  sorry

end total_money_shared_l533_53301


namespace days_per_book_l533_53351

theorem days_per_book (total_books : ℕ) (total_days : ℕ)
  (h1 : total_books = 41)
  (h2 : total_days = 492) :
  total_days / total_books = 12 :=
by
  -- proof goes here
  sorry

end days_per_book_l533_53351


namespace solve_for_x_and_calculate_l533_53398

theorem solve_for_x_and_calculate (x y : ℚ) 
  (h1 : 102 * x - 5 * y = 25) 
  (h2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 :=
by 
  -- These proof steps would solve the problem and validate the theorem
  sorry

end solve_for_x_and_calculate_l533_53398
