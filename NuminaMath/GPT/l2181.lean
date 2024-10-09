import Mathlib

namespace length_of_each_glass_pane_l2181_218177

theorem length_of_each_glass_pane (panes : ℕ) (width : ℕ) (total_area : ℕ) 
    (H_panes : panes = 8) (H_width : width = 8) (H_total_area : total_area = 768) : 
    ∃ length : ℕ, length = 12 := by
  sorry

end length_of_each_glass_pane_l2181_218177


namespace find_number_l2181_218178

theorem find_number (x : ℝ) : (x^2 + 4 = 5 * x) → (x = 4 ∨ x = 1) :=
by
  sorry

end find_number_l2181_218178


namespace jane_stopped_babysitting_l2181_218139

noncomputable def stopped_babysitting_years_ago := 12

-- Definitions for the problem conditions
def jane_age_started_babysitting := 20
def jane_current_age := 32
def oldest_child_current_age := 22

-- Final statement to prove the equivalence
theorem jane_stopped_babysitting : 
    ∃ (x : ℕ), 
    (jane_current_age - x = stopped_babysitting_years_ago) ∧
    (oldest_child_current_age - x ≤ 1/2 * (jane_current_age - x)) := 
sorry

end jane_stopped_babysitting_l2181_218139


namespace polynomial_division_l2181_218120

open Polynomial

theorem polynomial_division (a b : ℤ) (h : a^2 ≥ 4*b) :
  ∀ n : ℕ, ∃ (k l : ℤ), (x^2 + (C a) * x + (C b)) ∣ (x^2) * (x^2) ^ n + (C a) * x ^ n + (C b) ↔ 
    ((a = -2 ∧ b = 1) ∨ (a = 2 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
sorry

end polynomial_division_l2181_218120


namespace find_P2_l2181_218193

def P1 : ℕ := 64
def total_pigs : ℕ := 86

theorem find_P2 : ∃ (P2 : ℕ), P1 + P2 = total_pigs ∧ P2 = 22 :=
by 
  sorry

end find_P2_l2181_218193


namespace father_l2181_218147

-- Let s be the circumference of the circular rink.
-- Let x be the son's speed.
-- Let k be the factor by which the father's speed is greater than the son's speed.

-- Define a theorem to state that k = 3/2.
theorem father's_speed_is_3_over_2_times_son's_speed
  (s x : ℝ) (k : ℝ) (h : s / (k * x - x) = (s / (k * x + x)) * 5) :
  k = 3 / 2 :=
by {
  sorry
}

end father_l2181_218147


namespace ratio_c_d_l2181_218116

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x + 5 * y = c) (h2 : 8 * y - 10 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = 1 / 2 :=
by
  sorry

end ratio_c_d_l2181_218116


namespace print_time_including_warmup_l2181_218148

def warmUpTime : ℕ := 2
def pagesPerMinute : ℕ := 15
def totalPages : ℕ := 225

theorem print_time_including_warmup :
  (totalPages / pagesPerMinute) + warmUpTime = 17 := by
  sorry

end print_time_including_warmup_l2181_218148


namespace solution_set_empty_for_k_l2181_218153

theorem solution_set_empty_for_k (k : ℝ) :
  (∀ x : ℝ, ¬ (kx^2 - 2 * |x - 1| + 3 * k < 0)) ↔ (1 ≤ k) :=
by
  sorry

end solution_set_empty_for_k_l2181_218153


namespace wax_initial_amount_l2181_218128

def needed : ℕ := 17
def total : ℕ := 574
def initial : ℕ := total - needed

theorem wax_initial_amount :
  initial = 557 :=
by
  sorry

end wax_initial_amount_l2181_218128


namespace garden_area_proof_l2181_218163

def length_rect : ℕ := 20
def width_rect : ℕ := 18
def area_rect : ℕ := length_rect * width_rect

def side_square1 : ℕ := 4
def area_square1 : ℕ := side_square1 * side_square1

def side_square2 : ℕ := 5
def area_square2 : ℕ := side_square2 * side_square2

def area_remaining : ℕ := area_rect - area_square1 - area_square2

theorem garden_area_proof : area_remaining = 319 := by
  sorry

end garden_area_proof_l2181_218163


namespace leo_weight_l2181_218156

theorem leo_weight 
  (L K E : ℝ)
  (h1 : L + 10 = 1.5 * K)
  (h2 : L + 10 = 0.75 * E)
  (h3 : L + K + E = 210) :
  L = 63.33 := 
sorry

end leo_weight_l2181_218156


namespace solution_set_of_inequality_l2181_218121

theorem solution_set_of_inequality :
  {x : ℝ | 4*x^2 - 9*x > 5} = {x : ℝ | x < -1/4} ∪ {x : ℝ | x > 5} :=
by
  sorry

end solution_set_of_inequality_l2181_218121


namespace no_common_solution_l2181_218168

theorem no_common_solution 
  (x : ℝ) 
  (h1 : 8 * x^2 + 6 * x = 5) 
  (h2 : 3 * x + 2 = 0) : 
  False := 
by
  sorry

end no_common_solution_l2181_218168


namespace sqrt_180_simplified_l2181_218142

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l2181_218142


namespace maximum_temperature_difference_l2181_218129

theorem maximum_temperature_difference
  (highest_temp : ℝ) (lowest_temp : ℝ)
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 :=
by sorry

end maximum_temperature_difference_l2181_218129


namespace unique_element_a_values_set_l2181_218104

open Set

theorem unique_element_a_values_set :
  {a : ℝ | ∃! x : ℝ, a * x^2 + 2 * x - a = 0} = {0} :=
by
  sorry

end unique_element_a_values_set_l2181_218104


namespace total_value_of_button_collection_l2181_218122

theorem total_value_of_button_collection:
  (∀ (n : ℕ) (v : ℕ), n = 2 → v = 8 → has_same_value → total_value = 10 * (v / n)) →
  has_same_value :=
  sorry

end total_value_of_button_collection_l2181_218122


namespace solve_z_solutions_l2181_218141

noncomputable def z_solutions (z : ℂ) : Prop :=
  z ^ 6 = -16

theorem solve_z_solutions :
  {z : ℂ | z_solutions z} = {2 * Complex.I, -2 * Complex.I} :=
by {
  sorry
}

end solve_z_solutions_l2181_218141


namespace age_of_older_teenager_l2181_218144

theorem age_of_older_teenager
  (a b : ℕ) 
  (h1 : a^2 - b^2 = 4 * (a + b)) 
  (h2 : a + b = 8 * (a - b)) 
  (h3 : a > b) : 
  a = 18 :=
sorry

end age_of_older_teenager_l2181_218144


namespace distance_between_city_A_and_city_B_l2181_218157

noncomputable def eddyTravelTime : ℝ := 3  -- hours
noncomputable def freddyTravelTime : ℝ := 4  -- hours
noncomputable def constantDistance : ℝ := 300  -- km
noncomputable def speedRatio : ℝ := 2  -- Eddy:Freddy

theorem distance_between_city_A_and_city_B (D_B D_C : ℝ) (h1 : D_B = (3 / 2) * D_C) (h2 : D_C = 300) :
  D_B = 450 :=
by
  sorry

end distance_between_city_A_and_city_B_l2181_218157


namespace original_price_before_discounts_l2181_218106

theorem original_price_before_discounts (P : ℝ) 
  (h : 0.75 * (0.75 * P) = 18) : P = 32 :=
by
  sorry

end original_price_before_discounts_l2181_218106


namespace cans_ounces_per_day_l2181_218136

-- Definitions of the conditions
def daily_soda_cans : ℕ := 5
def daily_water_ounces : ℕ := 64
def weekly_fluid_ounces : ℕ := 868

-- Theorem statement proving the number of ounces per can of soda
theorem cans_ounces_per_day (h_soda_daily : daily_soda_cans * 7 = 35)
    (h_weekly_soda : weekly_fluid_ounces - daily_water_ounces * 7 = 420) 
    (h_total_weekly : 35 = ((daily_soda_cans * 7))):
  420 / 35 = 12 := by
  sorry

end cans_ounces_per_day_l2181_218136


namespace special_operation_value_l2181_218135

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end special_operation_value_l2181_218135


namespace initial_marbles_count_l2181_218191

-- Define the conditions
def marbles_given_to_mary : ℕ := 14
def marbles_remaining : ℕ := 50

-- Prove that Dan's initial number of marbles is 64
theorem initial_marbles_count : marbles_given_to_mary + marbles_remaining = 64 := 
by {
  sorry
}

end initial_marbles_count_l2181_218191


namespace find_exponent_l2181_218182

theorem find_exponent (y : ℕ) (h : (1/8) * (2: ℝ)^36 = (2: ℝ)^y) : y = 33 :=
by sorry

end find_exponent_l2181_218182


namespace max_value_condition_l2181_218167

variable {m n : ℝ}

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  m * n > 0 ∧ m + n = -1

-- Statement of the proof problem
theorem max_value_condition (h : conditions m n) : (1/m + 1/n) ≤ 4 :=
sorry

end max_value_condition_l2181_218167


namespace total_amount_paid_is_correct_l2181_218100

-- Definitions for the conditions
def original_price : ℝ := 150
def sale_discount : ℝ := 0.30
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

-- Calculation
def final_amount : ℝ :=
  let discounted_price := original_price * (1 - sale_discount)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price_after_tax := price_after_coupon * (1 + sales_tax)
  final_price_after_tax

-- Statement to prove
theorem total_amount_paid_is_correct : final_amount = 104.50 := by
  sorry

end total_amount_paid_is_correct_l2181_218100


namespace unique_solution_l2181_218192

theorem unique_solution (x : ℝ) (hx : x ≥ 0) : 2021 * x = 2022 * x ^ (2021 / 2022) - 1 → x = 1 :=
by
  intros h
  sorry

end unique_solution_l2181_218192


namespace coordinates_OQ_quadrilateral_area_range_l2181_218196

variables {p : ℝ} (p_pos : 0 < p)
variables {x0 x1 x2 y0 y1 y2 : ℝ} (h_parabola_A : y1^2 = 2*p*x1) (h_parabola_B : y2^2 = 2*p*x2) (h_parabola_M : y0^2 = 2*p*x0)
variables {a : ℝ} (h_focus_x : a = x0 + p) 

variables {FA FM FB : ℝ}
variables (h_arith_seq : ( FM = FA - (FA - FB) / 2 ))

-- Step 1: Prove the coordinates of OQ
theorem coordinates_OQ : (x0 + p, 0) = (a, 0) :=
by
  -- proof will be completed here
  sorry 

variables {x0_val : ℝ} (x0_eq : x0 = 2) {FM_val : ℝ} (FM_eq : FM = 5 / 2)

-- Step 2: Prove the area range of quadrilateral ABB1A1
theorem quadrilateral_area_range : ∀ (p : ℝ), 0 < p →
  ∀ (x0 x1 x2 y1 y2 FM OQ : ℝ), 
    x0 = 2 → FM = 5 / 2 → OQ = 3 → (y1^2 = 2*p*x1) → (y2^2 = 2*p*x2) →
  ( ∃ S : ℝ, 0 < S ∧ S ≤ 10) :=
by
  -- proof will be completed here
  sorry 

end coordinates_OQ_quadrilateral_area_range_l2181_218196


namespace perpendicular_slope_l2181_218159

variable (x y : ℝ)

def line_eq : Prop := 4 * x - 5 * y = 20

theorem perpendicular_slope (x y : ℝ) (h : line_eq x y) : - (1 / (4 / 5)) = -5 / 4 := by
  sorry

end perpendicular_slope_l2181_218159


namespace winning_percentage_l2181_218127

-- Defining the conditions
def election_conditions (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) : Prop :=
  total_candidates = 2 ∧ winner_votes = 864 ∧ win_margin = 288

-- Stating the question: What percentage of votes did the winner candidate receive?
theorem winning_percentage (V : ℕ) (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) :
  election_conditions winner_votes win_margin total_candidates → (winner_votes * 100 / V) = 60 :=
by
  sorry

end winning_percentage_l2181_218127


namespace handshakes_in_octagonal_shape_l2181_218155

-- Definitions
def number_of_students : ℕ := 8

def non_adjacent_handshakes_per_student : ℕ := number_of_students - 1 - 2

def total_handshakes : ℕ := (number_of_students * non_adjacent_handshakes_per_student) / 2

-- Theorem to prove
theorem handshakes_in_octagonal_shape : total_handshakes = 20 := 
by
  -- Provide the proof here.
  sorry

end handshakes_in_octagonal_shape_l2181_218155


namespace enclosed_area_correct_l2181_218194

noncomputable def enclosedArea : ℝ := ∫ x in (1 / Real.exp 1)..Real.exp 1, 1 / x

theorem enclosed_area_correct : enclosedArea = 2 := by
  sorry

end enclosed_area_correct_l2181_218194


namespace total_stamps_l2181_218105

-- Definitions based on the conditions
def AJ := 370
def KJ := AJ / 2
def CJ := 2 * KJ + 5

-- Proof Statement
theorem total_stamps : AJ + KJ + CJ = 930 := by
  sorry

end total_stamps_l2181_218105


namespace triangle_angle_A_triangle_bc_range_l2181_218102

theorem triangle_angle_A (a b c A B C : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (ha : a = b * Real.sin C + c * Real.sin B)
  (hb : b = c * Real.sin A + a * Real.sin C)
  (hc : c = a * Real.sin B + b * Real.sin A)
  (h_eq : (Real.sqrt 3) * a * Real.sin C + a * Real.cos C = c + b)
  (h_angles_sum : A + B + C = π) :
    A = π/3 := -- π/3 radians equals 60 degrees
sorry

theorem triangle_bc_range (a b c : ℝ) (h : a = Real.sqrt 3) :
  Real.sqrt 3 < b + c ∧ b + c ≤ 2 * Real.sqrt 3 := 
sorry

end triangle_angle_A_triangle_bc_range_l2181_218102


namespace solve_monomial_equation_l2181_218150

theorem solve_monomial_equation (x : ℝ) (m n : ℝ) (a b : ℝ) 
  (h1 : m = 2) (h2 : n = 3) 
  (h3 : (1/3) * a^m * b^3 + (-2) * a^2 * b^n = (1/3) * a^2 * b^3 + (-2) * a^2 * b^3) :
  (x - 7) / n - (1 + x) / m = 1 → x = -23 := 
by
  sorry

end solve_monomial_equation_l2181_218150


namespace years_between_2000_and_3000_with_property_l2181_218173

theorem years_between_2000_and_3000_with_property :
  ∃ n : ℕ, n = 143 ∧
  ∀ Y, 2000 ≤ Y ∧ Y ≤ 3000 → ∃ p q : ℕ, p + q = Y ∧ 2 * p = 5 * q →
  (2 * Y) % 7 = 0 :=
sorry

end years_between_2000_and_3000_with_property_l2181_218173


namespace find_added_number_l2181_218123

variable (x : ℝ) -- We define the variable x as a real number
-- We define the given conditions

def added_number (y : ℝ) : Prop :=
  (2 * (62.5 + y) / 5) - 5 = 22

theorem find_added_number : added_number x → x = 5 := by
  sorry

end find_added_number_l2181_218123


namespace blue_eyed_blonds_greater_than_population_proportion_l2181_218174

variables {G_B Γ B N : ℝ}

theorem blue_eyed_blonds_greater_than_population_proportion (h : G_B / Γ > B / N) : G_B / B > Γ / N :=
sorry

end blue_eyed_blonds_greater_than_population_proportion_l2181_218174


namespace walk_to_cafe_and_back_time_l2181_218183

theorem walk_to_cafe_and_back_time 
  (t_p : ℝ) (d_p : ℝ) (half_dp : ℝ) (pace : ℝ)
  (h1 : t_p = 30) 
  (h2 : d_p = 3) 
  (h3 : half_dp = d_p / 2) 
  (h4 : pace = t_p / d_p) :
  2 * half_dp * pace = 30 :=
by 
  sorry

end walk_to_cafe_and_back_time_l2181_218183


namespace largest_integer_a_l2181_218176

theorem largest_integer_a (x a : ℤ) :
  ∃ x : ℤ, (x - a) * (x - 7) + 3 = 0 → a ≤ 11 :=
sorry

end largest_integer_a_l2181_218176


namespace cows_problem_l2181_218110

theorem cows_problem :
  ∃ (M X : ℕ), 
  (5 * M = X + 30) ∧ 
  (5 * M + X = 570) ∧ 
  M = 60 :=
by
  sorry

end cows_problem_l2181_218110


namespace baker_work_alone_time_l2181_218146

theorem baker_work_alone_time 
  (rate_baker_alone : ℕ) 
  (rate_baker_with_helper : ℕ) 
  (total_time : ℕ) 
  (total_flour : ℕ)
  (time_with_helper : ℕ)
  (flour_used_baker_alone_time : ℕ)
  (flour_used_with_helper_time : ℕ)
  (total_flour_used : ℕ) 
  (h1 : rate_baker_alone = total_flour / 6) 
  (h2 : rate_baker_with_helper = total_flour / 2) 
  (h3 : total_time = 150)
  (h4 : flour_used_baker_alone_time = total_flour * flour_used_baker_alone_time / 6)
  (h5 : flour_used_with_helper_time = total_flour * (total_time - flour_used_baker_alone_time) / 2)
  (h6 : total_flour_used = total_flour) :
  flour_used_baker_alone_time = 45 :=
by
  sorry

end baker_work_alone_time_l2181_218146


namespace angle_x_l2181_218164

-- Conditions
variable (ABC BAC CDE DCE : ℝ)
variable (h1 : ABC = 70)
variable (h2 : BAC = 50)
variable (h3 : CDE = 90)
variable (h4 : ∃ BCA : ℝ, DCE = BCA ∧ ABC + BAC + BCA = 180)

-- The statement to prove
theorem angle_x (x : ℝ) (h : ∃ BCA : ℝ, (ABC = 70) ∧ (BAC = 50) ∧ (CDE = 90) ∧ (DCE = BCA ∧ ABC + BAC + BCA = 180) ∧ (DCE + x = 90)) :
  x = 30 := by
  sorry

end angle_x_l2181_218164


namespace find_a_plus_b_l2181_218161

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 1 = a - b) 
  (h2 : 5 = a - b / 5) : a + b = 11 :=
by
  sorry

end find_a_plus_b_l2181_218161


namespace sum_first_n_terms_geom_seq_l2181_218158

def geom_seq (n : ℕ) : ℕ :=
match n with
| 0     => 2
| k + 1 => 3 * geom_seq k

def sum_geom_seq (n : ℕ) : ℕ :=
(geom_seq 0) * (3 ^ n - 1) / (3 - 1)

theorem sum_first_n_terms_geom_seq (n : ℕ) :
sum_geom_seq n = 3 ^ n - 1 := by
sorry

end sum_first_n_terms_geom_seq_l2181_218158


namespace wrench_force_l2181_218188

def force_inversely_proportional (f1 f2 : ℝ) (L1 L2 : ℝ) : Prop :=
  f1 * L1 = f2 * L2

theorem wrench_force
  (f1 : ℝ) (L1 : ℝ) (f2 : ℝ) (L2 : ℝ)
  (h1 : L1 = 12) (h2 : f1 = 450) (h3 : L2 = 18) (h_prop : force_inversely_proportional f1 f2 L1 L2) :
  f2 = 300 :=
by
  sorry

end wrench_force_l2181_218188


namespace probability_one_painted_face_l2181_218198

def cube : ℕ := 5
def total_unit_cubes : ℕ := 125
def painted_faces_share_edge : Prop := true
def unit_cubes_with_one_painted_face : ℕ := 41

theorem probability_one_painted_face :
  ∃ (cube : ℕ) (total_unit_cubes : ℕ) (painted_faces_share_edge : Prop) (unit_cubes_with_one_painted_face : ℕ),
  cube = 5 ∧ total_unit_cubes = 125 ∧ painted_faces_share_edge ∧ unit_cubes_with_one_painted_face = 41 →
  (unit_cubes_with_one_painted_face : ℚ) / (total_unit_cubes : ℚ) = 41 / 125 :=
by 
  sorry

end probability_one_painted_face_l2181_218198


namespace sequence_a100_gt_14_l2181_218195

theorem sequence_a100_gt_14 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, 1 ≤ n → a (n+1) = a n + 1 / a n) :
  a 100 > 14 :=
by sorry

end sequence_a100_gt_14_l2181_218195


namespace fraction_of_robs_doubles_is_one_third_l2181_218143

theorem fraction_of_robs_doubles_is_one_third 
  (total_robs_cards : ℕ) (total_jess_doubles : ℕ) 
  (times_jess_doubles_robs : ℕ)
  (robs_doubles : ℕ) :
  total_robs_cards = 24 →
  total_jess_doubles = 40 →
  times_jess_doubles_robs = 5 →
  total_jess_doubles = times_jess_doubles_robs * robs_doubles →
  (robs_doubles : ℚ) / total_robs_cards = 1 / 3 := 
by 
  intros h1 h2 h3 h4
  sorry

end fraction_of_robs_doubles_is_one_third_l2181_218143


namespace solve_for_x_l2181_218119

theorem solve_for_x : ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 :=
by 
  intros x hx h
  sorry

end solve_for_x_l2181_218119


namespace area_percentage_decrease_42_l2181_218180

def radius_decrease_factor : ℝ := 0.7615773105863908

noncomputable def area_percentage_decrease : ℝ :=
  let k := radius_decrease_factor
  100 * (1 - k^2)

theorem area_percentage_decrease_42 :
  area_percentage_decrease = 42 := by
  sorry

end area_percentage_decrease_42_l2181_218180


namespace probability_of_top_grade_product_l2181_218181

-- Definitions for the problem conditions
def P_B : ℝ := 0.03
def P_C : ℝ := 0.01

-- Given that the sum of all probabilities is 1
axiom sum_of_probabilities (P_A P_B P_C : ℝ) : P_A + P_B + P_C = 1

-- Statement to be proved
theorem probability_of_top_grade_product : ∃ P_A : ℝ, P_A = 1 - P_B - P_C ∧ P_A = 0.96 :=
by
  -- Assuming the proof steps to derive the answer
  sorry

end probability_of_top_grade_product_l2181_218181


namespace divides_expression_l2181_218117

theorem divides_expression (x : ℕ) (hx : Even x) : 90 ∣ (15 * x + 3) * (15 * x + 9) * (5 * x + 10) :=
sorry

end divides_expression_l2181_218117


namespace wrapping_paper_fraction_each_present_l2181_218170

theorem wrapping_paper_fraction_each_present (total_fraction : ℚ) (num_presents : ℕ) 
  (H : total_fraction = 3/10) (H1 : num_presents = 3) :
  total_fraction / num_presents = 1/10 :=
by sorry

end wrapping_paper_fraction_each_present_l2181_218170


namespace actual_speed_of_car_l2181_218125

noncomputable def actual_speed (t : ℝ) (d : ℝ) (reduced_speed_factor : ℝ) : ℝ := 
  (d / t) * (1 / reduced_speed_factor)

noncomputable def time_in_hours : ℝ := 1 + (40 / 60) + (48 / 3600)

theorem actual_speed_of_car : 
  actual_speed time_in_hours 42 (5 / 7) = 35 :=
by
  sorry

end actual_speed_of_car_l2181_218125


namespace equations_not_equivalent_l2181_218184

variable {X : Type} [Field X]
variable (A B : X → X)

theorem equations_not_equivalent (h1 : ∀ x, A x ^ 2 = B x ^ 2) (h2 : ¬∀ x, A x = B x) :
  (∃ x, A x ≠ B x ∨ A x ≠ -B x) := 
sorry

end equations_not_equivalent_l2181_218184


namespace initial_parts_planned_l2181_218165

variable (x : ℕ)

theorem initial_parts_planned (x : ℕ) (h : 3 * x + (x + 5) + 100 = 675): x = 142 :=
by sorry

end initial_parts_planned_l2181_218165


namespace geometric_series_sum_l2181_218172

-- Define the terms of the series
def a : ℚ := 1 / 5
def r : ℚ := -1 / 3
def n : ℕ := 6

-- Define the expected sum
def expected_sum : ℚ := 182 / 1215

-- Prove that the sum of the geometric series equals the expected sum
theorem geometric_series_sum : 
  (a * (1 - r^n)) / (1 - r) = expected_sum := 
by
  sorry

end geometric_series_sum_l2181_218172


namespace pool_capacity_l2181_218179

theorem pool_capacity (C : ℝ) (initial_water : ℝ) :
  0.85 * C - 0.70 * C = 300 → C = 2000 :=
by
  intro h
  sorry

end pool_capacity_l2181_218179


namespace cost_of_bricks_l2181_218124

theorem cost_of_bricks
  (N: ℕ)
  (half_bricks:ℕ)
  (full_price: ℝ)
  (discount_percentage: ℝ)
  (n_half: half_bricks = N / 2)
  (P1: full_price = 0.5)
  (P2: discount_percentage = 0.5):
  (half_bricks * (full_price * discount_percentage) + 
  half_bricks * full_price = 375) := 
by sorry

end cost_of_bricks_l2181_218124


namespace clean_room_time_l2181_218111

theorem clean_room_time :
  let lisa_time := 8
  let kay_time := 12
  let ben_time := 16
  let combined_work_rate := (1 / lisa_time) + (1 / kay_time) + (1 / ben_time)
  let total_time := 1 / combined_work_rate
  total_time = 48 / 13 :=
by
  sorry

end clean_room_time_l2181_218111


namespace luke_total_coins_l2181_218132

def piles_coins_total (piles_quarters : ℕ) (coins_per_pile_quarters : ℕ) 
                      (piles_dimes : ℕ) (coins_per_pile_dimes : ℕ) 
                      (piles_nickels : ℕ) (coins_per_pile_nickels : ℕ) 
                      (piles_pennies : ℕ) (coins_per_pile_pennies : ℕ) : ℕ :=
  (piles_quarters * coins_per_pile_quarters) +
  (piles_dimes * coins_per_pile_dimes) +
  (piles_nickels * coins_per_pile_nickels) +
  (piles_pennies * coins_per_pile_pennies)

theorem luke_total_coins : 
  piles_coins_total 8 5 6 7 4 4 3 6 = 116 :=
by
  sorry

end luke_total_coins_l2181_218132


namespace joan_sandwiches_l2181_218140

theorem joan_sandwiches :
  ∀ (H : ℕ), (∀ (h_slice g_slice total_cheese num_grilled_cheese : ℕ),
  h_slice = 2 →
  g_slice = 3 →
  num_grilled_cheese = 10 →
  total_cheese = 50 →
  total_cheese - num_grilled_cheese * g_slice = H * h_slice →
  H = 10) :=
by
  intros H h_slice g_slice total_cheese num_grilled_cheese h_slice_eq g_slice_eq num_grilled_cheese_eq total_cheese_eq cheese_eq
  sorry

end joan_sandwiches_l2181_218140


namespace digits_solution_exists_l2181_218113

theorem digits_solution_exists (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : a = (b * (10 * b)) / (10 - b)) : a = 5 ∧ b = 2 :=
by
  sorry

end digits_solution_exists_l2181_218113


namespace least_number_to_subtract_l2181_218197

theorem least_number_to_subtract (x : ℕ) (h1 : 997 - x ≡ 3 [MOD 17]) (h2 : 997 - x ≡ 3 [MOD 19]) (h3 : 997 - x ≡ 3 [MOD 23]) : x = 3 :=
by
  sorry

end least_number_to_subtract_l2181_218197


namespace closest_correct_option_l2181_218118

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f x = f (-x + 16)) -- y = f(x + 8) is an even function
variable (h2 : ∀ a b, 8 < a → 8 < b → a < b → f b < f a) -- f is decreasing on (8, +∞)

theorem closest_correct_option :
  f 7 > f 10 := by
  -- Insert proof here
  sorry

end closest_correct_option_l2181_218118


namespace problem_l2181_218199

-- Definitions for the problem's conditions:
variables {a b c d : ℝ}

-- a and b are roots of x^2 + 68x + 1 = 0
axiom ha : a ^ 2 + 68 * a + 1 = 0
axiom hb : b ^ 2 + 68 * b + 1 = 0

-- c and d are roots of x^2 - 86x + 1 = 0
axiom hc : c ^ 2 - 86 * c + 1 = 0
axiom hd : d ^ 2 - 86 * d + 1 = 0

theorem problem : (a + c) * (b + c) * (a - d) * (b - d) = 2772 :=
sorry

end problem_l2181_218199


namespace find_difference_between_larger_and_fraction_smaller_l2181_218115

theorem find_difference_between_larger_and_fraction_smaller
  (x y : ℝ) 
  (h1 : x + y = 147)
  (h2 : x - 0.375 * y = 4) : x - 0.375 * y = 4 :=
by
  sorry

end find_difference_between_larger_and_fraction_smaller_l2181_218115


namespace hyperbola_equation_l2181_218185

noncomputable def h : ℝ := -4
noncomputable def k : ℝ := 2
noncomputable def a : ℝ := 1
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1

theorem hyperbola_equation :
  (h + k + a + b) = 0 := by
  have h := -4
  have k := 2
  have a := 1
  have b := 1
  show (-4 + 2 + 1 + 1) = 0
  sorry

end hyperbola_equation_l2181_218185


namespace fifth_group_pythagorean_triples_l2181_218189

theorem fifth_group_pythagorean_triples :
  ∃ (a b c : ℕ), (a, b, c) = (11, 60, 61) ∧ a^2 + b^2 = c^2 :=
by
  use 11, 60, 61
  sorry

end fifth_group_pythagorean_triples_l2181_218189


namespace other_x_intercept_l2181_218130

theorem other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, y = a * x ^ 2 + b * x + c → (x, y) = (4, -3)) (h_x_intercept : ∀ y, y = a * 1 ^ 2 + b * 1 + c → (1, y) = (1, 0)) : 
  ∃ x, x = 7 := by
sorry

end other_x_intercept_l2181_218130


namespace probability_odd_product_lt_one_eighth_l2181_218134

theorem probability_odd_product_lt_one_eighth :
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  p < 1 / 8 :=
by
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  sorry

end probability_odd_product_lt_one_eighth_l2181_218134


namespace multiplier_for_doberman_puppies_l2181_218190

theorem multiplier_for_doberman_puppies 
  (D : ℕ) (S : ℕ) (M : ℝ) 
  (hD : D = 20) 
  (hS : S = 55) 
  (h : D * M + (D - S) = 90) : 
  M = 6.25 := 
by 
  sorry

end multiplier_for_doberman_puppies_l2181_218190


namespace base_seven_to_ten_l2181_218166

theorem base_seven_to_ten :
  (6 * 7^4 + 5 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 7^0) = 16244 :=
by sorry

end base_seven_to_ten_l2181_218166


namespace inequality_abc_l2181_218133

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
sorry

end inequality_abc_l2181_218133


namespace square_area_EFGH_l2181_218103

theorem square_area_EFGH (AB BP : ℝ) (h1 : AB = Real.sqrt 72) (h2 : BP = 2) (x : ℝ)
  (h3 : AB + BP = 2 * x + 2) : x^2 = 18 :=
by
  sorry

end square_area_EFGH_l2181_218103


namespace hexagonalPrismCannotIntersectAsCircle_l2181_218109

-- Define each geometric shape as a type
inductive GeometricShape
| Sphere
| Cone
| Cylinder
| HexagonalPrism

-- Define a function that checks if a shape can be intersected by a plane to form a circular cross-section
def canIntersectAsCircle (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => True -- Sphere can always form a circular cross-section
  | GeometricShape.Cone => True -- Cone can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.Cylinder => True -- Cylinder can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.HexagonalPrism => False -- Hexagonal Prism cannot form a circular cross-section

-- The theorem to prove
theorem hexagonalPrismCannotIntersectAsCircle :
  ∀ shape : GeometricShape,
  (shape = GeometricShape.HexagonalPrism) ↔ ¬ canIntersectAsCircle shape := by
  sorry

end hexagonalPrismCannotIntersectAsCircle_l2181_218109


namespace number_of_insects_l2181_218101

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 30) (h2 : legs_per_insect = 6) :
  total_legs / legs_per_insect = 5 :=
by
  sorry

end number_of_insects_l2181_218101


namespace rain_at_least_once_l2181_218114

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l2181_218114


namespace average_monthly_growth_rate_equation_l2181_218112

-- Definitions directly from the conditions
def JanuaryOutput : ℝ := 50
def QuarterTotalOutput : ℝ := 175
def averageMonthlyGrowthRate (x : ℝ) : ℝ :=
  JanuaryOutput + JanuaryOutput * (1 + x) + JanuaryOutput * (1 + x) ^ 2

-- The statement to prove that the derived equation is correct
theorem average_monthly_growth_rate_equation (x : ℝ) :
  averageMonthlyGrowthRate x = QuarterTotalOutput :=
sorry

end average_monthly_growth_rate_equation_l2181_218112


namespace largest_integer_satisfying_inequality_l2181_218126

theorem largest_integer_satisfying_inequality :
  ∃ n : ℤ, n = 4 ∧ (1 / 4 + n / 8 < 7 / 8) ∧ ∀ m : ℤ, m > 4 → ¬(1 / 4 + m / 8 < 7 / 8) :=
by
  sorry

end largest_integer_satisfying_inequality_l2181_218126


namespace calculate_expression_l2181_218151

variable (x y : ℝ)

theorem calculate_expression (h1 : x + y = 5) (h2 : x * y = 3) : 
   x + (x^4 / y^3) + (y^4 / x^3) + y = 27665 / 27 :=
by
  sorry

end calculate_expression_l2181_218151


namespace birds_nest_building_area_scientific_notation_l2181_218108

theorem birds_nest_building_area_scientific_notation :
  (258000 : ℝ) = 2.58 * 10^5 :=
by sorry

end birds_nest_building_area_scientific_notation_l2181_218108


namespace cost_of_one_bag_l2181_218131

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l2181_218131


namespace find_roots_l2181_218171

theorem find_roots (x : ℝ) : (x^2 + x = 0) ↔ (x = 0 ∨ x = -1) := 
by sorry

end find_roots_l2181_218171


namespace clothing_percentage_l2181_218152

variable (T : ℝ) -- Total amount excluding taxes.
variable (C : ℝ) -- Percentage of total amount spent on clothing.

-- Conditions
def spent_on_food := 0.2 * T
def spent_on_other_items := 0.3 * T

-- Taxes
def tax_on_clothing := 0.04 * (C * T)
def tax_on_food := 0.0
def tax_on_other_items := 0.08 * (0.3 * T)
def total_tax_paid := 0.044 * T

-- Statement to prove
theorem clothing_percentage : 
  0.04 * (C * T) + 0.08 * (0.3 * T) = 0.044 * T ↔ C = 0.5 :=
by
  sorry

end clothing_percentage_l2181_218152


namespace arithmetic_square_root_l2181_218154

noncomputable def cube_root (x : ℝ) : ℝ :=
  x^(1/3)

noncomputable def sqrt_int_part (x : ℝ) : ℤ :=
  ⌊Real.sqrt x⌋

theorem arithmetic_square_root 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (h1 : cube_root a = 2) 
  (h2 : b = sqrt_int_part 5) 
  (h3 : c = 4 ∨ c = -4) : 
  Real.sqrt (a + ↑b + c) = Real.sqrt 14 ∨ Real.sqrt (a + ↑b + c) = Real.sqrt 6 := 
sorry

end arithmetic_square_root_l2181_218154


namespace Cora_book_reading_problem_l2181_218145

theorem Cora_book_reading_problem
  (total_pages: ℕ)
  (read_monday: ℕ)
  (read_tuesday: ℕ)
  (read_wednesday: ℕ)
  (H: total_pages = 158 ∧ read_monday = 23 ∧ read_tuesday = 38 ∧ read_wednesday = 61) :
  ∃ P: ℕ, 23 + 38 + 61 + P + 2 * P = total_pages ∧ P = 12 :=
  sorry

end Cora_book_reading_problem_l2181_218145


namespace ratio_a_c_l2181_218186

variables (a b c d : ℚ)

axiom ratio_a_b : a / b = 5 / 4
axiom ratio_c_d : c / d = 4 / 3
axiom ratio_d_b : d / b = 1 / 8

theorem ratio_a_c : a / c = 15 / 2 :=
by sorry

end ratio_a_c_l2181_218186


namespace debby_bottles_per_day_l2181_218138

theorem debby_bottles_per_day :
  let total_bottles := 153
  let days := 17
  total_bottles / days = 9 :=
by
  sorry

end debby_bottles_per_day_l2181_218138


namespace trigonometric_identity_l2181_218149

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / (Real.cos (3 * π / 2 - θ) - Real.sin (π - θ)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l2181_218149


namespace biology_marks_l2181_218162

theorem biology_marks (E M P C: ℝ) (A: ℝ) (N: ℕ) 
  (hE: E = 96) (hM: M = 98) (hP: P = 99) (hC: C = 100) (hA: A = 98.2) (hN: N = 5):
  (E + M + P + C + B) / N = A → B = 98 :=
by
  intro h
  sorry

end biology_marks_l2181_218162


namespace totalExerciseTime_l2181_218137

-- Define the conditions
def caloriesBurnedRunningPerMinute := 10
def caloriesBurnedWalkingPerMinute := 4
def totalCaloriesBurned := 450
def runningTime := 35

-- Define the problem as a theorem to be proven
theorem totalExerciseTime :
  ((runningTime * caloriesBurnedRunningPerMinute) + 
  ((totalCaloriesBurned - runningTime * caloriesBurnedRunningPerMinute) / caloriesBurnedWalkingPerMinute)) = 60 := 
sorry

end totalExerciseTime_l2181_218137


namespace largest_integer_divisible_example_1748_largest_n_1748_l2181_218160

theorem largest_integer_divisible (n : ℕ) (h : (n + 12) ∣ (n^3 + 160)) : n ≤ 1748 :=
by
  sorry

theorem example_1748 : 1748^3 + 160 = 1760 * 3045738 :=
by
  sorry

theorem largest_n_1748 (n : ℕ) (h : 1748 ≤ n) : (n + 12) ∣ (n^3 + 160) :=
by
  sorry

end largest_integer_divisible_example_1748_largest_n_1748_l2181_218160


namespace hyperbola_eccentricity_l2181_218187

theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : x₀^2 / a^2 - y₀^2 / b^2 = 1)
  (h₄ : a ≤ x₀ ∧ x₀ ≤ 2 * a)
  (h₅ : x₀ / a^2 * 0 - y₀ / b^2 * b = 1)
  (h₆ : - (a * a / (2 * b)) = 2) :
  (1 + b^2 / a^2 = 3) :=
sorry

end hyperbola_eccentricity_l2181_218187


namespace base_area_of_rect_prism_l2181_218175

theorem base_area_of_rect_prism (r : ℝ) (h : ℝ) (V : ℝ) (h_rate : ℝ) (V_rate : ℝ) (conversion : ℝ) :
  V_rate = conversion * V ∧ h_rate = h → ∃ A : ℝ, A = V / h ∧ A = 100 :=
by
  sorry

end base_area_of_rect_prism_l2181_218175


namespace range_of_a_l2181_218107

variable (a : ℝ)

theorem range_of_a (h : ∀ x : ℤ, 2 * (x:ℝ)^2 - 17 * x + a ≤ 0 →  (x = 3 ∨ x = 4 ∨ x = 5)) : 
  30 < a ∧ a ≤ 33 :=
sorry

end range_of_a_l2181_218107


namespace angle_between_diagonal_and_base_l2181_218169

theorem angle_between_diagonal_and_base 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ θ : ℝ, θ = Real.arctan (Real.sin (α / 2)) :=
sorry

end angle_between_diagonal_and_base_l2181_218169
