import Mathlib

namespace value_of_y_minus_x_l966_96659

theorem value_of_y_minus_x (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
sorry

end value_of_y_minus_x_l966_96659


namespace stratified_sampling_category_A_l966_96636

def total_students_A : ℕ := 2000
def total_students_B : ℕ := 3000
def total_students_C : ℕ := 4000
def total_students : ℕ := total_students_A + total_students_B + total_students_C
def total_selected : ℕ := 900

theorem stratified_sampling_category_A :
  (total_students_A * total_selected) / total_students = 200 :=
by
  sorry

end stratified_sampling_category_A_l966_96636


namespace dave_added_apps_l966_96689

-- Define the conditions as a set of given facts
def initial_apps : Nat := 10
def deleted_apps : Nat := 17
def remaining_apps : Nat := 4

-- The statement to prove
theorem dave_added_apps : ∃ x : Nat, initial_apps + x - deleted_apps = remaining_apps ∧ x = 11 :=
by
  use 11
  sorry

end dave_added_apps_l966_96689


namespace fraction_area_below_diagonal_is_one_l966_96640

noncomputable def fraction_below_diagonal (s : ℝ) : ℝ := 1

theorem fraction_area_below_diagonal_is_one (s : ℝ) :
  let long_side := 2 * s
  let P := (2 * s / 3, 0)
  let Q := (s, s / 2)
  -- Total area of the rectangle
  let total_area := s * 2 * s -- 2s^2
  -- Total area below the diagonal
  let area_below_diagonal := 2 * s * s  -- 2s^2
  -- Fraction of the area below diagonal
  fraction_below_diagonal s = area_below_diagonal / total_area := 
by 
  sorry

end fraction_area_below_diagonal_is_one_l966_96640


namespace number_of_real_roots_l966_96658

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then Real.exp x else -x^2 + 2.5 * x

theorem number_of_real_roots : ∃! x, f x = 0.5 * x + 1 :=
sorry

end number_of_real_roots_l966_96658


namespace twelve_women_reseated_l966_96620

def S (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 3
  else S (n - 1) + S (n - 2) + S (n - 3)

theorem twelve_women_reseated : S 12 = 1201 :=
by
  sorry

end twelve_women_reseated_l966_96620


namespace ratio_of_c_and_d_l966_96600

theorem ratio_of_c_and_d (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 3 * x + 2 * y = c) 
  (h2 : 4 * y - 6 * x = d) : c / d = -1 / 3 := 
sorry

end ratio_of_c_and_d_l966_96600


namespace initial_mean_of_observations_l966_96609

-- Definitions of the given conditions and proof of the correct initial mean
theorem initial_mean_of_observations 
  (M : ℝ) -- Mean of 50 observations
  (initial_sum := 50 * M) -- Initial sum of observations
  (wrong_observation : ℝ := 23) -- Wrong observation
  (correct_observation : ℝ := 45) -- Correct observation
  (understated_by := correct_observation - wrong_observation) -- Amount of understatement
  (correct_sum := initial_sum + understated_by) -- Corrected sum
  (corrected_mean : ℝ := 36.5) -- Corrected new mean
  (eq1 : correct_sum = 50 * corrected_mean) -- Equation from condition of corrected mean
  (eq2 : initial_sum = 50 * corrected_mean - understated_by) -- Restating in terms of initial sum
  : M = 36.06 := -- The initial mean of observations
  sorry -- Proof omitted

end initial_mean_of_observations_l966_96609


namespace population_net_increase_in_one_day_l966_96650

-- Definitions based on the conditions
def birth_rate_per_two_seconds : ℝ := 4
def death_rate_per_two_seconds : ℝ := 3
def seconds_in_a_day : ℝ := 86400

-- The main theorem to prove
theorem population_net_increase_in_one_day : 
  (birth_rate_per_two_seconds / 2 - death_rate_per_two_seconds / 2) * seconds_in_a_day = 43200 :=
by
  sorry

end population_net_increase_in_one_day_l966_96650


namespace g_neg_one_l966_96675

def g (d e f x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

theorem g_neg_one {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end g_neg_one_l966_96675


namespace hearing_news_probability_l966_96651

noncomputable def probability_of_hearing_news : ℚ :=
  let broadcast_cycle := 30 -- total time in minutes for each broadcast cycle
  let news_duration := 5  -- duration of each news broadcast in minutes
  news_duration / broadcast_cycle

theorem hearing_news_probability : probability_of_hearing_news = 1 / 6 := by
  sorry

end hearing_news_probability_l966_96651


namespace complex_norm_wz_l966_96697

open Complex

theorem complex_norm_wz (w z : ℂ) (h₁ : ‖w + z‖ = 2) (h₂ : ‖w^2 + z^2‖ = 8) : 
  ‖w^4 + z^4‖ = 56 := 
  sorry

end complex_norm_wz_l966_96697


namespace smallest_nat_divisible_by_48_squared_l966_96695

theorem smallest_nat_divisible_by_48_squared :
  ∃ n : ℕ, (n % (48^2) = 0) ∧ 
           (∀ (d : ℕ), d ∈ (Nat.digits n 10) → d = 0 ∨ d = 1) ∧ 
           (n = 11111111100000000) := sorry

end smallest_nat_divisible_by_48_squared_l966_96695


namespace francine_leave_time_earlier_l966_96687

-- Definitions for the conditions in the problem
def leave_time := "noon"  -- Francine and her father leave at noon every day.
def father_meet_time_shorten := 10  -- They arrived home 10 minutes earlier than usual.
def francine_walk_duration := 15  -- Francine walked for 15 minutes.

-- Premises based on the conditions
def usual_meet_time := 12 * 60  -- Meeting time in minutes from midnight (noon = 720 minutes)
def special_day_meet_time := usual_meet_time - father_meet_time_shorten / 2  -- 5 minutes earlier

-- The main theorem to prove: Francine leaves at 11:40 AM (700 minutes from midnight)
theorem francine_leave_time_earlier :
  usual_meet_time - (father_meet_time_shorten / 2 + francine_walk_duration) = (11 * 60 + 40) := by
  sorry

end francine_leave_time_earlier_l966_96687


namespace D_won_zero_matches_l966_96617

-- Define the players
inductive Player
| A | B | C | D deriving DecidableEq

-- Function to determine the winner of a match
def match_winner (p1 p2 : Player) : Option Player :=
  if p1 = Player.A ∧ p2 = Player.D then 
    some Player.A
  else if p2 = Player.A ∧ p1 = Player.D then 
    some Player.A
  else 
    none -- This represents that we do not know the outcome for matches not given

-- Assuming A, B, and C have won the same number of matches
def same_wins (w_A w_B w_C : Nat) : Prop := 
  w_A = w_B ∧ w_B = w_C

-- Define the problem statement
theorem D_won_zero_matches (w_D : Nat) (h_winner_AD: match_winner Player.A Player.D = some Player.A)
  (h_same_wins : ∃ w_A w_B w_C : Nat, same_wins w_A w_B w_C) : w_D = 0 :=
sorry

end D_won_zero_matches_l966_96617


namespace find_f_of_1_over_3_l966_96606

theorem find_f_of_1_over_3
  (g : ℝ → ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, g x = 1 - x^2)
  (h2 : ∀ x, x ≠ 0 → f (g x) = (1 - x^2) / x^2) :
  f (1 / 3) = 1 / 2 := by
  sorry -- Proof goes here

end find_f_of_1_over_3_l966_96606


namespace max_marks_l966_96645

variable (M : ℝ)

theorem max_marks (h1 : 0.35 * M = 175) : M = 500 := by
  -- Proof goes here
  sorry

end max_marks_l966_96645


namespace benny_gave_seashells_l966_96685

theorem benny_gave_seashells (original_seashells : ℕ) (remaining_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 66) 
  (h2 : remaining_seashells = 14) 
  (h3 : original_seashells - remaining_seashells = given_seashells) : 
  given_seashells = 52 := 
by
  sorry

end benny_gave_seashells_l966_96685


namespace combined_area_difference_l966_96676

theorem combined_area_difference :
  let rect1_len := 11
  let rect1_wid := 11
  let rect2_len := 5.5
  let rect2_wid := 11
  2 * (rect1_len * rect1_wid) - 2 * (rect2_len * rect2_wid) = 121 := by
  sorry

end combined_area_difference_l966_96676


namespace least_value_m_n_l966_96644

theorem least_value_m_n :
  ∃ m n : ℕ, (m > 0 ∧ n > 0) ∧
            (Nat.gcd (m + n) 231 = 1) ∧
            (n^n ∣ m^m) ∧
            ¬ (m % n = 0) ∧
            m + n = 377 :=
by 
  sorry

end least_value_m_n_l966_96644


namespace ratio_of_speeds_is_2_l966_96686

-- Definitions based on conditions
def rate_of_machine_B : ℕ := 100 / 40 -- Rate of Machine B (parts per minute)
def rate_of_machine_A : ℕ := 50 / 10 -- Rate of Machine A (parts per minute)
def ratio_of_speeds (rate_A rate_B : ℕ) : ℕ := rate_A / rate_B -- Ratio of speeds

-- Proof statement
theorem ratio_of_speeds_is_2 : ratio_of_speeds rate_of_machine_A rate_of_machine_B = 2 := by
  sorry

end ratio_of_speeds_is_2_l966_96686


namespace find_a_b_l966_96627

-- Define that the roots of the corresponding equality yield the specific conditions.
theorem find_a_b (a b : ℝ) :
    (∀ x : ℝ, x^2 + (a + 1) * x + ab > 0 ↔ (x < -1 ∨ x > 4)) →
    a = -4 ∧ b = 1 := 
by
    sorry

end find_a_b_l966_96627


namespace simplify_expression_l966_96601

noncomputable def simplify_expr (a b : ℝ) : ℝ :=
  (3 * a^5 * b^3 + a^4 * b^2) / (-(a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b)

theorem simplify_expression (a b : ℝ) :
  simplify_expr a b = 8 * a * b - 3 := 
by
  sorry

end simplify_expression_l966_96601


namespace film_cost_eq_five_l966_96604

variable (F : ℕ)

theorem film_cost_eq_five (H1 : 9 * F + 4 * 4 + 6 * 3 = 79) : F = 5 :=
by
  -- This is a placeholder for your proof
  sorry

end film_cost_eq_five_l966_96604


namespace find_initial_children_l966_96663

-- Definition of conditions
def initial_children_on_bus (X : ℕ) := 
  let final_children := (X + 40) - 60 
  final_children = 2

-- Theorem statement
theorem find_initial_children : 
  ∃ X : ℕ, initial_children_on_bus X ∧ X = 22 :=
by
  sorry

end find_initial_children_l966_96663


namespace point_on_parabola_distance_l966_96610

theorem point_on_parabola_distance (a b : ℝ) (h1 : a^2 = 20 * b) (h2 : |b + 5| = 25) : |a * b| = 400 :=
sorry

end point_on_parabola_distance_l966_96610


namespace factorize_expression_l966_96643

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 := 
  sorry

end factorize_expression_l966_96643


namespace closest_integer_to_99_times_9_l966_96642

theorem closest_integer_to_99_times_9 :
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  1000 ∈ choices ∧ ∀ (n : ℤ), n ∈ choices → dist 1000 ≤ dist n :=
by
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  sorry

end closest_integer_to_99_times_9_l966_96642


namespace average_speed_is_6_point_5_l966_96629

-- Define the given values
def total_distance : ℝ := 42
def riding_time : ℝ := 6
def break_time : ℝ := 0.5

-- Prove the average speed given the conditions
theorem average_speed_is_6_point_5 :
  (total_distance / (riding_time + break_time)) = 6.5 :=
by
  sorry

end average_speed_is_6_point_5_l966_96629


namespace train_crosses_signal_pole_l966_96656

theorem train_crosses_signal_pole 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (time_cross_platform : ℝ) 
  (speed : ℝ) 
  (time_cross_signal_pole : ℝ) : 
  length_train = 400 → 
  length_platform = 200 → 
  time_cross_platform = 45 → 
  speed = (length_train + length_platform) / time_cross_platform → 
  time_cross_signal_pole = length_train / speed -> 
  time_cross_signal_pole = 30 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1] at h5
  -- Add the necessary calculations here
  sorry

end train_crosses_signal_pole_l966_96656


namespace percent_cities_less_than_50000_l966_96614

-- Definitions of the conditions
def percent_cities_50000_to_149999 := 40
def percent_cities_less_than_10000 := 35
def percent_cities_10000_to_49999 := 10
def percent_cities_150000_or_more := 15

-- Prove that the total percentage of cities with fewer than 50,000 residents is 45%
theorem percent_cities_less_than_50000 :
  percent_cities_less_than_10000 + percent_cities_10000_to_49999 = 45 :=
by
  sorry

end percent_cities_less_than_50000_l966_96614


namespace max_value_g_l966_96688

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (4 - x))

theorem max_value_g : ∃ (x₁ N : ℝ), (0 ≤ x₁ ∧ x₁ ≤ 4) ∧ (N = 16) ∧ (x₁ = 2) ∧ (∀ x, 0 ≤ x ∧ x ≤ 4 → g x ≤ N) :=
by
  sorry

end max_value_g_l966_96688


namespace handed_out_apples_l966_96668

def total_apples : ℤ := 96
def pies : ℤ := 9
def apples_per_pie : ℤ := 6
def apples_for_pies : ℤ := pies * apples_per_pie
def apples_handed_out : ℤ := total_apples - apples_for_pies

theorem handed_out_apples : apples_handed_out = 42 := by
  sorry

end handed_out_apples_l966_96668


namespace Andrena_more_than_Debelyn_l966_96682

-- Definitions based on the problem conditions
def Debelyn_initial := 20
def Debelyn_gift_to_Andrena := 2
def Christel_initial := 24
def Christel_gift_to_Andrena := 5
def Andrena_more_than_Christel := 2

-- Calculating the number of dolls each person has after the gifts
def Debelyn_final := Debelyn_initial - Debelyn_gift_to_Andrena
def Christel_final := Christel_initial - Christel_gift_to_Andrena
def Andrena_final := Christel_final + Andrena_more_than_Christel

-- The proof problem statement
theorem Andrena_more_than_Debelyn : Andrena_final - Debelyn_final = 3 := by
  sorry

end Andrena_more_than_Debelyn_l966_96682


namespace class_weighted_average_l966_96654

theorem class_weighted_average
    (num_students : ℕ)
    (sect1_avg sect2_avg sect3_avg remainder_avg : ℝ)
    (sect1_pct sect2_pct sect3_pct remainder_pct : ℝ)
    (weight1 weight2 weight3 weight4 : ℝ)
    (h_total_students : num_students = 120)
    (h_sect1_avg : sect1_avg = 96.5)
    (h_sect2_avg : sect2_avg = 78.4)
    (h_sect3_avg : sect3_avg = 88.2)
    (h_remainder_avg : remainder_avg = 64.7)
    (h_sect1_pct : sect1_pct = 0.187)
    (h_sect2_pct : sect2_pct = 0.355)
    (h_sect3_pct : sect3_pct = 0.258)
    (h_remainder_pct : remainder_pct = 1 - (sect1_pct + sect2_pct + sect3_pct))
    (h_weight1 : weight1 = 0.35)
    (h_weight2 : weight2 = 0.25)
    (h_weight3 : weight3 = 0.30)
    (h_weight4 : weight4 = 0.10) :
    (sect1_avg * weight1 + sect2_avg * weight2 + sect3_avg * weight3 + remainder_avg * weight4) * 100 = 86 := 
sorry

end class_weighted_average_l966_96654


namespace work_completion_l966_96605

theorem work_completion (A B : ℝ → ℝ) (h1 : ∀ t, A t = B t) (h3 : A 4 + B 4 = 1) : B 1 = 1/2 :=
by {
  sorry
}

end work_completion_l966_96605


namespace circle_center_radius_l966_96622

theorem circle_center_radius (x y : ℝ) :
  (x^2 + y^2 + 4 * x - 6 * y = 11) →
  ∃ (h k r : ℝ), h = -2 ∧ k = 3 ∧ r = 2 * Real.sqrt 6 ∧
  (x+h)^2 + (y+k)^2 = r^2 :=
by
  sorry

end circle_center_radius_l966_96622


namespace solve_for_y_l966_96625

theorem solve_for_y (x y : ℝ) (h : 5 * x - y = 6) : y = 5 * x - 6 :=
sorry

end solve_for_y_l966_96625


namespace solve_eqn_l966_96671

theorem solve_eqn (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  3 ^ x = 2 ^ x * y + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end solve_eqn_l966_96671


namespace arithmetic_mean_of_14_22_36_l966_96657

theorem arithmetic_mean_of_14_22_36 : (14 + 22 + 36) / 3 = 24 := by
  sorry

end arithmetic_mean_of_14_22_36_l966_96657


namespace augmented_matrix_correct_l966_96681

-- Define the system of linear equations as a pair of equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x + y = 1) ∧ (3 * x - 2 * y = 0)

-- Define what it means to be the correct augmented matrix for the system
def is_augmented_matrix (A : Matrix (Fin 2) (Fin 3) ℝ) : Prop :=
  A = ![
    ![2, 1, 1],
    ![3, -2, 0]
  ]

-- The theorem states that the augmented matrix of the given system of equations is the specified matrix
theorem augmented_matrix_correct :
  ∃ x y : ℝ, system_of_equations x y ∧ is_augmented_matrix ![
    ![2, 1, 1],
    ![3, -2, 0]
  ] :=
sorry

end augmented_matrix_correct_l966_96681


namespace find_x_perpendicular_l966_96698

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 3)
def b (x : ℝ) : ℝ × ℝ := (-3, x)

-- Define the condition that the dot product of vectors a and b is zero
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Statement we need to prove
theorem find_x_perpendicular (x : ℝ) (h : perpendicular a (b x)) : x = -1 :=
by sorry

end find_x_perpendicular_l966_96698


namespace compare_charges_l966_96674

/-
Travel agencies A and B have group discount methods with the original price being $200 per person.
- Agency A: Buy 4 full-price tickets, the rest are half price.
- Agency B: All customers get a 30% discount.
Prove the given relationships based on the number of travelers.
-/

def agency_a_cost (x : ℕ) : ℕ :=
  if 0 < x ∧ x < 4 then 200 * x
  else if x ≥ 4 then 100 * x + 400
  else 0

def agency_b_cost (x : ℕ) : ℕ :=
  140 * x

theorem compare_charges (x : ℕ) :
  (agency_a_cost x < agency_b_cost x -> x > 10) ∧
  (agency_a_cost x = agency_b_cost x -> x = 10) ∧
  (agency_a_cost x > agency_b_cost x -> x < 10) :=
by
  sorry

end compare_charges_l966_96674


namespace total_tickets_l966_96608

theorem total_tickets (O B : ℕ) (h1 : 12 * O + 8 * B = 3320) (h2 : B = O + 90) : O + B = 350 := by
  sorry

end total_tickets_l966_96608


namespace find_base_l966_96693

theorem find_base (b : ℕ) : (b^3 ≤ 64 ∧ 64 < b^4) ↔ b = 4 := 
by
  sorry

end find_base_l966_96693


namespace find_speed_of_stream_l966_96690

def distance : ℝ := 24
def total_time : ℝ := 5
def rowing_speed : ℝ := 10

def speed_of_stream (v : ℝ) : Prop :=
  distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time

theorem find_speed_of_stream : ∃ v : ℝ, speed_of_stream v ∧ v = 2 :=
by
  exists 2
  unfold speed_of_stream
  simp
  sorry -- This would be the proof part which is not required here

end find_speed_of_stream_l966_96690


namespace color_coat_drying_time_l966_96661

theorem color_coat_drying_time : ∀ (x : ℕ), 2 + 2 * x + 5 = 13 → x = 3 :=
by
  intro x
  intro h
  sorry

end color_coat_drying_time_l966_96661


namespace min_segments_for_7_points_l966_96649

theorem min_segments_for_7_points (points : Fin 7 → ℝ × ℝ) : 
  ∃ (segments : Finset (Fin 7 × Fin 7)), 
    (∀ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a → (a, b) ∈ segments ∨ (b, c) ∈ segments ∨ (c, a) ∈ segments) ∧
    segments.card = 9 :=
sorry

end min_segments_for_7_points_l966_96649


namespace polygon_interior_angles_sum_l966_96632

theorem polygon_interior_angles_sum (n : ℕ) (hn : 180 * (n - 2) = 1980) : 180 * (n + 4 - 2) = 2700 :=
by
  sorry

end polygon_interior_angles_sum_l966_96632


namespace switches_connections_l966_96638

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l966_96638


namespace minimum_value_of_x_plus_y_l966_96648

theorem minimum_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (x - 1) * (y - 1) = 1) : x + y = 4 :=
sorry

end minimum_value_of_x_plus_y_l966_96648


namespace students_in_college_l966_96630

variable (P S : ℕ)

def condition1 : Prop := S = 15 * P
def condition2 : Prop := S + P = 40000

theorem students_in_college (h1 : condition1 S P) (h2 : condition2 S P) : S = 37500 := by
  sorry

end students_in_college_l966_96630


namespace determine_x_l966_96607

theorem determine_x (x : ℝ) (h : 9 * x^2 + 2 * x^2 + 3 * x^2 / 2 = 300) : x = 2 * Real.sqrt 6 :=
by sorry

end determine_x_l966_96607


namespace find_solutions_l966_96621

-- Definitions
def is_solution (x y z n : ℕ) : Prop :=
  x^3 + y^3 + z^3 = n * (x^2) * (y^2) * (z^2)

-- Theorem statement
theorem find_solutions :
  {sol : ℕ × ℕ × ℕ × ℕ | is_solution sol.1 sol.2.1 sol.2.2.1 sol.2.2.2} =
  {(1, 1, 1, 3), (1, 2, 3, 1), (2, 1, 3, 1)} :=
by sorry

end find_solutions_l966_96621


namespace cost_B_solution_l966_96667

variable (cost_B : ℝ)

/-- The number of items of type A that can be purchased with 1000 yuan 
is equal to the number of items of type B that can be purchased with 800 yuan. -/
def items_purchased_equality (cost_B : ℝ) : Prop :=
  1000 / (cost_B + 10) = 800 / cost_B

/-- The cost of each item of type A is 10 yuan more than the cost of each item of type B. -/
def cost_difference (cost_B : ℝ) : Prop :=
  cost_B + 10 - cost_B = 10

/-- The cost of each item of type B is 40 yuan. -/
theorem cost_B_solution (h1: items_purchased_equality cost_B) (h2: cost_difference cost_B) :
  cost_B = 40 := by
sorry

end cost_B_solution_l966_96667


namespace minimum_value_inequality_maximum_value_inequality_l966_96662

noncomputable def minimum_value (x1 x2 x3 : ℝ) : ℝ :=
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5)

theorem minimum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  1 ≤ minimum_value x1 x2 x3 :=
sorry

theorem maximum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  minimum_value x1 x2 x3 ≤ 9/5 :=
sorry

end minimum_value_inequality_maximum_value_inequality_l966_96662


namespace euler_polyhedron_problem_l966_96633

theorem euler_polyhedron_problem
  (V E F : ℕ)
  (t h T H : ℕ)
  (euler_formula : V - E + F = 2)
  (faces_count : F = 30)
  (tri_hex_faces : t + h = 30)
  (edges_equation : E = (3 * t + 6 * h) / 2)
  (vertices_equation1 : V = (3 * t) / T)
  (vertices_equation2 : V = (6 * h) / H)
  (T_val : T = 1)
  (H_val : H = 2)
  (t_val : t = 10)
  (h_val : h = 20)
  (edges_val : E = 75)
  (vertices_val : V = 60) :
  100 * H + 10 * T + V = 270 :=
by
  sorry

end euler_polyhedron_problem_l966_96633


namespace annalise_spending_l966_96673

theorem annalise_spending
  (n_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (cost_per_tissue : ℝ)
  (h1 : n_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : tissues_per_pack = 100)
  (h4 : cost_per_tissue = 0.05) :
  n_boxes * packs_per_box * tissues_per_pack * cost_per_tissue = 1000 := 
  by
  sorry

end annalise_spending_l966_96673


namespace correct_operation_l966_96637

theorem correct_operation (a : ℝ) : a^4 / a^2 = a^2 :=
by sorry

end correct_operation_l966_96637


namespace at_least_one_ge_two_l966_96692

theorem at_least_one_ge_two (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  (a + 1/b >= 2) ∨ (b + 1/c >= 2) ∨ (c + 1/a >= 2) :=
sorry

end at_least_one_ge_two_l966_96692


namespace f_of_g_of_2_l966_96646

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_2 : f (g 2) = 14 :=
by 
  sorry

end f_of_g_of_2_l966_96646


namespace license_plate_combinations_l966_96639

theorem license_plate_combinations :
  let letters := 26
  let two_other_letters := Nat.choose 25 2
  let repeated_positions := Nat.choose 4 2
  let arrange_two_letters := 2
  let first_digit_choices := 10
  let second_digit_choices := 9
  letters * two_other_letters * repeated_positions * arrange_two_letters * first_digit_choices * second_digit_choices = 8424000 :=
  sorry

end license_plate_combinations_l966_96639


namespace cuboid_first_edge_length_l966_96611

theorem cuboid_first_edge_length (x : ℝ) (hx : 180 = x * 5 * 6) : x = 6 :=
by
  sorry

end cuboid_first_edge_length_l966_96611


namespace find_angle_A_l966_96672

variable {A B C a b c : ℝ}
variable {triangle_ABC : Prop}

theorem find_angle_A
  (h1 : a^2 + c^2 = b^2 + 2 * a * c * Real.cos C)
  (h2 : a = 2 * b * Real.sin A)
  (h3 : Real.cos B = Real.cos C)
  (h_triangle_angles : triangle_ABC) : A = 2 * Real.pi / 3 := 
by
  sorry

end find_angle_A_l966_96672


namespace alex_average_speed_l966_96677

def total_distance : ℕ := 48
def biking_time : ℕ := 6

theorem alex_average_speed : (total_distance / biking_time) = 8 := 
by
  sorry

end alex_average_speed_l966_96677


namespace silas_payment_ratio_l966_96634

theorem silas_payment_ratio (total_bill : ℕ) (tip_rate : ℝ) (friend_payment : ℕ) (S : ℕ) :
  total_bill = 150 →
  tip_rate = 0.10 →
  friend_payment = 18 →
  (S + 5 * friend_payment = total_bill + total_bill * tip_rate) →
  (S : ℝ) / total_bill = 1 / 2 :=
by
  intros h_total_bill h_tip_rate h_friend_payment h_budget_eq
  sorry

end silas_payment_ratio_l966_96634


namespace contrapositive_equivalence_l966_96603

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1)) ↔ (∀ x : ℝ, (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0)) :=
by {
  sorry
}

end contrapositive_equivalence_l966_96603


namespace volume_of_cut_out_box_l966_96683

theorem volume_of_cut_out_box (x : ℝ) : 
  let l := 16
  let w := 12
  let new_l := l - 2 * x
  let new_w := w - 2 * x
  let height := x
  let V := new_l * new_w * height
  V = 4 * x^3 - 56 * x^2 + 192 * x :=
by
  sorry

end volume_of_cut_out_box_l966_96683


namespace division_correct_result_l966_96653

theorem division_correct_result (x : ℝ) (h : 8 * x = 56) : 42 / x = 6 := by
  sorry

end division_correct_result_l966_96653


namespace coffee_blend_price_l966_96666

theorem coffee_blend_price (x : ℝ) : 
  (9 * 8 + x * 12) / 20 = 8.4 → x = 8 :=
by
  intro h
  sorry

end coffee_blend_price_l966_96666


namespace football_team_practice_hours_l966_96696

-- Definitions for each day's practice adjusted for weather events
def monday_hours : ℕ := 4
def tuesday_hours : ℕ := 5 - 1
def wednesday_hours : ℕ := 0
def thursday_hours : ℕ := 5
def friday_hours : ℕ := 3 + 2
def saturday_hours : ℕ := 4
def sunday_hours : ℕ := 0

-- Total practice hours calculation
def total_practice_hours : ℕ := 
  monday_hours + tuesday_hours + wednesday_hours + 
  thursday_hours + friday_hours + saturday_hours + 
  sunday_hours

-- Statement to prove
theorem football_team_practice_hours : total_practice_hours = 22 := by
  sorry

end football_team_practice_hours_l966_96696


namespace product_of_mixed_numbers_l966_96669

theorem product_of_mixed_numbers :
  let fraction1 := (13 : ℚ) / 6
  let fraction2 := (29 : ℚ) / 9
  (fraction1 * fraction2) = 377 / 54 := 
by
  sorry

end product_of_mixed_numbers_l966_96669


namespace big_bottles_sold_percentage_l966_96631

-- Definitions based on conditions
def small_bottles_initial : ℕ := 5000
def big_bottles_initial : ℕ := 12000
def small_bottles_sold_percentage : ℝ := 0.15
def total_bottles_remaining : ℕ := 14090

-- Question in Lean 4
theorem big_bottles_sold_percentage : 
  (12000 - (12000 * x / 100) + 5000 - (5000 * 15 / 100)) = 14090 → x = 18 :=
by
  intros h
  sorry

end big_bottles_sold_percentage_l966_96631


namespace simplify_expression_l966_96699

theorem simplify_expression :
  (↑(Real.sqrt 648) / ↑(Real.sqrt 81) - ↑(Real.sqrt 245) / ↑(Real.sqrt 49)) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  -- proof omitted
  sorry

end simplify_expression_l966_96699


namespace problem1_problem2_l966_96655

-- Definition and proof statement for Problem 1
theorem problem1 (y : ℝ) : 
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 := 
by sorry

-- Definition and proof statement for Problem 2
theorem problem2 (x : ℝ) (h : x ≠ -1) :
  (1 + 2 / (x + 1)) / ((x^2 + 6 * x + 9) / (x + 1)) = 1 / (x + 3) :=
by sorry

end problem1_problem2_l966_96655


namespace books_left_over_after_repacking_l966_96623

def initial_boxes : ℕ := 1430
def books_per_initial_box : ℕ := 42
def weight_per_book : ℕ := 200 -- in grams
def books_per_new_box : ℕ := 45
def max_weight_per_new_box : ℕ := 9000 -- in grams (9 kg)

def total_books : ℕ := initial_boxes * books_per_initial_box

theorem books_left_over_after_repacking :
  total_books % books_per_new_box = 30 :=
by
  -- Proof goes here
  sorry

end books_left_over_after_repacking_l966_96623


namespace isosceles_triangle_properties_l966_96684

/--
  An isosceles triangle has a base of 6 units and legs of 5 units each.
  Prove:
  1. The area of the triangle is 12 square units.
  2. The radius of the inscribed circle is 1.5 units.
-/
theorem isosceles_triangle_properties (base : ℝ) (legs : ℝ) 
  (h_base : base = 6) (h_legs : legs = 5) : 
  ∃ (area : ℝ) (inradius : ℝ), 
  area = 12 ∧ inradius = 1.5 
  :=
by
  sorry

end isosceles_triangle_properties_l966_96684


namespace no_such_f_exists_l966_96612

theorem no_such_f_exists (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
  (h2 : ∀ x y, 0 < x → 0 < y → f x ^ 2 ≥ f (x + y) * (f x + y)) : false :=
sorry

end no_such_f_exists_l966_96612


namespace relationship_between_sums_l966_96664

-- Conditions: four distinct positive integers
variables {a b c d : ℕ}
-- additional conditions: positive integers
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Condition: a is the largest and d is the smallest
variables (a_largest : a > b ∧ a > c ∧ a > d)
variables (d_smallest : d < b ∧ d < c ∧ d < a)

-- Condition: a / b = c / d
variables (ratio_condition : a * d = b * c)

theorem relationship_between_sums :
  a + d > b + c :=
sorry

end relationship_between_sums_l966_96664


namespace sequence_inequality_l966_96619

theorem sequence_inequality (a : ℕ → ℕ) 
  (h_nonneg : ∀ n, 0 ≤ a n)
  (h_additive : ∀ m n, a (n + m) ≤ a n + a m) 
  (N n : ℕ) 
  (h_N_ge_n : N ≥ n) : 
  a n + a N ≤ n * a 1 + N / n * a n :=
sorry

end sequence_inequality_l966_96619


namespace number_of_performances_l966_96602

theorem number_of_performances (hanna_songs : ℕ) (mary_songs : ℕ) (alina_songs : ℕ) (tina_songs : ℕ)
    (hanna_cond : hanna_songs = 4)
    (mary_cond : mary_songs = 7)
    (alina_cond : 4 < alina_songs ∧ alina_songs < 7)
    (tina_cond : 4 < tina_songs ∧ tina_songs < 7) :
    ((hanna_songs + mary_songs + alina_songs + tina_songs) / 3) = 7 :=
by
  -- proof steps would go here
  sorry

end number_of_performances_l966_96602


namespace G_greater_F_l966_96615

theorem G_greater_F (x : ℝ) : 
  let F := 2*x^2 - 3*x - 2
  let G := 3*x^2 - 7*x + 5
  G > F := 
sorry

end G_greater_F_l966_96615


namespace how_many_pints_did_Annie_pick_l966_96626

theorem how_many_pints_did_Annie_pick (x : ℕ) (h1 : Kathryn = x + 2)
                                      (h2 : Ben = Kathryn - 3)
                                      (h3 : x + Kathryn + Ben = 25) : x = 8 :=
  sorry

end how_many_pints_did_Annie_pick_l966_96626


namespace triangle_interior_angle_l966_96691

-- Define the given values and equations
variables (x : ℝ) 
def arc_DE := x + 80
def arc_EF := 2 * x + 30
def arc_FD := 3 * x - 25

-- The main proof statement
theorem triangle_interior_angle :
  arc_DE x + arc_EF x + arc_FD x = 360 →
  0.5 * (arc_EF x) = 60.83 :=
by sorry

end triangle_interior_angle_l966_96691


namespace cost_of_article_l966_96678

-- Conditions as Lean definitions
def price_1 : ℝ := 340
def price_2 : ℝ := 350
def price_diff : ℝ := price_2 - price_1 -- Rs. 10
def gain_percent_increase : ℝ := 0.04

-- Question: What is the cost of the article?
-- Answer: Rs. 90

theorem cost_of_article : ∃ C : ℝ, 
  price_diff = gain_percent_increase * (price_1 - C) ∧ C = 90 := 
sorry

end cost_of_article_l966_96678


namespace minimum_greeting_pairs_l966_96670

def minimum_mutual_greetings (n: ℕ) (g: ℕ) : ℕ :=
  (n * g - (n * (n - 1)) / 2)

theorem minimum_greeting_pairs :
  minimum_mutual_greetings 400 200 = 200 :=
by 
  sorry

end minimum_greeting_pairs_l966_96670


namespace new_price_after_increase_l966_96652

def original_price (y : ℝ) : Prop := 2 * y = 540

theorem new_price_after_increase (y : ℝ) (h : original_price y) : 1.3 * y = 351 :=
by sorry

end new_price_after_increase_l966_96652


namespace new_year_markup_l966_96618

variable (C : ℝ) -- original cost of the turtleneck sweater
variable (N : ℝ) -- New Year season markup in decimal form
variable (final_price : ℝ) -- final price in February

-- Conditions
def initial_markup (C : ℝ) := 1.20 * C
def after_new_year_markup (C : ℝ) (N : ℝ) := (1 + N) * initial_markup C
def discount_in_february (C : ℝ) (N : ℝ) := 0.94 * after_new_year_markup C N
def profit_in_february (C : ℝ) := 1.41 * C

-- Mathematically equivalent proof problem (statement only)
theorem new_year_markup :
  ∀ C : ℝ, ∀ N : ℝ,
    discount_in_february C N = profit_in_february C →
    N = 0.5 :=
by
  sorry

end new_year_markup_l966_96618


namespace allen_reading_days_l966_96641

theorem allen_reading_days (pages_per_day : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 10) (h2 : total_pages = 120) : 
  (total_pages / pages_per_day) = 12 := by
  sorry

end allen_reading_days_l966_96641


namespace charlotte_total_dog_walking_time_l966_96679

def poodles_monday : ℕ := 4
def chihuahuas_monday : ℕ := 2
def poodles_tuesday : ℕ := 4
def chihuahuas_tuesday : ℕ := 2
def labradors_wednesday : ℕ := 4

def time_poodle : ℕ := 2
def time_chihuahua : ℕ := 1
def time_labrador : ℕ := 3

def total_time_monday : ℕ := poodles_monday * time_poodle + chihuahuas_monday * time_chihuahua
def total_time_tuesday : ℕ := poodles_tuesday * time_poodle + chihuahuas_tuesday * time_chihuahua
def total_time_wednesday : ℕ := labradors_wednesday * time_labrador

def total_time_week : ℕ := total_time_monday + total_time_tuesday + total_time_wednesday

theorem charlotte_total_dog_walking_time : total_time_week = 32 := by
  -- Lean allows us to state the theorem without proving it.
  sorry

end charlotte_total_dog_walking_time_l966_96679


namespace inequality_solution_l966_96694

theorem inequality_solution (x : ℝ) (h : ∀ (a b : ℝ) (ha : 0 < a) (hb : 0 < b), x^2 + x < a / b + b / a) : x ∈ Set.Ioo (-2 : ℝ) 1 := 
sorry

end inequality_solution_l966_96694


namespace line_parameterization_l966_96660

theorem line_parameterization (s m : ℝ) :
  (∃ t : ℝ, ∀ x y : ℝ, (x = s + 2 * t ∧ y = 3 + m * t) ↔ y = 5 * x - 7) →
  s = 2 ∧ m = 10 :=
by
  intro h_conditions
  sorry

end line_parameterization_l966_96660


namespace annual_income_calculation_l966_96647

noncomputable def annual_income (investment : ℝ) (price_per_share : ℝ) (dividend_rate : ℝ) (face_value : ℝ) : ℝ :=
  let number_of_shares := investment / price_per_share
  number_of_shares * face_value * dividend_rate

theorem annual_income_calculation :
  annual_income 4455 8.25 0.12 10 = 648 :=
by
  sorry

end annual_income_calculation_l966_96647


namespace contracting_schemes_l966_96635

theorem contracting_schemes :
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  (Nat.choose total_projects a_contracts) *
  (Nat.choose (total_projects - a_contracts) b_contracts) *
  (Nat.choose ((total_projects - a_contracts) - b_contracts) c_contracts) = 60 :=
by
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  sorry

end contracting_schemes_l966_96635


namespace ab_value_l966_96624

theorem ab_value (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by {
  sorry
}

end ab_value_l966_96624


namespace locus_equation_rectangle_perimeter_greater_l966_96665

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end locus_equation_rectangle_perimeter_greater_l966_96665


namespace robert_arrival_time_l966_96680

def arrival_time (T : ℕ) : Prop :=
  ∃ D : ℕ, D = 10 * (12 - T) ∧ D = 15 * (13 - T)

theorem robert_arrival_time : arrival_time 15 :=
by
  sorry

end robert_arrival_time_l966_96680


namespace solve_x_for_fraction_l966_96616

theorem solve_x_for_fraction :
  ∃ x : ℝ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 14.6 :=
by
  sorry

end solve_x_for_fraction_l966_96616


namespace quadratic_conversion_l966_96628

def quadratic_to_vertex_form (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 1

theorem quadratic_conversion :
  (∀ x : ℝ, quadratic_to_vertex_form x = 2 * (x - 2)^2 - 9) :=
by
  sorry

end quadratic_conversion_l966_96628


namespace graphene_scientific_notation_l966_96613

theorem graphene_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (0.00000000034 : ℝ) = a * 10^n ∧ a = 3.4 ∧ n = -10 :=
sorry

end graphene_scientific_notation_l966_96613
