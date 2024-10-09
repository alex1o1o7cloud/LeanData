import Mathlib

namespace ten_yuan_notes_count_l1071_107177

theorem ten_yuan_notes_count (total_notes : ℕ) (total_change : ℕ) (item_cost : ℕ) (change_given : ℕ → ℕ → ℕ) (is_ten_yuan_notes : ℕ → Prop) :
    total_notes = 16 →
    total_change = 95 →
    item_cost = 5 →
    change_given 10 5 = total_change →
    (∃ x y : ℕ, x + y = total_notes ∧ 10 * x + 5 * y = total_change ∧ is_ten_yuan_notes x) → is_ten_yuan_notes 3 :=
by
  sorry

end ten_yuan_notes_count_l1071_107177


namespace exponent_rule_example_l1071_107129

theorem exponent_rule_example : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end exponent_rule_example_l1071_107129


namespace movies_in_first_box_l1071_107106

theorem movies_in_first_box (x : ℕ) 
  (cost_first : ℕ) (cost_second : ℕ) 
  (num_second : ℕ) (avg_price : ℕ)
  (h_cost_first : cost_first = 2)
  (h_cost_second : cost_second = 5)
  (h_num_second : num_second = 5)
  (h_avg_price : avg_price = 3)
  (h_total_eq : cost_first * x + cost_second * num_second = avg_price * (x + num_second)) :
  x = 5 :=
by
  sorry

end movies_in_first_box_l1071_107106


namespace find_congruence_l1071_107166

theorem find_congruence (x : ℤ) (h : 4 * x + 9 ≡ 3 [ZMOD 17]) : 3 * x + 12 ≡ 16 [ZMOD 17] :=
sorry

end find_congruence_l1071_107166


namespace average_points_per_player_l1071_107157

theorem average_points_per_player (Lefty_points Righty_points OtherTeammate_points : ℕ)
  (hL : Lefty_points = 20)
  (hR : Righty_points = Lefty_points / 2)
  (hO : OtherTeammate_points = 6 * Righty_points) :
  (Lefty_points + Righty_points + OtherTeammate_points) / 3 = 30 :=
by
  sorry

end average_points_per_player_l1071_107157


namespace molecular_weight_of_compound_l1071_107127

def hydrogen_atomic_weight : ℝ := 1.008
def chromium_atomic_weight : ℝ := 51.996
def oxygen_atomic_weight : ℝ := 15.999

def compound_molecular_weight (h_atoms : ℕ) (cr_atoms : ℕ) (o_atoms : ℕ) : ℝ :=
  h_atoms * hydrogen_atomic_weight + cr_atoms * chromium_atomic_weight + o_atoms * oxygen_atomic_weight

theorem molecular_weight_of_compound :
  compound_molecular_weight 2 1 4 = 118.008 :=
by
  sorry

end molecular_weight_of_compound_l1071_107127


namespace minimum_shift_value_l1071_107170

theorem minimum_shift_value
    (m : ℝ) 
    (h1 : m > 0) :
    (∃ (k : ℤ), m = k * π - π / 3 ∧ k > 0) → (m = (2 * π) / 3) :=
sorry

end minimum_shift_value_l1071_107170


namespace max_value_of_symmetric_f_l1071_107101

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end max_value_of_symmetric_f_l1071_107101


namespace harmonic_mean_closest_to_2_l1071_107167

theorem harmonic_mean_closest_to_2 (a : ℝ) (b : ℝ) (h₁ : a = 1) (h₂ : b = 4032) : 
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  rw [h₁, h₂]
  -- The rest of the proof follows from here, skipped with sorry
  sorry

end harmonic_mean_closest_to_2_l1071_107167


namespace trajectory_of_center_l1071_107105

-- Define the fixed circle C as x^2 + (y + 3)^2 = 1
def fixed_circle (p : ℝ × ℝ) : Prop :=
  (p.1)^2 + (p.2 + 3)^2 = 1

-- Define the line y = 2
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2

-- The main theorem stating the trajectory of the center of circle M is x^2 = -12y
theorem trajectory_of_center :
  ∀ (M : ℝ × ℝ), 
  tangent_line M → (∃ r : ℝ, fixed_circle (M.1, M.2 - r) ∧ r > 0) →
  (M.1)^2 = -12 * M.2 :=
sorry

end trajectory_of_center_l1071_107105


namespace margarita_jumps_farther_l1071_107151

-- Definitions for conditions
def RiccianaTotalDistance := 24
def RiccianaRunDistance := 20
def RiccianaJumpDistance := 4

def MargaritaRunDistance := 18
def MargaritaJumpDistance := 2 * RiccianaJumpDistance - 1

-- Theorem statement to prove the question
theorem margarita_jumps_farther :
  (MargaritaRunDistance + MargaritaJumpDistance) - RiccianaTotalDistance = 1 :=
by
  -- Proof will be written here
  sorry

end margarita_jumps_farther_l1071_107151


namespace quadratic_no_real_roots_l1071_107196

open Real

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (hpq : p ≠ q)
  (hpositive_p : 0 < p)
  (hpositive_q : 0 < q)
  (hpositive_a : 0 < a)
  (hpositive_b : 0 < b)
  (hpositive_c : 0 < c)
  (h_geo_sequence : a^2 = p * q)
  (h_ari_sequence : b + c = p + q) :
  (a^2 - b * c) < 0 :=
by
  sorry

end quadratic_no_real_roots_l1071_107196


namespace power_function_properties_l1071_107140

theorem power_function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ a) (h2 : f 2 = Real.sqrt 2) : 
  a = 1 / 2 ∧ ∀ x, 0 ≤ x → f x ≤ f (x + 1) :=
by
  sorry

end power_function_properties_l1071_107140


namespace quadrilateral_area_l1071_107184

theorem quadrilateral_area 
  (AB BC DC : ℝ)
  (hAB_perp_BC : true)
  (hDC_perp_BC : true)
  (hAB_eq : AB = 8)
  (hDC_eq : DC = 3)
  (hBC_eq : BC = 10) : 
  (1 / 2 * (AB + DC) * BC = 55) :=
by 
  sorry

end quadrilateral_area_l1071_107184


namespace min_cards_to_guarantee_four_same_suit_l1071_107194

theorem min_cards_to_guarantee_four_same_suit (n : ℕ) (suits : Fin n) (cards_per_suit : ℕ) (total_cards : ℕ)
  (h1 : n = 4) (h2 : cards_per_suit = 13) : total_cards ≥ 13 :=
by
  sorry

end min_cards_to_guarantee_four_same_suit_l1071_107194


namespace cab_drivers_income_on_third_day_l1071_107150

theorem cab_drivers_income_on_third_day
  (day1 day2 day4 day5 avg_income n_days : ℝ)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 250)
  (h_day4 : day4 = 400)
  (h_day5 : day5 = 800)
  (h_avg_income : avg_income = 500)
  (h_n_days : n_days = 5) :
  ∃ day3 : ℝ, (day1 + day2 + day3 + day4 + day5) / n_days = avg_income ∧ day3 = 450 :=
by
  sorry

end cab_drivers_income_on_third_day_l1071_107150


namespace ratio_of_average_speeds_l1071_107197

-- Define the conditions as constants
def distance_ab : ℕ := 510
def distance_ac : ℕ := 300
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4

-- Define the speeds
def speed_eddy := distance_ab / time_eddy
def speed_freddy := distance_ac / time_freddy

-- The ratio calculation and verification function
def speed_ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- Define the main theorem to be proved
theorem ratio_of_average_speeds : speed_ratio speed_eddy speed_freddy = (34, 15) := by
  sorry

end ratio_of_average_speeds_l1071_107197


namespace domain_of_f1_x2_l1071_107178

theorem domain_of_f1_x2 (f : ℝ → ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → ∃ y, y = f x) → 
  (∀ x, -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → ∃ y, y = f (1 - x^2)) :=
by
  sorry

end domain_of_f1_x2_l1071_107178


namespace tan_sum_identity_l1071_107169

theorem tan_sum_identity : (1 + Real.tan (Real.pi / 180)) * (1 + Real.tan (44 * Real.pi / 180)) = 2 := 
by sorry

end tan_sum_identity_l1071_107169


namespace expenditure_recorded_neg_20_l1071_107128

-- Define the condition where income of 60 yuan is recorded as +60 yuan
def income_recorded (income : ℤ) : ℤ :=
  income

-- Define what expenditure is given the condition
def expenditure_recorded (expenditure : ℤ) : ℤ :=
  -expenditure

-- Prove that an expenditure of 20 yuan is recorded as -20 yuan
theorem expenditure_recorded_neg_20 :
  expenditure_recorded 20 = -20 :=
by
  sorry

end expenditure_recorded_neg_20_l1071_107128


namespace unobstructed_sight_l1071_107156

-- Define the curve C as y = 2x^2
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define point A and point B
def pointA : ℝ × ℝ := (0, -2)
def pointB (a : ℝ) : ℝ × ℝ := (3, a)

-- Statement of the problem
theorem unobstructed_sight {a : ℝ} (h : ∀ x : ℝ, 0 ≤ x → x ≤ 3 → 4 * x - 2 ≥ 2 * x^2) : a < 10 :=
sorry

end unobstructed_sight_l1071_107156


namespace technician_percent_round_trip_l1071_107188

noncomputable def round_trip_percentage_completed (D : ℝ) : ℝ :=
  let total_round_trip := 2 * D
  let distance_completed := D + 0.10 * D
  (distance_completed / total_round_trip) * 100

theorem technician_percent_round_trip (D : ℝ) (h : D > 0) : 
  round_trip_percentage_completed D = 55 := 
by 
  sorry

end technician_percent_round_trip_l1071_107188


namespace minimize_x_expr_minimized_l1071_107133

noncomputable def minimize_x_expr (x : ℝ) : ℝ :=
  x + 4 / (x + 1)

theorem minimize_x_expr_minimized 
  (hx : x > -1) 
  : x = 1 ↔ minimize_x_expr x = minimize_x_expr 1 :=
by
  sorry

end minimize_x_expr_minimized_l1071_107133


namespace rattlesnakes_count_l1071_107199

-- Definitions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def rattlesnakes : ℕ := total_snakes - (boa_constrictors + pythons)

-- Theorem to prove
theorem rattlesnakes_count : rattlesnakes = 40 := by
  -- provide proof here
  sorry

end rattlesnakes_count_l1071_107199


namespace prob_correct_l1071_107102

noncomputable def prob_train_there_when_sam_arrives : ℚ :=
  let total_area := (60 : ℚ) * 60
  let triangle_area := (1 / 2 : ℚ) * 15 * 15
  let parallelogram_area := (30 : ℚ) * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem prob_correct : prob_train_there_when_sam_arrives = 25 / 160 :=
  sorry

end prob_correct_l1071_107102


namespace video_duration_correct_l1071_107108

/-
Define the conditions as given:
1. Vasya's time from home to school
2. Petya's time from school to home
3. Meeting conditions
-/

-- Define the times for Vasya and Petya
def vasya_time : ℕ := 8
def petya_time : ℕ := 5

-- Define the total video duration when correctly merged
def video_duration : ℕ := 5

-- State the theorem to be proved in Lean:
theorem video_duration_correct : vasya_time = 8 → petya_time = 5 → video_duration = 5 :=
by
  intros h1 h2
  exact sorry

end video_duration_correct_l1071_107108


namespace remainder_when_divided_by_44_l1071_107135

theorem remainder_when_divided_by_44 (N Q R : ℕ) :
  (N = 44 * 432 + R) ∧ (N = 39 * Q + 15) → R = 0 :=
by
  sorry

end remainder_when_divided_by_44_l1071_107135


namespace sequence_tuple_l1071_107141

/-- Prove the unique solution to the system of equations derived from the sequence pattern. -/
theorem sequence_tuple (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 7) : (x, y) = (8, 1) :=
by
  sorry

end sequence_tuple_l1071_107141


namespace probability_A_seventh_week_l1071_107115

/-
Conditions:
1. There are four different passwords: A, B, C, and D.
2. Each week, one of these passwords is used.
3. Each week, the password is chosen at random and equally likely from the three passwords that were not used in the previous week.
4. Password A is used in the first week.

Goal:
Prove that the probability that password A will be used in the seventh week is 61/243.
-/

def prob_password_A_in_seventh_week : ℚ :=
  let Pk (k : ℕ) : ℚ := 
    if k = 1 then 1
    else if k >= 2 then ((-1 / 3)^(k - 1) * (3 / 4) + 1 / 4) else 0
  Pk 7

theorem probability_A_seventh_week : prob_password_A_in_seventh_week = 61 / 243 := by
  sorry

end probability_A_seventh_week_l1071_107115


namespace angle_measure_l1071_107107

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l1071_107107


namespace train_cross_pole_in_time_l1071_107148

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_pole_in_time :
  time_to_cross_pole 100 126 = 100 / (126 * (1000 / 3600)) :=
by
  -- this will unfold the calculation step-by-step
  unfold time_to_cross_pole
  sorry

end train_cross_pole_in_time_l1071_107148


namespace cuboid_surface_area_l1071_107182

-- Define the given conditions
def cuboid (a b c : ℝ) := 2 * (a + b + c)

-- Given areas of distinct sides
def area_face_1 : ℝ := 4
def area_face_2 : ℝ := 3
def area_face_3 : ℝ := 6

-- Prove the total surface area of the cuboid
theorem cuboid_surface_area : cuboid area_face_1 area_face_2 area_face_3 = 26 :=
by
  sorry

end cuboid_surface_area_l1071_107182


namespace basketball_shots_l1071_107132

variable (x y : ℕ)

theorem basketball_shots : 3 * x + 2 * y = 26 ∧ x + y = 11 → x = 4 :=
by
  intros h
  sorry

end basketball_shots_l1071_107132


namespace least_number_of_marbles_l1071_107145

def divisible_by (n : ℕ) (d : ℕ) : Prop := n % d = 0

theorem least_number_of_marbles 
  (n : ℕ)
  (h3 : divisible_by n 3)
  (h4 : divisible_by n 4)
  (h5 : divisible_by n 5)
  (h7 : divisible_by n 7)
  (h8 : divisible_by n 8) :
  n = 840 :=
sorry

end least_number_of_marbles_l1071_107145


namespace find_weight_l1071_107136

-- Define the weight of each box before taking out 20 kg as W
variable (W : ℚ)

-- Define the condition given in the problem
def condition : Prop := 7 * (W - 20) = 3 * W

-- The proof goal is to prove W = 35 under the given condition
theorem find_weight (h : condition W) : W = 35 := by
  sorry

end find_weight_l1071_107136


namespace bryce_received_raisins_l1071_107149

theorem bryce_received_raisins
  (C B : ℕ)
  (h1 : B = C + 8)
  (h2 : C = B / 3) :
  B = 12 :=
by sorry

end bryce_received_raisins_l1071_107149


namespace celina_paid_multiple_of_diego_l1071_107103

theorem celina_paid_multiple_of_diego
  (D : ℕ) (x : ℕ)
  (h_total : (x + 1) * D + 1000 = 50000)
  (h_positive : D > 0) :
  x = 48 :=
sorry

end celina_paid_multiple_of_diego_l1071_107103


namespace find_f_2_l1071_107168

-- Condition: f(x + 1) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Statement to prove
theorem find_f_2 : f 2 = -1 := by
  sorry

end find_f_2_l1071_107168


namespace rectangular_plot_breadth_l1071_107113

theorem rectangular_plot_breadth (b l : ℝ) (A : ℝ)
  (h1 : l = 3 * b)
  (h2 : A = l * b)
  (h3 : A = 2700) : b = 30 :=
by sorry

end rectangular_plot_breadth_l1071_107113


namespace max_b_in_box_l1071_107193

theorem max_b_in_box (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 := 
by
  sorry

end max_b_in_box_l1071_107193


namespace upstream_speed_l1071_107154

theorem upstream_speed (Vm Vdownstream Vupstream Vs : ℝ) 
  (h1 : Vm = 50) 
  (h2 : Vdownstream = 55) 
  (h3 : Vdownstream = Vm + Vs) 
  (h4 : Vupstream = Vm - Vs) : 
  Vupstream = 45 :=
by
  sorry

end upstream_speed_l1071_107154


namespace space_diagonal_of_prism_l1071_107189

theorem space_diagonal_of_prism (l w h : ℝ) (hl : l = 2) (hw : w = 3) (hh : h = 4) :
  (l ^ 2 + w ^ 2 + h ^ 2).sqrt = Real.sqrt 29 :=
by
  rw [hl, hw, hh]
  sorry

end space_diagonal_of_prism_l1071_107189


namespace arithmetic_sequence_general_term_l1071_107190

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_increasing : d > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = a 2 ^ 2 - 4) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l1071_107190


namespace floor_sum_proof_l1071_107142

noncomputable def floor_sum (x y z w : ℝ) : ℝ :=
  x + y + z + w

theorem floor_sum_proof
  (x y z w : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hw_pos : 0 < w)
  (h1 : x^2 + y^2 = 2010)
  (h2 : z^2 + w^2 = 2010)
  (h3 : x * z = 1008)
  (h4 : y * w = 1008) :
  ⌊floor_sum x y z w⌋ = 126 :=
by
  sorry

end floor_sum_proof_l1071_107142


namespace smallest_nat_satisfies_conditions_l1071_107120

theorem smallest_nat_satisfies_conditions : 
  ∃ x : ℕ, (∃ m : ℤ, x + 13 = 5 * m) ∧ (∃ n : ℤ, x - 13 = 6 * n) ∧ x = 37 := by
  sorry

end smallest_nat_satisfies_conditions_l1071_107120


namespace suff_and_necc_l1071_107137

variable (x : ℝ)

def A : Set ℝ := { x | x > 2 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem suff_and_necc : (x ∈ (A ∪ B)) ↔ (x ∈ C) := by
  sorry

end suff_and_necc_l1071_107137


namespace jelly_ratio_l1071_107123

theorem jelly_ratio (G S R P : ℕ) 
  (h1 : G = 2 * S)
  (h2 : R = 2 * P) 
  (h3 : P = 6) 
  (h4 : S = 18) : 
  R / G = 1 / 3 := by
  sorry

end jelly_ratio_l1071_107123


namespace sum_of_not_visible_faces_l1071_107187

-- Define the sum of the numbers on the faces of one die
def die_sum : ℕ := 21

-- List of visible numbers on the dice
def visible_faces_sum : ℕ := 4 + 3 + 2 + 5 + 1 + 3 + 1

-- Define the total sum of the numbers on the faces of three dice
def total_sum : ℕ := die_sum * 3

-- Statement to prove the sum of not-visible faces equals 44
theorem sum_of_not_visible_faces : 
  total_sum - visible_faces_sum = 44 :=
sorry

end sum_of_not_visible_faces_l1071_107187


namespace observations_number_l1071_107180

theorem observations_number 
  (mean : ℚ)
  (wrong_obs corrected_obs : ℚ)
  (new_mean : ℚ)
  (n : ℚ)
  (initial_mean : mean = 36)
  (wrong_obs_taken : wrong_obs = 23)
  (corrected_obs_value : corrected_obs = 34)
  (corrected_mean : new_mean = 36.5) :
  (n * mean + (corrected_obs - wrong_obs) = n * new_mean) → 
  n = 22 :=
by
  sorry

end observations_number_l1071_107180


namespace rent_increase_l1071_107100

theorem rent_increase (monthly_rent_first_3_years : ℕ) (months_first_3_years : ℕ) 
  (total_paid : ℕ) (total_years : ℕ) (months_in_a_year : ℕ) (new_monthly_rent : ℕ) :
  monthly_rent_first_3_years * (months_in_a_year * 3) + new_monthly_rent * (months_in_a_year * (total_years - 3)) = total_paid →
  new_monthly_rent = 350 :=
by
  intros h
  -- proof development
  sorry

end rent_increase_l1071_107100


namespace num_digits_c_l1071_107118

theorem num_digits_c (a b c : ℕ) (ha : 10 ^ 2010 ≤ a ∧ a < 10 ^ 2011)
  (hb : 10 ^ 2011 ≤ b ∧ b < 10 ^ 2012)
  (h1 : a < b) (h2 : b < c)
  (div1 : ∃ k : ℕ, b + a = k * (b - a))
  (div2 : ∃ m : ℕ, c + b = m * (c - b)) :
  10 ^ 4 ≤ c ∧ c < 10 ^ 5 :=
sorry

end num_digits_c_l1071_107118


namespace infinite_rational_points_on_circle_l1071_107161

noncomputable def exists_infinitely_many_rational_points_on_circle : Prop :=
  ∃ f : ℚ → ℚ × ℚ, (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                   (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m)

theorem infinite_rational_points_on_circle :
  ∃ (f : ℚ → ℚ × ℚ), (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                     (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m) := sorry

end infinite_rational_points_on_circle_l1071_107161


namespace find_value_of_sum_of_squares_l1071_107143

theorem find_value_of_sum_of_squares
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6 / 5 := by
  sorry

end find_value_of_sum_of_squares_l1071_107143


namespace problem_l1071_107111

theorem problem (w x y z : ℕ) (h : 3^w * 5^x * 7^y * 11^z = 2310) : 3 * w + 5 * x + 7 * y + 11 * z = 26 :=
sorry

end problem_l1071_107111


namespace solve_inequality_l1071_107162

theorem solve_inequality (k : ℝ) :
  (∀ (x : ℝ), (k + 2) * x > k + 2 → x < 1) → k = -3 :=
  by
  sorry

end solve_inequality_l1071_107162


namespace quadratic_eq_solutions_l1071_107179

theorem quadratic_eq_solutions : ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ x = 3 / 2 ∨ x = 1 :=
by
  sorry

end quadratic_eq_solutions_l1071_107179


namespace find_k_l1071_107139

theorem find_k
  (k : ℝ)
  (AB : ℝ × ℝ := (3, 1))
  (AC : ℝ × ℝ := (2, k))
  (BC : ℝ × ℝ := (2 - 3, k - 1))
  (h_perpendicular : AB.1 * BC.1 + AB.2 * BC.2 = 0)
  : k = 4 :=
sorry

end find_k_l1071_107139


namespace infinite_geometric_series_common_ratio_l1071_107117

theorem infinite_geometric_series_common_ratio 
  (a S r : ℝ) 
  (ha : a = 400) 
  (hS : S = 2500)
  (h_sum : S = a / (1 - r)) :
  r = 0.84 :=
by
  -- Proof will go here
  sorry

end infinite_geometric_series_common_ratio_l1071_107117


namespace initial_horses_to_cows_ratio_l1071_107163

theorem initial_horses_to_cows_ratio (H C : ℕ) (h₁ : (H - 15) / (C + 15) = 13 / 7) (h₂ : H - 15 = C + 45) :
  H / C = 4 / 1 := 
sorry

end initial_horses_to_cows_ratio_l1071_107163


namespace mean_score_calculation_l1071_107186

noncomputable def class_mean_score (total_students students_1 mean_score_1 students_2 mean_score_2 : ℕ) : ℚ :=
  ((students_1 * mean_score_1 + students_2 * mean_score_2) : ℚ) / total_students

theorem mean_score_calculation :
  class_mean_score 60 54 76 6 82 = 76.6 := 
sorry

end mean_score_calculation_l1071_107186


namespace find_multiple_of_larger_integer_l1071_107124

/--
The sum of two integers is 30. A certain multiple of the larger integer is 10 less than 5 times
the smaller integer. The smaller integer is 10. What is the multiple of the larger integer?
-/
theorem find_multiple_of_larger_integer
  (S L M : ℤ)
  (h1 : S + L = 30)
  (h2 : S = 10)
  (h3 : M * L = 5 * S - 10) :
  M = 2 :=
sorry

end find_multiple_of_larger_integer_l1071_107124


namespace numerator_of_fraction_l1071_107126

/-- 
Given:
1. The denominator of a fraction is 7 less than 3 times the numerator.
2. The fraction is equivalent to 2/5.
Prove that the numerator of the fraction is 14.
-/
theorem numerator_of_fraction {x : ℕ} (h : x / (3 * x - 7) = 2 / 5) : x = 14 :=
  sorry

end numerator_of_fraction_l1071_107126


namespace parabola_intersects_x_axis_l1071_107185

theorem parabola_intersects_x_axis 
  (a c : ℝ) 
  (h : ∃ x : ℝ, x = 1 ∧ (a * x^2 + x + c = 0)) : 
  a + c = -1 :=
sorry

end parabola_intersects_x_axis_l1071_107185


namespace change_in_spiders_l1071_107159

theorem change_in_spiders 
  (x a y b : ℤ) 
  (h1 : x + a = 20) 
  (h2 : y + b = 23) 
  (h3 : x - b = 5) :
  y - a = 8 := 
by
  sorry

end change_in_spiders_l1071_107159


namespace hexagon_pillar_height_l1071_107153

noncomputable def height_of_pillar_at_vertex_F (s : ℝ) (hA hB hC : ℝ) (A : ℝ × ℝ) : ℝ :=
  10

theorem hexagon_pillar_height :
  ∀ (s hA hB hC : ℝ) (A : ℝ × ℝ),
  s = 8 ∧ hA = 15 ∧ hB = 10 ∧ hC = 12 ∧ A = (3, 3 * Real.sqrt 3) →
  height_of_pillar_at_vertex_F s hA hB hC A = 10 := by
  sorry

end hexagon_pillar_height_l1071_107153


namespace number_of_people_is_ten_l1071_107191

-- Define the total number of Skittles and the number of Skittles per person.
def total_skittles : ℕ := 20
def skittles_per_person : ℕ := 2

-- Define the number of people as the total Skittles divided by the Skittles per person.
def number_of_people : ℕ := total_skittles / skittles_per_person

-- Theorem stating that the number of people is 10.
theorem number_of_people_is_ten : number_of_people = 10 := sorry

end number_of_people_is_ten_l1071_107191


namespace vasya_made_a_mistake_l1071_107158

theorem vasya_made_a_mistake (A B V G D E : ℕ)
  (h1 : A ≠ B)
  (h2 : V ≠ G)
  (h3 : (10 * A + B) * (10 * V + G) = 1000 * D + 100 * D + 10 * E + E)
  (h4 : ∀ {X Y : ℕ}, X ≠ Y → D ≠ E) :
  False :=
by
  -- Proof goes here (skipped)
  sorry

end vasya_made_a_mistake_l1071_107158


namespace flying_scotsman_more_carriages_l1071_107134

theorem flying_scotsman_more_carriages :
  ∀ (E N No F T D : ℕ),
    E = 130 →
    E = N + 20 →
    No = 100 →
    T = 460 →
    D = F - No →
    F + E + N + No = T →
    D = 20 :=
by
  intros E N No F T D hE1 hE2 hNo hT hD hSum
  sorry

end flying_scotsman_more_carriages_l1071_107134


namespace fixed_point_on_line_l1071_107116

theorem fixed_point_on_line (m x y : ℝ) (h : ∀ m : ℝ, m * x - y + 2 * m + 1 = 0) : 
  (x = -2 ∧ y = 1) :=
sorry

end fixed_point_on_line_l1071_107116


namespace minimum_distance_on_line_l1071_107122

-- Define the line as a predicate
def on_line (P : ℝ × ℝ) : Prop := P.1 - P.2 = 1

-- Define the expression to be minimized
def distance_squared (P : ℝ × ℝ) : ℝ := (P.1 - 2)^2 + (P.2 - 2)^2

theorem minimum_distance_on_line :
  ∃ P : ℝ × ℝ, on_line P ∧ distance_squared P = 1 / 2 :=
sorry

end minimum_distance_on_line_l1071_107122


namespace money_last_weeks_l1071_107144

-- Conditions
def money_from_mowing : ℕ := 14
def money_from_weeding : ℕ := 31
def weekly_spending : ℕ := 5

-- Total money made
def total_money : ℕ := money_from_mowing + money_from_weeding

-- Expected result
def expected_weeks : ℕ := 9

-- Prove the number of weeks the money will last Jerry
theorem money_last_weeks : (total_money / weekly_spending) = expected_weeks :=
by
  sorry

end money_last_weeks_l1071_107144


namespace xiao_ming_second_half_time_l1071_107155

theorem xiao_ming_second_half_time :
  ∀ (total_distance : ℕ) (speed1 : ℕ) (speed2 : ℕ), 
    total_distance = 360 →
    speed1 = 5 →
    speed2 = 4 →
    let t_total := total_distance / (speed1 + speed2) * 2
    let half_distance := total_distance / 2
    let t2 := half_distance / speed2
    half_distance / speed2 + (half_distance / speed1) = 44 :=
sorry

end xiao_ming_second_half_time_l1071_107155


namespace fraction_of_meat_used_for_meatballs_l1071_107164

theorem fraction_of_meat_used_for_meatballs
    (initial_meat : ℕ)
    (spring_rolls_meat : ℕ)
    (remaining_meat : ℕ)
    (total_meat_used : ℕ)
    (meatballs_meat : ℕ)
    (h_initial : initial_meat = 20)
    (h_spring_rolls : spring_rolls_meat = 3)
    (h_remaining : remaining_meat = 12) :
    (initial_meat - remaining_meat) = total_meat_used ∧
    (total_meat_used - spring_rolls_meat) = meatballs_meat ∧
    (meatballs_meat / initial_meat) = (1/4 : ℝ) :=
by
  sorry

end fraction_of_meat_used_for_meatballs_l1071_107164


namespace unit_digit_23_pow_100000_l1071_107131

theorem unit_digit_23_pow_100000 : (23^100000) % 10 = 1 := 
by
  -- Import necessary submodules and definitions

sorry

end unit_digit_23_pow_100000_l1071_107131


namespace correct_calculation_l1071_107160

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end correct_calculation_l1071_107160


namespace increase_by_150_percent_l1071_107109

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l1071_107109


namespace Larry_sessions_per_day_eq_2_l1071_107130

variable (x : ℝ)
variable (sessions_per_day_time : ℝ)
variable (feeding_time_per_day : ℝ)
variable (total_time_per_day : ℝ)

theorem Larry_sessions_per_day_eq_2
  (h1: sessions_per_day_time = 30 * x)
  (h2: feeding_time_per_day = 12)
  (h3: total_time_per_day = 72) :
  x = 2 := by
  sorry

end Larry_sessions_per_day_eq_2_l1071_107130


namespace average_of_values_l1071_107195

theorem average_of_values (z : ℝ) : 
  (0 + 1 + 2 + 4 + 8 + 32 : ℝ) * z / (6 : ℝ) = 47 * z / 6 :=
by
  sorry

end average_of_values_l1071_107195


namespace percent_is_250_l1071_107146

def part : ℕ := 150
def whole : ℕ := 60
def percent := (part : ℚ) / (whole : ℚ) * 100

theorem percent_is_250 : percent = 250 := 
by 
  sorry

end percent_is_250_l1071_107146


namespace angle_A_size_max_area_triangle_l1071_107152

open Real

variable {A B C a b c : ℝ}

-- Part 1: Prove the size of angle A given the conditions
theorem angle_A_size (h1 : (2 * c - b) / a = cos B / cos A) :
  A = π / 3 :=
sorry

-- Part 2: Prove the maximum area of triangle ABC
theorem max_area_triangle (h2 : a = 2 * sqrt 5) :
  ∃ (S : ℝ), S = 5 * sqrt 3 ∧ ∀ (b c : ℝ), S ≤ 1/2 * b * c * sin (π / 3) :=
sorry

end angle_A_size_max_area_triangle_l1071_107152


namespace simplify_expression_l1071_107138

theorem simplify_expression : 
  2^345 - 3^4 * (3^2)^2 = 2^345 - 6561 := by
sorry

end simplify_expression_l1071_107138


namespace Bill_original_profit_percentage_l1071_107147

theorem Bill_original_profit_percentage 
  (S : ℝ) 
  (h_S : S = 879.9999999999993) 
  (h_cond : ∀ (P : ℝ), 1.17 * P = S + 56) :
  ∃ (profit_percentage : ℝ), profit_percentage = 10 := 
by
  sorry

end Bill_original_profit_percentage_l1071_107147


namespace value_of_power_l1071_107181

theorem value_of_power (a : ℝ) (m n k : ℕ) (h1 : a ^ m = 2) (h2 : a ^ n = 4) (h3 : a ^ k = 32) : 
  a ^ (3 * m + 2 * n - k) = 4 := 
by sorry

end value_of_power_l1071_107181


namespace negation_of_exists_implies_forall_l1071_107183

theorem negation_of_exists_implies_forall :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry

end negation_of_exists_implies_forall_l1071_107183


namespace shari_total_distance_l1071_107172

theorem shari_total_distance (speed : ℝ) (time_1 : ℝ) (rest : ℝ) (time_2 : ℝ) (distance : ℝ) :
  speed = 4 ∧ time_1 = 2 ∧ rest = 0.5 ∧ time_2 = 1 ∧ distance = speed * time_1 + speed * time_2 → distance = 12 :=
by
  sorry

end shari_total_distance_l1071_107172


namespace f_2015_2016_l1071_107171

noncomputable def f : ℤ → ℤ := sorry

theorem f_2015_2016 (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (x + 2) = -f x) (h3 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l1071_107171


namespace center_of_circle_l1071_107119

theorem center_of_circle :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → (x, y) = (1, 1) :=
by
  sorry

end center_of_circle_l1071_107119


namespace probability_of_forming_phrase_l1071_107112

theorem probability_of_forming_phrase :
  let cards := ["中", "国", "梦"]
  let n := 6
  let m := 1
  ∃ (p : ℚ), p = (m / n : ℚ) ∧ p = 1 / 6 :=
by
  sorry

end probability_of_forming_phrase_l1071_107112


namespace constant_term_g_eq_l1071_107121

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

theorem constant_term_g_eq : 
  (h.coeff 0 = 2) ∧ (f.coeff 0 = -6) →  g.coeff 0 = -1/3 := by
  sorry

end constant_term_g_eq_l1071_107121


namespace part_I_part_II_l1071_107198

open Set

variable (a b : ℝ)

theorem part_I (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hB_sub_A : B ⊆ A) : a = -1 :=
  sorry

theorem part_II (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hA_inter_B_nonempty : A ∩ B ≠ ∅) : a^2 - b^2 + 2 * a = -1 :=
  sorry

end part_I_part_II_l1071_107198


namespace positive_when_x_negative_l1071_107175

theorem positive_when_x_negative (x : ℝ) (h : x < 0) : (x / |x|)^2 > 0 := by
  sorry

end positive_when_x_negative_l1071_107175


namespace thabo_books_l1071_107174

/-- Thabo's book count puzzle -/
theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 200) : H = 35 :=
by
  -- sorry is used to skip the proof, only state the theorem.
  sorry

end thabo_books_l1071_107174


namespace geometric_series_sum_l1071_107125

theorem geometric_series_sum :
  ∀ (a r : ℚ) (n : ℕ), 
  a = 1 / 5 → 
  r = -1 / 5 → 
  n = 6 →
  (a - a * r^n) / (1 - r) = 1562 / 9375 :=
by 
  intro a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l1071_107125


namespace prove_ordered_pair_l1071_107192

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

theorem prove_ordered_pair (h1 : p 0 = -24) (h2 : q 0 = 30) (h3 : ∀ x : ℝ, p (q x) = q (p x)) : (p 3, q 6) = (3, -24) := 
sorry

end prove_ordered_pair_l1071_107192


namespace businessman_expenditure_l1071_107110

theorem businessman_expenditure (P : ℝ) (h1 : P * 1.21 = 24200) : P = 20000 := 
by sorry

end businessman_expenditure_l1071_107110


namespace function_decreasing_in_interval_l1071_107176

theorem function_decreasing_in_interval :
  ∀ (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) → 
  (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0 :=
by
  intros x1 x2 hx
  sorry

end function_decreasing_in_interval_l1071_107176


namespace tan_theta_point_l1071_107173

open Real

theorem tan_theta_point :
  ∀ θ : ℝ,
  ∃ (x y : ℝ), x = -sqrt 3 / 2 ∧ y = 1 / 2 ∧ (tan θ) = y / x → (tan θ) = -sqrt 3 / 3 :=
by
  sorry

end tan_theta_point_l1071_107173


namespace ship_speed_in_still_water_l1071_107165

theorem ship_speed_in_still_water 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (x : ℝ) 
  (h1 : distance = 36)
  (h2 : time = 6)
  (h3 : current_speed = 3) 
  (h4 : (18 / (x + 3) + 18 / (x - 3) = 6)) 
  : x = 3 + 3 * Real.sqrt 2 :=
sorry

end ship_speed_in_still_water_l1071_107165


namespace max_dance_counts_possible_l1071_107104

noncomputable def max_dance_counts : ℕ := 29

theorem max_dance_counts_possible (boys girls : ℕ) (dance_count : ℕ → ℕ) :
   boys = 29 → girls = 15 → 
   (∀ b, b < boys → dance_count b ≤ girls) → 
   (∀ g, g < girls → ∃ d, d ≤ boys ∧ dance_count d = g) →
   (∃ d, d ≤ max_dance_counts ∧
     (∀ k, k ≤ d → (∃ b, b < boys ∧ dance_count b = k) ∨ (∃ g, g < girls ∧ dance_count g = k))) := 
sorry

end max_dance_counts_possible_l1071_107104


namespace binom_16_12_eq_1820_l1071_107114

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l1071_107114
