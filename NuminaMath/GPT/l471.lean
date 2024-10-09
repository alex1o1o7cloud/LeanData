import Mathlib

namespace MN_squared_l471_47184

theorem MN_squared (PQ QR RS SP : ℝ) (h1 : PQ = 15) (h2 : QR = 15) (h3 : RS = 20) (h4 : SP = 20) (angle_S : ℝ) (h5 : angle_S = 90)
(M N: ℝ) (Midpoint_M : M = (QR / 2)) (Midpoint_N : N = (SP / 2)) : 
MN^2 = 100 := by
  sorry

end MN_squared_l471_47184


namespace ellipse_focal_distance_l471_47125

theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), (x^2 / 36 + y^2 / 9 = 9) → (∃ c : ℝ, c = 2 * Real.sqrt 3) :=
by
  sorry

end ellipse_focal_distance_l471_47125


namespace number_div_by_3_l471_47169

theorem number_div_by_3 (x : ℕ) (h : 54 = x - 39) : x / 3 = 31 :=
by
  sorry

end number_div_by_3_l471_47169


namespace pentagon_largest_angle_l471_47180

variable (F G H I J : ℝ)

-- Define the conditions given in the problem
axiom angle_sum : F + G + H + I + J = 540
axiom angle_F : F = 80
axiom angle_G : G = 100
axiom angle_HI : H = I
axiom angle_J : J = 2 * H + 20

-- Statement that the largest angle in the pentagon is 190°
theorem pentagon_largest_angle : max F (max G (max H (max I J))) = 190 :=
sorry

end pentagon_largest_angle_l471_47180


namespace evaluate_expression_l471_47144

theorem evaluate_expression (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 + 7 * x = 696 :=
by
  have hx : x = 3 := h
  sorry

end evaluate_expression_l471_47144


namespace arc_length_calc_l471_47175

-- Defining the conditions
def circle_radius := 12 -- radius OR
def angle_RIP := 30 -- angle in degrees

-- Defining the goal
noncomputable def arc_length_RP := 4 * Real.pi -- length of arc RP

-- The statement to prove
theorem arc_length_calc :
  arc_length_RP = 4 * Real.pi :=
sorry

end arc_length_calc_l471_47175


namespace possible_values_of_sum_l471_47156

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l471_47156


namespace polynomial_expansion_l471_47113

variable (x : ℝ)

theorem polynomial_expansion :
  (7*x^2 + 3)*(5*x^3 + 4*x + 1) = 35*x^5 + 43*x^3 + 7*x^2 + 12*x + 3 := by
  sorry

end polynomial_expansion_l471_47113


namespace basketball_game_proof_l471_47120

-- Definition of the conditions
def num_teams (x : ℕ) : Prop := ∃ n : ℕ, n = x

def games_played (x : ℕ) (total_games : ℕ) : Prop := total_games = 28

def game_combinations (x : ℕ) : ℕ := (x * (x - 1)) / 2

-- Proof statement using the conditions
theorem basketball_game_proof (x : ℕ) (h1 : num_teams x) (h2 : games_played x 28) : 
  game_combinations x = 28 := by
  sorry

end basketball_game_proof_l471_47120


namespace total_number_of_toys_is_105_l471_47118

-- Definitions
variables {a k : ℕ}

-- Conditions
def condition_1 (a k : ℕ) : Prop := k ≥ 2
def katya_toys (a : ℕ) : ℕ := a
def lena_toys (a k : ℕ) : ℕ := k * a
def masha_toys (a k : ℕ) : ℕ := k^2 * a

def after_katya_gave_toys (a : ℕ) : ℕ := a - 2
def after_lena_received_toys (a k : ℕ) : ℕ := k * a + 5
def after_masha_gave_toys (a k : ℕ) : ℕ := k^2 * a - 3

def arithmetic_progression (x1 x2 x3 : ℕ) : Prop :=
  2 * x2 = x1 + x3

-- Problem statement to prove
theorem total_number_of_toys_is_105 (a k : ℕ) (h1 : condition_1 a k)
  (h2 : arithmetic_progression (after_katya_gave_toys a) (after_lena_received_toys a k) (after_masha_gave_toys a k)) :
  katya_toys a + lena_toys a k + masha_toys a k = 105 :=
sorry

end total_number_of_toys_is_105_l471_47118


namespace number_of_regions_on_sphere_l471_47130

theorem number_of_regions_on_sphere (n : ℕ) (h : ∀ {a b c: ℤ}, a ≠ b → b ≠ c → a ≠ c → True) : 
  ∃ a_n, a_n = n^2 - n + 2 := 
by
  sorry

end number_of_regions_on_sphere_l471_47130


namespace part_I_part_II_l471_47110

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part_I (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≥ (1 : ℝ) / Real.exp 1 :=
sorry

theorem part_II (a x1 x2 x : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) (hx : x1 < x ∧ x < x2) :
  (f x a - f x1 a) / (x - x1) < (f x a - f x2 a) / (x - x2) :=
sorry

end part_I_part_II_l471_47110


namespace find_set_B_l471_47160

def A : Set ℕ := {1, 2}
def B : Set (Set ℕ) := { x | x ⊆ A }

theorem find_set_B : B = { ∅, {1}, {2}, {1, 2} } :=
by
  sorry

end find_set_B_l471_47160


namespace tournament_player_count_l471_47134

theorem tournament_player_count (n : ℕ) :
  (∃ points_per_game : ℕ, points_per_game = (n * (n - 1)) / 2) →
  (∃ T : ℕ, T = 90) →
  (n * (n - 1)) / 4 = 90 →
  n = 19 :=
by
  intros h1 h2 h3
  sorry

end tournament_player_count_l471_47134


namespace chengdu_chongqing_scientific_notation_l471_47123

theorem chengdu_chongqing_scientific_notation:
  (185000 : ℝ) = 1.85 * 10^5 :=
sorry

end chengdu_chongqing_scientific_notation_l471_47123


namespace convex_polyhedron_P_T_V_sum_eq_34_l471_47143

theorem convex_polyhedron_P_T_V_sum_eq_34
  (F : ℕ) (V : ℕ) (E : ℕ) (T : ℕ) (P : ℕ) 
  (hF : F = 32)
  (hT1 : 3 * T + 5 * P = 960)
  (hT2 : 2 * E = V * (T + P))
  (hT3 : T + P - 2 = 60)
  (hT4 : F + V - E = 2) :
  P + T + V = 34 := by
  sorry

end convex_polyhedron_P_T_V_sum_eq_34_l471_47143


namespace value_of_m_l471_47186

noncomputable def A (m : ℝ) : Set ℝ := {3, m}
noncomputable def B (m : ℝ) : Set ℝ := {3 * m, 3}

theorem value_of_m (m : ℝ) (h : A m = B m) : m = 0 :=
by
  sorry

end value_of_m_l471_47186


namespace seventh_root_of_unity_problem_l471_47194

theorem seventh_root_of_unity_problem (q : ℂ) (h : q^7 = 1) :
  (q = 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = 3 / 2) ∧ 
  (q ≠ 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = -2) :=
by
  sorry

end seventh_root_of_unity_problem_l471_47194


namespace cos_sum_is_one_or_cos_2a_l471_47104

open Real

theorem cos_sum_is_one_or_cos_2a (a b : ℝ) (h : ∫ x in a..b, sin x = 0) : cos (a + b) = 1 ∨ cos (a + b) = cos (2 * a) :=
  sorry

end cos_sum_is_one_or_cos_2a_l471_47104


namespace train_platform_length_l471_47147

theorem train_platform_length 
  (speed_train_kmph : ℕ) 
  (time_cross_platform : ℕ) 
  (time_cross_man : ℕ) 
  (L_platform : ℕ) :
  speed_train_kmph = 72 ∧ 
  time_cross_platform = 34 ∧ 
  time_cross_man = 18 ∧ 
  L_platform = 320 :=
by
  sorry

end train_platform_length_l471_47147


namespace Julia_watch_collection_l471_47150

section
variable (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) (total_watches : ℕ)

theorem Julia_watch_collection :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = 10 * (silver_watches + bronze_watches) / 100 →
  total_watches = silver_watches + bronze_watches + gold_watches →
  total_watches = 88 :=
by
  intros
  sorry
end

end Julia_watch_collection_l471_47150


namespace measure_diagonal_without_pythagorean_theorem_l471_47161

variables (a b c : ℝ)

-- Definition of the function to measure the diagonal distance
def diagonal_method (a b c : ℝ) : ℝ :=
  -- by calculating the hypotenuse scaled by sqrt(3), we ignore using the Pythagorean theorem directly
  sorry

-- Calculate distance by arranging bricks
theorem measure_diagonal_without_pythagorean_theorem (distance_extreme_corners : ℝ) :
  distance_extreme_corners = (diagonal_method a b c) :=
  sorry

end measure_diagonal_without_pythagorean_theorem_l471_47161


namespace not_prime_1001_base_l471_47136

theorem not_prime_1001_base (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (n^3 + 1) :=
sorry

end not_prime_1001_base_l471_47136


namespace subtraction_example_l471_47154

theorem subtraction_example :
  145.23 - 0.07 = 145.16 :=
sorry

end subtraction_example_l471_47154


namespace sum_four_digit_integers_l471_47137

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l471_47137


namespace f_of_5_l471_47178

/- The function f(x) is defined by f(x) = x^2 - x. Prove that f(5) = 20. -/
def f (x : ℤ) : ℤ := x^2 - x

theorem f_of_5 : f 5 = 20 := by
  sorry

end f_of_5_l471_47178


namespace average_age_of_adults_l471_47114

theorem average_age_of_adults (n_total n_girls n_boys n_adults : ℕ) 
                              (avg_age_total avg_age_girls avg_age_boys avg_age_adults : ℕ)
                              (h1 : n_total = 60)
                              (h2 : avg_age_total = 18)
                              (h3 : n_girls = 30)
                              (h4 : avg_age_girls = 16)
                              (h5 : n_boys = 20)
                              (h6 : avg_age_boys = 17)
                              (h7 : n_adults = 10) :
                              avg_age_adults = 26 :=
sorry

end average_age_of_adults_l471_47114


namespace gcd_polynomial_l471_47103

theorem gcd_polynomial (b : ℤ) (hb : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + b^2 + 6 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l471_47103


namespace cheerleaders_uniforms_l471_47176

theorem cheerleaders_uniforms (total_cheerleaders : ℕ) (size_6_cheerleaders : ℕ) (half_size_6_cheerleaders : ℕ) (size_2_cheerleaders : ℕ) : 
  total_cheerleaders = 19 →
  size_6_cheerleaders = 10 →
  half_size_6_cheerleaders = size_6_cheerleaders / 2 →
  size_2_cheerleaders = total_cheerleaders - (size_6_cheerleaders + half_size_6_cheerleaders) →
  size_2_cheerleaders = 4 :=
by
  intros
  sorry

end cheerleaders_uniforms_l471_47176


namespace stream_speed_l471_47158

theorem stream_speed (v : ℝ) (boat_speed : ℝ) (distance : ℝ) (time : ℝ) 
    (h1 : boat_speed = 10) 
    (h2 : distance = 54) 
    (h3 : time = 3) 
    (h4 : distance = (boat_speed + v) * time) : 
    v = 8 :=
by
  sorry

end stream_speed_l471_47158


namespace sequences_equal_l471_47116

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2018 / (n + 1)) * a (n + 1) + a n

noncomputable def b : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2020 / (n + 1)) * b (n + 1) + b n

theorem sequences_equal :
  (a 1010) / 1010 = (b 1009) / 1009 :=
sorry

end sequences_equal_l471_47116


namespace inequality_solution_l471_47185

noncomputable def operation (a b : ℝ) : ℝ := (a + 3 * b) - a * b

theorem inequality_solution (x : ℝ) : operation 5 x < 13 → x > -4 := by
  sorry

end inequality_solution_l471_47185


namespace x_y_solution_l471_47182

variable (x y : ℕ)

noncomputable def x_wang_speed : ℕ := x - 6

theorem x_y_solution (hx : (5 : ℚ) / 6 * x = y) (hy : (2 : ℚ) / 3 * (x - 6) = y - 10) : x = 36 ∧ y = 30 :=
by {
  sorry
}

end x_y_solution_l471_47182


namespace triangle_area_is_two_l471_47121

noncomputable def triangle_area (b c : ℝ) (angle_A : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_A

theorem triangle_area_is_two
  (A B C : ℝ) (a b c : ℝ)
  (hA : A = π / 4)
  (hCondition : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B)
  (hBC : b * c = 4 * Real.sqrt 2) : 
  triangle_area b c A = 2 :=
by
  -- actual proof omitted
  sorry

end triangle_area_is_two_l471_47121


namespace tangent_line_ln_l471_47162

theorem tangent_line_ln (x y : ℝ) (h_curve : y = Real.log (x + 1)) (h_point : (1, Real.log 2) = (1, y)) :
  x - 2 * y - 1 + 2 * Real.log 2 = 0 :=
by
  sorry

end tangent_line_ln_l471_47162


namespace probability_of_union_l471_47105

-- Define the range of two-digit numbers
def digit_count : ℕ := 90

-- Define events A and B
def event_a (n : ℕ) : Prop := n % 2 = 0
def event_b (n : ℕ) : Prop := n % 5 = 0

-- Define the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℚ := 45 / digit_count
def P_B : ℚ := 18 / digit_count
def P_A_and_B : ℚ := 9 / digit_count

-- Prove the final probability using inclusion-exclusion principle
theorem probability_of_union : P_A + P_B - P_A_and_B = 0.6 := by
  sorry

end probability_of_union_l471_47105


namespace regression_estimate_l471_47115

theorem regression_estimate (x : ℝ) (h : x = 28) : 4.75 * x + 257 = 390 :=
by
  rw [h]
  norm_num

end regression_estimate_l471_47115


namespace find_sequence_term_l471_47164

noncomputable def sequence_sum (n : ℕ) : ℚ :=
  (2 / 3) * n^2 - (1 / 3) * n

def sequence_term (n : ℕ) : ℚ :=
  if n = 1 then (1 / 3) else (4 / 3) * n - 1

theorem find_sequence_term (n : ℕ) : sequence_term n = (sequence_sum n - sequence_sum (n - 1)) :=
by
  unfold sequence_sum
  unfold sequence_term
  sorry

end find_sequence_term_l471_47164


namespace product_has_trailing_zeros_l471_47191

theorem product_has_trailing_zeros (a b : ℕ) (h1 : a = 350) (h2 : b = 60) :
  ∃ (n : ℕ), (10^n ∣ a * b) ∧ n = 3 :=
by
  sorry

end product_has_trailing_zeros_l471_47191


namespace not_possible_to_fill_grid_l471_47165

theorem not_possible_to_fill_grid :
  ¬ ∃ (f : Fin 7 → Fin 7 → ℝ), ∀ i j : Fin 7,
    ((if j > 0 then f i (j - 1) else 0) +
     (if j < 6 then f i (j + 1) else 0) +
     (if i > 0 then f (i - 1) j else 0) +
     (if i < 6 then f (i + 1) j else 0)) = 1 :=
by
  sorry

end not_possible_to_fill_grid_l471_47165


namespace log5_6_identity_l471_47153

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 3

theorem log5_6_identity :
  Real.log 6 / Real.log 5 = ((a * b) + 1) / (b - (a * b)) :=
by sorry

end log5_6_identity_l471_47153


namespace chef_used_apples_l471_47138

theorem chef_used_apples (initial_apples remaining_apples used_apples : ℕ) 
  (h1 : initial_apples = 40) 
  (h2 : remaining_apples = 39) 
  (h3 : used_apples = initial_apples - remaining_apples) : 
  used_apples = 1 := 
  sorry

end chef_used_apples_l471_47138


namespace correct_quotient_of_original_division_operation_l471_47128

theorem correct_quotient_of_original_division_operation 
  (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 102)
  (h2 : correct_divisor = 201)
  (h3 : incorrect_quotient = 753)
  (h4 : ∃ k, k = incorrect_quotient * 3) :
  ∃ q, q = 1146 ∧ (correct_divisor * q = incorrect_divisor * (incorrect_quotient * 3)) :=
by
  sorry

end correct_quotient_of_original_division_operation_l471_47128


namespace male_students_count_l471_47127

variable (M F : ℕ)
variable (average_all average_male average_female : ℕ)
variable (total_male total_female total_all : ℕ)

noncomputable def male_students (M F : ℕ) : ℕ := 8

theorem male_students_count:
  F = 32 -> average_all = 90 -> average_male = 82 -> average_female = 92 ->
  total_male = average_male * M -> total_female = average_female * F -> 
  total_all = average_all * (M + F) -> total_male + total_female = total_all ->
  M = male_students M F := 
by
  intros hF hAvgAll hAvgMale hAvgFemale hTotalMale hTotalFemale hTotalAll hEqTotal
  sorry

end male_students_count_l471_47127


namespace sum_in_base_4_l471_47109

theorem sum_in_base_4 : 
  let n1 := 2
  let n2 := 23
  let n3 := 132
  let n4 := 1320
  let sum := 20200
  n1 + n2 + n3 + n4 = sum := 
by
  sorry

end sum_in_base_4_l471_47109


namespace set_inter_complement_U_B_l471_47197

-- Define sets U, A, B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- Statement to prove
theorem set_inter_complement_U_B :
  A ∩ (Uᶜ \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end set_inter_complement_U_B_l471_47197


namespace intersection_of_line_with_x_axis_l471_47166

theorem intersection_of_line_with_x_axis 
  (k : ℝ) 
  (h : ∀ x y : ℝ, y = k * x + 4 → (x = -1 ∧ y = 2)) 
  : ∃ x : ℝ, (2 : ℝ) * x + 4 = 0 ∧ x = -2 :=
by {
  sorry
}

end intersection_of_line_with_x_axis_l471_47166


namespace domain_of_function_l471_47190

theorem domain_of_function :
  {x : ℝ | 2 - x > 0 ∧ 1 + x > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end domain_of_function_l471_47190


namespace no_prime_p_satisfies_l471_47124

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_prime_p_satisfies (p : ℕ) (hp : Nat.Prime p) (hp1 : is_perfect_square (7 * p + 3 ^ p - 4)) : False :=
by
  sorry

end no_prime_p_satisfies_l471_47124


namespace original_population_has_factor_three_l471_47122

theorem original_population_has_factor_three (x y z : ℕ) 
  (hx : ∃ n : ℕ, x = n ^ 2) -- original population is a perfect square
  (h1 : x + 150 = y^2 - 1)  -- after increase of 150, population is one less than a perfect square
  (h2 : y^2 - 1 + 150 = z^2) -- after another increase of 150, population is a perfect square again
  : 3 ∣ x :=
sorry

end original_population_has_factor_three_l471_47122


namespace volume_of_convex_polyhedron_l471_47181

variables {S1 S2 S : ℝ} {h : ℝ}

theorem volume_of_convex_polyhedron (S1 S2 S h : ℝ) :
  (h > 0) → (S1 ≥ 0) → (S2 ≥ 0) → (S ≥ 0) →
  ∃ V, V = (h / 6) * (S1 + S2 + 4 * S) :=
by {
  sorry
}

end volume_of_convex_polyhedron_l471_47181


namespace complement_U_A_l471_47145

def U : Set ℝ := { x | x^2 ≤ 4 }
def A : Set ℝ := { x | abs (x + 1) ≤ 1 }

theorem complement_U_A :
  (U \ A) = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end complement_U_A_l471_47145


namespace captain_age_l471_47132

noncomputable def whole_team_age : ℕ := 253
noncomputable def remaining_players_age : ℕ := 198
noncomputable def captain_and_wicket_keeper_age : ℕ := whole_team_age - remaining_players_age
noncomputable def wicket_keeper_age (C : ℕ) : ℕ := C + 3

theorem captain_age (C : ℕ) (whole_team : whole_team_age = 11 * 23) (remaining_players : remaining_players_age = 9 * 22) 
    (sum_ages : captain_and_wicket_keeper_age = 55) (wicket_keeper : wicket_keeper_age C = C + 3) : C = 26 := 
  sorry

end captain_age_l471_47132


namespace m_equals_p_of_odd_prime_and_integers_l471_47108

theorem m_equals_p_of_odd_prime_and_integers (p m : ℕ) (x y : ℕ) (hp : p > 1 ∧ ¬ (p % 2 = 0)) 
    (hx : x > 1) (hy : y > 1) 
    (h : (x ^ p + y ^ p) / 2 = ((x + y) / 2) ^ m): 
    m = p := 
by 
  sorry

end m_equals_p_of_odd_prime_and_integers_l471_47108


namespace find_MT_square_l471_47193

-- Definitions and conditions
variables (P Q R S L O M N T U : Type*)
variables (x : ℝ)
variables (PL PQ PS QR RS LO : finset ℝ)
variable (side_length_PQRS : ℝ) (area_PLQ area_QMTL area_SNUL area_RNMUT : ℝ)
variables (LO_MT_perpendicular LO_NU_perpendicular : Prop)

-- Stating the problem
theorem find_MT_square :
  (side_length_PQRS = 3) →
  (PL ⊆ PQ) →
  (PO ⊆ PS) →
  (PL = PO) →
  (PL = x) →
  (U ∈ LO) →
  (T ∈ LO) →
  (LO_MT_perpendicular) →
  (LO_NU_perpendicular) →
  (area_PLQ = 1) →
  (area_QMTL = 1) →
  (area_SNUL = 2) →
  (area_RNMUT = 2) →
  (x^2 / 2 = 1) → 
  (PL * LO = 1) →
  MT^2 = 1 / 2 :=
sorry

end find_MT_square_l471_47193


namespace motel_percentage_reduction_l471_47188

theorem motel_percentage_reduction
  (x y : ℕ) 
  (h : 40 * x + 60 * y = 1000) :
  ((1000 - (40 * (x + 10) + 60 * (y - 10))) / 1000) * 100 = 20 := 
by
  sorry

end motel_percentage_reduction_l471_47188


namespace tractor_brigades_l471_47174
noncomputable def brigade_plowing : Prop :=
∃ x y : ℝ,
  x * y = 240 ∧
  (x + 3) * (y + 2) = 324 ∧
  x > 20 ∧
  (x + 3) > 20 ∧
  x = 24 ∧
  (x + 3) = 27

theorem tractor_brigades:
  brigade_plowing :=
sorry

end tractor_brigades_l471_47174


namespace largest_multiple_of_11_less_than_minus_150_l471_47100

theorem largest_multiple_of_11_less_than_minus_150 : 
  ∃ n : ℤ, (n * 11 < -150) ∧ (∀ m : ℤ, (m * 11 < -150) →  n * 11 ≥ m * 11) ∧ (n * 11 = -154) :=
by
  sorry

end largest_multiple_of_11_less_than_minus_150_l471_47100


namespace find_inlet_rate_l471_47195

-- definitions for the given conditions
def volume_cubic_feet : ℝ := 20
def conversion_factor : ℝ := 12^3
def volume_cubic_inches : ℝ := volume_cubic_feet * conversion_factor

def outlet_rate1 : ℝ := 9
def outlet_rate2 : ℝ := 8
def empty_time : ℕ := 2880

-- theorem that captures the proof problem
theorem find_inlet_rate (volume_cubic_inches : ℝ) (outlet_rate1 outlet_rate2 empty_time : ℝ) :
  ∃ (inlet_rate : ℝ), volume_cubic_inches = (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time ↔ inlet_rate = 5 := 
by
  sorry

end find_inlet_rate_l471_47195


namespace ratio_of_expenditures_l471_47196

variable (Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings: ℤ)
variable (ratio_incomes: ℚ)
variable (savings_amount: ℤ)

-- Given conditions
def conditions : Prop :=
  Rajan_income = 7000 ∧
  ratio_incomes = 7 / 6 ∧
  savings_amount = 1000 ∧
  Rajan_savings = Rajan_income - Rajan_expenditure ∧
  Balan_savings = Balan_income - Balan_expenditure ∧
  Rajan_savings = savings_amount ∧
  Balan_savings = savings_amount

-- The theorem we want to prove
theorem ratio_of_expenditures :
  conditions Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings ratio_incomes savings_amount →
  (Rajan_expenditure : ℚ) / (Balan_expenditure : ℚ) = 6 / 5 :=
by
  sorry

end ratio_of_expenditures_l471_47196


namespace probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l471_47155

noncomputable def probability_first_third_fifth_hit : ℚ :=
  (3 / 5) * (2 / 5) * (3 / 5) * (2 / 5) * (3 / 5)

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  ↑(Nat.factorial n) / (↑(Nat.factorial k) * ↑(Nat.factorial (n - k)))

noncomputable def probability_exactly_three_hits : ℚ :=
  binomial_coefficient 5 3 * (3 / 5)^3 * (2 / 5)^2

theorem probability_first_third_fifth_correct :
  probability_first_third_fifth_hit = 108 / 3125 :=
by sorry

theorem probability_exactly_three_hits_correct :
  probability_exactly_three_hits = 216 / 625 :=
by sorry

end probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l471_47155


namespace find_y_l471_47172

theorem find_y (y : ℝ) (h : (y + 10 + (5 * y) + 4 + (3 * y) + 12) / 3 = 6 * y - 8) :
  y = 50 / 9 := by
  sorry

end find_y_l471_47172


namespace james_money_left_no_foreign_currency_needed_l471_47168

noncomputable def JameMoneyLeftAfterPurchase : ℝ :=
  let usd_bills := 50 + 20 + 5 + 1 + 20 + 10 -- USD bills and coins
  let euro_in_usd := 5 * 1.20               -- €5 bill to USD
  let pound_in_usd := 2 * 1.35 - 0.8 / 100 * (2 * 1.35) -- £2 coin to USD after fee
  let yen_in_usd := 100 * 0.009 - 1.5 / 100 * (100 * 0.009) -- ¥100 coin to USD after fee
  let franc_in_usd := 2 * 1.08 - 1 / 100 * (2 * 1.08) -- 2₣ coins to USD after fee
  let total_usd := usd_bills + euro_in_usd + pound_in_usd + yen_in_usd + franc_in_usd
  let present_cost_with_tax := 88 * 1.08   -- Present cost after 8% tax
  total_usd - present_cost_with_tax        -- Amount left after purchasing the present

theorem james_money_left :
  JameMoneyLeftAfterPurchase = 22.6633 :=
by
  sorry

theorem no_foreign_currency_needed :
  (0 : ℝ)  = 0 :=
by
  sorry

end james_money_left_no_foreign_currency_needed_l471_47168


namespace geometric_seq_ratio_l471_47140

theorem geometric_seq_ratio : 
  ∀ (a : ℕ → ℝ) (q : ℝ), 
    (∀ n, a (n+1) = a n * q) → 
    q > 1 → 
    a 1 + a 6 = 8 → 
    a 3 * a 4 = 12 → 
    a 2018 / a 2013 = 3 :=
by
  intros a q h_geom h_q_pos h_sum_eq h_product_eq
  sorry

end geometric_seq_ratio_l471_47140


namespace max_value_of_xyz_l471_47148

theorem max_value_of_xyz (x y z : ℝ) (h : x + 3 * y + z = 5) : xy + xz + yz ≤ 125 / 4 := 
sorry

end max_value_of_xyz_l471_47148


namespace necessary_but_not_sufficient_condition_l471_47177

theorem necessary_but_not_sufficient_condition (a b c d : ℝ) : 
  (a + b < c + d) → (a < c ∨ b < d) :=
sorry

end necessary_but_not_sufficient_condition_l471_47177


namespace brownies_cut_into_pieces_l471_47170

theorem brownies_cut_into_pieces (total_amount_made : ℕ) (pans : ℕ) (cost_per_brownie : ℕ) (brownies_sold : ℕ) 
  (h1 : total_amount_made = 32) (h2 : pans = 2) (h3 : cost_per_brownie = 2) (h4 : brownies_sold = total_amount_made / cost_per_brownie) :
  16 = brownies_sold :=
by
  sorry

end brownies_cut_into_pieces_l471_47170


namespace triangle_rational_segments_l471_47119

theorem triangle_rational_segments (a b c : ℚ) (h : a + b > c ∧ a + c > b ∧ b + c > a):
  ∃ (ab1 cb1 : ℚ), (ab1 + cb1 = b) := sorry

end triangle_rational_segments_l471_47119


namespace least_area_in_rectangle_l471_47179

theorem least_area_in_rectangle
  (x y : ℤ)
  (h1 : 2 * (x + y) = 150)
  (h2 : x > 0)
  (h3 : y > 0) :
  ∃ x y : ℤ, (2 * (x + y) = 150) ∧ (x * y = 74) := by
  sorry

end least_area_in_rectangle_l471_47179


namespace pool_perimeter_l471_47129

theorem pool_perimeter (garden_length : ℝ) (plot_area : ℝ) (plot_count : ℕ) : 
  garden_length = 9 ∧ plot_area = 20 ∧ plot_count = 4 →
  ∃ (pool_perimeter : ℝ), pool_perimeter = 18 :=
by
  intros h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end pool_perimeter_l471_47129


namespace sum_of_integers_l471_47142

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 10) (h2 : x * y = 80) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 20 := by
  sorry

end sum_of_integers_l471_47142


namespace product_of_two_odd_numbers_not_always_composite_l471_47187

theorem product_of_two_odd_numbers_not_always_composite :
  ∃ (m n : ℕ), (¬ (2 ∣ m) ∧ ¬ (2 ∣ n)) ∧ (∀ d : ℕ, d ∣ (m * n) → d = 1 ∨ d = m * n) :=
by
  sorry

end product_of_two_odd_numbers_not_always_composite_l471_47187


namespace margo_total_distance_l471_47112

theorem margo_total_distance
  (t1 t2 : ℚ) (rate1 rate2 : ℚ)
  (h1 : t1 = 15 / 60)
  (h2 : t2 = 25 / 60)
  (r1 : rate1 = 5)
  (r2 : rate2 = 3) :
  (t1 * rate1 + t2 * rate2 = 2.5) :=
by
  sorry

end margo_total_distance_l471_47112


namespace fewer_onions_than_tomatoes_and_corn_l471_47159

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l471_47159


namespace number_of_female_students_l471_47107

theorem number_of_female_students
    (F : ℕ)  -- Number of female students
    (avg_all : ℝ)  -- Average score for all students
    (avg_male : ℝ)  -- Average score for male students
    (avg_female : ℝ)  -- Average score for female students
    (num_male : ℕ)  -- Number of male students
    (h_avg_all : avg_all = 90)
    (h_avg_male : avg_male = 82)
    (h_avg_female : avg_female = 92)
    (h_num_male : num_male = 8)
    (h_avg : avg_all * (num_male + F) = avg_male * num_male + avg_female * F) :
  F = 32 :=
by
  sorry

end number_of_female_students_l471_47107


namespace quadratic_inequality_l471_47111

theorem quadratic_inequality (c : ℝ) (h₁ : 0 < c) (h₂ : c < 16): ∃ x : ℝ, x^2 - 8 * x + c < 0 :=
sorry

end quadratic_inequality_l471_47111


namespace total_movie_hours_l471_47157

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end total_movie_hours_l471_47157


namespace valid_pairs_iff_l471_47126

noncomputable def valid_pairs (a b : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ a * (⌊ b * n ⌋ : ℝ) = b * (⌊ a * n ⌋ : ℝ)

theorem valid_pairs_iff (a b : ℝ) : valid_pairs a b ↔
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (m n : ℤ), a = m ∧ b = n)) :=
by sorry

end valid_pairs_iff_l471_47126


namespace james_needs_to_sell_12_coins_l471_47133

theorem james_needs_to_sell_12_coins:
  ∀ (num_coins : ℕ) (initial_price new_price : ℝ),
  num_coins = 20 ∧ initial_price = 15 ∧ new_price = initial_price + (2 / 3) * initial_price →
  (num_coins * initial_price) / new_price = 12 :=
by
  intros num_coins initial_price new_price h
  obtain ⟨hc1, hc2, hc3⟩ := h
  sorry

end james_needs_to_sell_12_coins_l471_47133


namespace weight_of_daughter_l471_47117

def mother_daughter_grandchild_weight (M D C : ℝ) :=
  M + D + C = 130 ∧
  D + C = 60 ∧
  C = 1/5 * M

theorem weight_of_daughter (M D C : ℝ) 
  (h : mother_daughter_grandchild_weight M D C) : D = 46 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_of_daughter_l471_47117


namespace count_positive_integers_satisfying_conditions_l471_47192

theorem count_positive_integers_satisfying_conditions :
  let condition1 (n : ℕ) := (169 * n) ^ 25 > n ^ 75
  let condition2 (n : ℕ) := n ^ 75 > 3 ^ 150
  ∃ (count : ℕ), count = 3 ∧ (∀ (n : ℕ), (condition1 n) ∧ (condition2 n) → 9 < n ∧ n < 13) :=
by
  sorry

end count_positive_integers_satisfying_conditions_l471_47192


namespace local_minimum_at_2_l471_47139

noncomputable def f (x : ℝ) : ℝ := (2 / x) + Real.log x

theorem local_minimum_at_2 : ∃ δ > 0, ∀ y, abs (y - 2) < δ → f y ≥ f 2 := by
  sorry

end local_minimum_at_2_l471_47139


namespace find_g_l471_47167

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g (g : ℝ → ℝ)
  (H : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  g = fun x => x + 5 :=
by
  sorry

end find_g_l471_47167


namespace jill_peaches_l471_47163

variable (S J : ℕ)

theorem jill_peaches (h1 : S = 19) (h2 : S = J + 13) : J = 6 :=
by
  sorry

end jill_peaches_l471_47163


namespace largest_r_l471_47101

theorem largest_r (p q r : ℝ) (h1 : p + q + r = 6) (h2 : p*q + p*r + q*r = 8) : 
  r ≤ 2 + Real.sqrt (20/3) := 
sorry

end largest_r_l471_47101


namespace joe_first_lift_weight_l471_47152

variables (x y : ℕ)

theorem joe_first_lift_weight (h1 : x + y = 600) (h2 : 2 * x = y + 300) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l471_47152


namespace cost_combination_exists_l471_47146

/-!
Given:
- Nadine spent a total of $105.
- The table costs $34.
- The mirror costs $15.
- The lamp costs $6.
- The total cost of the 2 chairs and 3 decorative vases is $50.

Prove:
- There are multiple combinations of individual chair cost (C) and individual vase cost (V) such that 2 * C + 3 * V = 50.
-/

theorem cost_combination_exists :
  ∃ (C V : ℝ), 2 * C + 3 * V = 50 :=
by {
  sorry
}

end cost_combination_exists_l471_47146


namespace factorize_polynomial_l471_47131

theorem factorize_polynomial (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 2 * y * (x - y)^2 :=
sorry

end factorize_polynomial_l471_47131


namespace journey_time_l471_47183

noncomputable def velocity_of_stream : ℝ := 4
noncomputable def speed_of_boat_in_still_water : ℝ := 14
noncomputable def distance_A_to_B : ℝ := 180
noncomputable def distance_B_to_C : ℝ := distance_A_to_B / 2
noncomputable def downstream_speed : ℝ := speed_of_boat_in_still_water + velocity_of_stream
noncomputable def upstream_speed : ℝ := speed_of_boat_in_still_water - velocity_of_stream

theorem journey_time : (distance_A_to_B / downstream_speed) + (distance_B_to_C / upstream_speed) = 19 := by
  sorry

end journey_time_l471_47183


namespace common_difference_is_7_l471_47173

-- Define the arithmetic sequence with common difference d
def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Define the conditions
variables (a1 d : ℕ)

-- Define the conditions provided in the problem
def condition1 := (arithmetic_seq a1 d 3) + (arithmetic_seq a1 d 6) = 11
def condition2 := (arithmetic_seq a1 d 5) + (arithmetic_seq a1 d 8) = 39

-- Prove that the common difference d is 7
theorem common_difference_is_7 : condition1 a1 d → condition2 a1 d → d = 7 :=
by
  intros cond1 cond2
  sorry

end common_difference_is_7_l471_47173


namespace sum_of_digits_of_m_eq_nine_l471_47135

theorem sum_of_digits_of_m_eq_nine
  (m : ℕ)
  (h1 : m * 3 / 2 - 72 = m) :
  1 + (m / 10 % 10) + (m % 10) = 9 :=
by
  sorry

end sum_of_digits_of_m_eq_nine_l471_47135


namespace orange_juice_fraction_l471_47171

theorem orange_juice_fraction 
    (capacity1 capacity2 : ℕ)
    (orange_fraction1 orange_fraction2 : ℚ)
    (h_capacity1 : capacity1 = 800)
    (h_capacity2 : capacity2 = 700)
    (h_orange_fraction1 : orange_fraction1 = 1/4)
    (h_orange_fraction2 : orange_fraction2 = 1/3) :
    (capacity1 * orange_fraction1 + capacity2 * orange_fraction2) / (capacity1 + capacity2) = 433.33 / 1500 :=
by sorry

end orange_juice_fraction_l471_47171


namespace distinct_triangles_in_3x3_grid_l471_47198

theorem distinct_triangles_in_3x3_grid : 
  let num_points := 9 
  let total_combinations := Nat.choose num_points 3 
  let degenerate_cases := 8
  total_combinations - degenerate_cases = 76 := 
by
  sorry

end distinct_triangles_in_3x3_grid_l471_47198


namespace min_minutes_for_B_cheaper_l471_47151

-- Define the relevant constants and costs associated with each plan
def cost_A (x : ℕ) : ℕ := 12 * x
def cost_B (x : ℕ) : ℕ := 2500 + 6 * x
def cost_C (x : ℕ) : ℕ := 9 * x

-- Lean statement for the proof problem
theorem min_minutes_for_B_cheaper : ∃ (x : ℕ), x = 834 ∧ cost_B x < cost_A x ∧ cost_B x < cost_C x := 
sorry

end min_minutes_for_B_cheaper_l471_47151


namespace probability_segments_length_l471_47149

theorem probability_segments_length (x y : ℝ) : 
    80 ≥ x ∧ x ≥ 20 ∧ 80 ≥ y ∧ y ≥ 20 ∧ 80 ≥ 80 - x - y ∧ 80 - x - y ≥ 20 → 
    (∃ (s : ℝ), s = (200 / 3200) ∧ s = (1 / 16)) :=
by
  intros h
  sorry

end probability_segments_length_l471_47149


namespace quadratic_range_l471_47189

open Real

theorem quadratic_range (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) : 
  a > -2 ∧ a ≠ 0 :=
by 
  sorry

end quadratic_range_l471_47189


namespace avg_age_of_children_l471_47106

theorem avg_age_of_children 
  (participants : ℕ) (women : ℕ) (men : ℕ) (children : ℕ)
  (overall_avg_age : ℕ) (avg_age_women : ℕ) (avg_age_men : ℕ)
  (hp : participants = 50) (hw : women = 22) (hm : men = 18) (hc : children = 10)
  (ho : overall_avg_age = 20) (haw : avg_age_women = 24) (ham : avg_age_men = 19) :
  ∃ (avg_age_children : ℕ), avg_age_children = 13 :=
by
  -- Proof will be here.
  sorry

end avg_age_of_children_l471_47106


namespace find_d_l471_47199

-- Define the polynomial g(x)
def g (d : ℚ) (x : ℚ) : ℚ := d * x^4 + 11 * x^3 + 5 * d * x^2 - 28 * x + 72

-- The main proof statement
theorem find_d (hd : g d 4 = 0) : d = -83 / 42 := by
  sorry -- proof not needed as per prompt

end find_d_l471_47199


namespace distance_from_center_to_plane_correct_l471_47102

noncomputable def distance_from_center_to_plane (O A B C : ℝ × ℝ × ℝ) (radius : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * K)
  let OD := Real.sqrt (radius^2 - R^2)
  OD

theorem distance_from_center_to_plane_correct (O A B C : ℝ × ℝ × ℝ) :
  (dist O A = 20) →
  (dist O B = 20) →
  (dist O C = 20) →
  (dist A B = 13) →
  (dist B C = 14) →
  (dist C A = 15) →
  let m := 15
  let n := 95
  let k := 8
  m + n + k = 118 := by
  sorry

end distance_from_center_to_plane_correct_l471_47102


namespace problem_l471_47141

theorem problem (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - x) = x^2 + 1) : f (-1) = 5 := 
  sorry

end problem_l471_47141
