import Mathlib

namespace three_pow_zero_eq_one_l263_263780

theorem three_pow_zero_eq_one : 3^0 = 1 :=
by {
  -- Proof would go here
  sorry
}

end three_pow_zero_eq_one_l263_263780


namespace average_length_of_two_strings_l263_263859

theorem average_length_of_two_strings (a b : ℝ) (h1 : a = 3.2) (h2 : b = 4.8) :
  (a + b) / 2 = 4.0 :=
by
  sorry

end average_length_of_two_strings_l263_263859


namespace Charlie_age_when_Jenny_twice_as_Bobby_l263_263217

theorem Charlie_age_when_Jenny_twice_as_Bobby (B C J : ℕ) 
  (h₁ : J = C + 5)
  (h₂ : C = B + 3)
  (h₃ : J = 2 * B) : 
  C = 11 :=
by
  sorry

end Charlie_age_when_Jenny_twice_as_Bobby_l263_263217


namespace number_of_solutions_to_eq_count_number_of_solutions_to_eq_l263_263563

theorem number_of_solutions_to_eq {x y : ℤ} (h : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 7) :
  (x, y) = (-42, 6) ∨ (x, y) = (6, -42) ∨ (x, y) = (8, 56) ∨ (x, y) = (14, 14) ∨ (x, y) = (56, 8) :=
begin
  -- proof goes here
  sorry
end

theorem count_number_of_solutions_to_eq :
  {p : ℤ × ℤ // 1 / (p.fst : ℚ) + 1 / (p.snd : ℚ) = 1 / 7}.to_finset.card = 5 :=
begin
  -- proof goes here
  sorry
end

end number_of_solutions_to_eq_count_number_of_solutions_to_eq_l263_263563


namespace sum_of_a_b_l263_263033

def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem sum_of_a_b (a b : ℝ) (h : symmetric_x_axis (3, a) (b, 4)) : a + b = -1 :=
by
  sorry

end sum_of_a_b_l263_263033


namespace angle_triple_supplement_l263_263122

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263122


namespace bread_loaves_l263_263594

theorem bread_loaves (loaf_cost : ℝ) (pb_cost : ℝ) (total_money : ℝ) (leftover_money : ℝ) : ℝ :=
  let spent_money := total_money - leftover_money
  let remaining_money := spent_money - pb_cost
  remaining_money / loaf_cost

example : bread_loaves 2.25 2 14 5.25 = 3 := by
  sorry

end bread_loaves_l263_263594


namespace solve_for_p_l263_263015

theorem solve_for_p (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end solve_for_p_l263_263015


namespace collinear_points_in_cube_l263_263437

def collinear_groups_in_cube : Prop :=
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let center_point := 1
  let total_groups :=
    (vertices * (vertices - 1) / 2) + (face_centers * 1 / 2) + (edge_midpoints * 3 / 2)
  total_groups = 49

theorem collinear_points_in_cube : collinear_groups_in_cube :=
  by
    sorry

end collinear_points_in_cube_l263_263437


namespace Emily_cleaning_time_in_second_room_l263_263693

/-
Lilly, Fiona, Jack, and Emily are cleaning 3 rooms.
For the first room: Lilly and Fiona together: 1/4 of the time, Jack: 1/3 of the time, Emily: the rest of the time.
In the second room: Jack: 25%, Emily: 25%, Lilly and Fiona: the remaining 50%.
In the third room: Emily: 40%, Lilly: 20%, Jack: 20%, Fiona: 20%.
Total time for all rooms: 12 hours.

Prove that the total time Emily spent cleaning in the second room is 60 minutes.
-/

theorem Emily_cleaning_time_in_second_room :
  let total_time := 12 -- total time in hours
  let time_per_room := total_time / 3 -- time per room in hours
  let time_per_room_minutes := time_per_room * 60 -- time per room in minutes
  let emily_cleaning_percentage := 0.25 -- Emily's cleaning percentage in the second room
  let emily_cleaning_time := emily_cleaning_percentage * time_per_room_minutes -- cleaning time in minutes
  emily_cleaning_time = 60 := by
  sorry

end Emily_cleaning_time_in_second_room_l263_263693


namespace jordan_rectangle_width_l263_263924

theorem jordan_rectangle_width
  (length_carol : ℕ) (width_carol : ℕ) (length_jordan : ℕ) (width_jordan : ℕ)
  (h1 : length_carol = 5) (h2 : width_carol = 24) (h3 : length_jordan = 2)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 60 := by
  sorry

end jordan_rectangle_width_l263_263924


namespace min_odd_solution_l263_263931

theorem min_odd_solution (a m1 m2 n1 n2 : ℕ)
  (h1: a = m1^2 + n1^2)
  (h2: a^2 = m2^2 + n2^2)
  (h3: m1 - n1 = m2 - n2)
  (h4: a > 5)
  (h5: a % 2 = 1) :
  a = 261 :=
sorry

end min_odd_solution_l263_263931


namespace find_prime_number_between_50_and_60_l263_263561

theorem find_prime_number_between_50_and_60 (n : ℕ) :
  (50 < n ∧ n < 60) ∧ Prime n ∧ n % 7 = 3 ↔ n = 59 :=
by
  sorry

end find_prime_number_between_50_and_60_l263_263561


namespace remainder_of_series_div_9_l263_263127

def sum (n : Nat) : Nat := n * (n + 1) / 2

theorem remainder_of_series_div_9 : (sum 20) % 9 = 3 :=
by
  -- The proof will go here
  sorry

end remainder_of_series_div_9_l263_263127


namespace return_trip_time_l263_263179

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l263_263179


namespace solve_x_if_alpha_beta_eq_8_l263_263630

variable (x : ℝ)

def alpha (x : ℝ) := 4 * x + 9
def beta (x : ℝ) := 9 * x + 6

theorem solve_x_if_alpha_beta_eq_8 (hx : alpha (beta x) = 8) : x = (-25 / 36) :=
by
  sorry

end solve_x_if_alpha_beta_eq_8_l263_263630


namespace quadratic_has_real_roots_l263_263667

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l263_263667


namespace no_pos_reals_floor_prime_l263_263439

open Real
open Nat

theorem no_pos_reals_floor_prime : 
  ∀ (a b : ℝ), (0 < a) → (0 < b) → ∃ n : ℕ, ¬ Prime (⌊a * n + b⌋) :=
by
  intro a b a_pos b_pos
  sorry

end no_pos_reals_floor_prime_l263_263439


namespace abc_divisible_by_6_l263_263043

theorem abc_divisible_by_6 (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) :=
by
  sorry

end abc_divisible_by_6_l263_263043


namespace necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l263_263811

-- Proof Problem 1
theorem necessary_condition_for_q_implies_m_in_range (m : ℝ) (h1 : 0 < m) :
  (∀ x : ℝ, 2 - m ≤ x ∧ x ≤ 2 + m → -2 ≤ x ∧ x ≤ 6) →
  0 < m ∧ m ≤ 4 :=
by
  sorry

-- Proof Problem 2
theorem neg_p_or_neg_q_false_implies_x_in_range (m : ℝ) (x : ℝ)
  (h2 : m = 2)
  (h3 : (x + 2) * (x - 6) ≤ 0)
  (h4 : 2 - m ≤ x ∧ x ≤ 2 + m)
  (h5 : ¬ ((x + 2) * (x - 6) > 0 ∨ x < 2 - m ∨ x > 2 + m)) :
  0 ≤ x ∧ x ≤ 4 :=
by
  sorry

end necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l263_263811


namespace right_triangle_incenter_distance_l263_263962

noncomputable def triangle_right_incenter_distance : ℝ :=
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let area := (1 / 2) * AB * BC
  let s := (AB + BC + AC) / 2
  let r := area / s
  r

theorem right_triangle_incenter_distance :
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := 2 * Real.sqrt 17
  let area := 12 * Real.sqrt 2
  let s := 2 * Real.sqrt 2 + 3 + Real.sqrt 17
  let BI := area / s
  BI = triangle_right_incenter_distance := sorry

end right_triangle_incenter_distance_l263_263962


namespace isabel_total_problems_l263_263490

theorem isabel_total_problems
  (math_pages : ℕ)
  (reading_pages : ℕ)
  (problems_per_page : ℕ)
  (h1 : math_pages = 2)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 5) :
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end isabel_total_problems_l263_263490


namespace contradiction_proof_l263_263582

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) :=
by
  sorry

end contradiction_proof_l263_263582


namespace angle_triple_supplementary_l263_263107

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l263_263107


namespace Sandy_phone_bill_expense_l263_263536
noncomputable def Sandy_age_now : ℕ := 34
noncomputable def Kim_age_now : ℕ := 10
noncomputable def Sandy_phone_bill : ℕ := 10 * Sandy_age_now

theorem Sandy_phone_bill_expense :
  (Sandy_age_now - 2 = 36 - 2) ∧ (Kim_age_now + 2 = 12) ∧ (36 = 3 * 12) ∧ (Sandy_phone_bill = 340) := by
sorry

end Sandy_phone_bill_expense_l263_263536


namespace negative_integer_is_minus_21_l263_263341

variable (n : ℤ) (hn : n < 0) (h : n * (-3) + 2 = 65)

theorem negative_integer_is_minus_21 : n = -21 :=
by
  sorry

end negative_integer_is_minus_21_l263_263341


namespace ordered_triples_count_l263_263205

noncomputable def count_valid_triples (n : ℕ) :=
  ∃ x y z : ℕ, ∃ k : ℕ, x * y * z = k ∧ k = 5 ∧ lcm x y = 48 ∧ lcm x z = 450 ∧ lcm y z = 600

theorem ordered_triples_count : count_valid_triples 5 := by
  sorry

end ordered_triples_count_l263_263205


namespace quadratic_real_roots_l263_263663

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l263_263663


namespace num_students_only_math_l263_263721

def oakwood_ninth_grade_problem 
  (total_students: ℕ)
  (students_in_math: ℕ)
  (students_in_foreign_language: ℕ)
  (students_in_science: ℕ)
  (students_in_all_three: ℕ)
  (students_total_from_ie: ℕ) :=
  (total_students = 120) ∧
  (students_in_math = 85) ∧
  (students_in_foreign_language = 65) ∧
  (students_in_science = 75) ∧
  (students_in_all_three = 20) ∧
  total_students = students_in_math + students_in_foreign_language + students_in_science 
  - (students_total_from_ie) + students_in_all_three - (students_in_all_three)

theorem num_students_only_math 
  (total_students: ℕ := 120)
  (students_in_math: ℕ := 85)
  (students_in_foreign_language: ℕ := 65)
  (students_in_science: ℕ := 75)
  (students_in_all_three: ℕ := 20)
  (students_total_from_ie: ℕ := 45) :
  oakwood_ninth_grade_problem total_students students_in_math students_in_foreign_language students_in_science students_in_all_three students_total_from_ie →
  ∃ (students_only_math: ℕ), students_only_math = 75 :=
by
  sorry

end num_students_only_math_l263_263721


namespace complex_addition_l263_263778

theorem complex_addition :
  (⟨6, -5⟩ : ℂ) + (⟨3, 2⟩ : ℂ) = ⟨9, -3⟩ := 
sorry

end complex_addition_l263_263778


namespace union_of_complements_eq_l263_263337

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_of_complements_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 4, 5, 7} →
  B = {3, 4, 5} →
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 6, 7}) :=
by
  intros hU hA hB
  sorry

end union_of_complements_eq_l263_263337


namespace tank_capacity_is_780_l263_263912

noncomputable def tank_capacity : ℕ := 
  let fill_rate_A := 40
  let fill_rate_B := 30
  let drain_rate_C := 20
  let cycle_minutes := 3
  let total_minutes := 48
  let net_fill_per_cycle := fill_rate_A + fill_rate_B - drain_rate_C
  let total_cycles := total_minutes / cycle_minutes
  let total_fill := total_cycles * net_fill_per_cycle
  let final_capacity := total_fill - drain_rate_C -- Adjust for the last minute where C opens
  final_capacity

theorem tank_capacity_is_780 : tank_capacity = 780 := by
  unfold tank_capacity
  -- Proof steps to be filled in
  sorry

end tank_capacity_is_780_l263_263912


namespace find_angle_C_l263_263839

variable {A B C a b c : ℝ}
variable (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variable (hTriangle : A + B + C = π)
variable (hSides : a > 0 ∧ b > 0 ∧ c > 0)
variable (hCondition : Real.sqrt 3 * a = 2 * c * Real.sin A)

theorem find_angle_C (hA_pos : A ≠ 0) : C = π / 3 :=
  sorry

end find_angle_C_l263_263839


namespace total_carriages_l263_263593

-- Definitions based on given conditions
def Euston_carriages := 130
def Norfolk_carriages := Euston_carriages - 20
def Norwich_carriages := 100
def Flying_Scotsman_carriages := Norwich_carriages + 20
def Victoria_carriages := Euston_carriages - 15
def Waterloo_carriages := Norwich_carriages * 2

-- Theorem to prove the total number of carriages is 775
theorem total_carriages : 
  Euston_carriages + Norfolk_carriages + Norwich_carriages + Flying_Scotsman_carriages + Victoria_carriages + Waterloo_carriages = 775 :=
by sorry

end total_carriages_l263_263593


namespace quadratic_has_real_roots_l263_263666

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l263_263666


namespace zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l263_263821

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x
noncomputable def g (a x : ℝ) : ℝ := Real.log (x^2 - 2*x + a)
noncomputable def F (a x : ℝ) : ℝ := f a x + g a x

theorem zero_of_F_when_a_is_zero (x : ℝ) : a = 0 → F a x = 0 → x = 3 := by
  sorry

theorem range_of_a_if_P_and_Q (a : ℝ) :
  (∀ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), a - 1/x ≤ 0) ∧
  (∀ x : ℝ, (x^2 - 2*x + a) > 0) →
  1 < a ∧ a ≤ 2 := by
  sorry

end zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l263_263821


namespace power_of_two_l263_263496

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n) 
  (hprime_divisors : ∀ p : ℕ, p.Prime → (p ∣ b ^ m - 1 ↔ p ∣ b ^ n - 1)) : 
  ∃ k : ℕ, b + 1 = 2 ^ k :=
by
  sorry

end power_of_two_l263_263496


namespace triple_supplementary_angle_l263_263098

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l263_263098


namespace angle_triple_supplement_l263_263119

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263119


namespace unique_solution_l263_263456

noncomputable def is_solution (f : ℝ → ℝ) : Prop :=
    (∀ x, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
    (∀ x, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

theorem unique_solution (f : ℝ → ℝ) :
    is_solution f → (∀ x, x ≥ 1 → f x = x + 1) := 
sorry

end unique_solution_l263_263456


namespace greatest_multiple_of_5_l263_263985

theorem greatest_multiple_of_5 (y : ℕ) (h1 : y > 0) (h2 : y % 5 = 0) (h3 : y^3 < 8000) : y ≤ 15 :=
by {
  sorry
}

end greatest_multiple_of_5_l263_263985


namespace global_maximum_condition_l263_263941

noncomputable def f (x m : ℝ) : ℝ :=
if x ≤ m then -x^2 - 2 * x else -x + 2

theorem global_maximum_condition (m : ℝ) (h : ∃ (x0 : ℝ), ∀ (x : ℝ), f x m ≤ f x0 m) : m ≥ 1 :=
sorry

end global_maximum_condition_l263_263941


namespace team_A_more_points_than_team_B_l263_263624

theorem team_A_more_points_than_team_B :
  let number_of_teams := 8
  let number_of_remaining_games := 6
  let win_probability_each_game := (1 : ℚ) / 2
  let team_A_beats_team_B_initial : Prop := True -- Corresponding to the condition team A wins the first game
  let probability_A_wins := 1087 / 2048
  team_A_beats_team_B_initial → win_probability_each_game = 1 / 2 → number_of_teams = 8 → 
    let A_more_points_than_B := team_A_beats_team_B_initial ∧ win_probability_each_game ^ number_of_remaining_games = probability_A_wins
    A_more_points_than_B :=
  sorry

end team_A_more_points_than_team_B_l263_263624


namespace sin_75_equals_sqrt_1_plus_sin_2_equals_l263_263920

noncomputable def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)
noncomputable def sqrt_1_plus_sin_2 : ℝ := Real.sqrt (1 + Real.sin 2)

theorem sin_75_equals :
  sin_75 = (Real.sqrt 2 + Real.sqrt 6) / 4 := 
sorry

theorem sqrt_1_plus_sin_2_equals :
  sqrt_1_plus_sin_2 = Real.sin 1 + Real.cos 1 := 
sorry

end sin_75_equals_sqrt_1_plus_sin_2_equals_l263_263920


namespace sum_of_roots_of_quadratic_eq_l263_263188

theorem sum_of_roots_of_quadratic_eq :
  ∀ x : ℝ, x^2 + 2023 * x - 2024 = 0 → 
  x = -2023 := 
sorry

end sum_of_roots_of_quadratic_eq_l263_263188


namespace ab_value_l263_263744

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by 
  sorry

end ab_value_l263_263744


namespace sum_of_roots_quadratic_eq_l263_263565

theorem sum_of_roots_quadratic_eq (x₁ x₂ : ℝ) (h : x₁^2 + 2 * x₁ - 4 = 0 ∧ x₂^2 + 2 * x₂ - 4 = 0) : 
  x₁ + x₂ = -2 :=
sorry

end sum_of_roots_quadratic_eq_l263_263565


namespace min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l263_263947

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * Real.log x + b

theorem min_value_f 
  (a b : ℝ) 
  (a_non_pos : a ≤ 1) : 
  f 1 a b = (1 / 2) + b :=
sorry

theorem min_value_f_sqrt 
  (a b : ℝ) 
  (a_pos_range : 1 < a ∧ a < 4) : 
  f (Real.sqrt a) a b = (a / 2) - a * Real.log (Real.sqrt a) + b :=
sorry

theorem min_value_f_2 
  (a b : ℝ) 
  (a_ge_4 : 4 ≤ a) : 
  f 2 a b = 2 - a * Real.log 2 + b :=
sorry

theorem min_m 
  (a : ℝ) 
  (a_range : -2 ≤ a ∧ a < 0):
  ∀x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 2) ∧ (0 < x2 ∧ x2 ≤ 2) →
  ∃m : ℝ, m = 12 ∧ abs (f x1 a 0 - f x2 a 0) ≤ m ^ abs (1 / x1 - 1 / x2) :=
sorry

end min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l263_263947


namespace number_of_students_more_than_pets_l263_263623

theorem number_of_students_more_than_pets 
  (students_per_classroom pets_per_classroom num_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : pets_per_classroom = 3)
  (h3 : num_classrooms = 5) :
  (students_per_classroom * num_classrooms) - (pets_per_classroom * num_classrooms) = 85 := 
by
  sorry

end number_of_students_more_than_pets_l263_263623


namespace prop_sufficient_not_necessary_l263_263685

-- Let p and q be simple propositions.
variables (p q : Prop)

-- Define the statement to be proved: 
-- "either p or q is false" is a sufficient but not necessary condition 
-- for "not p is true".
theorem prop_sufficient_not_necessary (hpq : ¬(p ∧ q)) : ¬ p :=
sorry

end prop_sufficient_not_necessary_l263_263685


namespace probability_diff_colors_l263_263960

-- Definitions based on the conditions provided.
-- Total number of chips
def total_chips := 15

-- Individual probabilities of drawing each color first
def prob_green_first := 6 / total_chips
def prob_purple_first := 5 / total_chips
def prob_orange_first := 4 / total_chips

-- Probabilities of drawing a different color second
def prob_not_green := 9 / total_chips
def prob_not_purple := 10 / total_chips
def prob_not_orange := 11 / total_chips

-- Combined probabilities for each case
def prob_green_then_diff := prob_green_first * prob_not_green
def prob_purple_then_diff := prob_purple_first * prob_not_purple
def prob_orange_then_diff := prob_orange_first * prob_not_orange

-- Total probability of drawing two chips of different colors
def total_prob_diff_colors := prob_green_then_diff + prob_purple_then_diff + prob_orange_then_diff

-- Theorem statement to be proved
theorem probability_diff_colors : total_prob_diff_colors = 148 / 225 :=
by
  -- Proof would go here
  sorry

end probability_diff_colors_l263_263960


namespace angle_triple_supplement_l263_263102

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l263_263102


namespace value_of_expression_l263_263021

theorem value_of_expression (a b c : ℝ) (h : a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7) :
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 :=
by sorry

end value_of_expression_l263_263021


namespace Eric_return_time_l263_263176

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l263_263176


namespace area_of_region_l263_263932

noncomputable def large_circle_radius : ℝ := 40

noncomputable def small_circle_radius : ℝ := large_circle_radius / (1 + (1 / Real.sin (Real.pi / 8)))

noncomputable def K : ℝ := 
  Real.pi * large_circle_radius^2 - 8 * Real.pi * small_circle_radius^2

theorem area_of_region : ⌊K⌋ = 2191 := by
  sorry

end area_of_region_l263_263932


namespace bankers_discount_l263_263289

/-- The banker’s gain on a sum due 3 years hence at 12% per annum is Rs. 360.
   The banker's discount is to be determined. -/
theorem bankers_discount (BG BD TD : ℝ) (R : ℝ := 12 / 100) (T : ℝ := 3) 
  (h1 : BG = 360) (h2 : BG = (BD * TD) / (BD - TD)) (h3 : TD = (P * R * T) / 100) 
  (h4 : BG = (TD * R * T) / 100) :
  BD = 562.5 :=
sorry

end bankers_discount_l263_263289


namespace product_of_possible_N_l263_263164

theorem product_of_possible_N (N : ℕ) (M L : ℕ) :
  (M = L + N) →
  (M - 5 = L + N - 5) →
  (L + 3 = L + 3) →
  |(L + N - 5) - (L + 3)| = 2 →
  (10 * 6 = 60) :=
by
  sorry

end product_of_possible_N_l263_263164


namespace neither_sufficient_nor_necessary_condition_l263_263635

-- Given conditions
def p (a : ℝ) : Prop := ∃ (x y : ℝ), a * x + y + 1 = 0 ∧ a * x - y + 2 = 0
def q : Prop := ∃ (a : ℝ), a = 1

-- The proof problem
theorem neither_sufficient_nor_necessary_condition : 
  ¬ ((∀ a, p a → q) ∧ (∀ a, q → p a)) :=
sorry

end neither_sufficient_nor_necessary_condition_l263_263635


namespace f_decreasing_on_0_1_l263_263822

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem f_decreasing_on_0_1 : ∀ (x1 x2 : ℝ), (x1 ∈ Set.Ioo 0 1) → (x2 ∈ Set.Ioo 0 1) → (x1 < x2) → (f x1 < f x2) := by
  sorry

end f_decreasing_on_0_1_l263_263822


namespace quadratic_min_value_l263_263200

theorem quadratic_min_value (p q : ℝ) (h : ∀ x : ℝ, 3 * x^2 + p * x + q ≥ 4) : q = p^2 / 12 + 4 :=
sorry

end quadratic_min_value_l263_263200


namespace intersecting_lines_l263_263089

theorem intersecting_lines {c d : ℝ} 
  (h₁ : 12 = 2 * 4 + c) 
  (h₂ : 12 = -4 + d) : 
  c + d = 20 := 
sorry

end intersecting_lines_l263_263089


namespace combine_heaps_l263_263503

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l263_263503


namespace number_of_blue_balls_l263_263745

theorem number_of_blue_balls (T : ℕ) (h1 : (1 / 4) * T = green) (h2 : (1 / 8) * T = blue)
    (h3 : (1 / 12) * T = yellow) (h4 : 26 = white) (h5 : green + blue + yellow + white = T) :
    blue = 6 :=
by
  sorry

end number_of_blue_balls_l263_263745


namespace kelly_total_snacks_l263_263222

theorem kelly_total_snacks (peanuts raisins : ℝ) (h₁ : peanuts = 0.1) (h₂ : raisins = 0.4) :
  peanuts + raisins = 0.5 :=
by
  simp [h₁, h₂]
  sorry

end kelly_total_snacks_l263_263222


namespace find_lamp_cost_l263_263682

def lamp_and_bulb_costs (L B : ℝ) : Prop :=
  B = L - 4 ∧ 2 * L + 6 * B = 32

theorem find_lamp_cost : ∃ L : ℝ, ∃ B : ℝ, lamp_and_bulb_costs L B ∧ L = 7 :=
by
  sorry

end find_lamp_cost_l263_263682


namespace factorization_eq_l263_263131

theorem factorization_eq :
  ∀ (a : ℝ), a^2 + 4 * a - 21 = (a - 3) * (a + 7) := by
  intro a
  sorry

end factorization_eq_l263_263131


namespace factor_expression_l263_263314

theorem factor_expression (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) :=
by
  sorry

end factor_expression_l263_263314


namespace max_value_sqrt_sum_l263_263814

theorem max_value_sqrt_sum {x y z : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  ∃ (M : ℝ), M = (Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x))) ∧ M = Real.sqrt 2 + 1 :=
by sorry

end max_value_sqrt_sum_l263_263814


namespace cloth_coloring_problem_l263_263147

theorem cloth_coloring_problem (lengthOfCloth : ℕ) 
  (women_can_color_100m_in_1_day : 5 * 1 = 100) 
  (women_can_color_in_3_days : 6 * 3 = lengthOfCloth) : lengthOfCloth = 360 := 
sorry

end cloth_coloring_problem_l263_263147


namespace valid_differences_of_squares_l263_263233

theorem valid_differences_of_squares (n : ℕ) (h : 2 * n + 1 < 150) :
    (2 * n + 1 = 129 ∨ 2 * n +1 = 147) :=
by
  sorry

end valid_differences_of_squares_l263_263233


namespace triple_supplementary_angle_l263_263093

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l263_263093


namespace angle_triple_supplement_l263_263099

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l263_263099


namespace annual_donation_amount_l263_263374

-- Define the conditions
variables (age_start age_end : ℕ)
variables (total_donations : ℕ)

-- Define the question (prove the annual donation amount) given these conditions
theorem annual_donation_amount (h1 : age_start = 13) (h2 : age_end = 33) (h3 : total_donations = 105000) :
  total_donations / (age_end - age_start) = 5250 :=
by
   sorry

end annual_donation_amount_l263_263374


namespace find_m_for_integer_solution_l263_263798

theorem find_m_for_integer_solution :
  ∀ (m x : ℤ), (x^3 - m*x^2 + m*x - (m^2 + 1) = 0) → (m = -3 ∨ m = 0) :=
by
  sorry

end find_m_for_integer_solution_l263_263798


namespace simplify_expression_l263_263585

theorem simplify_expression (x y : ℝ) (h_x_ne_0 : x ≠ 0) (h_y_ne_0 : y ≠ 0) :
  (25*x^3*y) * (8*x*y) * (1 / (5*x*y^2)^2) = 8*x^2 / y^2 :=
by
  sorry

end simplify_expression_l263_263585


namespace arithmetic_geometric_sequence_l263_263841

theorem arithmetic_geometric_sequence
    (a : ℕ → ℕ)
    (b : ℕ → ℕ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Definition of arithmetic sequence
    (h_geom_seq : ∀ n, b (n + 1) / b n = b 1 / b 0) -- Definition of geometric sequence
    (h_a3_a11 : a 3 + a 11 = 8) -- Condition a_3 + a_11 = 8
    (h_b7_a7 : b 7 = a 7) -- Condition b_7 = a_7
    : b 6 * b 8 = 16 := -- Prove that b_6 * b_8 = 16
sorry

end arithmetic_geometric_sequence_l263_263841


namespace find_minimum_r_l263_263958

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_minimum_r (r : ℕ) (h_pos : r > 0) (h_perfect : is_perfect_square (4^3 + 4^r + 4^4)) : r = 4 :=
sorry

end find_minimum_r_l263_263958


namespace number_of_marbles_removed_and_replaced_l263_263297

def bag_contains_red_marbles (r : ℕ) : Prop := r = 12
def total_marbles (t : ℕ) : Prop := t = 48
def probability_not_red_twice (r t : ℕ) : Prop := ((t - r) / t : ℝ) * ((t - r) / t) = 9 / 16

theorem number_of_marbles_removed_and_replaced (r t : ℕ)
  (hr : bag_contains_red_marbles r)
  (ht : total_marbles t)
  (hp : probability_not_red_twice r t) :
  2 = 2 := by
  sorry

end number_of_marbles_removed_and_replaced_l263_263297


namespace max_distance_with_optimal_swapping_l263_263464

-- Define the conditions
def front_tire_lifetime : ℕ := 24000
def rear_tire_lifetime : ℕ := 36000

-- Prove that the maximum distance the car can travel given optimal tire swapping is 48,000 km
theorem max_distance_with_optimal_swapping : 
    ∃ x : ℕ, x < 24000 ∧ x < 36000 ∧ (x + min (24000 - x) (36000 - x) = 48000) :=
by {
  sorry
}

end max_distance_with_optimal_swapping_l263_263464


namespace higher_profit_percentage_l263_263160

theorem higher_profit_percentage (P : ℝ) :
  (P / 100 * 800 = 144) ↔ (P = 18) :=
by
  sorry

end higher_profit_percentage_l263_263160


namespace cookies_ratio_l263_263049

theorem cookies_ratio (total_cookies sells_mr_stone brock_buys left_cookies katy_buys : ℕ)
  (h1 : total_cookies = 5 * 12)
  (h2 : sells_mr_stone = 2 * 12)
  (h3 : brock_buys = 7)
  (h4 : left_cookies = 15)
  (h5 : total_cookies - sells_mr_stone - brock_buys - left_cookies = katy_buys) :
  katy_buys / brock_buys = 2 :=
by sorry

end cookies_ratio_l263_263049


namespace matrix_pow_minus_l263_263684

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, 2]]

theorem matrix_pow_minus : B ^ 20 - 3 * (B ^ 19) = ![![0, 4 * (2 ^ 19)], ![0, -(2 ^ 19)]] :=
by
  sorry

end matrix_pow_minus_l263_263684


namespace point_value_of_other_questions_is_4_l263_263136

theorem point_value_of_other_questions_is_4
  (total_points : ℕ)
  (total_questions : ℕ)
  (points_from_2_point_questions : ℕ)
  (other_questions : ℕ)
  (points_each_2_point_question : ℕ)
  (points_from_2_point_questions_calc : ℕ)
  (remaining_points : ℕ)
  (point_value_of_other_type : ℕ)
  : total_points = 100 →
    total_questions = 40 →
    points_each_2_point_question = 2 →
    other_questions = 10 →
    points_from_2_point_questions = 30 →
    points_from_2_point_questions_calc = points_each_2_point_question * points_from_2_point_questions →
    remaining_points = total_points - points_from_2_point_questions_calc →
    remaining_points = other_questions * point_value_of_other_type →
    point_value_of_other_type = 4 := by
  sorry

end point_value_of_other_questions_is_4_l263_263136


namespace nine_distinct_numbers_product_l263_263053

variable (a b c d e f g h i : ℕ)

theorem nine_distinct_numbers_product (ha : a = 12) (hb : b = 9) (hc : c = 2)
                                      (hd : d = 1) (he : e = 6) (hf : f = 36)
                                      (hg : g = 18) (hh : h = 4) (hi : i = 3) :
  (a * b * c = 216) ∧ (d * e * f = 216) ∧ (g * h * i = 216) ∧
  (a * d * g = 216) ∧ (b * e * h = 216) ∧ (c * f * i = 216) ∧
  (a * e * i = 216) ∧ (c * e * g = 216) :=
by
  sorry

end nine_distinct_numbers_product_l263_263053


namespace quadratic_real_roots_l263_263664

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l263_263664


namespace number_of_dolls_is_18_l263_263080

def total_toys : ℕ := 24
def fraction_action_figures : ℚ := 1 / 4
def number_action_figures : ℕ := (fraction_action_figures * total_toys).to_nat
def number_dolls : ℕ := total_toys - number_action_figures

theorem number_of_dolls_is_18 :
  number_dolls = 18 :=
by
  sorry

end number_of_dolls_is_18_l263_263080


namespace liangliang_speed_l263_263718

theorem liangliang_speed (d_initial : ℝ) (t : ℝ) (d_final : ℝ) (v_mingming : ℝ) (v_liangliang : ℝ) :
  d_initial = 3000 →
  t = 20 →
  d_final = 2900 →
  v_mingming = 80 →
  (v_liangliang = 85 ∨ v_liangliang = 75) :=
by
  sorry

end liangliang_speed_l263_263718


namespace quadratic_equation_real_roots_l263_263660

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l263_263660


namespace min_distance_in_regular_tetrahedron_l263_263961

open Finset

theorem min_distance_in_regular_tetrahedron :
  let A := (0 : ℝ, 0 : ℝ, 0 : ℝ),
      B := (2 : ℝ, 0 : ℝ, 0 : ℝ),
      C := (1 : ℝ, Real.sqrt 3, 0 : ℝ),
      D := (1 : ℝ, Real.sqrt 3 / 3, Real.sqrt (8 / 3)) in
  let P := (1/2 : ℝ, 0 : ℝ, 0 : ℝ),
      Q := (1 : ℝ, Real.sqrt 3 / 3, 2 * Real.sqrt (2 / 3) / 3) in
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2) = Real.sqrt (11/12) :=
by sorry

end min_distance_in_regular_tetrahedron_l263_263961


namespace smallest_possible_value_of_b_l263_263072

theorem smallest_possible_value_of_b (a b x : ℕ) (h_pos_x : 0 < x)
  (h_gcd : Nat.gcd a b = x + 7)
  (h_lcm : Nat.lcm a b = x * (x + 7))
  (h_a : a = 56)
  (h_x : x = 21) :
  b = 294 := by
  sorry

end smallest_possible_value_of_b_l263_263072


namespace split_stones_l263_263513

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l263_263513


namespace intersection_with_complement_l263_263727

-- Definitions for the universal set and set A
def U : Set ℝ := Set.univ

def A : Set ℝ := { -1, 0, 1 }

-- Definition for set B using the given condition
def B : Set ℝ := { x : ℝ | (x - 2) / (x + 1) > 0 }

-- Definition for the complement of B
def B_complement : Set ℝ := { x : ℝ | -1 <= x ∧ x <= 0 }

-- Theorem stating the intersection of A and the complement of B equals {-1, 0, 1}
theorem intersection_with_complement : 
  A ∩ B_complement = { -1, 0, 1 } :=
by
  sorry

end intersection_with_complement_l263_263727


namespace gooGoo_buttons_l263_263242

theorem gooGoo_buttons (num_3_button_shirts : ℕ) (num_5_button_shirts : ℕ)
  (buttons_per_3_button_shirt : ℕ) (buttons_per_5_button_shirt : ℕ)
  (order_quantity : ℕ)
  (h1 : num_3_button_shirts = order_quantity)
  (h2 : num_5_button_shirts = order_quantity)
  (h3 : buttons_per_3_button_shirt = 3)
  (h4 : buttons_per_5_button_shirt = 5)
  (h5 : order_quantity = 200) :
  num_3_button_shirts * buttons_per_3_button_shirt + num_5_button_shirts * buttons_per_5_button_shirt = 1600 := by
  have h6 : 200 * 3 = 600 := by norm_num
  have h7 : 200 * 5 = 1000 := by norm_num
  have h8 : 600 + 1000 = 1600 := by norm_num
  rw [h1, h2, h3, h4, h5]
  rw [h6, h7]
  exact h8

end gooGoo_buttons_l263_263242


namespace total_repairs_cost_eq_l263_263052

-- Assume the initial cost of the scooter is represented by a real number C.
variable (C : ℝ)

-- Given conditions
def spent_on_first_repair := 0.05 * C
def spent_on_second_repair := 0.10 * C
def spent_on_third_repair := 0.07 * C

-- Total repairs expenditure
def total_repairs := spent_on_first_repair C + spent_on_second_repair C + spent_on_third_repair C

-- Selling price and profit
def selling_price := 1.25 * C
def profit := 1500
def profit_calc := selling_price C - (C + total_repairs C)

-- Statement to be proved: The total repairs is equal to $11,000.
theorem total_repairs_cost_eq : total_repairs 50000 = 11000 := by
  sorry

end total_repairs_cost_eq_l263_263052


namespace inequality_solution_l263_263626

theorem inequality_solution (x : ℝ) :
  (0 < x ∧ x ≤ 5 / 6 ∨ 2 < x) ↔ 
  ((2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2) :=
by
  sorry

end inequality_solution_l263_263626


namespace probability_interval_contains_p_l263_263195

theorem probability_interval_contains_p (P_A P_B p : ℝ) 
  (hA : P_A = 5 / 6) 
  (hB : P_B = 3 / 4) 
  (hp : p = P_A + P_B - 1) : 
  (5 / 12 ≤ p ∧ p ≤ 3 / 4) :=
by
  -- The proof is skipped by sorry as per the instructions.
  sorry

end probability_interval_contains_p_l263_263195


namespace find_first_term_l263_263569

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l263_263569


namespace A_eq_B_l263_263028

variables (α : Type) (Q : α → Prop)
variables (A B C : α → Prop)

-- Conditions
-- 1. For the questions where both B and C answered "yes", A also answered "yes".
axiom h1 : ∀ q, B q ∧ C q → A q
-- 2. For the questions where A answered "yes", B also answered "yes".
axiom h2 : ∀ q, A q → B q
-- 3. For the questions where B answered "yes", at least one of A and C answered "yes".
axiom h3 : ∀ q, B q → (A q ∨ C q)

-- Prove that A and B gave the same answer to all questions
theorem A_eq_B : ∀ q, A q ↔ B q :=
sorry

end A_eq_B_l263_263028


namespace construction_days_behind_without_additional_workers_l263_263420

-- Definitions for initial and additional workers and their respective efficiencies and durations.
def initial_workers : ℕ := 100
def initial_worker_efficiency : ℕ := 1
def total_days : ℕ := 150

def additional_workers_1 : ℕ := 50
def additional_worker_efficiency_1 : ℕ := 2
def additional_worker_start_day_1 : ℕ := 30

def additional_workers_2 : ℕ := 25
def additional_worker_efficiency_2 : ℕ := 3
def additional_worker_start_day_2 : ℕ := 45

def additional_workers_3 : ℕ := 15
def additional_worker_efficiency_3 : ℕ := 4
def additional_worker_start_day_3 : ℕ := 75

-- Define the total additional work units done by the extra workers.
def total_additional_work_units : ℕ := 
  (additional_workers_1 * additional_worker_efficiency_1 * (total_days - additional_worker_start_day_1)) +
  (additional_workers_2 * additional_worker_efficiency_2 * (total_days - additional_worker_start_day_2)) +
  (additional_workers_3 * additional_worker_efficiency_3 * (total_days - additional_worker_start_day_3))

-- Define the days the initial workers would have taken to do the additional work.
def initial_days_for_additional_work : ℕ := 
  (total_additional_work_units + (initial_workers * initial_worker_efficiency) - 1) / (initial_workers * initial_worker_efficiency)

-- Define the total days behind schedule.
def days_behind_schedule : ℕ := (total_days + initial_days_for_additional_work) - total_days

-- Define the theorem to prove.
theorem construction_days_behind_without_additional_workers : days_behind_schedule = 244 := 
  by 
  -- This translates to manually verifying the outcome.
  -- A detailed proof can be added later.
  sorry

end construction_days_behind_without_additional_workers_l263_263420


namespace johnny_red_pencils_l263_263355

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end johnny_red_pencils_l263_263355


namespace amount_of_salmon_sold_first_week_l263_263085

-- Define the conditions
def fish_sold_in_two_weeks (x : ℝ) := x + 3 * x = 200

-- Define the theorem we want to prove
theorem amount_of_salmon_sold_first_week (x : ℝ) (h : fish_sold_in_two_weeks x) : x = 50 :=
by
  sorry

end amount_of_salmon_sold_first_week_l263_263085


namespace evaluate_expression_l263_263796

theorem evaluate_expression : (532 * 532) - (531 * 533) = 1 := by
  sorry

end evaluate_expression_l263_263796


namespace tangent_line_correct_l263_263627

noncomputable def f (x : ℝ) : ℝ := exp (-5 * x) + 2
def point : ℝ × ℝ := (0, 3)
def tangent_line (x : ℝ) : ℝ := -5 * x + 3

theorem tangent_line_correct : 
    ∀ x y, (y = f x) → x = 0 → y = 3 → (∀ t, tangent_line t = -5 * t + 3) := 
by
  sorry

end tangent_line_correct_l263_263627


namespace greatest_possible_n_l263_263020

theorem greatest_possible_n (n : ℤ) (h1 : 102 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

end greatest_possible_n_l263_263020


namespace split_stones_l263_263512

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l263_263512


namespace ellipse_sum_l263_263773

theorem ellipse_sum (h k a b : ℝ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 6) (b_val : b = 2) : h + k + a + b = 6 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l263_263773


namespace deepak_present_age_l263_263287

theorem deepak_present_age (x : ℕ) (h1 : ∀ current_age_rahul current_age_deepak, 
  4 * x = current_age_rahul ∧ 3 * x = current_age_deepak)
  (h2 : ∀ current_age_rahul, current_age_rahul + 6 = 22) :
  3 * x = 12 :=
by
  have h3 : 4 * x + 6 = 22 := h2 (4 * x)
  linarith

end deepak_present_age_l263_263287


namespace total_red_pencils_l263_263358

theorem total_red_pencils (packs : ℕ) (normal_pencil_per_pack : ℕ) (extra_packs : ℕ) (extra_pencils_per_pack : ℕ) :
  packs = 15 →
  normal_pencil_per_pack = 1 →
  extra_packs = 3 →
  extra_pencils_per_pack = 2 →
  packs * normal_pencil_per_pack + extra_packs * extra_pencils_per_pack = 21 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_red_pencils_l263_263358


namespace inequality_range_l263_263938

theorem inequality_range (a : ℝ) : (-1 < a ∧ a ≤ 0) → ∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0 :=
by
  intro ha
  sorry

end inequality_range_l263_263938


namespace books_before_addition_l263_263221

-- Let b be the initial number of books on the shelf
variable (b : ℕ)

theorem books_before_addition (h : b + 10 = 19) : b = 9 := by
  sorry

end books_before_addition_l263_263221


namespace rhombus_perimeter_l263_263423

-- Define the lengths of the diagonals
def d1 : ℝ := 5  -- Length of the first diagonal
def d2 : ℝ := 12 -- Length of the second diagonal

-- Calculate the perimeter and state the theorem
theorem rhombus_perimeter : ((d1 / 2)^2 + (d2 / 2)^2).sqrt * 4 = 26 := by
  -- Sorry is placed here to denote the proof
  sorry

end rhombus_perimeter_l263_263423


namespace number_of_schools_l263_263883

def yellow_balloons := 3414
def additional_black_balloons := 1762
def balloons_per_school := 859

def black_balloons := yellow_balloons + additional_black_balloons
def total_balloons := yellow_balloons + black_balloons

theorem number_of_schools : total_balloons / balloons_per_school = 10 :=
by
  sorry

end number_of_schools_l263_263883


namespace albert_number_l263_263432

theorem albert_number :
  ∃ (n : ℕ), (1 / (n : ℝ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) ∧ 
             ∃ m : ℕ, (1 / (m : ℝ) + 1 / 2 = 1 / 3 + 2 / (m + 1)) ∧ m ≠ n :=
sorry

end albert_number_l263_263432


namespace find_fifth_month_sale_l263_263151

theorem find_fifth_month_sale (s1 s2 s3 s4 s6 A : ℝ) (h1 : s1 = 800) (h2 : s2 = 900) (h3 : s3 = 1000) (h4 : s4 = 700) (h5 : s6 = 900) (h6 : A = 850) :
  ∃ s5 : ℝ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = A ∧ s5 = 800 :=
by
  sorry

end find_fifth_month_sale_l263_263151


namespace solve_eq1_solve_eq2_l263_263954

theorem solve_eq1 : ∀ (x : ℚ), (3 / 5 - 5 / 8 * x = 2 / 5) → (x = 8 / 25) := by
  intro x
  intro h
  sorry

theorem solve_eq2 : ∀ (x : ℚ), (7 * (x - 2) = 8 * (x - 4)) → (x = 18) := by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l263_263954


namespace concert_duration_is_805_l263_263421

def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

def total_duration (hours : ℕ) (extra_minutes : ℕ) : ℕ :=
  hours_to_minutes hours + extra_minutes

theorem concert_duration_is_805 : total_duration 13 25 = 805 :=
by
  -- Proof skipped
  sorry

end concert_duration_is_805_l263_263421


namespace atlantic_call_charge_l263_263887

theorem atlantic_call_charge :
  let united_base := 6.00
  let united_per_min := 0.25
  let atlantic_base := 12.00
  let same_bill_minutes := 120
  let atlantic_total (charge_per_minute : ℝ) := atlantic_base + charge_per_minute * same_bill_minutes
  let united_total := united_base + united_per_min * same_bill_minutes
  united_total = atlantic_total 0.20 :=
by
  sorry

end atlantic_call_charge_l263_263887


namespace find_f2_l263_263810

namespace ProofProblem

-- Define the polynomial function f
def f (x a b : ℤ) : ℤ := x^5 + a * x^3 + b * x - 8

-- Conditions given in the problem
axiom f_neg2 : ∃ a b : ℤ, f (-2) a b = 10

-- Define the theorem statement
theorem find_f2 : ∃ a b : ℤ, f 2 a b = -26 :=
by
  sorry

end ProofProblem

end find_f2_l263_263810


namespace intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l263_263425

variables (x y z : ℕ)

-- Conditions
axiom condition_1 : ∀ (t: ℕ), t = (6 : ℕ) → y * z = 6 * (y - x)
axiom condition_2 : ∀ (t: ℕ), t = (3 : ℕ) → y * z = 3 * (y + x)

-- Proof statements
theorem intervals_between_trolleybuses : z = 4 :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

theorem sportsman_slower_than_trolleybus : y = 3 * x :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

end intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l263_263425


namespace quadratic_equation_real_roots_l263_263661

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l263_263661


namespace sum_of_solutions_eq_3_l263_263187

theorem sum_of_solutions_eq_3 (x y : ℝ) (h1 : x * y = 1) (h2 : x + y = 3) :
  x + y = 3 := sorry

end sum_of_solutions_eq_3_l263_263187


namespace split_piles_equiv_single_stone_heaps_l263_263500

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l263_263500


namespace trinomials_real_roots_inequality_l263_263448

theorem trinomials_real_roots_inequality :
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ¬ (∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q))) >
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q)) :=
sorry

end trinomials_real_roots_inequality_l263_263448


namespace junghyeon_stickers_l263_263069

def total_stickers : ℕ := 25
def junghyeon_sticker_count (yejin_stickers : ℕ) : ℕ := 2 * yejin_stickers + 1

theorem junghyeon_stickers (yejin_stickers : ℕ) (h : yejin_stickers + junghyeon_sticker_count yejin_stickers = total_stickers) : 
  junghyeon_sticker_count yejin_stickers = 17 :=
  by
  sorry

end junghyeon_stickers_l263_263069


namespace find_a_l263_263284

theorem find_a (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 89) : a = 34 :=
sorry

end find_a_l263_263284


namespace rope_cut_number_not_8_l263_263396

theorem rope_cut_number_not_8 (l : ℝ) (h1 : (1 : ℝ) % l = 0) (h2 : (2 : ℝ) % l = 0) (h3 : (3 / l) ≠ 8) : False :=
by
  sorry

end rope_cut_number_not_8_l263_263396


namespace rectangular_garden_width_l263_263871

theorem rectangular_garden_width (w : ℕ) (h1 : ∃ l : ℕ, l = 3 * w) (h2 : w * (3 * w) = 507) : w = 13 := 
by 
  sorry

end rectangular_garden_width_l263_263871


namespace product_of_possible_values_of_N_l263_263163

theorem product_of_possible_values_of_N (M L N : ℝ) (h1 : M = L + N) (h2 : M - 5 = (L + N) - 5) (h3 : L + 3 = L + 3) (h4 : |(L + N - 5) - (L + 3)| = 2) : 10 * 6 = 60 := by
  sorry

end product_of_possible_values_of_N_l263_263163


namespace number_equation_l263_263303

variable (x : ℝ)

theorem number_equation :
  5 * x - 2 * x = 10 :=
sorry

end number_equation_l263_263303


namespace min_value_geometric_sequence_l263_263226

theorem min_value_geometric_sequence (a_2 a_3 : ℝ) (r : ℝ) 
(h_a2 : a_2 = 2 * r) (h_a3 : a_3 = 2 * r^2) : 
  (6 * a_2 + 7 * a_3) = -18 / 7 :=
by
  sorry

end min_value_geometric_sequence_l263_263226


namespace angle_triple_supplementary_l263_263108

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l263_263108


namespace train_length_l263_263913

/-- Given a train traveling at 72 km/hr passing a pole in 8 seconds,
     prove that the length of the train in meters is 160. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (speed_m_s : ℝ) (distance_m : ℝ) :
  speed_kmh = 72 → 
  time_s = 8 → 
  speed_m_s = (speed_kmh * 1000) / 3600 → 
  distance_m = speed_m_s * time_s → 
  distance_m = 160 :=
by
  sorry

end train_length_l263_263913


namespace minimum_value_expression_l263_263733

noncomputable def expr (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) +
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3)

theorem minimum_value_expression : (∃ x y : ℝ, expr x y = 3*Real.sqrt 6 + 4*Real.sqrt 2) :=
sorry

end minimum_value_expression_l263_263733


namespace degree_f_x2_g_x3_l263_263548

open Polynomial

noncomputable def degree_of_composite_polynomials (f g : Polynomial ℝ) : ℕ :=
  let f_degree := Polynomial.degree f
  let g_degree := Polynomial.degree g
  match (f_degree, g_degree) with
  | (some 3, some 6) => 24
  | _ => 0

theorem degree_f_x2_g_x3 (f g : Polynomial ℝ) (h_f : Polynomial.degree f = 3) (h_g : Polynomial.degree g = 6) :
  Polynomial.degree (Polynomial.comp f (X^2) * Polynomial.comp g (X^3)) = 24 := by
  -- content Logic Here
  sorry

end degree_f_x2_g_x3_l263_263548


namespace triple_supplementary_angle_l263_263095

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l263_263095


namespace quiz_true_false_questions_l263_263159

theorem quiz_true_false_questions (n : ℕ) 
  (h1 : 2^n - 2 ≠ 0) 
  (h2 : (2^n - 2) * 16 = 224) : 
  n = 4 := 
sorry

end quiz_true_false_questions_l263_263159


namespace total_cost_chairs_l263_263759

def living_room_chairs : Nat := 3
def kitchen_chairs : Nat := 6
def dining_room_chairs : Nat := 8
def outdoor_patio_chairs : Nat := 12

def living_room_price : Nat := 75
def kitchen_price : Nat := 50
def dining_room_price : Nat := 100
def outdoor_patio_price : Nat := 60

theorem total_cost_chairs : 
  living_room_chairs * living_room_price + 
  kitchen_chairs * kitchen_price + 
  dining_room_chairs * dining_room_price + 
  outdoor_patio_chairs * outdoor_patio_price = 2045 := by
  sorry

end total_cost_chairs_l263_263759


namespace correct_equations_l263_263580

theorem correct_equations (m n : ℕ) (h1 : n = 4 * m - 2) (h2 : n = 2 * m + 58) :
  (4 * m - 2 = 2 * m + 58 ∨ (n + 2) / 4 = (n - 58) / 2) :=
by
  sorry

end correct_equations_l263_263580


namespace area_of_quadrilateral_ABDE_l263_263309

-- Definitions for the given problem
variable (AB CE AC DE : ℝ)
variable (parABCE parACDE : Prop)
variable (areaCOD : ℝ)

-- Lean 4 statement for the proof problem
theorem area_of_quadrilateral_ABDE
  (h1 : parABCE)
  (h2 : parACDE)
  (h3 : AB = 5)
  (h4 : AC = 5)
  (h5 : CE = 10)
  (h6 : DE = 10)
  (h7 : areaCOD = 10)
  : (AB + AC + CE + DE) / 2 + areaCOD = 52.5 := 
sorry

end area_of_quadrilateral_ABDE_l263_263309


namespace number_of_dogs_total_l263_263165

theorem number_of_dogs_total
  (A : Finset ℕ) (B : Finset ℕ) (C : Finset ℕ)
  (n_fetch : A.card = 40)
  (n_jump : B.card = 35)
  (n_playdead : C.card = 22)
  (n_fetch_jump : (A ∩ B).card = 14)
  (n_jump_playdead : (B ∩ C).card = 10)
  (n_fetch_playdead : (A ∩ C).card = 16)
  (n_all_three : (A ∩ B ∩ C).card = 6)
  (n_none : 12)
  : A.card + B.card + C.card - (A ∩ B).card - (B ∩ C).card - (A ∩ C).card + (A ∩ B ∩ C).card + n_none = 75 := by
  sorry

end number_of_dogs_total_l263_263165


namespace problem_l263_263241

variable (R S : Prop)

theorem problem (h1 : R → S) :
  ((¬S → ¬R) ∧ (¬R ∨ S)) :=
by
  sorry

end problem_l263_263241


namespace julies_balls_after_1729_steps_l263_263491

-- Define the process described
def increment_base_8 (n : ℕ) : List ℕ := 
by
  if n = 0 then
    exact [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 8) (n % 8 :: acc)
    exact loop n []

-- Define the total number of balls after 'steps' steps
def julies_total_balls (steps : ℕ) : ℕ :=
by 
  exact (increment_base_8 steps).sum

theorem julies_balls_after_1729_steps : julies_total_balls 1729 = 7 :=
by
  sorry

end julies_balls_after_1729_steps_l263_263491


namespace joan_games_attended_l263_263038
-- Mathematical definitions based on the provided conditions

def total_games_played : ℕ := 864
def games_missed_by_Joan : ℕ := 469

-- Theorem statement
theorem joan_games_attended : total_games_played - games_missed_by_Joan = 395 :=
by
  -- Proof omitted
  sorry

end joan_games_attended_l263_263038


namespace toothpicks_total_l263_263729

-- Definitions based on the conditions
def grid_length : ℕ := 50
def grid_width : ℕ := 40

-- Mathematical statement to prove
theorem toothpicks_total : (grid_length + 1) * grid_width + (grid_width + 1) * grid_length = 4090 := by
  sorry

end toothpicks_total_l263_263729


namespace bob_monthly_hours_l263_263453

noncomputable def total_hours_in_month : ℝ :=
  let daily_hours := 10
  let weekly_days := 5
  let weeks_in_month := 4.33
  daily_hours * weekly_days * weeks_in_month

theorem bob_monthly_hours :
  total_hours_in_month = 216.5 :=
by
  sorry

end bob_monthly_hours_l263_263453


namespace one_third_sugar_l263_263149

theorem one_third_sugar (sugar : ℚ) (h : sugar = 3 + 3 / 4) : sugar / 3 = 1 + 1 / 4 :=
by sorry

end one_third_sugar_l263_263149


namespace N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l263_263533

-- Define the number with 1986 ones
def N : ℕ := (10^1986 - 1) / 9

-- Definition of having at least n distinct divisors
def has_at_least_n_distinct_divisors (num : ℕ) (n : ℕ) :=
  ∃ (divisors : Finset ℕ), divisors.card ≥ n ∧ ∀ d ∈ divisors, d ∣ num

theorem N_has_at_least_8_distinct_divisors :
  has_at_least_n_distinct_divisors N 8 :=
sorry

theorem N_has_at_least_32_distinct_divisors :
  has_at_least_n_distinct_divisors N 32 :=
sorry


end N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l263_263533


namespace intersection_M_N_l263_263670

open Real

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 2 - abs x > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
sorry

end intersection_M_N_l263_263670


namespace fraction_of_total_cost_for_raisins_l263_263447

-- Define variables and constants
variable (R : ℝ) -- cost of a pound of raisins

-- Define the conditions as assumptions
variable (cost_of_nuts : ℝ := 4 * R)
variable (cost_of_dried_berries : ℝ := 2 * R)

variable (total_cost : ℝ := 3 * R + 4 * cost_of_nuts + 2 * cost_of_dried_berries)
variable (cost_of_raisins : ℝ := 3 * R)

-- Main statement that we want to prove
theorem fraction_of_total_cost_for_raisins :
  cost_of_raisins / total_cost = 3 / 23 := by
  sorry

end fraction_of_total_cost_for_raisins_l263_263447


namespace expansion_eq_l263_263797

variable (x y : ℝ) -- x and y are real numbers
def a := 5
def b := 3
def c := 15

theorem expansion_eq : (x + a) * (b * y + c) = 3 * x * y + 15 * x + 15 * y + 75 := by 
  sorry

end expansion_eq_l263_263797


namespace angle_triple_supplement_l263_263123

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263123


namespace prove_a_eq_b_l263_263645

theorem prove_a_eq_b 
  (p q a b : ℝ) 
  (h1 : p + q = 1) 
  (h2 : p * q ≠ 0) 
  (h3 : p / a + q / b = 1 / (p * a + q * b)) : 
  a = b := 
sorry

end prove_a_eq_b_l263_263645


namespace smallest_two_digit_number_l263_263307

theorem smallest_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100)
  (h2 : ∃ k : ℕ, (N - (N / 10 + (N % 10) * 10)) = k ∧ k > 0 ∧ (∃ m : ℕ, k = m * m))
  : N = 90 := 
sorry

end smallest_two_digit_number_l263_263307


namespace angle_triple_supplement_l263_263120

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263120


namespace last_four_digits_of_5_pow_2016_l263_263370

theorem last_four_digits_of_5_pow_2016 :
  (5^2016) % 10000 = 625 :=
by
  -- Establish periodicity of last four digits in powers of 5
  sorry

end last_four_digits_of_5_pow_2016_l263_263370


namespace equation_1_solutions_equation_2_solutions_l263_263709

-- Equation 1: Proving solutions for (x+8)(x+1) = -12
theorem equation_1_solutions (x : ℝ) :
  (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 :=
sorry

-- Equation 2: Proving solutions for (2x-3)^2 = 5(2x-3)
theorem equation_2_solutions (x : ℝ) :
  (2 * x - 3) ^ 2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
sorry

end equation_1_solutions_equation_2_solutions_l263_263709


namespace circle_center_radius_l263_263557

theorem circle_center_radius :
  ∀ x y : ℝ,
  x^2 + y^2 + 4 * x - 6 * y - 3 = 0 →
  (∃ h k r : ℝ, (x + h)^2 + (y + k)^2 = r^2 ∧ h = -2 ∧ k = 3 ∧ r = 4) :=
by
  intros x y hxy
  sorry

end circle_center_radius_l263_263557


namespace solve_quadratic_eq1_solve_quadratic_eq2_l263_263710

-- Define the statement for the first problem
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 49 = 0 → x = 7 ∨ x = -7 :=
by
  sorry

-- Define the statement for the second problem
theorem solve_quadratic_eq2 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 → x = 4 ∨ x = -6 :=
by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l263_263710


namespace percentage_paid_to_x_l263_263262

theorem percentage_paid_to_x (X Y : ℕ) (h₁ : Y = 350) (h₂ : X + Y = 770) :
  (X / Y) * 100 = 120 :=
by
  sorry

end percentage_paid_to_x_l263_263262


namespace ellipse_condition_l263_263483

theorem ellipse_condition (k : ℝ) : 
  (k > 1 ↔ 
  (k - 1 > 0 ∧ k + 1 > 0 ∧ k - 1 ≠ k + 1)) :=
by sorry

end ellipse_condition_l263_263483


namespace election_votes_l263_263485

theorem election_votes (T : ℝ) (Vf Va Vn : ℝ)
  (h1 : Va = 0.375 * T)
  (h2 : Vn = 0.125 * T)
  (h3 : Vf = Va + 78)
  (h4 : T = Vf + Va + Vn) :
  T = 624 :=
by
  sorry

end election_votes_l263_263485


namespace boat_speed_upstream_l263_263906

noncomputable def V_b : ℝ := 11
noncomputable def V_down : ℝ := 15
noncomputable def V_s : ℝ := V_down - V_b
noncomputable def V_up : ℝ := V_b - V_s

theorem boat_speed_upstream :
  V_up = 7 := by
  sorry

end boat_speed_upstream_l263_263906


namespace smallest_integer_representable_l263_263269

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l263_263269


namespace value_of_x_plus_y_l263_263653

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem value_of_x_plus_y
  (x y : ℝ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x * y = 10)
  (h4 : x^(lg x) * y^(lg y) ≥ 10) :
  x + y = 11 :=
  sorry

end value_of_x_plus_y_l263_263653


namespace Carl_avg_gift_bags_l263_263168

theorem Carl_avg_gift_bags :
  ∀ (known expected extravagant remaining : ℕ), 
  known = 50 →
  expected = 40 →
  extravagant = 10 →
  remaining = 60 →
  (known + expected) - extravagant - remaining = 30 := by
  intros
  sorry

end Carl_avg_gift_bags_l263_263168


namespace people_left_gym_l263_263442

theorem people_left_gym (initial : ℕ) (additional : ℕ) (current : ℕ) (H1 : initial = 16) (H2 : additional = 5) (H3 : current = 19) : (initial + additional - current) = 2 :=
by
  sorry

end people_left_gym_l263_263442


namespace average_percentage_popped_average_percentage_kernels_l263_263527

theorem average_percentage_popped (
  pops1 total1 pops2 total2 pops3 total3 : ℕ
) (h1 : pops1 = 60) (h2 : total1 = 75) 
  (h3 : pops2 = 42) (h4 : total2 = 50) 
  (h5 : pops3 = 82) (h6 : total3 = 100) : 
  ((pops1 : ℝ) / total1) * 100 + ((pops2 : ℝ) / total2) * 100 + ((pops3 : ℝ) / total3) * 100 = 246 := 
by
  sorry

theorem average_percentage_kernels (pops1 total1 pops2 total2 pops3 total3 : ℕ)
  (h1 : pops1 = 60) (h2 : total1 = 75)
  (h3 : pops2 = 42) (h4 : total2 = 50)
  (h5 : pops3 = 82) (h6 : total3 = 100) :
  ((
      (((pops1 : ℝ) / total1) * 100) + 
       (((pops2 : ℝ) / total2) * 100) + 
       (((pops3 : ℝ) / total3) * 100)
    ) / 3 = 82) :=
by
  sorry

end average_percentage_popped_average_percentage_kernels_l263_263527


namespace quadratic_has_real_roots_l263_263668

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l263_263668


namespace percentage_y_less_than_x_l263_263305

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 4 * y) : (x - y) / x * 100 = 75 := by
  sorry

end percentage_y_less_than_x_l263_263305


namespace isosceles_triangle_vertex_angle_l263_263567

theorem isosceles_triangle_vertex_angle (a b : ℕ) (h : a = 2 * b) 
  (h1 : a + b + b = 180): a = 90 ∨ a = 36 :=
by
  sorry

end isosceles_triangle_vertex_angle_l263_263567


namespace powerFunctionAtPoint_l263_263816

def powerFunction (n : ℕ) (x : ℕ) : ℕ := x ^ n

theorem powerFunctionAtPoint (n : ℕ) (h : powerFunction n 2 = 8) : powerFunction n 3 = 27 :=
  by {
    sorry
}

end powerFunctionAtPoint_l263_263816


namespace johns_average_speed_l263_263846

-- Definitions based on conditions
def cycling_distance_uphill := 3 -- in km
def cycling_time_uphill := 45 / 60 -- in hr (45 minutes)

def cycling_distance_downhill := 3 -- in km
def cycling_time_downhill := 15 / 60 -- in hr (15 minutes)

def walking_distance := 2 -- in km
def walking_time := 20 / 60 -- in hr (20 minutes)

-- Definition for total distance traveled
def total_distance := cycling_distance_uphill + cycling_distance_downhill + walking_distance

-- Definition for total time spent traveling
def total_time := cycling_time_uphill + cycling_time_downhill + walking_time

-- Definition for average speed
def average_speed := total_distance / total_time

-- Proof statement
theorem johns_average_speed : average_speed = 6 := by
  sorry

end johns_average_speed_l263_263846


namespace smallest_base10_integer_l263_263272

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l263_263272


namespace elvie_age_l263_263078

variable (E : ℕ) (A : ℕ)

theorem elvie_age (hA : A = 11) (h : E + A + (E * A) = 131) : E = 10 :=
by
  sorry

end elvie_age_l263_263078


namespace positive_integers_square_of_sum_of_digits_l263_263928

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem positive_integers_square_of_sum_of_digits :
  ∀ (n : ℕ), (n > 0) → (n = sum_of_digits n ^ 2) → (n = 1 ∨ n = 81) :=
by
  sorry

end positive_integers_square_of_sum_of_digits_l263_263928


namespace maximum_distance_with_tire_switching_l263_263463

theorem maximum_distance_with_tire_switching :
  ∀ (x y : ℕ),
    (∀ (front rear : ℕ), (front = 24000) ∧ (rear = 36000)) →
    x < 24000 →
    (y = min (24000 - x) (36000 - x)) →
    (x + y = 48000) :=
by {
  intros x y h_front_rear x_lt y_def,
  obtain ⟨front_eq, rear_eq⟩ := h_front_rear,
  rw [front_eq, rear_eq] at *,
  cases x_lt,
  sorry
}

end maximum_distance_with_tire_switching_l263_263463


namespace charlie_age_when_jenny_twice_as_old_as_bobby_l263_263220

-- Conditions as Definitions
def ageDifferenceJennyCharlie : ℕ := 5
def ageDifferenceCharlieBobby : ℕ := 3

-- Problem Statement as a Theorem
theorem charlie_age_when_jenny_twice_as_old_as_bobby (j c b : ℕ) 
  (H1 : j = c + ageDifferenceJennyCharlie) 
  (H2 : c = b + ageDifferenceCharlieBobby) : 
  j = 2 * b → c = 11 :=
by
  sorry

end charlie_age_when_jenny_twice_as_old_as_bobby_l263_263220


namespace six_people_acquaintance_or_strangers_l263_263531

theorem six_people_acquaintance_or_strangers (p : Fin 6 → Prop) :
  ∃ (A B C : Fin 6), (p A ∧ p B ∧ p C) ∨ (¬p A ∧ ¬p B ∧ ¬p C) :=
sorry

end six_people_acquaintance_or_strangers_l263_263531


namespace ln_1_2_over_6_gt_e_l263_263294

theorem ln_1_2_over_6_gt_e :
  let x := 1.2
  let exp1 := x^6
  let exp2 := (1.44)^2 * 1.44
  let final_val := 2.0736 * 1.44
  final_val > 2.718 :=
by {
  sorry
}

end ln_1_2_over_6_gt_e_l263_263294


namespace evaluate_fraction_l263_263183

theorem evaluate_fraction : 3 / (2 - 3 / 4) = 12 / 5 := by
  sorry

end evaluate_fraction_l263_263183


namespace alice_lawn_area_l263_263158

theorem alice_lawn_area (posts : ℕ) (distance : ℕ) (ratio : ℕ) : 
    posts = 24 → distance = 5 → ratio = 3 → 
    ∃ (short_side long_side : ℕ), 
        (2 * (short_side + long_side - 2) = posts) ∧
        (long_side = ratio * short_side) ∧
        (distance * (short_side - 1) * distance * (long_side - 1) = 825) :=
by
  intros h_posts h_distance h_ratio
  sorry

end alice_lawn_area_l263_263158


namespace distance_relationship_l263_263003

noncomputable def plane_parallel (α β : Type) : Prop := sorry
noncomputable def line_in_plane (m : Type) (α : Type) : Prop := sorry
noncomputable def point_on_line (A : Type) (m : Type) : Prop := sorry
noncomputable def distance (A B : Type) : ℝ := sorry
noncomputable def distance_point_to_line (A : Type) (n : Type) : ℝ := sorry
noncomputable def distance_between_lines (m n : Type) : ℝ := sorry

variables (α β m n A B : Type)
variables (a b c : ℝ)

axiom plane_parallel_condition : plane_parallel α β
axiom line_m_in_alpha : line_in_plane m α
axiom line_n_in_beta : line_in_plane n β
axiom point_A_on_m : point_on_line A m
axiom point_B_on_n : point_on_line B n
axiom distance_a : a = distance A B
axiom distance_b : b = distance_point_to_line A n
axiom distance_c : c = distance_between_lines m n

theorem distance_relationship : c ≤ b ∧ b ≤ a := by
  sorry

end distance_relationship_l263_263003


namespace factorize_expression_l263_263454

theorem factorize_expression (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 :=
by
  sorry

end factorize_expression_l263_263454


namespace simplify_and_evaluate_l263_263706

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l263_263706


namespace find_f_of_3pi_by_4_l263_263333

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)

theorem find_f_of_3pi_by_4 : f (3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end find_f_of_3pi_by_4_l263_263333


namespace triangle_area_ratio_l263_263395

theorem triangle_area_ratio (x y : ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  let A_area := (1/2) * (y/n) * (x/2)
  let B_area := (1/2) * (x/m) * (y/2)
  A_area / B_area = m / n := by
  sorry

end triangle_area_ratio_l263_263395


namespace coefficient_x3_l263_263843

open Polynomial

noncomputable def polynomial : Polynomial ℤ := (2 * X + 1) * (X - 2) * (X + 3) * (X - 4)

theorem coefficient_x3 : coeff polynomial 3 = -5 :=
sorry

end coefficient_x3_l263_263843


namespace find_b_l263_263828

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 :=
by
  sorry

end find_b_l263_263828


namespace total_red_pencils_l263_263359

theorem total_red_pencils (packs : ℕ) (normal_pencil_per_pack : ℕ) (extra_packs : ℕ) (extra_pencils_per_pack : ℕ) :
  packs = 15 →
  normal_pencil_per_pack = 1 →
  extra_packs = 3 →
  extra_pencils_per_pack = 2 →
  packs * normal_pencil_per_pack + extra_packs * extra_pencils_per_pack = 21 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_red_pencils_l263_263359


namespace brick_height_l263_263417

theorem brick_height (H : ℝ) 
    (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
    (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℝ)
    (volume_wall: wall_length = 900 ∧ wall_width = 500 ∧ wall_height = 1850)
    (volume_brick: brick_length = 21 ∧ brick_width = 10)
    (num_bricks_value: num_bricks = 4955.357142857142) :
    (H = 0.8) :=
by {
  sorry
}

end brick_height_l263_263417


namespace max_combinatorial_shapes_l263_263748

noncomputable def max_lines_planes_tetrahedrons 
  (α β : Type) [plane α] [plane β] 
  (points_α : finset α) (points_β : finset β) : ℕ × ℕ × ℕ :=
  let points := points_α ∪ points_β in
  let max_lines := (points.card.choose 2) in
  let max_planes := (points_α.card.choose 2 * points_β.card) + 
                    (points_α.card * points_β.card.choose 2) + 2 in
  let max_tetrahedrons := (points_α.card.choose 3 * points_β.card) + 
                          (points_α.card.choose 2 * points_β.card.choose 2) +
                          (points_α.card * points_β.card.choose 3) in
  (max_lines, max_planes, max_tetrahedrons)

theorem max_combinatorial_shapes {α β : Type} [plane α] [plane β]
  (points_α : finset α) (points_β : finset β)
  (hα : points_α.card = 4) (hβ : points_β.card = 5)
  (h_disjoint : points_α ∩ points_β = ∅)
  (h_not_coplanar : ∀ p ∈ points_α, ∀ q ∈ points_β, p ≠ q) :
  max_lines_planes_tetrahedrons α β points_α points_β = (36, 72, 120) :=
by sorry

end max_combinatorial_shapes_l263_263748


namespace probability_sum_is_odd_l263_263259

noncomputable def probability_sum_of_dice_rolls_odd : ℚ :=
let prob_coin := (1 : ℚ) / 2 in
let prob_tail := prob_coin in
let prob_head := prob_coin in
-- Probability of 0 heads (3 tails) -> No dice rolled -> Sum is 0 (even) -> P(odd) = 0
let P0 := (prob_tail ^ 3) * 0 in
-- Probability of 1 head (2 tails) -> 1 die rolled -> P(odd sum) = 3/8 * 1/2
let P1 := (3 * prob_head * prob_tail ^ 2) * (1 / 2) in
-- Probability of 2 heads (1 tail) -> 2 dice rolled -> P(odd sum) = 3/8 * 1/4
let P2 := (3 * prob_head ^ 2 * prob_tail) * (1 / 4) in
-- Probability of 3 heads -> 3 dice rolled -> P(odd sum) = 1/8 * 1/8 * 2
let P3 := (prob_head ^ 3) * (2 / 8) in
P0 + P1 + P2 + P3

theorem probability_sum_is_odd :
  probability_sum_of_dice_rolls_odd = 9 / 32 := by
  sorry

end probability_sum_is_odd_l263_263259


namespace range_of_k_for_ellipse_l263_263829

theorem range_of_k_for_ellipse (k : ℝ) :
  (4 - k > 0) ∧ (k - 1 > 0) ∧ (4 - k ≠ k - 1) ↔ (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  sorry

end range_of_k_for_ellipse_l263_263829


namespace meter_to_skips_l263_263712

/-!
# Math Proof Problem
Suppose hops, skips and jumps are specific units of length. Given the following conditions:
1. \( b \) hops equals \( c \) skips.
2. \( d \) jumps equals \( e \) hops.
3. \( f \) jumps equals \( g \) meters.

Prove that one meter equals \( \frac{cef}{bdg} \) skips.
-/

theorem meter_to_skips (b c d e f g : ℝ) (h1 : b ≠ 0) (h2 : c ≠ 0) (h3 : d ≠ 0) (h4 : e ≠ 0) (h5 : f ≠ 0) (h6 : g ≠ 0) :
  (1 : ℝ) = (cef) / (bdg) :=
by
  -- skipping the proof
  sorry

end meter_to_skips_l263_263712


namespace largest_digit_change_l263_263590

-- Definitions
def initial_number : ℝ := 0.12345

def change_digit (k : Fin 5) : ℝ :=
  match k with
  | 0 => 0.92345
  | 1 => 0.19345
  | 2 => 0.12945
  | 3 => 0.12395
  | 4 => 0.12349

theorem largest_digit_change :
  ∀ k : Fin 5, k ≠ 0 → change_digit 0 > change_digit k :=
by
  intros k hk
  sorry

end largest_digit_change_l263_263590


namespace number_of_cats_l263_263081

def number_of_dogs : ℕ := 43
def number_of_fish : ℕ := 72
def total_pets : ℕ := 149

theorem number_of_cats : total_pets - (number_of_dogs + number_of_fish) = 34 := 
by
  sorry

end number_of_cats_l263_263081


namespace integer_roots_abs_sum_l263_263802

theorem integer_roots_abs_sum (p q r n : ℤ) :
  (∃ n : ℤ, (∀ x : ℤ, x^3 - 2023 * x + n = 0) ∧ p + q + r = 0 ∧ p * q + q * r + r * p = -2023) →
  |p| + |q| + |r| = 102 :=
by
  sorry

end integer_roots_abs_sum_l263_263802


namespace simplify_exponent_l263_263545

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l263_263545


namespace spotted_and_fluffy_cats_l263_263769

theorem spotted_and_fluffy_cats (total_cats : ℕ) (h1 : total_cats = 120)
    (fraction_spotted : ℚ) (h2 : fraction_spotted = 1/3)
    (fraction_fluffy_of_spotted : ℚ) (h3 : fraction_fluffy_of_spotted = 1/4) :
    (total_cats * fraction_spotted * fraction_fluffy_of_spotted).toNat = 10 := by
  sorry

end spotted_and_fluffy_cats_l263_263769


namespace area_of_triangle_l263_263458

theorem area_of_triangle :
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  triangle_area = 34 :=
by {
  -- Definitions
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  -- Proof (normally written here, but omitted with 'sorry')
  sorry
}

end area_of_triangle_l263_263458


namespace proof_problem_l263_263409

-- Define sets
def N_plus : Set ℕ := {x | x > 0}  -- Positive integers
def Z : Set ℤ := {x | true}        -- Integers
def Q : Set ℚ := {x | true}        -- Rational numbers

-- Lean problem statement
theorem proof_problem : 
  (0 ∉ N_plus) ∧ 
  (((-1)^3 : ℤ) ∈ Z) ∧ 
  (π ∉ Q) :=
by
  sorry

end proof_problem_l263_263409


namespace part1_part2_l263_263488

noncomputable def A (x : ℝ) (k : ℝ) := -2 * x ^ 2 - (k - 1) * x + 1
noncomputable def B (x : ℝ) := -2 * (x ^ 2 - x + 2)

-- Part 1: If A is a quadratic binomial, then the value of k is 1
theorem part1 (x : ℝ) (k : ℝ) (h : ∀ x, A x k ≠ 0) : k = 1 :=
sorry

-- Part 2: When k = -1, C + 2A = B, then C = 2x^2 - 2x - 6
theorem part2 (x : ℝ) (C : ℝ → ℝ) (h1 : k = -1) (h2 : ∀ x, C x + 2 * A x k = B x) : (C x = 2 * x ^ 2 - 2 * x - 6) :=
sorry

end part1_part2_l263_263488


namespace fair_decision_l263_263024

def fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

theorem fair_decision (b c : ℕ) : fair_selection b c := by
  sorry

end fair_decision_l263_263024


namespace dance_team_recruitment_l263_263157

theorem dance_team_recruitment 
  (total_students choir_students track_field_students dance_students : ℕ)
  (h1 : total_students = 100)
  (h2 : choir_students = 2 * track_field_students)
  (h3 : dance_students = choir_students + 10)
  (h4 : total_students = track_field_students + choir_students + dance_students) : 
  dance_students = 46 :=
by {
  -- The proof goes here, but it is not required as per instructions
  sorry
}

end dance_team_recruitment_l263_263157


namespace paint_cans_needed_l263_263611

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l263_263611


namespace volleyball_team_girls_l263_263155

theorem volleyball_team_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : G = 15 :=
sorry

end volleyball_team_girls_l263_263155


namespace twice_x_minus_three_lt_zero_l263_263250

theorem twice_x_minus_three_lt_zero (x : ℝ) : (2 * x - 3 < 0) ↔ (2 * x < 3) :=
by
  sorry

end twice_x_minus_three_lt_zero_l263_263250


namespace ways_to_climb_four_steps_l263_263742

theorem ways_to_climb_four_steps (ways_to_climb : ℕ → ℕ) 
  (h1 : ways_to_climb 1 = 1) 
  (h2 : ways_to_climb 2 = 2) 
  (h3 : ways_to_climb 3 = 3) 
  (h_step : ∀ n, ways_to_climb n = ways_to_climb (n - 1) + ways_to_climb (n - 2)) : 
  ways_to_climb 4 = 5 := 
sorry

end ways_to_climb_four_steps_l263_263742


namespace only_k_equal_1_works_l263_263939

-- Define the first k prime numbers product
def prime_prod (k : ℕ) : ℕ :=
  Nat.recOn k 1 (fun n prod => prod * (Nat.factorial (n + 1) - Nat.factorial n))

-- Define a predicate for being the sum of two positive cubes
def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a^3 + b^3

-- The theorem statement
theorem only_k_equal_1_works :
  ∀ k : ℕ, (prime_prod k = 2 ↔ k = 1) :=
by
  sorry

end only_k_equal_1_works_l263_263939


namespace mother_age_l263_263857

theorem mother_age (x : ℕ) (h1 : 3 * x + x = 40) : 3 * x = 30 :=
by
  -- Here we should provide the proof but for now we use sorry to skip it
  sorry

end mother_age_l263_263857


namespace jake_total_work_hours_l263_263677

def initial_debt_A := 150
def payment_A := 60
def hourly_rate_A := 15
def remaining_debt_A := initial_debt_A - payment_A
def hours_to_work_A := remaining_debt_A / hourly_rate_A

def initial_debt_B := 200
def payment_B := 80
def hourly_rate_B := 20
def remaining_debt_B := initial_debt_B - payment_B
def hours_to_work_B := remaining_debt_B / hourly_rate_B

def initial_debt_C := 250
def payment_C := 100
def hourly_rate_C := 25
def remaining_debt_C := initial_debt_C - payment_C
def hours_to_work_C := remaining_debt_C / hourly_rate_C

def total_hours_to_work := hours_to_work_A + hours_to_work_B + hours_to_work_C

theorem jake_total_work_hours :
  total_hours_to_work = 18 :=
sorry

end jake_total_work_hours_l263_263677


namespace probability_diff_colors_l263_263389

/-!
There are 5 identical balls, including 3 white balls and 2 black balls. 
If 2 balls are drawn at once, the probability of the event "the 2 balls have different colors" 
occurring is \( \frac{3}{5} \).
-/

theorem probability_diff_colors 
    (white_balls : ℕ) (black_balls : ℕ) (total_balls : ℕ) (drawn_balls : ℕ) 
    (h_white : white_balls = 3) (h_black : black_balls = 2) (h_total : total_balls = 5) (h_drawn : drawn_balls = 2) :
    let total_ways := Nat.choose total_balls drawn_balls
    let diff_color_ways := (Nat.choose white_balls 1) * (Nat.choose black_balls 1)
    (diff_color_ways : ℚ) / (total_ways : ℚ) = 3 / 5 := 
by
    -- Step 1: Calculate total ways to draw 2 balls out of 5
    -- total_ways = 10 (by binomial coefficient)
    -- Step 2: Calculate favorable outcomes (1 white, 1 black)
    -- diff_color_ways = 6
    -- Step 3: Calculate probability
    -- Probability = 6 / 10 = 3 / 5
    sorry

end probability_diff_colors_l263_263389


namespace simplify_and_evaluate_l263_263707

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l263_263707


namespace PRINT_3_3_2_l263_263726

def PRINT (a b : Nat) : Nat × Nat := (a, b)

theorem PRINT_3_3_2 :
  PRINT 3 (3 + 2) = (3, 5) :=
by
  sorry

end PRINT_3_3_2_l263_263726


namespace product_without_zero_digits_l263_263965

def no_zero_digits (n : ℕ) : Prop :=
  ¬ ∃ d : ℕ, d ∈ n.digits 10 ∧ d = 0

theorem product_without_zero_digits :
  ∃ a b : ℕ, a * b = 1000000000 ∧ no_zero_digits a ∧ no_zero_digits b :=
by
  sorry

end product_without_zero_digits_l263_263965


namespace smallest_integer_representable_l263_263270

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l263_263270


namespace arithmetic_geometric_sequence_l263_263193

theorem arithmetic_geometric_sequence : 
  ∀ (a : ℤ), (∀ n : ℤ, a_n = a + (n-1) * 2) → 
  (a + 4)^2 = a * (a + 6) → 
  (a + 10 = 2) :=
by
  sorry

end arithmetic_geometric_sequence_l263_263193


namespace ratio_IM_IN_l263_263364

noncomputable def compute_ratio (IA IB IC ID : ℕ) (M N : ℕ) : ℚ :=
  (IA * IC : ℚ) / (IB * ID : ℚ)

theorem ratio_IM_IN (IA IB IC ID : ℕ) (hIA : IA = 12) (hIB : IB = 16) (hIC : IC = 14) (hID : ID = 11) :
  compute_ratio IA IB IC ID = 21 / 22 := by
  rw [hIA, hIB, hIC, hID]
  sorry

end ratio_IM_IN_l263_263364


namespace tangent_line_at_point_l263_263558

theorem tangent_line_at_point :
  let f := λ x : ℝ => x^3 - 3*x^2 + 3
  let f_deriv := deriv f
  let slope_at_1 := f_deriv 1
  let tangent_line := λ x : ℝ => slope_at_1 * (x - 1) + 1
  tangent_line = λ x : ℝ => -3 * x + 4 :=
by
  let f := λ x : ℝ => x^3 - 3*x^2 + 3
  let f_deriv := deriv f
  let slope_at_1 := f_deriv 1
  let tangent_line := λ x : ℝ => slope_at_1 * (x - 1) + 1
  show tangent_line = λ x : ℝ => -3 * x + 4
  sorry

end tangent_line_at_point_l263_263558


namespace pile_splitting_l263_263507

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l263_263507


namespace find_k_l263_263029

noncomputable def curve_C (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1

noncomputable def line_eq (k x y : ℝ) : Prop :=
  y = k * x + 1

theorem find_k (k : ℝ) :
  (∃ A B : ℝ × ℝ, (curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ 
   (A.1 * B.1 + A.2 * B.2 = 0))) ↔ (k = 1/2 ∨ k = -1/2) :=
sorry

end find_k_l263_263029


namespace ambulance_ride_cost_correct_l263_263973

noncomputable def total_bill : ℝ := 18000
noncomputable def medication_percentage : ℝ := 0.35
noncomputable def imaging_percentage : ℝ := 0.15
noncomputable def surgery_percentage : ℝ := 0.25
noncomputable def overnight_stays_percentage : ℝ := 0.10
noncomputable def doctors_fees_percentage : ℝ := 0.05

noncomputable def food_fee : ℝ := 300
noncomputable def consultation_fee : ℝ := 450
noncomputable def physical_therapy_fee : ℝ := 600

noncomputable def medication_cost : ℝ := medication_percentage * total_bill
noncomputable def imaging_cost : ℝ := imaging_percentage * total_bill
noncomputable def surgery_cost : ℝ := surgery_percentage * total_bill
noncomputable def overnight_stays_cost : ℝ := overnight_stays_percentage * total_bill
noncomputable def doctors_fees_cost : ℝ := doctors_fees_percentage * total_bill

noncomputable def percentage_based_costs : ℝ :=
  medication_cost + imaging_cost + surgery_cost + overnight_stays_cost + doctors_fees_cost

noncomputable def fixed_costs : ℝ :=
  food_fee + consultation_fee + physical_therapy_fee

noncomputable def total_known_costs : ℝ :=
  percentage_based_costs + fixed_costs

noncomputable def ambulance_ride_cost : ℝ :=
  total_bill - total_known_costs

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 450 := by
  sorry

end ambulance_ride_cost_correct_l263_263973


namespace sum_of_squares_of_roots_l263_263790

theorem sum_of_squares_of_roots :
  let a := 5
  let b := -7
  let c := 2
  let x1 := (-b + (b^2 - 4*a*c)^(1/2)) / (2*a)
  let x2 := (-b - (b^2 - 4*a*c)^(1/2)) / (2*a)
  x1^2 + x2^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

end sum_of_squares_of_roots_l263_263790


namespace half_angle_quadrant_l263_263827

variables {α : ℝ} {k : ℤ} {n : ℤ}

theorem half_angle_quadrant (h : ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270) :
  ∃ (n : ℤ), (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
      (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315) :=
by sorry

end half_angle_quadrant_l263_263827


namespace closest_perfect_square_l263_263280

theorem closest_perfect_square (n : ℕ) (h1 : n = 325) : 
    ∃ m : ℕ, m^2 = 324 ∧ 
    (∀ k : ℕ, (k^2 ≤ n ∨ k^2 ≥ n) → (k = 18 ∨ k^2 > 361 ∨ k^2 < 289)) := 
by
  sorry

end closest_perfect_square_l263_263280


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l263_263510

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l263_263510


namespace equation_of_trajectory_l263_263815

open Real

variable (P : ℝ → ℝ → Prop)
variable (C : ℝ → ℝ → Prop)
variable (L : ℝ → ℝ → Prop)

-- Definition of the fixed circle C
def fixed_circle (x y : ℝ) : Prop :=
  (x + 2) ^ 2 + y ^ 2 = 1

-- Definition of the fixed line L
def fixed_line (x y : ℝ) : Prop := 
  x = 1

noncomputable def moving_circle (P : ℝ → ℝ → Prop) (r : ℝ) : Prop :=
  ∃ x y : ℝ, P x y ∧ r > 0 ∧
  (∀ a b : ℝ, fixed_circle a b → ((x - a) ^ 2 + (y - b) ^ 2) = (r + 1) ^ 2) ∧
  (∀ a b : ℝ, fixed_line a b → (abs (x - a)) = (r + 1))

theorem equation_of_trajectory
  (P : ℝ → ℝ → Prop)
  (r : ℝ)
  (h : moving_circle P r) :
  ∀ x y : ℝ, P x y → y ^ 2 = -8 * x :=
by
  sorry

end equation_of_trajectory_l263_263815


namespace inequality_proof_l263_263700

theorem inequality_proof (a b : ℝ) (h : a + b ≠ 0) :
  (a + b) / (a^2 - a * b + b^2) ≤ 4 / |a + b| ∧
  ((a + b) / (a^2 - a * b + b^2) = 4 / |a + b| ↔ a = b) :=
by
  sorry

end inequality_proof_l263_263700


namespace bhanu_income_percentage_l263_263443

variable {I P : ℝ}

theorem bhanu_income_percentage (h₁ : 300 = (P / 100) * I)
                                  (h₂ : 210 = 0.3 * (I - 300)) :
  P = 30 :=
by
  sorry

end bhanu_income_percentage_l263_263443


namespace men_in_group_l263_263554

theorem men_in_group (A : ℝ) (n : ℕ) (h : n > 0) 
  (inc_avg : ↑n * A + 2 * 32 - (21 + 23) = ↑n * (A + 1)) : n = 20 :=
sorry

end men_in_group_l263_263554


namespace option_not_equal_to_three_halves_l263_263401

theorem option_not_equal_to_three_halves (d : ℚ) (h1 : d = 3/2) 
    (hA : 9/6 = 3/2) 
    (hB : 1 + 1/2 = 3/2) 
    (hC : 1 + 2/4 = 3/2)
    (hE : 1 + 6/12 = 3/2) :
  1 + 2/3 ≠ 3/2 :=
by
  sorry

end option_not_equal_to_three_halves_l263_263401


namespace find_a_in_triangle_l263_263215

theorem find_a_in_triangle (b c : ℝ) (cos_B_minus_C : ℝ) (a : ℝ) 
  (hb : b = 7) (hc : c = 6) (hcos : cos_B_minus_C = 15 / 16) :
  a = 5 * Real.sqrt 3 :=
by
  sorry

end find_a_in_triangle_l263_263215


namespace general_term_less_than_zero_from_13_l263_263196

-- Define the arithmetic sequence and conditions
def an (n : ℕ) : ℝ := 12 - n

-- Condition: a_3 = 9
def a3_condition : Prop := an 3 = 9

-- Condition: a_9 = 3
def a9_condition : Prop := an 9 = 3

-- Prove the general term of the sequence is 12 - n
theorem general_term (n : ℕ) (h3 : a3_condition) (h9 : a9_condition) :
  an n = 12 - n := 
sorry

-- Prove that the sequence becomes less than 0 starting from the 13th term
theorem less_than_zero_from_13 (h3 : a3_condition) (h9 : a9_condition) :
  ∀ n, n ≥ 13 → an n < 0 :=
sorry

end general_term_less_than_zero_from_13_l263_263196


namespace find_number_of_students_l263_263253

-- Conditions
def john_marks_wrongly_recorded : ℕ := 82
def john_actual_marks : ℕ := 62
def sarah_marks_wrongly_recorded : ℕ := 76
def sarah_actual_marks : ℕ := 66
def emily_marks_wrongly_recorded : ℕ := 92
def emily_actual_marks : ℕ := 78
def increase_in_average : ℚ := 1 / 2

-- Proof problem
theorem find_number_of_students (n : ℕ) 
    (h1 : john_marks_wrongly_recorded = 82)
    (h2 : john_actual_marks = 62)
    (h3 : sarah_marks_wrongly_recorded = 76)
    (h4 : sarah_actual_marks = 66)
    (h5 : emily_marks_wrongly_recorded = 92)
    (h6 : emily_actual_marks = 78) 
    (h7: increase_in_average = 1 / 2):
    n = 88 :=
by 
  sorry

end find_number_of_students_l263_263253


namespace rectangle_side_length_relation_l263_263959

variable (x y : ℝ)

-- Condition: The area of the rectangle is 10
def is_rectangle_area_10 (x y : ℝ) : Prop := x * y = 10

-- Theorem: Given the area condition, express y in terms of x
theorem rectangle_side_length_relation (h : is_rectangle_area_10 x y) : y = 10 / x :=
sorry

end rectangle_side_length_relation_l263_263959


namespace simplify_expression_l263_263547

theorem simplify_expression (x : ℝ) : 7 * x + 15 - 3 * x + 2 = 4 * x + 17 := 
by sorry

end simplify_expression_l263_263547


namespace miley_total_cost_l263_263694

-- Define the cost per cellphone
def cost_per_cellphone : ℝ := 800

-- Define the number of cellphones
def number_of_cellphones : ℝ := 2

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the total cost without discount
def total_cost_without_discount : ℝ := cost_per_cellphone * number_of_cellphones

-- Define the discount amount
def discount_amount : ℝ := total_cost_without_discount * discount_rate

-- Define the total cost with discount
def total_cost_with_discount : ℝ := total_cost_without_discount - discount_amount

-- Prove that the total amount Miley paid is $1520
theorem miley_total_cost : total_cost_with_discount = 1520 := by
  sorry

end miley_total_cost_l263_263694


namespace value_of_4x_l263_263957

variable (x : ℤ)

theorem value_of_4x (h : 2 * x - 3 = 10) : 4 * x = 26 := 
by
  sorry

end value_of_4x_l263_263957


namespace pile_splitting_l263_263508

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l263_263508


namespace geom_series_first_term_l263_263572

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l263_263572


namespace trig_problem_l263_263806

theorem trig_problem 
  (α : ℝ) 
  (h1 : Real.cos α = -1/2) 
  (h2 : 180 * (Real.pi / 180) < α ∧ α < 270 * (Real.pi / 180)) : 
  α = 240 * (Real.pi / 180) :=
sorry

end trig_problem_l263_263806


namespace no_natural_number_solution_l263_263451

theorem no_natural_number_solution :
  ¬∃ (n : ℕ), ∃ (k : ℕ), (n^5 - 5*n^3 + 4*n + 7 = k^2) :=
sorry

end no_natural_number_solution_l263_263451


namespace non_zero_real_solution_of_equation_l263_263738

noncomputable def equation_solution : Prop :=
  ∀ (x : ℝ), x ≠ 0 ∧ (7 * x) ^ 14 = (14 * x) ^ 7 → x = 2 / 7

theorem non_zero_real_solution_of_equation : equation_solution := sorry

end non_zero_real_solution_of_equation_l263_263738


namespace mod_81256_eq_16_l263_263732

theorem mod_81256_eq_16 : ∃ n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 % 31 = n := by
  use 16
  sorry

end mod_81256_eq_16_l263_263732


namespace max_value_x_1_minus_3x_is_1_over_12_l263_263466

open Real

noncomputable def max_value_of_x_1_minus_3x (x : ℝ) : ℝ :=
  x * (1 - 3 * x)

theorem max_value_x_1_minus_3x_is_1_over_12 :
  ∀ x : ℝ, 0 < x ∧ x < 1 / 3 → max_value_of_x_1_minus_3x x ≤ 1 / 12 :=
by
  intros x h
  sorry

end max_value_x_1_minus_3x_is_1_over_12_l263_263466


namespace cost_of_drapes_l263_263041

theorem cost_of_drapes (D: ℝ) (h1 : 3 * 40 = 120) (h2 : D * 3 + 120 = 300) : D = 60 :=
  sorry

end cost_of_drapes_l263_263041


namespace painting_two_sides_time_l263_263979

-- Definitions for the conditions
def time_to_paint_one_side_per_board : Nat := 1
def drying_time_per_board : Nat := 5

-- Definitions for the problem
def total_boards : Nat := 6

-- Main theorem statement
theorem painting_two_sides_time :
  (total_boards * time_to_paint_one_side_per_board) + drying_time_per_board + (total_boards * time_to_paint_one_side_per_board) = 12 :=
sorry

end painting_two_sides_time_l263_263979


namespace base9_to_base10_l263_263848

theorem base9_to_base10 : Nat.ofDigits 9 [3, 5, 6, 2] = 2648 := by
  sorry

end base9_to_base10_l263_263848


namespace friends_same_group_probability_l263_263070

noncomputable def probability_same_group (n : ℕ) (groups : ℕ) : ℚ :=
  1 / groups * 1 / groups

theorem friends_same_group_probability :
  ∀ (students groups : ℕ), 
  students = 900 → groups = 4 →
  (probability_same_group students groups = 1 / 16) :=
by
  intros students groups h_students h_groups
  rw [probability_same_group]
  have h1 : students = 900 := h_students
  have h2 : groups = 4 := h_groups
  simp [h1, h2]
  norm_num
  sorry

end friends_same_group_probability_l263_263070


namespace intersection_of_A_and_B_l263_263473

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l263_263473


namespace sandy_spent_on_repairs_l263_263234

theorem sandy_spent_on_repairs (initial_cost : ℝ) (selling_price : ℝ) (gain_percent : ℝ) (repair_cost : ℝ) :
  initial_cost = 800 → selling_price = 1400 → gain_percent = 40 → selling_price = 1.4 * (initial_cost + repair_cost) → repair_cost = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_spent_on_repairs_l263_263234


namespace max_sum_factors_of_60_exists_max_sum_factors_of_60_l263_263492

theorem max_sum_factors_of_60 (d Δ : ℕ) (h : d * Δ = 60) : (d + Δ) ≤ 61 :=
sorry

theorem exists_max_sum_factors_of_60 : ∃ d Δ : ℕ, d * Δ = 60 ∧ d + Δ = 61 :=
sorry

end max_sum_factors_of_60_exists_max_sum_factors_of_60_l263_263492


namespace angle_triple_supplement_l263_263114

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263114


namespace find_X_l263_263836

theorem find_X (X : ℕ) (h1 : 2 + 1 + 3 + X = 3 + 4 + 5) : X = 6 :=
by
  sorry

end find_X_l263_263836


namespace solve_for_a_l263_263388

theorem solve_for_a (x a : ℝ) (h1 : x + 2 * a - 6 = 0) (h2 : x = -2) : a = 4 :=
by
  sorry

end solve_for_a_l263_263388


namespace team_B_score_third_game_l263_263431

theorem team_B_score_third_game (avg_points : ℝ) (additional_needed : ℝ) (total_target : ℝ) (P : ℝ) :
  avg_points = 61.5 → additional_needed = 330 → total_target = 500 →
  2 * avg_points + P + additional_needed = total_target → P = 47 :=
by
  intros avg_points_eq additional_needed_eq total_target_eq total_eq
  rw [avg_points_eq, additional_needed_eq, total_target_eq] at total_eq
  sorry

end team_B_score_third_game_l263_263431


namespace route_C_is_quicker_l263_263695

/-
  Define the conditions based on the problem:
  - Route C: 8 miles at 40 mph.
  - Route D: 5 miles at 35 mph and 2 miles at 25 mph with an additional 3 minutes stop.
-/

def time_route_C : ℚ := (8 : ℚ) / (40 : ℚ) * 60  -- in minutes

def time_route_D : ℚ := ((5 : ℚ) / (35 : ℚ) * 60) + ((2 : ℚ) / (25 : ℚ) * 60) + 3  -- in minutes

def time_difference : ℚ := time_route_D - time_route_C  -- difference in minutes

theorem route_C_is_quicker : time_difference = 4.37 := 
by 
  sorry

end route_C_is_quicker_l263_263695


namespace add_to_fraction_eq_l263_263588

theorem add_to_fraction_eq (n : ℤ) : (3 + n : ℚ) / (5 + n) = 5 / 6 → n = 7 := 
by
  sorry

end add_to_fraction_eq_l263_263588


namespace total_coins_correct_l263_263521

-- Define basic parameters
def stacks_pennies : Nat := 3
def coins_per_penny_stack : Nat := 10
def stacks_nickels : Nat := 5
def coins_per_nickel_stack : Nat := 8
def stacks_dimes : Nat := 7
def coins_per_dime_stack : Nat := 4

-- Calculate total coins for each type
def total_pennies : Nat := stacks_pennies * coins_per_penny_stack
def total_nickels : Nat := stacks_nickels * coins_per_nickel_stack
def total_dimes : Nat := stacks_dimes * coins_per_dime_stack

-- Calculate total number of coins
def total_coins : Nat := total_pennies + total_nickels + total_dimes

-- Proof statement
theorem total_coins_correct : total_coins = 98 := by
  -- Proof steps go here (omitted)
  sorry

end total_coins_correct_l263_263521


namespace union_A_B_range_of_a_l263_263481

-- Definitions of sets A, B, and C
def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 9 }
def B : Set ℝ := { x | 2 < x ∧ x < 5 }
def C (a : ℝ) : Set ℝ := { x | x > a }

-- Problem 1: Proving A ∪ B = { x | 2 < x ≤ 9 }
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x ≤ 9 } :=
sorry

-- Problem 2: Proving the range of 'a' given B ∩ C = ∅
theorem range_of_a (a : ℝ) (h : B ∩ C a = ∅) : a ≥ 5 :=
sorry

end union_A_B_range_of_a_l263_263481


namespace split_into_similar_piles_l263_263515

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l263_263515


namespace simplify_expression_l263_263239

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 20 + 4 * y = 45 * x + 20 + 4 * y :=
by
  sorry

end simplify_expression_l263_263239


namespace coin_stack_l263_263674

def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75
def stack_height : ℝ := 14

theorem coin_stack (n_penny n_nickel n_dime n_quarter : ℕ) 
  (h : n_penny * penny_thickness + n_nickel * nickel_thickness + n_dime * dime_thickness + n_quarter * quarter_thickness = stack_height) :
  n_penny + n_nickel + n_dime + n_quarter = 8 :=
sorry

end coin_stack_l263_263674


namespace fraction_auto_installment_credit_extended_by_finance_companies_l263_263441

def total_consumer_installment_credit : ℝ := 291.6666666666667
def auto_instalment_percentage : ℝ := 0.36
def auto_finance_companies_credit_extended : ℝ := 35

theorem fraction_auto_installment_credit_extended_by_finance_companies :
  auto_finance_companies_credit_extended / (auto_instalment_percentage * total_consumer_installment_credit) = 1 / 3 :=
by
  sorry

end fraction_auto_installment_credit_extended_by_finance_companies_l263_263441


namespace find_first_term_l263_263570

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l263_263570


namespace total_cookies_baked_l263_263012

def cookies_baked_yesterday : ℕ := 435
def cookies_baked_today : ℕ := 139

theorem total_cookies_baked : cookies_baked_yesterday + cookies_baked_today = 574 := by
  sorry

end total_cookies_baked_l263_263012


namespace divides_of_exponentiation_l263_263237

theorem divides_of_exponentiation (n : ℕ) : 7 ∣ 3^(12 * n + 1) + 2^(6 * n + 2) := 
  sorry

end divides_of_exponentiation_l263_263237


namespace Kiran_money_l263_263997

theorem Kiran_money (R G K : ℕ) (h1 : R / G = 6 / 7) (h2 : G / K = 6 / 15) (h3 : R = 36) : K = 105 := by
  sorry

end Kiran_money_l263_263997


namespace ratio_of_good_states_l263_263584

theorem ratio_of_good_states (n : ℕ) :
  let total_states := 2^(2*n)
  let good_states := Nat.choose (2 * n) n
  good_states / total_states = (List.range n).foldr (fun i acc => acc * (2*i+1)) 1 / (2^n * Nat.factorial n) := sorry

end ratio_of_good_states_l263_263584


namespace tangent_line_equation_l263_263937

theorem tangent_line_equation (x y : ℝ) (h : y = x^3 + 1) (t : x = -1) :
  3*x - y + 3 = 0 :=
sorry

end tangent_line_equation_l263_263937


namespace carriages_people_equation_l263_263030

theorem carriages_people_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end carriages_people_equation_l263_263030


namespace A_lent_5000_to_B_l263_263418

noncomputable def principalAmountB
    (P_C : ℝ)
    (r : ℝ)
    (total_interest : ℝ)
    (P_B : ℝ) : Prop :=
  let I_B := P_B * r * 2
  let I_C := P_C * r * 4
  I_B + I_C = total_interest

theorem A_lent_5000_to_B :
  principalAmountB 3000 0.10 2200 5000 :=
by
  sorry

end A_lent_5000_to_B_l263_263418


namespace bank_exceeds_1600cents_in_9_days_after_Sunday_l263_263039

theorem bank_exceeds_1600cents_in_9_days_after_Sunday
  (a : ℕ)
  (r : ℕ)
  (initial_deposit : ℕ)
  (days_after_sunday : ℕ)
  (geometric_series : ℕ -> ℕ)
  (sum_geometric_series : ℕ -> ℕ)
  (geo_series_definition : ∀(n : ℕ), geometric_series n = 5 * 2^n)
  (sum_geo_series_definition : ∀(n : ℕ), sum_geometric_series n = 5 * (2^n - 1))
  (exceeds_condition : ∀(n : ℕ), sum_geometric_series n > 1600 -> n >= 9) :
  days_after_sunday = 9 → a = 5 → r = 2 → initial_deposit = 5 → days_after_sunday = 9 → geometric_series 1 = 10 → sum_geometric_series 9 > 1600 :=
by sorry

end bank_exceeds_1600cents_in_9_days_after_Sunday_l263_263039


namespace remaining_water_l263_263449

def initial_water : ℚ := 3
def water_used : ℚ := 4 / 3

theorem remaining_water : initial_water - water_used = 5 / 3 := 
by sorry -- skipping the proof for now

end remaining_water_l263_263449


namespace sum_of_a_b_c_l263_263948

theorem sum_of_a_b_c (a b c : ℝ) (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 :=
by
  -- The proof would go here
  sorry

end sum_of_a_b_c_l263_263948


namespace series_converges_to_one_l263_263450

noncomputable def infinite_series := ∑' n, (3^n) / (3^(2^n) + 2)

theorem series_converges_to_one :
  infinite_series = 1 := by
  sorry

end series_converges_to_one_l263_263450


namespace ratio_AB_AC_equals_FB_FC_l263_263687

theorem ratio_AB_AC_equals_FB_FC
  (A B C D E F: Point)
  (Γ: Circle)
  (h_scalene: scalene △ A B C)
  (h_circumcircle: \Gamma.circumscribes △ A B C)
  (h_bisector_A: internal_bisector A intersects [D, E])
  (h_diameter_DE: circle_with_diameter D E)
  (h_F_on_Γ: F ∈ \Gamma)
  (h_second_intersection: second_intersection F circle_with_diameter D E on_line [Γ]):
  ratio AB AC = ratio FB FC := 
begin
  sorry -- Proof goes here.
end

end ratio_AB_AC_equals_FB_FC_l263_263687


namespace simplify_exponent_l263_263543

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l263_263543


namespace B_oxen_count_l263_263139

/- 
  A puts 10 oxen for 7 months.
  B puts some oxen for 5 months.
  C puts 15 oxen for 3 months.
  The rent of the pasture is Rs. 175.
  C should pay Rs. 45 as his share of rent.
  We need to prove that B put 12 oxen for grazing.
-/

def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

def A_ox_months := oxen_months 10 7
def C_ox_months := oxen_months 15 3

def total_rent : ℕ := 175
def C_rent_share : ℕ := 45

theorem B_oxen_count (x : ℕ) : 
  (C_rent_share : ℝ) / total_rent = (C_ox_months : ℝ) / (A_ox_months + 5 * x + C_ox_months) →
  x = 12 := 
by
  sorry

end B_oxen_count_l263_263139


namespace find_first_term_l263_263574

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l263_263574


namespace sqrt_neg_squared_eq_two_l263_263923

theorem sqrt_neg_squared_eq_two : (-Real.sqrt 2) ^ 2 = 2 := by
  sorry

end sqrt_neg_squared_eq_two_l263_263923


namespace min_value_of_m_n_l263_263943

variable {a b : ℝ}
variable (ab_eq_4 : a * b = 4)
variable (m : ℝ := b + 1 / a)
variable (n : ℝ := a + 1 / b)

theorem min_value_of_m_n (h1 : 0 < a) (h2 : 0 < b) : m + n = 5 :=
sorry

end min_value_of_m_n_l263_263943


namespace painting_together_time_l263_263216

theorem painting_together_time (jamshid_time taimour_time time_together : ℝ) 
  (h1 : jamshid_time = taimour_time / 2)
  (h2 : taimour_time = 21)
  (h3 : time_together = 7) :
  (1 / taimour_time + 1 / jamshid_time) * time_together = 1 := 
sorry

end painting_together_time_l263_263216


namespace right_triangle_perimeter_l263_263383

-- Given conditions
variable (x y : ℕ)
def leg1 := 11
def right_triangle := (101 * 11 = 121)

-- The question and answer
theorem right_triangle_perimeter :
  (y + x = 121) ∧ (y - x = 1) → (11 + x + y = 132) :=
by
  sorry

end right_triangle_perimeter_l263_263383


namespace find_first_offset_l263_263799

theorem find_first_offset (d b : ℝ) (Area : ℝ) :
  d = 22 → b = 6 → Area = 165 → (first_offset : ℝ) → 22 * (first_offset + 6) / 2 = 165 → first_offset = 9 :=
by
  intros hd hb hArea first_offset heq
  sorry

end find_first_offset_l263_263799


namespace optimal_pricing_l263_263845

-- Define the conditions given in the problem
def cost_price : ℕ := 40
def selling_price : ℕ := 60
def weekly_sales : ℕ := 300

def sales_volume (price : ℕ) : ℕ := weekly_sales - 10 * (price - selling_price)
def profit (price : ℕ) : ℕ := (price - cost_price) * sales_volume price

-- Statement to prove
theorem optimal_pricing : ∃ (price : ℕ), price = 65 ∧ profit price = 6250 :=
by {
  sorry
}

end optimal_pricing_l263_263845


namespace part1_inequality_part2_range_of_a_l263_263823

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 1)

-- Part (1)
theorem part1_inequality (x : ℝ) (h : f x 2 < 5) : -2 < x ∧ x < 3 := sorry

-- Part (2)
theorem part2_range_of_a (x a : ℝ) (h : ∀ x, f x a ≥ 4 - abs (a - 1)) : a ≤ -2 ∨ a ≥ 2 := sorry

end part1_inequality_part2_range_of_a_l263_263823


namespace CarltonUniqueOutfits_l263_263788

theorem CarltonUniqueOutfits:
  ∀ (buttonUpShirts sweaterVests : ℕ), 
    buttonUpShirts = 3 →
    sweaterVests = 2 * buttonUpShirts →
    (sweaterVests * buttonUpShirts) = 18 :=
by
  intros buttonUpShirts sweaterVests h1 h2
  rw [h1, h2]
  simp
  sorry

end CarltonUniqueOutfits_l263_263788


namespace calculate_difference_of_squares_l263_263922

theorem calculate_difference_of_squares : (153^2 - 147^2) = 1800 := by
  sorry

end calculate_difference_of_squares_l263_263922


namespace sum_of_modulus_of_three_element_subsets_l263_263317

def P := { x : ℕ | ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 10 ∧ x = 2 * n - 1 }

def three_element_subsets : finset (finset ℕ) := (finset.powerset P.to_finset).filter (λ s, s.card = 3)

def sum_modulus (S : finset (finset ℕ)) : ℕ :=
  S.sum (λ s, s.sum id)

theorem sum_of_modulus_of_three_element_subsets :
  sum_modulus three_element_subsets = 3600 := 
  by
  -- meant to show that the statement holds
  -- proof steps would go here
  sorry

end sum_of_modulus_of_three_element_subsets_l263_263317


namespace fg_eval_at_3_l263_263688

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_eval_at_3 : f (g 3) = 99 := by
  sorry

end fg_eval_at_3_l263_263688


namespace expand_product_l263_263322

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  -- No proof required, just state the theorem
  sorry

end expand_product_l263_263322


namespace sum_abs_of_roots_l263_263804

variables {p q r : ℤ}

theorem sum_abs_of_roots:
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 94 := by
  intro h1 h2
  sorry

end sum_abs_of_roots_l263_263804


namespace find_four_digit_number_l263_263416

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l263_263416


namespace prices_and_subsidy_l263_263230

theorem prices_and_subsidy (total_cost : ℕ) (price_leather_jacket : ℕ) (price_sweater : ℕ) (subsidy_percentage : ℕ) 
  (leather_jacket_condition : price_leather_jacket = 5 * price_sweater + 600)
  (cost_condition : price_leather_jacket + price_sweater = total_cost)
  (total_sold : ℕ) (max_subsidy : ℕ) :
  (total_cost = 3000 ∧
   price_leather_jacket = 2600 ∧
   price_sweater = 400 ∧
   subsidy_percentage = 10) ∧ 
  ∃ a : ℕ, (2200 * a ≤ 50000 ∧ total_sold - a ≥ 128) :=
by
  sorry

end prices_and_subsidy_l263_263230


namespace smallest_base10_integer_l263_263278

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l263_263278


namespace solve_equation_l263_263375

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l263_263375


namespace min_n_satisfies_inequality_l263_263171

theorem min_n_satisfies_inequality :
  ∃ n : ℕ, 0 < n ∧ -3 * (n : ℤ) ^ 4 + 5 * (n : ℤ) ^ 2 - 199 < 0 ∧ (∀ m : ℕ, 0 < m ∧ -3 * (m : ℤ) ^ 4 + 5 * (m : ℤ) ^ 2 - 199 < 0 → 2 ≤ m) := 
  sorry

end min_n_satisfies_inequality_l263_263171


namespace survey_households_selected_l263_263025

theorem survey_households_selected 
    (total_households : ℕ) 
    (middle_income_families : ℕ) 
    (low_income_families : ℕ) 
    (high_income_selected : ℕ)
    (total_high_income_families : ℕ)
    (total_selected_households : ℕ) 
    (H1 : total_households = 480)
    (H2 : middle_income_families = 200)
    (H3 : low_income_families = 160)
    (H4 : high_income_selected = 6)
    (H5 : total_high_income_families = total_households - (middle_income_families + low_income_families))
    (H6 : total_selected_households * total_high_income_families = high_income_selected * total_households) :
    total_selected_households = 24 :=
by
  -- The actual proof will go here:
  sorry

end survey_households_selected_l263_263025


namespace fraction_of_males_l263_263919

theorem fraction_of_males (M F : ℝ) (h1 : M + F = 1) (h2 : (7/8 * M + 9/10 * (1 - M)) = 0.885) :
  M = 0.6 :=
sorry

end fraction_of_males_l263_263919


namespace find_Q_l263_263641

variable {x P Q : ℝ}

theorem find_Q (h₁ : x + 1 / x = P) (h₂ : P = 1) : x^6 + 1 / x^6 = 2 :=
by
  sorry

end find_Q_l263_263641


namespace most_lines_of_symmetry_circle_l263_263740

-- Define the figures and their lines of symmetry
def regular_pentagon_lines_of_symmetry : ℕ := 5
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def circle_lines_of_symmetry : ℕ := 0  -- Representing infinite lines of symmetry in Lean is unconventional; we'll use a special case.
def regular_hexagon_lines_of_symmetry : ℕ := 6
def ellipse_lines_of_symmetry : ℕ := 2

-- Define a predicate to check if one figure has more lines of symmetry than all others
def most_lines_of_symmetry {α : Type} [LinearOrder α] (f : α) (others : List α) : Prop :=
  ∀ x ∈ others, f ≥ x

-- Define the problem statement in Lean
theorem most_lines_of_symmetry_circle :
  most_lines_of_symmetry circle_lines_of_symmetry [
    regular_pentagon_lines_of_symmetry,
    isosceles_triangle_lines_of_symmetry,
    regular_hexagon_lines_of_symmetry,
    ellipse_lines_of_symmetry ] :=
by {
  -- To represent infinite lines, we consider 0 as a larger "dummy" number in this context,
  -- since in Lean we don't have a built-in representation for infinity in finite ordering.
  -- Replace with a suitable model if necessary.
  sorry
}

end most_lines_of_symmetry_circle_l263_263740


namespace first_player_wins_the_game_l263_263392

-- Define the game state with 1992 stones and rules for taking stones
structure GameState where
  stones : Nat

-- Game rule: Each player can take a number of stones that is a divisor of the number of stones the 
-- opponent took on the previous turn
def isValidMove (prevMove: Nat) (currentMove: Nat) : Prop :=
  currentMove > 0 ∧ prevMove % currentMove = 0

-- The first player can take any number of stones but not all at once on their first move
def isFirstMoveValid (move: Nat) : Prop :=
  move > 0 ∧ move < 1992

-- Define the initial state of the game with 1992 stones
def initialGameState : GameState := { stones := 1992 }

-- Definition of optimal play leading to the first player's victory
def firstPlayerWins (s : GameState) : Prop :=
  s.stones = 1992 →
  ∃ move: Nat, isFirstMoveValid move ∧
  ∃ nextState: GameState, nextState.stones = s.stones - move ∧ 
  -- The first player wins with optimal strategy
  sorry

-- Theorem statement in Lean 4 equivalent to the math problem
theorem first_player_wins_the_game :
  firstPlayerWins initialGameState :=
  sorry

end first_player_wins_the_game_l263_263392


namespace number_of_boys_l263_263142

def school_problem (x y : ℕ) : Prop :=
  (x + y = 400) ∧ (y = (x / 100) * 400)

theorem number_of_boys (x y : ℕ) (h : school_problem x y) : x = 80 :=
by
  sorry

end number_of_boys_l263_263142


namespace one_liter_fills_five_cups_l263_263861

-- Define the problem conditions and question in Lean 4
def one_liter_milliliters : ℕ := 1000
def cup_volume_milliliters : ℕ := 200

theorem one_liter_fills_five_cups : one_liter_milliliters / cup_volume_milliliters = 5 := 
by 
  sorry -- proof skipped

end one_liter_fills_five_cups_l263_263861


namespace arithmetic_sequence_l263_263495

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) (d a1 : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) (d a1 : ℤ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
theorem arithmetic_sequence (n : ℕ) (d a1 : ℤ) (S3 : ℤ) (h1 : a1 = 10) (h2 : S_n 3 d a1 = 24) :
  (a_n n d a1 = 12 - 2 * n) ∧ (S_n n (-2) 12 = -n^2 + 11 * n) ∧ (∀ k, S_n k (-2) 12 ≤ 30) :=
by
  sorry

end arithmetic_sequence_l263_263495


namespace simplified_expression_l263_263703

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end simplified_expression_l263_263703


namespace cricket_run_rate_l263_263746

theorem cricket_run_rate
  (run_rate_first_10_overs : ℝ)
  (overs_first_10_overs : ℕ)
  (target_runs : ℕ)
  (remaining_overs : ℕ)
  (run_rate_required : ℝ) :
  run_rate_first_10_overs = 3.2 →
  overs_first_10_overs = 10 →
  target_runs = 242 →
  remaining_overs = 40 →
  run_rate_required = 5.25 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) = 210 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) / remaining_overs = run_rate_required :=
by
  sorry

end cricket_run_rate_l263_263746


namespace number_equation_l263_263301

-- Lean statement equivalent to the mathematical problem
theorem number_equation (x : ℝ) (h : 5 * x - 2 * x = 10) : 5 * x - 2 * x = 10 :=
by exact h

end number_equation_l263_263301


namespace exponential_function_inequality_l263_263198

theorem exponential_function_inequality {a : ℝ} (h0 : 0 < a) (h1 : a < 1) :
  (a^3) * (a^2) < a^2 :=
by
  sorry

end exponential_function_inequality_l263_263198


namespace composite_function_evaluation_l263_263334

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem composite_function_evaluation : f (g 3) = 25 := by
  sorry

end composite_function_evaluation_l263_263334


namespace no_six_odd_numbers_sum_to_one_l263_263774

theorem no_six_odd_numbers_sum_to_one (a b c d e f : ℕ)
  (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) (hd : d % 2 = 1) (he : e % 2 = 1) (hf : f % 2 = 1)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) :
  (1 / a : ℝ) + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f ≠ 1 :=
by
  sorry

end no_six_odd_numbers_sum_to_one_l263_263774


namespace granola_bars_distribution_l263_263231

theorem granola_bars_distribution
  (total_bars : ℕ)
  (eaten_bars : ℕ)
  (num_children : ℕ)
  (remaining_bars := total_bars - eaten_bars)
  (bars_per_child := remaining_bars / num_children) :
  total_bars = 200 → eaten_bars = 80 → num_children = 6 → bars_per_child = 20 :=
by
  intros h1 h2 h3
  sorry

end granola_bars_distribution_l263_263231


namespace split_piles_equiv_single_stone_heaps_l263_263501

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l263_263501


namespace triangle_area_l263_263192

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_area :
  area_of_triangle 0 0 0 6 8 0 = 24 :=
by
  sorry

end triangle_area_l263_263192


namespace completing_square_eq_sum_l263_263009

theorem completing_square_eq_sum :
  ∃ (a b c : ℤ), a > 0 ∧ (∀ (x : ℝ), 36 * x^2 - 60 * x + 25 = (a * x + b)^2 - c) ∧ a + b + c = 26 :=
by
  sorry

end completing_square_eq_sum_l263_263009


namespace problem_a_b_c_d_l263_263042

open Real

/-- The main theorem to be proved -/
theorem problem_a_b_c_d
  (a b c d : ℝ)
  (hab : 0 < a) (hcd : 0 < c) (hab' : 0 < b) (hcd' : 0 < d)
  (h1 : a > c) (h2 : b < d)
  (h3 : a + sqrt b ≥ c + sqrt d)
  (h4 : sqrt a + b ≤ sqrt c + d) :
  a + b + c + d > 1 :=
by
  sorry

end problem_a_b_c_d_l263_263042


namespace carlton_outfits_l263_263783

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end carlton_outfits_l263_263783


namespace digit_sum_square_l263_263046

theorem digit_sum_square (n : ℕ) (hn : 0 < n) :
  let A := (4 * (10 ^ (2 * n) - 1)) / 9
  let B := (8 * (10 ^ n - 1)) / 9
  ∃ k : ℕ, A + 2 * B + 4 = k ^ 2 := 
by
  sorry

end digit_sum_square_l263_263046


namespace find_initial_number_of_girls_l263_263326

theorem find_initial_number_of_girls (b g : ℕ) : 
  (b = 3 * (g - 12)) ∧ (4 * (b - 36) = g - 12) → g = 25 :=
by
  intros h
  sorry

end find_initial_number_of_girls_l263_263326


namespace bear_hunting_l263_263411

theorem bear_hunting
    (mother_meat_req : ℕ) (cub_meat_req : ℕ) (num_cubs : ℕ) (num_animals_daily : ℕ)
    (weekly_meat_req : mother_meat_req = 210)
    (weekly_meat_per_cub : cub_meat_req = 35)
    (number_of_cubs : num_cubs = 4)
    (animals_hunted_daily : num_animals_daily = 10)
    (total_weekly_meat : mother_meat_req + num_cubs * cub_meat_req = 350) :
    ∃ w : ℕ, (w * num_animals_daily * 7 = 350) ∧ w = 5 :=
by
  sorry

end bear_hunting_l263_263411


namespace operation_proof_l263_263620

def operation (x y : ℤ) : ℤ := x * y - 3 * x - 4 * y

theorem operation_proof : (operation 7 2) - (operation 2 7) = 5 :=
by
  sorry

end operation_proof_l263_263620


namespace remaining_laps_l263_263525

theorem remaining_laps (total_laps_friday : ℕ)
                       (total_laps_saturday : ℕ)
                       (laps_sunday_morning : ℕ)
                       (total_required_laps : ℕ)
                       (total_laps_weekend : ℕ)
                       (remaining_laps : ℕ) :
  total_laps_friday = 63 →
  total_laps_saturday = 62 →
  laps_sunday_morning = 15 →
  total_required_laps = 198 →
  total_laps_weekend = total_laps_friday + total_laps_saturday + laps_sunday_morning →
  remaining_laps = total_required_laps - total_laps_weekend →
  remaining_laps = 58 := by
  intros
  sorry

end remaining_laps_l263_263525


namespace Carl_chops_more_onions_than_Brittney_l263_263777

theorem Carl_chops_more_onions_than_Brittney :
  let Brittney_rate := 15 / 5
  let Carl_rate := 20 / 5
  let Brittney_onions := Brittney_rate * 30
  let Carl_onions := Carl_rate * 30
  Carl_onions = Brittney_onions + 30 :=
by
  sorry

end Carl_chops_more_onions_than_Brittney_l263_263777


namespace angle_triple_supplementary_l263_263111

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l263_263111


namespace age_difference_l263_263308

variable (A J : ℕ)
variable (h1 : A + 5 = 40)
variable (h2 : J = 31)

theorem age_difference (h1 : A + 5 = 40) (h2 : J = 31) : A - J = 4 := by
  sorry

end age_difference_l263_263308


namespace discount_percentage_l263_263537

theorem discount_percentage (SP CP SP' discount_gain_percentage: ℝ) 
  (h1 : SP = 30) 
  (h2 : SP = CP + 0.25 * CP) 
  (h3 : SP' = CP + 0.125 * CP) 
  (h4 : discount_gain_percentage = ((SP - SP') / SP) * 100) :
  discount_gain_percentage = 10 :=
by
  -- Skipping the proof
  sorry

end discount_percentage_l263_263537


namespace concentration_of_salt_solution_l263_263150

-- Conditions:
def total_volume : ℝ := 1 + 0.25
def concentration_of_mixture : ℝ := 0.15
def volume_of_salt_solution : ℝ := 0.25

-- Expression for the concentration of the salt solution used, $C$:
theorem concentration_of_salt_solution (C : ℝ) :
  (volume_of_salt_solution * (C / 100)) = (total_volume * concentration_of_mixture) → C = 75 := by
  sorry

end concentration_of_salt_solution_l263_263150


namespace monthly_income_of_P_l263_263869

variable (P Q R : ℝ)

theorem monthly_income_of_P (h1 : (P + Q) / 2 = 5050) 
                           (h2 : (Q + R) / 2 = 6250) 
                           (h3 : (P + R) / 2 = 5200) : 
    P = 4000 := 
sorry

end monthly_income_of_P_l263_263869


namespace system1_l263_263867

theorem system1 {x y : ℝ} 
  (h1 : x + y = 3) 
  (h2 : x - y = 1) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end system1_l263_263867


namespace person_a_work_days_l263_263263

theorem person_a_work_days (x : ℝ) :
  (2 * (1 / x + 1 / 45) = 1 / 9) → (x = 30) :=
by
  sorry

end person_a_work_days_l263_263263


namespace simplified_expression_l263_263702

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end simplified_expression_l263_263702


namespace total_seats_l263_263916

theorem total_seats (s : ℕ) 
  (first_class : ℕ := 30) 
  (business_class : ℕ := (20 * s) / 100) 
  (premium_economy : ℕ := 15) 
  (economy_class : ℕ := s - first_class - business_class - premium_economy) 
  (total : first_class + business_class + premium_economy + economy_class = s) 
  : s = 288 := 
sorry

end total_seats_l263_263916


namespace machine_probabilities_at_least_one_first_class_component_l263_263579

theorem machine_probabilities : 
  (∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3) 
:=
sorry

theorem at_least_one_first_class_component : 
  ∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3 ∧ 
  1 - (1 - PA) * (1 - PB) * (1 - PC) = 5/6
:=
sorry

end machine_probabilities_at_least_one_first_class_component_l263_263579


namespace christen_peeled_20_potatoes_l263_263204

-- Define the conditions and question
def homer_rate : ℕ := 3
def time_alone : ℕ := 4
def christen_rate : ℕ := 5
def total_potatoes : ℕ := 44

noncomputable def christen_potatoes : ℕ :=
  (total_potatoes - (homer_rate * time_alone)) / (homer_rate + christen_rate) * christen_rate

theorem christen_peeled_20_potatoes :
  christen_potatoes = 20 := by
  -- Proof steps would go here
  sorry

end christen_peeled_20_potatoes_l263_263204


namespace no_solution_to_inequalities_l263_263057

theorem no_solution_to_inequalities :
  ∀ (x y z t : ℝ), 
    ¬ (|x| > |y - z + t| ∧
       |y| > |x - z + t| ∧
       |z| > |x - y + t| ∧
       |t| > |x - y + z|) :=
by
  intro x y z t
  sorry

end no_solution_to_inequalities_l263_263057


namespace find_four_digit_number_l263_263414

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l263_263414


namespace rectangular_garden_width_l263_263901

theorem rectangular_garden_width
  (w : ℝ)
  (h₁ : ∃ l, l = 3 * w)
  (h₂ : ∃ A, A = l * w ∧ A = 507) : 
  w = 13 :=
by
  sorry

end rectangular_garden_width_l263_263901


namespace number_equation_l263_263304

variable (x : ℝ)

theorem number_equation :
  5 * x - 2 * x = 10 :=
sorry

end number_equation_l263_263304


namespace complete_the_square_1_complete_the_square_2_complete_the_square_3_l263_263170

theorem complete_the_square_1 (x : ℝ) : 
  (x^2 - 2 * x + 3) = (x - 1)^2 + 2 :=
sorry

theorem complete_the_square_2 (x : ℝ) : 
  (3 * x^2 + 6 * x - 1) = 3 * (x + 1)^2 - 4 :=
sorry

theorem complete_the_square_3 (x : ℝ) : 
  (-2 * x^2 + 3 * x - 2) = -2 * (x - 3 / 4)^2 - 7 / 8 :=
sorry

end complete_the_square_1_complete_the_square_2_complete_the_square_3_l263_263170


namespace linda_max_servings_is_13_l263_263763

noncomputable def max_servings 
  (recipe_bananas : ℕ) (recipe_yogurt : ℕ) (recipe_honey : ℕ)
  (linda_bananas : ℕ) (linda_yogurt : ℕ) (linda_honey : ℕ)
  (servings_for_recipe : ℕ) : ℕ :=
  min 
    (linda_bananas * servings_for_recipe / recipe_bananas) 
    (min 
      (linda_yogurt * servings_for_recipe / recipe_yogurt)
      (linda_honey * servings_for_recipe / recipe_honey)
    )

theorem linda_max_servings_is_13 : 
  max_servings 3 2 1 10 9 4 4 = 13 :=
  sorry

end linda_max_servings_is_13_l263_263763


namespace game_ends_in_65_rounds_l263_263980

noncomputable def player_tokens_A : Nat := 20
noncomputable def player_tokens_B : Nat := 19
noncomputable def player_tokens_C : Nat := 18
noncomputable def player_tokens_D : Nat := 17

def rounds_until_game_ends (A B C D : Nat) : Nat :=
  -- Implementation to count the rounds will go here, but it is skipped for this statement-only task
  sorry

theorem game_ends_in_65_rounds : rounds_until_game_ends player_tokens_A player_tokens_B player_tokens_C player_tokens_D = 65 :=
  sorry

end game_ends_in_65_rounds_l263_263980


namespace problem1_problem2_l263_263597

-- Problem 1: Lean 4 Statement
theorem problem1 (n : ℕ) (hn : n > 0) : 20 ∣ (4 * 6^n + 5^(n + 1) - 9) :=
sorry

-- Problem 2: Lean 4 Statement
theorem problem2 : (3^100 % 7) = 4 :=
sorry

end problem1_problem2_l263_263597


namespace boys_in_class_l263_263724

theorem boys_in_class (students : ℕ) (ratio_girls_boys : ℕ → Prop)
  (h1 : students = 56)
  (h2 : ratio_girls_boys 4 ∧ ratio_girls_boys 3) :
  ∃ k : ℕ, 4 * k + 3 * k = students ∧ 3 * k = 24 :=
by
  sorry

end boys_in_class_l263_263724


namespace Jerry_walked_9_miles_l263_263678

theorem Jerry_walked_9_miles (x : ℕ) (h : 2 * x = 18) : x = 9 := 
by
  sorry

end Jerry_walked_9_miles_l263_263678


namespace sufficient_condition_l263_263399

theorem sufficient_condition 
  (x y z : ℤ)
  (H : x = y ∧ y = z)
  : x * (x - y) + y * (y - z) + z * (z - x) = 0 :=
by 
  sorry

end sufficient_condition_l263_263399


namespace annual_profit_function_correct_maximum_annual_profit_l263_263550

noncomputable def fixed_cost : ℝ := 60

noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 12 then 
    0.5 * x^2 + 4 * x 
  else 
    11 * x + 100 / x - 39

noncomputable def selling_price_per_thousand : ℝ := 10

noncomputable def sales_revenue (x : ℝ) : ℝ := selling_price_per_thousand * x

noncomputable def annual_profit (x : ℝ) : ℝ := sales_revenue x - fixed_cost - variable_cost x

theorem annual_profit_function_correct : 
∀ x : ℝ, (0 < x ∧ x < 12 → annual_profit x = -0.5 * x^2 + 6 * x - fixed_cost) ∧ 
        (x ≥ 12 → annual_profit x = -x - 100 / x + 33) :=
sorry

theorem maximum_annual_profit : 
∃ x : ℝ, x = 12 ∧ annual_profit x = 38 / 3 :=
sorry

end annual_profit_function_correct_maximum_annual_profit_l263_263550


namespace batter_sugar_is_one_l263_263229

-- Definitions based on the conditions given
def initial_sugar : ℕ := 3
def sugar_per_bag : ℕ := 6
def num_bags : ℕ := 2
def frosting_sugar_per_dozen : ℕ := 2
def total_dozen_cupcakes : ℕ := 5

-- Total sugar Lillian has
def total_sugar : ℕ := initial_sugar + num_bags * sugar_per_bag

-- Sugar needed for frosting
def frosting_sugar_needed : ℕ := frosting_sugar_per_dozen * total_dozen_cupcakes

-- Sugar used for the batter
def batter_sugar_total : ℕ := total_sugar - frosting_sugar_needed

-- Question asked in the problem
def batter_sugar_per_dozen : ℕ := batter_sugar_total / total_dozen_cupcakes

theorem batter_sugar_is_one :
  batter_sugar_per_dozen = 1 :=
by
  sorry -- Proof is not required here

end batter_sugar_is_one_l263_263229


namespace problem_statement_l263_263145

-- Define the universal set U, and sets A and B
def U : Set ℕ := { n | 1 ≤ n ∧ n ≤ 10 }
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of set A with respect to U
def complement_U_A : Set ℕ := { n | n ∈ U ∧ n ∉ A }

-- Define the intersection of complement_U_A and B
def intersection_complement_U_A_B : Set ℕ := { n | n ∈ complement_U_A ∧ n ∈ B }

-- Prove the given statement
theorem problem_statement : intersection_complement_U_A_B = {7, 9} := by
  sorry

end problem_statement_l263_263145


namespace angle_triple_supplement_l263_263118

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263118


namespace range_of_a_l263_263252

-- Given definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 1 

-- Monotonicity condition on the interval [1, 2]
def is_monotonic (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (x ≤ y → f x a ≤ f y a) ∨ (x ≤ y → f x a ≥ f y a)

-- The proof objective
theorem range_of_a (a : ℝ) : is_monotonic a → (a ≤ -2 ∨ a ≥ -1) := 
sorry

end range_of_a_l263_263252


namespace pool_full_capacity_is_2000_l263_263679

-- Definitions based on the conditions given
def water_loss_per_jump : ℕ := 400 -- in ml
def jumps_before_cleaning : ℕ := 1000
def cleaning_threshold : ℚ := 0.80 -- 80%
def total_water_loss : ℕ := water_loss_per_jump * jumps_before_cleaning -- in ml
def water_loss_liters : ℚ := total_water_loss / 1000 -- converting ml to liters
def cleaning_loss_fraction : ℚ := 1 - cleaning_threshold -- 20% loss

-- The actual proof statement
theorem pool_full_capacity_is_2000 :
  (water_loss_liters : ℚ) / cleaning_loss_fraction = 2000 :=
by
  sorry

end pool_full_capacity_is_2000_l263_263679


namespace inequality_result_l263_263809

theorem inequality_result
  (a b : ℝ) 
  (x y : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) :
  x + y ≤ 0 :=
sorry

end inequality_result_l263_263809


namespace abc_plus_2p_zero_l263_263258

variable (a b c p : ℝ)

-- Define the conditions
def cond1 : Prop := a + 2 / b = p
def cond2 : Prop := b + 2 / c = p
def cond3 : Prop := c + 2 / a = p
def nonzero_and_distinct : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main statement we want to prove
theorem abc_plus_2p_zero (h1 : cond1 a b p) (h2 : cond2 b c p) (h3 : cond3 c a p) (h4 : nonzero_and_distinct a b c) : 
  a * b * c + 2 * p = 0 := 
by 
  sorry

end abc_plus_2p_zero_l263_263258


namespace angle_triple_supplement_l263_263100

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l263_263100


namespace simplify_expression_l263_263538

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l263_263538


namespace equivalent_single_discount_rate_l263_263412

-- Definitions based on conditions
def original_price : ℝ := 120
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.15
def combined_discount_rate : ℝ := 0.3625  -- This is the expected result

-- The proof problem statement
theorem equivalent_single_discount_rate :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 
  (original_price * (1 - combined_discount_rate)) := 
sorry

end equivalent_single_discount_rate_l263_263412


namespace remaining_last_year_budget_is_13_l263_263767

-- Variables representing the conditions of the problem
variable (cost1 cost2 given_budget remaining this_year_spent remaining_last_year : ℤ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  cost1 = 13 ∧ cost2 = 24 ∧ 
  given_budget = 50 ∧ 
  remaining = 19 ∧ 
  (cost1 + cost2 = 37) ∧
  (this_year_spent = given_budget - remaining) ∧
  (remaining_last_year + (cost1 + cost2 - this_year_spent) = remaining)

-- The statement that needs to be proven
theorem remaining_last_year_budget_is_13 : conditions cost1 cost2 given_budget remaining this_year_spent remaining_last_year → remaining_last_year = 13 :=
by 
  intro h
  sorry

end remaining_last_year_budget_is_13_l263_263767


namespace line_condition_l263_263214

variable (m n Q : ℝ)

theorem line_condition (h1: m = 8 * n + 5) 
                       (h2: m + Q = 8 * (n + 0.25) + 5) 
                       (h3: p = 0.25) : Q = 2 :=
by
  sorry

end line_condition_l263_263214


namespace complete_the_square_l263_263991

theorem complete_the_square (x : ℝ) :
  (x^2 + 14*x + 60) = ((x + 7) ^ 2 + 11) :=
by
  sorry

end complete_the_square_l263_263991


namespace work_problem_l263_263413

theorem work_problem (days_B : ℝ) (h : (1 / 20) + (1 / days_B) = 1 / 8.571428571428571) : days_B = 15 :=
sorry

end work_problem_l263_263413


namespace min_value_of_expression_l263_263639

theorem min_value_of_expression (x y : ℝ) (hx : x > y) (hy : y > 0) (hxy : x + y ≤ 2) :
  ∃ m : ℝ, m = (2 / (x + 3 * y) + 1 / (x - y)) ∧ m = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_of_expression_l263_263639


namespace simplest_form_is_C_l263_263592

variables (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0)

def fraction_A := 3 * x * y / (x^2)
def fraction_B := (x - 1) / (x^2 - 1)
def fraction_C := (x + y) / (2 * x)
def fraction_D := (1 - x) / (x - 1)

theorem simplest_form_is_C : 
  ∀ (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0), 
  ¬ (3 * x * y / (x^2)).is_simplest ∧ 
  ¬ ((x - 1) / (x^2 - 1)).is_simplest ∧ 
  (x + y) / (2 * x).is_simplest ∧ 
  ¬ ((1 - x) / (x - 1)).is_simplest :=
by 
  sorry

end simplest_form_is_C_l263_263592


namespace weeks_project_lasts_l263_263013

-- Definition of the conditions
def meal_cost : ℤ := 4
def people : ℤ := 4
def days_per_week : ℤ := 5
def total_spent : ℤ := 1280
def weekly_cost : ℤ := meal_cost * people * days_per_week

-- Problem statement: prove that the number of weeks the project will last equals 16 weeks.
theorem weeks_project_lasts : total_spent / weekly_cost = 16 := by 
  sorry

end weeks_project_lasts_l263_263013


namespace ex1_ex2_l263_263894

-- Definition of the "multiplication-subtraction" operation.
def mult_sub (a b : ℚ) : ℚ :=
  if a = 0 then abs b else if b = 0 then abs a else if abs a = abs b then 0 else
  if (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) then abs a - abs b else -(abs a - abs b)

theorem ex1 : mult_sub (mult_sub (3) (-2)) (mult_sub (-9) 0) = -8 :=
  sorry

theorem ex2 : ∃ (a b c : ℚ), (mult_sub (mult_sub a b) c) ≠ (mult_sub a (mult_sub b c)) :=
  ⟨3, -2, 4, by simp [mult_sub]; sorry⟩

end ex1_ex2_l263_263894


namespace find_sum_uv_l263_263955

theorem find_sum_uv (u v : ℝ) (h1 : 3 * u - 7 * v = 29) (h2 : 5 * u + 3 * v = -9) : u + v = -3.363 := 
sorry

end find_sum_uv_l263_263955


namespace students_participated_in_both_l263_263609

theorem students_participated_in_both (total_students volleyball track field no_participation both: ℕ) 
  (h1 : total_students = 45) 
  (h2 : volleyball = 12) 
  (h3 : track = 20) 
  (h4 : no_participation = 19) 
  (h5 : both = volleyball + track - (total_students - no_participation)) 
  : both = 6 :=
by
  sorry

end students_participated_in_both_l263_263609


namespace radius_of_sphere_l263_263766

theorem radius_of_sphere {r x : ℝ} (h1 : 15^2 + x^2 = r^2) (h2 : r = x + 12) :
    r = 123 / 8 :=
  by
  sorry

end radius_of_sphere_l263_263766


namespace cone_section_area_half_base_ratio_l263_263989

theorem cone_section_area_half_base_ratio (h_base h_upper h_lower : ℝ) (A_base A_upper : ℝ) 
  (h_total : h_upper + h_lower = h_base)
  (A_upper : A_upper = A_base / 2) :
  h_upper = h_lower :=
by
  sorry

end cone_section_area_half_base_ratio_l263_263989


namespace am_gm_inequality_example_l263_263824

theorem am_gm_inequality_example (x y : ℝ) (hx : x = 16) (hy : y = 64) : 
  (x + y) / 2 ≥ Real.sqrt (x * y) :=
by
  rw [hx, hy]
  sorry

end am_gm_inequality_example_l263_263824


namespace missed_questions_l263_263524

-- Define variables
variables (a b c T : ℕ) (X Y Z : ℝ)
variables (h1 : a + b + c = T) 
          (h2 : 0 ≤ X ∧ X ≤ 100) 
          (h3 : 0 ≤ Y ∧ Y ≤ 100) 
          (h4 : 0 ≤ Z ∧ Z ≤ 100) 
          (h5 : 6 * (a * (100 - X) / 500 + 2 * b * (100 - Y) / 500 + 3 * c * (100 - Z) / 500) = 216)

-- Define the theorem
theorem missed_questions : 5 * (a * (100 - X) / 500 + b * (100 - Y) / 500 + c * (100 - Z) / 500) = 180 :=
by sorry

end missed_questions_l263_263524


namespace find_a1_and_d_l263_263638

-- Given conditions
variables {a : ℕ → ℤ} 
variables {a1 d : ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem find_a1_and_d 
  (h1 : is_arithmetic_sequence a a1 d)
  (h2 : (a 3) * (a 7) = -16)
  (h3 : (a 4) + (a 6) = 0)
  : (a1 = -8 ∧ d = 2) ∨ (a1 = 8 ∧ d = -2) :=
sorry

end find_a1_and_d_l263_263638


namespace enjoyable_gameplay_l263_263351

theorem enjoyable_gameplay (total_hours : ℕ) (boring_percentage : ℕ) (expansion_hours : ℕ)
  (h_total : total_hours = 100)
  (h_boring : boring_percentage = 80)
  (h_expansion : expansion_hours = 30) :
  ((1 - boring_percentage / 100) * total_hours + expansion_hours) = 50 := 
by
  sorry

end enjoyable_gameplay_l263_263351


namespace captain_age_is_24_l263_263248

theorem captain_age_is_24 (C W : ℕ) 
  (hW : W = C + 7)
  (h_total_team_age : 23 * 11 = 253)
  (h_total_9_players_age : 22 * 9 = 198)
  (h_team_age_equation : 253 = 198 + C + W)
  : C = 24 :=
sorry

end captain_age_is_24_l263_263248


namespace milk_production_l263_263713

variable (a b c d e : ℝ)

theorem milk_production (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  let rate_per_cow_per_day := b / (a * c)
  let production_per_day := d * rate_per_cow_per_day
  let total_production := production_per_day * e
  total_production = (b * d * e) / (a * c) :=
by
  sorry

end milk_production_l263_263713


namespace total_number_of_stickers_l263_263698

def sticker_count (sheets : ℕ) (stickers_per_sheet : ℕ) : ℕ := sheets * stickers_per_sheet

theorem total_number_of_stickers 
    (sheets_per_folder : ℕ)
    (red_folder_stickers_per_sheet : ℕ)
    (green_folder_stickers_per_sheet : ℕ)
    (blue_folder_stickers_per_sheet : ℕ) :
    sticker_count sheets_per_folder red_folder_stickers_per_sheet +
    sticker_count sheets_per_folder green_folder_stickers_per_sheet +
    sticker_count sheets_per_folder blue_folder_stickers_per_sheet = 60 := 
begin
    -- Given conditions
    let sheets := 10      -- Each folder contains 10 sheets of paper.
    let red := 3          -- Each sheet in the red folder gets 3 stickers.
    let green := 2        -- Each sheet in the green folder gets 2 stickers.
    let blue := 1         -- Each sheet in the blue folder gets 1 sticker.
    have h1 : sticker_count sheets red = 30, by sorry, -- Calculation omitted
    have h2 : sticker_count sheets green = 20, by sorry, -- Calculation omitted
    have h3 : sticker_count sheets blue = 10, by sorry, -- Calculation omitted

    -- Summing the stickers
    show h1 + h2 + h3 = 60, by sorry
end

end total_number_of_stickers_l263_263698


namespace derivative_at_3_l263_263194

noncomputable def f (x : ℝ) := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l263_263194


namespace packets_of_candy_bought_l263_263444

theorem packets_of_candy_bought
    (candies_per_day_weekday : ℕ)
    (candies_per_day_weekend : ℕ)
    (days_weekday : ℕ)
    (days_weekend : ℕ)
    (weeks : ℕ)
    (candies_per_packet : ℕ)
    (total_candies : ℕ)
    (packets_bought : ℕ) :
    candies_per_day_weekday = 2 →
    candies_per_day_weekend = 1 →
    days_weekday = 5 →
    days_weekend = 2 →
    weeks = 3 →
    candies_per_packet = 18 →
    total_candies = (candies_per_day_weekday * days_weekday + candies_per_day_weekend * days_weekend) * weeks →
    packets_bought = total_candies / candies_per_packet →
    packets_bought = 2 :=
by
  intros
  sorry

end packets_of_candy_bought_l263_263444


namespace fraction_power_multiplication_l263_263172

theorem fraction_power_multiplication :
  ((1 : ℝ) / 3) ^ 4 * ((1 : ℝ) / 5) = ((1 : ℝ) / 405) := by
  sorry

end fraction_power_multiplication_l263_263172


namespace non_zero_number_is_nine_l263_263717

theorem non_zero_number_is_nine {x : ℝ} (h1 : (x + x^2) / 2 = 5 * x) (h2 : x ≠ 0) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l263_263717


namespace wallet_amount_l263_263048

-- Definitions of given conditions
def num_toys := 28
def cost_per_toy := 10
def num_teddy_bears := 20
def cost_per_teddy_bear := 15

-- Calculation of total costs
def total_cost_of_toys := num_toys * cost_per_toy
def total_cost_of_teddy_bears := num_teddy_bears * cost_per_teddy_bear

-- Total amount of money in Louise's wallet
def total_cost := total_cost_of_toys + total_cost_of_teddy_bears

-- Proof that the total cost is $580
theorem wallet_amount : total_cost = 580 :=
by
  -- Skipping the proof for now
  sorry

end wallet_amount_l263_263048


namespace f_m_minus_1_pos_l263_263692

variable {R : Type*} [LinearOrderedField R]

def quadratic_function (x a : R) : R :=
  x^2 - x + a

theorem f_m_minus_1_pos {a m : R} (h_pos : 0 < a) (h_fm : quadratic_function m a < 0) :
  quadratic_function (m - 1 : R) a > 0 :=
sorry

end f_m_minus_1_pos_l263_263692


namespace spotted_and_fluffy_cats_l263_263770

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_eq : total_cats = 120) 
  (spotted_fraction : ℚ) (spotted_fraction_eq : spotted_fraction = 1/3)
  (fluffy_fraction : ℚ) (fluffy_fraction_eq : fluffy_fraction = 1/4) :
  let spotted_cats := (total_cats * spotted_fraction).natAbs in
  let fluffy_spotted_cats := (spotted_cats * fluffy_fraction).natAbs in
  fluffy_spotted_cats = 10 :=
by
  sorry

end spotted_and_fluffy_cats_l263_263770


namespace remainder_when_n_plus_5040_divided_by_7_l263_263338

theorem remainder_when_n_plus_5040_divided_by_7 (n : ℤ) (h: n % 7 = 2) : (n + 5040) % 7 = 2 :=
by
  sorry

end remainder_when_n_plus_5040_divided_by_7_l263_263338


namespace yellow_marbles_in_C_l263_263393

theorem yellow_marbles_in_C 
  (Y : ℕ)
  (conditionA : 4 - 2 ≠ 6)
  (conditionB : 6 - 1 ≠ 6)
  (conditionC1 : 3 > Y → 3 - Y = 6)
  (conditionC2 : Y > 3 → Y - 3 = 6) :
  Y = 9 :=
by
  sorry

end yellow_marbles_in_C_l263_263393


namespace average_annual_percent_change_l263_263381

-- Define the initial and final population, and the time period
def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def decade_years : ℕ := 10

-- Define the theorem to find the resulting average percent change per year
theorem average_annual_percent_change
    (P₀ : ℕ := initial_population)
    (P₁₀ : ℕ := final_population)
    (years : ℕ := decade_years) :
    ((P₁₀ - P₀ : ℝ) / P₀ * 100) / years = 7 := by
        sorry

end average_annual_percent_change_l263_263381


namespace train_speed_is_correct_l263_263428

noncomputable def train_length : ℕ := 900
noncomputable def platform_length : ℕ := train_length
noncomputable def time_in_minutes : ℕ := 1
noncomputable def distance_covered : ℕ := train_length + platform_length
noncomputable def speed_m_per_minute : ℕ := distance_covered / time_in_minutes
noncomputable def speed_km_per_hr : ℕ := (speed_m_per_minute * 60) / 1000

theorem train_speed_is_correct :
  speed_km_per_hr = 108 :=
by
  sorry

end train_speed_is_correct_l263_263428


namespace solution1_solution2_solution3_solution4_solution5_l263_263312

noncomputable def problem1 : ℤ :=
  -3 + 8 - 15 - 6

theorem solution1 : problem1 = -16 := by
  sorry

noncomputable def problem2 : ℚ :=
  -35 / -7 * (-1 / 7)

theorem solution2 : problem2 = -(5 / 7) := by
  sorry

noncomputable def problem3 : ℤ :=
  -2^2 - |2 - 5| / -3

theorem solution3 : problem3 = -3 := by
  sorry

noncomputable def problem4 : ℚ :=
  (1 / 2 + 5 / 6 - 7 / 12) * -24 

theorem solution4 : problem4 = -18 := by
  sorry

noncomputable def problem5 : ℚ :=
  (-99 - 6 / 11) * 22

theorem solution5 : problem5 = -2190 := by
  sorry

end solution1_solution2_solution3_solution4_solution5_l263_263312


namespace part_a_part_b_l263_263898

-- Part a: Prove for specific numbers 2015 and 2017
theorem part_a : ∃ (x y : ℕ), (2015^2 + 2017^2) / 2 = x^2 + y^2 := sorry

-- Part b: Prove for any two different odd natural numbers
theorem part_b (a b : ℕ) (h1 : a ≠ b) (h2 : a % 2 = 1) (h3 : b % 2 = 1) :
  ∃ (x y : ℕ), (a^2 + b^2) / 2 = x^2 + y^2 := sorry

end part_a_part_b_l263_263898


namespace village_population_l263_263754

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 := by
  sorry

end village_population_l263_263754


namespace geo_sequence_sum_l263_263675

theorem geo_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 2)
  (h2 : a 4 + a 5 = 4)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  a 10 + a 11 = 16 := by
  -- Insert proof here
  sorry  -- skipping the proof

end geo_sequence_sum_l263_263675


namespace statement_C_l263_263133

theorem statement_C (x : ℝ) (h : x^2 < 4) : x < 2 := 
sorry

end statement_C_l263_263133


namespace non_deg_ellipse_condition_l263_263993

theorem non_deg_ellipse_condition (k : ℝ) : k > -19 ↔ 
  (∃ x y : ℝ, 3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k) :=
sorry

end non_deg_ellipse_condition_l263_263993


namespace angle_triple_supplement_l263_263115

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263115


namespace geometric_sequence_sum_9000_l263_263878

noncomputable def sum_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_9000 (a r : ℝ) (h : r ≠ 1) 
  (h1 : sum_geometric_sequence a r 3000 = 1000)
  (h2 : sum_geometric_sequence a r 6000 = 1900) : 
  sum_geometric_sequence a r 9000 = 2710 :=
sorry

end geometric_sequence_sum_9000_l263_263878


namespace inscribed_circle_radius_eq_3_l263_263211

open Real

theorem inscribed_circle_radius_eq_3
  (a : ℝ) (A : ℝ) (p : ℝ) (r : ℝ)
  (h_eq_tri : ∀ (a : ℝ), A = (sqrt 3 / 4) * a^2)
  (h_perim : ∀ (a : ℝ), p = 3 * a)
  (h_area_perim : ∀ (a : ℝ), A = (3 / 2) * p) :
  r = 3 :=
by sorry

end inscribed_circle_radius_eq_3_l263_263211


namespace mikes_earnings_l263_263856

-- Definitions based on the conditions:
def blade_cost : ℕ := 47
def game_count : ℕ := 9
def game_cost : ℕ := 6

-- The total money Mike made:
def total_money (M : ℕ) : Prop :=
  M - (blade_cost + game_count * game_cost) = 0

theorem mikes_earnings (M : ℕ) : total_money M → M = 101 :=
by
  sorry

end mikes_earnings_l263_263856


namespace abc_ineq_l263_263063

theorem abc_ineq (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
by 
  sorry

end abc_ineq_l263_263063


namespace marians_groceries_l263_263367

variables (G : ℝ)

theorem marians_groceries :
  let initial_balance := 126
  let returned_amount := 45
  let new_balance := 171
  let gas_expense := G / 2
  initial_balance + G + gas_expense - returned_amount = new_balance → G = 60 :=
sorry

end marians_groceries_l263_263367


namespace charlie_age_when_jenny_twice_as_old_as_bobby_l263_263219

-- Conditions as Definitions
def ageDifferenceJennyCharlie : ℕ := 5
def ageDifferenceCharlieBobby : ℕ := 3

-- Problem Statement as a Theorem
theorem charlie_age_when_jenny_twice_as_old_as_bobby (j c b : ℕ) 
  (H1 : j = c + ageDifferenceJennyCharlie) 
  (H2 : c = b + ageDifferenceCharlieBobby) : 
  j = 2 * b → c = 11 :=
by
  sorry

end charlie_age_when_jenny_twice_as_old_as_bobby_l263_263219


namespace problem_statement_l263_263469

theorem problem_statement {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) ∧ 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) < 2 :=
by
  sorry

end problem_statement_l263_263469


namespace final_bill_correct_l263_263296

def initial_bill := 500.00
def late_charge_rate := 0.02
def final_bill := initial_bill * (1 + late_charge_rate) * (1 + late_charge_rate)

theorem final_bill_correct : final_bill = 520.20 := by
  sorry

end final_bill_correct_l263_263296


namespace probability_of_selecting_female_l263_263298

theorem probability_of_selecting_female (total_students female_students male_students : ℕ)
  (h_total : total_students = female_students + male_students)
  (h_female : female_students = 3)
  (h_male : male_students = 1) :
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end probability_of_selecting_female_l263_263298


namespace no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l263_263371

theorem no_positive_integer_solutions_m2_m3 (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (∃ m, m = 2 ∨ m = 3 → (x / y + y / z + z / t + t / x = m) → false) :=
sorry

theorem positive_integer_solutions_m4 (x y z t : ℕ) :
  x / y + y / z + z / t + t / x = 4 ↔ ∃ k : ℕ, k > 0 ∧ (x = k ∧ y = k ∧ z = k ∧ t = k) :=
sorry

end no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l263_263371


namespace remainder_of_polynomial_l263_263365

theorem remainder_of_polynomial 
  (P : ℝ → ℝ) 
  (h₁ : P 15 = 16)
  (h₂ : P 10 = 4) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 10) * (x - 15) * Q x + (12 / 5 * x - 20) :=
by
  sorry

end remainder_of_polynomial_l263_263365


namespace simplify_expression_l263_263539

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l263_263539


namespace colored_pencils_more_than_erasers_l263_263528

def colored_pencils_initial := 67
def erasers_initial := 38

def colored_pencils_final := 50
def erasers_final := 28

theorem colored_pencils_more_than_erasers :
  colored_pencils_final - erasers_final = 22 := by
  sorry

end colored_pencils_more_than_erasers_l263_263528


namespace not_true_n_gt_24_l263_263045

theorem not_true_n_gt_24 (n : ℕ) (h : 1/3 + 1/4 + 1/6 + 1/n = 1) : n ≤ 24 := 
by
  -- Placeholder for the proof
  sorry

end not_true_n_gt_24_l263_263045


namespace problem_part1_problem_part2_l263_263632

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def f (x : ℝ) : ℝ :=
  dot_product (Real.cos x, Real.cos x) (Real.sqrt 3 * Real.cos x, Real.sin x)

theorem problem_part1 :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, (x ∈ Set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) → MonotoneOn f (Set.Icc (k * π + π / 12) (k * π + 7 * π / 12))) :=
sorry

theorem problem_part2 (A : ℝ) (a b c : ℝ) (area : ℝ) :
  f (A / 2 - π / 6) = Real.sqrt 3 ∧ 
  c = 2 ∧ 
  area = 2 * Real.sqrt 3 →
  a = 2 * Real.sqrt 3 ∨ a = 2 * Real.sqrt 7 :=
sorry

end problem_part1_problem_part2_l263_263632


namespace fraction_of_25_l263_263753

theorem fraction_of_25 (x : ℝ) (h1 : 0.65 * 40 = 26) (h2 : 26 = x * 25 + 6) : x = 4 / 5 :=
sorry

end fraction_of_25_l263_263753


namespace problem_equivalence_l263_263191

-- Define the given circles and their properties
def E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C as the trajectory of the center of the moving circle P
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l intersecting curve C at points A and B with midpoint M(1,1)
def M (A B : ℝ × ℝ) : Prop := (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 2)
def l (x y : ℝ) : Prop := x + 4 * y - 5 = 0

theorem problem_equivalence :
  (∀ x y, E x y ∧ F x y → C x y) ∧
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ M A B → (∀ x y, l x y)) :=
sorry

end problem_equivalence_l263_263191


namespace students_not_picked_l263_263884

theorem students_not_picked (total_students groups group_size : ℕ) (h1 : total_students = 64)
(h2 : groups = 4) (h3 : group_size = 7) :
total_students - groups * group_size = 36 :=
by
  sorry

end students_not_picked_l263_263884


namespace circle_equation_l263_263605

theorem circle_equation :
  ∃ (C : ℝ × ℝ) (r : ℝ), C = (3, 0) ∧ r = sqrt 2 ∧
    (∀ (x y : ℝ), (x - 3)^2 + y^2 = 2 ↔ (C.1 - x)^2 + (C.2 - y)^2 = r^2) ∧
    (∃ (A B : ℝ × ℝ), A = (4, 1) ∧ B = (2, 1) ∧ 
     ((∃ (m : ℝ), m ≠ 0 ∧ (x - y = 1)) ∧
     (∃ (m : ℝ), m ≠ 0 ∧ (x + y = 3)))) :=
begin
  sorry
end

end circle_equation_l263_263605


namespace polar_to_cartesian_l263_263007

theorem polar_to_cartesian (θ : ℝ) (ρ : ℝ) (x y : ℝ) :
  (ρ = 2 * Real.sin θ + 4 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - 8)^2 + (y - 2)^2 = 68 :=
by
  intros hρ hx hy
  -- Proof steps would go here
  sorry

end polar_to_cartesian_l263_263007


namespace candidates_count_l263_263896

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 := 
sorry

end candidates_count_l263_263896


namespace bowling_average_l263_263152

theorem bowling_average (A : ℝ) (W : ℕ) (hW : W = 145) (hW7 : W + 7 ≠ 0)
  (h : ( A * W + 26 ) / ( W + 7 ) = A - 0.4) : A = 12.4 := 
by 
  sorry

end bowling_average_l263_263152


namespace smallest_fourth_number_l263_263281

-- Define the given conditions
def first_three_numbers_sum : ℕ := 28 + 46 + 59 
def sum_of_digits_of_first_three_numbers : ℕ := 2 + 8 + 4 + 6 + 5 + 9 

-- Define the condition for the fourth number represented as 10a + b and its digits 
def satisfies_condition (a b : ℕ) : Prop := 
  first_three_numbers_sum + 10 * a + b = 4 * (sum_of_digits_of_first_three_numbers + a + b)

-- Statement to prove the smallest fourth number
theorem smallest_fourth_number : ∃ (a b : ℕ), satisfies_condition a b ∧ 10 * a + b = 11 := 
sorry

end smallest_fourth_number_l263_263281


namespace jose_profit_share_l263_263581

def investment_share (toms_investment : ℕ) (jose_investment : ℕ) 
  (toms_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) : ℕ :=
  let toms_capital_months := toms_investment * toms_duration
  let jose_capital_months := jose_investment * jose_duration
  let total_capital_months := toms_capital_months + jose_capital_months
  let jose_share_ratio := jose_capital_months / total_capital_months
  jose_share_ratio * total_profit

theorem jose_profit_share 
  (toms_investment : ℕ := 3000)
  (jose_investment : ℕ := 4500)
  (toms_duration : ℕ := 12)
  (jose_duration : ℕ := 10)
  (total_profit : ℕ := 6300) :
  investment_share toms_investment jose_investment toms_duration jose_duration total_profit = 3500 := 
sorry

end jose_profit_share_l263_263581


namespace FGH_supermarkets_total_l263_263391

theorem FGH_supermarkets_total 
  (us_supermarkets : ℕ)
  (ca_supermarkets : ℕ)
  (h1 : us_supermarkets = 41)
  (h2 : us_supermarkets = ca_supermarkets + 22) :
  us_supermarkets + ca_supermarkets = 60 :=
by
  sorry

end FGH_supermarkets_total_l263_263391


namespace total_spent_correct_l263_263060

def shorts : ℝ := 13.99
def shirt : ℝ := 12.14
def jacket : ℝ := 7.43
def total_spent : ℝ := 33.56

theorem total_spent_correct : shorts + shirt + jacket = total_spent :=
by
  sorry

end total_spent_correct_l263_263060


namespace ants_in_park_l263_263422

theorem ants_in_park:
  let width_meters := 100
  let length_meters := 130
  let cm_per_meter := 100
  let ants_per_sq_cm := 1.2
  let width_cm := width_meters * cm_per_meter
  let length_cm := length_meters * cm_per_meter
  let area_sq_cm := width_cm * length_cm
  let total_ants := ants_per_sq_cm * area_sq_cm
  total_ants = 156000000 := by
  sorry

end ants_in_park_l263_263422


namespace fraction_operation_l263_263587

theorem fraction_operation : 
  let a := (2 : ℚ) / 9
  let b := (5 : ℚ) / 6
  let c := (1 : ℚ) / 18
  (a * b) + c = 13 / 54 :=
by
  sorry

end fraction_operation_l263_263587


namespace split_into_similar_piles_l263_263516

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l263_263516


namespace condition_for_equation_l263_263975

theorem condition_for_equation (a b c : ℕ) (ha : 0 < a ∧ a < 20) (hb : 0 < b ∧ b < 20) (hc : 0 < c ∧ c < 20) :
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 :=
by
  sorry

end condition_for_equation_l263_263975


namespace angle_triple_supplement_l263_263124

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263124


namespace angle_triple_supplement_l263_263116

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263116


namespace audrey_peaches_l263_263775

variable (A : ℕ)
variable (P : ℕ := 48)
variable (D : ℕ := 22)

theorem audrey_peaches : A - P = D → A = 70 :=
by
  intro h
  sorry

end audrey_peaches_l263_263775


namespace blue_marbles_difference_l263_263088

theorem blue_marbles_difference  (a b : ℚ) 
  (h1 : 3 * a + 2 * b = 80)
  (h2 : 2 * a = b) :
  (7 * a - 3 * b) = 80 / 7 := by
  sorry

end blue_marbles_difference_l263_263088


namespace smallest_base10_integer_l263_263265

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l263_263265


namespace complement_of_M_in_U_is_14_l263_263047

def U : Set ℕ := {x | x < 5 ∧ x > 0}

def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

theorem complement_of_M_in_U_is_14 : 
  {x | x ∈ U ∧ x ∉ M} = {1, 4} :=
by
  sorry

end complement_of_M_in_U_is_14_l263_263047


namespace equilateral_triangle_area_perimeter_l263_263868

theorem equilateral_triangle_area_perimeter (altitude : ℝ) : 
  altitude = Real.sqrt 12 →
  (exists area perimeter : ℝ, area = 4 * Real.sqrt 3 ∧ perimeter = 12) :=
by
  intro h_alt
  sorry

end equilateral_triangle_area_perimeter_l263_263868


namespace hyperbola_transverse_axis_l263_263478

noncomputable def hyperbola_transverse_axis_length (a b : ℝ) : ℝ :=
  2 * a

theorem hyperbola_transverse_axis {a b : ℝ} (h : a > 0) (h_b : b > 0) 
  (eccentricity_cond : Real.sqrt 2 = Real.sqrt (1 + b^2 / a^2))
  (area_cond : ∃ x y : ℝ, x^2 = -4 * Real.sqrt 3 * y ∧ y * y / a^2 - x^2 / b^2 = 1 ∧ 
                 Real.sqrt 3 = 1 / 2 * (2 * Real.sqrt (3 - a^2)) * Real.sqrt 3) :
  hyperbola_transverse_axis_length a b = 2 * Real.sqrt 2 :=
by
  sorry

end hyperbola_transverse_axis_l263_263478


namespace find_g_five_l263_263493

def g (a b c x : ℝ) : ℝ := a * x^7 + b * x^6 + c * x - 3

theorem find_g_five (a b c : ℝ) (h : g a b c (-5) = -3) : g a b c 5 = 31250 * b - 3 := 
sorry

end find_g_five_l263_263493


namespace odd_function_property_l263_263819

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x * Real.log (2 - x) else -x * Real.log (2 + x)

theorem odd_function_property (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_neg_interval : ∀ x : ℝ, x < 0 → f x = -x * Real.log (2 - x)) :
  ∀ x : ℝ, f x = (if x < 0 then -x * Real.log (2 - x) else -x * Real.log (2 + x)) :=
by
  sorry

end odd_function_property_l263_263819


namespace train_length_calculation_l263_263427

theorem train_length_calculation 
  (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) 
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 25) 
  (h_train_speed_kmph : train_speed_kmph = 57.6) : 
  ∃ train_length, train_length = 250 :=
by
  sorry

end train_length_calculation_l263_263427


namespace largest_invertible_interval_l263_263930

def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

theorem largest_invertible_interval (x : ℝ) (hx : x = 2) : 
  ∃ I : Set ℝ, (I = Set.univ ∩ {y | y ≥ 3 / 2}) ∧ ∀ y ∈ I, g y = 3 * (y - 3 / 2) ^ 2 - 11 / 4 ∧ g y ∈ I ∧ Function.Injective (g ∘ (fun z => z : I → ℝ)) :=
sorry

end largest_invertible_interval_l263_263930


namespace range_of_a_l263_263830

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + (a-1) * x + 1 ≤ 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l263_263830


namespace find_four_digit_number_l263_263415

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l263_263415


namespace find_a_20_l263_263335

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Definitions: The sequence is geometric: a_n = a_1 * r^(n-1)
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 1 * r^(n-1)

-- Conditions in the problem: a_10 and a_30 satisfy the quadratic equation
def satisfies_quadratic_roots (a10 a30 : ℝ) : Prop :=
  a10 + a30 = 11 ∧ a10 * a30 = 16

-- Question: Find a_20
theorem find_a_20 (h1 : is_geometric_sequence a r)
                  (h2 : satisfies_quadratic_roots (a 10) (a 30)) :
  a 20 = 4 :=
sorry

end find_a_20_l263_263335


namespace fencing_cost_l263_263994

theorem fencing_cost (L B: ℝ) (cost_per_meter : ℝ) (H1 : L = 58) (H2 : L = B + 16) (H3 : cost_per_meter = 26.50) : 
    2 * (L + B) * cost_per_meter = 5300 := by
  sorry

end fencing_cost_l263_263994


namespace triangle_parallel_side_l263_263360

variable {A B C E N M : Point}

-- Assume a triangle ABC
variables (hABC : Triangle A B C)
-- E is a point on AC, and N and M are defined as per the problem conditions
variables (hE : E ∈ Line A C)
          (l : Line)
          (hN : N ∈ l)
          (hM : M ∈ l)
          (hEN_BC : parallel (line_through E N) (Line_through B C))
          (hEM_AB : parallel (line_through E M) (line_through A B))

theorem triangle_parallel_side (hABC : Triangle A B C) (hE : E ∈ line_through A C)
    (hEN_BC : parallel (line_through E N) (Line_through B C))
    (hEM_AB : parallel (line_through E M) (line_through A B)) :
    parallel (line_through A N) (Line_through C M) := by
  sorry

end triangle_parallel_side_l263_263360


namespace fixed_point_on_line_AC_l263_263479

-- Given definitions and conditions directly from a)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line_through_P (x y : ℝ) : Prop := ∃ t : ℝ, x = t * y - 1
def reflection_across_x_axis (y : ℝ) : ℝ := -y

-- The final proof statement translating c)
theorem fixed_point_on_line_AC
  (A B C P : ℝ × ℝ)
  (hP : P = (-1, 0))
  (hA : parabola A.1 A.2)
  (hB : parabola B.1 B.2)
  (hAB : ∃ t : ℝ, line_through_P A.1 A.2 ∧ line_through_P B.1 B.2)
  (hRef : C = (B.1, reflection_across_x_axis B.2)) :
  ∃ x y : ℝ, (x, y) = (1, 0) ∧ line_through_P x y := 
sorry

end fixed_point_on_line_AC_l263_263479


namespace average_salary_all_workers_l263_263071

-- Define the given conditions as constants
def num_technicians : ℕ := 7
def avg_salary_technicians : ℕ := 12000

def num_workers_total : ℕ := 21
def num_workers_remaining := num_workers_total - num_technicians
def avg_salary_remaining_workers : ℕ := 6000

-- Define the statement we need to prove
theorem average_salary_all_workers :
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining_workers := num_workers_remaining * avg_salary_remaining_workers
  let total_salary_all_workers := total_salary_technicians + total_salary_remaining_workers
  let avg_salary_all_workers := total_salary_all_workers / num_workers_total
  avg_salary_all_workers = 8000 :=
by
  sorry

end average_salary_all_workers_l263_263071


namespace problem1_problem2_l263_263368

-- Problem 1: Sequence "Seven six five four three two one" is a descending order
theorem problem1 : ∃ term: String, term = "Descending Order" ∧ "Seven six five four three two one" = "Descending Order" := sorry

-- Problem 2: Describing a computing tool that knows 0 and 1 and can calculate large numbers (computer)
theorem problem2 : ∃ tool: String, tool = "Computer" ∧ "I only know 0 and 1, can calculate millions and billions, available in both software and hardware" = "Computer" := sorry

end problem1_problem2_l263_263368


namespace max_n_for_regular_polygons_l263_263361

theorem max_n_for_regular_polygons (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 3)
  (h3 : (7 * (m - 2) * n) = (8 * (n - 2) * m)) : 
  n ≤ 112 ∧ (∃ m, (14 * n = (n - 16) * m)) :=
by
  sorry

end max_n_for_regular_polygons_l263_263361


namespace find_a_l263_263209

theorem find_a (a : ℝ) 
  (line_through : ∃ (p1 p2 : ℝ × ℝ), p1 = (a-2, -1) ∧ p2 = (-a-2, 1)) 
  (perpendicular : ∀ (l1 l2 : ℝ × ℝ), l1 = (2, 3) → l2 = (-1/a, 1) → false) : 
  a = -2/3 :=
by 
  sorry

end find_a_l263_263209


namespace benny_has_24_books_l263_263235

def books_sandy : ℕ := 10
def books_tim : ℕ := 33
def total_books : ℕ := 67

def books_benny : ℕ := total_books - (books_sandy + books_tim)

theorem benny_has_24_books : books_benny = 24 := by
  unfold books_benny
  unfold total_books
  unfold books_sandy
  unfold books_tim
  sorry

end benny_has_24_books_l263_263235


namespace collinear_points_count_l263_263435

-- Definitions for the problem conditions
def vertices_count := 8
def midpoints_count := 12
def face_centers_count := 6
def cube_center_count := 1
def total_points_count := vertices_count + midpoints_count + face_centers_count + cube_center_count

-- Lean statement to express the proof problem
theorem collinear_points_count :
  (total_points_count = 27) →
  (vertices_count = 8) →
  (midpoints_count = 12) →
  (face_centers_count = 6) →
  (cube_center_count = 1) →
  ∃ n, n = 49 :=
by
  intros
  existsi 49
  sorry

end collinear_points_count_l263_263435


namespace union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l263_263818

open Set

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | x^2 - 12*x + 20 < 0 }
def C (a : ℝ) : Set ℝ := { x | x < a }

theorem union_of_A_and_B :
  A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
sorry

theorem complement_of_A_intersect_B :
  ((univ \ A) ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
sorry

theorem intersection_of_A_and_C (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a > 3 :=
sorry

end union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l263_263818


namespace output_increase_percentage_l263_263874

theorem output_increase_percentage (O : ℝ) (P : ℝ) (h : (O * (1 + P / 100) * 1.60) * 0.5682 = O) : P = 10.09 :=
by 
  sorry

end output_increase_percentage_l263_263874


namespace difference_face_local_value_8_l263_263140

theorem difference_face_local_value_8 :
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3  -- 0-indexed place for thousands
  let local_value := digit * 10^position
  local_value - face_value = 7992 :=
by
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3
  let local_value := digit * 10^position
  show local_value - face_value = 7992
  sorry

end difference_face_local_value_8_l263_263140


namespace escalator_length_l263_263917

theorem escalator_length :
  ∃ L : ℝ, L = 150 ∧ 
    (∀ t : ℝ, t = 10 → ∀ v_p : ℝ, v_p = 3 → ∀ v_e : ℝ, v_e = 12 → L = (v_p + v_e) * t) :=
by sorry

end escalator_length_l263_263917


namespace joe_probability_select_counsel_l263_263969

theorem joe_probability_select_counsel :
  let CANOE := ['C', 'A', 'N', 'O', 'E']
  let SHRUB := ['S', 'H', 'R', 'U', 'B']
  let FLOW := ['F', 'L', 'O', 'W']
  let COUNSEL := ['C', 'O', 'U', 'N', 'S', 'E', 'L']
  -- Probability of selecting C and O from CANOE
  let p_CANOE := 1 / (Nat.choose 5 2)
  -- Probability of selecting U, S, and E from SHRUB
  let comb_SHRUB := Nat.choose 5 3
  let count_USE := 3  -- Determined from the solution
  let p_SHRUB := count_USE / comb_SHRUB
  -- Probability of selecting L, O, W, F from FLOW
  let p_FLOW := 1 / 1
  -- Total probability
  let total_prob := p_CANOE * p_SHRUB * p_FLOW
  total_prob = 3 / 100 := by
    sorry

end joe_probability_select_counsel_l263_263969


namespace triangle_probability_l263_263933

open Classical

theorem triangle_probability : 
  let sticks := [1, 2, 4, 5, 8, 10, 12, 15] in
  let valid_sets := (sticks.toFinset.powerset.filter (λ x, x.card = 3)).filter(λ s, 
    ∃ a b c : ℕ, a = s.min' (by simp) ∧ c = s.max' (by simp) ∧ b = (s\\{a, c}.toFinset).min' (by simp) ∧ a + b > c ) in
  let total_sets := (sticks.toFinset.powerset.filter (λ x, x.card = 3)).card in 
  total_sets = 56 ∧ valid_sets.card = 9 →
  (valid_sets.card : ℚ) / (total_sets : ℚ) = 9 / 56 :=
by
  intros sticks valid_sets total_sets h
  sorry

end triangle_probability_l263_263933


namespace CarltonUniqueOutfits_l263_263787

theorem CarltonUniqueOutfits:
  ∀ (buttonUpShirts sweaterVests : ℕ), 
    buttonUpShirts = 3 →
    sweaterVests = 2 * buttonUpShirts →
    (sweaterVests * buttonUpShirts) = 18 :=
by
  intros buttonUpShirts sweaterVests h1 h2
  rw [h1, h2]
  simp
  sorry

end CarltonUniqueOutfits_l263_263787


namespace missing_digit_B_divisible_by_3_l263_263625

theorem missing_digit_B_divisible_by_3 (B : ℕ) (h1 : (2 * 10 + 8 + B) % 3 = 0) :
  B = 2 :=
sorry

end missing_digit_B_divisible_by_3_l263_263625


namespace _l263_263866

open Real

noncomputable def f (x : ℝ) : ℝ := x - sin x

lemma f_non_decreasing_in_0_to_pi_div_2 {a b : ℝ} (ha : a ∈ Icc (0 : ℝ) (π / 2)) (hb : b ∈ Icc (0 : ℝ) (π / 2)) (h : a < b) :
  f a < f b := by
  sorry

lemma f_increasing_in_pi_to_3pi_div_2 {a b : ℝ} (ha : a ∈ Icc (π : ℝ) (3 * π / 2)) (hb : b ∈ Icc (π : ℝ) (3 * π / 2)) (h : a < b) :
  f a < f b := by
  sorry

lemma main_theorem {a b : ℝ} (h1 : a < b)
  (h2 : (a ∈ Icc (0 : ℝ) (π / 2) ∧ b ∈ Icc (0 : ℝ) (π / 2)) ∨
        (a ∈ Icc (π : ℝ) (3 * π / 2) ∧ b ∈ Icc (π : ℝ) (3 * π / 2))) :
  f a < f b := by
  cases h2 with h2_1 h2_2
  case inl =>
    exact f_non_decreasing_in_0_to_pi_div_2 h2_1.left h2_1.right h1
  case inr =>
    exact f_increasing_in_pi_to_3pi_div_2 h2_2.left h2_2.right h1

-- main_theorem is the core statement derived from the problem

end _l263_263866


namespace ratio_area_rectangle_triangle_l263_263902

-- Define the lengths L and W as positive real numbers
variables {L W : ℝ} (hL : L > 0) (hW : W > 0)

-- Define the area of the rectangle
noncomputable def area_rectangle (L W : ℝ) : ℝ := L * W

-- Define the area of the triangle with base L and height W
noncomputable def area_triangle (L W : ℝ) : ℝ := (1 / 2) * L * W

-- Define the ratio between the area of the rectangle and the area of the triangle
noncomputable def area_ratio (L W : ℝ) : ℝ := area_rectangle L W / area_triangle L W

-- Prove that this ratio is equal to 2
theorem ratio_area_rectangle_triangle : area_ratio L W = 2 := by sorry

end ratio_area_rectangle_triangle_l263_263902


namespace power_sums_fifth_l263_263225

noncomputable def compute_power_sums (α β γ : ℂ) : ℂ :=
  α^5 + β^5 + γ^5

theorem power_sums_fifth (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  compute_power_sums α β γ = 47.2 :=
sorry

end power_sums_fifth_l263_263225


namespace nat_values_of_x_l263_263457

theorem nat_values_of_x :
  (∃ (x : ℕ), 2^(x - 5) = 2 ∧ x = 6) ∧
  (∃ (x : ℕ), 2^x = 512 ∧ x = 9) ∧
  (∃ (x : ℕ), x^5 = 243 ∧ x = 3) ∧
  (∃ (x : ℕ), x^4 = 625 ∧ x = 5) :=
  by {
    sorry
  }

end nat_values_of_x_l263_263457


namespace angle_triple_supplementary_l263_263110

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l263_263110


namespace total_amount_of_money_l263_263419

theorem total_amount_of_money (N50 N500 : ℕ) (h1 : N50 = 97) (h2 : N50 + N500 = 108) : 
  50 * N50 + 500 * N500 = 10350 := by
  sorry

end total_amount_of_money_l263_263419


namespace polynomial_simplification_l263_263888

theorem polynomial_simplification (w : ℝ) : 
  3 * w + 4 - 6 * w - 5 + 7 * w + 8 - 9 * w - 10 + 2 * w ^ 2 = 2 * w ^ 2 - 5 * w - 3 :=
by
  sorry

end polynomial_simplification_l263_263888


namespace survival_rate_is_98_l263_263708

def total_flowers := 150
def unsurviving_flowers := 3
def surviving_flowers := total_flowers - unsurviving_flowers

theorem survival_rate_is_98 : (surviving_flowers : ℝ) / total_flowers * 100 = 98 := by
  sorry

end survival_rate_is_98_l263_263708


namespace carlton_outfit_count_l263_263786

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end carlton_outfit_count_l263_263786


namespace sqrt_range_l263_263832

theorem sqrt_range (x : ℝ) : 3 - 2 * x ≥ 0 ↔ x ≤ 3 / 2 := 
    sorry

end sqrt_range_l263_263832


namespace shortest_altitude_l263_263075

/-!
  Prove that the shortest altitude of a right triangle with sides 9, 12, and 15 is 7.2.
-/

theorem shortest_altitude (a b c : ℕ) (h : a^2 + b^2 = c^2) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  7.2 ≤ a ∧ 7.2 ≤ b ∧ 7.2 ≤ (2 * (a * b) / c) := 
sorry

end shortest_altitude_l263_263075


namespace number_of_4_letter_words_with_vowel_l263_263648

def is_vowel (c : Char) : Bool :=
c = 'A' ∨ c = 'E'

def count_4letter_words_with_vowels : Nat :=
  let total_words := 5^4
  let words_without_vowels := 3^4
  total_words - words_without_vowels

theorem number_of_4_letter_words_with_vowel :
  count_4letter_words_with_vowels = 544 :=
by
  -- proof goes here
  sorry

end number_of_4_letter_words_with_vowel_l263_263648


namespace c_minus_b_seven_l263_263918

theorem c_minus_b_seven {a b c d : ℕ} (ha : a^6 = b^5) (hb : c^4 = d^3) (hc : c - a = 31) : c - b = 7 :=
sorry

end c_minus_b_seven_l263_263918


namespace average_score_of_male_students_l263_263036

theorem average_score_of_male_students
  (female_students : ℕ) (male_students : ℕ) (female_avg_score : ℕ) (class_avg_score : ℕ)
  (h_female_students : female_students = 20)
  (h_male_students : male_students = 30)
  (h_female_avg_score : female_avg_score = 75)
  (h_class_avg_score : class_avg_score = 72) :
  (30 * (((class_avg_score * (female_students + male_students)) - (female_avg_score * female_students)) / male_students) = 70) :=
by
  -- Sorry for the proof
  sorry

end average_score_of_male_students_l263_263036


namespace contradiction_example_l263_263055

theorem contradiction_example (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
by
  sorry

end contradiction_example_l263_263055


namespace sheila_hourly_wage_l263_263288

def sheila_works_hours : ℕ :=
  let monday_wednesday_friday := 8 * 3
  let tuesday_thursday := 6 * 2
  monday_wednesday_friday + tuesday_thursday

def sheila_weekly_earnings : ℕ := 396
def sheila_total_hours_worked := 36
def expected_hourly_earnings := sheila_weekly_earnings / sheila_total_hours_worked

theorem sheila_hourly_wage :
  sheila_works_hours = sheila_total_hours_worked ∧
  sheila_weekly_earnings / sheila_total_hours_worked = 11 :=
by
  sorry

end sheila_hourly_wage_l263_263288


namespace shortest_altitude_of_right_triangle_l263_263076

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Given conditions about the triangle
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of the triangle
def area (a b : ℕ) : ℝ := (1/2) * a * b

-- Define the altitude
noncomputable def altitude (area : ℝ) (c : ℕ) : ℝ := (2 * area) / c

-- Proving the length of the shortest altitude
theorem shortest_altitude_of_right_triangle 
  (h : ℝ) 
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15) 
  (rt : right_triangle a b c) : 
  altitude (area a b) c = 7.2 :=
sorry

end shortest_altitude_of_right_triangle_l263_263076


namespace original_price_of_dinosaur_model_l263_263244

-- Define the conditions
theorem original_price_of_dinosaur_model
  (P : ℝ) -- original price of each model
  (kindergarten_models : ℝ := 2)
  (elementary_models : ℝ := 2 * kindergarten_models)
  (total_models : ℝ := kindergarten_models + elementary_models)
  (reduction_percentage : ℝ := 0.05)
  (discounted_price : ℝ := P * (1 - reduction_percentage))
  (total_paid : ℝ := total_models * discounted_price)
  (total_paid_condition : total_paid = 570) :
  P = 100 :=
by
  sorry

end original_price_of_dinosaur_model_l263_263244


namespace log_value_comparison_l263_263293

theorem log_value_comparison :
  let e := Real.exp 1 in
  let initial_value := Log.log 1.2 * 1 / 6 in
  let transformed_value := 2.988 in
  transformed_value > e :=
by
  sorry

end log_value_comparison_l263_263293


namespace tennis_handshakes_l263_263310

theorem tennis_handshakes :
  let num_teams := 4
  let women_per_team := 2
  let total_women := num_teams * women_per_team
  let handshakes_per_woman := total_women - 2
  let total_handshakes_before_division := total_women * handshakes_per_woman
  let actual_handshakes := total_handshakes_before_division / 2
  actual_handshakes = 24 :=
by sorry

end tennis_handshakes_l263_263310


namespace fill_time_l263_263752

def inflow_rate : ℕ := 24 -- gallons per second
def outflow_rate : ℕ := 4 -- gallons per second
def basin_volume : ℕ := 260 -- gallons

theorem fill_time (inflow_rate outflow_rate basin_volume : ℕ) (h₁ : inflow_rate = 24) (h₂ : outflow_rate = 4) 
  (h₃ : basin_volume = 260) : basin_volume / (inflow_rate - outflow_rate) = 13 :=
by
  sorry

end fill_time_l263_263752


namespace inequality_on_abc_l263_263406

theorem inequality_on_abc (α β γ : ℝ) (h : α^2 + β^2 + γ^2 = 1) :
  -1/2 ≤ α * β + β * γ + γ * α ∧ α * β + β * γ + γ * α ≤ 1 :=
by {
  sorry -- Proof to be added
}

end inequality_on_abc_l263_263406


namespace ratio_paid_back_to_initial_debt_l263_263010

def initial_debt : ℕ := 40
def still_owed : ℕ := 30
def paid_back (initial_debt still_owed : ℕ) : ℕ := initial_debt - still_owed

theorem ratio_paid_back_to_initial_debt
  (initial_debt still_owed : ℕ) :
  (paid_back initial_debt still_owed : ℚ) / initial_debt = 1 / 4 :=
by 
  sorry

end ratio_paid_back_to_initial_debt_l263_263010


namespace equal_sum_sequence_even_odd_l263_263407

-- Define the sequence a_n
variable {a : ℕ → ℤ}

-- Define the condition of the equal-sum sequence
def equal_sum_sequence (a : ℕ → ℤ) : Prop := ∀ n, a n + a (n + 1) = a (n + 1) + a (n + 2)

-- Statement to prove the odd terms are equal and the even terms are equal
theorem equal_sum_sequence_even_odd (a : ℕ → ℤ) (h : equal_sum_sequence a) : (∀ n, a (2 * n) = a 0) ∧ (∀ n, a (2 * n + 1) = a 1) :=
by
  sorry

end equal_sum_sequence_even_odd_l263_263407


namespace find_first_term_l263_263568

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l263_263568


namespace trajectory_equation_minimum_AB_l263_263470

/-- Let a moving circle \( C \) passes through the point \( F(0, 1) \).
    The center of the circle \( C \), denoted as \( (x, y) \), is above the \( x \)-axis and the
    distance from \( (x, y) \) to \( F \) is greater than its distance to the \( x \)-axis by 1.
    We aim to prove that the trajectory of the center is \( x^2 = 4y \). -/
theorem trajectory_equation {x y : ℝ} (h : y > 0) (hCF : Real.sqrt (x^2 + (y - 1)^2) - y = 1) : 
  x^2 = 4 * y :=
sorry

/-- Suppose \( A \) and \( B \) are two distinct points on the curve \( x^2 = 4y \). 
    The tangents at \( A \) and \( B \) intersect at \( P \), and \( AP \perp BP \). 
    Then the minimum value of \( |AB| \) is 4. -/
theorem minimum_AB {x₁ x₂ : ℝ} 
  (h₁ : y₁ = (x₁^2) / 4) (h₂ : y₂ = (x₂^2) / 4)
  (h_perp : x₁ * x₂ = -4) : 
  ∃ (d : ℝ), d ≥ 0 ∧ d = 4 :=
sorry

end trajectory_equation_minimum_AB_l263_263470


namespace geometric_sequence_sum_9000_l263_263877

noncomputable def sum_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_9000 (a r : ℝ) (h : r ≠ 1) 
  (h1 : sum_geometric_sequence a r 3000 = 1000)
  (h2 : sum_geometric_sequence a r 6000 = 1900) : 
  sum_geometric_sequence a r 9000 = 2710 :=
sorry

end geometric_sequence_sum_9000_l263_263877


namespace shara_monthly_payment_l263_263236

theorem shara_monthly_payment : 
  ∀ (T M : ℕ), 
  (T / 2 = 6 * M) → 
  (T / 2 - 4 * M = 20) → 
  M = 10 :=
by
  intros T M h1 h2
  sorry

end shara_monthly_payment_l263_263236


namespace max_area_with_22_matches_l263_263731

-- Definitions based on the conditions
def perimeter := 22

def is_valid_length_width (l w : ℕ) : Prop := l + w = 11

def area (l w : ℕ) : ℕ := l * w

-- Statement of the proof problem
theorem max_area_with_22_matches : 
  ∃ (l w : ℕ), is_valid_length_width l w ∧ (∀ l' w', is_valid_length_width l' w' → area l w ≥ area l' w') ∧ area l w = 30 :=
  sorry

end max_area_with_22_matches_l263_263731


namespace average_marks_math_chem_l263_263143

variables (M P C : ℕ)

theorem average_marks_math_chem :
  (M + P = 20) → (C = P + 20) → (M + C) / 2 = 20 := 
by
  sorry

end average_marks_math_chem_l263_263143


namespace probability_rain_at_least_one_day_l263_263325

open ProbabilityTheory

variables {Ω : Type*} [MeasurableSpace Ω]
variable {P : Measure Ω}
variables {A B : Set Ω}

noncomputable def prob_saturday_rain := 0.6
noncomputable def prob_sunday_rain_given_saturday := 0.8
noncomputable def prob_sunday_rain_given_no_saturday := 0.4

theorem probability_rain_at_least_one_day : 
  P[A] = prob_saturday_rain →
  cond_prob P B A = prob_sunday_rain_given_saturday →
  cond_prob P B Aᶜ = prob_sunday_rain_given_no_saturday →
  (1 - ((1 - P A) * (1 - cond_prob P B Aᶜ))) = 0.76 :=
sorry

end probability_rain_at_least_one_day_l263_263325


namespace weight_of_replaced_student_l263_263555

variable (W : ℝ) -- total weight of the original 10 students
variable (new_student_weight : ℝ := 60) -- weight of the new student
variable (weight_decrease_per_student : ℝ := 6) -- average weight decrease per student

theorem weight_of_replaced_student (replaced_student_weight : ℝ) :
  (W - replaced_student_weight + new_student_weight = W - 10 * weight_decrease_per_student) →
  replaced_student_weight = 120 := by
  sorry

end weight_of_replaced_student_l263_263555


namespace fraction_of_red_knights_magical_l263_263073

def total_knights : ℕ := 28
def red_fraction : ℚ := 3 / 7
def magical_fraction : ℚ := 1 / 4
def red_magical_to_blue_magical_ratio : ℚ := 3

theorem fraction_of_red_knights_magical :
  let red_knights := red_fraction * total_knights
  let blue_knights := total_knights - red_knights
  let total_magical := magical_fraction * total_knights
  let red_magical_fraction := 21 / 52
  let blue_magical_fraction := red_magical_fraction / red_magical_to_blue_magical_ratio
  red_knights * red_magical_fraction + blue_knights * blue_magical_fraction = total_magical :=
by
  sorry

end fraction_of_red_knights_magical_l263_263073


namespace golf_money_l263_263657

-- Definitions based on conditions
def cost_per_round : ℤ := 80
def number_of_rounds : ℤ := 5

-- The theorem/problem statement
theorem golf_money : cost_per_round * number_of_rounds = 400 := 
by {
  -- Proof steps would go here, but to skip the proof, we use sorry
  sorry
}

end golf_money_l263_263657


namespace find_pages_revised_twice_l263_263566

def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_time : ℕ) (cost_revised_once : ℕ) (cost_revised_twice : ℕ) (total_cost : ℕ) :=
  ∃ (x : ℕ), 
    (total_pages - pages_revised_once - x) * cost_first_time
    + pages_revised_once * (cost_first_time + cost_revised_once)
    + x * (cost_first_time + cost_revised_once + cost_revised_once) = total_cost 

theorem find_pages_revised_twice :
  pages_revised_twice 100 35 6 4 4 860 ↔ ∃ x, x = 15 :=
by
  sorry

end find_pages_revised_twice_l263_263566


namespace red_pencils_count_l263_263356

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end red_pencils_count_l263_263356


namespace tiling_not_possible_l263_263264

-- Definitions for the puzzle pieces
inductive Piece
| L | T | I | Z | O

-- Function to check if tiling a rectangle is possible
noncomputable def can_tile_rectangle (pieces : List Piece) : Prop :=
  ∀ (width height : ℕ), width * height % 4 = 0 → ∃ (tiling : List (Piece × ℕ × ℕ)), sorry

theorem tiling_not_possible : ¬ can_tile_rectangle [Piece.L, Piece.T, Piece.I, Piece.Z, Piece.O] :=
sorry

end tiling_not_possible_l263_263264


namespace ratio_snakes_to_lions_is_S_per_100_l263_263701

variables {S G : ℕ}

/-- Giraffe count in Safari National Park is 10 fewer than snakes -/
def safari_giraffes_minus_ten (S G : ℕ) : Prop := G = S - 10

/-- The number of lions in Safari National Park -/
def safari_lions : ℕ := 100

/-- The ratio of number of snakes to number of lions in Safari National Park -/
def ratio_snakes_to_lions (S : ℕ) : ℕ := S / safari_lions

/-- Prove the ratio of the number of snakes to the number of lions in Safari National Park -/
theorem ratio_snakes_to_lions_is_S_per_100 :
  ∀ S G, safari_giraffes_minus_ten S G → (ratio_snakes_to_lions S = S / 100) :=
by
  intros S G h
  sorry

end ratio_snakes_to_lions_is_S_per_100_l263_263701


namespace angle_triple_supplement_l263_263113

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263113


namespace factor_expression_l263_263313

theorem factor_expression (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) :=
by
  sorry

end factor_expression_l263_263313


namespace simplify_expr1_simplify_expr2_l263_263373

-- Expression simplification proof statement 1
theorem simplify_expr1 (m n : ℤ) : 
  (5 * m + 3 * n - 7 * m - n) = (-2 * m + 2 * n) :=
sorry

-- Expression simplification proof statement 2
theorem simplify_expr2 (x : ℤ) : 
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2)) = (2 * x^2 - 5 * x + 6) :=
sorry

end simplify_expr1_simplify_expr2_l263_263373


namespace greatest_value_of_x_for_7x_factorial_100_l263_263890

open Nat

theorem greatest_value_of_x_for_7x_factorial_100 : 
  ∃ x : ℕ, (∀ y : ℕ, 7^y ∣ factorial 100 → y ≤ x) ∧ x = 16 :=
by
  sorry

end greatest_value_of_x_for_7x_factorial_100_l263_263890


namespace find_fraction_l263_263656

theorem find_fraction
  (w x y F : ℝ)
  (h1 : 5 / w + F = 5 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  F = 10 := 
sorry

end find_fraction_l263_263656


namespace seashells_given_to_Joan_l263_263535

def S_original : ℕ := 35
def S_now : ℕ := 17

theorem seashells_given_to_Joan :
  (S_original - S_now) = 18 := by
  sorry

end seashells_given_to_Joan_l263_263535


namespace at_least_one_shooter_hits_target_l263_263090

-- Definition stating the probability of the first shooter hitting the target
def prob_A1 : ℝ := 0.7

-- Definition stating the probability of the second shooter hitting the target
def prob_A2 : ℝ := 0.8

-- The event that at least one shooter hits the target
def prob_at_least_one_hit : ℝ := prob_A1 + prob_A2 - (prob_A1 * prob_A2)

-- Prove that the probability that at least one shooter hits the target is 0.94
theorem at_least_one_shooter_hits_target : prob_at_least_one_hit = 0.94 :=
by
  sorry

end at_least_one_shooter_hits_target_l263_263090


namespace solve_problem_l263_263655

theorem solve_problem (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 5) : (b - d) ^ 2 = 16 :=
by
  sorry

end solve_problem_l263_263655


namespace trig_quadrant_l263_263017

theorem trig_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * π + α / 2 :=
sorry

end trig_quadrant_l263_263017


namespace solve_for_s_l263_263202

theorem solve_for_s (r s : ℝ) (h1 : 1 < r) (h2 : r < s) (h3 : 1 / r + 1 / s = 3 / 4) (h4 : r * s = 8) : s = 4 :=
sorry

end solve_for_s_l263_263202


namespace find_t_l263_263523

theorem find_t (t : ℝ) :
  (2 * t - 7) * (3 * t - 4) = (3 * t - 9) * (2 * t - 6) →
  t = 26 / 7 := 
by 
  intro h
  sorry

end find_t_l263_263523


namespace red_pencils_count_l263_263357

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end red_pencils_count_l263_263357


namespace googoo_total_buttons_l263_263243

noncomputable def button_count_shirt_1 : ℕ := 3
noncomputable def button_count_shirt_2 : ℕ := 5
noncomputable def quantity_shirt_1 : ℕ := 200
noncomputable def quantity_shirt_2 : ℕ := 200

theorem googoo_total_buttons :
  (quantity_shirt_1 * button_count_shirt_1) + (quantity_shirt_2 * button_count_shirt_2) = 1600 := by
  sorry

end googoo_total_buttons_l263_263243


namespace inequality_proof_l263_263532

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : x1 > 0) (hx2 : x2 > 0) (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hx1y1_pos : x1 * y1 - z1^2 > 0) (hx2y2_pos : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 
    1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
by
  sorry

end inequality_proof_l263_263532


namespace paint_cans_needed_l263_263613

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l263_263613


namespace circle_equation_l263_263604

theorem circle_equation
  (a b r : ℝ)
  (ha : (4 - a)^2 + (1 - b)^2 = r^2)
  (hb : (2 - a)^2 + (1 - b)^2 = r^2)
  (ht : (b - 1) / (a - 2) = -1) :
  (a = 3) ∧ (b = 0) ∧ (r = 2) :=
by {
  sorry
}

-- Given the above values for a, b, r
def circle_equation_verified : Prop :=
  (∀ (x y : ℝ), ((x - 3)^2 + y^2) = 4)

example : circle_equation_verified :=
by {
  sorry
}

end circle_equation_l263_263604


namespace split_into_similar_piles_l263_263517

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l263_263517


namespace train_cross_time_l263_263429

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 255.03
noncomputable def train_speed_ms : ℝ := 12.5
noncomputable def distance_to_travel : ℝ := train_length + bridge_length
noncomputable def expected_time : ℝ := 30.0024

theorem train_cross_time :
  (distance_to_travel / train_speed_ms) = expected_time :=
by sorry

end train_cross_time_l263_263429


namespace required_fencing_l263_263282

-- Definitions from conditions
def length_uncovered : ℝ := 30
def area : ℝ := 720

-- Prove that the amount of fencing required is 78 feet
theorem required_fencing : 
  ∃ (W : ℝ), (area = length_uncovered * W) ∧ (2 * W + length_uncovered = 78) := 
sorry

end required_fencing_l263_263282


namespace geom_series_first_term_l263_263571

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l263_263571


namespace value_of_Y_l263_263319

theorem value_of_Y :
  let part1 := 15 * 180 / 100  -- 15% of 180
  let part2 := part1 - part1 / 3  -- one-third less than 15% of 180
  let part3 := 24.5 * (2 * 270 / 3) / 100  -- 24.5% of (2/3 * 270)
  let part4 := (5.4 * 2) / (0.25 * 0.25)  -- (5.4 * 2) / (0.25)^2
  let Y := part2 + part3 - part4
  Y = -110.7 := by
    -- proof skipped
    sorry

end value_of_Y_l263_263319


namespace return_trip_time_l263_263177

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l263_263177


namespace inheritance_amount_l263_263040

theorem inheritance_amount (x : ℝ) (hx1 : 0.25 * x + 0.1 * x = 15000) : x = 42857 := 
by
  -- Proof omitted
  sorry

end inheritance_amount_l263_263040


namespace paint_cans_needed_l263_263612

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l263_263612


namespace profit_relationship_max_profit_l263_263603

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
else if h : 2 < x ∧ x ≤ 5 then 50 * x / (1 + x)
else 0

noncomputable def f (x : ℝ) : ℝ :=
15 * W x - 10 * x - 20 * x

theorem profit_relationship:
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = 75 * x^2 - 30 * x + 225) ∧
  (∀ x, 2 < x ∧ x ≤ 5 → f x = (750 * x)/(1 + x) - 30 * x) :=
by
  -- to be proven
  sorry

theorem max_profit:
  ∃ x, 0 ≤ x ∧ x ≤ 5 ∧ f x = 480 ∧ 10 * x = 40 :=
by
  -- to be proven
  sorry

end profit_relationship_max_profit_l263_263603


namespace original_savings_calculation_l263_263977

theorem original_savings_calculation (S : ℝ) (F : ℝ) (T : ℝ) 
  (h1 : 0.8 * F = (3 / 4) * S)
  (h2 : 1.1 * T = 150)
  (h3 : (1 / 4) * S = T) :
  S = 545.44 :=
by
  sorry

end original_savings_calculation_l263_263977


namespace inverse_proportion_passes_first_and_third_quadrants_l263_263006

theorem inverse_proportion_passes_first_and_third_quadrants (m : ℝ) :
  ((∀ x : ℝ, x ≠ 0 → (x > 0 → (m - 3) / x > 0) ∧ (x < 0 → (m - 3) / x < 0)) → m = 5) := 
by 
  sorry

end inverse_proportion_passes_first_and_third_quadrants_l263_263006


namespace notepad_days_last_l263_263915

def fold_paper (n : Nat) : Nat := 2 ^ n

def lettersize_paper_pieces : Nat := 5
def folds : Nat := 3
def notes_per_day : Nat := 10

def smaller_note_papers_per_piece : Nat := fold_paper folds
def total_smaller_note_papers : Nat := lettersize_paper_pieces * smaller_note_papers_per_piece
def total_days : Nat := total_smaller_note_papers / notes_per_day

theorem notepad_days_last : total_days = 4 := by
  sorry

end notepad_days_last_l263_263915


namespace smallest_base10_integer_l263_263266

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l263_263266


namespace bus_stop_time_per_hour_l263_263403

theorem bus_stop_time_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h1 : speed_without_stoppages = 64)
  (h2 : speed_with_stoppages = 48) : 
  ∃ t : ℝ, t = 15 := 
by
  sorry

end bus_stop_time_per_hour_l263_263403


namespace angle_triple_supplement_l263_263101

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l263_263101


namespace rectangle_area_k_l263_263725

theorem rectangle_area_k (d : ℝ) (x : ℝ) (h_ratio : 5 * x > 0 ∧ 2 * x > 0) (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (∃ (h : k = 10 / 29), (5 * x) * (2 * x) = k * d^2) := by
  use 10 / 29
  sorry

end rectangle_area_k_l263_263725


namespace directrix_of_parabola_l263_263990

theorem directrix_of_parabola (y x : ℝ) (h : y = 4 * x^2) : y = - (1 / 16) :=
sorry

end directrix_of_parabola_l263_263990


namespace find_general_term_Tn_greater_than_one_l263_263472

open Real

variable {a : ℕ → ℝ}

-- Conditions for the arithmetic sequence
axiom a3 : a 3 = -4
axiom a1_a10 : a 1 + a 10 = 2

-- General formula for the sequence
def general_term (n : ℕ) : ℝ := 2 * n - 10

-- Proof that the derived general term matches the conditions.
theorem find_general_term (n : ℕ) :
  a 3 = -4 ∧ (a 1 + a 10 = 2) → a n = 2 * n - 10 := by
  intro h
  cases h with h1 h2
  have eq_a1 : a 1 = -8 := by sorry -- Derived from solving the equations
  have eq_d : (a (n + 1) - a n) = 2 := by sorry -- Derived from solving the equations
  induction n with n ih
  · sorry -- Base case for n = 0
  · sorry -- Inductive step

-- Conditions for the sequence {b_n}
def b (n : ℕ) : ℝ := 3 ^ (a n)
def T (n : ℕ) : ℝ := (List.range n).map b |>.prod

-- Proof that T_n > 1 for n > 9.
theorem Tn_greater_than_one {n : ℕ}:
  (∀ n, a n = log 3 (b n)) ∧ (∀ n, T n = (List.range n).map b |>.prod) → T n > 1 → n > 9 := by
  intro h _ 
  have t_eq : T n = 3 ^ (n ^ 2 - 9 * n) := by sorry
  have inequality : 3 ^ (n ^ 2 - 9 * n) > 1 → n ^ 2 - 9 * n > 0 := by sorry
  have factorization : n ^ 2 - 9 * n = n * (n - 9) := by sorry
  have final_step : n * (n - 9) > 0 → n > 9 := by sorry
  exact final_step

end find_general_term_Tn_greater_than_one_l263_263472


namespace eric_return_home_time_l263_263180

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l263_263180


namespace simplify_exponent_l263_263544

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l263_263544


namespace line_circle_no_intersection_l263_263650

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
sorry

end line_circle_no_intersection_l263_263650


namespace problem1_l263_263600

theorem problem1 (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^(n+1) - 9) := 
  sorry

end problem1_l263_263600


namespace steak_chicken_ratio_l263_263352

variable (S C : ℕ)

theorem steak_chicken_ratio (h1 : S + C = 80) (h2 : 25 * S + 18 * C = 1860) : S = 3 * C :=
by
  sorry

end steak_chicken_ratio_l263_263352


namespace company_x_total_employees_l263_263032

-- Definitions for conditions
def initial_percentage : ℝ := 0.60
def Q2_hiring_males : ℕ := 30
def Q2_new_percentage : ℝ := 0.57
def Q3_hiring_females : ℕ := 50
def Q3_new_percentage : ℝ := 0.62
def Q4_hiring_males : ℕ := 40
def Q4_hiring_females : ℕ := 10
def Q4_new_percentage : ℝ := 0.58

-- Statement of the proof problem
theorem company_x_total_employees :
  ∃ (E : ℕ) (F : ℕ), 
    (F = initial_percentage * E ∧
     F = Q2_new_percentage * (E + Q2_hiring_males) ∧
     F + Q3_hiring_females = Q3_new_percentage * (E + Q2_hiring_males + Q3_hiring_females) ∧
     F + Q3_hiring_females + Q4_hiring_females = Q4_new_percentage * (E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females)) →
    E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females = 700 :=
sorry

end company_x_total_employees_l263_263032


namespace greatest_possible_value_of_q_minus_r_l263_263722

theorem greatest_possible_value_of_q_minus_r :
  ∃ q r : ℕ, 0 < q ∧ 0 < r ∧ 852 = 21 * q + r ∧ q - r = 28 :=
by
  -- Proof goes here
  sorry

end greatest_possible_value_of_q_minus_r_l263_263722


namespace probability_of_yellow_l263_263148

-- Definitions of the given conditions
def red_jelly_beans := 4
def green_jelly_beans := 8
def yellow_jelly_beans := 9
def blue_jelly_beans := 5
def total_jelly_beans := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans

-- Theorem statement
theorem probability_of_yellow :
  (yellow_jelly_beans : ℚ) / total_jelly_beans = 9 / 26 :=
by
  sorry

end probability_of_yellow_l263_263148


namespace product_4_7_25_l263_263734

theorem product_4_7_25 : 4 * 7 * 25 = 700 :=
by sorry

end product_4_7_25_l263_263734


namespace eric_return_home_time_l263_263182

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l263_263182


namespace distance_between_Q_and_R_l263_263765

noncomputable def distance_QR : Real :=
  let YZ := 9
  let XZ := 12
  let XY := 15
  
  -- assume QY = QX and tangent to YZ at Y, and RX = RY and tangent to XZ at X
  let QY := 12.5
  let QX := 12.5
  let RY := 12.5
  let RX := 12.5

  -- calculate and return the distance QR based on these assumptions
  (QX^2 + RY^2 - 2 * QX * RX * Real.cos 90)^(1/2)

theorem distance_between_Q_and_R (YZ XZ XY : ℝ) (QY QX RY RX : ℝ) (h1 : YZ = 9) (h2 : XZ = 12) (h3 : XY = 15)
  (h4 : QY = 12.5) (h5 : QX = 12.5) (h6 : RY = 12.5) (h7 : RX = 12.5) :
  distance_QR = 15 :=
by
  sorry

end distance_between_Q_and_R_l263_263765


namespace not_multiple_of_3_l263_263223

noncomputable def exists_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n*(n + 3) = m^2

theorem not_multiple_of_3 
  (n : ℕ) (h1 : 0 < n) (h2 : exists_perfect_square n) : ¬ ∃ k : ℕ, n = 3 * k := 
sorry

end not_multiple_of_3_l263_263223


namespace james_total_payment_l263_263967

noncomputable def total_amount_paid : ℕ :=
  let dirt_bike_count := 3
  let off_road_vehicle_count := 4
  let atv_count := 2
  let moped_count := 5
  let scooter_count := 3
  let dirt_bike_cost := dirt_bike_count * 150
  let off_road_vehicle_cost := off_road_vehicle_count * 300
  let atv_cost := atv_count * 450
  let moped_cost := moped_count * 200
  let scooter_cost := scooter_count * 100
  let registration_dirt_bike := dirt_bike_count * 25
  let registration_off_road_vehicle := off_road_vehicle_count * 25
  let registration_atv := atv_count * 30
  let registration_moped := moped_count * 15
  let registration_scooter := scooter_count * 20
  let maintenance_dirt_bike := dirt_bike_count * 50
  let maintenance_off_road_vehicle := off_road_vehicle_count * 75
  let maintenance_atv := atv_count * 100
  let maintenance_moped := moped_count * 60
  let total_cost_of_vehicles := dirt_bike_cost + off_road_vehicle_cost + atv_cost + moped_cost + scooter_cost
  let total_registration_costs := registration_dirt_bike + registration_off_road_vehicle + registration_atv + registration_moped + registration_scooter
  let total_maintenance_costs := maintenance_dirt_bike + maintenance_off_road_vehicle + maintenance_atv + maintenance_moped
  total_cost_of_vehicles + total_registration_costs + total_maintenance_costs

theorem james_total_payment : total_amount_paid = 5170 := by
  -- The proof would be written here
  sorry

end james_total_payment_l263_263967


namespace g_9_pow_4_l263_263711

theorem g_9_pow_4 (f g : ℝ → ℝ) (h1 : ∀ x ≥ 1, f (g x) = x^2) (h2 : ∀ x ≥ 1, g (f x) = x^4) (h3 : g 81 = 81) : (g 9)^4 = 81 :=
sorry

end g_9_pow_4_l263_263711


namespace walter_age_in_2001_l263_263834

/-- In 1996, Walter was one-third as old as his grandmother, 
and the sum of the years in which they were born is 3864.
Prove that Walter will be 37 years old at the end of 2001. -/
theorem walter_age_in_2001 (y : ℕ) (H1 : ∃ g, g = 3 * y)
  (H2 : 1996 - y + (1996 - (3 * y)) = 3864) : y + 5 = 37 :=
by sorry

end walter_age_in_2001_l263_263834


namespace value_of_x_for_real_y_l263_263826

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) : x ≤ -3 ∨ x ≥ 12 :=
sorry

end value_of_x_for_real_y_l263_263826


namespace min_squared_sum_l263_263494

theorem min_squared_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  x^2 + y^2 + z^2 ≥ 9 := 
sorry

end min_squared_sum_l263_263494


namespace inequality_for_positive_nums_l263_263056

theorem inequality_for_positive_nums 
    (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    a^2 / b + c^2 / d ≥ (a + c)^2 / (b + d) :=
by
  sorry

end inequality_for_positive_nums_l263_263056


namespace max_daily_profit_l263_263430

noncomputable def daily_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then (5*x/3 - x^3/180)
  else if 12 < x ∧ x ≤ 20 then (1/2 * x)
  else 0

theorem max_daily_profit : ∃ (x : ℝ), 0 < x ∧ x ≤ 20 ∧ 
  (daily_profit x = 100 / 9) :=
begin
  use 10,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { rw daily_profit,
    simp,
    sorry, -- the proof goes here
  }
end

end max_daily_profit_l263_263430


namespace reciprocal_opposite_neg_two_thirds_l263_263998

noncomputable def opposite (a : ℚ) : ℚ := -a
noncomputable def reciprocal (a : ℚ) : ℚ := 1 / a

theorem reciprocal_opposite_neg_two_thirds : reciprocal (opposite (-2 / 3)) = 3 / 2 :=
by sorry

end reciprocal_opposite_neg_two_thirds_l263_263998


namespace factor_x4_minus_64_l263_263184

theorem factor_x4_minus_64 :
  ∀ (x : ℝ), (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by
  intro x
  sorry

end factor_x4_minus_64_l263_263184


namespace pairs_count_l263_263683

theorem pairs_count (A B : Set ℕ) (h1 : A ∪ B = {1, 2, 3, 4, 5}) (h2 : 3 ∈ A ∩ B) : 
  Nat.card {p : Set ℕ × Set ℕ | p.1 ∪ p.2 = {1, 2, 3, 4, 5} ∧ 3 ∈ p.1 ∩ p.2} = 81 := by
  sorry

end pairs_count_l263_263683


namespace evaluate_expression_l263_263795

theorem evaluate_expression :
  (2 / 10 + 3 / 100 + 5 / 1000 + 7 / 10000)^2 = 0.05555649 :=
by
  sorry

end evaluate_expression_l263_263795


namespace min_value_of_function_l263_263872

theorem min_value_of_function (p : ℝ) : 
  ∃ x : ℝ, (x^2 - 2 * p * x + 2 * p^2 + 2 * p - 1) = -2 := sorry

end min_value_of_function_l263_263872


namespace parallel_perpendicular_trans_l263_263474

variables {Plane Line : Type}

-- Definitions in terms of lines and planes
variables (α β γ : Plane) (a b : Line)

-- Definitions of parallel and perpendicular
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- The mathematical statement to prove
theorem parallel_perpendicular_trans :
  (parallel a b) → (perpendicular b α) → (perpendicular a α) :=
by sorry

end parallel_perpendicular_trans_l263_263474


namespace time_for_C_to_complete_work_l263_263602

variable (A B C : ℕ) (R : ℚ)

def work_completion_in_days (days : ℕ) (portion : ℚ) :=
  portion = 1 / days

theorem time_for_C_to_complete_work :
  work_completion_in_days A 8 →
  work_completion_in_days B 12 →
  work_completion_in_days (A + B + C) 4 →
  C = 24 :=
by
  sorry

end time_for_C_to_complete_work_l263_263602


namespace find_x_set_eq_l263_263689

noncomputable def f : ℝ → ℝ :=
sorry -- The actual definition of f according to its properties is omitted

lemma odd_function (x : ℝ) : f (-x) = -f x :=
sorry

lemma periodic_function (x : ℝ) : f (x + 2) = -f x :=
sorry

lemma f_definition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 1 / 2 * x :=
sorry

theorem find_x_set_eq (x : ℝ) : (f x = -1 / 2) ↔ (∃ k : ℤ, x = 4 * k - 1) :=
sorry

end find_x_set_eq_l263_263689


namespace find_S12_l263_263031

variable {a : Nat → Int} -- representing the arithmetic sequence {a_n}
variable {S : Nat → Int} -- representing the sums of the first n terms, S_n

-- Condition: a_1 = -9
axiom a1_def : a 1 = -9

-- Condition: (S_n / n) forms an arithmetic sequence
axiom arithmetic_s : ∃ d : Int, ∀ n : Nat, S n / n = -9 + (n - 1) * d

-- Condition: 2 = S9 / 9 - S7 / 7
axiom condition : S 9 / 9 - S 7 / 7 = 2

-- We want to prove: S_12 = 36
theorem find_S12 : S 12 = 36 := 
sorry

end find_S12_l263_263031


namespace enter_exit_ways_correct_l263_263146

-- Defining the problem conditions
def num_entrances := 4

-- Defining the problem question and answer
def enter_exit_ways (n : Nat) : Nat := n * (n - 1)

-- Statement: Prove the number of different ways to enter and exit is 12
theorem enter_exit_ways_correct : enter_exit_ways num_entrances = 12 := by
  -- Proof
  sorry

end enter_exit_ways_correct_l263_263146


namespace log_order_preservation_l263_263621

theorem log_order_preservation {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (Real.log a > Real.log b) → (a > b) :=
by
  sorry

end log_order_preservation_l263_263621


namespace geom_series_first_term_l263_263573

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l263_263573


namespace convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l263_263299

noncomputable def cost_per_pure_milk_box (x : ℕ) : ℝ := 2000 / x
noncomputable def cost_per_yogurt_box (x : ℕ) : ℝ := 4800 / (1.5 * x)

theorem convenience_store_pure_milk_quantity
  (x : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30) :
  x = 40 :=
by
  sorry

noncomputable def pure_milk_price := 80
noncomputable def yogurt_price (cost_per_yogurt_box : ℝ) : ℝ := cost_per_yogurt_box * 1.25

theorem convenience_store_yogurt_discount
  (x y : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30)
  (total_profit : ℕ)
  (profit_condition :
    pure_milk_price * x +
    yogurt_price (cost_per_yogurt_box x) * (1.5 * x - y) +
    yogurt_price (cost_per_yogurt_box x) * 0.9 * y - 2000 - 4800 = total_profit)
  (pure_milk_quantity : x = 40)
  (profit_value : total_profit = 2150) :
  y = 25 :=
by
  sorry

end convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l263_263299


namespace A_investment_l263_263156

variable (x : ℕ)
variable (A_share : ℕ := 3780)
variable (Total_profit : ℕ := 12600)
variable (B_invest : ℕ := 4200)
variable (C_invest : ℕ := 10500)

theorem A_investment :
  (A_share : ℝ) / (Total_profit : ℝ) = (x : ℝ) / (x + B_invest + C_invest) →
  x = 6300 :=
by
  sorry

end A_investment_l263_263156


namespace area_of_walkways_l263_263715

-- Define the dimensions of the individual flower bed
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3

-- Define the number of rows and columns of flower beds
def rows_of_beds : ℕ := 4
def cols_of_beds : ℕ := 3

-- Define the width of the walkways
def walkway_width : ℕ := 2

-- Calculate the total width and height of the garden including walkways
def total_width : ℕ := (cols_of_beds * flower_bed_width) + (cols_of_beds + 1) * walkway_width
def total_height : ℕ := (rows_of_beds * flower_bed_height) + (rows_of_beds + 1) * walkway_width

-- Calculate the area of the garden including walkways
def total_area : ℕ := total_width * total_height

-- Calculate the total area of all the flower beds
def total_beds_area : ℕ := (rows_of_beds * cols_of_beds) * (flower_bed_width * flower_bed_height)

-- Prove the area of walkways
theorem area_of_walkways : total_area - total_beds_area = 416 := by
  sorry

end area_of_walkways_l263_263715


namespace find_first_term_l263_263575

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l263_263575


namespace solve_functional_equation_l263_263455

theorem solve_functional_equation
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, (∀ x, f x = d * x^2 + c) ∧ (∀ x, g x = d * x^2 + c) :=
sorry

end solve_functional_equation_l263_263455


namespace inequality_1_system_of_inequalities_l263_263065

-- Statement for inequality (1)
theorem inequality_1 (x : ℝ) : 2 - x ≥ (x - 1) / 3 - 1 → x ≤ 2.5 := 
sorry

-- Statement for system of inequalities (2)
theorem system_of_inequalities (x : ℝ) : 
  (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) → false := 
sorry

end inequality_1_system_of_inequalities_l263_263065


namespace triple_supplementary_angle_l263_263092

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l263_263092


namespace neg_one_quadratic_residue_iff_l263_263853

theorem neg_one_quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) : 
  (∃ x : ℤ, x^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end neg_one_quadratic_residue_iff_l263_263853


namespace jessies_current_weight_l263_263772

theorem jessies_current_weight (initial_weight lost_weight : ℝ) (h1 : initial_weight = 69) (h2 : lost_weight = 35) :
  initial_weight - lost_weight = 34 :=
by sorry

end jessies_current_weight_l263_263772


namespace sum_of_a_equals_five_l263_263005

theorem sum_of_a_equals_five
  (f : ℕ → ℕ → ℕ)  -- Represents the function f defined by Table 1
  (a : ℕ → ℕ)  -- Represents the occurrences a₀, a₁, ..., a₄
  (h1 : a 0 + a 1 + a 2 + a 3 + a 4 = 5)  -- Condition 1
  (h2 : 0 * a 0 + 1 * a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 = 5)  -- Condition 2
  : a 0 + a 1 + a 2 + a 3 = 5 :=
sorry

end sum_of_a_equals_five_l263_263005


namespace donald_paul_ratio_l263_263320

-- Let P be the number of bottles Paul drinks in one day.
-- Let D be the number of bottles Donald drinks in one day.
def paul_bottles (P : ℕ) := P = 3
def donald_bottles (D : ℕ) := D = 9

theorem donald_paul_ratio (P D : ℕ) (hP : paul_bottles P) (hD : donald_bottles D) : D / P = 3 :=
by {
  -- Insert proof steps here using the conditions.
  sorry
}

end donald_paul_ratio_l263_263320


namespace investment_total_correct_l263_263438

-- Define the initial investment, interest rate, and duration
def initial_investment : ℝ := 300
def monthly_interest_rate : ℝ := 0.10
def duration_in_months : ℝ := 2

-- Define the total amount after 2 months
noncomputable def total_after_two_months : ℝ := initial_investment * (1 + monthly_interest_rate) * (1 + monthly_interest_rate)

-- Define the correct answer
def correct_answer : ℝ := 363

-- The proof problem
theorem investment_total_correct :
  total_after_two_months = correct_answer :=
sorry

end investment_total_correct_l263_263438


namespace find_first_term_l263_263576

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l263_263576


namespace square_side_length_l263_263384

theorem square_side_length (x y : ℕ) (h_gcd : Nat.gcd x y = 5) (h_area : ∃ a : ℝ, a^2 = (169 / 6) * ↑(Nat.lcm x y)) : ∃ a : ℝ, a = 65 * Real.sqrt 2 :=
by
  sorry

end square_side_length_l263_263384


namespace number_of_testing_methods_l263_263189

-- Definitions based on conditions
def num_genuine_items : ℕ := 6
def num_defective_items : ℕ := 4
def total_tests : ℕ := 5

-- Theorem stating the number of testing methods
theorem number_of_testing_methods 
    (h1 : total_tests = 5) 
    (h2 : num_genuine_items = 6) 
    (h3 : num_defective_items = 4) :
    ∃ n : ℕ, n = 576 := 
sorry

end number_of_testing_methods_l263_263189


namespace problem1_l263_263599

theorem problem1 (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^(n+1) - 9) := 
  sorry

end problem1_l263_263599


namespace length_of_AB_l263_263820

-- Define the parabola and the line passing through the focus F
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line (x y : ℝ) : Prop := y = x - 1

theorem length_of_AB : 
  (∃ F : ℝ × ℝ, F = (1, 0) ∧ line F.1 F.2) →
  (∃ A B : ℝ × ℝ, parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64)) :=
by
  sorry

end length_of_AB_l263_263820


namespace sum_abs_of_roots_l263_263803

variables {p q r : ℤ}

theorem sum_abs_of_roots:
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 94 := by
  intro h1 h2
  sorry

end sum_abs_of_roots_l263_263803


namespace parallelogram_altitude_base_ratio_l263_263986

theorem parallelogram_altitude_base_ratio 
  (area base : ℕ) (h : ℕ) 
  (h_base : base = 9)
  (h_area : area = 162)
  (h_area_eq : area = base * h) : 
  h / base = 2 := 
by 
  -- placeholder for the proof
  sorry

end parallelogram_altitude_base_ratio_l263_263986


namespace prob_B_hired_is_3_4_prob_at_least_two_hired_l263_263999

-- Definitions for the conditions
def prob_A_hired : ℚ := 2 / 3
def prob_neither_A_nor_B_hired : ℚ := 1 / 12
def prob_B_and_C_hired : ℚ := 3 / 8

-- Targets to prove
theorem prob_B_hired_is_3_4 (P_A_hired : ℚ) (P_neither_A_nor_B_hired : ℚ) (P_B_and_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_neither_A_nor_B_hired_eq : P_neither_A_nor_B_hired = prob_neither_A_nor_B_hired)
    (P_B_and_C_hired_eq : P_B_and_C_hired = prob_B_and_C_hired)
    : ∃ x y : ℚ, y = 1 / 2 ∧ x = 3 / 4 :=
by
  sorry
  
theorem prob_at_least_two_hired (P_A_hired : ℚ) (P_B_hired : ℚ) (P_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_B_hired_eq : P_B_hired = 3 / 4)
    (P_C_hired_eq : P_C_hired = 1 / 2)
    : (P_A_hired * P_B_hired * P_C_hired) + 
      ((1 - P_A_hired) * P_B_hired * P_C_hired) + 
      (P_A_hired * (1 - P_B_hired) * P_C_hired) + 
      (P_A_hired * P_B_hired * (1 - P_C_hired)) = 2 / 3 :=
by
  sorry

end prob_B_hired_is_3_4_prob_at_least_two_hired_l263_263999


namespace largest_n_for_divisibility_l263_263398

theorem largest_n_for_divisibility :
  ∃ n : ℕ, (n + 15) ∣ (n^3 + 250) ∧ ∀ m : ℕ, ((m + 15) ∣ (m^3 + 250)) → (m ≤ 10) → (n = 10) :=
by {
  sorry
}

end largest_n_for_divisibility_l263_263398


namespace part1_part2_l263_263329

variable (R : ℝ) -- radius of the sphere
variable (x : ℝ) -- semi-vertical angle of the cone

def V1 : ℝ := (1 / 3 : ℝ) * Real.pi * R^3 * (1 + Real.sin x)^3 / (Real.cos x)^2 / Real.sin x 
def V2 : ℝ := 2 * Real.pi * R^3 

theorem part1 : V1 R x ≠ V2 R :=
by
  sorry

theorem part2 :
  let λ := V1 R x / V2 R
  (∀ {λ}, λ = 4 / 3) → (V1 R x / V2 R = 4 / 3 ∧ 2 * Real.arcsin (1 / 3) = 2 * x) :=
by
  sorry

end part1_part2_l263_263329


namespace split_into_similar_heaps_l263_263519

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l263_263519


namespace max_sum_of_squares_eq_50_l263_263257

theorem max_sum_of_squares_eq_50 :
  ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 50 → x + y ≥ x' + y') ∧ x + y = 10 := 
sorry

end max_sum_of_squares_eq_50_l263_263257


namespace unique_f_satisfies_eq_l263_263936

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * (x^2 + 2 * x - 1)

theorem unique_f_satisfies_eq (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x + f (1 - x) = x^2) : 
  ∀ x : ℝ, f x = (1 / 3) * (x^2 + 2 * x - 1) :=
sorry

end unique_f_satisfies_eq_l263_263936


namespace second_solution_percentage_l263_263743

theorem second_solution_percentage (P : ℝ) : 
  (28 * 0.30 + 12 * P = 40 * 0.45) → P = 0.8 :=
by
  intros h
  sorry

end second_solution_percentage_l263_263743


namespace age_difference_l263_263378

variable (E Y : ℕ)

theorem age_difference (hY : Y = 35) (hE : E - 15 = 2 * (Y - 15)) : E - Y = 20 := by
  -- Assertions and related steps could be handled subsequently.
  sorry

end age_difference_l263_263378


namespace direct_proportion_function_l263_263339

theorem direct_proportion_function (m : ℝ) (h1 : m^2 - 8 = 1) (h2 : m ≠ 3) : m = -3 :=
by
  sorry

end direct_proportion_function_l263_263339


namespace sum_first_9000_terms_l263_263880

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
a * ((1 - r^n) / (1 - r))

theorem sum_first_9000_terms (a r : ℝ) (h1 : geom_sum a r 3000 = 1000) 
                              (h2 : geom_sum a r 6000 = 1900) : 
                              geom_sum a r 9000 = 2710 := 
by sorry

end sum_first_9000_terms_l263_263880


namespace linear_function_difference_l263_263851

variable (g : ℝ → ℝ)
variable (h_linear : ∀ x y, g (x + y) = g x + g y)
variable (h_value : g 8 - g 4 = 16)

theorem linear_function_difference : g 16 - g 4 = 48 := by
  sorry

end linear_function_difference_l263_263851


namespace eddie_rate_l263_263865

variables (hours_sam hours_eddie rate_sam total_crates rate_eddie : ℕ)

def sam_conditions :=
  hours_sam = 6 ∧ rate_sam = 60

def eddie_conditions :=
  hours_eddie = 4 ∧ total_crates = hours_sam * rate_sam

theorem eddie_rate (hs : sam_conditions hours_sam rate_sam)
                   (he : eddie_conditions hours_sam hours_eddie rate_sam total_crates) :
  rate_eddie = 90 :=
by sorry

end eddie_rate_l263_263865


namespace mike_picked_32_limes_l263_263433

theorem mike_picked_32_limes (total_limes : ℕ) (alyssa_limes : ℕ) (mike_limes : ℕ) 
  (h1 : total_limes = 57) (h2 : alyssa_limes = 25) (h3 : mike_limes = total_limes - alyssa_limes) : 
  mike_limes = 32 :=
by
  sorry

end mike_picked_32_limes_l263_263433


namespace tan_sin_cos_l263_263808

theorem tan_sin_cos (θ : ℝ) (h : Real.tan θ = 1 / 2) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = - 4 / 5 := by 
  sorry

end tan_sin_cos_l263_263808


namespace paint_cans_needed_l263_263619

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l263_263619


namespace angle_triple_supplement_l263_263104

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l263_263104


namespace original_peaches_l263_263051

theorem original_peaches (picked: ℕ) (current: ℕ) (initial: ℕ) : 
  picked = 52 → 
  current = 86 → 
  initial = current - picked → 
  initial = 34 := 
by intros h1 h2 h3
   subst h1
   subst h2
   subst h3
   simp

end original_peaches_l263_263051


namespace interest_rate_l263_263870

-- Define the sum of money
def P : ℝ := 1800

-- Define the time period in years
def T : ℝ := 2

-- Define the difference in interests
def interest_difference : ℝ := 18

-- Define the relationship between simple interest, compound interest, and the interest rate
theorem interest_rate (R : ℝ) 
  (h1 : SI = P * R * T / 100)
  (h2 : CI = P * (1 + R/100)^2 - P)
  (h3 : CI - SI = interest_difference) :
  R = 10 :=
by
  sorry

end interest_rate_l263_263870


namespace cos_B_eq_zero_l263_263000

variable {a b c A B C : ℝ}
variable (h1 : ∀ A B C, 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
variable (h2 : b * Real.cos A = c)

theorem cos_B_eq_zero (h1 : a = b) (h2 : b * Real.cos A = c) : Real.cos B = 0 :=
sorry

end cos_B_eq_zero_l263_263000


namespace positive_real_number_solution_l263_263629

theorem positive_real_number_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 11) (h3 : (x - 6) / 11 = 6 / (x - 11)) : x = 17 :=
sorry

end positive_real_number_solution_l263_263629


namespace smallest_base10_integer_l263_263279

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l263_263279


namespace eval_expression_eq_54_l263_263321

theorem eval_expression_eq_54 : (3 * 4 * 6) * ((1/3 : ℚ) + 1/4 + 1/6) = 54 := 
by
  sorry

end eval_expression_eq_54_l263_263321


namespace probability_of_selecting_male_l263_263926

-- We define the proportions and ratio given in the problem.
def proportion_obese_men := 1 / 5
def proportion_obese_women := 1 / 10
def ratio_men_to_women := 3 / 2

-- From the given conditions, prove that the probability of selecting a male given that the individual is obese is 3/4.
theorem probability_of_selecting_male (P_A B : Prop) 
  (h_ratio: ratio_men_to_women = 3 / 2)
  (h_obese_men: P_A → proportion_obese_men)
  (h_obese_women: P_A → proportion_obese_women):
  (proportion_obese_men) * (3 / 5 : ℝ) / ((proportion_obese_men) * (3 / 5 : ℝ) + (proportion_obese_women) * (2 / 5 : ℝ)) = 3 / 4 := 
by
  sorry

end probability_of_selecting_male_l263_263926


namespace algebraic_expression_value_l263_263467

-- Define the premises as a Lean statement
theorem algebraic_expression_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a * (b + c) + b * (a + c) + c * (a + b) = -1 :=
sorry

end algebraic_expression_value_l263_263467


namespace ellipse_equation_standard_form_l263_263610

theorem ellipse_equation_standard_form :
  ∃ (a b : ℝ) (h k : ℝ), 
    a = (Real.sqrt 146 + Real.sqrt 242) / 2 ∧ 
    b = Real.sqrt ((Real.sqrt 146 + Real.sqrt 242) / 2)^2 - 9 ∧ 
    h = 1 ∧ 
    k = 4 ∧ 
    (∀ x y : ℝ, (x, y) = (12, -4) → 
      ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) :=
  sorry

end ellipse_equation_standard_form_l263_263610


namespace sum_of_digits_of_largest_valid_n_l263_263852

open List

-- Defining the problem in Lean
def is_single_digit_prime (n: ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_valid_prime_triplet (d e: ℕ) : Prop :=
  is_single_digit_prime d ∧ is_single_digit_prime e ∧ Prime (10 * d + e)

def largest_valid_product : ℕ :=
  max (max (2 * 3 * 23) (3 * 7 * 37)) (max (5 * 3 * 53) (7 * 3 * 73))

def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |> foldl (·+·) 0

theorem sum_of_digits_of_largest_valid_n : sum_of_digits largest_valid_product = 12 := by
  sorry

end sum_of_digits_of_largest_valid_n_l263_263852


namespace sum_of_solutions_of_quadratic_l263_263385

theorem sum_of_solutions_of_quadratic (x : ℝ) :
  x^2 - 6*x + 5 = 2*x - 8 →
  let a := (1 : ℝ) in
  let b := (-8 : ℝ) in
  let sum_of_roots := -b / a in
  sum_of_roots = 8 := 
by
  intro h
  let a := (1 : ℝ)
  let b := (-8 : ℝ)
  let sum_of_roots := -b / a
  have : x^2 - 8*x + 13 = 0 := by
    linarith [h]
  have h_sum : sum_of_roots = 8 := by
    rw [sum_of_roots]
    norm_num
  exact h_sum

end sum_of_solutions_of_quadratic_l263_263385


namespace smallest_integer_representation_l263_263276

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l263_263276


namespace sum_of_234_and_142_in_base_4_l263_263560

theorem sum_of_234_and_142_in_base_4 :
  (234 + 142) = 376 ∧ (376 + 0) = 256 * 1 + 64 * 1 + 16 * 3 + 4 * 2 + 1 * 0 :=
by sorry

end sum_of_234_and_142_in_base_4_l263_263560


namespace remainder_3001_3005_mod_23_l263_263736

theorem remainder_3001_3005_mod_23 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 :=
by {
  sorry
}

end remainder_3001_3005_mod_23_l263_263736


namespace geometry_problem_l263_263343

open EuclideanGeometry

variables (O A B M T P : Point)

noncomputable def circle (center : Point) (radius : ℝ) := {p | dist center p = radius}

def midpoint (A B : Point) : Point := {
  x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2
}

theorem geometry_problem
  (hO : inside_circle O A B)   -- O is the center of circle C1, AB is a chord of C1
  (hM : M = midpoint A B)       -- M is the midpoint of chord AB
  (h2 : T ∈ circle M (dist O M / 2))  -- T lies on circle C2 with OM as diameter
  (h3 : is_tangent P T)         -- Tangent to C2 at T intersects C1 at P
  : dist P A ^ 2 + dist P B ^ 2 = 4 * dist P T ^ 2 :=
sorry

end geometry_problem_l263_263343


namespace move_symmetric_point_left_l263_263835

-- Define the original point and the operations
def original_point : ℝ × ℝ := (-2, 3)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Prove the resulting point after the operations
theorem move_symmetric_point_left : move_left (symmetric_point original_point) 2 = (0, -3) :=
by
  sorry

end move_symmetric_point_left_l263_263835


namespace angle_C_is_3pi_over_4_l263_263023

theorem angle_C_is_3pi_over_4 (A B C : ℝ) (a b c : ℝ) (h_tri : 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (h_eq : b * Real.cos C + c * Real.sin B = 0) : C = 3 * π / 4 :=
by
  sorry

end angle_C_is_3pi_over_4_l263_263023


namespace complex_arithmetic_l263_263206

def Q : ℂ := 7 + 3 * Complex.I
def E : ℂ := 2 * Complex.I
def D : ℂ := 7 - 3 * Complex.I
def F : ℂ := 1 + Complex.I

theorem complex_arithmetic : (Q * E * D) + F = 1 + 117 * Complex.I := by
  sorry

end complex_arithmetic_l263_263206


namespace commute_times_abs_diff_l263_263908

def commute_times_avg (x y : ℝ) : Prop := (x + y + 7 + 8 + 9) / 5 = 8
def commute_times_var (x y : ℝ) : Prop := ((x - 8)^2 + (y - 8)^2 + (7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2) / 5 = 4

theorem commute_times_abs_diff (x y : ℝ) (h_avg : commute_times_avg x y) (h_var : commute_times_var x y) :
  |x - y| = 6 :=
sorry

end commute_times_abs_diff_l263_263908


namespace translation_line_segment_l263_263963

theorem translation_line_segment (a b : ℝ) :
  (∃ A B A1 B1: ℝ × ℝ,
    A = (1,0) ∧ B = (3,2) ∧ A1 = (a, 1) ∧ B1 = (4,b) ∧
    ∃ t : ℝ × ℝ, A + t = A1 ∧ B + t = B1) →
  a = 2 ∧ b = 3 :=
by
  sorry

end translation_line_segment_l263_263963


namespace johnny_red_pencils_l263_263354

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end johnny_red_pencils_l263_263354


namespace charity_event_probability_l263_263440

theorem charity_event_probability :
  let A_days := 3
  let total_days := 5
  let A_total_ways := Nat.choose total_days A_days
  let consecutive_days := 3
  let probability := consecutive_days / A_total_ways

  A_total_ways = 10 → -- A₅³ is the number of ways B, C, and D can be chosen to participate.
  probability = 1 / 20
:=
by
  sorry

end charity_event_probability_l263_263440


namespace surface_area_of_interior_of_box_l263_263153

-- Definitions from conditions in a)
def length : ℕ := 25
def width : ℕ := 40
def cut_side : ℕ := 4

-- The proof statement we need to prove, using the correct answer from b)
theorem surface_area_of_interior_of_box : 
  (length - 2 * cut_side) * (width - 2 * cut_side) + 2 * (cut_side * (length + width - 2 * cut_side)) = 936 :=
by
  sorry

end surface_area_of_interior_of_box_l263_263153


namespace inequality_solution_l263_263929

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
    (15 ≤ x * (x - 2) / (x - 5) ^ 2) ↔ (4.1933 ≤ x ∧ x < 5 ∨ 5 < x ∧ x ≤ 6.3767) :=
by
  sorry

end inequality_solution_l263_263929


namespace bisect_segment_l263_263363

variables {A B C D E P : Point}
variables {α β γ δ ε : Real} -- angles in degrees
variables {BD CE : Line}

-- Geometric predicates
def Angle (x y z : Point) : Real := sorry -- calculates the angle ∠xyz

def isMidpoint (M A B : Point) : Prop := sorry -- M is the midpoint of segment AB

-- Given Conditions
variables (h1 : convex_pentagon A B C D E)
          (h2 : Angle B A C = Angle C A D ∧ Angle C A D = Angle D A E)
          (h3 : Angle A B C = Angle A C D ∧ Angle A C D = Angle A D E)
          (h4 : intersects BD CE P)

-- Conclusion to be proved
theorem bisect_segment : isMidpoint P C D :=
by {
  sorry -- proof to be filled in
}

end bisect_segment_l263_263363


namespace other_toys_cost_1000_l263_263353

-- Definitions of the conditions
def cost_of_other_toys : ℕ := sorry
def cost_of_lightsaber (cost_of_other_toys : ℕ) : ℕ := 2 * cost_of_other_toys
def total_spent (cost_of_lightsaber cost_of_other_toys : ℕ) : ℕ := cost_of_lightsaber + cost_of_other_toys

-- The proof goal
theorem other_toys_cost_1000 (T : ℕ) (H1 : cost_of_lightsaber T = 2 * T) 
                            (H2 : total_spent (cost_of_lightsaber T) T = 3000) : T = 1000 := by
  sorry

end other_toys_cost_1000_l263_263353


namespace sum_of_fractions_l263_263578

theorem sum_of_fractions : (1/2 + 1/2 + 1/3 + 1/3 + 1/3) = 2 :=
by
  -- Proof goes here
  sorry

end sum_of_fractions_l263_263578


namespace find_a_bi_c_l263_263376

theorem find_a_bi_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_eq : (a - (b : ℤ)*I)^2 + c = 13 - 8*I) :
  a = 2 ∧ b = 2 ∧ c = 13 :=
by
  sorry

end find_a_bi_c_l263_263376


namespace min_value_fraction_l263_263324

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∃ x_min, x_min = 12 ∧ (∀ x > 6, (x * x) / (x - 6) ≥ 18) ∧ (x * x) / (x - 6) = 18) :=
sorry

end min_value_fraction_l263_263324


namespace complex_fraction_eval_l263_263750

theorem complex_fraction_eval (i : ℂ) (hi : i^2 = -1) : (3 + i) / (1 + i) = 2 - i := 
by 
  sorry

end complex_fraction_eval_l263_263750


namespace point_B_possible_values_l263_263858

-- Define point A
def A : ℝ := 1

-- Define the condition that B is 3 units away from A
def units_away (a b : ℝ) : ℝ := abs (b - a)

theorem point_B_possible_values :
  ∃ B : ℝ, units_away A B = 3 ∧ (B = 4 ∨ B = -2) := by
  sorry

end point_B_possible_values_l263_263858


namespace min_blue_eyes_with_lunchbox_l263_263185

theorem min_blue_eyes_with_lunchbox (B L : Finset Nat) (hB : B.card = 15) (hL : L.card = 25) (students : Finset Nat) (hst : students.card = 35)  : 
  ∃ (x : Finset Nat), x ⊆ B ∧ x ⊆ L ∧ x.card ≥ 5 :=
by
  sorry

end min_blue_eyes_with_lunchbox_l263_263185


namespace product_of_repeating_decimal_and_22_l263_263460

noncomputable def repeating_decimal_to_fraction : ℚ :=
  0.45 + 0.0045 * (10 ^ (-2 : ℤ))

theorem product_of_repeating_decimal_and_22 : (repeating_decimal_to_fraction * 22 = 10) :=
by
  sorry

end product_of_repeating_decimal_and_22_l263_263460


namespace fill_time_with_leak_l263_263530

theorem fill_time_with_leak (A L : ℝ) (hA : A = 1 / 5) (hL : L = 1 / 10) :
  1 / (A - L) = 10 :=
by 
  sorry

end fill_time_with_leak_l263_263530


namespace trigonometric_identity_l263_263651

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 / 2 :=
by
  sorry

end trigonometric_identity_l263_263651


namespace A_wins_match_prob_correct_l263_263838

def probA_wins_game : ℝ := 0.6
def probB_wins_game : ℝ := 0.4

def probA_wins_match : ℝ :=
  let probA_wins_first_two := probA_wins_game * probA_wins_game
  let probA_wins_first_and_third := probA_wins_game * probB_wins_game * probA_wins_game
  let probA_wins_last_two := probB_wins_game * probA_wins_game * probA_wins_game
  probA_wins_first_two + probA_wins_first_and_third + probA_wins_last_two

theorem A_wins_match_prob_correct : probA_wins_match = 0.648 := by
  sorry

end A_wins_match_prob_correct_l263_263838


namespace remove_remaining_wallpaper_time_l263_263934

noncomputable def time_per_wall : ℕ := 2
noncomputable def walls_dining_room : ℕ := 4
noncomputable def walls_living_room : ℕ := 4
noncomputable def walls_completed : ℕ := 1

theorem remove_remaining_wallpaper_time : 
    time_per_wall * (walls_dining_room - walls_completed) + time_per_wall * walls_living_room = 14 :=
by
  sorry

end remove_remaining_wallpaper_time_l263_263934


namespace inequality_holds_for_positive_reals_equality_condition_l263_263062

theorem inequality_holds_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_holds_for_positive_reals_equality_condition_l263_263062


namespace geometric_sequence_sum_l263_263001

theorem geometric_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n+1) = a n * q) → -- geometric sequence condition
  a 2 = 6 → -- first condition
  6 * a 1 + a 3 = 30 → -- second condition
  (∀ n, S_n n = (if q = 2 then 3*(2^n - 1) else if q = 3 then 3^n - 1 else 0)) :=
by intros
   sorry

end geometric_sequence_sum_l263_263001


namespace data_point_frequency_l263_263728

theorem data_point_frequency 
  (data : Type) 
  (categories : data → Prop) 
  (group_counts : data → ℕ) :
  ∀ d, categories d → group_counts d = frequency := sorry

end data_point_frequency_l263_263728


namespace binom_12_9_plus_binom_12_3_l263_263794

theorem binom_12_9_plus_binom_12_3 : (Nat.choose 12 9) + (Nat.choose 12 3) = 440 := by
  sorry

end binom_12_9_plus_binom_12_3_l263_263794


namespace pile_splitting_l263_263506

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l263_263506


namespace intersection_point_of_curves_l263_263212

theorem intersection_point_of_curves :
  (∃ (θ t : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 ∧ (x = sqrt 5 * cos θ) ∧ (y = sqrt 5 * sin θ) ∧ 
  (x = 1 - (sqrt 2) / 2 * t) ∧ (y = -(sqrt 2) / 2 * t)) ↔ (2, 1) :=
by
  sorry

end intersection_point_of_curves_l263_263212


namespace negation_proposition_l263_263720

open Real

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x - 1 > 0) :
  ¬ (∀ x : ℝ, x^2 - 2*x - 1 > 0) = ∃ x_0 : ℝ, x_0^2 - 2*x_0 - 1 ≤ 0 :=
by 
  sorry

end negation_proposition_l263_263720


namespace ratio_yx_l263_263208

variable (c x y : ℝ)

theorem ratio_yx (h1: x = 0.80 * c) (h2: y = 1.25 * c) : y / x = 25 / 16 := by
  -- Proof to be written here
  sorry

end ratio_yx_l263_263208


namespace total_students_l263_263083

-- Definition of the conditions given in the problem
def num5 : ℕ := 12
def num6 : ℕ := 6 * num5

-- The theorem representing the mathematically equivalent proof problem
theorem total_students : num5 + num6 = 84 :=
by
  sorry

end total_students_l263_263083


namespace bill_needs_paint_cans_l263_263614

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l263_263614


namespace hazel_sold_18_cups_to_kids_l263_263951

theorem hazel_sold_18_cups_to_kids:
  ∀ (total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup: ℕ),
     total_cups = 56 →
     cups_sold_construction = 28 →
     crew_remaining = total_cups - cups_sold_construction →
     last_cup = 1 →
     crew_remaining = cups_sold_kids + (cups_sold_kids / 2) + last_cup →
     cups_sold_kids = 18 :=
by
  intros total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup h_total h_construction h_remaining h_last h_equation
  sorry

end hazel_sold_18_cups_to_kids_l263_263951


namespace smallest_integer_representation_l263_263275

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l263_263275


namespace employees_without_increase_l263_263860

-- Define the constants and conditions
def total_employees : ℕ := 480
def salary_increase_percentage : ℕ := 10
def travel_allowance_increase_percentage : ℕ := 20

-- Define the calculations derived from conditions
def employees_with_salary_increase : ℕ := (salary_increase_percentage * total_employees) / 100
def employees_with_travel_allowance_increase : ℕ := (travel_allowance_increase_percentage * total_employees) / 100

-- Total employees who got increases assuming no overlap
def employees_with_increases : ℕ := employees_with_salary_increase + employees_with_travel_allowance_increase

-- The proof statement
theorem employees_without_increase :
  total_employees - employees_with_increases = 336 := by
  sorry

end employees_without_increase_l263_263860


namespace fraction_white_tulips_l263_263377

theorem fraction_white_tulips : 
  ∀ (total_tulips yellow_fraction red_fraction pink_fraction white_fraction : ℝ),
  total_tulips = 60 →
  yellow_fraction = 1 / 2 →
  red_fraction = 1 / 3 →
  pink_fraction = 1 / 4 →
  white_fraction = 
    ((total_tulips * (1 - yellow_fraction)) * (1 - red_fraction) * (1 - pink_fraction)) / total_tulips →
  white_fraction = 1 / 4 :=
by
  intros total_tulips yellow_fraction red_fraction pink_fraction white_fraction 
    h_total h_yellow h_red h_pink h_white
  sorry

end fraction_white_tulips_l263_263377


namespace product_not_ending_in_1_l263_263864

theorem product_not_ending_in_1 : ∃ a b : ℕ, 111111 = a * b ∧ (a % 10 ≠ 1) ∧ (b % 10 ≠ 1) := 
sorry

end product_not_ending_in_1_l263_263864


namespace bill_needs_paint_cans_l263_263616

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l263_263616


namespace douglas_votes_in_county_D_l263_263487

noncomputable def percent_votes_in_county_D (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
    (total_votes : ℝ) (percent_A percent_B percent_C percent_D total_percent : ℝ) : Prop :=
  (votes_A / (5 * x) = 0.70) ∧
  (votes_B / (3 * x) = 0.58) ∧
  (votes_C / (2 * x) = 0.50) ∧
  (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62 ∧
  (votes_D / (4 * x) = percent_D)

theorem douglas_votes_in_county_D 
  (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
  (total_votes : ℝ := 14 * x) 
  (percent_A percent_B percent_C total_percent percent_D : ℝ)
  (h1 : votes_A / (5 * x) = 0.70) 
  (h2 : votes_B / (3 * x) = 0.58) 
  (h3 : votes_C / (2 * x) = 0.50) 
  (h4 : (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62) : 
  percent_votes_in_county_D x votes_A votes_B votes_C votes_D total_votes percent_A percent_B percent_C 0.61 total_percent :=
by
  constructor
  exact h1
  constructor
  exact h2
  constructor
  exact h3
  constructor
  exact h4
  sorry

end douglas_votes_in_county_D_l263_263487


namespace smallest_perfect_square_4_10_18_l263_263404

theorem smallest_perfect_square_4_10_18 :
  ∃ n : ℕ, (∃ k : ℕ, n = k^2) ∧ (4 ∣ n) ∧ (10 ∣ n) ∧ (18 ∣ n) ∧ n = 900 := 
  sorry

end smallest_perfect_square_4_10_18_l263_263404


namespace morning_routine_time_l263_263953

section

def time_for_teeth_and_face : ℕ := 3
def time_for_cooking : ℕ := 14
def time_for_reading_while_cooking : ℕ := time_for_cooking - time_for_teeth_and_face
def additional_time_for_reading : ℕ := 1
def total_time_for_reading : ℕ := time_for_reading_while_cooking + additional_time_for_reading
def time_for_eating : ℕ := 6

def total_time_to_school : ℕ := time_for_cooking + time_for_eating

theorem morning_routine_time :
  total_time_to_school = 21 := sorry

end

end morning_routine_time_l263_263953


namespace repeated_process_pure_alcohol_l263_263054

theorem repeated_process_pure_alcohol : 
  ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, 2 * (1 / 2 : ℝ)^(m : ℝ) ≥ 0.2 := by
  sorry

end repeated_process_pure_alcohol_l263_263054


namespace unit_digit_is_nine_l263_263771

theorem unit_digit_is_nine (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : a ≠ 0) (h4 : a + b + a * b = 10 * a + b) : b = 9 := 
by 
  sorry

end unit_digit_is_nine_l263_263771


namespace angle_triple_supplementary_l263_263106

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l263_263106


namespace prob_male_given_obese_correct_l263_263927

-- Definitions based on conditions
def ratio_male_female : ℚ := 3 / 2
def prob_obese_male : ℚ := 1 / 5
def prob_obese_female : ℚ := 1 / 10

-- Definition of events
def total_employees : ℚ := ratio_male_female + 1

-- Probability calculations
def prob_male : ℚ := ratio_male_female / total_employees
def prob_female : ℚ := 1 / total_employees

def prob_obese_and_male : ℚ := prob_male * prob_obese_male
def prob_obese_and_female : ℚ := prob_female * prob_obese_female

def prob_obese : ℚ := prob_obese_and_male + prob_obese_and_female

def prob_male_given_obese : ℚ := prob_obese_and_male / prob_obese

-- Theorem statement
theorem prob_male_given_obese_correct : prob_male_given_obese = 3 / 4 := sorry

end prob_male_given_obese_correct_l263_263927


namespace find_number_l263_263295

theorem find_number (x : ℝ) : 0.5 * 56 = 0.3 * x + 13 ↔ x = 50 :=
by
  -- Proof would go here
  sorry

end find_number_l263_263295


namespace carlton_outfits_l263_263784

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end carlton_outfits_l263_263784


namespace problem_statement_l263_263465

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ (ab < 1) :=
by
  sorry

end problem_statement_l263_263465


namespace triple_supplementary_angle_l263_263097

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l263_263097


namespace paint_cans_needed_l263_263617

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l263_263617


namespace trig_identity_evaluation_l263_263452

theorem trig_identity_evaluation :
  let θ1 := 70 * Real.pi / 180 -- angle 70 degrees in radians
  let θ2 := 10 * Real.pi / 180 -- angle 10 degrees in radians
  let θ3 := 20 * Real.pi / 180 -- angle 20 degrees in radians
  (Real.tan θ1 * Real.cos θ2 * (Real.sqrt 3 * Real.tan θ3 - 1) = -1) := 
by 
  sorry

end trig_identity_evaluation_l263_263452


namespace exists_n_satisfying_condition_l263_263984

-- Definition of the divisor function d(n)
def d (n : ℕ) : ℕ := Nat.divisors n |>.card

-- Theorem statement
theorem exists_n_satisfying_condition : ∃ n : ℕ, ∀ i : ℕ, i ≤ 1402 → (d n : ℚ) / d (n + i) > 1401 ∧ (d n : ℚ) / d (n - i) > 1401 :=
by
  sorry

end exists_n_satisfying_condition_l263_263984


namespace multiply_divide_repeating_decimals_l263_263737

theorem multiply_divide_repeating_decimals :
  (8 * (1 / 3) / 1) = 8 / 3 := by
  sorry

end multiply_divide_repeating_decimals_l263_263737


namespace right_angled_trapezoid_base_height_l263_263876

theorem right_angled_trapezoid_base_height {a b : ℝ} (h : a = b) :
  ∃ (base height : ℝ), base = a ∧ height = b := 
by
  sorry

end right_angled_trapezoid_base_height_l263_263876


namespace relationship_A_B_l263_263691

variable (x y : ℝ)

noncomputable def A : ℝ := (x + y) / (1 + x + y)

noncomputable def B : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem relationship_A_B (hx : 0 < x) (hy : 0 < y) : A x y < B x y := sorry

end relationship_A_B_l263_263691


namespace warriors_won_40_games_l263_263562

variable (H F W K R S : ℕ)

-- Conditions as given in the problem
axiom hawks_won_more_games_than_falcons : H > F
axiom knights_won_more_than_30 : K > 30
axiom warriors_won_more_than_knights_but_fewer_than_royals : W > K ∧ W < R
axiom squires_tied_with_falcons : S = F

-- The proof statement
theorem warriors_won_40_games : W = 40 :=
sorry

end warriors_won_40_games_l263_263562


namespace shaded_area_correct_l263_263755

noncomputable def total_shaded_area (floor_length : ℝ) (floor_width : ℝ) (tile_size : ℝ) (circle_radius : ℝ) : ℝ :=
  let tile_area := tile_size ^ 2
  let circle_area := Real.pi * circle_radius ^ 2
  let shaded_area_per_tile := tile_area - circle_area
  let floor_area := floor_length * floor_width
  let number_of_tiles := floor_area / tile_area
  number_of_tiles * shaded_area_per_tile 

theorem shaded_area_correct : total_shaded_area 12 15 2 1 = 180 - 45 * Real.pi := sorry

end shaded_area_correct_l263_263755


namespace zero_point_neg_x₀_l263_263067

-- Define odd function property
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define zero point condition for the function
def is_zero_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = Real.exp x₀

-- The main theorem to be proved
theorem zero_point_neg_x₀ (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : is_odd_function f)
  (h_zero : is_zero_point f x₀) :
  f (-x₀) * Real.exp x₀ + 1 = 0 :=
sorry

end zero_point_neg_x₀_l263_263067


namespace greatest_of_given_numbers_l263_263889

-- Defining the given conditions
def a := 1000 + 0.01
def b := 1000 * 0.01
def c := 1000 / 0.01
def d := 0.01 / 1000
def e := 1000 - 0.01

-- Prove that c is the greatest
theorem greatest_of_given_numbers : c = max a (max b (max d e)) :=
by
  -- Placeholder for the proof
  sorry

end greatest_of_given_numbers_l263_263889


namespace smallest_base10_integer_l263_263271

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l263_263271


namespace sum_div_9_remainder_l263_263128

theorem sum_div_9_remainder :
  ∑ i in Finset.range 21, i % 9 = 4 :=
  sorry

end sum_div_9_remainder_l263_263128


namespace sequence_value_l263_263348

theorem sequence_value (a : ℕ → ℕ) (h₁ : ∀ n, a (2 * n) = a (2 * n - 1) + (-1 : ℤ)^n) 
                        (h₂ : ∀ n, a (2 * n + 1) = a (2 * n) + n)
                        (h₃ : a 1 = 1) : a 20 = 46 :=
by 
  sorry

end sequence_value_l263_263348


namespace find_A_l263_263130

theorem find_A (A B : ℕ) (h1: 3 + 6 * (100 + 10 * A + B) = 691) (h2 : 100 ≤ 6 * (100 + 10 * A + B) ∧ 6 * (100 + 10 * A + B) < 1000) : 
A = 8 :=
sorry

end find_A_l263_263130


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l263_263509

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l263_263509


namespace certain_number_l263_263292

theorem certain_number (x : ℝ) (h : 4 * x = 200) : x = 50 :=
by
  sorry

end certain_number_l263_263292


namespace ratio_of_age_differences_l263_263059

variable (R J K : ℕ)

-- conditions
axiom h1 : R = J + 6
axiom h2 : R + 2 = 2 * (J + 2)
axiom h3 : (R + 2) * (K + 2) = 108

-- statement to prove
theorem ratio_of_age_differences : (R - J) = 2 * (R - K) := 
sorry

end ratio_of_age_differences_l263_263059


namespace example_problem_l263_263552

variables (a b : ℕ)

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem example_problem (hcf_ab : HCF 385 180 = 30) (a_def: a = 385) (b_def: b = 180) :
  LCM 385 180 = 2310 := 
by
  sorry

end example_problem_l263_263552


namespace only_function_l263_263596

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, divides (f m + f n) (m + n)

theorem only_function (f : ℕ → ℕ) (h : satisfies_condition f) : f = id :=
by
  -- Proof goes here.
  sorry

end only_function_l263_263596


namespace crayon_colors_correct_l263_263758

-- The Lean code will define the conditions and the proof statement as follows:
noncomputable def crayon_problem := 
  let crayons_per_box := (160 / (5 * 4)) -- Total crayons / Total boxes
  let colors := (crayons_per_box / 2) -- Crayons per box / Crayons per color
  colors = 4

-- This is the theorem that needs to be proven:
theorem crayon_colors_correct : crayon_problem := by
  sorry

end crayon_colors_correct_l263_263758


namespace population_of_metropolitan_county_l263_263673

theorem population_of_metropolitan_county : 
  let average_population := 5500
  let two_populous_cities_population := 2 * average_population
  let remaining_cities := 25 - 2
  let remaining_population := remaining_cities * average_population
  let total_population := (2 * two_populous_cities_population) + remaining_population
  total_population = 148500 := by
sorry

end population_of_metropolitan_county_l263_263673


namespace pencils_are_left_l263_263161

-- Define the conditions
def original_pencils : ℕ := 87
def removed_pencils : ℕ := 4

-- Define the expected outcome
def pencils_left : ℕ := original_pencils - removed_pencils

-- Prove that the number of pencils left in the jar is 83
theorem pencils_are_left : pencils_left = 83 := by
  -- Placeholder for the proof
  sorry

end pencils_are_left_l263_263161


namespace find_original_price_each_stocking_l263_263529

open Real

noncomputable def original_stocking_price (total_stockings total_cost_per_stocking discounted_cost monogramming_cost total_cost : ℝ) : ℝ :=
  let stocking_cost_before_monogramming := total_cost - (total_stockings * monogramming_cost)
  let original_price := stocking_cost_before_monogramming / (total_stockings * discounted_cost)
  original_price

theorem find_original_price_each_stocking :
  original_stocking_price 9 122.22 0.9 5 1035 = 122.22 := by
  sorry

end find_original_price_each_stocking_l263_263529


namespace side_length_of_S2_l263_263534

theorem side_length_of_S2 (r s : ℝ) 
  (h1 : 2 * r + s = 2025) 
  (h2 : 2 * r + 3 * s = 3320) :
  s = 647.5 :=
by {
  -- proof omitted
  sorry
}

end side_length_of_S2_l263_263534


namespace multiples_of_6_and_8_l263_263014

open Nat

theorem multiples_of_6_and_8 (n m k : ℕ) (h₁ : n = 33) (h₂ : m = 25) (h₃ : k = 8) :
  (n - k) + (m - k) = 42 :=
by
  sorry

end multiples_of_6_and_8_l263_263014


namespace carlton_outfit_count_l263_263785

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end carlton_outfit_count_l263_263785


namespace least_positive_x_multiple_of_53_l263_263892

theorem least_positive_x_multiple_of_53 :
  ∃ (x : ℕ), (x > 0) ∧ ((2 * x)^2 + 2 * 47 * (2 * x) + 47^2) % 53 = 0 ∧ x = 6 :=
by
  sorry

end least_positive_x_multiple_of_53_l263_263892


namespace lighter_boxes_weight_l263_263486

noncomputable def weight_lighter_boxes (W L H : ℕ) : Prop :=
  L + H = 30 ∧
  (L * W + H * 20) / 30 = 18 ∧
  (H - 15) = 0 ∧
  (15 + L - H = 15 ∧ 15 * 16 = 15 * W)

theorem lighter_boxes_weight :
  ∃ W, ∀ L H, weight_lighter_boxes W L H → W = 16 :=
by sorry

end lighter_boxes_weight_l263_263486


namespace calc_expression_l263_263782

theorem calc_expression : 
  (abs (Real.sqrt 2 - Real.sqrt 3) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3) :=
by
  -- Given that sqrt(3) > sqrt(2)
  have h1 : Real.sqrt 3 > Real.sqrt 2 := by sorry
  -- And cos(45°) = sqrt(2)/2
  have h2 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  -- Now prove the expression equivalency
  sorry

end calc_expression_l263_263782


namespace sally_took_out_5_onions_l263_263982

theorem sally_took_out_5_onions (X Y : ℕ) 
    (h1 : 4 + 9 - Y + X = X + 8) : Y = 5 := 
by
  sorry

end sally_took_out_5_onions_l263_263982


namespace factorial_div_power_of_two_odd_l263_263739

theorem factorial_div_power_of_two_odd (n k : ℕ) (h₁ : k = (nat.binary_length n).succ - nat.count_ones n) (h₂ : nat.count_ones n = k) :
  odd (n! / 2^(n - k)) :=
begin
  sorry
end

end factorial_div_power_of_two_odd_l263_263739


namespace assume_proof_by_contradiction_l263_263589

theorem assume_proof_by_contradiction (a b : ℤ) (hab : ∃ k : ℤ, ab = 3 * k) :
  (¬ (∃ k : ℤ, a = 3 * k) ∧ ¬ (∃ k : ℤ, b = 3 * k)) :=
sorry

end assume_proof_by_contradiction_l263_263589


namespace class_duration_l263_263681

theorem class_duration (x : ℝ) (h : 3 * x = 6) : x = 2 :=
by
  sorry

end class_duration_l263_263681


namespace factor_expression_l263_263315

theorem factor_expression (x : ℚ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := sorry

end factor_expression_l263_263315


namespace value_of_a_l263_263475

def f (x : ℝ) : ℝ := x^2 + 9
def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 25) : a = 3 :=
by
  sorry

end value_of_a_l263_263475


namespace blue_first_red_second_probability_l263_263905

-- Define the initial conditions
def initial_red_marbles : ℕ := 4
def initial_white_marbles : ℕ := 6
def initial_blue_marbles : ℕ := 2
def total_marbles : ℕ := initial_red_marbles + initial_white_marbles + initial_blue_marbles

-- Probability calculation under the given conditions
def probability_blue_first : ℚ := initial_blue_marbles / total_marbles
def remaining_marbles_after_blue : ℕ := total_marbles - 1
def remaining_red_marbles : ℕ := initial_red_marbles
def probability_red_second_given_blue_first : ℚ := remaining_red_marbles / remaining_marbles_after_blue

-- Combined probability
def combined_probability : ℚ := probability_blue_first * probability_red_second_given_blue_first

-- The statement to be proved
theorem blue_first_red_second_probability :
  combined_probability = 2 / 33 :=
sorry

end blue_first_red_second_probability_l263_263905


namespace correctStatement_l263_263741

def isValidInput : String → Bool
| "INPUT a, b, c;" => true
| "INPUT x=3;" => false
| _ => false

def isValidOutput : String → Bool
| "PRINT 20,3*2." => true
| "PRINT A=4;" => false
| _ => false

def isValidStatement : String → Bool
| stmt => (isValidInput stmt ∨ isValidOutput stmt)

theorem correctStatement : isValidStatement "PRINT 20,3*2." = true ∧ 
                           ¬(isValidStatement "INPUT a; b; c;" = true) ∧ 
                           ¬(isValidStatement "INPUT x=3;" = true) ∧ 
                           ¬(isValidStatement "PRINT A=4;" = true) := 
by sorry

end correctStatement_l263_263741


namespace volume_of_locations_eq_27sqrt6pi_over_8_l263_263408

noncomputable def volumeOfLocationSet : ℝ :=
  let sqrt2_inv := 1 / (2 * Real.sqrt 2)
  let points := [ (sqrt2_inv, sqrt2_inv, sqrt2_inv),
                  (sqrt2_inv, sqrt2_inv, -sqrt2_inv),
                  (sqrt2_inv, -sqrt2_inv, sqrt2_inv),
                  (-sqrt2_inv, sqrt2_inv, sqrt2_inv) ]
  let condition (x y z : ℝ) : Prop :=
    4 * (x^2 + y^2 + z^2) + 3 / 2 ≤ 15
  let r := Real.sqrt (27 / 8)
  let volume := (4/3) * Real.pi * r^3
  volume

theorem volume_of_locations_eq_27sqrt6pi_over_8 :
  volumeOfLocationSet = 27 * Real.sqrt 6 * Real.pi / 8 :=
sorry

end volume_of_locations_eq_27sqrt6pi_over_8_l263_263408


namespace beef_weight_after_processing_l263_263910

theorem beef_weight_after_processing
  (initial_weight : ℝ)
  (weight_loss_percentage : ℝ)
  (processed_weight : ℝ)
  (h1 : initial_weight = 892.31)
  (h2 : weight_loss_percentage = 0.35)
  (h3 : processed_weight = initial_weight * (1 - weight_loss_percentage)) :
  processed_weight = 579.5015 :=
by
  sorry

end beef_weight_after_processing_l263_263910


namespace a_3_and_a_4_sum_l263_263850

theorem a_3_and_a_4_sum (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℚ) :
  (1 - (1 / (2 * x))) ^ 6 = a_0 + a_1 * (1 / x) + a_2 * (1 / x) ^ 2 + a_3 * (1 / x) ^ 3 + 
  a_4 * (1 / x) ^ 4 + a_5 * (1 / x) ^ 5 + a_6 * (1 / x) ^ 6 →
  a_3 + a_4 = -25 / 16 :=
sorry

end a_3_and_a_4_sum_l263_263850


namespace minimum_value_of_function_l263_263459

theorem minimum_value_of_function : ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_of_function_l263_263459


namespace q_computation_l263_263499

def q : ℤ → ℤ → ℤ :=
  λ x y =>
    if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
    else if x < 0 ∧ y < 0 then x - 3 * y
    else 2 * x + y

theorem q_computation : q (q 2 (-2)) (q (-4) (-1)) = 3 :=
by {
  sorry
}

end q_computation_l263_263499


namespace garage_sale_items_l263_263162

-- Definition of conditions
def is_18th_highest (num_highest: ℕ) : Prop := num_highest = 17
def is_25th_lowest (num_lowest: ℕ) : Prop := num_lowest = 24

-- Theorem statement
theorem garage_sale_items (num_highest num_lowest total_items: ℕ) 
  (h1: is_18th_highest num_highest) (h2: is_25th_lowest num_lowest) :
  total_items = num_highest + num_lowest + 1 :=
by
  -- Proof omitted
  sorry

end garage_sale_items_l263_263162


namespace right_triangle_perimeter_l263_263424

noncomputable def perimeter_of_right_triangle (x : ℝ) : ℝ :=
  let y := x + 15
  let c := Real.sqrt (x^2 + y^2)
  x + y + c

theorem right_triangle_perimeter
  (h₁ : ∀ a b : ℝ, a * b = 2 * 150)  -- The area condition
  (h₂ : ∀ a b : ℝ, b = a + 15)       -- One leg is 15 units longer than the other
  : perimeter_of_right_triangle 11.375 = 66.47 :=
by
  sorry

end right_triangle_perimeter_l263_263424


namespace highest_number_paper_l263_263484

theorem highest_number_paper
  (n : ℕ)
  (P : ℝ)
  (hP : P = 0.010309278350515464)
  (hP_formula : 1 / n = P) :
  n = 97 :=
by
  -- Placeholder for proof
  sorry

end highest_number_paper_l263_263484


namespace bill_needs_paint_cans_l263_263615

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l263_263615


namespace total_enjoyable_gameplay_hours_l263_263350

def total_gameplay_hours : ℕ := 100
def grinding_percentage : ℝ := 0.8
def additional_enjoyable_hours : ℕ := 30

theorem total_enjoyable_gameplay_hours : 
  (total_gameplay_hours - (total_gameplay_hours * grinding_percentage).toNat + additional_enjoyable_hours = 50) :=
by
  sorry

end total_enjoyable_gameplay_hours_l263_263350


namespace probability_two_same_color_l263_263342

/-- There are 4 balls in a box, 2 red and 2 white. Two balls are to be drawn without replacement.
    The probability of drawing two balls of the same color is 1/3. -/
theorem probability_two_same_color (red white : ℕ)
    (h_red : red = 2) (h_white : white = 2) : 
    (probability (λ (event : Finset Ball), event.card = 2 ∧ 
    ∀ b ∈ event, b.color = Color.Red ∨ b.color = Color.White) = 1/3) :=
sorry

end probability_two_same_color_l263_263342


namespace roberto_outfits_l263_263058

-- Roberto's wardrobe constraints
def num_trousers : ℕ := 5
def num_shirts : ℕ := 6
def num_jackets : ℕ := 4
def num_shoes : ℕ := 3
def restricted_jacket_shoes : ℕ := 2

-- The total number of valid outfits
def total_outfits_with_constraint : ℕ := 330

-- Proving the equivalent of the problem statement
theorem roberto_outfits :
  (num_trousers * num_shirts * (num_jackets - 1) * num_shoes) + (num_trousers * num_shirts * 1 * restricted_jacket_shoes) = total_outfits_with_constraint :=
by
  sorry

end roberto_outfits_l263_263058


namespace one_angle_greater_135_l263_263002

noncomputable def angles_sum_not_form_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : Prop :=
  ∀ (A B C : ℝ), 
   (A < a + b ∧ A < a + c ∧ A < b + c) →
  (B < a + b ∧ B < a + c ∧ B < b + c) →
  (C < a + b ∧ C < a + c ∧ C < b + c) →
  ∃ α β γ, α > 135 ∧ β < 60 ∧ γ < 60 ∧ α + β + γ = 180

theorem one_angle_greater_135 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : angles_sum_not_form_triangle a b c ha hb hc) :
  ∃ α β γ, α > 135 ∧ α + β + γ = 180 :=
sorry

end one_angle_greater_135_l263_263002


namespace apples_more_than_grapes_l263_263402

theorem apples_more_than_grapes 
  (total_weight : ℕ) (weight_ratio_apples : ℕ) (weight_ratio_peaches : ℕ) (weight_ratio_grapes : ℕ) : 
  weight_ratio_apples = 12 → 
  weight_ratio_peaches = 8 → 
  weight_ratio_grapes = 7 → 
  total_weight = 54 →
  ((12 * total_weight / (12 + 8 + 7)) - (7 * total_weight / (12 + 8 + 7))) = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end apples_more_than_grapes_l263_263402


namespace total_people_on_hike_l263_263254

def cars : Nat := 3
def people_per_car : Nat := 4
def taxis : Nat := 6
def people_per_taxi : Nat := 6
def vans : Nat := 2
def people_per_van : Nat := 5

theorem total_people_on_hike :
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van = 58 := by
  sorry

end total_people_on_hike_l263_263254


namespace quadratic_min_value_l263_263207

theorem quadratic_min_value (p q r : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q + r ≥ -r) : q = p^2 / 4 :=
sorry

end quadratic_min_value_l263_263207


namespace eccentricity_theorem_l263_263224

noncomputable def ellipse : set (ℝ × ℝ) := 
  {p | p.1^2 / 16 + p.2^2 / b^2 = 1}

noncomputable def foci_1 : ℝ × ℝ := (-c, 0)
noncomputable def foci_2 : ℝ × ℝ := (c, 0)

def max_AF2_BF2_value : ℝ := 10

def eccentricity_of_ellipse : ℝ := 
  let a := 4 in
  let b := 2 * Real.sqrt 3 in
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_theorem (h : ellipse) (foci_1 foci_2) (A B : ℝ × ℝ)
  (l : set ℝ × ℝ) (hl : foci_1 ∈ l) 
  (interAB : A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ l ∧ B ∈ l)
  (hmax : ∀ A B, |(A - foci_2).length + (B - foci_2).length| ≤ max_AF2_BF2_value) :
  eccentricity_of_ellipse = 1/2 := 
sorry

end eccentricity_theorem_l263_263224


namespace cubic_equation_roots_l263_263318

theorem cubic_equation_roots (a b c d r s t : ℝ) (h_eq : a ≠ 0) 
(ht1 : a * r^3 + b * r^2 + c * r + d = 0)
(ht2 : a * s^3 + b * s^2 + c * s + d = 0)
(ht3 : a * t^3 + b * t^2 + c * t + d = 0)
(h1 : r * s = 3) 
(h2 : r * t = 3) 
(h3 : s * t = 3) : 
c = 3 * a := 
sorry

end cubic_equation_roots_l263_263318


namespace angle_triple_supplementary_l263_263112

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l263_263112


namespace split_stones_l263_263514

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l263_263514


namespace least_possible_sum_l263_263854

theorem least_possible_sum
  (a b x y z : ℕ)
  (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hpos_x : 0 < x) (hpos_y : 0 < y)
  (hpos_z : 0 < z)
  (h : 3 * a = 7 * b ∧ 7 * b = 5 * x ∧ 5 * x = 4 * y ∧ 4 * y = 6 * z) :
  a + b + x + y + z = 459 :=
by
  sorry

end least_possible_sum_l263_263854


namespace value_of_x_plus_y_l263_263016

variable {x y : ℝ}

theorem value_of_x_plus_y (h1 : 1 / x + 1 / y = 1) (h2 : 1 / x - 1 / y = 9) : x + y = -1 / 20 := 
sorry

end value_of_x_plus_y_l263_263016


namespace calculation_result_l263_263167

theorem calculation_result :
  1500 * 451 * 0.0451 * 25 = 7627537500 :=
by
  -- Simply state without proof as instructed
  sorry

end calculation_result_l263_263167


namespace four_leaved_clovers_percentage_l263_263210

noncomputable def percentage_of_four_leaved_clovers (clovers total_clovers purple_four_leaved_clovers : ℕ ) : ℝ := 
  (purple_four_leaved_clovers * 4 * 100) / total_clovers 

theorem four_leaved_clovers_percentage :
  percentage_of_four_leaved_clovers 500 500 25 = 20 := 
by
  -- application of conditions and arithmetic simplification.
  sorry

end four_leaved_clovers_percentage_l263_263210


namespace system_solution_l263_263789

theorem system_solution (x b y : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (h3 : x = 3) :
  b = -1 :=
by
  -- proof to be filled in
  sorry

end system_solution_l263_263789


namespace split_into_similar_heaps_l263_263520

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l263_263520


namespace calories_in_250_grams_is_106_l263_263972

noncomputable def total_calories_apple : ℝ := 150 * (46 / 100)
noncomputable def total_calories_orange : ℝ := 50 * (45 / 100)
noncomputable def total_calories_carrot : ℝ := 300 * (40 / 100)
noncomputable def total_calories_mix : ℝ := total_calories_apple + total_calories_orange + total_calories_carrot
noncomputable def total_weight_mix : ℝ := 150 + 50 + 300
noncomputable def caloric_density : ℝ := total_calories_mix / total_weight_mix
noncomputable def calories_in_250_grams : ℝ := 250 * caloric_density

theorem calories_in_250_grams_is_106 : calories_in_250_grams = 106 :=
by
  sorry

end calories_in_250_grams_is_106_l263_263972


namespace larger_number_is_34_l263_263881

theorem larger_number_is_34 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : y = 34 :=
by
  sorry

end larger_number_is_34_l263_263881


namespace min_value_expression_l263_263628

theorem min_value_expression (x y z : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : z > 1) : ∃ C, C = 12 ∧
  ∀ (x y z : ℝ), x > 1 → y > 1 → z > 1 → (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ C := by
  sorry

end min_value_expression_l263_263628


namespace infinite_representable_and_nonrepresentable_terms_l263_263793

def a (n : ℕ) : ℕ :=
  2^n + 2^(n / 2)

def is_representable (k : ℕ) : Prop :=   
  -- A nonnegative integer is defined to be representable if it can
  -- be expressed as a sum of distinct terms from the sequence a(n).
  sorry  -- Definition will depend on the specific notion of representability

theorem infinite_representable_and_nonrepresentable_terms :
  (∃ᶠ n in at_top, is_representable (a n)) ∧ (∃ᶠ n in at_top, ¬is_representable (a n)) :=
sorry  -- This is the main theorem claiming infinitely many representable and non-representable terms.

end infinite_representable_and_nonrepresentable_terms_l263_263793


namespace total_cows_l263_263760

theorem total_cows (cows : ℕ) (h1 : cows / 3 + cows / 5 + cows / 6 + 12 = cows) : cows = 40 :=
sorry

end total_cows_l263_263760


namespace phosphorus_atoms_l263_263607

theorem phosphorus_atoms (x : ℝ) : 122 = 26.98 + 30.97 * x + 64 → x = 1 := by
sorry

end phosphorus_atoms_l263_263607


namespace average_price_l263_263900

theorem average_price (books1 books2 : ℕ) (price1 price2 : ℝ)
  (h1 : books1 = 65) (h2 : price1 = 1380)
  (h3 : books2 = 55) (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2) = 19 :=
by
  sorry

end average_price_l263_263900


namespace mod_mult_congruence_l263_263066

theorem mod_mult_congruence (n : ℤ) (h1 : 215 ≡ 65 [ZMOD 75])
  (h2 : 789 ≡ 39 [ZMOD 75]) (h3 : 215 * 789 ≡ n [ZMOD 75]) (hn : 0 ≤ n ∧ n < 75) :
  n = 60 :=
by
  sorry

end mod_mult_congruence_l263_263066


namespace youngest_child_age_l263_263077

theorem youngest_child_age (x : ℝ) (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by sorry

end youngest_child_age_l263_263077


namespace cuberoot_eq_l263_263186

open Real

theorem cuberoot_eq (x : ℝ) (h: (5:ℝ) * x + 4 = (5:ℝ) ^ 3 / (2:ℝ) ^ 3) : x = 93 / 40 := by
  sorry

end cuberoot_eq_l263_263186


namespace ben_apples_difference_l263_263776

theorem ben_apples_difference (B P T : ℕ) (h1 : P = 40) (h2 : T = 18) (h3 : (3 / 8) * B = T) :
  B - P = 8 :=
sorry

end ben_apples_difference_l263_263776


namespace domain_of_x_l263_263964

-- Conditions
def is_defined_num (x : ℝ) : Prop := x + 1 >= 0
def not_zero_den (x : ℝ) : Prop := x ≠ 2

-- Proof problem statement
theorem domain_of_x (x : ℝ) : (is_defined_num x ∧ not_zero_den x) ↔ (x >= -1 ∧ x ≠ 2) := by
  sorry

end domain_of_x_l263_263964


namespace simplify_and_evaluate_l263_263704

-- Define the given expression
noncomputable def given_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m^2) + 3 * m) / (m + 1)

-- Define the condition
def condition (m : ℝ) : Prop :=
  m = Real.sqrt 3

-- Define the correct answer
def correct_answer : ℝ :=
  1 - Real.sqrt 3

-- State the theorem
theorem simplify_and_evaluate 
  (m : ℝ) (h : condition m) : 
  given_expression m = correct_answer := by
  sorry

end simplify_and_evaluate_l263_263704


namespace product_modulo_23_l263_263735

theorem product_modulo_23 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 0 :=
by {
  have h1 : 3001 % 23 = 19 := rfl,
  have h2 : 3002 % 23 = 20 := rfl,
  have h3 : 3003 % 23 = 21 := rfl,
  have h4 : 3004 % 23 = 22 := rfl,
  have h5 : 3005 % 23 = 0 := rfl,
  sorry
}

end product_modulo_23_l263_263735


namespace pizza_slices_per_pizza_l263_263232

theorem pizza_slices_per_pizza (h : ∀ (mrsKaplanSlices bobbySlices pizzas : ℕ), 
  mrsKaplanSlices = 3 ∧ mrsKaplanSlices = bobbySlices / 4 ∧ pizzas = 2 → bobbySlices / pizzas = 6) : 
  ∃ (bobbySlices pizzas : ℕ), bobbySlices / pizzas = 6 :=
by
  existsi (3 * 4)
  existsi 2
  sorry

end pizza_slices_per_pizza_l263_263232


namespace fill_time_l263_263751

def inflow_rate : ℕ := 24 -- gallons per second
def outflow_rate : ℕ := 4 -- gallons per second
def basin_volume : ℕ := 260 -- gallons

theorem fill_time (inflow_rate outflow_rate basin_volume : ℕ) (h₁ : inflow_rate = 24) (h₂ : outflow_rate = 4) 
  (h₃ : basin_volume = 260) : basin_volume / (inflow_rate - outflow_rate) = 13 :=
by
  sorry

end fill_time_l263_263751


namespace inequality_proof_l263_263477

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) : 
  1 / a + 4 / b ≥ 9 / 4 :=
by
  sorry

end inequality_proof_l263_263477


namespace det_of_matrix_l263_263779

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem det_of_matrix :
  determinant_2x2 5 (-2) 3 1 = 11 := by
  sorry

end det_of_matrix_l263_263779


namespace multiples_of_2_correct_multiples_of_3_correct_l263_263591

def numbers : Set ℕ := {28, 35, 40, 45, 53, 10, 78}

def multiples_of_2_in_numbers : Set ℕ := {n ∈ numbers | n % 2 = 0}
def multiples_of_3_in_numbers : Set ℕ := {n ∈ numbers | n % 3 = 0}

theorem multiples_of_2_correct :
  multiples_of_2_in_numbers = {28, 40, 10, 78} :=
sorry

theorem multiples_of_3_correct :
  multiples_of_3_in_numbers = {45, 78} :=
sorry

end multiples_of_2_correct_multiples_of_3_correct_l263_263591


namespace quadratic_solution_set_R_l263_263387

theorem quadratic_solution_set_R (a b c : ℝ) (h1 : a ≠ 0) (h2 : a < 0) (h3 : b^2 - 4 * a * c < 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
by sorry

end quadratic_solution_set_R_l263_263387


namespace radius_of_circle_l263_263716

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 81 * π) : r = 9 :=
by
  sorry

end radius_of_circle_l263_263716


namespace Eric_return_time_l263_263175

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l263_263175


namespace area_of_moon_slice_l263_263925

-- Definitions of the conditions
def larger_circle_radius := 5
def larger_circle_center := (2, 0)
def smaller_circle_radius := 2
def smaller_circle_center := (0, 0)

-- Prove the area of the moon slice
theorem area_of_moon_slice : 
  (1/4) * (larger_circle_radius^2 * Real.pi) - (1/4) * (smaller_circle_radius^2 * Real.pi) = (21 * Real.pi) / 4 :=
by
  sorry

end area_of_moon_slice_l263_263925


namespace avg_expenditure_Feb_to_July_l263_263283

noncomputable def avg_expenditure_Jan_to_Jun : ℝ := 4200
noncomputable def expenditure_January : ℝ := 1200
noncomputable def expenditure_July : ℝ := 1500
noncomputable def total_months_Jan_to_Jun : ℝ := 6
noncomputable def total_months_Feb_to_July : ℝ := 6

theorem avg_expenditure_Feb_to_July :
  (avg_expenditure_Jan_to_Jun * total_months_Jan_to_Jun - expenditure_January + expenditure_July) / total_months_Feb_to_July = 4250 :=
by sorry

end avg_expenditure_Feb_to_July_l263_263283


namespace elder_person_present_age_l263_263245

def younger_age : ℕ
def elder_age : ℕ

-- Conditions
axiom age_difference (y e : ℕ) : e = y + 16
axiom age_relation_6_years_ago (y e : ℕ) : e - 6 = 3 * (y - 6)

-- Proof of the present age of the elder person
theorem elder_person_present_age (y e : ℕ) (h1 : e = y + 16) (h2 : e - 6 = 3 * (y - 6)) : e = 30 :=
sorry

end elder_person_present_age_l263_263245


namespace cubic_polynomial_evaluation_l263_263652

theorem cubic_polynomial_evaluation
  (f : ℚ → ℚ)
  (cubic_f : ∃ a b c d : ℚ, ∀ x, f x = a*x^3 + b*x^2 + c*x + d)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 :=
sorry

end cubic_polynomial_evaluation_l263_263652


namespace max_marked_cells_100x100_board_l263_263891

theorem max_marked_cells_100x100_board : 
  ∃ n, (3 * n + 1 = 100) ∧ (2 * n + 1) * (n + 1) = 2278 :=
by
  sorry

end max_marked_cells_100x100_board_l263_263891


namespace tangent_function_intersection_l263_263644

theorem tangent_function_intersection (ω : ℝ) (hω : ω > 0) (h_period : (π / ω) = 3 * π) :
  let f (x : ℝ) := Real.tan (ω * x + π / 3)
  f π = -Real.sqrt 3 :=
by
  sorry

end tangent_function_intersection_l263_263644


namespace smith_family_mean_age_l263_263553

theorem smith_family_mean_age :
  let children_ages := [8, 8, 8, 12, 11]
  let dogs_ages := [3, 4]
  let all_ages := children_ages ++ dogs_ages
  let total_ages := List.sum all_ages
  let total_individuals := List.length all_ages
  (total_ages : ℚ) / (total_individuals : ℚ) = 7.71 :=
by
  sorry

end smith_family_mean_age_l263_263553


namespace simplify_expression_l263_263064

theorem simplify_expression :
  8 * (15 / 4) * (-45 / 50) = - (12 / 25) :=
by
  sorry

end simplify_expression_l263_263064


namespace pure_imaginary_number_l263_263658

theorem pure_imaginary_number (a : ℝ) (ha : (1 + a) / (1 + a^2) = 0) : a = -1 :=
sorry

end pure_imaginary_number_l263_263658


namespace circle_problem_l263_263035

theorem circle_problem
  (E F : ℝ × ℝ)
  (G C1_center : ℝ × ℝ)
  (P A B M N : ℝ × ℝ)
  (hE : E = (-2, 0))
  (hF : F = (-4, 2))
  (hA : A = (-6, 0))
  (hB : B = (-2, 0))
  (hG : G = (-2, -4))
  (h_c1_center : 2 * C1_center.1 - C1_center.2 + 8 = 0)
  (hP_not_A_B : P ≠ A ∧ P ≠ B)
  (C ON_PA : ∃ k : ℝ, M = (0, k*P.2 / (P.1 + 6)))
  (C ON_PB : ∃ k : ℝ, N = (0, k*P.2 / (P.1 + 2))) :
  ∃ Center : ℝ × ℝ, 
  (∀ x y : ℝ, (x + 4)^2 + y^2 = 4) ∧ 
  (∀ k : ℝ, 3*k + 4*-4 + 22 = 0 ∨ k = -2) ∧ 
  (∃ Point : ℝ × ℝ, Point = (-2*√3, 0)) :=
sorry

end circle_problem_l263_263035


namespace answered_both_questions_correctly_l263_263340

theorem answered_both_questions_correctly (P_A P_B P_A_prime_inter_B_prime : ℝ)
  (h1 : P_A = 70 / 100) (h2 : P_B = 55 / 100) (h3 : P_A_prime_inter_B_prime = 20 / 100) :
  P_A + P_B - (1 - P_A_prime_inter_B_prime) = 45 / 100 := 
by
  sorry

end answered_both_questions_correctly_l263_263340


namespace faster_train_speed_l263_263886

theorem faster_train_speed
  (slower_train_speed : ℝ := 60) -- speed of the slower train in km/h
  (length_train1 : ℝ := 1.10) -- length of the slower train in km
  (length_train2 : ℝ := 0.9) -- length of the faster train in km
  (cross_time_sec : ℝ := 47.99999999999999) -- crossing time in seconds
  (cross_time : ℝ := cross_time_sec / 3600) -- crossing time in hours
  (total_distance : ℝ := length_train1 + length_train2) -- total distance covered
  (relative_speed : ℝ := total_distance / cross_time) -- relative speed
  (faster_train_speed : ℝ := relative_speed - slower_train_speed) -- speed of the faster train
  : faster_train_speed = 90 :=
by
  sorry

end faster_train_speed_l263_263886


namespace average_weight_of_all_players_l263_263346

-- Definitions based on conditions
def num_forwards : ℕ := 8
def avg_weight_forwards : ℝ := 75
def num_defensemen : ℕ := 12
def avg_weight_defensemen : ℝ := 82

-- Total number of players
def total_players : ℕ := num_forwards + num_defensemen

-- Values derived from conditions
def total_weight_forwards : ℝ := avg_weight_forwards * num_forwards
def total_weight_defensemen : ℝ := avg_weight_defensemen * num_defensemen
def total_weight : ℝ := total_weight_forwards + total_weight_defensemen

-- Theorem to prove the average weight of all players
theorem average_weight_of_all_players : total_weight / total_players = 79.2 :=
by
  sorry

end average_weight_of_all_players_l263_263346


namespace range_of_x_l263_263812

theorem range_of_x (x : ℝ) (h1 : (x + 2) * (x - 3) ≤ 0) (h2 : |x + 1| ≥ 2) : 
  1 ≤ x ∧ x ≤ 3 :=
sorry

end range_of_x_l263_263812


namespace tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l263_263749

-- First problem: Given condition and expression to be proved equal to the correct answer.
theorem tan_alpha_eq_two_and_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * Real.pi - α) + Real.cos (Real.pi + α)) / 
  (Real.cos (α - Real.pi) - Real.cos (3 * Real.pi / 2 - α)) = -3 := sorry

-- Second problem: Given expression to be proved simplified to the correct answer.
theorem sin_tan_simplify :
  Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi/180)) = 1 := sorry

end tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l263_263749


namespace sum_digits_n_plus_one_l263_263044

/-- 
Let S(n) be the sum of the digits of a positive integer n.
Given S(n) = 29, prove that the possible values of S(n + 1) are 3, 12, or 30.
-/
theorem sum_digits_n_plus_one (S : ℕ → ℕ) (n : ℕ) (h : S n = 29) :
  S (n + 1) = 3 ∨ S (n + 1) = 12 ∨ S (n + 1) = 30 := 
sorry

end sum_digits_n_plus_one_l263_263044


namespace percent_with_university_diploma_l263_263286

theorem percent_with_university_diploma (a b c d : ℝ) (h1 : a = 0.12) (h2 : b = 0.25) (h3 : c = 0.40) 
    (h4 : d = c - a) (h5 : ¬c = 1) : 
    d + (b * (1 - c)) = 0.43 := 
by 
    sorry

end percent_with_university_diploma_l263_263286


namespace power_div_eq_l263_263091

theorem power_div_eq (a : ℕ) (h : 36 = 6^2) : (6^12 / 36^5) = 36 := by
  sorry

end power_div_eq_l263_263091


namespace cubes_sum_l263_263714

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 1) (h2 : ab + ac + bc = -4) (h3 : abc = -6) :
  a^3 + b^3 + c^3 = -5 :=
by
  sorry

end cubes_sum_l263_263714


namespace Keith_initial_picked_l263_263849

-- Definitions based on the given conditions
def Mike_picked := 12
def Keith_gave_away := 46
def remaining_pears := 13

-- Question: Prove that Keith initially picked 47 pears.
theorem Keith_initial_picked :
  ∃ K : ℕ, K = 47 ∧ (K - Keith_gave_away + Mike_picked = remaining_pears) :=
sorry

end Keith_initial_picked_l263_263849


namespace square_tiles_count_l263_263756

theorem square_tiles_count (a b : ℕ) (h1 : a + b = 25) (h2 : 3 * a + 4 * b = 84) : b = 9 := by
  sorry

end square_tiles_count_l263_263756


namespace sum_of_x_and_y_l263_263019

theorem sum_of_x_and_y (x y : ℤ) (h1 : 3 + x = 5) (h2 : -3 + y = 5) : x + y = 10 :=
by
  sorry

end sum_of_x_and_y_l263_263019


namespace hat_price_after_discounts_l263_263426

-- Defining initial conditions
def initial_price : ℝ := 15
def first_discount_percent : ℝ := 0.25
def second_discount_percent : ℝ := 0.50

-- Defining the expected final price after applying both discounts
def expected_final_price : ℝ := 5.625

-- Lean statement to prove the final price after both discounts is as expected
theorem hat_price_after_discounts : 
  let first_reduced_price := initial_price * (1 - first_discount_percent)
  let second_reduced_price := first_reduced_price * (1 - second_discount_percent)
  second_reduced_price = expected_final_price := sorry

end hat_price_after_discounts_l263_263426


namespace intersection_of_M_and_N_l263_263647

-- Define sets M and N
def M : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℤ := {0, 1, 2}

-- The theorem to be proven: M ∩ N = {0, 1, 2}
theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_of_M_and_N_l263_263647


namespace solve_for_x_l263_263631

def α(x : ℚ) : ℚ := 4 * x + 9
def β(x : ℚ) : ℚ := 9 * x + 6

theorem solve_for_x (x : ℚ) (h : α(β(x)) = 8) : x = -25 / 36 :=
by
  sorry

end solve_for_x_l263_263631


namespace maximum_marks_l263_263862

theorem maximum_marks (M : ℝ) (P : ℝ) 
  (h1 : P = 0.45 * M) -- 45% of the maximum marks to pass
  (h2 : P = 210 + 40) -- Pradeep's marks plus failed marks

  : M = 556 := 
sorry

end maximum_marks_l263_263862


namespace same_oxidation_state_HNO3_N2O5_l263_263034

def oxidation_state_HNO3 (H O: Int) : Int := 1 + 1 + (3 * (-2))
def oxidation_state_N2O5 (H O: Int) : Int := (2 * 1) + (5 * (-2))
def oxidation_state_substances_equal : Prop :=
  oxidation_state_HNO3 1 (-2) = oxidation_state_N2O5 1 (-2)

theorem same_oxidation_state_HNO3_N2O5 : oxidation_state_substances_equal :=
  by
  sorry

end same_oxidation_state_HNO3_N2O5_l263_263034


namespace intersection_A_B_union_A_B_complement_intersection_A_B_l263_263330

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def A_inter_B : Set ℝ := { x | 2 ≤ x ∧ x < 6 }
def A_union_B : Set ℝ := { x | 1 < x ∧ x ≤ 8 }
def A_compl_inter_B : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = A_inter_B := by
  sorry

theorem union_A_B :
  A ∪ B = A_union_B := by
  sorry

theorem complement_intersection_A_B :
  (Aᶜ ∩ B) = A_compl_inter_B := by
  sorry

end intersection_A_B_union_A_B_complement_intersection_A_B_l263_263330


namespace bc_product_l263_263082

theorem bc_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b * r - c = 0) → b * c = 30 :=
by
  sorry

end bc_product_l263_263082


namespace water_saving_percentage_l263_263847

/-- 
Given:
1. The old toilet uses 5 gallons of water per flush.
2. The household flushes 15 times per day.
3. John saved 1800 gallons of water in June.

Prove that the percentage of water saved per flush by the new toilet compared 
to the old one is 80%.
-/
theorem water_saving_percentage 
  (old_toilet_usage_per_flush : ℕ)
  (flushes_per_day : ℕ)
  (savings_in_june : ℕ)
  (days_in_june : ℕ) :
  old_toilet_usage_per_flush = 5 →
  flushes_per_day = 15 →
  savings_in_june = 1800 →
  days_in_june = 30 →
  (old_toilet_usage_per_flush * flushes_per_day * days_in_june - savings_in_june)
  * 100 / (old_toilet_usage_per_flush * flushes_per_day * days_in_june) = 80 :=
by 
  sorry

end water_saving_percentage_l263_263847


namespace elder_age_is_30_l263_263246

/-- The ages of two persons differ by 16 years, and 6 years ago, the elder one was 3 times as old as the younger one. 
Prove that the present age of the elder person is 30 years. --/
theorem elder_age_is_30 (y e: ℕ) (h₁: e = y + 16) (h₂: e - 6 = 3 * (y - 6)) : e = 30 := 
sorry

end elder_age_is_30_l263_263246


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l263_263511

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l263_263511


namespace collinear_points_in_cube_l263_263436

def collinear_groups_in_cube : Prop :=
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let center_point := 1
  let total_groups :=
    (vertices * (vertices - 1) / 2) + (face_centers * 1 / 2) + (edge_midpoints * 3 / 2)
  total_groups = 49

theorem collinear_points_in_cube : collinear_groups_in_cube :=
  by
    sorry

end collinear_points_in_cube_l263_263436


namespace sum_mod_1_to_20_l263_263129

theorem sum_mod_1_to_20 :
  (∑ i in finset.range 21, i) % 9 = 3 :=
by
  sorry

end sum_mod_1_to_20_l263_263129


namespace pipe_q_fills_in_9_hours_l263_263699

theorem pipe_q_fills_in_9_hours (x : ℝ) :
  (1 / 3 + 1 / x + 1 / 18 = 1 / 2) → x = 9 :=
by {
  sorry
}

end pipe_q_fills_in_9_hours_l263_263699


namespace average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l263_263873

-- Conditions from the problem statement
def initial_daily_sales : ℕ := 20
def profit_per_box : ℕ := 40
def additional_sales_per_yuan_reduction : ℕ := 2

-- Part 1: New average daily sales after a 10 yuan reduction
theorem average_daily_sales_after_10_yuan_reduction :
  (initial_daily_sales + 10 * additional_sales_per_yuan_reduction) = 40 :=
  sorry

-- Part 2: Price reduction needed to achieve a daily sales profit of 1200 yuan
theorem price_reduction_for_1200_yuan_profit :
  ∃ (x : ℕ), 
  (profit_per_box - x) * (initial_daily_sales + x * additional_sales_per_yuan_reduction) = 1200 ∧ x = 20 :=
  sorry

end average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l263_263873


namespace cube_root_inequality_l263_263956

theorem cube_root_inequality {a b : ℝ} (h : a > b) : (a^(1/3)) > (b^(1/3)) :=
sorry

end cube_root_inequality_l263_263956


namespace distinct_m_value_l263_263349

theorem distinct_m_value (a b : ℝ) (m : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_b_eq_2a : b = 2 * a) (h_m_eq_neg2a_b : m = -2 * a / b) : 
    ∃! (m : ℝ), m = -1 :=
by sorry

end distinct_m_value_l263_263349


namespace work_completion_l263_263747

noncomputable def efficiency (p q: ℕ) := q = 3 * p / 5

theorem work_completion (p q : ℕ) (h1 : efficiency p q) (h2: p * 24 = 100) :
  2400 / (p + q) = 15 :=
by 
  sorry

end work_completion_l263_263747


namespace paint_cans_needed_l263_263618

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l263_263618


namespace ratio_of_sum_l263_263899

theorem ratio_of_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := 
by 
  sorry

end ratio_of_sum_l263_263899


namespace geometric_progressions_common_ratio_l263_263893

theorem geometric_progressions_common_ratio (a b p q : ℝ) :
  (∀ n : ℕ, (a * p^n + b * q^n) = (a * b) * ((p^n + q^n)/a)) →
  p = q := by
  sorry

end geometric_progressions_common_ratio_l263_263893


namespace vasya_is_not_mistaken_l263_263394

theorem vasya_is_not_mistaken (X Y N A B : ℤ)
  (h_sum : X + Y = N)
  (h_tanya : A * X + B * Y ≡ 0 [ZMOD N]) :
  B * X + A * Y ≡ 0 [ZMOD N] :=
sorry

end vasya_is_not_mistaken_l263_263394


namespace count_squares_ending_in_4_l263_263649

theorem count_squares_ending_in_4 (n : ℕ) : 
  (∀ k : ℕ, (n^2 < 5000) → (n^2 % 10 = 4) → (k ≤ 70)) → 
  (∃ m : ℕ, m = 14) :=
by 
  sorry

end count_squares_ending_in_4_l263_263649


namespace max_m_value_l263_263831

variables {x y m : ℝ}

theorem max_m_value (h1 : 4 * x + 3 * y = 4 * m + 5)
                     (h2 : 3 * x - y = m - 1)
                     (h3 : x + 4 * y ≤ 3) :
                     m ≤ -1 :=
sorry

end max_m_value_l263_263831


namespace f_increasing_on_interval_l263_263950

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (x^2, x + 1)
noncomputable def vec_b (x t : ℝ) : ℝ × ℝ := (1 - x, t)

noncomputable def f (x t : ℝ) : ℝ :=
  let (a1, a2) := vec_a x
  let (b1, b2) := vec_b x t
  a1 * b1 + a2 * b2

noncomputable def f_prime (x t : ℝ) : ℝ :=
  2 * x - 3 * x^2 + t

theorem f_increasing_on_interval :
  ∀ t x, -1 < x → x < 1 → (0 ≤ f_prime x t) → (t ≥ 5) :=
sorry

end f_increasing_on_interval_l263_263950


namespace number_of_real_solutions_l263_263564

noncomputable def f (x : ℝ) : ℝ := 2^(-x) + x^2 - 3

theorem number_of_real_solutions :
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (∀ x : ℝ, f x = 0 → (x = x₁ ∨ x = x₂)) :=
by
  sorry

end number_of_real_solutions_l263_263564


namespace div_by_64_l263_263497

theorem div_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (5^n - 8*n^2 + 4*n - 1) :=
sorry

end div_by_64_l263_263497


namespace simplify_expression_l263_263540

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l263_263540


namespace sum_of_solutions_l263_263840

theorem sum_of_solutions (y : ℤ) (x1 x2 : ℤ) (h1 : y = 8) (h2 : x1^2 + y^2 = 145) (h3 : x2^2 + y^2 = 145) : x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l263_263840


namespace units_digit_base9_addition_l263_263800

theorem units_digit_base9_addition : 
  (∃ (d₁ d₂ : ℕ), d₁ < 9 ∧ d₂ < 9 ∧ (85 % 9 = d₁) ∧ (37 % 9 = d₂)) → ((d₁ + d₂) % 9 = 3) :=
by
  sorry

end units_digit_base9_addition_l263_263800


namespace orange_count_in_bin_l263_263144

-- Definitions of the conditions
def initial_oranges : Nat := 5
def oranges_thrown_away : Nat := 2
def new_oranges_added : Nat := 28

-- The statement of the proof problem
theorem orange_count_in_bin : initial_oranges - oranges_thrown_away + new_oranges_added = 31 :=
by
  sorry

end orange_count_in_bin_l263_263144


namespace grade_assignment_ways_l263_263306

theorem grade_assignment_ways : (4 ^ 12) = 16777216 := by
  sorry

end grade_assignment_ways_l263_263306


namespace sequence_terminates_final_value_l263_263855

-- Define the function Lisa uses to update the number
def f (x : ℕ) : ℕ :=
  let a := x / 10
  let b := x % 10
  a + 4 * b

-- Prove that for any initial value x0, the sequence eventually becomes periodic and ends.
theorem sequence_terminates (x0 : ℕ) : ∃ N : ℕ, ∃ j : ℕ, N ≠ j ∧ (Nat.iterate f N x0) = (Nat.iterate f j x0) :=
  by sorry

-- Given the starting value, show the sequence stabilizes at 39
theorem final_value (x0 : ℕ) (h : x0 = 53^2022 - 1) : ∃ N : ℕ, Nat.iterate f N x0 = 39 :=
  by sorry

end sequence_terminates_final_value_l263_263855


namespace min_number_of_4_dollar_frisbees_l263_263895

theorem min_number_of_4_dollar_frisbees 
  (x y : ℕ) 
  (h1 : x + y = 60)
  (h2 : 3 * x + 4 * y = 200) 
  : y = 20 :=
sorry

end min_number_of_4_dollar_frisbees_l263_263895


namespace max_variance_l263_263996

theorem max_variance (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) : 
  ∃ q, p * (1 - p) ≤ q ∧ q = 1 / 4 :=
by
  existsi (1 / 4)
  sorry

end max_variance_l263_263996


namespace calculate_fraction_l263_263921

theorem calculate_fraction :
  (10^9 / (2 * 10^5) = 5000) :=
  sorry

end calculate_fraction_l263_263921


namespace pencils_needed_l263_263672

theorem pencils_needed (pencilsA : ℕ) (pencilsB : ℕ) (classroomsA : ℕ) (classroomsB : ℕ) (total_shortage : ℕ)
  (hA : pencilsA = 480)
  (hB : pencilsB = 735)
  (hClassA : classroomsA = 6)
  (hClassB : classroomsB = 9)
  (hShortage : total_shortage = 85) 
  : 90 = 6 + 5 * ((total_shortage / (classroomsA + classroomsB)) + 1) * classroomsB :=
by {
  sorry
}

end pencils_needed_l263_263672


namespace abs_ineq_range_l263_263636

theorem abs_ineq_range (x : ℝ) : |x - 3| + |x + 1| ≥ 4 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

end abs_ineq_range_l263_263636


namespace angle_triple_supplement_l263_263125

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263125


namespace find_m_value_l263_263203

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)
def vector_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the condition that vector_sum is parallel to vector_b
def vectors_parallel (m : ℝ) : Prop :=
  let (x1, y1) := vector_sum m
  let (x2, y2) := vector_b
  x1 * y2 - x2 * y1 = 0

-- The statement to prove
theorem find_m_value : ∃ m : ℝ, vectors_parallel m ∧ m = -2 / 3 :=
by {
  sorry
}

end find_m_value_l263_263203


namespace ratio_fourth_to_sixth_l263_263978

-- Definitions from the conditions
def fourth_level_students := 40
def sixth_level_students := 40
def seventh_level_students := 2 * fourth_level_students

-- Statement to prove
theorem ratio_fourth_to_sixth : 
  fourth_level_students / sixth_level_students = 1 :=
by
  -- Proof skipped
  sorry

end ratio_fourth_to_sixth_l263_263978


namespace tan_sin_equality_l263_263781

theorem tan_sin_equality :
  (Real.tan (30 * Real.pi / 180))^2 + (Real.sin (45 * Real.pi / 180))^2 = 5 / 6 :=
by sorry

end tan_sin_equality_l263_263781


namespace mike_seashells_l263_263968

theorem mike_seashells (initial total : ℕ) (h1 : initial = 79) (h2 : total = 142) :
    total - initial = 63 :=
by
  sorry

end mike_seashells_l263_263968


namespace specialSignLanguage_l263_263256

theorem specialSignLanguage (S : ℕ) 
  (h1 : (S + 2) * (S + 2) = S * S + 1288) : S = 321 := 
by
  sorry

end specialSignLanguage_l263_263256


namespace valid_factorizations_of_1870_l263_263825

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_valid_factor1 (n : ℕ) : Prop := 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 * p2

def is_valid_factor2 (n : ℕ) : Prop := 
  ∃ (p k : ℕ), is_prime p ∧ (k = 4 ∨ k = 6 ∨ k = 8 ∨ k = 9) ∧ n = p * k

theorem valid_factorizations_of_1870 : 
  ∃ a b : ℕ, a * b = 1870 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ 
  ((is_valid_factor1 a ∧ is_valid_factor2 b) ∨ (is_valid_factor1 b ∧ is_valid_factor2 a)) ∧ 
  (a = 34 ∧ b = 55 ∨ a = 55 ∧ b = 34) ∧ 
  (¬∃ x y : ℕ, x * y = 1870 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ 
  ((is_valid_factor1 x ∧ is_valid_factor2 y) ∨ (is_valid_factor1 y ∧ is_valid_factor2 x)) ∧ 
  (x ≠ 34 ∨ y ≠ 55 ∨ x ≠ 55 ∨ y ≠ 34)) :=
sorry

end valid_factorizations_of_1870_l263_263825


namespace max_value_expression_l263_263586

theorem max_value_expression (r : ℝ) : ∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68 ∧ (∀ s : ℝ, -5 * s^2 + 40 * s - 12 ≤ 68) :=
sorry

end max_value_expression_l263_263586


namespace angle_C_length_CD_area_range_l263_263844

-- 1. Prove C = π / 3 given (2a - b)cos C = c cos B
theorem angle_C (a b c : ℝ) (A B C : ℝ) (h : (2 * a - b) * Real.cos C = c * Real.cos B) : 
  C = Real.pi / 3 := sorry

-- 2. Prove the length of CD is 6√3 / 5 given a = 2, b = 3, and CD is the angle bisector of angle C
theorem length_CD (a b x : ℝ) (C D : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : x = (6 * Real.sqrt 3) / 5) : 
  x = (6 * Real.sqrt 3) / 5 := sorry

-- 3. Prove the range of values for the area of acute triangle ABC is (8√3 / 3, 4√3] given a cos B + b cos A = 4
theorem area_range (a b : ℝ) (A B C : ℝ) (S : Set ℝ) (h1 : a * Real.cos B + b * Real.cos A = 4) 
  (h2 : S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3)) : 
  S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3) := sorry

end angle_C_length_CD_area_range_l263_263844


namespace cost_per_charge_l263_263011

theorem cost_per_charge
  (charges : ℕ) (budget left : ℝ) (cost_per_charge : ℝ)
  (charges_eq : charges = 4)
  (budget_eq : budget = 20)
  (left_eq : left = 6) :
  cost_per_charge = (budget - left) / charges :=
by
  apply sorry

end cost_per_charge_l263_263011


namespace parabola_distance_l263_263327

open Real

theorem parabola_distance (x₀ : ℝ) (h₁ : ∃ p > 0, (x₀^2 = 2 * p * 2) ∧ (2 + p / 2 = 5 / 2)) : abs (sqrt (x₀^2 + 4)) = 2 * sqrt 2 :=
by
  rcases h₁ with ⟨p, hp, h₀, h₂⟩
  sorry

end parabola_distance_l263_263327


namespace michael_remaining_money_l263_263050

variables (m b n : ℝ) (h1 : (1 : ℝ) / 3 * m = 1 / 2 * n * b) (h2 : 5 = m / 15)

theorem michael_remaining_money : m - (2 / 3 * m + m / 15) = 4 / 15 * m :=
by
  have hb1 : 2 / 3 * m = (2 * m) / 3 := by ring
  have hb2 : m / 15 = (1 * m) / 15 := by ring
  rw [hb1, hb2]
  sorry

end michael_remaining_money_l263_263050


namespace eric_return_home_time_l263_263181

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l263_263181


namespace simplest_common_denominator_fraction_exist_l263_263255

variable (x y : ℝ)

theorem simplest_common_denominator_fraction_exist :
  let d1 := x + y
  let d2 := x - y
  let d3 := x^2 - y^2
  (d3 = d1 * d2) → 
    ∀ n, (n = d1 * d2) → 
      (∃ m, (d1 * m = n) ∧ (d2 * m = n) ∧ (d3 * m = n)) :=
by
  sorry

end simplest_common_denominator_fraction_exist_l263_263255


namespace vector_magnitude_sum_l263_263332

noncomputable def magnitude_sum (a b : ℝ) (θ : ℝ) := by
  let dot_product := a * b * Real.cos θ
  let a_square := a ^ 2
  let b_square := b ^ 2
  let magnitude := Real.sqrt (a_square + 2 * dot_product + b_square)
  exact magnitude

theorem vector_magnitude_sum (a b : ℝ) (θ : ℝ)
  (ha : a = 2) (hb : b = 1) (hθ : θ = Real.pi / 4) :
  magnitude_sum a b θ = Real.sqrt (5 + 2 * Real.sqrt 2) := by
  rw [ha, hb, hθ, magnitude_sum]
  sorry

end vector_magnitude_sum_l263_263332


namespace tangerine_count_l263_263086

-- Definitions based directly on the conditions
def initial_oranges : ℕ := 5
def remaining_oranges : ℕ := initial_oranges - 2
def remaining_tangerines (T : ℕ) : ℕ := T - 10
def condition1 (T : ℕ) : Prop := remaining_tangerines T = remaining_oranges + 4

-- Theorem to prove the number of tangerines in the bag
theorem tangerine_count (T : ℕ) (h : condition1 T) : T = 17 :=
by
  sorry

end tangerine_count_l263_263086


namespace combine_heaps_l263_263504

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l263_263504


namespace recurring_decimal_to_rational_l263_263141

theorem recurring_decimal_to_rational : 
  (0.125125125 : ℝ) = 125 / 999 :=
sorry

end recurring_decimal_to_rational_l263_263141


namespace sum_of_possible_values_l263_263875

theorem sum_of_possible_values (M : ℝ) (h : M * (M + 4) = 12) : M + (if M = -6 then 2 else -6) = -4 :=
by
  sorry

end sum_of_possible_values_l263_263875


namespace smallest_integer_representable_l263_263268

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l263_263268


namespace incorrect_statement_among_options_l263_263291

/- Definitions and Conditions -/
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 1) + (n * (n - 1) / 2) * d

/- Conditions given in the problem -/
axiom S_6_gt_S_7 : S 6 > S 7
axiom S_7_gt_S_5 : S 7 > S 5

/- Incorrect statement to be proved -/
theorem incorrect_statement_among_options :
  ¬ (∀ n, S n ≤ S 11) := sorry

end incorrect_statement_among_options_l263_263291


namespace factor_expression_l263_263316

theorem factor_expression (x : ℚ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := sorry

end factor_expression_l263_263316


namespace x_ge_y_l263_263813

variable (a : ℝ)

def x : ℝ := 2 * a * (a + 3)
def y : ℝ := (a - 3) * (a + 3)

theorem x_ge_y : x a ≥ y a := 
by 
  sorry

end x_ge_y_l263_263813


namespace Cally_colored_shirts_l263_263446

theorem Cally_colored_shirts (C : ℕ) (hcally : 10 + 7 + 6 = 23) (hdanny : 6 + 8 + 10 + 6 = 30) (htotal : 23 + 30 + C = 58) : 
  C = 5 := 
by
  sorry

end Cally_colored_shirts_l263_263446


namespace xy_sum_eq_16_l263_263654

theorem xy_sum_eq_16 (x y : ℕ) (h1: x > 0) (h2: y > 0) (h3: x < 20) (h4: y < 20) (h5: x + y + x * y = 76) : x + y = 16 :=
  sorry

end xy_sum_eq_16_l263_263654


namespace ratio_of_ducks_to_total_goats_and_chickens_l263_263390

theorem ratio_of_ducks_to_total_goats_and_chickens 
    (goats chickens ducks pigs : ℕ) 
    (h1 : goats = 66)
    (h2 : chickens = 2 * goats)
    (h3 : pigs = ducks / 3)
    (h4 : goats = pigs + 33) :
    (ducks : ℚ) / (goats + chickens : ℚ) = 1 / 2 := 
by
  sorry

end ratio_of_ducks_to_total_goats_and_chickens_l263_263390


namespace People_Distribution_l263_263410

theorem People_Distribution 
  (total_people : ℕ) 
  (total_buses : ℕ) 
  (equal_distribution : ℕ) 
  (h1 : total_people = 219) 
  (h2 : total_buses = 3) 
  (h3 : equal_distribution = total_people / total_buses) : 
  equal_distribution = 73 :=
by 
  intros 
  sorry

end People_Distribution_l263_263410


namespace system_of_equations_solution_l263_263240

theorem system_of_equations_solution
  (a b c d e f g : ℝ)
  (x y z : ℝ)
  (h1 : a * x = b * y)
  (h2 : b * y = c * z)
  (h3 : d * x + e * y + f * z = g) :
  (x = g * b * c / (d * b * c + e * a * c + f * a * b)) ∧
  (y = g * a * c / (d * b * c + e * a * c + f * a * b)) ∧
  (z = g * a * b / (d * b * c + e * a * c + f * a * b)) :=
by
  sorry

end system_of_equations_solution_l263_263240


namespace circumscribed_sphere_surface_area_l263_263476

noncomputable def surface_area_of_circumscribed_sphere_from_volume (V : ℝ) : ℝ :=
  let s := V^(1/3 : ℝ)
  let d := s * Real.sqrt 3
  4 * Real.pi * (d / 2) ^ 2

theorem circumscribed_sphere_surface_area (V : ℝ) (h : V = 27) : surface_area_of_circumscribed_sphere_from_volume V = 27 * Real.pi :=
by
  rw [h]
  unfold surface_area_of_circumscribed_sphere_from_volume
  sorry

end circumscribed_sphere_surface_area_l263_263476


namespace johnny_tables_l263_263971

theorem johnny_tables :
  ∀ (T : ℕ),
  (∀ (T : ℕ), 4 * T + 5 * T = 45) →
  T = 5 :=
  sorry

end johnny_tables_l263_263971


namespace find_range_of_a_l263_263362

theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ∧ 
  ¬ ((∀ x : ℝ, x^2 - 2 * x > a) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)) → 
  a ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∪ Set.Ici (1:ℝ) :=
sorry

end find_range_of_a_l263_263362


namespace simplify_expression_l263_263541

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l263_263541


namespace midpoint_coordinates_l263_263995

theorem midpoint_coordinates :
  let A := (7, 8)
  let B := (1, 2)
  let midpoint (p1 p2 : ℕ × ℕ) : ℕ × ℕ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint A B = (4, 5) :=
by
  sorry

end midpoint_coordinates_l263_263995


namespace smallest_base10_integer_l263_263273

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l263_263273


namespace inequality_positive_reals_l263_263863

theorem inequality_positive_reals (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2 / 2) :=
sorry

end inequality_positive_reals_l263_263863


namespace constant_term_zero_l263_263400

theorem constant_term_zero (h1 : x^2 + x = 0)
                          (h2 : 2*x^2 - x - 12 = 0)
                          (h3 : 2*(x^2 - 1) = 3*(x - 1))
                          (h4 : 2*(x^2 + 1) = x + 4) :
                          (∃ (c : ℤ), c = 0 ∧ (c = 0 ∨ c = -12 ∨ c = 1 ∨ c = -2) → c = 0) :=
sorry

end constant_term_zero_l263_263400


namespace rectangular_prism_height_eq_17_l263_263135

-- Defining the lengths of the edges of the cubes and rectangular prism
def side_length_cube1 := 10
def edges_cube := 12
def length_rect_prism := 8
def width_rect_prism := 5

-- The total length of the wire used for each shape must be equal
def wire_length_cube1 := edges_cube * side_length_cube1
def wire_length_rect_prism (h : ℕ) := 4 * length_rect_prism + 4 * width_rect_prism + 4 * h

theorem rectangular_prism_height_eq_17 (h : ℕ) :
  wire_length_cube1 = wire_length_rect_prism h → h = 17 := 
by
  -- The proof goes here
  sorry

end rectangular_prism_height_eq_17_l263_263135


namespace sum_of_reciprocals_of_shifted_roots_l263_263686

theorem sum_of_reciprocals_of_shifted_roots (p q r : ℝ)
  (h1 : p^3 - 2 * p^2 - p + 3 = 0)
  (h2 : q^3 - 2 * q^2 - q + 3 = 0)
  (h3 : r^3 - 2 * r^2 - r + 3 = 0) :
  (1 / (p - 2)) + (1 / (q - 2)) + (1 / (r - 2)) = -3 :=
by
  sorry

end sum_of_reciprocals_of_shifted_roots_l263_263686


namespace intersection_correct_l263_263201

def setA : Set ℝ := { x | x - 1 ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x ≤ 0 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_correct : (setA ∩ setB) = expected_intersection :=
sorry

end intersection_correct_l263_263201


namespace complex_imaginary_axis_l263_263249

theorem complex_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) ↔ (a = 0 ∨ a = 2) := 
by
  sorry

end complex_imaginary_axis_l263_263249


namespace min_value_343_l263_263227

noncomputable def min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c)

theorem min_value_343 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ha hb hc = 343 :=
sorry

end min_value_343_l263_263227


namespace three_digit_number_l263_263462

theorem three_digit_number (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9) 
  (h : 100 * a + 10 * b + c = 3 * (10 * (a + b) + c)) : 100 * a + 10 * b + c = 135 :=
  sorry

end three_digit_number_l263_263462


namespace total_groups_l263_263026

-- Define the problem conditions
def boys : ℕ := 9
def girls : ℕ := 12

-- Calculate the required combinations
def C (n k: ℕ) : ℕ := n.choose k
def groups_with_two_boys_one_girl : ℕ := C boys 2 * C girls 1
def groups_with_two_girls_one_boy : ℕ := C girls 2 * C boys 1

-- Statement of the theorem to prove
theorem total_groups : groups_with_two_boys_one_girl + groups_with_two_girls_one_boy = 1026 := 
by sorry

end total_groups_l263_263026


namespace daily_production_l263_263154

theorem daily_production (x : ℕ) (hx1 : 216 / x > 4)
  (hx2 : 3 * x + (x + 8) * ((216 / x) - 4) = 232) : 
  x = 24 := by
sorry

end daily_production_l263_263154


namespace angle_triple_supplement_l263_263126

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263126


namespace average_percentage_of_popped_kernels_l263_263526

theorem average_percentage_of_popped_kernels (k1 k2 k3 p1 p2 p3 : ℕ) (h1 : k1 = 75) (h2 : k2 = 50) (h3 : k3 = 100)
    (h1_pop : p1 = 60) (h2_pop : p2 = 42) (h3_pop : p3 = 82) :
    ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) * 100 = 82 :=
by
  -- The proportion for each bag
  have prop1 : p1 / (k1 : ℝ) = 60 / 75 := by rw [h1, h1_pop]
  have prop2 : p2 / (k2 : ℝ) = 42 / 50 := by rw [h2, h2_pop]
  have prop3 : p3 / (k3 : ℝ) = 82 / 100 := by rw [h3, h3_pop]
  -- Sum the proportions
  have total_props : (p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) = 0.8 + 0.84 + 0.82 := by
    rw [prop1, prop2, prop3]
  -- Calculating the average proportion
  have avg_prop : ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) = 0.82 := by
    rw [total_props]
  -- Finally multiply the average by 100 to get the percentage
  have avg_percentage : ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) * 100 = 82 := by
    rw [avg_prop]
    norm_num
  exact avg_percentage

end average_percentage_of_popped_kernels_l263_263526


namespace ratio_of_hair_lengths_l263_263974

theorem ratio_of_hair_lengths 
  (logan_hair : ℕ)
  (emily_hair : ℕ)
  (kate_hair : ℕ)
  (h1 : logan_hair = 20)
  (h2 : emily_hair = logan_hair + 6)
  (h3 : kate_hair = 7)
  : kate_hair / emily_hair = 7 / 26 :=
by sorry

end ratio_of_hair_lengths_l263_263974


namespace diagonal_square_grid_size_l263_263914

theorem diagonal_square_grid_size (n : ℕ) (h : 2 * n - 1 = 2017) : n = 1009 :=
by
  sorry

end diagonal_square_grid_size_l263_263914


namespace parallel_lines_l263_263382

theorem parallel_lines (m : ℝ) :
    (∀ x y : ℝ, x + (m+1) * y - 1 = 0 → mx + 2 * y - 1 = 0 → (m = 1 → False)) → m = -2 :=
by
  sorry

end parallel_lines_l263_263382


namespace integer_roots_abs_sum_l263_263801

theorem integer_roots_abs_sum (p q r n : ℤ) :
  (∃ n : ℤ, (∀ x : ℤ, x^3 - 2023 * x + n = 0) ∧ p + q + r = 0 ∧ p * q + q * r + r * p = -2023) →
  |p| + |q| + |r| = 102 :=
by
  sorry

end integer_roots_abs_sum_l263_263801


namespace number_equation_l263_263302

-- Lean statement equivalent to the mathematical problem
theorem number_equation (x : ℝ) (h : 5 * x - 2 * x = 10) : 5 * x - 2 * x = 10 :=
by exact h

end number_equation_l263_263302


namespace peg_stickers_total_l263_263697

def stickers_in_red_folder : ℕ := 10 * 3
def stickers_in_green_folder : ℕ := 10 * 2
def stickers_in_blue_folder : ℕ := 10 * 1

def total_stickers : ℕ := stickers_in_red_folder + stickers_in_green_folder + stickers_in_blue_folder

theorem peg_stickers_total : total_stickers = 60 := by
  sorry

end peg_stickers_total_l263_263697


namespace find_X_l263_263461

theorem find_X : ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ X = 0.3 :=
by
  sorry

end find_X_l263_263461


namespace triple_supplementary_angle_l263_263094

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l263_263094


namespace sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l263_263489

-- Definitions based on conditions
def standard_flavor_to_water_ratio := 1 / 30
def standard_flavor_to_corn_syrup_ratio := 1 / 12
def sport_water_amount := 60
def sport_corn_syrup_amount := 4
def sport_flavor_to_water_ratio := 1 / 60
def sport_flavor_amount := 1 -- derived from sport_water_amount * sport_flavor_to_water_ratio

-- The main theorem to prove
theorem sport_flavoring_to_corn_syrup_ratio_is_three_times_standard :
  1 / 4 = 3 * (1 / 12) :=
by
  sorry

end sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l263_263489


namespace factor_difference_of_squares_example_l263_263132

theorem factor_difference_of_squares_example :
    (m : ℝ) → (m ^ 2 - 4 = (m + 2) * (m - 2)) :=
by
    intro m
    sorry

end factor_difference_of_squares_example_l263_263132


namespace range_of_a_l263_263659

noncomputable section

open Real

def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - (a + 1 / 2) * x^2 + (a^2 + a) * x - (1 / 2) * a^2 + 1 / 2

theorem range_of_a (a : ℝ) (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) :
  -7 / 2 < a ∧ a < -1 :=
by
  sorry

end range_of_a_l263_263659


namespace termite_ridden_not_collapsing_fraction_l263_263696

theorem termite_ridden_not_collapsing_fraction (h1 : (5 : ℚ) / 8) (h2 : (11 : ℚ) / 16) :
  (5 : ℚ) / 8 - ((5 : ℚ) / 8) * ((11 : ℚ) / 16) = 25 / 128 :=
by
  sorry

end termite_ridden_not_collapsing_fraction_l263_263696


namespace rate_per_meter_for_fencing_l263_263719

theorem rate_per_meter_for_fencing
  (w : ℕ) (length : ℕ) (perimeter : ℕ) (cost : ℕ)
  (h1 : length = w + 10)
  (h2 : perimeter = 2 * (length + w))
  (h3 : perimeter = 340)
  (h4 : cost = 2210) : (cost / perimeter : ℝ) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l263_263719


namespace g_at_5_l263_263690

def g (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 28 * x^2 - 20 * x - 80

theorem g_at_5 : g 5 = -5 := 
  by 
  -- Proof goes here
  sorry

end g_at_5_l263_263690


namespace complement_of_A_in_I_l263_263480

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6, 7}
def C_I_A : Set ℕ := {1, 3, 5}

theorem complement_of_A_in_I :
  (I \ A) = C_I_A := by
  sorry

end complement_of_A_in_I_l263_263480


namespace bob_second_week_hours_l263_263166

theorem bob_second_week_hours (total_earnings : ℕ) (total_hours_first_week : ℕ) (regular_hours_pay : ℕ) 
  (overtime_hours_pay : ℕ) (regular_hours_max : ℕ) (total_hours_overtime_first_week : ℕ) 
  (earnings_first_week : ℕ) (earnings_second_week : ℕ) : 
  total_earnings = 472 →
  total_hours_first_week = 44 →
  regular_hours_pay = 5 →
  overtime_hours_pay = 6 →
  regular_hours_max = 40 →
  total_hours_overtime_first_week = total_hours_first_week - regular_hours_max →
  earnings_first_week = regular_hours_max * regular_hours_pay + 
                          total_hours_overtime_first_week * overtime_hours_pay →
  earnings_second_week = total_earnings - earnings_first_week → 
  ∃ h, earnings_second_week = h * regular_hours_pay ∨ 
  earnings_second_week = (regular_hours_max * regular_hours_pay + (h - regular_hours_max) * overtime_hours_pay) ∧ 
  h = 48 :=
by 
  intros 
  sorry 

end bob_second_week_hours_l263_263166


namespace pizza_topping_slices_l263_263601

theorem pizza_topping_slices 
  (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ)
  (pepperoni_slices_has_at_least_one_topping : pepperoni_slices = 8)
  (mushroom_slices_has_at_least_one_topping : mushroom_slices = 12)
  (olive_slices_has_at_least_one_topping : olive_slices = 14)
  (total_slices_has_one_topping : total_slices = 16)
  (slices_with_at_least_one_topping : 8 + 12 + 14 - 2 * x = 16) :
  x = 9 :=
by
  sorry

end pizza_topping_slices_l263_263601


namespace area_of_dodecagon_l263_263764

theorem area_of_dodecagon (r : ℝ) : 
  ∃ A : ℝ, (∃ n : ℕ, n = 12) ∧ (A = 3 * r^2) := 
by
  sorry

end area_of_dodecagon_l263_263764


namespace sum_of_numbers_l263_263577

-- Define the given conditions.
def S : ℕ := 30
def F : ℕ := 2 * S
def T : ℕ := F / 3

-- State the proof problem.
theorem sum_of_numbers : F + S + T = 110 :=
by
  -- Assume the proof here.
  sorry

end sum_of_numbers_l263_263577


namespace dolls_total_correct_l263_263037

def Jazmin_dolls : Nat := 1209
def Geraldine_dolls : Nat := 2186
def total_dolls : Nat := Jazmin_dolls + Geraldine_dolls

theorem dolls_total_correct : total_dolls = 3395 := by
  sorry

end dolls_total_correct_l263_263037


namespace checkerboard_black_squares_count_l263_263169

namespace Checkerboard

def is_black (n : ℕ) : Bool :=
  -- Define the alternating pattern of the checkerboard
  (n % 2 = 0)

def black_square_count (n : ℕ) : ℕ :=
  -- Calculate the number of black squares in a checkerboard of size n x n
  if n % 2 = 0 then n * n / 2 else n * n / 2 + n / 2 + 1

def additional_black_squares (n : ℕ) : ℕ :=
  -- Calculate the additional black squares due to modification of every 33rd square in every third row
  ((n - 1) / 3 + 1)

def total_black_squares (n : ℕ) : ℕ :=
  -- Calculate the total black squares considering the modified hypothesis
  black_square_count n + additional_black_squares n

theorem checkerboard_black_squares_count : total_black_squares 33 = 555 := 
  by sorry

end Checkerboard

end checkerboard_black_squares_count_l263_263169


namespace S8_value_l263_263976

theorem S8_value (x : ℝ) (h : x + 1/x = 4) (S : ℕ → ℝ) (S_def : ∀ m, S m = x^m + 1/x^m) :
  S 8 = 37634 :=
sorry

end S8_value_l263_263976


namespace fraction_of_number_l263_263903

variable (N : ℝ) (F : ℝ)

theorem fraction_of_number (h1 : 0.5 * N = F * N + 2) (h2 : N = 8.0) : F = 0.25 := by
  sorry

end fraction_of_number_l263_263903


namespace a_b_total_money_l263_263137

variable (A B : ℝ)

theorem a_b_total_money (h1 : (4 / 15) * A = (2 / 5) * 484) (h2 : B = 484) : A + B = 1210 := by
  sorry

end a_b_total_money_l263_263137


namespace value_of_f_neg_a_l263_263634

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -2 := 
by 
  sorry

end value_of_f_neg_a_l263_263634


namespace Eric_return_time_l263_263174

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l263_263174


namespace collinear_points_count_l263_263434

-- Definitions for the problem conditions
def vertices_count := 8
def midpoints_count := 12
def face_centers_count := 6
def cube_center_count := 1
def total_points_count := vertices_count + midpoints_count + face_centers_count + cube_center_count

-- Lean statement to express the proof problem
theorem collinear_points_count :
  (total_points_count = 27) →
  (vertices_count = 8) →
  (midpoints_count = 12) →
  (face_centers_count = 6) →
  (cube_center_count = 1) →
  ∃ n, n = 49 :=
by
  intros
  existsi 49
  sorry

end collinear_points_count_l263_263434


namespace problem_solution_l263_263940

theorem problem_solution
  (m : ℝ) (n : ℝ)
  (h1 : m = 1 / (Real.sqrt 3 + Real.sqrt 2))
  (h2 : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) :
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 :=
by sorry

end problem_solution_l263_263940


namespace quadratic_equation_real_roots_l263_263662

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l263_263662


namespace percentage_increase_of_y_over_x_l263_263285

variable (x y : ℝ) (h : x > 0 ∧ y > 0) 

theorem percentage_increase_of_y_over_x
  (h_ratio : (x / 8) = (y / 7)) :
  ((y - x) / x) * 100 = 12.5 := 
sorry

end percentage_increase_of_y_over_x_l263_263285


namespace statement_A_statement_B_statement_C_statement_D_l263_263642

variable (a b : ℝ)

-- Given conditions
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom condition : a + 2 * b = 2 * a * b

-- Prove the statements
theorem statement_A : a + 2 * b ≥ 4 := sorry
theorem statement_B : ¬ (a + b ≥ 4) := sorry
theorem statement_C : ¬ (a * b ≤ 2) := sorry
theorem statement_D : a^2 + 4 * b^2 ≥ 8 := sorry

end statement_A_statement_B_statement_C_statement_D_l263_263642


namespace algebraic_expression_value_l263_263018

-- Define the given condition
def condition (a b : ℝ) : Prop := a + b - 2 = 0

-- State the theorem to prove the algebraic expression value
theorem algebraic_expression_value (a b : ℝ) (h : condition a b) : a^2 - b^2 + 4 * b = 4 := by
  sorry

end algebraic_expression_value_l263_263018


namespace gcd_1337_382_l263_263559

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end gcd_1337_382_l263_263559


namespace true_proposition_l263_263817

variable (p q : Prop)
variable (hp : p = true)
variable (hq : q = false)

theorem true_proposition : (¬p ∨ ¬q) = true := by
  sorry

end true_proposition_l263_263817


namespace hurricane_damage_in_GBP_l263_263761

def damage_in_AUD : ℤ := 45000000
def conversion_rate : ℚ := 1 / 2 -- 1 AUD = 1/2 GBP

theorem hurricane_damage_in_GBP : 
  (damage_in_AUD : ℚ) * conversion_rate = 22500000 := 
by
  sorry

end hurricane_damage_in_GBP_l263_263761


namespace simplify_power_of_product_l263_263372

theorem simplify_power_of_product (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 :=
by
  -- hint: begin proof here
  sorry

end simplify_power_of_product_l263_263372


namespace part1_part2_l263_263549

-- Definitions
def p (t : ℝ) := ∀ x : ℝ, x^2 + 2 * x + 2 * t - 4 ≠ 0
def q (t : ℝ) := (4 - t > 0) ∧ (t - 2 > 0)

-- Theorem statements
theorem part1 (t : ℝ) (hp : p t) : t > 5 / 2 := sorry

theorem part2 (t : ℝ) (h : p t ∨ q t) (h_and : ¬ (p t ∧ q t)) : (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) := sorry

end part1_part2_l263_263549


namespace simplify_expression_l263_263238

-- Define the given expression
def expr : ℚ := (5^6 + 5^3) / (5^5 - 5^2)

-- State the proof problem
theorem simplify_expression : expr = 315 / 62 := 
by sorry

end simplify_expression_l263_263238


namespace problem1_problem2_l263_263471

def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We add this case for Lean to handle zero index
  else if n = 1 then 2
  else 2^(n-1)

def S (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) a

theorem problem1 (n : ℕ) :
  a n = 
  if n = 1 then 2
  else 2^(n-1) :=
sorry

theorem problem2 (n : ℕ) :
  S n = 2^n :=
sorry

end problem1_problem2_l263_263471


namespace split_piles_equiv_single_stone_heaps_l263_263502

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l263_263502


namespace subtracted_number_divisible_by_5_l263_263405

theorem subtracted_number_divisible_by_5 : ∃ k : ℕ, 9671 - 1 = 5 * k :=
by
  sorry

end subtracted_number_divisible_by_5_l263_263405


namespace plastic_bag_estimation_l263_263087

theorem plastic_bag_estimation (a b c d e f : ℕ) (class_size : ℕ) (h1 : a = 33) 
  (h2 : b = 25) (h3 : c = 28) (h4 : d = 26) (h5 : e = 25) (h6 : f = 31) (h_class_size : class_size = 45) :
  let count := a + b + c + d + e + f
  let average := count / 6
  average * class_size = 1260 := by
{ 
  sorry 
}

end plastic_bag_estimation_l263_263087


namespace johns_initial_playtime_l263_263970

theorem johns_initial_playtime :
  ∃ (x : ℝ), (14 * x = 0.40 * (14 * x + 84)) → x = 4 :=
by
  sorry

end johns_initial_playtime_l263_263970


namespace christel_gave_andrena_l263_263792

theorem christel_gave_andrena (d m c a: ℕ) (h1: d = 20 - 2) (h2: c = 24) 
  (h3: a = c + 2) (h4: a = d + 3) : (24 - c = 5) :=
by { sorry }

end christel_gave_andrena_l263_263792


namespace sum_of_powers_of_two_l263_263336

theorem sum_of_powers_of_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 511) : 
  ∃ (S : Finset ℕ), S ⊆ ({2^8, 2^7, 2^6, 2^5, 2^4, 2^3, 2^2, 2^1, 2^0} : Finset ℕ) ∧ 
  S.sum id = n :=
by
  sorry

end sum_of_powers_of_two_l263_263336


namespace geometric_progression_product_l263_263966

theorem geometric_progression_product (n : ℕ) (S R : ℝ) (hS : S > 0) (hR : R > 0)
  (h_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ S = a * (q^n - 1) / (q - 1))
  (h_reciprocal_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ R = (1 - q^n) / (a * q^(n-1) * (q - 1))) :
  ∃ P : ℝ, P = (S / R)^(n / 2) := sorry

end geometric_progression_product_l263_263966


namespace solve_system_equations_l263_263882

theorem solve_system_equations :
  ∃ x y : ℚ, (5 * x * (y + 6) = 0 ∧ 2 * x + 3 * y = 1) ∧
  (x = 0 ∧ y = 1 / 3 ∨ x = 19 / 2 ∧ y = -6) :=
by
  sorry

end solve_system_equations_l263_263882


namespace quadratic_real_roots_l263_263665

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l263_263665


namespace max_profit_l263_263757

-- Definition of the conditions
def production_requirements (tonAprodA tonAprodB tonBprodA tonBprodB: ℕ )
  := tonAprodA = 3 ∧ tonAprodB = 1 ∧ tonBprodA = 2 ∧ tonBprodB = 3

def profit_per_ton ( profitA profitB: ℕ )
  := profitA = 50000 ∧ profitB = 30000

def raw_material_limits ( rawA rawB: ℕ)
  := rawA = 13 ∧ rawB = 18

theorem max_profit 
  (production_requirements: production_requirements 3 1 2 3)
  (profit_per_ton: profit_per_ton 50000 30000)
  (raw_material_limits: raw_material_limits 13 18)
: ∃ (maxProfit: ℕ), maxProfit = 270000 := 
by 
  sorry

end max_profit_l263_263757


namespace cheapest_third_company_l263_263885

theorem cheapest_third_company (x : ℕ) :
  (120 + 18 * x ≥ 150 + 15 * x) ∧ (220 + 13 * x ≥ 150 + 15 * x) → 36 ≤ x :=
by
  intro h
  cases h with
  | intro h1 h2 =>
    sorry

end cheapest_third_company_l263_263885


namespace not_beautiful_739_and_741_l263_263379

-- Define the function g and its properties
variable (g : ℤ → ℤ)

-- Condition: g(x) ≠ x
axiom g_neq_x (x : ℤ) : g x ≠ x

-- Definition of "beautiful"
def beautiful (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

-- The theorem to prove
theorem not_beautiful_739_and_741 :
  ¬ (beautiful g 739 ∧ beautiful g 741) :=
sorry

end not_beautiful_739_and_741_l263_263379


namespace trigonometric_inequality_l263_263004

-- Let \( f(x) \) be defined as \( cos \, x \)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Given a, b, c are the sides of triangle ∆ABC opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}

-- Condition: \( 3a^2 + 3b^2 - c^2 = 4ab \)
variable (h : 3 * a^2 + 3 * b^2 - c^2 = 4 * a * b)

-- Goal: Prove that \( f(\cos A) \leq f(\sin B) \)
theorem trigonometric_inequality (h1 : A + B + C = π) (h2 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) : 
  f (Real.cos A) ≤ f (Real.sin B) :=
by
  sorry

end trigonometric_inequality_l263_263004


namespace geometric_progression_value_l263_263331

variable (a : ℕ → ℕ)
variable (r : ℕ)
variable (h_geo : ∀ n, a (n + 1) = a n * r)

theorem geometric_progression_value (h2 : a 2 = 2) (h6 : a 6 = 162) : a 10 = 13122 :=
by
  sorry

end geometric_progression_value_l263_263331


namespace ratio_of_men_to_women_l263_263907

theorem ratio_of_men_to_women (C W M : ℕ) 
  (hC : C = 30) 
  (hW : W = 3 * C) 
  (hTotal : M + W + C = 300) : 
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l263_263907


namespace train_speed_l263_263138

/-
Problem Statement:
Prove that the speed of a train is 26.67 meters per second given:
  1. The length of the train is 320 meters.
  2. The time taken to cross the telegraph post is 12 seconds.
-/

theorem train_speed (distance time : ℝ) (h1 : distance = 320) (h2 : time = 12) :
  (distance / time) = 26.67 :=
by
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l263_263138


namespace correct_factoring_example_l263_263251

-- Define each option as hypotheses
def optionA (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2
def optionB (a b : ℝ) : Prop := 2 * a ^ 2 - a * b - a = a * (2 * a - b - 1)
def optionC (a b : ℝ) : Prop := 8 * a ^ 5 * b ^ 2 = 4 * a ^ 3 * b * 2 * a ^ 2 * b
def optionD (a : ℝ) : Prop := a ^ 2 - 4 * a + 3 = (a - 1) * (a - 3)

-- The goal is to prove that optionD is the correct example of factoring
theorem correct_factoring_example (a b : ℝ) : optionD a ↔ (∀ a b, ¬ optionA a b) ∧ (∀ a b, ¬ optionB a b) ∧ (∀ a b, ¬ optionC a b) :=
by
  sorry

end correct_factoring_example_l263_263251


namespace find_length_l263_263669

-- Define the perimeter and breadth as constants
def P : ℕ := 950
def B : ℕ := 100

-- State the theorem
theorem find_length (L : ℕ) (H : 2 * (L + B) = P) : L = 375 :=
by sorry

end find_length_l263_263669


namespace find_y_given_x_eq_neg6_l263_263197

theorem find_y_given_x_eq_neg6 :
  ∀ (y : ℤ), (∃ (x : ℤ), x = -6 ∧ x^2 - x + 6 = y - 6) → y = 54 :=
by
  intros y h
  obtain ⟨x, hx1, hx2⟩ := h
  rw [hx1] at hx2
  simp at hx2
  linarith

end find_y_given_x_eq_neg6_l263_263197


namespace sad_outcome_probability_l263_263988

theorem sad_outcome_probability : 
  let total_outcomes := 3^6 in
  let sad_outcomes := 156 in
  (sad_outcomes / total_outcomes : ℚ) = 0.214 := 
by
  /-
  Given conditions:
  - The company consists of three boys and three girls.
  - Each boy loves one of the three girls.
  - Each girl loves one of the boys.
  - In a sad outcome, nobody is loved by the one they love.
  - Using the properties of derangements and additional condition counts.
  - Total number of sad outcomes = 156.
  - Total possible outcomes = 3^6 = 729.
  - Final probability of sad outcome = 156 / 729 = 0.214.
  -/
  sorry

end sad_outcome_probability_l263_263988


namespace cost_price_of_watch_l263_263595

theorem cost_price_of_watch (CP : ℝ) (h_loss : 0.54 * CP = SP_loss)
                            (h_gain : 1.04 * CP = SP_gain)
                            (h_diff : SP_gain - SP_loss = 140) :
                            CP = 280 :=
by {
    sorry
}

end cost_price_of_watch_l263_263595


namespace team_A_wins_2_1_team_B_wins_l263_263730

theorem team_A_wins_2_1 (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (2 * p_a * p_b) * p_a = 0.288 := by
  sorry

theorem team_B_wins (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (p_b * p_b) + (2 * p_a * p_b * p_b) = 0.352 := by
  sorry

end team_A_wins_2_1_team_B_wins_l263_263730


namespace simplify_exponent_l263_263542

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l263_263542


namespace percentage_increase_after_decrease_l263_263723

variable (P : ℝ) (x : ℝ)

-- Conditions
def decreased_price : ℝ := 0.80 * P
def final_price_condition : Prop := 0.80 * P + (x / 100) * (0.80 * P) = 1.04 * P
def correct_answer : Prop := x = 30

-- The proof goal
theorem percentage_increase_after_decrease : final_price_condition P x → correct_answer x :=
by sorry

end percentage_increase_after_decrease_l263_263723


namespace return_trip_time_l263_263178

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l263_263178


namespace algebraic_expression_evaluation_l263_263633

-- Given condition and goal statement
theorem algebraic_expression_evaluation (a b : ℝ) (h : a - 2 * b + 3 = 0) : 5 + 2 * b - a = 8 :=
by sorry

end algebraic_expression_evaluation_l263_263633


namespace unique_root_a_b_values_l263_263008

theorem unique_root_a_b_values {a b : ℝ} (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = 1) : a = -2 ∧ b = 1 := by
  sorry

end unique_root_a_b_values_l263_263008


namespace ellipse_proof_l263_263945

noncomputable def ellipse_equation {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
    (line_slope : ℝ) (dist_from_center : ℝ) (chord_length : ℝ) (major_axis : ℝ) : Prop :=
  line_slope = 1/2 ∧
  dist_from_center = 1 ∧
  chord_length = (4/5) * major_axis ∧
  major_axis = 2 * a ∧
  36 = 5 * b^2 → 
  (a = 3 ∧ b = 2 ∧ (eq : ( ∀ x y : ℝ, (x^2) / 9 + (y^2) / 4 = 1 )))

theorem ellipse_proof : ellipse_equation (a := 3) (b := 2) (h1 := by norm_num) (h2 := by norm_num) (h3 := by linarith)
    1/2 1 ((4/5) * 6) 6 :=
by
  sorry

end ellipse_proof_l263_263945


namespace neg_i_pow_four_l263_263173

-- Define i as the imaginary unit satisfying i^2 = -1
def i : ℂ := Complex.I

-- The proof problem: Prove (-i)^4 = 1 given i^2 = -1
theorem neg_i_pow_four : (-i)^4 = 1 :=
by
  -- sorry is used to skip proof
  sorry

end neg_i_pow_four_l263_263173


namespace parallel_vectors_solution_l263_263949

noncomputable def vector_a : (ℝ × ℝ) := (1, 2)
noncomputable def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -4)

def vectors_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_solution (x : ℝ) (h : vectors_parallel vector_a (vector_b x)) : x = -2 :=
sorry

end parallel_vectors_solution_l263_263949


namespace integer_pairs_m_n_l263_263323

theorem integer_pairs_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (cond1 : ∃ k1 : ℕ, k1 * m = 3 * n ^ 2)
  (cond2 : ∃ k2 : ℕ, k2 ^ 2 = n ^ 2 + m) :
  ∃ a : ℕ, m = 3 * a ^ 2 ∧ n = a :=
by
  sorry

end integer_pairs_m_n_l263_263323


namespace train_speed_l263_263897

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) (h_train_length : train_length = 100) (h_bridge_length : bridge_length = 300) (h_crossing_time : crossing_time = 12) : 
  (train_length + bridge_length) / crossing_time = 33.33 := 
by 
  -- sorry allows us to skip the proof
  sorry

end train_speed_l263_263897


namespace power_eq_l263_263498

open Real

theorem power_eq {x : ℝ} (h : x^3 + 4 * x = 8) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end power_eq_l263_263498


namespace smallest_base10_integer_l263_263277

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l263_263277


namespace shaded_region_area_l263_263842

theorem shaded_region_area (r : ℝ) (π : ℝ) (h1 : r = 5) : 
  4 * ((1/2 * π * r * r) - (1/2 * r * r)) = 50 * π - 50 :=
by 
  sorry

end shaded_region_area_l263_263842


namespace inequality_solution_l263_263983

theorem inequality_solution (x : ℝ) :
  (-1 : ℝ) < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < (1 : ℝ) ↔
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by
  sorry

end inequality_solution_l263_263983


namespace number_of_foxes_l263_263522

-- Define the conditions as given in the problem
def num_cows : ℕ := 20
def num_sheep : ℕ := 20
def total_animals : ℕ := 100
def num_zebras (F : ℕ) := 3 * F

-- The theorem we want to prove based on the conditions
theorem number_of_foxes (F : ℕ) :
  num_cows + num_sheep + F + num_zebras F = total_animals → F = 15 :=
by
  sorry

end number_of_foxes_l263_263522


namespace triangle_angles_geometric_progression_l263_263833

-- Theorem: If the sides of a triangle whose angles form an arithmetic progression are in geometric progression, then all three angles are 60°.
theorem triangle_angles_geometric_progression (A B C : ℝ) (a b c : ℝ)
  (h_arith_progression : 2 * B = A + C)
  (h_sum_angles : A + B + C = 180)
  (h_geo_progression : (a / b) = (b / c))
  (h_b_angle : B = 60) :
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_angles_geometric_progression_l263_263833


namespace probability_of_neither_solving_l263_263300

def prob_solve_A : ℝ := 1 / 2
def prob_solve_B : ℝ := 1 / 3

def prob_not_solve_A : ℝ := 1 - prob_solve_A
def prob_not_solve_B : ℝ := 1 - prob_solve_B

def prob_neither_solve : ℝ := prob_not_solve_A * prob_not_solve_B

theorem probability_of_neither_solving (hA : prob_solve_A = 1 / 2) (hB : prob_solve_B = 1 / 3) 
  (indep : true) : prob_neither_solve = 1 / 3 :=
by
  sorry

end probability_of_neither_solving_l263_263300


namespace john_total_distance_l263_263680

theorem john_total_distance : 
  let daily_distance := 1700
  let days_run := 6
  daily_distance * days_run = 10200 :=
by
  sorry

end john_total_distance_l263_263680


namespace eleven_pow_four_l263_263134

theorem eleven_pow_four : 11 ^ 4 = 14641 := 
by sorry

end eleven_pow_four_l263_263134


namespace multiply_fractions_l263_263445

theorem multiply_fractions :
  (2 / 9) * (5 / 14) = 5 / 63 :=
by
  sorry

end multiply_fractions_l263_263445


namespace new_train_distance_l263_263762

-- Given conditions
def distance_older_train : ℝ := 200
def percent_more : ℝ := 0.20

-- Conclusion to prove
theorem new_train_distance : (distance_older_train * (1 + percent_more)) = 240 := by
  -- Placeholder to indicate that we are skipping the actual proof steps
  sorry

end new_train_distance_l263_263762


namespace angle_triple_supplement_l263_263103

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l263_263103


namespace geometric_series_q_and_S6_l263_263676

theorem geometric_series_q_and_S6 (a : ℕ → ℝ) (q : ℝ) (S_6 : ℝ) 
  (ha_pos : ∀ n, a n > 0)
  (ha2 : a 2 = 3)
  (ha4 : a 4 = 27) :
  q = 3 ∧ S_6 = 364 :=
by
  sorry

end geometric_series_q_and_S6_l263_263676


namespace largest_common_value_l263_263247

theorem largest_common_value (a : ℕ) (h1 : a % 4 = 3) (h2 : a % 9 = 5) (h3 : a < 600) :
  a = 599 :=
sorry

end largest_common_value_l263_263247


namespace pies_sold_l263_263311

theorem pies_sold (apple_slices : ℕ) (peach_slices : ℕ) (apple_customers : ℕ) (peach_customers : ℕ)
  (h1 : apple_slices = 8) (h2 : peach_slices = 6)
  (h3 : apple_customers = 56) (h4 : peach_customers = 48) : 
  (apple_customers / apple_slices + peach_customers / peach_slices) = 15 := 
by
  have h5 : apple_customers / apple_slices = 7 := by sorry
  have h6 : peach_customers / peach_slices = 8 := by sorry
  calc
    (apple_customers / apple_slices + peach_customers / peach_slices) = (7 + 8) : by
      rw [h5, h6]
    ... = 15 : by
      norm_num

end pies_sold_l263_263311


namespace general_term_of_sequence_l263_263637

-- Definition of arithmetic sequence with positive common difference
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℤ} {d : ℤ}
axiom positive_common_difference : d > 0
axiom cond1 : a 3 * a 4 = 117
axiom cond2 : a 2 + a 5 = 22

-- Target statement to prove
theorem general_term_of_sequence : is_arithmetic_sequence a d → a n = 4 * n - 3 :=
by sorry

end general_term_of_sequence_l263_263637


namespace problem1_problem2_l263_263598

-- Problem 1: Lean 4 Statement
theorem problem1 (n : ℕ) (hn : n > 0) : 20 ∣ (4 * 6^n + 5^(n + 1) - 9) :=
sorry

-- Problem 2: Lean 4 Statement
theorem problem2 : (3^100 % 7) = 4 :=
sorry

end problem1_problem2_l263_263598


namespace find_m_n_l263_263290

theorem find_m_n (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_gcd : m.gcd n = 1) (h_div : (m^3 + n^3) ∣ (m^2 + 20 * m * n + n^2)) :
  (m, n) ∈ [(1, 2), (2, 1), (2, 3), (3, 2), (1, 5), (5, 1)] :=
by
  sorry

end find_m_n_l263_263290


namespace fraction_simplification_l263_263546

theorem fraction_simplification : 
  (320 / 18) * (9 / 144) * (4 / 5) = 1 / 2 :=
by sorry

end fraction_simplification_l263_263546


namespace student_B_more_stable_l263_263397

-- Definitions as stated in the conditions
def student_A_variance : ℝ := 0.3
def student_B_variance : ℝ := 0.1

-- Theorem stating that student B has more stable performance than student A
theorem student_B_more_stable : student_B_variance < student_A_variance :=
by
  sorry

end student_B_more_stable_l263_263397


namespace statement_b_statement_e_l263_263622

-- Statement (B): ∀ x, if x^3 > 0 then x > 0.
theorem statement_b (x : ℝ) : x^3 > 0 → x > 0 := sorry

-- Statement (E): ∀ x, if x < 1 then x^3 < x.
theorem statement_e (x : ℝ) : x < 1 → x^3 < x := sorry

end statement_b_statement_e_l263_263622


namespace evaluate_f_g3_l263_263482

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 1
def g (x : ℝ) : ℝ := x + 3

theorem evaluate_f_g3 : f (g 3) = 97 := by
  sorry

end evaluate_f_g3_l263_263482


namespace subset_m_values_l263_263646

theorem subset_m_values
  {A B : Set ℝ}
  (hA : A = { x | x^2 + x - 6 = 0 })
  (hB : ∃ m, B = { x | m * x + 1 = 0 })
  (h_subset : ∀ {x}, x ∈ B → x ∈ A) :
  (∃ m, m = -1/2 ∨ m = 0 ∨ m = 1/3) :=
sorry

end subset_m_values_l263_263646


namespace club_members_problem_l263_263344

theorem club_members_problem 
    (T : ℕ) (C : ℕ) (D : ℕ) (B : ℕ) 
    (h_T : T = 85) (h_C : C = 45) (h_D : D = 32) (h_B : B = 18) :
    let Cₒ := C - B
    let Dₒ := D - B
    let N := T - (Cₒ + Dₒ + B)
    N = 26 :=
by
  sorry

end club_members_problem_l263_263344


namespace ellipse_iff_constant_sum_l263_263345

-- Let F_1 and F_2 be two fixed points in the plane.
variables (F1 F2 : Point)
-- Let d be a constant.
variable (d : ℝ)

-- A point M in a plane
variable (M : Point)

-- Define the distance function between two points.
def dist (P Q : Point) : ℝ := sorry

-- Definition: M is on an ellipse with foci F1 and F2
def on_ellipse (M F1 F2 : Point) (d : ℝ) : Prop :=
  dist M F1 + dist M F2 = d

-- Proof that shows the two parts of the statement
theorem ellipse_iff_constant_sum :
  (∀ M, on_ellipse M F1 F2 d) ↔ (∀ M, dist M F1 + dist M F2 = d) ∧ d > dist F1 F2 :=
sorry

end ellipse_iff_constant_sum_l263_263345


namespace Charlie_age_when_Jenny_twice_as_Bobby_l263_263218

theorem Charlie_age_when_Jenny_twice_as_Bobby (B C J : ℕ) 
  (h₁ : J = C + 5)
  (h₂ : C = B + 3)
  (h₃ : J = 2 * B) : 
  C = 11 :=
by
  sorry

end Charlie_age_when_Jenny_twice_as_Bobby_l263_263218


namespace smallest_A_is_144_l263_263084

noncomputable def smallest_A (B : ℕ) := B * 28 + 4

theorem smallest_A_is_144 :
  ∃ (B : ℕ), smallest_A B = 144 ∧ ∀ (B' : ℕ), B' * 28 + 4 < 144 → false :=
by
  sorry

end smallest_A_is_144_l263_263084


namespace probability_same_outcomes_l263_263911

-- Let us define the event space for a fair coin
inductive CoinTossOutcome
| H : CoinTossOutcome
| T : CoinTossOutcome

open CoinTossOutcome

-- Definition of an event where the outcomes are the same (HHH or TTT)
def same_outcomes (t1 t2 t3 : CoinTossOutcome) : Prop :=
  (t1 = H ∧ t2 = H ∧ t3 = H) ∨ (t1 = T ∧ t2 = T ∧ t3 = T)

-- Number of all possible outcomes for three coin tosses
def total_outcomes : ℕ := 2 ^ 3

-- Number of favorable outcomes where all outcomes are the same
def favorable_outcomes : ℕ := 2

-- Calculation of probability
def prob_same_outcomes : ℚ := favorable_outcomes / total_outcomes

-- The statement to be proved in Lean 4
theorem probability_same_outcomes : prob_same_outcomes = 1 / 4 := 
by sorry

end probability_same_outcomes_l263_263911


namespace angle_triple_supplement_l263_263105

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l263_263105


namespace div_by_3_implies_one_div_by_3_l263_263583

theorem div_by_3_implies_one_div_by_3 (a b : ℕ) (h_ab : 3 ∣ (a * b)) (h_na : ¬ 3 ∣ a) (h_nb : ¬ 3 ∣ b) : false :=
sorry

end div_by_3_implies_one_div_by_3_l263_263583


namespace slope_of_line_n_l263_263944

noncomputable def tan_double_angle (t : ℝ) : ℝ := (2 * t) / (1 - t^2)

theorem slope_of_line_n :
  let slope_m := 6
  let alpha := Real.arctan slope_m
  let slope_n := tan_double_angle slope_m
  slope_n = -12 / 35 :=
by
  sorry

end slope_of_line_n_l263_263944


namespace find_a_l263_263366

def M : Set ℝ := {-1, 0, 1}

def N (a : ℝ) : Set ℝ := {a, a^2}

theorem find_a (a : ℝ) : N a ⊆ M → a = -1 :=
by
  sorry

end find_a_l263_263366


namespace combined_weight_of_three_new_people_l263_263556

theorem combined_weight_of_three_new_people 
  (W : ℝ) 
  (h_avg_increase : (W + 80) / 20 = W / 20 + 4) 
  (h_replaced_weights : 60 + 75 + 85 = 220) : 
  220 + 80 = 300 :=
by
  sorry

end combined_weight_of_three_new_people_l263_263556


namespace min_value_of_a_l263_263643

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + 2 * a * x + 1 ≥ 0) → a ≥ -5 / 4 := 
sorry

end min_value_of_a_l263_263643


namespace sum_first_9000_terms_l263_263879

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
a * ((1 - r^n) / (1 - r))

theorem sum_first_9000_terms (a r : ℝ) (h1 : geom_sum a r 3000 = 1000) 
                              (h2 : geom_sum a r 6000 = 1900) : 
                              geom_sum a r 9000 = 2710 := 
by sorry

end sum_first_9000_terms_l263_263879


namespace smallest_integer_representation_l263_263274

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l263_263274


namespace rectangle_length_width_difference_l263_263328

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : x + y = 40)
  (h2 : x^2 + y^2 = 800) :
  x - y = 0 :=
sorry

end rectangle_length_width_difference_l263_263328


namespace total_number_of_boys_in_camp_l263_263671

theorem total_number_of_boys_in_camp (T : ℕ)
  (hA1 : ∃ (boysA : ℕ), boysA = 20 * T / 100)
  (hA2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 30 * boysA / 100 ∧ boysM = 40 * boysA / 100)
  (hB1 : ∃ (boysB : ℕ), boysB = 30 * T / 100)
  (hB2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 25 * boysB / 100 ∧ boysM = 35 * boysB / 100)
  (hC1 : ∃ (boysC : ℕ), boysC = 50 * T / 100)
  (hC2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 15 * boysC / 100 ∧ boysM = 45 * boysC / 100)
  (hA_no_SM : 77 = 70 * boysA / 100)
  (hB_no_SM : 72 = 60 * boysB / 100)
  (hC_no_SM : 98 = 60 * boysC / 100) :
  T = 535 :=
by
  sorry

end total_number_of_boys_in_camp_l263_263671


namespace angle_triple_supplementary_l263_263109

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l263_263109


namespace max_value_of_expr_l263_263468

theorem max_value_of_expr (x : ℝ) (h : x ≠ 0) : 
  (∀ y : ℝ, y = (x^2) / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) → y ≤ 1/8) :=
sorry

end max_value_of_expr_l263_263468


namespace lcm_factors_l263_263380

theorem lcm_factors (A B : ℕ) (hcf lcm_factor other_factor : ℕ) (hcf_is_10 : hcf = 10) (larger_is_150 : A = 150) (lcm_factor_is_15 : lcm_factor = 15) (lcm_def : A = hcf * other_factor * lcm_factor) 
  : other_factor = 1 :=
by
  sorry

end lcm_factors_l263_263380


namespace part_I_solution_part_II_solution_l263_263199

-- Defining f(x) given parameters a and b
def f (x a b : ℝ) := |x - a| + |x + b|

-- Part (I): Given a = 1 and b = 2, solve the inequality f(x) ≤ 5
theorem part_I_solution (x : ℝ) : 
  (f x 1 2) ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

-- Part (II): Given the minimum value of f(x) is 3, find min (a^2 / b + b^2 / a)
theorem part_II_solution (a b : ℝ) (h : 3 = |a| + |b|) (ha : a > 0) (hb : b > 0) : 
  (min (a^2 / b + b^2 / a)) = 3 := 
by
  sorry

end part_I_solution_part_II_solution_l263_263199


namespace feet_per_inch_of_model_l263_263551

theorem feet_per_inch_of_model 
  (height_tower : ℝ)
  (height_model : ℝ)
  (height_tower_eq : height_tower = 984)
  (height_model_eq : height_model = 6)
  : (height_tower / height_model) = 164 :=
by
  -- Assume the proof here
  sorry

end feet_per_inch_of_model_l263_263551


namespace divisible_by_condition_a_l263_263981

theorem divisible_by_condition_a (a b c k : ℤ) 
  (h : ∃ k : ℤ, a - b * c = (10 * c + 1) * k) : 
  ∃ k : ℤ, 10 * a + b = (10 * c + 1) * k :=
by
  sorry

end divisible_by_condition_a_l263_263981


namespace number_of_height_groups_l263_263074

theorem number_of_height_groups
  (max_height : ℕ) (min_height : ℕ) (class_width : ℕ)
  (h_max : max_height = 186)
  (h_min : min_height = 167)
  (h_class_width : class_width = 3) :
  (max_height - min_height + class_width - 1) / class_width = 7 := by
  sorry

end number_of_height_groups_l263_263074


namespace capacity_of_buckets_l263_263904

theorem capacity_of_buckets :
  (∃ x : ℝ, 26 * x = 39 * 9) → (∃ x : ℝ, 26 * x = 351 ∧ x = 13.5) :=
by
  sorry

end capacity_of_buckets_l263_263904


namespace number_of_dolls_l263_263079

theorem number_of_dolls (total_toys : ℕ) (fraction_action_figures : ℚ) 
  (remaining_fraction_action_figures : fraction_action_figures = 1 / 4) 
  (remaining_fraction_dolls : 1 - fraction_action_figures = 3 / 4) 
  (total_toys_eq : total_toys = 24) : 
  (total_toys - total_toys * fraction_action_figures) = 18 := 
by 
  sorry

end number_of_dolls_l263_263079


namespace part_one_max_value_range_of_a_l263_263228

def f (x a : ℝ) : ℝ := |x + 2| - |x - 3| - a

theorem part_one_max_value (a : ℝ) (h : a = 1) : ∃ x : ℝ, f x a = 4 := 
by sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 4 / a) :  (0 < a ∧ a ≤ 1) ∨ 4 ≤ a :=
by sorry

end part_one_max_value_range_of_a_l263_263228


namespace sum_of_decimals_l263_263952

theorem sum_of_decimals : (0.305 : ℝ) + (0.089 : ℝ) + (0.007 : ℝ) = 0.401 := by
  sorry

end sum_of_decimals_l263_263952


namespace magic_square_sum_l263_263213

-- Definitions based on the conditions outlined in the problem
def magic_sum := 83
def a := 42
def b := 26
def c := 29
def e := 34
def d := 36

theorem magic_square_sum :
  d + e = 70 :=
by
  -- Proof is omitted as per instructions
  sorry

end magic_square_sum_l263_263213


namespace value_of_expression_l263_263190

variable (a b : ℝ)

def system_of_equations : Prop :=
  (2 * a - b = 12) ∧ (a + 2 * b = 8)

theorem value_of_expression (h : system_of_equations a b) : 3 * a + b = 20 :=
  sorry

end value_of_expression_l263_263190


namespace spotted_and_fluffy_cats_l263_263768

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_equiv : total_cats = 120) (one_third_spotted : ℕ → ℕ) (one_fourth_fluffy_spotted : ℕ → ℕ) :
  (one_third_spotted total_cats * one_fourth_fluffy_spotted (one_third_spotted total_cats) = 10) :=
by
  sorry

end spotted_and_fluffy_cats_l263_263768


namespace solve_fraction_l263_263022

theorem solve_fraction (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : (x - 2) * (x + 1) ≠ 0) : x = 1 := 
sorry

end solve_fraction_l263_263022


namespace arithmetic_geometric_product_l263_263992

theorem arithmetic_geometric_product :
  let a (n : ℕ) := 2 * n - 1
  let b (n : ℕ) := 2 ^ (n - 1)
  b (a 1) * b (a 3) * b (a 5) = 4096 :=
by 
  sorry

end arithmetic_geometric_product_l263_263992


namespace sticker_ratio_l263_263791

variable (Dan Tom Bob : ℕ)

theorem sticker_ratio 
  (h1 : Dan = 2 * Tom) 
  (h2 : Tom = Bob) 
  (h3 : Bob = 12) 
  (h4 : Dan = 72) : 
  Tom = Bob :=
by
  sorry

end sticker_ratio_l263_263791


namespace boys_other_communities_l263_263837

/-- 
In a school of 850 boys, 44% are Muslims, 28% are Hindus, 
10% are Sikhs, and the remaining belong to other communities.
Prove that the number of boys belonging to other communities is 153.
-/
theorem boys_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℚ)
  (h_total_boys : total_boys = 850)
  (h_percentage_muslims : percentage_muslims = 44)
  (h_percentage_hindus : percentage_hindus = 28)
  (h_percentage_sikhs : percentage_sikhs = 10) :
  let percentage_others := 100 - (percentage_muslims + percentage_hindus + percentage_sikhs)
  let number_others := (percentage_others / 100) * total_boys
  number_others = 153 := 
by
  sorry

end boys_other_communities_l263_263837


namespace expression_evaluation_l263_263935

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end expression_evaluation_l263_263935


namespace combine_heaps_l263_263505

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l263_263505


namespace smallest_base10_integer_l263_263267

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l263_263267


namespace proposition2_and_4_correct_l263_263946

theorem proposition2_and_4_correct (a b : ℝ) : 
  (a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧ 
  (a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → a^2 + b^2 = 9) :=
by
  sorry

end proposition2_and_4_correct_l263_263946


namespace split_into_similar_heaps_l263_263518

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l263_263518


namespace trig_identity_l263_263807

theorem trig_identity (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : π/2 < α ∧ α < π) : 
  - (Real.sin (2 * α) / Real.cos α) = -6/5 :=
by
  sorry

end trig_identity_l263_263807


namespace range_of_f_on_nonneg_reals_l263_263640

theorem range_of_f_on_nonneg_reals (k : ℕ) (h_even : k % 2 = 0) (h_pos : 0 < k) :
    ∀ y : ℝ, 0 ≤ y ↔ ∃ x : ℝ, 0 ≤ x ∧ x^k = y :=
by
  sorry

end range_of_f_on_nonneg_reals_l263_263640


namespace angle_triple_supplement_l263_263117

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263117


namespace simplify_and_evaluate_l263_263705

-- Define the given expression
noncomputable def given_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m^2) + 3 * m) / (m + 1)

-- Define the condition
def condition (m : ℝ) : Prop :=
  m = Real.sqrt 3

-- Define the correct answer
def correct_answer : ℝ :=
  1 - Real.sqrt 3

-- State the theorem
theorem simplify_and_evaluate 
  (m : ℝ) (h : condition m) : 
  given_expression m = correct_answer := by
  sorry

end simplify_and_evaluate_l263_263705


namespace arithmetic_sequence_properties_l263_263347

theorem arithmetic_sequence_properties
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * ((a_n 0 + a_n (n-1)) / 2))
  (h2 : S 6 < S 7)
  (h3 : S 7 > S 8) :
  (a_n 8 - a_n 7 < 0) ∧ (S 9 < S 6) ∧ (∀ m, S m ≤ S 7) :=
by
  sorry

end arithmetic_sequence_properties_l263_263347


namespace largest_class_students_l263_263027

theorem largest_class_students (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = x) (h2 : n2 = x - 2) (h3 : n3 = x - 4) (h4 : n4 = x - 6) (h5 : n5 = x - 8) (h_sum : n1 + n2 + n3 + n4 + n5 = 140) : x = 32 :=
by {
  sorry
}

end largest_class_students_l263_263027


namespace no_integer_solutions_l263_263068

theorem no_integer_solutions (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hq : Nat.Prime (2*p + 1)) :
  ∀ (x y z : ℤ), x^p + 2 * y^p + 5 * z^p = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_integer_solutions_l263_263068


namespace playground_ratio_l263_263987

theorem playground_ratio (L B : ℕ) (playground_area landscape_area : ℕ) 
  (h1 : B = 8 * L)
  (h2 : B = 480)
  (h3 : playground_area = 3200)
  (h4 : landscape_area = L * B) : 
  (playground_area : ℚ) / landscape_area = 1 / 9 :=
by
  sorry

end playground_ratio_l263_263987


namespace percent_of_area_triangle_in_pentagon_l263_263608

-- Defining a structure for the problem statement
structure PentagonAndTriangle where
  s : ℝ -- side length of the equilateral triangle
  side_square : ℝ -- side of the square
  area_triangle : ℝ
  area_square : ℝ
  area_pentagon : ℝ

noncomputable def calculate_areas (s : ℝ) : PentagonAndTriangle :=
  let height_triangle := s * (Real.sqrt 3) / 2
  let area_triangle := Real.sqrt 3 / 4 * s^2
  let area_square := height_triangle^2
  let area_pentagon := area_square + area_triangle
  { s := s, side_square := height_triangle, area_triangle := area_triangle, area_square := area_square, area_pentagon := area_pentagon }

/--
Prove that the percentage of the pentagon's area that is the area of the equilateral triangle is (3 * (Real.sqrt 3 - 1)) / 6 * 100%.
-/
theorem percent_of_area_triangle_in_pentagon 
  (s : ℝ) 
  (pt : PentagonAndTriangle)
  (h₁ : pt = calculate_areas s)
  : pt.area_triangle / pt.area_pentagon = (3 * (Real.sqrt 3 - 1)) / 6 * 100 :=
by
  sorry

end percent_of_area_triangle_in_pentagon_l263_263608


namespace systematic_sampling_selects_616_l263_263909

theorem systematic_sampling_selects_616 (n : ℕ) (h₁ : n = 1000) (h₂ : (∀ i : ℕ, ∃ j : ℕ, i = 46 + j * 10) → True) :
  (∃ m : ℕ, m = 616) :=
  by
  sorry

end systematic_sampling_selects_616_l263_263909


namespace club_members_l263_263606

variable (x : ℕ)

theorem club_members (h1 : 2 * x + 5 = x + 15) : x = 10 := by
  sorry

end club_members_l263_263606


namespace probability_of_square_product_is_17_over_96_l263_263261

def num_tiles : Nat := 12
def num_die_faces : Nat := 8

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def favorable_outcomes_count : Nat :=
  -- Valid pairs where tile's number and die's number product is a perfect square
  List.length [ (1, 1), (1, 4), (2, 2), (4, 1),
                (1, 9), (3, 3), (9, 1), (4, 4),
                (2, 8), (8, 2), (5, 5), (6, 6),
                (4, 9), (9, 4), (7, 7), (8, 8),
                (9, 9) ] -- Equals 17 pairs

def total_outcomes_count : Nat :=
  num_tiles * num_die_faces

def probability_square_product : ℚ :=
  favorable_outcomes_count / total_outcomes_count

theorem probability_of_square_product_is_17_over_96 :
  probability_square_product = (17 : ℚ) / 96 := 
  by sorry

end probability_of_square_product_is_17_over_96_l263_263261


namespace angle_triple_supplement_l263_263121

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l263_263121


namespace gas_cost_per_gallon_l263_263260

theorem gas_cost_per_gallon (mpg : ℝ) (miles_per_day : ℝ) (days : ℝ) (total_cost : ℝ) : 
  mpg = 50 ∧ miles_per_day = 75 ∧ days = 10 ∧ total_cost = 45 → 
  (total_cost / ((miles_per_day * days) / mpg)) = 3 :=
by
  sorry

end gas_cost_per_gallon_l263_263260


namespace rainwater_cows_l263_263369

theorem rainwater_cows (chickens goats cows : ℕ) 
  (h1 : chickens = 18) 
  (h2 : goats = 2 * chickens) 
  (h3 : goats = 4 * cows) : 
  cows = 9 := 
sorry

end rainwater_cows_l263_263369


namespace common_ratio_of_geometric_sequence_l263_263942

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_nonzero : d ≠ 0) 
  (h_geom : (a 1)^2 = a 0 * a 2) :
  (a 2) / (a 0) = 3 / 2 := 
sorry

end common_ratio_of_geometric_sequence_l263_263942


namespace length_MN_l263_263386

variables {A B C D M N : Type}
variables {BC AD AB : ℝ} -- Lengths of sides
variables {a b : ℝ}

-- Given conditions
def is_trapezoid (a b BC AD AB : ℝ) : Prop :=
  BC = a ∧ AD = b ∧ AB = AD + BC

-- Given, side AB is divided into 5 equal parts and a line parallel to bases is drawn through the 3rd division point
def is_divided (AB : ℝ) : Prop := ∃ P_1 P_2 P_3 P_4, AB = P_4 + P_3 + P_2 + P_1

-- Prove the length of MN
theorem length_MN (a b : ℝ) (h_trapezoid : is_trapezoid a b BC AD AB) (h_divided : is_divided AB) : 
  MN = (2 * BC + 3 * AD) / 5 :=
sorry

end length_MN_l263_263386


namespace kona_additional_miles_l263_263805

theorem kona_additional_miles 
  (d_apartment_to_bakery : ℕ := 9) 
  (d_bakery_to_grandmother : ℕ := 24) 
  (d_grandmother_to_apartment : ℕ := 27) : 
  (d_apartment_to_bakery + d_bakery_to_grandmother + d_grandmother_to_apartment) - (2 * d_grandmother_to_apartment) = 6 := 
by 
  sorry

end kona_additional_miles_l263_263805


namespace triple_supplementary_angle_l263_263096

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l263_263096


namespace sandy_gave_puppies_l263_263061

theorem sandy_gave_puppies 
  (original_puppies : ℕ) 
  (puppies_with_spots : ℕ) 
  (puppies_left : ℕ) 
  (h1 : original_puppies = 8) 
  (h2 : puppies_with_spots = 3) 
  (h3 : puppies_left = 4) : 
  original_puppies - puppies_left = 4 := 
by {
  -- This is a placeholder for the proof.
  sorry
}

end sandy_gave_puppies_l263_263061
