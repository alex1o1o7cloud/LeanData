import Mathlib

namespace repeating_decimals_for_n_div_18_l980_98029

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l980_98029


namespace tan_105_eq_neg2_sub_sqrt3_l980_98002

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l980_98002


namespace alice_has_ball_after_two_turns_l980_98061

noncomputable def prob_alice_has_ball_after_two_turns : ℚ :=
  let p_A_B := (3 : ℚ) / 5 -- Probability Alice tosses to Bob
  let p_B_A := (1 : ℚ) / 3 -- Probability Bob tosses to Alice
  let p_A_A := (2 : ℚ) / 5 -- Probability Alice keeps the ball
  (p_A_B * p_B_A) + (p_A_A * p_A_A)

theorem alice_has_ball_after_two_turns :
  prob_alice_has_ball_after_two_turns = 9 / 25 :=
by
  -- skipping the proof
  sorry

end alice_has_ball_after_two_turns_l980_98061


namespace simplify_evaluate_expression_l980_98053

theorem simplify_evaluate_expression (a b : ℚ) (h1 : a = -2) (h2 : b = 1/5) :
    2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - (1/2) * a^3 * b)) = 8 := 
by
  sorry

end simplify_evaluate_expression_l980_98053


namespace teal_more_green_count_l980_98033

open Set

-- Define the survey data structure
def Survey : Type := {p : ℕ // p ≤ 150}

def people_surveyed : ℕ := 150
def more_blue (s : Survey) : Prop := sorry
def more_green (s : Survey) : Prop := sorry

-- Define the given conditions
def count_more_blue : ℕ := 90
def count_more_both : ℕ := 40
def count_neither : ℕ := 20

-- Define the proof statement
theorem teal_more_green_count :
  (count_more_both + (people_surveyed - (count_neither + (count_more_blue - count_more_both)))) = 80 :=
by {
  -- Sorry is used as a placeholder for the proof
  sorry
}

end teal_more_green_count_l980_98033


namespace qin_jiushao_algorithm_correct_operations_l980_98084

def qin_jiushao_algorithm_operations (f : ℝ → ℝ) (x : ℝ) : ℕ × ℕ := sorry

def f (x : ℝ) : ℝ := 4 * x^5 - x^2 + 2
def x : ℝ := 3

theorem qin_jiushao_algorithm_correct_operations :
  qin_jiushao_algorithm_operations f x = (5, 2) :=
sorry

end qin_jiushao_algorithm_correct_operations_l980_98084


namespace find_x_l980_98028

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log_base_5 (x : ℝ) : ℝ := Real.log x / Real.log 5
noncomputable def log_base_4 (x : ℝ) : ℝ := Real.log x / Real.log 4

noncomputable def right_triangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b + 3 * c * (a + b + c)

noncomputable def right_triangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b * c

noncomputable def rectangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  2 * (a * b + a * a + b * a)

noncomputable def rectangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  a * b * a

theorem find_x (x : ℝ) (h : right_triangular_prism_area x + rectangular_prism_area x = right_triangular_prism_volume x + rectangular_prism_volume x) :
  x = 1152 := by
sorry

end find_x_l980_98028


namespace time_released_rope_first_time_l980_98035

theorem time_released_rope_first_time :
  ∀ (rate_ascent : ℕ) (rate_descent : ℕ) (time_first_ascent : ℕ) (time_second_ascent : ℕ) (highest_elevation : ℕ)
    (total_elevation_gained : ℕ) (elevation_difference : ℕ) (time_descent : ℕ),
  rate_ascent = 50 →
  rate_descent = 10 →
  time_first_ascent = 15 →
  time_second_ascent = 15 →
  highest_elevation = 1400 →
  total_elevation_gained = (rate_ascent * time_first_ascent) + (rate_ascent * time_second_ascent) →
  elevation_difference = total_elevation_gained - highest_elevation →
  time_descent = elevation_difference / rate_descent →
  time_descent = 10 :=
by
  intros rate_ascent rate_descent time_first_ascent time_second_ascent highest_elevation total_elevation_gained elevation_difference time_descent
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end time_released_rope_first_time_l980_98035


namespace find_number_l980_98096

theorem find_number (x : ℕ) (h1 : x - 13 = 31) : x + 11 = 55 :=
  sorry

end find_number_l980_98096


namespace remainder_of_large_power_l980_98023

theorem remainder_of_large_power :
  (2^(2^(2^2))) % 500 = 36 :=
sorry

end remainder_of_large_power_l980_98023


namespace polygon_with_given_angle_sum_l980_98008

-- Definition of the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Definition of the sum of exterior angles of a polygon
def sum_exterior_angles : ℝ := 360

-- Given condition: the sum of the interior angles is four times the sum of the exterior angles
def sum_condition (n : ℕ) : Prop :=
  sum_interior_angles n = 4 * sum_exterior_angles

-- The main theorem we want to prove
theorem polygon_with_given_angle_sum : 
  ∃ n : ℕ, sum_condition n ∧ n = 10 :=
by
  sorry

end polygon_with_given_angle_sum_l980_98008


namespace remainder_prod_mod_10_l980_98034

theorem remainder_prod_mod_10 :
  (2457 * 7963 * 92324) % 10 = 4 :=
  sorry

end remainder_prod_mod_10_l980_98034


namespace angle_of_inclination_l980_98010

theorem angle_of_inclination (θ : ℝ) (h_range : 0 ≤ θ ∧ θ < 180)
  (h_line : ∀ x y : ℝ, x + y - 1 = 0 → x = -y + 1) :
  θ = 135 :=
by 
  sorry

end angle_of_inclination_l980_98010


namespace otimes_example_l980_98083

def otimes (a b : ℤ) : ℤ := a^2 - a * b

theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end otimes_example_l980_98083


namespace right_drawing_num_triangles_l980_98017

-- Given the conditions:
-- 1. Nine distinct lines in the right drawing
-- 2. Any combination of 3 lines out of these 9 forms a triangle
-- 3. Count of intersections of these lines where exactly three lines intersect

def num_triangles : Nat := 84 -- Calculated via binomial coefficient
def num_intersections : Nat := 61 -- Given or calculated from the problem

-- The target theorem to prove that the number of triangles is equal to 23
theorem right_drawing_num_triangles :
  num_triangles - num_intersections = 23 :=
by
  -- Proof would go here, but we skip it as per the instructions
  sorry

end right_drawing_num_triangles_l980_98017


namespace total_work_completion_days_l980_98068

theorem total_work_completion_days :
  let Amit_work_rate := 1 / 15
  let Ananthu_work_rate := 1 / 90
  let Chandra_work_rate := 1 / 45

  let Amit_days_worked_alone := 3
  let Ananthu_days_worked_alone := 6
  
  let work_by_Amit := Amit_days_worked_alone * Amit_work_rate
  let work_by_Ananthu := Ananthu_days_worked_alone * Ananthu_work_rate
  
  let initial_work_done := work_by_Amit + work_by_Ananthu
  let remaining_work := 1 - initial_work_done

  let combined_work_rate := Amit_work_rate + Ananthu_work_rate + Chandra_work_rate
  let days_all_worked_together := remaining_work / combined_work_rate

  Amit_days_worked_alone + Ananthu_days_worked_alone + days_all_worked_together = 17 :=
by
  sorry

end total_work_completion_days_l980_98068


namespace vertex_on_x_axis_iff_t_eq_neg_4_l980_98041

theorem vertex_on_x_axis_iff_t_eq_neg_4 (t : ℝ) :
  (∃ x : ℝ, (4 + t) = 0) ↔ t = -4 :=
by
  sorry

end vertex_on_x_axis_iff_t_eq_neg_4_l980_98041


namespace new_individuals_weight_l980_98081

variables (W : ℝ) (A B C : ℝ)

-- Conditions
def original_twelve_people_weight : ℝ := W
def weight_leaving_1 : ℝ := 64
def weight_leaving_2 : ℝ := 75
def weight_leaving_3 : ℝ := 81
def average_increase : ℝ := 3.6
def total_weight_increase : ℝ := 12 * average_increase
def weight_leaving_sum : ℝ := weight_leaving_1 + weight_leaving_2 + weight_leaving_3

-- Equation derived from the problem conditions
def new_individuals_weight_sum : ℝ := weight_leaving_sum + total_weight_increase

-- Theorem to prove
theorem new_individuals_weight :
  A + B + C = 263.2 :=
by
  sorry

end new_individuals_weight_l980_98081


namespace average_age_union_l980_98059

open Real

variables {a b c d A B C D : ℝ}

theorem average_age_union (h1 : A / a = 40)
                         (h2 : B / b = 30)
                         (h3 : C / c = 45)
                         (h4 : D / d = 35)
                         (h5 : (A + B) / (a + b) = 37)
                         (h6 : (A + C) / (a + c) = 42)
                         (h7 : (A + D) / (a + d) = 39)
                         (h8 : (B + C) / (b + c) = 40)
                         (h9 : (B + D) / (b + d) = 37)
                         (h10 : (C + D) / (c + d) = 43) : 
  (A + B + C + D) / (a + b + c + d) = 44.5 := 
sorry

end average_age_union_l980_98059


namespace determine_constants_l980_98067

theorem determine_constants (a b c d : ℝ) 
  (periodic : (2 * (2 * Real.pi / b) = 4 * Real.pi))
  (vert_shift : d = 3)
  (max_val : (d + a = 8))
  (min_val : (d - a = -2)) :
  a = 5 ∧ b = 1 :=
by
  sorry

end determine_constants_l980_98067


namespace percentage_of_total_money_raised_from_donations_l980_98095

-- Define the conditions
def max_donation := 1200
def num_donors_max := 500
def half_donation := max_donation / 2
def num_donors_half := 3 * num_donors_max
def total_money_raised := 3750000

-- Define the amounts collected from each group
def amount_from_max_donors := num_donors_max * max_donation
def amount_from_half_donors := num_donors_half * half_donation
def total_amount_from_donations := amount_from_max_donors + amount_from_half_donors

-- Define the percentage calculation
def percentage_of_total := (total_amount_from_donations / total_money_raised) * 100

-- State the theorem (but not the proof)
theorem percentage_of_total_money_raised_from_donations : 
  percentage_of_total = 40 := by
  sorry

end percentage_of_total_money_raised_from_donations_l980_98095


namespace sum_series_eq_11_div_18_l980_98047

theorem sum_series_eq_11_div_18 :
  (∑' n : ℕ, if n = 0 then 0 else 1 / (n * (n + 3))) = 11 / 18 :=
by
  sorry

end sum_series_eq_11_div_18_l980_98047


namespace find_a2_b2_l980_98013

theorem find_a2_b2 (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 :=
by
  sorry

end find_a2_b2_l980_98013


namespace successive_percentage_reduction_l980_98011

theorem successive_percentage_reduction (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  a + b - (a * b) / 100 = 40 := by
  sorry

end successive_percentage_reduction_l980_98011


namespace find_m_if_f_even_l980_98090

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_f_even :
  (∀ x : ℝ, f m (-x) = f m x) → m = 2 :=
by 
  intro h
  sorry

end find_m_if_f_even_l980_98090


namespace factorization_of_x4_plus_81_l980_98045

theorem factorization_of_x4_plus_81 :
  ∀ x : ℝ, x^4 + 81 = (x^2 - 3 * x + 4.5) * (x^2 + 3 * x + 4.5) :=
by
  intros x
  sorry

end factorization_of_x4_plus_81_l980_98045


namespace total_books_l980_98097

-- Define the number of books Victor originally had and the number he bought
def original_books : ℕ := 9
def bought_books : ℕ := 3

-- The proof problem statement: Prove Victor has a total of original_books + bought_books books
theorem total_books : original_books + bought_books = 12 := by
  -- proof will go here, using sorry to indicate it's omitted
  sorry

end total_books_l980_98097


namespace sqrt_expression_l980_98015

theorem sqrt_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2 * x + 1)) :=
by
  sorry

end sqrt_expression_l980_98015


namespace determinant_matrix_equivalence_l980_98006

variable {R : Type} [CommRing R]

theorem determinant_matrix_equivalence
  (x y z w : R)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 4 * w) - z * (5 * x + 4 * y) = 12) :=
by sorry

end determinant_matrix_equivalence_l980_98006


namespace ranking_l980_98077

variables (score : string → ℝ)
variables (Hannah Cassie Bridget David : string)

-- Conditions based on the problem statement
axiom Hannah_shows_her_test_to_everyone : ∀ x, x ≠ Hannah → x = Cassie ∨ x = Bridget ∨ x = David
axiom David_shows_his_test_only_to_Bridget : ∀ x, x ≠ Bridget → x ≠ David
axiom Cassie_does_not_show_her_test : ∀ x, x = Hannah ∨ x = Bridget ∨ x = David → x ≠ Cassie

-- Statements based on what Cassie and Bridget claim
axiom Cassie_statement : score Cassie > min (score Hannah) (score Bridget)
axiom Bridget_statement : score David > score Bridget

-- Final ranking to be proved
theorem ranking : score David > score Bridget ∧ score Bridget > score Cassie ∧ score Cassie > score Hannah := sorry

end ranking_l980_98077


namespace number_of_eggs_l980_98026

-- Define the conditions as assumptions
variables (marbles : ℕ) (eggs : ℕ)
variables (eggs_A eggs_B eggs_C : ℕ)
variables (marbles_A marbles_B marbles_C : ℕ)

-- Conditions from the problem
axiom eggs_total : marbles = 4
axiom marbles_total : eggs = 15
axiom eggs_groups : eggs_A ≠ eggs_B ∧ eggs_B ≠ eggs_C ∧ eggs_A ≠ eggs_C
axiom marbles_diff1 : marbles_B - marbles_A = eggs_B
axiom marbles_diff2 : marbles_C - marbles_B = eggs_C

-- Prove that the number of eggs in each group is as specified in the answer
theorem number_of_eggs :
  eggs_A = 12 ∧ eggs_B = 1 ∧ eggs_C = 2 :=
by {
  sorry
}

end number_of_eggs_l980_98026


namespace proper_subset_of_A_l980_98027

def A : Set ℝ := {x | x^2 < 5 * x}

theorem proper_subset_of_A :
  (∀ x, x ∈ Set.Ioc 1 5 → x ∈ A ∧ ∀ y, y ∈ A → y ∉ Set.Ioc 1 5 → ¬(Set.Ioc 1 5 = A)) :=
sorry

end proper_subset_of_A_l980_98027


namespace range_k_l980_98009

noncomputable def point (α : Type*) := (α × α)

def M : point ℝ := (0, 2)
def N : point ℝ := (-2, 0)

def line (k : ℝ) (P : point ℝ) := k * P.1 - P.2 - 2 * k + 2 = 0
def angle_condition (M N P : point ℝ) := true -- placeholder for the condition that ∠MPN ≥ π/2

theorem range_k (k : ℝ) (P : point ℝ)
  (hP_on_line : line k P)
  (h_angle_cond : angle_condition M N P) :
  (1 / 7 : ℝ) ≤ k ∧ k ≤ 1 :=
sorry

end range_k_l980_98009


namespace prove_f_neg_2_l980_98012

noncomputable def f (a b x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Main theorem statement
theorem prove_f_neg_2 (a b : ℝ) (h : f a b 2 = 9) : f a b (-2) = 13 := 
by
  sorry

end prove_f_neg_2_l980_98012


namespace teacher_total_score_l980_98079

/-- Conditions -/
def written_test_score : ℝ := 80
def interview_score : ℝ := 60
def written_test_weight : ℝ := 0.6
def interview_weight : ℝ := 0.4

/-- Prove the total score -/
theorem teacher_total_score :
  written_test_score * written_test_weight + interview_score * interview_weight = 72 :=
by
  sorry

end teacher_total_score_l980_98079


namespace twelfth_equation_l980_98021

theorem twelfth_equation : (14 : ℤ)^2 - (12 : ℤ)^2 = 4 * 13 := by
  sorry

end twelfth_equation_l980_98021


namespace solve_expression_l980_98063

noncomputable def expression : ℝ := 5 * 1.6 - 2 * 1.4 / 1.3

theorem solve_expression : expression = 5.8462 := 
by 
  sorry

end solve_expression_l980_98063


namespace find_x_l980_98000

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end find_x_l980_98000


namespace final_price_after_increase_and_decrease_l980_98058

variable (P : ℝ)

theorem final_price_after_increase_and_decrease (h : P > 0) : 
  let increased_price := P * 1.15
  let final_price := increased_price * 0.85
  final_price = P * 0.9775 :=
by
  sorry

end final_price_after_increase_and_decrease_l980_98058


namespace less_than_n_repetitions_l980_98049

variable {n : ℕ} (a : Fin n.succ → ℕ)

def is_repetition (a : Fin n.succ → ℕ) (k l p : ℕ) : Prop :=
  p ≤ (l - k) / 2 ∧
  (∀ i : ℕ, k + 1 ≤ i ∧ i ≤ l - p → a ⟨i, sorry⟩ = a ⟨i + p, sorry⟩) ∧
  (k > 0 → a ⟨k, sorry⟩ ≠ a ⟨k + p, sorry⟩) ∧
  (l < n → a ⟨l - p + 1, sorry⟩ ≠ a ⟨l + 1, sorry⟩)

theorem less_than_n_repetitions (a : Fin n.succ → ℕ) :
  ∃ r : ℕ, r < n ∧ ∀ k l : ℕ, is_repetition a k l r → r < n :=
sorry

end less_than_n_repetitions_l980_98049


namespace mod11_residue_l980_98039

theorem mod11_residue :
  (305 % 11 = 8) →
  (44 % 11 = 0) →
  (176 % 11 = 0) →
  (18 % 11 = 7) →
  (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end mod11_residue_l980_98039


namespace Jermaine_more_than_Terrence_l980_98091

theorem Jermaine_more_than_Terrence :
  ∀ (total_earnings Terrence_earnings Emilee_earnings : ℕ),
    total_earnings = 90 →
    Terrence_earnings = 30 →
    Emilee_earnings = 25 →
    (total_earnings - Terrence_earnings - Emilee_earnings) - Terrence_earnings = 5 := by
  sorry

end Jermaine_more_than_Terrence_l980_98091


namespace geometric_seq_increasing_condition_not_sufficient_nor_necessary_l980_98050

-- Definitions based on conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = q * a n
def monotonically_increasing (a : ℕ → ℝ) := ∀ n : ℕ, a n ≤ a (n + 1)
def common_ratio_gt_one (q : ℝ) := q > 1

-- Proof statement of the problem
theorem geometric_seq_increasing_condition_not_sufficient_nor_necessary 
    (a : ℕ → ℝ) (q : ℝ) 
    (h1 : geometric_sequence a q) : 
    ¬(common_ratio_gt_one q ↔ monotonically_increasing a) :=
sorry

end geometric_seq_increasing_condition_not_sufficient_nor_necessary_l980_98050


namespace verka_digit_sets_l980_98060

-- Define the main conditions as:
def is_three_digit_number (a b c : ℕ) : Prop :=
  let num1 := 100 * a + 10 * b + c
  let num2 := 100 * a + 10 * c + b
  let num3 := 100 * b + 10 * a + c
  let num4 := 100 * b + 10 * c + a
  let num5 := 100 * c + 10 * a + b
  let num6 := 100 * c + 10 * b + a
  num1 + num2 + num3 + num4 + num5 + num6 = 1221

-- Prove the main theorem
theorem verka_digit_sets :
  ∃ (a b c : ℕ), is_three_digit_number a a c ∧
                 ((a, c) = (1, 9) ∨ (a, c) = (2, 7) ∨ (a, c) = (3, 5) ∨ (a, c) = (4, 3) ∨ (a, c) = (5, 1)) :=
by sorry

end verka_digit_sets_l980_98060


namespace percentage_sophia_ate_l980_98025

theorem percentage_sophia_ate : 
  ∀ (caden zoe noah sophia : ℝ),
    caden = 20 / 100 →
    zoe = caden + (0.5 * caden) →
    noah = zoe + (0.5 * zoe) →
    caden + zoe + noah + sophia = 1 →
    sophia = 5 / 100 :=
by
  intros
  sorry

end percentage_sophia_ate_l980_98025


namespace seeds_in_big_garden_is_correct_l980_98080

def total_seeds : ℕ := 41
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 4

def seeds_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden
def seeds_in_big_garden : ℕ := total_seeds - seeds_in_small_gardens

theorem seeds_in_big_garden_is_correct : seeds_in_big_garden = 29 := by
  -- proof goes here
  sorry

end seeds_in_big_garden_is_correct_l980_98080


namespace cost_of_tree_planting_l980_98036

theorem cost_of_tree_planting 
  (initial_temp final_temp : ℝ) (temp_drop_per_tree cost_per_tree : ℝ) 
  (h_initial: initial_temp = 80) (h_final: final_temp = 78.2) 
  (h_temp_drop_per_tree: temp_drop_per_tree = 0.1) 
  (h_cost_per_tree: cost_per_tree = 6) : 
  (final_temp - initial_temp) / temp_drop_per_tree * cost_per_tree = 108 := 
by
  sorry

end cost_of_tree_planting_l980_98036


namespace parabola_c_value_l980_98099

theorem parabola_c_value (b c : ℝ) 
  (h1 : 5 = 2 * 1^2 + b * 1 + c)
  (h2 : 17 = 2 * 3^2 + b * 3 + c) : 
  c = 5 := 
by
  sorry

end parabola_c_value_l980_98099


namespace ratio_of_candies_l980_98040

theorem ratio_of_candies (emily_candies jennifer_candies bob_candies : ℕ)
  (h1 : emily_candies = 6)
  (h2 : bob_candies = 4)
  (h3 : jennifer_candies = 2 * emily_candies) : 
  jennifer_candies / bob_candies = 3 := 
by
  sorry

end ratio_of_candies_l980_98040


namespace decimal_subtraction_l980_98085

theorem decimal_subtraction (a b : ℝ) (h1 : a = 3.79) (h2 : b = 2.15) : a - b = 1.64 := by
  rw [h1, h2]
  -- This follows from the correct calculation rule
  sorry

end decimal_subtraction_l980_98085


namespace books_read_l980_98024

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l980_98024


namespace Emily_money_made_l980_98044

def price_per_bar : ℕ := 4
def total_bars : ℕ := 8
def bars_sold : ℕ := total_bars - 3
def money_made : ℕ := bars_sold * price_per_bar

theorem Emily_money_made : money_made = 20 :=
by
  sorry

end Emily_money_made_l980_98044


namespace min_value_of_expression_l980_98031

open Real

noncomputable def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / (x + 2) + 1 / (y + 2) = 1 / 4)

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 4) :
  2 * x + 3 * y = 5 + 4 * sqrt 3 :=
sorry

end min_value_of_expression_l980_98031


namespace percentage_defective_meters_l980_98001

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (h1 : total_meters = 150) (h2 : defective_meters = 15) : 
  (defective_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
sorry

end percentage_defective_meters_l980_98001


namespace muffin_machine_completion_time_l980_98042

theorem muffin_machine_completion_time :
  let start_time := 9 * 60 -- minutes
  let partial_completion_time := (12 * 60) + 15 -- minutes
  let partial_duration := partial_completion_time - start_time
  let fraction_of_day := 1 / 4
  let total_duration := partial_duration / fraction_of_day
  start_time + total_duration = (22 * 60) := -- 10:00 PM in minutes
by
  sorry

end muffin_machine_completion_time_l980_98042


namespace calculation_result_l980_98038

def initial_number : ℕ := 15
def subtracted_value : ℕ := 2
def added_value : ℕ := 4
def divisor : ℕ := 1
def second_divisor : ℕ := 2
def multiplier : ℕ := 8

theorem calculation_result : 
  (initial_number - subtracted_value + (added_value / divisor : ℕ)) / second_divisor * multiplier = 68 :=
by
  sorry

end calculation_result_l980_98038


namespace base_n_multiple_of_5_l980_98052

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 4 + n + 3 * n^2 + 5 * n^3 + n^4 + 4 * n^5

-- The main theorem to be proven
theorem base_n_multiple_of_5 (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 100) : 
  f n % 5 ≠ 0 :=
by sorry

end base_n_multiple_of_5_l980_98052


namespace exists_int_x_l980_98032

theorem exists_int_x (K M N : ℤ) (h1 : K ≠ 0) (h2 : M ≠ 0) (h3 : N ≠ 0) (h_coprime : Int.gcd K M = 1) :
  ∃ x : ℤ, K ∣ (M * x + N) :=
by
  sorry

end exists_int_x_l980_98032


namespace good_walker_catches_up_l980_98022

-- Definitions based on the conditions in the problem
def steps_good_walker := 100
def steps_bad_walker := 60
def initial_lead := 100

-- Mathematical proof problem statement
theorem good_walker_catches_up :
  ∃ x : ℕ, x = initial_lead + (steps_bad_walker * x / steps_good_walker) :=
sorry

end good_walker_catches_up_l980_98022


namespace right_angled_triangle_max_area_l980_98076

theorem right_angled_triangle_max_area (a b : ℝ) (h : a + b = 4) : (1 / 2) * a * b ≤ 2 :=
by 
  sorry

end right_angled_triangle_max_area_l980_98076


namespace initial_guinea_fowls_l980_98004

theorem initial_guinea_fowls (initial_chickens initial_turkeys : ℕ) 
  (initial_guinea_fowls : ℕ) (lost_chickens lost_turkeys lost_guinea_fowls : ℕ) 
  (total_birds_end : ℕ) (days : ℕ)
  (hc : initial_chickens = 300) (ht : initial_turkeys = 200) 
  (lc : lost_chickens = 20) (lt : lost_turkeys = 8) (lg : lost_guinea_fowls = 5) 
  (d : days = 7) (tb : total_birds_end = 349) :
  initial_guinea_fowls = 80 := 
by 
  sorry

end initial_guinea_fowls_l980_98004


namespace arithmetic_sequence_a3_l980_98073

theorem arithmetic_sequence_a3 (a : ℕ → ℕ) (h1 : a 6 = 6) (h2 : a 9 = 9) : a 3 = 3 :=
by
  -- proof goes here
  sorry

end arithmetic_sequence_a3_l980_98073


namespace sequence_of_arrows_512_to_517_is_B_C_D_E_A_l980_98092

noncomputable def sequence_from_512_to_517 : List Char :=
  let pattern := ['A', 'B', 'C', 'D', 'E']
  pattern.drop 2 ++ pattern.take 2

theorem sequence_of_arrows_512_to_517_is_B_C_D_E_A : sequence_from_512_to_517 = ['B', 'C', 'D', 'E', 'A'] :=
  sorry

end sequence_of_arrows_512_to_517_is_B_C_D_E_A_l980_98092


namespace total_repair_cost_l980_98064

theorem total_repair_cost :
  let rate1 := 60
  let hours1 := 8
  let days1 := 14
  let rate2 := 75
  let hours2 := 6
  let days2 := 10
  let parts_cost := 3200
  let first_mechanic_cost := rate1 * hours1 * days1
  let second_mechanic_cost := rate2 * hours2 * days2
  let total_cost := first_mechanic_cost + second_mechanic_cost + parts_cost
  total_cost = 14420 := by
  sorry

end total_repair_cost_l980_98064


namespace correct_multiplier_l980_98056

theorem correct_multiplier (x : ℕ) 
  (h1 : 137 * 34 + 1233 = 137 * x) : 
  x = 43 := 
by 
  sorry

end correct_multiplier_l980_98056


namespace total_items_at_bakery_l980_98019

theorem total_items_at_bakery (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ) (h1 : bread_rolls = 49) (h2 : croissants = 19) (h3 : bagels = 22) : bread_rolls + croissants + bagels = 90 :=
by
  sorry

end total_items_at_bakery_l980_98019


namespace max_covered_squares_l980_98037

-- Definitions representing the conditions
def checkerboard_squares : ℕ := 1 -- side length of each square on the checkerboard
def card_side_len : ℕ := 2 -- side length of the card

-- Theorem statement representing the question and answer
theorem max_covered_squares : ∀ n, 
  (∃ board_side squared_len, 
    checkerboard_squares = 1 ∧ card_side_len = 2 ∧
    (board_side = checkerboard_squares ∧ squared_len = card_side_len) ∧
    n ≤ 16) →
  n = 16 :=
  sorry

end max_covered_squares_l980_98037


namespace find_replacement_percentage_l980_98054

noncomputable def final_percentage_replacement_alcohol_solution (a₁ p₁ p₂ x : ℝ) : Prop :=
  let d := 0.4 -- gallons
  let final_solution := 1 -- gallon
  let initial_pure_alcohol := a₁ * p₁ / 100
  let remaining_pure_alcohol := initial_pure_alcohol - (d * p₁ / 100)
  let added_pure_alcohol := d * x / 100
  remaining_pure_alcohol + added_pure_alcohol = final_solution * p₂ / 100

theorem find_replacement_percentage :
  final_percentage_replacement_alcohol_solution 1 75 65 50 :=
by
  sorry

end find_replacement_percentage_l980_98054


namespace gcd_lcm_sum_8_12_l980_98069

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l980_98069


namespace page_cost_in_cents_l980_98074

theorem page_cost_in_cents (notebooks pages_per_notebook total_cost : ℕ)
  (h_notebooks : notebooks = 2)
  (h_pages_per_notebook : pages_per_notebook = 50)
  (h_total_cost : total_cost = 5 * 100) :
  (total_cost / (notebooks * pages_per_notebook)) = 5 :=
by
  sorry

end page_cost_in_cents_l980_98074


namespace max_f_value_l980_98046

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (- (4 / 3) * x + 3) ((1 / 3) * x + 9))

theorem max_f_value : ∃ x : ℝ, f x = 31 / 13 :=
by 
  sorry

end max_f_value_l980_98046


namespace determine_n_l980_98078

theorem determine_n (x n : ℝ) : 
  (∃ c d : ℝ, G = (c * x + d) ^ 2) ∧ (G = (8 * x^2 + 24 * x + 3 * n) / 8) → n = 6 :=
by {
  sorry
}

end determine_n_l980_98078


namespace initial_crayons_l980_98062

theorem initial_crayons {C : ℕ} (h : C + 12 = 53) : C = 41 :=
by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end initial_crayons_l980_98062


namespace lattice_point_in_PQE_l980_98071

-- Define points and their integer coordinates
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a convex quadrilateral with integer coordinates
structure ConvexQuadrilateral :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

-- Define the intersection point of diagonals as another point
def diagIntersection (quad: ConvexQuadrilateral) : Point := sorry

-- Define the condition for the sum of angles at P and Q being less than 180 degrees
def sumAnglesLessThan180 (quad : ConvexQuadrilateral) : Prop := sorry

-- Define a function to check if a point is a lattice point
def isLatticePoint (p : Point) : Prop := true  -- Since all points are lattice points by definition

-- Define the proof problem
theorem lattice_point_in_PQE (quad : ConvexQuadrilateral) (E : Point) :
  sumAnglesLessThan180 quad →
  ∃ p : Point, p ≠ quad.P ∧ p ≠ quad.Q ∧ isLatticePoint p ∧ sorry := sorry -- (prove the point is in PQE)

end lattice_point_in_PQE_l980_98071


namespace shape_at_22_l980_98007

-- Define the pattern
def pattern : List String := ["triangle", "square", "diamond", "diamond", "circle"]

-- Function to get the nth shape in the repeated pattern sequence
def getShape (n : Nat) : String :=
  pattern.get! (n % pattern.length)

-- Statement to prove
theorem shape_at_22 : getShape 21 = "square" :=
by
  sorry

end shape_at_22_l980_98007


namespace arithmetic_sequence_a9_l980_98020

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a 1)
  (h2 : a 2 + a 4 = 2)
  (h5 : a 5 = 3) :
  a 9 = 7 :=
by
  sorry

end arithmetic_sequence_a9_l980_98020


namespace speed_of_second_train_l980_98043

-- Definitions of conditions
def distance_train1 : ℝ := 200
def speed_train1 : ℝ := 50
def distance_train2 : ℝ := 240
def time_train1_and_train2 : ℝ := 4

-- Statement of the problem
theorem speed_of_second_train : (distance_train2 / time_train1_and_train2) = 60 := by
  sorry

end speed_of_second_train_l980_98043


namespace num_enemies_left_l980_98057

-- Definitions of conditions
def points_per_enemy : Nat := 5
def total_enemies : Nat := 8
def earned_points : Nat := 10

-- Theorem statement to prove the number of undefeated enemies
theorem num_enemies_left (points_per_enemy total_enemies earned_points : Nat) : 
    (earned_points / points_per_enemy) <= total_enemies →
    total_enemies - (earned_points / points_per_enemy) = 6 := by
  sorry

end num_enemies_left_l980_98057


namespace moles_NaOH_to_form_H2O_2_moles_l980_98094

-- Define the reaction and moles involved
def reaction : String := "NH4NO3 + NaOH -> NaNO3 + NH3 + H2O"
def moles_H2O_produced : Nat := 2
def moles_NaOH_required (moles_H2O : Nat) : Nat := moles_H2O

-- Theorem stating the required moles of NaOH to produce 2 moles of H2O
theorem moles_NaOH_to_form_H2O_2_moles : moles_NaOH_required moles_H2O_produced = 2 := 
by
  sorry

end moles_NaOH_to_form_H2O_2_moles_l980_98094


namespace cylinder_surface_area_l980_98065

theorem cylinder_surface_area (h : ℝ) (r : ℝ) (h_eq : h = 12) (r_eq : r = 4) : 
  2 * π * r * (r + h) = 128 * π := 
by
  rw [h_eq, r_eq]
  sorry

end cylinder_surface_area_l980_98065


namespace total_vegetables_l980_98055

theorem total_vegetables (b k r : ℕ) (broccoli_weight_kg : ℝ) (broccoli_weight_g : ℝ) 
  (kohlrabi_mult : ℕ) (radish_mult : ℕ) :
  broccoli_weight_kg = 5 ∧ 
  broccoli_weight_g = 0.25 ∧ 
  kohlrabi_mult = 4 ∧ 
  radish_mult = 3 ∧ 
  b = broccoli_weight_kg / broccoli_weight_g ∧ 
  k = kohlrabi_mult * b ∧ 
  r = radish_mult * k →
  b + k + r = 340 := 
by
  sorry

end total_vegetables_l980_98055


namespace find_f_neg_one_l980_98003

theorem find_f_neg_one (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + x^2 = - (f x + x^2)) (h2 : f 1 = 1) : f (-1) = -3 := by
  sorry

end find_f_neg_one_l980_98003


namespace problem_conditions_l980_98018

noncomputable def f (a b x : ℝ) : ℝ := abs (x + a) + abs (2 * x - b)

theorem problem_conditions (ha : 0 < a) (hb : 0 < b) 
  (hmin : ∃ x : ℝ, f a b x = 1) : 
  2 * a + b = 2 ∧ 
  ∀ (t : ℝ), (∀ a b : ℝ, 
    (0 < a) → (0 < b) → (a + 2 * b ≥ t * a * b)) → 
  t ≤ 9 / 2 :=
by
  sorry

end problem_conditions_l980_98018


namespace four_divides_sum_of_squares_iff_even_l980_98086

theorem four_divides_sum_of_squares_iff_even (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 ∣ (a^2 + b^2 + c^2)) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end four_divides_sum_of_squares_iff_even_l980_98086


namespace fraction_not_covered_correct_l980_98089

def area_floor : ℕ := 64
def width_rug : ℕ := 2
def length_rug : ℕ := 7
def area_rug := width_rug * length_rug
def area_not_covered := area_floor - area_rug
def fraction_not_covered := (area_not_covered : ℚ) / area_floor

theorem fraction_not_covered_correct :
  fraction_not_covered = 25 / 32 :=
by
  -- Proof goes here
  sorry

end fraction_not_covered_correct_l980_98089


namespace solve_gcd_problem_l980_98088

def gcd_problem : Prop :=
  gcd 153 119 = 17

theorem solve_gcd_problem : gcd_problem :=
  by
    sorry

end solve_gcd_problem_l980_98088


namespace third_divisor_l980_98051

theorem third_divisor (x : ℕ) (h1 : x - 16 = 136) (h2 : ∃ y, y = x - 16) (h3 : 4 ∣ x) (h4 : 6 ∣ x) (h5 : 10 ∣ x) : 19 ∣ x := 
by
  sorry

end third_divisor_l980_98051


namespace find_x_squared_plus_y_squared_l980_98005

theorem find_x_squared_plus_y_squared (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y + x + y = 17) (h4 : x^2 * y + x * y^2 = 72) : x^2 + y^2 = 65 := 
  sorry

end find_x_squared_plus_y_squared_l980_98005


namespace vector_addition_AC_l980_98048

def vector := (ℝ × ℝ)

def AB : vector := (0, 1)
def BC : vector := (1, 0)

def AC (AB BC : vector) : vector := (AB.1 + BC.1, AB.2 + BC.2) 

theorem vector_addition_AC (AB BC : vector) (h1 : AB = (0, 1)) (h2 : BC = (1, 0)) : 
  AC AB BC = (1, 1) :=
by
  sorry

end vector_addition_AC_l980_98048


namespace seq_formula_l980_98066

theorem seq_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n - 1) + 1 := 
by 
  sorry

end seq_formula_l980_98066


namespace find_h_parallel_line_l980_98087

theorem find_h_parallel_line:
  ∃ h : ℚ, (3 * (h : ℚ) - 2 * (24 : ℚ) = 7) → (h = 47 / 3) :=
by
  sorry

end find_h_parallel_line_l980_98087


namespace factorization_of_expression_l980_98070

theorem factorization_of_expression (a b c : ℝ) : 
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
by
  sorry

end factorization_of_expression_l980_98070


namespace domain_log_function_l980_98014

theorem domain_log_function :
  {x : ℝ | 1 < x ∧ x < 3 ∧ x ≠ 2} = {x : ℝ | (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1)} :=
sorry

end domain_log_function_l980_98014


namespace max_min_values_l980_98072

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 8

theorem max_min_values :
  ∃ x_max x_min : ℝ, x_max ∈ Set.Icc (-3 : ℝ) 3 ∧ x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧
    f (-2) = 24 ∧ f 2 = -6 := sorry

end max_min_values_l980_98072


namespace jiaqi_grade_is_95_3_l980_98098

def extracurricular_score : ℝ := 96
def mid_term_score : ℝ := 92
def final_exam_score : ℝ := 97

def extracurricular_weight : ℝ := 0.2
def mid_term_weight : ℝ := 0.3
def final_exam_weight : ℝ := 0.5

def total_grade : ℝ :=
  extracurricular_score * extracurricular_weight +
  mid_term_score * mid_term_weight +
  final_exam_score * final_exam_weight

theorem jiaqi_grade_is_95_3 : total_grade = 95.3 :=
by
  simp [total_grade, extracurricular_score, mid_term_score, final_exam_score,
    extracurricular_weight, mid_term_weight, final_exam_weight]
  sorry

end jiaqi_grade_is_95_3_l980_98098


namespace tangent_line_equation_l980_98082

theorem tangent_line_equation (y : ℝ → ℝ) (x : ℝ) (dy_dx : ℝ → ℝ) (tangent_eq : ℝ → ℝ → Prop):
  (∀ x, y x = x^2 + Real.log x) →
  (∀ x, dy_dx x = (deriv y) x) →
  (dy_dx 1 = 3) →
  (tangent_eq x (y x) ↔ (3 * x - y x - 2 = 0)) →
  tangent_eq 1 (y 1) :=
by
  intros y_def dy_dx_def slope_at_1 tangent_line_char
  sorry

end tangent_line_equation_l980_98082


namespace func_value_sum_l980_98016

noncomputable def f (x : ℝ) : ℝ :=
  -x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2 + 1

theorem func_value_sum : f (1/2) + f (-1/2) = 2 :=
by
  sorry

end func_value_sum_l980_98016


namespace triangle_acute_angles_integer_solution_l980_98093

theorem triangle_acute_angles_integer_solution :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), (20 < x ∧ x < 27) ∧ (12 < x ∧ x < 36) ↔ (x = 21 ∨ x = 22 ∨ x = 23 ∨ x = 24 ∨ x = 25 ∨ x = 26) :=
by
  sorry

end triangle_acute_angles_integer_solution_l980_98093


namespace shaded_area_is_correct_l980_98075

-- Define the basic constants and areas
def grid_length : ℝ := 15
def grid_height : ℝ := 5
def total_grid_area : ℝ := grid_length * grid_height

def large_triangle_base : ℝ := 15
def large_triangle_height : ℝ := 3
def large_triangle_area : ℝ := 0.5 * large_triangle_base * large_triangle_height

def small_triangle_base : ℝ := 3
def small_triangle_height : ℝ := 4
def small_triangle_area : ℝ := 0.5 * small_triangle_base * small_triangle_height

-- Define the total shaded area
def shaded_area : ℝ := total_grid_area - large_triangle_area + small_triangle_area

-- Theorem stating that the shaded area is 58.5 square units
theorem shaded_area_is_correct : shaded_area = 58.5 := 
by 
  -- proof will be provided here
  sorry

end shaded_area_is_correct_l980_98075


namespace no_solutions_triples_l980_98030

theorem no_solutions_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a! + b^3 ≠ 18 + c^3 :=
by
  sorry

end no_solutions_triples_l980_98030
