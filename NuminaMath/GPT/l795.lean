import Mathlib

namespace NUMINAMATH_GPT_part1_part2_part3_l795_79557

open Set

variable (x : ℝ)

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 2 < x ∧ x < 10}

theorem part1 : A ∩ B = {x | 3 ≤ x ∧ x < 7} :=
sorry

theorem part2 : (Aᶜ : Set ℝ) = {x | x < 3 ∨ x ≥ 7} :=
sorry

theorem part3 : (A ∪ B)ᶜ = {x | x ≤ 2 ∨ x ≥ 10} :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l795_79557


namespace NUMINAMATH_GPT_scientific_notation_of_284000000_l795_79548

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_284000000_l795_79548


namespace NUMINAMATH_GPT_expression_in_multiply_form_l795_79507

def a : ℕ := 3 ^ 1005
def b : ℕ := 7 ^ 1006
def m : ℕ := 114337548

theorem expression_in_multiply_form : 
  (a + b)^2 - (a - b)^2 = m * 10 ^ 1006 :=
by
  sorry

end NUMINAMATH_GPT_expression_in_multiply_form_l795_79507


namespace NUMINAMATH_GPT_power_function_properties_l795_79567

theorem power_function_properties (α : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ α) 
    (h_point : f (1/2) = 2 ) :
    (∀ x : ℝ, f x = 1 / x) ∧ (∀ x : ℝ, 0 < x → (f x) < (f (x / 2))) ∧ (∀ x : ℝ, f (-x) = - (f x)) :=
by
  sorry

end NUMINAMATH_GPT_power_function_properties_l795_79567


namespace NUMINAMATH_GPT_probability_at_least_one_correct_l795_79533

theorem probability_at_least_one_correct :
  let p_a := 12 / 20
  let p_b := 8 / 20
  let prob_neither := (1 - p_a) * (1 - p_b)
  let prob_at_least_one := 1 - prob_neither
  prob_at_least_one = 19 / 25 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_correct_l795_79533


namespace NUMINAMATH_GPT_incenter_correct_l795_79550

variable (P Q R : Type) [AddCommGroup P] [Module ℝ P]
variable (p q r : ℝ)
variable (P_vec Q_vec R_vec : P)

noncomputable def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (p / (p + q + r), q / (p + q + r), r / (p + q + r))

theorem incenter_correct : 
  incenter_coordinates 8 10 6 = (1/3, 5/12, 1/4) := by
  sorry

end NUMINAMATH_GPT_incenter_correct_l795_79550


namespace NUMINAMATH_GPT_vector2d_propositions_l795_79555

-- Define the vector structure in ℝ²
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define the relation > on Vector2D
def Vector2D.gt (a1 a2 : Vector2D) : Prop :=
  a1.x > a2.x ∨ (a1.x = a2.x ∧ a1.y > a2.y)

-- Define vectors e1, e2, and 0
def e1 : Vector2D := ⟨ 1, 0 ⟩
def e2 : Vector2D := ⟨ 0, 1 ⟩
def zero : Vector2D := ⟨ 0, 0 ⟩

-- Define propositions
def prop1 : Prop := Vector2D.gt e1 e2 ∧ Vector2D.gt e2 zero
def prop2 (a1 a2 a3 : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt a2 a3 → Vector2D.gt a1 a3
def prop3 (a1 a2 a : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a1.x + a.x) (a1.y + a.y)) (Vector2D.mk (a2.x + a.x) (a2.y + a.y))
def prop4 (a a1 a2 : Vector2D) : Prop := Vector2D.gt a zero → Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a.x * a1.x + a.y * a1.y) (0)) (Vector2D.mk (a.x * a2.x + a.y * a2.y) 0)

-- Main theorem to prove
theorem vector2d_propositions : prop1 ∧ (∀ a1 a2 a3, prop2 a1 a2 a3) ∧ (∀ a1 a2 a, prop3 a1 a2 a) := 
by
  sorry

end NUMINAMATH_GPT_vector2d_propositions_l795_79555


namespace NUMINAMATH_GPT_geom_seq_11th_term_l795_79536

/-!
The fifth and eighth terms of a geometric sequence are -2 and -54, respectively. 
What is the 11th term of this progression?
-/
theorem geom_seq_11th_term {a : ℕ → ℤ} (r : ℤ) 
  (h1 : a 5 = -2) (h2 : a 8 = -54) 
  (h3 : ∀ n : ℕ, a (n + 3) = a n * r ^ 3) : 
  a 11 = -1458 :=
sorry

end NUMINAMATH_GPT_geom_seq_11th_term_l795_79536


namespace NUMINAMATH_GPT_least_n_froods_l795_79568

theorem least_n_froods (n : ℕ) : (∃ n, n ≥ 30 ∧ (n * (n + 1)) / 2 > 15 * n) ∧ (∀ m < 30, (m * (m + 1)) / 2 ≤ 15 * m) :=
sorry

end NUMINAMATH_GPT_least_n_froods_l795_79568


namespace NUMINAMATH_GPT_star_polygon_points_l795_79589

theorem star_polygon_points (p : ℕ) (ϕ : ℝ) :
  (∀ i : Fin p, ∃ Ci Di : ℝ, Ci = Di + 15) →
  (p * ϕ + p * (ϕ + 15) = 360) →
  p = 24 :=
by
  sorry

end NUMINAMATH_GPT_star_polygon_points_l795_79589


namespace NUMINAMATH_GPT_function_range_y_eq_1_div_x_minus_2_l795_79577

theorem function_range_y_eq_1_div_x_minus_2 (x : ℝ) : (∀ y : ℝ, y = 1 / (x - 2) ↔ x ∈ {x : ℝ | x ≠ 2}) :=
sorry

end NUMINAMATH_GPT_function_range_y_eq_1_div_x_minus_2_l795_79577


namespace NUMINAMATH_GPT_equation_solution_l795_79558

noncomputable def solve_equation : Set ℝ := {x : ℝ | (3 * x + 2) / (x ^ 2 + 5 * x + 6) = 3 * x / (x - 1)
                                             ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1}

theorem equation_solution (r : ℝ) (h : r ∈ solve_equation) : 3 * r ^ 3 + 12 * r ^ 2 + 19 * r + 2 = 0 :=
sorry

end NUMINAMATH_GPT_equation_solution_l795_79558


namespace NUMINAMATH_GPT_andrena_has_more_dolls_than_debelyn_l795_79572

-- Definitions based on the given conditions
def initial_dolls_debelyn := 20
def initial_gift_debelyn_to_andrena := 2

def initial_dolls_christel := 24
def gift_christel_to_andrena := 5
def gift_christel_to_belissa := 3

def initial_dolls_belissa := 15
def gift_belissa_to_andrena := 4

-- Final number of dolls after exchanges
def final_dolls_debelyn := initial_dolls_debelyn - initial_gift_debelyn_to_andrena
def final_dolls_christel := initial_dolls_christel - gift_christel_to_andrena - gift_christel_to_belissa
def final_dolls_belissa := initial_dolls_belissa - gift_belissa_to_andrena + gift_christel_to_belissa
def final_dolls_andrena := initial_gift_debelyn_to_andrena + gift_christel_to_andrena + gift_belissa_to_andrena

-- Additional conditions
def andrena_more_than_christel := final_dolls_andrena = final_dolls_christel + 2
def belissa_equals_debelyn := final_dolls_belissa = final_dolls_debelyn

-- Proof Statement
theorem andrena_has_more_dolls_than_debelyn :
  andrena_more_than_christel →
  belissa_equals_debelyn →
  final_dolls_andrena - final_dolls_debelyn = 4 :=
by
  sorry

end NUMINAMATH_GPT_andrena_has_more_dolls_than_debelyn_l795_79572


namespace NUMINAMATH_GPT_exam_correct_answers_l795_79588

theorem exam_correct_answers (C W : ℕ) 
  (h1 : C + W = 60)
  (h2 : 4 * C - W = 160) : 
  C = 44 :=
sorry

end NUMINAMATH_GPT_exam_correct_answers_l795_79588


namespace NUMINAMATH_GPT_percentage_of_40_l795_79537

theorem percentage_of_40 (P : ℝ) (h1 : 8/100 * 24 = 1.92) (h2 : P/100 * 40 + 1.92 = 5.92) : P = 10 :=
sorry

end NUMINAMATH_GPT_percentage_of_40_l795_79537


namespace NUMINAMATH_GPT_part1_part2_l795_79554

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a+1)*x + a

theorem part1 (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f a x < 0) → a ≤ -2 := sorry

theorem part2 (a x : ℝ) :
  f a x > 0 ↔
  (a > 1 ∧ (x < -a ∨ x > -1)) ∨
  (a = 1 ∧ x ≠ -1) ∨
  (a < 1 ∧ (x < -1 ∨ x > -a)) := sorry

end NUMINAMATH_GPT_part1_part2_l795_79554


namespace NUMINAMATH_GPT_linear_equation_m_value_l795_79512

theorem linear_equation_m_value (m : ℝ) (x : ℝ) (h : (m - 1) * x ^ |m| - 2 = 0) : m = -1 :=
sorry

end NUMINAMATH_GPT_linear_equation_m_value_l795_79512


namespace NUMINAMATH_GPT_height_percentage_difference_l795_79520

theorem height_percentage_difference (H : ℝ) (p r q : ℝ) 
  (hp : p = 0.60 * H) 
  (hr : r = 1.30 * H) : 
  (r - p) / p * 100 = 116.67 :=
by
  sorry

end NUMINAMATH_GPT_height_percentage_difference_l795_79520


namespace NUMINAMATH_GPT_lower_limit_for_x_l795_79518

variable {n : ℝ} {x : ℝ} {y : ℝ}

theorem lower_limit_for_x (h1 : x > n) (h2 : x < 8) (h3 : y > 8) (h4 : y < 13) (h5 : y - x = 7) : x = 2 :=
sorry

end NUMINAMATH_GPT_lower_limit_for_x_l795_79518


namespace NUMINAMATH_GPT_product_of_p_r_s_l795_79529

theorem product_of_p_r_s (p r s : ℕ) 
  (h1 : 4^p + 4^3 = 280)
  (h2 : 3^r + 29 = 56) 
  (h3 : 7^s + 6^3 = 728) : 
  p * r * s = 27 :=
by
  sorry

end NUMINAMATH_GPT_product_of_p_r_s_l795_79529


namespace NUMINAMATH_GPT_vacation_cost_division_l795_79510

theorem vacation_cost_division (n : ℕ) (h1 : 720 / 4 = 60 + 720 / n) : n = 3 := by
  sorry

end NUMINAMATH_GPT_vacation_cost_division_l795_79510


namespace NUMINAMATH_GPT_cost_per_square_meter_l795_79506

theorem cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (hlength : length = 20) 
  (hwidth : width = 15) 
  (hheight : height = 5) 
  (hmoney : total_expenditure = 38000) : 
  58.46 = total_expenditure / (length * width + 2 * length * height + 2 * width * height) :=
by 
  -- Let's assume our definitions and use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_cost_per_square_meter_l795_79506


namespace NUMINAMATH_GPT_opposite_of_neg_6_l795_79522

theorem opposite_of_neg_6 : ∀ (n : ℤ), n = -6 → -n = 6 :=
by
  intro n h
  rw [h]
  sorry

end NUMINAMATH_GPT_opposite_of_neg_6_l795_79522


namespace NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l795_79562

theorem opposite_of_2023_is_neg_2023 : (2023 + (-2023) = 0) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l795_79562


namespace NUMINAMATH_GPT_directrix_of_parabola_l795_79565

theorem directrix_of_parabola :
  ∀ (x : ℝ), y = x^2 / 4 → y = -1 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l795_79565


namespace NUMINAMATH_GPT_passed_in_both_subjects_l795_79578

theorem passed_in_both_subjects (A B C : ℝ)
  (hA : A = 0.25)
  (hB : B = 0.48)
  (hC : C = 0.27) :
  1 - (A + B - C) = 0.54 := by
  sorry

end NUMINAMATH_GPT_passed_in_both_subjects_l795_79578


namespace NUMINAMATH_GPT_tan_cos_identity_l795_79524

theorem tan_cos_identity :
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  (tanθ^2 - cosθ^2) / (tanθ^2 * cosθ^2) = -5 / 3 :=
by
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  have h_tan : tanθ^2 = (Real.sin θ)^2 / (Real.cos θ)^2 := by sorry; -- Given condition 1
  have h_cos : cosθ^2 = 3 / 4 := by sorry; -- Given condition 2
  -- Prove the statement
  sorry

end NUMINAMATH_GPT_tan_cos_identity_l795_79524


namespace NUMINAMATH_GPT_find_alpha_l795_79543

noncomputable section

open Real 

def curve_C1 (x y : ℝ) : Prop := x + y = 1
def curve_C2 (x y φ : ℝ) : Prop := x = 2 + 2 * cos φ ∧ y = 2 * sin φ 

def polar_coordinate_eq1 (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = sqrt 2 / 2
def polar_coordinate_eq2 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ

def line_l (ρ θ α : ℝ)  (hα: α > 0 ∧ α < π / 2) : Prop := θ = α ∧ ρ > 0 

def OB_div_OA_eq_4 (ρA ρB α : ℝ) : Prop := ρB / ρA = 4

theorem find_alpha (α : ℝ) (hα: α > 0 ∧ α < π / 2)
  (h₁: ∀ (x y ρ θ: ℝ), curve_C1 x y → polar_coordinate_eq1 ρ θ) 
  (h₂: ∀ (x y φ ρ θ: ℝ), curve_C2 x y φ → polar_coordinate_eq2 ρ θ) 
  (h₃: ∀ (ρ θ: ℝ), line_l ρ θ α hα) 
  (h₄: ∀ (ρA ρB : ℝ), OB_div_OA_eq_4 ρA ρB α → ρA = 1 / (cos α + sin α) ∧ ρB = 4 * cos α ): 
  α = 3 * π / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_l795_79543


namespace NUMINAMATH_GPT_tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l795_79584

-- Define the function f
noncomputable def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 3

-- First proof problem: Equation of the tangent line at (2, 7)
theorem tangent_line_at_2_is_12x_minus_y_minus_17_eq_0 :
  ∀ x y : ℝ, y = f x → (x = 2) → y = 7 → (∃ (m b : ℝ), (m = 12) ∧ (b = -17) ∧ (∀ x, 12 * x - y - 17 = 0)) :=
by
  sorry

-- Second proof problem: Range of m for three distinct real roots
theorem range_of_m_for_three_distinct_real_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) → -3 < m ∧ m < -2 :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_at_2_is_12x_minus_y_minus_17_eq_0_range_of_m_for_three_distinct_real_roots_l795_79584


namespace NUMINAMATH_GPT_find_salary_l795_79571

def salary_remaining (S : ℝ) (food : ℝ) (house_rent : ℝ) (clothes : ℝ) (remaining : ℝ) : Prop :=
  S - food * S - house_rent * S - clothes * S = remaining

theorem find_salary :
  ∀ S : ℝ, 
  salary_remaining S (1/5) (1/10) (3/5) 15000 → 
  S = 150000 :=
by
  intros S h
  sorry

end NUMINAMATH_GPT_find_salary_l795_79571


namespace NUMINAMATH_GPT_volleyball_match_probabilities_l795_79511

noncomputable def probability_of_team_A_winning : ℚ := (2 / 3) ^ 3
noncomputable def probability_of_team_B_winning_3_0 : ℚ := 1 / 3
noncomputable def probability_of_team_B_winning_3_1 : ℚ := (2 / 3) * (1 / 3)
noncomputable def probability_of_team_B_winning_3_2 : ℚ := (2 / 3) ^ 2 * (1 / 3)

theorem volleyball_match_probabilities :
  probability_of_team_A_winning = 8 / 27 ∧
  probability_of_team_B_winning_3_0 = 1 / 3 ∧
  probability_of_team_B_winning_3_1 ≠ 1 / 9 ∧
  probability_of_team_B_winning_3_2 ≠ 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_match_probabilities_l795_79511


namespace NUMINAMATH_GPT_initial_overs_l795_79576

theorem initial_overs {x : ℝ} (h1 : 4.2 * x + (83 / 15) * 30 = 250) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_overs_l795_79576


namespace NUMINAMATH_GPT_triangle_inequality_inequality_l795_79528

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 > 0 := 
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_inequality_l795_79528


namespace NUMINAMATH_GPT_relationship_abc_l795_79580

theorem relationship_abc (a b c : ℝ) 
  (h₁ : a = Real.log 0.5 / Real.log 2) 
  (h₂ : b = Real.sqrt 2) 
  (h₃ : c = 0.5 ^ 2) : 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_relationship_abc_l795_79580


namespace NUMINAMATH_GPT_quadratic_equation_from_absolute_value_l795_79563

theorem quadratic_equation_from_absolute_value :
  ∃ b c : ℝ, (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b * x + c = 0) ∧ (b, c) = (-16, 55) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_from_absolute_value_l795_79563


namespace NUMINAMATH_GPT_soccer_team_games_played_l795_79516

theorem soccer_team_games_played 
  (players : ℕ) (total_goals : ℕ) (third_players_goals_per_game : ℕ → ℕ) (other_players_goals : ℕ) (G : ℕ)
  (h1 : players = 24)
  (h2 : total_goals = 150)
  (h3 : ∃ n, n = players / 3 ∧ ∀ g, third_players_goals_per_game g = n * g)
  (h4 : other_players_goals = 30)
  (h5 : total_goals = third_players_goals_per_game G + other_players_goals) :
  G = 15 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_soccer_team_games_played_l795_79516


namespace NUMINAMATH_GPT_fraction_sum_l795_79547

theorem fraction_sum : ((10 : ℚ) / 9 + (9 : ℚ) / 10 = 2.0 + (0.1 + 0.1 / 9)) :=
by sorry

end NUMINAMATH_GPT_fraction_sum_l795_79547


namespace NUMINAMATH_GPT_housewife_oil_cost_l795_79574

theorem housewife_oil_cost (P R M : ℝ) (hR : R = 45) (hReduction : (P - R) = (15 / 100) * P)
  (hMoreOil : M / P = M / R + 4) : M = 150.61 := 
by
  sorry

end NUMINAMATH_GPT_housewife_oil_cost_l795_79574


namespace NUMINAMATH_GPT_sum_of_digits_smallest_N_l795_79564

theorem sum_of_digits_smallest_N :
  ∃ (N : ℕ), N ≤ 999 ∧ 72 * N < 1000 ∧ (N = 13) ∧ (1 + 3 = 4) := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_smallest_N_l795_79564


namespace NUMINAMATH_GPT_avg_speed_while_climbing_l795_79527

-- Definitions for conditions
def totalClimbTime : ℝ := 4
def restBreaks : ℝ := 0.5
def descentTime : ℝ := 2
def avgSpeedWholeJourney : ℝ := 1.5
def totalDistance : ℝ := avgSpeedWholeJourney * (totalClimbTime + descentTime)

-- The question: Prove Natasha's average speed while climbing to the top, excluding the rest breaks duration.
theorem avg_speed_while_climbing :
  (totalDistance / 2) / (totalClimbTime - restBreaks) = 1.29 := 
sorry

end NUMINAMATH_GPT_avg_speed_while_climbing_l795_79527


namespace NUMINAMATH_GPT_batsman_average_l795_79583

variable (x : ℝ)

theorem batsman_average (h1 : ∀ x, 11 * x + 55 = 12 * (x + 1)) : 
  x = 43 → (x + 1 = 44) :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_l795_79583


namespace NUMINAMATH_GPT_find_number_and_remainder_l795_79546

theorem find_number_and_remainder :
  ∃ (N r : ℕ), (3927 + 2873) * (3 * (3927 - 2873)) + r = N ∧ r < (3927 + 2873) :=
sorry

end NUMINAMATH_GPT_find_number_and_remainder_l795_79546


namespace NUMINAMATH_GPT_simplify_evaluate_expression_l795_79549

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -2) (h2 : b = 4) : 
  (-(3 * a)^2 + 6 * a * b - (a^2 + 3 * (a - 2 * a * b))) = 14 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expression_l795_79549


namespace NUMINAMATH_GPT_total_women_attendees_l795_79508

theorem total_women_attendees 
  (adults : ℕ) (adult_women : ℕ) (student_offset : ℕ) (total_students : ℕ)
  (male_students : ℕ) :
  adults = 1518 →
  adult_women = 536 →
  student_offset = 525 →
  total_students = adults + student_offset →
  total_students = 2043 →
  male_students = 1257 →
  (adult_women + (total_students - male_students) = 1322) :=
by
  sorry

end NUMINAMATH_GPT_total_women_attendees_l795_79508


namespace NUMINAMATH_GPT_wolves_heads_count_l795_79502

/-- 
A person goes hunting in the jungle and discovers a pack of wolves.
It is known that this person has one head and two legs, 
an ordinary wolf has one head and four legs, and a mutant wolf has two heads and three legs.
The total number of heads of all the people and wolves combined is 21,
and the total number of legs is 57.
-/
theorem wolves_heads_count :
  ∃ (x y : ℕ), (x + 2 * y = 20) ∧ (4 * x + 3 * y = 55) ∧ (x + y > 0) ∧ (x + 2 * y + 1 = 21) ∧ (4 * x + 3 * y + 2 = 57) := 
by {
  sorry
}

end NUMINAMATH_GPT_wolves_heads_count_l795_79502


namespace NUMINAMATH_GPT_find_y_l795_79594

noncomputable def x : ℝ := (4 / 25)^(1 / 3)

theorem find_y (y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x^x = y^y) : y = (32 / 3125)^(1 / 3) :=
sorry

end NUMINAMATH_GPT_find_y_l795_79594


namespace NUMINAMATH_GPT_probability_three_white_two_black_l795_79534

-- Define the total number of balls
def total_balls : ℕ := 17

-- Define the number of white balls
def white_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 9

-- Define the number of balls drawn
def balls_drawn : ℕ := 5

-- Define three white balls drawn
def three_white_drawn : ℕ := 3

-- Define two black balls drawn
def two_black_drawn : ℕ := 2

-- Define the combination formula
noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Calculate the probability
noncomputable def probability : ℚ :=
  (combination white_balls three_white_drawn * combination black_balls two_black_drawn : ℚ) 
  / combination total_balls balls_drawn

-- Statement to prove
theorem probability_three_white_two_black :
  probability = 672 / 2063 := by
  sorry

end NUMINAMATH_GPT_probability_three_white_two_black_l795_79534


namespace NUMINAMATH_GPT_greater_number_l795_79586

theorem greater_number (x y : ℕ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := 
by 
  sorry

end NUMINAMATH_GPT_greater_number_l795_79586


namespace NUMINAMATH_GPT_exists_increasing_triplet_l795_79591

theorem exists_increasing_triplet (f : ℕ → ℕ) (bij : Function.Bijective f) :
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ f a < f (a + d) ∧ f (a + d) < f (a + 2 * d) :=
by
  sorry

end NUMINAMATH_GPT_exists_increasing_triplet_l795_79591


namespace NUMINAMATH_GPT_park_area_is_correct_l795_79532

-- Define the side of the square
def side_length : ℕ := 30

-- Define the area function for a square
def area_of_square (side: ℕ) : ℕ := side * side

-- State the theorem we're going to prove
theorem park_area_is_correct : area_of_square side_length = 900 := 
sorry -- proof not required

end NUMINAMATH_GPT_park_area_is_correct_l795_79532


namespace NUMINAMATH_GPT_carousel_seats_count_l795_79559

theorem carousel_seats_count :
  ∃ (yellow blue red : ℕ), 
  (yellow + blue + red = 100) ∧ 
  (yellow = 34) ∧ 
  (blue = 20) ∧ 
  (red = 46) ∧ 
  (∀ i : ℕ, i < yellow → ∃ j : ℕ, j = yellow.succ * j ∧ (j < 100 ∧ j ≠ yellow.succ * j)) ∧ 
  (∀ k : ℕ, k < blue → ∃ m : ℕ, m = blue.succ * m ∧ (m < 100 ∧ m ≠ blue.succ * m)) ∧ 
  (∀ n : ℕ, n < red → ∃ p : ℕ, p = red.succ * p ∧ (p < 100 ∧ p ≠ red.succ * p)) :=
sorry

end NUMINAMATH_GPT_carousel_seats_count_l795_79559


namespace NUMINAMATH_GPT_find_a_plus_b_l795_79595

def cubic_function (a b : ℝ) (x : ℝ) := x^3 - x^2 - a * x + b

def tangent_line (x : ℝ) := 2 * x + 1

theorem find_a_plus_b (a b : ℝ) 
  (h1 : tangent_line 0 = 1)
  (h2 : cubic_function a b 0 = 1)
  (h3 : deriv (cubic_function a b) 0 = 2) :
  a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l795_79595


namespace NUMINAMATH_GPT_min_a_b_l795_79569

theorem min_a_b : 
  (∀ x : ℝ, 3 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) ≤ 3) →
  a + b = -2 →
  a = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_min_a_b_l795_79569


namespace NUMINAMATH_GPT_circle_equation_proof_l795_79505

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Define a predicate for the circle being tangent to the y-axis
def tangent_y_axis (center : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r = abs center.1

-- Define the equation of the circle given center and radius
def circle_eqn (center : ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2

-- State the theorem
theorem circle_equation_proof :
  tangent_y_axis circle_center →
  ∃ r, r = 1 ∧ circle_eqn circle_center r :=
sorry

end NUMINAMATH_GPT_circle_equation_proof_l795_79505


namespace NUMINAMATH_GPT_evaluate_tensor_expression_l795_79542

-- Define the tensor operation
def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- The theorem we want to prove
theorem evaluate_tensor_expression : tensor (tensor 5 3) 2 = 293 / 15 := by
  sorry

end NUMINAMATH_GPT_evaluate_tensor_expression_l795_79542


namespace NUMINAMATH_GPT_add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l795_79553

theorem add_neg_eq_neg_add (a b : Int) : a + -b = a - b := by
  sorry

theorem neg_ten_plus_neg_twelve : -10 + (-12) = -22 := by
  have h1 : -10 + (-12) = -10 - 12 := add_neg_eq_neg_add _ _
  have h2 : -10 - 12 = -(10 + 12) := by
    sorry -- This step corresponds to recognizing the arithmetic rule for subtraction.
  have h3 : -(10 + 12) = -22 := by
    sorry -- This step is the concrete calculation.
  exact Eq.trans h1 (Eq.trans h2 h3)

end NUMINAMATH_GPT_add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l795_79553


namespace NUMINAMATH_GPT_men_with_tv_at_least_11_l795_79593

-- Definitions for the given conditions
def total_men : ℕ := 100
def married_men : ℕ := 81
def men_with_radio : ℕ := 85
def men_with_ac : ℕ := 70
def men_with_tv_radio_ac_and_married : ℕ := 11

-- The proposition to prove the minimum number of men with TV
theorem men_with_tv_at_least_11 :
  ∃ (T : ℕ), T ≥ men_with_tv_radio_ac_and_married := 
by
  sorry

end NUMINAMATH_GPT_men_with_tv_at_least_11_l795_79593


namespace NUMINAMATH_GPT_angle_supplement_complement_l795_79541

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end NUMINAMATH_GPT_angle_supplement_complement_l795_79541


namespace NUMINAMATH_GPT_space_convex_polyhedron_euler_characteristic_l795_79592

-- Definition of space convex polyhedron
structure Polyhedron where
  F : ℕ    -- number of faces
  V : ℕ    -- number of vertices
  E : ℕ    -- number of edges

-- Problem statement: Prove that for any space convex polyhedron, F + V - E = 2
theorem space_convex_polyhedron_euler_characteristic (P : Polyhedron) : P.F + P.V - P.E = 2 := by
  sorry

end NUMINAMATH_GPT_space_convex_polyhedron_euler_characteristic_l795_79592


namespace NUMINAMATH_GPT_find_m_value_l795_79556

theorem find_m_value (m : ℚ) :
  (m * 2 / 3 + m * 4 / 9 + m * 8 / 27 = 1) → m = 27 / 38 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_m_value_l795_79556


namespace NUMINAMATH_GPT_gcd_168_486_l795_79526

theorem gcd_168_486 : gcd 168 486 = 6 := 
by sorry

end NUMINAMATH_GPT_gcd_168_486_l795_79526


namespace NUMINAMATH_GPT_magic_square_exists_l795_79598

theorem magic_square_exists : 
  ∃ (a b c d e f g h : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ 
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c = 12 ∧ d + e + f = 12 ∧ g + h + 0 = 12 ∧
    a + d + g = 12 ∧ b + 0 + h = 12 ∧ c + f + 0 = 12 :=
sorry

end NUMINAMATH_GPT_magic_square_exists_l795_79598


namespace NUMINAMATH_GPT_no_pos_int_solutions_l795_79539

theorem no_pos_int_solutions (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + 2^(2 * k) + 1 ≠ y^3 := by
  sorry

end NUMINAMATH_GPT_no_pos_int_solutions_l795_79539


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l795_79566

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}

-- Given two arithmetic sequences a_n and b_n, their sums of the first n terms are S_n and T_n respectively.
-- Given that S_n / T_n = (2n + 2) / (n + 3).
-- Prove that a_10 / b_10 = 20 / 11.

theorem arithmetic_sequence_ratio (h : ∀ n, S_n n / T_n n = (2 * n + 2) / (n + 3)) : (a_n 10) / (b_n 10) = 20 / 11 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l795_79566


namespace NUMINAMATH_GPT_sum_R1_R2_eq_19_l795_79585

-- Definitions for F_1 and F_2 in base R_1 and R_2
def F1_R1 : ℚ := 37 / 99
def F2_R1 : ℚ := 73 / 99
def F1_R2 : ℚ := 25 / 99
def F2_R2 : ℚ := 52 / 99

-- Prove that the sum of R1 and R2 is 19
theorem sum_R1_R2_eq_19 (R1 R2 : ℕ) (hF1R1 : F1_R1 = (3 * R1 + 7) / (R1^2 - 1))
  (hF2R1 : F2_R1 = (7 * R1 + 3) / (R1^2 - 1))
  (hF1R2 : F1_R2 = (2 * R2 + 5) / (R2^2 - 1))
  (hF2R2 : F2_R2 = (5 * R2 + 2) / (R2^2 - 1)) :
  R1 + R2 = 19 :=
  sorry

end NUMINAMATH_GPT_sum_R1_R2_eq_19_l795_79585


namespace NUMINAMATH_GPT_probability_at_most_six_distinct_numbers_l795_79517

def roll_eight_dice : ℕ := 6^8

def favorable_cases : ℕ := 3628800

def probability_six_distinct_numbers (n : ℕ) (f : ℕ) : ℚ :=
  f / n

theorem probability_at_most_six_distinct_numbers :
  probability_six_distinct_numbers roll_eight_dice favorable_cases = 45 / 52 := by
  sorry

end NUMINAMATH_GPT_probability_at_most_six_distinct_numbers_l795_79517


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l795_79515

theorem solve_eq1 (x : ℝ) : 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : x^2 + 4 * x + 3 = 0 ↔ x = -3 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l795_79515


namespace NUMINAMATH_GPT_quadratic_properties_l795_79538

noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x => a * x^2 + b * x + c

def min_value_passing_point (f : ℝ → ℝ) : Prop :=
  (f (-1) = -4) ∧ (f 0 = -3)

def intersects_x_axis (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  (f p1.1 = p1.2) ∧ (f p2.1 = p2.2)

def max_value_in_interval (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ max_val

theorem quadratic_properties :
  ∃ f : ℝ → ℝ,
    min_value_passing_point f ∧
    intersects_x_axis f (1, 0) (-3, 0) ∧
    max_value_in_interval f (-2) 2 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_properties_l795_79538


namespace NUMINAMATH_GPT_valid_parameterizations_l795_79540

-- Define the parameterization as a structure
structure LineParameterization where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

-- Define the line equation
def line_eq (p : ℝ × ℝ) : Prop :=
  p.snd = -(2/3) * p.fst + 4

-- Proving which parameterizations are valid
theorem valid_parameterizations :
  (line_eq (3 + t * 3, 4 + t * (-2)) ∧
   line_eq (0 + t * 1.5, 4 + t * (-1)) ∧
   line_eq (1 + t * (-6), 3.33 + t * 4) ∧
   line_eq (5 + t * 1.5, (2/3) + t * (-1)) ∧
   line_eq (-6 + t * 9, 8 + t * (-6))) = 
  false ∧ true ∧ false ∧ true ∧ false :=
by
  sorry

end NUMINAMATH_GPT_valid_parameterizations_l795_79540


namespace NUMINAMATH_GPT_sequence_properties_l795_79535

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 3 ∧ ∀ n, a (n + 1) = a n + 2

theorem sequence_properties {a : ℕ → ℤ} (h : arithmetic_sequence a) :
  a 2 + a 4 = 6 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l795_79535


namespace NUMINAMATH_GPT_initial_population_l795_79545

theorem initial_population (P : ℝ) (h1 : ∀ t : ℕ, P * (1.10 : ℝ) ^ t = 26620 → t = 3) : P = 20000 := by
  have h2 : P * (1.10) ^ 3 = 26620 := sorry
  sorry

end NUMINAMATH_GPT_initial_population_l795_79545


namespace NUMINAMATH_GPT_volume_of_given_cuboid_l795_79509

-- Definition of the function to compute the volume of a cuboid
def volume_of_cuboid (length width height : ℝ) : ℝ :=
  length * width * height

-- Given conditions and the proof target
theorem volume_of_given_cuboid : volume_of_cuboid 2 5 3 = 30 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_given_cuboid_l795_79509


namespace NUMINAMATH_GPT_number_of_arrangements_l795_79531

theorem number_of_arrangements (A B : Type) (individuals : Fin 6 → Type)
  (adjacent_condition : ∃ (i : Fin 5), individuals i = B ∧ individuals (i + 1) = A) :
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l795_79531


namespace NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_part1_problem2_part2_l795_79552

open Set

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | -2 < x ∧ x < 5 }
def B : Set ℝ := { x | -1 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem1_part1 : A ∪ B = { x | -2 < x ∧ x < 5 } := sorry
theorem problem1_part2 : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } := sorry

def B_c : Set ℝ := { x | x < 0 ∨ 3 < x }

theorem problem2_part1 : A ∪ B_c = U := sorry
theorem problem2_part2 : A ∩ B_c = { x | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 5) } := sorry

end NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_part1_problem2_part2_l795_79552


namespace NUMINAMATH_GPT_avg_fish_in_bodies_of_water_l795_79579

def BoastPoolFish : ℕ := 75
def OnumLakeFish : ℕ := BoastPoolFish + 25
def RiddlePondFish : ℕ := OnumLakeFish / 2
def RippleCreekFish : ℕ := 2 * (OnumLakeFish - BoastPoolFish)
def WhisperingSpringsFish : ℕ := (3 * RiddlePondFish) / 2

def totalFish : ℕ := BoastPoolFish + OnumLakeFish + RiddlePondFish + RippleCreekFish + WhisperingSpringsFish
def averageFish : ℕ := totalFish / 5

theorem avg_fish_in_bodies_of_water : averageFish = 68 :=
by
  sorry

end NUMINAMATH_GPT_avg_fish_in_bodies_of_water_l795_79579


namespace NUMINAMATH_GPT_jen_age_proof_l795_79570

variable (JenAge : ℕ) (SonAge : ℕ)

theorem jen_age_proof (h1 : SonAge = 16) (h2 : JenAge = 3 * SonAge - 7) : JenAge = 41 :=
by
  -- conditions
  rw [h1] at h2
  -- substitution and simplification
  have h3 : JenAge = 3 * 16 - 7 := h2
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_jen_age_proof_l795_79570


namespace NUMINAMATH_GPT_evaluate_expression_l795_79513

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l795_79513


namespace NUMINAMATH_GPT_problem_gets_solved_prob_l795_79525

-- Define conditions for probabilities
def P_A_solves := 2 / 3
def P_B_solves := 3 / 4

-- Calculate the probability that the problem is solved
theorem problem_gets_solved_prob :
  let P_A_not_solves := 1 - P_A_solves
  let P_B_not_solves := 1 - P_B_solves
  let P_both_not_solve := P_A_not_solves * P_B_not_solves
  let P_solved := 1 - P_both_not_solve
  P_solved = 11 / 12 :=
by
  -- Skip proof
  sorry

end NUMINAMATH_GPT_problem_gets_solved_prob_l795_79525


namespace NUMINAMATH_GPT_costOfBrantsRoyalBananaSplitSundae_l795_79599

-- Define constants for the prices of the known sundaes
def yvette_sundae_cost : ℝ := 9.00
def alicia_sundae_cost : ℝ := 7.50
def josh_sundae_cost : ℝ := 8.50

-- Define the tip percentage
def tip_percentage : ℝ := 0.20

-- Define the final bill amount
def final_bill : ℝ := 42.00

-- Calculate the total known sundaes cost
def total_known_sundaes_cost : ℝ := yvette_sundae_cost + alicia_sundae_cost + josh_sundae_cost

-- Define a proof to show that the cost of Brant's sundae is $10.00
theorem costOfBrantsRoyalBananaSplitSundae : 
  total_known_sundaes_cost + b = final_bill / (1 + tip_percentage) → b = 10 :=
sorry

end NUMINAMATH_GPT_costOfBrantsRoyalBananaSplitSundae_l795_79599


namespace NUMINAMATH_GPT_ratio_of_horns_l795_79544

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_harps := 0

def total_instruments := 7

def charlie_instruments := charlie_flutes + charlie_horns + charlie_harps
def carli_instruments := total_instruments - charlie_instruments

def carli_horns := carli_instruments - carli_flutes

theorem ratio_of_horns : (carli_horns : ℚ) / charlie_horns = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_horns_l795_79544


namespace NUMINAMATH_GPT_no_valid_solutions_l795_79582

theorem no_valid_solutions (x : ℝ) (h : x ≠ 1) : 
  ¬(3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) :=
sorry

end NUMINAMATH_GPT_no_valid_solutions_l795_79582


namespace NUMINAMATH_GPT_jessica_earned_from_washing_l795_79590

-- Conditions defined as per Problem a)
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def remaining_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 11
def earned_from_washing : ℕ := final_amount - remaining_after_movies

-- Lean statement to prove Jessica earned $6 from washing the family car
theorem jessica_earned_from_washing :
  earned_from_washing = 6 := 
by
  -- Proof to be filled in later (skipped here with sorry)
  sorry

end NUMINAMATH_GPT_jessica_earned_from_washing_l795_79590


namespace NUMINAMATH_GPT_tangent_line_a_value_l795_79523

theorem tangent_line_a_value (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y - 1 = 0 → x^2 + y^2 + 4 * x = 0) → a = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_a_value_l795_79523


namespace NUMINAMATH_GPT_cylinder_volume_eq_sphere_volume_l795_79551

theorem cylinder_volume_eq_sphere_volume (a h R x : ℝ) (h_pos : h > 0) (a_pos : a > 0) (R_pos : R > 0)
  (h_volume_eq : (a - h) * x^2 - a * h * x + 2 * h * R^2 = 0) :
  ∃ x : ℝ, a > h ∧ x > 0 ∧ x < h ∧ x = 2 * R^2 / a ∨ 
           h < a ∧ 0 < x ∧ x = (a * h / (a - h)) - h ∧ R^2 < h^2 / 2 :=
sorry

end NUMINAMATH_GPT_cylinder_volume_eq_sphere_volume_l795_79551


namespace NUMINAMATH_GPT_white_marbles_bagA_eq_fifteen_l795_79503

noncomputable def red_marbles_bagA := 5
def rw_ratio_bagA := (1, 3)
def wb_ratio_bagA := (2, 3)

theorem white_marbles_bagA_eq_fifteen :
  let red_to_white := rw_ratio_bagA.1 * red_marbles_bagA
  red_to_white * rw_ratio_bagA.2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_white_marbles_bagA_eq_fifteen_l795_79503


namespace NUMINAMATH_GPT_complement_intersection_l795_79575

def P : Set ℝ := {y | ∃ x, y = (1 / 2) ^ x ∧ 0 < x}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_intersection :
  (Set.univ \ P) ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l795_79575


namespace NUMINAMATH_GPT_total_questions_solved_l795_79501

-- Define the number of questions Taeyeon solved in a day and the number of days
def Taeyeon_questions_per_day : ℕ := 16
def Taeyeon_days : ℕ := 7

-- Define the number of questions Yura solved in a day and the number of days
def Yura_questions_per_day : ℕ := 25
def Yura_days : ℕ := 6

-- Define the total number of questions Taeyeon and Yura solved
def Total_questions_Taeyeon : ℕ := Taeyeon_questions_per_day * Taeyeon_days
def Total_questions_Yura : ℕ := Yura_questions_per_day * Yura_days
def Total_questions : ℕ := Total_questions_Taeyeon + Total_questions_Yura

-- Prove that the total number of questions solved by Taeyeon and Yura is 262
theorem total_questions_solved : Total_questions = 262 := by
  sorry

end NUMINAMATH_GPT_total_questions_solved_l795_79501


namespace NUMINAMATH_GPT_solution_set_of_inequality_l795_79500

theorem solution_set_of_inequality (x : ℝ) : |2 * x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l795_79500


namespace NUMINAMATH_GPT_value_of_k_l795_79561

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k * x + 7

theorem value_of_k (k : ℝ) : f 5 - g 5 k = 40 → k = 1.4 := by
  sorry

end NUMINAMATH_GPT_value_of_k_l795_79561


namespace NUMINAMATH_GPT_most_likely_outcome_is_D_l795_79560

-- Define the basic probability of rolling any specific number with a fair die
def probability_of_specific_roll : ℚ := 1/6

-- Define the probability of each option
def P_A : ℚ := probability_of_specific_roll
def P_B : ℚ := 2 * probability_of_specific_roll
def P_C : ℚ := 3 * probability_of_specific_roll
def P_D : ℚ := 4 * probability_of_specific_roll

-- Define the proof problem statement
theorem most_likely_outcome_is_D : P_D = max P_A (max P_B (max P_C P_D)) :=
sorry

end NUMINAMATH_GPT_most_likely_outcome_is_D_l795_79560


namespace NUMINAMATH_GPT_skittles_total_correct_l795_79519

def number_of_students : ℕ := 9
def skittles_per_student : ℕ := 3
def total_skittles : ℕ := 27

theorem skittles_total_correct : number_of_students * skittles_per_student = total_skittles := by
  sorry

end NUMINAMATH_GPT_skittles_total_correct_l795_79519


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l795_79587

theorem isosceles_triangle_base_angle (α β γ : ℝ) 
  (h_triangle: α + β + γ = 180) 
  (h_isosceles: α = β ∨ α = γ ∨ β = γ) 
  (h_one_angle: α = 80 ∨ β = 80 ∨ γ = 80) : 
  (α = 50 ∨ β = 50 ∨ γ = 50) ∨ (α = 80 ∨ β = 80 ∨ γ = 80) :=
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l795_79587


namespace NUMINAMATH_GPT_interval_of_increase_logb_l795_79581

noncomputable def f (x : ℝ) := Real.logb 5 (2 * x + 1)

-- Define the domain
def domain : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the interval of monotonic increase for the function
def interval_of_increase (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x < y → f x < f y}

-- Statement of the problem
theorem interval_of_increase_logb :
  interval_of_increase f = {x | x > - (1 / 2)} :=
by
  have h_increase : ∀ x y, x < y → f x < f y := sorry
  exact sorry

end NUMINAMATH_GPT_interval_of_increase_logb_l795_79581


namespace NUMINAMATH_GPT_point_coordinates_l795_79514

theorem point_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : abs y = 5) (h4 : abs x = 2) : x = -2 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_l795_79514


namespace NUMINAMATH_GPT_tan_11_pi_over_4_l795_79521

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_tan_11_pi_over_4_l795_79521


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l795_79573

def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

theorem instantaneous_velocity_at_2 : 
  (deriv s 2) = 29 :=
by
  -- The proof is skipped by using sorry
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l795_79573


namespace NUMINAMATH_GPT_max_c_for_log_inequality_l795_79596

theorem max_c_for_log_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / (3 + Real.log b / Real.log a) + 1 / (3 + Real.log a / Real.log b) ≥ c) :=
by
  use 1 / 3
  sorry

end NUMINAMATH_GPT_max_c_for_log_inequality_l795_79596


namespace NUMINAMATH_GPT_monotonicity_intervals_range_of_m_l795_79504

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem monotonicity_intervals (m : ℝ) (x : ℝ) (hx : x > 1):
  (m >= 1 → ∀ x' > 1, f m x' ≤ f m x) ∧
  (m < 1 → (∀ x' ∈ Set.Ioo 1 (Real.exp (1 - m)), f m x' > f m x) ∧
            (∀ x' ∈ Set.Ioi (Real.exp (1 - m)), f m x' < f m x)) := by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x > 1, f m x < m * x) ↔ m ≥ 1/2 := by
  sorry

end NUMINAMATH_GPT_monotonicity_intervals_range_of_m_l795_79504


namespace NUMINAMATH_GPT_binomial_fermat_l795_79597

theorem binomial_fermat (p : ℕ) (a b : ℤ) (hp : p.Prime) : 
  ((a + b)^p - a^p - b^p) % p = 0 := by
  sorry

end NUMINAMATH_GPT_binomial_fermat_l795_79597


namespace NUMINAMATH_GPT_perimeter_of_triangle_l795_79530

-- Defining the basic structure of the problem
theorem perimeter_of_triangle (A B C : Type)
  (distance_AB distance_AC distance_BC : ℝ)
  (angle_B : ℝ)
  (h1 : distance_AB = distance_AC)
  (h2 : angle_B = 60)
  (h3 : distance_BC = 4) :
  distance_AB + distance_AC + distance_BC = 12 :=
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l795_79530
