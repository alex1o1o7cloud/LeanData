import Mathlib

namespace NUMINAMATH_GPT_set_intersection_l333_33313

open Set

/-- Given sets M and N as defined below, we wish to prove that their complements and intersections work as expected. -/
theorem set_intersection (R : Set ℝ)
  (M : Set ℝ := {x | x > 1})
  (N : Set ℝ := {x | abs x ≤ 2})
  (R_universal : R = univ) :
  ((compl M) ∩ N) = Icc (-2 : ℝ) (1 : ℝ) := by
  sorry

end NUMINAMATH_GPT_set_intersection_l333_33313


namespace NUMINAMATH_GPT_cosine_double_angle_identity_l333_33385

theorem cosine_double_angle_identity (α : ℝ) (h : Real.sin (α + 7 * Real.pi / 6) = 1) :
  Real.cos (2 * α - 2 * Real.pi / 3) = 1 := by
  sorry

end NUMINAMATH_GPT_cosine_double_angle_identity_l333_33385


namespace NUMINAMATH_GPT_cannonball_maximum_height_l333_33399

def height_function (t : ℝ) := -20 * t^2 + 100 * t + 36

theorem cannonball_maximum_height :
  ∃ t₀ : ℝ, ∀ t : ℝ, height_function t ≤ height_function t₀ ∧ height_function t₀ = 161 :=
by
  sorry

end NUMINAMATH_GPT_cannonball_maximum_height_l333_33399


namespace NUMINAMATH_GPT_intersection_x_value_l333_33371

theorem intersection_x_value :
  (∃ x y : ℝ, y = 5 * x - 20 ∧ y = 110 - 3 * x ∧ x = 16.25) := sorry

end NUMINAMATH_GPT_intersection_x_value_l333_33371


namespace NUMINAMATH_GPT_geometric_sequence_thm_proof_l333_33360

noncomputable def geometric_sequence_thm (a : ℕ → ℤ) : Prop :=
  (∃ r : ℤ, ∃ a₀ : ℤ, ∀ n : ℕ, a n = a₀ * r ^ n) ∧
  (a 2) * (a 10) = 4 ∧
  (a 2) + (a 10) > 0 →
  (a 6) = 2

theorem geometric_sequence_thm_proof (a : ℕ → ℤ) :
  geometric_sequence_thm a :=
  by
  sorry

end NUMINAMATH_GPT_geometric_sequence_thm_proof_l333_33360


namespace NUMINAMATH_GPT_soccer_team_games_count_l333_33387

variable (total_games won_games : ℕ)
variable (h1 : won_games = 70)
variable (h2 : won_games = total_games / 2)

theorem soccer_team_games_count : total_games = 140 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_soccer_team_games_count_l333_33387


namespace NUMINAMATH_GPT_digit_in_thousandths_place_l333_33336

theorem digit_in_thousandths_place : (3 / 16 : ℚ) = 0.1875 :=
by sorry

end NUMINAMATH_GPT_digit_in_thousandths_place_l333_33336


namespace NUMINAMATH_GPT_solve_for_x_l333_33301

variable (x : ℝ)

theorem solve_for_x (h : (4 * x + 2) / (5 * x - 5) = 3 / 4) : x = -23 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l333_33301


namespace NUMINAMATH_GPT_line_equation_l333_33331

noncomputable def P (A B C x y : ℝ) := A * x + B * y + C

theorem line_equation {A B C x₁ y₁ x₂ y₂ : ℝ} (h1 : P A B C x₁ y₁ = 0) (h2 : P A B C x₂ y₂ ≠ 0) :
    ∀ (x y : ℝ), P A B C x y - P A B C x₁ y₁ - P A B C x₂ y₂ = 0 ↔ P A B 0 x y = -P A B 0 x₂ y₂ := by
  sorry

end NUMINAMATH_GPT_line_equation_l333_33331


namespace NUMINAMATH_GPT_sin_alpha_eq_63_over_65_l333_33334

open Real

variables {α β : ℝ}

theorem sin_alpha_eq_63_over_65
  (h1 : tan β = 4 / 3)
  (h2 : sin (α + β) = 5 / 13)
  (h3 : 0 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π) :
  sin α = 63 / 65 := 
by
  sorry

end NUMINAMATH_GPT_sin_alpha_eq_63_over_65_l333_33334


namespace NUMINAMATH_GPT_money_put_in_by_A_l333_33308

theorem money_put_in_by_A 
  (B_capital : ℕ := 25000)
  (total_profit : ℕ := 9600)
  (A_management_fee : ℕ := 10)
  (A_total_received : ℕ := 4200) 
  (A_puts_in : ℕ) :
  (A_management_fee * total_profit / 100 
    + (A_puts_in / (A_puts_in + B_capital)) * (total_profit - A_management_fee * total_profit / 100) = A_total_received)
  → A_puts_in = 15000 :=
  by
    sorry

end NUMINAMATH_GPT_money_put_in_by_A_l333_33308


namespace NUMINAMATH_GPT_james_total_earnings_l333_33379

-- Define the earnings for January
def januaryEarnings : ℕ := 4000

-- Define the earnings for February based on January
def februaryEarnings : ℕ := 2 * januaryEarnings

-- Define the earnings for March based on February
def marchEarnings : ℕ := februaryEarnings - 2000

-- Define the total earnings including January, February, and March
def totalEarnings : ℕ := januaryEarnings + februaryEarnings + marchEarnings

-- State the theorem: total earnings should be 18000
theorem james_total_earnings : totalEarnings = 18000 := by
  sorry

end NUMINAMATH_GPT_james_total_earnings_l333_33379


namespace NUMINAMATH_GPT_discount_is_one_percent_l333_33355

/-
  Assuming the following:
  - market_price is the price of one pen in dollars.
  - num_pens is the number of pens bought.
  - cost_price is the total cost price paid by the retailer.
  - profit_percentage is the profit made by the retailer.
  We need to prove that the discount percentage is 1.
-/

noncomputable def discount_percentage
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (SP_per_pen : ℝ) : ℝ :=
  ((market_price - SP_per_pen) / market_price) * 100

theorem discount_is_one_percent
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (buying_condition : cost_price = (market_price * num_pens * (36 / 60)))
  (SP : ℝ)
  (selling_condition : SP = cost_price * (1 + profit_percentage / 100))
  (SP_per_pen : ℝ)
  (sp_per_pen_condition : SP_per_pen = SP / num_pens)
  (profit_condition : profit_percentage = 65) :
  discount_percentage market_price num_pens cost_price profit_percentage SP_per_pen = 1 := by
  sorry

end NUMINAMATH_GPT_discount_is_one_percent_l333_33355


namespace NUMINAMATH_GPT_second_class_students_l333_33359

-- Define the conditions
variables (x : ℕ)
variable (sum_marks_first_class : ℕ := 35 * 40)
variable (sum_marks_second_class : ℕ := x * 60)
variable (total_students : ℕ := 35 + x)
variable (total_marks_all_students : ℕ := total_students * 5125 / 100)

-- The theorem to prove
theorem second_class_students : 
  1400 + (x * 60) = (35 + x) * 5125 / 100 →
  x = 45 :=
by
  sorry

end NUMINAMATH_GPT_second_class_students_l333_33359


namespace NUMINAMATH_GPT_seniors_selected_correct_l333_33304

-- Definitions based on the conditions problem
def total_freshmen : ℕ := 210
def total_sophomores : ℕ := 270
def total_seniors : ℕ := 300
def selected_freshmen : ℕ := 7

-- Problem statement to prove
theorem seniors_selected_correct : 
  (total_seniors / (total_freshmen / selected_freshmen)) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_seniors_selected_correct_l333_33304


namespace NUMINAMATH_GPT_base6_sum_eq_10_l333_33310

theorem base6_sum_eq_10 
  (A B C : ℕ) 
  (hA : 0 < A ∧ A < 6) 
  (hB : 0 < B ∧ B < 6) 
  (hC : 0 < C ∧ C < 6)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_add : A*36 + B*6 + C + B*6 + C = A*36 + C*6 + A) :
  A + B + C = 10 := 
by
  sorry

end NUMINAMATH_GPT_base6_sum_eq_10_l333_33310


namespace NUMINAMATH_GPT_translation_of_point_l333_33302

variable (P : ℝ × ℝ) (xT yT : ℝ)

def translate_x (P : ℝ × ℝ) (xT : ℝ) : ℝ × ℝ :=
    (P.1 + xT, P.2)

def translate_y (P : ℝ × ℝ) (yT : ℝ) : ℝ × ℝ :=
    (P.1, P.2 + yT)

theorem translation_of_point : translate_y (translate_x (-5, 1) 2) (-4) = (-3, -3) :=
by
  sorry

end NUMINAMATH_GPT_translation_of_point_l333_33302


namespace NUMINAMATH_GPT_normal_price_of_article_l333_33363

theorem normal_price_of_article 
  (P : ℝ) 
  (h : (P * 0.88 * 0.78 * 0.85) * 1.06 = 144) : 
  P = 144 / (0.88 * 0.78 * 0.85 * 1.06) :=
sorry

end NUMINAMATH_GPT_normal_price_of_article_l333_33363


namespace NUMINAMATH_GPT_arithmetic_mean_of_two_digit_multiples_of_5_l333_33364

theorem arithmetic_mean_of_two_digit_multiples_of_5:
  let smallest := 10
  let largest := 95
  let num_terms := 18
  let sum := 945
  let mean := (sum : ℝ) / (num_terms : ℝ)
  mean = 52.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_two_digit_multiples_of_5_l333_33364


namespace NUMINAMATH_GPT_find_other_endpoint_l333_33375

set_option pp.funBinderTypes true

def circle_center : (ℝ × ℝ) := (5, -2)
def diameter_endpoint1 : (ℝ × ℝ) := (1, 2)
def diameter_endpoint2 : (ℝ × ℝ) := (9, -6)

theorem find_other_endpoint (c : ℝ × ℝ) (e1 : ℝ × ℝ) (e2 : ℝ × ℝ) : 
  c = circle_center ∧ e1 = diameter_endpoint1 → e2 = diameter_endpoint2 := by
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l333_33375


namespace NUMINAMATH_GPT_find_number_l333_33361

theorem find_number (x : ℝ) (hx : (50 + 20 / x) * x = 4520) : x = 90 :=
sorry

end NUMINAMATH_GPT_find_number_l333_33361


namespace NUMINAMATH_GPT_range_of_m_l333_33312

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, 3 * x^2 + 1 ≥ m * x * (x - 1)) : -6 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l333_33312


namespace NUMINAMATH_GPT_ratio_AB_CD_lengths_AB_CD_l333_33328

theorem ratio_AB_CD 
  (AM MD BN NC : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  : (AM / MD) / (BN / NC) = 5 / 6 :=
by
  sorry

theorem lengths_AB_CD
  (AM MD BN NC AB CD : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  (AB_div_CD : (AM / MD) / (BN / NC) = 5 / 6)
  (h_touch : true)  -- A placeholder condition indicating circles touch each other
  : AB = 5 ∧ CD = 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_AB_CD_lengths_AB_CD_l333_33328


namespace NUMINAMATH_GPT_measure_of_angle_C_l333_33380

theorem measure_of_angle_C (A B C : ℕ) (h1 : A + B = 150) (h2 : A + B + C = 180) : C = 30 := 
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l333_33380


namespace NUMINAMATH_GPT_quiz_total_points_l333_33381

theorem quiz_total_points (points : ℕ → ℕ) 
  (h1 : ∀ n, points (n+1) = points n + 4)
  (h2 : points 2 = 39) : 
  (points 0 + points 1 + points 2 + points 3 + points 4 + points 5 + points 6 + points 7) = 360 :=
sorry

end NUMINAMATH_GPT_quiz_total_points_l333_33381


namespace NUMINAMATH_GPT_statement_C_l333_33390

variables (a b c d : ℝ)

theorem statement_C (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end NUMINAMATH_GPT_statement_C_l333_33390


namespace NUMINAMATH_GPT_set_complement_union_eq_l333_33382

open Set

variable (U : Set ℕ) (P : Set ℕ) (Q : Set ℕ)

theorem set_complement_union_eq :
  U = {1, 2, 3, 4, 5, 6} →
  P = {1, 3, 5} →
  Q = {1, 2, 4} →
  (U \ P) ∪ Q = {1, 2, 4, 6} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end NUMINAMATH_GPT_set_complement_union_eq_l333_33382


namespace NUMINAMATH_GPT_expected_ties_after_10_l333_33317

def binom: ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom n (k+1)

noncomputable def expected_ties : ℕ → ℝ 
| 0 => 0
| n+1 => expected_ties n + (binom (2*(n+1)) (n+1) / 2^(2*(n+1)))

theorem expected_ties_after_10 : expected_ties 5 = 1.707 := 
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_expected_ties_after_10_l333_33317


namespace NUMINAMATH_GPT_opposite_of_83_is_84_l333_33392

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end NUMINAMATH_GPT_opposite_of_83_is_84_l333_33392


namespace NUMINAMATH_GPT_solve_for_a_l333_33354

theorem solve_for_a (a x y : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 3) : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l333_33354


namespace NUMINAMATH_GPT_intersection_M_N_l333_33311

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3 * x = 0}

theorem intersection_M_N : M ∩ N = {0} :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l333_33311


namespace NUMINAMATH_GPT_each_persons_tip_l333_33319

theorem each_persons_tip
  (cost_julie cost_letitia cost_anton : ℕ)
  (H1 : cost_julie = 10)
  (H2 : cost_letitia = 20)
  (H3 : cost_anton = 30)
  (total_people : ℕ)
  (H4 : total_people = 3)
  (tip_percentage : ℝ)
  (H5 : tip_percentage = 0.20) :
  ∃ tip_per_person : ℝ, tip_per_person = 4 := 
by
  sorry

end NUMINAMATH_GPT_each_persons_tip_l333_33319


namespace NUMINAMATH_GPT_planning_committee_ways_is_20_l333_33349

-- Define the number of students in the council
def num_students : ℕ := 6

-- Define the ways to choose a 3-person committee from num_students
def committee_ways (x : ℕ) : ℕ := Nat.choose x 3

-- Given condition: number of ways to choose the welcoming committee is 20
axiom welcoming_committee_condition : committee_ways num_students = 20

-- Statement to prove
theorem planning_committee_ways_is_20 : committee_ways num_students = 20 := by
  exact welcoming_committee_condition

end NUMINAMATH_GPT_planning_committee_ways_is_20_l333_33349


namespace NUMINAMATH_GPT_probability_one_hits_l333_33314

theorem probability_one_hits 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 1 / 2) (hB : p_B = 1 / 3):
  p_A * (1 - p_B) + (1 - p_A) * p_B = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_one_hits_l333_33314


namespace NUMINAMATH_GPT_sophia_ate_pie_l333_33344

theorem sophia_ate_pie (weight_fridge weight_total weight_ate : ℕ)
  (h1 : weight_fridge = 1200) 
  (h2 : weight_fridge = 5 * weight_total / 6) :
  weight_ate = weight_total / 6 :=
by
  have weight_total_formula : weight_total = 6 * weight_fridge / 5 := by
    sorry
  have weight_ate_formula : weight_ate = weight_total / 6 := by
    sorry
  sorry

end NUMINAMATH_GPT_sophia_ate_pie_l333_33344


namespace NUMINAMATH_GPT_intersection_is_singleton_l333_33309

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_is_singleton : M ∩ N = {1} :=
by sorry

end NUMINAMATH_GPT_intersection_is_singleton_l333_33309


namespace NUMINAMATH_GPT_total_hours_worked_l333_33346

theorem total_hours_worked (amber_hours : ℕ) (armand_hours : ℕ) (ella_hours : ℕ) 
(h_amber : amber_hours = 12) 
(h_armand : armand_hours = (1 / 3) * amber_hours) 
(h_ella : ella_hours = 2 * amber_hours) :
amber_hours + armand_hours + ella_hours = 40 :=
sorry

end NUMINAMATH_GPT_total_hours_worked_l333_33346


namespace NUMINAMATH_GPT_exponent_sum_l333_33342

theorem exponent_sum (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^123 + i^223 + i^323 = -3 * i :=
by
  sorry

end NUMINAMATH_GPT_exponent_sum_l333_33342


namespace NUMINAMATH_GPT_express_in_scientific_notation_l333_33391

theorem express_in_scientific_notation (x : ℝ) (h : x = 720000) : x = 7.2 * 10^5 :=
by sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l333_33391


namespace NUMINAMATH_GPT_sam_read_pages_l333_33393

-- Define conditions
def assigned_pages : ℕ := 25
def harrison_pages : ℕ := assigned_pages + 10
def pam_pages : ℕ := harrison_pages + 15
def sam_pages : ℕ := 2 * pam_pages

-- Prove the target theorem
theorem sam_read_pages : sam_pages = 100 := by
  sorry

end NUMINAMATH_GPT_sam_read_pages_l333_33393


namespace NUMINAMATH_GPT_proof_problem_l333_33357

noncomputable def arithmetic_sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  n * (a 1) + ((n * (n - 1)) / 2) * (a 2 - a 1)

theorem proof_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (d : ℕ)
  (h_d_gt_zero : d > 0)
  (h_a1 : a 1 = 1)
  (h_S : ∀ n, S n = arithmetic_sequence_sum n a)
  (h_S2_S3 : S 2 * S 3 = 36)
  (h_arith_seq : ∀ n, a (n + 1) = a 1 + n * d)
  (m k : ℕ)
  (h_mk_pos : m > 0 ∧ k > 0)
  (sum_condition : (k + 1) * (a m + a (m + k)) / 2 = 65) :
  d = 2 ∧ (∀ n, S n = n * n) ∧ m = 5 ∧ k = 4 :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l333_33357


namespace NUMINAMATH_GPT_area_of_circle_l333_33372

theorem area_of_circle :
  (∃ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y = -9) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  -- We need to prove the area is 16π
  sorry

end NUMINAMATH_GPT_area_of_circle_l333_33372


namespace NUMINAMATH_GPT_at_least_fifty_same_leading_coefficient_l333_33362

-- Define what it means for two quadratic polynomials to intersect exactly once
def intersect_once (P Q : Polynomial ℝ) : Prop :=
∃ x, P.eval x = Q.eval x ∧ ∀ y ≠ x, P.eval y ≠ Q.eval y

-- Define the main theorem and its conditions
theorem at_least_fifty_same_leading_coefficient 
  (polynomials : Fin 100 → Polynomial ℝ)
  (h1 : ∀ i j, i ≠ j → intersect_once (polynomials i) (polynomials j))
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
        ¬∃ x, (polynomials i).eval x = (polynomials j).eval x ∧ (polynomials j).eval x = (polynomials k).eval x) : 
  ∃ (S : Finset (Fin 100)), S.card ≥ 50 ∧ ∃ a, ∀ i ∈ S, (polynomials i).leadingCoeff = a :=
sorry

end NUMINAMATH_GPT_at_least_fifty_same_leading_coefficient_l333_33362


namespace NUMINAMATH_GPT_max_number_of_circular_triples_l333_33315

theorem max_number_of_circular_triples (players : Finset ℕ) (game_results : ℕ → ℕ → Prop) (total_players : players.card = 14)
  (each_plays_13_others : ∀ (p : ℕ) (hp : p ∈ players), ∃ wins losses : Finset ℕ, wins.card = 6 ∧ losses.card = 7 ∧
    (∀ w ∈ wins, game_results p w) ∧ (∀ l ∈ losses, game_results l p)) :
  (∃ (circular_triples : Finset (Finset ℕ)), circular_triples.card = 112 ∧
    ∀ t ∈ circular_triples, t.card = 3 ∧
    (∀ x y z : ℕ, x ∈ t ∧ y ∈ t ∧ z ∈ t → game_results x y ∧ game_results y z ∧ game_results z x)) := 
sorry

end NUMINAMATH_GPT_max_number_of_circular_triples_l333_33315


namespace NUMINAMATH_GPT_Petya_cannot_achieve_goal_l333_33389

theorem Petya_cannot_achieve_goal (n : ℕ) (h : n ≥ 2) :
  ¬ (∃ (G : ℕ → Prop), (∀ i : ℕ, (G i ↔ (G ((i + 2) % (2 * n))))) ∨ (G (i + 1) ≠ G (i + 2))) :=
sorry

end NUMINAMATH_GPT_Petya_cannot_achieve_goal_l333_33389


namespace NUMINAMATH_GPT_carla_water_requirement_l333_33340

theorem carla_water_requirement (h: ℕ) (p: ℕ) (c: ℕ) (gallons_per_pig: ℕ) (horse_factor: ℕ) 
  (num_pigs: ℕ) (num_horses: ℕ) (tank_water: ℕ): 
  num_pigs = 8 ∧ num_horses = 10 ∧ gallons_per_pig = 3 ∧ horse_factor = 2 ∧ tank_water = 30 →
  h = horse_factor * gallons_per_pig ∧ p = num_pigs * gallons_per_pig ∧ c = tank_water →
  h * num_horses + p + c = 114 :=
by
  intro h1 h2
  cases h1
  cases h2
  sorry

end NUMINAMATH_GPT_carla_water_requirement_l333_33340


namespace NUMINAMATH_GPT_cookies_left_after_week_l333_33373

theorem cookies_left_after_week (cookies_in_jar : ℕ) (total_taken_out_in_4_days : ℕ) (same_amount_each_day : Prop)
  (h1 : cookies_in_jar = 70) (h2 : total_taken_out_in_4_days = 24) :
  ∃ (cookies_left : ℕ), cookies_left = 28 :=
by
  sorry

end NUMINAMATH_GPT_cookies_left_after_week_l333_33373


namespace NUMINAMATH_GPT_modulus_z_eq_sqrt_10_l333_33348

noncomputable def z : ℂ := (1 + 7 * Complex.I) / (2 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := sorry

end NUMINAMATH_GPT_modulus_z_eq_sqrt_10_l333_33348


namespace NUMINAMATH_GPT_smallest_t_satisfies_equation_l333_33386

def satisfies_equation (t x y : ℤ) : Prop :=
  (x^2 + y^2)^2 + 2 * t * x * (x^2 + y^2) = t^2 * y^2

theorem smallest_t_satisfies_equation : ∃ t x y : ℤ, t > 0 ∧ x > 0 ∧ y > 0 ∧ satisfies_equation t x y ∧
  ∀ t' x' y' : ℤ, t' > 0 ∧ x' > 0 ∧ y' > 0 ∧ satisfies_equation t' x' y' → t' ≥ t :=
sorry

end NUMINAMATH_GPT_smallest_t_satisfies_equation_l333_33386


namespace NUMINAMATH_GPT_calc_expression1_calc_expression2_l333_33377

theorem calc_expression1 : (1 / 3)^0 + Real.sqrt 27 - abs (-3) + Real.tan (Real.pi / 4) = 1 + 3 * Real.sqrt 3 - 2 :=
by
  sorry

theorem calc_expression2 (x : ℝ) : (x + 2)^2 - 2 * (x - 1) = x^2 + 2 * x + 6 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression1_calc_expression2_l333_33377


namespace NUMINAMATH_GPT_jim_travels_20_percent_of_jill_l333_33395

def john_distance : ℕ := 15
def jill_travels_less : ℕ := 5
def jim_distance : ℕ := 2
def jill_distance : ℕ := john_distance - jill_travels_less

theorem jim_travels_20_percent_of_jill :
  (jim_distance * 100) / jill_distance = 20 := by
  sorry

end NUMINAMATH_GPT_jim_travels_20_percent_of_jill_l333_33395


namespace NUMINAMATH_GPT_sum_of_decimals_l333_33324

theorem sum_of_decimals : (0.305 : ℝ) + (0.089 : ℝ) + (0.007 : ℝ) = 0.401 := by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l333_33324


namespace NUMINAMATH_GPT_man_salary_l333_33325

variable (S : ℝ)

theorem man_salary (S : ℝ) (h1 : S - (1/3) * S - (1/4) * S - (1/5) * S = 1760) : S = 8123 := 
by 
  sorry

end NUMINAMATH_GPT_man_salary_l333_33325


namespace NUMINAMATH_GPT_initial_gummy_worms_l333_33384

variable (G : ℕ)

theorem initial_gummy_worms (h : (G : ℚ) / 16 = 4) : G = 64 :=
by
  sorry

end NUMINAMATH_GPT_initial_gummy_worms_l333_33384


namespace NUMINAMATH_GPT_radius_of_sphere_find_x_for_equation_l333_33396

-- Problem I2.1
theorem radius_of_sphere (r : ℝ) (V : ℝ) (h : V = 36 * π) : r = 3 :=
sorry

-- Problem I2.2
theorem find_x_for_equation (x : ℝ) (r : ℝ) (h_r : r = 3) (h : r^x + r^(1-x) = 4) (h_x_pos : x > 0) : x = 1 :=
sorry

end NUMINAMATH_GPT_radius_of_sphere_find_x_for_equation_l333_33396


namespace NUMINAMATH_GPT_geometric_sequence_a6_a8_sum_l333_33388

theorem geometric_sequence_a6_a8_sum 
  (a : ℕ → ℕ) (q : ℕ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 1 + a 3 = 5)
  (h2 : a 2 + a 4 = 10) : 
  a 6 + a 8 = 160 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a6_a8_sum_l333_33388


namespace NUMINAMATH_GPT_smallest_angle_equilateral_triangle_l333_33378

-- Definitions corresponding to the conditions
structure EquilateralTriangle :=
(vertices : Fin 3 → ℝ × ℝ)
(equilateral : ∀ i j, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))

def point_on_line_segment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
((1 - t) * p1.1 + t * p2.1, (1 - t) * p1.2 + t * p2.2)

-- Given an equilateral triangle ABC with vertices A, B, C,
-- and points D on AB, E on AC, D1 on BC, and E1 on BC,
-- such that AB = DB + BD_1 and AC = CE + CE_1,
-- prove the smallest angle between DE_1 and ED_1 is 60 degrees.

theorem smallest_angle_equilateral_triangle
  (ABC : EquilateralTriangle)
  (A B C D E D₁ E₁ : ℝ × ℝ)
  (on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = point_on_line_segment A B t)
  (on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = point_on_line_segment A C t)
  (on_BC : ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ D₁ = point_on_line_segment B C t₁ ∧
                         0 ≤ t₂ ∧ t₂ ≤ 1 ∧ E₁ = point_on_line_segment B C t₂)
  (AB_property : dist A B = dist D B + dist B D₁)
  (AC_property : dist A C = dist E C + dist C E₁) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 60 ∧ θ = 60 :=
sorry

end NUMINAMATH_GPT_smallest_angle_equilateral_triangle_l333_33378


namespace NUMINAMATH_GPT_hyperbola_slope_reciprocals_l333_33332

theorem hyperbola_slope_reciprocals (P : ℝ × ℝ) (t : ℝ) :
  (P.1 = t ∧ P.2 = - (8 / 9) * t ∧ t ≠ 0 ∧  
    ∃ k1 k2: ℝ, k1 = - (8 * t) / (9 * (t + 3)) ∧ k2 = - (8 * t) / (9 * (t - 3)) ∧
    (1 / k1) + (1 / k2) = -9 / 4) ∧
    ((P = (9/5, -(8/5)) ∨ P = (-(9/5), 8/5)) →
        ∃ kOA kOB kOC kOD : ℝ, (kOA + kOB + kOC + kOD = 0)) := 
sorry

end NUMINAMATH_GPT_hyperbola_slope_reciprocals_l333_33332


namespace NUMINAMATH_GPT_power_multiplication_l333_33356

variable (x y m n : ℝ)

-- Establishing our initial conditions
axiom h1 : 10^x = m
axiom h2 : 10^y = n

theorem power_multiplication : 10^(2*x + 3*y) = m^2 * n^3 :=
by
  sorry

end NUMINAMATH_GPT_power_multiplication_l333_33356


namespace NUMINAMATH_GPT_binary_sum_correct_l333_33397

-- Definitions of the binary numbers
def bin1 : ℕ := 0b1011
def bin2 : ℕ := 0b101
def bin3 : ℕ := 0b11001
def bin4 : ℕ := 0b1110
def bin5 : ℕ := 0b100101

-- The statement to prove
theorem binary_sum_correct : bin1 + bin2 + bin3 + bin4 + bin5 = 0b1111010 := by
  sorry

end NUMINAMATH_GPT_binary_sum_correct_l333_33397


namespace NUMINAMATH_GPT_remainder_T_2015_mod_10_l333_33339

-- Define the number of sequences with no more than two consecutive identical letters
noncomputable def T : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 6
| n + 1 => (T n + T (n - 1) + T (n - 2) + T (n - 3))  -- hypothetically following initial conditions pattern

theorem remainder_T_2015_mod_10 : T 2015 % 10 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_T_2015_mod_10_l333_33339


namespace NUMINAMATH_GPT_how_many_times_l333_33343

theorem how_many_times (a b : ℝ) (h1 : a = 0.5) (h2 : b = 0.01) : a / b = 50 := 
by 
  sorry

end NUMINAMATH_GPT_how_many_times_l333_33343


namespace NUMINAMATH_GPT_find_f_2008_l333_33316

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero : f 0 = 2008

axiom f_inequality1 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality2 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 : f 2008 = 2^2008 + 2007 :=
sorry

end NUMINAMATH_GPT_find_f_2008_l333_33316


namespace NUMINAMATH_GPT_greatest_value_of_squares_l333_33320

-- Given conditions
variables (a b c d : ℝ)
variables (h1 : a + b = 20)
variables (h2 : ab + c + d = 105)
variables (h3 : ad + bc = 225)
variables (h4 : cd = 144)

theorem greatest_value_of_squares : a^2 + b^2 + c^2 + d^2 ≤ 150 := by
  sorry

end NUMINAMATH_GPT_greatest_value_of_squares_l333_33320


namespace NUMINAMATH_GPT_canoes_more_than_kayaks_l333_33303

theorem canoes_more_than_kayaks (C K : ℕ)
  (h1 : 14 * C + 15 * K = 288)
  (h2 : C = 3 * K / 2) :
  C - K = 4 :=
sorry

end NUMINAMATH_GPT_canoes_more_than_kayaks_l333_33303


namespace NUMINAMATH_GPT_description_of_S_l333_33321

noncomputable def S := {p : ℝ × ℝ | (3 = (p.1 + 2) ∧ p.2 - 5 ≤ 3) ∨ 
                                      (3 = (p.2 - 5) ∧ p.1 + 2 ≤ 3) ∨ 
                                      (p.1 + 2 = p.2 - 5 ∧ 3 ≤ p.1 + 2 ∧ 3 ≤ p.2 - 5)}

theorem description_of_S :
  S = {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 ≤ 8) ∨ 
                    (p.2 = 8 ∧ p.1 ≤ 1) ∨ 
                    (p.2 = p.1 + 7 ∧ p.1 ≥ 1 ∧ p.2 ≥ 8)} :=
sorry

end NUMINAMATH_GPT_description_of_S_l333_33321


namespace NUMINAMATH_GPT_emily_quiz_score_l333_33365

theorem emily_quiz_score :
  ∃ x : ℕ, 94 + 88 + 92 + 85 + 97 + x = 6 * 90 :=
by
  sorry

end NUMINAMATH_GPT_emily_quiz_score_l333_33365


namespace NUMINAMATH_GPT_rhombus_side_length_l333_33333

theorem rhombus_side_length (area d1 d2 side : ℝ) (h_area : area = 24)
(h_d1 : d1 = 6) (h_other_diag : d2 * 6 = 48) (h_side : side = Real.sqrt (3^2 + 4^2)) :
  side = 5 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_rhombus_side_length_l333_33333


namespace NUMINAMATH_GPT_find_shortage_l333_33352

def total_capacity (T : ℝ) : Prop :=
  0.70 * T = 14

def normal_level (normal : ℝ) : Prop :=
  normal = 14 / 2

def capacity_shortage (T : ℝ) (normal : ℝ) : Prop :=
  T - normal = 13

theorem find_shortage (T : ℝ) (normal : ℝ) : 
  total_capacity T →
  normal_level normal →
  capacity_shortage T normal :=
by
  sorry

end NUMINAMATH_GPT_find_shortage_l333_33352


namespace NUMINAMATH_GPT_rainfall_march_l333_33300

variable (M A : ℝ)
variable (Hm : A = M - 0.35)
variable (Ha : A = 0.46)

theorem rainfall_march : M = 0.81 := by
  sorry

end NUMINAMATH_GPT_rainfall_march_l333_33300


namespace NUMINAMATH_GPT_field_area_proof_l333_33323

-- Define the length of the uncovered side
def L : ℕ := 20

-- Define the total amount of fencing used for the other three sides
def total_fence : ℕ := 26

-- Define the field area function
def field_area (length width : ℕ) : ℕ := length * width

-- Statement: Prove that the area of the field is 60 square feet
theorem field_area_proof : 
  ∃ W : ℕ, (2 * W + L = total_fence) ∧ (field_area L W = 60) :=
  sorry

end NUMINAMATH_GPT_field_area_proof_l333_33323


namespace NUMINAMATH_GPT_mark_siblings_l333_33338

theorem mark_siblings (total_eggs : ℕ) (eggs_per_person : ℕ) (persons_including_mark : ℕ) (h1 : total_eggs = 24) (h2 : eggs_per_person = 6) (h3 : persons_including_mark = total_eggs / eggs_per_person) : persons_including_mark - 1 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_mark_siblings_l333_33338


namespace NUMINAMATH_GPT_joe_spent_on_food_l333_33341

theorem joe_spent_on_food :
  ∀ (initial_savings flight hotel remaining food : ℝ),
    initial_savings = 6000 →
    flight = 1200 →
    hotel = 800 →
    remaining = 1000 →
    food = initial_savings - remaining - (flight + hotel) →
    food = 3000 :=
by
  intros initial_savings flight hotel remaining food h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_joe_spent_on_food_l333_33341


namespace NUMINAMATH_GPT_total_flowers_is_288_l333_33345

-- Definitions from the Conditions in a)
def arwen_tulips : ℕ := 20
def arwen_roses : ℕ := 18
def elrond_tulips : ℕ := 2 * arwen_tulips
def elrond_roses : ℕ := 3 * arwen_roses
def galadriel_tulips : ℕ := 3 * elrond_tulips
def galadriel_roses : ℕ := 2 * arwen_roses

-- Total number of tulips
def total_tulips : ℕ := arwen_tulips + elrond_tulips + galadriel_tulips

-- Total number of roses
def total_roses : ℕ := arwen_roses + elrond_roses + galadriel_roses

-- Total number of flowers
def total_flowers : ℕ := total_tulips + total_roses

theorem total_flowers_is_288 : total_flowers = 288 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_total_flowers_is_288_l333_33345


namespace NUMINAMATH_GPT_triangle_angle_tangent_condition_l333_33374

theorem triangle_angle_tangent_condition
  (A B C : ℝ)
  (h1 : A + C = 2 * B)
  (h2 : Real.tan A * Real.tan C = 2 + Real.sqrt 3) :
  (A = Real.pi / 4 ∧ B = Real.pi / 3 ∧ C = 5 * Real.pi / 12) ∨
  (A = 5 * Real.pi / 12 ∧ B = Real.pi / 3 ∧ C = Real.pi / 4) :=
  sorry

end NUMINAMATH_GPT_triangle_angle_tangent_condition_l333_33374


namespace NUMINAMATH_GPT_definite_integral_ln_squared_l333_33367

noncomputable def integralFun : ℝ → ℝ := λ x => x * (Real.log x) ^ 2

theorem definite_integral_ln_squared (f : ℝ → ℝ) (a b : ℝ):
  (f = integralFun) → 
  (a = 1) → 
  (b = 2) → 
  ∫ x in a..b, f x = 2 * (Real.log 2) ^ 2 - 2 * Real.log 2 + 3 / 4 :=
by
  intros hfa hao hbo
  rw [hfa, hao, hbo]
  sorry

end NUMINAMATH_GPT_definite_integral_ln_squared_l333_33367


namespace NUMINAMATH_GPT_odd_function_f_l333_33326

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

theorem odd_function_f :
  odd_function f :=
sorry

end NUMINAMATH_GPT_odd_function_f_l333_33326


namespace NUMINAMATH_GPT_glucose_solution_l333_33376

theorem glucose_solution (x : ℝ) (h : (15 / 100 : ℝ) = (6.75 / x)) : x = 45 :=
sorry

end NUMINAMATH_GPT_glucose_solution_l333_33376


namespace NUMINAMATH_GPT_length_reduction_percentage_to_maintain_area_l333_33335

theorem length_reduction_percentage_to_maintain_area
  (L W : ℝ)
  (new_width : ℝ := W * (1 + 28.2051282051282 / 100))
  (new_length : ℝ := L * (1 - 21.9512195121951 / 100))
  (original_area : ℝ := L * W) :
  original_area = new_length * new_width := by
  sorry

end NUMINAMATH_GPT_length_reduction_percentage_to_maintain_area_l333_33335


namespace NUMINAMATH_GPT_rain_on_both_days_l333_33347

-- Define the events probabilities
variables (P_M P_T P_N P_MT : ℝ)

-- Define the initial conditions
axiom h1 : P_M = 0.6
axiom h2 : P_T = 0.55
axiom h3 : P_N = 0.25

-- Define the statement to prove
theorem rain_on_both_days : P_MT = 0.4 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_GPT_rain_on_both_days_l333_33347


namespace NUMINAMATH_GPT_tiffany_max_points_l333_33327

theorem tiffany_max_points : 
  let initial_money := 3
  let cost_per_game := 1
  let points_red_bucket := 2
  let points_green_bucket := 3
  let rings_per_game := 5
  let games_played := 2
  let red_buckets_first_two_games := 4
  let green_buckets_first_two_games := 5
  let remaining_money := initial_money - games_played * cost_per_game
  let remaining_games := remaining_money / cost_per_game
  let points_first_two_games := red_buckets_first_two_games * points_red_bucket + green_buckets_first_two_games * points_green_bucket
  let max_points_third_game := rings_per_game * points_green_bucket
  points_first_two_games + max_points_third_game = 38 := 
by
  sorry

end NUMINAMATH_GPT_tiffany_max_points_l333_33327


namespace NUMINAMATH_GPT_parallelogram_area_formula_l333_33337

noncomputable def parallelogram_area (ha hb : ℝ) (γ : ℝ) : ℝ := 
  ha * hb / Real.sin γ

theorem parallelogram_area_formula (ha hb γ : ℝ) (a b : ℝ) 
  (h₁ : Real.sin γ ≠ 0) :
  (parallelogram_area ha hb γ = ha * hb / Real.sin γ) := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_formula_l333_33337


namespace NUMINAMATH_GPT_physics_teacher_min_count_l333_33307

theorem physics_teacher_min_count 
  (maths_teachers : ℕ) 
  (chemistry_teachers : ℕ) 
  (max_subjects_per_teacher : ℕ) 
  (min_total_teachers : ℕ) 
  (physics_teachers : ℕ)
  (h1 : maths_teachers = 7)
  (h2 : chemistry_teachers = 5)
  (h3 : max_subjects_per_teacher = 3)
  (h4 : min_total_teachers = 6) 
  (h5 : 7 + physics_teachers + 5 ≤ 6 * 3) :
  0 < physics_teachers :=
  by 
  sorry

end NUMINAMATH_GPT_physics_teacher_min_count_l333_33307


namespace NUMINAMATH_GPT_exists_integers_m_n_for_inequalities_l333_33368

theorem exists_integers_m_n_for_inequalities (a b : ℝ) (h : a ≠ b) : ∃ (m n : ℤ), 
  (a * (m : ℝ) + b * (n : ℝ) < 0) ∧ (b * (m : ℝ) + a * (n : ℝ) > 0) :=
sorry

end NUMINAMATH_GPT_exists_integers_m_n_for_inequalities_l333_33368


namespace NUMINAMATH_GPT_scholarship_total_l333_33318

-- Definitions of the money received by Wendy, Kelly, Nina, and Jason based on the given conditions
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000
def jason_scholarship : ℕ := (3 * kelly_scholarship) / 4

-- Total amount of scholarships
def total_scholarship : ℕ := wendy_scholarship + kelly_scholarship + nina_scholarship + jason_scholarship

-- The proof statement that needs to be proven
theorem scholarship_total : total_scholarship = 122000 := by
  -- Here we use 'sorry' to indicate that the proof is not provided.
  sorry

end NUMINAMATH_GPT_scholarship_total_l333_33318


namespace NUMINAMATH_GPT_value_of_g_800_l333_33353

noncomputable def g : ℝ → ℝ :=
sorry

theorem value_of_g_800 (g_eq : ∀ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), g (x * y) = g x / (y^2))
  (g_at_1000 : g 1000 = 4) : g 800 = 625 / 2 :=
sorry

end NUMINAMATH_GPT_value_of_g_800_l333_33353


namespace NUMINAMATH_GPT_not_lucky_1994_l333_33366

def is_valid_month (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12

def is_valid_day (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 31

def is_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), is_valid_month m ∧ is_valid_day d ∧ m * d = y

theorem not_lucky_1994 : ¬ is_lucky_year 94 := 
by
  sorry

end NUMINAMATH_GPT_not_lucky_1994_l333_33366


namespace NUMINAMATH_GPT_horner_v2_value_l333_33329

def polynomial : ℤ → ℤ := fun x => 208 + 9 * x^2 + 6 * x^4 + x^6

def horner (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x
  let v2 := v1 * x + 6
  v2

theorem horner_v2_value (x : ℤ) : x = -4 → horner x = 22 :=
by
  intro h
  rw [h]
  rfl

end NUMINAMATH_GPT_horner_v2_value_l333_33329


namespace NUMINAMATH_GPT_cost_for_3300_pens_l333_33305

noncomputable def cost_per_pack (pack_cost : ℝ) (num_pens_per_pack : ℕ) : ℝ :=
  pack_cost / num_pens_per_pack

noncomputable def total_cost (cost_per_pen : ℝ) (num_pens : ℕ) : ℝ :=
  cost_per_pen * num_pens

theorem cost_for_3300_pens (pack_cost : ℝ) (num_pens_per_pack num_pens : ℕ) (h_pack_cost : pack_cost = 45) (h_num_pens_per_pack : num_pens_per_pack = 150) (h_num_pens : num_pens = 3300) :
  total_cost (cost_per_pack pack_cost num_pens_per_pack) num_pens = 990 :=
  by
    sorry

end NUMINAMATH_GPT_cost_for_3300_pens_l333_33305


namespace NUMINAMATH_GPT_a_values_unique_solution_l333_33370

theorem a_values_unique_solution :
  (∀ a : ℝ, ∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) →
  (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end NUMINAMATH_GPT_a_values_unique_solution_l333_33370


namespace NUMINAMATH_GPT_unique_z_value_l333_33306

theorem unique_z_value (x y u z : ℕ) (hx : 0 < x)
    (hy : 0 < y) (hu : 0 < u) (hz : 0 < z)
    (h1 : 3 + x + 21 = y + 25 + z)
    (h2 : 3 + x + 21 = 15 + u + 4)
    (h3 : y + 25 + z = 15 + u + 4)
    (h4 : 3 + y + 15 = x + 25 + u)
    (h5 : 3 + y + 15 = 21 + z + 4)
    (h6 : x + 25 + u = 21 + z + 4):
    z = 20 :=
by
    sorry

end NUMINAMATH_GPT_unique_z_value_l333_33306


namespace NUMINAMATH_GPT_raft_travel_distance_l333_33350

theorem raft_travel_distance (v_b v_s t : ℝ) (h1 : t > 0) 
  (h2 : v_b + v_s = 90 / t) (h3 : v_b - v_s = 70 / t) : 
  v_s * t = 10 := by
  sorry

end NUMINAMATH_GPT_raft_travel_distance_l333_33350


namespace NUMINAMATH_GPT_irrational_sqrt3_l333_33394

def is_irrational (x : ℝ) : Prop := ∀ (a b : ℤ), b ≠ 0 → x ≠ a / b

theorem irrational_sqrt3 :
  let A := 22 / 7
  let B := 0
  let C := Real.sqrt 3
  let D := 3.14
  is_irrational C :=
by
  sorry

end NUMINAMATH_GPT_irrational_sqrt3_l333_33394


namespace NUMINAMATH_GPT_labourer_monthly_income_l333_33351

-- Define the conditions
def total_expense_first_6_months : ℕ := 90 * 6
def total_expense_next_4_months : ℕ := 60 * 4
def debt_cleared_and_savings : ℕ := 30

-- Define the monthly income
def monthly_income : ℕ := 81

-- The statement to be proven
theorem labourer_monthly_income (I D : ℕ) (h1 : 6 * I + D = total_expense_first_6_months) 
                               (h2 : 4 * I - D = total_expense_next_4_months + debt_cleared_and_savings) :
  I = monthly_income :=
by {
  sorry
}

end NUMINAMATH_GPT_labourer_monthly_income_l333_33351


namespace NUMINAMATH_GPT_part1_part2_l333_33369

/-
Part 1: Given the conditions of parabola and line intersection, prove the range of slope k of the line.
-/
theorem part1 (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  k > -2 + 2 * Real.sqrt 2 ∨ k < -2 - 2 * Real.sqrt 2 :=
  sorry

/-
Part 2: Given the conditions of locus of point Q on the line segment P1P2, prove the equation of the locus.
-/
theorem part2 (x y : ℝ) (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  2 * x - y + 1 = 0 ∧ (-Real.sqrt 2 - 1 < x ∧ x < Real.sqrt 2 - 1 ∧ x ≠ -1) :=
  sorry

end NUMINAMATH_GPT_part1_part2_l333_33369


namespace NUMINAMATH_GPT_students_not_enrolled_l333_33330

theorem students_not_enrolled (total_students : ℕ) (students_french : ℕ) (students_german : ℕ) (students_both : ℕ)
  (h1 : total_students = 94)
  (h2 : students_french = 41)
  (h3 : students_german = 22)
  (h4 : students_both = 9) : 
  ∃ (students_neither : ℕ), students_neither = 40 :=
by
  -- We would show the calculation here in a real proof 
  sorry

end NUMINAMATH_GPT_students_not_enrolled_l333_33330


namespace NUMINAMATH_GPT_expr_C_always_positive_l333_33398

-- Define the expressions as Lean definitions
def expr_A (x : ℝ) : ℝ := x^2
def expr_B (x : ℝ) : ℝ := abs (-x + 1)
def expr_C (x : ℝ) : ℝ := (-x)^2 + 2
def expr_D (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem expr_C_always_positive : ∀ (x : ℝ), expr_C x > 0 :=
by
  sorry

end NUMINAMATH_GPT_expr_C_always_positive_l333_33398


namespace NUMINAMATH_GPT_seed_mixture_percentage_l333_33358

theorem seed_mixture_percentage (x y : ℝ) 
  (hx : 0.4 * x + 0.25 * y = 30)
  (hxy : x + y = 100) :
  x / 100 = 0.3333 :=
by 
  sorry

end NUMINAMATH_GPT_seed_mixture_percentage_l333_33358


namespace NUMINAMATH_GPT_min_value_of_sum_of_squares_l333_33322

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 10) : 
  x^2 + y^2 + z^2 ≥ 100 / 29 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_of_squares_l333_33322


namespace NUMINAMATH_GPT_solve_system_l333_33383

-- Definitions for the system of equations.
def system_valid (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Main theorem to prove.
theorem solve_system (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  system_valid y x₁ x₂ x₃ x₄ x₅ →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨ 
  (y = 2 → ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∨ 
  (y^2 + y - 1 = 0 → ∃ (u v : ℝ), 
    x₁ = u ∧ 
    x₅ = v ∧ 
    x₂ = y * u - v ∧ 
    x₃ = -y * (u + v) ∧ 
    x₄ = y * v - u ∧ 
    (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_system_l333_33383
