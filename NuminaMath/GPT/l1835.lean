import Mathlib

namespace proportion_decrease_l1835_183533

open Real

/-- 
Given \(x\) and \(y\) are directly proportional and positive,
if \(x\) decreases by \(q\%\), then \(y\) decreases by \(q\%\).
-/
theorem proportion_decrease (c x q : ℝ) (h_pos : x > 0) (h_q_pos : q > 0)
    (h_direct : ∀ x y, y = c * x) :
    ((x * (1 - q / 100)) = y) → ((y * (1 - q / 100)) = (c * x * (1 - q / 100))) := by
  sorry

end proportion_decrease_l1835_183533


namespace sin_alpha_through_point_l1835_183595

theorem sin_alpha_through_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (-3, -Real.sqrt 3)) :
    Real.sin α = -1 / 2 :=
by
  sorry

end sin_alpha_through_point_l1835_183595


namespace larger_tent_fabric_amount_l1835_183581

-- Define the fabric used for the small tent
def small_tent_fabric : ℝ := 4

-- Define the fabric computation for the larger tent
def larger_tent_fabric (small_tent_fabric : ℝ) : ℝ :=
  2 * small_tent_fabric

-- Theorem stating the amount of fabric needed for the larger tent
theorem larger_tent_fabric_amount : larger_tent_fabric small_tent_fabric = 8 :=
by
  -- Skip the actual proof
  sorry

end larger_tent_fabric_amount_l1835_183581


namespace correct_conclusions_count_l1835_183598

theorem correct_conclusions_count :
  (¬ (¬ p → (q ∨ r)) ↔ (¬ p → ¬ q ∧ ¬ r)) = false ∧
  ((¬ p → q) ↔ (p → ¬ q)) = false ∧
  (¬ ∃ n : ℕ, n > 0 ∧ (n ^ 2 + 3 * n) % 10 = 0 ∧ (∀ n : ℕ, n > 0 → (n ^ 2 + 3 * n) % 10 ≠ 0)) = true ∧
  (¬ ∀ x, x ^ 2 - 2 * x + 3 > 0 ∧ (∃ x, x ^ 2 - 2 * x + 3 < 0)) = false :=
by
  sorry

end correct_conclusions_count_l1835_183598


namespace reflected_coordinates_l1835_183579

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, -3)

-- Define the function for reflection across the origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- State the theorem to prove
theorem reflected_coordinates :
  reflect_origin point_P = (2, 3) := by
  sorry

end reflected_coordinates_l1835_183579


namespace isosceles_triangle_l1835_183580

variable (a b c : ℝ)
variable (α β γ : ℝ)
variable (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β))
variable (triangle_angles : γ = π - (α + β))

theorem isosceles_triangle : α = β :=
by
  sorry

end isosceles_triangle_l1835_183580


namespace max_area_225_l1835_183534

noncomputable def max_area_rect_perim60 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) : ℝ :=
max (x * y) (30 - x)

theorem max_area_225 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) :
  max_area_rect_perim60 x y h1 h2 = 225 :=
sorry

end max_area_225_l1835_183534


namespace simplify_expr_C_l1835_183592

theorem simplify_expr_C (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y :=
by
  sorry

end simplify_expr_C_l1835_183592


namespace simplify_expression_l1835_183571

theorem simplify_expression (y : ℝ) : (5 * y) ^ 3 + (4 * y) * (y ^ 2) = 129 * (y ^ 3) := by
  sorry

end simplify_expression_l1835_183571


namespace average_age_combined_l1835_183536

theorem average_age_combined (n1 n2 : ℕ) (avg1 avg2 : ℕ) 
  (h1 : n1 = 45) (h2 : n2 = 60) (h3 : avg1 = 12) (h4 : avg2 = 40) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 28 :=
by
  sorry

end average_age_combined_l1835_183536


namespace roots_sum_of_squares_l1835_183522

theorem roots_sum_of_squares {p q r : ℝ} 
  (h₁ : ∀ x : ℝ, (x - p) * (x - q) * (x - r) = x^3 - 24 * x^2 + 50 * x - 35) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 1052 :=
by
  have h_sum : p + q + r = 24 := by sorry
  have h_product : p * q + q * r + r * p = 50 := by sorry
  sorry

end roots_sum_of_squares_l1835_183522


namespace even_poly_iff_a_zero_l1835_183518

theorem even_poly_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 3) = (x^2 - a*x + 3)) → a = 0 :=
by
  sorry

end even_poly_iff_a_zero_l1835_183518


namespace part1_part2_l1835_183512

-- Let's define the arithmetic sequence and conditions
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + d * (n - 1)
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables (a1 a4 a3 a5 : ℕ)
variable (d : ℕ)

-- Additional conditions for the problem  
axiom h1 : a1 = 2
axiom h2 : a4 = 8
axiom h3 : arithmetic_seq a1 d 3 + arithmetic_seq a1 d 5 = a4 + 8

-- Define S7
def S7 : ℕ := sum_arithmetic_seq a1 d 7

-- Part I: Prove S7 = 56
theorem part1 : S7 = 56 := 
by
  sorry

-- Part II: Prove k = 2 given additional conditions
variable (k : ℕ)

-- Given that a_3, a_{k+1}, S_k are a geometric sequence
def is_geom_seq (a b s : ℕ) : Prop := b*b = a * s

axiom h4 : a3 = arithmetic_seq a1 d 3
axiom h5 : ∃ k, 0 < k ∧ is_geom_seq a3 (arithmetic_seq a1 d (k + 1)) (sum_arithmetic_seq a1 d k)

theorem part2 : ∃ k, 0 < k ∧ k = 2 := 
by
  sorry

end part1_part2_l1835_183512


namespace ben_final_amount_l1835_183525

-- Definition of the conditions
def daily_start := 50
def daily_spent := 15
def daily_saving := daily_start - daily_spent
def days := 7
def mom_double (s : ℕ) := 2 * s
def dad_addition := 10

-- Total amount calculation based on the conditions
noncomputable def total_savings := daily_saving * days
noncomputable def after_mom := mom_double total_savings
noncomputable def total_amount := after_mom + dad_addition

-- The final theorem to prove Ben's final amount is $500 after the given conditions
theorem ben_final_amount : total_amount = 500 :=
by sorry

end ben_final_amount_l1835_183525


namespace radius_of_circle_with_chords_l1835_183561

theorem radius_of_circle_with_chords 
  (chord1_length : ℝ) (chord2_length : ℝ) (distance_between_midpoints : ℝ) 
  (h1 : chord1_length = 9) (h2 : chord2_length = 17) (h3 : distance_between_midpoints = 5) : 
  ∃ r : ℝ, r = 85 / 8 :=
by
  sorry

end radius_of_circle_with_chords_l1835_183561


namespace simplify_fraction_l1835_183596

theorem simplify_fraction (a b m n : ℕ) (h : a ≠ 0 ∧ b ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0) : 
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by
  sorry

end simplify_fraction_l1835_183596


namespace speed_of_policeman_l1835_183558

theorem speed_of_policeman 
  (d_initial : ℝ) 
  (v_thief : ℝ) 
  (d_thief : ℝ)
  (d_policeman : ℝ)
  (h_initial : d_initial = 100) 
  (h_v_thief : v_thief = 8) 
  (h_d_thief : d_thief = 400) 
  (h_d_policeman : d_policeman = 500) 
  : ∃ (v_p : ℝ), v_p = 10 :=
by
  -- Use the provided conditions
  sorry

end speed_of_policeman_l1835_183558


namespace rabbits_to_hamsters_l1835_183578

theorem rabbits_to_hamsters (rabbits hamsters : ℕ) (h_ratio : 3 * hamsters = 4 * rabbits) (h_rabbits : rabbits = 18) : hamsters = 24 :=
by
  sorry

end rabbits_to_hamsters_l1835_183578


namespace marks_change_factor_l1835_183545

def total_marks (n : ℕ) (avg : ℝ) : ℝ := n * avg

theorem marks_change_factor 
  (n : ℕ) (initial_avg new_avg : ℝ) 
  (initial_total := total_marks n initial_avg) 
  (new_total := total_marks n new_avg)
  (h1 : initial_avg = 36)
  (h2 : new_avg = 72)
  (h3 : n = 12):
  (new_total / initial_total) = 2 :=
by
  sorry

end marks_change_factor_l1835_183545


namespace correct_answers_unanswered_minimum_correct_answers_l1835_183555

-- Definition of the conditions in the problem
def total_questions := 25
def unanswered_questions := 1
def correct_points := 4
def wrong_points := -1
def total_score_1 := 86
def total_score_2 := 90

-- Part 1: Define the conditions and prove that x = 22
theorem correct_answers_unanswered (x : ℕ) (h1 : total_questions - unanswered_questions = 24)
  (h2 : 4 * x + wrong_points * (total_questions - unanswered_questions - x) = total_score_1) : x = 22 :=
sorry

-- Part 2: Define the conditions and prove that at least 23 correct answers are needed
theorem minimum_correct_answers (a : ℕ)
  (h3 : correct_points * a + wrong_points * (total_questions - a) ≥ total_score_2) : a ≥ 23 :=
sorry

end correct_answers_unanswered_minimum_correct_answers_l1835_183555


namespace Benny_and_Tim_have_47_books_together_l1835_183504

/-
  Definitions and conditions:
  1. Benny_has_24_books : Benny has 24 books.
  2. Benny_gave_10_books_to_Sandy : Benny gave Sandy 10 books.
  3. Tim_has_33_books : Tim has 33 books.
  
  Goal:
  Prove that together Benny and Tim have 47 books.
-/

def Benny_has_24_books : ℕ := 24
def Benny_gave_10_books_to_Sandy : ℕ := 10
def Tim_has_33_books : ℕ := 33

def Benny_remaining_books : ℕ := Benny_has_24_books - Benny_gave_10_books_to_Sandy

def Benny_and_Tim_together : ℕ := Benny_remaining_books + Tim_has_33_books

theorem Benny_and_Tim_have_47_books_together :
  Benny_and_Tim_together = 47 := by
  sorry

end Benny_and_Tim_have_47_books_together_l1835_183504


namespace equilateral_division_l1835_183557

theorem equilateral_division (k : ℕ) :
  (k = 1 ∨ k = 3 ∨ k = 4 ∨ k = 9 ∨ k = 12 ∨ k = 36) ↔
  (k ∣ 36 ∧ ¬ (k = 2 ∨ k = 6 ∨ k = 18)) := by
  sorry

end equilateral_division_l1835_183557


namespace solution_set_inequality_l1835_183582

def custom_op (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem solution_set_inequality : {x : ℝ | custom_op x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_inequality_l1835_183582


namespace piggy_bank_dimes_diff_l1835_183515

theorem piggy_bank_dimes_diff :
  ∃ (a b c : ℕ), a + b + c = 100 ∧ 5 * a + 10 * b + 25 * c = 1005 ∧ (∀ lo hi, 
  (lo = 1 ∧ hi = 101) → (hi - lo = 100)) :=
by
  sorry

end piggy_bank_dimes_diff_l1835_183515


namespace points_on_same_circle_l1835_183516
open Real

theorem points_on_same_circle (m : ℝ) :
  ∃ D E F, 
  (2^2 + 1^2 + 2 * D + 1 * E + F = 0) ∧
  (4^2 + 2^2 + 4 * D + 2 * E + F = 0) ∧
  (3^2 + 4^2 + 3 * D + 4 * E + F = 0) ∧
  (1^2 + m^2 + 1 * D + m * E + F = 0) →
  (m = 2 ∨ m = 3) := 
sorry

end points_on_same_circle_l1835_183516


namespace oranges_in_bowl_l1835_183546

theorem oranges_in_bowl (bananas : Nat) (apples : Nat) (pears : Nat) (total_fruits : Nat) (h_bananas : bananas = 4) (h_apples : apples = 3 * bananas) (h_pears : pears = 5) (h_total_fruits : total_fruits = 30) :
  total_fruits - (bananas + apples + pears) = 9 :=
by
  subst h_bananas
  subst h_apples
  subst h_pears
  subst h_total_fruits
  sorry

end oranges_in_bowl_l1835_183546


namespace wrong_conclusion_l1835_183575

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem wrong_conclusion {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : 2 * a + b = 0) (h₂ : a + b + c = 3) (h₃ : 4 * a + 2 * b + c = 8) :
  quadratic a b c (-1) ≠ 0 :=
sorry

end wrong_conclusion_l1835_183575


namespace workers_contribution_l1835_183566

theorem workers_contribution (W C : ℕ) 
    (h1 : W * C = 300000) 
    (h2 : W * (C + 50) = 325000) : 
    W = 500 :=
by
    sorry

end workers_contribution_l1835_183566


namespace AllieMoreGrapes_l1835_183526

-- Definitions based on conditions
def RobBowl : ℕ := 25
def TotalGrapes : ℕ := 83
def AllynBowl (A : ℕ) : ℕ := A + 4

-- The proof statement that must be shown.
theorem AllieMoreGrapes (A : ℕ) (h1 : A + (AllynBowl A) + RobBowl = TotalGrapes) : A - RobBowl = 2 :=
by {
  sorry
}

end AllieMoreGrapes_l1835_183526


namespace cubic_roots_quadratic_l1835_183586

theorem cubic_roots_quadratic (A B C p : ℚ)
  (hA : A ≠ 0)
  (h1 : (∀ x : ℚ, A * x^2 + B * x + C = 0 ↔ x = (root1) ∨ x = (root2)))
  (h2 : root1 + root2 = - B / A)
  (h3 : root1 * root2 = C / A)
  (new_eq : ∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = root1^3 ∨ x = root2^3) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by
  sorry

end cubic_roots_quadratic_l1835_183586


namespace anne_clean_house_in_12_hours_l1835_183563

theorem anne_clean_house_in_12_hours (B A : ℝ) (h1 : 4 * (B + A) = 1) (h2 : 3 * (B + 2 * A) = 1) : A = 1 / 12 ∧ (1 / A) = 12 :=
by
  -- We will leave the proof as a placeholder
  sorry

end anne_clean_house_in_12_hours_l1835_183563


namespace calculate_expression_l1835_183548

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end calculate_expression_l1835_183548


namespace Natasha_avg_speed_climb_l1835_183532

-- Definitions for conditions
def distance_to_top : ℝ := sorry -- We need to find this
def time_up := 3 -- time in hours to climb up
def time_down := 2 -- time in hours to climb down
def avg_speed_journey := 3 -- avg speed in km/hr for the whole journey

-- Equivalent math proof problem statement
theorem Natasha_avg_speed_climb (distance_to_top : ℝ) 
  (h1 : time_up = 3)
  (h2 : time_down = 2)
  (h3 : avg_speed_journey = 3)
  (h4 : (2 * distance_to_top) / (time_up + time_down) = avg_speed_journey) : 
  (distance_to_top / time_up) = 2.5 :=
sorry -- Proof not required

end Natasha_avg_speed_climb_l1835_183532


namespace mrs_hilt_water_fountain_trips_l1835_183572

theorem mrs_hilt_water_fountain_trips (d : ℕ) (t : ℕ) (n : ℕ) 
  (h1 : d = 30) 
  (h2 : t = 120) 
  (h3 : 2 * d * n = t) : 
  n = 2 :=
by
  -- Proof omitted
  sorry

end mrs_hilt_water_fountain_trips_l1835_183572


namespace symmetry_x_axis_l1835_183588

theorem symmetry_x_axis (a b : ℝ) (h1 : a - 3 = 2) (h2 : 1 = -(b + 1)) : a + b = 3 :=
by
  sorry

end symmetry_x_axis_l1835_183588


namespace cos_diff_proof_l1835_183510

theorem cos_diff_proof (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5 / 8 := 
by
  sorry

end cos_diff_proof_l1835_183510


namespace take_home_pay_is_correct_l1835_183599

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l1835_183599


namespace sqrt_of_4_eq_2_l1835_183554

theorem sqrt_of_4_eq_2 : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_4_eq_2_l1835_183554


namespace problem_solution_set_l1835_183587

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^3 - 8 else (-x)^3 - 8

theorem problem_solution_set : 
  { x : ℝ | f (x-2) > 0 } = {x : ℝ | x < 0} ∪ {x : ℝ | x > 4} :=
by sorry

end problem_solution_set_l1835_183587


namespace simplify_expression_l1835_183502

theorem simplify_expression (m : ℝ) (h1 : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6 * m + 9)) = m - 3 := 
by
  sorry

end simplify_expression_l1835_183502


namespace down_payment_calculation_l1835_183553

theorem down_payment_calculation 
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (n : ℕ)
  (interest_rate : ℝ)
  (down_payment : ℝ) :
  purchase_price = 127 ∧ 
  monthly_payment = 10 ∧ 
  n = 12 ∧ 
  interest_rate = 0.2126 ∧
  down_payment + (n * monthly_payment) = purchase_price * (1 + interest_rate) 
  → down_payment = 34 := 
sorry

end down_payment_calculation_l1835_183553


namespace joined_toucans_is_1_l1835_183531

-- Define the number of toucans initially
def initial_toucans : ℕ := 2

-- Define the total number of toucans after some join
def total_toucans : ℕ := 3

-- Define the number of toucans that joined
def toucans_joined : ℕ := total_toucans - initial_toucans

-- State the theorem to prove that 1 toucan joined
theorem joined_toucans_is_1 : toucans_joined = 1 :=
by
  sorry

end joined_toucans_is_1_l1835_183531


namespace ratio_of_sam_to_sue_l1835_183524

-- Definitions
def Sam_age (S : ℕ) : Prop := 3 * S = 18
def Kendra_age (K : ℕ) : Prop := K = 18
def total_age_in_3_years (S U K : ℕ) : Prop := (S + 3) + (U + 3) + (K + 3) = 36

-- Theorem statement
theorem ratio_of_sam_to_sue (S U K : ℕ) (h1 : Sam_age S) (h2 : Kendra_age K) (h3 : total_age_in_3_years S U K) :
  S / U = 2 :=
sorry

end ratio_of_sam_to_sue_l1835_183524


namespace find_x_l1835_183530

theorem find_x :
  ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 ∧
  x = 69.11764705882353 :=
by
  sorry

end find_x_l1835_183530


namespace sum_of_coefficients_eq_10_l1835_183556

theorem sum_of_coefficients_eq_10 
  (s : ℕ → ℝ) 
  (a b c : ℝ) 
  (h0 : s 0 = 3) 
  (h1 : s 1 = 5) 
  (h2 : s 2 = 9)
  (h : ∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) : 
  a + b + c = 10 :=
sorry

end sum_of_coefficients_eq_10_l1835_183556


namespace yoongi_initial_books_l1835_183559

theorem yoongi_initial_books 
  (Y E U : ℕ)
  (h1 : Y - 5 + 15 = 45)
  (h2 : E + 5 - 10 = 45)
  (h3 : U - 15 + 10 = 45) : 
  Y = 35 := 
by 
  -- To be completed with proof
  sorry

end yoongi_initial_books_l1835_183559


namespace union_P_Q_l1835_183597

noncomputable def P : Set ℝ := {x : ℝ | abs x ≥ 3}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x - 1}

theorem union_P_Q :
  (P ∪ Q) = Set.Iic (-3) ∪ Set.Ici (-1) :=
by {
  sorry
}

end union_P_Q_l1835_183597


namespace shiela_neighbors_l1835_183538

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) : neighbors = total_drawings / drawings_per_neighbor :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end shiela_neighbors_l1835_183538


namespace june_earnings_l1835_183508

theorem june_earnings (total_clovers : ℕ) (percent_three : ℝ) (percent_two : ℝ) (percent_four : ℝ) :
  total_clovers = 200 →
  percent_three = 0.75 →
  percent_two = 0.24 →
  percent_four = 0.01 →
  (total_clovers * percent_three + total_clovers * percent_two + total_clovers * percent_four) = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end june_earnings_l1835_183508


namespace probability_of_3_black_2_white_l1835_183573

def total_balls := 15
def black_balls := 10
def white_balls := 5
def drawn_balls := 5
def drawn_black_balls := 3
def drawn_white_balls := 2

noncomputable def probability_black_white_draw : ℝ :=
  (Nat.choose black_balls drawn_black_balls * Nat.choose white_balls drawn_white_balls : ℝ) /
  (Nat.choose total_balls drawn_balls : ℝ)

theorem probability_of_3_black_2_white :
  probability_black_white_draw = 400 / 1001 := by
  sorry

end probability_of_3_black_2_white_l1835_183573


namespace find_abcdef_l1835_183500

def repeating_decimal_to_fraction_abcd (a b c d : ℕ) : ℚ :=
  (1000 * a + 100 * b + 10 * c + d) / 9999

def repeating_decimal_to_fraction_abcdef (a b c d e f : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) / 999999

theorem find_abcdef :
  ∀ a b c d e f : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  (repeating_decimal_to_fraction_abcd a b c d + repeating_decimal_to_fraction_abcdef a b c d e f = 49 / 999) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 490) :=
by
  repeat {sorry}

end find_abcdef_l1835_183500


namespace trigonometric_expression_simplification_l1835_183511

theorem trigonometric_expression_simplification
  (α : ℝ) 
  (hα : α = 49 * Real.pi / 48) :
  4 * (Real.sin α ^ 3 * Real.cos (3 * α) + 
       Real.cos α ^ 3 * Real.sin (3 * α)) * 
  Real.cos (4 * α) = 0.75 := 
by 
  sorry

end trigonometric_expression_simplification_l1835_183511


namespace probability_not_paired_shoes_l1835_183520

noncomputable def probability_not_pair (total_shoes : ℕ) (pairs : ℕ) (shoes_drawn : ℕ) : ℚ :=
  let total_ways := Nat.choose total_shoes shoes_drawn
  let pair_ways := pairs * Nat.choose 2 2
  let not_pair_ways := total_ways - pair_ways
  not_pair_ways / total_ways

theorem probability_not_paired_shoes (total_shoes pairs shoes_drawn : ℕ) (h1 : total_shoes = 6) 
(h2 : pairs = 3) (h3 : shoes_drawn = 2) :
  probability_not_pair total_shoes pairs shoes_drawn = 4 / 5 :=
by 
  rw [h1, h2, h3]
  simp [probability_not_pair, Nat.choose]
  sorry

end probability_not_paired_shoes_l1835_183520


namespace least_five_digit_congruent_to_8_mod_17_l1835_183537

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end least_five_digit_congruent_to_8_mod_17_l1835_183537


namespace geometric_series_six_terms_l1835_183560

theorem geometric_series_six_terms :
  (1/4 - 1/16 + 1/64 - 1/256 + 1/1024 - 1/4096 : ℚ) = 4095 / 20480 :=
by
  sorry

end geometric_series_six_terms_l1835_183560


namespace combined_eel_length_l1835_183568

def Lengths : Type := { j : ℕ // j = 16 }

def jenna_eel_length : Lengths := ⟨16, rfl⟩

def bill_eel_length (j : Lengths) : ℕ := 3 * j.val

#check bill_eel_length

theorem combined_eel_length (j : Lengths) :
  j.val + bill_eel_length j = 64 :=
by
  -- The proof would go here
  sorry

end combined_eel_length_l1835_183568


namespace map_area_l1835_183528

def length : ℕ := 5
def width : ℕ := 2
def area_of_map (length width : ℕ) : ℕ := length * width

theorem map_area : area_of_map length width = 10 := by
  sorry

end map_area_l1835_183528


namespace overall_average_score_l1835_183527

theorem overall_average_score (students_total : ℕ) (scores_day1 : ℕ) (avg1 : ℝ)
  (scores_day2 : ℕ) (avg2 : ℝ) (scores_day3 : ℕ) (avg3 : ℝ)
  (h1 : students_total = 45)
  (h2 : scores_day1 = 35)
  (h3 : avg1 = 0.65)
  (h4 : scores_day2 = 8)
  (h5 : avg2 = 0.75)
  (h6 : scores_day3 = 2)
  (h7 : avg3 = 0.85) :
  (scores_day1 * avg1 + scores_day2 * avg2 + scores_day3 * avg3) / students_total = 0.68 :=
by
  -- Lean proof goes here
  sorry

end overall_average_score_l1835_183527


namespace brother_catch_up_in_3_minutes_l1835_183585

variables (v_s v_b : ℝ) (t t_new : ℝ)

-- Conditions
def brother_speed_later_leaves_catch (v_b : ℝ) (v_s : ℝ) : Prop :=
18 * v_s = 12 * v_b

def new_speed_of_brother (v_b v_s : ℝ) : ℝ :=
2 * v_b

def time_to_catch_up (v_s : ℝ) (t_new : ℝ) : Prop :=
6 + t_new = 3 * t_new

-- Goal: prove that t_new = 3
theorem brother_catch_up_in_3_minutes (v_s v_b : ℝ) (t_new : ℝ) :
  (brother_speed_later_leaves_catch v_b v_s) → 
  (new_speed_of_brother v_b v_s) = 3 * v_s → 
  time_to_catch_up v_s t_new → 
  t_new = 3 :=
by sorry

end brother_catch_up_in_3_minutes_l1835_183585


namespace no_solution_for_k_eq_2_l1835_183583

theorem no_solution_for_k_eq_2 :
  ∀ m n : ℕ, m ≠ n → ¬ (lcm m n - gcd m n = 2 * (m - n)) :=
by
  sorry

end no_solution_for_k_eq_2_l1835_183583


namespace fraction_to_decimal_l1835_183584

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l1835_183584


namespace number_of_books_l1835_183549

theorem number_of_books (original_books new_books : ℕ) (h1 : original_books = 35) (h2 : new_books = 56) : 
  original_books + new_books = 91 :=
by {
  -- the proof will go here, but is not required for the statement
  sorry
}

end number_of_books_l1835_183549


namespace english_score_l1835_183551

theorem english_score (s1 s2 s3 e : ℕ) :
  (s1 + s2 + s3) = 276 → (s1 + s2 + s3 + e) = 376 → e = 100 :=
by
  intros h1 h2
  sorry

end english_score_l1835_183551


namespace derivative_of_y_l1835_183523

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (log 7) * (sin (7 * x)) ^ 2) / (7 * cos (14 * x))

theorem derivative_of_y (x : ℝ) : deriv y x = (cos (log 7) * tan (14 * x)) / cos (14 * x) := sorry

end derivative_of_y_l1835_183523


namespace intersection_A_B_l1835_183594

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l1835_183594


namespace factorize_polynomial_find_value_l1835_183509

-- Problem 1: Factorize a^3 - 3a^2 - 4a + 12
theorem factorize_polynomial (a : ℝ) :
  a^3 - 3 * a^2 - 4 * a + 12 = (a - 3) * (a - 2) * (a + 2) :=
sorry

-- Problem 2: Given m + n = 5 and m - n = 1, prove m^2 - n^2 + 2m - 2n = 7
theorem find_value (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) :
  m^2 - n^2 + 2 * m - 2 * n = 7 :=
sorry

end factorize_polynomial_find_value_l1835_183509


namespace remainder_of_3_pow_101_plus_4_mod_5_l1835_183547

theorem remainder_of_3_pow_101_plus_4_mod_5 :
  (3^101 + 4) % 5 = 2 :=
by
  have h1 : 3 % 5 = 3 := by sorry
  have h2 : (3^2) % 5 = 4 := by sorry
  have h3 : (3^3) % 5 = 2 := by sorry
  have h4 : (3^4) % 5 = 1 := by sorry
  -- more steps to show the pattern and use it to prove the final statement
  sorry

end remainder_of_3_pow_101_plus_4_mod_5_l1835_183547


namespace minimum_score_for_advanced_course_l1835_183506

theorem minimum_score_for_advanced_course (q1 q2 q3 q4 : ℕ) (H1 : q1 = 88) (H2 : q2 = 84) (H3 : q3 = 82) :
  (q1 + q2 + q3 + q4) / 4 ≥ 85 → q4 = 86 := by
  sorry

end minimum_score_for_advanced_course_l1835_183506


namespace elly_candies_l1835_183540

theorem elly_candies (a b c : ℝ) (h1 : a * b * c = 216) : 
  24 * 216 = 5184 :=
by
  sorry

end elly_candies_l1835_183540


namespace length_of_AE_l1835_183569

/-- Given the conditions on the pentagon ABCDE:
1. AB = 2, BC = 2, CD = 5, DE = 7
2. AC is the largest side in triangle ABC
3. CE is the smallest side in triangle ECD
4. In triangle ACE all sides are integers and have distinct lengths,
prove that the length of side AE is 5. -/
theorem length_of_AE
  (AB BC CD DE : ℕ)
  (hAB : AB = 2)
  (hBC : BC = 2)
  (hCD : CD = 5)
  (hDE : DE = 7)
  (AC : ℕ) 
  (hAC_large : AB < AC ∧ BC < AC)
  (CE : ℕ)
  (hCE_small : CE < CD ∧ CE < DE)
  (AE : ℕ)
  (distinct_sides : ∀ x y z : ℕ, x ≠ y → x ≠ z → y ≠ z → (AC = x ∨ CE = x ∨ AE = x) → (AC = y ∨ CE = y ∨ AE = y) → (AC = z ∨ CE = z ∨ AE = z)) :
  AE = 5 :=
sorry

end length_of_AE_l1835_183569


namespace vec_expression_l1835_183552

def vec_a : ℝ × ℝ := (1, -2)
def vec_b : ℝ × ℝ := (3, 5)

theorem vec_expression : 2 • vec_a + vec_b = (5, 1) := by
  sorry

end vec_expression_l1835_183552


namespace cost_of_orange_juice_l1835_183564

theorem cost_of_orange_juice (O : ℝ) (H1 : ∀ (apple_juice_cost : ℝ), apple_juice_cost = 0.60 ):
  let total_bottles := 70
  let total_cost := 46.20
  let orange_juice_bottles := 42
  let apple_juice_bottles := total_bottles - orange_juice_bottles
  let equation := (orange_juice_bottles * O + apple_juice_bottles * 0.60 = total_cost)
  equation -> O = 0.70 := by
  sorry

end cost_of_orange_juice_l1835_183564


namespace standard_deviation_of_data_l1835_183517

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (fun x => (x - m)^2)).sum / data.length

noncomputable def std_dev (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

theorem standard_deviation_of_data :
  std_dev [5, 7, 7, 8, 10, 11] = 2 := 
sorry

end standard_deviation_of_data_l1835_183517


namespace parabola_unique_solution_l1835_183501

theorem parabola_unique_solution (a : ℝ) :
  (∃ x : ℝ, (0 ≤ x^2 + a * x + 5) ∧ (x^2 + a * x + 5 ≤ 4)) → (a = 2 ∨ a = -2) :=
by
  sorry

end parabola_unique_solution_l1835_183501


namespace average_speed_is_correct_l1835_183535
noncomputable def average_speed_trip : ℝ :=
  let distance_AB := 240 * 5
  let distance_BC := 300 * 3
  let distance_CD := 400 * 4
  let total_distance := distance_AB + distance_BC + distance_CD
  let flight_time_AB := 5
  let layover_B := 2
  let flight_time_BC := 3
  let layover_C := 1
  let flight_time_CD := 4
  let total_time := (flight_time_AB + flight_time_BC + flight_time_CD) + (layover_B + layover_C)
  total_distance / total_time

theorem average_speed_is_correct :
  average_speed_trip = 246.67 := sorry

end average_speed_is_correct_l1835_183535


namespace hyperbola_eccentricity_l1835_183570

theorem hyperbola_eccentricity (a b c e : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 → (x, y) ≠ (3, -4))
  (h2 : b / a = 4 / 3)
  (h3 : b^2 = c^2 - a^2)
  (h4 : c / a = e):
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l1835_183570


namespace conor_chop_eggplants_l1835_183590

theorem conor_chop_eggplants (E : ℕ) 
  (condition1 : E + 9 + 8 = (E + 17))
  (condition2 : 4 * (E + 9 + 8) = 116) :
  E = 12 :=
by {
  sorry
}

end conor_chop_eggplants_l1835_183590


namespace modular_inverse_l1835_183513

theorem modular_inverse :
  (24 * 22) % 53 = 1 :=
by
  have h1 : (24 * -29) % 53 = (53 * 0 - 29 * 24) % 53 := by sorry
  have h2 : (24 * -29) % 53 = (-29 * 24) % 53 := by sorry
  have h3 : (-29 * 24) % 53 = (-29 % 53 * 24 % 53 % 53) := by sorry
  have h4 : -29 % 53 = 53 - 24 := by sorry
  have h5 : (53 - 29) % 53 = (22 * 22) % 53 := by sorry
  have h6 : (22 * 22) % 53 = (24 * 22) % 53 := by sorry
  have h7 : (24 * 22) % 53 = 1 := by sorry
  exact h7

end modular_inverse_l1835_183513


namespace functional_equation_solution_l1835_183519

theorem functional_equation_solution (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (f x = (1/2) * (x + 1 - 1/x - 1/(1-x))) →
  (f x + f (1 / (1 - x)) = x) :=
sorry

end functional_equation_solution_l1835_183519


namespace polynomial_discriminant_l1835_183543

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l1835_183543


namespace building_height_l1835_183505

-- Definitions of the conditions
def wooden_box_height : ℝ := 3
def wooden_box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- The statement that needs to be proved
theorem building_height : ∃ (height : ℝ), height = 9 ∧ wooden_box_height / wooden_box_shadow = height / building_shadow :=
by
  sorry

end building_height_l1835_183505


namespace find_garden_perimeter_l1835_183503

noncomputable def garden_perimeter (a : ℝ) (P : ℝ) : Prop :=
  a = 2 * P + 14.25 ∧ a = 90.25

theorem find_garden_perimeter :
  ∃ P : ℝ, garden_perimeter 90.25 P ∧ P = 38 :=
by
  sorry

end find_garden_perimeter_l1835_183503


namespace correct_choice_option_D_l1835_183529

theorem correct_choice_option_D : (500 - 9 * 7 = 437) := by sorry

end correct_choice_option_D_l1835_183529


namespace sunzi_problem_solution_l1835_183550

theorem sunzi_problem_solution (x y : ℝ) :
  (y = x + 4.5) ∧ (0.5 * y = x - 1) ↔ (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by 
  sorry

end sunzi_problem_solution_l1835_183550


namespace complex_series_sum_l1835_183574

theorem complex_series_sum (ω : ℂ) (h₁ : ω^7 = 1) (h₂ : ω ≠ 1) :
  (ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + 
   ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + ω^44 + ω^46 + ω^48 + ω^50 + 
   ω^52 + ω^54) = -1 :=
by
  sorry

end complex_series_sum_l1835_183574


namespace quadratic_int_roots_iff_n_eq_3_or_4_l1835_183593

theorem quadratic_int_roots_iff_n_eq_3_or_4 (n : ℕ) (hn : 0 < n) :
    (∃ m k : ℤ, (m ≠ k) ∧ (m^2 - 4 * m + n = 0) ∧ (k^2 - 4 * k + n = 0)) ↔ (n = 3 ∨ n = 4) := sorry

end quadratic_int_roots_iff_n_eq_3_or_4_l1835_183593


namespace proof_of_problem_l1835_183589

noncomputable def problem : Prop :=
  (1 + Real.cos (20 * Real.pi / 180)) / (2 * Real.sin (20 * Real.pi / 180)) -
  (Real.sin (10 * Real.pi / 180) * 
  (1 / Real.tan (5 * Real.pi / 180) - Real.tan (5 * Real.pi / 180))) =
  (Real.sqrt 3) / 2

theorem proof_of_problem : problem :=
by
  sorry

end proof_of_problem_l1835_183589


namespace fruit_punch_total_l1835_183514

section fruit_punch
variable (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) (total_punch : ℝ)

axiom h1 : orange_punch = 4.5
axiom h2 : cherry_punch = 2 * orange_punch
axiom h3 : apple_juice = cherry_punch - 1.5
axiom h4 : total_punch = orange_punch + cherry_punch + apple_juice

theorem fruit_punch_total : total_punch = 21 := sorry

end fruit_punch

end fruit_punch_total_l1835_183514


namespace no_such_function_exists_l1835_183565

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∃ M > 0, ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧
                    (f 1 = 1) ∧
                    (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end no_such_function_exists_l1835_183565


namespace find_quotient_l1835_183567

def dividend : ℕ := 55053
def divisor : ℕ := 456
def remainder : ℕ := 333

theorem find_quotient (Q : ℕ) (h : dividend = (divisor * Q) + remainder) : Q = 120 := by
  sorry

end find_quotient_l1835_183567


namespace spheres_volume_ratio_l1835_183576

theorem spheres_volume_ratio (S1 S2 V1 V2 : ℝ)
  (h1 : S1 / S2 = 1 / 9) 
  (h2a : S1 = 4 * π * r1^2) 
  (h2b : S2 = 4 * π * r2^2)
  (h3a : V1 = 4 / 3 * π * r1^3)
  (h3b : V2 = 4 / 3 * π * r2^3)
  : V1 / V2 = 1 / 27 :=
by
  sorry

end spheres_volume_ratio_l1835_183576


namespace range_of_a_l1835_183577

/-- Given that the point (1, 1) is located inside the circle (x - a)^2 + (y + a)^2 = 4, 
    proving that the range of values for a is -1 < a < 1. -/
theorem range_of_a (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → 
  (-1 < a ∧ a < 1) :=
by
  intro h
  sorry

end range_of_a_l1835_183577


namespace sum_of_first_four_terms_l1835_183539

noncomputable def sum_first_n_terms (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_first_four_terms :
  ∀ (a q : ℝ), a * (1 + q) = 7 → a * (q^6 - 1) / (q - 1) = 91 →
  a * (1 + q + q^2 + q^3) = 28 :=
by
  intros a q h₁ h₂
  -- Proof omitted
  sorry

end sum_of_first_four_terms_l1835_183539


namespace complex_addition_l1835_183591

def imag_unit_squared (i : ℂ) : Prop := i * i = -1

theorem complex_addition (a b : ℝ) (i : ℂ)
  (h1 : a + b * i = i * i)
  (h2 : imag_unit_squared i) : a + b = -1 := 
sorry

end complex_addition_l1835_183591


namespace remaining_candy_l1835_183521

def initial_candy : ℕ := 36
def ate_candy1 : ℕ := 17
def ate_candy2 : ℕ := 15
def total_ate_candy : ℕ := ate_candy1 + ate_candy2

theorem remaining_candy : initial_candy - total_ate_candy = 4 := by
  sorry

end remaining_candy_l1835_183521


namespace part_I_part_II_l1835_183541

noncomputable def vector_a : ℝ × ℝ := (4, 3)
noncomputable def vector_b : ℝ × ℝ := (5, -12)
noncomputable def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vector_magnitude_sum := magnitude vector_sum
noncomputable def magnitude_a := magnitude vector_a
noncomputable def magnitude_b := magnitude vector_b
noncomputable def cos_theta := dot_product vector_a vector_b / (magnitude_a * magnitude_b)

-- Prove the magnitude of the sum of vectors is 9√2
theorem part_I : vector_magnitude_sum = 9 * Real.sqrt 2 :=
by
  sorry

-- Prove the cosine of the angle between the vectors is -16/65
theorem part_II : cos_theta = -16 / 65 :=
by
  sorry

end part_I_part_II_l1835_183541


namespace malia_berries_second_bush_l1835_183507

theorem malia_berries_second_bush :
  ∀ (b2 : ℕ), ∃ (d1 d2 d3 d4 : ℕ),
  d1 = 3 → d2 = 7 → d3 = 12 → d4 = 19 →
  d2 - d1 = (d3 - d2) - 2 →
  d3 - d2 = (d4 - d3) - 2 →
  b2 = d1 + (d2 - d1 - 2) →
  b2 = 6 :=
by
  sorry

end malia_berries_second_bush_l1835_183507


namespace simplify_expression_l1835_183544

variable (y : ℝ)
variable (h : y ≠ 0)

theorem simplify_expression : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry

end simplify_expression_l1835_183544


namespace total_items_proof_l1835_183542

noncomputable def totalItemsBought (budget : ℕ) (sandwichCost : ℕ) 
  (pastryCost : ℕ) (maxSandwiches : ℕ) : ℕ :=
  let s := min (budget / sandwichCost) maxSandwiches
  let remainingMoney := budget - s * sandwichCost
  let p := remainingMoney / pastryCost
  s + p

theorem total_items_proof : totalItemsBought 50 6 2 7 = 11 := by
  sorry

end total_items_proof_l1835_183542


namespace inequality_solution_set_range_of_k_l1835_183562

variable {k m x : ℝ}

theorem inequality_solution_set (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k)) 
  (sol_set_f_x_gt_m : ∀ x, f x > m ↔ (x < -3 ∨ x > -2)) :
  -1 < x ∧ x < 3 / 2 := 
sorry

theorem range_of_k (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k))
  (exists_f_x_gt_1 : ∃ x > 3, f x > 1) : 
  k > 12 :=
sorry

end inequality_solution_set_range_of_k_l1835_183562
