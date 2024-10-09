import Mathlib

namespace pencils_per_student_l1227_122780

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ)
    (h1 : total_pencils = 125)
    (h2 : students = 25)
    (h3 : pencils_per_student = total_pencils / students) :
    pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l1227_122780


namespace visible_during_metaphase_l1227_122792

-- Define the structures which could be present in a plant cell during mitosis.
inductive Structure
| Chromosomes
| Spindle
| CellWall
| MetaphasePlate
| CellMembrane
| Nucleus
| Nucleolus

open Structure

-- Define what structures are visible during metaphase.
def visibleStructures (phase : String) : Set Structure :=
  if phase = "metaphase" then
    {Chromosomes, Spindle, CellWall}
  else
    ∅

-- The proof statement
theorem visible_during_metaphase :
  visibleStructures "metaphase" = {Chromosomes, Spindle, CellWall} :=
by
  sorry

end visible_during_metaphase_l1227_122792


namespace mean_of_five_integers_l1227_122743

theorem mean_of_five_integers
  (p q r s t : ℤ)
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 :=
by
  sorry

end mean_of_five_integers_l1227_122743


namespace quadratic_real_solutions_l1227_122758

theorem quadratic_real_solutions (m : ℝ) :
  (∃ (x : ℝ), m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_solutions_l1227_122758


namespace hypotenuse_length_triangle_l1227_122731

theorem hypotenuse_length_triangle (a b c : ℝ) (h1 : a + b + c = 40) (h2 : (1/2) * a * b = 30) 
  (h3 : a = b) : c = 2 * Real.sqrt 30 :=
by
  sorry

end hypotenuse_length_triangle_l1227_122731


namespace monthly_installments_l1227_122781

theorem monthly_installments (cash_price deposit installment saving : ℕ) (total_paid installments_made : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment = 300 →
  saving = 4000 →
  total_paid = cash_price + saving →
  installments_made = (total_paid - deposit) / installment →
  installments_made = 30 :=
by
  intros h_cash_price h_deposit h_installment h_saving h_total_paid h_installments_made
  sorry

end monthly_installments_l1227_122781


namespace chemistry_club_student_count_l1227_122767

theorem chemistry_club_student_count (x : ℕ) (h1 : x % 3 = 0)
  (h2 : x % 4 = 0) (h3 : x % 6 = 0)
  (h4 : (x / 3) = (x / 4) + 3) :
  (x / 6) = 6 :=
by {
  -- Proof goes here
  sorry
}

end chemistry_club_student_count_l1227_122767


namespace compare_y1_y2_l1227_122757

theorem compare_y1_y2 (a : ℝ) (y1 y2 : ℝ) (h₁ : a < 0) (h₂ : y1 = a * (-1 - 1)^2 + 3) (h₃ : y2 = a * (2 - 1)^2 + 3) : 
  y1 < y2 :=
by
  sorry

end compare_y1_y2_l1227_122757


namespace det_of_matrix_M_l1227_122704

open Matrix

def M : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![2, -4, 4], 
    ![0, 6, -2], 
    ![5, -3, 2]]

theorem det_of_matrix_M : Matrix.det M = -68 :=
by
  sorry

end det_of_matrix_M_l1227_122704


namespace max_gcd_of_13n_plus_3_and_7n_plus_1_l1227_122790

theorem max_gcd_of_13n_plus_3_and_7n_plus_1 (n : ℕ) (hn : 0 < n) :
  ∃ d, d = Nat.gcd (13 * n + 3) (7 * n + 1) ∧ ∀ m, m = Nat.gcd (13 * n + 3) (7 * n + 1) → m ≤ 8 := 
sorry

end max_gcd_of_13n_plus_3_and_7n_plus_1_l1227_122790


namespace triangle_area_is_4_l1227_122747

-- Define the lines
def line1 (x : ℝ) : ℝ := 4
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define intersection points
def intersection1 : ℝ × ℝ := (2, 4)
def intersection2 : ℝ × ℝ := (-2, 4)
def intersection3 : ℝ × ℝ := (0, 2)

-- Function to calculate the area of a triangle using its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Statement of the proof problem
theorem triangle_area_is_4 :
  ∀ A B C : ℝ × ℝ, A = intersection1 → B = intersection2 → C = intersection3 →
  triangle_area A B C = 4 := by
  sorry

end triangle_area_is_4_l1227_122747


namespace second_divisor_27_l1227_122749

theorem second_divisor_27 (N : ℤ) (D : ℤ) (k : ℤ) (q : ℤ) (h1 : N = 242 * k + 100) (h2 : N = D * q + 19) : D = 27 := by
  sorry

end second_divisor_27_l1227_122749


namespace faculty_reduction_l1227_122752

theorem faculty_reduction (x : ℝ) (h1 : 0.75 * x = 195) : x = 260 :=
by sorry

end faculty_reduction_l1227_122752


namespace real_solution_exists_l1227_122746

theorem real_solution_exists (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) :
  (x^3 - 4*x^2) / (x^2 - 5*x + 6) - x = 9 → x = 9/2 :=
by sorry

end real_solution_exists_l1227_122746


namespace hyperbola_eccentricity_l1227_122726

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), a = 3 → b = 4 → c = Real.sqrt (a^2 + b^2) → c / a = 5 / 3 :=
by
  intros a b c ha hb h_eq
  sorry

end hyperbola_eccentricity_l1227_122726


namespace part_a_part_b_l1227_122700

variable (p : ℕ)
variable (h1 : prime p)
variable (h2 : p > 3)

theorem part_a : (p + 1) % 4 = 0 ∨ (p - 1) % 4 = 0 :=
sorry

theorem part_b : ¬ ((p + 1) % 5 = 0 ∨ (p - 1) % 5 = 0) :=
sorry

end part_a_part_b_l1227_122700


namespace simple_interest_rate_l1227_122713

/-- Prove that given Principal (P) = 750, Amount (A) = 900, and Time (T) = 5 years,
    the rate (R) such that the Simple Interest formula holds is 4 percent. -/
theorem simple_interest_rate :
  ∀ (P A T : ℕ) (R : ℕ),
    P = 750 → 
    A = 900 → 
    T = 5 → 
    A = P + (P * R * T / 100) →
    R = 4 :=
by
  intros P A T R hP hA hT h_si
  sorry

end simple_interest_rate_l1227_122713


namespace number_of_people_l1227_122710

theorem number_of_people
  (x y : ℕ)
  (h1 : x + y = 28)
  (h2 : 2 * x + 4 * y = 92) :
  x = 10 :=
by
  sorry

end number_of_people_l1227_122710


namespace student_avg_always_greater_l1227_122701

theorem student_avg_always_greater (x y z : ℝ) (h1 : x < y) (h2 : y < z) : 
  ( ( (x + y) / 2 + z) / 2 ) > ( (x + y + z) / 3 ) :=
by
  sorry

end student_avg_always_greater_l1227_122701


namespace proof_PQ_expression_l1227_122734

theorem proof_PQ_expression (P Q : ℝ) (h1 : P^2 - P * Q = 1) (h2 : 4 * P * Q - 3 * Q^2 = 2) : 
  P^2 + 3 * P * Q - 3 * Q^2 = 3 :=
by
  sorry

end proof_PQ_expression_l1227_122734


namespace a_minus_b_eq_three_l1227_122719

theorem a_minus_b_eq_three (a b : ℝ) (h : (a+bi) * i = 1 + 2 * i) : a - b = 3 :=
by
  sorry

end a_minus_b_eq_three_l1227_122719


namespace combined_final_selling_price_correct_l1227_122772

def itemA_cost : Float := 180.0
def itemB_cost : Float := 220.0
def itemC_cost : Float := 130.0

def itemA_profit_margin : Float := 0.15
def itemB_profit_margin : Float := 0.20
def itemC_profit_margin : Float := 0.25

def itemA_tax_rate : Float := 0.05
def itemB_discount_rate : Float := 0.10
def itemC_tax_rate : Float := 0.08

def itemA_selling_price_before_tax := itemA_cost * (1 + itemA_profit_margin)
def itemB_selling_price_before_discount := itemB_cost * (1 + itemB_profit_margin)
def itemC_selling_price_before_tax := itemC_cost * (1 + itemC_profit_margin)

def itemA_final_price := itemA_selling_price_before_tax * (1 + itemA_tax_rate)
def itemB_final_price := itemB_selling_price_before_discount * (1 - itemB_discount_rate)
def itemC_final_price := itemC_selling_price_before_tax * (1 + itemC_tax_rate)

def combined_final_price := itemA_final_price + itemB_final_price + itemC_final_price

theorem combined_final_selling_price_correct : 
  combined_final_price = 630.45 :=
by
  -- proof would go here
  sorry

end combined_final_selling_price_correct_l1227_122772


namespace min_perimeter_l1227_122745

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the coordinates of the right focus, point on the hyperbola, and point M
def right_focus (F : ℝ × ℝ) : Prop := F = (2, 0)
def point_on_left_branch (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ hyperbola P.1 P.2
def point_M (M : ℝ × ℝ) : Prop := M = (0, 2)

-- Perimeter of ΔPFM
noncomputable def perimeter (P F M : ℝ × ℝ) : ℝ :=
  let PF := (P.1 - F.1)^2 + (P.2 - F.2)^2
  let PM := (P.1 - M.1)^2 + (P.2 - M.2)^2
  let MF := (M.1 - F.1)^2 + (M.2 - F.2)^2
  PF.sqrt + PM.sqrt + MF.sqrt

-- Theorem statement
theorem min_perimeter (P F M : ℝ × ℝ) 
  (hF : right_focus F)
  (hP : point_on_left_branch P)
  (hM : point_M M) :
  ∃ P, perimeter P F M = 2 + 4 * Real.sqrt 2 :=
sorry

end min_perimeter_l1227_122745


namespace volume_ratio_l1227_122793

variable (A B : ℝ)

theorem volume_ratio (h1 : (3 / 4) * A = (5 / 8) * B) :
  A / B = 5 / 6 :=
by
  sorry

end volume_ratio_l1227_122793


namespace a_share_is_1400_l1227_122711

-- Definitions for the conditions
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def share_B : ℕ := 2200

-- Definition for the ratios
def ratio_A : ℚ := investment_A / 1000
def ratio_B : ℚ := investment_B / 1000
def ratio_C : ℚ := investment_C / 1000

-- Sum of ratios
def sum_ratios : ℚ := ratio_A + ratio_B + ratio_C

-- Total profit P can be deduced from B's share
def total_profit : ℚ := share_B * sum_ratios / ratio_B

-- Goal: Prove that A's share is $1400
def share_A : ℚ := ratio_A * total_profit / sum_ratios

theorem a_share_is_1400 : share_A = 1400 :=
sorry

end a_share_is_1400_l1227_122711


namespace ratio_lcm_gcf_256_162_l1227_122799

theorem ratio_lcm_gcf_256_162 : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := 
by 
  sorry

end ratio_lcm_gcf_256_162_l1227_122799


namespace direct_proportion_function_l1227_122744

theorem direct_proportion_function (m : ℝ) : 
  (m^2 + 2 * m ≠ 0) ∧ (m^2 - 3 = 1) → m = 2 :=
by {
  sorry
}

end direct_proportion_function_l1227_122744


namespace prob_correct_l1227_122761

-- Define percentages as ratio values
def prob_beginner_excel : ℝ := 0.35
def prob_intermediate_excel : ℝ := 0.25
def prob_advanced_excel : ℝ := 0.20
def prob_no_excel : ℝ := 0.20

def prob_day_shift : ℝ := 0.70
def prob_night_shift : ℝ := 0.30

def prob_weekend : ℝ := 0.40
def prob_not_weekend : ℝ := 0.60

-- Define the target probability calculation
def prob_intermediate_or_advanced_excel : ℝ := prob_intermediate_excel + prob_advanced_excel
def prob_combined : ℝ := prob_intermediate_or_advanced_excel * prob_night_shift * prob_not_weekend

-- The proof problem statement
theorem prob_correct : prob_combined = 0.081 :=
by
  sorry

end prob_correct_l1227_122761


namespace rhombus_area_l1227_122717

theorem rhombus_area (s d1 d2 : ℝ)
  (h1 : s = Real.sqrt 113)
  (h2 : abs (d1 - d2) = 8)
  (h3 : s^2 = (d1 / 2)^2 + (d2 / 2)^2) :
  (d1 * d2) / 2 = 194 := by
  sorry

end rhombus_area_l1227_122717


namespace function_two_common_points_with_xaxis_l1227_122755

theorem function_two_common_points_with_xaxis (c : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x + c = 0 → x = -1 ∨ x = 1) → (c = -2 ∨ c = 2) :=
by
  sorry

end function_two_common_points_with_xaxis_l1227_122755


namespace monotonicity_of_f_odd_function_a_value_l1227_122796

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

-- Part 1: Prove that f(x) is monotonically increasing
theorem monotonicity_of_f (a : ℝ) : 
  ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := by
  intro x1 x2 hx
  sorry

-- Part 2: If f(x) is an odd function, find the value of a
theorem odd_function_a_value (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : 
  f a 0 = 0 → a = 1 / 2 := by
  intro h
  sorry

end monotonicity_of_f_odd_function_a_value_l1227_122796


namespace find_x_l1227_122795

theorem find_x (x : ℝ) (h : (0.4 + x) / 2 = 0.2025) : x = 0.005 :=
by
  sorry

end find_x_l1227_122795


namespace xiao_li_estimate_l1227_122750

variable (x y z : ℝ)

theorem xiao_li_estimate (h1 : x > y) (h2 : y > 0) (h3 : 0 < z):
    (x + z) + (y - z) = x + y := 
by 
sorry

end xiao_li_estimate_l1227_122750


namespace simplify_expression_l1227_122759

theorem simplify_expression (x y : ℝ) :
  3 * (x + y) ^ 2 - 7 * (x + y) + 8 * (x + y) ^ 2 + 6 * (x + y) = 
  11 * (x + y) ^ 2 - (x + y) :=
by
  sorry

end simplify_expression_l1227_122759


namespace right_triangle_legs_sum_l1227_122718

theorem right_triangle_legs_sum
  (x : ℕ)
  (h_even : Even x)
  (h_eq : x^2 + (x + 2)^2 = 34^2) :
  x + (x + 2) = 50 := 
by
  sorry

end right_triangle_legs_sum_l1227_122718


namespace angle_BDC_is_15_degrees_l1227_122708

theorem angle_BDC_is_15_degrees (A B C D : Type) (AB AC AD CD : ℝ) (angle_BAC : ℝ) :
  AB = AC → AC = AD → CD = 2 * AC → angle_BAC = 30 →
  ∃ angle_BDC, angle_BDC = 15 := 
by
  sorry

end angle_BDC_is_15_degrees_l1227_122708


namespace white_balls_count_l1227_122773

theorem white_balls_count (n : ℕ) (h : 8 / (8 + n : ℝ) = 0.4) : n = 12 := by
  sorry

end white_balls_count_l1227_122773


namespace arithmetic_sequence_sum_l1227_122735

theorem arithmetic_sequence_sum (x y : ℕ)
  (h₁ : ∃ d, 9 = 3 + d)  -- Common difference exists, d = 6
  (h₂ : ∃ n, 15 = 3 + n * 6)  -- Arithmetic sequence term verification
  (h₃ : y = 33 - 6)
  (h₄ : x = 27 - 6) : x + y = 48 :=
sorry

end arithmetic_sequence_sum_l1227_122735


namespace value_of_A_l1227_122714

-- Definitions for values in the factor tree, ensuring each condition is respected.
def D : ℕ := 3 * 2 * 2
def E : ℕ := 5 * 2
def B : ℕ := 3 * D
def C : ℕ := 5 * E
def A : ℕ := B * C

-- Assertion of the correct value for A
theorem value_of_A : A = 1800 := by
  -- Mathematical equivalence proof problem placeholder
  sorry

end value_of_A_l1227_122714


namespace prism_volume_l1227_122764

theorem prism_volume
  (l w h : ℝ)
  (h1 : l * w = 6.5)
  (h2 : w * h = 8)
  (h3 : l * h = 13) :
  l * w * h = 26 :=
by
  sorry

end prism_volume_l1227_122764


namespace geometric_sequence_S4_l1227_122709

/-
In the geometric sequence {a_n}, S_2 = 7, S_6 = 91. Prove that S_4 = 28.
-/

theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h_S2 : S 2 = 7) 
  (h_S6 : S 6 = 91) :
  S 4 = 28 := 
sorry

end geometric_sequence_S4_l1227_122709


namespace elias_purchased_50cent_items_l1227_122763

theorem elias_purchased_50cent_items :
  ∃ (a b c : ℕ), a + b + c = 50 ∧ (50 * a + 250 * b + 400 * c = 5000) ∧ (a = 40) :=
by {
  sorry
}

end elias_purchased_50cent_items_l1227_122763


namespace fraction_division_l1227_122739

theorem fraction_division: 
  ((3 + 1 / 2) / 7) / (5 / 3) = 3 / 10 := 
by 
  sorry

end fraction_division_l1227_122739


namespace minimize_squares_in_rectangle_l1227_122733

theorem minimize_squares_in_rectangle (w h : ℕ) (hw : w = 63) (hh : h = 42) : 
  ∃ s : ℕ, s = Nat.gcd w h ∧ s = 21 :=
by
  sorry

end minimize_squares_in_rectangle_l1227_122733


namespace problem_1_problem_2_l1227_122754

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem problem_1 (x : ℝ) : f x + x^2 - 4 > 0 ↔ (x > 2 ∨ x < -1) := sorry

theorem problem_2 {m : ℝ} (h : m > 3) : ∃ x : ℝ, f x < g x m := sorry

end problem_1_problem_2_l1227_122754


namespace find_vertex_C_l1227_122762

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)
def euler_line (x y : ℝ) : Prop := x - y + 2 = 0

theorem find_vertex_C 
  (C : ℝ × ℝ)
  (h_centroid : (2 + C.1) / 3 = (4 + C.2) / 3)
  (h_euler_line : euler_line ((2 + C.1) / 3) ((4 + C.2) / 3))
  (h_circumcenter : (C.1 + 1)^2 + (C.2 - 1)^2 = 10) :
  C = (-4, 0) :=
sorry

end find_vertex_C_l1227_122762


namespace probability_three_white_balls_l1227_122716

def total_balls := 11
def white_balls := 5
def black_balls := 6
def balls_drawn := 5
def white_balls_drawn := 3
def black_balls_drawn := 2

theorem probability_three_white_balls :
  let total_outcomes := Nat.choose total_balls balls_drawn
  let favorable_outcomes := (Nat.choose white_balls white_balls_drawn) * (Nat.choose black_balls black_balls_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 77 :=
by
  sorry

end probability_three_white_balls_l1227_122716


namespace B_can_win_with_initial_config_B_l1227_122786

def initial_configuration_B := (6, 2, 1)

def A_starts_and_B_wins (config : (Nat × Nat × Nat)) : Prop := sorry

theorem B_can_win_with_initial_config_B : A_starts_and_B_wins initial_configuration_B :=
sorry

end B_can_win_with_initial_config_B_l1227_122786


namespace zero_cleverly_numbers_l1227_122774

theorem zero_cleverly_numbers (n : ℕ) : 
  (1000 ≤ n ∧ n < 10000) ∧ (∃ a b c, n = 1000 * a + 10 * b + c ∧ b = 0 ∧ 9 * (100 * a + 10 * b + c) = n) ↔ (n = 2025 ∨ n = 4050 ∨ n = 6075) := 
sorry

end zero_cleverly_numbers_l1227_122774


namespace calculate_expression_l1227_122730

-- Define the numerator and denominator
def numerator := 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1
def denominator := 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10

-- Prove the expression equals 1
theorem calculate_expression : (numerator / denominator) = 1 := by
  sorry

end calculate_expression_l1227_122730


namespace not_m_gt_132_l1227_122776

theorem not_m_gt_132 (m : ℕ) (hm : 0 < m)
  (H : ∃ (k : ℕ), 1 / 2 + 1 / 3 + 1 / 11 + 1 / (m:ℚ) = k) :
  m ≤ 132 :=
sorry

end not_m_gt_132_l1227_122776


namespace q1_monotonic_increasing_intervals_q2_proof_l1227_122760

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x^2 - (2 * a + 1) * x + 2 * Real.log x

theorem q1_monotonic_increasing_intervals (a : ℝ) (h : a > 0) :
  (a > 1/2 ∧ (∀ x, (0 < x ∧ x < 1/a) ∨ (2 < x) → f a x > 0)) ∨
  (a = 1/2 ∧ (∀ x, 0 < x → f a x ≥ 0)) ∨
  (0 < a ∧ a < 1/2 ∧ (∀ x, (0 < x ∧ x < 2) ∨ (1/a < x) → f a x > 0)) := sorry

theorem q2_proof (x : ℝ) :
  (a = 0 ∧ x > 0 → f 0 x < 2 * Real.exp x - x - 4) := sorry

end q1_monotonic_increasing_intervals_q2_proof_l1227_122760


namespace find_n_l1227_122723

theorem find_n (n : ℕ) (hn : (n - 2) * (n - 3) / 12 = 14 / 3) : n = 10 := by
  sorry

end find_n_l1227_122723


namespace irrationals_l1227_122784

open Classical

variable (x : ℝ)

theorem irrationals (h : x^3 + 2 * x^2 + 10 * x = 20) : Irrational x ∧ Irrational (x^2) :=
by
  sorry

end irrationals_l1227_122784


namespace problem1_problem2_problem3_l1227_122738

-- Definition of the polynomial expansion
def poly (x : ℝ) := (1 - 2*x)^7

-- Definitions capturing the conditions directly
def a_0 := 1
def sum_a_1_to_a_7 := -2
def sum_a_1_3_5_7 := -1094
def sum_abs_a_0_to_a_7 := 2187

-- Lean statements for the proof problems
theorem problem1 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = sum_a_1_to_a_7 :=
sorry

theorem problem2 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 3 + a 5 + a 7 = sum_a_1_3_5_7 :=
sorry

theorem problem3 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  abs (a 0) + abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) + abs (a 6) + abs (a 7) = sum_abs_a_0_to_a_7 :=
sorry

end problem1_problem2_problem3_l1227_122738


namespace div_iff_div_l1227_122779

theorem div_iff_div {a b : ℤ} : (29 ∣ (3 * a + 2 * b)) ↔ (29 ∣ (11 * a + 17 * b)) := 
by sorry

end div_iff_div_l1227_122779


namespace no_such_six_tuples_exist_l1227_122703

theorem no_such_six_tuples_exist :
  ∀ (a b c x y z : ℕ),
    1 ≤ c → c ≤ b → b ≤ a →
    1 ≤ z → z ≤ y → y ≤ x →
    2 * a + b + 4 * c = 4 * x * y * z →
    2 * x + y + 4 * z = 4 * a * b * c →
    False :=
by
  intros a b c x y z h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end no_such_six_tuples_exist_l1227_122703


namespace distance_A_B_l1227_122765

noncomputable def distance_between_points (v_A v_B : ℝ) (t : ℝ) : ℝ := 5 * (6 * t / (2 / 3 * t))

theorem distance_A_B
  (v_A v_B : ℝ)
  (t : ℝ)
  (h1 : v_A = 1.2 * v_B)
  (h2 : ∃ distance_broken, distance_broken = 5)
  (h3 : ∃ delay, delay = (1 / 6) * 6 * t)
  (h4 : ∃ v_B_new, v_B_new = 1.6 * v_B)
  (h5 : distance_between_points v_A v_B t = 45) :
  distance_between_points v_A v_B t = 45 :=
sorry

end distance_A_B_l1227_122765


namespace distinct_solutions_difference_l1227_122753

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l1227_122753


namespace frequency_of_third_group_l1227_122766

theorem frequency_of_third_group (total_data first_group second_group fourth_group third_group : ℕ) 
    (h1 : total_data = 40)
    (h2 : first_group = 5)
    (h3 : second_group = 12)
    (h4 : fourth_group = 8) :
    third_group = 15 :=
by
  sorry

end frequency_of_third_group_l1227_122766


namespace max_unique_solution_l1227_122756

theorem max_unique_solution (x y : ℕ) (m : ℕ) (h : 2005 * x + 2007 * y = m) : 
  m = 2 * 2005 * 2007 ↔ ∃! (x y : ℕ), 2005 * x + 2007 * y = m :=
sorry

end max_unique_solution_l1227_122756


namespace zachary_needs_more_money_l1227_122725

def cost_of_football : ℝ := 3.75
def cost_of_shorts : ℝ := 2.40
def cost_of_shoes : ℝ := 11.85
def zachary_money : ℝ := 10.00
def total_cost : ℝ := cost_of_football + cost_of_shorts + cost_of_shoes
def amount_needed : ℝ := total_cost - zachary_money

theorem zachary_needs_more_money : amount_needed = 7.00 := by
  sorry

end zachary_needs_more_money_l1227_122725


namespace find_z_l1227_122788

theorem find_z (z : ℝ) 
    (cos_angle : (2 + 2 * z) / ((Real.sqrt (1 + z^2)) * 3) = 2 / 3) : 
    z = 0 := 
sorry

end find_z_l1227_122788


namespace bookshelf_prices_purchasing_plans_l1227_122705

/-
We are given the following conditions:
1. 3 * x + 2 * y = 1020
2. 4 * x + 3 * y = 1440

From these conditions, we need to prove that:
1. Price of type A bookshelf (x) is 180 yuan.
2. Price of type B bookshelf (y) is 240 yuan.

Given further conditions:
1. The school plans to purchase a total of 20 bookshelves.
2. Type B bookshelves not less than type A bookshelves.
3. Maximum budget of 4320 yuan.

We need to prove that the following plans are valid:
1. 8 type A bookshelves, 12 type B bookshelves.
2. 9 type A bookshelves, 11 type B bookshelves.
3. 10 type A bookshelves, 10 type B bookshelves.
-/

theorem bookshelf_prices (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 1020) 
  (h2 : 4 * x + 3 * y = 1440) : 
  x = 180 ∧ y = 240 :=
by sorry

theorem purchasing_plans (m : ℕ) 
  (h3 : 8 ≤ m ∧ m ≤ 10) 
  (h4 : 180 * m + 240 * (20 - m) ≤ 4320) 
  (h5 : 20 - m ≥ m) : 
  m = 8 ∨ m = 9 ∨ m = 10 :=
by sorry

end bookshelf_prices_purchasing_plans_l1227_122705


namespace perpendicular_line_through_point_l1227_122789

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l1227_122789


namespace smallest_rectangles_to_cover_square_l1227_122727

theorem smallest_rectangles_to_cover_square :
  ∃ n : ℕ, 
    (∃ a : ℕ, a = 3 * 4) ∧
    (∃ k : ℕ, k = lcm 3 4) ∧
    (∃ s : ℕ, s = k * k) ∧
    (s / a = n) ∧
    n = 12 :=
by
  sorry

end smallest_rectangles_to_cover_square_l1227_122727


namespace john_max_correct_answers_l1227_122740

theorem john_max_correct_answers 
  (c w b : ℕ) -- define c, w, b as natural numbers
  (h1 : c + w + b = 30) -- condition 1: total questions
  (h2 : 4 * c - 3 * w = 36) -- condition 2: scoring equation
  : c ≤ 12 := -- statement to prove
sorry

end john_max_correct_answers_l1227_122740


namespace simplify_expression_l1227_122706

theorem simplify_expression (x y z : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 :=
by
  sorry

end simplify_expression_l1227_122706


namespace tip_percentage_is_30_l1227_122751

theorem tip_percentage_is_30
  (appetizer_cost : ℝ)
  (entree_cost : ℝ)
  (num_entrees : ℕ)
  (dessert_cost : ℝ)
  (total_price_including_tip : ℝ)
  (h_appetizer : appetizer_cost = 9.0)
  (h_entree : entree_cost = 20.0)
  (h_num_entrees : num_entrees = 2)
  (h_dessert : dessert_cost = 11.0)
  (h_total : total_price_including_tip = 78.0) :
  let total_before_tip := appetizer_cost + num_entrees * entree_cost + dessert_cost
  let tip_amount := total_price_including_tip - total_before_tip
  let tip_percentage := (tip_amount / total_before_tip) * 100
  tip_percentage = 30 :=
by
  sorry

end tip_percentage_is_30_l1227_122751


namespace brian_needs_some_cartons_l1227_122791

def servings_per_person : ℕ := sorry -- This should be defined with the actual number of servings per person.
def family_members : ℕ := 8
def us_cup_in_ml : ℕ := 250
def ml_per_serving : ℕ := us_cup_in_ml / 2
def ml_per_liter : ℕ := 1000

def total_milk_needed (servings_per_person : ℕ) : ℕ :=
  family_members * servings_per_person * ml_per_serving

def cartons_of_milk_needed (servings_per_person : ℕ) : ℕ :=
  total_milk_needed servings_per_person / ml_per_liter + if total_milk_needed servings_per_person % ml_per_liter = 0 then 0 else 1

theorem brian_needs_some_cartons (servings_per_person : ℕ) : 
  cartons_of_milk_needed servings_per_person = (family_members * servings_per_person * ml_per_serving / ml_per_liter + 
  if (family_members * servings_per_person * ml_per_serving) % ml_per_liter = 0 then 0 else 1) := 
by 
  sorry

end brian_needs_some_cartons_l1227_122791


namespace least_possible_value_of_y_l1227_122797

theorem least_possible_value_of_y
  (x y z : ℤ)
  (hx : Even x)
  (hy : Odd y)
  (hz : Odd z)
  (h1 : y - x > 5)
  (h2 : ∀ z', z' - x ≥ 9 → z' ≥ 9) :
  y ≥ 7 :=
by
  -- Proof is not required here
  sorry

end least_possible_value_of_y_l1227_122797


namespace beth_marbles_left_l1227_122783

theorem beth_marbles_left :
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  T - (L_red + L_blue + L_yellow) = 42 :=
by
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  have h1 : T - (L_red + L_blue + L_yellow) = 42 := rfl
  exact h1

end beth_marbles_left_l1227_122783


namespace net_change_in_onions_l1227_122736

-- Definitions for the given conditions
def onions_added_by_sara : ℝ := 4.5
def onions_taken_by_sally : ℝ := 5.25
def onions_added_by_fred : ℝ := 9.75

-- Statement of the problem to be proved
theorem net_change_in_onions : 
  onions_added_by_sara - onions_taken_by_sally + onions_added_by_fred = 9 := 
by
  sorry -- hint that proof is required

end net_change_in_onions_l1227_122736


namespace janet_clarinet_hours_l1227_122720

theorem janet_clarinet_hours 
  (C : ℕ)  -- number of clarinet lessons hours per week
  (clarinet_cost_per_hour : ℕ := 40)
  (piano_cost_per_hour : ℕ := 28)
  (hours_of_piano_per_week : ℕ := 5)
  (annual_extra_piano_cost : ℕ := 1040) :
  52 * (piano_cost_per_hour * hours_of_piano_per_week - clarinet_cost_per_hour * C) = annual_extra_piano_cost → 
  C = 3 :=
by
  sorry

end janet_clarinet_hours_l1227_122720


namespace product_modulo_seven_l1227_122729

theorem product_modulo_seven (a b c d : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3)
(h3 : c % 7 = 4) (h4 : d % 7 = 5) : (a * b * c * d) % 7 = 1 := 
sorry

end product_modulo_seven_l1227_122729


namespace estimate_total_number_of_fish_l1227_122770

-- Define the conditions
variables (totalMarked : ℕ) (secondSample : ℕ) (markedInSecondSample : ℕ) (N : ℕ)

-- Assume the conditions
axiom condition1 : totalMarked = 60
axiom condition2 : secondSample = 80
axiom condition3 : markedInSecondSample = 5

-- Lean theorem statement proving N = 960 given the conditions
theorem estimate_total_number_of_fish (totalMarked secondSample markedInSecondSample N : ℕ)
  (h1 : totalMarked = 60)
  (h2 : secondSample = 80)
  (h3 : markedInSecondSample = 5) :
  N = 960 :=
sorry

end estimate_total_number_of_fish_l1227_122770


namespace probability_divisible_by_25_is_zero_l1227_122785

-- Definitions of spinner outcomes and the function to generate four-digit numbers
def is_valid_spinner_outcome (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3

def generate_four_digit_number (spin1 spin2 spin3 spin4 : ℕ) : ℕ :=
  spin1 * 1000 + spin2 * 100 + spin3 * 10 + spin4

-- Condition stating that all outcomes of each spin are equally probable among {1, 2, 3}
def valid_outcome_condition (spin1 spin2 spin3 spin4 : ℕ) : Prop :=
  is_valid_spinner_outcome spin1 ∧ is_valid_spinner_outcome spin2 ∧
  is_valid_spinner_outcome spin3 ∧ is_valid_spinner_outcome spin4

-- Probability condition for the number being divisible by 25
def is_divisible_by_25 (n : ℕ) : Prop := n % 25 = 0

-- Main theorem: proving the probability is 0
theorem probability_divisible_by_25_is_zero :
  ∀ spin1 spin2 spin3 spin4,
    valid_outcome_condition spin1 spin2 spin3 spin4 →
    ¬ is_divisible_by_25 (generate_four_digit_number spin1 spin2 spin3 spin4) :=
by
  intros spin1 spin2 spin3 spin4 h
  -- Sorry for the proof details
  sorry

end probability_divisible_by_25_is_zero_l1227_122785


namespace rectangle_ratio_l1227_122722

/-- Conditions:
1. There are three identical squares and two rectangles forming a large square.
2. Each rectangle shares one side with a square and another side with the edge of the large square.
3. The side length of each square is 1 unit.
4. The total side length of the large square is 5 units.
Question:
What is the ratio of the length to the width of one of the rectangles? --/

theorem rectangle_ratio (sq_len : ℝ) (large_sq_len : ℝ) (side_ratio : ℝ) :
  sq_len = 1 ∧ large_sq_len = 5 ∧ 
  (∀ (rect_len rect_wid : ℝ), 3 * sq_len + 2 * rect_len = large_sq_len ∧ side_ratio = rect_len / rect_wid) →
  side_ratio = 1 / 2 :=
by
  sorry

end rectangle_ratio_l1227_122722


namespace lesser_fraction_l1227_122742

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 14 / 15) (h2 : x * y = 1 / 10) : min x y = 1 / 5 :=
sorry

end lesser_fraction_l1227_122742


namespace find_p_l1227_122794

theorem find_p (m n p : ℚ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + 18) / 6 - 2 / 5) : 
  p = 3 := 
by 
  sorry

end find_p_l1227_122794


namespace calculate_f_value_l1227_122712

def f (x y : ℚ) : ℚ := x - y * ⌈x / y⌉

theorem calculate_f_value :
  f (1/3) (-3/7) = -2/21 := by
  sorry

end calculate_f_value_l1227_122712


namespace test_unanswered_one_way_l1227_122771

theorem test_unanswered_one_way (Q A : ℕ) (hQ : Q = 4) (hA : A = 5):
  ∀ (unanswered : ℕ), (unanswered = 1) :=
by
  intros
  sorry

end test_unanswered_one_way_l1227_122771


namespace sum_first_twelve_terms_of_arithmetic_sequence_l1227_122777

theorem sum_first_twelve_terms_of_arithmetic_sequence :
    let a1 := -3
    let a12 := 48
    let n := 12
    let Sn := (n * (a1 + a12)) / 2
    Sn = 270 := 
by
  sorry

end sum_first_twelve_terms_of_arithmetic_sequence_l1227_122777


namespace mixtape_first_side_songs_l1227_122769

theorem mixtape_first_side_songs (total_length : ℕ) (second_side_songs : ℕ) (song_length : ℕ) :
  total_length = 40 → second_side_songs = 4 → song_length = 4 → (total_length - second_side_songs * song_length) / song_length = 6 := 
by
  intros h1 h2 h3
  sorry

end mixtape_first_side_songs_l1227_122769


namespace solve_for_A_l1227_122724

def clubsuit (A B : ℤ) : ℤ := 3 * A + 2 * B + 7

theorem solve_for_A (A : ℤ) : (clubsuit A 6 = 70) -> (A = 17) :=
by
  sorry

end solve_for_A_l1227_122724


namespace max_subway_riders_l1227_122741

theorem max_subway_riders:
  ∃ (P F : ℕ), P + F = 251 ∧ (1 / 11) * P + (1 / 13) * F = 22 := sorry

end max_subway_riders_l1227_122741


namespace rectangle_area_percentage_increase_l1227_122702

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let len_inc := 1.3 * l
  let wid_inc := 1.15 * w
  let A_new := len_inc * wid_inc
  let percentage_increase := ((A_new - A) / A) * 100
  percentage_increase = 49.5 :=
by
  sorry

end rectangle_area_percentage_increase_l1227_122702


namespace question_true_l1227_122787
noncomputable def a := (1/2) * Real.cos (7 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * Real.pi / 180)
noncomputable def b := (2 * Real.tan (12 * Real.pi / 180)) / (1 + Real.tan (12 * Real.pi / 180)^2)
noncomputable def c := Real.sqrt ((1 - Real.cos (44 * Real.pi / 180)) / 2)

theorem question_true :
  b > a ∧ a > c :=
by
  sorry

end question_true_l1227_122787


namespace event_B_C_mutually_exclusive_l1227_122728

-- Define the events based on the given conditions
def EventA (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬is_defective x ∧ ¬is_defective y

def EventB (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  is_defective x ∧ is_defective y

def EventC (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬(is_defective x ∧ is_defective y)

-- Prove that Event B and Event C are mutually exclusive
theorem event_B_C_mutually_exclusive (products : Type) (is_defective : products → Prop) (x y : products) :
  (EventB products is_defective x y) → ¬(EventC products is_defective x y) :=
sorry

end event_B_C_mutually_exclusive_l1227_122728


namespace fraction_members_absent_l1227_122707

variable (p : ℕ) -- Number of persons in the office
variable (W : ℝ) -- Total work amount
variable (x : ℝ) -- Fraction of members absent

theorem fraction_members_absent (h : W / (p * (1 - x)) = W / p + W / (6 * p)) : x = 1 / 7 :=
by
  sorry

end fraction_members_absent_l1227_122707


namespace complete_the_square_correct_l1227_122798

noncomputable def complete_the_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 1 = 0 ↔ (x - 1)^2 = 2

theorem complete_the_square_correct : ∀ x : ℝ, complete_the_square x := by
  sorry

end complete_the_square_correct_l1227_122798


namespace solving_inequality_l1227_122775

theorem solving_inequality (x : ℝ) : 
  (x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1)) ↔ ((x^2 - 4) / (x^2 - 1) > 0) :=
by 
  sorry

end solving_inequality_l1227_122775


namespace complete_task_in_3_days_l1227_122748

theorem complete_task_in_3_days (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0)
  (h1 : 1 / x + 1 / y + 1 / z = 1 / 7.5)
  (h2 : 1 / x + 1 / z + 1 / v = 1 / 5)
  (h3 : 1 / x + 1 / z + 1 / w = 1 / 6)
  (h4 : 1 / y + 1 / w + 1 / v = 1 / 4) :
  1 / (1 / x + 1 / z + 1 / v + 1 / w + 1 / y) = 3 :=
sorry

end complete_task_in_3_days_l1227_122748


namespace cost_per_mile_l1227_122715

def miles_per_week : ℕ := 3 * 50 + 4 * 100
def weeks_per_year : ℕ := 52
def miles_per_year : ℕ := miles_per_week * weeks_per_year
def weekly_fee : ℕ := 100
def yearly_total_fee : ℕ := 7800
def yearly_weekly_fees : ℕ := 52 * weekly_fee
def yearly_mile_fees := yearly_total_fee - yearly_weekly_fees
def pay_per_mile := yearly_mile_fees / miles_per_year

theorem cost_per_mile : pay_per_mile = 909 / 10000 := by
  -- proof will be added here
  sorry

end cost_per_mile_l1227_122715


namespace value_of_a_plus_b_l1227_122768

theorem value_of_a_plus_b (a b x y : ℝ) 
  (h1 : 2 * x + 4 * y = 20)
  (h2 : a * x + b * y = 1)
  (h3 : 2 * x - y = 5)
  (h4 : b * x + a * y = 6) : a + b = 1 := 
sorry

end value_of_a_plus_b_l1227_122768


namespace option_A_is_correct_l1227_122782

theorem option_A_is_correct (a b : ℝ) (h : a ≠ 0) : (a^2 / (a * b)) = (a / b) :=
by
  -- Proof will be filled in here
  sorry

end option_A_is_correct_l1227_122782


namespace negation_of_existence_l1227_122721

theorem negation_of_existence :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_existence_l1227_122721


namespace speedster_convertibles_proof_l1227_122778

-- Definitions based on conditions
def total_inventory (T : ℕ) : Prop := 2 / 3 * T = 2 / 3 * T
def not_speedsters (T : ℕ) : Prop := 1 / 3 * T = 60
def speedsters (T : ℕ) (S : ℕ) : Prop := S = 2 / 3 * T
def speedster_convertibles (S : ℕ) (C : ℕ) : Prop := C = 4 / 5 * S

theorem speedster_convertibles_proof (T S C : ℕ) (hT : total_inventory T) (hNS : not_speedsters T) (hS : speedsters T S) (hSC : speedster_convertibles S C) : C = 96 :=
by
  -- Proof goes here
  sorry

end speedster_convertibles_proof_l1227_122778


namespace problem_solution_eq_l1227_122732

theorem problem_solution_eq : 
  { x : ℝ | (x ^ 2 - 9) / (x ^ 2 - 1) > 0 } = { x : ℝ | x > 3 ∨ x < -3 } :=
by
  sorry

end problem_solution_eq_l1227_122732


namespace crayons_lost_or_given_away_l1227_122737

theorem crayons_lost_or_given_away (given_away lost : ℕ) (H_given_away : given_away = 213) (H_lost : lost = 16) :
  given_away + lost = 229 :=
by
  sorry

end crayons_lost_or_given_away_l1227_122737
