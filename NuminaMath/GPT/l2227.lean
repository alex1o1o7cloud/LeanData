import Mathlib

namespace NUMINAMATH_GPT_find_term_number_l2227_222778

noncomputable def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem find_term_number
  (a₁ : ℤ)
  (d : ℤ)
  (n : ℕ)
  (h₀ : a₁ = 1)
  (h₁ : d = 3)
  (h₂ : arithmetic_sequence a₁ d n = 2011) :
  n = 671 :=
  sorry

end NUMINAMATH_GPT_find_term_number_l2227_222778


namespace NUMINAMATH_GPT_number_of_unique_intersections_l2227_222710

-- Definitions for the given lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 3
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 5 * x - 3 * y = 6

-- The problem is to show the number of unique intersection points is 2
theorem number_of_unique_intersections : ∃ p1 p2 : ℝ × ℝ,
  p1 ≠ p2 ∧
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line1 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (p1 ≠ p2 → ∀ p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2 ∨ line3 p.1 p.2) →
    (p = p1 ∨ p = p2)) :=
sorry

end NUMINAMATH_GPT_number_of_unique_intersections_l2227_222710


namespace NUMINAMATH_GPT_angle_BDC_is_55_l2227_222754

def right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] : Prop :=
  ∃ (angle_A angle_B angle_C : ℝ), angle_A + angle_B + angle_C = 180 ∧
  angle_A = 20 ∧ angle_C = 90

def bisector (B D : Type) [Inhabited B] [Inhabited D] (angle_ABC : ℝ) : Prop :=
  ∃ (angle_DBC : ℝ), angle_DBC = angle_ABC / 2

theorem angle_BDC_is_55 (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] :
  right_triangle A B C →
  bisector B D 70 →
  ∃ angle_BDC : ℝ, angle_BDC = 55 :=
by sorry

end NUMINAMATH_GPT_angle_BDC_is_55_l2227_222754


namespace NUMINAMATH_GPT_solve_equation_l2227_222728

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_solve_equation_l2227_222728


namespace NUMINAMATH_GPT_no_single_x_for_doughnut_and_syrup_l2227_222797

theorem no_single_x_for_doughnut_and_syrup :
  ¬ ∃ x : ℝ, (x^2 - 9 * x + 13 < 0) ∧ (x^2 + x - 5 < 0) :=
sorry

end NUMINAMATH_GPT_no_single_x_for_doughnut_and_syrup_l2227_222797


namespace NUMINAMATH_GPT_g_sum_even_l2227_222791

def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5

theorem g_sum_even (a b c d : ℝ) (h : g 42 a b c d = 3) : g 42 a b c d + g (-42) a b c d = 6 := by
  sorry

end NUMINAMATH_GPT_g_sum_even_l2227_222791


namespace NUMINAMATH_GPT_table_capacity_l2227_222722

def invited_people : Nat := 18
def no_show_people : Nat := 12
def number_of_tables : Nat := 2
def attendees := invited_people - no_show_people
def people_per_table : Nat := attendees / number_of_tables

theorem table_capacity : people_per_table = 3 :=
by
  sorry

end NUMINAMATH_GPT_table_capacity_l2227_222722


namespace NUMINAMATH_GPT_parallel_line_plane_l2227_222751

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry

-- Predicate for parallel lines
noncomputable def is_parallel_line (a b : line) : Prop := sorry

-- Predicate for parallel line and plane
noncomputable def is_parallel_plane (a : line) (α : plane) : Prop := sorry

-- Predicate for line contained within the plane
noncomputable def contained_in_plane (b : line) (α : plane) : Prop := sorry

theorem parallel_line_plane
  (a b : line) (α : plane)
  (h1 : is_parallel_line a b)
  (h2 : ¬ contained_in_plane a α)
  (h3 : contained_in_plane b α) :
  is_parallel_plane a α :=
sorry

end NUMINAMATH_GPT_parallel_line_plane_l2227_222751


namespace NUMINAMATH_GPT_find_term_in_sequence_l2227_222763

theorem find_term_in_sequence (n : ℕ) (k : ℕ) (term_2020: ℚ) : 
  (3^7 = 2187) → 
  (2020 : ℕ) / (2187 : ℕ) = term_2020 → 
  (term_2020 = 2020 / 2187) →
  (∃ (k : ℕ), k = 2020 ∧ (2 ≤ k ∧ k < 2187 ∧ k % 3 ≠ 0)) → 
  (2020 / 2187 = (1347 / 2187 : ℚ)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_term_in_sequence_l2227_222763


namespace NUMINAMATH_GPT_smallest_solution_l2227_222779

theorem smallest_solution (x : ℝ) : 
  (∃ x, (3 * x / (x - 3)) + ((3 * x^2 - 27) / x) = 15 ∧ ∀ y, (3 * y / (y - 3)) + ((3 * y^2 - 27) / y) = 15 → y ≥ x) → 
  x = -1 := 
by
  sorry

end NUMINAMATH_GPT_smallest_solution_l2227_222779


namespace NUMINAMATH_GPT_ratio_of_areas_of_similar_triangles_l2227_222730

theorem ratio_of_areas_of_similar_triangles (a b a1 b1 S S1 : ℝ) (α k : ℝ) :
  S = (1/2) * a * b * (Real.sin α) →
  S1 = (1/2) * a1 * b1 * (Real.sin α) →
  a1 = k * a →
  b1 = k * b →
  S1 / S = k^2 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_similar_triangles_l2227_222730


namespace NUMINAMATH_GPT_iodine_initial_amount_l2227_222767

theorem iodine_initial_amount (half_life : ℕ) (days_elapsed : ℕ) (final_amount : ℕ) (initial_amount : ℕ) :
  half_life = 8 → days_elapsed = 24 → final_amount = 2 → initial_amount = final_amount * 2 ^ (days_elapsed / half_life) → initial_amount = 16 :=
by
  intros h_half_life h_days_elapsed h_final_amount h_initial_exp
  rw [h_half_life, h_days_elapsed, h_final_amount] at h_initial_exp
  norm_num at h_initial_exp
  exact h_initial_exp

end NUMINAMATH_GPT_iodine_initial_amount_l2227_222767


namespace NUMINAMATH_GPT_xiao_ming_selects_cooking_probability_l2227_222755

theorem xiao_ming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let probability (event: String) := if event ∈ courses then 1 / (courses.length : ℝ) else 0
  probability "cooking" = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_selects_cooking_probability_l2227_222755


namespace NUMINAMATH_GPT_correct_average_of_ten_numbers_l2227_222784

theorem correct_average_of_ten_numbers :
  let incorrect_average := 20 
  let num_values := 10 
  let incorrect_number := 26
  let correct_number := 86 
  let incorrect_total_sum := incorrect_average * num_values
  let correct_total_sum := incorrect_total_sum - incorrect_number + correct_number 
  (correct_total_sum / num_values) = 26 := 
by
  sorry

end NUMINAMATH_GPT_correct_average_of_ten_numbers_l2227_222784


namespace NUMINAMATH_GPT_distinct_solutions_square_l2227_222783

theorem distinct_solutions_square (α β : ℝ) (h₁ : α ≠ β)
    (h₂ : α^2 = 2 * α + 2 ∧ β^2 = 2 * β + 2) : (α - β) ^ 2 = 12 := by
  sorry

end NUMINAMATH_GPT_distinct_solutions_square_l2227_222783


namespace NUMINAMATH_GPT_birch_trees_probability_l2227_222738

/--
A gardener plants four pine trees, five oak trees, and six birch trees in a row. He plants them in random order, each arrangement being equally likely.
Prove that no two birch trees are next to one another is \(\frac{2}{45}\).
--/
theorem birch_trees_probability: (∃ (m n : ℕ), (m = 2) ∧ (n = 45) ∧ (no_two_birch_trees_adjacent_probability = m / n)) := 
sorry

end NUMINAMATH_GPT_birch_trees_probability_l2227_222738


namespace NUMINAMATH_GPT_jeans_price_increase_l2227_222732

theorem jeans_price_increase
  (C R P : ℝ)
  (h1 : P = 1.15 * R)
  (h2 : P = 1.6100000000000001 * C) :
  R = 1.4 * C :=
by
  sorry

end NUMINAMATH_GPT_jeans_price_increase_l2227_222732


namespace NUMINAMATH_GPT_passing_marks_l2227_222741

theorem passing_marks
  (T P : ℝ)
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) :
  P = 160 :=
by
  sorry

end NUMINAMATH_GPT_passing_marks_l2227_222741


namespace NUMINAMATH_GPT_length_of_AB_l2227_222704

theorem length_of_AB {A B P Q : ℝ} (h1 : P = 3 / 5 * B)
                    (h2 : Q = 2 / 5 * A + 3 / 5 * B)
                    (h3 : dist P Q = 5) :
  dist A B = 25 :=
by sorry

end NUMINAMATH_GPT_length_of_AB_l2227_222704


namespace NUMINAMATH_GPT_shooting_prob_l2227_222705

theorem shooting_prob (p q : ℚ) (h: p + q = 1) (n : ℕ) 
  (cond1: p = 2/3) 
  (cond2: q = 1 - p) 
  (cond3: n = 5) : 
  (q ^ (n-1)) = 1/81 := 
by 
  sorry

end NUMINAMATH_GPT_shooting_prob_l2227_222705


namespace NUMINAMATH_GPT_multiplier_of_difference_l2227_222702

variable (x y : ℕ)
variable (h : x + y = 49) (h1 : x > y)

theorem multiplier_of_difference (h2 : x^2 - y^2 = k * (x - y)) : k = 49 :=
by sorry

end NUMINAMATH_GPT_multiplier_of_difference_l2227_222702


namespace NUMINAMATH_GPT_smaller_angle_of_parallelogram_l2227_222715

theorem smaller_angle_of_parallelogram (x : ℝ) (h : x + 3 * x = 180) : x = 45 :=
sorry

end NUMINAMATH_GPT_smaller_angle_of_parallelogram_l2227_222715


namespace NUMINAMATH_GPT_dealer_purchased_articles_l2227_222795

/-
The dealer purchases some articles for Rs. 25 and sells 12 articles for Rs. 38. 
The dealer has a profit percentage of 90%. Prove that the number of articles 
purchased by the dealer is 14.
-/

theorem dealer_purchased_articles (x : ℕ) 
    (total_cost : ℝ) (group_selling_price : ℝ) (group_size : ℕ) (profit_percentage : ℝ) 
    (h1 : total_cost = 25)
    (h2 : group_selling_price = 38)
    (h3 : group_size = 12)
    (h4 : profit_percentage = 90 / 100) :
    x = 14 :=
by
  sorry

end NUMINAMATH_GPT_dealer_purchased_articles_l2227_222795


namespace NUMINAMATH_GPT_statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l2227_222768

theorem statement_A_correct :
  (∃ x0 : ℝ, x0^2 + 2 * x0 + 2 < 0) ↔ (¬ ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0) :=
sorry

theorem statement_B_incorrect :
  ¬ (∀ x y : ℝ, x > y → |x| > |y|) :=
sorry

theorem statement_C_incorrect :
  ¬ ∀ x : ℤ, x^2 > 0 :=
sorry

theorem statement_D_correct :
  (∀ m : ℝ, (∃ x1 x2 : ℝ, x1 + x2 = 2 ∧ x1 * x2 = m ∧ x1 * x2 > 0) ↔ m < 0) :=
sorry

end NUMINAMATH_GPT_statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l2227_222768


namespace NUMINAMATH_GPT_intersection_M_N_l2227_222727

noncomputable def set_M : Set ℝ := {x | x^2 - 3 * x - 4 ≤ 0}
noncomputable def set_N : Set ℝ := {x | Real.log x ≥ 0}

theorem intersection_M_N :
  {x | x ∈ set_M ∧ x ∈ set_N} = {x | 1 ≤ x ∧ x ≤ 4} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l2227_222727


namespace NUMINAMATH_GPT_incorrect_statement_d_l2227_222765

noncomputable def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem incorrect_statement_d (n : ℤ) :
  (n < cbrt 9 ∧ cbrt 9 < n+1) → n ≠ 3 :=
by
  intro h
  have h2 : (2 : ℤ) < cbrt 9 := sorry
  have h3 : cbrt 9 < (3 : ℤ) := sorry
  exact sorry

end NUMINAMATH_GPT_incorrect_statement_d_l2227_222765


namespace NUMINAMATH_GPT_combined_weight_of_boxes_l2227_222735

def first_box_weight := 2
def second_box_weight := 11
def last_box_weight := 5

theorem combined_weight_of_boxes :
  first_box_weight + second_box_weight + last_box_weight = 18 := by
  sorry

end NUMINAMATH_GPT_combined_weight_of_boxes_l2227_222735


namespace NUMINAMATH_GPT_solve_quadratic_difference_l2227_222717

theorem solve_quadratic_difference :
  ∀ x : ℝ, (x^2 - 7*x - 48 = 0) → 
  let x1 := (7 + Real.sqrt 241) / 2
  let x2 := (7 - Real.sqrt 241) / 2
  abs (x1 - x2) = Real.sqrt 241 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_difference_l2227_222717


namespace NUMINAMATH_GPT_sin_A_mul_sin_B_find_c_l2227_222760

-- Definitions for the triangle and the given conditions
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Opposite sides of the triangle

-- Given conditions
axiom h1 : c^2 = 4 * a * b * (Real.sin C)^2

-- The first proof problem statement
theorem sin_A_mul_sin_B (ha : A + B + C = π) (h2 : Real.sin C ≠ 0) :
  Real.sin A * Real.sin B = 1/4 :=
by
  sorry

-- The second proof problem statement with additional given conditions
theorem find_c (ha : A = π / 6) (ha2 : a = 3) (hb2 : b = 3) : 
  c = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_A_mul_sin_B_find_c_l2227_222760


namespace NUMINAMATH_GPT_perimeter_of_quadrilateral_l2227_222757

theorem perimeter_of_quadrilateral 
  (WXYZ_area : ℝ)
  (h_area : WXYZ_area = 2500)
  (WQ XQ YQ ZQ : ℝ)
  (h_WQ : WQ = 30)
  (h_XQ : XQ = 40)
  (h_YQ : YQ = 35)
  (h_ZQ : ZQ = 50) :
  ∃ (P : ℝ), P = 155 + 10 * Real.sqrt 34 + 5 * Real.sqrt 113 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_quadrilateral_l2227_222757


namespace NUMINAMATH_GPT_height_of_windows_l2227_222716

theorem height_of_windows
  (L W H d_l d_w w_w : ℕ)
  (C T : ℕ)
  (hl : L = 25)
  (hw : W = 15)
  (hh : H = 12)
  (hdl : d_l = 6)
  (hdw : d_w = 3)
  (hww : w_w = 3)
  (hc : C = 3)
  (ht : T = 2718):
  ∃ h : ℕ, 960 - (18 + 9 * h) = 906 ∧ 
  (T = C * (960 - (18 + 9 * h))) ∧
  (960 = 2 * (L * H) + 2 * (W * H)) ∧ 
  (18 = d_l * d_w) ∧ 
  (9 * h = 3 * (h * w_w)) := 
sorry

end NUMINAMATH_GPT_height_of_windows_l2227_222716


namespace NUMINAMATH_GPT_angle_B_value_l2227_222790

theorem angle_B_value (a b c B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
    B = (Real.pi / 3) ∨ B = (2 * Real.pi / 3) :=
by
    sorry

end NUMINAMATH_GPT_angle_B_value_l2227_222790


namespace NUMINAMATH_GPT_problem_statement_l2227_222714

noncomputable def a_b (a b : ℚ) : Prop :=
  a + b = 6 ∧ a / b = 6

theorem problem_statement (a b : ℚ) (h : a_b a b) : 
  (a * b - (a - b)) = 6 / 49 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2227_222714


namespace NUMINAMATH_GPT_max_value_of_expression_l2227_222742

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2227_222742


namespace NUMINAMATH_GPT_solve_for_x_l2227_222711

theorem solve_for_x (x : ℤ) (h : 15 * 2 = x - 3 + 5) : x = 28 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2227_222711


namespace NUMINAMATH_GPT_range_of_a_if_p_is_false_l2227_222737

theorem range_of_a_if_p_is_false :
  (∀ x : ℝ, x^2 + a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) := 
sorry

end NUMINAMATH_GPT_range_of_a_if_p_is_false_l2227_222737


namespace NUMINAMATH_GPT_mean_proportional_l2227_222708

theorem mean_proportional (x : ℝ) (h : (72.5:ℝ) = Real.sqrt (x * 81)): x = 64.9 := by
  sorry

end NUMINAMATH_GPT_mean_proportional_l2227_222708


namespace NUMINAMATH_GPT_logically_follows_l2227_222744

-- Define the predicates P and Q
variables {Student : Type} {P Q : Student → Prop}

-- The given condition
axiom Turner_statement : ∀ (x : Student), P x → Q x

-- The statement that necessarily follows
theorem logically_follows : (∀ (x : Student), ¬ Q x → ¬ P x) :=
sorry

end NUMINAMATH_GPT_logically_follows_l2227_222744


namespace NUMINAMATH_GPT_decimal_15_to_binary_l2227_222799

theorem decimal_15_to_binary : (15 : ℕ) = (4*1 + 2*1 + 1*1)*2^3 + (4*1 + 2*1 + 1*1)*2^2 + (4*1 + 2*1 + 1*1)*2 + 1 := by
  sorry

end NUMINAMATH_GPT_decimal_15_to_binary_l2227_222799


namespace NUMINAMATH_GPT_term_2_6_position_l2227_222739

theorem term_2_6_position : 
  ∃ (seq : ℕ → ℚ), 
    (seq 23 = 2 / 6) ∧ 
    (∀ n, ∃ k, (n = (k * (k + 1)) / 2 ∧ k > 0 ∧ k <= n)) :=
by sorry

end NUMINAMATH_GPT_term_2_6_position_l2227_222739


namespace NUMINAMATH_GPT_find_annual_interest_rate_l2227_222770

noncomputable def compound_interest_problem : Prop :=
  ∃ (r : ℝ),
    let P := 8000
    let CI := 3109
    let t := 2.3333
    let A := 11109
    let n := 1
    A = P * (1 + r/n)^(n*t) ∧ r = 0.1505

theorem find_annual_interest_rate : compound_interest_problem :=
by sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l2227_222770


namespace NUMINAMATH_GPT_middle_school_mentoring_l2227_222729

theorem middle_school_mentoring (s n : ℕ) (h1 : s ≠ 0) (h2 : n ≠ 0) 
  (h3 : (n : ℚ) / 3 = (2 : ℚ) * (s : ℚ) / 5) : 
  (n / 3 + 2 * s / 5) / (n + s) = 4 / 11 := by
  sorry

end NUMINAMATH_GPT_middle_school_mentoring_l2227_222729


namespace NUMINAMATH_GPT_cos_identity_l2227_222758

theorem cos_identity (θ : ℝ) (h : Real.cos (π / 6 + θ) = (Real.sqrt 3) / 3) : 
  Real.cos (5 * π / 6 - θ) = - (Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_cos_identity_l2227_222758


namespace NUMINAMATH_GPT_positive_difference_of_b_values_l2227_222725

noncomputable def g (n : ℤ) : ℤ :=
if n ≤ 0 then n^2 + 3 * n + 2 else 3 * n - 15

theorem positive_difference_of_b_values : 
  abs (-5 - 9) = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_difference_of_b_values_l2227_222725


namespace NUMINAMATH_GPT_correlation_coefficients_l2227_222749

-- Definition of the variables and constants
def relative_risks_starting_age : List (ℕ × ℝ) := [(16, 15.10), (18, 12.81), (20, 9.72), (22, 3.21)]
def relative_risks_cigarettes_per_day : List (ℕ × ℝ) := [(10, 7.5), (20, 9.5), (30, 16.6)]

def r1 : ℝ := -- The correlation coefficient between starting age and relative risk
  sorry

def r2 : ℝ := -- The correlation coefficient between number of cigarettes per day and relative risk
  sorry

theorem correlation_coefficients :
  r1 < 0 ∧ 0 < r2 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end NUMINAMATH_GPT_correlation_coefficients_l2227_222749


namespace NUMINAMATH_GPT_wheat_flour_used_l2227_222701

-- Conditions and definitions
def total_flour_used : ℝ := 0.3
def white_flour_used : ℝ := 0.1

-- Statement of the problem
theorem wheat_flour_used : 
  (total_flour_used - white_flour_used) = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_wheat_flour_used_l2227_222701


namespace NUMINAMATH_GPT_percentage_of_cobalt_is_15_l2227_222700

-- Define the given percentages of lead and copper
def percent_lead : ℝ := 25
def percent_copper : ℝ := 60

-- Define the weights of lead and copper used in the mixture
def weight_lead : ℝ := 5
def weight_copper : ℝ := 12

-- Define the total weight of the mixture
def total_weight : ℝ := weight_lead + weight_copper

-- Prove that the percentage of cobalt is 15%
theorem percentage_of_cobalt_is_15 :
  (100 - (percent_lead + percent_copper) = 15) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_cobalt_is_15_l2227_222700


namespace NUMINAMATH_GPT_power_function_nature_l2227_222740

def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_nature:
  (f 3 = Real.sqrt 3) ∧
  (¬ (∀ x, f (-x) = f x)) ∧
  (¬ (∀ x, f (-x) = -f x)) ∧
  (∀ x, 0 < x → 0 < f x) := 
by
  sorry

end NUMINAMATH_GPT_power_function_nature_l2227_222740


namespace NUMINAMATH_GPT_solve_trig_eq_l2227_222752

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = (π / 12) + 2 * k * π ∨
   x = (7 * π / 12) + 2 * k * π ∨
   x = (7 * π / 6) + 2 * k * π ∨
   x = -(5 * π / 6) + 2 * k * π) →
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_solve_trig_eq_l2227_222752


namespace NUMINAMATH_GPT_power_of_two_divisor_l2227_222713

theorem power_of_two_divisor {n : ℕ} (h_pos : n > 0) : 
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) → ∃ r : ℕ, n = 2^r :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_divisor_l2227_222713


namespace NUMINAMATH_GPT_man_l2227_222731

theorem man's_speed_downstream (v : ℕ) (h1 : v - 3 = 8) (s : ℕ := 3) : v + s = 14 :=
by
  sorry

end NUMINAMATH_GPT_man_l2227_222731


namespace NUMINAMATH_GPT_blue_pill_cost_correct_l2227_222780

-- Defining the conditions
def num_days : Nat := 21
def total_cost : Nat := 672
def red_pill_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost - 2
def daily_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost + red_pill_cost blue_pill_cost

-- The statement to prove
theorem blue_pill_cost_correct : ∃ (y : Nat), daily_cost y * num_days = total_cost ∧ y = 17 :=
by
  sorry

end NUMINAMATH_GPT_blue_pill_cost_correct_l2227_222780


namespace NUMINAMATH_GPT_maximum_value_of_expression_l2227_222776

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression (x y z : ℝ ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) :
  problem_statement x y z ≤ 81 / 4 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l2227_222776


namespace NUMINAMATH_GPT_seq_15_l2227_222775

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 2 else 2 * (n - 1) + 1 -- form inferred from solution

theorem seq_15 : seq 15 = 29 := by
  sorry

end NUMINAMATH_GPT_seq_15_l2227_222775


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2227_222796

variable (x a : ℝ)

def p := x ≤ -1
def q := a ≤ x ∧ x < a + 2

-- If q is sufficient but not necessary for p, then the range of a is (-∞, -3]
theorem sufficient_not_necessary_condition : 
  (∀ x, q x a → p x) ∧ ∃ x, p x ∧ ¬ q x a → a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2227_222796


namespace NUMINAMATH_GPT_probability_within_circle_eq_pi_over_nine_l2227_222788

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let circle_area := Real.pi * (2 ^ 2)
  let square_area := 6 * 6
  circle_area / square_area

theorem probability_within_circle_eq_pi_over_nine :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end NUMINAMATH_GPT_probability_within_circle_eq_pi_over_nine_l2227_222788


namespace NUMINAMATH_GPT_measure_angle_BCA_l2227_222723

theorem measure_angle_BCA 
  (BCD_angle : ℝ)
  (CBA_angle : ℝ)
  (sum_angles : BCD_angle + CBA_angle + BCA_angle = 190)
  (BCD_right : BCD_angle = 90)
  (CBA_given : CBA_angle = 70) :
  BCA_angle = 30 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_BCA_l2227_222723


namespace NUMINAMATH_GPT_find_s_l2227_222726

theorem find_s (n : ℤ) (hn : n ≠ 0) (s : ℝ)
  (hs : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1 / n)) :
  s = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l2227_222726


namespace NUMINAMATH_GPT_range_of_a_l2227_222707

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2227_222707


namespace NUMINAMATH_GPT_employee_payment_sum_l2227_222712

theorem employee_payment_sum :
  ∀ (A B : ℕ), 
  (A = 3 * B / 2) → 
  (B = 180) → 
  (A + B = 450) :=
by
  intros A B hA hB
  sorry

end NUMINAMATH_GPT_employee_payment_sum_l2227_222712


namespace NUMINAMATH_GPT_trigonometric_identity_l2227_222720

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2227_222720


namespace NUMINAMATH_GPT_q_domain_range_l2227_222782

open Set

-- Given the function h with the specified domain and range
variable (h : ℝ → ℝ) (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3 → h x ∈ Icc 0 2)

def q (x : ℝ) : ℝ := 2 - h (x - 2)

theorem q_domain_range :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → (q h x) ∈ Icc 0 2) ∧
  (∀ y, q h y ∈ Icc 0 2 ↔ y ∈ Icc 1 5) :=
by
  sorry

end NUMINAMATH_GPT_q_domain_range_l2227_222782


namespace NUMINAMATH_GPT_student_needs_33_percent_to_pass_l2227_222772

-- Define the conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def max_marks : ℕ := 500

-- The Lean statement to prove the required percentage
theorem student_needs_33_percent_to_pass : (obtained_marks + failed_by) * 100 / max_marks = 33 := by
  sorry

end NUMINAMATH_GPT_student_needs_33_percent_to_pass_l2227_222772


namespace NUMINAMATH_GPT_baskets_and_remainder_l2227_222793

-- Define the initial conditions
def cucumbers : ℕ := 216
def basket_capacity : ℕ := 23

-- Define the expected calculations
def expected_baskets : ℕ := cucumbers / basket_capacity
def expected_remainder : ℕ := cucumbers % basket_capacity

-- Theorem to prove the output values
theorem baskets_and_remainder :
  expected_baskets = 9 ∧ expected_remainder = 9 := by
  sorry

end NUMINAMATH_GPT_baskets_and_remainder_l2227_222793


namespace NUMINAMATH_GPT_paige_scored_17_points_l2227_222798

def paige_points (total_points : ℕ) (num_players : ℕ) (points_per_player_exclusive : ℕ) : ℕ :=
  total_points - ((num_players - 1) * points_per_player_exclusive)

theorem paige_scored_17_points :
  paige_points 41 5 6 = 17 :=
by
  sorry

end NUMINAMATH_GPT_paige_scored_17_points_l2227_222798


namespace NUMINAMATH_GPT_channels_taken_away_l2227_222734

theorem channels_taken_away (X : ℕ) : 
  (150 - X + 12 - 10 + 8 + 7 = 147) -> X = 20 :=
by
  sorry

end NUMINAMATH_GPT_channels_taken_away_l2227_222734


namespace NUMINAMATH_GPT_solve_inequality_l2227_222709

theorem solve_inequality (x : ℝ) : (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ (-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2227_222709


namespace NUMINAMATH_GPT_greatest_number_of_dimes_l2227_222792

-- Definitions according to the conditions in a)
def total_value_in_cents : ℤ := 485
def dime_value_in_cents : ℤ := 10
def nickel_value_in_cents : ℤ := 5

-- The proof problem in Lean 4
theorem greatest_number_of_dimes : 
  ∃ (d : ℤ), (dime_value_in_cents * d + nickel_value_in_cents * d = total_value_in_cents) ∧ d = 32 := 
by
  sorry

end NUMINAMATH_GPT_greatest_number_of_dimes_l2227_222792


namespace NUMINAMATH_GPT_total_produce_of_mangoes_is_400_l2227_222762

variable (A M O : ℕ)  -- Defines variables for total produce of apples, mangoes, and oranges respectively
variable (P : ℕ := 50)  -- Price per kg
variable (R : ℕ := 90000)  -- Total revenue

-- Definition of conditions
def apples_total_produce := 2 * M
def oranges_total_produce := M + 200
def total_weight_of_fruits := apples_total_produce + M + oranges_total_produce

-- Statement to prove
theorem total_produce_of_mangoes_is_400 :
  (total_weight_of_fruits = R / P) → (M = 400) :=
by
  sorry

end NUMINAMATH_GPT_total_produce_of_mangoes_is_400_l2227_222762


namespace NUMINAMATH_GPT_speed_ratio_l2227_222750

variable (v_A v_B : ℝ)

def equidistant_3min : Prop := 3 * v_A = abs (-800 + 3 * v_B)
def equidistant_8min : Prop := 8 * v_A = abs (-800 + 8 * v_B)
def speed_ratio_correct : Prop := v_A / v_B = 1 / 2

theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_8min v_A v_B) : speed_ratio_correct v_A v_B :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l2227_222750


namespace NUMINAMATH_GPT_marble_weight_l2227_222756

-- Define the conditions
def condition1 (m k : ℝ) : Prop := 9 * m = 5 * k
def condition2 (k : ℝ) : Prop := 4 * k = 120

-- Define the main goal, i.e., proving m = 50/3 given the conditions
theorem marble_weight (m k : ℝ) 
  (h1 : condition1 m k) 
  (h2 : condition2 k) : 
  m = 50 / 3 := by 
  sorry

end NUMINAMATH_GPT_marble_weight_l2227_222756


namespace NUMINAMATH_GPT_jackie_more_apples_oranges_l2227_222759

-- Definitions of initial conditions
def adams_apples : ℕ := 25
def adams_oranges : ℕ := 34
def jackies_apples : ℕ := 43
def jackies_oranges : ℕ := 29

-- The proof statement
theorem jackie_more_apples_oranges :
  (jackies_apples - adams_apples) + (jackies_oranges - adams_oranges) = 13 :=
by
  sorry

end NUMINAMATH_GPT_jackie_more_apples_oranges_l2227_222759


namespace NUMINAMATH_GPT_infinite_series_sum_zero_l2227_222733

theorem infinite_series_sum_zero : ∑' n : ℕ, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_zero_l2227_222733


namespace NUMINAMATH_GPT_intersection_product_l2227_222746

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 9 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 - 8 * x + y^2 - 6 * y + 25 = 0

-- Define the theorem to prove the product of the coordinates of the intersection points
theorem intersection_product : ∀ x y : ℝ, circle1 x y → circle2 x y → x * y = 12 :=
by
  intro x y h1 h2
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_intersection_product_l2227_222746


namespace NUMINAMATH_GPT_perimeter_after_adding_tiles_l2227_222721

-- Initial perimeter given
def initial_perimeter : ℕ := 20

-- Number of initial tiles
def initial_tiles : ℕ := 10

-- Number of additional tiles to be added
def additional_tiles : ℕ := 2

-- New tile side must be adjacent to an existing tile
def adjacent_tile_side : Prop := true

-- Condition about the tiles being 1x1 squares
def sq_tile (n : ℕ) : Prop := n = 1

-- The perimeter should be calculated after adding the tiles
def new_perimeter_after_addition : ℕ := 19

theorem perimeter_after_adding_tiles :
  ∃ (new_perimeter : ℕ), 
    new_perimeter = 19 ∧ 
    initial_perimeter = 20 ∧ 
    initial_tiles = 10 ∧ 
    additional_tiles = 2 ∧ 
    adjacent_tile_side ∧ 
    sq_tile 1 :=
sorry

end NUMINAMATH_GPT_perimeter_after_adding_tiles_l2227_222721


namespace NUMINAMATH_GPT_purely_imaginary_z_l2227_222718

open Complex

theorem purely_imaginary_z (b : ℝ) (h : z = (1 + b * I) / (2 + I) ∧ im z = 0) : z = -I :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_z_l2227_222718


namespace NUMINAMATH_GPT_length_of_garden_l2227_222785

-- Definitions based on conditions
def P : ℕ := 600
def b : ℕ := 200

-- Theorem statement
theorem length_of_garden : ∃ L : ℕ, 2 * (L + b) = P ∧ L = 100 :=
by
  existsi 100
  simp
  sorry

end NUMINAMATH_GPT_length_of_garden_l2227_222785


namespace NUMINAMATH_GPT_triangle_inequality_difference_l2227_222748

theorem triangle_inequality_difference :
  (∀ (x : ℤ), (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) → (3 ≤ x ∧ x ≤ 15) ∧ (15 - 3 = 12)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_difference_l2227_222748


namespace NUMINAMATH_GPT_education_fund_growth_l2227_222771

theorem education_fund_growth (x : ℝ) :
  2500 * (1 + x)^2 = 3600 :=
sorry

end NUMINAMATH_GPT_education_fund_growth_l2227_222771


namespace NUMINAMATH_GPT_problem1_problem2_l2227_222719

-- Define the conditions as noncomputable definitions
noncomputable def A : Real := sorry
noncomputable def tan_A : Real := 2
noncomputable def sin_A_plus_cos_A : Real := 1 / 5

-- Define the trigonometric identities
noncomputable def sin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry
noncomputable def tan (x : Real) : Real := sin x / cos x

-- Ensure the conditions
axiom tan_A_condition : tan A = tan_A
axiom sin_A_plus_cos_A_condition : sin A + cos A = sin_A_plus_cos_A

-- Proof problem 1:
theorem problem1 : 
  (sin (π - A) + cos (-A)) / (sin A - sin (π / 2 + A)) = 3 := by
  sorry

-- Proof problem 2:
theorem problem2 : 
  sin A - cos A = 7 / 5 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2227_222719


namespace NUMINAMATH_GPT_unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l2227_222769

-- Definitions based on conditions
variable (unit_price quantity total_price : ℕ)
variable (map_distance actual_distance scale : ℕ)

-- Given conditions
def total_price_fixed := unit_price * quantity = total_price
def scale_fixed := map_distance * scale = actual_distance

-- Proof problem statements
theorem unit_price_quantity_inverse_proportion (h : total_price_fixed unit_price quantity total_price) :
  ∃ k : ℕ, unit_price = k / quantity := sorry

theorem map_distance_actual_distance_direct_proportion (h : scale_fixed map_distance actual_distance scale) :
  ∃ k : ℕ, map_distance * scale = k * actual_distance := sorry

end NUMINAMATH_GPT_unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l2227_222769


namespace NUMINAMATH_GPT_power_inequality_l2227_222761

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := 
by 
  sorry

end NUMINAMATH_GPT_power_inequality_l2227_222761


namespace NUMINAMATH_GPT_polygon_with_three_times_exterior_angle_sum_is_octagon_l2227_222787

theorem polygon_with_three_times_exterior_angle_sum_is_octagon
  (n : ℕ)
  (h : (n - 2) * 180 = 3 * 360) : n = 8 := by
  sorry

end NUMINAMATH_GPT_polygon_with_three_times_exterior_angle_sum_is_octagon_l2227_222787


namespace NUMINAMATH_GPT_evaluate_expr_l2227_222777

theorem evaluate_expr :
  (150^2 - 12^2) / (90^2 - 21^2) * ((90 + 21) * (90 - 21)) / ((150 + 12) * (150 - 12)) = 2 :=
by sorry

end NUMINAMATH_GPT_evaluate_expr_l2227_222777


namespace NUMINAMATH_GPT_plane_equation_l2227_222766

theorem plane_equation
  (A B C D : ℤ)
  (hA : A > 0)
  (h_gcd : Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1)
  (h_point : (A * 4 + B * (-4) + C * 5 + D = 0)) :
  A = 4 ∧ B = -4 ∧ C = 5 ∧ D = -57 :=
  sorry

end NUMINAMATH_GPT_plane_equation_l2227_222766


namespace NUMINAMATH_GPT_length_cut_XY_l2227_222764

theorem length_cut_XY (a x : ℝ) (h1 : 4 * a = 100) (h2 : a + a + 2 * x = 56) : x = 3 :=
by { sorry }

end NUMINAMATH_GPT_length_cut_XY_l2227_222764


namespace NUMINAMATH_GPT_loss_percentage_l2227_222794

theorem loss_percentage (CP SP : ℝ) (hCP : CP = 1500) (hSP : SP = 1200) : 
  (CP - SP) / CP * 100 = 20 :=
by
  -- Proof would be provided here
  sorry

end NUMINAMATH_GPT_loss_percentage_l2227_222794


namespace NUMINAMATH_GPT_minimum_days_to_owe_double_l2227_222774

/-- Kim borrows $100$ dollars from Sam with a simple interest rate of $10\%$ per day.
    There's a one-time borrowing fee of $10$ dollars that is added to the debt immediately.
    We need to prove that the least integer number of days after which Kim will owe 
    Sam at least twice as much as she borrowed is 9 days.
-/
theorem minimum_days_to_owe_double :
  ∀ (x : ℕ), 100 + 10 + 10 * x ≥ 200 → x ≥ 9 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_minimum_days_to_owe_double_l2227_222774


namespace NUMINAMATH_GPT_lateral_surface_area_of_cone_l2227_222789

-- Definitions from the conditions
def base_radius : ℝ := 6
def slant_height : ℝ := 15

-- Theorem statement to be proved
theorem lateral_surface_area_of_cone (r l : ℝ) (hr : r = base_radius) (hl : l = slant_height) : 
  (π * r * l) = 90 * π :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cone_l2227_222789


namespace NUMINAMATH_GPT_children_on_playground_l2227_222773

theorem children_on_playground (boys_soccer girls_soccer boys_swings girls_swings boys_snacks girls_snacks : ℕ)
(h1 : boys_soccer = 27) (h2 : girls_soccer = 35)
(h3 : boys_swings = 15) (h4 : girls_swings = 20)
(h5 : boys_snacks = 10) (h6 : girls_snacks = 5) :
boys_soccer + girls_soccer + boys_swings + girls_swings + boys_snacks + girls_snacks = 112 := by
  sorry

end NUMINAMATH_GPT_children_on_playground_l2227_222773


namespace NUMINAMATH_GPT_squad_sizes_l2227_222743

-- Definitions for conditions
def total_students (x y : ℕ) : Prop := x + y = 146
def equal_after_transfer (x y : ℕ) : Prop := x - 11 = y + 11

-- Theorem to prove the number of students in first and second-year squads
theorem squad_sizes (x y : ℕ) (h1 : total_students x y) (h2 : equal_after_transfer x y) : 
  x = 84 ∧ y = 62 :=
by
  sorry

end NUMINAMATH_GPT_squad_sizes_l2227_222743


namespace NUMINAMATH_GPT_additional_people_proof_l2227_222724

variable (initialPeople additionalPeople mowingHours trimmingRate totalNewPeople totalMowingPeople requiredPersonHours totalPersonHours: ℕ)

noncomputable def mowingLawn (initialPeople mowingHours : ℕ) : ℕ :=
  initialPeople * mowingHours

noncomputable def mowingRate (requiredPersonHours : ℕ) (mowingHours : ℕ) : ℕ :=
  (requiredPersonHours / mowingHours)

noncomputable def trimmingEdges (totalMowingPeople trimmingRate : ℕ) : ℕ :=
  (totalMowingPeople / trimmingRate)

noncomputable def totalPeople (mowingPeople trimmingPeople : ℕ) : ℕ :=
  (mowingPeople + trimmingPeople)

noncomputable def additionalPeopleNeeded (totalPeople initialPeople : ℕ) : ℕ :=
  (totalPeople - initialPeople)

theorem additional_people_proof :
  initialPeople = 8 →
  mowingHours = 3 →
  totalPersonHours = mowingLawn initialPeople mowingHours →
  totalMowingPeople = mowingRate totalPersonHours 2 →
  trimmingRate = 3 →
  requiredPersonHours = totalPersonHours →
  totalNewPeople = totalPeople totalMowingPeople (trimmingEdges totalMowingPeople trimmingRate) →
  additionalPeople = additionalPeopleNeeded totalNewPeople initialPeople →
  additionalPeople = 8 :=
by
  sorry

end NUMINAMATH_GPT_additional_people_proof_l2227_222724


namespace NUMINAMATH_GPT_unique_function_satisfying_equation_l2227_222786

theorem unique_function_satisfying_equation :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → ∀ x : ℝ, f x = x :=
by
  intro f h
  sorry

end NUMINAMATH_GPT_unique_function_satisfying_equation_l2227_222786


namespace NUMINAMATH_GPT_sarah_bottle_caps_l2227_222747

theorem sarah_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) : initial_caps = 26 → additional_caps = 3 → total_caps = initial_caps + additional_caps → total_caps = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_sarah_bottle_caps_l2227_222747


namespace NUMINAMATH_GPT_complete_collection_prob_l2227_222703

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end NUMINAMATH_GPT_complete_collection_prob_l2227_222703


namespace NUMINAMATH_GPT_range_of_m_l2227_222706

variable {x m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) : (¬ p m ∨ ¬ q m) → m ≥ 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2227_222706


namespace NUMINAMATH_GPT_hired_waiters_l2227_222781

theorem hired_waiters (W H : Nat) (hcooks : Nat := 9) 
                      (initial_ratio : 3 * W = 11 * hcooks)
                      (new_ratio : 9 = 5 * (W + H)) 
                      (original_waiters : W = 33) 
                      : H = 12 :=
by
  sorry

end NUMINAMATH_GPT_hired_waiters_l2227_222781


namespace NUMINAMATH_GPT_total_books_l2227_222736

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) : 
  joan_books + tom_books + sarah_books + alex_books = 118 := 
by 
  sorry

end NUMINAMATH_GPT_total_books_l2227_222736


namespace NUMINAMATH_GPT_parts_per_hour_equality_l2227_222753

variable {x : ℝ}

theorem parts_per_hour_equality (h1 : x - 4 > 0) :
  (100 / x) = (80 / (x - 4)) :=
sorry

end NUMINAMATH_GPT_parts_per_hour_equality_l2227_222753


namespace NUMINAMATH_GPT_circle_area_ratio_l2227_222745

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_circle_area_ratio_l2227_222745
