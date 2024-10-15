import Mathlib

namespace NUMINAMATH_GPT_expression_divisible_by_7_l445_44506

theorem expression_divisible_by_7 (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ m : ℤ, 3^(6*n-1) - k * 2^(3*n-2) + 1 = 7 * m) ↔ ∃ m' : ℤ, k = 7 * m' + 3 := 
by
  sorry

end NUMINAMATH_GPT_expression_divisible_by_7_l445_44506


namespace NUMINAMATH_GPT_can_form_triangle_l445_44515

theorem can_form_triangle (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_condition : c^2 ≤ 4 * a * b) : 
  a + b > c ∧ a + c > b ∧ b + c > a := 
sorry

end NUMINAMATH_GPT_can_form_triangle_l445_44515


namespace NUMINAMATH_GPT_equal_distribution_l445_44508

def earnings : List ℕ := [30, 35, 45, 55, 65]

def total_earnings : ℕ := earnings.sum

def equal_share (total: ℕ) : ℕ := total / earnings.length

def redistribution_amount (earner: ℕ) (equal: ℕ) : ℕ := earner - equal

theorem equal_distribution :
  redistribution_amount 65 (equal_share total_earnings) = 19 :=
by
  sorry

end NUMINAMATH_GPT_equal_distribution_l445_44508


namespace NUMINAMATH_GPT_width_decrease_l445_44509

-- Given conditions and known values
variable (L W : ℝ) -- original length and width
variable (P : ℝ)   -- percentage decrease in width

-- The known condition for the area comparison
axiom area_condition : 1.4 * (L * (W * (1 - P / 100))) = 1.1199999999999999 * (L * W)

-- The property we want to prove
theorem width_decrease (L W: ℝ) (h : L > 0) (h1 : W > 0) :
  P = 20 := 
by
  sorry

end NUMINAMATH_GPT_width_decrease_l445_44509


namespace NUMINAMATH_GPT_john_amount_share_l445_44582

theorem john_amount_share {total_amount : ℕ} {total_parts john_share : ℕ} (h1 : total_amount = 4200) (h2 : total_parts = 2 + 4 + 6) (h3 : john_share = 2) :
  john_share * (total_amount / total_parts) = 700 :=
by
  sorry

end NUMINAMATH_GPT_john_amount_share_l445_44582


namespace NUMINAMATH_GPT_negation_of_proposition_l445_44580

theorem negation_of_proposition :
    (¬ ∃ (x : ℝ), (Real.exp x - x - 1 < 0)) ↔ (∀ (x : ℝ), Real.exp x - x - 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l445_44580


namespace NUMINAMATH_GPT_range_of_m_l445_44513

theorem range_of_m (x y m : ℝ) (h1 : x - 2 * y = 1) (h2 : 2 * x + y = 4 * m) (h3 : x + 3 * y < 6) : m < 7 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l445_44513


namespace NUMINAMATH_GPT_max_arithmetic_subsequences_l445_44559

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d c : ℤ), ∀ n : ℕ, a n = d * n + c

-- Condition that the sum of the indices is even
def sum_indices_even (n m : ℕ) : Prop :=
  (n % 2 = 0 ∧ m % 2 = 0) ∨ (n % 2 = 1 ∧ m % 2 = 1)

-- Maximum count of 3-term arithmetic sequences in a sequence of 20 terms
theorem max_arithmetic_subsequences (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) :
  ∃ n : ℕ, n = 180 :=
by
  sorry

end NUMINAMATH_GPT_max_arithmetic_subsequences_l445_44559


namespace NUMINAMATH_GPT_op_4_3_equals_23_l445_44527

def op (a b : ℕ) : ℕ := a ^ 2 + a * b + a - b ^ 2

theorem op_4_3_equals_23 : op 4 3 = 23 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_op_4_3_equals_23_l445_44527


namespace NUMINAMATH_GPT_rectangle_area_l445_44587

theorem rectangle_area (side_of_square := 45)
  (radius_of_circle := side_of_square)
  (length_of_rectangle := (2/5 : ℚ) * radius_of_circle)
  (breadth_of_rectangle := 10) :
  breadth_of_rectangle * length_of_rectangle = 180 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l445_44587


namespace NUMINAMATH_GPT_range_abs_plus_one_l445_44543

 theorem range_abs_plus_one : 
   ∀ y : ℝ, (∃ x : ℝ, y = |x| + 1) ↔ y ≥ 1 := 
 by
   sorry
 
end NUMINAMATH_GPT_range_abs_plus_one_l445_44543


namespace NUMINAMATH_GPT_three_digit_number_base_10_l445_44548

theorem three_digit_number_base_10 (A B C : ℕ) (x : ℕ)
  (h1 : x = 100 * A + 10 * B + 6)
  (h2 : x = 82 * C + 36)
  (hA : 1 ≤ A ∧ A ≤ 9)
  (hB : 0 ≤ B ∧ B ≤ 9)
  (hC : 0 ≤ C ∧ C ≤ 8) :
  x = 446 := by
  sorry

end NUMINAMATH_GPT_three_digit_number_base_10_l445_44548


namespace NUMINAMATH_GPT_grill_run_time_l445_44584

def time_burn (coals : ℕ) (burn_rate : ℕ) (interval : ℕ) : ℚ :=
  (coals / burn_rate) * interval

theorem grill_run_time :
  let time_a1 := time_burn 60 15 20
  let time_a2 := time_burn 75 12 20
  let time_a3 := time_burn 45 15 20
  let time_b1 := time_burn 50 10 30
  let time_b2 := time_burn 70 8 30
  let time_b3 := time_burn 40 10 30
  let time_b4 := time_burn 80 8 30
  time_a1 + time_a2 + time_a3 + time_b1 + time_b2 + time_b3 + time_b4 = 1097.5 := sorry

end NUMINAMATH_GPT_grill_run_time_l445_44584


namespace NUMINAMATH_GPT_problem1_problem2_l445_44503

section
variable {α : Real}
variable (tan_α : Real)
variable (sin_α cos_α : Real)

def trigonometric_identities (tan_α sin_α cos_α : Real) : Prop :=
  tan_α = 2 ∧ sin_α = tan_α * cos_α

theorem problem1 (h : trigonometric_identities tan_α sin_α cos_α) :
  (4 * sin_α - 2 * cos_α) / (5 * cos_α + 3 * sin_α) = 6 / 11 := by
  sorry

theorem problem2 (h : trigonometric_identities tan_α sin_α cos_α) :
  (1 / 4 * sin_α^2 + 1 / 3 * sin_α * cos_α + 1 / 2 * cos_α^2) = 13 / 30 := by
  sorry
end

end NUMINAMATH_GPT_problem1_problem2_l445_44503


namespace NUMINAMATH_GPT_gcd_of_gx_and_x_l445_44520

theorem gcd_of_gx_and_x (x : ℕ) (h : 7200 ∣ x) : Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 5) * (4 * x + 7)) x = 30 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_gx_and_x_l445_44520


namespace NUMINAMATH_GPT_jane_baking_time_l445_44540

-- Definitions based on the conditions
variables (J : ℝ) (J_time : J > 0) -- J is the time it takes Jane to bake cakes individually
variables (Roy_time : 5 > 0) -- Roy can bake cakes in 5 hours
variables (together_time : 2 > 0) -- They work together for 2 hours
variables (remaining_time : 0.4 > 0) -- Jane completes the remaining task in 0.4 hours alone

-- Lean statement to prove Jane's individual baking time
theorem jane_baking_time : 
  (2 * (1 / J + 1 / 5) + 0.4 * (1 / J) = 1) → 
  J = 4 :=
by 
  sorry

end NUMINAMATH_GPT_jane_baking_time_l445_44540


namespace NUMINAMATH_GPT_arc_length_of_octagon_side_l445_44589

-- Define the conditions
def is_regular_octagon (side_length : ℝ) (angle_subtended : ℝ) := side_length = 5 ∧ angle_subtended = 2 * Real.pi / 8

-- Define the property to be proved
theorem arc_length_of_octagon_side :
  ∀ (side_length : ℝ) (angle_subtended : ℝ), 
    is_regular_octagon side_length angle_subtended →
    (angle_subtended / (2 * Real.pi)) * (2 * Real.pi * side_length) = 5 * Real.pi / 4 :=
by
  intros side_length angle_subtended h
  unfold is_regular_octagon at h
  sorry

end NUMINAMATH_GPT_arc_length_of_octagon_side_l445_44589


namespace NUMINAMATH_GPT_trig_identity_example_l445_44553

theorem trig_identity_example (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (4 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_example_l445_44553


namespace NUMINAMATH_GPT_problem_1_problem_2_l445_44594

def is_in_solution_set (x : ℝ) : Prop := -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0

variables {a b : ℝ}

theorem problem_1 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |(1 / 3) * a + (1 / 6) * b| < 1 / 4 :=
sorry

theorem problem_2 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l445_44594


namespace NUMINAMATH_GPT_triangle_inequality_l445_44561

variable (a b c : ℝ)

theorem triangle_inequality (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b) : 
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l445_44561


namespace NUMINAMATH_GPT_max_value_q_l445_44544

noncomputable def q (A M C : ℕ) : ℕ :=
  A * M * C + A * M + M * C + C * A + A + M + C

theorem max_value_q : ∀ A M C : ℕ, A + M + C = 15 → q A M C ≤ 215 :=
by 
  sorry

end NUMINAMATH_GPT_max_value_q_l445_44544


namespace NUMINAMATH_GPT_sum_of_coefficients_l445_44556

theorem sum_of_coefficients (d : ℤ) (h : d ≠ 0) :
    let a := 3 + 2
    let b := 17 + 2
    let c := 10 + 5
    let e := 16 + 4
    a + b + c + e = 59 :=
by
  let a := 3 + 2
  let b := 17 + 2
  let c := 10 + 5
  let e := 16 + 4
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l445_44556


namespace NUMINAMATH_GPT_cannot_cut_out_rect_l445_44519

noncomputable def square_area : ℝ := 400
noncomputable def rect_area : ℝ := 300
noncomputable def length_to_width_ratio : ℝ × ℝ := (3, 2)

theorem cannot_cut_out_rect (h1: square_area = 400) (h2: rect_area = 300) (h3: length_to_width_ratio = (3, 2)) : 
  false := sorry

end NUMINAMATH_GPT_cannot_cut_out_rect_l445_44519


namespace NUMINAMATH_GPT_find_tan_G_l445_44583

def right_triangle (FG GH FH : ℕ) : Prop :=
  FG^2 = GH^2 + FH^2

def tan_ratio (GH FH : ℕ) : ℚ :=
  FH / GH

theorem find_tan_G
  (FG GH : ℕ)
  (H1 : FG = 13)
  (H2 : GH = 12)
  (FH : ℕ)
  (H3 : right_triangle FG GH FH) :
  tan_ratio GH FH = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_find_tan_G_l445_44583


namespace NUMINAMATH_GPT_simplify_expression_l445_44595

theorem simplify_expression (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) = 
  (y^6 - y^5 + 2 * y^4 + y^3 - 2) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l445_44595


namespace NUMINAMATH_GPT_parabola_focus_coords_l445_44511

theorem parabola_focus_coords :
  ∀ (x y : ℝ), y^2 = -4 * x → (x, y) = (-1, 0) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_parabola_focus_coords_l445_44511


namespace NUMINAMATH_GPT_electronics_weight_l445_44539

variable (B C E : ℝ)

-- Conditions
def initial_ratio : Prop := B / 5 = C / 4 ∧ C / 4 = E / 2
def removed_clothes : Prop := B / 10 = (C - 9) / 4

-- Proof statement
theorem electronics_weight (h1 : initial_ratio B C E) (h2 : removed_clothes B C) : E = 9 := 
by
  sorry

end NUMINAMATH_GPT_electronics_weight_l445_44539


namespace NUMINAMATH_GPT_percentage_area_covered_by_pentagons_l445_44563

theorem percentage_area_covered_by_pentagons :
  ∀ (a : ℝ), (∃ (large_square_area small_square_area pentagon_area : ℝ),
    large_square_area = 16 * a^2 ∧
    small_square_area = a^2 ∧
    pentagon_area = 10 * small_square_area ∧
    (pentagon_area / large_square_area) * 100 = 62.5) :=
sorry

end NUMINAMATH_GPT_percentage_area_covered_by_pentagons_l445_44563


namespace NUMINAMATH_GPT_shop_owner_profitable_l445_44529

noncomputable def shop_owner_profit (CP_SP_difference_percentage: ℚ) (CP: ℚ) (buy_cheat_percentage: ℚ) (sell_cheat_percentage: ℚ) (buy_discount_percentage: ℚ) (sell_markup_percentage: ℚ) : ℚ := 
  CP_SP_difference_percentage * 100

theorem shop_owner_profitable :
  shop_owner_profit ((114 * (110 / 80 / 100) - 90) / 90) 1 0.14 0.20 0.10 0.10 = 74.17 := 
by
  sorry

end NUMINAMATH_GPT_shop_owner_profitable_l445_44529


namespace NUMINAMATH_GPT_reciprocal_of_one_is_one_l445_44599

def is_reciprocal (x y : ℝ) : Prop := x * y = 1

theorem reciprocal_of_one_is_one : is_reciprocal 1 1 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_one_is_one_l445_44599


namespace NUMINAMATH_GPT_zero_in_interval_l445_44593

noncomputable def f (x : ℝ) := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end NUMINAMATH_GPT_zero_in_interval_l445_44593


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l445_44514

theorem sum_of_digits_of_N (N : ℕ) (hN : N * (N + 1) / 2 = 3003) :
  (Nat.digits 10 N).sum = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l445_44514


namespace NUMINAMATH_GPT_probability_y_eq_2x_l445_44592

/-- Two fair cubic dice each have six faces labeled with the numbers 1, 2, 3, 4, 5, and 6. 
Rolling these dice sequentially, find the probability that the number on the top face 
of the second die (y) is twice the number on the top face of the first die (x). --/
noncomputable def dice_probability : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem probability_y_eq_2x : dice_probability = 1 / 12 :=
  by sorry

end NUMINAMATH_GPT_probability_y_eq_2x_l445_44592


namespace NUMINAMATH_GPT_sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l445_44586

def row_10_pascals_triangle : List ℕ := [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]

theorem sum_of_row_10_pascals_triangle :
  (List.sum row_10_pascals_triangle) = 1024 := by
  sorry

theorem sum_of_squares_of_row_10_pascals_triangle :
  (List.sum (List.map (fun x => x * x) row_10_pascals_triangle)) = 183756 := by
  sorry

end NUMINAMATH_GPT_sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l445_44586


namespace NUMINAMATH_GPT_nth_equation_l445_44550

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - 1 = 4 * n * (n + 1) := 
by
  sorry

end NUMINAMATH_GPT_nth_equation_l445_44550


namespace NUMINAMATH_GPT_cost_per_liter_of_gas_today_l445_44552

-- Definition of the conditions
def oil_price_rollback : ℝ := 0.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters := liters_today + liters_friday
def total_cost : ℝ := 39

-- The theorem to prove
theorem cost_per_liter_of_gas_today (C : ℝ) :
  (liters_today * C) + (liters_friday * (C - oil_price_rollback)) = total_cost →
  C = 1.4 := 
by 
  sorry

end NUMINAMATH_GPT_cost_per_liter_of_gas_today_l445_44552


namespace NUMINAMATH_GPT_fraction_ratio_equivalence_l445_44537

theorem fraction_ratio_equivalence :
  ∃ (d : ℚ), d = 240 / 1547 ∧ ((2 / 13) / d) = ((5 / 34) / (7 / 48)) := 
by
  sorry

end NUMINAMATH_GPT_fraction_ratio_equivalence_l445_44537


namespace NUMINAMATH_GPT_unique_ordered_triple_l445_44578

theorem unique_ordered_triple (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ab : Nat.lcm a b = 500) (h_bc : Nat.lcm b c = 2000) (h_ca : Nat.lcm c a = 2000) :
  (a = 100 ∧ b = 2000 ∧ c = 2000) :=
by
  sorry

end NUMINAMATH_GPT_unique_ordered_triple_l445_44578


namespace NUMINAMATH_GPT_sum_arithmetic_series_l445_44507

theorem sum_arithmetic_series :
  let a1 := 1000
  let an := 5000
  let d := 4
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 3003000 := by
    sorry

end NUMINAMATH_GPT_sum_arithmetic_series_l445_44507


namespace NUMINAMATH_GPT_exponential_fraction_l445_44536

theorem exponential_fraction :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_exponential_fraction_l445_44536


namespace NUMINAMATH_GPT_diameter_percentage_l445_44533

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.25 * π * (d_S / 2)^2) : 
  d_R = 0.5 * d_S :=
by 
  sorry

end NUMINAMATH_GPT_diameter_percentage_l445_44533


namespace NUMINAMATH_GPT_simple_interest_difference_l445_44568

theorem simple_interest_difference :
  let P : ℝ := 900
  let R1 : ℝ := 4
  let R2 : ℝ := 4.5
  let T : ℝ := 7
  let SI1 := P * R1 * T / 100
  let SI2 := P * R2 * T / 100
  SI2 - SI1 = 31.50 := by
  sorry

end NUMINAMATH_GPT_simple_interest_difference_l445_44568


namespace NUMINAMATH_GPT_proof_problem_l445_44590

-- Given conditions
variables {a b c : ℕ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a > b) (h5 : a^2 - a * b - a * c + b * c = 7)

-- Statement to prove
theorem proof_problem : a - c = 1 ∨ a - c = 7 :=
sorry

end NUMINAMATH_GPT_proof_problem_l445_44590


namespace NUMINAMATH_GPT_largest_x_l445_44524

theorem largest_x (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 := 
sorry

end NUMINAMATH_GPT_largest_x_l445_44524


namespace NUMINAMATH_GPT_symmetric_linear_functions_l445_44576

theorem symmetric_linear_functions :
  (∃ (a b : ℝ), ∀ x y : ℝ, (y = a * x + 2 ∧ y = 3 * x - b) → a = 1 / 3 ∧ b = 6) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_linear_functions_l445_44576


namespace NUMINAMATH_GPT_simplify_expression_l445_44597

theorem simplify_expression (x : ℝ) (h : x^2 - x - 1 = 0) :
  ( ( (x - 1) / x - (x - 2) / (x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) ) = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l445_44597


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l445_44588

noncomputable def equilateral_side_length (p_eq : ℕ) : ℕ := p_eq / 3

theorem isosceles_triangle_base_length (p_eq p_iso s b : ℕ) 
  (h1 : p_eq = 45)
  (h2 : p_iso = 40)
  (h3 : s = equilateral_side_length p_eq)
  (h4 : p_iso = s + s + b)
  : b = 10 :=
by
  simp [h1, h2, h3] at h4
  -- steps to solve for b would be written here
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l445_44588


namespace NUMINAMATH_GPT_infinite_geometric_series_l445_44523

theorem infinite_geometric_series
  (p q r : ℝ)
  (h_series : ∑' n : ℕ, p / q^(n+1) = 9) :
  (∑' n : ℕ, p / (p + r)^(n+1)) = (9 * (q - 1)) / (9 * q + r - 10) :=
by 
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_l445_44523


namespace NUMINAMATH_GPT_matt_total_score_l445_44579

-- Definitions from the conditions
def num_2_point_shots : ℕ := 4
def num_3_point_shots : ℕ := 2
def score_per_2_point_shot : ℕ := 2
def score_per_3_point_shot : ℕ := 3

-- Proof statement
theorem matt_total_score : 
  (num_2_point_shots * score_per_2_point_shot) + 
  (num_3_point_shots * score_per_3_point_shot) = 14 := 
by 
  sorry  -- placeholder for the actual proof

end NUMINAMATH_GPT_matt_total_score_l445_44579


namespace NUMINAMATH_GPT_find_minimal_product_l445_44542

theorem find_minimal_product : ∃ x y : ℤ, (20 * x + 19 * y = 2019) ∧ (x * y = 2623) ∧ (∀ z w : ℤ, (20 * z + 19 * w = 2019) → |x - y| ≤ |z - w|) :=
by
  -- definitions and theorems to prove the problem would be placed here
  sorry

end NUMINAMATH_GPT_find_minimal_product_l445_44542


namespace NUMINAMATH_GPT_food_remaining_l445_44581

-- Definitions for conditions
def first_week_donations : ℕ := 40
def second_week_donations := 2 * first_week_donations
def total_donations := first_week_donations + second_week_donations
def percentage_given_out : ℝ := 0.70
def amount_given_out := percentage_given_out * total_donations

-- Proof goal
theorem food_remaining (h1 : first_week_donations = 40)
                      (h2 : second_week_donations = 2 * first_week_donations)
                      (h3 : percentage_given_out = 0.70) :
                      total_donations - amount_given_out = 36 := by
  sorry

end NUMINAMATH_GPT_food_remaining_l445_44581


namespace NUMINAMATH_GPT_probability_first_spade_second_ace_l445_44557

theorem probability_first_spade_second_ace :
  let n : ℕ := 52
  let spades : ℕ := 13
  let aces : ℕ := 4
  let ace_of_spades : ℕ := 1
  let non_ace_spades : ℕ := spades - ace_of_spades
  (non_ace_spades / n : ℚ) * (aces / (n - 1) : ℚ) +
  (ace_of_spades / n : ℚ) * ((aces - 1) / (n - 1) : ℚ) =
  (1 / n : ℚ) :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_probability_first_spade_second_ace_l445_44557


namespace NUMINAMATH_GPT_probability_of_one_triplet_without_any_pairs_l445_44516

noncomputable def probability_one_triplet_no_pairs : ℚ :=
  let total_outcomes := 6^5
  let choices_for_triplet := 6
  let ways_to_choose_triplet_dice := Nat.choose 5 3
  let choices_for_remaining_dice := 5 * 4
  let successful_outcomes := choices_for_triplet * ways_to_choose_triplet_dice * choices_for_remaining_dice
  successful_outcomes / total_outcomes

theorem probability_of_one_triplet_without_any_pairs :
  probability_one_triplet_no_pairs = 25 / 129 := by
  sorry

end NUMINAMATH_GPT_probability_of_one_triplet_without_any_pairs_l445_44516


namespace NUMINAMATH_GPT_james_ride_time_l445_44570

theorem james_ride_time :
  let distance := 80 
  let speed := 16 
  distance / speed = 5 := 
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_james_ride_time_l445_44570


namespace NUMINAMATH_GPT_numberOfBoysInClass_l445_44530

-- Define the problem condition: students sit in a circle and boy at 5th position is opposite to boy at 20th position
def studentsInCircle (n : ℕ) : Prop :=
  (n > 5) ∧ (n > 20) ∧ ((20 - 5) * 2 + 2 = n)

-- The main theorem: Given the conditions, prove the total number of boys equals 32
theorem numberOfBoysInClass : ∀ n : ℕ, studentsInCircle n → n = 32 :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_numberOfBoysInClass_l445_44530


namespace NUMINAMATH_GPT_problem1_l445_44538

theorem problem1 :
  (2021 - Real.pi)^0 + (Real.sqrt 3 - 1) - 2 + (2 * Real.sqrt 3) = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l445_44538


namespace NUMINAMATH_GPT_Harkamal_total_payment_l445_44528

theorem Harkamal_total_payment :
  let cost_grapes := 10 * 70
  let cost_mangoes := 9 * 55
  let cost_apples := 12 * 80
  let cost_papayas := 7 * 45
  let cost_oranges := 15 * 30
  let cost_bananas := 5 * 25
  cost_grapes + cost_mangoes + cost_apples + cost_papayas + cost_oranges + cost_bananas = 3045 := by
  sorry

end NUMINAMATH_GPT_Harkamal_total_payment_l445_44528


namespace NUMINAMATH_GPT_peter_read_more_books_l445_44585

/-
Given conditions:
  Peter has 20 books.
  Peter has read 40% of them.
  Peter's brother has read 10% of them.
We aim to prove that Peter has read 6 more books than his brother.
-/

def total_books : ℕ := 20
def peter_read_fraction : ℚ := 0.4
def brother_read_fraction : ℚ := 0.1

def books_read_by_peter := total_books * peter_read_fraction
def books_read_by_brother := total_books * brother_read_fraction

theorem peter_read_more_books :
  books_read_by_peter - books_read_by_brother = 6 := by
  sorry

end NUMINAMATH_GPT_peter_read_more_books_l445_44585


namespace NUMINAMATH_GPT_cheeseburger_cost_l445_44555

-- Definitions for given conditions
def milkshake_price : ℝ := 5
def cheese_fries_price : ℝ := 8
def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money := jim_money + cousin_money
def spending_percentage : ℝ := 0.80
def total_spent := spending_percentage * combined_money
def number_of_milkshakes : ℝ := 2
def number_of_cheeseburgers : ℝ := 2

-- Prove the cost of one cheeseburger
theorem cheeseburger_cost : (total_spent - (number_of_milkshakes * milkshake_price) - cheese_fries_price) / number_of_cheeseburgers = 3 :=
by
  sorry

end NUMINAMATH_GPT_cheeseburger_cost_l445_44555


namespace NUMINAMATH_GPT_completing_the_square_l445_44591

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end NUMINAMATH_GPT_completing_the_square_l445_44591


namespace NUMINAMATH_GPT_find_x_l445_44596

theorem find_x (x y : ℕ) (h1 : x / y = 6 / 3) (h2 : y = 27) : x = 54 :=
sorry

end NUMINAMATH_GPT_find_x_l445_44596


namespace NUMINAMATH_GPT_tank_capacity_l445_44525

theorem tank_capacity (C : ℝ) (h_leak : ∀ t, t = 6 -> C / 6 = C / t)
    (h_inlet : ∀ r, r = 240 -> r = 4 * 60)
    (h_net : ∀ t, t = 8 -> 240 - C / 6 = C / 8) :
    C = 5760 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_tank_capacity_l445_44525


namespace NUMINAMATH_GPT_cyclic_quadrilateral_angle_D_l445_44541

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h₁ : A + B + C + D = 360) (h₂ : ∃ x, A = 3 * x ∧ B = 4 * x ∧ C = 6 * x) :
  D = 100 :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_angle_D_l445_44541


namespace NUMINAMATH_GPT_num_seven_digit_palindromes_l445_44517

theorem num_seven_digit_palindromes : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices = 9000 :=
by
  sorry

end NUMINAMATH_GPT_num_seven_digit_palindromes_l445_44517


namespace NUMINAMATH_GPT_exists_lcm_lt_l445_44546

theorem exists_lcm_lt (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) (hp_gt_one : p > 1) (hq_gt_one : q > 1) (hpq_diff_gt_one : (p < q ∧ q - p > 1) ∨ (p > q ∧ p - q > 1)) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q := by
  sorry

end NUMINAMATH_GPT_exists_lcm_lt_l445_44546


namespace NUMINAMATH_GPT_correct_operation_l445_44562

theorem correct_operation :
  (∀ a : ℝ, (a^4)^2 ≠ a^6) ∧
  (∀ a b : ℝ, (a - b)^2 ≠ a^2 - ab + b^2) ∧
  (∀ a b : ℝ, 6 * a^2 * b / (2 * a * b) = 3 * a) ∧
  (∀ a : ℝ, a^2 + a^4 ≠ a^6) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_operation_l445_44562


namespace NUMINAMATH_GPT_purchasing_options_count_l445_44572

theorem purchasing_options_count : ∃ (s : Finset (ℕ × ℕ)), s.card = 4 ∧
  ∀ (a : ℕ × ℕ), a ∈ s ↔ 
    (80 * a.1 + 120 * a.2 = 1000) 
    ∧ (a.1 > 0) ∧ (a.2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_purchasing_options_count_l445_44572


namespace NUMINAMATH_GPT_smallest_n_divisible_by_24_and_864_l445_44521

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ, (0 < n) ∧ (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ (∀ m : ℕ, (0 < m) → (24 ∣ m^2) → (864 ∣ m^3) → (n ≤ m)) :=
sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_24_and_864_l445_44521


namespace NUMINAMATH_GPT_minimize_f_at_a_l445_44569

def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f_at_a (a : ℝ) (h : a = 82 / 43) :
  ∃ x, ∀ y, f x a ≤ f y a :=
sorry

end NUMINAMATH_GPT_minimize_f_at_a_l445_44569


namespace NUMINAMATH_GPT_ratio_girls_to_boys_l445_44549

theorem ratio_girls_to_boys (g b : ℕ) (h1 : g = b + 4) (h2 : g + b = 28) :
  g / gcd g b = 4 ∧ b / gcd g b = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_girls_to_boys_l445_44549


namespace NUMINAMATH_GPT_total_amount_correct_l445_44554

-- Define the prices of jeans and tees
def price_jean : ℕ := 11
def price_tee : ℕ := 8

-- Define the quantities sold
def quantity_jeans_sold : ℕ := 4
def quantity_tees_sold : ℕ := 7

-- Calculate the total amount earned
def total_amount : ℕ := (price_jean * quantity_jeans_sold) + (price_tee * quantity_tees_sold)

-- Now, we state and prove the theorem
theorem total_amount_correct : total_amount = 100 :=
by
  -- Here we assert the correctness of the calculation
  sorry

end NUMINAMATH_GPT_total_amount_correct_l445_44554


namespace NUMINAMATH_GPT_median_on_hypotenuse_length_l445_44564

theorem median_on_hypotenuse_length
  (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) (right_triangle : (a ^ 2 + b ^ 2) = c ^ 2) :
  (1 / 2) * c = 5 :=
  sorry

end NUMINAMATH_GPT_median_on_hypotenuse_length_l445_44564


namespace NUMINAMATH_GPT_sheryll_paid_total_l445_44547

-- Variables/conditions
variables (cost_per_book : ℝ) (num_books : ℕ) (discount_per_book : ℝ)

-- Given conditions
def assumption1 : cost_per_book = 5 := by sorry
def assumption2 : num_books = 10 := by sorry
def assumption3 : discount_per_book = 0.5 := by sorry

-- Theorem statement
theorem sheryll_paid_total : cost_per_book = 5 → num_books = 10 → discount_per_book = 0.5 → 
  (cost_per_book - discount_per_book) * num_books = 45 := by
  sorry

end NUMINAMATH_GPT_sheryll_paid_total_l445_44547


namespace NUMINAMATH_GPT_range_of_m_l445_44501

variable (x m : ℝ)

theorem range_of_m (h1 : ∀ x : ℝ, 2 * x^2 - 2 * m * x + m < 0) 
    (h2 : ∃ a b : ℤ, a ≠ b ∧ ∀ x : ℝ, (a < x ∧ x < b) → 2 * x^2 - 2 * m * x + m < 0): 
    -8 / 5 ≤ m ∧ m < -2 / 3 ∨ 8 / 3 < m ∧ m ≤ 18 / 5 :=
sorry

end NUMINAMATH_GPT_range_of_m_l445_44501


namespace NUMINAMATH_GPT_max_min_values_of_f_l445_44545

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≥ -18) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = -18)
:= by
  sorry  -- To be replaced with the actual proof

end NUMINAMATH_GPT_max_min_values_of_f_l445_44545


namespace NUMINAMATH_GPT_geometric_sequence_sum_inv_l445_44522

theorem geometric_sequence_sum_inv
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_inv_l445_44522


namespace NUMINAMATH_GPT_cos_double_angle_given_tan_l445_44500

theorem cos_double_angle_given_tan (x : ℝ) (h : Real.tan x = 2) : Real.cos (2 * x) = -3 / 5 :=
by sorry

end NUMINAMATH_GPT_cos_double_angle_given_tan_l445_44500


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l445_44560

theorem number_of_sides_of_polygon :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 2 * n + 7 ∧ n = 8 := 
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l445_44560


namespace NUMINAMATH_GPT_max_value_of_S_n_divided_l445_44526

noncomputable def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def S_n (a₁ d n : ℕ) : ℕ :=
  n * (n + 4)

theorem max_value_of_S_n_divided (a₁ d : ℕ) (h₁ : ∀ n, a₁ + (2 * n - 1) * d = 2 * (a₁ + (n - 1) * d) - 3)
  (h₂ : (a₁ + 5 * d)^2 = a₁ * (a₁ + 20 * d)) :
  ∃ n, 2 * S_n a₁ d n / 2^n = 6 := 
sorry

end NUMINAMATH_GPT_max_value_of_S_n_divided_l445_44526


namespace NUMINAMATH_GPT_arithmetic_seq_a2_a8_a5_l445_44566

-- Define the sequence and sum conditions
variable {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Define the given conditions
axiom seq_condition (n : ℕ) : (1 - q) * S n + q * a n = 1
axiom q_nonzero : q * (q - 1) ≠ 0
axiom geom_seq : ∀ n, a n = q^(n - 1)

-- Main theorem (consistent with both parts (Ⅰ) and (Ⅱ) results)
theorem arithmetic_seq_a2_a8_a5 (S_arith : S 3 + S 6 = 2 * S 9) : a 2 + a 5 = 2 * a 8 :=
by
    sorry

end NUMINAMATH_GPT_arithmetic_seq_a2_a8_a5_l445_44566


namespace NUMINAMATH_GPT_allocation_schemes_correct_l445_44531

noncomputable def allocation_schemes : Nat :=
  let C (n k : Nat) : Nat := Nat.choose n k
  -- Calculate category 1: one school gets 1 professor, two get 2 professors each
  let category1 := C 3 1 * C 5 1 * C 4 2 * C 2 2 / 2
  -- Calculate category 2: one school gets 3 professors, two get 1 professor each
  let category2 := C 3 1 * C 5 3 * C 2 1 * C 1 1 / 2
  -- Total allocation ways
  let totalWays := 6 * (category1 + category2)
  totalWays

theorem allocation_schemes_correct : allocation_schemes = 900 := by
  sorry

end NUMINAMATH_GPT_allocation_schemes_correct_l445_44531


namespace NUMINAMATH_GPT_num_five_letter_words_correct_l445_44575

noncomputable def num_five_letter_words : ℕ := 1889568

theorem num_five_letter_words_correct :
  let a := 3
  let e := 4
  let i := 2
  let o := 5
  let u := 4
  (a + e + i + o + u) ^ 5 = num_five_letter_words :=
by
  sorry

end NUMINAMATH_GPT_num_five_letter_words_correct_l445_44575


namespace NUMINAMATH_GPT_circle_O₁_equation_sum_of_squares_constant_l445_44567

-- Given conditions
def circle_O (x y : ℝ) := x^2 + y^2 = 25
def center_O₁ (m : ℝ) : ℝ × ℝ := (m, 0) 
def intersect_point := (3, 4)
def is_intersection (x y : ℝ) := circle_O x y ∧ (x - intersect_point.1)^2 + (y - intersect_point.2)^2 = 0
def line_passing_P (k : ℝ) (x y : ℝ) := y - intersect_point.2 = k * (x - intersect_point.1)
def point_on_circle (circle : ℝ × ℝ → Prop) (x y : ℝ) := circle (x, y)
def distance_squared (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Problem statements
theorem circle_O₁_equation (k : ℝ) (m : ℝ) (x y : ℝ) (h : k = 1) (h_intersect: is_intersection 3 4)
  (h_BP_distance : distance_squared (3, 4) (x, y) = (7 * Real.sqrt 2)^2) : 
  (x - 14)^2 + y^2 = 137 := sorry

theorem sum_of_squares_constant (k m : ℝ) (h : k ≠ 0) (h_perpendicular : line_passing_P (-1/k) 3 4)
  (A B C D : ℝ × ℝ) (h_AB_distance : distance_squared A B = 4 * m^2 / (1 + k^2)) 
  (h_CD_distance : distance_squared C D = 4 * m^2 * k^2 / (1 + k^2)) : 
  distance_squared A B + distance_squared C D = 4 * m^2 := sorry

end NUMINAMATH_GPT_circle_O₁_equation_sum_of_squares_constant_l445_44567


namespace NUMINAMATH_GPT_rectangle_dimension_l445_44571

theorem rectangle_dimension (x : ℝ) (h : (x^2) * (x + 5) = 3 * (2 * (x^2) + 2 * (x + 5))) : x = 3 :=
by
  have eq1 : (x^2) * (x + 5) = x^3 + 5 * x^2 := by ring
  have eq2 : 3 * (2 * (x^2) + 2 * (x + 5)) = 6 * x^2 + 6 * x + 30 := by ring
  rw [eq1, eq2] at h
  sorry  -- Proof details omitted

end NUMINAMATH_GPT_rectangle_dimension_l445_44571


namespace NUMINAMATH_GPT_lcm_of_12_and_15_l445_44574
-- Import the entire Mathlib library

-- Define the given conditions
def HCF (a b : ℕ) : ℕ := gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / (gcd a b)

-- Given the values
def a := 12
def b := 15
def hcf := 3

-- State the proof problem
theorem lcm_of_12_and_15 : LCM a b = 60 :=
by
  -- Proof goes here (skipped)
  sorry

end NUMINAMATH_GPT_lcm_of_12_and_15_l445_44574


namespace NUMINAMATH_GPT_prime_divides_3np_minus_3n1_l445_44573

theorem prime_divides_3np_minus_3n1 (p n : ℕ) (hp : Prime p) : p ∣ (3^(n + p) - 3^(n + 1)) :=
sorry

end NUMINAMATH_GPT_prime_divides_3np_minus_3n1_l445_44573


namespace NUMINAMATH_GPT_larger_number_solution_l445_44551

theorem larger_number_solution (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_solution_l445_44551


namespace NUMINAMATH_GPT_xiao_cong_math_score_l445_44598

theorem xiao_cong_math_score :
  ∀ (C M E : ℕ),
    (C + M + E) / 3 = 122 → C = 118 → E = 125 → M = 123 :=
by
  intros C M E h1 h2 h3
  sorry

end NUMINAMATH_GPT_xiao_cong_math_score_l445_44598


namespace NUMINAMATH_GPT_saturated_function_2014_l445_44565

def saturated (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f^[f^[f n] n] n = n

theorem saturated_function_2014 (f : ℕ → ℕ) (m : ℕ) (h : saturated f) :
  (m ∣ 2014) ↔ (f^[2014] m = m) :=
sorry

end NUMINAMATH_GPT_saturated_function_2014_l445_44565


namespace NUMINAMATH_GPT_intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l445_44512

noncomputable def A := {x : ℝ | -4 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem intersection_A_B_when_m_eq_2 : (A ∩ B 2) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

theorem range_of_m_for_p_implies_q : {m : ℝ | m ≥ 5} = {m : ℝ | ∀ x, ((x^2 + 2 * x - 8 < 0) → ((x - 1 + m) * (x - 1 - m) ≤ 0)) ∧ ¬((x - 1 + m) * (x - 1 - m) ≤ 0 → (x^2 + 2 * x - 8 < 0))} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l445_44512


namespace NUMINAMATH_GPT_largest_k_sum_of_consecutive_odds_l445_44558

theorem largest_k_sum_of_consecutive_odds (k m : ℕ) (h1 : k * (2 * m + k) = 2^15) : k ≤ 128 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_k_sum_of_consecutive_odds_l445_44558


namespace NUMINAMATH_GPT_smallest_solution_eq_l445_44532

theorem smallest_solution_eq :
  (∀ x : ℝ, x ≠ 3 →
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 15) → 
  x = 1 - Real.sqrt 10 ∨ (∃ y : ℝ, y ≤ 1 - Real.sqrt 10 ∧ y ≠ 3 ∧ 3 * y / (y - 3) + (3 * y^2 - 27) / y = 15)) :=
sorry

end NUMINAMATH_GPT_smallest_solution_eq_l445_44532


namespace NUMINAMATH_GPT_value_expression_l445_44535

theorem value_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by 
  sorry

end NUMINAMATH_GPT_value_expression_l445_44535


namespace NUMINAMATH_GPT_factor_expression_l445_44505

variable (x : ℝ)

theorem factor_expression : 
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := 
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l445_44505


namespace NUMINAMATH_GPT_equivalence_of_statements_l445_44510

variable (S M : Prop)

theorem equivalence_of_statements : 
  (S → M) ↔ ((¬M → ¬S) ∧ (¬S ∨ M)) :=
by
  sorry

end NUMINAMATH_GPT_equivalence_of_statements_l445_44510


namespace NUMINAMATH_GPT_foci_distance_l445_44518

variable (x y : ℝ)

def ellipse_eq : Prop := (x^2 / 45) + (y^2 / 5) = 9

theorem foci_distance : ellipse_eq x y → (distance_between_foci : ℝ) = 12 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_foci_distance_l445_44518


namespace NUMINAMATH_GPT_triangle_area_is_9_point_5_l445_44504

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 1)
def B : Point := (4, 0)
def C : Point := (3, 5)

noncomputable def areaOfTriangle (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_is_9_point_5 :
  areaOfTriangle A B C = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_9_point_5_l445_44504


namespace NUMINAMATH_GPT_average_paper_tape_length_l445_44577

-- Define the lengths of the paper tapes as given in the conditions
def red_tape_length : ℝ := 20
def purple_tape_length : ℝ := 16

-- State the proof problem
theorem average_paper_tape_length : 
  (red_tape_length + purple_tape_length) / 2 = 18 := 
by
  sorry

end NUMINAMATH_GPT_average_paper_tape_length_l445_44577


namespace NUMINAMATH_GPT_geom_seq_sum_l445_44534

theorem geom_seq_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 + a 2 = 16) 
  (h2 : a 3 + a 4 = 24) 
  (h_geom : ∀ n, a (n+1) = r * a n):
  a 7 + a 8 = 54 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_l445_44534


namespace NUMINAMATH_GPT_condition1_condition2_condition3_condition4_l445_44502

-- Proof for the equivalence of conditions and point descriptions

theorem condition1 (x y : ℝ) : 
  (x >= -2) ↔ ∃ y : ℝ, x = -2 ∨ x > -2 := 
by
  sorry

theorem condition2 (x y : ℝ) : 
  (-2 < x ∧ x < 2) ↔ ∃ y : ℝ, -2 < x ∧ x < 2 := 
by
  sorry

theorem condition3 (x y : ℝ) : 
  (|x| < 2) ↔ -2 < x ∧ x < 2 :=
by
  sorry

theorem condition4 (x y : ℝ) : 
  (|x| ≥ 2) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by 
  sorry

end NUMINAMATH_GPT_condition1_condition2_condition3_condition4_l445_44502
