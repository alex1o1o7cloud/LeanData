import Mathlib

namespace NUMINAMATH_GPT_smallest_q_difference_l2027_202757

theorem smallest_q_difference (p q : ℕ) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_fraction1 : 3 * q < 5 * p)
  (h_fraction2 : 5 * p < 6 * q)
  (h_smallest : ∀ r s : ℕ, 0 < s → 3 * s < 5 * r → 5 * r < 6 * s → q ≤ s) :
  q - p = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_q_difference_l2027_202757


namespace NUMINAMATH_GPT_rectangle_ratio_width_length_l2027_202731

variable (w : ℝ)

theorem rectangle_ratio_width_length (h1 : w + 8 + w + 8 = 24) : 
  w / 8 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_width_length_l2027_202731


namespace NUMINAMATH_GPT_find_b_l2027_202725

theorem find_b 
  (a b : ℚ)
  (h_root : (1 + Real.sqrt 5) ^ 3 + a * (1 + Real.sqrt 5) ^ 2 + b * (1 + Real.sqrt 5) - 60 = 0) :
  b = 26 :=
sorry

end NUMINAMATH_GPT_find_b_l2027_202725


namespace NUMINAMATH_GPT_rabbit_wins_race_l2027_202707

theorem rabbit_wins_race :
  ∀ (rabbit_speed1 rabbit_speed2 snail_speed rest_time total_distance : ℕ)
  (rabbit_time1 rabbit_time2 : ℚ),
  rabbit_speed1 = 20 →
  rabbit_speed2 = 30 →
  snail_speed = 2 →
  rest_time = 3 →
  total_distance = 100 →
  rabbit_time1 = (30 : ℚ) / rabbit_speed1 →
  rabbit_time2 = (70 : ℚ) / rabbit_speed2 →
  (rabbit_time1 + rest_time + rabbit_time2 < total_distance / snail_speed) :=
by
  intros
  sorry

end NUMINAMATH_GPT_rabbit_wins_race_l2027_202707


namespace NUMINAMATH_GPT_trees_planted_in_garden_l2027_202726

theorem trees_planted_in_garden (yard_length : ℕ) (tree_distance : ℕ) (h₁ : yard_length = 500) (h₂ : tree_distance = 20) :
  ((yard_length / tree_distance) + 1) = 26 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_trees_planted_in_garden_l2027_202726


namespace NUMINAMATH_GPT_expression_f_range_a_l2027_202783

noncomputable def f (x : ℝ) : ℝ :=
if h : -1 ≤ x ∧ x ≤ 1 then x^3
else if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
else (x-4)^3

theorem expression_f (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) :
  f x =
    if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
    else (x-4)^3 :=
by sorry

theorem range_a (a : ℝ) : 
  (∃ x, f x > a) ↔ a < 1 :=
by sorry

end NUMINAMATH_GPT_expression_f_range_a_l2027_202783


namespace NUMINAMATH_GPT_smallest_interesting_number_l2027_202781

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_interesting_number_l2027_202781


namespace NUMINAMATH_GPT_amber_age_l2027_202701

theorem amber_age 
  (a g : ℕ)
  (h1 : g = 15 * a)
  (h2 : g - a = 70) :
  a = 5 :=
by
  sorry

end NUMINAMATH_GPT_amber_age_l2027_202701


namespace NUMINAMATH_GPT_max_stamps_l2027_202711

theorem max_stamps (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 45) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, n ≤ total_cents / price_per_stamp ∧ n = 111 :=
by
  sorry

end NUMINAMATH_GPT_max_stamps_l2027_202711


namespace NUMINAMATH_GPT_angle_of_inclination_l2027_202746

noncomputable def line_slope (a b : ℝ) : ℝ := 1  -- The slope of the line y = x + 1 is 1
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m -- angle of inclination is arctan of the slope

theorem angle_of_inclination (θ : ℝ) : 
  inclination_angle (line_slope 1 1) = 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l2027_202746


namespace NUMINAMATH_GPT_optimal_addition_amount_l2027_202732

def optimal_material_range := {x : ℝ | 100 ≤ x ∧ x ≤ 200}

def second_trial_amounts := {x : ℝ | x = 138.2 ∨ x = 161.8}

theorem optimal_addition_amount (
  h1 : ∀ x ∈ optimal_material_range, x ∈ second_trial_amounts
  ) :
  138.2 ∈ second_trial_amounts ∧ 161.8 ∈ second_trial_amounts :=
by
  sorry

end NUMINAMATH_GPT_optimal_addition_amount_l2027_202732


namespace NUMINAMATH_GPT_trigonometric_identity_l2027_202722

theorem trigonometric_identity :
  (1 / 2 - (Real.cos (15 * Real.pi / 180)) ^ 2) = - (Real.sqrt 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2027_202722


namespace NUMINAMATH_GPT_find_P_l2027_202762

theorem find_P (P : ℕ) (h : P^2 + P = 30) : P = 5 :=
sorry

end NUMINAMATH_GPT_find_P_l2027_202762


namespace NUMINAMATH_GPT_pow_div_pow_eq_l2027_202749

theorem pow_div_pow_eq :
  (3^12) / (27^2) = 729 :=
by
  -- We'll use the provided conditions and proof outline
  -- 1. 27 = 3^3
  -- 2. (a^b)^c = a^{bc}
  -- 3. a^b \div a^c = a^{b-c}
  sorry

end NUMINAMATH_GPT_pow_div_pow_eq_l2027_202749


namespace NUMINAMATH_GPT_diagonal_crosses_768_unit_cubes_l2027_202702

-- Defining the dimensions of the rectangular prism
def a : ℕ := 150
def b : ℕ := 324
def c : ℕ := 375

-- Computing the gcd values
def gcd_ab : ℕ := Nat.gcd a b
def gcd_ac : ℕ := Nat.gcd a c
def gcd_bc : ℕ := Nat.gcd b c
def gcd_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- Using the formula to compute the number of unit cubes the diagonal intersects
def num_unit_cubes : ℕ := a + b + c - gcd_ab - gcd_ac - gcd_bc + gcd_abc

-- Stating the theorem to prove
theorem diagonal_crosses_768_unit_cubes : num_unit_cubes = 768 := by
  sorry

end NUMINAMATH_GPT_diagonal_crosses_768_unit_cubes_l2027_202702


namespace NUMINAMATH_GPT_solve_problems_l2027_202735

variable (initial_problems : ℕ) 
variable (additional_problems : ℕ)

theorem solve_problems
  (h1 : initial_problems = 12) 
  (h2 : additional_problems = 7) : 
  initial_problems + additional_problems = 19 := 
by 
  sorry

end NUMINAMATH_GPT_solve_problems_l2027_202735


namespace NUMINAMATH_GPT_candy_left_l2027_202710

-- Definitions according to the conditions
def initialCandy : ℕ := 15
def candyGivenToHaley : ℕ := 6

-- Theorem statement formalizing the proof problem
theorem candy_left (c : ℕ) (h₁ : c = initialCandy - candyGivenToHaley) : c = 9 :=
by
  -- The proof is omitted as instructed.
  sorry

end NUMINAMATH_GPT_candy_left_l2027_202710


namespace NUMINAMATH_GPT_flat_fee_l2027_202716

theorem flat_fee (f n : ℝ) 
  (h1 : f + 3 * n = 205) 
  (h2 : f + 6 * n = 350) : 
  f = 60 := 
by
  sorry

end NUMINAMATH_GPT_flat_fee_l2027_202716


namespace NUMINAMATH_GPT_total_rent_of_pasture_l2027_202718

theorem total_rent_of_pasture 
  (oxen_A : ℕ) (months_A : ℕ) (oxen_B : ℕ) (months_B : ℕ)
  (oxen_C : ℕ) (months_C : ℕ) (share_C : ℕ) (total_rent : ℕ) :
  oxen_A = 10 →
  months_A = 7 →
  oxen_B = 12 →
  months_B = 5 →
  oxen_C = 15 →
  months_C = 3 →
  share_C = 72 →
  total_rent = 280 :=
by
  intros hA1 hA2 hB1 hB2 hC1 hC2 hC3
  sorry

end NUMINAMATH_GPT_total_rent_of_pasture_l2027_202718


namespace NUMINAMATH_GPT_solve_for_x_l2027_202700

theorem solve_for_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2027_202700


namespace NUMINAMATH_GPT_lottery_not_guaranteed_to_win_l2027_202765

theorem lottery_not_guaranteed_to_win (total_tickets : ℕ) (winning_rate : ℚ) (num_purchased : ℕ) :
  total_tickets = 100000 ∧ winning_rate = 1 / 1000 ∧ num_purchased = 2000 → 
  ∃ (outcome : ℕ), outcome = 0 := by
  sorry

end NUMINAMATH_GPT_lottery_not_guaranteed_to_win_l2027_202765


namespace NUMINAMATH_GPT_bombardment_deaths_l2027_202786

variable (initial_population final_population : ℕ)
variable (fear_factor death_percentage : ℝ)

theorem bombardment_deaths (h1 : initial_population = 4200)
                           (h2 : final_population = 3213)
                           (h3 : fear_factor = 0.15)
                           (h4 : ∃ x, death_percentage = x / 100 ∧ 
                                       4200 - (x / 100) * 4200 - fear_factor * (4200 - (x / 100) * 4200) = 3213) :
                           death_percentage = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_bombardment_deaths_l2027_202786


namespace NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l2027_202703

theorem parallel_vectors_m_eq_neg3 : 
  ∀ m : ℝ, (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (1 + m, 1 - m) → a.1 * b.2 - a.2 * b.1 = 0) → m = -3 :=
by
  intros m h_par
  specialize h_par (1, -2) (1 + m, 1 - m) rfl rfl
  -- We need to show m = -3
  sorry

end NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l2027_202703


namespace NUMINAMATH_GPT_factorize_polynomial_l2027_202755

theorem factorize_polynomial (c : ℝ) :
  (x : ℝ) → (x - 1) * (x - 3) = x^2 - 4 * x + c → c = 3 :=
by 
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l2027_202755


namespace NUMINAMATH_GPT_pure_imaginary_value_l2027_202793

theorem pure_imaginary_value (a : ℝ) : (z = (0 : ℝ) + (a^2 + 2 * a - 3) * I) → (a = 0 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_value_l2027_202793


namespace NUMINAMATH_GPT_intersection_points_x_axis_vertex_on_line_inequality_c_l2027_202799

section
variable {r : ℝ}
def quadratic_function (x m : ℝ) : ℝ := -0.5 * (x - 2*m)^2 + 3 - m

theorem intersection_points_x_axis (m : ℝ) (h : m = 2) : 
  ∃ x1 x2 : ℝ, quadratic_function x1 m = 0 ∧ quadratic_function x2 m = 0 ∧ x1 ≠ x2 :=
by
  sorry

theorem vertex_on_line (m : ℝ) (h : true) : 
  ∀ m : ℝ, (2*m, 3-m) ∈ {p : ℝ × ℝ | p.2 = -0.5 * p.1 + 3} :=
by
  sorry

theorem inequality_c (a c m : ℝ) (hP : quadratic_function (a+1) m = c) (hQ : quadratic_function ((4*m-5)+a) m = c) : 
  c ≤ 13/8 :=
by
  sorry
end

end NUMINAMATH_GPT_intersection_points_x_axis_vertex_on_line_inequality_c_l2027_202799


namespace NUMINAMATH_GPT_sum_of_triangulars_15_to_20_l2027_202767

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_triangulars_15_to_20 : 
  (triangular_number 15 + triangular_number 16 + triangular_number 17 + triangular_number 18 + triangular_number 19 + triangular_number 20) = 980 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_triangulars_15_to_20_l2027_202767


namespace NUMINAMATH_GPT_value_at_zero_eq_sixteen_l2027_202782

-- Define the polynomial P(x)
def P (x : ℚ) : ℚ := x ^ 4 - 20 * x ^ 2 + 16

-- Theorem stating the value of P(0)
theorem value_at_zero_eq_sixteen :
  P 0 = 16 :=
by
-- We know the polynomial P(x) is x^4 - 20x^2 + 16
-- When x = 0, P(0) = 0^4 - 20 * 0^2 + 16 = 16
sorry

end NUMINAMATH_GPT_value_at_zero_eq_sixteen_l2027_202782


namespace NUMINAMATH_GPT_find_truck_weight_l2027_202737

variable (T Tr : ℝ)

def weight_condition_1 : Prop := T + Tr = 7000
def weight_condition_2 : Prop := Tr = 0.5 * T - 200

theorem find_truck_weight (h1 : weight_condition_1 T Tr) 
                           (h2 : weight_condition_2 T Tr) : 
  T = 4800 :=
sorry

end NUMINAMATH_GPT_find_truck_weight_l2027_202737


namespace NUMINAMATH_GPT_marks_difference_is_140_l2027_202769

noncomputable def marks_difference (P C M : ℕ) : ℕ :=
  (P + C + M) - P

theorem marks_difference_is_140 (P C M : ℕ) (h1 : (C + M) / 2 = 70) :
  marks_difference P C M = 140 := by
  sorry

end NUMINAMATH_GPT_marks_difference_is_140_l2027_202769


namespace NUMINAMATH_GPT_colored_paper_distribution_l2027_202719

theorem colored_paper_distribution (F M : ℕ) (h1 : F + M = 24) (h2 : M = 2 * F) (total_sheets : ℕ) (distributed_sheets : total_sheets = 48) : 
  (48 / F) = 6 := by
  sorry

end NUMINAMATH_GPT_colored_paper_distribution_l2027_202719


namespace NUMINAMATH_GPT_max_value_f_l2027_202794

noncomputable def op_add (a b : ℝ) : ℝ :=
if a >= b then a else b^2

noncomputable def f (x : ℝ) : ℝ :=
(op_add 1 x) + (op_add 2 x)

theorem max_value_f :
  ∃ x ∈ Set.Icc (-2 : ℝ) 3, ∀ y ∈ Set.Icc (-2 : ℝ) 3, f y ≤ f x := 
sorry

end NUMINAMATH_GPT_max_value_f_l2027_202794


namespace NUMINAMATH_GPT_mother_hubbard_children_l2027_202791

theorem mother_hubbard_children :
  (∃ c : ℕ, (2 / 3 : ℚ) = c * (1 / 12 : ℚ)) → c = 8 :=
by
  sorry

end NUMINAMATH_GPT_mother_hubbard_children_l2027_202791


namespace NUMINAMATH_GPT_value_of_a_minus_b_l2027_202734

theorem value_of_a_minus_b 
  (a b : ℤ) 
  (x y : ℤ)
  (h1 : x = -2)
  (h2 : y = 1)
  (h3 : a * x + b * y = 1)
  (h4 : b * x + a * y = 7) : 
  a - b = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l2027_202734


namespace NUMINAMATH_GPT_part_one_part_two_l2027_202798

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem part_one:
  f a b (-1) = 0 → f a b x = x^2 + 2 * x + 1 :=
by
  sorry

theorem part_two:
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f 1 2 x > x + k) ↔ k < 1 :=
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l2027_202798


namespace NUMINAMATH_GPT_three_digit_number_is_112_l2027_202754

theorem three_digit_number_is_112 (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 1 ≤ c ∧ c ≤ 9) (h4 : 100 * a + 10 * b + c = 56 * c) :
  100 * a + 10 * b + c = 112 :=
by sorry

end NUMINAMATH_GPT_three_digit_number_is_112_l2027_202754


namespace NUMINAMATH_GPT_find_b_minus_a_l2027_202743

/-- Proof to find the value of b - a given the inequality conditions on x.
    The conditions are:
    1. x - a < 1
    2. x + b > 2
    3. 0 < x < 4
    We need to show that b - a = -1.
-/
theorem find_b_minus_a (a b x : ℝ) 
  (h1 : x - a < 1) 
  (h2 : x + b > 2) 
  (h3 : 0 < x) 
  (h4 : x < 4) 
  : b - a = -1 := 
sorry

end NUMINAMATH_GPT_find_b_minus_a_l2027_202743


namespace NUMINAMATH_GPT_Will_Had_28_Bottles_l2027_202721

-- Definitions based on conditions
-- Let days be the number of days water lasted (4 days)
def days : ℕ := 4

-- Let bottles_per_day be the number of bottles Will drank each day (7 bottles/day)
def bottles_per_day : ℕ := 7

-- Correct answer defined as total number of bottles (28 bottles)
def total_bottles : ℕ := 28

-- The proof statement to show that the total number of bottles is equal to 28
theorem Will_Had_28_Bottles :
  (bottles_per_day * days = total_bottles) :=
by
  sorry

end NUMINAMATH_GPT_Will_Had_28_Bottles_l2027_202721


namespace NUMINAMATH_GPT_intersection_point_l2027_202775

variable (x y : ℝ)

-- Definitions given by the conditions
def line1 (x y : ℝ) := 3 * y = -2 * x + 6
def line2 (x y : ℝ) := -2 * y = 6 * x + 4

-- The theorem we want to prove
theorem intersection_point : ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ x = -12/7 ∧ y = 22/7 := 
sorry

end NUMINAMATH_GPT_intersection_point_l2027_202775


namespace NUMINAMATH_GPT_system_of_equations_solution_l2027_202736

theorem system_of_equations_solution (x y z : ℝ) 
  (h : ∀ (n : ℕ), x * (1 - 1 / 2^(n : ℝ)) + y * (1 - 1 / 2^(n+1 : ℝ)) + z * (1 - 1 / 2^(n+2 : ℝ)) = 0) : 
  y = -3 * x ∧ z = 2 * x :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2027_202736


namespace NUMINAMATH_GPT_sam_age_l2027_202773

theorem sam_age (drew_current_age : ℕ) (drew_future_age : ℕ) (sam_future_age : ℕ) : 
  (drew_current_age = 12) → 
  (drew_future_age = drew_current_age + 5) → 
  (sam_future_age = 3 * drew_future_age) → 
  (sam_future_age - 5 = 46) := 
by sorry

end NUMINAMATH_GPT_sam_age_l2027_202773


namespace NUMINAMATH_GPT_remainder_of_x_div_9_is_8_l2027_202784

variable (x y r : ℕ)
variable (r_lt_9 : r < 9)
variable (h1 : x = 9 * y + r)
variable (h2 : 2 * x = 14 * y + 1)
variable (h3 : 5 * y - x = 3)

theorem remainder_of_x_div_9_is_8 : r = 8 := by
  sorry

end NUMINAMATH_GPT_remainder_of_x_div_9_is_8_l2027_202784


namespace NUMINAMATH_GPT_sequence_general_formula_l2027_202709

theorem sequence_general_formula (a : ℕ → ℕ) 
    (h₀ : a 1 = 3) 
    (h : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) : 
    ∀ n : ℕ, a n = 2^(n+1) - 1 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l2027_202709


namespace NUMINAMATH_GPT_alpha_plus_beta_l2027_202770

theorem alpha_plus_beta (α β : ℝ) 
  (hα : 0 < α ∧ α < Real.pi / 2) 
  (hβ : 0 < β ∧ β < Real.pi / 2)
  (h_sin_alpha : Real.sin α = Real.sqrt 10 / 10)
  (h_cos_beta : Real.cos β = 2 * Real.sqrt 5 / 5) :
  α + β = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_alpha_plus_beta_l2027_202770


namespace NUMINAMATH_GPT_proof_problem_l2027_202778

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ -1}

theorem proof_problem :
  ((A ∩ {x | x > -1}) ∪ (B ∩ {x | x ≤ 0})) = {x | x > 0 ∨ x ≤ -1} :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l2027_202778


namespace NUMINAMATH_GPT_coconut_grove_l2027_202763

theorem coconut_grove (x Y : ℕ) (h1 : 3 * x ≠ 0) (h2 : (x+3) * 60 + x * Y + (x-3) * 180 = 3 * x * 100) (hx : x = 6) : Y = 120 :=
by 
  sorry

end NUMINAMATH_GPT_coconut_grove_l2027_202763


namespace NUMINAMATH_GPT_solve_for_y_l2027_202738

theorem solve_for_y (y : ℕ) (h : 9^y = 3^14) : y = 7 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2027_202738


namespace NUMINAMATH_GPT_max_value_of_expression_l2027_202728

noncomputable def max_expression_value (a b c : ℝ) : ℝ :=
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2)))

theorem max_value_of_expression (a b c : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) (hc : -1 < c ∧ c < 1) :
  max_expression_value a b c ≤ 2 :=
by sorry

end NUMINAMATH_GPT_max_value_of_expression_l2027_202728


namespace NUMINAMATH_GPT_ab_is_4_l2027_202776

noncomputable def ab_value (a b : ℝ) : ℝ :=
  8 / (0.5 * (8 / a) * (8 / b))

theorem ab_is_4 (a b : ℝ) (ha : a > 0) (hb : b > 0) (area_condition : ab_value a b = 8) : a * b = 4 :=
  by
  sorry

end NUMINAMATH_GPT_ab_is_4_l2027_202776


namespace NUMINAMATH_GPT_task_completion_choice_l2027_202795

theorem task_completion_choice (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 := by
  sorry

end NUMINAMATH_GPT_task_completion_choice_l2027_202795


namespace NUMINAMATH_GPT_minimize_expression_pos_int_l2027_202739

theorem minimize_expression_pos_int (n : ℕ) (hn : 0 < n) : 
  (∀ m : ℕ, 0 < m → (m / 3 + 27 / m : ℝ) ≥ (9 / 3 + 27 / 9)) :=
sorry

end NUMINAMATH_GPT_minimize_expression_pos_int_l2027_202739


namespace NUMINAMATH_GPT_parabola_directrix_l2027_202758

theorem parabola_directrix (x y : ℝ) (h_eqn : y = -3 * x^2 + 6 * x - 5) :
  y = -23 / 12 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l2027_202758


namespace NUMINAMATH_GPT_part_one_part_two_l2027_202779
-- Import the Mathlib library for necessary definitions and theorems.

-- Define the conditions as hypotheses.
variables {a b c : ℝ} (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (1): State the inequality involving sums of reciprocals.
theorem part_one : (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 3 / 2 := 
by
  sorry

-- Part (2): Define the range for m in terms of the inequality condition.
theorem part_two : ∃m: ℝ, (∀a b c : ℝ, a + b + c = 3 → 0 < a → 0 < b → 0 < c → (-x^2 + m*x + 2 ≤ a^2 + b^2 + c^2)) ↔ (-2 ≤ m) ∧ (m ≤ 2) :=
by 
  sorry

end NUMINAMATH_GPT_part_one_part_two_l2027_202779


namespace NUMINAMATH_GPT_find_a9_l2027_202761

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- conditions
def is_arithmetic_sequence := ∀ n : ℕ, a (n + 1) = a n + d
def given_condition1 := a 5 + a 7 = 16
def given_condition2 := a 3 = 4

-- theorem
theorem find_a9 (h1 : is_arithmetic_sequence a d) (h2 : given_condition1 a) (h3 : given_condition2 a) :
  a 9 = 12 :=
sorry

end NUMINAMATH_GPT_find_a9_l2027_202761


namespace NUMINAMATH_GPT_initial_cars_l2027_202745

theorem initial_cars (X : ℕ) : (X - 13 + (13 + 5) = 85) → (X = 80) :=
by
  sorry

end NUMINAMATH_GPT_initial_cars_l2027_202745


namespace NUMINAMATH_GPT_tammy_trees_l2027_202751

-- Define the conditions as Lean definitions and the final statement to prove
theorem tammy_trees :
  (∀ (days : ℕ) (earnings : ℕ) (pricePerPack : ℕ) (orangesPerPack : ℕ) (orangesPerTree : ℕ),
    days = 21 →
    earnings = 840 →
    pricePerPack = 2 →
    orangesPerPack = 6 →
    orangesPerTree = 12 →
    (earnings / days) / (pricePerPack / orangesPerPack) / orangesPerTree = 10) :=
by
  intros days earnings pricePerPack orangesPerPack orangesPerTree
  sorry

end NUMINAMATH_GPT_tammy_trees_l2027_202751


namespace NUMINAMATH_GPT_part1_part2_l2027_202774

noncomputable def f (x : Real) : Real :=
  2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x) - 1

noncomputable def h (x t : Real) : Real :=
  f (x + t)

theorem part1 (t : Real) (ht : 0 < t ∧ t < Real.pi / 2) :
  (h (-Real.pi / 6) t = 0) → t = Real.pi / 3 :=
sorry

theorem part2 (A B C : Real) (hA : 0 < A ∧ A < Real.pi / 2) (hA1 : h A (Real.pi / 3) = 1) :
  1 < ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ∧
  ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ≤ 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2027_202774


namespace NUMINAMATH_GPT_problem_solution_l2027_202733

theorem problem_solution (N : ℚ) (h : (4/5) * (3/8) * N = 24) : 2.5 * N = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_solution_l2027_202733


namespace NUMINAMATH_GPT_probability_first_prize_l2027_202756

-- Define the total number of tickets
def total_tickets : ℕ := 150

-- Define the number of first prizes
def first_prizes : ℕ := 5

-- Define the probability calculation as a theorem
theorem probability_first_prize : (first_prizes : ℚ) / total_tickets = 1 / 30 := 
by sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_probability_first_prize_l2027_202756


namespace NUMINAMATH_GPT_no_common_root_l2027_202727

variables {R : Type*} [OrderedRing R]

def f (x m n : R) := x^2 + m*x + n
def p (x k l : R) := x^2 + k*x + l

theorem no_common_root (k m n l : R) (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬ ∃ x : R, (f x m n = 0 ∧ p x k l = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_common_root_l2027_202727


namespace NUMINAMATH_GPT_find_a_2b_l2027_202723

theorem find_a_2b 
  (a b : ℤ) 
  (h1 : a * b = -150) 
  (h2 : a + b = -23) : 
  a + 2 * b = -55 :=
sorry

end NUMINAMATH_GPT_find_a_2b_l2027_202723


namespace NUMINAMATH_GPT_number_of_pups_in_second_round_l2027_202747

-- Define the conditions
variable (initialMice : Nat := 8)
variable (firstRoundPupsPerMouse : Nat := 6)
variable (secondRoundEatenPupsPerMouse : Nat := 2)
variable (finalMice : Nat := 280)

-- Define the proof problem
theorem number_of_pups_in_second_round (P : Nat) :
  initialMice + initialMice * firstRoundPupsPerMouse = 56 → 
  56 + 56 * P - 56 * secondRoundEatenPupsPerMouse = finalMice →
  P = 6 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_number_of_pups_in_second_round_l2027_202747


namespace NUMINAMATH_GPT_rope_cut_probability_l2027_202789

theorem rope_cut_probability (L : ℝ) (cut_position : ℝ) (P : ℝ) :
  L = 4 → (∀ cut_position, 0 ≤ cut_position ∧ cut_position ≤ L →
  (cut_position ≥ 1.5 ∧ (L - cut_position) ≥ 1.5)) → P = 1 / 4 :=
by
  intros hL hcut
  sorry

end NUMINAMATH_GPT_rope_cut_probability_l2027_202789


namespace NUMINAMATH_GPT_central_angle_is_2_radians_l2027_202777

namespace CircleAngle

def radius : ℝ := 2
def arc_length : ℝ := 4

theorem central_angle_is_2_radians : arc_length / radius = 2 := by
  sorry

end CircleAngle

end NUMINAMATH_GPT_central_angle_is_2_radians_l2027_202777


namespace NUMINAMATH_GPT_initial_people_in_gym_l2027_202788

variable (W A : ℕ)

theorem initial_people_in_gym (W A : ℕ) (h : W + A + 5 + 2 - 3 - 4 + 2 = 20) : W + A = 18 := by
  sorry

end NUMINAMATH_GPT_initial_people_in_gym_l2027_202788


namespace NUMINAMATH_GPT_cryptarithm_solution_l2027_202797

theorem cryptarithm_solution :
  ∃ A B C D E F G H J : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ J ∧
  H ≠ J ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ J < 10 ∧
  (10 * A + B) * (10 * C + A) = 100 * D + 10 * E + B ∧
  (10 * F + C) - (10 * D + G) = D ∧
  (10 * E + G) + (10 * H + J) = 100 * A + 10 * A + G ∧
  A = 1 ∧ B = 7 ∧ C = 2 ∧ D = 3 ∧ E = 5 ∧ F = 4 ∧ G = 9 ∧ H = 6 ∧ J = 0 :=
by
  sorry

end NUMINAMATH_GPT_cryptarithm_solution_l2027_202797


namespace NUMINAMATH_GPT_range_m_l2027_202760

variable {x m : ℝ}

theorem range_m (h1 : m / (1 - x) - 2 / (x - 1) = 1) (h2 : x ≥ 0) (h3 : x ≠ 1) : m ≤ -1 ∧ m ≠ -2 := 
sorry

end NUMINAMATH_GPT_range_m_l2027_202760


namespace NUMINAMATH_GPT_triangle_perimeter_l2027_202715

def triangle_side_lengths : ℕ × ℕ × ℕ := (10, 6, 7)

def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter (a b c : ℕ) (h : (a, b, c) = triangle_side_lengths) : 
  perimeter a b c = 23 := by
  -- We formulate the statement and leave the proof for later
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2027_202715


namespace NUMINAMATH_GPT_noemi_initial_amount_l2027_202724

theorem noemi_initial_amount : 
  ∀ (rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount : ℕ), 
    rouletteLoss = 600 → 
    blackjackLoss = 800 → 
    pokerLoss = 400 → 
    baccaratLoss = 700 → 
    remainingAmount = 1500 → 
    initialAmount = rouletteLoss + blackjackLoss + pokerLoss + baccaratLoss + remainingAmount →
    initialAmount = 4000 :=
by
  intros rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  exact h6

end NUMINAMATH_GPT_noemi_initial_amount_l2027_202724


namespace NUMINAMATH_GPT_dice_roll_probability_is_correct_l2027_202713

/-- Define the probability calculation based on conditions of the problem. --/
def dice_rolls_probability_diff_by_two (successful_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

/-- Given the problem conditions, there are 8 successful outcomes and 36 total outcomes. --/
theorem dice_roll_probability_is_correct :
  dice_rolls_probability_diff_by_two 8 36 = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_dice_roll_probability_is_correct_l2027_202713


namespace NUMINAMATH_GPT_perimeter_of_square_l2027_202704

theorem perimeter_of_square (a : ℤ) (h : a * a = 36) : 4 * a = 24 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l2027_202704


namespace NUMINAMATH_GPT_find_number_l2027_202759

theorem find_number (x : ℤ) (h : 3 * x + 4 = 19) : x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l2027_202759


namespace NUMINAMATH_GPT_find_FC_l2027_202742

theorem find_FC 
(DC CB AD ED FC : ℝ)
(h1 : DC = 7) 
(h2 : CB = 8) 
(h3 : AB = (1 / 4) * AD)
(h4 : ED = (4 / 5) * AD) : 
FC = 10.4 :=
sorry

end NUMINAMATH_GPT_find_FC_l2027_202742


namespace NUMINAMATH_GPT_decreasing_power_function_on_interval_l2027_202796

noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem decreasing_power_function_on_interval (m : ℝ) :
  (∀ x : ℝ, (0 < x) -> power_function m x < 0) ↔ m = -1 := 
by 
  sorry

end NUMINAMATH_GPT_decreasing_power_function_on_interval_l2027_202796


namespace NUMINAMATH_GPT_no_integer_solution_l2027_202705

theorem no_integer_solution (y : ℤ) : ¬ (-3 * y ≥ y + 9 ∧ 2 * y ≥ 14 ∧ -4 * y ≥ 2 * y + 21) :=
sorry

end NUMINAMATH_GPT_no_integer_solution_l2027_202705


namespace NUMINAMATH_GPT_tree_height_l2027_202740

theorem tree_height (BR MH MB MR TB : ℝ)
  (h_cond1 : BR = 5)
  (h_cond2 : MH = 1.8)
  (h_cond3 : MB = 1)
  (h_cond4 : MR = BR - MB)
  (h_sim : TB / BR = MH / MR)
  : TB = 2.25 :=
by sorry

end NUMINAMATH_GPT_tree_height_l2027_202740


namespace NUMINAMATH_GPT_complement_angle_l2027_202706

theorem complement_angle (A : ℝ) (hA : A = 35) : 90 - A = 55 := by
  sorry

end NUMINAMATH_GPT_complement_angle_l2027_202706


namespace NUMINAMATH_GPT_original_price_second_store_l2027_202764

-- Definitions of the conditions
def price_first_store : ℝ := 950
def discount_first_store : ℝ := 0.06
def discount_second_store : ℝ := 0.05
def price_difference : ℝ := 19

-- Define the discounted price function
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- State the main theorem
theorem original_price_second_store :
  ∃ P : ℝ, 
    (discounted_price price_first_store discount_first_store - discounted_price P discount_second_store = price_difference) ∧ 
    P = 960 :=
by
  sorry

end NUMINAMATH_GPT_original_price_second_store_l2027_202764


namespace NUMINAMATH_GPT_expenses_notation_l2027_202717

theorem expenses_notation (income expense : ℤ) (h_income : income = 6) (h_expense : -expense = income) : expense = -4 := 
by
  sorry

end NUMINAMATH_GPT_expenses_notation_l2027_202717


namespace NUMINAMATH_GPT_count_primes_1021_eq_one_l2027_202771

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_primes_1021_eq_one :
  (∃ n : ℕ, 3 ≤ n ∧ is_prime (n^3 + 2*n + 1) ∧
  ∀ m : ℕ, (3 ≤ m ∧ m ≠ n) → ¬ is_prime (m^3 + 2*m + 1)) :=
sorry

end NUMINAMATH_GPT_count_primes_1021_eq_one_l2027_202771


namespace NUMINAMATH_GPT_ratio_Florence_Rene_l2027_202780

theorem ratio_Florence_Rene :
  ∀ (I F R : ℕ), R = 300 → F = k * R → I = 1/3 * (F + R + I) → F + R + I = 1650 → F / R = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_Florence_Rene_l2027_202780


namespace NUMINAMATH_GPT_garden_area_increase_l2027_202708

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end NUMINAMATH_GPT_garden_area_increase_l2027_202708


namespace NUMINAMATH_GPT_evaluate_polynomial_at_2_l2027_202748

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 1) = 31 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_2_l2027_202748


namespace NUMINAMATH_GPT_gcd_min_value_l2027_202712

theorem gcd_min_value {a b c : ℕ} (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (gcd_ab : Nat.gcd a b = 210) (gcd_ac : Nat.gcd a c = 770) : Nat.gcd b c = 10 :=
sorry

end NUMINAMATH_GPT_gcd_min_value_l2027_202712


namespace NUMINAMATH_GPT_find_length_AX_l2027_202768

theorem find_length_AX 
  (A B C X : Type)
  (BC BX AC : ℝ)
  (h_BC : BC = 36)
  (h_BX : BX = 30)
  (h_AC : AC = 27)
  (h_bisector : ∃ (x : ℝ), x = BX / BC ∧ x = AX / AC ) :
  ∃ AX : ℝ, AX = 22.5 := 
sorry

end NUMINAMATH_GPT_find_length_AX_l2027_202768


namespace NUMINAMATH_GPT_f1_neither_even_nor_odd_f2_min_value_l2027_202792

noncomputable def f1 (x : ℝ) : ℝ :=
  x^2 + abs (x - 2) - 1

theorem f1_neither_even_nor_odd : ¬(∀ x : ℝ, f1 x = f1 (-x)) ∧ ¬(∀ x : ℝ, f1 x = -f1 (-x)) :=
sorry

noncomputable def f2 (x a : ℝ) : ℝ :=
  x^2 + abs (x - a) + 1

theorem f2_min_value (a : ℝ) :
  (if a < -1/2 then (∃ x, f2 x a = 3/4 - a)
  else if -1/2 ≤ a ∧ a ≤ 1/2 then (∃ x, f2 x a = a^2 + 1)
  else (∃ x, f2 x a = 3/4 + a)) :=
sorry

end NUMINAMATH_GPT_f1_neither_even_nor_odd_f2_min_value_l2027_202792


namespace NUMINAMATH_GPT_max_value_frac_inv_sum_l2027_202753

theorem max_value_frac_inv_sum (x y : ℝ) (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b)
  (h3 : a^x = 6) (h4 : b^y = 6) (h5 : a + b = 2 * Real.sqrt 6) :
  ∃ m, m = 1 ∧ (∀ x y a b, (1 < a) → (1 < b) → (a^x = 6) → (b^y = 6) → (a + b = 2 * Real.sqrt 6) → 
  (∃ n, (n = (1/x + 1/y)) → n ≤ m)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_frac_inv_sum_l2027_202753


namespace NUMINAMATH_GPT_sale_in_third_month_l2027_202741

def grocer_sales (s1 s2 s4 s5 s6 : ℕ) (average : ℕ) (num_months : ℕ) (total_sales : ℕ) : Prop :=
  s1 = 5266 ∧ s2 = 5768 ∧ s4 = 5678 ∧ s5 = 6029 ∧ s6 = 4937 ∧ average = 5600 ∧ num_months = 6 ∧ total_sales = average * num_months

theorem sale_in_third_month
  (s1 s2 s4 s5 s6 total_sales : ℕ)
  (h : grocer_sales s1 s2 s4 s5 s6 5600 6 total_sales) :
  ∃ s3 : ℕ, total_sales - (s1 + s2 + s4 + s5 + s6) = s3 ∧ s3 = 5922 := 
by {
  sorry
}

end NUMINAMATH_GPT_sale_in_third_month_l2027_202741


namespace NUMINAMATH_GPT_find_min_n_l2027_202729

theorem find_min_n (n k : ℕ) (h : 14 * n = k^2) : n = 14 := sorry

end NUMINAMATH_GPT_find_min_n_l2027_202729


namespace NUMINAMATH_GPT_given_polynomial_l2027_202787

noncomputable def f (x : ℝ) := x^3 - 2

theorem given_polynomial (x : ℝ) : 
  8 * f (x^3) - x^6 * f (2 * x) - 2 * f (x^2) + 12 = 0 :=
by
  sorry

end NUMINAMATH_GPT_given_polynomial_l2027_202787


namespace NUMINAMATH_GPT_c_ge_one_l2027_202785

theorem c_ge_one (a b : ℕ) (c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (a + 1) / (b + c) = b / a) : c ≥ 1 := 
sorry

end NUMINAMATH_GPT_c_ge_one_l2027_202785


namespace NUMINAMATH_GPT_dogwood_tree_count_l2027_202720

def initial_dogwoods : ℕ := 34
def additional_dogwoods : ℕ := 49
def total_dogwoods : ℕ := initial_dogwoods + additional_dogwoods

theorem dogwood_tree_count :
  total_dogwoods = 83 :=
by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_dogwood_tree_count_l2027_202720


namespace NUMINAMATH_GPT_intermediate_circle_radius_l2027_202752

theorem intermediate_circle_radius (r1 r3: ℝ) (h1: r1 = 5) (h2: r3 = 13) 
  (h3: π * r1 ^ 2 = π * r3 ^ 2 - π * r2 ^ 2) : r2 = 12 := sorry


end NUMINAMATH_GPT_intermediate_circle_radius_l2027_202752


namespace NUMINAMATH_GPT_percentage_returned_l2027_202766

theorem percentage_returned (R : ℕ) (S : ℕ) (total : ℕ) (least_on_lot : ℕ) (max_rented : ℕ)
  (h1 : total = 20) (h2 : least_on_lot = 10) (h3 : max_rented = 20) (h4 : R = 20) (h5 : S ≥ 10) :
  (S / R) * 100 ≥ 50 := sorry

end NUMINAMATH_GPT_percentage_returned_l2027_202766


namespace NUMINAMATH_GPT_simplify_fraction_l2027_202730

theorem simplify_fraction (h1 : 222 = 2 * 3 * 37) (h2 : 8888 = 8 * 11 * 101) :
  (222 / 8888) * 22 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2027_202730


namespace NUMINAMATH_GPT_opposite_of_neg_two_l2027_202790

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_l2027_202790


namespace NUMINAMATH_GPT_twice_x_minus_3_l2027_202714

theorem twice_x_minus_3 (x : ℝ) : (2 * x) - 3 = 2 * x - 3 := 
by 
  -- This proof is trivial and we can assert equality directly
  sorry

end NUMINAMATH_GPT_twice_x_minus_3_l2027_202714


namespace NUMINAMATH_GPT_workshop_total_workers_l2027_202750

theorem workshop_total_workers
  (avg_salary_per_head : ℕ)
  (num_technicians num_managers num_apprentices total_workers : ℕ)
  (avg_tech_salary avg_mgr_salary avg_appr_salary : ℕ) 
  (h1 : avg_salary_per_head = 700)
  (h2 : num_technicians = 5)
  (h3 : num_managers = 3)
  (h4 : avg_tech_salary = 800)
  (h5 : avg_mgr_salary = 1200)
  (h6 : avg_appr_salary = 650)
  (h7 : total_workers = num_technicians + num_managers + num_apprentices)
  : total_workers = 48 := 
sorry

end NUMINAMATH_GPT_workshop_total_workers_l2027_202750


namespace NUMINAMATH_GPT_men_entered_count_l2027_202772

variable (M W x : ℕ)

noncomputable def initial_ratio : Prop := M = 4 * W / 5
noncomputable def men_entered : Prop := M + x = 14
noncomputable def women_double : Prop := 2 * (W - 3) = 14

theorem men_entered_count (M W x : ℕ) (h1 : initial_ratio M W) (h2 : men_entered M x) (h3 : women_double W) : x = 6 := by
  sorry

end NUMINAMATH_GPT_men_entered_count_l2027_202772


namespace NUMINAMATH_GPT_angle_BC₁_plane_BBD₁D_l2027_202744

-- Define all the necessary components of the cube and its geometry
variables {A B C D A₁ B₁ C₁ D₁ : ℝ} -- placeholders for points, represented by real coordinates

def is_cube (A B C D A₁ B₁ C₁ D₁ : ℝ) : Prop := sorry -- Define the cube property (this would need a proper definition)

def space_diagonal (B C₁ : ℝ) : Prop := sorry -- Define the property of being a space diagonal

def plane (B B₁ D₁ D : ℝ) : Prop := sorry -- Define a plane through these points (again needs a definition)

-- Define the angle between a line and a plane
def angle_between_line_and_plane (BC₁ B B₁ D₁ D : ℝ) : ℝ := sorry -- Define angle calculation (requires more context)

-- The proof statement, which is currently not proven (contains 'sorry')
theorem angle_BC₁_plane_BBD₁D (s : ℝ):
  is_cube A B C D A₁ B₁ C₁ D₁ →
  space_diagonal B C₁ →
  plane B B₁ D₁ D →
  angle_between_line_and_plane B C₁ B₁ D₁ D = π / 6 :=
sorry

end NUMINAMATH_GPT_angle_BC₁_plane_BBD₁D_l2027_202744
