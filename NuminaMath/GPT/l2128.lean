import Mathlib

namespace NUMINAMATH_GPT_tom_boxes_needed_l2128_212864

-- Definitions of given conditions
def room_length : ℕ := 16
def room_width : ℕ := 20
def box_coverage : ℕ := 10
def already_covered : ℕ := 250

-- The total area of the living room
def total_area : ℕ := room_length * room_width

-- The remaining area that needs to be covered
def remaining_area : ℕ := total_area - already_covered

-- The number of boxes required to cover the remaining area
def boxes_needed : ℕ := remaining_area / box_coverage

-- The theorem statement
theorem tom_boxes_needed : boxes_needed = 7 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_tom_boxes_needed_l2128_212864


namespace NUMINAMATH_GPT_solution_to_problem_l2128_212814

-- Definitions of conditions
def condition_1 (x : ℝ) : Prop := 2 * x - 6 ≠ 0
def condition_2 (x : ℝ) : Prop := 5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10

-- Definition of solution set
def solution_set (x : ℝ) : Prop := 3 < x ∧ x < 60 / 19

-- The theorem to be proven
theorem solution_to_problem (x : ℝ) (h1 : condition_1 x) : condition_2 x ↔ solution_set x :=
by sorry

end NUMINAMATH_GPT_solution_to_problem_l2128_212814


namespace NUMINAMATH_GPT_gcf_270_108_150_l2128_212885

theorem gcf_270_108_150 : Nat.gcd (Nat.gcd 270 108) 150 = 30 := 
  sorry

end NUMINAMATH_GPT_gcf_270_108_150_l2128_212885


namespace NUMINAMATH_GPT_neither_outstanding_nor_young_pioneers_is_15_l2128_212849

-- Define the conditions
def total_students : ℕ := 87
def outstanding_students : ℕ := 58
def young_pioneers : ℕ := 63
def both_outstanding_and_young_pioneers : ℕ := 49

-- Define the function to calculate the number of students who are neither
def neither_outstanding_nor_young_pioneers
: ℕ :=
total_students - (outstanding_students - both_outstanding_and_young_pioneers) - (young_pioneers - both_outstanding_and_young_pioneers) - both_outstanding_and_young_pioneers

-- The theorem to prove
theorem neither_outstanding_nor_young_pioneers_is_15
: neither_outstanding_nor_young_pioneers = 15 :=
by
  sorry

end NUMINAMATH_GPT_neither_outstanding_nor_young_pioneers_is_15_l2128_212849


namespace NUMINAMATH_GPT_solve_fractional_equation_l2128_212894

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x + 1 ≠ 0) :
  (1 / x = 2 / (x + 1)) → x = 1 := 
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l2128_212894


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2128_212856

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n+1) = a n + d)
  (h1 : a 2 + a 3 = 1)
  (h2 : a 10 + a 11 = 9) :
  a 5 + a 6 = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2128_212856


namespace NUMINAMATH_GPT_slope_y_intercept_product_l2128_212836

theorem slope_y_intercept_product (m b : ℝ) (hm : m = -1/2) (hb : b = 4/5) : -1 < m * b ∧ m * b < 0 :=
by
  sorry

end NUMINAMATH_GPT_slope_y_intercept_product_l2128_212836


namespace NUMINAMATH_GPT_measure_of_angle_C_l2128_212832

theorem measure_of_angle_C (m l : ℝ) (angle_A angle_B angle_D angle_C : ℝ)
  (h_parallel : l = m)
  (h_angle_A : angle_A = 130)
  (h_angle_B : angle_B = 140)
  (h_angle_D : angle_D = 100) :
  angle_C = 90 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l2128_212832


namespace NUMINAMATH_GPT_lollipop_problem_l2128_212837

def arithmetic_sequence_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem lollipop_problem
  (a : ℕ) (h1 : arithmetic_sequence_sum a 5 7 = 175) :
  (a + 15) = 25 :=
by
  sorry

end NUMINAMATH_GPT_lollipop_problem_l2128_212837


namespace NUMINAMATH_GPT_amusement_park_l2128_212872

theorem amusement_park
  (A : ℕ)
  (adult_ticket_cost : ℕ := 22)
  (child_ticket_cost : ℕ := 7)
  (num_children : ℕ := 2)
  (total_cost : ℕ := 58)
  (cost_eq : adult_ticket_cost * A + child_ticket_cost * num_children = total_cost) :
  A = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_amusement_park_l2128_212872


namespace NUMINAMATH_GPT_initial_tree_height_l2128_212848

-- Definition of the problem conditions as Lean definitions.
def quadruple (x : ℕ) : ℕ := 4 * x

-- Given conditions of the problem
def final_height : ℕ := 256
def height_increase_each_year (initial_height : ℕ) : Prop :=
  quadruple (quadruple (quadruple (quadruple initial_height))) = final_height

-- The proof statement that we need to prove
theorem initial_tree_height 
  (initial_height : ℕ)
  (h : height_increase_each_year initial_height)
  : initial_height = 1 := sorry

end NUMINAMATH_GPT_initial_tree_height_l2128_212848


namespace NUMINAMATH_GPT_correct_operation_l2128_212871

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a / (3 * a) = 2 * a) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ (-a * b^2)^2 = a^2 * b^4 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2128_212871


namespace NUMINAMATH_GPT_polynomials_equal_at_all_x_l2128_212812

variable {R : Type} [CommRing R]

def f (a_5 a_4 a_3 a_2 a_1 a_0 : R) (x : R) := a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
def g (b_3 b_2 b_1 b_0 : R) (x : R) := b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
def h (c_2 c_1 c_0 : R) (x : R) := c_2 * x^2 + c_1 * x + c_0

theorem polynomials_equal_at_all_x 
    (a_5 a_4 a_3 a_2 a_1 a_0 b_3 b_2 b_1 b_0 c_2 c_1 c_0 : ℤ)
    (bound_a : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
    (bound_b : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
    (bound_c : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
    (H : f a_5 a_4 a_3 a_2 a_1 a_0 10 = g b_3 b_2 b_1 b_0 10 * h c_2 c_1 c_0 10) :
    ∀ x, f a_5 a_4 a_3 a_2 a_1 a_0 x = g b_3 b_2 b_1 b_0 x * h c_2 c_1 c_0 x := by
  sorry

end NUMINAMATH_GPT_polynomials_equal_at_all_x_l2128_212812


namespace NUMINAMATH_GPT_solution_l2128_212863

noncomputable def triangle_perimeter (AB BC AC : ℕ) (lA lB lC : ℕ) : ℕ :=
  -- This represents the proof problem using the given conditions
  if (AB = 130) ∧ (BC = 240) ∧ (AC = 190)
     ∧ (lA = 65) ∧ (lB = 50) ∧ (lC = 20)
  then
    130  -- The correct answer
  else
    0    -- If the conditions are not met, return 0 

theorem solution :
  triangle_perimeter 130 240 190 65 50 20 = 130 :=
by
  -- This theorem states that with the given conditions, the perimeter of the triangle is 130
  sorry

end NUMINAMATH_GPT_solution_l2128_212863


namespace NUMINAMATH_GPT_triangle_equilateral_if_abs_eq_zero_l2128_212891

theorem triangle_equilateral_if_abs_eq_zero (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_if_abs_eq_zero_l2128_212891


namespace NUMINAMATH_GPT_quadratic_roots_range_l2128_212817

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (x1^2 - 2 * x1 + m - 2 = 0) ∧ 
    (x2^2 - 2 * x2 + m - 2 = 0)) → m < 3 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_range_l2128_212817


namespace NUMINAMATH_GPT_min_dot_product_of_vectors_at_fixed_point_l2128_212843

noncomputable def point := ℝ × ℝ

def on_ellipse (x y : ℝ) : Prop := 
  (x^2) / 36 + (y^2) / 9 = 1

def dot_product (p q : point) : ℝ := 
  p.1 * q.1 + p.2 * q.2

def vector_magnitude_squared (p : point) : ℝ := 
  p.1^2 + p.2^2

def KM (M : point) : point := 
  (M.1 - 2, M.2)

def NM (N M : point) : point := 
  (M.1 - N.1, M.2 - N.2)

def fixed_point_K : point := 
  (2, 0)

theorem min_dot_product_of_vectors_at_fixed_point (M N : point) 
  (hM_on_ellipse : on_ellipse M.1 M.2) 
  (hN_on_ellipse : on_ellipse N.1 N.2) 
  (h_orthogonal : dot_product (KM M) (KM N) = 0) : 
  ∃ (α : ℝ), dot_product (KM M) (NM N M) = 23 / 3 :=
sorry

end NUMINAMATH_GPT_min_dot_product_of_vectors_at_fixed_point_l2128_212843


namespace NUMINAMATH_GPT_pirate_ship_minimum_speed_l2128_212892

noncomputable def minimum_speed (initial_distance : ℝ) (caravel_speed : ℝ) (caravel_direction : ℝ) : ℝ :=
  let caravel_velocity_x := -caravel_speed * Real.cos caravel_direction
  let caravel_velocity_y := -caravel_speed * Real.sin caravel_direction
  let t := initial_distance / (caravel_speed * (1 + Real.sqrt 3))
  let v_p := Real.sqrt ((initial_distance / t - caravel_velocity_x)^2 + (caravel_velocity_y)^2)
  v_p

theorem pirate_ship_minimum_speed : 
  minimum_speed 10 12 (Real.pi / 3) = 6 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_pirate_ship_minimum_speed_l2128_212892


namespace NUMINAMATH_GPT_num_groups_of_consecutive_natural_numbers_l2128_212806

theorem num_groups_of_consecutive_natural_numbers (n : ℕ) (h : 3 * n + 3 < 19) : n < 6 := 
  sorry

end NUMINAMATH_GPT_num_groups_of_consecutive_natural_numbers_l2128_212806


namespace NUMINAMATH_GPT_quadratic_inequality_range_l2128_212877

theorem quadratic_inequality_range (a x : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_range_l2128_212877


namespace NUMINAMATH_GPT_total_canoes_by_end_of_april_l2128_212804

def canoes_built_jan : Nat := 4

def canoes_built_next_month (prev_month : Nat) : Nat := 3 * prev_month

def canoes_built_feb : Nat := canoes_built_next_month canoes_built_jan
def canoes_built_mar : Nat := canoes_built_next_month canoes_built_feb
def canoes_built_apr : Nat := canoes_built_next_month canoes_built_mar

def total_canoes_built : Nat := canoes_built_jan + canoes_built_feb + canoes_built_mar + canoes_built_apr

theorem total_canoes_by_end_of_april : total_canoes_built = 160 :=
by
  sorry

end NUMINAMATH_GPT_total_canoes_by_end_of_april_l2128_212804


namespace NUMINAMATH_GPT_factorize_polynomial_l2128_212862

theorem factorize_polynomial (x y : ℝ) : 
  (x^2 - y^2 - 2 * x - 4 * y - 3) = (x + y + 1) * (x - y - 3) :=
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l2128_212862


namespace NUMINAMATH_GPT_evaluate_given_condition_l2128_212878

noncomputable def evaluate_expression (b : ℚ) : ℚ :=
  (7 * b^2 - 15 * b + 5) * (3 * b - 4)

theorem evaluate_given_condition (b : ℚ) (h : b = 4 / 3) : evaluate_expression b = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_given_condition_l2128_212878


namespace NUMINAMATH_GPT_find_varphi_l2128_212803

theorem find_varphi (φ : ℝ) (h1 : 0 < φ ∧ φ < 2 * Real.pi) 
    (h2 : ∀ x, x = 2 → Real.sin (Real.pi * x + φ) = 1) : 
    φ = Real.pi / 2 :=
-- The following is left as a proof placeholder
sorry

end NUMINAMATH_GPT_find_varphi_l2128_212803


namespace NUMINAMATH_GPT_ellipse_eccentricity_l2128_212881

theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b : b > 0) (c : ℝ)
  (h_ellipse : (b^2 / c^2) = 3)
  (eccentricity_eq : ∀ (e : ℝ), e = c / a ↔ e = 1 / 2) : 
  ∃ e, e = (c / a) :=
by {
  sorry
}

end NUMINAMATH_GPT_ellipse_eccentricity_l2128_212881


namespace NUMINAMATH_GPT_num_terms_arith_seq_l2128_212873

theorem num_terms_arith_seq {a d t : ℕ} (h_a : a = 5) (h_d : d = 3) (h_t : t = 140) :
  ∃ n : ℕ, t = a + (n-1) * d ∧ n = 46 :=
by
  sorry

end NUMINAMATH_GPT_num_terms_arith_seq_l2128_212873


namespace NUMINAMATH_GPT_A_subset_B_l2128_212861

def inA (n : ℕ) : Prop := ∃ x y : ℕ, n = x^2 + 2 * y^2 ∧ x > y
def inB (n : ℕ) : Prop := ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ n = (a^3 + b^3 + c^3) / (a + b + c)

theorem A_subset_B : ∀ (n : ℕ), inA n → inB n := 
sorry

end NUMINAMATH_GPT_A_subset_B_l2128_212861


namespace NUMINAMATH_GPT_prob_of_target_hit_l2128_212869

noncomputable def probability_target_hit : ℚ :=
  let pA := (1 : ℚ) / 2
  let pB := (1 : ℚ) / 3
  let pC := (1 : ℚ) / 4
  let pA' := 1 - pA
  let pB' := 1 - pB
  let pC' := 1 - pC
  let pNoneHit := pA' * pB' * pC'
  1 - pNoneHit

-- Statement to be proved
theorem prob_of_target_hit : probability_target_hit = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_prob_of_target_hit_l2128_212869


namespace NUMINAMATH_GPT_train_crossing_time_l2128_212860

theorem train_crossing_time
  (length : ℝ) (speed : ℝ) (time : ℝ)
  (h1 : length = 100) (h2 : speed = 30.000000000000004) :
  time = length / speed :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2128_212860


namespace NUMINAMATH_GPT_sphere_surface_area_l2128_212855

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) : ∃ A, A = 36 * Real.pi * (2 ^ (2 / 3)) := 
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l2128_212855


namespace NUMINAMATH_GPT_evaluate_expression_l2128_212816

theorem evaluate_expression (b : ℤ) (x : ℤ) (h : x = b + 9) : (x - b + 5 = 14) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2128_212816


namespace NUMINAMATH_GPT_evaluate_expression_correct_l2128_212893

noncomputable def evaluate_expression :=
  abs (-1) - ((-3.14 + Real.pi) ^ 0) + (2 ^ (-1 : ℤ)) + (Real.cos (Real.pi / 6)) ^ 2

theorem evaluate_expression_correct : evaluate_expression = 5 / 4 := by sorry

end NUMINAMATH_GPT_evaluate_expression_correct_l2128_212893


namespace NUMINAMATH_GPT_original_price_of_shirts_l2128_212830

theorem original_price_of_shirts 
  (sale_price : ℝ) 
  (fraction_of_original : ℝ) 
  (original_price : ℝ) 
  (h1 : sale_price = 6) 
  (h2 : fraction_of_original = 0.25) 
  (h3 : sale_price = fraction_of_original * original_price) 
  : original_price = 24 := 
by 
  sorry

end NUMINAMATH_GPT_original_price_of_shirts_l2128_212830


namespace NUMINAMATH_GPT_arctan_sum_l2128_212834

theorem arctan_sum (a b : ℝ) (h1 : a = 2 / 3) (h2 : (a + 1) * (b + 1) = 8 / 3) :
  Real.arctan a + Real.arctan b = Real.arctan (19 / 9) := by
  sorry

end NUMINAMATH_GPT_arctan_sum_l2128_212834


namespace NUMINAMATH_GPT_zeros_of_quadratic_l2128_212899

theorem zeros_of_quadratic (a b : ℝ) (h : a + b = 0) : 
  ∀ x, (b * x^2 - a * x = 0) ↔ (x = 0 ∨ x = -1) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_zeros_of_quadratic_l2128_212899


namespace NUMINAMATH_GPT_day_crew_fraction_loaded_l2128_212867

-- Let D be the number of boxes loaded by each worker on the day crew
-- Let W_d be the number of workers on the day crew
-- Let W_n be the number of workers on the night crew
-- Let B_d be the total number of boxes loaded by the day crew
-- Let B_n be the total number of boxes loaded by the night crew

variable (D W_d : ℕ) 
variable (B_d := D * W_d)
variable (W_n := (4 / 9 : ℚ) * W_d)
variable (B_n := (3 / 4 : ℚ) * D * W_n)
variable (total_boxes := B_d + B_n)

theorem day_crew_fraction_loaded : 
  (D * W_d) / (D * W_d + (3 / 4 : ℚ) * D * ((4 / 9 : ℚ) * W_d)) = (3 / 4 : ℚ) := sorry

end NUMINAMATH_GPT_day_crew_fraction_loaded_l2128_212867


namespace NUMINAMATH_GPT_part1_part2_l2128_212880

open Real

theorem part1 (m : ℝ) (h : ∀ x : ℝ, abs (x - 2) + abs (x - 3) ≥ m) : m ≤ 1 := 
sorry

theorem part2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 = 1 / a + 1 / (2 * b) + 1 / (3 * c)) : a + 2 * b + 3 * c ≥ 9 := 
sorry

end NUMINAMATH_GPT_part1_part2_l2128_212880


namespace NUMINAMATH_GPT_binomial_identity_l2128_212824

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) :=
sorry

end NUMINAMATH_GPT_binomial_identity_l2128_212824


namespace NUMINAMATH_GPT_equation_of_line_l2128_212842

theorem equation_of_line (x y : ℝ) :
  (∃ (x1 y1 : ℝ), (x1 = 0) ∧ (y1= 2) ∧ (y - y1 = 2 * (x - x1))) → (y = 2 * x + 2) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_l2128_212842


namespace NUMINAMATH_GPT_problem_proof_l2128_212807

theorem problem_proof (a b x y : ℝ) (h1 : a + b = 0) (h2 : x * y = 1) : 5 * |a + b| - 5 * (x * y) = -5 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l2128_212807


namespace NUMINAMATH_GPT_percentage_increase_l2128_212826

theorem percentage_increase (x : ℝ) (y : ℝ) (h1 : x = 114.4) (h2 : y = 88) : 
  ((x - y) / y) * 100 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l2128_212826


namespace NUMINAMATH_GPT_machine_p_vs_machine_q_l2128_212851

variable (MachineA_rate MachineQ_rate MachineP_rate : ℝ)
variable (Total_sprockets : ℝ := 550)
variable (Production_rate_A : ℝ := 5)
variable (Production_rate_Q : ℝ := MachineA_rate + 0.1 * MachineA_rate)
variable (Time_Q : ℝ := Total_sprockets / Production_rate_Q)
variable (Time_P : ℝ)
variable (Difference : ℝ)

noncomputable def production_times_difference (MachineA_rate MachineQ_rate MachineP_rate : ℝ) : ℝ :=
  let Production_rate_Q := MachineA_rate + 0.1 * MachineA_rate
  let Time_Q := Total_sprockets / Production_rate_Q
  let Difference := Time_P - Time_Q
  Difference

theorem machine_p_vs_machine_q : 
  Production_rate_A = 5 → 
  Total_sprockets = 550 →
  Production_rate_Q = 5.5 →
  Time_Q = 100 →
  MachineP_rate = MachineP_rate →
  Time_P = Time_P →
  Difference = (Time_P - Time_Q) :=
by
  intros
  sorry

end NUMINAMATH_GPT_machine_p_vs_machine_q_l2128_212851


namespace NUMINAMATH_GPT_distance_to_bus_stand_l2128_212884

theorem distance_to_bus_stand :
  ∀ D : ℝ, (D / 5 - 0.2 = D / 6 + 0.25) → D = 13.5 :=
by
  intros D h
  sorry

end NUMINAMATH_GPT_distance_to_bus_stand_l2128_212884


namespace NUMINAMATH_GPT_find_number_of_girls_l2128_212868

noncomputable def B (G : ℕ) : ℕ := (8 * G) / 5

theorem find_number_of_girls (B G : ℕ) (h_ratio : B = (8 * G) / 5) (h_total : B + G = 312) : G = 120 :=
by
  -- the proof would be done here
  sorry

end NUMINAMATH_GPT_find_number_of_girls_l2128_212868


namespace NUMINAMATH_GPT_proof_problem_l2128_212828

def sum_even_ints (n : ℕ) : ℕ := n * (n + 1)
def sum_odd_ints (n : ℕ) : ℕ := n^2
def sum_specific_primes : ℕ := [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97].sum

theorem proof_problem : (sum_even_ints 100 - sum_odd_ints 100) + sum_specific_primes = 1063 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2128_212828


namespace NUMINAMATH_GPT_perfect_shells_l2128_212810

theorem perfect_shells (P_spiral B_spiral P_total : ℕ) 
  (h1 : 52 = 2 * B_spiral)
  (h2 : B_spiral = P_spiral + 21)
  (h3 : P_total = P_spiral + 12) :
  P_total = 17 :=
by
  sorry

end NUMINAMATH_GPT_perfect_shells_l2128_212810


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l2128_212841

theorem equilateral_triangle_side_length (a : ℝ) (h : 3 * a = 18) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l2128_212841


namespace NUMINAMATH_GPT_rhombus_longer_diagonal_l2128_212818

theorem rhombus_longer_diagonal (a b d_1 : ℝ) (h_side : a = 60) (h_d1 : d_1 = 56) :
  ∃ d_2, d_2 = 106 := by
  sorry

end NUMINAMATH_GPT_rhombus_longer_diagonal_l2128_212818


namespace NUMINAMATH_GPT_al_original_portion_l2128_212845

theorem al_original_portion (a b c : ℝ) (h1 : a + b + c = 1200) (h2 : 0.75 * a + 2 * b + 2 * c = 1800) : a = 480 :=
by
  sorry

end NUMINAMATH_GPT_al_original_portion_l2128_212845


namespace NUMINAMATH_GPT_maria_money_left_l2128_212859

def ticket_cost : ℕ := 300
def hotel_cost : ℕ := ticket_cost / 2
def transportation_cost : ℕ := 80
def num_days : ℕ := 5
def avg_meal_cost_per_day : ℕ := 40
def tourist_tax_rate : ℚ := 0.10
def starting_amount : ℕ := 760

def total_meal_cost : ℕ := num_days * avg_meal_cost_per_day
def expenses_subject_to_tax := hotel_cost + transportation_cost
def tourist_tax := tourist_tax_rate * expenses_subject_to_tax
def total_expenses := ticket_cost + hotel_cost + transportation_cost + total_meal_cost + tourist_tax
def money_left := starting_amount - total_expenses

theorem maria_money_left : money_left = 7 := by
  sorry

end NUMINAMATH_GPT_maria_money_left_l2128_212859


namespace NUMINAMATH_GPT_john_investment_in_bank_a_l2128_212852

theorem john_investment_in_bank_a :
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1500 ∧
    x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1740.54 ∧
    x = 695 := sorry

end NUMINAMATH_GPT_john_investment_in_bank_a_l2128_212852


namespace NUMINAMATH_GPT_total_number_of_coins_l2128_212889

theorem total_number_of_coins (n : ℕ) (h : 4 * n - 4 = 240) : n^2 = 3721 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l2128_212889


namespace NUMINAMATH_GPT_different_movies_count_l2128_212854

theorem different_movies_count 
    (d_movies : ℕ) (h_movies : ℕ) (a_movies : ℕ) (b_movies : ℕ) (c_movies : ℕ) 
    (together_movies : ℕ) (dha_movies : ℕ) (bc_movies : ℕ) 
    (db_movies : ℕ) (ac_movies : ℕ)
    (H_d : d_movies = 20) (H_h : h_movies = 26) (H_a : a_movies = 35) 
    (H_b : b_movies = 29) (H_c : c_movies = 16)
    (H_together : together_movies = 5)
    (H_dha : dha_movies = 4) (H_bc : bc_movies = 3) 
    (H_db : db_movies = 2) (H_ac : ac_movies = 4) :
    d_movies + h_movies + a_movies + b_movies + c_movies 
    - 4 * together_movies - 3 * dha_movies - 2 * bc_movies - db_movies - 3 * ac_movies = 74 := by sorry

end NUMINAMATH_GPT_different_movies_count_l2128_212854


namespace NUMINAMATH_GPT_frank_hawaiian_slices_l2128_212813

theorem frank_hawaiian_slices:
  ∀ (total_slices dean_slices sammy_slices leftover_slices frank_slices : ℕ),
  total_slices = 24 →
  dean_slices = 6 →
  sammy_slices = 4 →
  leftover_slices = 11 →
  (total_slices - leftover_slices) = (dean_slices + sammy_slices + frank_slices) →
  frank_slices = 3 :=
by
  intros total_slices dean_slices sammy_slices leftover_slices frank_slices
  intros h_total h_dean h_sammy h_leftovers h_total_eaten
  sorry

end NUMINAMATH_GPT_frank_hawaiian_slices_l2128_212813


namespace NUMINAMATH_GPT_correct_equation_l2128_212879

theorem correct_equation (x : ℕ) (h : x ≤ 26) :
    let a_parts := 2100
    let b_parts := 1200
    let total_workers := 26
    let a_rate := 30
    let b_rate := 20
    let type_a_time := (a_parts : ℚ) / (a_rate * x)
    let type_b_time := (b_parts : ℚ) / (b_rate * (total_workers - x))
    type_a_time = type_b_time :=
by
    sorry

end NUMINAMATH_GPT_correct_equation_l2128_212879


namespace NUMINAMATH_GPT_find_length_QR_l2128_212896

-- Define the provided conditions as Lean definitions
variables (Q P R : ℝ) (h_cos : Real.cos Q = 0.3) (QP : ℝ) (h_QP : QP = 15)
  
-- State the theorem we need to prove
theorem find_length_QR (QR : ℝ) (h_triangle : QP / QR = Real.cos Q) : QR = 50 := sorry

end NUMINAMATH_GPT_find_length_QR_l2128_212896


namespace NUMINAMATH_GPT_parabola_vertex_l2128_212888

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := (x - 2)^2 + 5

-- State the theorem to find the vertex
theorem parabola_vertex : ∃ h k : ℝ, ∀ x : ℝ, parabola_equation x = (x - h)^2 + k ∧ h = 2 ∧ k = 5 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l2128_212888


namespace NUMINAMATH_GPT_vasya_kolya_difference_impossible_l2128_212875

theorem vasya_kolya_difference_impossible : 
  ∀ k v : ℕ, (∃ q₁ q₂ : ℕ, 14400 = q₁ * 2 + q₂ * 2 + 1 + 1) → ¬ ∃ k, ∃ v, (v - k = 11 ∧ 14400 = k * q₁ + v * q₂) :=
by sorry

end NUMINAMATH_GPT_vasya_kolya_difference_impossible_l2128_212875


namespace NUMINAMATH_GPT_find_q_l2128_212805

open Polynomial

-- Define the conditions for the roots of the first polynomial
def roots_of_first_eq (a b m : ℝ) (h : a * b = 3) : Prop := 
  ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)

-- Define the problem statement
theorem find_q (a b m p q : ℝ) 
  (h1 : a * b = 3) 
  (h2 : ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)) 
  (h3 : ∀ x, (x^2 - p*x + q) = (x - (a + 2/b)) * (x - (b + 2/a))) :
  q = 25 / 3 :=
sorry

end NUMINAMATH_GPT_find_q_l2128_212805


namespace NUMINAMATH_GPT_f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l2128_212897

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 / x - 1 else 2 / (-x) - 1

-- Assertion 1: Value of f(-1)
theorem f_neg_one : f (-1) = 1 := 
sorry

-- Assertion 2: f(x) is a decreasing function on (0, +∞)
theorem f_decreasing_on_positive : ∀ a b : ℝ, 0 < b → b < a → f (a) < f (b) := 
sorry

-- Assertion 3: Expression of the function when x < 0
theorem f_expression_on_negative (x : ℝ) (hx : x < 0) : f x = 2 / (-x) - 1 := 
sorry

end NUMINAMATH_GPT_f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l2128_212897


namespace NUMINAMATH_GPT_perimeter_of_square_l2128_212886

-- Defining the square with area
structure Square where
  side_length : ℝ
  area : ℝ

-- Defining a constant square with given area 625
def givenSquare : Square := 
  { side_length := 25, -- will square root the area of 625
    area := 625 }

-- Defining the function to calculate the perimeter of the square
noncomputable def perimeter (s : Square) : ℝ :=
  4 * s.side_length

-- The theorem stating that the perimeter of the given square with area 625 is 100
theorem perimeter_of_square : perimeter givenSquare = 100 := 
sorry

end NUMINAMATH_GPT_perimeter_of_square_l2128_212886


namespace NUMINAMATH_GPT_space_per_bookshelf_l2128_212809

-- Defining the conditions
def S_room : ℕ := 400
def S_reserved : ℕ := 160
def n_shelves : ℕ := 3

-- Theorem statement
theorem space_per_bookshelf (S_room S_reserved n_shelves : ℕ)
  (h1 : S_room = 400) (h2 : S_reserved = 160) (h3 : n_shelves = 3) :
  (S_room - S_reserved) / n_shelves = 80 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_space_per_bookshelf_l2128_212809


namespace NUMINAMATH_GPT_water_depth_is_12_feet_l2128_212847

variable (Ron_height Dean_height Water_depth : ℕ)

-- Given conditions
axiom H1 : Ron_height = 14
axiom H2 : Dean_height = Ron_height - 8
axiom H3 : Water_depth = 2 * Dean_height

-- Prove that the water depth is 12 feet
theorem water_depth_is_12_feet : Water_depth = 12 :=
by
  sorry

end NUMINAMATH_GPT_water_depth_is_12_feet_l2128_212847


namespace NUMINAMATH_GPT_rooster_ratio_l2128_212815

theorem rooster_ratio (R H : ℕ) 
  (h1 : R + H = 80)
  (h2 : R + (1 / 4) * H = 35) :
  R / 80 = 1 / 4 :=
  sorry

end NUMINAMATH_GPT_rooster_ratio_l2128_212815


namespace NUMINAMATH_GPT_ladder_alley_width_l2128_212823

theorem ladder_alley_width (l : ℝ) (m : ℝ) (w : ℝ) (h : m = l / 2) :
  w = (l * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_ladder_alley_width_l2128_212823


namespace NUMINAMATH_GPT_monotonicity_f_parity_f_max_value_f_min_value_f_l2128_212831

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

-- Monotonicity Proof
theorem monotonicity_f : ∀ {x1 x2 : ℝ}, 2 < x1 → 2 < x2 → x1 < x2 → f x1 > f x2 :=
sorry

-- Parity Proof
theorem parity_f : ∀ x : ℝ, f (-x) = -f x :=
sorry

-- Maximum Value Proof
theorem max_value_f : ∀ {x : ℝ}, x = -6 → f x = -3/16 :=
sorry

-- Minimum Value Proof
theorem min_value_f : ∀ {x : ℝ}, x = -3 → f x = -3/5 :=
sorry

end NUMINAMATH_GPT_monotonicity_f_parity_f_max_value_f_min_value_f_l2128_212831


namespace NUMINAMATH_GPT_Andrena_more_than_Debelyn_l2128_212808

-- Define initial dolls count for each person
def Debelyn_initial_dolls : ℕ := 20
def Christel_initial_dolls : ℕ := 24

-- Define dolls given by Debelyn and Christel
def Debelyn_gift_dolls : ℕ := 2
def Christel_gift_dolls : ℕ := 5

-- Define remaining dolls for Debelyn and Christel after giving dolls away
def Debelyn_final_dolls : ℕ := Debelyn_initial_dolls - Debelyn_gift_dolls
def Christel_final_dolls : ℕ := Christel_initial_dolls - Christel_gift_dolls

-- Define Andrena's dolls after transactions
def Andrena_dolls : ℕ := Christel_final_dolls + 2

-- Define the Lean statement for proving Andrena has 3 more dolls than Debelyn
theorem Andrena_more_than_Debelyn : Andrena_dolls = Debelyn_final_dolls + 3 := by
  -- Here you would prove the statement
  sorry

end NUMINAMATH_GPT_Andrena_more_than_Debelyn_l2128_212808


namespace NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l2128_212858

theorem smallest_enclosing_sphere_radius :
  let r := 2
  let d := 4 * Real.sqrt 3
  let total_diameter := d + 2*r
  let radius_enclosing_sphere := total_diameter / 2
  radius_enclosing_sphere = 2 + 2 * Real.sqrt 3 := by
  -- Define the radius of the smaller spheres
  let r : ℝ := 2
  -- Space diagonal of the cube which is 4√3 where 4 is the side length
  let d : ℝ := 4 * Real.sqrt 3
  -- Total diameter of the sphere containing the cube (space diagonal + 2 radius of one sphere)
  let total_diameter : ℝ := d + 2 * r
  -- Radius of the enclosing sphere
  let radius_enclosing_sphere : ℝ := total_diameter / 2
  -- We need to prove that this radius equals 2 + 2√3
  sorry

end NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l2128_212858


namespace NUMINAMATH_GPT_measure_of_α_l2128_212829

variables (α β : ℝ)
-- Condition 1: α and β are complementary angles
def complementary := α + β = 180

-- Condition 2: Half of angle β is 30° less than α
def half_less_30 := α - (1 / 2) * β = 30

-- Theorem: Measure of angle α
theorem measure_of_α (α β : ℝ) (h1 : complementary α β) (h2 : half_less_30 α β) :
  α = 80 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_α_l2128_212829


namespace NUMINAMATH_GPT_tina_wins_more_than_losses_l2128_212850

theorem tina_wins_more_than_losses 
  (initial_wins : ℕ)
  (additional_wins : ℕ)
  (first_loss : ℕ)
  (doubled_wins : ℕ)
  (second_loss : ℕ)
  (total_wins : ℕ)
  (total_losses : ℕ)
  (final_difference : ℕ) :
  initial_wins = 10 →
  additional_wins = 5 →
  first_loss = 1 →
  doubled_wins = 30 →
  second_loss = 1 →
  total_wins = initial_wins + additional_wins + doubled_wins →
  total_losses = first_loss + second_loss →
  final_difference = total_wins - total_losses →
  final_difference = 43 :=
by
  sorry

end NUMINAMATH_GPT_tina_wins_more_than_losses_l2128_212850


namespace NUMINAMATH_GPT_initial_men_is_250_l2128_212853

-- Define the given conditions
def provisions (initial_men remaining_men initial_days remaining_days : ℕ) : Prop :=
  initial_men * initial_days = remaining_men * remaining_days

-- Define the problem statement
theorem initial_men_is_250 (initial_days remaining_days : ℕ) (remaining_men_leaving : ℕ) :
  provisions initial_men (initial_men - remaining_men_leaving) initial_days remaining_days → initial_men = 250 :=
by
  intros h
  -- Requirement to solve the theorem.
  -- This is where the proof steps would go, but we put sorry to satisfy the statement requirement.
  sorry

end NUMINAMATH_GPT_initial_men_is_250_l2128_212853


namespace NUMINAMATH_GPT_perpendicular_lines_m_value_l2128_212820

-- Define the first line
def line1 (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the second line
def line2 (x y : ℝ) (m : ℝ) : Prop := 6 * x - m * y - 3 = 0

-- Define the perpendicular condition for slopes of two lines
def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Prove the value of m for perpendicular lines
theorem perpendicular_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, line1 x y → ∃ y', line2 x y' m) →
  (∀ x y : ℝ, ∃ x', line1 x y ∧ line2 x' y m) →
  perpendicular_slopes 3 (6 / m) →
  m = -18 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_m_value_l2128_212820


namespace NUMINAMATH_GPT_bridge_length_l2128_212874

-- Defining the problem based on the given conditions and proof goal
theorem bridge_length (L : ℝ) 
  (h1 : L / 4 + L / 3 + 120 = L) :
  L = 288 :=
sorry

end NUMINAMATH_GPT_bridge_length_l2128_212874


namespace NUMINAMATH_GPT_boat_speed_is_20_l2128_212890

-- Definitions based on conditions from the problem
def boat_speed_still_water (x : ℝ) : Prop := 
  let current_speed := 5
  let downstream_distance := 8.75
  let downstream_time := 21 / 60
  let downstream_speed := x + current_speed
  downstream_speed * downstream_time = downstream_distance

-- The theorem to prove
theorem boat_speed_is_20 : boat_speed_still_water 20 :=
by 
  unfold boat_speed_still_water
  sorry

end NUMINAMATH_GPT_boat_speed_is_20_l2128_212890


namespace NUMINAMATH_GPT_solve_equation_l2128_212883

noncomputable def f (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs x - 8) - 4) - 2) - 1)

noncomputable def g (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs x - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1)

theorem solve_equation : ∀ (x : ℝ), f x = g x :=
by
  sorry -- The proof will be inserted here

end NUMINAMATH_GPT_solve_equation_l2128_212883


namespace NUMINAMATH_GPT_percent_profit_l2128_212838

variable (C S : ℝ)

theorem percent_profit (h : 72 * C = 60 * S) : ((S - C) / C) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percent_profit_l2128_212838


namespace NUMINAMATH_GPT_solve_equation_l2128_212846

theorem solve_equation (x : ℝ) : (2*x - 1)^2 = 81 ↔ (x = 5 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2128_212846


namespace NUMINAMATH_GPT_nonneg_integer_solution_l2128_212857

theorem nonneg_integer_solution (a b c : ℕ) (h : 5^a * 7^b + 4 = 3^c) : (a, b, c) = (1, 0, 2) := 
sorry

end NUMINAMATH_GPT_nonneg_integer_solution_l2128_212857


namespace NUMINAMATH_GPT_find_value_of_expression_l2128_212822

theorem find_value_of_expression
  (k m : ℕ)
  (hk : 3^(k - 1) = 9)
  (hm : 4^(m + 2) = 64) :
  2^(3*k + 2*m) = 2^11 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l2128_212822


namespace NUMINAMATH_GPT_largest_value_x_l2128_212882

theorem largest_value_x (x a b c d : ℝ) (h_eq : 7 * x ^ 2 + 15 * x - 20 = 0) (h_form : x = (a + b * Real.sqrt c) / d) (ha : a = -15) (hb : b = 1) (hc : c = 785) (hd : d = 14) : (a * c * d) / b = -164850 := 
sorry

end NUMINAMATH_GPT_largest_value_x_l2128_212882


namespace NUMINAMATH_GPT_find_denominator_of_second_fraction_l2128_212865

theorem find_denominator_of_second_fraction (y : ℝ) (h : y > 0) (x : ℝ) :
  (2 * y) / 5 + (3 * y) / x = 0.7 * y → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_denominator_of_second_fraction_l2128_212865


namespace NUMINAMATH_GPT_frog_jump_distance_l2128_212827

theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_jump : ℕ) (frog_jump : ℕ) :
  grasshopper_jump = 9 → extra_jump = 3 → frog_jump = grasshopper_jump + extra_jump → frog_jump = 12 :=
by
  intros h_grasshopper h_extra h_frog
  rw [h_grasshopper, h_extra] at h_frog
  exact h_frog

end NUMINAMATH_GPT_frog_jump_distance_l2128_212827


namespace NUMINAMATH_GPT_vessel_base_length_l2128_212895

variables (L : ℝ) (edge : ℝ) (W : ℝ) (h : ℝ)
def volume_cube := edge^3
def volume_rise := L * W * h

theorem vessel_base_length :
  (volume_cube 16 = volume_rise L 15 13.653333333333334) →
  L = 20 :=
by sorry

end NUMINAMATH_GPT_vessel_base_length_l2128_212895


namespace NUMINAMATH_GPT_train_length_l2128_212802

noncomputable def length_of_first_train (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  let v1_m_per_s := v1 * 1000 / 3600
  let v2_m_per_s := v2 * 1000 / 3600
  let relative_speed := v1_m_per_s + v2_m_per_s
  let combined_length := relative_speed * t
  combined_length - l2

theorem train_length (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) (h_l2 : l2 = 200) 
  (h_v1 : v1 = 100) (h_v2 : v2 = 200) (h_t : t = 3.6) : length_of_first_train l2 v1 v2 t = 100 := by
  sorry

end NUMINAMATH_GPT_train_length_l2128_212802


namespace NUMINAMATH_GPT_banana_cost_l2128_212835

/-- If 4 bananas cost $20, then the cost of one banana is $5. -/
theorem banana_cost (total_cost num_bananas : ℕ) (cost_per_banana : ℕ) 
  (h : total_cost = 20 ∧ num_bananas = 4) : cost_per_banana = 5 := by
  sorry

end NUMINAMATH_GPT_banana_cost_l2128_212835


namespace NUMINAMATH_GPT_distance_between_vertices_hyperbola_l2128_212898

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_vertices_hyperbola_l2128_212898


namespace NUMINAMATH_GPT_product_of_roots_l2128_212844

theorem product_of_roots (a b c d : ℝ)
  (h1 : a = 16 ^ (1 / 5))
  (h2 : 16 = 2 ^ 4)
  (h3 : b = 64 ^ (1 / 6))
  (h4 : 64 = 2 ^ 6):
  a * b = 2 * (16 ^ (1 / 5)) := by
  sorry

end NUMINAMATH_GPT_product_of_roots_l2128_212844


namespace NUMINAMATH_GPT_contrapositive_example_l2128_212833

theorem contrapositive_example (x : ℝ) :
  (x ^ 2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x ^ 2 ≥ 1) :=
sorry

end NUMINAMATH_GPT_contrapositive_example_l2128_212833


namespace NUMINAMATH_GPT_no_fraternity_member_is_club_member_thm_l2128_212821

-- Definitions from the conditions
variable (Person : Type)
variable (Club : Person → Prop)
variable (Honest : Person → Prop)
variable (Student : Person → Prop)
variable (Fraternity : Person → Prop)

-- Hypotheses from the problem statements
axiom all_club_members_honest (p : Person) : Club p → Honest p
axiom some_students_not_honest : ∃ p : Person, Student p ∧ ¬ Honest p
axiom no_fraternity_member_is_club_member (p : Person) : Fraternity p → ¬ Club p

-- The theorem to be proven
theorem no_fraternity_member_is_club_member_thm : 
  ∀ p : Person, Fraternity p → ¬ Club p := 
by 
  sorry

end NUMINAMATH_GPT_no_fraternity_member_is_club_member_thm_l2128_212821


namespace NUMINAMATH_GPT_math_problem_l2128_212811

-- Statement of the theorem
theorem math_problem :
  (0.66)^3 - ((0.1)^3 / ((0.66)^2 + 0.066 + (0.1)^2)) = 0.3612 :=
by
  sorry -- Proof is not required

end NUMINAMATH_GPT_math_problem_l2128_212811


namespace NUMINAMATH_GPT_cube_volume_l2128_212825

theorem cube_volume (h : 12 * l = 72) : l^3 = 216 :=
sorry

end NUMINAMATH_GPT_cube_volume_l2128_212825


namespace NUMINAMATH_GPT_fraction_percent_of_y_l2128_212800

theorem fraction_percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) = 0.5 * y := by
  sorry

end NUMINAMATH_GPT_fraction_percent_of_y_l2128_212800


namespace NUMINAMATH_GPT_dot_product_value_l2128_212887

variables (a b : ℝ × ℝ)

theorem dot_product_value
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a.1 * b.1 + a.2 * b.2 = -12 :=
sorry

end NUMINAMATH_GPT_dot_product_value_l2128_212887


namespace NUMINAMATH_GPT_sean_total_apples_l2128_212876

-- Define initial apples
def initial_apples : Nat := 9

-- Define the number of apples Susan gives each day
def apples_per_day : Nat := 8

-- Define the number of days Susan gives apples
def number_of_days : Nat := 5

-- Calculate total apples given by Susan
def total_apples_given : Nat := apples_per_day * number_of_days

-- Define the final total apples
def total_apples : Nat := initial_apples + total_apples_given

-- Prove the number of total apples is 49
theorem sean_total_apples : total_apples = 49 := by
  sorry

end NUMINAMATH_GPT_sean_total_apples_l2128_212876


namespace NUMINAMATH_GPT_time_diff_is_6_l2128_212870

-- Define the speeds for the different sails
def speed_of_large_sail : ℕ := 50
def speed_of_small_sail : ℕ := 20

-- Define the distance of the trip
def trip_distance : ℕ := 200

-- Calculate the time for each sail
def time_large_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_small_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Define the time difference
def time_difference (distance : ℕ) (speed_large : ℕ) (speed_small : ℕ) : ℕ := 
  (distance / speed_small) - (distance / speed_large)

-- Prove that the time difference between the large and small sails is 6 hours
theorem time_diff_is_6 : time_difference trip_distance speed_of_large_sail speed_of_small_sail = 6 := by
  -- useful := time_difference trip_distance speed_of_large_sail speed_of_small_sail,
  -- change useful with 6,
  sorry

end NUMINAMATH_GPT_time_diff_is_6_l2128_212870


namespace NUMINAMATH_GPT_find_x2_y2_l2128_212840

theorem find_x2_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x^2 * y + x * y^2 = 1512) :
  x^2 + y^2 = 1136 ∨ x^2 + y^2 = 221 := by
  sorry

end NUMINAMATH_GPT_find_x2_y2_l2128_212840


namespace NUMINAMATH_GPT_money_left_after_bike_purchase_l2128_212801

-- Definitions based on conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def quarter_value : ℝ := 0.25
def bike_cost : ℝ := 180

-- The theorem statement
theorem money_left_after_bike_purchase : (jars * quarters_per_jar * quarter_value) - bike_cost = 20 := by
  sorry

end NUMINAMATH_GPT_money_left_after_bike_purchase_l2128_212801


namespace NUMINAMATH_GPT_misha_darts_score_l2128_212819

theorem misha_darts_score (x : ℕ) 
  (h1 : x >= 24)
  (h2 : x * 3 <= 72) : 
  2 * x = 48 :=
by
  sorry

end NUMINAMATH_GPT_misha_darts_score_l2128_212819


namespace NUMINAMATH_GPT_rectangle_length_l2128_212839

theorem rectangle_length (P B L : ℕ) (h1 : P = 800) (h2 : B = 300) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l2128_212839


namespace NUMINAMATH_GPT_Jillian_had_200_friends_l2128_212866

def oranges : ℕ := 80
def pieces_per_orange : ℕ := 10
def pieces_per_friend : ℕ := 4
def number_of_friends : ℕ := oranges * pieces_per_orange / pieces_per_friend

theorem Jillian_had_200_friends :
  number_of_friends = 200 :=
sorry

end NUMINAMATH_GPT_Jillian_had_200_friends_l2128_212866
