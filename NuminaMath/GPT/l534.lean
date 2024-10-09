import Mathlib

namespace cos_300_eq_half_l534_53461

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end cos_300_eq_half_l534_53461


namespace necessary_but_not_sufficient_condition_l534_53480

variable (a b : ℝ)

theorem necessary_but_not_sufficient_condition : (a > b) → ((a > b) ↔ ((a - b) * b^2 > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l534_53480


namespace Jirina_number_l534_53476

theorem Jirina_number (a b c d : ℕ) (h_abcd : 1000 * a + 100 * b + 10 * c + d = 1468) :
  (1000 * a + 100 * b + 10 * c + d) +
  (1000 * a + 100 * d + 10 * c + b) = 3332 ∧ 
  (1000 * a + 100 * b + 10 * c + d)+
  (1000 * c + 100 * b + 10 * a + d) = 7886 :=
by
  sorry

end Jirina_number_l534_53476


namespace fixed_amount_at_least_190_l534_53479

variable (F S : ℝ)

theorem fixed_amount_at_least_190
  (h1 : S = 7750)
  (h2 : F + 0.04 * S ≥ 500) :
  F ≥ 190 := by
  sorry

end fixed_amount_at_least_190_l534_53479


namespace negation_proposition_l534_53424

theorem negation_proposition : ¬ (∀ x : ℝ, (1 < x) → x^3 > x^(1/3)) ↔ ∃ x : ℝ, (1 < x) ∧ x^3 ≤ x^(1/3) := by
  sorry

end negation_proposition_l534_53424


namespace radius_of_circle_l534_53486

theorem radius_of_circle (P : ℝ) (PQ QR : ℝ) (distance_center_P : ℝ) (r : ℝ) :
  P = 17 ∧ PQ = 12 ∧ QR = 8 ∧ (PQ * (PQ + QR) = (distance_center_P - r) * (distance_center_P + r)) → r = 7 :=
by
  sorry

end radius_of_circle_l534_53486


namespace line_passes_fixed_point_max_distance_eqn_l534_53484

-- Definition of the line equation
def line_eq (a b x y : ℝ) : Prop :=
  (2 * a + b) * x + (a + b) * y + a - b = 0

-- Point P
def point_P : ℝ × ℝ :=
  (3, 4)

-- Fixed point that the line passes through
def fixed_point : ℝ × ℝ :=
  (-2, 3)

-- Statement that the line passes through the fixed point
theorem line_passes_fixed_point (a b : ℝ) :
  line_eq a b (-2) 3 :=
sorry

-- Equation of the line when distance from point P to line is maximized
def line_max_distance (a b : ℝ) : Prop :=
  5 * 3 + 4 + 7 = 0

-- Statement that the equation of the line is as given when distance is maximized
theorem max_distance_eqn (a b : ℝ) :
  line_max_distance a b :=
sorry

end line_passes_fixed_point_max_distance_eqn_l534_53484


namespace range_of_a_l534_53469

def f (x a : ℝ) : ℝ := x^2 + a * x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ∧ (∃ x : ℝ, f (f x a) a = 0) → (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l534_53469


namespace common_difference_arith_seq_l534_53458

theorem common_difference_arith_seq (a : ℕ → ℝ) (d : ℝ)
    (h₀ : a 1 + a 5 = 10)
    (h₁ : a 4 = 7)
    (h₂ : ∀ n, a (n + 1) = a n + d) : 
    d = 2 := by
  sorry

end common_difference_arith_seq_l534_53458


namespace minimum_apples_l534_53493

theorem minimum_apples (n : ℕ) (A : ℕ) (h1 : A = 25 * n + 24) (h2 : A > 300) : A = 324 :=
sorry

end minimum_apples_l534_53493


namespace diesel_train_slower_l534_53408

theorem diesel_train_slower
    (t_cattle_speed : ℕ)
    (t_cattle_early_hours : ℕ)
    (t_diesel_hours : ℕ)
    (total_distance : ℕ)
    (diesel_speed : ℕ) :
  t_cattle_speed = 56 →
  t_cattle_early_hours = 6 →
  t_diesel_hours = 12 →
  total_distance = 1284 →
  diesel_speed = 23 →
  t_cattle_speed - diesel_speed = 33 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end diesel_train_slower_l534_53408


namespace find_ordered_pair_l534_53422

theorem find_ordered_pair :
  ∃ (x y : ℚ), 7 * x - 3 * y = 6 ∧ 4 * x + 5 * y = 23 ∧ 
               x = 99 / 47 ∧ y = 137 / 47 :=
by
  sorry

end find_ordered_pair_l534_53422


namespace total_savings_in_2_months_l534_53439

def students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_in_month : ℕ := 4
def months : ℕ := 2

def total_contribution_per_week : ℕ := students * contribution_per_student_per_week
def total_weeks : ℕ := months * weeks_in_month
def total_savings : ℕ := total_contribution_per_week * total_weeks

theorem total_savings_in_2_months : total_savings = 480 := by
  -- Proof goes here
  sorry

end total_savings_in_2_months_l534_53439


namespace overall_average_marks_l534_53497

theorem overall_average_marks
  (avg_A : ℝ) (n_A : ℕ) (avg_B : ℝ) (n_B : ℕ) (avg_C : ℝ) (n_C : ℕ)
  (h_avg_A : avg_A = 40) (h_n_A : n_A = 12)
  (h_avg_B : avg_B = 60) (h_n_B : n_B = 28)
  (h_avg_C : avg_C = 55) (h_n_C : n_C = 15) :
  ((n_A * avg_A) + (n_B * avg_B) + (n_C * avg_C)) / (n_A + n_B + n_C) = 54.27 := by
  sorry

end overall_average_marks_l534_53497


namespace eggs_volume_correct_l534_53490

def raw_spinach_volume : ℕ := 40
def cooking_reduction_ratio : ℚ := 0.20
def cream_cheese_volume : ℕ := 6
def total_quiche_volume : ℕ := 18
def cooked_spinach_volume := (raw_spinach_volume : ℚ) * cooking_reduction_ratio
def combined_spinach_and_cream_cheese_volume := cooked_spinach_volume + (cream_cheese_volume : ℚ)
def eggs_volume := (total_quiche_volume : ℚ) - combined_spinach_and_cream_cheese_volume

theorem eggs_volume_correct : eggs_volume = 4 := by
  sorry

end eggs_volume_correct_l534_53490


namespace P_necessary_but_not_sufficient_for_q_l534_53452

def M : Set ℝ := {x : ℝ | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x : ℝ | x^2 + x < 0}

theorem P_necessary_but_not_sufficient_for_q :
  (∀ x, x ∈ N → x ∈ M) ∧ (∃ x, x ∈ M ∧ x ∉ N) :=
by
  sorry

end P_necessary_but_not_sufficient_for_q_l534_53452


namespace new_average_increased_by_40_percent_l534_53467

theorem new_average_increased_by_40_percent 
  (n : ℕ) (initial_avg : ℝ) (initial_marks : ℝ) (new_marks : ℝ) (new_avg : ℝ)
  (h1 : n = 37)
  (h2 : initial_avg = 73)
  (h3 : initial_marks = (initial_avg * n))
  (h4 : new_marks = (initial_marks * 1.40))
  (h5 : new_avg = (new_marks / n)) :
  new_avg = 102.2 :=
sorry

end new_average_increased_by_40_percent_l534_53467


namespace simplify_fraction_l534_53412

theorem simplify_fraction (k : ℝ) : 
  (∃ a b : ℝ, (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a = 1 ∧ b = 3 ∧ (a / b) = 1/3) := by
  sorry

end simplify_fraction_l534_53412


namespace arithmetic_geometric_sequence_ab_l534_53483

theorem arithmetic_geometric_sequence_ab :
  ∀ (a l m b n : ℤ), 
    (b < 0) → 
    (2 * a = -10) → 
    (b^2 = 9) → 
    ab = 15 :=
by
  intros a l m b n hb ha hb_eq
  sorry

end arithmetic_geometric_sequence_ab_l534_53483


namespace quadratic_always_positive_l534_53423

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - k + 4 > 0) ↔ -2 * Real.sqrt 3 < k ∧ k < 2 * Real.sqrt 3 := by
  sorry

end quadratic_always_positive_l534_53423


namespace amber_worked_hours_l534_53436

-- Define the variables and conditions
variables (A : ℝ) (Armand_hours : ℝ) (Ella_hours : ℝ)
variables (h1 : Armand_hours = A / 3) (h2 : Ella_hours = 2 * A)
variables (h3 : A + Armand_hours + Ella_hours = 40)

-- Prove the statement
theorem amber_worked_hours : A = 12 :=
by
  sorry

end amber_worked_hours_l534_53436


namespace sampling_probabilities_equal_l534_53401

variables (total_items first_grade_items second_grade_items equal_grade_items substandard_items : ℕ)
variables (p_1 p_2 p_3 : ℚ)

-- Conditions given in the problem
def conditions := 
  total_items = 160 ∧ 
  first_grade_items = 48 ∧ 
  second_grade_items = 64 ∧ 
  equal_grade_items = 3 ∧ 
  substandard_items = 1 ∧ 
  p_1 = 1 / 8 ∧ 
  p_2 = 1 / 8 ∧ 
  p_3 = 1 / 8

-- The theorem to be proved
theorem sampling_probabilities_equal (h : conditions total_items first_grade_items second_grade_items equal_grade_items substandard_items p_1 p_2 p_3) :
  p_1 = p_2 ∧ p_2 = p_3 :=
sorry

end sampling_probabilities_equal_l534_53401


namespace P_investment_time_l534_53403

noncomputable def investment_in_months 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop)
  (time_Q : ℕ)
  (time_P : ℕ)
  (x : ℕ) : Prop :=
  investment_ratio_PQ 7 5 ∧ 
  profit_ratio_PQ 7 9 ∧ 
  time_Q = 9 ∧ 
  (7 * time_P) / (5 * time_Q) = 7 / 9

theorem P_investment_time 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop) 
  (x : ℕ) : Prop :=
  ∀ (t : ℕ), investment_in_months investment_ratio_PQ profit_ratio_PQ 9 t x → t = 5

end P_investment_time_l534_53403


namespace solve_xyz_l534_53485

theorem solve_xyz (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : z > 0) (h4 : x^2 = y * 2^z + 1) :
  (z ≥ 4 ∧ x = 2^(z-1) + 1 ∧ y = 2^(z-2) + 1) ∨
  (z ≥ 5 ∧ x = 2^(z-1) - 1 ∧ y = 2^(z-2) - 1) ∨
  (z ≥ 3 ∧ x = 2^z - 1 ∧ y = 2^z - 2) :=
sorry

end solve_xyz_l534_53485


namespace minimum_yellow_balls_l534_53489

theorem minimum_yellow_balls (g o y : ℕ) :
  (o ≥ (1/3:ℝ) * g) ∧ (o ≤ (1/4:ℝ) * y) ∧ (g + o ≥ 75) → y ≥ 76 :=
sorry

end minimum_yellow_balls_l534_53489


namespace inequality_proof_l534_53498

theorem inequality_proof 
  (a b c d : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (sum_eq : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := 
by 
  sorry

end inequality_proof_l534_53498


namespace ensure_two_different_colors_ensure_two_yellow_balls_l534_53496

-- First statement: Ensuring two balls of different colors
theorem ensure_two_different_colors (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 11 ∧ 
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, draws i ≠ draws j := 
sorry

-- Second statement: Ensuring two yellow balls
theorem ensure_two_yellow_balls (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 22 ∧
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, (draws i).val - balls_red - balls_white < balls_yellow ∧ 
              (draws j).val - balls_red - balls_white < balls_yellow ∧
              draws i = draws j := 
sorry

end ensure_two_different_colors_ensure_two_yellow_balls_l534_53496


namespace mildred_weight_is_correct_l534_53425

noncomputable def carol_weight := 9
noncomputable def mildred_weight := carol_weight + 50

theorem mildred_weight_is_correct : mildred_weight = 59 :=
by 
  -- the proof is omitted
  sorry

end mildred_weight_is_correct_l534_53425


namespace three_digit_oddfactors_count_is_22_l534_53445

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l534_53445


namespace claudia_fills_4ounce_glasses_l534_53455

theorem claudia_fills_4ounce_glasses :
  ∀ (total_water : ℕ) (five_ounce_glasses : ℕ) (eight_ounce_glasses : ℕ) 
    (four_ounce_glass_volume : ℕ),
  total_water = 122 →
  five_ounce_glasses = 6 →
  eight_ounce_glasses = 4 →
  four_ounce_glass_volume = 4 →
  (total_water - (five_ounce_glasses * 5 + eight_ounce_glasses * 8)) / four_ounce_glass_volume = 15 :=
by
  intros _ _ _ _ _ _ _ _ 
  sorry

end claudia_fills_4ounce_glasses_l534_53455


namespace file_organization_ratio_l534_53435

variable (X : ℕ) -- The number of files organized in the morning
variable (total_files morning_files afternoon_files missing_files : ℕ)

-- Conditions
def condition1 : total_files = 60 := by sorry
def condition2 : afternoon_files = 15 := by sorry
def condition3 : missing_files = 15 := by sorry
def condition4 : morning_files = X := by sorry
def condition5 : morning_files + afternoon_files + missing_files = total_files := by sorry

-- Question
def ratio_morning_to_total : Prop :=
  let organized_files := total_files - afternoon_files - missing_files
  (organized_files / total_files : ℚ) = 1 / 2

-- Proof statement
theorem file_organization_ratio : 
  ∀ (X total_files morning_files afternoon_files missing_files : ℕ), 
    total_files = 60 → 
    afternoon_files = 15 → 
    missing_files = 15 → 
    morning_files = X → 
    morning_files + afternoon_files + missing_files = total_files → 
    (X / 60 : ℚ) = 1 / 2 := by 
  sorry

end file_organization_ratio_l534_53435


namespace problem_1_problem_2_problem_3_l534_53428

def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def notU (s : Set ℝ) : Set ℝ := { x | x ∉ s ∧ x ∈ U }

theorem problem_1 : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
sorry

theorem problem_2 : notU A ∪ B = { x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7) } :=
sorry

theorem problem_3 : A ∩ notU B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end problem_1_problem_2_problem_3_l534_53428


namespace line_BC_eq_l534_53463

def altitude1 (x y : ℝ) : Prop := x + y = 0
def altitude2 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def point_A : ℝ × ℝ := (1, 2)

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem line_BC_eq (x y : ℝ) :
  (∃ b c : ℝ × ℝ, altitude1 b.1 b.2 ∧ altitude2 c.1 c.2 ∧
                   line_eq 2 3 7 b.1 b.2 ∧ line_eq 2 3 7 c.1 c.2 ∧
                   b ≠ c) → 
    line_eq 2 3 7 x y :=
by sorry

end line_BC_eq_l534_53463


namespace units_digit_of_x4_plus_inv_x4_l534_53454

theorem units_digit_of_x4_plus_inv_x4 (x : ℝ) (hx : x^2 - 13 * x + 1 = 0) : 
  (x^4 + x⁻¹ ^ 4) % 10 = 7 := sorry

end units_digit_of_x4_plus_inv_x4_l534_53454


namespace vector_orthogonality_l534_53450

variables (x : ℝ)

def vec_a := (x - 1, 2)
def vec_b := (1, x)

theorem vector_orthogonality :
  (vec_a x).fst * (vec_b x).fst + (vec_a x).snd * (vec_b x).snd = 0 ↔ x = 1 / 3 := by
  sorry

end vector_orthogonality_l534_53450


namespace globe_division_l534_53474

theorem globe_division (parallels meridians : ℕ)
  (h_parallels : parallels = 17)
  (h_meridians : meridians = 24) :
  let slices_per_sector := parallels + 1
  let sectors := meridians
  let total_parts := slices_per_sector * sectors
  total_parts = 432 := by
  sorry

end globe_division_l534_53474


namespace Billy_age_l534_53410

-- Defining the ages of Billy, Joe, and Sam
variable (B J S : ℕ)

-- Conditions given in the problem
axiom Billy_twice_Joe : B = 2 * J
axiom sum_BJ_three_times_S : B + J = 3 * S
axiom Sam_age : S = 27

-- Statement to prove
theorem Billy_age : B = 54 :=
by
  sorry

end Billy_age_l534_53410


namespace distinct_roots_of_quadratic_l534_53481

variable {a b : ℝ}
-- condition: a and b are distinct
variable (h_distinct: a ≠ b)

theorem distinct_roots_of_quadratic (a b : ℝ) (h_distinct : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x + a)*(x + b) = 2*x + a + b :=
by
  sorry

end distinct_roots_of_quadratic_l534_53481


namespace total_value_of_goods_l534_53449

theorem total_value_of_goods (V : ℝ) (tax_paid : ℝ) (tax_exemption : ℝ) (tax_rate : ℝ) :
  tax_exemption = 600 → tax_rate = 0.11 → tax_paid = 123.2 → 0.11 * (V - 600) = tax_paid → V = 1720 :=
by
  sorry

end total_value_of_goods_l534_53449


namespace street_sweeper_routes_l534_53419

def num_routes (A B C : Type) :=
  -- Conditions: Starts from point A, 
  -- travels through all streets exactly once, 
  -- and returns to point A.
  -- Correct Answer: Total routes = 12
  2 * 6 = 12

theorem street_sweeper_routes (A B C : Type) : num_routes A B C := by
  -- The proof is omitted as per instructions
  sorry

end street_sweeper_routes_l534_53419


namespace beads_bracelet_rotational_symmetry_l534_53499

theorem beads_bracelet_rotational_symmetry :
  let n := 8
  let factorial := Nat.factorial
  (factorial n / n = 5040) := by
  sorry

end beads_bracelet_rotational_symmetry_l534_53499


namespace ages_of_siblings_l534_53491

-- Define the variables representing the ages of the siblings
variables (R D S E : ℕ)

-- Define the conditions
def conditions := 
  R = D + 6 ∧ 
  D = S + 8 ∧ 
  E = R - 5 ∧ 
  R + 8 = 2 * (S + 8)

-- Define the statement to be proved
theorem ages_of_siblings (h : conditions R D S E) : 
  R = 20 ∧ D = 14 ∧ S = 6 ∧ E = 15 :=
sorry

end ages_of_siblings_l534_53491


namespace like_terms_set_l534_53413

theorem like_terms_set (a b : ℕ) (x y : ℝ) : 
  (¬ (a = b)) ∧
  ((-2 * x^3 * y^3 = y^3 * x^3)) ∧ 
  (¬ (1 * x * y = 2 * x * y^3)) ∧ 
  (¬ (-6 = x)) :=
by
  sorry

end like_terms_set_l534_53413


namespace arithmetic_sequence_50th_term_l534_53453

theorem arithmetic_sequence_50th_term :
  let a1 := 3
  let d := 2
  let n := 50
  let a_n := a1 + (n - 1) * d
  a_n = 101 :=
by
  sorry

end arithmetic_sequence_50th_term_l534_53453


namespace max_tickets_l534_53433

theorem max_tickets (n : ℕ) (H : 15 * n ≤ 120) : n ≤ 8 :=
by sorry

end max_tickets_l534_53433


namespace find_triplets_satisfying_equation_l534_53457

theorem find_triplets_satisfying_equation :
  ∃ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧ (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triplets_satisfying_equation_l534_53457


namespace speed_conversion_l534_53471

noncomputable def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_conversion (h : 1 = 3.6) : mps_to_kmph 12.7788 = 45.96 :=
  by
    sorry

end speed_conversion_l534_53471


namespace find_constant_term_of_polynomial_with_negative_integer_roots_l534_53475

theorem find_constant_term_of_polynomial_with_negative_integer_roots
  (p q r s : ℝ) (t1 t2 t3 t4 : ℝ)
  (h_roots : ∀ {x : ℝ}, x^4 + p*x^3 + q*x^2 + r*x + s = (x + t1)*(x + t2)*(x + t3)*(x + t4))
  (h_neg_int_roots : ∀ {i : ℕ}, i < 4 → t1 = i ∨ t2 = i ∨ t3 = i ∨ t4 = i)
  (h_sum_coeffs : p + q + r + s = 168) :
  s = 144 :=
by
  sorry

end find_constant_term_of_polynomial_with_negative_integer_roots_l534_53475


namespace original_bill_l534_53442

theorem original_bill (n : ℕ) (d : ℝ) (p : ℝ) (B : ℝ) (h1 : n = 5) (h2 : d = 0.06) (h3 : p = 18.8)
  (h4 : 0.94 * B = n * p) :
  B = 100 :=
sorry

end original_bill_l534_53442


namespace problem1_problem2_l534_53421

-- Problem1
theorem problem1 (a : ℤ) (h : a = -2) :
    ( (a^2 + a) / (a^2 - 3 * a) / (a^2 - 1) / (a - 3) - 1 / (a + 1) = 2 / 3) :=
by 
  sorry

-- Problem2
theorem problem2 (x : ℤ) :
    ( (x^2 - 1) / (x - 4) / (x + 1) / (4 - x) = 1 - x) :=
by 
  sorry

end problem1_problem2_l534_53421


namespace basketball_starting_lineups_l534_53444

theorem basketball_starting_lineups (n_players n_guards n_forwards n_centers : ℕ)
  (h_players : n_players = 12)
  (h_guards : n_guards = 2)
  (h_forwards : n_forwards = 2)
  (h_centers : n_centers = 1) :
  (Nat.choose n_players n_guards) * (Nat.choose (n_players - n_guards) n_forwards) * (Nat.choose (n_players - n_guards - n_forwards) n_centers) = 23760 := by
  sorry

end basketball_starting_lineups_l534_53444


namespace payroll_amount_l534_53406

theorem payroll_amount (P : ℝ) 
  (h1 : P > 500000) 
  (h2 : 0.004 * (P - 500000) - 1000 = 600) :
  P = 900000 :=
by
  sorry

end payroll_amount_l534_53406


namespace quadratic_distinct_roots_l534_53451

theorem quadratic_distinct_roots (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
sorry

end quadratic_distinct_roots_l534_53451


namespace ratio_aerobics_to_weight_training_l534_53420

def time_spent_exercising : ℕ := 250
def time_spent_aerobics : ℕ := 150
def time_spent_weight_training : ℕ := 100

theorem ratio_aerobics_to_weight_training :
    (time_spent_aerobics / gcd time_spent_aerobics time_spent_weight_training) = 3 ∧
    (time_spent_weight_training / gcd time_spent_aerobics time_spent_weight_training) = 2 :=
by
    sorry

end ratio_aerobics_to_weight_training_l534_53420


namespace net_gain_difference_l534_53407

def first_applicant_salary : ℝ := 42000
def first_applicant_training_cost_per_month : ℝ := 1200
def first_applicant_training_months : ℝ := 3
def first_applicant_revenue : ℝ := 93000

def second_applicant_salary : ℝ := 45000
def second_applicant_hiring_bonus_percentage : ℝ := 0.01
def second_applicant_revenue : ℝ := 92000

def first_applicant_total_cost : ℝ := first_applicant_salary + first_applicant_training_cost_per_month * first_applicant_training_months
def first_applicant_net_gain : ℝ := first_applicant_revenue - first_applicant_total_cost

def second_applicant_hiring_bonus : ℝ := second_applicant_salary * second_applicant_hiring_bonus_percentage
def second_applicant_total_cost : ℝ := second_applicant_salary + second_applicant_hiring_bonus
def second_applicant_net_gain : ℝ := second_applicant_revenue - second_applicant_total_cost

theorem net_gain_difference :
  first_applicant_net_gain - second_applicant_net_gain = 850 := by
  sorry

end net_gain_difference_l534_53407


namespace general_form_line_eq_line_passes_fixed_point_l534_53440

-- (Ⅰ) Prove that if m = 1/2 and point P (1/2, 2), the general form equation of line l is 2x - y + 1 = 0
theorem general_form_line_eq (m n : ℝ) (h1 : m = 1/2) (h2 : n = 1 / (1 - m)) (h3 : n = 2) (P : (ℝ × ℝ)) (hP : P = (1/2, 2)) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 1 := sorry

-- (Ⅱ) Prove that if point P(m,n) is on the line l0, then the line mx + (n-1)y + n + 5 = 0 passes through a fixed point, coordinates (1,1)
theorem line_passes_fixed_point (m n : ℝ) (h1 : m + 2 * n + 4 = 0) :
  ∀ (x y : ℝ), (m * x + (n - 1) * y + n + 5 = 0) ↔ (x = 1) ∧ (y = 1) := sorry

end general_form_line_eq_line_passes_fixed_point_l534_53440


namespace deliveries_conditions_l534_53465

variables (M P D : ℕ)
variables (MeMa MeBr MeQu MeBx: ℕ)

def distribution := (MeMa = 3 * MeBr) ∧ (MeBr = MeBr) ∧ (MeQu = MeBr) ∧ (MeBx = MeBr)

theorem deliveries_conditions 
  (h1 : P = 8 * M) 
  (h2 : D = 4 * M) 
  (h3 : M + P + D = 75) 
  (h4 : MeMa + MeBr + MeQu + MeBx = M)
  (h5 : distribution MeMa MeBr MeQu MeBx) :
  M = 5 ∧ MeMa = 2 ∧ MeBr = 1 ∧ MeQu = 1 ∧ MeBx = 1 :=
    sorry 

end deliveries_conditions_l534_53465


namespace probability_two_red_two_green_l534_53417

theorem probability_two_red_two_green (total_red total_blue total_green : ℕ)
  (total_marbles total_selected : ℕ) (probability : ℚ)
  (h_total_marbles: total_marbles = total_red + total_blue + total_green)
  (h_total_selected: total_selected = 4)
  (h_red_selected: 2 ≤ total_red)
  (h_green_selected: 2 ≤ total_green)
  (h_total_selected_le: total_selected ≤ total_marbles)
  (h_probability: probability = (Nat.choose total_red 2 * Nat.choose total_green 2) / (Nat.choose total_marbles total_selected))
  (h_total_red: total_red = 12)
  (h_total_blue: total_blue = 8)
  (h_total_green: total_green = 5):
  probability = 2 / 39 :=
by
  sorry

end probability_two_red_two_green_l534_53417


namespace find_C_plus_D_l534_53456

noncomputable def polynomial_divisible (x : ℝ) (C : ℝ) (D : ℝ) : Prop := 
  ∃ (ω : ℝ), ω^2 + ω + 1 = 0 ∧ ω^104 + C*ω + D = 0

theorem find_C_plus_D (C D : ℝ) : 
  (∃ x : ℝ, polynomial_divisible x C D) → C + D = 2 :=
by
  sorry

end find_C_plus_D_l534_53456


namespace condition_sufficient_not_necessary_l534_53426

theorem condition_sufficient_not_necessary
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B ↔ C)
  (h3 : C → D) :
  (A → D) ∧ ¬(D → A) :=
by
  sorry

end condition_sufficient_not_necessary_l534_53426


namespace felix_trees_per_sharpening_l534_53473

theorem felix_trees_per_sharpening (dollars_spent : ℕ) (cost_per_sharpen : ℕ) (trees_chopped : ℕ) 
  (h1 : dollars_spent = 35) (h2 : cost_per_sharpen = 5) (h3 : trees_chopped ≥ 91) :
  (91 / (35 / 5)) = 13 := 
by 
  sorry

end felix_trees_per_sharpening_l534_53473


namespace compute_infinite_series_l534_53482

noncomputable def infinite_series (c d : ℝ) (hcd : c > d) : ℝ :=
  ∑' n, 1 / (((n - 1 : ℝ) * c - (n - 2 : ℝ) * d) * (n * c - (n - 1 : ℝ) * d))

theorem compute_infinite_series (c d : ℝ) (hcd : c > d) :
  infinite_series c d hcd = 1 / ((c - d) * d) :=
by
  sorry

end compute_infinite_series_l534_53482


namespace express_308_million_in_scientific_notation_l534_53466

theorem express_308_million_in_scientific_notation :
    (308000000 : ℝ) = 3.08 * (10 ^ 8) :=
by
  sorry

end express_308_million_in_scientific_notation_l534_53466


namespace inequality_solution_empty_set_l534_53441

theorem inequality_solution_empty_set : ∀ x : ℝ, ¬ (x * (2 - x) > 3) :=
by
  -- Translate the condition and show that there are no x satisfying the inequality
  sorry

end inequality_solution_empty_set_l534_53441


namespace sin_2alpha_pos_if_tan_alpha_pos_l534_53462

theorem sin_2alpha_pos_if_tan_alpha_pos (α : ℝ) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_if_tan_alpha_pos_l534_53462


namespace necessary_but_not_sufficient_condition_l534_53470

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≤ 0

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, p x → q x a) ∧ ¬ (∀ x : ℝ, q x a → p x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end necessary_but_not_sufficient_condition_l534_53470


namespace no_integer_solutions_l534_53460

theorem no_integer_solutions (x y : ℤ) : 19 * x^3 - 84 * y^2 ≠ 1984 :=
by
  sorry

end no_integer_solutions_l534_53460


namespace total_steps_eliana_walked_l534_53432

-- Define the conditions of the problem.
def first_day_exercise_steps : Nat := 200
def first_day_additional_steps : Nat := 300
def second_day_multiplier : Nat := 2
def third_day_additional_steps : Nat := 100

-- Define the steps calculation for each day.
def first_day_total_steps : Nat := first_day_exercise_steps + first_day_additional_steps
def second_day_total_steps : Nat := second_day_multiplier * first_day_total_steps
def third_day_total_steps : Nat := second_day_total_steps + third_day_additional_steps

-- Prove that the total number of steps Eliana walked during these three days is 1600.
theorem total_steps_eliana_walked :
  first_day_total_steps + second_day_total_steps + third_day_additional_steps = 1600 :=
by
  -- Conditional values are constants. We can use Lean's deterministic evaluator here.
  -- Hence, there's no need to write out full proof for now. Using sorry to bypass actual proof.
  sorry

end total_steps_eliana_walked_l534_53432


namespace mona_biked_monday_l534_53415

-- Define the constants and conditions
def distance_biked_weekly : ℕ := 30
def distance_biked_wednesday : ℕ := 12
def speed_flat_road : ℕ := 15
def speed_reduction_percentage : ℕ := 20

-- Define the main problem and conditions in Lean
theorem mona_biked_monday (M : ℕ)
  (h1 : 2 * M + distance_biked_wednesday + M = distance_biked_weekly)  -- total distance biked in the week
  (h2 : 2 * M * (100 - speed_reduction_percentage) / 100 / 15 = 2 * M / 12)  -- speed reduction effect
  : M = 6 :=
sorry 

end mona_biked_monday_l534_53415


namespace yoonseok_handshakes_l534_53416

-- Conditions
def totalFriends : ℕ := 12
def yoonseok := "Yoonseok"
def adjacentFriends (i : ℕ) : Prop := i = 1 ∨ i = (totalFriends - 1)

-- Problem Statement
theorem yoonseok_handshakes : 
  ∀ (totalFriends : ℕ) (adjacentFriends : ℕ → Prop), 
    totalFriends = 12 → 
    (∀ i, adjacentFriends i ↔ i = 1 ∨ i = (totalFriends - 1)) → 
    (totalFriends - 1 - 2 = 9) := by
  intros totalFriends adjacentFriends hTotal hAdjacent
  have hSub : totalFriends - 1 - 2 = 9 := by sorry
  exact hSub

end yoonseok_handshakes_l534_53416


namespace smallest_degree_of_f_l534_53488

theorem smallest_degree_of_f (p : Polynomial ℂ) (hp_deg : p.degree < 1992)
  (hp0 : p.eval 0 ≠ 0) (hp1 : p.eval 1 ≠ 0) (hp_1 : p.eval (-1) ≠ 0) :
  ∃ f g : Polynomial ℂ, 
    (Polynomial.derivative^[1992] (p / (X^3 - X))) = f / g ∧ f.degree = 3984 := 
sorry

end smallest_degree_of_f_l534_53488


namespace student_rank_from_left_l534_53492

theorem student_rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_right = 16) 
  (h3 : total_students = rank_from_right + rank_from_left - 1) 
  : rank_from_left = 6 := 
by 
  sorry

end student_rank_from_left_l534_53492


namespace inequality_sum_l534_53418

theorem inequality_sum {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) : a + c > b + d :=
by
  sorry

end inequality_sum_l534_53418


namespace completion_days_together_l534_53405

-- Definitions based on given conditions
variable (W : ℝ) -- Total work
variable (A : ℝ) -- Work done by A in one day
variable (B : ℝ) -- Work done by B in one day

-- Condition 1: A alone completes the work in 20 days
def work_done_by_A := A = W / 20

-- Condition 2: A and B working with B half a day complete the work in 15 days
def work_done_by_A_and_half_B := A + (1 / 2) * B = W / 15

-- Prove: A and B together will complete the work in 60 / 7 days if B works full time
theorem completion_days_together (h1 : work_done_by_A W A) (h2 : work_done_by_A_and_half_B W A B) :
  ∃ D : ℝ, D = 60 / 7 :=
by 
  sorry

end completion_days_together_l534_53405


namespace sequence_a4_value_l534_53431

theorem sequence_a4_value :
  ∀ {a : ℕ → ℚ}, (a 1 = 3) → ((∀ n, a (n + 1) = 3 * a n / (a n + 3))) → (a 4 = 3 / 4) :=
by
  intros a h1 hRec
  sorry

end sequence_a4_value_l534_53431


namespace not_perfect_square_7_pow_2025_all_others_perfect_squares_l534_53400

theorem not_perfect_square_7_pow_2025 :
  ¬ (∃ x : ℕ, x^2 = 7^2025) :=
sorry

theorem all_others_perfect_squares :
  (∃ x : ℕ, x^2 = 6^2024) ∧
  (∃ x : ℕ, x^2 = 8^2026) ∧
  (∃ x : ℕ, x^2 = 9^2027) ∧
  (∃ x : ℕ, x^2 = 10^2028) :=
sorry

end not_perfect_square_7_pow_2025_all_others_perfect_squares_l534_53400


namespace expand_product_l534_53468

theorem expand_product (x : ℝ) : (x + 2) * (x^2 + 3 * x + 4) = x^3 + 5 * x^2 + 10 * x + 8 := 
by
  sorry

end expand_product_l534_53468


namespace price_per_kilo_of_bananas_l534_53427

def initial_money : ℕ := 500
def potatoes_cost : ℕ := 6 * 2
def tomatoes_cost : ℕ := 9 * 3
def cucumbers_cost : ℕ := 5 * 4
def bananas_weight : ℕ := 3
def remaining_money : ℕ := 426

-- Defining total cost of all items
def total_item_cost : ℕ := initial_money - remaining_money

-- Defining the total cost of bananas
def cost_bananas : ℕ := total_item_cost - (potatoes_cost + tomatoes_cost + cucumbers_cost)

-- Final question: Prove that the price per kilo of bananas is $5
theorem price_per_kilo_of_bananas : cost_bananas / bananas_weight = 5 :=
by
  sorry

end price_per_kilo_of_bananas_l534_53427


namespace relationship_of_ys_l534_53472

variable (f : ℝ → ℝ)

def inverse_proportion := ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k / x

theorem relationship_of_ys
  (h_inv_prop : inverse_proportion f)
  (h_pts1 : f (-2) = 3)
  (h_pts2 : f (-3) = y₁)
  (h_pts3 : f 1 = y₂)
  (h_pts4 : f 2 = y₃) :
  y₂ < y₃ ∧ y₃ < y₁ :=
sorry

end relationship_of_ys_l534_53472


namespace wage_difference_l534_53446

variable (P Q : ℝ)
variable (h : ℝ)
axiom wage_relation : P = 1.5 * Q
axiom time_relation : 360 = P * h
axiom time_relation_q : 360 = Q * (h + 10)

theorem wage_difference : P - Q = 6 :=
  by
  sorry

end wage_difference_l534_53446


namespace new_planet_volume_eq_l534_53430

noncomputable def volume_of_new_planet (V_earth : ℝ) (scaling_factor : ℝ) : ℝ :=
  V_earth * (scaling_factor^3)

theorem new_planet_volume_eq 
  (V_earth : ℝ)
  (scaling_factor : ℝ)
  (hV_earth : V_earth = 1.08 * 10^12)
  (h_scaling_factor : scaling_factor = 10^4) :
  volume_of_new_planet V_earth scaling_factor = 1.08 * 10^24 :=
by
  sorry

end new_planet_volume_eq_l534_53430


namespace Collin_total_petals_l534_53495

-- Definitions of the conditions
def initial_flowers_Collin : ℕ := 25
def flowers_Ingrid : ℕ := 33
def petals_per_flower : ℕ := 4
def third_of_flowers_Ingrid : ℕ := flowers_Ingrid / 3

-- Total number of flowers Collin has after receiving from Ingrid
def total_flowers_Collin : ℕ := initial_flowers_Collin + third_of_flowers_Ingrid

-- Total number of petals Collin has
def total_petals_Collin : ℕ := total_flowers_Collin * petals_per_flower

-- The theorem to be proved
theorem Collin_total_petals : total_petals_Collin = 144 := by
  -- Proof goes here
  sorry

end Collin_total_petals_l534_53495


namespace find_c_k_l534_53477

-- Definitions of the arithmetic and geometric sequences
def a (n d : ℕ) := 1 + (n - 1) * d
def b (n r : ℕ) := r ^ (n - 1)
def c (n d r : ℕ) := a n d + b n r

-- Conditions for the specific problem
theorem find_c_k (k d r : ℕ) (h1 : 1 + (k - 2) * d + r ^ (k - 2) = 150) (h2 : 1 + k * d + r ^ k = 1500) : c k d r = 314 :=
by
  sorry

end find_c_k_l534_53477


namespace annual_income_of_A_l534_53437

theorem annual_income_of_A 
  (ratio_AB : ℕ → ℕ → Prop)
  (income_C : ℕ)
  (income_B_more_C : ℕ → ℕ → Prop)
  (income_B_from_ratio : ℕ → ℕ → Prop)
  (income_C_value : income_C = 16000)
  (income_B_condition : ∀ c, income_B_more_C 17920 c)
  (income_A_condition : ∀ b, ratio_AB 5 (b/2))
  : ∃ a, a = 537600 :=
by
  sorry

end annual_income_of_A_l534_53437


namespace vanessa_recycled_correct_l534_53447

-- Define conditions as separate hypotheses
variable (weight_per_point : ℕ := 9)
variable (points_earned : ℕ := 4)
variable (friends_recycled : ℕ := 16)

-- Define the total weight recycled as points earned times the weight per point
def total_weight_recycled (points_earned weight_per_point : ℕ) : ℕ := points_earned * weight_per_point

-- Define the weight recycled by Vanessa
def vanessa_recycled (total_recycled friends_recycled : ℕ) : ℕ := total_recycled - friends_recycled

-- Main theorem statement
theorem vanessa_recycled_correct (weight_per_point points_earned friends_recycled : ℕ) 
    (hw : weight_per_point = 9) (hp : points_earned = 4) (hf : friends_recycled = 16) : 
    vanessa_recycled (total_weight_recycled points_earned weight_per_point) friends_recycled = 20 := 
by 
  sorry

end vanessa_recycled_correct_l534_53447


namespace larger_number_of_ratio_and_lcm_l534_53464

theorem larger_number_of_ratio_and_lcm (x : ℕ) (h1 : (2 * x) % (5 * x) = 160) : (5 * x) = 160 := by
  sorry

end larger_number_of_ratio_and_lcm_l534_53464


namespace initial_plank_count_l534_53404

def Bedroom := 8
def LivingRoom := 20
def Kitchen := 11
def DiningRoom := 13
def Hallway := 4
def GuestBedroom := Bedroom - 2
def Study := GuestBedroom + 3
def BedroomReplacements := 3
def LivingRoomReplacements := 2
def StudyReplacements := 1
def LeftoverPlanks := 7

def TotalPlanksUsed := 
  (Bedroom + BedroomReplacements) +
  (LivingRoom + LivingRoomReplacements) +
  (Kitchen) +
  (DiningRoom) +
  (GuestBedroom + BedroomReplacements) +
  (Hallway * 2) +
  (Study + StudyReplacements)

theorem initial_plank_count : 
  TotalPlanksUsed + LeftoverPlanks = 91 := 
by
  sorry

end initial_plank_count_l534_53404


namespace no_positive_a_for_inequality_l534_53414

theorem no_positive_a_for_inequality (a : ℝ) (h : 0 < a) : 
  ¬ ∀ x : ℝ, |Real.cos x| + |Real.cos (a * x)| > Real.sin x + Real.sin (a * x) := by
  sorry

end no_positive_a_for_inequality_l534_53414


namespace rhombus_side_length_l534_53448

theorem rhombus_side_length (total_length : ℕ) (num_sides : ℕ) (h1 : total_length = 32) (h2 : num_sides = 4) :
    total_length / num_sides = 8 :=
by
  -- Proof will be provided here
  sorry

end rhombus_side_length_l534_53448


namespace unique_solution_l534_53402

noncomputable def check_triplet (a b c : ℕ) : Prop :=
  5^a + 3^b - 2^c = 32

theorem unique_solution : ∀ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ check_triplet a b c ↔ (a = 2 ∧ b = 2 ∧ c = 1) :=
  by sorry

end unique_solution_l534_53402


namespace product_of_coordinates_of_D_l534_53443

theorem product_of_coordinates_of_D 
  (x y : ℝ)
  (midpoint_x : (5 + x) / 2 = 4)
  (midpoint_y : (3 + y) / 2 = 7) : 
  x * y = 33 := 
by 
  sorry

end product_of_coordinates_of_D_l534_53443


namespace soybean_cornmeal_proof_l534_53409

theorem soybean_cornmeal_proof :
  ∃ (x y : ℝ), 
    (0.14 * x + 0.07 * y = 0.13 * 280) ∧
    (x + y = 280) ∧
    (x = 240) ∧
    (y = 40) :=
by
  sorry

end soybean_cornmeal_proof_l534_53409


namespace num_factors_of_90_multiple_of_6_l534_53429

def is_factor (m n : ℕ) : Prop := n % m = 0
def is_multiple_of (m n : ℕ) : Prop := n % m = 0

theorem num_factors_of_90_multiple_of_6 : 
  ∃ (count : ℕ), count = 4 ∧ ∀ x, is_factor x 90 → is_multiple_of 6 x → x > 0 :=
sorry

end num_factors_of_90_multiple_of_6_l534_53429


namespace total_time_taken_l534_53459

theorem total_time_taken (b km : ℝ) : 
  (b / 50 + km / 80) = (8 * b + 5 * km) / 400 := 
sorry

end total_time_taken_l534_53459


namespace maximum_bags_of_milk_l534_53487

theorem maximum_bags_of_milk (bag_cost : ℚ) (promotion : ℕ → ℕ) (total_money : ℚ) 
  (h1 : bag_cost = 2.5) 
  (h2 : promotion 2 = 3) 
  (h3 : total_money = 30) : 
  ∃ n, n = 18 ∧ (total_money >= n * bag_cost - (n / 3) * bag_cost) :=
by
  sorry

end maximum_bags_of_milk_l534_53487


namespace equation_pattern_l534_53494

theorem equation_pattern (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 :=
by
  sorry

end equation_pattern_l534_53494


namespace temperature_on_fourth_day_l534_53438

theorem temperature_on_fourth_day
  (t₁ t₂ t₃ : ℤ) 
  (avg : ℤ)
  (h₁ : t₁ = -36) 
  (h₂ : t₂ = 13) 
  (h₃ : t₃ = -10) 
  (h₄ : avg = -12) 
  : ∃ t₄ : ℤ, t₄ = -15 :=
by
  sorry

end temperature_on_fourth_day_l534_53438


namespace solution_l534_53411

def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^5 + p*x^3 + q*x - 8

theorem solution (p q : ℝ) (h : f (-2) p q = 10) : f 2 p q = -26 := by
  sorry

end solution_l534_53411


namespace triangle_altitude_length_l534_53478

variable (AB AC BC BA1 AA1 : ℝ)
variable (eq1 : AB = 8)
variable (eq2 : AC = 10)
variable (eq3 : BC = 12)

theorem triangle_altitude_length (h : ∃ AA1, AA1 * AA1 + BA1 * BA1 = 64 ∧ 
                                AA1 * AA1 + (BC - BA1) * (BC - BA1) = 100) :
    BA1 = 4.5 := by
  sorry 

end triangle_altitude_length_l534_53478


namespace stream_speed_l534_53434

theorem stream_speed (v : ℝ) : (24 + v) = 168 / 6 → v = 4 :=
by
  intro h
  sorry

end stream_speed_l534_53434
