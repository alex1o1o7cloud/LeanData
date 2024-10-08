import Mathlib

namespace rectangle_area_l57_57899

theorem rectangle_area (x : ℝ) (h : (x - 3) * (2 * x + 3) = 4 * x - 9) : x = 7 / 2 :=
sorry

end rectangle_area_l57_57899


namespace min_value_of_expr_l57_57722

theorem min_value_of_expr (a : ℝ) (h : a > 3) : ∃ m, (∀ b > 3, b + 4 / (b - 3) ≥ m) ∧ m = 7 :=
sorry

end min_value_of_expr_l57_57722


namespace final_selling_price_l57_57317

-- Conditions
variable (x : ℝ)
def original_price : ℝ := x
def first_discount : ℝ := 0.8 * x
def additional_reduction : ℝ := 10

-- Statement of the problem
theorem final_selling_price (x : ℝ) : (0.8 * x) - 10 = 0.8 * x - 10 :=
by sorry

end final_selling_price_l57_57317


namespace binary_addition_and_subtraction_correct_l57_57394

def add_binary_and_subtract : ℕ :=
  let n1 := 0b1101  -- binary for 1101_2
  let n2 := 0b0010  -- binary for 10_2
  let n3 := 0b0101  -- binary for 101_2
  let n4 := 0b1011  -- expected result 1011_2
  n1 + n2 + n3 - 0b0011  -- subtract binary for 11_2

theorem binary_addition_and_subtraction_correct : add_binary_and_subtract = 0b1011 := 
by 
  sorry

end binary_addition_and_subtraction_correct_l57_57394


namespace quadratic_other_root_l57_57883

theorem quadratic_other_root (a : ℝ) (h1 : ∃ (x : ℝ), x^2 - 2 * x + a = 0 ∧ x = -1) :
  ∃ (x2 : ℝ), x2^2 - 2 * x2 + a = 0 ∧ x2 = 3 :=
sorry

end quadratic_other_root_l57_57883


namespace trajectory_of_midpoint_l57_57496

theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) :
  (y₀ = 2 * x₀ ^ 2 + 1) ∧ (x = (x₀ + 0) / 2) ∧ (y = (y₀ + 1) / 2) →
  y = 4 * x ^ 2 + 1 :=
by sorry

end trajectory_of_midpoint_l57_57496


namespace find_mn_l57_57627

theorem find_mn (m n : ℕ) (h : m > 0 ∧ n > 0) (eq1 : m^2 + n^2 + 4 * m - 46 = 0) :
  mn = 5 ∨ mn = 15 := by
  sorry

end find_mn_l57_57627


namespace geometric_sequence_product_l57_57329

theorem geometric_sequence_product (b : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, b (n+1) = b n * r)
  (h_b9 : b 9 = (3 + 5) / 2) : b 1 * b 17 = 16 :=
by
  sorry

end geometric_sequence_product_l57_57329


namespace greatest_length_of_cords_l57_57537

theorem greatest_length_of_cords (a b c : ℝ) (h₁ : a = Real.sqrt 20) (h₂ : b = Real.sqrt 50) (h₃ : c = Real.sqrt 98) :
  ∃ (d : ℝ), d = 1 ∧ ∀ (k : ℝ), (k = a ∨ k = b ∨ k = c) → ∃ (n m : ℕ), k = d * (n : ℝ) ∧ d * (m : ℝ) = (m : ℝ) := by
sorry

end greatest_length_of_cords_l57_57537


namespace books_written_by_Zig_l57_57255

theorem books_written_by_Zig (F Z : ℕ) (h1 : Z = 4 * F) (h2 : F + Z = 75) : Z = 60 := by
  sorry

end books_written_by_Zig_l57_57255


namespace math_problem_l57_57962

theorem math_problem (x : ℤ) (h : x = 9) :
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 :=
by
  sorry

end math_problem_l57_57962


namespace inverse_implies_negation_l57_57850

-- Let's define p as a proposition
variable (p : Prop)

-- The inverse of a proposition p, typically the implication of not p implies not q
def inverse (p q : Prop) := ¬p → ¬q

-- The negation of a proposition p is just ¬p
def negation (p : Prop) := ¬p

-- The math problem statement. Prove that if the inverse of p is true, the negation of p is true.
theorem inverse_implies_negation (q : Prop) (h : inverse p q) : negation q := by
  sorry

end inverse_implies_negation_l57_57850


namespace speed_of_A_l57_57177

theorem speed_of_A :
  ∀ (v_A : ℝ), 
    (v_A * 2 + 7 * 2 = 24) → 
    v_A = 5 :=
by
  intro v_A
  intro h
  have h1 : v_A * 2 = 10 := by linarith
  have h2 : v_A = 5 := by linarith
  exact h2

end speed_of_A_l57_57177


namespace candy_cost_l57_57012

theorem candy_cost (C : ℝ) 
  (h1 : 20 * C + 80 * 5 = 100 * 6) : 
  C = 10 := 
by
  sorry

end candy_cost_l57_57012


namespace sequence_product_l57_57836

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def a4_value (a : ℕ → ℕ) : Prop :=
a 4 = 2

-- The statement to be proven
theorem sequence_product (a : ℕ → ℕ) (q : ℕ) (h_geo_seq : geometric_sequence a q) (h_a4 : a4_value a) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l57_57836


namespace letters_containing_only_dot_l57_57707

theorem letters_containing_only_dot (DS S_only : ℕ) (total : ℕ) (h1 : DS = 20) (h2 : S_only = 36) (h3 : total = 60) :
  total - (DS + S_only) = 4 :=
by
  sorry

end letters_containing_only_dot_l57_57707


namespace integer_coordinates_point_exists_l57_57035

theorem integer_coordinates_point_exists (p q : ℤ) (h : p^2 - 4 * q = 0) :
  ∃ a b : ℤ, b = a^2 + p * a + q ∧ (a = -p ∧ b = q) ∧ (a ≠ -p → (a = p ∧ b = q) → (p^2 - 4 * b = 0)) :=
by
  sorry

end integer_coordinates_point_exists_l57_57035


namespace arithmetic_sequence_30th_term_l57_57230

theorem arithmetic_sequence_30th_term :
  let a₁ := 4
  let d₁ := 6
  let n := 30
  (a₁ + (n - 1) * d₁) = 178 :=
by
  sorry

end arithmetic_sequence_30th_term_l57_57230


namespace arithmetic_expression_proof_l57_57243

theorem arithmetic_expression_proof : 4 * 6 * 8 + 18 / 3 ^ 2 = 194 := by
  sorry

end arithmetic_expression_proof_l57_57243


namespace processing_rates_and_total_cost_l57_57512

variables (products total_days total_days_A total_days_B daily_capacity_A daily_capacity_B total_cost_A total_cost_B : ℝ)

noncomputable def A_processing_rate : ℝ := daily_capacity_A
noncomputable def B_processing_rate : ℝ := daily_capacity_B

theorem processing_rates_and_total_cost
  (h1 : products = 1000)
  (h2 : total_days_A = total_days_B + 10)
  (h3 : daily_capacity_B = 1.25 * daily_capacity_A)
  (h4 : total_cost_A = 100 * total_days_A)
  (h5 : total_cost_B = 125 * total_days_B) :
  (daily_capacity_A = 20) ∧ (daily_capacity_B = 25) ∧ (total_cost_A + total_cost_B = 5000) :=
by
  sorry

end processing_rates_and_total_cost_l57_57512


namespace find_a_for_odd_function_l57_57234

noncomputable def f (a x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem find_a_for_odd_function (a : ℝ) :
  (∀ x : ℝ, f a x + f a (-x) = 0) ↔ a = -1 := sorry

end find_a_for_odd_function_l57_57234


namespace find_n_l57_57703

theorem find_n {n : ℕ} (avg1 : ℕ) (avg2 : ℕ) (S : ℕ) :
  avg1 = 7 →
  avg2 = 6 →
  S = 7 * n →
  6 = (S - 11) / (n + 1) →
  n = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end find_n_l57_57703


namespace find_a10_l57_57037

noncomputable def ladder_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), (a (n + 3))^2 = a n * a (n + 6)

theorem find_a10 {a : ℕ → ℝ} (h1 : ladder_geometric_sequence a) 
(h2 : a 1 = 1) 
(h3 : a 4 = 2) : a 10 = 8 :=
sorry

end find_a10_l57_57037


namespace triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l57_57328

-- Given the conditions: two sides of one triangle are equal to two sides of another triangle.
-- And an angle opposite to one of these sides is equal to the angle opposite to the corresponding side.
variables {A B C D E F : Type}
variables {AB DE BC EF : ℝ} (h_AB_DE : AB = DE) (h_BC_EF : BC = EF)
variables {angle_A angle_D : ℝ} (h_angle_A_D : angle_A = angle_D)

-- Prove that the triangles may or may not be congruent
theorem triangles_may_or_may_not_be_congruent :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_be_congruent_or_not : Prop) :=
sorry

-- Prove that the triangles may have equal areas
theorem triangles_may_have_equal_areas :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_have_equal_areas : Prop) :=
sorry

end triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l57_57328


namespace slower_ball_speed_l57_57725

open Real

variables (v u C : ℝ)

theorem slower_ball_speed :
  (20 * (v - u) = C) → (4 * (v + u) = C) → ((v + u) * 3 = 75) → u = 10 :=
by
  intros h1 h2 h3
  sorry

end slower_ball_speed_l57_57725


namespace remainder_eval_at_4_l57_57117

def p : ℚ → ℚ := sorry

def r (x : ℚ) : ℚ := sorry

theorem remainder_eval_at_4 :
  (p 1 = 2) →
  (p 3 = 5) →
  (p (-2) = -2) →
  (∀ x, ∃ q : ℚ → ℚ, p x = (x - 1) * (x - 3) * (x + 2) * q x + r x) →
  r 4 = 38 / 7 :=
sorry

end remainder_eval_at_4_l57_57117


namespace inverse_at_neg_two_l57_57389

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_at_neg_two :
  g (-2) = -43 :=
by
  -- sorry here to skip the proof, as instructed.
  sorry

end inverse_at_neg_two_l57_57389


namespace digit_7_occurrences_in_range_20_to_199_l57_57517

open Set

noncomputable def countDigitOccurrences (low high : ℕ) (digit : ℕ) : ℕ :=
  sorry

theorem digit_7_occurrences_in_range_20_to_199 : 
  countDigitOccurrences 20 199 7 = 38 := 
by
  sorry

end digit_7_occurrences_in_range_20_to_199_l57_57517


namespace original_price_l57_57908

-- Definitions based on the conditions
def selling_price : ℝ := 1080
def gain_percent : ℝ := 80

-- The proof problem: Prove that the cost price is Rs. 600
theorem original_price (CP : ℝ) (h_sp : CP + CP * (gain_percent / 100) = selling_price) : CP = 600 :=
by
  -- We skip the proof itself
  sorry

end original_price_l57_57908


namespace initial_population_l57_57282

theorem initial_population (P : ℝ) (h : (0.9 : ℝ)^2 * P = 4860) : P = 6000 :=
by
  sorry

end initial_population_l57_57282


namespace stagePlayRolesAssignment_correct_l57_57002

noncomputable def stagePlayRolesAssignment : ℕ :=
  let male_roles : ℕ := 4 * 3 -- ways to assign male roles
  let female_roles : ℕ := 5 * 4 -- ways to assign female roles
  let either_gender_roles : ℕ := 5 * 4 * 3 -- ways to assign either-gender roles
  male_roles * female_roles * either_gender_roles -- total assignments

theorem stagePlayRolesAssignment_correct : stagePlayRolesAssignment = 14400 := by
  sorry

end stagePlayRolesAssignment_correct_l57_57002


namespace combined_weight_of_daughter_and_child_l57_57851

theorem combined_weight_of_daughter_and_child 
  (G D C : ℝ)
  (h1 : G + D + C = 110)
  (h2 : C = 1/5 * G)
  (h3 : D = 50) :
  D + C = 60 :=
sorry

end combined_weight_of_daughter_and_child_l57_57851


namespace no_zonk_probability_l57_57484

theorem no_zonk_probability (Z C G : ℕ) (total_boxes : ℕ := 3) (tables : ℕ := 3)
  (no_zonk_prob : ℚ := 2 / 3) : (no_zonk_prob ^ tables) = 8 / 27 :=
by
  -- Here we would prove the theorem, but for the purpose of this task, we skip the proof.
  sorry

end no_zonk_probability_l57_57484


namespace largest_divisor_of_n_l57_57607

theorem largest_divisor_of_n (n : ℕ) (h_pos: n > 0) (h_div: 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l57_57607


namespace find_m_l57_57964

theorem find_m (x y m : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : 3 * x - 4 * (m - 1) * y + 30 = 0) : m = -2 :=
by
  sorry

end find_m_l57_57964


namespace seven_pow_k_eq_two_l57_57218

theorem seven_pow_k_eq_two {k : ℕ} (h : 7 ^ (4 * k + 2) = 784) : 7 ^ k = 2 := 
by 
  sorry

end seven_pow_k_eq_two_l57_57218


namespace problem1_problem2_l57_57041

-- Problem 1
theorem problem1 : 2023^2 - 2024 * 2022 = 1 :=
sorry

-- Problem 2
variables (a b c : ℝ)
theorem problem2 : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c :=
sorry

end problem1_problem2_l57_57041


namespace no_integer_pairs_satisfy_equation_l57_57793

theorem no_integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), a^3 + 3 * a^2 + 2 * a ≠ 125 * b^3 + 75 * b^2 + 15 * b + 2 :=
by
  intro a b
  sorry

end no_integer_pairs_satisfy_equation_l57_57793


namespace number_of_paintings_l57_57124

def is_valid_painting (grid : Matrix (Fin 3) (Fin 3) Bool) : Prop :=
  ∀ i j, grid i j = true → 
    (∀ k, k.succ < 3 → grid k j = true → ¬ grid (k.succ) j = false) ∧
    (∀ l, l.succ < 3 → grid i l = true → ¬ grid i (l.succ) = false)

theorem number_of_paintings : 
  ∃ n, n = 50 ∧ 
       ∃ f : Finset (Matrix (Fin 3) (Fin 3) Bool), 
         (∀ grid ∈ f, is_valid_painting grid) ∧ 
         Finset.card f = n :=
sorry

end number_of_paintings_l57_57124


namespace train_passing_time_l57_57324

theorem train_passing_time
  (length_A : ℝ) (length_B : ℝ) (time_A : ℝ) (speed_B : ℝ) 
  (Dir_opposite : true) 
  (passenger_on_A_time : time_A = 10)
  (length_of_A : length_A = 150)
  (length_of_B : length_B = 200)
  (relative_speed : speed_B = length_B / time_A) :
  ∃ x : ℝ, length_A / x = length_B / time_A ∧ x = 7.5 :=
by
  -- conditions stated
  sorry

end train_passing_time_l57_57324


namespace sequence_sum_general_term_l57_57180

theorem sequence_sum_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n) : ∀ n, a n = 2 * n :=
by 
  sorry

end sequence_sum_general_term_l57_57180


namespace part1_part2_l57_57843

-- Part 1
theorem part1 (n : ℕ) (hn : n ≠ 0) (d : ℕ) (hd : d ∣ 2 * n^2) : 
  ∀ m : ℕ, ¬ (m ≠ 0 ∧ m^2 = n^2 + d) :=
by
  sorry 

-- Part 2
theorem part2 (n : ℕ) (hn : n ≠ 0) : 
  ∀ d : ℕ, (d ∣ 3 * n^2 ∧ ∃ m : ℕ, m ≠ 0 ∧ m^2 = n^2 + d) → d = 3 * n^2 :=
by
  sorry

end part1_part2_l57_57843


namespace smallest_possible_sector_angle_l57_57316

theorem smallest_possible_sector_angle : ∃ a₁ d : ℕ, 2 * a₁ + 9 * d = 72 ∧ a₁ = 9 :=
by
  sorry

end smallest_possible_sector_angle_l57_57316


namespace minimum_value_of_expression_l57_57887

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ m : ℝ, (m = 8) ∧ (∀ z : ℝ, z = (y / x) + (4 / y) → z ≥ m) :=
sorry

end minimum_value_of_expression_l57_57887


namespace original_planned_length_l57_57999

theorem original_planned_length (x : ℝ) (h1 : x > 0) (total_length : ℝ := 3600) (efficiency_ratio : ℝ := 1.8) (time_saved : ℝ := 20) 
  (h2 : total_length / x - total_length / (efficiency_ratio * x) = time_saved) :
  x = 80 :=
sorry

end original_planned_length_l57_57999


namespace complement_of_M_l57_57598

open Set

def M : Set ℝ := { x | (2 - x) / (x + 3) < 0 }

theorem complement_of_M : (Mᶜ = { x : ℝ | -3 ≤ x ∧ x ≤ 2 }) :=
by
  sorry

end complement_of_M_l57_57598


namespace probability_exactly_one_each_is_correct_l57_57816

def probability_one_each (total forks spoons knives teaspoons : ℕ) : ℚ :=
  (forks * spoons * knives * teaspoons : ℚ) / ((total.choose 4) : ℚ)

theorem probability_exactly_one_each_is_correct :
  probability_one_each 34 8 9 10 7 = 40 / 367 :=
by sorry

end probability_exactly_one_each_is_correct_l57_57816


namespace smallest_number_divisible_l57_57464

theorem smallest_number_divisible (n : ℕ) : (∃ n : ℕ, (n + 3) % 27 = 0 ∧ (n + 3) % 35 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0) ∧ n = 4722 :=
by
  sorry

end smallest_number_divisible_l57_57464


namespace original_number_of_men_l57_57040

/-- 
Given:
1. A group of men decided to do a work in 20 days,
2. When 2 men became absent, the remaining men did the work in 22 days,

Prove:
The original number of men in the group was 22.
-/
theorem original_number_of_men (x : ℕ) (h : 20 * x = 22 * (x - 2)) : x = 22 :=
by
  sorry

end original_number_of_men_l57_57040


namespace find_point_B_l57_57812

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (-3, -1)
def line_y_eq_2x (x : ℝ) : ℝ × ℝ := (x, 2 * x)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1 

theorem find_point_B (B : ℝ × ℝ) (hB : B = line_y_eq_2x B.1) (h_parallel : is_parallel (B.1 + 3, B.2 + 1) vector_a) :
  B = (2, 4) := 
  sorry

end find_point_B_l57_57812


namespace calculate_E_l57_57292

theorem calculate_E (P J T B A E : ℝ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : B = 1.40 * J)
  (h4 : A = 0.85 * B)
  (h5 : T = P - (E / 100) * P)
  (h6 : E = 2 * ((P - A) / P) * 100) : 
  E = 21.5 := 
sorry

end calculate_E_l57_57292


namespace height_percentage_differences_l57_57662

variable (B : ℝ) (A : ℝ) (R : ℝ)
variable (h1 : A = 1.25 * B) (h2 : R = 1.0625 * B)

theorem height_percentage_differences :
  (100 * (A - B) / B = 25) ∧
  (100 * (A - R) / A = 15) ∧
  (100 * (R - B) / B = 6.25) :=
by
  sorry

end height_percentage_differences_l57_57662


namespace number_of_people_in_group_is_21_l57_57458

-- Definitions based directly on the conditions
def pins_contribution_per_day := 10
def pins_deleted_per_week_per_person := 5
def group_initial_pins := 1000
def final_pins_after_month := 6600
def weeks_in_a_month := 4

-- To be proved: number of people in the group is 21
theorem number_of_people_in_group_is_21 (P : ℕ)
  (h1 : final_pins_after_month - group_initial_pins = 5600)
  (h2 : weeks_in_a_month * (pins_contribution_per_day * 7 - pins_deleted_per_week_per_person) = 260)
  (h3 : 5600 / 260 = 21) :
  P = 21 := 
sorry

end number_of_people_in_group_is_21_l57_57458


namespace problem_area_triangle_PNT_l57_57946

noncomputable def area_triangle_PNT (PQ QR x : ℝ) : ℝ :=
  let PS := Real.sqrt (PQ^2 + QR^2)
  let PN := PS / 2
  let area := (PN * Real.sqrt (61 - x^2)) / 4
  area

theorem problem_area_triangle_PNT :
  ∀ (PQ QR : ℝ) (x : ℝ), PQ = 10 → QR = 12 → 0 ≤ x ∧ x ≤ 10 → area_triangle_PNT PQ QR x = 
  (Real.sqrt (244) * Real.sqrt (61 - x^2)) / 4 :=
by
  intros PQ QR x hPQ hQR hx
  sorry

end problem_area_triangle_PNT_l57_57946


namespace number_of_common_points_l57_57562

-- Define the circle equation
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Define the vertical line equation
def is_on_line (x : ℝ) : Prop :=
  x = 3

-- Prove that the number of distinct points common to both graphs is two
theorem number_of_common_points : 
  ∃ y1 y2 : ℝ, is_on_circle 3 y1 ∧ is_on_circle 3 y2 ∧ y1 ≠ y2 :=
by {
  sorry
}

end number_of_common_points_l57_57562


namespace range_of_x1_f_x2_l57_57396

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 * Real.exp 1 else Real.exp x / x^2

theorem range_of_x1_f_x2:
  ∃ (x1 x2 : ℝ), x1 ≤ 0 ∧ 0 < x2 ∧ f x1 = f x2 ∧ -4 * (Real.exp 1)^2 ≤ x1 * f x2 ∧ x1 * f x2 ≤ 0 :=
sorry

end range_of_x1_f_x2_l57_57396


namespace rowing_speed_downstream_correct_l57_57919

/-- Given:
- The speed of the man upstream V_upstream is 20 kmph.
- The speed of the man in still water V_man is 40 kmph.
Prove:
- The speed of the man rowing downstream V_downstream is 60 kmph.
-/
def rowing_speed_downstream : Prop :=
  let V_upstream := 20
  let V_man := 40
  let V_s := V_man - V_upstream
  let V_downstream := V_man + V_s
  V_downstream = 60

theorem rowing_speed_downstream_correct : rowing_speed_downstream := by
  sorry

end rowing_speed_downstream_correct_l57_57919


namespace rancher_unique_solution_l57_57092

-- Defining the main problem statement
theorem rancher_unique_solution : ∃! (b h : ℕ), 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end rancher_unique_solution_l57_57092


namespace distance_is_correct_l57_57241

noncomputable def distance_from_home_to_forest_park : ℝ := 11  -- distance in kilometers

structure ProblemData where
  v : ℝ                  -- Xiao Wu's bicycling speed (in meters per minute)
  t_catch_up : ℝ          -- time it takes for father to catch up (in minutes)
  d_forest : ℝ            -- distance from catch-up point to forest park (in kilometers)
  t_remaining : ℝ        -- time remaining for Wu to reach park after wallet delivered (in minutes)
  bike_speed_factor : ℝ   -- speed factor of father's car compared to Wu's bike
  
open ProblemData

def problem_conditions : ProblemData :=
  { v := 350,
    t_catch_up := 7.5,
    d_forest := 3.5,
    t_remaining := 10,
    bike_speed_factor := 5 }

theorem distance_is_correct (data : ProblemData) :
  data.v = 350 →
  data.t_catch_up = 7.5 →
  data.d_forest = 3.5 →
  data.t_remaining = 10 →
  data.bike_speed_factor = 5 →
  distance_from_home_to_forest_park = 11 := 
by
  intros
  sorry

end distance_is_correct_l57_57241


namespace martin_total_distance_l57_57149

noncomputable def calculate_distance_traveled : ℕ :=
  let segment1 := 70 * 3 -- 210 km
  let segment2 := 80 * 4 -- 320 km
  let segment3 := 65 * 3 -- 195 km
  let segment4 := 50 * 2 -- 100 km
  let segment5 := 90 * 4 -- 360 km
  segment1 + segment2 + segment3 + segment4 + segment5

theorem martin_total_distance : calculate_distance_traveled = 1185 :=
by
  sorry

end martin_total_distance_l57_57149


namespace expression_equals_64_l57_57359

theorem expression_equals_64 :
  let a := 2^3 + 2^3
  let b := 2^3 * 2^3
  let c := (2^3)^3
  let d := 2^12 / 2^2
  b = 2^6 :=
by
  sorry

end expression_equals_64_l57_57359


namespace Tim_eats_91_pickle_slices_l57_57837

theorem Tim_eats_91_pickle_slices :
  let Sammy := 25
  let Tammy := 3 * Sammy
  let Ron := Tammy - 0.15 * Tammy
  let Amy := Sammy + 0.50 * Sammy
  let CombinedTotal := Ron + Amy
  let Tim := CombinedTotal - 0.10 * CombinedTotal
  Tim = 91 :=
by
  admit

end Tim_eats_91_pickle_slices_l57_57837


namespace rice_and_wheat_grains_division_l57_57525

-- Definitions for the conditions in the problem
def total_grains : ℕ := 1534
def sample_size : ℕ := 254
def wheat_in_sample : ℕ := 28

-- Proving the approximate amount of wheat grains in the batch  
theorem rice_and_wheat_grains_division : total_grains * (wheat_in_sample / sample_size) = 169 := by 
  sorry

end rice_and_wheat_grains_division_l57_57525


namespace exists_integer_solution_l57_57762

theorem exists_integer_solution (x : ℤ) (h : x - 1 < 0) : ∃ y : ℤ, y < 1 :=
by
  sorry

end exists_integer_solution_l57_57762


namespace ram_krish_task_completion_l57_57977

/-!
  Given:
  1. Ram's efficiency (R) is half of Krish's efficiency (K).
  2. Ram can complete the task alone in 24 days.

  To Prove:
  Ram and Krish will complete the task together in 8 days.
-/

theorem ram_krish_task_completion {R K : ℝ} (hR : R = 1 / 2 * K)
  (hRAMalone : R ≠ 0) (hRAMtime : 24 * R = 1) :
  1 / (R + K) = 8 := by
  sorry

end ram_krish_task_completion_l57_57977


namespace puppies_per_cage_l57_57436

-- Conditions
variables (total_puppies sold_puppies cages initial_puppies per_cage : ℕ)
variables (h_total : total_puppies = 13)
variables (h_sold : sold_puppies = 7)
variables (h_cages : cages = 3)
variables (h_equal_cages : total_puppies - sold_puppies = cages * per_cage)

-- Question
theorem puppies_per_cage :
  per_cage = 2 :=
by {
  sorry
}

end puppies_per_cage_l57_57436


namespace digits_sum_l57_57231

theorem digits_sum (P Q R : ℕ) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10)
  (h_eq : 100 * P + 10 * Q + R + 10 * Q + R = 1012) :
  P + Q + R = 20 :=
by {
  -- Implementation of the proof will go here
  sorry
}

end digits_sum_l57_57231


namespace find_fraction_B_minus_1_over_A_l57_57421

variable (A B : ℝ) (a_n S_n : ℕ → ℝ)
variable (h1 : ∀ n, a_n n + S_n n = A * (n ^ 2) + B * n + 1)
variable (h2 : A ≠ 0)

theorem find_fraction_B_minus_1_over_A : (B - 1) / A = 3 := by
  sorry

end find_fraction_B_minus_1_over_A_l57_57421


namespace students_play_neither_sport_l57_57224

def total_students : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def both_players : ℕ := 10

theorem students_play_neither_sport :
  total_students - (hockey_players + basketball_players - both_players) = 4 :=
by
  sorry

end students_play_neither_sport_l57_57224


namespace solution_set_f_x_leq_x_range_of_a_l57_57293

-- Definition of the function f
def f (x : ℝ) : ℝ := |2 * x - 7| + 1

-- Proof Problem for Question (1):
-- Given: f(x) = |2x - 7| + 1
-- Prove: The solution set of the inequality f(x) <= x is {x | 8/3 <= x <= 6}
theorem solution_set_f_x_leq_x :
  { x : ℝ | f x ≤ x } = { x : ℝ | 8 / 3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Definition of the function g
def g (x : ℝ) : ℝ := f x - 2 * |x - 1|

-- Proof Problem for Question (2):
-- Given: f(x) = |2x - 7| + 1 and g(x) = f(x) - 2 * |x - 1|
-- Prove: If ∃ x, g(x) <= a, then a >= -4
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 :=
sorry

end solution_set_f_x_leq_x_range_of_a_l57_57293


namespace Jerry_paid_more_last_month_l57_57135

def Debt_total : ℕ := 50
def Debt_remaining : ℕ := 23
def Paid_2_months_ago : ℕ := 12
def Paid_last_month : ℕ := 27 - Paid_2_months_ago

theorem Jerry_paid_more_last_month :
  Paid_last_month - Paid_2_months_ago = 3 :=
by
  -- Calculation for Paid_last_month
  have h : Paid_last_month = 27 - 12 := by rfl
  -- Compute the difference
  have diff : 15 - 12 = 3 := by rfl
  exact diff

end Jerry_paid_more_last_month_l57_57135


namespace roots_polynomial_value_l57_57470

theorem roots_polynomial_value (r s t : ℝ) (h₁ : r + s + t = 15) (h₂ : r * s + s * t + t * r = 25) (h₃ : r * s * t = 10) :
  (1 + r) * (1 + s) * (1 + t) = 51 :=
by
  sorry

end roots_polynomial_value_l57_57470


namespace nathan_ate_100_gumballs_l57_57634

/-- Define the number of gumballs per package. -/
def gumballs_per_package : ℝ := 5.0

/-- Define the number of packages Nathan ate. -/
def number_of_packages : ℝ := 20.0

/-- Define the total number of gumballs Nathan ate. -/
def total_gumballs : ℝ := number_of_packages * gumballs_per_package

/-- Prove that Nathan ate 100.0 gumballs. -/
theorem nathan_ate_100_gumballs : total_gumballs = 100.0 :=
sorry

end nathan_ate_100_gumballs_l57_57634


namespace mother_gave_80_cents_l57_57894

theorem mother_gave_80_cents (father_uncles_gift : Nat) (spent_on_candy current_amount : Nat) (gift_from_father gift_from_uncle add_gift_from_uncle : Nat) (x : Nat) :
  father_uncles_gift = gift_from_father + gift_from_uncle ∧
  father_uncles_gift = 110 ∧
  spent_on_candy = 50 ∧
  current_amount = 140 ∧
  gift_from_father = 40 ∧
  gift_from_uncle = 70 ∧
  add_gift_from_uncle = 70 ∧
  x = current_amount + spent_on_candy - father_uncles_gift ∧
  x = 190 - 110 ∨
  x = 80 :=
  sorry

end mother_gave_80_cents_l57_57894


namespace train_length_l57_57095

theorem train_length (time_crossing : ℝ) (speed_train : ℝ) (speed_man : ℝ) (rel_speed : ℝ) (length_train : ℝ) 
    (h1 : time_crossing = 39.99680025597952)
    (h2 : speed_train = 56)
    (h3 : speed_man = 2)
    (h4 : rel_speed = (speed_train - speed_man) * (1000 / 3600))
    (h5 : length_train = rel_speed * time_crossing):
 length_train = 599.9520038396928 :=
by 
  sorry

end train_length_l57_57095


namespace calc_g_f_3_l57_57430

def f (x : ℕ) : ℕ := x^3 + 3

def g (x : ℕ) : ℕ := 2 * x^2 + 3 * x + 2

theorem calc_g_f_3 : g (f 3) = 1892 := by
  sorry

end calc_g_f_3_l57_57430


namespace calculate_difference_of_squares_l57_57935

theorem calculate_difference_of_squares : (640^2 - 360^2) = 280000 := by
  sorry

end calculate_difference_of_squares_l57_57935


namespace LCM_of_fractions_l57_57553

noncomputable def LCM (a b : Rat) : Rat :=
  a * b / (gcd a.num b.num / gcd a.den b.den : Int)

theorem LCM_of_fractions (x : ℤ) (h : x ≠ 0) :
  LCM (1 / (4 * x : ℚ)) (LCM (1 / (6 * x : ℚ)) (1 / (9 * x : ℚ))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l57_57553


namespace total_fish_caught_l57_57859

-- Definitions based on conditions
def sums : List ℕ := [7, 9, 14, 14, 19, 21]

-- Statement of the proof problem
theorem total_fish_caught : 
  (∃ (a b c d : ℕ), [a+b, a+c, a+d, b+c, b+d, c+d] = sums) → 
  ∃ (a b c d : ℕ), a + b + c + d = 28 :=
by 
  sorry

end total_fish_caught_l57_57859


namespace baskets_weight_l57_57961

theorem baskets_weight 
  (weight_per_basket : ℕ)
  (num_baskets : ℕ)
  (total_weight : ℕ) 
  (h1 : weight_per_basket = 30)
  (h2 : num_baskets = 8)
  (h3 : total_weight = weight_per_basket * num_baskets) :
  total_weight = 240 := 
by
  sorry

end baskets_weight_l57_57961


namespace correct_equation_l57_57546

theorem correct_equation (x : ℝ) : (-x^2)^2 = x^4 := by sorry

end correct_equation_l57_57546


namespace shopkeeper_gain_l57_57365

noncomputable def gain_percent (cost_per_kg : ℝ) (claimed_weight : ℝ) (actual_weight : ℝ) : ℝ :=
  let gain := cost_per_kg - (actual_weight / claimed_weight) * cost_per_kg
  (gain / ((actual_weight / claimed_weight) * cost_per_kg)) * 100

theorem shopkeeper_gain (c : ℝ) (cw aw : ℝ) (h : c = 1) (hw : cw = 1) (ha : aw = 0.75) : 
  gain_percent c cw aw = 33.33 :=
by sorry

end shopkeeper_gain_l57_57365


namespace book_sale_revenue_l57_57540

noncomputable def total_amount_received (price_per_book : ℝ) (B : ℕ) (sold_fraction : ℝ) :=
  sold_fraction * B * price_per_book

theorem book_sale_revenue (B : ℕ) (price_per_book : ℝ) (unsold_books : ℕ) (sold_fraction : ℝ) :
  (1 / 3 : ℝ) * B = unsold_books →
  price_per_book = 3.50 →
  unsold_books = 36 →
  sold_fraction = 2 / 3 →
  total_amount_received price_per_book B sold_fraction = 252 :=
by
  intros h1 h2 h3 h4
  sorry

end book_sale_revenue_l57_57540


namespace not_all_pieces_found_l57_57034

theorem not_all_pieces_found (k p v : ℕ) (h1 : p + v > 0) (h2 : k % 2 = 1) : k + 4 * p + 8 * v ≠ 1988 :=
by
  sorry

end not_all_pieces_found_l57_57034


namespace shaded_region_area_l57_57084

def isosceles_triangle (AB AC BC : ℝ) (BAC : ℝ) : Prop :=
  AB = AC ∧ BAC = 120 ∧ BC = 32

def circle_with_diameter (diameter : ℝ) (radius : ℝ) : Prop :=
  radius = diameter / 2

theorem shaded_region_area :
  ∀ (AB AC BC : ℝ) (BAC : ℝ) (O : Type) (a b c : ℕ),
    isosceles_triangle AB AC BC BAC →
    circle_with_diameter BC 8 →
    (a = 43) ∧ (b = 128) ∧ (c = 3) →
    a + b + c = 174 :=
by
  sorry

end shaded_region_area_l57_57084


namespace tank_capacity_l57_57667

theorem tank_capacity (C : ℝ) (h₁ : 3/4 * C + 7 = 7/8 * C) : C = 56 :=
by
  sorry

end tank_capacity_l57_57667


namespace part1_l57_57482

variable (a b c : ℝ) (A B : ℝ)
variable (triangle_abc : Triangle ABC)
variable (cos : ℝ → ℝ)

axiom law_of_cosines : ∀ {a b c A : ℝ}, a^2 = b^2 + c^2 - 2 * b * c * cos A

theorem part1 (h1 : b^2 + 3 * a * c * (a^2 + c^2 - b^2) / (2 * a * c) = 2 * c^2) (h2 : a = c) : A = π / 4 := 
sorry

end part1_l57_57482


namespace chandler_saves_for_laptop_l57_57491

theorem chandler_saves_for_laptop :
  ∃ x : ℕ, 140 + 20 * x = 800 ↔ x = 33 :=
by
  use 33
  sorry

end chandler_saves_for_laptop_l57_57491


namespace find_smaller_number_l57_57242

-- Define the two numbers such that one is 3 times the other
def numbers (x : ℝ) := (x, 3 * x)

-- Define the condition that the sum of the two numbers is 14
def sum_condition (x y : ℝ) : Prop := x + y = 14

-- The theorem we want to prove
theorem find_smaller_number (x : ℝ) (hx : sum_condition x (3 * x)) : x = 3.5 :=
by
  -- Proof goes here
  sorry

end find_smaller_number_l57_57242


namespace regular_polygon_sides_l57_57116

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l57_57116


namespace maximize_take_home_pay_l57_57760

def tax_collected (x : ℝ) : ℝ :=
  10 * x^2

def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - tax_collected x

theorem maximize_take_home_pay : ∃ x : ℝ, (x * 1000 = 50000) ∧ (∀ y : ℝ, take_home_pay x ≥ take_home_pay y) := 
sorry

end maximize_take_home_pay_l57_57760


namespace percentage_of_y_l57_57583

theorem percentage_of_y (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y :=
by
  sorry

end percentage_of_y_l57_57583


namespace radius_of_inscribed_circle_l57_57156

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (triangle : Triangle A B C)

-- Given conditions
def AC : ℝ := 24
def BC : ℝ := 10
def AB : ℝ := 26

-- Statement to be proved
theorem radius_of_inscribed_circle (hAC : triangle.side_length A C = AC)
                                   (hBC : triangle.side_length B C = BC)
                                   (hAB : triangle.side_length A B = AB) :
  triangle.incircle_radius = 4 :=
by sorry

end radius_of_inscribed_circle_l57_57156


namespace range_of_m_l57_57733

theorem range_of_m (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (hx : ∃ x < 0, a^x = 3 * m - 2) :
  1 < m :=
sorry

end range_of_m_l57_57733


namespace percent_of_150_is_60_l57_57431

def percent_is_correct (Part Whole : ℝ) : Prop :=
  (Part / Whole) * 100 = 250

theorem percent_of_150_is_60 :
  percent_is_correct 150 60 :=
by
  sorry

end percent_of_150_is_60_l57_57431


namespace fermat_prime_divisibility_l57_57404

def F (k : ℕ) : ℕ := 2 ^ 2 ^ k + 1

theorem fermat_prime_divisibility {m n : ℕ} (hmn : m > n) : F n ∣ (F m - 2) :=
sorry

end fermat_prime_divisibility_l57_57404


namespace John_walked_miles_to_park_l57_57888

theorem John_walked_miles_to_park :
  ∀ (total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles : ℕ),
    total_skateboarded_miles = 24 →
    skateboarded_first_leg = 10 →
    skateboarded_return_leg = 10 →
    total_skateboarded_miles = skateboarded_first_leg + skateboarded_return_leg + walked_miles →
    walked_miles = 4 :=
by
  intros total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles
  intro h1 h2 h3 h4
  sorry

end John_walked_miles_to_park_l57_57888


namespace drawings_with_colored_pencils_l57_57802

-- Definitions based on conditions
def total_drawings : Nat := 25
def blending_markers_drawings : Nat := 7
def charcoal_drawings : Nat := 4
def colored_pencils_drawings : Nat := total_drawings - (blending_markers_drawings + charcoal_drawings)

-- Theorem to be proven
theorem drawings_with_colored_pencils : colored_pencils_drawings = 14 :=
by
  sorry

end drawings_with_colored_pencils_l57_57802


namespace abs_sum_values_l57_57696

theorem abs_sum_values (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := 
by
  sorry

end abs_sum_values_l57_57696


namespace sum_of_distinct_prime_factors_l57_57647

-- Definition of the expression
def expression : ℤ := 7^4 - 7^2

-- Statement of the theorem
theorem sum_of_distinct_prime_factors : 
  Nat.sum (List.eraseDup (Nat.factors expression.natAbs)) = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l57_57647


namespace intersection_and_area_l57_57664

theorem intersection_and_area (A B : ℝ × ℝ) (x y : ℝ):
  (x - 2 * y - 5 = 0) → (x ^ 2 + y ^ 2 = 50) →
  (A = (-5, -5) ∨ A = (7, 1)) → (B = (-5, -5) ∨ B = (7, 1)) →
  (A ≠ B) →
  ∃ (area : ℝ), area = 15 :=
by
  sorry

end intersection_and_area_l57_57664


namespace point_on_graph_of_inverse_proportion_l57_57367

theorem point_on_graph_of_inverse_proportion :
  ∃ x y : ℝ, (x = 2 ∧ y = 4) ∧ y = 8 / x :=
by
  sorry

end point_on_graph_of_inverse_proportion_l57_57367


namespace distinct_units_digits_perfect_cube_l57_57355

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l57_57355


namespace sin_minus_cos_eq_sqrt3_div2_l57_57600

theorem sin_minus_cos_eq_sqrt3_div2
  (α : ℝ) 
  (h_range : (Real.pi / 4) < α ∧ α < (Real.pi / 2))
  (h_sincos : Real.sin α * Real.cos α = 1 / 8) :
  Real.sin α - Real.cos α = Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_eq_sqrt3_div2_l57_57600


namespace statues_added_in_third_year_l57_57765

/-
Definition of the turtle statues problem:

1. Initially, there are 4 statues in the first year.
2. In the second year, the number of statues quadruples.
3. In the third year, x statues are added, and then 3 statues are broken.
4. In the fourth year, 2 * 3 new statues are added.
5. In total, at the end of the fourth year, there are 31 statues.
-/

def year1_statues : ℕ := 4
def year2_statues : ℕ := 4 * year1_statues
def before_hailstorm_year3_statues (x : ℕ) : ℕ := year2_statues + x
def after_hailstorm_year3_statues (x : ℕ) : ℕ := before_hailstorm_year3_statues x - 3
def total_year4_statues (x : ℕ) : ℕ := after_hailstorm_year3_statues x + 2 * 3

theorem statues_added_in_third_year (x : ℕ) (h : total_year4_statues x = 31) : x = 12 :=
by
  sorry

end statues_added_in_third_year_l57_57765


namespace percentage_increase_is_2_l57_57447

def alan_price := 2000
def john_price := 2040
def percentage_increase (alan_price : ℕ) (john_price : ℕ) : ℕ := (john_price - alan_price) * 100 / alan_price

theorem percentage_increase_is_2 (alan_price john_price : ℕ) (h₁ : alan_price = 2000) (h₂ : john_price = 2040) :
  percentage_increase alan_price john_price = 2 := by
  rw [h₁, h₂]
  sorry

end percentage_increase_is_2_l57_57447


namespace jack_second_half_time_l57_57080

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end jack_second_half_time_l57_57080


namespace rate_of_interest_is_8_l57_57936

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def principal_C : ℕ := 3000
def time_C : ℕ := 4
def total_interest : ℕ := 1760

theorem rate_of_interest_is_8 :
  ∃ (R : ℝ), ((principal_B * R * time_B) / 100 + (principal_C * R * time_C) / 100 = total_interest) → R = 8 := 
by
  sorry

end rate_of_interest_is_8_l57_57936


namespace school_choir_robe_cost_l57_57981

theorem school_choir_robe_cost :
  ∀ (total_robes_needed current_robes cost_per_robe : ℕ), 
  total_robes_needed = 30 → 
  current_robes = 12 → 
  cost_per_robe = 2 → 
  (total_robes_needed - current_robes) * cost_per_robe = 36 :=
by
  intros total_robes_needed current_robes cost_per_robe h1 h2 h3
  sorry

end school_choir_robe_cost_l57_57981


namespace sequence_count_even_odd_l57_57240

/-- The number of 8-digit sequences such that no two adjacent digits have the same parity
    and the sequence starts with an even number. -/
theorem sequence_count_even_odd : 
  let choices_for_even := 5
  let choices_for_odd := 5
  let total_positions := 8
  (choices_for_even * (choices_for_odd * choices_for_even) ^ (total_positions / 2 - 1)) = 390625 :=
by
  sorry

end sequence_count_even_odd_l57_57240


namespace womenInBusinessClass_l57_57514

-- Given conditions
def totalPassengers : ℕ := 300
def percentageWomen : ℚ := 70 / 100
def percentageWomenBusinessClass : ℚ := 15 / 100

def numberOfWomen (totalPassengers : ℕ) (percentageWomen : ℚ) : ℚ := 
  totalPassengers * percentageWomen

def numberOfWomenBusinessClass (numberOfWomen : ℚ) (percentageWomenBusinessClass : ℚ) : ℚ := 
  numberOfWomen * percentageWomenBusinessClass

-- Theorem to prove
theorem womenInBusinessClass (totalPassengers : ℕ) (percentageWomen : ℚ) (percentageWomenBusinessClass : ℚ) :
  numberOfWomenBusinessClass (numberOfWomen totalPassengers percentageWomen) percentageWomenBusinessClass = 32 := 
by 
  -- The proof steps would go here
  sorry

end womenInBusinessClass_l57_57514


namespace minimal_board_size_for_dominoes_l57_57606

def board_size_is_minimal (n: ℕ) (total_area: ℕ) (domino_size: ℕ) (num_dominoes: ℕ) : Prop :=
  ∀ m: ℕ, m < n → ¬ (total_area ≥ m * m ∧ m * m = num_dominoes * domino_size)

theorem minimal_board_size_for_dominoes (n: ℕ) :
  board_size_is_minimal 77 2008 2 1004 :=
by
  sorry

end minimal_board_size_for_dominoes_l57_57606


namespace greatest_multiple_of_5_l57_57207

theorem greatest_multiple_of_5 (y : ℕ) (h1 : y > 0) (h2 : y % 5 = 0) (h3 : y^3 < 8000) : y ≤ 15 :=
by {
  sorry
}

end greatest_multiple_of_5_l57_57207


namespace box_cubes_no_green_face_l57_57403

theorem box_cubes_no_green_face (a b c : ℕ) (h_a2 : a > 2) (h_b2 : b > 2) (h_c2 : c > 2)
  (h_no_green_face : (a-2)*(b-2)*(c-2) = (a*b*c) / 3) :
  (a, b, c) = (7, 30, 4) ∨ (a, b, c) = (8, 18, 4) ∨ (a, b, c) = (9, 14, 4) ∨
  (a, b, c) = (10, 12, 4) ∨ (a, b, c) = (5, 27, 5) ∨ (a, b, c) = (6, 12, 5) ∨
  (a, b, c) = (7, 9, 5) ∨ (a, b, c) = (6, 8, 6) :=
sorry

end box_cubes_no_green_face_l57_57403


namespace project_completion_rate_l57_57596

variables {a b c d e : ℕ} {f g : ℚ}  -- Assuming efficiency ratings can be represented by rational numbers.

theorem project_completion_rate (h : (a * f / c) = b / c) 
: (d * g / e) = bdge / ca := 
sorry

end project_completion_rate_l57_57596


namespace original_amount_l57_57875

theorem original_amount (X : ℝ) (h : 0.05 * X = 25) : X = 500 :=
sorry

end original_amount_l57_57875


namespace oldest_child_age_l57_57896

def avg (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem oldest_child_age (a b : ℕ) (h1 : avg a b x = 10) (h2 : a = 8) (h3 : b = 11) : x = 11 :=
by
  sorry

end oldest_child_age_l57_57896


namespace find_y_value_l57_57874

theorem find_y_value (x y : ℝ) (k : ℝ) 
  (h1 : 5 * y = k / x^2)
  (h2 : y = 4)
  (h3 : x = 2)
  (h4 : k = 80) :
  ( ∃ y : ℝ, 5 * y = k / 4^2 ∧ y = 1) :=
by
  sorry

end find_y_value_l57_57874


namespace arithmetic_seq_sum_l57_57558

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 5 + a 6 + a 7 = 1) : a 3 + a 9 = 2 / 3 :=
sorry

end arithmetic_seq_sum_l57_57558


namespace numberOfWaysToChoose4Cards_l57_57909

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l57_57909


namespace average_weight_of_whole_class_l57_57248

theorem average_weight_of_whole_class :
  let students_A := 26
  let students_B := 34
  let avg_weight_A := 50
  let avg_weight_B := 30
  let total_weight_A := avg_weight_A * students_A
  let total_weight_B := avg_weight_B * students_B
  let total_weight_class := total_weight_A + total_weight_B
  let total_students_class := students_A + students_B
  let avg_weight_class := total_weight_class / total_students_class
  avg_weight_class = 38.67 :=
by {
  sorry -- Proof is not required as per instructions
}

end average_weight_of_whole_class_l57_57248


namespace fifth_score_l57_57832

theorem fifth_score (r : ℕ) 
  (h1 : r % 5 = 0)
  (h2 : (60 + 75 + 85 + 95 + r) / 5 = 80) : 
  r = 85 := by 
  sorry

end fifth_score_l57_57832


namespace other_group_land_l57_57371

def total_land : ℕ := 900
def remaining_land : ℕ := 385
def lizzies_group_land : ℕ := 250

theorem other_group_land :
  total_land - remaining_land - lizzies_group_land = 265 :=
by
  sorry

end other_group_land_l57_57371


namespace average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l57_57915

theorem average_children_per_grade (G3_girls G3_boys G3_club : ℕ) 
                                  (G4_girls G4_boys G4_club : ℕ) 
                                  (G5_girls G5_boys G5_club : ℕ) 
                                  (H1 : G3_girls = 28) 
                                  (H2 : G3_boys = 35) 
                                  (H3 : G3_club = 12) 
                                  (H4 : G4_girls = 45) 
                                  (H5 : G4_boys = 42) 
                                  (H6 : G4_club = 15) 
                                  (H7 : G5_girls = 38) 
                                  (H8 : G5_boys = 51) 
                                  (H9 : G5_club = 10) :
   (63 + 87 + 89) / 3 = 79.67 :=
by sorry

theorem average_girls_per_grade (G3_girls G4_girls G5_girls : ℕ) 
                                (H1 : G3_girls = 28) 
                                (H2 : G4_girls = 45) 
                                (H3 : G5_girls = 38) :
   (28 + 45 + 38) / 3 = 37 :=
by sorry

theorem average_boys_per_grade (G3_boys G4_boys G5_boys : ℕ)
                               (H1 : G3_boys = 35) 
                               (H2 : G4_boys = 42) 
                               (H3 : G5_boys = 51) :
   (35 + 42 + 51) / 3 = 42.67 :=
by sorry

theorem average_club_members_per_grade (G3_club G4_club G5_club : ℕ) 
                                       (H1 : G3_club = 12)
                                       (H2 : G4_club = 15)
                                       (H3 : G5_club = 10) :
   (12 + 15 + 10) / 3 = 12.33 :=
by sorry

end average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l57_57915


namespace cody_games_still_has_l57_57950

def initial_games : ℕ := 9
def games_given_away_to_jake : ℕ := 4
def games_given_away_to_sarah : ℕ := 2
def games_bought_over_weekend : ℕ := 3

theorem cody_games_still_has : 
  initial_games - (games_given_away_to_jake + games_given_away_to_sarah) + games_bought_over_weekend = 6 := 
by
  sorry

end cody_games_still_has_l57_57950


namespace expectedValueProof_l57_57865

-- Definition of the problem conditions
def veryNormalCoin {n : ℕ} : Prop :=
  ∀ t : ℕ, (5 < t → (t - 5) = n → (t+1 = t + 1)) ∧ (t ≤ 5 ∨ n = t)

-- Definition of the expected value calculation
def expectedValue (n : ℕ) : ℚ :=
  if n > 0 then (1/2)^n else 0

-- Expected value for the given problem
def expectedValueProblem : ℚ := 
  let a1 := -2/683
  let expectedFirstFlip := 1/2 - 1/(2 * 683)
  100 * 341 + 683

-- Main statement to prove
theorem expectedValueProof : expectedValueProblem = 34783 := 
  sorry -- Proof omitted

end expectedValueProof_l57_57865


namespace megan_average_speed_l57_57296

theorem megan_average_speed :
  ∃ s : ℕ, s = 100 / 3 ∧ ∃ (o₁ o₂ : ℕ), o₁ = 27472 ∧ o₂ = 27572 ∧ o₂ - o₁ = 100 :=
by
  sorry

end megan_average_speed_l57_57296


namespace total_prize_amount_l57_57143

theorem total_prize_amount:
  ∃ P : ℝ, 
  (∃ n m : ℝ, n = 15 ∧ m = 15 ∧ ((2 / 5) * P = (3 / 5) * n * 285) ∧ P = 2565 * 2.5 + 6 * 15 ∧ ∀ i : ℕ, i < m → i ≥ 0 → P ≥ 15)
  ∧ P = 6502.5 :=
sorry

end total_prize_amount_l57_57143


namespace democrats_ratio_l57_57542

variable (F M D_F D_M TotalParticipants : ℕ)

-- Assume the following conditions
variables (H1 : F + M = 660)
variables (H2 : D_F = 1 / 2 * F)
variables (H3 : D_F = 110)
variables (H4 : D_M = 1 / 4 * M)
variables (H5 : TotalParticipants = 660)

theorem democrats_ratio 
  (H1 : F + M = 660)
  (H2 : D_F = 1 / 2 * F)
  (H3 : D_F = 110)
  (H4 : D_M = 1 / 4 * M)
  (H5 : TotalParticipants = 660) :
  (D_F + D_M) / TotalParticipants = 1 / 3
:= 
  sorry

end democrats_ratio_l57_57542


namespace olivia_spent_89_l57_57841

-- Define initial and subsequent amounts
def initial_amount : ℕ := 100
def atm_amount : ℕ := 148
def after_supermarket : ℕ := 159

-- Total amount before supermarket
def total_before_supermarket : ℕ := initial_amount + atm_amount

-- Amount spent
def amount_spent : ℕ := total_before_supermarket - after_supermarket

-- Proof that Olivia spent 89 dollars
theorem olivia_spent_89 : amount_spent = 89 := sorry

end olivia_spent_89_l57_57841


namespace range_of_b_l57_57211

theorem range_of_b (a b c : ℝ) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 24) : 
  1 ≤ b ∧ b ≤ 5 := 
sorry

end range_of_b_l57_57211


namespace repeating_decimal_base_l57_57705

theorem repeating_decimal_base (k : ℕ) (h_pos : 0 < k) (h_repr : (9 : ℚ) / 61 = (3 * k + 4) / (k^2 - 1)) : k = 21 :=
  sorry

end repeating_decimal_base_l57_57705


namespace find_y_payment_l57_57557

-- Definitions for the conditions in the problem
def total_payment (X Y : ℝ) : Prop := X + Y = 560
def x_is_120_percent_of_y (X Y : ℝ) : Prop := X = 1.2 * Y

-- Problem statement converted to a Lean proof problem
theorem find_y_payment (X Y : ℝ) (h1 : total_payment X Y) (h2 : x_is_120_percent_of_y X Y) : Y = 255 := 
by sorry

end find_y_payment_l57_57557


namespace total_packs_sold_l57_57136

theorem total_packs_sold (lucy_packs : ℕ) (robyn_packs : ℕ) (h1 : lucy_packs = 19) (h2 : robyn_packs = 16) : lucy_packs + robyn_packs = 35 :=
by
  sorry

end total_packs_sold_l57_57136


namespace parabola_equation_l57_57609

open Classical

noncomputable def circle_center : ℝ × ℝ := (2, 0)

theorem parabola_equation (vertex : ℝ × ℝ) (focus : ℝ × ℝ) :
  vertex = (0, 0) ∧ focus = circle_center → ∀ x y : ℝ, y^2 = 8 * x := by
  intro h
  sorry

end parabola_equation_l57_57609


namespace ratio_cereal_A_to_B_l57_57903

-- Definitions translated from conditions
def sugar_percentage_A : ℕ := 10
def sugar_percentage_B : ℕ := 2
def desired_sugar_percentage : ℕ := 6

-- The theorem based on the question and correct answer
theorem ratio_cereal_A_to_B :
  let difference_A := sugar_percentage_A - desired_sugar_percentage
  let difference_B := desired_sugar_percentage - sugar_percentage_B
  difference_A = 4 ∧ difference_B = 4 → 
  difference_B / difference_A = 1 :=
by
  intros
  sorry

end ratio_cereal_A_to_B_l57_57903


namespace range_of_a_l57_57361

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a - 1 / Real.exp x)

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ deriv (f a) x₁ = 0 ∧ deriv (f a) x₂ = 0) ↔ -1 / Real.exp 2 < a ∧ a < 0 := 
sorry

end range_of_a_l57_57361


namespace part1_part2_part3_l57_57052

theorem part1 {k b : ℝ} (h₀ : k ≠ 0) (h₁ : b = 1) (h₂ : k + b = 2) : 
  ∀ x : ℝ, y = k * x + b → y = x + 1 :=
by sorry

theorem part2 : ∃ (C : ℝ × ℝ), 
  C.1 = 3 ∧ C.2 = 4 ∧ y = x + 1 :=
by sorry

theorem part3 {n k : ℝ} (h₀ : k ≠ 0) 
  (h₁ : ∀ x : ℝ, x < 3 → (2 / 3) * x + n > x + 1 ∧ (2 / 3) * x + n < 4) 
  (h₂ : ∀ x : ℝ, y = (2 / 3) * x + n → y = 4 ∧ x = 3) :
  n = 2 :=
by sorry

end part1_part2_part3_l57_57052


namespace increase_in_value_l57_57093

-- Define the conditions
def starting_weight : ℝ := 400
def weight_multiplier : ℝ := 1.5
def price_per_pound : ℝ := 3

-- Define new weight and values
def new_weight : ℝ := starting_weight * weight_multiplier
def value_at_starting_weight : ℝ := starting_weight * price_per_pound
def value_at_new_weight : ℝ := new_weight * price_per_pound

-- Theorem to prove
theorem increase_in_value : value_at_new_weight - value_at_starting_weight = 600 := by
  sorry

end increase_in_value_l57_57093


namespace simplify_fraction_l57_57005

theorem simplify_fraction : (3^9 / 9^3) = 27 :=
by
  sorry

end simplify_fraction_l57_57005


namespace color_copies_comparison_l57_57777

theorem color_copies_comparison (n : ℕ) (pX pY : ℝ) (charge_diff : ℝ) 
  (h₀ : pX = 1.20) (h₁ : pY = 1.70) (h₂ : charge_diff = 35) 
  (h₃ : pY * n = pX * n + charge_diff) : n = 70 :=
by
  -- proof steps would go here
  sorry

end color_copies_comparison_l57_57777


namespace clock_angle_at_3_40_l57_57952

theorem clock_angle_at_3_40
  (hour_position : ℕ → ℝ)
  (minute_position : ℕ → ℝ)
  (h_hour : hour_position 3 = 3 * 30)
  (h_minute : minute_position 40 = 40 * 6)
  : abs (minute_position 40 - (hour_position 3 + 20 * 30 / 60)) = 130 :=
by
  -- Insert proof here
  sorry

end clock_angle_at_3_40_l57_57952


namespace football_total_points_l57_57784

theorem football_total_points :
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  Zach_points + Ben_points + Sarah_points + Emily_points = 109.0 :=
by
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  have h : Zach_points + Ben_points + Sarah_points + Emily_points = 42.0 + 21.0 + 18.5 + 27.5 := by rfl
  have total_points := 42.0 + 21.0 + 18.5 + 27.5
  have result := 109.0
  sorry

end football_total_points_l57_57784


namespace maximize_distance_l57_57978

theorem maximize_distance (front_tires_lifetime: ℕ) (rear_tires_lifetime: ℕ):
  front_tires_lifetime = 20000 → rear_tires_lifetime = 30000 → 
  ∃ D, D = 30000 :=
by
  sorry

end maximize_distance_l57_57978


namespace min_value_of_reciprocal_sum_l57_57814

noncomputable def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ((2016 * (a 1 + a 2016)) / 2 = 1008)

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) (h : arithmetic_sequence_condition a) :
  ∃ x : ℝ, x = 4 ∧ (∀ y, y = (1 / a 1001 + 1 / a 1016) → x ≤ y) :=
sorry

end min_value_of_reciprocal_sum_l57_57814


namespace approximate_number_of_fish_in_pond_l57_57310

theorem approximate_number_of_fish_in_pond :
  (∃ N : ℕ, 
  (∃ tagged1 tagged2 : ℕ, tagged1 = 50 ∧ tagged2 = 10) ∧
  (∃ caught1 caught2 : ℕ, caught1 = 50 ∧ caught2 = 50) ∧
  ((tagged2 : ℝ) / caught2 = (tagged1 : ℝ) / (N : ℝ)) ∧
  N = 250) :=
sorry

end approximate_number_of_fish_in_pond_l57_57310


namespace find_smallest_number_l57_57892

theorem find_smallest_number (a b c : ℕ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : b = 31)
  (h4 : c = b + 6)
  (h5 : (a + b + c) / 3 = 30) :
  a = 22 := 
sorry

end find_smallest_number_l57_57892


namespace min_correct_all_four_l57_57099

def total_questions : ℕ := 15
def correct_xiaoxi : ℕ := 11
def correct_xiaofei : ℕ := 12
def correct_xiaomei : ℕ := 13
def correct_xiaoyang : ℕ := 14

theorem min_correct_all_four : 
(∀ total_questions correct_xiaoxi correct_xiaofei correct_xiaomei correct_xiaoyang, 
  total_questions = 15 → correct_xiaoxi = 11 → 
  correct_xiaofei = 12 → correct_xiaomei = 13 → 
  correct_xiaoyang = 14 → 
  ∃ k : ℕ, k = 5 ∧ 
    k = total_questions - ((total_questions - correct_xiaoxi) + 
    (total_questions - correct_xiaofei) + 
    (total_questions - correct_xiaomei) + 
    (total_questions - correct_xiaoyang)) / 4) := 
sorry

end min_correct_all_four_l57_57099


namespace find_m_l57_57959

theorem find_m (m : ℝ) (x1 x2 : ℝ) 
  (h_eq : x1 ^ 2 - 4 * x1 - 2 * m + 5 = 0)
  (h_distinct : x1 ≠ x2)
  (h_product_sum_eq : x1 * x2 + x1 + x2 = m ^ 2 + 6) : 
  m = 1 ∧ m > 1/2 :=
sorry

end find_m_l57_57959


namespace team_b_can_serve_on_submarine_l57_57373

   def can_serve_on_submarine (height : ℝ) : Prop := height ≤ 168

   def average_height_condition (avg_height : ℝ) : Prop := avg_height = 166

   def median_height_condition (median_height : ℝ) : Prop := median_height = 167

   def tallest_height_condition (max_height : ℝ) : Prop := max_height = 169

   def mode_height_condition (mode_height : ℝ) : Prop := mode_height = 167

   theorem team_b_can_serve_on_submarine (H : median_height_condition 167) :
     ∀ (h : ℝ), can_serve_on_submarine h :=
   sorry
   
end team_b_can_serve_on_submarine_l57_57373


namespace sequence_strictly_monotonic_increasing_l57_57434

noncomputable def a (n : ℕ) : ℝ := ((n + 1) ^ n * n ^ (2 - n)) / (7 * n ^ 2 + 1)

theorem sequence_strictly_monotonic_increasing :
  ∀ n : ℕ, a n < a (n + 1) := 
by {
  sorry
}

end sequence_strictly_monotonic_increasing_l57_57434


namespace number_of_fences_painted_l57_57053

-- Definitions based on the problem conditions
def meter_fee : ℝ := 0.2
def fence_length : ℝ := 500
def total_earnings : ℝ := 5000

-- Target statement
theorem number_of_fences_painted : (total_earnings / (fence_length * meter_fee)) = 50 := by
sorry

end number_of_fences_painted_l57_57053


namespace min_value_change_when_2x2_added_l57_57031

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_change_when_2x2_added
  (a b : ℝ)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∀ x : ℝ, (a + 1) * x^2 + b * x + c > a * x^2 + b * x + c + 1)
  (h3 : ∀ x : ℝ, (a - 1) * x^2 + b * x + c < a * x^2 + b * x + c - 3) :
  ∀ x : ℝ, (a + 2) * x^2 + b * x + c = a * x^2 + b * x + (c + 1.5) :=
sorry

end min_value_change_when_2x2_added_l57_57031


namespace jade_transactions_l57_57693

theorem jade_transactions :
  ∀ (transactions_mabel transactions_anthony transactions_cal transactions_jade : ℕ),
    transactions_mabel = 90 →
    transactions_anthony = transactions_mabel + transactions_mabel / 10 →
    transactions_cal = (transactions_anthony * 2) / 3 →
    transactions_jade = transactions_cal + 19 →
    transactions_jade = 85 :=
by
  intros transactions_mabel transactions_anthony transactions_cal transactions_jade
  intros h_mabel h_anthony h_cal h_jade
  sorry

end jade_transactions_l57_57693


namespace min_fraction_value_l57_57669

theorem min_fraction_value 
    (a : ℕ → ℝ) 
    (S : ℕ → ℝ) 
    (d : ℝ) 
    (n : ℕ) 
    (h1 : ∀ {n}, a n = 5 + (n - 1) * d)
    (h2 : (a 2) * (a 10) = (a 4 - 1)^2) 
    (h3 : S n = (n * (a 1 + a n)) / 2)
    (h4 : a 1 = 5)
    (h5 : d > 0) :
    2 * S n + n + 32 ≥ (20 / 3) * (a n + 1) := sorry

end min_fraction_value_l57_57669


namespace simplify_expression_l57_57573

variable (x : ℝ)

theorem simplify_expression : (5 * x + 2 * (4 + x)) = (7 * x + 8) := 
by
  sorry

end simplify_expression_l57_57573


namespace function_symmetric_and_monotonic_l57_57165

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^4 - 2 * Real.sin x * Real.cos x - (Real.sin x)^4

theorem function_symmetric_and_monotonic :
  (∀ x, f (x + (3/8) * π) = f (x - (3/8) * π)) ∧
  (∀ x y, x ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → y ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → x < y → f x > f y) :=
by
  sorry

end function_symmetric_and_monotonic_l57_57165


namespace probability_correct_guesses_l57_57538

theorem probability_correct_guesses:
  let p_wrong := (5/6 : ℚ)
  let p_miss_all := p_wrong ^ 5
  let p_at_least_one_correct := 1 - p_miss_all
  p_at_least_one_correct = 4651/7776 := by
  sorry

end probability_correct_guesses_l57_57538


namespace find_numbers_l57_57111

theorem find_numbers (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : 1000 * x + y = 7 * x * y) :
  x = 143 ∧ y = 143 :=
by
  sorry

end find_numbers_l57_57111


namespace friendly_snakes_not_blue_l57_57803

variable (Snakes : Type)
variable (sally_snakes : Finset Snakes)
variable (blue : Snakes → Prop)
variable (friendly : Snakes → Prop)
variable (can_swim : Snakes → Prop)
variable (can_climb : Snakes → Prop)

variable [DecidablePred blue] [DecidablePred friendly] [DecidablePred can_swim] [DecidablePred can_climb]

-- The number of snakes in Sally's collection
axiom h_snakes_count : sally_snakes.card = 20
-- There are 7 blue snakes
axiom h_blue : (sally_snakes.filter blue).card = 7
-- There are 10 friendly snakes
axiom h_friendly : (sally_snakes.filter friendly).card = 10
-- All friendly snakes can swim
axiom h1 : ∀ s ∈ sally_snakes, friendly s → can_swim s
-- No blue snakes can climb
axiom h2 : ∀ s ∈ sally_snakes, blue s → ¬ can_climb s
-- Snakes that can't climb also can't swim
axiom h3 : ∀ s ∈ sally_snakes, ¬ can_climb s → ¬ can_swim s

theorem friendly_snakes_not_blue :
  ∀ s ∈ sally_snakes, friendly s → ¬ blue s :=
by
  sorry

end friendly_snakes_not_blue_l57_57803


namespace imaginary_part_of_z_l57_57917

open Complex

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : 
  im ((x + I) / (y - I)) = 1 :=
by
  sorry

end imaginary_part_of_z_l57_57917


namespace Romeo_bars_of_chocolate_l57_57139

theorem Romeo_bars_of_chocolate 
  (cost_per_bar : ℕ) (packaging_cost : ℕ) (total_sale : ℕ) (profit : ℕ) (x : ℕ) :
  cost_per_bar = 5 →
  packaging_cost = 2 →
  total_sale = 90 →
  profit = 55 →
  (total_sale - (cost_per_bar + packaging_cost) * x = profit) →
  x = 5 :=
by
  sorry

end Romeo_bars_of_chocolate_l57_57139


namespace domain_of_f_zeros_of_f_l57_57419

def log_a (a : ℝ) (x : ℝ) : ℝ := sorry -- Assume definition of logarithm base 'a'.

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_a a (2 - x)

theorem domain_of_f (a : ℝ) : ∀ x : ℝ, 2 - x > 0 ↔ x < 2 :=
by
  sorry

theorem zeros_of_f (a : ℝ) : f a 1 = 0 :=
by
  sorry

end domain_of_f_zeros_of_f_l57_57419


namespace find_s_for_g_eq_0_l57_57349

def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

theorem find_s_for_g_eq_0 : ∃ (s : ℝ), g 3 s = 0 → s = -867 :=
by
  sorry

end find_s_for_g_eq_0_l57_57349


namespace inequality_solution_set_l57_57998

theorem inequality_solution_set (a : ℝ) : 
    (a = 0 → (∃ x : ℝ, x > 1 ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a < 0 → (∃ x : ℝ, (x < 2/a ∨ x > 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (0 < a ∧ a < 2 → (∃ x : ℝ, (1 < x ∧ x < 2/a) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a = 2 → ¬(∃ x : ℝ, ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a > 2 → (∃ x : ℝ, (2/a < x ∧ x < 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) :=
by sorry

end inequality_solution_set_l57_57998


namespace find_xyz_sum_l57_57376

theorem find_xyz_sum
  (x y z : ℝ)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 + x * y + y^2 = 108)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + z * x + x^2 = 124) :
  x * y + y * z + z * x = 48 := 
  sorry

end find_xyz_sum_l57_57376


namespace multiple_of_3_l57_57472

theorem multiple_of_3 (a b : ℤ) (h1 : ∃ m : ℤ, a = 3 * m) (h2 : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end multiple_of_3_l57_57472


namespace find_k_l57_57862

-- Define the vectors a, b, and c
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (0, 1)

-- Define the vector c involving variable k
variables (k : ℝ)
def vec_c : ℝ × ℝ := (k, -2)

-- Define the combined vector (a + 2b)
def combined_vec : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem to prove
theorem find_k (h : dot_product combined_vec (vec_c k) = 0) : k = 8 :=
by sorry

end find_k_l57_57862


namespace speed_of_current_is_2_l57_57854

noncomputable def speed_current : ℝ :=
  let still_water_speed := 14  -- kmph
  let distance_m := 40         -- meters
  let time_s := 8.9992800576   -- seconds
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let downstream_speed := distance_km / time_h
  downstream_speed - still_water_speed

theorem speed_of_current_is_2 :
  speed_current = 2 :=
by
  sorry

end speed_of_current_is_2_l57_57854


namespace jovana_added_shells_l57_57769

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h1 : initial_amount = 5) 
  (h2 : final_amount = 17) 
  (h3 : added_amount = final_amount - initial_amount) : 
  added_amount = 12 := 
by 
  -- Since the proof is not required, we add sorry here to skip the proof.
  sorry 

end jovana_added_shells_l57_57769


namespace find_c_l57_57397

theorem find_c (c : ℝ) (h : ∀ x : ℝ, ∃ a : ℝ, (x + a)^2 = x^2 + 200 * x + c) : c = 10000 :=
sorry

end find_c_l57_57397


namespace triangle_angle_property_l57_57702

variables {a b c : ℝ}
variables {A B C : ℝ} -- angles in triangle ABC

-- definition of a triangle side condition
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- condition given in the problem
def satisfies_condition (a b c : ℝ) : Prop := b^2 = a^2 + c^2

-- angle property based on given problem
def angle_B_is_right (A B C : ℝ) : Prop := B = 90

theorem triangle_angle_property (a b c : ℝ) (A B C : ℝ)
  (ht : triangle a b c) 
  (hc : satisfies_condition a b c) : 
  angle_B_is_right A B C :=
sorry

end triangle_angle_property_l57_57702


namespace rectangle_area_is_588_l57_57338

-- Definitions based on the conditions of the problem
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- The statement to prove that the area of the rectangle is 588
theorem rectangle_area_is_588 : length * width = 588 :=
by
  -- Omitted proof
  sorry

end rectangle_area_is_588_l57_57338


namespace grocery_cost_l57_57460

/-- Potatoes and celery costs problem. -/
theorem grocery_cost (a b : ℝ) (potato_cost_per_kg celery_cost_per_kg : ℝ) 
(h1 : potato_cost_per_kg = 1) (h2 : celery_cost_per_kg = 0.7) :
  potato_cost_per_kg * a + celery_cost_per_kg * b = a + 0.7 * b :=
by
  rw [h1, h2]
  sorry

end grocery_cost_l57_57460


namespace necessary_condition_for_inequality_l57_57258

-- Definitions based on the conditions in a)
variables (A B C D : ℝ)

-- Main statement translating c) into Lean
theorem necessary_condition_for_inequality (h : C < D) : A > B :=
by sorry

end necessary_condition_for_inequality_l57_57258


namespace round_robin_tournament_l57_57767

theorem round_robin_tournament (n k : ℕ) (h : (n-2) * (n-3) = 2 * 3^k): n = 5 :=
sorry

end round_robin_tournament_l57_57767


namespace calculate_floor_100_p_l57_57200

noncomputable def max_prob_sum_7 : ℝ := 
  let p1 := 0.2
  let p6 := 0.1
  let p2_p5_p3_p4 := 0.7 - p1 - p6
  2 * (p1 * p6 + p2_p5_p3_p4 / 2 ^ 2)

theorem calculate_floor_100_p : ∃ p : ℝ, (⌊100 * max_prob_sum_7⌋ = 28) :=
  by
  sorry

end calculate_floor_100_p_l57_57200


namespace average_speed_calculation_l57_57856

-- Define constants and conditions
def speed_swimming : ℝ := 1
def speed_running : ℝ := 6
def distance : ℝ := 1  -- We use a generic distance d = 1 (assuming normalized unit distance)

-- Proof statement
theorem average_speed_calculation :
  (2 * distance) / ((distance / speed_swimming) + (distance / speed_running)) = 12 / 7 :=
by
  sorry

end average_speed_calculation_l57_57856


namespace total_painting_cost_l57_57049

variable (house_area : ℕ) (price_per_sqft : ℕ)

theorem total_painting_cost (h1 : house_area = 484) (h2 : price_per_sqft = 20) :
  house_area * price_per_sqft = 9680 :=
by
  sorry

end total_painting_cost_l57_57049


namespace number_of_possible_lists_l57_57068

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l57_57068


namespace find_w_l57_57168

theorem find_w {w : ℝ} : (3, w^3) ∈ {p : ℝ × ℝ | ∃ x, p = (x, x^2 - 1)} → w = 2 :=
by
  sorry

end find_w_l57_57168


namespace min_max_f_l57_57222

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - Real.cos x

theorem min_max_f :
  (∀ x, 2 * (Real.sin (x / 2))^2 = 1 - Real.cos x) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 5 / 4) :=
by 
  intros h x
  sorry

end min_max_f_l57_57222


namespace perpendicular_lines_a_l57_57066

theorem perpendicular_lines_a {a : ℝ} :
  ((∀ x y : ℝ, (2 * a - 1) * x + a * y + a = 0) → (∀ x y : ℝ, a * x - y + 2 * a = 0) → a = 0 ∨ a = 1) :=
by
  intro h₁ h₂
  sorry

end perpendicular_lines_a_l57_57066


namespace simplify_nested_fraction_l57_57885

theorem simplify_nested_fraction :
  (1 : ℚ) / (1 + (1 / (3 + (1 / 4)))) = 13 / 17 :=
by
  sorry

end simplify_nested_fraction_l57_57885


namespace min_spend_for_free_delivery_l57_57442

theorem min_spend_for_free_delivery : 
  let chicken_price := 1.5 * 6.00
  let lettuce_price := 3.00
  let tomato_price := 2.50
  let sweet_potato_price := 4 * 0.75
  let broccoli_price := 2 * 2.00
  let brussel_sprouts_price := 2.50
  let current_total := chicken_price + lettuce_price + tomato_price + sweet_potato_price + broccoli_price + brussel_sprouts_price
  let additional_needed := 11.00 
  let minimum_spend := current_total + additional_needed
  minimum_spend = 35.00 :=
by
  sorry

end min_spend_for_free_delivery_l57_57442


namespace pizza_eaten_after_six_trips_l57_57710

theorem pizza_eaten_after_six_trips :
  (1 / 3) + (1 / 3) / 2 + (1 / 3) / 2 / 2 + (1 / 3) / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 / 2 = 21 / 32 :=
by
  sorry

end pizza_eaten_after_six_trips_l57_57710


namespace mod_congruence_zero_iff_l57_57232

theorem mod_congruence_zero_iff
  (a b c d n : ℕ)
  (h1 : a * c ≡ 0 [MOD n])
  (h2 : b * c + a * d ≡ 0 [MOD n]) :
  b * c ≡ 0 [MOD n] ∧ a * d ≡ 0 [MOD n] :=
by
  sorry

end mod_congruence_zero_iff_l57_57232


namespace geometric_sequence_a5_l57_57960

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a3 : a 3 = -1)
  (h_a7 : a 7 = -9) : a 5 = -3 := 
sorry

end geometric_sequence_a5_l57_57960


namespace complex_power_difference_l57_57289

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := 
by sorry

end complex_power_difference_l57_57289


namespace calculate_expression_l57_57913

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 :=
by {
  -- hint to the Lean prover to consider associative property
  sorry
}

end calculate_expression_l57_57913


namespace find_number_l57_57151

theorem find_number : ∃ n : ℕ, n = (15 * 6) + 5 := 
by sorry

end find_number_l57_57151


namespace trigonometric_identity_l57_57775

theorem trigonometric_identity
  (θ : ℝ)
  (h : (2 + (1 / (Real.sin θ) ^ 2)) / (1 + Real.sin θ) = 1) :
  (1 + Real.sin θ) * (2 + Real.cos θ) = 4 :=
sorry

end trigonometric_identity_l57_57775


namespace impossible_odd_n_m_l57_57465

theorem impossible_odd_n_m (n m : ℤ) (h : Even (n^2 + m + n * m)) : ¬ (Odd n ∧ Odd m) :=
by
  intro h1
  sorry

end impossible_odd_n_m_l57_57465


namespace area_XMY_l57_57805

-- Definitions
structure Triangle :=
(area : ℝ)

def ratio (a b : ℝ) : Prop := ∃ k : ℝ, (a = k * b)

-- Given conditions
variables {XYZ XMY YZ MY : ℝ}
variables (h1 : ratio XYZ 35)
variables (h2 : ratio (XM / MY) (5 / 2))

-- Theorem to prove
theorem area_XMY (hYZ_ratio : YZ = XM + MY) (hshared_height : true) : XMY = 10 :=
by
  sorry

end area_XMY_l57_57805


namespace largest_divisor_of_expression_l57_57895

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 - n) := 
sorry

end largest_divisor_of_expression_l57_57895


namespace smallest_n_congruence_l57_57443

theorem smallest_n_congruence :
  ∃ n : ℕ+, 537 * (n : ℕ) % 30 = 1073 * (n : ℕ) % 30 ∧ (∀ m : ℕ+, 537 * (m : ℕ) % 30 = 1073 * (m : ℕ) % 30 → (m : ℕ) < n → false) :=
  sorry

end smallest_n_congruence_l57_57443


namespace basketball_children_l57_57326

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l57_57326


namespace part1_profit_in_april_part2_price_reduction_l57_57564

-- Given conditions
def cost_per_bag : ℕ := 16
def original_price_per_bag : ℕ := 30
def reduction_amount : ℕ := 5
def increase_in_sales_rate : ℕ := 20
def original_sales_volume : ℕ := 200
def target_profit : ℕ := 2860

-- Part 1: When the price per bag of noodles is reduced by 5 yuan
def profit_in_april_when_reduced_by_5 (cost_per_bag original_price_per_bag reduction_amount increase_in_sales_rate original_sales_volume : ℕ) : ℕ := 
  let new_price := original_price_per_bag - reduction_amount
  let new_sales_volume := original_sales_volume + (increase_in_sales_rate * reduction_amount)
  let profit_per_bag := new_price - cost_per_bag
  profit_per_bag * new_sales_volume

theorem part1_profit_in_april :
  profit_in_april_when_reduced_by_5 16 30 5 20 200 = 2700 :=
sorry

-- Part 2: Determine the price reduction for a specific target profit
def price_reduction_for_profit (cost_per_bag original_price_per_bag increase_in_sales_rate original_sales_volume target_profit : ℕ) : ℕ :=
  let x := (target_profit - (original_sales_volume * (original_price_per_bag - cost_per_bag))) / (increase_in_sales_rate * (original_price_per_bag - cost_per_bag) - increase_in_sales_rate - original_price_per_bag)
  x

theorem part2_price_reduction :
  price_reduction_for_profit 16 30 20 200 2860 = 3 :=
sorry

end part1_profit_in_april_part2_price_reduction_l57_57564


namespace donut_distribution_l57_57975

theorem donut_distribution :
  ∃ (Alpha Beta Gamma Delta Epsilon : ℕ), 
    Delta = 8 ∧ 
    Beta = 3 * Gamma ∧ 
    Alpha = 2 * Delta ∧ 
    Epsilon = Gamma - 4 ∧ 
    Alpha + Beta + Gamma + Delta + Epsilon = 60 ∧ 
    Alpha = 16 ∧ 
    Beta = 24 ∧ 
    Gamma = 8 ∧ 
    Delta = 8 ∧ 
    Epsilon = 4 :=
by
  sorry

end donut_distribution_l57_57975


namespace place_sweet_hexagons_l57_57523

def sweetHexagon (h : ℝ) : Prop := h = 1
def convexPolygon (A : ℝ) : Prop := A ≥ 1900000
def hexagonPlacementPossible (N : ℕ) : Prop := N ≤ 2000000

theorem place_sweet_hexagons:
  (∀ h, sweetHexagon h) →
  (∃ A, convexPolygon A) →
  (∃ N, hexagonPlacementPossible N) →
  True :=
by
  intros _ _ _ 
  exact True.intro

end place_sweet_hexagons_l57_57523


namespace arithmetic_mean_18_27_45_l57_57526

theorem arithmetic_mean_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_18_27_45_l57_57526


namespace intersection_complement_range_m_l57_57001

open Set

variable (A : Set ℝ) (B : ℝ → Set ℝ) (m : ℝ)

def setA : Set ℝ := Icc (-1 : ℝ) (3 : ℝ)
def setB (m : ℝ) : Set ℝ := Icc m (m + 6)

theorem intersection_complement (m : ℝ) (h : m = 2) : 
  (setA ∩ (setB 2)ᶜ) = Ico (-1 : ℝ) (2 : ℝ) :=
by
  sorry

theorem range_m (m : ℝ) : 
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 :=
by
  sorry

end intersection_complement_range_m_l57_57001


namespace quadratic_real_roots_l57_57161

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_l57_57161


namespace radius_of_circle_eqn_zero_l57_57973

def circle_eqn (x y : ℝ) := x^2 + 8*x + y^2 - 4*y + 20 = 0

theorem radius_of_circle_eqn_zero :
  ∀ x y : ℝ, circle_eqn x y → ∃ r : ℝ, r = 0 :=
by
  intros x y h
  -- Sorry to skip the proof as per instructions
  sorry

end radius_of_circle_eqn_zero_l57_57973


namespace proof_equivalent_l57_57957

variables {α : Type*} [Field α]

theorem proof_equivalent (a b c d e f : α)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 :=
by sorry

end proof_equivalent_l57_57957


namespace train_crosses_bridge_in_30_seconds_l57_57594

/--
A train 155 metres long, travelling at 45 km/hr, can cross a bridge with length 220 metres in 30 seconds.
-/
theorem train_crosses_bridge_in_30_seconds
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_km_per_hr : ℕ)
  (total_distance : ℕ)
  (speed_m_per_s : ℚ)
  (time_seconds : ℚ) 
  (h1 : length_train = 155)
  (h2 : length_bridge = 220)
  (h3 : speed_km_per_hr = 45)
  (h4 : total_distance = length_train + length_bridge)
  (h5 : speed_m_per_s = (speed_km_per_hr * 1000) / 3600)
  (h6 : time_seconds = total_distance / speed_m_per_s) :
  time_seconds = 30 :=
sorry

end train_crosses_bridge_in_30_seconds_l57_57594


namespace find_geometric_arithmetic_progressions_l57_57675

theorem find_geometric_arithmetic_progressions
    (b1 b2 b3 : ℚ)
    (h1 : b2^2 = b1 * b3)
    (h2 : b2 + 2 = (b1 + b3) / 2)
    (h3 : (b2 + 2)^2 = b1 * (b3 + 16)) :
    (b1 = 1 ∧ b2 = 3 ∧ b3 = 9) ∨ (b1 = 1/9 ∧ b2 = -5/9 ∧ b3 = 25/9) :=
  sorry

end find_geometric_arithmetic_progressions_l57_57675


namespace evaluate_expression_eq_neg_one_evaluate_expression_only_value_l57_57251

variable (a y : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : a ≠ 2 * y)
variable (h3 : a ≠ -2 * y)

theorem evaluate_expression_eq_neg_one
  (h : y = -a / 3) :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) ) = -1 := 
sorry

theorem evaluate_expression_only_value :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) = -1 ) ↔ 
  y = -a / 3 := 
sorry

end evaluate_expression_eq_neg_one_evaluate_expression_only_value_l57_57251


namespace initial_students_count_l57_57026

theorem initial_students_count (N T : ℕ) (h1 : T = N * 90) (h2 : (T - 120) / (N - 3) = 95) : N = 33 :=
by
  sorry

end initial_students_count_l57_57026


namespace polygon_sides_l57_57210

-- Definitions based on the conditions provided
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

def sum_exterior_angles : ℝ := 360 

def condition (n : ℕ) : Prop :=
  sum_interior_angles n = 2 * sum_exterior_angles + 180

-- Main theorem based on the correct answer
theorem polygon_sides (n : ℕ) (h : condition n) : n = 7 :=
sorry

end polygon_sides_l57_57210


namespace employee_B_payment_l57_57554

theorem employee_B_payment (x : ℝ) (h1 : ∀ A B : ℝ, A + B = 580) (h2 : A = 1.5 * B) : B = 232 :=
by
  sorry

end employee_B_payment_l57_57554


namespace integer_solution_unique_l57_57110

theorem integer_solution_unique (w x y z : ℤ) :
  w^2 + 11 * x^2 - 8 * y^2 - 12 * y * z - 10 * z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry
 
end integer_solution_unique_l57_57110


namespace min_value_of_expression_l57_57489

theorem min_value_of_expression {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) = 50/9 :=
sorry

end min_value_of_expression_l57_57489


namespace smallest_m_satisfying_condition_l57_57137

def D (n : ℕ) : Finset ℕ := (n.divisors : Finset ℕ)

def F (n i : ℕ) : Finset ℕ :=
  (D n).filter (λ a => a % 4 = i)

def f (n i : ℕ) : ℕ :=
  (F n i).card

theorem smallest_m_satisfying_condition :
  ∃ m : ℕ, f m 0 + f m 1 - f m 2 - f m 3 = 2017 ∧
           m = 2^34 * 3^6 * 7^2 * 11^2 :=
by
  sorry

end smallest_m_satisfying_condition_l57_57137


namespace second_year_undeclared_fraction_l57_57588

def total_students := 12

def fraction_first_year : ℚ := 1 / 4
def fraction_second_year : ℚ := 1 / 2
def fraction_third_year : ℚ := 1 / 6
def fraction_fourth_year : ℚ := 1 / 12

def fraction_undeclared_first_year : ℚ := 4 / 5
def fraction_undeclared_second_year : ℚ := 3 / 4
def fraction_undeclared_third_year : ℚ := 1 / 3
def fraction_undeclared_fourth_year : ℚ := 1 / 6

def students_first_year : ℚ := total_students * fraction_first_year
def students_second_year : ℚ := total_students * fraction_second_year
def students_third_year : ℚ := total_students * fraction_third_year
def students_fourth_year : ℚ := total_students * fraction_fourth_year

def undeclared_first_year : ℚ := students_first_year * fraction_undeclared_first_year
def undeclared_second_year : ℚ := students_second_year * fraction_undeclared_second_year
def undeclared_third_year : ℚ := students_third_year * fraction_undeclared_third_year
def undeclared_fourth_year : ℚ := students_fourth_year * fraction_undeclared_fourth_year

theorem second_year_undeclared_fraction :
  (undeclared_second_year / total_students) = 1 / 3 :=
by
  sorry  -- Proof to be provided

end second_year_undeclared_fraction_l57_57588


namespace width_of_room_l57_57017

theorem width_of_room 
  (length : ℝ) 
  (cost : ℝ) 
  (rate : ℝ) 
  (h_length : length = 6.5) 
  (h_cost : cost = 10725) 
  (h_rate : rate = 600) 
  : (cost / rate) / length = 2.75 :=
by
  rw [h_length, h_cost, h_rate]
  norm_num

end width_of_room_l57_57017


namespace find_k_l57_57584

-- Define the vector structures for i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define the vectors a and b based on i, j, and k
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Statement of the theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by sorry

end find_k_l57_57584


namespace factorize_square_diff_factorize_common_factor_l57_57690

-- Problem 1: Difference of squares
theorem factorize_square_diff (x : ℝ) : 4 * x^2 - 9 = (2 * x + 3) * (2 * x - 3) := 
by
  sorry

-- Problem 2: Factoring out common terms
theorem factorize_common_factor (a b x y : ℝ) (h : y - x = -(x - y)) : 
  2 * a * (x - y) - 3 * b * (y - x) = (x - y) * (2 * a + 3 * b) := 
by
  sorry

end factorize_square_diff_factorize_common_factor_l57_57690


namespace inequality_proof_l57_57882

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a + 1 / b + 9 / c + 25 / d) ≥ (100 / (a + b + c + d)) :=
by
  sorry

end inequality_proof_l57_57882


namespace average_marks_first_class_l57_57455

theorem average_marks_first_class (A : ℝ) :
  let students_class1 := 55
  let students_class2 := 48
  let avg_class2 := 58
  let avg_all := 59.067961165048544
  let total_students := 103
  let total_marks := avg_all * total_students
  total_marks = (A * students_class1) + (avg_class2 * students_class2) 
  → A = 60 :=
by
  sorry

end average_marks_first_class_l57_57455


namespace pebble_sequence_10_l57_57532

-- A definition for the sequence based on the given conditions and pattern.
def pebble_sequence : ℕ → ℕ
| 0 => 1
| 1 => 5
| 2 => 12
| 3 => 22
| (n + 4) => pebble_sequence (n + 3) + (3 * (n + 1) + 1)

-- Theorem that states the value at the 10th position in the sequence.
theorem pebble_sequence_10 : pebble_sequence 9 = 145 :=
sorry

end pebble_sequence_10_l57_57532


namespace determine_a_l57_57612

theorem determine_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x + 1 > 0) ↔ a = 2 := by
  sorry

end determine_a_l57_57612


namespace prove_a_is_perfect_square_l57_57889

-- Definition of a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Main theorem statement
theorem prove_a_is_perfect_square 
  (a b : ℕ) 
  (hb_odd : b % 2 = 1) 
  (h_integer : ∃ k : ℕ, ((a + b) * (a + b) + 4 * a) = k * a * b) :
  is_perfect_square a :=
sorry

end prove_a_is_perfect_square_l57_57889


namespace trajectory_of_M_l57_57129

-- Define the conditions: P moves on the circle, and Q is fixed
variable (P Q M : ℝ × ℝ)
variable (P_moves_on_circle : P.1^2 + P.2^2 = 1)
variable (Q_fixed : Q = (3, 0))
variable (M_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))

-- Theorem statement
theorem trajectory_of_M :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end trajectory_of_M_l57_57129


namespace yard_length_calculation_l57_57364

theorem yard_length_calculation (n_trees : ℕ) (distance : ℕ) (h1 : n_trees = 26) (h2 : distance = 32) : (n_trees - 1) * distance = 800 :=
by
  -- This is where the proof would go.
  sorry

end yard_length_calculation_l57_57364


namespace blue_balls_prob_l57_57968

def prob_same_color (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

theorem blue_balls_prob {n : ℕ} (h : prob_same_color n = 1 / 2) : n = 1 ∨ n = 9 :=
by
  sorry

end blue_balls_prob_l57_57968


namespace Sequential_structure_not_conditional_l57_57260

-- Definitions based on provided conditions
def is_conditional (s : String) : Prop :=
  s = "Loop structure" ∨ s = "If structure" ∨ s = "Until structure"

-- Theorem stating that Sequential structure is the one that doesn't contain a conditional judgment box
theorem Sequential_structure_not_conditional :
  ¬ is_conditional "Sequential structure" :=
by
  intro h
  cases h <;> contradiction

end Sequential_structure_not_conditional_l57_57260


namespace n_squared_plus_2n_plus_3_mod_50_l57_57738

theorem n_squared_plus_2n_plus_3_mod_50 (n : ℤ) (hn : n % 50 = 49) : (n^2 + 2 * n + 3) % 50 = 2 := 
sorry

end n_squared_plus_2n_plus_3_mod_50_l57_57738


namespace moles_of_NaCl_l57_57541

def moles_of_reactants (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

theorem moles_of_NaCl (NaCl KNO3 NaNO3 KCl : ℕ) 
  (h : moles_of_reactants NaCl KNO3 NaNO3 KCl) 
  (h2 : KNO3 = 1)
  (h3 : NaNO3 = 1) :
  NaCl = 1 :=
by
  sorry

end moles_of_NaCl_l57_57541


namespace greater_solution_of_quadratic_eq_l57_57927

theorem greater_solution_of_quadratic_eq (x : ℝ) : 
  (∀ y : ℝ, y^2 + 20 * y - 96 = 0 → (y = 4)) :=
sorry

end greater_solution_of_quadratic_eq_l57_57927


namespace Emily_walks_more_distance_than_Troy_l57_57451

theorem Emily_walks_more_distance_than_Troy (Troy_distance Emily_distance : ℕ) (days : ℕ) 
  (hTroy : Troy_distance = 75) (hEmily : Emily_distance = 98) (hDays : days = 5) : 
  ((Emily_distance * 2 - Troy_distance * 2) * days) = 230 :=
by
  sorry

end Emily_walks_more_distance_than_Troy_l57_57451


namespace p_adic_valuation_of_factorial_l57_57314

noncomputable def digit_sum (n p : ℕ) : ℕ :=
  -- Definition for sum of digits of n in base p
  sorry

def p_adic_valuation (n factorial : ℕ) (p : ℕ) : ℕ :=
  -- Representation of p-adic valuation of n!
  sorry

theorem p_adic_valuation_of_factorial (n p : ℕ) (hp: p > 1):
  p_adic_valuation n.factorial p = (n - digit_sum n p) / (p - 1) :=
sorry

end p_adic_valuation_of_factorial_l57_57314


namespace gasoline_price_decrease_l57_57801

theorem gasoline_price_decrease (a : ℝ) (h : 0 ≤ a) :
  8.1 * (1 - a / 100) ^ 2 = 7.8 :=
sorry

end gasoline_price_decrease_l57_57801


namespace polynomial_integer_roots_l57_57635

theorem polynomial_integer_roots :
  ∀ x : ℤ, (x^3 - 3*x^2 - 10*x + 20 = 0) ↔ (x = -2 ∨ x = 5) :=
by
  sorry

end polynomial_integer_roots_l57_57635


namespace calculate_area_of_pentagon_l57_57630

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℝ :=
  let triangle_area := (1/2 : ℝ) * b * a
  let trapezoid_area := (1/2 : ℝ) * (c + e) * d
  triangle_area + trapezoid_area

theorem calculate_area_of_pentagon : area_of_pentagon 18 25 28 30 25 = 1020 :=
sorry

end calculate_area_of_pentagon_l57_57630


namespace problem1_problem2_problem3_problem4_l57_57299

variable (a b c : ℝ)

theorem problem1 : a^4 * (a^2)^3 = a^10 :=
by
  sorry

theorem problem2 : 2 * a^3 * b^2 * c / (1 / 3 * a^2 * b) = 6 * a * b * c :=
by
  sorry

theorem problem3 : 6 * a * (1 / 3 * a * b - b) - (2 * a * b + b) * (a - 1) = -5 * a * b + b :=
by
  sorry

theorem problem4 : (a - 2)^2 - (3 * a + 2 * b) * (3 * a - 2 * b) = -8 * a^2 - 4 * a + 4 + 4 * b^2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l57_57299


namespace probability_sum_of_two_draws_is_three_l57_57199

theorem probability_sum_of_two_draws_is_three :
  let outcomes := [(1, 1), (1, 2), (2, 1), (2, 2)]
  let favorable := [(1, 2), (2, 1)]
  (favorable.length : ℚ) / (outcomes.length : ℚ) = 1 / 2 :=
by
  sorry

end probability_sum_of_two_draws_is_three_l57_57199


namespace concave_quadrilateral_area_l57_57178

noncomputable def area_of_concave_quadrilateral (AB BC CD AD : ℝ) (angle_BCD : ℝ) : ℝ :=
  let BD := Real.sqrt (BC * BC + CD * CD)
  let area_ABD := 0.5 * AB * BD
  let area_BCD := 0.5 * BC * CD
  area_ABD - area_BCD

theorem concave_quadrilateral_area :
  ∀ (AB BC CD AD : ℝ) (angle_BCD : ℝ),
    angle_BCD = Real.pi / 2 ∧ AB = 12 ∧ BC = 4 ∧ CD = 3 ∧ AD = 13 → 
    area_of_concave_quadrilateral AB BC CD AD angle_BCD = 24 :=
by
  intros AB BC CD AD angle_BCD h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end concave_quadrilateral_area_l57_57178


namespace total_trip_length_l57_57516

theorem total_trip_length :
  ∀ (d : ℝ), 
    (∀ fuel_per_mile : ℝ, fuel_per_mile = 0.03 →
      ∀ battery_miles : ℝ, battery_miles = 50 →
      ∀ avg_miles_per_gallon : ℝ, avg_miles_per_gallon = 50 →
      (d / (fuel_per_mile * (d - battery_miles))) = avg_miles_per_gallon →
      d = 150) := 
by
  intros d fuel_per_mile fuel_per_mile_eq battery_miles battery_miles_eq avg_miles_per_gallon avg_miles_per_gallon_eq trip_condition
  sorry

end total_trip_length_l57_57516


namespace number_of_winning_scores_l57_57060

theorem number_of_winning_scores : 
  ∃ (scores: ℕ), scores = 19 := by
  sorry

end number_of_winning_scores_l57_57060


namespace largest_multiple_of_15_less_than_500_l57_57381

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l57_57381


namespace trees_left_after_typhoon_and_growth_l57_57065

-- Conditions
def initial_trees : ℕ := 9
def trees_died_in_typhoon : ℕ := 4
def new_trees : ℕ := 5

-- Question (Proof Problem)
theorem trees_left_after_typhoon_and_growth : 
  initial_trees - trees_died_in_typhoon + new_trees = 10 := 
by
  sorry

end trees_left_after_typhoon_and_growth_l57_57065


namespace Mikail_birthday_money_l57_57568

theorem Mikail_birthday_money (x : ℕ) (h1 : x = 3 + 3 * 3) : 5 * x = 60 := 
by 
  sorry

end Mikail_birthday_money_l57_57568


namespace petes_original_number_l57_57661

theorem petes_original_number (x : ℤ) (h : 4 * (2 * x + 20) = 200) : x = 15 :=
sorry

end petes_original_number_l57_57661


namespace volume_ratio_of_cubes_l57_57330

-- Given conditions
def edge_length_smaller_cube : ℝ := 6
def edge_length_larger_cube : ℝ := 12

-- Problem statement
theorem volume_ratio_of_cubes : 
  (edge_length_smaller_cube / edge_length_larger_cube) ^ 3 = (1 / 8) := 
by
  sorry

end volume_ratio_of_cubes_l57_57330


namespace inequality_solution_l57_57405

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (2021 * (x ^ 10) - 1 ≥ 2020 * x) ↔ (x = 1) :=
sorry

end inequality_solution_l57_57405


namespace scientific_notation_of_634000000_l57_57428

theorem scientific_notation_of_634000000 :
  634000000 = 6.34 * 10 ^ 8 := 
sorry

end scientific_notation_of_634000000_l57_57428


namespace mike_age_l57_57864

theorem mike_age : ∀ (m M : ℕ), m = M - 18 ∧ m + M = 54 → m = 18 :=
by
  intros m M
  intro h
  sorry

end mike_age_l57_57864


namespace multiplication_modulo_l57_57006

theorem multiplication_modulo :
  ∃ n : ℕ, (253 * 649 ≡ n [MOD 100]) ∧ (0 ≤ n) ∧ (n < 100) ∧ (n = 97) := 
by
  sorry

end multiplication_modulo_l57_57006


namespace mandatory_state_tax_rate_l57_57469

theorem mandatory_state_tax_rate 
  (MSRP : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) (tax_rate : ℝ) 
  (insurance_cost : ℝ := insurance_rate * MSRP)
  (cost_before_tax : ℝ := MSRP + insurance_cost)
  (tax_amount : ℝ := total_paid - cost_before_tax) :
  MSRP = 30 → total_paid = 54 → insurance_rate = 0.2 → 
  tax_amount / cost_before_tax * 100 = tax_rate →
  tax_rate = 50 :=
by
  intros MSRP_val paid_val ins_rate_val comp_tax_rate
  sorry

end mandatory_state_tax_rate_l57_57469


namespace problem_solution_l57_57674

-- We assume x and y are real numbers.
variables (x y : ℝ)

-- Our conditions
def condition1 : Prop := |x| - x + y = 6
def condition2 : Prop := x + |y| + y = 8

-- The goal is to prove that x + y = 30 under the given conditions.
theorem problem_solution (hx : condition1 x y) (hy : condition2 x y) : x + y = 30 :=
sorry

end problem_solution_l57_57674


namespace min_bills_required_l57_57601

-- Conditions
def ten_dollar_bills := 13
def five_dollar_bills := 11
def one_dollar_bills := 17
def total_amount := 128

-- Prove that Tim can pay exactly $128 with the minimum number of bills being 16
theorem min_bills_required : (∃ ten five one : ℕ, 
    ten ≤ ten_dollar_bills ∧
    five ≤ five_dollar_bills ∧
    one ≤ one_dollar_bills ∧
    ten * 10 + five * 5 + one = total_amount ∧
    ten + five + one = 16) :=
by
  -- We will skip the proof for now
  sorry

end min_bills_required_l57_57601


namespace intersection_M_N_l57_57357

-- Definitions for the sets M and N based on the given conditions
def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }

-- The statement we need to prove
theorem intersection_M_N : M ∩ N = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l57_57357


namespace tomatoes_ready_for_sale_l57_57221

-- Define all conditions
def initial_shipment := 1000 -- kg of tomatoes on Friday
def sold_on_saturday := 300 -- kg of tomatoes sold on Saturday
def rotten_on_sunday := 200 -- kg of tomatoes rotted on Sunday
def additional_shipment := 2 * initial_shipment -- kg of tomatoes arrived on Monday

-- Define the final calculation to prove
theorem tomatoes_ready_for_sale : 
  initial_shipment - sold_on_saturday - rotten_on_sunday + additional_shipment = 2500 := 
by
  sorry

end tomatoes_ready_for_sale_l57_57221


namespace total_distance_combined_l57_57697

/-- The conditions for the problem
Each car has 50 liters of fuel.
Car U has a fuel efficiency of 20 liters per 100 kilometers.
Car V has a fuel efficiency of 25 liters per 100 kilometers.
Car W has a fuel efficiency of 5 liters per 100 kilometers.
Car X has a fuel efficiency of 10 liters per 100 kilometers.
-/
theorem total_distance_combined (fuel_U fuel_V fuel_W fuel_X : ℕ) (eff_U eff_V eff_W eff_X : ℕ) (fuel : ℕ)
  (hU : fuel_U = 50) (hV : fuel_V = 50) (hW : fuel_W = 50) (hX : fuel_X = 50)
  (eU : eff_U = 20) (eV : eff_V = 25) (eW : eff_W = 5) (eX : eff_X = 10) :
  (fuel_U * 100 / eff_U) + (fuel_V * 100 / eff_V) + (fuel_W * 100 / eff_W) + (fuel_X * 100 / eff_X) = 1950 := by 
  sorry

end total_distance_combined_l57_57697


namespace number_of_books_before_purchase_l57_57628

theorem number_of_books_before_purchase (x : ℕ) (h1 : x + 140 = (27 / 25) * x) : x = 1750 :=
by
  sorry

end number_of_books_before_purchase_l57_57628


namespace rhombus_area_l57_57750

-- Define the rhombus with given conditions
def rhombus (a d1 d2 : ℝ) : Prop :=
  a = 9 ∧ abs (d1 - d2) = 10 

-- The theorem stating the area of the rhombus
theorem rhombus_area (a d1 d2 : ℝ) (h : rhombus a d1 d2) : 
  (d1 * d2) / 2 = 72 :=
by
  sorry

#check rhombus_area

end rhombus_area_l57_57750


namespace expression_value_l57_57201

def a : ℝ := 0.96
def b : ℝ := 0.1

theorem expression_value : (a^3 - (b^3 / a^2) + 0.096 + b^2) = 0.989651 :=
by
  sorry

end expression_value_l57_57201


namespace initial_number_of_boarders_l57_57543

theorem initial_number_of_boarders (B D : ℕ) (h1 : B / D = 2 / 5) (h2 : (B + 15) / D = 1 / 2) : B = 60 :=
by
  -- Proof needs to be provided here
  sorry

end initial_number_of_boarders_l57_57543


namespace small_bonsai_sold_eq_l57_57021

-- Define the conditions
def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- The proof problem: Prove that the number of small bonsai sold is 3
theorem small_bonsai_sold_eq : ∃ x : ℕ, 30 * x + 20 * 5 = 190 ∧ x = 3 :=
by
  sorry

end small_bonsai_sold_eq_l57_57021


namespace evaluate_expr_at_neg3_l57_57749

-- Define the expression
def expr (x : ℤ) : ℤ := (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)

-- Define the proposition to be proven
theorem evaluate_expr_at_neg3 : expr (-3) = -26 := by
  sorry

end evaluate_expr_at_neg3_l57_57749


namespace ratio_e_a_l57_57452

theorem ratio_e_a (a b c d e : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4) :
  e / a = 8 / 15 := 
by
  sorry

end ratio_e_a_l57_57452


namespace bob_paid_24_percent_of_SRP_l57_57113

theorem bob_paid_24_percent_of_SRP
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ) -- Marked Price (MP)
  (price_bob_paid : ℝ) -- Price Bob Paid
  (h1 : MP = 0.60 * P) -- Condition 1: MP is 60% of SRP
  (h2 : price_bob_paid = 0.40 * MP) -- Condition 2: Bob paid 40% of the MP
  : (price_bob_paid / P) * 100 = 24 := -- Bob paid 24% of the SRP
by
  sorry

end bob_paid_24_percent_of_SRP_l57_57113


namespace sum_of_edges_of_rectangular_solid_l57_57797

theorem sum_of_edges_of_rectangular_solid 
(volume : ℝ) (surface_area : ℝ) (a b c : ℝ)
(h1 : volume = a * b * c)
(h2 : surface_area = 2 * (a * b + b * c + c * a))
(h3 : ∃ s : ℝ, s ≠ 0 ∧ a = b / s ∧ c = b * s)
(h4 : volume = 512)
(h5 : surface_area = 384) :
a + b + c = 24 := 
sorry

end sum_of_edges_of_rectangular_solid_l57_57797


namespace lcm_subtract100_correct_l57_57187

noncomputable def lcm1364_884_subtract_100 : ℕ :=
  let a := 1364
  let b := 884
  let lcm_ab := Nat.lcm a b
  lcm_ab - 100

theorem lcm_subtract100_correct : lcm1364_884_subtract_100 = 1509692 := by
  sorry

end lcm_subtract100_correct_l57_57187


namespace multiples_of_4_between_88_and_104_l57_57877

theorem multiples_of_4_between_88_and_104 : 
  ∃ n, (104 - 4 * 23 = n) ∧ n = 88 ∧ ( ∀ x, (x ≥ 88 ∧ x ≤ 104 ∧ x % 4 = 0) → ( x - 88) / 4 < 24) :=
by
  sorry

end multiples_of_4_between_88_and_104_l57_57877


namespace cross_section_perimeter_l57_57256

-- Define the lengths of the diagonals AC and BD.
def length_AC : ℝ := 8
def length_BD : ℝ := 12

-- Define the perimeter calculation for the cross-section quadrilateral
-- that passes through the midpoint E of AB and is parallel to BD and AC.
theorem cross_section_perimeter :
  let side1 := length_AC / 2
  let side2 := length_BD / 2
  let perimeter := 2 * (side1 + side2)
  perimeter = 20 :=
by
  sorry

end cross_section_perimeter_l57_57256


namespace sum_of_abc_l57_57305

theorem sum_of_abc (a b c : ℝ) (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0) :
  a + b + c = 18 :=
sorry

end sum_of_abc_l57_57305


namespace part1_solution_set_part2_range_of_a_l57_57450

noncomputable def f (x a : ℝ) : ℝ := -x^2 + a * x + 4

def g (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem part1_solution_set (a : ℝ := 1) :
  {x : ℝ | f x a ≥ g x} = { x : ℝ | -1 ≤ x ∧ x ≤ (Real.sqrt 17 - 1) / 2 } :=
by
  sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x ∈ [-1,1], f x a ≥ g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end part1_solution_set_part2_range_of_a_l57_57450


namespace max_wx_xy_yz_zt_l57_57759

theorem max_wx_xy_yz_zt {w x y z t : ℕ} (h_sum : w + x + y + z + t = 120)
  (hnn_w : 0 ≤ w) (hnn_x : 0 ≤ x) (hnn_y : 0 ≤ y) (hnn_z : 0 ≤ z) (hnn_t : 0 ≤ t) :
  wx + xy + yz + zt ≤ 3600 := 
sorry

end max_wx_xy_yz_zt_l57_57759


namespace revised_lemonade_calories_l57_57411

def lemonade (lemon_grams sugar_grams water_grams lemon_calories_per_50grams sugar_calories_per_100grams : ℕ) :=
  let lemon_cals := lemon_calories_per_50grams
  let sugar_cals := (sugar_grams / 100) * sugar_calories_per_100grams
  let water_cals := 0
  lemon_cals + sugar_cals + water_cals

def lemonade_weight (lemon_grams sugar_grams water_grams : ℕ) :=
  lemon_grams + sugar_grams + water_grams

def caloric_density (total_calories : ℕ) (total_weight : ℕ) := (total_calories : ℚ) / total_weight

def calories_in_serving (density : ℚ) (serving : ℕ) := density * serving

theorem revised_lemonade_calories :
  let lemon_calories := 32
  let sugar_calories := 579
  let total_calories := lemonade 50 150 300 lemon_calories sugar_calories
  let total_weight := lemonade_weight 50 150 300
  let density := caloric_density total_calories total_weight
  let serving_calories := calories_in_serving density 250
  serving_calories = 305.5 := sorry

end revised_lemonade_calories_l57_57411


namespace trig_identity_proof_l57_57152

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def sin_30 := Real.sin (Real.pi / 6)
noncomputable def cos_60 := Real.cos (Real.pi / 3)

theorem trig_identity_proof :
  (1 - (1 / cos_30)) * (1 + (2 / sin_60)) * (1 - (1 / sin_30)) * (1 + (2 / cos_60)) = (25 - 10 * Real.sqrt 3) / 3 := by
  sorry

end trig_identity_proof_l57_57152


namespace eq_y_as_x_l57_57783

theorem eq_y_as_x (y x : ℝ) : 
  (y = 2*x - 3*y) ∨ (x = 2 - 3*y) ∨ (-y = 2*x - 1) ∨ (y = x) → (y = x) :=
by
  sorry

end eq_y_as_x_l57_57783


namespace tangent_normal_at_t1_l57_57591

noncomputable def curve_param_x (t: ℝ) : ℝ := Real.arcsin (t / Real.sqrt (1 + t^2))
noncomputable def curve_param_y (t: ℝ) : ℝ := Real.arccos (1 / Real.sqrt (1 + t^2))

theorem tangent_normal_at_t1 : 
  curve_param_x 1 = Real.pi / 4 ∧
  curve_param_y 1 = Real.pi / 4 ∧
  ∃ (x y : ℝ), (y = 2*x - Real.pi/4) ∧ (y = -x/2 + 3*Real.pi/8) :=
  sorry

end tangent_normal_at_t1_l57_57591


namespace jane_mean_after_extra_credit_l57_57446

-- Define Jane's original scores
def original_scores : List ℤ := [82, 90, 88, 95, 91]

-- Define the extra credit points
def extra_credit : ℤ := 2

-- Define the mean calculation after extra credit
def mean_after_extra_credit (scores : List ℤ) (extra : ℤ) : ℚ :=
  let total_sum := scores.sum + (scores.length * extra)
  total_sum / scores.length

theorem jane_mean_after_extra_credit :
  mean_after_extra_credit original_scores extra_credit = 91.2 := by
  sorry

end jane_mean_after_extra_credit_l57_57446


namespace cos_identity_l57_57603

theorem cos_identity (α : ℝ) (h : Real.cos (Real.pi / 8 - α) = 1 / 6) :
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end cos_identity_l57_57603


namespace find_simple_interest_sum_l57_57758

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

noncomputable def simple_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * r * n / 100

theorem find_simple_interest_sum (P CIsum : ℝ)
  (simple_rate : ℝ) (simple_years : ℕ)
  (compound_rate : ℝ) (compound_years : ℕ)
  (compound_principal : ℝ)
  (hP : simple_interest P simple_rate simple_years = CIsum)
  (hCI : CIsum = (compound_interest compound_principal compound_rate compound_years - compound_principal) / 2) :
  P = 1272 :=
by
  sorry

end find_simple_interest_sum_l57_57758


namespace find_the_number_l57_57413

theorem find_the_number :
  ∃ x : ℤ, 65 + (x * 12) / (180 / 3) = 66 ∧ x = 5 :=
by
  existsi (5 : ℤ)
  sorry

end find_the_number_l57_57413


namespace find_theta_ratio_l57_57115

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem find_theta_ratio (θ : ℝ) 
  (h : det2x2 (Real.sin θ) 2 (Real.cos θ) 3 = 0) : 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 := 
by 
  sorry

end find_theta_ratio_l57_57115


namespace coat_price_reduction_l57_57911

theorem coat_price_reduction 
    (original_price : ℝ) 
    (reduction_amount : ℝ) 
    (h1 : original_price = 500) 
    (h2 : reduction_amount = 300) : 
    (reduction_amount / original_price) * 100 = 60 := 
by 
  sorry

end coat_price_reduction_l57_57911


namespace unique_solution_implies_relation_l57_57592

open Nat

noncomputable def unique_solution (a b : ℤ) :=
  ∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b

theorem unique_solution_implies_relation (a b : ℤ) :
  unique_solution a b → b = (a * a) / 4 := sorry

end unique_solution_implies_relation_l57_57592


namespace petya_winning_probability_l57_57500

noncomputable def petya_wins_probability : ℚ :=
  (1 / 4) ^ 4

-- The main theorem statement
theorem petya_winning_probability :
  petya_wins_probability = 1 / 256 :=
by sorry

end petya_winning_probability_l57_57500


namespace probability_of_color_difference_l57_57173

noncomputable def probability_of_different_colors (n m : ℕ) : ℚ :=
  (Nat.choose n m : ℚ) * (1/2)^n

theorem probability_of_color_difference :
  probability_of_different_colors 8 4 = 35/128 :=
by
  sorry

end probability_of_color_difference_l57_57173


namespace train_passing_time_l57_57984

theorem train_passing_time
  (length_of_train : ℝ)
  (speed_in_kmph : ℝ)
  (conversion_factor : ℝ)
  (speed_in_mps : ℝ)
  (time : ℝ)
  (H1 : length_of_train = 65)
  (H2 : speed_in_kmph = 36)
  (H3 : conversion_factor = 5 / 18)
  (H4 : speed_in_mps = speed_in_kmph * conversion_factor)
  (H5 : time = length_of_train / speed_in_mps) :
  time = 6.5 :=
by
  sorry

end train_passing_time_l57_57984


namespace proof_statements_correct_l57_57417

variable (candidates : Nat) (sample_size : Nat)

def is_sampling_survey (survey_type : String) : Prop :=
  survey_type = "sampling"

def is_population (pop_size sample_size : Nat) : Prop :=
  (pop_size = 60000) ∧ (sample_size = 1000)

def is_sample (sample_size pop_size : Nat) : Prop :=
  sample_size < pop_size

def sample_size_correct (sample_size : Nat) : Prop :=
  sample_size = 1000

theorem proof_statements_correct :
  ∀ (survey_type : String) (pop_size sample_size : Nat),
  is_sampling_survey survey_type →
  is_population pop_size sample_size →
  is_sample sample_size pop_size →
  sample_size_correct sample_size →
  survey_type = "sampling" ∧
  pop_size = 60000 ∧
  sample_size = 1000 :=
by
  intros survey_type pop_size sample_size hs hp hsamp hsiz
  sorry

end proof_statements_correct_l57_57417


namespace find_c_for_two_solutions_in_real_l57_57333

noncomputable def system_two_solutions (x y c : ℝ) : Prop := (|x + y| = 2007 ∧ |x - y| = c)

theorem find_c_for_two_solutions_in_real : ∃ c : ℝ, (∀ x y : ℝ, system_two_solutions x y c) ↔ (c = 0) :=
by
  sorry

end find_c_for_two_solutions_in_real_l57_57333


namespace quadratic_prime_roots_l57_57774

theorem quadratic_prime_roots (k : ℕ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p + q = 101 → p * q = k → False :=
by
  sorry

end quadratic_prime_roots_l57_57774


namespace ladder_base_l57_57291

theorem ladder_base (h : ℝ) (b : ℝ) (l : ℝ)
  (h_eq : h = 12) (l_eq : l = 15) : b = 9 :=
by
  have hypotenuse := l
  have height := h
  have base := b
  have pythagorean_theorem : height^2 + base^2 = hypotenuse^2 := by sorry 
  sorry

end ladder_base_l57_57291


namespace sin_cos_alpha_eq_fifth_l57_57745

variable {α : ℝ}
variable (h : Real.sin α = 2 * Real.cos α)

theorem sin_cos_alpha_eq_fifth : Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end sin_cos_alpha_eq_fifth_l57_57745


namespace add_salt_solution_l57_57925

theorem add_salt_solution
  (initial_amount : ℕ) (added_concentration : ℕ) (desired_concentration : ℕ)
  (initial_concentration : ℝ) :
  initial_amount = 50 ∧ initial_concentration = 0.4 ∧ added_concentration = 10 ∧ desired_concentration = 25 →
  (∃ (x : ℕ), x = 50 ∧ 
    (initial_concentration * initial_amount + 0.1 * x) / (initial_amount + x) = 0.25) :=
by
  sorry

end add_salt_solution_l57_57925


namespace probability_escher_consecutive_l57_57551

def total_pieces : Nat := 12
def escher_pieces : Nat := 4

theorem probability_escher_consecutive :
  (Nat.factorial 9 * Nat.factorial 4 : ℚ) / Nat.factorial 12 = 1 / 55 := 
sorry

end probability_escher_consecutive_l57_57551


namespace probability_same_color_boxes_l57_57407

def num_neckties := 6
def num_shirts := 5
def num_hats := 4
def num_socks := 3

def num_common_colors := 3

def total_combinations : ℕ := num_neckties * num_shirts * num_hats * num_socks

def same_color_combinations : ℕ := num_common_colors

def same_color_probability : ℚ :=
  same_color_combinations / total_combinations

theorem probability_same_color_boxes :
  same_color_probability = 1 / 120 :=
  by
    -- Proof would go here
    sorry

end probability_same_color_boxes_l57_57407


namespace probability_of_matching_pair_l57_57827

theorem probability_of_matching_pair (blackSocks blueSocks : ℕ) (h_black : blackSocks = 12) (h_blue : blueSocks = 10) : 
  let totalSocks := blackSocks + blueSocks
  let totalWays := Nat.choose totalSocks 2
  let blackPairWays := Nat.choose blackSocks 2
  let bluePairWays := Nat.choose blueSocks 2
  let matchingPairWays := blackPairWays + bluePairWays
  totalWays = 231 ∧ matchingPairWays = 111 → (matchingPairWays : ℚ) / totalWays = 111 / 231 := 
by
  intros
  sorry

end probability_of_matching_pair_l57_57827


namespace circle_origin_range_l57_57701

theorem circle_origin_range (m : ℝ) : 
  (0 - m)^2 + (0 + m)^2 < 4 → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end circle_origin_range_l57_57701


namespace maximum_value_abs_difference_l57_57579

theorem maximum_value_abs_difference (x y : ℝ) 
  (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : 
  |x - y + 1| ≤ 2 :=
sorry

end maximum_value_abs_difference_l57_57579


namespace sum_of_fractions_l57_57306

theorem sum_of_fractions :
  (1/15 + 2/15 + 3/15 + 4/15 + 5/15 + 6/15 + 7/15 + 8/15 + 9/15 + 46/15) = (91/15) := by
  sorry

end sum_of_fractions_l57_57306


namespace boys_in_class_l57_57756

theorem boys_in_class (total_students : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ)
    (h_ratio : ratio_girls = 3) (h_ratio_boys : ratio_boys = 4)
    (h_total_students : total_students = 35) :
    ∃ boys, boys = 20 :=
by
  let k := total_students / (ratio_girls + ratio_boys)
  have hk : k = 5 := by sorry
  let boys := ratio_boys * k
  have h_boys : boys = 20 := by sorry
  exact ⟨boys, h_boys⟩

end boys_in_class_l57_57756


namespace year_proof_l57_57101

variable (n : ℕ)

def packaging_waste_exceeds_threshold (y0 : ℝ) (rate : ℝ) (threshold : ℝ) : Prop :=
  let y := y0 * (rate^n)
  y > threshold

noncomputable def year_when_waste_exceeds := 
  let initial_year := 2015
  let y0 := 4 * 10^6 -- in tons
  let rate := (3.0 / 2.0) -- growth rate per year
  let threshold := 40 * 10^6 -- threshold in tons
  ∃ n, packaging_waste_exceeds_threshold n y0 rate threshold ∧ (initial_year + n = 2021)

theorem year_proof : year_when_waste_exceeds :=
  sorry

end year_proof_l57_57101


namespace second_candidate_extra_marks_l57_57833

theorem second_candidate_extra_marks (T : ℝ) (marks_40_percent : ℝ) (marks_passing : ℝ) (marks_60_percent : ℝ) 
  (h1 : marks_40_percent = 0.40 * T)
  (h2 : marks_passing = 160)
  (h3 : marks_60_percent = 0.60 * T)
  (h4 : marks_passing = marks_40_percent + 40) :
  (marks_60_percent - marks_passing) = 20 :=
by
  sorry

end second_candidate_extra_marks_l57_57833


namespace meetings_percent_l57_57479

/-- Define the lengths of the meetings and total workday in minutes -/
def first_meeting : ℕ := 40
def second_meeting : ℕ := 80
def second_meeting_overlap : ℕ := 10
def third_meeting : ℕ := 30
def workday_minutes : ℕ := 8 * 60

/-- Define the effective duration of the second meeting -/
def effective_second_meeting : ℕ := second_meeting - second_meeting_overlap

/-- Define the total time spent in meetings -/
def total_meeting_time : ℕ := first_meeting + effective_second_meeting + third_meeting

/-- Define the percentage of the workday spent in meetings -/
noncomputable def percent_meeting_time : ℚ := (total_meeting_time * 100 : ℕ) / workday_minutes

/-- Theorem: Given Laura's workday and meeting durations, prove that the percent of her workday spent in meetings is approximately 29.17%. -/
theorem meetings_percent {epsilon : ℚ} (h : epsilon = 0.01) : abs (percent_meeting_time - 29.17) < epsilon :=
sorry

end meetings_percent_l57_57479


namespace probability_of_selecting_quarter_l57_57032

theorem probability_of_selecting_quarter 
  (value_quarters value_nickels value_pennies total_value : ℚ)
  (coin_value_quarter coin_value_nickel coin_value_penny : ℚ) 
  (h1 : value_quarters = 10)
  (h2 : value_nickels = 10)
  (h3 : value_pennies = 10)
  (h4 : coin_value_quarter = 0.25)
  (h5 : coin_value_nickel = 0.05)
  (h6 : coin_value_penny = 0.01)
  (total_coins : ℚ) 
  (h7 : total_coins = (value_quarters / coin_value_quarter) + (value_nickels / coin_value_nickel) + (value_pennies / coin_value_penny)) : 
  (value_quarters / coin_value_quarter) / total_coins = 1 / 31 :=
by
  sorry

end probability_of_selecting_quarter_l57_57032


namespace g_at_6_l57_57462

def g (x : ℝ) : ℝ := 2 * x^4 - 13 * x^3 + 28 * x^2 - 32 * x - 48

theorem g_at_6 : g 6 = 552 :=
by sorry

end g_at_6_l57_57462


namespace julien_contribution_l57_57486

def exchange_rate : ℝ := 1.5
def cost_of_pie : ℝ := 12
def lucas_cad : ℝ := 10

theorem julien_contribution : (cost_of_pie - lucas_cad / exchange_rate) = 16 / 3 := by
  sorry

end julien_contribution_l57_57486


namespace range_of_a_l57_57825

noncomputable def A (x : ℝ) : Prop := x < -2 ∨ x ≥ 1
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) : (∀ x, A x ∨ B x a) ↔ a ≤ -2 :=
by sorry

end range_of_a_l57_57825


namespace geometric_progression_terms_l57_57298

theorem geometric_progression_terms (a b r : ℝ) (n : ℕ) (h1 : 0 < r) (h2: a ≠ 0) (h3 : b = a * r^(n-1)) :
  n = 1 + (Real.log (b / a)) / (Real.log r) :=
by sorry

end geometric_progression_terms_l57_57298


namespace find_a_b_l57_57791

theorem find_a_b (a b : ℕ) (h1 : (a^3 - a^2 + 1) * (b^3 - b^2 + 2) = 2020) : 10 * a + b = 53 :=
by {
  -- Proof to be completed
  sorry
}

end find_a_b_l57_57791


namespace exists_four_distinct_indices_l57_57215

theorem exists_four_distinct_indices
  (a : Fin 5 → ℝ)
  (h : ∀ i, 0 < a i) :
  ∃ i j k l : (Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < 1 / 2 :=
by
  sorry

end exists_four_distinct_indices_l57_57215


namespace maggie_bouncy_balls_l57_57402

theorem maggie_bouncy_balls (yellow_packs green_pack_given green_pack_bought : ℝ)
    (balls_per_pack : ℝ)
    (hy : yellow_packs = 8.0)
    (hg_given : green_pack_given = 4.0)
    (hg_bought : green_pack_bought = 4.0)
    (hbp : balls_per_pack = 10.0) :
    (yellow_packs * balls_per_pack + green_pack_bought * balls_per_pack - green_pack_given * balls_per_pack = 80.0) :=
by
  sorry

end maggie_bouncy_balls_l57_57402


namespace find_number_l57_57350

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
by
  sorry

end find_number_l57_57350


namespace sufficient_and_necessary_condition_l57_57197

theorem sufficient_and_necessary_condition (x : ℝ) :
  x^2 - 4 * x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 :=
sorry

end sufficient_and_necessary_condition_l57_57197


namespace part1_unique_zero_part2_inequality_l57_57785

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x + 1 / x

theorem part1_unique_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

theorem part2_inequality (n : ℕ) (h : n > 0) : 
  Real.log ((n + 1) / n) < 1 / Real.sqrt (n^2 + n) := by
  sorry

end part1_unique_zero_part2_inequality_l57_57785


namespace sum_of_factorials_is_perfect_square_l57_57399

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map factorial |>.sum

theorem sum_of_factorials_is_perfect_square (n : ℕ) (h : n > 0) :
  (∃ m : ℕ, m * m = sum_of_factorials n) ↔ (n = 1 ∨ n = 3) := 
sorry

end sum_of_factorials_is_perfect_square_l57_57399


namespace sum_of_cubes_l57_57616

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := 
sorry

end sum_of_cubes_l57_57616


namespace y_works_in_40_days_l57_57332

theorem y_works_in_40_days :
  ∃ d, (d > 0) ∧ 
  (1/20 + 1/d = 3/40) ∧ 
  d = 40 :=
by
  use 40
  sorry

end y_works_in_40_days_l57_57332


namespace columns_contain_all_numbers_l57_57427

def rearrange (n m k : ℕ) (a : ℕ → ℕ) : ℕ → ℕ :=
  λ i => if i < n - m then a (i + m + 1)
         else if i < n - k - m then a (i - (n - m) + k + 1)
         else a (i - (n - k))

theorem columns_contain_all_numbers
  (n m k: ℕ)
  (h1 : n > 0)
  (h2 : m < n)
  (h3 : k < n)
  (a : ℕ → ℕ)
  (h4 : ∀ i : ℕ, i < n → a i = i + 1) :
  ∀ j : ℕ, j < n → ∃ i : ℕ, i < n ∧ rearrange n m k a i = j + 1 :=
by
  sorry

end columns_contain_all_numbers_l57_57427


namespace weight_of_each_dumbbell_l57_57933

-- Definitions based on conditions
def initial_dumbbells : Nat := 4
def added_dumbbells : Nat := 2
def total_dumbbells : Nat := initial_dumbbells + added_dumbbells -- 6
def total_weight : Nat := 120

-- Theorem statement
theorem weight_of_each_dumbbell (h : total_dumbbells = 6) (w : total_weight = 120) :
  total_weight / total_dumbbells = 20 :=
by
  -- Proof is to be written here
  sorry

end weight_of_each_dumbbell_l57_57933


namespace intersection_complement_l57_57527

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B := {x : ℝ | x < 1}
def complement_B := {x : ℝ | x ≥ 1}

theorem intersection_complement :
  (set_A ∩ complement_B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_l57_57527


namespace rate_of_interest_is_5_percent_l57_57789

-- Defining the conditions as constants
def simple_interest : ℝ := 4016.25
def principal : ℝ := 16065
def time_period : ℝ := 5

-- Proving that the rate of interest is 5%
theorem rate_of_interest_is_5_percent (R : ℝ) : 
  simple_interest = (principal * R * time_period) / 100 → 
  R = 5 :=
by
  intro h
  sorry

end rate_of_interest_is_5_percent_l57_57789


namespace ratio_of_costs_l57_57057

theorem ratio_of_costs (R N : ℝ) (hR : 3 * R = 0.25 * (3 * R + 3 * N)) : N / R = 3 := 
sorry

end ratio_of_costs_l57_57057


namespace count_valid_numbers_l57_57170

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l57_57170


namespace trig_problem_l57_57412

theorem trig_problem 
  (α : ℝ) 
  (h1 : Real.cos α = -1/2) 
  (h2 : 180 * (Real.pi / 180) < α ∧ α < 270 * (Real.pi / 180)) : 
  α = 240 * (Real.pi / 180) :=
sorry

end trig_problem_l57_57412


namespace remainder_of_5_pow_2023_mod_6_l57_57694

theorem remainder_of_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := 
by sorry

end remainder_of_5_pow_2023_mod_6_l57_57694


namespace calculate_fraction_l57_57867

theorem calculate_fraction :
  (-1 / 42) / (1 / 6 - 3 / 14 + 2 / 3 - 2 / 7) = -1 / 14 :=
by
  sorry

end calculate_fraction_l57_57867


namespace largest_n_for_factored_quad_l57_57574

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l57_57574


namespace prob_three_blue_is_correct_l57_57327

-- Definitions corresponding to the problem conditions
def total_jellybeans : ℕ := 20
def blue_jellybeans_start : ℕ := 10
def red_jellybeans : ℕ := 10

-- Probabilities calculation steps as definitions
def prob_first_blue : ℚ := blue_jellybeans_start / total_jellybeans
def prob_second_blue_given_first_blue : ℚ := (blue_jellybeans_start - 1) / (total_jellybeans - 1)
def prob_third_blue_given_first_two_blue : ℚ := (blue_jellybeans_start - 2) / (total_jellybeans - 2)

-- Total probability of drawing three blue jellybeans
def prob_three_blue : ℚ := 
  prob_first_blue *
  prob_second_blue_given_first_blue *
  prob_third_blue_given_first_two_blue

-- Formal statement of the proof problem
theorem prob_three_blue_is_correct : prob_three_blue = 2 / 19 :=
by
  -- Fill the proof here
  sorry

end prob_three_blue_is_correct_l57_57327


namespace cosine_identity_l57_57089

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l57_57089


namespace circles_5_and_8_same_color_l57_57384

-- Define the circles and colors
inductive Color
  | red
  | yellow
  | blue

def circles : Nat := 8

-- Define the adjacency relationship (i.e., directly connected)
-- This is a placeholder. In practice, this would be defined based on the problem's diagram.
def directly_connected (c1 c2 : Nat) : Prop := sorry

-- Simulate painting circles with given constraints
def painted (c : Nat) : Color := sorry

-- Define the conditions
axiom paint_condition (c1 c2 : Nat) (h : directly_connected c1 c2) : painted c1 ≠ painted c2

-- The proof problem: show that circles 5 and 8 must be painted the same color
theorem circles_5_and_8_same_color : painted 5 = painted 8 := 
sorry

end circles_5_and_8_same_color_l57_57384


namespace sara_ticket_cost_l57_57891

noncomputable def calc_ticket_price : ℝ :=
  let rented_movie_cost := 1.59
  let bought_movie_cost := 13.95
  let total_cost := 36.78
  let total_tickets := 2
  let spent_on_tickets := total_cost - (rented_movie_cost + bought_movie_cost)
  spent_on_tickets / total_tickets

theorem sara_ticket_cost : calc_ticket_price = 10.62 := by
  sorry

end sara_ticket_cost_l57_57891


namespace P_inequality_l57_57620

variable {α : Type*} [LinearOrderedField α]

def P (a b c : α) (x : α) : α := a * x^2 + b * x + c

theorem P_inequality (a b c x y : α) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (P a b c (x * y))^2 ≤ (P a b c (x^2)) * (P a b c (y^2)) :=
sorry

end P_inequality_l57_57620


namespace kishore_savings_l57_57902

noncomputable def total_expenses : ℝ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

def percentage_saved : ℝ := 0.10

theorem kishore_savings (salary : ℝ) :
  (total_expenses + percentage_saved * salary) = salary → 
  (percentage_saved * salary = 2077.78) :=
by
  intros h
  rw [← h]
  sorry

end kishore_savings_l57_57902


namespace find_k_for_equation_l57_57539

theorem find_k_for_equation : 
  ∃ k : ℤ, -x^2 - (k + 7) * x - 8 = -(x - 2) * (x - 4) → k = -13 := 
by
  sorry

end find_k_for_equation_l57_57539


namespace greatest_b_not_in_range_l57_57203

theorem greatest_b_not_in_range : ∃ b : ℤ, b = 10 ∧ ∀ x : ℝ, x^2 + (b:ℝ) * x + 20 ≠ -7 := sorry

end greatest_b_not_in_range_l57_57203


namespace pie_eating_contest_l57_57928

theorem pie_eating_contest :
  let first_student_round1 := (5 : ℚ) / 6
  let first_student_round2 := (1 : ℚ) / 6
  let second_student_total := (2 : ℚ) / 3
  let first_student_total := first_student_round1 + first_student_round2
  first_student_total - second_student_total = 1 / 3 :=
by
  sorry

end pie_eating_contest_l57_57928


namespace flat_rate_65_l57_57263

noncomputable def flat_rate_first_night (f n : ℝ) : Prop := 
  (f + 4 * n = 245) ∧ (f + 9 * n = 470)

theorem flat_rate_65 :
  ∃ (f n : ℝ), flat_rate_first_night f n ∧ f = 65 := 
by
  sorry

end flat_rate_65_l57_57263


namespace harmonic_mean_pairs_count_l57_57335

theorem harmonic_mean_pairs_count :
  ∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, p.1 < p.2 ∧ 2 * p.1 * p.2 = 4^15 * (p.1 + p.2)) ∧ s.card = 29 :=
sorry

end harmonic_mean_pairs_count_l57_57335


namespace third_side_length_of_triangle_l57_57105

theorem third_side_length_of_triangle {a b c : ℝ} (h1 : a^2 - 7 * a + 12 = 0) (h2 : b^2 - 7 * b + 12 = 0) 
  (h3 : a ≠ b) (h4 : a = 3 ∨ a = 4) (h5 : b = 3 ∨ b = 4) : 
  (c = 5 ∨ c = Real.sqrt 7) := by
  sorry

end third_side_length_of_triangle_l57_57105


namespace simplify_expression_l57_57522

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 20 + 4 * y = 45 * x + 20 + 4 * y :=
by
  sorry

end simplify_expression_l57_57522


namespace sum_of_infinite_series_l57_57246

theorem sum_of_infinite_series :
  ∑' n, (1 : ℝ) / ((2 * n + 1)^2 - (2 * n - 1)^2) * ((1 : ℝ) / (2 * n - 1)^2 - (1 : ℝ) / (2 * n + 1)^2) = 1 :=
sorry

end sum_of_infinite_series_l57_57246


namespace units_digit_of_exponentiated_product_l57_57142

theorem units_digit_of_exponentiated_product :
  (2 ^ 2101 * 5 ^ 2102 * 11 ^ 2103) % 10 = 0 := 
sorry

end units_digit_of_exponentiated_product_l57_57142


namespace xy_series_16_l57_57643

noncomputable def series (x y : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * (x * y)^n

theorem xy_series_16 (x y : ℝ) (h_series : series x y = 16) (h_abs : |x * y| < 1) :
  (x = 3 / 4 ∧ (y = 1 ∨ y = -1)) :=
sorry

end xy_series_16_l57_57643


namespace length_of_DC_l57_57820

noncomputable def AB : ℝ := 30
noncomputable def sine_A : ℝ := 4 / 5
noncomputable def sine_C : ℝ := 1 / 4
noncomputable def angle_ADB : ℝ := Real.pi / 2

theorem length_of_DC (h_AB : AB = 30) (h_sine_A : sine_A = 4 / 5) (h_sine_C : sine_C = 1 / 4) (h_angle_ADB : angle_ADB = Real.pi / 2) :
  ∃ DC : ℝ, DC = 24 * Real.sqrt 15 :=
by sorry

end length_of_DC_l57_57820


namespace solutions_of_system_l57_57301

theorem solutions_of_system (x y z : ℝ) :
    (x^2 - y = z^2) → (y^2 - z = x^2) → (z^2 - x = y^2) →
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
    (x = 1 ∧ y = 0 ∧ z = -1) ∨ 
    (x = 0 ∧ y = -1 ∧ z = 1) ∨ 
    (x = -1 ∧ y = 1 ∧ z = 0) := by
  sorry

end solutions_of_system_l57_57301


namespace equation_solutions_l57_57976

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃ x : ℝ, ax + b = 0) ∨ (∃ x : ℝ, ∀ y : ℝ, ax + b = 0 → x = y) :=
sorry

end equation_solutions_l57_57976


namespace total_weight_apples_l57_57198

variable (Minjae_weight : ℝ) (Father_weight : ℝ)

theorem total_weight_apples (h1 : Minjae_weight = 2.6) (h2 : Father_weight = 5.98) :
  Minjae_weight + Father_weight = 8.58 :=
by 
  sorry

end total_weight_apples_l57_57198


namespace compound_interest_1200_20percent_3years_l57_57509

noncomputable def compoundInterest (P r : ℚ) (n t : ℕ) : ℚ :=
  let A := P * (1 + r / n) ^ (n * t)
  A - P

theorem compound_interest_1200_20percent_3years :
  compoundInterest 1200 0.20 1 3 = 873.6 :=
by
  sorry

end compound_interest_1200_20percent_3years_l57_57509


namespace sticker_ratio_l57_57444

variable (Dan Tom Bob : ℕ)

theorem sticker_ratio 
  (h1 : Dan = 2 * Tom) 
  (h2 : Tom = Bob) 
  (h3 : Bob = 12) 
  (h4 : Dan = 72) : 
  Tom = Bob :=
by
  sorry

end sticker_ratio_l57_57444


namespace certain_number_l57_57261

-- Define the conditions as variables
variables {x : ℝ}

-- Define the proof problem
theorem certain_number (h : 0.15 * x = 0.025 * 450) : x = 75 :=
sorry

end certain_number_l57_57261


namespace optimal_strategy_for_father_l57_57618

-- Define the individual players
inductive player
| Father 
| Mother 
| Son

open player

-- Define the probabilities of player defeating another
def prob_defeat (p1 p2 : player) : ℝ := sorry  -- These will be defined as per the problem's conditions.

-- Define the probability of father winning given the first matchups
def P_father_vs_mother : ℝ :=
  prob_defeat Father Mother * prob_defeat Father Son +
  prob_defeat Father Mother * prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother +
  prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son * prob_defeat Father Mother

def P_father_vs_son : ℝ :=
  prob_defeat Father Son * prob_defeat Father Mother +
  prob_defeat Father Son * prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son +
  prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother * prob_defeat Father Son

-- Define the optimality condition
theorem optimal_strategy_for_father :
  P_father_vs_mother > P_father_vs_son :=
sorry

end optimal_strategy_for_father_l57_57618


namespace percentage_students_on_trip_l57_57313

variable (total_students : ℕ)
variable (students_more_than_100 : ℕ)
variable (students_on_trip : ℕ)
variable (percentage_more_than_100 : ℝ)
variable (percentage_not_more_than_100 : ℝ)

-- Given conditions
def condition_1 := percentage_more_than_100 = 0.16
def condition_2 := percentage_not_more_than_100 = 0.75

-- The final proof statement
theorem percentage_students_on_trip :
  percentage_more_than_100 * (total_students : ℝ) /
  ((1 - percentage_not_more_than_100)) / (total_students : ℝ) * 100 = 64 :=
by
  sorry

end percentage_students_on_trip_l57_57313


namespace min_value_a_2b_l57_57195

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 3 / b = 1) :
  a + 2 * b = 7 + 2 * Real.sqrt 6 :=
sorry

end min_value_a_2b_l57_57195


namespace termites_ate_black_squares_l57_57334

def chessboard_black_squares_eaten : Nat :=
  12

theorem termites_ate_black_squares :
  let rows := 8;
  let cols := 8;
  let total_squares := rows * cols / 2; -- This simplistically assumes half the squares are black.
  (total_squares = 32) → 
  chessboard_black_squares_eaten = 12 :=
by
  intros h
  sorry

end termites_ate_black_squares_l57_57334


namespace area_of_enclosed_region_l57_57425

theorem area_of_enclosed_region :
  ∃ (r : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 5 = 0 ↔ (x + 3)^2 + (y - 2)^2 = r^2) ∧ (π * r^2 = 14 * π) := by
  sorry

end area_of_enclosed_region_l57_57425


namespace find_a6_plus_a7_plus_a8_l57_57923

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l57_57923


namespace quotient_remainder_difference_l57_57279

theorem quotient_remainder_difference :
  ∀ (N Q Q' R : ℕ), 
    N = 75 →
    N = 5 * Q →
    N = 34 * Q' + R →
    Q > R →
    Q - R = 8 :=
by
  intros N Q Q' R hN hDiv5 hDiv34 hGt
  sorry

end quotient_remainder_difference_l57_57279


namespace expr_undefined_iff_l57_57360

theorem expr_undefined_iff (b : ℝ) : ¬ ∃ y : ℝ, y = (b - 1) / (b^2 - 9) ↔ b = -3 ∨ b = 3 :=
by 
  sorry

end expr_undefined_iff_l57_57360


namespace angle_in_second_quadrant_l57_57311

/-- If α is an angle in the first quadrant, then π - α is an angle in the second quadrant -/
theorem angle_in_second_quadrant (α : Real) (h : 0 < α ∧ α < π / 2) : π - α > π / 2 ∧ π - α < π :=
by
  sorry

end angle_in_second_quadrant_l57_57311


namespace largest_among_a_b_c_d_l57_57091

noncomputable def a : ℝ := Real.log 2022 / Real.log 2021
noncomputable def b : ℝ := Real.log 2023 / Real.log 2022
noncomputable def c : ℝ := 2022 / 2021
noncomputable def d : ℝ := 2023 / 2022

theorem largest_among_a_b_c_d : max a (max b (max c d)) = c := 
sorry

end largest_among_a_b_c_d_l57_57091


namespace dinosaur_book_cost_l57_57826

theorem dinosaur_book_cost (D : ℕ) : 
  (11 + D + 7 = 37) → (D = 19) := 
by 
  intro h
  sorry

end dinosaur_book_cost_l57_57826


namespace min_average_annual_growth_rate_l57_57159

theorem min_average_annual_growth_rate (M : ℝ) (x : ℝ) (h : M * (1 + x)^2 = 2 * M) : x = Real.sqrt 2 - 1 :=
by
  sorry

end min_average_annual_growth_rate_l57_57159


namespace cricket_match_count_l57_57589

theorem cricket_match_count (x : ℕ) (h_avg_1 : ℕ → ℕ) (h_avg_2 : ℕ) (h_avg_all : ℕ) (h_eq : 50 * x + 26 * 15 = 42 * (x + 15)) : x = 30 :=
by
  sorry

end cricket_match_count_l57_57589


namespace value_of_question_l57_57954

noncomputable def value_of_approx : ℝ := 0.2127541038062284

theorem value_of_question :
  ((0.76^3 - 0.1^3) / (0.76^2) + value_of_approx + 0.1^2) = 0.66 :=
by
  sorry

end value_of_question_l57_57954


namespace cost_of_paints_is_5_l57_57740

-- Define folders due to 6 classes
def folder_cost_per_item := 6
def num_classes := 6
def total_folder_cost : ℕ := folder_cost_per_item * num_classes

-- Define pencils due to the 6 classes and need per class
def pencil_cost_per_item := 2
def pencil_per_class := 3
def total_pencils : ℕ := pencil_per_class * num_classes
def total_pencil_cost : ℕ := pencil_cost_per_item * total_pencils

-- Define erasers needed based on pencils and their cost
def eraser_cost_per_item := 1
def pencils_per_eraser := 6
def total_erasers : ℕ := total_pencils / pencils_per_eraser
def total_eraser_cost : ℕ := eraser_cost_per_item * total_erasers

-- Total cost spent on folders, pencils, and erasers
def total_spent : ℕ := 80
def total_cost_supplies : ℕ := total_folder_cost + total_pencil_cost + total_eraser_cost

-- Cost of paints is the remaining amount when total cost is subtracted from total spent
def cost_of_paints : ℕ := total_spent - total_cost_supplies

-- The goal is to prove the cost of paints
theorem cost_of_paints_is_5 : cost_of_paints = 5 := by
  sorry

end cost_of_paints_is_5_l57_57740


namespace line_circle_separation_l57_57015

theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
    let d := 1 / (Real.sqrt (a^2 + b^2))
    d > 1 := by
    sorry

end line_circle_separation_l57_57015


namespace PersonYs_speed_in_still_water_l57_57252

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end PersonYs_speed_in_still_water_l57_57252


namespace min_value_inequality_l57_57868

open Real

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 3 * a + 2 * b + c ≥ 18 := 
sorry

end min_value_inequality_l57_57868


namespace isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l57_57216

-- Definitions for number of valence electrons
def valence_electrons (atom : String) : ℕ :=
  if atom = "C" then 4
  else if atom = "N" then 5
  else if atom = "O" then 6
  else if atom = "F" then 7
  else if atom = "S" then 6
  else 0

-- Definitions for molecular valence count
def molecule_valence_electrons (molecule : List String) : ℕ :=
  molecule.foldr (λ x acc => acc + valence_electrons x) 0

-- Definitions for specific molecules
def N2_molecule := ["N", "N"]
def CO_molecule := ["C", "O"]
def N2O_molecule := ["N", "N", "O"]
def CO2_molecule := ["C", "O", "O"]
def NO2_minus_molecule := ["N", "O", "O"]
def SO2_molecule := ["S", "O", "O"]
def O3_molecule := ["O", "O", "O"]

-- Isoelectronic property definition
def isoelectronic (mol1 mol2 : List String) : Prop :=
  molecule_valence_electrons mol1 = molecule_valence_electrons mol2

theorem isoelectronic_problem_1_part_1 :
  isoelectronic N2_molecule CO_molecule := sorry

theorem isoelectronic_problem_1_part_2 :
  isoelectronic N2O_molecule CO2_molecule := sorry

theorem isoelectronic_problem_2 :
  isoelectronic NO2_minus_molecule SO2_molecule ∧
  isoelectronic NO2_minus_molecule O3_molecule := sorry

end isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l57_57216


namespace hyperbola_condition_l57_57686

theorem hyperbola_condition (m n : ℝ) : (m < 0 ∧ 0 < n) → (∀ x y : ℝ, nx^2 + my^2 = 1 → (n * x^2 - m * y^2 > 0)) :=
by
  sorry

end hyperbola_condition_l57_57686


namespace f_le_2x_f_not_le_1_9x_l57_57751

-- Define the function f and conditions
def f : ℝ → ℝ := sorry

axiom non_neg_f : ∀ x, 0 ≤ x → 0 ≤ f x
axiom f_at_1 : f 1 = 1
axiom f_additivity : ∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2

-- Proof for part (1): f(x) ≤ 2x for all x in [0, 1]
theorem f_le_2x : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x := 
by
  sorry

-- Part (2): The inequality f(x) ≤ 1.9x does not hold for all x
theorem f_not_le_1_9x : ¬ (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 1.9 * x) := 
by
  sorry

end f_le_2x_f_not_le_1_9x_l57_57751


namespace container_volume_ratio_l57_57731

theorem container_volume_ratio
  (C D : ℕ)
  (h1 : (3 / 5 : ℚ) * C = (1 / 2 : ℚ) * D)
  (h2 : (1 / 3 : ℚ) * ((1 / 2 : ℚ) * D) + (3 / 5 : ℚ) * C = C) :
  (C : ℚ) / D = 5 / 6 :=
by {
  sorry
}

end container_volume_ratio_l57_57731


namespace emily_initial_cards_l57_57140

theorem emily_initial_cards (x : ℤ) (h1 : x + 7 = 70) : x = 63 :=
by
  sorry

end emily_initial_cards_l57_57140


namespace number_of_triples_l57_57461

theorem number_of_triples : 
  {n : ℕ // ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ n = 4} :=
sorry

end number_of_triples_l57_57461


namespace steve_book_sales_l57_57699

theorem steve_book_sales
  (copies_price : ℝ)
  (agent_rate : ℝ)
  (total_earnings : ℝ)
  (net_per_copy : ℝ := copies_price * (1 - agent_rate))
  (total_copies_sold : ℝ := total_earnings / net_per_copy) :
  copies_price = 2 → agent_rate = 0.10 → total_earnings = 1620000 → total_copies_sold = 900000 :=
by
  intros
  sorry

end steve_book_sales_l57_57699


namespace rectangle_area_correct_l57_57493

noncomputable def rectangle_area (x: ℚ) : ℚ :=
  let length := 5 * x - 18
  let width := 25 - 4 * x
  length * width

theorem rectangle_area_correct (x: ℚ) (h1: 3.6 < x) (h2: x < 6.25) :
  rectangle_area (43 / 9) = (2809 / 81) := 
  by
    sorry

end rectangle_area_correct_l57_57493


namespace moskvich_halfway_from_zhiguli_to_b_l57_57398

-- Define the Moskvich's and Zhiguli's speeds as real numbers
variables (u v : ℝ)

-- Define the given conditions as named hypotheses
axiom speed_condition : u = v
axiom halfway_condition : u = (1 / 2) * (u + v) 

-- The mathematical statement we want to prove
theorem moskvich_halfway_from_zhiguli_to_b (speed_condition : u = v) (halfway_condition : u = (1 / 2) * (u + v)) : 
  ∃ t : ℝ, t = 2 := 
sorry -- Proof omitted

end moskvich_halfway_from_zhiguli_to_b_l57_57398


namespace dice_sum_prime_probability_l57_57098

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roll_dice_prob_prime : ℚ :=
  let total_outcomes := 6^7
  let prime_sums := [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  let P := 80425 -- Assume pre-computed sum counts based on primes
  (P : ℚ) / total_outcomes

theorem dice_sum_prime_probability :
  roll_dice_prob_prime = 26875 / 93312 :=
by
  sorry

end dice_sum_prime_probability_l57_57098


namespace length_of_platform_proof_l57_57050

def convert_speed_to_mps (kmph : Float) : Float := kmph * (5/18)

def distance_covered (speed : Float) (time : Float) : Float := speed * time

def length_of_platform (total_distance : Float) (train_length : Float) : Float := total_distance - train_length

theorem length_of_platform_proof :
  let speed_kmph := 72.0
  let speed_mps := convert_speed_to_mps speed_kmph
  let time_seconds := 36.0
  let train_length := 470.06
  let total_distance := distance_covered speed_mps time_seconds
  length_of_platform total_distance train_length = 249.94 :=
by
  sorry

end length_of_platform_proof_l57_57050


namespace swimming_pool_surface_area_l57_57459

def length : ℝ := 20
def width : ℝ := 15

theorem swimming_pool_surface_area : length * width = 300 := 
by
  -- The mathematical proof would go here; we'll skip it with "sorry" per instructions.
  sorry

end swimming_pool_surface_area_l57_57459


namespace student_correct_answers_l57_57737

-- Defining the conditions as variables and equations
def correct_answers (c w : ℕ) : Prop :=
  c + w = 60 ∧ 4 * c - w = 160

-- Stating the problem: proving the number of correct answers is 44
theorem student_correct_answers (c w : ℕ) (h : correct_answers c w) : c = 44 :=
by 
  sorry

end student_correct_answers_l57_57737


namespace find_points_l57_57893

def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem find_points (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (y = x ∨ y = -x) := by
  sorry

end find_points_l57_57893


namespace least_number_leaving_remainder_4_l57_57947

theorem least_number_leaving_remainder_4 (x : ℤ) : 
  (x % 6 = 4) ∧ (x % 9 = 4) ∧ (x % 12 = 4) ∧ (x % 18 = 4) → x = 40 :=
by
  sorry

end least_number_leaving_remainder_4_l57_57947


namespace perpendicular_bisector_eq_l57_57418

theorem perpendicular_bisector_eq (A B: (ℝ × ℝ)) (hA: A = (1, 3)) (hB: B = (-5, 1)) :
  ∃ m c, (m = -3) ∧ (c = 4) ∧ (∀ x y, y = m * x + c ↔ 3 * x + y + 4 = 0) := 
by
  sorry

end perpendicular_bisector_eq_l57_57418


namespace circle_tangent_l57_57300

variables {O M : ℝ} {R : ℝ}

theorem circle_tangent
  (r : ℝ)
  (hOM_pos : O ≠ M)
  (hO : O > 0)
  (hR : R > 0)
  (h_distinct : ∀ (m n : ℝ), m ≠ n → abs (m - n) ≠ 0) :
  (r = abs (O - M) - R) ∨ (r = abs (O - M) + R) ∨ (r = R - abs (O - M)) →
  (abs ((O - M)^2 + r^2 - R^2) = 2 * R * r) :=
sorry

end circle_tangent_l57_57300


namespace circle_condition_tangent_lines_right_angle_triangle_l57_57094

-- Part (1): Range of m for the equation to represent a circle
theorem circle_condition {m : ℝ} : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*m*y + m^2 - 2*m - 2 = 0 →
  (m > -3 / 2)) :=
sorry

-- Part (2): Equation of tangent line to circle C
theorem tangent_lines {m : ℝ} (h : m = -1) : 
  ∀ x y : ℝ,
  ((x - 1)^2 + (y - 1)^2 = 1 →
  ((x = 2) ∨ (4*x - 3*y + 4 = 0))) :=
sorry

-- Part (3): Value of t for the line intersecting circle at a right angle
theorem right_angle_triangle {t : ℝ} :
  (∀ x y : ℝ, 
  (x + y + t = 0) →
  (t = -3 ∨ t = -1)) :=
sorry

end circle_condition_tangent_lines_right_angle_triangle_l57_57094


namespace solve_fraction_eq_l57_57096

theorem solve_fraction_eq :
  ∀ x : ℝ, (x - 3 ≠ 0) → ((x + 6) / (x - 3) = 4) → x = 6 :=
by
  intros x h_nonzero h_eq
  sorry

end solve_fraction_eq_l57_57096


namespace time_to_empty_is_109_89_hours_l57_57445

noncomputable def calculate_time_to_empty_due_to_leak : ℝ :=
  let R := 1 / 10 -- filling rate in tank/hour
  let Reffective := 1 / 11 -- effective filling rate in tank/hour
  let L := R - Reffective -- leak rate in tank/hour
  1 / L -- time to empty in hours

theorem time_to_empty_is_109_89_hours : calculate_time_to_empty_due_to_leak = 109.89 :=
by
  rw [calculate_time_to_empty_due_to_leak]
  sorry -- Proof steps can be filled in later

end time_to_empty_is_109_89_hours_l57_57445


namespace part_one_solution_set_part_two_range_of_m_l57_57718

noncomputable def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

/- Part I -/
theorem part_one_solution_set (x : ℝ) : 
  (f x (-1) <= 2) ↔ (0 <= x ∧ x <= 4 / 3) := 
sorry

/- Part II -/
theorem part_two_range_of_m (m : ℝ) : 
  (∀ x ∈ (Set.Icc 1 2), f x m <= |2 * x + 1|) ↔ (-3 <= m ∧ m <= 0) := 
sorry

end part_one_solution_set_part_two_range_of_m_l57_57718


namespace area_triangle_CMB_eq_105_l57_57687

noncomputable def area_of_triangle (C M B : ℝ × ℝ) : ℝ :=
  0.5 * (M.1 * B.2 - M.2 * B.1)

theorem area_triangle_CMB_eq_105 :
  let C : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (10, 0)
  let B : ℝ × ℝ := (10, 21)
  area_of_triangle C M B = 105 := by
  sorry

end area_triangle_CMB_eq_105_l57_57687


namespace find_13th_result_l57_57273

theorem find_13th_result
  (avg_25 : ℕ → ℕ)
  (avg_1_to_12 : ℕ → ℕ)
  (avg_14_to_25 : ℕ → ℕ)
  (h1 : avg_25 25 = 50)
  (h2 : avg_1_to_12 12 = 14)
  (h3 : avg_14_to_25 12 = 17) :
  ∃ (X : ℕ), X = 878 := sorry

end find_13th_result_l57_57273


namespace probability_of_different_suits_l57_57944

-- Let’s define the parameters of the problem
def total_cards : ℕ := 104
def first_card_remaining : ℕ := 103
def same_suit_cards : ℕ := 26
def different_suit_cards : ℕ := first_card_remaining - same_suit_cards

-- The probability that the two cards drawn are of different suits
def probability_different_suits : ℚ := different_suit_cards / first_card_remaining

-- The main statement to prove
theorem probability_of_different_suits :
  probability_different_suits = 78 / 103 :=
by {
  -- The proof would go here
  sorry
}

end probability_of_different_suits_l57_57944


namespace mans_rate_in_still_water_l57_57966

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h1 : V_m + V_s = 20)
  (h2 : V_m - V_s = 4) :
  V_m = 12 :=
by
  sorry

end mans_rate_in_still_water_l57_57966


namespace tina_first_hour_coins_l57_57346

variable (X : ℕ)

theorem tina_first_hour_coins :
  let first_hour_coins := X
  let second_third_hour_coins := 30 + 30
  let fourth_hour_coins := 40
  let fifth_hour_removed_coins := 20
  let total_coins := first_hour_coins + second_third_hour_coins + fourth_hour_coins - fifth_hour_removed_coins
  total_coins = 100 → X = 20 :=
by
  intro h
  sorry

end tina_first_hour_coins_l57_57346


namespace how_many_cubes_needed_l57_57534

def cube_volume (side_len : ℕ) : ℕ :=
  side_len ^ 3

theorem how_many_cubes_needed (Vsmall Vlarge Vsmall_cube num_small_cubes : ℕ) 
  (h1 : Vsmall = cube_volume 8) 
  (h2 : Vlarge = cube_volume 12) 
  (h3 : Vsmall_cube = cube_volume 2) 
  (h4 : num_small_cubes = (Vlarge - Vsmall) / Vsmall_cube) :
  num_small_cubes = 152 :=
by
  sorry

end how_many_cubes_needed_l57_57534


namespace mul_102_102_l57_57336

theorem mul_102_102 : 102 * 102 = 10404 := by
  sorry

end mul_102_102_l57_57336


namespace ny_sales_tax_l57_57728

theorem ny_sales_tax {x : ℝ} 
  (h1 : 100 + x * 1 + 6/100 * (100 + x * 1) = 110) : 
  x = 3.77 :=
by
  sorry

end ny_sales_tax_l57_57728


namespace pages_wed_calculation_l57_57510

def pages_mon : ℕ := 23
def pages_tue : ℕ := 38
def pages_thu : ℕ := 12
def pages_fri : ℕ := 2 * pages_thu
def total_pages : ℕ := 158

theorem pages_wed_calculation (pages_wed : ℕ) : 
  pages_mon + pages_tue + pages_wed + pages_thu + pages_fri = total_pages → pages_wed = 61 :=
by
  intros h
  sorry

end pages_wed_calculation_l57_57510


namespace points_collinear_l57_57611

theorem points_collinear 
  {a b c : ℝ} (h1 : 0 < b) (h2 : b < a) (h3 : c = Real.sqrt (a^2 - b^2))
  (α β : ℝ)
  (P : ℝ × ℝ) (hP : P = (a^2 / c, 0)) 
  (A : ℝ × ℝ) (hA : A = (a * Real.cos α, b * Real.sin α)) 
  (B : ℝ × ℝ) (hB : B = (a * Real.cos β, b * Real.sin β)) 
  (Q : ℝ × ℝ) (hQ : Q = (a * Real.cos α, -b * Real.sin α)) 
  (F : ℝ × ℝ) (hF : F = (c, 0))
  (line_through_F : (A.1 - F.1) * (B.2 - F.2) = (A.2 - F.2) * (B.1 - F.1)) :
  ∃ (k : ℝ), k * (Q.1 - P.1) = Q.2 - P.2 ∧ k * (B.1 - P.1) = B.2 - P.2 :=
by {
  sorry
}

end points_collinear_l57_57611


namespace find_n_l57_57732

theorem find_n (n : ℕ) : (256 : ℝ)^(1/4) = (4 : ℝ)^n → n = 1 := 
by
  sorry

end find_n_l57_57732


namespace find_x_intervals_l57_57633

theorem find_x_intervals :
  {x : ℝ | x^3 - x^2 + 11*x - 42 < 0} = { x | -2 < x ∧ x < 3 ∨ 3 < x ∧ x < 7 } :=
by sorry

end find_x_intervals_l57_57633


namespace number_of_kids_per_day_l57_57637

theorem number_of_kids_per_day (K : ℕ) 
    (kids_charge : ℕ := 3) 
    (adults_charge : ℕ := kids_charge * 2) 
    (daily_earnings_from_adults : ℕ := 10 * adults_charge) 
    (weekly_earnings : ℕ := 588) 
    (daily_earnings : ℕ := weekly_earnings / 7) :
    (daily_earnings - daily_earnings_from_adults) / kids_charge = 8 :=
by
  sorry

end number_of_kids_per_day_l57_57637


namespace problem1_l57_57649

theorem problem1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : a^2 + b^2 = 4 ∧ a * b = 1 := 
by
  sorry

end problem1_l57_57649


namespace paco_initial_cookies_l57_57846

-- Define the given conditions
def cookies_given : ℕ := 14
def cookies_eaten : ℕ := 10
def cookies_left : ℕ := 12

-- Proposition to prove: Paco initially had 36 cookies
theorem paco_initial_cookies : (cookies_given + cookies_eaten + cookies_left = 36) :=
by
  sorry

end paco_initial_cookies_l57_57846


namespace divisor_of_a_l57_57083

theorem divisor_of_a (a b : ℕ) (hx : a % x = 3) (hb : b % 6 = 5) (hab : (a * b) % 48 = 15) : x = 48 :=
by sorry

end divisor_of_a_l57_57083


namespace yellow_highlighters_l57_57901

def highlighters (pink blue yellow total : Nat) : Prop :=
  (pink + blue + yellow = total)

theorem yellow_highlighters (h : highlighters 3 5 y 15) : y = 7 :=
by 
  sorry

end yellow_highlighters_l57_57901


namespace carlos_fraction_l57_57734

theorem carlos_fraction (f : ℝ) :
  (1 - f) ^ 4 * 64 = 4 → f = 1 / 2 :=
by
  intro h
  sorry

end carlos_fraction_l57_57734


namespace combined_probability_l57_57028

-- Definitions:
def number_of_ways_to_get_3_heads_and_1_tail := Nat.choose 4 3
def probability_of_specific_sequence_of_3_heads_and_1_tail := (1/2) ^ 4
def probability_of_3_heads_and_1_tail := number_of_ways_to_get_3_heads_and_1_tail * probability_of_specific_sequence_of_3_heads_and_1_tail

def favorable_outcomes_die := 2
def total_outcomes_die := 6
def probability_of_number_greater_than_4 := favorable_outcomes_die / total_outcomes_die

-- Proof statement:
theorem combined_probability : probability_of_3_heads_and_1_tail * probability_of_number_greater_than_4 = 1/12 := by
  sorry

end combined_probability_l57_57028


namespace proof_problem_l57_57372

-- Definitions of points and vectors
def C : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (3, 4)
def N : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (0, 1)

-- Definition of vector operations
def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2)

-- Vectors needed
def AC : ℝ × ℝ := vector_sub C A
def AM : ℝ × ℝ := vector_sub M A
def AN : ℝ × ℝ := vector_sub N A

-- The Lean proof statement
theorem proof_problem :
  (∃ (x y : ℝ), AC = (x * AM.1 + y * AN.1, x * AM.2 + y * AN.2) ∧
     (x, y) = (2 / 3, 1 / 2)) ∧
  (9 * (2 / 3:ℝ) ^ 2 + 16 * (1 / 2:ℝ) ^ 2 = 8) :=
by
  sorry

end proof_problem_l57_57372


namespace quadratic_two_distinct_real_roots_l57_57290

theorem quadratic_two_distinct_real_roots (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + 2 * x1 - 3 = 0) ∧ (a * x2^2 + 2 * x2 - 3 = 0)) ↔ a > -1 / 3 := by
  sorry

end quadratic_two_distinct_real_roots_l57_57290


namespace correct_statement_A_l57_57629

-- Definitions for conditions
def general_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

def actinomycetes_dilution_range : Set ℕ := {10^3, 10^4, 10^5}

def fungi_dilution_range : Set ℕ := {10^2, 10^3, 10^4}

def first_experiment_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

-- Statement to prove
theorem correct_statement_A : 
  (general_dilution_range = {10^3, 10^4, 10^5, 10^6, 10^7}) :=
sorry

end correct_statement_A_l57_57629


namespace michael_height_l57_57262

theorem michael_height (flagpole_height flagpole_shadow michael_shadow : ℝ) 
                        (h1 : flagpole_height = 50) 
                        (h2 : flagpole_shadow = 25) 
                        (h3 : michael_shadow = 5) : 
                        (michael_shadow * (flagpole_height / flagpole_shadow) = 10) :=
by
  sorry

end michael_height_l57_57262


namespace ratio_solves_for_x_l57_57898

theorem ratio_solves_for_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
by
  -- The formal proof would go here.
  sorry

end ratio_solves_for_x_l57_57898


namespace max_tickets_jane_can_buy_l57_57320

-- Define ticket prices and Jane's budget
def ticket_price := 15
def discounted_price := 12
def discount_threshold := 5
def jane_budget := 150

-- Prove that the maximum number of tickets Jane can buy is 11
theorem max_tickets_jane_can_buy : 
  ∃ (n : ℕ), n ≤ 11 ∧ (if n ≤ discount_threshold then ticket_price * n ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (n - discount_threshold)) ≤ jane_budget)
  ∧ ∀ m : ℕ, (if m ≤ 11 then (if m ≤ discount_threshold then ticket_price * m ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (m - discount_threshold)) ≤ jane_budget) else false)  → m ≤ 11 := 
by
  sorry

end max_tickets_jane_can_buy_l57_57320


namespace fraction_zero_imp_x_eq_two_l57_57420
open Nat Real

theorem fraction_zero_imp_x_eq_two (x : ℝ) (h: (2 - abs x) / (x + 2) = 0) : x = 2 :=
by
  have h1 : 2 - abs x = 0 := sorry
  have h2 : x + 2 ≠ 0 := sorry
  sorry

end fraction_zero_imp_x_eq_two_l57_57420


namespace sum_of_cubes_divisible_by_9n_l57_57772

theorem sum_of_cubes_divisible_by_9n (n : ℕ) (h : n % 3 ≠ 0) : 
  ((n - 1)^3 + n^3 + (n + 1)^3) % (9 * n) = 0 := by
  sorry

end sum_of_cubes_divisible_by_9n_l57_57772


namespace jason_retirement_age_l57_57132

def age_at_retirement (initial_age years_to_chief extra_years_ratio years_after_masterchief : ℕ) : ℕ :=
  initial_age + years_to_chief + (years_to_chief * extra_years_ratio / 100) + years_after_masterchief

theorem jason_retirement_age :
  age_at_retirement 18 8 25 10 = 46 :=
by
  sorry

end jason_retirement_age_l57_57132


namespace smallest_n_for_common_factor_l57_57318

theorem smallest_n_for_common_factor : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (11 * n - 3) (8 * n + 4) > 1) ∧ n = 42 := 
by
  sorry

end smallest_n_for_common_factor_l57_57318


namespace kendra_words_learned_l57_57069

theorem kendra_words_learned (Goal : ℕ) (WordsNeeded : ℕ) (WordsAlreadyLearned : ℕ) 
  (h1 : Goal = 60) (h2 : WordsNeeded = 24) :
  WordsAlreadyLearned = Goal - WordsNeeded :=
sorry

end kendra_words_learned_l57_57069


namespace line_representation_l57_57106

variable {R : Type*} [Field R]
variable (f : R → R → R)
variable (x0 y0 : R)

def not_on_line (P : R × R) (f : R → R → R) : Prop :=
  f P.1 P.2 ≠ 0

theorem line_representation (P : R × R) (hP : not_on_line P f) :
  ∃ l : R → R → Prop, (∀ x y, l x y ↔ f x y - f P.1 P.2 = 0) ∧ (l P.1 P.2) ∧ 
  ∀ x y, f x y = 0 → ∃ n : R, ∀ x1 y1, (l x1 y1 → f x1 y1 = n * (f x y)) :=
sorry

end line_representation_l57_57106


namespace fraction_difference_is_correct_l57_57575

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l57_57575


namespace smallest_floor_sum_l57_57058

theorem smallest_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
sorry

end smallest_floor_sum_l57_57058


namespace laborer_monthly_income_l57_57779

theorem laborer_monthly_income
  (I : ℕ)
  (D : ℕ)
  (h1 : 6 * I + D = 510)
  (h2 : 4 * I - D = 270) : I = 78 := by
  sorry

end laborer_monthly_income_l57_57779


namespace simplify_and_evaluate_expression_l57_57700

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = (Real.sqrt 2) + 1) : 
  (1 - (1 / a)) / ((a ^ 2 - 2 * a + 1) / a) = (Real.sqrt 2) / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l57_57700


namespace jims_investment_l57_57023

theorem jims_investment (total_investment : ℝ) (john_ratio : ℝ) (james_ratio : ℝ) (jim_ratio : ℝ) 
                        (h_total_investment : total_investment = 80000)
                        (h_ratio_john : john_ratio = 4)
                        (h_ratio_james : james_ratio = 7)
                        (h_ratio_jim : jim_ratio = 9) : 
    jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 :=
by 
  sorry

end jims_investment_l57_57023


namespace people_in_first_group_l57_57937

-- Conditions
variables (P W : ℕ) (people_work_rate same_work_rate : ℕ)

-- Given conditions as Lean definitions
-- P people can do 3W in 3 days implies the work rate of the group is W per day
def first_group_work_rate : ℕ := 3 * W / 3

-- 9 people can do 9W in 3 days implies the work rate of these 9 people is 3W per day
def second_group_work_rate : ℕ := 9 * W / 3

-- The work rates are proportional to the number of people
def proportional_work_rate : Prop := P / 9 = first_group_work_rate / second_group_work_rate

-- Lean theorem statement for proof
theorem people_in_first_group (h1 : first_group_work_rate = W) (h2 : second_group_work_rate = 3 * W) :
  P = 3 :=
by
  sorry

end people_in_first_group_l57_57937


namespace simplify_expression_l57_57501

theorem simplify_expression (x : ℤ) : (3 * x) ^ 3 + (2 * x) * (x ^ 4) = 27 * x ^ 3 + 2 * x ^ 5 :=
by sorry

end simplify_expression_l57_57501


namespace inequality_has_exactly_one_solution_l57_57265

-- Definitions based on the conditions
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 3 * a

-- The main theorem that encodes the proof problem
theorem inequality_has_exactly_one_solution (a : ℝ) : 
  (∃! x : ℝ, |f x a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end inequality_has_exactly_one_solution_l57_57265


namespace n_squared_plus_d_not_square_l57_57227

theorem n_squared_plus_d_not_square 
  (n : ℕ) (d : ℕ)
  (h_pos_n : n > 0) 
  (h_pos_d : d > 0) 
  (h_div : d ∣ 2 * n^2) : 
  ¬ ∃ m : ℕ, n^2 + d = m^2 := 
sorry

end n_squared_plus_d_not_square_l57_57227


namespace triangle_angle_not_greater_than_60_l57_57926

theorem triangle_angle_not_greater_than_60 (A B C : Real) (h1 : A + B + C = 180) 
  : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
by {
  sorry
}

end triangle_angle_not_greater_than_60_l57_57926


namespace monkey_climbing_time_l57_57104

theorem monkey_climbing_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) (net_gain : ℕ) :
  tree_height = 19 →
  hop_distance = 3 →
  slip_distance = 2 →
  net_gain = hop_distance - slip_distance →
  final_hop = hop_distance →
  (tree_height - final_hop) % net_gain = 0 →
  18 / net_gain + 1 = (tree_height - final_hop) / net_gain + 1 := 
by {
  sorry
}

end monkey_climbing_time_l57_57104


namespace amazing_squares_exist_l57_57970

structure Quadrilateral :=
(A B C D : Point)

def diagonals_not_perpendicular (quad : Quadrilateral) : Prop := sorry -- The precise definition will abstractly represent the non-perpendicularity of diagonals.

def amazing_square (quad : Quadrilateral) (square : Square) : Prop :=
  -- Definition stating that the sides of the square (extended if necessary) pass through distinct vertices of the quadrilateral
  sorry

theorem amazing_squares_exist (quad : Quadrilateral) (h : diagonals_not_perpendicular quad) :
  ∃ squares : Finset Square, squares.card ≥ 6 ∧ ∀ square ∈ squares, amazing_square quad square :=
by sorry

end amazing_squares_exist_l57_57970


namespace hundred_chicken_problem_l57_57238

theorem hundred_chicken_problem :
  ∃ (x y : ℕ), x + y + 81 = 100 ∧ 5 * x + 3 * y + 81 / 3 = 100 := 
by
  sorry

end hundred_chicken_problem_l57_57238


namespace max_g_f_inequality_l57_57682

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - x / 4 - 1

theorem max_g : ∃ x : ℝ, g x = 2 * Real.log 2 - 7 / 4 :=
sorry

theorem f_inequality (x : ℝ) (hx : 0 < x) : f x < (Real.exp x - 1) / x^2 :=
sorry

end max_g_f_inequality_l57_57682


namespace solve_for_x_l57_57988

theorem solve_for_x (x : ℝ) : 3^(4 * x) = (81 : ℝ)^(1 / 4) → x = 1 / 4 :=
by
  intros
  sorry

end solve_for_x_l57_57988


namespace equilibrium_constant_l57_57414

theorem equilibrium_constant (C_NO2 C_O2 C_NO : ℝ) (h_NO2 : C_NO2 = 0.4) (h_O2 : C_O2 = 0.3) (h_NO : C_NO = 0.2) :
  (C_NO2^2 / (C_O2 * C_NO^2)) = 13.3 := by
  rw [h_NO2, h_O2, h_NO]
  sorry

end equilibrium_constant_l57_57414


namespace geometric_sequence_sum_l57_57339

theorem geometric_sequence_sum {a : ℕ → ℤ} (r : ℤ) (h1 : a 1 = 1) (h2 : r = -2) 
(h3 : ∀ n, a (n + 1) = a n * r) : 
  a 1 + |a 2| + |a 3| + a 4 = 15 := 
by sorry

end geometric_sequence_sum_l57_57339


namespace sampling_probabilities_equal_l57_57521

-- Definitions according to the problem conditions
def population_size := ℕ
def sample_size := ℕ
def simple_random_sampling (N n : ℕ) : Prop := sorry
def systematic_sampling (N n : ℕ) : Prop := sorry
def stratified_sampling (N n : ℕ) : Prop := sorry

-- Probabilities
def P1 : ℝ := sorry -- Probability for simple random sampling
def P2 : ℝ := sorry -- Probability for systematic sampling
def P3 : ℝ := sorry -- Probability for stratified sampling

-- Each definition directly corresponds to a condition in the problem statement.
-- Now, we summarize the equivalent proof problem in Lean.

theorem sampling_probabilities_equal (N n : ℕ) (h1 : simple_random_sampling N n) (h2 : systematic_sampling N n) (h3 : stratified_sampling N n) :
  P1 = P2 ∧ P2 = P3 :=
by sorry

end sampling_probabilities_equal_l57_57521


namespace brad_trips_to_fill_barrel_l57_57148

noncomputable def bucket_volume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

noncomputable def barrel_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem brad_trips_to_fill_barrel :
  let r_bucket := 8  -- radius of the hemisphere bucket in inches
  let r_barrel := 8  -- radius of the cylindrical barrel in inches
  let h_barrel := 20 -- height of the cylindrical barrel in inches
  let V_bucket := bucket_volume r_bucket
  let V_barrel := barrel_volume r_barrel h_barrel
  (Nat.ceil (V_barrel / V_bucket) = 4) :=
by
  sorry

end brad_trips_to_fill_barrel_l57_57148


namespace pyramid_volume_QEFGH_l57_57508

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * EF * FG * QE

theorem pyramid_volume_QEFGH :
  let EF := 10
  let FG := 5
  let QE := 9
  volume_of_pyramid EF FG QE = 150 := by
  sorry

end pyramid_volume_QEFGH_l57_57508


namespace soda_choosers_l57_57529

-- Definitions based on conditions
def total_people := 600
def soda_angle := 108
def full_circle := 360

-- Statement to prove the number of people who referred to soft drinks as "Soda"
theorem soda_choosers : total_people * (soda_angle / full_circle) = 180 :=
by
  sorry

end soda_choosers_l57_57529


namespace digits_of_2_120_l57_57504

theorem digits_of_2_120 (h : ∀ n : ℕ, (10 : ℝ)^(n - 1) ≤ (2 : ℝ)^200 ∧ (2 : ℝ)^200 < (10 : ℝ)^n → n = 61) :
  ∀ m : ℕ, (10 : ℝ)^(m - 1) ≤ (2 : ℝ)^120 ∧ (2 : ℝ)^120 < (10 : ℝ)^m → m = 37 :=
by
  sorry

end digits_of_2_120_l57_57504


namespace lowest_number_of_students_l57_57683

theorem lowest_number_of_students (n : ℕ) (h1 : n % 18 = 0) (h2 : n % 24 = 0) : n = 72 := by
  sorry

end lowest_number_of_students_l57_57683


namespace trigonometric_identity_l57_57945

theorem trigonometric_identity (α : Real) (h : (1 + Real.sin α) / Real.cos α = -1 / 2) :
  (Real.cos α) / (Real.sin α - 1) = 1 / 2 :=
sorry

end trigonometric_identity_l57_57945


namespace necessary_but_not_sufficient_range_m_l57_57916

namespace problem

variable (m x y : ℝ)

/-- Propositions for m -/
def P := (1 < m ∧ m < 4) 
def Q := (2 < m ∧ m < 3) ∨ (3 < m ∧  m < 4)

/-- Statements that P => Q is necessary but not sufficient -/
theorem necessary_but_not_sufficient (hP : 1 < m ∧ m < 4) : 
  ((m-1) * (m-4) < 0) ∧ (Q m) :=
by 
  sorry

theorem range_m (h1 : ¬ (P m ∧ Q m)) (h2 : P m ∨ Q m) : 
  1 < m ∧ m ≤ 2 ∨ m = 3 :=
by
  sorry

end problem

end necessary_but_not_sufficient_range_m_l57_57916


namespace parts_of_diagonal_in_rectangle_l57_57297

/-- Proving that a 24x60 rectangle divided by its diagonal results in 1512 parts --/

theorem parts_of_diagonal_in_rectangle :
  let m := 24
  let n := 60
  let gcd_mn := gcd m n
  let unit_squares := m * n
  let diagonal_intersections := m + n - gcd_mn
  unit_squares + diagonal_intersections = 1512 :=
by
  sorry

end parts_of_diagonal_in_rectangle_l57_57297


namespace festival_second_day_attendance_l57_57169

-- Define the conditions
variables (X Y Z A : ℝ)
variables (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z)

-- Theorem stating the question and the conditions result in the correct answer
theorem festival_second_day_attendance (X Y Z A : ℝ) 
  (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z) : 
  Y = 300 :=
sorry

end festival_second_day_attendance_l57_57169


namespace sum_of_solutions_l57_57658

theorem sum_of_solutions : 
  (∀ x : ℝ, (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1) → 
  (∃ s : ℝ, s = 16) :=
by
  sorry

end sum_of_solutions_l57_57658


namespace least_lcm_possible_l57_57476

theorem least_lcm_possible (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) : Nat.lcm a c = 12 :=
sorry

end least_lcm_possible_l57_57476


namespace cube_volume_from_surface_area_l57_57640

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l57_57640


namespace slope_line_point_l57_57871

theorem slope_line_point (m b : ℝ) (h_slope : m = 3) (h_point : 2 = m * 5 + b) : m + b = -10 :=
by
  sorry

end slope_line_point_l57_57871


namespace range_of_m_l57_57817

noncomputable def isEllipse (m : ℝ) : Prop := (m^2 > 2 * m + 8) ∧ (2 * m + 8 > 0)
noncomputable def intersectsXAxisAtTwoPoints (m : ℝ) : Prop := (2 * m - 3)^2 - 1 > 0

theorem range_of_m (m : ℝ) :
  ((m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∨ (2 * m - 3)^2 - 1 > 0) ∧
  ¬ (m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∧ (2 * m - 3)^2 - 1 > 0)) →
  (m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)) :=
by sorry

end range_of_m_l57_57817


namespace geometric_sequence_k_eq_6_l57_57830

theorem geometric_sequence_k_eq_6 
  (a : ℕ → ℝ) (q : ℝ) (k : ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n = a 1 * q ^ (n - 1))
  (h3 : q ≠ 1)
  (h4 : q ≠ -1)
  (h5 : a k = a 2 * a 5) :
  k = 6 :=
sorry

end geometric_sequence_k_eq_6_l57_57830


namespace range_of_m_correct_l57_57743

noncomputable def range_of_m (x : ℝ) (m : ℝ) : Prop :=
  (x + m) / (x - 2) - (2 * m) / (x - 2) = 3 ∧ x > 0 ∧ x ≠ 2

theorem range_of_m_correct (m : ℝ) : 
  (∃ x : ℝ, range_of_m x m) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_correct_l57_57743


namespace given_conditions_l57_57208

theorem given_conditions :
  ∀ (t : ℝ), t > 0 → t ≠ 1 → 
  let x := t^(2/(t-1))
  let y := t^((t+1)/(t-1))
  ¬ ((y * x^(1/y) = x * y^(1/x)) ∨ (y * x^y = x * y^x) ∨ (y^x = x^y) ∨ (x^(x+y) = y^(x+y))) :=
by
  intros t ht_pos ht_ne_1 x_def y_def
  let x := x_def
  let y := y_def
  sorry

end given_conditions_l57_57208


namespace extra_coverage_calculation_l57_57287

/-- Define the conditions -/
def bag_coverage : ℕ := 500
def lawn_length : ℕ := 35
def lawn_width : ℕ := 48
def number_of_bags : ℕ := 6

/-- Define the main theorem to prove -/
theorem extra_coverage_calculation :
  number_of_bags * bag_coverage - (lawn_length * lawn_width) = 1320 := 
by
  sorry

end extra_coverage_calculation_l57_57287


namespace sum_of_numbers_greater_than_or_equal_to_0_1_l57_57636

def num1 : ℝ := 0.8
def num2 : ℝ := 0.5  -- converting 1/2 to 0.5
def num3 : ℝ := 0.6

def is_greater_than_or_equal_to_0_1 (n : ℝ) : Prop :=
  n ≥ 0.1

theorem sum_of_numbers_greater_than_or_equal_to_0_1 :
  is_greater_than_or_equal_to_0_1 num1 ∧ 
  is_greater_than_or_equal_to_0_1 num2 ∧ 
  is_greater_than_or_equal_to_0_1 num3 →
  num1 + num2 + num3 = 1.9 :=
by
  sorry

end sum_of_numbers_greater_than_or_equal_to_0_1_l57_57636


namespace bus_passengers_total_l57_57160

theorem bus_passengers_total (children_percent : ℝ) (adults_number : ℝ) (H1 : children_percent = 0.25) (H2 : adults_number = 45) :
  ∃ T : ℝ, T = 60 :=
by
  sorry

end bus_passengers_total_l57_57160


namespace proposition_2_proposition_4_l57_57304

variable {m n : Line}
variable {α β : Plane}

-- Define predicates for perpendicularity, parallelism, and containment
axiom line_parallel_plane (n : Line) (α : Plane) : Prop
axiom line_perp_plane (n : Line) (α : Plane) : Prop
axiom plane_perp_plane (α β : Plane) : Prop
axiom line_in_plane (m : Line) (β : Plane) : Prop

-- State the correct propositions
theorem proposition_2 (m n : Line) (α β : Plane)
  (h1 : line_perp_plane m n)
  (h2 : line_perp_plane n α)
  (h3 : line_perp_plane m β) :
  plane_perp_plane α β := sorry

theorem proposition_4 (n : Line) (α β : Plane)
  (h1 : line_perp_plane n β)
  (h2 : plane_perp_plane α β) :
  line_parallel_plane n α ∨ line_in_plane n α := sorry

end proposition_2_proposition_4_l57_57304


namespace marble_selection_probability_l57_57264

theorem marble_selection_probability :
  let total_marbles := 9
  let selected_marbles := 4
  let total_ways := Nat.choose total_marbles selected_marbles
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 3
  let ways_one_red := Nat.choose red_marbles 1
  let ways_two_blue := Nat.choose blue_marbles 2
  let ways_one_green := Nat.choose green_marbles 1
  let favorable_outcomes := ways_one_red * ways_two_blue * ways_one_green
  (favorable_outcomes : ℚ) / total_ways = 3 / 14 :=
by
  sorry

end marble_selection_probability_l57_57264


namespace garden_area_remaining_l57_57986

variable (d : ℕ) (w : ℕ) (t : ℕ)

theorem garden_area_remaining (r : Real) (A_circle : Real) 
                              (A_path : Real) (A_remaining : Real) :
  r = 10 →
  A_circle = 100 * Real.pi →
  A_path = 66.66 * Real.pi - 50 * Real.sqrt 3 →
  A_remaining = 33.34 * Real.pi + 50 * Real.sqrt 3 :=
by
  -- Given the radius of the garden
  let r := (d : Real) / 2
  -- Calculate the total area of the garden
  let A_circle := Real.pi * r^2
  -- Area covered by the path computed using circular segments
  let A_path := 66.66 * Real.pi - 50 * Real.sqrt 3
  -- Remaining garden area
  let A_remaining := A_circle - A_path
  -- Statement to prove correct
  sorry 

end garden_area_remaining_l57_57986


namespace area_triangle_AEB_l57_57646

theorem area_triangle_AEB :
  ∀ (A B C D F G E : Type)
    (AB AD BC CD : ℝ) 
    (AF BG : ℝ) 
    (triangle_AEB : ℝ),
  (AB = 7) →
  (BC = 4) →
  (CD = 7) →
  (AD = 4) →
  (DF = 2) →
  (GC = 1) →
  (triangle_AEB = 1/2 * 7 * (4 + 16/3)) →
  (triangle_AEB = 98 / 3) :=
by
  intros A B C D F G E AB AD BC CD AF BG triangle_AEB
  sorry

end area_triangle_AEB_l57_57646


namespace proper_fraction_and_condition_l57_57645

theorem proper_fraction_and_condition (a b : ℤ) (h1 : 1 < a) (h2 : b = 2 * a - 1) :
  0 < a ∧ a < b ∧ (a - 1 : ℚ) / (b - 1) = 1 / 2 :=
by
  sorry

end proper_fraction_and_condition_l57_57645


namespace find_number_l57_57112

-- Define the main condition and theorem.
theorem find_number (x : ℤ) : 45 - (x - (37 - (15 - 19))) = 58 ↔ x = 28 :=
by
  sorry  -- placeholder for the proof

end find_number_l57_57112


namespace incorrect_method_D_l57_57878

-- Conditions definitions
def conditionA (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus ↔ cond p)

def conditionB (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

def conditionC (locus : Set α) (cond : α → Prop) :=
  ∀ p, (¬ (p ∈ locus) ↔ ¬ (cond p))

def conditionD (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus → cond p) ∧ (∃ p, cond p ∧ ¬ (p ∈ locus))

def conditionE (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

-- Main theorem
theorem incorrect_method_D {α : Type} (locus : Set α) (cond : α → Prop) :
  conditionD locus cond →
  ¬ (conditionA locus cond) ∧
  ¬ (conditionB locus cond) ∧
  ¬ (conditionC locus cond) ∧
  ¬ (conditionE locus cond) :=
  sorry

end incorrect_method_D_l57_57878


namespace wire_length_ratio_l57_57872

open Real

noncomputable def bonnie_wire_length : ℝ := 12 * 8
noncomputable def bonnie_cube_volume : ℝ := 8^3
noncomputable def roark_unit_cube_volume : ℝ := 2^3
noncomputable def roark_number_of_cubes : ℝ := bonnie_cube_volume / roark_unit_cube_volume
noncomputable def roark_wire_length_per_cube : ℝ := 12 * 2
noncomputable def roark_total_wire_length : ℝ := roark_number_of_cubes * roark_wire_length_per_cube
noncomputable def bonnie_to_roark_wire_ratio := bonnie_wire_length / roark_total_wire_length

theorem wire_length_ratio : bonnie_to_roark_wire_ratio = (1 : ℝ) / 16 :=
by
  sorry

end wire_length_ratio_l57_57872


namespace polynomial_expansion_sum_eq_l57_57942

theorem polynomial_expansion_sum_eq :
  (∀ (x : ℝ), (2 * x - 1)^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 243) :=
by
  sorry

end polynomial_expansion_sum_eq_l57_57942


namespace positive_integer_solution_l57_57679

theorem positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≤ y ∧ y ≤ z) (h_eq : 5 * (x * y + y * z + z * x) = 4 * x * y * z) :
  (x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20) :=
sorry

end positive_integer_solution_l57_57679


namespace smallest_positive_integer_l57_57684

-- Given integers m and n, prove the smallest positive integer of the form 2017m + 48576n
theorem smallest_positive_integer (m n : ℤ) : 
  ∃ m n : ℤ, 2017 * m + 48576 * n = 1 := by
sorry

end smallest_positive_integer_l57_57684


namespace scientific_notation_of_19672_l57_57930

theorem scientific_notation_of_19672 :
  ∃ a b, 19672 = a * 10^b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.9672 ∧ b = 4 :=
sorry

end scientific_notation_of_19672_l57_57930


namespace find_number_l57_57924

theorem find_number (x : ℤ) (h : 22 * (x - 36) = 748) : x = 70 :=
sorry

end find_number_l57_57924


namespace max_two_digit_number_divisible_by_23_l57_57268

theorem max_two_digit_number_divisible_by_23 :
  ∃ n : ℕ, 
    (n < 100) ∧ 
    (1000 ≤ n * 109) ∧ 
    (n * 109 < 10000) ∧ 
    (n % 23 = 0) ∧ 
    (n / 23 < 10) ∧ 
    (n = 69) :=
by {
  sorry
}

end max_two_digit_number_divisible_by_23_l57_57268


namespace class_committee_selection_l57_57691

theorem class_committee_selection :
  let members := ["A", "B", "C", "D", "E"]
  let admissible_entertainment_candidates := ["C", "D", "E"]
  ∃ (entertainment : String) (study : String) (sports : String),
    entertainment ∈ admissible_entertainment_candidates ∧
    study ∈ members.erase entertainment ∧
    sports ∈ (members.erase entertainment).erase study ∧
    (3 * 4 * 3 = 36) :=
sorry

end class_committee_selection_l57_57691


namespace function_increasing_l57_57742

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem function_increasing (a b c : ℝ) (h : a^2 - 3 * b < 0) : 
  ∀ x y : ℝ, x < y → f x a b c < f y a b c := sorry

end function_increasing_l57_57742


namespace mr_smith_spends_l57_57400

def buffet_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (senior_discount : ℕ) 
  (num_full_price_adults : ℕ) 
  (num_children : ℕ) 
  (num_seniors : ℕ) : ℕ :=
  num_full_price_adults * adult_price + num_children * child_price + num_seniors * (adult_price - (adult_price * senior_discount / 100))

theorem mr_smith_spends (adult_price : ℕ) (child_price : ℕ) (senior_discount : ℕ) (num_full_price_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) : 
  adult_price = 30 → 
  child_price = 15 → 
  senior_discount = 10 → 
  num_full_price_adults = 3 → 
  num_children = 3 → 
  num_seniors = 1 → 
  buffet_price adult_price child_price senior_discount num_full_price_adults num_children num_seniors = 162 :=
by 
  intros h_adult_price h_child_price h_senior_discount h_num_full_price_adults h_num_children h_num_seniors
  rw [h_adult_price, h_child_price, h_senior_discount, h_num_full_price_adults, h_num_children, h_num_seniors]
  sorry

end mr_smith_spends_l57_57400


namespace combined_degrees_l57_57174

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l57_57174


namespace largest_square_side_l57_57024

variable (length width : ℕ)
variable (h_length : length = 54)
variable (h_width : width = 20)
variable (num_squares : ℕ)
variable (h_num_squares : num_squares = 3)

theorem largest_square_side : (length : ℝ) / num_squares = 18 := by
  sorry

end largest_square_side_l57_57024


namespace inequality_solution_l57_57185

noncomputable def solve_inequality (a : ℝ) : Set ℝ :=
  if a = 0 then 
    {x : ℝ | 1 < x}
  else if 0 < a ∧ a < 2 then 
    {x : ℝ | 1 < x ∧ x < (2 / a)}
  else if a = 2 then 
    ∅
  else if a > 2 then 
    {x : ℝ | (2 / a) < x ∧ x < 1}
  else 
    {x : ℝ | x < (2 / a)} ∪ {x : ℝ | 1 < x}

theorem inequality_solution (a : ℝ) :
  ∀ x : ℝ, (ax^2 - (a + 2) * x + 2 < 0) ↔ (x ∈ solve_inequality a) :=
sorry

end inequality_solution_l57_57185


namespace right_triangle_area_l57_57064

theorem right_triangle_area (a b c : ℝ)
    (h1 : a = 16)
    (h2 : ∃ r, r = 6)
    (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a^2 + b^2 = c^2) :
    1/2 * a * b = 240 := 
by
  -- given:
  -- a = 16
  -- ∃ r, r = 6
  -- c = Real.sqrt (a^2 + b^2)
  -- a^2 + b^2 = c^2
  -- Prove: 1/2 * a * b = 240
  sorry

end right_triangle_area_l57_57064


namespace new_assistant_draw_time_l57_57581

-- Definitions based on conditions
def capacity : ℕ := 36
def halfway : ℕ := capacity / 2
def rate_top : ℕ := 1 / 6
def rate_bottom : ℕ := 1 / 4
def extra_time : ℕ := 24

-- The proof statement
theorem new_assistant_draw_time : 
  ∃ t : ℕ, ((capacity - (extra_time * rate_bottom * 1)) - halfway) = (t * rate_bottom * 1) ∧ t = 48 := by
sorry

end new_assistant_draw_time_l57_57581


namespace solution_set_inequality_l57_57671

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

theorem solution_set_inequality (a x : ℝ) (h : Set.Ioo (-1 : ℝ) (2 : ℝ) = {x | |f a x| < 6}) : 
    {x | f a x ≤ 1} = {x | x ≥ 1 / 4} :=
sorry

end solution_set_inequality_l57_57671


namespace number_of_solutions_l57_57983

theorem number_of_solutions (h₁ : ∀ x, 50 * x % 100 = 0 → (x % 2 = 0)) 
                            (h₂ : ∀ x, (x % 2 = 0) → (∀ k, 1 ≤ k ∧ k ≤ 49 → (k * x % 100 ≠ 0)))
                            (h₃ : ∀ x, 1 ≤ x ∧ x ≤ 100) : 
  ∃ count, count = 20 := 
by {
  -- Here, we usually would provide a method to count all valid x values meeting the conditions,
  -- but we skip the proof as instructed.
  sorry
}

end number_of_solutions_l57_57983


namespace a_2009_eq_1_a_2014_eq_0_l57_57219

section
variable (a : ℕ → ℕ)
variable (n : ℕ)

-- Condition 1: a_{4n-3} = 1
axiom cond1 : ∀ n : ℕ, a (4 * n - 3) = 1

-- Condition 2: a_{4n-1} = 0
axiom cond2 : ∀ n : ℕ, a (4 * n - 1) = 0

-- Condition 3: a_{2n} = a_n
axiom cond3 : ∀ n : ℕ, a (2 * n) = a n

-- Theorem: a_{2009} = 1
theorem a_2009_eq_1 : a 2009 = 1 := by
  sorry

-- Theorem: a_{2014} = 0
theorem a_2014_eq_0 : a 2014 = 0 := by
  sorry

end

end a_2009_eq_1_a_2014_eq_0_l57_57219


namespace g_range_l57_57286

variable {R : Type*} [LinearOrderedRing R]

-- Let y = f(x) be a function defined on R with a period of 1
def periodic (f : R → R) : Prop :=
  ∀ x, f (x + 1) = f x

-- If g(x) = f(x) + 2x
def g (f : R → R) (x : R) : R := f x + 2 * x

-- If the range of g(x) on the interval [1,2] is [-1,5]
def rangeCondition (f : R → R) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → -1 ≤ g f x ∧ g f x ≤ 5

-- Then the range of the function g(x) on the interval [-2020,2020] is [-4043,4041]
theorem g_range (f : R → R) 
  (hf_periodic : periodic f) 
  (hf_range : rangeCondition f) : 
  ∀ x, -2020 ≤ x ∧ x ≤ 2020 → -4043 ≤ g f x ∧ g f x ≤ 4041 :=
sorry

end g_range_l57_57286


namespace unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l57_57953

noncomputable def f (x k : ℝ) : ℝ := (Real.log x) - k * x + k

theorem unique_solution_f_geq_0 {k : ℝ} :
  (∃! x : ℝ, 0 < x ∧ f x k ≥ 0) ↔ k = 1 :=
sorry

theorem inequality_hold_for_a_leq_1 {a x : ℝ} (h₀ : a ≤ 1) :
  x * (f x 1 + x - 1) < Real.exp x - a * x^2 - 1 :=
sorry

end unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l57_57953


namespace peter_age_l57_57036

theorem peter_age (P Q : ℕ) (h1 : Q - P = P / 2) (h2 : P + Q = 35) : Q = 21 :=
  sorry

end peter_age_l57_57036


namespace total_rehabilitation_centers_l57_57572

def lisa_visits : ℕ := 6
def jude_visits (lisa : ℕ) : ℕ := lisa / 2
def han_visits (jude : ℕ) : ℕ := 2 * jude - 2
def jane_visits (han : ℕ) : ℕ := 2 * han + 6
def total_visits (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

theorem total_rehabilitation_centers :
  total_visits lisa_visits (jude_visits lisa_visits) (han_visits (jude_visits lisa_visits)) 
    (jane_visits (han_visits (jude_visits lisa_visits))) = 27 :=
by
  sorry

end total_rehabilitation_centers_l57_57572


namespace remaining_pencils_l57_57717

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end remaining_pencils_l57_57717


namespace more_radishes_correct_l57_57154

def total_radishes : ℕ := 88
def radishes_first_basket : ℕ := 37

def more_radishes_in_second_basket := total_radishes - radishes_first_basket - radishes_first_basket

theorem more_radishes_correct : more_radishes_in_second_basket = 14 :=
by
  sorry

end more_radishes_correct_l57_57154


namespace temperature_drop_change_l57_57285

theorem temperature_drop_change (T : ℝ) (h1 : T + 2 = T + 2) :
  (T - 4) - T = -4 :=
by
  sorry

end temperature_drop_change_l57_57285


namespace base_85_solution_l57_57375

theorem base_85_solution (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 16) :
  (352936524 - b) % 17 = 0 ↔ b = 4 :=
by
  sorry

end base_85_solution_l57_57375


namespace last_digit_fib_mod_12_l57_57086

noncomputable def F : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => (F n + F (n + 1)) % 12

theorem last_digit_fib_mod_12 : ∃ N, ∀ n < N, (∃ k, F k % 12 = n) ∧ ∀ m > N, F m % 12 ≠ 11 :=
sorry

end last_digit_fib_mod_12_l57_57086


namespace bus_ride_cost_l57_57904

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.85) (h2 : T + B = 9.65) : B = 1.40 :=
sorry

end bus_ride_cost_l57_57904


namespace red_gumballs_count_l57_57108

def gumballs_problem (R B G : ℕ) : Prop :=
  B = R / 2 ∧
  G = 4 * B ∧
  R + B + G = 56

theorem red_gumballs_count (R B G : ℕ) (h : gumballs_problem R B G) : R = 16 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end red_gumballs_count_l57_57108


namespace rosemary_leaves_count_l57_57794

-- Define the number of pots for each plant type
def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6

-- Define the number of leaves each plant type has
def basil_leaves : ℕ := 4
def thyme_leaves : ℕ := 30
def total_leaves : ℕ := 354

-- Prove that the number of leaves on each rosemary plant is 18
theorem rosemary_leaves_count (R : ℕ) (h : basil_pots * basil_leaves + rosemary_pots * R + thyme_pots * thyme_leaves = total_leaves) : R = 18 :=
by {
  -- Following steps are within the theorem's proof
  sorry
}

end rosemary_leaves_count_l57_57794


namespace problem_statement_l57_57176

noncomputable def two_arccos_equals_arcsin : Prop :=
  2 * Real.arccos (3 / 5) = Real.arcsin (24 / 25)

theorem problem_statement : two_arccos_equals_arcsin :=
  sorry

end problem_statement_l57_57176


namespace percentage_of_sikhs_l57_57641

theorem percentage_of_sikhs
  (total_boys : ℕ := 400)
  (percent_muslims : ℕ := 44)
  (percent_hindus : ℕ := 28)
  (other_boys : ℕ := 72) :
  ((total_boys - (percent_muslims * total_boys / 100 + percent_hindus * total_boys / 100 + other_boys)) * 100 / total_boys) = 10 :=
by
  -- proof goes here
  sorry

end percentage_of_sikhs_l57_57641


namespace combined_value_of_silver_and_gold_l57_57074

noncomputable def silver_cube_side : ℝ := 3
def silver_weight_per_cubic_inch : ℝ := 6
def silver_price_per_ounce : ℝ := 25
def gold_layer_fraction : ℝ := 0.5
def gold_weight_per_square_inch : ℝ := 0.1
def gold_price_per_ounce : ℝ := 1800
def markup_percentage : ℝ := 1.10

def calculate_combined_value (side weight_per_cubic_inch silver_price layer_fraction weight_per_square_inch gold_price markup : ℝ) : ℝ :=
  let volume := side^3
  let weight_silver := volume * weight_per_cubic_inch
  let value_silver := weight_silver * silver_price
  let surface_area := 6 * side^2
  let area_gold := surface_area * layer_fraction
  let weight_gold := area_gold * weight_per_square_inch
  let value_gold := weight_gold * gold_price
  let total_value_before_markup := value_silver + value_gold
  let selling_price := total_value_before_markup * (1 + markup)
  selling_price

theorem combined_value_of_silver_and_gold :
  calculate_combined_value silver_cube_side silver_weight_per_cubic_inch silver_price_per_ounce gold_layer_fraction gold_weight_per_square_inch gold_price_per_ounce markup_percentage = 18711 :=
by
  sorry

end combined_value_of_silver_and_gold_l57_57074


namespace at_least_one_equals_a_l57_57720

theorem at_least_one_equals_a (x y z a : ℝ) (hx_ne_0 : x ≠ 0) (hy_ne_0 : y ≠ 0) (hz_ne_0 : z ≠ 0) (ha_ne_0 : a ≠ 0)
  (h1 : x + y + z = a) (h2 : 1/x + 1/y + 1/z = 1/a) : x = a ∨ y = a ∨ z = a :=
  sorry

end at_least_one_equals_a_l57_57720


namespace manuscript_pages_count_l57_57229

theorem manuscript_pages_count
  (P : ℕ)
  (cost_first_time : ℕ := 5 * P)
  (cost_once_revised : ℕ := 4 * 30)
  (cost_twice_revised : ℕ := 8 * 20)
  (total_cost : ℕ := 780)
  (h : cost_first_time + cost_once_revised + cost_twice_revised = total_cost) :
  P = 100 :=
sorry

end manuscript_pages_count_l57_57229


namespace jimmy_eats_7_cookies_l57_57424

def cookies_and_calories (c: ℕ) : Prop :=
  50 * c + 150 = 500

theorem jimmy_eats_7_cookies : cookies_and_calories 7 :=
by {
  -- This would be where the proof steps go, but we replace it with:
  sorry
}

end jimmy_eats_7_cookies_l57_57424


namespace pandas_bamboo_consumption_l57_57043

def small_pandas : ℕ := 4
def big_pandas : ℕ := 5
def daily_bamboo_small : ℕ := 25
def daily_bamboo_big : ℕ := 40
def days_in_week : ℕ := 7

theorem pandas_bamboo_consumption : 
  (small_pandas * daily_bamboo_small + big_pandas * daily_bamboo_big) * days_in_week = 2100 := by
  sorry

end pandas_bamboo_consumption_l57_57043


namespace necessary_but_not_sufficient_condition_l57_57621

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem necessary_but_not_sufficient_condition (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f := 
sorry

end necessary_but_not_sufficient_condition_l57_57621


namespace percentage_decrease_revenue_l57_57659

theorem percentage_decrease_revenue (old_revenue new_revenue : Float) (h_old : old_revenue = 69.0) (h_new : new_revenue = 42.0) : 
  (old_revenue - new_revenue) / old_revenue * 100 = 39.13 := by
  rw [h_old, h_new]
  norm_num
  sorry

end percentage_decrease_revenue_l57_57659


namespace complement_A_eq_interval_l57_57039

open Set

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := {x : ℝ | True}

-- Define the set A according to the given conditions
def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x ≤ 0}

-- State the theorem that the complement of A with respect to U is (0, 1)
theorem complement_A_eq_interval : ∀ x : ℝ, x ∈ U \ A ↔ x ∈ Ioo 0 1 := by
  intros x
  -- Proof skipped
  sorry

end complement_A_eq_interval_l57_57039


namespace alice_expected_games_l57_57940

-- Defining the initial conditions
def skill_levels := Fin 21

def initial_active_player := 0

-- Defining Alice's skill level
def Alice_skill_level := 11

-- Define the tournament structure and conditions
def tournament_round (active: skill_levels) (inactive: Set skill_levels) : skill_levels :=
  sorry

-- Define the expected number of games Alice plays
noncomputable def expected_games_Alice_plays : ℚ :=
  sorry

-- Statement of the problem proving the expected number of games Alice plays
theorem alice_expected_games : expected_games_Alice_plays = 47 / 42 :=
sorry

end alice_expected_games_l57_57940


namespace gunther_typing_l57_57870

theorem gunther_typing :
  ∀ (wpm : ℚ), (wpm = 160 / 3) → 480 * wpm = 25598 :=
by
  intros wpm h
  sorry

end gunther_typing_l57_57870


namespace geometric_series_sum_l57_57190

theorem geometric_series_sum : 
  ∑' n : ℕ, (5 / 3) * (-1 / 3) ^ n = (5 / 4) := by
  sorry

end geometric_series_sum_l57_57190


namespace base_of_second_term_l57_57580

theorem base_of_second_term (e : ℕ) (base : ℝ) 
  (h1 : e = 35) 
  (h2 : (1/5)^e * base^18 = 1 / (2 * (10)^35)) : 
  base = 1/4 :=
by
  sorry

end base_of_second_term_l57_57580


namespace num_chickens_is_one_l57_57549

-- Define the number of dogs and the number of total legs
def num_dogs := 2
def total_legs := 10

-- Define the number of legs per dog and per chicken
def legs_per_dog := 4
def legs_per_chicken := 2

-- Define the number of chickens
def num_chickens := (total_legs - num_dogs * legs_per_dog) / legs_per_chicken

-- Prove that the number of chickens is 1
theorem num_chickens_is_one : num_chickens = 1 := by
  -- This is the proof placeholder
  sorry

end num_chickens_is_one_l57_57549


namespace find_prime_and_integer_l57_57819

theorem find_prime_and_integer (p x : ℕ) (hp : Nat.Prime p) 
  (hx1 : 1 ≤ x) (hx2 : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (p, x) = (2, 1) ∨ (p, x) = (2, 2) ∨ (p, x) = (3, 1) ∨ (p, x) = (3, 3) ∨ ((p ≥ 5) ∧ (x = 1)) :=
by
  sorry

end find_prime_and_integer_l57_57819


namespace triangle_angle_ge_60_l57_57033

theorem triangle_angle_ge_60 {A B C : ℝ} (h : A + B + C = 180) :
  A < 60 ∧ B < 60 ∧ C < 60 → false :=
by
  sorry

end triangle_angle_ge_60_l57_57033


namespace exists_square_with_only_invisible_points_l57_57955

def is_invisible (p q : ℤ) : Prop := Int.gcd p q > 1

def all_points_in_square_invisible (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 2 ∧ ∀ x y : ℕ, (x < n ∧ y < n) → is_invisible (k*x) (k*y)

theorem exists_square_with_only_invisible_points (n : ℕ) :
  all_points_in_square_invisible n := sorry

end exists_square_with_only_invisible_points_l57_57955


namespace factor_of_7_l57_57473

theorem factor_of_7 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 7 ∣ (a + 2 * b)) : 7 ∣ (100 * a + 11 * b) :=
by sorry

end factor_of_7_l57_57473


namespace sequence_a4_value_l57_57595

theorem sequence_a4_value :
  ∀ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + 3) → a 4 = 29 :=
by sorry

end sequence_a4_value_l57_57595


namespace arithmetic_sequence_common_difference_l57_57395

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 5 = 3) (h2 : a_n 6 = -2) : a_n 6 - a_n 5 = -5 :=
by
  sorry

end arithmetic_sequence_common_difference_l57_57395


namespace find_p_l57_57692

theorem find_p (m n p : ℝ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by sorry

end find_p_l57_57692


namespace hyperbola_condition_l57_57676

theorem hyperbola_condition (k : ℝ) : 
  (-1 < k ∧ k < 1) ↔ (∃ x y : ℝ, (x^2 / (k-1) + y^2 / (k+1)) = 1) := 
sorry

end hyperbola_condition_l57_57676


namespace correct_quotient_is_48_l57_57528

theorem correct_quotient_is_48 (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 72 → 
  incorrect_quotient = 24 → 
  correct_divisor = 36 →
  dividend = incorrect_divisor * incorrect_quotient →
  correct_quotient = dividend / correct_divisor →
  correct_quotient = 48 :=
by
  sorry

end correct_quotient_is_48_l57_57528


namespace probability_first_grade_probability_at_least_one_second_grade_l57_57565

-- Define conditions
def total_products : ℕ := 10
def first_grade_products : ℕ := 8
def second_grade_products : ℕ := 2
def inspected_products : ℕ := 2
def total_combinations : ℕ := Nat.choose total_products inspected_products
def first_grade_combinations : ℕ := Nat.choose first_grade_products inspected_products
def mixed_combinations : ℕ := first_grade_products * second_grade_products
def second_grade_combinations : ℕ := Nat.choose second_grade_products inspected_products

-- Define probabilities
def P_A : ℚ := first_grade_combinations / total_combinations
def P_B1 : ℚ := mixed_combinations / total_combinations
def P_B2 : ℚ := second_grade_combinations / total_combinations
def P_B : ℚ := P_B1 + P_B2

-- Statements
theorem probability_first_grade : P_A = 28 / 45 := sorry
theorem probability_at_least_one_second_grade : P_B = 17 / 45 := sorry

end probability_first_grade_probability_at_least_one_second_grade_l57_57565


namespace part1_part2_l57_57014

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem part1 (x : ℝ) : 
  (∀ (x : ℝ), f x 1 ≥ 1 → x ≤ -3 / 2) :=
sorry

theorem part2 (x t : ℝ) (h : ∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) : 
  0 < m ∧ m < 3 / 4 :=
sorry

end part1_part2_l57_57014


namespace largest_value_n_under_100000_l57_57780

theorem largest_value_n_under_100000 :
  ∃ n : ℕ,
    0 ≤ n ∧
    n < 100000 ∧
    (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 ∧
    n = 99999 :=
sorry

end largest_value_n_under_100000_l57_57780


namespace product_remainder_l57_57302

theorem product_remainder (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) (h4 : (a + b + c) % 7 = 3) : 
  (a * b * c) % 7 = 2 := 
by sorry

end product_remainder_l57_57302


namespace average_percentage_reduction_equation_l57_57130

theorem average_percentage_reduction_equation (x : ℝ) : 200 * (1 - x)^2 = 162 :=
by 
  sorry

end average_percentage_reduction_equation_l57_57130


namespace math_problem_l57_57029

variables (a b c d m : ℤ)

theorem math_problem (h1 : a = -b) (h2 : c * d = 1) (h3 : m = -1) : c * d - a - b + m^2022 = 2 :=
by
  sorry

end math_problem_l57_57029


namespace inequality_solution_l57_57386

noncomputable def condition (x : ℝ) : Prop :=
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))
  ∧ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2

theorem inequality_solution (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) (h₁ : condition x) :
  Real.cos x ≤ Real.sqrt (2:ℝ) / 2 ∧ x ∈ [Real.pi/4, 7 * Real.pi/4] := sorry

end inequality_solution_l57_57386


namespace sqrt_43_between_6_and_7_l57_57184

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 :=
by
  sorry

end sqrt_43_between_6_and_7_l57_57184


namespace sqrt_five_squared_minus_four_squared_eq_three_l57_57625

theorem sqrt_five_squared_minus_four_squared_eq_three : Real.sqrt (5 ^ 2 - 4 ^ 2) = 3 := by
  sorry

end sqrt_five_squared_minus_four_squared_eq_three_l57_57625


namespace geom_seq_sum_half_l57_57503

theorem geom_seq_sum_half (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∃ L, L = ∑' n, a n ∧ L = 1 / 2) (h_abs : |q| < 1) :
  a 0 ∈ (Set.Ioo 0 (1 / 2)) ∪ (Set.Ioo (1 / 2) 1) :=
sorry

end geom_seq_sum_half_l57_57503


namespace product_of_D_coordinates_l57_57385

theorem product_of_D_coordinates 
  (M D : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hC : C = (5, 3))
  (hM : M = (3, 7))
  (h_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
  D.1 * D.2 = 11 :=
by
  sorry

end product_of_D_coordinates_l57_57385


namespace right_triangle_area_l57_57809

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 21) (h2 : c = 15) (h3 : a^2 + b^2 = c^2):
  (1/2) * a * b = 54 :=
by
  sorry

end right_triangle_area_l57_57809


namespace CatsFavoriteNumber_l57_57191

theorem CatsFavoriteNumber :
  ∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧ 
    (∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n = p1 * p2 * p3) ∧ 
    (∀ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      n ≠ a ∧ n ≠ b ∧ n ≠ c ∧ n ≠ d ∧
      a + b - c = d ∨ b + c - d = a ∨ c + d - a = b ∨ d + a - b = c →
      (a = 30 ∧ b = 42 ∧ c = 66 ∧ d = 78)) ∧
    (n = 70) := by
  sorry

end CatsFavoriteNumber_l57_57191


namespace discount_percentage_of_sale_l57_57866

theorem discount_percentage_of_sale (initial_price sale_coupon saved_amount final_price : ℝ)
    (h1 : initial_price = 125)
    (h2 : sale_coupon = 10)
    (h3 : saved_amount = 44)
    (h4 : final_price = 81) :
    ∃ x : ℝ, x = 0.20 ∧ 
             (initial_price - initial_price * x - sale_coupon) - 
             0.10 * (initial_price - initial_price * x - sale_coupon) = final_price :=
by
  -- Proof should be constructed here
  sorry

end discount_percentage_of_sale_l57_57866


namespace last_four_digits_of_5_pow_2018_l57_57380

theorem last_four_digits_of_5_pow_2018 : 
  (5^2018) % 10000 = 5625 :=
by {
  sorry
}

end last_four_digits_of_5_pow_2018_l57_57380


namespace product_of_terms_l57_57179

variable (a : ℕ → ℝ)

-- Conditions: the sequence is geometric, a_1 = 1, a_10 = 3.
axiom geometric_sequence : ∀ n m : ℕ, a n * a m = a 1 * a (n + m - 1)

axiom a_1_eq_one : a 1 = 1
axiom a_10_eq_three : a 10 = 3

-- We need to prove that the product a_2a_3a_4a_5a_6a_7a_8a_9 = 81.
theorem product_of_terms : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end product_of_terms_l57_57179


namespace exists_linear_function_l57_57183

-- Define the properties of the function f
def is_contraction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f x - f y| ≤ |x - y|

-- Define the property of an arithmetic progression
def is_arith_seq (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ d : ℝ, ∀ n : ℕ, (f^[n] x) = x + n * d

-- Main theorem to prove
theorem exists_linear_function (f : ℝ → ℝ) (h1 : is_contraction f) (h2 : is_arith_seq f) : ∃ a : ℝ, ∀ x : ℝ, f x = x + a :=
sorry

end exists_linear_function_l57_57183


namespace slope_of_line_l57_57615

variable (s : ℝ) -- real number s

def line1 (x y : ℝ) := x + 3 * y = 9 * s + 4
def line2 (x y : ℝ) := x - 2 * y = 3 * s - 3

theorem slope_of_line (s : ℝ) :
  ∀ (x y : ℝ), (line1 s x y ∧ line2 s x y) → y = (2 / 9) * x + (13 / 9) :=
sorry

end slope_of_line_l57_57615


namespace sum_of_cubes_l57_57845

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 20) : x^3 + y^3 = 87.5 := 
by 
  sorry

end sum_of_cubes_l57_57845


namespace explicit_expression_solve_inequality_l57_57852

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := (n^2 - 3*n + 3) * x^(n+1)

theorem explicit_expression (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x)) :
  (∀ n x, f n x = x^3) :=
by
  sorry

theorem solve_inequality (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x))
  (f_eq : ∀ n x, f n x = x^3) :
  ∀ x, (x + 1)^3 + (3 - 2*x)^3 > 0 → x < 4 :=
by
  sorry

end explicit_expression_solve_inequality_l57_57852


namespace range_of_m_no_zeros_inequality_when_m_zero_l57_57858

-- Statement for Problem 1
theorem range_of_m_no_zeros (m : ℝ) (h : ∀ x : ℝ, (x^2 + m * x + m) * Real.exp x ≠ 0) : 0 < m ∧ m < 4 :=
sorry

-- Statement for Problem 2
theorem inequality_when_m_zero (x : ℝ) : 
  (x^2) * (Real.exp x) ≥ x^2 + x^3 :=
sorry

end range_of_m_no_zeros_inequality_when_m_zero_l57_57858


namespace equal_share_of_tea_l57_57295

def totalCups : ℕ := 10
def totalPeople : ℕ := 5
def cupsPerPerson : ℕ := totalCups / totalPeople

theorem equal_share_of_tea : cupsPerPerson = 2 := by
  sorry

end equal_share_of_tea_l57_57295


namespace commission_amount_l57_57550

theorem commission_amount 
  (new_avg_commission : ℤ) (increase_in_avg : ℤ) (sales_count : ℤ) 
  (total_commission_before : ℤ) (total_commission_after : ℤ) : 
  new_avg_commission = 400 → increase_in_avg = 150 → sales_count = 6 → 
  total_commission_before = (sales_count - 1) * (new_avg_commission - increase_in_avg) → 
  total_commission_after = sales_count * new_avg_commission → 
  total_commission_after - total_commission_before = 1150 :=
by 
  sorry

end commission_amount_l57_57550


namespace quadrilateral_possible_rods_l57_57307

theorem quadrilateral_possible_rods (rods : Finset ℕ) (a b c : ℕ) (ha : a = 3) (hb : b = 7) (hc : c = 15)
  (hrods : rods = (Finset.range 31 \ {3, 7, 15})) :
  ∃ d, d ∈ rods ∧ 5 < d ∧ d < 25 ∧ rods.card - 2 = 17 := 
by
  sorry

end quadrilateral_possible_rods_l57_57307


namespace fred_grew_38_cantelopes_l57_57171

def total_cantelopes : Nat := 82
def tim_cantelopes : Nat := 44
def fred_cantelopes : Nat := total_cantelopes - tim_cantelopes

theorem fred_grew_38_cantelopes : fred_cantelopes = 38 :=
by
  sorry

end fred_grew_38_cantelopes_l57_57171


namespace second_person_days_l57_57078

theorem second_person_days (x : ℕ) (h1 : ∀ y : ℝ, y = 24 → 1 / y = 1 / 24)
  (h2 : ∀ z : ℝ, z = 15 → 1 / z = 1 / 15) :
  (1 / 24 + 1 / x = 1 / 15) → x = 40 :=
by
  intro h
  have h3 : 15 * (x + 24) = 24 * x := sorry
  have h4 : 15 * x + 360 = 24 * x := sorry
  have h5 : 360 = 24 * x - 15 * x := sorry
  have h6 : 360 = 9 * x := sorry
  have h7 : x = 360 / 9 := sorry
  have h8 : x = 40 := sorry
  exact h8

end second_person_days_l57_57078


namespace domain_log_function_l57_57881

/-- The quadratic expression x^2 - 2x + 3 is always positive. -/
lemma quadratic_positive (x : ℝ) : x^2 - 2*x + 3 > 0 :=
by
  sorry

/-- The domain of the function y = log(x^2 - 2x + 3) is all real numbers. -/
theorem domain_log_function : ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*x + 3) :=
by
  have h := quadratic_positive
  sorry

end domain_log_function_l57_57881


namespace count_valid_n_l57_57800

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, 300 < n^2 ∧ n^2 < 1200 ∧ n % 3 = 0) ∧
                     S.card = 6 := sorry

end count_valid_n_l57_57800


namespace equivalence_of_complements_union_l57_57653

open Set

-- Definitions as per the conditions
def U : Set ℝ := univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def complement_U (S : Set ℝ) : Set ℝ := U \ S

-- Mathematical statement to be proved
theorem equivalence_of_complements_union :
  (complement_U M ∪ complement_U N) = { x : ℝ | x < 1 ∨ x ≥ 5 } :=
by
  -- Non-trivial proof, hence skipped with sorry
  sorry

end equivalence_of_complements_union_l57_57653


namespace determine_m_l57_57249

theorem determine_m (x m : ℝ) (h₁ : 2 * x + m = 6) (h₂ : x = 2) : m = 2 := by
  sorry

end determine_m_l57_57249


namespace problem_1_problem_2_l57_57545

-- First Problem
theorem problem_1 (f : ℝ → ℝ) (a : ℝ) (h : ∃ x : ℝ, f x - 2 * |x - 7| ≤ 0) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → a ≥ -12 :=
by
  intros
  sorry

-- Second Problem
theorem problem_2 (f : ℝ → ℝ) (a m : ℝ) (h1 : a = 1) 
  (h2 : ∀ x : ℝ, f x + |x + 7| ≥ m) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → m ≤ 7 :=
by
  intros
  sorry

end problem_1_problem_2_l57_57545


namespace bald_eagle_dive_time_l57_57303

-- Definitions as per the conditions in the problem
def speed_bald_eagle : ℝ := 100
def speed_peregrine_falcon : ℝ := 2 * speed_bald_eagle
def time_peregrine_falcon : ℝ := 15

-- The theorem to prove
theorem bald_eagle_dive_time : (speed_bald_eagle * 30) = (speed_peregrine_falcon * time_peregrine_falcon) := by
  sorry

end bald_eagle_dive_time_l57_57303


namespace solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l57_57748

variable (a x : ℝ)

theorem solve_inequality_case_a_lt_neg1 (h : a < -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

theorem solve_inequality_case_a_eq_neg1 (h : a = -1) :
  ((x - 1) * (x + a) > 0) ↔ (x ≠ 1) := sorry

theorem solve_inequality_case_a_gt_neg1 (h : a > -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

end solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l57_57748


namespace minimum_value_of_ratio_l57_57141

theorem minimum_value_of_ratio 
  {a b c : ℝ} (h_a : a ≠ 0) 
  (h_f'0 : 2 * a * 0 + b > 0)
  (h_f_nonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (∃ x : ℝ, a * x^2 + b * x + c ≥ 0) ∧ (1 + (a + c) / b = 2) := sorry

end minimum_value_of_ratio_l57_57141


namespace depth_of_channel_l57_57212

theorem depth_of_channel (top_width bottom_width : ℝ) (area : ℝ) (h : ℝ) 
  (h_top : top_width = 14) (h_bottom : bottom_width = 8) (h_area : area = 770) :
  (1 / 2) * (top_width + bottom_width) * h = area → h = 70 :=
by
  intros h_trapezoid
  sorry

end depth_of_channel_l57_57212


namespace job_completion_l57_57125

theorem job_completion (x y z : ℝ) 
  (h1 : 1/x + 1/y = 1/2) 
  (h2 : 1/y + 1/z = 1/4) 
  (h3 : 1/z + 1/x = 1/2.4) 
  (h4 : 1/x + 1/y + 1/z = 7/12) : 
  x = 3 := 
sorry

end job_completion_l57_57125


namespace range_of_a_l57_57511

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (2 * a + 1) * x + a^2 + a < 0 → 0 < 2 * x - 1 ∧ 2 * x - 1 ≤ 10) →
  (∃ l u : ℝ, (l = 1/2) ∧ (u = 9/2) ∧ (l ≤ a ∧ a ≤ u)) :=
by
  sorry

end range_of_a_l57_57511


namespace water_needed_quarts_l57_57569

-- Definitions from conditions
def ratio_water : ℕ := 8
def ratio_lemon : ℕ := 1
def total_gallons : ℚ := 1.5
def gallons_to_quarts : ℚ := 4

-- State what needs to be proven
theorem water_needed_quarts : 
  (total_gallons * gallons_to_quarts * (ratio_water / (ratio_water + ratio_lemon))) = 16 / 3 :=
by
  sorry

end water_needed_quarts_l57_57569


namespace cover_black_squares_with_L_shape_l57_57617

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the main theorem
theorem cover_black_squares_with_L_shape (n : ℕ) (h_odd : is_odd n) (h_corner_black : ∀i j, (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 1) : n ≥ 7 :=
sorry

end cover_black_squares_with_L_shape_l57_57617


namespace min_green_beads_l57_57128

theorem min_green_beads (B R G : ℕ)
  (h_total : B + R + G = 80)
  (h_red_blue : ∀ i j, B ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < R)
  (h_green_red : ∀ i j, R ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < G)
  : G = 27 := 
sorry

end min_green_beads_l57_57128


namespace NaOH_HCl_reaction_l57_57709

theorem NaOH_HCl_reaction (m : ℝ) (HCl : ℝ) (NaCl : ℝ) 
  (reaction_eq : NaOH + HCl = NaCl + H2O)
  (HCl_combined : HCl = 1)
  (NaCl_produced : NaCl = 1) :
  m = 1 := by
  sorry

end NaOH_HCl_reaction_l57_57709


namespace outerCircumference_is_correct_l57_57914

noncomputable def π : ℝ := Real.pi  
noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def width : ℝ := 4.001609997739084

noncomputable def radius_inner : ℝ := innerCircumference / (2 * π)
noncomputable def radius_outer : ℝ := radius_inner + width
noncomputable def outerCircumference : ℝ := 2 * π * radius_outer

theorem outerCircumference_is_correct : outerCircumference = 341.194 := by
  sorry

end outerCircumference_is_correct_l57_57914


namespace total_movies_shown_l57_57410

-- Define the conditions of the problem
def screens := 6
def open_hours := 8
def movie_duration := 2

-- Define the statement to prove
theorem total_movies_shown : screens * (open_hours / movie_duration) = 24 := 
by
  sorry

end total_movies_shown_l57_57410


namespace one_and_one_third_of_x_is_36_l57_57163

theorem one_and_one_third_of_x_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := 
sorry

end one_and_one_third_of_x_is_36_l57_57163


namespace cookies_per_pack_l57_57730

theorem cookies_per_pack
  (trays : ℕ) (cookies_per_tray : ℕ) (packs : ℕ)
  (h1 : trays = 8) (h2 : cookies_per_tray = 36) (h3 : packs = 12) :
  (trays * cookies_per_tray) / packs = 24 :=
by
  sorry

end cookies_per_pack_l57_57730


namespace find_c_l57_57382

   noncomputable def c_value (c : ℝ) : Prop :=
     ∃ (x y : ℝ), (x^2 - 8*x + y^2 + 10*y + c = 0) ∧ (x - 4)^2 + (y + 5)^2 = 25

   theorem find_c (c : ℝ) : c_value c → c = 16 := by
     sorry
   
end find_c_l57_57382


namespace incorrect_statements_are_1_2_4_l57_57994

theorem incorrect_statements_are_1_2_4:
    let statements := ["Inductive reasoning and analogical reasoning both involve reasoning from specific to general.",
                       "When making an analogy, it is more appropriate to use triangles in a plane and parallelepipeds in space as the objects of analogy.",
                       "'All multiples of 9 are multiples of 3, if a number m is a multiple of 9, then m must be a multiple of 3' is an example of syllogistic reasoning.",
                       "In deductive reasoning, as long as it follows the form of deductive reasoning, the conclusion is always correct."]
    let incorrect_statements := {1, 2, 4}
    incorrect_statements = {i | i ∈ [1, 2, 3, 4] ∧
                             ((i = 1 → ¬(∃ s, s ∈ statements ∧ s = statements[0])) ∧ 
                              (i = 2 → ¬(∃ s, s ∈ statements ∧ s = statements[1])) ∧ 
                              (i = 3 → ∃ s, s ∈ statements ∧ s = statements[2]) ∧ 
                              (i = 4 → ¬(∃ s, s ∈ statements ∧ s = statements[3])))} :=
by
  sorry

end incorrect_statements_are_1_2_4_l57_57994


namespace plain_b_area_l57_57453

theorem plain_b_area : 
  ∃ x : ℕ, (x + (x - 50) = 350) ∧ x = 200 :=
by
  sorry

end plain_b_area_l57_57453


namespace password_problem_l57_57283

theorem password_problem (n : ℕ) :
  (n^4 - n * (n - 1) * (n - 2) * (n - 3) = 936) → n = 6 :=
by
  sorry

end password_problem_l57_57283


namespace alexa_weight_proof_l57_57677

variable (totalWeight katerinaWeight alexaWeight : ℕ)

def weight_relation (totalWeight katerinaWeight alexaWeight : ℕ) : Prop :=
  totalWeight = katerinaWeight + alexaWeight

theorem alexa_weight_proof (h1 : totalWeight = 95) (h2 : katerinaWeight = 49) : alexaWeight = 46 :=
by
  have h : alexaWeight = totalWeight - katerinaWeight := by
    sorry
  rw [h1, h2] at h
  exact h

end alexa_weight_proof_l57_57677


namespace power_of_xy_l57_57822

-- Problem statement: Given a condition on x and y, find x^y.
theorem power_of_xy (x y : ℝ) (h : x^2 + y^2 + 4 * x - 6 * y + 13 = 0) : x^y = -8 :=
by {
  -- Proof will be added here
  sorry
}

end power_of_xy_l57_57822


namespace problem_conditions_l57_57007

theorem problem_conditions (x y : ℝ) (h : x^2 + y^2 - x * y = 1) :
  ¬ (x + y ≤ 1) ∧ (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) ∧ ¬ (x^2 + y^2 ≥ 1) :=
by
  sorry

end problem_conditions_l57_57007


namespace female_guests_from_jays_family_l57_57076

theorem female_guests_from_jays_family (total_guests : ℕ) (percent_females : ℝ) (percent_from_jays_family : ℝ)
    (h1 : total_guests = 240) (h2 : percent_females = 0.60) (h3 : percent_from_jays_family = 0.50) :
    total_guests * percent_females * percent_from_jays_family = 72 := by
  sorry

end female_guests_from_jays_family_l57_57076


namespace teta_beta_gamma_l57_57103

theorem teta_beta_gamma : 
  ∃ T E T' A B E' T'' A' G A'' M M' A''' A'''' : ℕ, 
  TETA = T * 1000 + E * 100 + T' * 10 + A ∧ 
  BETA = B * 1000 + E' * 100 + T'' * 10 + A' ∧ 
  GAMMA = G * 10000 + A'' * 1000 + M * 100 + M' * 10 + A''' ∧
  TETA + BETA = GAMMA ∧ 
  A = A'''' ∧ E = E' ∧ T = T' ∧ T' = T'' ∧ A = A' ∧ A = A'' ∧ A = A''' ∧ M = M' ∧ 
  T ≠ E ∧ T ≠ A ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧
  E ≠ A ∧ E ≠ B ∧ E ≠ G ∧ E ≠ M ∧
  A ≠ B ∧ A ≠ G ∧ A ≠ M ∧
  B ≠ G ∧ B ≠ M ∧
  G ≠ M ∧
  TETA = 4940 ∧ BETA = 5940 ∧ GAMMA = 10880
  :=
sorry

end teta_beta_gamma_l57_57103


namespace incorrect_scientific_statement_is_D_l57_57468

-- Define the number of colonies screened by Student A and other students
def studentA_colonies := 150
def other_students_colonies := 50

-- Define the descriptions
def descriptionA := "The reason Student A had such results could be due to different soil samples or problems in the experimental operation."
def descriptionB := "Student A's prepared culture medium could be cultured without adding soil as a blank control, to demonstrate whether the culture medium is contaminated."
def descriptionC := "If other students use the same soil as Student A for the experiment and get consistent results with Student A, it can be proven that Student A's operation was without error."
def descriptionD := "Both experimental approaches described in options B and C follow the principle of control in the experiment."

-- The incorrect scientific statement identified
def incorrect_statement := descriptionD

-- The main theorem statement
theorem incorrect_scientific_statement_is_D : incorrect_statement = descriptionD := by
  sorry

end incorrect_scientific_statement_is_D_l57_57468


namespace average_speed_whole_journey_l57_57244

theorem average_speed_whole_journey (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 54
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  let V_avg := total_distance / total_time
  V_avg = 64.8 :=
by
  sorry

end average_speed_whole_journey_l57_57244


namespace ratio_of_DN_NF_l57_57109

theorem ratio_of_DN_NF (D E F N : Type) (DE EF DF DN NF p q: ℕ) (h1 : DE = 18) (h2 : EF = 28) (h3 : DF = 34) 
(h4 : DN + NF = DF) (h5 : DN = 22) (h6 : NF = 11) (h7 : p = 101) (h8 : q = 50) : p + q = 151 := 
by 
  sorry

end ratio_of_DN_NF_l57_57109


namespace number_of_5_digit_numbers_l57_57799

/-- There are 324 five-digit numbers starting with 2 that have exactly three identical digits which are not 2. -/
theorem number_of_5_digit_numbers : ∃ n : ℕ, n = 324 ∧ ∀ (d₁ d₂ : ℕ), 
  (d₁ ≠ 2) ∧ (d₁ ≠ d₂) ∧ (0 ≤ d₁ ∧ d₁ < 10) ∧ (0 ≤ d₂ ∧ d₂ < 10) → 
  n = 4 * 9 * 9 := by
  sorry

end number_of_5_digit_numbers_l57_57799


namespace number_of_liars_l57_57392

theorem number_of_liars {n : ℕ} (h1 : n ≥ 1) (h2 : n ≤ 200) (h3 : ∃ k : ℕ, k < n ∧ k ≥ 1) : 
  (∃ l : ℕ, l = 199 ∨ l = 200) := 
sorry

end number_of_liars_l57_57392


namespace fourth_guard_distance_l57_57654

theorem fourth_guard_distance (d1 d2 d3 : ℕ) (d4 : ℕ) (h1 : d1 + d2 + d3 + d4 = 1000) (h2 : d1 + d2 + d3 = 850) : d4 = 150 :=
sorry

end fourth_guard_distance_l57_57654


namespace shaded_square_ratio_l57_57172

theorem shaded_square_ratio (side_length : ℝ) (H : side_length = 5) :
  let large_square_area := side_length ^ 2
  let shaded_square_area := (side_length / 2) ^ 2
  shaded_square_area / large_square_area = 1 / 4 :=
by
  sorry

end shaded_square_ratio_l57_57172


namespace determine_N_l57_57719

theorem determine_N (N : ℕ) : 995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end determine_N_l57_57719


namespace two_digit_number_digits_34_l57_57971

theorem two_digit_number_digits_34 :
  let x := (34 / 99.0)
  ∃ n : ℕ, n = 34 ∧ (48 * x - 48 * 0.34 = 0.2) := 
by
  let x := (34.0 / 99.0)
  use 34
  sorry

end two_digit_number_digits_34_l57_57971


namespace sum_of_integers_l57_57123

theorem sum_of_integers (x y : ℕ) (h1 : x = y + 3) (h2 : x^3 - y^3 = 63) : x + y = 5 :=
by
  sorry

end sum_of_integers_l57_57123


namespace sum_of_cubes_of_roots_l57_57897

theorem sum_of_cubes_of_roots (r1 r2 r3 : ℂ) (h1 : r1 + r2 + r3 = 3) (h2 : r1 * r2 + r1 * r3 + r2 * r3 = 0) (h3 : r1 * r2 * r3 = -1) : 
  r1^3 + r2^3 + r3^3 = 24 :=
  sorry

end sum_of_cubes_of_roots_l57_57897


namespace solve_quadratic_l57_57869

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) := by
  intro x
  sorry

end solve_quadratic_l57_57869


namespace cos_alpha_sub_beta_cos_alpha_l57_57708

section

variables (α β : ℝ)
variables (cos_α : ℝ) (sin_α : ℝ) (cos_β : ℝ) (sin_β : ℝ)

-- The given conditions as premises
variable (h1: cos_α = Real.cos α)
variable (h2: sin_α = Real.sin α)
variable (h3: cos_β = Real.cos β)
variable (h4: sin_β = Real.sin β)
variable (h5: 0 < α ∧ α < π / 2)
variable (h6: -π / 2 < β ∧ β < 0)
variable (h7: (cos_α - cos_β)^2 + (sin_α - sin_β)^2 = 4 / 5)

-- Part I: Prove that cos(α - β) = 3/5
theorem cos_alpha_sub_beta : Real.cos (α - β) = 3 / 5 :=
by
  sorry

-- Additional condition for Part II
variable (h8: cos_β = 12 / 13)

-- Part II: Prove that cos α = 56 / 65
theorem cos_alpha : Real.cos α = 56 / 65 :=
by
  sorry

end

end cos_alpha_sub_beta_cos_alpha_l57_57708


namespace least_possible_b_l57_57987

theorem least_possible_b (a b : ℕ) (h1 : a + b = 120) (h2 : (Prime a ∨ ∃ p : ℕ, Prime p ∧ a = 2 * p)) (h3 : Prime b) (h4 : a > b) : b = 7 :=
sorry

end least_possible_b_l57_57987


namespace parabola_focus_eq_l57_57788

/-- Given the equation of a parabola y = -4x^2 - 8x + 1, prove that its focus is at (-1, 79/16). -/
theorem parabola_focus_eq :
  ∀ x y : ℝ, y = -4 * x ^ 2 - 8 * x + 1 → 
  ∃ h k p : ℝ, y = -4 * (x + 1)^2 + 5 ∧ 
  h = -1 ∧ k = 5 ∧ p = -1 / 16 ∧ (h, k + p) = (-1, 79/16) :=
by
  sorry

end parabola_focus_eq_l57_57788


namespace mean_of_combined_sets_l57_57276

theorem mean_of_combined_sets (A : Finset ℝ) (B : Finset ℝ)
  (hA_len : A.card = 7) (hB_len : B.card = 8)
  (hA_mean : (A.sum id) / 7 = 15) (hB_mean : (B.sum id) / 8 = 22) :
  (A.sum id + B.sum id) / 15 = 18.73 :=
by sorry

end mean_of_combined_sets_l57_57276


namespace solve_equation_l57_57673

theorem solve_equation (x : ℝ) : (x + 1) * (x - 3) = 5 ↔ (x = 4 ∨ x = -2) :=
by
  sorry

end solve_equation_l57_57673


namespace max_smoothie_servings_l57_57979

def servings (bananas yogurt strawberries : ℕ) : ℕ :=
  min (bananas * 4 / 3) (min (yogurt * 4 / 2) (strawberries * 4 / 1))

theorem max_smoothie_servings :
  servings 9 10 3 = 12 :=
by
  -- Proof steps would be inserted here
  sorry

end max_smoothie_servings_l57_57979


namespace enthalpy_change_correct_l57_57678

def CC_bond_energy : ℝ := 347
def CO_bond_energy : ℝ := 358
def OH_bond_energy_CH2OH : ℝ := 463
def CO_double_bond_energy_COOH : ℝ := 745
def OH_bond_energy_COOH : ℝ := 467
def OO_double_bond_energy : ℝ := 498
def OH_bond_energy_H2O : ℝ := 467

def total_bond_energy_reactants : ℝ :=
  CC_bond_energy + CO_bond_energy + OH_bond_energy_CH2OH + 1.5 * OO_double_bond_energy

def total_bond_energy_products : ℝ :=
  CO_double_bond_energy_COOH + OH_bond_energy_COOH + OH_bond_energy_H2O

def deltaH : ℝ := total_bond_energy_reactants - total_bond_energy_products

theorem enthalpy_change_correct :
  deltaH = 236 := by
  sorry

end enthalpy_change_correct_l57_57678


namespace sum_divisors_of_24_is_60_and_not_prime_l57_57798

def divisors (n : Nat) : List Nat :=
  List.filter (λ d => n % d = 0) (List.range (n + 1))

def sum_divisors (n : Nat) : Nat :=
  (divisors n).sum

def is_prime (n : Nat) : Bool :=
  n > 1 ∧ (List.filter (λ d => d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))).length = 0

theorem sum_divisors_of_24_is_60_and_not_prime :
  sum_divisors 24 = 60 ∧ ¬ is_prime 60 := 
by
  sorry

end sum_divisors_of_24_is_60_and_not_prime_l57_57798


namespace max_value_ineq_l57_57876

theorem max_value_ineq (x y : ℝ) (h : x^2 + y^2 = 20) : xy + 8*x + y ≤ 42 := by
  sorry

end max_value_ineq_l57_57876


namespace percent_increase_l57_57681

theorem percent_increase (new_value old_value : ℕ) (h_new : new_value = 480) (h_old : old_value = 320) :
  ((new_value - old_value) / old_value) * 100 = 50 := by
  sorry

end percent_increase_l57_57681


namespace decryption_correct_l57_57457

theorem decryption_correct (a b : ℤ) (h1 : a - 2 * b = 1) (h2 : 2 * a + b = 7) : a = 3 ∧ b = 1 :=
by
  sorry

end decryption_correct_l57_57457


namespace sqrt_sum_eval_l57_57604

theorem sqrt_sum_eval : 
  (Real.sqrt 50 + Real.sqrt 72) = 11 * Real.sqrt 2 := 
by 
  sorry

end sqrt_sum_eval_l57_57604


namespace bread_baked_on_monday_l57_57704

def loaves_wednesday : ℕ := 5
def loaves_thursday : ℕ := 7
def loaves_friday : ℕ := 10
def loaves_saturday : ℕ := 14
def loaves_sunday : ℕ := 19

def increment (n m : ℕ) : ℕ := m - n

theorem bread_baked_on_monday : 
  increment loaves_wednesday loaves_thursday = 2 →
  increment loaves_thursday loaves_friday = 3 →
  increment loaves_friday loaves_saturday = 4 →
  increment loaves_saturday loaves_sunday = 5 →
  loaves_sunday + 6 = 25 :=
by 
  sorry

end bread_baked_on_monday_l57_57704


namespace center_and_radius_of_circle_l57_57739

def circle_equation := ∀ (x y : ℝ), x^2 + y^2 - 2*x - 3 = 0

theorem center_and_radius_of_circle :
  (∃ h k r : ℝ, (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x - 3 = 0) ∧ h = 1 ∧ k = 0 ∧ r = 2) :=
sorry

end center_and_radius_of_circle_l57_57739


namespace abs_diff_of_two_numbers_l57_57614

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : |x - y| = 3 :=
by
  sorry

end abs_diff_of_two_numbers_l57_57614


namespace count_congruent_3_mod_8_l57_57409

theorem count_congruent_3_mod_8 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 300) :
  ∃ k : ℕ, (1 ≤ 8 * k + 3 ∧ 8 * k + 3 ≤ 300) ∧ n = 38 :=
by
  sorry

end count_congruent_3_mod_8_l57_57409


namespace simplify_polynomial_l57_57824

theorem simplify_polynomial :
  (6 * p ^ 4 + 2 * p ^ 3 - 8 * p + 9) + (-3 * p ^ 3 + 7 * p ^ 2 - 5 * p - 1) = 
  6 * p ^ 4 - p ^ 3 + 7 * p ^ 2 - 13 * p + 8 :=
by
  sorry

end simplify_polynomial_l57_57824


namespace inequality_holds_if_and_only_if_l57_57948

noncomputable def absolute_inequality (x a : ℝ) : Prop :=
  |x - 3| + |x - 4| + |x - 5| < a

theorem inequality_holds_if_and_only_if (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, absolute_inequality x a) ↔ a > 4 := 
sorry

end inequality_holds_if_and_only_if_l57_57948


namespace card_problem_l57_57561

-- Define the variables
variables (x y : ℕ)

-- Conditions given in the problem
theorem card_problem 
  (h1 : x - 1 = y + 1) 
  (h2 : x + 1 = 2 * (y - 1)) : 
  x + y = 12 :=
sorry

end card_problem_l57_57561


namespace knights_win_35_l57_57433

noncomputable def Sharks : ℕ := sorry
noncomputable def Falcons : ℕ := sorry
noncomputable def Knights : ℕ := 35
noncomputable def Wolves : ℕ := sorry
noncomputable def Royals : ℕ := sorry

-- Conditions
axiom h1 : Sharks > Falcons
axiom h2 : Wolves > 25
axiom h3 : Wolves < Knights ∧ Knights < Royals

-- Prove: Knights won 35 games
theorem knights_win_35 : Knights = 35 := 
by sorry

end knights_win_35_l57_57433


namespace distilled_water_required_l57_57059

theorem distilled_water_required :
  ∀ (nutrient_concentrate distilled_water : ℝ) (total_solution prep_solution : ℝ), 
    nutrient_concentrate = 0.05 →
    distilled_water = 0.025 →
    total_solution = 0.075 → 
    prep_solution = 0.6 →
    (prep_solution * (distilled_water / total_solution)) = 0.2 :=
by
  intros nutrient_concentrate distilled_water total_solution prep_solution
  sorry

end distilled_water_required_l57_57059


namespace problem1_problem2_l57_57906

theorem problem1 (n : ℕ) : 2^n + 3 = k * k → n = 0 :=
by
  intros
  sorry 

theorem problem2 (n : ℕ) : 2^n + 1 = x * x → n = 3 :=
by
  intros
  sorry 

end problem1_problem2_l57_57906


namespace ordered_triples_unique_solution_l57_57020

theorem ordered_triples_unique_solution :
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ a + b + c = 2 :=
sorry

end ordered_triples_unique_solution_l57_57020


namespace find_g_at_1_l57_57466

theorem find_g_at_1 (g : ℝ → ℝ) (h : ∀ x, x ≠ 1/2 → g x + g ((2*x + 1)/(1 - 2*x)) = x) : 
  g 1 = 15 / 7 :=
sorry

end find_g_at_1_l57_57466


namespace bob_plate_price_correct_l57_57010

-- Assuming units and specific values for the problem
def anne_plate_area : ℕ := 20 -- in square units
def bob_clay_usage : ℕ := 600 -- total clay used by Bob in square units
def bob_number_of_plates : ℕ := 15
def anne_plate_price : ℕ := 50 -- in cents
def anne_number_of_plates : ℕ := 30
def total_anne_earnings : ℕ := anne_number_of_plates * anne_plate_price

-- Condition
def bob_plate_area : ℕ := bob_clay_usage / bob_number_of_plates

-- Prove the price of one of Bob's plates
theorem bob_plate_price_correct : bob_number_of_plates * bob_plate_area = bob_clay_usage →
                                  bob_number_of_plates * 100 = total_anne_earnings :=
by
  intros 
  sorry

end bob_plate_price_correct_l57_57010


namespace range_of_eccentricity_l57_57164

theorem range_of_eccentricity
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2)
  (h4 : c^2 - b^2 + a * c < 0) :
  0 < c / a ∧ c / a < 1 / 2 :=
sorry

end range_of_eccentricity_l57_57164


namespace trapezoid_angles_sum_l57_57665

theorem trapezoid_angles_sum {α β γ δ : ℝ} (h : α + β + γ + δ = 360) (h1 : α = 60) (h2 : β = 120) :
  γ + δ = 180 :=
by
  sorry

end trapezoid_angles_sum_l57_57665


namespace tire_circumference_l57_57003

/-- 
Given:
1. The tire rotates at 400 revolutions per minute.
2. The car is traveling at a speed of 168 km/h.

Prove that the circumference of the tire is 7 meters.
-/
theorem tire_circumference (rpm : ℕ) (speed_km_h : ℕ) (C : ℕ) 
  (h1 : rpm = 400) 
  (h2 : speed_km_h = 168)
  (h3 : C = 7) : 
  C = (speed_km_h * 1000 / 60) / rpm :=
by
  rw [h1, h2]
  exact h3

end tire_circumference_l57_57003


namespace dice_sum_probability_l57_57186

theorem dice_sum_probability (n : ℕ) (h : ∃ k : ℕ, (8 : ℕ) * k + k = 12) : n = 330 :=
sorry

end dice_sum_probability_l57_57186


namespace h_j_h_of_3_l57_57922

def h (x : ℤ) : ℤ := 5 * x + 2
def j (x : ℤ) : ℤ := 3 * x + 4

theorem h_j_h_of_3 : h (j (h 3)) = 277 := by
  sorry

end h_j_h_of_3_l57_57922


namespace aira_rubber_bands_l57_57162

theorem aira_rubber_bands (total_bands : ℕ) (bands_each : ℕ) (samantha_extra : ℕ) (aira_fewer : ℕ)
  (h1 : total_bands = 18) 
  (h2 : bands_each = 6) 
  (h3 : samantha_extra = 5) 
  (h4 : aira_fewer = 1) : 
  ∃ x : ℕ, x + (x + samantha_extra) + (x + aira_fewer) = total_bands ∧ x = 4 :=
by
  sorry

end aira_rubber_bands_l57_57162


namespace cubes_with_one_painted_side_l57_57202

theorem cubes_with_one_painted_side (side_length : ℕ) (one_cm_cubes : ℕ) : 
  side_length = 5 → one_cm_cubes = 54 :=
by 
  intro h 
  sorry

end cubes_with_one_painted_side_l57_57202


namespace twelve_edge_cubes_painted_faces_l57_57157

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end twelve_edge_cubes_painted_faces_l57_57157


namespace joan_video_game_spending_l57_57831

theorem joan_video_game_spending:
  let basketball_game := 5.20
  let racing_game := 4.23
  basketball_game + racing_game = 9.43 := 
by
  sorry

end joan_video_game_spending_l57_57831


namespace solve_equation_l57_57934

theorem solve_equation (x : ℝ) :
    x^6 - 22 * x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5) / 2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5) / 2) := by
  sorry

end solve_equation_l57_57934


namespace area_of_square_on_AD_l57_57347

theorem area_of_square_on_AD :
  ∃ (AB BC CD AD : ℝ),
    (∃ AB_sq BC_sq CD_sq AD_sq : ℝ,
      AB_sq = 25 ∧ BC_sq = 49 ∧ CD_sq = 64 ∧ 
      AB = Real.sqrt AB_sq ∧ BC = Real.sqrt BC_sq ∧ CD = Real.sqrt CD_sq ∧
      AD_sq = AB^2 + BC^2 + CD^2 ∧ AD = Real.sqrt AD_sq ∧ AD_sq = 138
    ) :=
by
  sorry

end area_of_square_on_AD_l57_57347


namespace plus_signs_count_l57_57582

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l57_57582


namespace solve_for_x_l57_57519

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 4 * x = 0) (h₁ : x ≠ 0) : x = 4 := 
by
  sorry

end solve_for_x_l57_57519


namespace sin_cos_sum_inequality_l57_57167

theorem sin_cos_sum_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := 
sorry

end sin_cos_sum_inequality_l57_57167


namespace product_of_values_l57_57853

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := |2 * x| + 4 = 38

-- State the theorem
theorem product_of_values : ∃ x1 x2 : ℝ, satisfies_eq x1 ∧ satisfies_eq x2 ∧ x1 * x2 = -289 := 
by
  sorry

end product_of_values_l57_57853


namespace percentage_increase_l57_57217

theorem percentage_increase (use_per_six_months : ℝ) (new_annual_use : ℝ) : 
  use_per_six_months = 90 →
  new_annual_use = 216 →
  ((new_annual_use - 2 * use_per_six_months) / (2 * use_per_six_months)) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_increase_l57_57217


namespace original_amount_of_rice_l57_57792

theorem original_amount_of_rice
  (x : ℕ) -- the total amount of rice in kilograms
  (h1 : x = 10 * 500) -- statement that needs to be proven
  (h2 : 210 = x * (21 / 50)) -- remaining rice condition after given fractions are consumed
  (consume_day_one : x - (3 / 10) * x  = (7 / 10) * x) -- after the first day's consumption
  (consume_day_two : ((7 / 10) * x) - ((2 / 5) * ((7 / 10) * x)) = 210) -- after the second day's consumption
  : x = 500 :=
by
  sorry

end original_amount_of_rice_l57_57792


namespace perfect_square_for_x_l57_57345

def expr (x : ℝ) : ℝ := 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02

theorem perfect_square_for_x : expr 0.04 = (11.98 + 0.02) ^ 2 :=
by
  sorry

end perfect_square_for_x_l57_57345


namespace find_common_difference_l57_57536

noncomputable def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 4 = 7 ∧ a 3 + a 6 = 16

theorem find_common_difference (a : ℕ → ℝ) (d : ℝ) (h : common_difference a d) : d = 2 :=
by
  sorry

end find_common_difference_l57_57536


namespace problem_B_height_l57_57088

noncomputable def point_B_height (cos : ℝ → ℝ) : ℝ :=
  let θ := 30 * (Real.pi / 180)
  let cos30 := cos θ
  let original_vertical_height := 1 / 2
  let additional_height := cos30 * (1 / 2)
  original_vertical_height + additional_height

theorem problem_B_height : 
  point_B_height Real.cos = (2 + Real.sqrt 3) / 4 := 
by 
  sorry

end problem_B_height_l57_57088


namespace difference_in_areas_l57_57155

def S1 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 2 + Real.log (x + y) / Real.log 2

def S2 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 3 + Real.log (x + y) / Real.log 2

theorem difference_in_areas : 
  let area_S1 := π * 1 ^ 2
  let area_S2 := π * (Real.sqrt 13) ^ 2
  area_S2 - area_S1 = 12 * π :=
by
  sorry

end difference_in_areas_l57_57155


namespace find_f_of_3_l57_57331

theorem find_f_of_3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x * f y - y) = x * y - f y) 
  (h2 : f 0 = 0) (h3 : ∀ x : ℝ, f (-x) = -f x) : f 3 = 3 :=
sorry

end find_f_of_3_l57_57331


namespace non_formable_triangle_sticks_l57_57651

theorem non_formable_triangle_sticks 
  (sticks : Fin 8 → ℕ) 
  (h_no_triangle : ∀ (i j k : Fin 8), i < j → j < k → sticks i + sticks j ≤ sticks k) : 
  ∃ (max_length : ℕ), (max_length = sticks (Fin.mk 7 (by norm_num))) ∧ max_length = 21 := 
by 
  sorry

end non_formable_triangle_sticks_l57_57651


namespace alpha_eq_pi_over_3_l57_57046

theorem alpha_eq_pi_over_3 (α β γ : ℝ) (h1 : 0 < α ∧ α < π) (h2 : α + β + γ = π) 
    (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
    α = π / 3 :=
by
  sorry

end alpha_eq_pi_over_3_l57_57046


namespace one_cow_one_bag_l57_57463

-- Define parameters
def cows : ℕ := 26
def bags : ℕ := 26
def days_for_all_cows : ℕ := 26

-- Theorem to prove the number of days for one cow to eat one bag of husk
theorem one_cow_one_bag (cows bags days_for_all_cows : ℕ) (h : cows = bags) (h2 : days_for_all_cows = 26) : days_for_one_cow_one_bag = 26 :=
by {
    sorry -- Proof to be filled in
}

end one_cow_one_bag_l57_57463


namespace circus_total_tickets_sold_l57_57639

-- Definitions from the conditions
def revenue_total : ℕ := 2100
def lower_seat_tickets_sold : ℕ := 50
def price_lower : ℕ := 30
def price_upper : ℕ := 20

-- Definition derived from the conditions
def tickets_total (L U : ℕ) : ℕ := L + U

-- The theorem we need to prove
theorem circus_total_tickets_sold (L U : ℕ) (hL: L = lower_seat_tickets_sold)
    (h₁ : price_lower * L + price_upper * U = revenue_total) : 
    tickets_total L U = 80 :=
by
  sorry  -- Proof omitted

end circus_total_tickets_sold_l57_57639


namespace B_finishes_work_in_54_days_l57_57061

-- The problem statement rewritten in Lean 4.
theorem B_finishes_work_in_54_days
  (A_eff : ℕ) -- amount of work A can do in one day
  (B_eff : ℕ) -- amount of work B can do in one day
  (work_days_together : ℕ) -- number of days A and B work together to finish the work
  (h1 : A_eff = 2 * B_eff)
  (h2 : A_eff + B_eff = 3)
  (h3 : work_days_together = 18) :
  work_days_together * (A_eff + B_eff) / B_eff = 54 :=
by
  sorry

end B_finishes_work_in_54_days_l57_57061


namespace polyhedron_calculation_l57_57490

def faces := 32
def triangular := 10
def pentagonal := 8
def hexagonal := 14
def edges := 79
def vertices := 49
def T := 1
def P := 2

theorem polyhedron_calculation : 
  100 * P + 10 * T + vertices = 249 := 
sorry

end polyhedron_calculation_l57_57490


namespace a4_equals_zero_l57_57990

-- Define the general term of the sequence
def a (n : ℕ) (h : n > 0) : ℤ := n^2 - 3 * n - 4

-- The theorem statement to prove a_4 = 0
theorem a4_equals_zero : a 4 (by norm_num) = 0 :=
sorry

end a4_equals_zero_l57_57990


namespace problem_statement_l57_57448

variable (x : ℝ)

-- Definitions based on the conditions
def a := 2005 * x + 2009
def b := 2005 * x + 2010
def c := 2005 * x + 2011

-- Assertion for the problem
theorem problem_statement : a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = 3 := by
  sorry

end problem_statement_l57_57448


namespace cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l57_57454

theorem cos_alpha_plus_5pi_over_12_eq_neg_1_over_3
  (α : ℝ)
  (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l57_57454


namespace caleb_counted_right_angles_l57_57715

-- Definitions for conditions
def rectangular_park_angles : ℕ := 4
def square_field_angles : ℕ := 4
def total_angles (x y : ℕ) : ℕ := x + y

-- Theorem stating the problem
theorem caleb_counted_right_angles (h : total_angles rectangular_park_angles square_field_angles = 8) : 
   "type of anges Caleb counted" = "right angles" :=
sorry

end caleb_counted_right_angles_l57_57715


namespace range_of_a_l57_57194

noncomputable def A : Set ℝ := {x | -2 ≤ x ∧ x < 4 }

noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (a : ℝ) : (B a ⊆ A) ↔ (0 ≤ a ∧ a < 3) := sorry

end range_of_a_l57_57194


namespace least_area_of_prime_dim_l57_57471

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_area_of_prime_dim (l w : ℕ) (h_perimeter : 2 * (l + w) = 120)
    (h_integer_dims : l > 0 ∧ w > 0) (h_prime_dim : is_prime l ∨ is_prime w) :
    l * w = 116 :=
sorry

end least_area_of_prime_dim_l57_57471


namespace seating_arrangement_l57_57752

theorem seating_arrangement (x y : ℕ) (h1 : x * 8 + y * 7 = 55) : x = 6 :=
by
  sorry

end seating_arrangement_l57_57752


namespace orthogonal_lines_solution_l57_57578

theorem orthogonal_lines_solution (a b c d : ℝ)
  (h1 : b - a = 0)
  (h2 : c - a = 2)
  (h3 : 12 * d - a = 1)
  : d = 3 / 11 :=
by {
  sorry
}

end orthogonal_lines_solution_l57_57578


namespace gecko_sales_ratio_l57_57980

theorem gecko_sales_ratio (x : ℕ) (h1 : 86 + x = 258) : 86 / Nat.gcd 172 86 = 1 ∧ 172 / Nat.gcd 172 86 = 2 := by
  sorry

end gecko_sales_ratio_l57_57980


namespace auction_site_TVs_correct_l57_57502

-- Define the number of TVs Beatrice looked at in person
def in_person_TVs : Nat := 8

-- Define the number of TVs Beatrice looked at online
def online_TVs : Nat := 3 * in_person_TVs

-- Define the total number of TVs Beatrice looked at
def total_TVs : Nat := 42

-- Define the number of TVs Beatrice looked at on the auction site
def auction_site_TVs : Nat := total_TVs - (in_person_TVs + online_TVs)

-- Prove that the number of TVs Beatrice looked at on the auction site is 10
theorem auction_site_TVs_correct : auction_site_TVs = 10 :=
by
  sorry

end auction_site_TVs_correct_l57_57502


namespace jared_popcorn_l57_57011

-- Define the given conditions
def pieces_per_serving := 30
def number_of_friends := 3
def pieces_per_friend := 60
def servings_ordered := 9

-- Define the total pieces of popcorn
def total_pieces := servings_ordered * pieces_per_serving

-- Define the total pieces of popcorn eaten by Jared's friends
def friends_total_pieces := number_of_friends * pieces_per_friend

-- State the theorem
theorem jared_popcorn : total_pieces - friends_total_pieces = 90 :=
by 
  -- The detailed proof would go here.
  sorry

end jared_popcorn_l57_57011


namespace M_squared_is_odd_l57_57744

theorem M_squared_is_odd (a b : ℤ) (h1 : a = b + 1) (c : ℤ) (h2 : c = a * b) (M : ℤ) (h3 : M^2 = a^2 + b^2 + c^2) : M^2 % 2 = 1 := 
by
  sorry

end M_squared_is_odd_l57_57744


namespace total_time_to_complete_project_l57_57196

-- Define the initial conditions
def initial_people : ℕ := 6
def initial_days : ℕ := 35
def fraction_completed : ℚ := 1 / 3

-- Define the additional conditions after more people joined
def additional_people : ℕ := initial_people
def total_people : ℕ := initial_people + additional_people
def remaining_fraction : ℚ := 1 - fraction_completed

-- Total time taken to complete the project
theorem total_time_to_complete_project (initial_people initial_days additional_people : ℕ) (fraction_completed remaining_fraction : ℚ)
  (h1 : initial_people * initial_days * fraction_completed = 1/3) 
  (h2 : additional_people = initial_people) 
  (h3 : total_people = initial_people + additional_people)
  (h4 : remaining_fraction = 1 - fraction_completed) : 
  (initial_days + (remaining_fraction / (total_people * (fraction_completed / (initial_people * initial_days)))) = 70) :=
sorry

end total_time_to_complete_project_l57_57196


namespace range_of_a_l57_57941

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a > 0

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a) (h2 : q a) : a ≤ -2 :=
by
  sorry

end range_of_a_l57_57941


namespace map_length_representation_l57_57920

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l57_57920


namespace log_inequality_l57_57266

theorem log_inequality (x y : ℝ) :
  let log2 := Real.log 2
  let log5 := Real.log 5
  let log3 := Real.log 3
  let log2_3 := log3 / log2
  let log5_3 := log3 / log5
  (log2_3 ^ x - log5_3 ^ x ≥ log2_3 ^ (-y) - log5_3 ^ (-y)) → (x + y ≥ 0) :=
by
  intros h
  sorry

end log_inequality_l57_57266


namespace combined_return_percentage_l57_57341

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return1_percent return2_percent : ℝ) (total_investment total_return : ℝ) :
  investment1 = 500 → 
  return1_percent = 0.07 → 
  investment2 = 1500 → 
  return2_percent = 0.09 → 
  total_investment = investment1 + investment2 → 
  total_return = investment1 * return1_percent + investment2 * return2_percent → 
  (total_return / total_investment) * 100 = 8.5 :=
by 
  sorry

end combined_return_percentage_l57_57341


namespace minimum_value_of_reciprocal_product_l57_57939

theorem minimum_value_of_reciprocal_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + a * b + 2 * b = 30) : 
  ∃ m : ℝ, m = 1 / (a * b) ∧ m = 1 / 18 :=
sorry

end minimum_value_of_reciprocal_product_l57_57939


namespace problem_lean_statement_l57_57294

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6)

theorem problem_lean_statement : 
  (∀ x, g x = 2 * cos (2 * x)) ∧ (∀ x, g (x) = g (-x)) ∧ (∀ x, g (x + π) = g (x)) :=
  sorry

end problem_lean_statement_l57_57294


namespace number_of_people_in_group_l57_57071

/-- The number of people in the group N is such that when one of the people weighing 65 kg is replaced
by a new person weighing 100 kg, the average weight of the group increases by 3.5 kg. -/
theorem number_of_people_in_group (N : ℕ) (W : ℝ) 
  (h1 : (W + 35) / N = W / N + 3.5) 
  (h2 : W + 35 = W - 65 + 100) : 
  N = 10 :=
sorry

end number_of_people_in_group_l57_57071


namespace solution_set_empty_l57_57257

variable (m x : ℝ)
axiom no_solution (h : (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) : (1 + m = 0)

theorem solution_set_empty :
  (∀ x, (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end solution_set_empty_l57_57257


namespace proof_of_intersection_l57_57122

open Set

theorem proof_of_intersection :
  let U := ℝ
  let M := compl { x : ℝ | x^2 > 4 }
  let N := { x : ℝ | 1 < x ∧ x ≤ 3 }
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := by
sorry

end proof_of_intersection_l57_57122


namespace find_a6_l57_57834

variable (a : ℕ → ℝ)

-- condition: a_2 + a_8 = 16
axiom h1 : a 2 + a 8 = 16

-- condition: a_4 = 1
axiom h2 : a 4 = 1

-- question: Prove that a_6 = 15
theorem find_a6 : a 6 = 15 :=
sorry

end find_a6_l57_57834


namespace mice_meet_after_three_days_l57_57918

theorem mice_meet_after_three_days 
  (thickness : ℕ) 
  (first_day_distance : ℕ) 
  (big_mouse_double_progress : ℕ → ℕ) 
  (small_mouse_half_remain_distance : ℕ → ℕ) 
  (days : ℕ) 
  (big_mouse_distance : ℚ) : 
  thickness = 5 ∧ 
  first_day_distance = 1 ∧ 
  (∀ n, big_mouse_double_progress n = 2 ^ (n - 1)) ∧ 
  (∀ n, small_mouse_half_remain_distance n = 5 - (5 / 2 ^ (n - 1))) ∧ 
  days = 3 → 
  big_mouse_distance = 3 + 8 / 17 := 
by
  sorry

end mice_meet_after_three_days_l57_57918


namespace percentage_difference_l57_57555

theorem percentage_difference :
  (0.50 * 56 - 0.30 * 50) = 13 := 
by
  -- sorry is used to skip the actual proof steps
  sorry 

end percentage_difference_l57_57555


namespace find_principal_amount_l57_57118

theorem find_principal_amount
  (r : ℝ := 0.05)  -- Interest rate (5% per annum)
  (t : ℕ := 2)    -- Time period (2 years)
  (diff : ℝ := 20) -- Given difference between CI and SI
  (P : ℝ := 8000) -- Principal amount to prove
  : P * (1 + r) ^ t - P - P * r * t = diff :=
by
  sorry

end find_principal_amount_l57_57118


namespace sign_of_c_l57_57280

/-
Define the context and conditions as Lean axioms.
-/

variables (a b c : ℝ)

-- Axiom: The sum of coefficients is less than zero
axiom h1 : a + b + c < 0

-- Axiom: The quadratic equation has no real roots, thus the discriminant is less than zero
axiom h2 : (b^2 - 4*a*c) < 0

/-
Formal statement of the proof problem:
-/

theorem sign_of_c : c < 0 :=
by
  -- We state that the proof of c < 0 follows from the given axioms
  sorry

end sign_of_c_l57_57280


namespace range_of_a_l57_57481

theorem range_of_a (a : ℝ) (h : ∅ ⊂ {x : ℝ | x^2 ≤ a}) : 0 ≤ a :=
by
  sorry

end range_of_a_l57_57481


namespace pies_sold_in_week_l57_57857

def daily_pies : ℕ := 8
def days_in_week : ℕ := 7

theorem pies_sold_in_week : daily_pies * days_in_week = 56 := by
  sorry

end pies_sold_in_week_l57_57857


namespace cupcakes_sold_l57_57312

theorem cupcakes_sold (initial_made sold additional final : ℕ) (h1 : initial_made = 42) (h2 : additional = 39) (h3 : final = 59) :
  (initial_made - sold + additional = final) -> sold = 22 :=
by
  intro h
  rw [h1, h2, h3] at h
  sorry

end cupcakes_sold_l57_57312


namespace area_of_walkways_is_214_l57_57082

-- Definitions for conditions
def width_of_flower_beds : ℕ := 2 * 7  -- two beds each 7 feet wide
def walkways_between_beds_width : ℕ := 3 * 2  -- three walkways each 2 feet wide (one on each side and one in between)
def total_width : ℕ := width_of_flower_beds + walkways_between_beds_width  -- Total width

def height_of_flower_beds : ℕ := 3 * 3  -- three rows of beds each 3 feet high
def walkways_between_beds_height : ℕ := 4 * 2  -- four walkways each 2 feet wide (one on each end and one between each row)
def total_height : ℕ := height_of_flower_beds + walkways_between_beds_height  -- Total height

def total_area_of_garden : ℕ := total_width * total_height  -- Total area of the garden including walkways

def area_of_one_flower_bed : ℕ := 7 * 3  -- Area of one flower bed
def total_area_of_flower_beds : ℕ := 6 * area_of_one_flower_bed  -- Total area of six flower beds

def total_area_walkways : ℕ := total_area_of_garden - total_area_of_flower_beds  -- Total area of the walkways

-- Theorem to prove the area of the walkways
theorem area_of_walkways_is_214 : total_area_walkways = 214 := sorry

end area_of_walkways_is_214_l57_57082


namespace find_N_sum_e_l57_57884

theorem find_N_sum_e (N : ℝ) (e1 e2 : ℝ) :
  (2 * abs (2 - e1) = N) ∧
  (2 * abs (2 - e2) = N) ∧
  (e1 ≠ e2) ∧
  (e1 + e2 = 4) →
  N = 0 :=
by
  sorry

end find_N_sum_e_l57_57884


namespace solution_set_of_inequality_l57_57515

theorem solution_set_of_inequality:
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | (-7 < x ∧ x ≤ -1) ∨ (5 ≤ x ∧ x < 11)} :=
by
  sorry

end solution_set_of_inequality_l57_57515


namespace quadratic_inequality_m_range_l57_57480

theorem quadratic_inequality_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ (m ≠ 0) :=
by
  sorry

end quadratic_inequality_m_range_l57_57480


namespace integer_values_in_interval_l57_57815

theorem integer_values_in_interval : (∃ n : ℕ, n = 25 ∧ ∀ x : ℤ, abs x < 4 * π ↔ -12 ≤ x ∧ x ≤ 12) :=
by
  sorry

end integer_values_in_interval_l57_57815


namespace walt_total_interest_l57_57689

noncomputable def total_investment : ℝ := 12000
noncomputable def investment_at_7_percent : ℝ := 5500
noncomputable def investment_at_9_percent : ℝ := total_investment - investment_at_7_percent
noncomputable def rate_7_percent : ℝ := 0.07
noncomputable def rate_9_percent : ℝ := 0.09

theorem walt_total_interest :
  let interest_7 : ℝ := investment_at_7_percent * rate_7_percent
  let interest_9 : ℝ := investment_at_9_percent * rate_9_percent
  interest_7 + interest_9 = 970 := by
  sorry

end walt_total_interest_l57_57689


namespace inequality_proof_l57_57342

theorem inequality_proof (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) : 
  2 * Real.sin α + Real.tan α > 3 * α := 
by
  sorry

end inequality_proof_l57_57342


namespace train_speed_proof_l57_57570

def identical_trains_speed : Real :=
  11.11

theorem train_speed_proof :
  ∀ (v : ℝ),
  (∀ (t t' : ℝ), 
  (t = 150 / v) ∧ 
  (t' = 300 / v) ∧ 
  ((t' + 100 / v) = 36)) → v = identical_trains_speed :=
by
  sorry

end train_speed_proof_l57_57570


namespace great_white_shark_teeth_is_420_l57_57352

-- Define the number of teeth in a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Define the number of teeth in a hammerhead shark based on the tiger shark's teeth
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Define the number of teeth in a great white shark based on the sum of tiger and hammerhead shark's teeth
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- The theorem statement that we need to prove
theorem great_white_shark_teeth_is_420 : great_white_shark_teeth = 420 :=
by
  -- Provide space for the proof
  sorry

end great_white_shark_teeth_is_420_l57_57352


namespace total_letters_correct_l57_57995

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l57_57995


namespace number_of_matches_in_first_set_l57_57804

theorem number_of_matches_in_first_set
  (x : ℕ)
  (h1 : (30 : ℚ) * x + 15 * 10 = 25 * (x + 10)) :
  x = 20 :=
by
  -- The proof will be filled in here
  sorry

end number_of_matches_in_first_set_l57_57804


namespace percentage_decrease_stock_l57_57000

theorem percentage_decrease_stock (F J M : ℝ)
  (h1 : J = F - 0.10 * F)
  (h2 : M = J - 0.20 * J) :
  (F - M) / F * 100 = 28 := by
sorry

end percentage_decrease_stock_l57_57000


namespace proposition_q_false_for_a_lt_2_l57_57613

theorem proposition_q_false_for_a_lt_2 (a : ℝ) (h : a < 2) : 
  ¬ ∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1 :=
sorry

end proposition_q_false_for_a_lt_2_l57_57613


namespace divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l57_57556

theorem divides_2_pow_n_sub_1 (n : ℕ) : 7 ∣ (2 ^ n - 1) ↔ 3 ∣ n := by
  sorry

theorem no_n_divides_2_pow_n_add_1 (n : ℕ) : ¬ 7 ∣ (2 ^ n + 1) := by
  sorry

end divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l57_57556


namespace time_for_a_to_complete_one_round_l57_57189

theorem time_for_a_to_complete_one_round (T_a T_b : ℝ) 
  (h1 : 4 * T_a = 3 * T_b)
  (h2 : T_b = T_a + 10) : 
  T_a = 30 := by
  sorry

end time_for_a_to_complete_one_round_l57_57189


namespace sugar_amount_l57_57520

theorem sugar_amount (S F B : ℝ) 
    (h_ratio1 : S = F) 
    (h_ratio2 : F = 10 * B) 
    (h_ratio3 : F / (B + 60) = 8) : S = 2400 := 
by
  sorry

end sugar_amount_l57_57520


namespace range_of_m_l57_57602

theorem range_of_m {x1 x2 y1 y2 m : ℝ} 
  (h1 : x1 > x2) 
  (h2 : y1 > y2) 
  (ha : y1 = (m - 3) * x1 - 4) 
  (hb : y2 = (m - 3) * x2 - 4) : 
  m > 3 :=
sorry

end range_of_m_l57_57602


namespace cows_total_l57_57374

theorem cows_total (A M R : ℕ) (h1 : A = 4 * M) (h2 : M = 60) (h3 : A + M = R + 30) : 
  A + M + R = 570 := by
  sorry

end cows_total_l57_57374


namespace average_student_headcount_proof_l57_57587

def average_student_headcount : ℕ := (11600 + 11800 + 12000 + 11400) / 4

theorem average_student_headcount_proof :
  average_student_headcount = 11700 :=
by
  -- calculation here
  sorry

end average_student_headcount_proof_l57_57587


namespace compound_interest_time_period_l57_57552

theorem compound_interest_time_period (P r I : ℝ) (n A t : ℝ) 
(hP : P = 6000) 
(hr : r = 0.10) 
(hI : I = 1260.000000000001) 
(hn : n = 1)
(hA : A = P + I)
(ht_eqn: (A / P) = (1 + r / n) ^ t) :
t = 2 := 
by sorry

end compound_interest_time_period_l57_57552


namespace minimum_value_ineq_l57_57787

theorem minimum_value_ineq (x : ℝ) (hx : 0 < x) :
  3 * Real.sqrt x + 4 / x ≥ 4 * Real.sqrt 2 :=
by
  sorry

end minimum_value_ineq_l57_57787


namespace square_side_length_l57_57228

theorem square_side_length (s : ℝ) (h : s^2 + s - 4 * s = 4) : s = 4 :=
sorry

end square_side_length_l57_57228


namespace polar_to_rectangular_4sqrt2_pi_over_4_l57_57383

theorem polar_to_rectangular_4sqrt2_pi_over_4 :
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (4, 4) :=
by
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  sorry

end polar_to_rectangular_4sqrt2_pi_over_4_l57_57383


namespace perimeter_of_first_square_l57_57075

theorem perimeter_of_first_square
  (s1 s2 s3 : ℝ)
  (P1 P2 P3 : ℝ)
  (A1 A2 A3 : ℝ)
  (hs2 : s2 = 8)
  (hs3 : s3 = 10)
  (hP2 : P2 = 4 * s2)
  (hP3 : P3 = 4 * s3)
  (hP2_val : P2 = 32)
  (hP3_val : P3 = 40)
  (hA2 : A2 = s2^2)
  (hA3 : A3 = s3^2)
  (hA1_A2_A3 : A3 = A1 + A2)
  (hA3_val : A3 = 100)
  (hA2_val : A2 = 64) :
  P1 = 24 := by
  sorry

end perimeter_of_first_square_l57_57075


namespace largest_lcm_l57_57377

theorem largest_lcm :
  max (max (max (max (Nat.lcm 18 4) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 14)) (Nat.lcm 18 18) = 126 :=
by
  sorry

end largest_lcm_l57_57377


namespace simplify_expression_l57_57370

variable (x y : ℝ)

theorem simplify_expression :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 :=
by
  sorry

end simplify_expression_l57_57370


namespace unique_prime_satisfying_condition_l57_57608

theorem unique_prime_satisfying_condition :
  ∃! p : ℕ, Prime p ∧ (∀ q : ℕ, Prime q ∧ q < p → ∀ k r : ℕ, p = k * q + r ∧ 0 ≤ r ∧ r < q → ∀ a : ℕ, a > 1 → ¬ a^2 ∣ r) ∧ p = 13 :=
sorry

end unique_prime_satisfying_condition_l57_57608


namespace combined_weight_l57_57766

variable (a b c d : ℕ)

theorem combined_weight :
  a + b = 260 →
  b + c = 245 →
  c + d = 270 →
  a + d = 285 :=
by
  intros hab hbc hcd
  sorry

end combined_weight_l57_57766


namespace find_n_l57_57004

open Nat

def is_solution_of_comb_perm (n : ℕ) : Prop :=
    3 * (factorial (n-1) / (factorial (n-5) * factorial 4)) = 5 * (n-2) * (n-3)

theorem find_n (n : ℕ) (h : is_solution_of_comb_perm n) (hn : n ≠ 0) : n = 9 :=
by
  -- will fill proof steps if required
  sorry

end find_n_l57_57004


namespace tenth_term_of_arithmetic_sequence_l57_57085

theorem tenth_term_of_arithmetic_sequence :
  ∃ a : ℕ → ℤ, (∀ n : ℕ, a n + 1 - a n = 2) ∧ a 1 = 1 ∧ a 10 = 19 :=
sorry

end tenth_term_of_arithmetic_sequence_l57_57085


namespace floss_per_student_l57_57632

theorem floss_per_student
  (students : ℕ)
  (yards_per_packet : ℕ)
  (floss_left_over : ℕ)
  (total_packets : ℕ)
  (total_floss : ℕ)
  (total_floss_bought : ℕ)
  (smallest_multiple_of_35 : ℕ)
  (each_student_needs : ℕ)
  (hs1 : students = 20)
  (hs2 : yards_per_packet = 35)
  (hs3 : floss_left_over = 5)
  (hs4 : total_floss = total_packets * yards_per_packet)
  (hs5 : total_floss_bought = total_floss + floss_left_over)
  (hs6 : total_floss_bought % 35 = 0)
  (hs7 : smallest_multiple_of_35 > total_packets * yards_per_packet - floss_left_over)
  (hs8 : 20 * each_student_needs + 5 = smallest_multiple_of_35)
  : each_student_needs = 5 :=
by
  sorry

end floss_per_student_l57_57632


namespace circle_through_points_eq_l57_57499

noncomputable def circle_eqn (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_through_points_eq {h k r : ℝ} :
  circle_eqn h k r (-1) 0 ∧
  circle_eqn h k r 0 2 ∧
  circle_eqn h k r 2 0 → 
  (h = 2 / 3 ∧ k = 2 / 3 ∧ r^2 = 29 / 9) :=
sorry

end circle_through_points_eq_l57_57499


namespace exists_contiguous_figure_l57_57907

-- Definition of the type for different types of rhombuses
inductive RhombusType
| wide
| narrow

-- Definition of a figure composed of rhombuses
structure Figure where
  count_wide : ℕ
  count_narrow : ℕ
  connected : Prop

-- Statement of the proof problem
theorem exists_contiguous_figure : ∃ (f : Figure), f.count_wide = 3 ∧ f.count_narrow = 8 ∧ f.connected :=
sorry

end exists_contiguous_figure_l57_57907


namespace dash_cam_mounts_max_profit_l57_57439

noncomputable def monthly_profit (x t : ℝ) : ℝ :=
  (48 + t / (2 * x)) * x - 32 * x - 3 - t

theorem dash_cam_mounts_max_profit :
  ∃ (x t : ℝ), 1 < x ∧ x < 3 ∧ x = 3 - 2 / (t + 1) ∧
  monthly_profit x t = 37.5 := by
sorry

end dash_cam_mounts_max_profit_l57_57439


namespace area_of_square_KLMN_is_25_l57_57209

-- Given a square ABCD with area 25
def ABCD_area_is_25 : Prop :=
  ∃ s : ℝ, (s * s = 25)

-- Given points K, L, M, and N forming isosceles right triangles with the sides of the square
def isosceles_right_triangles_at_vertices (A B C D K L M N : ℝ) : Prop :=
  ∃ (a b c d : ℝ),
    (a = b) ∧ (c = d) ∧
    (K - A)^2 + (B - K)^2 = (A - B)^2 ∧  -- AKB
    (L - B)^2 + (C - L)^2 = (B - C)^2 ∧  -- BLC
    (M - C)^2 + (D - M)^2 = (C - D)^2 ∧  -- CMD
    (N - D)^2 + (A - N)^2 = (D - A)^2    -- DNA

-- Given that KLMN is a square
def KLMN_is_square (K L M N : ℝ) : Prop :=
  (K - L)^2 + (L - M)^2 = (M - N)^2 + (N - K)^2

-- Proving that the area of square KLMN is 25 given the conditions
theorem area_of_square_KLMN_is_25 (A B C D K L M N : ℝ) :
  ABCD_area_is_25 → isosceles_right_triangles_at_vertices A B C D K L M N → KLMN_is_square K L M N → ∃s, s * s = 25 :=
by
  intro h1 h2 h3
  sorry

end area_of_square_KLMN_is_25_l57_57209


namespace factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l57_57921

theorem factorize_polynomial_1 (x y : ℝ) : 
  12 * x ^ 3 * y - 3 * x * y ^ 2 = 3 * x * y * (4 * x ^ 2 - y) := 
by sorry

theorem factorize_polynomial_2 (x : ℝ) : 
  x - 9 * x ^ 3 = x * (1 + 3 * x) * (1 - 3 * x) :=
by sorry

theorem factorize_polynomial_3 (a b : ℝ) : 
  3 * a ^ 2 - 12 * a * b * (a - b) = 3 * (a - 2 * b) ^ 2 := 
by sorry

end factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l57_57921


namespace find_m_l57_57498

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l57_57498


namespace gcd_mn_mn_squared_l57_57147

theorem gcd_mn_mn_squared (m n : ℕ) (h : Nat.gcd m n = 1) : ({d : ℕ | d = Nat.gcd (m + n) (m ^ 2 + n ^ 2)} ⊆ {1, 2}) := 
sorry

end gcd_mn_mn_squared_l57_57147


namespace sum_of_squares_of_four_integers_equals_175_l57_57989

theorem sum_of_squares_of_four_integers_equals_175 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a^2 + b^2 + c^2 + d^2 = 175 ∧ a + b + c + d = 23 :=
sorry

end sum_of_squares_of_four_integers_equals_175_l57_57989


namespace scott_invests_l57_57840

theorem scott_invests (x r : ℝ) (h1 : 2520 = x + 1260) (h2 : 2520 * 0.08 = x * r) : r = 0.16 :=
by
  -- Proof goes here
  sorry

end scott_invests_l57_57840


namespace express_recurring_decimal_as_fraction_l57_57967

theorem express_recurring_decimal_as_fraction (h : 0.01 = (1 : ℚ) / 99) : 2.02 = (200 : ℚ) / 99 :=
by 
  sorry

end express_recurring_decimal_as_fraction_l57_57967


namespace circle_area_is_323pi_l57_57388

-- Define points A and B
def A : ℝ × ℝ := (2, 9)
def B : ℝ × ℝ := (14, 7)

-- Define that points A and B lie on circle ω
def on_circle_omega (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = r ^ 2 ∧
  (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = r ^ 2

-- Define the tangent lines intersect at a point on the x-axis
def tangents_intersect_on_x_axis (A B : ℝ × ℝ) (C : ℝ × ℝ) (ω : (ℝ × ℝ) → ℝ): Prop := 
  ∃ x : ℝ, (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 ∧
             C.2 = 0

-- Problem statement to prove
theorem circle_area_is_323pi (C : ℝ × ℝ) (radius : ℝ) (on_circle_omega : on_circle_omega A B C radius)
  (tangents_intersect_on_x_axis : tangents_intersect_on_x_axis A B C omega) :
  π * radius ^ 2 = 323 * π :=
sorry

end circle_area_is_323pi_l57_57388


namespace soda_cost_l57_57849

variable (b s : ℕ)

theorem soda_cost (h1 : 2 * b + s = 210) (h2 : b + 2 * s = 240) : s = 90 := by
  sorry

end soda_cost_l57_57849


namespace div_five_times_eight_by_ten_l57_57932

theorem div_five_times_eight_by_ten : (5 * 8) / 10 = 4 := by
  sorry

end div_five_times_eight_by_ten_l57_57932


namespace arithmetic_seq_general_term_geometric_seq_general_term_l57_57910

theorem arithmetic_seq_general_term (a : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2) :
  ∀ n, a n = 2 * n + 2 :=
by sorry

theorem geometric_seq_general_term (a b : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2)
  (h3 : b 2 = a 3) (h4 : b 3 = a 7) :
  ∀ n, b n = 2 ^ (n + 1) :=
by sorry

end arithmetic_seq_general_term_geometric_seq_general_term_l57_57910


namespace range_sum_of_h_l57_57182

noncomputable def h (x : ℝ) : ℝ := 5 / (5 + 3 * x^2)

theorem range_sum_of_h : 
  (∃ a b : ℝ, (∀ x : ℝ, 0 < h x ∧ h x ≤ 1) ∧ a = 0 ∧ b = 1 ∧ a + b = 1) :=
sorry

end range_sum_of_h_l57_57182


namespace bus_total_capacity_l57_57746

-- Definitions based on conditions in a)
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seats_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 12

-- Proof statement
theorem bus_total_capacity : (left_side_seats + right_side_seats) * seats_per_seat + back_seat_capacity = 93 := by
  sorry

end bus_total_capacity_l57_57746


namespace total_steps_traveled_l57_57478

def steps_per_mile : ℕ := 2000
def walk_to_subway : ℕ := 2000
def subway_ride_miles : ℕ := 7
def walk_to_rockefeller : ℕ := 3000
def cab_ride_miles : ℕ := 3

theorem total_steps_traveled :
  walk_to_subway +
  (subway_ride_miles * steps_per_mile) +
  walk_to_rockefeller +
  (cab_ride_miles * steps_per_mile)
  = 24000 := 
by 
  sorry

end total_steps_traveled_l57_57478


namespace binomial_n_choose_n_sub_2_l57_57237

theorem binomial_n_choose_n_sub_2 (n : ℕ) (h : 2 ≤ n) : Nat.choose n (n - 2) = n * (n - 1) / 2 :=
by
  sorry

end binomial_n_choose_n_sub_2_l57_57237


namespace nina_money_l57_57379

-- Definitions based on the problem's conditions
def original_widgets := 15
def reduced_widgets := 25
def price_reduction := 5

-- The statement
theorem nina_money : 
  ∃ (W : ℝ), 15 * W = 25 * (W - 5) ∧ 15 * W = 187.5 :=
by
  sorry

end nina_money_l57_57379


namespace johns_watermelon_weight_l57_57055

theorem johns_watermelon_weight (michael_weight clay_weight john_weight : ℕ)
  (h1 : michael_weight = 8)
  (h2 : clay_weight = 3 * michael_weight)
  (h3 : john_weight = clay_weight / 2) :
  john_weight = 12 :=
by
  sorry

end johns_watermelon_weight_l57_57055


namespace solve_system_of_equations_l57_57423

theorem solve_system_of_equations :
  ∃ y : ℝ, (2 * 2 + y = 0) ∧ (2 + y = 3) :=
by
  sorry

end solve_system_of_equations_l57_57423


namespace divisors_congruent_mod8_l57_57477

theorem divisors_congruent_mod8 (n : ℕ) (hn : n % 2 = 1) :
  ∀ d, d ∣ (2^n - 1) → d % 8 = 1 ∨ d % 8 = 7 :=
by
  sorry

end divisors_congruent_mod8_l57_57477


namespace mason_courses_not_finished_l57_57416

-- Each necessary condition is listed as a definition.
def coursesPerWall := 6
def bricksPerCourse := 10
def numOfWalls := 4
def totalBricksUsed := 220

-- Creating an entity to store the problem and prove it.
theorem mason_courses_not_finished : 
  (numOfWalls * coursesPerWall * bricksPerCourse - totalBricksUsed) / bricksPerCourse = 2 := 
by
  sorry

end mason_courses_not_finished_l57_57416


namespace probability_div_int_l57_57494

theorem probability_div_int
    (r : ℤ) (k : ℤ)
    (hr : -5 < r ∧ r < 10)
    (hk : 1 < k ∧ k < 8)
    (hk_prime : Nat.Prime (Int.natAbs k)) :
    ∃ p q : ℕ, (p = 3 ∧ q = 14) ∧ p / q = 3 / 14 := 
by {
  sorry
}

end probability_div_int_l57_57494


namespace original_rectangle_area_l57_57134

theorem original_rectangle_area : 
  ∃ (a b : ℤ), (a + b = 20) ∧ (a * b = 96) := by
  sorry

end original_rectangle_area_l57_57134


namespace tap_filling_time_l57_57223

theorem tap_filling_time (T : ℝ) 
  (h_total : (1 / 3) = (1 / T + 1 / 15 + 1 / 6)) : T = 10 := 
sorry

end tap_filling_time_l57_57223


namespace tripling_base_exponent_l57_57848

variables (a b x : ℝ)

theorem tripling_base_exponent (b_ne_zero : b ≠ 0) (r_def : (3 * a)^(3 * b) = a^b * x^b) : x = 27 * a^2 :=
by
  -- Proof omitted as requested
  sorry

end tripling_base_exponent_l57_57848


namespace percentage_increase_second_year_l57_57056

theorem percentage_increase_second_year :
  let initial_deposit : ℤ := 1000
  let balance_first_year : ℤ := 1100
  let total_balance_two_years : ℤ := 1320
  let percent_increase_first_year : ℚ := ((balance_first_year - initial_deposit) / initial_deposit) * 100
  let percent_increase_total : ℚ := ((total_balance_two_years - initial_deposit) / initial_deposit) * 100
  let increase_second_year : ℤ := total_balance_two_years - balance_first_year
  let percent_increase_second_year : ℚ := (increase_second_year / balance_first_year) * 100
  percent_increase_first_year = 10 ∧
  percent_increase_total = 32 ∧
  increase_second_year = 220 → 
  percent_increase_second_year = 20 := by
  intros initial_deposit balance_first_year total_balance_two_years percent_increase_first_year
         percent_increase_total increase_second_year percent_increase_second_year
  sorry

end percentage_increase_second_year_l57_57056


namespace problem_l57_57038

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l57_57038


namespace DeansCalculatorGame_l57_57796

theorem DeansCalculatorGame (r : ℕ) (c1 c2 c3 : ℤ) (h1 : r = 45) (h2 : c1 = 1) (h3 : c2 = 0) (h4 : c3 = -2) : 
  let final1 := (c1 ^ 3)
  let final2 := (c2 ^ 2)
  let final3 := (-c3)^45
  final1 + final2 + final3 = 3 := 
by
  sorry

end DeansCalculatorGame_l57_57796


namespace division_problem_l57_57426

theorem division_problem (D : ℕ) (Quotient Dividend Remainder : ℕ) 
    (h1 : Quotient = 36) 
    (h2 : Dividend = 3086) 
    (h3 : Remainder = 26) 
    (h_div : Dividend = (D * Quotient) + Remainder) : 
    D = 85 := 
by 
  -- Steps to prove the theorem will go here
  sorry

end division_problem_l57_57426


namespace period_of_f_is_4_and_f_2pow_n_zero_l57_57267

noncomputable def f : ℝ → ℝ := sorry

variables (hf_diff : differentiable ℝ f)
          (hf_nonzero : ∃ x, f x ≠ 0)
          (hf_odd_2 : ∀ x, f (x + 2) = -f (-x - 2))
          (hf_even_2x1 : ∀ x, f (2 * x + 1) = f (-(2 * x + 1)))

theorem period_of_f_is_4_and_f_2pow_n_zero (n : ℕ) (hn : 0 < n) :
  (∀ x, f (x + 4) = f x) ∧ f (2^n) = 0 :=
sorry

end period_of_f_is_4_and_f_2pow_n_zero_l57_57267


namespace find_alpha_l57_57422

theorem find_alpha (n : ℕ) (h : ∀ x : ℤ, x * x * x + α * x + 4 - 2 * 2016 ^ n = 0 → ∀ r : ℤ, x = r)
  : α = -3 :=
sorry

end find_alpha_l57_57422


namespace percentage_reduction_l57_57531

-- Define the problem within given conditions
def original_length := 30 -- original length in seconds
def new_length := 21 -- new length in seconds

-- State the theorem that needs to be proved
theorem percentage_reduction (original_length new_length : ℕ) : 
  original_length = 30 → 
  new_length = 21 → 
  ((original_length - new_length) / original_length: ℚ) * 100 = 30 :=
by 
  sorry

end percentage_reduction_l57_57531


namespace candy_store_truffle_price_l57_57281

def total_revenue : ℝ := 212
def fudge_revenue : ℝ := 20 * 2.5
def pretzels_revenue : ℝ := 3 * 12 * 2.0
def truffles_quantity : ℕ := 5 * 12

theorem candy_store_truffle_price (total_revenue fudge_revenue pretzels_revenue truffles_quantity : ℝ) : 
  (total_revenue - (fudge_revenue + pretzels_revenue)) / truffles_quantity = 1.50 := 
by 
  sorry

end candy_store_truffle_price_l57_57281


namespace mary_circus_change_l57_57559

theorem mary_circus_change :
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  change = 15 :=
by
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  sorry

end mary_circus_change_l57_57559


namespace find_x_values_l57_57269

theorem find_x_values (x : ℝ) (h : x + 60 / (x - 3) = -12) : x = -3 ∨ x = -6 :=
sorry

end find_x_values_l57_57269


namespace sequence_general_term_formula_l57_57051

-- Definitions based on conditions
def alternating_sign (n : ℕ) : ℤ := (-1) ^ n
def arithmetic_sequence (n : ℕ) : ℤ := 4 * n - 3

-- Definition for the general term formula
def general_term (n : ℕ) : ℤ := alternating_sign n * arithmetic_sequence n

-- Theorem stating that the given sequence's general term formula is a_n = (-1)^n * (4n - 3)
theorem sequence_general_term_formula (n : ℕ) : general_term n = (-1) ^ n * (4 * n - 3) :=
by
  -- Proof logic will go here
  sorry

end sequence_general_term_formula_l57_57051


namespace distance_diff_is_0_point3_l57_57429

def john_walk_distance : ℝ := 0.7
def nina_walk_distance : ℝ := 0.4
def distance_difference_john_nina : ℝ := john_walk_distance - nina_walk_distance

theorem distance_diff_is_0_point3 : distance_difference_john_nina = 0.3 :=
by
  -- proof goes here
  sorry

end distance_diff_is_0_point3_l57_57429


namespace new_average_is_minus_one_l57_57188

noncomputable def new_average_of_deducted_sequence : ℤ :=
  let n := 15
  let avg := 20
  let seq_sum := n * avg
  let x := (seq_sum - (n * (n-1) / 2)) / n
  let deductions := (n-1) * n * 3 / 2
  let new_sum := seq_sum - deductions
  new_sum / n

theorem new_average_is_minus_one : new_average_of_deducted_sequence = -1 := 
  sorry

end new_average_is_minus_one_l57_57188


namespace tangent_division_l57_57233

theorem tangent_division (a b c d e : ℝ) (h0 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) :
  ∃ t1 t5 : ℝ, t1 = (a + b - c - d + e) / 2 ∧ t5 = (a - b - c + d + e) / 2 ∧ t1 + t5 = a :=
by
  sorry

end tangent_division_l57_57233


namespace total_food_items_donated_l57_57408

def FosterFarmsDonation : ℕ := 45
def AmericanSummitsDonation : ℕ := 2 * FosterFarmsDonation
def HormelDonation : ℕ := 3 * FosterFarmsDonation
def BoudinButchersDonation : ℕ := HormelDonation / 3
def DelMonteFoodsDonation : ℕ := AmericanSummitsDonation - 30

theorem total_food_items_donated :
  FosterFarmsDonation + AmericanSummitsDonation + HormelDonation + BoudinButchersDonation + DelMonteFoodsDonation = 375 :=
by
  sorry

end total_food_items_donated_l57_57408


namespace inequality_condition_l57_57714

theorem inequality_condition (a x : ℝ) : 
  x^3 + 13 * a^2 * x > 5 * a * x^2 + 9 * a^3 ↔ x > a := 
by
  sorry

end inequality_condition_l57_57714


namespace math_proof_l57_57838

noncomputable def problem (a b : ℝ) : Prop :=
  a - b = 2 ∧ a^2 + b^2 = 25 → a * b = 10.5

-- We state the problem as a theorem:
theorem math_proof (a b : ℝ) (h1: a - b = 2) (h2: a^2 + b^2 = 25) : a * b = 10.5 :=
by {
  sorry -- Proof goes here
}

end math_proof_l57_57838


namespace next_in_step_distance_l57_57133

theorem next_in_step_distance
  (jack_stride jill_stride : ℕ)
  (h1 : jack_stride = 64)
  (h2 : jill_stride = 56) :
  Nat.lcm jack_stride jill_stride = 448 := by
  sorry

end next_in_step_distance_l57_57133


namespace sin_A_and_height_on_AB_l57_57821

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l57_57821


namespace union_M_N_equals_0_1_5_l57_57193

def M : Set ℝ := { x | x^2 - 6 * x + 5 = 0 }
def N : Set ℝ := { x | x^2 - 5 * x = 0 }

theorem union_M_N_equals_0_1_5 : M ∪ N = {0, 1, 5} := by
  sorry

end union_M_N_equals_0_1_5_l57_57193


namespace sin_gt_cos_range_l57_57507

theorem sin_gt_cos_range (x : ℝ) : 
  0 < x ∧ x < 2 * Real.pi → (Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4)) := by
  sorry

end sin_gt_cos_range_l57_57507


namespace phil_quarters_collection_l57_57965

theorem phil_quarters_collection
    (initial_quarters : ℕ)
    (doubled_quarters : ℕ)
    (additional_quarters_per_month : ℕ)
    (total_quarters_end_of_second_year : ℕ)
    (quarters_collected_every_third_month : ℕ)
    (total_quarters_end_of_third_year : ℕ)
    (remaining_quarters_after_loss : ℕ)
    (quarters_left : ℕ) :
    initial_quarters = 50 →
    doubled_quarters = 2 * initial_quarters →
    additional_quarters_per_month = 3 →
    total_quarters_end_of_second_year = doubled_quarters + 12 * additional_quarters_per_month →
    total_quarters_end_of_third_year = total_quarters_end_of_second_year + 4 * quarters_collected_every_third_month →
    remaining_quarters_after_loss = (3 / 4 : ℚ) * total_quarters_end_of_third_year → 
    quarters_left = 105 →
    quarters_collected_every_third_month = 1 := 
by
  sorry

end phil_quarters_collection_l57_57965


namespace abigail_fence_building_time_l57_57277

def abigail_time_per_fence (total_built: ℕ) (additional_hours: ℕ) (total_fences: ℕ): ℕ :=
  (additional_hours * 60) / (total_fences - total_built)

theorem abigail_fence_building_time :
  abigail_time_per_fence 10 8 26 = 30 :=
sorry

end abigail_fence_building_time_l57_57277


namespace unique_bounded_sequence_exists_l57_57114

variable (a : ℝ) (n : ℕ) (hn_pos : n > 0)

theorem unique_bounded_sequence_exists :
  ∃! (x : ℕ → ℝ), (x 0 = 0) ∧ (x (n+1) = 0) ∧
                   (∀ i, 1 ≤ i ∧ i ≤ n → (1/2) * (x (i+1) + x (i-1)) = x i + x i ^ 3 - a ^ 3) ∧
                   (∀ i, i ≤ n + 1 → |x i| ≤ |a|) := by
  sorry

end unique_bounded_sequence_exists_l57_57114


namespace lollipops_left_for_becky_l57_57175
-- Import the Mathlib library

-- Define the conditions as given in the problem
def lemon_lollipops : ℕ := 75
def peppermint_lollipops : ℕ := 210
def watermelon_lollipops : ℕ := 6
def marshmallow_lollipops : ℕ := 504
def friends : ℕ := 13

-- Total number of lollipops
def total_lollipops : ℕ := lemon_lollipops + peppermint_lollipops + watermelon_lollipops + marshmallow_lollipops

-- Statement to prove that the remainder after distributing the total lollipops among friends is 2
theorem lollipops_left_for_becky : total_lollipops % friends = 2 := by
  -- Proof goes here
  sorry

end lollipops_left_for_becky_l57_57175


namespace negation_of_proposition_l57_57090

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l57_57090


namespace number_of_small_pipes_needed_l57_57610

theorem number_of_small_pipes_needed :
  let diameter_large := 8
  let diameter_small := 1
  let radius_large := diameter_large / 2
  let radius_small := diameter_small / 2
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let num_small_pipes := area_large / area_small
  num_small_pipes = 64 :=
by
  sorry

end number_of_small_pipes_needed_l57_57610


namespace work_hours_l57_57847

namespace JohnnyWork

variable (dollarsPerHour : ℝ) (totalDollars : ℝ)

theorem work_hours 
  (h_wage : dollarsPerHour = 3.25)
  (h_earned : totalDollars = 26) 
  : (totalDollars / dollarsPerHour) = 8 := 
by
  rw [h_wage, h_earned]
  -- proof goes here
  sorry

end JohnnyWork

end work_hours_l57_57847


namespace max_H2O_produced_l57_57042

theorem max_H2O_produced :
  ∀ (NaOH H2SO4 H2O : ℝ)
  (n_NaOH : NaOH = 1.5)
  (n_H2SO4 : H2SO4 = 1)
  (balanced_reaction : 2 * NaOH + H2SO4 = 2 * H2O + 1 * (NaOH + H2SO4)),
  H2O = 1.5 :=
by
  intros NaOH H2SO4 H2O n_NaOH n_H2SO4 balanced_reaction
  sorry

end max_H2O_produced_l57_57042


namespace xiaoming_original_phone_number_l57_57236

variable (d1 d2 d3 d4 d5 d6 : Nat)

def original_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

def upgraded_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  20000000 + 1000000 * d1 + 80000 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem xiaoming_original_phone_number :
  let x := original_phone_number d1 d2 d3 d4 d5 d6
  let x' := upgraded_phone_number d1 d2 d3 d4 d5 d6
  (x' = 81 * x) → (x = 282500) :=
by
  sorry

end xiaoming_original_phone_number_l57_57236


namespace union_sets_a_l57_57655

theorem union_sets_a (P S : Set ℝ) (a : ℝ) :
  P = {1, 5, 10} →
  S = {1, 3, a^2 + 1} →
  S ∪ P = {1, 3, 5, 10} →
  a = 2 ∨ a = -2 ∨ a = 3 ∨ a = -3 :=
by
  intros hP hS hUnion 
  sorry

end union_sets_a_l57_57655


namespace walnut_trees_l57_57322

theorem walnut_trees (logs_per_pine logs_per_maple logs_per_walnut pine_trees maple_trees total_logs walnut_trees : ℕ)
  (h1 : logs_per_pine = 80)
  (h2 : logs_per_maple = 60)
  (h3 : logs_per_walnut = 100)
  (h4 : pine_trees = 8)
  (h5 : maple_trees = 3)
  (h6 : total_logs = 1220)
  (h7 : total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut) :
  walnut_trees = 4 :=
by
  sorry

end walnut_trees_l57_57322


namespace number_of_days_A_left_l57_57763

noncomputable def work_problem (W : ℝ) : Prop :=
  let A_rate := W / 45
  let B_rate := W / 40
  let days_B_alone := 23
  ∃ x : ℝ, x * (A_rate + B_rate) + days_B_alone * B_rate = W ∧ x = 9

theorem number_of_days_A_left (W : ℝ) : work_problem W :=
  sorry

end number_of_days_A_left_l57_57763


namespace product_is_cube_l57_57844

/-
  Given conditions:
    - a, b, and c are distinct composite natural numbers.
    - None of a, b, and c are divisible by any of the integers from 2 to 100 inclusive.
    - a, b, and c are the smallest possible numbers satisfying the above conditions.

  We need to prove that their product a * b * c is a cube of a natural number.
-/

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q

theorem product_is_cube (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : is_composite a) (h5 : is_composite b) (h6 : is_composite c)
  (h7 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ a))
  (h8 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ b))
  (h9 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ c))
  (h10 : ∀ (d e f : ℕ), is_composite d → is_composite e → is_composite f → d ≠ e → e ≠ f → d ≠ f → 
         (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ d)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ e)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ f)) →
         (d * e * f ≥ a * b * c)) :
  ∃ (n : ℕ), a * b * c = n ^ 3 :=
by
  sorry

end product_is_cube_l57_57844


namespace tangent_points_l57_57226

noncomputable def curve (x : ℝ) : ℝ := x^3 - x - 1

theorem tangent_points (x y : ℝ) (h : y = curve x) (slope_line : ℝ) (h_slope : slope_line = -1/2)
  (tangent_perpendicular : (3 * x^2 - 1) = 2) :
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := sorry

end tangent_points_l57_57226


namespace smallest_of_three_consecutive_odd_numbers_l57_57754

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) 
(h_sum : x + (x+2) + (x+4) = 69) : x = 21 :=
by
  sorry

end smallest_of_three_consecutive_odd_numbers_l57_57754


namespace unit_vector_parallel_to_a_l57_57969

theorem unit_vector_parallel_to_a (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 12 * y = 5 * x) :
  (x = 12 / 13 ∧ y = 5 / 13) ∨ (x = -12 / 13 ∧ y = -5 / 13) := by
  sorry

end unit_vector_parallel_to_a_l57_57969


namespace simplify_expression_l57_57943

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 6) * (2 * x + 8) - (x + 6) * (3 * x + 1) = 3 * x^2 - 7 * x - 54 :=
by
  sorry

end simplify_expression_l57_57943


namespace mixed_number_subtraction_l57_57166

theorem mixed_number_subtraction :
  2 + 5 / 6 - (1 + 1 / 3) = 3 / 2 := by
sorry

end mixed_number_subtraction_l57_57166


namespace fraction_comparison_l57_57072

theorem fraction_comparison (a b c d : ℝ) (h1 : a / b < c / d) (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < 1/2 * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l57_57072


namespace function_equality_l57_57488

theorem function_equality (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x : ℝ, f x = (1/2) * x^2 - x + (3/2) :=
by
  sorry

end function_equality_l57_57488


namespace Lisa_total_spoons_l57_57997

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l57_57997


namespace product_of_three_consecutive_integers_l57_57806

theorem product_of_three_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 740)
    (x1 : ℕ := x - 1) (x2 : ℕ := x) (x3 : ℕ := x + 1) :
    x1 * x2 * x3 = 17550 :=
by
  sorry

end product_of_three_consecutive_integers_l57_57806


namespace intersection_M_N_l57_57432

noncomputable def M : Set ℝ := { x | x^2 + x - 2 = 0 }
def N : Set ℝ := { x | x < 0 }

theorem intersection_M_N : M ∩ N = { -2 } := by
  sorry

end intersection_M_N_l57_57432


namespace find_f_ln2_l57_57956

noncomputable def f : ℝ → ℝ := sorry

axiom fx_monotonic : Monotone f
axiom fx_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - Real.exp 1

theorem find_f_ln2 : f (Real.log 2) = -1 := 
sorry

end find_f_ln2_l57_57956


namespace area_of_rhombus_l57_57992

-- Defining the lengths of the diagonals
variable (d1 d2 : ℝ)
variable (d1_eq : d1 = 15)
variable (d2_eq : d2 = 20)

-- Goal is to prove the area given the diagonal lengths
theorem area_of_rhombus (d1 d2 : ℝ) (d1_eq : d1 = 15) (d2_eq : d2 = 20) : 
  (d1 * d2) / 2 = 150 := 
by
  -- Using the given conditions for the proof
  sorry

end area_of_rhombus_l57_57992


namespace find_pairs_of_positive_numbers_l57_57577

theorem find_pairs_of_positive_numbers
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (exists_triangle : ∃ (C D E A B : ℝ), true)
  (points_on_hypotenuse : ∀ (C D E A B : ℝ), A ∈ [D, E] ∧ B ∈ [D, E]) 
  (equal_vectors : ∀ (D A B E : ℝ), (D - A) = (A - B) ∧ (A - B) = (B - E))
  (AC_eq_a : (C - A) = a)
  (BC_eq_b : (C - B) = b) :
  (1 / 2) < (a / b) ∧ (a / b) < 2 :=
by {
  sorry
}

end find_pairs_of_positive_numbers_l57_57577


namespace at_least_two_foxes_met_same_number_of_koloboks_l57_57624

-- Define the conditions
def number_of_foxes : ℕ := 14
def number_of_koloboks : ℕ := 92

-- The theorem statement to be proven
theorem at_least_two_foxes_met_same_number_of_koloboks :
  ∃ (f : Fin number_of_foxes.succ → ℕ), 
    (∀ i, f i ≤ number_of_koloboks) ∧ 
    ∃ i j, i ≠ j ∧ f i = f j :=
by
  sorry

end at_least_two_foxes_met_same_number_of_koloboks_l57_57624


namespace area_of_highest_points_l57_57205

noncomputable def highest_point_area (u g : ℝ) : ℝ :=
  let x₁ := u^2 / (2 * g)
  let x₂ := 2 * u^2 / g
  (1/4) * ((x₂^2) - (x₁^2))

theorem area_of_highest_points (u g : ℝ) : highest_point_area u g = 3 * u^4 / (4 * g^2) :=
by
  sorry

end area_of_highest_points_l57_57205


namespace expression_simplification_l57_57706

def base_expr := (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) *
                (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64)

theorem expression_simplification :
  base_expr = 3^128 - 4^128 := by
  sorry

end expression_simplification_l57_57706


namespace horner_value_v2_l57_57348

def poly (x : ℤ) : ℤ := 208 + 9 * x^2 + 6 * x^4 + x^6

theorem horner_value_v2 : poly (-4) = ((((0 + -4) * -4 + 6) * -4 + 9) * -4 + 208) :=
by
  sorry

end horner_value_v2_l57_57348


namespace games_lost_l57_57126

theorem games_lost (total_games won_games : ℕ) (h_total : total_games = 12) (h_won : won_games = 8) :
  (total_games - won_games) = 4 :=
by
  -- Placeholder for the proof
  sorry

end games_lost_l57_57126


namespace quadratic_solution_l57_57931

theorem quadratic_solution (a b : ℚ) (h : a * 1^2 + b * 1 + 1 = 0) : 3 - a - b = 4 := 
by
  sorry

end quadratic_solution_l57_57931


namespace eccentricity_range_of_isosceles_right_triangle_l57_57813

theorem eccentricity_range_of_isosceles_right_triangle
  (a : ℝ) (e : ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x^2)/(a^2) + y^2 = 1)
  (h_a_gt_1 : a > 1)
  (B C : ℝ × ℝ)
  (isosceles_right_triangle : ∀ (A B C : ℝ × ℝ), ∃ k : ℝ, k > 0 ∧ 
    B = (-(2*k*a^2)/(1 + a^2*k^2), 0) ∧ 
    C = ((2*k*a^2)/(a^2 + k^2), 0) ∧ 
    (B.1^2 + B.2^2 = C.1^2 + C.2^2 + 1))
  (unique_solution : ∀ (k : ℝ), ∃! k', k' = 1)
  : 0 < e ∧ e ≤ (Real.sqrt 6) / 3 :=
sorry

end eccentricity_range_of_isosceles_right_triangle_l57_57813


namespace rate_percent_is_10_l57_57344

theorem rate_percent_is_10
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ) 
  (h1 : SI = 2500) (h2 : P = 5000) (h3 : T = 5) :
  R = 10 :=
by
  sorry

end rate_percent_is_10_l57_57344


namespace max_abs_value_inequality_l57_57153

theorem max_abs_value_inequality (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ (a b : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) ∧ |20 * a + 14 * b| + |20 * a - 14 * b| = 80 := 
sorry

end max_abs_value_inequality_l57_57153


namespace purely_imaginary_condition_l57_57657

-- Define the necessary conditions
def real_part_eq_zero (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0
def imaginary_part_neq_zero (m : ℝ) : Prop := m^2 - 3 * m + 2 ≠ 0

-- State the theorem to be proved
theorem purely_imaginary_condition (m : ℝ) :
  real_part_eq_zero m ∧ imaginary_part_neq_zero m ↔ m = -1/2 :=
sorry

end purely_imaginary_condition_l57_57657


namespace geometric_sequence_common_ratio_l57_57860

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 5/4) 
  (h_sequence : ∀ n, a n = a 1 * q ^ (n - 1)) : 
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l57_57860


namespace fraction_exists_l57_57073

theorem fraction_exists (n d k : ℕ) (h₁ : n = k * d) (h₂ : d > 0) (h₃ : k > 0) : 
  ∃ (i j : ℕ), i < n ∧ j < n ∧ i + j = n ∧ i/j = d-1 :=
by
  sorry

end fraction_exists_l57_57073


namespace distance_proof_l57_57340

/-- Maxwell's walking speed in km/h. -/
def Maxwell_speed := 4

/-- Time Maxwell walks before meeting Brad in hours. -/
def Maxwell_time := 10

/-- Brad's running speed in km/h. -/
def Brad_speed := 6

/-- Time Brad runs before meeting Maxwell in hours. -/
def Brad_time := 9

/-- Distance between Maxwell and Brad's homes in km. -/
def distance_between_homes : ℕ := 94

/-- Prove the distance between their homes is 94 km given the conditions. -/
theorem distance_proof 
  (h1 : Maxwell_speed * Maxwell_time = 40)
  (h2 : Brad_speed * Brad_time = 54) :
  Maxwell_speed * Maxwell_time + Brad_speed * Brad_time = distance_between_homes := 
by 
  sorry

end distance_proof_l57_57340


namespace simplify_expression_l57_57905

variable (a : ℝ)

theorem simplify_expression (a : ℝ) : (3 * a) ^ 2 * a ^ 5 = 9 * a ^ 7 :=
by sorry

end simplify_expression_l57_57905


namespace snow_probability_first_week_l57_57668

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l57_57668


namespace solve_fraction_inequality_l57_57081

theorem solve_fraction_inequality :
  { x : ℝ | x / (x + 5) ≥ 0 } = { x : ℝ | x < -5 } ∪ { x : ℝ | x ≥ 0 } := by
  sorry

end solve_fraction_inequality_l57_57081


namespace largest_integer_satisfying_conditions_l57_57019

theorem largest_integer_satisfying_conditions (n : ℤ) (m : ℤ) :
  n^2 = (m + 1)^3 - m^3 ∧ ∃ k : ℤ, 2 * n + 103 = k^2 → n = 313 := 
by 
  sorry

end largest_integer_satisfying_conditions_l57_57019


namespace sufficient_but_not_necessary_condition_l57_57393

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3
def g (k x : ℝ) : ℝ := k * x - 1

theorem sufficient_but_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, f x ≥ g k x) ↔ (-6 ≤ k ∧ k ≤ 2) :=
sorry

end sufficient_but_not_necessary_condition_l57_57393


namespace rationalize_denominator_sqrt_l57_57253

theorem rationalize_denominator_sqrt (x y : ℝ) (hx : x = 5) (hy : y = 12) :
  Real.sqrt (x / y) = Real.sqrt 15 / 6 :=
by
  rw [hx, hy]
  sorry

end rationalize_denominator_sqrt_l57_57253


namespace number_minus_six_l57_57441

variable (x : ℤ)

theorem number_minus_six
  (h : x / 5 = 2) : x - 6 = 4 := 
sorry

end number_minus_six_l57_57441


namespace greatest_common_multiple_less_than_bound_l57_57206

-- Define the numbers and the bound
def num1 : ℕ := 15
def num2 : ℕ := 10
def bound : ℕ := 150

-- Define the LCM of num1 and num2
def lcm_num1_num2 : ℕ := Nat.lcm num1 num2

-- Define the greatest multiple of LCM less than bound
def greatest_multiple_less_than_bound (lcm : ℕ) (b : ℕ) : ℕ :=
  (b / lcm) * lcm

-- Main theorem
theorem greatest_common_multiple_less_than_bound :
  greatest_multiple_less_than_bound lcm_num1_num2 bound = 120 :=
by
  sorry

end greatest_common_multiple_less_than_bound_l57_57206


namespace length_MN_of_circle_l57_57487

def point := ℝ × ℝ

def circle_passing_through (A B C: point) :=
  ∃ (D E F : ℝ), ∀ (p : point), p = A ∨ p = B ∨ p = C →
    (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)

theorem length_MN_of_circle (A B C : point) (H : circle_passing_through A B C) :
  A = (1, 3) → B = (4, 2) → C = (1, -7) →
  ∃ M N : ℝ, (A.1 * 0 + N^2 + D * 0 + E * N + F = 0) ∧ (A.1 * 0 + M^2 + D * 0 + E * M + F = 0) ∧
  abs (M - N) = 4 * Real.sqrt 6 := 
sorry

end length_MN_of_circle_l57_57487


namespace least_n_questions_l57_57723

theorem least_n_questions {n : ℕ} : 
  (1/2 : ℝ)^n < 1/10 → n ≥ 4 :=
by
  sorry

end least_n_questions_l57_57723


namespace right_triangle_min_perimeter_multiple_13_l57_57272

theorem right_triangle_min_perimeter_multiple_13 :
  ∃ (a b c : ℕ), 
    (a^2 + b^2 = c^2) ∧ 
    (a % 13 = 0 ∨ b % 13 = 0) ∧
    (a < b) ∧ 
    (a + b > c) ∧ 
    (a + b + c = 24) :=
sorry

end right_triangle_min_perimeter_multiple_13_l57_57272


namespace num_positive_divisors_of_720_multiples_of_5_l57_57121

theorem num_positive_divisors_of_720_multiples_of_5 :
  (∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 2 ∧ c = 1) →
  ∃ (n : ℕ), n = 15 :=
by
  -- Proof will go here
  sorry

end num_positive_divisors_of_720_multiples_of_5_l57_57121


namespace product_of_x_and_y_l57_57818

variables (EF FG GH HE : ℕ) (x y : ℕ)

theorem product_of_x_and_y (h1: EF = 42) (h2: FG = 4 * y^3) (h3: GH = 2 * x + 10) (h4: HE = 32) (h5: EF = GH) (h6: FG = HE) :
  x * y = 32 :=
by
  sorry

end product_of_x_and_y_l57_57818


namespace remainder_of_five_consecutive_odds_mod_12_l57_57378

/-- Let x be an odd integer. Prove that (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 
    when x ≡ 5 (mod 12). -/
theorem remainder_of_five_consecutive_odds_mod_12 {x : ℤ} (h : x % 12 = 5) :
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 :=
sorry

end remainder_of_five_consecutive_odds_mod_12_l57_57378


namespace toy_store_bears_shelves_l57_57782

theorem toy_store_bears_shelves (initial_stock shipment bears_per_shelf total_bears number_of_shelves : ℕ)
  (h1 : initial_stock = 17)
  (h2 : shipment = 10)
  (h3 : bears_per_shelf = 9)
  (h4 : total_bears = initial_stock + shipment)
  (h5 : number_of_shelves = total_bears / bears_per_shelf) :
  number_of_shelves = 3 :=
by
  sorry

end toy_store_bears_shelves_l57_57782


namespace polynomial_factorization_l57_57729

theorem polynomial_factorization (a b : ℤ) (h : (x^2 + x - 6) = (x + a) * (x + b)) :
  (a + b)^2023 = 1 :=
sorry

end polynomial_factorization_l57_57729


namespace spinner_final_direction_l57_57764

-- Define the directions as an enumeration
inductive Direction
| north
| east
| south
| west

-- Convert between revolution fractions to direction
def direction_after_revolutions (initial : Direction) (revolutions : ℚ) : Direction :=
  let quarters := (revolutions * 4) % 4
  match initial with
  | Direction.south => if quarters == 0 then Direction.south
                       else if quarters == 1 then Direction.west
                       else if quarters == 2 then Direction.north
                       else Direction.east
  | Direction.east  => if quarters == 0 then Direction.east
                       else if quarters == 1 then Direction.south
                       else if quarters == 2 then Direction.west
                       else Direction.north
  | Direction.north => if quarters == 0 then Direction.north
                       else if quarters == 1 then Direction.east
                       else if quarters == 2 then Direction.south
                       else Direction.west
  | Direction.west  => if quarters == 0 then Direction.west
                       else if quarters == 1 then Direction.north
                       else if quarters == 2 then Direction.east
                       else Direction.south

-- Final proof statement
theorem spinner_final_direction : direction_after_revolutions Direction.south (4 + 3/4 - (6 + 1/2)) = Direction.east := 
by 
  sorry

end spinner_final_direction_l57_57764


namespace smallest_base_conversion_l57_57880

theorem smallest_base_conversion :
  let n1 := 8 * 9 + 5 -- 85 in base 9
  let n2 := 2 * 6^2 + 1 * 6 -- 210 in base 6
  let n3 := 1 * 4^3 -- 1000 in base 4
  let n4 := 1 * 2^7 - 1 -- 1111111 in base 2
  n3 < n1 ∧ n3 < n2 ∧ n3 < n4 :=
by
  let n1 := 8 * 9 + 5
  let n2 := 2 * 6^2 + 1 * 6
  let n3 := 1 * 4^3
  let n4 := 1 * 2^7 - 1
  sorry

end smallest_base_conversion_l57_57880


namespace triangle_sine_ratio_l57_57993

-- Define points A and C
def A : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the condition of point B being on the ellipse
def isOnEllipse (B : ℝ × ℝ) : Prop :=
  (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1

-- Define the sin law ratio we need to prove
noncomputable def sin_ratio (sin_A sin_C sin_B : ℝ) : ℝ := 
  (sin_A + sin_C) / sin_B

-- Prove the required sine ratio condition
theorem triangle_sine_ratio (B : ℝ × ℝ) (sin_A sin_C sin_B : ℝ)
  (hB : isOnEllipse B) (hA : sin_A = 0) (hC : sin_C = 0) (hB_nonzero : sin_B ≠ 0) :
  sin_ratio sin_A sin_C sin_B = 2 :=
by
  -- Skipping proof
  sorry

end triangle_sine_ratio_l57_57993


namespace union_A_B_l57_57790

noncomputable def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
noncomputable def B : Set ℝ := {x | x^2 - 1 < 0}

theorem union_A_B : A ∪ B = {x : ℝ | -1 < x} := by
  sorry

end union_A_B_l57_57790


namespace simplify_sub_polynomials_l57_57204

def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 5 * r - 4
def g (r : ℝ) : ℝ := r^3 + 3 * r^2 + 7 * r - 2

theorem simplify_sub_polynomials (r : ℝ) : f r - g r = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end simplify_sub_polynomials_l57_57204


namespace volume_region_inequality_l57_57047

theorem volume_region_inequality : 
  ∃ (V : ℝ), V = (20 / 3) ∧ 
    ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 4 
    → x^2 + y^2 + z^2 ≤ V :=
sorry

end volume_region_inequality_l57_57047


namespace sum_of_extreme_values_of_x_l57_57087

open Real

theorem sum_of_extreme_values_of_x 
  (x y z : ℝ)
  (h1 : x + y + z = 6)
  (h2 : x^2 + y^2 + z^2 = 14) : 
  (min x + max x) = (10 / 3) :=
sorry

end sum_of_extreme_values_of_x_l57_57087


namespace sum_rows_7_8_pascal_triangle_l57_57663

theorem sum_rows_7_8_pascal_triangle : (2^7 + 2^8 = 384) :=
by
  sorry

end sum_rows_7_8_pascal_triangle_l57_57663


namespace rosa_bonheur_birth_day_l57_57712

/--
Given that Rosa Bonheur's 210th birthday was celebrated on a Wednesday,
prove that she was born on a Sunday.
-/
theorem rosa_bonheur_birth_day :
  let anniversary_year := 2022
  let birth_year := 1812
  let total_years := anniversary_year - birth_year
  let leap_years := (total_years / 4) - (total_years / 100) + (total_years / 400)
  let regular_years := total_years - leap_years
  let day_shifts := regular_years + 2 * leap_years
  (3 - day_shifts % 7) % 7 = 0 := 
sorry

end rosa_bonheur_birth_day_l57_57712


namespace max_value_sin2x_cos2x_l57_57337

open Real

theorem max_value_sin2x_cos2x (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :
  (sin (2 * x) + cos (2 * x) ≤ sqrt 2) ∧
  (∃ y, (0 ≤ y ∧ y ≤ π / 2) ∧ (sin (2 * y) + cos (2 * y) = sqrt 2)) :=
by
  sorry

end max_value_sin2x_cos2x_l57_57337


namespace percentage_problem_l57_57077

theorem percentage_problem (x : ℝ)
  (h : 0.70 * 600 = 0.40 * x) : x = 1050 :=
sorry

end percentage_problem_l57_57077


namespace points_on_line_l57_57356

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l57_57356


namespace actual_price_of_good_l57_57680

theorem actual_price_of_good (P : ℝ) (h : 0.684 * P = 6600) : P = 9649.12 :=
sorry

end actual_price_of_good_l57_57680


namespace simplify_expression_correct_l57_57807

noncomputable def simplify_expression (α : ℝ) : ℝ :=
    (2 * (Real.cos (2 * α))^2 - 1) / 
    (2 * Real.tan ((Real.pi / 4) - 2 * α) * (Real.sin ((3 * Real.pi / 4) - 2 * α))^2) -
    Real.tan (2 * α) + Real.cos (2 * α) - Real.sin (2 * α)

theorem simplify_expression_correct (α : ℝ) : 
    simplify_expression α = 
    (2 * Real.sqrt 2 * Real.sin ((Real.pi / 4) - 2 * α) * (Real.cos α)^2) /
    Real.cos (2 * α) := by
    sorry

end simplify_expression_correct_l57_57807


namespace part1_solution_part2_solution_l57_57063

def f (x : ℝ) (a : ℝ) := |x + 1| - |a * x - 1|

-- Statement for part 1
theorem part1_solution (x : ℝ) : (f x 1 > 1) ↔ (x > 1 / 2) := sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (f x a > x) ↔ (0 < a ∧ a ≤ 2) := sorry

end part1_solution_part2_solution_l57_57063


namespace arrows_from_530_to_535_l57_57566

def cyclic_arrows (n : Nat) : Nat :=
  n % 5

theorem arrows_from_530_to_535 : 
  cyclic_arrows 530 = 0 ∧ cyclic_arrows 531 = 1 ∧ cyclic_arrows 532 = 2 ∧
  cyclic_arrows 533 = 3 ∧ cyclic_arrows 534 = 4 ∧ cyclic_arrows 535 = 0 :=
by
  sorry

end arrows_from_530_to_535_l57_57566


namespace percentage_of_invalid_papers_l57_57127

theorem percentage_of_invalid_papers (total_papers : ℕ) (valid_papers : ℕ) (invalid_papers : ℕ) (percentage_invalid : ℚ) 
  (h1 : total_papers = 400) 
  (h2 : valid_papers = 240) 
  (h3 : invalid_papers = total_papers - valid_ppapers)
  (h4 : percentage_invalid = (invalid_papers : ℚ) / total_papers * 100) : 
  percentage_invalid = 40 :=
by
  sorry

end percentage_of_invalid_papers_l57_57127


namespace fraction_zero_implies_x_is_neg_2_l57_57861

theorem fraction_zero_implies_x_is_neg_2 {x : ℝ} 
  (h₁ : x^2 - 4 = 0)
  (h₂ : x^2 - 4 * x + 4 ≠ 0) 
  : x = -2 := 
by
  sorry

end fraction_zero_implies_x_is_neg_2_l57_57861


namespace min_time_one_ball_l57_57547

noncomputable def children_circle_min_time (n : ℕ) := 98

theorem min_time_one_ball (n : ℕ) (h1 : n = 99) : 
  children_circle_min_time n = 98 := 
by 
  sorry

end min_time_one_ball_l57_57547


namespace dvd_count_correct_l57_57158

def total_dvds (store_dvds online_dvds : Nat) : Nat :=
  store_dvds + online_dvds

theorem dvd_count_correct :
  total_dvds 8 2 = 10 :=
by
  sorry

end dvd_count_correct_l57_57158


namespace maximum_value_is_one_div_sqrt_two_l57_57656

noncomputable def maximum_value_2ab_root2_plus_2ac_plus_2bc (a b c : ℝ) : ℝ :=
  2 * a * b * Real.sqrt 2 + 2 * a * c + 2 * b * c

theorem maximum_value_is_one_div_sqrt_two (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h : a^2 + b^2 + c^2 = 1) :
  maximum_value_2ab_root2_plus_2ac_plus_2bc a b c ≤ 1 / Real.sqrt 2 :=
by
  sorry

end maximum_value_is_one_div_sqrt_two_l57_57656


namespace shared_total_l57_57275

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end shared_total_l57_57275


namespace y_completion_time_l57_57467

noncomputable def work_done (days : ℕ) (rate : ℚ) : ℚ := days * rate

theorem y_completion_time (X_days Y_remaining_days : ℕ) (X_rate Y_days : ℚ) :
  X_days = 40 →
  work_done 8 (1 / X_days) = 1 / 5 →
  work_done Y_remaining_days (4 / 5 / Y_remaining_days) = 4 / 5 →
  Y_days = 35 :=
by
  intros hX hX_work_done hY_work_done
  -- With the stated conditions, we should be able to conclude that Y_days is 35.
  sorry

end y_completion_time_l57_57467


namespace value_of_b_l57_57070

theorem value_of_b (b : ℚ) (h : b + b / 4 = 3) : b = 12 / 5 := by
  sorry

end value_of_b_l57_57070


namespace two_solutions_for_positive_integer_m_l57_57548

theorem two_solutions_for_positive_integer_m :
  ∃ k : ℕ, k = 2 ∧ (∀ m : ℕ, 0 < m → 990 % (m^2 - 2) = 0 → m = 2 ∨ m = 3) := 
sorry

end two_solutions_for_positive_integer_m_l57_57548


namespace overall_percentage_decrease_l57_57362

-- Define the initial pay cut percentages as given in the conditions.
def first_pay_cut := 5.25 / 100
def second_pay_cut := 9.75 / 100
def third_pay_cut := 14.6 / 100
def fourth_pay_cut := 12.8 / 100

-- Define the single shot percentage decrease we want to prove.
def single_shot_decrease := 36.73 / 100

-- Calculate the cumulative multiplier from individual pay cuts.
def cumulative_multiplier := 
  (1 - first_pay_cut) * (1 - second_pay_cut) * (1 - third_pay_cut) * (1 - fourth_pay_cut)

-- Statement: Prove the overall percentage decrease using cumulative multiplier is equal to single shot decrease.
theorem overall_percentage_decrease :
  1 - cumulative_multiplier = single_shot_decrease :=
by sorry

end overall_percentage_decrease_l57_57362


namespace bench_cost_150_l57_57713

-- Define the conditions
def combined_cost (bench_cost table_cost : ℕ) : Prop := bench_cost + table_cost = 450
def table_cost_eq_twice_bench (bench_cost table_cost : ℕ) : Prop := table_cost = 2 * bench_cost

-- Define the main statement, which includes the goal of the proof.
theorem bench_cost_150 (bench_cost table_cost : ℕ) (h_combined_cost : combined_cost bench_cost table_cost)
  (h_table_cost_eq_twice_bench : table_cost_eq_twice_bench bench_cost table_cost) : bench_cost = 150 :=
by
  sorry

end bench_cost_150_l57_57713


namespace p_necessary_not_sufficient_q_l57_57698

def condition_p (x : ℝ) : Prop := abs x ≤ 2
def condition_q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_not_sufficient_q (x : ℝ) :
  (condition_p x → condition_q x) = false ∧ (condition_q x → condition_p x) = true :=
by
  sorry

end p_necessary_not_sufficient_q_l57_57698


namespace equation_has_three_real_roots_l57_57685

noncomputable def f (x : ℝ) : ℝ := 2^x - x^2 - 1

theorem equation_has_three_real_roots : ∃! (x : ℝ), f x = 0 :=
by sorry

end equation_has_three_real_roots_l57_57685


namespace sum_of_fractions_l57_57586

theorem sum_of_fractions :
  (1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6) + 1 / (5 * 6 * 7) + 1 / (6 * 7 * 8)) = 3 / 16 := 
by
  sorry

end sum_of_fractions_l57_57586


namespace boxes_same_number_oranges_l57_57102

theorem boxes_same_number_oranges 
  (total_boxes : ℕ) (min_oranges : ℕ) (max_oranges : ℕ) 
  (boxes : ℕ) (range_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 130 →
  max_oranges = 160 →
  range_oranges = max_oranges - min_oranges + 1 →
  boxes = total_boxes / range_oranges →
  31 = range_oranges →
  4 ≤ boxes :=
by sorry

end boxes_same_number_oranges_l57_57102


namespace remainder_expression_mod_l57_57254

/-- 
Let the positive integers s, t, u, and v leave remainders of 6, 9, 13, and 17, respectively, 
when divided by 23. Also, let s > t > u > v.
We want to prove that the remainder when 2 * (s - t) - 3 * (t - u) + 4 * (u - v) is divided by 23 is 12.
-/
theorem remainder_expression_mod (s t u v : ℕ) (hs : s % 23 = 6) (ht : t % 23 = 9) (hu : u % 23 = 13) (hv : v % 23 = 17)
  (h_gt : s > t ∧ t > u ∧ u > v) : (2 * (s - t) - 3 * (t - u) + 4 * (u - v)) % 23 = 12 :=
by
  sorry

end remainder_expression_mod_l57_57254


namespace student_contribution_is_4_l57_57886

-- Definitions based on the conditions in the problem statement
def total_contribution := 90
def available_class_funds := 14
def number_of_students := 19

-- The theorem statement to be proven
theorem student_contribution_is_4 : 
  (total_contribution - available_class_funds) / number_of_students = 4 :=
by
  sorry  -- Proof is not required as per the instructions

end student_contribution_is_4_l57_57886


namespace largest_perimeter_polygons_meeting_at_A_l57_57474

theorem largest_perimeter_polygons_meeting_at_A
  (n : ℕ) 
  (r : ℝ)
  (h1 : n ≥ 3)
  (h2 : 2 * 180 * (n - 2) / n + 60 = 360) :
  2 * n * 2 = 24 := 
by
  sorry

end largest_perimeter_polygons_meeting_at_A_l57_57474


namespace problem1_proof_problem2_proof_l57_57672

noncomputable def problem1 : Real :=
  Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24

theorem problem1_proof : problem1 = 3 * Real.sqrt 6 :=
  sorry

noncomputable def problem2 : Real :=
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3)

theorem problem2_proof : problem2 = 6 :=
  sorry

end problem1_proof_problem2_proof_l57_57672


namespace a2_plus_a3_eq_40_l57_57067

theorem a2_plus_a3_eq_40 : 
  ∀ (a a1 a2 a3 a4 a5 : ℤ), 
  (2 * x - 1)^5 = a * x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5 → 
  a2 + a3 = 40 :=
by
  sorry

end a2_plus_a3_eq_40_l57_57067


namespace unique_root_in_interval_l57_57401

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 2

theorem unique_root_in_interval (n : ℤ) (h_root : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0) :
  n = 1 := 
sorry

end unique_root_in_interval_l57_57401


namespace decimal_to_fraction_correct_l57_57308

-- Define a structure representing our initial decimal to fraction conversion
structure DecimalFractionConversion :=
  (decimal: ℚ)
  (vulgar_fraction: ℚ)
  (simplified_fraction: ℚ)

-- Define the conditions provided in the problem
def conversion_conditions : DecimalFractionConversion :=
  { decimal := 35 / 100,
    vulgar_fraction := 35 / 100,
    simplified_fraction := 7 / 20 }

-- State the theorem we aim to prove
theorem decimal_to_fraction_correct :
  conversion_conditions.simplified_fraction = 7 / 20 := by
  sorry

end decimal_to_fraction_correct_l57_57308


namespace counterexample_to_prime_condition_l57_57495

theorem counterexample_to_prime_condition :
  ¬(Prime 54) ∧ ¬(Prime 52) ∧ ¬(Prime 51) := by
  -- Proof not required
  sorry

end counterexample_to_prime_condition_l57_57495


namespace closest_point_on_parabola_to_line_l57_57354

noncomputable def line := { P : ℝ × ℝ | 2 * P.1 - P.2 = 4 }
noncomputable def parabola := { P : ℝ × ℝ | P.2 = P.1^2 }

theorem closest_point_on_parabola_to_line : 
  ∃ P : ℝ × ℝ, P ∈ parabola ∧ 
  (∀ Q ∈ parabola, ∀ R ∈ line, dist P R ≤ dist Q R) ∧ 
  P = (1, 1) := 
sorry

end closest_point_on_parabola_to_line_l57_57354


namespace treasures_coins_count_l57_57597

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l57_57597


namespace no_integer_roots_of_polynomial_l57_57670

theorem no_integer_roots_of_polynomial :
  ¬ ∃ x : ℤ, x^3 - 4 * x^2 - 14 * x + 28 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l57_57670


namespace balls_in_rightmost_box_l57_57890

theorem balls_in_rightmost_box (a : ℕ → ℕ)
  (h₀ : a 1 = 7)
  (h₁ : ∀ i, 1 ≤ i ∧ i ≤ 1990 → a i + a (i + 1) + a (i + 2) + a (i + 3) = 30) :
  a 1993 = 7 :=
sorry

end balls_in_rightmost_box_l57_57890


namespace vasya_claim_false_l57_57288

theorem vasya_claim_false :
  ∀ (weights : List ℕ), weights = [1, 2, 3, 4, 5, 6, 7] →
  (¬ ∃ (subset : List ℕ), subset.length = 3 ∧ 1 ∈ subset ∧
  ((weights.sum - subset.sum) = 14) ∧ (14 = 14)) :=
by
  sorry

end vasya_claim_false_l57_57288


namespace unique_solution_of_fraction_eq_l57_57271

theorem unique_solution_of_fraction_eq (x : ℝ) : (1 / (x - 1) = 2 / (x - 2)) ↔ (x = 0) :=
by
  sorry

end unique_solution_of_fraction_eq_l57_57271


namespace edward_dunk_a_clown_tickets_l57_57599

-- Definitions for conditions
def total_tickets : ℕ := 79
def rides : ℕ := 8
def tickets_per_ride : ℕ := 7

-- Theorem statement
theorem edward_dunk_a_clown_tickets :
  let tickets_spent_on_rides := rides * tickets_per_ride
  let tickets_remaining := total_tickets - tickets_spent_on_rides
  tickets_remaining = 23 :=
by
  sorry

end edward_dunk_a_clown_tickets_l57_57599


namespace combined_weight_l57_57747

-- We define the variables and the conditions
variables (x y : ℝ)

-- First condition 
def condition1 : Prop := y = (16 - 4) + (30 - 6) + (x - 3)

-- Second condition
def condition2 : Prop := y = 12 + 24 + (x - 3)

-- The statement to prove
theorem combined_weight (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : y = x + 33 :=
by
  -- Skipping the proof part
  sorry

end combined_weight_l57_57747


namespace range_of_a_for_function_min_max_l57_57437

theorem range_of_a_for_function_min_max 
  (a : ℝ) 
  (h_min : ∀ x ∈ [-1, 1], x = -1 → x^2 + a * x + 3 ≤ y) 
  (h_max : ∀ x ∈ [-1, 1], x = 1 → x^2 + a * x + 3 ≥ y) : 
  2 ≤ a := 
sorry

end range_of_a_for_function_min_max_l57_57437


namespace vacation_cost_eq_l57_57650

theorem vacation_cost_eq (C : ℕ) (h : C / 3 - C / 5 = 50) : C = 375 :=
sorry

end vacation_cost_eq_l57_57650


namespace dilation_0_minus_2i_to_neg3_minus_14i_l57_57497

open Complex

def dilation_centered (z_center z zk : ℂ) (factor : ℝ) : ℂ :=
  z_center + factor * (zk - z_center)

theorem dilation_0_minus_2i_to_neg3_minus_14i :
  dilation_centered (1 + 2 * I) (0 - 2 * I) (1 + 2 * I) 4 = -3 - 14 * I :=
by
  sorry

end dilation_0_minus_2i_to_neg3_minus_14i_l57_57497


namespace amelia_drove_tuesday_l57_57524

-- Define the known quantities
def total_distance : ℕ := 8205
def distance_monday : ℕ := 907
def remaining_distance : ℕ := 6716

-- Define the distance driven on Tuesday and state the theorem
def distance_tuesday : ℕ := total_distance - (distance_monday + remaining_distance)

-- Theorem stating the distance driven on Tuesday is 582 kilometers
theorem amelia_drove_tuesday : distance_tuesday = 582 := 
by
  -- We skip the proof for now
  sorry

end amelia_drove_tuesday_l57_57524


namespace sum_super_cool_rectangle_areas_eq_84_l57_57781

theorem sum_super_cool_rectangle_areas_eq_84 :
  ∀ (a b : ℕ), 
  (a * b = 3 * (a + b)) → 
  ∃ (S : ℕ), 
  S = 84 :=
by
  sorry

end sum_super_cool_rectangle_areas_eq_84_l57_57781


namespace chris_money_before_birthday_l57_57391

-- Define the given amounts of money from each source
def money_from_grandmother : ℕ := 25
def money_from_aunt_and_uncle : ℕ := 20
def money_from_parents : ℕ := 75
def total_money_now : ℕ := 279

-- Calculate the total birthday money
def total_birthday_money := money_from_grandmother + money_from_aunt_and_uncle + money_from_parents

-- Define the amount of money Chris had before his birthday
def money_before_birthday := total_money_now - total_birthday_money

-- The proof statement
theorem chris_money_before_birthday : money_before_birthday = 159 :=
by
  sorry

end chris_money_before_birthday_l57_57391


namespace problem_solution_l57_57619

noncomputable def length_segment_AB : ℝ :=
  let k : ℝ := 1 -- derived from 3k - 3 = 0
  let A : ℝ × ℝ := (0, k) -- point (0, k)
  let C : ℝ × ℝ := (3, -1) -- center of the circle
  let r : ℝ := 1 -- radius of the circle
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -- distance formula
  Real.sqrt (AC^2 - r^2)

theorem problem_solution :
  length_segment_AB = 2 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l57_57619


namespace train_distance_proof_l57_57828

theorem train_distance_proof (c₁ c₂ c₃ : ℝ) : 
  (5 / c₁ + 5 / c₂ = 15) →
  (5 / c₂ + 5 / c₃ = 11) →
  ∀ (x : ℝ), (x / c₁ = 10 / c₂ + (10 + x) / c₃) →
  x = 27.5 := 
by
  sorry

end train_distance_proof_l57_57828


namespace acute_angles_theorem_l57_57900

open Real

variable (α β : ℝ)

-- Given conditions
def conditions : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  tan α = 1 / 7 ∧
  sin β = sqrt 10 / 10

-- Proof goal
def proof_goal : Prop :=
  α + 2 * β = π / 4

-- The final theorem
theorem acute_angles_theorem (h : conditions α β) : proof_goal α β :=
  sorry

end acute_angles_theorem_l57_57900


namespace total_chickens_l57_57810

-- Definitions from conditions
def ducks : ℕ := 40
def rabbits : ℕ := 30
def hens : ℕ := ducks + 20
def roosters : ℕ := rabbits - 10

-- Theorem statement: total number of chickens
theorem total_chickens : hens + roosters = 80 := 
sorry

end total_chickens_l57_57810


namespace murtha_pebbles_after_20_days_l57_57120

/- Define the sequence function for the pebbles collected each day -/
def pebbles_collected_day (n : ℕ) : ℕ :=
  if (n = 0) then 0 else 1 + pebbles_collected_day (n - 1)

/- Define the total pebbles collected by the nth day -/
def total_pebbles_collected (n : ℕ) : ℕ :=
  (n * (pebbles_collected_day n)) / 2

/- Define the total pebbles given away by the nth day -/
def pebbles_given_away (n : ℕ) : ℕ :=
  (n / 5) * 3

/- Define the net total of pebbles Murtha has on the nth day -/
def pebbles_net (n : ℕ) : ℕ :=
  total_pebbles_collected (n + 1) - pebbles_given_away (n + 1)

/- The main theorem about the pebbles Murtha has after the 20th day -/
theorem murtha_pebbles_after_20_days : pebbles_net 19 = 218 := 
  by sorry

end murtha_pebbles_after_20_days_l57_57120


namespace probability_of_at_least_one_l57_57146

theorem probability_of_at_least_one (P_1 P_2 : ℝ) (h1 : 0 ≤ P_1 ∧ P_1 ≤ 1) (h2 : 0 ≤ P_2 ∧ P_2 ≤ 1) :
  1 - (1 - P_1) * (1 - P_2) = P_1 + P_2 - P_1 * P_2 :=
by
  sorry

end probability_of_at_least_one_l57_57146


namespace count_obtuse_triangle_values_k_l57_57150

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  if a ≥ b ∧ a ≥ c then a * a > b * b + c * c 
  else if b ≥ a ∧ b ≥ c then b * b > a * a + c * c
  else c * c > a * a + b * b

theorem count_obtuse_triangle_values_k :
  ∃! (k : ℕ), is_triangle 8 18 k ∧ is_obtuse_triangle 8 18 k :=
sorry

end count_obtuse_triangle_values_k_l57_57150


namespace find_length_of_DE_l57_57771

-- Define the setup: five points A, B, C, D, E on a circle
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Define the given distances 
def AB : ℝ := 7
def BC : ℝ := 7
def AD : ℝ := 10

-- Define the total distance AC
def AC : ℝ := AB + BC

-- Define the length DE to be solved
def DE : ℝ := 0.2

-- State the theorem to be proved given the conditions
theorem find_length_of_DE : 
  DE = 0.2 :=
sorry

end find_length_of_DE_l57_57771


namespace greatest_4_digit_number_l57_57225

theorem greatest_4_digit_number
  (n : ℕ)
  (h1 : n % 5 = 3)
  (h2 : n % 9 = 2)
  (h3 : 1000 ≤ n)
  (h4 : n < 10000) :
  n = 9962 := 
sorry

end greatest_4_digit_number_l57_57225


namespace minimum_flower_cost_l57_57274

def vertical_strip_width : ℝ := 3
def horizontal_strip_height : ℝ := 2
def bed_width : ℝ := 11
def bed_height : ℝ := 6

def easter_lily_cost : ℝ := 3
def dahlia_cost : ℝ := 2.5
def canna_cost : ℝ := 2

def vertical_strip_area : ℝ := vertical_strip_width * bed_height
def horizontal_strip_area : ℝ := horizontal_strip_height * bed_width
def overlap_area : ℝ := vertical_strip_width * horizontal_strip_height
def remaining_area : ℝ := (bed_width * bed_height) - vertical_strip_area - (horizontal_strip_area - overlap_area)

def easter_lily_area : ℝ := horizontal_strip_area - overlap_area
def dahlia_area : ℝ := vertical_strip_area
def canna_area : ℝ := remaining_area

def easter_lily_total_cost : ℝ := easter_lily_area * easter_lily_cost
def dahlia_total_cost : ℝ := dahlia_area * dahlia_cost
def canna_total_cost : ℝ := canna_area * canna_cost

def total_cost : ℝ := easter_lily_total_cost + dahlia_total_cost + canna_total_cost

theorem minimum_flower_cost : total_cost = 157 := by
  sorry

end minimum_flower_cost_l57_57274


namespace wheel_distance_travelled_l57_57144

noncomputable def radius : ℝ := 3
noncomputable def num_revolutions : ℝ := 3
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def total_distance (r : ℝ) (n : ℝ) : ℝ := n * circumference r

theorem wheel_distance_travelled :
  total_distance radius num_revolutions = 18 * Real.pi :=
by 
  sorry

end wheel_distance_travelled_l57_57144


namespace find_d_and_a11_l57_57795

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_d_and_a11 (a : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence a d →
  a 5 = 6 →
  a 8 = 15 →
  d = 3 ∧ a 11 = 24 :=
by
  intros h_seq h_a5 h_a8
  sorry

end find_d_and_a11_l57_57795


namespace positive_integers_satisfy_eq_l57_57623

theorem positive_integers_satisfy_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + 1 = c! → (a = 2 ∧ b = 1 ∧ c = 3) ∨ (a = 1 ∧ b = 2 ∧ c = 3) :=
by sorry

end positive_integers_satisfy_eq_l57_57623


namespace not_perfect_square_n_l57_57622

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, m * m = x

theorem not_perfect_square_n (n : ℕ) : ¬ isPerfectSquare (4 * n^2 + 4 * n + 4) :=
sorry

end not_perfect_square_n_l57_57622


namespace num_people_in_group_l57_57213

-- Define constants and conditions
def cost_per_set : ℕ := 3  -- $3 to make 4 S'mores
def smores_per_set : ℕ := 4
def total_cost : ℕ := 18   -- $18 total cost
def smores_per_person : ℕ := 3

-- Calculate total S'mores that can be made
def total_sets : ℕ := total_cost / cost_per_set
def total_smores : ℕ := total_sets * smores_per_set

-- Proof problem statement
theorem num_people_in_group : (total_smores / smores_per_person) = 8 :=
by
  sorry

end num_people_in_group_l57_57213


namespace max_value_of_8q_minus_9p_is_zero_l57_57406

theorem max_value_of_8q_minus_9p_is_zero (p : ℝ) (q : ℝ) (h1 : 0 < p) (h2 : p < 1) (hq : q = 3 * p ^ 2 - 2 * p ^ 3) : 
  8 * q - 9 * p ≤ 0 :=
by
  sorry

end max_value_of_8q_minus_9p_is_zero_l57_57406


namespace find_xyz_sum_l57_57018

variables {x y z : ℝ}

def system_of_equations (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + x * y + y^2 = 12) ∧
  (y^2 + y * z + z^2 = 9) ∧
  (z^2 + z * x + x^2 = 21)

theorem find_xyz_sum (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 12 :=
sorry

end find_xyz_sum_l57_57018


namespace commute_weeks_per_month_l57_57016

variable (total_commute_one_way : ℕ)
variable (gas_cost_per_gallon : ℝ)
variable (car_mileage : ℝ)
variable (commute_days_per_week : ℕ)
variable (individual_monthly_payment : ℝ)
variable (number_of_people : ℕ)

theorem commute_weeks_per_month :
  total_commute_one_way = 21 →
  gas_cost_per_gallon = 2.5 →
  car_mileage = 30 →
  commute_days_per_week = 5 →
  individual_monthly_payment = 14 →
  number_of_people = 5 →
  (individual_monthly_payment * number_of_people) / 
  ((total_commute_one_way * 2 / car_mileage) * gas_cost_per_gallon * commute_days_per_week) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end commute_weeks_per_month_l57_57016


namespace sum_geometric_sequence_l57_57985

theorem sum_geometric_sequence {n : ℕ} (S : ℕ → ℝ) (h1 : S n = 10) (h2 : S (2 * n) = 30) : 
  S (3 * n) = 70 := 
by 
  sorry

end sum_geometric_sequence_l57_57985


namespace supermarket_sold_54_pints_l57_57736

theorem supermarket_sold_54_pints (x s : ℝ) 
  (h1 : x * s = 216)
  (h2 : x * (s + 2) = 324) : 
  x = 54 := 
by 
  sorry

end supermarket_sold_54_pints_l57_57736


namespace area_of_trapezium_l57_57013

-- Definitions based on conditions
def length_parallel_side1 : ℝ := 20 -- length of the first parallel side
def length_parallel_side2 : ℝ := 18 -- length of the second parallel side
def distance_between_sides : ℝ := 5 -- distance between the parallel sides

-- Statement to prove
theorem area_of_trapezium (a b h : ℝ) :
  a = length_parallel_side1 → b = length_parallel_side2 → h = distance_between_sides →
  (a + b) * h / 2 = 95 :=
by
  intros ha hb hh
  rw [ha, hb, hh]
  sorry

end area_of_trapezium_l57_57013


namespace find_values_of_a_b_solve_inequality_l57_57863

variable (a b : ℝ)
variable (h1 : ∀ x : ℝ, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 2)

theorem find_values_of_a_b (h2 : a = -2) (h3 : b = 3) : 
  a = -2 ∧ b = 3 :=
by
  constructor
  exact h2
  exact h3


theorem solve_inequality 
  (h2 : a = -2) (h3 : b = 3) :
  ∀ x : ℝ, (a * x^2 + b * x - 1 > 0) ↔ (1/2 < x ∧ x < 1) :=
by
  sorry

end find_values_of_a_b_solve_inequality_l57_57863


namespace liza_final_balance_l57_57321

theorem liza_final_balance :
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries (balance : ℝ) := 0.2 * balance
  let friday_interest (balance : ℝ) := 0.02 * balance
  let saturday_phone_bill := 70
  let saturday_additional_deposit := 300
  let tuesday_balance := monday_balance - tuesday_rent
  let wednesday_balance := tuesday_balance + wednesday_deposit
  let thursday_balance_before_groceries := wednesday_balance - thursday_electric_bill - thursday_internet_bill
  let thursday_balance_after_groceries := thursday_balance_before_groceries - thursday_groceries thursday_balance_before_groceries
  let friday_balance := thursday_balance_after_groceries + friday_interest thursday_balance_after_groceries
  let saturday_balance_after_phone := friday_balance - saturday_phone_bill
  let final_balance := saturday_balance_after_phone + saturday_additional_deposit
  final_balance = 1562.528 :=
by
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries := 0.2 * (800 - 450 + 1500 - 117 - 100)
  let friday_interest := 0.02 * (800 - 450 + 1500 - 117 - 100 - 0.2 * (800 - 450 + 1500 - 117 - 100))
  let final_balance := 800 - 450 + 1500 - 117 - 100 - thursday_groceries + friday_interest - 70 + 300
  sorry

end liza_final_balance_l57_57321


namespace quadruple_solution_l57_57415

theorem quadruple_solution (x y z w : ℝ) (h1: x + y + z + w = 0) (h2: x^7 + y^7 + z^7 + w^7 = 0) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨ (x = -y ∧ z = -w) ∨ (x = -z ∧ y = -w) ∨ (x = -w ∧ y = -z) :=
by
  sorry

end quadruple_solution_l57_57415


namespace total_amount_withdrawn_l57_57022

def principal : ℤ := 20000
def interest_rate : ℚ := 3.33 / 100
def term : ℤ := 3

theorem total_amount_withdrawn :
  principal + (principal * interest_rate * term) = 21998 := by
  sorry

end total_amount_withdrawn_l57_57022


namespace series_remainder_is_zero_l57_57563

theorem series_remainder_is_zero :
  let a : ℕ := 4
  let d : ℕ := 6
  let n : ℕ := 17
  let l : ℕ := a + d * (n - 1) -- last term
  let S : ℕ := n * (a + l) / 2 -- sum of the series
  S % 17 = 0 := by
  sorry

end series_remainder_is_zero_l57_57563


namespace find_q_l57_57456

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by
  sorry

end find_q_l57_57456


namespace polynomial_division_l57_57192

variable (x : ℝ)

theorem polynomial_division :
  ((3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 - x^2 + 6 * x - 8)) / (x + 1) 
  = (x^2 + 4 * x - 15 + 25 / (x+1)) :=
by sorry

end polynomial_division_l57_57192


namespace min_folds_to_exceed_thickness_l57_57030

def initial_thickness : ℝ := 0.1
def desired_thickness : ℝ := 12

theorem min_folds_to_exceed_thickness : ∃ (n : ℕ), initial_thickness * 2^n > desired_thickness ∧ ∀ m < n, initial_thickness * 2^m ≤ desired_thickness := by
  sorry

end min_folds_to_exceed_thickness_l57_57030


namespace expected_number_of_socks_l57_57438

noncomputable def expected_socks_to_pick (n : ℕ) : ℚ := (2 * (n + 1)) / 3

theorem expected_number_of_socks (n : ℕ) (h : n ≥ 2) : 
  (expected_socks_to_pick n) = (2 * (n + 1)) / 3 := 
by
  sorry

end expected_number_of_socks_l57_57438


namespace snail_crawl_distance_l57_57963

theorem snail_crawl_distance
  (α : ℕ → ℝ)  -- α represents the snail's position at each minute
  (crawls_forward : ∀ n m : ℕ, n < m → α n ≤ α m)  -- The snail moves forward (without going backward)
  (observer_finds : ∀ n : ℕ, α (n + 1) - α n = 1) -- Every observer finds that the snail crawled exactly 1 meter per minute
  (time_span : ℕ := 6)  -- Total observation period is 6 minutes
  : α time_span - α 0 ≤ 10 :=  -- The distance crawled in 6 minutes does not exceed 10 meters
by
  -- Proof goes here
  sorry

end snail_crawl_distance_l57_57963


namespace remainder_of_98_mul_102_mod_9_l57_57721

theorem remainder_of_98_mul_102_mod_9 : (98 * 102) % 9 = 6 := 
by 
  -- Introducing the variables and arithmetic
  let x := 98 * 102 
  have h1 : x = 9996 := 
    by norm_num
  have h2 : x % 9 = 6 := 
    by norm_num
  -- Result
  exact h2

end remainder_of_98_mul_102_mod_9_l57_57721


namespace third_place_amount_l57_57605

noncomputable def total_people : ℕ := 13
noncomputable def money_per_person : ℝ := 5
noncomputable def total_money : ℝ := total_people * money_per_person

noncomputable def first_place_percentage : ℝ := 0.65
noncomputable def second_third_place_percentage : ℝ := 0.35
noncomputable def split_factor : ℝ := 0.5

noncomputable def first_place_money : ℝ := first_place_percentage * total_money
noncomputable def second_third_place_money : ℝ := second_third_place_percentage * total_money
noncomputable def third_place_money : ℝ := split_factor * second_third_place_money

theorem third_place_amount : third_place_money = 11.38 := by
  sorry

end third_place_amount_l57_57605


namespace pencil_count_l57_57079

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end pencil_count_l57_57079


namespace work_days_together_l57_57449

theorem work_days_together (p_rate q_rate : ℝ) (fraction_left : ℝ) (d : ℝ) 
  (h₁ : p_rate = 1/15) (h₂ : q_rate = 1/20) (h₃ : fraction_left = 8/15)
  (h₄ : (p_rate + q_rate) * d = 1 - fraction_left) : d = 4 :=
by
  sorry

end work_days_together_l57_57449


namespace inexperienced_sailors_count_l57_57873

theorem inexperienced_sailors_count
  (I E : ℕ)
  (h1 : I + E = 17)
  (h2 : ∀ (rate_inexperienced hourly_rate experienced_rate : ℕ), hourly_rate = 10 → experienced_rate = 12 → rate_inexperienced = 2400)
  (h3 : ∀ (total_income experienced_salary : ℕ), total_income = 34560 → experienced_salary = 2880)
  (h4 : ∀ (monthly_income : ℕ), monthly_income = 34560)
  : I = 5 := sorry

end inexperienced_sailors_count_l57_57873


namespace geometric_sequence_a5_l57_57062

variable {a : Nat → ℝ} {q : ℝ}

-- Conditions
def is_geometric_sequence (a : Nat → ℝ) (q : ℝ) :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = q * a n

def condition_eq (a : Nat → ℝ) :=
  a 5 + a 4 = 3 * (a 3 + a 2)

-- Proof statement
theorem geometric_sequence_a5 (hq : q ≠ -1)
  (hg : is_geometric_sequence a q)
  (hc : condition_eq a) : a 5 = 9 :=
  sorry

end geometric_sequence_a5_l57_57062


namespace zero_sum_of_squares_l57_57284

theorem zero_sum_of_squares {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_l57_57284


namespace prove_expression_l57_57626

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 5)

lemma root_of_unity : omega^5 = 1 := sorry
lemma sum_of_roots : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sorry

noncomputable def z := omega + omega^2 + omega^3 + omega^4

theorem prove_expression : z^2 + z + 1 = 1 :=
by 
  have h1 : omega^5 = 1 := root_of_unity
  have h2 : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sum_of_roots
  show z^2 + z + 1 = 1
  {
    -- Proof omitted
    sorry
  }

end prove_expression_l57_57626


namespace john_behind_steve_l57_57773

theorem john_behind_steve
  (vJ : ℝ) (vS : ℝ) (ahead : ℝ) (t : ℝ) (d : ℝ)
  (hJ : vJ = 4.2) (hS : vS = 3.8) (hA : ahead = 2) (hT : t = 42.5)
  (h1 : vJ * t = d + ahead)
  (h2 : vS * t + ahead = vJ * t - ahead) :
  d = 15 :=
by
  -- Proof omitted
  sorry

end john_behind_steve_l57_57773


namespace average_speed_without_stoppages_l57_57235

variables (d : ℝ) (t : ℝ) (v_no_stop : ℝ)

-- The train stops for 12 minutes per hour
def stoppage_per_hour := 12 / 60
def moving_fraction := 1 - stoppage_per_hour

-- Given speed with stoppages is 160 km/h
def speed_with_stoppage := 160

-- Average speed of the train without stoppages
def speed_without_stoppage := speed_with_stoppage / moving_fraction

-- The average speed without stoppages should equal 200 km/h
theorem average_speed_without_stoppages : speed_without_stoppage = 200 :=
by
  unfold speed_without_stoppage
  unfold moving_fraction
  unfold stoppage_per_hour
  norm_num
  sorry

end average_speed_without_stoppages_l57_57235


namespace min_value_expression_l57_57666

theorem min_value_expression (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 14) ∧ (∀ z : ℝ, (z = (x + 10) / Real.sqrt (x - 4)) → y ≤ z) := sorry

end min_value_expression_l57_57666


namespace smallest_possible_AAB_l57_57518

-- Definitions of the digits A and B
def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

-- Definition of the condition AB equals 1/7 of AAB
def condition (A B : ℕ) : Prop := 10 * A + B = (1 / 7) * (110 * A + B)

theorem smallest_possible_AAB (A B : ℕ) : is_valid_digit A ∧ is_valid_digit B ∧ condition A B → 110 * A + B = 664 := sorry

end smallest_possible_AAB_l57_57518


namespace gcd_80_180_450_l57_57786

theorem gcd_80_180_450 : Int.gcd (Int.gcd 80 180) 450 = 10 := by
  sorry

end gcd_80_180_450_l57_57786


namespace lowest_possible_score_l57_57027

def total_points_first_four_tests : ℕ := 82 + 90 + 78 + 85
def required_total_points_for_seven_tests : ℕ := 80 * 7
def points_needed_for_last_three_tests : ℕ :=
  required_total_points_for_seven_tests - total_points_first_four_tests

theorem lowest_possible_score 
  (max_points_per_test : ℕ)
  (points_first_four_tests : ℕ := total_points_first_four_tests)
  (required_points : ℕ := required_total_points_for_seven_tests)
  (total_points_needed_last_three : ℕ := points_needed_for_last_three_tests) :
  ∃ (lowest_score : ℕ), 
    max_points_per_test = 100 ∧
    points_first_four_tests = 335 ∧
    required_points = 560 ∧
    total_points_needed_last_three = 225 ∧
    lowest_score = 25 :=
by
  sorry

end lowest_possible_score_l57_57027


namespace discount_is_5_percent_l57_57951

-- Defining the conditions
def cost_per_iphone : ℕ := 600
def total_cost_3_iphones : ℕ := 3 * cost_per_iphone
def savings : ℕ := 90

-- Calculating the discount percentage
def discount_percentage : ℕ := (savings * 100) / total_cost_3_iphones

-- Stating the theorem
theorem discount_is_5_percent : discount_percentage = 5 :=
  sorry

end discount_is_5_percent_l57_57951


namespace gcd_m_n_l57_57319

def m := 122^2 + 234^2 + 346^2 + 458^2
def n := 121^2 + 233^2 + 345^2 + 457^2

theorem gcd_m_n : Int.gcd m n = 1 := 
by sorry

end gcd_m_n_l57_57319


namespace total_dots_l57_57045

def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

theorem total_dots :
  (ladybugs_monday + ladybugs_tuesday) * dots_per_ladybug = 78 :=
by
  sorry

end total_dots_l57_57045


namespace joey_speed_on_way_back_eq_six_l57_57505

theorem joey_speed_on_way_back_eq_six :
  ∃ (v : ℝ), 
    (∀ (d t : ℝ), 
      d = 2 ∧ t = 1 →  -- Joey runs a 2-mile distance in 1 hour
      (∀ (d_total t_avg : ℝ),
        d_total = 4 ∧ t_avg = 3 →  -- Round trip distance is 4 miles with average speed 3 mph
        (3 = 4 / (1 + 2 / v) → -- Given average speed equation
         v = 6))) := sorry

end joey_speed_on_way_back_eq_six_l57_57505


namespace problem_statement_l57_57638

noncomputable def a : ℝ := (Real.tan 23) / (1 - (Real.tan 23) ^ 2)
noncomputable def b : ℝ := 2 * Real.sin 13 * Real.cos 13
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos 50) / 2)

theorem problem_statement : c < b ∧ b < a :=
by
  -- Proof omitted
  sorry

end problem_statement_l57_57638


namespace box_volume_l57_57829

theorem box_volume (a b c : ℝ) (H1 : a * b = 15) (H2 : b * c = 10) (H3 : c * a = 6) : a * b * c = 30 := 
sorry

end box_volume_l57_57829


namespace original_number_of_men_l57_57839

theorem original_number_of_men (M : ℤ) (h1 : 8 * M = 5 * (M + 10)) : M = 17 := by
  -- Proof goes here
  sorry

end original_number_of_men_l57_57839


namespace solution_set_f_x_leq_m_solution_set_inequality_a_2_l57_57644

-- Part (I)
theorem solution_set_f_x_leq_m (a m : ℝ) (h : ∀ x : ℝ, |x - a| ≤ m ↔ -1 ≤ x ∧ x ≤ 5) :
  a = 2 ∧ m = 3 :=
sorry

-- Part (II)
theorem solution_set_inequality_a_2 (t : ℝ) (h_t : t ≥ 0) :
  (∀ x : ℝ, |x - 2| + t ≥ |x + 2 * t - 2| ↔ t = 0 ∧ (∀ x : ℝ, True) ∨ t > 0 ∧ ∀ x : ℝ, x ≤ 2 - t / 2) :=
sorry

end solution_set_f_x_leq_m_solution_set_inequality_a_2_l57_57644


namespace smallest_n_l57_57642

theorem smallest_n (n : ℕ) (h : 10 - n ≥ 0) : 
  (9 / 10) * (8 / 9) * (7 / 8) * (6 / 7) * (5 / 6) * (4 / 5) < 0.5 → n = 6 :=
by
  sorry

end smallest_n_l57_57642


namespace solution_set_inequality_l57_57247

theorem solution_set_inequality (a c : ℝ)
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2)) :
  (∀ x : ℝ, (cx^2 - 2*x + a ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3)) :=
sorry

end solution_set_inequality_l57_57247


namespace winning_candidate_percentage_l57_57533

theorem winning_candidate_percentage (total_membership: ℕ)
  (votes_cast: ℕ) (winning_percentage: ℝ) (h1: total_membership = 1600)
  (h2: votes_cast = 525) (h3: winning_percentage = 19.6875)
  : (winning_percentage / 100 * total_membership / votes_cast * 100 = 60) :=
by
  sorry

end winning_candidate_percentage_l57_57533


namespace find_missing_coordinates_l57_57368

def parallelogram_area (A B : ℝ × ℝ) (C D : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (D.2 - A.2))

theorem find_missing_coordinates :
  ∃ (x y : ℝ), (x, y) ≠ (4, 4) ∧ (x, y) ≠ (5, 9) ∧ (x, y) ≠ (8, 9) ∧
  parallelogram_area (4, 4) (5, 9) (8, 9) (x, y) = 5 :=
sorry

end find_missing_coordinates_l57_57368


namespace perfect_square_trinomial_m_l57_57119

theorem perfect_square_trinomial_m (m : ℝ) :
  (∀ x : ℝ, ∃ b : ℝ, x^2 + 2 * (m - 3) * x + 16 = (1 * x + b)^2) → (m = 7 ∨ m = -1) :=
by 
  intro h
  sorry

end perfect_square_trinomial_m_l57_57119


namespace tan_x_y_l57_57353

theorem tan_x_y (x y : ℝ) (h : Real.sin (2 * x + y) = 5 * Real.sin y) :
  Real.tan (x + y) = (3 / 2) * Real.tan x :=
sorry

end tan_x_y_l57_57353


namespace hourly_wage_12_5_l57_57716

theorem hourly_wage_12_5 
  (H : ℝ)
  (work_hours : ℝ := 40)
  (widgets_per_week : ℝ := 1000)
  (widget_earnings_per_widget : ℝ := 0.16)
  (total_earnings : ℝ := 660) :
  (40 * H + 1000 * 0.16 = 660) → (H = 12.5) :=
by
  sorry

end hourly_wage_12_5_l57_57716


namespace exists_subset_sum_2n_l57_57535

theorem exists_subset_sum_2n (n : ℕ) (h : n > 3) (s : Finset ℕ)
  (hs : ∀ x ∈ s, x < 2 * n) (hs_card : s.card = 2 * n)
  (hs_sum : s.sum id = 4 * n) :
  ∃ t ⊆ s, t.sum id = 2 * n :=
by sorry

end exists_subset_sum_2n_l57_57535


namespace seq_prime_l57_57560

/-- A strictly increasing sequence of positive integers. -/
def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

/-- An infinite strictly increasing sequence of positive integers. -/
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a n ∧ is_strictly_increasing a

/-- A sequence of distinct primes. -/
def distinct_primes (p : ℕ → ℕ) : Prop :=
  ∀ m n, m ≠ n → p m ≠ p n ∧ Nat.Prime (p n)

/-- The main theorem to be proved. -/
theorem seq_prime (a p : ℕ → ℕ) (h1 : strictly_increasing_sequence a) (h2 : distinct_primes p)
  (h3 : ∀ n, p n ∣ a n) (h4 : ∀ n k, a n - a k = p n - p k) : ∀ n, Nat.Prime (a n) := 
by
  sorry

end seq_prime_l57_57560


namespace ram_efficiency_eq_27_l57_57325

theorem ram_efficiency_eq_27 (R : ℕ) (h1 : ∀ Krish, 2 * (1 / (R : ℝ)) = 1 / Krish) 
  (h2 : ∀ s, 3 * (1 / (R : ℝ)) * s = 1 ↔ s = (9 : ℝ)) : R = 27 :=
sorry

end ram_efficiency_eq_27_l57_57325


namespace remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l57_57768

section Doughnuts

variable (initial_glazed : Nat := 10)
variable (initial_chocolate : Nat := 8)
variable (initial_raspberry : Nat := 6)

variable (personA_glazed : Nat := 2)
variable (personA_chocolate : Nat := 1)
variable (personB_glazed : Nat := 1)
variable (personC_chocolate : Nat := 3)
variable (personD_glazed : Nat := 1)
variable (personD_raspberry : Nat := 1)
variable (personE_raspberry : Nat := 1)
variable (personF_raspberry : Nat := 2)

def remaining_glazed : Nat :=
  initial_glazed - (personA_glazed + personB_glazed + personD_glazed)

def remaining_chocolate : Nat :=
  initial_chocolate - (personA_chocolate + personC_chocolate)

def remaining_raspberry : Nat :=
  initial_raspberry - (personD_raspberry + personE_raspberry + personF_raspberry)

theorem remaining_glazed_correct :
  remaining_glazed initial_glazed personA_glazed personB_glazed personD_glazed = 6 :=
by
  sorry

theorem remaining_chocolate_correct :
  remaining_chocolate initial_chocolate personA_chocolate personC_chocolate = 4 :=
by
  sorry

theorem remaining_raspberry_correct :
  remaining_raspberry initial_raspberry personD_raspberry personE_raspberry personF_raspberry = 2 :=
by
  sorry

end Doughnuts

end remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l57_57768


namespace cannot_form_62_cents_with_six_coins_l57_57823

-- Define the coin denominations and their values
structure Coin :=
  (value : ℕ)
  (count : ℕ)

def penny : Coin := ⟨1, 6⟩
def nickel : Coin := ⟨5, 6⟩
def dime : Coin := ⟨10, 6⟩
def quarter : Coin := ⟨25, 6⟩
def halfDollar : Coin := ⟨50, 6⟩

-- Define the main theorem statement
theorem cannot_form_62_cents_with_six_coins :
  ¬ (∃ (p n d q h : ℕ),
      p + n + d + q + h = 6 ∧
      1 * p + 5 * n + 10 * d + 25 * q + 50 * h = 62) :=
sorry

end cannot_form_62_cents_with_six_coins_l57_57823


namespace degrees_to_radians_l57_57755

theorem degrees_to_radians (π_radians : ℝ) : 150 * π_radians / 180 = 5 * π_radians / 6 :=
by sorry

end degrees_to_radians_l57_57755


namespace inverse_of_k_l57_57631

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end inverse_of_k_l57_57631


namespace cistern_emptying_l57_57181

theorem cistern_emptying (h: (3 / 4) / 12 = 1 / 16) : (8 * (1 / 16) = 1 / 2) :=
by sorry

end cistern_emptying_l57_57181


namespace equation_of_line_passing_through_and_parallel_l57_57974

theorem equation_of_line_passing_through_and_parallel :
  ∀ (x y : ℝ), (x = -3 ∧ y = -1) → (∃ (C : ℝ), x - 2 * y + C = 0) → C = 1 :=
by
  intros x y h₁ h₂
  sorry

end equation_of_line_passing_through_and_parallel_l57_57974


namespace sharona_bought_more_pencils_l57_57245

-- Define constants for the amounts paid
def amount_paid_jamar : ℚ := 1.43
def amount_paid_sharona : ℚ := 1.87

-- Define the function that computes the number of pencils given the price per pencil and total amount paid
def num_pencils (amount_paid : ℚ) (price_per_pencil : ℚ) : ℚ := amount_paid / price_per_pencil

-- Define the theorem stating that Sharona bought 4 more pencils than Jamar
theorem sharona_bought_more_pencils {price_per_pencil : ℚ} (h_price : price_per_pencil > 0) :
  num_pencils amount_paid_sharona price_per_pencil = num_pencils amount_paid_jamar price_per_pencil + 4 :=
sorry

end sharona_bought_more_pencils_l57_57245


namespace solution_set_of_inequality_l57_57239

theorem solution_set_of_inequality (x : ℝ) : |5 * x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_of_inequality_l57_57239


namespace min_value_of_xy_l57_57652

theorem min_value_of_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 2 * x + y + 6 = x * y) : 18 ≤ x * y :=
by
  sorry

end min_value_of_xy_l57_57652


namespace train_length_is_correct_l57_57440

-- Defining the initial conditions
def train_speed_km_per_hr : Float := 90.0
def time_seconds : Float := 5.0

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : Float) : Float :=
  speed_km_per_hr * (1000.0 / 3600.0)

-- Calculate the length of the train in meters
def length_of_train (speed_km_per_hr : Float) (time_s : Float) : Float :=
  km_per_hr_to_m_per_s speed_km_per_hr * time_s

-- Theorem statement
theorem train_length_is_correct : length_of_train train_speed_km_per_hr time_seconds = 125.0 :=
by
  sorry

end train_length_is_correct_l57_57440


namespace Shekar_marks_in_English_l57_57727

theorem Shekar_marks_in_English 
  (math_marks : ℕ) (science_marks : ℕ) (socialstudies_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (num_subjects : ℕ) 
  (mathscore : math_marks = 76)
  (sciencescore : science_marks = 65)
  (socialstudiesscore : socialstudies_marks = 82)
  (biologyscore : biology_marks = 85)
  (averagescore : average_marks = 74)
  (numsubjects : num_subjects = 5) :
  ∃ (english_marks : ℕ), english_marks = 62 :=
by
  sorry

end Shekar_marks_in_English_l57_57727


namespace part_a_part_b_part_c_l57_57485

def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1) ^ n - x ^ n - 1
def P (x : ℝ) : ℝ := x ^ 2 + x + 1

theorem part_a (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) ↔ (∀ x : ℝ, P x ∣ Q x n) := sorry

theorem part_b (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1) ↔ (∀ x : ℝ, (P x)^2 ∣ Q x n) := sorry

theorem part_c (n : ℕ) : 
  n = 1 ↔ (∀ x : ℝ, (P x)^3 ∣ Q x n) := sorry

end part_a_part_b_part_c_l57_57485


namespace total_seats_value_l57_57648

noncomputable def students_per_bus : ℝ := 14.0
noncomputable def number_of_buses : ℝ := 2.0
noncomputable def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_value : total_seats = 28.0 :=
by
  sorry

end total_seats_value_l57_57648


namespace contradiction_assumption_l57_57544

theorem contradiction_assumption (a b : ℝ) (h : a ≤ 2 ∧ b ≤ 2) : (a > 2 ∨ b > 2) -> false :=
by
  sorry

end contradiction_assumption_l57_57544


namespace intersection_P_M_l57_57343

open Set Int

def P : Set ℤ := {x | 0 ≤ x ∧ x < 3}

def M : Set ℤ := {x | x^2 ≤ 9}

theorem intersection_P_M : P ∩ M = {0, 1, 2} := by
  sorry

end intersection_P_M_l57_57343


namespace find_n_l57_57855

def valid_n (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [MOD 15]

theorem find_n : ∃ n, valid_n n ∧ n = 8 :=
by
  sorry

end find_n_l57_57855


namespace proof_problem_l57_57369

variable {R : Type*} [Field R] {x y z w N : R}

theorem proof_problem 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 :=
by sorry

end proof_problem_l57_57369


namespace sufficient_condition_for_ellipse_with_foci_y_axis_l57_57054

theorem sufficient_condition_for_ellipse_with_foci_y_axis (m n : ℝ) (h : m > n ∧ n > 0) :
  (∃ a b : ℝ, (a^2 = m / n) ∧ (b^2 = 1 / n) ∧ (a > b)) ∧ ¬(∀ u v : ℝ, (u^2 = m / v) → (v^2 = 1 / v) → (u > v) → (v = n ∧ u = m)) :=
by
  sorry

end sufficient_condition_for_ellipse_with_foci_y_axis_l57_57054


namespace find_principal_l57_57912

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h₁ : SI = 8625) (h₂ : R = 50 / 3) (h₃ : T = 3 / 4) :
  SI = (P * R * T) / 100 → P = 69000 := sorry

end find_principal_l57_57912


namespace total_flowers_l57_57131

def pieces (f : String) : Nat :=
  if f == "roses" ∨ f == "lilies" ∨ f == "sunflowers" ∨ f == "daisies" then 40 else 0

theorem total_flowers : 
  pieces "roses" + pieces "lilies" + pieces "sunflowers" + pieces "daisies" = 160 := 
by
  sorry


end total_flowers_l57_57131


namespace problem_solution_l57_57259

variables {f : ℝ → ℝ}

-- f is monotonically decreasing on [1, 3]
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

-- f(x+3) is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = f (3 - x)

-- Given conditions
axiom mono_dec : monotone_decreasing_on f 1 3
axiom even_f : even_function f

-- To prove: f(π) < f(2) < f(5)
theorem problem_solution : f π < f 2 ∧ f 2 < f 5 :=
by
  sorry

end problem_solution_l57_57259


namespace second_number_value_l57_57506

theorem second_number_value (x y : ℝ) (h1 : (1/5) * x = (5/8) * y) 
                                      (h2 : x + 35 = 4 * y) : y = 40 := 
by 
  sorry

end second_number_value_l57_57506


namespace darwin_spending_fraction_l57_57982

theorem darwin_spending_fraction {x : ℝ} (h1 : 600 - 600 * x - (1 / 4) * (600 - 600 * x) = 300) :
  x = 1 / 3 :=
sorry

end darwin_spending_fraction_l57_57982


namespace area_of_circle_eq_sixteen_pi_l57_57991

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end area_of_circle_eq_sixteen_pi_l57_57991


namespace cost_of_concessions_l57_57660

theorem cost_of_concessions (total_cost : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  total_cost = 76 →
  adult_ticket_cost = 10 →
  child_ticket_cost = 7 →
  num_adults = 5 →
  num_children = 2 →
  total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_concessions_l57_57660


namespace find_a_l57_57879

open Set

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) (h : (A ∪ B a) ⊆ (A ∩ B a)) : a = 1 :=
sorry

end find_a_l57_57879


namespace max_gcd_of_linear_combinations_l57_57358

theorem max_gcd_of_linear_combinations (a b c : ℕ) (h1 : a + b + c ≤ 3000000) (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  gcd (a * b + 1) (gcd (a * c + 1) (b * c + 1)) ≤ 998285 :=
sorry

end max_gcd_of_linear_combinations_l57_57358


namespace sin_cos_eq_one_l57_57390

theorem sin_cos_eq_one (x : ℝ) (hx0 : 0 ≤ x) (hx2pi : x < 2 * Real.pi) :
  (Real.sin x - Real.cos x = 1) ↔ (x = Real.pi / 2 ∨ x = Real.pi) :=
by
  sorry

end sin_cos_eq_one_l57_57390


namespace clark_family_ticket_cost_l57_57778

theorem clark_family_ticket_cost
  (regular_price children's_price seniors_price : ℝ)
  (number_youngest_gen number_second_youngest_gen number_second_oldest_gen number_oldest_gen : ℕ)
  (h_senior_discount : seniors_price = 0.7 * regular_price)
  (h_senior_ticket_cost : seniors_price = 7)
  (h_child_discount : children's_price = 0.6 * regular_price)
  (h_number_youngest_gen : number_youngest_gen = 3)
  (h_number_second_youngest_gen : number_second_youngest_gen = 1)
  (h_number_second_oldest_gen : number_second_oldest_gen = 2)
  (h_number_oldest_gen : number_oldest_gen = 1)
  : 3 * children's_price + 1 * regular_price + 2 * seniors_price + 1 * regular_price = 52 := by
  sorry

end clark_family_ticket_cost_l57_57778


namespace sphere_cone_radius_ratio_l57_57958

-- Define the problem using given conditions and expected outcome.
theorem sphere_cone_radius_ratio (r R h : ℝ)
  (h1 : h = 2 * r)
  (h2 : (1/3) * π * R^2 * h = 3 * (4/3) * π * r^3) :
  r / R = 1 / Real.sqrt 6 :=
by
  sorry

end sphere_cone_radius_ratio_l57_57958


namespace problem_statement_l57_57009

theorem problem_statement (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := 
by
  sorry

end problem_statement_l57_57009


namespace cos_double_angle_l57_57044

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) :
  Real.cos (2 * θ) = -7 / 9 :=
sorry

end cos_double_angle_l57_57044


namespace number_of_lemons_l57_57996

theorem number_of_lemons
  (total_fruits : ℕ)
  (mangoes : ℕ)
  (pears : ℕ)
  (pawpaws : ℕ)
  (kiwis : ℕ)
  (lemons : ℕ)
  (h_total : total_fruits = 58)
  (h_mangoes : mangoes = 18)
  (h_pears : pears = 10)
  (h_pawpaws : pawpaws = 12)
  (h_kiwis_lemons_equal : kiwis = lemons) :
  lemons = 9 :=
by
  sorry

end number_of_lemons_l57_57996


namespace find_missing_exponent_l57_57048

theorem find_missing_exponent (b e₁ e₂ e₃ e₄ : ℝ) (h1 : e₁ = 5.6) (h2 : e₂ = 10.3) (h3 : e₃ = 13.33744) (h4 : e₄ = 2.56256) :
  (b ^ e₁ * b ^ e₂) / b ^ e₄ = b ^ e₃ :=
by
  have h5 : e₁ + e₂ = 15.9 := sorry
  have h6 : 15.9 - e₄ = 13.33744 := sorry
  exact sorry

end find_missing_exponent_l57_57048


namespace remainder_of_exponentiation_is_correct_l57_57530

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l57_57530


namespace compute_value_l57_57097

-- Definitions based on problem conditions
def x : ℤ := (150 - 100 + 1) * (100 + 150) / 2  -- Sum of integers from 100 to 150

def y : ℤ := (150 - 100) / 2 + 1  -- Number of even integers from 100 to 150

def z : ℤ := 0  -- Product of odd integers from 100 to 150 (including even numbers makes the product 0)

-- The theorem to prove
theorem compute_value : x + y - z = 6401 :=
by
  sorry

end compute_value_l57_57097


namespace students_like_burgers_l57_57513

theorem students_like_burgers (total_students : ℕ) (french_fries_likers : ℕ) (both_likers : ℕ) (neither_likers : ℕ) 
    (h1 : total_students = 25) (h2 : french_fries_likers = 15) (h3 : both_likers = 6) (h4 : neither_likers = 6) : 
    (total_students - neither_likers) - (french_fries_likers - both_likers) = 10 :=
by
  -- The proof will go here.
  sorry

end students_like_burgers_l57_57513


namespace maya_total_pages_read_l57_57571

def last_week_books : ℕ := 5
def pages_per_book : ℕ := 300
def this_week_multiplier : ℕ := 2

theorem maya_total_pages_read : 
  (last_week_books * pages_per_book * (1 + this_week_multiplier)) = 4500 :=
by
  sorry

end maya_total_pages_read_l57_57571


namespace area_of_triangle_l57_57214

theorem area_of_triangle (s1 s2 s3 : ℕ) (h1 : s1^2 = 36) (h2 : s2^2 = 64) (h3 : s3^2 = 100) (h4 : s1^2 + s2^2 = s3^2) :
  (1 / 2 : ℚ) * s1 * s2 = 24 := by
  sorry

end area_of_triangle_l57_57214


namespace total_area_covered_is_60_l57_57808

-- Declare the dimensions of the strips
def length_strip : ℕ := 12
def width_strip : ℕ := 2
def num_strips : ℕ := 3

-- Define the total area covered without overlaps
def total_area_no_overlap := num_strips * (length_strip * width_strip)

-- Define the area of overlap for each pair of strips
def overlap_area_per_pair := width_strip * width_strip

-- Define the total overlap area given 3 pairs
def total_overlap_area := 3 * overlap_area_per_pair

-- Define the actual total covered area
def total_covered_area := total_area_no_overlap - total_overlap_area

-- Prove that the total covered area is 60 square units
theorem total_area_covered_is_60 : total_covered_area = 60 := by 
  sorry

end total_area_covered_is_60_l57_57808


namespace convex_parallelogram_faces_1992_l57_57309

theorem convex_parallelogram_faces_1992 (n : ℕ) (h : n > 0) : (n * (n - 1) ≠ 1992) := 
by
  sorry

end convex_parallelogram_faces_1992_l57_57309


namespace class_grades_l57_57475

theorem class_grades (boys girls n : ℕ) (h1 : girls = boys + 3) (h2 : ∀ (fours fives : ℕ), fours = fives + 6) (h3 : ∀ (threes : ℕ), threes = 2 * (fives + 6)) : ∃ k, k = 2 ∨ k = 1 :=
by
  sorry

end class_grades_l57_57475


namespace emily_garden_larger_l57_57776

-- Define the dimensions and conditions given in the problem
def john_length : ℕ := 30
def john_width : ℕ := 60
def emily_length : ℕ := 35
def emily_width : ℕ := 55

-- Define the effective area for John’s garden given the double space requirement
def john_usable_area : ℕ := (john_length * john_width) / 2

-- Define the total area for Emily’s garden
def emily_usable_area : ℕ := emily_length * emily_width

-- State the theorem to be proved
theorem emily_garden_larger : emily_usable_area - john_usable_area = 1025 :=
by
  sorry

end emily_garden_larger_l57_57776


namespace fraction_of_repeating_decimal_l57_57761

theorem fraction_of_repeating_decimal:
  let a := (4 / 10 : ℝ)
  let r := (1 / 10 : ℝ)
  (∑' n:ℕ, a * r^n) = (4 / 9 : ℝ) := by
  sorry

end fraction_of_repeating_decimal_l57_57761


namespace sufficient_and_necessary_condition_l57_57724

def A : Set ℝ := { x | x - 2 > 0 }

def B : Set ℝ := { x | x < 0 }

def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem sufficient_and_necessary_condition :
  ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C :=
sorry

end sufficient_and_necessary_condition_l57_57724


namespace movie_final_length_l57_57972

theorem movie_final_length (original_length : ℕ) (cut_length : ℕ) (final_length : ℕ) 
  (h1 : original_length = 60) (h2 : cut_length = 8) : 
  final_length = 52 :=
by
  sorry

end movie_final_length_l57_57972


namespace cos_of_sin_given_l57_57741

theorem cos_of_sin_given (θ : ℝ) (h : Real.sin (88 * Real.pi / 180 + θ) = 2 / 3) :
  Real.cos (178 * Real.pi / 180 + θ) = - (2 / 3) :=
by
  sorry

end cos_of_sin_given_l57_57741


namespace solveTheaterProblem_l57_57835

open Nat

def theaterProblem : Prop :=
  ∃ (A C : ℕ), (A + C = 80) ∧ (12 * A + 5 * C = 519) ∧ (C = 63)

theorem solveTheaterProblem : theaterProblem :=
  by
  sorry

end solveTheaterProblem_l57_57835


namespace inscribed_circle_radius_l57_57250

theorem inscribed_circle_radius (a b c : ℝ) (R : ℝ) (r : ℝ) :
  a = 20 → b = 20 → d = 25 → r = 6 := 
by
  -- conditions of the problem
  sorry

end inscribed_circle_radius_l57_57250


namespace find_prime_p_l57_57100

open Int

theorem find_prime_p (p k m n : ℕ) (hp : Nat.Prime p) 
  (hk : 0 < k) (hm : 0 < m)
  (h_eq : (mk^2 + 2 : ℤ) * p - (m^2 + 2 * k^2 : ℤ) = n^2 * (mp + 2 : ℤ)) :
  p = 3 ∨ p = 1 := sorry

end find_prime_p_l57_57100


namespace ordering_of_abc_l57_57585

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem ordering_of_abc : b < a ∧ a < c := by
  sorry

end ordering_of_abc_l57_57585


namespace greatest_n_for_xy_le_0_l57_57576

theorem greatest_n_for_xy_le_0
  (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) :
  ∃ n : ℕ, (n = a * b ∧ ∃ x y : ℤ, n = a * x + b * y ∧ x * y ≤ 0) :=
sorry

end greatest_n_for_xy_le_0_l57_57576


namespace area_of_plot_area_in_terms_of_P_l57_57351

-- Conditions and definitions.
variables (P : ℝ) (l w : ℝ)
noncomputable def perimeter := 2 * (l + w)
axiom h_perimeter : perimeter l w = 120
axiom h_equality : l = 2 * w

-- Proofs statements
theorem area_of_plot : l + w = 60 → l = 2 * w → (4 * w)^2 = 6400 := by
  sorry

theorem area_in_terms_of_P : (4 * (P / 6))^2 = (2 * P / 3)^2 → (2 * P / 3)^2 = 4 * P^2 / 9 := by
  sorry

end area_of_plot_area_in_terms_of_P_l57_57351


namespace quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l57_57315

theorem quad_eq1_solution (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  sorry

theorem quad_eq2_solution (x : ℝ) : 2 * x^2 - 7 * x + 5 = 0 → x = 5 / 2 ∨ x = 1 :=
by
  sorry

theorem quad_eq3_solution (x : ℝ) : (x + 3)^2 - 2 * (x + 3) = 0 → x = -3 ∨ x = -1 :=
by
  sorry

end quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l57_57315


namespace sandy_total_money_received_l57_57363

def sandy_saturday_half_dollars := 17
def sandy_sunday_half_dollars := 6
def half_dollar_value : ℝ := 0.50

theorem sandy_total_money_received :
  (sandy_saturday_half_dollars * half_dollar_value) +
  (sandy_sunday_half_dollars * half_dollar_value) = 11.50 :=
by
  sorry

end sandy_total_money_received_l57_57363


namespace sin_cos_value_l57_57138

noncomputable def tan_plus_pi_div_two_eq_two (θ : ℝ) : Prop :=
  Real.tan (θ + Real.pi / 2) = 2

theorem sin_cos_value (θ : ℝ) (h : tan_plus_pi_div_two_eq_two θ) :
  Real.sin θ * Real.cos θ = -2 / 5 :=
sorry

end sin_cos_value_l57_57138


namespace angles_arith_prog_triangle_l57_57770

noncomputable def a : ℕ := 8
noncomputable def b : ℕ := 37
noncomputable def c : ℕ := 0

theorem angles_arith_prog_triangle (y : ℝ) (h1 : y = 8 ∨ y * y = 37) :
  a + b + c = 45 := by
  -- skipping the detailed proof steps
  sorry

end angles_arith_prog_triangle_l57_57770


namespace work_completion_days_l57_57929

-- We assume D is a certain number of days and W is some amount of work
variables (D W : ℕ)

-- Define the rate at which 3 people can do 3W work in D days
def rate_3_people : ℚ := 3 * W / D

-- Define the rate at which 5 people can do 5W work in D days
def rate_5_people : ℚ := 5 * W / D

-- The problem states that both rates must be equal
theorem work_completion_days : (3 * D) = D / 3 :=
by sorry

end work_completion_days_l57_57929


namespace alex_buys_15_pounds_of_corn_l57_57107

theorem alex_buys_15_pounds_of_corn:
  ∃ (c b : ℝ), c + b = 30 ∧ 1.20 * c + 0.60 * b = 27.00 ∧ c = 15.0 :=
by
  sorry

end alex_buys_15_pounds_of_corn_l57_57107


namespace nails_painted_purple_l57_57366

variable (P S : ℕ)

theorem nails_painted_purple :
  (P + 8 + S = 20) ∧ ((8 / 20 : ℚ) * 100 - (S / 20 : ℚ) * 100 = 10) → P = 6 :=
by
  sorry

end nails_painted_purple_l57_57366


namespace quadratic_no_real_roots_l57_57567

theorem quadratic_no_real_roots (m : ℝ) : (∀ x, x^2 - 2 * x + m ≠ 0) ↔ m > 1 := 
by sorry

end quadratic_no_real_roots_l57_57567


namespace victor_percentage_80_l57_57753

def percentage_of_marks (marks_obtained : ℕ) (maximum_marks : ℕ) : ℕ :=
  (marks_obtained * 100) / maximum_marks

theorem victor_percentage_80 :
  percentage_of_marks 240 300 = 80 := by
  sorry

end victor_percentage_80_l57_57753


namespace kitchen_width_l57_57842

theorem kitchen_width (length : ℕ) (height : ℕ) (rate : ℕ) (hours : ℕ) (coats : ℕ) 
  (total_painted : ℕ) (half_walls_area : ℕ) (total_walls_area : ℕ)
  (width : ℕ) : 
  length = 12 ∧ height = 10 ∧ rate = 40 ∧ hours = 42 ∧ coats = 3 ∧ 
  total_painted = rate * hours ∧ total_painted = coats * total_walls_area ∧
  half_walls_area = 2 * length * height ∧ total_walls_area = half_walls_area + 2 * width * height ∧
  2 * (total_walls_area - half_walls_area / 2) = 2 * width * height →
  width = 16 := 
by
  sorry

end kitchen_width_l57_57842


namespace temperature_at_midnight_l57_57811

-- Define the variables for initial conditions and changes
def T_morning : ℤ := 7 -- Morning temperature in degrees Celsius
def ΔT_noon : ℤ := 2   -- Temperature increase at noon in degrees Celsius
def ΔT_midnight : ℤ := -10  -- Temperature drop at midnight in degrees Celsius

-- Calculate the temperatures at noon and midnight
def T_noon := T_morning + ΔT_noon
def T_midnight := T_noon + ΔT_midnight

-- State the theorem to prove the temperature at midnight
theorem temperature_at_midnight : T_midnight = -1 := by
  sorry

end temperature_at_midnight_l57_57811


namespace satisfy_eqn_l57_57270

/-- 
  Prove that the integer pairs (0, 1), (0, -1), (1, 0), (-1, 0), (2, 2), (-2, -2)
  are the only pairs that satisfy x^5 + y^5 = (x + y)^3
-/
theorem satisfy_eqn (x y : ℤ) : 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (1, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (2, 2) ∨ (x, y) = (-2, -2) ↔ 
  x^5 + y^5 = (x + y)^3 := 
by 
  sorry

end satisfy_eqn_l57_57270


namespace cost_price_of_watch_l57_57323

theorem cost_price_of_watch :
  ∃ (CP : ℝ), (CP * 1.07 = CP * 0.88 + 250) ∧ CP = 250 / 0.19 :=
sorry

end cost_price_of_watch_l57_57323


namespace smaller_angle_at_3_15_l57_57483

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end smaller_angle_at_3_15_l57_57483


namespace hypotenuse_length_50_l57_57726

theorem hypotenuse_length_50 (a b : ℕ) (h₁ : a = 14) (h₂ : b = 48) :
  ∃ c : ℕ, c = 50 ∧ c = Nat.sqrt (a^2 + b^2) :=
by
  sorry

end hypotenuse_length_50_l57_57726


namespace solve_inequality_I_solve_inequality_II_l57_57695

def f (x : ℝ) : ℝ := |x - 1| - |2 * x + 3|

theorem solve_inequality_I (x : ℝ) : f x > 2 ↔ -2 < x ∧ x < -4 / 3 :=
by sorry

theorem solve_inequality_II (a : ℝ) : ∀ x, f x ≤ (3 / 2) * a^2 - a ↔ a ≥ 5 / 3 :=
by sorry

end solve_inequality_I_solve_inequality_II_l57_57695


namespace mul_18396_9999_l57_57492

theorem mul_18396_9999 :
  18396 * 9999 = 183941604 :=
by
  sorry

end mul_18396_9999_l57_57492


namespace Maaza_liters_l57_57025

theorem Maaza_liters 
  (M L : ℕ)
  (Pepsi : ℕ := 144)
  (Sprite : ℕ := 368)
  (total_liters := M + Pepsi + Sprite)
  (cans_required : ℕ := 281)
  (H : total_liters = cans_required * L)
  : M = 50 :=
by
  sorry

end Maaza_liters_l57_57025


namespace area_of_triangle_l57_57387

def triangle (α β γ : Type) : (α ≃ β) ≃ γ ≃ Prop := sorry

variables (α β γ : Type) (AB AC AM : ℝ)
variables (ha : AB = 9) (hb : AC = 17) (hc : AM = 12)

theorem area_of_triangle (α β γ : Type) (AB AC AM : ℝ)
  (ha : AB = 9) (hb : AC = 17) (hc : AM = 12) : 
  ∃ A : ℝ, A = 74 :=
sorry

end area_of_triangle_l57_57387


namespace fraction_sum_5625_l57_57590

theorem fraction_sum_5625 : 
  ∃ (a b : ℕ), 0.5625 = (9 : ℚ) / 16 ∧ (a + b = 25) := 
by 
  sorry

end fraction_sum_5625_l57_57590


namespace decision_has_two_exit_paths_l57_57220

-- Define types representing different flowchart symbols
inductive FlowchartSymbol
| Terminal
| InputOutput
| Process
| Decision

-- Define a function that states the number of exit paths given a flowchart symbol
def exit_paths (s : FlowchartSymbol) : Nat :=
  match s with
  | FlowchartSymbol.Terminal   => 1
  | FlowchartSymbol.InputOutput => 1
  | FlowchartSymbol.Process    => 1
  | FlowchartSymbol.Decision   => 2

-- State the theorem that Decision has two exit paths
theorem decision_has_two_exit_paths : exit_paths FlowchartSymbol.Decision = 2 := by
  sorry

end decision_has_two_exit_paths_l57_57220


namespace count_three_digit_integers_with_product_thirty_l57_57278

theorem count_three_digit_integers_with_product_thirty :
  (∃ S : Finset (ℕ × ℕ × ℕ),
      (∀ (a b c : ℕ), (a, b, c) ∈ S → a * b * c = 30 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9) 
    ∧ S.card = 12) :=
by
  sorry

end count_three_digit_integers_with_product_thirty_l57_57278


namespace find_x0_and_m_l57_57757

theorem find_x0_and_m (x : ℝ) (m : ℝ) (x0 : ℝ) :
  (abs (x + 3) - 2 * x - 1 < 0 ↔ x > 2) ∧ 
  (∃ x, abs (x - m) + abs (x + 1 / m) - 2 = 0) → 
  (x0 = 2 ∧ m = 1) := 
by
  sorry

end find_x0_and_m_l57_57757


namespace cubic_root_identity_l57_57938

theorem cubic_root_identity (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = -3)
  (h3 : a * b * c = -2) : 
  a * (b + c) ^ 2 + b * (c + a) ^ 2 + c * (a + b) ^ 2 = -6 := 
by
  sorry

end cubic_root_identity_l57_57938


namespace stream_current_rate_proof_l57_57435

noncomputable def stream_current_rate (c : ℝ) : Prop :=
  ∃ (c : ℝ), (6 / (8 - c) + 6 / (8 + c) = 2) ∧ c = 4

theorem stream_current_rate_proof : stream_current_rate 4 :=
by {
  -- Proof to be provided here.
  sorry
}

end stream_current_rate_proof_l57_57435


namespace find_integers_divisible_by_18_in_range_l57_57008

theorem find_integers_divisible_by_18_in_range :
  ∃ n : ℕ, (n % 18 = 0) ∧ (n ≥ 900) ∧ (n ≤ 930) ∧ (n = 900 ∨ n = 918) :=
sorry

end find_integers_divisible_by_18_in_range_l57_57008


namespace binom_identity_l57_57711

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) : k * binom n k = n * binom (n - 1) (k - 1) := by
  sorry

end binom_identity_l57_57711


namespace Farrah_total_match_sticks_l57_57593

def boxes := 4
def matchboxes_per_box := 20
def sticks_per_matchbox := 300

def total_matchboxes : Nat :=
  boxes * matchboxes_per_box

def total_match_sticks : Nat :=
  total_matchboxes * sticks_per_matchbox

theorem Farrah_total_match_sticks : total_match_sticks = 24000 := sorry

end Farrah_total_match_sticks_l57_57593


namespace handshake_count_l57_57145

def gathering_handshakes (total_people : ℕ) (know_each_other : ℕ) (know_no_one : ℕ) : ℕ :=
  let group2_handshakes := know_no_one * (total_people - 1)
  group2_handshakes / 2

theorem handshake_count :
  gathering_handshakes 30 20 10 = 145 :=
by
  sorry

end handshake_count_l57_57145


namespace angles_cosine_sum_l57_57688

theorem angles_cosine_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 :=
sorry

end angles_cosine_sum_l57_57688


namespace correct_flag_positions_l57_57735

-- Definitions for the gears and their relations
structure Gear where
  flag_position : ℝ -- position of the flag in degrees

-- Condition: Two identical gears
def identical_gears (A B : Gear) : Prop := true

-- Conditions: Initial positions and gear interaction
def initial_position_A (A : Gear) : Prop := A.flag_position = 0
def initial_position_B (B : Gear) : Prop := B.flag_position = 180
def gear_interaction (A B : Gear) (theta : ℝ) : Prop :=
  A.flag_position = -theta ∧ B.flag_position = theta

-- Definition for the final positions given a rotation angle θ
def final_position (A B : Gear) (theta : ℝ) : Prop :=
  identical_gears A B ∧ initial_position_A A ∧ initial_position_B B ∧ gear_interaction A B theta

-- Theorem stating the positions after some rotation θ
theorem correct_flag_positions (A B : Gear) (theta : ℝ) : final_position A B theta → 
  A.flag_position = -theta ∧ B.flag_position = theta :=
by
  intro h
  cases h
  sorry

end correct_flag_positions_l57_57735


namespace Chris_age_l57_57949

theorem Chris_age 
  (a b c : ℝ)
  (h1 : a + b + c = 36)
  (h2 : c - 5 = a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 15.5454545454545 :=
by
  sorry

end Chris_age_l57_57949
