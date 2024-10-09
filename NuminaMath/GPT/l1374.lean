import Mathlib

namespace min_value_n_l1374_137444

theorem min_value_n (n : ℕ) (h1 : 4 ∣ 60 * n) (h2 : 8 ∣ 60 * n) : n = 1 := 
  sorry

end min_value_n_l1374_137444


namespace quadratic_no_real_roots_l1374_137422

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 - 3 * x - k ≠ 0) → k < -9 / 4 :=
by
  sorry

end quadratic_no_real_roots_l1374_137422


namespace zero_in_interval_l1374_137412

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 3

theorem zero_in_interval : 
    (∀ x y : ℝ, 0 < x → x < y → f x < f y) → 
    (f 1 = -2) →
    (f 2 = Real.log 2 + 5) →
    (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by 
    sorry

end zero_in_interval_l1374_137412


namespace solution_ne_zero_l1374_137475

theorem solution_ne_zero (a x : ℝ) (h : x = a * x + 1) : x ≠ 0 := sorry

end solution_ne_zero_l1374_137475


namespace Triamoeba_Count_After_One_Week_l1374_137450

def TriamoebaCount (n : ℕ) : ℕ :=
  3 ^ n

theorem Triamoeba_Count_After_One_Week : TriamoebaCount 7 = 2187 :=
by
  -- This is the statement to be proved
  sorry

end Triamoeba_Count_After_One_Week_l1374_137450


namespace rhombus_properties_l1374_137483

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2
noncomputable def side_length_of_rhombus (d1 d2 : ℝ) : ℝ := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)

theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 16) :
  area_of_rhombus d1 d2 = 144 ∧ side_length_of_rhombus d1 d2 = Real.sqrt 145 := by
  sorry

end rhombus_properties_l1374_137483


namespace geometric_sequence_ratio_l1374_137457

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
(h_arith : 2 * a 1 * q = a 0 + a 0 * q * q) :
  q = 2 + Real.sqrt 3 ∨ q = 2 - Real.sqrt 3 := 
by
  sorry

end geometric_sequence_ratio_l1374_137457


namespace factorization_problem_l1374_137455

theorem factorization_problem 
  (C D : ℤ)
  (h1 : 15 * y ^ 2 - 76 * y + 48 = (C * y - 16) * (D * y - 3))
  (h2 : C * D = 15)
  (h3 : C * (-3) + D * (-16) = -76)
  (h4 : (-16) * (-3) = 48) : 
  C * D + C = 20 :=
by { sorry }

end factorization_problem_l1374_137455


namespace ordered_pairs_count_l1374_137409

theorem ordered_pairs_count :
  ∃ (p : Finset (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ p → a * b + 45 = 10 * Nat.lcm a b + 18 * Nat.gcd a b) ∧
  p.card = 4 :=
by
  sorry

end ordered_pairs_count_l1374_137409


namespace tan_angle_addition_l1374_137423

theorem tan_angle_addition (y : ℝ) (hyp : Real.tan y = -3) : 
  Real.tan (y + Real.pi / 3) = - (5 * Real.sqrt 3 - 6) / 13 := 
by 
  sorry

end tan_angle_addition_l1374_137423


namespace mary_final_weight_l1374_137410

theorem mary_final_weight : 
  let initial_weight := 99
  let weight_loss1 := 12
  let weight_gain1 := 2 * weight_loss1
  let weight_loss2 := 3 * weight_loss1
  let weight_gain2 := 6
  initial_weight - weight_loss1 + weight_gain1 - weight_loss2 + weight_gain2 = 81 := by 
  sorry

end mary_final_weight_l1374_137410


namespace value_of_x_add_y_l1374_137449

theorem value_of_x_add_y (x y : ℝ) 
  (h1 : x + Real.sin y = 2023)
  (h2 : x + 2023 * Real.cos y = 2021)
  (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (3 * Real.pi / 4)) : 
  x + y = 2023 - (Real.sqrt 2) / 2 + (3 * Real.pi) / 4 := 
sorry

end value_of_x_add_y_l1374_137449


namespace speed_of_second_train_40_kmph_l1374_137469

noncomputable def length_train_1 : ℝ := 140
noncomputable def length_train_2 : ℝ := 160
noncomputable def crossing_time : ℝ := 10.799136069114471
noncomputable def speed_train_1 : ℝ := 60

theorem speed_of_second_train_40_kmph :
  let total_distance := length_train_1 + length_train_2
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  let speed_train_2 := relative_speed_kmph - speed_train_1
  speed_train_2 = 40 :=
by
  sorry

end speed_of_second_train_40_kmph_l1374_137469


namespace ratio_of_M_to_R_l1374_137447

variable (M Q P N R : ℝ)

theorem ratio_of_M_to_R :
      M = 0.40 * Q →
      Q = 0.25 * P →
      N = 0.60 * P →
      R = 0.30 * N →
      M / R = 5 / 9 := by
  sorry

end ratio_of_M_to_R_l1374_137447


namespace probability_of_selecting_letter_a_l1374_137435

def total_ways := Nat.choose 5 2
def ways_to_select_a := 4
def probability_of_selecting_a := (ways_to_select_a : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_letter_a :
  probability_of_selecting_a = 2 / 5 :=
by
  -- proof steps will be filled in here
  sorry

end probability_of_selecting_letter_a_l1374_137435


namespace part1_part2_l1374_137438

-- Part 1: Positive integers with leading digit 6 that become 1/25 of the original number when the leading digit is removed.
theorem part1 (n : ℕ) (m : ℕ) (h1 : m = 6 * 10^n + m) (h2 : m = (6 * 10^n + m) / 25) :
  m = 625 * 10^(n - 2) ∨
  m = 625 * 10^(n - 2 + 1) ∨
  ∃ k : ℕ, m = 625 * 10^(n - 2 + k) :=
sorry

-- Part 2: No positive integer exists which becomes 1/35 of the original number when its leading digit is removed.
theorem part2 (n : ℕ) (m : ℕ) (h : m = 6 * 10^n + m) :
  m ≠ (6 * 10^n + m) / 35 :=
sorry

end part1_part2_l1374_137438


namespace smallest_three_digit_solution_l1374_137482

theorem smallest_three_digit_solution (n : ℕ) : 
  75 * n ≡ 225 [MOD 345] → 100 ≤ n ∧ n ≤ 999 → n = 118 :=
by
  intros h1 h2
  sorry

end smallest_three_digit_solution_l1374_137482


namespace find_missing_number_l1374_137487

theorem find_missing_number 
  (x : ℕ) 
  (avg : (744 + 745 + 747 + 748 + 749 + some_num + 753 + 755 + x) / 9 = 750)
  (hx : x = 755) : 
  some_num = 804 := 
  sorry

end find_missing_number_l1374_137487


namespace geometric_sequence_sum_inequality_l1374_137428

open Classical

variable (a_1 q : ℝ) (h1 : a_1 > 0) (h2 : q > 0) (h3 : q ≠ 1)

theorem geometric_sequence_sum_inequality :
  a_1 + a_1 * q^3 > a_1 * q + a_1 * q^2 :=
by
  sorry

end geometric_sequence_sum_inequality_l1374_137428


namespace minimum_value_of_f_is_15_l1374_137415

noncomputable def f (x : ℝ) : ℝ := 9 * x + (1 / (x - 1))

theorem minimum_value_of_f_is_15 (h : ∀ x, x > 1) : ∃ x, x > 1 ∧ f x = 15 :=
by sorry

end minimum_value_of_f_is_15_l1374_137415


namespace inequality_nonnegative_reals_l1374_137494

theorem inequality_nonnegative_reals (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 :=
sorry

end inequality_nonnegative_reals_l1374_137494


namespace find_m_l1374_137402

open Classical

variable {d : ℤ} (h₁ : d ≠ 0) (a : ℕ → ℤ)

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∃ a₀ : ℤ, ∀ n, a n = a₀ + n * d

theorem find_m 
  (h_seq : arithmetic_sequence a d)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : ∃ m, a m = 8) :
  ∃ m, m = 8 :=
sorry

end find_m_l1374_137402


namespace solve_inequality_l1374_137470

noncomputable def g (x : ℝ) := Real.arcsin x + x^3

theorem solve_inequality (x : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1)
    (h2 : Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3 > 0) :
    0 < x ∧ x ≤ 1 :=
by
  sorry

end solve_inequality_l1374_137470


namespace oil_price_reduction_l1374_137420

theorem oil_price_reduction (P P_reduced : ℝ) (h1 : P_reduced = 50) (h2 : 1000 / P_reduced - 5 = 5) :
  ((P - P_reduced) / P) * 100 = 25 := by
  sorry

end oil_price_reduction_l1374_137420


namespace probability_of_event_A_l1374_137437

def probability_event_A : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes

-- Statement of the theorem
theorem probability_of_event_A :
  probability_event_A = 1 / 6 :=
by
  -- This is where the proof would go, replaced with sorry for now.
  sorry

end probability_of_event_A_l1374_137437


namespace sqrt_88200_simplified_l1374_137416

theorem sqrt_88200_simplified : Real.sqrt 88200 = 210 * Real.sqrt 6 :=
by sorry

end sqrt_88200_simplified_l1374_137416


namespace total_pencils_correct_l1374_137463
  
def original_pencils : ℕ := 2
def added_pencils : ℕ := 3
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 5 := 
by
  -- proof state will be filled here 
  sorry

end total_pencils_correct_l1374_137463


namespace find_abc_l1374_137429

theorem find_abc (a b c : ℝ) (x y : ℝ) :
  (x^2 + y^2 + 2*a*x - b*y + c = 0) ∧
  ((-a, b / 2) = (2, 2)) ∧
  (4 = b^2 / 4 + a^2 - c) →
  a = -2 ∧ b = 4 ∧ c = 4 := by
  sorry

end find_abc_l1374_137429


namespace find_x_l1374_137440

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

noncomputable def lcm_of_10_to_15 : ℕ :=
  leastCommonMultiple 10 (leastCommonMultiple 11 (leastCommonMultiple 12 (leastCommonMultiple 13 (leastCommonMultiple 14 15))))

theorem find_x :
  (lcm_of_10_to_15 / 2310 = 26) := by
  sorry

end find_x_l1374_137440


namespace total_pages_in_book_is_250_l1374_137497

-- Definitions
def avg_pages_first_part := 36
def days_first_part := 3
def avg_pages_second_part := 44
def days_second_part := 3
def pages_last_day := 10

-- Calculate total pages
def total_pages := (days_first_part * avg_pages_first_part) + (days_second_part * avg_pages_second_part) + pages_last_day

-- Theorem statement
theorem total_pages_in_book_is_250 : total_pages = 250 := by
  sorry

end total_pages_in_book_is_250_l1374_137497


namespace average_number_of_glasses_per_box_l1374_137476

-- Definitions and conditions
variables (S L : ℕ) -- S is the number of smaller boxes, L is the number of larger boxes

-- Condition 1: One box contains 12 glasses, and the other contains 16 glasses.
-- (This is implicitly understood in the equation for total glasses)

-- Condition 3: There are 16 more larger boxes than smaller smaller boxes
def condition_3 := L = S + 16

-- Condition 4: The total number of glasses is 480.
def condition_4 := 12 * S + 16 * L = 480

-- Proving the average number of glasses per box is 15
theorem average_number_of_glasses_per_box (h1 : condition_3 S L) (h2 : condition_4 S L) :
  (480 : ℝ) / (S + L) = 15 :=
by 
  -- Assuming S and L are natural numbers 
  sorry

end average_number_of_glasses_per_box_l1374_137476


namespace total_cost_l1374_137405

def num_professionals := 2
def hours_per_professional_per_day := 6
def days_worked := 7
def hourly_rate := 15

theorem total_cost : 
  (num_professionals * hours_per_professional_per_day * days_worked * hourly_rate) = 1260 := by
  sorry

end total_cost_l1374_137405


namespace number_of_tables_l1374_137436

-- Define conditions
def chairs_in_base5 : ℕ := 310  -- chairs in base-5
def chairs_base10 : ℕ := 3 * 5^2 + 1 * 5^1 + 0 * 5^0  -- conversion to base-10
def people_per_table : ℕ := 3

-- The theorem to prove
theorem number_of_tables : chairs_base10 / people_per_table = 26 := by
  -- include the automatic proof here
  sorry

end number_of_tables_l1374_137436


namespace profit_calculation_l1374_137461

variable (x y : ℝ)

-- Conditions
def fabric_constraints_1 : Prop := (0.5 * x + 0.9 * (50 - x) ≤ 38)
def fabric_constraints_2 : Prop := (x + 0.2 * (50 - x) ≤ 26)
def x_range : Prop := (17.5 ≤ x ∧ x ≤ 20)

-- Goal
def profit_expression : ℝ := 15 * x + 1500

theorem profit_calculation (h1 : fabric_constraints_1 x) (h2 : fabric_constraints_2 x) (h3 : x_range x) : y = profit_expression x :=
by
  sorry

end profit_calculation_l1374_137461


namespace no_partition_equal_product_l1374_137492

theorem no_partition_equal_product (n : ℕ) (h_pos : 0 < n) :
  ¬∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧
  A.prod id = B.prod id := sorry

end no_partition_equal_product_l1374_137492


namespace number_of_jump_sequences_l1374_137488

def jump_sequences (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (a 3 = 3) ∧
  (∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 2))

theorem number_of_jump_sequences :
  ∃ a : ℕ → ℕ, jump_sequences a ∧ a 11 = 60 :=
by
  sorry

end number_of_jump_sequences_l1374_137488


namespace divides_expression_l1374_137467

theorem divides_expression (n : ℕ) : 7 ∣ (3^(12 * n^2 + 1) + 2^(6 * n + 2)) := sorry

end divides_expression_l1374_137467


namespace range_m_l1374_137489

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_m (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ (-1 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_m_l1374_137489


namespace beef_weight_loss_percentage_l1374_137454

noncomputable def weight_after_processing : ℝ := 570
noncomputable def weight_before_processing : ℝ := 876.9230769230769

theorem beef_weight_loss_percentage :
  (weight_before_processing - weight_after_processing) / weight_before_processing * 100 = 35 :=
by
  sorry

end beef_weight_loss_percentage_l1374_137454


namespace quadrilateral_area_l1374_137418

theorem quadrilateral_area (a b c d e f : ℝ) : 
    (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 :=
    by sorry

noncomputable def quadrilateral_area_formula (a b c d e f : ℝ) : ℝ :=
    if H : (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 then 
    (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2)
    else 0

-- Ensure that the computed area matches the expected value
example (a b c d e f : ℝ) (H : (a^2 + c^2 - b^2 - d^2)^2 ≤ 4 * e^2 * f^2) : 
    quadrilateral_area_formula a b c d e f = 
        (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2) :=
by simp [quadrilateral_area_formula, H]

end quadrilateral_area_l1374_137418


namespace grid_black_probability_l1374_137404

theorem grid_black_probability :
  let p_black_each_cell : ℝ := 1 / 3 
  let p_not_black : ℝ := (2 / 3) * (2 / 3)
  let p_one_black : ℝ := 1 - p_not_black
  let total_pairs : ℕ := 8
  (p_one_black ^ total_pairs) = (5 / 9) ^ 8 :=
sorry

end grid_black_probability_l1374_137404


namespace line_equation_l1374_137452

theorem line_equation
  (P : ℝ × ℝ) (hP : P = (1, -1))
  (h_perp : ∀ x y : ℝ, 3 * x - 2 * y = 0 → 2 * x + 3 * y = 0):
  ∃ m : ℝ, (2 * P.1 + 3 * P.2 + m = 0) ∧ m = 1 :=
by
  sorry

end line_equation_l1374_137452


namespace sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l1374_137479

noncomputable def P (x : ℝ) : Prop := (x - 1)^2 > 16
noncomputable def Q (x a : ℝ) : Prop := x^2 + (a - 8) * x - 8 * a ≤ 0

theorem sufficient_not_necessary (a : ℝ) (x : ℝ) :
  a = 3 →
  (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem necessary_and_sufficient (a : ℝ) :
  (-5 ≤ a ∧ a ≤ 3) ↔ ∀ x, (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem P_inter_Q (a : ℝ) (x : ℝ) :
  (a > 3 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a) ∨ (5 < x ∧ x ≤ 8)) ∧
  (-5 ≤ a ∧ a ≤ 3 → (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8)) ∧
  (-8 ≤ a ∧ a < -5 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) ∧
  (a < -8 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) :=
sorry

end sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l1374_137479


namespace jason_manager_years_l1374_137472

-- Definitions based on the conditions
def jason_bartender_years : ℕ := 9
def jason_total_months : ℕ := 150
def additional_months_excluded : ℕ := 6

-- Conversion from months to years
def total_years := jason_total_months / 12
def excluded_years := additional_months_excluded / 12

-- Lean statement for the proof problem
theorem jason_manager_years :
  total_years - jason_bartender_years - excluded_years = 3 := by
  sorry

end jason_manager_years_l1374_137472


namespace students_at_start_of_year_l1374_137458

theorem students_at_start_of_year (S : ℝ) (h1 : S + 46.0 = 56) : S = 10 :=
sorry

end students_at_start_of_year_l1374_137458


namespace triangle_side_BC_length_l1374_137419

noncomputable def triangle_side_length
  (AB : ℝ) (angle_a : ℝ) (angle_c : ℝ) : ℝ := 
  let sin_a := Real.sin angle_a
  let sin_c := Real.sin angle_c
  (AB * sin_a) / sin_c

theorem triangle_side_BC_length (AB : ℝ) (angle_a angle_c : ℝ) :
  AB = (Real.sqrt 6) / 2 →
  angle_a = (45 * Real.pi / 180) →
  angle_c = (60 * Real.pi / 180) →
  triangle_side_length AB angle_a angle_c = 1 :=
sorry

end triangle_side_BC_length_l1374_137419


namespace equalize_costs_l1374_137499

theorem equalize_costs (X Y Z : ℝ) (h1 : Y > X) (h2 : Z > Y) : 
  (Y + (Z - (X + Z - 2 * Y) / 3) = Z) → 
   (Y - (Y + Z - (X + Z - 2 * Y)) / 3 = (X + Z - 2 * Y) / 3) := sorry

end equalize_costs_l1374_137499


namespace train_speed_l1374_137498

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l1374_137498


namespace meal_combinations_l1374_137406

def number_of_menu_items : ℕ := 15

theorem meal_combinations (different_orderings : ∀ Yann Camille : ℕ, Yann ≠ Camille → Yann ≤ number_of_menu_items ∧ Camille ≤ number_of_menu_items) : 
  (number_of_menu_items * (number_of_menu_items - 1)) = 210 :=
by sorry

end meal_combinations_l1374_137406


namespace smallest_possible_norm_l1374_137459

-- Defining the vector \begin{pmatrix} -2 \\ 4 \end{pmatrix}
def vec_a : ℝ × ℝ := (-2, 4)

-- Condition: the norm of \mathbf{v} + \begin{pmatrix} -2 \\ 4 \end{pmatrix} = 10
def satisfies_condition (v : ℝ × ℝ) : Prop :=
  (Real.sqrt ((v.1 + vec_a.1) ^ 2 + (v.2 + vec_a.2) ^ 2)) = 10

noncomputable def smallest_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_possible_norm (v : ℝ × ℝ) (h : satisfies_condition v) : smallest_norm v = 10 - 2 * Real.sqrt 5 := by
  sorry

end smallest_possible_norm_l1374_137459


namespace ava_average_speed_l1374_137411

noncomputable def initial_odometer : ℕ := 14941
noncomputable def final_odometer : ℕ := 15051
noncomputable def elapsed_time : ℝ := 4 -- hours

theorem ava_average_speed :
  (final_odometer - initial_odometer) / elapsed_time = 27.5 :=
by
  sorry

end ava_average_speed_l1374_137411


namespace exists_multiple_with_sum_divisible_l1374_137413

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := -- Implementation of sum_of_digits function is omitted here
sorry

-- Main theorem statement
theorem exists_multiple_with_sum_divisible (n : ℕ) (hn : n > 0) : 
  ∃ k, k % n = 0 ∧ sum_of_digits k ∣ k :=
sorry

end exists_multiple_with_sum_divisible_l1374_137413


namespace cost_per_bag_proof_minimize_total_cost_l1374_137490

-- Definitions of given conditions
variable (x y : ℕ) -- cost per bag for brands A and B respectively
variable (m : ℕ) -- number of bags of brand B

def first_purchase_eq := 100 * x + 150 * y = 7000
def second_purchase_eq := 180 * x + 120 * y = 8100
def cost_per_bag_A : ℕ := 25
def cost_per_bag_B : ℕ := 30
def total_bags := 300
def constraint := (300 - m) ≤ 2 * m

-- Prove the costs per bag
theorem cost_per_bag_proof (h1 : first_purchase_eq x y)
                           (h2 : second_purchase_eq x y) :
  x = cost_per_bag_A ∧ y = cost_per_bag_B :=
sorry

-- Define the cost function and prove the purchase strategy
def total_cost (m : ℕ) : ℕ := 25 * (300 - m) + 30 * m

theorem minimize_total_cost (h : constraint m) :
  m = 100 ∧ total_cost 100 = 8000 :=
sorry

end cost_per_bag_proof_minimize_total_cost_l1374_137490


namespace isosceles_triangle_angle_split_l1374_137464

theorem isosceles_triangle_angle_split (A B C1 C2 : ℝ)
  (h_isosceles : A = B)
  (h_greater_than_third : A > C1)
  (h_split : C1 + C2 = C) :
  C1 = C2 :=
sorry

end isosceles_triangle_angle_split_l1374_137464


namespace quadratic_inequality_solution_l1374_137425

theorem quadratic_inequality_solution:
  ∀ x : ℝ, -x^2 + 3 * x - 2 ≥ 0 ↔ (1 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l1374_137425


namespace triangle_inequality_l1374_137443

variables {R : Type*} [LinearOrderedField R]

theorem triangle_inequality 
  (a b c u v w : R)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (a + b + c) * (1 / u + 1 / v + 1 / w) ≤ 3 * (a / u + b / v + c / w) :=
sorry

end triangle_inequality_l1374_137443


namespace proof_problem_l1374_137445

theorem proof_problem (α : ℝ) (h1 : 0 < α ∧ α < π)
    (h2 : Real.sin α + Real.cos α = 1 / 5) :
    (Real.tan α = -4 / 3) ∧ 
    ((Real.sin (3 * Real.pi / 2 + α) * Real.sin (Real.pi / 2 - α) * (Real.tan (Real.pi - α))^3) / 
    (Real.cos (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α)) = -4 / 3) :=
by
  sorry

end proof_problem_l1374_137445


namespace hundred_div_point_two_five_eq_four_hundred_l1374_137426

theorem hundred_div_point_two_five_eq_four_hundred : 100 / 0.25 = 400 := by
  sorry

end hundred_div_point_two_five_eq_four_hundred_l1374_137426


namespace sequence_sum_zero_l1374_137401

-- Define the sequence as a function
def seq (n : ℕ) : ℤ :=
  if (n-1) % 8 < 4
  then (n+1) / 2
  else - (n / 2)

-- Define the sum of the sequence up to a given number
def seq_sum (m : ℕ) : ℤ :=
  (Finset.range (m+1)).sum (λ n => seq n)

-- The actual problem statement
theorem sequence_sum_zero : seq_sum 2012 = 0 :=
  sorry

end sequence_sum_zero_l1374_137401


namespace find_m_for_increasing_graph_l1374_137480

theorem find_m_for_increasing_graph (m : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → (m + 1) * x ^ (3 - m^2) < (m + 1) * y ^ (3 - m^2) → x < y) ↔ m = -2 :=
by
  sorry

end find_m_for_increasing_graph_l1374_137480


namespace probability_no_order_l1374_137434

theorem probability_no_order (P : ℕ) 
  (h1 : 60 ≤ 100) (h2 : 10 ≤ 100) (h3 : 15 ≤ 100) 
  (h4 : 5 ≤ 100) (h5 : 3 ≤ 100) (h6 : 2 ≤ 100) :
  P = 100 - (60 + 10 + 15 + 5 + 3 + 2) :=
by 
  sorry

end probability_no_order_l1374_137434


namespace smallest_square_area_l1374_137432

theorem smallest_square_area :
  (∀ (x y : ℝ), (∃ (x1 x2 y1 y2 : ℝ), y1 = 3 * x1 - 4 ∧ y2 = 3 * x2 - 4 ∧ y = x^2 + 5 ∧ 
  ∀ (k : ℝ), x1 + x2 = 3 ∧ x1 * x2 = 5 - k ∧ 16 * k^2 - 332 * k + 396 = 0 ∧ 
  ((k = 1.5 ∧ 10 * (4 * k - 11) = 50) ∨ 
  (k = 16.5 ∧ 10 * (4 * k - 11) ≠ 50))) → 
  ∃ (A: Real), A = 50) :=
sorry

end smallest_square_area_l1374_137432


namespace arith_seq_general_term_sum_b_n_l1374_137460

-- Definitions and conditions
structure ArithSeq (f : ℕ → ℕ) :=
  (d : ℕ)
  (d_ne_zero : d ≠ 0)
  (Sn : ℕ → ℕ)
  (a3_plus_S5 : f 3 + Sn 5 = 42)
  (geom_seq : (f 4)^2 = (f 1) * (f 13))

-- Given the definitions and conditions, prove the general term formula of the sequence
theorem arith_seq_general_term (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℕ) 
  (d_ne_zero : d ≠ 0) (a3_plus_S5 : a_n 3 + S_n 5 = 42)
  (geom_seq : (a_n 4)^2 = (a_n 1) * (a_n 13)) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- Prove the sum of the first n terms of the sequence b_n
theorem sum_b_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℕ) (n : ℕ):
  b_n n = 1 / (a_n (n - 1) * a_n n) →
  T_n n = (1 / 2) * (1 - (1 / (2 * n - 1))) →
  T_n n = (n - 1) / (2 * n - 1) :=
sorry

end arith_seq_general_term_sum_b_n_l1374_137460


namespace hyperbola_equation_is_correct_l1374_137477

-- Given Conditions
def hyperbola_eq (x y : ℝ) (a : ℝ) : Prop := (x^2) / (a^2) - (y^2) / 4 = 1
def asymptote_eq (x y : ℝ) : Prop := y = (1 / 2) * x

-- Correct answer to be proven
def hyperbola_correct (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 4 = 1

theorem hyperbola_equation_is_correct (x y : ℝ) (a : ℝ) :
  (hyperbola_eq x y a) → (asymptote_eq x y) → (a = 4) → hyperbola_correct x y :=
by 
  intros h_hyperbola h_asymptote h_a
  sorry

end hyperbola_equation_is_correct_l1374_137477


namespace find_a_even_function_l1374_137424

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f x = (x + 1) * (x + a))  
  (h2 : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end find_a_even_function_l1374_137424


namespace tetrahedron_volume_l1374_137478

variable {R : ℝ}
variable {S1 S2 S3 S4 : ℝ}
variable {V : ℝ}

theorem tetrahedron_volume (R : ℝ) (S1 S2 S3 S4 V : ℝ) :
  V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l1374_137478


namespace no_real_roots_in_interval_l1374_137473

variable {a b c : ℝ}

theorem no_real_roots_in_interval (ha : 0 < a) (h : 12 * a + 5 * b + 2 * c > 0) :
  ¬ ∃ α β, (2 < α ∧ α < 3) ∧ (2 < β ∧ β < 3) ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 := by
  sorry

end no_real_roots_in_interval_l1374_137473


namespace find_number_and_n_l1374_137491

def original_number (x y z n : ℕ) : Prop := 
  n = 2 ∧ 100 * x + 10 * y + z = 178

theorem find_number_and_n (x y z n : ℕ) :
  (∀ x y z n, original_number x y z n) ↔ (n = 2 ∧ 100 * x + 10 * y + z = 178) := 
sorry

end find_number_and_n_l1374_137491


namespace power_eq_45_l1374_137484

theorem power_eq_45 (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end power_eq_45_l1374_137484


namespace circle_properties_l1374_137448

theorem circle_properties :
  ∃ p q s : ℝ, 
  (∀ x y : ℝ, x^2 + 16 * y + 89 = -y^2 - 12 * x ↔ (x + p)^2 + (y + q)^2 = s^2) ∧ 
  p + q + s = -14 + Real.sqrt 11 :=
by
  use -6, -8, Real.sqrt 11
  sorry

end circle_properties_l1374_137448


namespace JakePresentWeight_l1374_137427

def JakeWeight (J S : ℕ) : Prop :=
  J - 33 = 2 * S ∧ J + S = 153

theorem JakePresentWeight : ∃ (J : ℕ), ∃ (S : ℕ), JakeWeight J S ∧ J = 113 := 
by
  sorry

end JakePresentWeight_l1374_137427


namespace intersection_with_xz_plane_l1374_137485

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def direction_vector (p1 p2 : Point3D) : Point3D :=
  Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def parametric_eqn (p : Point3D) (d : Point3D) (t : ℝ) : Point3D :=
  Point3D.mk (p.x + t * d.x) (p.y + t * d.y) (p.z + t * d.z)

theorem intersection_with_xz_plane (p1 p2 : Point3D) :
  let d := direction_vector p1 p2
  let t := (p1.y / d.y)
  parametric_eqn p1 d t = Point3D.mk 4 0 9 :=
sorry

#check intersection_with_xz_plane

end intersection_with_xz_plane_l1374_137485


namespace symmetric_line_equation_l1374_137456

theorem symmetric_line_equation (x y : ℝ) :
  (∀ x y : ℝ, x - 3 * y + 5 = 0 ↔ 3 * x - y - 5 = 0) :=
by 
  sorry

end symmetric_line_equation_l1374_137456


namespace halfway_fraction_l1374_137421

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l1374_137421


namespace system_of_equations_l1374_137430

theorem system_of_equations (x y z : ℝ) (h1 : 4 * x - 6 * y - 2 * z = 0) (h2 : 2 * x + 6 * y - 28 * z = 0) (hz : z ≠ 0) :
  (x^2 - 6 * x * y) / (y^2 + 4 * z^2) = -5 :=
by
  sorry

end system_of_equations_l1374_137430


namespace value_of_expression_l1374_137466

theorem value_of_expression : 48^2 - 2 * 48 * 3 + 3^2 = 2025 :=
by
  sorry

end value_of_expression_l1374_137466


namespace percentage_of_fish_gone_bad_l1374_137465

-- Definitions based on conditions
def fish_per_roll : ℕ := 40
def total_fish_bought : ℕ := 400
def sushi_rolls_made : ℕ := 8

-- Definition of fish calculations
def total_fish_used (rolls: ℕ) (per_roll: ℕ) : ℕ := rolls * per_roll
def fish_gone_bad (total : ℕ) (used : ℕ) : ℕ := total - used
def percentage (part : ℕ) (whole : ℕ) : ℚ := (part : ℚ) / (whole : ℚ) * 100

-- Theorem to prove the percentage of bad fish
theorem percentage_of_fish_gone_bad :
  percentage (fish_gone_bad total_fish_bought (total_fish_used sushi_rolls_made fish_per_roll)) total_fish_bought = 20 := by
  sorry

end percentage_of_fish_gone_bad_l1374_137465


namespace cauchy_schwarz_inequality_l1374_137400

theorem cauchy_schwarz_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by
  sorry

end cauchy_schwarz_inequality_l1374_137400


namespace soda_cost_l1374_137441

variable (b s f : ℝ)

noncomputable def keegan_equation : Prop :=
  3 * b + 2 * s + f = 975

noncomputable def alex_equation : Prop :=
  2 * b + 3 * s + f = 900

theorem soda_cost (h1 : keegan_equation b s f) (h2 : alex_equation b s f) : s = 18.75 :=
by
  sorry

end soda_cost_l1374_137441


namespace distance_between_parallel_lines_l1374_137439

class ParallelLines (A B c1 c2 : ℝ)

theorem distance_between_parallel_lines (A B c1 c2 : ℝ)
  [h : ParallelLines A B c1 c2] : 
  A = 4 → B = 3 → c1 = 1 → c2 = -9 → 
  (|c1 - c2| / Real.sqrt (A^2 + B^2)) = 2 :=
by
  intros hA hB hc1 hc2
  rw [hA, hB, hc1, hc2]
  norm_num
  sorry

end distance_between_parallel_lines_l1374_137439


namespace james_vs_combined_l1374_137468

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687
def olivia_balloons : ℕ := 395
def combined_balloons : ℕ := amy_balloons + felix_balloons + olivia_balloons

theorem james_vs_combined :
  1222 = 1222 ∧ 513 = 513 ∧ 687 = 687 ∧ 395 = 395 → combined_balloons - james_balloons = 373 := by
  sorry

end james_vs_combined_l1374_137468


namespace initial_money_l1374_137417

theorem initial_money (spent allowance total initial : ℕ) 
  (h1 : spent = 2) 
  (h2 : allowance = 26) 
  (h3 : total = 29) 
  (h4 : initial - spent + allowance = total) : 
  initial = 5 := 
by 
  sorry

end initial_money_l1374_137417


namespace find_d_l1374_137474

theorem find_d
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd4 : 4 = a * Real.sin 0 + d)
  (hdm2 : -2 = a * Real.sin (π) + d) :
  d = 1 := by
  sorry

end find_d_l1374_137474


namespace kite_area_is_192_l1374_137462

-- Define the points with doubled dimensions
def A : (ℝ × ℝ) := (0, 16)
def B : (ℝ × ℝ) := (8, 24)
def C : (ℝ × ℝ) := (16, 16)
def D : (ℝ × ℝ) := (8, 0)

-- Calculate the area of the kite
noncomputable def kiteArea (A B C D : ℝ × ℝ) : ℝ :=
  let baseUpper := abs (C.1 - A.1)
  let heightUpper := abs (B.2 - A.2)
  let areaUpper := 1 / 2 * baseUpper * heightUpper
  let baseLower := baseUpper
  let heightLower := abs (B.2 - D.2)
  let areaLower := 1 / 2 * baseLower * heightLower
  areaUpper + areaLower

-- State the theorem to prove the kite area is 192 square inches
theorem kite_area_is_192 : kiteArea A B C D = 192 := 
  sorry

end kite_area_is_192_l1374_137462


namespace maximum_n_l1374_137433

theorem maximum_n (n : ℕ) (G : SimpleGraph (Fin n)) :
  (∃ (A : Fin n → Set (Fin 2020)),  ∀ i j, (G.Adj i j ↔ (A i ∩ A j ≠ ∅)) →
  n ≤ 89) := sorry

end maximum_n_l1374_137433


namespace linear_eq_implies_m_eq_1_l1374_137446

theorem linear_eq_implies_m_eq_1 (x y m : ℝ) (h : 3 * (x ^ |m|) + (m + 1) * y = 6) (hm_abs : |m| = 1) (hm_ne_zero : m + 1 ≠ 0) : m = 1 :=
  sorry

end linear_eq_implies_m_eq_1_l1374_137446


namespace number_of_sides_l1374_137493

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l1374_137493


namespace sum_of_first_50_digits_is_216_l1374_137486

noncomputable def sum_first_50_digits_of_fraction : Nat :=
  let repeating_block := [0, 0, 0, 9, 9, 9]
  let full_cycles := 8
  let remaining_digits := [0, 0]
  let sum_full_cycles := full_cycles * (repeating_block.sum)
  let sum_remaining_digits := remaining_digits.sum
  sum_full_cycles + sum_remaining_digits

theorem sum_of_first_50_digits_is_216 :
  sum_first_50_digits_of_fraction = 216 := by
  sorry

end sum_of_first_50_digits_is_216_l1374_137486


namespace inequality_problem_l1374_137408

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end inequality_problem_l1374_137408


namespace opponent_score_l1374_137471

theorem opponent_score (s g c total opponent : ℕ)
  (h1 : s = 20)
  (h2 : g = 2 * s)
  (h3 : c = 2 * g)
  (h4 : total = s + g + c)
  (h5 : total - 55 = opponent) :
  opponent = 85 := by
  sorry

end opponent_score_l1374_137471


namespace triangle_count_lower_bound_l1374_137414

theorem triangle_count_lower_bound (n m : ℕ) (S : Finset (ℕ × ℕ))
  (hS : ∀ (a b : ℕ), (a, b) ∈ S → 1 ≤ a ∧ a < b ∧ b ≤ n) (hm : S.card = m) :
  ∃T, T ≥ 4 * m * (m - n^2 / 4) / (3 * n) := 
by 
  sorry

end triangle_count_lower_bound_l1374_137414


namespace boxes_amount_l1374_137431

/-- 
  A food company has 777 kilograms of food to put into boxes. 
  If each box gets a certain amount of kilograms, they will have 388 full boxes.
  Prove that each box gets 2 kilograms of food.
-/
theorem boxes_amount (total_food : ℕ) (boxes : ℕ) (kilograms_per_box : ℕ) 
  (h_total : total_food = 777)
  (h_boxes : boxes = 388) :
  total_food / boxes = kilograms_per_box :=
by {
  -- Skipped proof
  sorry 
}

end boxes_amount_l1374_137431


namespace triangle_inequality_l1374_137451

theorem triangle_inequality (A B C : ℝ) (k : ℝ) (hABC : A + B + C = π) (h1 : 1 ≤ k) (h2 : k ≤ 2) :
  (1 / (k - Real.cos A)) + (1 / (k - Real.cos B)) + (1 / (k - Real.cos C)) ≥ 6 / (2 * k - 1) := 
by
  sorry

end triangle_inequality_l1374_137451


namespace squares_in_50th_ring_l1374_137403

-- Define the problem using the given conditions
def centered_square_3x3 : ℕ := 3 -- Represent the 3x3 centered square

-- Define the function that computes the number of unit squares in the nth ring
def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  if n = 1 then 16
  else 24 + 8 * (n - 2)

-- Define the accumulation of unit squares up to the 50th ring
def total_squares_in_50th_ring : ℕ :=
  33 + 24 * 49

theorem squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1209 :=
by
  -- Ensure that the correct value for the 50th ring can be verified
  sorry

end squares_in_50th_ring_l1374_137403


namespace height_of_sky_island_l1374_137496

theorem height_of_sky_island (day_climb : ℕ) (night_slide : ℕ) (days : ℕ) (final_day_climb : ℕ) :
  day_climb = 25 →
  night_slide = 3 →
  days = 64 →
  final_day_climb = 25 →
  (days - 1) * (day_climb - night_slide) + final_day_climb = 1411 :=
by
  -- Add the formal proof here
  sorry

end height_of_sky_island_l1374_137496


namespace number_of_functions_with_given_range_l1374_137442

theorem number_of_functions_with_given_range : 
  let S := {2, 5, 10}
  let R (x : ℤ) := x^2 + 1
  ∃ f : ℤ → ℤ, (∀ y ∈ S, ∃ x : ℤ, f x = y) ∧ (f '' {x | R x ∈ S} = S) :=
    sorry

end number_of_functions_with_given_range_l1374_137442


namespace probability_red_side_l1374_137481

theorem probability_red_side (total_cards : ℕ)
  (cards_black_black : ℕ) (cards_black_red : ℕ) (cards_red_red : ℕ)
  (h_total : total_cards = 9)
  (h_black_black : cards_black_black = 4)
  (h_black_red : cards_black_red = 2)
  (h_red_red : cards_red_red = 3) :
  let total_sides := (cards_black_black * 2) + (cards_black_red * 2) + (cards_red_red * 2)
  let red_sides := (cards_black_red * 1) + (cards_red_red * 2)
  (red_sides > 0) →
  ((cards_red_red * 2) / red_sides : ℚ) = 3 / 4 := 
by
  intros
  sorry

end probability_red_side_l1374_137481


namespace savings_after_increase_l1374_137453

theorem savings_after_increase (salary savings_rate increase_rate : ℝ) (old_savings old_expenses new_expenses new_savings : ℝ)
  (h_salary : salary = 6000)
  (h_savings_rate : savings_rate = 0.2)
  (h_increase_rate : increase_rate = 0.2)
  (h_old_savings : old_savings = savings_rate * salary)
  (h_old_expenses : old_expenses = salary - old_savings)
  (h_new_expenses : new_expenses = old_expenses * (1 + increase_rate))
  (h_new_savings : new_savings = salary - new_expenses) :
  new_savings = 240 :=
by sorry

end savings_after_increase_l1374_137453


namespace find_value_l1374_137495

open Classical

variables (a b c : ℝ)

-- Assume a, b, c are roots of the polynomial x^3 - 24x^2 + 50x - 42
def is_root (x : ℝ) : Prop := x^3 - 24*x^2 + 50*x - 42 = 0

-- Vieta's formulas for the given polynomial
axiom h1 : is_root a
axiom h2 : is_root b
axiom h3 : is_root c
axiom h4 : a + b + c = 24
axiom h5 : a * b + b * c + c * a = 50
axiom h6 : a * b * c = 42

-- We want to prove the given expression equals 476/43
theorem find_value : 
  (a/(1/a + b*c) + b/(1/b + c*a) + c/(1/c + a*b) = 476/43) :=
sorry

end find_value_l1374_137495


namespace intersection_primes_evens_l1374_137407

open Set

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def evens : Set ℕ := {n | n % 2 = 0}
def primes : Set ℕ := {n | is_prime n}

theorem intersection_primes_evens :
  primes ∩ evens = {2} :=
by sorry

end intersection_primes_evens_l1374_137407
