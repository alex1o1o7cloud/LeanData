import Mathlib

namespace tan_alpha_value_l286_28601

variables (α β : ℝ)

theorem tan_alpha_value
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) :
  Real.tan α = 13 / 16 :=
sorry

end tan_alpha_value_l286_28601


namespace john_finishes_fourth_task_at_12_18_PM_l286_28629

theorem john_finishes_fourth_task_at_12_18_PM :
  let start_time := 8 * 60 + 45 -- Start time in minutes from midnight
  let third_task_time := 11 * 60 + 25 -- End time of the third task in minutes from midnight
  let total_time_three_tasks := third_task_time - start_time -- Total time in minutes to complete three tasks
  let time_per_task := total_time_three_tasks / 3 -- Time per task in minutes
  let fourth_task_end_time := third_task_time + time_per_task -- End time of the fourth task in minutes from midnight
  fourth_task_end_time = 12 * 60 + 18 := -- Expected end time in minutes from midnight
  sorry

end john_finishes_fourth_task_at_12_18_PM_l286_28629


namespace andy_correct_answer_l286_28645

-- Let y be the number Andy is using
def y : ℕ := 13  -- Derived from the conditions

-- Given condition based on Andy's incorrect operation
def condition : Prop := 4 * y + 5 = 57

-- Statement of the proof problem
theorem andy_correct_answer : condition → ((y + 5) * 4 = 72) := by
  intros h
  sorry

end andy_correct_answer_l286_28645


namespace translate_graph_cos_l286_28644

/-- Let f(x) = cos(2x). 
    Translate f(x) to the left by π/6 units to get g(x), 
    then translate g(x) upwards by 1 unit to get h(x). 
    Prove that h(x) = cos(2x + π/3) + 1. -/
theorem translate_graph_cos :
  let f (x : ℝ) := Real.cos (2 * x)
  let g (x : ℝ) := f (x + Real.pi / 6)
  let h (x : ℝ) := g x + 1
  ∀ (x : ℝ), h x = Real.cos (2 * x + Real.pi / 3) + 1 :=
by
  sorry

end translate_graph_cos_l286_28644


namespace seq_a_n_100th_term_l286_28618

theorem seq_a_n_100th_term :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ 
  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) ∧ 
  a 100 = -3 := 
sorry

end seq_a_n_100th_term_l286_28618


namespace sum_of_digits_eleven_l286_28685

-- Definitions for the problem conditions
def distinct_digits (p q r : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ p < 10 ∧ q < 10 ∧ r < 10

def is_two_digit_prime (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n.Prime

def concat_digits (x y : Nat) : Nat :=
  10 * x + y

def problem_conditions (p q r : Nat) : Prop :=
  distinct_digits p q r ∧
  is_two_digit_prime (concat_digits p q) ∧
  is_two_digit_prime (concat_digits p r) ∧
  is_two_digit_prime (concat_digits q r) ∧
  (concat_digits p q) * (concat_digits p r) = 221

-- Lean 4 statement to prove the sum of p, q, r is 11
theorem sum_of_digits_eleven (p q r : Nat) (h : problem_conditions p q r) : p + q + r = 11 :=
sorry

end sum_of_digits_eleven_l286_28685


namespace initial_cakes_l286_28624

variable (friend_bought : Nat) (baker_has : Nat)

theorem initial_cakes (h1 : friend_bought = 140) (h2 : baker_has = 15) : 
  (friend_bought + baker_has = 155) := 
by
  sorry

end initial_cakes_l286_28624


namespace total_visitors_l286_28612

noncomputable def visitors_questionnaire (V E U : ℕ) : Prop :=
  (130 ≠ E ∧ E ≠ U) ∧ 
  (E = U) ∧ 
  (3 * V = 4 * E) ∧ 
  (V = 130 + 3 / 4 * V)

theorem total_visitors (V : ℕ) : visitors_questionnaire V V V → V = 520 :=
by sorry

end total_visitors_l286_28612


namespace required_moles_h2so4_l286_28689

-- Defining chemical equation conditions
def balanced_reaction (nacl h2so4 hcl nahso4 : ℕ) : Prop :=
  nacl = h2so4 ∧ hcl = nacl ∧ nahso4 = nacl

-- Theorem statement
theorem required_moles_h2so4 (nacl_needed moles_h2so4 : ℕ) (hcl_produced nahso4_produced : ℕ)
  (h : nacl_needed = 2 ∧ balanced_reaction nacl_needed moles_h2so4 hcl_produced nahso4_produced) :
  moles_h2so4 = 2 :=
  sorry

end required_moles_h2so4_l286_28689


namespace determine_h_l286_28635

theorem determine_h (x : ℝ) (h : ℝ → ℝ) :
  2 * x ^ 5 + 4 * x ^ 3 + h x = 7 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 →
  h x = -2 * x ^ 5 + 3 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 :=
by
  intro h_eq
  sorry

end determine_h_l286_28635


namespace loss_percent_l286_28678

theorem loss_percent (CP SP Loss : ℝ) (h1 : CP = 600) (h2 : SP = 450) (h3 : Loss = CP - SP) : (Loss / CP) * 100 = 25 :=
by
  sorry

end loss_percent_l286_28678


namespace map_distance_l286_28606

theorem map_distance (scale_cm : ℝ) (scale_km : ℝ) (actual_distance_km : ℝ) 
  (h1 : scale_cm = 0.4) (h2 : scale_km = 5.3) (h3 : actual_distance_km = 848) :
  actual_distance_km / (scale_km / scale_cm) = 64 :=
by
  rw [h1, h2, h3]
  -- Further steps would follow here, but to ensure code compiles
  -- and there is no assumption directly from solution steps, we use sorry.
  sorry

end map_distance_l286_28606


namespace ratio_geometric_sequence_of_arithmetic_l286_28617

variable {d : ℤ}
variable {a : ℕ → ℤ}

-- definition of an arithmetic sequence with common difference d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- definition of a geometric sequence for a_5, a_9, a_{15}
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 9 * a 9 = a 5 * a 15

theorem ratio_geometric_sequence_of_arithmetic
  (h_arith : arithmetic_sequence a d) (h_nonzero : d ≠ 0) (h_geom : geometric_sequence a) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end ratio_geometric_sequence_of_arithmetic_l286_28617


namespace total_legs_in_park_l286_28640

theorem total_legs_in_park :
  let dogs := 109
  let cats := 37
  let birds := 52
  let spiders := 19
  let dog_legs := 4
  let cat_legs := 4
  let bird_legs := 2
  let spider_legs := 8
  dogs * dog_legs + cats * cat_legs + birds * bird_legs + spiders * spider_legs = 840 := by
  sorry

end total_legs_in_park_l286_28640


namespace problem_statement_l286_28670

variable (n : ℕ)
variable (op : ℕ → ℕ → ℕ)
variable (h1 : op 1 1 = 1)
variable (h2 : ∀ n, op (n+1) 1 = 3 * op n 1)

theorem problem_statement : op 5 1 - op 2 1 = 78 := by
  sorry

end problem_statement_l286_28670


namespace value_of_2a_minus_1_l286_28603

theorem value_of_2a_minus_1 (a : ℝ) (h : ∀ x : ℝ, (x = 2 → (3 / 2) * x - 2 * a = 0)) : 2 * a - 1 = 2 :=
sorry

end value_of_2a_minus_1_l286_28603


namespace three_digit_number_addition_l286_28694

theorem three_digit_number_addition (a b : ℕ) (ha : a < 10) (hb : b < 10) (h1 : 307 + 294 = 6 * 100 + b * 10 + 1)
  (h2 : (6 * 100 + b * 10 + 1) % 7 = 0) : a + b = 8 :=
by {
  sorry  -- Proof steps not needed
}

end three_digit_number_addition_l286_28694


namespace find_group_2018_l286_28682

-- Definition of the conditions
def group_size (n : Nat) : Nat := 3 * n - 2

def total_numbers (n : Nat) : Nat := 
  (3 * n * n - n) / 2

theorem find_group_2018 : ∃ n : Nat, total_numbers (n - 1) < 1009 ∧ total_numbers n ≥ 1009 ∧ n = 27 :=
  by
  -- This forms the structure for the proof
  sorry

end find_group_2018_l286_28682


namespace smallest_number_among_l286_28650

theorem smallest_number_among
  (π : ℝ) (Hπ_pos : π > 0) :
  ∀ (a b c d : ℝ), 
    (a = 0) → 
    (b = -1) → 
    (c = -1.5) → 
    (d = π) → 
    (∀ (x y : ℝ), (x > 0) → (y > 0) → (x > y) ↔ x - y > 0) → 
    (∀ (x : ℝ), x < 0 → x < 0) → 
    (∀ (x y : ℝ), (x > 0) → (y < 0) → x > y) → 
    (∀ (x y : ℝ), (x < 0) → (y < 0) → (|x| > |y|) → x < y) → 
  c = -1.5 := 
by
  intros a b c d Ha Hb Hc Hd Hpos Hneg HposNeg Habs
  sorry

end smallest_number_among_l286_28650


namespace total_percentage_of_failed_candidates_l286_28641

theorem total_percentage_of_failed_candidates :
  ∀ (total_candidates girls boys : ℕ) (passed_boys passed_girls : ℝ),
    total_candidates = 2000 →
    girls = 900 →
    boys = total_candidates - girls →
    passed_boys = 0.34 * boys →
    passed_girls = 0.32 * girls →
    (total_candidates - (passed_boys + passed_girls)) / total_candidates * 100 = 66.9 :=
by
  intros total_candidates girls boys passed_boys passed_girls
  intro h_total_candidates
  intro h_girls
  intro h_boys
  intro h_passed_boys
  intro h_passed_girls
  sorry

end total_percentage_of_failed_candidates_l286_28641


namespace part1_part2_l286_28631

variable (A B C : ℝ) (a b c : ℝ)
variable (h1 : a = 5) (h2 : c = 6) (h3 : Real.sin B = 3 / 5) (h4 : b < a)

-- Part 1: Prove b = sqrt(13) and sin A = (3 * sqrt(13)) / 13
theorem part1 : b = Real.sqrt 13 ∧ Real.sin A = (3 * Real.sqrt 13) / 13 := sorry

-- Part 2: Prove sin (2A + π / 4) = 7 * sqrt(2) / 26
theorem part2 (h5 : b = Real.sqrt 13) (h6 : Real.sin A = (3 * Real.sqrt 13) / 13) : 
  Real.sin (2 * A + Real.pi / 4) = (7 * Real.sqrt 2) / 26 := sorry

end part1_part2_l286_28631


namespace minimum_value_sum_l286_28669

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a / (2 * b)) + (b / (4 * c)) + (c / (8 * a))

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c >= 3/4 :=
by
  sorry

end minimum_value_sum_l286_28669


namespace max_value_of_f_l286_28677

noncomputable def f (x : ℝ) : ℝ := x * (4 - x)

theorem max_value_of_f : ∃ y, ∀ x ∈ Set.Ioo 0 4, f x ≤ y ∧ y = 4 :=
by
  sorry

end max_value_of_f_l286_28677


namespace train_cross_pole_in_5_seconds_l286_28690

/-- A train 100 meters long traveling at 72 kilometers per hour 
    will cross an electric pole in 5 seconds. -/
theorem train_cross_pole_in_5_seconds (L : ℝ) (v : ℝ) (t : ℝ) : 
  L = 100 → v = 72 * (1000 / 3600) → t = L / v → t = 5 :=
by
  sorry

end train_cross_pole_in_5_seconds_l286_28690


namespace ratio_of_visible_spots_l286_28661

theorem ratio_of_visible_spots (S S1 : ℝ) (h1 : ∀ (fold_type : ℕ), 
  (fold_type = 1 ∨ fold_type = 2 ∨ fold_type = 3) → 
  (if fold_type = 1 ∨ fold_type = 2 then S1 else S) = S1) : S1 / S = 2 / 3 := 
sorry

end ratio_of_visible_spots_l286_28661


namespace total_gold_coins_l286_28614

/--
An old man distributed all the gold coins he had to his two sons into 
two different numbers such that the difference between the squares 
of the two numbers is 49 times the difference between the two numbers. 
Prove that the total number of gold coins the old man had is 49.
-/
theorem total_gold_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 :=
sorry

end total_gold_coins_l286_28614


namespace simplify_expression_l286_28605

theorem simplify_expression (x y : ℝ) : 
  (x - y) * (x + y) + (x - y) ^ 2 = 2 * x ^ 2 - 2 * x * y :=
sorry

end simplify_expression_l286_28605


namespace find_central_angle_l286_28688

theorem find_central_angle
  (θ r : ℝ)
  (h1 : r * θ = 2 * π)
  (h2 : (1 / 2) * r^2 * θ = 3 * π) :
  θ = 2 * π / 3 := 
sorry

end find_central_angle_l286_28688


namespace least_value_q_minus_p_l286_28679

def p : ℝ := 2
def q : ℝ := 5

theorem least_value_q_minus_p (y : ℝ) (h : p < y ∧ y < q) : q - p = 3 :=
by
  sorry

end least_value_q_minus_p_l286_28679


namespace shopper_total_payment_l286_28628

theorem shopper_total_payment :
  let original_price := 150
  let discount_rate := 0.25
  let coupon_discount := 10
  let sales_tax_rate := 0.10
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price := price_after_coupon * (1 + sales_tax_rate)
  final_price = 112.75 := by
{
  sorry
}

end shopper_total_payment_l286_28628


namespace no_solution_a_squared_plus_b_squared_eq_2023_l286_28630

theorem no_solution_a_squared_plus_b_squared_eq_2023 :
  ∀ (a b : ℤ), a^2 + b^2 ≠ 2023 := 
by
  sorry

end no_solution_a_squared_plus_b_squared_eq_2023_l286_28630


namespace Damien_jogs_miles_over_three_weeks_l286_28625

theorem Damien_jogs_miles_over_three_weeks :
  (5 * 5) * 3 = 75 :=
by sorry

end Damien_jogs_miles_over_three_weeks_l286_28625


namespace factor_expression_l286_28615

theorem factor_expression:
  ∀ (x : ℝ), (10 * x^3 + 50 * x^2 - 4) - (3 * x^3 - 5 * x^2 + 2) = 7 * x^3 + 55 * x^2 - 6 :=
by
  sorry

end factor_expression_l286_28615


namespace determine_a_value_l286_28663

-- Define the initial equation and conditions
def fractional_equation (x a : ℝ) : Prop :=
  (x - a) / (x - 1) - 3 / x = 1

-- Define the existence of a positive root
def has_positive_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ fractional_equation x a

-- The main theorem stating the correct value of 'a' for the given condition
theorem determine_a_value (x : ℝ) : has_positive_root 1 :=
sorry

end determine_a_value_l286_28663


namespace intersection_M_N_l286_28646

def M := { x : ℝ | -1 < x ∧ x < 2 }
def N := { x : ℝ | x ≤ 1 }
def expectedIntersection := { x : ℝ | -1 < x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = expectedIntersection :=
by
  sorry

end intersection_M_N_l286_28646


namespace exists_two_people_with_property_l286_28600

theorem exists_two_people_with_property (n : ℕ) (P : Fin (2 * n + 2) → Fin (2 * n + 2) → Prop) :
  ∃ A B : Fin (2 * n + 2), 
    A ≠ B ∧
    (∃ S : Finset (Fin (2 * n + 2)), 
      S.card = n ∧
      ∀ C ∈ S, (P C A ∧ P C B) ∨ (¬P C A ∧ ¬P C B)) :=
sorry

end exists_two_people_with_property_l286_28600


namespace pow_two_sub_one_not_square_l286_28684

theorem pow_two_sub_one_not_square (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end pow_two_sub_one_not_square_l286_28684


namespace point_in_first_quadrant_l286_28676

theorem point_in_first_quadrant (m : ℝ) (h : m < 0) : 
  (-m > 0) ∧ (-m + 1 > 0) :=
by 
  sorry

end point_in_first_quadrant_l286_28676


namespace find_side_length_of_largest_square_l286_28672

theorem find_side_length_of_largest_square (A : ℝ) (hA : A = 810) :
  ∃ a : ℝ, (5 / 8) * a ^ 2 = A ∧ a = 36 := by
  sorry

end find_side_length_of_largest_square_l286_28672


namespace three_x_plus_four_l286_28686

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end three_x_plus_four_l286_28686


namespace range_of_a_l286_28657

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x | x^2 ≤ 1} ∪ {a} ↔ x ∈ {x | x^2 ≤ 1}) → (-1 ≤ a ∧ a ≤ 1) :=
by
  intro h
  sorry

end range_of_a_l286_28657


namespace find_base_solve_inequality_case1_solve_inequality_case2_l286_28664

noncomputable def log_function (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_base (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : log_function a 8 = 3 → a = 2 :=
by sorry

theorem solve_inequality_case1 (a : ℝ) (h₁ : 1 < a) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 0 < x ∧ x ≤ 1 / 2 :=
by sorry

theorem solve_inequality_case2 (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 1 / 2 ≤ x ∧ x < 2 / 3 :=
by sorry

end find_base_solve_inequality_case1_solve_inequality_case2_l286_28664


namespace h_at_neg_eight_l286_28660

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + x + 1

noncomputable def h (x : ℝ) (a b c : ℝ) : ℝ := (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_neg_eight (a b c : ℝ) (hf : f a = 0) (hf_b : f b = 0) (hf_c : f c = 0) :
  h (-8) a b c = -115 :=
  sorry

end h_at_neg_eight_l286_28660


namespace fibers_below_20_count_l286_28643

variable (fibers : List ℕ)

-- Conditions
def total_fibers := fibers.length = 100
def length_interval (f : ℕ) := 5 ≤ f ∧ f ≤ 40
def fibers_within_interval := ∀ f ∈ fibers, length_interval f

-- Question
def fibers_less_than_20 (fibers : List ℕ) : Nat :=
  (fibers.filter (λ f => f < 20)).length

theorem fibers_below_20_count (h_total : total_fibers fibers)
  (h_interval : fibers_within_interval fibers)
  (histogram_data : fibers_less_than_20 fibers = 30) :
  fibers_less_than_20 fibers = 30 :=
by
  sorry

end fibers_below_20_count_l286_28643


namespace sum_of_interior_angles_of_special_regular_polygon_l286_28656

theorem sum_of_interior_angles_of_special_regular_polygon (n : ℕ) (h1 : n = 4 ∨ n = 5) :
  ((n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540) :=
by sorry

end sum_of_interior_angles_of_special_regular_polygon_l286_28656


namespace option_d_correct_l286_28619

theorem option_d_correct (m n : ℝ) : (m + n) * (m - 2 * n) = m^2 - m * n - 2 * n^2 :=
by
  sorry

end option_d_correct_l286_28619


namespace sum_of_first_n_terms_l286_28608

-- Definitions for the sequences and the problem conditions.
def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := 2 * n - 1
def c (n : ℕ) : ℕ := a n * b n
def T (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ (n + 1) + 6

-- The theorem statement
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range n).sum c = T n :=
  sorry

end sum_of_first_n_terms_l286_28608


namespace problem1_problem2_problem3_l286_28687

-- Definitions of sets A, B, and C as per given conditions
def set_A (a : ℝ) : Set ℝ :=
  {x | x^2 - a * x + a^2 - 19 = 0}

def set_B : Set ℝ :=
  {x | x^2 - 5 * x + 6 = 0}

def set_C : Set ℝ :=
  {x | x^2 + 2 * x - 8 = 0}

-- Questions reformulated as proof problems
theorem problem1 (a : ℝ) (h : set_A a = set_B) : a = 5 :=
sorry

theorem problem2 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : ∀ x, x ∈ set_A a → x ∉ set_C) : a = -2 :=
sorry

theorem problem3 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : set_A a ∩ set_B = set_A a ∩ set_C) : a = -3 :=
sorry

end problem1_problem2_problem3_l286_28687


namespace tangent_line_eq_max_min_values_l286_28658

noncomputable def f (x : ℝ) : ℝ := (1 / (3:ℝ)) * x^3 - 4 * x + 4

theorem tangent_line_eq (x y : ℝ) : 
    y = f 1 → 
    y = -3 * (x - 1) + f 1 → 
    3 * x + y - 10 / 3 = 0 := 
sorry

theorem max_min_values (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) : 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≤ 4) ∧ 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≥ -4 / 3) := 
sorry

end tangent_line_eq_max_min_values_l286_28658


namespace blanket_rate_l286_28623

/-- 
A man purchased 4 blankets at Rs. 100 each, 
5 blankets at Rs. 150 each, 
and two blankets at an unknown rate x. 
If the average price of the blankets was Rs. 150, 
prove that the unknown rate x is 250. 
-/
theorem blanket_rate (x : ℝ) 
  (h1 : 4 * 100 + 5 * 150 + 2 * x = 11 * 150) : 
  x = 250 := 
sorry

end blanket_rate_l286_28623


namespace ducks_and_geese_difference_l286_28639

variable (d g d' l : ℕ)
variables (hd : d = 25)
variables (hg : g = 2 * d - 10)
variables (hd' : d' = d + 4)
variables (hl : l = 15 - 5)

theorem ducks_and_geese_difference :
  let geese_remain := g - l
  let ducks_remain := d'
  geese_remain - ducks_remain = 1 :=
by
  sorry

end ducks_and_geese_difference_l286_28639


namespace possible_values_of_n_l286_28609

-- Conditions: Definition of equilateral triangles and squares with side length 1
def equilateral_triangle_side_length_1 : Prop := ∀ (a : ℕ), 
  ∃ (triangle : ℕ), triangle * 60 = 180 * (a - 2)

def square_side_length_1 : Prop := ∀ (b : ℕ), 
  ∃ (square : ℕ), square * 90 = 180 * (b - 2)

-- Definition of convex n-sided polygon formed using these pieces
def convex_polygon_formed (n : ℕ) : Prop := 
  ∃ (a b c d : ℕ), 
    a + b + c + d = n ∧ 
    60 * a + 90 * b + 120 * c + 150 * d = 180 * (n - 2)

-- Equivalent proof problem
theorem possible_values_of_n :
  ∃ (n : ℕ), (5 ≤ n ∧ n ≤ 12) ∧ convex_polygon_formed n :=
sorry

end possible_values_of_n_l286_28609


namespace num_bad_oranges_l286_28681

theorem num_bad_oranges (G B : ℕ) (hG : G = 24) (ratio : G / B = 3) : B = 8 :=
by
  sorry

end num_bad_oranges_l286_28681


namespace weight_of_5_moles_BaO_molar_concentration_BaO_l286_28671

-- Definitions based on conditions
def atomic_mass_Ba : ℝ := 137.33
def atomic_mass_O : ℝ := 16.00
def molar_mass_BaO : ℝ := atomic_mass_Ba + atomic_mass_O
def moles_BaO : ℝ := 5
def volume_solution : ℝ := 3

-- Theorem statements
theorem weight_of_5_moles_BaO : moles_BaO * molar_mass_BaO = 766.65 := by
  sorry

theorem molar_concentration_BaO : moles_BaO / volume_solution = 1.67 := by
  sorry

end weight_of_5_moles_BaO_molar_concentration_BaO_l286_28671


namespace cells_count_at_day_8_l286_28642

theorem cells_count_at_day_8 :
  let initial_cells := 3
  let common_ratio := 2
  let days := 8
  let interval := 2
  ∃ days_intervals, days_intervals = days / interval ∧ initial_cells * common_ratio ^ days_intervals = 48 :=
by
  sorry

end cells_count_at_day_8_l286_28642


namespace probability_two_points_one_unit_apart_l286_28626

theorem probability_two_points_one_unit_apart :
  let total_points := 10
  let total_ways := (total_points * (total_points - 1)) / 2
  let favorable_horizontal_pairs := 8
  let favorable_vertical_pairs := 5
  let favorable_pairs := favorable_horizontal_pairs + favorable_vertical_pairs
  let probability := (favorable_pairs : ℚ) / total_ways
  probability = 13 / 45 :=
by
  sorry

end probability_two_points_one_unit_apart_l286_28626


namespace value_of_b_l286_28620

theorem value_of_b (x y b : ℝ) (h1: 7^(3 * x - 1) * b^(4 * y - 3) = 49^x * 27^y) (h2: x + y = 4) : b = 3 :=
by
  sorry

end value_of_b_l286_28620


namespace sequence_sum_l286_28673

theorem sequence_sum (r : ℝ) (x y : ℝ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 4096)
  (h2 : a 2 = 1024)
  (h3 : a 3 = 256)
  (h4 : a 6 = 4)
  (h5 : a 7 = 1)
  (h6 : a 8 = 0.25)
  (h_sequence : ∀ n, a (n + 1) = r * a n)
  (h_r : r = 1 / 4) :
  x + y = 80 :=
sorry

end sequence_sum_l286_28673


namespace remainder_23_2057_mod_25_l286_28695

theorem remainder_23_2057_mod_25 : (23^2057) % 25 = 16 := 
by
  sorry

end remainder_23_2057_mod_25_l286_28695


namespace no_integers_exist_l286_28698

theorem no_integers_exist :
  ¬ (∃ x y : ℤ, (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2) :=
by
  sorry

end no_integers_exist_l286_28698


namespace ratio_d_e_l286_28667

theorem ratio_d_e (a b c d e f : ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  d / e = 1 / 4 :=
sorry

end ratio_d_e_l286_28667


namespace power_of_expression_l286_28674

theorem power_of_expression (a b c d e : ℝ)
  (h1 : a - b - c + d = 18)
  (h2 : a + b - c - d = 6)
  (h3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 :=
by
  sorry

end power_of_expression_l286_28674


namespace find_a_l286_28666

-- Define the constants b and the asymptote equation
def asymptote_eq (x y : ℝ) := 3 * x + 2 * y = 0

-- Define the hyperbola equation and the condition
def hyperbola_eq (x y a : ℝ) := x^2 / a^2 - y^2 / 9 = 1
def hyperbola_condition (a : ℝ) := a > 0

-- Theorem stating the value of a given the conditions
theorem find_a (a : ℝ) (hcond : hyperbola_condition a) 
  (h_asymp : ∀ x y : ℝ, asymptote_eq x y → y = -(3/2) * x) :
  a = 2 := 
sorry

end find_a_l286_28666


namespace min_initial_bags_l286_28632

theorem min_initial_bags :
  ∃ x : ℕ, (∃ y : ℕ, (y + 90 = 2 * (x - 90) ∧ x + (11 * x - 1620) / 7 = 6 * (2 * x - 270 - (11 * x - 1620) / 7))
             ∧ x = 153) :=
by { sorry }

end min_initial_bags_l286_28632


namespace monthly_growth_rate_optimal_selling_price_l286_28648

-- Conditions
def april_sales : ℕ := 150
def june_sales : ℕ := 216
def cost_price_per_unit : ℕ := 30
def initial_selling_price : ℕ := 40
def initial_sales_vol : ℕ := 300
def sales_decrease_rate : ℕ := 10
def desired_profit : ℕ := 3960

-- Questions (Proof statements)
theorem monthly_growth_rate :
  ∃ (x : ℝ), (1 + x) ^ 2 = (june_sales:ℝ) / (april_sales:ℝ) ∧ x = 0.2 := by
  sorry

theorem optimal_selling_price :
  ∃ (y : ℝ), (y - cost_price_per_unit) * (initial_sales_vol - sales_decrease_rate * (y - initial_selling_price)) = desired_profit ∧ y = 48 := by
  sorry

end monthly_growth_rate_optimal_selling_price_l286_28648


namespace probability_shots_result_l286_28611

open ProbabilityTheory

noncomputable def P_A := 3 / 4
noncomputable def P_B := 4 / 5
noncomputable def P_not_A := 1 - P_A
noncomputable def P_not_B := 1 - P_B

theorem probability_shots_result :
    (P_not_A * P_not_B * P_A) + (P_not_A * P_not_B * P_not_A * P_B) = 19 / 400 :=
    sorry

end probability_shots_result_l286_28611


namespace visitors_answered_questionnaire_l286_28680

theorem visitors_answered_questionnaire (V : ℕ) (h : (3 / 4 : ℝ) * V = (V : ℝ) - 110) : V = 440 :=
sorry

end visitors_answered_questionnaire_l286_28680


namespace parabola_focus_value_of_a_l286_28636

theorem parabola_focus_value_of_a :
  (∀ a : ℝ, (∃ y : ℝ, y = a * (0^2) ∧ (0, y) = (0, 3 / 8)) → a = 2 / 3) := by
sorry

end parabola_focus_value_of_a_l286_28636


namespace at_least_one_not_solved_l286_28602

theorem at_least_one_not_solved (p q : Prop) : (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by sorry

end at_least_one_not_solved_l286_28602


namespace value_of_expression_l286_28649

theorem value_of_expression (x y z : ℕ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 1) : 
  3 * x - 2 * y + 4 * z = 9 := 
by
  sorry

end value_of_expression_l286_28649


namespace count_solid_circles_among_first_2006_l286_28692

-- Definition of the sequence sum for location calculation
def sequence_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2 - 1

-- Main theorem
theorem count_solid_circles_among_first_2006 : 
  ∃ n : ℕ, sequence_sum (n - 1) < 2006 ∧ 2006 ≤ sequence_sum n ∧ n = 62 :=
by {
  sorry
}

end count_solid_circles_among_first_2006_l286_28692


namespace matinee_receipts_l286_28633

theorem matinee_receipts :
  let child_ticket_cost := 4.50
  let adult_ticket_cost := 6.75
  let num_children := 48
  let num_adults := num_children - 20
  total_receipts = num_children * child_ticket_cost + num_adults * adult_ticket_cost :=
by 
  sorry

end matinee_receipts_l286_28633


namespace cylinder_surface_area_l286_28627

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area (h r : ℕ) (h_eq : h = 8) (r_eq : r = 3) :
  2 * Real.pi * r * h + 2 * Real.pi * r ^ 2 = 66 * Real.pi := by
  sorry

end cylinder_surface_area_l286_28627


namespace necessary_but_not_sufficient_condition_l286_28655

variable {a b : ℝ}

theorem necessary_but_not_sufficient_condition
    (h1 : a ≠ 0)
    (h2 : b ≠ 0) :
    (a^2 + b^2 ≥ 2 * a * b) → 
    (¬(a^2 + b^2 ≥ 2 * a * b) → ¬(a / b + b / a ≥ 2)) ∧ 
    ((a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2 * a * b)) :=
sorry

end necessary_but_not_sufficient_condition_l286_28655


namespace simplify_expr_l286_28696

noncomputable def expr1 : ℝ := 3 * Real.sqrt 8 / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7)
noncomputable def expr2 : ℝ := -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7)

theorem simplify_expr : expr1 = expr2 := by
  sorry

end simplify_expr_l286_28696


namespace sequence_formula_l286_28604

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n →  1 / a (n + 1) = 1 / a n + 1) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by {
  sorry
}

end sequence_formula_l286_28604


namespace average_weight_of_children_l286_28675

theorem average_weight_of_children 
  (average_weight_boys : ℝ)
  (number_of_boys : ℕ)
  (average_weight_girls : ℝ)
  (number_of_girls : ℕ)
  (total_children : ℕ)
  (average_weight_children : ℝ) :
  average_weight_boys = 160 →
  number_of_boys = 8 →
  average_weight_girls = 130 →
  number_of_girls = 6 →
  total_children = number_of_boys + number_of_girls →
  average_weight_children = 
    (number_of_boys * average_weight_boys + number_of_girls * average_weight_girls) / total_children →
  average_weight_children = 147 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_weight_of_children_l286_28675


namespace regular_octagon_interior_angle_l286_28662

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l286_28662


namespace math_and_science_students_l286_28659

theorem math_and_science_students (x y : ℕ) 
  (h1 : x + y + 2 = 30)
  (h2 : y = 3 * x + 4) :
  y - 2 = 20 :=
by {
  sorry
}

end math_and_science_students_l286_28659


namespace division_of_repeating_decimals_l286_28668

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end division_of_repeating_decimals_l286_28668


namespace wendy_dentist_bill_l286_28683

theorem wendy_dentist_bill : 
  let cost_cleaning := 70
  let cost_filling := 120
  let num_fillings := 3
  let cost_root_canal := 400
  let cost_dental_crown := 600
  let total_bill := 9 * cost_root_canal
  let known_costs := cost_cleaning + (num_fillings * cost_filling) + cost_root_canal + cost_dental_crown
  let cost_tooth_extraction := total_bill - known_costs
  cost_tooth_extraction = 2170 := by
  sorry

end wendy_dentist_bill_l286_28683


namespace average_age_increase_l286_28653

variable (A : ℝ) -- Original average age of 8 men
variable (age1 age2 : ℝ) -- The ages of the two men being replaced
variable (avg_women : ℝ) -- The average age of the two women

-- Conditions as hypotheses
def conditions : Prop :=
  8 * A - age1 - age2 + avg_women * 2 = 8 * (A + 2)

-- The theorem that needs to be proved
theorem average_age_increase (h1 : age1 = 20) (h2 : age2 = 28) (h3 : avg_women = 32) (h4 : conditions A age1 age2 avg_women) : (8 * A + 16) / 8 - A = 2 :=
by
  sorry

end average_age_increase_l286_28653


namespace four_clique_exists_in_tournament_l286_28654

open Finset

/-- Given a graph G with 9 vertices and 28 edges, prove that G contains a 4-clique. -/
theorem four_clique_exists_in_tournament 
  (V : Finset ℕ) (E : Finset (ℕ × ℕ)) 
  (hV : V.card = 9) 
  (hE : E.card = 28) :
  ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ (v₁ v₂ : ℕ), v₁ ∈ S → v₂ ∈ S → v₁ ≠ v₂ → (v₁, v₂) ∈ E ∨ (v₂, v₁) ∈ E :=
sorry

end four_clique_exists_in_tournament_l286_28654


namespace sequence_a4_eq_5_over_3_l286_28665

theorem sequence_a4_eq_5_over_3 :
  ∀ (a : ℕ → ℚ), a 1 = 1 → (∀ n > 1, a n = 1 / a (n - 1) + 1) → a 4 = 5 / 3 :=
by
  intro a ha1 H
  sorry

end sequence_a4_eq_5_over_3_l286_28665


namespace solve_for_x_l286_28693

theorem solve_for_x (x : ℝ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -(2 / 11) :=
by
  sorry

end solve_for_x_l286_28693


namespace total_cherry_tomatoes_l286_28699

-- Definitions based on the conditions
def cherryTomatoesPerJar : Nat := 8
def numberOfJars : Nat := 7

-- The statement we want to prove
theorem total_cherry_tomatoes : cherryTomatoesPerJar * numberOfJars = 56 := by
  sorry

end total_cherry_tomatoes_l286_28699


namespace solve_trig_eq_l286_28610

open Real

theorem solve_trig_eq (k : ℤ) : 
  (∃ x : ℝ, 
    (|cos x| + cos (3 * x)) / (sin x * cos (2 * x)) = -2 * sqrt 3 
    ∧ (x = -π/6 + 2 * k * π ∨ x = 2 * π/3 + 2 * k * π ∨ x = 7 * π/6 + 2 * k * π)) :=
sorry

end solve_trig_eq_l286_28610


namespace factor_x6_plus_8_l286_28691

theorem factor_x6_plus_8 : (x^2 + 2) ∣ (x^6 + 8) :=
by
  sorry

end factor_x6_plus_8_l286_28691


namespace pow_mod_sub_remainder_l286_28634

theorem pow_mod_sub_remainder :
  (10^23 - 7) % 6 = 3 :=
sorry

end pow_mod_sub_remainder_l286_28634


namespace three_digit_number_l286_28697

/-- 
Prove there exists three-digit number N such that 
1. N is of form 100a + 10b + c
2. 1 ≤ a ≤ 9
3. 0 ≤ b, c ≤ 9
4. N = 11 * (a + b + c)
--/
theorem three_digit_number (N a b c : ℕ) 
  (hN: N = 100 * a + 10 * b + c) 
  (h_a: 1 ≤ a ∧ a ≤ 9)
  (h_b_c: 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9)
  (h_condition: N = 11 * (a + b + c)) :
  N = 198 := 
sorry

end three_digit_number_l286_28697


namespace photo_album_slots_l286_28607

def photos_from_cristina : Nat := 7
def photos_from_john : Nat := 10
def photos_from_sarah : Nat := 9
def photos_from_clarissa : Nat := 14

theorem photo_album_slots :
  photos_from_cristina + photos_from_john + photos_from_sarah + photos_from_clarissa = 40 :=
by
  sorry

end photo_album_slots_l286_28607


namespace somu_present_age_l286_28637

variable (S F : ℕ)

-- Conditions from the problem
def condition1 : Prop := S = F / 3
def condition2 : Prop := S - 10 = (F - 10) / 5

-- The statement we need to prove
theorem somu_present_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 20 := 
by sorry

end somu_present_age_l286_28637


namespace range_of_m_for_hyperbola_l286_28651

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ u v : ℝ, (∀ x y : ℝ, x^2/(m+2) + y^2/(m+1) = 1) → (m > -2) ∧ (m < -1)) := by
  sorry

end range_of_m_for_hyperbola_l286_28651


namespace men_to_complete_work_l286_28622

theorem men_to_complete_work (x : ℕ) (h1 : 10 * 80 = x * 40) : x = 20 :=
by
  sorry

end men_to_complete_work_l286_28622


namespace sum_of_distinct_integers_l286_28616

theorem sum_of_distinct_integers (a b c d : ℤ) (h : (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25) (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : a + b + c + d = 4 :=
by
    sorry

end sum_of_distinct_integers_l286_28616


namespace center_determines_position_l286_28652

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for the Circle's position being determined by its center.
theorem center_determines_position (c : Circle) : c.center = c.center :=
by
  sorry

end center_determines_position_l286_28652


namespace sin_cos_inequality_l286_28621

theorem sin_cos_inequality (α : ℝ) 
  (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  (Real.pi / 3 < α ∧ α < 4 * Real.pi / 3) :=
sorry

end sin_cos_inequality_l286_28621


namespace solution_l286_28613

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution (x : ℝ) : g (g x) = g x ↔ x = 0 ∨ x = 4 ∨ x = 5 ∨ x = -1 :=
by
  sorry

end solution_l286_28613


namespace range_of_m_l286_28638

theorem range_of_m (m x : ℝ) :
  (m-1 < x ∧ x < m+1) → (2 < x ∧ x < 6) → (3 ≤ m ∧ m ≤ 5) :=
by
  intros hp hq
  sorry

end range_of_m_l286_28638


namespace total_marbles_l286_28647

-- Definitions based on the given conditions
def jars : ℕ := 16
def pots : ℕ := jars / 2
def marbles_in_jar : ℕ := 5
def marbles_in_pot : ℕ := 3 * marbles_in_jar

-- Main statement to be proved
theorem total_marbles : 
  5 * jars + marbles_in_pot * pots = 200 := 
by
  sorry

end total_marbles_l286_28647
