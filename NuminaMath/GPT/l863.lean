import Mathlib

namespace hyperbola_equation_l863_86304

theorem hyperbola_equation
  (a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (focus_at_five : a^2 + b^2 = 25) 
  (asymptote_ratio : b / a = 3 / 4) :
  (a = 4 ∧ b = 3 ∧ ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1) ↔ ( ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 ):=
sorry 

end hyperbola_equation_l863_86304


namespace transformed_ellipse_l863_86351

-- Define the original equation and the transformation
def orig_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def trans_x (x' : ℝ) : ℝ := x' / 5
noncomputable def trans_y (y' : ℝ) : ℝ := y' / 4

-- Prove that the transformed equation is an ellipse with specified properties
theorem transformed_ellipse :
  (∃ x' y' : ℝ, (trans_x x')^2 + (trans_y y')^2 = 1) →
  ∃ a b : ℝ, (a = 10) ∧ (b = 8) ∧ (∀ x' y' : ℝ, x'^2 / (a/2)^2 + y'^2 / (b/2)^2 = 1) :=
sorry

end transformed_ellipse_l863_86351


namespace bobby_consumption_l863_86342

theorem bobby_consumption :
  let initial_candy := 28
  let additional_candy_portion := 3/4 * 42
  let chocolate_portion := 1/2 * 63
  initial_candy + additional_candy_portion + chocolate_portion = 91 := 
by {
  let initial_candy : ℝ := 28
  let additional_candy_portion : ℝ := 3/4 * 42
  let chocolate_portion : ℝ := 1/2 * 63
  sorry
}

end bobby_consumption_l863_86342


namespace point_in_second_quadrant_l863_86329

theorem point_in_second_quadrant (m : ℝ) (h : 2 > 0 ∧ m < 0) : m < 0 :=
by
  sorry

end point_in_second_quadrant_l863_86329


namespace multiply_correct_l863_86393

theorem multiply_correct : 2.4 * 0.2 = 0.48 := by
  sorry

end multiply_correct_l863_86393


namespace hexagonal_H5_find_a_find_t_find_m_l863_86382

section problem1

-- Define the hexagonal number formula
def hexagonal_number (n : ℕ) : ℕ :=
  2 * n^2 - n

-- Define that H_5 should equal 45
theorem hexagonal_H5 : hexagonal_number 5 = 45 := sorry

end problem1

section problem2

variables (a b c : ℕ)

-- Given hexagonal number equations
def H1 := a + b + c
def H2 := 4 * a + 2 * b + c
def H3 := 9 * a + 3 * b + c

-- Conditions given in problem
axiom H1_def : H1 = 1
axiom H2_def : H2 = 7
axiom H3_def : H3 = 19

-- Prove that a = 3
theorem find_a : a = 3 := sorry

end problem2

section problem3

variables (p q r t : ℕ)

-- Given ratios in problem
axiom ratio1 : p * 3 = 2 * q
axiom ratio2 : q * 5 = 4 * r

-- Prove that t = 12
theorem find_t : t = 12 := sorry

end problem3

section problem4

variables (x y m : ℕ)

-- Given proportional conditions
axiom ratio3 : x * 3 = y * 4
axiom ratio4 : (x + y) * 3 = x * m

-- Prove that m = 7
theorem find_m : m = 7 := sorry

end problem4

end hexagonal_H5_find_a_find_t_find_m_l863_86382


namespace remaining_lawn_mowing_l863_86311

-- Definitions based on the conditions in the problem.
def Mary_mowing_time : ℝ := 3  -- Mary can mow the lawn in 3 hours
def John_mowing_time : ℝ := 6  -- John can mow the lawn in 6 hours
def John_work_time : ℝ := 3    -- John works for 3 hours

-- Question: How much of the lawn remains to be mowed?
theorem remaining_lawn_mowing : (Mary_mowing_time = 3) ∧ (John_mowing_time = 6) ∧ (John_work_time = 3) →
  (1 - (John_work_time / John_mowing_time) = 1 / 2) :=
by
  sorry

end remaining_lawn_mowing_l863_86311


namespace division_result_l863_86345

theorem division_result : 180 / 6 / 3 / 2 = 5 := by
  sorry

end division_result_l863_86345


namespace find_d_minus_b_l863_86396

theorem find_d_minus_b (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^5 = b^4) (h2 : c^3 = d^2) (h3 : c - a = 19) : d - b = 757 := 
by sorry

end find_d_minus_b_l863_86396


namespace not_possible_perimeter_l863_86392

theorem not_possible_perimeter :
  ∀ (x : ℝ), 13 < x ∧ x < 37 → ¬ (37 + x = 50) :=
by
  intros x h
  sorry

end not_possible_perimeter_l863_86392


namespace distribute_tickets_among_people_l863_86387

noncomputable def distribution_ways : ℕ := 84

theorem distribute_tickets_among_people (tickets : Fin 5 → ℕ) (persons : Fin 4 → ℕ)
  (h1 : ∀ p : Fin 4, ∃ t : Fin 5, tickets t = persons p)
  (h2 : ∀ p : Fin 4, ∀ t1 t2 : Fin 5, tickets t1 = persons p ∧ tickets t2 = persons p → (t1.val + 1 = t2.val ∨ t2.val + 1 = t1.val)) :
  ∃ n : ℕ, n = distribution_ways := by
  use 84
  trivial

end distribute_tickets_among_people_l863_86387


namespace approximation_example1_approximation_example2_approximation_example3_l863_86347

theorem approximation_example1 (α β : ℝ) (hα : α = 0.0023) (hβ : β = 0.0057) :
  (1 + α) * (1 + β) = 1.008 := sorry

theorem approximation_example2 (α β : ℝ) (hα : α = 0.05) (hβ : β = -0.03) :
  (1 + α) * (10 + β) = 10.02 := sorry

theorem approximation_example3 (α β γ : ℝ) (hα : α = 0.03) (hβ : β = -0.01) (hγ : γ = -0.02) :
  (1 + α) * (1 + β) * (1 + γ) = 1 := sorry

end approximation_example1_approximation_example2_approximation_example3_l863_86347


namespace determine_N_l863_86370

/-- 
Each row and two columns in the grid forms distinct arithmetic sequences.
Given:
- First column values: 10 and 18 (arithmetic sequence).
- Second column top value: N, bottom value: -23 (arithmetic sequence).
Prove that N = -15.
 -/
theorem determine_N : ∃ N : ℤ, (∀ n : ℕ, 10 + n * 8 = 10 ∨ 10 + n * 8 = 18) ∧ (∀ m : ℕ, N + m * 8 = N ∨ N + m * 8 = -23) ∧ N = -15 :=
by {
  sorry
}

end determine_N_l863_86370


namespace product_of_prs_l863_86374

theorem product_of_prs
  (p r s : ℕ)
  (H1 : 4 ^ p + 4 ^ 3 = 272)
  (H2 : 3 ^ r + 27 = 54)
  (H3 : 2 ^ (s + 2) + 10 = 42) : 
  p * r * s = 27 :=
sorry

end product_of_prs_l863_86374


namespace magazine_cost_l863_86322

theorem magazine_cost (C M : ℝ) 
  (h1 : 4 * C = 8 * M) 
  (h2 : 12 * C = 24) : 
  M = 1 :=
by
  sorry

end magazine_cost_l863_86322


namespace range_of_m_cond_l863_86333

noncomputable def quadratic_inequality (x m : ℝ) : Prop :=
  x^2 + m * x + 2 * m - 3 ≥ 0

theorem range_of_m_cond (m : ℝ) (h1 : 2 ≤ m) (h2 : m ≤ 6) (x : ℝ) :
  quadratic_inequality x m :=
sorry

end range_of_m_cond_l863_86333


namespace positive_number_l863_86365

theorem positive_number (n : ℕ) (h : n^2 + 2 * n = 170) : n = 12 :=
sorry

end positive_number_l863_86365


namespace geometric_sequence_sum_l863_86355

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l863_86355


namespace Fermat_numbers_are_not_cubes_l863_86386

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem Fermat_numbers_are_not_cubes : ∀ n : ℕ, ¬ ∃ k : ℕ, F n = k^3 :=
by
  sorry

end Fermat_numbers_are_not_cubes_l863_86386


namespace C_increases_with_n_l863_86364

noncomputable def C (e n R r : ℝ) : ℝ := (e * n) / (R + n * r)

theorem C_increases_with_n (e R r : ℝ) (h_e : 0 < e) (h_R : 0 < R) (h_r : 0 < r) :
  ∀ {n₁ n₂ : ℝ}, 0 < n₁ → n₁ < n₂ → C e n₁ R r < C e n₂ R r :=
by
  sorry

end C_increases_with_n_l863_86364


namespace inner_cube_surface_area_l863_86395

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l863_86395


namespace find_triangle_sides_l863_86303

noncomputable def triangle_sides (x : ℝ) : Prop :=
  let a := x - 2
  let b := x
  let c := x + 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a + 2 = b ∧ b + 2 = c ∧ area = 6 ∧
  a = 2 * Real.sqrt 6 - 2 ∧
  b = 2 * Real.sqrt 6 ∧
  c = 2 * Real.sqrt 6 + 2

theorem find_triangle_sides :
  ∃ x : ℝ, triangle_sides x := by
  sorry

end find_triangle_sides_l863_86303


namespace card_statements_true_l863_86316

def statement1 (statements : Fin 5 → Prop) : Prop :=
  ∃! i, i < 5 ∧ statements i

def statement2 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ i ≠ j ∧ statements i ∧ statements j) ∧ ¬(∃ h k l, h < 5 ∧ k < 5 ∧ l < 5 ∧ h ≠ k ∧ h ≠ l ∧ k ≠ l ∧ statements h ∧ statements k ∧ statements l)

def statement3 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k, i < 5 ∧ j < 5 ∧ k < 5 ∧ i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ statements i ∧ statements j ∧ statements k) ∧ ¬(∃ a b c d, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ statements a ∧ statements b ∧ statements c ∧ statements d)

def statement4 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k l, i < 5 ∧ j < 5 ∧ k < 5 ∧ l < 5 ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ statements i ∧ statements j ∧ statements k ∧ statements l) ∧ ¬(∃ a b c d e, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ e < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ statements a ∧ statements b ∧ statements c ∧ statements d ∧ statements e)

def statement5 (statements : Fin 5 → Prop) : Prop :=
  ∀ i, i < 5 → statements i

theorem card_statements_true : ∃ (statements : Fin 5 → Prop), 
  statement1 statements ∨ statement2 statements ∨ statement3 statements ∨ statement4 statements ∨ statement5 statements 
  ∧ statement3 statements := 
sorry

end card_statements_true_l863_86316


namespace pool_capacity_percentage_l863_86384

noncomputable def hose_rate := 60 -- cubic feet per minute
noncomputable def pool_width := 80 -- feet
noncomputable def pool_length := 150 -- feet
noncomputable def pool_depth := 10 -- feet
noncomputable def drainage_time := 2000 -- minutes
noncomputable def pool_volume := pool_width * pool_length * pool_depth -- cubic feet
noncomputable def removed_water_volume := hose_rate * drainage_time -- cubic feet

theorem pool_capacity_percentage :
  (removed_water_volume / pool_volume) * 100 = 100 :=
by
  -- the proof steps would go here
  sorry

end pool_capacity_percentage_l863_86384


namespace negation_of_exactly_one_even_l863_86390

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

theorem negation_of_exactly_one_even :
  ¬ exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
                                 (is_even a ∧ is_even b) ∨
                                 (is_even a ∧ is_even c) ∨
                                 (is_even b ∧ is_even c) :=
by sorry

end negation_of_exactly_one_even_l863_86390


namespace change_is_13_82_l863_86369

def sandwich_cost : ℝ := 5
def num_sandwiches : ℕ := 3
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05
def payment : ℝ := 20 + 5 + 3

def total_cost_before_discount : ℝ := num_sandwiches * sandwich_cost
def discount_amount : ℝ := total_cost_before_discount * discount_rate
def discounted_cost : ℝ := total_cost_before_discount - discount_amount
def tax_amount : ℝ := discounted_cost * tax_rate
def total_cost_after_tax : ℝ := discounted_cost + tax_amount

def change (payment total_cost : ℝ) : ℝ := payment - total_cost

theorem change_is_13_82 : change payment total_cost_after_tax = 13.82 := 
by
  -- Proof will be provided here
  sorry

end change_is_13_82_l863_86369


namespace cost_per_order_of_pakoras_l863_86348

noncomputable def samosa_cost : ℕ := 2
noncomputable def samosa_count : ℕ := 3
noncomputable def mango_lassi_cost : ℕ := 2
noncomputable def pakora_count : ℕ := 4
noncomputable def tip_percentage : ℚ := 0.25
noncomputable def total_cost_with_tax : ℚ := 25

theorem cost_per_order_of_pakoras (P : ℚ)
  (h1 : samosa_cost * samosa_count = 6)
  (h2 : mango_lassi_cost = 2)
  (h3 : 1.25 * (samosa_cost * samosa_count + mango_lassi_cost + pakora_count * P) = total_cost_with_tax) :
  P = 3 :=
by
  -- sorry ⟹ sorry
  sorry

end cost_per_order_of_pakoras_l863_86348


namespace true_statement_for_f_l863_86388

variable (c : ℝ) (f : ℝ → ℝ)

theorem true_statement_for_f :
  (∀ x : ℝ, f x = x^2 - 2 * x + c) → (∀ x : ℝ, f x ≥ c - 1) :=
by
  sorry

end true_statement_for_f_l863_86388


namespace proof_problem_l863_86362

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < (π / 2))
variable (htan : tan α = (1 + sin β) / cos β)

theorem proof_problem : 2 * α - β = π / 2 :=
by
  sorry

end proof_problem_l863_86362


namespace remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l863_86318

def p (x : ℝ) : ℝ := x^3 - 4 * x^2 + 3 * x + 2

theorem remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1 :
  p 1 = 2 := by
  -- solution needed, for now we put a placeholder
  sorry

end remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l863_86318


namespace at_least_one_non_negative_l863_86327

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 := 
sorry

end at_least_one_non_negative_l863_86327


namespace arithmetic_sequence_proof_l863_86368

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_proof
  (a₁ d : ℝ)
  (h : a 4 a₁ d + a 6 a₁ d + a 8 a₁ d + a 10 a₁ d + a 12 a₁ d = 120) :
  a 7 a₁ d - (1 / 3) * a 5 a₁ d = 16 :=
by
  sorry

end arithmetic_sequence_proof_l863_86368


namespace remainder_modulo_l863_86378

theorem remainder_modulo (N k q r : ℤ) (h1 : N = 1423 * k + 215) (h2 : N = 109 * q + r) : 
  (N - q ^ 2) % 109 = 106 := by
  sorry

end remainder_modulo_l863_86378


namespace initial_bacteria_count_l863_86336

theorem initial_bacteria_count (doubling_time : ℕ) (initial_time : ℕ) (initial_bacteria : ℕ) 
(final_bacteria : ℕ) (doubling_rate : initial_time / doubling_time = 8 ∧ final_bacteria = 524288) : 
  initial_bacteria = 2048 :=
by
  sorry

end initial_bacteria_count_l863_86336


namespace angle_A_is_pi_over_4_l863_86309

theorem angle_A_is_pi_over_4
  (A B C : ℝ)
  (a b c : ℝ)
  (h : a^2 = b^2 + c^2 - 2 * b * c * Real.sin A) :
  A = Real.pi / 4 :=
  sorry

end angle_A_is_pi_over_4_l863_86309


namespace alpha_beta_sum_pi_over_2_l863_86312

theorem alpha_beta_sum_pi_over_2 (α β : ℝ) (hα : 0 < α) (hα_lt : α < π / 2) (hβ : 0 < β) (hβ_lt : β < π / 2) (h : Real.sin (α + β) = Real.sin α ^ 2 + Real.sin β ^ 2) : α + β = π / 2 :=
by
  -- Proof steps would go here
  sorry

end alpha_beta_sum_pi_over_2_l863_86312


namespace sufficient_but_not_necessary_condition_l863_86377

variable (a₁ d : ℝ)

def S₄ := 4 * a₁ + 6 * d
def S₅ := 5 * a₁ + 10 * d
def S₆ := 6 * a₁ + 15 * d

theorem sufficient_but_not_necessary_condition (h : d > 1) :
  S₄ a₁ d + S₆ a₁ d > 2 * S₅ a₁ d :=
by
  -- proof omitted
  sorry

end sufficient_but_not_necessary_condition_l863_86377


namespace highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l863_86335

-- Definitions for cumulative visitors entering and leaving
def y (x : ℕ) : ℕ := 850 * x + 100
def z (x : ℕ) : ℕ := 200 * x - 200

-- Definition for total number of visitors at time x
def w (x : ℕ) : ℕ := y x - z x

-- Proof problem statements
theorem highest_visitors_at_4pm :
  ∀x, x ≤ 9 → w 9 ≥ w x :=
sorry

theorem yellow_warning_time_at_12_30pm :
  ∃x, w x = 2600 :=
sorry

end highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l863_86335


namespace find_a4_l863_86363

theorem find_a4 (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n - 3) : a 4 = -8 :=
by {
  sorry
}

end find_a4_l863_86363


namespace amount_over_budget_l863_86385

-- Define the prices of each item
def cost_necklace_A : ℕ := 34
def cost_necklace_B : ℕ := 42
def cost_necklace_C : ℕ := 50
def cost_first_book := cost_necklace_A + 20
def cost_second_book := cost_necklace_C - 10

-- Define Bob's budget
def budget : ℕ := 100

-- Define the total cost
def total_cost := cost_necklace_A + cost_necklace_B + cost_necklace_C + cost_first_book + cost_second_book

-- Prove the amount over budget
theorem amount_over_budget : total_cost - budget = 120 := by
  sorry

end amount_over_budget_l863_86385


namespace half_radius_circle_y_l863_86360

-- Conditions
def circle_x_circumference (C : ℝ) : Prop :=
  C = 20 * Real.pi

def circle_x_and_y_same_area (r R : ℝ) : Prop :=
  Real.pi * r^2 = Real.pi * R^2

-- Problem statement: Prove that half the radius of circle y is 5
theorem half_radius_circle_y (r R : ℝ) (hx : circle_x_circumference (2 * Real.pi * r)) (hy : circle_x_and_y_same_area r R) : R / 2 = 5 :=
by sorry

end half_radius_circle_y_l863_86360


namespace paper_clips_distribution_l863_86341

theorem paper_clips_distribution (total_clips : ℕ) (num_boxes : ℕ) (clip_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : num_boxes = 9) : clip_per_box = 9 :=
by sorry

end paper_clips_distribution_l863_86341


namespace min_value_of_expr_l863_86306

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  ((x^2 + 1 / y^2 + 1) * (x^2 + 1 / y^2 - 1000)) +
  ((y^2 + 1 / x^2 + 1) * (y^2 + 1 / x^2 - 1000))

theorem min_value_of_expr :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ min_value_expr x y = -498998 :=
by
  sorry

end min_value_of_expr_l863_86306


namespace odd_expression_is_odd_l863_86397

theorem odd_expression_is_odd (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : (4 * p * q + 1) % 2 = 1 :=
sorry

end odd_expression_is_odd_l863_86397


namespace wire_attachment_distance_l863_86307

theorem wire_attachment_distance :
  ∃ x : ℝ, 
    (∀ z y : ℝ, z = Real.sqrt (x ^ 2 + 3.6 ^ 2) ∧ y = Real.sqrt ((x + 5) ^ 2 + 3.6 ^ 2) →
      z + y = 13) ∧
    abs ((x : ℝ) - 2.7) < 0.01 := -- Assuming numerical closeness within a small epsilon for practical solutions.
sorry -- Proof not provided.

end wire_attachment_distance_l863_86307


namespace functional_equation_solution_l863_86320

open Function

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x ^ 2 + f y) = y + f x ^ 2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end functional_equation_solution_l863_86320


namespace inequality_solution_l863_86361

-- Define the inequality condition
def fraction_inequality (x : ℝ) : Prop :=
  (3 * x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 / 3 ≤ x ∧ x < 2

-- The theorem to prove that the inequality's solution matches the given solution set
theorem inequality_solution (x : ℝ) (h : fraction_inequality x) : solution_set x :=
  sorry

end inequality_solution_l863_86361


namespace g_inverse_sum_l863_86373

-- Define the function g and its inverse
def g (x : ℝ) : ℝ := x ^ 3
noncomputable def g_inv (y : ℝ) : ℝ := y ^ (1/3 : ℝ)

-- State the theorem to be proved
theorem g_inverse_sum : g_inv 8 + g_inv (-64) = -2 := by 
  sorry

end g_inverse_sum_l863_86373


namespace total_amount_invested_l863_86325

variable (T : ℝ)

def income_first (T : ℝ) : ℝ :=
  0.10 * (T - 700)

def income_second : ℝ :=
  0.08 * 700

theorem total_amount_invested :
  income_first T - income_second = 74 → T = 2000 :=
by
  intros h
  sorry 

end total_amount_invested_l863_86325


namespace shortest_path_l863_86331

noncomputable def diameter : ℝ := 18
noncomputable def radius : ℝ := diameter / 2
noncomputable def AC : ℝ := 7
noncomputable def BD : ℝ := 7
noncomputable def CD : ℝ := diameter - AC - BD
noncomputable def CP : ℝ := Real.sqrt (radius ^ 2 - (CD / 2) ^ 2)
noncomputable def DP : ℝ := CP

theorem shortest_path (C P D : ℝ) :
  (C - 7) ^ 2 + (D - 7) ^ 2 = CD ^ 2 →
  (C = AC) ∧ (D = BD) →
  2 * CP = 2 * Real.sqrt 77 :=
by
  intros h1 h2
  sorry

end shortest_path_l863_86331


namespace garden_area_l863_86376

theorem garden_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end garden_area_l863_86376


namespace quotient_of_large_div_small_l863_86381

theorem quotient_of_large_div_small (L S : ℕ) (h1 : L - S = 1365)
  (h2 : L = S * (L / S) + 20) (h3 : L = 1634) : (L / S) = 6 := by
  sorry

end quotient_of_large_div_small_l863_86381


namespace initial_fish_l863_86391

-- Define the conditions of the problem
def fish_bought : Float := 280.0
def current_fish : Float := 492.0

-- Define the question to be proved
theorem initial_fish (x : Float) (h : x + fish_bought = current_fish) : x = 212 :=
by 
  sorry

end initial_fish_l863_86391


namespace paula_candies_distribution_l863_86301

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l863_86301


namespace mark_total_cents_l863_86346

theorem mark_total_cents (dimes nickels : ℕ) (h1 : nickels = dimes + 3) (h2 : dimes = 5) : 
  dimes * 10 + nickels * 5 = 90 := by
  sorry

end mark_total_cents_l863_86346


namespace junior_girls_count_l863_86340

theorem junior_girls_count 
  (total_players : ℕ) 
  (boys_percentage : ℝ) 
  (junior_girls : ℕ)
  (h_team : total_players = 50)
  (h_boys_pct : boys_percentage = 0.6)
  (h_junior_girls : junior_girls = ((total_players : ℝ) * (1 - boys_percentage) * 0.5)) : 
  junior_girls = 10 := 
by 
  sorry

end junior_girls_count_l863_86340


namespace problem_divisibility_l863_86328

theorem problem_divisibility (n : ℕ) : ∃ k : ℕ, 2 ^ (3 ^ n) + 1 = 3 ^ (n + 1) * k :=
sorry

end problem_divisibility_l863_86328


namespace solve_for_M_l863_86332

def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 2 * x + y = 2 ∧ x - y = 1 }

theorem solve_for_M : M = { (1, 0) } := by
  sorry

end solve_for_M_l863_86332


namespace probability_X_eq_Y_l863_86334

theorem probability_X_eq_Y
  (x y : ℝ)
  (h1 : -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi)
  (h2 : -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi)
  (h3 : Real.cos (Real.cos x) = Real.cos (Real.cos y)) :
  (∃ N : ℕ, N = 100 ∧ ∃ M : ℕ, M = 11 ∧ M / N = (11 : ℝ) / 100) :=
by sorry

end probability_X_eq_Y_l863_86334


namespace new_water_intake_recommendation_l863_86357

noncomputable def current_consumption : ℝ := 25
noncomputable def increase_percentage : ℝ := 0.75
noncomputable def increased_amount : ℝ := increase_percentage * current_consumption
noncomputable def new_recommended_consumption : ℝ := current_consumption + increased_amount

theorem new_water_intake_recommendation :
  new_recommended_consumption = 43.75 := 
by 
  sorry

end new_water_intake_recommendation_l863_86357


namespace probability_white_ball_is_two_fifths_l863_86372

-- Define the total number of each type of balls.
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Calculate the total number of balls in the bag.
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability calculation.
noncomputable def probability_of_white_ball : ℚ := white_balls / total_balls

-- The theorem statement asserting the probability of drawing a white ball.
theorem probability_white_ball_is_two_fifths :
  probability_of_white_ball = 2 / 5 :=
sorry

end probability_white_ball_is_two_fifths_l863_86372


namespace complex_value_of_product_l863_86350

theorem complex_value_of_product (r : ℂ) (hr : r^7 = 1) (hr1 : r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := 
by sorry

end complex_value_of_product_l863_86350


namespace baseball_batter_at_bats_left_l863_86356

theorem baseball_batter_at_bats_left (L R H_L H_R : ℕ) (h1 : L + R = 600)
    (h2 : H_L + H_R = 192) (h3 : H_L = 25 / 100 * L) (h4 : H_R = 35 / 100 * R) : 
    L = 180 :=
by
  sorry

end baseball_batter_at_bats_left_l863_86356


namespace smallest_n_terminating_contains_9_l863_86380

def isTerminatingDecimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2 ^ a * 5 ^ b

def containsDigit9 (n : ℕ) : Prop :=
  (Nat.digits 10 n).contains 9

theorem smallest_n_terminating_contains_9 : ∃ n : ℕ, 
  isTerminatingDecimal n ∧
  containsDigit9 n ∧
  (∀ m : ℕ, isTerminatingDecimal m ∧ containsDigit9 m → n ≤ m) ∧
  n = 5120 :=
  sorry

end smallest_n_terminating_contains_9_l863_86380


namespace z_max_plus_z_min_l863_86394

theorem z_max_plus_z_min {x y z : ℝ} 
  (h1 : x^2 + y^2 + z^2 = 3) 
  (h2 : x + 2 * y - 2 * z = 4) : 
  z + z = -4 :=
by 
  sorry

end z_max_plus_z_min_l863_86394


namespace flight_height_l863_86323

theorem flight_height (flights : ℕ) (step_height_in_inches : ℕ) (total_steps : ℕ) 
    (H1 : flights = 9) (H2 : step_height_in_inches = 18) (H3 : total_steps = 60) : 
    (total_steps * step_height_in_inches) / 12 / flights = 10 :=
by
  sorry

end flight_height_l863_86323


namespace dividend_rate_correct_l863_86305

-- Define the stock's yield and market value
def stock_yield : ℝ := 0.08
def market_value : ℝ := 175

-- Dividend rate definition based on given yield and market value
def dividend_rate (yield market_value : ℝ) : ℝ :=
  (yield * market_value)

-- The problem statement to be proven in Lean
theorem dividend_rate_correct :
  dividend_rate stock_yield market_value = 14 := by
  sorry

end dividend_rate_correct_l863_86305


namespace comic_cost_is_4_l863_86343

-- Define initial amount of money Raul had.
def initial_money : ℕ := 87

-- Define number of comics bought by Raul.
def num_comics : ℕ := 8

-- Define the amount of money left after buying comics.
def money_left : ℕ := 55

-- Define the hypothesis condition about the money spent.
def total_spent : ℕ := initial_money - money_left

-- Define the main assertion that each comic cost $4.
def cost_per_comic (total_spent : ℕ) (num_comics : ℕ) : Prop :=
  total_spent / num_comics = 4

-- Main theorem statement
theorem comic_cost_is_4 : cost_per_comic total_spent num_comics :=
by
  -- Here we're skipping the proof for this exercise.
  sorry

end comic_cost_is_4_l863_86343


namespace percentage_deducted_from_list_price_l863_86321

-- Definitions based on conditions
def cost_price : ℝ := 85.5
def marked_price : ℝ := 112.5
def profit_rate : ℝ := 0.25 -- 25% profit

noncomputable def selling_price : ℝ := cost_price * (1 + profit_rate)

theorem percentage_deducted_from_list_price:
  ∃ d : ℝ, d = 5 ∧ selling_price = marked_price * (1 - d / 100) :=
by
  sorry

end percentage_deducted_from_list_price_l863_86321


namespace circle_equation_center_line_l863_86359

theorem circle_equation_center_line (x y : ℝ) :
  -- Conditions
  (∀ (x1 y1 : ℝ), x1 + y1 - 2 = 0 → (x = 1 ∧ y = 1)) ∧
  ((x - 1)^2 + (y - 1)^2 = 4) ∧
  -- Points A and B
  (∀ (xA yA : ℝ), xA = 1 ∧ yA = -1 ∨ xA = -1 ∧ yA = 1 →
    ((xA - x)^2 + (yA - y)^2 = 4)) :=
by
  sorry

end circle_equation_center_line_l863_86359


namespace solution_1_solution_2_l863_86337

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (2 * x + 3)

lemma f_piecewise (x : ℝ) : 
  f x = if x ≤ -3 / 2 then -4 * x - 2
        else if -3 / 2 < x ∧ x < 1 / 2 then 4
        else 4 * x + 2 := 
by
-- This lemma represents the piecewise definition of f(x)
sorry

theorem solution_1 : 
  (∀ x : ℝ, f x < 5 ↔ (-7 / 4 < x ∧ x < 3 / 4)) := 
by 
-- Proof of the inequality solution
sorry

theorem solution_2 : 
  (∀ t : ℝ, (∀ x : ℝ, f x - t ≥ 0) → t ≤ 4) :=
by
-- Proof that the maximum value of t is 4
sorry

end solution_1_solution_2_l863_86337


namespace field_trip_buses_l863_86330

-- Definitions of conditions
def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def seats_per_bus : ℕ := 72

-- Total calculations
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_people : ℕ := total_students + total_chaperones
def buses_needed : ℕ := (total_people + seats_per_bus - 1) / seats_per_bus

theorem field_trip_buses : buses_needed = 6 := by
  unfold buses_needed
  unfold total_people total_students total_chaperones chaperones_per_grade
  norm_num
  sorry

end field_trip_buses_l863_86330


namespace soda_cost_l863_86317

theorem soda_cost (b s f : ℕ) (h1 : 3 * b + 2 * s + 2 * f = 590) (h2 : 2 * b + 3 * s + f = 610) : s = 140 :=
sorry

end soda_cost_l863_86317


namespace same_terminal_side_angles_l863_86352

theorem same_terminal_side_angles (α : ℝ) : 
  (∃ k : ℤ, α = -457 + k * 360) ↔ (∃ k : ℤ, α = 263 + k * 360) :=
sorry

end same_terminal_side_angles_l863_86352


namespace zoey_preparation_months_l863_86338
open Nat

-- Define months as integers assuming 1 = January, 5 = May, 9 = September, etc.
def month_start : ℕ := 5 -- May
def month_exam : ℕ := 9 -- September

-- The function to calculate the number of months of preparation excluding the exam month.
def months_of_preparation (start : ℕ) (exam : ℕ) : ℕ := (exam - start)

theorem zoey_preparation_months :
  months_of_preparation month_start month_exam = 4 := by
  sorry

end zoey_preparation_months_l863_86338


namespace tan_150_eq_neg_inv_sqrt3_l863_86324

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l863_86324


namespace cookies_left_over_l863_86371

def abigail_cookies : Nat := 53
def beatrice_cookies : Nat := 65
def carson_cookies : Nat := 26
def pack_size : Nat := 10

theorem cookies_left_over : (abigail_cookies + beatrice_cookies + carson_cookies) % pack_size = 4 := 
by
  sorry

end cookies_left_over_l863_86371


namespace donuts_eaten_on_monday_l863_86302

theorem donuts_eaten_on_monday (D : ℕ) (h1 : D + D / 2 + 4 * D = 49) : 
  D = 9 :=
sorry

end donuts_eaten_on_monday_l863_86302


namespace monotonically_decreasing_intervals_max_and_min_values_on_interval_l863_86379

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem monotonically_decreasing_intervals (a : ℝ) : 
  ∀ x : ℝ, (x < -1 ∨ x > 3) → f x a < f (x+1) a :=
sorry

theorem max_and_min_values_on_interval : 
  (f (-1) (-2) = -7) ∧ (max (f (-2) (-2)) (f 2 (-2)) = 20) :=
sorry

end monotonically_decreasing_intervals_max_and_min_values_on_interval_l863_86379


namespace kay_exercise_time_l863_86344

variable (A W : ℕ)
variable (exercise_total : A + W = 250) 
variable (ratio_condition : A * 2 = 3 * W)

theorem kay_exercise_time :
  A = 150 ∧ W = 100 :=
by
  sorry

end kay_exercise_time_l863_86344


namespace cube_volume_in_pyramid_and_cone_l863_86375

noncomputable def volume_of_cube
  (base_side : ℝ)
  (pyramid_height : ℝ)
  (cone_radius : ℝ)
  (cone_height : ℝ)
  (cube_side_length : ℝ) : ℝ := 
  cube_side_length^3

theorem cube_volume_in_pyramid_and_cone :
  let base_side := 2
  let pyramid_height := Real.sqrt 3
  let cone_radius := Real.sqrt 2
  let cone_height := Real.sqrt 3
  let cube_side_length := (Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3)
  volume_of_cube base_side pyramid_height cone_radius cone_height cube_side_length = (6 * Real.sqrt 6) / 17 :=
by sorry

end cube_volume_in_pyramid_and_cone_l863_86375


namespace simplify_and_evaluate_l863_86366

noncomputable def simplifyExpression (a : ℚ) : ℚ :=
  (a - 3 + (1 / (a - 1))) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2))

theorem simplify_and_evaluate
  (h : ∀ a, a ∈ [-2, -1, 0, 1, 2]) :
  ∀ a, (a - 1) ≠ 0 → a ≠ 0 → a ≠ 2  →
  simplifyExpression a = a / (a - 1) ∧ simplifyExpression (-1) = 1 / 2 :=
by
  intro a ha_ne_zero ha_ne_two
  sorry

end simplify_and_evaluate_l863_86366


namespace true_proposition_B_l863_86353

theorem true_proposition_B : (3 > 4) ∨ (3 < 4) :=
sorry

end true_proposition_B_l863_86353


namespace geometric_sequence_a5_l863_86349

theorem geometric_sequence_a5 {a : ℕ → ℝ} 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := 
sorry

end geometric_sequence_a5_l863_86349


namespace total_chocolate_bars_l863_86399

theorem total_chocolate_bars :
  let num_large_boxes := 45
  let num_small_boxes_per_large_box := 36
  let num_chocolate_bars_per_small_box := 72
  num_large_boxes * num_small_boxes_per_large_box * num_chocolate_bars_per_small_box = 116640 :=
by
  sorry

end total_chocolate_bars_l863_86399


namespace sum_ge_six_l863_86398

theorem sum_ge_six (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 :=
by
  sorry

end sum_ge_six_l863_86398


namespace buttons_on_first_type_of_shirt_l863_86308

/--
The GooGoo brand of clothing manufactures two types of shirts.
- The first type of shirt has \( x \) buttons.
- The second type of shirt has 5 buttons.
- The department store ordered 200 shirts of each type.
- A total of 1600 buttons are used for the entire order.

Prove that the first type of shirt has exactly 3 buttons.
-/
theorem buttons_on_first_type_of_shirt (x : ℕ) 
  (h1 : 200 * x + 200 * 5 = 1600) : 
  x = 3 :=
  sorry

end buttons_on_first_type_of_shirt_l863_86308


namespace network_structure_l863_86339

theorem network_structure 
  (n : ℕ)
  (is_acquainted : Fin n → Fin n → Prop)
  (H_symmetric : ∀ x y, is_acquainted x y = is_acquainted y x) 
  (H_common_acquaintance : ∀ x y, ¬ is_acquainted x y → ∃! z : Fin n, is_acquainted x z ∧ is_acquainted y z) :
  ∃ (G : SimpleGraph (Fin n)), (∀ x y, G.Adj x y = is_acquainted x y) ∧
    (∀ x y, ¬ G.Adj x y → (∃ (z1 z2 : Fin n), G.Adj x z1 ∧ G.Adj y z1 ∧ G.Adj x z2 ∧ G.Adj y z2)) :=
by
  sorry

end network_structure_l863_86339


namespace inequality_satisfaction_l863_86367

theorem inequality_satisfaction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / y + 1 / x + y ≥ y / x + 1 / y + x) ↔ 
  ((x = y) ∨ (x = 1 ∧ y ≠ 0) ∨ (y = 1 ∧ x ≠ 0)) ∧ (x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end inequality_satisfaction_l863_86367


namespace determine_v6_l863_86389

variable (v : ℕ → ℝ)

-- Given initial conditions: v₄ = 12 and v₇ = 471
def initial_conditions := v 4 = 12 ∧ v 7 = 471

-- Recurrence relation definition: vₙ₊₂ = 3vₙ₊₁ + vₙ
def recurrence_relation := ∀ n : ℕ, v (n + 2) = 3 * v (n + 1) + v n

-- The target is to prove that v₆ = 142.5
theorem determine_v6 (h1 : initial_conditions v) (h2 : recurrence_relation v) : 
  v 6 = 142.5 :=
sorry

end determine_v6_l863_86389


namespace Jill_talking_time_total_l863_86300

-- Definition of the sequence of talking times
def talking_time : ℕ → ℕ 
| 0 => 5
| (n+1) => 2 * talking_time n

-- The statement we need to prove
theorem Jill_talking_time_total :
  (talking_time 0) + (talking_time 1) + (talking_time 2) + (talking_time 3) + (talking_time 4) = 155 :=
by
  sorry

end Jill_talking_time_total_l863_86300


namespace sequence_sum_l863_86319

def alternating_sum : List ℤ := [2, -7, 10, -15, 18, -23, 26, -31, 34, -39, 40, -45, 48]

theorem sequence_sum : alternating_sum.sum = 13 := by
  sorry

end sequence_sum_l863_86319


namespace cos_90_eq_zero_l863_86313

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l863_86313


namespace sequence_term_500_l863_86383

theorem sequence_term_500 (a : ℕ → ℤ) (h1 : a 1 = 3009) (h2 : a 2 = 3010) 
  (h3 : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 500 = 3341 := 
sorry

end sequence_term_500_l863_86383


namespace find_value_of_d_l863_86358

theorem find_value_of_d
  (a b c d : ℕ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : c < d) 
  (h5 : ab + bc + ac = abc) 
  (h6 : abc = d) : 
  d = 36 := 
sorry

end find_value_of_d_l863_86358


namespace pairs_of_old_roller_skates_l863_86310

def cars := 2
def bikes := 2
def trash_can := 1
def tricycle := 1
def car_wheels := 4
def bike_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def total_wheels := 25

def roller_skates_wheels := 2
def skates_per_pair := 2

theorem pairs_of_old_roller_skates : (total_wheels - (cars * car_wheels + bikes * bike_wheels + trash_can * trash_can_wheels + tricycle * tricycle_wheels)) / roller_skates_wheels / skates_per_pair = 2 := by
  sorry

end pairs_of_old_roller_skates_l863_86310


namespace range_of_a_l863_86314

theorem range_of_a
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x + 2 * y + 4 = 4 * x * y)
  (h2 : ∀ a : ℝ, (x + 2 * y) * a ^ 2 + 2 * a + 2 * x * y - 34 ≥ 0) : 
  ∀ a : ℝ, a ≤ -3 ∨ a ≥ 5 / 2 :=
by
  sorry

end range_of_a_l863_86314


namespace line_quadrants_l863_86354

theorem line_quadrants (k b : ℝ) (h : ∃ x y : ℝ, y = k * x + b ∧ 
                                          ((x > 0 ∧ y > 0) ∧   -- First quadrant
                                           (x < 0 ∧ y < 0) ∧   -- Third quadrant
                                           (x > 0 ∧ y < 0))) : -- Fourth quadrant
  k > 0 :=
sorry

end line_quadrants_l863_86354


namespace tom_should_pay_times_original_price_l863_86326

-- Definitions of the given conditions
def original_price : ℕ := 3
def amount_paid : ℕ := 9

-- The theorem to prove
theorem tom_should_pay_times_original_price : ∃ k : ℕ, amount_paid = k * original_price ∧ k = 3 :=
by 
  -- Using sorry to skip the proof for now
  sorry

end tom_should_pay_times_original_price_l863_86326


namespace tan_alpha_minus_beta_l863_86315

theorem tan_alpha_minus_beta
  (α β : ℝ)
  (tan_alpha : Real.tan α = 2)
  (tan_beta : Real.tan β = -7) :
  Real.tan (α - β) = -9 / 13 :=
by sorry

end tan_alpha_minus_beta_l863_86315
