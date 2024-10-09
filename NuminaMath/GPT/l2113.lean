import Mathlib

namespace alpha_plus_beta_l2113_211360

theorem alpha_plus_beta (α β : ℝ) 
  (hα : 0 < α ∧ α < Real.pi / 2) 
  (hβ : 0 < β ∧ β < Real.pi / 2)
  (h_sin_alpha : Real.sin α = Real.sqrt 10 / 10)
  (h_cos_beta : Real.cos β = 2 * Real.sqrt 5 / 5) :
  α + β = Real.pi / 4 :=
sorry

end alpha_plus_beta_l2113_211360


namespace original_price_second_store_l2113_211377

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

end original_price_second_store_l2113_211377


namespace ratio_Florence_Rene_l2113_211357

theorem ratio_Florence_Rene :
  ∀ (I F R : ℕ), R = 300 → F = k * R → I = 1/3 * (F + R + I) → F + R + I = 1650 → F / R = 3 / 2 := 
by 
  sorry

end ratio_Florence_Rene_l2113_211357


namespace marks_difference_is_140_l2113_211347

noncomputable def marks_difference (P C M : ℕ) : ℕ :=
  (P + C + M) - P

theorem marks_difference_is_140 (P C M : ℕ) (h1 : (C + M) / 2 = 70) :
  marks_difference P C M = 140 := by
  sorry

end marks_difference_is_140_l2113_211347


namespace solve_problems_l2113_211301

variable (initial_problems : ℕ) 
variable (additional_problems : ℕ)

theorem solve_problems
  (h1 : initial_problems = 12) 
  (h2 : additional_problems = 7) : 
  initial_problems + additional_problems = 19 := 
by 
  sorry

end solve_problems_l2113_211301


namespace no_integer_solution_l2113_211328

theorem no_integer_solution (y : ℤ) : ¬ (-3 * y ≥ y + 9 ∧ 2 * y ≥ 14 ∧ -4 * y ≥ 2 * y + 21) :=
sorry

end no_integer_solution_l2113_211328


namespace circle_area_in_square_centimeters_l2113_211323

theorem circle_area_in_square_centimeters (d_meters : ℤ) (h : d_meters = 8) :
  ∃ (A : ℤ), A = 160000 * Real.pi ∧ 
  A = π * (d_meters / 2) ^ 2 * 10000 :=
by
  sorry

end circle_area_in_square_centimeters_l2113_211323


namespace ab_is_4_l2113_211391

noncomputable def ab_value (a b : ℝ) : ℝ :=
  8 / (0.5 * (8 / a) * (8 / b))

theorem ab_is_4 (a b : ℝ) (ha : a > 0) (hb : b > 0) (area_condition : ab_value a b = 8) : a * b = 4 :=
  by
  sorry

end ab_is_4_l2113_211391


namespace unique_solution_values_l2113_211336

theorem unique_solution_values (x y a : ℝ) :
  (∀ x y a, x^2 + y^2 + 2 * x ≤ 1 ∧ x - y + a = 0) → (a = -1 ∨ a = 3) :=
by
  intro h
  sorry

end unique_solution_values_l2113_211336


namespace solve_for_x_l2113_211321

theorem solve_for_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end solve_for_x_l2113_211321


namespace geom_seq_proof_l2113_211381

noncomputable def geom_seq (a q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n - 1)

variables {a q : ℝ}

theorem geom_seq_proof (h1 : geom_seq a q 7 = 4) (h2 : geom_seq a q 5 + geom_seq a q 9 = 10) :
  geom_seq a q 3 + geom_seq a q 11 = 17 :=
by
  sorry

end geom_seq_proof_l2113_211381


namespace Will_Had_28_Bottles_l2113_211330

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

end Will_Had_28_Bottles_l2113_211330


namespace star_value_l2113_211307

variable (a b : ℤ)
noncomputable def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem star_value
  (h1 : a + b = 11)
  (h2 : a * b = 24)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0) :
  star a b = 11 / 24 := by
  sorry

end star_value_l2113_211307


namespace sequence_general_formula_l2113_211312

theorem sequence_general_formula (a : ℕ → ℕ) 
    (h₀ : a 1 = 3) 
    (h : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) : 
    ∀ n : ℕ, a n = 2^(n+1) - 1 :=
by 
  sorry

end sequence_general_formula_l2113_211312


namespace find_min_n_l2113_211317

theorem find_min_n (n k : ℕ) (h : 14 * n = k^2) : n = 14 := sorry

end find_min_n_l2113_211317


namespace flat_fee_l2113_211303

theorem flat_fee (f n : ℝ) 
  (h1 : f + 3 * n = 205) 
  (h2 : f + 6 * n = 350) : 
  f = 60 := 
by
  sorry

end flat_fee_l2113_211303


namespace jessica_has_three_dozens_of_red_marbles_l2113_211338

-- Define the number of red marbles Sandy has
def sandy_red_marbles : ℕ := 144

-- Define the relationship between Sandy's and Jessica's red marbles
def relationship (jessica_red_marbles : ℕ) : Prop :=
  sandy_red_marbles = 4 * jessica_red_marbles

-- Define the question to find out how many dozens of red marbles Jessica has
def jessica_dozens (jessica_red_marbles : ℕ) := jessica_red_marbles / 12

-- Theorem stating that given the conditions, Jessica has 3 dozens of red marbles
theorem jessica_has_three_dozens_of_red_marbles (jessica_red_marbles : ℕ)
  (h : relationship jessica_red_marbles) : jessica_dozens jessica_red_marbles = 3 :=
by
  -- The proof is omitted
  sorry

end jessica_has_three_dozens_of_red_marbles_l2113_211338


namespace optimal_addition_amount_l2113_211341

def optimal_material_range := {x : ℝ | 100 ≤ x ∧ x ≤ 200}

def second_trial_amounts := {x : ℝ | x = 138.2 ∨ x = 161.8}

theorem optimal_addition_amount (
  h1 : ∀ x ∈ optimal_material_range, x ∈ second_trial_amounts
  ) :
  138.2 ∈ second_trial_amounts ∧ 161.8 ∈ second_trial_amounts :=
by
  sorry

end optimal_addition_amount_l2113_211341


namespace grid_permutation_exists_l2113_211364

theorem grid_permutation_exists (n : ℕ) (grid : Fin n → Fin n → ℤ) 
  (cond1 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = 1)
  (cond2 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = -1)
  (cond3 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = 1)
  (cond4 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = -1)
  (cond5 : ∀ i j, grid i j = 0 ∨ grid i j = 1 ∨ grid i j = -1) :
  ∃ (perm_rows perm_cols : Fin n → Fin n),
    (∀ i j, grid (perm_rows i) (perm_cols j) = -grid i j) :=
by
  -- Proof goes here
  sorry

end grid_permutation_exists_l2113_211364


namespace geometric_sequence_ratio_l2113_211345

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

-- Definitions based on given conditions
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement
theorem geometric_sequence_ratio :
  is_geometric_seq a q →
  q = -1/3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros
  sorry

end geometric_sequence_ratio_l2113_211345


namespace find_number_l2113_211350

theorem find_number (x : ℤ) (h : 3 * x + 4 = 19) : x = 5 :=
by {
  sorry
}

end find_number_l2113_211350


namespace percentage_returned_l2113_211387

theorem percentage_returned (R : ℕ) (S : ℕ) (total : ℕ) (least_on_lot : ℕ) (max_rented : ℕ)
  (h1 : total = 20) (h2 : least_on_lot = 10) (h3 : max_rented = 20) (h4 : R = 20) (h5 : S ≥ 10) :
  (S / R) * 100 ≥ 50 := sorry

end percentage_returned_l2113_211387


namespace bombardment_deaths_l2113_211361

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

end bombardment_deaths_l2113_211361


namespace angle_of_inclination_l2113_211388

noncomputable def line_slope (a b : ℝ) : ℝ := 1  -- The slope of the line y = x + 1 is 1
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m -- angle of inclination is arctan of the slope

theorem angle_of_inclination (θ : ℝ) : 
  inclination_angle (line_slope 1 1) = 45 :=
by
  sorry

end angle_of_inclination_l2113_211388


namespace triangle_perimeter_l2113_211316

def triangle_side_lengths : ℕ × ℕ × ℕ := (10, 6, 7)

def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter (a b c : ℕ) (h : (a, b, c) = triangle_side_lengths) : 
  perimeter a b c = 23 := by
  -- We formulate the statement and leave the proof for later
  sorry

end triangle_perimeter_l2113_211316


namespace colored_paper_distribution_l2113_211342

theorem colored_paper_distribution (F M : ℕ) (h1 : F + M = 24) (h2 : M = 2 * F) (total_sheets : ℕ) (distributed_sheets : total_sheets = 48) : 
  (48 / F) = 6 := by
  sorry

end colored_paper_distribution_l2113_211342


namespace system_of_equations_solution_l2113_211302

theorem system_of_equations_solution (x y z : ℝ) 
  (h : ∀ (n : ℕ), x * (1 - 1 / 2^(n : ℝ)) + y * (1 - 1 / 2^(n+1 : ℝ)) + z * (1 - 1 / 2^(n+2 : ℝ)) = 0) : 
  y = -3 * x ∧ z = 2 * x :=
sorry

end system_of_equations_solution_l2113_211302


namespace max_value_of_expression_l2113_211332

noncomputable def max_expression_value (a b c : ℝ) : ℝ :=
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2)))

theorem max_value_of_expression (a b c : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) (hc : -1 < c ∧ c < 1) :
  max_expression_value a b c ≤ 2 :=
by sorry

end max_value_of_expression_l2113_211332


namespace decreasing_power_function_on_interval_l2113_211367

noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem decreasing_power_function_on_interval (m : ℝ) :
  (∀ x : ℝ, (0 < x) -> power_function m x < 0) ↔ m = -1 := 
by 
  sorry

end decreasing_power_function_on_interval_l2113_211367


namespace max_value_f_l2113_211385

noncomputable def op_add (a b : ℝ) : ℝ :=
if a >= b then a else b^2

noncomputable def f (x : ℝ) : ℝ :=
(op_add 1 x) + (op_add 2 x)

theorem max_value_f :
  ∃ x ∈ Set.Icc (-2 : ℝ) 3, ∀ y ∈ Set.Icc (-2 : ℝ) 3, f y ≤ f x := 
sorry

end max_value_f_l2113_211385


namespace trees_planted_in_garden_l2113_211309

theorem trees_planted_in_garden (yard_length : ℕ) (tree_distance : ℕ) (h₁ : yard_length = 500) (h₂ : tree_distance = 20) :
  ((yard_length / tree_distance) + 1) = 26 :=
by
  -- The proof goes here
  sorry

end trees_planted_in_garden_l2113_211309


namespace part1_part2_l2113_211393

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

end part1_part2_l2113_211393


namespace solve_for_y_l2113_211305

theorem solve_for_y (y : ℕ) (h : 9^y = 3^14) : y = 7 := 
by
  sorry

end solve_for_y_l2113_211305


namespace c_ge_one_l2113_211386

theorem c_ge_one (a b : ℕ) (c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (a + 1) / (b + c) = b / a) : c ≥ 1 := 
sorry

end c_ge_one_l2113_211386


namespace least_integer_to_multiple_of_3_l2113_211339

theorem least_integer_to_multiple_of_3 : ∃ n : ℕ, n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ m : ℕ, m > 0 → (527 + m) % 3 = 0 → m ≥ n :=
sorry

end least_integer_to_multiple_of_3_l2113_211339


namespace initial_people_in_gym_l2113_211376

variable (W A : ℕ)

theorem initial_people_in_gym (W A : ℕ) (h : W + A + 5 + 2 - 3 - 4 + 2 = 20) : W + A = 18 := by
  sorry

end initial_people_in_gym_l2113_211376


namespace part_one_part_two_l2113_211356
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

end part_one_part_two_l2113_211356


namespace range_m_l2113_211351

variable {x m : ℝ}

theorem range_m (h1 : m / (1 - x) - 2 / (x - 1) = 1) (h2 : x ≥ 0) (h3 : x ≠ 1) : m ≤ -1 ∧ m ≠ -2 := 
sorry

end range_m_l2113_211351


namespace parabola_directrix_l2113_211372

theorem parabola_directrix (x y : ℝ) (h_eqn : y = -3 * x^2 + 6 * x - 5) :
  y = -23 / 12 :=
sorry

end parabola_directrix_l2113_211372


namespace find_b_minus_a_l2113_211325

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

end find_b_minus_a_l2113_211325


namespace parallel_vectors_m_eq_neg3_l2113_211300

theorem parallel_vectors_m_eq_neg3 : 
  ∀ m : ℝ, (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (1 + m, 1 - m) → a.1 * b.2 - a.2 * b.1 = 0) → m = -3 :=
by
  intros m h_par
  specialize h_par (1, -2) (1 + m, 1 - m) rfl rfl
  -- We need to show m = -3
  sorry

end parallel_vectors_m_eq_neg3_l2113_211300


namespace jakes_weight_l2113_211380

theorem jakes_weight (J S B : ℝ) 
  (h1 : 0.8 * J = 2 * S)
  (h2 : J + S = 168)
  (h3 : B = 1.25 * (J + S))
  (h4 : J + S + B = 221) : 
  J = 120 :=
by
  sorry

end jakes_weight_l2113_211380


namespace cryptarithm_solution_l2113_211368

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

end cryptarithm_solution_l2113_211368


namespace sum_of_triangulars_15_to_20_l2113_211362

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_triangulars_15_to_20 : 
  (triangular_number 15 + triangular_number 16 + triangular_number 17 + triangular_number 18 + triangular_number 19 + triangular_number 20) = 980 :=
by
  sorry

end sum_of_triangulars_15_to_20_l2113_211362


namespace max_value_frac_inv_sum_l2113_211395

theorem max_value_frac_inv_sum (x y : ℝ) (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b)
  (h3 : a^x = 6) (h4 : b^y = 6) (h5 : a + b = 2 * Real.sqrt 6) :
  ∃ m, m = 1 ∧ (∀ x y a b, (1 < a) → (1 < b) → (a^x = 6) → (b^y = 6) → (a + b = 2 * Real.sqrt 6) → 
  (∃ n, (n = (1/x + 1/y)) → n ≤ m)) :=
by
  sorry

end max_value_frac_inv_sum_l2113_211395


namespace gcd_min_value_l2113_211313

theorem gcd_min_value {a b c : ℕ} (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (gcd_ab : Nat.gcd a b = 210) (gcd_ac : Nat.gcd a c = 770) : Nat.gcd b c = 10 :=
sorry

end gcd_min_value_l2113_211313


namespace pow_div_pow_eq_l2113_211346

theorem pow_div_pow_eq :
  (3^12) / (27^2) = 729 :=
by
  -- We'll use the provided conditions and proof outline
  -- 1. 27 = 3^3
  -- 2. (a^b)^c = a^{bc}
  -- 3. a^b \div a^c = a^{b-c}
  sorry

end pow_div_pow_eq_l2113_211346


namespace lottery_not_guaranteed_to_win_l2113_211378

theorem lottery_not_guaranteed_to_win (total_tickets : ℕ) (winning_rate : ℚ) (num_purchased : ℕ) :
  total_tickets = 100000 ∧ winning_rate = 1 / 1000 ∧ num_purchased = 2000 → 
  ∃ (outcome : ℕ), outcome = 0 := by
  sorry

end lottery_not_guaranteed_to_win_l2113_211378


namespace intersection_points_x_axis_vertex_on_line_inequality_c_l2113_211365

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

end intersection_points_x_axis_vertex_on_line_inequality_c_l2113_211365


namespace f_2019_eq_2019_l2113_211308

def f : ℝ → ℝ := sorry

axiom f_pos : ∀ x, x > 0 → f x > 0
axiom f_one : f 1 = 1
axiom f_eq : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

theorem f_2019_eq_2019 : f 2019 = 2019 :=
by sorry

end f_2019_eq_2019_l2113_211308


namespace intersection_point_l2113_211394

variable (x y : ℝ)

-- Definitions given by the conditions
def line1 (x y : ℝ) := 3 * y = -2 * x + 6
def line2 (x y : ℝ) := -2 * y = 6 * x + 4

-- The theorem we want to prove
theorem intersection_point : ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ x = -12/7 ∧ y = 22/7 := 
sorry

end intersection_point_l2113_211394


namespace trigonometric_identity_l2113_211331

theorem trigonometric_identity :
  (1 / 2 - (Real.cos (15 * Real.pi / 180)) ^ 2) = - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_identity_l2113_211331


namespace find_a9_l2113_211352

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

end find_a9_l2113_211352


namespace evaluate_polynomial_at_2_l2113_211355

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 1) = 31 := 
by 
  sorry

end evaluate_polynomial_at_2_l2113_211355


namespace initial_cars_l2113_211382

theorem initial_cars (X : ℕ) : (X - 13 + (13 + 5) = 85) → (X = 80) :=
by
  sorry

end initial_cars_l2113_211382


namespace value_at_zero_eq_sixteen_l2113_211363

-- Define the polynomial P(x)
def P (x : ℚ) : ℚ := x ^ 4 - 20 * x ^ 2 + 16

-- Theorem stating the value of P(0)
theorem value_at_zero_eq_sixteen :
  P 0 = 16 :=
by
-- We know the polynomial P(x) is x^4 - 20x^2 + 16
-- When x = 0, P(0) = 0^4 - 20 * 0^2 + 16 = 16
sorry

end value_at_zero_eq_sixteen_l2113_211363


namespace amber_age_l2113_211322

theorem amber_age 
  (a g : ℕ)
  (h1 : g = 15 * a)
  (h2 : g - a = 70) :
  a = 5 :=
by
  sorry

end amber_age_l2113_211322


namespace given_polynomial_l2113_211375

noncomputable def f (x : ℝ) := x^3 - 2

theorem given_polynomial (x : ℝ) : 
  8 * f (x^3) - x^6 * f (2 * x) - 2 * f (x^2) + 12 = 0 :=
by
  sorry

end given_polynomial_l2113_211375


namespace num_isosceles_right_triangles_in_ellipse_l2113_211324

theorem num_isosceles_right_triangles_in_ellipse
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ ∃ t : ℝ, (x, y) = (a * Real.cos t, b * Real.sin t))
  :
  (∃ n : ℕ,
    (n = 3 ∧ a > Real.sqrt 3 * b) ∨
    (n = 1 ∧ (b < a ∧ a ≤ Real.sqrt 3 * b))
  ) :=
sorry

end num_isosceles_right_triangles_in_ellipse_l2113_211324


namespace number_of_pups_in_second_round_l2113_211354

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

end number_of_pups_in_second_round_l2113_211354


namespace minimize_expression_pos_int_l2113_211306

theorem minimize_expression_pos_int (n : ℕ) (hn : 0 < n) : 
  (∀ m : ℕ, 0 < m → (m / 3 + 27 / m : ℝ) ≥ (9 / 3 + 27 / 9)) :=
sorry

end minimize_expression_pos_int_l2113_211306


namespace pure_imaginary_value_l2113_211397

theorem pure_imaginary_value (a : ℝ) : (z = (0 : ℝ) + (a^2 + 2 * a - 3) * I) → (a = 0 ∨ a = -2) :=
by
  sorry

end pure_imaginary_value_l2113_211397


namespace dogwood_tree_count_l2113_211315

def initial_dogwoods : ℕ := 34
def additional_dogwoods : ℕ := 49
def total_dogwoods : ℕ := initial_dogwoods + additional_dogwoods

theorem dogwood_tree_count :
  total_dogwoods = 83 :=
by
  -- omitted proof
  sorry

end dogwood_tree_count_l2113_211315


namespace sandy_phone_bill_expense_l2113_211390

def sandy_age_now (kim_age : ℕ) : ℕ := 3 * (kim_age + 2) - 2

def sandy_phone_bill (sandy_age : ℕ) : ℕ := 10 * sandy_age

theorem sandy_phone_bill_expense
  (kim_age : ℕ)
  (kim_age_condition : kim_age = 10)
  : sandy_phone_bill (sandy_age_now kim_age) = 340 := by
  sorry

end sandy_phone_bill_expense_l2113_211390


namespace opposite_of_neg_two_l2113_211399

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l2113_211399


namespace value_of_a_minus_b_l2113_211314

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

end value_of_a_minus_b_l2113_211314


namespace rectangle_ratio_width_length_l2113_211320

variable (w : ℝ)

theorem rectangle_ratio_width_length (h1 : w + 8 + w + 8 = 24) : 
  w / 8 = 1 / 2 :=
by
  sorry

end rectangle_ratio_width_length_l2113_211320


namespace find_b_l2113_211340

theorem find_b 
  (a b : ℚ)
  (h_root : (1 + Real.sqrt 5) ^ 3 + a * (1 + Real.sqrt 5) ^ 2 + b * (1 + Real.sqrt 5) - 60 = 0) :
  b = 26 :=
sorry

end find_b_l2113_211340


namespace task_completion_choice_l2113_211369

theorem task_completion_choice (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 := by
  sorry

end task_completion_choice_l2113_211369


namespace find_length_AX_l2113_211383

theorem find_length_AX 
  (A B C X : Type)
  (BC BX AC : ℝ)
  (h_BC : BC = 36)
  (h_BX : BX = 30)
  (h_AC : AC = 27)
  (h_bisector : ∃ (x : ℝ), x = BX / BC ∧ x = AX / AC ) :
  ∃ AX : ℝ, AX = 22.5 := 
sorry

end find_length_AX_l2113_211383


namespace geom_seq_common_ratio_q_l2113_211379

-- Define the geometric sequence
def geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- State the theorem
theorem geom_seq_common_ratio_q {a₁ q : ℝ} :
  (a₁ = 2) → (geom_seq a₁ q 4 = 16) → (q = 2) :=
by
  intros h₁ h₂
  sorry

end geom_seq_common_ratio_q_l2113_211379


namespace expenses_notation_l2113_211304

theorem expenses_notation (income expense : ℤ) (h_income : income = 6) (h_expense : -expense = income) : expense = -4 := 
by
  sorry

end expenses_notation_l2113_211304


namespace find_FC_l2113_211318

theorem find_FC 
(DC CB AD ED FC : ℝ)
(h1 : DC = 7) 
(h2 : CB = 8) 
(h3 : AB = (1 / 4) * AD)
(h4 : ED = (4 / 5) * AD) : 
FC = 10.4 :=
sorry

end find_FC_l2113_211318


namespace debby_remaining_pictures_l2113_211333

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (deleted_pictures : ℕ)

def initial_pictures (zoo_pictures museum_pictures : ℕ) : ℕ :=
  zoo_pictures + museum_pictures

def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (initial_pictures zoo_pictures museum_pictures) - deleted_pictures

theorem debby_remaining_pictures :
  remaining_pictures 24 12 14 = 22 :=
by
  sorry

end debby_remaining_pictures_l2113_211333


namespace sale_in_third_month_l2113_211326

def grocer_sales (s1 s2 s4 s5 s6 : ℕ) (average : ℕ) (num_months : ℕ) (total_sales : ℕ) : Prop :=
  s1 = 5266 ∧ s2 = 5768 ∧ s4 = 5678 ∧ s5 = 6029 ∧ s6 = 4937 ∧ average = 5600 ∧ num_months = 6 ∧ total_sales = average * num_months

theorem sale_in_third_month
  (s1 s2 s4 s5 s6 total_sales : ℕ)
  (h : grocer_sales s1 s2 s4 s5 s6 5600 6 total_sales) :
  ∃ s3 : ℕ, total_sales - (s1 + s2 + s4 + s5 + s6) = s3 ∧ s3 = 5922 := 
by {
  sorry
}

end sale_in_third_month_l2113_211326


namespace simplify_fraction_l2113_211334

theorem simplify_fraction (h1 : 222 = 2 * 3 * 37) (h2 : 8888 = 8 * 11 * 101) :
  (222 / 8888) * 22 = 1 / 2 :=
by
  sorry

end simplify_fraction_l2113_211334


namespace find_a_2b_l2113_211319

theorem find_a_2b 
  (a b : ℤ) 
  (h1 : a * b = -150) 
  (h2 : a + b = -23) : 
  a + 2 * b = -55 :=
sorry

end find_a_2b_l2113_211319


namespace central_angle_is_2_radians_l2113_211358

namespace CircleAngle

def radius : ℝ := 2
def arc_length : ℝ := 4

theorem central_angle_is_2_radians : arc_length / radius = 2 := by
  sorry

end CircleAngle

end central_angle_is_2_radians_l2113_211358


namespace constant_term_expansion_l2113_211311

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : k ≤ n then Nat.choose n k else 0

theorem constant_term_expansion :
    ∀ x: ℂ, (x ≠ 0) → ∃ term: ℂ, 
    term = (-1 : ℂ) * binom 6 4 ∧ term = -15 := 
by
  intros x hx
  use (-1 : ℂ) * binom 6 4
  constructor
  · rfl
  · sorry

end constant_term_expansion_l2113_211311


namespace passengers_landed_in_virginia_l2113_211366

theorem passengers_landed_in_virginia
  (P_start : ℕ) (D_Texas : ℕ) (C_Texas : ℕ) (D_NC : ℕ) (C_NC : ℕ) (C : ℕ)
  (hP_start : P_start = 124)
  (hD_Texas : D_Texas = 58)
  (hC_Texas : C_Texas = 24)
  (hD_NC : D_NC = 47)
  (hC_NC : C_NC = 14)
  (hC : C = 10) :
  P_start - D_Texas + C_Texas - D_NC + C_NC + C = 67 := by
  sorry

end passengers_landed_in_virginia_l2113_211366


namespace remainder_of_x_div_9_is_8_l2113_211374

variable (x y r : ℕ)
variable (r_lt_9 : r < 9)
variable (h1 : x = 9 * y + r)
variable (h2 : 2 * x = 14 * y + 1)
variable (h3 : 5 * y - x = 3)

theorem remainder_of_x_div_9_is_8 : r = 8 := by
  sorry

end remainder_of_x_div_9_is_8_l2113_211374


namespace factorize_polynomial_l2113_211392

theorem factorize_polynomial (c : ℝ) :
  (x : ℝ) → (x - 1) * (x - 3) = x^2 - 4 * x + c → c = 3 :=
by 
  sorry

end factorize_polynomial_l2113_211392


namespace perimeter_of_square_l2113_211327

theorem perimeter_of_square (a : ℤ) (h : a * a = 36) : 4 * a = 24 := 
by
  sorry

end perimeter_of_square_l2113_211327


namespace smallest_q_difference_l2113_211371

theorem smallest_q_difference (p q : ℕ) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_fraction1 : 3 * q < 5 * p)
  (h_fraction2 : 5 * p < 6 * q)
  (h_smallest : ∀ r s : ℕ, 0 < s → 3 * s < 5 * r → 5 * r < 6 * s → q ≤ s) :
  q - p = 3 :=
by
  sorry

end smallest_q_difference_l2113_211371


namespace tree_height_l2113_211329

theorem tree_height (BR MH MB MR TB : ℝ)
  (h_cond1 : BR = 5)
  (h_cond2 : MH = 1.8)
  (h_cond3 : MB = 1)
  (h_cond4 : MR = BR - MB)
  (h_sim : TB / BR = MH / MR)
  : TB = 2.25 :=
by sorry

end tree_height_l2113_211329


namespace probability_first_prize_l2113_211398

-- Define the total number of tickets
def total_tickets : ℕ := 150

-- Define the number of first prizes
def first_prizes : ℕ := 5

-- Define the probability calculation as a theorem
theorem probability_first_prize : (first_prizes : ℚ) / total_tickets = 1 / 30 := 
by sorry  -- Placeholder for the proof

end probability_first_prize_l2113_211398


namespace three_digit_number_is_112_l2113_211370

theorem three_digit_number_is_112 (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 1 ≤ c ∧ c ≤ 9) (h4 : 100 * a + 10 * b + c = 56 * c) :
  100 * a + 10 * b + c = 112 :=
by sorry

end three_digit_number_is_112_l2113_211370


namespace neutralization_reaction_l2113_211335

/-- When combining 2 moles of CH3COOH and 2 moles of NaOH, 2 moles of H2O are formed
    given the balanced chemical reaction CH3COOH + NaOH → CH3COONa + H2O 
    with a molar ratio of 1:1:1 (CH3COOH:NaOH:H2O). -/
theorem neutralization_reaction
  (mCH3COOH : ℕ) (mNaOH : ℕ) :
  (mCH3COOH = 2) → (mNaOH = 2) → (mCH3COOH = mNaOH) →
  ∃ mH2O : ℕ, mH2O = 2 :=
by intros; existsi 2; sorry

end neutralization_reaction_l2113_211335


namespace part_one_part_two_l2113_211353

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

end part_one_part_two_l2113_211353


namespace total_shaded_area_l2113_211337

theorem total_shaded_area (S T : ℝ) (h1 : 16 / S = 4) (h2 : S / T = 4) : 
    S^2 + 16 * T^2 = 32 := 
by {
    sorry
}

end total_shaded_area_l2113_211337


namespace rope_cut_probability_l2113_211384

theorem rope_cut_probability (L : ℝ) (cut_position : ℝ) (P : ℝ) :
  L = 4 → (∀ cut_position, 0 ≤ cut_position ∧ cut_position ≤ L →
  (cut_position ≥ 1.5 ∧ (L - cut_position) ≥ 1.5)) → P = 1 / 4 :=
by
  intros hL hcut
  sorry

end rope_cut_probability_l2113_211384


namespace find_ABC_sum_l2113_211344

theorem find_ABC_sum (A B C : ℤ) (h : ∀ x : ℤ, x = -3 ∨ x = 0 ∨ x = 4 → x^3 + A * x^2 + B * x + C = 0) : 
  A + B + C = -13 := 
by 
  sorry

end find_ABC_sum_l2113_211344


namespace workshop_total_workers_l2113_211348

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

end workshop_total_workers_l2113_211348


namespace tammy_trees_l2113_211349

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

end tammy_trees_l2113_211349


namespace no_common_root_l2113_211310

variables {R : Type*} [OrderedRing R]

def f (x m n : R) := x^2 + m*x + n
def p (x k l : R) := x^2 + k*x + l

theorem no_common_root (k m n l : R) (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬ ∃ x : R, (f x m n = 0 ∧ p x k l = 0) :=
by
  sorry

end no_common_root_l2113_211310


namespace expression_f_range_a_l2113_211373

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

end expression_f_range_a_l2113_211373


namespace find_truck_weight_l2113_211343

variable (T Tr : ℝ)

def weight_condition_1 : Prop := T + Tr = 7000
def weight_condition_2 : Prop := Tr = 0.5 * T - 200

theorem find_truck_weight (h1 : weight_condition_1 T Tr) 
                           (h2 : weight_condition_2 T Tr) : 
  T = 4800 :=
sorry

end find_truck_weight_l2113_211343


namespace proof_problem_l2113_211359

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ -1}

theorem proof_problem :
  ((A ∩ {x | x > -1}) ∪ (B ∩ {x | x ≤ 0})) = {x | x > 0 ∨ x ≤ -1} :=
by 
  sorry

end proof_problem_l2113_211359


namespace angle_BC₁_plane_BBD₁D_l2113_211389

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

end angle_BC₁_plane_BBD₁D_l2113_211389


namespace number_of_factors_and_perfect_square_factors_l2113_211396

open Nat

-- Define the number 1320 and its prime factorization.
def n : ℕ := 1320
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (11, 1)]

-- Define a function to count factors.
def count_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

-- Define a function to count perfect square factors.
def count_perfect_square_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨prime, exp⟩ => acc * (if exp % 2 == 0 then exp / 2 + 1 else 1)) 1

theorem number_of_factors_and_perfect_square_factors :
  count_factors prime_factors = 24 ∧ count_perfect_square_factors prime_factors = 2 :=
by
  sorry

end number_of_factors_and_perfect_square_factors_l2113_211396
