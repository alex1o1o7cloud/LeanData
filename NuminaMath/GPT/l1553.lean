import Mathlib

namespace height_of_middle_brother_l1553_155385

theorem height_of_middle_brother (h₁ h₂ h₃ : ℝ) (h₁_le_h₂ : h₁ ≤ h₂) (h₂_le_h₃ : h₂ ≤ h₃)
  (avg_height : (h₁ + h₂ + h₃) / 3 = 1.74) (avg_height_tallest_shortest : (h₁ + h₃) / 2 = 1.75) :
  h₂ = 1.72 :=
by
  -- Proof to be filled here
  sorry

end height_of_middle_brother_l1553_155385


namespace max_value_a_n_l1553_155350

noncomputable def a_seq : ℕ → ℕ
| 0     => 0  -- By Lean's 0-based indexing, a_1 corresponds to a_seq 1
| 1     => 3
| (n+2) => a_seq (n+1) + 1

def S_n (n : ℕ) : ℕ := (n * (n + 5)) / 2

theorem max_value_a_n : 
  ∃ n : ℕ, S_n n = 2023 ∧ a_seq n = 73 :=
by
  sorry

end max_value_a_n_l1553_155350


namespace greatest_large_chips_l1553_155392

theorem greatest_large_chips (s l p : ℕ) (h1 : s + l = 80) (h2 : s = l + p) (hp : Nat.Prime p) : l ≤ 39 :=
by
  sorry

end greatest_large_chips_l1553_155392


namespace diagonal_cannot_be_good_l1553_155374

def is_good (table : ℕ → ℕ → ℕ) (i j : ℕ) :=
  ∀ x y, (x = i ∨ y = j) → ∀ x' y', (x' = i ∨ y' = j) → (x ≠ x' ∨ y ≠ y') → table x y ≠ table x' y'

theorem diagonal_cannot_be_good :
  ∀ (table : ℕ → ℕ → ℕ), (∀ i j, 1 ≤ table i j ∧ table i j ≤ 25) →
  ¬ ∀ k, (is_good table k k) :=
by
  sorry

end diagonal_cannot_be_good_l1553_155374


namespace range_of_m_l1553_155344

theorem range_of_m (m : ℝ) (h : 2 * m + 3 < 4) : m < 1 / 2 :=
by
  sorry

end range_of_m_l1553_155344


namespace truncated_cone_sphere_radius_l1553_155300

theorem truncated_cone_sphere_radius :
  ∀ (r1 r2 h : ℝ), 
  r1 = 24 → 
  r2 = 6 → 
  h = 20 → 
  ∃ r, 
  r = 17 * Real.sqrt 2 / 2 := by
  intros r1 r2 h hr1 hr2 hh
  sorry

end truncated_cone_sphere_radius_l1553_155300


namespace find_a_and_b_maximize_profit_l1553_155307

variable (a b x : ℝ)

-- The given conditions
def condition1 : Prop := 2 * a + b = 120
def condition2 : Prop := 4 * a + 3 * b = 270
def constraint : Prop := 75 ≤ 300 - x

-- The questions translated into a proof problem
theorem find_a_and_b :
  condition1 a b ∧ condition2 a b → a = 45 ∧ b = 30 :=
by
  intros h
  sorry

theorem maximize_profit (a : ℝ) (b : ℝ) (x : ℝ) :
  condition1 a b → condition2 a b → constraint x →
  x = 75 → (300 - x) = 225 → 
  (10 * x + 20 * (300 - x) = 5250) :=
by
  intros h1 h2 hc hx hx1
  sorry

end find_a_and_b_maximize_profit_l1553_155307


namespace find_tangent_points_l1553_155346

def f (x : ℝ) : ℝ := x^3 + x - 2
def tangent_parallel_to_line (x : ℝ) : Prop := deriv f x = 4

theorem find_tangent_points :
  (tangent_parallel_to_line 1 ∧ f 1 = 0) ∧ 
  (tangent_parallel_to_line (-1) ∧ f (-1) = -4) :=
by
  sorry

end find_tangent_points_l1553_155346


namespace lower_side_length_is_correct_l1553_155362

noncomputable def length_of_lower_side
  (a b h : ℝ) (A : ℝ) 
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62) : ℝ :=
b

theorem lower_side_length_is_correct
  (a b h : ℝ) (A : ℝ)
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62)
  (ha : A = (1/2) * (a + b) * h) : b = 17.65 :=
by
  sorry

end lower_side_length_is_correct_l1553_155362


namespace find_m_l1553_155311

noncomputable def is_power_function (y : ℝ → ℝ) := 
  ∃ (c : ℝ), ∃ (n : ℝ), ∀ x : ℝ, y x = c * x ^ n

theorem find_m (m : ℝ) :
  (∀ x : ℝ, (∃ c : ℝ, (m^2 - 2 * m + 1) * x^(m - 1) = c * x^n) ∧ (∀ x : ℝ, true)) → m = 2 :=
sorry

end find_m_l1553_155311


namespace harry_annual_pet_feeding_cost_l1553_155381

def monthly_cost_snake := 10
def monthly_cost_iguana := 5
def monthly_cost_gecko := 15
def num_snakes := 4
def num_iguanas := 2
def num_geckos := 3
def months_in_year := 12

theorem harry_annual_pet_feeding_cost :
  (num_snakes * monthly_cost_snake + 
   num_iguanas * monthly_cost_iguana + 
   num_geckos * monthly_cost_gecko) * 
   months_in_year = 1140 := 
sorry

end harry_annual_pet_feeding_cost_l1553_155381


namespace total_digits_first_2003_even_integers_l1553_155317

theorem total_digits_first_2003_even_integers : 
  let even_integers := (List.range' 1 (2003 * 2)).filter (λ n => n % 2 = 0)
  let one_digit_count := List.filter (λ n => n < 10) even_integers |>.length
  let two_digit_count := List.filter (λ n => 10 ≤ n ∧ n < 100) even_integers |>.length
  let three_digit_count := List.filter (λ n => 100 ≤ n ∧ n < 1000) even_integers |>.length
  let four_digit_count := List.filter (λ n => 1000 ≤ n) even_integers |>.length
  let total_digits := one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3 + four_digit_count * 4
  total_digits = 7460 :=
by
  sorry

end total_digits_first_2003_even_integers_l1553_155317


namespace intersection_complement_eq_l1553_155357

open Set

variable (U M N : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5} →
  M = {1, 4} →
  N = {1, 3, 5} →
  N ∩ (U \ M) = {3, 5} := by 
sorry

end intersection_complement_eq_l1553_155357


namespace volume_of_S_l1553_155330

-- Define the region S in terms of the conditions
def region_S (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1.5 ∧ 
  abs x + abs y ≤ 1 ∧ 
  abs z ≤ 0.5

-- Define the volume calculation function
noncomputable def volume_S : ℝ :=
  sorry -- This is where the computation/theorem proving for volume would go

-- The theorem stating the volume of S
theorem volume_of_S : volume_S = 2 / 3 :=
  sorry

end volume_of_S_l1553_155330


namespace hyperbola_standard_equation_l1553_155384

theorem hyperbola_standard_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h_real_axis : 2 * a = 4 * Real.sqrt 2) (h_eccentricity : a / Real.sqrt (a^2 + b^2) = Real.sqrt 6 / 2) :
    (a = 2 * Real.sqrt 2) ∧ (b = 2) → ∀ x y : ℝ, (x^2)/8 - (y^2)/4 = 1 :=
sorry

end hyperbola_standard_equation_l1553_155384


namespace ab_eq_zero_l1553_155368

theorem ab_eq_zero (a b : ℤ) (h : ∀ m n : ℕ, ∃ k : ℤ, a * (m^2 : ℤ) + b * (n^2 : ℤ) = k^2) : a * b = 0 :=
by
  sorry

end ab_eq_zero_l1553_155368


namespace evaluate_g_inv_l1553_155372

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 6)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 7)
variable (h_inv1 : g_inv 6 = 4)
variable (h_inv2 : g_inv 7 = 3)
variable (h_inv_eq : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x)

theorem evaluate_g_inv :
  g_inv (g_inv 6 + g_inv 7) = 3 :=
by
  sorry

end evaluate_g_inv_l1553_155372


namespace cos_double_angle_example_l1553_155397

theorem cos_double_angle_example (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 := by
  sorry

end cos_double_angle_example_l1553_155397


namespace geometric_progression_product_sum_sumrecip_l1553_155331

theorem geometric_progression_product_sum_sumrecip (P S S' : ℝ) (n : ℕ)
  (hP : P = a ^ n * r ^ ((n * (n - 1)) / 2))
  (hS : S = a * (1 - r ^ n) / (1 - r))
  (hS' : S' = (r ^ n - 1) / (a * (r - 1))) :
  P = (S / S') ^ (1 / 2 * n) :=
  sorry

end geometric_progression_product_sum_sumrecip_l1553_155331


namespace selection_probability_equal_l1553_155353

theorem selection_probability_equal :
  let n := 2012
  let eliminated := 12
  let remaining := n - eliminated
  let selected := 50
  let probability := (remaining / n) * (selected / remaining)
  probability = 25 / 1006 :=
by
  sorry

end selection_probability_equal_l1553_155353


namespace factorize_1_factorize_2_l1553_155354

variable {a x y : ℝ}

theorem factorize_1 : 2 * a * x^2 - 8 * a * x * y + 8 * a * y^2 = 2 * a * (x - 2 * y)^2 := 
by
  sorry

theorem factorize_2 : 6 * x * y^2 - 9 * x^2 * y - y^3 = -y * (3 * x - y)^2 := 
by
  sorry

end factorize_1_factorize_2_l1553_155354


namespace variable_swap_l1553_155303

theorem variable_swap (x y t : Nat) (h1 : x = 5) (h2 : y = 6) (h3 : t = x) (h4 : x = y) (h5 : y = t) : 
  x = 6 ∧ y = 5 := 
by
  sorry

end variable_swap_l1553_155303


namespace fermat_numbers_pairwise_coprime_l1553_155371

theorem fermat_numbers_pairwise_coprime :
  ∀ i j : ℕ, i ≠ j → Nat.gcd (2 ^ (2 ^ i) + 1) (2 ^ (2 ^ j) + 1) = 1 :=
sorry

end fermat_numbers_pairwise_coprime_l1553_155371


namespace max_non_overlapping_areas_l1553_155318

theorem max_non_overlapping_areas (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, k = 4 * n + 4 := 
sorry

end max_non_overlapping_areas_l1553_155318


namespace bill_before_tax_l1553_155305

theorem bill_before_tax (T E : ℝ) (h1 : E = 2) (h2 : 3 * T + 5 * E = 12.70) : 2 * T + 3 * E = 7.80 :=
by
  sorry

end bill_before_tax_l1553_155305


namespace probability_red_or_blue_l1553_155373

noncomputable def total_marbles : ℕ := 100

noncomputable def probability_white : ℚ := 1 / 4

noncomputable def probability_green : ℚ := 1 / 5

theorem probability_red_or_blue :
  (1 - (probability_white + probability_green)) = 11 / 20 :=
by
  -- Proof is omitted
  sorry

end probability_red_or_blue_l1553_155373


namespace line_tangent_ellipse_l1553_155352

-- Define the conditions of the problem
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x + 2
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

-- Prove the statement about the intersection of the line and ellipse
theorem line_tangent_ellipse (m : ℝ) :
  (∀ x y, line m x y → ellipse x y → x = 0.0 ∧ y = 2.0)
  ↔ m^2 = 1 / 3 :=
sorry

end line_tangent_ellipse_l1553_155352


namespace roots_equation_l1553_155389

theorem roots_equation (m n : ℝ) (h1 : ∀ x, (x - m) * (x - n) = x^2 + 2 * x - 2025) : m^2 + 3 * m + n = 2023 :=
by
  sorry

end roots_equation_l1553_155389


namespace find_x_l1553_155321

theorem find_x :
  ∀ (x : ℝ), 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 → x = 13.26 :=
by
  intro x
  intro h
  sorry

end find_x_l1553_155321


namespace chocolate_factory_production_l1553_155334

theorem chocolate_factory_production
  (candies_per_hour : ℕ)
  (total_candies : ℕ)
  (days : ℕ)
  (total_hours : ℕ := total_candies / candies_per_hour)
  (hours_per_day : ℕ := total_hours / days)
  (h1 : candies_per_hour = 50)
  (h2 : total_candies = 4000)
  (h3 : days = 8) :
  hours_per_day = 10 := by
  sorry

end chocolate_factory_production_l1553_155334


namespace no_real_solution_condition_l1553_155356

def no_real_solution (k : ℝ) : Prop :=
  let discriminant := 25 + 4 * k
  discriminant < 0

theorem no_real_solution_condition (k : ℝ) : no_real_solution k ↔ k < -25 / 4 := 
sorry

end no_real_solution_condition_l1553_155356


namespace solve_for_x_l1553_155324

theorem solve_for_x : 
  ∃ x : ℝ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1 / 2 ∧ x = -5.375 :=
by
  sorry

end solve_for_x_l1553_155324


namespace ratio_of_toys_l1553_155337

theorem ratio_of_toys (total_toys : ℕ) (num_friends : ℕ) (toys_D : ℕ) 
  (h1 : total_toys = 118) 
  (h2 : num_friends = 4) 
  (h3 : toys_D = total_toys / num_friends) : 
  (toys_D / total_toys : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_toys_l1553_155337


namespace m_gt_n_l1553_155340

noncomputable def m : ℕ := 2015 ^ 2016
noncomputable def n : ℕ := 2016 ^ 2015

theorem m_gt_n : m > n := by
  sorry

end m_gt_n_l1553_155340


namespace unique_solution_implies_a_eq_pm_b_l1553_155320

theorem unique_solution_implies_a_eq_pm_b 
  (a b : ℝ) 
  (h_nonzero_a : a ≠ 0) 
  (h_nonzero_b : b ≠ 0) 
  (h_unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) : 
  a = b ∨ a = -b :=
sorry

end unique_solution_implies_a_eq_pm_b_l1553_155320


namespace employee_salary_l1553_155329

theorem employee_salary (x y : ℝ) (h1 : x + y = 770) (h2 : x = 1.2 * y) : y = 350 :=
by
  sorry

end employee_salary_l1553_155329


namespace max_marks_l1553_155365

theorem max_marks (M p : ℝ) (h1 : p = 0.60 * M) (h2 : p = 160 + 20) : M = 300 := by
  sorry

end max_marks_l1553_155365


namespace initial_antifreeze_percentage_l1553_155322

-- Definitions of conditions
def total_volume : ℚ := 10
def replaced_volume : ℚ := 2.85714285714
def final_percentage : ℚ := 50 / 100

-- Statement to prove
theorem initial_antifreeze_percentage (P : ℚ) :
  10 * P / 100 - P / 100 * 2.85714285714 + 2.85714285714 = 5 → 
  P = 30 :=
sorry

end initial_antifreeze_percentage_l1553_155322


namespace distance_from_apex_l1553_155335

theorem distance_from_apex (A B : ℝ)
  (h_A : A = 216 * Real.sqrt 3)
  (h_B : B = 486 * Real.sqrt 3)
  (distance_planes : ℝ)
  (h_distance_planes : distance_planes = 8) :
  ∃ h : ℝ, h = 24 :=
by
  sorry

end distance_from_apex_l1553_155335


namespace book_cost_l1553_155360

variable {b m : ℝ}

theorem book_cost (h1 : b + m = 2.10) (h2 : b = m + 2) : b = 2.05 :=
by
  sorry

end book_cost_l1553_155360


namespace proposition_p_and_not_q_is_true_l1553_155314

-- Define proposition p
def p : Prop := ∀ x > 0, Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : Real, a > b → a^2 > b^2

-- State the theorem to be proven in Lean
theorem proposition_p_and_not_q_is_true : p ∧ ¬q :=
by
  -- Sorry placeholder for the proof
  sorry

end proposition_p_and_not_q_is_true_l1553_155314


namespace infinite_series_equivalence_l1553_155312

theorem infinite_series_equivalence (x y : ℝ) (hy : y ≠ 0 ∧ y ≠ 1) 
  (series_cond : ∑' n : ℕ, x / (y^(n+1)) = 3) :
  ∑' n : ℕ, x / ((x + 2*y)^(n+1)) = 3 * (y - 1) / (5*y - 4) := 
by
  sorry

end infinite_series_equivalence_l1553_155312


namespace find_a_l1553_155326

theorem find_a (a n : ℝ) (p : ℝ) (hp : p = 2 / 3)
  (h₁ : a = 3 * n + 5)
  (h₂ : a + 2 = 3 * (n + p) + 5) : a = 3 * n + 5 :=
by 
  sorry

end find_a_l1553_155326


namespace find_f_l1553_155347

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x / (a * x + b)

theorem find_f (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f 2 a b = 1) (h₂ : ∃! x, f x a b = x) :
  f x (1/2) 1 = 2 * x / (x + 2) :=
by
  sorry

end find_f_l1553_155347


namespace problem_statement_l1553_155386

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {4, 5}
def C_U (B : Set ℕ) : Set ℕ := U \ B

-- Statement
theorem problem_statement : A ∩ (C_U B) = {2} :=
  sorry

end problem_statement_l1553_155386


namespace ingrid_tax_rate_l1553_155393

def john_income : ℝ := 57000
def ingrid_income : ℝ := 72000
def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_tax_rate :
  let john_tax := john_tax_rate * john_income
  let combined_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * combined_income
  let ingrid_tax := total_tax - john_tax
  let ingrid_tax_rate := ingrid_tax / ingrid_income
  ingrid_tax_rate = 0.40 :=
by
  sorry

end ingrid_tax_rate_l1553_155393


namespace starters_choice_l1553_155302

/-- There are 18 players including a set of quadruplets: Bob, Bill, Ben, and Bert. -/
def total_players : ℕ := 18

/-- The set of quadruplets: Bob, Bill, Ben, and Bert. -/
def quadruplets : Finset (String) := {"Bob", "Bill", "Ben", "Bert"}

/-- We need to choose 7 starters, exactly 3 of which are from the set of quadruplets. -/
def ways_to_choose_starters : ℕ :=
  let quadruplet_combinations := Nat.choose 4 3
  let remaining_spots := 4
  let remaining_players := total_players - 4
  quadruplet_combinations * Nat.choose remaining_players remaining_spots

theorem starters_choice (h1 : total_players = 18)
                        (h2 : quadruplets.card = 4) :
  ways_to_choose_starters = 4004 :=
by 
  -- conditional setups here
  sorry

end starters_choice_l1553_155302


namespace decreasing_on_transformed_interval_l1553_155377

theorem decreasing_on_transformed_interval
  (f : ℝ → ℝ)
  (h : ∀ ⦃x₁ x₂ : ℝ⦄, 1 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 2 → f x₁ ≤ f x₂) :
  ∀ ⦃x₁ x₂ : ℝ⦄, -1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 0 → f (1 - x₂) ≤ f (1 - x₁) :=
sorry

end decreasing_on_transformed_interval_l1553_155377


namespace jill_arrives_before_jack_l1553_155380

theorem jill_arrives_before_jack {distance speed_jill speed_jack : ℝ} (h1 : distance = 1) 
  (h2 : speed_jill = 10) (h3 : speed_jack = 4) :
  (60 * (distance / speed_jack) - 60 * (distance / speed_jill)) = 9 :=
by
  sorry

end jill_arrives_before_jack_l1553_155380


namespace problem_statement_l1553_155376

theorem problem_statement (a b : ℝ) (h1 : 1 + b = 0) (h2 : a - 3 = 0) : 
  3 * (a^2 - 2 * a * b + b^2) - (4 * a^2 - 2 * (1 / 2 * a^2 + a * b - 3 / 2 * b^2)) = 12 :=
by
  sorry

end problem_statement_l1553_155376


namespace slope_of_line_I_l1553_155363

-- Line I intersects y = 1 at point P
def intersects_y_eq_one (I P : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, P (x, 1) ↔ I (x, y) ∧ y = 1

-- Line I intersects x - y - 7 = 0 at point Q
def intersects_x_minus_y_eq_seven (I Q : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, Q (x, y) ↔ I (x, y) ∧ x - y - 7 = 0

-- The coordinates of the midpoint of segment PQ are (1, -1)
def midpoint_eq (P Q : ℝ × ℝ) : Prop :=
∃ x1 y1 x2 y2 : ℝ,
  P = (x1, y1) ∧ Q = (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)

-- We need to show that the slope of line I is -2/3
def slope_of_I (I : ℝ × ℝ → Prop) (k : ℝ) : Prop :=
∀ x y : ℝ, I (x, y) → y + 1 = k * (x - 1)

theorem slope_of_line_I :
  ∃ I P Q : (ℝ × ℝ → Prop),
    intersects_y_eq_one I P ∧
    intersects_x_minus_y_eq_seven I Q ∧
    (∃ x1 y1 x2 y2 : ℝ, P (x1, y1) ∧ Q (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)) →
    slope_of_I I (-2/3) :=
by
  sorry

end slope_of_line_I_l1553_155363


namespace fraction_simplification_l1553_155395

theorem fraction_simplification :
  (20 / 21) * (35 / 54) * (63 / 50) = (7 / 9) :=
by
  sorry

end fraction_simplification_l1553_155395


namespace tan_alpha_minus_pi_div_4_l1553_155345

open Real

theorem tan_alpha_minus_pi_div_4 (α : ℝ) (h : (cos α * 2 + (-1) * sin α = 0)) : 
  tan (α - π / 4) = 1 / 3 :=
sorry

end tan_alpha_minus_pi_div_4_l1553_155345


namespace find_f_10_l1553_155379

variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, f x = 2 * x^2 + y
def condition2 : Prop := f 2 = 30

-- Theorem to prove
theorem find_f_10 (h1 : condition1 f y) (h2 : condition2 f) : f 10 = 222 := 
sorry

end find_f_10_l1553_155379


namespace min_value_expression_eq_2sqrt3_l1553_155313

noncomputable def min_value_expression (c d : ℝ) : ℝ :=
  c^2 + d^2 + 4 / c^2 + 2 * d / c

theorem min_value_expression_eq_2sqrt3 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, (∀ d : ℝ, min_value_expression c d ≥ y) ∧ y = 2 * Real.sqrt 3 :=
sorry

end min_value_expression_eq_2sqrt3_l1553_155313


namespace quadratic_roots_unique_pair_l1553_155327

theorem quadratic_roots_unique_pair (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h_root1 : p * q = q)
  (h_root2 : p + q = -p)
  (h_rel : q = -2 * p) : 
(p, q) = (1, -2) :=
  sorry

end quadratic_roots_unique_pair_l1553_155327


namespace jasmine_money_left_l1553_155308

theorem jasmine_money_left 
  (initial_amount : ℝ)
  (apple_cost : ℝ) (num_apples : ℕ)
  (orange_cost : ℝ) (num_oranges : ℕ)
  (pear_cost : ℝ) (num_pears : ℕ)
  (h_initial : initial_amount = 100.00)
  (h_apple_cost : apple_cost = 1.50)
  (h_num_apples : num_apples = 5)
  (h_orange_cost : orange_cost = 2.00)
  (h_num_oranges : num_oranges = 10)
  (h_pear_cost : pear_cost = 2.25)
  (h_num_pears : num_pears = 4) : 
  initial_amount - (num_apples * apple_cost + num_oranges * orange_cost + num_pears * pear_cost) = 63.50 := 
by 
  sorry

end jasmine_money_left_l1553_155308


namespace standard_circle_eq_l1553_155319

noncomputable def circle_equation : String :=
  "The standard equation of the circle whose center lies on the line y = -4x and is tangent to the line x + y - 1 = 0 at point P(3, -2) is (x - 1)^2 + (y + 4)^2 = 8"

theorem standard_circle_eq
  (center_x : ℝ)
  (center_y : ℝ)
  (tangent_line : ℝ → ℝ → Prop)
  (point : ℝ × ℝ)
  (eqn_line : ∀ x y, tangent_line x y ↔ x + y - 1 = 0)
  (center_on_line : ∀ x y, y = -4 * x → center_y = y)
  (point_on_tangent : point = (3, -2))
  (tangent_at_point : tangent_line (point.1) (point.2)) :
  (center_x = 1 ∧ center_y = -4 ∧ (∃ r : ℝ, r = 2 * Real.sqrt 2)) →
  (∀ x y, (x - 1)^2 + (y + 4)^2 = 8) := by
  sorry

end standard_circle_eq_l1553_155319


namespace a_received_share_l1553_155336

def a_inv : ℕ := 7000
def b_inv : ℕ := 11000
def c_inv : ℕ := 18000

def b_share : ℕ := 2200

def total_profit : ℕ := (b_share / (b_inv / 1000)) * 36
def a_ratio : ℕ := a_inv / 1000
def total_ratio : ℕ := (a_inv / 1000) + (b_inv / 1000) + (c_inv / 1000)

def a_share : ℕ := (a_ratio / total_ratio) * total_profit

theorem a_received_share :
  a_share = 1400 := 
sorry

end a_received_share_l1553_155336


namespace probability_same_color_l1553_155375

theorem probability_same_color :
  let bagA_white := 8
  let bagA_red := 4
  let bagB_white := 6
  let bagB_red := 6
  let totalA := bagA_white + bagA_red
  let totalB := bagB_white + bagB_red
  let prob_white_white := (bagA_white / totalA) * (bagB_white / totalB)
  let prob_red_red := (bagA_red / totalA) * (bagB_red / totalB)
  let total_prob := prob_white_white + prob_red_red
  total_prob = 1 / 2 := 
by 
  sorry

end probability_same_color_l1553_155375


namespace prime_15p_plus_one_l1553_155339

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_15p_plus_one (p q : ℕ) 
  (hp : is_prime p) 
  (hq : q = 15 * p + 1) 
  (hq_prime : is_prime q) :
  q = 31 :=
sorry

end prime_15p_plus_one_l1553_155339


namespace leah_earned_initially_l1553_155316

noncomputable def initial_money (x : ℝ) : Prop :=
  let amount_after_milkshake := (6 / 7) * x
  let amount_left_wallet := (3 / 7) * x
  amount_left_wallet = 12

theorem leah_earned_initially (x : ℝ) (h : initial_money x) : x = 28 :=
by
  sorry

end leah_earned_initially_l1553_155316


namespace problem_1_problem_2_problem_3_l1553_155382

-- Definitions based on problem conditions
def total_people := 12
def choices := 5
def special_people_count := 3

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Proof problem 1: A, B, and C must be chosen, so select 2 more from the remaining 9 people
theorem problem_1 : choose 9 2 = 36 :=
by sorry

-- Proof problem 2: Only one among A, B, and C is chosen, so select 4 more from the remaining 9 people
theorem problem_2 : choose 3 1 * choose 9 4 = 378 :=
by sorry

-- Proof problem 3: At most two among A, B, and C are chosen
theorem problem_3 : choose 12 5 - choose 9 2 = 756 :=
by sorry

end problem_1_problem_2_problem_3_l1553_155382


namespace boys_meet_once_excluding_start_finish_l1553_155398

theorem boys_meet_once_excluding_start_finish 
    (d : ℕ) 
    (h1 : 0 < d) 
    (boy1_speed : ℕ) (boy2_speed : ℕ) 
    (h2 : boy1_speed = 6) (h3 : boy2_speed = 10)
    (relative_speed : ℕ) (h4 : relative_speed = boy1_speed + boy2_speed) 
    (time_to_meet_A_again : ℕ) (h5 : time_to_meet_A_again = d / relative_speed) 
    (boy1_laps_per_sec boy2_laps_per_sec : ℕ) 
    (h6 : boy1_laps_per_sec = boy1_speed / d) 
    (h7 : boy2_laps_per_sec = boy2_speed / d)
    (lcm_laps : ℕ) (h8 : lcm_laps = Nat.lcm 6 10)
    (meetings_per_lap : ℕ) (h9 : meetings_per_lap = lcm_laps / d)
    (total_meetings : ℕ) (h10 : total_meetings = meetings_per_lap * time_to_meet_A_again)
  : total_meetings = 1 := by
  sorry

end boys_meet_once_excluding_start_finish_l1553_155398


namespace total_bills_combined_l1553_155323

theorem total_bills_combined
  (a b c : ℝ)
  (H1 : 0.15 * a = 3)
  (H2 : 0.25 * b = 5)
  (H3 : 0.20 * c = 4) :
  a + b + c = 60 := 
sorry

end total_bills_combined_l1553_155323


namespace joe_initial_tests_l1553_155383

theorem joe_initial_tests (S n : ℕ) (h1 : S = 60 * n) (h2 : (S - 45) = 65 * (n - 1)) : n = 4 :=
by {
  sorry
}

end joe_initial_tests_l1553_155383


namespace chord_length_of_intersecting_line_and_circle_l1553_155355

theorem chord_length_of_intersecting_line_and_circle :
  ∀ (x y : ℝ), (3 * x + 4 * y - 5 = 0) ∧ (x^2 + y^2 = 4) →
  ∃ (AB : ℝ), AB = 2 * Real.sqrt 3 := 
sorry

end chord_length_of_intersecting_line_and_circle_l1553_155355


namespace new_profit_percentage_l1553_155338

theorem new_profit_percentage (P : ℝ) (h1 : 1.10 * P = 990) (h2 : 0.90 * P * (1 + 0.30) = 1053) : 0.30 = 0.30 :=
by sorry

end new_profit_percentage_l1553_155338


namespace min_k_spherical_cap_cylinder_l1553_155310

/-- Given a spherical cap and a cylinder sharing a common inscribed sphere with volumes V1 and V2 respectively,
we show that the minimum value of k such that V1 = k * V2 is 4/3. -/
theorem min_k_spherical_cap_cylinder (R : ℝ) (V1 V2 : ℝ) (h1 : V1 = (4/3) * π * R^3) 
(h2 : V2 = 2 * π * R^3) : 
∃ k : ℝ, V1 = k * V2 ∧ k = 4/3 := 
by 
  use (4/3)
  constructor
  . sorry
  . sorry

end min_k_spherical_cap_cylinder_l1553_155310


namespace adam_initial_books_l1553_155342

theorem adam_initial_books (B : ℕ) (h1 : B - 11 + 23 = 45) : B = 33 := 
by
  sorry

end adam_initial_books_l1553_155342


namespace smallest_x_mod_equation_l1553_155301

theorem smallest_x_mod_equation : ∃ x : ℕ, 42 * x + 10 ≡ 5 [MOD 15] ∧ ∀ y : ℕ, 42 * y + 10 ≡ 5 [MOD 15] → x ≤ y :=
by
sorry

end smallest_x_mod_equation_l1553_155301


namespace renaldo_distance_l1553_155364

theorem renaldo_distance (R : ℕ) (h : R + (1/3 : ℝ) * R + 7 = 27) : R = 15 :=
by sorry

end renaldo_distance_l1553_155364


namespace smallest_range_possible_l1553_155358

-- Definition of the problem conditions
def seven_observations (x1 x2 x3 x4 x5 x6 x7 : ℝ) :=
  (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 9 ∧
  x4 = 10

noncomputable def smallest_range : ℝ :=
  5

-- Lean statement asserting the proof problem
theorem smallest_range_possible (x1 x2 x3 x4 x5 x6 x7 : ℝ) (h : seven_observations x1 x2 x3 x4 x5 x6 x7) :
  ∃ x1' x2' x3' x4' x5' x6' x7', seven_observations x1' x2' x3' x4' x5' x6' x7' ∧ (x7' - x1') = smallest_range :=
sorry

end smallest_range_possible_l1553_155358


namespace new_boarder_ratio_l1553_155343

structure School where
  initial_boarders : ℕ
  day_students : ℕ
  boarders_ratio : ℚ

theorem new_boarder_ratio (S : School) (additional_boarders : ℕ) :
  S.initial_boarders = 60 →
  S.boarders_ratio = 2 / 5 →
  additional_boarders = 15 →
  S.day_students = (60 * 5) / 2 →
  (S.initial_boarders + additional_boarders) / S.day_students = 1 / 2 :=
by
  sorry

end new_boarder_ratio_l1553_155343


namespace downstream_speed_l1553_155361

noncomputable def speed_downstream (Vu Vs : ℝ) : ℝ :=
  2 * Vs - Vu

theorem downstream_speed (Vu Vs : ℝ) (hVu : Vu = 30) (hVs : Vs = 45) :
  speed_downstream Vu Vs = 60 := by
  rw [hVu, hVs]
  dsimp [speed_downstream]
  linarith

end downstream_speed_l1553_155361


namespace solve_for_x_l1553_155370

theorem solve_for_x (x : ℤ) (h : 3 * x + 7 = -2) : x = -3 :=
by
  sorry

end solve_for_x_l1553_155370


namespace greatest_product_three_integers_sum_2000_l1553_155332

noncomputable def maxProduct (s : ℝ) : ℝ := 
  s * s * (2000 - 2 * s)

theorem greatest_product_three_integers_sum_2000 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 / 2 ∧ maxProduct x = 8000000000 / 27 := sorry

end greatest_product_three_integers_sum_2000_l1553_155332


namespace original_distance_between_Stacy_and_Heather_l1553_155367

theorem original_distance_between_Stacy_and_Heather
  (H_speed : ℝ := 5)  -- Heather's speed in miles per hour
  (S_speed : ℝ := 6)  -- Stacy's speed in miles per hour
  (delay : ℝ := 0.4)  -- Heather's start delay in hours
  (H_distance : ℝ := 1.1818181818181817)  -- Distance Heather walked when they meet
  : H_speed * (H_distance / H_speed) + S_speed * ((H_distance / H_speed) + delay) = 5 := by
  sorry

end original_distance_between_Stacy_and_Heather_l1553_155367


namespace wine_distribution_l1553_155391

theorem wine_distribution (m n k s : ℕ) (h : Nat.gcd m (Nat.gcd n k) = 1) (h_s : s < m + n + k) :
  ∃ g : ℕ, g = s := by
  sorry

end wine_distribution_l1553_155391


namespace students_per_class_l1553_155388

theorem students_per_class
  (cards_per_student : Nat)
  (periods_per_day : Nat)
  (cost_per_pack : Nat)
  (total_spent : Nat)
  (cards_per_pack : Nat)
  (students_per_class : Nat)
  (H1 : cards_per_student = 10)
  (H2 : periods_per_day = 6)
  (H3 : cost_per_pack = 3)
  (H4 : total_spent = 108)
  (H5 : cards_per_pack = 50)
  (H6 : students_per_class = 30)
  :
  students_per_class = (total_spent / cost_per_pack * cards_per_pack / cards_per_student / periods_per_day) :=
sorry

end students_per_class_l1553_155388


namespace female_employees_l1553_155349

theorem female_employees (total_employees male_employees : ℕ) 
  (advanced_degree_male_adv: ℝ) (advanced_degree_female_adv: ℝ) (prob: ℝ) 
  (h1 : total_employees = 450) 
  (h2 : male_employees = 300)
  (h3 : advanced_degree_male_adv = 0.10) 
  (h4 : advanced_degree_female_adv = 0.40)
  (h5 : prob = 0.4) : 
  ∃ F : ℕ, 0.10 * male_employees + (advanced_degree_female_adv * F + (1 - advanced_degree_female_adv) * F) / total_employees = prob ∧ F = 150 :=
by
  sorry

end female_employees_l1553_155349


namespace polygon_with_interior_sum_1260_eq_nonagon_l1553_155399

theorem polygon_with_interior_sum_1260_eq_nonagon :
  ∃ n : ℕ, (n-2) * 180 = 1260 ∧ n = 9 := by
  sorry

end polygon_with_interior_sum_1260_eq_nonagon_l1553_155399


namespace range_of_k_for_ellipse_l1553_155359

def represents_ellipse (x y k : ℝ) : Prop :=
  (k^2 - 3 > 0) ∧ 
  (k - 1 > 0) ∧ 
  (k - 1 ≠ k^2 - 3)

theorem range_of_k_for_ellipse (k : ℝ) : 
  represents_ellipse x y k → k ∈ Set.Ioo (-Real.sqrt 3) (-1) ∪ Set.Ioo (-1) 1 :=
by
  sorry

end range_of_k_for_ellipse_l1553_155359


namespace work_completed_in_30_days_l1553_155306

theorem work_completed_in_30_days (ravi_days : ℕ) (prakash_days : ℕ)
  (h1 : ravi_days = 50) (h2 : prakash_days = 75) : 
  let ravi_rate := (1 / 50 : ℚ)
  let prakash_rate := (1 / 75 : ℚ)
  let combined_rate := ravi_rate + prakash_rate
  let days_to_complete := 1 / combined_rate
  days_to_complete = 30 := by
  sorry

end work_completed_in_30_days_l1553_155306


namespace max_value_expression_l1553_155351

theorem max_value_expression : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 →
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 256 / 243 :=
by
  intros x y z hx hy hz hsum
  sorry

end max_value_expression_l1553_155351


namespace polynomial_A_polynomial_B_l1553_155366

-- Problem (1): Prove that A = 6x^3 + 8x^2 + x - 1 given the conditions.
theorem polynomial_A :
  ∀ (x : ℝ),
  (2 * x^2 * (3 * x + 4) + (x - 1) = 6 * x^3 + 8 * x^2 + x - 1) :=
by
  intro x
  sorry

-- Problem (2): Prove that B = 6x^2 - 19x + 9 given the conditions.
theorem polynomial_B :
  ∀ (x : ℝ),
  ((2 * x - 6) * (3 * x - 1) + (x + 3) = 6 * x^2 - 19 * x + 9) :=
by
  intro x
  sorry

end polynomial_A_polynomial_B_l1553_155366


namespace evaluate_expression_l1553_155394

theorem evaluate_expression : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) :=
by
  sorry

end evaluate_expression_l1553_155394


namespace find_solutions_of_x4_minus_16_l1553_155333

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l1553_155333


namespace seventh_observation_l1553_155387

-- Declare the conditions with their definitions
def average_of_six (sum6 : ℕ) : Prop := sum6 = 6 * 14
def new_average_decreased (sum6 sum7 : ℕ) : Prop := sum7 = sum6 + 7 ∧ 13 = (sum6 + 7) / 7

-- The main statement to prove that the seventh observation is 7
theorem seventh_observation (sum6 sum7 : ℕ) (h_avg6 : average_of_six sum6) (h_new_avg : new_average_decreased sum6 sum7) :
  sum7 - sum6 = 7 := 
  sorry

end seventh_observation_l1553_155387


namespace kamal_age_problem_l1553_155325

theorem kamal_age_problem (K S : ℕ) 
  (h1 : K - 8 = 4 * (S - 8)) 
  (h2 : K + 8 = 2 * (S + 8)) : 
  K = 40 := 
by sorry

end kamal_age_problem_l1553_155325


namespace find_constants_l1553_155328

theorem find_constants (P Q R : ℤ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) →
  P = -6 ∧ Q = 8 ∧ R = -5 :=
by
  sorry

end find_constants_l1553_155328


namespace cube_edge_length_eq_six_l1553_155348

theorem cube_edge_length_eq_six {s : ℝ} (h : s^3 = 6 * s^2) : s = 6 :=
sorry

end cube_edge_length_eq_six_l1553_155348


namespace unique_n_for_given_divisors_l1553_155390

theorem unique_n_for_given_divisors :
  ∃! (n : ℕ), 
    ∀ (k : ℕ) (d : ℕ → ℕ), 
      k ≥ 22 ∧ 
      d 1 = 1 ∧ d k = n ∧ 
      (∀ i j, i < j → d i < d j) ∧ 
      (d 7) ^ 2 + (d 10) ^ 2 = (n / d 22) ^ 2 →
      n = 2^3 * 3 * 5 * 17 :=
sorry

end unique_n_for_given_divisors_l1553_155390


namespace alan_needs_more_wings_l1553_155378

theorem alan_needs_more_wings 
  (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) (target_wings : ℕ) : 
  kevin_wings = 64 → kevin_time = 8 → alan_rate = 5 → target_wings = 3 → 
  (kevin_wings / kevin_time < alan_rate + target_wings) :=
by
  intros kevin_eq time_eq rate_eq target_eq
  sorry

end alan_needs_more_wings_l1553_155378


namespace solution_set_inequality_l1553_155369

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x

axiom mono_increasing (x y : ℝ) (hxy : 0 < x ∧ x < y) : f x < f y

axiom f_2_eq_0 : f 2 = 0

theorem solution_set_inequality :
  { x : ℝ | (x - 1) * f x < 0 } = { x : ℝ | -2 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } :=
by {
  sorry
}

end solution_set_inequality_l1553_155369


namespace ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l1553_155396

theorem ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : 0 < a * b * c)
  : a * b + b * c + c * a < (Real.sqrt (a * b * c)) / 2 + 1 / 4 := 
sorry

end ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l1553_155396


namespace arrange_numbers_l1553_155304

namespace MathProofs

theorem arrange_numbers (a b : ℚ) (h1 : a > 0) (h2 : b < 0) (h3 : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by
  -- Proof to be completed
  sorry

end MathProofs

end arrange_numbers_l1553_155304


namespace simplify_expression_l1553_155341

theorem simplify_expression (a : ℝ) (h : a ≠ 1/2) : 1 - (2 / (1 + (2 * a) / (1 - 2 * a))) = 4 * a - 1 :=
by
  sorry

end simplify_expression_l1553_155341


namespace vector_addition_l1553_155309

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

-- State the theorem to prove 2a + b = (7, -3, 2)
theorem vector_addition : (2 • a + b) = (7, -3, 2) := by
  sorry

end vector_addition_l1553_155309


namespace Randy_blocks_used_l1553_155315

theorem Randy_blocks_used (blocks_tower : ℕ) (blocks_house : ℕ) (total_blocks_used : ℕ) :
  blocks_tower = 27 → blocks_house = 53 → total_blocks_used = (blocks_tower + blocks_house) → total_blocks_used = 80 :=
by
  sorry

end Randy_blocks_used_l1553_155315
