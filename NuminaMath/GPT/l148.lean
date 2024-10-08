import Mathlib

namespace product_of_five_consecutive_integers_not_square_l148_148712

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (ha : 0 < a) : ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) := sorry

end product_of_five_consecutive_integers_not_square_l148_148712


namespace even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l148_148823

theorem even_product_implies_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 = 0 → ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

theorem odd_product_implies_no_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 ≠ 0 → ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

end even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l148_148823


namespace x1x2_lt_one_l148_148420

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  |exp x - exp 1| + exp x + a * x

theorem x1x2_lt_one (a x1 x2 : ℝ) 
  (ha : a < -exp 1) 
  (hzero1 : f a x1 = 0) 
  (hzero2 : f a x2 = 0) 
  (h_order : x1 < x2) : x1 * x2 < 1 := 
sorry

end x1x2_lt_one_l148_148420


namespace gcd_9240_12240_33720_l148_148434

theorem gcd_9240_12240_33720 : Nat.gcd (Nat.gcd 9240 12240) 33720 = 240 := by
  sorry

end gcd_9240_12240_33720_l148_148434


namespace convex_quadrilateral_diagonal_l148_148889

theorem convex_quadrilateral_diagonal (P : ℝ) (d1 d2 : ℝ) (hP : P = 2004) (hd1 : d1 = 1001) :
  (d2 = 1 → False) ∧ 
  (d2 = 2 → True) ∧ 
  (d2 = 1001 → True) :=
by
  sorry

end convex_quadrilateral_diagonal_l148_148889


namespace percentage_of_men_l148_148023

theorem percentage_of_men (E M W : ℝ) 
  (h1 : M + W = E)
  (h2 : 0.5 * M + 0.1666666666666669 * W = 0.4 * E)
  (h3 : W = E - M) : 
  (M / E = 0.70) :=
by
  sorry

end percentage_of_men_l148_148023


namespace diameter_of_circle_l148_148026

theorem diameter_of_circle (a b : ℕ) (r : ℝ) (h_a : a = 6) (h_b : b = 8) (h_triangle : a^2 + b^2 = r^2) : r = 10 :=
by 
  rw [h_a, h_b] at h_triangle
  sorry

end diameter_of_circle_l148_148026


namespace smallest_possible_students_group_l148_148758

theorem smallest_possible_students_group 
  (students : ℕ) :
  (∀ n, 2 ≤ n ∧ n ≤ 15 → ∃ k, students = k * n) ∧
  ¬∃ k, students = k * 10 ∧ ¬∃ k, students = k * 25 ∧ ¬∃ k, students = k * 50 ∧
  ∀ m n, 1 ≤ m ∧ m ≤ 15 ∧ 1 ≤ n ∧ n ≤ 15 ∧ (students ≠ m * n) → (m = n ∨ m ≠ n)
  → students = 120 := sorry

end smallest_possible_students_group_l148_148758


namespace ax2_x_plus_1_positive_l148_148740

theorem ax2_x_plus_1_positive (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ (a > 1/4) :=
by {
  sorry
}

end ax2_x_plus_1_positive_l148_148740


namespace side_length_a_l148_148297

theorem side_length_a (a b c : ℝ) (B : ℝ) (h1 : a = c - 2 * a * Real.cos B) (h2 : c = 5) (h3 : 3 * a = 2 * b) :
  a = 4 := by
  sorry

end side_length_a_l148_148297


namespace arithmetic_progression_25th_term_l148_148259

def arithmetic_progression_nth_term (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_progression_25th_term : arithmetic_progression_nth_term 5 7 25 = 173 := by
  sorry

end arithmetic_progression_25th_term_l148_148259


namespace chocolate_chip_cookies_l148_148180

theorem chocolate_chip_cookies (chocolate_chips_per_recipe : ℕ) (num_recipes : ℕ) (total_chocolate_chips : ℕ) 
  (h1 : chocolate_chips_per_recipe = 2) 
  (h2 : num_recipes = 23) 
  (h3 : total_chocolate_chips = chocolate_chips_per_recipe * num_recipes) : 
  total_chocolate_chips = 46 :=
by
  rw [h1, h2] at h3
  exact h3

-- sorry

end chocolate_chip_cookies_l148_148180


namespace geometric_progression_first_term_one_l148_148388

theorem geometric_progression_first_term_one (a r : ℝ) (gp : ℕ → ℝ)
  (h_gp : ∀ n, gp n = a * r^(n - 1))
  (h_product_in_gp : ∀ i j, ∃ k, gp i * gp j = gp k) :
  a = 1 := 
sorry

end geometric_progression_first_term_one_l148_148388


namespace counting_unit_difference_l148_148455

-- Definitions based on conditions
def magnitude_equality : Prop := 75 = 75.0
def counting_unit_75 : Nat := 1
def counting_unit_75_0 : Nat := 1 / 10

-- Proof problem stating that 75 and 75.0 do not have the same counting units.
theorem counting_unit_difference : 
  ¬ (counting_unit_75 = counting_unit_75_0) :=
by sorry

end counting_unit_difference_l148_148455


namespace find_g_3_l148_148479

def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 3) : g 3 = 0 := 
by
  sorry

end find_g_3_l148_148479


namespace integer_part_inequality_l148_148943

theorem integer_part_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
 (h_cond : (x + y + z) * ((1 / x) + (1 / y) + (1 / z)) = (91 / 10)) :
  (⌊(x^3 + y^3 + z^3) * ((1 / x^3) + (1 / y^3) + (1 / z^3))⌋) = 9 :=
by
  -- proof here
  sorry

end integer_part_inequality_l148_148943


namespace sum_mod_20_l148_148781

/-- Define the elements that are summed. -/
def elements : List ℤ := [82, 83, 84, 85, 86, 87, 88, 89]

/-- The problem statement to prove. -/
theorem sum_mod_20 : (elements.sum % 20) = 15 := by
  sorry

end sum_mod_20_l148_148781


namespace sara_initial_peaches_l148_148513

variable (p : ℕ)

def initial_peaches (picked_peaches total_peaches : ℕ) :=
  total_peaches - picked_peaches

theorem sara_initial_peaches :
  initial_peaches 37 61 = 24 :=
by
  -- This follows directly from the definition of initial_peaches
  sorry

end sara_initial_peaches_l148_148513


namespace arithmetic_sequence_length_l148_148869

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a_1 d a_n : ℤ), a_1 = -3 ∧ d = 4 ∧ a_n = 45 → n = 13 :=
by
  sorry

end arithmetic_sequence_length_l148_148869


namespace white_tshirts_l148_148024

theorem white_tshirts (packages shirts_per_package : ℕ) (h1 : packages = 71) (h2 : shirts_per_package = 6) : packages * shirts_per_package = 426 := 
by 
  sorry

end white_tshirts_l148_148024


namespace eval_expr_l148_148131

theorem eval_expr : (900 ^ 2) / (262 ^ 2 - 258 ^ 2) = 389.4 := 
by
  sorry

end eval_expr_l148_148131


namespace directrix_of_parabola_l148_148691

theorem directrix_of_parabola (y x : ℝ) (p : ℝ) (h₁ : y = 8 * x ^ 2) (h₂ : y = 4 * p * x) : 
  p = 2 ∧ (y = -p ↔ y = -2) :=
by
  sorry

end directrix_of_parabola_l148_148691


namespace am_gm_inequality_example_l148_148912

theorem am_gm_inequality_example (x y : ℝ) (hx : x = 16) (hy : y = 64) : 
  (x + y) / 2 ≥ Real.sqrt (x * y) :=
by
  rw [hx, hy]
  sorry

end am_gm_inequality_example_l148_148912


namespace seonho_original_money_l148_148792

variable (X : ℝ)
variable (spent_snacks : ℝ := (1/4) * X)
variable (remaining_after_snacks : ℝ := X - spent_snacks)
variable (spent_food : ℝ := (2/3) * remaining_after_snacks)
variable (final_remaining : ℝ := remaining_after_snacks - spent_food)

theorem seonho_original_money :
  final_remaining = 2500 -> X = 10000 := by
  -- Proof goes here
  sorry

end seonho_original_money_l148_148792


namespace general_term_of_sequence_l148_148216

theorem general_term_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S (n + 1) = 3 * (n + 1) ^ 2 - 2 * (n + 1)) →
  a 1 = 1 →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  (∀ n, a n = 6 * n - 5) := 
by
  intros hS ha1 ha
  sorry

end general_term_of_sequence_l148_148216


namespace line_equation_intercept_twice_x_intercept_l148_148932

theorem line_equation_intercept_twice_x_intercept 
  {x y : ℝ}
  (intersection_point : ∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0) 
  (y_intercept_is_twice_x_intercept : ∃ (a : ℝ), ∀ (x y : ℝ), y = 2 * a ∧ x = a) :
  (∃ (x y : ℝ), 2 * x - 3 * y = 0) ∨ (∃ (x y : ℝ), 2 * x + y - 8 = 0) :=
sorry

end line_equation_intercept_twice_x_intercept_l148_148932


namespace factory_ill_days_l148_148154

theorem factory_ill_days
  (average_first_25_days : ℝ)
  (total_days : ℝ)
  (overall_average : ℝ)
  (ill_days_average : ℝ)
  (production_first_25_days_total : ℝ)
  (production_ill_days_total : ℝ)
  (x : ℝ) :
  average_first_25_days = 50 →
  total_days = 25 + x →
  overall_average = 48 →
  ill_days_average = 38 →
  production_first_25_days_total = 25 * 50 →
  production_ill_days_total = x * 38 →
  (25 * 50 + x * 38 = (25 + x) * 48) →
  x = 5 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end factory_ill_days_l148_148154


namespace person_before_you_taller_than_you_l148_148689

-- Define the persons involved in the problem.
variable (Person : Type)
variable (Taller : Person → Person → Prop)
variable (P Q You : Person)

-- The conditions given in the problem.
axiom standing_queue : Taller P Q
axiom queue_structure : You = Q

-- The question we need to prove, which is the correct answer to the problem.
theorem person_before_you_taller_than_you : Taller P You :=
by
  sorry

end person_before_you_taller_than_you_l148_148689


namespace peter_runs_more_than_andrew_each_day_l148_148555

-- Define the constants based on the conditions
def miles_andrew : ℕ := 2
def total_days : ℕ := 5
def total_miles : ℕ := 35

-- Define a theorem to prove the number of miles Peter runs more than Andrew each day
theorem peter_runs_more_than_andrew_each_day : 
  ∃ x : ℕ, total_days * (miles_andrew + x) + total_days * miles_andrew = total_miles ∧ x = 3 :=
by
  sorry

end peter_runs_more_than_andrew_each_day_l148_148555


namespace determine_disco_ball_price_l148_148128

variable (x y z : ℝ)

-- Given conditions
def budget_constraint : Prop := 4 * x + 10 * y + 20 * z = 600
def food_cost : Prop := y = 0.85 * x
def decoration_cost : Prop := z = x / 2 - 10

-- Goal
theorem determine_disco_ball_price (h1 : budget_constraint x y z) (h2 : food_cost x y) (h3 : decoration_cost x z) :
  x = 35.56 :=
sorry 

end determine_disco_ball_price_l148_148128


namespace sum_of_squares_of_roots_l148_148977

theorem sum_of_squares_of_roots :
  ∀ (r₁ r₂ : ℝ), (r₁ + r₂ = 15) → (r₁ * r₂ = 6) → (r₁^2 + r₂^2 = 213) :=
by
  intros r₁ r₂ h_sum h_prod
  -- Proof goes here, but skipping it for now
  sorry

end sum_of_squares_of_roots_l148_148977


namespace value_of_angle_C_perimeter_range_l148_148222

-- Part (1): Prove angle C value
theorem value_of_angle_C
  {a b c : ℝ} {A B C : ℝ}
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (m : ℝ × ℝ := (Real.sin C, Real.cos C))
  (n : ℝ × ℝ := (2 * Real.sin A - Real.cos B, -Real.sin B))
  (orthogonal_mn : m.1 * n.1 + m.2 * n.2 = 0) 
  : C = π / 6 := sorry

-- Part (2): Prove perimeter range
theorem perimeter_range
  {a b c : ℝ} {A B C : ℝ}
  (A_range : π / 3 < A ∧ A < π / 2)
  (C_value : C = π / 6)
  (a_value : a = 2)
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  : 3 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c < 2 + 3 * Real.sqrt 3 := sorry

end value_of_angle_C_perimeter_range_l148_148222


namespace problem_1_problem_2_problem_3_l148_148810

section Problem

-- Initial conditions
variable (a : ℕ → ℝ) (t m : ℝ)
def a_1 : ℝ := 3
def a_n (n : ℕ) (h : 2 ≤ n) : ℝ := 2 * a (n - 1) + (t + 1) * 2^n + 3 * m + t

-- Problem 1:
theorem problem_1 (h : t = 0) (h' : m = 0) :
  ∃ d, ∀ n, 2 ≤ n → (a n / 2^n) = (a (n - 1) / 2^(n-1)) + d := sorry

-- Problem 2:
theorem problem_2 (h : t = -1) (h' : m = 4/3) :
  ∃ r, ∀ n, 2 ≤ n → a n + 3 = r * (a (n - 1) + 3) := sorry

-- Problem 3:
theorem problem_3 (h : t = 0) (h' : m = 1) :
  (∀ n, 1 ≤ n → a n = (n + 2) * 2^n - 3) ∧
  (∃ S : ℕ → ℝ, ∀ n, S n = (n + 1) * 2^(n + 1) - 2 - 3 * n) := sorry

end Problem

end problem_1_problem_2_problem_3_l148_148810


namespace unique_solution_positive_n_l148_148451

theorem unique_solution_positive_n (n : ℝ) : 
  ( ∃ x : ℝ, 4 * x^2 + n * x + 16 = 0 ∧ ∀ y : ℝ, 4 * y^2 + n * y + 16 = 0 → y = x ) → n = 16 := 
by {
  sorry
}

end unique_solution_positive_n_l148_148451


namespace max_cookies_andy_could_have_eaten_l148_148510

theorem max_cookies_andy_could_have_eaten (x k : ℕ) (hk : k > 0) 
  (h_total : x + k * x + 2 * x = 36) : x ≤ 9 :=
by
  -- Using the conditions to construct the proof (which is not required based on the instructions)
  sorry

end max_cookies_andy_could_have_eaten_l148_148510


namespace solve_equation_l148_148280

theorem solve_equation (x : ℤ) : x * (x + 2) + 1 = 36 ↔ x = 5 :=
by sorry

end solve_equation_l148_148280


namespace probability_two_or_more_women_l148_148084

-- Definitions based on the conditions
def men : ℕ := 8
def women : ℕ := 4
def total_people : ℕ := men + women
def chosen_people : ℕ := 4

-- Function to calculate the probability of a specific event
noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ :=
  event_count / total_count

-- Function to calculate the combination (binomial coefficient)
noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability calculations based on steps given in the solution:
noncomputable def prob_no_women : ℚ :=
  probability_event ((men - 0) * (men - 1) * (men - 2) * (men - 3)) (total_people * (total_people - 1) * (total_people - 2) * (total_people - 3))

noncomputable def prob_exactly_one_woman : ℚ :=
  probability_event (binom women 1 * binom men 3) (binom total_people chosen_people)

noncomputable def prob_fewer_than_two_women : ℚ :=
  prob_no_women + prob_exactly_one_woman

noncomputable def prob_at_least_two_women : ℚ :=
  1 - prob_fewer_than_two_women

-- The main theorem to be proved
theorem probability_two_or_more_women :
  prob_at_least_two_women = 67 / 165 :=
sorry

end probability_two_or_more_women_l148_148084


namespace bob_eats_10_apples_l148_148284

variable (B C : ℕ)
variable (h1 : B + C = 30)
variable (h2 : C = 2 * B)

theorem bob_eats_10_apples : B = 10 :=
by sorry

end bob_eats_10_apples_l148_148284


namespace stretched_curve_l148_148853

noncomputable def transformed_curve (x : ℝ) : ℝ :=
  2 * Real.sin (x / 3 + Real.pi / 3)

theorem stretched_curve (y x : ℝ) :
  y = 2 * Real.sin (x + Real.pi / 3) → y = transformed_curve x := by
  intro h
  sorry

end stretched_curve_l148_148853


namespace solution_set_inequality_l148_148435

noncomputable def f (x : ℝ) : ℝ := x * (1 - 3 * x)

theorem solution_set_inequality : {x : ℝ | f x > 0} = { x | (0 < x) ∧ (x < 1/3) } := by
  sorry

end solution_set_inequality_l148_148435


namespace no_intersection_of_sets_l148_148359

noncomputable def A (a b x y : ℝ) :=
  a * (Real.sin x + Real.sin y) + (b - 1) * (Real.cos x + Real.cos y) = 0

noncomputable def B (a b x y : ℝ) :=
  (b + 1) * Real.sin (x + y) - a * Real.cos (x + y) = a

noncomputable def C (a b : ℝ) :=
  ∀ z : ℝ, z^2 - 2 * (a - b) * z + (a + b)^2 - 2 > 0

theorem no_intersection_of_sets (a b x y : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : 0 < y) (h4 : y < Real.pi / 2) :
  (C a b) → ¬(∃ x y, A a b x y ∧ B a b x y) :=
by 
  sorry

end no_intersection_of_sets_l148_148359


namespace max_integer_valued_fractions_l148_148763

-- Problem Statement:
-- Given a set of natural numbers from 1 to 22,
-- the maximum number of fractions that can be formed such that each fraction is an integer
-- (where an integer fraction is defined as a/b being an integer if and only if b divides a) is 10.

open Nat

theorem max_integer_valued_fractions : 
  ∀ (S : Finset ℕ), (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 22) →
  ∃ P : Finset (ℕ × ℕ), (∀ (a b : ℕ), (a, b) ∈ P → b ∣ a) ∧ P.card = 11 → 
  10 ≤ (P.filter (λ p => p.1 % p.2 = 0)).card :=
by
  -- proof goes here
  sorry

end max_integer_valued_fractions_l148_148763


namespace inclination_angle_of_line_m_l148_148481

theorem inclination_angle_of_line_m
  (m : ℝ → ℝ → Prop)
  (l₁ l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ x - y + 1 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x - y - 1 = 0)
  (intersect_segment_length : ℝ)
  (h₃ : intersect_segment_length = 2 * Real.sqrt 2) :
  (∃ α : ℝ, (α = 15 ∨ α = 75) ∧ (∃ k : ℝ, ∀ x y, m x y ↔ y = k * x)) :=
by
  sorry

end inclination_angle_of_line_m_l148_148481


namespace greatest_least_S_T_l148_148352

theorem greatest_least_S_T (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 :=
by sorry

end greatest_least_S_T_l148_148352


namespace find_pairs_l148_148642

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l148_148642


namespace polynomial_factor_pair_l148_148381

theorem polynomial_factor_pair (a b : ℝ) :
  (∃ (c d : ℝ), 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 6)) →
  (a, b) = (-26.5, -40) :=
by
  sorry

end polynomial_factor_pair_l148_148381


namespace inequality_proof_l148_148735

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l148_148735


namespace parabola_directrix_l148_148418

theorem parabola_directrix (a : ℝ) :
  (∃ y : ℝ, y = ax^2 ∧ y = -2) → a = 1/8 :=
by
  -- Solution steps are omitted.
  sorry

end parabola_directrix_l148_148418


namespace determine_sum_of_digits_l148_148658

theorem determine_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10)
  (h : ∃ a b c d : ℕ, 
       a = 30 + x ∧ b = 10 * y + 4 ∧
       c = (a * (b % 10)) % 100 ∧ 
       d = (a * (b % 10)) / 100 ∧ 
       10 * d + c = 156) :
  x + y = 13 :=
by
  sorry

end determine_sum_of_digits_l148_148658


namespace vanya_speed_l148_148761

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l148_148761


namespace survey_min_people_l148_148046

theorem survey_min_people (p : ℕ) : 
  (∃ p, ∀ k ∈ [18, 10, 5, 9], k ∣ p) → p = 90 :=
by sorry

end survey_min_people_l148_148046


namespace minimum_quotient_value_l148_148271

-- Helper definition to represent the quotient 
def quotient (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d) / (a + b + c + d)

-- Conditions: digits are distinct and non-zero 
def distinct_and_nonzero (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem minimum_quotient_value :
  ∀ (a b c d : ℕ), distinct_and_nonzero a b c d → quotient a b c d = 71.9 :=
by sorry

end minimum_quotient_value_l148_148271


namespace abs_inequality_range_l148_148264

theorem abs_inequality_range (x : ℝ) (b : ℝ) (h : 0 < b) : (b > 2) ↔ ∃ x : ℝ, |x - 5| + |x - 7| < b :=
sorry

end abs_inequality_range_l148_148264


namespace quadratic_function_passing_through_origin_l148_148580

-- Define the quadratic function y
def quadratic_function (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

-- State the problem as a theorem
theorem quadratic_function_passing_through_origin (m : ℝ) (h: quadratic_function m 0 = 0) : m = -4 :=
by
  -- Since we only need the statement, we put sorry here
  sorry

end quadratic_function_passing_through_origin_l148_148580


namespace simplify_expression_l148_148298

theorem simplify_expression (x : ℝ) :
  (3 * x)^5 + (4 * x^2) * (3 * x^2) = 243 * x^5 + 12 * x^4 :=
by
  sorry

end simplify_expression_l148_148298


namespace factorization_problem1_factorization_problem2_l148_148140

-- Define the first problem: Factorization of 3x^2 - 27
theorem factorization_problem1 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) :=
by
  sorry 

-- Define the second problem: Factorization of (a + 1)(a - 5) + 9
theorem factorization_problem2 (a : ℝ) : (a + 1) * (a - 5) + 9 = (a - 2) ^ 2 :=
by
  sorry

end factorization_problem1_factorization_problem2_l148_148140


namespace find_x_for_given_y_l148_148087

theorem find_x_for_given_y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_initial : x = 2 ∧ y = 8) (h_inverse : (2 ^ 3) * 8 = 128) :
  y = 1728 → x = (1 / (13.5) ^ (1 / 3)) :=
by
  sorry

end find_x_for_given_y_l148_148087


namespace find_number_l148_148648

theorem find_number (N p q : ℝ) (h₁ : N / p = 8) (h₂ : N / q = 18) (h₃ : p - q = 0.2777777777777778) : N = 4 :=
sorry

end find_number_l148_148648


namespace find_x_l148_148313

-- Definitions of the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_a_minus_b (x : ℝ) : ℝ × ℝ := ((1 - x), (4))

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The given condition of perpendicular vectors
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The theorem to prove
theorem find_x : ∃ x : ℝ, is_perpendicular vector_a (vector_a_minus_b x) ∧ x = 9 :=
by {
  -- Sorry statement used to skip proof
  sorry
}

end find_x_l148_148313


namespace algebraic_expression_opposite_l148_148018

theorem algebraic_expression_opposite (a b x : ℝ) (h : b^2 * x^2 + |a| = -(b^2 * x^2 + |a|)) : a * b = 0 :=
by 
  sorry

end algebraic_expression_opposite_l148_148018


namespace complement_union_l148_148651

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4, 5}

theorem complement_union :
  (U \ A) ∪ (U \ B) = {1, 2, 3, 6} := 
by 
  sorry

end complement_union_l148_148651


namespace money_given_to_last_set_l148_148270

theorem money_given_to_last_set (total first second third fourth last : ℝ) 
  (h_total : total = 4500) 
  (h_first : first = 725) 
  (h_second : second = 1100) 
  (h_third : third = 950) 
  (h_fourth : fourth = 815) 
  (h_sum: total = first + second + third + fourth + last) : 
  last = 910 :=
sorry

end money_given_to_last_set_l148_148270


namespace relationship_among_abc_l148_148212

noncomputable def a : ℝ := 20.3
noncomputable def b : ℝ := 0.32
noncomputable def c : ℝ := Real.log 25 / Real.log 10

theorem relationship_among_abc : b < a ∧ a < c :=
by
  -- Proof needs to be filled in here
  sorry

end relationship_among_abc_l148_148212


namespace trig_expression_value_l148_148569

theorem trig_expression_value
  (x : ℝ)
  (h : Real.tan (x + Real.pi / 4) = -3) :
  (Real.sin x + 2 * Real.cos x) / (3 * Real.sin x + 4 * Real.cos x) = 2 / 5 :=
by
  sorry

end trig_expression_value_l148_148569


namespace divisor_in_second_division_is_19_l148_148061

theorem divisor_in_second_division_is_19 (n d : ℕ) (h1 : n % 25 = 4) (h2 : (n + 15) % d = 4) : d = 19 :=
sorry

end divisor_in_second_division_is_19_l148_148061


namespace total_enemies_l148_148601

theorem total_enemies (E : ℕ) (h : 8 * (E - 2) = 40) : E = 7 := sorry

end total_enemies_l148_148601


namespace x_solves_quadratic_and_sum_is_75_l148_148064

theorem x_solves_quadratic_and_sum_is_75
  (x a b : ℕ) (h : x^2 + 10 * x = 45) (hx_pos : 0 < x) (hx_form : x = Nat.sqrt a - b) 
  (ha_pos : 0 < a) (hb_pos : 0 < b)
  : a + b = 75 := 
sorry

end x_solves_quadratic_and_sum_is_75_l148_148064


namespace find_exponent_l148_148661

theorem find_exponent (n : ℕ) (some_number : ℕ) (h1 : n = 27) 
  (h2 : 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) = 4 ^ some_number) :
  some_number = 28 :=
by 
  sorry

end find_exponent_l148_148661


namespace volume_of_hemisphere_l148_148726

theorem volume_of_hemisphere (d : ℝ) (h : d = 10) : 
  let r := d / 2
  let V := (2 / 3) * π * r^3
  V = 250 / 3 * π := by
sorry

end volume_of_hemisphere_l148_148726


namespace sum_of_roots_l148_148937

theorem sum_of_roots : 
  ( ∀ x : ℝ, x^2 - 7*x + 10 = 0 → x = 2 ∨ x = 5 ) → 
  ( 2 + 5 = 7 ) := 
by
  sorry

end sum_of_roots_l148_148937


namespace area_of_field_l148_148338

theorem area_of_field (w l A : ℝ) 
    (h1 : l = 2 * w + 35) 
    (h2 : 2 * (w + l) = 700) : 
    A = 25725 :=
by sorry

end area_of_field_l148_148338


namespace rate_of_stream_l148_148846

-- Definitions from problem conditions
def rowing_speed_still_water : ℕ := 24

-- Assume v is the rate of the stream
variable (v : ℕ)

-- Time taken to row up is three times the time taken to row down
def rowing_time_condition : Prop :=
  1 / (rowing_speed_still_water - v) = 3 * (1 / (rowing_speed_still_water + v))

-- The rate of the stream (v) should be 12 kmph
theorem rate_of_stream (h : rowing_time_condition v) : v = 12 :=
  sorry

end rate_of_stream_l148_148846


namespace opposite_of_minus_one_third_l148_148124

theorem opposite_of_minus_one_third :
  -(- (1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_minus_one_third_l148_148124


namespace minimum_value_of_expression_l148_148886

theorem minimum_value_of_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 3) : 
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 72 :=
sorry

end minimum_value_of_expression_l148_148886


namespace initial_population_l148_148133

theorem initial_population (P : ℝ) 
    (h1 : 1.25 * P * 0.70 = 363650) : 
    P = 415600 :=
sorry

end initial_population_l148_148133


namespace linear_function_increasing_l148_148165

theorem linear_function_increasing (x1 x2 y1 y2 : ℝ) (h1 : y1 = 2 * x1 - 1) (h2 : y2 = 2 * x2 - 1) (h3 : x1 > x2) : y1 > y2 :=
by
  sorry

end linear_function_increasing_l148_148165


namespace find_m_ineq_soln_set_min_value_a2_b2_l148_148364

-- Problem 1
theorem find_m_ineq_soln_set (m x : ℝ) (h1 : m - |x - 2| ≥ 1) (h2 : x ∈ Set.Icc 0 4) : m = 3 := by
  sorry

-- Problem 2
theorem min_value_a2_b2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) : a^2 + b^2 ≥ 9 / 2 := by
  sorry

end find_m_ineq_soln_set_min_value_a2_b2_l148_148364


namespace base_of_first_term_l148_148787

-- Define the necessary conditions
def equation (x s : ℝ) : Prop :=
  x^16 * 25^s = 5 * 10^16

-- The proof goal
theorem base_of_first_term (x s : ℝ) (h : equation x s) : x = 2 / 5 :=
by
  sorry

end base_of_first_term_l148_148787


namespace positive_real_solution_l148_148972

def polynomial (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem positive_real_solution (h : polynomial 1 = 0) : polynomial 1 > 0 := sorry

end positive_real_solution_l148_148972


namespace fundraiser_contribution_l148_148702

theorem fundraiser_contribution :
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  total_muffins * price_per_muffin = 900 :=
by
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  sorry

end fundraiser_contribution_l148_148702


namespace cookies_left_at_end_of_week_l148_148292

def trays_baked_each_day : List Nat := [2, 3, 4, 5, 3, 4, 4]
def cookies_per_tray : Nat := 12
def cookies_eaten_by_frank : Nat := 2 * 7
def cookies_eaten_by_ted : Nat := 3 + 5
def cookies_eaten_by_jan : Nat := 5
def cookies_eaten_by_tom : Nat := 8
def cookies_eaten_by_neighbours_kids : Nat := 20

def total_cookies_baked : Nat :=
  (trays_baked_each_day.map (λ trays => trays * cookies_per_tray)).sum

def total_cookies_eaten : Nat :=
  cookies_eaten_by_frank + cookies_eaten_by_ted + cookies_eaten_by_jan +
  cookies_eaten_by_tom + cookies_eaten_by_neighbours_kids

def cookies_left : Nat := total_cookies_baked - total_cookies_eaten

theorem cookies_left_at_end_of_week : cookies_left = 245 :=
by
  sorry

end cookies_left_at_end_of_week_l148_148292


namespace shirts_per_minute_l148_148237

theorem shirts_per_minute (S : ℕ) 
  (h1 : 12 * S + 14 = 156) : S = 11 := 
by
  sorry

end shirts_per_minute_l148_148237


namespace tangent_line_at_P_exists_c_for_a_l148_148191

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_P :
  ∀ x y : ℝ, y = f x → x = 1 → y = 0 → x - y - 1 = 0 := 
by 
  sorry

theorem exists_c_for_a :
  ∀ a : ℝ, 1 < a → ∃ c : ℝ, 0 < c ∧ c < 1 / a ∧ ∀ x : ℝ, c < x → x < 1 → f x > a * x * (x - 1) :=
by 
  sorry

end tangent_line_at_P_exists_c_for_a_l148_148191


namespace cookies_per_day_l148_148568

theorem cookies_per_day (cost_per_cookie : ℕ) (total_spent : ℕ) (days_in_march : ℕ) (h1 : cost_per_cookie = 16) (h2 : total_spent = 992) (h3 : days_in_march = 31) :
  (total_spent / cost_per_cookie) / days_in_march = 2 :=
by sorry

end cookies_per_day_l148_148568


namespace ratio_greater_than_one_ratio_greater_than_one_neg_l148_148806

theorem ratio_greater_than_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b > 1) : a > b :=
by
  sorry

theorem ratio_greater_than_one_neg (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a / b > 1) : a < b :=
by
  sorry

end ratio_greater_than_one_ratio_greater_than_one_neg_l148_148806


namespace find_p_l148_148157

variables (p q : ℚ)
variables (h1 : 2 * p + 5 * q = 10) (h2 : 5 * p + 2 * q = 20)

theorem find_p : p = 80 / 21 :=
by sorry

end find_p_l148_148157


namespace find_number_l148_148769

theorem find_number (a b x : ℝ) (H1 : 2 * a = x * b) (H2 : a * b ≠ 0) (H3 : (a / 3) / (b / 2) = 1) : x = 3 :=
by
  sorry

end find_number_l148_148769


namespace remainder_of_base12_2563_mod_17_l148_148047

-- Define the base-12 number 2563 in decimal.
def base12_to_decimal : ℕ := 2 * 12^3 + 5 * 12^2 + 6 * 12^1 + 3 * 12^0

-- Define the number 17.
def divisor : ℕ := 17

-- Prove that the remainder when base12_to_decimal is divided by divisor is 1.
theorem remainder_of_base12_2563_mod_17 : base12_to_decimal % divisor = 1 :=
by
  sorry

end remainder_of_base12_2563_mod_17_l148_148047


namespace num_positive_integers_l148_148189

-- Definitions
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Problem statement
theorem num_positive_integers (n : ℕ) (h : n = 2310) :
  (∃ count, count = 3 ∧ (∀ m : ℕ, m > 0 → is_divisor (m^2 - 2) n → count = 3)) := by
  sorry

end num_positive_integers_l148_148189


namespace division_remainder_example_l148_148486

theorem division_remainder_example :
  ∃ n, n = 20 * 10 + 10 ∧ n = 210 :=
by
  sorry

end division_remainder_example_l148_148486


namespace max_a4_l148_148558

theorem max_a4 (a1 d a4 : ℝ) 
  (h1 : 2 * a1 + 3 * d ≥ 5) 
  (h2 : a1 + 2 * d ≤ 3) 
  (ha4 : a4 = a1 + 3 * d) : 
  a4 ≤ 4 := 
by 
  sorry

end max_a4_l148_148558


namespace ratio_of_areas_of_squares_l148_148858

theorem ratio_of_areas_of_squares (a_side b_side : ℕ) (h_a : a_side = 36) (h_b : b_side = 42) : 
  (a_side ^ 2 : ℚ) / (b_side ^ 2 : ℚ) = 36 / 49 :=
by
  sorry

end ratio_of_areas_of_squares_l148_148858


namespace arithmetic_sequence_common_difference_l148_148999

theorem arithmetic_sequence_common_difference
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (a₁ d : ℤ)
  (h1 : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h2 : ∀ n, a n = a₁ + (n - 1) * d)
  (h3 : S 5 = 5 * (a 4) - 10) :
  d = 2 := sorry

end arithmetic_sequence_common_difference_l148_148999


namespace janice_bottle_caps_l148_148445

-- Define the conditions
def num_boxes : ℕ := 79
def caps_per_box : ℕ := 4

-- Define the question as a theorem to prove
theorem janice_bottle_caps : num_boxes * caps_per_box = 316 :=
by
  sorry

end janice_bottle_caps_l148_148445


namespace find_y_l148_148300

noncomputable def x : Real := 2.6666666666666665

theorem find_y (y : Real) (h : (x * y) / 3 = x^2) : y = 8 :=
sorry

end find_y_l148_148300


namespace rowing_distance_l148_148619

theorem rowing_distance (D : ℝ) : 
  (D / 14 + D / 2 = 120) → D = 210 := by
  sorry

end rowing_distance_l148_148619


namespace three_digit_number_divisible_by_8_and_even_tens_digit_l148_148372

theorem three_digit_number_divisible_by_8_and_even_tens_digit (d : ℕ) (hd : d % 2 = 0) (hdiv : (100 * 5 + 10 * d + 4) % 8 = 0) :
  100 * 5 + 10 * d + 4 = 544 :=
by
  sorry

end three_digit_number_divisible_by_8_and_even_tens_digit_l148_148372


namespace min_values_of_exprs_l148_148946

theorem min_values_of_exprs (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (h : (r + s - r * s) * (r + s + r * s) = r * s) :
  (r + s - r * s) = -3 + 2 * Real.sqrt 3 ∧ (r + s + r * s) = 3 + 2 * Real.sqrt 3 :=
by sorry

end min_values_of_exprs_l148_148946


namespace gcd_polynomial_l148_148634

open Nat

theorem gcd_polynomial (b : ℤ) (hb : 1632 ∣ b) : gcd (b^2 + 11 * b + 30) (b + 6) = 6 := by
  sorry

end gcd_polynomial_l148_148634


namespace avg_transformation_l148_148478

theorem avg_transformation
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) :
  ((3 * x₁ + 1) + (3 * x₂ + 1) + (3 * x₃ + 1) + (3 * x₄ + 1) + (3 * x₅ + 1)) / 5 = 7 :=
by
  sorry

end avg_transformation_l148_148478


namespace value_expression_possible_values_l148_148090

open Real

noncomputable def value_expression (a b : ℝ) : ℝ :=
  a^2 + 2 * a * b + b^2 + 2 * a^2 * b + 2 * a * b^2 + a^2 * b^2

theorem value_expression_possible_values (a b : ℝ)
  (h1 : (a / b) + (b / a) = 5 / 2)
  (h2 : a - b = 3 / 2) :
  value_expression a b = 0 ∨ value_expression a b = 81 :=
sorry

end value_expression_possible_values_l148_148090


namespace four_gt_sqrt_fourteen_l148_148997

theorem four_gt_sqrt_fourteen : 4 > Real.sqrt 14 := 
  sorry

end four_gt_sqrt_fourteen_l148_148997


namespace evaporation_period_l148_148875

theorem evaporation_period
  (initial_amount : ℚ)
  (evaporation_rate : ℚ)
  (percentage_evaporated : ℚ)
  (actual_days : ℚ)
  (h_initial : initial_amount = 10)
  (h_evap_rate : evaporation_rate = 0.007)
  (h_percentage : percentage_evaporated = 3.5000000000000004)
  (h_days : actual_days = (percentage_evaporated / 100) * initial_amount / evaporation_rate):
  actual_days = 50 := by
  sorry

end evaporation_period_l148_148875


namespace expected_value_of_winnings_l148_148739

noncomputable def winnings (n : ℕ) : ℕ := 2 * n - 1

theorem expected_value_of_winnings : 
  (1 / 6 : ℚ) * ((winnings 1) + (winnings 2) + (winnings 3) + (winnings 4) + (winnings 5) + (winnings 6)) = 6 :=
by
  sorry

end expected_value_of_winnings_l148_148739


namespace odd_function_f_a_zero_l148_148682

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (a + 1) * Real.cos x + x

theorem odd_function_f_a_zero (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : f a a = 0 := 
sorry

end odd_function_f_a_zero_l148_148682


namespace solve_for_x_l148_148615

-- Let us state and prove that x = 495 / 13 is a solution to the equation 3x + 5 = 500 - (4x + 6x)
theorem solve_for_x (x : ℝ) : 3 * x + 5 = 500 - (4 * x + 6 * x) → x = 495 / 13 :=
by
  sorry

end solve_for_x_l148_148615


namespace AC_eq_200_l148_148871

theorem AC_eq_200 (A B C : ℕ) (h1 : A + B + C = 500) (h2 : B + C = 330) (h3 : C = 30) : A + C = 200 := by
  sorry

end AC_eq_200_l148_148871


namespace maximum_marks_l148_148777

theorem maximum_marks (M : ℝ) :
  (0.45 * M = 80) → (M = 180) :=
by
  sorry

end maximum_marks_l148_148777


namespace fair_coin_second_head_l148_148742

theorem fair_coin_second_head (P : ℝ) 
  (fair_coin : ∀ outcome : ℝ, outcome = 0.5) :
  P = 0.5 :=
by
  sorry

end fair_coin_second_head_l148_148742


namespace pure_imaginary_condition_l148_148176

theorem pure_imaginary_condition (a b : ℝ) : 
  (a = 0) ↔ (∃ b : ℝ, b ≠ 0 ∧ z = a + b * I) :=
sorry

end pure_imaginary_condition_l148_148176


namespace correct_quotient_remainder_sum_l148_148166

theorem correct_quotient_remainder_sum :
  ∃ N : ℕ, (N % 23 = 17 ∧ N / 23 = 3) ∧ (∃ q r : ℕ, N = 32 * q + r ∧ r < 32 ∧ q + r = 24) :=
by
  sorry

end correct_quotient_remainder_sum_l148_148166


namespace rhombus_shorter_diagonal_l148_148312

variable (d1 d2 : ℝ) (Area : ℝ)

def is_rhombus (Area : ℝ) (d1 d2 : ℝ) : Prop := Area = (d1 * d2) / 2

theorem rhombus_shorter_diagonal
  (h_d2 : d2 = 20)
  (h_Area : Area = 110)
  (h_rhombus : is_rhombus Area d1 d2) :
  d1 = 11 := by
  sorry

end rhombus_shorter_diagonal_l148_148312


namespace cistern_fill_time_l148_148543

theorem cistern_fill_time (F : ℝ) (E : ℝ) (net_rate : ℝ) (time : ℝ)
  (h_F : F = 1 / 4)
  (h_E : E = 1 / 8)
  (h_net : net_rate = F - E)
  (h_time : time = 1 / net_rate) :
  time = 8 := 
sorry

end cistern_fill_time_l148_148543


namespace tallest_giraffe_height_l148_148848

theorem tallest_giraffe_height :
  ∃ (height : ℕ), height = 96 ∧ (height = 68 + 28) := by
  sorry

end tallest_giraffe_height_l148_148848


namespace larry_channel_reduction_l148_148609

theorem larry_channel_reduction
  (initial_channels new_channels final_channels sports_package supreme_sports_package channels_at_end : ℕ)
  (h_initial : initial_channels = 150)
  (h_adjustment : new_channels = initial_channels - 20 + 12)
  (h_sports : sports_package = 8)
  (h_supreme_sports : supreme_sports_package = 7)
  (h_channels_at_end : channels_at_end = 147)
  (h_final : final_channels = channels_at_end - sports_package - supreme_sports_package) :
  initial_channels - 20 + 12 - final_channels = 10 := 
sorry

end larry_channel_reduction_l148_148609


namespace allan_plums_l148_148646

theorem allan_plums (A : ℕ) (h1 : 7 - A = 3) : A = 4 :=
sorry

end allan_plums_l148_148646


namespace conference_duration_l148_148576

theorem conference_duration (hours minutes lunch_break total_minutes active_session : ℕ) 
  (h1 : hours = 8) 
  (h2 : minutes = 40) 
  (h3 : lunch_break = 15) 
  (h4 : total_minutes = hours * 60 + minutes)
  (h5 : active_session = total_minutes - lunch_break) :
  active_session = 505 := 
by {
  sorry
}

end conference_duration_l148_148576


namespace initial_violet_balloons_l148_148885

-- Define initial conditions and variables
def red_balloons := 4
def violet_balloons_lost := 3
def current_violet_balloons := 4

-- Define the theorem we want to prove
theorem initial_violet_balloons (red_balloons : ℕ) (violet_balloons_lost : ℕ) (current_violet_balloons : ℕ) : 
  red_balloons = 4 → violet_balloons_lost = 3 → current_violet_balloons = 4 → (current_violet_balloons + violet_balloons_lost) = 7 :=
by
  intros
  sorry

end initial_violet_balloons_l148_148885


namespace number_of_valid_triangles_l148_148143

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l148_148143


namespace abs_diff_roots_eq_sqrt_13_l148_148538

theorem abs_diff_roots_eq_sqrt_13 {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  |x₁ - x₂| = Real.sqrt 13 :=
sorry

end abs_diff_roots_eq_sqrt_13_l148_148538


namespace trigonometric_identity_l148_148638

theorem trigonometric_identity
  (θ : ℝ) 
  (h_tan : Real.tan θ = 3) :
  (1 - Real.cos θ) / (Real.sin θ) - (Real.sin θ) / (1 + (Real.cos θ)^2) = (11 * Real.sqrt 10 - 101) / 33 := 
by
  sorry

end trigonometric_identity_l148_148638


namespace xiaoming_wait_probability_l148_148102

-- Conditions
def green_light_duration : ℕ := 40
def red_light_duration : ℕ := 50
def total_light_cycle : ℕ := green_light_duration + red_light_duration
def waiting_time_threshold : ℕ := 20
def long_wait_interval : ℕ := 30 -- from problem (20 seconds to wait corresponds to 30 seconds interval)

-- Probability calculation
theorem xiaoming_wait_probability :
  ∀ (arrival_time : ℕ), arrival_time < total_light_cycle →
    (30 : ℝ) / (total_light_cycle : ℝ) = 1 / 3 := by sorry

end xiaoming_wait_probability_l148_148102


namespace ratio_of_fractions_l148_148935

-- Given conditions
variables {x y : ℚ}
variables (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0)

-- Assertion to be proved
theorem ratio_of_fractions (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 18 / 25 :=
sorry

end ratio_of_fractions_l148_148935


namespace not_possible_values_l148_148872

theorem not_possible_values (t h d : ℕ) (ht : 3 * t - 6 * h = 2001) (hd : t - h = d) (hh : 6 * h > 0) :
  ∃ n, n = 667 ∧ ∀ d : ℕ, d ≤ 667 → ¬ (t = h + d ∧ 3 * (h + d) - 6 * h = 2001) :=
by
  sorry

end not_possible_values_l148_148872


namespace prime_power_minus_l148_148927

theorem prime_power_minus (p : ℕ) (hp : Nat.Prime p) (hps : Nat.Prime (p + 3)) : p ^ 11 - 52 = 1996 := by
  -- this is where the proof would go
  sorry

end prime_power_minus_l148_148927


namespace market_survey_l148_148841

theorem market_survey (X Y Z : ℕ) (h1 : X / Y = 3)
  (h2 : X / Z = 2 / 3) (h3 : X = 60) : X + Y + Z = 170 :=
by
  sorry

end market_survey_l148_148841


namespace gross_revenue_is_47_l148_148498

def total_net_profit : ℤ := 44
def babysitting_profit : ℤ := 31
def lemonade_stand_expense : ℤ := 34

def gross_revenue_from_lemonade_stand (P_t P_b E : ℤ) : ℤ :=
  P_t - P_b + E

theorem gross_revenue_is_47 :
  gross_revenue_from_lemonade_stand total_net_profit babysitting_profit lemonade_stand_expense = 47 :=
by
  sorry

end gross_revenue_is_47_l148_148498


namespace find_k_l148_148400

noncomputable def k_val : ℝ := 19.2

theorem find_k (k : ℝ) :
  (4 + ∑' n : ℕ, (4 + n * k) / (5^(n + 1))) = 10 ↔ k = k_val :=
  sorry

end find_k_l148_148400


namespace no_sphinx_tiling_l148_148425

def equilateral_triangle_tiling_problem (side_length : ℕ) (pointing_up : ℕ) (pointing_down : ℕ) : Prop :=
  let total_triangles := side_length * side_length
  pointing_up + pointing_down = total_triangles ∧ 
  total_triangles = 36 ∧
  pointing_down = 1 + 2 + 3 + 4 + 5 ∧
  pointing_up = 1 + 2 + 3 + 4 + 5 + 6 ∧
  (pointing_up % 2 = 1) ∧
  (pointing_down % 2 = 1) ∧
  (2 * pointing_up + 4 * pointing_down ≠ total_triangles ∧ 4 * pointing_up + 2 * pointing_down ≠ total_triangles)

theorem no_sphinx_tiling : ¬equilateral_triangle_tiling_problem 6 21 15 :=
by
  sorry

end no_sphinx_tiling_l148_148425


namespace millicent_fraction_books_l148_148896

variable (M H : ℝ)
variable (F : ℝ)

-- Conditions
def harold_has_half_books (M H : ℝ) : Prop := H = (1 / 2) * M
def harold_brings_one_third_books (M H : ℝ) : Prop := (1 / 3) * H = (1 / 6) * M
def new_library_capacity (M F : ℝ) : Prop := (1 / 6) * M + F * M = (5 / 6) * M

-- Target Proof Statement
theorem millicent_fraction_books (M H F : ℝ) 
    (h1 : harold_has_half_books M H) 
    (h2 : harold_brings_one_third_books M H) 
    (h3 : new_library_capacity M F) : 
    F = 2 / 3 :=
sorry

end millicent_fraction_books_l148_148896


namespace find_percentage_decrease_in_fourth_month_l148_148262

theorem find_percentage_decrease_in_fourth_month
  (P0 : ℝ) (P1 : ℝ) (P2 : ℝ) (P3 : ℝ) (x : ℝ) :
  (P0 = 100) →
  (P1 = P0 + 0.30 * P0) →
  (P2 = P1 - 0.15 * P1) →
  (P3 = P2 + 0.10 * P2) →
  (P0 = P3 - x / 100 * P3) →
  x = 18 :=
by
  sorry

end find_percentage_decrease_in_fourth_month_l148_148262


namespace isosceles_triangle_perimeter_l148_148548

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_perimeter 
  (a b c : ℝ) 
  (h_iso : is_isosceles a b c) 
  (h1 : a = 2 ∨ a = 4) 
  (h2 : b = 2 ∨ b = 4) 
  (h3 : c = 2 ∨ c = 4) :
  a + b + c = 10 :=
  sorry

end isosceles_triangle_perimeter_l148_148548


namespace hotel_charge_per_hour_morning_l148_148827

noncomputable def charge_per_hour_morning := 2 -- The correct answer

theorem hotel_charge_per_hour_morning
  (cost_night : ℝ)
  (initial_money : ℝ)
  (hours_night : ℝ)
  (hours_morning : ℝ)
  (remaining_money : ℝ)
  (total_cost : ℝ)
  (M : ℝ)
  (H1 : cost_night = 1.50)
  (H2 : initial_money = 80)
  (H3 : hours_night = 6)
  (H4 : hours_morning = 4)
  (H5 : remaining_money = 63)
  (H6 : total_cost = initial_money - remaining_money)
  (H7 : total_cost = hours_night * cost_night + hours_morning * M) :
  M = charge_per_hour_morning :=
by
  sorry

end hotel_charge_per_hour_morning_l148_148827


namespace simplify_and_evaluate_expression_l148_148241

theorem simplify_and_evaluate_expression (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) : 
  2 * x * y - (1 / 2) * (4 * x * y - 8 * x^2 * y^2) + 2 * (3 * x * y - 5 * x^2 * y^2) = -36 := by
  sorry

end simplify_and_evaluate_expression_l148_148241


namespace product_of_two_equal_numbers_l148_148606

theorem product_of_two_equal_numbers :
  ∃ (x : ℕ), (5 * 20 = 12 + 22 + 16 + 2 * x) ∧ (x * x = 625) :=
by
  sorry

end product_of_two_equal_numbers_l148_148606


namespace fraction_exponentiation_multiplication_l148_148252

theorem fraction_exponentiation_multiplication :
  (1 / 3) ^ 4 * (1 / 8) = 1 / 648 :=
by
  sorry

end fraction_exponentiation_multiplication_l148_148252


namespace equation_of_line_l_equations_of_line_m_l148_148911

-- Define the point P and condition for line l
def P := (2, (7 : ℚ)/4)
def l_slope : ℚ := 3 / 4

-- Define the given equation form and conditions for line l
def condition_l (x y : ℚ) : Prop := y - (7 / 4) = (3 / 4) * (x - 2)
def equation_l (x y : ℚ) : Prop := 3 * x - 4 * y = 5

theorem equation_of_line_l :
  ∀ x y : ℚ, condition_l x y → equation_l x y :=
sorry

-- Define the distance condition for line m
def equation_m (x y n : ℚ) : Prop := 3 * x - 4 * y + n = 0
def distance_condition_m (n : ℚ) : Prop := 
  |(-1 + n : ℚ)| / 5 = 3

theorem equations_of_line_m :
  ∃ n : ℚ, distance_condition_m n ∧ (equation_m 2 (7/4) n) ∨ 
            equation_m 2 (7/4) (-14) :=
sorry

end equation_of_line_l_equations_of_line_m_l148_148911


namespace middle_number_is_five_l148_148099

theorem middle_number_is_five
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 20)
  (h_sorted : a < b ∧ b < c)
  (h_bella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → x = a → y = b ∧ z = c)
  (h_della : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → y = b → x = a ∧ z = c)
  (h_nella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → z = c → x = a ∧ y = b) :
  b = 5 := sorry

end middle_number_is_five_l148_148099


namespace wall_building_time_l148_148341

theorem wall_building_time
  (m1 m2 : ℕ) 
  (d1 d2 : ℝ)
  (h1 : m1 = 20)
  (h2 : d1 = 3.0)
  (h3 : m2 = 30)
  (h4 : ∃ k, m1 * d1 = k ∧ m2 * d2 = k) :
  d2 = 2.0 :=
by
  sorry

end wall_building_time_l148_148341


namespace base_k_representation_l148_148755

theorem base_k_representation (k : ℕ) (hk : k > 0) (hk_exp : 7 / 51 = (2 * k + 3 : ℚ) / (k ^ 2 - 1 : ℚ)) : k = 16 :=
by {
  sorry
}

end base_k_representation_l148_148755


namespace remainder_when_divided_by_8_l148_148206

theorem remainder_when_divided_by_8 :
  (481207 % 8) = 7 :=
by
  sorry

end remainder_when_divided_by_8_l148_148206


namespace calculate_expression_l148_148175

theorem calculate_expression : 
  -3^2 + Real.sqrt ((-2)^4) - (-27)^(1/3 : ℝ) = -2 := 
by
  sorry

end calculate_expression_l148_148175


namespace amount_of_pizza_needed_l148_148118

theorem amount_of_pizza_needed :
  (1 / 2 + 1 / 3 + 1 / 6) = 1 := by
  sorry

end amount_of_pizza_needed_l148_148118


namespace proof_problem_l148_148306

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem proof_problem (h_even : even_function f)
                      (h_period : ∀ x, f (x + 2) = -f x)
                      (h_incr : increasing_on f (-2) 0) :
                      periodic_function f 4 ∧ symmetric_about f 2 :=
by { sorry }

end proof_problem_l148_148306


namespace ed_lost_seven_marbles_l148_148214

theorem ed_lost_seven_marbles (D L : ℕ) (h1 : ∃ (Ed_init Tim_init : ℕ), Ed_init = D + 19 ∧ Tim_init = D - 10)
(h2 : ∃ (Ed_final Tim_final : ℕ), Ed_final = D + 19 - L - 4 ∧ Tim_final = D - 10 + 4 + 3)
(h3 : ∀ (Ed_final : ℕ), Ed_final = D + 8)
(h4 : ∀ (Tim_final : ℕ), Tim_final = D):
  L = 7 :=
by
  sorry

end ed_lost_seven_marbles_l148_148214


namespace solve_quartic_eq_l148_148650

theorem solve_quartic_eq {x : ℝ} : (x - 4)^4 + (x - 6)^4 = 16 → (x = 4 ∨ x = 6) :=
by
  sorry

end solve_quartic_eq_l148_148650


namespace P_works_alone_l148_148011

theorem P_works_alone (P : ℝ) (hP : 2 * (1 / P + 1 / 15) + 0.6 * (1 / P) = 1) : P = 3 :=
by sorry

end P_works_alone_l148_148011


namespace discount_on_item_l148_148456

noncomputable def discount_percentage : ℝ := 20
variable (total_cart_value original_price final_amount : ℝ)
variable (coupon_discount : ℝ)

axiom cart_value : total_cart_value = 54
axiom item_price : original_price = 20
axiom coupon : coupon_discount = 0.10
axiom final_price : final_amount = 45

theorem discount_on_item :
  ∃ x : ℝ, (total_cart_value - (x / 100) * original_price) * (1 - coupon_discount) = final_amount ∧ x = discount_percentage :=
by
  have eq1 := cart_value
  have eq2 := item_price
  have eq3 := coupon
  have eq4 := final_price
  sorry

end discount_on_item_l148_148456


namespace angle_W_in_quadrilateral_l148_148211

theorem angle_W_in_quadrilateral 
  (W X Y Z : ℝ) 
  (h₀ : W + X + Y + Z = 360) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) : 
  W = 206 :=
by
  sorry

end angle_W_in_quadrilateral_l148_148211


namespace initial_number_of_apples_l148_148776

-- Definitions based on the conditions
def number_of_trees : ℕ := 3
def apples_picked_per_tree : ℕ := 8
def apples_left_on_trees : ℕ := 9

-- The theorem to prove
theorem initial_number_of_apples (t: ℕ := number_of_trees) (a: ℕ := apples_picked_per_tree) (l: ℕ := apples_left_on_trees) : t * a + l = 33 :=
by
  sorry

end initial_number_of_apples_l148_148776


namespace log_domain_inequality_l148_148151

theorem log_domain_inequality {a : ℝ} : 
  (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ a > 1 :=
sorry

end log_domain_inequality_l148_148151


namespace total_distance_covered_l148_148551

variable (h : ℝ) (initial_height : ℝ := h) (bounce_ratio : ℝ := 0.8)

theorem total_distance_covered :
  initial_height + 2 * initial_height * bounce_ratio / (1 - bounce_ratio) = 13 * h :=
by 
  -- Proof omitted for now
  sorry

end total_distance_covered_l148_148551


namespace area_difference_quarter_circles_l148_148593

theorem area_difference_quarter_circles :
  let r1 := 28
  let r2 := 14
  let pi := (22 / 7)
  let quarter_area_big := (1 / 4) * pi * r1^2
  let quarter_area_small := (1 / 4) * pi * r2^2
  let rectangle_area := r1 * r2
  (quarter_area_big - (quarter_area_small + rectangle_area)) = 70 := by
  -- Placeholder for the proof
  sorry

end area_difference_quarter_circles_l148_148593


namespace evaluate_at_10_l148_148311

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem evaluate_at_10 : f 10 = 756 := by
  -- the proof is omitted
  sorry

end evaluate_at_10_l148_148311


namespace symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l148_148663

-- Definitions of sequences of events and symmetric difference
variable (A : ℕ → Set α) (B : ℕ → Set α)

-- Definition of symmetric difference
def symm_diff (S T : Set α) : Set α := (S \ T) ∪ (T \ S)

-- Theorems to be proven
theorem symm_diff_complement (A1 B1 : Set α) :
  symm_diff A1 B1 = symm_diff (Set.compl A1) (Set.compl B1) := sorry

theorem symm_diff_union_subset :
  symm_diff (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

theorem symm_diff_inter_subset :
  symm_diff (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

end symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l148_148663


namespace find_prices_and_max_basketballs_l148_148392

def unit_price_condition (x : ℕ) (y : ℕ) : Prop :=
  y = 2*x - 30

def cost_ratio_condition (x : ℕ) (y : ℕ) : Prop :=
  3 * x = 2 * y - 60

def total_cost_condition (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ) : Prop :=
  total_cost ≤ 15500 ∧ num_basketballs + num_soccerballs = 200

theorem find_prices_and_max_basketballs
  (x y : ℕ) (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ)
  (h1 : unit_price_condition x y)
  (h2 : cost_ratio_condition x y)
  (h3 : total_cost_condition total_cost num_basketballs num_soccerballs)
  (h4 : total_cost = 90 * num_basketballs + 60 * num_soccerballs)
  : x = 60 ∧ y = 90 ∧ num_basketballs ≤ 116 :=
sorry

end find_prices_and_max_basketballs_l148_148392


namespace matrix_calculation_l148_148797

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l148_148797


namespace all_perfect_squares_l148_148397

theorem all_perfect_squares (a b c : ℕ) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) 
  (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 2 * (a * b + b * c + c * a)) : 
  ∃ (k l m : ℕ), a = k ^ 2 ∧ b = l ^ 2 ∧ c = m ^ 2 :=
sorry

end all_perfect_squares_l148_148397


namespace proof_problem_l148_148019

/-- Definition of the problem -/
def problem_statement : Prop :=
  ∃(a b c : ℝ) (A B C : ℝ) (D : ℝ),
    -- Conditions:
    ((b ^ 2 = a * c) ∧
     (2 * Real.cos (A - C) - 2 * Real.cos B = 1) ∧
     (D = 5) ∧
     -- Questions:
     (B = Real.pi / 3) ∧
     (∀ (AC CD : ℝ), (a = b ∧ b = c) → -- Equilateral triangle
       (AC * CD = (1/2) * (5 * AC - AC ^ 2) ∧
       (0 < AC * CD ∧ AC * CD ≤ 25/8))))

-- Lean 4 statement
theorem proof_problem : problem_statement := sorry

end proof_problem_l148_148019


namespace order_of_products_l148_148345

theorem order_of_products (x a b : ℝ) (h1 : x < a) (h2 : a < b) (h3 : b < 0) : b * x > a * x ∧ a * x > a ^ 2 :=
by
  sorry

end order_of_products_l148_148345


namespace length_of_faster_train_is_correct_l148_148849

def speed_faster_train := 54 -- kmph
def speed_slower_train := 36 -- kmph
def crossing_time := 27 -- seconds

def kmph_to_mps (s : ℕ) : ℕ :=
  s * 1000 / 3600

def relative_speed_faster_train := kmph_to_mps (speed_faster_train - speed_slower_train)

def length_faster_train := relative_speed_faster_train * crossing_time

theorem length_of_faster_train_is_correct : length_faster_train = 135 := 
  by
  sorry

end length_of_faster_train_is_correct_l148_148849


namespace angle_ratio_l148_148112

theorem angle_ratio (x y α β : ℝ)
  (h1 : y = x + β)
  (h2 : 2 * y = 2 * x + α) :
  α / β = 2 :=
by
  sorry

end angle_ratio_l148_148112


namespace find_abc_l148_148278

theorem find_abc :
  ∃ a b c : ℝ, (∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - a) * (x - b) / (x - c) ≤ 0)) ∧ a < b ∧ a + 2 * b + 3 * c = 74 :=
by
  sorry

end find_abc_l148_148278


namespace quadratic_nonneg_iff_m_in_range_l148_148716

theorem quadratic_nonneg_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 2 * m + 5 ≥ 0) ↔ (-2 : ℝ) ≤ m ∧ m ≤ 10 :=
by sorry

end quadratic_nonneg_iff_m_in_range_l148_148716


namespace sum_center_radius_eq_neg2_l148_148906

theorem sum_center_radius_eq_neg2 (c d s : ℝ) (h_eq : ∀ x y : ℝ, x^2 + 14 * x + y^2 - 8 * y = -64 ↔ (x + c)^2 + (y + d)^2 = s^2) :
  c + d + s = -2 :=
sorry

end sum_center_radius_eq_neg2_l148_148906


namespace average_mark_second_class_l148_148272

theorem average_mark_second_class
  (avg_mark_class1 : ℝ)
  (num_students_class1 : ℕ)
  (num_students_class2 : ℕ)
  (combined_avg_mark : ℝ) 
  (total_students : ℕ)
  (total_marks_combined : ℝ) :
  avg_mark_class1 * num_students_class1 + x * num_students_class2 = total_marks_combined →
  num_students_class1 + num_students_class2 = total_students →
  combined_avg_mark * total_students = total_marks_combined →
  avg_mark_class1 = 40 →
  num_students_class1 = 30 →
  num_students_class2 = 50 →
  combined_avg_mark = 58.75 →
  total_students = 80 →
  total_marks_combined = 4700 →
  x = 70 :=
by
  intros
  sorry

end average_mark_second_class_l148_148272


namespace number_of_strings_is_multiple_of_3_l148_148323

theorem number_of_strings_is_multiple_of_3 (N : ℕ) :
  (∀ (avg_total avg_one_third avg_two_third : ℚ), 
    avg_total = 80 ∧ avg_one_third = 70 ∧ avg_two_third = 85 →
    (∃ k : ℕ, N = 3 * k)) :=
by
  intros avg_total avg_one_third avg_two_third h
  sorry

end number_of_strings_is_multiple_of_3_l148_148323


namespace electric_guitar_count_l148_148004

theorem electric_guitar_count (E A : ℤ) (h1 : E + A = 9) (h2 : 479 * E + 339 * A = 3611) (hE_nonneg : E ≥ 0) (hA_nonneg : A ≥ 0) : E = 4 :=
by
  sorry

end electric_guitar_count_l148_148004


namespace largest_possible_a_l148_148097

theorem largest_possible_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) (hp : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a ≤ 8924 :=
sorry

end largest_possible_a_l148_148097


namespace intersection_x_val_l148_148318

theorem intersection_x_val (x y : ℝ) (h1 : y = 3 * x - 24) (h2 : 5 * x + 2 * y = 102) : x = 150 / 11 :=
by
  sorry

end intersection_x_val_l148_148318


namespace octal_addition_correct_l148_148687

def octal_to_decimal (n : ℕ) : ℕ := 
  /- function to convert an octal number to decimal goes here -/
  sorry

def decimal_to_octal (n : ℕ) : ℕ :=
  /- function to convert a decimal number to octal goes here -/
  sorry

theorem octal_addition_correct :
  let a := 236 
  let b := 521
  let c := 74
  let sum_decimal := octal_to_decimal a + octal_to_decimal b + octal_to_decimal c
  decimal_to_octal sum_decimal = 1063 :=
by
  sorry

end octal_addition_correct_l148_148687


namespace geometric_sequence_first_term_l148_148992

theorem geometric_sequence_first_term (a : ℕ) (r : ℕ)
    (h1 : a * r^2 = 27) 
    (h2 : a * r^3 = 81) : 
    a = 3 :=
by
  sorry

end geometric_sequence_first_term_l148_148992


namespace sqrt_mixed_number_simplification_l148_148616

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_mixed_number_simplification_l148_148616


namespace sin_double_angle_neg_l148_148649

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l148_148649


namespace sequences_of_length_15_l148_148224

def odd_runs_of_A_even_runs_of_B (n : ℕ) : ℕ :=
  (if n = 1 then 1 else 0) + (if n = 2 then 1 else 0)

theorem sequences_of_length_15 : 
  odd_runs_of_A_even_runs_of_B 15 = 47260 :=
  sorry

end sequences_of_length_15_l148_148224


namespace find_a_l148_148043

theorem find_a (a : ℝ) (h : ∃ b : ℝ, (4:ℝ)*x^2 - (12:ℝ)*x + a = (2*x + b)^2) : a = 9 :=
sorry

end find_a_l148_148043


namespace max_abs_f_lower_bound_l148_148968

theorem max_abs_f_lower_bound (a b M : ℝ) (hM : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → abs (x^2 + a*x + b) ≤ M) : 
  M ≥ 1/2 :=
sorry

end max_abs_f_lower_bound_l148_148968


namespace problem1_problem2_l148_148179

namespace TriangleProofs

-- Problem 1: Prove that A + B = π / 2
theorem problem1 (a b c : ℝ) (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (a, Real.cos B))
  (h2 : n = (b, Real.cos A))
  (h_parallel : m.1 * n.2 = m.2 * n.1)
  (h_neq : m ≠ n)
  : A + B = Real.pi / 2 :=
sorry

-- Problem 2: Determine the range of x
theorem problem2 (A B : ℝ) (x : ℝ) 
  (h : A + B = Real.pi / 2) 
  (hx : x * Real.sin A * Real.sin B = Real.sin A + Real.sin B) 
  : 2 * Real.sqrt 2 ≤ x :=
sorry

end TriangleProofs

end problem1_problem2_l148_148179


namespace parallelogram_base_length_l148_148055

theorem parallelogram_base_length (Area Height : ℝ) (h1 : Area = 216) (h2 : Height = 18) : 
  Area / Height = 12 := 
by 
  sorry

end parallelogram_base_length_l148_148055


namespace arc_length_parametric_l148_148528

open Real Interval

noncomputable def arc_length (f_x f_y : ℝ → ℝ) (t1 t2 : ℝ) :=
  ∫ t in Set.Icc t1 t2, sqrt ((deriv f_x t)^2 + (deriv f_y t)^2)

theorem arc_length_parametric :
  arc_length
    (λ t => 2.5 * (t - sin t))
    (λ t => 2.5 * (1 - cos t))
    (π / 2) π = 5 * sqrt 2 :=
by
  sorry

end arc_length_parametric_l148_148528


namespace rotate_parabola_180_l148_148065

theorem rotate_parabola_180 (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 2) → 
  (∃ x' y', x' = -x ∧ y' = -y ∧ y' = -2 * (x' + 1)^2 - 2) := 
sorry

end rotate_parabola_180_l148_148065


namespace total_amount_saved_l148_148210

def priceX : ℝ := 575
def surcharge_rateX : ℝ := 0.04
def installation_chargeX : ℝ := 82.50
def total_chargeX : ℝ := priceX + surcharge_rateX * priceX + installation_chargeX

def priceY : ℝ := 530
def surcharge_rateY : ℝ := 0.03
def installation_chargeY : ℝ := 93.00
def total_chargeY : ℝ := priceY + surcharge_rateY * priceY + installation_chargeY

def savings : ℝ := total_chargeX - total_chargeY

theorem total_amount_saved : savings = 41.60 :=
by
  sorry

end total_amount_saved_l148_148210


namespace g_of_g_of_g_of_g_of_3_l148_148171

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_of_g_of_g_of_g_of_3 : g (g (g (g 3))) = 3 :=
by sorry

end g_of_g_of_g_of_g_of_3_l148_148171


namespace tan_alpha_equiv_l148_148700

theorem tan_alpha_equiv (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end tan_alpha_equiv_l148_148700


namespace sin_double_angle_cos_condition_l148_148909

theorem sin_double_angle_cos_condition (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) :
  Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_cos_condition_l148_148909


namespace find_a_l148_148457

theorem find_a (a x : ℝ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 :=
by
  sorry

end find_a_l148_148457


namespace not_sufficient_nor_necessary_geometric_seq_l148_148181

theorem not_sufficient_nor_necessary_geometric_seq {a : ℕ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q) :
    (a 1 < a 3) ↔ (¬(a 2 < a 4) ∨ ¬(a 4 < a 2)) :=
by
  sorry

end not_sufficient_nor_necessary_geometric_seq_l148_148181


namespace difference_in_dimes_l148_148640

variables (q : ℝ)

def samantha_quarters : ℝ := 3 * q + 2
def bob_quarters : ℝ := 2 * q + 8
def quarter_to_dimes : ℝ := 2.5

theorem difference_in_dimes :
  quarter_to_dimes * (samantha_quarters q - bob_quarters q) = 2.5 * q - 15 :=
by sorry

end difference_in_dimes_l148_148640


namespace expected_intersections_100gon_l148_148765

noncomputable def expected_intersections : ℝ :=
  let n := 100
  let total_pairs := (n * (n - 3) / 2)
  total_pairs * (1/3)

theorem expected_intersections_100gon :
  expected_intersections = 4850 / 3 :=
by
  sorry

end expected_intersections_100gon_l148_148765


namespace polynomials_equal_l148_148834

noncomputable def P : ℝ → ℝ := sorry -- assume P is a nonconstant polynomial
noncomputable def Q : ℝ → ℝ := sorry -- assume Q is a nonconstant polynomial

axiom floor_eq_for_all_y (y : ℝ) : ⌊P y⌋ = ⌊Q y⌋

theorem polynomials_equal (x : ℝ) : P x = Q x :=
by
  sorry

end polynomials_equal_l148_148834


namespace first_problem_solution_set_second_problem_a_range_l148_148430

-- Define the function f(x) = |2x - a| + |x - 1|
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 1)

-- First problem: When a = 3, the solution set of the inequality f(x) ≥ 2
theorem first_problem_solution_set (x : ℝ) : (f x 3 ≥ 2) ↔ (x ≤ 2 / 3 ∨ x ≥ 2) :=
by sorry

-- Second problem: If f(x) ≥ 5 - x for ∀ x ∈ ℝ, find the range of the real number a
theorem second_problem_a_range (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ (6 ≤ a) :=
by sorry

end first_problem_solution_set_second_problem_a_range_l148_148430


namespace complex_number_real_imag_equal_l148_148192

theorem complex_number_real_imag_equal (a : ℝ) (h : (a + 6) = (3 - 2 * a)) : a = -1 :=
by
  sorry

end complex_number_real_imag_equal_l148_148192


namespace a2_add_a8_l148_148417

variable (a : ℕ → ℝ) -- a_n is an arithmetic sequence
variable (d : ℝ) -- common difference

-- Condition stating that a_n is an arithmetic sequence with common difference d
axiom arithmetic_sequence : ∀ n, a (n + 1) = a n + d

-- Given condition a_3 + a_4 + a_5 + a_6 + a_7 = 450
axiom given_condition : a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem a2_add_a8 : a 2 + a 8 = 180 :=
by
  sorry

end a2_add_a8_l148_148417


namespace total_balloons_l148_148951

theorem total_balloons
  (g b y r : ℕ)  -- Number of green, blue, yellow, and red balloons respectively
  (equal_groups : g = b ∧ b = y ∧ y = r)
  (anya_took : y / 2 = 84) :
  g + b + y + r = 672 := by
sorry

end total_balloons_l148_148951


namespace minimum_value_at_x_eq_3_l148_148001

theorem minimum_value_at_x_eq_3 (b : ℝ) : 
  ∃ m : ℝ, (∀ x : ℝ, 3 * x^2 - 18 * x + b ≥ m) ∧ (3 * 3^2 - 18 * 3 + b = m) :=
by
  sorry

end minimum_value_at_x_eq_3_l148_148001


namespace ceil_neg_sqrt_frac_l148_148335

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l148_148335


namespace evaluate_expression_l148_148169

theorem evaluate_expression : 
  (196 * (1 / 17 - 1 / 21) + 361 * (1 / 21 - 1 / 13) + 529 * (1 / 13 - 1 / 17)) /
    (14 * (1 / 17 - 1 / 21) + 19 * (1 / 21 - 1 / 13) + 23 * (1 / 13 - 1 / 17)) = 56 :=
by
  sorry

end evaluate_expression_l148_148169


namespace total_number_of_bottles_l148_148474

def water_bottles := 2 * 12
def orange_juice_bottles := (7 / 4) * 12
def apple_juice_bottles := water_bottles + 6
def total_bottles := water_bottles + orange_juice_bottles + apple_juice_bottles

theorem total_number_of_bottles :
  total_bottles = 75 :=
by
  sorry

end total_number_of_bottles_l148_148474


namespace solution_l148_148866

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, g (-y) = g y

def problem (f g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y) ∧
  (f 0 = 0) ∧
  (∃ x : ℝ, f x ≠ 0)

theorem solution (f g : ℝ → ℝ) (h : problem f g) : is_odd f ∧ is_even g :=
sorry

end solution_l148_148866


namespace find_y_l148_148407

theorem find_y (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 :=
by
  sorry

end find_y_l148_148407


namespace remainder_101_pow_47_mod_100_l148_148519

theorem remainder_101_pow_47_mod_100 : (101 ^ 47) % 100 = 1 := by 
  sorry

end remainder_101_pow_47_mod_100_l148_148519


namespace train_length_l148_148168

theorem train_length {L : ℝ} (h_equal_lengths : ∃ (L: ℝ), L = L) (h_cross_time : ∃ (t : ℝ), t = 60) (h_speed : ∃ (v : ℝ), v = 20) : L = 600 :=
by
  sorry

end train_length_l148_148168


namespace tickets_to_buy_l148_148053

theorem tickets_to_buy
  (ferris_wheel_cost : Float := 2.0)
  (roller_coaster_cost : Float := 7.0)
  (multiple_rides_discount : Float := 1.0)
  (newspaper_coupon : Float := 1.0) :
  (ferris_wheel_cost + roller_coaster_cost - multiple_rides_discount - newspaper_coupon = 7.0) :=
by
  sorry

end tickets_to_buy_l148_148053


namespace lawsuit_win_probability_l148_148944

theorem lawsuit_win_probability (P_L1 P_L2 P_W1 P_W2 : ℝ) (h1 : P_L2 = 0.5) 
  (h2 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20 * P_W1 * P_W2)
  (h3 : P_W1 + P_L1 = 1)
  (h4 : P_W2 + P_L2 = 1) : 
  P_W1 = 1 / 2.20 :=
by
  sorry

end lawsuit_win_probability_l148_148944


namespace final_price_correct_l148_148431

open BigOperators

-- Define the constants used in the problem
def original_price : ℝ := 500
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def state_tax : ℝ := 0.05

-- Define the calculation steps
def price_after_first_discount : ℝ := original_price * (1 - first_discount)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - second_discount)
def final_price : ℝ := price_after_second_discount * (1 + state_tax)

-- Prove that the final price is 354.375
theorem final_price_correct : final_price = 354.375 :=
by
  sorry

end final_price_correct_l148_148431


namespace no_solution_inequalities_l148_148940

theorem no_solution_inequalities (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  intro h
  sorry

end no_solution_inequalities_l148_148940


namespace number_of_grade2_students_l148_148127

theorem number_of_grade2_students (ratio1 ratio2 ratio3 : ℕ) (total_students : ℕ) (ratio_sum : ratio1 + ratio2 + ratio3 = 12)
  (total_sample_size : total_students = 240) : 
  total_students * ratio2 / (ratio1 + ratio2 + ratio3) = 80 :=
by
  have ratio1_val : ratio1 = 5 := sorry
  have ratio2_val : ratio2 = 4 := sorry
  have ratio3_val : ratio3 = 3 := sorry
  rw [ratio1_val, ratio2_val, ratio3_val] at ratio_sum
  rw [ratio1_val, ratio2_val, ratio3_val]
  exact sorry

end number_of_grade2_students_l148_148127


namespace smaller_square_area_percentage_l148_148624

noncomputable def percent_area_of_smaller_square (side_length_larger_square : ℝ) : ℝ :=
  let diagonal_larger_square := side_length_larger_square * Real.sqrt 2
  let radius_circle := diagonal_larger_square / 2
  let x := (2 + 4 * (side_length_larger_square / 2)) / ((side_length_larger_square / 2) * 2) -- Simplified quadratic solution
  let side_length_smaller_square := side_length_larger_square * x
  let area_smaller_square := side_length_smaller_square ^ 2
  let area_larger_square := side_length_larger_square ^ 2
  (area_smaller_square / area_larger_square) * 100

-- Statement to show that under given conditions, the area of the smaller square is 4% of the larger square's area
theorem smaller_square_area_percentage :
  percent_area_of_smaller_square 4 = 4 := 
sorry

end smaller_square_area_percentage_l148_148624


namespace area_acpq_eq_sum_areas_aekl_cdmn_l148_148144

variables (A B C D E P Q M N K L : Point)

def is_acute_angled_triangle (A B C : Point) : Prop := sorry
def is_altitude (A B C D : Point) : Prop := sorry
def is_square (A P Q C : Point) : Prop := sorry
def is_rectangle (A E K L : Point) : Prop := sorry
def is_rectangle' (C D M N : Point) : Prop := sorry
def length (P Q : Point) : Real := sorry
def area (P Q R S : Point) : Real := sorry

-- Conditions
axiom abc_acute : is_acute_angled_triangle A B C
axiom ad_altitude : is_altitude A B C D
axiom ce_altitude : is_altitude C A B E
axiom acpq_square : is_square A P Q C
axiom aekl_rectangle : is_rectangle A E K L
axiom cdmn_rectangle : is_rectangle' C D M N
axiom al_eq_ab : length A L = length A B
axiom cn_eq_cb : length C N = length C B

-- Question proof statement
theorem area_acpq_eq_sum_areas_aekl_cdmn :
  area A C P Q = area A E K L + area C D M N :=
sorry

end area_acpq_eq_sum_areas_aekl_cdmn_l148_148144


namespace price_reduction_equation_l148_148585

variable (x : ℝ)

theorem price_reduction_equation 
    (original_price : ℝ)
    (final_price : ℝ)
    (two_reductions : original_price * (1 - x) ^ 2 = final_price) :
    100 * (1 - x) ^ 2 = 81 :=
by
  sorry

end price_reduction_equation_l148_148585


namespace gcf_180_270_450_l148_148040

theorem gcf_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  sorry

end gcf_180_270_450_l148_148040


namespace cattle_transport_problem_l148_148524

noncomputable def truck_capacity 
    (total_cattle : ℕ)
    (distance_one_way : ℕ)
    (speed : ℕ)
    (total_time : ℕ) : ℕ :=
  total_cattle / (total_time / ((distance_one_way * 2) / speed))

theorem cattle_transport_problem :
  truck_capacity 400 60 60 40 = 20 := by
  -- The theorem statement follows the structure from the conditions and question
  sorry

end cattle_transport_problem_l148_148524


namespace evaluate_fraction_l148_148588

variable (a b x : ℝ)
variable (h1 : a ≠ b)
variable (h2 : b ≠ 0)
variable (h3 : x = a / b)

theorem evaluate_fraction :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_fraction_l148_148588


namespace ball_distribution_ways_l148_148547

theorem ball_distribution_ways :
  ∃ (ways : ℕ), ways = 10 ∧
    ∀ (balls boxes : ℕ), 
    balls = 6 ∧ boxes = 4 ∧ 
    (∀ (b : ℕ), b < boxes → b > 0) →
    ways = 10 :=
sorry

end ball_distribution_ways_l148_148547


namespace petya_vasya_meet_at_lamp_64_l148_148485

-- Definitions of positions of Petya and Vasya
def Petya_position (x : ℕ) : ℕ := x - 21 -- Petya starts from the 1st lamp and is at the 22nd lamp
def Vasya_position (x : ℕ) : ℕ := 88 - x -- Vasya starts from the 100th lamp and is at the 88th lamp

-- Condition that both lanes add up to 64
theorem petya_vasya_meet_at_lamp_64 : ∀ x y : ℕ, 
    Petya_position x = Vasya_position y -> x = 64 :=
by
  intro x y
  rw [Petya_position, Vasya_position]
  sorry

end petya_vasya_meet_at_lamp_64_l148_148485


namespace division_value_l148_148172

theorem division_value (x : ℝ) (h : 1376 / x - 160 = 12) : x = 8 := 
by sorry

end division_value_l148_148172


namespace factor_x4_plus_64_monic_real_l148_148258

theorem factor_x4_plus_64_monic_real :
  ∀ x : ℝ, x^4 + 64 = (x^2 + 4 * x + 8) * (x^2 - 4 * x + 8) := 
by
  intros
  sorry

end factor_x4_plus_64_monic_real_l148_148258


namespace fraction_exponent_multiplication_l148_148562

theorem fraction_exponent_multiplication :
  ( (8/9 : ℚ)^2 * (1/3 : ℚ)^2 = (64/729 : ℚ) ) :=
by
  -- here we would write out the detailed proof
  sorry

end fraction_exponent_multiplication_l148_148562


namespace probability_of_drawing_white_ball_l148_148028

theorem probability_of_drawing_white_ball 
  (total_balls : ℕ) (white_balls : ℕ) 
  (h_total : total_balls = 9) (h_white : white_balls = 4) : 
  (white_balls : ℚ) / total_balls = 4 / 9 := 
by 
  sorry

end probability_of_drawing_white_ball_l148_148028


namespace find_a_value_l148_148828

theorem find_a_value 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 2 * y - 1 = 0)
  (M N : ℝ × ℝ)
  (tangent_condition : (N.snd - M.snd) / (N.fst - M.fst) + (M.fst + N.fst - 2) / (M.snd + N.snd) = 0) : 
  a = 3 ∨ a = -2 := 
sorry

end find_a_value_l148_148828


namespace smallest_part_is_correct_l148_148343

-- Conditions
def total_value : ℕ := 360
def proportion1 : ℕ := 5
def proportion2 : ℕ := 7
def proportion3 : ℕ := 4
def proportion4 : ℕ := 8
def total_parts := proportion1 + proportion2 + proportion3 + proportion4
def value_per_part := total_value / total_parts
def smallest_proportion : ℕ := proportion3

-- Theorem to prove
theorem smallest_part_is_correct : value_per_part * smallest_proportion = 60 := by
  dsimp [total_value, total_parts, value_per_part, smallest_proportion]
  norm_num
  sorry

end smallest_part_is_correct_l148_148343


namespace no_two_digit_numbers_satisfy_condition_l148_148764

theorem no_two_digit_numbers_satisfy_condition :
  ¬ ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by
  sorry

end no_two_digit_numbers_satisfy_condition_l148_148764


namespace height_inradius_ratio_is_7_l148_148033

-- Definitions of geometric entities and given conditions.
variable (h r : ℝ)
variable (cos_theta : ℝ)
variable (cos_theta_eq : cos_theta = 1 / 6)

-- Theorem statement: Ratio of height to inradius is 7 given the cosine condition.
theorem height_inradius_ratio_is_7
  (h r : ℝ)
  (cos_theta : ℝ)
  (cos_theta_eq : cos_theta = 1 / 6)
  (prism_def : true) -- Added to mark the geometric nature properly
: h / r = 7 :=
sorry  -- Placeholder for the actual proof.

end height_inradius_ratio_is_7_l148_148033


namespace problem_evaluation_l148_148161

theorem problem_evaluation (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (ab + bc + cd + da + ac + bd)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (cd)⁻¹ + (da)⁻¹ + (ac)⁻¹ + (bd)⁻¹) = 
  (1 / (a * b * c * d)) * (1 / (a * b * c * d)) :=
by
  sorry

end problem_evaluation_l148_148161


namespace find_x_l148_148104

-- Definition of the problem conditions
def angle_ABC : ℝ := 85
def angle_BAC : ℝ := 55
def sum_angles_triangle (a b c : ℝ) : Prop := a + b + c = 180
def corresponding_angle (a b : ℝ) : Prop := a = b
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90

-- The theorem to prove
theorem find_x :
  ∀ (x BCA : ℝ), sum_angles_triangle angle_ABC angle_BAC BCA ∧ corresponding_angle BCA 40 ∧ right_triangle_sum BCA x → x = 50 :=
by
  intros x BCA h
  sorry

end find_x_l148_148104


namespace part1_part2_l148_148003

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Part 1: Prove the range of k such that f(x) < k * x for all x
theorem part1 (k : ℝ) : (∀ x : ℝ, x > 0 → f x < k * x) ↔ k > 1 / (2 * Real.exp 1) :=
by sorry

-- Part 2: Define the function g(x) = f(x) - k * x and prove the range of k for which g(x) has two zeros in the interval [1/e, e^2]
noncomputable def g (x k : ℝ) : ℝ := f x - k * x

theorem part2 (k : ℝ) : (∃ x1 x2 : ℝ, 1 / Real.exp 1 ≤ x1 ∧ x1 ≤ Real.exp 2 ∧
                                 1 / Real.exp 1 ≤ x2 ∧ x2 ≤ Real.exp 2 ∧
                                 g x1 k = 0 ∧ g x2 k = 0 ∧ x1 ≠ x2)
                               ↔ 2 / (Real.exp 4) ≤ k ∧ k < 1 / (2 * Real.exp 1) :=
by sorry

end part1_part2_l148_148003


namespace number_of_girls_in_class_l148_148553

theorem number_of_girls_in_class (B G : ℕ) (h1 : G = 4 * B / 10) (h2 : B + G = 35) : G = 10 :=
by
  sorry

end number_of_girls_in_class_l148_148553


namespace find_center_of_circle_l148_148855

noncomputable def center_of_circle (θ ρ : ℝ) : Prop :=
  ρ = (1 : ℝ) ∧ θ = (-Real.pi / (3 : ℝ))

theorem find_center_of_circle (θ ρ : ℝ) (h : ρ = Real.cos θ - Real.sqrt 3 * Real.sin θ) :
  center_of_circle θ ρ := by
  sorry

end find_center_of_circle_l148_148855


namespace find_added_number_l148_148891

theorem find_added_number 
  (initial_number : ℕ)
  (final_result : ℕ)
  (h : initial_number = 8)
  (h_result : 3 * (2 * initial_number + final_result) = 75) : 
  final_result = 9 := by
  sorry

end find_added_number_l148_148891


namespace quadratic_form_m_neg3_l148_148534

theorem quadratic_form_m_neg3
  (m : ℝ)
  (h_exp : m^2 - 7 = 2)
  (h_coef : m ≠ 3) :
  m = -3 := by
  sorry

end quadratic_form_m_neg3_l148_148534


namespace Patriots_won_30_games_l148_148021

def Tigers_won_more_games_than_Eagles (games_tigers games_eagles : ℕ) : Prop :=
games_tigers > games_eagles

def Patriots_won_more_than_Cubs_less_than_Mounties (games_patriots games_cubs games_mounties : ℕ) : Prop :=
games_cubs < games_patriots ∧ games_patriots < games_mounties

def Cubs_won_more_than_20_games (games_cubs : ℕ) : Prop :=
games_cubs > 20

theorem Patriots_won_30_games (games_tigers games_eagles games_patriots games_cubs games_mounties : ℕ)  :
  Tigers_won_more_games_than_Eagles games_tigers games_eagles →
  Patriots_won_more_than_Cubs_less_than_Mounties games_patriots games_cubs games_mounties →
  Cubs_won_more_than_20_games games_cubs →
  ∃ games_patriots, games_patriots = 30 := 
by
  sorry

end Patriots_won_30_games_l148_148021


namespace min_y_value_l148_148814

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16 * x + 50 * y + 64) : y ≥ 0 :=
sorry

end min_y_value_l148_148814


namespace find_n_positive_integer_l148_148931

theorem find_n_positive_integer:
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, 2^n + 12^n + 2011^n = k^2) ↔ n = 1 := 
by
  sorry

end find_n_positive_integer_l148_148931


namespace cost_per_meter_of_fencing_l148_148439

/-- A rectangular farm has area 1200 m², a short side of 30 m, and total job cost 1560 Rs.
    Prove that the cost of fencing per meter is 13 Rs. -/
theorem cost_per_meter_of_fencing
  (A : ℝ := 1200)
  (W : ℝ := 30)
  (job_cost : ℝ := 1560)
  (L : ℝ := A / W)
  (D : ℝ := Real.sqrt (L^2 + W^2))
  (total_length : ℝ := L + W + D) :
  job_cost / total_length = 13 := 
sorry

end cost_per_meter_of_fencing_l148_148439


namespace ajays_monthly_income_l148_148561

theorem ajays_monthly_income :
  ∀ (I : ℝ), 
  (0.50 * I) + (0.25 * I) + (0.15 * I) + 9000 = I → I = 90000 :=
by
  sorry

end ajays_monthly_income_l148_148561


namespace family_visit_cost_is_55_l148_148768

def num_children := 4
def num_parents := 2
def num_grandmother := 1
def num_people := num_children + num_parents + num_grandmother

def entrance_ticket_cost := 5
def attraction_ticket_cost_kid := 2
def attraction_ticket_cost_adult := 4

def entrance_total_cost := num_people * entrance_ticket_cost
def attraction_total_cost_kids := num_children * attraction_ticket_cost_kid
def adults := num_parents + num_grandmother
def attraction_total_cost_adults := adults * attraction_ticket_cost_adult

def total_cost := entrance_total_cost + attraction_total_cost_kids + attraction_total_cost_adults

theorem family_visit_cost_is_55 : total_cost = 55 := by
  sorry

end family_visit_cost_is_55_l148_148768


namespace problem_I5_1_l148_148864

theorem problem_I5_1 (a : ℝ) (h : a^2 - 8^2 = 12^2 + 9^2) : a = 17 := 
sorry

end problem_I5_1_l148_148864


namespace total_ice_cream_sales_l148_148217

theorem total_ice_cream_sales (tuesday_sales : ℕ) (h1 : tuesday_sales = 12000)
    (wednesday_sales : ℕ) (h2 : wednesday_sales = 2 * tuesday_sales) :
    tuesday_sales + wednesday_sales = 36000 := by
  -- This is the proof statement
  sorry

end total_ice_cream_sales_l148_148217


namespace value_of_expression_l148_148644

theorem value_of_expression : 30 - 5^2 = 5 := by
  sorry

end value_of_expression_l148_148644


namespace rounding_bounds_l148_148941

theorem rounding_bounds:
  ∃ (max min : ℕ), (∀ x : ℕ, (x >= 1305000) → (x < 1305000) -> false) ∧ 
  (max = 1304999) ∧ 
  (min = 1295000) :=
by
  -- Proof steps would go here
  sorry

end rounding_bounds_l148_148941


namespace angles_of_triangle_arith_seq_l148_148357

theorem angles_of_triangle_arith_seq (A B C a b c : ℝ) (h1 : A + B + C = 180) (h2 : A = B - (B - C)) (h3 : (1 / a + 1 / c) / 2 = 1 / b) : 
  A = 60 ∧ B = 60 ∧ C = 60 :=
sorry

end angles_of_triangle_arith_seq_l148_148357


namespace radius_of_circle_l148_148218

theorem radius_of_circle (A C : ℝ) (h1 : A = π * (r : ℝ)^2) (h2 : C = 2 * π * r) (h3 : A / C = 10) :
  r = 20 :=
by
  sorry

end radius_of_circle_l148_148218


namespace dan_blue_marbles_l148_148249

variable (m d : ℕ)
variable (h1 : m = 2 * d)
variable (h2 : m = 10)

theorem dan_blue_marbles : d = 5 :=
by
  sorry

end dan_blue_marbles_l148_148249


namespace marble_ratio_l148_148859

-- Definitions and assumptions from the conditions
def my_marbles : ℕ := 16
def total_marbles : ℕ := 63
def transfer_amount : ℕ := 2

-- After transferring marbles to my brother
def my_marbles_after_transfer := my_marbles - transfer_amount
def brother_marbles (B : ℕ) := B + transfer_amount

-- Friend's marbles
def friend_marbles (F : ℕ) := F = 3 * my_marbles_after_transfer

-- Prove the ratio of marbles after transfer
theorem marble_ratio (B F : ℕ) (hf : F = 3 * my_marbles_after_transfer) (h_total : my_marbles + B + F = total_marbles)
  (h_multiple : ∃ M : ℕ, my_marbles_after_transfer = M * brother_marbles B) :
  (my_marbles_after_transfer : ℚ) / (brother_marbles B : ℚ) = 2 / 1 :=
by
  sorry

end marble_ratio_l148_148859


namespace not_even_not_odd_neither_even_nor_odd_l148_148926

noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ + 1 / 2

theorem not_even (x : ℝ) : f (-x) ≠ f x := sorry
theorem not_odd (x : ℝ) : f (0) ≠ 0 ∨ f (-x) ≠ -f x := sorry

theorem neither_even_nor_odd : ∀ x : ℝ, f (-x) ≠ f x ∧ (f (0) ≠ 0 ∨ f (-x) ≠ -f x) :=
by
  intros x
  exact ⟨not_even x, not_odd x⟩

end not_even_not_odd_neither_even_nor_odd_l148_148926


namespace rowing_rate_in_still_water_l148_148754

theorem rowing_rate_in_still_water (R C : ℝ) 
  (h1 : (R + C) * 2 = 26)
  (h2 : (R - C) * 4 = 26) : 
  R = 26 / 3 :=
by
  sorry

end rowing_rate_in_still_water_l148_148754


namespace girls_more_than_boys_l148_148049

theorem girls_more_than_boys : ∃ (b g x : ℕ), b = 3 * x ∧ g = 4 * x ∧ b + g = 35 ∧ g - b = 5 :=
by  -- We just define the theorem, no need for a proof, added "by sorry"
  sorry

end girls_more_than_boys_l148_148049


namespace max_value_of_a_l148_148108

theorem max_value_of_a (a b c d : ℤ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 := by 
  sorry

end max_value_of_a_l148_148108


namespace option_B_correct_l148_148641

theorem option_B_correct (x m : ℕ) : (x^3)^m / (x^m)^2 = x^m := sorry

end option_B_correct_l148_148641


namespace bus_ride_difference_l148_148577

def oscars_bus_ride : ℝ := 0.75
def charlies_bus_ride : ℝ := 0.25

theorem bus_ride_difference :
  oscars_bus_ride - charlies_bus_ride = 0.50 :=
by
  sorry

end bus_ride_difference_l148_148577


namespace points_lost_calculation_l148_148365

variable (firstRound secondRound finalScore : ℕ)
variable (pointsLost : ℕ)

theorem points_lost_calculation 
  (h1 : firstRound = 40) 
  (h2 : secondRound = 50) 
  (h3 : finalScore = 86) 
  (h4 : pointsLost = firstRound + secondRound - finalScore) :
  pointsLost = 4 := 
sorry

end points_lost_calculation_l148_148365


namespace point_C_coordinates_line_MN_equation_area_triangle_ABC_l148_148386

-- Define the points A and B
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (7, 3)

-- Let C be an unknown point that we need to determine
variables (x y : ℝ)

-- Define the conditions given in the problem
axiom midpoint_M : (x + 5) / 2 = 0 ∧ (y + 3) / 2 = 0 -- Midpoint M lies on the y-axis
axiom midpoint_N : (x + 7) / 2 = 1 ∧ (y + 3) / 2 = 0 -- Midpoint N lies on the x-axis

-- The problem consists of proving three assertions
theorem point_C_coordinates :
  ∃ (x y : ℝ), (x, y) = (-5, -3) :=
by
  sorry

theorem line_MN_equation :
  ∃ (a b c : ℝ), a = 5 ∧ b = -2 ∧ c = -5 :=
by
  sorry

theorem area_triangle_ABC :
  ∃ (S : ℝ), S = 841 / 20 :=
by
  sorry

end point_C_coordinates_line_MN_equation_area_triangle_ABC_l148_148386


namespace part1_part2_l148_148060

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.log x)

theorem part1 : 
  (∀ x, 0 < x → x < 1 → (f x) < f (1)) ∧ 
  (∀ x, 1 < x → x < Real.exp 1 → (f x) < f (Real.exp 1)) :=
sorry

theorem part2 :
  ∃ k, k = 2 ∧ ∀ x, 0 < x → (f x) > (k / (Real.log x)) + 2 * Real.sqrt x :=
sorry

end part1_part2_l148_148060


namespace solve_equation_l148_148985

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end solve_equation_l148_148985


namespace spam_ratio_l148_148913

theorem spam_ratio (total_emails important_emails promotional_fraction promotional_emails spam_emails : ℕ) 
  (h1 : total_emails = 400) 
  (h2 : important_emails = 180) 
  (h3 : promotional_fraction = 2/5) 
  (h4 : total_emails - important_emails = spam_emails + promotional_emails) 
  (h5 : promotional_emails = promotional_fraction * (total_emails - important_emails)) 
  : spam_emails / total_emails = 33 / 100 := 
by {
  sorry
}

end spam_ratio_l148_148913


namespace total_distance_traveled_l148_148983

-- Definitions of conditions
def bess_throw_distance : ℕ := 20
def bess_throws : ℕ := 4
def holly_throw_distance : ℕ := 8
def holly_throws : ℕ := 5
def bess_effective_throw_distance : ℕ := 2 * bess_throw_distance

-- Theorem statement
theorem total_distance_traveled :
  (bess_throws * bess_effective_throw_distance + holly_throws * holly_throw_distance) = 200 := 
  by sorry

end total_distance_traveled_l148_148983


namespace niko_percentage_profit_l148_148954

theorem niko_percentage_profit
    (pairs_sold : ℕ)
    (cost_per_pair : ℕ)
    (profit_5_pairs : ℕ)
    (total_profit : ℕ)
    (num_pairs_remaining : ℕ)
    (cost_remaining_pairs : ℕ)
    (profit_remaining_pairs : ℕ)
    (percentage_profit : ℕ)
    (cost_5_pairs : ℕ):
    pairs_sold = 9 →
    cost_per_pair = 2 →
    profit_5_pairs = 1 →
    total_profit = 3 →
    num_pairs_remaining = 4 →
    cost_remaining_pairs = 8 →
    profit_remaining_pairs = 2 →
    percentage_profit = 25 →
    cost_5_pairs = 10 →
    (profit_remaining_pairs * 100 / cost_remaining_pairs) = percentage_profit :=
by
    intros
    sorry

end niko_percentage_profit_l148_148954


namespace maximum_value_of_omega_l148_148088

variable (A ω : ℝ)

theorem maximum_value_of_omega (hA : 0 < A) (hω_pos : 0 < ω)
  (h1 : ω * (-π / 2) ≥ -π / 2) 
  (h2 : ω * (2 * π / 3) ≤ π / 2) :
  ω = 3 / 4 :=
sorry

end maximum_value_of_omega_l148_148088


namespace number_of_valid_three_digit_numbers_l148_148296

def valid_three_digit_numbers : Nat :=
  -- Proving this will be the task: showing that there are precisely 24 such numbers
  24

theorem number_of_valid_three_digit_numbers : valid_three_digit_numbers = 24 :=
by
  -- Proof would go here.
  sorry

end number_of_valid_three_digit_numbers_l148_148296


namespace ratio_of_lengths_l148_148520

noncomputable def total_fence_length : ℝ := 640
noncomputable def short_side_length : ℝ := 80

theorem ratio_of_lengths (L S : ℝ) (h1 : 2 * L + 2 * S = total_fence_length) (h2 : S = short_side_length) :
  L / S = 3 :=
by {
  sorry
}

end ratio_of_lengths_l148_148520


namespace range_of_a_l148_148201

theorem range_of_a (a : ℝ) :
  (∃ (x : ℝ), (2 - 2^(-|x - 3|))^2 = 3 + a) ↔ -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l148_148201


namespace initial_peanuts_l148_148231

theorem initial_peanuts (x : ℕ) (h : x + 4 = 8) : x = 4 :=
sorry

end initial_peanuts_l148_148231


namespace symmetric_parabola_equation_l148_148667

theorem symmetric_parabola_equation (x y : ℝ) (h : y^2 = 2 * x) : (y^2 = -2 * (x + 2)) :=
by
  sorry

end symmetric_parabola_equation_l148_148667


namespace mary_needs_more_cups_l148_148247

theorem mary_needs_more_cups (total_cups required_cups added_cups : ℕ) (h1 : required_cups = 8) (h2 : added_cups = 2) : total_cups = 6 :=
by
  sorry

end mary_needs_more_cups_l148_148247


namespace find_length_d_l148_148780

theorem find_length_d :
  ∀ (A B C P: Type) (AB AC BC : ℝ) (d : ℝ),
    AB = 425 ∧ BC = 450 ∧ AC = 510 ∧
    (∃ (JG FI HE : ℝ), JG = FI ∧ FI = HE ∧ JG = d ∧ 
      (d / BC + d / AC + d / AB = 2)) 
    → d = 306 :=
by {
  sorry
}

end find_length_d_l148_148780


namespace paving_cost_l148_148685

theorem paving_cost (l w r : ℝ) (h_l : l = 5.5) (h_w : w = 4) (h_r : r = 700) :
  l * w * r = 15400 :=
by sorry

end paving_cost_l148_148685


namespace general_term_of_sequence_l148_148238

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else
  if n = 2 then 2 else
  sorry -- the recurrence relation will go here, but we'll skip its implementation

theorem general_term_of_sequence :
  ∀ n : ℕ, n ≥ 1 → a n = 3 - (2 / n) :=
by sorry

end general_term_of_sequence_l148_148238


namespace grasshopper_twenty_five_jumps_l148_148950

noncomputable def sum_natural (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem grasshopper_twenty_five_jumps :
  let total_distance := sum_natural 25
  total_distance % 2 = 1 -> 0 % 2 = 0 -> total_distance ≠ 0 :=
by
  intros total_distance_odd zero_even
  sorry

end grasshopper_twenty_five_jumps_l148_148950


namespace monochromatic_triangle_probability_correct_l148_148876

noncomputable def monochromatic_triangle_probability (p : ℝ) : ℝ :=
  1 - (3 * (p^2) * (1 - p) + 3 * ((1 - p)^2) * p)^20

theorem monochromatic_triangle_probability_correct :
  monochromatic_triangle_probability (1/2) = 1 - (3/4)^20 :=
by
  sorry

end monochromatic_triangle_probability_correct_l148_148876


namespace correct_expansion_l148_148690

variables {x y : ℝ}

theorem correct_expansion : 
  (-x + y)^2 = x^2 - 2 * x * y + y^2 := sorry

end correct_expansion_l148_148690


namespace jill_sales_goal_l148_148480

def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def boxes_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer
def boxes_left : ℕ := 75
def sales_goal : ℕ := boxes_sold + boxes_left

theorem jill_sales_goal : sales_goal = 150 := by
  sorry

end jill_sales_goal_l148_148480


namespace triangle_area_50_l148_148560

theorem triangle_area_50 :
  let A := (0, 0)
  let B := (0, 10)
  let C := (-10, 0)
  let base := 10
  let height := 10
  0 + base * height / 2 = 50 := by
sorry

end triangle_area_50_l148_148560


namespace algebraic_expression_value_l148_148697

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 19 - 1) : x^2 + 2 * x + 2 = 20 := by
  sorry

end algebraic_expression_value_l148_148697


namespace dvd_player_movie_ratio_l148_148076

theorem dvd_player_movie_ratio (M D : ℝ) (h1 : D = M + 63) (h2 : D = 81) : D / M = 4.5 :=
by
  sorry

end dvd_player_movie_ratio_l148_148076


namespace exactly_one_valid_N_l148_148800

def four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

def condition (N x a : ℕ) : Prop := 
  N = 1000 * a + x ∧ x = N / 7

theorem exactly_one_valid_N : 
  ∃! N : ℕ, ∃ x a : ℕ, four_digit_number N ∧ condition N x a :=
sorry

end exactly_one_valid_N_l148_148800


namespace total_selling_price_l148_148597

theorem total_selling_price
  (meters_cloth : ℕ)
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter)
  (total_selling_price : ℕ := selling_price_per_meter * meters_cloth)
  (h_mc : meters_cloth = 75)
  (h_ppm : profit_per_meter = 15)
  (h_cppm : cost_price_per_meter = 51)
  (h_spm : selling_price_per_meter = 66)
  (h_tsp : total_selling_price = 4950) : 
  total_selling_price = 4950 := 
  by
  -- Skipping the actual proof
  trivial

end total_selling_price_l148_148597


namespace max_t_for_real_root_l148_148340

theorem max_t_for_real_root (t : ℝ) (x : ℝ) 
  (h : 0 < x ∧ x < π ∧ (t+1) * Real.cos x - t * Real.sin x = t + 2) : t = -1 :=
sorry

end max_t_for_real_root_l148_148340


namespace M_intersect_N_l148_148301

-- Definition of the sets M and N
def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≤ x}

-- Proposition to be proved
theorem M_intersect_N : M ∩ N = {0, 1} := 
by 
  sorry

end M_intersect_N_l148_148301


namespace fraction_equiv_subtract_l148_148879

theorem fraction_equiv_subtract (n : ℚ) : (4 - n) / (7 - n) = 3 / 5 → n = 0.5 :=
by
  intros h
  sorry

end fraction_equiv_subtract_l148_148879


namespace problem1_problem2_l148_148287

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)

-- Problem 1: Proving the range of x
theorem problem1 (x : ℝ) (h₁ : a = -1) (h₂ : ∀ (x : ℝ), p x a → q x) : 
  x ∈ {x : ℝ | -6 ≤ x ∧ x < -3} ∨ x ∈ {x : ℝ | 1 < x ∧ x ≤ 12} := sorry

-- Problem 2: Proving the range of a
theorem problem2 (a : ℝ) (h₃ : (∀ x, q x → p x a) ∧ ¬ (∀ x, ¬q x → ¬p x a)) : 
  -4 ≤ a ∧ a ≤ -2 := sorry

end problem1_problem2_l148_148287


namespace max_fraction_sum_l148_148254

theorem max_fraction_sum (a b c : ℝ) 
  (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum: a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_fraction_sum_l148_148254


namespace find_t_correct_l148_148822

theorem find_t_correct : 
  ∃ t : ℝ, (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 15) = 15 * x^4 - 47 * x^3 + 115 * x^2 - 110 * x + 75) ∧ t = -10 :=
sorry

end find_t_correct_l148_148822


namespace runner_speed_comparison_l148_148708

theorem runner_speed_comparison
  (t1 t2 : ℕ → ℝ) -- function to map lap-time.
  (s v1 v2 : ℝ)  -- speed of runners v1 and v2 respectively, and the street distance s.
  (h1 : t1 1 < t2 1) -- first runner overtakes the second runner twice implying their lap-time comparison.
  (h2 : ∀ n, t1 (n + 1) = t1 n + t1 1) -- lap time consistency for runner 1
  (h3 : ∀ n, t2 (n + 1) = t2 n + t2 1) -- lap time consistency for runner 2
  (h4 : t1 3 < t2 2) -- first runner completes 3 laps faster than second runner completes 2 laps
   : 2 * v2 ≤ v1 := sorry

end runner_speed_comparison_l148_148708


namespace jellybean_addition_l148_148332

-- Definitions related to the problem
def initial_jellybeans : ℕ := 37
def removed_jellybeans_initial : ℕ := 15
def added_jellybeans (x : ℕ) : ℕ := x
def removed_jellybeans_again : ℕ := 4
def final_jellybeans : ℕ := 23

-- Prove that the number of jellybeans added back (x) is 5
theorem jellybean_addition (x : ℕ) 
  (h1 : initial_jellybeans - removed_jellybeans_initial + added_jellybeans x - removed_jellybeans_again = final_jellybeans) : 
  x = 5 :=
sorry

end jellybean_addition_l148_148332


namespace unique_solution_conditions_l148_148205

-- Definitions based on the conditions
variables {x y a : ℝ}

def inequality_condition (x y a : ℝ) : Prop := 
  x^2 + y^2 + 2 * x ≤ 1

def equation_condition (x y a : ℝ) : Prop := 
  x - y = -a

-- Main Theorem Statement
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, inequality_condition x y a ∧ equation_condition x y a) ↔ (a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2) :=
sorry

end unique_solution_conditions_l148_148205


namespace program_output_l148_148405

theorem program_output :
  let a := 1
  let b := 3
  let a := a + b
  let b := b * a
  a = 4 ∧ b = 12 :=
by
  sorry

end program_output_l148_148405


namespace quadratic_inequality_solution_set_l148_148956

variable (a b : ℝ)

theorem quadratic_inequality_solution_set :
  (∀ x : ℝ, (a + b) * x + 2 * a - 3 * b < 0 ↔ x > -(3 / 4)) →
  (∀ x : ℝ, (a - 2 * b) * x ^ 2 + 2 * (a - b - 1) * x + (a - 2) > 0 ↔ -3 + 2 / b < x ∧ x < -1) :=
by
  sorry

end quadratic_inequality_solution_set_l148_148956


namespace construction_costs_correct_l148_148433

structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPerThousand : ℕ
  tileCostPerTile : ℕ
  landRequired : ℕ
  bricksRequired : ℕ
  tilesRequired : ℕ

noncomputable def totalConstructionCost (cc : ConstructionCosts) : ℕ :=
  let landCost := cc.landRequired * cc.landCostPerSqMeter
  let brickCost := (cc.bricksRequired / 1000) * cc.brickCostPerThousand
  let tileCost := cc.tilesRequired * cc.tileCostPerTile
  landCost + brickCost + tileCost

theorem construction_costs_correct (cc : ConstructionCosts)
  (h1 : cc.landCostPerSqMeter = 50)
  (h2 : cc.brickCostPerThousand = 100)
  (h3 : cc.tileCostPerTile = 10)
  (h4 : cc.landRequired = 2000)
  (h5 : cc.bricksRequired = 10000)
  (h6 : cc.tilesRequired = 500) :
  totalConstructionCost cc = 106000 := 
  by 
    sorry

end construction_costs_correct_l148_148433


namespace lives_per_player_l148_148986

theorem lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8) (h2 : additional_players = 2) (h3 : total_lives = 60) : 
  total_lives / (initial_players + additional_players) = 6 :=
by 
  sorry

end lives_per_player_l148_148986


namespace minimum_positive_period_of_f_l148_148310

noncomputable def f (x : ℝ) : ℝ := (1 + (Real.sqrt 3) * Real.tan x) * Real.cos x

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

end minimum_positive_period_of_f_l148_148310


namespace infinite_fractions_2_over_odd_l148_148437

theorem infinite_fractions_2_over_odd (a b : ℕ) (n : ℕ) : 
  (a = 2 → 2 * b + 1 ≠ 0) ∧ ((b = 2 * n + 1) → (2 + 2) / (2 * (2 * n + 1)) = 2 / (2 * n + 1)) ∧ (a / b = 2 / (2 * n + 1)) :=
by
  sorry

end infinite_fractions_2_over_odd_l148_148437


namespace bert_total_stamps_l148_148082

theorem bert_total_stamps (bought_stamps : ℕ) (half_stamps_before : ℕ) (total_stamps_after : ℕ) :
  (bought_stamps = 300) ∧ (half_stamps_before = bought_stamps / 2) → (total_stamps_after = half_stamps_before + bought_stamps) → (total_stamps_after = 450) :=
by
  sorry

end bert_total_stamps_l148_148082


namespace probability_reaching_five_without_returning_to_zero_l148_148173

def reach_position_without_return_condition (tosses : ℕ) (target : ℤ) (return_limit : ℤ) : ℕ :=
  -- Ideally we should implement the logic to find the number of valid paths here (as per problem constraints)
  sorry

theorem probability_reaching_five_without_returning_to_zero {a b : ℕ} (h_rel_prime : Nat.gcd a b = 1)
    (h_paths_valid : reach_position_without_return_condition 10 5 3 = 15) :
    a = 15 ∧ b = 256 ∧ a + b = 271 :=
by
  sorry

end probability_reaching_five_without_returning_to_zero_l148_148173


namespace average_salary_of_employees_l148_148904

theorem average_salary_of_employees (A : ℝ) 
  (h1 : (20 : ℝ) * A + 3400 = 21 * (A + 100)) : 
  A = 1300 := 
by 
  -- proof goes here 
  sorry

end average_salary_of_employees_l148_148904


namespace miles_monday_calculation_l148_148532

-- Define the constants
def flat_fee : ℕ := 150
def cost_per_mile : ℝ := 0.50
def miles_thursday : ℕ := 744
def total_cost : ℕ := 832

-- Define the equation to be proved
theorem miles_monday_calculation :
  ∃ M : ℕ, (flat_fee + (M : ℝ) * cost_per_mile + (miles_thursday : ℝ) * cost_per_mile = total_cost) ∧ M = 620 :=
by
  sorry

end miles_monday_calculation_l148_148532


namespace preferred_point_condition_l148_148674

theorem preferred_point_condition (x y : ℝ) (h₁ : x^2 + y^2 ≤ 2008)
  (cond : ∀ x' y', (x'^2 + y'^2 ≤ 2008) → (x' ≤ x → y' ≥ y) → (x = x' ∧ y = y')) :
  x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 :=
by
  sorry

end preferred_point_condition_l148_148674


namespace validColoringsCount_l148_148054

-- Define the initial conditions
def isValidColoring (n : ℕ) (color : ℕ → ℕ) : Prop :=
  ∀ i ∈ Finset.range (n - 1), 
    (i % 2 = 1 → (color i = 1 ∨ color i = 3)) ∧
    color i ≠ color (i + 1)

noncomputable def countValidColorings : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => 
    match n % 2 with
    | 0 => 2 * 3^(n/2)
    | _ => 4 * 3^((n-1)/2)

-- Main theorem
theorem validColoringsCount (n : ℕ) :
  (∀ color : ℕ → ℕ, isValidColoring n color) →
  (if n % 2 = 0 then countValidColorings n = 4 * 3^((n / 2) - 1) 
     else countValidColorings n = 2 * 3^(n / 2)) :=
by
  sorry

end validColoringsCount_l148_148054


namespace bullet_train_speed_is_70kmph_l148_148502

noncomputable def bullet_train_speed (train_length time_man  : ℚ) (man_speed_kmph : ℕ) : ℚ :=
  let man_speed_ms : ℚ := man_speed_kmph * 1000 / 3600
  let relative_speed : ℚ := train_length / time_man
  let train_speed_ms : ℚ := relative_speed - man_speed_ms
  train_speed_ms * 3600 / 1000

theorem bullet_train_speed_is_70kmph :
  bullet_train_speed 160 7.384615384615384 8 = 70 :=
by {
  -- Proof is omitted
  sorry
}

end bullet_train_speed_is_70kmph_l148_148502


namespace solutions_to_x_squared_eq_x_l148_148544

theorem solutions_to_x_squared_eq_x (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := 
sorry

end solutions_to_x_squared_eq_x_l148_148544


namespace max_cross_section_area_l148_148050

noncomputable def prism_cross_section_area : ℝ :=
  let z_axis_parallel := true
  let square_base := 8
  let plane := ∀ x y z, 3 * x - 5 * y + 2 * z = 20
  121.6

theorem max_cross_section_area :
  prism_cross_section_area = 121.6 :=
sorry

end max_cross_section_area_l148_148050


namespace Carlos_gave_Rachel_21_blocks_l148_148710

def initial_blocks : Nat := 58
def remaining_blocks : Nat := 37
def given_blocks : Nat := initial_blocks - remaining_blocks

theorem Carlos_gave_Rachel_21_blocks : given_blocks = 21 :=
by
  sorry

end Carlos_gave_Rachel_21_blocks_l148_148710


namespace rhind_papyrus_prob_l148_148705

theorem rhind_papyrus_prob (a₁ a₂ a₃ a₄ a₅ : ℝ) (q : ℝ) 
  (h_geom_seq : a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3 ∧ a₅ = a₁ * q^4)
  (h_loaves_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 93)
  (h_condition : a₁ + a₂ = (3/4) * a₃) 
  (q_gt_one : q > 1) :
  a₃ = 12 :=
sorry

end rhind_papyrus_prob_l148_148705


namespace find_denominators_l148_148959

theorem find_denominators (f1 f2 f3 f4 f5 f6 f7 f8 f9 : ℚ)
  (h1 : f1 = 1/3) (h2 : f2 = 1/7) (h3 : f3 = 1/9) (h4 : f4 = 1/11) (h5 : f5 = 1/33)
  (h6 : ∃ (d₁ d₂ d₃ d₄ : ℕ), f6 = 1/d₁ ∧ f7 = 1/d₂ ∧ f8 = 1/d₃ ∧ f9 = 1/d₄ ∧
    (∀ d, d ∈ [d₁, d₂, d₃, d₄] → d % 10 = 5))
  (h7 : f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 = 1) :
  ∃ (d₁ d₂ d₃ d₄ : ℕ), (d₁ = 5) ∧ (d₂ = 15) ∧ (d₃ = 45) ∧ (d₄ = 385) :=
by
  sorry

end find_denominators_l148_148959


namespace problem_1_problem_2_l148_148121

theorem problem_1 {m : ℝ} (h₁ : 0 < m) (h₂ : ∀ x : ℝ, (m - |x + 2| ≥ 0) ↔ (-3 ≤ x ∧ x ≤ -1)) :
  m = 1 :=
sorry

theorem problem_2 {a b c : ℝ} (h₃ : 0 < a ∧ 0 < b ∧ 0 < c) (h₄ : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1)
  : a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_1_problem_2_l148_148121


namespace find_b_l148_148369

noncomputable def f (x : ℝ) : ℝ := (x+1)^3 + (x / (x + 1))

theorem find_b (b : ℝ) (h_sum : ∃ x1 x2 : ℝ, f x1 = -x1 + b ∧ f x2 = -x2 + b ∧ x1 + x2 = -2) : b = 0 :=
by
  sorry

end find_b_l148_148369


namespace solve_for_m_l148_148466

theorem solve_for_m (m : ℝ) (f g : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^2 - 2 * x + m) →
  (∀ x : ℝ, g x = x^2 - 2 * x + 9 * m) →
  f 2 = 2 * g 2 →
  m = 0 :=
  by
    intros hf hg hs
    sorry

end solve_for_m_l148_148466


namespace simplify_fraction_l148_148723

theorem simplify_fraction :
  (3 * (Real.sqrt 3 + Real.sqrt 8)) / (2 * Real.sqrt (3 + Real.sqrt 5)) = 
  (297 - 99 * Real.sqrt 5 + 108 * Real.sqrt 6 - 36 * Real.sqrt 30) / 16 := by
  sorry

end simplify_fraction_l148_148723


namespace athlete_heartbeats_l148_148027

def heart_beats_per_minute : ℕ := 120
def running_pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 30
def total_heartbeats : ℕ := 21600

theorem athlete_heartbeats :
  (running_pace_minutes_per_mile * race_distance_miles * heart_beats_per_minute) = total_heartbeats :=
by
  sorry

end athlete_heartbeats_l148_148027


namespace shoe_size_combination_l148_148686

theorem shoe_size_combination (J A : ℕ) (hJ : J = 7) (hA : A = 2 * J) : J + A = 21 := by
  sorry

end shoe_size_combination_l148_148686


namespace inequality_proof_l148_148291

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l148_148291


namespace multiply_negatives_l148_148960

theorem multiply_negatives : (-2) * (-3) = 6 :=
  by 
  sorry

end multiply_negatives_l148_148960


namespace maddox_theo_equal_profit_l148_148915

-- Definitions based on the problem conditions
def maddox_initial_cost := 10 * 35
def theo_initial_cost := 15 * 30
def maddox_revenue := 10 * 50
def theo_revenue := 15 * 40

-- Define profits based on the revenues and costs
def maddox_profit := maddox_revenue - maddox_initial_cost
def theo_profit := theo_revenue - theo_initial_cost

-- The theorem to be proved
theorem maddox_theo_equal_profit : maddox_profit = theo_profit :=
by
  -- Omitted proof steps
  sorry

end maddox_theo_equal_profit_l148_148915


namespace initial_birds_in_cage_l148_148522

-- Define a theorem to prove the initial number of birds in the cage
theorem initial_birds_in_cage (B : ℕ) 
  (H1 : 2 / 15 * B = 8) : B = 60 := 
by sorry

end initial_birds_in_cage_l148_148522


namespace well_diameter_l148_148373

theorem well_diameter 
  (h : ℝ) 
  (P : ℝ) 
  (C : ℝ) 
  (V : ℝ) 
  (r : ℝ) 
  (d : ℝ) 
  (π : ℝ) 
  (h_eq : h = 14)
  (P_eq : P = 15)
  (C_eq : C = 1484.40)
  (V_eq : V = C / P)
  (volume_eq : V = π * r^2 * h)
  (radius_eq : r^2 = V / (π * h))
  (diameter_eq : d = 2 * r) : 
  d = 3 :=
by
  sorry

end well_diameter_l148_148373


namespace roses_in_each_bouquet_l148_148193

theorem roses_in_each_bouquet (R : ℕ)
(roses_bouquets daisies_bouquets total_bouquets total_flowers daisies_per_bouquet total_daisies : ℕ)
(h1 : total_bouquets = 20)
(h2 : roses_bouquets = 10)
(h3 : daisies_bouquets = 10)
(h4 : total_flowers = 190)
(h5 : daisies_per_bouquet = 7)
(h6 : total_daisies = daisies_bouquets * daisies_per_bouquet)
(h7 : total_flowers - total_daisies = roses_bouquets * R) :
R = 12 :=
by
  sorry

end roses_in_each_bouquet_l148_148193


namespace scientific_notation_of_192M_l148_148138

theorem scientific_notation_of_192M : 192000000 = 1.92 * 10^8 :=
by 
  sorry

end scientific_notation_of_192M_l148_148138


namespace Dana_Colin_relationship_l148_148250

variable (C : ℝ) -- Let C be the number of cards Colin has.

def Ben_cards (C : ℝ) : ℝ := 1.20 * C -- Ben has 20% more cards than Colin
def Dana_cards (C : ℝ) : ℝ := 1.40 * Ben_cards C + Ben_cards C -- Dana has 40% more cards than Ben

theorem Dana_Colin_relationship : Dana_cards C = 1.68 * C := by
  sorry

end Dana_Colin_relationship_l148_148250


namespace cost_of_one_dozen_pens_l148_148971

theorem cost_of_one_dozen_pens (pen pencil : ℝ) (h_ratios : pen = 5 * pencil) (h_total : 3 * pen + 5 * pencil = 240) :
  12 * pen = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l148_148971


namespace consumer_installment_credit_l148_148288

theorem consumer_installment_credit (A C : ℝ) 
  (h1 : A = 0.36 * C) 
  (h2 : 57 = 1 / 3 * A) : 
  C = 475 := 
by 
  sorry

end consumer_installment_credit_l148_148288


namespace problem_solution_l148_148699

variable (a b c : ℝ)

theorem problem_solution (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) :
  a + b ≤ 3 * c := 
sorry

end problem_solution_l148_148699


namespace units_digit_2016_pow_2017_add_2017_pow_2016_l148_148186

theorem units_digit_2016_pow_2017_add_2017_pow_2016 :
  (2016 ^ 2017 + 2017 ^ 2016) % 10 = 7 :=
by
  sorry

end units_digit_2016_pow_2017_add_2017_pow_2016_l148_148186


namespace tom_driving_speed_l148_148410

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end tom_driving_speed_l148_148410


namespace find_constant_l148_148566

theorem find_constant
  (k : ℝ)
  (r : ℝ := 36)
  (C : ℝ := 72 * k)
  (h1 : C = 2 * Real.pi * r)
  : k = Real.pi := by
  sorry

end find_constant_l148_148566


namespace apples_added_l148_148925

theorem apples_added (initial_apples added_apples final_apples : ℕ) 
  (h1 : initial_apples = 8) 
  (h2 : final_apples = 13) 
  (h3 : final_apples = initial_apples + added_apples) : 
  added_apples = 5 :=
by
  sorry

end apples_added_l148_148925


namespace isosceles_triangle_l148_148354

theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ) (hAcosB : a * Real.cos B = b * Real.cos A) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end isosceles_triangle_l148_148354


namespace number_of_feet_on_branches_l148_148734

def number_of_birds : ℕ := 46
def feet_per_bird : ℕ := 2

theorem number_of_feet_on_branches : number_of_birds * feet_per_bird = 92 := 
by 
  sorry

end number_of_feet_on_branches_l148_148734


namespace area_union_square_circle_l148_148360

noncomputable def side_length_square : ℝ := 12
noncomputable def radius_circle : ℝ := 15
noncomputable def area_union : ℝ := 144 + 168.75 * Real.pi

theorem area_union_square_circle : 
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * radius_circle ^ 2
  let area_quarter_circle := area_circle / 4
  area_union = area_square + area_circle - area_quarter_circle :=
by
  -- The actual proof is omitted
  sorry

end area_union_square_circle_l148_148360


namespace total_percentage_increase_l148_148109

def initial_salary : Float := 60
def first_raise (s : Float) : Float := s + 0.10 * s
def second_raise (s : Float) : Float := s + 0.15 * s
def deduction (s : Float) : Float := s - 0.05 * s
def promotion_raise (s : Float) : Float := s + 0.20 * s
def final_salary (s : Float) : Float := promotion_raise (deduction (second_raise (first_raise s)))

theorem total_percentage_increase :
  final_salary initial_salary = initial_salary * 1.4421 :=
by
  sorry

end total_percentage_increase_l148_148109


namespace union_sets_l148_148863

-- Given sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 4, 5} := by
  sorry

end union_sets_l148_148863


namespace expression_value_l148_148554

theorem expression_value : (19 + 12) ^ 2 - (12 ^ 2 + 19 ^ 2) = 456 := 
by sorry

end expression_value_l148_148554


namespace integer_pairs_satisfy_equation_l148_148567

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), (x^2 * y + y^2 = x^3) → (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) :=
by
  sorry

end integer_pairs_satisfy_equation_l148_148567


namespace polygon_interior_exterior_relation_l148_148032

theorem polygon_interior_exterior_relation (n : ℕ) (h1 : (n-2) * 180 = 2 * 360) : n = 6 :=
by sorry

end polygon_interior_exterior_relation_l148_148032


namespace possible_values_of_a_l148_148376

theorem possible_values_of_a (a : ℝ) :
  (∃ x, ∀ y, (y = x) ↔ (a * y^2 + 2 * y + a = 0))
  → (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end possible_values_of_a_l148_148376


namespace tangent_line_equation_l148_148348

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P = (-4, -3)) :
  ∃ (a b c : ℝ), a * -4 + b * -3 + c = 0 ∧ a * a + b * b = (5:ℝ)^2 ∧ 
                 a = 4 ∧ b = 3 ∧ c = 25 := 
sorry

end tangent_line_equation_l148_148348


namespace two_baskets_of_peaches_l148_148221

theorem two_baskets_of_peaches (R G : ℕ) (h1 : G = R + 2) (h2 : 2 * R + 2 * G = 12) : R = 2 :=
by
  sorry

end two_baskets_of_peaches_l148_148221


namespace hyperbola_eccentricity_l148_148782

theorem hyperbola_eccentricity : 
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e = Real.sqrt 5 / 2 := 
by
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  sorry

end hyperbola_eccentricity_l148_148782


namespace smallest_number_of_fruits_l148_148228

theorem smallest_number_of_fruits 
  (n_apple_slices : ℕ) (n_grapes : ℕ) (n_orange_wedges : ℕ) (n_cherries : ℕ)
  (h_apple : n_apple_slices = 18)
  (h_grape : n_grapes = 9)
  (h_orange : n_orange_wedges = 12)
  (h_cherry : n_cherries = 6)
  : ∃ (n : ℕ), n = 36 ∧ (n % n_apple_slices = 0) ∧ (n % n_grapes = 0) ∧ (n % n_orange_wedges = 0) ∧ (n % n_cherries = 0) :=
sorry

end smallest_number_of_fruits_l148_148228


namespace breadth_increase_25_percent_l148_148507

variable (L B : ℝ) 

-- Conditions
def original_area := L * B
def increased_length := 1.10 * L
def increased_area := 1.375 * (original_area L B)

-- The breadth increase percentage (to be proven as 25)
def percentage_increase_breadth (p : ℝ) := 
  increased_area L B = increased_length L * (B * (1 + p/100))

-- The statement to be proven
theorem breadth_increase_25_percent : 
  percentage_increase_breadth L B 25 := 
sorry

end breadth_increase_25_percent_l148_148507


namespace original_three_digit_number_a_original_three_digit_number_b_l148_148275

section ProblemA

variables {x y z : ℕ}

/-- In a three-digit number, the first digit on the left was erased. Then, the resulting
  two-digit number was multiplied by 7, and the original three-digit number was obtained. -/
theorem original_three_digit_number_a (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  N = 7 * (10 * y + z)) : ∃ (N : ℕ), N = 350 :=
sorry

end ProblemA

section ProblemB

variables {x y z : ℕ}

/-- In a three-digit number, the middle digit was erased, and the resulting number 
  is 6 times smaller than the original. --/
theorem original_three_digit_number_b (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  6 * (10 * x + z) = N) : ∃ (N : ℕ), N = 108 :=
sorry

end ProblemB

end original_three_digit_number_a_original_three_digit_number_b_l148_148275


namespace calculate_expression_l148_148805

def seq (k : Nat) : Nat := 2^k + 3^k

def product_seq : Nat :=
  (2 + 3) * (2^3 + 3^3) * (2^6 + 3^6) * (2^12 + 3^12) * (2^24 + 3^24)

theorem calculate_expression :
  product_seq = (3^47 - 2^47) :=
sorry

end calculate_expression_l148_148805


namespace probability_no_neighbouring_same_color_l148_148416

-- Given conditions
def red_beads : ℕ := 4
def white_beads : ℕ := 2
def blue_beads : ℕ := 2
def total_beads : ℕ := red_beads + white_beads + blue_beads

-- Total permutations
def total_orderings : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- Probability calculation proof
theorem probability_no_neighbouring_same_color : (30 / 420 : ℚ) = (1 / 14 : ℚ) :=
by
  -- proof steps
  sorry

end probability_no_neighbouring_same_color_l148_148416


namespace min_function_value_in_domain_l148_148536

theorem min_function_value_in_domain :
  ∃ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (∀ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) → (xy / (x^2 + y^2)) ≥ (60 / 169)) :=
sorry

end min_function_value_in_domain_l148_148536


namespace find_number_of_students_l148_148326

open Nat

theorem find_number_of_students :
  ∃ n : ℕ, 35 < n ∧ n < 70 ∧ n % 6 = 3 ∧ n % 8 = 1 ∧ n = 57 :=
by
  use 57
  sorry

end find_number_of_students_l148_148326


namespace find_k_l148_148523

-- Definitions of the conditions as given in the problem
def total_amount (A B C : ℕ) : Prop := A + B + C = 585
def c_share (C : ℕ) : Prop := C = 260
def equal_shares (A B C k : ℕ) : Prop := 4 * A = k * C ∧ 6 * B = k * C

-- The theorem we need to prove
theorem find_k (A B C k : ℕ) (h_tot: total_amount A B C)
  (h_c: c_share C) (h_eq: equal_shares A B C k) : k = 3 := by 
  sorry

end find_k_l148_148523


namespace jack_walked_time_l148_148494

def jack_distance : ℝ := 9
def jack_rate : ℝ := 7.2
def jack_time : ℝ := 1.25

theorem jack_walked_time : jack_time = jack_distance / jack_rate := by
  sorry

end jack_walked_time_l148_148494


namespace triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l148_148856

noncomputable def a : ℝ := sorry
noncomputable def d : ℝ := a / 4
noncomputable def half_perimeter : ℝ := (a - d + a + (a + d)) / 2
noncomputable def r : ℝ := ((a - d) + a + (a + d)) / 2

theorem triangle_right_angled_and_common_difference_equals_inscribed_circle_radius :
  (half_perimeter > a + d) →
  ((a - d) + a + (a + d) = 2 * half_perimeter) →
  (a - d)^2 + a^2 = (a + d)^2 →
  d = r :=
by
  intros h1 h2 h3
  sorry

end triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l148_148856


namespace remaining_amount_correct_l148_148148

-- Definitions for the given conditions
def deposit_percentage : ℝ := 0.05
def deposit_amount : ℝ := 50

-- The correct answer we need to prove
def remaining_amount_to_be_paid : ℝ := 950

-- Stating the theorem (proof not required)
theorem remaining_amount_correct (total_price : ℝ) 
    (H1 : deposit_amount = total_price * deposit_percentage) : 
    total_price - deposit_amount = remaining_amount_to_be_paid :=
by
  sorry

end remaining_amount_correct_l148_148148


namespace correct_equation_is_x2_sub_10x_add_9_l148_148015

-- Define the roots found by Student A and Student B
def roots_A := (8, 2)
def roots_B := (-9, -1)

-- Define the incorrect equation by student A from given roots
def equation_A (x : ℝ) := x^2 - 10 * x + 16

-- Define the incorrect equation by student B from given roots
def equation_B (x : ℝ) := x^2 + 10 * x + 9

-- Define the correct quadratic equation
def correct_quadratic_equation (x : ℝ) := x^2 - 10 * x + 9

-- Theorem stating that the correct quadratic equation balances the errors of both students
theorem correct_equation_is_x2_sub_10x_add_9 :
  ∃ (eq_correct : ℝ → ℝ), 
    eq_correct = correct_quadratic_equation :=
by
  -- proof will go here
  sorry

end correct_equation_is_x2_sub_10x_add_9_l148_148015


namespace customer_payment_strawberries_watermelons_max_discount_value_l148_148581

-- Definitions for prices
def price_strawberries : ℕ := 60
def price_jingbai_pears : ℕ := 65
def price_watermelons : ℕ := 80
def price_peaches : ℕ := 90

-- Definition for condition on minimum purchase for promotion
def min_purchase_for_promotion : ℕ := 120

-- Definition for percentage Li Ming receives
def li_ming_percentage : ℕ := 80
def customer_percentage : ℕ := 100

-- Proof problem for part 1
theorem customer_payment_strawberries_watermelons (x : ℕ) (total_price : ℕ) :
  x = 10 →
  total_price = price_strawberries + price_watermelons →
  total_price >= min_purchase_for_promotion →
  total_price - x = 130 :=
  by sorry

-- Proof problem for part 2
theorem max_discount_value (m x : ℕ) :
  m >= min_purchase_for_promotion →
  (m - x) * li_ming_percentage / customer_percentage ≥ m * 7 / 10 →
  x ≤ m / 8 :=
  by sorry

end customer_payment_strawberries_watermelons_max_discount_value_l148_148581


namespace paul_baseball_cards_l148_148916

-- Define the necessary variables and statements
variable {n : ℕ}

-- State the problem and the proof target
theorem paul_baseball_cards : ∃ k, k = 3 * n + 1 := sorry

end paul_baseball_cards_l148_148916


namespace total_legs_and_hands_on_ground_is_118_l148_148379

-- Definitions based on the conditions given
def total_dogs := 20
def dogs_on_two_legs := total_dogs / 2
def dogs_on_four_legs := total_dogs / 2

def total_cats := 10
def cats_on_two_legs := total_cats / 3
def cats_on_four_legs := total_cats - cats_on_two_legs

def total_horses := 5
def horses_on_two_legs := 2
def horses_on_four_legs := total_horses - horses_on_two_legs

def total_acrobats := 6
def acrobats_on_one_hand := 4
def acrobats_on_two_hands := 2

-- Functions to calculate the number of legs/paws/hands on the ground
def dogs_legs_on_ground := (dogs_on_two_legs * 2) + (dogs_on_four_legs * 4)
def cats_legs_on_ground := (cats_on_two_legs * 2) + (cats_on_four_legs * 4)
def horses_legs_on_ground := (horses_on_two_legs * 2) + (horses_on_four_legs * 4)
def acrobats_hands_on_ground := (acrobats_on_one_hand * 1) + (acrobats_on_two_hands * 2)

-- Total legs/paws/hands on the ground
def total_legs_on_ground := dogs_legs_on_ground + cats_legs_on_ground + horses_legs_on_ground + acrobats_hands_on_ground

-- The theorem to prove
theorem total_legs_and_hands_on_ground_is_118 : total_legs_on_ground = 118 :=
by sorry

end total_legs_and_hands_on_ground_is_118_l148_148379


namespace eval_expression_l148_148036

open Real

theorem eval_expression :
  (0.8^5 - (0.5^6 / 0.8^4) + 0.40 + 0.5^3 - log 0.3 + sin (π / 6)) = 2.51853302734375 :=
  sorry

end eval_expression_l148_148036


namespace polygon_sides_eq_seven_l148_148440

theorem polygon_sides_eq_seven (n : ℕ) (h : 2 * n - (n * (n - 3)) / 2 = 0) : n = 7 :=
by sorry

end polygon_sides_eq_seven_l148_148440


namespace reciprocal_sum_l148_148391

variable {x y z a b c : ℝ}

-- The function statement where we want to show the equivalence.
theorem reciprocal_sum (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxy : (x * y) / (x - y) = a)
  (hxz : (x * z) / (x - z) = b)
  (hyz : (y * z) / (y - z) = c) :
  (1/x + 1/y + 1/z) = ((1/a + 1/b + 1/c) / 2) :=
sorry

end reciprocal_sum_l148_148391


namespace population_multiple_of_seven_l148_148243

theorem population_multiple_of_seven 
  (a b c : ℕ) 
  (h1 : a^2 + 100 = b^2 + 1) 
  (h2 : b^2 + 1 + 100 = c^2) : 
  (∃ k : ℕ, a = 7 * k) :=
sorry

end population_multiple_of_seven_l148_148243


namespace piggy_bank_after_8_weeks_l148_148612

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end piggy_bank_after_8_weeks_l148_148612


namespace lifespan_of_bat_l148_148542

theorem lifespan_of_bat (B : ℕ) (h₁ : ∀ B, B - 6 < B)
    (h₂ : ∀ B, 4 * (B - 6) < 4 * B)
    (h₃ : B + (B - 6) + 4 * (B - 6) = 30) :
    B = 10 := by
  sorry

end lifespan_of_bat_l148_148542


namespace negation_of_exists_l148_148048

theorem negation_of_exists (x : ℝ) :
  ¬ (∃ x > 0, 2 * x + 3 ≤ 0) ↔ ∀ x > 0, 2 * x + 3 > 0 :=
by
  sorry

end negation_of_exists_l148_148048


namespace solve_for_x_l148_148308

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end solve_for_x_l148_148308


namespace expand_expression_l148_148115

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 :=
by
  sorry

end expand_expression_l148_148115


namespace avg_price_pen_is_correct_l148_148152

-- Definitions for the total numbers and expenses:
def number_of_pens : ℕ := 30
def number_of_pencils : ℕ := 75
def total_cost : ℕ := 630
def avg_price_pencil : ℝ := 2.00

-- Calculation of total cost for pencils and pens
def total_cost_pencils : ℝ := number_of_pencils * avg_price_pencil
def total_cost_pens : ℝ := total_cost - total_cost_pencils

-- Statement to prove:
theorem avg_price_pen_is_correct :
  total_cost_pens / number_of_pens = 16 :=
by
  sorry

end avg_price_pen_is_correct_l148_148152


namespace difference_of_squares_is_149_l148_148315

-- Definitions of the conditions
def are_consecutive (n m : ℤ) : Prop := m = n + 1
def sum_less_than_150 (n : ℤ) : Prop := (n + (n + 1)) < 150

-- The difference of their squares
def difference_of_squares (n m : ℤ) : ℤ := (m * m) - (n * n)

-- Stating the problem where the answer expected is 149
theorem difference_of_squares_is_149 :
  ∀ n : ℤ, 
  ∀ m : ℤ,
  are_consecutive n m →
  sum_less_than_150 n →
  difference_of_squares n m = 149 :=
by
  sorry

end difference_of_squares_is_149_l148_148315


namespace point_A_coordinates_l148_148146

variable (a x y : ℝ)

def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

theorem point_A_coordinates (h1 : ∃ t : ℝ, ∀ x : ℝ, f a x = t * x + t) (h2 : x = 0) : (0, 2) = (0, f a 0) :=
by
  sorry

end point_A_coordinates_l148_148146


namespace max_crosses_in_grid_l148_148500

theorem max_crosses_in_grid : ∀ (n : ℕ), n = 16 → (∃ X : ℕ, X = 30 ∧
  ∀ (i j : ℕ), i < n → j < n → 
    (∀ k, k < n → (i ≠ k → X ≠ k)) ∧ 
    (∀ l, l < n → (j ≠ l → X ≠ l))) :=
by
  sorry

end max_crosses_in_grid_l148_148500


namespace continuity_necessity_not_sufficiency_l148_148514

theorem continuity_necessity_not_sufficiency (f : ℝ → ℝ) (x₀ : ℝ) :
  ((∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) → f x₀ = f x₀) ∧ ¬ ((f x₀ = f x₀) → (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε)) := 
sorry

end continuity_necessity_not_sufficiency_l148_148514


namespace radius_of_base_of_cone_is_3_l148_148022

noncomputable def radius_of_base_of_cone (θ R : ℝ) : ℝ :=
  ((θ / 360) * 2 * Real.pi * R) / (2 * Real.pi)

theorem radius_of_base_of_cone_is_3 :
  radius_of_base_of_cone 120 9 = 3 := 
by 
  simp [radius_of_base_of_cone]
  sorry

end radius_of_base_of_cone_is_3_l148_148022


namespace negation_of_existence_statement_l148_148299

theorem negation_of_existence_statement :
  (¬ ∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_existence_statement_l148_148299


namespace ann_age_is_26_l148_148995

theorem ann_age_is_26
  (a b : ℕ)
  (h1 : a + b = 50)
  (h2 : b = 2 * a / 3 + 2 * (a - b)) :
  a = 26 :=
by
  sorry

end ann_age_is_26_l148_148995


namespace factor_and_divisor_statements_l148_148598

theorem factor_and_divisor_statements :
  (∃ n : ℕ, 25 = 5 * n) ∧
  ((∃ n : ℕ, 209 = 19 * n) ∧ ¬ (∃ n : ℕ, 63 = 19 * n)) ∧
  (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end factor_and_divisor_statements_l148_148598


namespace marble_count_l148_148045

theorem marble_count (a : ℕ) (h1 : a + 3 * a + 6 * a + 30 * a = 120) : a = 3 :=
  sorry

end marble_count_l148_148045


namespace sum_of_terms_in_fractional_array_l148_148964

theorem sum_of_terms_in_fractional_array :
  (∑' (r : ℕ) (c : ℕ), (1 : ℝ) / ((3 * 4) ^ r) * (1 / (4 ^ c))) = (1 / 33) := sorry

end sum_of_terms_in_fractional_array_l148_148964


namespace find_a_plus_b_l148_148665

noncomputable def f (a b x : ℝ) := a ^ x + b

theorem find_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (dom1 : f a b (-2) = -2) (dom2 : f a b 0 = 0) :
  a + b = (Real.sqrt 3) / 3 - 3 :=
by
  unfold f at dom1 dom2
  sorry

end find_a_plus_b_l148_148665


namespace time_to_write_all_rearrangements_in_hours_l148_148071

/-- Michael's name length is 7 (number of unique letters) -/
def name_length : Nat := 7

/-- Michael can write 10 rearrangements per minute -/
def write_rate : Nat := 10

/-- Number of rearrangements of Michael's name -/
def num_rearrangements : Nat := (name_length.factorial)

theorem time_to_write_all_rearrangements_in_hours :
  (num_rearrangements / write_rate : ℚ) / 60 = 8.4 := by
  sorry

end time_to_write_all_rearrangements_in_hours_l148_148071


namespace sum_three_positive_numbers_ge_three_l148_148694

theorem sum_three_positive_numbers_ge_three 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 :=
sorry

end sum_three_positive_numbers_ge_three_l148_148694


namespace triangle_inequality_holds_l148_148251

theorem triangle_inequality_holds (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^3 + b^3 + c^3 + 4 * a * b * c ≤ (9 / 32) * (a + b + c)^3 :=
by {
  sorry
}

end triangle_inequality_holds_l148_148251


namespace pebbles_divisibility_impossibility_l148_148670

def initial_pebbles (K A P D : Nat) := K + A + P + D

theorem pebbles_divisibility_impossibility 
  (K A P D : Nat)
  (hK : K = 70)
  (hA : A = 30)
  (hP : P = 21)
  (hD : D = 45) :
  ¬ (∃ n : Nat, initial_pebbles K A P D = 4 * n) :=
by
  sorry

end pebbles_divisibility_impossibility_l148_148670


namespace angle_sum_around_point_l148_148595

theorem angle_sum_around_point (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) : 
    x + y + 130 = 360 → x + y = 230 := by
  sorry

end angle_sum_around_point_l148_148595


namespace cos_C_in_triangle_l148_148497

theorem cos_C_in_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_cos_A : Real.cos A = 3/5)
  (h_sin_B : Real.sin B = 12/13) :
  Real.cos C = 63/65 ∨ Real.cos C = 33/65 :=
sorry

end cos_C_in_triangle_l148_148497


namespace jesters_on_stilts_count_l148_148156

theorem jesters_on_stilts_count :
  ∃ j e : ℕ, 3 * j + 4 * e = 50 ∧ j + e = 18 ∧ j = 22 :=
by 
  sorry

end jesters_on_stilts_count_l148_148156


namespace largest_integer_y_l148_148408

theorem largest_integer_y (y : ℤ) : 
  (∃ k : ℤ, (y^2 + 3*y + 10) = k * (y - 4)) → y ≤ 42 :=
sorry

end largest_integer_y_l148_148408


namespace area_of_cross_section_l148_148086

noncomputable def area_cross_section (H α : ℝ) : ℝ :=
  let AC := 2 * H * Real.sqrt 3 * Real.tan (Real.pi / 2 - α)
  let MK := (H / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2)
  (1 / 2) * AC * MK

theorem area_of_cross_section (H α : ℝ) :
  area_cross_section H α = (H^2 * Real.sqrt 3 * Real.tan (Real.pi / 2 - α) / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2) :=
sorry

end area_of_cross_section_l148_148086


namespace proof_of_diagonal_length_l148_148647

noncomputable def length_of_diagonal (d : ℝ) : Prop :=
  d^2 = 325 ∧ 17^2 + 36 = 325

theorem proof_of_diagonal_length (d : ℝ) : length_of_diagonal d → d = 5 * Real.sqrt 13 :=
by
  intro h
  sorry

end proof_of_diagonal_length_l148_148647


namespace hydrogen_atomic_weight_is_correct_l148_148924

-- Definitions and assumptions based on conditions
def molecular_weight : ℝ := 68
def number_of_hydrogen_atoms : ℕ := 1
def number_of_chlorine_atoms : ℕ := 1
def number_of_oxygen_atoms : ℕ := 2
def atomic_weight_chlorine : ℝ := 35.45
def atomic_weight_oxygen : ℝ := 16.00

-- Definition for the atomic weight of hydrogen to be proved
def atomic_weight_hydrogen (w : ℝ) : Prop :=
  w * number_of_hydrogen_atoms
  + atomic_weight_chlorine * number_of_chlorine_atoms
  + atomic_weight_oxygen * number_of_oxygen_atoms = molecular_weight

-- The theorem to prove the atomic weight of hydrogen
theorem hydrogen_atomic_weight_is_correct : atomic_weight_hydrogen 1.008 :=
by
  unfold atomic_weight_hydrogen
  simp
  sorry

end hydrogen_atomic_weight_is_correct_l148_148924


namespace polynomial_at_one_l148_148452

def f (x : ℝ) : ℝ := x^4 - 7*x^3 - 9*x^2 + 11*x + 7

theorem polynomial_at_one :
  f 1 = 3 := 
by
  sorry

end polynomial_at_one_l148_148452


namespace tangent_lines_count_l148_148503

noncomputable def number_of_tangent_lines (r1 r2 : ℝ) (k : ℕ) : ℕ :=
if r1 = 2 ∧ r2 = 3 then 5 else 0

theorem tangent_lines_count: 
∃ k : ℕ, number_of_tangent_lines 2 3 k = 5 :=
by sorry

end tangent_lines_count_l148_148503


namespace area_of_rectangular_plot_l148_148622

-- Defining the breadth
def breadth : ℕ := 26

-- Defining the length as thrice the breadth
def length : ℕ := 3 * breadth

-- Defining the area as the product of length and breadth
def area : ℕ := length * breadth

-- The theorem stating the problem to prove
theorem area_of_rectangular_plot : area = 2028 := by
  -- Initial proof step skipped
  sorry

end area_of_rectangular_plot_l148_148622


namespace square_areas_l148_148398

theorem square_areas (s1 s2 s3 : ℕ)
  (h1 : s3 = s2 + 1)
  (h2 : s3 = s1 + 2)
  (h3 : s2 = 18)
  (h4 : s1 = s2 - 1) :
  s3^2 = 361 ∧ s2^2 = 324 ∧ s1^2 = 289 :=
by {
sorry
}

end square_areas_l148_148398


namespace find_rs_l148_148530

theorem find_rs :
  ∃ r s : ℝ, ∀ x : ℝ, 8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 8 * (x - r) ^ 2 * (x - s) * (x - 1) :=
sorry

end find_rs_l148_148530


namespace difference_between_picked_and_left_is_five_l148_148377

theorem difference_between_picked_and_left_is_five :
  let dave_sticks := 14
  let amy_sticks := 9
  let ben_sticks := 12
  let total_initial_sticks := 65
  let total_picked_up := dave_sticks + amy_sticks + ben_sticks
  let sticks_left := total_initial_sticks - total_picked_up
  total_picked_up - sticks_left = 5 :=
by
  sorry

end difference_between_picked_and_left_is_five_l148_148377


namespace spent_on_books_l148_148870

theorem spent_on_books (allowance games_fraction snacks_fraction toys_fraction : ℝ)
  (h_allowance : allowance = 50)
  (h_games : games_fraction = 1/4)
  (h_snacks : snacks_fraction = 1/5)
  (h_toys : toys_fraction = 2/5) :
  allowance - (allowance * games_fraction + allowance * snacks_fraction + allowance * toys_fraction) = 7.5 :=
by
  sorry

end spent_on_books_l148_148870


namespace batsman_average_increase_l148_148436

theorem batsman_average_increase (A : ℕ) (H1 : 16 * A + 85 = 17 * (A + 3)) : A + 3 = 37 :=
by {
  sorry
}

end batsman_average_increase_l148_148436


namespace simplify_and_evaluate_l148_148550

noncomputable def a := 2 * Real.sqrt 3 + 3
noncomputable def expr := (1 - 1 / (a - 2)) / ((a ^ 2 - 6 * a + 9) / (2 * a - 4))

theorem simplify_and_evaluate : expr = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l148_148550


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_l148_148673

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4 : 
  Nat.minFac (11^5 - 11^4) = 2 := 
by sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_l148_148673


namespace find_m_and_n_l148_148854

theorem find_m_and_n (x y m n : ℝ) 
  (h1 : 5 * x - 2 * y = 3) 
  (h2 : m * x + 5 * y = 4) 
  (h3 : x - 4 * y = -3) 
  (h4 : 5 * x + n * y = 1) :
  m = -1 ∧ n = -4 :=
by
  sorry

end find_m_and_n_l148_148854


namespace matvey_healthy_diet_l148_148134

theorem matvey_healthy_diet (n b_1 p_1 : ℕ) (h1 : n * b_1 - (n * (n - 1)) / 2 = 264) (h2 : n * p_1 + (n * (n - 1)) / 2 = 187) :
  n = 11 :=
by
  let buns_diff_pears := b_1 - p_1 - (n - 1)
  have buns_def : 264 = n * buns_diff_pears + n * (n - 1) / 2 := sorry
  have pears_def : 187 = n * buns_diff_pears - n * (n - 1) / 2 := sorry
  have diff : 77 = n * buns_diff_pears := sorry
  sorry

end matvey_healthy_diet_l148_148134


namespace elder_person_age_l148_148070

open Nat

variable (y e : ℕ)

-- Conditions
def age_difference := e = y + 16
def age_relation := e - 6 = 3 * (y - 6)

theorem elder_person_age
  (h1 : age_difference y e)
  (h2 : age_relation y e) :
  e = 30 :=
sorry

end elder_person_age_l148_148070


namespace repeating_decimal_fraction_value_l148_148257

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  d

theorem repeating_decimal_fraction_value :
  repeating_decimal_to_fraction (73 / 100 + 246 / 999000) = 731514 / 999900 :=
by
  sorry

end repeating_decimal_fraction_value_l148_148257


namespace find_abc_solutions_l148_148469

theorem find_abc_solutions :
  ∀ (a b c : ℕ),
    (2^(a) * 3^(b) = 7^(c) - 1) ↔
    ((a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 2)) :=
by
  sorry

end find_abc_solutions_l148_148469


namespace simplify_exponents_product_l148_148506

theorem simplify_exponents_product :
  (10^0.5) * (10^0.25) * (10^0.15) * (10^0.05) * (10^1.05) = 100 := by
sorry

end simplify_exponents_product_l148_148506


namespace difference_of_squares_eval_l148_148934

-- Define the conditions
def a : ℕ := 81
def b : ℕ := 49

-- State the corresponding problem and its equivalence
theorem difference_of_squares_eval : (a^2 - b^2) = 4160 := by
  sorry -- Placeholder for the proof

end difference_of_squares_eval_l148_148934


namespace LynsDonation_l148_148208

theorem LynsDonation (X : ℝ)
  (h1 : 1/3 * X + 1/2 * X + 1/4 * (X - (1/3 * X + 1/2 * X)) = 3/4 * X)
  (h2 : (X - 3/4 * X)/4 = 30) :
  X = 240 := by
  sorry

end LynsDonation_l148_148208


namespace determine_omega_l148_148656

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

-- Conditions
variables (ω : ℝ) (ϕ : ℝ)
axiom omega_pos : ω > 0
axiom phi_bound : abs ϕ < Real.pi / 2
axiom symm_condition1 : ∀ x, f ω ϕ (Real.pi / 4 - x) = -f ω ϕ (Real.pi / 4 + x)
axiom symm_condition2 : ∀ x, f ω ϕ (-Real.pi / 2 - x) = f ω ϕ x
axiom monotonic_condition : ∀ x1 x2, 0 < x1 → x1 < x2 → x2 < Real.pi / 8 → f ω ϕ x1 < f ω ϕ x2

theorem determine_omega : ω = 1 ∨ ω = 5 :=
sorry

end determine_omega_l148_148656


namespace find_number_l148_148590

theorem find_number (x : ℕ) (h : 8 * x = 64) : x = 8 :=
sorry

end find_number_l148_148590


namespace fraction_equality_l148_148432

theorem fraction_equality (x y z : ℝ) (k : ℝ) (hx : x = 3 * k) (hy : y = 5 * k) (hz : z = 7 * k) :
  (x - y + z) / (x + y - z) = 5 := 
  sorry

end fraction_equality_l148_148432


namespace mass_of_three_packages_l148_148020

noncomputable def total_mass {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : ℝ := 
  x + y + z

theorem mass_of_three_packages {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : total_mass h1 h2 h3 = 175 :=
by
  sorry

end mass_of_three_packages_l148_148020


namespace spiderCanEatAllFlies_l148_148225

-- Define the number of nodes in the grid.
def numNodes := 100

-- Define initial conditions.
def cornerStart := true
def numFlies := 100
def fliesAtNodes (nodes : ℕ) : Prop := nodes = numFlies

-- Define the predicate for whether the spider can eat all flies within a certain number of moves.
def canEatAllFliesWithinMoves (maxMoves : ℕ) : Prop :=
  ∃ (moves : ℕ), moves ≤ maxMoves

-- The theorem we need to prove in Lean 4.
theorem spiderCanEatAllFlies (h1 : cornerStart) (h2 : fliesAtNodes numFlies) : canEatAllFliesWithinMoves 2000 :=
by
  sorry

end spiderCanEatAllFlies_l148_148225


namespace probability_diagonals_intersect_l148_148770

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l148_148770


namespace B_work_days_l148_148279

-- Define work rates and conditions
def A_work_rate : ℚ := 1 / 18
def B_work_rate : ℚ := 1 / 15
def A_days_after_B_left : ℚ := 6
def total_work : ℚ := 1

-- Theorem statement
theorem B_work_days : ∃ x : ℚ, (x * B_work_rate + A_days_after_B_left * A_work_rate = total_work) → x = 10 := by
  sorry

end B_work_days_l148_148279


namespace quadratic_equation_standard_form_quadratic_equation_coefficients_l148_148052

theorem quadratic_equation_standard_form : 
  ∀ (x : ℝ), (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (a = 2 ∧ b = -6 ∧ c = -1) :=
by
  sorry

end quadratic_equation_standard_form_quadratic_equation_coefficients_l148_148052


namespace find_b_in_triangle_l148_148293

theorem find_b_in_triangle
  (a b c A B C : ℝ)
  (cos_A : ℝ) (cos_C : ℝ)
  (ha : a = 1)
  (hcos_A : cos_A = 4 / 5)
  (hcos_C : cos_C = 5 / 13) :
  b = 21 / 13 :=
by
  sorry

end find_b_in_triangle_l148_148293


namespace ratio_joe_sara_l148_148818

variables (S J : ℕ) (k : ℕ)

-- Conditions
#check J + S = 120
#check J = k * S + 6
#check J = 82

-- The goal is to prove the ratio J / S = 41 / 19
theorem ratio_joe_sara (h1 : J + S = 120) (h2 : J = k * S + 6) (h3 : J = 82) : J / S = 41 / 19 :=
sorry

end ratio_joe_sara_l148_148818


namespace evaluate_expression_l148_148621

variable (a : ℕ)

theorem evaluate_expression (h : a = 2) : a^3 * a^4 = 128 :=
by
  sorry

end evaluate_expression_l148_148621


namespace problem_b_problem_d_l148_148837

variable (x y t : ℝ)

def condition_curve (t : ℝ) : Prop :=
  ∃ C : ℝ × ℝ → Prop, ∀ x y : ℝ, C (x, y) ↔ (x^2 / (5 - t) + y^2 / (t - 1) = 1)

theorem problem_b (h1 : t < 1) : condition_curve t → ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → ¬(5 - t) < 0 ∧ (t - 1) < 0 := 
sorry

theorem problem_d (h1 : 3 < t) (h2 : t < 5) (h3 : condition_curve t) : ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → 0 < (t - 1) ∧ (t - 1) > (5 - t) := 
sorry

end problem_b_problem_d_l148_148837


namespace jake_watching_hours_l148_148604

theorem jake_watching_hours
    (monday_hours : ℕ := 12) -- Half of 24 hours in a day is 12 hours for Monday
    (wednesday_hours : ℕ := 6) -- A quarter of 24 hours in a day is 6 hours for Wednesday
    (friday_hours : ℕ := 19) -- Jake watched 19 hours on Friday
    (total_hours : ℕ := 52) -- The entire show is 52 hours long
    (T : ℕ) -- To find the total number of hours on Tuesday
    (h : monday_hours + T + wednesday_hours + (monday_hours + T + wednesday_hours) / 2 + friday_hours = total_hours) :
    T = 4 := sorry

end jake_watching_hours_l148_148604


namespace jerry_reaches_3_at_some_time_l148_148938

def jerry_reaches_3_probability (n : ℕ) (k : ℕ) : ℚ :=
  -- This function represents the probability that Jerry reaches 3 at some point during n coin tosses
  if n = 7 ∧ k = 3 then (21 / 64 : ℚ) else 0

theorem jerry_reaches_3_at_some_time :
  jerry_reaches_3_probability 7 3 = (21 / 64 : ℚ) :=
sorry

end jerry_reaches_3_at_some_time_l148_148938


namespace find_a1_l148_148684

theorem find_a1 (a : ℕ → ℝ) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) (h_init : a 3 = 1 / 5) : a 1 = 1 := by
  sorry

end find_a1_l148_148684


namespace dave_tray_problem_l148_148963

theorem dave_tray_problem (n_trays_per_trip : ℕ) (n_trips : ℕ) (n_second_table : ℕ) : 
  (n_trays_per_trip = 9) → (n_trips = 8) → (n_second_table = 55) → 
  (n_trays_per_trip * n_trips - n_second_table = 17) :=
by
  sorry

end dave_tray_problem_l148_148963


namespace area_ratio_equilateral_triangle_extension_l148_148825

variable (s : ℝ)

theorem area_ratio_equilateral_triangle_extension :
  (let A := (0, 0)
   let B := (s, 0)
   let C := (s / 2, s * (Real.sqrt 3 / 2))
   let A' := (0, -4 * s * (Real.sqrt 3 / 2))
   let B' := (3 * s, 0)
   let C' := (s / 2, s * (Real.sqrt 3 / 2) + 3 * s * (Real.sqrt 3 / 2))
   let area_ABC := (Real.sqrt 3 / 4) * s^2
   let area_A'B'C' := (Real.sqrt 3 / 4) * 60 * s^2
   area_A'B'C' / area_ABC = 60) :=
sorry

end area_ratio_equilateral_triangle_extension_l148_148825


namespace sufficient_not_necessary_condition_l148_148845

noncomputable def sufficient_but_not_necessary (x y : ℝ) : Prop :=
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ (x + y > 2 → ¬(x > 1 ∧ y > 1))

theorem sufficient_not_necessary_condition (x y : ℝ) :
  sufficient_but_not_necessary x y :=
sorry

end sufficient_not_necessary_condition_l148_148845


namespace part_a_region_part_b_region_part_c_region_l148_148732

-- Definitions for Part (a)
def surface1a (x y z : ℝ) := 2 * y = x ^ 2 + z ^ 2
def surface2a (x y z : ℝ) := x ^ 2 + z ^ 2 = 1
def region_a (x y z : ℝ) := surface1a x y z ∧ surface2a x y z

-- Definitions for Part (b)
def surface1b (x y z : ℝ) := z = 0
def surface2b (x y z : ℝ) := y + z = 2
def surface3b (x y z : ℝ) := y = x ^ 2
def region_b (x y z : ℝ) := surface1b x y z ∧ surface2b x y z ∧ surface3b x y z

-- Definitions for Part (c)
def surface1c (x y z : ℝ) := z = 6 - x ^ 2 - y ^ 2
def surface2c (x y z : ℝ) := x ^ 2 + y ^ 2 = z ^ 2
def region_c (x y z : ℝ) := surface1c x y z ∧ surface2c x y z

-- The formal theorem statements
theorem part_a_region : ∃x y z : ℝ, region_a x y z := by
  sorry

theorem part_b_region : ∃x y z : ℝ, region_b x y z := by
  sorry

theorem part_c_region : ∃x y z : ℝ, region_c x y z := by
  sorry

end part_a_region_part_b_region_part_c_region_l148_148732


namespace remainder_when_divided_l148_148801

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l148_148801


namespace evaluate_y_correct_l148_148103

noncomputable def evaluate_y (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - 4 * x + 4) + Real.sqrt (x^2 + 6 * x + 9) - 2

theorem evaluate_y_correct (x : ℝ) : 
  evaluate_y x = |x - 2| + |x + 3| - 2 :=
by 
  sorry

end evaluate_y_correct_l148_148103


namespace largest_among_five_numbers_l148_148894

theorem largest_among_five_numbers :
  max (max (max (max (12345 + 1 / 3579) 
                       (12345 - 1 / 3579))
                   (12345 ^ (1 / 3579)))
               (12345 / (1 / 3579)))
           12345.3579 = 12345 / (1 / 3579) := sorry

end largest_among_five_numbers_l148_148894


namespace abs_algebraic_expression_l148_148219

theorem abs_algebraic_expression (x : ℝ) (h : |2 * x - 3| - 3 + 2 * x = 0) : |2 * x - 5| = 5 - 2 * x := 
by sorry

end abs_algebraic_expression_l148_148219


namespace discount_percentage_l148_148155

theorem discount_percentage (discount amount_paid : ℝ) (h_discount : discount = 40) (h_paid : amount_paid = 120) : 
  (discount / (discount + amount_paid)) * 100 = 25 := by
  sorry

end discount_percentage_l148_148155


namespace distance_between_trees_l148_148438

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 255) (h2 : num_trees = 18) : yard_length / (num_trees - 1) = 15 := by
  sorry

end distance_between_trees_l148_148438


namespace negation_universal_proposition_l148_148273

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, |x| + x^4 ≥ 0) ↔ ∃ x₀ : ℝ, |x₀| + x₀^4 < 0 :=
by
  sorry

end negation_universal_proposition_l148_148273


namespace sin_square_general_proposition_l148_148945

-- Definitions for the given conditions
def sin_square_sum_30_90_150 : Prop :=
  (Real.sin (30 * Real.pi / 180))^2 + (Real.sin (90 * Real.pi / 180))^2 + (Real.sin (150 * Real.pi / 180))^2 = 3/2

def sin_square_sum_5_65_125 : Prop :=
  (Real.sin (5 * Real.pi / 180))^2 + (Real.sin (65 * Real.pi / 180))^2 + (Real.sin (125 * Real.pi / 180))^2 = 3/2

-- The general proposition we want to prove
theorem sin_square_general_proposition (α : ℝ) : 
  sin_square_sum_30_90_150 ∧ sin_square_sum_5_65_125 →
  (Real.sin (α * Real.pi / 180 - 60 * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180 + 60 * Real.pi / 180))^2 = 3/2 :=
by
  intro h
  -- Proof goes here
  sorry

end sin_square_general_proposition_l148_148945


namespace range_of_x_when_m_eq_4_range_of_m_given_conditions_l148_148056

-- Definitions of p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Question 1: Given m = 4 and conditions p ∧ q being true, prove the range of x is 4 < x < 5
theorem range_of_x_when_m_eq_4 (x m : ℝ) (h_m : m = 4) (h : p x ∧ q x m) : 4 < x ∧ x < 5 := 
by
  sorry

-- Question 2: Given conditions ⟪¬q ⟫is a sufficient but not necessary condition for ⟪¬p ⟫and constraints, prove the range of m is 5/3 ≤ m ≤ 2
theorem range_of_m_given_conditions (m : ℝ) (h_sufficient : ∀ (x : ℝ), ¬q x m → ¬p x) (h_constraints : m > 0) : 5 / 3 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_x_when_m_eq_4_range_of_m_given_conditions_l148_148056


namespace length_GH_of_tetrahedron_l148_148467

noncomputable def tetrahedron_edge_length : ℕ := 24

theorem length_GH_of_tetrahedron
  (a b c d e f : ℕ)
  (h1 : a = 8) 
  (h2 : b = 16) 
  (h3 : c = 24) 
  (h4 : d = 35) 
  (h5 : e = 45) 
  (h6 : f = 55)
  (hEF : f = 55)
  (hEGF : e + b > f)
  (hEHG: e + c > a ∧ e + c > d) 
  (hFHG : b + c > a ∧ b + f > c ∧ c + a > b):
   tetrahedron_edge_length = c := 
sorry

end length_GH_of_tetrahedron_l148_148467


namespace binomial_12_10_eq_66_l148_148620

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l148_148620


namespace part1_tangent_circles_part2_chords_l148_148895

theorem part1_tangent_circles (t : ℝ) : 
  t = 1 → 
  ∃ (a b : ℝ), 
    (x + 1)^2 + y^2 = 1 ∨ 
    (x + (2/5))^2 + (y - (9/5))^2 = (1 : ℝ) :=
by
  sorry

theorem part2_chords (t : ℝ) : 
  (∀ (k1 k2 : ℝ), 
    k1 + k2 = -3 * t / 4 ∧ 
    k1 * k2 = (t^2 - 1) / 8 ∧ 
    |k1 - k2| = 3 / 4) → 
    t = 1 ∨ t = -1 :=
by
  sorry

end part1_tangent_circles_part2_chords_l148_148895


namespace anne_speed_ratio_l148_148188

theorem anne_speed_ratio (B A A' : ℝ) (h_A : A = 1/12) (h_together_current : (B + A) * 4 = 1) (h_together_new : (B + A') * 3 = 1) :
  A' / A = 2 := 
by
  sorry

end anne_speed_ratio_l148_148188


namespace range_of_k_l148_148325

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*y^2 = 2 ∧ 
  (∀ e : ℝ, (x^2 / 2 + y^2 / (2 / e) = 1 → (2 / e) > 2))) → 
  0 < k ∧ k < 1 :=
by 
sorry

end range_of_k_l148_148325


namespace range_of_a_decreasing_function_l148_148233

theorem range_of_a_decreasing_function (a : ℝ) :
  (∀ x < 1, ∀ y < x, (3 * a - 1) * x + 4 * a ≥ (3 * a - 1) * y + 4 * a) ∧ 
  (∀ x ≥ 1, ∀ y > x, -a * x ≤ -a * y) ∧
  (∀ x < 1, ∀ y ≥ 1, (3 * a - 1) * x + 4 * a ≥ -a * y)  →
  (1 / 8 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
sorry

end range_of_a_decreasing_function_l148_148233


namespace centrally_symmetric_equidecomposable_l148_148704

-- Assume we have a type for Polyhedra
variable (Polyhedron : Type)

-- Conditions
variable (sameVolume : Polyhedron → Polyhedron → Prop)
variable (centrallySymmetricFaces : Polyhedron → Prop)
variable (equidecomposable : Polyhedron → Polyhedron → Prop)

-- Theorem statement
theorem centrally_symmetric_equidecomposable 
  (P Q : Polyhedron) 
  (h1 : sameVolume P Q) 
  (h2 : centrallySymmetricFaces P) 
  (h3 : centrallySymmetricFaces Q) :
  equidecomposable P Q := 
sorry

end centrally_symmetric_equidecomposable_l148_148704


namespace sequence_sum_l148_148139

-- Assume the sum of first n terms of the sequence {a_n} is given by S_n = n^2 + n + 1
def S (n : ℕ) : ℕ := n^2 + n + 1

-- The sequence a_8 + a_9 + a_10 + a_11 + a_12 is what we want to prove equals 100.
theorem sequence_sum : S 12 - S 7 = 100 :=
by
  sorry

end sequence_sum_l148_148139


namespace smallest_angle_of_triangle_l148_148246

theorem smallest_angle_of_triangle (y : ℝ) (h : 40 + 70 + y = 180) : 
  ∃ smallest_angle : ℝ, smallest_angle = 40 ∧ smallest_angle = min 40 (min 70 y) := 
by
  use 40
  sorry

end smallest_angle_of_triangle_l148_148246


namespace solve_system_of_equations_l148_148611

theorem solve_system_of_equations : ∃ x y : ℤ, 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 ∧ x = 4 ∧ y = 3 :=
by
  sorry

end solve_system_of_equations_l148_148611


namespace a_in_range_l148_148092

noncomputable def kOM (t : ℝ) : ℝ := (Real.log t) / t
noncomputable def kON (a t : ℝ) : ℝ := (a + a * t - t^2) / t

theorem a_in_range (a : ℝ) : 
  (∀ t ∈ Set.Ici 1, 0 ≤ (1 - Real.log t + a) / t^2 + 1) →
  a ∈ Set.Ici (-2) := 
by
  sorry

end a_in_range_l148_148092


namespace area_inside_Z_outside_X_l148_148743

structure Circle :=
  (center : Real × Real)
  (radius : ℝ)

def tangent (A B : Circle) : Prop :=
  dist A.center B.center = A.radius + B.radius

theorem area_inside_Z_outside_X (X Y Z : Circle)
  (hX : X.radius = 1) 
  (hY : Y.radius = 1) 
  (hZ : Z.radius = 1)
  (tangent_XY : tangent X Y)
  (tangent_XZ : tangent X Z)
  (non_intersect_YZ : dist Z.center Y.center > Z.radius + Y.radius) :
  π - 1/2 * π = 1/2 * π := 
by
  sorry

end area_inside_Z_outside_X_l148_148743


namespace min_fraction_sum_is_15_l148_148370

theorem min_fraction_sum_is_15
  (A B C D : ℕ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
  (h_nonzero_int : ∃ k : ℤ, k ≠ 0 ∧ (A + B : ℤ) = k * (C + D))
  : C + D = 15 :=
sorry

end min_fraction_sum_is_15_l148_148370


namespace product_of_solutions_l148_148578

theorem product_of_solutions (x : ℝ) :
  ∃ (α β : ℝ), (x^2 - 4*x - 21 = 0) ∧ α * β = -21 := sorry

end product_of_solutions_l148_148578


namespace total_attendance_l148_148741

theorem total_attendance (A C : ℕ) (adult_ticket_price child_ticket_price total_revenue : ℕ) 
(h1 : adult_ticket_price = 11) (h2 : child_ticket_price = 10) (h3 : total_revenue = 246) 
(h4 : C = 7) (h5 : adult_ticket_price * A + child_ticket_price * C = total_revenue) : 
A + C = 23 :=
by {
  sorry
}

end total_attendance_l148_148741


namespace min_value_of_expression_l148_148623

theorem min_value_of_expression (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (hpar : ∀ x y : ℝ, 2 * x + (n - 1) * y - 2 = 0 → ∃ c : ℝ, mx + ny + c = 0) :
  2 * m + n = 9 :=
by
  sorry

end min_value_of_expression_l148_148623


namespace unique_function_satisfies_sum_zero_l148_148126

theorem unique_function_satisfies_sum_zero 
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x^3) = (f x)^3)
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2) : 
  f 0 + f 1 + f (-1) = 0 :=
sorry

end unique_function_satisfies_sum_zero_l148_148126


namespace solve_inequality_l148_148013

theorem solve_inequality (x : Real) : 
  x^2 - 48 * x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 :=
by
  sorry

end solve_inequality_l148_148013


namespace binary_arithmetic_l148_148058

-- Define the binary numbers 11010_2, 11100_2, and 100_2
def x : ℕ := 0b11010 -- base 2 number 11010 in base 10 representation
def y : ℕ := 0b11100 -- base 2 number 11100 in base 10 representation
def d : ℕ := 0b100   -- base 2 number 100 in base 10 representation

-- Define the correct answer
def correct_answer : ℕ := 0b10101101 -- base 2 number 10101101 in base 10 representation

-- The proof problem statement
theorem binary_arithmetic : (x * y) / d = correct_answer := by
  sorry

end binary_arithmetic_l148_148058


namespace domain_of_function_l148_148900

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ x + 2 ≠ 0} = {x : ℝ | x ≥ -3 ∧ x ≠ -2} :=
by
  sorry

end domain_of_function_l148_148900


namespace systematic_sampling_result_l148_148330

theorem systematic_sampling_result :
  ∀ (total_students sample_size selected1_16 selected33_48 : ℕ),
  total_students = 800 →
  sample_size = 50 →
  selected1_16 = 11 →
  selected33_48 = selected1_16 + 32 →
  selected33_48 = 43 := by
  intros
  sorry

end systematic_sampling_result_l148_148330


namespace john_spending_l148_148867

variable (initial_cost : ℕ) (sale_price : ℕ) (new_card_cost : ℕ)

theorem john_spending (h1 : initial_cost = 1200) (h2 : sale_price = 300) (h3 : new_card_cost = 500) :
  initial_cost - sale_price + new_card_cost = 1400 := 
by
  sorry

end john_spending_l148_148867


namespace probability_fly_reaches_8_10_l148_148795

theorem probability_fly_reaches_8_10 :
  let total_steps := 2^18
  let right_up_combinations := Nat.choose 18 8
  (right_up_combinations / total_steps : ℚ) = Nat.choose 18 8 / 2^18 := 
sorry

end probability_fly_reaches_8_10_l148_148795


namespace angle_conversion_l148_148695

theorem angle_conversion : (1 : ℝ) * (π / 180) * (-225) = - (5 * π / 4) :=
by
  sorry

end angle_conversion_l148_148695


namespace general_term_l148_148920

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n / (2 + a n)

theorem general_term (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n > 0 → a n = 2 / (n + 1) :=
by
sorry

end general_term_l148_148920


namespace true_statements_count_is_two_l148_148596

def original_proposition (a : ℝ) : Prop :=
  a < 0 → ∃ x : ℝ, x^2 + x + a = 0

def contrapositive (a : ℝ) : Prop :=
  ¬ (∃ x : ℝ, x^2 + x + a = 0) → a ≥ 0

def converse (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 + x + a = 0) → a < 0

def negation (a : ℝ) : Prop :=
  a < 0 → ¬ ∃ x : ℝ, x^2 + x + a = 0

-- Prove that there are exactly 2 true statements among the four propositions: 
-- original_proposition, contrapositive, converse, and negation.

theorem true_statements_count_is_two : 
  ∀ (a : ℝ), original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a) → 
  (original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a)) ↔ (2 = 2) := 
by
  sorry

end true_statements_count_is_two_l148_148596


namespace option_D_not_necessarily_true_l148_148614

variable {a b c : ℝ}

theorem option_D_not_necessarily_true 
  (h1 : c < b)
  (h2 : b < a)
  (h3 : a * c < 0) : ¬((c * b^2 < a * b^2) ↔ (b ≠ 0 ∨ b = 0 ∧ (c * b^2 < a * b^2))) := 
sorry

end option_D_not_necessarily_true_l148_148614


namespace cot_half_angle_product_geq_3sqrt3_l148_148980

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_half_angle_product_geq_3sqrt3 {A B C : ℝ} (h : A + B + C = π) :
    cot (A / 2) * cot (B / 2) * cot (C / 2) ≥ 3 * Real.sqrt 3 := 
  sorry

end cot_half_angle_product_geq_3sqrt3_l148_148980


namespace quadruplets_satisfy_l148_148987

-- Define the condition in the problem
def equation (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

-- State the theorem
theorem quadruplets_satisfy (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  equation x y z w ↔ (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) :=
by
  sorry

end quadruplets_satisfy_l148_148987


namespace hyperbola_eccentricity_l148_148628

theorem hyperbola_eccentricity (a b c : ℚ) (h1 : (c : ℚ) = 5)
  (h2 : (b / a) = 3 / 4) (h3 : c^2 = a^2 + b^2) :
  (c / a : ℚ) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l148_148628


namespace bodhi_yacht_animals_l148_148882

def total_animals (cows foxes zebras sheep : ℕ) : ℕ :=
  cows + foxes + zebras + sheep

theorem bodhi_yacht_animals :
  ∀ (cows foxes sheep : ℕ), foxes = 15 → cows = 20 → sheep = 20 → total_animals cows foxes (3 * foxes) sheep = 100 :=
by
  intros cows foxes sheep h1 h2 h3
  rw [h1, h2, h3]
  show total_animals 20 15 (3 * 15) 20 = 100
  sorry

end bodhi_yacht_animals_l148_148882


namespace games_draw_fraction_l148_148063

-- Definitions from the conditions in the problems
def ben_win_fraction : ℚ := 4 / 9
def tom_win_fraction : ℚ := 1 / 3

-- The theorem we want to prove
theorem games_draw_fraction : 1 - (ben_win_fraction + (1 / 3)) = 2 / 9 := by
  sorry

end games_draw_fraction_l148_148063


namespace polygon_perimeter_l148_148116

theorem polygon_perimeter (a b : ℕ) (h : adjacent_sides_perpendicular) :
  perimeter = 2 * (a + b) :=
sorry

end polygon_perimeter_l148_148116


namespace Carol_weight_equals_nine_l148_148366

-- conditions in Lean definitions
def Mildred_weight : ℤ := 59
def weight_difference : ℤ := 50

-- problem statement to prove in Lean 4
theorem Carol_weight_equals_nine (Carol_weight : ℤ) :
  Mildred_weight = Carol_weight + weight_difference → Carol_weight = 9 :=
by
  sorry

end Carol_weight_equals_nine_l148_148366


namespace total_campers_correct_l148_148773

-- Definitions for the conditions
def campers_morning : ℕ := 15
def campers_afternoon : ℕ := 17

-- Define total campers, question is to prove it is indeed 32
def total_campers : ℕ := campers_morning + campers_afternoon

theorem total_campers_correct : total_campers = 32 :=
by
  -- Proof omitted
  sorry

end total_campers_correct_l148_148773


namespace max_value_expression_l148_148158

theorem max_value_expression (x : ℝ) : 
  ∃ m : ℝ, m = 1 / 37 ∧ ∀ x : ℝ, (x^6) / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ m :=
sorry

end max_value_expression_l148_148158


namespace even_sum_probability_l148_148772

theorem even_sum_probability :
  let p_even_w1 := 3 / 4
  let p_even_w2 := 1 / 2
  let p_even_w3 := 1 / 4
  let p_odd_w1 := 1 - p_even_w1
  let p_odd_w2 := 1 - p_even_w2
  let p_odd_w3 := 1 - p_even_w3
  (p_even_w1 * p_even_w2 * p_even_w3) +
  (p_odd_w1 * p_odd_w2 * p_even_w3) +
  (p_odd_w1 * p_even_w2 * p_odd_w3) +
  (p_even_w1 * p_odd_w2 * p_odd_w3) = 1 / 2 := by
    sorry

end even_sum_probability_l148_148772


namespace inscribed_circle_radius_l148_148659

variable (A p s r : ℝ)

-- Condition: Area is twice the perimeter
def twice_perimeter_condition : Prop := A = 2 * p

-- Condition: The formula connecting the area, inradius, and semiperimeter
def area_inradius_semiperimeter_relation : Prop := A = r * s

-- Condition: The perimeter is twice the semiperimeter
def perimeter_semiperimeter_relation : Prop := p = 2 * s

-- Prove the radius of the inscribed circle is 4
theorem inscribed_circle_radius (h1 : twice_perimeter_condition A p)
                                (h2 : area_inradius_semiperimeter_relation A r s)
                                (h3 : perimeter_semiperimeter_relation p s) :
  r = 4 :=
by
  sorry

end inscribed_circle_radius_l148_148659


namespace line_equation_unique_l148_148356

theorem line_equation_unique (m b k : ℝ) (h_intersect_dist : |(k^2 + 6*k + 5) - (m*k + b)| = 7)
  (h_passing_point : 8 = 2*m + b) (hb_nonzero : b ≠ 0) :
  y = 10*x - 12 :=
by
  sorry

end line_equation_unique_l148_148356


namespace num_cells_after_10_moves_l148_148559

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l148_148559


namespace remainder_when_divided_by_9_l148_148857

open Nat

theorem remainder_when_divided_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 :=
by
  sorry

end remainder_when_divided_by_9_l148_148857


namespace part1_part2_i_part2_ii_l148_148989

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x * Real.log x - 1

theorem part1 (a : ℝ) (x : ℝ) : f x a + x^2 * f (1 / x) a = 0 :=
by sorry

theorem part2_i (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : 2 < a :=
by sorry

theorem part2_ii (a : ℝ) (x1 x2 x3 : ℝ) (h : x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : x1 + x3 > 2 * a - 2 :=
by sorry

end part1_part2_i_part2_ii_l148_148989


namespace transport_tax_to_be_paid_l148_148696

noncomputable def engine_power : ℕ := 150
noncomputable def tax_rate : ℕ := 20
noncomputable def annual_tax : ℕ := engine_power * tax_rate
noncomputable def months_used : ℕ := 8
noncomputable def prorated_tax : ℕ := (months_used * annual_tax) / 12

theorem transport_tax_to_be_paid : prorated_tax = 2000 := 
by 
  -- sorry is used to skip the proof step
  sorry

end transport_tax_to_be_paid_l148_148696


namespace jog_time_each_morning_is_1_5_hours_l148_148263

-- Define the total time Mr. John spent jogging
def total_time_spent_jogging : ℝ := 21

-- Define the number of days Mr. John jogged
def number_of_days_jogged : ℕ := 14

-- Define the time Mr. John jogs each morning
noncomputable def time_jogged_each_morning : ℝ := total_time_spent_jogging / number_of_days_jogged

-- State the theorem that the time jogged each morning is 1.5 hours
theorem jog_time_each_morning_is_1_5_hours : time_jogged_each_morning = 1.5 := by
  sorry

end jog_time_each_morning_is_1_5_hours_l148_148263


namespace Kim_morning_routine_time_l148_148888

theorem Kim_morning_routine_time :
  let senior_employees := 3
  let junior_employees := 3
  let interns := 3

  let senior_overtime := 2
  let junior_overtime := 3
  let intern_overtime := 1
  let senior_not_overtime := senior_employees - senior_overtime
  let junior_not_overtime := junior_employees - junior_overtime
  let intern_not_overtime := interns - intern_overtime

  let coffee_time := 5
  let email_time := 10
  let supplies_time := 8
  let meetings_time := 6
  let reports_time := 5

  let status_update_time := 3 * senior_employees + 2 * junior_employees + 1 * interns
  let payroll_update_time := 
    4 * senior_overtime + 2 * senior_not_overtime +
    3 * junior_overtime + 1 * junior_not_overtime +
    2 * intern_overtime + 0.5 * intern_not_overtime
  let daily_tasks_time :=
    4 * senior_employees + 3 * junior_employees + 2 * interns

  let total_time := coffee_time + status_update_time + payroll_update_time + daily_tasks_time + email_time + supplies_time + meetings_time + reports_time
  total_time = 101 := by
  sorry

end Kim_morning_routine_time_l148_148888


namespace sum_of_coefficients_eq_39_l148_148239

theorem sum_of_coefficients_eq_39 :
  5 * (2 * 1^8 - 3 * 1^3 + 4) - 6 * (1^6 + 4 * 1^3 - 9) = 39 :=
by
  sorry

end sum_of_coefficients_eq_39_l148_148239


namespace spent_on_basil_seeds_l148_148786

-- Define the variables and conditions
variables (S cost_soil num_plants price_per_plant net_profit total_revenue total_expenses : ℝ)
variables (h1 : cost_soil = 8)
variables (h2 : num_plants = 20)
variables (h3 : price_per_plant = 5)
variables (h4 : net_profit = 90)

-- Definition of total revenue as the multiplication of number of plants and price per plant
def revenue_eq : Prop := total_revenue = num_plants * price_per_plant

-- Definition of total expenses as the sum of soil cost and cost of basil seeds
def expenses_eq : Prop := total_expenses = cost_soil + S

-- Definition of net profit
def profit_eq : Prop := net_profit = total_revenue - total_expenses

-- The theorem to prove
theorem spent_on_basil_seeds : S = 2 :=
by
  -- Since we define variables and conditions as inputs,
  -- the proof itself is omitted as per instructions
  sorry

end spent_on_basil_seeds_l148_148786


namespace gcd_factorial_eight_nine_eq_8_factorial_l148_148344

theorem gcd_factorial_eight_nine_eq_8_factorial : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := 
by 
  sorry

end gcd_factorial_eight_nine_eq_8_factorial_l148_148344


namespace width_of_wall_is_6_l148_148526

-- Definitions of the conditions given in the problem
def height_of_wall (w : ℝ) := 4 * w
def length_of_wall (h : ℝ) := 3 * h
def volume_of_wall (w h l : ℝ) := w * h * l

-- Proof statement that the width of the wall is 6 meters given the conditions
theorem width_of_wall_is_6 :
  ∃ w : ℝ, 
  (height_of_wall w = 4 * w) ∧ 
  (length_of_wall (height_of_wall w) = 3 * (height_of_wall w)) ∧ 
  (volume_of_wall w (height_of_wall w) (length_of_wall (height_of_wall w)) = 10368) ∧ 
  (w = 6) :=
sorry

end width_of_wall_is_6_l148_148526


namespace chickens_count_l148_148396

-- Define conditions
def cows : Nat := 4
def sheep : Nat := 3
def bushels_per_cow : Nat := 2
def bushels_per_sheep : Nat := 2
def bushels_per_chicken : Nat := 3
def total_bushels_needed : Nat := 35

-- The main theorem to be proven
theorem chickens_count : 
  (total_bushels_needed - ((cows * bushels_per_cow) + (sheep * bushels_per_sheep))) / bushels_per_chicken = 7 :=
by
  sorry

end chickens_count_l148_148396


namespace students_average_vegetables_l148_148714

variable (points_needed : ℕ) (points_per_vegetable : ℕ) (students : ℕ) (school_days : ℕ) (school_weeks : ℕ)

def average_vegetables_per_student_per_week (points_needed points_per_vegetable students school_days school_weeks : ℕ) : ℕ :=
  let total_vegetables := points_needed / points_per_vegetable
  let vegetables_per_student := total_vegetables / students
  vegetables_per_student / school_weeks

theorem students_average_vegetables 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : students = 25) 
  (h4 : school_days = 10) 
  (h5 : school_weeks = 2) : 
  average_vegetables_per_student_per_week points_needed points_per_vegetable students school_days school_weeks = 2 :=
by
  sorry

end students_average_vegetables_l148_148714


namespace cos_pi_minus_double_alpha_l148_148290

theorem cos_pi_minus_double_alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_double_alpha_l148_148290


namespace solve_equation_l148_148981

theorem solve_equation (x : ℝ) (h₁ : x ≠ -11) (h₂ : x ≠ -5) (h₃ : x ≠ -12) (h₄ : x ≠ -4) :
  (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ↔ x = -8 :=
by
  sorry

end solve_equation_l148_148981


namespace find_fraction_l148_148899

noncomputable def fraction_of_eighths (N : ℝ) (a b : ℝ) : Prop :=
  (3/8) * N * (a/b) = 24

noncomputable def two_fifty_percent (N : ℝ) : Prop :=
  2.5 * N = 199.99999999999997

theorem find_fraction {N a b : ℝ} (h1 : fraction_of_eighths N a b) (h2 : two_fifty_percent N) :
  a/b = 4/5 :=
sorry

end find_fraction_l148_148899


namespace max_alpha_value_l148_148669

variable (a b x y α : ℝ)

theorem max_alpha_value (h1 : a = 2 * b)
    (h2 : a^2 + y^2 = b^2 + x^2)
    (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
    (h4 : 0 ≤ x) (h5 : x < a) (h6 : 0 ≤ y) (h7 : y < b) :
    α = a / b → α^2 = 4 := 
by
  sorry

end max_alpha_value_l148_148669


namespace find_a_if_lines_perpendicular_l148_148929

theorem find_a_if_lines_perpendicular (a : ℝ) :
  (∀ x, (y1 : ℝ) = a * x - 2 → (y2 : ℝ) = (a + 2) * x + 1 → y1 * y2 = -1) → a = -1 :=
by {
  sorry
}

end find_a_if_lines_perpendicular_l148_148929


namespace inequality_proof_l148_148269

variable (a b c d : ℝ)
variable (habcda : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ ab + bc + cd + da = 1)

theorem inequality_proof :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ (ab + bc + cd + da = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by sorry

end inequality_proof_l148_148269


namespace asia_paid_140_l148_148187

noncomputable def original_price : ℝ := 350
noncomputable def discount_percentage : ℝ := 0.60
noncomputable def discount_amount : ℝ := original_price * discount_percentage
noncomputable def final_price : ℝ := original_price - discount_amount

theorem asia_paid_140 : final_price = 140 := by
  unfold final_price
  unfold discount_amount
  unfold original_price
  unfold discount_percentage
  sorry

end asia_paid_140_l148_148187


namespace inequality_xyz_l148_148178

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l148_148178


namespace price_reduction_is_50_rubles_l148_148877

theorem price_reduction_is_50_rubles :
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  P_Feb - P_Mar = 50 :=
by
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  sorry

end price_reduction_is_50_rubles_l148_148877


namespace total_accidents_l148_148897

-- Define the given vehicle counts for the highways
def total_vehicles_A : ℕ := 4 * 10^9
def total_vehicles_B : ℕ := 2 * 10^9
def total_vehicles_C : ℕ := 1 * 10^9

-- Define the accident ratios per highway
def accident_ratio_A : ℕ := 80
def accident_ratio_B : ℕ := 120
def accident_ratio_C : ℕ := 65

-- Define the number of vehicles in millions
def million := 10^6

-- Define the accident calculations per highway
def accidents_A : ℕ := (total_vehicles_A / (100 * million)) * accident_ratio_A
def accidents_B : ℕ := (total_vehicles_B / (200 * million)) * accident_ratio_B
def accidents_C : ℕ := (total_vehicles_C / (50 * million)) * accident_ratio_C

-- Prove the total number of accidents across all highways
theorem total_accidents : accidents_A + accidents_B + accidents_C = 5700 := by
  have : accidents_A = 3200 := by sorry
  have : accidents_B = 1200 := by sorry
  have : accidents_C = 1300 := by sorry
  sorry

end total_accidents_l148_148897


namespace smallest_multiple_5_711_l148_148085

theorem smallest_multiple_5_711 : ∃ n : ℕ, n = Nat.lcm 5 711 ∧ n = 3555 := 
by
  sorry

end smallest_multiple_5_711_l148_148085


namespace cloves_needed_l148_148423

theorem cloves_needed (cv_fp : 3 / 2 = 1.5) (cw_fp : 3 / 3 = 1) (vc_fp : 3 / 8 = 0.375) : 
  let cloves_for_vampires := 45
  let cloves_for_wights := 12
  let cloves_for_bats := 15
  30 * (3 / 2) + 12 * (3 / 3) + 40 * (3 / 8) = 72 := by
  sorry

end cloves_needed_l148_148423


namespace output_y_for_x_eq_5_l148_148096

def compute_y (x : Int) : Int :=
  if x > 0 then 3 * x + 1 else -2 * x + 3

theorem output_y_for_x_eq_5 : compute_y 5 = 16 := by
  sorry

end output_y_for_x_eq_5_l148_148096


namespace seven_lines_regions_l148_148662

theorem seven_lines_regions (n : ℕ) (hn : n = 7) (h1 : ¬ ∃ l1 l2 : ℝ, l1 = l2) (h2 : ∀ l1 l2 l3 : ℝ, ¬ (l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ (l1 = l2 ∧ l2 = l3))) :
  ∃ R : ℕ, R = 29 :=
by
  sorry

end seven_lines_regions_l148_148662


namespace n_mod_9_eq_6_l148_148767

def n : ℕ := 2 + 333 + 44444 + 555555 + 6666666 + 77777777 + 888888888 + 9999999999

theorem n_mod_9_eq_6 : n % 9 = 6 :=
by
  sorry

end n_mod_9_eq_6_l148_148767


namespace product_PA_PB_eq_nine_l148_148914

theorem product_PA_PB_eq_nine 
  (P A B : ℝ × ℝ) 
  (hP : P = (3, 1)) 
  (h1 : A ≠ B)
  (h2 : ∃ L : ℝ × ℝ → Prop, L P ∧ L A ∧ L B) 
  (h3 : A.fst ^ 2 + A.snd ^ 2 = 1) 
  (h4 : B.fst ^ 2 + B.snd ^ 2 = 1) : 
  |((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)| * |((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2)| = 9 := 
sorry

end product_PA_PB_eq_nine_l148_148914


namespace fraction_simplifiable_by_7_l148_148736

theorem fraction_simplifiable_by_7 (a b c : ℤ) (h : (100 * a + 10 * b + c) % 7 = 0) : 
  ((10 * b + c + 16 * a) % 7 = 0) ∧ ((10 * b + c - 61 * a) % 7 = 0) :=
by
  sorry

end fraction_simplifiable_by_7_l148_148736


namespace thirteen_pow_seven_mod_eight_l148_148167

theorem thirteen_pow_seven_mod_eight : 
  (13^7) % 8 = 5 := by
  sorry

end thirteen_pow_seven_mod_eight_l148_148167


namespace initial_mixture_volume_l148_148990

/--
Given:
1. A mixture initially contains 20% water.
2. When 13.333333333333334 liters of water is added, water becomes 25% of the new mixture.

Prove that the initial volume of the mixture is 200 liters.
-/
theorem initial_mixture_volume (V : ℝ) (h1 : V > 0) (h2 : 0.20 * V + 13.333333333333334 = 0.25 * (V + 13.333333333333334)) : V = 200 :=
sorry

end initial_mixture_volume_l148_148990


namespace problem_a_range_l148_148358

theorem problem_a_range (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x^2 - 2 * (a - 1) * x - 2 < 0) ↔ (-1 < a ∧ a ≤ 1) :=
by
  sorry

end problem_a_range_l148_148358


namespace oranges_purchase_cost_l148_148816

/-- 
Oranges are sold at a rate of $3$ per three pounds.
If a customer buys 18 pounds and receives a discount of $5\%$ for buying more than 15 pounds,
prove that the total amount the customer pays is $17.10.
-/
theorem oranges_purchase_cost (rate : ℕ) (base_weight : ℕ) (discount_rate : ℚ)
  (total_weight : ℕ) (discount_threshold : ℕ) (final_cost : ℚ) :
  rate = 3 → base_weight = 3 → discount_rate = 0.05 → 
  total_weight = 18 → discount_threshold = 15 → final_cost = 17.10 := by
  sorry

end oranges_purchase_cost_l148_148816


namespace cost_for_23_days_l148_148921

-- Define the cost structure
def costFirstWeek : ℕ → ℝ := λ days => if days <= 7 then days * 18 else 7 * 18
def costAdditionalDays : ℕ → ℝ := λ days => if days > 7 then (days - 7) * 14 else 0

-- Total cost equation
def totalCost (days : ℕ) : ℝ := costFirstWeek days + costAdditionalDays days

-- Declare the theorem to prove
theorem cost_for_23_days : totalCost 23 = 350 := by
  sorry

end cost_for_23_days_l148_148921


namespace total_cost_of_aquarium_l148_148277

variable (original_price discount_rate sales_tax_rate : ℝ)
variable (original_cost : original_price = 120)
variable (discount : discount_rate = 0.5)
variable (tax : sales_tax_rate = 0.05)

theorem total_cost_of_aquarium : 
  (original_price * (1 - discount_rate) * (1 + sales_tax_rate) = 63) :=
by
  rw [original_cost, discount, tax]
  sorry

end total_cost_of_aquarium_l148_148277


namespace find_k_l148_148831

variables {r k : ℝ}
variables {O A B C D : EuclideanSpace ℝ (Fin 3)}

-- Points A, B, C, and D lie on a sphere centered at O with radius r
variables (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
-- The given vector equation
variables (h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3)))

theorem find_k (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
(h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3))) : 
k = -7 :=
sorry

end find_k_l148_148831


namespace find_k_l148_148753

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 3) (h2 : m + 2 = 2 * (n + k) + 3) : k = 1 :=
by
  -- Proof is omitted
  sorry

end find_k_l148_148753


namespace Martha_reading_challenge_l148_148635

theorem Martha_reading_challenge :
  ∀ x : ℕ,
  (12 + 18 + 14 + 20 + 11 + 13 + 19 + 15 + 17 + x) / 10 = 15 ↔ x = 11 :=
by sorry

end Martha_reading_challenge_l148_148635


namespace sum_is_zero_l148_148571

-- Define the conditions: the function f is invertible, and f(a) = 3, f(b) = 7
variables {α β : Type} [Inhabited α] [Inhabited β]

def invertible {α β : Type} (f : α → β) :=
  ∃ g : β → α, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

variables (f : ℝ → ℝ) (a b : ℝ)

-- Assume f is invertible and the given conditions f(a) = 3 and f(b) = 7
axiom f_invertible : invertible f
axiom f_a : f a = 3
axiom f_b : f b = 7

-- Prove that a + b = 0
theorem sum_is_zero : a + b = 0 :=
sorry

end sum_is_zero_l148_148571


namespace impossible_coins_l148_148066

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l148_148066


namespace raft_travel_time_l148_148582

-- Define the problem conditions:
def steamboat_time (distance : ℕ) := 1 -- in hours
def motorboat_time (distance : ℕ) : ℚ := 3 / 4 -- in hours
def speed_ratio := 2 -- motorboat speed is twice the speed of steamboat

-- Define the time for the raft to travel the distance:
def raft_time (distance : ℕ) (current_speed : ℚ) := distance / current_speed

-- Given the conditions, prove that the raft time equals to 90 minutes
theorem raft_travel_time (distance : ℕ) (rafter_speed : ℚ) (current_speed : ℚ) :
  steamboat_time distance = 1 ∧ motorboat_time distance = 3 / 4 ∧ rafter_speed = current_speed →
  rafter_speed = current_speed ∧ raft_time distance current_speed = 3 / 2 → -- hours
  raft_time distance current_speed * 60 = 90 := -- convert hours to minutes
by
  intros h1 h2
  sorry

end raft_travel_time_l148_148582


namespace cars_meeting_time_l148_148591

def problem_statement (V_A V_B V_C V_D : ℝ) :=
  (V_A ≠ V_B) ∧ (V_A ≠ V_C) ∧ (V_A ≠ V_D) ∧
  (V_B ≠ V_C) ∧ (V_B ≠ V_D) ∧ (V_C ≠ V_D) ∧
  (V_A + V_C = V_B + V_D) ∧
  (53 * (V_A - V_B) / 46 = 7) ∧
  (53 * (V_D - V_C) / 46 = 7)

theorem cars_meeting_time (V_A V_B V_C V_D : ℝ) (h : problem_statement V_A V_B V_C V_D) : 
  ∃ t : ℝ, t = 53 := 
sorry

end cars_meeting_time_l148_148591


namespace new_trailer_homes_added_l148_148100

theorem new_trailer_homes_added
  (n : ℕ) (avg_age_3_years_ago avg_age_today age_increase new_home_age : ℕ) (k : ℕ) :
  n = 30 → avg_age_3_years_ago = 15 → avg_age_today = 12 → age_increase = 3 → new_home_age = 3 →
  (n * (avg_age_3_years_ago + age_increase) + k * new_home_age) / (n + k) = avg_age_today →
  k = 20 :=
by
  intros h_n h_avg_age_3y h_avg_age_today h_age_increase h_new_home_age h_eq
  sorry

end new_trailer_homes_added_l148_148100


namespace smallest_root_equation_l148_148403

theorem smallest_root_equation :
  ∃ x : ℝ, (3 * x) / (x - 2) + (2 * x^2 - 28) / x = 11 ∧ ∀ y, (3 * y) / (y - 2) + (2 * y^2 - 28) / y = 11 → x ≤ y ∧ x = (-1 - Real.sqrt 17) / 2 :=
sorry

end smallest_root_equation_l148_148403


namespace chandra_pairings_l148_148428

variable (bowls : ℕ) (glasses : ℕ)

theorem chandra_pairings : 
  bowls = 5 → 
  glasses = 4 → 
  bowls * glasses = 20 :=
by intros; 
    sorry

end chandra_pairings_l148_148428


namespace man_speed_in_still_water_l148_148838

theorem man_speed_in_still_water (V_m V_s : ℝ) 
  (h1 : V_m + V_s = 8)
  (h2 : V_m - V_s = 6) : 
  V_m = 7 := 
by
  sorry

end man_speed_in_still_water_l148_148838


namespace gcd_54_180_l148_148693

theorem gcd_54_180 : Nat.gcd 54 180 = 18 := by
  sorry

end gcd_54_180_l148_148693


namespace sum_products_of_chords_l148_148970

variable {r x y u v : ℝ}

theorem sum_products_of_chords (h1 : x * y = u * v) (h2 : 4 * r^2 = (x + y)^2 + (u + v)^2) :
  x * (x + y) + u * (u + v) = 4 * r^2 := by
sorry

end sum_products_of_chords_l148_148970


namespace sacksPerSectionDaily_l148_148303

variable (totalSacks : ℕ) (sections : ℕ) (sacksPerSection : ℕ)

-- Conditions from the problem
variables (h1 : totalSacks = 360) (h2 : sections = 8)

-- The theorem statement
theorem sacksPerSectionDaily : sacksPerSection = 45 :=
by
  have h3 : totalSacks / sections = 45 := by sorry
  have h4 : sacksPerSection = totalSacks / sections := by sorry
  exact Eq.trans h4 h3

end sacksPerSectionDaily_l148_148303


namespace xy_range_l148_148448

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_eqn : x + 3 * y + 2 / x + 4 / y = 10) :
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
  sorry

end xy_range_l148_148448


namespace number_of_women_l148_148751

theorem number_of_women (n_men n_women n_dances men_partners women_partners : ℕ) 
  (h_men_partners : men_partners = 4)
  (h_women_partners : women_partners = 3)
  (h_n_men : n_men = 15)
  (h_total_dances : n_dances = n_men * men_partners)
  (h_women_calc : n_women = n_dances / women_partners) :
  n_women = 20 :=
sorry

end number_of_women_l148_148751


namespace am_gm_inequality_l148_148367

noncomputable def arithmetic_mean (a c : ℝ) : ℝ := (a + c) / 2

noncomputable def geometric_mean (a c : ℝ) : ℝ := Real.sqrt (a * c)

theorem am_gm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  (arithmetic_mean a c - geometric_mean a c < (c - a)^2 / (8 * a)) :=
sorry

end am_gm_inequality_l148_148367


namespace sum_of_six_primes_even_l148_148850

/-- If A, B, and C are positive integers such that A, B, C, A-B, A+B, and A+B+C are all prime numbers, 
    and B is specifically the prime number 2,
    then the sum of these six primes is even. -/
theorem sum_of_six_primes_even (A B C : ℕ) (hA : Prime A) (hB : Prime B) (hC : Prime C) 
    (h1 : Prime (A - B)) (h2 : Prime (A + B)) (h3 : Prime (A + B + C)) (hB_eq_two : B = 2) : 
    Even (A + B + C + (A - B) + (A + B) + (A + B + C)) :=
by
  sorry

end sum_of_six_primes_even_l148_148850


namespace johns_allowance_is_3_45_l148_148197

noncomputable def johns_weekly_allowance (A : ℝ) : Prop :=
  -- Condition 1: John spent 3/5 of his allowance at the arcade
  let spent_at_arcade := (3/5) * A
  -- Remaining allowance
  let remaining_after_arcade := A - spent_at_arcade
  -- Condition 2: He spent 1/3 of the remaining allowance at the toy store
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  -- Condition 3: He spent his last $0.92 at the candy store
  let spent_at_candy_store := 0.92
  -- Remaining amount after the candy store expenditure should be 0
  remaining_after_toy_store = spent_at_candy_store

theorem johns_allowance_is_3_45 : johns_weekly_allowance 3.45 :=
sorry

end johns_allowance_is_3_45_l148_148197


namespace problem_eval_at_x_eq_3_l148_148724

theorem problem_eval_at_x_eq_3 : ∀ x : ℕ, x = 3 → (x^x)^(x^x) = 27^27 :=
by
  intros x hx
  rw [hx]
  sorry

end problem_eval_at_x_eq_3_l148_148724


namespace sum_first_13_terms_l148_148227

theorem sum_first_13_terms
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₁ : a 4 + a 10 - (a 7)^2 + 15 = 0)
  (h₂ : ∀ n : ℕ, a n > 0) :
  S 13 = 65 :=
sorry

end sum_first_13_terms_l148_148227


namespace calculate_expression_l148_148475

theorem calculate_expression (x : ℝ) (h : x + 1/x = 3) : x^12 - 7 * x^6 + x^2 = 45363 * x - 17327 :=
by
  sorry

end calculate_expression_l148_148475


namespace minimize_fees_at_5_l148_148556

noncomputable def minimize_costs (x : ℝ) (y1 y2 : ℝ) : Prop :=
  let k1 := 40
  let k2 := 8 / 5
  y1 = k1 / x ∧ y2 = k2 * x ∧ (∀ x, y1 + y2 ≥ 16 ∧ (y1 + y2 = 16 ↔ x = 5))

theorem minimize_fees_at_5 :
  minimize_costs 5 4 16 :=
sorry

end minimize_fees_at_5_l148_148556


namespace decimal_equivalent_one_half_pow_five_l148_148256

theorem decimal_equivalent_one_half_pow_five :
  (1 / 2) ^ 5 = 0.03125 :=
by sorry

end decimal_equivalent_one_half_pow_five_l148_148256


namespace set_intersection_complement_l148_148835

variable (U : Set ℝ := Set.univ)
variable (M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)})
variable (N : Set ℝ := {x | 0 < x ∧ x < 2})

theorem set_intersection_complement :
  N ∩ (U \ M) = {x | 0 < x ∧ x ≤ 1} :=
  sorry

end set_intersection_complement_l148_148835


namespace solution_set_inequality_l148_148331

theorem solution_set_inequality
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → ax^2 + bx + c > 0) :
  ∃ s : Set ℝ, s = {x | (1/2) < x ∧ x < 1} ∧ ∀ x : ℝ, x ∈ s → cx^2 + bx + a > 0 := by
sorry

end solution_set_inequality_l148_148331


namespace intersection_is_correct_complement_is_correct_l148_148617

open Set

variable {U : Set ℝ} (A B : Set ℝ)

-- Define the universal set U
def U_def : Set ℝ := { x | 1 < x ∧ x < 7 }

-- Define set A
def A_def : Set ℝ := { x | 2 ≤ x ∧ x < 5 }

-- Define set B using the simplified condition from the inequality
def B_def : Set ℝ := { x | x ≥ 3 }

-- Proof statement that A ∩ B is as specified
theorem intersection_is_correct :
  (A_def ∩ B_def) = { x : ℝ | 3 ≤ x ∧ x < 5 } := by
  sorry

-- Proof statement for the complement of A relative to U
theorem complement_is_correct :
  (U_def \ A_def) = { x : ℝ | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7) } := by
  sorry

end intersection_is_correct_complement_is_correct_l148_148617


namespace abc_greater_than_n_l148_148470

theorem abc_greater_than_n
  (a b c n : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : 1 < n)
  (h5 : a ^ n + b ^ n = c ^ n) :
  a > n ∧ b > n ∧ c > n :=
sorry

end abc_greater_than_n_l148_148470


namespace suff_not_nec_l148_148679

theorem suff_not_nec (x : ℝ) : (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬(x ≤ 0)) :=
by
  sorry

end suff_not_nec_l148_148679


namespace max_xyz_l148_148833

theorem max_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : (x * y) + 3 * z = (x + 3 * z) * (y + 3 * z)) 
: ∀ x y z, ∃ (a : ℝ), a = (x * y * z) ∧ a ≤ (1/81) :=
sorry

end max_xyz_l148_148833


namespace completing_square_transformation_l148_148874

theorem completing_square_transformation : ∀ x : ℝ, x^2 - 4 * x - 7 = 0 → (x - 2)^2 = 11 :=
by
  intros x h
  sorry

end completing_square_transformation_l148_148874


namespace hyperbola_equation_l148_148789

theorem hyperbola_equation {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)
    (hfocal : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5)
    (hslope : b / a = 1 / 8) :
    (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :=
by
  -- Goals and conditions to handle proof
  sorry

end hyperbola_equation_l148_148789


namespace original_price_l148_148471

theorem original_price (P : ℝ) (h : P * 0.5 = 1200) : P = 2400 := 
by
  sorry

end original_price_l148_148471


namespace third_term_of_arithmetic_sequence_l148_148041

variable (a : ℕ → ℤ)
variable (a1_eq_2 : a 1 = 2)
variable (a2_eq_8 : a 2 = 8)
variable (arithmetic_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))

theorem third_term_of_arithmetic_sequence :
  a 3 = 14 :=
by
  sorry

end third_term_of_arithmetic_sequence_l148_148041


namespace vector_expression_simplification_l148_148245

variable (a b : Type)
variable (α : Type) [Field α]
variable [AddCommGroup a] [Module α a]

theorem vector_expression_simplification
  (vector_a vector_b : a) :
  (1/3 : α) • (vector_a - (2 : α) • vector_b) + vector_b = (1/3 : α) • vector_a + (1/3 : α) • vector_b :=
by
  sorry

end vector_expression_simplification_l148_148245


namespace sum_of_digits_next_exact_multiple_l148_148495

noncomputable def Michael_next_age_sum_of_digits (L M T n : ℕ) : ℕ :=
  let next_age := M + n
  ((next_age / 10) % 10) + (next_age % 10)

theorem sum_of_digits_next_exact_multiple :
  ∀ (L M T n : ℕ),
    T = 2 →
    M = L + 4 →
    (∀ k : ℕ, k < 8 → ∃ m : ℕ, L = m * T + k * T) →
    (∃ n, (M + n) % (T + n) = 0) →
    Michael_next_age_sum_of_digits L M T n = 9 :=
by
  intros
  sorry

end sum_of_digits_next_exact_multiple_l148_148495


namespace division_remainder_correct_l148_148382

def polynomial_div_remainder (x : ℝ) : ℝ :=
  3 * x^4 + 14 * x^3 - 50 * x^2 - 72 * x + 55

def divisor (x : ℝ) : ℝ :=
  x^2 + 8 * x - 4

theorem division_remainder_correct :
  ∀ x : ℝ, polynomial_div_remainder x % divisor x = 224 * x - 113 :=
by
  sorry

end division_remainder_correct_l148_148382


namespace unique_intersection_value_k_l148_148984

theorem unique_intersection_value_k (k : ℝ) : (∀ x y: ℝ, (y = x^2) ∧ (y = 3*x + k) ↔ k = -9/4) :=
by
  sorry

end unique_intersection_value_k_l148_148984


namespace extremum_areas_extremum_areas_case_b_equal_areas_l148_148200

variable (a b x : ℝ)
variable (h1 : b > 0) (h2 : a ≥ b) (h_cond : 0 < x ∧ x ≤ b)

def area_t1 (a b x : ℝ) : ℝ := 2 * x^2 - (a + b) * x + a * b
def area_t2 (a b x : ℝ) : ℝ := -2 * x^2 + (a + b) * x

noncomputable def x0 (a b : ℝ) : ℝ := (a + b) / 4

-- Problem 1
theorem extremum_areas :
  b ≥ a / 3 → area_t1 a b (x0 a b) ≤ area_t1 a b x ∧ area_t2 a b (x0 a b) ≥ area_t2 a b x :=
sorry

theorem extremum_areas_case_b :
  b < a / 3 → (area_t1 a b b = b^2) ∧ (area_t2 a b b = a * b - b^2) :=
sorry

-- Problem 2
theorem equal_areas :
  b ≤ a ∧ a ≤ 2 * b → (area_t1 a b (a / 2) = area_t2 a b (a / 2)) ∧ (area_t1 a b (b / 2) = area_t2 a b (b / 2)) :=
sorry

end extremum_areas_extremum_areas_case_b_equal_areas_l148_148200


namespace evaluate_expression_l148_148347

def a := 3 + 6 + 9
def b := 2 + 5 + 8
def c := 3 + 6 + 9
def d := 2 + 5 + 8

theorem evaluate_expression : (a / b) - (d / c) = 11 / 30 :=
by
  sorry

end evaluate_expression_l148_148347


namespace snakes_hiding_l148_148654

/-- The statement that given the total number of snakes and the number of snakes not hiding,
we can determine the number of snakes hiding. -/
theorem snakes_hiding (total_snakes : ℕ) (snakes_not_hiding : ℕ) (h1 : total_snakes = 95) (h2 : snakes_not_hiding = 31) :
  total_snakes - snakes_not_hiding = 64 :=
by {
  sorry
}

end snakes_hiding_l148_148654


namespace number_of_questions_per_survey_is_10_l148_148327

variable {Q : ℕ}  -- Q: Number of questions in each survey

def money_per_question : ℝ := 0.2
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4
def total_money_earned : ℝ := 14

theorem number_of_questions_per_survey_is_10 :
    (surveys_on_monday + surveys_on_tuesday) * Q * money_per_question = total_money_earned → Q = 10 :=
by
  sorry

end number_of_questions_per_survey_is_10_l148_148327


namespace max_alpha_for_2_alpha_divides_3n_plus_1_l148_148629

theorem max_alpha_for_2_alpha_divides_3n_plus_1 (n : ℕ) (hn : n > 0) : ∃ α : ℕ, (2 ^ α ∣ (3 ^ n + 1)) ∧ ¬ (2 ^ (α + 1) ∣ (3 ^ n + 1)) ∧ α = 1 :=
by
  sorry

end max_alpha_for_2_alpha_divides_3n_plus_1_l148_148629


namespace find_x_six_l148_148069

noncomputable def positive_real : Type := { x : ℝ // 0 < x }

theorem find_x_six (x : positive_real)
  (h : (1 - x.val ^ 3) ^ (1/3) + (1 + x.val ^ 3) ^ (1/3) = 1) :
  x.val ^ 6 = 28 / 27 := 
sorry

end find_x_six_l148_148069


namespace rectangular_plot_area_l148_148539

theorem rectangular_plot_area (breadth length : ℕ) (h1 : breadth = 14) (h2 : length = 3 * breadth) : (length * breadth) = 588 := 
by 
  -- imports, noncomputable keyword, and placeholder proof for compilation
  sorry

end rectangular_plot_area_l148_148539


namespace max_correct_answers_l148_148039

theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 80) (h2 : 5 * a - 2 * c = 150) : a ≤ 44 :=
by
  sorry

end max_correct_answers_l148_148039


namespace total_amount_l148_148533

theorem total_amount (W X Y Z : ℝ) (h1 : X = 0.8 * W) (h2 : Y = 0.65 * W) (h3 : Z = 0.45 * W) (h4 : Y = 78) : 
  W + X + Y + Z = 348 := by
  sorry

end total_amount_l148_148533


namespace negate_original_is_correct_l148_148361

-- Define the original proposition
def original_proposition (a b : ℕ) : Prop := (a * b = 0) → (a = 0 ∨ b = 0)

-- Define the negated proposition
def negated_proposition (a b : ℕ) : Prop := (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)

-- The theorem stating that the negation of the original proposition is the given negated proposition
theorem negate_original_is_correct (a b : ℕ) : ¬ original_proposition a b ↔ negated_proposition a b := by
  sorry

end negate_original_is_correct_l148_148361


namespace a_n_formula_l148_148803

open Nat

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n * (n + 1) / 2

theorem a_n_formula (n : ℕ) (h : n > 0) 
  (S_n : ℕ → ℕ)
  (hS : ∀ n, S_n n = (n + 2) / 3 * a_n n) 
  : a_n n = n * (n + 1) / 2 := sorry

end a_n_formula_l148_148803


namespace average_eq_5_times_non_zero_l148_148698

theorem average_eq_5_times_non_zero (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := 
by sorry

end average_eq_5_times_non_zero_l148_148698


namespace projection_is_orthocenter_l148_148942

-- Define a structure for a point in 3D space.
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

-- Define mutually perpendicular edges condition.
def mutually_perpendicular {α : Type} [Field α] (A B C D : Point α) :=
(A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) + (A.z - D.z) * (B.z - D.z) = 0 ∧
(A.x - D.x) * (C.x - D.x) + (A.y - D.y) * (C.y - D.y) + (A.z - D.z) * (C.z - D.z) = 0 ∧
(B.x - D.x) * (C.x - D.x) + (B.y - D.y) * (C.y - D.y) + (B.z - D.z) * (C.z - D.z) = 0

-- The main theorem statement.
theorem projection_is_orthocenter {α : Type} [Field α]
    (A B C D : Point α) (h : mutually_perpendicular A B C D) :
    ∃ O : Point α, -- there exists a point O (the orthocenter)
    (O.x * (B.y - A.y) + O.y * (A.y - B.y) + O.z * (A.y - B.y)) = 0 ∧
    (O.x * (C.y - B.y) + O.y * (B.y - C.y) + O.z * (B.y - C.y)) = 0 ∧
    (O.x * (A.y - C.y) + O.y * (C.y - A.y) + O.z * (C.y - A.y)) = 0 := 
sorry

end projection_is_orthocenter_l148_148942


namespace outfit_combinations_l148_148149

def shirts : ℕ := 6
def pants : ℕ := 4
def hats : ℕ := 6

def pant_colors : Finset String := {"tan", "black", "blue", "gray"}
def shirt_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}

def total_combinations : ℕ := shirts * pants * hats
def restricted_combinations : ℕ := pant_colors.card

theorem outfit_combinations
    (hshirts : shirts = 6)
    (hpants : pants = 4)
    (hhats : hats = 6)
    (hpant_colors : pant_colors.card = 4)
    (hshirt_colors : shirt_colors.card = 6)
    (hhat_colors : hat_colors.card = 6)
    (hrestricted : restricted_combinations = pant_colors.card) :
    total_combinations - restricted_combinations = 140 := by
  sorry

end outfit_combinations_l148_148149


namespace three_digit_numbers_div_by_17_l148_148688

theorem three_digit_numbers_div_by_17 : ∃ n : ℕ, n = 53 ∧ 
  let min_k := Nat.ceil (100 / 17)
  let max_k := Nat.floor (999 / 17)
  min_k = 6 ∧ max_k = 58 ∧ (max_k - min_k + 1) = n :=
by
  sorry

end three_digit_numbers_div_by_17_l148_148688


namespace volume_of_pyramid_SPQR_l148_148072

variable (P Q R S : Type)
variable (SP SQ SR : ℝ)
variable (is_perpendicular_SP_SQ : SP * SQ = 0)
variable (is_perpendicular_SQ_SR : SQ * SR = 0)
variable (is_perpendicular_SR_SP : SR * SP = 0)
variable (SP_eq_9 : SP = 9)
variable (SQ_eq_8 : SQ = 8)
variable (SR_eq_7 : SR = 7)

theorem volume_of_pyramid_SPQR : 
  ∃ V : ℝ, V = 84 := by
  -- Conditions and assumption
  sorry

end volume_of_pyramid_SPQR_l148_148072


namespace product_of_roots_l148_148557

noncomputable def quadratic_has_product_of_roots (A B C : ℤ) : ℚ :=
  C / A

theorem product_of_roots (α β : ℚ) (h : 12 * α^2 + 28 * α - 320 = 0) (h2 : 12 * β^2 + 28 * β - 320 = 0) :
  quadratic_has_product_of_roots 12 28 (-320) = -80 / 3 :=
by
  -- Insert proof here
  sorry

end product_of_roots_l148_148557


namespace regular_ducks_sold_l148_148675

theorem regular_ducks_sold (R : ℕ) (h1 : 3 * R + 5 * 185 = 1588) : R = 221 :=
by {
  sorry
}

end regular_ducks_sold_l148_148675


namespace total_length_correct_l148_148552

def segment_lengths_Figure1 : List ℕ := [10, 3, 1, 1, 5, 7]

def removed_segments : List ℕ := [3, 1, 1, 5]

def remaining_segments_Figure2 : List ℕ := [10, (3 + 1 + 1), 7, 1]

def total_length_Figure2 : ℕ := remaining_segments_Figure2.sum

theorem total_length_correct :
  total_length_Figure2 = 23 :=
by
  sorry

end total_length_correct_l148_148552


namespace greatest_product_l148_148775

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l148_148775


namespace rent_percentage_l148_148892

-- Define Elaine's earnings last year
def E : ℝ := sorry

-- Define last year's rent expenditure
def rentLastYear : ℝ := 0.20 * E

-- Define this year's earnings
def earningsThisYear : ℝ := 1.35 * E

-- Define this year's rent expenditure
def rentThisYear : ℝ := 0.30 * earningsThisYear

-- Prove the required percentage
theorem rent_percentage : ((rentThisYear / rentLastYear) * 100) = 202.5 := by
  sorry

end rent_percentage_l148_148892


namespace find_e_l148_148908

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_e (r s d e : ℝ) 
  (h1 : quadratic 2 (-4) (-6) r = 0)
  (h2 : quadratic 2 (-4) (-6) s = 0)
  (h3 : r + s = 2) 
  (h4 : r * s = -3)
  (h5 : d = -(r + s - 6))
  (h6 : e = (r - 3) * (s - 3)) : 
  e = 0 :=
sorry

end find_e_l148_148908


namespace percentage_is_12_l148_148901

variable (x : ℝ) (p : ℝ)

-- Given the conditions
def condition_1 : Prop := 0.25 * x = (p / 100) * 1500 - 15
def condition_2 : Prop := x = 660

-- We need to prove that the percentage p is 12
theorem percentage_is_12 (h1 : condition_1 x p) (h2 : condition_2 x) : p = 12 := by
  sorry

end percentage_is_12_l148_148901


namespace remainder_of_x_divided_by_30_l148_148307

theorem remainder_of_x_divided_by_30:
  ∀ x : ℤ,
    (4 + x ≡ 9 [ZMOD 8]) ∧ 
    (6 + x ≡ 8 [ZMOD 27]) ∧ 
    (8 + x ≡ 49 [ZMOD 125]) ->
    (x ≡ 17 [ZMOD 30]) :=
by
  intros x h
  sorry

end remainder_of_x_divided_by_30_l148_148307


namespace last_digit_of_sum_is_four_l148_148230

theorem last_digit_of_sum_is_four (x y z : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9)
  (h : 1950 ≤ 200 * x + 11 * y + 11 * z ∧ 200 * x + 11 * y + 11 * z < 2000) :
  (200 * x + 11 * y + 11 * z) % 10 = 4 :=
sorry

end last_digit_of_sum_is_four_l148_148230


namespace proof_problem_l148_148652

noncomputable def log2 (n : ℝ) : ℝ := Real.log n / Real.log 2

theorem proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/2 * log2 x + 1/3 * log2 y = 1) : x^3 * y^2 = 64 := 
sorry 

end proof_problem_l148_148652


namespace vector_magnitude_l148_148016

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l148_148016


namespace subtract_some_number_l148_148305

theorem subtract_some_number
  (x : ℤ)
  (h : 913 - x = 514) :
  514 - x = 115 :=
by {
  sorry
}

end subtract_some_number_l148_148305


namespace probability_of_death_each_month_l148_148204

-- Defining the variables and expressions used in conditions
def p : ℝ := 0.1
def N : ℝ := 400
def surviving_after_3_months : ℝ := 291.6

-- The main theorem to be proven
theorem probability_of_death_each_month (prob : ℝ) :
  (N * (1 - prob)^3 = surviving_after_3_months) → (prob = p) :=
by
  sorry

end probability_of_death_each_month_l148_148204


namespace decrypted_plaintext_l148_148145

theorem decrypted_plaintext (a b c d : ℕ) : 
  (a + 2 * b = 14) → (2 * b + c = 9) → (2 * c + 3 * d = 23) → (4 * d = 28) → 
  (a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7) :=
by 
  intros h1 h2 h3 h4
  -- Proof steps go here
  sorry

end decrypted_plaintext_l148_148145


namespace daily_sales_profit_45_selling_price_for_1200_profit_l148_148709

-- Definitions based on given conditions

def cost_price : ℤ := 30
def base_selling_price : ℤ := 40
def base_sales_volume : ℤ := 80
def price_increase_effect : ℤ := 2
def max_selling_price : ℤ := 55

-- Part (1): Prove that for a selling price of 45 yuan, the daily sales profit is 1050 yuan.
theorem daily_sales_profit_45 :
  let selling_price := 45
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = 1050 := by sorry

-- Part (2): Prove that to achieve a daily profit of 1200 yuan, the selling price should be 50 yuan.
theorem selling_price_for_1200_profit :
  let target_profit := 1200
  ∃ (selling_price : ℤ), 
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = target_profit ∧ selling_price ≤ max_selling_price ∧ selling_price = 50 := by sorry

end daily_sales_profit_45_selling_price_for_1200_profit_l148_148709


namespace extremum_f_at_neg_four_thirds_monotonicity_g_l148_148164

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f a x) * Real.exp x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 2 * x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 
  let f_a_x := f a x
  ( f' a x * Real.exp x ) + ( f_a_x * Real.exp x)

theorem extremum_f_at_neg_four_thirds (a : ℝ) :
  f' a (-4/3) = 0 ↔ a = 1/2 := sorry

-- Assuming a = 1/2 from the previous theorem
theorem monotonicity_g :
  let a := 1/2
  ∀ x : ℝ, 
    ((x < -4 → g' a x < 0) ∧ 
     (-4 < x ∧ x < -1 → g' a x > 0) ∧
     (-1 < x ∧ x < 0 → g' a x < 0) ∧
     (x > 0 → g' a x > 0)) := sorry

end extremum_f_at_neg_four_thirds_monotonicity_g_l148_148164


namespace total_distance_l148_148957

variable {D : ℝ}

theorem total_distance (h1 : D / 3 > 0)
                       (h2 : (2 / 3 * D) - (1 / 6 * D) > 0)
                       (h3 : (1 / 2 * D) - (1 / 10 * D) = 180) :
    D = 450 := 
sorry

end total_distance_l148_148957


namespace find_y_coordinate_of_first_point_l148_148380

theorem find_y_coordinate_of_first_point :
  ∃ y1 : ℝ, ∀ k : ℝ, (k = 0.8) → (k = (0.8 - y1) / (5 - (-1))) → y1 = 4 :=
by
  sorry

end find_y_coordinate_of_first_point_l148_148380


namespace series_largest_prime_factor_of_111_l148_148504

def series := [368, 689, 836]  -- given sequence series

def div_condition (n : Nat) := 
  ∃ k : Nat, n = 111 * k

def largest_prime_factor (n : Nat) (p : Nat) := 
  Prime p ∧ ∀ q : Nat, Prime q → q ∣ n → q ≤ p

theorem series_largest_prime_factor_of_111 :
  largest_prime_factor 111 37 := 
by
  sorry

end series_largest_prime_factor_of_111_l148_148504


namespace negation_of_proposition_l148_148294

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 1) ↔ ∃ x : ℝ, x^2 + 1 < 1 :=
by sorry

end negation_of_proposition_l148_148294


namespace total_students_l148_148483

variables (B G : ℕ)
variables (two_thirds_boys : 2 * B = 3 * 400)
variables (three_fourths_girls : 3 * G = 4 * 150)
variables (total_participants : B + G = 800)

theorem total_students (B G : ℕ)
  (two_thirds_boys : 2 * B = 3 * 400)
  (three_fourths_girls : 3 * G = 4 * 150)
  (total_participants : B + G = 800) :
  B + G = 800 :=
by
  sorry

end total_students_l148_148483


namespace fractional_eq_no_real_roots_l148_148583

theorem fractional_eq_no_real_roots (k : ℝ) :
  (∀ x : ℝ, (x - 1) ≠ 0 → (k / (x - 1) + 3 ≠ x / (1 - x))) → k = -1 :=
by
  sorry

end fractional_eq_no_real_roots_l148_148583


namespace leah_ride_time_l148_148549

theorem leah_ride_time (x y : ℝ) (h1 : 90 * x = y) (h2 : 30 * (x + 2 * x) = y)
: ∃ t : ℝ, t = 67.5 :=
by
  -- Define 50% increase in length
  let y' := 1.5 * y
  -- Define escalator speed without Leah walking
  let k := 2 * x
  -- Calculate the time taken
  let t := y' / k
  -- Prove that this time is 67.5 seconds
  have ht : t = 67.5 := sorry
  exact ⟨t, ht⟩

end leah_ride_time_l148_148549


namespace divide_circle_into_parts_l148_148808

theorem divide_circle_into_parts : 
    ∃ (divide : ℕ → ℕ), 
        (divide 3 = 4 ∧ divide 3 = 5 ∧ divide 3 = 6 ∧ divide 3 = 7) :=
by
  -- This illustrates that we require a proof to show that for 3 straight cuts ('n = 3'), 
  -- we can achieve 4, 5, 6, and 7 segments in different settings (circle with strategic line placements).
  sorry

end divide_circle_into_parts_l148_148808


namespace sum_of_medians_powers_l148_148586

noncomputable def median_length_squared (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / 4

noncomputable def sum_of_fourth_powers_of_medians (a b c : ℝ) : ℝ :=
  let mAD := (median_length_squared a b c)^2
  let mBE := (median_length_squared b c a)^2
  let mCF := (median_length_squared c a b)^2
  mAD^2 + mBE^2 + mCF^2

theorem sum_of_medians_powers :
  sum_of_fourth_powers_of_medians 13 14 15 = 7644.25 :=
by
  sorry

end sum_of_medians_powers_l148_148586


namespace other_bill_denomination_l148_148385

-- Define the conditions of the problem
def cost_shirt : ℕ := 80
def ten_dollar_bills : ℕ := 2
def other_bills (x : ℕ) : ℕ := ten_dollar_bills + 1

-- The amount paid with $10 bills
def amount_with_ten_dollar_bills : ℕ := ten_dollar_bills * 10

-- The total amount should match the cost of the shirt
def total_amount (x : ℕ) : ℕ := amount_with_ten_dollar_bills + (other_bills x) * x

-- Statement to prove
theorem other_bill_denomination : 
  ∃ (x : ℕ), total_amount x = cost_shirt ∧ x = 20 :=
by
  sorry

end other_bill_denomination_l148_148385


namespace discount_store_purchase_l148_148276

theorem discount_store_purchase (n x y : ℕ) (hn : 2 * n + (x + y) = 2 * n) 
(h1 : 8 * x + 9 * y = 172) (hx : 0 ≤ x) (hy : 0 ≤ y): 
x = 8 ∧ y = 12 :=
sorry

end discount_store_purchase_l148_148276


namespace triangle_third_side_length_l148_148613

theorem triangle_third_side_length
  (AC BC : ℝ)
  (h_a h_b h_c : ℝ)
  (half_sum_heights_eq : (h_a + h_b) / 2 = h_c) :
  AC = 6 → BC = 3 → AB = 4 :=
by
  sorry

end triangle_third_side_length_l148_148613


namespace part1_part2_l148_148183

noncomputable def f (a c x : ℝ) : ℝ :=
  if x >= c then a * Real.log x + (x - c) ^ 2
  else a * Real.log x - (x - c) ^ 2

theorem part1 (a c : ℝ)
  (h_a : a = 2 * c - 2)
  (h_c_gt_0 : c > 0)
  (h_f_geq : ∀ x, x ∈ (Set.Ioi c) → f a c x >= 1 / 4) :
    a ∈ Set.Icc (-2 : ℝ) (-1 : ℝ) :=
  sorry

theorem part2 (a c x1 x2 : ℝ)
  (h_a_lt_0 : a < 0)
  (h_c_gt_0 : c > 0)
  (h_x1 : x1 = Real.sqrt (- a / 2))
  (h_x2 : x2 = c)
  (h_tangents_intersect : deriv (f a c) x1 * deriv (f a c) x2 = -1) :
    c >= 3 * Real.sqrt 3 / 2 :=
  sorry

end part1_part2_l148_148183


namespace range_of_2a_minus_b_l148_148820

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := 
sorry

end range_of_2a_minus_b_l148_148820


namespace terminal_side_third_quadrant_l148_148607

noncomputable def angle_alpha : ℝ := (7 * Real.pi) / 5

def is_in_third_quadrant (angle : ℝ) : Prop :=
  ∃ k : ℤ, (3 * Real.pi) / 2 < angle + 2 * k * Real.pi ∧ angle + 2 * k * Real.pi < 2 * Real.pi

theorem terminal_side_third_quadrant : is_in_third_quadrant angle_alpha :=
sorry

end terminal_side_third_quadrant_l148_148607


namespace find_F_of_circle_l148_148194

def circle_equation (x y F : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + F = 0

def is_circle_with_radius (x y F r : ℝ) : Prop := 
  ∃ k h, (x - k)^2 + (y + h)^2 = r

theorem find_F_of_circle {F : ℝ} :
  (∀ x y : ℝ, circle_equation x y F) ∧ 
  is_circle_with_radius 1 1 F 4 → F = -2 := 
by
  sorry

end find_F_of_circle_l148_148194


namespace relationship_among_mnr_l148_148722

-- Definitions of the conditions
variables {a b c : ℝ}
variables (m n r : ℝ)

-- Assumption given by the conditions
def conditions (a b c : ℝ) := 0 < a ∧ a < b ∧ b < 1 ∧ 1 < c
def log_equations (a b c m n : ℝ) := m = Real.log c / Real.log a ∧ n = Real.log c / Real.log b
def r_definition (a c r : ℝ) := r = a^c

-- Statement: If the conditions are satisfied, then the relationship holds
theorem relationship_among_mnr (a b c m n r : ℝ)
  (h1 : conditions a b c)
  (h2 : log_equations a b c m n)
  (h3 : r_definition a c r) :
  n < m ∧ m < r := by
  sorry

end relationship_among_mnr_l148_148722


namespace find_interest_rate_l148_148209

noncomputable def amount : ℝ := 896
noncomputable def principal : ℝ := 799.9999999999999
noncomputable def time : ℝ := 2 + 2 / 5
noncomputable def interest : ℝ := amount - principal
noncomputable def rate : ℝ := interest / (principal * time)

theorem find_interest_rate :
  rate * 100 = 5 := by
  sorry

end find_interest_rate_l148_148209


namespace intersection_S_T_l148_148037

def setS (x : ℝ) : Prop := (x - 1) * (x - 3) ≥ 0
def setT (x : ℝ) : Prop := x > 0

theorem intersection_S_T : {x : ℝ | setS x} ∩ {x : ℝ | setT x} = {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (3 ≤ x)} := 
sorry

end intersection_S_T_l148_148037


namespace Q_eq_G_l148_148226

def P := {y | ∃ x, y = x^2 + 1}
def Q := {y : ℝ | ∃ x, y = x^2 + 1}
def E := {x : ℝ | ∃ y, y = x^2 + 1}
def F := {(x, y) | y = x^2 + 1}
def G := {x : ℝ | x ≥ 1}

theorem Q_eq_G : Q = G := by
  sorry

end Q_eq_G_l148_148226


namespace min_value_sum_pos_int_l148_148213

theorem min_value_sum_pos_int 
  (a b c : ℕ)
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots: ∃ (A B : ℝ), A < 0 ∧ A > -1 ∧ B > 0 ∧ B < 1 ∧ (∀ x : ℝ, x^2*x*a + x*b + c = 0 → x = A ∨ x = B))
  : a + b + c = 11 :=
sorry

end min_value_sum_pos_int_l148_148213


namespace polynomial_roots_arithmetic_progression_l148_148512

theorem polynomial_roots_arithmetic_progression (m n : ℝ)
  (h : ∃ a : ℝ, ∃ d : ℝ, ∃ b : ℝ,
   (a = b ∧ (b + d) + (b + 2*d) + (b + 3*d) + b = 0) ∧
   (b * (b + d) * (b + 2*d) * (b + 3*d) = 144) ∧
   b ≠ (b + d) ∧ (b + d) ≠ (b + 2*d) ∧ (b + 2*d) ≠ (b + 3*d)) :
  m = -40 := sorry

end polynomial_roots_arithmetic_progression_l148_148512


namespace polynomial_positive_values_l148_148531

noncomputable def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_values :
  ∀ (z : ℝ), (∃ (x y : ℝ), P x y = z) ↔ z > 0 :=
by
  sorry

end polynomial_positive_values_l148_148531


namespace piles_stones_l148_148091

theorem piles_stones (a b c d : ℕ)
  (h₁ : a = 2011)
  (h₂ : b = 2010)
  (h₃ : c = 2009)
  (h₄ : d = 2008) :
  ∃ (k l m n : ℕ), (k, l, m, n) = (0, 0, 0, 2) ∧
  ((∃ x y z w : ℕ, k = x - y ∧ l = y - z ∧ m = z - w ∧ x + l + m + w = 0) ∨
   (∃ u : ℕ, k = a - u ∧ l = b - u ∧ m = c - u ∧ n = d - u)) :=
sorry

end piles_stones_l148_148091


namespace diamond_fifteen_two_l148_148389

def diamond (a b : ℤ) : ℤ := a + (a / (b + 1))

theorem diamond_fifteen_two : diamond 15 2 = 20 := 
by 
    sorry

end diamond_fifteen_two_l148_148389


namespace perfect_square_m_value_l148_148421

theorem perfect_square_m_value (y m : ℤ) (h : ∃ k : ℤ, y^2 - 8 * y + m = (y - k)^2) : m = 16 :=
sorry

end perfect_square_m_value_l148_148421


namespace triangle_inequality_right_triangle_l148_148535

theorem triangle_inequality_right_triangle
  (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a + b) / Real.sqrt 2 ≤ c :=
by sorry

end triangle_inequality_right_triangle_l148_148535


namespace angelina_journey_equation_l148_148337

theorem angelina_journey_equation (t : ℝ) :
    4 = t + 15/60 + (4 - 15/60 - t) →
    60 * t + 90 * (15/4 - t) = 255 :=
    by
    sorry

end angelina_journey_equation_l148_148337


namespace map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l148_148666

theorem map_x_eq_3_and_y_eq_2_under_z_squared_to_uv :
  (∀ (z : ℂ), (z = 3 + I * z.im) → ((z^2).re = 9 - (9*z.im^2) / 36)) ∧
  (∀ (z : ℂ), (z = z.re + I * 2) → ((z^2).re = (4*z.re^2) / 16 - 4)) :=
by 
  sorry

end map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l148_148666


namespace determine_m_value_l148_148804

theorem determine_m_value (m : ℤ) (A : Set ℤ) : 
  A = {1, m + 2, m^2 + 4} → 5 ∈ A → m = 3 ∨ m = 1 := 
by
  sorry

end determine_m_value_l148_148804


namespace tank_breadth_l148_148465

/-
  We need to define the conditions:
  1. The field dimensions.
  2. The tank dimensions (length and depth), and the unknown breadth.
  3. The relationship after the tank is dug.
-/

noncomputable def field_length : ℝ := 90
noncomputable def field_breadth : ℝ := 50
noncomputable def tank_length : ℝ := 25
noncomputable def tank_depth : ℝ := 4
noncomputable def rise_in_level : ℝ := 0.5

theorem tank_breadth (B : ℝ) (h : 100 * B = (field_length * field_breadth - tank_length * B) * rise_in_level) : B = 20 :=
by sorry

end tank_breadth_l148_148465


namespace circle_condition_l148_148132

theorem circle_condition (m : ℝ) :
    (4 * m) ^ 2 + 4 - 4 * 5 * m > 0 ↔ (m < 1 / 4 ∨ m > 1) := sorry

end circle_condition_l148_148132


namespace length_of_hallway_is_six_l148_148537

noncomputable def length_of_hallway (total_area_square_feet : ℝ) (central_area_side_length : ℝ) (hallway_width : ℝ) : ℝ :=
  (total_area_square_feet - (central_area_side_length * central_area_side_length)) / hallway_width

theorem length_of_hallway_is_six 
  (total_area_square_feet : ℝ)
  (central_area_side_length : ℝ)
  (hallway_width : ℝ)
  (h1 : total_area_square_feet = 124)
  (h2 : central_area_side_length = 10)
  (h3 : hallway_width = 4) :
  length_of_hallway total_area_square_feet central_area_side_length hallway_width = 6 := by
  sorry

end length_of_hallway_is_six_l148_148537


namespace negation_of_sin_le_one_l148_148890

theorem negation_of_sin_le_one : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end negation_of_sin_le_one_l148_148890


namespace third_term_is_five_l148_148898

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Suppose S_n = n^2 for n ∈ ℕ*
axiom H1 : ∀ n : ℕ, n > 0 → S n = n * n

-- The relationship a_n = S_n - S_(n-1) for n ≥ 2
axiom H2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)

-- Prove that the third term is 5
theorem third_term_is_five : a 3 = 5 := by
  sorry

end third_term_is_five_l148_148898


namespace probability_of_diamond_ace_joker_l148_148068

noncomputable def probability_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  event_cards / total_cards

noncomputable def probability_not_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_event total_cards event_cards

noncomputable def probability_none_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  (probability_not_event total_cards event_cards) * (probability_not_event total_cards event_cards)

noncomputable def probability_at_least_one_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_none_event_two_trials total_cards event_cards

theorem probability_of_diamond_ace_joker 
  (total_cards : ℕ := 54) (event_cards : ℕ := 18) :
  probability_at_least_one_event_two_trials total_cards event_cards = 5 / 9 :=
by
  sorry

end probability_of_diamond_ace_joker_l148_148068


namespace combined_age_l148_148632

variable (m y o : ℕ)

noncomputable def younger_brother_age := 5

noncomputable def older_brother_age_based_on_younger := 3 * younger_brother_age

noncomputable def older_brother_age_based_on_michael (m : ℕ) := 1 + 2 * (m - 1)

theorem combined_age (m y o : ℕ) (h1 : y = younger_brother_age) (h2 : o = older_brother_age_based_on_younger) (h3 : o = older_brother_age_based_on_michael m) :
  y + o + m = 28 := by
  sorry

end combined_age_l148_148632


namespace difference_of_numbers_l148_148517

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := 
by
  sorry

end difference_of_numbers_l148_148517


namespace num_children_l148_148444

-- Defining the conditions
def num_adults : Nat := 10
def price_adult_ticket : Nat := 8
def total_bill : Nat := 124
def price_child_ticket : Nat := 4

-- Statement to prove: Number of children
theorem num_children (num_adults : Nat) (price_adult_ticket : Nat) (total_bill : Nat) (price_child_ticket : Nat) : Nat :=
  let cost_adults := num_adults * price_adult_ticket
  let cost_child := total_bill - cost_adults
  cost_child / price_child_ticket

example : num_children 10 8 124 4 = 11 := sorry

end num_children_l148_148444


namespace average_score_l148_148748

variable (u v A : ℝ)
variable (h1 : v / u = 1/3)
variable (h2 : A = (u + v) / 2)

theorem average_score : A = (2/3) * u := by
  sorry

end average_score_l148_148748


namespace second_divisor_is_24_l148_148525

theorem second_divisor_is_24 (m n k l : ℤ) (hm : m = 288 * k + 47) (hn : m = n * l + 23) : n = 24 :=
by
  sorry

end second_divisor_is_24_l148_148525


namespace product_of_roots_l148_148031

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 9 * x + 20

-- The main statement for the Lean theorem
theorem product_of_roots : (∃ x₁ x₂ : ℝ, quadratic x₁ = 0 ∧ quadratic x₂ = 0 ∧ x₁ * x₂ = 20) :=
by
  sorry

end product_of_roots_l148_148031


namespace average_annual_growth_rate_sales_revenue_2018_l148_148005

-- Define the conditions as hypotheses
def initial_sales := 200000
def final_sales := 800000
def years := 2
def growth_rate := 1.0 -- representing 100%

theorem average_annual_growth_rate (x : ℝ) :
  (initial_sales : ℝ) * (1 + x)^years = final_sales → x = 1 :=
by
  intro h1
  sorry

theorem sales_revenue_2018 (x : ℝ) (revenue_2017 : ℝ) :
  x = 1 → revenue_2017 = final_sales → revenue_2017 * (1 + x) = 1600000 :=
by
  intros h1 h2
  sorry

end average_annual_growth_rate_sales_revenue_2018_l148_148005


namespace cost_of_gas_used_l148_148599

theorem cost_of_gas_used (initial_odometer final_odometer fuel_efficiency cost_per_gallon : ℝ)
  (h₀ : initial_odometer = 82300)
  (h₁ : final_odometer = 82335)
  (h₂ : fuel_efficiency = 22)
  (h₃ : cost_per_gallon = 3.80) :
  (final_odometer - initial_odometer) / fuel_efficiency * cost_per_gallon = 6.04 :=
by
  sorry

end cost_of_gas_used_l148_148599


namespace find_x_l148_148847

theorem find_x (x : ℝ) (h : 0.60 / x = 6 / 2) : x = 0.2 :=
by {
  sorry
}

end find_x_l148_148847


namespace area_of_triangle_formed_by_tangent_line_l148_148336
-- Import necessary libraries from Mathlib

-- Set up the problem
theorem area_of_triangle_formed_by_tangent_line
  (f : ℝ → ℝ) (h_f : ∀ x, f x = x^2) :
  let slope := (deriv f 1)
  let tangent_line (x : ℝ) := slope * (x - 1) + f 1
  let x_intercept := (0 : ℝ)
  let y_intercept := tangent_line 0
  let area := 0.5 * abs x_intercept * abs y_intercept
  area = 1 / 4 :=
by
  sorry -- Proof to be completed

end area_of_triangle_formed_by_tangent_line_l148_148336


namespace range_of_a_for_monotonic_increasing_f_l148_148660

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x - 2 * Real.log x

theorem range_of_a_for_monotonic_increasing_f (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → (x - a - 2 / x) ≥ 0) : a ≤ -1 :=
by {
  -- Placeholder for the detailed proof steps
  sorry
}

end range_of_a_for_monotonic_increasing_f_l148_148660


namespace undefined_values_l148_148953

theorem undefined_values (a : ℝ) : a = -3 ∨ a = 3 ↔ (a^2 - 9 = 0) := sorry

end undefined_values_l148_148953


namespace fried_busy_frog_l148_148633

open ProbabilityTheory

def initial_position : (ℤ × ℤ) := (0, 0)

def possible_moves : List (ℤ × ℤ) := [(0, 0), (1, 0), (0, 1)]

def p (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = initial_position then 1 else 0

noncomputable def transition (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = (0, 0) then 1/3 * p n (0, 0)
  else if pos = (0, 1) then 1/3 * p n (0, 0) + 1/3 * p n (0, 1)
  else if pos = (1, 0) then 1/3 * p n (0, 0) + 1/3 * p n (1, 0)
  else 0

noncomputable def p_1 (pos : ℤ × ℤ) : ℚ := transition 0 pos

noncomputable def p_2 (pos : ℤ × ℤ) : ℚ := transition 1 pos

noncomputable def p_3 (pos : ℤ × ℤ) : ℚ := transition 2 pos

theorem fried_busy_frog :
  p_3 (0, 0) = 1/27 :=
by
  sorry

end fried_busy_frog_l148_148633


namespace incorrect_statement_C_l148_148422

noncomputable def f (a x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem incorrect_statement_C :
  ¬ (∀ a : ℝ, a > 0 → ∀ x : ℝ, x > 0 → f a x ≥ 0) := sorry

end incorrect_statement_C_l148_148422


namespace part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l148_148412

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem part1_smallest_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem part1_monotonic_interval :
  ∀ k : ℤ, ∀ x, (k * Real.pi - Real.pi / 3) ≤ x ∧ x ≤ (k * Real.pi + Real.pi / 6) →
  ∃ (b a c : ℝ) (A : ℝ), b + c = 2 * a ∧ 2 * A = A + Real.pi / 3 ∧ 
  f A = 1 / 2 ∧ a = 3 * Real.sqrt 2 := 
sorry

theorem part2_value_of_a :
  ∀ (A b c : ℝ), 
  (∃ (a : ℝ), 2 * a = b + c ∧ 
  f A = 1 / 2 ∧ 
  b * c = 18 ∧ 
  Real.cos A = 1 / 2) → 
  ∃ a, a = 3 * Real.sqrt 2 := 
sorry

end part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l148_148412


namespace find_coefficients_l148_148203

variables {x1 x2 x3 x4 x5 x6 x7 : ℝ}

theorem find_coefficients
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 14)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 30)
  (h4 : 16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 70) :
  25*x1 + 36*x2 + 49*x3 + 64*x4 + 81*x5 + 100*x6 + 121*x7 = 130 :=
sorry

end find_coefficients_l148_148203


namespace find_number_l148_148784

theorem find_number (x : ℝ) (h : 2 = 0.04 * x) : x = 50 := 
sorry

end find_number_l148_148784


namespace total_books_after_loss_l148_148861

-- Define variables for the problem
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

-- Prove the final number of books together
theorem total_books_after_loss : (sandy_books + tim_books - benny_lost_books) = 19 := by
  sorry

end total_books_after_loss_l148_148861


namespace alec_string_ways_l148_148973

theorem alec_string_ways :
  let letters := ['A', 'C', 'G', 'N']
  let num_ways := 24 * 2 * 2
  num_ways = 96 := 
by
  sorry

end alec_string_ways_l148_148973


namespace sale_price_correct_l148_148044

noncomputable def original_price : ℝ := 600.00
noncomputable def first_discount_factor : ℝ := 0.75
noncomputable def second_discount_factor : ℝ := 0.90
noncomputable def final_price : ℝ := original_price * first_discount_factor * second_discount_factor
noncomputable def expected_final_price : ℝ := 0.675 * original_price

theorem sale_price_correct : final_price = expected_final_price := sorry

end sale_price_correct_l148_148044


namespace gunther_cleaning_free_time_l148_148626

theorem gunther_cleaning_free_time :
  let vacuum := 45
  let dusting := 60
  let mopping := 30
  let bathroom := 40
  let windows := 15
  let brushing_per_cat := 5
  let cats := 4

  let free_time_hours := 4
  let free_time_minutes := 25

  let cleaning_time := vacuum + dusting + mopping + bathroom + windows + (brushing_per_cat * cats)
  let free_time_total := (free_time_hours * 60) + free_time_minutes

  free_time_total - cleaning_time = 55 :=
by
  sorry

end gunther_cleaning_free_time_l148_148626


namespace line_y2_does_not_pass_through_fourth_quadrant_l148_148928

theorem line_y2_does_not_pass_through_fourth_quadrant (k b : ℝ) (h1 : k < 0) (h2 : b > 0) : 
  ¬(∃ x y : ℝ, (y = b * x - k ∧ x > 0 ∧ y < 0)) := 
by 
  sorry

end line_y2_does_not_pass_through_fourth_quadrant_l148_148928


namespace arithmetic_sequence_75th_term_diff_l148_148429

noncomputable def sum_arith_sequence (n : ℕ) (a d : ℚ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_75th_term_diff {n : ℕ} {a d : ℚ}
  (hn : n = 150)
  (sum_seq : sum_arith_sequence n a d = 15000)
  (term_range : ∀ k, 0 ≤ k ∧ k < n → 20 ≤ a + k * d ∧ a + k * d ≤ 150)
  (t75th : ∃ L G, L = a + 74 * d ∧ G = a + 74 * d) :
  G - L = (7500 / 149) :=
sorry

end arithmetic_sequence_75th_term_diff_l148_148429


namespace find_a_value_l148_148771

def quadratic_vertex_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = 2) → (y = 5) →
  a * (x - 2)^2 + 5 = y

def quadratic_passing_point_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = -1) → (y = -20) →
  a * (x - 2)^2 + 5 = y

theorem find_a_value : ∃ a : ℚ, quadratic_vertex_condition a ∧ quadratic_passing_point_condition a ∧ a = (-25)/9 := 
by 
  sorry

end find_a_value_l148_148771


namespace miki_pear_juice_l148_148488

def total_pears : ℕ := 18
def total_oranges : ℕ := 10
def pear_juice_per_pear : ℚ := 10 / 2
def orange_juice_per_orange : ℚ := 12 / 3
def max_blend_volume : ℚ := 44

theorem miki_pear_juice : (total_oranges * orange_juice_per_orange = 40) ∧ (max_blend_volume - 40 = 4) → 
  ∃ p : ℚ, p * pear_juice_per_pear = 4 ∧ p = 0 :=
by
  sorry

end miki_pear_juice_l148_148488


namespace perimeter_of_region_l148_148725

-- Define the conditions as Lean definitions
def area_of_region (a : ℝ) := a = 400
def number_of_squares (n : ℕ) := n = 8
def arrangement := "2x4 rectangle"

-- Define the statement we need to prove
theorem perimeter_of_region (a : ℝ) (n : ℕ) (s : ℝ) 
  (h_area_region : area_of_region a) 
  (h_number_of_squares : number_of_squares n) 
  (h_arrangement : arrangement = "2x4 rectangle")
  (h_area_one_square : a / n = s^2) :
  4 * 10 * (s) = 80 * 2^(1/2)  :=
by sorry

end perimeter_of_region_l148_148725


namespace james_collects_15_gallons_per_inch_l148_148839

def rain_gallons_per_inch (G : ℝ) : Prop :=
  let monday_rain := 4
  let tuesday_rain := 3
  let price_per_gallon := 1.2
  let total_money := 126
  let total_rain := monday_rain + tuesday_rain
  (total_rain * G = total_money / price_per_gallon)

theorem james_collects_15_gallons_per_inch : rain_gallons_per_inch 15 :=
by
  -- This is the theorem statement; the proof is not required.
  sorry

end james_collects_15_gallons_per_inch_l148_148839


namespace find_B_l148_148289

variable {A B C D : ℕ}

-- Condition 1: The first dig site (A) was dated 352 years more recent than the second dig site (B)
axiom h1 : A = B + 352

-- Condition 2: The third dig site (C) was dated 3700 years older than the first dig site (A)
axiom h2 : C = A - 3700

-- Condition 3: The fourth dig site (D) was twice as old as the third dig site (C)
axiom h3 : D = 2 * C

-- Condition 4: The age difference between the second dig site (B) and the third dig site (C) was four times the difference between the fourth dig site (D) and the first dig site (A)
axiom h4 : B - C = 4 * (D - A)

-- Condition 5: The fourth dig site is dated 8400 BC.
axiom h5 : D = 8400

-- Prove the question
theorem find_B : B = 7548 :=
by
  sorry

end find_B_l148_148289


namespace no_integer_solution_mx2_minus_sy2_eq_3_l148_148150

theorem no_integer_solution_mx2_minus_sy2_eq_3 (m s : ℤ) (x y : ℤ) (h : m * s = 2000 ^ 2001) :
  ¬ (m * x ^ 2 - s * y ^ 2 = 3) :=
sorry

end no_integer_solution_mx2_minus_sy2_eq_3_l148_148150


namespace total_birds_remaining_l148_148007

theorem total_birds_remaining (grey_birds_in_cage : ℕ) (white_birds_next_to_cage : ℕ) :
  (grey_birds_in_cage = 40) →
  (white_birds_next_to_cage = grey_birds_in_cage + 6) →
  (1/2 * grey_birds_in_cage = 20) →
  (1/2 * grey_birds_in_cage + white_birds_next_to_cage = 66) :=
by 
  intros h_grey_birds h_white_birds h_grey_birds_freed
  sorry

end total_birds_remaining_l148_148007


namespace basketball_count_l148_148067

theorem basketball_count (s b v : ℕ) 
  (h1 : s = b + 23) 
  (h2 : v = s - 18)
  (h3 : v = 40) : b = 35 :=
by sorry

end basketball_count_l148_148067


namespace eggs_collected_l148_148680

def total_eggs_collected (b1 e1 b2 e2 : ℕ) : ℕ :=
  b1 * e1 + b2 * e2

theorem eggs_collected :
  total_eggs_collected 450 36 405 42 = 33210 :=
by
  sorry

end eggs_collected_l148_148680


namespace find_percentage_l148_148159

noncomputable def percentage_condition (P : ℝ) : Prop :=
  9000 + (P / 100) * 9032 = 10500

theorem find_percentage (P : ℝ) (h : percentage_condition P) : P = 16.61 :=
sorry

end find_percentage_l148_148159


namespace find_d_l148_148757

noncomputable def problem_condition :=
  ∃ (v d : ℝ × ℝ) (t : ℝ) (x y : ℝ),
  (y = (5 * x - 7) / 6) ∧ 
  ((x, y) = (v.1 + t * d.1, v.2 + t * d.2)) ∧ 
  (x ≥ 4) ∧ 
  (dist (x, y) (4, 2) = t)

noncomputable def correct_answer : ℝ × ℝ := ⟨6 / 7, 5 / 7⟩

theorem find_d 
  (h : problem_condition) : 
  ∃ (d : ℝ × ℝ), d = correct_answer :=
sorry

end find_d_l148_148757


namespace number_of_roses_l148_148229

def total_flowers : ℕ := 10
def carnations : ℕ := 5
def roses : ℕ := total_flowers - carnations

theorem number_of_roses : roses = 5 := by
  sorry

end number_of_roses_l148_148229


namespace fencers_count_l148_148083

theorem fencers_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 :=
sorry

end fencers_count_l148_148083


namespace probability_meeting_proof_l148_148930

noncomputable def probability_meeting (arrival_time_paul arrival_time_caroline : ℝ) : Prop :=
  arrival_time_paul ≤ arrival_time_caroline + 1 / 4 ∧ arrival_time_paul ≥ arrival_time_caroline - 1 / 4

theorem probability_meeting_proof :
  ∀ (arrival_time_paul arrival_time_caroline : ℝ)
    (h_paul_range : 0 ≤ arrival_time_paul ∧ arrival_time_paul ≤ 1)
    (h_caroline_range: 0 ≤ arrival_time_caroline ∧ arrival_time_caroline ≤ 1),
  (probability_meeting arrival_time_paul arrival_time_caroline) → 
  ∃ p, p = 7/16 :=
by
  sorry

end probability_meeting_proof_l148_148930


namespace toy_factory_days_per_week_l148_148778

theorem toy_factory_days_per_week (toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : toys_per_week = 4560) (h₂ : toys_per_day = 1140) : toys_per_week / toys_per_day = 4 := 
by {
  -- Proof to be provided
  sorry
}

end toy_factory_days_per_week_l148_148778


namespace hyperbola_circle_intersection_l148_148232

open Real

theorem hyperbola_circle_intersection (a r : ℝ) (P Q R S : ℝ × ℝ) 
  (hP : P.1^2 - P.2^2 = a^2) (hQ : Q.1^2 - Q.2^2 = a^2) (hR : R.1^2 - R.2^2 = a^2) (hS : S.1^2 - S.2^2 = a^2)
  (hO : r ≥ 0)
  (hPQRS : (P.1 - 0)^2 + (P.2 - 0)^2 = r^2 ∧
            (Q.1 - 0)^2 + (Q.2 - 0)^2 = r^2 ∧
            (R.1 - 0)^2 + (R.2 - 0)^2 = r^2 ∧
            (S.1 - 0)^2 + (S.2 - 0)^2 = r^2) : 
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end hyperbola_circle_intersection_l148_148232


namespace will_can_buy_correct_amount_of_toys_l148_148415

-- Define the initial conditions as constants
def initial_amount : Int := 57
def amount_spent : Int := 27
def cost_per_toy : Int := 6

-- Lemma stating the problem to prove.
theorem will_can_buy_correct_amount_of_toys : (initial_amount - amount_spent) / cost_per_toy = 5 :=
by
  sorry

end will_can_buy_correct_amount_of_toys_l148_148415


namespace initial_group_size_l148_148760

theorem initial_group_size (W : ℝ) : 
  (∃ n : ℝ, (W + 15) / n = W / n + 2.5) → n = 6 :=
by
  sorry

end initial_group_size_l148_148760


namespace smallest_n_for_partition_condition_l148_148014

theorem smallest_n_for_partition_condition :
  ∃ n : ℕ, n = 4 ∧ ∀ T, (T = {i : ℕ | 2 ≤ i ∧ i ≤ n}) →
  (∀ A B, (T = A ∪ B ∧ A ∩ B = ∅) →
   (∃ a b c, (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B) ∧ (a + b = c))) := sorry

end smallest_n_for_partition_condition_l148_148014


namespace function_is_odd_and_increasing_l148_148584

theorem function_is_odd_and_increasing :
  (∀ x : ℝ, (x^(1/3) : ℝ) = -( (-x)^(1/3) : ℝ)) ∧ (∀ x y : ℝ, x < y → (x^(1/3) : ℝ) < (y^(1/3) : ℝ)) :=
by
  sorry

end function_is_odd_and_increasing_l148_148584


namespace automobile_travel_distance_l148_148051

theorem automobile_travel_distance 
  (a r : ℝ) 
  (travel_rate : ℝ) (h1 : travel_rate = a / 6)
  (time_in_seconds : ℝ) (h2 : time_in_seconds = 180):
  (3 * time_in_seconds * travel_rate) * (1 / r) * (1 / 3) = 10 * a / r :=
by
  sorry

end automobile_travel_distance_l148_148051


namespace calculate_weekly_charge_l148_148394

-- Defining conditions as constraints
def daily_charge : ℕ := 30
def total_days : ℕ := 11
def total_cost : ℕ := 310

-- Defining the weekly charge
def weekly_charge : ℕ := 190

-- Prove that the weekly charge for the first week of rental is $190
theorem calculate_weekly_charge (daily_charge total_days total_cost weekly_charge: ℕ) (daily_charge_eq : daily_charge = 30) (total_days_eq : total_days = 11) (total_cost_eq : total_cost = 310) : 
  weekly_charge = 190 :=
by
  sorry

end calculate_weekly_charge_l148_148394


namespace average_is_5x_minus_10_implies_x_is_50_l148_148790

theorem average_is_5x_minus_10_implies_x_is_50 (x : ℝ) 
  (h : (1 / 3) * ((3 * x + 8) + (7 * x + 3) + (4 * x + 9)) = 5 * x - 10) : 
  x = 50 :=
by
  sorry

end average_is_5x_minus_10_implies_x_is_50_l148_148790


namespace coords_reflect_origin_l148_148207

def P : Type := (ℤ × ℤ)

def reflect_origin (p : P) : P :=
  (-p.1, -p.2)

theorem coords_reflect_origin (p : P) (hx : p = (2, -1)) : reflect_origin p = (-2, 1) :=
by
  sorry

end coords_reflect_origin_l148_148207


namespace cone_base_radius_l148_148939

variable (s : ℝ) (A : ℝ) (r : ℝ)

theorem cone_base_radius (h1 : s = 5) (h2 : A = 15 * Real.pi) : r = 3 :=
by
  sorry

end cone_base_radius_l148_148939


namespace original_average_weight_l148_148733

theorem original_average_weight 
  (W : ℝ)  -- Define W as the original average weight
  (h1 : 0 < W)  -- Define conditions
  (w_new1 : ℝ := 110)
  (w_new2 : ℝ := 60)
  (num_initial_players : ℝ := 7)
  (num_total_players : ℝ := 9)
  (new_average_weight : ℝ := 92)
  (total_weight_initial := num_initial_players * W)
  (total_weight_additional := w_new1 + w_new2)
  (total_weight_total := new_average_weight * num_total_players) : 
  total_weight_initial + total_weight_additional = total_weight_total → W = 94 :=
by 
  sorry

end original_average_weight_l148_148733


namespace ann_frosting_time_l148_148030

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end ann_frosting_time_l148_148030


namespace total_license_groups_l148_148153

-- Defining the given conditions
def letter_choices : Nat := 3
def digit_choices_per_slot : Nat := 10
def number_of_digit_slots : Nat := 5

-- Statement to prove that the total number of different license groups is 300000
theorem total_license_groups : letter_choices * (digit_choices_per_slot ^ number_of_digit_slots) = 300000 := by
  sorry

end total_license_groups_l148_148153


namespace problem_l148_148919

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a (n : ℕ) : ℤ := sorry -- Define the arithmetic sequence a_n based on conditions

-- Problem statement
theorem problem : 
  (a 1 = 4) ∧
  (a 2 + a 4 = 4) →
  (∃ d : ℤ, arithmetic_sequence a d ∧ a 10 = -5) :=
by {
  sorry
}

end problem_l148_148919


namespace primes_satisfying_equation_l148_148286

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l148_148286


namespace smaller_bills_denomination_correct_l148_148463

noncomputable def denomination_of_smaller_bills : ℕ :=
  let total_money := 1000
  let part_smaller_bills := 3 / 10
  let smaller_bills_amount := part_smaller_bills * total_money
  let rest_of_money := total_money - smaller_bills_amount
  let bill_100_denomination := 100
  let total_bills := 13
  let num_100_bills := rest_of_money / bill_100_denomination
  let num_smaller_bills := total_bills - num_100_bills
  let denomination := smaller_bills_amount / num_smaller_bills
  denomination

theorem smaller_bills_denomination_correct : denomination_of_smaller_bills = 50 := by
  sorry

end smaller_bills_denomination_correct_l148_148463


namespace local_maximum_at_1_2_l148_148147

noncomputable def f (x1 x2 : ℝ) : ℝ := x2^2 - x1^2
def constraint (x1 x2 : ℝ) : Prop := x1 - 2 * x2 + 3 = 0
def is_local_maximum (f : ℝ → ℝ → ℝ) (x1 x2 : ℝ) : Prop := 
∃ ε > 0, ∀ (y1 y2 : ℝ), (constraint y1 y2 ∧ (y1 - x1)^2 + (y2 - x2)^2 < ε^2) → f y1 y2 ≤ f x1 x2

theorem local_maximum_at_1_2 : is_local_maximum f 1 2 :=
sorry

end local_maximum_at_1_2_l148_148147


namespace smallest_possible_value_of_EF_minus_DE_l148_148098

theorem smallest_possible_value_of_EF_minus_DE :
  ∃ (DE EF FD : ℤ), DE + EF + FD = 2010 ∧ DE < EF ∧ EF ≤ FD ∧ 1 = EF - DE ∧ DE > 0 ∧ EF > 0 ∧ FD > 0 ∧ 
  DE + EF > FD ∧ DE + FD > EF ∧ EF + FD > DE :=
by {
  sorry
}

end smallest_possible_value_of_EF_minus_DE_l148_148098


namespace find_kg_of_mangoes_l148_148529

-- Define the conditions
def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 965
def cost_of_mangoes (m : ℕ) : ℕ := 45 * m

-- Formalize the proof problem
theorem find_kg_of_mangoes (m : ℕ) :
  cost_of_grapes + cost_of_mangoes m = total_amount_paid → m = 9 :=
by
  intros h
  sorry

end find_kg_of_mangoes_l148_148529


namespace max_crates_first_trip_l148_148266

theorem max_crates_first_trip (x : ℕ) : (∀ w, w ≥ 120) ∧ (600 ≥ x * 120) → x = 5 := 
by
  -- Condition: The weight of any crate is no less than 120 kg
  intro h
  have h1 : ∀ w, w ≥ 120 := h.left
  
  -- Condition: The maximum weight for the first trip
  have h2 : 600 ≥ x * 120 := h.right 
  
  -- Derivation of maximum crates
  have h3 : x ≤ 600 / 120 := by sorry  -- This inequality follows from h2 by straightforward division
  
  have h4 : x ≤ 5 := by sorry  -- This follows from evaluating 600 / 120 = 5
  
  -- Knowing x is an integer and the maximum possible value is 5
  exact by sorry

end max_crates_first_trip_l148_148266


namespace union_of_sets_l148_148910

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} := 
by
  sorry

end union_of_sets_l148_148910


namespace remainder_of_polynomial_division_l148_148975

theorem remainder_of_polynomial_division :
  ∀ (x : ℂ), ((x + 2) ^ 2023) % (x^2 + x + 1) = 1 :=
by
  sorry

end remainder_of_polynomial_division_l148_148975


namespace Force_Inversely_Proportional_l148_148565

theorem Force_Inversely_Proportional
  (L₁ F₁ L₂ F₂ : ℝ)
  (h₁ : L₁ = 12)
  (h₂ : F₁ = 480)
  (h₃ : L₂ = 18)
  (h_inv : F₁ * L₁ = F₂ * L₂) :
  F₂ = 320 :=
by
  sorry

end Force_Inversely_Proportional_l148_148565


namespace reservoir_shortage_l148_148427

noncomputable def reservoir_information := 
  let current_level := 14 -- million gallons
  let normal_level_due_to_yield := current_level / 2
  let percentage_of_capacity := 0.70
  let evaporation_factor := 0.90
  let total_capacity := current_level / percentage_of_capacity
  let normal_level_after_evaporation := normal_level_due_to_yield * evaporation_factor
  let shortage := total_capacity - normal_level_after_evaporation
  shortage

theorem reservoir_shortage :
  reservoir_information = 13.7 := 
by
  sorry

end reservoir_shortage_l148_148427


namespace range_of_a_l148_148844

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ (a ≤ 1) :=
by
  sorry

end range_of_a_l148_148844


namespace relationship_among_a_b_c_l148_148637

noncomputable def a : ℝ := Real.logb 0.5 0.2
noncomputable def b : ℝ := Real.logb 2 0.2
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 2)

theorem relationship_among_a_b_c : b < c ∧ c < a :=
by
  sorry

end relationship_among_a_b_c_l148_148637


namespace total_baseball_cards_l148_148917
-- Import the broad Mathlib library

-- The conditions stating the number of cards each person has
def melanie_cards : ℕ := 3
def benny_cards : ℕ := 3
def sally_cards : ℕ := 3
def jessica_cards : ℕ := 3

-- The theorem to prove the total number of cards they have is 12
theorem total_baseball_cards : melanie_cards + benny_cards + sally_cards + jessica_cards = 12 := by
  sorry

end total_baseball_cards_l148_148917


namespace total_cows_l148_148521

def number_of_cows_in_herd : ℕ := 40
def number_of_herds : ℕ := 8
def total_number_of_cows (cows_per_herd herds : ℕ) : ℕ := cows_per_herd * herds

theorem total_cows : total_number_of_cows number_of_cows_in_herd number_of_herds = 320 := by
  sorry

end total_cows_l148_148521


namespace digging_depth_l148_148511

theorem digging_depth :
  (∃ (D : ℝ), 750 * D = 75000) → D = 100 :=
by
  sorry

end digging_depth_l148_148511


namespace reinforcement_correct_l148_148655

-- Conditions
def initial_men : ℕ := 2000
def initial_days : ℕ := 54
def days_before_reinforcement : ℕ := 18
def days_after_reinforcement : ℕ := 20

-- Define the remaining provisions after 18 days
def provisions_left : ℕ := initial_men * (initial_days - days_before_reinforcement)

-- Define reinforcement
def reinforcement : ℕ := 
  sorry -- placeholder for the definition

-- Theorem to prove
theorem reinforcement_correct :
  reinforcement = 1600 :=
by
  -- Use the given conditions to derive the reinforcement value
  let total_provision := initial_men * initial_days
  let remaining_provision := provisions_left
  let men_after_reinforcement := initial_men + reinforcement
  have h := remaining_provision = men_after_reinforcement * days_after_reinforcement
  sorry -- placeholder for the proof

end reinforcement_correct_l148_148655


namespace claire_photos_l148_148025

theorem claire_photos (C L R : ℕ) 
  (h1 : L = 3 * C) 
  (h2 : R = C + 12)
  (h3 : L = R) : C = 6 := 
by
  sorry

end claire_photos_l148_148025


namespace line_plane_parallelism_l148_148731

variables {Point : Type} [LinearOrder Point] -- Assuming Point is a Type with some linear order.

-- Definitions for line and plane
-- These definitions need further libraries or details depending on actual Lean geometry library support
@[ext] structure Line (P : Type) := (contains : P → Prop)
@[ext] structure Plane (P : Type) := (contains : P → Prop)

variables {a b : Line Point} {α β : Plane Point} {l : Line Point}

-- Conditions (as in part a)
axiom lines_are_different : a ≠ b
axiom planes_are_different : α ≠ β
axiom planes_intersect_in_line : ∃ l, α.contains l ∧ β.contains l
axiom a_parallel_l : ∀ p : Point, a.contains p → l.contains p
axiom b_within_plane : ∀ p : Point, b.contains p → β.contains p
axiom b_parallel_alpha : ∀ p q : Point, β.contains p → β.contains q → α.contains p → α.contains q

-- Define the theorem statement
theorem line_plane_parallelism : a ≠ b ∧ α ≠ β ∧ (∃ l, α.contains l ∧ β.contains l) 
  ∧ (∀ p, a.contains p → l.contains p) 
  ∧ (∀ p, b.contains p → β.contains p) 
  ∧ (∀ p q, β.contains p → β.contains q → α.contains p → α.contains q) → a = b :=
by sorry

end line_plane_parallelism_l148_148731


namespace number_of_ordered_triples_l148_148809

noncomputable def count_triples : Nat := 50

theorem number_of_ordered_triples 
    (x y z : Nat)
    (hx : x > 0)
    (hy : y > 0)
    (hz : z > 0)
    (H1 : Nat.lcm x y = 500)
    (H2 : Nat.lcm y z = 1000)
    (H3 : Nat.lcm z x = 1000) :
    ∃ (n : Nat), n = count_triples := 
by
    use 50
    sorry

end number_of_ordered_triples_l148_148809


namespace fraction_sum_eq_l148_148399

-- Given conditions
variables (w x y : ℝ)
axiom hx : w / x = 1 / 6
axiom hy : w / y = 1 / 5

-- Proof goal
theorem fraction_sum_eq : (x + y) / y = 11 / 5 :=
by sorry

end fraction_sum_eq_l148_148399


namespace heights_proportional_l148_148991

-- Define the problem conditions
def sides_ratio (a b c : ℕ) : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5

-- Define the heights
def heights_ratio (h1 h2 h3 : ℕ) : Prop := h1 / h2 = 20 / 15 ∧ h2 / h3 = 15 / 12

-- Problem statement: Given the sides ratio, prove the heights ratio
theorem heights_proportional {a b c h1 h2 h3 : ℕ} (h : sides_ratio a b c) :
  heights_ratio h1 h2 h3 :=
sorry

end heights_proportional_l148_148991


namespace range_of_a_l148_148163

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x^2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 4 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, x0 > 0 ∧ f x0 = g (-x0) a) → a ≤ -1 := 
by
  sorry

end range_of_a_l148_148163


namespace range_of_a_for_increasing_function_l148_148184

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a ^ x

theorem range_of_a_for_increasing_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (3/2 ≤ a ∧ a < 6) := sorry

end range_of_a_for_increasing_function_l148_148184


namespace logic_problem_l148_148610

variables (p q : Prop)

theorem logic_problem (hnp : ¬ p) (hpq : ¬ (p ∧ q)) : ¬ (p ∨ q) ∨ (p ∨ q) :=
by 
  sorry

end logic_problem_l148_148610


namespace range_of_x_l148_148587

section
  variable (f : ℝ → ℝ)

  -- Conditions:
  -- 1. f is an even function
  def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

  -- 2. f is monotonically increasing on [0, +∞)
  def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

  -- Range of x
  def in_range (x : ℝ) : Prop := (1 : ℝ) / 3 < x ∧ x < (2 : ℝ) / 3

  -- Main statement
  theorem range_of_x (f_is_even : is_even f) (f_is_mono : mono_increasing_on_nonneg f) :
    ∀ x, f (2 * x - 1) < f ((1 : ℝ) / 3) ↔ in_range x := 
  by
    sorry
end

end range_of_x_l148_148587


namespace math_problem_solution_l148_148059

noncomputable def problem_statement : Prop :=
  let AB := 4
  let AC := 6
  let BC := 5
  let area_ABC := 9.9216 -- Using the approximated area directly for simplicity
  let K_div3 := area_ABC / 3
  let GP := (2 * K_div3) / BC
  let GQ := (2 * K_div3) / AC
  let GR := (2 * K_div3) / AB
  GP + GQ + GR = 4.08432

theorem math_problem_solution : problem_statement :=
by
  sorry

end math_problem_solution_l148_148059


namespace trapezium_area_l148_148505

theorem trapezium_area (a b d : ℕ) (h₁ : a = 28) (h₂ : b = 18) (h₃ : d = 15) :
  (a + b) * d / 2 = 345 := by
{
  sorry
}

end trapezium_area_l148_148505


namespace find_total_bricks_l148_148545

variable (y : ℕ)
variable (B_rate : ℕ)
variable (N_rate : ℕ)
variable (eff_rate : ℕ)
variable (time : ℕ)
variable (reduction : ℕ)

-- The wall is completed in 6 hours
def completed_in_time (y B_rate N_rate eff_rate time reduction : ℕ) : Prop := 
  time = 6 ∧
  reduction = 8 ∧
  B_rate = y / 8 ∧
  N_rate = y / 12 ∧
  eff_rate = (B_rate + N_rate) - reduction ∧
  y = eff_rate * time

-- Prove that the number of bricks in the wall is 192
theorem find_total_bricks : 
  ∀ (y B_rate N_rate eff_rate time reduction : ℕ), 
  completed_in_time y B_rate N_rate eff_rate time reduction → 
  y = 192 := 
by 
  sorry

end find_total_bricks_l148_148545


namespace missing_fraction_is_correct_l148_148829

def sum_of_fractions (x : ℚ) : Prop :=
  (1/3 : ℚ) + (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + x = (45/100 : ℚ)

theorem missing_fraction_is_correct : sum_of_fractions (27/60 : ℚ) :=
  by sorry

end missing_fraction_is_correct_l148_148829


namespace dogsled_race_time_difference_l148_148785

theorem dogsled_race_time_difference :
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  T_W - T_A = 3 :=
by
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  sorry

end dogsled_race_time_difference_l148_148785


namespace people_in_line_l148_148618

theorem people_in_line (initially_in_line : ℕ) (left_line : ℕ) (after_joined_line : ℕ) 
  (h1 : initially_in_line = 12) (h2 : left_line = 10) (h3 : after_joined_line = 17) : 
  initially_in_line - left_line + 15 = after_joined_line := by
  sorry

end people_in_line_l148_148618


namespace find_p_probability_of_match_ending_after_4_games_l148_148812

variables (p : ℚ)

-- Conditions translated to Lean definitions
def probability_first_game_win : ℚ := 1 / 2

def probability_consecutive_games_win : ℚ := 5 / 16

-- Definitions based on conditions
def prob_second_game_win_if_won_first : ℚ := (1 + p) / 2

def prob_winning_consecutive_games (prob_first_game : ℚ) (prob_second_game_if_won_first : ℚ) : ℚ :=
prob_first_game * prob_second_game_if_won_first

-- Main Theorem Statements to be proved
theorem find_p 
    (h_eq : prob_winning_consecutive_games probability_first_game_win (prob_second_game_win_if_won_first p) = probability_consecutive_games_win) :
    p = 1 / 4 :=
sorry

-- Given p = 1/4, probabilities for each scenario the match ends after 4 games
def prob_scenario1 : ℚ := (1 / 2) * ((1 + 1/4) / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2)
def prob_scenario2 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2)
def prob_scenario3 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2) * ((1 + 1/4) / 2)

def total_probability_ending_in_4_games : ℚ :=
2 * (prob_scenario1 + prob_scenario2 + prob_scenario3)

theorem probability_of_match_ending_after_4_games (hp : p = 1 / 4) :
    total_probability_ending_in_4_games = 165 / 512 :=
sorry

end find_p_probability_of_match_ending_after_4_games_l148_148812


namespace expected_value_of_geometric_variance_of_geometric_l148_148322

noncomputable def expected_value (p : ℝ) : ℝ :=
  1 / p

noncomputable def variance (p : ℝ) : ℝ :=
  (1 - p) / (p ^ 2)

theorem expected_value_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, (n + 1 : ℝ) * (1 - p) ^ n * p = expected_value p := by
  sorry

theorem variance_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, ((n + 1 : ℝ) ^ 2) * (1 - p) ^ n * p - (expected_value p) ^ 2 = variance p := by
  sorry

end expected_value_of_geometric_variance_of_geometric_l148_148322


namespace a1_greater_than_500_l148_148283

-- Set up conditions
variables (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n ∧ a n < 20000)
variables (h2 : ∀ i j, i < j → gcd (a i) (a j) < a i)
variables (h3 : ∀ i j, i < j ∧ 1 ≤ i ∧ j ≤ 10000 → a i < a j)

/-- Statement to prove / lean concept as per mathematical problem  --/
theorem a1_greater_than_500 : 500 < a 1 :=
sorry

end a1_greater_than_500_l148_148283


namespace calculate_A_share_l148_148902

variable (x : ℝ) (total_gain : ℝ)
variable (h_b_invests : 2 * x)  -- B invests double the amount after 6 months
variable (h_c_invests : 3 * x)  -- C invests thrice the amount after 8 months

/-- Calculate the share of A from the total annual gain -/
theorem calculate_A_share (h_total_gain : total_gain = 18600) :
  let a_investmentMonths := x * 12
  let b_investmentMonths := (2 * x) * 6
  let c_investmentMonths := (3 * x) * 4
  let total_investmentMonths := a_investmentMonths + b_investmentMonths + c_investmentMonths
  let a_share := (a_investmentMonths / total_investmentMonths) * total_gain
  a_share = 6200 :=
by
  sorry

end calculate_A_share_l148_148902


namespace round_trip_percentage_l148_148404

-- Definitions based on the conditions
variable (P : ℝ) -- Total number of passengers
variable (R : ℝ) -- Number of round-trip ticket holders

-- First condition: 20% of passengers held round-trip tickets and took their cars aboard
def condition1 := 0.20 * P = 0.60 * R

-- Second condition: 40% of passengers with round-trip tickets did not take their cars aboard (implies 60% did)
theorem round_trip_percentage (h1 : condition1 P R) : (R / P) * 100 = 33.33 := by
  sorry

end round_trip_percentage_l148_148404


namespace first_player_wins_l148_148676

-- Define the initial conditions
def initial_pieces : ℕ := 1
def final_pieces (m n : ℕ) : ℕ := m * n
def num_moves (pieces : ℕ) : ℕ := pieces - 1

-- Theorem statement: Given the initial dimensions and the game rules,
-- prove that the first player will win.
theorem first_player_wins (m n : ℕ) (h_m : m = 6) (h_n : n = 8) : 
  (num_moves (final_pieces m n)) % 2 = 0 → false :=
by
  -- The solution details and the proof will be here.
  sorry

end first_player_wins_l148_148676


namespace tan_angle_addition_l148_148668

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 2) : Real.tan (x + Real.pi / 3) = (5 * Real.sqrt 3 + 8) / -11 := by
  sorry

end tan_angle_addition_l148_148668


namespace simplify_expression_l148_148073

theorem simplify_expression :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := 
by
  sorry

end simplify_expression_l148_148073


namespace space_is_volume_stuff_is_capacity_film_is_surface_area_l148_148516

-- Let's define the properties based on the conditions
def size_of_space (box : Type) : Type := 
  sorry -- This will be volume later

def stuff_can_hold (box : Type) : Type :=
  sorry -- This will be capacity later

def film_needed_to_cover (box : Type) : Type :=
  sorry -- This will be surface area later

-- Now prove the correspondences
theorem space_is_volume (box : Type) :
  size_of_space box = volume := 
by 
  sorry

theorem stuff_is_capacity (box : Type) :
  stuff_can_hold box = capacity := 
by 
  sorry

theorem film_is_surface_area (box : Type) :
  film_needed_to_cover box = surface_area := 
by 
  sorry

end space_is_volume_stuff_is_capacity_film_is_surface_area_l148_148516


namespace part_a_part_b_l148_148346

theorem part_a (n : ℕ) (hn : n % 2 = 1) (h_pos : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n-1 ∧ ∃ f : (ℕ → ℕ), f k ≥ (n - 1) / 2 :=
sorry

theorem part_b : ∃ᶠ n in at_top, ∃ f : (ℕ → ℕ), ∀ k : ℕ, 1 ≤ k ∧ k ≤ n-1 → f k ≤ (n - 1) / 2 :=
sorry

end part_a_part_b_l148_148346


namespace mixed_doubles_teams_l148_148842

theorem mixed_doubles_teams (males females : ℕ) (hm : males = 6) (hf : females = 7) : (males * females) = 42 :=
by
  sorry

end mixed_doubles_teams_l148_148842


namespace angle_AOC_is_minus_150_l148_148329

-- Define the conditions.
def rotate_counterclockwise (angle1 : Int) (angle2 : Int) : Int :=
  angle1 + angle2

-- The initial angle starts at 0°, rotates 120° counterclockwise, and then 270° clockwise
def angle_OA := 0
def angle_OB := rotate_counterclockwise angle_OA 120
def angle_OC := rotate_counterclockwise angle_OB (-270)

-- The theorem stating the resulting angle between OA and OC.
theorem angle_AOC_is_minus_150 : angle_OC = -150 := by
  sorry

end angle_AOC_is_minus_150_l148_148329


namespace domain_of_function_l148_148681

theorem domain_of_function :
  { x : ℝ | 0 ≤ 2 * x - 10 ∧ 2 * x - 10 ≠ 0 } = { x : ℝ | x > 5 } :=
by
  sorry

end domain_of_function_l148_148681


namespace sin_2B_minus_5pi_over_6_area_of_triangle_l148_148162

-- Problem (I)
theorem sin_2B_minus_5pi_over_6 {A B C : ℝ} (a b c : ℝ)
  (h: 3 * Real.cos A * Real.cos C * (Real.tan A * Real.tan C - 1) = 1) :
  Real.sin (2 * B - 5 * Real.pi / 6) = (7 - 4 * Real.sqrt 6) / 18 :=
sorry

-- Problem (II)
theorem area_of_triangle {A B C : ℝ} (a b c : ℝ)
  (h1: a + c = 3 * Real.sqrt 3 / 2) (h2: b = Real.sqrt 3) :
  Real.sqrt (a * c) * Real.sin B / 2 = 15 * Real.sqrt 2 / 32 :=
sorry

end sin_2B_minus_5pi_over_6_area_of_triangle_l148_148162


namespace bus_dispatch_interval_l148_148952

-- Variables representing the speeds of Xiao Nan and the bus
variable (V_1 V_2 : ℝ)
-- The interval between the dispatch of two buses
variable (interval : ℝ)

-- Stating the conditions in Lean

-- Xiao Nan notices a bus catches up with him every 10 minutes
def cond1 : Prop := ∃ s, s = 10 * (V_1 - V_2)

-- Xiao Yu notices he encounters a bus every 5 minutes
def cond2 : Prop := ∃ s, s = 5 * (V_1 + 3 * V_2)

-- Proof statement
theorem bus_dispatch_interval (h1 : cond1 V_1 V_2) (h2 : cond2 V_1 V_2) : interval = 8 := by
  -- Proof would be provided here
  sorry

end bus_dispatch_interval_l148_148952


namespace tangents_from_point_to_circle_l148_148603

theorem tangents_from_point_to_circle (x y k : ℝ) (
    P : ℝ × ℝ)
    (h₁ : P = (1, -1))
    (circle_eq : x^2 + y^2 + 2*x + 2*y + k = 0)
    (h₂ : P = (1, -1))
    (has_two_tangents : 1^2 + (-1)^2 - k / 2 > 0):
  -2 < k ∧ k < 2 :=
by 
    sorry

end tangents_from_point_to_circle_l148_148603


namespace find_number_that_gives_200_9_when_8_036_divided_by_it_l148_148160

theorem find_number_that_gives_200_9_when_8_036_divided_by_it (
  x : ℝ
) : (8.036 / x = 200.9) → (x = 0.04) :=
by
  intro h
  sorry

end find_number_that_gives_200_9_when_8_036_divided_by_it_l148_148160


namespace find_F_neg_a_l148_148923

-- Definitions of odd functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of F
def F (f g : ℝ → ℝ) (x : ℝ) := 3 * f x + 5 * g x + 2

theorem find_F_neg_a (f g : ℝ → ℝ) (a : ℝ)
  (hf : is_odd f) (hg : is_odd g) (hFa : F f g a = 3) : F f g (-a) = 1 :=
by
  sorry

end find_F_neg_a_l148_148923


namespace minimum_unused_area_for_given_shapes_l148_148453

def remaining_area (side_length : ℕ) (total_area used_area : ℕ) : ℕ :=
  total_area - used_area

theorem minimum_unused_area_for_given_shapes : (remaining_area 5 (5 * 5) (2 * 2 + 1 * 3 + 2 * 1) = 16) :=
by
  -- We skip the proof here, as instructed.
  sorry

end minimum_unused_area_for_given_shapes_l148_148453


namespace parallel_vectors_x_value_l148_148374

def vectors_are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, vectors_are_parallel (-1, 4) (x, 2) → x = -1 / 2 := 
by 
  sorry

end parallel_vectors_x_value_l148_148374


namespace probability_multiple_of_100_is_zero_l148_148462

def singleDigitMultiplesOf5 : Set ℕ := {5}
def primeNumbersLessThan50 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
def isMultipleOf100 (n : ℕ) : Prop := 100 ∣ n

theorem probability_multiple_of_100_is_zero :
  (∀ m ∈ singleDigitMultiplesOf5, ∀ p ∈ primeNumbersLessThan50, ¬ isMultipleOf100 (m * p)) →
  r = 0 :=
sorry

end probability_multiple_of_100_is_zero_l148_148462


namespace primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l148_148409

-- Part 1: Prove that every prime number >= 3 is of the form 4k-1 or 4k+1
theorem primes_ge_3_are_4k_pm1 (p : ℕ) (hp_prime: Nat.Prime p) (hp_ge_3: p ≥ 3) : 
  ∃ k : ℕ, p = 4 * k + 1 ∨ p = 4 * k - 1 :=
by
  sorry

-- Part 2: Prove that there are infinitely many primes of the form 4k-1
theorem infinitely_many_primes_4k_minus1 : 
  ∀ (n : ℕ), ∃ (p : ℕ), Nat.Prime p ∧ p = 4 * k - 1 ∧ p > n :=
by
  sorry

end primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l148_148409


namespace compound_interest_principal_l148_148136

theorem compound_interest_principal 
  (CI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hCI : CI = 315)
  (hR : R = 10)
  (hT : T = 2) :
  CI = P * ((1 + R / 100)^T - 1) → P = 1500 := by
  sorry

end compound_interest_principal_l148_148136


namespace max_min_values_of_f_l148_148813

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_min_values_of_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ x : ℝ, f x = 2) ∧
  (∀ x : ℝ, -2 ≤ f x) ∧ (∃ x : ℝ, f x = -2) :=
by 
  sorry

end max_min_values_of_f_l148_148813


namespace constants_solution_l148_148117

theorem constants_solution : ∀ (x : ℝ), x ≠ 0 ∧ x^2 ≠ 2 →
  (2 * x^2 - 5 * x + 1) / (x^3 - 2 * x) = (-1 / 2) / x + (2.5 * x - 5) / (x^2 - 2) := by
  intros x hx
  sorry

end constants_solution_l148_148117


namespace average_speed_additional_hours_l148_148631

theorem average_speed_additional_hours
  (time_first_part : ℝ) (speed_first_part : ℝ) (total_time : ℝ) (avg_speed_total : ℝ)
  (additional_hours : ℝ) (speed_additional_hours : ℝ) :
  time_first_part = 4 → speed_first_part = 35 → total_time = 24 → avg_speed_total = 50 →
  additional_hours = total_time - time_first_part →
  (time_first_part * speed_first_part + additional_hours * speed_additional_hours) / total_time = avg_speed_total →
  speed_additional_hours = 53 :=
by intros; sorry

end average_speed_additional_hours_l148_148631


namespace sum_of_first_five_terms_sequence_l148_148496

-- Definitions derived from conditions
def seventh_term : ℤ := 4
def eighth_term : ℤ := 10
def ninth_term : ℤ := 16

-- The main theorem statement
theorem sum_of_first_five_terms_sequence : 
  ∃ (a d : ℤ), 
    a + 6 * d = seventh_term ∧
    a + 7 * d = eighth_term ∧
    a + 8 * d = ninth_term ∧
    (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = -100) :=
by
  sorry

end sum_of_first_five_terms_sequence_l148_148496


namespace M_inter_N_l148_148527

namespace ProofProblem

def M : Set ℝ := { x | 3 * x - x^2 > 0 }
def N : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem M_inter_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
sorry

end ProofProblem

end M_inter_N_l148_148527


namespace compressor_distances_distances_when_a_15_l148_148355

theorem compressor_distances (a : ℝ) (x y z : ℝ) (h1 : x + y = 2 * z) (h2 : x + z = y + a) (h3 : x + z = 75) :
  0 < a ∧ a < 100 → 
  let x := (75 + a) / 3;
  let y := 75 - a;
  let z := 75 - x;
  x + y = 2 * z ∧ x + z = y + a ∧ x + z = 75 :=
sorry

theorem distances_when_a_15 (x y z : ℝ) (h : 15 = 15) :
  let x := (75 + 15) / 3;
  let y := 75 - 15;
  let z := 75 - x;
  x = 30 ∧ y = 60 ∧ z = 45 :=
sorry

end compressor_distances_distances_when_a_15_l148_148355


namespace solve_for_x_l148_148729

theorem solve_for_x (x : ℝ) (h : (3 / 4) + (1 / x) = 7 / 8) : x = 8 :=
sorry

end solve_for_x_l148_148729


namespace garden_ratio_l148_148314

theorem garden_ratio (L W : ℕ) (h1 : L = 50) (h2 : 2 * L + 2 * W = 150) : L / W = 2 :=
by
  sorry

end garden_ratio_l148_148314


namespace larger_number_is_22_l148_148362

theorem larger_number_is_22 (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 :=
by
  sorry

end larger_number_is_22_l148_148362


namespace centroid_triangle_PQR_l148_148965

theorem centroid_triangle_PQR (P Q R S : ℝ × ℝ) 
  (P_coord : P = (2, 5)) 
  (Q_coord : Q = (9, 3)) 
  (R_coord : R = (4, -4))
  (S_is_centroid : S = (
    (P.1 + Q.1 + R.1) / 3,
    (P.2 + Q.2 + R.2) / 3)) :
  9 * S.1 + 4 * S.2 = 151 / 3 :=
by
  sorry

end centroid_triangle_PQR_l148_148965


namespace roof_ratio_l148_148350

theorem roof_ratio (L W : ℝ) (h1 : L * W = 576) (h2 : L - W = 36) : L / W = 4 := 
by
  sorry

end roof_ratio_l148_148350


namespace log_expression_evaluation_l148_148484

noncomputable def log2 : ℝ := Real.log 2
noncomputable def log5 : ℝ := Real.log 5

theorem log_expression_evaluation (condition : log2 + log5 = 1) :
  log2^2 + log2 * log5 + log5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  sorry

end log_expression_evaluation_l148_148484


namespace sum_of_reciprocals_factors_12_l148_148142

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l148_148142


namespace trader_gain_percentage_is_25_l148_148713

noncomputable def trader_gain_percentage (C : ℝ) : ℝ :=
  ((22 * C) / (88 * C)) * 100

theorem trader_gain_percentage_is_25 (C : ℝ) (h : C ≠ 0) : trader_gain_percentage C = 25 := by
  unfold trader_gain_percentage
  field_simp [h]
  norm_num
  sorry

end trader_gain_percentage_is_25_l148_148713


namespace arithmetic_progression_numbers_l148_148883

theorem arithmetic_progression_numbers :
  ∃ (a d : ℚ), (3 * (2 * a - d) = 2 * (a + d)) ∧ ((a - d) * (a + d) = (a - 2)^2) ∧
  ((a = 5 ∧ d = 4 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 ∧ c = 9) 
   ∨ (a = 5 / 4 ∧ d = 1 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 / 4 ∧ c = 9 / 4)) :=
by
  sorry

end arithmetic_progression_numbers_l148_148883


namespace unique_solution_of_quadratic_l148_148878

theorem unique_solution_of_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9 / 8) :=
by
  sorry

end unique_solution_of_quadratic_l148_148878


namespace jeffrey_fills_crossword_l148_148994

noncomputable def prob_fill_crossword : ℚ :=
  let total_clues := 10
  let prob_knowing_all_clues := (1 / 2) ^ total_clues
  let prob_case_1 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_2 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_3 := 25 / (2 ^ total_clues)
  let overcounted_case := prob_knowing_all_clues
  (prob_case_1 + prob_case_2 + prob_case_3 - overcounted_case)

theorem jeffrey_fills_crossword : prob_fill_crossword = 11 / 128 := by
  sorry

end jeffrey_fills_crossword_l148_148994


namespace calories_consumed_in_week_l148_148316

-- Define the calorie content of each type of burger
def calorie_A := 350
def calorie_B := 450
def calorie_C := 550

-- Define Dimitri's burger consumption over the 7 days
def consumption_day1 := (2 * calorie_A) + (1 * calorie_B)
def consumption_day2 := (1 * calorie_A) + (2 * calorie_B) + (1 * calorie_C)
def consumption_day3 := (1 * calorie_A) + (1 * calorie_B) + (2 * calorie_C)
def consumption_day4 := (3 * calorie_B)
def consumption_day5 := (1 * calorie_A) + (1 * calorie_B) + (1 * calorie_C)
def consumption_day6 := (2 * calorie_A) + (3 * calorie_C)
def consumption_day7 := (1 * calorie_B) + (2 * calorie_C)

-- Define the total weekly calorie consumption
def total_weekly_calories :=
  consumption_day1 + consumption_day2 + consumption_day3 +
  consumption_day4 + consumption_day5 + consumption_day6 + consumption_day7

-- State and prove the main theorem
theorem calories_consumed_in_week :
  total_weekly_calories = 11450 := 
by
  sorry

end calories_consumed_in_week_l148_148316


namespace downstream_speed_l148_148962

variable (Vu : ℝ) (Vs : ℝ)

theorem downstream_speed (h1 : Vu = 25) (h2 : Vs = 35) : (2 * Vs - Vu = 45) :=
by
  sorry

end downstream_speed_l148_148962


namespace cubic_eq_root_nature_l148_148185

-- Definitions based on the problem statement
def cubic_eq (x : ℝ) : Prop := x^3 + 3 * x^2 - 4 * x - 12 = 0

-- The main theorem statement
theorem cubic_eq_root_nature :
  (∃ p n₁ n₂ : ℝ, cubic_eq p ∧ cubic_eq n₁ ∧ cubic_eq n₂ ∧ p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧ p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂) :=
sorry

end cubic_eq_root_nature_l148_148185


namespace remaining_regular_toenails_l148_148464

def big_toenail_space := 2
def total_capacity := 100
def big_toenails_count := 20
def regular_toenails_count := 40

theorem remaining_regular_toenails : 
  total_capacity - (big_toenails_count * big_toenail_space + regular_toenails_count) = 20 := by
  sorry

end remaining_regular_toenails_l148_148464


namespace area_of_cross_l148_148501

-- Definitions based on the conditions
def congruent_squares (n : ℕ) := n = 5
def perimeter_of_cross (p : ℕ) := p = 72

-- Targeting the proof that the area of the cross formed by the squares is 180 square units
theorem area_of_cross (n p : ℕ) (h1 : congruent_squares n) (h2 : perimeter_of_cross p) : 
  5 * (p / 12) ^ 2 = 180 := 
by 
  sorry

end area_of_cross_l148_148501


namespace clock_angle_at_3_20_is_160_l148_148873

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l148_148873


namespace total_cost_of_bicycles_is_2000_l148_148182

noncomputable def calculate_total_cost_of_bicycles (SP1 SP2 : ℝ) (profit1 profit2 : ℝ) : ℝ :=
  let C1 := SP1 / (1 + profit1)
  let C2 := SP2 / (1 - profit2)
  C1 + C2

theorem total_cost_of_bicycles_is_2000 :
  calculate_total_cost_of_bicycles 990 990 0.10 0.10 = 2000 :=
by
  -- Proof will be provided here
  sorry

end total_cost_of_bicycles_is_2000_l148_148182


namespace order_of_fractions_l148_148933

theorem order_of_fractions (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0)
(hab : a > b) : (b / a) < (b + c) / (a + c) ∧ (b + c) / (a + c) < (a + d) / (b + d) ∧ (a + d) / (b + d) < (a / b) :=
by
  sorry

end order_of_fractions_l148_148933


namespace first_group_men_8_l148_148120

variable (x : ℕ)

theorem first_group_men_8 (h1 : x * 80 = 20 * 32) : x = 8 := by
  -- provide the proof here
  sorry

end first_group_men_8_l148_148120


namespace system_of_linear_equations_m_l148_148988

theorem system_of_linear_equations_m (x y m : ℝ) :
  (2 * x + y = 1 + 2 * m) →
  (x + 2 * y = 2 - m) →
  (x + y > 0) →
  ((2 * m + 1) * x - 2 * m < 1) →
  (x > 1) →
  (-3 < m ∧ m < -1/2) ∧ (m = -2 ∨ m = -1) :=
by
  intros h1 h2 h3 h4 h5
  -- Placeholder for proof steps
  sorry

end system_of_linear_equations_m_l148_148988


namespace number_of_integers_satisfying_l148_148717

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end number_of_integers_satisfying_l148_148717


namespace arianna_sleeping_hours_l148_148840

def hours_in_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_on_chores : ℕ := 5
def hours_sleeping : ℕ := hours_in_day - (hours_at_work + hours_on_chores)

theorem arianna_sleeping_hours : hours_sleeping = 13 := by
  sorry

end arianna_sleeping_hours_l148_148840


namespace y_neither_directly_nor_inversely_proportional_l148_148907

theorem y_neither_directly_nor_inversely_proportional (x y : ℝ) :
  ¬((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) ↔ 2 * x + 3 * y = 6 :=
by 
  sorry

end y_neither_directly_nor_inversely_proportional_l148_148907


namespace locus_eqn_l148_148304

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  ∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 2)^2 + b^2 = (5 - r)^2)

theorem locus_eqn (a b : ℝ) : 
  locus_of_centers a b ↔ 3 * a^2 + b^2 + 44 * a + 121 = 0 :=
by
  -- Proof omitted
  sorry

end locus_eqn_l148_148304


namespace find_a_l148_148443

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = 3 * x^(a-2) - 2) (h_cond : f 2 = 4) : a = 3 :=
by
  sorry

end find_a_l148_148443


namespace expected_faces_rolled_six_times_l148_148832

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l148_148832


namespace total_money_shared_l148_148122

/-- Assume there are four people Amanda, Ben, Carlos, and David, sharing an amount of money.
    Their portions are in the ratio 1:2:7:3.
    Amanda's portion is $20.
    Prove that the total amount of money shared by them is $260. -/
theorem total_money_shared (A B C D : ℕ) (h_ratio : A = 20 ∧ B = 2 * A ∧ C = 7 * A ∧ D = 3 * A) :
  A + B + C + D = 260 := by 
  sorry

end total_money_shared_l148_148122


namespace find_original_number_l148_148492

def digitsGPA (A B C : ℕ) : Prop := B^2 = A * C
def digitsAPA (X Y Z : ℕ) : Prop := 2 * Y = X + Z

theorem find_original_number (A B C X Y Z : ℕ) :
  100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C ≤ 999 ∧
  digitsGPA A B C ∧
  100 * X + 10 * Y + Z = (100 * A + 10 * B + C) - 200 ∧
  digitsAPA X Y Z →
  (100 * A + 10 * B + C) = 842 :=
sorry

end find_original_number_l148_148492


namespace max_product_xy_l148_148573

theorem max_product_xy (x y : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 7 * x + 4 * y = 150) : xy = 200 :=
by
  sorry

end max_product_xy_l148_148573


namespace find_daily_wage_of_c_l148_148719

def dailyWagesInRatio (a b c : ℕ) : Prop :=
  4 * a = 3 * b ∧ 5 * a = 3 * c

def totalEarnings (a b c : ℕ) (total : ℕ) : Prop :=
  6 * a + 9 * b + 4 * c = total

theorem find_daily_wage_of_c (a b c : ℕ) (total : ℕ) 
  (h1 : dailyWagesInRatio a b c) 
  (h2 : totalEarnings a b c total) 
  (h3 : total = 1406) : 
  c = 95 :=
by
  -- We assume the conditions and solve the required proof.
  sorry

end find_daily_wage_of_c_l148_148719


namespace smallest_nine_digit_times_smallest_seven_digit_l148_148446

theorem smallest_nine_digit_times_smallest_seven_digit :
  let smallest_nine_digit := 100000000
  let smallest_seven_digit := 1000000
  smallest_nine_digit = 100 * smallest_seven_digit :=
by
  sorry

end smallest_nine_digit_times_smallest_seven_digit_l148_148446


namespace monotonicity_intervals_f_above_g_l148_148017

noncomputable def f (x m : ℝ) := (Real.exp x) / (x^2 - m * x + 1)

theorem monotonicity_intervals (m : ℝ) (h : m ∈ Set.Ioo (-2 : ℝ) 2) :
  (m = 0 → ∀ x y : ℝ, x ≤ y → f x m ≤ f y m) ∧ 
  (0 < m ∧ m < 2 → ∀ x : ℝ, (x < 1 → f x m < f (x + 1) m) ∧
    (1 < x ∧ x < m + 1 → f x m > f (x + 1) m) ∧
    (x > m + 1 → f x m < f (x + 1) m)) ∧
  (-2 < m ∧ m < 0 → ∀ x : ℝ, (x < m + 1 → f x m < f (x + 1) m) ∧
    (m + 1 < x ∧ x < 1 → f x m > f (x + 1) m) ∧
    (x > 1 → f x m < f (x + 1) m)) :=
sorry

theorem f_above_g (m : ℝ) (hm : m ∈ Set.Ioo (0 : ℝ) (1/2 : ℝ)) (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (m + 1)) :
  f x m > x :=
sorry

end monotonicity_intervals_f_above_g_l148_148017


namespace find_a_equiv_l148_148454

noncomputable def A (a : ℝ) : Set ℝ := {1, 3, a^2}
noncomputable def B (a : ℝ) : Set ℝ := {1, 2 + a}

theorem find_a_equiv (a : ℝ) (h : A a ∪ B a = A a) : a = 2 :=
by
  sorry

end find_a_equiv_l148_148454


namespace part1_l148_148978

def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

def setB (x : ℝ) : Prop := x ≠ 0 ∧ x ≤ 5 ∧ 0 < x

def setC (a x : ℝ) : Prop := 3 * a ≤ x ∧ x ≤ 2 * a + 1

def setInter (x : ℝ) : Prop := setA x ∧ setB x

theorem part1 (a : ℝ) : (∀ x, setC a x → setInter x) ↔ (0 < a ∧ a ≤ 1 / 2 ∨ 1 < a) :=
sorry

end part1_l148_148978


namespace smallest_x_for_equation_l148_148998

theorem smallest_x_for_equation :
  ∃ x : ℝ, x = -15 ∧ (∀ y : ℝ, 3*y^2 + 39*y - 75 = y*(y + 16) → x ≤ y) ∧ 
  3*(-15)^2 + 39*(-15) - 75 = -15*(-15 + 16) :=
sorry

end smallest_x_for_equation_l148_148998


namespace percentage_increase_direct_proportionality_l148_148461

variable (x y k q : ℝ)
variable (h1 : x = k * y)
variable (h2 : x' = x * (1 + q / 100))

theorem percentage_increase_direct_proportionality :
  ∃ q_percent : ℝ, y' = y * (1 + q_percent / 100) ∧ q_percent = q := sorry

end percentage_increase_direct_proportionality_l148_148461


namespace no_duplicate_among_expressions_l148_148413

theorem no_duplicate_among_expressions
  (N a1 a2 b1 b2 c1 c2 d1 d2 : ℕ)
  (ha : a1 = x^2)
  (hb : b1 = y^3)
  (hc : c1 = z^5)
  (hd : d1 = w^7)
  (ha2 : a2 = m^2)
  (hb2 : b2 = n^3)
  (hc2 : c2 = p^5)
  (hd2 : d2 = q^7)
  (h1 : N = a1 - a2)
  (h2 : N = b1 - b2)
  (h3 : N = c1 - c2)
  (h4 : N = d1 - d2) :
  ¬ (a1 = b1 ∨ a1 = c1 ∨ a1 = d1 ∨ b1 = c1 ∨ b1 = d1 ∨ c1 = d1) :=
by
  -- Begin proof here
  sorry

end no_duplicate_among_expressions_l148_148413


namespace shortest_distance_to_circle_l148_148482

variable (A O T : Type)
variable (r d : ℝ)
variable [MetricSpace A]
variable [MetricSpace O]
variable [MetricSpace T]

open Real

theorem shortest_distance_to_circle (h : d = (4 / 3) * r) : 
  OA = (5 / 3) * r → shortest_dist = (2 / 3) * r :=
by
  sorry

end shortest_distance_to_circle_l148_148482


namespace least_three_digit_multiple_of_3_4_7_l148_148411

theorem least_three_digit_multiple_of_3_4_7 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 → m % 3 = 0 ∧ m % 4 = 0 ∧ m % 7 = 0 → n ≤ m :=
  sorry

end least_three_digit_multiple_of_3_4_7_l148_148411


namespace domain_of_f_exp_l148_148563

theorem domain_of_f_exp (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x + 1 ∧ x + 1 < 4 → ∃ y, f y = f (x + 1)) →
  (∀ x, 1 ≤ 2^x ∧ 2^x < 4 → ∃ y, f y = f (2^x)) :=
by
  sorry

end domain_of_f_exp_l148_148563


namespace trig_identity_solution_l148_148009

theorem trig_identity_solution
  (α : ℝ) (β : ℝ)
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan β = -1 / 3) :
  (3 * Real.sin α * Real.cos β - Real.sin β * Real.cos α) / (Real.cos α * Real.cos β + 2 * Real.sin α * Real.sin β) = 11 / 4 :=
by
  sorry

end trig_identity_solution_l148_148009


namespace correct_sum_of_integers_l148_148798

theorem correct_sum_of_integers
  (x y : ℕ)
  (h1 : x - y = 5)
  (h2 : x * y = 84) :
  x + y = 19 :=
sorry

end correct_sum_of_integers_l148_148798


namespace trig_comparison_l148_148862

theorem trig_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) :
  a = Real.sin (3 * Real.pi / 5) → 
  b = Real.cos (2 * Real.pi / 5) → 
  c = Real.tan (2 * Real.pi / 5) → 
  b < a ∧ a < c :=
by
  intro ha hb hc
  sorry

end trig_comparison_l148_148862


namespace isosceles_triangle_largest_angle_l148_148851

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_triangle : A + B + C = 180) 
  (h_isosceles : A = B) (h_given_angle : A = 40) : C = 100 :=
by
  sorry

end isosceles_triangle_largest_angle_l148_148851


namespace minimum_unit_cubes_l148_148062

theorem minimum_unit_cubes (n : ℕ) (N : ℕ) : 
  (n ≥ 3) → (N = n^3) → ((n - 2)^3 > (1/2) * n^3) → 
  ∃ n : ℕ, N = n^3 ∧ (n - 2)^3 > (1/2) * n^3 ∧ N = 1000 :=
by
  intros
  sorry

end minimum_unit_cubes_l148_148062


namespace find_sin_beta_l148_148706

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π/2) -- α is acute
variable (hβ : 0 < β ∧ β < π/2) -- β is acute

variable (hcosα : Real.cos α = 4/5)
variable (hcosαβ : Real.cos (α + β) = 5/13)

theorem find_sin_beta (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
    (hcosα : Real.cos α = 4/5) (hcosαβ : Real.cos (α + β) = 5/13) : 
    Real.sin β = 33/65 := 
sorry

end find_sin_beta_l148_148706


namespace knights_on_red_chairs_l148_148458

theorem knights_on_red_chairs (K L K_r L_b : ℕ) (h1: K + L = 20)
  (h2: K - K_r + L_b = 10) (h3: K_r + L - L_b = 10) (h4: K_r = L_b) : K_r = 5 := by
  sorry

end knights_on_red_chairs_l148_148458


namespace angle_sum_90_l148_148752

theorem angle_sum_90 (A B : ℝ) (h : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2) : A + B = Real.pi / 2 :=
sorry

end angle_sum_90_l148_148752


namespace unique_a_for_fx_eq_2ax_l148_148174

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * Real.log x

theorem unique_a_for_fx_eq_2ax (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, f x a = 2 * a * x → x = (a + Real.sqrt (a^2 + 4 * a)) / 2) →
  a = 1 / 2 :=
sorry

end unique_a_for_fx_eq_2ax_l148_148174


namespace function_intersection_le_one_l148_148351

theorem function_intersection_le_one (f : ℝ → ℝ)
  (h : ∀ x t : ℝ, t ≠ 0 → t * (f (x + t) - f x) > 0) :
  ∀ a : ℝ, ∃! x : ℝ, f x = a :=
by 
sorry

end function_intersection_le_one_l148_148351


namespace man_speed_is_4_kmph_l148_148745

noncomputable def speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass_seconds : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := train_length / time_to_pass_seconds
  let relative_speed_kmph := relative_speed_mps * 3600 / 1000
  relative_speed_kmph - train_speed_kmph

theorem man_speed_is_4_kmph : speed_of_man 140 50 9.332586726395222 = 4 := by
  sorry

end man_speed_is_4_kmph_l148_148745


namespace calculate_pens_l148_148627

theorem calculate_pens (P : ℕ) (Students : ℕ) (Pencils : ℕ) (h1 : Students = 40) (h2 : Pencils = 920) (h3 : ∃ k : ℕ, Pencils = Students * k) 
(h4 : ∃ m : ℕ, P = Students * m) : ∃ k : ℕ, P = 40 * k := by
  sorry

end calculate_pens_l148_148627


namespace all_Xanths_are_Yelps_and_Wicks_l148_148958

-- Definitions for Zorbs, Yelps, Xanths, and Wicks
variable {U : Type} (Zorb Yelp Xanth Wick : U → Prop)

-- Conditions from the problem
axiom all_Zorbs_are_Yelps : ∀ u, Zorb u → Yelp u
axiom all_Xanths_are_Zorbs : ∀ u, Xanth u → Zorb u
axiom all_Xanths_are_Wicks : ∀ u, Xanth u → Wick u

-- The goal is to prove that all Xanths are Yelps and are Wicks
theorem all_Xanths_are_Yelps_and_Wicks : ∀ u, Xanth u → Yelp u ∧ Wick u := sorry

end all_Xanths_are_Yelps_and_Wicks_l148_148958


namespace chess_games_total_l148_148884

-- Conditions
def crowns_per_win : ℕ := 8
def uncle_wins : ℕ := 4
def draws : ℕ := 5
def father_net_gain : ℤ := 24

-- Let total_games be the total number of games played
def total_games : ℕ := sorry

-- Proof that under the given conditions, total_games equals 16
theorem chess_games_total :
  total_games = uncle_wins + (father_net_gain + uncle_wins * crowns_per_win) / crowns_per_win + draws := by
  sorry

end chess_games_total_l148_148884


namespace proportion_Q_to_R_l148_148677

theorem proportion_Q_to_R (q r : ℕ) (h1 : 3 * q + 5 * r = 1000) (h2 : 4 * r - 2 * q = 250) : q = r :=
by sorry

end proportion_Q_to_R_l148_148677


namespace exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l148_148811

theorem exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles
    (a b c d α β γ δ: ℝ) (h_conv: a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c)
    (h_angles: α < β + γ + δ ∧ β < α + γ + δ ∧ γ < α + β + δ ∧ δ < α + β + γ) :
    ∃ (a' b' c' d' α' β' γ' δ' : ℝ),
      (a' / b' = α / β) ∧ (b' / c' = β / γ) ∧ (c' / d' = γ / δ) ∧ (d' / a' = δ / α) ∧
      (a' < b' + c' + d') ∧ (b' < a' + c' + d') ∧ (c' < a' + b' + d') ∧ (d' < a' + b' + c') ∧
      (α' < β' + γ' + δ') ∧ (β' < α' + γ' + δ') ∧ (γ' < α' + β' + δ') ∧ (δ' < α' + β' + γ') :=
  sorry

end exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l148_148811


namespace power_function_half_l148_148215

theorem power_function_half (a : ℝ) (ha : (4 : ℝ)^a / (2 : ℝ)^a = 3) : (1 / 2 : ℝ) ^ a = 1 / 3 := 
by
  sorry

end power_function_half_l148_148215


namespace ratio_james_paid_l148_148077

-- Define the parameters of the problem
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def james_paid : ℚ := 6

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack
-- Total cost of stickers
def total_cost : ℚ := total_stickers * cost_per_sticker

-- Theorem stating that the ratio of the amount James paid to the total cost of the stickers is 1:2
theorem ratio_james_paid : james_paid / total_cost = 1 / 2 :=
by 
  -- proof goes here
  sorry

end ratio_james_paid_l148_148077


namespace phone_not_answered_prob_l148_148657

noncomputable def P_not_answered_within_4_rings : ℝ :=
  let P1 := 1 - 0.1
  let P2 := 1 - 0.3
  let P3 := 1 - 0.4
  let P4 := 1 - 0.1
  P1 * P2 * P3 * P4

theorem phone_not_answered_prob : 
  P_not_answered_within_4_rings = 0.3402 := 
by 
  -- The detailed steps and proof will be implemented here 
  sorry

end phone_not_answered_prob_l148_148657


namespace power_multiplication_l148_148720

theorem power_multiplication :
  3^5 * 6^5 = 1889568 :=
by
  sorry

end power_multiplication_l148_148720


namespace canoes_vs_kayaks_l148_148967

theorem canoes_vs_kayaks (C K : ℕ) (h1 : 9 * C + 12 * K = 432) (h2 : C = 4 * K / 3) : C - K = 6 :=
sorry

end canoes_vs_kayaks_l148_148967


namespace batsman_average_proof_l148_148260

noncomputable def batsman_average_after_17th_inning (A : ℝ) : ℝ :=
  (A * 16 + 87) / 17

theorem batsman_average_proof (A : ℝ) (h1 : 16 * A + 87 = 17 * (A + 2)) : batsman_average_after_17th_inning 53 = 55 :=
by
  sorry

end batsman_average_proof_l148_148260


namespace polynomial_satisfies_condition_l148_148575

open Polynomial

noncomputable def polynomial_f : Polynomial ℝ := 6 * X ^ 2 + 5 * X + 1
noncomputable def polynomial_g : Polynomial ℝ := 3 * X ^ 2 + 7 * X + 2

def sum_of_squares (p : Polynomial ℝ) : ℝ :=
  p.coeff 0 ^ 2 + p.coeff 1 ^ 2 + p.coeff 2 ^ 2 + p.coeff 3 ^ 2 + -- ...
  sorry -- Extend as necessary for the degree of the polynomial

theorem polynomial_satisfies_condition :
  (∀ n : ℕ, sum_of_squares (polynomial_f ^ n) = sum_of_squares (polynomial_g ^ n)) :=
by
  sorry

end polynomial_satisfies_condition_l148_148575


namespace range_of_m_l148_148236

-- Definitions used to state conditions of the problem.
def fractional_equation (m x : ℝ) : Prop := (m / (2 * x - 1)) + 2 = 0
def positive_solution (x : ℝ) : Prop := x > 0

-- The Lean 4 theorem statement
theorem range_of_m (m x : ℝ) (h : fractional_equation m x) (hx : positive_solution x) : m < 2 ∧ m ≠ 0 :=
by
  sorry

end range_of_m_l148_148236


namespace find_g_five_l148_148703

def g (a b c x : ℝ) : ℝ := a * x^7 + b * x^6 + c * x - 3

theorem find_g_five (a b c : ℝ) (h : g a b c (-5) = -3) : g a b c 5 = 31250 * b - 3 := 
sorry

end find_g_five_l148_148703


namespace largest_prime_factor_of_4752_l148_148449

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬ (m ∣ n)

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p)

def pf_4752 : ℕ := 4752

theorem largest_prime_factor_of_4752 : largest_prime_factor pf_4752 11 :=
  by
  sorry

end largest_prime_factor_of_4752_l148_148449


namespace find_k_l148_148796

theorem find_k (x : ℝ) (a h k : ℝ) (h1 : 9 * x^2 - 12 * x = a * (x - h)^2 + k) : k = -4 := by
  sorry

end find_k_l148_148796


namespace regression_shows_positive_correlation_l148_148034

-- Define the regression equations as constants
def reg_eq_A (x : ℝ) : ℝ := -2.1 * x + 1.8
def reg_eq_B (x : ℝ) : ℝ := 1.2 * x + 1.5
def reg_eq_C (x : ℝ) : ℝ := -0.5 * x + 2.1
def reg_eq_D (x : ℝ) : ℝ := -0.6 * x + 3

-- Define the condition for positive correlation
def positive_correlation (b : ℝ) : Prop := b > 0

-- The theorem statement to prove
theorem regression_shows_positive_correlation : 
  positive_correlation 1.2 := 
by
  sorry

end regression_shows_positive_correlation_l148_148034


namespace geometric_sequence_arithmetic_progression_l148_148447

open Nat

/--
Given a geometric sequence \( \{a_n\} \) where \( a_1 = 1 \) and the sequence terms
\( 4a_1 \), \( 2a_2 \), \( a_3 \) form an arithmetic progression, prove that
the common ratio \( q = 2 \) and the sum of the first four terms \( S_4 = 15 \).
-/
theorem geometric_sequence_arithmetic_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h₀ : a 1 = 1)
    (h₁ : ∀ n, S n = (1 - q^n) / (1 - q)) 
    (h₂ : ∀ k n, a (k + n) = a k * q ^ n) 
    (h₃ : 4 * a 1 + a 3 = 4 * a 2) :
  q = 2 ∧ S 4 = 15 := 
sorry

end geometric_sequence_arithmetic_progression_l148_148447


namespace _l148_148393

noncomputable def waiter_fraction_from_tips (S T I : ℝ) : Prop :=
  T = (5 / 2) * S ∧
  I = S + T ∧
  T / I = 5 / 7

lemma waiter_tips_fraction_theorem (S T I : ℝ) : waiter_fraction_from_tips S T I → T / I = 5 / 7 :=
by
  intro h
  rw [waiter_fraction_from_tips] at h
  obtain ⟨h₁, h₂, h₃⟩ := h
  exact h₃

end _l148_148393


namespace second_sheet_width_l148_148282

theorem second_sheet_width :
  ∃ w : ℝ, (286 = 22 * w + 100) ∧ w = 8.5 :=
by
  -- Proof goes here
  sorry

end second_sheet_width_l148_148282


namespace remainder_when_divided_by_32_l148_148129

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l148_148129


namespace evaluate_expression_l148_148353

theorem evaluate_expression : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end evaluate_expression_l148_148353


namespace certain_number_l148_148081

theorem certain_number (x y : ℝ) (h1 : 0.65 * x = 0.20 * y) (h2 : x = 210) : y = 682.5 :=
by
  sorry

end certain_number_l148_148081


namespace number_of_girls_attending_winter_festival_l148_148936

variables (g b : ℝ)
variables (totalStudents attendFestival: ℝ)

theorem number_of_girls_attending_winter_festival
  (H1 : g + b = 1500)
  (H2 : (3/5) * g + (2/5) * b = 800) :
  (3/5 * g) = 600 :=
sorry

end number_of_girls_attending_winter_festival_l148_148936


namespace largest_three_digit_divisible_by_13_l148_148010

theorem largest_three_digit_divisible_by_13 :
  ∃ n, (n ≤ 999 ∧ n ≥ 100 ∧ 13 ∣ n) ∧ (∀ m, m ≤ 999 ∧ m ≥ 100 ∧ 13 ∣ m → m ≤ 987) :=
by
  sorry

end largest_three_digit_divisible_by_13_l148_148010


namespace inequality_relationship_l148_148080

noncomputable def even_function_periodic_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (x + 2) = f x) ∧
  (∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 1 → f x1 > f x2)

theorem inequality_relationship (f : ℝ → ℝ) (h : even_function_periodic_decreasing f) : 
  f (-1) < f (2.5) ∧ f (2.5) < f 0 :=
by 
  sorry

end inequality_relationship_l148_148080


namespace solve_system_l148_148265

def system_of_equations (x y : ℝ) : Prop :=
  (4 * (x - y) = 8 - 3 * y) ∧ (x / 2 + y / 3 = 1)

theorem solve_system : ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 0 := 
  by
  sorry

end solve_system_l148_148265


namespace students_side_by_side_with_A_and_B_l148_148114

theorem students_side_by_side_with_A_and_B (total students_from_club_A students_from_club_B: ℕ) 
    (h1 : total = 100)
    (h2 : students_from_club_A = 62)
    (h3 : students_from_club_B = 54) :
  ∃ p q r : ℕ, p + q + r = 100 ∧ p + q = 62 ∧ p + r = 54 ∧ p = 16 :=
by
  sorry

end students_side_by_side_with_A_and_B_l148_148114


namespace range_of_8x_plus_y_l148_148678

theorem range_of_8x_plus_y (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_condition : 1 / x + 2 / y = 2) : 8 * x + y ≥ 9 :=
by
  sorry

end range_of_8x_plus_y_l148_148678


namespace problem_solution_l148_148198

def sequence_graphical_representation_isolated (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ x : ℝ, x = a n

def sequence_terms_infinite (a : ℕ → ℝ) : Prop :=
  ∃ l : List ℝ, ∃ n : ℕ, l.length = n

def sequence_general_term_formula_unique (a : ℕ → ℝ) : Prop :=
  ∀ f g : ℕ → ℝ, (∀ n, f n = g n) → f = g

theorem problem_solution
  (h1 : ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a)
  (h2 : ¬ ∀ a : ℕ → ℝ, sequence_terms_infinite a)
  (h3 : ¬ ∀ a : ℕ → ℝ, sequence_general_term_formula_unique a) :
  ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a ∧ 
                ¬ (sequence_terms_infinite a) ∧
                ¬ (sequence_general_term_formula_unique a) := by
  sorry

end problem_solution_l148_148198


namespace halfway_miles_proof_l148_148947

def groceries_miles : ℕ := 10
def haircut_miles : ℕ := 15
def doctor_miles : ℕ := 5

def total_miles : ℕ := groceries_miles + haircut_miles + doctor_miles

theorem halfway_miles_proof : total_miles / 2 = 15 := by
  -- calculation to follow
  sorry

end halfway_miles_proof_l148_148947


namespace correct_value_wrongly_copied_l148_148887

theorem correct_value_wrongly_copied 
  (mean_initial : ℕ)
  (mean_correct : ℕ)
  (wrong_value : ℕ) 
  (n : ℕ) 
  (initial_mean : mean_initial = 250)
  (correct_mean : mean_correct = 251)
  (wrongly_copied : wrong_value = 135)
  (number_of_values : n = 30) : 
  ∃ x : ℕ, x = 165 := 
by
  use (wrong_value + (mean_correct - mean_initial) * n / n)
  sorry

end correct_value_wrongly_copied_l148_148887


namespace math_proof_l148_148961

theorem math_proof :
  ∀ (x y z : ℚ), (2 * x - 3 * y - 2 * z = 0) →
                  (x + 3 * y - 28 * z = 0) →
                  (z ≠ 0) →
                  (x^2 + 3 * x * y * z) / (y^2 + z^2) = 280 / 37 :=
by
  intros x y z h1 h2 h3
  sorry

end math_proof_l148_148961


namespace cooking_people_count_l148_148826

variables (P Y W : ℕ)

def people_practicing_yoga := 25
def people_studying_weaving := 8
def people_studying_only_cooking := 2
def people_studying_cooking_and_yoga := 7
def people_studying_cooking_and_weaving := 3
def people_studying_all_curriculums := 3

theorem cooking_people_count :
  P = people_studying_only_cooking + (people_studying_cooking_and_yoga - people_studying_all_curriculums)
    + (people_studying_cooking_and_weaving - people_studying_all_curriculums) + people_studying_all_curriculums →
  P = 9 :=
by
  intro h
  unfold people_studying_only_cooking people_studying_cooking_and_yoga people_studying_cooking_and_weaving people_studying_all_curriculums at h
  sorry

end cooking_people_count_l148_148826


namespace lost_weights_l148_148110

-- Define the weights
def weights : List ℕ := [43, 70, 57]

-- Total remaining weight after loss
def remaining_weight : ℕ := 20172

-- Number of weights lost
def weights_lost : ℕ := 4

-- Whether a given number of weights and types of weights match the remaining weight
def valid_loss (initial_count : ℕ) (lost_weight_count : ℕ) : Prop :=
  let total_initial_weight := initial_count * (weights.sum)
  let lost_weight := lost_weight_count * 57
  total_initial_weight - lost_weight = remaining_weight

-- Proposition we need to prove
theorem lost_weights (initial_count : ℕ) (h : valid_loss initial_count weights_lost) : ∀ w ∈ weights, w = 57 :=
by {
  sorry
}

end lost_weights_l148_148110


namespace neznaika_mistake_l148_148605

-- Let's define the conditions
variables {X A Y M E O U : ℕ} -- Represents distinct digits

-- Ascending order of the numbers
variables (XA AY AX OY EM EY MU : ℕ)
  (h1 : XA < AY)
  (h2 : AY < AX)
  (h3 : AX < OY)
  (h4 : OY < EM)
  (h5 : EM < EY)
  (h6 : EY < MU)

-- Identical digits replaced with the same letters
variables (h7 : XA = 10 * X + A)
  (h8 : AY = 10 * A + Y)
  (h9 : AX = 10 * A + X)
  (h10 : OY = 10 * O + Y)
  (h11 : EM = 10 * E + M)
  (h12 : EY = 10 * E + Y)
  (h13 : MU = 10 * M + U)

-- Each letter represents a different digit
variables (h_distinct : X ≠ A ∧ X ≠ Y ∧ X ≠ M ∧ X ≠ E ∧ X ≠ O ∧ X ≠ U ∧
                       A ≠ Y ∧ A ≠ M ∧ A ≠ E ∧ A ≠ O ∧ A ≠ U ∧
                       Y ≠ M ∧ Y ≠ E ∧ Y ≠ O ∧ Y ≠ U ∧
                       M ≠ E ∧ M ≠ O ∧ M ≠ U ∧
                       E ≠ O ∧ E ≠ U ∧
                       O ≠ U)

-- Prove Neznaika made a mistake
theorem neznaika_mistake : false :=
by
  -- Here we'll reach a contradiction, proving false.
  sorry

end neznaika_mistake_l148_148605


namespace statement_B_statement_C_l148_148589

variable (a b c : ℝ)

-- Condition: a > b
def condition1 := a > b

-- Condition: a / c^2 > b / c^2
def condition2 := a / c^2 > b / c^2

-- Statement B: If a > b, then a - 1 > b - 2
theorem statement_B (ha_gt_b : condition1 a b) : a - 1 > b - 2 :=
by sorry

-- Statement C: If a / c^2 > b / c^2, then a > b
theorem statement_C (ha_div_csqr_gt_hb_div_csqr : condition2 a b c) : a > b :=
by sorry

end statement_B_statement_C_l148_148589


namespace typeA_selling_price_maximize_profit_l148_148625

theorem typeA_selling_price (sales_last_year : ℝ) (sales_increase_rate : ℝ) (price_increase : ℝ) 
                            (cars_sold_last_year : ℝ) : 
                            (sales_last_year = 32000) ∧ (sales_increase_rate = 1.25) ∧ 
                            (price_increase = 400) ∧ 
                            (sales_last_year / cars_sold_last_year = (sales_last_year * sales_increase_rate) / (cars_sold_last_year + price_increase)) → 
                            (cars_sold_last_year = 1600) :=
by
  sorry

theorem maximize_profit (typeA_price : ℝ) (typeB_price : ℝ) (typeA_cost : ℝ) (typeB_cost : ℝ) 
                        (total_cars : ℕ) :
                        (typeA_price = 2000) ∧ (typeB_price = 2400) ∧ 
                        (typeA_cost = 1100) ∧ (typeB_cost = 1400) ∧ 
                        (total_cars = 50) ∧ 
                        (∀ m : ℕ, m ≤ 50 / 3) → 
                        ∃ m : ℕ, (m = 17) ∧ (50 - m * 2 ≤ 33) :=
by
  sorry

end typeA_selling_price_maximize_profit_l148_148625


namespace range_of_a_l148_148574

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x + x^2 else x - x^2

theorem range_of_a (a : ℝ) : (∀ x, -1/2 ≤ x ∧ x ≤ 1/2 → f (x^2 + 1) > f (a * x)) ↔ -5/2 < a ∧ a < 5/2 := 
sorry

end range_of_a_l148_148574


namespace probability_odd_even_draw_correct_l148_148105

noncomputable def probability_odd_even_draw : ℚ := sorry

theorem probability_odd_even_draw_correct :
  probability_odd_even_draw = 17 / 45 := 
sorry

end probability_odd_even_draw_correct_l148_148105


namespace distinct_numbers_in_union_set_l148_148339

def first_seq_term (k : ℕ) : ℤ := 5 * ↑k - 3
def second_seq_term (m : ℕ) : ℤ := 9 * ↑m - 3

def first_seq_set : Finset ℤ := ((Finset.range 1003).image first_seq_term)
def second_seq_set : Finset ℤ := ((Finset.range 1003).image second_seq_term)

def union_set : Finset ℤ := first_seq_set ∪ second_seq_set

theorem distinct_numbers_in_union_set : union_set.card = 1895 := by
  sorry

end distinct_numbers_in_union_set_l148_148339


namespace jill_water_stored_l148_148636

theorem jill_water_stored (n : ℕ) (h : n = 24) : 
  8 * (1 / 4 : ℝ) + 8 * (1 / 2 : ℝ) + 8 * 1 = 14 :=
by
  sorry

end jill_water_stored_l148_148636


namespace bridge_length_problem_l148_148807

noncomputable def length_of_bridge (num_carriages : ℕ) (length_carriage : ℕ) (length_engine : ℕ) (speed_kmph : ℕ) (crossing_time_min : ℕ) : ℝ :=
  let total_train_length := (num_carriages + 1) * length_carriage
  let speed_mps := (speed_kmph * 1000) / 3600
  let crossing_time_secs := crossing_time_min * 60
  let total_distance := speed_mps * crossing_time_secs
  let bridge_length := total_distance - total_train_length
  bridge_length

theorem bridge_length_problem :
  length_of_bridge 24 60 60 60 5 = 3501 :=
by
  sorry

end bridge_length_problem_l148_148807


namespace lesser_solution_of_quadratic_l148_148830

theorem lesser_solution_of_quadratic :
  (∃ x y: ℝ, x ≠ y ∧ x^2 + 10*x - 24 = 0 ∧ y^2 + 10*y - 24 = 0 ∧ min x y = -12) :=
by {
  sorry
}

end lesser_solution_of_quadratic_l148_148830


namespace probability_all_correct_l148_148955

noncomputable def probability_mcq : ℚ := 1 / 3
noncomputable def probability_true_false : ℚ := 1 / 2

theorem probability_all_correct :
  (probability_mcq * probability_true_false * probability_true_false) = (1 / 12) :=
by
  sorry

end probability_all_correct_l148_148955


namespace cost_of_each_pair_of_shorts_l148_148473

variable (C : ℝ)
variable (h_discount : 3 * C - 2.7 * C = 3)

theorem cost_of_each_pair_of_shorts : C = 10 :=
by 
  sorry

end cost_of_each_pair_of_shorts_l148_148473


namespace min_value_am_hm_inequality_l148_148793

theorem min_value_am_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
    (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end min_value_am_hm_inequality_l148_148793


namespace inequality_proof_l148_148711

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l148_148711


namespace can_capacity_l148_148993

-- Definition for the capacity of the can
theorem can_capacity 
  (milk_ratio water_ratio : ℕ) 
  (add_milk : ℕ) 
  (final_milk_ratio final_water_ratio : ℕ) 
  (capacity : ℕ) 
  (initial_milk initial_water : ℕ) 
  (h_initial_ratio : milk_ratio = 4 ∧ water_ratio = 3) 
  (h_additional_milk : add_milk = 8) 
  (h_final_ratio : final_milk_ratio = 2 ∧ final_water_ratio = 1) 
  (h_initial_amounts : initial_milk = 4 * (capacity - add_milk) / 7 ∧ initial_water = 3 * (capacity - add_milk) / 7) 
  (h_full_capacity : (initial_milk + add_milk) / initial_water = 2) 
  : capacity = 36 :=
sorry

end can_capacity_l148_148993


namespace problem_l148_148540

theorem problem (p q : ℕ) (hp: p > 1) (hq: q > 1) (h1 : (2 * p - 1) % q = 0) (h2 : (2 * q - 1) % p = 0) : p + q = 8 := 
sorry

end problem_l148_148540


namespace solve_equation_l148_148095

def euler_totient (n : ℕ) : ℕ := sorry  -- Placeholder, Euler's φ function definition
def sigma_function (n : ℕ) : ℕ := sorry  -- Placeholder, σ function definition

theorem solve_equation (x : ℕ) : euler_totient (sigma_function (2^x)) = 2^x → x = 1 := by
  sorry

end solve_equation_l148_148095


namespace number_of_a_values_l148_148426

theorem number_of_a_values (a : ℝ) :
  (∃ x : ℝ, y = x + 2*a ∧ y = x^3 - 3*a*x + a^3) → a = 0 :=
by
  sorry

end number_of_a_values_l148_148426


namespace find_b_minus_d_squared_l148_148979

theorem find_b_minus_d_squared (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3) :
  (b - d) ^ 2 = 25 :=
sorry

end find_b_minus_d_squared_l148_148979


namespace boat_speed_in_still_water_l148_148240

/-- Prove the speed of the boat in still water given the conditions -/
theorem boat_speed_in_still_water (V_s : ℝ) (T : ℝ) (D : ℝ) (V_b : ℝ) :
  V_s = 4 ∧ T = 4 ∧ D = 112 ∧ (D / T = V_b + V_s) → V_b = 24 := sorry

end boat_speed_in_still_water_l148_148240


namespace find_pairs_l148_148645

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

theorem find_pairs {a b : ℝ} :
  (0 < b) → (b ≤ 1) → (0 < a) → (a < 1) → (2 * a + b ≤ 2) →
  (∀ x y : ℝ, f a b (x * y) + f a b (x + y) ≥ f a b x * f a b y) :=
by
  intros h_b_gt_zero h_b_le_one h_a_gt_zero h_a_lt_one h_2a_b_le_2
  sorry

end find_pairs_l148_148645


namespace waiter_total_customers_l148_148639

def numCustomers (T : ℕ) (totalTips : ℕ) (tipPerCustomer : ℕ) (numNoTipCustomers : ℕ) : ℕ :=
  T + numNoTipCustomers

theorem waiter_total_customers
  (T : ℕ)
  (h1 : 3 * T = 6)
  (numNoTipCustomers : ℕ := 5)
  (total := numCustomers T 6 3 numNoTipCustomers) :
  total = 7 := by
  sorry

end waiter_total_customers_l148_148639


namespace probability_red_blue_yellow_l148_148715

-- Define the probabilities for white, green, and black marbles
def p_white : ℚ := 1/4
def p_green : ℚ := 1/6
def p_black : ℚ := 1/8

-- Define the problem: calculating the probability of drawing a red, blue, or yellow marble
theorem probability_red_blue_yellow : 
  p_white = 1/4 → p_green = 1/6 → p_black = 1/8 →
  (1 - (p_white + p_green + p_black)) = 11/24 := 
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end probability_red_blue_yellow_l148_148715


namespace percentage_increase_in_savings_l148_148113

theorem percentage_increase_in_savings
  (I : ℝ) -- Original income of Paulson
  (E : ℝ) -- Original expenditure of Paulson
  (hE : E = 0.75 * I) -- Paulson spends 75% of his income
  (h_inc_income : 1.2 * I = I + 0.2 * I) -- Income is increased by 20%
  (h_inc_expenditure : 0.825 * I = 0.75 * I + 0.1 * (0.75 * I)) -- Expenditure is increased by 10%
  : (0.375 * I - 0.25 * I) / (0.25 * I) * 100 = 50 := by
  sorry

end percentage_increase_in_savings_l148_148113


namespace find_y_common_solution_l148_148948

theorem find_y_common_solution (y : ℝ) :
  (∃ x : ℝ, x^2 + y^2 = 11 ∧ x^2 = 4*y - 7) ↔ (7/4 ≤ y ∧ y ≤ Real.sqrt 11) :=
by
  sorry

end find_y_common_solution_l148_148948


namespace line_parallelism_theorem_l148_148949

-- Definitions of the relevant geometric conditions
variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions as hypotheses
axiom line_parallel_plane (m : Line) (α : Plane) : Prop
axiom line_in_plane (n : Line) (α : Plane) : Prop
axiom plane_intersection_line (α β : Plane) : Line
axiom line_parallel (m n : Line) : Prop

-- The problem statement in Lean 4
theorem line_parallelism_theorem 
  (h1 : line_parallel_plane m α) 
  (h2 : line_in_plane n β) 
  (h3 : plane_intersection_line α β = n) 
  (h4 : line_parallel_plane m β) : line_parallel m n :=
sorry

end line_parallelism_theorem_l148_148949


namespace percent_decrease_l148_148242

-- Definitions based on conditions
def originalPrice : ℝ := 100
def salePrice : ℝ := 10

-- The percentage decrease is the main statement to prove
theorem percent_decrease : ((originalPrice - salePrice) / originalPrice) * 100 = 90 := 
by
  -- Placeholder for proof
  sorry

end percent_decrease_l148_148242


namespace parametric_to_standard_l148_148865

theorem parametric_to_standard (t : ℝ) : 
  (x = (2 + 3 * t) / (1 + t)) ∧ (y = (1 - 2 * t) / (1 + t)) → (3 * x + y - 7 = 0) ∧ (x ≠ 3) := 
by 
  sorry

end parametric_to_standard_l148_148865


namespace find_sum_of_a_b_c_l148_148802

theorem find_sum_of_a_b_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(h4 : (a + b + c) ^ 3 - a ^ 3 - b ^ 3 - c ^ 3 = 210) : a + b + c = 11 :=
sorry

end find_sum_of_a_b_c_l148_148802


namespace root_expression_value_l148_148378

noncomputable def value_of_expression (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) : ℝ :=
  sorry

theorem root_expression_value (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) :
  value_of_expression p q r h1 h2 h3 = 367 / 183 :=
sorry

end root_expression_value_l148_148378


namespace total_guppies_l148_148572

-- Define conditions
def Haylee_guppies : ℕ := 3 * 12
def Jose_guppies : ℕ := Haylee_guppies / 2
def Charliz_guppies : ℕ := Jose_guppies / 3
def Nicolai_guppies : ℕ := Charliz_guppies * 4

-- Theorem statement: total number of guppies is 84
theorem total_guppies : Haylee_guppies + Jose_guppies + Charliz_guppies + Nicolai_guppies = 84 := 
by 
  sorry

end total_guppies_l148_148572


namespace least_n_for_distance_l148_148852

-- Definitions ensuring our points and distances
def A_0 : (ℝ × ℝ) := (0, 0)

-- Assume we have distance function and equilateral triangles on given coordinates
def is_on_x_axis (p : ℕ → ℝ × ℝ) : Prop := ∀ n, (p n).snd = 0
def is_on_parabola (q : ℕ → ℝ × ℝ) : Prop := ∀ n, (q n).snd = (q n).fst^2
def is_equilateral (p : ℕ → ℝ × ℝ) (q : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
  let d1 := dist (p (n-1)) (q n)
  let d2 := dist (q n) (p n)
  let d3 := dist (p (n-1)) (p n)
  d1 = d2 ∧ d2 = d3

-- Define the main property we want to prove
def main_property (n : ℕ) (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) : Prop :=
  A 0 = A_0 ∧ is_on_x_axis A ∧ is_on_parabola B ∧
  (∀ k, is_equilateral A B (k+1)) ∧
  dist A_0 (A n) ≥ 200

-- Final theorem statement
theorem least_n_for_distance (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) :
  (∃ n, main_property n A B ∧ (∀ m, main_property m A B → n ≤ m)) ↔ n = 24 := by
  sorry

end least_n_for_distance_l148_148852


namespace completion_time_B_l148_148261

-- Definitions based on conditions
def work_rate_A : ℚ := 1 / 10 -- A's rate of completing work per day

def efficiency_B : ℚ := 1.75 -- B is 75% more efficient than A

def work_rate_B : ℚ := efficiency_B * work_rate_A -- B's work rate per day

-- The main theorem that we need to prove
theorem completion_time_B : (1 : ℚ) / work_rate_B = 40 / 7 :=
by 
  sorry

end completion_time_B_l148_148261


namespace calculate_speed_l148_148419

theorem calculate_speed :
  ∀ (distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph : ℚ),
  distance_ft = 200 →
  time_sec = 2 →
  miles_per_ft = 1 / 5280 →
  hours_per_sec = 1 / 3600 →
  approx_speed_mph = 68.1818181818 →
  (distance_ft * miles_per_ft) / (time_sec * hours_per_sec) = approx_speed_mph :=
by
  intros distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph
  intro h_distance_eq h_time_eq h_miles_eq h_hours_eq h_speed_eq
  sorry

end calculate_speed_l148_148419


namespace remainder_identity_l148_148235

variable {n : ℕ}

theorem remainder_identity
  (a b a_1 b_1 a_2 b_2 : ℕ)
  (ha : a = a_1 + a_2 * n)
  (hb : b = b_1 + b_2 * n) :
  (((a + b) % n = (a_1 + b_1) % n) ∧ ((a - b) % n = (a_1 - b_1) % n)) ∧ ((a * b) % n = (a_1 * b_1) % n) := by
  sorry

end remainder_identity_l148_148235


namespace solve_inequality_correct_l148_148406

noncomputable def solve_inequality (a x : ℝ) : Set ℝ :=
  if a > 1 ∨ a < 0 then {x | x ≤ a ∨ x ≥ a^2 }
  else if a = 1 ∨ a = 0 then {x | True}
  else {x | x ≤ a^2 ∨ x ≥ a}

theorem solve_inequality_correct (a x : ℝ) :
  (x^2 - (a^2 + a) * x + a^3 ≥ 0) ↔ 
    (if a > 1 ∨ a < 0 then x ≤ a ∨ x ≥ a^2
      else if a = 1 ∨ a = 0 then True
      else x ≤ a^2 ∨ x ≥ a) :=
by sorry

end solve_inequality_correct_l148_148406


namespace find_x2_y2_and_xy_l148_148747

-- Problem statement
theorem find_x2_y2_and_xy (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 :=
by
  sorry -- Proof omitted

end find_x2_y2_and_xy_l148_148747


namespace cubic_roots_identity_l148_148390

theorem cubic_roots_identity 
  (x1 x2 x3 p q : ℝ) 
  (hq : ∀ x, x^3 + p * x + q = (x - x1) * (x - x2) * (x - x3))
  (h_sum : x1 + x2 + x3 = 0)
  (h_prod : x1 * x2 + x2 * x3 + x3 * x1 = p):
  x2^2 + x2 * x3 + x3^2 = -p ∧ x1^2 + x1 * x3 + x3^2 = -p ∧ x1^2 + x1 * x2 + x2^2 = -p :=
sorry

end cubic_roots_identity_l148_148390


namespace degrees_subtraction_l148_148608

theorem degrees_subtraction :
  (108 * 3600 + 18 * 60 + 25) - (56 * 3600 + 23 * 60 + 32) = (51 * 3600 + 54 * 60 + 53) :=
by sorry

end degrees_subtraction_l148_148608


namespace value_of_f_at_6_l148_148860

-- The condition that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- The condition that f(x + 2) = -f(x)
def periodic_sign_flip (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = -f (x)

-- The theorem statement
theorem value_of_f_at_6 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : periodic_sign_flip f) : f 6 = 0 :=
sorry

end value_of_f_at_6_l148_148860


namespace range_of_a_l148_148602

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x < 3) → (4 * a * x + 4 * (a - 3)) ≤ 0) ↔ (0 ≤ a ∧ a ≤ 3 / 4) :=
by
  sorry

end range_of_a_l148_148602


namespace optimal_order_for_ostap_l148_148309

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l148_148309


namespace combined_cost_price_correct_l148_148594

def face_value_A : ℝ := 100
def discount_A : ℝ := 0.02
def face_value_B : ℝ := 100
def premium_B : ℝ := 0.015
def brokerage : ℝ := 0.002

def purchase_price_A := face_value_A * (1 - discount_A)
def brokerage_fee_A := purchase_price_A * brokerage
def total_cost_price_A := purchase_price_A + brokerage_fee_A

def purchase_price_B := face_value_B * (1 + premium_B)
def brokerage_fee_B := purchase_price_B * brokerage
def total_cost_price_B := purchase_price_B + brokerage_fee_B

def combined_cost_price := total_cost_price_A + total_cost_price_B

theorem combined_cost_price_correct :
  combined_cost_price = 199.899 :=
by
  sorry

end combined_cost_price_correct_l148_148594


namespace larger_cylinder_volume_l148_148402

theorem larger_cylinder_volume (v: ℝ) (r: ℝ) (R: ℝ) (h: ℝ) (hR : R = 2 * r) (hv : v = 100) : 
  π * R^2 * h = 4 * v := 
by 
  sorry

end larger_cylinder_volume_l148_148402


namespace translation_result_l148_148383

variables (P : ℝ × ℝ) (P' : ℝ × ℝ)

def translate_left (P : ℝ × ℝ) (units : ℝ) := (P.1 - units, P.2)
def translate_down (P : ℝ × ℝ) (units : ℝ) := (P.1, P.2 - units)

theorem translation_result :
    P = (-4, 3) -> P' = translate_down (translate_left P 2) 2 -> P' = (-6, 1) :=
by
  intros h1 h2
  sorry

end translation_result_l148_148383


namespace heather_average_balance_l148_148008

theorem heather_average_balance :
  let balance_J := 150
  let balance_F := 250
  let balance_M := 100
  let balance_A := 200
  let balance_May := 300
  let total_balance := balance_J + balance_F + balance_M + balance_A + balance_May
  let avg_balance := total_balance / 5
  avg_balance = 200 :=
by
  sorry

end heather_average_balance_l148_148008


namespace at_least_one_l148_148490

axiom P : Prop  -- person A is an outstanding student
axiom Q : Prop  -- person B is an outstanding student

theorem at_least_one (H : ¬(¬P ∧ ¬Q)) : P ∨ Q :=
sorry

end at_least_one_l148_148490


namespace part_a_l148_148880

theorem part_a (a x y : ℕ) (h_a_pos : a > 0) (h_x_pos : x > 0) (h_y_pos : y > 0) (h_neq : x ≠ y) :
  (a * x + Nat.gcd a x + Nat.lcm a x) ≠ (a * y + Nat.gcd a y + Nat.lcm a y) := sorry

end part_a_l148_148880


namespace max_planes_determined_l148_148267

-- Definitions for conditions
variables (Point Line Plane : Type)
variables (l : Line) (A B C : Point)
variables (contains : Point → Line → Prop)
variables (plane_contains_points : Plane → Point → Point → Point → Prop)
variables (plane_contains_line_and_point : Plane → Line → Point → Prop)
variables (non_collinear : Point → Point → Point → Prop)
variables (not_on_line : Point → Line → Prop)

-- Hypotheses based on the conditions
axiom three_non_collinear_points : non_collinear A B C
axiom point_not_on_line (P : Point) : not_on_line P l

-- Goal: Prove that the number of planes is 4
theorem max_planes_determined : 
  ∃ total_planes : ℕ, total_planes = 4 :=
sorry

end max_planes_determined_l148_148267


namespace percentage_of_left_handed_women_l148_148468

variable (x y : Nat) (h_ratio_rh_lh : 3 * x = 1 * x)
variable (h_ratio_men_women : 3 * y = 2 * y)
variable (h_rh_men_max : True)

theorem percentage_of_left_handed_women :
  (x / (4 * x)) * 100 = 25 :=
by sorry

end percentage_of_left_handed_women_l148_148468


namespace f3_is_ideal_function_l148_148815

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + f (-x) = 0

def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

noncomputable def f3 (x : ℝ) : ℝ :=
  if x < 0 then x ^ 2 else -x ^ 2

theorem f3_is_ideal_function : is_odd_function f3 ∧ is_strictly_decreasing f3 := 
  sorry

end f3_is_ideal_function_l148_148815


namespace inequality_solution_l148_148982

theorem inequality_solution (m : ℝ) (x : ℝ) (hm : 0 ≤ m ∧ m ≤ 1) (ineq : m * x^2 - 2 * x - m ≥ 2) : x ≤ -1 :=
sorry

end inequality_solution_l148_148982


namespace quadratic_trinomial_has_two_roots_l148_148868

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l148_148868


namespace empty_rooms_le_1000_l148_148130

/--
In a 50x50 grid where each cell can contain at most one tree, 
with the following rules: 
1. A pomegranate tree has at least one apple neighbor
2. A peach tree has at least one apple neighbor and one pomegranate neighbor
3. An empty room has at least one apple neighbor, one pomegranate neighbor, and one peach neighbor
Show that the number of empty rooms is not greater than 1000.
-/
theorem empty_rooms_le_1000 (apple pomegranate peach : ℕ) (empty : ℕ)
  (h1 : apple + pomegranate + peach + empty = 2500)
  (h2 : ∀ p, pomegranate ≥ p → apple ≥ 1)
  (h3 : ∀ p, peach ≥ p → apple ≥ 1 ∧ pomegranate ≥ 1)
  (h4 : ∀ e, empty ≥ e → apple ≥ 1 ∧ pomegranate ≥ 1 ∧ peach ≥ 1) :
  empty ≤ 1000 :=
sorry

end empty_rooms_le_1000_l148_148130


namespace amgm_inequality_proof_l148_148905

noncomputable def amgm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  1 < (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ∧ (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ≤ (3 * Real.sqrt 2) / 2

theorem amgm_inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  amgm_inequality a b c ha hb hc := 
sorry

end amgm_inequality_proof_l148_148905


namespace binom_n_n_sub_2_l148_148477

theorem binom_n_n_sub_2 (n : ℕ) (h : n > 0) : (Nat.choose n (n - 2)) = (n * (n - 1)) / 2 := by
  sorry

end binom_n_n_sub_2_l148_148477


namespace other_leg_length_l148_148564

theorem other_leg_length (a b c : ℕ) (ha : a = 24) (hc : c = 25) 
  (h : a * a + b * b = c * c) : b = 7 := 
by 
  sorry

end other_leg_length_l148_148564


namespace tan_theta_sub_pi_over_4_l148_148074

open Real

theorem tan_theta_sub_pi_over_4 (θ : ℝ) (h1 : -π / 2 < θ ∧ θ < 0) 
  (h2 : sin (θ + π / 4) = 3 / 5) : tan (θ - π / 4) = -4 / 3 :=
by
  sorry

end tan_theta_sub_pi_over_4_l148_148074


namespace worker_days_total_l148_148817

theorem worker_days_total
  (W I : ℕ)
  (hw : 20 * W - 3 * I = 280)
  (hi : I = 40) :
  W + I = 60 :=
by
  sorry

end worker_days_total_l148_148817


namespace discount_correct_l148_148918

def normal_cost : ℝ := 80
def discount_rate : ℝ := 0.45
def discounted_cost : ℝ := normal_cost - (discount_rate * normal_cost)

theorem discount_correct : discounted_cost = 44 := by
  -- By computation, 0.45 * 80 = 36 and 80 - 36 = 44
  sorry

end discount_correct_l148_148918


namespace find_max_marks_l148_148078

variable (M : ℕ) (P : ℕ)

theorem find_max_marks (h1 : M = 332) (h2 : P = 83) : 
  let Max_Marks := M / (P / 100)
  Max_Marks = 400 := 
by 
  sorry

end find_max_marks_l148_148078


namespace bob_initial_cats_l148_148489

theorem bob_initial_cats (B : ℕ) (h : 21 - 4 = B + 14) : B = 3 := 
by
  -- Placeholder for the proof
  sorry

end bob_initial_cats_l148_148489


namespace task1_on_time_task2_not_on_time_prob_l148_148321

def task1_on_time_prob : ℚ := 3 / 8
def task2_on_time_prob : ℚ := 3 / 5

theorem task1_on_time_task2_not_on_time_prob :
  task1_on_time_prob * (1 - task2_on_time_prob) = 3 / 20 := by
  sorry

end task1_on_time_task2_not_on_time_prob_l148_148321


namespace pqr_value_l148_148125

theorem pqr_value
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 29)
  (h_eq : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 :=
by
  sorry

end pqr_value_l148_148125


namespace person_b_days_work_alone_l148_148791

theorem person_b_days_work_alone (B : ℕ) (h1 : (1 : ℚ) / 40 + 1 / B = 1 / 24) : B = 60 := 
by
  sorry

end person_b_days_work_alone_l148_148791


namespace solution_product_l148_148707

theorem solution_product (p q : ℝ) (hpq : p ≠ q) (h1 : (x-3)*(3*x+18) = x^2-15*x+54) (hp : (x - p) * (x - q) = x^2 - 12 * x + 54) :
  (p + 2) * (q + 2) = -80 := sorry

end solution_product_l148_148707


namespace k_squared_minus_3k_minus_4_l148_148592

theorem k_squared_minus_3k_minus_4 (a b c d k : ℚ)
  (h₁ : (2 * a) / (b + c + d) = k)
  (h₂ : (2 * b) / (a + c + d) = k)
  (h₃ : (2 * c) / (a + b + d) = k)
  (h₄ : (2 * d) / (a + b + c) = k) :
  k^2 - 3 * k - 4 = -50 / 9 ∨ k^2 - 3 * k - 4 = 6 :=
  sorry

end k_squared_minus_3k_minus_4_l148_148592


namespace minimum_value_expression_l148_148317

theorem minimum_value_expression 
  (a b c : ℝ) 
  (h1 : 3 * a + 2 * b + c = 5) 
  (h2 : 2 * a + b - 3 * c = 1) 
  (h3 : 0 ≤ a) 
  (h4 : 0 ≤ b) 
  (h5 : 0 ≤ c) : 
  ∃(c : ℝ), (c ≥ 3/7 ∧ c ≤ 7/11) ∧ (3 * a + b - 7 * c = -5/7) :=
sorry 

end minimum_value_expression_l148_148317


namespace function_form_l148_148038

def satisfies_condition (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 → n > 0 → ⌊ (f (m * n) : ℚ) / n ⌋ = f m

theorem function_form (f : ℕ → ℤ) (h : satisfies_condition f) :
  ∃ r : ℝ, ∀ n : ℕ, 
    (f n = ⌊ (r * n : ℝ) ⌋) ∨ (f n = ⌈ (r * n : ℝ) ⌉ - 1) := 
  sorry

end function_form_l148_148038


namespace seating_arrangement_7_people_l148_148093

theorem seating_arrangement_7_people (n : Nat) (h1 : n = 7) :
  let m := n - 1
  (m.factorial / m) * 2 = 240 :=
by
  sorry

end seating_arrangement_7_people_l148_148093


namespace youngest_age_is_29_l148_148746

-- Define that the ages form an arithmetic sequence
def arithmetic_sequence (a1 a2 a3 a4 : ℕ) : Prop :=
  ∃ (d : ℕ), a2 = a1 + d ∧ a3 = a1 + 2*d ∧ a4 = a1 + 3*d

-- Define the problem statement
theorem youngest_age_is_29 (a1 a2 a3 a4 : ℕ) (h_seq : arithmetic_sequence a1 a2 a3 a4) (h_oldest : a4 = 50) (h_sum : a1 + a2 + a3 + a4 = 158) :
  a1 = 29 :=
by
  sorry

end youngest_age_is_29_l148_148746


namespace ellipse_foci_y_axis_range_l148_148274

theorem ellipse_foci_y_axis_range (k : ℝ) : 
  (2*k - 1 > 2 - k) → (2 - k > 0) → (1 < k ∧ k < 2) := 
by 
  intros h1 h2
  -- We use the assumptions to derive the target statement.
  sorry

end ellipse_foci_y_axis_range_l148_148274


namespace sqrt_one_sixty_four_l148_148414

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 :=
sorry

end sqrt_one_sixty_four_l148_148414


namespace complex_z_modulus_l148_148779

open Complex

theorem complex_z_modulus (z : ℂ) (h1 : (z + 2 * I).re = z + 2 * I) (h2 : (z / (2 - I)).re = z / (2 - I)) :
  (z = 4 - 2 * I) ∧ abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end complex_z_modulus_l148_148779


namespace find_y_given_conditions_l148_148387

theorem find_y_given_conditions (x y : ℝ) (hx : x = 102) 
                                (h : x^3 * y - 3 * x^2 * y + 3 * x * y = 106200) : 
  y = 10 / 97 :=
by
  sorry

end find_y_given_conditions_l148_148387


namespace proof_by_contradiction_l148_148094

-- Definitions for the conditions
inductive ContradictionType
| known          -- ① Contradictory to what is known
| assumption     -- ② Contradictory to the assumption
| definitions    -- ③ Contradictory to definitions, theorems, axioms, laws
| facts          -- ④ Contradictory to facts

open ContradictionType

-- Proving that in proof by contradiction, a contradiction can be of type 1, 2, 3, or 4
theorem proof_by_contradiction :
  (∃ ct : ContradictionType, 
    ct = known ∨ 
    ct = assumption ∨ 
    ct = definitions ∨ 
    ct = facts) :=
by
  sorry

end proof_by_contradiction_l148_148094


namespace math_problem_l148_148472

variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def odd_function (g : ℝ → ℝ) := ∀ x : ℝ, g x = -g (-x)

theorem math_problem
  (hf_even : even_function f)
  (hf_0 : f 0 = 1)
  (hg_odd : odd_function g)
  (hgf : ∀ x : ℝ, g x = f (x - 1)) :
  f 2011 + f 2012 + f 2013 = 1 := sorry

end math_problem_l148_148472


namespace value_of_y_l148_148333

theorem value_of_y (x y : ℝ) (hx : x = 3) (h : x^(3 * y) = 9) : y = 2 / 3 := by
  sorry

end value_of_y_l148_148333


namespace integer_average_problem_l148_148783

theorem integer_average_problem (a b c d : ℤ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
(h_max : max a (max b (max c d)) = 90) (h_min : min a (min b (min c d)) = 29) : 
(a + b + c + d) / 4 = 45 := 
sorry

end integer_average_problem_l148_148783


namespace toys_produced_per_day_l148_148683

theorem toys_produced_per_day :
  (3400 / 5 = 680) :=
by
  sorry

end toys_produced_per_day_l148_148683


namespace range_of_x_satisfying_inequality_l148_148302

def otimes (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x_satisfying_inequality :
  { x : ℝ | otimes x (x - 2) < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end range_of_x_satisfying_inequality_l148_148302


namespace combined_students_yellow_blue_l148_148195

theorem combined_students_yellow_blue {total_students blue_percent red_percent yellow_combined : ℕ} :
  total_students = 200 →
  blue_percent = 30 →
  red_percent = 40 →
  yellow_combined = (total_students * 3 / 10) + ((total_students - (total_students * 3 / 10)) * 6 / 10) →
  yellow_combined = 144 :=
by
  intros
  sorry

end combined_students_yellow_blue_l148_148195


namespace inequality_proof_l148_148006

variables {a1 a2 a3 b1 b2 b3 : ℝ}

theorem inequality_proof (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) 
                         (h4 : 0 < b1) (h5 : 0 < b2) (h6 : 0 < b3):
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 
  ≥ 4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) := 
sorry

end inequality_proof_l148_148006


namespace total_annual_cost_l148_148106

def daily_pills : ℕ := 2
def pill_cost : ℕ := 5
def medication_cost (daily_pills : ℕ) (pill_cost : ℕ) : ℕ := daily_pills * pill_cost
def insurance_coverage : ℚ := 0.80
def visit_cost : ℕ := 400
def visits_per_year : ℕ := 2
def annual_medication_cost (medication_cost : ℕ) (insurance_coverage : ℚ) : ℚ :=
  medication_cost * 365 * (1 - insurance_coverage)
def annual_visit_cost (visit_cost : ℕ) (visits_per_year : ℕ) : ℕ :=
  visit_cost * visits_per_year

theorem total_annual_cost : annual_medication_cost (medication_cost daily_pills pill_cost) insurance_coverage
  + annual_visit_cost visit_cost visits_per_year = 1530 := by
  sorry

end total_annual_cost_l148_148106


namespace paths_from_A_to_B_l148_148012

def path_count_A_to_B : Nat :=
  let red_to_blue_ways := [2, 3]  -- 2 ways to first blue, 3 ways to second blue
  let blue_to_green_ways_first := 4 * 2  -- Each of the 2 green arrows from first blue, 4 ways each
  let blue_to_green_ways_second := 5 * 2 -- Each of the 2 green arrows from second blue, 5 ways each
  let green_to_B_ways_first := 2 * blue_to_green_ways_first  -- Each of the first green, 2 ways each
  let green_to_B_ways_second := 3 * blue_to_green_ways_second  -- Each of the second green, 3 ways each
  green_to_B_ways_first + green_to_B_ways_second  -- Total paths from green arrows to B

theorem paths_from_A_to_B : path_count_A_to_B = 46 := by
  sorry

end paths_from_A_to_B_l148_148012


namespace range_of_m_l148_148570

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m
  (m : ℝ)
  (hθ : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2)
  (h : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) :
  m < 1 :=
by
  sorry

end range_of_m_l148_148570


namespace unique_triangle_with_consecutive_sides_and_angle_condition_l148_148750

theorem unique_triangle_with_consecutive_sides_and_angle_condition
    (a b c : ℕ) (A B C : ℝ) (h1 : a < b ∧ b < c)
    (h2 : b = a + 1 ∧ c = a + 2)
    (h3 : C = 2 * B)
    (h4 : ∀ x y z : ℕ, x < y ∧ y < z → y = x + 1 ∧ z = x + 2 → 2 * B = C)
    : ∃! (a b c : ℕ) (A B C : ℝ), (a < b ∧ b < c) ∧ (b = a + 1 ∧ c = a + 2) ∧ (C = 2 * B) :=
  sorry

end unique_triangle_with_consecutive_sides_and_angle_condition_l148_148750


namespace percentage_decrease_l148_148762

theorem percentage_decrease (P : ℝ) (new_price : ℝ) (x : ℝ) (h1 : new_price = 320) (h2 : P = 421.05263157894734) : x = 24 :=
by
  sorry

end percentage_decrease_l148_148762


namespace tournament_players_l148_148220

theorem tournament_players (n : ℕ) :
  (∃ k : ℕ, k = n + 12 ∧
    -- Exactly one-third of the points earned by each player were earned against the twelve players with the least number of points.
    (2 * (1 / 3 * (n * (n - 1) / 2)) + 2 / 3 * 66 + 66 = (k * (k - 1)) / 2) ∧
    --- Solving the quadratic equation derived
    (n = 4)) → 
    k = 16 :=
by
  sorry

end tournament_players_l148_148220


namespace cyclic_quad_angles_l148_148721

theorem cyclic_quad_angles (A B C D : ℝ) (x : ℝ)
  (h_ratio : A = 5 * x ∧ B = 6 * x ∧ C = 4 * x)
  (h_cyclic : A + D = 180 ∧ B + C = 180):
  (B = 108) ∧ (C = 72) :=
by
  sorry

end cyclic_quad_angles_l148_148721


namespace cocos_August_bill_l148_148384

noncomputable def total_cost (a_monthly_cost: List (Float × Float)) :=
a_monthly_cost.foldr (fun x acc => (x.1 * x.2 * 0.09) + acc) 0

theorem cocos_August_bill :
  let oven        := (2.4, 25)
  let air_cond    := (1.6, 150)
  let refrigerator := (0.15, 720)
  let washing_mach := (0.5, 20) 
  total_cost [oven, air_cond, refrigerator, washing_mach] = 37.62 :=
by
  sorry

end cocos_August_bill_l148_148384


namespace james_older_brother_age_l148_148460

def johnAge : ℕ := 39

def ageCondition (johnAge : ℕ) (jamesAgeIn6 : ℕ) : Prop :=
  johnAge - 3 = 2 * jamesAgeIn6

def jamesOlderBrother (james : ℕ) : ℕ :=
  james + 4

theorem james_older_brother_age (johnAge jamesOlderBrotherAge : ℕ) (james : ℕ) :
  johnAge = 39 →
  (johnAge - 3 = 2 * (james + 6)) →
  jamesOlderBrotherAge = jamesOlderBrother james →
  jamesOlderBrotherAge = 16 :=
by
  sorry

end james_older_brother_age_l148_148460


namespace alex_charge_per_trip_l148_148328

theorem alex_charge_per_trip (x : ℝ)
  (savings_needed : ℝ) (n_trips : ℝ) (worth_groceries : ℝ) (charge_per_grocery_percent : ℝ) :
  savings_needed = 100 → 
  n_trips = 40 →
  worth_groceries = 800 →
  charge_per_grocery_percent = 0.05 →
  n_trips * x + charge_per_grocery_percent * worth_groceries = savings_needed →
  x = 1.5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end alex_charge_per_trip_l148_148328


namespace largest_number_l148_148342

theorem largest_number (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29) :
  d = 21 := 
sorry

end largest_number_l148_148342


namespace carrie_spent_l148_148137

-- Definitions derived from the problem conditions
def cost_of_one_tshirt : ℝ := 9.65
def number_of_tshirts : ℕ := 12

-- The statement to prove
theorem carrie_spent :
  cost_of_one_tshirt * number_of_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l148_148137


namespace max_distance_origin_perpendicular_bisector_l148_148799

theorem max_distance_origin_perpendicular_bisector :
  ∀ (k m : ℝ), k ≠ 0 → 
  (|m| = Real.sqrt (1 + k^2)) → 
  ∃ (d : ℝ), d = 4 / 3 :=
by
  sorry

end max_distance_origin_perpendicular_bisector_l148_148799


namespace infinite_series_value_l148_148794

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3)) = 1 / 2 :=
sorry

end infinite_series_value_l148_148794


namespace probability_distribution_m_l148_148255

theorem probability_distribution_m (m : ℚ) : 
  (m + m / 2 + m / 3 + m / 4 = 1) → m = 12 / 25 :=
by sorry

end probability_distribution_m_l148_148255


namespace mehki_age_l148_148671

theorem mehki_age (Z J M : ℕ) (h1 : Z = 6) (h2 : J = Z - 4) (h3 : M = 2 * (J + Z)) : M = 16 := by
  sorry

end mehki_age_l148_148671


namespace rectangle_area_l148_148718

theorem rectangle_area (w l d : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : d = 10)
  (h3 : d^2 = w^2 + l^2) : 
  l * w = 40 := 
by
  sorry

end rectangle_area_l148_148718


namespace compare_negative_sqrt_values_l148_148493

theorem compare_negative_sqrt_values : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := 
sorry

end compare_negative_sqrt_values_l148_148493


namespace base_conversion_proof_l148_148996

-- Definitions of the base-converted numbers
def b1463_7 := 3 * 7^0 + 6 * 7^1 + 4 * 7^2 + 1 * 7^3  -- 1463 in base 7
def b121_5 := 1 * 5^0 + 2 * 5^1 + 1 * 5^2  -- 121 in base 5
def b1754_6 := 4 * 6^0 + 5 * 6^1 + 7 * 6^2 + 1 * 6^3  -- 1754 in base 6
def b3456_7 := 6 * 7^0 + 5 * 7^1 + 4 * 7^2 + 3 * 7^3  -- 3456 in base 7

-- Formalizing the proof goal
theorem base_conversion_proof : (b1463_7 / b121_5 : ℤ) - b1754_6 * 2 + b3456_7 = 278 := by
  sorry  -- Proof is omitted

end base_conversion_proof_l148_148996


namespace geometric_probability_l148_148664

noncomputable def probability_point_within_rectangle (l w : ℝ) (A_rectangle A_circle : ℝ) : ℝ :=
  A_rectangle / A_circle

theorem geometric_probability (l w : ℝ) (r : ℝ) (A_rectangle : ℝ) (h_length : l = 4) 
  (h_width : w = 3) (h_radius : r = 2.5) (h_area_rectangle : A_rectangle = 12) :
  A_rectangle / (Real.pi * r^2) = 48 / (25 * Real.pi) :=
by
  sorry

end geometric_probability_l148_148664


namespace ellipse_standard_equation_l148_148248

theorem ellipse_standard_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (-4)^2 / a^2 + 3^2 / b^2 = 1) 
    (h4 : a^2 = b^2 + 5^2) : 
    ∃ (a b : ℝ), a^2 = 40 ∧ b^2 = 15 ∧ 
    (∀ x y : ℝ, x^2 / 40 + y^2 / 15 = 1 → (∃ f1 f2 : ℝ, f1 = 5 ∧ f2 = -5)) :=
by {
    sorry
}

end ellipse_standard_equation_l148_148248


namespace find_two_digit_number_l148_148000

def tens_digit (n: ℕ) := n / 10
def unit_digit (n: ℕ) := n % 10
def is_required_number (n: ℕ) : Prop :=
  tens_digit n + 2 = unit_digit n ∧ n < 30 ∧ 10 ≤ n

theorem find_two_digit_number (n : ℕ) :
  is_required_number n → n = 13 ∨ n = 24 :=
by
  -- Proof placeholder
  sorry

end find_two_digit_number_l148_148000


namespace real_solutions_of_equation_l148_148459

theorem real_solutions_of_equation :
  (∃! x : ℝ, (5 * x) / (x^2 + 2 * x + 4) + (6 * x) / (x^2 - 6 * x + 4) = -4 / 3) :=
sorry

end real_solutions_of_equation_l148_148459


namespace evaluate_nested_radical_l148_148630

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l148_148630


namespace average_price_of_pen_l148_148268

theorem average_price_of_pen (c_total : ℝ) (n_pens n_pencils : ℕ) (p_pencil : ℝ)
  (h1 : c_total = 450) (h2 : n_pens = 30) (h3 : n_pencils = 75) (h4 : p_pencil = 2) :
  (c_total - (n_pencils * p_pencil)) / n_pens = 10 :=
by
  sorry

end average_price_of_pen_l148_148268


namespace total_teeth_cleaned_l148_148727

/-
  Given:
   1. Dogs have 42 teeth.
   2. Cats have 30 teeth.
   3. Pigs have 28 teeth.
   4. There are 5 dogs.
   5. There are 10 cats.
   6. There are 7 pigs.
  Prove: The total number of teeth Vann will clean today is 706.
-/

theorem total_teeth_cleaned :
  let dogs: Nat := 5
  let cats: Nat := 10
  let pigs: Nat := 7
  let dog_teeth: Nat := 42
  let cat_teeth: Nat := 30
  let pig_teeth: Nat := 28
  (dogs * dog_teeth) + (cats * cat_teeth) + (pigs * pig_teeth) = 706 := by
  -- Proof goes here
  sorry

end total_teeth_cleaned_l148_148727


namespace b_divisible_by_8_l148_148821

theorem b_divisible_by_8 (b : ℕ) (h_even: ∃ k : ℕ, b = 2 * k) (h_square: ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (b ^ n - 1) / (b - 1) = m ^ 2) : b % 8 = 0 := 
by
  sorry

end b_divisible_by_8_l148_148821


namespace total_pieces_gum_is_correct_l148_148281

-- Define the number of packages and pieces per package
def packages : ℕ := 27
def pieces_per_package : ℕ := 18

-- Define the total number of pieces of gum Robin has
def total_pieces_gum : ℕ :=
  packages * pieces_per_package

-- State the theorem and proof obligation
theorem total_pieces_gum_is_correct : total_pieces_gum = 486 := by
  -- Proof omitted
  sorry

end total_pieces_gum_is_correct_l148_148281


namespace difference_in_ages_27_l148_148653

def conditions (a b : ℕ) : Prop :=
  10 * b + a = (1 / 2) * (10 * a + b) + 6 ∧
  10 * a + b + 2 = 5 * (10 * b + a - 4)

theorem difference_in_ages_27 {a b : ℕ} (h : conditions a b) :
  (10 * a + b) - (10 * b + a) = 27 :=
sorry

end difference_in_ages_27_l148_148653


namespace d_n_2_d_n_3_l148_148042

def d (n k : ℕ) : ℕ :=
  if k = 0 then 1
  else if n = 1 then 0
  else (0:ℕ) -- Placeholder to demonstrate that we need a recurrence relation, not strictly necessary here for the statement.

theorem d_n_2 (n : ℕ) (hn : n ≥ 2) : 
  d n 2 = (n^2 - 3*n + 2) / 2 := 
by 
  sorry

theorem d_n_3 (n : ℕ) (hn : n ≥ 3) : 
  d n 3 = (n^3 - 7*n + 6) / 6 := 
by 
  sorry

end d_n_2_d_n_3_l148_148042


namespace min_area_triangle_ABC_l148_148922

theorem min_area_triangle_ABC :
  let A := (0, 0) 
  let B := (42, 18)
  (∃ p q : ℤ, let C := (p, q) 
              ∃ area : ℝ, area = (1 / 2 : ℝ) * |42 * q - 18 * p| 
              ∧ area = 3) := 
sorry

end min_area_triangle_ABC_l148_148922


namespace worker_followed_instructions_l148_148196

def initial_trees (grid_size : ℕ) : ℕ := grid_size * grid_size

noncomputable def rows_of_trees (rows left each_row : ℕ) : ℕ := rows * each_row

theorem worker_followed_instructions :
  initial_trees 7 = 49 →
  rows_of_trees 5 20 4 = 20 →
  rows_of_trees 5 10 4 = 39 →
  (∃ T : Finset (Fin 7 × Fin 7), T.card = 10) :=
by
  sorry

end worker_followed_instructions_l148_148196


namespace p_sufficient_but_not_necessary_for_q_l148_148324

def condition_p (x : ℝ) : Prop := x^2 - 9 > 0
def condition_q (x : ℝ) : Prop := x^2 - (5 / 6) * x + (1 / 6) > 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x, condition_p x → condition_q x) ∧ ¬(∀ x, condition_q x → condition_p x) :=
sorry

end p_sufficient_but_not_necessary_for_q_l148_148324


namespace value_of_V3_l148_148442

-- Define the polynomial function using Horner's rule
def f (x : ℤ) := (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Define the value of x
def x : ℤ := 2

-- Prove the value of V_3 when x = 2
theorem value_of_V3 : f x = 12 := by
  sorry

end value_of_V3_l148_148442


namespace david_chemistry_marks_l148_148756

theorem david_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℝ)
  (average_marks: ℝ) (marks_english_val: marks_english = 72) (marks_math_val: marks_math = 45)
  (marks_physics_val: marks_physics = 72) (marks_biology_val: marks_biology = 75)
  (average_marks_val: average_marks = 68.2) : 
  ∃ marks_chemistry : ℝ, (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) / 5 = average_marks ∧ 
    marks_chemistry = 77 := 
by
  sorry

end david_chemistry_marks_l148_148756


namespace probability_girl_selection_l148_148141

-- Define the conditions
def total_candidates : ℕ := 3 + 1
def girl_candidates : ℕ := 1

-- Define the question in terms of probability
def probability_of_selecting_girl (total: ℕ) (girl: ℕ) : ℚ :=
  girl / total

-- Lean statement to prove
theorem probability_girl_selection : probability_of_selecting_girl total_candidates girl_candidates = 1 / 4 :=
by
  sorry

end probability_girl_selection_l148_148141


namespace tangent_lines_parallel_to_line_l148_148730

theorem tangent_lines_parallel_to_line (a : ℝ) (b : ℝ)
  (h1 : b = a^3 + a - 2)
  (h2 : 3 * a^2 + 1 = 4) :
  (b = 4 * a - 4 ∨ b = 4 * a) :=
sorry

end tangent_lines_parallel_to_line_l148_148730


namespace angle_in_second_quadrant_l148_148177

def inSecondQuadrant (θ : ℤ) : Prop :=
  90 < θ ∧ θ < 180

theorem angle_in_second_quadrant :
  ∃ k : ℤ, inSecondQuadrant (-2015 + 360 * k) :=
by {
  sorry
}

end angle_in_second_quadrant_l148_148177


namespace unique_identity_function_l148_148836

theorem unique_identity_function (f : ℝ → ℝ) (H : ∀ x y z : ℝ, (x^3 + f y * x + f z = 0) → (f x ^ 3 + y * f x + z = 0)) :
  f = id :=
by sorry

end unique_identity_function_l148_148836


namespace binom_12_6_eq_924_l148_148819

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l148_148819


namespace geom_seq_a3_a5_product_l148_148969

-- Defining the conditions: a sequence and its sum formula
def geom_seq (a : ℕ → ℕ) := ∃ r : ℕ, ∀ n, a (n+1) = a n * r

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n-1) + a 1

-- The theorem statement
theorem geom_seq_a3_a5_product (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : geom_seq a) (h2 : sum_first_n_terms a S) : a 3 * a 5 = 16 := 
sorry

end geom_seq_a3_a5_product_l148_148969


namespace digits_base8_2015_l148_148600

theorem digits_base8_2015 : ∃ n : Nat, (8^n ≤ 2015 ∧ 2015 < 8^(n+1)) ∧ n + 1 = 4 := 
by 
  sorry

end digits_base8_2015_l148_148600


namespace train_speed_clicks_l148_148541

theorem train_speed_clicks (x : ℝ) (v : ℝ) (t : ℝ) 
  (h1 : v = x * 5280 / 60) 
  (h2 : t = 25) 
  (h3 : 70 * t = v * 25) : v = 70 := sorry

end train_speed_clicks_l148_148541


namespace daniel_stickers_l148_148976

def stickers_data 
    (total_stickers : Nat)
    (fred_extra : Nat)
    (andrew_kept : Nat) : Prop :=
  total_stickers = 750 ∧ fred_extra = 120 ∧ andrew_kept = 130

theorem daniel_stickers (D : Nat) :
  stickers_data 750 120 130 → D + (D + 120) = 750 - 130 → D = 250 :=
by
  intros h_data h_eq
  sorry

end daniel_stickers_l148_148976


namespace butterfingers_count_l148_148075

theorem butterfingers_count (total_candy_bars : ℕ) (snickers : ℕ) (mars_bars : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_mars : mars_bars = 2) : 
  ∃ (butterfingers : ℕ), butterfingers = 7 :=
by
  sorry

end butterfingers_count_l148_148075


namespace sufficient_cond_l148_148774

theorem sufficient_cond (x : ℝ) (h : 1/x > 2) : x < 1/2 := 
by {
  sorry 
}

end sufficient_cond_l148_148774


namespace customers_served_total_l148_148320

theorem customers_served_total :
  let Ann_hours := 8
  let Ann_rate := 7
  let Becky_hours := 7
  let Becky_rate := 8
  let Julia_hours := 6
  let Julia_rate := 6
  let lunch_break := 0.5
  let Ann_customers := (Ann_hours - lunch_break) * Ann_rate
  let Becky_customers := (Becky_hours - lunch_break) * Becky_rate
  let Julia_customers := (Julia_hours - lunch_break) * Julia_rate
  Ann_customers + Becky_customers + Julia_customers = 137 := by
  sorry

end customers_served_total_l148_148320


namespace minimum_value_l148_148371

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 / x + 9 / y) = 16 :=
sorry

end minimum_value_l148_148371


namespace length_of_PR_l148_148107

-- Define the entities and conditions
variables (x y : ℝ)
variables (xy_area : ℝ := 125)
variables (PR_length : ℝ := 10 * Real.sqrt 5)

-- State the problem in Lean
theorem length_of_PR (x y : ℝ) (hxy : x * y = 125) :
  x^2 + (125 / x)^2 = (10 * Real.sqrt 5)^2 :=
sorry

end length_of_PR_l148_148107


namespace trigonometric_identity_l148_148111

theorem trigonometric_identity (x : ℝ) (h : Real.tan (x + Real.pi / 2) = 5) : 
  1 / (Real.sin x * Real.cos x) = -26 / 5 :=
by
  sorry

end trigonometric_identity_l148_148111


namespace inverse_proposition_of_parallel_lines_l148_148057

theorem inverse_proposition_of_parallel_lines 
  (P : Prop) (Q : Prop) 
  (h : P ↔ Q) : 
  (Q ↔ P) :=
by 
  sorry

end inverse_proposition_of_parallel_lines_l148_148057


namespace tom_total_dimes_l148_148319

-- Define the original and additional dimes Tom received.
def original_dimes : ℕ := 15
def additional_dimes : ℕ := 33

-- Define the total number of dimes Tom has now.
def total_dimes : ℕ := original_dimes + additional_dimes

-- Statement to prove that the total number of dimes Tom has is 48.
theorem tom_total_dimes : total_dimes = 48 := by
  sorry

end tom_total_dimes_l148_148319


namespace janet_home_time_l148_148672

def blocks_north := 3
def blocks_west := 7 * blocks_north
def blocks_south := blocks_north
def blocks_east := 2 * blocks_south -- Initially mistaken, recalculating needed
def remaining_blocks_west := blocks_west - blocks_east
def total_blocks_home := blocks_south + remaining_blocks_west
def walking_speed := 2 -- blocks per minute

theorem janet_home_time :
  (blocks_south + remaining_blocks_west) / walking_speed = 9 := by
  -- We assume that Lean can handle the arithmetic properly here.
  sorry

end janet_home_time_l148_148672


namespace prove_b_value_l148_148395

theorem prove_b_value (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end prove_b_value_l148_148395


namespace not_necessarily_a_squared_lt_b_squared_l148_148824
-- Import the necessary library

-- Define the variables and the condition
variables {a b : ℝ}
axiom h : a < b

-- The theorem statement that needs to be proved/disproved
theorem not_necessarily_a_squared_lt_b_squared (a b : ℝ) (h : a < b) : ¬ (a^2 < b^2) :=
sorry

end not_necessarily_a_squared_lt_b_squared_l148_148824


namespace points_on_circle_l148_148375

theorem points_on_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1);
  let y := (2 * t^3) / (t^3 + 1);
  x^2 + y^2 = 1 :=
by
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2 * t^3) / (t^3 + 1)
  have h1 : x^2 + y^2 = ((t^3 - 1) / (t^3 + 1))^2 + ((2 * t^3) / (t^3 + 1))^2 := by rfl
  have h2 : (x^2 + y^2) = ( (t^3 - 1)^2 + (2 * t^3)^2 ) / (t^3 + 1)^2 := by sorry
  have h3 : (x^2 + y^2) = ( t^6 - 2 * t^3 + 1 + 4 * t^6 ) / (t^3 + 1)^2 := by sorry
  have h4 : (x^2 + y^2) = 1 := by sorry
  exact h4

end points_on_circle_l148_148375


namespace probability_of_one_red_ball_is_one_third_l148_148903

-- Define the number of red and black balls
def red_balls : Nat := 2
def black_balls : Nat := 4
def total_balls : Nat := red_balls + black_balls

-- Define the probability calculation
def probability_red_ball : ℚ := red_balls / (red_balls + black_balls)

-- State the theorem
theorem probability_of_one_red_ball_is_one_third :
  probability_red_ball = 1 / 3 :=
by
  sorry

end probability_of_one_red_ball_is_one_third_l148_148903


namespace nap_time_is_correct_l148_148135

-- Define the total trip time and the hours spent on each activity
def total_trip_time : ℝ := 15
def reading_time : ℝ := 2
def eating_time : ℝ := 1
def movies_time : ℝ := 3
def chatting_time : ℝ := 1
def browsing_time : ℝ := 0.75
def waiting_time : ℝ := 0.5
def working_time : ℝ := 2

-- Define the total activity time
def total_activity_time : ℝ := reading_time + eating_time + movies_time + chatting_time + browsing_time + waiting_time + working_time

-- Define the nap time as the difference between total trip time and total activity time
def nap_time : ℝ := total_trip_time - total_activity_time

-- Prove that the nap time is 4.75 hours
theorem nap_time_is_correct : nap_time = 4.75 :=
by
  -- Calculation hint, can be ignored
  -- nap_time = 15 - (2 + 1 + 3 + 1 + 0.75 + 0.5 + 2) = 15 - 10.25 = 4.75
  sorry

end nap_time_is_correct_l148_148135


namespace angle_RPS_is_27_l148_148701

theorem angle_RPS_is_27 (PQ BP PR QS QS PSQ QPRS : ℝ) :
  PQ + PSQ + QS = 180 ∧ 
  QS = 48 ∧ 
  PSQ = 38 ∧ 
  QPRS = 67
  → (QS - QPRS = 27) := 
by {
  sorry
}

end angle_RPS_is_27_l148_148701


namespace determine_b_l148_148476

theorem determine_b (A B C : ℝ) (a b c : ℝ)
  (angle_C_eq_4A : C = 4 * A)
  (a_eq_30 : a = 30)
  (c_eq_48 : c = 48)
  (law_of_sines : ∀ x y, x / Real.sin A = y / Real.sin (4 * A))
  (cos_eq_solution : 4 * Real.cos A ^ 3 - 4 * Real.cos A = 8 / 5) :
  ∃ b : ℝ, b = 30 * (5 - 20 * (1 - Real.cos A ^ 2) + 16 * (1 - Real.cos A ^ 2) ^ 2) :=
by 
  sorry

end determine_b_l148_148476


namespace toy_store_revenue_fraction_l148_148749

theorem toy_store_revenue_fraction (N D J : ℝ) 
  (h1 : J = N / 3) 
  (h2 : D = 3.75 * (N + J) / 2) : 
  (N / D) = 2 / 5 :=
by sorry

end toy_store_revenue_fraction_l148_148749


namespace packs_used_after_6_weeks_l148_148737

-- Define the conditions as constants or definitions.
def pages_per_class_per_day : ℕ := 2
def num_classes : ℕ := 5
def days_per_week : ℕ := 5
def weeks : ℕ := 6
def pages_per_pack : ℕ := 100

-- The total number of packs of notebook paper Chip will use after 6 weeks
theorem packs_used_after_6_weeks : (pages_per_class_per_day * num_classes * days_per_week * weeks) / pages_per_pack = 3 := 
by
  -- skip the proof
  sorry

end packs_used_after_6_weeks_l148_148737


namespace distance_traveled_l148_148029

theorem distance_traveled (speed time : ℕ) (h_speed : speed = 20) (h_time : time = 8) : 
  speed * time = 160 := 
by
  -- Solution proof goes here
  sorry

end distance_traveled_l148_148029


namespace max_fraction_l148_148244

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) : 
  ∃ k, k = (x + y) / x ∧ k ≤ -2 := 
sorry

end max_fraction_l148_148244


namespace mark_total_young_fish_l148_148223

-- Define the conditions
def num_tanks : ℕ := 5
def fish_per_tank : ℕ := 6
def young_per_fish : ℕ := 25

-- Define the total number of young fish
def total_young_fish := num_tanks * fish_per_tank * young_per_fish

-- The theorem statement
theorem mark_total_young_fish : total_young_fish = 750 :=
  by
    sorry

end mark_total_young_fish_l148_148223


namespace miles_driven_on_Monday_l148_148119

def miles_Tuesday : ℕ := 18
def miles_Wednesday : ℕ := 21
def avg_miles_per_day : ℕ := 17

theorem miles_driven_on_Monday (miles_Monday : ℕ) :
  (miles_Monday + miles_Tuesday + miles_Wednesday) / 3 = avg_miles_per_day →
  miles_Monday = 12 :=
by
  intro h
  sorry

end miles_driven_on_Monday_l148_148119


namespace remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l148_148728

theorem remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0:
  (7 * 12^24 + 3^24) % 11 = 0 := sorry

end remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l148_148728


namespace max_distance_right_triangle_l148_148546

theorem max_distance_right_triangle (a b : ℝ) 
  (h1: ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
    (a * A.1 + 2 * b * A.2 = 1) ∧ (a * B.1 + 2 * b * B.2 = 1) ∧ 
    ∃ (C : ℝ × ℝ), C = (0,0) ∧ (A.1 * B.1 + A.2 * B.2 = 0)): 
  ∃ (d : ℝ), d = (Real.sqrt (a^2 + b^2)) ∧ d ≤ Real.sqrt 2 :=
sorry

end max_distance_right_triangle_l148_148546


namespace min_sum_of_factors_l148_148974

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 1806) (h2 : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) : a + b + c ≥ 112 :=
sorry

end min_sum_of_factors_l148_148974


namespace largest_of_three_consecutive_integers_l148_148643

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l148_148643


namespace minimum_of_quadratic_l148_148424

theorem minimum_of_quadratic : ∀ x : ℝ, 1 ≤ x^2 - 6 * x + 10 :=
by
  intro x
  have h : x^2 - 6 * x + 10 = (x - 3)^2 + 1 := by ring
  rw [h]
  have h_nonneg : (x - 3)^2 ≥ 0 := by apply sq_nonneg
  linarith

end minimum_of_quadratic_l148_148424


namespace part1_A_intersect_B_l148_148487

def setA : Set ℝ := { x | x ^ 2 - 2 * x - 3 ≤ 0 }
def setB (m : ℝ) : Set ℝ := { x | (x - (m - 1)) * (x - (m + 1)) > 0 }

theorem part1_A_intersect_B (m : ℝ) (h : m = 0) : 
  setA ∩ setB m = { x | 1 < x ∧ x ≤ 3 } :=
sorry

end part1_A_intersect_B_l148_148487


namespace simplify_expr_l148_148253

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l148_148253


namespace bacteria_growth_rate_l148_148738

theorem bacteria_growth_rate (B G : ℝ) (h : B * G^16 = 2 * B * G^15) : G = 2 :=
by
  sorry

end bacteria_growth_rate_l148_148738


namespace factorize_l148_148123

theorem factorize (x : ℝ) : 72 * x ^ 11 + 162 * x ^ 22 = 18 * x ^ 11 * (4 + 9 * x ^ 11) :=
by
  sorry

end factorize_l148_148123


namespace least_num_to_divisible_l148_148295

theorem least_num_to_divisible (n : ℕ) : (1056 + n) % 27 = 0 → n = 24 :=
by
  sorry

end least_num_to_divisible_l148_148295


namespace find_number_l148_148881

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 :=
sorry

end find_number_l148_148881


namespace percent_gold_coins_l148_148508

variables (total_objects : ℝ) (coins_beads_percent beads_percent gold_coins_percent : ℝ)
           (h1 : coins_beads_percent = 0.75)
           (h2 : beads_percent = 0.15)
           (h3 : gold_coins_percent = 0.60)

theorem percent_gold_coins : (gold_coins_percent * (coins_beads_percent - beads_percent)) = 0.36 :=
by
  have coins_percent := coins_beads_percent - beads_percent
  have gold_coins_total_percent := gold_coins_percent * coins_percent
  exact sorry

end percent_gold_coins_l148_148508


namespace average_stoppage_time_per_hour_l148_148518

theorem average_stoppage_time_per_hour :
    ∀ (v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl : ℝ),
    v1_excl = 54 → v1_incl = 36 →
    v2_excl = 72 → v2_incl = 48 →
    v3_excl = 90 → v3_incl = 60 →
    ( ((54 / v1_excl - 54 / v1_incl) + (72 / v2_excl - 72 / v2_incl) + (90 / v3_excl - 90 / v3_incl)) / 3 = 0.5 ) := 
by
    intros v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl
    sorry

end average_stoppage_time_per_hour_l148_148518


namespace relationship_y1_y2_y3_l148_148079

-- Define the function y = 3(x + 1)^2 - 8
def quadratic_fn (x : ℝ) : ℝ := 3 * (x + 1)^2 - 8

-- Define points A, B, and C on the graph of the quadratic function
def y1 := quadratic_fn 1
def y2 := quadratic_fn 2
def y3 := quadratic_fn (-2)

-- The goal is to prove the relationship y2 > y1 > y3
theorem relationship_y1_y2_y3 :
  y2 > y1 ∧ y1 > y3 :=
by sorry

end relationship_y1_y2_y3_l148_148079


namespace find_cost_prices_l148_148893

noncomputable def cost_price_per_meter
  (selling_price_per_meter : ℕ) (loss_per_meter : ℕ) : ℕ :=
  selling_price_per_meter + loss_per_meter

theorem find_cost_prices
  (selling_A : ℕ) (meters_A : ℕ) (loss_A : ℕ)
  (selling_B : ℕ) (meters_B : ℕ) (loss_B : ℕ)
  (selling_C : ℕ) (meters_C : ℕ) (loss_C : ℕ)
  (H_A : selling_A = 9000) (H_meters_A : meters_A = 300) (H_loss_A : loss_A = 6)
  (H_B : selling_B = 7000) (H_meters_B : meters_B = 250) (H_loss_B : loss_B = 4)
  (H_C : selling_C = 12000) (H_meters_C : meters_C = 400) (H_loss_C : loss_C = 8) :
  cost_price_per_meter (selling_A / meters_A) loss_A = 36 ∧
  cost_price_per_meter (selling_B / meters_B) loss_B = 32 ∧
  cost_price_per_meter (selling_C / meters_C) loss_C = 38 :=
by {
  sorry
}

end find_cost_prices_l148_148893


namespace no_roots_of_form_one_over_n_l148_148759

theorem no_roots_of_form_one_over_n (a b c : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_c : c % 2 = 1) :
  ∀ n : ℕ, ¬(a * (1 / (n:ℚ))^2 + b * (1 / (n:ℚ)) + c = 0) := by
  sorry

end no_roots_of_form_one_over_n_l148_148759


namespace fraction_of_sum_l148_148234

theorem fraction_of_sum (numbers : List ℝ) (h_len : numbers.length = 21)
  (n : ℝ) (h_n : n ∈ numbers)
  (h_avg : n = 5 * ((numbers.sum - n) / 20)) :
  n / numbers.sum = 1 / 5 :=
by
  sorry

end fraction_of_sum_l148_148234


namespace part1_part2_part3_l148_148202

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

theorem part1 (h_odd : ∀ x : ℝ, f x b = -f (-x) b) : b = 1 :=
sorry

theorem part2 (h_b : b = 1) : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1 :=
sorry

theorem part3 (h_monotonic : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1) 
  : ∀ t : ℝ, f (t^2 - 2 * t) 1 + f (2 * t^2 - k) 1 < 0 → k < -1/3 :=
sorry

end part1_part2_part3_l148_148202


namespace ratio_cars_to_dogs_is_two_l148_148766

-- Definitions of the conditions
def initial_dogs : ℕ := 90
def initial_cars : ℕ := initial_dogs / 3
def additional_cars : ℕ := 210
def current_dogs : ℕ := 120
def current_cars : ℕ := initial_cars + additional_cars

-- The statement to be proven
theorem ratio_cars_to_dogs_is_two :
  (current_cars : ℚ) / (current_dogs : ℚ) = 2 := by
  sorry

end ratio_cars_to_dogs_is_two_l148_148766


namespace cost_of_pink_notebook_l148_148349

theorem cost_of_pink_notebook
    (total_cost : ℕ) 
    (black_cost : ℕ) 
    (green_cost : ℕ) 
    (num_green : ℕ) 
    (num_black : ℕ) 
    (num_pink : ℕ)
    (total_notebooks : ℕ)
    (h_total_cost : total_cost = 45)
    (h_black_cost : black_cost = 15) 
    (h_green_cost : green_cost = 10) 
    (h_num_green : num_green = 2) 
    (h_num_black : num_black = 1) 
    (h_num_pink : num_pink = 1)
    (h_total_notebooks : total_notebooks = 4) 
    : (total_cost - (num_green * green_cost + black_cost) = 10) :=
by
  sorry

end cost_of_pink_notebook_l148_148349


namespace gcd_g_y_l148_148441

def g (y : ℕ) : ℕ := (3*y + 4) * (8*y + 3) * (14*y + 9) * (y + 17)

theorem gcd_g_y (y : ℕ) (h : y % 42522 = 0) : Nat.gcd (g y) y = 102 := by
  sorry

end gcd_g_y_l148_148441


namespace books_remaining_after_second_day_l148_148368

variable (x a b c d : ℕ)

theorem books_remaining_after_second_day :
  let books_borrowed_first_day := a * b
  let books_borrowed_second_day := c
  let books_returned_second_day := (d * books_borrowed_first_day) / 100
  x - books_borrowed_first_day - books_borrowed_second_day + books_returned_second_day =
  x - (a * b) - c + ((d * (a * b)) / 100) :=
sorry

end books_remaining_after_second_day_l148_148368


namespace distance_from_point_to_directrix_l148_148035

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l148_148035


namespace crystal_final_segment_distance_l148_148450

theorem crystal_final_segment_distance :
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2 -- as nx, ny
  let southwest_component := southwest_distance / Real.sqrt 2 -- as sx, sy
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  Real.sqrt (net_north^2 + net_west^2) = 2 * Real.sqrt 3 :=
by
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2
  let southwest_component := southwest_distance / Real.sqrt 2
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  exact sorry

end crystal_final_segment_distance_l148_148450


namespace angle_DNE_l148_148285

theorem angle_DNE (DE EF FD : ℝ) (EFD END FND : ℝ) 
  (h1 : DE = 2 * EF) 
  (h2 : EF = FD) 
  (h3 : EFD = 34) 
  (h4 : END = 3) 
  (h5 : FND = 18) : 
  ∃ DNE : ℝ, DNE = 104 :=
by 
  sorry

end angle_DNE_l148_148285


namespace inequality_solution_l148_148692

theorem inequality_solution :
  {x : ℝ | ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0} = 
  {x : ℝ | (1 < x ∧ x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7)} :=
sorry

end inequality_solution_l148_148692


namespace intersection_in_first_quadrant_l148_148334

theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax - y + 2 = 0 ∧ x + y - a = 0 ∧ x > 0 ∧ y > 0) ↔ a > 2 := 
by
  sorry

end intersection_in_first_quadrant_l148_148334


namespace total_items_and_cost_per_pet_l148_148788

theorem total_items_and_cost_per_pet
  (treats_Jane : ℕ)
  (treats_Wanda : ℕ := treats_Jane / 2)
  (bread_Jane : ℕ := (3 * treats_Jane) / 4)
  (bread_Wanda : ℕ := 90)
  (bread_Carla : ℕ := 40)
  (treats_Carla : ℕ := 5 * bread_Carla / 2)
  (items_Peter : ℕ := 140)
  (treats_Peter : ℕ := items_Peter / 3)
  (bread_Peter : ℕ := 2 * treats_Peter)
  (x y z : ℕ) :
  (∀ B : ℕ, B = bread_Jane + bread_Wanda + bread_Carla + bread_Peter) ∧
  (∀ T : ℕ, T = treats_Jane + treats_Wanda + treats_Carla + treats_Peter) ∧
  (∀ Total : ℕ, Total = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter)) ∧
  (∀ ExpectedTotal : ℕ, ExpectedTotal = 427) ∧
  (∀ Cost : ℕ, Cost = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) * x + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter) * y) ∧
  (∀ CostPerPet : ℕ, CostPerPet = Cost / z) ∧
  (B + T = 427) ∧
  ((Cost / z) = (235 * x + 192 * y) / z)
:=
  by
  sorry

end total_items_and_cost_per_pet_l148_148788


namespace necessary_condition_to_contain_circle_in_parabola_l148_148843

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem necessary_condition_to_contain_circle_in_parabola (a : ℝ) : 
  (∀ x y, N x y a → M x y) ↔ a ≥ 5 / 4 := 
sorry

end necessary_condition_to_contain_circle_in_parabola_l148_148843


namespace at_least_one_greater_than_zero_l148_148170

noncomputable def a (x : ℝ) : ℝ := x^2 - 2 * x + (Real.pi / 2)
noncomputable def b (y : ℝ) : ℝ := y^2 - 2 * y + (Real.pi / 2)
noncomputable def c (z : ℝ) : ℝ := z^2 - 2 * z + (Real.pi / 2)

theorem at_least_one_greater_than_zero (x y z : ℝ) : (a x > 0) ∨ (b y > 0) ∨ (c z > 0) :=
by sorry

end at_least_one_greater_than_zero_l148_148170


namespace vector_properties_l148_148101

-- Definitions of vectors
def vec_a : ℝ × ℝ := (3, 11)
def vec_b : ℝ × ℝ := (-1, -4)
def vec_c : ℝ × ℝ := (1, 3)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Linear combination of vector scaling and addition
def vec_sub_scal (u v : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (u.1 - k * v.1, u.2 - k * v.2)

-- Check if two vectors are parallel
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2

-- Lean statement for the proof problem
theorem vector_properties :
  dot_product vec_a vec_b = -47 ∧
  vec_sub_scal vec_a vec_b 2 = (5, 19) ∧
  dot_product (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2) vec_c ≠ 0 ∧
  parallel (vec_sub_scal vec_a vec_c 1) vec_b :=
by sorry

end vector_properties_l148_148101


namespace ketchup_bottles_count_l148_148002

def ratio_ketchup_mustard_mayo : Nat × Nat × Nat := (3, 3, 2)
def num_mayo_bottles : Nat := 4

theorem ketchup_bottles_count 
  (r : Nat × Nat × Nat)
  (m : Nat)
  (h : r = ratio_ketchup_mustard_mayo)
  (h2 : m = num_mayo_bottles) :
  ∃ k : Nat, k = 6 := by
sorry

end ketchup_bottles_count_l148_148002


namespace largest_nonrepresentable_by_17_11_l148_148966

/--
In the USA, standard letter-size paper is 8.5 inches wide and 11 inches long. The largest integer that cannot be written as a sum of a whole number (possibly zero) of 17's and a whole number (possibly zero) of 11's is 159.
-/
theorem largest_nonrepresentable_by_17_11 : 
  ∀ (a b : ℕ), (∀ (n : ℕ), n = 17 * a + 11 * b -> n ≠ 159) ∧ 
               ¬ (∃ (a b : ℕ), 17 * a + 11 * b = 159) :=
by
  sorry

end largest_nonrepresentable_by_17_11_l148_148966


namespace exist_n_l148_148744

theorem exist_n : ∃ n : ℕ, n > 1 ∧ ¬(Nat.Prime n) ∧ ∀ a : ℤ, (a^n - a) % n = 0 :=
by
  sorry

end exist_n_l148_148744


namespace red_marked_area_on_larger_sphere_l148_148089

-- Define the conditions
def r1 : ℝ := 4 -- radius of the smaller sphere
def r2 : ℝ := 6 -- radius of the larger sphere
def A1 : ℝ := 37 -- area marked on the smaller sphere

-- State the proportional relationship as a Lean theorem
theorem red_marked_area_on_larger_sphere : 
  let A2 := A1 * (r2^2 / r1^2)
  A2 = 83.25 :=
by
  sorry

end red_marked_area_on_larger_sphere_l148_148089


namespace lana_extra_flowers_l148_148363

theorem lana_extra_flowers (tulips roses used total extra : ℕ) 
  (h1 : tulips = 36) 
  (h2 : roses = 37) 
  (h3 : used = 70) 
  (h4 : total = tulips + roses) 
  (h5 : extra = total - used) : 
  extra = 3 := 
sorry

end lana_extra_flowers_l148_148363


namespace ice_cream_flavors_l148_148199

theorem ice_cream_flavors (n k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n + k - 1).choose (k - 1) = 84 :=
by
  have h3 : n = 6 := h1
  have h4 : k = 4 := h2
  rw [h3, h4]
  sorry

end ice_cream_flavors_l148_148199


namespace find_x_l148_148509

theorem find_x (x y z : ℤ) (h1 : 4 * x + y + z = 80) (h2 : 2 * x - y - z = 40) (h3 : 3 * x + y - z = 20) : x = 20 := by
  sorry

end find_x_l148_148509


namespace Mickey_horses_per_week_l148_148491

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l148_148491


namespace magazine_ad_extra_cost_l148_148515

/--
The cost of purchasing a laptop through a magazine advertisement includes four monthly 
payments of $60.99 each and a one-time shipping and handling fee of $19.99. The in-store 
price of the laptop is $259.99. Prove that purchasing the laptop through the magazine 
advertisement results in an extra cost of 396 cents.
-/
theorem magazine_ad_extra_cost : 
  let in_store_price := 259.99
  let monthly_payment := 60.99
  let num_payments := 4
  let shipping_handling := 19.99
  let total_magazine_cost := (num_payments * monthly_payment) + shipping_handling
  (total_magazine_cost - in_store_price) * 100 = 396 := 
by
  sorry

end magazine_ad_extra_cost_l148_148515


namespace variable_cost_per_book_fixed_cost_l148_148401

theorem variable_cost_per_book_fixed_cost (fixed_costs : ℝ) (selling_price_per_book : ℝ) 
(number_of_books : ℝ) (total_costs total_revenue : ℝ) (variable_cost_per_book : ℝ) 
(h1 : fixed_costs = 35630) (h2 : selling_price_per_book = 20.25) (h3 : number_of_books = 4072)
(h4 : total_costs = fixed_costs + variable_cost_per_book * number_of_books)
(h5 : total_revenue = selling_price_per_book * number_of_books)
(h6 : total_costs = total_revenue) : variable_cost_per_book = 11.50 := by
  sorry

end variable_cost_per_book_fixed_cost_l148_148401


namespace am_gm_inequality_l148_148499

open Real

theorem am_gm_inequality (
    a b c d e f : ℝ
) (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_nonneg_c : 0 ≤ c)
  (h_nonneg_d : 0 ≤ d)
  (h_nonneg_e : 0 ≤ e)
  (h_nonneg_f : 0 ≤ f)
  (h_cond_ab : a + b ≤ e)
  (h_cond_cd : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) := 
  by sorry

end am_gm_inequality_l148_148499


namespace math_proof_problem_l148_148579

variable (a d e : ℝ)

theorem math_proof_problem (h1 : a < 0) (h2 : a < d) (h3 : d < e) :
  (a * d < a * e) ∧ (a + d < d + e) ∧ (e / a < 1) :=
by {
  sorry
}

end math_proof_problem_l148_148579


namespace abc_inequality_l148_148190

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
    (ab / (a^5 + ab + b^5)) + (bc / (b^5 + bc + c^5)) + (ca / (c^5 + ca + a^5)) ≤ 1 := 
sorry

end abc_inequality_l148_148190
