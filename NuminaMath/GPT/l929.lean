import Mathlib

namespace necessary_but_not_sufficient_condition_for_x_gt_2_l929_92909

theorem necessary_but_not_sufficient_condition_for_x_gt_2 :
  ∀ (x : ℝ), (2 / x < 1 → x > 2) ∧ (x > 2 → 2 / x < 1) → (¬ (x > 2 → 2 / x < 1) ∨ ¬ (2 / x < 1 → x > 2)) :=
by
  intro x h
  sorry

end necessary_but_not_sufficient_condition_for_x_gt_2_l929_92909


namespace M_empty_iff_k_range_M_interval_iff_k_range_l929_92970

-- Part 1
theorem M_empty_iff_k_range (k : ℝ) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 ≤ 0) ↔ -3 ≤ k ∧ k ≤ 1 / 5 := sorry

-- Part 2
theorem M_interval_iff_k_range (k a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_ab : a < b) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 > 0 ↔ a < x ∧ x < b) ↔ 1 / 5 < k ∧ k < 1 := sorry

end M_empty_iff_k_range_M_interval_iff_k_range_l929_92970


namespace total_cost_textbooks_l929_92990

theorem total_cost_textbooks :
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  sale_books + online_books + bookstore_books = 210 :=
by
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  show sale_books + online_books + bookstore_books = 210
  sorry

end total_cost_textbooks_l929_92990


namespace find_angle_C_max_area_triangle_l929_92992

-- Part I: Proving angle C
theorem find_angle_C (a b c : ℝ) (A B C : ℝ)
    (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
    C = Real.pi / 3 :=
sorry

-- Part II: Finding maximum area of triangle ABC
theorem max_area_triangle (a b : ℝ) (c : ℝ) (h_c : c = 2 * Real.sqrt 3) (A B C : ℝ)
    (h_A : A > 0) (h_B : B > 0) (h_C : C = Real.pi / 3)
    (h : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
    0.5 * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
sorry

end find_angle_C_max_area_triangle_l929_92992


namespace maximum_k_value_l929_92945

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / (x - 1)
noncomputable def g (x : ℝ) (k : ℕ) : ℝ := k / x

theorem maximum_k_value (c : ℝ) (h_c : c > 1) : 
  (∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b 3) ∧ 
  (∀ k : ℕ, k > 3 → ¬ ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b k) :=
sorry

end maximum_k_value_l929_92945


namespace arithmetic_sequence_problem_l929_92960

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : ∀ k, k ≥ 2 → a (k + 1) - a k^2 + a (k - 1) = 0) (h2 : ∀ k, a k ≠ 0) (h3 : ∀ k ≥ 2, a (k + 1) + a (k - 1) = 2 * a k) :
  S (2 * n - 1) - 4 * n = -2 :=
by
  sorry

end arithmetic_sequence_problem_l929_92960


namespace rectangular_plot_dimensions_l929_92939

theorem rectangular_plot_dimensions (a b : ℝ) 
  (h_area : a * b = 800) 
  (h_perimeter_fencing : 2 * a + b = 100) :
  (a = 40 ∧ b = 20) ∨ (a = 10 ∧ b = 80) := 
sorry

end rectangular_plot_dimensions_l929_92939


namespace grid_with_value_exists_possible_values_smallest_possible_value_l929_92979

open Nat

def isGridValuesP (P : ℕ) (a b c d e f g h i : ℕ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = P) ∧ (d * e * f = P) ∧
  (g * h * i = P) ∧ (a * d * g = P) ∧
  (b * e * h = P) ∧ (c * f * i = P)

theorem grid_with_value_exists (P : ℕ) :
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem possible_values (P : ℕ) :
  P ∈ [1992, 1995] ↔ 
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem smallest_possible_value : 
  ∃ P a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i ∧ 
  ∀ Q, (∃ w x y z u v s t q : ℕ, isGridValuesP Q w x y z u v s t q) → Q ≥ 120 :=
sorry

end grid_with_value_exists_possible_values_smallest_possible_value_l929_92979


namespace find_k_l929_92903

theorem find_k (k : ℕ) (h : (64 : ℕ) / k = 4) : k = 16 := by
  sorry

end find_k_l929_92903


namespace price_decrease_l929_92907

theorem price_decrease (current_price original_price : ℝ) (h1 : current_price = 684) (h2 : original_price = 900) :
  ((original_price - current_price) / original_price) * 100 = 24 :=
by
  sorry

end price_decrease_l929_92907


namespace pentagon_interior_angles_l929_92962

theorem pentagon_interior_angles
  (x y : ℝ)
  (H_eq_triangle : ∀ (angle : ℝ), angle = 60)
  (H_rect_QT : ∀ (angle : ℝ), angle = 90)
  (sum_interior_angles_pentagon : ∀ (n : ℕ), (n - 2) * 180 = 3 * 180) :
  x + y = 60 :=
by
  sorry

end pentagon_interior_angles_l929_92962


namespace certain_number_divided_by_10_l929_92922
-- Broad import to bring in necessary libraries

-- Define the constants and hypotheses
variable (x : ℝ)
axiom condition : 5 * x = 100

-- Theorem to prove the required equality
theorem certain_number_divided_by_10 : (x / 10) = 2 :=
by
  -- The proof is skipped by sorry
  sorry

end certain_number_divided_by_10_l929_92922


namespace min_value_expression_l929_92935

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + 4 / y) ≥ 9 :=
sorry

end min_value_expression_l929_92935


namespace blister_slowdown_l929_92984

theorem blister_slowdown
    (old_speed new_speed time : ℕ) (new_speed_initial : ℕ) (blister_freq : ℕ)
    (distance_old : ℕ) (blister_per_hour_slowdown : ℝ):
    -- Given conditions
    old_speed = 6 →
    new_speed = 11 →
    new_speed_initial = 11 →
    time = 4 →
    blister_freq = 2 →
    distance_old = old_speed * time →
    -- Prove that each blister slows Candace down by 10 miles per hour
    blister_per_hour_slowdown = 10 :=
  by
    sorry

end blister_slowdown_l929_92984


namespace probability_first_4_second_club_third_2_l929_92914

theorem probability_first_4_second_club_third_2 :
  let deck_size := 52
  let prob_4_first := 4 / deck_size
  let deck_minus_first_card := deck_size - 1
  let prob_club_second := 13 / deck_minus_first_card
  let deck_minus_two_cards := deck_minus_first_card - 1
  let prob_2_third := 4 / deck_minus_two_cards
  prob_4_first * prob_club_second * prob_2_third = 1 / 663 :=
by
  sorry

end probability_first_4_second_club_third_2_l929_92914


namespace find_solutions_l929_92994

theorem find_solutions :
  ∀ (x n : ℕ), 0 < x → 0 < n → x^(n+1) - (x + 1)^n = 2001 → (x, n) = (13, 2) :=
by
  intros x n hx hn heq
  sorry

end find_solutions_l929_92994


namespace value_of_e_l929_92972

theorem value_of_e (a : ℕ) (e : ℕ) 
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * 45 * e) : 
  e = 49 := 
by 
  sorry

end value_of_e_l929_92972


namespace division_identity_l929_92948

theorem division_identity (h : 6 / 3 = 2) : 72 / (6 / 3) = 36 := by
  sorry

end division_identity_l929_92948


namespace point_side_opposite_l929_92989

def equation_lhs (x y : ℝ) : ℝ := 2 * y - 6 * x + 1

theorem point_side_opposite : 
  (equation_lhs 0 0 * equation_lhs 2 1 < 0) := 
by 
   sorry

end point_side_opposite_l929_92989


namespace find_positive_number_l929_92905

theorem find_positive_number (x n : ℝ) (h₁ : (x + 1) ^ 2 = n) (h₂ : (x - 5) ^ 2 = n) : n = 9 := 
sorry

end find_positive_number_l929_92905


namespace dolphins_score_l929_92967

theorem dolphins_score (S D : ℕ) (h1 : S + D = 48) (h2 : S = D + 20) : D = 14 := by
    sorry

end dolphins_score_l929_92967


namespace octagon_diagonals_l929_92927

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l929_92927


namespace simplify_expression_l929_92911

theorem simplify_expression (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x := 
by
  sorry

end simplify_expression_l929_92911


namespace div_m_by_18_equals_500_l929_92937

-- Define the conditions
noncomputable def m : ℕ := 9000 -- 'm' is given as 9000 since it fulfills all conditions described
def is_multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def all_digits_9_or_0 (n : ℕ) : Prop := ∀ (d : ℕ), (∃ (k : ℕ), n = 10^k * d) → (d = 0 ∨ d = 9)

-- Define the proof problem statement
theorem div_m_by_18_equals_500 
  (h1 : is_multiple_of_18 m) 
  (h2 : all_digits_9_or_0 m) 
  (h3 : ∀ n, is_multiple_of_18 n ∧ all_digits_9_or_0 n → n ≤ m) : 
  m / 18 = 500 :=
sorry

end div_m_by_18_equals_500_l929_92937


namespace dawns_earnings_per_hour_l929_92987

variable (hours_per_painting : ℕ) (num_paintings : ℕ) (total_earnings : ℕ)

def total_hours (hours_per_painting num_paintings : ℕ) : ℕ :=
  hours_per_painting * num_paintings

def earnings_per_hour (total_earnings total_hours : ℕ) : ℕ :=
  total_earnings / total_hours

theorem dawns_earnings_per_hour :
  hours_per_painting = 2 →
  num_paintings = 12 →
  total_earnings = 3600 →
  earnings_per_hour total_earnings (total_hours hours_per_painting num_paintings) = 150 :=
by
  intros h1 h2 h3
  sorry

end dawns_earnings_per_hour_l929_92987


namespace max_integer_value_of_x_l929_92904

theorem max_integer_value_of_x (x : ℤ) : 3 * x - (1 / 4 : ℚ) ≤ (1 / 3 : ℚ) * x - 2 → x ≤ -1 :=
by
  intro h
  sorry

end max_integer_value_of_x_l929_92904


namespace average_coins_per_day_l929_92983

theorem average_coins_per_day :
  let a := 10
  let d := 10
  let n := 7
  let extra := 20
  let total_coins := a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) + (a + 6 * d + extra)
  total_coins = 300 →
  total_coins / n = 300 / 7 :=
by
  sorry

end average_coins_per_day_l929_92983


namespace increasing_sequence_a_range_l929_92999

theorem increasing_sequence_a_range (a : ℝ) (a_seq : ℕ → ℝ) (h_def : ∀ n, a_seq n = 
  if n ≤ 2 then a * n^2 - ((7 / 8) * a + 17 / 4) * n + 17 / 2
  else a ^ n) : 
  (∀ n, a_seq n < a_seq (n + 1)) → a > 2 :=
by
  sorry

end increasing_sequence_a_range_l929_92999


namespace range_of_y_given_x_l929_92936

theorem range_of_y_given_x (x : ℝ) (h₁ : x > 3) : 0 < (6 / x) ∧ (6 / x) < 2 :=
by 
  sorry

end range_of_y_given_x_l929_92936


namespace revenue_increase_20_percent_l929_92944

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q
def new_price (P : ℝ) : ℝ := P * 1.5
def new_quantity (Q : ℝ) : ℝ := Q * 0.8
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q)

theorem revenue_increase_20_percent (P Q : ℝ) : 
  (new_revenue P Q) = 1.2 * (original_revenue P Q) := by
  sorry

end revenue_increase_20_percent_l929_92944


namespace determine_vector_p_l929_92996

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def vector_operation (m p : Vector2D) : Vector2D :=
  Vector2D.mk (m.x * p.x + m.y * p.y) (m.x * p.y + m.y * p.x)

theorem determine_vector_p (p : Vector2D) : 
  (∀ (m : Vector2D), vector_operation m p = m) → p = Vector2D.mk 1 0 :=
by
  sorry

end determine_vector_p_l929_92996


namespace cost_of_large_tubs_l929_92925

theorem cost_of_large_tubs (L : ℝ) (h1 : 3 * L + 6 * 5 = 48) : L = 6 :=
by {
  sorry
}

end cost_of_large_tubs_l929_92925


namespace king_arthur_round_table_seats_l929_92955

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l929_92955


namespace sqrt_quartic_equiv_l929_92910

-- Define x as a positive real number
variable (x : ℝ)
variable (hx : 0 < x)

-- Statement of the problem to prove
theorem sqrt_quartic_equiv (x : ℝ) (hx : 0 < x) : (x^2 * x^(1/2))^(1/4) = x^(5/8) :=
sorry

end sqrt_quartic_equiv_l929_92910


namespace polynomial_satisfies_condition_l929_92930

-- Define P as a real polynomial
def P (a : ℝ) (X : ℝ) : ℝ := a * X

-- Define a statement that needs to be proven
theorem polynomial_satisfies_condition (P : ℝ → ℝ) :
  (∀ X : ℝ, P (2 * X) = 2 * P X) ↔ ∃ a : ℝ, ∀ X : ℝ, P X = a * X :=
by
  sorry

end polynomial_satisfies_condition_l929_92930


namespace coeff_x2_product_l929_92982

open Polynomial

noncomputable def poly1 : Polynomial ℤ := -5 * X^3 - 5 * X^2 - 7 * X + 1
noncomputable def poly2 : Polynomial ℤ := -X^2 - 6 * X + 1

theorem coeff_x2_product : (poly1 * poly2).coeff 2 = 36 := by
  sorry

end coeff_x2_product_l929_92982


namespace no_valid_n_l929_92988

theorem no_valid_n (n : ℕ) : (100 ≤ n / 4 ∧ n / 4 ≤ 999) → (100 ≤ 4 * n ∧ 4 * n ≤ 999) → false :=
by
  intro h1 h2
  sorry

end no_valid_n_l929_92988


namespace intersection_when_m_eq_2_range_of_m_l929_92993

open Set

variables (m x : ℝ)

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}
def intersection (m : ℝ) : Set ℝ := A m ∩ B

-- First proof: When m = 2, the intersection of A and B is [1,2].
theorem intersection_when_m_eq_2 : intersection 2 = {x | 1 ≤ x ∧ x ≤ 2} :=
sorry

-- Second proof: The range of m such that A ⊆ A ∩ B
theorem range_of_m : {m | A m ⊆ B} = {m | -2 ≤ m ∧ m ≤ 1 / 2} :=
sorry

end intersection_when_m_eq_2_range_of_m_l929_92993


namespace comparison_of_a_b_c_l929_92978

theorem comparison_of_a_b_c (a b c : ℝ) (h_a : a = Real.log 2) (h_b : b = 5^(-1/2 : ℝ)) (h_c : c = Real.sin (Real.pi / 6)) : 
  b < c ∧ c < a :=
by
  sorry

end comparison_of_a_b_c_l929_92978


namespace stratified_sampling_difference_l929_92908

theorem stratified_sampling_difference
  (male_athletes : ℕ := 56)
  (female_athletes : ℕ := 42)
  (sample_size : ℕ := 28)
  (H_total : male_athletes + female_athletes = 98)
  (H_sample_frac : sample_size = 28)
  : (56 * (sample_size / 98) - 42 * (sample_size / 98) = 4) :=
sorry

end stratified_sampling_difference_l929_92908


namespace max_shortest_side_decagon_inscribed_circle_l929_92971

noncomputable def shortest_side_decagon : ℝ :=
  2 * Real.sin (36 * Real.pi / 180 / 2)

theorem max_shortest_side_decagon_inscribed_circle :
  shortest_side_decagon = (Real.sqrt 5 - 1) / 2 :=
by {
  -- Proof details here
  sorry
}

end max_shortest_side_decagon_inscribed_circle_l929_92971


namespace arithmetic_sequence_common_diff_l929_92940

noncomputable def variance (s : List ℝ) : ℝ :=
  let mean := (s.sum) / (s.length : ℝ)
  (s.map (λ x => (x - mean) ^ 2)).sum / (s.length : ℝ)

theorem arithmetic_sequence_common_diff (a1 a2 a3 a4 a5 a6 a7 d : ℝ) 
(h_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d ∧ a5 = a1 + 4 * d ∧ a6 = a1 + 5 * d ∧ a7 = a1 + 6 * d)
(h_var : variance [a1, a2, a3, a4, a5, a6, a7] = 1) : 
d = 1 / 2 ∨ d = -1 / 2 := 
sorry

end arithmetic_sequence_common_diff_l929_92940


namespace first_number_in_set_l929_92961

theorem first_number_in_set (x : ℝ)
  (h : (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5) :
  x = 20 := by
  sorry

end first_number_in_set_l929_92961


namespace total_bags_l929_92906

-- Definitions based on the conditions
def bags_on_monday : ℕ := 4
def bags_next_day : ℕ := 8

-- Theorem statement
theorem total_bags : bags_on_monday + bags_next_day = 12 :=
by
  -- Proof will be added here
  sorry

end total_bags_l929_92906


namespace solve_system_l929_92946

theorem solve_system :
  ∃ x y : ℝ, (x^2 - 9 * y^2 = 0 ∧ 2 * x - 3 * y = 6) ∧ (x = 6 ∧ y = 2) ∨ (x = 2 ∧ y = -2 / 3) :=
by
  -- The proof will go here
  sorry

end solve_system_l929_92946


namespace roots_of_quadratic_eq_l929_92985

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l929_92985


namespace math_problem_l929_92901

theorem math_problem (a b c d x : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |x| = 2) :
  x^4 + c * d * x^2 - a - b = 20 :=
sorry

end math_problem_l929_92901


namespace distance_difference_l929_92920

-- Definitions related to the problem conditions
variables (v D_AB D_BC D_AC : ℝ)

-- Conditions
axiom h1 : D_AB = v * 7
axiom h2 : D_BC = v * 5
axiom h3 : D_AC = 6
axiom h4 : D_AC = D_AB + D_BC

-- Theorem for proof problem
theorem distance_difference : D_AB - D_BC = 1 :=
by sorry

end distance_difference_l929_92920


namespace minimum_value_of_expression_l929_92974

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) :
  ∃ P, (P = (x / y + y / z + z / x) * (y / x + z / y + x / z)) ∧ P = 25 := 
by sorry

end minimum_value_of_expression_l929_92974


namespace value_of_e_is_91_l929_92947

noncomputable def value_of_e (a b c d e : ℤ) (k : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧
  b = a + 2 * k ∧ c = a + 4 * k ∧ d = a + 6 * k ∧ e = a + 8 * k ∧
  a + c = 146 ∧ k > 0 ∧ 2 * k ≥ 4 ∧ k ≠ 2

theorem value_of_e_is_91 (a b c d e k : ℤ) (h : value_of_e a b c d e k) : e = 91 :=
  sorry

end value_of_e_is_91_l929_92947


namespace inequality_solution_set_l929_92969

theorem inequality_solution_set (a c x : ℝ) 
  (h1 : -1/3 < x ∧ x < 1/2 → 0 < a * x^2 + 2 * x + c) :
  -2 < x ∧ x < 3 ↔ -c * x^2 + 2 * x - a > 0 :=
by sorry

end inequality_solution_set_l929_92969


namespace ribbons_at_start_l929_92932

theorem ribbons_at_start (morning_ribbons : ℕ) (afternoon_ribbons : ℕ) (left_ribbons : ℕ)
  (h_morning : morning_ribbons = 14) (h_afternoon : afternoon_ribbons = 16) (h_left : left_ribbons = 8) :
  morning_ribbons + afternoon_ribbons + left_ribbons = 38 :=
by
  sorry

end ribbons_at_start_l929_92932


namespace m_range_l929_92931

variable (a1 b1 : ℝ)

def arithmetic_sequence (n : ℕ) : ℝ := a1 + 2 * (n - 1)
def geometric_sequence (n : ℕ) : ℝ := b1 * 2^(n - 1)

def a2_condition : Prop := arithmetic_sequence a1 2 + geometric_sequence b1 2 < -2
def a1_b1_condition : Prop := a1 + b1 > 0

theorem m_range : a1_b1_condition a1 b1 ∧ a2_condition a1 b1 → 
  let a4 := arithmetic_sequence a1 4 
  let b3 := geometric_sequence b1 3 
  let m := a4 + b3 
  m < 0 := 
by
  sorry

end m_range_l929_92931


namespace number_of_customers_l929_92998

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end number_of_customers_l929_92998


namespace fred_now_has_l929_92986

-- Definitions based on conditions
def original_cards : ℕ := 40
def purchased_cards : ℕ := 22

-- Theorem to prove the number of cards Fred has now
theorem fred_now_has (original_cards : ℕ) (purchased_cards : ℕ) : original_cards - purchased_cards = 18 :=
by
  sorry

end fred_now_has_l929_92986


namespace time_to_run_home_l929_92942

-- Define the conditions
def blocks_run_per_time : ℚ := 2 -- Justin runs 2 blocks
def time_per_blocks : ℚ := 1.5 -- in 1.5 minutes
def blocks_to_home : ℚ := 8 -- Justin is 8 blocks from home

-- Define the theorem to prove the time taken for Justin to run home
theorem time_to_run_home : (blocks_to_home / blocks_run_per_time) * time_per_blocks = 6 :=
by
  sorry

end time_to_run_home_l929_92942


namespace postage_problem_l929_92956

noncomputable def sum_all_positive_integers (n1 n2 : ℕ) : ℕ :=
  n1 + n2

theorem postage_problem : sum_all_positive_integers 21 22 = 43 :=
by
  have h1 : ∀ x y z : ℕ, 7 * x + 21 * y + 23 * z ≠ 120 := sorry
  have h2 : ∀ x y z : ℕ, 7 * x + 22 * y + 24 * z ≠ 120 := sorry
  exact rfl

end postage_problem_l929_92956


namespace minimal_area_circle_equation_circle_equation_center_on_line_l929_92957

-- Question (1): Prove the equation of the circle with minimal area
theorem minimal_area_circle_equation :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  C = (0, -4) ∧ r = Real.sqrt 5 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → P.1 ^ 2 + (P.2 + 4) ^ 2 = 5) :=
sorry

-- Question (2): Prove the equation of a circle with the center on a specific line
theorem circle_equation_center_on_line :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  (C.1 - 2 * C.2 - 3 = 0) ∧
  C = (-1, -2) ∧ r = Real.sqrt 10 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → (P.1 + 1) ^ 2 + (P.2 + 2) ^ 2 = 10) :=
sorry

end minimal_area_circle_equation_circle_equation_center_on_line_l929_92957


namespace transformations_map_onto_self_l929_92919

/-- Define the transformations -/
def T1 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a 90 degree rotation around the center of a square
  sorry

def T2 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a translation parallel to line ℓ
  sorry

def T3 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across line ℓ
  sorry

def T4 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across a line perpendicular to line ℓ
  sorry

/-- Define the pattern -/
def pattern (p : ℝ × ℝ) : Type :=
  -- Representation of alternating right triangles and squares along line ℓ
  sorry

/-- The main theorem:
    Prove that there are exactly 3 transformations (T1, T2, T3) that will map the pattern onto itself. -/
theorem transformations_map_onto_self : (∃ pattern : ℝ × ℝ → Type,
  (T1 pattern = pattern) ∧
  (T2 pattern = pattern) ∧
  (T3 pattern = pattern) ∧
  ¬ (T4 pattern = pattern)) → (3 = 3) :=
by
  sorry

end transformations_map_onto_self_l929_92919


namespace cloves_used_for_roast_chicken_l929_92966

section
variable (total_cloves : ℕ)
variable (remaining_cloves : ℕ)

theorem cloves_used_for_roast_chicken (h1 : total_cloves = 93) (h2 : remaining_cloves = 7) : total_cloves - remaining_cloves = 86 := 
by 
  have h : total_cloves - remaining_cloves = 93 - 7 := by rw [h1, h2]
  exact h
-- sorry
end

end cloves_used_for_roast_chicken_l929_92966


namespace point_reflection_example_l929_92963

def point := ℝ × ℝ

def reflect_x_axis (p : point) : point := (p.1, -p.2)

theorem point_reflection_example : reflect_x_axis (1, -2) = (1, 2) := sorry

end point_reflection_example_l929_92963


namespace four_digit_divisors_l929_92912

theorem four_digit_divisors :
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * d + 100 * a + 10 * b + c) →
  ∃ (e f : ℕ), e = a ∧ f = b ∧ (e ≠ 0 ∧ f ≠ 0) ∧ (1000 * e + 100 * e + 10 * f + f = 1000 * a + 100 * b + 10 * a + b) ∧
  (1000 * e + 100 * e + 10 * f + f ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * d + 100 * a + 10 * b + c) := 
by
  sorry

end four_digit_divisors_l929_92912


namespace jessica_total_spent_l929_92933

noncomputable def catToyCost : ℝ := 10.22
noncomputable def cageCost : ℝ := 11.73
noncomputable def totalCost : ℝ := 21.95

theorem jessica_total_spent :
  catToyCost + cageCost = totalCost :=
sorry

end jessica_total_spent_l929_92933


namespace pair_with_15_l929_92958

theorem pair_with_15 (s : List ℕ) (h : s = [49, 29, 9, 40, 22, 15, 53, 33, 13, 47]) :
  ∃ (t : List (ℕ × ℕ)), (∀ (x y : ℕ), (x, y) ∈ t → x + y = 62) ∧ (15, 47) ∈ t := by
  sorry

end pair_with_15_l929_92958


namespace added_number_is_four_l929_92965

theorem added_number_is_four :
  ∃ x y, 2 * x < 3 * x ∧ (3 * x - 2 * x = 8) ∧ 
         ((2 * x + y) * 7 = 5 * (3 * x + y)) ∧ y = 4 :=
  sorry

end added_number_is_four_l929_92965


namespace BD_distance_16_l929_92941

noncomputable def distanceBD (DA AB : ℝ) (angleBDA : ℝ) : ℝ :=
  (DA^2 + AB^2 - 2 * DA * AB * Real.cos angleBDA).sqrt

theorem BD_distance_16 :
  distanceBD 10 14 (60 * Real.pi / 180) = 16 := by
  sorry

end BD_distance_16_l929_92941


namespace cosine_of_acute_angle_l929_92980

theorem cosine_of_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 4 / 5) : Real.cos α = 3 / 5 :=
by
  sorry

end cosine_of_acute_angle_l929_92980


namespace sum_of_three_numbers_l929_92924

variable (x y z : ℝ)

theorem sum_of_three_numbers :
  y = 5 → 
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 :=
by
  intros hy h1 h2
  rw [hy] at h1 h2
  sorry

end sum_of_three_numbers_l929_92924


namespace brothers_work_rate_l929_92981

theorem brothers_work_rate (A B C : ℝ) :
  (1 / A + 1 / B = 1 / 8) ∧ (1 / A + 1 / C = 1 / 9) ∧ (1 / B + 1 / C = 1 / 10) →
  A = 160 / 19 ∧ B = 160 / 9 ∧ C = 32 / 3 :=
by
  sorry

end brothers_work_rate_l929_92981


namespace sum_of_f_values_l929_92950

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f_values :
  (∀ x : ℝ, f x + f (-x) = 0) →
  (∀ x : ℝ, f x = f (x + 2)) →
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = 2^x - 1) →
  f (1/2) + f 1 + f (3/2) + f 2 + f (5/2) = Real.sqrt 2 - 1 :=
by
  intros h1 h2 h3
  sorry

end sum_of_f_values_l929_92950


namespace initial_water_amount_l929_92973

theorem initial_water_amount (W : ℝ) (h1 : 0.006 * 50 = 0.03 * W) : W = 10 :=
by
  -- Proof steps would go here
  sorry

end initial_water_amount_l929_92973


namespace square_area_l929_92991

noncomputable def line_lies_on_square_side (a b : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A = (a, a + 4) ∧ B = (b, b + 4)

noncomputable def points_on_parabola (x y : ℝ) : Prop :=
  ∃ (C D : ℝ × ℝ), C = (y^2, y) ∧ D = (x^2, x)

theorem square_area (a b : ℝ) (x y : ℝ)
  (h1 : line_lies_on_square_side a b)
  (h2 : points_on_parabola x y) :
  ∃ (s : ℝ), s^2 = (boxed_solution) :=
sorry

end square_area_l929_92991


namespace plywood_width_is_5_l929_92923

theorem plywood_width_is_5 (length width perimeter : ℕ) (h1 : length = 6) (h2 : perimeter = 2 * (length + width)) (h3 : perimeter = 22) : width = 5 :=
by {
  -- proof steps would go here, but are omitted per instructions
  sorry
}

end plywood_width_is_5_l929_92923


namespace perp_lines_solution_l929_92959

theorem perp_lines_solution (a : ℝ) :
  ((a+2) * (a-1) + (1-a) * (2*a + 3) = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end perp_lines_solution_l929_92959


namespace fraction_habitable_surface_l929_92915

noncomputable def fraction_land_not_covered_by_water : ℚ := 1 / 3
noncomputable def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_land_not_covered_by_water * fraction_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_habitable_surface_l929_92915


namespace algebraic_expression_value_l929_92902

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 :=
by
  sorry

end algebraic_expression_value_l929_92902


namespace domain_of_function_l929_92921

noncomputable def function_domain := {x : ℝ | 1 + 1 / x > 0 ∧ 1 - x^2 ≥ 0}

theorem domain_of_function : function_domain = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l929_92921


namespace roots_of_cubic_l929_92977

theorem roots_of_cubic (a b c d r s t : ℝ) 
  (h1 : r + s + t = -b / a)
  (h2 : r * s + r * t + s * t = c / a)
  (h3 : r * s * t = -d / a) :
  1 / (r ^ 2) + 1 / (s ^ 2) + 1 / (t ^ 2) = (c ^ 2 - 2 * b * d) / (d ^ 2) := 
sorry

end roots_of_cubic_l929_92977


namespace find_number_l929_92995

theorem find_number (N: ℕ): (N % 131 = 112) ∧ (N % 132 = 98) → 1000 ≤ N ∧ N ≤ 9999 ∧ N = 1946 :=
sorry

end find_number_l929_92995


namespace balls_distribution_l929_92934

def balls_into_boxes : Nat := 6
def boxes : Nat := 3
def at_least_one_in_first (n m : Nat) : ℕ := sorry -- Use a function with appropriate constraints to ensure at least 1 ball is in the first box

theorem balls_distribution (n m : Nat) (h: n = 6) (h2: m = 3) :
  at_least_one_in_first n m = 665 :=
by
  sorry

end balls_distribution_l929_92934


namespace pond_ratios_l929_92951

theorem pond_ratios (T A : ℕ) (h1 : T = 48) (h2 : A = 32) : A / (T - A) = 2 :=
by
  sorry

end pond_ratios_l929_92951


namespace ben_initial_marbles_l929_92913

theorem ben_initial_marbles (B : ℕ) (John_initial_marbles : ℕ) (H1 : John_initial_marbles = 17) (H2 : John_initial_marbles + B / 2 = B / 2 + B / 2 + 17) : B = 34 := by
  sorry

end ben_initial_marbles_l929_92913


namespace arrange_books_l929_92900

open Nat

theorem arrange_books :
    let german_books := 3
    let spanish_books := 4
    let french_books := 3
    let total_books := german_books + spanish_books + french_books
    (total_books == 10) →
    let units := 2
    let items_to_arrange := units + german_books
    factorial items_to_arrange * factorial spanish_books * factorial french_books = 17280 :=
by 
    intros
    sorry

end arrange_books_l929_92900


namespace gamma_lt_delta_l929_92997

open Real

variables (α β γ δ : ℝ)

-- Hypotheses as given in the problem
axiom h1 : 0 < α 
axiom h2 : α < β
axiom h3 : β < π / 2
axiom hg1 : 0 < γ
axiom hg2 : γ < π / 2
axiom htan_gamma_eq : tan γ = (tan α + tan β) / 2
axiom hd1 : 0 < δ
axiom hd2 : δ < π / 2
axiom hcos_delta_eq : (1 / cos δ) = (1 / 2) * (1 / cos α + 1 / cos β)

-- Goal to prove
theorem gamma_lt_delta : γ < δ := 
by 
sorry

end gamma_lt_delta_l929_92997


namespace toby_steps_needed_l929_92928

noncomputable def total_steps_needed : ℕ := 10000 * 9

noncomputable def first_sunday_steps : ℕ := 10200
noncomputable def first_monday_steps : ℕ := 10400
noncomputable def tuesday_steps : ℕ := 9400
noncomputable def wednesday_steps : ℕ := 9100
noncomputable def thursday_steps : ℕ := 8300
noncomputable def friday_steps : ℕ := 9200
noncomputable def saturday_steps : ℕ := 8900
noncomputable def second_sunday_steps : ℕ := 9500

noncomputable def total_steps_walked := 
  first_sunday_steps + 
  first_monday_steps + 
  tuesday_steps + 
  wednesday_steps + 
  thursday_steps + 
  friday_steps + 
  saturday_steps + 
  second_sunday_steps

noncomputable def remaining_steps_needed := total_steps_needed - total_steps_walked

noncomputable def days_left : ℕ := 3

noncomputable def average_steps_needed := remaining_steps_needed / days_left

theorem toby_steps_needed : average_steps_needed = 5000 := by
  sorry

end toby_steps_needed_l929_92928


namespace smallest_four_digit_multiple_of_18_l929_92952

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l929_92952


namespace first_range_is_30_l929_92954

theorem first_range_is_30 
  (R2 R3 : ℕ)
  (h1 : R2 = 26)
  (h2 : R3 = 32)
  (h3 : min 26 (min 30 32) = 30) : 
  ∃ R1 : ℕ, R1 = 30 :=
  sorry

end first_range_is_30_l929_92954


namespace max_card_count_sum_l929_92938

theorem max_card_count_sum (W B R : ℕ) (total_cards : ℕ) 
  (white_cards black_cards red_cards : ℕ) : 
  total_cards = 300 ∧ white_cards = 100 ∧ black_cards = 100 ∧ red_cards = 100 ∧
  (∀ w, w < white_cards → ∃ b, b < black_cards) ∧ 
  (∀ b, b < black_cards → ∃ r, r < red_cards) ∧ 
  (∀ r, r < red_cards → ∃ w, w < white_cards) →
  ∃ max_sum, max_sum = 20000 :=
by
  sorry

end max_card_count_sum_l929_92938


namespace reciprocal_of_2022_l929_92975

theorem reciprocal_of_2022 : 1 / 2022 = (1 : ℝ) / 2022 :=
sorry

end reciprocal_of_2022_l929_92975


namespace john_buys_spools_l929_92916

theorem john_buys_spools (spool_length necklace_length : ℕ) 
  (necklaces : ℕ) 
  (total_length := necklaces * necklace_length) 
  (spools := total_length / spool_length) :
  spool_length = 20 → 
  necklace_length = 4 → 
  necklaces = 15 → 
  spools = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end john_buys_spools_l929_92916


namespace office_distance_eq_10_l929_92964

noncomputable def distance_to_office (D T : ℝ) : Prop :=
  D = 10 * (T + 10 / 60) ∧ D = 15 * (T - 10 / 60)

theorem office_distance_eq_10 (D T : ℝ) (h : distance_to_office D T) : D = 10 :=
by
  sorry

end office_distance_eq_10_l929_92964


namespace parallel_planes_imply_l929_92918

variable {Point Line Plane : Type}

-- Definitions of parallelism and perpendicularity between lines and planes
variables {parallel_perpendicular : Line → Plane → Prop}
variables {parallel_lines : Line → Line → Prop}
variables {parallel_planes : Plane → Plane → Prop}

-- Given conditions
variable {m n : Line}
variable {α β : Plane}

-- Conditions
axiom m_parallel_n : parallel_lines m n
axiom m_perpendicular_α : parallel_perpendicular m α
axiom n_perpendicular_β : parallel_perpendicular n β

-- The statement to be proven
theorem parallel_planes_imply (m_parallel_n : parallel_lines m n)
  (m_perpendicular_α : parallel_perpendicular m α)
  (n_perpendicular_β : parallel_perpendicular n β) :
  parallel_planes α β :=
sorry

end parallel_planes_imply_l929_92918


namespace triangle_inequality_1_triangle_inequality_2_l929_92926

variable (a b c : ℝ)

theorem triangle_inequality_1 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b * c + 28 / 27 ≥ a * b + b * c + c * a :=
by
  sorry

theorem triangle_inequality_2 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b + b * c + c * a ≥ a * b * c + 1 :=
by
  sorry

end triangle_inequality_1_triangle_inequality_2_l929_92926


namespace min_sum_of_dimensions_l929_92968

theorem min_sum_of_dimensions 
  (a b c : ℕ) 
  (h_pos : a > 0) 
  (h_pos_2 : b > 0) 
  (h_pos_3 : c > 0) 
  (h_even : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) 
  (h_vol : a * b * c = 1806) 
  : a + b + c = 56 :=
sorry

end min_sum_of_dimensions_l929_92968


namespace sin_14pi_div_3_eq_sqrt3_div_2_l929_92949

theorem sin_14pi_div_3_eq_sqrt3_div_2 : Real.sin (14 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_14pi_div_3_eq_sqrt3_div_2_l929_92949


namespace greatest_integer_radius_l929_92953

theorem greatest_integer_radius (r : ℕ) :
  (π * (r: ℝ)^2 < 30 * π) ∧ (2 * π * (r: ℝ) > 10 * π) → r = 5 :=
by
  sorry

end greatest_integer_radius_l929_92953


namespace product_of_solutions_eq_zero_l929_92943

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (3 * x + 5) / (6 * x + 5) = (5 * x + 4) / (9 * x + 4) → (x = 0 ∨ x = 8 / 3)) →
  0 * (8 / 3) = 0 :=
by
  intro h
  sorry

end product_of_solutions_eq_zero_l929_92943


namespace minimize_quadratic_l929_92917

theorem minimize_quadratic : ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12*x + 28 ≤ y^2 - 12*y + 28) :=
by
  use 6
  sorry

end minimize_quadratic_l929_92917


namespace Z_4_3_eq_neg11_l929_92976

def Z (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2

theorem Z_4_3_eq_neg11 : Z 4 3 = -11 := 
by
  sorry

end Z_4_3_eq_neg11_l929_92976


namespace calc_mixed_number_expr_l929_92929

theorem calc_mixed_number_expr :
  53 * (3 + 1 / 4 - (3 + 3 / 4)) / (1 + 2 / 3 + (2 + 2 / 5)) = -6 - 57 / 122 := 
by
  sorry

end calc_mixed_number_expr_l929_92929
