import Mathlib

namespace rectangle_minimal_area_l2130_213008

theorem rectangle_minimal_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (l + w) = 120) : l * w = 675 :=
by
  -- Proof will go here
  sorry

end rectangle_minimal_area_l2130_213008


namespace smallest_bottom_right_value_l2130_213037

theorem smallest_bottom_right_value :
  ∃ (grid : ℕ × ℕ × ℕ → ℕ), -- grid as a function from row/column pairs to natural numbers
    (∀ i j, 1 ≤ i ∧ i ≤ 3 → 1 ≤ j ∧ j ≤ 3 → grid (i, j) ≠ 0) ∧ -- all grid values are non-zero
    (grid (1, 1) ≠ grid (1, 2) ∧ grid (1, 1) ≠ grid (1, 3) ∧ grid (1, 2) ≠ grid (1, 3) ∧
     grid (2, 1) ≠ grid (2, 2) ∧ grid (2, 1) ≠ grid (2, 3) ∧ grid (2, 2) ≠ grid (2, 3) ∧
     grid (3, 1) ≠ grid (3, 2) ∧ grid (3, 1) ≠ grid (3, 3) ∧ grid (3, 2) ≠ grid (3, 3)) ∧ -- all grid values are distinct
    (grid (1, 1) + grid (1, 2) = grid (1, 3)) ∧ 
    (grid (2, 1) + grid (2, 2) = grid (2, 3)) ∧ 
    (grid (3, 1) + grid (3, 2) = grid (3, 3)) ∧ -- row sum conditions
    (grid (1, 1) + grid (2, 1) = grid (3, 1)) ∧ 
    (grid (1, 2) + grid (2, 2) = grid (3, 2)) ∧ 
    (grid (1, 3) + grid (2, 3) = grid (3, 3)) ∧ -- column sum conditions
    (grid (3, 3) = 12) :=
by
  sorry

end smallest_bottom_right_value_l2130_213037


namespace correct_statement_is_B_l2130_213020

def coefficient_of_x : Int := 1
def is_monomial (t : String) : Bool := t = "1x^0"
def coefficient_of_neg_3x : Int := -3
def degree_of_5x2y : Int := 3

theorem correct_statement_is_B :
  (coefficient_of_x = 0) = false ∧ 
  (is_monomial "1x^0" = true) ∧ 
  (coefficient_of_neg_3x = 3) = false ∧ 
  (degree_of_5x2y = 2) = false ∧ 
  (B = "1 is a monomial") :=
by {
  sorry
}

end correct_statement_is_B_l2130_213020


namespace probability_at_least_one_bean_distribution_of_X_expectation_of_X_l2130_213094

noncomputable def total_ways := Nat.choose 6 3
noncomputable def ways_select_2_egg_1_bean := (Nat.choose 4 2) * (Nat.choose 2 1)
noncomputable def ways_select_1_egg_2_bean := (Nat.choose 4 1) * (Nat.choose 2 2)
noncomputable def at_least_one_bean_probability := (ways_select_2_egg_1_bean + ways_select_1_egg_2_bean) / total_ways

theorem probability_at_least_one_bean : at_least_one_bean_probability = 4 / 5 :=
by sorry

noncomputable def p_X_eq_0 := (Nat.choose 4 3) / total_ways
noncomputable def p_X_eq_1 := ways_select_2_egg_1_bean / total_ways
noncomputable def p_X_eq_2 := ways_select_1_egg_2_bean / total_ways

theorem distribution_of_X : p_X_eq_0 = 1 / 5 ∧ p_X_eq_1 = 3 / 5 ∧ p_X_eq_2 = 1 / 5 :=
by sorry

noncomputable def E_X := (0 * p_X_eq_0) + (1 * p_X_eq_1) + (2 * p_X_eq_2)

theorem expectation_of_X : E_X = 1 :=
by sorry

end probability_at_least_one_bean_distribution_of_X_expectation_of_X_l2130_213094


namespace evaluate_expression_l2130_213087

theorem evaluate_expression (a : ℤ) : ((a + 10) - a + 3) * ((a + 10) - a - 2) = 104 := by
  sorry

end evaluate_expression_l2130_213087


namespace Robert_diff_C_l2130_213023

/- Define the conditions as hypotheses -/
variables (C : ℕ) -- Assuming the number of photos Claire has taken as a natural number.

-- Lisa has taken 3 times as many photos as Claire.
def Lisa_photos := 3 * C

-- Robert has taken the same number of photos as Lisa.
def Robert_photos := Lisa_photos C -- which will be 3 * C

-- Proof of the difference.
theorem Robert_diff_C : (Robert_photos C) - C = 2 * C :=
by
  sorry

end Robert_diff_C_l2130_213023


namespace pencil_length_difference_l2130_213034

theorem pencil_length_difference (a b : ℝ) (h1 : a = 1) (h2 : b = 4/9) :
  a - b - b = 1/9 :=
by
  rw [h1, h2]
  sorry

end pencil_length_difference_l2130_213034


namespace max_books_borrowed_l2130_213002

theorem max_books_borrowed (total_students books_per_student : ℕ) (students_with_no_books: ℕ) (students_with_one_book students_with_two_books: ℕ) (rest_at_least_three_books students : ℕ) :
  total_students = 20 →
  books_per_student = 2 →
  students_with_no_books = 2 →
  students_with_one_book = 8 →
  students_with_two_books = 3 →
  rest_at_least_three_books = total_students - (students_with_no_books + students_with_one_book + students_with_two_books) →
  (students_with_no_books * 0 + students_with_one_book * 1 + students_with_two_books * 2 + students * books_per_student = total_students * books_per_student) →
  (students * 3 + some_student_max = 26) →
  some_student_max ≥ 8 :=
by
  introv h1 h2 h3 h4 h5 h6 h7
  sorry

end max_books_borrowed_l2130_213002


namespace not_entire_field_weedy_l2130_213012

-- Define the conditions
def field_divided_into_100_plots : Prop :=
  ∃ (a b : ℕ), a * b = 100

def initial_weedy_plots : Prop :=
  ∃ (weedy_plots : Finset (ℕ × ℕ)), weedy_plots.card = 9

def plot_becomes_weedy (weedy_plots : Finset (ℕ × ℕ)) (p : ℕ × ℕ) : Prop :=
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots)

-- Theorem statement
theorem not_entire_field_weedy :
  field_divided_into_100_plots →
  initial_weedy_plots →
  (∀ weedy_plots : Finset (ℕ × ℕ), (∀ p : ℕ × ℕ, plot_becomes_weedy weedy_plots p → weedy_plots ∪ {p} = weedy_plots) → weedy_plots.card < 100) :=
  sorry

end not_entire_field_weedy_l2130_213012


namespace sin_cos_identity_l2130_213055

theorem sin_cos_identity :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) -
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end sin_cos_identity_l2130_213055


namespace simplify_fraction_expression_l2130_213059

theorem simplify_fraction_expression :
  5 * (12 / 7) * (49 / (-60)) = -7 := 
sorry

end simplify_fraction_expression_l2130_213059


namespace problem_sol_l2130_213074

-- Defining the operations as given
def operation_hash (a b c : ℤ) : ℤ := 4 * a ^ 3 + 4 * b ^ 3 + 8 * a ^ 2 * b + c
def operation_star (a b d : ℤ) : ℤ := 2 * a ^ 2 - 3 * b ^ 2 + d ^ 3

-- Main theorem statement
theorem problem_sol (a b x c d : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (hc : c > 0) (hd : d > 0) 
  (h3 : operation_hash a x c = 250)
  (h4 : operation_star a b d + x = 50) :
  False := sorry

end problem_sol_l2130_213074


namespace linear_coefficient_l2130_213088

theorem linear_coefficient (m x : ℝ) (h1 : (m - 3) * x ^ (m^2 - 2 * m - 1) - m * x + 6 = 0) (h2 : (m^2 - 2 * m - 1 = 2)) (h3 : m ≠ 3) : 
  ∃ a b c : ℝ, a * x ^ 2 + b * x + c = 0 ∧ b = 1 :=
by
  sorry

end linear_coefficient_l2130_213088


namespace jacob_age_l2130_213079

/- Conditions:
1. Rehana's current age is 25.
2. In five years, Rehana's age is three times Phoebe's age.
3. Jacob's current age is 3/5 of Phoebe's current age.

Prove that Jacob's current age is 3.
-/

theorem jacob_age (R P J : ℕ) (h1 : R = 25) (h2 : R + 5 = 3 * (P + 5)) (h3 : J = 3 / 5 * P) : J = 3 :=
by
  sorry

end jacob_age_l2130_213079


namespace sin_160_eq_sin_20_l2130_213096

theorem sin_160_eq_sin_20 : Real.sin (160 * Real.pi / 180) = Real.sin (20 * Real.pi / 180) :=
by
  sorry

end sin_160_eq_sin_20_l2130_213096


namespace induction_step_n_eq_1_l2130_213058

theorem induction_step_n_eq_1 : (1 + 2 + 3 = (1+1)*(2*1+1)) :=
by
  -- Proof would go here
  sorry

end induction_step_n_eq_1_l2130_213058


namespace a_perpendicular_to_a_minus_b_l2130_213066

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2

def a : vector := (-2, 1)
def b : vector := (-1, 3)

def a_minus_b : vector := (a.1 - b.1, a.2 - b.2) 

theorem a_perpendicular_to_a_minus_b : dot_product a a_minus_b = 0 := by
  sorry

end a_perpendicular_to_a_minus_b_l2130_213066


namespace games_went_this_year_l2130_213041

theorem games_went_this_year (t l : ℕ) (h1 : t = 13) (h2 : l = 9) : (t - l = 4) :=
by
  sorry

end games_went_this_year_l2130_213041


namespace solve_quadratic_l2130_213067

theorem solve_quadratic (x : ℝ) : x^2 - 4 * x + 3 = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end solve_quadratic_l2130_213067


namespace distance_missouri_to_new_york_by_car_l2130_213027

variable (d_flight d_car : ℚ)

theorem distance_missouri_to_new_york_by_car :
  d_car = 1.4 * d_flight → 
  d_car = 1400 → 
  (d_car / 2 = 700) :=
by
  intros h1 h2
  sorry

end distance_missouri_to_new_york_by_car_l2130_213027


namespace flowers_bees_butterflies_comparison_l2130_213050

def num_flowers : ℕ := 12
def num_bees : ℕ := 7
def num_butterflies : ℕ := 4
def difference_flowers_bees : ℕ := num_flowers - num_bees

theorem flowers_bees_butterflies_comparison :
  difference_flowers_bees - num_butterflies = 1 :=
by
  -- The proof will go here
  sorry

end flowers_bees_butterflies_comparison_l2130_213050


namespace john_horizontal_distance_l2130_213049

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end john_horizontal_distance_l2130_213049


namespace max_A_value_l2130_213014

-- Variables
variables {x1 x2 x3 y1 y2 y3 z1 z2 z3 : ℝ}

-- Assumptions
axiom pos_x1 : 0 < x1
axiom pos_x2 : 0 < x2
axiom pos_x3 : 0 < x3
axiom pos_y1 : 0 < y1
axiom pos_y2 : 0 < y2
axiom pos_y3 : 0 < y3
axiom pos_z1 : 0 < z1
axiom pos_z2 : 0 < z2
axiom pos_z3 : 0 < z3

-- Statement
theorem max_A_value :
  ∃ A : ℝ, 
    (∀ x1 x2 x3 y1 y2 y3 z1 z2 z3, 
    (0 < x1) → (0 < x2) → (0 < x3) →
    (0 < y1) → (0 < y2) → (0 < y3) →
    (0 < z1) → (0 < z2) → (0 < z3) →
    (x1^3 + x2^3 + x3^3 + 1) * (y1^3 + y2^3 + y3^3 + 1) * (z1^3 + z2^3 + z3^3 + 1) ≥
    A * (x1 + y1 + z1) * (x2 + y2 + z2) * (x3 + y3 + z3)) ∧ 
    A = 9/2 := 
by 
  exists 9/2 
  sorry

end max_A_value_l2130_213014


namespace ratio_time_A_to_B_l2130_213026

-- Definition of total examination time in minutes
def total_time : ℕ := 180

-- Definition of time spent on type A problems
def time_A : ℕ := 40

-- Definition of time spent on type B problems as total_time - time_A
def time_B : ℕ := total_time - time_A

-- Statement that we need to prove
theorem ratio_time_A_to_B : time_A * 7 = time_B * 2 :=
by
  -- Implementation of the proof will go here
  sorry

end ratio_time_A_to_B_l2130_213026


namespace initial_amount_proof_l2130_213064

noncomputable def initial_amount (A B : ℝ) : ℝ :=
  A + B

theorem initial_amount_proof :
  ∃ (A B : ℝ), B = 4000.0000000000005 ∧ 
               (A * 0.15 * 2 = B * 0.18 * 2 + 360) ∧ 
               initial_amount A B = 10000.000000000002 :=
by
  sorry

end initial_amount_proof_l2130_213064


namespace dave_apps_files_difference_l2130_213091

theorem dave_apps_files_difference :
  let initial_apps := 15
  let initial_files := 24
  let final_apps := 21
  let final_files := 4
  final_apps - final_files = 17 :=
by
  intros
  sorry

end dave_apps_files_difference_l2130_213091


namespace charles_ate_no_bananas_l2130_213090

theorem charles_ate_no_bananas (W C B : ℝ) (h1 : W = 48) (h2 : C = 35) (h3 : W + C = 83) : B = 0 :=
by
  -- Proof goes here
  sorry

end charles_ate_no_bananas_l2130_213090


namespace remainder_18_l2130_213072

theorem remainder_18 (x : ℤ) (k : ℤ) (h : x = 62 * k + 7) :
  (x + 11) % 31 = 18 :=
by
  sorry

end remainder_18_l2130_213072


namespace region_area_l2130_213029

noncomputable def area_of_region := 4 * Real.pi

theorem region_area :
  (∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0) →
  Real.pi * 4 = area_of_region :=
by
  sorry

end region_area_l2130_213029


namespace lower_limit_of_range_with_multiples_l2130_213048

theorem lower_limit_of_range_with_multiples (n : ℕ) (h : 2000 - n ≥ 198 * 10 ∧ n % 10 = 0 ∧ n + 1980 ≤ 2000) :
  n = 30 :=
by
  sorry

end lower_limit_of_range_with_multiples_l2130_213048


namespace part_a_part_b_l2130_213092

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l2130_213092


namespace natural_numbers_between_sqrt_100_and_101_l2130_213025

theorem natural_numbers_between_sqrt_100_and_101 :
  ∃ (n : ℕ), n = 200 ∧ (∀ k : ℕ, 100 < Real.sqrt k ∧ Real.sqrt k < 101 -> 10000 < k ∧ k < 10201) := 
by
  sorry

end natural_numbers_between_sqrt_100_and_101_l2130_213025


namespace minimum_y_l2130_213054

theorem minimum_y (x : ℝ) (h : x > 1) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y = 3) :=
by
  sorry

end minimum_y_l2130_213054


namespace unique_triple_solution_l2130_213075

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (y > 1) ∧ Prime y ∧
                  (¬(3 ∣ z ∧ y ∣ z)) ∧
                  (x^3 - y^3 = z^2) ∧
                  (x = 8 ∧ y = 7 ∧ z = 13) :=
by
  sorry

end unique_triple_solution_l2130_213075


namespace turban_as_part_of_salary_l2130_213060

-- Definitions of the given conditions
def annual_salary (T : ℕ) : ℕ := 90 + 70 * T
def nine_month_salary (T : ℕ) : ℕ := 3 * (90 + 70 * T) / 4
def leaving_amount : ℕ := 50 + 70

-- Proof problem statement in Lean 4
theorem turban_as_part_of_salary (T : ℕ) (h : nine_month_salary T = leaving_amount) : T = 1 := 
sorry

end turban_as_part_of_salary_l2130_213060


namespace polynomial_roots_correct_l2130_213085

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end polynomial_roots_correct_l2130_213085


namespace range_of_a_l2130_213080

noncomputable def f (a x : ℝ) : ℝ := min (Real.exp x - 2) (Real.exp (2 * x) - a * Real.exp x + a + 24)

def has_three_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0

theorem range_of_a (a : ℝ) :
  has_three_zeros (f a) ↔ 12 < a ∧ a < 28 :=
sorry

end range_of_a_l2130_213080


namespace geom_S4_eq_2S2_iff_abs_q_eq_1_l2130_213028

variable {α : Type*} [LinearOrderedField α]

-- defining the sum of first n terms of a geometric sequence
def geom_series_sum (a q : α) (n : ℕ) :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

noncomputable def S (a q : α) (n : ℕ) := geom_series_sum a q n

theorem geom_S4_eq_2S2_iff_abs_q_eq_1 
  (a q : α) : 
  S a q 4 = 2 * S a q 2 ↔ |q| = 1 :=
sorry

end geom_S4_eq_2S2_iff_abs_q_eq_1_l2130_213028


namespace kwik_e_tax_revenue_l2130_213093

def price_federal : ℕ := 50
def price_state : ℕ := 30
def price_quarterly : ℕ := 80

def num_federal : ℕ := 60
def num_state : ℕ := 20
def num_quarterly : ℕ := 10

def revenue_federal := num_federal * price_federal
def revenue_state := num_state * price_state
def revenue_quarterly := num_quarterly * price_quarterly

def total_revenue := revenue_federal + revenue_state + revenue_quarterly

theorem kwik_e_tax_revenue : total_revenue = 4400 := by
  sorry

end kwik_e_tax_revenue_l2130_213093


namespace chris_wins_l2130_213095

noncomputable def chris_heads : ℚ := 1 / 4
noncomputable def drew_heads : ℚ := 1 / 3
noncomputable def both_tails : ℚ := (1 - chris_heads) * (1 - drew_heads)

/-- The probability that Chris wins comparing with relatively prime -/
theorem chris_wins (p q : ℕ) (hpq : Nat.Coprime p q) (hq0 : q ≠ 0) :
  (chris_heads * (1 + both_tails)) = (p : ℚ) / q ∧ (q - p = 1) :=
sorry

end chris_wins_l2130_213095


namespace rhombus_area_2sqrt2_l2130_213097

structure Rhombus (α : Type _) :=
  (side_length : ℝ)
  (angle : ℝ)

theorem rhombus_area_2sqrt2 (R : Rhombus ℝ) (h_side : R.side_length = 2) (h_angle : R.angle = 45) :
  ∃ A : ℝ, A = 2 * Real.sqrt 2 :=
by
  let A := 2 * Real.sqrt 2
  existsi A
  sorry

end rhombus_area_2sqrt2_l2130_213097


namespace solution_set_inequality_l2130_213042

theorem solution_set_inequality (x : ℝ) : (x ≠ 1) → 
  ((x - 3) * (x + 2) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3) :=
by
  intros h
  sorry

end solution_set_inequality_l2130_213042


namespace fraction_equivalent_to_decimal_l2130_213051

theorem fraction_equivalent_to_decimal : 
  (0.4 -- using appropriate representation for repeating decimal 0.4\overline{13}
      + 13 / 990) = 409 / 990 ∧ Nat.gcd 409 990 = 1 := 
sorry

end fraction_equivalent_to_decimal_l2130_213051


namespace contrapositive_lemma_l2130_213018

theorem contrapositive_lemma (a : ℝ) (h : a^2 ≤ 9) : a < 4 := 
sorry

end contrapositive_lemma_l2130_213018


namespace bakery_water_requirement_l2130_213036

theorem bakery_water_requirement (flour water : ℕ) (total_flour : ℕ) (h : flour = 300) (w : water = 75) (t : total_flour = 900) : 
  225 = (total_flour / flour) * water :=
by
  sorry

end bakery_water_requirement_l2130_213036


namespace probability_is_correct_l2130_213057

noncomputable def probability_total_more_than_7 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 15
  favorable_outcomes / total_outcomes

theorem probability_is_correct :
  probability_total_more_than_7 = 5 / 12 :=
by
  sorry

end probability_is_correct_l2130_213057


namespace equal_chord_segments_l2130_213062

theorem equal_chord_segments 
  (a x y : ℝ) 
  (AM CM : ℝ → ℝ → Prop) 
  (AB CD : ℝ → Prop)
  (intersect_chords_theorem : AM x (a - x) = CM y (a - y)) :
  x = y ∨ x = a - y :=
by
  sorry

end equal_chord_segments_l2130_213062


namespace area_of_region_l2130_213098

theorem area_of_region (x y : ℝ) : |4 * x - 24| + |3 * y + 10| ≤ 6 → ∃ A : ℝ, A = 12 :=
by
  sorry

end area_of_region_l2130_213098


namespace tan_A_area_triangle_ABC_l2130_213006
open Real

-- Define the given conditions
def conditions (A : ℝ) (AC AB : ℝ) : Prop :=
  (sin A + cos A = sqrt 2 / 2) ∧ (AC = 2) ∧ (AB = 3)

-- State the first proof problem for tan A
theorem tan_A (A : ℝ) (hcond : conditions A 2 3) : tan A = -(2 + sqrt 3) := 
by 
  -- sorry for the proof placeholder
  sorry

-- State the second proof problem for the area of triangle ABC
theorem area_triangle_ABC (A B C : ℝ) (C_eq : C = 90) 
  (hcond : conditions A 2 3)
  (hBC : BC = sqrt ((AC^2) + (AB^2) - 2 * AC * AB * cos B)) : 
  (1/2) * AC * AB * sin A = (3 / 4) * (sqrt 6 + sqrt 2) := 
by 
  -- sorry for the proof placeholder
  sorry

end tan_A_area_triangle_ABC_l2130_213006


namespace intersection_P_Q_l2130_213073

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l2130_213073


namespace coffee_shop_lattes_l2130_213038

theorem coffee_shop_lattes (T : ℕ) (L : ℕ) (hT : T = 6) (hL : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_lattes_l2130_213038


namespace parabola_passing_through_4_neg2_l2130_213084

theorem parabola_passing_through_4_neg2 :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ y = -2 ∧ x = 4 ∧ (y^2 = x)) ∨
  (∃ p : ℝ, x^2 = -2 * p * y ∧ y = -2 ∧ x = 4 ∧ (x^2 = -8 * y)) :=
by
  sorry

end parabola_passing_through_4_neg2_l2130_213084


namespace fraction_oj_is_5_over_13_l2130_213019

def capacity_first_pitcher : ℕ := 800
def capacity_second_pitcher : ℕ := 500
def fraction_oj_first_pitcher : ℚ := 1 / 4
def fraction_oj_second_pitcher : ℚ := 3 / 5

def amount_oj_first_pitcher : ℚ := capacity_first_pitcher * fraction_oj_first_pitcher
def amount_oj_second_pitcher : ℚ := capacity_second_pitcher * fraction_oj_second_pitcher

def total_amount_oj : ℚ := amount_oj_first_pitcher + amount_oj_second_pitcher
def total_capacity : ℚ := capacity_first_pitcher + capacity_second_pitcher

def fraction_oj_large_container : ℚ := total_amount_oj / total_capacity

theorem fraction_oj_is_5_over_13 : fraction_oj_large_container = (5 / 13) := by
  -- Proof would go here
  sorry

end fraction_oj_is_5_over_13_l2130_213019


namespace toy_robot_shipment_l2130_213082

-- Define the conditions provided in the problem
def thirty_percent_displayed (total: ℕ) : ℕ := (3 * total) / 10
def seventy_percent_stored (total: ℕ) : ℕ := (7 * total) / 10

-- The main statement to prove: if 70% of the toy robots equal 140, then the total number of toy robots is 200
theorem toy_robot_shipment (total : ℕ) (h : seventy_percent_stored total = 140) : total = 200 :=
by
  -- We will fill in the proof here
  sorry

end toy_robot_shipment_l2130_213082


namespace least_number_divisible_by_digits_and_5_l2130_213052

/-- Define a predicate to check if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10 % 10, n / 10 % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

/-- Define the main theorem stating the least four-digit number divisible by 5 and each of its digits is 1425 -/
theorem least_number_divisible_by_digits_and_5 
  (n : ℕ) (hn : 1000 ≤ n ∧ n < 10000)
  (hd : (∀ i j : ℕ, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)))
  (hdiv5 : n % 5 = 0)
  (hdiv_digits : divisible_by_digits n) 
  : n = 1425 :=
sorry

end least_number_divisible_by_digits_and_5_l2130_213052


namespace product_remainder_l2130_213047

theorem product_remainder (a b : ℕ) (m n : ℤ) (ha : a = 3 * m + 2) (hb : b = 3 * n + 2) : 
  (a * b) % 3 = 1 := 
by 
  sorry

end product_remainder_l2130_213047


namespace average_ratio_one_l2130_213053

theorem average_ratio_one (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / 50)
  let scores_with_averages := scores ++ [A, A]
  let A' := (scores_with_averages.sum / 52)
  A' = A :=
by
  sorry

end average_ratio_one_l2130_213053


namespace num_nat_numbers_divisible_by_7_between_100_and_250_l2130_213003

noncomputable def countNatNumbersDivisibleBy7InRange : ℕ :=
  let smallest := Nat.ceil (100 / 7) * 7
  let largest := Nat.floor (250 / 7) * 7
  (largest - smallest) / 7 + 1

theorem num_nat_numbers_divisible_by_7_between_100_and_250 :
  countNatNumbersDivisibleBy7InRange = 21 :=
by
  -- Placeholder for the proof steps
  sorry

end num_nat_numbers_divisible_by_7_between_100_and_250_l2130_213003


namespace negation_of_universal_prop_l2130_213039

variable (P : ∀ x : ℝ, Real.cos x ≤ 1)

theorem negation_of_universal_prop : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end negation_of_universal_prop_l2130_213039


namespace total_cost_of_panels_l2130_213040

theorem total_cost_of_panels
    (sidewall_width : ℝ)
    (sidewall_height : ℝ)
    (triangle_base : ℝ)
    (triangle_height : ℝ)
    (panel_width : ℝ)
    (panel_height : ℝ)
    (panel_cost : ℝ)
    (total_cost : ℝ)
    (h_sidewall : sidewall_width = 9)
    (h_sidewall_height : sidewall_height = 7)
    (h_triangle_base : triangle_base = 9)
    (h_triangle_height : triangle_height = 6)
    (h_panel_width : panel_width = 10)
    (h_panel_height : panel_height = 15)
    (h_panel_cost : panel_cost = 32)
    (h_total_cost : total_cost = 32) :
    total_cost = panel_cost :=
by
  sorry

end total_cost_of_panels_l2130_213040


namespace luke_clothing_distribution_l2130_213099

theorem luke_clothing_distribution (total_clothing: ℕ) (first_load: ℕ) (num_loads: ℕ) 
  (remaining_clothing : total_clothing - first_load = 30)
  (equal_load_per_small_load: (total_clothing - first_load) / num_loads = 6) : 
  total_clothing = 47 ∧ first_load = 17 ∧ num_loads = 5 :=
by
  have h1 : total_clothing - first_load = 30 := remaining_clothing
  have h2 : (total_clothing - first_load) / num_loads = 6 := equal_load_per_small_load
  sorry

end luke_clothing_distribution_l2130_213099


namespace find_line_equation_through_ellipse_midpoint_l2130_213004

theorem find_line_equation_through_ellipse_midpoint {A B : ℝ × ℝ} 
  (hA : (A.fst^2 / 2) + A.snd^2 = 1) 
  (hB : (B.fst^2 / 2) + B.snd^2 = 1) 
  (h_midpoint : (A.fst + B.fst) / 2 = 1 ∧ (A.snd + B.snd) / 2 = 1 / 2) : 
  ∃ k : ℝ, (k = -1) ∧ (∀ x y : ℝ, (y - 1/2 = k * (x - 1)) → 2*x + 2*y - 3 = 0) :=
sorry

end find_line_equation_through_ellipse_midpoint_l2130_213004


namespace isosceles_triangle_leg_l2130_213030

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ a = c ∨ b = c)

theorem isosceles_triangle_leg
  (a b c : ℝ)
  (h1 : is_isosceles_triangle a b c)
  (h2 : a + b + c = 18)
  (h3 : a = 8 ∨ b = 8 ∨ c = 8) :
  (a = 5 ∨ b = 5 ∨ c = 5 ∨ a = 8 ∨ b = 8 ∨ c = 8) :=
sorry

end isosceles_triangle_leg_l2130_213030


namespace pieces_of_paper_picked_up_l2130_213076

theorem pieces_of_paper_picked_up (Olivia : ℕ) (Edward : ℕ) (h₁ : Olivia = 16) (h₂ : Edward = 3) : Olivia + Edward = 19 :=
by
  sorry

end pieces_of_paper_picked_up_l2130_213076


namespace neither_sufficient_nor_necessary_l2130_213017

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0) ↔ (ab < ((a + b) / 2)^2)) :=
sorry

end neither_sufficient_nor_necessary_l2130_213017


namespace Levi_has_5_lemons_l2130_213007

theorem Levi_has_5_lemons
  (Levi Jayden Eli Ian : ℕ)
  (h1 : Jayden = Levi + 6)
  (h2 : Eli = 3 * Jayden)
  (h3 : Ian = 2 * Eli)
  (h4 : Levi + Jayden + Eli + Ian = 115) :
  Levi = 5 := 
sorry

end Levi_has_5_lemons_l2130_213007


namespace inequality_proof_l2130_213045

variable (a b : Real)
variable (θ : Real)

-- Line equation and point condition
def line_eq := ∀ x y, x / a + y / b = 1 → (x, y) = (Real.cos θ, Real.sin θ)
-- Main theorem to prove
theorem inequality_proof : (line_eq a b θ) → 1 / (a^2) + 1 / (b^2) ≥ 1 := sorry

end inequality_proof_l2130_213045


namespace opposite_of_2023_l2130_213086

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l2130_213086


namespace fraction_simplification_l2130_213043

theorem fraction_simplification 
  (a b c : ℝ)
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : a^2 + b^2 + c^2 ≠ 0) :
  (a^2 * b^2 + 2 * a^2 * b * c + a^2 * c^2 - b^4) / (a^4 - b^2 * c^2 + 2 * a * b * c^2 + c^4) =
  ((a * b + a * c + b^2) * (a * b + a * c - b^2)) / ((a^2 + b^2 - c^2) * (a^2 - b^2 + c^2)) :=
sorry

end fraction_simplification_l2130_213043


namespace member_sum_of_two_others_l2130_213069

def numMembers : Nat := 1978
def numCountries : Nat := 6

theorem member_sum_of_two_others :
  ∃ m : ℕ, m ∈ Finset.range numMembers.succ ∧
  ∃ a b : ℕ, a ∈ Finset.range numMembers.succ ∧ b ∈ Finset.range numMembers.succ ∧ 
  ∃ country : Fin (numCountries + 1), (a = m + b ∧ country = country) :=
by
  sorry

end member_sum_of_two_others_l2130_213069


namespace find_x_l2130_213033

theorem find_x (x : ℝ) (h : x - 1/10 = x / 10) : x = 1 / 9 := 
  sorry

end find_x_l2130_213033


namespace initially_caught_and_tagged_fish_l2130_213016

theorem initially_caught_and_tagged_fish (N T : ℕ) (hN : N = 800) (h_ratio : 2 / 40 = T / N) : T = 40 :=
by
  have hN : N = 800 := hN
  have h_ratio : 2 / 40 = T / 800 := by rw [hN] at h_ratio; exact h_ratio
  sorry

end initially_caught_and_tagged_fish_l2130_213016


namespace neg_p_exists_x_l2130_213078

-- Let p be the proposition: For all x in ℝ, x^2 - 3x + 3 > 0
def p : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 > 0

-- Prove that the negation of p implies that there exists some x in ℝ such that x^2 - 3x + 3 ≤ 0
theorem neg_p_exists_x : ¬p ↔ ∃ x : ℝ, x^2 - 3 * x + 3 ≤ 0 :=
by {
  sorry
}

end neg_p_exists_x_l2130_213078


namespace midpoint_one_sixth_one_ninth_l2130_213035

theorem midpoint_one_sixth_one_ninth : (1 / 6 + 1 / 9) / 2 = 5 / 36 := by
  sorry

end midpoint_one_sixth_one_ninth_l2130_213035


namespace custom_mul_expansion_l2130_213031

variable {a b x y : ℝ}

def custom_mul (a b : ℝ) : ℝ := (a - b)^2

theorem custom_mul_expansion (x y : ℝ) : custom_mul (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end custom_mul_expansion_l2130_213031


namespace find_f_2017_l2130_213063

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2017 (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_func_eq : ∀ x : ℝ, f (x + 3) * f x = -1)
  (h_val : f (-1) = 2) :
  f 2017 = -2 := sorry

end find_f_2017_l2130_213063


namespace math_problem_l2130_213046

def cond1 (R r a b c p : ℝ) : Prop := R * r = (a * b * c) / (4 * p)
def cond2 (a b c p : ℝ) : Prop := a * b * c ≤ 8 * p^3
def cond3 (a b c p : ℝ) : Prop := p^2 ≤ (3 * (a^2 + b^2 + c^2)) / 4
def cond4 (m_a m_b m_c R : ℝ) : Prop := m_a^2 + m_b^2 + m_c^2 ≤ (27 * R^2) / 4

theorem math_problem (R r a b c p m_a m_b m_c : ℝ) 
  (h1 : cond1 R r a b c p)
  (h2 : cond2 a b c p)
  (h3 : cond3 a b c p)
  (h4 : cond4 m_a m_b m_c R) : 
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ (27 * R^2) / 2 :=
by 
  sorry

end math_problem_l2130_213046


namespace turnip_bag_weighs_l2130_213001

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l2130_213001


namespace solution_proof_l2130_213024

def count_multiples (n : ℕ) (m : ℕ) (limit : ℕ) : ℕ :=
  (limit - 1) / m + 1

def problem_statement : Prop :=
  let multiples_of_10 := count_multiples 1 10 300
  let multiples_of_10_and_6 := count_multiples 1 30 300
  let multiples_of_10_and_11 := count_multiples 1 110 300
  let unwanted_multiples := multiples_of_10_and_6 + multiples_of_10_and_11
  multiples_of_10 - unwanted_multiples = 20

theorem solution_proof : problem_statement :=
  by {
    sorry
  }

end solution_proof_l2130_213024


namespace number_of_cutlery_pieces_added_l2130_213021

-- Define the initial conditions
def forks_initial := 6
def knives_initial := forks_initial + 9
def spoons_initial := 2 * knives_initial
def teaspoons_initial := forks_initial / 2
def total_initial_cutlery := forks_initial + knives_initial + spoons_initial + teaspoons_initial
def total_final_cutlery := 62

-- Define the total number of cutlery pieces added
def cutlery_added := total_final_cutlery - total_initial_cutlery

-- Define the theorem to prove
theorem number_of_cutlery_pieces_added : cutlery_added = 8 := by
  sorry

end number_of_cutlery_pieces_added_l2130_213021


namespace P_eq_Q_l2130_213011

open Set Real

def P : Set ℝ := {m | -1 < m ∧ m ≤ 0}
def Q : Set ℝ := {m | ∀ (x : ℝ), m * x^2 + 4 * m * x - 4 < 0}

theorem P_eq_Q : P = Q :=
by
  sorry

end P_eq_Q_l2130_213011


namespace eliza_irons_dress_in_20_minutes_l2130_213089

def eliza_iron_time : Prop :=
∃ d : ℕ, 
  (d ≠ 0 ∧  -- To avoid division by zero
  8 + 180 / d = 17 ∧
  d = 20)

theorem eliza_irons_dress_in_20_minutes : eliza_iron_time :=
sorry

end eliza_irons_dress_in_20_minutes_l2130_213089


namespace final_jacket_price_is_correct_l2130_213000

-- Define the initial price, the discounts, and the tax rate
def initial_price : ℝ := 120
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def sales_tax : ℝ := 0.05

-- Calculate the final price using the given conditions
noncomputable def price_after_first_discount := initial_price * (1 - first_discount)
noncomputable def price_after_second_discount := price_after_first_discount * (1 - second_discount)
noncomputable def final_price := price_after_second_discount * (1 + sales_tax)

-- The theorem to prove
theorem final_jacket_price_is_correct : final_price = 75.60 := by
  -- The proof is omitted
  sorry

end final_jacket_price_is_correct_l2130_213000


namespace factorize_x_cubed_minus_9x_l2130_213071

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l2130_213071


namespace difference_is_693_l2130_213044

noncomputable def one_tenth_of_seven_thousand : ℕ := 1 / 10 * 7000
noncomputable def one_tenth_percent_of_seven_thousand : ℕ := (1 / 10 / 100) * 7000
noncomputable def difference : ℕ := one_tenth_of_seven_thousand - one_tenth_percent_of_seven_thousand

theorem difference_is_693 :
  difference = 693 :=
by
  sorry

end difference_is_693_l2130_213044


namespace minimum_value_of_f_on_neg_interval_l2130_213005

theorem minimum_value_of_f_on_neg_interval (f : ℝ → ℝ) 
    (h_even : ∀ x, f (-x) = f x) 
    (h_increasing : ∀ x y, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y) 
  : ∀ x, -2 ≤ x → x ≤ -1 → f (-1) ≤ f x := 
by
  sorry

end minimum_value_of_f_on_neg_interval_l2130_213005


namespace combined_population_of_New_England_and_New_York_l2130_213015

noncomputable def population_of_New_England : ℕ := 2100000

noncomputable def population_of_New_York := (2/3 : ℚ) * population_of_New_England

theorem combined_population_of_New_England_and_New_York :
  population_of_New_England + population_of_New_York = 3500000 :=
by sorry

end combined_population_of_New_England_and_New_York_l2130_213015


namespace smallest_angle_opposite_smallest_side_l2130_213009

theorem smallest_angle_opposite_smallest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_inequality_proof)
  (h_condition : 3 * a = b + c) :
  smallest_angle_proof :=
sorry

end smallest_angle_opposite_smallest_side_l2130_213009


namespace sum_of_roots_l2130_213056

theorem sum_of_roots : 
  let a := 1
  let b := 2001
  let c := -2002
  ∀ x y: ℝ, (x^2 + b*x + c = 0) ∧ (y^2 + b*y + c = 0) -> (x + y = -b) :=
by
  sorry

end sum_of_roots_l2130_213056


namespace sequence_divisibility_l2130_213010

theorem sequence_divisibility (a b c : ℤ) (u v : ℕ → ℤ) (N : ℕ)
  (hu0 : u 0 = 1) (hu1 : u 1 = 1)
  (hu : ∀ n ≥ 2, u n = 2 * u (n - 1) - 3 * u (n - 2))
  (hv0 : v 0 = a) (hv1 : v 1 = b) (hv2 : v 2 = c)
  (hv : ∀ n ≥ 3, v n = v (n - 1) - 3 * v (n - 2) + 27 * v (n - 3))
  (hdiv : ∀ n ≥ N, u n ∣ v n) : 3 * a = 2 * b + c :=
by
  sorry

end sequence_divisibility_l2130_213010


namespace relationship_l2130_213065

noncomputable def a : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def b : ℝ := (3 / 5) ^ (2 / 5)
noncomputable def c : ℝ := Real.logb (3 / 5) (2 / 5)

theorem relationship : a < b ∧ b < c :=
by
  -- proof will go here
  sorry


end relationship_l2130_213065


namespace jars_contain_k_balls_eventually_l2130_213022

theorem jars_contain_k_balls_eventually
  (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hkp : k < 2 * p + 1) :
  ∃ n : ℕ, ∃ x y : ℕ, x + y = 2 * p + 1 ∧ (x = k ∨ y = k) :=
by
  sorry

end jars_contain_k_balls_eventually_l2130_213022


namespace precision_of_rounded_value_l2130_213083

-- Definition of the original problem in Lean 4
def original_value := 27390000000

-- Proof statement to check the precision of the rounded value to the million place
theorem precision_of_rounded_value :
  (original_value % 1000000 = 0) :=
sorry

end precision_of_rounded_value_l2130_213083


namespace sprinkles_remaining_l2130_213032

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) 
  (h1 : initial_cans = 12) 
  (h2 : remaining_cans = (initial_cans / 2) - 3) : 
  remaining_cans = 3 := 
by
  sorry

end sprinkles_remaining_l2130_213032


namespace cricket_team_members_l2130_213061

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ := 27)
  (wk_age : ℕ := captain_age + 1)
  (total_avg_age : ℕ := 23)
  (remaining_avg_age : ℕ := total_avg_age - 1)
  (total_age : ℕ := n * total_avg_age)
  (captain_and_wk_age : ℕ := captain_age + wk_age)
  (remaining_age : ℕ := (n - 2) * remaining_avg_age) : n = 11 := 
by
  sorry

end cricket_team_members_l2130_213061


namespace systematic_sampling_selects_616_l2130_213077

theorem systematic_sampling_selects_616 (n : ℕ) (h₁ : n = 1000) (h₂ : (∀ i : ℕ, ∃ j : ℕ, i = 46 + j * 10) → True) :
  (∃ m : ℕ, m = 616) :=
  by
  sorry

end systematic_sampling_selects_616_l2130_213077


namespace correct_statements_l2130_213013

-- Statement B
def statementB : Prop := 
∀ x : ℝ, x < 1/2 → (∃ y : ℝ, y = 2 * x + 1 / (2 * x - 1) ∧ y = -1)

-- Statement D
def statementD : Prop :=
∃ y : ℝ, (∀ x : ℝ, y = 1 / (Real.sin x) ^ 2 + 4 / (Real.cos x) ^ 2) ∧ y = 9

-- Combined proof problem
theorem correct_statements : statementB ∧ statementD :=
sorry

end correct_statements_l2130_213013


namespace percent_increase_output_l2130_213070

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l2130_213070


namespace min_value_of_squares_l2130_213068

theorem min_value_of_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1 / 3 := sorry

end min_value_of_squares_l2130_213068


namespace range_of_m_for_false_p_and_q_l2130_213081

theorem range_of_m_for_false_p_and_q (m : ℝ) :
  (¬ (∀ x y : ℝ, (x^2 / (1 - m) + y^2 / (m + 2) = 1) ∧ ∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (2 - m) = 1))) →
  (m ≤ 1 ∨ m ≥ 2) :=
sorry

end range_of_m_for_false_p_and_q_l2130_213081
