import Mathlib

namespace maciek_total_cost_l1401_140111

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l1401_140111


namespace find_k_l1401_140139

noncomputable def vec_na (x1 k : ℝ) : ℝ × ℝ := (x1 - k/4, 2 * x1^2)
noncomputable def vec_nb (x2 k : ℝ) : ℝ × ℝ := (x2 - k/4, 2 * x2^2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem find_k (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k / 2) 
  (h2 : x1 * x2 = -1) 
  (h3 : dot_product (vec_na x1 k) (vec_nb x2 k) = 0) : 
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 :=
by
  sorry

end find_k_l1401_140139


namespace people_in_third_row_l1401_140164

theorem people_in_third_row (row1_ini row2_ini left_row1 left_row2 total_left : ℕ) (h1 : row1_ini = 24) (h2 : row2_ini = 20) (h3 : left_row1 = row1_ini - 3) (h4 : left_row2 = row2_ini - 5) (h_total : total_left = 54) :
  total_left - (left_row1 + left_row2) = 18 := 
by
  sorry

end people_in_third_row_l1401_140164


namespace nancy_pots_created_on_Wednesday_l1401_140190

def nancy_pots_conditions (pots_Monday pots_Tuesday total_pots : ℕ) : Prop :=
  pots_Monday = 12 ∧ pots_Tuesday = 2 * pots_Monday ∧ total_pots = 50

theorem nancy_pots_created_on_Wednesday :
  ∀ pots_Monday pots_Tuesday total_pots,
  nancy_pots_conditions pots_Monday pots_Tuesday total_pots →
  (total_pots - (pots_Monday + pots_Tuesday) = 14) := by
  intros pots_Monday pots_Tuesday total_pots h
  -- proof would go here
  sorry

end nancy_pots_created_on_Wednesday_l1401_140190


namespace range_of_m_l1401_140150

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) m, 1 ≤ f x ∧ f x ≤ 10) ↔ 2 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l1401_140150


namespace problem_statement_problem_statement_2_l1401_140147

noncomputable def A (m : ℝ) : Set ℝ := {x | x > 2^m}
noncomputable def B : Set ℝ := {x | -4 < x - 4 ∧ x - 4 < 4}

theorem problem_statement (m : ℝ) (h1 : m = 2) :
  (A m ∪ B = {x | x > 0}) ∧ (A m ∩ B = {x | 4 < x ∧ x < 8}) :=
by sorry

theorem problem_statement_2 (m : ℝ) (h2 : A m ⊆ {x | x ≤ 0 ∨ 8 ≤ x}) :
  3 ≤ m :=
by sorry

end problem_statement_problem_statement_2_l1401_140147


namespace avg_marks_calculation_l1401_140182

theorem avg_marks_calculation (max_score : ℕ)
    (gibi_percent jigi_percent mike_percent lizzy_percent : ℚ)
    (hg : gibi_percent = 0.59) (hj : jigi_percent = 0.55) 
    (hm : mike_percent = 0.99) (hl : lizzy_percent = 0.67)
    (hmax : max_score = 700) :
    ((gibi_percent * max_score + jigi_percent * max_score +
      mike_percent * max_score + lizzy_percent * max_score) / 4 = 490) :=
by
  sorry

end avg_marks_calculation_l1401_140182


namespace solve_for_y_l1401_140121

theorem solve_for_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end solve_for_y_l1401_140121


namespace man_half_father_age_in_years_l1401_140144

theorem man_half_father_age_in_years
  (M F Y : ℕ) 
  (h1: M = (2 * F) / 5) 
  (h2: F = 25) 
  (h3: M + Y = (F + Y) / 2) : 
  Y = 5 := by 
  sorry

end man_half_father_age_in_years_l1401_140144


namespace arithmetic_sequence_15th_term_l1401_140157

theorem arithmetic_sequence_15th_term :
  let a1 := 3
  let d := 7
  let n := 15
  a1 + (n - 1) * d = 101 :=
by
  let a1 := 3
  let d := 7
  let n := 15
  sorry

end arithmetic_sequence_15th_term_l1401_140157


namespace evaluate_expression_l1401_140125

theorem evaluate_expression :
  (4 * 6) / (12 * 14) * ((8 * 12 * 14) / (4 * 6 * 8)) = 1 := 
by 
  sorry

end evaluate_expression_l1401_140125


namespace polynomial_irreducible_segment_intersect_l1401_140108

-- Part (a)
theorem polynomial_irreducible 
  (f : Polynomial ℤ) 
  (h_def : f = Polynomial.C 12 + Polynomial.X * Polynomial.C 9 + Polynomial.X^2 * Polynomial.C 6 + Polynomial.X^3 * Polynomial.C 3 + Polynomial.X^4) : 
  ¬ ∃ (p q : Polynomial ℤ), (Polynomial.degree p = 2) ∧ (Polynomial.degree q = 2) ∧ (f = p * q) :=
sorry

-- Part (b)
theorem segment_intersect 
  (n : ℕ) 
  (segments : Fin (2*n+1) → Set (ℝ × ℝ)) 
  (h_intersect : ∀ i, ∃ n_indices : Finset (Fin (2*n+1)), n_indices.card = n ∧ ∀ j ∈ n_indices, (segments i ∩ segments j).Nonempty) :
  ∃ i, ∀ j, i ≠ j → (segments i ∩ segments j).Nonempty :=
sorry


end polynomial_irreducible_segment_intersect_l1401_140108


namespace no_high_quality_triangle_exist_high_quality_quadrilateral_l1401_140114

-- Define the necessary predicate for a number being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the property of being a high-quality triangle
def high_quality_triangle (a b c : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + a)

-- Define the property of non-existence of a high-quality triangle
theorem no_high_quality_triangle (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) : 
  ¬high_quality_triangle a b c := by sorry

-- Define the property of being a high-quality quadrilateral
def high_quality_quadrilateral (a b c d : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + d) ∧ is_perfect_square (d + a)

-- Define the property of existence of a high-quality quadrilateral
theorem exist_high_quality_quadrilateral (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) : 
  high_quality_quadrilateral a b c d := by sorry

end no_high_quality_triangle_exist_high_quality_quadrilateral_l1401_140114


namespace isosceles_triangle_congruent_l1401_140123

theorem isosceles_triangle_congruent (A B C C1 : ℝ) 
(h₁ : A = B) 
(h₂ : C = C1) 
: A = B ∧ C = C1 :=
by
  sorry

end isosceles_triangle_congruent_l1401_140123


namespace volume_is_six_l1401_140153

-- Define the polygons and their properties
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0)
def rectangle (l w : ℝ) := (l > 0 ∧ w > 0)
def equilateral_triangle (s : ℝ) := (s > 0)

-- The given polygons
def A := right_triangle 1 2 (Real.sqrt 5)
def E := right_triangle 1 2 (Real.sqrt 5)
def F := right_triangle 1 2 (Real.sqrt 5)
def B := rectangle 1 2
def C := rectangle 2 3
def D := rectangle 1 3
def G := equilateral_triangle (Real.sqrt 5)

-- The volume of the polyhedron
-- Assume the largest rectangle C forms the base and a reasonable height
def volume_of_polyhedron : ℝ := 6

theorem volume_is_six : 
  (right_triangle 1 2 (Real.sqrt 5)) → 
  (rectangle 1 2) → 
  (rectangle 2 3) → 
  (rectangle 1 3) → 
  (equilateral_triangle (Real.sqrt 5)) → 
  volume_of_polyhedron = 6 := 
by 
  sorry

end volume_is_six_l1401_140153


namespace solve_for_k_l1401_140152

theorem solve_for_k : 
  ∃ (k : ℕ), k > 0 ∧ k * k = 2012 * 2012 + 2010 * 2011 * 2013 * 2014 ∧ k = 4048142 :=
sorry

end solve_for_k_l1401_140152


namespace arithmetic_sequence_geometric_sequence_added_number_l1401_140113

theorem arithmetic_sequence_geometric_sequence_added_number 
  (a : ℕ → ℤ)
  (h1 : a 1 = -8)
  (h2 : a 2 = -6)
  (h_arith : ∀ n, a n = -8 + (n-1) * 2)  -- derived from the conditions
  (x : ℤ)
  (h_geo : (-8 + x) * x = (-2 + x) * (-2 + x)) :
  x = -1 := 
sorry

end arithmetic_sequence_geometric_sequence_added_number_l1401_140113


namespace minimize_at_five_halves_five_sixths_l1401_140196

noncomputable def minimize_expression (x y : ℝ) : ℝ :=
  (y - 1)^2 + (x + y - 3)^2 + (2 * x + y - 6)^2

theorem minimize_at_five_halves_five_sixths (x y : ℝ) :
  minimize_expression x y = 1 / 6 ↔ (x = 5 / 2 ∧ y = 5 / 6) :=
sorry

end minimize_at_five_halves_five_sixths_l1401_140196


namespace jesus_squares_l1401_140189

theorem jesus_squares (J : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : linden_squares = 75)
  (h2 : pedro_squares = 200)
  (h3 : pedro_squares = J + linden_squares + 65) : 
  J = 60 := 
by
  sorry

end jesus_squares_l1401_140189


namespace inequality_of_pos_reals_l1401_140141

open Real

theorem inequality_of_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤
  (1 / 4) * (a + b + c) :=
by
  sorry

end inequality_of_pos_reals_l1401_140141


namespace sum_of_digits_of_fraction_is_nine_l1401_140154

theorem sum_of_digits_of_fraction_is_nine : 
  ∃ (x y : Nat), (4 / 11 : ℚ) = x / 10 + y / 100 + x / 1000 + y / 10000 + (x + y) / 100000 -- and other terms
  ∧ x + y = 9 := 
sorry

end sum_of_digits_of_fraction_is_nine_l1401_140154


namespace simplify_expression_l1401_140171

theorem simplify_expression (a b : ℤ) (h1 : a = 1) (h2 : b = -4) :
  4 * (a^2 * b + a * b^2) - 3 * (a^2 * b - 1) + 2 * a * b^2 - 6 = 89 := by
  sorry

end simplify_expression_l1401_140171


namespace john_max_books_l1401_140122

theorem john_max_books (h₁ : 4575 ≥ 0) (h₂ : 325 > 0) : 
  ∃ (x : ℕ), x = 14 ∧ ∀ n : ℕ, n ≤ x ↔ n * 325 ≤ 4575 := 
  sorry

end john_max_books_l1401_140122


namespace olympic_volunteers_selection_l1401_140133

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem olympic_volunteers_selection :
  (choose 4 3 * choose 3 1) + (choose 4 2 * choose 3 2) + (choose 4 1 * choose 3 3) = 34 := 
by
  sorry

end olympic_volunteers_selection_l1401_140133


namespace minimum_slit_length_l1401_140127

theorem minimum_slit_length (circumference : ℝ) (speed_ratio : ℝ) (reliability : ℝ) :
  circumference = 1 → speed_ratio = 2 → (∀ (s : ℝ), (s < 2/3) → (¬ reliable)) → reliability =
    2 / 3 :=
by
  intros hcirc hspeed hrel
  have s := (2 : ℝ) / 3
  sorry

end minimum_slit_length_l1401_140127


namespace product_of_possible_values_of_N_l1401_140103

theorem product_of_possible_values_of_N (M L N : ℝ) (h1 : M = L + N) (h2 : M - 5 = (L + N) - 5) (h3 : L + 3 = L + 3) (h4 : |(L + N - 5) - (L + 3)| = 2) : 10 * 6 = 60 := by
  sorry

end product_of_possible_values_of_N_l1401_140103


namespace solve_inequalities_l1401_140138

theorem solve_inequalities (x : ℝ) :
    ((x / 2 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x))) ↔ (-6 ≤ x ∧ x < -3 / 2) :=
by
  sorry

end solve_inequalities_l1401_140138


namespace product_of_ratios_eq_l1401_140106

theorem product_of_ratios_eq :
  (∃ x_1 y_1 x_2 y_2 x_3 y_3 : ℝ,
    (x_1^3 - 3 * x_1 * y_1^2 = 2006) ∧
    (y_1^3 - 3 * x_1^2 * y_1 = 2007) ∧
    (x_2^3 - 3 * x_2 * y_2^2 = 2006) ∧
    (y_2^3 - 3 * x_2^2 * y_2 = 2007) ∧
    (x_3^3 - 3 * x_3 * y_3^2 = 2006) ∧
    (y_3^3 - 3 * x_3^2 * y_3 = 2007)) →
    (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end product_of_ratios_eq_l1401_140106


namespace trading_organization_increase_price_l1401_140117

theorem trading_organization_increase_price 
  (initial_moisture_content : ℝ)
  (final_moisture_content : ℝ)
  (solid_mass : ℝ)
  (initial_total_mass final_total_mass : ℝ) :
  initial_moisture_content = 0.99 → 
  final_moisture_content = 0.98 →
  initial_total_mass = 100 →
  solid_mass = initial_total_mass * (1 - initial_moisture_content) →
  final_total_mass = solid_mass / (1 - final_moisture_content) →
  (final_total_mass / initial_total_mass) = 0.5 →
  100 * (1 - (final_total_mass / initial_total_mass)) = 100 :=
by sorry

end trading_organization_increase_price_l1401_140117


namespace domain_of_f_log2x_is_0_4_l1401_140173

def f : ℝ → ℝ := sorry

-- Given condition: domain of y = f(2x) is (-1, 1)
def dom_f_2x (x : ℝ) : Prop := -1 < 2 * x ∧ 2 * x < 1

-- Conclusion: domain of y = f(log_2 x) is (0, 4)
def dom_f_log2x (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem domain_of_f_log2x_is_0_4 (x : ℝ) :
  (dom_f_2x x) → (dom_f_log2x x) :=
by
  sorry

end domain_of_f_log2x_is_0_4_l1401_140173


namespace abs_neg_is_2_l1401_140101

theorem abs_neg_is_2 (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 :=
by sorry

end abs_neg_is_2_l1401_140101


namespace three_inequalities_true_l1401_140169

variables {x y a b : ℝ}
-- Declare the conditions as hypotheses
axiom h₁ : 0 < x
axiom h₂ : 0 < y
axiom h₃ : 0 < a
axiom h₄ : 0 < b
axiom hx : x^2 < a^2
axiom hy : y^2 < b^2

theorem three_inequalities_true : 
  (x^2 + y^2 < a^2 + b^2) ∧ 
  (x^2 * y^2 < a^2 * b^2) ∧ 
  (x^2 / y^2 < a^2 / b^2) :=
sorry

end three_inequalities_true_l1401_140169


namespace painter_total_cost_l1401_140183

def south_seq (n : Nat) : Nat :=
  4 + 6 * (n - 1)

def north_seq (n : Nat) : Nat :=
  5 + 6 * (n - 1)

noncomputable def digit_cost (n : Nat) : Nat :=
  String.length (toString n)

noncomputable def total_cost : Nat :=
  let south_cost := (List.range 25).map south_seq |>.map digit_cost |>.sum
  let north_cost := (List.range 25).map north_seq |>.map digit_cost |>.sum
  south_cost + north_cost

theorem painter_total_cost : total_cost = 116 := by
  sorry

end painter_total_cost_l1401_140183


namespace contrapositive_equivalence_l1401_140167

variable (Person : Type)
variable (Happy Have : Person → Prop)

theorem contrapositive_equivalence :
  (∀ (x : Person), Happy x → Have x) ↔ (∀ (x : Person), ¬Have x → ¬Happy x) :=
by
  sorry

end contrapositive_equivalence_l1401_140167


namespace cheetahs_pandas_ratio_l1401_140191

-- Let C denote the number of cheetahs 5 years ago.
-- Let P denote the number of pandas 5 years ago.
-- The conditions given are:
-- 1. The ratio of cheetahs to pandas 5 years ago was the same as it is now.
-- 2. The number of cheetahs has increased by 2.
-- 3. The number of pandas has increased by 6.
-- We need to prove that the current ratio of cheetahs to pandas is C / P.

theorem cheetahs_pandas_ratio
  (C P : ℕ)
  (h1 : C / P = (C + 2) / (P + 6)) :
  (C + 2) / (P + 6) = C / P :=
by sorry

end cheetahs_pandas_ratio_l1401_140191


namespace rate_of_stream_is_5_l1401_140119

-- Define the conditions
def boat_speed : ℝ := 16  -- Boat speed in still water
def time_downstream : ℝ := 3  -- Time taken downstream
def distance_downstream : ℝ := 63  -- Distance covered downstream

-- Define the rate of the stream as an unknown variable
def rate_of_stream (v : ℝ) : Prop := 
  distance_downstream = (boat_speed + v) * time_downstream

-- Statement to prove
theorem rate_of_stream_is_5 : 
  ∃ (v : ℝ), rate_of_stream v ∧ v = 5 :=
by
  use 5
  simp [boat_speed, time_downstream, distance_downstream, rate_of_stream]
  sorry

end rate_of_stream_is_5_l1401_140119


namespace smallest_solution_to_equation_l1401_140145

theorem smallest_solution_to_equation :
  let x := 4 - Real.sqrt 2
  ∃ x, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
       ∀ y, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y :=
  by
    let x := 4 - Real.sqrt 2
    sorry

end smallest_solution_to_equation_l1401_140145


namespace exists_base_for_part_a_not_exists_base_for_part_b_l1401_140151

theorem exists_base_for_part_a : ∃ b : ℕ, (3 + 4 = b) ∧ (3 * 4 = 1 * b + 5) := 
by
  sorry

theorem not_exists_base_for_part_b : ¬ ∃ b : ℕ, (2 + 3 = b) ∧ (2 * 3 = 1 * b + 1) :=
by
  sorry

end exists_base_for_part_a_not_exists_base_for_part_b_l1401_140151


namespace shaded_area_represents_correct_set_l1401_140116

theorem shaded_area_represents_correct_set :
  ∀ (U A B : Set ℕ), 
    U = {0, 1, 2, 3, 4} → 
    A = {1, 2, 3} → 
    B = {2, 4} → 
    (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} :=
by
  intros U A B hU hA hB
  -- The rest of the proof would go here
  sorry

end shaded_area_represents_correct_set_l1401_140116


namespace set_intersection_complement_l1401_140188

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {1, 3, 4, 6, 7}

theorem set_intersection_complement :
  A ∩ (U \ B) = {2, 5} := 
by
  sorry

end set_intersection_complement_l1401_140188


namespace sum_of_absolute_values_l1401_140187

variables {a : ℕ → ℤ} {S₁₀ S₁₈ : ℤ} {T₁₈ : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

def sum_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem sum_of_absolute_values 
  (h1 : a 0 > 0) 
  (h2 : a 9 * a 10 < 0) 
  (h3 : sum_n_terms a 9 = 36) 
  (h4 : sum_n_terms a 17 = 12) :
  (sum_n_terms a 9) - (sum_n_terms a 17 - sum_n_terms a 9) = 60 :=
sorry

end sum_of_absolute_values_l1401_140187


namespace rectangle_perimeter_gt_16_l1401_140165

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_area_gt_perim : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
by
  sorry

end rectangle_perimeter_gt_16_l1401_140165


namespace find_c_d_l1401_140181

theorem find_c_d (y : ℝ) (c d : ℕ) (hy : y^2 + 4*y + 4/y + 1/y^2 = 35)
  (hform : ∃ (c d : ℕ), y = c + Real.sqrt d) : c + d = 42 :=
sorry

end find_c_d_l1401_140181


namespace suitable_b_values_l1401_140102

theorem suitable_b_values (b : ℤ) :
  (∃ (c d e f : ℤ), 35 * c * d + (c * f + d * e) * b + 35 = 0 ∧
    c * e = 35 ∧ d * f = 35) →
  (∃ (k : ℤ), b = 2 * k) :=
by
  intro h
  sorry

end suitable_b_values_l1401_140102


namespace ajay_gain_l1401_140163

-- Definitions of the problem conditions as Lean variables/constants.
variables (kg1 kg2 kg_total : ℕ) 
variables (price1 price2 price3 cost1 cost2 total_cost selling_price gain : ℝ)

-- Conditions of the problem.
def conditions : Prop :=
  kg1 = 15 ∧ 
  kg2 = 10 ∧ 
  kg_total = kg1 + kg2 ∧ 
  price1 = 14.5 ∧ 
  price2 = 13 ∧ 
  price3 = 15 ∧ 
  cost1 = kg1 * price1 ∧ 
  cost2 = kg2 * price2 ∧ 
  total_cost = cost1 + cost2 ∧ 
  selling_price = kg_total * price3 ∧ 
  gain = selling_price - total_cost 

-- The theorem for the gain amount proof.
theorem ajay_gain (h : conditions kg1 kg2 kg_total price1 price2 price3 cost1 cost2 total_cost selling_price gain) : 
  gain = 27.50 :=
  sorry

end ajay_gain_l1401_140163


namespace prob1_prob2_prob3_l1401_140140

-- Problem (1)
theorem prob1 (a b : ℝ) :
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4) ∧ (2 * (a / 4 - 1) + (b / 3 + 2) = 5) →
  a = 12 ∧ b = -3 :=
by { sorry }

-- Problem (2)
theorem prob2 (m n x y a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (x = 10) ∧ (y = 6) ∧ 
  (5 * a₁ * (m - 3) + 3 * b₁ * (n + 2) = c₁) ∧ (5 * a₂ * (m - 3) + 3 * b₂ * (n + 2) = c₂) →
  (m = 5) ∧ (n = 0) :=
by { sorry }

-- Problem (3)
theorem prob3 (x y z : ℝ) :
  (3 * x - 2 * z + 12 * y = 47) ∧ (2 * x + z + 8 * y = 36) → z = 2 :=
by { sorry }

end prob1_prob2_prob3_l1401_140140


namespace investment_plan_optimization_l1401_140162

-- Define the given conditions.
def max_investment : ℝ := 100000
def max_loss : ℝ := 18000
def max_profit_A_rate : ℝ := 1.0     -- 100%
def max_profit_B_rate : ℝ := 0.5     -- 50%
def max_loss_A_rate : ℝ := 0.3       -- 30%
def max_loss_B_rate : ℝ := 0.1       -- 10%

-- Define the investment amounts.
def invest_A : ℝ := 40000
def invest_B : ℝ := 60000

-- Calculate profit and loss.
def profit : ℝ := (invest_A * max_profit_A_rate) + (invest_B * max_profit_B_rate)
def loss : ℝ := (invest_A * max_loss_A_rate) + (invest_B * max_loss_B_rate)
def total_investment : ℝ := invest_A + invest_B

-- Prove the required statement.
theorem investment_plan_optimization : 
    total_investment ≤ max_investment ∧ loss ≤ max_loss ∧ profit = 70000 :=
by
  simp [total_investment, profit, loss, invest_A, invest_B, 
    max_investment, max_profit_A_rate, max_profit_B_rate, 
    max_loss_A_rate, max_loss_B_rate, max_loss]
  sorry

end investment_plan_optimization_l1401_140162


namespace range_of_expr_l1401_140131

theorem range_of_expr (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
by
  sorry

end range_of_expr_l1401_140131


namespace trigonometric_identity_l1401_140124

theorem trigonometric_identity :
  7 * 6 * (1 / Real.tan (2 * Real.pi * 10 / 360) + Real.tan (2 * Real.pi * 5 / 360)) 
  = 7 * 6 * (1 / Real.sin (2 * Real.pi * 10 / 360)) := 
sorry

end trigonometric_identity_l1401_140124


namespace vendor_pepsi_volume_l1401_140107

theorem vendor_pepsi_volume 
    (liters_maaza : ℕ)
    (liters_sprite : ℕ)
    (num_cans : ℕ)
    (h1 : liters_maaza = 40)
    (h2 : liters_sprite = 368)
    (h3 : num_cans = 69)
    (volume_pepsi : ℕ)
    (total_volume : ℕ)
    (h4 : total_volume = liters_maaza + liters_sprite + volume_pepsi)
    (h5 : total_volume = num_cans * n)
    (h6 : 408 % num_cans = 0) :
  volume_pepsi = 75 :=
sorry

end vendor_pepsi_volume_l1401_140107


namespace apples_final_count_l1401_140168

theorem apples_final_count :
  let initial_apples := 200
  let shared_apples := 5
  let remaining_after_share := initial_apples - shared_apples
  let sister_takes := remaining_after_share / 2
  let half_rounded_down := 97 -- explicitly rounding down since 195 cannot be split exactly
  let remaining_after_sister := remaining_after_share - half_rounded_down
  let received_gift := 7
  let final_count := remaining_after_sister + received_gift
  final_count = 105 :=
by
  sorry

end apples_final_count_l1401_140168


namespace sum_of_coefficients_l1401_140176

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) 
  (h1 : quadratic a b c 3 = 0) 
  (h2 : quadratic a b c 7 = 0)
  (h3 : ∃ x0, (∀ x, quadratic a b c x ≥ quadratic a b c x0) ∧ quadratic a b c x0 = 20) :
  a + b + c = -105 :=
by 
  sorry

end sum_of_coefficients_l1401_140176


namespace maximize_product_numbers_l1401_140110

theorem maximize_product_numbers (a b : ℕ) (ha : a = 96420) (hb : b = 87531) (cond: a * b = 96420 * 87531):
  b = 87531 := 
by sorry

end maximize_product_numbers_l1401_140110


namespace boat_speed_in_still_water_l1401_140148

theorem boat_speed_in_still_water (B S : ℕ) (h1 : B + S = 13) (h2 : B - S = 5) : B = 9 :=
by
  sorry

end boat_speed_in_still_water_l1401_140148


namespace integral_log_eq_ln2_l1401_140105

theorem integral_log_eq_ln2 :
  ∫ x in (0 : ℝ)..(1 : ℝ), (1 / (x + 1)) = Real.log 2 :=
by
  sorry

end integral_log_eq_ln2_l1401_140105


namespace binary_representation_of_fourteen_l1401_140172

theorem binary_representation_of_fourteen :
  (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by
  sorry

end binary_representation_of_fourteen_l1401_140172


namespace regular_pay_correct_l1401_140158

noncomputable def regular_pay_per_hour (total_payment : ℝ) (regular_hours : ℕ) (overtime_hours : ℕ) (overtime_rate : ℝ) : ℝ :=
  let R := total_payment / (regular_hours + overtime_rate * overtime_hours)
  R

theorem regular_pay_correct :
  regular_pay_per_hour 198 40 13 2 = 3 :=
by
  sorry

end regular_pay_correct_l1401_140158


namespace solve_fractional_equation_l1401_140156

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 0) : (x + 1) / x = 2 / 3 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l1401_140156


namespace shire_total_population_l1401_140109

theorem shire_total_population :
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  n * avg_pop = 138750 :=
by
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  show n * avg_pop = 138750
  sorry

end shire_total_population_l1401_140109


namespace find_k_l1401_140115

-- Define vector a and vector b
def vec_a : (ℝ × ℝ) := (1, 1)
def vec_b : (ℝ × ℝ) := (-3, 1)

-- Define the expression for k * vec_a - vec_b
def k_vec_a_minus_vec_b (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - vec_b.1, k * vec_a.2 - vec_b.2)

-- Define the dot product condition for perpendicular vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved: k = -1 is the value that makes the dot product zero
theorem find_k : ∃ k : ℝ, dot_product (k_vec_a_minus_vec_b k) vec_a = 0 :=
by
  use -1
  sorry

end find_k_l1401_140115


namespace number_of_cows_on_boat_l1401_140155

-- Definitions based on conditions
def number_of_sheep := 20
def number_of_dogs := 14
def sheep_drowned := 3
def cows_drowned := 2 * sheep_drowned  -- Twice as many cows drowned as did sheep.
def dogs_made_it_shore := number_of_dogs  -- All dogs made it to shore.
def total_animals_shore := 35
def total_sheep_shore := number_of_sheep - sheep_drowned
def total_sheep_cows_shore := total_animals_shore - dogs_made_it_shore
def cows_made_it_shore := total_sheep_cows_shore - total_sheep_shore

-- Theorem stating the problem
theorem number_of_cows_on_boat : 
  (cows_made_it_shore + cows_drowned) = 10 := by
  sorry

end number_of_cows_on_boat_l1401_140155


namespace intersection_M_N_l1401_140178

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l1401_140178


namespace integer_solution_exists_l1401_140134

theorem integer_solution_exists : ∃ n : ℤ, (⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ∧ n = 6 := by
  sorry

end integer_solution_exists_l1401_140134


namespace drinks_left_for_Seungwoo_l1401_140185

def coke_taken_liters := 35 + 0.5
def cider_taken_liters := 27 + 0.2
def coke_drank_liters := 1 + 0.75

theorem drinks_left_for_Seungwoo :
  (coke_taken_liters - coke_drank_liters) + cider_taken_liters = 60.95 := by
  sorry

end drinks_left_for_Seungwoo_l1401_140185


namespace weight_of_a_l1401_140179

variables (a b c d e : ℝ)

theorem weight_of_a (h1 : (a + b + c) / 3 = 80)
                    (h2 : (a + b + c + d) / 4 = 82)
                    (h3 : e = d + 3)
                    (h4 : (b + c + d + e) / 4 = 81) :
  a = 95 :=
by
  sorry

end weight_of_a_l1401_140179


namespace handshaking_remainder_l1401_140104

-- Define number of people
def num_people := 11

-- Define N as the number of possible handshaking ways
def N : ℕ :=
sorry -- This will involve complicated combinatorial calculations

-- Define the target result to be proven
theorem handshaking_remainder : N % 1000 = 120 :=
sorry

end handshaking_remainder_l1401_140104


namespace glasses_total_l1401_140198

theorem glasses_total :
  ∃ (S L e : ℕ), 
    (L = S + 16) ∧ 
    (12 * S + 16 * L) / (S + L) = 15 ∧ 
    (e = 12 * S + 16 * L) ∧ 
    e = 480 :=
by
  sorry

end glasses_total_l1401_140198


namespace molecular_weight_4_benzoic_acid_l1401_140136

def benzoic_acid_molecular_weight : Float := (7 * 12.01) + (6 * 1.008) + (2 * 16.00)

def molecular_weight_4_moles_benzoic_acid (molecular_weight : Float) : Float := molecular_weight * 4

theorem molecular_weight_4_benzoic_acid :
  molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight = 488.472 :=
by
  unfold molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight
  -- rest of the proof
  sorry

end molecular_weight_4_benzoic_acid_l1401_140136


namespace sum_of_edges_of_square_l1401_140142

theorem sum_of_edges_of_square (u v w x : ℕ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) 
(hsum : u * x + u * v + v * w + w * x = 15) : u + v + w + x = 8 :=
by
  sorry

end sum_of_edges_of_square_l1401_140142


namespace num_men_in_first_group_l1401_140126

variable {x m w : ℝ}

theorem num_men_in_first_group (h1 : x * m + 8 * w = 6 * m + 2 * w)
  (h2 : 2 * m + 3 * w = 0.5 * (x * m + 8 * w)) : 
  x = 3 :=
sorry

end num_men_in_first_group_l1401_140126


namespace circle_area_l1401_140180

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end circle_area_l1401_140180


namespace solve_for_a_l1401_140149

theorem solve_for_a
  (h : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (x^2 - a * x + 2 < 0)) :
  a = 3 :=
sorry

end solve_for_a_l1401_140149


namespace printer_Z_time_l1401_140194

theorem printer_Z_time (T_Z : ℝ) (h1 : (1.0 / 15.0 : ℝ) = (15.0 * ((1.0 / 12.0) + (1.0 / T_Z))) / 2.0833333333333335) : 
  T_Z = 18.0 :=
sorry

end printer_Z_time_l1401_140194


namespace arithmetic_twelfth_term_l1401_140161

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l1401_140161


namespace urn_gold_coins_percentage_l1401_140193

noncomputable def percentage_gold_coins_in_urn
  (total_objects : ℕ)
  (beads_percentage : ℝ)
  (rings_percentage : ℝ)
  (coins_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  : ℝ := 
  let gold_coins_percentage := 100 - silver_coins_percentage
  let coins_total_percentage := total_objects * coins_percentage / 100
  coins_total_percentage * gold_coins_percentage / 100

theorem urn_gold_coins_percentage 
  (total_objects : ℕ)
  (beads_percentage rings_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  (h1 : beads_percentage = 15)
  (h2 : rings_percentage = 15)
  (h3 : beads_percentage + rings_percentage = 30)
  (h4 : coins_percentage = 100 - 30)
  (h5 : silver_coins_percentage = 35)
  : percentage_gold_coins_in_urn total_objects beads_percentage rings_percentage (100 - 30) 35 = 45.5 :=
sorry

end urn_gold_coins_percentage_l1401_140193


namespace gcd_1458_1479_l1401_140129

def a : ℕ := 1458
def b : ℕ := 1479
def gcd_ab : ℕ := 21

theorem gcd_1458_1479 : Nat.gcd a b = gcd_ab := sorry

end gcd_1458_1479_l1401_140129


namespace part_a_part_b_l1401_140174

-- Define n_mid_condition
def n_mid_condition (n : ℕ) : Prop := n % 2 = 1 ∧ n ∣ 2023^n - 1

-- Part a:
theorem part_a : ∃ (k₁ k₂ : ℕ), k₁ = 3 ∧ k₂ = 9 ∧ n_mid_condition k₁ ∧ n_mid_condition k₂ := by
  sorry

-- Part b:
theorem part_b : ∀ k, k ≥ 1 → n_mid_condition (3^k) := by
  sorry

end part_a_part_b_l1401_140174


namespace product_of_possible_values_l1401_140118

theorem product_of_possible_values :
  (∀ x : ℝ, abs (18 / x + 4) = 3 → x = -18 ∨ x = -18 / 7) →
  (∀ x1 x2 : ℝ, x1 = -18 → x2 = -18 / 7 → x1 * x2 = 324 / 7) :=
by
  intros h x1 x2 hx1 hx2
  rw [hx1, hx2]
  norm_num

end product_of_possible_values_l1401_140118


namespace parabola_find_c_l1401_140166

theorem parabola_find_c (b c : ℝ) 
  (h1 : (1 : ℝ)^2 + b * 1 + c = 2)
  (h2 : (5 : ℝ)^2 + b * 5 + c = 2) : 
  c = 7 := by
  sorry

end parabola_find_c_l1401_140166


namespace exponential_function_inequality_l1401_140100

theorem exponential_function_inequality {a : ℝ} (h0 : 0 < a) (h1 : a < 1) :
  (a^3) * (a^2) < a^2 :=
by
  sorry

end exponential_function_inequality_l1401_140100


namespace number_of_grey_birds_l1401_140146

variable (G : ℕ)

def grey_birds_condition1 := G + 6
def grey_birds_condition2 := G / 2

theorem number_of_grey_birds
  (H1 : G + 6 + G / 2 = 66) :
  G = 40 :=
by
  sorry

end number_of_grey_birds_l1401_140146


namespace initial_principal_amount_l1401_140177

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_principal_amount :
  let P := 4410 / (compound_interest 1 0.07 4 2 * compound_interest 1 0.09 2 2)
  abs (P - 3238.78) < 0.01 :=
by
  sorry

end initial_principal_amount_l1401_140177


namespace total_seashells_l1401_140160

theorem total_seashells 
  (sally_seashells : ℕ)
  (tom_seashells : ℕ)
  (jessica_seashells : ℕ)
  (h1 : sally_seashells = 9)
  (h2 : tom_seashells = 7)
  (h3 : jessica_seashells = 5) : 
  sally_seashells + tom_seashells + jessica_seashells = 21 :=
by
  sorry

end total_seashells_l1401_140160


namespace three_digit_numbers_count_correct_l1401_140175

def digits : List ℕ := [2, 3, 4, 5, 5, 5, 6, 6]

def three_digit_numbers_count (d : List ℕ) : ℕ := 
  -- To be defined: Full implementation for counting matching three-digit numbers
  sorry

theorem three_digit_numbers_count_correct :
  three_digit_numbers_count digits = 85 :=
sorry

end three_digit_numbers_count_correct_l1401_140175


namespace four_digit_numbers_count_l1401_140159

theorem four_digit_numbers_count : 
  (∀ d1 d2 d3 d4 : Fin 4, 
    (d1 = 1 ∨ d1 = 2 ∨ d1 = 3) ∧ 
    d2 ≠ d1 ∧ d2 ≠ 0 ∧ 
    d3 ≠ d1 ∧ d3 ≠ d2 ∧ 
    d4 ≠ d1 ∧ d4 ≠ d2 ∧ d4 ≠ d3) →
  3 * 6 = 18 := 
by
  sorry

end four_digit_numbers_count_l1401_140159


namespace division_quotient_example_l1401_140130

theorem division_quotient_example :
  ∃ q : ℕ,
    let dividend := 760
    let divisor := 36
    let remainder := 4
    dividend = divisor * q + remainder ∧ q = 21 :=
by
  sorry

end division_quotient_example_l1401_140130


namespace solve_quadratic_1_solve_quadratic_2_l1401_140128

open Real

theorem solve_quadratic_1 :
  (∃ x : ℝ, x^2 - 2 * x - 7 = 0) ∧
  (∀ x : ℝ, x^2 - 2 * x - 7 = 0 → x = 1 + 2 * sqrt 2 ∨ x = 1 - 2 * sqrt 2) :=
sorry

theorem solve_quadratic_2 :
  (∃ x : ℝ, 3 * (x - 2)^2 = x * (x - 2)) ∧
  (∀ x : ℝ, 3 * (x - 2)^2 = x * (x - 2) → x = 2 ∨ x = 3) :=
sorry

end solve_quadratic_1_solve_quadratic_2_l1401_140128


namespace player_A_wins_even_n_l1401_140135

theorem player_A_wins_even_n (n : ℕ) (hn : n > 0) (even_n : Even n) :
  ∃ strategy_A : ℕ → Bool, 
    ∀ (P Q : ℕ), P % 2 = 0 → (Q + P) % 2 = 0 :=
by 
  sorry

end player_A_wins_even_n_l1401_140135


namespace intersection_A_B_l1401_140143

def A : Set ℝ := { x : ℝ | |x - 1| < 2 }
def B : Set ℝ := { x : ℝ | x^2 - x - 2 > 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l1401_140143


namespace solution_set_of_inequality_l1401_140186

theorem solution_set_of_inequality :
  { x : ℝ | (2 * x - 1) / (x + 1) ≤ 1 } = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end solution_set_of_inequality_l1401_140186


namespace all_pets_combined_l1401_140132

def Teddy_initial_dogs : Nat := 7
def Teddy_initial_cats : Nat := 8
def Teddy_initial_rabbits : Nat := 6

def Teddy_adopted_dogs : Nat := 2
def Teddy_adopted_rabbits : Nat := 4

def Ben_dogs : Nat := 3 * Teddy_initial_dogs
def Ben_cats : Nat := 2 * Teddy_initial_cats

def Dave_dogs : Nat := (Teddy_initial_dogs + Teddy_adopted_dogs) - 4
def Dave_cats : Nat := Teddy_initial_cats + 13
def Dave_rabbits : Nat := 3 * Teddy_initial_rabbits

def Teddy_current_dogs : Nat := Teddy_initial_dogs + Teddy_adopted_dogs
def Teddy_current_cats : Nat := Teddy_initial_cats
def Teddy_current_rabbits : Nat := Teddy_initial_rabbits + Teddy_adopted_rabbits

def Teddy_total : Nat := Teddy_current_dogs + Teddy_current_cats + Teddy_current_rabbits
def Ben_total : Nat := Ben_dogs + Ben_cats
def Dave_total : Nat := Dave_dogs + Dave_cats + Dave_rabbits

def total_pets_combined : Nat := Teddy_total + Ben_total + Dave_total

theorem all_pets_combined : total_pets_combined = 108 :=
by
  sorry

end all_pets_combined_l1401_140132


namespace n_minus_two_is_square_of_natural_number_l1401_140112

theorem n_minus_two_is_square_of_natural_number 
  (n m : ℕ) 
  (hn: n ≥ 3) 
  (hm: m = n * (n - 1) / 2) 
  (hm_odd: m % 2 = 1)
  (unique_rem: ∀ i j : ℕ, i ≠ j → (i + j) % m ≠ (i + j) % m) :
  ∃ k : ℕ, n - 2 = k * k := 
sorry

end n_minus_two_is_square_of_natural_number_l1401_140112


namespace sequence_sum_after_operations_l1401_140199

-- Define the initial sequence length
def initial_sequence := [1, 9, 8, 8]

-- Define the sum of initial sequence
def initial_sum := initial_sequence.sum

-- Define the number of operations
def ops := 100

-- Define the increase per operation
def increase_per_op := 7

-- Define the final sum after operations
def final_sum := initial_sum + (increase_per_op * ops)

-- Prove the final sum is 726 after 100 operations
theorem sequence_sum_after_operations : final_sum = 726 := by
  -- Proof omitted as per instructions
  sorry

end sequence_sum_after_operations_l1401_140199


namespace geometric_series_sum_l1401_140170

theorem geometric_series_sum:
  let a := 1
  let r := 5
  let n := 5
  (1 - r^n) / (1 - r) = 781 :=
by
  let a := 1
  let r := 5
  let n := 5
  sorry

end geometric_series_sum_l1401_140170


namespace find_equation_of_perpendicular_line_l1401_140120

noncomputable def line_through_point_perpendicular
    (A : ℝ × ℝ) (a b c : ℝ) (hA : A = (2, 3)) (hLine : a = 2 ∧ b = 1 ∧ c = -5) :
    Prop :=
  ∃ (m : ℝ) (b1 : ℝ), (m = (1 / 2)) ∧
    (b1 = 3 - m * 2) ∧
    (∀ (x y : ℝ), y = m * (x - 2) + 3 → a * x + b * y + c = 0 → x - 2 * y + 4 = 0)

theorem find_equation_of_perpendicular_line :
  line_through_point_perpendicular (2, 3) 2 1 (-5) rfl ⟨rfl, rfl, rfl⟩ :=
sorry

end find_equation_of_perpendicular_line_l1401_140120


namespace simplify_expression_l1401_140184

theorem simplify_expression (y : ℝ) : 
  (3 * y) ^ 3 - 2 * y * y ^ 2 + y ^ 4 = 25 * y ^ 3 + y ^ 4 :=
by
  sorry

end simplify_expression_l1401_140184


namespace theater_seat_count_l1401_140197

theorem theater_seat_count (number_of_people : ℕ) (empty_seats : ℕ) (total_seats : ℕ) 
  (h1 : number_of_people = 532) 
  (h2 : empty_seats = 218) 
  (h3 : total_seats = number_of_people + empty_seats) : 
  total_seats = 750 := 
by 
  sorry

end theater_seat_count_l1401_140197


namespace reduced_price_per_kg_l1401_140192

variable (P R Q : ℝ)

theorem reduced_price_per_kg :
  R = 0.75 * P →
  1200 = (Q + 5) * R →
  Q * P = 1200 →
  R = 60 :=
by
  intro h₁ h₂ h₃
  sorry

end reduced_price_per_kg_l1401_140192


namespace distinct_real_roots_range_l1401_140137

theorem distinct_real_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ax^2 + 2 * x + 1 = 0 ∧ ay^2 + 2 * y + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
by
  sorry

end distinct_real_roots_range_l1401_140137


namespace find_a_l1401_140195

theorem find_a
  (a : ℝ)
  (h1 : ∃ P Q : ℝ × ℝ, (P.1 ^ 2 + P.2 ^ 2 - 2 * P.1 + 4 * P.2 + 1 = 0) ∧ (Q.1 ^ 2 + Q.2 ^ 2 - 2 * Q.1 + 4 * Q.2 + 1 = 0) ∧
                         (a * P.1 + 2 * P.2 + 6 = 0) ∧ (a * Q.1 + 2 * Q.2 + 6 = 0) ∧
                         ((P.1 - 1) * (Q.1 - 1) + (P.2 + 2) * (Q.2 + 2) = 0)) :
  a = 2 :=
by
  sorry

end find_a_l1401_140195
