import Complex
import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Probability.ProbD
import Mathlib.Analysis
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.OuterProduct
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Graph
import Mathlib.Combinatorics.HallMarriage
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Comb
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Order
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Real.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.GCD.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology
import Mathlib.Topology.Algebra.InfiniteSum

namespace four_digit_integers_with_4_or_5_l307_307119

theorem four_digit_integers_with_4_or_5 : 
  (finset.range 10000).filter (λ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ d, d ∈ [4, 5] ∧ d ∈ n.digits 10)).card = 5416 :=
sorry

end four_digit_integers_with_4_or_5_l307_307119


namespace smallest_n_for_grid_coloring_l307_307073

theorem smallest_n_for_grid_coloring :
  ∃ n : ℕ, (n >= 338 ∧ ∀ grid : array (fin 2023) (array (fin 2023) (fin n)), 
  (∀ i : fin 2023, 
  ∀ j₁ j₂ : fin i.val, 
  ∀ k₁ k₂ : fin j₂.val, 
  (grid[i][j₁] = grid[i][k₁] ∧ 
  grid[i][j₂] = grid[i][k₂] ∧ 
  j₁ + 1 < j₂ ∧ 
  k₁ + 1 < k₂ 
  → ∃ m₁ m₂ : fin i.val, grid[m₁][j₁] ≠ grid[m₂][k₂]))): n = 338 :=
begin
  sorry
end

end smallest_n_for_grid_coloring_l307_307073


namespace letters_identity_l307_307230

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l307_307230


namespace number_of_four_digit_numbers_with_two_identical_digits_l307_307277

/-- Four-digit numbers starting with 2 and having exactly two identical digits. -/
def four_digit_numbers_with_two_identical_digits : set ℕ :=
  {n | 2000 ≤ n ∧ n < 3000 ∧
    (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
     digits.count (λ d, d = 2) = 1 ∧ 
     (digits.remove 2).count (λ d, d = (digits.remove 2).head!) = 1)}

/-- The number of four-digit numbers starting with 2 that have exactly two identical digits is 384. -/
theorem number_of_four_digit_numbers_with_two_identical_digits :
  (four_digit_numbers_with_two_identical_digits.to_finset.card = 384) :=
sorry

end number_of_four_digit_numbers_with_two_identical_digits_l307_307277


namespace largest_fraction_sum_l307_307388

noncomputable def max_fraction_sum : ℚ :=
  max (finset.image (λ x, (1/3 : ℚ) + x) (finset.of_list [1/4, 1/5, 1/2, 1/7, 1/8]))

theorem largest_fraction_sum : max_fraction_sum = 5/6 := by
  sorry

end largest_fraction_sum_l307_307388


namespace probability_slope_ge_one_l307_307184

theorem probability_slope_ge_one :
  let P : ℝ × ℝ := (x, y) in
  let point := (2/3, 1/3) in
  ∀ x y : ℝ,
    0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
    (∃ m n : ℕ, m.is_coprime n ∧ ((P ∈ {point | slope (P, point) ≥ 1})) = (1/18)) :=
sorry

end probability_slope_ge_one_l307_307184


namespace quadratic_polynomial_l307_307050

noncomputable def p (x : ℝ) : ℝ := (14 * x^2 + 4 * x + 12) / 15

theorem quadratic_polynomial :
  p (-2) = 4 ∧ p 1 = 2 ∧ p 3 = 10 :=
by
  have : p (-2) = (14 * (-2 : ℝ) ^ 2 + 4 * (-2 : ℝ) + 12) / 15 := rfl
  have : p 1 = (14 * (1 : ℝ) ^ 2 + 4 * (1 : ℝ) + 12) / 15 := rfl
  have : p 3 = (14 * (3 : ℝ) ^ 2 + 4 * (3 : ℝ) + 12) / 15 := rfl
  -- You can directly state the equalities or keep track of the computation steps.
  sorry

end quadratic_polynomial_l307_307050


namespace find_b_l307_307656

theorem find_b (b : ℝ) (h1 : 0 < b) (h2 : b < 6)
  (h_ratio : ∃ (QRS QOP : ℝ), QRS / QOP = 4 / 25) : b = 6 :=
sorry

end find_b_l307_307656


namespace greatest_distance_P_D_l307_307786

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def A : point := (0, 0)
def B : point := (2, 0)
def C : point := (2, 2)
def D : point := (0, 2)

noncomputable def u (P : point) : ℝ := distance P A
noncomputable def v (P : point) : ℝ := distance P B
noncomputable def w (P : point) : ℝ := distance P C

theorem greatest_distance_P_D (P : point) 
  (h : u P ^ 2 + v P ^ 2 = 2 * (w P ^ 2)) :
  ∃ PD, PD = distance P D ∧ PD ≤ 2 * real.sqrt 2 :=
sorry

end greatest_distance_P_D_l307_307786


namespace p_value_correct_positive_m_exists_l307_307497

-- Define the parabola and its properties
def parabola (p : ℝ) : set (ℝ × ℝ) :=
  { P | ∃ (x y : ℝ), P = (x, y) ∧ y^2 = 2 * p * x ∧ p > 0 }

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- Define the condition involving the distance to the y-axis and the focus F
def distance_condition (P : ℝ × ℝ) (p : ℝ) (Fp : ℝ × ℝ) : Prop :=
  let distance_to_y_axis := abs P.1 in
  let distance_to_focus := Real.sqrt ((P.1 - Fp.1)^2 + P.2^2) in
  distance_to_y_axis = abs (distance_to_focus - 1)

-- Prove that p = 2 given the distance condition
theorem p_value_correct : ∀ (p : ℝ), p > 0 →
  (∀ (P : ℝ × ℝ), P ∈ parabola p → distance_condition P p (focus p)) →
  p = 2 :=
by
  sorry

-- Define the condition involving the vectors FA and FB
def vector_condition (A B F : ℝ × ℝ) : Prop :=
  let FA := (A.1 - F.1, A.2) in
  let FB := (B.1 - F.1, B.2) in
  (FA.1 * FB.1 + FA.2 * FB.2) < 0

-- Prove the existence of a positive m within a specific range
theorem positive_m_exists :
  ∃ (m : ℝ), (3 - 2 * Real.sqrt 2) < m ∧ m < (3 + 2 * Real.sqrt 2) ∧
  (∀ (t : ℝ), ∃ (A B : ℝ × ℝ), A.1^2 = 4*A.2 ∧ B.1^2 = 4*B.2 ∧
  vector_condition A B (focus 2)) :=
by
  sorry

end p_value_correct_positive_m_exists_l307_307497


namespace natural_number_with_property_l307_307782

theorem natural_number_with_property :
  ∃ n a b c : ℕ, (n = 10 * a + b) ∧ (100 * a + 10 * c + b = 6 * n) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (n = 18) :=
sorry

end natural_number_with_property_l307_307782


namespace diff_six_point_twenty_five_star_l307_307055

def star (z : ℝ) : ℕ :=
  (Nat.floor (z / 2) * 2 : ℕ)

theorem diff_six_point_twenty_five_star : (6.25 - star 6.25 = 2.25) := by
  sorry

end diff_six_point_twenty_five_star_l307_307055


namespace no_real_solution_for_eq_l307_307912

theorem no_real_solution_for_eq (y : ℝ) : ¬ ∃ y : ℝ, ((y - 4 * y + 10)^2 + 4 = -2 * |y|) :=
by
  sorry

end no_real_solution_for_eq_l307_307912


namespace find_a_cubed_l307_307861

-- Define the given condition
axiom condition (a : ℝ) : (a + 1 / a) ^ 2 = 4

-- Define the theorem to prove
theorem find_a_cubed (a : ℝ) [h : condition a] : a^3 + (1 / a)^3 = 2 := 
by
  sorry

end find_a_cubed_l307_307861


namespace sum_of_roots_of_P_l307_307849

noncomputable def P (x : ℝ) : ℝ :=
  (x - 1) ^ 1005 + 2 * (x - 2) ^ 1004 + 3 * (x - 3) ^ 1003 + 
  ∑ i in (1004), (i + 1) * (x - (i + 1)) ^ (1005 - (i + 1))
  -- ∑ notation is used for the summation pattern described

theorem sum_of_roots_of_P : 
  ∑ α in (roots P), α = 1003 :=
by 
  sorry

end sum_of_roots_of_P_l307_307849


namespace chance_variables_related_l307_307883

theorem chance_variables_related (k : ℝ) (P1 P2 : ℝ) 
  (h_k : k = 4.073) 
  (h_P1 : P1 = 0.05) 
  (h_P2 : P2 = 0.025)
  (H1 : ∀ x, x ≥ 3.841 → P1 ≤ 0.05) 
  (H2 : ∀ x, x ≥ 5.024 → P2 ≤ 0.025) : 
  P(k^2 ≥ 3.841) = 0.95 := 
  sorry

end chance_variables_related_l307_307883


namespace num_five_digit_div_by_12_l307_307509

theorem num_five_digit_div_by_12 : 
  let smallest_div_12 := 1008
  let largest_div_12 := 9996
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  in total_count = 6750 := by
  let smallest_div_12 := 1008
  let largest_div_12 := 9996 
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  have : total_count = 6750 := by
    sorry
  exact this

end num_five_digit_div_by_12_l307_307509


namespace expressible_as_sum_l307_307591

theorem expressible_as_sum (n : ℕ) (hn : n ≥ 50) : 
  ∃ x y : ℕ, n = x + y ∧ ∀ p : ℕ, p.prime → (p ∣ x ∨ p ∣ y) → (p ≤ Int.sqrt n) :=
sorry

end expressible_as_sum_l307_307591


namespace find_s_l307_307677

-- Conditions
def P := (0, 10)
def Q := (3, 0)
def R := (10, 0)
def horizontal_line (s : ℝ) := ∀ x, (x, s)

-- Statement with conditions and conclusion
theorem find_s (s : ℝ) (V W : ℝ × ℝ) (hV : V ∈ line_segment P Q) (hW : W ∈ line_segment P R) 
  (h_horizontal_line : ∀ x, V = (x, s) ∧ W = (x, s))
  (h_area : (1/2) * (distance V W) *  (10 - s) = 20) :
  s = 3.65 :=
begin
  sorry
end

end find_s_l307_307677


namespace shortest_distance_to_curve_l307_307109

-- Define the variables and conditions
variables {m k a b : ℝ}
noncomputable def y_curve (x : ℝ) : ℝ := m * x^2 + k
noncomputable def distance_sq (c : ℝ) : ℝ := (c - a)^2 + (y_curve c - b)^2

-- Proposition to prove the shortest distance
theorem shortest_distance_to_curve (h : ∀ c, distance_sq c ≥ distance_sq a) : 
  (abs (m * a^2 + k - b) = √(distance_sq a)) :=
by
  sorry

end shortest_distance_to_curve_l307_307109


namespace number_of_possible_values_l307_307344

def k_squared (x : ℕ) : ℝ :=
  let a := 5 * x
  let b := 5 * x
  let c := 4 * x
  let d := 6 * x
  let n := a + b + c + d
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def x_values : List ℕ := [14, 15, 16, 17, 18, 19]

theorem number_of_possible_values (x : ℕ) (hx : x ∈ x_values)
  (h1 : k_squared x > 2.706) : ∃ n, n = 6 := 
  sorry

end number_of_possible_values_l307_307344


namespace coefficient_of_m5n7_l307_307310

theorem coefficient_of_m5n7 (m n : ℕ) :
  (binomial 12 5) = 792 := by 
  sorry

end coefficient_of_m5n7_l307_307310


namespace sqrt_expression_l307_307409

theorem sqrt_expression (x y z : ℤ) (h1 : x = 5) (h2 : y = 2) (h3 : z = 6) :
  sqrt (114 + 44 * sqrt 6) = x + y * sqrt z ∧ x + y + z = 13 :=
  by
    sorry

end sqrt_expression_l307_307409


namespace chameleons_all_red_l307_307625

theorem chameleons_all_red (Y G R : ℕ) (total : ℕ) (P : Y = 7) (Q : G = 10) (R_cond : R = 17) (total_cond : Y + G + R = total) (total_value : total = 34) :
  ∃ x, x = R ∧ x = total ∧ ∀ z : ℕ, z ≠ 0 → total % 3 = z % 3 → ((R : ℕ) % 3 = z) :=
by
  sorry

end chameleons_all_red_l307_307625


namespace smallest_positive_period_and_range_monotonically_increasing_intervals_l307_307895

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - 2 * sin x ^ 2 + 1

theorem smallest_positive_period_and_range : 
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧ (∀ y, y ∈ set.range f ↔ y ∈ set.Icc (-real.sqrt 2) (real.sqrt 2)) :=
sorry

theorem monotonically_increasing_intervals : 
  ∀ k : ℤ, 
  ∃ a b, a = ↑k * π - (3 * π / 8) ∧ b = ↑k * π + (π / 8) ∧ (∀ x ∈ set.Icc a b, f' x > 0) :=
sorry

end smallest_positive_period_and_range_monotonically_increasing_intervals_l307_307895


namespace exam_passing_marks_l307_307760

theorem exam_passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.40 * T + 10 = P) 
  (h3 : 0.50 * T - 5 = P + 40) : 
  P = 210 := 
sorry

end exam_passing_marks_l307_307760


namespace simplify_expression_l307_307261

theorem simplify_expression (x : ℝ) : 8 * x + 15 - 3 * x + 27 = 5 * x + 42 := 
by
  sorry

end simplify_expression_l307_307261


namespace negation_of_p_l307_307901

variable (x : ℝ)

-- Define the original proposition p
def p := ∀ x, x^2 < 1 → x < 1

-- Define the negation of p
def neg_p := ∃ x₀, x₀^2 ≥ 1 ∧ x₀ < 1

-- State the theorem that negates p
theorem negation_of_p : ¬ p ↔ neg_p :=
by
  sorry

end negation_of_p_l307_307901


namespace james_sells_boxes_l307_307582

theorem james_sells_boxes (profit_per_candy_bar : ℝ) (total_profit : ℝ) 
                          (candy_bars_per_box : ℕ) (x : ℕ)
                          (h1 : profit_per_candy_bar = 1.5 - 1)
                          (h2 : total_profit = 25)
                          (h3 : candy_bars_per_box = 10) 
                          (h4 : total_profit = (x * candy_bars_per_box) * profit_per_candy_bar) :
                          x = 5 :=
by
  sorry

end james_sells_boxes_l307_307582


namespace manicure_cost_per_person_l307_307356

-- Definitions based on given conditions
def fingers_per_person : ℕ := 10
def total_fingers : ℕ := 210
def total_revenue : ℕ := 200  -- in dollars
def non_clients : ℕ := 11

-- Statement we want to prove
theorem manicure_cost_per_person :
  (total_revenue : ℚ) / (total_fingers / fingers_per_person - non_clients) = 9.52 :=
by
  sorry

end manicure_cost_per_person_l307_307356


namespace arithmetic_first_term_l307_307186

theorem arithmetic_first_term (a : ℕ) (d : ℕ) (T : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, T n = n * (2 * a + (n - 1) * d) / 2) →
  (∀ n : ℕ, T (4 * n) / T n = k) →
  d = 5 →
  k = 16 →
  a = 3 := 
by
  sorry

end arithmetic_first_term_l307_307186


namespace max_sin_angle_MCN_l307_307579

noncomputable def maximum_sin_angle_MCN (p : ℝ) (hp : p > 0) : ℝ :=
  let C := {x : ℝ × ℝ | x.1^2 = 2 * p * x.2} in
  let A := (0, p : ℝ) in
  let valid_circles := {r : ℝ | ∃ (x y : ℝ), (x, y) ∈ C ∧ (x^2 + y^2 = r^2) ∧ (0, p) ∈ circle (x, y) r} in
  ⨆ r ∈ valid_circles, 1 / Real.sqrt 2 -- maximum value of sin

theorem max_sin_angle_MCN (p : ℝ) (hp : p > 0) :
  maximum_sin_angle_MCN p hp = 1 / Real.sqrt 2 :=
sorry

end max_sin_angle_MCN_l307_307579


namespace five_n_plus_3_composite_l307_307087

theorem five_n_plus_3_composite (n : ℕ)
  (h1 : ∃ k : ℤ, 2 * n + 1 = k^2)
  (h2 : ∃ m : ℤ, 3 * n + 1 = m^2) :
  ¬ Prime (5 * n + 3) :=
by
  sorry

end five_n_plus_3_composite_l307_307087


namespace shea_buys_corn_l307_307831

noncomputable def num_pounds_corn (c b : ℚ) : ℚ :=
  if b + c = 24 ∧ 45 * b + 99 * c = 1809 then c else -1

theorem shea_buys_corn (c b : ℚ) : b + c = 24 ∧ 45 * b + 99 * c = 1809 → c = 13.5 :=
by
  intros h
  sorry

end shea_buys_corn_l307_307831


namespace conjugate_div_length_diagonal_OW_length_diagonal_Z1Z2_l307_307079

-- Definitions for given complex numbers
noncomputable def z1 : ℂ := 2 * complex.I
noncomputable def z2 : ℂ := 1 - complex.I

-- Proof statements
theorem conjugate_div (z1 z2 : ℂ) (hz1 : z1 = 2 * complex.I) (hz2 : z2 = 1 - complex.I) :
  complex.conj (z1 / z2) = -1 - complex.I :=
by sorry

theorem length_diagonal_OW (z1 z2 : ℂ) (hz1 : z1 = 2 * complex.I) (hz2 : z2 = 1 - complex.I) :
  complex.abs (z1 + z2) = real.sqrt 2 :=
by sorry

theorem length_diagonal_Z1Z2 (z1 z2 : ℂ) (hz1 : z1 = 2 * complex.I) (hz2 : z2 = 1 - complex.I) :
  complex.abs (z2 - z1) = real.sqrt 10 :=
by sorry

end conjugate_div_length_diagonal_OW_length_diagonal_Z1Z2_l307_307079


namespace solve_congruences_l307_307035

theorem solve_congruences :
  ∃ x : ℤ, 
  x ≡ 3 [ZMOD 7] ∧ 
  x^2 ≡ 44 [ZMOD 49] ∧ 
  x^3 ≡ 111 [ZMOD 343] ∧ 
  x ≡ 17 [ZMOD 343] :=
sorry

end solve_congruences_l307_307035


namespace convex_quad_area_le_half_sum_opposite_sides_l307_307628

variable (A B C D : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable {AB BC CD DA : ℝ}
variable {area_ABCD : ℝ}

/-- The area of any convex quadrilateral ABCD does not exceed the half-sum of the products of its opposite sides. -/
theorem convex_quad_area_le_half_sum_opposite_sides 
  (h : convex_quadrilateral A B C D)
  (h_area : area_ABCD = area_convex_quad A B C D) :
  area_ABCD ≤ 1/2 * (AB * CD + BC * AD) :=
sorry

end convex_quad_area_le_half_sum_opposite_sides_l307_307628


namespace max_stories_on_odd_pages_l307_307366

/-- Define the length of stories as a list from 1 to 30 -/
def story_lengths : List ℕ := List.range' 1 30

/-- Define a predicate to check if a page number is odd -/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- Define a function to calculate starting pages for each story -/
def starting_pages (story_lengths : List ℕ) : List ℕ :=
  List.scanl (λ acc len, acc + len) 1 story_lengths

/-- State the theorem: Maximum number of stories that start on an odd page -/
theorem max_stories_on_odd_pages :
  (starting_pages story_lengths).countp is_odd = 23 := by
  sorry

end max_stories_on_odd_pages_l307_307366


namespace percent_answered_second_correctly_l307_307535

theorem percent_answered_second_correctly
  (nA : ℝ) (nAB : ℝ) (n_neither : ℝ) :
  nA = 0.80 → nAB = 0.60 → n_neither = 0.05 → 
  (nA + nB - nAB + n_neither = 1) → 
  ((1 - n_neither) = nA + nB - nAB) → 
  nB = 0.75 :=
by
  intros h1 h2 h3 hUnion hInclusion
  sorry

end percent_answered_second_correctly_l307_307535


namespace dog_grouping_l307_307266

theorem dog_grouping 
  (dogs : Finset String)
  (Fluffy Nipper Spot : String)
  (h_size : dogs.card = 12)
  (h_dogs : Fluffy ∈ dogs ∧ Nipper ∈ dogs ∧ Spot ∈ dogs) :
  let remaining_dogs := dogs.erase Fluffy
  let remaining_dogs := remaining_dogs.erase Nipper
  let remaining_dogs := remaining_dogs.erase Spot
  let choose_3_dog_group := (remaining_dogs.erase Spot).choose 2
  let choose_4_dog_group := (remaining_dogs.erase choose_3_dog_group).choose 3
  let choose_5_dog_group := (remaining_dogs.erase choose_3_dog_group.erase choose_4_dog_group).choose 3
  choose_3_dog_group.card * 
  choose_4_dog_group.card * 
  choose_5_dog_group.card = 20160 := by
  let remaining_dogs := dogs.erase Fluffy
  let remaining_dogs := remaining_dogs.erase Nipper
  let remaining_dogs := remaining_dogs.erase Spot
  let choose_3_dog_group := remaining_dogs.erase Spot
  let choose_4_dog_group := remaining_dogs.erase choose_3_dog_group
  let choose_5_dog_group := remaining_dogs.erase choose_3_dog_group.erase choose_4_dog_group
  let binom_9_2 := binomial 9 2
  let binom_8_3 := binomial 8 3
  let binom_5_3 := binomial 5 3
  have : binom_9_2 * binom_8_3 * binom_5_3 = 20160 := sorry
  exact this

end dog_grouping_l307_307266


namespace round_trip_ticket_percentage_l307_307211

variable (P : ℕ) -- Total number of passengers on board
variable (x : ℕ) -- Percentage of passengers with round-trip tickets who did not take their cars

theorem round_trip_ticket_percentage (h1 : ∃ (p_round_trip_with_cars : ℕ), p_round_trip_with_cars = 20 * P / 100)
                                      (h2 : ∃ (p_round_trip_without_cars : ℕ), p_round_trip_without_cars = x * P / 100) :
    ∃ (total_percent_round_trip : ℕ), total_percent_round_trip = 20 + x :=
by
  intro h1 h2
  sorry

end round_trip_ticket_percentage_l307_307211


namespace projection_correct_l307_307422

open Real

def vector_projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let dot_vv := v.1 * v.1 + v.2 * v.2 + v.3 * v.3
  let scalar := dot_uv / dot_vv
  (scalar * v.1, scalar * v.2, scalar * v.3)

theorem projection_correct :
  vector_projection (4, -1, 3) (2, 1, -1) = (4 / 3, 2 / 3, -2 / 3) :=
by
  sorry

end projection_correct_l307_307422


namespace angle_between_unit_vectors_l307_307081

open Real

variables {a b : ℝ^3} (θ : ℝ)

-- Define unit vectors
def is_unit_vector (v : ℝ^3) := ∥v∥ = 1

-- Given conditions
variables (ha : is_unit_vector a) (hb : is_unit_vector b)
          (ha_b: ∥a - 2 • b∥ = √3)

-- The goal: prove the angle θ between a and b is π/3
theorem angle_between_unit_vectors : θ = π / 3 :=
  sorry

end angle_between_unit_vectors_l307_307081


namespace power_mod_remainder_l307_307320

theorem power_mod_remainder : (3^20) % 7 = 2 :=
by {
  -- condition: 3^6 ≡ 1 (mod 7)
  have h1 : (3^6) % 7 = 1 := by norm_num,
  -- we now use this to show 3^20 ≡ 2 (mod 7)
  calc
    (3^20) % 7 = ((3^6)^3 * 3^2) % 7 : by norm_num
          ... = (1^3 * 3^2) % 7       : by rw [←nat.modeq.modeq_iff_dvd, h1]
          ... =  (3^2) % 7            : by norm_num
          ... = 2                    : by norm_num
}

end power_mod_remainder_l307_307320


namespace axis_of_symmetry_parabola_l307_307647

def axis_of_symmetry (f : ℝ → ℝ) (a b c : ℝ) : ℝ :=
  if h : f = λ x, a*x^2 + b*x + c then -b / (2*a) else 0

theorem axis_of_symmetry_parabola :
  axis_of_symmetry (λ x, x^2 - 6*x + 1) 1 (-6) 1 = 3 :=
by
  -- Proof is omitted
  sorry

end axis_of_symmetry_parabola_l307_307647


namespace probability_fraction_sum_l307_307054

def problem_statement (a b : Finset ℕ) : Prop :=
a.card = 5 ∧
b.card = 5 ∧
a ∪ b = (Finset.range 500).erase 0 ∧

-- Conditions if a brick can fit in a box
∃ (a1 a2 a3 : ℕ), a \ {a1, a2, a3} ∪ {Inf a \ {a1, a2, a3}} = a ∧
∃ (b1 b2 b3 : ℕ), b \ {b1, b2, b3} ∪ {Inf b \ {b1, b2, b3}} = b ∧
a1 * a2 * a3 ≤ b1 * b2 * b3 ∧

-- Condition for products
Inf a \ {a1, a2, a3} * Inf(b \ {b1, b2, b3}) < Inf b \ {b1, b2, b3} *

-- Probability condition
(q : ℚ) ∧ q.num = 1 ∧ q.denom = 7

theorem probability_fraction_sum : ∃ q : ℚ, problem_statement a b ∧ q.num + q.denom = 8 :=
begin
  sorry
end

end probability_fraction_sum_l307_307054


namespace sam_vs_sarah_running_distance_l307_307929

theorem sam_vs_sarah_running_distance
  (street_width : ℝ)
  (block_side : ℝ)
  (sarah_side : ℝ)
  (sam_side : ℝ)
  (perimeter_sarah : sarah_side = block_side)
  (perimeter_sam_extension : sam_side = block_side + 2 * street_width) :
  let sarah_perimeter := 4 * block_side in
  let sam_perimeter := 4 * (block_side + 2 * street_width) in
  (sam_perimeter - sarah_perimeter) = 240 := 
by
  sorry

end sam_vs_sarah_running_distance_l307_307929


namespace product_squared_inequality_l307_307661

theorem product_squared_inequality (n : ℕ) (a : Fin n → ℝ) (h : (Finset.univ.prod (λ i => a i)) = 1) :
    (Finset.univ.prod (λ i => (1 + (a i)^2))) ≥ 2^n := 
sorry

end product_squared_inequality_l307_307661


namespace total_candies_in_third_set_l307_307710

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l307_307710


namespace domain_f_f_is_odd_range_of_x_l307_307467

noncomputable def f (x : ℝ) := Real.log (1 - x) - Real.log (1 + x)

-- Prove the domain of f(x) is (-1, 1)
theorem domain_f : ∀ x, f x ∈ ℝ → -1 < x ∧ x < 1 := sorry

-- Prove that f(x) is odd
theorem f_is_odd : ∀ x, f(-x) = -f(x) := sorry

-- Prove the range of x that satisfies lg( (1-x)/(1+x) ) < lg(2) is (-1/3, 1)
theorem range_of_x : ∀ x, Real.log(1 - x) - Real.log(1 + x) < Real.log(2) → -1/3 < x ∧ x < 1 := sorry

end domain_f_f_is_odd_range_of_x_l307_307467


namespace five_digit_numbers_correct_five_digit_numbers_div_by_5_correct_l307_307679

-- Define the universe of digits available
def digits : List ℕ := [0, 1, 2, 3, 4, 5]

-- Define function to generate permutations
def permutations {α : Type*} [DecidableEq α] : ℕ → List α → List (List α)
| 0, _ => [[]]
| _, [] => []
| k, (x :: xs) => (permutations k xs).map (cons x) ++ (permutations k xs).map (cons x)

-- Problem (1): The total number of different five-digit numbers without repeated digits.
def count_five_digit_numbers : ℕ :=
  (permutations 5 digits.erase 0).length

-- Problem (2): The number of different five-digit numbers without repeated digits that are divisible by 5.
def count_five_digit_numbers_div_by_5 : ℕ :=
  let nums_ending_in_0 := (permutations 4 (digits.erase 5)).length
  let nums_ending_in_5 := (permutations 4 (digits.erase 0).erase 1).length
  nums_ending_in_0 + nums_ending_in_5

-- Expected results for verification (these must match the computed values)
def expected_five_digit_numbers : ℕ := 600
def expected_five_digit_numbers_div_by_5 : ℕ := 216

-- Theorems to prove
theorem five_digit_numbers_correct : count_five_digit_numbers = expected_five_digit_numbers := sorry

theorem five_digit_numbers_div_by_5_correct : count_five_digit_numbers_div_by_5 = expected_five_digit_numbers_div_by_5 := sorry

end five_digit_numbers_correct_five_digit_numbers_div_by_5_correct_l307_307679


namespace c_geq_one_l307_307963

open Real

theorem c_geq_one (a b : ℕ) (c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h_eqn : (a + 1) / (b + c) = b / a) : c ≥ 1 :=
by
  sorry

end c_geq_one_l307_307963


namespace polynomial_roots_identity_l307_307191

theorem polynomial_roots_identity (a b c : ℝ) 
  (h_roots : (Polynomial.X^3 - 8 * Polynomial.X^2 + 11 * Polynomial.X - 3).roots = {a, b, c}) :
  (a / (b * c - 1) + b / (a * c - 1) + c / (a * b - 1)) = 13 :=
by
  sorry

end polynomial_roots_identity_l307_307191


namespace milojka_can_rule_milisav_l307_307754

theorem milojka_can_rule_milisav (f : ℕ) (s : ℕ) 
  (a : fin f → ℕ) (b : fin f → ℕ) 
  (hf : f = 26^(5^2019)) 
  (hs : s = 27^(5^2019)) 
  (a_sum : (∑ i in finset.range f, a i) = s) 
  (b_sum : (∑ i in finset.range f, b i) = s) : 
  ∃ b' : fin f → ℕ, 
    (∑ i in finset.range f, b' i) = s ∧ 
    (∑ i in finset.range f, ite (b' i > a i) 1 0) > (∑ i in finset.range f, ite (a i > b' i) 1 0) :=
sorry

end milojka_can_rule_milisav_l307_307754


namespace terrell_reps_necessary_l307_307642

noncomputable def repetitions_to_lift_same_weight (original_reps : ℕ) (original_weight : ℕ) (new_weight : ℕ) : ℕ :=
  (2 * original_weight * original_reps) / (2 * new_weight)

theorem terrell_reps_necessary :
  let original_reps := 10
  let original_weight := 25
  let new_weight := 20
  repetitions_to_lift_same_weight original_reps original_weight new_weight = 12.5 := by
  sorry

end terrell_reps_necessary_l307_307642


namespace towers_of_hanoi_minimal_moves_towers_of_hanoi_minimal_moves_with_intermediate_peg_towers_of_hanoi_minimal_moves_with_restriction_l307_307742

-- Part (a)
theorem towers_of_hanoi_minimal_moves :
  ∀ n, ∃ K, (K = 2^n - 1) :=
by
  sorry

-- Test case for n = 8
#eval towers_of_hanoi_minimal_moves 8 -- Expected: ∃ K, (K = 255)

-- Part (b)
theorem towers_of_hanoi_minimal_moves_with_intermediate_peg :
  ∀ n, ∃ K, (K = 3^n - 1) :=
by
  sorry

-- Test case for n = 8
#eval towers_of_hanoi_minimal_moves_with_intermediate_peg 8 -- Expected: ∃ K, (K = 6560)

-- Part (c)
theorem towers_of_hanoi_minimal_moves_with_restriction :
  ∀ n, ∃ K, (K = 2 * 3^(n-1) - 1) :=
by
  sorry

-- Test case for n = 8
#eval towers_of_hanoi_minimal_moves_with_restriction 8 -- Expected: ∃ K, (K = 4373)

end towers_of_hanoi_minimal_moves_towers_of_hanoi_minimal_moves_with_intermediate_peg_towers_of_hanoi_minimal_moves_with_restriction_l307_307742


namespace tangent_line_circle_m_values_l307_307655

theorem tangent_line_circle_m_values {m : ℝ} :
  (∀ (x y: ℝ), 3 * x + 4 * y + m = 0 → (x - 1)^2 + (y + 2)^2 = 4) →
  (m = 15 ∨ m = -5) :=
by
  sorry

end tangent_line_circle_m_values_l307_307655


namespace division_of_funds_l307_307342

theorem division_of_funds :
  ∃ (x : ℚ),
    (∃ (A B C : ℚ),
      (A + B + C = 1260) ∧
      (A = B + 35) ∧
      (A + C = (7/2) * B) ∧
      A / (B + C) = x) ∧
    x = 63 / 119 :=
begin
  sorry
end

end division_of_funds_l307_307342


namespace remainder_3_pow_20_mod_7_l307_307315

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end remainder_3_pow_20_mod_7_l307_307315


namespace sum_first_n_sequence_l307_307871

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n+1) = a n + d

def geometric_mean (x y z : ℝ) :=
  z^2 = x * y

def sequence_b (a : ℕ → ℝ) (n : ℕ) :=
  1 / (a n * a (n + 1))

def sum_sequence (b : ℕ → ℝ) (n : ℕ) :=
  ∑ i in finset.range n, b (i + 1)

theorem sum_first_n_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : a 3 + a 8 = 20)
  (h_geo : geometric_mean (a 2) (a 14) (a 5))
  (b : ℕ → ℝ := sequence_b a) :
  sum_sequence b n = n / (2 * n + 1) := by
  -- proof goes here
  sorry

end sum_first_n_sequence_l307_307871


namespace tangent_ellipse_isosceles_l307_307617

noncomputable def ellipse_eccentricity (a b : ℝ) (hab : a > b) (hb : b > 0) : ℝ :=
let e := Real.sqrt (1 - (b^2 / a^2)) in
if a > b ∧ b > 0 then e else 0

theorem tangent_ellipse_isosceles {a b : ℝ} (hab : a > b) (hb : b > 0) :
  ellipse_eccentricity a b hab hb = Real.sqrt 2 - 1 :=
sorry

end tangent_ellipse_isosceles_l307_307617


namespace infinitely_many_not_representable_l307_307630

def can_be_represented_as_p_n_2k (c : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ c = p + n^(2 * k)

theorem infinitely_many_not_representable :
  ∃ᶠ m in at_top, ¬ can_be_represented_as_p_n_2k (2^m + 1) := 
sorry

end infinitely_many_not_representable_l307_307630


namespace total_candies_third_set_l307_307706

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l307_307706


namespace find_expression_l307_307551

variable (x y k : ℝ)

def operation (x y k : ℝ) : ℝ := (x + y)^2 - k^2

theorem find_expression :
  operation (sqrt 11) (sqrt 11) k = 44 → k = 0 :=
by
  sorry

end find_expression_l307_307551


namespace part1_q1_l307_307876

open Set Real

def A (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def U : Set ℝ := univ

theorem part1_q1 (m : ℝ) (h : m = -1) : 
  A m ∪ B = {x | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

end part1_q1_l307_307876


namespace vector_n_value_l307_307098

theorem vector_n_value {n : ℤ} (hAB : (2, 4) = (2, 4)) (hBC : (-2, n) = (-2, n)) (hAC : (0, 2) = (2 + -2, 4 + n)) : n = -2 :=
by
  sorry

end vector_n_value_l307_307098


namespace angle_CED_greater_45_l307_307157

noncomputable theory
open Classical

-- Define an acute-angled triangle ABC
structure Triangle :=
  (A B C : Point)
  (acute : ∠BAC < 90 ∧ ∠ABC < 90 ∧ ∠ACB < 90)

-- Define the angle bisector AD
structure AngleBisector (T : Triangle) :=
  (D : Point)
  (on_bisector : A ∣ D ∈ bisector ∠BAC)

-- Define the altitude BE
structure Altitude (T : Triangle) :=
  (E : Point)
  (on_altitude : perp E ∣ B ∈ line BC)

-- Proof statement
theorem angle_CED_greater_45 (T : Triangle) (AB: AngleBisector T) (AE: Altitude T) :
  ∠CED > 45 :=
begin
  sorry
end

end angle_CED_greater_45_l307_307157


namespace total_candies_in_third_set_l307_307713

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l307_307713


namespace jean_jail_time_l307_307174

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l307_307174


namespace square_in_6_operations_multiply_in_20_operations_l307_307741

noncomputable def can_square_in_6_operations (x : ℝ) (h : 0 < x) : Prop := 
  ∃ (a b c d e f : ℝ → ℝ → ℝ) (g: ℝ → ℝ), 
    (∀ y1 y2 : ℝ, a y1 y2 = y1 + y2) ∧
    (∀ y1 y2 : ℝ, b y1 y2 = y1 - y2) ∧
    (∀ y : ℝ, c y = 1 / y) ∧
    (∀ y1 y2 : ℝ, d y1 y2 = c (a y1 y2)) ∧
    (∀ y1 y2 : ℝ, e y1 y2 = b (c y1) (d y1 y2)) ∧
    (∀ y1 : ℝ, f y1 = c (e y1 (a y1 1))) ∧
    (g x = b (f x) x) ∧
    (g x = x^2)

noncomputable def can_multiply_in_20_operations (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : Prop :=
  ∃ (a b c d e f g h i j k l m n o p q r s t : ℝ → ℝ → ℝ) (u: ℝ → ℝ),
    (∀ y1 y2 : ℝ, a y1 y2 = y1 + y2) ∧
    (∀ y1 y2 : ℝ, b y1 y2 = y1 - y2) ∧
    (∀ y1 y2 : ℝ, c y1 y2 = (a y1 y2) + 1) ∧
    (∀ y : ℝ, d y = 1 / y) ∧
    (∀ y1 y2 : ℝ, e y1 y2 = d (a y1 y2)) ∧
    (∀ y1 y2 : ℝ, f y1 y2 = b (d y1) (e y1 y2)) ∧
    (∀ y1 : ℝ, g y1 = d (f y1 (a y1 1))) ∧
    (∀ y1 y2 : ℝ, h y1 y2 = b (g y1) y1) ∧
    (∀ y1 : ℝ, i y1 = g ((a y1 y2))) ∧
    (∀ y1 : ℝ, j y1 = g ((b y1 y2))) ∧
    (∀ y1 y2 : ℝ, k y1 y2 = a (i y1) (j y1)) ∧
    (∀ y1 y2 : ℝ, l y1 y2 = d (k y1 y2)) ∧
    (∀ y : ℝ, m y = d (l y y1)) ∧
    (∀ y : ℝ, n y = e (m y) (h y y2)) ∧
    (∀ y : ℝ, o y = e (n y) (m y)) ∧
    (∀ y1 y2 : ℝ, p y1 y2 = b (o y1) y1) ∧
    (∀ y1 y2 : ℝ, r y1 y2 = (a y1 y2) - (b y1 y2)) ∧
    (∀ y : ℝ, s y = q (r y y2)) ∧
    (∀ y1 y2 : ℝ, t y1 y2 = d (s ((a y1 y2) - (b y1 y2)))) ∧
    (u y = d (s (u y1 y2 / q ((1 / (4 * x * y)))))) ∧
    (u (q (1 / tt (p x y)))) = x * y)

theorem square_in_6_operations (x : ℝ) (hx : 0 < x) : can_square_in_6_operations x hx := 
  sorry

theorem multiply_in_20_operations (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : can_multiply_in_20_operations x y hx hy := 
  sorry

end square_in_6_operations_multiply_in_20_operations_l307_307741


namespace count_ways_to_choose_and_discard_l307_307914

theorem count_ways_to_choose_and_discard :
  let suits := 4 
  let cards_per_suit := 13
  let ways_to_choose_4_different_suits := Nat.choose 4 4
  let ways_to_choose_4_cards := cards_per_suit ^ 4
  let ways_to_discard_1_card := 4
  1 * ways_to_choose_4_cards * ways_to_discard_1_card = 114244 :=
by
  sorry

end count_ways_to_choose_and_discard_l307_307914


namespace number_of_right_triangles_with_conditions_l307_307123

theorem number_of_right_triangles_with_conditions :
  let count := ∑ b in Finset.range 100, if ∃ a : ℕ, a^2 = 6 * b + 9 then 1 else 0
  in count = 6 :=
by
  sorry

end number_of_right_triangles_with_conditions_l307_307123


namespace total_candies_in_third_set_l307_307714

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l307_307714


namespace num_five_digit_div_by_12_l307_307511

theorem num_five_digit_div_by_12 : 
  let smallest_div_12 := 1008
  let largest_div_12 := 9996
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  in total_count = 6750 := by
  let smallest_div_12 := 1008
  let largest_div_12 := 9996 
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  have : total_count = 6750 := by
    sorry
  exact this

end num_five_digit_div_by_12_l307_307511


namespace present_price_after_discount_l307_307975

theorem present_price_after_discount :
  ∀ (P : ℝ), (∀ x : ℝ, (3 * x = P - 0.20 * P) ∧ (x = (P / 3) - 4)) → P = 60 → 0.80 * P = 48 :=
by
  intros P hP h60
  sorry

end present_price_after_discount_l307_307975


namespace trapezoid_count_in_pentagon_l307_307940

theorem trapezoid_count_in_pentagon (A B C D E : Point) (pentagon : Polygon) (is_regular : Regular pentagon) (midpoints : {A, B, C, D, E} = set.map Midpoint pentagon.sides) : 
  countTrapezoids pentagon midpoints = 35 :=
sorry

end trapezoid_count_in_pentagon_l307_307940


namespace pipes_needed_for_equal_volume_l307_307369

theorem pipes_needed_for_equal_volume (h : ℝ) : 
  let r1 := 4 -- radius of 8-inch diameter channel
  let r2 := 0.75 -- radius of 1.5-inch diameter pipes
  let V8 := π * r1^2 * h -- volume of the 8-inch channel
  let V1_5 := π * r2^2 * h -- volume of one 1.5-inch pipe
  let num_pipes := V8 / V1_5 -- the number of small pipes needed
  num_pipes.ceil = 29 :=
by
  sorry

end pipes_needed_for_equal_volume_l307_307369


namespace find_fx_for_l307_307275

theorem find_fx_for {f : ℕ → ℤ} (h1 : f 0 = 1) (h2 : ∀ x, f (x + 1) = f x + 2 * x + 3) : f 2012 = 4052169 :=
by
  sorry

end find_fx_for_l307_307275


namespace part_1_part_2_l307_307104

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x + a) + 2 * a

theorem part_1 (h : ∀ x : ℝ, f x a = f (3 - x) a) : a = -3 :=
by
  sorry

theorem part_2 (h : ∃ x : ℝ, f x a ≤ -abs (2 * x - 1) + a) : a ≤ -1 / 2 :=
by
  sorry

end part_1_part_2_l307_307104


namespace unique_reconstruction_l307_307219

-- Definition of the sums on the edges given the face values
variables (a b c d e f : ℤ)

-- The 12 edge sums
variables (e₁ e₂ e₃ e₄ e₅ e₆ e₇ e₈ e₉ e₁₀ e₁₁ e₁₂ : ℤ)
variables (h₁ : e₁ = a + b) (h₂ : e₂ = a + c) (h₃ : e₃ = a + d) 
          (h₄ : e₄ = a + e) (h₅ : e₅ = b + c) (h₆ : e₆ = b + f) 
          (h₇ : e₇ = c + f) (h₈ : e₈ = d + f) (h₉ : e₉ = d + e)
          (h₁₀ : e₁₀ = e + f) (h₁₁ : e₁₁ = b + d) (h₁₂ : e₁₂ = c + e)

-- Proving that the face values can be uniquely determined given the edge sums
theorem unique_reconstruction :
  ∃ a' b' c' d' e' f' : ℤ, 
    (e₁ = a' + b') ∧ (e₂ = a' + c') ∧ (e₃ = a' + d') ∧ (e₄ = a' + e') ∧ 
    (e₅ = b' + c') ∧ (e₆ = b' + f') ∧ (e₇ = c' + f') ∧ (e₈ = d' + f') ∧ 
    (e₉ = d' + e') ∧ (e₁₀ = e' + f') ∧ (e₁₁ = b' + d') ∧ (e₁₂ = c' + e') ∧ 
    (a = a') ∧ (b = b') ∧ (c = c') ∧ (d = d') ∧ (e = e') ∧ (f = f') := by
  sorry

end unique_reconstruction_l307_307219


namespace intersection_P_on_line_l307_307199

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-1, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨2, 0⟩

variable (D : Point)
variable (E : Point)

axiom D_on_hyperbola : D.x^2 - D.y^2 = 1
axiom D_not_A : D ≠ A
axiom E_on_hyperbola : E.x^2 - E.y^2 = 1

def line_eq (P Q : Point) (x : ℝ) : ℝ :=
  ((Q.y - P.y) / (Q.x - P.x)) * (x - P.x) + P.y

def line_AD := line_eq A D
def line_BE := line_eq B E

theorem intersection_P_on_line :
  ∃ (P : Point), P.x = 1/2 ∧ line_AD P.x = P.y ∧ line_BE P.x = P.y :=
sorry

end intersection_P_on_line_l307_307199


namespace candy_count_in_third_set_l307_307694

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l307_307694


namespace range_of_m_l307_307891

theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ y = 0) ∧ 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ x = 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l307_307891


namespace ryegrass_percentage_l307_307634

noncomputable def percentage_ryegrass (w total_weight : ℝ) : ℝ :=
  (w / total_weight) * 100

theorem ryegrass_percentage :
  ((percentage_ryegrass ((0.40 * 0.6667 + 0.25 * 0.3333) * W) W) = 35) :=
by
  -- Conditions
  let W : ℝ := 1 -- We can assume total_weight W is 1 for simplicity.
  let X_weight := 0.6667 * W
  let Y_weight := 0.3333 * W
  let ryegrass_X := 0.40 * X_weight
  let ryegrass_Y := 0.25 * Y_weight
  let total_ryegrass := ryegrass_X + ryegrass_Y
  let expected_percent := percentage_ryegrass total_ryegrass W
  -- Conclusion
  have h : expected_percent = 35 := sorry
  exact h

end ryegrass_percentage_l307_307634


namespace count_arithmetic_progression_integers_l307_307910

-- Define what it means for digits to form an arithmetic progression
def is_arithmetic_progression (a b c d : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (c < d) ∧ (b - a = c - b) ∧ (c - b = d - c)

-- Define the predicate for valid integers within the range
def valid_int (n : ℕ) : Prop :=
  1500 ≤ n ∧ n < 2000 ∧
  let ds := (List.range 4).map (λ i, (n / (10 ^ i)) % 10) in
  ds.nodup ∧ is_arithmetic_progression ds.nth 0 ds.nth 1 ds.nth 2 ds.nth 3

-- Main theorem: Prove the number of valid integers is 2
theorem count_arithmetic_progression_integers : 
  {n : ℕ | valid_int n}.to_finset.card = 2 :=
by sorry

end count_arithmetic_progression_integers_l307_307910


namespace q_is_false_l307_307142

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q) = false) (h2 : ¬p = false) : q = false :=
by
  sorry

end q_is_false_l307_307142


namespace union_of_sets_l307_307904

def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }

theorem union_of_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x : ℝ | 2 < x ∧ x < 10 }) :=
by
  sorry

end union_of_sets_l307_307904


namespace m_in_A_then_m_squared_in_A_area_of_triangle_range_of_t_l307_307594

-- Defining the set A
def A : Set ℝ := {a | ∃ x y : ℕ, a = x + sqrt 2 * y}

-- Condition for problem (1)
theorem m_in_A_then_m_squared_in_A (m : ℝ) (h : m ∈ A) : m^2 ∈ A := by
  sorry

-- Coordinates for points P2, P4, and P6 based on the description
def P2 : ℝ × ℝ := (1, 0)
def P4 : ℝ × ℝ := (2, 0)
def P6 : ℝ × ℝ := (0, 2)

-- Problem (2) - Area of the triangle
theorem area_of_triangle : Float :=
  let base := P4.1 - P2.1
  let height := P6.2
  (1 / 2) * base * height

-- Problem (3) - Range of t
def B (t : ℝ) : Set ℝ := {a | a = t ∨ a = 1}
theorem range_of_t (t : ℝ) : 2 < t ∧ t ≤ 2 + sqrt 2 :=
  sorry

#eval area_of_triangle -- Expected: 1.0

end m_in_A_then_m_squared_in_A_area_of_triangle_range_of_t_l307_307594


namespace fraction_sum_l307_307533

variable (x y : ℚ)

theorem fraction_sum (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := 
by
  sorry

end fraction_sum_l307_307533


namespace cone_volume_l307_307090

-- Define the base area and lateral area as given conditions
def base_area (r : ℝ) : ℝ := π * r^2
def lateral_area (r l : ℝ) : ℝ := π * r * l

-- State the problem: Given these conditions, prove the volume of the cone
theorem cone_volume (r l h V : ℝ) 
  (h_base : base_area r = 2 * π) 
  (h_lateral : lateral_area r l = 2 * sqrt 3 * π) 
  (h_height : h = 2)
  (h_volume : V = (1/3) * π * r^2 * h) :
  V = (4/3) * π :=
sorry

end cone_volume_l307_307090


namespace min_tan_sum_l307_307086

theorem min_tan_sum (A B C : ℝ) (h : True) 
  (h1: O = circumcenter (△ ABC))
  (h2: areas_of_BOC_COA_AOB_form_arith_seq (△ ABC) O)
  (h3: acute_triangle (△ ABC)): ∃ A C, tan A + 2 * tan C = 2 √6 := 
begin

sorry
end

end min_tan_sum_l307_307086


namespace find_a_plus_b_l307_307056

-- Define the operation x ⊕ y
def op (x y : ℝ) : ℝ := x + 2*y + 3

-- Given real numbers a and b that satisfy the condition
def condition (a b : ℝ) : Prop :=
  (op (op a^3 a^2) a = b) ∧ (op a^3 (op a^2 a) = b)

-- Prove the main statement
theorem find_a_plus_b (a b : ℝ) (h : condition a b) : a + b = 21 / 8 :=
sorry

end find_a_plus_b_l307_307056


namespace line_parabola_intersect_l307_307549

theorem line_parabola_intersect {k : ℝ} 
    (h1: ∀ x y : ℝ, y = k*x - 2 → y^2 = 8*x → x ≠ y)
    (h2: ∀ x1 x2 y1 y2 : ℝ, y1 = k*x1 - 2 → y2 = k*x2 - 2 → y1^2 = 8*x1 → y2^2 = 8*x2 → (x1 + x2) / 2 = 2) : 
    k = 2 := 
sorry

end line_parabola_intersect_l307_307549


namespace cricket_bat_cost_price_CPA_is_154_l307_307734

variable (CP_A SP_B SP_C : ℝ)
variable (condition1 : SP_B = 1.20 * CP_A)
variable (condition2 : SP_C = 1.25 * SP_B)
variable (condition3 : SP_C = 231)

theorem cricket_bat_cost_price_CPA_is_154 (h1 : SP_B = 1.20 * CP_A) (h2 : SP_C = 1.25 * SP_B) (h3 : SP_C = 231) : CP_A = 154 := by
  have h4 : SP_C = 1.25 * (1.20 * CP_A) := by rw [h1, h2]
  have h5 : SP_C = 1.50 * CP_A := by linarith
  rw [h5] at h3
  linarith

end cricket_bat_cost_price_CPA_is_154_l307_307734


namespace candy_count_in_third_set_l307_307698

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l307_307698


namespace smallest_prime_from_fraction_l307_307652

noncomputable def fractional_unit (a b : ℕ) : ℚ :=
1 / b

theorem smallest_prime_from_fraction (a b smallest_prime : ℕ) (h1 : a = 13) (h2 : b = 5) (h3 : smallest_prime = 2) (n : ℕ) : 
  let frac_unit := fractional_unit a b in
  (a - 3 * b) / b = smallest_prime → 
  (frac_unit = 1 / 5) := 
by
  sorry

end smallest_prime_from_fraction_l307_307652


namespace arccos_neg_half_eq_two_thirds_pi_l307_307823

theorem arccos_neg_half_eq_two_thirds_pi : 
  ∃ θ : ℝ, θ ∈ set.Icc 0 real.pi ∧ real.cos θ = -1/2 ∧ real.arccos (-1/2) = θ := 
sorry

end arccos_neg_half_eq_two_thirds_pi_l307_307823


namespace line_through_intersection_line_max_distance_l307_307069

theorem line_through_intersection (x y : ℝ) :
  (∃ a b c : ℝ, (2 * x + y - 5 = 0) ∧ (x - 2 * y = 0) ∧ (a ≠ 0 → (∃ c : ℝ, (x * a⁻¹ + y * a⁻¹ = 1) → (x + y - 3 = 0 ∨ x - 2 * y = 0))) ∧ (a = 0 → x - 2 * y = 0)): Prop :=
sorry

theorem line_max_distance (x y : ℝ) :
  (∃ a b c : ℝ, (5 = a * b) ∧ (c = b * y) ∧ (2 * c + x - 5 = 0) ∧ 
  ((abs ((b * y - x) / sqrt (b ^ 2 + 1)) = max (abs ((b * y - x) / sqrt (b ^ 2 + 1))))) → (3 * x - y = -5)) :=
sorry

end line_through_intersection_line_max_distance_l307_307069


namespace production_rate_l307_307919

theorem production_rate (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * x * x = x) → (y * y * z) / x^2 = y^2 * z / x^2 :=
by
  intro h
  sorry

end production_rate_l307_307919


namespace min_disks_needed_to_store_files_l307_307949

theorem min_disks_needed_to_store_files :
  ∃ d : ℕ, 
    (∀ (files : List ℝ), files = (List.replicate 5 0.95 ++ List.replicate 15 0.65 ++ List.replicate 20 0.55) → 
     (∀ (disk_capacity : ℝ), disk_capacity = 2.0 →
     (∀ (total_files : ℕ), total_files = 40 →
     (∀ (file_count : ℕ), file_count = (List.length files) →
     file_count = total_files →
     disks_needed files disk_capacity ≤ d) → d = 14))
  :=
begin
  sorry
end

def disks_needed (files : List ℝ) (disk_capacity : ℝ) : ℕ :=
  -- Function to calculate the minimum number of disks needed, we leave it as a stub since implementation is not required
  sorry

end min_disks_needed_to_store_files_l307_307949


namespace equal_probabilities_l307_307294

-- Definitions based on the conditions in the problem

def total_parts : ℕ := 160
def first_class_parts : ℕ := 48
def second_class_parts : ℕ := 64
def third_class_parts : ℕ := 32
def substandard_parts : ℕ := 16
def sample_size : ℕ := 20

-- Define the probabilities for each sampling method
def p1 : ℚ := sample_size / total_parts
def p2 : ℚ := (6 : ℚ) / first_class_parts  -- Given the conditions, this will hold for all classes
def p3 : ℚ := 1 / 8

theorem equal_probabilities :
  p1 = p2 ∧ p2 = p3 :=
by
  -- This is the end of the statement as no proof is required
  sorry

end equal_probabilities_l307_307294


namespace radius_of_second_circle_is_correct_l307_307794

noncomputable def radius_second_circle (AB BC : ℝ) (hAB_pos : 0 < AB) (hBC_pos : 0 < BC) : ℝ :=
  let AC := 2 * AB in
  let radius_first_circle := AB / 2 in
  let AO1 := AB + radius_first_circle in
  (AC / 3)

theorem radius_of_second_circle_is_correct :
  let AB := real.sqrt 3 in
  let BC := 3 in
  radius_second_circle AB BC _ _ = real.sqrt 3 / 6 :=
by
  -- proof goes here
  sorry

end radius_of_second_circle_is_correct_l307_307794


namespace sqrt_nested_eq_l307_307838

theorem sqrt_nested_eq (x : ℝ) (h : x = sqrt (2 - x)) : x = 1 := 
by 
  sorry

end sqrt_nested_eq_l307_307838


namespace number_of_arithmetic_sequences_l307_307203

theorem number_of_arithmetic_sequences (n : ℕ) (S : finset ℕ) (A : finset ℕ) :
  S = finset.range (n + 1) ∧ (A ⊆ S) ∧ (∀ x ∈ S, x ∉ A → ¬(exists d, d > 0 ∧ (finset.image ((+) d) A) = A ∪ {x})) →
  S = finset.range (n + 1) → 
  ∑ i in finset.range (n+1), i / 2 = nat.floor (n ^ 2 / 4) :=
by sorry

end number_of_arithmetic_sequences_l307_307203


namespace mutually_exclusive_events_l307_307931

-- Define the conditions
variable (redBalls greenBalls : ℕ)
variable (n : ℕ) -- Number of balls drawn
variable (event_one_red_ball event_two_green_balls : Prop)

-- Assumptions: more than two red balls and more than two green balls
axiom H1 : 2 < redBalls
axiom H2 : 2 < greenBalls

-- Assume that exactly one red ball and exactly two green balls are events
axiom H3 : event_one_red_ball = (n = 2 ∧ 1 ≤ redBalls ∧ 1 ≤ greenBalls)
axiom H4 : event_two_green_balls = (n = 2 ∧ greenBalls ≥ 2)

-- Definition of mutually exclusive events
def mutually_exclusive (A B : Prop) : Prop :=
  A ∧ B → false

-- Statement of the theorem
theorem mutually_exclusive_events :
  mutually_exclusive event_one_red_ball event_two_green_balls :=
by {
  sorry
}

end mutually_exclusive_events_l307_307931


namespace parallel_condition_coincide_condition_perpendicular_condition_l307_307907

-- Define the equations of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y = 8

-- Parallel lines condition
theorem parallel_condition (m : ℝ) : (l1 m = l2 m ↔ m = -7) →
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y) → False := sorry

-- Coincidence condition
theorem coincide_condition (m : ℝ) : 
  (l1 (-1) = l2 (-1)) :=
sorry

-- Perpendicular lines condition
theorem perpendicular_condition (m : ℝ) : 
  (m = - 13 / 3 ↔ (2 * (m + 3) + 4 * (m + 5) = 0)) :=
sorry

end parallel_condition_coincide_condition_perpendicular_condition_l307_307907


namespace total_people_in_tour_group_l307_307793

noncomputable def tour_group_total_people (θ : ℝ) (N : ℕ) (children_percentage young_adults_percentage older_people_percentage : ℝ) : Prop :=
  (older_people_percentage = (θ + 9) / 3.6) ∧
  (young_adults_percentage = (θ + 27) / 3.6) ∧
  (N * young_adults_percentage / 100 = N * children_percentage / 100 + 9) ∧
  (children_percentage = θ / 3.6) →
  N = 120

theorem total_people_in_tour_group (θ : ℝ) (N : ℕ) (children_percentage young_adults_percentage older_people_percentage : ℝ) :
  tour_group_total_people θ N children_percentage young_adults_percentage older_people_percentage :=
sorry

end total_people_in_tour_group_l307_307793


namespace pyramid_volume_calculation_l307_307361

noncomputable def pyramid_volume (a b c h: ℝ) : ℝ := (1/3) * (1/2) * a * b * h

theorem pyramid_volume_calculation :
  let a := 20
  let b := 20
  let c := 24
  let h := 25
  let n := 3
  let m := 800
  let volume := pyramid_volume a b c h
  let m_plus_n := m + n
  in 
  volume = 800 * real.sqrt 3 ∧ 
  m_plus_n = 803 :=
by {
  sorry
}

end pyramid_volume_calculation_l307_307361


namespace jean_total_jail_time_l307_307172

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l307_307172


namespace set_intersection_complement_l307_307501

variable (U : Set ℕ)
variable (P Q : Set ℕ)

theorem set_intersection_complement {U : Set ℕ} {P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4, 5, 6}) 
  (hP : P = {1, 2, 3, 4}) 
  (hQ : Q = {3, 4, 5, 6}) : 
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l307_307501


namespace find_alpha_l307_307082

theorem find_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h: sin α = 1 - sqrt 3 * tan (10 * real.pi / 180) * sin α) : α = 50 * real.pi / 180 :=
sorry

end find_alpha_l307_307082


namespace jelly_bean_matching_probability_l307_307371

-- Define the conditions
def abe_jelly_beans := [2/5, 3/5] -- probabilities for green and red respectively
def bob_jelly_beans := [2/7, 3/7] -- probabilities for green and red respectively

-- Define the event of matching colors
def prob_match_color : ℚ :=
  (abe_jelly_beans.head * bob_jelly_beans.head) + (abe_jelly_beans.tail.head * bob_jelly_beans.tail.head)

-- The proof goal
theorem jelly_bean_matching_probability :
  prob_match_color = 13 / 35 :=
by
  sorry

end jelly_bean_matching_probability_l307_307371


namespace five_digit_divisible_by_twelve_count_l307_307514

theorem five_digit_divisible_by_twelve_count : 
  let count_four_digit_multiples_of_12 := 
      ((9996 - 1008) / 12 + 1) in
  let count_five_digit_multiples_of_12 := 
      (9 * count_four_digit_multiples_of_12) in
  count_five_digit_multiples_of_12 = 6732 :=
by
  sorry

end five_digit_divisible_by_twelve_count_l307_307514


namespace find_a_l307_307966

noncomputable def a_real_number (a : ℝ) : Prop :=
  let z : ℂ := a * complex.I - ↑(10 : ℝ) / (3 - complex.I)
  z.im = 0

theorem find_a (a : ℝ) : a_real_number a ↔ a = -1 :=
by
  sorry

end find_a_l307_307966


namespace minimum_value_of_f_l307_307743

noncomputable def f (x y z : ℝ) := (x^2) / (1 + x) + (y^2) / (1 + y) + (z^2) / (1 + z)

theorem minimum_value_of_f (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
  (h7 : b * z + c * y = a) (h8 : a * z + c * x = b) (h9 : a * y + b * x = c) : 
  f x y z ≥ 1 / 2 :=
sorry

end minimum_value_of_f_l307_307743


namespace find_angle_between_vectors_l307_307114

-- Definitions and conditions
variables {a b : EuclideanSpace ℝ (Fin 3)}

-- |a| = 1
hypothesis h1 : ∥a∥ = 1

-- |b| = 2
hypothesis h2 : ∥b∥ = 2

-- (a + 2b) · (a - b) = -6
hypothesis h3 : (a + 2 • b) ⬝ (a - b) = -6

-- The statement to prove: θ = π/3
theorem find_angle_between_vectors : ∃ (θ : ℝ), θ = Real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)) ∧ θ = π / 3 :=
by
  exists (Real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)))
  split
  . sorry  -- Proof that θ = Real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥))
  . sorry  -- Proof that θ = π / 3 based on hypotheses and the dot product calculation

end find_angle_between_vectors_l307_307114


namespace total_food_each_day_l307_307660

-- Definitions as per conditions
def soldiers_first_side : Nat := 4000
def food_per_soldier_first_side : Nat := 10
def soldiers_difference : Nat := 500
def food_difference : Nat := 2

-- Proving the total amount of food
theorem total_food_each_day : 
  let soldiers_second_side := soldiers_first_side - soldiers_difference
  let food_per_soldier_second_side := food_per_soldier_first_side - food_difference
  let total_food_first_side := soldiers_first_side * food_per_soldier_first_side
  let total_food_second_side := soldiers_second_side * food_per_soldier_second_side
  total_food_first_side + total_food_second_side = 68000 := by
  -- Proof is omitted
  sorry

end total_food_each_day_l307_307660


namespace cary_net_calorie_deficit_l307_307812

noncomputable def net_calorie_deficit
  (distance : ℝ) -- Total round-trip distance in miles
  (speed_to : ℝ) -- Speed to the store in mph
  (speed_back : ℝ) -- Speed back from the store in mph
  (cal_per_mile : ℝ) -- Usual calories burned per mile
  (weight_increase_factor : ℝ) -- Calorie burn rate increase due to weight
  (candy_calories : ℕ) -- Calories in the candy

  (distance = 1.5) -- one-way distance in miles
  (speed_to = 3) -- mph
  (speed_back = 4) -- mph
  (cal_per_mile = 150) -- Calories per mile
  (weight_increase_factor = 1.2) -- 20% increase
  (candy_calories = 200) -- Candy calories

  : ℕ :=
  let cals_to_store := distance * cal_per_mile in
  let new_cal_per_mile := cal_per_mile * (speed_back / speed_to) in
  let cals_per_mile_back := new_cal_per_mile * weight_increase_factor in
  let cals_from_store := distance * cals_per_mile_back in
  let total_calories := cals_to_store + cals_from_store in
  let net_deficit := total_calories - candy_calories in
  net_deficit

theorem cary_net_calorie_deficit : net_calorie_deficit 1.5 3 4 150 1.2 200 = 385 :=
  by
    -- skipped proof
    sorry

end cary_net_calorie_deficit_l307_307812


namespace pool_filling_water_amount_l307_307302

theorem pool_filling_water_amount (Tina_pail Tommy_pail Timmy_pail Trudy_pail : ℕ) 
  (h1 : Tina_pail = 4)
  (h2 : Tommy_pail = Tina_pail + 2)
  (h3 : Timmy_pail = 2 * Tommy_pail)
  (h4 : Trudy_pail = (3 * Timmy_pail) / 2)
  (Timmy_trips Trudy_trips Tommy_trips Tina_trips: ℕ)
  (h5 : Timmy_trips = 4)
  (h6 : Trudy_trips = 4)
  (h7 : Tommy_trips = 6)
  (h8 : Tina_trips = 6) :
  Timmy_trips * Timmy_pail + Trudy_trips * Trudy_pail + Tommy_trips * Tommy_pail + Tina_trips * Tina_pail = 180 := by
  sorry

end pool_filling_water_amount_l307_307302


namespace option_a_is_incorrect_l307_307504

noncomputable section

variables {α β : Plane} {m n : Line}

-- Conditions
axiom planes_different : α ≠ β
axiom lines_non_coincident : m ≠ n

-- Given conditions for option A
axiom m_parallel_α : m ∥ α
axiom alpha_cap_beta_eq_n : α ∩ β = n

-- The incorrect proposition
theorem option_a_is_incorrect : ¬ (m ∥ n) :=
by
  sorry

end option_a_is_incorrect_l307_307504


namespace right_triangle_count_l307_307651

theorem right_triangle_count (A P B C Q D : ℝ × ℝ)
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : D ≠ A)
  (h5 : A ≠ C) (h6 : B ≠ D)
  (rectangle : A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2)
  (PQ_on_AC : ∃ t : ℝ , P = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
    ∧ Q = (A.1 + (1 - t) * (C.1 - A.1), A.2 + (1 - t) * (C.2 - A.2)))
  (PQ_perpendicular_BD : ∃ k : ℝ, P.2 - Q.2 = k * (B.2 - D.2) ∧ P.1 - Q.1 = -k * (B.1 - D.1)):
  (number_of_right_triangles A P B C Q D = 12) :=
by
  sorry

end right_triangle_count_l307_307651


namespace num_four_digit_int_with_4_or_5_correct_l307_307121

def num_four_digit_int_with_4_or_5 : ℕ :=
  5416

theorem num_four_digit_int_with_4_or_5_correct (A B : ℕ) (hA : A = 9000) (hB : B = 3584) :
  num_four_digit_int_with_4_or_5 = A - B :=
by
  rw [hA, hB]
  sorry

end num_four_digit_int_with_4_or_5_correct_l307_307121


namespace equal_IO_IH_l307_307446

-- Define the Triangle with given angle
variables {A B C : Type}
variables [Triangle A B C]
variables H : Orthocenter A B C
variables I : Incenter A B C
variables O : Circumcenter A B C

def angle_BAC : angle A B C := 60


theorem equal_IO_IH (H : Orthocenter A B C) (I : Incenter A B C) (O : Circumcenter A B C) :
  IO = IH := sorry

end equal_IO_IH_l307_307446


namespace yoojeong_rabbits_l307_307209

theorem yoojeong_rabbits :
  ∀ (R C : ℕ), 
  let minyoung_dogs := 9
  let minyoung_cats := 3
  let minyoung_rabbits := 5
  let minyoung_total := minyoung_dogs + minyoung_cats + minyoung_rabbits
  let yoojeong_total := minyoung_total + 2
  let yoojeong_dogs := 7
  let yoojeong_cats := R - 2
  yoojeong_total = yoojeong_dogs + (R - 2) + R → 
  R = 7 :=
by
  intros R C minyoung_dogs minyoung_cats minyoung_rabbits minyoung_total yoojeong_total yoojeong_dogs yoojeong_cats
  have h1 : minyoung_total = 9 + 3 + 5 := rfl
  have h2 : yoojeong_total = minyoung_total + 2 := by sorry
  have h3 : yoojeong_dogs = 7 := rfl
  have h4 : yoojeong_cats = R - 2 := by sorry
  sorry

end yoojeong_rabbits_l307_307209


namespace sum_first_5_terms_l307_307554

variable {a : ℕ → ℝ}

-- Each term is positive
axiom pos_geometric_sequence : ∀ n, a n > 0

-- Definition of a geometric sequence
axiom geometric_sequence : ∃ q > 0, ∀ n, a (n + 1) = a n * q

-- Given conditions
axiom a1_eq_2 : a 1 = 2
axiom arithmetic_cond : a 2 = 2 * (a 1 * q) ∧ ∀ n q, a (n + 3) + 2 = a n * (q^3 + 1)
axiom a5_term : a 5 = a 4 * (q^4 + 1)

-- Define the sum S_n of the first n terms
noncomputable def sum_first_n_terms (n : ℕ) : ℝ := 
  (a 1 * (1 - q^n)) / (1 - q)

-- Prove that S_5 equals 62 given the conditions
theorem sum_first_5_terms : sum_first_n_terms 5 = 62 := by
  sorry

end sum_first_5_terms_l307_307554


namespace find_sample_size_l307_307027

-- Define the frequencies
def frequencies (k : ℕ) : List ℕ := [2 * k, 3 * k, 4 * k, 6 * k, 4 * k, k]

-- Define the sum of the first three frequencies
def sum_first_three_frequencies (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k

-- Define the total number of data points
def total_data_points (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k + 6 * k + 4 * k + k

-- Define the main theorem
theorem find_sample_size (n k : ℕ) (h1 : sum_first_three_frequencies k = 27)
  (h2 : total_data_points k = n) : n = 60 := by
  sorry

end find_sample_size_l307_307027


namespace sum_of_coordinates_l307_307264

variable (f : ℝ → ℝ)

/-- Given that the point (2, 3) is on the graph of y = f(x) / 3,
    show that (9, 2/3) must be on the graph of y = f⁻¹(x) / 3 and the
    sum of its coordinates is 29/3. -/
theorem sum_of_coordinates (h : 3 = f 2 / 3) : (9 : ℝ) + (2 / 3 : ℝ) = 29 / 3 :=
by
  have h₁ : f 2 = 9 := by
    linarith
    
  have h₂ : f⁻¹ 9 = 2 := by
    -- We assume that f has an inverse and it is well-defined
    sorry

  have point_on_graph : (9, (2 / 3)) ∈ { p : ℝ × ℝ | p.2 = f⁻¹ p.1 / 3 } := by
    sorry

  show 9 + 2 / 3 = 29 / 3
  norm_num

end sum_of_coordinates_l307_307264


namespace polynomial_has_one_real_root_l307_307842

theorem polynomial_has_one_real_root (a : ℝ) :
  (∃! x : ℝ, x^3 - 2 * a * x^2 + 3 * a * x + a^2 - 2 = 0) :=
sorry

end polynomial_has_one_real_root_l307_307842


namespace product_probability_l307_307429

def set := {-2, -1, 0, 3, 4, 5}
def n_elements := 6
def n_combinations := 15
def favorable_outcomes := 5
def probability := 1/3

theorem product_probability : 
  let total_ways := choose n_elements 2 in
  let ways_product_zero := favorable_outcomes in
  (ways_product_zero / total_ways) = probability := 
by
  sorry

end product_probability_l307_307429


namespace inequality_solution_l307_307494

noncomputable def f (x : ℝ) : ℝ := log (|x| + 1) + 2^x + 2^(-x)

theorem inequality_solution (x : ℝ) :
  f (x + 1) < f (2 * x) ↔ x < -1/3 ∨ x > 1 := 
sorry

end inequality_solution_l307_307494


namespace negation_P_l307_307111

variable {x : ℝ}

def P : Prop := 2 * x + 1 ≤ 0

theorem negation_P : ¬P ↔ 2 * x + 1 > 0 :=
by sorry

end negation_P_l307_307111


namespace subsets_with_odd_elements_count_l307_307293

theorem subsets_with_odd_elements_count (S : Finset ℕ) (hS : S.card = 13) : 
  ∃ n : ℕ, n = 2^12 ∧ (Finset.filter (λ t, t.card % 2 = 1) (Finset.powerset S)).card = n := 
sorry

end subsets_with_odd_elements_count_l307_307293


namespace range_of_a_l307_307101

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

def piecewise_function (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (a + 1) * x + 1 else x^2 - 2 * a * x + 2

theorem range_of_a (a : ℝ) :
  is_increasing (piecewise_function a) ↔ (-1 < a ∧ a ≤ 1/3) :=
sorry

end range_of_a_l307_307101


namespace simplify_expr_l307_307725

-- Define the expression to be simplified
def simplify_expression (x y : ℝ) : ℝ := (x⁻¹ + y⁻¹)⁻¹

-- State the theorem
theorem simplify_expr (x y : ℝ) : simplify_expression x y = (x * y) / (x + y) :=
by 
  sorry  -- proof is omitted

end simplify_expr_l307_307725


namespace Jennie_material_needed_l307_307177

variable (quilts yards : ℕ)
variable (h : 7 * quilts = 21)

/-- Jennie makes quilts. She can make 7 quilts with 21 yards of material. Therefore, she needs 36 yards of material to make 12 quilts. -/
theorem Jennie_material_needed : quilts = 3 ∧ 12 * quilts = 36 := by
  split
  { sorry }
  { sorry }

end Jennie_material_needed_l307_307177


namespace least_value_a_plus_b_l307_307132

theorem least_value_a_plus_b (a b : ℝ) (h : log 10 a + log 10 b ≥ 9) : 
  a + b ≥ 2 * 10 ^ 4.5 := sorry

end least_value_a_plus_b_l307_307132


namespace sum_of_two_longest_altitudes_l307_307124

theorem sum_of_two_longest_altitudes (a b c : ℕ) (h : a = 9 ∧ b = 40 ∧ c = 41) (right_triangle : a^2 + b^2 = c^2) : 
  9 + 40 = 49 :=
by
  cases h with ha hbc
  cases hbc with hb hc
  rw [ha, hb, hc] at *
  sorry

end sum_of_two_longest_altitudes_l307_307124


namespace find_angle_C_l307_307945

-- conditions of the problem expressed as a Lean 4 theorem
theorem find_angle_C
  (A B C a b c : ℝ) -- Define angles A, B, C and sides a, b, c as real numbers
  (triangle_abc : a * sin A = c * sin C ∧ b * sin B = c * sin A)
  (given_condition : c * sin A = √3 * a * cos C) :
  C = π / 3 :=
by
  sorry

end find_angle_C_l307_307945


namespace dan_finishes_first_l307_307836

variable {x y : ℕ} -- ℕ represents natural numbers, assuming positive area and mowing rate

-- Define the areas of the gardens
def ella_garden_area := 3 * x
def dan_garden_area := x
def fran_garden_area := 6 * x

-- Define the mowing rates
def dan_mowing_rate := y
def fran_mowing_rate := 2 * y
def ella_mowing_rate := 0.5 * y

-- Define the mowing times
def ella_mowing_time := ella_garden_area / ella_mowing_rate
def dan_mowing_time := dan_garden_area / dan_mowing_rate
def fran_mowing_time := fran_garden_area / fran_mowing_rate

-- Prove that Dan finishes mowing his garden first
theorem dan_finishes_first (h1 : ella_garden_area = 3 * x)
                           (h2 : fran_garden_area = 6 * x)
                           (h3 : fran_mowing_rate = 2 * y)
                           (h4 : ella_mowing_rate = 0.5 * y) :
  dan_mowing_time < ella_mowing_time ∧ dan_mowing_time < fran_mowing_time :=
by
  sorry

end dan_finishes_first_l307_307836


namespace number_of_dials_l307_307998

theorem number_of_dials (k : ℕ) (aligned_sums : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → aligned_sums i % 12 = aligned_sums j % 12) ↔ k = 12 :=
by
  sorry

end number_of_dials_l307_307998


namespace rose_paid_after_discount_l307_307996

noncomputable def discount_percentage : ℝ := 0.1
noncomputable def original_price : ℝ := 10
noncomputable def discount_amount := discount_percentage * original_price
noncomputable def final_price := original_price - discount_amount

theorem rose_paid_after_discount : final_price = 9 := by
  sorry

end rose_paid_after_discount_l307_307996


namespace Alyssa_needs_to_cut_15_roses_l307_307670

/--
There are 3 roses from other sources in the vase.
Alyssa wants the ratio of roses from her garden to roses from other sources in the vase to be 5:1.
Prove that the number of roses Alyssa needs to cut from her garden is 15.
-/
theorem Alyssa_needs_to_cut_15_roses :
  ∃ x : ℕ, (∃ from_other_sources : ℕ, from_other_sources = 3) → (ratio : ℕ × ℕ, ratio = (5, 1)) → x = 15 :=
by
  sorry

end Alyssa_needs_to_cut_15_roses_l307_307670


namespace average_income_l307_307759

theorem average_income :
  let income_day1 := 300
  let income_day2 := 150
  let income_day3 := 750
  let income_day4 := 200
  let income_day5 := 600
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = 400 := by
  sorry

end average_income_l307_307759


namespace third_set_candies_l307_307692

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l307_307692


namespace s2_c2_minus_t2_negative_l307_307604

variable (m n : ℝ)
def u (m n : ℝ) := real.sqrt (m^2 + n^2)
def s (m n : ℝ) := n / u m n
def c (m n : ℝ) := m / u m n
def t (m n : ℝ) := n / m

theorem s2_c2_minus_t2_negative (m n : ℝ) : (s m n)^2 * (c m n)^2 - (t m n)^2 < 0.25 := 
sorry

end s2_c2_minus_t2_negative_l307_307604


namespace volume_of_regular_triangular_pyramid_l307_307287

variable (a b : ℝ)

theorem volume_of_regular_triangular_pyramid (ha : 0 < a) (hb : 0 < b) :
  let V := (a^3 / (12 * Real.sqrt 3)) in V = a^3 / (12 * Real.sqrt 3) :=
by
  -- Proof can be added here
  sorry

end volume_of_regular_triangular_pyramid_l307_307287


namespace letters_identity_l307_307239

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l307_307239


namespace points_in_quadrants_l307_307024

open Real

-- Define equidistance condition
def equidistant_from_axes (p : ℝ × ℝ) : Prop :=
  abs p.1 = abs p.2

-- Define the line equation condition
def on_line (p : ℝ × ℝ) : Prop :=
  4 * p.1 + 6 * p.2 = 18

-- Define quadrants 
def in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Main statement: the points on the line that are equidistant from axes lie in quadrants I or II.
theorem points_in_quadrants (p : ℝ × ℝ) : equidistant_from_axes(p) ∧ on_line(p) → (in_first_quadrant(p) ∨ in_second_quadrant(p)) :=
  sorry

end points_in_quadrants_l307_307024


namespace total_candies_third_set_l307_307701

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l307_307701


namespace largest_n_distinct_natural_numbers_l307_307437

def digits_permuted (m n: ℕ) : Prop :=
  ∃ l: list ℕ, l.perm m.digits ∧ l.perm n.digits ∧ l.head ≠ 0

theorem largest_n_distinct_natural_numbers
  (n : ℕ) (a : ℕ → ℕ)
  (h_distinct : ∀ (i j : ℕ), i ≠ j → a i ≠ a j)
  (h_permuted : ∀ (i j : ℕ), digits_permuted (a i) (a j))
  (h_divisible : ∀ (i : ℕ), a i % (a 0) = 0) :
  n ≤ 9 :=
by
  sorry

end largest_n_distinct_natural_numbers_l307_307437


namespace number_of_outcomes_l307_307337

theorem number_of_outcomes (n : ℕ) : 
  let outcomes := 2^n in
  outcomes = 2^n :=
begin
  sorry
end

end number_of_outcomes_l307_307337


namespace kayla_apples_correct_l307_307589

-- Definition of Kylie and Kayla's apples
def total_apples : ℕ := 340
def kaylas_apples (k : ℕ) : ℕ := 4 * k + 10

-- The main statement to prove
theorem kayla_apples_correct :
  ∃ K : ℕ, K + kaylas_apples K = total_apples ∧ kaylas_apples K = 274 :=
sorry

end kayla_apples_correct_l307_307589


namespace order_of_magnitude_l307_307835

theorem order_of_magnitude :
  let pi := Real.pi in
  let sin_T := Real.sin (20 * pi / 3) in
  let x := 0.3^pi in
  let y := pi^0.3 in
  x < sin_T ∧ sin_T < y :=
by
  sorry

end order_of_magnitude_l307_307835


namespace complement_union_eq_l307_307204

variable (U M P : Set ℕ)

def M := {2, 4, 6}
def P := {3, 4, 5}
def U := {x ∈ Nat | x > 0 ∧ x ≤ 7}

theorem complement_union_eq :
  (U \ (M ∪ P)) = {1, 7} :=
by
  sorry

end complement_union_eq_l307_307204


namespace largest_prime_divisor_25_sq_plus_72_sq_l307_307044

theorem largest_prime_divisor_25_sq_plus_72_sq : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (25^2 + 72^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (25^2 + 72^2) → q ≤ p :=
sorry

end largest_prime_divisor_25_sq_plus_72_sq_l307_307044


namespace determine_a_l307_307545

def f (x a : ℝ) : ℝ := x^2 - a * x

theorem determine_a (a : ℝ) :
  (∀ x, x ≤ 2 → (f x a)' ≤ 0) ∧ (∀ x, 2 < x → (f x a)' ≥ 0) →
  a = 4 :=
by
  sorry

end determine_a_l307_307545


namespace symmetry_condition_l307_307658

theorem symmetry_condition 
  (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : 
  (∀ a b : ℝ, b = 2 * a → (∃ y, y = (p * (b/2) + 2*q) / (r * (b/2) + 2*s) ∧  b = 2*(y/2) )) → 
  p + r = 0 :=
by
  sorry

end symmetry_condition_l307_307658


namespace unripe_apples_count_l307_307947

def total_apples : ℕ := 34
def apples_per_pie : ℕ := 4
def number_of_pies : ℕ := 7
def apples_needed_for_pies : ℕ := number_of_pies * apples_per_pie
def unripe_apples : ℕ := total_apples - apples_needed_for_pies

theorem unripe_apples_count : unripe_apples = 6 := by simp [unripe_apples, apples_needed_for_pies, total_apples]

end unripe_apples_count_l307_307947


namespace count_integers_leq_zero_l307_307021

def Q (x : ℕ) : ℕ := (List.range 50).map (λ i, x - (i + 1)^2).prod
def R (x : ℕ) : ℕ := (List.range 51).map (λ i, x - i).prod

theorem count_integers_leq_zero :
  {n : ℕ | Q n * R n ≤ 0}.to_finset.card = 3051 :=
sorry

end count_integers_leq_zero_l307_307021


namespace no_outliers_l307_307646

def data : List ℕ := [2, 11, 23, 23, 25, 35, 41, 41, 55, 67, 85]

def Q1 : ℕ := 23
def Q3 : ℕ := 55

def IQR : ℕ := Q3 - Q1
def factor : ℕ := 2

def lower_threshold : Int := Q1 - Int.ofNat (factor * IQR)
def upper_threshold : Int := Q3 + Int.ofNat (factor * IQR)

def is_outlier (x : Int) : Bool := x < lower_threshold ∨ x > upper_threshold

def count_outliers (l : List ℕ) : ℕ :=
  l.countp (fun x => is_outlier (Int.ofNat x))

theorem no_outliers : count_outliers data = 0 :=
by
  sorry

end no_outliers_l307_307646


namespace arcsin_neg_sqrt_two_over_two_eq_neg_pi_over_four_l307_307006

theorem arcsin_neg_sqrt_two_over_two_eq_neg_pi_over_four :
  ∀ (x : ℝ),
  arcsin x = -π / 4 → x = -√2 / 2 := 
by
  -- The statement says that if arcsin(x) = -π/4, then x must be -√2 / 2
  intro x h
  rw [← arcsin_eq, sin_arcsin] at h
  -- Here, we should translate the previous conditions and use them in the proof
  sorry

end arcsin_neg_sqrt_two_over_two_eq_neg_pi_over_four_l307_307006


namespace estimate_height_of_student_l307_307675

theorem estimate_height_of_student
  (x_values : List ℝ)
  (y_values : List ℝ)
  (h_sum_x : x_values.sum = 225)
  (h_sum_y : y_values.sum = 1600)
  (h_length : x_values.length = 10 ∧ y_values.length = 10)
  (b : ℝ := 4) :
  ∃ a : ℝ, ∀ x : ℝ, x = 24 → (b * x + a = 166) :=
by
  have avg_x := (225 / 10 : ℝ)
  have avg_y := (1600 / 10 : ℝ)
  have a := avg_y - b * avg_x
  use a
  intro x h
  rw [h]
  sorry

end estimate_height_of_student_l307_307675


namespace arithmetic_sequence_S10_l307_307557

-- Definition of an arithmetic sequence and the corresponding sums S_n.
def is_arithmetic_sequence (S : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, S (n + 1) = S n + d

theorem arithmetic_sequence_S10 
  (S : ℕ → ℕ)
  (h1 : S 1 = 10)
  (h2 : S 2 = 20)
  (h_arith : is_arithmetic_sequence S) :
  S 10 = 100 :=
sorry

end arithmetic_sequence_S10_l307_307557


namespace find_missing_number_l307_307368

-- Define the given numbers as a list
def given_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

-- Define the arithmetic mean condition
def arithmetic_mean (xs : List ℕ) (mean : ℕ) : Prop :=
  (xs.sum + mean) / xs.length.succ = 12

-- Define the proof problem
theorem find_missing_number (x : ℕ) (h : arithmetic_mean given_numbers x) : x = 7 := 
sorry

end find_missing_number_l307_307368


namespace find_k_l307_307890

open Real

variables {k : ℝ}
def a : (ℝ × ℝ) := (1, -1)
def b : (ℝ × ℝ) := (1/2 * (2 - k), 1/2 * (-2 * k - 3))

noncomputable def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
noncomputable def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (h1 : vector_sub a (scalar_mul 2 b) = (k - 1, 2 * k + 2))
               (h2 : dot_product a b = 0) : k = -5 :=
by
  sorry

end find_k_l307_307890


namespace bridge_length_proof_l307_307654

noncomputable def speed (v_km_hr : ℕ) : ℝ := v_km_hr * (1000.0 / 3600.0)

noncomputable def total_distance (speed_m_s : ℝ) (time_s : ℕ) : ℝ := speed_m_s * time_s

noncomputable def bridge_length (total_dist_m : ℝ) (train_length_m : ℕ) : ℝ := total_dist_m - train_length_m

theorem bridge_length_proof
  (train_length : ℕ)
  (speed_km_hr : ℕ)
  (time_s : ℕ)
  (total_dist : ℝ := total_distance (speed speed_km_hr) time_s)
  (bridge_len : ℝ := bridge_length total_dist train_length) :
  train_length = 160 →
  speed_km_hr = 45 →
  time_s = 30 →
  bridge_len = 215 :=
by
  intros train_length_eq speed_km_hr_eq time_s_eq
  rw [train_length_eq, speed_km_hr_eq, time_s_eq]
  sorry

end bridge_length_proof_l307_307654


namespace cosine_alpha_l307_307160

noncomputable def r : ℝ := real.sqrt ((-5:ℝ)^2 + (-12:ℝ)^2)

theorem cosine_alpha : (let α := real.acos (-5 / r) in cos α) = -(5:ℝ) / r :=
by
  -- This part essentially encapsulates the conditions within 'let α := ... in cos α'.
  -- The proof and computational steps would be included here, ending with showing the desired equation.
  sorry

end cosine_alpha_l307_307160


namespace alex_goal_possible_l307_307445

noncomputable def Alex_Can_Fill : Prop :=
  ∀ (x y z : ℕ), (x + y + z = 100) →
  ∃ (second_row : Fin 100 → ℕ),
    (∀ (i : Fin 100), (second_row i) ∈ {1, 2, 3}) ∧
    (∀ (i : Fin 100), (∃ (first_row : Fin 100 → ℕ), first_row i ≠ second_row i)) ∧
    (Finset.univ.sum second_row = 200)

theorem alex_goal_possible : Alex_Can_Fill :=
  sorry

end alex_goal_possible_l307_307445


namespace identity_of_letters_l307_307249

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l307_307249


namespace minimum_value_frac_sum_l307_307729

-- Define the statement problem C and proof outline skipping the steps
theorem minimum_value_frac_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 :=
by
  -- Proof is to be constructed here
  sorry

end minimum_value_frac_sum_l307_307729


namespace triangles_centroid_square_divisible_by_three_l307_307416

theorem triangles_centroid_square_divisible_by_three (k : ℕ) :
  (∀ (T : list (list (ℕ × ℕ))), 
    (∀ t ∈ T, ∃ (A B C: ℕ × ℕ), t = [A, B, C] ∧ 
      (A.1 + B.1 + C.1) / 3 ∈ ℕ ∧ (A.2 + B.2 + C.2) / 3 ∈ ℕ ∧ (∃ D: ℕ × ℕ, Int.gcd k (D.1 + D.2) = 1)) ∧ 
    (∃ (P : ℕ × ℕ), ∀ p ∈ P, 0 ≤ p.1 ∧ p.1 ≤ k ∧ 0 ≤ p.2 ∧ p.2 ≤ k ∧
      (∀ t1 t2 ∈ T, t1 ≠ t2 → t1 ∩ t2 = ∅ ∨ t1 ∩ t2 = {p} ∨ ∃ q1 q2 : ℕ × ℕ, t1 ∩ t2 = [q1, q2]))) →
  k % 3 = 0 :=
begin
  sorry
end

end triangles_centroid_square_divisible_by_three_l307_307416


namespace three_solutions_no_solutions_2891_l307_307480

theorem three_solutions (n : ℤ) (hpos : n > 0) (hx : ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3 * x1 * y1^2 + y1^3 = n ∧ 
    x2^3 - 3 * x2 * y2^2 + y2^3 = n ∧ 
    x3^3 - 3 * x3 * y3^2 + y3^3 = n := 
sorry

theorem no_solutions_2891 : ¬ ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end three_solutions_no_solutions_2891_l307_307480


namespace absolute_value_h_of_quadratic_l307_307666

theorem absolute_value_h_of_quadratic : ∀ (h : ℝ),
  let r := - h + sqrt(h^2 - 2)
  let s := - h - sqrt(h^2 - 2)
  (r^(2:ℝ) + s^(2:ℝ) = 8) → |h| = sqrt 3 :=
by
  sorry

end absolute_value_h_of_quadratic_l307_307666


namespace value_of_expression_l307_307528

theorem value_of_expression (x : ℝ) (hx : 23 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 5 :=
by
  sorry

end value_of_expression_l307_307528


namespace number_of_teachers_under_40_in_sample_l307_307155

def proportion_teachers_under_40 (total_teachers teachers_under_40 : ℕ) : ℚ :=
  teachers_under_40 / total_teachers

def sample_teachers_under_40 (sample_size : ℕ) (proportion : ℚ) : ℚ :=
  sample_size * proportion

theorem number_of_teachers_under_40_in_sample
(total_teachers teachers_under_40 teachers_40_and_above sample_size : ℕ)
(h_total : total_teachers = 400)
(h_under_40 : teachers_under_40 = 250)
(h_40_and_above : teachers_40_and_above = 150)
(h_sample_size : sample_size = 80)
: sample_teachers_under_40 sample_size 
  (proportion_teachers_under_40 total_teachers teachers_under_40) = 50 := by
sorry

end number_of_teachers_under_40_in_sample_l307_307155


namespace parallel_lines_distance_l307_307921

noncomputable def distance_between_lines (a b c d e f : ℝ) : ℝ :=
  abs (a * d + b * e + c) / sqrt (a^2 + b^2)

theorem parallel_lines_distance :
  ∀ (x y : ℝ), x - 2*y + 1 = 0 → 2*x - 4*y - 2 = 0 → distance_between_lines 1 (-2) 1 1 (-2) (-1) = (2*sqrt 5/5) :=
by
  sorry

end parallel_lines_distance_l307_307921


namespace angle_B_is_60_degrees_l307_307165

def cos_law_b_triangle (a b c : ℝ) (B : ℝ) : Prop :=
  a = 3 ∧ b = sqrt 7 ∧ c = 2 → cos B = (a^2 + c^2 - b^2) / (2 * a * c)

theorem angle_B_is_60_degrees : 
  ∀ a b c B, cos_law_b_triangle a b c B → B = 60 :=
by
  intros a b c B h
  have h1: cos B = (a^2 + c^2 - b^2) / (2 * a * c), from h.2.2
  have h2: a = 3, from h.1
  have h3: b = sqrt 7, from h.2.1
  have h4: c = 2, from h.2.2.1
  sorry

end angle_B_is_60_degrees_l307_307165


namespace slope_tangent_line_at_point_l307_307325

-- Conditions: circle's center is (2, -1) and point of tangency is (7, 3).
def center : ℝ × ℝ := (2, -1)
def point_of_tangency : ℝ × ℝ := (7, 3)

-- The negative reciprocal of the slope of the radius.
def slope_of_radius : ℝ := (point_of_tangency.2 - center.2) / (point_of_tangency.1 - center.1)
def slope_of_tangent_line : ℝ := -1 / slope_of_radius

-- The theorem to prove the slope of the tangent line is -5/4.
theorem slope_tangent_line_at_point : slope_of_tangent_line = -5 / 4 := by
  sorry

end slope_tangent_line_at_point_l307_307325


namespace convex_quad_area_le_half_sum_opposite_sides_l307_307629

variable (A B C D : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable {AB BC CD DA : ℝ}
variable {area_ABCD : ℝ}

/-- The area of any convex quadrilateral ABCD does not exceed the half-sum of the products of its opposite sides. -/
theorem convex_quad_area_le_half_sum_opposite_sides 
  (h : convex_quadrilateral A B C D)
  (h_area : area_ABCD = area_convex_quad A B C D) :
  area_ABCD ≤ 1/2 * (AB * CD + BC * AD) :=
sorry

end convex_quad_area_le_half_sum_opposite_sides_l307_307629


namespace find_a_l307_307886

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Condition: f(x+2) = 1/f(x) for any x
axiom condition1 : ∀ x : ℝ, f(x + 2) = 1 / f(x)

-- Condition: f(-5) = a
axiom condition2 : f (-5) = a

-- Theorem statement: a = 1/f(1)
theorem find_a : a = 1 / f 1 :=
by
  sorry

end find_a_l307_307886


namespace sqrt_a_b_l307_307879

theorem sqrt_a_b (a b : ℕ) (h₁ : a = ⌊real.sqrt 17⌋) (h₂ : b - 1 = real.sqrt 121) : real.sqrt (a + b) = 4 :=
  sorry

end sqrt_a_b_l307_307879


namespace modulus_of_z_l307_307477

-- Definitions of the problem conditions
def z := Complex.mk 1 (-1)

-- Statement of the math proof problem
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry -- Proof placeholder

end modulus_of_z_l307_307477


namespace round_to_nearest_hundred_l307_307997

-- Define the number x
def x : ℝ := 1278365.7422389

-- Define the rounded result y
def y : ℝ := 1278400

-- The statement that x rounded to the nearest hundred is y
theorem round_to_nearest_hundred : round_nearest_hundred x = y := 
begin
  sorry
end

end round_to_nearest_hundred_l307_307997


namespace total_candies_in_third_set_l307_307686

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l307_307686


namespace total_candies_third_set_l307_307703

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l307_307703


namespace five_digit_integers_divisible_by_12_count_l307_307519

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

def two_digit_multiples_of_12 : List ℕ :=
  [12, 24, 36, 48, 60, 72, 84, 96]

def count_five_digit_integers_divisible_by_12 : ℕ :=
  9 * 10 * 10 * 10 * 8

theorem five_digit_integers_divisible_by_12_count :
  (count_five_digit_integers_divisible_by_12 = 72000) :=
by
  rw [count_five_digit_integers_divisible_by_12]
  norm_num
  -- We skip the detailed proof steps here
  sorry

end five_digit_integers_divisible_by_12_count_l307_307519


namespace inequality_solution_l307_307493

noncomputable def f (x : ℝ) : ℝ := log (|x| + 1) + 2^x + 2^(-x)

theorem inequality_solution (x : ℝ) :
  f (x + 1) < f (2 * x) ↔ x < -1/3 ∨ x > 1 := 
sorry

end inequality_solution_l307_307493


namespace identity_of_letters_l307_307250

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l307_307250


namespace expression_defined_if_x_not_3_l307_307541

theorem expression_defined_if_x_not_3 (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end expression_defined_if_x_not_3_l307_307541


namespace equal_distances_of_arc_midpoint_and_circumcenter_l307_307939

namespace Geometry

-- Definitions of the geometric entities involved in the problem
variables {A B C D O H X : Point}

-- Conditions based on the problem statement
def acute_triangle_with_angle_45 (ABC : Triangle) : Prop :=
  (acute_triangle ABC) ∧ (angle A = 45)

def circumcenter_of_triangle (O : Point) (ABC : Triangle) : Prop :=
  is_circumcenter O ABC

def orthocenter_of_triangle (H : Point) (ABC : Triangle) : Prop :=
  is_orthocenter H ABC

def altitude_with_foot (B D : Point) (AC : Segment) : Prop :=
  is_foot_of_altitude D B AC

def midpoint_of_arc_ADH (X : Point) (ADH : Circle) : Prop :=
  is_midpoint_of_arc X ADH

-- Main theorem to prove DX = DO under given conditions
theorem equal_distances_of_arc_midpoint_and_circumcenter 
  (ABC : Triangle) (AC : Segment) (ADH : Circle)
  (h1 : acute_triangle_with_angle_45 ABC)
  (h2 : circumcenter_of_triangle O ABC)
  (h3 : orthocenter_of_triangle H ABC)
  (h4 : altitude_with_foot B D AC)
  (h5 : midpoint_of_arc_ADH X ADH) :
  dist D X = dist D O :=
sorry

end Geometry

end equal_distances_of_arc_midpoint_and_circumcenter_l307_307939


namespace sum_first_15_terms_l307_307144

-- Define the nth term of an arithmetic progression
def a_n (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic progression
def S_n (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

-- Given condition: the sum of the 4th term and the 12th term is 30
def cond (a d : ℤ) : Prop := a_n a d 4 + a_n a d 12 = 30

-- Theorem to prove: the sum of the first 15 terms is 225, given the condition
theorem sum_first_15_terms (a d : ℤ) (h : cond a d) : S_n a d 15 = 225 := 
sorry

end sum_first_15_terms_l307_307144


namespace train_speed_l307_307773

theorem train_speed (length_train length_platform : ℝ) (time : ℝ) 
  (h_length_train : length_train = 170.0416) 
  (h_length_platform : length_platform = 350) 
  (h_time : time = 26) : 
  (length_train + length_platform) / time * 3.6 = 72 :=
by 
  sorry

end train_speed_l307_307773


namespace wrens_below_twenty_percent_l307_307032

noncomputable def wrens_population_decreasing : ℕ → ℝ
| 0 := 1
| (n+1) := (wrens_population_decreasing n) / 2

theorem wrens_below_twenty_percent :
  ∃ n, wrens_population_decreasing (n + 3) < 0.2 :=
begin
  use 0,
  simp [wrens_population_decreasing],
  norm_num,
end

end wrens_below_twenty_percent_l307_307032


namespace constant_term_correct_l307_307644

-- Defining the problem: finding the constant term in the expansion of (x + 2/x + 1)^6
def constant_term_of_expansion : ℕ := by
  -- It is non-computable because the exact method of expansion and identification of terms is skipped
  noncomputable theory
  -- Defining the expansion function and answering the question directly
  let expr := (fun x : ℚ, (x + 2 / x + 1) ^ 6)
  -- Using sorry to skip directly to the answer for statement purpose
  sorry

theorem constant_term_correct :
  constant_term_of_expansion = 581 :=
by
  exact constant_term_of_expansion
  sorry

end constant_term_correct_l307_307644


namespace k_divides_99_l307_307783

theorem k_divides_99 (k : ℕ) :
  (∀ n : ℕ, n % k = 0 → reverse_digits n % k = 0) →
  coprime k 10 →
  99 % k = 0 :=
by
  sorry

end k_divides_99_l307_307783


namespace bowling_ball_weight_l307_307406

-- Define the weights of the bowling balls and canoes
variables (b c : ℝ)

-- Conditions provided by the problem statement
axiom eq1 : 8 * b = 4 * c
axiom eq2 : 3 * c = 108

-- Prove that one bowling ball weighs 18 pounds
theorem bowling_ball_weight : b = 18 :=
by
  sorry

end bowling_ball_weight_l307_307406


namespace shortest_distance_to_circle_l307_307324

theorem shortest_distance_to_circle :
  let P := (3, -4)
  let C := (7, 4)
  (dist (3, -4) (7, 4) = 4 * Real.sqrt 5) :=
by
  let dist := λ (p1 p2 : ℝ × ℝ), Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  have center : (x - 7)^2 + (y - 4)^2 = 0 -> (x, y) = (7, 4), from sorry
  sorry

end shortest_distance_to_circle_l307_307324


namespace stratified_sample_pines_l307_307350

theorem stratified_sample_pines (N P S : ℕ) (h1: N = 20000) (h2: P = 4000) (h3: S = 100) :
  (S * P) / N = 20 := by
  rw [h1, h2, h3]
  norm_num
  sorry

end stratified_sample_pines_l307_307350


namespace daily_water_intake_l307_307637

noncomputable def daily_water_consumption
    (x : ℕ)                  -- Number of 8-ounce servings Simeon used to drink
    (h8x : ℕ := 8 * x)       -- Total fluid ounces of water Simeon used to drink per day in 8-ounce servings
    (h16 : ℕ := 16 * (x - 4)) -- Total fluid ounces of water Simeon drinks now per day in 16-ounce servings
    (hw : h8x = h16)         -- Equation representing equal total water consumption in both cases
    : ℕ :=
64                           -- Total daily fluid ounces of water Simeon drinks

-- Statement to be proved: daily water consumption equals 64 fluid ounces
theorem daily_water_intake : 
∀ x : ℕ, 
let h8x := 8 * x           in 
let h16 := 16 * (x - 4)    in 
h8x = h16 → daily_water_consumption x = 64 :=
by
  intros x h8x h16 hw
  unfold daily_water_consumption
  exact sorry

end daily_water_intake_l307_307637


namespace polar_to_rectangular_l307_307830

open Real

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 8) (h_θ : θ = π / 4) :
    (r * cos θ, r * sin θ) = (4 * sqrt 2, 4 * sqrt 2) :=
by
  rw [h_r, h_θ]
  rw [cos_pi_div_four, sin_pi_div_four]
  norm_num
  field_simp [sqrt_eq_rpow]
  sorry

end polar_to_rectangular_l307_307830


namespace total_candies_in_third_set_l307_307682

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l307_307682


namespace tim_kittens_l307_307301

theorem tim_kittens (K : ℕ) (h1 : (3 / 5 : ℚ) * (2 / 3 : ℚ) * K = 12) : K = 30 :=
sorry

end tim_kittens_l307_307301


namespace prob_complement_union_l307_307154

-- Given conditions
variable {Ω : Type*} -- Universe of events
variable (A B : Ω → Prop) -- Events A and B
variable (P : ProbabilityMassFunction Ω) -- Probability mass function
variable (m n : ℝ) -- Probabilities of events A and B
variable (hA : P.event A = m) -- Given probability of event A
variable (hB : P.event B = n) -- Given probability of event B
variable (hAB : ∀ ω, ¬ (A ω ∧ B ω)) -- A and B are mutually exclusive events

-- Desired conclusion
theorem prob_complement_union :
  P.event (λ ω, ¬ (A ω ∨ B ω)) = 1 - m - n :=
sorry

end prob_complement_union_l307_307154


namespace unique_reconstruction_possible_l307_307213

def can_reconstruct_faces (a b c d e f : ℤ) : 
  Prop :=
  let edges := [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]
  in ∀ (sums : list ℤ), sums = edges →
    ∃ (a' b' c' d' e' f' : ℤ), a = a' ∧ b = b' ∧ c = c' ∧ d = d' ∧ e = e' ∧ f = f'

theorem unique_reconstruction_possible :
  ∀ (a b c d e f : ℤ),
    can_reconstruct_faces a b c d e f :=
begin
  sorry
end

end unique_reconstruction_possible_l307_307213


namespace arithmetic_mean_18_27_45_l307_307717

theorem arithmetic_mean_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_18_27_45_l307_307717


namespace Amy_balloons_l307_307581

-- Defining the conditions
def James_balloons : ℕ := 1222
def more_balloons : ℕ := 208

-- Defining Amy's balloons as a proof goal
theorem Amy_balloons : ∀ (Amy_balloons : ℕ), James_balloons - more_balloons = Amy_balloons → Amy_balloons = 1014 :=
by
  intros Amy_balloons h
  sorry

end Amy_balloons_l307_307581


namespace common_tangent_ln_x_ax2_l307_307478

theorem common_tangent_ln_x_ax2 (a : ℝ) (h : 0 < a) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ a > 0 ∧ ∃ m b, 
    (∀ x, ln x1 + (m * (x - x1)) = ln x1 + (1/x1 * (x - x1))) ∧ 
    (y = a * x1^2 + 2 * a * x1 * (x - x1))) ↔ a ∈ Ici (1/(2*exp(1))) :=
sorry

end common_tangent_ln_x_ax2_l307_307478


namespace parabola_intersection_sum_l307_307015

theorem parabola_intersection_sum :
  let P1 := fun x => (x - 2) ^ 2
  let P2 := fun y => (y + 1) ^ 2 - 6
  let X := { (x, y) : ℝ × ℝ | y = P1 x ∧ x = P2 y }
  ∑ p in X, (p.1 + p.2) = 6 :=
by
  sorry

end parabola_intersection_sum_l307_307015


namespace arithmetic_mean_18_27_45_l307_307718

theorem arithmetic_mean_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_18_27_45_l307_307718


namespace matrix_projection_l307_307420

open Matrix

noncomputable def Q : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![1/14, 3/14, -2/14],
    ![3/14, 9/14, -6/14],
    ![-2/14, -6/14, 4/14]
  ]

def u : Vector ℚ 3 :=
  ![1, 3, -2]

theorem matrix_projection (w : Vector ℚ 3) :
  (Q.mulVec w) = ((w ⬝ u) / (u ⬝ u) : ℚ) • u :=
by {
  sorry
}

end matrix_projection_l307_307420


namespace sum_of_b_values_l307_307020

noncomputable def f (b : ℚ) (x : ℚ) : ℚ := b / (3 * x - 4)

theorem sum_of_b_values (b : ℚ) (h1 : f b 3 = f⁻¹ (b + 2)) : b = 8 ∨ b = -5/3 → b ∈ {8, -5/3} →
  b = 8 + (-5/3) → ∑ b ∈ {8, -5/3}, b = 19/3 :=
by
  -- We would undertake proving here, but for now, we skip the proof.
  sorry

end sum_of_b_values_l307_307020


namespace distance_from_F_to_midpoint_DE_l307_307937

-- Conditions
variables (DE DF EF : ℝ) (is_right_triangle : DE^2 = DF^2 + EF^2) (DE_eq : DE = 15) (DF_eq : DF = 9) (EF_eq : EF = 12)

-- The problem statement
theorem distance_from_F_to_midpoint_DE : (distance (F : ℝ × ℝ) (midpoint (DE : ℝ × ℝ) (D : ℝ × ℝ) E) = 7.5) :=
by sorry

end distance_from_F_to_midpoint_DE_l307_307937


namespace number_of_middle_schools_in_sample_correct_l307_307766

-- Conditions
def total_schools : ℕ := 700
def universities : ℕ := 20
def middle_schools : ℕ := 200
def primary_schools : ℕ := 480
def sample_size : ℕ := 70

-- Calculation of the number of middle schools in the sample
def proportion_middle_schools : ℚ := middle_schools / total_schools
def middle_schools_in_sample : ℕ := (sample_size : ℚ) * proportion_middle_schools

-- Proof statement
theorem number_of_middle_schools_in_sample_correct :
  middle_schools_in_sample = 20 :=
sorry

end number_of_middle_schools_in_sample_correct_l307_307766


namespace athlete_target_heart_rate_l307_307376

noncomputable def target_heart_rate (age : ℕ) : ℕ :=
  let max_heart_rate := 230 - age
  let target_heart_rate := 0.85 * max_heart_rate
  Nat.round target_heart_rate

theorem athlete_target_heart_rate : target_heart_rate 40 = 162 := by
  sorry

end athlete_target_heart_rate_l307_307376


namespace solution_mixing_l307_307333

/-- 
Proof that adding 150 milliliters of solution y (30% alcohol by volume) 
to 50 milliliters of solution x (10% alcohol by volume) results 
in a solution that is 25% alcohol by volume.
-/
theorem solution_mixing (x y : ℕ) (hx : x = 50) (hy : y = 150) :
  ((0.10 * x + 0.30 * y) / (x + y)) = 0.25 :=
by
  rw [hx, hy]
  norm_num
  sorry

end solution_mixing_l307_307333


namespace find_value_of_a_plus_b_l307_307080

variables (a b : ℝ)

theorem find_value_of_a_plus_b
  (h1 : a^3 - 3 * a^2 + 5 * a = 1)
  (h2 : b^3 - 3 * b^2 + 5 * b = 5) :
  a + b = 2 := 
sorry

end find_value_of_a_plus_b_l307_307080


namespace one_over_98_squared_sum_l307_307384

noncomputable def repeating_decimal_sum (d : ℕ) (k : ℕ) : ℕ :=
  -- This function calculates the sum of digits in the repeating decimal representation
  -- of 1 / (d^k). For this task, specifics of how to compute this are not needed.
  -- It is left noncomputable for simplicity.
  sorry

theorem one_over_98_squared_sum :
  repeating_decimal_sum 98 2 = 882 :=
begin
  -- Proof is omitted.
  sorry
end

end one_over_98_squared_sum_l307_307384


namespace sum_first_10_terms_l307_307498

def sequence (n : ℕ) : ℕ → ℝ
| 0       := 2
| (n + 1) := 0.5 * sequence n + 0.5

theorem sum_first_10_terms :
  (finset.range 10).sum (λ n, 1 / (sequence n - 1)) = 1023 := 
sorry

end sum_first_10_terms_l307_307498


namespace initial_quarters_l307_307620

-- Define the value of each type of coin
def dime_value : ℝ := 0.10
def nickel_value : ℝ := 0.05
def quarter_value : ℝ := 0.25

-- Define the number of each type of coin Maria has initially
def num_dimes : ℕ := 4
def num_nickels : ℕ := 7

-- Define the number of quarters Maria's mom gives her
def num_quarters_given : ℕ := 5

-- Define the total amount of money Maria has after receiving the quarters
def total_amount : ℝ := 3.00

-- Prove the number of quarters Maria had initially
theorem initial_quarters :
  let initial_quarters_value := total_amount
                              - (num_dimes * dime_value)
                              - (num_nickels * nickel_value)
                              - (num_quarters_given * quarter_value) in
  initial_quarters_value / quarter_value = 4 := by
  sorry

end initial_quarters_l307_307620


namespace find_lambda_l307_307454

section

variables {Point : Type} [MetricSpace Point] [VectorSpace ℝ Point] (A B C P Q : Point)

axiom midpoint_AC : (P + P) = (A + C)
axiom midpoint_AB : ((Q + Q) + B + C) = (B + C)
axiom dist_PQ_BC_eq : (dist P Q) = λ * (dist B C)

theorem find_lambda (λ : ℝ) (h_mid_AC : midpoint_AC) (h_mid_AB : midpoint_AB) (h_dist_PQ_BC : dist_PQ_BC_eq λ) : 
  λ = (1 / 2) :=
sorry

end

end find_lambda_l307_307454


namespace measure_15_minutes_l307_307116

/--
Using a 7-minute and an 11-minute hourglass, it is possible to measure exactly 15 minutes.
-/
theorem measure_15_minutes (HG_7 HG_11 : ℕ) (h7 : HG_7 = 7) (h11 : HG_11 = 11) : ∃ t : ℕ, t = 15 :=
by
  use 15
  sorry

end measure_15_minutes_l307_307116


namespace solution_set_f_lt_0_l307_307470

variable {f : ℝ → ℝ}

-- Definitions that directly appear in conditions
def domain_f : Set ℝ := {x | x > 0}

def derivative_f_exists : Prop := ∀ (x : ℝ), x ∈ domain_f → DifferentiableAt ℝ f x

def derivative_condition : Prop := ∀ (x : ℝ), x ∈ domain_f → x * (deriv f x) > f x

def f_at_2 : Prop := f 2 = 0

-- The theorem to be proven
theorem solution_set_f_lt_0 (hf : derivative_f_exists) (hcond : derivative_condition) (hf2 : f_at_2) :
  {x : ℝ | x > 0 ∧ f x < 0} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end solution_set_f_lt_0_l307_307470


namespace four_digit_integers_with_4_or_5_l307_307118

theorem four_digit_integers_with_4_or_5 : 
  (finset.range 10000).filter (λ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ d, d ∈ [4, 5] ∧ d ∈ n.digits 10)).card = 5416 :=
sorry

end four_digit_integers_with_4_or_5_l307_307118


namespace remaining_integers_in_T_l307_307286

def T := {n : ℕ | 1 ≤ n ∧ n ≤ 60}

def multiples_of_4 := {n : ℕ | 1 ≤ n ∧ n ≤ 60 ∧ n % 4 = 0}

def multiples_of_5 := {n : ℕ | 1 ≤ n ∧ n ≤ 60 ∧ n % 5 = 0}

def multiples_of_20 := {n : ℕ | 1 ≤ n ∧ n ≤ 60 ∧ n % 20 = 0}

theorem remaining_integers_in_T :
  (60 - ((multiples_of_4.card + multiples_of_5.card) - multiples_of_20.card)) = 36 :=
by
  sorry

end remaining_integers_in_T_l307_307286


namespace trapezoid_area_l307_307571

theorem trapezoid_area (A B : ℝ) (n : ℕ) (hA : A = 36) (hB : B = 4) (hn : n = 6) :
    (A - B) / n = 5.33 := 
by 
  -- Given conditions and the goal
  sorry

end trapezoid_area_l307_307571


namespace triangle_area_and_point_check_l307_307795

noncomputable def is_point_in_triangle (x y : ℝ) : Prop := 
  (0 ≤ x) ∧ 
  (0 ≤ y) ∧ 
  (3 * x + y ≤ 9)

theorem triangle_area_and_point_check :
  let area := (3 : ℝ) * (9 : ℝ) / 2 in
  area = 27 / 2 ∧ ¬ is_point_in_triangle 1 1 :=
by
  sorry

end triangle_area_and_point_check_l307_307795


namespace gcd_possible_values_count_l307_307726

theorem gcd_possible_values_count :
  ∃ (a b : ℕ), a * b = 72 ∧ 
              (∃ (m n p q : ℕ), a = 2^m * 3^n ∧ b = 2^p * 3^q ∧ 
                               finset.card (finset.image (λ (x y : ℕ), gcd x y) 
                                                      (finset.product (finset.range 4) (finset.range 3))) = 10) :=
by
  sorry

end gcd_possible_values_count_l307_307726


namespace cross_product_zero_l307_307598

open Matrix

noncomputable def vector := Matrix (Fin 3) (Fin 1) ℝ
noncomputable def v_cross_w : vector := ![\[7], \[-3], \[5]]
noncomputable def u : vector := ![\[1], \[2], \[-1]]

theorem cross_product_zero (v w : vector):
  (v + w + u) ⬝ ((v + w + u) ⬝ ![![0, -1, 1], ![1, 0, -1], ![-1, 1, 0]]) = 0 :=
by
  sorry

end cross_product_zero_l307_307598


namespace evaluate_expression_l307_307918

theorem evaluate_expression (m n : ℝ) (h : m - n = 2) :
  (2 * m^2 - 4 * m * n + 2 * n^2 - 1) = 7 := by
  sorry

end evaluate_expression_l307_307918


namespace find_annual_income_l307_307291

theorem find_annual_income (p : ℝ) (A : ℝ) (h : 0) :
  0.01 * p * 35000 + 0.01 * (p + 3) * (A - 35000) = 0.01 * (p + 0.5) * A → A = 42000 :=
sorry

end find_annual_income_l307_307291


namespace unique_face_numbers_l307_307216

-- Define the problem statement and conditions
theorem unique_face_numbers (a b c d e f : ℤ) (sums : list ℤ) (h : sums = [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]) : 
  (∃ (n : ℕ → ℤ), (n 0 = a ∧ n 1 = b ∧ n 2 = c ∧ n 3 = d ∧ n 4 = e ∧ n 5 = f)) :=
by 
  rw h
  -- Additional detailed steps are omitted
  sorry

end unique_face_numbers_l307_307216


namespace max_value_sqrt_expr_l307_307593

theorem max_value_sqrt_expr (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) (h5 : 0 ≤ c) (h6 : c ≤ 1) (h7 : 0 ≤ d) (h8 : d ≤ 1) :
  (∛√ (abcd) + ∛√ ((1-a)*(1-b)*(1-c)*(1-d))) ≤ 1 := sorry

end max_value_sqrt_expr_l307_307593


namespace identify_letters_l307_307255

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l307_307255


namespace log_comparison_l307_307727

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := 
sorry

end log_comparison_l307_307727


namespace ratio_DO_OP_l307_307627

variable {A B C D P Q O : Type*}
variable [AddGroup (A → ℝ)]
variable [AddGroup (B → ℝ)]
variable [AddGroup (C → ℝ)]
variable [AddGroup (D → ℝ)]
variable [AddGroup (P → ℝ)]
variable [AddGroup (Q → ℝ)]
variable [AddGroup (O → ℝ)]

-- Conditions: parallelogram, positioning of points, and given segment ratios
variable (parallelogram_ABCD : IsParallelogram A B C D)
variable (on_AB_P : IsOnLineSegment A B P)
variable (on_BC_Q : IsOnLineSegment B C Q)
variable (ratio_AB_BP : 3 * distance A B = 7 * distance B P)
variable (ratio_BC_BQ : 3 * distance B C = 4 * distance B Q)

-- Conclude the ratio DO : OP is 7 : 3
theorem ratio_DO_OP : distance D O / distance O P = 7 / 3 := sorry

end ratio_DO_OP_l307_307627


namespace prime_implies_power_of_three_l307_307902

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 7 ∧ ∀ n, n ≥ 2 → a n ^ 2 + 5 = a (n - 1) * a (n + 1)

def is_prime (p : ℕ) : Prop :=
  p ≥ 2 ∧ ∀ m, m ≥ 2 → m ∣ p → m = p

theorem prime_implies_power_of_three {a : ℕ → ℕ} (H : sequence a) (n : ℕ) :
  is_prime (a n + (-1) ^ n) → ∃ m, n = 3 ^ m :=
by
  sorry

end prime_implies_power_of_three_l307_307902


namespace planter_cost_l307_307307

-- Define costs
def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny : ℝ := 4.00
def cost_geranium : ℝ := 3.50

-- Define quantities
def num_creeping_jennies : ℝ := 4
def num_geraniums : ℝ := 4
def num_corners : ℝ := 4

-- Define the total cost
def total_cost : ℝ :=
  (cost_palm_fern
   + (cost_creeping_jenny * num_creeping_jennies)
   + (cost_geranium * num_geraniums))
  * num_corners

-- Prove the total cost is $180.00
theorem planter_cost : total_cost = 180.00 :=
by
  sorry

end planter_cost_l307_307307


namespace range_of_a_l307_307487

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x then log a x else abs (x + 3)

theorem range_of_a (a : ℝ) : 
  (0 < a ∧ a ≠ 1) ↔ 
  (∃ x : ℝ, 0 < x ∧ x < 3 ∧ log a x = abs (x - 3)) :=
by
  sorry

end range_of_a_l307_307487


namespace simplify_abs1_simplify_abs2_simplify_abs3_l307_307638

-- Proof for simplification of |2x - 1|
theorem simplify_abs1 (x : ℝ) : 
  (x ≥ 1/2 → |2*x - 1| = 2*x - 1) ∧ 
  (x < 1/2 → |2*x - 1| = 1 - 2*x) :=
by
  sorry

-- Proof for simplification of |x - 1| + |x - 3|
theorem simplify_abs2 (x : ℝ) : 
  (x < 1 → |x - 1| + |x - 3| = 4 - 2 * x) ∧ 
  (1 ≤ x ∧ x < 3 → |x - 1| + |x - 3| = 2) ∧ 
  (x ≥ 3 → |x - 1| + |x - 3| = 2 * x - 4) :=
by
  sorry

-- Proof for simplification of ||x-1|-2| + |x+1|
theorem simplify_abs3 (x : ℝ) : 
  (x ≥ 3 → ||x - 1| - 2| + |x + 1| = 2 * x - 2) ∧ 
  (1 ≤ x ∧ x < 3 → ||x - 1| - 2| + |x + 1| = 4) ∧ 
  (-1 ≤ x ∧ x < 1 → ||x - 1| - 2| + |x + 1| = 2 * x + 2) ∧
  (x < -1 → ||x - 1| - 2| + |x + 1| = -2 * x - 2) :=
by
  sorry

end simplify_abs1_simplify_abs2_simplify_abs3_l307_307638


namespace combinatorial_identity_example_l307_307745

theorem combinatorial_identity_example :
  nat.choose 12 5 + nat.choose 12 6 = nat.choose 13 6 :=
by
  sorry

end combinatorial_identity_example_l307_307745


namespace identity_of_letters_l307_307248

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l307_307248


namespace part_I_solution_part_II_solution_l307_307106

def f (x a : ℝ) : ℝ := abs (2 * x + a) + abs (x - 1)

theorem part_I_solution (a : ℝ) (h : a = 3) :
  { x : ℝ | f x a < 6 } = set.Ioo (-8/3 : ℝ) (4/3 : ℝ) := sorry

theorem part_II_solution (a : ℝ) :
  (∀ x : ℝ, f x a + f (-x) a ≥ 5) ↔ (a ≤ -3/2 ∨ a ≥ 3/2) := sorry

end part_I_solution_part_II_solution_l307_307106


namespace cars_meet_second_time_at_30_minutes_l307_307762

theorem cars_meet_second_time_at_30_minutes :
  let time_car1 := 150 / 60
  let time_car2 := 60 / 40
  let lcm_time := Nat.lcm (Real.to_rat time_car1).denom (Real.to_rat time_car2).denom
  lcm_time / 60 = 30 := by
  let time_car1 := 2.5
  let time_car2 := 1.5
  have h1 : time_car1 = 150 / 60 := by sorry
  have h2 : time_car2 = 60 / 40 := by sorry
  have h3 : lcm (2.5, 1.5)= 15 := by sorry
  have h4 : lcm_time = 15 := by sorry
  have h5 : lcm_time * 2 = 30 := by sorry
  show 30 := by sorry

end cars_meet_second_time_at_30_minutes_l307_307762


namespace number_of_remaining_grandchildren_l307_307507

-- Defining the given values and conditions
def total_amount : ℕ := 124600
def half_amount : ℕ := total_amount / 2
def amount_per_remaining_grandchild : ℕ := 6230

-- Defining the goal to prove the number of remaining grandchildren
theorem number_of_remaining_grandchildren : (half_amount / amount_per_remaining_grandchild) = 10 := by
  sorry

end number_of_remaining_grandchildren_l307_307507


namespace total_candies_in_third_set_l307_307684

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l307_307684


namespace toby_friends_girls_count_l307_307751

noncomputable def percentage_of_boys : ℚ := 55 / 100
noncomputable def boys_count : ℕ := 33
noncomputable def total_friends : ℚ := boys_count / percentage_of_boys
noncomputable def percentage_of_girls : ℚ := 1 - percentage_of_boys
noncomputable def girls_count : ℚ := percentage_of_girls * total_friends

theorem toby_friends_girls_count : girls_count = 27 := by
  sorry

end toby_friends_girls_count_l307_307751


namespace sum_of_x_values_l307_307665

def sum_of_values (x : ℚ) : Prop := 
  (x - 3) * (x + 4) = 3 * (x - 2) * (x - 2)

theorem sum_of_x_values : 
  let s := {x : ℚ | sum_of_values x} in
  ∑ x in s.to_finset, x = 19/2 :=
sorry

end sum_of_x_values_l307_307665


namespace original_divisor_in_terms_of_Y_l307_307866

variables (N D Y : ℤ)
variables (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4)

theorem original_divisor_in_terms_of_Y (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4) : 
  D = (2 * Y - 3) / 15 :=
sorry

end original_divisor_in_terms_of_Y_l307_307866


namespace intervals_bound_l307_307608

variables {n : ℕ} (N : ℕ) (A : Finset (Fin n)) (A_i : Fin n → Finset (Fin n))

-- Condition: n ≥ 2
axiom h_n : n ≥ 2

-- Condition: Definition of interval
def interval (A : Finset (Fin n)) : Prop :=
  ∃ a b : Fin n, a.1 < b.1 ∧ A = Finset.Icc a b

-- Condition: A₁, ..., Aₙ are subsets such that A_i ∩ A_j is an interval
axiom h_intervals :
  ∀ {i j : Fin N}, i ≠ j → interval (A_i i ∩ A_i j)

-- Problem Statement: N ≤ ⌊ n^2 / 4 ⌋
theorem intervals_bound :
  N ≤ n * n / 4 :=
sorry

end intervals_bound_l307_307608


namespace f_f_neg1_eq_45_l307_307864

def f (x : ℝ) : ℝ := if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem f_f_neg1_eq_45 : f (f (-1)) = 45 := by
  sorry

end f_f_neg1_eq_45_l307_307864


namespace product_xy_l307_307987

noncomputable def positive_integers := {n : ℕ // 0 < n}

theorem product_xy (x y : positive_integers) 
  (h1 : ∃ p q : ℕ, p + q + p^2 + q^2 = 50 
        ∧ x.val = 10^(p^2) 
        ∧ y.val = 10^(q^2) 
        ∧ ∀ z : ℕ, z = sqrt(log x.val) ∨ z = sqrt(log y.val) ∨ z = log(sqrt x.val) ∨ z = log(sqrt y.val)) 
  : x.val * y.val = 10^41 :=
sorry

end product_xy_l307_307987


namespace angle_ABC_is_50_l307_307166

theorem angle_ABC_is_50 {A B C N M : Type} [PlaneGeom A B C N M]
  (h1 : ∠ A C B = 40)
  (h2 : midpoint N A B)
  (h3 : midpoint M A C)
  (h4 : distance N M = distance B C) :
  ∠ A B C = 50 := 
sorry

end angle_ABC_is_50_l307_307166


namespace can_obtain_number_l307_307626

-- Define the condition for generating a new number on the blackboard
def generates (a b c : ℕ) : Prop :=
  c = a * b + a + b

-- Define what it means for a number to be obtained on the blackboard
def can_be_written (n : ℕ) : Prop :=
  ∃ f : ℕ → ℕ → ℕ, -- f is the function following the generate rule
    (f 1 2 = n) ∨
    (∃ a b : ℕ, f a b = n ∧ can_be_written a ∧ can_be_written b)
    
-- Theorem we want to prove
theorem can_obtain_number (n : ℕ) : can_be_written n ↔ (n = 13121 ∨ n ≠ 12131) := by
  sorry

end can_obtain_number_l307_307626


namespace ratio_implies_sum_ratio_l307_307531

theorem ratio_implies_sum_ratio (x y : ℝ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_implies_sum_ratio_l307_307531


namespace sum_of_integers_satisfying_inequality_l307_307052

theorem sum_of_integers_satisfying_inequality :
  let n_set := {n : ℕ | 1.3 * n - 5 < 7 ∧ n > 0} in
  ∑ n in n_set, n = 45 :=
by
  sorry

end sum_of_integers_satisfying_inequality_l307_307052


namespace largest_n_divides_factorial_l307_307041

theorem largest_n_divides_factorial (n : ℕ) : 
  (18^(n:Int)) ∣ factorial 30 ↔ n <= 7 := 
sorry

end largest_n_divides_factorial_l307_307041


namespace min_value_of_2a7_a11_l307_307881

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the sequence terms

-- Conditions
axiom geometric_sequence (n m : ℕ) (r : ℝ) (h : ∀ k, a k > 0) : a n = a 0 * r^n
axiom geometric_mean_condition : a 4 * a 14 = 8

-- Theorem to Prove
theorem min_value_of_2a7_a11 : ∀ n : ℕ, (∀ k, a k > 0) → 2 * a 7 + a 11 ≥ 8 :=
by
  intros
  sorry

end min_value_of_2a7_a11_l307_307881


namespace sum_of_recip_sqrt_radii_l307_307769

open Real

-- Legend for the conditions
structure Circle where
  radius : ℝ
  tangent_point : ℝ

def L0 : List Circle := 
  [ { radius := 50^2, tangent_point := 50 }, { radius := 55^2, tangent_point := 55 } ]

def recursive_circle (C1 C2 : Circle) : Circle :=
  { radius := (C1.radius * C2.radius) / ((sqrt C1.radius + sqrt C2.radius)^2)
  , tangent_point := (C1.tangent_point + C2.tangent_point) / 2 }

def next_layer (L : List Circle) : List Circle :=
  List.init L.zip (List.tail L) (λ ⟨C1, C2⟩, recursive_circle C1 C2)

def L : ℕ → List Circle
| 0       := L0
| (k + 1) := next_layer (Array.fold (List.range k) (λ acc x, acc ++ L x)) []

def S : List Circle := Array.fold (List.range 8) (λ acc x, acc ++ L x) []

def sum_recip_sqrt_radii (S : List Circle) : ℝ :=
  List.sum (List.map (λ C, 1 / sqrt C.radius) S)

-- Lean statement for the given problem
theorem sum_of_recip_sqrt_radii : 
  sum_recip_sqrt_radii S = 2688 / 550 :=
sorry

end sum_of_recip_sqrt_radii_l307_307769


namespace jean_total_jail_time_l307_307173

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l307_307173


namespace general_term_formula_l307_307663

-- Defining the sequence according to the problem statement
def seq (a : ℕ → ℕ) : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := let an := a (n + 2) in
           let a_n1 := a (n + 1) in
           let a_n2 := a n in
           if h : (a n ≠ 0) ∧ (a (n + 1) ≠ 0) then
             2 * (a (n + 1)) +
             (a (n + 1) * (a n)) -
             (a (n + 1) * a n)
           else
             1 -- The problem states positivity, thus this should not occur

-- Conjecturing the general term formula
theorem general_term_formula :
  ∀ n : ℕ, (seq (λ k, ∏ i in finset.range (k + 1), (2^i - 1)^2) n) = (λ k, ∏ i in finset.range (k + 1), (2^i - 1)^2) n :=
begin
  sorry
end

end general_term_formula_l307_307663


namespace like_terms_exponents_l307_307916

theorem like_terms_exponents (m n : ℤ) (x y : ℝ) :
  (∀ x y, -3 * x ^ m * y ^ 3 = 2 * x ^ 4 * y ^ n) → m - n = 1 :=
by
  sorry

end like_terms_exponents_l307_307916


namespace parabola_focus_ellipse_focus_l307_307544

theorem parabola_focus_ellipse_focus (p : ℝ) :
  (let ellipse_focus := (2, 0)
   in y^2 = 2 * p * x ∧ x^2 / 6 + y^2 / 2 = 1) →
  p = 4 :=
by
  sorry

end parabola_focus_ellipse_focus_l307_307544


namespace total_candies_in_third_set_l307_307683

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l307_307683


namespace total_num_players_l307_307298

-- Definitions of the problem conditions
def num_throwers : ℕ := 40
def total_right_handed : ℕ := 60
def left_handed_fraction : ℚ := 1 / 3

-- Statement of the problem:
theorem total_num_players : ∃ T : ℕ, T = 70 :=
by
  -- define number of non-throwers (N) and equations used
  let N := 20 * (3/2) -- based on the given 20 right-handed non-throwers, solve N
  let num_players := num_throwers + N.to_nat
  existsi num_players
  sorry

end total_num_players_l307_307298


namespace floor_sum_even_l307_307953

theorem floor_sum_even (a b c : ℕ) (h1 : a^2 + b^2 + 1 = c^2) : 
    ((a / 2) + (c / 2)) % 2 = 0 := 
  sorry

end floor_sum_even_l307_307953


namespace proof_problem_l307_307534

-- Given conditions for the problem
variables {X Y : Type} [Inhabited X] [Inhabited Y]
variables (f : X → Y) (g : X → Y)
variable [IsInverse (Function.LeftInverse) (f)]
variable [IsInverse (Function.LeftInverse) (g)]

-- Definition of the conditions in Lean 4
def condition1 : Prop := ∀ x, (f ∘ g⁻¹) (x) = x^2 - 1
def condition2 : Prop := Function.Bijective g

-- The proof goal.
theorem proof_problem (h1 : condition1) (h2 : condition2) : 
  g⁻¹ (f 8) = 3 ∨ g⁻¹ (f 8) = -3 :=
sorry

end proof_problem_l307_307534


namespace inf_sum_calc_l307_307410

noncomputable def infinite_sum : ℕ → ℝ 
  | n := (n : ℝ) / (n^4 + 1 : ℝ)

theorem inf_sum_calc : 
  (∑' n, infinite_sum n) = 1 :=
sorry

end inf_sum_calc_l307_307410


namespace max_single_salary_l307_307781

-- Define the constants and conditions
def num_players : ℕ := 18
def min_salary : ℕ := 20000
def total_cap : ℕ := 600000

-- Theorem stating the maximum salary for a single player given the conditions
theorem max_single_salary : 
  (∃ (p : ℕ → ℕ), 
    (∀ i, i < num_players → min_salary ≤ p i) ∧ 
    (∑ i in Finset.range num_players, p i <= total_cap) ∧ 
    (∃ j, j < num_players ∧ p j = 260000)) :=
by
  -- skipping the proof
  sorry

end max_single_salary_l307_307781


namespace complex_number_real_condition_l307_307463

theorem complex_number_real_condition (a : ℝ) (h : ∃ (z : ℤ), (1 + a * complex.I) * (2 - complex.I) = z) : a = 1 / 2 :=
by
  sorry

end complex_number_real_condition_l307_307463


namespace sum_of_possible_values_of_d_l307_307341

theorem sum_of_possible_values_of_d (d : ℕ) : 
  (∃ n : ℕ, 512 ≤ n ∧ n < 4096 ∧ d = Nat.logBase 2 n + 1) → 
  d = 10 ∨ d = 11 ∨ d = 12 → 
  ∑ x in {10, 11, 12}.toFinset, x = 33 :=
by 
  sorry

end sum_of_possible_values_of_d_l307_307341


namespace solve_f_x_leq_f_1_l307_307613

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4 * x + 6 else x + 6

theorem solve_f_x_leq_f_1 : { x : ℝ | f x ≤ f 1 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | 1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end solve_f_x_leq_f_1_l307_307613


namespace find_complex_number_find_modulus_l307_307066

variable (b : ℝ)
variable (b_pos : b > 0)
variable (z : ℂ)
variable (hz : z = 1 + b * Complex.i)
variable (h_pure_imaginary : (z - 2) ^ 2 = Complex.im ((z - 2) ^ 2) * Complex.i)

/-- 
Proof problem:
Given a complex number z = 1 + b * i (where b is a positive real number), 
and (z - 2)^2 is a pure imaginary number. 
Prove that z = 1 + i. 
--/
theorem find_complex_number : z = 1 + Complex.i := 
by 
  -- proof will go here 
  sorry

variable (ω : ℂ)

/-- 
Proof problem:
Given z = 1 + i, and ω = z / (2 + i),
Prove that the modulus |ω| of the complex number ω is sqrt(10) / 5.
--/
theorem find_modulus (hz : z = 1 + Complex.i) (hω : ω = z / (2 + Complex.i)) : ω.abs = Real.sqrt 10 / 5 := 
by 
  -- proof will go here 
  sorry

end find_complex_number_find_modulus_l307_307066


namespace solve_x_given_y_l307_307536

theorem solve_x_given_y (x : ℝ) (h : 2 = 2 / (5 * x + 3)) : x = -2 / 5 :=
sorry

end solve_x_given_y_l307_307536


namespace first_discount_percentage_l307_307282

theorem first_discount_percentage (x : ℕ) :
  let original_price := 175
  let discounted_price := original_price * (100 - x) / 100
  let final_price := discounted_price * 95 / 100
  final_price = 133 → x = 20 :=
by
  sorry

end first_discount_percentage_l307_307282


namespace problem_proof_l307_307448

open Nat

def arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (λ : ℕ) : Prop :=
  (a 3 = 3) ∧ (∀ n, S n = (n * (n + 1)) / 2) ∧ (∀ n, λ * S n = a n * a (n + 1))

def geometric_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  (b 1 = 2 * 2) ∧ (b 3 = a 15 + 1) ∧
  (∀ n, b n = 2^(n+1) ∨ b n = (-2)^(n+1))

def sequence_cn (c : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℝ) : Prop :=
  (∀ n, c n = 2 / (n * (n + 2))) ∧
  (∀ n, T n = (3 / 2) - (2 * n + 3) / (n^2 + 3 * n + 2)) ∧ 
  (∀ n, (S n + n / 2) * c n = 1)

theorem problem_proof :
  ∃ (a b c : ℕ → ℕ) (S T : ℕ → ℝ) (λ : ℕ),
    arithmetic_sequence a S λ ∧ 
    geometric_sequence b a ∧ 
    sequence_cn c S T :=
by
  sorry

end problem_proof_l307_307448


namespace remainder_of_3_pow_20_mod_7_l307_307323

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end remainder_of_3_pow_20_mod_7_l307_307323


namespace f_decreasing_on_0_1_l307_307834

noncomputable def f : ℝ → ℝ := λ x, x + (1 / x)

theorem f_decreasing_on_0_1 : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 := by
  sorry

end f_decreasing_on_0_1_l307_307834


namespace fruit_bowl_oranges_l307_307150

theorem fruit_bowl_oranges :
  ∀ (bananas apples oranges : ℕ),
    bananas = 2 →
    apples = 2 * bananas →
    bananas + apples + oranges = 12 →
    oranges = 6 :=
by
  intros bananas apples oranges h1 h2 h3
  sorry

end fruit_bowl_oranges_l307_307150


namespace altitude_inequality_l307_307951

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables {a b c h_a : ℝ}

theorem altitude_inequality (h : h_a = (dist A B) * (dist A C) / (dist B C)) :
  (b + c)^2 ≥ a^2 + 4 * h_a^2 :=
sorry

end altitude_inequality_l307_307951


namespace no_upper_bound_l307_307367

-- Given Conditions
variables {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {M : ℝ}

-- Condition: widths and lengths of plates are 1 and a1, a2, a3, ..., respectively
axiom width_1 : ∀ n, (S n > 0)

-- Condition: a1 ≠ 1
axiom a1_neq_1 : a 1 ≠ 1

-- Condition: plates are similar but not congruent starting from the second
axiom similar_not_congruent : ∀ n > 1, (a (n+1) > a n)

-- Condition: S_n denotes the length covered after placing n plates
axiom Sn_length : ∀ n, S (n+1) = S n + a (n+1)

-- Condition: a_{n+1} = 1 / S_n
axiom an_reciprocal : ∀ n, a (n+1) = 1 / S n

-- The final goal: no such real number exists that S_n does not exceed
theorem no_upper_bound : ∀ M : ℝ, ∃ n : ℕ, S n > M := 
sorry

end no_upper_bound_l307_307367


namespace determine_x_l307_307396

namespace Proof

def f (x : ℝ) : ℝ :=
  if x < 0 then (x + 1) ^ 2 else 2 * Real.cos x

theorem determine_x (x : ℝ) (hx : x ∈ Set.Icc 0 (2 * Real.pi)) (h : f x = 1) :
  x = -2 ∨ x = Real.pi / 3 ∨ x = 5 * Real.pi / 3 := by
  sorry

end Proof

end determine_x_l307_307396


namespace convert_to_cylindrical_coords_l307_307394

theorem convert_to_cylindrical_coords : 
  let x := -3
  let y := -3 * Real.sqrt 3
  let z := 2
  let r := Real.sqrt ((x:ℝ) ^ 2 + (y:ℝ) ^ 2)
  let θ := Real.acos (x / r)
  (r > 0) ∧ (0 ≤ θ) ∧ (θ < 2 * Real.pi) ∧
  ((Int.ofReal θ = 4 * Real.pi / 3) ∧ 
  (Real.sqrt x ^ 2 + y ^ 2 = 36) ∧ 
  (r = 6) ∧ 
  (z = 2)) := by
  sorry

end convert_to_cylindrical_coords_l307_307394


namespace maximum_students_l307_307378

theorem maximum_students (n : ℕ) :
  (∀ (s1 s2 : fin n → fin 3), s1 ≠ s2 →
    (∃! k, k < 4 ∧ (∃ i j : fin 6, i ≠ j ∧ s1 i = s2 i ∧ s1 j = s2 j) = (k = 2) )) →
  n ≤ 18 :=
by
  sorry -- Proof to be provided

end maximum_students_l307_307378


namespace find_AX_bisect_angle_bisector_l307_307415

theorem find_AX_bisect_angle_bisector (AC BC BX : ℝ) (CX_bisects_angle : ∀ AX, (AC / AX = BC / BX)) :
    ∃ AX, AX = 108 / 5 :=
by
  have AC := 27
  have BC := 45
  have BX := 36
  use 108 / 5
  sorry

end find_AX_bisect_angle_bisector_l307_307415


namespace find_a_minus_b_l307_307924

theorem find_a_minus_b (a b : ℝ)
  (h1 : 6 = a * 3 + b)
  (h2 : 26 = a * 7 + b) :
  a - b = 14 := 
sorry

end find_a_minus_b_l307_307924


namespace equal_BDE_EDC_l307_307577

-- Definitions for angle trisection and point placement
variables {A B C D E : Type*} [eq A B C] [eq A C B] [eq D B E C] [eq E B C System.qfe] (trisect : Prop)

-- Assumptions
def angle_trisected (P Q R : Type*) [triangle P Q R] (S : Type*) : Prop := 
  trisect => 
    ∃ p q r, 
  angle P S = 1/3 * angle P Q, 
  angle S R = 1/3 * angle P Q

def point_lies_interior (P Q R S : Type*) [triangle P Q R] (S : Type*) : Prop := 
  ∃ p, 
  p ∈ triangle P Q R, 
  p = interior P Q R, 
  closer_to_side P Q

-- Conditions
variables (BDtrisect : angle_trisected B A C D)
variables (BEtrisect : angle_trisected B A C E)
variables (CDtrisect : angle_trisected C A B D)
variables (CEtrisect : angle_trisected C A B E)
variables (E_position : point_lies_interior A B C E)

-- Main statement
theorem equal_BDE_EDC (BDtrisect : angle_trisected B A C D)
                     (BEtrisect : angle_trisected B A C E)
                     (CDtrisect : angle_trisected C A B D)
                     (CEtrisect : angle_trisected C A B E)
                     (E_position : point_lies_interior A B C E) :
  angle B D E = angle E D C :=
sorry

end equal_BDE_EDC_l307_307577


namespace coeff_of_x90_l307_307037

noncomputable def coeff_x90 : ℕ :=
  let poly := (list.range 14).foldl (λ p n, p * (Polynomial.X ^ (n + 1) - Polynomial.C (n + 1))) (1 : Polynomial ℚ) in
  Polynomial.coeff poly 90

theorem coeff_of_x90 :
  coeff_x90 = -1511 := 
sorry

end coeff_of_x90_l307_307037


namespace xiaodong_sister_age_correct_l307_307731

/-- Let's define the conditions as Lean definitions -/
def sister_age := 13
def xiaodong_age := sister_age - 8
def sister_age_in_3_years := sister_age + 3
def xiaodong_age_in_3_years := xiaodong_age + 3

/-- We need to prove that in 3 years, the sister's age will be twice Xiaodong's age -/
theorem xiaodong_sister_age_correct :
  (sister_age_in_3_years = 2 * xiaodong_age_in_3_years) → sister_age = 13 :=
by
  sorry

end xiaodong_sister_age_correct_l307_307731


namespace Toby_friends_girls_l307_307750

theorem Toby_friends_girls (F G : ℕ) (h1 : 0.55 * F = 33) (h2 : F - 33 = G) : G = 27 := 
by
  sorry

end Toby_friends_girls_l307_307750


namespace oblique_axonometric_area_l307_307162

-- defining conditions as constants
def side_length (a : ℝ) : ℝ := a
def area_ratio : ℝ := 2 * Real.sqrt 2

-- theorem statement
theorem oblique_axonometric_area (a : ℝ) (h : a > 0) :
  let planar_area := (side_length a) ^ 2 in
  let intuitive_area := planar_area / area_ratio in
  intuitive_area = (Real.sqrt 2 / 4) * (side_length a) ^ 2 :=
by
  -- proof goes here
  sorry

end oblique_axonometric_area_l307_307162


namespace number_of_integers_l307_307747

theorem number_of_integers (n : ℤ) : 
  (set_of (λ n, -125 < n^3 ∧ n^3 < 125)).card = 9 :=
sorry

end number_of_integers_l307_307747


namespace average_speed_is_correct_l307_307982

def total_distance : ℝ := 450
def distance_1 : ℝ := 300
def distance_2 : ℝ := total_distance - distance_1
def speed_1 : ℝ := 20
def speed_2 : ℝ := 15

def time_1 : ℝ := distance_1 / speed_1
def time_2 : ℝ := distance_2 / speed_2
def total_time : ℝ := time_1 + time_2

def average_speed : ℝ := total_distance / total_time

theorem average_speed_is_correct : average_speed = 18 :=
by
  sorry

end average_speed_is_correct_l307_307982


namespace find_DP_l307_307738

-- Define the conditions
variable (A B C D K P N : Type)
variable (angle : Type → Type → Type)
variable (tangent : Type → Type → Prop)
variable (similar : Type → Type → Prop)
variable (med : Type → Type → Type)
variable (sqrt : ℝ → ℝ)
variable (AC CD AB AD BD x y : ℝ)

-- Given angle conditions
axiom angle_APBAC : angle P B = angle A C
axiom angle_APBAKC : angle P B = angle A K C
axiom angle_AKCBAC : angle A K C = angle B A C
axiom angle_KACABC : angle K A C = angle A B C

-- Given geometric properties
axiom tangent_AC_circle : tangent A C
axiom sim_ABC_AKC : similar (triangle A B C) (triangle A K C)

-- Given side lengths and relationships
axiom ratio_AB_4 : AB / 4 = AC / 3
axiom ratio_AC_3 : AC / 3 = 12 / AC
axiom length_AC : AC = 6
axiom length_AB : AB = 8

-- Define the median CD
axiom CD_median : med C D = true
axiom length_CD : CD = sqrt 74

-- Power of point theorem conditions
axiom power_AD_DB : AD * BD = D P * D N
axiom product_xy : x * y = 16
axiom tangent_seacant : (sqrt 74 - y) * (sqrt 74 + x) = 36

-- Define the correct answer
axiom DP_value : x = (-11 + 3 * sqrt 145) / sqrt 74

theorem find_DP : x = (-11 + 3 * sqrt 145) / sqrt 74 := by
  sorry

end find_DP_l307_307738


namespace circle_eq_standard_intersecting_line_values_l307_307882

-- Definitions for geometric entities used in the problem
structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def circle (center : Point) (radius : ℝ) : set Point :=
  {p | (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2}

noncomputable def line (a b c : ℝ) : set Point :=
  {p | a * p.x + b * p.y + c = 0}

-- Condition: Circle passing through points (2,1) and (3,2) and being symmetric wrt line x - 3y = 0
def C := circle ⟨3, 1⟩ 1
def P1 := Point.mk 2 1
def P2 := Point.mk 3 2
def symmetry_line := line 1 (-3) 0

-- Question (1): Prove the standard equation of circle C is (x-3)^2 + (y-1)^2 = 1
theorem circle_eq_standard :
  ∃ (a b r : ℝ), (a = 3 ∧ b = 1 ∧ r = 1) ∧ ∀ (p : Point), p ∈ C ↔ (p.x - a)^2 + (p.y - b)^2 = r^2 := sorry

-- Condition: Circle intersects the line x + 2y + m = 0 at points A and B with |AB| = 4√5/5
def intersect_line (m : ℝ) := line 1 2 m
def dist_AB := 4 * real.sqrt 5 / 5

-- Question (2): Prove the value of m is -4 or -6
theorem intersecting_line_values :
  ∃ m : ℝ, (m = -4 ∨ m = -6) ∧ (∃ A B : Point, 
  A ∈ (intersect_line m) ∧ A ∈ C ∧ B ∈ (intersect_line m) ∧ B ∈ C ∧
  (real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)) = dist_AB) := sorry

end circle_eq_standard_intersecting_line_values_l307_307882


namespace Qn_divisible_by_Q1_l307_307226

noncomputable def P : ℕ → ℝ → ℝ := sorry
def Q1 (x : ℝ) : ℝ := P 1 x - x

def Qn (n : ℕ) : (ℝ → ℝ) :=
  if n = 0 then λ x, x
  else if n = 1 then Q1
  else λ x, P n (Qn (n-1) x) - x

theorem Qn_divisible_by_Q1 (n : ℕ) (hn : n ∈ ℕ) : 
  ∃ Rn : ℝ → ℝ, ∀ x : ℝ, Qn n x = Rn x * Q1 x :=
sorry

end Qn_divisible_by_Q1_l307_307226


namespace RSTU_bicentric_iff_AC_perp_BD_l307_307182

variables {A B C D P R S T U : Type} [Nonempty R] [Nonempty S] [Nonempty T] [Nonempty U]
variables (cyclicQuad : CyclicQuadrilateral A B C D)
variables (P_intersection : ∃ (P : Type), IsIntersection P A C B D)
variables (perpendiculars : Π (P : Type), (P ⊥ A ∧ P ⊥ B) ∧ (P ⊥ C ∧ P ⊥ D) ∧ (P ⊥ R ∧ P ⊥ S) ∧ (P ⊥ T ∧ P ⊥ U))

theorem RSTU_bicentric_iff_AC_perp_BD :
  BicentricQuadrilateral R S T U ↔ Perpendicular AC BD :=
begin
  sorry
end

end RSTU_bicentric_iff_AC_perp_BD_l307_307182


namespace correct_propositions_l307_307892

theorem correct_propositions (A B : Prop) (P : Prop → ℝ) :
  (0 ≤ P A ∧ P A ≤ 1) ∧
  ¬(A ∧ (A → B) ∧ (B → ¬A)) ∧
  ¬((A ∨ B) ∧ (A → B) ∧ (B → ¬A)) ∧
  (¬A = B ∧ (A ↔ ¬B)) :=
begin 
  -- The Lean proof omitted
  sorry
end

end correct_propositions_l307_307892


namespace find_rate_of_stream_l307_307330

noncomputable def rate_of_stream (v : ℝ) : Prop :=
  let rowing_speed := 36
  let downstream_speed := rowing_speed + v
  let upstream_speed := rowing_speed - v
  (1 / upstream_speed) = 3 * (1 / downstream_speed)

theorem find_rate_of_stream : ∃ v : ℝ, rate_of_stream v ∧ v = 18 :=
by
  use 18
  unfold rate_of_stream
  sorry

end find_rate_of_stream_l307_307330


namespace max_value_of_x_minus_y_l307_307605

theorem max_value_of_x_minus_y
  (x y : ℝ)
  (h : 2 * (x ^ 2 + y ^ 2 - x * y) = x + y) :
  x - y ≤ 1 / 2 := 
sorry

end max_value_of_x_minus_y_l307_307605


namespace expected_scurried_home_mn_sum_l307_307297

theorem expected_scurried_home_mn_sum : 
  let expected_fraction : ℚ := (1/2 + 2/3 + 3/4 + 4/5 + 5/6 + 6/7 + 7/8)
  let m : ℕ := 37
  let n : ℕ := 7
  m + n = 44 := by
  sorry

end expected_scurried_home_mn_sum_l307_307297


namespace taizhou_investment_scientific_notation_l307_307147

theorem taizhou_investment_scientific_notation :
  (314.86 * 10^8 : ℝ) = 3.1486 * 10^10 :=
begin
  sorry
end

end taizhou_investment_scientific_notation_l307_307147


namespace right_triangle_with_a_as_hypotenuse_l307_307537

theorem right_triangle_with_a_as_hypotenuse
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a = (b^2 + c^2 - a^2) / (2 * b * c))
  (h2 : b = (a^2 + c^2 - b^2) / (2 * a * c))
  (h3 : c = (a^2 + b^2 - c^2) / (2 * a * b))
  (h4 : a * ((b^2 + c^2 - a^2) / (2 * b * c)) + b * ((a^2 + c^2 - b^2) / (2 * a * c)) = c * ((a^2 + b^2 - c^2) / (2 * a * b))) :
  a^2 = b^2 + c^2 :=
by
  sorry

end right_triangle_with_a_as_hypotenuse_l307_307537


namespace candy_count_in_third_set_l307_307696

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l307_307696


namespace number_of_four_digit_numbers_with_hundreds_digit_3_l307_307909

theorem number_of_four_digit_numbers_with_hundreds_digit_3 : 
  (number_of_four_digit_numbers_with_hundreds_digit_3 = 900) := by 
sorry

end number_of_four_digit_numbers_with_hundreds_digit_3_l307_307909


namespace compute_sum_l307_307639

-- Define the variables as real numbers
variables (x y z : ℝ)

-- Define the conditions as hypotheses
def hyp1 : Prop := (xz / (x + y)) + (yx / (y + z)) + (zy / (z + x)) = -5
def hyp2 : Prop := (yz / (x + y)) + (zx / (y + z)) + (xy / (z + x)) = 7

-- Define the theorem to prove
theorem compute_sum (h1 : hyp1) (h2 : hyp2) : x + y + z = 2 :=
sorry

end compute_sum_l307_307639


namespace part1_part2_l307_307112

noncomputable def U := ℝ
noncomputable def A := {x : ℝ | x^2 - 3 * x - 18 ≥ 0}
noncomputable def B := {x : ℝ | (x + 5) / (x - 14) ≤ 0}

theorem part1 : (A ∩ (U \ B) = (Iic (-5) ∪ Ici 14)) :=
  by sorry

noncomputable def C (a : ℝ) := {x : ℝ | 2 * a < x ∧ x < a + 1}

theorem part2 (a : ℝ) : (C a ⊆ B) → a ≥ -5 / 2 :=
  by sorry

end part1_part2_l307_307112


namespace find_extrema_l307_307496

noncomputable def f (x m : ℝ) : ℝ := x ^ 2 - 2 * m * x + m - 1

theorem find_extrema (x : ℝ) (f : ℝ → ℝ) (m : ℝ) (hx : 0 ≤ x ∧ x ≤ 4) :
  (m < 0 → ∃ a b, f(a) = m - 1 ∧ f(b) = 15 - 7 * m) ∧ 
  (0 ≤ m ∧ m ≤ 4 → ∃ c, f(c) = -m ^ 2 + m - 1 ∧ 
    ((0 ≤ m ∧ m ≤ 2 → ∃ d, f(d) = 15 - 7 * m) ∧ 
     (2 ≤ m ∧ m ≤ 4 → ∃ e, f(e) = m - 1))) ∧ 
  (m > 4 → ∃ g h, f(g) = 15 - 7 * m ∧ f(h) = m - 1) := 
by 
  sorry

end find_extrema_l307_307496


namespace intersection_M_N_l307_307285

def M : Set ℝ := {x : ℝ | log (1 - x) < 0}

def N : Set ℝ := {x : ℝ | x^2 ≤ 1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
  sorry

end intersection_M_N_l307_307285


namespace letters_identity_l307_307232

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l307_307232


namespace range_of_a_l307_307143

-- Given conditions
def condition1 (x : ℝ) := (4 + x) / 3 > (x + 2) / 2
def condition2 (x : ℝ) (a : ℝ) := (x + a) / 2 < 0

-- The statement to prove
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, condition1 x → condition2 x a → x < 2) → a ≤ -2 :=
sorry

end range_of_a_l307_307143


namespace find_a_plus_b_l307_307345

-- Define the problem constraints
def ellipse_foci_circle_intersection (r a b : ℝ) : Prop :=
  ellipse_equation ∧ circle_passing_foci ∧ circle_intersects_ellipse

-- We need to specify the mathematical conditions for the Lean theorem.
noncomputable def ellipse_equation := ∀ (x y : ℝ), x^2 + 16 * y^2 = 16

noncomputable def circle_passing_foci (r : ℝ) := (0 ≤ r) ∧ (r ∈ [sqrt 15, 8)) 

noncomputable def circle_intersects_ellipse (r : ℝ) : Prop :=
  ∃ (h : ℝ), sqrt (h^2 + 15) = r ∧ (r < 8) ∧ (r ≥ sqrt 15)

-- Define the theorem statement.
theorem find_a_plus_b (a b : ℝ) (r : ℝ) (h : ℝ) 
  (h_eq : ellipse_equation) 
  (h_circle : circle_passing_foci r)
  (h_intersect : circle_intersects_ellipse r)
  : a + b = sqrt 15 + 8 :=
sorry

end find_a_plus_b_l307_307345


namespace alpha_in_second_quadrant_l307_307126

theorem alpha_in_second_quadrant (α : ℝ) 
  (h1 : Real.sin α > Real.cos α)
  (h2 : Real.sin α * Real.cos α < 0) : 
  (Real.sin α > 0) ∧ (Real.cos α < 0) :=
by 
  -- Proof omitted
  sorry

end alpha_in_second_quadrant_l307_307126


namespace joseph_total_socks_is_32_l307_307584

variables (pairs_of_blue pairs_of_black pairs_of_red pairs_of_white : ℕ)
variable (total_socks : ℕ)

-- Conditions
axiom cond1 : pairs_of_blue = pairs_of_black + 3
axiom cond2 : pairs_of_red = pairs_of_white - 1
axiom cond3 : 6 = pairs_of_red * 2
axiom cond4 : pairs_of_red * 2 = 6

-- Definition of total number of socks
def compute_total_socks (pairs_of_red pairs_of_blue pairs_of_black pairs_of_white : ℕ) :=
  (pairs_of_red * 2) + (pairs_of_blue * 2) + (pairs_of_black * 2) + (pairs_of_white * 2)

-- Theorem stating the total number of socks
theorem joseph_total_socks_is_32 : total_socks = 32 :=
by {
  -- Calculate the pairs of each type of socks
  have h_pairs_of_red : pairs_of_red = 3, from sorry,
  have h_pairs_of_blue : pairs_of_blue = 6, from sorry,
  have h_pairs_of_black : pairs_of_black = 3, from sorry,
  have h_pairs_of_white : pairs_of_white = 4, from sorry,

  -- Calculate total socks
  let total_socks := compute_total_socks pairs_of_red pairs_of_blue pairs_of_black pairs_of_white,

  -- Prove total number of socks is 32
  show total_socks = 32, from sorry
}

end joseph_total_socks_is_32_l307_307584


namespace probability_longer_piece_at_least_x_squared_l307_307792

noncomputable def probability_longer_piece (x : ℝ) : ℝ :=
  if x = 0 then 1 else (2 / (x^2 + 1))

theorem probability_longer_piece_at_least_x_squared (x : ℝ) :
  probability_longer_piece x = (2 / (x^2 + 1)) :=
sorry

end probability_longer_piece_at_least_x_squared_l307_307792


namespace divisor_sequence_probability_l307_307185

noncomputable def m := sorry

theorem divisor_sequence_probability:
  let S := (24 : ℤ) ^ 9 in
  let divisors := {d : ℤ | d ∣ S ∧ 0 < d} in
  let chosen_list := (list ℤ) := sorry in
  let probability := sorry in
  let m := sorry in
  ∃ (m n : ℕ) (hm : nat.coprime m n), probability = (m:ℚ)/n ∧ nat.prime m := sorry

end divisor_sequence_probability_l307_307185


namespace arccos_neg_half_eq_two_pi_div_three_l307_307815

theorem arccos_neg_half_eq_two_pi_div_three :
  ∀ x : ℝ, (cos x = -1 / 2 ∧ 0 ≤ x ∧ x ≤ π) → x = 2 * π / 3 :=
by
  intro x
  intro h
  sorry

end arccos_neg_half_eq_two_pi_div_three_l307_307815


namespace volume_of_salt_proof_l307_307347

noncomputable def volume_of_salt {r h : ℝ} (π : ℝ) (ratio : ℝ) (radius : r = 1.5) (height : h = 3) (salt_ratio : ratio = 1 / 6) : ℝ :=
let V_solution := π * r^2 * h in
let V_salt := V_solution * ratio in
V_salt

theorem volume_of_salt_proof (π approx_value : ℝ) (V_salt_rounded : V_salt_rounded = 3.53) : V_salt_rounded = (volume_of_salt π 1/6 1.5 3 /1/6 /approx_value/π/π *1.125 *1.125  : ∃ (volume_of_salt V_salt_rounded) →/volumeL_value_salt => sorry π volume_of_lean 1/6 ) :=
begin
  sorry
end

end volume_of_salt_proof_l307_307347


namespace letters_identity_l307_307243

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l307_307243


namespace determine_set_l307_307885

noncomputable def f (x : ℝ) : ℝ := 2^|x| - 4

theorem determine_set (x : ℝ) :
  f(x) = 2^x - 4 → (f x = f (-x)) → {x | f(x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  intro h1 h2
  sorry

end determine_set_l307_307885


namespace swimming_time_per_style_l307_307673

theorem swimming_time_per_style (d v1 v2 v3 v4 t: ℝ) 
    (h1: d = 600) 
    (h2: v1 = 45) 
    (h3: v2 = 35) 
    (h4: v3 = 40) 
    (h5: v4 = 30)
    (h6: t = 15) 
    (h7: d / 4 = 150) 
    : (t / 4 = 3.75) :=
by
  sorry

end swimming_time_per_style_l307_307673


namespace proof_problem1_proof_problem2_l307_307004

noncomputable def problem1 : ℝ :=
  (1) * (Real.sqrt (2 * Real.sqrt 2)) ^ (4 / 3) -
  4 * (16 / 49) ^ (-1 / 2) -
  (2 ^ (1 / 4)) * (8 ^ (1 / 4)) +
  (-2014) ^ 0

noncomputable def problem2 : ℝ :=
  Real.log (6.25) / Real.log (2.5) +
  Real.log10 (1 / 100) +
  Real.log (exp (1) * sqrt (exp (1))) +
  Real.log (Real.log 16 / Real.log 2) / Real.log 2

theorem proof_problem1 : problem1 = -6 :=
  by
  sorry

theorem proof_problem2 : problem2 = 7 / 2 :=
  by
  sorry

end proof_problem1_proof_problem2_l307_307004


namespace third_set_candies_l307_307688

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l307_307688


namespace arithmetic_sequence_property_l307_307188

variable {α : Type}
variable [LinearOrderedField α]

-- Defining an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a(n+1) = a(n) + d

-- The statement to prove
theorem arithmetic_sequence_property (a : ℕ → α) (h : is_arithmetic_sequence a) :
  a 1 ≥ a 2 → (a 2) ^ 2 ≥ a 1 * a 3 :=
by
  sorry

end arithmetic_sequence_property_l307_307188


namespace coupon_savings_l307_307791

/--
Problem: Determine the smallest and largest prices, \( x \) and \( y \) respectively, 
for which Coupon A saves at least as much money as Coupon B or C. What is the value of \( y - x \)?
Conditions:
1. Listed price \( P \) is greater than \$100 (\( P > 100 \)).
2. Coupon A gives 20% off the listed price.
3. Coupon B gives \$40 off the listed price.
4. Coupon C gives 30% off the amount by which the listed price exceeds \$100.
-/
theorem coupon_savings (P : ℕ) (h : P > 100) :
  let A_savings := 0.20 * P,
      B_savings := 40,
      C_savings := 0.30 * (P - 100) in
  (A_savings ≥ B_savings) ∧ (A_savings ≥ C_savings) → 
  let x := 200, y := 300 in y - x = 100 :=
by 
  sorry

end coupon_savings_l307_307791


namespace length_of_AB_l307_307574

-- Define the problem variables
variables (AB CD : ℝ)
variables (h : ℝ)

-- Define the conditions
def ratio_condition (AB CD : ℝ) : Prop :=
  AB / CD = 7 / 3

def length_condition (AB CD : ℝ) : Prop :=
  AB + CD = 210

-- Lean statement combining the conditions and the final result
theorem length_of_AB (h : ℝ) (AB CD : ℝ) (h_ratio : ratio_condition AB CD) (h_length : length_condition AB CD) : 
  AB = 147 :=
by
  -- Definitions and proof would go here
  sorry

end length_of_AB_l307_307574


namespace five_digit_integers_divisible_by_12_count_l307_307518

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

def two_digit_multiples_of_12 : List ℕ :=
  [12, 24, 36, 48, 60, 72, 84, 96]

def count_five_digit_integers_divisible_by_12 : ℕ :=
  9 * 10 * 10 * 10 * 8

theorem five_digit_integers_divisible_by_12_count :
  (count_five_digit_integers_divisible_by_12 = 72000) :=
by
  rw [count_five_digit_integers_divisible_by_12]
  norm_num
  -- We skip the detailed proof steps here
  sorry

end five_digit_integers_divisible_by_12_count_l307_307518


namespace necessary_and_sufficient_condition_holds_l307_307659

noncomputable def necessary_and_sufficient_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + m > 0

theorem necessary_and_sufficient_condition_holds (m : ℝ) :
  necessary_and_sufficient_condition m ↔ m > 1 :=
by
  sorry

end necessary_and_sufficient_condition_holds_l307_307659


namespace arccos_neg_half_eq_two_pi_div_three_l307_307816

theorem arccos_neg_half_eq_two_pi_div_three :
  ∀ x : ℝ, (cos x = -1 / 2 ∧ 0 ≤ x ∧ x ≤ π) → x = 2 * π / 3 :=
by
  intro x
  intro h
  sorry

end arccos_neg_half_eq_two_pi_div_three_l307_307816


namespace christmas_not_one_seventh_probability_l307_307372

-- Definitions for conditions
def is_common_year (y : ℕ) : Prop := y % 4 ≠ 0 ∨ (y % 100 = 0 ∧ y % 400 ≠ 0)

def is_leap_year (y : ℕ) : Prop := y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)

def days_in_common_year : ℕ := 365

def days_in_leap_year : ℕ := 366

-- Main theorem statement
theorem christmas_not_one_seventh_probability :
  ∀ (prob : ℚ), prob = 1/7 → 
  ¬ (∃ N : ℕ, christmas_count N / 400 = prob) :=
by sorry

-- Definition of Christmas count in 400 years
noncomputable def christmas_count (N : ℕ) : ℕ :=
  let common_days := 303 * days_in_common_year,
      leap_days := 97 * days_in_leap_year,
      total_days := common_days + leap_days in
  total_days % 7


end christmas_not_one_seventh_probability_l307_307372


namespace measure_angle_ABC_l307_307569

-- Mathematical Definitions
def is_isosceles_trapezoid (a b c d : ℝ) : Prop :=
  ∃ (parallel_pairs: b = d ∧ a = c), true

def is_regular_nonagon (angles: List ℝ) : Prop :=
  angles.length = 9 ∧ (∀ angle, angle ∈ angles → angle = 140)

theorem measure_angle_ABC :
  ∃ (trapezoids: List (ℝ×ℝ×ℝ×ℝ)), 
    (trapezoids.length = 9 ∧ ∀ t ∈ trapezoids, is_isosceles_trapezoid t.1 t.2 t.3 t.4) → 
    ∃ angle_ABC : ℝ, angle_ABC = 110 :=
by
  intros
  sorry

end measure_angle_ABC_l307_307569


namespace museum_schedule_solutions_l307_307667

-- The number of students available.
def num_students : ℕ := 11

-- The number of hours the event runs.
def hours : ℕ := 8

-- a_n denotes the number of valid schedules for n hours.
def valid_schedules : ℕ → ℕ
| 1       := num_students
| 2       := num_students * (num_students - 1)
| (n + 1) := valid_schedules n + 10 ^ n

-- We need to prove this statement for the specific case when n = 8.
theorem museum_schedule_solutions :
  valid_schedules hours = 100000010 :=
by
  sorry

end museum_schedule_solutions_l307_307667


namespace total_candies_third_set_l307_307704

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l307_307704


namespace maddox_theo_equal_profit_l307_307974

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

end maddox_theo_equal_profit_l307_307974


namespace distance_from_point_to_line_polar_l307_307900

theorem distance_from_point_to_line_polar 
  (ρ θ : ℝ) 
  (hline : ∀ ρ θ, ρ * Real.sin(θ + π / 4) = √2 / 2) 
  (A_coord : (2, π / 6)) 
  (A_rect : (Real.sqrt 3, 1)) 
  :
  let distance := (Real.abs (Real.sqrt 3 + 1 - 1)) / Real.sqrt 2 in
  distance = Real.sqrt 6 / 2 := 
by 
  sorry

end distance_from_point_to_line_polar_l307_307900


namespace arithmetic_mean_of_18_27_45_l307_307720

theorem arithmetic_mean_of_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_of_18_27_45_l307_307720


namespace area_of_disks_l307_307404

theorem area_of_disks : 
  (∀ (n : ℕ) (s : ℝ) (r : ℝ), 
    n = 8 ∧ s = 2 ∧ r = √2/2 → 
    8 * (π * r^2) = 4 * π) :=
by 
  intros n s r h 
  cases h 
  .1 
  .2.symm ▸ rfl 
  case h.left => simp
  case h.right => simp
  sorry

end area_of_disks_l307_307404


namespace binom_13_10_eq_286_l307_307806

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_13_10_eq_286 : binomial 13 10 = 286 := by
  sorry

end binom_13_10_eq_286_l307_307806


namespace distance_between_foci_of_hyperbola_l307_307268

theorem distance_between_foci_of_hyperbola :
  (∀ x y : ℝ, (y = 2 * x + 3) ∨ (y = -2 * x + 1)) →
  ∀ p : ℝ × ℝ, (p = (2, 1)) →
  ∃ d : ℝ, d = 2 * Real.sqrt 30 :=
by
  sorry

end distance_between_foci_of_hyperbola_l307_307268


namespace meaningful_expression_range_l307_307542

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by 
  sorry

end meaningful_expression_range_l307_307542


namespace probability_diamond_or_club_l307_307365

/--
Given a standard deck of 52 cards randomly shuffled, with each suit
(diamonds, clubs, hearts, and spades) having 13 cards, prove that 
the probability of the top card being either a diamond or a club is 1/2.
-/
theorem probability_diamond_or_club :
  let total_cards := 52
  let suit_card_count := 13
  let favorables := 2 * suit_card_count -- (13 diamonds + 13 clubs)
  let probability := (favorables : ℝ) / total_cards
  probability = 1 / 2 :=
begin
  sorry
end

end probability_diamond_or_club_l307_307365


namespace problem_1_problem_2_l307_307468

variable (x y z : ℝ)

-- The first part of the problem
theorem problem_1 (h : x + y + z = 1) : 
  sqrt (3 * x + 1) + sqrt (3 * y + 2) + sqrt (3 * z + 3) ≤ 3 * sqrt 3 := 
  sorry

-- The second part of the problem
theorem problem_2 (h : x + 2 * y + 3 * z = 6) : 
  ∃ m : ℝ, m = x^2 + y^2 + z^2 ∧ m = 18 / 7 :=
  sorry

end problem_1_problem_2_l307_307468


namespace darnel_lap_difference_l307_307395

theorem darnel_lap_difference (sprint jog : ℝ) (h_sprint : sprint = 0.88) (h_jog : jog = 0.75) : sprint - jog = 0.13 := 
by 
  rw [h_sprint, h_jog] 
  norm_num

end darnel_lap_difference_l307_307395


namespace max_distance_parallel_tangents_l307_307085

theorem max_distance_parallel_tangents (l1 l2 : ℝ → ℝ)
    (h_tangent_l1 : ∃ x1 : ℝ, ∀ y, y = (1 / x1) ∧ l1 y = - (1 / x1^2) * (x - x1) + (1 / x1))
    (h_tangent_l2 : ∃ x2 : ℝ, ∀ y, y = (1 / x2) ∧ l2 y = - (1 / x2^2) * (x - x2) + (1 / x2))
    (h_parallel : ∀ x1 x2, l1 = l2) :
  max (dist l1 l2) = 2 * sqrt 2 := sorry

end max_distance_parallel_tangents_l307_307085


namespace rectangular_coordinates_new_point_l307_307441

def given_point := (12, 5)
def polar_coordinates (x y : ℝ) : ℝ × ℝ := (real.sqrt (x*x + y*y), real.atan (y / x))
def new_polar_coordinates (r θ : ℝ) : ℝ × ℝ := (2 * r^2, 3 * θ)
def rectangular_coordinates (r θ : ℝ) : ℝ × ℝ := 
  let cosθ := real.cos θ
  let sinθ := real.sin θ
  (r * cosθ, r * sinθ)

theorem rectangular_coordinates_new_point :
  let (x, y) := given_point
  let (r, θ) := polar_coordinates x y
  let (nr, nθ) := new_polar_coordinates r θ
  let cosθ := real.cos θ
  let sinθ := real.sin θ
  let nx := nr * (cosθ^3 - 3 * cosθ * (sinθ^2))
  let ny := nr * (3 * sinθ * (cosθ^2) - (sinθ^3))
  (nx, ny) = (338 * (828 / 2197), 338 * (2035 / 2197)) :=
by
  sorry

end rectangular_coordinates_new_point_l307_307441


namespace cos_neg_45_eq_one_over_sqrt_two_l307_307008

theorem cos_neg_45_eq_one_over_sqrt_two : Real.cos (-(45 : ℝ)) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_neg_45_eq_one_over_sqrt_two_l307_307008


namespace sqrt_meaningful_iff_l307_307139

theorem sqrt_meaningful_iff (x : ℝ) : (∃ (y : ℝ), y = sqrt (x - 6)) ↔ x ≥ 6 :=
by
  sorry

end sqrt_meaningful_iff_l307_307139


namespace angle_equality_acute_triangle_l307_307447

noncomputable def acute_triangle (A B C : Type) : Prop :=
  ∃ (triangle_acute : ∀ (A B C : Type), ∃ (tangents_of_circumcircle : Type),
  ∃ (D E P R Q S : Type),
    D = tangent_of_circumcircle_from A ∧
    E = tangent_of_circumcircle_from B ∧
    P = (line AE) ∩ (line BC) ∧
    R = (line BD) ∩ (line AC) ∧
    Q = midpoint (segment AP) ∧
    S = midpoint (segment BR))

theorem angle_equality_acute_triangle {A B C D E P R Q S : Type}
  (h1 : acute_triangle A B C)
  (h2 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h3 : line AE ∩ line BC = P)
  (h4 : line BD ∩ line AC = R)
  (h5 : Q = midpoint (segment AP))
  (h6 : S = midpoint (segment BR)) :
  ∠ (A B Q) = ∠ (A B S) :=
sorry

end angle_equality_acute_triangle_l307_307447


namespace letters_identity_l307_307245

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l307_307245


namespace conic_sections_hyperbola_and_ellipse_l307_307526

theorem conic_sections_hyperbola_and_ellipse
  (x y : ℝ) (h : y^4 - 9 * x^4 = 3 * y^2 - 3) :
  (∃ a b c : ℝ, a * y^2 - b * x^2 = c ∧ a = b ∧ c ≠ 0) ∨ (∃ a b c : ℝ, a * y^2 + b * x^2 = c ∧ a ≠ b ∧ c ≠ 0) :=
by
  sorry

end conic_sections_hyperbola_and_ellipse_l307_307526


namespace geom_seq_log_sum_l307_307926

noncomputable def geom_ln_sum (a : ℕ → ℝ) :=
  (∀ n, 0 < a n) ∧ (∀ m n, a (m + n) = a m * a n) ∧ (a 10 * a 11 + a 9 * a 12 + a 8 * a 13 = 3 * real.exp 5)

theorem geom_seq_log_sum (a : ℕ → ℝ) (h : geom_ln_sum a) :
  ∑ i in finset.range 20, real.log (a (i + 1)) = 50 :=
sorry

end geom_seq_log_sum_l307_307926


namespace kayla_apples_l307_307586

variable (x y : ℕ)
variable (h1 : x + (10 + 4 * x) = 340)
variable (h2 : y = 10 + 4 * x)

theorem kayla_apples : y = 274 :=
by
  sorry

end kayla_apples_l307_307586


namespace rabbit_wins_race_l307_307787

theorem rabbit_wins_race :
  ∀ (rabbit_speed1 rabbit_speed2 snail_speed rest_time total_distance : ℕ)
  (rabbit_time1 rabbit_time2 : ℚ),
  rabbit_speed1 = 20 →
  rabbit_speed2 = 30 →
  snail_speed = 2 →
  rest_time = 3 →
  total_distance = 100 →
  rabbit_time1 = (30 : ℚ) / rabbit_speed1 →
  rabbit_time2 = (70 : ℚ) / rabbit_speed2 →
  (rabbit_time1 + rest_time + rabbit_time2 < total_distance / snail_speed) :=
by
  intros
  sorry

end rabbit_wins_race_l307_307787


namespace exists_l307_307423

def smallest_a_composite_expression : ℕ :=
  8

theorem exists smallest_a_composite : ∀ x : ℤ, ∃ a : ℕ, (∀ y : ℤ, ¬ prime (y^4 + a^2 + 16)) ∧ a = 8 :=
by
  sorry

end exists_l307_307423


namespace cards_received_while_in_hospital_l307_307621

theorem cards_received_while_in_hospital (T H C : ℕ) (hT : T = 690) (hC : C = 287) (hH : H = T - C) : H = 403 :=
by
  sorry

end cards_received_while_in_hospital_l307_307621


namespace product_sequence_equality_l307_307382

theorem product_sequence_equality :
  (∏ n in finset.range 2 (7+1), (1 - 1 / (n^3 : ℝ))) = 
  (7 * 26 * 63 * 124 * 215 * 342) / (8 * 27 * 64 * 125 * 216 * 343) := by
  sorry

end product_sequence_equality_l307_307382


namespace distance_from_F_to_midpoint_DE_l307_307936

-- Conditions
variables (DE DF EF : ℝ) (is_right_triangle : DE^2 = DF^2 + EF^2) (DE_eq : DE = 15) (DF_eq : DF = 9) (EF_eq : EF = 12)

-- The problem statement
theorem distance_from_F_to_midpoint_DE : (distance (F : ℝ × ℝ) (midpoint (DE : ℝ × ℝ) (D : ℝ × ℝ) E) = 7.5) :=
by sorry

end distance_from_F_to_midpoint_DE_l307_307936


namespace length_OQ_eq_third_OP_l307_307576

theorem length_OQ_eq_third_OP 
    (A B C P Q R O : euclidean_space ℝ 2) 
    (hAB : dist A B = 10)
    (hBC : dist B C = 11)
    (hAC : dist A C = 12)
    (hP : midpoint ℝ A B P)
    (hQ : midpoint ℝ B C Q) 
    (hR : midpoint ℝ A C R)
    (hO : centroid ℝ A B C O) :
    dist O Q = (1 / 3) * dist O P :=
sorry

end length_OQ_eq_third_OP_l307_307576


namespace range_of_m_l307_307099

theorem range_of_m {m : ℝ} (h : ∀ x : ℝ, (3 * m - 1) ^ x = (3 * m - 1) ^ x ∧ (3 * m - 1) > 0 ∧ (3 * m - 1) < 1) :
  1 / 3 < m ∧ m < 2 / 3 :=
by
  sorry

end range_of_m_l307_307099


namespace exists_unique_L_shapes_5x5_l307_307636

def Cell := ℕ × ℕ

def Grid := { cells : Finset Cell // ∀ cell ∈ cells, cell.1 < 5 ∧ cell.2 < 5 }

def is_L_shape (corner : Finset Cell) : Prop := ∃ (a b : Cell), 
  (a.1 = b.1 ∨ a.2 = b.2) ∧ corner = {a, b} ∧ a ≠ b

def unique_L_shapes (grid : Grid) : Prop :=
  ∃ (corners : list (Finset Cell)),
  (∀ corner ∈ corners, is_L_shape corner) ∧ 
  (corners.Nodup)

theorem exists_unique_L_shapes_5x5 :
    ∃ (corners : list (Finset Cell)),
    length corners = 5 ∧
    (∀ (corner : Finset Cell), corner ∈ corners → ∃ n, corner.card = n ∧ 1 ≤ n ∧ n ≤ 5) ∧
    (∀ i j, i < j → ∀ {A}, A ∈ corners.to_list → A ≠ A.shift i j) ∧ 
    (∀ {cell} (corner ∈ corners), cell ∈ corner → cell.1 < 5 ∧ cell.2 < 5) :=
sorry

end exists_unique_L_shapes_5x5_l307_307636


namespace fn_seq_correct_l307_307284

noncomputable def fn_seq (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then x / Real.sqrt(1 + x^2)
  else fn_seq (n - 1) (x / Real.sqrt(1 + (fn_seq (n - 1) x)^2))

theorem fn_seq_correct (n : ℕ) (x : ℝ) (hx : x > 0) : 
  fn_seq n x = x / Real.sqrt(1 + n * x^2) :=
begin
  induction n with k ih,
  { -- base case n = 1 if not in ℕ*
    sorry },
  { -- induction step
    sorry }
end

end fn_seq_correct_l307_307284


namespace weight_of_each_package_l307_307279

theorem weight_of_each_package (W : ℝ) 
  (h1: 10 * W + 7 * W + 8 * W = 100) : W = 4 :=
by
  sorry

end weight_of_each_package_l307_307279


namespace largest_n_divides_30_factorial_l307_307043

theorem largest_n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ nat.factorial 30 → m ≤ 7) ∧ (18^7 ∣ nat.factorial 30) :=
by
  sorry

end largest_n_divides_30_factorial_l307_307043


namespace even_function_f_l307_307471

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem even_function_f (hx : ∀ x : ℝ, f (-x) = f x) 
  (hg : ∀ x : ℝ, g (-x) = -g x)
  (h_pass : g (-1) = 1)
  (hg_eq_f : ∀ x : ℝ, g x = f (x - 1)) 
  : f 7 + f 8 = -1 := 
by
  sorry

end even_function_f_l307_307471


namespace jean_jail_time_l307_307168

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l307_307168


namespace smaller_square_area_l307_307803

theorem smaller_square_area
  (side_length : ℝ)
  (cut_length : ℝ)
  (angle : ℝ)
  (hypotenuse : ℝ)
  (new_side_length : ℝ)
  (new_area : ℝ)
  (h1 : side_length = 15)
  (h2 : cut_length = 4)
  (h3 : angle = real.pi / 4)
  (h4 : hypotenuse = 4 * real.sqrt 2)
  (h5 : new_side_length = hypotenuse)
  (h6 : new_area = new_side_length * new_side_length) :
  new_area = 32 := 
sorry

end smaller_square_area_l307_307803


namespace equal_distances_l307_307935

-- Define the problem statement in Lean

variable {α : Type*}
variables (A B C P K H : α) (AC AB CP BP : α → α) (E F : α)

-- Assume necessary properties of the points and lines
variable [AffinePlane α]

-- Assume that triangle ABC is acute
def is_acute_triangle (A B C: α) : Prop := sorry

-- Define midpoint E of AC and midpoint F of AB
def is_midpoint (E : α) (A C: α) : Prop := sorry
def is_midpoint (F : α) (A B: α) : Prop := sorry

-- Define that P is on the altitude AH
def on_altitude (P H A : α) : Prop := sorry

-- Define perpendicularities and intersection at K
def is_perpendicular (E : α) (CP : α) : Prop := sorry
def is_perpendicular (F : α) (BP : α) : Prop := sorry
def intersects_at (E_cp F_bp K : α) : Prop := sorry

theorem equal_distances 
  (h1 : is_acute_triangle A B C)
  (h2 : is_midpoint E A C)
  (h3 : is_midpoint F A B)
  (h4 : on_altitude P H A)
  (h5 : is_perpendicular E (CP P C))
  (h6 : is_perpendicular F (BP P B))
  (h7 : intersects_at (CP P K) (BP P K) K) :
  dist K B = dist K C :=
begin
  sorry
end

end equal_distances_l307_307935


namespace valerie_needs_72_stamps_l307_307305

noncomputable def total_stamps_needed : ℕ :=
  let thank_you_cards := 5
  let stamps_per_thank_you := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebates_more_than_bills := 3
  let rebate_stamps := 2
  let job_applications_factor := 2
  let job_application_stamps := 1

  let total_thank_you_stamps := thank_you_cards * stamps_per_thank_you
  let total_bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let total_rebates := total_bill_stamps + rebates_more_than_bills
  let total_rebate_stamps := total_rebates * rebate_stamps
  let total_job_applications := total_rebates * job_applications_factor
  let total_job_application_stamps := total_job_applications * job_application_stamps

  total_thank_you_stamps + total_bill_stamps + total_rebate_stamps + total_job_application_stamps

theorem valerie_needs_72_stamps : total_stamps_needed = 72 :=
  by
    sorry

end valerie_needs_72_stamps_l307_307305


namespace rodney_probability_correct_guess_l307_307632

noncomputable def two_digit_integer (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

noncomputable def tens_digit (n : ℕ) : Prop :=
  (n / 10 = 7 ∨ n / 10 = 8 ∨ n / 10 = 9)

noncomputable def units_digit_even (n : ℕ) : Prop :=
  (n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8)

noncomputable def greater_than_seventy_five (n : ℕ) : Prop := n > 75

theorem rodney_probability_correct_guess (n : ℕ) :
  two_digit_integer n →
  tens_digit n →
  units_digit_even n →
  greater_than_seventy_five n →
  (∃ m, m = 1 / 12) :=
sorry

end rodney_probability_correct_guess_l307_307632


namespace find_LN_l307_307263

noncomputable def LM : ℝ := 9
noncomputable def sin_N : ℝ := 3 / 5
noncomputable def LN : ℝ := 15

theorem find_LN (h₁ : sin_N = 3 / 5) (h₂ : LM = 9) (h₃ : sin_N = LM / LN) : LN = 15 :=
by
  sorry

end find_LN_l307_307263


namespace p3_mp_odd_iff_m_even_l307_307197

theorem p3_mp_odd_iff_m_even (p m : ℕ) (hp : p % 2 = 1) : (p^3 + m * p) % 2 = 1 ↔ m % 2 = 0 := sorry

end p3_mp_odd_iff_m_even_l307_307197


namespace u_n_divisible_by_3_power_l307_307592

def largest_divisor (n : ℕ) : ℕ := 
  if n == 1 then 1 else 
  let divisors := finset.filter (λ d, d ≠ n ∧ n % d = 0) (finset.range (n + 1)) in
  finset.max' divisors (finset.nonempty_of_mem (finset.mem_filter.mpr ⟨1, n.div_pos (nat.pos_of_ne_zero (ne_of_gt (nat.one_lt_iff_ne_zero.mpr (nat.ne_of_gt (nat.lt_succ_self n)))), nat.dvd_one⟩))

def sequence_u (u0 : ℕ) : ℕ → ℕ
| 0     := u0
| (n+1) := let prev := sequence_u n in prev + largest_divisor prev

theorem u_n_divisible_by_3_power (u0 : ℕ) (h : u0 > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → 3^2019 ∣ sequence_u u0 n :=
sorry

end u_n_divisible_by_3_power_l307_307592


namespace derivative_at_zero_l307_307484
noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem derivative_at_zero : (deriv f 0) = -120 :=
by
  -- The proof is omitted
  sorry

end derivative_at_zero_l307_307484


namespace garin_homework_pages_l307_307858

theorem garin_homework_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
    pages_per_day = 19 → 
    days = 24 → 
    total_pages = pages_per_day * days → 
    total_pages = 456 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end garin_homework_pages_l307_307858


namespace production_difference_correct_l307_307019

variable (w t M T : ℕ)

-- Condition: w = 2t
def condition_w := w = 2 * t

-- Widgets produced on Monday
def widgets_monday := M = w * t

-- Widgets produced on Tuesday
def widgets_tuesday := T = (w + 5) * (t - 3)

-- Difference in production
def production_difference := M - T = t + 15

theorem production_difference_correct
  (h1 : condition_w w t)
  (h2 : widgets_monday M w t)
  (h3 : widgets_tuesday T w t) :
  production_difference M T t :=
sorry

end production_difference_correct_l307_307019


namespace partition_X_l307_307452

namespace ProofProblem

open Finset

-- Given integer n ≥ 3
variable (n : ℤ) (h_n : n ≥ 3)

-- Define the set X = {1, 2, ..., n^2 - n}
def X : Finset ℤ := (range (n^2 - n + 1)).filter (λ x, x > 0)

-- Define the condition that no subset with n elements satisfies the given inequality
def invalid_subset (s : Finset ℤ) : Prop :=
  ∃ (a : ℤ → ℤ), ∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ s ∧ a 1 < a 2 ∧ a 2 < a 3 ∧ a (n - 1) < a n ∧
    ∀ k, 2 ≤ k ∧ k < n → a k ≤ (a (k - 1) + a (k + 1)) / 2

-- Prove that X can be partitioned into two disjoint subsets such that neither contains the invalid subset
theorem partition_X : 
  ∃ (S T : Finset ℤ), S ≠ ∅ ∧ T ≠ ∅ ∧ S ∩ T = ∅ ∧ S ∪ T = X n ∧ ¬invalid_subset n S ∧ ¬invalid_subset n T :=
sorry

end ProofProblem

end partition_X_l307_307452


namespace num_five_digit_div_by_12_l307_307508

theorem num_five_digit_div_by_12 : 
  let smallest_div_12 := 1008
  let largest_div_12 := 9996
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  in total_count = 6750 := by
  let smallest_div_12 := 1008
  let largest_div_12 := 9996 
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  have : total_count = 6750 := by
    sorry
  exact this

end num_five_digit_div_by_12_l307_307508


namespace find_costs_of_A_and_B_find_price_reduction_l307_307764

-- Definitions for part 1
def cost_of_type_A_and_B (x y : ℕ) : Prop :=
  (5 * x + 3 * y = 450) ∧ (10 * x + 8 * y = 1000)

-- Part 1: Prove that x and y satisfy the cost conditions
theorem find_costs_of_A_and_B (x y : ℕ) (hx : 5 * x + 3 * y = 450) (hy : 10 * x + 8 * y = 1000) : 
  x = 60 ∧ y = 50 :=
sorry

-- Definitions for part 2
def daily_profit_condition (m : ℕ) : Prop :=
  (100 + 20 * m > 200) ∧ ((80 - m) * (100 + 20 * m) + 7000 = 10000)

-- Part 2: Prove that the price reduction m meets the profit condition
theorem find_price_reduction (m : ℕ) (hm : 100 + 20 * m > 200) (hp : (80 - m) * (100 + 20 * m) + 7000 = 10000) : 
  m = 10 :=
sorry

end find_costs_of_A_and_B_find_price_reduction_l307_307764


namespace range_of_a_l307_307611

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, ax^2 - x + (1/16)*a > 0

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - (3/2))^x < (a - (3/2))^(x + 1)

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔
  ((3/2 < a ∧ a ≤ 2) ∨ a ≥ 5/2) :=
by
  sorry

end range_of_a_l307_307611


namespace eleven_percent_greater_than_seventy_l307_307736

theorem eleven_percent_greater_than_seventy : ∀ x : ℝ, (x = 70 * (1 + 11 / 100)) → (x = 77.7) :=
by
  intro x
  intro h
  sorry

end eleven_percent_greater_than_seventy_l307_307736


namespace kayla_apples_correct_l307_307588

-- Definition of Kylie and Kayla's apples
def total_apples : ℕ := 340
def kaylas_apples (k : ℕ) : ℕ := 4 * k + 10

-- The main statement to prove
theorem kayla_apples_correct :
  ∃ K : ℕ, K + kaylas_apples K = total_apples ∧ kaylas_apples K = 274 :=
sorry

end kayla_apples_correct_l307_307588


namespace binary_to_hexadecimal_l307_307016

theorem binary_to_hexadecimal (b : String) (h : b = "1011101") :
  (bitvec.to_nat (bitvec.of_string b) = 93) ∧ (nat.to_hex (bitvec.to_nat (bitvec.of_string b)) = "5D") :=
by
  sorry

end binary_to_hexadecimal_l307_307016


namespace pyramid_height_correct_l307_307442

-- Define the given conditions
structure Pyramid (A P B C : Point) where
  PA_perpendicular_ABC : IsPerpendicular (Line P A) (Plane B C A)
  AB_perpendicular_AC : IsPerpendicular (Line B A) (Line C A)
  BA_eq_CA : Distance B A = Distance C A
  BA_eq_2PA : Distance B A = 2 * Distance P A

-- Our goal is to prove that the height from the base PBC to the apex A is sqrt(6) / 3
def height_from_base_to_apex (A P B C : Point) (pyramid : Pyramid A P B C) : Real :=
  sqrt 6 / 3

-- State the theorem
theorem pyramid_height_correct (A P B C : Point) (pyramid : Pyramid A P B C) :
  height_from_base_to_apex A P B C pyramid = sqrt 6 / 3 := by
  sorry

end pyramid_height_correct_l307_307442


namespace inequality_solution_l307_307601

theorem inequality_solution
  (f : ℝ → ℝ)
  (h_deriv : ∀ x : ℝ, deriv f x > 2 * f x)
  (h_value : f (1/2) = Real.exp 1)
  (x : ℝ)
  (h_pos : 0 < x) :
  f (Real.log x) < x^2 ↔ x < Real.exp (1/2) :=
sorry

end inequality_solution_l307_307601


namespace intersection_eq_0_to_1_l307_307431

def setM : set ℝ := { y | ∃ x : ℝ, y = x ^ 2 }
def setN : set ℝ := { y | ∃ x : ℝ, x ^ 2 + y ^ 2 = 1 }

theorem intersection_eq_0_to_1 : setM ∩ setN = { y | 0 ≤ y ∧ y ≤ 1 } := by
  sorry

end intersection_eq_0_to_1_l307_307431


namespace jean_total_jail_time_l307_307171

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l307_307171


namespace minimum_edges_to_monochromatic_triangle_l307_307716

theorem minimum_edges_to_monochromatic_triangle (n : ℕ) 
  (h_points : ∀ (S : finite_set point), S.card = 9 → ∀ (a b c : point), a ≠ b → b ≠ c → a ≠ c → collinear {a, b, c} → false)
  (h_coloring : ∀ (f : edge → bool), ∃ (u v w : point), u ≠ v → v ≠ w → u ≠ w → triangle_monochrome f u v w) :
  n = 33 :=
by
  sorry

end minimum_edges_to_monochromatic_triangle_l307_307716


namespace five_digit_divisible_by_twelve_count_l307_307515

theorem five_digit_divisible_by_twelve_count : 
  let count_four_digit_multiples_of_12 := 
      ((9996 - 1008) / 12 + 1) in
  let count_five_digit_multiples_of_12 := 
      (9 * count_four_digit_multiples_of_12) in
  count_five_digit_multiples_of_12 = 6732 :=
by
  sorry

end five_digit_divisible_by_twelve_count_l307_307515


namespace validate_propositions_l307_307746

-- Define the basic entities: lines and planes
variable (l : Line)
variable (α β γ : Plane)

-- Define the propositions
def proposition_1 : Prop := (α ⊥ γ ∧ β ⊥ γ) → α ⊥ β
def proposition_2 : Prop := (α ⊥ γ ∧ β ∥ γ) → α ⊥ β
def proposition_3 : Prop := (l ∥ α ∧ l ⊥ β) → α ⊥ β
def proposition_4 : Prop := l ∥ α → ∀ (m : Line), m ∈ α → l ∥ m

-- The proof problem: validate the correctness of propositions 2 and 3,
-- and the incorrectness of propositions 1 and 4.
theorem validate_propositions :
  (¬ proposition_1) ∧ proposition_2 ∧ proposition_3 ∧ (¬ proposition_4) :=
by
  sorry

end validate_propositions_l307_307746


namespace customer_saves_7_906304_percent_l307_307784

variable {P : ℝ} -- Define the base retail price as a variable

-- Define the percentage reductions and additions
def reduced_price (P : ℝ) : ℝ := 0.88 * P
def further_discount_price (P : ℝ) : ℝ := reduced_price P * 0.95
def price_with_dealers_fee (P : ℝ) : ℝ := further_discount_price P * 1.02
def final_price (P : ℝ) : ℝ := price_with_dealers_fee P * 1.08

-- Define the final price factor
def final_price_factor : ℝ := 0.88 * 0.95 * 1.02 * 1.08

noncomputable def total_savings (P : ℝ) : ℝ :=
  P - (final_price_factor * P)

theorem customer_saves_7_906304_percent (P : ℝ) :
  total_savings P = P * 0.07906304 := by
  sorry -- Proof to be added

end customer_saves_7_906304_percent_l307_307784


namespace eval_diff_squares_l307_307411

theorem eval_diff_squares : 81^2 - 49^2 = 4160 :=
by
  sorry

end eval_diff_squares_l307_307411


namespace letters_identity_l307_307233

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l307_307233


namespace smallest_color_count_l307_307074

-- Define the size of the grid and the condition on the colors.
def grid_size : ℕ := 2023
def color_count (n : ℕ) : Prop :=
  ∀ c : ℕ, 
  (∃ f : ℕ × ℕ → ℕ,
    (∀ i j, f (i, j) < n) ∧
    (∀ i k l s, (k < l) ∨ (s < i) → f (i, k) = f (s, l) → False) ∧
    (∃ r, ∃ c₁ c₂: ℕ, r < grid_size → r < grid_size → 
      (c₁ = 0 ∨ c₂ = grid_size - 1) ∧ 
      (∀ i, c₁ ≤ i ∧ i ≤ c₂ → f (r, i) = c)))

-- Define the main theorem to find the minimum possible value of n
theorem smallest_color_count : ∃ n, color_count n ∧ n = 338 :=
begin
  sorry
end

end smallest_color_count_l307_307074


namespace letters_identity_l307_307246

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l307_307246


namespace num_five_digit_div_by_12_l307_307510

theorem num_five_digit_div_by_12 : 
  let smallest_div_12 := 1008
  let largest_div_12 := 9996
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  in total_count = 6750 := by
  let smallest_div_12 := 1008
  let largest_div_12 := 9996 
  let num_four_digit_multiples := (largest_div_12 - smallest_div_12) / 12 + 1
  let num_ten_thousands_choices := 9
  let total_count := num_ten_thousands_choices * num_four_digit_multiples
  have : total_count = 6750 := by
    sorry
  exact this

end num_five_digit_div_by_12_l307_307510


namespace distance_proof_l307_307398

def degrees (d m s : ℝ) : ℝ := d + m/60 + s/3600

def delta_time_to_degrees (m s : ℝ) : ℝ := (m + s/60) / 60 * 15 

noncomputable def distance_between_locations 
  (lat_A_d lat_A_m : ℝ)
  (lat_B_d lat_B_m lat_B_s : ℝ)
  (time_diff_m time_diff_s : ℝ)
  (r : ℝ) : ℝ :=
  let φ₁ := degrees lat_A_d lat_A_m 0
  let φ₂ := degrees lat_B_d lat_B_m lat_B_s
  let Δt := delta_time_to_degrees time_diff_m time_diff_s
  let ψ := Real.arctan ((Real.cos(Δt)) / (Real.tan(Real.toRadians φ₁)))
  let cos_D := (Real.sin (Real.toRadians φ₁) * Real.sin (Real.toRadians (φ₂ + ψ))) / (Real.cos ψ)
  let D := Real.acos cos_D
  (2 * Math.PI * r * D) / 360 

theorem distance_proof : distance_between_locations 47 6 59 56 30 59 29 6378.3 = 1727 :=
  by
    sorry

end distance_proof_l307_307398


namespace largest_n_divides_30_factorial_l307_307042

theorem largest_n_divides_30_factorial :
  ∃ n : ℕ, (∀ m : ℕ, 18^m ∣ nat.factorial 30 → m ≤ 7) ∧ (18^7 ∣ nat.factorial 30) :=
by
  sorry

end largest_n_divides_30_factorial_l307_307042


namespace line_slope_intercept_sum_l307_307354

theorem line_slope_intercept_sum (m b : ℝ)
    (h1 : m = 4)
    (h2 : ∃ b, ∀ x y : ℝ, y = mx + b → y = 5 ∧ x = -2)
    : m + b = 17 := by
  sorry

end line_slope_intercept_sum_l307_307354


namespace find_p_if_geometric_exists_p_arithmetic_sequence_l307_307943

variable (a : ℕ → ℝ) (p : ℝ)

-- Condition 1: a_1 = 1
axiom a1_eq_1 : a 1 = 1

-- Condition 2: a_n + a_{n+1} = pn + 1
axiom a_recurrence : ∀ n : ℕ, a n + a (n + 1) = p * n + 1

-- Question 1: If a_1, a_2, a_4 form a geometric sequence, find p
theorem find_p_if_geometric (h_geometric : (a 2)^2 = (a 1) * (a 4)) : p = 2 := by
  -- Proof goes here
  sorry

-- Question 2: Does there exist a p such that the sequence {a_n} is an arithmetic sequence?
theorem exists_p_arithmetic_sequence : ∃ p : ℝ, (∀ n : ℕ, a n + a (n + 1) = p * n + 1) ∧ 
                                         (∀ m n : ℕ, a (m + n) - a n = m * p) := by
  -- Proof goes here
  exists 2
  sorry

end find_p_if_geometric_exists_p_arithmetic_sequence_l307_307943


namespace arccos_neg_half_eq_two_thirds_pi_l307_307821

theorem arccos_neg_half_eq_two_thirds_pi : 
  ∃ θ : ℝ, θ ∈ set.Icc 0 real.pi ∧ real.cos θ = -1/2 ∧ real.arccos (-1/2) = θ := 
sorry

end arccos_neg_half_eq_two_thirds_pi_l307_307821


namespace h_0_is_neg30_l307_307434

-- Given Conditions: 
variables {h : ℝ → ℝ}
axiom monic_quartic : ∃ p : polynomial ℝ, p.degree = 4 ∧ p.leading_coeff = 1 ∧ (∀ x, h x = polynomial.eval x p)
axiom h_neg2 : h (-2) = -12
axiom h_1 : h 1 = -3
axiom h_3 : h 3 = -27
axiom h_5 : h 5 = -75

-- To Prove:
theorem h_0_is_neg30 : h 0 = -30 := 
sorry

end h_0_is_neg30_l307_307434


namespace B_initial_investment_l307_307756

theorem B_initial_investment (B : ℝ) :
  let A_initial := 2000
  let A_months := 12
  let A_withdraw := 1000
  let B_advanced := 1000
  let months_before_change := 8
  let months_after_change := 4
  let total_profit := 630
  let A_share := 175
  let B_share := total_profit - A_share
  let A_investment := A_initial * A_months
  let B_investment := (B * months_before_change) + ((B + B_advanced) * months_after_change)
  (B_share / A_share = B_investment / A_investment) →
  B = 4866.67 :=
sorry

end B_initial_investment_l307_307756


namespace exists_eleven_consecutive_numbers_sum_cube_l307_307401

theorem exists_eleven_consecutive_numbers_sum_cube :
  ∃ (n k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) + (n+9) + (n+10)) = k^3 :=
by
  sorry

end exists_eleven_consecutive_numbers_sum_cube_l307_307401


namespace linear_eq_rewrite_l307_307229

theorem linear_eq_rewrite (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end linear_eq_rewrite_l307_307229


namespace largest_lambda_inequality_l307_307045

theorem largest_lambda_inequality :
  ∃ λ : ℝ, (∀ (a b c d : ℝ), (0 ≤ a) → (0 ≤ b) → (0 ≤ c) → (0 ≤ d) →
  (a^2 + b^2 + 2*c^2 + 2*d^2 ≥ 2*a*b + λ*b*d + 2*c*d)) ∧ (∀ μ : ℝ, (∀ (a b c d : ℝ), (0 ≤ a) → (0 ≤ b) → (0 ≤ c) → (0 ≤ d) →
  (a^2 + b^2 + 2*c^2 + 2*d^2 ≥ 2*a*b + μ*b*d + 2*c*d)) → μ ≤ λ) :=
begin
  use 2,
  split,
  { intros a b c d ha hb hc hd,
    calc a^2 + b^2 + 2*c^2 + 2*d^2
      : _ ≥ 2*a*b + 2*b*d + 2*c*d, 
    sorry }, -- Here we would need the actual proof details
  { intros μ hμ,
    have h := hμ 1 1 1 1 (le_refl 1) (le_refl 1) (le_refl 1) (le_refl 1),
    linarith, }
end

end largest_lambda_inequality_l307_307045


namespace letters_identity_l307_307234

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l307_307234


namespace comb_comb_l307_307807

theorem comb_comb (n1 k1 n2 k2 : ℕ) (h1 : n1 = 10) (h2 : k1 = 3) (h3 : n2 = 8) (h4 : k2 = 4) :
  (Nat.choose n1 k1) * (Nat.choose n2 k2) = 8400 := by
  rw [h1, h2, h3, h4]
  change Nat.choose 10 3 * Nat.choose 8 4 = 8400
  -- Adding the proof steps is not necessary as per instructions
  sorry

end comb_comb_l307_307807


namespace letters_identity_l307_307237

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l307_307237


namespace cos_alpha_pi_over_3_decreasing_interval_f_tangent_line_origin_l307_307103

def f (x : ℝ) := 2 * sin x * cos x

-- Question 1
theorem cos_alpha_pi_over_3 (α : ℝ) (h1 : α ∈ Ioo (π / 2) π) (h2 : f (α / 2) = 3 / 5) : 
  cos (α - π / 3) = (3 * sqrt 3 - 4) / 10 :=
sorry

-- Question 2
theorem decreasing_interval_f (k : ℤ) : 
  ∃ I : set ℝ, I = Icc (k * π + π / 4) (k * π + 3 * π / 4) ∧ 
  ∀ x ∈ I, f' x < 0 :=
sorry

-- Question 3
theorem tangent_line_origin (h : f 0 = 0) (h' : deriv f 0 = 2) : 
  ∀ x : ℝ, tangent_line f 0 x = 2 * x :=
sorry

end cos_alpha_pi_over_3_decreasing_interval_f_tangent_line_origin_l307_307103


namespace number_of_sequences_with_at_least_two_reds_l307_307403

theorem number_of_sequences_with_at_least_two_reds (n : ℕ) (h : n ≥ 2) :
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2 :=
by
  intros
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  show T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2
  sorry

end number_of_sequences_with_at_least_two_reds_l307_307403


namespace volume_ratio_l307_307373

-- Given conditions
def radius_from_diameter (d : ℝ) : ℝ := d / 2
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
def cone_volume (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

-- Main statement to prove
theorem volume_ratio (d h : ℝ) (d = 8) (h = 10) : 
  let r := radius_from_diameter d in
  let VA := cylinder_volume r h in
  let VF := cone_volume r h in
  VA / VF = 3 :=
by
  sorry

end volume_ratio_l307_307373


namespace problem_l307_307459

-- Define the polynomial x^2 - 3x + 1
noncomputable def polynomial : Polynomial ℝ := X^2 - 3 * X + 1

-- Define α and β as the roots of the polynomial
variable (α β : ℝ)

-- Conditions stating that α and β are roots of the polynomial
axiom root1 : polynomial.eval α = 0
axiom root2 : polynomial.eval β = 0

-- Using these roots, we need to prove that 3α^3 + 7β^4 = 448
theorem problem :
    3 * α ^ 3 + 7 * β ^ 4 = 448 :=
begin
  sorry
end

end problem_l307_307459


namespace letters_identity_l307_307247

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l307_307247


namespace red_ball_higher_probability_l307_307678

theorem red_ball_higher_probability (prob_red_bin : (k : ℕ) → ℝ) (prob_blue_bin : (k : ℕ) → ℝ)
  (h_red : ∀ k, prob_red_bin k = 1 / 2^k)
  (h_blue : ∀ k, prob_blue_bin k = 1 / 3^k) :
  let P := ∑ k, (prob_red_bin k) * (prob_blue_bin k)
  in (P = 1/5) → (∑ k, (prob_red_bin k * (∑ n ≤ k, prob_blue_bin n))) = 2/5 :=
by
  sorry

end red_ball_higher_probability_l307_307678


namespace haircut_cost_l307_307355

variable (counterfeit_value real_value barber_lost_change : ℕ)

-- Conditions
def man_pays_counterfeit : Prop := counterfeit_value = 0
def barber_gets_real_money_from_flower_shop : Prop := real_value = 20
def barber_gives_change : Prop := barber_lost_change = 5
def barber_replaces_counterfeit_to_flower_shop : Prop := real_value = 20

-- Theorem statement
theorem haircut_cost :
  man_pays_counterfeit ∧ 
  barber_gets_real_money_from_flower_shop ∧ 
  barber_gives_change ∧
  barber_replaces_counterfeit_to_flower_shop →
  (real_value - barber_lost_change) = 20 :=
by
  sorry

end haircut_cost_l307_307355


namespace highway_speed_l307_307761

theorem highway_speed 
  (local_distance : ℝ) (local_speed : ℝ)
  (highway_distance : ℝ) (avg_speed : ℝ)
  (h_local : local_distance = 90) 
  (h_local_speed : local_speed = 30)
  (h_highway : highway_distance = 75)
  (h_avg : avg_speed = 38.82) :
  ∃ v : ℝ, v = 60 := 
sorry

end highway_speed_l307_307761


namespace candy_count_in_third_set_l307_307700

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l307_307700


namespace modular_inverse_of_34_mod_35_l307_307048

theorem modular_inverse_of_34_mod_35 : ∃ a : ℤ, 0 ≤ a ∧ a < 35 ∧ (34 * a) % 35 = 1 ∧ a = 34 :=
by
  -- Conditions
  have h : 34 % 35 = -1 % 35, by nat_mod_eq_neg,
  -- Proofs leading to conclusion a = 34 omitted
  use 34
  split; linarith
  split; linarith
  split
  -- modulo conditions
  calc 34 * 34 % 35 = 34 % 35
  sorry

end modular_inverse_of_34_mod_35_l307_307048


namespace finitely_many_elements_sum_conditions_l307_307606

noncomputable theory

open Nat

theorem finitely_many_elements_sum_conditions
  (A : Set ℕ) (hA_inf : ∀ p : ℕ, Prime p ∧ ¬(p ∣ n) → ∃ᶠ a in cofinite, a ∈ A ∧ ¬(p ∣ a))
  (n : ℕ) (hn : n > 1) :
  ∀ (m : ℕ), m > 1 ∧ gcd m n = 1 →
  ∃ finite_set : Finset ℕ, 
    (∀ s ∈ finite_set, s ∈ A) ∧ 
    let S := finite_set.sum id in 
    S % m = 1 ∧ S % n = 0 :=
begin
  sorry
end

end finitely_many_elements_sum_conditions_l307_307606


namespace inverse_value_l307_307105

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value :
  g (-3) = -103 :=
by
  sorry

end inverse_value_l307_307105


namespace point_B_not_on_curve_C_l307_307481

theorem point_B_not_on_curve_C {a : ℝ} : 
  ¬ ((2 * a) ^ 2 + (4 * a) ^ 2 + 6 * a * (2 * a) - 8 * a * (4 * a) = 0) :=
by 
  sorry

end point_B_not_on_curve_C_l307_307481


namespace find_rate_per_kg_mangoes_l307_307000

noncomputable def rate_per_kg_mangoes
  (cost_grapes_rate : ℕ)
  (quantity_grapes : ℕ)
  (quantity_mangoes : ℕ)
  (total_paid : ℕ)
  (rate_grapes : ℕ)
  (rate_mangoes : ℕ) :=
  total_paid = (rate_grapes * quantity_grapes) + (rate_mangoes * quantity_mangoes)

theorem find_rate_per_kg_mangoes :
  rate_per_kg_mangoes 70 8 11 1165 70 55 :=
by
  sorry

end find_rate_per_kg_mangoes_l307_307000


namespace weights_partition_l307_307063

theorem weights_partition (weights : Finset ℕ) (h_weights : weights = (Finset.range 82).map (λ n, n^2)) :
  ∃ (A B C : Finset ℕ), A.card = 27 ∧ B.card = 27 ∧ C.card = 27 ∧
  A ∪ B ∪ C = weights ∧
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id :=
sorry

end weights_partition_l307_307063


namespace fibonable_count_l307_307360

-- Definition for a number being 5-fibonable.
def is_fibonable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ ∀ (d : ℕ), d ∈ (n.digits 10) → d ∈ [1, 2, 3, 5, 8]

-- The property we're proving.
theorem fibonable_count : { n : ℕ | is_fibonable n ∧ n < 1000 }.to_finset.card = 31 :=
by sorry

end fibonable_count_l307_307360


namespace count_pairs_l307_307181

theorem count_pairs (X : Set ℕ) (A B : Set ℕ) :
  X = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} →
  A ⊆ X →
  B ⊆ X →
  A ≠ B →
  A ∩ B = {5, 7, 8} →
  ∃ n : ℕ, n = 2186 :=
by
  intro hX hA hB hNeq hInt
  use 2186
  sorry

end count_pairs_l307_307181


namespace geom_sequence_sum_l307_307475

theorem geom_sequence_sum (n : ℕ) (a : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 4 ^ n + a) : 
  a = -1 := 
by
  sorry

end geom_sequence_sum_l307_307475


namespace sum_values_l307_307379

def v (x : Real) : Real := 2 * Real.cos x - x

theorem sum_values :
  v (-3) + v (-1) + v (1) + v (3) = 4 * Real.cos 3 + 4 * Real.cos 1 :=
by
  sorry

end sum_values_l307_307379


namespace proposition_A_proposition_B_proposition_C_l307_307060

variables {a b c : ℝ}

theorem proposition_A (h : b * c^2 < a * c^2) : b < a :=
sorry

theorem proposition_B (h1 : a^3 > b^3) (h2 : a * b < 0) : 1 / a > 1 / b :=
sorry

theorem proposition_C (h : a > b ∧ b > c ∧ c > 0) : a / b > (a + c) / (b + c) :=
sorry

example : proposition_A ∧ proposition_B ∧ proposition_C :=
⟨proposition_A, proposition_B, proposition_C⟩  

end proposition_A_proposition_B_proposition_C_l307_307060


namespace range_of_g_l307_307051

noncomputable def g (x : ℝ) : ℝ :=
  (sin x)^4 + 3*(sin x)^3 + 5*(sin x)^2 + 4*(sin x) + 3*(cos x)^2 - 9 / (sin x - 1)

theorem range_of_g :
  ∀ x : ℝ, sin x ≠ 1 → 2 ≤ g x ∧ g x < 15 :=
sorry

end range_of_g_l307_307051


namespace sum_of_reciprocals_l307_307616

-- Defining the polynomial
def poly : Polynomial ℝ := 7 * Polynomial.X^2 + 4 * Polynomial.X + 9

-- Vieta's formulas give us the sum and product of the roots
def sum_roots (p : Polynomial ℝ) : ℝ := -(Polynomial.coeff p 1) / (Polynomial.coeff p 2)
def prod_roots (p : Polynomial ℝ) : ℝ := (Polynomial.coeff p 0) / (Polynomial.coeff p 2)

-- Defining the reciprocals of the roots
def alpha (a b : ℝ) := 1 / a
def beta (a b : ℝ) := 1 / b

-- The proof statement
theorem sum_of_reciprocals :
  let a := sum_roots poly in
  let b := prod_roots poly in
  alpha a b + beta a b = -4 / 9 :=
by
  let a := sum_roots poly
  let b := prod_roots poly
  have h1 : a = -4 / 7 := by sorry
  have h2 : b = 9 / 7 := by sorry
  sorry

end sum_of_reciprocals_l307_307616


namespace box_height_l307_307353

variables (length width : ℕ) (cube_volume cubes total_volume : ℕ)
variable (height : ℕ)

theorem box_height :
  length = 12 →
  width = 16 →
  cube_volume = 3 →
  cubes = 384 →
  total_volume = cubes * cube_volume →
  total_volume = length * width * height →
  height = 6 :=
by
  intros
  sorry

end box_height_l307_307353


namespace power_mod_remainder_l307_307318

theorem power_mod_remainder : (3^20) % 7 = 2 :=
by {
  -- condition: 3^6 ≡ 1 (mod 7)
  have h1 : (3^6) % 7 = 1 := by norm_num,
  -- we now use this to show 3^20 ≡ 2 (mod 7)
  calc
    (3^20) % 7 = ((3^6)^3 * 3^2) % 7 : by norm_num
          ... = (1^3 * 3^2) % 7       : by rw [←nat.modeq.modeq_iff_dvd, h1]
          ... =  (3^2) % 7            : by norm_num
          ... = 2                    : by norm_num
}

end power_mod_remainder_l307_307318


namespace continuous_at_1_l307_307228

theorem continuous_at_1 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → |(-4 * x^2 - 6) - (-10)| < ε :=
by
  sorry

end continuous_at_1_l307_307228


namespace system_solution_l307_307058

theorem system_solution (m n : ℝ) (h1 : -2 * m * 5 + 5 * 2 = 15) (h2 : 5 + 7 * n * 2 = 14) :
  ∃ (a b : ℝ), (-2 * m * (a + b) + 5 * (a - 2 * b) = 15) ∧ ((a + b) + 7 * n * (a - 2 * b) = 14) ∧ (a = 4) ∧ (b = 1) :=
by
  -- The proof is intentionally omitted
  sorry

end system_solution_l307_307058


namespace min_max_argument_l307_307421

noncomputable def min_max_arg_y (b y : ℂ) (r : ℝ) (φ : ℝ) : Prop :=
  abs (b * y + 1 / y) = Real.sqrt 2 ∧
  y = r * (Complex.cos φ + Complex.sin φ * Complex.I) ∧
  (φ = Real.pi / 4 ∨ φ = 7 * Real.pi / 4)

theorem min_max_argument (b y : ℂ) (r : ℝ) (φ : ℝ) :
  abs (b * y + 1 / y) = Real.sqrt 2 →
  y = r * (Complex.cos φ + Complex.sin φ * Complex.I) →
  φ = Real.pi / 4 ∨ φ = 7 * Real.pi / 4 :=
begin
  sorry
end

end min_max_argument_l307_307421


namespace hilt_combinations_l307_307623

def num_combinations : Nat := 12

theorem hilt_combinations :
  {P E N : Nat // 5 * P + 10 * E + 20 * N = 50}.card = num_combinations := 
by sorry

end hilt_combinations_l307_307623


namespace profit_percentage_l307_307358

theorem profit_percentage (CP SP : ℝ) (h₁ : CP = 400) (h₂ : SP = 560) : 
  ((SP - CP) / CP) * 100 = 40 := by 
  sorry

end profit_percentage_l307_307358


namespace product_value_4_l307_307965

noncomputable def product_of_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ℝ :=
(x - 1) * (y - 1)

theorem product_value_4 (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ∃ v : ℝ, product_of_values x y h = v ∧ v = 4 :=
sorry

end product_value_4_l307_307965


namespace jenny_questions_wrong_l307_307583

variable (j k l m : ℕ)

theorem jenny_questions_wrong
  (h1 : j + k = l + m)
  (h2 : j + m = k + l + 6)
  (h3 : l = 7) : j = 10 := by
  sorry

end jenny_questions_wrong_l307_307583


namespace unique_reconstruction_l307_307218

-- Definition of the sums on the edges given the face values
variables (a b c d e f : ℤ)

-- The 12 edge sums
variables (e₁ e₂ e₃ e₄ e₅ e₆ e₇ e₈ e₉ e₁₀ e₁₁ e₁₂ : ℤ)
variables (h₁ : e₁ = a + b) (h₂ : e₂ = a + c) (h₃ : e₃ = a + d) 
          (h₄ : e₄ = a + e) (h₅ : e₅ = b + c) (h₆ : e₆ = b + f) 
          (h₇ : e₇ = c + f) (h₈ : e₈ = d + f) (h₉ : e₉ = d + e)
          (h₁₀ : e₁₀ = e + f) (h₁₁ : e₁₁ = b + d) (h₁₂ : e₁₂ = c + e)

-- Proving that the face values can be uniquely determined given the edge sums
theorem unique_reconstruction :
  ∃ a' b' c' d' e' f' : ℤ, 
    (e₁ = a' + b') ∧ (e₂ = a' + c') ∧ (e₃ = a' + d') ∧ (e₄ = a' + e') ∧ 
    (e₅ = b' + c') ∧ (e₆ = b' + f') ∧ (e₇ = c' + f') ∧ (e₈ = d' + f') ∧ 
    (e₉ = d' + e') ∧ (e₁₀ = e' + f') ∧ (e₁₁ = b' + d') ∧ (e₁₂ = c' + e') ∧ 
    (a = a') ∧ (b = b') ∧ (c = c') ∧ (d = d') ∧ (e = e') ∧ (f = f') := by
  sorry

end unique_reconstruction_l307_307218


namespace textbook_weight_difference_l307_307585

variable (chemWeight : ℝ) (geomWeight : ℝ)

def chem_weight := chemWeight = 7.12
def geom_weight := geomWeight = 0.62

theorem textbook_weight_difference : chemWeight - geomWeight = 6.50 :=
by
  sorry

end textbook_weight_difference_l307_307585


namespace child_picks_3_toys_to_have_0_5_probability_l307_307757

theorem child_picks_3_toys_to_have_0_5_probability :
  ∃ (n : ℕ), (n ≤ 4) ∧ (∃ (P : ℚ), P = (Nat.choose 2 (n - 2)).to_rational / (Nat.choose 4 n).to_rational) ∧ P = 0.5 → n = 3 :=
by
  sorry

end child_picks_3_toys_to_have_0_5_probability_l307_307757


namespace monotonicity_intervals_range_of_a_l307_307057

noncomputable def f (x a b : ℝ) : ℝ := - (1 / 3) * x ^ 3 + 2 * a * x ^ 2 - 3 * a ^ 2 * x + b

def f_prime (x a : ℝ) : ℝ := -(x - 3 * a) * (x - a)

theorem monotonicity_intervals (a b : ℝ) (h : 0 < a ∧ a < 1) :
    (∀ x, a < x ∧ x < 3 * a → f_prime x a > 0) ∧
    (∀ x, x < a → f_prime x a < 0) ∧
    (∀ x, x > 3 * a → f_prime x a < 0) ∧
    f (3 * a) a b = b ∧
    f a a b = - (4 / 3) * a ^ 3 + b :=
begin
  sorry
end

def abs_le (x a : ℝ) := abs (f_prime x a) ≤ a

theorem range_of_a (a : ℝ) (h : 0 < a ∧ a < 1) :
    (∀ x, a + 1 ≤ x ∧ x ≤ a + 2 → abs_le x a) → (4 / 5) ≤ a ∧ a < 1 :=
begin
  sorry
end

end monotonicity_intervals_range_of_a_l307_307057


namespace no_knight_cycle_on_5x5_l307_307578

-- Definitions of board size, knight's move and colors
def is_possible_knight_cycle (board_size : ℕ) (knight_moves : (ℕ × ℕ) → (ℕ × ℕ) → Prop) (black_squares white_squares : ℕ) : Prop :=
  ∀ (path : List (ℕ × ℕ)),
    path.length = board_size * board_size + 1 ∧
    (∀ i, i < path.length - 1 → knight_moves (path.nth_le i sorry) (path.nth_le (i + 1) sorry)) ∧
    (∃ i, i < path.length - 1 → path.nth_le i sorry = path.head sorry) →
    false

-- Specific statement for the 5x5 chessboard
theorem no_knight_cycle_on_5x5 :
  ¬ is_possible_knight_cycle 5 (λ (p1 p2 : ℕ × ℕ), 
    let (x1, y1) := p1;
        (x2, y2) := p2 in
    (abs (x1 - x2) = 2 ∧ abs (y1 - y2) = 1) ∨ (abs (x1 - x2) = 1 ∧ abs (y1 - y2) = 2)) 13 12 :=
by
  sorry

end no_knight_cycle_on_5x5_l307_307578


namespace scenario1_scenario2_scenario3_l307_307851

-- Assuming the permutation functions and concepts such as "choose" are predefined in the relevant library.

def scenario1_ways (A B C D E : Type) : ℕ := 72
def scenario2_ways (A B C D E : Type) : ℕ := 12
def scenario3_ways (A B C D E : Type) : ℕ := 78

theorem scenario1 (A B C D E : Type) (h1 : ¬ (A (list.head [_ D B])) ∧ ¬ (A (list.last [B C D E] _))) :
  scenario1_ways A B C D E = 72 :=
sorry

theorem scenario2 (A B C D E : Type) (h2 : (A (list.head [A B C D E])) ∧ (B (list.last [A B C D E] _))) :
  scenario2_ways A B C D E = 12 :=
sorry

theorem scenario3 (A B C D E : Type) (h3 : ¬ (A (list.head [A B C D E])) ∧ ¬ (B (list.last [A B C D E] _))) :
  scenario3_ways A B C D E = 78 :=
sorry

end scenario1_scenario2_scenario3_l307_307851


namespace james_trip_time_l307_307580

def speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

theorem james_trip_time:
  (distance / speed) + stop_time = 7 := 
by
  sorry

end james_trip_time_l307_307580


namespace sum_of_reciprocals_of_shifted_roots_l307_307390

noncomputable def cubic_poly (x : ℝ) := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) 
  (ha : cubic_poly a = 0) 
  (hb : cubic_poly b = 0) 
  (hc : cubic_poly c = 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_bounds_a : 0 < a ∧ a < 1)
  (h_bounds_b : 0 < b ∧ b < 1)
  (h_bounds_c : 0 < c ∧ c < 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 4 / 3 := 
sorry

end sum_of_reciprocals_of_shifted_roots_l307_307390


namespace remainder_3_pow_20_mod_7_l307_307316

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end remainder_3_pow_20_mod_7_l307_307316


namespace five_digit_divisible_by_twelve_count_l307_307513

theorem five_digit_divisible_by_twelve_count : 
  let count_four_digit_multiples_of_12 := 
      ((9996 - 1008) / 12 + 1) in
  let count_five_digit_multiples_of_12 := 
      (9 * count_four_digit_multiples_of_12) in
  count_five_digit_multiples_of_12 = 6732 :=
by
  sorry

end five_digit_divisible_by_twelve_count_l307_307513


namespace smallest_largest_interesting_l307_307206

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := toString n in s = s.reverse

def is_interesting (n : ℕ) : Prop :=
  is_palindrome n ∧ is_palindrome (n + 2023)

theorem smallest_largest_interesting :
  ∃ n₁ n₂ : ℕ, is_interesting n₁ ∧ is_interesting n₂ ∧
               (∀ m : ℕ, is_interesting m → n₁ ≤ m) ∧
               (∀ m : ℕ, is_interesting m → m ≤ n₂) ∧
               n₁ = 969 ∧ n₂ = 8778 :=
by
  sorry

end smallest_largest_interesting_l307_307206


namespace f_succ_l307_307602

def f (n : ℕ) : ℝ := ∑ i in finset.range (3 * n), (1 : ℝ) / (i + 1)

theorem f_succ (k : ℕ) (hk : 0 < k) : 
  f (k + 1) = f k + (1 / (3 * k) : ℝ) + (1 / (3 * k + 1) : ℝ) + (1 / (3 * k + 2) : ℝ) :=
by 
  sorry

end f_succ_l307_307602


namespace tangent_line_equation_l307_307100

noncomputable def f (f'₀ : ℝ) (c : ℝ) : ℝ → ℝ := λ x, f'₀ * x^2 - 2 * x + c * Real.log x

theorem tangent_line_equation : 
  ∃ f'₀ : ℝ, 
  ∃ c : ℝ, 
  ∀ x : ℝ, 
  c = 1 ∧ 
  f'₀ = 3 ∧
  let f' := (λ x, 2 * f'₀ * x - 2 + c / x) in 
  let f := (λ x, f'₀ * x^2 - 2 * x + c * Real.log x) in 
  (5 * 1 - (6 * 1 - 2 + 1 / 1) * 1 - f 1) = 0 :=
begin
  sorry,
end

end tangent_line_equation_l307_307100


namespace pure_imaginary_complex_number_l307_307538

-- Define the imaginary unit and the statement of the problem.
def i : ℂ := complex.I

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem pure_imaginary_complex_number (a : ℝ) (h : is_pure_imaginary ((a + i) / (1 - i))) : a = 1 :=
by
  sorry

end pure_imaginary_complex_number_l307_307538


namespace domain_of_f_l307_307311

noncomputable def f (x : ℝ) : ℝ := logBase 2 (logBase 3 (logBase 4 (logBase 6 x)))

theorem domain_of_f : ∀ x : ℝ, f x ∈ (1296, ∞) ↔ x > 1296 :=
by
  sorry

end domain_of_f_l307_307311


namespace even_abundant_numbers_lt_30_eq_4_l307_307117

def is_abundant (n : ℕ) : Prop :=
  (∑ d in (Finset.filter (λ d, d ∣ n) (Finset.range n)), d) > n

noncomputable def count_even_abundant_numbers_lt_30 : ℕ :=
  Finset.card $ Finset.filter (λ n, n % 2 = 0 ∧ is_abundant n) (Finset.range 30)

theorem even_abundant_numbers_lt_30_eq_4 : count_even_abundant_numbers_lt_30 = 4 := 
by
  sorry

end even_abundant_numbers_lt_30_eq_4_l307_307117


namespace distribute_teachers_l307_307300

theorem distribute_teachers :
  let math_teachers := 3
  let lang_teachers := 3 
  let schools := 2
  let teachers_each_school := 3
  let distribution_plans := 
    (math_teachers.choose 2) * (lang_teachers.choose 1) + 
    (math_teachers.choose 1) * (lang_teachers.choose 2)
  distribution_plans = 18 := 
by
  sorry

end distribute_teachers_l307_307300


namespace beth_finds_packs_l307_307380

theorem beth_finds_packs (initial_crayons : ℝ) (people : ℕ) (total_crayons_after : ℝ) (beth_total_crayons_after : ℝ) : 
  initial_crayons = 4 → people = 10 → total_crayons_after = 4 / 10 → beth_total_crayons_after = 6.4 → 
  ∃ found_crayons, beth_total_crayons_after = total_crayons_after + found_crayons ∧ found_crayons = 6 :=
by
  intros h1 h2 h3 h4
  use beth_total_crayons_after - total_crayons_after
  rw [h4, h3]
  norm_num
  split
  sorry

end beth_finds_packs_l307_307380


namespace cat_mouse_position_after_307_moves_l307_307340

-- Definitions based on the conditions
def cat_position (n : ℕ) : Fin 6 := ⟨n % 6, Nat.mod_lt n (by norm_num)⟩
def mouse_position (n : ℕ) : Fin 12 := ⟨n % 12, Nat.mod_lt n (by norm_num)⟩

-- Problem Statement
theorem cat_mouse_position_after_307_moves :
  cat_position 307 = 0 ∧ mouse_position 307 = 6 :=
begin
  -- Use sorry to bypass the proof
  sorry
end

end cat_mouse_position_after_307_moves_l307_307340


namespace focus_of_parabola_tangent_to_circle_directrix_l307_307091

theorem focus_of_parabola_tangent_to_circle_directrix :
  ∃ p : ℝ, p > 0 ∧
  (∃ (x y : ℝ), x ^ 2 + y ^ 2 - 6 * x - 7 = 0 ∧
  ∀ x y : ℝ, y ^ 2 = 2 * p * x → x = -p) →
  (1, 0) = (p, 0) :=
by
  sorry

end focus_of_parabola_tangent_to_circle_directrix_l307_307091


namespace area_identity_tg_cos_l307_307260

variable (a b c α β γ : Real)
variable (s t : Real) (area_of_triangle : Real)

-- Assume t is the area of the triangle and s is the semiperimeter
axiom area_of_triangle_eq_heron :
  t = Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Assume trigonometric identities for tangents and cosines of half-angles
axiom tg_half_angle_α : Real.tan (α / 2) = Real.sqrt ((s - b) * (s - c) / (s * (s - a)))
axiom tg_half_angle_β : Real.tan (β / 2) = Real.sqrt ((s - c) * (s - a) / (s * (s - b)))
axiom tg_half_angle_γ : Real.tan (γ / 2) = Real.sqrt ((s - a) * (s - b) / (s * (s - c)))

axiom cos_half_angle_α : Real.cos (α / 2) = Real.sqrt (s * (s - a) / (b * c))
axiom cos_half_angle_β : Real.cos (β / 2) = Real.sqrt (s * (s - b) / (c * a))
axiom cos_half_angle_γ : Real.cos (γ / 2) = Real.sqrt (s * (s - c) / (a * b))

theorem area_identity_tg_cos :
  t = s^2 * Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) ∧
  t = (a * b * c / s) * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) :=
by
  sorry

end area_identity_tg_cos_l307_307260


namespace polynomial_sequence_conditions_l307_307662

-- Definitions of the polynomial sequence
def p (n : ℕ) : Polynomial ℝ :=
  match n with
  | 0 => 0
  | 1 => Polynomial.X
  | 2 => Polynomial.X ^ 2 - 2
  | n + 2 => Polynomial.X * p (n + 1) - p n

-- Theorem statement: proving the conditions for the polynomial sequence
theorem polynomial_sequence_conditions (n m : ℕ) (h1 : n > 1) (h2 : m > 1) (h3 : n ≠ m) : 
  p 2 = Polynomial.X ^ 2 - 2 ∧ 
  (∀ i j : ℕ, 1 < i → 1 < j → i < j → p i (p j Polynomial.X) = p j (p i Polynomial.X)) ∧
  (∀ i : ℕ, Polynomial.degree (p i) = i) :=
sorry

end polynomial_sequence_conditions_l307_307662


namespace infinitely_many_n_neq_l307_307854

theorem infinitely_many_n_neq
  (a b : ℤ)
  (ha : a > 1)
  (hb : b > 1) :
  ∃ᶠ n in Filter.at_top, ∀ m t : ℤ, m > 0 → t > 0 → EulerTotient (a^n - 1) ≠ b^m - b^t :=
begin
  sorry
end

end infinitely_many_n_neq_l307_307854


namespace coefficient_correct_l307_307309

noncomputable theory

open BigOperators

def coeff_x3y7_in_expansion : ℚ :=
  let binom := nat.choose 10 3
  let fraction1 := (4/7 : ℚ) ^ 3
  let fraction2 := (-2/3 : ℚ) ^ 7
  binom * fraction1 * fraction2

theorem coefficient_correct :
  coeff_x3y7_in_expansion = -983040 / 746649 :=
by
  sorry

end coefficient_correct_l307_307309


namespace sqrt_meaningful_iff_l307_307138

theorem sqrt_meaningful_iff (x : ℝ) : (∃ (y : ℝ), y = sqrt (x - 6)) ↔ x ≥ 6 :=
by
  sorry

end sqrt_meaningful_iff_l307_307138


namespace right_triangle_angles_l307_307932

noncomputable def right_triangle_of_leg_half_hypotenuse_angles
  (a b c : ℝ) (α β γ : ℝ) : Prop :=
  (a = c / 2) ∧ (b = c * real.sqrt 3 / 2) ∧ (α = 30) ∧ (β = 60) ∧ (γ = 90) ∧ (α + β + γ = 180)

theorem right_triangle_angles
  (c : ℝ) (h : c > 0) :
  ∃ α β γ : ℝ, right_triangle_of_leg_half_hypotenuse_angles (c / 2) (c * real.sqrt 3 / 2) c α β γ :=
begin
  sorry
end

end right_triangle_angles_l307_307932


namespace quadratic_min_value_l307_307495

theorem quadratic_min_value (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : 
  (∀ x ∈ Set.Icc (-(b / (2 * a))) ((2 * a - b) / (2 * a)), 
  f x ≤ (4 * a^2 + 4 * a * c - b^2) / (4 * a)) 
  ∧ (∀ x ∈ Set.Icc (-(b / (2 * a))) ((2 * a - b) / (2 * a)), 
  f x ≠ (4 * a * c - b^2) / (4 * a))) :
  (∀ x ∈ Set.Icc (-(b / (2 * a))) ((2 * a - b) / (2 * a)), 
  f x = (4 * a^2 + 4 * a * c - b^2) / (4 * a)) :=
sorry

end quadratic_min_value_l307_307495


namespace math_problem_l307_307867

noncomputable def parabola_eq (p : ℝ) := (λ x y, x^2 = 2 * p * y)
noncomputable def line_eq (k : ℝ) := (λ x y, y = k * x + 1)
noncomputable def circle_eq (cx cy r : ℝ) := (λ x y, (x - cx)^2 + (y - cy)^2 = r^2)

theorem math_problem (p > 0) (A B F N Q : ℝ × ℝ) (k : ℝ) :
  parabola_eq p = (λ x y, x^2 = 4 * y) ∧
  line_eq k = (λ x y, x - 2 * y + 2 = 0) ∧
  circle_eq 1 (3/2) (5/2) = (λ x y, (x - 1)^2 + (y - (3/2))^2 = (5/2)^2) :=
by
  sorry

end math_problem_l307_307867


namespace no_common_roots_of_trinomials_l307_307428

theorem no_common_roots_of_trinomials 
  (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ∀ x, ¬(x^2 + b * x + c = 0 ∧ x^2 + a * x + d = 0) :=
by
  intro x
  assume h
  cases h with h₁ h₂
  have h₅ : (b - a) * x = (d - c) := sorry
  have h₆ : (b - a) > 0 := sorry
  have h₇ : (d - c) > 0 := sorry
  have h₈ : x = (d - c) / (b - a) := sorry
  have h₉ : (d - c) / (b - a) > 0 := sorry
  have h₀ : ¬ (x > 0) := sorry
  contradiction

end no_common_roots_of_trinomials_l307_307428


namespace bowling_ball_weight_l307_307408

theorem bowling_ball_weight (b c : ℕ) (h1 : 8 * b = 4 * c) (h2 : 3 * c = 108) : b = 18 := 
by 
  sorry

end bowling_ball_weight_l307_307408


namespace arccos_neg_half_l307_307820

-- Defining the problem in Lean 4
theorem arccos_neg_half : 
  ∃ θ ∈ set.Icc 0 Real.pi, Real.arccos (-1 / 2) = θ ∧ Real.cos θ = -1 / 2 := 
by
  use Real.pi * 2 / 3
  split
  { sorry } -- Proof that θ is in [0, π]
  { split
    { sorry } -- Proof that θ = arccos(-1 / 2)
    { sorry } -- Proof that cos(θ) = -1/2
  }


end arccos_neg_half_l307_307820


namespace hyperbola_perimeter_proof_l307_307458

noncomputable def hyperbola_perimeter_problem : Prop :=
∃ (F1 F2 P : ℝ × ℝ) (a b : ℝ),
  a = 3 ∧
  b = real.sqrt 7 ∧
  F1 = (-4, 0) ∧  -- These coordinates can be derived based on standard position of the hyperbola
  F2 = (4, 0) ∧  -- As c = 4, Foci at (-c, 0) and (c, 0)
  (P.1 = 4 + ε ∨ P.1 = 4 - ε) ∧  -- P is on the right branch of hyperbola
  (0 ≤ ε) ∧
  abs (real.sqrt ((P.1 - (-4))^2 + P.2^2)) = 8 ∧
  (abs (real.sqrt ((P.1 - 4)^2 + P.2^2)) = 2) ∧  -- deduced from |PF1| - |PF2| = 6
  abs (real.sqrt ((-4 - 4)^2 + 0^2)) = 8 →   -- |F1F2|
  8 + 8 + 2 = 18

theorem hyperbola_perimeter_proof : hyperbola_perimeter_problem :=
by sorry -- Proof to be completed

end hyperbola_perimeter_proof_l307_307458


namespace find_BP_l307_307224

variables (A B C D P O1 O2 : ℝ) 
variables (angle_O1PO2 : ℝ)

-- Given conditions
def conditions (AB:=8) (P_on_BD := P ∈ line_segment B D) (BP_gt_DP := BP > DP) 
               (circumcenter_O1 := O1 = midpoint A B) (circumcenter_O2 := O2 = midpoint C D) 
               (angle_135 := angle_O1PO2 = 135) : Prop :=
   (AB := 8) ∧ P_on_BD ∧ BP_gt_DP ∧ circumcenter_O1 ∧ circumcenter_O2 ∧ angle_135

-- Theorem statement, given conditions then BP == 4 * sqrt 2
theorem find_BP (h : conditions AB P_on_BD BP_gt_DP circumcenter_O1 circumcenter_O2 angle_135) :
  ∃ c: ℝ, BP = sqrt c ∧ c = 32 :=
sorry

end find_BP_l307_307224


namespace triangle_statements_correct_l307_307088

-- Definitions for the conditions in the problem
variables {A B C a b c : ℝ} {triangle_ABC : Prop}

-- Condition that in triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively
axiom triangle_property : triangle_ABC → ( ∃ (a b c : ℝ), true)

-- Proof problem statement
theorem triangle_statements_correct 
  (h1 : triangle_ABC)
  (h2 : triangle_property h1) :
  ( ( ∀ {T : Type} (a_ b_ c_ : T), a_ / (Real.sin A) = (a_ + b_ + c_) / (Real.sin A + Real.sin B + Real.sin C) ) ∧
    ( T (angle A) (T (angle B) ( T (angle C) is an oblique triangle), 
    Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C ) ∧
    ( ∃ {A B C : ℝ} [DecidableEq T1], (a_ / (Real.cos A) = b_ / (Real.cos B) ∧ b_ / (Real.cos B) = c_ / (Real.cos C)) → True )) := 
sorry

end triangle_statements_correct_l307_307088


namespace find_k_base_representation_l307_307852

theorem find_k_base_representation (k : ℕ) (h₁ : k > 0) 
  (h₂ : ∀ n, (((1 / ↑k) + ((6 : ℤ) / (↑k^2))) + 
  (1 / (↑k^3) + (6 / (↑k^4))) = 0.161616..._k)) : k = 18 :=
sorry

end find_k_base_representation_l307_307852


namespace candy_count_in_third_set_l307_307697

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l307_307697


namespace jean_jail_time_l307_307176

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l307_307176


namespace probability_of_forming_triangle_l307_307013

noncomputable def calculate_probability_of_triangle (s : finset (ℕ × ℕ)) : ℝ :=
  let total_segments := (15.choose 2) in
  let total_ways := finset.powersetLen 3 s in
  let triangle_conditions := total_ways.filter (λ t,
    let l := t.to_list in
    match l with
    | [a, b, c] := a + b > c ∧ a + c > b ∧ b + c > a
    | _ := false
    end) in
  (triangle_conditions.card : ℝ) / (total_ways.card : ℝ)

#eval calculate_probability_of_triangle (finset.univ : finset (ℕ × ℕ))

theorem probability_of_forming_triangle (prob : ℝ) :
  prob = calculate_probability_of_triangle (finset.univ : finset (ℕ × ℕ)) :=
sorry

end probability_of_forming_triangle_l307_307013


namespace log_a_properties_l307_307102

noncomputable def log_a (a x : ℝ) (h : 0 < a ∧ a < 1) : ℝ := Real.log x / Real.log a

theorem log_a_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x : ℝ, 1 < x → log_a a x h < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → log_a a x h > 0) ∧
  (¬ ∀ x1 x2 : ℝ, log_a a x1 h > log_a a x2 h → x1 > x2) ∧
  (∀ x y : ℝ, log_a a (x * y) h = log_a a x h + log_a a y h) :=
by
  sorry

end log_a_properties_l307_307102


namespace correct_function_l307_307374

-- Definitions of the functions
def f1 (x : ℝ) : ℝ := -x^2 + 2*x - 1
def f2 (x : ℝ) : ℝ := Real.cos x
def f3 (x : ℝ) : ℝ := Real.log (Real.abs (x - 1))
def f4 (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

-- Main theorem statement
theorem correct_function : 
  (∀ x y : ℝ, 1 < x → x < y → y < ∞ → f3 x < f3 y) ∧ 
  (∀ x : ℝ, f3 (-x) = f3 x) := 
sorry

end correct_function_l307_307374


namespace cubes_with_4_neighbors_l307_307430

theorem cubes_with_4_neighbors (a b c : ℕ) (h₁ : 3 < a) (h₂ : 3 < b) (h₃ : 3 < c)
  (h₄ : (a - 2) * (b - 2) * (c - 2) = 429) : 
  4 * ((a - 2) + (b - 2) + (c - 2)) = 108 := by
  sorry

end cubes_with_4_neighbors_l307_307430


namespace distance_between_intersections_l307_307829

-- Definitions of the conditions
def eq1 (x y : ℝ) : Prop := x^2 + y = 12
def eq2 (x y : ℝ) : Prop := x + y = 12

-- The distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ := ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).sqrt

-- The math proof problem:
theorem distance_between_intersections : ∃ (x1 x2 y1 y2 : ℝ), eq1 x1 y1 ∧ eq2 x1 y1 ∧ eq1 x2 y2 ∧ eq2 x2 y2 ∧ distance (x1, y1) (x2, y2) = Real.sqrt 2 :=
by
  sorry

end distance_between_intersections_l307_307829


namespace trigonometric_identity_l307_307744

theorem trigonometric_identity : 1 - 2 * sin (real.pi * 67.5 / 180)^2 = - (real.sqrt 2) / 2 :=
by
  sorry

end trigonometric_identity_l307_307744


namespace probability_journalist_A_to_group_A_l307_307274

open Nat

theorem probability_journalist_A_to_group_A :
  let group_A := 0
  let group_B := 1
  let group_C := 2
  let journalists := [0, 1, 2, 3]  -- four journalists

  -- total number of ways to distribute 4 journalists into 3 groups such that each group has at least one journalist
  let total_ways := 36

  -- number of ways to assign journalist 0 to group A specifically
  let favorable_ways := 12

  -- probability calculation
  ∃ (prob : ℚ), prob = favorable_ways / total_ways ∧ prob = 1 / 3 :=
sorry

end probability_journalist_A_to_group_A_l307_307274


namespace letters_identity_l307_307231

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l307_307231


namespace tan_theta_half_l307_307860

theorem tan_theta_half (θ : ℝ) (h : (sin θ - 1) + (sin θ - cos θ) * complex.i ∈ {z : ℂ | ∃ (x y : ℝ), z = x + y * complex.i ∧ x + y + 1 = 0}) : 
  real.tan θ = 1 / 2 :=
by
  sorry

end tan_theta_half_l307_307860


namespace geometric_sequence_b_value_l307_307888

theorem geometric_sequence_b_value :
  ∀ (a b c : ℝ),
  (a = 5 + 2 * Real.sqrt 6) →
  (c = 5 - 2 * Real.sqrt 6) →
  (b * b = a * c) →
  (b = 1 ∨ b = -1) :=
by
  intros a b c ha hc hgeometric
  sorry

end geometric_sequence_b_value_l307_307888


namespace variance_translated_l307_307927

variable {α : Type*}

noncomputable def variance (s : Finset α) (f : α → ℝ) :=
  (s.sum (λ x, (f x - s.avg f) ^ 2)) / s.card

theorem variance_translated (s : Finset α) (f : α → ℝ) (h : variance s f = 2) :
  variance s (λ x, f x + 3) = 2 :=
  by sorry

end variance_translated_l307_307927


namespace sum_of_angles_of_roots_eq_1020_l307_307002

noncomputable def sum_of_angles_of_roots : ℝ :=
  60 + 132 + 204 + 276 + 348

theorem sum_of_angles_of_roots_eq_1020 :
  (∑ θ in {60, 132, 204, 276, 348}, θ) = 1020 := by
  sorry

end sum_of_angles_of_roots_eq_1020_l307_307002


namespace divisor_sum_lt_n_squared_and_if_prime_l307_307954

theorem divisor_sum_lt_n_squared_and_if_prime (n : ℕ) (h_n : n ≥ 2) :
  let d : ℕ → ℕ := λ i, if i = 1 then 1 else if i = k then n else choose_some_divisor (i, n)
  in let S : ℕ := (finset.range k).sum (λ i, d i * d (i + 1))
  in (S < n^2) ∧ (S ∣ n^2 ↔ nat.prime n) := 
by {
  -- Define necessary variables and assumptions
  let d : ℕ → ℕ := λ i,
    if i = 1 then 1
    else if i = nat.factors n).length then n
    else nat.factors n !! (i - 1),
  let S := (list.range ((nat.factors n).length - 1)).sum (λ i, d i * d (i + 1)),

  -- We'll add "sorry" to skip the proof part
  sorry,
}

end divisor_sum_lt_n_squared_and_if_prime_l307_307954


namespace rays_form_straight_lines_l307_307059

theorem rays_form_straight_lines
  (α β : ℝ)
  (h1 : 2 * α + 2 * β = 360) :
  α + β = 180 :=
by
  -- proof details are skipped using sorry
  sorry

end rays_form_straight_lines_l307_307059


namespace imo1988_p12_l307_307609

-- Definitions
def consecutive_product (k : ℕ) : ℕ := k * (k + 1)

-- Main theorem statement
theorem imo1988_p12 (p : ℕ) (k : ℕ) 
  (hk : k ≥ 3) 
  (hp : p = consecutive_product k) :
  ¬ ∃ (x : fin p → ℤ), 
    (∑ i, (x i)^2 - (4 / (4 * p + 1)) * (∑ i, x i)^2 = 1) :=
sorry

end imo1988_p12_l307_307609


namespace bishop_configurations_are_perfect_square_l307_307624

-- Define the chessboard as a grid with specific properties
def chessboard := fin 8 × fin 8

-- Define what it means for bishops not to threaten each other
def bishop_safe (b1 b2 : chessboard) : Prop :=
  (b1.1 ≠ b2.1) ∧ (b1.2 ≠ b2.2) ∧ (b1.1 + b1.2 ≠ b2.1 + b2.2) ∧ (b1.1 - b1.2 ≠ b2.1 - b2.2)

-- Define a configuration of bishops
def bishop_configuration (b : set chessboard) : Prop :=
  ∀ b1 b2 ∈ b, b1 ≠ b2 → bishop_safe b1 b2

-- The main theorem statement: the number of maximal bishop configurations is a perfect square
theorem bishop_configurations_are_perfect_square :
  ∃ (n : ℕ), n * n = fintype.card {b : set chessboard // bishop_configuration b} :=
sorry

end bishop_configurations_are_perfect_square_l307_307624


namespace arccos_neg_half_eq_two_pi_div_three_l307_307814

theorem arccos_neg_half_eq_two_pi_div_three :
  ∀ x : ℝ, (cos x = -1 / 2 ∧ 0 ≤ x ∧ x ≤ π) → x = 2 * π / 3 :=
by
  intro x
  intro h
  sorry

end arccos_neg_half_eq_two_pi_div_three_l307_307814


namespace minimize_distance_AB_l307_307125

theorem minimize_distance_AB :
  let A := (x : ℝ) → (x, 5 - x, 2 * x - 1)
  let B := (x : ℝ) → (1, x + 2, 2 - x)
  let distance := (x : ℝ) → (A x, B x) → ℝ :=
    λ x p₁ p₂, (let (a1, a2, a3) := p₁ in let (b1, b2, b3) := p₂ in
      real.sqrt ((b1 - a1)^2 + (b2 - a2)^2 + (b3 - a3)^2))
  in (∀ x, distance x (A x) (B x) ≥ distance (8 / 7) (A (8 / 7)) (B (8 / 7)))
  sorry

end minimize_distance_AB_l307_307125


namespace all_develop_at_least_one_develop_at_most_one_develop_l307_307840

open ProbabilityTheory

noncomputable def P_A : ℝ := 1/5
noncomputable def P_B : ℝ := 1/4
noncomputable def P_C : ℝ := 1/3

axiom independence : Independent [prob_event P_A, prob_event P_B, prob_event P_C]

theorem all_develop : P_A * P_B * P_C = 1/60 := sorry

theorem at_least_one_develop : 1 - ((1 - P_A) * (1 - P_B) * (1 - P_C)) = 3/5 := sorry

theorem at_most_one_develop : 
  P_A * (1 - P_B) * (1 - P_C) + (1 - P_A) * P_B * (1 - P_C) + (1 - P_A) * (1 - P_B) * P_C + 
  (1 - P_A) * (1 - P_B) * (1 - P_C) = 5/6 := sorry

end all_develop_at_least_one_develop_at_most_one_develop_l307_307840


namespace product_negative_probability_l307_307850

noncomputable def chosen_set : Set ℤ := { -7, -3, 0, 5, 6 }

theorem product_negative_probability :
  ∑ (subset: Finset ℤ) in (chosen_set.powerset.filter (λ s, s.card = 5)), if (∃ a ∈ subset, a < 0) ∧ (∃ b ∈ subset, b > 0) then 1 else 0 
  / 
  ∑ (subset: Finset ℤ) in (chosen_set.powerset.filter (λ s, s.card = 5)), 1 = 
  (2 : ℚ) / 5 :=
by
  sorry

end product_negative_probability_l307_307850


namespace card_diff_sets_l307_307049

def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 5}

theorem card_diff_sets : (set_A \ set_B).card = 2 := by
  sorry

end card_diff_sets_l307_307049


namespace king_path_min_max_length_l307_307775

-- Defining the chessboard conditions and king's moves
structure KingPath :=
  (cells_visited : Finset (Fin 8 × Fin 8))
  (start_pos : Fin 8 × Fin 8)
  (end_pos : Fin 8 × Fin 8)
  (moves : List ((Fin 8 × Fin 8) × (Fin 8 × Fin 8)))
  (is_closed : end_pos = start_pos)
  (no_self_intersections : ∀ (m1 m2 : (Fin 8 × Fin 8) × (Fin 8 × Fin 8)), m1 ∈ moves → m2 ∈ moves → m1 ≠ m2 → ¬ (m1.1 = m2.1 ∧ m1.2 = m2.2))

-- Definitions of useful constants
noncomputable def unit_step : ℝ := 1
noncomputable def diag_step : ℝ := Real.sqrt 2

-- Length calculation for a given path
noncomputable def path_length (kp : KingPath) : ℝ :=
  kp.moves.foldl (λ acc (m : (Fin 8 × Fin 8) × (Fin 8 × Fin 8)) =>
    let dx := abs (m.1.1 - m.2.1)
    let dy := abs (m.1.2 - m.2.2)
    acc + if dx = 1 ∧ dy = 1 then diag_step else unit_step) 0

-- Minimum and Maximum length for king's path
theorem king_path_min_max_length (kp : KingPath) (hcp : |kp.cells_visited| = 64) :
  64 ≤ path_length kp ∧ path_length kp ≤ 28 + 36 * Real.sqrt 2 := by
  sorry

end king_path_min_max_length_l307_307775


namespace complex_line_product_l307_307272

noncomputable def u : ℂ := 4 * complex.I
noncomputable def v : ℂ := 2 + 2 * complex.I
noncomputable def a : ℂ := 2 + 2 * complex.I
noncomputable def b : ℂ := 2 - 2 * complex.I

theorem complex_line_product (z : ℂ) (k : ℝ) 
    (h : (z - u) / (v - u) = (complex.conj z - complex.conj u) / (complex.conj v - complex.conj u))
    (eqn : a * z + b * complex.conj z = k) : 
  a * b = 8 :=
by
  sorry

end complex_line_product_l307_307272


namespace rearrangements_wxyz_no_adjacent_alphabet_l307_307524

def is_not_adjacent (a b : Char) : Prop :=
  (a = 'w' ∧ b ≠ 'x')
  ∨ (a = 'x' ∧ b ≠ 'w' ∧ b ≠ 'y')
  ∨ (a = 'y' ∧ b ≠ 'x' ∧ b ≠ 'z')
  ∨ (a = 'z' ∧ b ≠ 'y')

def all_chars := ['w', 'x', 'y', 'z']

def valid_rearrangements : List (List Char) :=
  (all_chars.permutations.filter (λ perm, ∀ i, i < perm.length - 1 → is_not_adjacent (perm.nthLe i sorry) (perm.nthLe (i + 1) sorry)))

theorem rearrangements_wxyz_no_adjacent_alphabet :
  valid_rearrangements.length = 8 :=
  -- proof goes here
  sorry

end rearrangements_wxyz_no_adjacent_alphabet_l307_307524


namespace toby_friends_girls_count_l307_307752

noncomputable def percentage_of_boys : ℚ := 55 / 100
noncomputable def boys_count : ℕ := 33
noncomputable def total_friends : ℚ := boys_count / percentage_of_boys
noncomputable def percentage_of_girls : ℚ := 1 - percentage_of_boys
noncomputable def girls_count : ℚ := percentage_of_girls * total_friends

theorem toby_friends_girls_count : girls_count = 27 := by
  sorry

end toby_friends_girls_count_l307_307752


namespace find_angle_APN_and_length_MN_min_l307_307455

-- Definitions of all given conditions
def AX_within_angle_NAM : Prop := -- (Defining "AX within angle NAM")
  sorry

def XAM_eq_30_deg : Prop := (angle X A M) = 30
def XAN_eq_45_deg : Prop := (angle X A N) = 45
def P_on_AX_and_AP_eq_1 : Prop := (P ∈ ray A X) ∧ (dist A P = 1)
def line_through_P_perpendicular_to_AX : Prop :=
  ∃ (M N : Point), M ∈ AM ∧ N ∈ AN ∧ (perpendicular (line P M) AX) ∧ (perpendicular (line P N) AX)

-- Theorem statement for the proof problem
theorem find_angle_APN_and_length_MN_min :
  AX_within_angle_NAM →
  XAM_eq_30_deg →
  XAN_eq_45_deg →
  P_on_AX_and_AP_eq_1 →
  line_through_P_perpendicular_to_AX →
  (∃ θ, θ = arccot ((√3 - 1) / 2) ∧ MN = √(8 - 2√3)) :=
by
  sorry

end find_angle_APN_and_length_MN_min_l307_307455


namespace positive_perfect_squares_l307_307522

theorem positive_perfect_squares (M : ℕ) (hM : M = 10^8) :
  ∃ n, n = 416 ∧ ∀ (k : ℕ), k * k < M → (576 ∣ k * k) ↔ (24 ∣ k) ∧ k < 10000 := 
begin
  use 416,
  split,
  { refl },
  { intros k h_k,
    split,
    { intro h_576_div_k2,
      have h_mod_0 : 576 ∣ k * k := h_576_div_k2,
      obtain ⟨m, h_eq⟩ := h_mod_0,
      have : 24 ∣ k,
      { rw [← h_eq] at h_mod_0,
        exact nat.dvd_of_mul_right_dvd h_mod_0, },
      split,
      { exact this, },
      { exact (nat.lt_pow_self_of_lt_pow_self_of_pos k 2 6 dec_trivial hM).1, } },
    { rintro ⟨h_dvd, h_lt⟩,
      have : 24 * 24 ∣ k * k,
      { exact nat.mul_dvd_mul h_dvd h_dvd, },
      use k / 24,
      rw [← nat.mul_div_assoc _ h_dvd, nat.mul_comm],
      exact nat.div_mul_cancel h_dvd, } }
end

end positive_perfect_squares_l307_307522


namespace point_in_fourth_quadrant_l307_307986

theorem point_in_fourth_quadrant (x y : Real) (hx : x = 2) (hy : y = Real.tan 300) : 
  (0 < x) → (y < 0) → (x = 2 ∧ y = -Real.sqrt 3) :=
by
  intro hx_trans hy_trans
  -- Here you will provide statements or tactics to assist the proof if you were completing it
  sorry

end point_in_fourth_quadrant_l307_307986


namespace intersection_result_l307_307134

open Set

namespace ProofProblem

def A : Set ℝ := {x | |x| ≤ 4}
def B : Set ℝ := {x | 4 ≤ x ∧ x < 5}

theorem intersection_result : A ∩ B = {4} :=
  sorry

end ProofProblem

end intersection_result_l307_307134


namespace sin_sum_zero_cos_sum_nonneg_l307_307739

open_locale real

variable 
  (n : ℕ) (hn: n > 0)
  (X : Type*) [fintype X]
  (f : X → X)
  (hfn : ∀ x : X, (finrec id f n x) = x)
  (k : ℤ)

def m_j (j : ℕ) : ℕ := fintype.card { x : X // finrec id f j x = x }

theorem sin_sum_zero : 
  (1 / n) * ∑ j in finset.range n, (m_j f j) * real.sin (2 * ↑j * ↑k * real.pi / n) = 0 := 
sorry

theorem cos_sum_nonneg : 
  ∃ c : ℕ, (1 / n) * ∑ j in finset.range n, (m_j f j * real.cos (2 * ↑j * ↑k * real.pi / n)) = c := 
sorry

end sin_sum_zero_cos_sum_nonneg_l307_307739


namespace work_completion_in_combined_days_l307_307737

-- Definitions of conditions as parameters
variables {W : ℝ} (x y D : ℝ)
-- Define the conditions
def x_work_rate := W / 30
def y_work_rate := W / 45

-- Combined work rates when x and y work together
def combined_work_rate := x_work_rate W + y_work_rate W

-- Proof that x and y can complete the work in 18 days
theorem work_completion_in_combined_days :
  combined_work_rate W * 18 = W :=
by
  unfold x_work_rate y_work_rate combined_work_rate
  -- Unfolding defined vars for clarity
  sorry

end work_completion_in_combined_days_l307_307737


namespace solve_for_x_l307_307928

theorem solve_for_x : ∀ (x : ℝ), (x = 3 / 4) →
  3 - (1 / (4 * (1 - x))) = 2 * (1 / (4 * (1 - x))) :=
by
  intros x h
  rw [h]
  sorry

end solve_for_x_l307_307928


namespace triangle_ratio_AH_HD_l307_307164

noncomputable def ratio_AH_HD (BC AC : ℝ) (angle_C : ℝ) := 
  let H := sorry in -- Orthocenter
  let AD := sorry in -- Altitude from A
  let HD := sorry in -- Segment HD
  AD / HD

theorem triangle_ratio_AH_HD (BC AC : ℝ) (angle_C : ℝ) (hBC : BC = 6) (hAC : AC = 4 * real.sqrt 2) (h_angle_C : angle_C = 60)
    : ratio_AH_HD BC AC angle_C = 12 + 16 * real.sqrt 2 :=
  sorry

end triangle_ratio_AH_HD_l307_307164


namespace tangency_triangle_area_l307_307671

noncomputable def mutual_tangent_circles_area : ℚ :=
sorry

theorem tangency_triangle_area :
  mutual_tangent_circles_area 2 3 4 = 25 / 14 :=
sorry

end tangency_triangle_area_l307_307671


namespace tour_routes_l307_307359

-- Definitions based on conditions
def choose (n k : ℕ) : ℕ := nat.choose n k
def factorial (n : ℕ) : ℕ := nat.factorial n

-- Problem statement
theorem tour_routes (a b c d e f g : Type) 
  (A B : a) (C D E F G : b) : 
  choose 5 3 = 10 →
  factorial 3 = 6 →
  ∑ n in (finset.range 10), n = 10 →

  -- There exist 600 different tour routes
  600 :=
by
  sorry

end tour_routes_l307_307359


namespace largest_divisor_540_315_l307_307313

theorem largest_divisor_540_315 : ∃ d : ℕ, d ∣ 540 ∧ d ∣ 315 ∧ d = 45 := by
  sorry

end largest_divisor_540_315_l307_307313


namespace find_w_l307_307265

-- Define the first polynomial's coefficients and roots
def poly1 : Polynomial ℤ := Polynomial.C (-7) + Polynomial.X * (Polynomial.C 6 + Polynomial.X * (Polynomial.C 5 + Polynomial.X))

-- Define the second polynomial's coefficient representation
noncomputable def poly2 (w : ℤ) : Polynomial ℤ := Polynomial.C w + Polynomial.X * (Polynomial.C 0 + Polynomial.X * (Polynomial.C 0 + Polynomial.X))

-- State the problem in Lean as a theorem
theorem find_w (p q r u v w : ℤ) (h1 : poly1.eval p = 0) (h2 : poly1.eval q = 0) (h3 : poly1.eval r = 0) 
  (h4 : poly2 w.eval (p + q) = 0) (h5 : poly2 w.eval (q + r) = 0) (h6 : poly2 w.eval (r + p) = 0) : 
  w = 37 := 
sorry

end find_w_l307_307265


namespace median_values_count_l307_307955

theorem median_values_count (R : Set Int) (hR_card : Set.toFinset R.card = 9) 
  (hR_subset : {2, 3, 4, 6, 9, 14} ⊆ R) : ∃! n, (n = 7) :=
sorry

end median_values_count_l307_307955


namespace find_x0_l307_307466

noncomputable def f (x : ℝ) : ℝ := 13 - 8 * x + x^2

theorem find_x0 :
  (∃ x0 : ℝ, deriv f x0 = 4) → ∃ x0 : ℝ, x0 = 6 :=
by
  sorry

end find_x0_l307_307466


namespace inverse_proportion_quadrants_l307_307141

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∃ (x y : ℝ), x = 3 ∧ y = -2 ∧ y = k / x) → (k < 0 → (∀ (x : ℝ), (x > 0 → ((k / x < 0) ∨ (x < 0 → (k / x > 0))))) :=
by
  intro h1 h2
  sorry

end inverse_proportion_quadrants_l307_307141


namespace simplify_expression_l307_307386

theorem simplify_expression (a b : ℝ) : 
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b :=
by
  sorry

end simplify_expression_l307_307386


namespace vasya_number_l307_307306

theorem vasya_number (a b c d : ℕ) (h1 : a * b = 21) (h2 : b * c = 20) (h3 : ∃ x, x ∈ [4, 7] ∧ a ≠ c ∧ b = 7 ∧ c = 4 ∧ d = 5) : (1000 * a + 100 * b + 10 * c + d) = 3745 :=
sorry

end vasya_number_l307_307306


namespace andrea_last_number_probability_l307_307412

open_locale big_operators

theorem andrea_last_number_probability :
  let p : ℕ → ℚ := λ n, if n = 1 then 3 / 11 else if n = 3 then 1 / 11 else 1 / 2 in
  (1/4) * (p 1 + p 2 + p 3 + p 4) = 17 / 44 :=
by
  -- Definitions and conditions
  let sum_to_prime := {x | x ∈ {3, 5, 7}},
  let chosen_digit := [1, 2, 3, 4],
  sorry

end andrea_last_number_probability_l307_307412


namespace min_chips_needed_l307_307669

theorem min_chips_needed (colors : ℕ) (adjacent : (ℕ → ℕ → Prop)) (chips : list ℕ) (h_colors : colors = 6)
  (unlimited_chips : ∀ c, c ≤ colors → ∃ l, chips = list.repeat c l) : 
  (∀ (c₁ c₂ : ℕ), c₁ ≠ c₂ → ∃ i, adjacent (chips.nth_le i (sorry)) (chips.nth_le (i + 1) sorry) ∧ 
    (chips.nth_le i sorry = c₁ ∨ chips.nth_le (i + 1) sorry = c₁) ∧ 
    (chips.nth_le i sorry = c₂ ∨ chips.nth_le (i + 1) sorry = c₂)) → chips.length ≥ 18 := sorry

end min_chips_needed_l307_307669


namespace solve_system_l307_307500

noncomputable def sqrt_cond (x y : ℝ) : Prop :=
  Real.sqrt ((3 * x - 2 * y) / (2 * x)) + Real.sqrt ((2 * x) / (3 * x - 2 * y)) = 2

noncomputable def quad_cond (x y : ℝ) : Prop :=
  x^2 - 18 = 2 * y * (4 * y - 9)

theorem solve_system (x y : ℝ) : sqrt_cond x y ∧ quad_cond x y ↔ (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 1.5) :=
by
  sorry

end solve_system_l307_307500


namespace non_monotonic_implies_a_gt_2_div_3_l307_307201

theorem non_monotonic_implies_a_gt_2_div_3
  (a : ℝ) (ha : a > 0) :
  (∃ (x : ℝ), (0 < x ∧ x < 3 ∧ x = 2/a)) ↔ a > (2 / 3) :=
begin
  sorry
end

end non_monotonic_implies_a_gt_2_div_3_l307_307201


namespace find_a_l307_307546

theorem find_a (a : ℝ) (h : ∀ x, (x, a^-x) = (1, 1/2)) : a = 2 :=
sorry

end find_a_l307_307546


namespace smallest_pos_integer_h_l307_307853

theorem smallest_pos_integer_h (n : ℕ) : 
  ∃ h : ℕ, 
    (∀ A : Finset (Finset ℕ), A.card = n → 
      (∀ partition : A → ℕ → Prop, 
        (∃ a x y : ℕ, 
          1 ≤ x ∧ x ≤ y ∧ y ≤ h ∧ 
          (∃ i : A, partition i (a + x) ∧ partition i (a + y) ∧ partition i (a + x + y))))) → 
    h = 2 * n :=
by
  sorry

end smallest_pos_integer_h_l307_307853


namespace find_f_neg_2_l307_307198

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom function_definition : ∀ x : ℝ, x ≥ 0 → f(x) = 3^x - 2*x + a
axiom a_value : a = -1

-- Problem statement
theorem find_f_neg_2 : f (-2) = -4 := by
  sorry

end find_f_neg_2_l307_307198


namespace difference_of_distances_l307_307336

-- Definition of John's walking distance to school
def John_distance : ℝ := 0.7

-- Definition of Nina's walking distance to school
def Nina_distance : ℝ := 0.4

-- Assertion that the difference in walking distance is 0.3 miles
theorem difference_of_distances : (John_distance - Nina_distance) = 0.3 := 
by 
  sorry

end difference_of_distances_l307_307336


namespace students_to_add_l307_307733

theorem students_to_add (students := 1049) (teachers := 9) : ∃ n, students + n ≡ 0 [MOD teachers] ∧ n = 4 :=
by
  use 4
  sorry

end students_to_add_l307_307733


namespace bowling_ball_weight_l307_307407

theorem bowling_ball_weight (b c : ℕ) (h1 : 8 * b = 4 * c) (h2 : 3 * c = 108) : b = 18 := 
by 
  sorry

end bowling_ball_weight_l307_307407


namespace sum_of_real_solutions_to_eq_l307_307025

noncomputable def sum_of_solutions : Real :=
  let eq1 (x : Real) : Prop := (x^2 - 5*x + 3)^(x^2 - 6*x + 3) = 1
  let solutions := {x : Real | eq1 x}
  solutions.sum sorry

theorem sum_of_real_solutions_to_eq : sum_of_solutions = 11 :=
sorry

end sum_of_real_solutions_to_eq_l307_307025


namespace geometric_sequence_term_l307_307889

theorem geometric_sequence_term {
  a : ℕ → ℝ,
  q : ℝ
} (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a4 : a 4 = 2)
  (h_a8 : a 8 = 32) :
  a 6 = 8 :=
sorry

end geometric_sequence_term_l307_307889


namespace identity_of_letters_l307_307251

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l307_307251


namespace exists_l307_307424

def smallest_a_composite_expression : ℕ :=
  8

theorem exists smallest_a_composite : ∀ x : ℤ, ∃ a : ℕ, (∀ y : ℤ, ¬ prime (y^4 + a^2 + 16)) ∧ a = 8 :=
by
  sorry

end exists_l307_307424


namespace find_square_of_chord_length_l307_307562

noncomputable def square_of_chord_length (r1 r2 d : ℝ) (x : ℝ) :=
∃ x : ℝ, (r1 = 10 ∧ r2 = 8 ∧ d = 15 ∧ (x = √250) ∧ 
          ∀ {a b c C : ℝ}, (a = r1 ∧ b = r2 ∧ c = d ∧ 
          cos C = (a^2 + b^2 - c^2) / (2 * a * b)) → 
          (cos (arc cos (x / (2 * r1)) + arc cos (x / (2 * r2))) = cos (π - arc cos ((c^2 - a^2 - b^2) / (-2 * a * b))) → 
          x^2 = 250))

theorem find_square_of_chord_length : square_of_chord_length 10 8 15 (√250) :=
sorry

end find_square_of_chord_length_l307_307562


namespace rose_paid_after_discount_l307_307995

noncomputable def discount_percentage : ℝ := 0.1
noncomputable def original_price : ℝ := 10
noncomputable def discount_amount := discount_percentage * original_price
noncomputable def final_price := original_price - discount_amount

theorem rose_paid_after_discount : final_price = 9 := by
  sorry

end rose_paid_after_discount_l307_307995


namespace problem_statement_l307_307131

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^2021 + a^2022 = 2 := 
by
  sorry

end problem_statement_l307_307131


namespace part1_tangent_line_eq_part2_range_of_a_l307_307486

-- Part 1: Tangent line to the function at (2, f(2)) for a = 1
theorem part1_tangent_line_eq (x : ℝ) (hx : x = 2 ∨ f x = (a-1) * real.log x + x + a / x)
    (FA : ∀ (x : ℝ), f x = (1-1) * real.log x + x + 1 / x)
    (y : ℝ) 
    (h2 : y = f 2):
  3 * x - 4 * y + 4 = 0 :=
sorry

-- Part 2: Range of values for a such that f(x) - a/x > 0 for all x in (1,e]
theorem part2_range_of_a (a : ℝ) :
    (∀ x : ℝ, 1 < x ∧ x ≤ real.exp 1 → (f x - a / x > 0)) ↔ (a ∈ set.Ioi (1 - real.exp 1)) :=
sorry

end part1_tangent_line_eq_part2_range_of_a_l307_307486


namespace arccos_neg_half_l307_307819

-- Defining the problem in Lean 4
theorem arccos_neg_half : 
  ∃ θ ∈ set.Icc 0 Real.pi, Real.arccos (-1 / 2) = θ ∧ Real.cos θ = -1 / 2 := 
by
  use Real.pi * 2 / 3
  split
  { sorry } -- Proof that θ is in [0, π]
  { split
    { sorry } -- Proof that θ = arccos(-1 / 2)
    { sorry } -- Proof that cos(θ) = -1/2
  }


end arccos_neg_half_l307_307819


namespace cube_volume_from_diagonal_l307_307346

-- Definition of the cube's side length given its space diagonal
def cube_side_length (d : ℝ) : ℝ := d / Real.sqrt 3

-- The theorem stating that given a space diagonal of 10√3, the volume is 1000
theorem cube_volume_from_diagonal (d : ℝ) (h : d = 10 * Real.sqrt 3) : 
  (cube_side_length d) ^ 3 = 1000 :=
by
  sorry

end cube_volume_from_diagonal_l307_307346


namespace symmetric_matrix_count_lower_bound_l307_307426

-- Define the Euler's totient function φ(n)
def euler_totient (n : ℕ) : ℕ :=
  (Finset.range n).filter (n.gcd · = 1).card

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the number f(n) according to the conditions
noncomputable def f (n : ℕ) : ℕ :=
  sorry  -- Placeholder to be filled with the actual counting function

-- The main theorem statement
theorem symmetric_matrix_count_lower_bound (n : ℕ) (h1 : n % 2 = 1) :
  f(n) ≥ (factorial n * factorial (n - 1)) / euler_totient n :=
sorry

end symmetric_matrix_count_lower_bound_l307_307426


namespace ratio_of_shaded_to_non_shaded_l307_307676

open Real

-- Define the midpoints and the necessary variables
structure Point (x y : ℝ)

def midpoint (P Q : Point) : Point :=
  Point ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Let Triangle ABC has coordinates A(0, 0), B(6, 0), C(0, 8)
def A : Point := ⟨0, 0⟩
def B : Point := ⟨6, 0⟩
def C : Point := ⟨0, 8⟩
def D : Point := midpoint A B -- midpoint of AB
def F : Point := midpoint A C -- midpoint of AC
def E : Point := midpoint B C -- midpoint of BC
def G : Point := midpoint D F -- midpoint of DF
def H : Point := midpoint F E -- midpoint of FE

noncomputable def triangle_area (P Q R : Point) : ℝ :=
  abs ((P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)) / 2)

-- Calculate areas
noncomputable def area_ABC : ℝ := triangle_area A B C
noncomputable def area_DFG : ℝ := triangle_area D F G
noncomputable def area_FEH : ℝ := triangle_area F E H

noncomputable def shaded_area : ℝ := area_DFG + area_FEH
noncomputable def non_shaded_area : ℝ := area_ABC - shaded_area

noncomputable def ratio_shaded_to_non_shaded : ℚ :=
  (shaded_area / non_shaded_area).toRat

theorem ratio_of_shaded_to_non_shaded :
  ratio_shaded_to_non_shaded = (23 / 25 : ℚ) := by
    -- This is where the proof would go
    sorry

end ratio_of_shaded_to_non_shaded_l307_307676


namespace power_mod_remainder_l307_307319

theorem power_mod_remainder : (3^20) % 7 = 2 :=
by {
  -- condition: 3^6 ≡ 1 (mod 7)
  have h1 : (3^6) % 7 = 1 := by norm_num,
  -- we now use this to show 3^20 ≡ 2 (mod 7)
  calc
    (3^20) % 7 = ((3^6)^3 * 3^2) % 7 : by norm_num
          ... = (1^3 * 3^2) % 7       : by rw [←nat.modeq.modeq_iff_dvd, h1]
          ... =  (3^2) % 7            : by norm_num
          ... = 2                    : by norm_num
}

end power_mod_remainder_l307_307319


namespace minimum_tangent_slope_angle_l307_307548

/-- The function given in the problem -/
def f (x : ℝ) : ℝ := x^3 / 3 - x^2 + 1

/-- The derivative of the function f -/
def f' (x : ℝ) : ℝ := x^2 - 2 * x

/-- The tangent slope angle at any point on the graph of f -/
noncomputable def α (x : ℝ) : ℝ := Real.arctan (f' x)

theorem minimum_tangent_slope_angle :
  (∀ x ∈ Ioo 0 2, α x ≥ 3 * Real.pi / 4) :=
sorry

end minimum_tangent_slope_angle_l307_307548


namespace percent_decrease_computer_price_l307_307146

theorem percent_decrease_computer_price (price_1990 price_2010 : ℝ) (h1 : price_1990 = 1200) (h2 : price_2010 = 600) :
  ((price_1990 - price_2010) / price_1990) * 100 = 50 := 
  sorry

end percent_decrease_computer_price_l307_307146


namespace shift_sine_graph_l307_307674

theorem shift_sine_graph :
  ∀ x : ℝ, (sin (2 * (x - π / 8)) = sin (2 * x - π / 4)) := by
  intro x
  sorry

end shift_sine_graph_l307_307674


namespace coordinates_of_Q_l307_307161

-- Given points P and R
def P := (1, 1)
def R := (5, 3)

-- Defining the coordinates of Q based on the conditions
def Q : ℝ × ℝ := (5, 1)

-- Proving that Q has the correct coordinates given the conditions
theorem coordinates_of_Q : 
  (∃ Qx Qy : ℝ, Q = (Qx, Qy) ∧ Qy = 1 ∧ Qx = 5) :=
by
  use 5, 1
  split
  . refl
  split
  . exact rfl
  . exact rfl

end coordinates_of_Q_l307_307161


namespace identity_of_letters_l307_307253

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l307_307253


namespace problem1_problem2_problem3_l307_307865

section

-- Define the function f with the given properties
variable {f : ℝ → ℝ}
variable (hf_nonzero : ∀ x, f x ≠ 0)
variable (hf_add : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ * f x₂)
variable (hf_pos : ∀ x > 0, f x > 1)

-- Problem 1: Monotonicity
theorem problem1 : ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ :=
sorry

-- Problem 2: Finding θ
theorem problem2 (θ : ℝ) (h : f (4 * (Real.cos θ) ^ 2) * f (4 * (Real.sin θ) * (Real.cos θ)) = 1) :
    ∃ k : ℤ, θ = k * Real.pi + Real.pi / 2 ∨ θ = k * Real.pi - Real.pi / 4 :=
sorry

-- Problem 3: Existence of m
theorem problem3 : ∃ (m : ℝ), (∀ θ ∈ Icc 0 (Real.pi / 2), f (Real.cos θ ^ 2 - (2 + m) * Real.sin θ) * f (3 + 2 * m) > 1) ∧ m > -1 :=
sorry

end

end problem1_problem2_problem3_l307_307865


namespace geom_seq_formula_arith_seq_formula_sum_C_l307_307440

-- Given conditions as definitions
def geom_seq (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n : ℕ, a (n+1) = q * a n

def arith_seq (a b c : ℕ) : Prop :=
2 * b = a + c - 1

def sum_geom_seq (a : ℕ → ℕ) (n : ℕ) (S : ℕ → ℕ) : Prop :=
S n = a 0 * ((1 - (q : ℕ) ^ n) / (1 - q))

noncomputable def S_n := λ n : ℕ, ((2 ^ n) - 1)

def sum_arith_seq (b : ℕ → ℕ) (n : ℕ) (T : ℕ → ℕ) : Prop :=
6 * T n = (3 * n + 1) * b n + 2

-- Results to prove
theorem geom_seq_formula : (∀ n : ℕ, geom_seq a 2 → a n = 2^(n-1)) :=
sorry

theorem arith_seq_formula : (∀ n : ℕ, sum_arith_seq b n T → b n = 3 * n - 2) :=
sorry

theorem sum_C : let A := {a 1, a 2, a 3, a 4, a 5, a 6, a 7, a 8, a 9, a 10},
                   B := {b 1, b 2, b 3, b 4, b 5, b 6, b 7, b 8, b 9, b 10,
                         b 11, b 12, b 13, b 14, b 15, b 16, b 17, b 18, b 19, b 20,
                         b 21, b 22, b 23, b 24, b 25, b 26, b 27, b 28, b 29, b 30,
                         b 31, b 32, b 33, b 34, b 35, b 36, b 37, b 38, b 39, b 40}
               in  sum (A ∪ B) = 3318 :=
sorry

end geom_seq_formula_arith_seq_formula_sum_C_l307_307440


namespace seating_arrangements_l307_307980

-- Definitions:
def children : Type := {i : ℕ // i < 9}

def sibling_sets {n : ℕ} (sets : list (list children)) : Prop :=
  sets.length = 3 ∧ ∀ s ∈ sets, s.length = 3

def rows (R : fin 3 → fin 3 → option children) : Prop :=
  ∀ i j, ∀ (c : children), R i j = some c → (R i (j+1) ≠ some c) ∧ (R (i+1) j ≠ some c)

-- Problem statement:
theorem seating_arrangements :
  ∃ (R : fin 3 → fin 3 → option children),
    sibling_sets [[⟨0, nat.lt_succ_self 8⟩, ⟨1, nat.lt_succ_self 8⟩, ⟨2, nat.lt_succ_self 8⟩], 
                   [⟨3, nat.lt_succ_succ_self 8⟩, ⟨4, nat.lt_succ_succ_self 8⟩, ⟨5, nat.lt_succ_succ_self 8⟩], 
                   [⟨6, nat.lt_succ_self 8⟩, ⟨7, nat.lt_succ_self 8⟩, ⟨8, nat.lt_succ_self 8⟩]] ∧
    rows R ∧
    card (R) = 648 :=
sorry

end seating_arrangements_l307_307980


namespace a_5_over_b_5_l307_307596

variables {a b : ℕ → ℝ}
variables {S T : ℕ → ℝ}
variables {n : ℕ}

-- Conditions
def is_arithmetic_sequence (s : ℕ → ℝ) := ∃ d : ℝ, ∀ n, s (n + 1) = s n + d

def sum_first_n_terms (s : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, s i

def S_n (n : ℕ) := sum_first_n_terms a n
def T_n (n : ℕ) := sum_first_n_terms b n

axiom S_n_T_n_relation : ∀ n, S_n n / T_n n = (7 * n + 2 : ℝ) / (n + 3)

-- The statement to prove
theorem a_5_over_b_5 :
  (a 5 / b 5) = 65 / 12 :=
sorry

end a_5_over_b_5_l307_307596


namespace triangle_side_range_l307_307292

theorem triangle_side_range (a : ℝ) :
  1 < a ∧ a < 4 ↔ 3 + (2 * a - 1) > 4 ∧ 3 + 4 > 2 * a - 1 ∧ 4 + (2 * a - 1) > 3 :=
by
  sorry

end triangle_side_range_l307_307292


namespace rectangle_area_change_l307_307276

def percentage_change_area (L B : ℝ) (L' B' : ℝ) : ℝ :=
  ((L' * B' - L * B) / (L * B)) * 100

theorem rectangle_area_change (L B : ℝ) (hL : L' = (3/4 : ℝ) * L) (hB : B' = (8/5 : ℝ) * B) :
  percentage_change_area L B L' B' = 20 := by
  sorry

end rectangle_area_change_l307_307276


namespace three_numbers_sum_multiple_of_three_l307_307857

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0

def count_ways_to_choose_three_diff_numbers (s : Finset ℕ) (cond : ∀ x ∈ s, x < 31) : ℕ :=
  let A := s.filter (λ x, x % 3 = 1)
  let B := s.filter (λ x, x % 3 = 2)
  let C := s.filter (λ x, x % 3 = 0)
  (((A.card * B.card * C.card) : ℕ) : ℕ) + (A.card.choose 3) + (B.card.choose 3) + (C.card.choose 3)

theorem three_numbers_sum_multiple_of_three :
  count_ways_to_choose_three_diff_numbers (Finset.range 30) (by simp) = 1360 :=
sorry

end three_numbers_sum_multiple_of_three_l307_307857


namespace exists_n_divisible_by_2019_by_num_divisors_l307_307028

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem exists_n_divisible_by_2019_by_num_divisors : ∃ (n : ℕ+), 2019 ∣ num_divisors (factorial n) :=
  sorry

end exists_n_divisible_by_2019_by_num_divisors_l307_307028


namespace solution_set_l307_307092

variable {f : ℝ → ℝ}

axiom domain_real : ∀ x : ℝ, x ∈ set.univ
axiom symmetric_about_neg1 : ∀ x : ℝ, f(-1 - x) = -f(-1 + x)
axiom derivative_f : ∀ x : ℝ, deriv f x = f' x
axiom condition_neg1 : ∀ x : ℝ, x < -1 → (x + 1) * (f x + (x + 1) * f' x) < 0

theorem solution_set (f0 : f 0) : {x : ℝ | x * f (x - 1) > f 0} = Ioo (-1) 1 :=
sorry

end solution_set_l307_307092


namespace final_length_of_movie_l307_307672

theorem final_length_of_movie :
  let original_length := 3600 -- original movie length in seconds
  let cut_1 := 3 * 60 -- first scene cut in seconds
  let cut_2 := (5 * 60) + 30 -- second scene cut in seconds
  let cut_3 := (2 * 60) + 15 -- third scene cut in seconds
  let total_cut := cut_1 + cut_2 + cut_3 -- total cut time in seconds
  let final_length_seconds := original_length - total_cut -- final length in seconds
  final_length_seconds = 2955 ∧ final_length_seconds / 60 = 49 ∧ final_length_seconds % 60 = 15
:= by
  sorry

end final_length_of_movie_l307_307672


namespace matrix_operation_correct_l307_307383

open Matrix

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 5]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 4], ![0, -3]]
def matrix3 : Matrix (Fin 2) (Fin 2) ℤ := ![![6, 0], ![-1, 8]]
def result : Matrix (Fin 2) (Fin 2) ℤ := ![![12, -7], ![1, 16]]

theorem matrix_operation_correct:
  matrix1 - matrix2 + matrix3 = result :=
by
  sorry

end matrix_operation_correct_l307_307383


namespace original_price_l307_307758

theorem original_price (selling_price profit_percent : ℝ) (h_sell : selling_price = 63) (h_profit : profit_percent = 5) : 
  selling_price / (1 + profit_percent / 100) = 60 :=
by sorry

end original_price_l307_307758


namespace sunflower_height_l307_307208

theorem sunflower_height (H : ℝ) 
  (height_A : ℝ = 192)
  (height_relationship : height_A = H + 0.20 * H) : 
  H = 160 := 
by
  sorry

end sunflower_height_l307_307208


namespace convert_rectangular_to_polar_l307_307017

noncomputable def rectangular_to_polar_coords (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let θ := if y ≥ 0 then real.arctan (y / x) else 2 * real.pi + real.arctan (y / x)
  (r, θ)

theorem convert_rectangular_to_polar :
  rectangular_to_polar_coords 3 (-3) = (3 * real.sqrt 2, 7 * real.pi / 4) :=
by
  sorry

end convert_rectangular_to_polar_l307_307017


namespace sequence_sum_inequality_l307_307110

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Definitions from the conditions
def a_1 : ℚ := 2 / 5
def a_seq_formula (n : ℕ) (a_n : ℚ) : ℚ := 2 * a_n / (3 - a_n)
def S_sum {n : ℕ} : ℚ := ∑ i in Finset.range n + 1, a i

-- Statement of the proof problem
theorem sequence_sum_inequality
  (ha1 : a 1 = a_1)
  (hrec : ∀ n, a (n + 1) = a_seq_formula n (a n))
  (hS : ∀ n, S n = S_sum n) :
  ∀ n, (6 / 5) * (1 - (2 / 3) ^ n) ≤ S n ∧ S n < 21 / 13 :=
by
  sorry

end sequence_sum_inequality_l307_307110


namespace cos_neg_45_eq_one_over_sqrt_two_l307_307007

theorem cos_neg_45_eq_one_over_sqrt_two : Real.cos (-(45 : ℝ)) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_neg_45_eq_one_over_sqrt_two_l307_307007


namespace angle_A_is_pi_over_3_l307_307465

theorem angle_A_is_pi_over_3 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C)
  (h2 : a ^ 2 = b ^ 2 + c ^ 2 - bc * (2 * Real.cos A))
  (triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ A + B + C = π) :
  A = π / 3 :=
by
  sorry

end angle_A_is_pi_over_3_l307_307465


namespace minimum_pairs_of_cities_l307_307568

/-- In the country of Alfya, there are 150 cities, and any four cities can be divided 
    into two pairs such that there is an express train running between the cities of each pair. 
    We need to prove that the minimum number of pairs of cities connected by express trains is 11025. -/
theorem minimum_pairs_of_cities : 
  ∀ (V : Type) [Fintype V], 
  Fintype.card V = 150 →
  (∀ (A B C D : V), ∃ (X Y : set (V × V)), {A, B, C, D}.pairwise (λ x y, (x, y) ∈ X ∪ Y)) →
  ∃ (E : set (V × V)), 
  (∀ (u : V), Fintype.card {v : V | (u, v) ∈ E ∨ (v, u) ∈ E} ≥ 147) ∧ 
  Fintype.card E = 11025 :=
µ sorry

end minimum_pairs_of_cities_l307_307568


namespace find_values_of_abc_and_root_l307_307115

theorem find_values_of_abc_and_root (a b c : ℕ) : 
  (a = 2 ∧ b = 5 ∧ c = 0) ∧ (sqrt (3 * a + 10 * b + c) = 2 * sqrt 14 ∨ sqrt (3 * a + 10 * b + c) = -2 * sqrt 14) :=
by 
  have h1 : (∛(3 * a + 21) = 3) := sorry,
  have h2 : (sqrt (b - 1) = 2) := sorry,
  have h3 : (sqrt c = c) := sorry,
  sorry

end find_values_of_abc_and_root_l307_307115


namespace nadia_flower_shop_l307_307979

theorem nadia_flower_shop :
  let roses := 20
  let lilies := (3 / 4) * roses
  let cost_per_rose := 5
  let cost_per_lily := 2 * cost_per_rose
  let total_cost := roses * cost_per_rose + lilies * cost_per_lily
  total_cost = 250 := by
    sorry

end nadia_flower_shop_l307_307979


namespace min_express_pairs_l307_307565

universe u

/-- In the country of Alfya, there are 150 cities, some of which are connected by express trains that do not stop at intermediate stations. It is known that any four cities can be divided into two pairs such that there is an express running between the cities of each pair. What is the minimum number of pairs of cities connected by expresses? -/
theorem min_express_pairs (C : Type u) [Fintype C] (hC : Fintype.card C = 150)
    (express : C → C → Prop) (h3 : ∀ a b : C, express a b → express b a)
    (h4 : ∀ (A B C D : C), ∃ (P : {A B} → {C D}), ∀ (s t : {A B} → {C D}), ∃ (x y : C), x ≠ y ∧ express x y) :
  ∃ m, m = 11025 :=
by
  sorry

end min_express_pairs_l307_307565


namespace sum_of_squares_of_roots_l307_307280

theorem sum_of_squares_of_roots (x_1 x_2 : ℚ) (h1 : 6 * x_1^2 - 13 * x_1 + 5 = 0)
                                (h2 : 6 * x_2^2 - 13 * x_2 + 5 = 0) 
                                (h3 : x_1 ≠ x_2) :
  x_1^2 + x_2^2 = 109 / 36 :=
sorry

end sum_of_squares_of_roots_l307_307280


namespace solution_set_of_inequality_l307_307884

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom condition1 : ∀ x : ℝ, f(x) < f'(x)
axiom condition2 : f 0 = 2

theorem solution_set_of_inequality : { x : ℝ | f(x) > 2 * Real.exp x } = { x : ℝ | 0 < x } :=
by sorry

end solution_set_of_inequality_l307_307884


namespace find_g_of_fx_is_odd_l307_307084

noncomputable def intgrl := ∫ (t : ℝ) in (0 : ℝ)..(1 : ℝ), 2 * t

theorem find_g_of_fx_is_odd (f g : ℝ → ℝ) (h1 : ∀ x, f x + g x = ∫ t in x..(x+1), 2 * t)
  (h2 : ∀ x, f (-x) = -f x) : g = (λ x, 1 + x) := 
sorry

end find_g_of_fx_is_odd_l307_307084


namespace tangency_points_of_incircle_l307_307202

-- Definitions for the hyperbola-related entities
structure Hyperbola where
  F1 F2: Point -- The foci
  M N: Point  -- The vertices

structure Triangle (P F1 F2 : Point) where
  P F1 F2 : Point
    
-- Definition of incircle and point of tangency
def incircle_tangency_point (triangle : Triangle): Point := sorry

-- Conditions for the problem
axiom hyperbola (h : Hyperbola) : ∀ P : Point, (lies_on_hyperbola h P) →
  incircle_tangency_point (Triangle.mk P h.F1 h.F2) = h.M ∨ 
  incircle_tangency_point (Triangle.mk P h.F1 h.F2) = h.N

-- Final statement we want to prove
theorem tangency_points_of_incircle (h : Hyperbola) (P : Point)
  (hP : lies_on_hyperbola h P) :
  incircle_tangency_point (Triangle.mk P h.F1 h.F2) = h.M ∨ 
  incircle_tangency_point (Triangle.mk P h.F1 h.F2) = h.N :=
  hyperbola h P hP

end tangency_points_of_incircle_l307_307202


namespace arithmetic_sequence_sum_l307_307200

noncomputable def f (x : ℝ) : ℝ := (x - 3) ^ 3 + x - 1

theorem arithmetic_sequence_sum :
  ∃ (a : ℝ) (d : ℝ), d ≠ 0 ∧ 
  let a_n := λ n : ℕ, a + n * d in
  f (a_n 0) + f (a_n 1) + f (a_n 2) + f (a_n 3) + f (a_n 4) + f (a_n 5) + f (a_n 6) = 14 →
  (a_n 0) + (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) + (a_n 6) = 21 :=
by
  sorry

end arithmetic_sequence_sum_l307_307200


namespace find_min_f_l307_307194

open BigOperators

variable (S : Finset ℕ) (A B : Finset ℕ) (f : ℕ) (C : Finset ℕ)

def symmetric_diff (X Y : Finset ℕ) : Finset ℕ := (X \ Y) ∪ (Y \ X)
def S := (Finset.range 2017).erase 0
def C := A.product B |>.image (λ p => p.1 + p.2)

theorem find_min_f (A_nonempty : A.nonempty) (B_nonempty : B.nonempty) :
  f = |symmetric_diff A S| + |symmetric_diff B S| + |symmetric_diff C S| →
  f = 2017 := sorry

end find_min_f_l307_307194


namespace six_people_arrangement_not_next_to_each_other_l307_307296

theorem six_people_arrangement_not_next_to_each_other :
  let total_arrangements := Nat.factorial 6,
      together_arrangements := Nat.factorial 5 * 2 in
  (total_arrangements - together_arrangements) = 480 :=
by
  sorry

end six_people_arrangement_not_next_to_each_other_l307_307296


namespace probability_multiple_of_2_3_5_l307_307797

theorem probability_multiple_of_2_3_5 :
  let cards := (1 : ℕ) :: (List.range 99).map (λ n, n+2)  -- cards from 1 to 100
  let favorable := cards.filter (λ n, n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0)
  let probability := favorable.length.toRat / cards.length.toRat
  probability = (37 : ℚ) / 50 :=
by
  sorry

end probability_multiple_of_2_3_5_l307_307797


namespace EmmaBalanceIsCorrect_l307_307837

-- Definitions of the conditions
def EmmaInitialBalance : ℝ := 1200
def compound_interest_rate_annual : ℝ := 0.03
def compound_interest_rate_monthly : ℝ := compound_interest_rate_annual / 12
def withdrawal_shoes_percent : ℝ := 0.08
def deposit_after_withdrawal_percent : ℝ := 0.25
def paycheck_deposit_percent : ℝ := 1.5
def withdrawal_gift_percent : ℝ := 0.05

-- Definition of the transactions
def EmmaBalanceOnApril1 : ℝ :=
  let balance_after_withdrawal_shoes := EmmaInitialBalance * (1 - withdrawal_shoes_percent)
  let withdrawal_shoes := EmmaInitialBalance * withdrawal_shoes_percent
  let balance_after_deposit := balance_after_withdrawal_shoes + (withdrawal_shoes * deposit_after_withdrawal_percent)
  let balance_after_january_interest := balance_after_deposit * (1 + compound_interest_rate_monthly)
  let balance_after_paycheck_deposit := balance_after_january_interest + (withdrawal_shoes * paycheck_deposit_percent)
  let balance_after_february_interest := balance_after_paycheck_deposit * (1 + compound_interest_rate_monthly)
  let balance_after_withdrawal_gift := balance_after_february_interest * (1 - withdrawal_gift_percent)
  let balance_after_march_interest := balance_after_withdrawal_gift * (1 + compound_interest_rate_monthly)
  balance_after_march_interest

-- Lean statement to prove the problem
theorem EmmaBalanceIsCorrect :
  EmmaBalanceOnApril1 = 1217.15 := by
  sorry

end EmmaBalanceIsCorrect_l307_307837


namespace third_set_candies_l307_307693

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l307_307693


namespace cosine_between_vectors_A_B_and_A_C_l307_307038

def vector_sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (b.1 - a.1, b.2 - a.2, b.3 - a.3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

def cos_angle (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_between_vectors_A_B_and_A_C :
  let A := (1, -2, 3)
  let B := (0, -1, 2)
  let C := (3, -4, 5)
  let AB := vector_sub A B
  let AC := vector_sub A C
  cos_angle AB AC = -1 := by
  sorry

end cosine_between_vectors_A_B_and_A_C_l307_307038


namespace solve_S_l307_307590

variable {S : ℝ}
variable h : S = (1 / (4 - real.sqrt 9)) - (1 / (real.sqrt 9 - real.sqrt 8)) + (1 / (real.sqrt 8 - real.sqrt 7)) - (1 / (real.sqrt 7 - real.sqrt 6)) + (1 / (real.sqrt 6 - 3))

theorem solve_S : S = 7 :=
by
  sorry

end solve_S_l307_307590


namespace solve_modified_system_l307_307289

theorem solve_modified_system (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : 4 * a1 + 6 * b1 = c1) 
  (h2 : 4 * a2 + 6 * b2 = c2) :
  (4 * a1 * 5 + 3 * b1 * 10 = 5 * c1) ∧ (4 * a2 * 5 + 3 * b2 * 10 = 5 * c2) :=
by
  sorry

end solve_modified_system_l307_307289


namespace beautiful_fold_probability_l307_307223

noncomputable def probability_beautiful_fold (a : ℝ) : ℝ := 1 / 2

theorem beautiful_fold_probability 
  (A B C D F : ℝ × ℝ) 
  (ABCD_square : (A.1 = 0) ∧ (A.2 = 0) ∧ 
                 (B.1 = a) ∧ (B.2 = 0) ∧ 
                 (C.1 = a) ∧ (C.2 = a) ∧ 
                 (D.1 = 0) ∧ (D.2 = a))
  (F_in_square : 0 ≤ F.1 ∧ F.1 ≤ a ∧ 0 ≤ F.2 ∧ F.2 ≤ a):
  probability_beautiful_fold a = 1 / 2 :=
sorry

end beautiful_fold_probability_l307_307223


namespace greatest_x_l307_307722

theorem greatest_x (x : ℕ) : (x^6 / x^3 ≤ 27) → x ≤ 3 :=
by sorry

end greatest_x_l307_307722


namespace white_cannot_lose_l307_307402

-- Define a type to represent the game state
structure Game :=
  (state : Type)
  (white_move : state → state)
  (black_move : state → state)
  (initial : state)

-- Define a type to represent the double chess game conditions
structure DoubleChess extends Game :=
  (double_white_move : state → state)
  (double_black_move : state → state)

-- Define the hypothesis based on the conditions
noncomputable def white_has_no_losing_strategy (g : DoubleChess) : Prop :=
  ∃ s, g.double_white_move (g.double_white_move s) = g.initial

theorem white_cannot_lose (g : DoubleChess) :
  white_has_no_losing_strategy g :=
sorry

end white_cannot_lose_l307_307402


namespace wristwatch_intersection_points_lie_on_circle_l307_307370

-- Definitions for axes and hands
variables (O_s O_m : Point) -- axes for stopwatch hand and second hand
variable (t : ℕ) -- different start time in seconds

-- Conditions as given in the problem
def wristwatch_conditions (O_s O_m : Point) : Prop :=
  O_s ≠ O_m -- Stopwatc and second hands share the same axis but different from each 

-- Proof problem statement
noncomputable def intersection_points_lie_on_circle (O_s O_m : Point) (t : ℕ) 
  (h_cond : wristwatch_conditions O_s O_m) : Prop :=
  ∃ K : Point, ∀ α : ℝ, ((α = 6 * t) → (∃ M : Point, 
    is_on_circle K (distance K M) M))
    
-- Main theorem statement that needs proof
theorem wristwatch_intersection_points_lie_on_circle 
  (O_s O_m : Point) (t : ℕ) (h_cond : wristwatch_conditions O_s O_m) :
  intersection_points_lie_on_circle O_s O_m t h_cond :=
sorry

end wristwatch_intersection_points_lie_on_circle_l307_307370


namespace olympics_year_zodiac_l307_307938

-- Define the list of zodiac signs
def zodiac_cycle : List String :=
  ["rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "goat", "monkey", "rooster", "dog", "pig"]

-- Function to compute the zodiac sign for a given year
def zodiac_sign (start_year : ℕ) (year : ℕ) : String :=
  let index := (year - start_year) % 12
  zodiac_cycle.getD index "unknown"

-- Proof statement: the zodiac sign of the year 2008 is "rabbit"
theorem olympics_year_zodiac :
  zodiac_sign 1 2008 = "rabbit" :=
by
  -- Proof omitted
  sorry

end olympics_year_zodiac_l307_307938


namespace remainder_of_3_pow_20_mod_7_l307_307321

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end remainder_of_3_pow_20_mod_7_l307_307321


namespace lock_can_open_toggle_single_switch_possible_l307_307349

-- Definitions for conditions
def initial_state : matrix (fin 4) (fin 4) bool := 
  ![(1,0,1,0), 
    (1,0,1,0), 
    (0,0,1,1), 
    (0,0,1,1)]

def final_state : matrix (fin 4) (fin 4) bool :=
  Matrix.ones (fin 4) (fin 4)

-- Proving the questions based on the conditions
theorem lock_can_open (M : matrix (fin 4) (fin 4) bool) : 
  (∃ moves : list (fin 4 × fin 4), 
    ∀ m : fin 4 × fin 4, m ∈ moves → true -- This would be the toggling mechanism
    ∧ final_state = apply_moves M moves)
:=
sorry

theorem toggle_single_switch_possible (M : matrix (fin 4) (fin 4) bool) :
  ∃ move : fin 4 × fin 4, 
    (∃ moves : list (fin 4 × fin 4), 
    ∀ m : fin 4 × fin 4, m ∈ moves → true -- Again, the exact toggling step-by-step
    ∧ final_state = apply_moves_single M move moves)
:=
sorry

-- Auxiliary functions (expected to be defined elsewhere)
noncomputable def apply_moves (M : matrix (fin 4) (fin 4) bool) 
  (moves : list (fin 4 × fin 4)) : matrix (fin 4) (fin 4) bool :=
sorry

noncomputable def apply_moves_single (M : matrix (fin 4) (fin 4) bool) 
  (move : fin 4 × fin 4) 
  (moves : list (fin 4 × fin 4)) : matrix (fin 4) (fin 4) bool :=
sorry

end lock_can_open_toggle_single_switch_possible_l307_307349


namespace find_H_coordinates_l307_307906

def E : ℝ × ℝ × ℝ := (2, 3, -1)
def F : ℝ × ℝ × ℝ := (0, 5, 3)
def G : ℝ × ℝ × ℝ := (4, 2, 5)

theorem find_H_coordinates : ∃ H : ℝ × ℝ × ℝ, H = (6, 0, 1) ∧
  ((E.1 + G.1) / 2 = (F.1 + H.1) / 2) ∧
  ((E.2 + G.2) / 2 = (F.2 + H.2) / 2) ∧
  ((E.3 + G.3) / 2 = (F.3 + H.3) / 2) :=
by
  let H := (6, 0, 1)
  use H
  apply And.intro rfl
  repeat { apply And.intro <;>
           field_simp <;>
           rfl }

end find_H_coordinates_l307_307906


namespace sequence_general_formula_l307_307547

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 12)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) :
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 12 :=
sorry

end sequence_general_formula_l307_307547


namespace obtuse_angle_at_515_l307_307314

-- Definitions derived from conditions
def minuteHandDegrees (minute: ℕ) : ℝ := minute * 6.0
def hourHandDegrees (hour: ℕ) (minute: ℕ) : ℝ := hour * 30.0 + (minute * 0.5)

-- Main statement to be proved
theorem obtuse_angle_at_515 : 
  let hour := 5
  let minute := 15
  let minute_pos := minuteHandDegrees minute
  let hour_pos := hourHandDegrees hour minute
  let angle := abs (minute_pos - hour_pos)
  angle = 67.5 :=
by
  sorry

end obtuse_angle_at_515_l307_307314


namespace problem_proof_l307_307183

noncomputable def C1_equation (p : ℝ) : ℝ → ℝ → Prop :=
 λ x y, y^2 = 2 * p * x

noncomputable def H_asymptote_1 (x y : ℝ) : Prop := 2 * x = sqrt 3 * y
noncomputable def H_asymptote_2 (x y : ℝ) : Prop := 2 * x = -sqrt 3 * y
noncomputable def H_focus (x y : ℝ) : Prop := x = 0 ∧ y = sqrt 7

noncomputable def C2_equation : ℝ → ℝ → Prop :=
 λ x y, y^2 / 4 - x^2 / 3 = 1

noncomputable def dot_product (u v : (ℝ × ℝ)) : ℝ :=
 u.1 * v.1 + u.2 * v.2

noncomputable def F (p : ℝ) : (ℝ × ℝ) := (p/2, 0)

theorem problem_proof :
  ∀ (p : ℝ),
  (p > 0) →
  (∀ x y, C1_equation p x y → C2_equation x y → (x > 0 ∧ y > 0)) →
  (∀ A B : (ℝ × ℝ),
    (p > 4*sqrt 3/3 →
    let FA := (A.1 - F p .1, A.2 - F p .2),
        FB := (B.1 - F p .1, B.2 - F p .2)
    in (dot_product FA FB) ≤ 9)) →
  (∀ A B : (ℝ × ℝ),
    let FA := (A.1 - F p .1, A.2 - F p .2),
        FB := (B.1 - F p .1, B.2 - F p .2),
        S := 1/4 * (2*sqrt 3 + p) * sqrt (3*p^2 - 4*sqrt 3*p)
    in (S = 2/3 * (dot_product FA FB) → p = 2*sqrt 3)) :=
sorry

end problem_proof_l307_307183


namespace isosceles_tetrahedron_faces_acute_l307_307352

def is_isosceles_tetrahedron (A B C D : ℝ × ℝ × ℝ) : Prop :=
  dist A B = dist C D ∧ dist A C = dist B D ∧ dist A D = dist B C

-- Definition to check if a triangle is acute-angled (in 3D)
def is_acute_triangle (A B C : ℝ × ℝ × ℝ) : Prop :=
  let a := dist A B in
  let b := dist A C in
  let c := dist B C in
  (a^2 + b^2 > c^2) ∧
  (a^2 + c^2 > b^2) ∧
  (b^2 + c^2 > a^2)

-- Definition to check if all faces of a tetrahedron are acute-angled
def tetrahedron_faces_acute (A B C D : ℝ × ℝ × ℝ) : Prop :=
  is_acute_triangle A B C ∧
  is_acute_triangle A B D ∧
  is_acute_triangle A C D ∧
  is_acute_triangle B C D

theorem isosceles_tetrahedron_faces_acute
  {A B C D : ℝ × ℝ × ℝ} (h : is_isosceles_tetrahedron A B C D) :
  tetrahedron_faces_acute A B C D :=
sorry

end isosceles_tetrahedron_faces_acute_l307_307352


namespace original_number_increased_by_45_percent_is_870_l307_307785

theorem original_number_increased_by_45_percent_is_870 (x : ℝ) (h : x * 1.45 = 870) : x = 870 / 1.45 :=
by sorry

end original_number_increased_by_45_percent_is_870_l307_307785


namespace calculate_f_neg_two_l307_307472

variable {R : Type*} [LinearOrderedField R]

-- Definition of an odd function
def is_odd_function (f : R -> R) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Function f(x)
def f (x : R) : R := if x > 0 then x^2 + 1/x else -(abs x^2 + 1/abs x)

theorem calculate_f_neg_two (h_odd : is_odd_function f) (h_pos : ∀ x > 0, f x = x^2 + 1/x) : 
  f (-2 : R) = - (9 / 2 : R) := 
by
  sorry

end calculate_f_neg_two_l307_307472


namespace five_digit_divisible_by_twelve_count_l307_307512

theorem five_digit_divisible_by_twelve_count : 
  let count_four_digit_multiples_of_12 := 
      ((9996 - 1008) / 12 + 1) in
  let count_five_digit_multiples_of_12 := 
      (9 * count_four_digit_multiples_of_12) in
  count_five_digit_multiples_of_12 = 6732 :=
by
  sorry

end five_digit_divisible_by_twelve_count_l307_307512


namespace identify_letters_l307_307257

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l307_307257


namespace total_candies_third_set_l307_307702

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l307_307702


namespace one_leg_divisible_by_3_l307_307653

theorem one_leg_divisible_by_3 (a b c : ℕ) (h : a^2 + b^2 = c^2) : (3 ∣ a) ∨ (3 ∣ b) :=
by sorry

end one_leg_divisible_by_3_l307_307653


namespace find_m_of_pure_imaginary_l307_307610

theorem find_m_of_pure_imaginary (m : ℝ) (h1 : (m^2 + m - 2) = 0) (h2 : (m^2 - 1) ≠ 0) : m = -2 :=
by
  sorry

end find_m_of_pure_imaginary_l307_307610


namespace find_rationals_l307_307874

theorem find_rationals (
  a b c d m n : ℤ)
  (ε : ℝ) 
  (h_det_nonzero : a * d - b * c ≠ 0)
  (h_ε_pos : ε > 0) :
  ∃ x y : ℚ, 0 < |(a : ℚ) * x + b * y - (m : ℚ)| ∧ |(a : ℚ) * x + b * y - (m : ℚ)| < ε ∧
             0 < |(c : ℚ) * x + d * y - (n : ℚ)| ∧ |(c : ℚ) * x + d * y - (n : ℚ)| < ε :=
begin
  sorry
end

end find_rationals_l307_307874


namespace tangent_line_ln_l307_307657

theorem tangent_line_ln (a : ℝ) : 
  (∀ x > 0, (∃ m : ℝ, m > 0 ∧ (1 + real.log m = 1 + real.log x ∧ 
    ∀ y, y = 1 + real.log x → y - 1 - real.log m = (1 / m) * (x - m))) → a = 1) :=
by sorry

end tangent_line_ln_l307_307657


namespace clock_angle_at_3_40_l307_307308

def hour_hand_position (hour : ℕ) (minute : ℕ) : ℝ :=
  30 * hour + 0.5 * minute

def minute_hand_position (minute : ℕ) : ℝ :=
  6 * minute

def calculate_angle (h_pos : ℝ) (m_pos : ℝ) : ℝ :=
  let angle := abs (m_pos - h_pos) in
  min angle (360 - angle)

theorem clock_angle_at_3_40 : calculate_angle (hour_hand_position 3 40) (minute_hand_position 40) = 130 :=
by
  sorry

end clock_angle_at_3_40_l307_307308


namespace jose_speed_l307_307179

theorem jose_speed
  (distance : ℕ) (time : ℕ)
  (h_distance : distance = 4)
  (h_time : time = 2) :
  distance / time = 2 := by
  sorry

end jose_speed_l307_307179


namespace count_pairs_l307_307911

theorem count_pairs (a b : ℤ) (ha : 1 ≤ a ∧ a ≤ 42) (hb : 1 ≤ b ∧ b ≤ 42) (h : a^9 % 43 = b^7 % 43) : (∃ (n : ℕ), n = 42) :=
  sorry

end count_pairs_l307_307911


namespace letters_identity_l307_307235

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l307_307235


namespace value_of_x_squared_plus_one_over_x_squared_l307_307917

noncomputable def x: ℝ := sorry

theorem value_of_x_squared_plus_one_over_x_squared (h : 20 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = 23 :=
sorry

end value_of_x_squared_plus_one_over_x_squared_l307_307917


namespace parametric_curve_length_l307_307843

theorem parametric_curve_length :
  let x (t : ℝ) := 3 * Real.sin t
  let y (t : ℝ) := 3 * Real.cos t
  t ∈ Icc (0:ℝ) (2 * Real.pi)
  ∃ L, L = 6 * Real.pi := sorry

end parametric_curve_length_l307_307843


namespace expression_defined_if_x_not_3_l307_307540

theorem expression_defined_if_x_not_3 (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end expression_defined_if_x_not_3_l307_307540


namespace ab_necessary_not_sufficient_l307_307065

theorem ab_necessary_not_sufficient (a b : ℝ) : 
  (ab > 0) ↔ ((a ≠ 0) ∧ (b ≠ 0) ∧ ((b / a + a / b > 2) → (ab > 0))) := 
sorry

end ab_necessary_not_sufficient_l307_307065


namespace correct_calculation_l307_307326

theorem correct_calculation :
  (-real.cbrt 5) * (1 / (real.cbrt (-5))) = 1 :=
by
  sorry

end correct_calculation_l307_307326


namespace triangle_angles_l307_307270

theorem triangle_angles 
  (R r : ℝ) (p : ℝ)
  (hR : R = 170) 
  (hr : r = 12) 
  (hp : p = 416) :
  let s := p / 2,
      a := 52.0,
      b := 160.0,
      c := 204.0,
      alpha := real.arcsin (a / (2 * R)) * (180 / real.pi), 
      beta := real.arcsin (b / (2 * R)) * (180 / real.pi), 
      gamma := 180.0 - (alpha + beta)
  in 
  (alpha ≈ 8 + 47.8 / 60) ∧ (beta ≈ 28 + 4.4 / 60) ∧ (gamma ≈ 143 + 7.8 / 60) :=
by
  sorry

end triangle_angles_l307_307270


namespace inequality_system_solution_l307_307262

theorem inequality_system_solution (x : ℤ) (h1 : 4 * (x + 1) ≤ 7 * x + 10)
                                  (h2 : x - 5 < (x - 8) / 3) :
                                  x ∈ {0, 1, 2, 3} :=
by
  -- proof steps can be filled here
  sorry

end inequality_system_solution_l307_307262


namespace curve_equation_and_range_of_m_l307_307948

theorem curve_equation_and_range_of_m:
  (∀ P : ℝ × ℝ, let x := P.1, y := P.2 in 
    (sqrt (x^2 + (y-1)^2) - abs y = 1) 
    ↔ (x^2 = 4*y ∧ y ≥ 0) ∨ (x = 0 ∧ y < 0))
  ∧ (∀ k : ℝ, ∀ m : ℝ, m > 0 → 
    (∀ A B : ℝ × ℝ, 
        let x1 := A.1, y1 := A.2, x2 := B.1, y2 := B.2 in 
        (y1 = k*x1 + m ∧ (x1^2 = 4*y1 ∧ y1 ≥ 0) ∨ (x1 = 0 ∧ y1 < 0))
        ∧ (y2 = k*x2 + m ∧ (x2^2 = 4*y2 ∧ y2 ≥ 0) ∨ (x2 = 0 ∧ y2 < 0)) →
           ((m-1)^2 - 4*m < 0)) 
    ↔ (3 - 2*sqrt (2) < m) ∧ (m < 3 + 2*sqrt (2)))
:=
begin
  sorry
end

end curve_equation_and_range_of_m_l307_307948


namespace lcm_48_147_l307_307046

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := sorry

end lcm_48_147_l307_307046


namespace cos_neg_45_degree_l307_307010

theorem cos_neg_45_degree :
  real.cos (-π / 4) = real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_45_degree_l307_307010


namespace line_y_axis_intersection_l307_307777

-- Conditions: Line contains points (3, 20) and (-9, -6)
def line_contains_points : Prop :=
  ∃ m b : ℚ, ∀ (x y : ℚ), ((x = 3 ∧ y = 20) ∨ (x = -9 ∧ y = -6)) → (y = m * x + b)

-- Question: Prove that the line intersects the y-axis at (0, 27/2)
theorem line_y_axis_intersection :
  line_contains_points → (∃ (y : ℚ), y = 27/2) :=
by
  sorry

end line_y_axis_intersection_l307_307777


namespace unique_reconstruction_possible_l307_307214

def can_reconstruct_faces (a b c d e f : ℤ) : 
  Prop :=
  let edges := [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]
  in ∀ (sums : list ℤ), sums = edges →
    ∃ (a' b' c' d' e' f' : ℤ), a = a' ∧ b = b' ∧ c = c' ∧ d = d' ∧ e = e' ∧ f = f'

theorem unique_reconstruction_possible :
  ∀ (a b c d e f : ℤ),
    can_reconstruct_faces a b c d e f :=
begin
  sorry
end

end unique_reconstruction_possible_l307_307214


namespace arithmetic_sequence_properties_l307_307563

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 + a 3 = 21) 
  (h2 : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n, a n = -4 * n + 15 ∨ a n = 4 * n - 1) := 
by
  sorry

end arithmetic_sequence_properties_l307_307563


namespace smallest_positive_x_floor_value_l307_307961

def g (x : ℝ) : ℝ := Real.sin x + 3 * Real.cos x + 4 * Real.tan x

theorem smallest_positive_x_floor_value :
  ∃ s > 0, g s = 0 ∧ Int.floor s = 4 :=
sorry

end smallest_positive_x_floor_value_l307_307961


namespace problem1_problem2_l307_307490
-- Import Mathlib

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x^2 + 1

-- Problem 1: Proving the monotonicity condition implies a ≤ e/2
theorem problem1 (a : ℝ) : (∀ x : ℝ, 0 < x → (exp x - 2 * a * x) ≥ 0) → a ≤ (exp 1) / 2 :=
sorry

-- Problem 2: Prove the given inequality when a = 1
theorem problem2 (x1 x2 : ℝ) (h : x1 ≠ x2) : 
  (exp x1 - 1 * x1^2 + 1 - (exp x2 - 1 * x2^2 + 1)) / (x1 - x2) > 2 - 2 * log 2 :=
sorry

end problem1_problem2_l307_307490


namespace identify_letters_l307_307258

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l307_307258


namespace f_increasing_on_real_l307_307227

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem: The function f(x) = x^3 + x is increasing on the entire real line ℝ.
theorem f_increasing_on_real : ∀ x y : ℝ, x < y → f(x) < f(y) :=
by
  sorry

end f_increasing_on_real_l307_307227


namespace systematic_sampling_correct_l307_307770

-- Define the conditions for the problem
def num_employees : ℕ := 840
def num_selected : ℕ := 42
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Define systematic sampling interval
def sampling_interval := num_employees / num_selected

-- Define the length of the given interval
def interval_length := interval_end - interval_start + 1

-- The theorem to prove
theorem systematic_sampling_correct :
  (interval_length / sampling_interval) = 12 := sorry

end systematic_sampling_correct_l307_307770


namespace solve_x_squared_eq_four_l307_307920

theorem solve_x_squared_eq_four (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 := 
by sorry

end solve_x_squared_eq_four_l307_307920


namespace large_buckets_needed_l307_307776

def capacity_large_bucket (S: ℚ) : ℚ := 2 * S + 3

theorem large_buckets_needed (n : ℕ) (L S : ℚ) (h1 : L = capacity_large_bucket S) (h2 : L = 4) (h3 : 2 * S + n * L = 63)
: n = 16 := sorry

end large_buckets_needed_l307_307776


namespace jean_jail_time_l307_307175

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l307_307175


namespace baseball_team_groups_l307_307283

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) (h_new : new_players = 48) (h_return : returning_players = 6) (h_per_group : players_per_group = 6) : (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end baseball_team_groups_l307_307283


namespace general_formula_l307_307070

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 0 then 0 else sorry -- a placeholder; zero is invalid in this context but required by ℕ type

theorem general_formula (n : ℕ) (h : n > 0) : 
  a_seq (a_seq 1 = 0 ∧ (∀ k > 0, a_seq (k + 1) = a_seq k + 2 * k - 1)) = 
  (n - 1) ^ 2 := 
sorry

end general_formula_l307_307070


namespace liam_draws_segments_l307_307972

-- Define the parameters of the problem
def concentric_circles (draw_chords : ℕ → ℕ → Prop) := 
  ∀ n m, draw_chords n m
 
def angle_ABC (abc_angle : angle) := 
  abc_angle = 60

-- State the theorem using the above definitions
theorem liam_draws_segments (draw_chords : ℕ → ℕ → Prop) (abc_angle : angle)
  [conc_circles : concentric_circles draw_chords] [abc_cond : angle_ABC abc_angle]:
  ∃ n, n = 3 :=
by
  -- proof will eventually go here
  sorry

end liam_draws_segments_l307_307972


namespace number_of_lattice_points_on_line_segment_l307_307844

-- Define the gcd function using the built-in method from mathlib
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the coordinates and the differences
def x1 := 5
def y1 := 13
def x2 := 47
def y2 := 275
def Δx := x2 - x1
def Δy := y2 - y1

-- Define the gcd of Δx and Δy
def gcd_val := gcd Δy Δx

-- Define the number of lattice points
def num_lattice_points : ℕ := (gcd_val + 1)

-- The theorem statement
theorem number_of_lattice_points_on_line_segment :
  num_lattice_points = 3 :=
by
  -- Proof omitted
  sorry

end number_of_lattice_points_on_line_segment_l307_307844


namespace quadratic_two_distinct_real_roots_l307_307925

theorem quadratic_two_distinct_real_roots (k : ℝ) : (k < 1) ↔ (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 ^ 2 - 6 * x1 + 9 * k = 0) ∧ (x2 ^ 2 - 6 * x2 + 9 * k = 0)) :=
by
  have hΔ : 36 - 36 * k > 0 ↔ k < 1 := 
    sorry
  exact hΔ.sorry

end quadratic_two_distinct_real_roots_l307_307925


namespace bushes_needed_for_circular_garden_l307_307133

noncomputable def num_bushes_needed (radius : ℝ) (spacing : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  circumference / spacing

theorem bushes_needed_for_circular_garden :
  num_bushes_needed 15 2 ≈ 47 :=
begin
  sorry
end

end bushes_needed_for_circular_garden_l307_307133


namespace range_of_m_l307_307923

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → (7 / 4) ≤ (x^2 - 3 * x + 4) ∧ (x^2 - 3 * x + 4) ≤ 4) ↔ (3 / 2 ≤ m ∧ m ≤ 3) := 
sorry

end range_of_m_l307_307923


namespace percent_alcohol_in_new_solution_l307_307755

theorem percent_alcohol_in_new_solution (orig_vol : ℝ) (orig_percent : ℝ) (add_alc : ℝ) (add_water : ℝ) :
  orig_percent = 5 → orig_vol = 40 → add_alc = 5.5 → add_water = 4.5 →
  (((orig_vol * (orig_percent / 100) + add_alc) / (orig_vol + add_alc + add_water)) * 100) = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end percent_alcohol_in_new_solution_l307_307755


namespace altitudes_of_acute_triangle_l307_307076

variables {A B C D E F : Type*}
variables [linear_ordered_field ℝ] [add_group ℝ] [linear_ordered_add_comm_group ℝ]
variables [module ℝ D] [module ℝ E] [module ℝ F]
variables {ABC : triangle ℝ}
variables {triangleABC : ∀ {p1 p2 : ℝ}, p1 * p2 > 0}

variables (angleEDC : ∀ {p1 p2 : ℝ}, p1 = p2)
variables (angleCDF : ∀ {p1 p2 : ℝ}, p1 = p2)
variables (angleFEA : ∀ {p1 p2 : ℝ}, p1 = p2)
variables (angleAED : ∀ {p1 p2 : ℝ}, p1 = p2)
variables (angleDFB : ∀ {p1 p2 : ℝ}, p1 = p2)
variables (angleBFE : ∀ {p1 p2 : ℝ}, p1 = p2)

theorem altitudes_of_acute_triangle :
  ∀ (ABC : triangle ℝ),
  ∀ (D E F : ℝ),
  acute_triangle ABC →
  on_side_ABC D E F ABC →
  angleEDC D E F ABC →
  angleCDF D E F ABC →
  angleFEA D E F ABC →
  angleAED D E F ABC →
  angleDFB D E F ABC →
  angleBFE D E F ABC →
  are_altitudes ABC D E F :=
sorry

end altitudes_of_acute_triangle_l307_307076


namespace greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l307_307721

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem greatest_divisor_of_546_smaller_than_30_and_factor_of_126 :
  ∃ (d : ℕ), d < 30 ∧ is_factor d 546 ∧ is_factor d 126 ∧ ∀ e : ℕ, e < 30 ∧ is_factor e 546 ∧ is_factor e 126 → e ≤ d := 
sorry

end greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l307_307721


namespace find_expression_value_l307_307474

-- Given conditions
variables {a b : ℝ}

-- Perimeter condition
def perimeter_condition (a b : ℝ) : Prop := 2 * (a + b) = 10

-- Area condition
def area_condition (a b : ℝ) : Prop := a * b = 6

-- Goal statement
theorem find_expression_value (h1 : perimeter_condition a b) (h2 : area_condition a b) :
  a^3 * b + 2 * a^2 * b^2 + a * b^3 = 150 :=
sorry

end find_expression_value_l307_307474


namespace difference_in_hop_and_glide_lengths_l307_307993

theorem difference_in_hop_and_glide_lengths : 
  ∀ (P Q : ℕ) (D : ℝ), D = 7920 →
  51 = P →
  ∀ (Rita_hops Gideon_glides markers : ℕ), Rita_hops = 56 → Gideon_glides = 16 → markers = P-1 →
  let hops_total := (markers * Rita_hops : ℕ) in
  let glides_total := (markers * Gideon_glides : ℕ) in
  let hop_length := D / hops_total in
  let glide_length := D / glides_total in
  let length_difference := glide_length - hop_length in
  length_difference ≈ 7.0714
  := 
begin
  intros P Q D hD hP Rita_hops Gideon_glides markers hRita_hops hGideon_glides hmarkers,
  let hop_total := markers * hRita_hops,
  let glide_total := markers * hGideon_glides,
  let hop_length := D / hop_total,
  let glide_length := D / glide_total,
  let length_difference := glide_length - hop_length,
  sorry
end

end difference_in_hop_and_glide_lengths_l307_307993


namespace find_parabola_eq_find_vector_dot_product_l307_307868

noncomputable section
open Classical

-- Definitions of necessary points and functions
def parabola_eq (p : ℝ) : set (ℝ × ℝ) := { q | q.2 ^ 2 = 2 * p * q.1 }

def point_inside_parabola : ℝ × ℝ := (3, 1)

def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def min_sum_of_distances (p : ℝ) (Q : ℝ × ℝ) : ℝ := 
  if h : Q ∈ (parabola_eq p)
  then dist Q (focus p) + dist Q point_inside_parabola
  else 0

-- Prove the equation of the parabola given the conditions
theorem find_parabola_eq (p : ℝ) (hp : p > 0) (hmin : min_sum_of_distances p = 4) : parabola_eq p = { q | q.2 ^ 2 = 4 * q.1 } :=
by sorry

-- Definitions for vectors OA and OB, line through focus
def line_through_focus (k : ℝ) (p : ℝ) : set (ℝ × ℝ) := 
  { q | q.2 = k * (q.1 - focus p).1 }

def intersection_with_parabola (k : ℝ) (p : ℝ) : set (ℝ × ℝ) := (parabola_eq p) ∩ (line_through_focus k p)

def vector_dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def OA : ℝ × ℝ := (1, 2)
def OB : ℝ × ℝ := (1, -2)

-- Prove the value of OA . OB
theorem find_vector_dot_product (p : ℝ) (hp : p > 0) : vector_dot_product OA OB = -3 :=
by sorry

end find_parabola_eq_find_vector_dot_product_l307_307868


namespace median_of_S_l307_307550

noncomputable def S (x : ℕ) : set ℝ := {x - 1, 3 * x + 3, 2 * x - 4}

theorem median_of_S (x : ℕ) (h_prime : Nat.Prime x) (h_mean : (x - 1 + (3 * x + 3) + (2 * x - 4)) / 3 = 3.3333333333333335) :
  ∃ m, m = 2 * x - 4 ∧ m = 0 :=
by
  sorry

end median_of_S_l307_307550


namespace solution_set_l307_307893

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem solution_set (x : ℝ) : f (Real.log2 x) < f 2 ↔ (4 < x ∨ (0 < x ∧ x < 1)) := sorry

end solution_set_l307_307893


namespace real_axis_length_of_hyperbola_l307_307269

-- Define the parabola y^2 = 16x and its directrix.
def parabola (x y : ℝ) : Prop := y^2 = 16 * x
def directrix (x : ℝ) : Prop := x = -4

-- Define the hyperbola C
structure Hyperbola :=
  (center : ℝ × ℝ := (0, 0))
  (fociOnXAxis : Prop := True) -- Foci are on the x-axis

-- Define the points A and B and the distance between them being 4.
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def pointsAandB (A B : ℝ × ℝ) : Prop :=
  (parabola A.1 A.2) ∧
  (parabola B.1 B.2) ∧
  (directrix A.1) ∧
  (directrix B.1) ∧
  (distance A B = 4)

-- The theorem to be proved.
theorem real_axis_length_of_hyperbola :
  ∀ (C : Hyperbola) (A B : ℝ × ℝ),
  pointsAandB A B →
  (distance A B = 4) →
  fociOnXAxis C →
  length_of_real_axis C = 4 :=
sorry

end real_axis_length_of_hyperbola_l307_307269


namespace sin_sum_positive_l307_307989

theorem sin_sum_positive (x : ℝ) (h₀ : 0 < x) (h₁ : x < real.pi) : 
  sin x + (1 / 2) * sin (2 * x) + (1 / 3) * sin (3 * x) > 0 := 
sorry

end sin_sum_positive_l307_307989


namespace radical_center_on_Euler_line_l307_307438

-- Definitions for geometric entities and conditions
variables {A B C D E F : Type} [euclidean_geometry A B C D E F]

-- Points and their equal distances
variables (D E F : Point)
variables (DB DC EC EA FA FB : ℝ)
variable (BDC_equals_CEA_equals_AFB : angle A B D = angle C E A ∧ angle B D C = angle C E A ∧ angle A F B = angle A F B)

-- Circles defined with centers D, E, F and passing through specific points
def Omega_D : Circle := Circle.centered_at D (DB)
def Omega_E : Circle := Circle.centered_at E (EC)
def Omega_F : Circle := Circle.centered_at F (FA)

-- To prove: The radical center of Ω_D, Ω_E, Ω_F lies on the Euler line of ΔDEF
theorem radical_center_on_Euler_line (H : orthocenter (triangle DEF)) :
  radical_center Omega_D Omega_E Omega_F ∈ Euler_line triangle DEF :=
sorry

end radical_center_on_Euler_line_l307_307438


namespace specific_gravity_is_0_6734_l307_307364

noncomputable def sphere_radius : ℝ := 8 -- Since diameter is 16 cm

noncomputable def dry_surface_area : ℝ := 307.2 -- cm²

noncomputable def buoyant_force_balance (x : ℝ) : Prop :=
  let volume_sphere := (4 / 3) * Real.pi * (sphere_radius ^ 3) in
  let height_dry_cap := dry_surface_area / (2 * Real.pi * sphere_radius) in
  let height_submerged_cap := 2 * sphere_radius - height_dry_cap in
  let volume_submerged_cap := (1 / 3) * Real.pi * (height_submerged_cap ^ 2) * (3 * sphere_radius - height_submerged_cap) in
  x = volume_submerged_cap / volume_sphere

theorem specific_gravity_is_0_6734 : ∃ x, buoyant_force_balance x ∧ x = 0.6734 :=
by 
  -- The proof would go here
  sorry

end specific_gravity_is_0_6734_l307_307364


namespace max_value_f_range_g_l307_307894

-- Define the function f(x) = sqrt(3) * sin x + cos x
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * Real.sin x + Real.cos x

-- Prove that the maximum value of f(x) is 2
theorem max_value_f : ∃ x : ℝ, f x = 2 :=
sorry

-- Define the function g(x) = f(x) * cos x
noncomputable def g (x : ℝ) : ℝ := f x * Real.cos x

-- Prove that the range of g(x) for x ∈ [0, π/2] is [1, 3/2]
theorem range_g : ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → 1 ≤ g x ∧ g x ≤ 3 / 2 :=
sorry

end max_value_f_range_g_l307_307894


namespace community_families_l307_307796

theorem community_families (A B X : ℕ) (hA : A = 35) (hB : B = 65) (hX : X = 20) : A + B - X = 80 := 
by 
  rw [hA, hB, hX]
  exact rfl

end community_families_l307_307796


namespace sampling_method_is_systematic_l307_307774

-- Define the conditions:
def num_classes : Nat := 20
def students_per_class : Nat := 50
def selected_student_number : Nat := 16

-- The theorem statement asserting that the sampling method is "Systematic sampling" given the conditions.
theorem sampling_method_is_systematic :
  (selected_student_number = 16) → (students_per_class = 50) → (num_classes = 20) → 
  "The sampling method used is Systematic sampling" :=
by
  intros h1 h2 h3
  sorry -- Proof goes here.

end sampling_method_is_systematic_l307_307774


namespace sufficient_not_necessary_condition_l307_307599

variable {a : ℝ}

theorem sufficient_not_necessary_condition (ha : a > 1 / a^2) :
  a^2 > 1 / a ∧ ∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2) :=
by
  sorry

end sufficient_not_necessary_condition_l307_307599


namespace square_term_decomposition_l307_307273

theorem square_term_decomposition :
  ∀ x : ℝ, ∃ c : ℝ, x^2 + 8 * x + 20 = (x + 4) ^ 2 + c :=
begin
  intro x,
  use 4,
  calc
    x^2 + 8 * x + 20
        = x^2 + 8 * x + 16 + 4 : by ring
    ... = (x + 4) ^ 2 + 4     : by rw add_assoc (x^2 + 8 * x) 16 4
end

end square_term_decomposition_l307_307273


namespace inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l307_307456

theorem inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared 
  (a b c : ℝ)
  (h_sum : a + b + c = 0)
  (d : ℝ) 
  (h_d : d = max (abs a) (max (abs b) (abs c))) : 
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by 
  sorry

end inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l307_307456


namespace general_solution_to_differential_eq_l307_307039

theorem general_solution_to_differential_eq :
  ∃ (C1 C2 : ℝ), ∀ (y : ℝ → ℝ),
    (∀ x, deriv y x = x) →
    (∀ x, deriv (deriv y) x + 2 * deriv y x + 5 * y x = Real.exp (-x) * Real.cos (2 * x)) →
    y = λ x, (C1 * Real.cos (2 * x) + C2 * Real.sin (2 * x)) * Real.exp (-x) + (1/4 * x * Real.exp (-x) * Real.sin (2 * x))
:= by
  sorry

end general_solution_to_differential_eq_l307_307039


namespace a_100_is_rational_simplified_l307_307790

/-- A sequence defined recursively with initial terms a1 and a2, and a recursive relation -/
def a : ℕ → ℚ
| 1 := 2
| 2 := 7 / 5
| n := a (n - 2) * a (n - 1) / (3 * a (n - 2) - a (n - 1))

/-- Main theorem stating the required property of the sequence -/
theorem a_100_is_rational_simplified :
  ∃ (p q : ℕ), Nat.coprime p q ∧ a 100 = p / q ∧ p + q = 97 := sorry

end a_100_is_rational_simplified_l307_307790


namespace chord_lengths_arithmetic_sequence_l307_307328

theorem chord_lengths_arithmetic_sequence 
  (n : ℕ) (a₁ aₙ : ℝ) (d : ℝ)
  (h₁ : n > 1)
  (h₂ : ∃ p : ℝ × ℝ, p = (5 / 2, 3 / 2))
  (h₃ : ∀ k, 1 ≤ k ∧ k ≤ n → a₁ + (k - 1) * d = aₖ)
  (h₄ : (1 / 6 : ℝ) < d ∧ d ≤ (1 / 3 : ℝ))
  (h₅ : x ^ 2 + y ^ 2 = 5 * x) :
  n ∈ {4, 5, 6} :=
sorry

end chord_lengths_arithmetic_sequence_l307_307328


namespace max_distinct_prime_factors_a_l307_307957

-- Definitions:
variables {a b : ℕ}

-- Conditions:
def positive_integer (n : ℕ) : Prop := n > 0

def gcd_has_10_distinct_primes (a b : ℕ) : Prop :=
∃ ps : Finset ℕ, ps.card = 10 ∧ ∀ p ∈ ps, p ∣ gcd a b

def lcm_has_35_distinct_primes (a b : ℕ) : Prop :=
∃ ps : Finset ℕ, ps.card = 35 ∧ ∀ p ∈ ps, p ∣ lcm a b

def fewer_distinct_prime_factors (a b : ℕ) : Prop :=
((Finset.filter nat.prime (factors a)).card) < ((Finset.filter nat.prime (factors b)).card)

-- Theorem Statement:
theorem max_distinct_prime_factors_a (a b : ℕ)
  (H1: positive_integer a)
  (H2: positive_integer b)
  (H3: gcd_has_10_distinct_primes a b)
  (H4: lcm_has_35_distinct_primes a b)
  (H5: fewer_distinct_prime_factors a b) :
  ((Finset.filter nat.prime (factors a)).card) ≤ 22 :=
  sorry

end max_distinct_prime_factors_a_l307_307957


namespace base8_arithmetic_l307_307805

/-- Given three numbers in base 8: 10_8, 26_8, and 13_8, we want to prove that
    (10_8 + 26_8) - 13_8 in base 8 equals 23_8. -/

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let digits := n.digits 8
  digits.foldl (λ acc d, acc * 8 + d) 0

def convert_base10_to_base8 (n : ℕ) : ℕ :=
  nat.foldr (λ d acc, acc * 10 + d) 0 (n.digits 8)

/-- The main theorem stating that (10_8 + 26_8) - 13_8 equals 23_8 in base 8. -/
theorem base8_arithmetic : 
  convert_base10_to_base8 ((convert_base8_to_base10 10 + convert_base8_to_base10 26) - convert_base8_to_base10 13) = 23 :=
by sorry

end base8_arithmetic_l307_307805


namespace nested_fraction_simplifies_l307_307385

theorem nested_fraction_simplifies : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := 
by 
  sorry

end nested_fraction_simplifies_l307_307385


namespace total_candies_in_third_set_l307_307685

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l307_307685


namespace sequence_general_formula_l307_307942

theorem sequence_general_formula (a : ℕ → ℚ) (h₁ : a 1 = 2 / 3)
  (h₂ : ∀ n : ℕ, a (n + 1) = a n + a n * a (n + 1)) : 
  ∀ n : ℕ, a n = 2 / (5 - 2 * n) :=
by 
  sorry

end sequence_general_formula_l307_307942


namespace percentage_of_trucks_is_8_l307_307985

variables (T R C : ℕ)

def total_matchbox_cars : ℕ := 125
def regular_car_percentage : ℕ := 64
def regular_cars (total : ℕ) (percent : ℕ) : ℕ := (percent * total) / 100
def convertibles : ℕ := 35
def trucks (total : ℕ) (regular : ℕ) (convertible : ℕ) : ℕ := total - regular - convertible
def percentage_trucks (trucks : ℕ) (total : ℕ) : ℕ := (trucks * 100) / total

theorem percentage_of_trucks_is_8 :
  percentage_trucks (trucks total_matchbox_cars (regular_cars total_matchbox_cars regular_car_percentage) convertibles) total_matchbox_cars = 8 :=
by
  -- Definitions for clarity
  let total := total_matchbox_cars
  let regular := regular_cars total regular_car_percentage
  let convertible := convertibles
  let truck := trucks total regular convertible
  -- The main proof goal
  have h1 : regular = 80 := sorry,
  have h2 : truck = 10 := sorry,
  have h3 : percentage_trucks truck total = 8 := sorry,
  exact h3

end percentage_of_trucks_is_8_l307_307985


namespace sum_of_sequence_d_n_l307_307097

variable {ℕ : Type}

noncomputable def S (n : ℕ) : ℕ := 2 * (2 ^ n) - 2
def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := 2 * n - 1
def c (n : ℕ) : ℕ := b (a n)
def d (n : ℕ) : ℕ := (c n + 1) / (c n * c (n + 1))
def T (n : ℕ) : ℕ := Σ i in Finset.range n, d i

theorem sum_of_sequence_d_n (n : ℕ) (hS : ∀ n, S n = 2 * a n - 2) (hb3 : b 3 = 5) (hb5 : b 5 = 9):
  T n = 1 / 3 - 1 / (2 ^ (n + 2) - 1) :=
  by
  sorry

end sum_of_sequence_d_n_l307_307097


namespace tan_addition_l307_307127

noncomputable def tan (x : ℝ) : ℝ := Math.tan x
noncomputable def cot (x : ℝ) : ℝ := 1 / Math.tan x

theorem tan_addition
  (x y : ℝ)
  (h1 : tan x + tan y = 25)
  (h2 : cot x + cot y = 30) : 
  tan (x + y) = 150 := by
  sorry

end tan_addition_l307_307127


namespace modular_inverse_of_34_mod_35_l307_307047

theorem modular_inverse_of_34_mod_35 : ∃ a : ℤ, 0 ≤ a ∧ a < 35 ∧ (34 * a) % 35 = 1 ∧ a = 34 :=
by
  -- Conditions
  have h : 34 % 35 = -1 % 35, by nat_mod_eq_neg,
  -- Proofs leading to conclusion a = 34 omitted
  use 34
  split; linarith
  split; linarith
  split
  -- modulo conditions
  calc 34 * 34 % 35 = 34 % 35
  sorry

end modular_inverse_of_34_mod_35_l307_307047


namespace multiply_negatives_l307_307808

theorem multiply_negatives : (-2) * (-3) = 6 :=
  by 
  sorry

end multiply_negatives_l307_307808


namespace relationship_among_abc_l307_307859

noncomputable def a : ℝ := 2 ^ (1 / 2)
noncomputable def b : ℝ := (2 ^ (Real.log 3 / Real.log 2)) ^ (-1 / 2)
noncomputable def c : ℝ := Real.cos (50 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) + Real.cos (140 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

theorem relationship_among_abc : a > b ∧ b > c := 
by
  sorry

end relationship_among_abc_l307_307859


namespace third_set_candies_l307_307691

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l307_307691


namespace minimize_cost_and_piers_l307_307763

def total_cost (x : ℝ) : ℝ := 12000 * (50 / x + Real.log (x + 12)) - 36500

def num_piers (x : ℝ) : ℕ := Nat.floor (1200 / x - 1)

theorem minimize_cost_and_piers :
  (∃ x : ℝ, x = 60 ∧ num_piers x = 19 ∧ total_cost x ≈ 24740) :=
by 
  have hx : 60 ≠ 0 := by norm_num
  use 60
  split
  . exact rfl
  split
  . unfold num_piers; simp_rw [Nat.floor, sub_eq_add_neg, add_neg_iff_eq, add_comm]; simp
  . unfold total_cost; sorry

end minimize_cost_and_piers_l307_307763


namespace matching_pair_probability_l307_307348

theorem matching_pair_probability (m n : ℕ) : 
  let deck_size := 50,
      remaining_deck_size := deck_size - 3,
      total_ways_to_select_two := Nat.choose remaining_deck_size 2,
      pair_ways_triplet_removed := Nat.choose 2 2,
      pair_ways_remaining := 9 * Nat.choose 5 2,
      total_eligible_pairs := pair_ways_triplet_removed + pair_ways_remaining,
      probability_num := total_eligible_pairs,
      probability_denom := total_ways_to_select_two
  in Nat.gcd probability_num probability_denom = 1 →
     m = probability_num →
     n = probability_denom →
     m + n = 1172 :=
by
  trivial

end matching_pair_probability_l307_307348


namespace triangle_inequality_l307_307991

-- Define basic geometric objects and properties
variable {α : Type*} [LinearOrder α]

-- Define points A, B, C and M
variables (A B C M : α)

-- Define distances from M to sides BC, CA, AB
variables (R_a R_b R_c : ℝ₀)

-- Define perpendicular distances from vertices to line through M
variables (d_a d_b d_c : ℝ₀)

-- Define corresponding side lengths
variables (a b c : ℝ₀)

theorem triangle_inequality (A B C M : α) (R_a R_b R_c d_a d_b d_c a b c : ℝ₀) :
  a * R_a + b * R_b + c * R_c ≥ 2 * (a * d_a + b * d_b + c * d_c) :=
sorry

end triangle_inequality_l307_307991


namespace remainder_3_pow_20_mod_7_l307_307317

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end remainder_3_pow_20_mod_7_l307_307317


namespace perfect_square_form_l307_307990

theorem perfect_square_form (k : ℕ) (hk : k > 0) : 
  ∃ m : ℕ, n = m^2 :=
begin
  let n := (10^(2 * k) + 4 * 10^k + 4) / 9,
  have h₁ : (10^k + 2) % 3 = 0 := by sorry,
  use (10^k + 2) / 3,
  use n,
  have h₂ : n = ((10^k + 2) / 3)^2 := by sorry,
  exact ⟨(10^k + 2) / 3, h₂⟩,
end

end perfect_square_form_l307_307990


namespace letters_identity_l307_307238

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l307_307238


namespace carl_total_cost_l307_307811

namespace IndexCards

open Nat

def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

def cards_needed (num_students: ℕ) (cards_per_student: ℕ) : ℕ :=
  num_students * cards_per_student

def cards_per_grade_6 (students_6th graders cards_6th_grade: ℕ) : ℕ :=
  cards_needed 20 8

def cards_per_grade_7 (students_7th graders cards_7th_grade: ℕ) : ℕ :=
  cards_needed 25 10

def cards_per_grade_8 (students_8th graders cards_8th_grade: ℕ) : ℕ :=
  cards_needed 30 12

def cards_per_period (cards_6th students_7th graders cards_8th: ℕ) : ℕ :=
  cards_per_grade_6 20 8 + cards_per_grade_7 25 10 + cards_per_grade_8 30 12

def total_cards_needed (cards_per_period_num_periods: ℕ) : ℕ :=
  cards_per_period 1 2 3 * 6

def total_packs_needed (total_cards cards_per_pack: ℕ) : ℕ :=
  (total_cards + (cards_per_pack - 1)) / cards_per_pack  -- Ceiling division

def total_cost (packs_needed cost_per_pack: ℕ) : ℕ :=
  packs_needed * cost_per_pack

theorem carl_total_cost : total_cost (total_packs_needed (total_cards_needed (cards_per_period (20 8) (25 10) (30 12)) cards_per_pack)) cost_per_pack = 279 :=
  by simp [cards_per_period, total_cards_needed, total_packs_needed, total_cost, cards_per_grade_6, cards_per_grade_7, cards_per_grade_8, cards_needed]; sorry

end IndexCards

end carl_total_cost_l307_307811


namespace montys_reunion_family_members_l307_307779

noncomputable def cost_per_bucket (price : ℕ) : ℕ := 12
noncomputable def people_per_bucket (people : ℕ) : ℕ := 6
noncomputable def total_cost (cost : ℕ) : ℕ := 72
noncomputable def total_family_members (cost people : ℕ) : ℕ := (total_cost cost) / (cost_per_bucket cost) * (people_per_bucket people)

theorem montys_reunion_family_members (cost price people : ℕ) (H1 : price = 12) (H2 : people = 6) (H3 : cost = 72) : total_family_members cost people = 36 := 
by 
  rw [total_family_members, H1, H2, H3]
  -- The proof calculation goes here
  sorry

end montys_reunion_family_members_l307_307779


namespace arithmetic_mean_of_18_27_45_l307_307719

theorem arithmetic_mean_of_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_of_18_27_45_l307_307719


namespace area_of_triangle_line_equation_l307_307872

noncomputable def ellipse_C (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Definitions of foci and the given condition on the ratio
def foci (a b c : ℝ) : Prop := 
  a^2 = b^2 + c^2

def focal_ratio (a c : ℝ) : Prop := 
  (2 * a) / (2 * c) = sqrt 2 / 1

-- Given constants
def given_consts : Prop :=
  ∃ a b c : ℝ, 
    a > b ∧ b > 0 ∧ foci a b c ∧ focal_ratio a c ∧ c = 3 ∧ 
    ellipse_C a b 3 0

-- Part (1) Area of triangle F1AB when slope k = 1
theorem area_of_triangle (A B F₁ F₂ : ℝ × ℝ) (a b : ℝ) 
  (h : given_consts)
  (slope_eq_one : ∃ l : ℝ → ℝ, (∀ x, l x = x - 3) ∧ ∀ p ∈ ({A, B}), (∃ q, ellipse_C a b (p.fst) (p.snd))) : 
  let area := 12 in
  F₁ ≠ F₂ ∧ F₂ = (3, 0) ∧ 
  (∀ A B : ℝ × ℝ, 
    (A.snd = -3 ∧ B.snd = 1) → abs (A.snd - B.snd) = 4) → 
  let dist := 6 in
  let calculated_area := 0.5 * dist * 4 in
  calculated_area = area
:= sorry

-- Part (2) Equation of line l when perpendicular bisector has smallest intercept on y-axis
theorem line_equation (a b k l : ℝ) (h : given_consts)
  (slope_condition : ∃ k : ℝ, k < 0 ∧ ∀ x : ℝ, l x = k * (x - 3)) 
  (intercept_condition : ∃ m : ℝ, ∀ y, y = (3 * k) / (2 * k^2 + 1) ∧ m = -(3 * (sqrt 2)) / 4 ∧ k = -(sqrt 2) / 2) : 
  let line_eq := λ x y, x + sqrt 2 * y - 3 = 0 in
  line_eq
:= sorry

end area_of_triangle_line_equation_l307_307872


namespace fifth_smallest_prime_factor_of_f20_plus_101_l307_307922

def p(n : ℕ) : ℕ := ∏ x in (Finset.filter Nat.Prime (Finset.range (n + 1))), x

def smallest_prime_factor(n : ℕ) : ℕ := if hn : n > 1 then
  Classical.find (Nat.exists_prime_and_dvd hn)
else n

def f(n : ℕ) : ℕ := p(n) ^ smallest_prime_factor(p(n))

theorem fifth_smallest_prime_factor_of_f20_plus_101 : 
  Nat.find (λ x, Prime x ∧ ∃ (a : ℕ), (f 20 + 101) = x * a) 4 = 11 := 
by
  sorry

end fifth_smallest_prime_factor_of_f20_plus_101_l307_307922


namespace factorization_correct_l307_307327

theorem factorization_correct {m : ℝ} : 
  (m^2 - 4) = (m + 2) * (m - 2) := 
by
  sorry

end factorization_correct_l307_307327


namespace gear_wheels_rotation_l307_307930

theorem gear_wheels_rotation (n : ℕ) : (∃ arrangement : list ℕ, arrangement.length = n ∧ ∀ (i : ℕ), i < n → (arrangement.nth i % 2 = 0 → arrangement.nth ((i + 1) % n) % 2 = 1) ∧ (arrangement.nth i % 2 = 1 → arrangement.nth ((i + 1) % n) % 2 = 0)) ↔ even n :=
sorry

end gear_wheels_rotation_l307_307930


namespace area_of_blackboard_l307_307664

-- Define the problem's conditions
def width : ℝ := 5.4  -- The width of the rectangle in meters
def height : ℝ := 2.5 -- The height of the rectangle in meters

-- Define the problem's statement: prove that the area is 13.5 square meters
theorem area_of_blackboard : width * height = 13.5 :=
by
  sorry

end area_of_blackboard_l307_307664


namespace letters_identity_l307_307240

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l307_307240


namespace arccos_neg_half_l307_307817

-- Defining the problem in Lean 4
theorem arccos_neg_half : 
  ∃ θ ∈ set.Icc 0 Real.pi, Real.arccos (-1 / 2) = θ ∧ Real.cos θ = -1 / 2 := 
by
  use Real.pi * 2 / 3
  split
  { sorry } -- Proof that θ is in [0, π]
  { split
    { sorry } -- Proof that θ = arccos(-1 / 2)
    { sorry } -- Proof that cos(θ) = -1/2
  }


end arccos_neg_half_l307_307817


namespace common_difference_and_first_three_terms_l307_307800

-- Given condition that for any n, the sum of the first n terms of an arithmetic progression is equal to 5n^2.
def arithmetic_sum_property (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 5 * n ^ 2

-- Define the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n-1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (a1 d n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d)/2

-- Conditions and prove that common difference d is 10 and the first three terms are 5, 15, and 25
theorem common_difference_and_first_three_terms :
  (∃ (a1 d : ℕ), arithmetic_sum_property (sum_first_n_terms a1 d) ∧ d = 10 ∧ nth_term a1 d 1 = 5 ∧ nth_term a1 d 2 = 15 ∧ nth_term a1 d 3  = 25) :=
sorry

end common_difference_and_first_three_terms_l307_307800


namespace yolanda_walking_rate_l307_307732

/-- Yolanda and Bob's walking rates and their meeting conditions -/
theorem yolanda_walking_rate
    (distance_XY : ℕ)
    (bob_rate : ℕ)
    (bob_walked : ℕ)
    (meet_time_from_Bob : bob_walking_time = bob_walked / bob_rate)
    (yolanda_start_time : ℕ)
    (yolanda_walked_time : yolanda_total_time = meet_time_from_Bob + yolanda_start_time)
    (meeting_point_distance_from_Y : ℕ)
    (yolanda_distance_covered : ℕ)
    (yolanda_rate : yolanda_rate = yolanda_distance_covered / yolanda_walked_time) :
    yolanda_rate = 3 := 
by
  sorry

-- with the asserted specifics from the problem:
#check @yolanda_walking_rate 24 4 12 3 1 4 12 12 3

end yolanda_walking_rate_l307_307732


namespace expansion_term_count_l307_307023

theorem expansion_term_count (N : ℕ) (a b c d e : ℕ → ℤ) :
  (N = 10) →
  ∃ T : finset (set (list ℕ)), T.card = 252 ∧ 
  ∀ term ∈ T, (∀ i, 1 ≤ term[i] ∧ term.sum = N) := 
sorry

end expansion_term_count_l307_307023


namespace number_of_four_digit_numbers_with_two_identical_digits_l307_307278

/-- Four-digit numbers starting with 2 and having exactly two identical digits. -/
def four_digit_numbers_with_two_identical_digits : set ℕ :=
  {n | 2000 ≤ n ∧ n < 3000 ∧
    (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
     digits.count (λ d, d = 2) = 1 ∧ 
     (digits.remove 2).count (λ d, d = (digits.remove 2).head!) = 1)}

/-- The number of four-digit numbers starting with 2 that have exactly two identical digits is 384. -/
theorem number_of_four_digit_numbers_with_two_identical_digits :
  (four_digit_numbers_with_two_identical_digits.to_finset.card = 384) :=
sorry

end number_of_four_digit_numbers_with_two_identical_digits_l307_307278


namespace problem_statement_l307_307970

open Set

-- Definitions based on the problem's conditions
def U : Set ℕ := { x | 0 < x ∧ x ≤ 8 }
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}
def complement_U_T : Set ℕ := U \ T

-- The Lean 4 statement to prove
theorem problem_statement : S ∩ complement_U_T = {1, 2, 4} :=
by sorry

end problem_statement_l307_307970


namespace positive_perfect_squares_l307_307523

theorem positive_perfect_squares (M : ℕ) (hM : M = 10^8) :
  ∃ n, n = 416 ∧ ∀ (k : ℕ), k * k < M → (576 ∣ k * k) ↔ (24 ∣ k) ∧ k < 10000 := 
begin
  use 416,
  split,
  { refl },
  { intros k h_k,
    split,
    { intro h_576_div_k2,
      have h_mod_0 : 576 ∣ k * k := h_576_div_k2,
      obtain ⟨m, h_eq⟩ := h_mod_0,
      have : 24 ∣ k,
      { rw [← h_eq] at h_mod_0,
        exact nat.dvd_of_mul_right_dvd h_mod_0, },
      split,
      { exact this, },
      { exact (nat.lt_pow_self_of_lt_pow_self_of_pos k 2 6 dec_trivial hM).1, } },
    { rintro ⟨h_dvd, h_lt⟩,
      have : 24 * 24 ∣ k * k,
      { exact nat.mul_dvd_mul h_dvd h_dvd, },
      use k / 24,
      rw [← nat.mul_div_assoc _ h_dvd, nat.mul_comm],
      exact nat.div_mul_cancel h_dvd, } }
end

end positive_perfect_squares_l307_307523


namespace garden_area_increase_l307_307788

-- Definitions derived directly from the conditions
def length := 50
def width := 10
def perimeter := 2 * (length + width)
def side_length_square := perimeter / 4
def area_rectangle := length * width
def area_square := side_length_square * side_length_square

-- The proof statement
theorem garden_area_increase :
  area_square - area_rectangle = 400 := 
by
  sorry

end garden_area_increase_l307_307788


namespace towers_per_castle_jeff_is_5_l307_307668

-- Define the number of sandcastles on Mark's beach
def num_castles_mark : ℕ := 20

-- Define the number of towers per sandcastle on Mark's beach
def towers_per_castle_mark : ℕ := 10

-- Calculate the total number of towers on Mark's beach
def total_towers_mark : ℕ := num_castles_mark * towers_per_castle_mark

-- Define the number of sandcastles on Jeff's beach (3 times that of Mark's)
def num_castles_jeff : ℕ := 3 * num_castles_mark

-- Define the total number of sandcastles on both beaches
def total_sandcastles : ℕ := num_castles_mark + num_castles_jeff
  
-- Define the combined total number of sandcastles and towers on both beaches
def combined_total : ℕ := 580

-- Define the number of towers per sandcastle on Jeff's beach
def towers_per_castle_jeff : ℕ := sorry

-- Define the total number of towers on Jeff's beach
def total_towers_jeff (T : ℕ) : ℕ := num_castles_jeff * T

-- Prove that the number of towers per sandcastle on Jeff's beach is 5
theorem towers_per_castle_jeff_is_5 : 
    200 + total_sandcastles + total_towers_jeff towers_per_castle_jeff = combined_total → 
    towers_per_castle_jeff = 5
:= by
    sorry

end towers_per_castle_jeff_is_5_l307_307668


namespace probability_of_neither_tamil_nor_english_l307_307934

-- Definitions based on the conditions
def TotalPopulation := 1500
def SpeakTamil := 800
def SpeakEnglish := 650
def SpeakTamilAndEnglish := 250

-- Use Inclusion-Exclusion Principle
def SpeakTamilOrEnglish : ℕ := SpeakTamil + SpeakEnglish - SpeakTamilAndEnglish

-- Number of people who speak neither Tamil nor English
def SpeakNeitherTamilNorEnglish : ℕ := TotalPopulation - SpeakTamilOrEnglish

-- The probability calculation
def Probability := (SpeakNeitherTamilNorEnglish : ℚ) / (TotalPopulation : ℚ)

-- Theorem to prove
theorem probability_of_neither_tamil_nor_english : Probability = (1/5 : ℚ) :=
sorry

end probability_of_neither_tamil_nor_english_l307_307934


namespace ratio_implies_sum_ratio_l307_307530

theorem ratio_implies_sum_ratio (x y : ℝ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_implies_sum_ratio_l307_307530


namespace jean_jail_time_l307_307170

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l307_307170


namespace compute_expression_l307_307389

theorem compute_expression (x : ℝ) (h : x = 7) : (x^6 - 36*x^3 + 324) / (x^3 - 18) = 325 := 
by
  sorry

end compute_expression_l307_307389


namespace sum_possible_values_n_l307_307827

theorem sum_possible_values_n : ∃ (n_values : List ℕ), 
  (∀ n ∈ n_values, ∃ m : ℕ, 0 < m ∧ 0 < n ∧ 59 * m - 68 * n = m * n) ∧ 
  n_values.Sum = 237 :=
by
  sorry

end sum_possible_values_n_l307_307827


namespace complement_of_A_in_U_l307_307113

def U := {-1, 0, 1, 2}
def A := {-1, 2}

theorem complement_of_A_in_U : ∁U A = {0, 1} := by
  sorry

end complement_of_A_in_U_l307_307113


namespace total_candies_in_third_set_l307_307708

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l307_307708


namespace cos_plus_sin_eq_sqrt_five_over_two_l307_307460

theorem cos_plus_sin_eq_sqrt_five_over_two (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
(h : sin α * cos α = 1 / 8) : 
cos α + sin α = real.sqrt (5) / 2 :=
by
  sorry

end cos_plus_sin_eq_sqrt_five_over_two_l307_307460


namespace number_of_questionnaires_from_D_l307_307559

variables (A B C D : ℕ)

-- Hypotheses and conditions
hypothesis (h_arithmetic_seq : ∃ d : ℕ, B = A + d ∧ C = A + 2*d ∧ D = A + 3*d)
hypothesis (h_sample_total : A + B + C + D = 100)
hypothesis (h_B_sample : B = 20)

-- Goal
theorem number_of_questionnaires_from_D : D = 40 :=
sorry

end number_of_questionnaires_from_D_l307_307559


namespace alligators_not_hiding_l307_307804

-- Definitions derived from conditions
def total_alligators : ℕ := 75
def hiding_alligators : ℕ := 19

-- Theorem statement matching the mathematically equivalent proof problem.
theorem alligators_not_hiding : (total_alligators - hiding_alligators) = 56 := by
  -- Sorry skips the proof. Replace with actual proof if required.
  sorry

end alligators_not_hiding_l307_307804


namespace remaining_interval_length_after_batch2_l307_307767

-- Define the initial conditions
def initial_interval : ℝ × ℝ := (10, 28)
def batch1_divisions : ℝ := 9
def batch1_length_after : ℝ := 4
def remaining_interval_after_batch1 (n : ℝ) : set ℝ := set.Icc (n - 2) (n + 2)
def batch2_divisions : ℝ := 10

theorem remaining_interval_length_after_batch2
    (n : ℝ)
    (length_after_batch1 : (remaining_interval_after_batch1 n).diam = batch1_length_after) :
    (remaining_interval_after_batch1 n).diam / batch2_divisions = 0.8 :=
by
  -- Given initial conditions and correct computations
  sorry

end remaining_interval_length_after_batch2_l307_307767


namespace expected_value_of_unfair_coin_l307_307802

theorem expected_value_of_unfair_coin:
  let P_heads := (2 : ℚ) / 3
  let P_tails := (1 : ℚ) / 3
  let V_heads := 5
  let V_tails := -10
  E = P_heads * V_heads + P_tails * V_tails
  in E = 0 :=
by
  let P_heads := (2 : ℚ) / 3
  let P_tails := (1 : ℚ) / 3
  let V_heads := 5
  let V_tails := -10
  let E := P_heads * V_heads + P_tails * V_tails
  show E = 0
  sorry

end expected_value_of_unfair_coin_l307_307802


namespace part1_part2_l307_307295

def traditional_chinese_paintings : ℕ := 6
def oil_paintings : ℕ := 4
def watercolor_paintings : ℕ := 5

theorem part1 :
  traditional_chinese_paintings * oil_paintings * watercolor_paintings = 120 :=
by
  sorry

theorem part2 :
  (traditional_chinese_paintings * oil_paintings) + 
  (traditional_chinese_paintings * watercolor_paintings) + 
  (oil_paintings * watercolor_paintings) = 74 :=
by
  sorry

end part1_part2_l307_307295


namespace find_beta_l307_307941

theorem find_beta (x y α t β : ℝ) 
  (hC : x = 2 + 2 * cos α ∧ y = 2 + 2 * sin α) 
  (hl : x = t * cos β ∧ y = t * sin β) 
  (hβ : 0 ≤ β ∧ β < real.pi)
  (h : (t * t + t * t) = 16) :
  β = real.pi / 12 ∨ β = 5 * real.pi / 12 := 
sorry

end find_beta_l307_307941


namespace extremum_at_1_l307_307898

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_1 (a b : ℝ) :
  (∃ c : ℝ, c = 10 ∧
  (∃ x, x = 1 ∧ f' x a b = 0 ∧ f x a b = c)) → a + b = -7 :=
begin
  sorry
end

end extremum_at_1_l307_307898


namespace relative_sizes_l307_307062

-- Define the function f
def f (x : ℝ) : ℝ := x * sin x + cos x

-- Specific points to be evaluated
def f1 := f 1
def f_pi_2 := f (π / 2)
def f_3pi_2 := f (3 * π / 2)

-- Assertion of relative sizes
theorem relative_sizes :
  f (3 * π / 2) < f (π / 2) ∧ f 1 < f (3 * π / 2) :=
by
  -- Explicit values based on trigonometric properties:
  have h1 : f1 = 1 * sin 1 + cos 1 := by rfl
  have h2 : f_pi_2 = π / 2 := by
    unfold f
    rw [sin_pi_div_two, cos_pi_div_two, zero_add, one_mul]
  have h3 : f_3pi_2 = -3 * π / 2 := by
    unfold f
    rw [sin_three_pi_div_two, cos_three_pi_div_two, zero_add, one_mul, neg_mul]
  -- Inequality holds
  sorry

end relative_sizes_l307_307062


namespace polygon_diagonals_l307_307728

theorem polygon_diagonals (n : ℕ) : 
  n(n - 3) / 2 = 2 ∨ n(n - 3) / 2 = 54 ↔ n = 4 ∨ n = 12 :=
sorry

end polygon_diagonals_l307_307728


namespace intersection_A_B_l307_307457

open Set

def A : Set ℝ := {x | Real.log x > 1 ∧ x ∈ Set.Ioi 0 ∩ {n | ∃ m : ℕ, n = m + 1}}
def B : Set ℝ := {x | x^2 - 16 < 0}

theorem intersection_A_B :
  B ∩ A = {3} :=
by
  sorry

end intersection_A_B_l307_307457


namespace sum_possible_values_b_l307_307847

theorem sum_possible_values_b : 
  (∑ b in {b : ℕ | ∃ (x : ℚ), 3 * x^3 + 7 * x^2 + 6 * x + b = 0}, b) = 10 :=
by
  sorry

end sum_possible_values_b_l307_307847


namespace range_of_a_l307_307896

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp (2 * x) / x) - 2 * x + Real.log x

theorem range_of_a (a : ℝ) :
  (∀ s ∈ Ioi 0, ∃ t ∈ Ioi 0, f a t < f a s) ↔ a ∈ Iic 0 := by
  sorry

end range_of_a_l307_307896


namespace gcd_poly_l307_307880

-- Defining the conditions
def is_odd_multiple_of_17 (b : ℤ) : Prop := ∃ k : ℤ, b = 17 * (2 * k + 1)

theorem gcd_poly (b : ℤ) (h : is_odd_multiple_of_17 b) : 
  Int.gcd (12 * b^3 + 7 * b^2 + 49 * b + 106) 
          (3 * b + 7) = 1 :=
by sorry

end gcd_poly_l307_307880


namespace problem_statement_l307_307435

variable (m n : ℝ)
noncomputable def sqrt_2_minus_1_inv := (Real.sqrt 2 - 1)⁻¹
noncomputable def sqrt_2_plus_1_inv := (Real.sqrt 2 + 1)⁻¹

theorem problem_statement 
  (hm : m = sqrt_2_minus_1_inv) 
  (hn : n = sqrt_2_plus_1_inv) : 
  m + n = 2 * Real.sqrt 2 := 
sorry

end problem_statement_l307_307435


namespace units_digit_of_n_l307_307400

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 11^4) (h2 : m % 10 = 9) : n % 10 = 9 := 
sorry

end units_digit_of_n_l307_307400


namespace third_set_candies_l307_307690

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l307_307690


namespace triangles_congruent_by_medians_and_angles_l307_307631

theorem triangles_congruent_by_medians_and_angles
  {A B C A1 B1 C1 M M1 : Type}
  (BM : Segment B M)
  (B1M1 : Segment B1 M1)
  (median_B : IsMedian B (A, C))
  (median_B1 : IsMedian B1 (A1, C1))
  (BM_eq_B1M1 : BM.length = B1M1.length)
  (angle_ABM_eq_angle_A1B1M1 : Angle (A, B, M) = Angle (A1, B1, M1))
  (angle_CBM_eq_angle_C1B1M1 : Angle (C, B, M) = Angle (C1, B1, M1)) :
  Triangle ABC ≃ Triangle A1B1C1 := 
sorry

end triangles_congruent_by_medians_and_angles_l307_307631


namespace letters_identity_l307_307242

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l307_307242


namespace correct_answers_max_l307_307555

def max_correct_answers (c w b : ℕ) : Prop :=
  c + w + b = 25 ∧ 4 * c - 3 * w = 40

theorem correct_answers_max : ∃ c w b : ℕ, max_correct_answers c w b ∧ ∀ c', max_correct_answers c' w b → c' ≤ 13 :=
by
  sorry

end correct_answers_max_l307_307555


namespace volume_of_rectangular_prism_l307_307267

theorem volume_of_rectangular_prism (x y z : ℝ) 
  (h1 : x * y = 30) 
  (h2 : x * z = 45) 
  (h3 : y * z = 75) : 
  x * y * z = 150 :=
sorry

end volume_of_rectangular_prism_l307_307267


namespace trigonometric_identity_l307_307600

noncomputable def c : ℝ := (2 * Real.pi) / 11

theorem trigonometric_identity :
  ( sin (3 * c) * sin (6 * c) * sin (9 * c) * sin (12 * c) * sin (15 * c) ) / 
  ( sin c * sin (2 * c) * sin (3 * c) * sin (4 * c) * sin (5 * c) ) = 1 :=
by
  sorry

end trigonometric_identity_l307_307600


namespace find_f_5_l307_307482

def f : ℕ → ℕ
| x := if x ≥ 6 then x - 3 else f (f (x + 5))

theorem find_f_5 : f 5 = 4 := sorry

end find_f_5_l307_307482


namespace third_set_candies_l307_307689

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l307_307689


namespace num_mountain_numbers_l307_307715

def is_mountain_number (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10] in
  n / 100 > 0 ∧
  n / 100 < 10 ∧
  (n / 10) % 10 > 0 ∧
  (digits.nth! 1 > digits.nth! 0) ∧ (digits.nth! 1 > digits.nth! 2)

def count_mountain_numbers : ℕ :=
  Finset.range 1000 |>.filter is_mountain_number |>.card

theorem num_mountain_numbers : count_mountain_numbers = 240 :=
  sorry

end num_mountain_numbers_l307_307715


namespace candy_count_in_third_set_l307_307699

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l307_307699


namespace selling_price_including_tax_l307_307994

theorem selling_price_including_tax
  (units : ℕ) (initial_investment profit_percentage packaging_cost shipping_fee sales_tax_percentage: ℝ) 
  (units_purchase : units = 200) 
  (initial_investment_purchase : initial_investment = 3000) 
  (profit_percentage_def : profit_percentage = 1 / 3) 
  (packaging_cost_def : packaging_cost = 2) 
  (shipping_fee_def : shipping_fee = 500) 
  (sales_tax_percentage_def : sales_tax_percentage = 0.08) :
  let cost_per_unit := shipping_fee / units + packaging_cost,
      total_cost_before_profit_and_tax := initial_investment + cost_per_unit * units,
      desired_profit := profit_percentage * initial_investment,
      total_amount_needed_from_sales := total_cost_before_profit_and_tax + desired_profit,
      selling_price_per_unit_before_tax := total_amount_needed_from_sales / units,
      sales_tax_per_unit := selling_price_per_unit_before_tax * sales_tax_percentage,
      selling_price_per_unit_including_tax := selling_price_per_unit_before_tax + sales_tax_per_unit
  in selling_price_per_unit_including_tax = 26.46 :=
by
  -- The proof goes here
  sorry

end selling_price_including_tax_l307_307994


namespace identity_of_letters_l307_307252

def first_letter : Type := Prop
def second_letter : Type := Prop
def third_letter : Type := Prop

axiom first_statement : first_letter → (first_letter = false)
axiom second_statement : second_letter → ∃! (x : second_letter), true
axiom third_statement : third_letter → (∃! (x : third_letter), x = true)

theorem identity_of_letters (A B : Prop) (is_A_is_true : ∀ x, x = A → x) (is_B_is_false : ∀ x, x = B → ¬x) :
  (first_letter = B) ∧ (second_letter = A) ∧ (third_letter = B) :=
sorry

end identity_of_letters_l307_307252


namespace sin_B_value_triangle_area_l307_307944

-- Problem 1: sine value of angle B given the conditions
theorem sin_B_value (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C) :
  Real.sin B = (4 * Real.sqrt 5) / 9 :=
sorry

-- Problem 2: Area of triangle ABC given the conditions and b = 4
theorem triangle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C)
  (h3 : b = 4) :
  (1 / 2) * b * c * Real.sin A = (14 * Real.sqrt 5) / 9 :=
sorry

end sin_B_value_triangle_area_l307_307944


namespace quadratic_expression_binomial_square_l307_307026

theorem quadratic_expression_binomial_square (a : ℚ) :
  (∃ b : ℚ, 4x^2 + 18x + a = (2x + b)^2) ↔ a = 81 / 4 :=
by
  sorry

end quadratic_expression_binomial_square_l307_307026


namespace kayla_apples_l307_307587

variable (x y : ℕ)
variable (h1 : x + (10 + 4 * x) = 340)
variable (h2 : y = 10 + 4 * x)

theorem kayla_apples : y = 274 :=
by
  sorry

end kayla_apples_l307_307587


namespace oranges_in_bowl_l307_307152

-- Definitions (conditions)
def bananas : Nat := 2
def apples : Nat := 2 * bananas
def total_fruits : Nat := 12

-- Theorem (proof goal)
theorem oranges_in_bowl : 
  apples + bananas + oranges = total_fruits → oranges = 6 :=
by
  intro h
  sorry

end oranges_in_bowl_l307_307152


namespace meaningful_expression_range_l307_307543

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by 
  sorry

end meaningful_expression_range_l307_307543


namespace MissyTotalTVTime_l307_307976

theorem MissyTotalTVTime :
  let reality_shows := [28, 35, 42, 39, 29]
  let cartoons := [10, 10]
  let ad_breaks := [8, 6, 12]
  let total_time := reality_shows.sum + cartoons.sum + ad_breaks.sum
  total_time = 219 := by
{
  -- Lean proof logic goes here (proof not requested)
  sorry
}

end MissyTotalTVTime_l307_307976


namespace solution_set_of_inequality_l307_307969

variables {R : Type*} [LinearOrderedField R]

-- Define f as an even function
def even_function (f : R → R) := ∀ x : R, f x = f (-x)

-- Define f as an increasing function on [0, +∞)
def increasing_on_nonneg (f : R → R) := ∀ ⦃x y : R⦄, 0 ≤ x → x ≤ y → f x ≤ f y

-- Define the hypothesis and the theorem
theorem solution_set_of_inequality (f : R → R)
  (h_even : even_function f)
  (h_inc : increasing_on_nonneg f) :
  { x : R | f x > f 1 } = { x : R | x > 1 ∨ x < -1 } :=
by {
  sorry
}

end solution_set_of_inequality_l307_307969


namespace correct_range_omega_l307_307140

-- Define the given function
def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x)

-- Define the problem conditions and the target range for ω
def two_extreme_points (ω : ℝ) := (ω > 0) ∧ 
  (∃−π/3 < x₁ < π/4, ∃−π/3 < x₂ < π/4, x₁ ≠ x₂ ∧ 
    (∂/∂x in x₁ and ∂/∂x in x₂ of f ω x are both 0))

-- The theorem to prove
theorem correct_range_omega (ω : ℝ) : 
  two_extreme_points(ω) → (3 < ω ∧ ω ≤ 4) := 
sorry

end correct_range_omega_l307_307140


namespace circumference_of_flower_bed_l307_307933

noncomputable def square_garden_circumference (a p s r C : ℝ) : Prop :=
  a = s^2 ∧
  p = 4 * s ∧
  a = 2 * p + 14.25 ∧
  r = s / 4 ∧
  C = 2 * Real.pi * r

theorem circumference_of_flower_bed (a p s r : ℝ) (h : square_garden_circumference a p s r (4.75 * Real.pi)) : 
  ∃ C, square_garden_circumference a p s r C ∧ C = 4.75 * Real.pi :=
sorry

end circumference_of_flower_bed_l307_307933


namespace proof_problem_l307_307439

def circleO_eq : Prop :=
  let M : ℝ × ℝ → ℝ := λ p, (p.1 + 2)^2 + (p.2 + 2)^2
  let sym_line : ℝ × ℝ → Prop := λ p, p.1 + p.2 + 2 = 0
  let A : ℝ × ℝ := (1, 1)
  let O : ℝ × ℝ → ℝ := λ p, p.1^2 + p.2^2
  ∀ a b r, O (a, b) = r^2 ∧ O A = 2

def max_area_quad : Prop :=
  let EF : ℝ × ℝ := (1, real.sqrt 2 / 2)
  let GH : ℝ × ℝ := (1, real.sqrt 2 / 2)
  ∀ d1 d2, d1^2 + d2^2 = 3/2 → 2 * real.sqrt(2 - d1^2) * 2 * real.sqrt(2 - d2^2) ≤ 5/2

def line_CD_fixed_point : Prop :=
  let l : ℝ × ℝ → Prop := λ p, p.2 = 1/2 * p.1 - 2
  let P : ℝ → ℝ × ℝ := λ t, (t, 1/2 * t - 2)
  let O : ℝ × ℝ := (0, 0)
  let CD passes_through : ℝ × ℝ := (1/2, -1)
  ∀ t, O × P t ∧ (CD_contains (1/2, -1))

theorem proof_problem :
  circleO_eq →
  max_area_quad →
  line_CD_fixed_point :=
by
  intros circleO maxArea lineCD
  exact sorry

end proof_problem_l307_307439


namespace problem_solution_l307_307432

theorem problem_solution (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 14) (h2 : a = b + c) : ab - bc + ac = 7 :=
  sorry

end problem_solution_l307_307432


namespace measure_angle_B_max_perimeter_l307_307502

variables {A B C : Angle} {a b c : Real}

/- Problem (I) -/
theorem measure_angle_B (h : 2 * a * Real.cos B + b * Real.cos C + c * Real.cos B = 0) : 
  B = (2 * Real.pi) / 3 :=
sorry

/- Problem (II) -/
theorem max_perimeter (b : Real) (hb : b = Real.sqrt 3) 
  (a c : Real) (ha : a > 0) (hc : c > 0)
  : a + b + c ≤ Real.sqrt 3 + 2 :=
sorry

end measure_angle_B_max_perimeter_l307_307502


namespace james_new_fuel_cost_l307_307167

def original_cost : ℕ := 200
def price_increase_rate : ℕ := 20
def extra_tank_factor : ℕ := 2

theorem james_new_fuel_cost :
  let new_price := original_cost + (price_increase_rate * original_cost / 100)
  let total_cost := extra_tank_factor * new_price
  total_cost = 480 :=
by
  sorry

end james_new_fuel_cost_l307_307167


namespace exists_player_P_l307_307153

-- Definitions to represent the conditions
variable (Player : Type) [Fintype Player]

-- Relations for winning and losing
variable (loses_to : Player → Player → Prop)
variable [decidable_rel loses_to]

-- The main theorem statement
theorem exists_player_P (h_tournament : ∀ (P Q : Player), loses_to P Q ∨ loses_to Q P)
  (h_no_draw : ∀ (P Q : Player), loses_to P Q → ¬ loses_to Q P) : 
  ∃ P : Player, ∀ Q : Player, Q ≠ P → 
    (loses_to Q P ∨ ∃ R : Player, loses_to Q R ∧ loses_to R P) :=
begin
  sorry
end

end exists_player_P_l307_307153


namespace root_implies_value_l307_307130

theorem root_implies_value (b c : ℝ) (h : 2 * b - c = 4) : 4 * b - 2 * c + 1 = 9 :=
by
  sorry

end root_implies_value_l307_307130


namespace problem1_problem2_problem3_l307_307005

-- Problem 1
theorem problem1 : 13 + (-7) - (-9) + 5 * (-2) = 5 :=
by 
  sorry

-- Problem 2
theorem problem2 : abs (-7 / 2) * (12 / 7) / (4 / 3) / (3 ^ 2) = 1 / 2 :=
by 
  sorry

-- Problem 3
theorem problem3 : -1^4 - (1 / 6) * (2 - (-3)^2) = 1 / 6 :=
by 
  sorry

end problem1_problem2_problem3_l307_307005


namespace cubes_penetrated_by_diagonal_l307_307753

theorem cubes_penetrated_by_diagonal (a b c : ℕ) (h₁ : a = 120) (h₂ : b = 260) (h₃ : c = 300) :
  let gcd_ab := Nat.gcd a b,
      gcd_bc := Nat.gcd b c,
      gcd_ca := Nat.gcd c a,
      gcd_abc := Nat.gcd (Nat.gcd a b) c
  in a + b + c - (gcd_ab + gcd_bc + gcd_ca) + gcd_abc = 520 :=
by
  sorry

end cubes_penetrated_by_diagonal_l307_307753


namespace f_unbounded_l307_307192

-- Define the function f
def f (n : ℕ) : ℕ :=
  -- This is where the definition of how to calculate f(n) goes
  sorry

-- State the problem as a Lean theorem
theorem f_unbounded : ¬ (∃ x : ℕ, ∀ n : ℕ, f(n) ≤ x) :=
by
  -- Proof goes here
  sorry

end f_unbounded_l307_307192


namespace range_of_x_l307_307492

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x + 1) + 2^x + 2^(-x)

theorem range_of_x (x : ℝ) :
  f (x + 1) < f (2 * x) ↔ x ∈ Iio (-1/3) ∪ Ioi 1 :=
sorry

end range_of_x_l307_307492


namespace total_rows_chairs_l307_307299

theorem total_rows_chairs (R : ℕ) (h1 : ∀ i, i < R → (i < 3 ∨ i ≥ R - 2) ∨ (15 - (4 / 5) * 15) (R - 5) = 15) : R = 10 :=
sorry

end total_rows_chairs_l307_307299


namespace monotonicity_of_f_range_of_a_l307_307960

noncomputable def f (x a : ℝ) := Real.exp x - a * x
noncomputable def g (x a : ℝ) := Real.exp x - (a + 2) * x

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (Real.log a) (Real.log a) → deriv (f x a) x ≥ 0) ∨ 
  (a ≤ 0 ∧ ∀ x : ℝ, deriv (f x a) x ≥ 0) :=
sorry

theorem range_of_a : {a : ℝ | ∀ x : ℝ, f x a ≥ 2 * x } = Set.Icc (-2) (Real.exp 1 - 2) :=
sorry

end monotonicity_of_f_range_of_a_l307_307960


namespace smallest_k_gt_10_periodic_fraction_represents_17_over_85_l307_307845

theorem smallest_k_gt_10_periodic_fraction_represents_17_over_85 : 
  ∃ (k : ℤ), (k > 10) ∧ (0.\overline{(41)}_k = fractional_representation (17 / 85)) ∧ is_smallest_k (k = 19) := 
sorry

end smallest_k_gt_10_periodic_fraction_represents_17_over_85_l307_307845


namespace average_divisible_by_3_and_4_l307_307001

-- Define the conditions
def in_range (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 50
def divisible_by_3_and_4 (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0

-- Define the set of numbers
def numbers : List ℕ := List.filter (λ n, in_range n ∧ divisible_by_3_and_4 n) (List.range 51)

-- Prove the main statement
theorem average_divisible_by_3_and_4 : (List.sum numbers : ℚ) / List.length numbers = 30 := by
  sorry

end average_divisible_by_3_and_4_l307_307001


namespace original_cylinder_weight_is_24_l307_307832

noncomputable def weight_of_original_cylinder (cylinder_weight cone_weight : ℝ) : Prop :=
  cylinder_weight = 3 * cone_weight

-- Given conditions in Lean 4
variables (cone_weight : ℝ) (h_cone_weight : cone_weight = 8)

-- Proof problem statement
theorem original_cylinder_weight_is_24 :
  weight_of_original_cylinder 24 cone_weight :=
by
  sorry

end original_cylinder_weight_is_24_l307_307832


namespace circumcenter_in_angle_bisector_l307_307222

noncomputable def circumcenter (A D F : Point) : Point := sorry

variables {A B C D E F O : Point}

-- Conditions
def condition1 : (D ∈ Segment A B) := sorry
def condition2 : (E ∈ Segment B C) := sorry
def condition3 : (F ∈ Segment A C) := sorry
def condition4 : (dist D E = dist B E) := sorry
def condition5 : (dist F E = dist C E) := sorry
def circumcenter_ADF := circumcenter A D F

-- The statement to be proven
theorem circumcenter_in_angle_bisector :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 →
  is_on_angle_bisector circumcenter_ADF (∠DEF) :=
sorry

end circumcenter_in_angle_bisector_l307_307222


namespace num_partitions_of_staircase_l307_307971

-- Definition of a staircase
def is_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ j → j ≤ i → i ≤ n → cells (i, j)

-- Number of partitions of a staircase of height n
def num_partitions (n : ℕ) : ℕ :=
  2^(n-1)

theorem num_partitions_of_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) :
  is_staircase n cells → (∃ p : ℕ, p = num_partitions n) :=
by
  intro h
  use (2^(n-1))
  sorry

end num_partitions_of_staircase_l307_307971


namespace candy_count_in_third_set_l307_307695

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l307_307695


namespace range_of_x_l307_307491

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x + 1) + 2^x + 2^(-x)

theorem range_of_x (x : ℝ) :
  f (x + 1) < f (2 * x) ↔ x ∈ Iio (-1/3) ∪ Ioi 1 :=
sorry

end range_of_x_l307_307491


namespace dot_product_of_BA_and_BC_l307_307552

noncomputable def dot_product_triangle (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (AB BC AC : ℝ) (hAB : AB = 3) (hBC : BC = 2) (hAC : AC = real.sqrt 7) :
  ℝ :=
  let cos_B := (AB^2 + BC^2 - AC^2) / (2 * AB * BC) in
  AB * BC * cos_B

theorem dot_product_of_BA_and_BC :
  ∀ (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C],
  dot_product_triangle A B C 3 2 (real.sqrt 7) rfl rfl rfl = 3 :=
  by
  intros
  exact sorry

end dot_product_of_BA_and_BC_l307_307552


namespace union_of_A_and_B_l307_307595

/-- Given sets A and B defined as follows: A = {x | -1 <= x <= 3} and B = {x | 0 < x < 4}.
Prove that their union A ∪ B is the interval [-1, 4). -/
theorem union_of_A_and_B :
  let A := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
  let B := {x : ℝ | 0 < x ∧ x < 4}
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 4} :=
by
  sorry

end union_of_A_and_B_l307_307595


namespace find_sum_mnp_l307_307357

-- Define the problem parameters and conditions
def side_length : ℝ := 12
def distance_BD : ℝ := 9
def distance_DC : ℝ := side_length - distance_BD

-- Define variables for the length of the fold
variables (m n p : ℕ)

-- State that m, n, p are positive integers, m and n are relatively prime,
-- and p is not divisible by the square of any prime
-- Also, m + n + p is the value to be proven
def is_prime (x : ℕ) : Prop := Nat.Prime x
def is_irreducible (x : ℤ) : Prop := ¬(∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ a * b = x)
def relatively_prime (a b : ℕ) : Prop := Nat.coprime a b
def square_free (p : ℕ) : Prop := ∀ (q : ℕ), is_prime q → q^2 ∣ p → false

theorem find_sum_mnp : ∃ m n p : ℕ, relatively_prime m n ∧ square_free p ∧ (m * m + n * n = 13) ∧ m + n + p = 17 :=
by
    sorry

end find_sum_mnp_l307_307357


namespace third_set_candies_l307_307687

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l307_307687


namespace min_express_pairs_l307_307566

universe u

/-- In the country of Alfya, there are 150 cities, some of which are connected by express trains that do not stop at intermediate stations. It is known that any four cities can be divided into two pairs such that there is an express running between the cities of each pair. What is the minimum number of pairs of cities connected by expresses? -/
theorem min_express_pairs (C : Type u) [Fintype C] (hC : Fintype.card C = 150)
    (express : C → C → Prop) (h3 : ∀ a b : C, express a b → express b a)
    (h4 : ∀ (A B C D : C), ∃ (P : {A B} → {C D}), ∀ (s t : {A B} → {C D}), ∃ (x y : C), x ≠ y ∧ express x y) :
  ∃ m, m = 11025 :=
by
  sorry

end min_express_pairs_l307_307566


namespace function_monotonic_increasing_iff_l307_307489

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x + Real.cos x

def is_monotonically_increasing (f : ℝ → ℝ) (I : Set ℝ) := ∀ x y ∈ I, x ≤ y → f x ≤ f y

theorem function_monotonic_increasing_iff (a : ℝ) :
  is_monotonically_increasing (f a) (Set.Icc 0 (Real.pi / 4)) ↔ a ≥ 1 :=
by
  sorry

end function_monotonic_increasing_iff_l307_307489


namespace krystiana_earnings_l307_307180

def earning_building1_first_floor : ℝ := 5 * 15 * 0.8
def earning_building1_second_floor : ℝ := 6 * 25 * 0.75
def earning_building1_third_floor : ℝ := 9 * 30 * 0.5
def earning_building1_fourth_floor : ℝ := 4 * 60 * 0.85
def earnings_building1 : ℝ := earning_building1_first_floor + earning_building1_second_floor + earning_building1_third_floor + earning_building1_fourth_floor

def earning_building2_first_floor : ℝ := 7 * 20 * 0.9
def earning_building2_second_floor : ℝ := (25 + 30 + 35 + 40 + 45 + 50 + 55 + 60) * 0.7
def earning_building2_third_floor : ℝ := 6 * 60 * 0.6
def earnings_building2 : ℝ := earning_building2_first_floor + earning_building2_second_floor + earning_building2_third_floor

def total_earnings : ℝ := earnings_building1 + earnings_building2

theorem krystiana_earnings : total_earnings = 1091.5 := by
  sorry

end krystiana_earnings_l307_307180


namespace minimum_quadrilateral_area_l307_307089

noncomputable def minimum_area_quadrilateral : ℝ :=
  let line_eq : ℝ → ℝ → Prop := λ x y, 3 * x + 4 * y + 8 = 0
  let circle_eq : ℝ → ℝ → Prop := λ x y, (x - 1)^2 + (y - 1)^2 = 1
  let center : (ℝ × ℝ) := (1, 1)
  let PA_PB_tangents : Π (P : ℝ × ℝ), Prop := λ P,
    let (Px, Py) := P in
    line_eq Px Py ∧ ∃ (A B : ℝ × ℝ), 
      circle_eq (fst A) (snd A) ∧ circle_eq (fst B) (snd B) ∧
      (Px - (fst A)) * (Py - (snd A)) = (Px - (fst B)) * (Py - (snd B))
  ∃ (P : ℝ × ℝ), PA_PB_tangents P ∧ 
    let A B : ℝ × ℝ := (P.1 - 1, P.2 - 1) in
    let area_PACB := 2 * 0.5 * sqrt((center.1 - P.1)^2 + (center.2 - P.2)^2) * 1 in
    area_PACB = 2 * sqrt 2

theorem minimum_quadrilateral_area : minimum_area_quadrilateral = 2 * sqrt 2 := 
sorry

end minimum_quadrilateral_area_l307_307089


namespace tangent_line_ln_l307_307053

theorem tangent_line_ln (b : ℝ) :
  (∀ x > 0, ∀ y, y = Real.log x → (deriv (λ x : ℝ, Real.log x) x = 1 / x)) →
  (∃ x > 0, ∃ y, y = Real.log x ∧ y = 1 / 2 * x + b) →
  b = Real.log 2 - 1 :=
by
  sorry

end tangent_line_ln_l307_307053


namespace total_candies_in_third_set_l307_307709

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l307_307709


namespace total_candies_third_set_l307_307705

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l307_307705


namespace f_f_neg1_eq_45_l307_307863

def f (x : ℝ) : ℝ := if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem f_f_neg1_eq_45 : f (f (-1)) = 45 := by
  sorry

end f_f_neg1_eq_45_l307_307863


namespace area_trapezoid_ABND_l307_307765

theorem area_trapezoid_ABND (A B C D N : Point) (r : ℝ) (circle : Circle) (AB AD : ℝ) :
  rectangle A B C D →
  circle.touches (line A B) →
  circle.touches (line A D) →
  circle.passes_through C →
  circle.intersects_at (line D C) N →
  AB = 9 →
  AD = 8 →
  area_trapezoid A B N D = 40 :=
by
  sorry

end area_trapezoid_ABND_l307_307765


namespace cos_75_degree_l307_307825

theorem cos_75_degree :
  ∃ x : ℝ, x = real.cos (75 * real.pi / 180) ∧ x = (real.sqrt 6 - real.sqrt 2) / 4 :=
sorry

end cos_75_degree_l307_307825


namespace median_number_of_children_is_three_l307_307978

def families : List (Nat × Nat) :=
[(1, 4), (2, 3), (3, 5), (4, 2), (5, 1)]

/-- The median number of children in the families of Ms. Thompson's math class students is 3. -/
theorem median_number_of_children_is_three : 
  List.median families = 3 :=
by
  -- Proof goes here
  sorry

end median_number_of_children_is_three_l307_307978


namespace sum_of_squares_of_solutions_l307_307848

theorem sum_of_squares_of_solutions (x : ℝ) (h : 2 * x - 1 / x = x) : ∑ (r : ℝ) in {x | 2 * x - 1 / x = x}.to_finset, r^2 = 2 := by
  sorry

end sum_of_squares_of_solutions_l307_307848


namespace proof_problem_l307_307443

noncomputable theory
open Classical

def quadratic_symmetric (a b : ℝ) := ∀ x, a * x^2 + b * x = a * (-x-2)^2 + b * (-x-2)

def tangent_graph (a b : ℝ) := ∃ x, a * x^2 + b * x = x

def analytical_expression_correct := ∃ a b : ℝ, a ≠ 0 ∧ quadratic_symmetric a b ∧ tangent_graph a b ∧ (a = 1/2 ∧ b = 1)

def inequality_solution := ∀ t x : ℝ, abs t ≤ 2 → (π^(1/2 * x^2 + x) > (1/π)^(2 - t*x)) ↔ ((x < -3 - real.sqrt 5) ∨ (x > -3 + real.sqrt 5))

theorem proof_problem : analytical_expression_correct ∧ inequality_solution :=
sorry

end proof_problem_l307_307443


namespace simple_numbers_less_than_million_l307_307619

/- Define what it means for an integer to be simple -/
def is_simple (n : ℕ) : Prop :=
  ∀ d : ℕ, (d ∈ n.digits 10) → (d = 1 ∨ d = 2)

/- The main theorem statement -/
theorem simple_numbers_less_than_million : 
  (finset.filter is_simple (finset.range 1000000)).card = 126 :=
sorry

end simple_numbers_less_than_million_l307_307619


namespace find_positive_real_number_solution_l307_307417

theorem find_positive_real_number_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) (hx : x > 0) : x = 15 :=
sorry

end find_positive_real_number_solution_l307_307417


namespace percent_increase_calculation_l307_307205

variable (x y : ℝ) -- Declare x and y as real numbers representing the original salary and increment

-- The statement that the percent increase z follows from the given conditions
theorem percent_increase_calculation (h : y + x = x + y) : (y / x) * 100 = ((y / x) * 100) := by
  sorry

end percent_increase_calculation_l307_307205


namespace a_values_l307_307903

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem a_values (a : ℝ) : A a ∩ B a = {x} → (a = 0 ∧ x = 1) ∨ (a = -2 ∧ x = 5) := sorry

end a_values_l307_307903


namespace smallest_n_for_grid_coloring_l307_307072

theorem smallest_n_for_grid_coloring :
  ∃ n : ℕ, (n >= 338 ∧ ∀ grid : array (fin 2023) (array (fin 2023) (fin n)), 
  (∀ i : fin 2023, 
  ∀ j₁ j₂ : fin i.val, 
  ∀ k₁ k₂ : fin j₂.val, 
  (grid[i][j₁] = grid[i][k₁] ∧ 
  grid[i][j₂] = grid[i][k₂] ∧ 
  j₁ + 1 < j₂ ∧ 
  k₁ + 1 < k₂ 
  → ∃ m₁ m₂ : fin i.val, grid[m₁][j₁] ≠ grid[m₂][k₂]))): n = 338 :=
begin
  sorry
end

end smallest_n_for_grid_coloring_l307_307072


namespace count_three_digit_numbers_divisible_by_seventeen_l307_307913

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_divisible_by_seventeen (n : ℕ) : Prop := n % 17 = 0

theorem count_three_digit_numbers_divisible_by_seventeen : 
  ∃ (count : ℕ), count = 53 ∧ 
    (∀ (n : ℕ), is_three_digit_number n → is_divisible_by_seventeen n → response) := 
sorry

end count_three_digit_numbers_divisible_by_seventeen_l307_307913


namespace closely_related_interval_unique_l307_307505

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

theorem closely_related_interval_unique :
  closely_related f g 2 3 :=
sorry

end closely_related_interval_unique_l307_307505


namespace letters_identity_l307_307241

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l307_307241


namespace percentage_of_students_at_camping_trip_l307_307331

theorem percentage_of_students_at_camping_trip (total_students : ℕ) (total_students > 0) 
    (h1 : 15% of total_students went to a camping trip and took more than $100)
    (h2 : 75% of the students who went to the camping trip did not take more than $100) :
    percentage_of_students_at_camping_trip = 15% :=
begin
    sorry
end

end percentage_of_students_at_camping_trip_l307_307331


namespace rate_of_increase_of_area_l307_307281

open Real

-- Define the conditions
def radius_velocity : ℝ := 50 -- cm/s
def radius_at_time : ℝ := 250 -- cm

-- Define the problem statement as a Lean 4 theorem
theorem rate_of_increase_of_area :
  let v := radius_velocity
  let r := radius_at_time
  let t := r / v in
  deriv (λ t : ℝ, π * (v * t)^2) t = 25000 * π :=
by
  -- The proof would go here
  sorry

end rate_of_increase_of_area_l307_307281


namespace appropriate_instrument_needed_l307_307572

variable {HCl37 : Type} [measure37 : Measurement HCl37 0.37]
variable {HCl15 : Type} [measure15 : Measurement HCl15 0.15]
variable {Container : Type} [beaker : Beaker Container]
variable (additionally_needed : Type) 

-- Definitions to be used from the problem description:
def prepared (acid : Type) [measure : Measurement acid] [container : Beaker Container] [stir : StirLect acid container] :=
  true

axiom prepare_15pct_HCl_from_37pct_HCl (M C : Type) [measureM : Measurement M 0.37] [measureC : Measurement C 0.15]
  [beaker : Beaker Container] [rod : StirLect M beaker] : prepared (C : Type)

-- The original problem statement translated into a Lean statement.
theorem appropriate_instrument_needed (M C : Type) [measureM : Measurement M 0.37] [measureC : Measurement C 0.15]
  [beaker : Beaker Container] [additionally_needed_stir : StirLect M beaker]: 
  M = HCl37 ∧ C = HCl15 ∧ additionally_needed = GlassRod :=
sorry

end appropriate_instrument_needed_l307_307572


namespace equation_represents_point_l307_307271

theorem equation_represents_point 
  (a b x y : ℝ) 
  (h : (x - a) ^ 2 + (y + b) ^ 2 = 0) : 
  x = a ∧ y = -b := 
by
  sorry

end equation_represents_point_l307_307271


namespace paths_with_consecutive_ups_l307_307122

theorem paths_with_consecutive_ups (w h : ℕ) (w_condition : w = 7) (h_condition : h = 6) : 
  let total_paths := Nat.choose (w + h) h,
      paths_with_1_consec_ups := Nat.choose (w + h - 1) h,
      paths_with_2_consec_ups := Nat.choose (w + h - 2) h
  in total_paths - paths_with_1_consec_ups + paths_with_2_consec_ups = 1254 :=
by
  have w := w_condition
  have h := h_condition
  sorry

end paths_with_consecutive_ups_l307_307122


namespace find_vector_b_norm_l307_307469

noncomputable def vector_norm {n : ℕ} (v : Fin n → ℝ) : ℝ :=
Real.sqrt (Finset.univ.sum (λ i, v i ^ 2))

noncomputable def vector_dot_product {n : ℕ} (v w : Fin n → ℝ) : ℝ :=
Finset.univ.sum (λ i, v i * w i)

noncomputable def angle_between_vectors (θ : ℝ) (a b : Fin 3 → ℝ) : Prop :=
vector_dot_product a b = vector_norm a * vector_norm b * Real.cos θ

theorem find_vector_b_norm
  (a b : Fin 3 → ℝ)
  (h1 : angle_between_vectors (Real.pi / 3) a b)
  (h2 : vector_norm a = 2)
  (h3 : vector_norm (λ i, a i - 2 * b i) = 2 * Real.sqrt 7) :
  vector_norm b = 3 :=
sorry

end find_vector_b_norm_l307_307469


namespace find_value_of_expression_l307_307877

variable (α β : ℝ)

-- Defining the conditions
def is_root (α : ℝ) : Prop := α^2 - 3 * α + 1 = 0
def add_roots_eq (α β : ℝ) : Prop := α + β = 3
def mult_roots_eq (α β : ℝ) : Prop := α * β = 1

-- The main statement we want to prove
theorem find_value_of_expression {α β : ℝ} 
  (hα : is_root α) 
  (hβ : is_root β)
  (h_add : add_roots_eq α β)
  (h_mul : mult_roots_eq α β) :
  3 * α^5 + 7 * β^4 = 817 := 
sorry

end find_value_of_expression_l307_307877


namespace intersection_locus_is_circle_l307_307067

-- Definition of a point in a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a trapezoid with base AD, and sides AB, BC with constant lengths
structure Trapezoid where
  A B C D : Point
  length_AB : ℝ
  length_BC : ℝ
  base_AD : ℝ
  conditions : A.dist B = length_AB ∧ B.dist C = length_BC ∧ A.dist D = base_AD

-- Function to get the intersection of diagonals of a trapezoid
def intersection_of_diagonals (T : Trapezoid) : Point := sorry

-- Statement to prove the locus of intersection points of diagonals is a circle
theorem intersection_locus_is_circle (T : Trapezoid) 
  (h : ∀ T1 T2, T1.conditions → T2.conditions → T1.base_AD = T2.base_AD →  
        (intersection_of_diagonals T1).dist (intersection_of_diagonals T2) = (T.base_AD / 2)) :
  ∃ center : Point, ∃ radius : ℝ, ∀ T, 
    T.conditions → (intersection_of_diagonals T).dist center = radius := 
sorry

end intersection_locus_is_circle_l307_307067


namespace students_facing_outward_after_12_turns_l307_307414

theorem students_facing_outward_after_12_turns :
  let turns (n : Nat) := (List.range (n+1)).tail 
    in let facing_out_after_12_turns (student : Nat) := 
      (List.foldl (λ facing round => if round ≥ student then !facing else facing) false (turns 12)) 
        in List.countP facing_out_after_12_turns (List.range 15).tail = 12 :=
by 
  let turns (n : Nat) := (List.range (n+1)).tail
  let facing_out_after_12_turns (student : Nat) :=
    (List.foldl (λ facing round => if round ≥ student then !facing else facing) false (turns 12))
  have : List.countP facing_out_after_12_turns (List.range 15).tail = 12 := sorry
  exact this

end students_facing_outward_after_12_turns_l307_307414


namespace equal_sides_if_collinear_l307_307952

-- Definitions of the geometric setup
variables {A B C D E F I : Type} [Triangle ABC] 
variables (r : Line) [is_angle_bisector r ABC] 
variables (s : Line) [is_angle_bisector s BCA]
variables (AD_line BE_line AE_line CD_line : Line)
variables [is_parallel AD_line BW] [is_parallel AE_line CD]

-- conditions
variables [contains_point r E] [contains_point s D]
variables [lines_intersect BD CE F]
variables [is_incenter I ABC]
variables [collinear A F I]

-- Goal statement
theorem equal_sides_if_collinear :
  AB = AC :=
sorry

end equal_sides_if_collinear_l307_307952


namespace not_power_of_two_l307_307641

-- Definitions corresponding to the conditions.
def is_arrangement_of_five_digit_numbers (A : ℕ) : Prop :=
  ∃ (f : Fin 88889 → ℕ), (∀ i, 11111 ≤ f i ∧ f i ≤ 99999) ∧
  (A = ∑ i in Finset.range 88889, f i * 10^(5 * i))

-- Statement of the theorem
theorem not_power_of_two (A : ℕ) (h : is_arrangement_of_five_digit_numbers A) :
  ∀ k : ℕ, A ≠ 2^k :=
sorry

end not_power_of_two_l307_307641


namespace part1_part2_l307_307196

noncomputable def f (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), (1 / (i + 1))

theorem part1 (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : n > m) :
  f(n) - f(m) ≥ (n - m) / n := 
sorry

theorem part2 (n : ℕ) (hn : n > 1) :
  f(2^n) > (n + 2) / 2 := 
sorry

end part1_part2_l307_307196


namespace area_inequality_l307_307444

variable (A B C D : Point)
variable (S : ℝ)
variable (AB BC CD DA : ℝ)

axiom AB_CD_congruent : dist A B = AB ∧ dist B C = BC ∧ dist C D = CD ∧ dist D A = DA
axiom area_ABC_correct: areaOfQuadrilateral A B C D = S

theorem area_inequality : S ≤ 1 / 2 * (AB * CD + DA * BC) := by
  sorry

end area_inequality_l307_307444


namespace cos_A_zero_l307_307529

theorem cos_A_zero (A : ℝ) (h : Real.tan A + (1 / Real.tan A) + 2 / (Real.cos A) = 4) : Real.cos A = 0 :=
sorry

end cos_A_zero_l307_307529


namespace arithmetic_sequence_a9_l307_307462

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + d * (n - 1)

-- The sum of the first n terms of an arithmetic sequence.
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a9
  (a1 d : ℤ)
  (h1 : a1 + (a1 + d)^2 = -3)
  (h2 : S 5 a1 d = 10) :
  a 9 a1 d = 20 :=
begin
  sorry
end

end arithmetic_sequence_a9_l307_307462


namespace perp_implies_parallel_l307_307875

variables (a b : Type) [line a] [line b] (alpha : Type) [plane alpha]

-- defining the perpendicular relation
def perp (l : Type) [line l] (p : Type) [plane p] : Prop := sorry

-- defining the parallel relation
def parallel (l1 l2 : Type) [line l1] [line l2] : Prop := sorry

-- stating the proof goal
theorem perp_implies_parallel (ha : perp a alpha) (hb : perp b alpha) : parallel a b := sorry

end perp_implies_parallel_l307_307875


namespace total_candies_third_set_l307_307707

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l307_307707


namespace total_candies_in_third_set_l307_307711

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l307_307711


namespace intersection_sum_zero_l307_307564

-- Definitions from conditions:
def lineA (x : ℝ) : ℝ := -x
def lineB (x : ℝ) : ℝ := 5 * x - 10

-- Declaration of the theorem:
theorem intersection_sum_zero : ∃ a b : ℝ, lineA a = b ∧ lineB a = b ∧ a + b = 0 := sorry

end intersection_sum_zero_l307_307564


namespace five_digit_integers_divisible_by_12_count_l307_307516

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

def two_digit_multiples_of_12 : List ℕ :=
  [12, 24, 36, 48, 60, 72, 84, 96]

def count_five_digit_integers_divisible_by_12 : ℕ :=
  9 * 10 * 10 * 10 * 8

theorem five_digit_integers_divisible_by_12_count :
  (count_five_digit_integers_divisible_by_12 = 72000) :=
by
  rw [count_five_digit_integers_divisible_by_12]
  norm_num
  -- We skip the detailed proof steps here
  sorry

end five_digit_integers_divisible_by_12_count_l307_307516


namespace base8_sum_to_base16_equiv_l307_307846

-- Define the numbers in base 8
def n1_base8 := "5214"
def n2_base8 := "1742"

-- Function to convert base 8 to base 10
def base8_to_base10 (s : String) : ℕ :=
  s.foldr (λ c acc, acc * 8 + (c.toNat - '0'.toNat)) 0

-- Base 16 representation
def n1_base10 := base8_to_base10 n1_base8
def n2_base10 := base8_to_base10 n2_base8

-- Function to convert base 10 to base 16
def base10_to_base16 (n : ℕ) : String :=
  if n = 0 then "0" else
  let rec aux (n : ℕ) (acc : String) :=
    if n = 0 then acc else
    let r := n % 16
    let n' := n / 16
    let r_str := if r < 10 then toString r else (Char.ofNat (r - 10 + 'A'.toNat)).toString
    aux n' (r_str ++ acc)
  aux n ""

-- Variables for the base 8 sum
def sum_base8 := base8_to_base10 n1_base8 + base8_to_base10 n2_base8

-- Expected result in base 16
def expected_base16 := "E6E"

-- The statement we want to prove
theorem base8_sum_to_base16_equiv :
  base10_to_base16 sum_base8 = expected_base16 :=
sorry

end base8_sum_to_base16_equiv_l307_307846


namespace power_function_monotonic_solution_l307_307887

theorem power_function_monotonic_solution 
  (a : ℝ) (h1 : ∀ x : ℝ, monotonic_on (fun (y : ℝ) ↦ (a^2 - 2*a - 2) * y^a) (set.univ : set ℝ)) : 
  a = 3 := 
sorry

end power_function_monotonic_solution_l307_307887


namespace identify_letters_l307_307256

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l307_307256


namespace remainder_problem_l307_307723

theorem remainder_problem : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end remainder_problem_l307_307723


namespace no_real_solutions_for_equation_l307_307014

theorem no_real_solutions_for_equation (x : ℝ) :
  y = 3 * x ∧ y = (x^3 - 8) / (x - 2) → false :=
by {
  sorry
}

end no_real_solutions_for_equation_l307_307014


namespace restaurant_bill_split_l307_307210

def original_bill : ℝ := 514.16
def tip_rate : ℝ := 0.18
def number_of_people : ℕ := 9
def final_amount_per_person : ℝ := 67.41

theorem restaurant_bill_split :
  final_amount_per_person = (1 + tip_rate) * original_bill / number_of_people :=
by
  sorry

end restaurant_bill_split_l307_307210


namespace find_x_l307_307983

theorem find_x : ∃ x : ℚ, x^2 + 100 = (x - 12)^2 ∧ x = 11 / 6 :=
by
  existsi (11 / 6 : ℚ)
  split
  sorry

end find_x_l307_307983


namespace apples_count_l307_307525

theorem apples_count (n : ℕ) (h₁ : n > 2)
  (h₂ : 144 / n - 144 / (n + 2) = 1) :
  n + 2 = 18 :=
by
  sorry

end apples_count_l307_307525


namespace min_value_expression_l307_307190

def positive_real (x : ℝ) : Prop := 0 < x

theorem min_value_expression (a b c : ℝ) (ha : positive_real a) (hb : positive_real b) (hc : positive_real c) :
  ∃ v, v = 64 ∧ 
    (a^2 + 4 * a + 4) * (b^2 + 4 * b + 4) * (c^2 + 4 * c + 4) / (a * b * c) ≥ v := 
begin
  use 64,
  split,
  { refl },
  { sorry }
end

end min_value_expression_l307_307190


namespace no_student_received_score_of_4_or_5_l307_307156

theorem no_student_received_score_of_4_or_5
  (n_problems : ℕ) (n_excluding_petya : ℕ) (petya_solves : ℕ) 
  (students_solved : ℕ → ℕ) : 
  n_problems = 5 →
  ∀ i : ℕ, i < n_problems → 
    (students_solved i = 
      [9, 7, 5, 3, 1].nth i .iget) →
  ¬ ∃ students_with_score_4_and_5 : ℕ, 
    students_with_score_4_and_5 = 4 ∨ 
    students_with_score_4_and_5 = 5 := 
by
  sorry

end no_student_received_score_of_4_or_5_l307_307156


namespace dogs_not_eating_either_l307_307556

variable (U : Finset α) (A B : Finset α)
variable (hU : U.card = 75) (hA : A.card = 18) (hB : B.card = 55) (hAB : (A ∩ B).card = 10)

theorem dogs_not_eating_either (U A B : Finset α) (hU : U.card = 75) (hA : A.card = 18) (hB : B.card = 55) (hAB : (A ∩ B).card = 10) :
  (U.card - (A ∪ B).card) = 12 :=
by
  --Proof goes here
  sorry

end dogs_not_eating_either_l307_307556


namespace platform_length_approx_correct_l307_307338

noncomputable def speed_of_train (train_length time_to_cross_signal: ℝ) : ℝ :=
  train_length / time_to_cross_signal

noncomputable def length_of_platform (train_length time_to_cross_signal time_to_cross_platform: ℝ) : ℝ :=
  (speed_of_train train_length time_to_cross_signal) * time_to_cross_platform - train_length

theorem platform_length_approx_correct :
  let train_length := 300.0 in
  let time_to_cross_signal := 14.0 in
  let time_to_cross_platform := 39.0 in
  let P := length_of_platform train_length time_to_cross_signal time_to_cross_platform in
  abs (P - 535.77) < 0.01 :=
sorry

end platform_length_approx_correct_l307_307338


namespace compute_ratio_of_a_and_b_l307_307128

theorem compute_ratio_of_a_and_b
  (a b : ℝ) (n : ℕ)
  (h1 : 0 < a) (h2 : 0 < b)
  (h3 : (a + b * complex.i)^n = -(a - b * complex.i)^n)
  (h4 : ∀ m < n, ¬((a + b * complex.i)^m = -(a - b * complex.i)^m)) :
  a / b = real.sqrt 3 := 
sorry

end compute_ratio_of_a_and_b_l307_307128


namespace fraction_sum_l307_307532

variable (x y : ℚ)

theorem fraction_sum (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := 
by
  sorry

end fraction_sum_l307_307532


namespace total_candies_in_third_set_l307_307681

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l307_307681


namespace find_alpha_l307_307433

def f (x : ℝ) (α : ℝ) : ℝ :=
if x < 3 then 3 * x + 1 else x^2 - α * x

theorem find_alpha (α : ℝ) : 
  (f (f (2/3) α) α = 3) ↔ (α = 2) :=
by
  sorry

end find_alpha_l307_307433


namespace ferris_wheel_capacity_l307_307643

theorem ferris_wheel_capacity :
  let seats := 14
  let people_per_seat := 6
  let fill_percentage := 80 / 100
  let total_spaces := seats * people_per_seat
  let total_people := total_spaces * fill_percentage
  let rounded_total_people := Int.floor total_people
  rounded_total_people = 67 := 
by
  sorry

end ferris_wheel_capacity_l307_307643


namespace intersection_of_A_and_B_l307_307905

-- Define the set A as the solutions to the equation x^2 - 4 = 0
def A : Set ℝ := { x | x^2 - 4 = 0 }

-- Define the set B as the explicit set {1, 2}
def B : Set ℝ := {1, 2}

-- Prove that the intersection of sets A and B is {2}
theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  unfold A B
  sorry

end intersection_of_A_and_B_l307_307905


namespace union_M_N_l307_307612

noncomputable def M : Set ℝ := { x | x^2 - x - 12 = 0 }
noncomputable def N : Set ℝ := { x | x^2 + 3x = 0 }

theorem union_M_N : M ∪ N = { 0, -3, 4 } :=
by
  sorry

end union_M_N_l307_307612


namespace slope_MN_is_1_l307_307288

-- Define the points M and N
def M : ℝ × ℝ := (-Real.sqrt 3, Real.sqrt 2)
def N : ℝ × ℝ := (-Real.sqrt 2, Real.sqrt 3)

-- Define the slope function
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Problem statement
theorem slope_MN_is_1 : slope M N = 1 := by
  sorry

end slope_MN_is_1_l307_307288


namespace c_plus_d_is_even_l307_307640

-- Define the conditions
variables {c d : ℕ}
variables (m n : ℕ) (hc : c = 6 * m) (hd : d = 9 * n)

-- State the theorem to be proven
theorem c_plus_d_is_even : 
  (c = 6 * m) → (d = 9 * n) → Even (c + d) :=
by
  -- Proof steps would go here
  sorry

end c_plus_d_is_even_l307_307640


namespace new_lamp_taller_than_old_lamp_l307_307946

theorem new_lamp_taller_than_old_lamp :
  let old_lamp_in_inches := 12
  let new_lamp_in_cm := 55.56666666666667
  let cm_to_inches_conversion := 2.54
  let new_lamp_in_inches := new_lamp_in_cm / cm_to_inches_conversion
  in new_lamp_in_inches - old_lamp_in_inches = 9.875 :=
by {
  sorry
}

end new_lamp_taller_than_old_lamp_l307_307946


namespace f_one_eq_zero_f_decreasing_range_of_m_l307_307603

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: For any x, y in (0, +∞), f(xy) = f(x) + f(y)
axiom f_multiplicative (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x + f y

-- Condition 2: f(x) < 0 if and only if x > 1
axiom f_negative_iff_gt_one (x : ℝ) (hx : 0 < x) : f x < 0 ↔ 1 < x

-- Question 1: Prove f(1) = 0
theorem f_one_eq_zero : f 1 = 0 := 
  sorry

-- Question 2: Prove that f(x) is decreasing on (0, +∞)
theorem f_decreasing (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 > x2) : f x1 < f x2 :=
  sorry

-- Question 3: Prove the range of m for which the inequality f(x^2 + 1/x^2) ≥ f[m(x + 1/x) - 4] holds for all x in [1, 2]
theorem range_of_m (m : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f (x ^ 2 + 1 / x ^ 2) ≥ f (m * (x + 1 / x) - 4)) ↔ m ∈ set.Ici (33 / 10) :=
  sorry

end f_one_eq_zero_f_decreasing_range_of_m_l307_307603


namespace logical_equivalence_l307_307399

variable (R S T : Prop)

theorem logical_equivalence :
  (R → ¬S ∧ ¬T) ↔ ((S ∨ T) → ¬R) :=
by
  sorry

end logical_equivalence_l307_307399


namespace count_irrational_l307_307375

theorem count_irrational : 
  let numbers := [4, 0, 12/7, 101/1000, Real.sqrt 3, Real.pi / 2] in
  let irrationals := filter (λ x, ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (x : ℝ) = (a : ℝ) / (b : ℝ)) numbers in
  irrationals.length = 2 :=
by
  sorry

end count_irrational_l307_307375


namespace sum_sequence_a_equals_f_one_four_l307_307195

noncomputable def f : ℝ → ℝ := sorry

def sequence_a (n : ℕ) : ℝ := f (1 / (n^2 + 5 * n + 5))

axiom function_property :
  ∀ (x y : ℝ), 1 < x → x < y → (f (1/x) - f (1/y) = f ((x - y) / (1 - x * y)))

theorem sum_sequence_a_equals_f_one_four :
  (sequence_a 1 + sequence_a 2 + sequence_a 3 + sequence_a 4 +
   sequence_a 5 + sequence_a 6 + sequence_a 7 + sequence_a 8) = f (1/4) :=
by
  sorry

end sum_sequence_a_equals_f_one_four_l307_307195


namespace sufficient_monotonic_decreasing_condition_l307_307135

theorem sufficient_monotonic_decreasing_condition (f : ℝ → ℝ) 
  (h_deriv : ∀ x, deriv f x = x^2 - 4*x + 3) : 
  is_decreasing (f ∘ (λ x, x + 1)) (set.Ioo 0 1) :=
by sorry

end sufficient_monotonic_decreasing_condition_l307_307135


namespace recurrence_sequence_a5_l307_307392

theorem recurrence_sequence_a5 :
  ∃ a : ℕ → ℚ, (a 1 = 5 ∧ (∀ n, a (n + 1) = 1 + 1 / a n) ∧ a 5 = 28 / 17) :=
  sorry

end recurrence_sequence_a5_l307_307392


namespace perfect_matching_exists_l307_307553

-- Define the set of boys and girls
def boys : Finset ℕ := Finset.range 10
def girls : Finset ℕ := Finset.range 10

-- Define the friendship relation as a set of pairs
variable (friends : ℕ → Finset ℕ)

-- Condition: For each 1 ≤ k ≤ 10 and for each group of k boys, the number of girls
-- who are friends with at least one boy in the group is not less than k.
def friendship_condition : Prop :=
  ∀ (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 10) (b : Finset ℕ) (hb : b.card = k ∧ b ⊆ boys),
    (b.bUnion friends).card ≥ k

-- The theorem to be proven
theorem perfect_matching_exists (friends : ℕ → Finset ℕ) (h : friendship_condition friends) :
  ∃ (matching : boys ↪ girls), ∀ b ∈ boys, (friends b).count (matching b) ≥ 1 := 
by
  sorry

end perfect_matching_exists_l307_307553


namespace additional_students_needed_l307_307730

theorem additional_students_needed 
  (n : ℕ) 
  (r : ℕ) 
  (t : ℕ) 
  (h_n : n = 82) 
  (h_r : r = 2) 
  (h_t : t = 49) : 
  (t - n / r) * r = 16 := 
by 
  sorry

end additional_students_needed_l307_307730


namespace function_relation_l307_307873

theorem function_relation (f : ℝ → ℝ) 
  (h0 : ∀ x, f (-x) = f x)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) :
  f 0 < f (-6.5) ∧ f (-6.5) < f (-1) := 
by
  sorry

end function_relation_l307_307873


namespace polynomial_conditions_l307_307841

theorem polynomial_conditions (P : ℝ → ℝ) :
  (∀ a b c : ℝ, P(a + b - 2 * c) + P(b + c - 2 * a) + P(c + a - 2 * b) = 3 * P(a - b) + 3 * P(b - c) + 3 * P(c - a)) →
  ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x :=
begin
  sorry
end

end polynomial_conditions_l307_307841


namespace Toby_friends_girls_l307_307749

theorem Toby_friends_girls (F G : ℕ) (h1 : 0.55 * F = 33) (h2 : F - 33 = G) : G = 27 := 
by
  sorry

end Toby_friends_girls_l307_307749


namespace bowling_ball_weight_l307_307405

-- Define the weights of the bowling balls and canoes
variables (b c : ℝ)

-- Conditions provided by the problem statement
axiom eq1 : 8 * b = 4 * c
axiom eq2 : 3 * c = 108

-- Prove that one bowling ball weighs 18 pounds
theorem bowling_ball_weight : b = 18 :=
by
  sorry

end bowling_ball_weight_l307_307405


namespace arithmetic_sequence_proof_l307_307503

-- Definitions of arithmetic sequences {a_n} and {b_n}
noncomputable def a : ℕ → ℝ := sorry
noncomputable def b : ℕ → ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := ∑ i in range n, a i
noncomputable def T (n : ℕ) : ℝ := ∑ i in range n, b i

-- Given condition ∀ n, S_n / T_n = (2n - 3) / (4n - 3)
axiom given_condition (n : ℕ) : S n / T n = (2 * n - 3 : ℝ) / (4 * n - 3)

-- Question: Prove the required value equals 19 / 41
theorem arithmetic_sequence_proof :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 41 :=
by
  sorry

end arithmetic_sequence_proof_l307_307503


namespace sin_sum_to_product_l307_307413

theorem sin_sum_to_product (x : ℝ) : sin (6 * x) + sin (8 * x) = 2 * sin (7 * x) * cos x := 
by 
  sorry

end sin_sum_to_product_l307_307413


namespace female_democrats_l307_307334

theorem female_democrats (F M D_f: ℕ) 
  (h1 : F + M = 780)
  (h2 : D_f = (1/2) * F)
  (h3 : (1/3) * 780 = 260)
  (h4 : 260 = (1/2) * F + (1/4) * M) : 
  D_f = 130 := 
by
  sorry

end female_democrats_l307_307334


namespace correct_option_D_l307_307479

def X_distribution : List (ℤ × ℝ) := [(0, 0.2), (1, 0.8)]
def Y_function (x : ℤ) : ℤ := -3 * x + 1

noncomputable def E (X : List (ℤ × ℝ)) : ℝ :=
  X.foldr (λ p acc, (p.fst : ℝ) * p.snd + acc) 0

noncomputable def D (X : List (ℤ × ℝ)) (E_X : ℝ) : ℝ :=
  X.foldr (λ p acc, ((p.fst : ℝ) - E_X) ^ 2 * p.snd + acc) 0

theorem correct_option_D :
  let E_X := E X_distribution
  let D_X := D X_distribution E_X
  (D_X = 0.16) ∧ ((-3 : ℝ) ^ 2 * D_X = 1.44) :=
by
  let E_X := E X_distribution
  have h0 : E_X = 0.8 := sorry
  let D_X := D X_distribution E_X
  have h1 : D_X = 0.16 := sorry
  have h2 : (-3 : ℝ) ^ 2 * D_X = 1.44 := sorry
  exact ⟨h1, h2⟩

end correct_option_D_l307_307479


namespace unique_a_value_l307_307958

-- Definitions as per conditions
variables {a c x y : ℝ}

-- Main statement
theorem unique_a_value (h1 : a > 1)
  (h2 : ∃ c, ∀ x : ℝ, ∃ y : ℝ, log a x + log a y = c) : a = 2 :=
by sorry

end unique_a_value_l307_307958


namespace arithmetic_sequence_general_term_and_sum_l307_307077

theorem arithmetic_sequence_general_term_and_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) 
  (h1 : a 1 = 1)
  (h2 : a 2 > 1)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d)  -- Defining arithmetic progression
  (h4 : a (4 : ℕ) * a (4 : ℕ) = a (2 : ℕ) * a (9 : ℕ)) :                              
  (∀ n, a n = 3 * n - 2) ∧ (∀ n, S n = (3/2) * n^2 - (1/2) * n) :=
begin
  sorry
end

end arithmetic_sequence_general_term_and_sum_l307_307077


namespace min_p_value_l307_307962

variable (p q r s : ℝ)

theorem min_p_value (h1 : p + q + r + s = 10)
                    (h2 : pq + pr + ps + qr + qs + rs = 20)
                    (h3 : p^2 * q^2 * r^2 * s^2 = 16) :
  p ≥ 2 ∧ ∃ q r s, q + r + s = 10 - p ∧ pq + pr + ps + qr + qs + rs = 20 ∧ (p^2 * q^2 * r^2 * s^2 = 16) :=
by
  sorry  -- proof goes here

end min_p_value_l307_307962


namespace prob_sum_six_l307_307335

def dice_face : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

def is_sum_six (a b : dice_face) : Prop := a.val + b.val = 6

def total_outcomes : ℕ := 36

noncomputable def favorable_outcomes : ℕ := (do
  a ← [⟨1, sorry⟩, ⟨2, sorry⟩, ⟨3, sorry⟩, ⟨4, sorry⟩, ⟨5, sorry⟩, ⟨6, sorry⟩],
  b ← [⟨1, sorry⟩, ⟨2, sorry⟩, ⟨3, sorry⟩, ⟨4, sorry⟩, ⟨5, sorry⟩, ⟨6, sorry⟩],
  [if is_sum_six a b then some () else none]
  ).length

theorem prob_sum_six : favorable_outcomes / total_outcomes = 5 / 36 := sorry

end prob_sum_six_l307_307335


namespace problem1_problem2_l307_307387

-- Problem 1: Prove that (-11) + 8 + (-4) = -7
theorem problem1 : (-11) + 8 + (-4) = -7 := by
  sorry

-- Problem 2: Prove that -1^2023 - |1 - 1/3| * (-3/2)^2 = -(5/2)
theorem problem2 : (-1 : ℚ)^2023 - abs (1 - 1/3) * (-3/2)^2 = -(5/2) := by
  sorry

end problem1_problem2_l307_307387


namespace problem_condition_l307_307959

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp (1 + Real.sin x) + Real.exp (1 - Real.sin x)

theorem problem_condition
  (x1 x2 : ℝ)
  (h1 : x1 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : x2 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : f x1 > f x2) :
  x1^2 > x2^2 :=
sorry

end problem_condition_l307_307959


namespace arccos_neg_half_eq_two_pi_div_three_l307_307813

theorem arccos_neg_half_eq_two_pi_div_three :
  ∀ x : ℝ, (cos x = -1 / 2 ∧ 0 ≤ x ∧ x ≤ π) → x = 2 * π / 3 :=
by
  intro x
  intro h
  sorry

end arccos_neg_half_eq_two_pi_div_three_l307_307813


namespace find_a_if_factor_l307_307129

theorem find_a_if_factor (a : ℚ) :
  (λ x : ℚ, a * x^4 + 12 * x^2 - 5 * a * x + 42) (-5) = 0 →
  a = -57 / 100 :=
by sorry

end find_a_if_factor_l307_307129


namespace count_unit_squares_50th_ring_l307_307012

theorem count_unit_squares_50th_ring (n : ℕ) :
  let w_outer := 2 + 2*n,
      h_outer := 1 + 2*n,
      w_inner := 2 + 2*(n - 1),
      h_inner := 1 + 2*(n - 1) in
  w_outer * h_outer - w_inner * h_inner = 402 :=
by
  sorry

end count_unit_squares_50th_ring_l307_307012


namespace straighten_road_shortens_distance_l307_307801

axiom shortest_distance_between_two_points (A B : ℝ^2) : ∃ (L : ℝ => ℝ^2), (∀ (t : ℝ), (L t = [A,B],) ↔ (dist A B = straight_line_dist A B))

theorem straighten_road_shortens_distance
  (curved_road : ℝ)
  (goal : ∀ (A B : ℝ^2), dist A B < curved_road) :
  (∃ (A B : ℝ^2), (dist A B = straight_line_dist A B)) :=
by
  sorry

end straighten_road_shortens_distance_l307_307801


namespace fruit_bowl_oranges_l307_307149

theorem fruit_bowl_oranges :
  ∀ (bananas apples oranges : ℕ),
    bananas = 2 →
    apples = 2 * bananas →
    bananas + apples + oranges = 12 →
    oranges = 6 :=
by
  intros bananas apples oranges h1 h2 h3
  sorry

end fruit_bowl_oranges_l307_307149


namespace sum_of_first_n_natural_numbers_l307_307964

theorem sum_of_first_n_natural_numbers (n : ℕ) : (∑ i in Finset.range (n + 1), i) = n * (n + 1) / 2 :=
by
  -- Proof skipped for the purpose of this statement
  sorry

end sum_of_first_n_natural_numbers_l307_307964


namespace count_squares_divisible_by_48_l307_307521

theorem count_squares_divisible_by_48 (N : ℕ) (h : N^2 < 100000000) :
  (∃ k : ℕ, N = 24 * k) → (nat.filter (λ n, n * n < 100000000 ∧ (∃ k, n = 24 * k)) (list.range 10000)).length = 416 :=
by sorry

end count_squares_divisible_by_48_l307_307521


namespace point_division_ratio_l307_307221

theorem point_division_ratio {A B C K L M : Type} [EquilateralTriangle A B C]
    (hK : K ∈ Segment A B) (hL : L ∈ Segment B C) (hM : M ∈ Segment A C)
    (hKL_perp_BC : Perpendicular KL BC) (hLM_perp_AC : Perpendicular LM AC) (hMK_perp_AB : Perpendicular MK AB) :
    divides_segment_in_ratio A B K 1 2 ∧ divides_segment_in_ratio B C L 1 2 ∧ divides_segment_in_ratio A C M 1 2 := 
sorry

end point_division_ratio_l307_307221


namespace unique_reconstruction_l307_307220

-- Definition of the sums on the edges given the face values
variables (a b c d e f : ℤ)

-- The 12 edge sums
variables (e₁ e₂ e₃ e₄ e₅ e₆ e₇ e₈ e₉ e₁₀ e₁₁ e₁₂ : ℤ)
variables (h₁ : e₁ = a + b) (h₂ : e₂ = a + c) (h₃ : e₃ = a + d) 
          (h₄ : e₄ = a + e) (h₅ : e₅ = b + c) (h₆ : e₆ = b + f) 
          (h₇ : e₇ = c + f) (h₈ : e₈ = d + f) (h₉ : e₉ = d + e)
          (h₁₀ : e₁₀ = e + f) (h₁₁ : e₁₁ = b + d) (h₁₂ : e₁₂ = c + e)

-- Proving that the face values can be uniquely determined given the edge sums
theorem unique_reconstruction :
  ∃ a' b' c' d' e' f' : ℤ, 
    (e₁ = a' + b') ∧ (e₂ = a' + c') ∧ (e₃ = a' + d') ∧ (e₄ = a' + e') ∧ 
    (e₅ = b' + c') ∧ (e₆ = b' + f') ∧ (e₇ = c' + f') ∧ (e₈ = d' + f') ∧ 
    (e₉ = d' + e') ∧ (e₁₀ = e' + f') ∧ (e₁₁ = b' + d') ∧ (e₁₂ = c' + e') ∧ 
    (a = a') ∧ (b = b') ∧ (c = c') ∧ (d = d') ∧ (e = e') ∧ (f = f') := by
  sorry

end unique_reconstruction_l307_307220


namespace sum_first_n_c_l307_307096

noncomputable def general_term_a (n : ℕ) := 2 * n
noncomputable def general_term_b (n : ℕ) := 2^(n - 1)
noncomputable def sum_Sn (n : ℕ) := n * (n + 1)
noncomputable def c_n (n : ℕ) := (-1)^n * (general_term_a n * general_term_b n + Real.log (sum_Sn n))

theorem sum_first_n_c (n : ℕ) : 
  ∑ k in Finset.range n, c_n (k + 1) = (-1)^n * Real.log (n + 1) - 2/9 - (3 * n + 1) / 9 * (-2)^(n + 1) :=
sorry

end sum_first_n_c_l307_307096


namespace sum_of_even_cubes_is_zero_l307_307809

theorem sum_of_even_cubes_is_zero :
  ∑ i in (finset.range (50)).image (λ k, (k+1)*2), ↑((2*(k+1))^3) +
  ∑ i in (finset.range (50)).image (λ k, -((k+1)*2)), ↑((-(2*(k+1)))^3) = 0 :=
begin
  sorry
end

end sum_of_even_cubes_is_zero_l307_307809


namespace find_BD_l307_307145

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (a b c d : A)
variables (ac bc ab cd bd : ℝ)

noncomputable def Triangle := {A B C : Type} × Real
def conditions (TA : Triangle) := 
  let ⟨A, B, C, AB⟩ := TA in
  ac = 7 ∧ bc = 7 ∧ ab = 2 ∧ cd = 8 ∧ (Bize AB) ∧
  CD = 8

theorem find_BD 
  (h : conditions ⟨A, B, C⟩) : 
  BD = 3 := sorry

end find_BD_l307_307145


namespace find_C_l307_307163

def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 0)
def D : ℝ × ℝ := (0, 1)

theorem find_C :
  ∃ C : ℝ × ℝ, parallelogram A B C D ∧ C = (3, 2) := sorry

-- Definition of a parallelogram (this is a simple placeholder, you might need a detailed definition)
def parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - A.1 = C.1 - B.1) ∧ (D.2 - A.2 = C.2 - B.2)

end find_C_l307_307163


namespace shift_sine_graph_left_l307_307635

theorem shift_sine_graph_left (x : ℝ) :
  let f := λ x, 2 * Real.sin (2 * x - Real.pi / 4)
  let g := λ x, 2 * Real.sin (2 * (x + Real.pi / 4) - Real.pi / 4)
  g 0 = Real.sqrt 2 :=
by
  sorry

end shift_sine_graph_left_l307_307635


namespace solution_is_bx_l307_307436

noncomputable def target_function (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, 0 < x → f(f(x)) + a * f(x) = b * (a + b) * x

theorem solution_is_bx {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (f : ℝ → ℝ) (hf : target_function a b h1 h2 f) :
  ∀ x : ℝ, 0 < x → f(x) = b * x :=
sorry

end solution_is_bx_l307_307436


namespace max_min_difference_l307_307473

noncomputable def f (x : ℝ) : ℝ := x + 9 / x

theorem max_min_difference :
  let M := max (max (f 1) (f 4)) (f 3) in
  let m := min (min (f 1) (f 4)) (f 3) in
  [1, 4].contains 1 → [1, 4].contains 4 → [1, 4].contains 3 →
  M - m = 4 :=
by
  sorry

end max_min_difference_l307_307473


namespace identify_letters_l307_307259

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l307_307259


namespace polar_eq_C1_proof_cartesian_eq_C2_proof_length_MN_l307_307561

noncomputable def polar_eq_C1 (φ : Real) : Real := 2 * Real.cos φ
noncomputable def cartesian_eq_C2 (x y : Real) : Prop := x^2 + y^2 = y

theorem polar_eq_C1_proof : 
  ∀ x y φ : Real, (x = 1 + Real.cos φ) → (y = Real.sin φ) → (Real.sqrt (x^2 + y^2) = 2 * Real.cos (Real.arctan2 y x)) :=
by 
  sorry

theorem cartesian_eq_C2_proof : 
  ∀ ρ θ : Real, (ρ = Real.sin θ) → (ρ^2 = ρ * Real.sin θ) → (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) → (x^2 + y^2 = y) :=
by 
  sorry

theorem length_MN : 
  ∀ θ : Real, (Real.tan θ = 2) → (|MN| = √5) :=
by 
  sorry

end polar_eq_C1_proof_cartesian_eq_C2_proof_length_MN_l307_307561


namespace minimum_value_of_f_l307_307064

def f (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2) + Real.sqrt ((x-1)^2 + (y-1)^2) + Real.sqrt ((x+2)^2 + (y+2)^2)

theorem minimum_value_of_f : ∀ x y : ℝ, f x y ≥ 3 * Real.sqrt 2 :=
by {
  intro x,
  intro y,
  sorry,
}

end minimum_value_of_f_l307_307064


namespace part1_part2_l307_307908

noncomputable def a : ℝ := 2 + Real.sqrt 3
noncomputable def b : ℝ := 2 - Real.sqrt 3

theorem part1 : a * b = 1 := 
by 
  unfold a b
  sorry

theorem part2 : a^2 + b^2 - a * b = 13 :=
by 
  unfold a b
  sorry

end part1_part2_l307_307908


namespace tangent_line_at_neg1_l307_307649

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x - x^3

theorem tangent_line_at_neg1 :
  let slope := (deriv f) (-1)
  let y := f (-1)
  slope = -1 ∧ y = -1 ∧ 
  (∀ x y : ℝ, y = slope * (x + 1) - 1 → x + y + 2 = 0) :=
by 
  let slope := (deriv f) (-1)
  let y := f (-1)
  have h1 : slope = -1 := by sorry
  have h2 : y = -1 := by sorry
  split
  case left => exact h1
  case right =>
    split
    case left => exact h2
    case right =>
      intros "x" "y" h3
      apply_fun (λ e => e + (x + 2) - (y + 1)) -- manipulate the equation to form x + y + 2
      exact sorry

end tangent_line_at_neg1_l307_307649


namespace K_lies_on_fixed_circle_l307_307304

-- Define points and circles
variables (A B K : Point)
variables (Φ1 Φ2 Ω1 Ω2 : Circle)

-- Define the conditions as predicates
def fixed_circles_intersect (Φ1 Φ2 : Circle) (A B : Point) : Prop :=
  Φ1.contains A ∧ Φ1.contains B ∧ Φ2.contains A ∧ Φ2.contains B

def moving_circles_tangent_at (Ω1 Ω2 : Circle) (K : Point) : Prop :=
  Ω1.externa_tangent_to Ω2 ∧ Ω1.contains K ∧ Ω2.contains K

def moving_circle_tangency (Ω : Circle) (Φ1 Φ2 : Circle) : Prop :=
  Ω.internally_tangential_to Φ1 ∧ Ω.externally_tangential_to Φ2

-- The Lean 4 statement
theorem K_lies_on_fixed_circle :
  fixed_circles_intersect Φ1 Φ2 A B →
  moving_circles_tangent_at Ω1 Ω2 K →
  moving_circle_tangency Ω1 Φ1 Φ2 →
  moving_circle_tangency Ω2 Φ2 Φ1 →
  (Φ1.contains K ∨ Φ2.contains K) := 
begin
  sorry,
end

end K_lies_on_fixed_circle_l307_307304


namespace fido_yard_access_fraction_l307_307033

theorem fido_yard_access_fraction (s r : ℝ) (h1 : s = r * Real.sqrt 2) : 
  let aoct := 4 * (1 + Real.sqrt 2) * r^2 in
  let acircle := π * r^2 in
  ∃ (a b : ℤ), (a = 2) ∧ (b = 8) ∧ (acircle / aoct = (π * Real.sqrt a) / b) :=
by
  sorry

end fido_yard_access_fraction_l307_307033


namespace fib_poly_ineq_l307_307187

-- Definitions of Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- Definitions of the polynomial p(x)
variable (p : ℕ → ℕ)

-- Conditions: p(k) = a_k for k = 992 to 1982
def condition (p : ℕ → ℕ) : Prop :=
  ∀ k ∈ (992 : ℕ)..1982, p k = fib k

-- The goal to prove
theorem fib_poly_ineq (p : ℕ → ℕ) : condition p → p 1983 = fib 1983 - 1 :=
by
  sorry

end fib_poly_ineq_l307_307187


namespace graph_transformation_shift_l307_307303

theorem graph_transformation_shift :
  ∀ x, (2 * cos (2 * x)) = (cos (2 * x) - sqrt 3 * sin (2 * x)) → 
        ((2 * cos (2 * x)) = (2 * cos (2 * (x - π / 6)))) :=
by
  intro x
  sorry

end graph_transformation_shift_l307_307303


namespace sqrt_meaningful_range_l307_307137

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 6)) ↔ (x ≥ 6) :=
by
    sorry

end sqrt_meaningful_range_l307_307137


namespace initial_total_coins_l307_307810

theorem initial_total_coins (x : ℕ) (h1 : ∑ i in (finset.range (x + 1)), i = x * (x + 1) / 2)
    (h2 : x > 0) (h3 : 5 * x = x * (x + 1) / 2) : 6 * x = 54 :=
by
    have h4: x * (x + 1) / 2 = 5 * x := h3
    have h5: x * (x + 1) / 2 = 5 * x
    sorry

end initial_total_coins_l307_307810


namespace dish_choice_problem_l307_307362

theorem dish_choice_problem (types_of_dishes : ℕ) (students : ℕ) (choices_per_student : ℕ) 
  (h1 : types_of_dishes = 5) (h2 : students = 3) (h3 : choices_per_student = types_of_dishes)
  : (choices_per_student ^ students) = 125 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end dish_choice_problem_l307_307362


namespace amount_after_two_years_l307_307839

def amount_after_years (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * ((r + 1) ^ n) / (r ^ n)

theorem amount_after_two_years :
  let P : ℕ := 70400
  let r : ℕ := 8
  amount_after_years P r 2 = 89070 :=
  by
    sorry

end amount_after_two_years_l307_307839


namespace power_equivalence_l307_307527

theorem power_equivalence (y : ℝ) (h : 128^3 = 16^y) : 2^(-3 * y) = 1 / 2^15.75 :=
by sorry

end power_equivalence_l307_307527


namespace cos_neg_45_degree_l307_307009

theorem cos_neg_45_degree :
  real.cos (-π / 4) = real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_45_degree_l307_307009


namespace find_C_for_chord_length_l307_307476

/-- Given the circle M: x^2 - 2x + y^2 + 4y - 10 = 0.
   The line x + 3y + C = 0 intersects circle M at points A and B.
   We need to prove that if |AB| = 2∇5, then the possible values of C are 15 and -5. -/
theorem find_C_for_chord_length (C : ℝ) :
  let M := (circle.mk {x : ℝ // x^2 - 2 * x + y^2 + 4 * y - 10 = 0})
      center := (1, -2)  -- Coordinates obtained from completing the square
      radius := sqrt 15 :(sqrt ((1 - 6 + C)/sqrt( 1^2 + 3^2)) = sqrt 10) ->
      | 1 - 6 + C | = 10  -> |AB| = 2 sqrt 5 ) then
      (C = 15 ∨ C = -5) :=
begin  
  sorry  
end

end find_C_for_chord_length_l307_307476


namespace lukas_avg_points_per_game_l307_307207

theorem lukas_avg_points_per_game (total_points games_played : ℕ) (h_total_points : total_points = 60) (h_games_played : games_played = 5) :
  (total_points / games_played = 12) :=
by
  sorry

end lukas_avg_points_per_game_l307_307207


namespace linda_colleges_l307_307973

theorem linda_colleges :
  let wage_per_hour := 10.00
  let app_fee_per_college := 25.00
  let total_hours := 15
  in total_hours * wage_per_hour / app_fee_per_college = 6 :=
by
  sorry

end linda_colleges_l307_307973


namespace triangle_inequality_l307_307778

variables {A B C D E : Type} [ordered_ring A]

-- Conditions
variables (triangle_ABC : Type) (line_DE : triangle_ABC → triangle_ABC)
variables {p1 p2 : triangle_ABC}
variable (area : triangle_ABC → A)

-- Given: D is on AB and E is on AC

def on_AB (D : triangle_ABC) : Prop :=
-- Provide the definition of D being on AB

def on_AC (E : triangle_ABC) : Prop :=
-- Provide the definition of E being on AC

-- Given: Area of triangle ADE equals area of quadrilateral BCED
def area_eq (ADE BCED : triangle_ABC) : Prop := 
  area ADE = area BCED

-- Conclusion: Prove the inequality
theorem triangle_inequality (hd : on_AB p1) (he : on_AC p2) (harea : area_eq (λ x, x) (λ x, x)) :
  ∀ (AD AE BD DE EC : A),
    AD + AE ≥ (BD + DE + EC) / 3 :=
begin
  sorry -- Proof goes here
end

end triangle_inequality_l307_307778


namespace quadratic_real_roots_l307_307539

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x1 x2 : ℝ, k * x1^2 - 6 * x1 - 1 = 0 ∧ k * x2^2 - 6 * x2 - 1 = 0 ∧ x1 ≠ x2) ↔ k ≥ -9 := 
by
  sorry

end quadratic_real_roots_l307_307539


namespace symmetry_center_sum_l307_307615

open Real

noncomputable def f (x : ℝ) := x + sin (π * x) - 3

theorem symmetry_center_sum :
  (∑ k in finset.range 4033, f ((k + 1) / 2017)) = -8066 :=
by
  sorry

end symmetry_center_sum_l307_307615


namespace possible_values_of_a_l307_307614

noncomputable def f (x : ℝ) : ℝ := sorry

theorem possible_values_of_a (a : ℝ) (h1 : ∀ x : ℝ, f x + f (-x) = x^2)
                            (h2 : ∀ x : ℝ, x ≥ 0 → f' x > x)
                            (h3 : f (2 - a) + 2 * a > f a + 2) :
                            a < 1 :=
sorry

end possible_values_of_a_l307_307614


namespace area_shaded_region_A_area_shaded_region_B_area_triangle_PQR_area_shaded_region_D_l307_307748

-- Part (a)
theorem area_shaded_region_A (r : ℝ) (h : r = 2) : 
  let circle_area := π * r^2 in
  let sector_area := (1 / 4) * circle_area in
  sector_area = π :=
by sorry

-- Part (b)
theorem area_shaded_region_B (r : ℝ) (h : r = 2) : 
  let circle_area := π * r^2 in
  let sector_area := (1 / 4) * circle_area in
  let triangle_area := (1 / 2) * r * r in
  (sector_area - triangle_area) = (π - 2) :=
by sorry

-- Part (c)
theorem area_triangle_PQR (a : ℝ) (h : a = 2) : 
  let height := (sqrt (a^2 - (a / 2)^2)) in
  let area := (1 / 2) * a * height in
  area = sqrt 3 :=
by sorry

-- Part (d)
theorem area_shaded_region_D (r : ℝ) (h : r = 2) : 
  let circle_area := π * r^2 in
  let sector_area := (1 / 6) * circle_area in
  let triangle_area := sqrt 3 in
  (sector_area - triangle_area) = (2 / 3 * π - sqrt 3) :=
by sorry

end area_shaded_region_A_area_shaded_region_B_area_triangle_PQR_area_shaded_region_D_l307_307748


namespace intersection_point_correct_l307_307158

-- Points in 3D coordinate space
def P : ℝ × ℝ × ℝ := (3, -9, 6)
def Q : ℝ × ℝ × ℝ := (13, -19, 11)
def R : ℝ × ℝ × ℝ := (1, 4, -7)
def S : ℝ × ℝ × ℝ := (3, -6, 9)

-- Vectors for parameterization
def pq_vector (t : ℝ) : ℝ × ℝ × ℝ := (3 + 10 * t, -9 - 10 * t, 6 + 5 * t)
def rs_vector (s : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - 10 * s, -7 + 16 * s)

-- The proof of the intersection point equals the correct answer
theorem intersection_point_correct : 
  ∃ t s : ℝ, pq_vector t = rs_vector s ∧ 
  pq_vector t = (-19 / 3, 10 / 3, 4 / 3) := 
by
  sorry

end intersection_point_correct_l307_307158


namespace sum_of_interior_angles_l307_307499

def f (n : ℕ) : ℚ := (n - 2) * 180

theorem sum_of_interior_angles (n : ℕ) : f (n + 1) = f n + 180 :=
by
  unfold f
  sorry

end sum_of_interior_angles_l307_307499


namespace non_degenerate_ellipse_l307_307397

theorem non_degenerate_ellipse (k : ℝ) : (∃ a, a = -21) ↔ (k > -21) := by
  sorry

end non_degenerate_ellipse_l307_307397


namespace line_equation_through_points_l307_307648

-- Define the points
def point1 : ℝ × ℝ := (-3, 0)
def point2 : ℝ × ℝ := (0, 4)

-- Prove the line equation passing through these points
theorem line_equation_through_points (x y : ℝ) :
  ∀ (p1 p2 : ℝ×ℝ), p1 = (-3, 0) → p2 = (0, 4) → 
  (∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ a = 4 ∧ b = -3 ∧ c = 12) := 
by
  intros p1 p2 hp1 hp2
  use 4, -3, 12  -- these are a, b, and c in the equation ax + by + c = 0
  split, 
  { -- check the equation 4x - 3y + 12 = 0 holds
    sorry },
  split,
  { -- check a = 4
    refl },
  split,
  { -- check b = -3
    refl },
  { -- check c = 12
    refl }

end line_equation_through_points_l307_307648


namespace boat_speed_with_stream_l307_307780

variable (man_rate_in_still_water : ℝ)
variable (speed_against_stream : ℝ)
variable (speed_of_stream : ℝ)
variable (speed_with_stream : ℝ)

theorem boat_speed_with_stream :
  man_rate_in_still_water = 7 →
  speed_against_stream = 4 →
  speed_of_stream = man_rate_in_still_water - speed_against_stream →
  speed_with_stream = man_rate_in_still_water + speed_of_stream →
  speed_with_stream = 10 :=
by
  intros man_rate_eq speed_against_eq stream_speed_eq with_stream_eq
  rw [man_rate_eq, speed_against_eq] at stream_speed_eq
  rw stream_speed_eq at with_stream_eq
  exact with_stream_eq

end boat_speed_with_stream_l307_307780


namespace number_of_dials_l307_307999

theorem number_of_dials (k : ℕ) (aligned_sums : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → aligned_sums i % 12 = aligned_sums j % 12) ↔ k = 12 :=
by
  sorry

end number_of_dials_l307_307999


namespace arithmetic_sequence_a9_l307_307461

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + d * (n - 1)

-- The sum of the first n terms of an arithmetic sequence.
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a9
  (a1 d : ℤ)
  (h1 : a1 + (a1 + d)^2 = -3)
  (h2 : S 5 a1 d = 10) :
  a 9 a1 d = 20 :=
begin
  sorry
end

end arithmetic_sequence_a9_l307_307461


namespace broken_line_in_same_plane_l307_307560

-- Define the space as R^3 and points as vectors
variables (A B C D E F : EuclideanSpace ℝ (fin 3))

-- Define segment parallelism
def parallel (u v : EuclideanSpace ℝ (fin 3)) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u = k • v

-- Given conditions
variables (hABDE : parallel (B - A) (E - D))
variables (hBCEF : parallel (C - B) (F - E))
variables (hCDFA : parallel (D - C) (A - F))
variables (hABneqDE : (B - A) ≠ (E - D))

-- Theorem to prove all points lie in the same plane
theorem broken_line_in_same_plane :
  ∃ (P : affine_subspace ℝ (EuclideanSpace ℝ (fin 3))),
    A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ D ∈ P ∧ E ∈ P ∧ F ∈ P :=
by sorry

end broken_line_in_same_plane_l307_307560


namespace unique_reconstruction_possible_l307_307212

def can_reconstruct_faces (a b c d e f : ℤ) : 
  Prop :=
  let edges := [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]
  in ∀ (sums : list ℤ), sums = edges →
    ∃ (a' b' c' d' e' f' : ℤ), a = a' ∧ b = b' ∧ c = c' ∧ d = d' ∧ e = e' ∧ f = f'

theorem unique_reconstruction_possible :
  ∀ (a b c d e f : ℤ),
    can_reconstruct_faces a b c d e f :=
begin
  sorry
end

end unique_reconstruction_possible_l307_307212


namespace jean_jail_time_l307_307169

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l307_307169


namespace trailing_zeros_of_10_trailing_zeros_of_2008_l307_307915

-- Definition to count trailing zeros in n!
def trailing_zeros (n : ℕ) : ℕ :=
  let f (n k) := n / (5^k)
  (range (n+1)).sum (λ k, f n k)

theorem trailing_zeros_of_10 : trailing_zeros 10 = 2 := by 
  sorry

theorem trailing_zeros_of_2008 : trailing_zeros 2008 = 500 := by 
  sorry

end trailing_zeros_of_10_trailing_zeros_of_2008_l307_307915


namespace find_ratio_l307_307451

-- Define the condition of the equilateral triangle
structure EquilateralTriangle (A B C : Type) :=
(ac : dist A C = 1)

-- Define the inscribed circle in trapezoid condition
structure InscribedCircle (A K F C : Type) :=
( is_inscribed : ∃ O : Type, Circle O ∧ tangent O A ∧ tangent O K ∧ tangent O F ∧ tangent O C)

-- Define the angle condition
structure AngleCondition (K C F : Type) :=
(angle_30 : ∠ K C F = 30)

-- Define the key points and the ratio theorem
noncomputable def ratio_bk_ck {A B C K F : Type} 
  (e : EquilateralTriangle A B C) 
  (ic : InscribedCircle A K F C)
  (ac: AngleCondition K C F) : ℝ :=
BK / CK = 0.5 + √3

-- Top-level theorem statement
theorem find_ratio {A B C K F : Type} 
  (e : EquilateralTriangle A B C) 
  (ic : InscribedCircle A K F C)
  (ac : AngleCondition K C F) : ℝ :=
begin
  exact 0.5 + √3,
end

end find_ratio_l307_307451


namespace min_rice_weight_l307_307633

theorem min_rice_weight (o r : ℝ) (h1 : o ≥ 4 + 2 * r) (h2 : o ≤ 3 * r) : r ≥ 4 :=
sorry

end min_rice_weight_l307_307633


namespace letters_identity_l307_307244

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l307_307244


namespace discount_calculation_l307_307950

namespace StereoSystem

def old_system_cost : ℝ := 250
def trade_in_percentage : ℝ := 0.80
def new_system_cost : ℝ := 600
def out_of_pocket : ℝ := 250

def trade_in_value := trade_in_percentage * old_system_cost
def total_amount_paid := out_of_pocket + trade_in_value
def discount_amount := new_system_cost - total_amount_paid
def discount_percentage := (discount_amount / new_system_cost) * 100

theorem discount_calculation : discount_percentage = 25 :=
by
  sorry

end StereoSystem

end discount_calculation_l307_307950


namespace sum_of_squares_of_logs_l307_307483

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := log x / log α

theorem sum_of_squares_of_logs (α : ℝ) (x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℝ)
  (h1 : α > 0) (h2 : α ≠ 1)
  (h3 : f x1 α + f x2 α + f x3 α + f x4 α + f x5 α + f x6 α + f x7 α + f x8 α + f x9 α + f x10 α = 50) :
  f (x1 ^ 2) α + f (x2 ^ 2) α + f (x3 ^ 2) α + f (x4 ^ 2) α + f (x5 ^ 2) α +
  f (x6 ^ 2) α + f (x7 ^ 2) α + f (x8 ^ 2) α + f (x9 ^ 2) α + f (x10 ^ 2) α = 100 := by
  sorry

end sum_of_squares_of_logs_l307_307483


namespace largest_power_of_three_in_s_l307_307381

noncomputable def q : ℝ :=
∑ k in Finset.range 10, (k + 1 : ℝ) * Real.log (k + 1)

noncomputable def s : ℝ := Real.exp q

theorem largest_power_of_three_in_s :
  ∃ n : ℕ, s = 3^27 * n ∧ ∀ m : ℕ, s = 3^m * n → m ≤ 27 :=
sorry

end largest_power_of_three_in_s_l307_307381


namespace count_squares_divisible_by_48_l307_307520

theorem count_squares_divisible_by_48 (N : ℕ) (h : N^2 < 100000000) :
  (∃ k : ℕ, N = 24 * k) → (nat.filter (λ n, n * n < 100000000 ∧ (∃ k, n = 24 * k)) (list.range 10000)).length = 416 :=
by sorry

end count_squares_divisible_by_48_l307_307520


namespace quadratic_equation_conditions_l307_307094

theorem quadratic_equation_conditions :
  ∃ (a b c : ℝ), a = 3 ∧ c = 1 ∧ (a * x^2 + b * x + c = 0 ↔ 3 * x^2 + 1 = 0) :=
by
  use 3, 0, 1
  sorry

end quadratic_equation_conditions_l307_307094


namespace sqrt_meaningful_range_l307_307136

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 6)) ↔ (x ≥ 6) :=
by
    sorry

end sqrt_meaningful_range_l307_307136


namespace mr_green_expected_potato_yield_l307_307977

-- Definitions
def length_in_steps : ℕ := 18
def width_in_steps : ℕ := 25
def stride_length_in_inches : ℕ := 30
def stride_length_in_feet : ℝ := (stride_length_in_inches : ℝ) / 12
def yield_rate_1 : ℝ := 0.4
def yield_rate_2 : ℝ := 0.6
def area_threshold : ℝ := 1200

-- Length and width in feet
def length_in_feet := (length_in_steps : ℝ) * stride_length_in_feet
def width_in_feet := (width_in_steps : ℝ) * stride_length_in_feet

-- Garden area in square feet
def garden_area := length_in_feet * width_in_feet

-- Expected potato yield from the garden
def expected_yield := if garden_area > area_threshold then garden_area * yield_rate_2 else garden_area * yield_rate_1

-- Theorem statement
theorem mr_green_expected_potato_yield : expected_yield = 1687.5 := by
  sorry

end mr_green_expected_potato_yield_l307_307977


namespace geom_seq_frac_l307_307095

noncomputable def geom_seq_sum (a1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_frac (a1 q : ℕ) (hq : q > 1) (h_sum : a1 * (q ^ 3 + q ^ 6 + 1 + q + q ^ 2 + q ^ 5) = 20)
  (h_prod : a1 ^ 7 * q ^ (3 + 6) = 64) :
  geom_seq_sum a1 q 6 / geom_seq_sum a1 q 9 = 5 / 21 :=
by
  sorry

end geom_seq_frac_l307_307095


namespace total_face_value_of_notes_l307_307148

theorem total_face_value_of_notes :
  let face_value := 5
  let number_of_notes := 440 * 10^6
  face_value * number_of_notes = 2200000000 := 
by
  sorry

end total_face_value_of_notes_l307_307148


namespace binom_10_5_l307_307826

theorem binom_10_5 : nat.choose 10 5 = 252 := 
by
  sorry

end binom_10_5_l307_307826


namespace find_a_b_transform_line_l307_307061

theorem find_a_b_transform_line (a b : ℝ) (hA : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, a], ![b, 3]]) :
  (∀ x y : ℝ, (2 * (-(x) + a*y) - (b*x + 3*y) - 3 = 0) → (2*x - y - 3 = 0)) →
  a = 1 ∧ b = -4 :=
by {
  sorry
}

end find_a_b_transform_line_l307_307061


namespace matrix_multiplication_correct_l307_307011

def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![4, 2],
  ![-2, 3]
]

def matrix_b : Matrix (Fin 2) (Fin 1) ℤ := ![
  ![5],
  ![-3]
]

def result : Matrix (Fin 2) (Fin 1) ℤ := ![
  ![14],
  ![-19]
]

theorem matrix_multiplication_correct :
  matrix_a.mul matrix_b = result :=
sorry

end matrix_multiplication_correct_l307_307011


namespace function_properties_l307_307899

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x^2)

theorem function_properties : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x < y) → f x > f y) :=
by
  sorry

end function_properties_l307_307899


namespace not_always_possible_repaint_all_white_l307_307377

-- Define the conditions and the problem
def equilateral_triangle_division (n: ℕ) : Prop := 
  ∀ m, m > 1 → m = n^2

def line_parallel_repaint (triangles : List ℕ) : Prop :=
  -- Definition of how the repaint operation affects the triangle colors
  sorry

theorem not_always_possible_repaint_all_white (n : ℕ) (h: equilateral_triangle_division n) :
  ¬∀ triangles, line_parallel_repaint triangles → (∀ t ∈ triangles, t = 0) := 
sorry

end not_always_possible_repaint_all_white_l307_307377


namespace proof_problem_l307_307597

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as {y | y = 2^x, x ∈ ℝ}
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the set B as {x ∈ ℤ | x^2 - 4 ≤ 0}
def B : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}

-- Define the complement of A relative to U (universal set)
def CU_A : Set ℝ := {x | x ≤ 0}

-- Define the proposition to be proved
theorem proof_problem :
  (CU_A ∩ (Set.image (coe : ℤ → ℝ) B)) = {-2.0, 1.0, 0.0} :=
by 
  sorry

end proof_problem_l307_307597


namespace determine_set_B_l307_307618

open Set Finset

variable {α : Type*} [DecidableEq α]

def U : Finset α := {1,2,3,4,5,6,7,8,9}
def A : Finset α
def B : Finset α

theorem determine_set_B (h1 : U \ (A ∪ B) = {1, 3}) (h2 : (U \ A) ∩ B = {2, 4}) : 
  B = {5, 6, 7, 8, 9} := 
  sorry

end determine_set_B_l307_307618


namespace total_candies_in_third_set_l307_307680

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l307_307680


namespace greatest_integer_y_l307_307312

theorem greatest_integer_y (y : ℤ) : 
  (5 : ℝ) / 8 > y / 15 → y ≤ ⌊75 / 8⌋ :=
by
  intro h
  sorry

end greatest_integer_y_l307_307312


namespace value_of_k_range_of_k_l307_307869

noncomputable def quadratic_eq_has_real_roots (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 ∧
    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0

def roots_condition (x₁ x₂ : ℝ) : Prop :=
  |(x₁ + x₂)| + 1 = x₁ * x₂

theorem value_of_k (k : ℝ) :
  quadratic_eq_has_real_roots k →
  (∀ (x₁ x₂ : ℝ), roots_condition x₁ x₂ → x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 →
                    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0 → k = -3) :=
by sorry

theorem range_of_k :
  ∃ (k : ℝ), quadratic_eq_has_real_roots k → k ≤ 1 :=
by sorry

end value_of_k_range_of_k_l307_307869


namespace sum_first_8_terms_l307_307107

def sequence (n : ℕ) : ℤ :=
  if n % 2 = 1 then 2 * n - 3 else 2 ^ (n - 1)

theorem sum_first_8_terms :
  (sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5 + sequence 6 + sequence 7 + sequence 8) = 190 :=
by sorry

end sum_first_8_terms_l307_307107


namespace sin_double_angle_solution_l307_307878

theorem sin_double_angle_solution :
  (cos (4 * Real.pi / 5) * cos (7 * Real.pi / 15) - sin (9 * Real.pi / 5) * sin (7 * Real.pi / 15) =
  cos (x + Real.pi / 2) * cos x + 2 / 3) -> sin (2 * x) = 1 / 3 :=
by
  intro h
  sorry

end sin_double_angle_solution_l307_307878


namespace find_projection_vector_l307_307393

noncomputable def vector_a : ℝ × ℝ × ℝ := (2, -2, 1)
noncomputable def vector_b : ℝ × ℝ × ℝ := (-1, 2, 1)
noncomputable def vector_p : ℝ × ℝ × ℝ := (8/25, 6/25, 1)

theorem find_projection_vector
  (a b : ℝ × ℝ × ℝ)
  (p : ℝ × ℝ × ℝ)
  (collinear : ∃ t : ℝ, p = (a.fst - 3 * t, a.snd + 4 * t, a.snd + 0))
  (ortho : (p.fst - a.fst) * (-3) + (p.snd - a.snd) * 4 = 0):
  p = (8/25, 6/25, 1) :=
sorry

end find_projection_vector_l307_307393


namespace find_p_not_geometric_sequence_arithmetic_sequence_when_r_2_l307_307071

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {r p : ℝ}
variables [ordered_field ℝ]

-- Given conditions
axiom abs_a1_ne_abs_a2 : |a 1| ≠ |a 2|
axiom condition (n : ℕ) (h_pos : 0 < n) : r * (n - p) * S (n + 1) = n^2 * a n + (n^2 - n - 2) * a 1

-- Proofs to be completed
theorem find_p : p = 1 :=
sorry

theorem not_geometric_sequence : ¬ (∀ n, (a (n + 1) = k * a n) ∧ k ≠ 1 ∧ k ≠ -1) :=
sorry

theorem arithmetic_sequence_when_r_2 (h_r_2 : r = 2) : ∀ n, a (n + 1) = a 1 + n * (a 2 - a 1) :=
sorry

end find_p_not_geometric_sequence_arithmetic_sequence_when_r_2_l307_307071


namespace five_digit_integers_divisible_by_12_count_l307_307517

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

def two_digit_multiples_of_12 : List ℕ :=
  [12, 24, 36, 48, 60, 72, 84, 96]

def count_five_digit_integers_divisible_by_12 : ℕ :=
  9 * 10 * 10 * 10 * 8

theorem five_digit_integers_divisible_by_12_count :
  (count_five_digit_integers_divisible_by_12 = 72000) :=
by
  rw [count_five_digit_integers_divisible_by_12]
  norm_num
  -- We skip the detailed proof steps here
  sorry

end five_digit_integers_divisible_by_12_count_l307_307517


namespace find_rotation_center_l307_307391
noncomputable def f (z : ℂ) : ℂ := ((-2 + Complex.I * Real.sqrt 2) * z + (- Real.sqrt 2 - 10 * Complex.I)) / 2

theorem find_rotation_center : ∃ c : ℂ, f c = c ∧ c = - (3 * Real.sqrt 2) / 2 - 10 * Complex.I / 9 :=
by
  sorry

end find_rotation_center_l307_307391


namespace angle_HIM_eq_90_iff_l307_307022

-- Definitions of various geometric entities
variables {A B C H I M : Point}
variable [Orthocenter H (Triangle A B C)]
variable [Incenter I (Triangle A B C)]
variable [Midpoint M B C]

-- The proof statement
theorem angle_HIM_eq_90_iff :
  angle H I M = 90 * degree ↔ distance A B + distance A C = 2 * distance B C :=
sorry

end angle_HIM_eq_90_iff_l307_307022


namespace largest_possible_S_l307_307159

theorem largest_possible_S :
  ∃ (S : ℕ), 
  (∀ (C : ℕ → ℕ) (H : bij (C : fin 9 → ℕ) (finset.range 1 10)), 
  (sum (set_of (λ i, C i)) = S) ∧ 
  (sum (set_of (λ i, C (i + 1))) = S) ∧ 
  (sum (set_of (λ i, C (i + 2))) = S)) ∧
  (∀ (S' : ℕ), (∀ (C : ℕ → ℕ) (H : bij (C : fin 9 → ℕ) (finset.range 1 10)), 
  (sum (set_of (λ i, C i)) = S') ∧ 
  (sum (set_of (λ i, C (i + 1))) = S') ∧ 
  (sum (set_of (λ i, C (i + 2))) = S')) → S' ≤ 28) :=
begin
  sorry
end

end largest_possible_S_l307_307159


namespace largest_n_divides_factorial_l307_307040

theorem largest_n_divides_factorial (n : ℕ) : 
  (18^(n:Int)) ∣ factorial 30 ↔ n <= 7 := 
sorry

end largest_n_divides_factorial_l307_307040


namespace fran_uniform_cost_l307_307855

def pants_cost : ℝ := 20
def shirt_cost : ℝ := 2 * pants_cost
def tie_cost : ℝ := (1 / 5) * shirt_cost
def socks_cost : ℝ := 3
def jacket_cost : ℝ := 3 * shirt_cost
def shoes_cost : ℝ := 40

def uniform_cost : ℝ := pants_cost + shirt_cost + tie_cost + socks_cost + jacket_cost + shoes_cost
def discount_per_uniform (cost : ℝ) : ℝ := 0.10 * cost
def discounted_uniform_cost (cost : ℝ) : ℝ := cost - discount_per_uniform(cost)

def total_discounted_cost (num_uniforms : ℕ) (uniform_cost : ℝ) : ℝ := num_uniforms * discounted_uniform_cost(uniform_cost)

theorem fran_uniform_cost : total_discounted_cost 5 uniform_cost = 1039.50 := by
  -- proof steps should go here
  sorry

end fran_uniform_cost_l307_307855


namespace incenter_orthocenter_inequality_l307_307607

noncomputable theory
open_locale classical

variables {A B C A1 B1 C1 : Type*} [metric_space A]
  {triangle_ABC : triangle A B C}
  (AA1 BB1 CC1 : line_segment_in_circle triangle_ABC)
  (I : incenter triangle_ABC)
  (H : orthocenter (triangle A1 B1 C1))

-- The required hypothesis:
-- A1, B1, and C1 are points on sides BC, CA, and AB of acute triangle ABC respectively.
-- AA1, BB1, and CC1 are the internal angle bisectors of triangle ABC.
-- I is the incenter of triangle ABC.
-- H is the orthocenter of triangle A1 B1 C1.

theorem incenter_orthocenter_inequality :
  dist A H + dist B H + dist C H ≥ dist A I + dist B I + dist C I :=
sorry

end incenter_orthocenter_inequality_l307_307607


namespace pairs_integer_criterion_l307_307036

theorem pairs_integer_criterion (c d : ℕ) (h1 : c > 1) (h2 : d > 1) : 
  (d ≤ c) ↔ (∀ (Q : ℤ[X]), monic Q → degree Q = d → ∀ (p : ℕ), 
  p > c * (2 * c + 1) → 
  ∃ (S : set ℤ), S.finite ∧ S.card ≤ (2 * c - 1) * p / (2 * c + 1) ∧ 
  (∀ (x : ℤ), ∃ (s : ℤ), s ∈ S ∧ ∃ (n : ℕ), Q^[n] x ≡ s [MOD p])) := 
sorry

end pairs_integer_criterion_l307_307036


namespace boat_distance_downstream_l307_307290

theorem boat_distance_downstream :
  ∃ (d : ℝ) (c : ℝ), let boat_speed := 12 in
                     let time_downstream := 3 in
                     let time_upstream := 4.2 in
                     d = (boat_speed + c) * time_downstream ∧
                     d = (boat_speed - c) * time_upstream ∧
                     d = 42 :=
by
  sorry

end boat_distance_downstream_l307_307290


namespace projection_norm_eq_three_l307_307506

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions based on conditions
variables (v w : V)
variable (h_v_norm : ∥v∥ = 6)
variable (h_w_norm : ∥w∥ = 4)
variable (h_dot : ⟪v, w⟫ = 12)

-- Statement of the problem
theorem projection_norm_eq_three :
  ∥(⟪v, w⟫ / ∥w∥^2) • w∥ = 3 :=
sorry

end projection_norm_eq_three_l307_307506


namespace line_equation_y_intercept_distance_l307_307068

theorem line_equation_y_intercept_distance :
  ∃ (a b c : ℝ), (b = 10) ∧ (|c| / real.sqrt (a^2 + b^2) = 8) ∧ 
  ((a = 3 ∧ c = -40) ∨ (a = 3 ∧ c = 40) ∨ (a = -3 ∧ c = 40) ∨ (a = -3 ∧ c = -40)) :=
by
  use [3, 10, 40]
  sorry

end line_equation_y_intercept_distance_l307_307068


namespace smallest_color_count_l307_307075

-- Define the size of the grid and the condition on the colors.
def grid_size : ℕ := 2023
def color_count (n : ℕ) : Prop :=
  ∀ c : ℕ, 
  (∃ f : ℕ × ℕ → ℕ,
    (∀ i j, f (i, j) < n) ∧
    (∀ i k l s, (k < l) ∨ (s < i) → f (i, k) = f (s, l) → False) ∧
    (∃ r, ∃ c₁ c₂: ℕ, r < grid_size → r < grid_size → 
      (c₁ = 0 ∨ c₂ = grid_size - 1) ∧ 
      (∀ i, c₁ ≤ i ∧ i ≤ c₂ → f (r, i) = c)))

-- Define the main theorem to find the minimum possible value of n
theorem smallest_color_count : ∃ n, color_count n ∧ n = 338 :=
begin
  sorry
end

end smallest_color_count_l307_307075


namespace find_values_l307_307650

theorem find_values (a b c : ℤ)
  (h1 : ∀ x, x^2 + 9 * x + 14 = (x + a) * (x + b))
  (h2 : ∀ x, x^2 + 4 * x - 21 = (x + b) * (x - c)) :
  a + b + c = 12 :=
sorry

end find_values_l307_307650


namespace cubic_diff_l307_307083

theorem cubic_diff (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : a^3 - b^3 = 208 :=
by
  sorry

end cubic_diff_l307_307083


namespace minimize_distance_sum_l307_307453

-- Define the point A
structure Point :=
  (x : ℝ) (y : ℝ)

def A : Point := { x := 4, y := 2 }

-- Define the parabola in terms of a predicate
def isParabolaPoint (P : Point) : Prop :=
  P.y ^ 2 = 4 * P.x

-- Define the focus of the parabola
def focus : Point := { x := 1, y := 0 }

-- Define the distance between two points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- The proof problem statement
theorem minimize_distance_sum (M : Point) (hM : isParabolaPoint M) :
  ∀ N : Point, isParabolaPoint N → (distance M A + distance M focus) ≤ (distance N A + distance N focus) :=
sorry

end minimize_distance_sum_l307_307453


namespace problem_statement_l307_307418

noncomputable def valid_functions (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = sqrt 1990 * Real.exp x ∨ f x = -sqrt 1990 * Real.exp x ∨
          f x = sqrt 1990 * Real.exp (-x) ∨ f x = -sqrt 1990 * Real.exp (-x)

theorem problem_statement (f : ℝ → ℝ) (h : ∀ α : ℝ, f.derivative α * f.derivative α = 1990 + ∫ x in 0..α, f x * f x + f.derivative x * f.derivative x) : 
  valid_functions f :=
sorry

end problem_statement_l307_307418


namespace prove_angle_equality_l307_307956

noncomputable def circle (P : Type*) := {center : P × P, radius : ℝ}

variables (P : Type*) [metric_space P]

def is_tangent (C₁ C₂ : circle P) (T : P) : Prop :=
  dist C₁.center.1 T = C₁.radius ∧ dist C₂.center.1 T = C₂.radius

def is_on (p : P) (C : circle P) : Prop :=
  dist C.center.1 p = C.radius

def tangent_points (C : circle P) (B : P) :=
  {M N : P // dist C.center.1 M = C.radius ∧ dist C.center.1 N = C.radius}

variables (Γ Γ' : circle P) (T B M N : P)

axiom tangent_at_T : is_tangent Γ Γ' T
axiom B_on_GammaPrime : is_on B Γ'
axiom tangents_intersect (h : B_on_GammaPrime) : tangent_points Γ B

-- Axiom for angles to be equal
axiom angle_equality : ∀ (C : circle P) (A B C D : P),
  is_tangent C Γ' D → is_on B C → is_on A C → is_on D Γ →
  angle M T B = angle B T N

-- The proof statement:
theorem prove_angle_equality : angle M T B = angle B T N :=
begin
  apply angle_equality,
  all_goals { sorry }
end

end prove_angle_equality_l307_307956


namespace rod_balance_equilibrium_l307_307789

-- Define variables and constants to represent the problem conditions
variables (x : ℝ)
constants 
  (weight_0 weight_1 weight_2 weight_3 weight_4 : ℝ)
  (dist_1 dist_2 dist_3 dist_4 : ℝ)

-- Assign the given weight values
def w0 := 20
def w1 := 30
def w2 := 40
def w3 := 50
def w4 := 60

-- Assign the given distances
def d1 := 1
def d2 := 2
def d3 := 3
def d4 := 4

-- Non-computable definition for the equilibrium equation
noncomputable def equilibrium_eq := w0 * x + w1 * (x - d1) + w2 * (x - d2) + w3 * (x - d3) + w4 * (x - d4)

-- The theorem stating the equilibrium point of the rod
theorem rod_balance_equilibrium : equilibrium_eq x = 0 → x = 2.5 :=
  by 
  skip

-- Proof placeholder
sorry

end rod_balance_equilibrium_l307_307789


namespace f_relation_l307_307897

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_relation :
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by
  sorry

end f_relation_l307_307897


namespace tangent_line_eqn_unique_local_minimum_l307_307485

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + 2) / x

def tangent_line_at_1 (x y : ℝ) : Prop :=
  2 * x + y - Real.exp 1 - 4 = 0

theorem tangent_line_eqn :
  tangent_line_at_1 1 (f 1) :=
sorry

noncomputable def h (x : ℝ) : ℝ := Real.exp x * (x - 1) - 2

theorem unique_local_minimum :
  ∃! c : ℝ, 1 < c ∧ c < 2 ∧ (∀ x < c, f x > f c) ∧ (∀ x > c, f c < f x) :=
sorry

end tangent_line_eqn_unique_local_minimum_l307_307485


namespace sum_of_ages_l307_307984

variable (P_years Q_years : ℝ) (D_years : ℝ)

-- conditions
def condition_1 : Prop := Q_years = 37.5
def condition_2 : Prop := P_years = 3 * (Q_years - D_years)
def condition_3 : Prop := P_years - Q_years = D_years

-- statement to prove
theorem sum_of_ages (h1 : condition_1 Q_years) (h2 : condition_2 P_years Q_years D_years) (h3 : condition_3 P_years Q_years D_years) :
  P_years + Q_years = 93.75 :=
by sorry

end sum_of_ages_l307_307984


namespace cannot_determine_order_l307_307425

theorem cannot_determine_order (m : Stone → ℝ) (问 : (Stone → Stone → Stone → Prop) → Prop) :
  (∀ {A B C : Stone}, (问 (λ x y z, m x < m y ∧ m y < m z) A B C) ∨ (问 (λ x y z, ¬ (m x < m y ∧ m y < m z)) A B C)) →
  (∃ s : Finset (Equiv.Perm Stone), s.card ≥ 2) :=
by
  sorry

end cannot_determine_order_l307_307425


namespace total_candies_in_third_set_l307_307712

-- Definitions for the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Conditions based on the problem statement
def conditions : Prop :=
  (L1 + L2 + L3 = S1 + S2 + S3) ∧ 
  (S1 + S2 + S3 = M1 + M2 + M3) ∧
  (S1 = M1) ∧ 
  (L1 = S1 + 7) ∧ 
  (L2 = S2) ∧
  (M2 = L2 - 15) ∧ 
  (L3 = 0)

-- Statement to verify the total number of candies in the third set is 29
theorem total_candies_in_third_set (h : conditions) : L3 + S3 + M3 = 29 := 
sorry

end total_candies_in_third_set_l307_307712


namespace smallest_abs_term_is_a7_l307_307449

noncomputable def a_sequence (a_1 : ℕ) (d : ℕ) : ℕ → ℕ
| 0     => a_1
| (n+1) => a_sequence a_1 d n + d

def S_n (a_1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n + 1) * a_1 + (n * (n + 1) / 2) * d

theorem smallest_abs_term_is_a7 
  (a_1 : ℕ) (d : ℕ) 
  (h_a1_pos : a_1 > 0) 
  (h_n : n ∈ Set.Ioo (0: ℕ) (13: ℕ))
  (h_S12_pos : S_n a_1 d 12 > 0) 
  (h_S13_neg : S_n a_1 d 13 < 0) : 
  ∃ m ∈ Set.Ioo (0: ℕ) (13: ℕ), |a_sequence a_1 d m| = |a_sequence a_1 d 7| :=
begin
  sorry
end

end smallest_abs_term_is_a7_l307_307449


namespace area_of_triangle_AEC_l307_307570

theorem area_of_triangle_AEC (BE EC : ℝ) (h_ratio : BE / EC = 3 / 2) (area_abe : ℝ) (h_area_abe : area_abe = 27) : 
  ∃ area_aec, area_aec = 18 :=
by
  sorry

end area_of_triangle_AEC_l307_307570


namespace cylinder_properties_l307_307771

theorem cylinder_properties (h r : ℝ) (h_eq : h = 15) (r_eq : r = 5) :
  let total_surface_area := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  let volume := Real.pi * r^2 * h
  total_surface_area = 200 * Real.pi ∧ volume = 375 * Real.pi :=
by
  sorry

end cylinder_properties_l307_307771


namespace length_of_BE_l307_307828

/-- Definition of a cyclic quadrilateral ABCD with given conditions and properties --/
def CyclicQuadrilateral (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
  AB = AD ∧ BC = CD ∧
  ∃ E BD BD_length AC_length : Type,  
    (is_intersection AC BD E) ∧
    AB = 3 ∧ BD = 5
                         
/-- Lean 4 statement to prove BE = 2.5 given the conditions --/
theorem length_of_BE {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (h : CyclicQuadrilateral A B C D) : 
  ∃ BD BE : ℝ, is_perpendicular_bisector BD ∧ BD = 5 ∧ BE = 2.5 :=
sorry

end length_of_BE_l307_307828


namespace unique_face_numbers_l307_307215

-- Define the problem statement and conditions
theorem unique_face_numbers (a b c d e f : ℤ) (sums : list ℤ) (h : sums = [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]) : 
  (∃ (n : ℕ → ℤ), (n 0 = a ∧ n 1 = b ∧ n 2 = c ∧ n 3 = d ∧ n 4 = e ∧ n 5 = f)) :=
by 
  rw h
  -- Additional detailed steps are omitted
  sorry

end unique_face_numbers_l307_307215


namespace arccos_neg_half_eq_two_thirds_pi_l307_307824

theorem arccos_neg_half_eq_two_thirds_pi : 
  ∃ θ : ℝ, θ ∈ set.Icc 0 real.pi ∧ real.cos θ = -1/2 ∧ real.arccos (-1/2) = θ := 
sorry

end arccos_neg_half_eq_two_thirds_pi_l307_307824


namespace num_four_digit_int_with_4_or_5_correct_l307_307120

def num_four_digit_int_with_4_or_5 : ℕ :=
  5416

theorem num_four_digit_int_with_4_or_5_correct (A B : ℕ) (hA : A = 9000) (hB : B = 3584) :
  num_four_digit_int_with_4_or_5 = A - B :=
by
  rw [hA, hB]
  sorry

end num_four_digit_int_with_4_or_5_correct_l307_307120


namespace eccentricity_relationship_l307_307029

variable {R : Type*} [LinearOrderedField R]
variables (e₁ e₂ : R)
variables (m n : R)
variables (lenPQ lenAB lenMN : R)

def chord_ratio_conditions :=
  ∃ (lenPQ lenAB lenMN : R),
    (lenPQ / lenAB) ≥ m ∧ 
    (lenMN / lenPQ) ≥ n ∧
    m = n

theorem eccentricity_relationship 
  (e₁ e₂ : R) 
  (m n : R)
  (hc : chord_ratio_conditions e₁ e₂ m n) : 
  e₁ * e₂ = 1 :=
  sorry

end eccentricity_relationship_l307_307029


namespace negation_of_p_l307_307108

-- Given proposition
def p := ∀ (a : ℝ), (0 < a) → (real.exp a ≥ 1)

-- Negation of the proposition
def not_p := ¬ p

-- Formulating the negation
theorem negation_of_p : not_p ↔ ∃ (a : ℝ), 0 < a ∧ real.exp a < 1 :=
by
  sorry

end negation_of_p_l307_307108


namespace no_closed_non_intersecting_path_on_pipe_l307_307856

-- Define the structure of the "pipe" cube.
structure Cube (n : ℕ) :=
  (vertices : Fin n -> Fin n -> Fin n -> Type)
  (edges : (Fin n -> Fin n -> Fin n) -> (Fin n -> Fin n -> Fin n) -> Prop)

def pipe_surface (c : Cube 3) : Prop :=
  let vertices_per_face := 5
  let surface_vertices := vertices_per_face * vertices_per_face * 6
  let diagonals := 4 * 9 + 2 * 8 + 12 -- 4 sides, 2 faces, 12 edge cubes
  diagonals = 64

theorem no_closed_non_intersecting_path_on_pipe : ∀ (c : Cube 3), pipe_surface c → ¬∃ (p : list (Fin 3 × Fin 3 × Fin 3)), 
  (∀ v ∈ p, v ∈ c.vertices) ∧ 
  (∀ (u v : Fin 3 × Fin 3 × Fin 3), (u ∈ p → v ∈ p → c.edges u v)) ∧ 
  (∀ u ∈ p, ∀ v ∈ p, u ≠ v → c.edges u v → u ≠ v) :=
begin
  intros c h,
  sorry
end

end no_closed_non_intersecting_path_on_pipe_l307_307856


namespace identify_letters_l307_307254

/-- Each letter tells the truth if it is an A and lies if it is a B. -/
axiom letter (i : ℕ) : bool
def is_A (i : ℕ) : bool := letter i
def is_B (i : ℕ) : bool := ¬letter i

/-- First letter: "I am the only letter like me here." -/
def first_statement : ℕ → Prop := 
  λ i, (is_A i → ∀ j, (i = j) ∨ is_B j)

/-- Second letter: "There are fewer than two A's here." -/
def second_statement : ℕ → Prop := 
  λ i, is_A i → ∃ j, ∀ k, j ≠ k → is_B j

/-- Third letter: "There is one B among us." -/
def third_statement : ℕ → Prop := 
  λ i, is_A i → ∃ ! j, is_B j

/-- Each letter statement being true if the letter is A, and false if the letter is B. -/
def statement_truth (i : ℕ) (statement : ℕ → Prop) : Prop := 
  is_A i ↔ statement i

/-- Given conditions, prove the identity of the three letters is B, A, A. -/
theorem identify_letters : 
  ∃ (letters : ℕ → bool), 
    (letters 0 = false) ∧ -- B
    (letters 1 = true) ∧ -- A
    (letters 2 = true) ∧ -- A
    (statement_truth 0 first_statement) ∧
    (statement_truth 1 second_statement) ∧
    (statement_truth 2 third_statement) :=
by
  sorry

end identify_letters_l307_307254


namespace weight_of_new_man_l307_307735

noncomputable def newManWeight (W : ℝ) := sorry

theorem weight_of_new_man (W : ℝ) (H : W' = W - 45 + N) 
  (H1 : W' = W + 30): 
  newManWeight W = 75 := sorry

end weight_of_new_man_l307_307735


namespace categorize_numbers_l307_307034

def is_integer (n : ℚ) : Prop := (n.den = 1)
def is_fraction (n : ℚ) : Prop := true -- Any rational number can be considered a fraction
def is_negative_rational (n : ℚ) : Prop := n < 0

axiom given_numbers : list ℚ := [-7, 3.01, 2015, -0.142, 0.1, 0, 99, -7/5]

theorem categorize_numbers :
  (∀ n ∈ given_numbers, (is_integer n ↔ n ∈ [-7, 2015, 0, 99]) ∧
                        (is_fraction n ↔ n ∈ [3.01, -0.142, 0.1, -7/5]) ∧
                        (is_negative_rational n ↔ n ∈ [-7, -0.142, -7/5])) :=
by {
  sorry
}

end categorize_numbers_l307_307034


namespace log_base_change_proof_l307_307030

theorem log_base_change_proof : 
  ∀ (b x : ℝ), (b = 16) → (x = 4) → (16 = 2 ^ 4) → (4 = 2 ^ 2) → log 16 4 = 1 / 2 :=
by 
  intros b x h_b h_x eq1 eq2
  sorry

end log_base_change_proof_l307_307030


namespace task_completion_equation_l307_307343

-- Conditions stated in the problem
def A_task_completion_rate : ℝ := 1 / 3
def B_task_completion_rate : ℝ := 1 / 5
def A_work_days (x : ℝ) : ℝ := x + 1
def B_work_days (x : ℝ) : ℝ := x

-- The theorem stating the equation to be proven
theorem task_completion_equation (x : ℝ) :
  (A_work_days x * A_task_completion_rate) + (B_work_days x * B_task_completion_rate) = 1 :=
by
  sorry

end task_completion_equation_l307_307343


namespace minimum_surface_area_of_sphere_l307_307870

theorem minimum_surface_area_of_sphere
  (a h : ℝ)
  (ha : a > 0)
  (hh : h > 0)
  (volume_prism : (1/3) * (sqrt 3 / 4) * a^2 * h = 3 * sqrt 3)
  (radius_of_sphere : ℝ := sqrt((1/3) * (a^2 / 2) + (18 / a^2)^2)) :
  4 * π * radius_of_sphere^2 = 12 * π * 324 :=
begin
  sorry
end

end minimum_surface_area_of_sphere_l307_307870


namespace max_unique_sums_l307_307351

-- Definitions for the values of the coins
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

-- The list of sums calculated from pairs of the coin values
def sums : List ℕ := [penny + penny, penny + nickel, penny + dime, penny + quarter, penny + half_dollar,
                      nickel + nickel, nickel + dime, nickel + quarter, nickel + half_dollar,
                      dime + dime, dime + quarter, dime + half_dollar,
                      quarter + quarter, quarter + half_dollar, half_dollar + half_dollar]

-- The propositions stating the unique sums
def unique_sums (s : List ℕ) : Prop := s.nodup

-- The main statement to be proven
theorem max_unique_sums :
  unique_sums sums ∧ sums.length = 15 := 
sorry

end max_unique_sums_l307_307351


namespace letters_identity_l307_307236

-- Let's define the types of letters.
inductive Letter
| A
| B

-- Predicate indicating whether a letter tells the truth or lies.
def tells_truth : Letter → Prop
| Letter.A := True
| Letter.B := False

-- Define the three letters
def first_letter : Letter := Letter.B
def second_letter : Letter := Letter.A
def third_letter : Letter := Letter.A

-- Conditions from the problem.
def condition1 : Prop := ¬ (tells_truth first_letter)
def condition2 : Prop := tells_truth second_letter → (first ≠ Letter.A ∧ second ≠ Letter.A → True)
def condition3 : Prop := tells_truth third_letter ↔ second = Letter.A → True

-- Proof statement
theorem letters_identity : 
  first_letter = Letter.B ∧ 
  second_letter = Letter.A ∧ 
  third_letter = Letter.A  :=
by
  split; try {sorry}

end letters_identity_l307_307236


namespace ellipse_eccentricity_l307_307450

noncomputable def right_focus (a b : ℝ) (h : a > b  ∧ b > 0) : ℝ := real.sqrt (a^2 - b^2)

theorem ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) :
  let c := right_focus a b h in
  let e := c / a in
  ∀ x y : ℝ, (c * (2 * x) = x + a) ∧ (2 * y = - b) ∧ (x, y) ∈ set_of (λ (p : ℝ × ℝ), (p.1^2 / a^2) + (p.2^2 / b^2) = 1) → 
  e = real.sqrt (1/3) :=
by 
  sorry

end ellipse_eccentricity_l307_307450


namespace circles_disjoint_l307_307427

theorem circles_disjoint (a : ℝ) : ((x - 1)^2 + (y - 1)^2 = 4) ∧ (x^2 + (y - a)^2 = 1) → (a < 1 - 2 * Real.sqrt 2 ∨ a > 1 + 2 * Real.sqrt 2) :=
by sorry

end circles_disjoint_l307_307427


namespace ratio_of_spent_to_left_after_video_game_l307_307981

-- Definitions based on conditions
def total_money : ℕ := 100
def spent_on_video_game : ℕ := total_money * 1 / 4
def money_left_after_video_game : ℕ := total_money - spent_on_video_game
def money_left_after_goggles : ℕ := 60
def spent_on_goggles : ℕ := money_left_after_video_game - money_left_after_goggles

-- Statement to prove the ratio
theorem ratio_of_spent_to_left_after_video_game :
  (spent_on_goggles : ℚ) / (money_left_after_video_game : ℚ) = 1 / 5 := 
sorry

end ratio_of_spent_to_left_after_video_game_l307_307981


namespace roots_complex_conjugates_l307_307189

noncomputable def complex_roots (a b : ℝ) : Prop :=
  ∃ z : ℂ, (z + conj(z) = -(6 + a * complex.I)) ∧ (z * conj(z) = (13 + b * complex.I)) ∧ z ≠ conj(z)

theorem roots_complex_conjugates (a b : ℝ) :
  complex_roots a b → a = 0 ∧ b = 0 :=
by
  sorry

end roots_complex_conjugates_l307_307189


namespace perimeter_rhombus_l307_307093

-- Given conditions
def rhombus (ABCD : Type) : Prop :=
  ∀ A B C D : ABCD,
    -- All sides are equal
    ∀ (side : ℝ), 
      side = 8 / 2 ∧ 
      side = 6 / 2

-- Define points A, B, C, D
variables {A B C D : Type}
variables [linear_ordered_semiring ℝ]

-- Proposition to be proved
theorem perimeter_rhombus (H : rhombus ABCD) 
    (diagonal_AC : ℝ := 8) 
    (diagonal_BD : ℝ := 6) : 
  ∃ (perimeter : ℝ), perimeter = 20 :=
by
  -- Compute the lengths of halves of the diagonals
  let AO := 8 / 2
  let BO := 6 / 2
  -- Compute the side length using Pythagoras theorem
  let side := real.sqrt (AO^2 + BO^2)
  -- The perimeter of the rhombus is 4 times the side length
  let perimeter := 4 * side
  exists perimeter
  sorry

end perimeter_rhombus_l307_307093


namespace coin_flip_probability_l307_307724

theorem coin_flip_probability (p : ℝ) 
  (h : p^2 + (1 - p)^2 = 4 * p * (1 - p)) : 
  p = (3 + Real.sqrt 3) / 6 :=
sorry

end coin_flip_probability_l307_307724


namespace binomial_inequality_l307_307225

open BigOperators

theorem binomial_inequality (n x : ℕ) : 
  (nat.choose (2*n + x) n) * (nat.choose (2*n - x) n) ≤ (nat.choose (2*n) n) ^ 2 :=
by
  sorry

end binomial_inequality_l307_307225


namespace minimum_c_value_l307_307988

theorem minimum_c_value
  (a b c k : ℕ) (h1 : b = a + k) (h2 : c = b + k) (h3 : a < b) (h4 : b < c) (h5 : k > 0) :
  c = 6005 :=
sorry

end minimum_c_value_l307_307988


namespace train_pass_time_l307_307332

def train_length : ℝ := 64
def train_speed_kmph : ℝ := 46
def kmph_to_mps (v : ℝ) : ℝ := (v * 1000) / 3600
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

theorem train_pass_time (ε : ℝ) (hε : ε = 5.01) : (train_length / train_speed_mps) ≈ ε := 
sorry -- Proof not provided

end train_pass_time_l307_307332


namespace remainder_of_3_pow_20_mod_7_l307_307322

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end remainder_of_3_pow_20_mod_7_l307_307322


namespace oranges_in_bowl_l307_307151

-- Definitions (conditions)
def bananas : Nat := 2
def apples : Nat := 2 * bananas
def total_fruits : Nat := 12

-- Theorem (proof goal)
theorem oranges_in_bowl : 
  apples + bananas + oranges = total_fruits → oranges = 6 :=
by
  intro h
  sorry

end oranges_in_bowl_l307_307151


namespace arccos_neg_half_eq_two_thirds_pi_l307_307822

theorem arccos_neg_half_eq_two_thirds_pi : 
  ∃ θ : ℝ, θ ∈ set.Icc 0 real.pi ∧ real.cos θ = -1/2 ∧ real.arccos (-1/2) = θ := 
sorry

end arccos_neg_half_eq_two_thirds_pi_l307_307822


namespace school_election_votes_l307_307573

theorem school_election_votes (E S R L : ℕ)
  (h1 : E = 2 * S)
  (h2 : E = 4 * R)
  (h3 : S = 5 * R)
  (h4 : S = 3 * L)
  (h5 : R = 16) :
  E = 64 ∧ S = 80 ∧ R = 16 ∧ L = 27 := by
  sorry

end school_election_votes_l307_307573


namespace zhang_age_in_multiple_of_9_l307_307329
open Nat 

theorem zhang_age_in_multiple_of_9 (A B : ℕ) (h1 : 1953 < 1900 + 10 * A + B)
    (h2 : 1900 + 10 * A + B - 1953 = 10 + A + B) 
    (h3 : (1 + 9 + A + B) % 9 = 0) : 
    1900 + 10 * 7 + 1 - 1953 = 1 + 9 + 7 + 1 := by
sry

end zhang_age_in_multiple_of_9_l307_307329


namespace definite_integral_example_l307_307031

theorem definite_integral_example : ∫ x in 1..2, 2 * x = 3 := by
  sorry

end definite_integral_example_l307_307031


namespace inequality_chain_l307_307464

open Real

theorem inequality_chain (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_chain_l307_307464


namespace trigonometric_identity_l307_307862

variables (α : Real) (h : sin α + cos α = 2 / 3)

theorem trigonometric_identity :
  (1 + tan α) / (2 * sin α ^ 2 + 2 * sin α * cos α) = -9 / 5 :=
by
  sorry

end trigonometric_identity_l307_307862


namespace probability_intersection_interval_l307_307967

noncomputable def P (e : Prop) : ℝ := sorry
def eventA : Prop := sorry
def eventB : Prop := sorry

def P_A : ℝ := 3 / 5
def P_B : ℝ := 4 / 5

theorem probability_intersection_interval :
  (2 / 5 : ℝ) ≤ P (eventA ∧ eventB) ∧ P (eventA ∧ eventB) ≤ (3 / 5 : ℝ) :=
by
  have hA : P eventA = P_A := sorry
  have hB : P eventB = P_B := sorry
  sorry

end probability_intersection_interval_l307_307967


namespace max_k_for_arithmetic_sum_l307_307799

theorem max_k_for_arithmetic_sum : ∃ k : ℕ, (∀ k', k' < 45 → (∃ s ⊆ (finset.range (52)).filter (λ n, (n % 2) = 1), 
    (finset.sum s id = 1949 ∧ finset.card s = k')) → k' ≤ k) ∧ 
    (∃ s ⊆ (finset.range (52)).filter (λ n, (n % 2) = 1), finset.sum s id = 1949 ∧ finset.card s = k) :=
by
  sorry

end max_k_for_arithmetic_sum_l307_307799


namespace intersection_symmetric_midpoint_l307_307740

variables {A B C D A' B' C' D' M PQ : Point}
variables {AD BC : Line}
variables {ABCD : quadrilateral}
variables {AA' BB' CC' DD' : Line}

-- Definitions of the conditions
def point_on_sides (p : Point) (q : Point) (L : Line) : Prop :=
    on_line p L ∧ on_line q L

def bisects_area (l : Line) (q : quadrilateral) : Prop :=
    divides_area l q (area q / 2)

def midpoint (M : Point) (PQ : Line) : Prop :=
    on_line M PQ ∧ divides_line M PQ (length PQ / 2)

theorem intersection_symmetric_midpoint :
    point_on_sides A' AD ∧ point_on_sides B' BC ∧ point_on_sides C' AD ∧ point_on_sides D' BC
    ∧ bisects_area AA' ABCD ∧ bisects_area BB' ABCD ∧ bisects_area CC' ABCD ∧ bisects_area DD' ABCD
    ∧ larger_base AD BC ∧ smaller_base BC AD
    ∧ midpoint M PQ →
    symmetric_intersection_diagonals ABCD A'B'C'D' M :=
    sorry

end intersection_symmetric_midpoint_l307_307740


namespace min_e1_plus_2e2_l307_307078

noncomputable def e₁ (r : ℝ) : ℝ := 2 / (4 - r)
noncomputable def e₂ (r : ℝ) : ℝ := 2 / (4 + r)

theorem min_e1_plus_2e2 (r : ℝ) (h₀ : 0 < r) (h₂ : r < 2) :
  e₁ r + 2 * e₂ r = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_e1_plus_2e2_l307_307078


namespace func_monotonic_in_interval_l307_307488

noncomputable def f (x : ℝ) : ℝ := Real.sin(2 * x + Real.pi / 6)

theorem func_monotonic_in_interval :
  ∀ x y : ℝ, (-Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 6) →
            (-Real.pi / 3 ≤ y ∧ y ≤ Real.pi / 6) →
            x < y → f x < f y :=
by
  sorry

end func_monotonic_in_interval_l307_307488


namespace valid_triangle_classification_l307_307575

-- Define the structure of a Triangle
structure Triangle :=
  (A B C : Point)
  (median_not_perpendicular_AB : ¬ Perpendicular (median_from B) AB)
  (median_not_perpendicular_BC : ¬ Perpendicular (median_from B) BC)

-- Define the function for point lying on the axis of the median from B
def on_axis_of_median_from_B (T : Triangle) (X Y : Point) : Prop :=
  on_axis_median (median_from T.B) X ∧ on_axis_median (median_from T.B) Y

-- Define the function to check if four points lie on the same circumference
def are_concyclic (A B X Y : Point) : Prop :=
  lies_on_same_circumcircle A B X Y

-- Noncomputable to define the triangles that meet the problem's requirements
noncomputable def valid_triangles (T : Triangle) (X Y : Point) : Prop :=
  on_axis_of_median_from_B T X Y ∧ are_concyclic T.A T.C X Y

-- Lean 4 statement to prove the problem
theorem valid_triangle_classification (T: Triangle)
  (X Y : Point)
  (h_on_median_axis : on_axis_of_median_from_B T X Y)
  (h_concyclic : are_concyclic T.A T.C X Y):
  (isosceles T.B T.A T.C) ∨ (right_angled_at T.B):=
sorry

end valid_triangle_classification_l307_307575


namespace unique_face_numbers_l307_307217

-- Define the problem statement and conditions
theorem unique_face_numbers (a b c d e f : ℤ) (sums : list ℤ) (h : sums = [a + b, a + c, a + d, a + e, b + c, b + f, c + f, d + f, d + e, e + f, b + d, c + e]) : 
  (∃ (n : ℕ → ℤ), (n 0 = a ∧ n 1 = b ∧ n 2 = c ∧ n 3 = d ∧ n 4 = e ∧ n 5 = f)) :=
by 
  rw h
  -- Additional detailed steps are omitted
  sorry

end unique_face_numbers_l307_307217


namespace minimum_pairs_of_cities_l307_307567

/-- In the country of Alfya, there are 150 cities, and any four cities can be divided 
    into two pairs such that there is an express train running between the cities of each pair. 
    We need to prove that the minimum number of pairs of cities connected by express trains is 11025. -/
theorem minimum_pairs_of_cities : 
  ∀ (V : Type) [Fintype V], 
  Fintype.card V = 150 →
  (∀ (A B C D : V), ∃ (X Y : set (V × V)), {A, B, C, D}.pairwise (λ x y, (x, y) ∈ X ∪ Y)) →
  ∃ (E : set (V × V)), 
  (∀ (u : V), Fintype.card {v : V | (u, v) ∈ E ∨ (v, u) ∈ E} ≥ 147) ∧ 
  Fintype.card E = 11025 :=
µ sorry

end minimum_pairs_of_cities_l307_307567


namespace probability_single_trial_l307_307558

theorem probability_single_trial (p : ℚ) (h₁ : (1 - p)^4 = 16 / 81) : p = 1 / 3 :=
sorry

end probability_single_trial_l307_307558


namespace president_vice_president_count_l307_307768

/-- The club consists of 24 members, split evenly with 12 boys and 12 girls. 
    There are also two classes, each containing 6 boys and 6 girls. 
    Prove that the number of ways to choose a president and a vice-president 
    if they must be of the same gender and from different classes is 144. -/
theorem president_vice_president_count :
  ∃ n : ℕ, n = 144 ∧ 
  (∀ (club : Finset ℕ) (boys girls : Finset ℕ) 
     (class1_boys class1_girls class2_boys class2_girls : Finset ℕ),
     club.card = 24 →
     boys.card = 12 → girls.card = 12 →
     class1_boys.card = 6 → class1_girls.card = 6 →
     class2_boys.card = 6 → class2_girls.card = 6 →
     (∃ president vice_president : ℕ,
     president ∈ club ∧ vice_president ∈ club ∧
     ((president ∈ boys ∧ vice_president ∈ boys) ∨ 
      (president ∈ girls ∧ vice_president ∈ girls)) ∧
     ((president ∈ class1_boys ∧ vice_president ∈ class2_boys) ∨
      (president ∈ class2_boys ∧ vice_president ∈ class1_boys) ∨
      (president ∈ class1_girls ∧ vice_president ∈ class2_girls) ∨
      (president ∈ class2_girls ∧ vice_president ∈ class1_girls)) →
     n = 144)) :=
by
  sorry

end president_vice_president_count_l307_307768


namespace factorization_squared_sums_compare_expressions_l307_307992

-- (1) Factorizing using completing the square method
theorem factorization (a : ℝ) : a^2 - 6 * a + 8 = (a - 4) * (a - 2) :=
sorry

-- (2) Finding a² + b² and a⁴ + b⁴ given a + b = 5 and ab = 6
theorem squared_sums (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 6) :
  a^2 + b^2 = 13 ∧ a^4 + b^4 = 97 :=
sorry

-- (3) Comparing x² - 4x + 5 with -x² + 4x -4
theorem compare_expressions (x : ℝ) : x^2 - 4 * x + 5 > -x^2 + 4 * x - 4 :=
sorry

end factorization_squared_sums_compare_expressions_l307_307992


namespace sampling_method_is_systematic_l307_307772

-- Define the conditions of the problem
def conveyor_belt_transport : Prop := true
def inspectors_sampling_every_ten_minutes : Prop := true

-- Define what needs to be proved
theorem sampling_method_is_systematic :
  conveyor_belt_transport ∧ inspectors_sampling_every_ten_minutes → is_systematic_sampling :=
by
  sorry

-- Example definition that could be used in the proof
def is_systematic_sampling : Prop := true

end sampling_method_is_systematic_l307_307772


namespace arccos_neg_half_l307_307818

-- Defining the problem in Lean 4
theorem arccos_neg_half : 
  ∃ θ ∈ set.Icc 0 Real.pi, Real.arccos (-1 / 2) = θ ∧ Real.cos θ = -1 / 2 := 
by
  use Real.pi * 2 / 3
  split
  { sorry } -- Proof that θ is in [0, π]
  { split
    { sorry } -- Proof that θ = arccos(-1 / 2)
    { sorry } -- Proof that cos(θ) = -1/2
  }


end arccos_neg_half_l307_307818


namespace system_of_equations_exactly_two_solutions_l307_307419

theorem system_of_equations_exactly_two_solutions (a : ℝ) :
  (∀ x y : ℝ,
    (a^2 - 2*a*x + 10*y + x^2 + y^2 = 0) ↔
    ((|x| - 12)^2 + (|y| - 5)^2 = 169)) ↔
    (a ∈ set.Icc (-30) (-20) ∪ {0} ∪ set.Icc 20 30) :=
sorry

end system_of_equations_exactly_two_solutions_l307_307419


namespace sum_telescoping_series_l307_307003

theorem sum_telescoping_series :
  (\sum n in (finset.range 498).map (finset.nat.cast_add 3), 1 / (n * nroot (n - 2) 3 + (n - 2) * nroot n 3)) 
  = (1 / 2) * (1 - 1 / nroot 500 3) :=
sorry

end sum_telescoping_series_l307_307003


namespace find_n_l307_307193

theorem find_n (x y n : ℤ) (hx : x = 3) (hy : y = -1)
  (hn : n = x - y^(x - 2 * y) + 2 * x) : n = 10 := by 
  sorry

end find_n_l307_307193


namespace sums_have_same_remainder_l307_307798

theorem sums_have_same_remainder (n : ℕ) (a : Fin (2 * n) → ℕ) : 
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i.val) % (2 * n) = (a j + j.val) % (2 * n)) := 
sorry

end sums_have_same_remainder_l307_307798


namespace john_weekly_earnings_l307_307178

theorem john_weekly_earnings :
  (4 * 4 * 10 = 160) :=
by
  -- Proposition: John makes $160 a week from streaming
  -- Condition 1: John streams for 4 days a week
  let days_of_streaming := 4
  -- Condition 2: He streams 4 hours each day.
  let hours_per_day := 4
  -- Condition 3: He makes $10 an hour.
  let earnings_per_hour := 10

  -- Now, calculate the weekly earnings
  -- Weekly earnings = 4 days/week * 4 hours/day * $10/hour
  have weekly_earnings : days_of_streaming * hours_per_day * earnings_per_hour = 160 := sorry
  exact weekly_earnings


end john_weekly_earnings_l307_307178


namespace additional_flowers_grew_l307_307018

-- Define the initial conditions
def initial_flowers : ℕ := 10  -- Dane’s two daughters planted 5 flowers each (5 + 5).
def flowers_died : ℕ := 10     -- 10 flowers died.
def baskets : ℕ := 5
def flowers_per_basket : ℕ := 4

-- Total flowers harvested (from the baskets)
def total_harvested : ℕ := baskets * flowers_per_basket  -- 5 * 4 = 20

-- The proof to show additional flowers grown
theorem additional_flowers_grew : (total_harvested - initial_flowers + flowers_died) = 10 :=
by
  -- The final number of flowers and the initial number of flowers are known
  have final_flowers : ℕ := total_harvested
  have initial_plus_grown : ℕ := initial_flowers + (total_harvested - initial_flowers)
  -- Show the equality that defines the additional flowers grown
  show (total_harvested - initial_flowers + flowers_died) = 10
  sorry

end additional_flowers_grew_l307_307018


namespace mike_total_hours_l307_307622

-- Define the number of hours Mike worked each day.
def hours_per_day : ℕ := 3

-- Define the number of days Mike worked.
def days : ℕ := 5

-- Define the total number of hours Mike worked.
def total_hours : ℕ := hours_per_day * days

-- State and prove that the total hours Mike worked is 15.
theorem mike_total_hours : total_hours = 15 := by
  -- Proof goes here
  sorry

end mike_total_hours_l307_307622


namespace celine_change_l307_307363

theorem celine_change :
  let laptop_price := 600
  let smartphone_price := 400
  let tablet_price := 250
  let headphone_price := 100
  let laptops_purchased := 2
  let smartphones_purchased := 4
  let tablets_purchased := 3
  let headphones_purchased := 5
  let discount_rate := 0.10
  let sales_tax_rate := 0.05
  let initial_amount := 5000
  let laptop_total := laptops_purchased * laptop_price
  let smartphone_total := smartphones_purchased * smartphone_price
  let tablet_total := tablets_purchased * tablet_price
  let headphone_total := headphones_purchased * headphone_price
  let discount := discount_rate * (laptop_total + tablet_total)
  let total_before_discount := laptop_total + smartphone_total + tablet_total + headphone_total
  let total_after_discount := total_before_discount - discount
  let sales_tax := sales_tax_rate * total_after_discount
  let final_price := total_after_discount + sales_tax
  let change := initial_amount - final_price
  change = 952.25 :=
  sorry

end celine_change_l307_307363


namespace find_a_l307_307968

def f (x a : ℝ) := |x + 1| + |x - a|

theorem find_a (a : ℝ) : (∀ x, f x a ≥ 5 ↔ x ∈ Iic (-2) ∨ x ∈ Ioi 3) → a = 2 :=
by
  rw [f]
  sorry

end find_a_l307_307968


namespace length_block_correct_l307_307339

-- Define the box dimensions
def height_box : ℕ := 8
def width_box : ℕ := 10
def length_box : ℕ := 12

-- Define the building block dimensions
def height_block : ℕ := 3
def width_block : ℕ := 2

-- Number of building blocks that fit into the box
def number_blocks : ℕ := 40

-- Goal: Prove that the length of the building block is 4 inches
theorem length_block_correct : 
  let volume_box := height_box * width_box * length_box in
  let volume_block := volume_box / number_blocks in
  let length_block := volume_block / (height_block * width_block) in
  length_block = 4 := 
by
  sorry

end length_block_correct_l307_307339


namespace total_distance_traveled_l307_307833

-- Definitions of distances in km
def ZX : ℝ := 4000
def XY : ℝ := 5000
def YZ : ℝ := (XY^2 - ZX^2)^(1/2)

-- Prove the total distance traveled
theorem total_distance_traveled :
  XY + YZ + ZX = 11500 := by
  have h1 : ZX = 4000 := rfl
  have h2 : XY = 5000 := rfl
  have h3 : YZ = (5000^2 - 4000^2)^(1/2) := rfl
  -- Continue the proof showing the calculation of each step
  sorry

end total_distance_traveled_l307_307833


namespace parabola_vertex_coordinates_l307_307645

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = -(x - 1) ^ 2 + 3 → (1, 3) = (1, 3) :=
by
  intros x y h
  sorry

end parabola_vertex_coordinates_l307_307645
