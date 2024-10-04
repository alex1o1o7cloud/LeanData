import Mathlib
import Mathlib.Algebra.Arithmetic.Sequence
import Mathlib.Algebra.Finset.Pi
import Mathlib.Algebra.Functions
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.StarsAndBars
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Polynomial
import Mathlib.Data.ProbTheory
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution.Normal
import Mathlib.Probability.Distributions.Normal
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.ProbabilitySpace
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology
import Mathlib.Topology.EuclideanMetricSpace
import Mathlib.Topology.MetricSpace.Basic

namespace tic_tac_toe_tie_fraction_l778_778506

theorem tic_tac_toe_tie_fraction :
  let amys_win : ℚ := 5 / 12
  let lilys_win : ℚ := 1 / 4
  1 - (amys_win + lilys_win) = 1 / 3 :=
by
  sorry

end tic_tac_toe_tie_fraction_l778_778506


namespace relation_between_M_and_N_l778_778360

open Set

def U : Set ℝ := univ

def M : Set ℝ := { x : ℝ | x^2 - 2 * x > 0 }

def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = log (x - 2) }

theorem relation_between_M_and_N : M ⊇ N := 
by 
  sorry

end relation_between_M_and_N_l778_778360


namespace domain_p_l778_778021

def h : ℝ → ℝ := sorry -- assume some function definition

noncomputable def p (x : ℝ) := h (-3 * x + 1)

theorem domain_p :
  (∀ x, -10 ≤ -3 * x + 1 ∧ -3 * x + 1 ≤ 6 ↔ -5/3 ≤ x ∧ x ≤ 11/3) :=
by
  intro x
  constructor
  {
    intro h_cond
    cases h_cond with left right
    split
    {
      have : -11 ≤ -3 * x := by
        linarith [left]
      linarith
    }
    {
      have : -3 * x ≤ 5 := by
        linarith [right]
      linarith
    }
  }
  {
    intro h_cond
    cases h_cond with left right
    split
    {
      have : -11 ≤ -3 * x := by
        linarith [left]
      linarith
    }
    {
      have : -3 * x ≤ 5 := by
        linarith [right]
      linarith
    }
  }

end domain_p_l778_778021


namespace meal_tables_needed_l778_778116

theorem meal_tables_needed : ceil (156 / 8) ≤ 20 := sorry

end meal_tables_needed_l778_778116


namespace function_characterization_l778_778634

theorem function_characterization (f : ℤ → ℤ) :
  (∀ m n : ℤ, f(m + f(f(n))) = -f(f(m + 1)) - n) ↔ (∀ n : ℤ, f(n) = -n - 1) :=
by
  sorry

end function_characterization_l778_778634


namespace ellipse_equation_fixed_point_l778_778274

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778274


namespace total_lemons_l778_778805

theorem total_lemons (x : ℕ) : 
  let Levi := x,
      Jayden := x + 6,
      Alexandra := (4/3) * (x + 6 : ℝ),
      Eli := (2/3) * (x + 6 : ℝ),
      Ian := (4/3) * (x + 6 : ℝ),
      Nathan := (3/4) * (x : ℝ),
      Olivia := (4/5) * (x + 6 : ℝ)
  in Levi + Jayden + Alexandra + Eli + Ian + Nathan + Olivia = (413/60) * (x : ℝ) + 30.8 := by sorry

end total_lemons_l778_778805


namespace parabola_intercepts_sum_l778_778843

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end parabola_intercepts_sum_l778_778843


namespace inequality_solution_system_l778_778465

variable (x : ℝ)

theorem inequality_solution_system :
  (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 3 ≥ x) ↔ (-2 < x ∧ x ≤ 3) :=
by
  intro h
  change sorry with the proof. -- proof is not provided here

end inequality_solution_system_l778_778465


namespace num_sequences_with_perfect_square_l778_778162

theorem num_sequences_with_perfect_square : 
  let sequences_with_perfect_square_count := 
    (λ a d, ∃ n : ℕ, (a + n * d)^2 ∈ {1,4,9,16,25,36}) in
  let count := 
    List.sum (List.map (λ a, 
      List.sum (List.map (λ d, if sequences_with_perfect_square_count a d then 1 else 0) (List.range 1 7))
    ) (List.range 1 7)) in
  count = 27 :=
by
  sorry

end num_sequences_with_perfect_square_l778_778162


namespace sum_of_auspicious_numbers_l778_778145

open Int

def is_auspicious (n : ℕ) : Prop :=
  ∀ (x y : ℕ), (x * (10^(log10 n + 1)) + n) * (y * (10^(log10 n + 1)) + n) % (10^(log10 n + 1)) = n

def list_of_auspicious_numbers : List ℕ := [1, 5, 6, 25, 76, 625, 376]

theorem sum_of_auspicious_numbers : list_of_auspicious_numbers.sum = 1114 := by
  intros
  sorry

end sum_of_auspicious_numbers_l778_778145


namespace sum_of_possible_values_of_x_l778_778491

theorem sum_of_possible_values_of_x (x : ℝ) :
  let square_side := x - 3 in
  let rectangle_length := x - 4 in
  let rectangle_width := x + 5 in
  let square_area := square_side^2 in
  let rectangle_area := rectangle_length * rectangle_width in
  (rectangle_area = 3 * square_area) →
  (x^2 - 7*x + 3.5 = 0) →
  (∀ p q : ℝ, p * q = 3.5 ∧ p + q = 7 → 
    let roots_sum := p + q in roots_sum = 7) :=
by {
  sorry
}

end sum_of_possible_values_of_x_l778_778491


namespace distance_triangle_four_points_l778_778242

variable {X : Type*} [MetricSpace X]

theorem distance_triangle_four_points (A B C D : X) :
  dist A D ≤ dist A B + dist B C + dist C D :=
by
  sorry

end distance_triangle_four_points_l778_778242


namespace IMO1987Q28_l778_778683

theorem IMO1987Q28 (n : ℕ) (h_n : 2 ≤ n)
    (h_prime : ∀ k : ℕ, 0 ≤ k ∧ k ≤ nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
    ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := 
by
  sorry

end IMO1987Q28_l778_778683


namespace gcd_of_repeated_three_digit_integers_l778_778589

theorem gcd_of_repeated_three_digit_integers :
  ∀ m : ℕ, (100 ≤ m ∧ m < 1000) → ∃ d : ℕ, (d = gcd (1001 * m) (1001 * m.next)) ∧ d = 1001 :=
by
  intro m
  intro h
  use 1001
  sorry

end gcd_of_repeated_three_digit_integers_l778_778589


namespace inequality_problem_l778_778702

theorem inequality_problem (a b c : ℝ) (h : a < b ∧ b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  -- The proof is supposed to be here
  sorry

end inequality_problem_l778_778702


namespace power_computation_l778_778538

theorem power_computation : (12 ^ (12 / 2)) = 2985984 := by
  sorry

end power_computation_l778_778538


namespace plane_speeds_l778_778507

theorem plane_speeds (v : ℕ) 
    (h1 : ∀ (t : ℕ), t = 5 → 20 * v = 4800): 
  v = 240 ∧ 3 * v = 720 := by
  sorry

end plane_speeds_l778_778507


namespace ellipse_and_fixed_point_l778_778303

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778303


namespace parabola_intercept_sum_l778_778847

theorem parabola_intercept_sum (a b c : ℝ) : 
  (∃ y : ℝ, a = 3 * y^2 - 9 * y + 5) ∧ (∀ x : ℝ, x = 0 → b ≠ c → 3 * b^2 - 9 * b + 5 = 0 ∧ 3 * c^2 - 9 * c + 5 = 0 ∧ b + c = 3) → 
  a + b + c = 8 :=
begin
  sorry
end

end parabola_intercept_sum_l778_778847


namespace hn_passes_fixed_point_l778_778263

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778263


namespace inequality_product_sum_l778_778956

theorem inequality_product_sum (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i ∧ x i < 1) :
  2 ≤ (∏ i, (1 + x i)) + (∏ i, (1 - x i)) ∧ (∏ i, (1 + x i)) + (∏ i, (1 - x i)) ≤ 2^n :=
sorry

end inequality_product_sum_l778_778956


namespace evaluate_g_l778_778188

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_l778_778188


namespace ellipse_and_fixed_point_l778_778295

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778295


namespace maximize_tetrahedron_volume_l778_778955

noncomputable def volume_maximized_ob_length : ℝ :=
  let PA := 4 in
  let PC := PA / 2 in
  let OC := PC in
  let HC := OC in
  let OP := OC in
  OP * real.tan (real.pi / 6)

theorem maximize_tetrahedron_volume :
  let OB := volume_maximized_ob_length
  in OB = 2 * real.sqrt 6 / 3 :=
begin
  sorry
end

end maximize_tetrahedron_volume_l778_778955


namespace valid_triangle_count_l778_778367

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_not_right (a b c : ℕ) : Prop :=
  a^2 + b^2 ≠ c^2 ∧ a^2 + c^2 ≠ b^2 ∧ b^2 + c^2 ≠ a^2

def is_distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c

noncomputable def count_valid_triangles (perimeter_bound : ℕ) : ℕ :=
  (Σ' a b c : ℕ, a + b + c < perimeter_bound ∧ is_triangle a b c ∧ is_not_right a b c ∧ is_distinct a b c).toFinset.card

theorem valid_triangle_count : count_valid_triangles 20 = 9 := 
by 
  -- The full proof would be placed here.
  sorry

end valid_triangle_count_l778_778367


namespace equivalent_after_eliminating_denominators_l778_778091

theorem equivalent_after_eliminating_denominators (x : ℝ) (h : 1 + 2 / (x - 1) = (x - 5) / (x - 3)) :
  (x - 1) * (x - 3) + 2 * (x - 3) = (x - 5) * (x - 1) :=
sorry

end equivalent_after_eliminating_denominators_l778_778091


namespace handshakes_total_l778_778606

-- Definition of the scenario with Group A and Group B
def groupA_size : ℕ := 20
def groupB_size : ℕ := 10
def total_people : ℕ := 30
def handshakes_between_groups := groupB_size * groupA_size
def handshakes_within_groupB := groupB_size.choose 2

theorem handshakes_total : handshakes_between_groups + handshakes_within_groupB = 245 := by
  have h1 : handshakes_between_groups = 200 := by
    sorry
  have h2 : handshakes_within_groupB = 45 := by
    sorry
  rw [h1, h2]
  norm_num

end handshakes_total_l778_778606


namespace eventually_periodic_l778_778948

theorem eventually_periodic (n : ℕ) (h : n > 0) (s : ℕ → (fin 2)) 
  (h_blocks : ∀ nblock1 nblock2 : fin 2 ^ n, (∃ i j, nblock1 = vector.of_fn (λ k, s (i + k)) ∧ nblock2 = vector.of_fn (λ k, s (j + k)) → i = j)) : 
  ∃ p q : ℕ, p < q ∧ (∀ i, s (p + i) = s (q + i)) :=
sorry

end eventually_periodic_l778_778948


namespace evaluate_g_expression_l778_778190

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g_expression :
  3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_expression_l778_778190


namespace probability_correct_l778_778523

open Finset

def standard_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

noncomputable def probability_abc_144 : ℚ :=
  let outcomes := (standard_die × standard_die × standard_die).filter (λ (t : ℕ × ℕ × ℕ), t.1 * t.2 * t.3 = 144)
  1 / 6 * 1 / 6 * 1 / 6 * outcomes.card

theorem probability_correct : probability_abc_144 = 1 / 72 := by
  unfold probability_abc_144
  sorry

end probability_correct_l778_778523


namespace complex_number_solution_l778_778682

-- Conditions
variable (z : ℂ)
def imaginary_unit := Complex.i

-- Question and Answer
theorem complex_number_solution (h : z / (z - imaginary_unit) = imaginary_unit) : z = (1 / 2 + 1 / 2 * imaginary_unit) :=
sorry

end complex_number_solution_l778_778682


namespace analytical_expression_of_f_intervals_of_monotonic_increase_range_of_m_l778_778245

-- Definitions for the given problem
def function_f (x : ℝ) : ℝ := 2 * Real.sin (3 * x - Real.pi / 3)

theorem analytical_expression_of_f :
  function_f = (λ x, 2 * Real.sin (3 * x - Real.pi / 3)) :=
sorry

theorem intervals_of_monotonic_increase :
  ∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) →
  (x ∈ [0, 5 * Real.pi / 18] ∨ x ∈ [11 * Real.pi / 18, 2 * Real.pi / 3]) →
  (StrictMonoOn function_f [0, 5 * Real.pi / 18] ∨
   StrictMonoOn function_f [11 * Real.pi / 18, 2 * Real.pi / 3]) :=
sorry

theorem range_of_m (m : ℝ) :
  ∀ x, (0 ≤ x ∧ x ≤ Real.pi / 6) →
  (m * function_f x + 2 * m ≥ function_f x) ↔ (m ≥ 1 / 3) :=
sorry

end analytical_expression_of_f_intervals_of_monotonic_increase_range_of_m_l778_778245


namespace problem_l778_778324

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778324


namespace all_stars_seating_fixed_order_l778_778392

theorem all_stars_seating_fixed_order :
  let total_blocks : Nat := 4,
      cubs : Nat := 3,
      red_sox : Nat := 3,
      yankees : Nat := 2,
      dodger : Nat := 1 in
  Nat.factorial total_blocks * Nat.factorial cubs * Nat.factorial red_sox * Nat.factorial yankees * Nat.factorial dodger = 1728 :=
by
  simp only [Nat.factorial]
  simp
  sorry

end all_stars_seating_fixed_order_l778_778392


namespace mixed_groups_count_l778_778891

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l778_778891


namespace tan_a6_a7_a8_is_neg1_l778_778339

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range n, a i

-- Given conditions
axiom sum_first_13_terms (h_arith : is_arithmetic_sequence a) : 
  sum_of_first_n_terms S a → S 13 = 13 * (a 6)

axiom middle_term_value (h_arith : is_arithmetic_sequence a) : 
  sum_of_first_n_terms S a → S 13 = (13 * π) / 4 → a 6 = π / 4

-- Mathematical proof problem
theorem tan_a6_a7_a8_is_neg1 (h_arith : is_arithmetic_sequence a) 
  (h_sum : sum_of_first_n_terms S a) (h_sum_value : S 13 = (13 * π) / 4) :
  Real.tan (a 5 + a 6 + a 7) = -1 :=
by sorry

end tan_a6_a7_a8_is_neg1_l778_778339


namespace Brad_has_9_green_balloons_l778_778111

theorem Brad_has_9_green_balloons
  (total_balloons : ℕ)
  (red_balloons : ℕ)
  (green_balloons : ℕ)
  (h1 : total_balloons = 17)
  (h2 : red_balloons = 8)
  (h3 : total_balloons = red_balloons + green_balloons) :
  green_balloons = 9 := 
sorry

end Brad_has_9_green_balloons_l778_778111


namespace part_a_part_b_l778_778532

-- Part (a)
theorem part_a (a : ℕ) (h_odd : a % 2 = 1) (m : ℕ) :
  ∃ (infinitely_many_k : ℕ → ℕ), ∀ n : ℕ, 2^m ∣ (a^(infinitely_many_k n) - 1) :=
sorry

-- Part (b)
theorem part_b (a : ℕ) (h_odd : a % 2 = 1) :
  ∃ (finite_m_set : set ℕ), ∀ m : ℕ, (2^m ∣ (a^m - 1)) → m ∈ finite_m_set :=
sorry

end part_a_part_b_l778_778532


namespace find_quadratic_polynomial_with_root_l778_778661

noncomputable def quadratic_polynomial (a b c : ℝ) : ℝ := 3 * a^2 - 30 * b + 87

theorem find_quadratic_polynomial_with_root (x : ℂ) (h₁ : x = 5 + 2 * complex.I) 
        (h₂ : x.conj = 5 - 2 * complex.I) : 
        quadratic_polynomial x.re x.im (3) = 3 * (x^2).re - 30 * x.re + 87  :=
by
  -- Proof goes here
  sorry

end find_quadratic_polynomial_with_root_l778_778661


namespace segment_midpoints_and_distance_l778_778389

noncomputable def segments_and_points (AB A'B' : ℝ) (D'_ratio A'B'_value AB_value x : ℝ) (D F: ℝ) := 
  let D := AB / 2
  let D' := D'_ratio * A'B'
  let x := x
  let y := (A'B' / AB) * x -- since the mapping ratio is A'B'/AB
  x + y = 3 * x

theorem segment_midpoints_and_distance (a : ℝ) :
  segments_and_points 3 6 (1/3) 6 3 a = 3 * a := 
  sorry

end segment_midpoints_and_distance_l778_778389


namespace ellipse_equation_fixed_point_l778_778276

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778276


namespace HN_passes_through_fixed_point_l778_778259

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778259


namespace problem_l778_778327

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778327


namespace man_walking_speed_l778_778972

def distance_when_running : ℝ := 15 * (36 / 60)
def distance_when_walking (time_walking : ℝ) : ℝ := distance_when_running

-- Define the man's walking speed
def walking_speed (distance : ℝ) (time_walking : ℝ) : ℝ := distance / time_walking

-- The Lean statement to prove
theorem man_walking_speed (time_walking : ℝ) (time_running_hr : ℝ := 36 / 60) :
  walking_speed distance_when_running time_walking = 1.8 := by
  sorry

end man_walking_speed_l778_778972


namespace max_profit_max_profit_price_l778_778149

-- Definitions based on the conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 120
def initial_sales : ℕ := 20
def extra_sales_per_unit_decrease : ℕ := 2
def cost_price_constraint (x : ℝ) : Prop := 0 < x ∧ x ≤ 40

-- Expression for the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Prove the maximum profit given the conditions
theorem max_profit : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 :=
by
  sorry

-- Proving that the selling price for max profit is 105 yuan
theorem max_profit_price : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 ∧ (initial_selling_price - x) = 105 :=
by
  sorry

end max_profit_max_profit_price_l778_778149


namespace popularity_order_l778_778049

def chess_popularity := 5 / 16
def drama_popularity := 7 / 24
def music_popularity := 11 / 32
def art_popularity := 13 / 48

theorem popularity_order :
  (31 / 96 < 34 / 96) ∧ (34 / 96 < 35 / 96) ∧ (35 / 96 < 36 / 96) ∧ 
  (chess_popularity < music_popularity) ∧ 
  (drama_popularity < music_popularity) ∧ 
  (music_popularity > art_popularity) ∧ 
  (chess_popularity > drama_popularity) ∧ 
  (drama_popularity > art_popularity) := 
sorry

end popularity_order_l778_778049


namespace compute_DP_l778_778430

variable (A B C : Point)
variable (AB AC BC : Length)
variable (I D E F X P : Point)
variable (r : Length)
variable (line_XI : Line)
variable (circumcircle : Circle)
variable (incircle : Circle)

-- Given conditions
axiom AB_length : AB = 14
axiom AC_length : AC = 15
axiom BC_length : BC = 13
axiom incircle_center_is_I : incircle.center = I
axiom incircle_tangents : incircle.isTangentAt BC D ∧ incircle.isTangentAt CA E ∧ incircle.isTangentAt AB F
axiom X_midpoint_major_arc : X = circumcircle.majorArcMidpoint BAC
axiom P_on_line_XI : P ∈ line_XI
axiom DP_perpendicular_EF : perpendicular DP EF

-- Define the theorem to prove DP length
theorem compute_DP : DP = (4 * real.sqrt 5) / 5 := by 
  sorry

end compute_DP_l778_778430


namespace set_A_membership_l778_778106

theorem set_A_membership (U : Finset ℕ) (A : Finset ℕ) (B : Finset ℕ)
  (hU : U.card = 193)
  (hB : B.card = 49)
  (hneither : (U \ (A ∪ B)).card = 59)
  (hAandB : (A ∩ B).card = 25) :
  A.card = 110 := sorry

end set_A_membership_l778_778106


namespace proof_problem_l778_778556

-- Definitions for the conditions and the events in the problem
def P_A : ℚ := 2 / 3
def P_B : ℚ := 1 / 4
def P_not_any_module : ℚ := 1 - (P_A + P_B)

-- Definition for the binomial coefficient
def C (n k : ℕ) := Nat.choose n k

-- Definition for the event where at least 3 out of 4 students have taken "Selected Topics in Geometric Proofs"
def P_at_least_three_taken : ℚ := 
  C 4 3 * (P_A ^ 3) * ((1 - P_A) ^ 1) + C 4 4 * (P_A ^ 4)

-- The main theorem to prove
theorem proof_problem : 
  P_not_any_module = 1 / 12 ∧ P_at_least_three_taken = 16 / 27 :=
by
  sorry

end proof_problem_l778_778556


namespace prime_divisor_of_sum_of_four_consecutive_integers_l778_778915

theorem prime_divisor_of_sum_of_four_consecutive_integers : 
  ∃ p : ℕ, Prime p ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) ∧ p = 2 := 
by
  sorry

end prime_divisor_of_sum_of_four_consecutive_integers_l778_778915


namespace sum_of_roots_of_quadratic_l778_778483

theorem sum_of_roots_of_quadratic :
  let f := λ x : ℝ, 3 * x^2 + 5 * x + 2 in
  let roots := {x : ℝ | f x = 0} in
  ∀ a b : ℝ, a ∈ roots → b ∈ roots → a ≠ b → a + b = -5/3 :=
by 
  let a := (-5:ℝ)/(3:ℝ) * 2 in
  let b := - (1 : ℝ) in
  sorry

end sum_of_roots_of_quadratic_l778_778483


namespace parabola_intercepts_sum_l778_778846

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end parabola_intercepts_sum_l778_778846


namespace sum_arithmetic_sequence_S_101_zero_l778_778050

-- Define parameters
variables (a_1 S_101 : ℤ)
variables (S : ℕ → ℤ)
variables (n : ℕ) (d : ℤ)

-- Given conditions
def condition_1 := a_1 = -100
def condition_2 := 5 * S 1 - 7 * S 5 = 70
def S_formula (n : ℕ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- Statement to prove
theorem sum_arithmetic_sequence_S_101_zero :
  condition_1 → condition_2 → (∀ n, S n = S_formula n) → S 101 = 0 :=
by
  intros h1 h2 h3
  sorry

end sum_arithmetic_sequence_S_101_zero_l778_778050


namespace first_year_exceeds_threshold_l778_778965

def P (n : ℕ) : ℝ := 40000 * (1 + 0.2) ^ n
def exceeds_threshold (n : ℕ) : Prop := P n > 120000

theorem first_year_exceeds_threshold : ∃ n : ℕ, exceeds_threshold n ∧ 2013 + n = 2020 := 
by
  sorry

end first_year_exceeds_threshold_l778_778965


namespace solution_b_2023_l778_778193

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if n = 2 then 4 / 9
  else 2 * seq (n - 2) * seq (n - 1) / (3 * seq (n - 2) - 2 * seq (n - 1))

theorem solution_b_2023 :
  let p := 8 in
  let q := 8092 in
  seq 2023 = p / q ∧ Nat.coprime p q ∧ p + q = 8100 :=
begin
  sorry,
end

end solution_b_2023_l778_778193


namespace x_pairs_cond_l778_778223

variable (n : ℕ)
variable (x : Fin n → ℤ)

-- Define the condition that x_i is either +1 or -1
def isPlusOrMinusOne (x : ℤ) : Prop :=
  x = 1 ∨ x = -1

-- The main hypothesis
def mainHypothesis (x : Fin n → ℤ) : Prop :=
  (∑ i in Finset.range n, x i * x ((i + 1) % n)) = (0 : ℤ)

-- One hypothesis is that each x_i is either +1 or -1
def allElementsArePlusOrMinusOne (x: Fin n → ℤ) : Prop :=
  ∀ i : Fin n, isPlusOrMinusOne (x i)

theorem x_pairs_cond (n : ℕ) (x : Fin n → ℤ) 
  (hyp1 : allElementsArePlusOrMinusOne x)
  (hyp2 : mainHypothesis x) : 
  ∃ k : ℕ, n = 4 * k := 
sorry

end x_pairs_cond_l778_778223


namespace digital_clock_even_digits_plim_sounds_in_one_day_l778_778098

-- Problem (a): Digital Clock with all even digits
theorem digital_clock_even_digits :
  ∃ (count : ℕ), count = 105 ∧
  ∀ (h1 h2 m1 m2 : ℕ), 
    (h1 = 0 ∨ h1 = 1 ∨ h1 = 2) →
    (h1 = 2 → h2 < 4) →
    (m1 < 6) →
    (even h1 ∧ even h2 ∧ even m1 ∧ even m2 → (h1 = 0 ∨ h1 = 1 ∨ h1 = 2) ∧ (h2 = 0 ∨ h2 = 2 ∨ h2 = 4 ∨ h2 = 6 ∨ h2 = 8) ∧ (m1 = 0 ∨ m1 = 2 ∨ m1 = 4) ∧ (m2 = 0 ∨ m2 = 2 ∨ m2 = 4 ∨ m2 = 6 ∨ m2 = 8) →
    count = 105) :=
begin
  sorry
end

-- Problem (b): Plim sounds in one day
theorem plim_sounds_in_one_day :
  ∃ (count : ℕ), count = 1438 ∧
  ∀ (p q : ℕ), 
    (hour_second_passes_per_day = 719) →
    (minute_second_passes_per_day = 708) →
    (hour_minute_passes_per_day = 11) →
    (p = 719 ∧ q = 1438 ∧ r = 11 → count = p + q + r → 
    count = 1438) :=
begin
  sorry
end

end digital_clock_even_digits_plim_sounds_in_one_day_l778_778098


namespace square_area_from_circle_area_l778_778592

variable (square_area : ℝ) (circle_area : ℝ)

theorem square_area_from_circle_area 
  (h1 : circle_area = 9 * Real.pi) 
  (h2 : square_area = (2 * Real.sqrt (circle_area / Real.pi))^2) : 
  square_area = 36 := 
by
  sorry

end square_area_from_circle_area_l778_778592


namespace possible_new_perimeter_l778_778940

-- Definitions of the problem conditions
def original_tiles : ℕ := 10
def original_perimeter : ℕ := 16
def new_tiles : ℕ := 2

-- A tile is represented as a 1x1 unit square
structure tile :=
  (x : ℕ)  -- x-coordinate
  (y : ℕ)  -- y-coordinate

-- Predicate indicating that two tiles share at least one side
def share_side (t1 t2 : tile) : Prop :=
  (t1.x = t2.x ∧ (t1.y = t2.y + 1 ∨ t1.y = t2.y - 1)) ∨
  (t1.y = t2.y ∧ (t1.x = t2.x + 1 ∨ t1.x = t2.x - 1))

-- Main theorem stating the new perimeter can be either 23 or 25
theorem possible_new_perimeter : 
  ∀ (original_shape : list tile) (new_tile1 new_tile2 : tile),
  length original_shape = original_tiles →
  original_perimeter = 16 →
  share_side new_tile1 (original_shape.head) →
  share_side new_tile2 (original_shape.head) →
  (new_perimeter original_shape [new_tile1, new_tile2] = 23 ∨ new_perimeter original_shape [new_tile1, new_tile2] = 25) :=
sorry

end possible_new_perimeter_l778_778940


namespace roots_magnitudes_less_than_one_l778_778706

theorem roots_magnitudes_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + A * r + B = 0))
  (h2 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + C * r + D = 0)) :
  ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + (1 / 2 * (A + C)) * r + (1 / 2 * (B + D)) = 0) :=
by
  sorry

end roots_magnitudes_less_than_one_l778_778706


namespace geometric_sequence_nth_term_divisible_by_10_8_l778_778434

theorem geometric_sequence_nth_term_divisible_by_10_8 {n : ℕ} (hn : n ≥ 1) :
    let a := (5 : ℚ) / 8,
        r := 50 / a in
    (∀ m : ℕ, m ≥ 1 → a * r^(m - 1) ≥ 10^8 → m = 9) :=
by
  intros
  let a := (5 / 8 : ℚ)
  let r := (50 : ℚ) / a
  admit

end geometric_sequence_nth_term_divisible_by_10_8_l778_778434


namespace find_quadratic_polynomial_l778_778657

-- Given conditions to construct a quadratic polynomial with real coefficients
noncomputable def quadratic_polynomial (a b c : ℂ) (h1 : a = 5 + 2 * complex.I) : polynomial ℂ :=
  3 * (X - C a) * (X - C (conj a))

-- The required proof problem statement
theorem find_quadratic_polynomial (x : ℂ) :
    quadratic_polynomial 5 2 0 rfl = 3 * X^2 - 30 * X + 87 :=
sorry

end find_quadratic_polynomial_l778_778657


namespace marble_drawing_probability_l778_778962

-- Define the number of marbles of each color
def blue_marbles : Nat := 4
def orange_marbles : Nat := 3
def green_marbles : Nat := 5

-- Define the total number of marbles
def total_marbles : Nat := blue_marbles + orange_marbles + green_marbles

-- Define the probability calculation problem
theorem marble_drawing_probability :
  (blue_marbles = 4)
  ∧ (orange_marbles = 3)
  ∧ (green_marbles = 5)
  ∧ (total_marbles = 12)
  → (1 / 4620) = (3! * 4! * 3! * 5!) / 12! :=
by
  sorry

end marble_drawing_probability_l778_778962


namespace non_zero_real_x_solution_l778_778196

noncomputable section

variables {x : ℝ} (hx : x ≠ 0)

theorem non_zero_real_x_solution 
  (h : (3 * x)^5 = (9 * x)^4) : 
  x = 27 := by
  sorry

end non_zero_real_x_solution_l778_778196


namespace ellipse_and_fixed_point_l778_778292

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778292


namespace original_curve_equation_l778_778241

theorem original_curve_equation (x y : ℝ) (θ : ℝ) (hθ : θ = π / 4)
  (h : (∃ P : ℝ × ℝ, P = (x, y) ∧ (∃ P' : ℝ × ℝ, P' = (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ) ∧ ((P'.fst)^2 - (P'.snd)^2 = 2)))) :
  x * y = -1 :=
sorry

end original_curve_equation_l778_778241


namespace trig_identity_simplification_l778_778017

theorem trig_identity_simplification (α : ℝ) :
  (cos (π + α) * sin (α + 2 * π)) / (sin (-α - π) * cos (-π - α)) = 1 :=
by
  sorry

end trig_identity_simplification_l778_778017


namespace perpendicular_lines_condition_l778_778542

theorem perpendicular_lines_condition (k : ℝ) : 
  (k = 5 → (∃ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 ∧ x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 → (k = 5 ∨ k = -1)) :=
sorry

end perpendicular_lines_condition_l778_778542


namespace problem_equivalent_l778_778480

-- Define variables and conditions
variables {a b : ℤ}

-- Given conditions as per the problem statement
def condition1 : Prop := 4 * a + 6 * b = -98
def condition2 : Prop := a * b = -168

-- Theorem statement combining conditions and asserting the final result
theorem problem_equivalent (h1 : condition1) (h2 : condition2) : a + 2 * b = 10 :=
sorry

end problem_equivalent_l778_778480


namespace area_of_triangle_AME_is_75_over_8_l778_778014

noncomputable def rectangle_area_of_AME : ℝ :=
  let AB := 8
  let BC := 6
  let AC := real.sqrt (AB^2 + BC^2)
  let M := AC / 2
  let AE := 25 / 4
  let EM := real.sqrt((AC / 2)^2 - AE^2)
  (1 / 2) * AE * EM

theorem area_of_triangle_AME_is_75_over_8 : rectangle_area_of_AME = 75 / 8 := by
  sorry

end area_of_triangle_AME_is_75_over_8_l778_778014


namespace total_expected_rain_l778_778981

noncomputable def ev_day := 
  (0.40 * 0) + (0.25 * 4) + (0.35 * 10)

noncomputable def ev_total_days (days : ℕ) := 
  ev_day * days

theorem total_expected_rain : ev_total_days 5 = 22.5 := 
by
  sorry

end total_expected_rain_l778_778981


namespace number_of_solutions_l778_778368

theorem number_of_solutions (a : ℝ) :
  let sys_eqns (x y : ℝ) := (x^2 - y^2 = 0) ∧ ((x - a)^2 + y^2 = 1)
  in
  (a > real.sqrt 2 ∨ (1 < abs a ∧ abs a < real.sqrt 2) ∨ abs a < 1) →
  (∃! xy1 xy2 xy3 xy4 : ℝ × ℝ, sys_eqns xy1.fst xy1.snd ∧ sys_eqns xy2.fst xy2.snd ∧ sys_eqns xy3.fst xy3.snd ∧ sys_eqns xy4.fst xy4.snd) ∨
  (a = real.sqrt 2 ∨ a = 1) →
  (∃! xy1 xy2 : ℝ × ℝ, sys_eqns xy1.fst xy1.snd ∧ sys_eqns xy2.fst xy2.snd) :=
by
  sorry

end number_of_solutions_l778_778368


namespace ellipse_equation_fixed_point_l778_778275

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778275


namespace infinite_composite_in_sequence_l778_778455

/-
Prove that there are infinitely many composite numbers in the sequence 1, 31, 331, 3331, ...
-/

theorem infinite_composite_in_sequence : 
  ∀ n : ℕ, ∃ k > n, composite (sequence_k k) :=
by sorry

/-
Helper definitions:
-/

def sequence_k (k : ℕ) : ℕ :=
  if k = 0 then 1 else (10 ^ k) + 1

def composite (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ n = p * q

end infinite_composite_in_sequence_l778_778455


namespace greatest_divisor_l778_778105

theorem greatest_divisor (d : ℕ) (h1 : 1246 % d = 38) (h2 : 2550 % d = 58) : d = Nat.gcd (1246 - 38) (2550 - 58) := by
  have h3 : 1246 - 38 = 1208 := rfl
  have h4 : 2550 - 58 = 2492 := rfl
  rw [h3, h4]
  exact Nat.gcd_eq_right h2

end greatest_divisor_l778_778105


namespace no_unsatisfactory_grades_l778_778448

theorem no_unsatisfactory_grades (total_students : ℕ)
  (top_marks : ℕ) (average_marks : ℕ) (good_marks : ℕ)
  (h1 : top_marks = total_students / 6)
  (h2 : average_marks = total_students / 3)
  (h3 : good_marks = total_students / 2) :
  total_students = top_marks + average_marks + good_marks := by
  sorry

end no_unsatisfactory_grades_l778_778448


namespace quadratic_solution_l778_778048

theorem quadratic_solution (x : ℝ) : (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) :=
by
  sorry

end quadratic_solution_l778_778048


namespace angle_BCA_l778_778387

noncomputable def midpoint {α : Type*} [field α] {V : Type*} [add_comm_group V] [module α V]
  (A D : V) : V := (1/2) • (A + D)

-- Let M be the midpoint of AD
variable {A B C D M O : Point}
variable [convex_quad A B C D]

-- Conditions from problem
variable (hm : midpoint A D = M)
variable (h1 : ang A B M = 55)
variable (h2 : ang A M B = 70)
variable (h3 : ang B O C = 80)
variable (h4 : ang A D C = 60)
variable (h5 : line_intersect B M A C O)

theorem angle_BCA : ang B C A = 35 := 
by
  sorry

end angle_BCA_l778_778387


namespace HN_passes_through_fixed_point_l778_778254

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778254


namespace sum_is_composite_l778_778785

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ x * y = a + b + c + d :=
sorry

end sum_is_composite_l778_778785


namespace distinct_prime_factors_product_of_divisors_90_l778_778417

theorem distinct_prime_factors_product_of_divisors_90 : 
  let B := (∏ d in ({1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90} : Finset ℕ), d)
  in (∃ p : Finset ℕ, p = {2, 3, 5} ∧ ∃ n, B = 2^n * 3^(2*n) * 5^n) → 
    p.card = 3 := by
sorry

end distinct_prime_factors_product_of_divisors_90_l778_778417


namespace multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l778_778788

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem multiple_of_4 : y % 4 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_8 : y % 8 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_16 : y % 16 = 0 := by
  -- proof needed
  sorry

theorem not_multiple_of_32 : y % 32 ≠ 0 := by
  -- proof needed
  sorry

end multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l778_778788


namespace find_quadratic_polynomial_with_root_l778_778658

noncomputable def quadratic_polynomial (a b c : ℝ) : ℝ := 3 * a^2 - 30 * b + 87

theorem find_quadratic_polynomial_with_root (x : ℂ) (h₁ : x = 5 + 2 * complex.I) 
        (h₂ : x.conj = 5 - 2 * complex.I) : 
        quadratic_polynomial x.re x.im (3) = 3 * (x^2).re - 30 * x.re + 87  :=
by
  -- Proof goes here
  sorry

end find_quadratic_polynomial_with_root_l778_778658


namespace min_cos_squared_theta_l778_778335

-- Define unit vectors e1 and e2
variables {e1 e2 : ℝ^2}
-- Assume e1 and e2 are unit vectors
  (unit_e1 : ‖e1‖ = 1)
  (unit_e2 : ‖e2‖ = 1)
-- Given condition |2e1 - e2| ≤ √2
  (condition : ‖2 • e1 - e2‖ ≤ real.sqrt 2)

-- Define vectors a and b
def a : ℝ^2 := e1 + e2
def b : ℝ^2 := 3 • e1 + e2

-- Define the angle θ between vectors a and b
noncomputable def θ : ℝ := real.angle a b

-- Prove the minimum value of cos²θ
theorem min_cos_squared_theta : min (real.cos θ)^2 = 28 / 29 :=
sorry

end min_cos_squared_theta_l778_778335


namespace geometric_sequence_sum_l778_778403

open Nat

noncomputable def geometric_sum (a q n : ℕ) : ℕ :=
  a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (S : ℕ → ℕ) (q a₁ : ℕ)
  (h_q: q = 2)
  (h_S5: S 5 = 1)
  (h_S: ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) :
  S 10 = 33 :=
by
  sorry

end geometric_sequence_sum_l778_778403


namespace prime_factor_of_sum_of_four_consecutive_integers_l778_778913

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) :
  ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ ((n - 2) + (n - 1) + n + (n + 1)) :=
by
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_l778_778913


namespace tree_planting_equation_l778_778755

variables (x : ℝ)

theorem tree_planting_equation (h1 : x > 50) :
  (300 / (x - 50) = 400 / x) ≠ False :=
by
  sorry

end tree_planting_equation_l778_778755


namespace BF_equals_AF_plus_CF_l778_778381

variables {A B C D E F : Type} [metric_space A]

-- Conditions
variables (ABC : Triangle A) (AB AC BC : Segment A)

variables (D : Point BC) (E : Point (extension BA))

axiom AB_lt_AC : AB < AC
axiom AC_lt_BC : AC < BC
axiom BD_eq_BE : BD = BE
axiom BD_eq_AC : BD = AC

-- Circumcircles intersection
axiom circumcircle_BDE_intersects_ABC : 
  intersects (circumcircle (to_tris B D E)) (circumcircle ABC) = F

-- Question to Prove
theorem BF_equals_AF_plus_CF : 
  BF = AF + CF :=
sorry

end BF_equals_AF_plus_CF_l778_778381


namespace number_of_games_l778_778938

-- Definitions based on the conditions
def initial_money : ℕ := 104
def cost_of_blades : ℕ := 41
def cost_per_game : ℕ := 9

-- Lean 4 statement asserting the number of games Will can buy is 7
theorem number_of_games : (initial_money - cost_of_blades) / cost_per_game = 7 := by
  sorry

end number_of_games_l778_778938


namespace solve_for_y_l778_778821

variable (y : ℝ)

theorem solve_for_y (h : 64^(3 * y) = 16^(2 * y + 3)) : y = 6 / 5 :=
sorry

end solve_for_y_l778_778821


namespace standard_deviation_is_three_l778_778476

noncomputable def standard_deviation_of_data (x : ℝ) (h_mode_median : x ≠ 5 ∧ 2 / 3 * ((x + 2) / 2) = 2) : ℝ :=
  let d := [10, 5, x, 2, 2, 1]
  let mean := d.sum / 6
  let variance := (d.map (λ xi, (xi - mean)^2)).sum / 6
  real.sqrt variance

theorem standard_deviation_is_three :
  ∃ x : ℝ, x ≠ 5 ∧ 2 / 3 * ((x + 2) / 2) = 2 ∧ standard_deviation_of_data x ⟨x ≠ 5, 2 / 3 * ((x + 2) / 2) = 2⟩ = 3 :=
by {
  sorry
}

end standard_deviation_is_three_l778_778476


namespace debby_photos_of_friends_l778_778809

theorem debby_photos_of_friends (F : ℕ) (h1 : 23 + F = 86) : F = 63 := by
  -- Proof steps will go here
  sorry

end debby_photos_of_friends_l778_778809


namespace constant_function_of_inequality_l778_778646

theorem constant_function_of_inequality (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_inequality_l778_778646


namespace problem_statement_l778_778138

noncomputable def line_intersects_parabola (m t : ℝ) : Prop :=
  let x1 : ℝ := -8 * t + sqrt (64 * t^2 - 64 * m) / 2
  let x2 : ℝ := -8 * t - sqrt (64 * t^2 - 64 * m) / 2
  let AP_squared : ℝ := (1 + t^2) * x1^2
  let BP_squared : ℝ := (1 + t^2) * x2^2
  let sum_reciprocals : ℝ := 1 / AP_squared + 1 / BP_squared
  sum_reciprocals = 1 / (1 + t^2) * (8 * t^2 - m) / (8 * m^2)

-- The problem statement
theorem problem_statement (m : ℝ) :
  (∃ t : ℝ, line_intersects_parabola m t) → m = -8 :=
sorry

end problem_statement_l778_778138


namespace calculate_loss_percentage_l778_778978

def cost_price_radio : ℝ := 1500
def cost_price_tv : ℝ := 8000
def cost_price_fridge : ℝ := 25000

def selling_price_radio : ℝ := 1110
def selling_price_tv : ℝ := 7500
def selling_price_fridge : ℝ := 23000

def total_cost_price : ℝ := cost_price_radio + cost_price_tv + cost_price_fridge
def total_selling_price : ℝ := selling_price_radio + selling_price_tv + selling_price_fridge
def total_loss : ℝ := total_cost_price - total_selling_price
def loss_percentage : ℝ := (total_loss / total_cost_price) * 100

theorem calculate_loss_percentage : abs (loss_percentage - 8.38) < 0.01 :=
by {
  have h1 : total_cost_price = 1500 + 8000 + 25000 := rfl,
  have h2 : total_selling_price = 1110 + 7500 + 23000 := rfl,
  have h3 : total_loss = total_cost_price - total_selling_price := rfl,
  have h4 : loss_percentage = (total_loss / total_cost_price) * 100 := rfl,
  have h5 : (34500 - 31610) / 34500 * 100 = 8.376811594202898 := by norm_num,
  have h6 : abs (8.376811594202898 - 8.38) < 0.01 := by norm_num,
  exact h6,
}

end calculate_loss_percentage_l778_778978


namespace chess_club_mixed_groups_l778_778890

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l778_778890


namespace probability_of_summing_to_four_l778_778470

open Set

noncomputable def probability_sum_to_four : ℝ :=
  measure.univ.sum $ λ x : ℝ, 
    if (0 <= x ∧ x <= 3.5 ∧
        -- first number interval checks
        ((x < 0.25 ∧ round_half_int x = 0) ∨ 
         (0.25 ≤ x ∧ x < 0.75 ∧ round_half_int x = 0.5) ∨ 
         (0.75 ≤ x ∧ x < 1.25 ∧ round_half_int x = 1) ∨ 
         (1.25 ≤ x ∧ x < 1.75 ∧ round_half_int x = 1.5) ∨ 
         (1.75 ≤ x ∧ x < 2.25 ∧ round_half_int x = 2) ∨ 
         (2.25 ≤ x ∧ x < 2.75 ∧ round_half_int x = 2.5) ∨ 
         (2.75 ≤ x ∧ x < 3.25 ∧ round_half_int x = 3) ∨ 
         (3.25 ≤ x ∧ x ≤ 3.5 ∧ round_half_int x = 3.5)) ∧
        -- second number interval checks
        let y := 3.5 - x in
        (0 <= y ∧ y <= 3.5 ∧
        ((y < 0.25 ∧ round_half_int y = 0) ∨ 
         (0.25 ≤ y ∧ y < 0.75 ∧ round_half_int y = 0.5) ∨ 
         (0.75 ≤ y ∧ y < 1.25 ∧ round_half_int y = 1) ∨ 
         (1.25 ≤ y ∧ y < 1.75 ∧ round_half_int y = 1.5) ∨ 
         (1.75 ≤ y ∧ y < 2.25 ∧ round_half_int y = 2) ∨ 
         (2.25 ≤ y ∧ y < 2.75 ∧ round_half_int y = 2.5) ∨ 
         (2.75 ≤ y ∧ y < 3.25 ∧ round_half_int y = 3) ∨ 
         (3.25 ≤ y ∧ y ≤ 3.5 ∧ round_half_int y = 3.5)) ∧
        -- sum condition
        round_half_int x + round_half_int y = 4)
    then 1 / 3.5 else 0

theorem probability_of_summing_to_four : 
  probability_sum_to_four = 2 / 7 := 
sorry

-- To define the rounding to the nearest half-integer
def round_half_int (x : ℝ) : ℝ :=
  (2 * x).round / 2

end probability_of_summing_to_four_l778_778470


namespace sum_of_super_cool_triangle_areas_l778_778583

noncomputable def super_cool_triangle_sum_area : ℕ :=
  let leg_pairs := [(7, 42), (8, 24), (9, 18), (10, 15), (12, 12)] in
  let areas := leg_pairs.map (λ p, (p.1 * p.2) / 2) in
  areas.sum

theorem sum_of_super_cool_triangle_areas : super_cool_triangle_sum_area = 471 :=
by
  sorry

end sum_of_super_cool_triangle_areas_l778_778583


namespace p_plus_q_l778_778484

-- Definitions of the conditions
def horizontal_asymptote (p q : ℚ[X]) : Prop :=
  degree p < degree q

def vertical_asymptote (p q : ℚ[X]) (c : ℚ) : Prop :=
  q.eval c = 0 ∧ p.eval c ≠ 0

def is_quadratic (q : ℚ[X]) : Prop :=
  degree q = 2

def p_val (p : ℚ[X]) (c : ℚ) (k : ℚ) : Prop :=
  p.eval c = k

def q_val (q : ℚ[X]) (c : ℚ) (k : ℚ) : Prop :=
  q.eval c = k

-- Statement of the math proof problem
theorem p_plus_q
  (p q : ℚ[X])
  (hp : horizontal_asymptote p q)
  (vq : vertical_asymptote p q (-2))
  (hq : is_quadratic q)
  (hp3 : p_val p 3 2)
  (hq3 : q_val q 3 5) :
  p + q = X^2 + 3*X - 5 :=
sorry

end p_plus_q_l778_778484


namespace problem1_problem2_problem3_l778_778725

-- Problem 1
theorem problem1
  (α : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (hα : 0 < α ∧ α < 2 * Real.pi / 3) :
  (a + b) • (a - b) = 0 :=
sorry

-- Problem 2
theorem problem2
  (α k : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (x : ℝ × ℝ := k • a + 3 • b)
  (y : ℝ × ℝ := a + (1 / k) • b)
  (hk : 0 < k)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hxy : x • y = 0) :
  k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0 :=
sorry

-- Problem 3
theorem problem3
  (α k : ℝ)
  (h_eq : k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hk : 0 < k) :
  Real.pi / 2 ≤ α ∧ α < 2 * Real.pi / 3 :=
sorry

end problem1_problem2_problem3_l778_778725


namespace sum_of_coefficients_l778_778179

def P (x : ℝ) : ℝ := 3 * (x^8 - 2 * x^5 + x^3 - 7) - 5 * (x^6 + 3 * x^2 - 6) + 2 * (x^4 - 5)

theorem sum_of_coefficients : P 1 = -19 := by
  sorry

end sum_of_coefficients_l778_778179


namespace polygon_sides_l778_778747

theorem polygon_sides (h : 900 = ∑ angles in polygon ∧ exterior_sum = 360) : sides = 5 := sorry

end polygon_sides_l778_778747


namespace problem_l778_778704

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 4*x + 3 else (x^2 - 4*(-x) + 3)

theorem problem (h1 : ∀ x, f x = f (-x)) : 
  (f (-3) = 0) ∧ 
  ({x : ℝ // -2 ≤ x ∨ 2 ≤ x} ∀ a b : ℝ, a < b → {f a ≤ f b}). 
sorry

end problem_l778_778704


namespace factor_of_sum_of_consecutive_integers_l778_778918

theorem factor_of_sum_of_consecutive_integers :
  ∀ n : ℤ, ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by
  intro n
  use 2
  split
  exact prime_two
  split
  rfl
  calc
    2 ∣ 2 * (2 * n + 1) : by apply dvd_mul_right

end factor_of_sum_of_consecutive_integers_l778_778918


namespace harmonic_mean_ordered_pairs_l778_778635

theorem harmonic_mean_ordered_pairs (H : ℕ) : 
  ( ∀ (x y : ℕ), 0 < x ∧ 0 < y ∧ x < y ∧ (2 * x * y) / (x + y) = H ) → 
  H = 24 ^ 10 → 
  ∃ n, n = 619 :=
by {
  intro h1 H_eq,
  have h2 : (2 * x * y) / (x + y) = H := h1,
  unfold H_eq,
  sorry
}

end harmonic_mean_ordered_pairs_l778_778635


namespace probability_of_P_eq_neg1_l778_778495

open Complex

noncomputable def V : Finset ℂ :=
  {c : ℂ | c ∈ [Complex.I * Real.sqrt 2,
                 -Complex.I * Real.sqrt 2,
                 (1 + Complex.I) / Real.sqrt 8,
                 (-1 + Complex.I) / Real.sqrt 8,
                 (1 - Complex.I) / Real.sqrt 8,
                 (-1 - Complex.I) / Real.sqrt 8]}

theorem probability_of_P_eq_neg1 :
  let z := (finset.replica 12 V).filter_map id in
  z.prod id = -1 → true := by
  sorry

end probability_of_P_eq_neg1_l778_778495


namespace infinite_natural_sequence_l778_778831

theorem infinite_natural_sequence (a : ℕ) (q : ℕ) (hq : q > 1) :
  ∃ (seq : ℕ → ℕ), 
  (seq 0 = a) ∧ 
  (seq 1 = a * q) ∧ 
  (seq 2 = a * q^2) ∧
  (∀ n : ℕ, ∃ k : ℕ, k > n ∧
    ((seq (2 * k + 1) = (2 * seq (2 * k) - seq (2 * k - 1))) ∧
     (seq (2 * k + 2) = ((seq (2 * k + 1) ^ 2) / seq (2 * k - 1))))
  :=
sorry

end infinite_natural_sequence_l778_778831


namespace parabola_intercept_sum_l778_778850

theorem parabola_intercept_sum (a b c : ℝ) : 
  (∃ y : ℝ, a = 3 * y^2 - 9 * y + 5) ∧ (∀ x : ℝ, x = 0 → b ≠ c → 3 * b^2 - 9 * b + 5 = 0 ∧ 3 * c^2 - 9 * c + 5 = 0 ∧ b + c = 3) → 
  a + b + c = 8 :=
begin
  sorry
end

end parabola_intercept_sum_l778_778850


namespace min_value_expr_l778_778226

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 :=
sorry

end min_value_expr_l778_778226


namespace parabola_intercepts_sum_l778_778842

noncomputable def a : ℝ := 5

noncomputable def b : ℝ := (9 + Real.sqrt 21) / 6

noncomputable def c : ℝ := (9 - Real.sqrt 21) / 6

theorem parabola_intercepts_sum : a + b + c = 8 := by
  -- definition of a
  have ha : a = 5 := rfl
  
  -- definitions of b and c from roots of 3y^2 - 9y + 5 = 0
  have hb : b = (9 + Real.sqrt 21) / 6 := rfl
  have hc : c = (9 - Real.sqrt 21) / 6 := rfl
  
  -- Vieta's formulas implies b + c = 3
  have hb_c : b + c = 3 := by
    calc
    b + c = (9 + Real.sqrt 21) / 6 + (9 - Real.sqrt 21) / 6 : by rw [hb, hc]
    ... = (9 + 9) / 6 : by ring
    ... = 18 / 6 : by norm_num
    ... = 3 : by norm_num
  
  -- Sum a + b + c
  calc
  a + b + c = 5 + (b + c) : by rw [ha]
  ... = 5 + 3 : by rw [hb_c]
  ... = 8 : by norm_num

end parabola_intercepts_sum_l778_778842


namespace correct_factorization_l778_778935

theorem correct_factorization :
  (∀ a b : ℝ, ¬ (a^2 + b^2 = (a + b) * (a - b))) ∧
  (∀ a : ℝ, ¬ (a^4 - 1 = (a^2 + 1) * (a^2 - 1))) ∧
  (∀ x : ℝ, ¬ (x^2 + 2 * x + 4 = (x + 2)^2)) ∧
  (∀ x : ℝ, x^2 - 3 * x + 2 = (x - 1) * (x - 2)) :=
by
  sorry

end correct_factorization_l778_778935


namespace HN_passes_through_fixed_point_l778_778256

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778256


namespace min_trips_to_transport_400_tons_l778_778065

-- Define a cargo type that has a weight
structure Cargo where
  weight : ℝ
  is_multiple_of_quintal : ∃ (k : ℕ), weight = k * 0.1
  weight_leq_10tons : weight ≤ 10

-- Define the total cargo weight and distinctness constraint
def cargo_set : set Cargo := {
  c | c.weight ∈ { w : ℝ | ∃ k : ℕ, w = k * 0.1 ∧ w ≤ 10}
} 

-- Assert the total weight constraint and uniqueness
def total_weight (cargos : set Cargo) : ℝ :=
  ∑ cargo in cargos, cargo.weight

def all_weights_distinct (cargos : set Cargo) : Prop :=
  ∀ (c1 c2 : Cargo), c1 ∈ cargos → c2 ∈ cargos → c1 ≠ c2 → c1.weight ≠ c2.weight

-- Statement of the main theorem
theorem min_trips_to_transport_400_tons (cargos : set Cargo) :
  total_weight cargos = 400 ∧ all_weights_distinct cargos →
  ∃ min_trips : ℕ, min_trips = 51 ∧ (∀ t, t < min_trips → ¬ (t = min_trips)) ∧ 
  ∑ c in (c : set Cargo), c.weight ≤ 10 * min_trips :=
sorry

end min_trips_to_transport_400_tons_l778_778065


namespace sum_of_super_cool_areas_l778_778582

def is_super_cool (a b : ℕ) : Prop :=
  (a - 9) * (b - 9) = 81

theorem sum_of_super_cool_areas : 
  let areas := [(90 * 10) / 2, (36 * 12) / 2, (18 * 18) / 2].erase_dup
  areas.sum = 828 :=
by
  sorry

end sum_of_super_cool_areas_l778_778582


namespace fraction_replaced_by_second_solution_l778_778128

theorem fraction_replaced_by_second_solution
    (initial_salt : ℝ)
    (added_salt_percent : ℝ)
    (removed_salt_percent : ℝ)
    (final_salt_percent : ℝ) :
    initial_salt = 0.14 → 
    added_salt_percent = 0.22 → 
    removed_salt_percent = 0.14 → 
    final_salt_percent = 0.16 →
    ∃ x : ℝ, x = 1/4 :=
by
  intros h1 h2 h3 h4
  use 1/4
  -- Proof follows here
  sorry

end fraction_replaced_by_second_solution_l778_778128


namespace cats_center_l778_778990

def totalCatsInCenter 
  (jump: ℕ) (fetch: ℕ) (spin: ℕ)
  (jump_and_fetch: ℕ) (fetch_and_spin: ℕ) (jump_and_spin: ℕ)
  (all_three: ℕ) (none: ℕ) :
  ℕ :=
  let only_jump := jump - (jump_and_fetch + jump_and_spin - all_three)
  let only_fetch := fetch - (jump_and_fetch + fetch_and_spin - all_three)
  let only_spin := spin - (jump_and_spin + fetch_and_spin - all_three)
  let jump_and_fetch_only := jump_and_fetch - all_three
  let fetch_and_spin_only := fetch_and_spin - all_three
  let jump_and_spin_only := jump_and_spin - all_three
  only_jump + only_fetch + only_spin + jump_and_fetch_only + fetch_and_spin_only + jump_and_spin_only + all_three + none

theorem cats_center : totalCatsInCenter 40 25 30 15 10 12 5 7 = 70 :=
by
  let only_jump := 40 - (15 - 5 + 12 - 5)
  let only_fetch := 25 - (15 - 5 + 10 - 5)
  let only_spin := 30 - (10 - 5 + 12 - 5)
  let jump_and_fetch_only := 15 - 5
  let fetch_and_spin_only := 10 - 5
  let jump_and_spin_only := 12 - 5
  have total_cats := only_jump + only_fetch + only_spin + jump_and_fetch_only + fetch_and_spin_only + jump_and_spin_only + 5 + 7
  show total_cats = 70 from sorry

end cats_center_l778_778990


namespace intercepts_sum_eq_eight_l778_778835

theorem intercepts_sum_eq_eight :
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  a + b + c = 8 :=
by
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  -- Proof will go here
  sorry

end intercepts_sum_eq_eight_l778_778835


namespace HN_passes_through_fixed_point_l778_778258

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778258


namespace ellipse_solution_geometry_solution_l778_778310

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778310


namespace parallelogram_area_l778_778489

open Complex

theorem parallelogram_area :
  let roots := [2 * sqrt 3 - Complex.I, -2 * sqrt 3 - Complex.I, sqrt 3 + Complex.I, -sqrt 3 + Complex.I]
  ∃ (h : roots.perm ≃ Multiset.ofList [2 * sqrt 3 - Complex.I, -2 * sqrt 3 - Complex.I, sqrt 3 + Complex.I, -sqrt 3 + Complex.I]),
  ∃ (a b c d : ℂ), List.perm [a, b, c, d] (roots : List ℂ) →
  (a - c).re = 0 ∧ (a - d).im = 0 →
  (b - d).im = 0 ∧ (c - d).re * (d - b).im = 8 * sqrt 3 :=
sorry

end parallelogram_area_l778_778489


namespace trig_solution_l778_778492

noncomputable def solution_set := {x : Real | x = (5 * Real.pi)/6 ∨ x = (3 * Real.pi)/2}

theorem trig_solution (x : Real) (h : x ∈ Set.Icc 0 (2 * Real.pi)) (h_eq : sin x + sqrt 3 * cos x + 1 = 0) : 
  x ∈ solution_set := sorry

end trig_solution_l778_778492


namespace probability_harmonic_set_is_one_seventeenth_l778_778736

def is_harmonic_set (A : Set ℝ) : Prop :=
  ∀ ⦃x⦄, x ∈ A → (1 / x) ∈ A

def M : Set ℝ := {-1, 0, 1/3, 1/2, 1, 2, 3, 4}

def num_nonempty_subsets (S : Set ℝ) : ℕ := 
  (Nat.pow 2 S.card) - 1

def num_harmonic_nonempty_subsets (S : Set ℝ) : ℕ :=
  let reciprocals := S.filter (λ x, 1 / x ∈ S)
  num_nonempty_subsets reciprocals

noncomputable def probability_harmonic_set : ℚ :=
  num_harmonic_nonempty_subsets M / num_nonempty_subsets M

theorem probability_harmonic_set_is_one_seventeenth : 
  probability_harmonic_set = 1 / 17 := by
  sorry

end probability_harmonic_set_is_one_seventeenth_l778_778736


namespace min_sum_first_n_terms_l778_778352

theorem min_sum_first_n_terms {a : ℕ → ℤ} (h₁ : ∀ n, a n = 2 * n - 48) : 
  (∃ n, (n = 23 ∨ n = 24) ∧ 
         ∀ m, sum_first_n a n ≤ sum_first_n a m) :=
by
  sorry

end min_sum_first_n_terms_l778_778352


namespace correct_equation_l778_778934

theorem correct_equation (a b : ℝ) : (a - b) ^ 3 * (b - a) ^ 4 = (a - b) ^ 7 :=
sorry

end correct_equation_l778_778934


namespace left_building_percentage_l778_778012

theorem left_building_percentage (L R : ℝ)
  (middle_building_height : ℝ := 100)
  (total_height : ℝ := 340)
  (condition1 : L + middle_building_height + R = total_height)
  (condition2 : R = L + middle_building_height - 20) :
  (L / middle_building_height) * 100 = 80 := by
  sorry

end left_building_percentage_l778_778012


namespace problem1_problem2_l778_778714

-- Function definition
def f (x : ℝ) : ℝ := 1 / 2 - Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- Definitions for specific constants
def interval1 (k : ℤ) : Set ℝ :=
  { x | (Real.pi / 3 + k * Real.pi) ≤ x ∧ x ≤ (5 * Real.pi / 6 + k * Real.pi) }

def interval2 : Set ℝ := Ioo (-1 / 2) 1

-- Problem 1: 
theorem problem1 (k : ℤ) : 
  ∀ x, x ∈ interval1 k → ∀ u v, u = f x → v = f (x + Real.pi) → u = v :=
sorry

-- Problem 2:
theorem problem2 {a b c : ℝ} (h : b^2 + c^2 - a^2 > b * c) (A : ℝ) :
  0 < A ∧ A < Real.pi / 3 → f A ∈ interval2 :=
sorry

end problem1_problem2_l778_778714


namespace ABCD_is_trapezoid_l778_778767

-- Definitions based on given conditions
variables (A B C D O O1 : Type)
variables [CyclicQuadrilateral A B C D] [IntersectionPoint O A C B D]
variables [Circumcenter O1 A B C D] [LiesOnCircumcircle O1 C O D]

-- The statement to be proven
theorem ABCD_is_trapezoid (h1 : CyclicQuadrilateral A B C D)
  (h2 : IntersectionPoint O A C B D)
  (h3 : Circumcenter O1 A B C D)
  (h4 : LiesOnCircumcircle O1 C O D) : IsTrapezoid A B C D :=
  sorry

end ABCD_is_trapezoid_l778_778767


namespace greatest_c_value_l778_778512

theorem greatest_c_value (c : ℤ) : 
  (∀ (x : ℝ), x^2 + (c : ℝ) * x + 20 ≠ -7) → c = 10 :=
by
  sorry

end greatest_c_value_l778_778512


namespace find_f_expression_l778_778347

noncomputable def f : ℝ → ℝ := sorry  -- Definition of f, to be specified later

theorem find_f_expression (x : ℝ) (h : ∀ x, f x = lg(x^2 + 1)) : 
  f x = lg(x^2 - 4*x + 5) :=
sorry  -- Proof omitted

end find_f_expression_l778_778347


namespace pool_fill_time_l778_778219

-- Definitions according to conditions
def pool_volume : ℝ := 15000  -- pool volume in gallons

def hose_rate1 : ℝ := 2       -- rate of first type of hoses in gallons per minute
def hose_rate2 : ℝ := 3       -- rate of second type of hoses in gallons per minute

def hoses_count1 : ℕ := 2     -- number of first type of hoses
def hoses_count2 : ℕ := 2     -- number of second type of hoses

-- The main theorem to be proved
theorem pool_fill_time (volume : ℝ) (rate1 rate2 : ℝ) (count1 count2 : ℕ) :
  let total_rate := (rate1 * count1) + (rate2 * count2) in
  let time_minutes := volume / total_rate in
  let time_hours := time_minutes / 60 in
  volume = pool_volume →
  rate1 = hose_rate1 →
  rate2 = hose_rate2 →
  count1 = hoses_count1 →
  count2 = hoses_count2 →
  time_hours = 25 := 
sorry

end pool_fill_time_l778_778219


namespace interior_sum_nine_l778_778772

-- Defining the function for the sum of the interior numbers in the nth row of Pascal's Triangle
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

-- Given conditions
axiom interior_sum_4 : interior_sum 4 = 6
axiom interior_sum_5 : interior_sum 5 = 14

-- Goal to prove
theorem interior_sum_nine : interior_sum 9 = 254 := by
  sorry

end interior_sum_nine_l778_778772


namespace percentage_of_red_non_honda_cars_l778_778494

-- Define the conditions
def total_cars : ℕ := 900
def honda_cars : ℕ := 500
def red_per_100_honda_cars : ℕ := 90
def red_percent_total := 60

-- Define the question we want to answer
theorem percentage_of_red_non_honda_cars : 
  let red_honda_cars := (red_per_100_honda_cars / 100 : ℚ) * honda_cars
  let total_red_cars := (red_percent_total / 100 : ℚ) * total_cars
  let red_non_honda_cars := total_red_cars - red_honda_cars
  let non_honda_cars := total_cars - honda_cars
  (red_non_honda_cars / non_honda_cars) * 100 = (22.5 : ℚ) :=
by
  sorry

end percentage_of_red_non_honda_cars_l778_778494


namespace number_of_arrangements_l778_778758

theorem number_of_arrangements (A B C D K1 K2 : Prop) :
  let original_acts := [A, B, C, D]
  let total_acts := [A, B, C, D, K1, K2]
  let possible_gaps := 5 in
  (possible_gaps * (possible_gaps + 1)) = 30 := by sorry

end number_of_arrangements_l778_778758


namespace orange_face_probability_correct_l778_778022

-- Define the number of faces
def total_faces : ℕ := 12
def green_faces : ℕ := 5
def orange_faces : ℕ := 4
def purple_faces : ℕ := 3

-- Define the probability of rolling an orange face
def probability_of_orange_face : ℚ := orange_faces / total_faces

-- Statement of the theorem
theorem orange_face_probability_correct :
  probability_of_orange_face = 1 / 3 :=
by
  sorry

end orange_face_probability_correct_l778_778022


namespace inequality_log_l778_778684

theorem inequality_log (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hab : a > b) (hc : 0 < c) (hc1 : c < 1) : log a c > log b c :=
sorry

end inequality_log_l778_778684


namespace alice_savings_l778_778598

noncomputable def monthly_savings_after_first : ℝ :=
  let x := 13.33 -- This is the correct answer identified as 13.33 or 40/3
  x

theorem alice_savings :
  let x := monthly_savings_after_first in
  10 + (10 + x) + (10 + 2 * x) = 70 :=
by
  sorry

end alice_savings_l778_778598


namespace ellipse_equation_proof_HN_fixed_point_l778_778321

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778321


namespace remainder_sum_b_1_to_28_mod_1000_l778_778975

noncomputable def b : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 1
| 3     := 1
| (n+4) := b (n+3) + 2 * b (n+2) + b (n+1)

theorem remainder_sum_b_1_to_28_mod_1000 :
  b 28 = 121393 ∧ b 29 = 196418 ∧ b 30 = 317811 →
  (∑ k in finset.range 28, b (k + 1)) % 1000 = 253 :=
by
  sorry

end remainder_sum_b_1_to_28_mod_1000_l778_778975


namespace find_extrema_l778_778207

noncomputable def max_min_values (x y : ℝ) : Prop :=
  (sqrt (x - 2) + sqrt (y - 3) = 3) ∧ 
  ((∀ x y, sqrt (x - 2) + sqrt (y - 3) = 3 → x - 2y ≤ 5 ∧ x = 11 ∧ y = 3) ∧
   (∀ x y, sqrt (x - 2) + sqrt (y - 3) = 3 → x - 2y ≥ -22 ∧ x = 2 ∧ y = 12))

theorem find_extrema : max_min_values :=
by
  sorry

end find_extrema_l778_778207


namespace solve_fractional_equation_l778_778018

theorem solve_fractional_equation (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (3 / (x^2 - x) + 1 = x / (x - 1)) → x = 3 :=
by
  sorry -- Placeholder for the actual proof

end solve_fractional_equation_l778_778018


namespace common_external_tangent_y_intercept_l778_778926

def circle1_center : (ℝ × ℝ) := (1, 3)
def circle1_radius : ℝ := 3
def circle2_center : (ℝ × ℝ) := (10, 6)
def circle2_radius : ℝ := 5

theorem common_external_tangent_y_intercept :
  (∃ m b : ℝ, y = m * x + b ∧ 
    b = 11 ∧
    -- additional conditions to represent the tangent properties
  ) :=
begin
  sorry
end

end common_external_tangent_y_intercept_l778_778926


namespace no_possible_solution_l778_778110

def is_strictly_increasing (l : List ℕ) : Prop := 
  ∀ i j, i < j → l.nth i < l.nth j

theorem no_possible_solution :
  ∀ (z u a n e l c i s m : ℕ), 
  z ≠ u ∧ z ≠ a ∧ z ≠ n ∧ z ≠ e ∧ z ≠ l ∧ z ≠ c ∧ z ≠ i ∧ z ≠ s ∧ z ≠ m ∧
  u ≠ a ∧ u ≠ n ∧ u ≠ e ∧ u ≠ l ∧ u ≠ c ∧ u ≠ i ∧ u ≠ s ∧ u ≠ m ∧
  a ≠ n ∧ a ≠ e ∧ a ≠ l ∧ a ≠ c ∧ a ≠ i ∧ a ≠ s ∧ a ≠ m ∧
  n ≠ e ∧ n ≠ l ∧ n ≠ c ∧ n ≠ i ∧ n ≠ s ∧ n ≠ m ∧
  e ≠ l ∧ e ≠ c ∧ e ≠ i ∧ e ≠ s ∧ e ≠ m ∧
  l ≠ c ∧ l ≠ i ∧ l ≠ s ∧ l ≠ m ∧
  c ≠ i ∧ c ≠ s ∧ c ≠ m ∧
  i ≠ s ∧ i ≠ m ∧
  s ≠ m ∧
  ∀ d, d ∈ [z, u, a, n, e, l, c, i, s, m] → 0 ≤ d ∧ d ≤ 9 →
  ¬ is_strictly_increasing [10*z + u, 10*z + a, 10*n + a, 10*n + e, 10*l + a, 10*l + u, 10*c + i, 10*s + a, 10*m + u, 10*e + l] :=
sorry

end no_possible_solution_l778_778110


namespace triangle_inscribed_circle_area_radius_l778_778979

noncomputable def radius (c : ℝ) : ℝ := c / (2 * Real.pi)
noncomputable def angle (a : ℝ) (c : ℝ) : ℝ := (a / c) * 360
noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ := (r^2 / 2) * (Real.sin A + Real.sin B + Real.sin C)

theorem triangle_inscribed_circle_area_radius :
  let c := 20 -- circumference of circle
  let r := radius c -- radius of circle
  let a := 5
  let b := 7
  let d := 8 -- lengths of arcs
  let A := angle a c
  let B := angle b c
  let C := angle d c -- angles of triangle
  r = 10 / Real.pi ∧ triangle_area r A B C = 239.7 / (2 * Real.pi ^ 2)
  sorry

end triangle_inscribed_circle_area_radius_l778_778979


namespace problem_statement_l778_778334

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem problem_statement
  (f_even : even_function f)
  (f_increasing : increasing_on f {x | x ≤ -2}) :
  f (-3) > f (-7/2) ∧ f (-7/2) > f 4 :=
sorry

end problem_statement_l778_778334


namespace sum_of_ns_l778_778089

theorem sum_of_ns (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999) (hn : ∃ k : ℕ, n = 7 * k + 4 ∧ 14 ≤ k ∧ k ≤ 142) : 
  ∑ k in Finset.Icc 14 142, (7 * k + 4) = 70950 :=
by
  sorry

end sum_of_ns_l778_778089


namespace sum_2011_terms_eq_1004_l778_778632

def equal_sum_sequence (a : ℕ → ℤ) (s : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = s

def sequence_nth_term (n : ℕ) : ℤ :=
  if n % 2 = 0 then 2 else -1

theorem sum_2011_terms_eq_1004 :
  equal_sum_sequence sequence_nth_term 1 →
  sequence_nth_term 0 = -1 →
  (∑ i in finset.range 2011, sequence_nth_term i) = 1004 := by
sorry

end sum_2011_terms_eq_1004_l778_778632


namespace FE_perpendicular_AC_l778_778601

-- Define the entities and conditions
variables {A B C D E F S : Point} [Line A B C D E F S]
  (h1: acute_triangle A B C)
  (h2: Length A B < Length A C)
  (h3: OnLine E A C)
  (h4: Length A B = Length A E)
  (h5: is_diameter A D (circumcircle A B C))
  (h6: is_center_arc S B C (circumcircle A B C) A)
  (h7: symmetric_point F D S)

-- Define the theorem
theorem FE_perpendicular_AC :
  Perpendicular (Line F E) (Line A C) :=
sorry

end FE_perpendicular_AC_l778_778601


namespace construct_rhombus_with_specific_properties_l778_778235

theorem construct_rhombus_with_specific_properties (a : ℝ) :
  ∃ (rhombus : Set (ℝ × ℝ)), 
    (∃ (area : ℝ), area = (3 * a^2 * real.sqrt 3) / 2) ∧ 
    ∃ (angle : ℝ), angle = 60 :=
begin
  sorry
end

end construct_rhombus_with_specific_properties_l778_778235


namespace Jane_age_l778_778008

theorem Jane_age (x : ℕ) 
  (h1 : ∃ n1 : ℕ, x - 1 = n1 ^ 2) 
  (h2 : ∃ n2 : ℕ, x + 1 = n2 ^ 3) : 
  x = 26 :=
sorry

end Jane_age_l778_778008


namespace ellipse_equation_and_fixed_point_proof_l778_778279

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778279


namespace polar_to_rect_l778_778630

theorem polar_to_rect (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (2.5, 5 * Real.sqrt 3 / 2) :=
by
  rw [hr, hθ]
  sorry

end polar_to_rect_l778_778630


namespace quartile_median_relationship_l778_778985

-- Define the conditions
variable (data : List ℝ)
variable (h_distinct : data.nodup)
variable (h_len : data.length = 25)

-- Define the quartile/percentile functions
def fourth_quartile (data : List ℝ) : ℝ := data[data.length * 3 / 4]
def median (data : List ℝ) : ℝ := data[data.length / 2]
def seventy_fifth_percentile (data : List ℝ) : ℝ := data[data.length * 3 / 4]

-- Prove the main statement
theorem quartile_median_relationship (a b c : ℝ)
  (h_a : a = fourth_quartile data)
  (h_b : b = median data)
  (h_c : c = seventy_fifth_percentile data) :
  b < a ∧ a = c := by
  sorry

end quartile_median_relationship_l778_778985


namespace ratio_of_areas_l778_778790

variable (A B C P : Type) [AddCommMonoid A] [AddCommMonoid B] [AddCommMonoid C] 
variable [Module ℝ A] [Module ℝ B] [Module ℝ C]

open scoped affine

def condition (P A B C : A) : Prop :=
  (P -ᵥ A) +ᵥ (3 • (P -ᵥ B)) +ᵥ (4 • (P -ᵥ C)) = 0

theorem ratio_of_areas (A B C P : A) (h : condition P A B C) :
  (area P A B) / (area A B C) = 3 :=
sorry

end ratio_of_areas_l778_778790


namespace pathway_width_l778_778140

theorem pathway_width {r1 r2 : ℝ} 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : r1 - r2 = 10) :
  r1 - r2 + 4 = 14 := 
by 
  sorry

end pathway_width_l778_778140


namespace unique_perpendicular_through_point_l778_778769

theorem unique_perpendicular_through_point
  (P : Point) (L : Line) : ∃! M : Line, passes_through M P ∧ perpendicular M L :=
sorry

end unique_perpendicular_through_point_l778_778769


namespace minimum_distance_ellipse_line_l778_778710

theorem minimum_distance_ellipse_line :
  let ellipse (x y : ℝ) := (x^2 / 25) + (y^2 / 9) = 1
  let line (x y : ℝ) := 4 * x - 5 * y + 40 = 0
  (∃ (P : ℝ × ℝ), ellipse (P.1) (P.2) ∧ (∃ m : ℝ, 4 * P.1 - 5 * P.2 + m = 0 ∧
   abs (40 - m) / real.sqrt (4^2 + 5^2) = 15 / real.sqrt (41))) :=
sorry

end minimum_distance_ellipse_line_l778_778710


namespace Problem1_Problem2_l778_778951

-- For Problem 1
theorem Problem1 (x : ℝ) (h : x^2 - sqrt 5 * x - x - 1 = 0) (hx : x ≠ 0) : x^2 + (1 / x^2) = 8 + 2 * sqrt 5 := 
sorry

-- For Problem 2
theorem Problem2 (x : ℝ) (h : x^2 + 1 = 3 * x) (hx : x ≠ 0) : sqrt x + (1 / sqrt x) = sqrt 5 := 
sorry

end Problem1_Problem2_l778_778951


namespace max_k_inequality_l778_778652

theorem max_k_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) 
                                      (h₂ : 0 ≤ b) (h₃ : b ≤ 1) 
                                      (h₄ : 0 ≤ c) (h₅ : c ≤ 1) 
                                      (h₆ : 0 ≤ d) (h₇ : d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) :=
sorry

end max_k_inequality_l778_778652


namespace chess_club_mixed_groups_l778_778889

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l778_778889


namespace hn_passes_fixed_point_l778_778268

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778268


namespace locus_of_point_C_l778_778726

noncomputable theory
open Real

theorem locus_of_point_C 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ C : ℝ × ℝ, C = (x, (b / a) * x) ∧ (ab / sqrt (a^2 + b^2)) ≤ x ∧ x ≤ a := 
sorry

end locus_of_point_C_l778_778726


namespace max_tan_alpha_l778_778738

theorem max_tan_alpha (α β : ℝ) (hα : 0 < α) (hα' : α < π/2) (hβ : 0 < β) (hβ' : β < π/2)
  (h : sin α = cos (α + β) * sin β) : tan α ≤ 1/2 :=
begin
  sorry,
end

end max_tan_alpha_l778_778738


namespace mixed_groups_count_l778_778904

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l778_778904


namespace ratio_singh_bhatia_correct_l778_778609

-- Define the initial conditions
def initial_amount : ℕ := 70
def singh_gain : ℕ := 50
def singh_initial : ℕ := initial_amount
def singh_final : ℕ := singh_initial + singh_gain
def ratio_ashtikar_singh : ℚ := 1 / 2

-- Define relevant amounts
def ashtikar_final : ℕ := (ratio_ashtikar_singh * singh_final).nat_abs
def total_initial_amount : ℕ := initial_amount * 3
def bhatia_final : ℕ := total_initial_amount - (ashtikar_final + singh_final)

-- Define the ratio to prove
def ratio_singh_bhatia : ℚ := singh_final / bhatia_final

theorem ratio_singh_bhatia_correct : ratio_singh_bhatia = 4 := by
  sorry

end ratio_singh_bhatia_correct_l778_778609


namespace total_money_l778_778617

-- Definitions from conditions
def cecil := 600
def catherine := 2 * cecil - 250
def carmela := 2 * cecil + 50
def average := (cecil + catherine + carmela) / 3
def carlos := average + 200

-- Total amount of money
def total := cecil + catherine + carmela + carlos

-- Proof statement
theorem total_money : total = 3933.33 := by
  sorry

end total_money_l778_778617


namespace max_mom_money_on_thursday_l778_778003

theorem max_mom_money_on_thursday : 
  let tues := 8 in
  let wed := 5 * tues in
  let thurs := tues + 41 in
  thurs = 49 :=
by
  sorry

end max_mom_money_on_thursday_l778_778003


namespace alice_correct_percentage_l778_778599

def aliceTestResults (scores: List (ℚ × ℕ)) : ℚ :=
  let correctAnswers := scores.map (fun (percentage, total) => percentage * total)
  let sumCorrect := correctAnswers.sum
  let totalQuestions := scores.map Prod.snd.sum
  sumCorrect / totalQuestions

theorem alice_correct_percentage :
  aliceTestResults [(0.75, 20), (0.85, 50), (0.60, 30), (0.90, 40)] = 0.80 :=
by
  sorry

end alice_correct_percentage_l778_778599


namespace upstream_distance_18_l778_778971

theorem upstream_distance_18 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (still_water_speed : ℝ) : 
  upstream_distance = 18 :=
by
  have v := (downstream_distance / downstream_time) - still_water_speed
  have upstream_distance := (still_water_speed - v) * upstream_time
  sorry

end upstream_distance_18_l778_778971


namespace quadratic_one_pos_one_neg_l778_778112

theorem quadratic_one_pos_one_neg (a : ℝ) : 
  (a < -1) → (∃ x1 x2 : ℝ, x1 * x2 < 0 ∧ x1 + x2 > 0 ∧ (x1^2 + x1 + a = 0 ∧ x2^2 + x2 + a = 0)) :=
sorry

end quadratic_one_pos_one_neg_l778_778112


namespace max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l778_778716

noncomputable def f (a x : ℝ) := a * x + Real.log x
noncomputable def g (a x : ℝ) := x * f a x
noncomputable def e := Real.exp 1

-- Statement for part (1)
theorem max_value_fx_when_a_neg1 : 
  ∀ x : ℝ, 0 < x → (f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement for part (2)
theorem find_a_when_max_fx_is_neg3 : 
  (∀ x : ℝ, 0 < x ∧ x ≤ e → (f (-e^2) x ≤ -3)) →
  (∃ a : ℝ, a = -e^2) :=
sorry

-- Statement for part (3)
theorem inequality_gx_if_a_pos (a : ℝ) (hapos : 0 < a) 
  (x1 x2 : ℝ) (hxpos1 : 0 < x1) (hxpos2 : 0 < x2) (hx12 : x1 ≠ x2) :
  2 * g a ((x1 + x2) / 2) < g a x1 + g a x2 :=
sorry

end max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l778_778716


namespace mike_typing_time_l778_778441

-- Definitions based on the given conditions
def original_speed : ℕ := 65
def speed_reduction : ℕ := 20
def document_words : ℕ := 810
def reduced_speed : ℕ := original_speed - speed_reduction

-- The statement to prove
theorem mike_typing_time : (document_words / reduced_speed) = 18 :=
  by
    sorry

end mike_typing_time_l778_778441


namespace sum_of_super_cool_areas_l778_778580

def is_super_cool (a b : ℕ) : Prop :=
  (a - 9) * (b - 9) = 81

theorem sum_of_super_cool_areas : 
  let areas := [(90 * 10) / 2, (36 * 12) / 2, (18 * 18) / 2].erase_dup
  areas.sum = 828 :=
by
  sorry

end sum_of_super_cool_areas_l778_778580


namespace ellipse_solution_geometry_solution_l778_778307

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778307


namespace perimeter_ratio_of_similar_triangles_l778_778547

  -- Define the given conditions in Lean 4
  variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] 
    [MetricSpace C] [MetricSpace D] [MetricSpace E]

  -- Assume ∆ ADE ∼ ∆ ABC with a similarity ratio of 1:2
  def triangles_similar_with_ratio (triangle1 triangle2 : Type) (ratio : ℚ) : Prop :=
    ∀ (a b c : triangle1) (d e f : triangle2), 
      ∃ (h : SimilarTriangles a b c d e f), 
      h.ratio = ratio

  -- The goal is to prove the ratio of the perimeters of ∆ ADE and ∆ ABC is 1:2
  theorem perimeter_ratio_of_similar_triangles 
    (similar_triangles : triangles_similar_with_ratio ΔADE ΔABC (1:2)) 
    : (perimeter ΔADE) / (perimeter ΔABC) = (1:2) := 
  sorry
  
end perimeter_ratio_of_similar_triangles_l778_778547


namespace trigonometric_expression_eq_neg3_l778_778679

theorem trigonometric_expression_eq_neg3
  {α : ℝ} (h : Real.tan α = 1 / 2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) /
  ((Real.sin (-α))^2 - (Real.sin (5 * π / 2 - α))^2) = -3 :=
sorry

end trigonometric_expression_eq_neg3_l778_778679


namespace ellipse_equation_and_fixed_point_proof_l778_778284

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778284


namespace ellipse_and_fixed_point_l778_778304

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778304


namespace yard_length_300_l778_778759

theorem yard_length_300 {n : ℕ} {d : ℕ} (h1 : n = 26) (h2 : d = 12) : (n - 1) * d = 300 :=
by
  simp [h1, h2]
  sorry

end yard_length_300_l778_778759


namespace probability_is_correct_l778_778921

noncomputable def probability_perpendicular_vectors : ℚ :=
let possible_pairs := [(a, b) | a in finset.range(7) \ {0}, b in finset.range(7) \ {0}, a = 2 * b] in
let total_pairs := (finset.range(7) \ {0}) × (finset.range(7) \ {0}) in
(finset.card possible_pairs : ℚ) / (finset.card total_pairs : ℚ)

theorem probability_is_correct : probability_perpendicular_vectors = 1 / 12 :=
by
  sorry

end probability_is_correct_l778_778921


namespace find_r_l778_778215

-- Define the polynomial
def polynomial := λ x : ℝ, 12 * x^4 - 6 * x^3 - 63 * x^2 + 90 * x - 45

-- Define the condition that the polynomial is divisible by (x - r)²
def divisible_by_square (r : ℝ) := ∃ (s t : ℝ), polynomial = λ x : ℝ, 12 * (x - r)^2 * (x - s) * (x - t)

-- The proof problem statement
theorem find_r (r : ℝ) (h : divisible_by_square r) : r = 7 / 2 :=
sorry

end find_r_l778_778215


namespace set_intersection_complement_l778_778361

theorem set_intersection_complement (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = Set.univ) 
  (hA : ∀ x : ℝ, A x ↔ x^2 - x - 6 ≤ 0) 
  (hB : ∀ x : ℝ, B x ↔ Real.log x / Real.log (1/2) ≥ -1) :
  A ∩ (U \ B) = (Set.Icc (-2 : ℝ) 0 ∪ Set.Ioc 2 3) :=
by
  ext x
  -- Proof here would follow
  sorry

end set_intersection_complement_l778_778361


namespace ellipse_equation_min_area_OPAB_l778_778341

section GeometryProblem

variable (m : ℝ) (h_m : m > 0)
variable (x y : ℝ) (h_ellipse : m * x^2 + 3 * m * y^2 = 1)
variable (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (P : ℝ × ℝ)
variable (hxA : A = (3, 0)) (hB_on_y_axis : B.1 = 0) (hP_on_C : m * P.1^2 + 3 * m * P.2^2 = 1) (hP_right_y : P.1 > 0)
variable (h_BA_BP : (B.1 - 0)^2 + (B.2 - 0)^2 = (P.1 - 3)^2 + (P.2 - 0)^2)

-- Step 1: Prove the equation of the ellipse C
theorem ellipse_equation :
  (m = 1 / 6) → (∀ x y, (x^2 / 6 + y^2 / 2 = 1) ↔ (m * x^2 + 3 * m * y^2 = 1)) :=
by
  intro h_m_val
  intro x y
  rw [←h_m_val]
  sorry

-- Step 2: Prove the minimum area of quadrilateral OPAB
theorem min_area_OPAB :
  ∃ y0 : ℝ, (abs y0 = (√3 / 2)) → (area_OPAB (O, A, B, P) = 3 * (√3)) :=
by
  sorry

end GeometryProblem

end ellipse_equation_min_area_OPAB_l778_778341


namespace ellipse_equation_and_fixed_point_proof_l778_778280

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778280


namespace platform_length_l778_778096

theorem platform_length 
  (train_length : ℝ)
  (signal_time : ℝ)
  (platform_time : ℝ)
  (speed : ℝ):
  train_length = 300 →
  signal_time = 20 →
  platform_time = 39 →
  speed = train_length / signal_time →
  let platform_length := speed * platform_time - train_length in
  platform_length = 285 :=
by
  intros train_length_eq signal_time_eq platform_time_eq speed_eq
  have h : platform_length = speed * platform_time - train_length := rfl
  rw [train_length_eq, signal_time_eq, platform_time_eq, speed_eq]
  exact sorry

end platform_length_l778_778096


namespace arithmetic_square_root_of_3_neg_2_l778_778995

theorem arithmetic_square_root_of_3_neg_2 : Real.sqrt (3 ^ (-2: Int)) = 1 / 3 := 
by 
  sorry

end arithmetic_square_root_of_3_neg_2_l778_778995


namespace new_variance_correct_l778_778690

-- Define some helpers for handling statistics
def mean (s : List ℝ) : ℝ := s.sum / s.length
def variance (s : List ℝ) : ℝ := (s.map (λ x => (x - mean s)^2)).sum / s.length
def extended_set (s : List ℝ) (k : ℝ) := s ++ [k]

-- Define given conditions
def initial_set : List ℝ := [a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9]
axiom initial_avg : mean initial_set = 8
axiom initial_var : variance initial_set = 12
axiom new_avg : mean (extended_set initial_set 18) = 9 -- derived from solution; added element is 18

-- Theorem to prove the variance of the new sample data set is 19.8
theorem new_variance_correct :
  variance (extended_set initial_set 18) = 19.8 := 
sorry 

end new_variance_correct_l778_778690


namespace part1_B_correct_part2_at_least_two_correct_l778_778756

noncomputable def P_A : ℚ := 1 / 2
noncomputable def P_C : ℚ := 3 / 4
noncomputable def P_B : ℚ := 2 / 3

def prob_A_correct : ℚ := P_A
def prob_AC_incorrect : ℚ := (1 - P_A) * (1 - P_C)
def prob_BC_correct : ℚ := P_B * P_C

-- Question(I): Prove the probabilities of B and C answering correctly
theorem part1_B_correct :
  prob_AC_incorrect = 1 / 8 ∧ 
  prob_BC_correct = 1 / 2 → 
  P_B = 2 / 3 ∧ P_C = 3 / 4 :=
sorry

-- Question(II): Prove the probability that at least 2 out of A, B, and C answer correctly
theorem part2_at_least_two_correct :
  P_A = 1 / 2 ∧ 
  P_B = 2 / 3 ∧ 
  P_C = 3 / 4 →
  let P_at_least_two := P_A * P_B * P_C + (1 - P_A) * P_B * P_C + P_A * (1 - P_B) * P_C + P_A * P_B * (1 - P_C) in
  P_at_least_two = 17 / 24 :=
sorry

end part1_B_correct_part2_at_least_two_correct_l778_778756


namespace polygon_sides_l778_778744

theorem polygon_sides (h : 900 = interior_angles_sum + 360) (interior_angles_sum = (n-2) * 180) : n = 5 :=
by {
  sorry
}

end polygon_sides_l778_778744


namespace area_of_triangle_l778_778418

noncomputable def ellipse (x y : ℝ) : Prop :=
  4 * x^2 / 49 + y^2 / 6 = 1

def foci (F1 F2 : ℝ × ℝ) : Prop :=
  true -- This is a placeholder. The actual coordinates of F1 and F2 can be derived.

def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse P.1 P.2

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def ratio_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  distance P F1 / distance P F2 = 4 / 3

def length_condition (F1 F2 : ℝ × ℝ) : Prop :=
  distance F1 F2 = 5

theorem area_of_triangle {F1 F2 P : ℝ × ℝ} 
  (h₁ : is_on_ellipse P)
  (h₂ : ratio_condition P F1 F2)
  (h₃ : length_condition F1 F2) :
  1 / 2 * distance P F1 * distance P F2 = 6 :=
sorry

end area_of_triangle_l778_778418


namespace f_periodic_l778_778798

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_periodic (f : ℝ → ℝ)
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) :
  ∀ x : ℝ, f (x + 1) = f x :=
sorry

end f_periodic_l778_778798


namespace bill_score_l778_778004

theorem bill_score (B J S E : ℕ)
                   (h1 : B = J + 20)
                   (h2 : B = S / 2)
                   (h3 : E = B + J - 10)
                   (h4 : B + J + S + E = 250) :
                   B = 50 := 
by sorry

end bill_score_l778_778004


namespace sqrt_and_cuberoot_integers_l778_778485

noncomputable def is_integer (x : ℚ) : Prop :=
  ∃ (z : ℤ), x = z

theorem sqrt_and_cuberoot_integers (m n : ℤ) (k : ℤ)
  (h : (Real.sqrt (n : ℚ) + (m : ℚ)^(1/3) : ℚ) = k) :
  ∃ (a b : ℤ), n = a^2 ∧ m = b^3 :=
begin
  -- Proof goes here
  sorry
end

end sqrt_and_cuberoot_integers_l778_778485


namespace limit_of_sequence_l778_778623

noncomputable def a_n (n : ℕ) : ℝ := (2^n + 7^n) / (2^n - 7^(n-1))

theorem limit_of_sequence :
  (filter.at_top.map a_n).tendsto at_bot :=
begin
  sorry
end

end limit_of_sequence_l778_778623


namespace vector_b_bisects_angle_l778_778431

noncomputable def vector_a : ℝ³ := ⟨4, -3, 1⟩
noncomputable def vector_b : ℝ³ := ⟨2, -2, 2⟩
noncomputable def vector_v : ℝ³ := ⟨0, -(1/Real.sqrt 26), 3/(Real.sqrt 26)⟩

theorem vector_b_bisects_angle 
  (a : ℝ³) (b : ℝ³) (v : ℝ³) 
  (ha : a = vector_a) 
  (hb : b = vector_b) 
  (hv : v = vector_v) : 
  b = (2 / Real.sqrt 26) • (a + (Real.sqrt 26) • v) := sorry

end vector_b_bisects_angle_l778_778431


namespace discriminant_eq_M_l778_778376

theorem discriminant_eq_M (a b c x0 : ℝ) (h1: a ≠ 0) (h2: a * x0^2 + b * x0 + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * x0 + b)^2 :=
by
  sorry

end discriminant_eq_M_l778_778376


namespace factorial_product_not_perfect_power_l778_778452

noncomputable def is_factorial_product_not_perfect_power : Prop :=
  ¬ ∃ (n k : ℕ), k ≥ 2 ∧ n ^ k = ∏ i in finset.range 2022, (i + 1)!

theorem factorial_product_not_perfect_power :
  is_factorial_product_not_perfect_power :=
begin
  sorry
end

end factorial_product_not_perfect_power_l778_778452


namespace min_trips_theorem_l778_778070

noncomputable def min_trips (weights : List ℕ) (capacity : ℕ) (total_weight : ℕ) : ℕ :=
  if total_weight = 0 then 0
  else let weight_sum := List.sum weights
       if weight_sum <= capacity then 1
       else if weights.length = 1 then weights.length / capacity
       else (weight_sum + capacity - 1) / capacity

theorem min_trips_theorem (weights : List ℕ) (h_distinct : weights.nodup)
                          (h_max_weight : ∀ w ∈ weights, w ≤ 10)
                          (h_total_weight : List.sum weights = 400)
                          (h_multiple_of_quintal : ∀ w ∈ weights, w % 100 = 0) :
  min_trips weights 10 400 = 51 := 
sorry

end min_trips_theorem_l778_778070


namespace mixed_groups_count_l778_778907

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l778_778907


namespace square_root_of_25_squared_l778_778520

theorem square_root_of_25_squared :
  Real.sqrt (25 ^ 2) = 25 :=
sorry

end square_root_of_25_squared_l778_778520


namespace solve_m_value_l778_778033

-- Definitions for conditions
def hyperbola_eq (m : ℝ) : Prop := ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3
def has_focus (m : ℝ) : Prop := (∃ f1 f2 : ℝ, f1 = 0 ∧ f2 = 2)

-- Statement of the problem to prove
theorem solve_m_value (m : ℝ) (h_eq : hyperbola_eq m) (h_focus : has_focus m) : m = -1 :=
sorry

end solve_m_value_l778_778033


namespace area_per_tree_l778_778749

theorem area_per_tree :
  (∃ (area per square_meter per_tree : ℕ),
   area = 10_000 ∧
   per_tree = 40 ∧
   area / 250 = per_tree ∧
   250 = 100 / 0.4) :=
sorry

end area_per_tree_l778_778749


namespace super_cool_triangles_area_sum_l778_778574

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end super_cool_triangles_area_sum_l778_778574


namespace lines_meet_on_bisector_l778_778783

open Set
open Classical
open Function
noncomputable theory

variables {A B C M A1 A2 C1 C2 : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variables [MetricSpace A1] [MetricSpace A2] [MetricSpace C1] [MetricSpace C2]

-- Define the right-angle triangle ABC with ∠B = 90°
def right_angled_triangle (ABC : Triangle A B C) : Prop :=
  ∠ B = 90° ∧ B ∈ ABC.points

-- Define the median BM such that MA = MB = MC
def median_BM (BM : Segment B M) : Prop :=
  BM.is_median ∧ distance A M = distance B M ∧ distance C M = distance B M

-- Define the incircle touch points A1, A2, C1, C2
def incircle_touch_points (A1 A2 C1 C2 : Point) (ABC ABM BCM : Triangle) : Prop :=
  incircle ABM.touching_sides {A1, A2} ∧
  incircle BCM.touching_sides {C1, C2}

-- Statement to prove that lines A1A2 and C1C2 meet on the bisector of angle ABC
theorem lines_meet_on_bisector {ABC : triangle A B C} {BM : segment B M}
  (h1 : right_angled_triangle ABC)
  (h2 : median_BM BM)
  (h3 : incircle_touch_points A1 A2 C1 C2 ABC ABM BCM) :
  exists P : Point, P ∈ bisector ABC (A1A2 ∩ C1C2) :=
sorry

end lines_meet_on_bisector_l778_778783


namespace distance_from_point_to_circumcenter_l778_778677

/-- Given a triangle ABC with a point P and the perpendiculars PP₁, PP₂, PP₃ dropped to the sides 
BC, CA, AB respectively, and R the circumradius of triangle ABC. -/
theorem distance_from_point_to_circumcenter
  (A B C P P1 P2 P3 : Point)
  (h_in_circumcircle : P ∈ circumcircle A B C)
  (h_not_in_circumcircle : P ∉ circumcircle A B C)
  (h_perp_pp1 : ⊥ PP1 BC)
  (h_perp_pp2 : ⊥ PP2 CA)
  (h_perp_pp3 : ⊥ PP3 AB)
  (R : ℝ)
  (area_ABC : real)
  (area_P1P2P3 : real) :
  (d : ℝ) =
    R ^ 2 * (1 - 4 * (area_P1P2P3 / area_ABC)) ↔ h_in_circumcircle ∨
    R ^ 2 * (1 + 4 * (area_P1P2P3 / area_ABC)) ↔ h_not_in_circumcircle :=
sorry

end distance_from_point_to_circumcenter_l778_778677


namespace monotonicity_of_F_common_tangent_line_l778_778719

def f (m x : ℝ) : ℝ := m * Real.log (x + 1)
def g (x : ℝ) : ℝ := x / (x + 1)

theorem monotonicity_of_F (m : ℝ) :
  (∀ x, x > -1 → derivative (λ x, f m x - g x) x < 0 → m ≤ 0 ∨
      (m > 0 ∧ derivative (λ x, f m x - g x) x < 0 → x < -1 + 1 / m ∧
      derivative (λ x, f m x - g x) x > 0 → x > -1 + 1 / m)) :=
sorry

theorem common_tangent_line (m : ℝ) :
  (∀ a b : ℝ, (a > -1 ∧ b > -1) →
    (m > 0 → tangent_at (λ x, f m x) a = tangent_at (λ x, g x) b → m = 1)) :=
sorry

end monotonicity_of_F_common_tangent_line_l778_778719


namespace MK_perp_NK_l778_778167

open EuclideanGeometry

/-- Definition of the geometric setup. -/
variables {ℝ : Type*} [MetricSpace ℝ] [InnerProductSpace ℝ]
variables {O1 O2 A B C D M N K : Point ℝ}

/-- Conditions -/
axiom circle_O1 : circle ℝ O1
axiom circle_O2 : circle ℝ O2
axiom A_B_inters : A ∈ circle_O1 ∧ A ∈ circle_O2 ∧ B ∈ circle_O1 ∧ B ∈ circle_O2
axiom line_A_CD : ∃ line, (A ∈ line) ∧ ends_in (line_meets_circles line A circle_O1 circle_O2) C D
axiom M_midpoint_arc_BC : is_midpoint_arc M B C circle_O1
axiom N_midpoint_arc_BD : is_midpoint_arc N B D circle_O2
axiom K_midpoint_CD : is_midpoint K C D

/-- Result to be proved -/
theorem MK_perp_NK : ∃ (MK_perp_NK : Π (M K N : Point ℝ), is_perpendicular (M - K) (N - K)), 
  MK_perp_NK M K N M K N :=
sorry

end MK_perp_NK_l778_778167


namespace second_more_than_third_l778_778119

def firstChapterPages : ℕ := 35
def secondChapterPages : ℕ := 18
def thirdChapterPages : ℕ := 3

theorem second_more_than_third : secondChapterPages - thirdChapterPages = 15 := by
  sorry

end second_more_than_third_l778_778119


namespace f_prime_at_1_equals_4_over_3_l778_778344

noncomputable def f (x : ℝ) (f_prime_1 : ℝ) : ℝ :=
  Real.log x - f_prime_1 * x^2 + 3 * x - 4

theorem f_prime_at_1_equals_4_over_3 :
  (deriv (λ x : ℝ, f x (f' 1)) 1) = 4/3 :=
by
  let f' := deriv (λ x, f x (f' 1)) in
  have h : f' x = 1 / x - 2 * (f' 1) * x + 3, from sorry,
  have h_subst : f' 1 = 1 - 2 * (f' 1) + 3, from sorry,
  exact eq_of_linear_system h_subst (by norm_num : -2 * (f' 1) = -8/3) (by norm_num : 4/3)

end f_prime_at_1_equals_4_over_3_l778_778344


namespace largest_circle_area_in_region_S_l778_778446

-- Define the region S
def region_S (x y : ℝ) : Prop :=
  |x + (1 / 2) * y| ≤ 10 ∧ |x| ≤ 10 ∧ |y| ≤ 10

-- The question is to determine the value of k such that the area of the largest circle 
-- centered at (0, 0) fitting inside region S is k * π.
theorem largest_circle_area_in_region_S :
  ∃ k : ℝ, k = 80 :=
sorry

end largest_circle_area_in_region_S_l778_778446


namespace ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l778_778453

theorem ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one
  (m n : ℕ) : (10 ^ m + 1) % (10 ^ n - 1) ≠ 0 := 
  sorry

end ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l778_778453


namespace aileen_badminton_l778_778983

theorem aileen_badminton (m n : ℕ) (hmn_coprime : Nat.coprime m n)
  (h_mn_prob : m > 0 ∧ n > 0)
  (h_aileen_prob : ((m : ℚ) / n) * (1 / 4) + ((m : ℚ) / n) * ((3 / 4) * (m / n) * (1 / 4)) + ((m / n) * (3 / 4))^2 * (m / n) * (1 / 4) + ... = 9 / 10) :
  m + n = 73 := sorry

end aileen_badminton_l778_778983


namespace true_false_question_count_l778_778567

theorem true_false_question_count (n : ℕ) (h : (1 / 3) * (1 / 2)^n = 1 / 12) : n = 2 := by
  sorry

end true_false_question_count_l778_778567


namespace number_of_points_A_l778_778851

theorem number_of_points_A (B C : ℤ × ℤ) (hBC: B.1 ≠ C.1 ∨ B.2 ≠ C.2) : 
  ∃ A : ℤ × ℤ, ∥ A - B ∥ ≠ 0 ∧ ∥ A - C ∥ ≠ 0 ∧ 
  2 * (B.1 * C.2 - B.2 * C.1 + C.1 * A.2 - C.2 * A.1 + A.1 * B.2 - A.2 * B.1) = 6 :=
by sorry

end number_of_points_A_l778_778851


namespace find_b_for_perpendicular_lines_l778_778929

theorem find_b_for_perpendicular_lines (b : ℝ) :
  let v1 := ![2, 5]
  let v2 := ![b, -3]
  (v1.dot_product v2 = 0) → (b = 15/2) :=
by
  find_b_for_perpendicular_lines. sorry

end find_b_for_perpendicular_lines_l778_778929


namespace product_of_AB_equals_product_of_AD_l778_778686

-- Assume the definitions of the convex polygon and lines with given conditions
-- Define a noncomputable proof environment
noncomputable theory

variable {n : ℕ} (A B D : Fin n.succ → ℝ)

-- Define the given conditions
variables 
  (convex_polygon : ∀ i, 1 ≤ i → i < n.succ → convex A i)
  (P : ∀ i : Fin n, parallelogram (A i) (B i) (C i) (D i))
  (C : ∀ i, point_on_line (A i) (C i) (O))

-- The theorem statement
theorem product_of_AB_equals_product_of_AD 
    (h:  ∀ i : Fin n.succ, ∃ j : Fin n.succ, A i = A (i + 1) ∧ B j = B (i + 1)) 
    (hO : ∀ i, line_intersection (A i) (C i) (O)) 
    : (∏ (i : Fin n.succ), dist (A i) (B i)) = (∏ (i : Fin n.succ), dist (A i) (D i)) :=
  sorry

end product_of_AB_equals_product_of_AD_l778_778686


namespace sum_c_n_eq_l778_778405

-- Define sequences a and b recursively
def a : Nat → ℝ
| 0 => 1
| n+1 => a n + b n + Real.sqrt ((a n)^2 + (b n)^2)

def b : Nat → ℝ
| 0 => 1
| n+1 => a n + b n - Real.sqrt ((a n)^2 + (b n)^2)

-- Define the sequence c based on sequences a and b
def c (n : Nat) : ℝ := 2^n * (1 / (a n) + 1 / (b n))

-- Define the sum of the first n terms of sequence c
def S (n : Nat) : ℝ := (List.range n).sum (λ i => c i)

-- The theorem we need to prove
theorem sum_c_n_eq (n : ℕ) : S n = 2^(n + 2) - 4 := by
  sorry

end sum_c_n_eq_l778_778405


namespace lithium_hydroxide_formed_l778_778203

theorem lithium_hydroxide_formed :
  ∀ (Li3N H2O LiOH NH3 : Type) (n_Li3N n_H2O n_LiOH n_NH3 : ℕ),
  (n_Li3N = 1) →
  (n_H2O = 54) →
  (n_NH3 = 1) →
  (3 * n_LiOH = 3 * n_Li3N) →
  (n_LiOH = 3) :=
by
  intros Li3N H2O LiOH NH3 n_Li3N n_H2O n_LiOH n_NH3 h1 h2 h3 h4
  have h5: n_Li3N = 1 := h1
  have h6: 3 * n_LiOH = 3 * 1 := h4
  have h7: n_LiOH = 3 := sorry
  exact h7

end lithium_hydroxide_formed_l778_778203


namespace quadrilateral_impossible_l778_778031

theorem quadrilateral_impossible (a b c d : ℕ) (h1 : 2 * a ^ 2 - 18 * a + 36 = 0)
    (h2 : b ^ 2 - 20 * b + 75 = 0) (h3 : c ^ 2 - 20 * c + 75 = 0) (h4 : 2 * d ^ 2 - 18 * d + 36 = 0) :
    ¬(a + b > d ∧ a + c > d ∧ b + c > d ∧ a + d > c ∧ b + d > c ∧ c + d > b ∧
      a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by
  sorry

end quadrilateral_impossible_l778_778031


namespace min_fraction_value_l778_778208

theorem min_fraction_value (x : ℝ) (hx : x > 9) : ∃ y, y = 36 ∧ (∀ z, z = (x^2 / (x - 9)) → y ≤ z) :=
by
  sorry

end min_fraction_value_l778_778208


namespace space_convex_polyhedron_euler_characteristic_l778_778757

-- Definition of space convex polyhedron
structure Polyhedron where
  F : ℕ    -- number of faces
  V : ℕ    -- number of vertices
  E : ℕ    -- number of edges

-- Problem statement: Prove that for any space convex polyhedron, F + V - E = 2
theorem space_convex_polyhedron_euler_characteristic (P : Polyhedron) : P.F + P.V - P.E = 2 := by
  sorry

end space_convex_polyhedron_euler_characteristic_l778_778757


namespace sum_problem_l778_778697

variable {α : Type*} [linear_ordered_field α] {a d : α}

-- Definition of S_n
def S (n : ℕ) (a d : α) : α := n * (2 * a + (n - 1) * d) / 2

-- Definitions of a_n based on a_1, d and the term position
def a_n (a d : α) (n : ℕ) : α := a + (n - 1) * d

theorem sum_problem
  (h : S 9 a d = 18) :
  a_n a d 3 + a_n a d 5 + a_n a d 7 = 6 :=
by
  sorry

end sum_problem_l778_778697


namespace limit_of_sequence_l778_778624

noncomputable def a_n (n : ℕ) : ℝ := (2^n + 7^n) / (2^n - 7^(n-1))

theorem limit_of_sequence :
  (filter.at_top.map a_n).tendsto at_bot :=
begin
  sorry
end

end limit_of_sequence_l778_778624


namespace imo_1986_p3_l778_778427

def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

def elements : set ℤ := {2, 5, 13}

noncomputable def contains_non_square_prod (d : ℤ) : Prop :=
  d > 0 ∧ d ≠ 2 ∧ d ≠ 5 ∧ d ≠ 13 ∧
  (∃ a ∈ (elements ∪ {d} : set ℤ), 
   ∃ b ∈ (elements ∪ {d} : set ℤ),
   a ≠ b ∧ ¬is_perfect_square (a * b - 1))

theorem imo_1986_p3 (d : ℤ) : contains_non_square_prod d := 
sorry

end imo_1986_p3_l778_778427


namespace triangle_inequality_proof_l778_778909

/-
  Given 8 points in space, no four of which are coplanar, and 17 line segments connecting these points,
  prove that there exist three segments with lengths a, b, and c such that:
  (a^2 + b^2 + c^2) / 4 >= sqrt(3 * p * (p - a) * (p - b) * (p - c)),
  where p = (a + b + c) / 2.
-/
theorem triangle_inequality_proof :
  ∃ (a b c : ℝ), (∀ (points : set ℝ^3), points.card = 8 ∧ (∀ (s : finset ℝ^3), s.card = 4 → ¬affine_independent ℝ (s : set ℝ^3)) ∧ ∃ (edges : finset (ℝ^3 × ℝ^3)), edges.card = 17 ∧ (∀ (e : ℝ^3 × ℝ^3), e ∈ edges → e.1 ≠ e.2 ∧ e.1 ∈ points ∧ e.2 ∈ points)) →
  let p := (a + b + c) / 2 in
  (a^2 + b^2 + c^2) / 4 ≥ sqrt (3 * p * (p - a) * (p - b) * (p - c)) :=
by
  sorry

end triangle_inequality_proof_l778_778909


namespace tan_2x_period_l778_778853

-- Define the tangent function and its properties
def tan := Real.tan

-- Define the problem
theorem tan_2x_period :
  (∃ P > 0, ∀ x, tan (2 * x) = tan (2 * x + P)) → P = π / 2 :=
by
  sorry

end tan_2x_period_l778_778853


namespace speed_difference_l778_778777

theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no_traffic : ℝ) (d : distance = 200) (th : time_heavy = 5) (tn : time_no_traffic = 4) :
  (distance / time_no_traffic) - (distance / time_heavy) = 10 :=
by
  -- Proof goes here
  sorry

end speed_difference_l778_778777


namespace circle_radius_tangent_to_semicircles_and_sides_l778_778976

noncomputable def side_length_of_square : ℝ := 4
noncomputable def side_length_of_smaller_square : ℝ := side_length_of_square / 2
noncomputable def radius_of_semicircle : ℝ := side_length_of_smaller_square / 2
noncomputable def distance_from_center_to_tangent_point : ℝ := Real.sqrt (side_length_of_smaller_square^2 + radius_of_semicircle^2)

theorem circle_radius_tangent_to_semicircles_and_sides : 
  ∃ (r : ℝ), r = (Real.sqrt 5 - 1) / 2 :=
by
  have r : ℝ := (Real.sqrt 5 - 1) / 2
  use r
  sorry -- Proof omitted

end circle_radius_tangent_to_semicircles_and_sides_l778_778976


namespace slices_in_each_loaf_l778_778444

-- Define the conditions
def slices_per_weekend (s: ℕ) (h: ℕ) (d: ℕ) : ℕ := s + h + d

def total_slices_in_52_weeks (slices_per_weekend: ℕ): ℕ := slices_per_weekend * 52

def slices_per_loaf (total_slices: ℕ) (loaves: ℕ): ℕ := total_slices / loaves

-- Define the question as a statement
theorem slices_in_each_loaf :
  let s := 1, h := 1, d := 1,
      slices_per_weekend := slices_per_weekend s h d,
      total_slices := total_slices_in_52_weeks slices_per_weekend,
      loaves := 26
  in slices_per_loaf total_slices loaves = 6 := by
  -- Insert proof here
  sorry

end slices_in_each_loaf_l778_778444


namespace range_of_t_l778_778705

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

variable {f : ℝ → ℝ}

theorem range_of_t (h_odd : odd_function f) 
  (h_decreasing : decreasing_function f)
  (h_ineq : ∀ t, -1 < t → t < 1 → f (1 - t) + f (1 - t^2) < 0) 
  : ∀ t, 0 < t → t < 1 :=
by sorry

end range_of_t_l778_778705


namespace ellipse_and_fixed_point_l778_778289

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778289


namespace find_p_l778_778353

-- Definitions based on the given conditions
def hyperbola : set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 / 4 - y^2 / 12 = 1 }
def parabola (p : ℝ) : set (ℝ × ℝ) := { q | ∃ x y, q = (x, y) ∧ x = 2 * p * y^2 }
def focus := (2, 0)

-- Eccentricity of the hyperbola
def eccentricity (a c : ℝ) := c / a

theorem find_p (p : ℝ) : 
  (∃ a b c : ℝ, a = 2 ∧ b = 2 * real.sqrt 3 ∧ c^2 = a^2 + b^2 ∧ (eccentricity a c) = 2) 
  → p = 1 / 16 :=
by
  sorry

end find_p_l778_778353


namespace oz_max_score_l778_778768

theorem oz_max_score :
  ∀ (scores : list (list ℤ)), (∀ s ∈ scores, s.length = 3 ∧ ∀ x ∈ s, x = 0 ∨ x = 1)
  → (∃ (triple_count : ℕ), triple_count = scores.length ∧
                            (∀ (t ∈ scores), t.sum ∈ {0, 1, 2, 3}) ∧
                            triple_count = 3 ∧
                            (∀ (t ∈ scores), t.sum = 3 → round (t.sum / 3 : ℝ) = 1) ∧
                            (∀ (t ∈ scores), t.sum = 0 → round (t.sum / 3 : ℝ) = 0))
  → (∑ t in scores, round (t.sum / 3 : ℝ)) = 4 :=
by
  intros scores hscores htriple_count
  sorry

end oz_max_score_l778_778768


namespace no_finite_true_statements_no_finite_false_statements_l778_778942

-- Definitions based on the conditions
def is_normal_person (person : Type) : Prop := 
  true -- Normal person can tell both true and false statements

def is_knight (person : Type) : Prop := 
  ∀ (statement : Prop), statement = true

def is_liar (person : Type) : Prop := 
  ∀ (statement : Prop), statement = false

def not_normal (person : Type) : Prop :=
  is_knight person ∨ is_liar person

-- Propositions based on the questions and correct answers
theorem no_finite_true_statements (n : ℕ) :
  ¬ (∃ (person : Type) (statements : fin n → Prop), 
    (∀ i, statements i = true) ∧ not_normal person) :=
sorry

theorem no_finite_false_statements (n : ℕ) :
  ¬ (∃ (person : Type) (statements : fin n → Prop), 
    (∀ i, statements i = false) ∧ not_normal person) :=
sorry

end no_finite_true_statements_no_finite_false_statements_l778_778942


namespace correct_time_fraction_l778_778563

-- Definitions based on conditions
def is_incorrect_hour (h : ℕ) : Prop := (h = 2) ∨ (h = 12)
def is_incorrect_minute (m : ℕ) : Prop := (m = 20 ∨ m = 21 ∨ m = 22 ∨ m = 23 ∨ m = 24 ∨ m = 25 ∨ m = 26 ∨ m = 27 ∨ m = 28 ∨ m = 29 ∨ m = 2 ∨ m = 12 ∨ m = 32 ∨ m = 42 ∨ m = 52)

-- The proof problem
theorem correct_time_fraction : 
  (let correct_hours := (12 - 2) / 12 in
   let correct_minutes := (60 - 15) / 60 in
   correct_hours * correct_minutes = 5 / 8) := sorry

end correct_time_fraction_l778_778563


namespace count_graphs_l778_778366

open Finset

def is_triangle_free (G : SimpleGraph (Fin 10)) : Prop :=
  ∀ (v1 v2 v3 : Fin 10), G.adj v1 v2 → G.adj v2 v3 → G.adj v3 v1 → False

def satisfies_conditions (G : SimpleGraph (Fin 10)) : Prop :=
  G.edgeSet.card = 23 ∧ is_triangle_free G

theorem count_graphs (G : SimpleGraph (Fin 10)) :
  ∃! G, satisfies_conditions G :=
sorry

end count_graphs_l778_778366


namespace probability_x_plus_y_leq_5_is_3_over_5_l778_778143

noncomputable def probability_in_rectangle (x y : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5 then 1 / 20 else 0

noncomputable def probability_x_plus_y_leq_5 (x y : ℝ) : ℝ :=
  if x + y ≤ 5 then probability_in_rectangle x y else 0

theorem probability_x_plus_y_leq_5_is_3_over_5 :
  (∫∫ (x y : ℝ), probability_x_plus_y_leq_5 x y) = 3 / 5 :=
sorry

end probability_x_plus_y_leq_5_is_3_over_5_l778_778143


namespace mixed_groups_count_l778_778894

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l778_778894


namespace probability_heads_l778_778823

theorem probability_heads :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2^2 in
  (successful_outcomes / total_outcomes : ℚ) = 1 / 8 :=
by
  let total_outcomes := 2^5
  let successful_outcomes := 2^2
  have h : (successful_outcomes : ℚ) / total_outcomes = 1 / 8 := sorry
  exact h

end probability_heads_l778_778823


namespace polynomial_properties_l778_778653

noncomputable def p (x : ℤ) : ℤ := x^3 + 2 * x^2 - 2 * x

theorem polynomial_properties :
  (∀ x, p x ∈ ℤ) ∧
  (∀ a b c : ℤ, (x^3 + a * x^2 + b * x + c = p x) -> (p 0 = 0) -> (p 1 = 1) -> (p (-1) = 3)) ∧
  (p 0 = 0) ∧
  (p 1 = 1) ∧
  (p (-1) = 3) :=
by
  sorry

end polynomial_properties_l778_778653


namespace slope_angle_of_line_l778_778047

theorem slope_angle_of_line (a : ℝ) (h : atan a = π / 4) : a = 1 :=
sorry

end slope_angle_of_line_l778_778047


namespace kristin_annual_income_l778_778873

theorem kristin_annual_income (p : ℝ) :
  ∃ A : ℝ, 
  (0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = (0.01 * (p + 0.25) * A)) ∧
  A = 32000 :=
by
  sorry

end kristin_annual_income_l778_778873


namespace library_books_difference_proof_l778_778036

def initial_science_books : ℕ := 400
def initial_history_books : ℕ := 250
def second_graders_borrowed_percentage : ℚ := 0.30
def third_graders_reserved_percentage : ℚ := 0.40

def second_graders_borrowed_books : ℕ := (second_graders_borrowed_percentage * initial_science_books).toNat
def third_graders_reserved_books : ℕ := (third_graders_reserved_percentage * initial_history_books).toNat

def total_books_initial : ℕ := initial_science_books + initial_history_books
def total_books_borrowed_reserved : ℕ := second_graders_borrowed_books + third_graders_reserved_books
def difference_of_books : ℕ := total_books_initial - total_books_borrowed_reserved

theorem library_books_difference_proof : difference_of_books = 430 :=
by
  -- Proof omitted
  sorry

end library_books_difference_proof_l778_778036


namespace total_books_l778_778174

theorem total_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total : ℕ) :
  books_per_shelf = 56 → 
  num_shelves = 9 → 
  total = 504 →
  books_per_shelf * num_shelves = total :=
by
  intros h1 h2 h3
  rw [h1, h2]
  exact h3

end total_books_l778_778174


namespace base_14_div_9_remainder_zero_l778_778092

noncomputable def base_14_to_decimal (n : ℕ) : ℕ :=
  let digits := [1, 7, 2, 5]
  digits.foldl (λ acc d, acc * 14 + d) 0

theorem base_14_div_9_remainder_zero : base_14_to_decimal 1725 % 9 = 0 := by
  sorry

end base_14_div_9_remainder_zero_l778_778092


namespace remainder_101_pow_50_mod_100_l778_778518

theorem remainder_101_pow_50_mod_100 : (101 ^ 50) % 100 = 1 := by
  sorry

end remainder_101_pow_50_mod_100_l778_778518


namespace probability_correct_l778_778524

open Finset

def standard_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

noncomputable def probability_abc_144 : ℚ :=
  let outcomes := (standard_die × standard_die × standard_die).filter (λ (t : ℕ × ℕ × ℕ), t.1 * t.2 * t.3 = 144)
  1 / 6 * 1 / 6 * 1 / 6 * outcomes.card

theorem probability_correct : probability_abc_144 = 1 / 72 := by
  unfold probability_abc_144
  sorry

end probability_correct_l778_778524


namespace pipe_B_fill_time_l778_778503

theorem pipe_B_fill_time (t : ℝ) :
  (1/10) + (2/t) - (2/15) = 1 ↔ t = 60/31 :=
by
  sorry

end pipe_B_fill_time_l778_778503


namespace line_slope_and_angle_l778_778666

open Real

-- Define the points C and D
def C (m n : ℝ) := (m, n)
def D (m n : ℝ) := (m, -n)

-- Definition of undefined slope and angle of inclination 90 degrees
def slope_undefined (m n : ℝ) := ∀ x : ℝ, (C m n).1 = (D m (-n)).1 → False
def angle_of_inclination_90 (θ : ℝ) := θ = 90

theorem line_slope_and_angle (m n : ℝ) (h : n ≠ 0) :
  slope_undefined m n ∧ angle_of_inclination_90 90 :=
by
  split
  -- Prove that the slope of the line is undefined
  case h1 =>
    intros x hx
    rw [C, D] at hx
    simp at hx
    contradiction
  -- Prove that the angle of inclination is 90 degrees
  case h2 =>
    exact rfl

end line_slope_and_angle_l778_778666


namespace problem_sum_decomposition_l778_778791

theorem problem_sum_decomposition :
  let T := ∑ n in Finset.range 9999, (λ n, 3 / (Real.sqrt (n + 2 + Real.sqrt ((n + 2)^2 - 1))))
  ∃ (a b c : ℕ), (T = a + b * Real.sqrt c) ∧ (c ≠ 0) ∧ (∀ p, (Nat.prime p ∧ p * p ∣ b) → False) ∧ (a + b + c = 364) :=
by 
  let T := ∑ n in Finset.range 9999, (λ n, 3 / (Real.sqrt (n + 2 + Real.sqrt ((n + 2)^2 - 1))))
  have h_T : T = 212 + 150 * Real.sqrt 2, sorry,
  use (212, 150, 2),
  simp only [h_T],
  split,
  { refl },
  split,
  { exact two_ne_zero.symm },
  split,
  { intros p h_prime h_sq_div,
    exfalso,
    exact h_sq_div },
  { ring },
  sorry

end problem_sum_decomposition_l778_778791


namespace ellipse_equation_proof_HN_fixed_point_l778_778323

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778323


namespace triangle_angles_arithmetic_progression_l778_778471

theorem triangle_angles_arithmetic_progression (α β γ : ℝ) (a c : ℝ) :
  (α < β) ∧ (β < γ) ∧ (α + β + γ = 180) ∧
  (∃ x : ℝ, β = α + x ∧ γ = β + x) ∧
  (a = c / 2) → 
  (α = 30) ∧ (β = 60) ∧ (γ = 90) :=
by
  intros h
  sorry

end triangle_angles_arithmetic_progression_l778_778471


namespace chess_club_mixed_groups_l778_778885

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l778_778885


namespace complex_number_problem_l778_778340

open Complex -- Open the complex numbers namespace

theorem complex_number_problem 
  (z1 z2 : ℂ) 
  (h_z1 : z1 = 2 - I) 
  (h_z2 : z2 = -I) : 
  z1 / z2 + Complex.abs z2 = 2 + 2 * I := by
-- Definitions and conditions directly from (a)
  rw [h_z1, h_z2] -- Replace z1 and z2 with their given values
  sorry -- Proof to be filled in place of the solution steps

end complex_number_problem_l778_778340


namespace max_mn_value_min_4m_square_n_square_l778_778737

variable {m n : ℝ}
variable (h_cond1 : m > 0)
variable (h_cond2 : n > 0)
variable (h_eq : 2 * m + n = 1)

theorem max_mn_value : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ m * n = 1/8) := 
  sorry

theorem min_4m_square_n_square : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ 4 * m^2 + n^2 = 1/2) := 
  sorry

end max_mn_value_min_4m_square_n_square_l778_778737


namespace f_x1_positive_l778_778832

noncomputable def f : ℝ → ℝ := λ x, (1 / 3) ^ x - Real.log x / Real.log 2

theorem f_x1_positive (x0 x1 : ℝ) (h0 : f x0 = 0) (h1 : 0 < x1) (h2 : x1 < x0) : f x1 > 0 := by
  -- Monotonicity of f on (0, ∞) not used directly in proof here
  sorry

end f_x1_positive_l778_778832


namespace minimum_trips_proof_l778_778057

noncomputable def minimum_trips (cargo : ℕ) (weight_limit : ℕ) : ℕ :=
  by 
  have h1 : cargo = 400 := sorry -- given condition: there are 400 tons of cargo
  have h2 : weight_limit = 10 := sorry -- given condition: truck capacity is 10 tons
  have h3 : ∀ w, w % 1 = 0 ∧ w ≤ 10 ∧ (distinct_weights w) := sorry -- conditions: weights are multiples of quintals, distinct, and ≤ 10 tons
  sorry -- prove that minimum trips is 51

theorem minimum_trips_proof :
  minimum_trips 400 10 = 51 := 
  sorry

end minimum_trips_proof_l778_778057


namespace operation_is_commutative_and_associative_l778_778459

variables {S : Type} (op : S → S → S)

-- defining the properties given in the conditions
def idempotent (op : S → S → S) : Prop :=
  ∀ (a : S), op a a = a

def medial (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op (op b c) a

-- defining commutative and associative properties
def commutative (op : S → S → S) : Prop :=
  ∀ (a b : S), op a b = op b a

def associative (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op a (op b c)

-- statement of the theorem to prove
theorem operation_is_commutative_and_associative 
  (idemp : idempotent op) 
  (med : medial op) : commutative op ∧ associative op :=
sorry

end operation_is_commutative_and_associative_l778_778459


namespace minimum_trips_l778_778062

/-- 
Problem: There are 400 tons of cargo stored in a warehouse, each with a weight that is a multiple 
of a quintal and does not exceed 10 tons. It is known that any two pieces of cargo have different 
weights. Prove that the minimum number of trips a 10-ton truck must make to guarantee the 
transportation of all this cargo from the warehouse is 51.
-/

theorem minimum_trips (n : ℕ) (weights : fin n → ℕ)
  (h1 : ∀ i, weights i % 10 = 0)
  (h2 : ∀ i j, i ≠ j → weights i ≠ weights j)
  (h3 : ∑ i, weights i = 4000)
  (h4 : ∀ i, weights i ≤ 100) :
  ∃ t : ℕ, t = 51 ∧ ∑ i in finset.range t, weights i ≤ 10000 :=
sorry

end minimum_trips_l778_778062


namespace sum_of_adjacent_to_five_is_seven_l778_778029

variable (n : ℕ)
variable (circle : ℕ → ℕ)

axiom pentagon_placement : ∀ i, 1 ≤ circle i ∧ circle i ≤ 5
axiom all_diff : ∀ i j, i ≠ j → circle i ≠ circle j
axiom diff_by_more_than_one : ∀ i j, (i ≈ j) → abs (circle i - circle j) > 1
axiom contains_five : ∃ i, circle i = 5
axiom adjacent : (5 ≈ 4) ∨ (5 ≈ 3)

theorem sum_of_adjacent_to_five_is_seven : 
  (∃ i j, circle i = 5 ∧ (i ≈ j) ∧ (circle j = 4 ∨ circle j = 3)) → 
  ∃ k l, circle k = 5 ∧ (adjacent k l) → circle (adjacent_circle k l) = 7 :=
sorry

end sum_of_adjacent_to_five_is_seven_l778_778029


namespace three_digit_number_permutation_sum_l778_778152

theorem three_digit_number_permutation_sum (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) (h2 : ∀ d ∈ (nat.digits 10 n), d ≠ 0) 
(h3 : ∑ m in (list.permutations (nat.digits 10 n)).erase_dup, nat.of_digits 10 m = 2775) : 
n = 889 ∨ n = 997 :=
sorry

end three_digit_number_permutation_sum_l778_778152


namespace minimum_trips_proof_l778_778058

noncomputable def minimum_trips (cargo : ℕ) (weight_limit : ℕ) : ℕ :=
  by 
  have h1 : cargo = 400 := sorry -- given condition: there are 400 tons of cargo
  have h2 : weight_limit = 10 := sorry -- given condition: truck capacity is 10 tons
  have h3 : ∀ w, w % 1 = 0 ∧ w ≤ 10 ∧ (distinct_weights w) := sorry -- conditions: weights are multiples of quintals, distinct, and ≤ 10 tons
  sorry -- prove that minimum trips is 51

theorem minimum_trips_proof :
  minimum_trips 400 10 = 51 := 
  sorry

end minimum_trips_proof_l778_778058


namespace exists_at_least_n_ubiquitous_words_l778_778429

open Nat

def isPeriodicWord (W : ℤ → char) (N : ℕ) : Prop :=
  ∀ i, W (i + N) = W i

def appearsInPeriod (U : List char) (W : ℤ → char) (k ℓ : ℤ) : Prop :=
  U = List.map W (List.range (ℓ - k + 1))

def isUbiquitous (U : List char) (W : ℤ → char) : Prop :=
  (∃ k ℓ, appearsInPeriod (U ++ ['a']) W k ℓ) ∧
  (∃ k ℓ, appearsInPeriod (U ++ ['b']) W k ℓ) ∧
  (∃ k ℓ, appearsInPeriod ('a' :: U) W k ℓ) ∧
  (∃ k ℓ, appearsInPeriod ('b' :: U) W k ℓ)

theorem exists_at_least_n_ubiquitous_words (n : ℕ) (W : ℤ → char) (N : ℕ) :
  0 < n → 2^n < N → isPeriodicWord W N →
  ∃ (U : Fin n → List char), (∀ i, U i ≠ []) ∧ (∀ i, isUbiquitous (U i) W) ∧ (Function.Injective U) :=
by
  sorry

end exists_at_least_n_ubiquitous_words_l778_778429


namespace remainder_sum_first_150_div_11300_l778_778087

theorem remainder_sum_first_150_div_11300 :
  let n := 150 in
  let S := n * (n + 1) / 2 in
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l778_778087


namespace count_polynomials_degree_at_most_3_l778_778185

theorem count_polynomials_degree_at_most_3 :
  let coeffs := {0, 1, 2, 3, 4},
      P := λ (a b c d : ℤ), a * (-1)^3 + b * (-1)^2 + c * (-1) + d = 1 in
    (∃ (a b c d : ℤ), a ∈ coeffs ∧ b ∈ coeffs ∧ c ∈ coeffs ∧ d ∈ coeffs ∧ P a b c d) ∧
    ((finset.univ.filter (λ (abcd : ℤ × ℤ × ℤ × ℤ),
      abcd.1 ∈ coeffs ∧ abcd.2.1 ∈ coeffs ∧ abcd.2.2.1 ∈ coeffs ∧ abcd.2.2.2 ∈ coeffs ∧
      P abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2)).card = 80) :=
by {
  let coeffs := {0, 1, 2, 3, 4},
  let P := λ (a b c d : ℤ), a * (-1)^3 + b * (-1)^2 + c * (-1) + d = 1,
  have h : (∃ (a b c d : ℤ), a ∈ coeffs ∧ b ∈ coeffs ∧ c ∈ coeffs ∧ d ∈ coeffs ∧ P a b c d),
  { sorry },
  have hc : (finset.univ.filter (λ (abcd : ℤ × ℤ × ℤ × ℤ),
    abcd.1 ∈ coeffs ∧ abcd.2.1 ∈ coeffs ∧ abcd.2.2.1 ∈ coeffs ∧ abcd.2.2.2 ∈ coeffs ∧
    P abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2)).card = 80,
  { sorry },
  exact ⟨h, hc⟩
}

end count_polynomials_degree_at_most_3_l778_778185


namespace ellipse_solution_geometry_solution_l778_778308

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778308


namespace nth_root_not_fractional_l778_778043

theorem nth_root_not_fractional (n A : ℕ) (hn : n > 0) (hA : A > 0) : 
  ∀ p q : ℕ, (nat.gcd p q = 1) → (∃ r : ℕ, (p * r) ^ n = A * q ^ n) → q = 1 :=
by sorry

end nth_root_not_fractional_l778_778043


namespace find_angle_BXY_l778_778398

theorem find_angle_BXY (AB CD : Set Point) (AXE CYX : ℝ) (h1 : AB ∥ CD) (h2 : AXE = 2 * CYX - 72) :
  BXY = 72 :=
sorry

end find_angle_BXY_l778_778398


namespace necessary_not_sufficient_for_circle_l778_778028

theorem necessary_not_sufficient_for_circle (a : ℝ) :
  (a ≤ 2 → (x^2 + y^2 - 2*x + 2*y + a = 0 → ∃ r : ℝ, r > 0)) ∧
  (a ≤ 2 ∧ ∃ b, b < 2 → a = b) := sorry

end necessary_not_sufficient_for_circle_l778_778028


namespace modulo_inverse_problem_l778_778423

open BigOperators

theorem modulo_inverse_problem :
  let b := ((7 + 8 + 8)⁻¹ : ℤ) in (b % 13 = 4) :=
sorry

end modulo_inverse_problem_l778_778423


namespace chessboard_movement_l778_778016

-- Defining the problem as described in the transformed proof problem

theorem chessboard_movement (pieces : Nat) (adjacent_empty_square : Nat → Nat → Bool) (visited_all_squares : Nat → Bool)
  (returns_to_starting_square : Nat → Bool) :
  (∃ (moment : Nat), ∀ (piece : Nat), ¬ returns_to_starting_square piece) :=
by
  -- Here we state that there exists a moment when each piece (checker) is not on its starting square
  sorry

end chessboard_movement_l778_778016


namespace days_elapsed_l778_778994

theorem days_elapsed : ∃ d : ℕ, (2 * 35 * d + 10 = 500) ∧ (d = 7) :=
by {
  use 7,
  split,
  { linarith, },
  { refl, }
}

end days_elapsed_l778_778994


namespace count_four_digit_numbers_with_2_and_3_l778_778629

theorem count_four_digit_numbers_with_2_and_3 : 
  -- Define the set of digits and the length of number
  ∀ (digits : set ℕ), 
  digits = {2, 3} → 
  ∀ (len : ℕ), 
  len = 4 → 
  -- Ensure that each digit appears at least once
  (∀ n : list ℕ, n.length = len → (∀ d ∈ digits, ∃ x ∈ n, x = d)) → 
  -- Define the condition that we must have only these digits
  (∀ n : list ℕ, n.length = len → (∀ x ∈ n, x ∈ digits)) → 
  -- The conclusion is that the number of valid configurations is 14
  (∃ n : ℕ, n = 14) :=
begin
  intros digits h_digits len h_len has_each appears_in_digits,
  use 14,
  sorry -- proof to be completed
end

end count_four_digit_numbers_with_2_and_3_l778_778629


namespace min_n_101_l778_778447

noncomputable def find_minimum_n (n : ℕ) (T : Matrix ℕ ℕ (Fin n → Fin n → Bool)) : ℕ :=
  if h : ∃ p : List (Fin n → Fin n → Prop), unique_partition T p ∧ 
                                             (∀ r ∈ p, counts_blue_squares r = 1)
  then 101
  else n

theorem min_n_101 {n : ℕ} (T : Matrix ℕ ℕ (Fin n → Fin n → Bool)) :
  (∃ p : List (Fin n → Fin n → Prop), unique_partition T p ∧ 
                                       (∀ r ∈ p, counts_blue_squares r = 1)
   ∧ counts_blue_squares T = 101) → (find_minimum_n n T = 101) :=
by
  sorry

/-- Definitions used in the Lean theorem -/

def counts_blue_squares (T : Matrix ℕ ℕ (Fin n → Fin n → Bool)) : ℕ := 
  ∑ (i j : Fin n), if T i j then 1 else 0

def unique_partition (T : Matrix ℕ ℕ (Fin n → Fin n → Bool)) (p : List (Fin n → Fin n → Prop)) : Prop := 
  -- A formal definition to ensure a partition is unique
  sorry

end min_n_101_l778_778447


namespace max_volume_frustum_l778_778953

noncomputable def volume_frustum (S₁ S₂ h : ℝ) : ℝ :=
  (1 / 3) * h * (S₁ + S₂ + real.sqrt (S₁ * S₂))

noncomputable def lower_base_area (α : ℝ) : ℝ :=
  2 * ( (real.sqrt 2 / 2) + real.cos α ) ^ 2

noncomputable def height (α : ℝ) : ℝ := real.sin α

noncomputable def volume_expr (α : ℝ) : ℝ :=
  volume_frustum 1 (lower_base_area α) (height α)

theorem max_volume_frustum : ∃ α, volume_expr α = 1.674 :=
sorry

end max_volume_frustum_l778_778953


namespace triangle_count_l778_778808

theorem triangle_count (points : Finset ℕ) (h_card : points.card = 9) (distinct_points : ∀ i j k ∈ points, (i ≠ j ∧ j ≠ k ∧ k ≠ i) → (i + 1 ≠ j ∨ j + 1 ≠ k ∨ k + 1 ≠ i)) : 
  Finset.card {T : Finset (Fin ℕ) | T ⊆ points ∧ T.card = 3 ∧ ∀ i j k ∈ T, i + 1 ≠ j ∨ j + 1 ≠ k ∨ k + 1 ≠ i} = 75 :=
sorry

end triangle_count_l778_778808


namespace ellipse_and_fixed_point_l778_778299

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778299


namespace minimum_period_tan_2x_l778_778861

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem minimum_period_tan_2x : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi / 2 :=
by
  sorry

end minimum_period_tan_2x_l778_778861


namespace math_problem_l778_778246

variables (a b : ℝ)
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variables (h_condition : a + 2 * b = 1)

noncomputable def ab_max_value : Prop := ab ≤ 1 / 8 :=
begin
  -- Proof goes here, skipped with sorry
  sorry
end

noncomputable def not_a_inv_plus_2b_inv_has_min_value_8 : Prop := ¬ (1 / a + 2 / b ≥ 8) :=
begin
  -- Proof goes here, skipped with sorry
  sorry
end

noncomputable def b_inv_plus_ba_has_min_value_4 : Prop := 1 / b + b / a ≥ 4 :=
begin
  -- Proof goes here, skipped with sorry
  sorry
end

noncomputable def a_squared_plus_b_squared_min_value : Prop := ∃ inf, inf = 1 / 5 ∧ ∀ ε > 0, a^2 + b^2 > 1 / 5 - ε :=
begin
  -- Proof goes here, skipped with sorry
  sorry
end

theorem math_problem :
  ab_max_value a b h_pos_a h_pos_b h_condition ∧
  not_a_inv_plus_2b_inv_has_min_value_8 a b h_pos_a h_pos_b h_condition ∧
  b_inv_plus_ba_has_min_value_4 a b h_pos_a h_pos_b h_condition ∧
  a_squared_plus_b_squared_min_value a b h_pos_a h_pos_b h_condition :=
begin
  -- Proof goes here, skipped with sorry
  sorry
end

end math_problem_l778_778246


namespace hn_passes_fixed_point_l778_778266

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778266


namespace supermarket_mooncakes_proof_l778_778197

-- Define the variables and conditions
variables (a b x m : ℕ) (priceA_before priceB_before : ℚ)

-- Conditions
def purchase_conditions : Prop :=
  (a = 4 * b) ∧
  (priceA_before * a = 1200) ∧ 
  (priceB_before * b = 600) ∧
  (priceA_before + priceB_before = 9)

-- Results from Part 1
def results_part1 : Prop :=
  a = 400 ∧ b = 100

-- Conditions for Part 2
def promotions_conditions : Prop :=
  (priceA_before = 3) ∧ 
  (priceB_before = 6) ∧
  let priceA_after := priceA_before * (1 - 1/3 : ℚ) in
  let priceB_after := priceB_before - m / 5 in
  let increased_b := b + 15 / 2 * m in
  let decreased_total_cost := (priceA_after * a + priceB_after * increased_b = 1400 - 2 * m) in
  decreased_total_cost

-- Result from Part 2
def result_part2 : Prop :=
  m = 8

-- Final proof problem
theorem supermarket_mooncakes_proof :
  purchase_conditions a b priceA_before priceB_before →
  results_part1 a b →
  promotions_conditions a b priceA_before priceB_before m →
  result_part2 m :=
sorry

end supermarket_mooncakes_proof_l778_778197


namespace length_BD_l778_778928

theorem length_BD (P A C B D : Point) (α β : Plane)
  (h1 : ∀ {X}, X ∈ α ∨ X ∈ β → P ≠ X ∧ X != A ∧ X != B ∧ X != C ∧ X != D)
  (h2 : A ∈ α) (h3 : B ∈ β) (h4 : C ∈ α) (h5 : D ∈ β)
  (h6 : line_through P A) (h7 : line_through P B) (h8 : line_through P C) (h9 : line_through P D)
  (h10 : dist P A = 6) (h11 : dist A C = 9) (h12 : dist P B = 8) :
  dist B D = 12 := sorry

end length_BD_l778_778928


namespace final_value_ge_one_over_n_l778_778811

theorem final_value_ge_one_over_n (n : ℕ) (h : n ≥ 1) :
    ∃ x ≥ 0, (x ∈ set.range (λ i : ℕ, 1)) ∧ (∀ a b ∈ set.range (λ i : ℕ, 1),
    (a ≠ b → ∃ c, c = (a + b) / 4)) ∧ x ≥ 1 / n := by sorry

end final_value_ge_one_over_n_l778_778811


namespace annual_income_is_32000_l778_778871

noncomputable def compute_tax (p A: ℝ) : ℝ := 
  0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000)

noncomputable def stated_tax (p A: ℝ) : ℝ := 
  0.01 * (p + 0.25) * A

theorem annual_income_is_32000 (p : ℝ) (A : ℝ) :
  compute_tax p A = stated_tax p A → A = 32000 :=
by
  intros h
  have : 0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = 0.01 * (p + 0.25) * A := h
  sorry

end annual_income_is_32000_l778_778871


namespace selection_assignment_ways_l778_778555

-- Define the group of students
def male_students : ℕ := 4
def female_students : ℕ := 3

-- Define the selection conditions
def selected_people : ℕ := 4
def min_females : ℕ := 2

-- Prove the number of different ways to select and assign the individuals
theorem selection_assignment_ways : (C 4 2 * C 3 2 + C 3 3 * C 4 1) * (C 4 2 * A 3 3) = 792 :=
by
  -- Sorry to omit the proof details
  sorry

end selection_assignment_ways_l778_778555


namespace mixed_groups_count_l778_778897

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l778_778897


namespace product_of_all_possible_values_of_c_l778_778794

theorem product_of_all_possible_values_of_c (b c : ℝ) 
  (h_b_eq_c_plus_one : b = c + 1)
  (h_poly_has_one_root : ∃ x, x^2 + b * x + c = 0 ∧ ∀ y, y^2 + b * y + c = 0 → y = x) :
  ∏ c in {c}, c = 1 :=
by
  sorry

end product_of_all_possible_values_of_c_l778_778794


namespace ending_number_of_set_Y_l778_778015

theorem ending_number_of_set_Y :
  ∃ (Y_end : ℕ),
    ∀ X Y : set ℕ,
      (X = { n | 1 ≤ n ∧ n ≤ 12 }) →
      (Y = { n | 0 ≤ n ∧ n ≤ Y_end }) →
      (∃ S : finset ℕ, S.card = 12 ∧ ↑S ⊆ X ∧ ↑S ⊆ Y) →
      Y_end = 12 :=
begin
  sorry,
end

end ending_number_of_set_Y_l778_778015


namespace flowers_per_bouquet_l778_778081

theorem flowers_per_bouquet (total_flowers wilted_flowers : ℕ) (bouquets : ℕ) (remaining_flowers : ℕ)
    (h1 : total_flowers = 45)
    (h2 : wilted_flowers = 35)
    (h3 : bouquets = 2)
    (h4 : remaining_flowers = total_flowers - wilted_flowers)
    (h5 : bouquets * (remaining_flowers / bouquets) = remaining_flowers) :
  remaining_flowers / bouquets = 5 :=
by
  sorry

end flowers_per_bouquet_l778_778081


namespace pentagon_octagon_position_l778_778570

open Real

-- Definitions and assumptions
def regular_polygon_angle (n : ℕ) : Real :=
  (n - 2) * 180 / n

def rotation_per_movement (angle_octagon angle_pentagon : Real) : Real :=
  360 - (angle_octagon + angle_pentagon)

def total_rotation (movements : ℕ) (rotation : Real) : Real :=
  movements * rotation

def final_position (rotation : Real) : String :=
  if rotation % 360 == 0 then "At the starting vertex"
  else if rotation % 360 == 180 then "Opposite to the starting vertex"
  else if rotation % 360 < 180 then "Right next to the starting vertex in clockwise direction"
  else "Right next to the starting vertex in counter-clockwise direction"

theorem pentagon_octagon_position :
  let angle_octagon := regular_polygon_angle 8 in
  let angle_pentagon := regular_polygon_angle 5 in
  let rotation := rotation_per_movement angle_octagon angle_pentagon in
  let total_rot := total_rotation 3 rotation in
  final_position total_rot = "Right next to the starting vertex in clockwise direction" :=
by
  sorry

end pentagon_octagon_position_l778_778570


namespace mixed_groups_count_l778_778903

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l778_778903


namespace remainder_of_division_l778_778388

theorem remainder_of_division (dividend divisor quotient remainder : ℕ)
  (h1 : dividend = 55053)
  (h2 : divisor = 456)
  (h3 : quotient = 120)
  (h4 : remainder = dividend - divisor * quotient) : 
  remainder = 333 := by
  sorry

end remainder_of_division_l778_778388


namespace mixed_groups_count_l778_778900

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l778_778900


namespace least_cost_l778_778602

def total_students : Nat := 25
def vampire_students : Nat := 11
def pumpkin_students : Nat := 14
def pack_size : Nat := 5
def pack_cost : Nat := 3
def individual_bag_cost : Nat := 1

/- 
  Theorem: Given 25 students, where 11 want vampire-themed bags and 14 want pumpkin-themed bags,
  and the cost structure of the bags, the least amount of money the teacher can spend 
  to satisfy the preferences is $17.
-/
theorem least_cost (total_students = 25) (vampire_students = 11) (pumpkin_students = 14) 
  (pack_size = 5) (pack_cost = 3) (individual_bag_cost = 1) : 
  (2 * pack_cost + individual_bag_cost) + (2 * pack_cost + 4 * individual_bag_cost) = 17 :=
by
  sorry

end least_cost_l778_778602


namespace mixed_groups_count_l778_778908

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l778_778908


namespace no_int_solutions_for_equation_l778_778730

theorem no_int_solutions_for_equation : 
  ∀ x y : ℤ, x ^ 2022 + y^2 = 2 * y + 2 → false := 
by
  -- By the given steps in the solution, we can conclude that no integer solutions exist
  sorry

end no_int_solutions_for_equation_l778_778730


namespace general_term_formula_find_T_n_l778_778240

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n, a (n + 1) = q * a n

variables {a : ℕ → ℝ} (h_seq : arithmetic_sequence a)

axiom a1_eq_2 : a 1 = 2
axiom a4_eq_4a6 : a 4 = 4 * a 6
axiom a_pos : ∀ n, a n > 0

def b (n : ℕ) : ℝ := -real.logb 2 (a (2 * n + 1))
def frac_seq (n : ℕ) : ℝ := 2 / (b n * b (n + 1))

noncomputable def T (n : ℕ) : ℝ := ∑ i in finset.range n, frac_seq i

theorem general_term_formula :
  (∀ n, a n = 2 * (1 / 2) ^ (n - 1)) ∨ (∀ n, a n = 2 * (-1 / 2) ^ (n - 1)) :=
sorry

theorem find_T_n :
  ∀ n : ℕ, T n = 2 * (n : ℝ) / (2 * n + 1) :=
sorry

end general_term_formula_find_T_n_l778_778240


namespace rectangle_area_constant_l778_778042

noncomputable def k (d : ℝ) : ℝ :=
  let x := d / Real.sqrt 29
  10 / 29

theorem rectangle_area_constant (d : ℝ) : 
  let k := 10 / 29
  let length := 5 * (d / Real.sqrt 29)
  let width := 2 * (d / Real.sqrt 29)
  let diagonal := d
  let area := length * width
  area = k * d^2 :=
by
  sorry

end rectangle_area_constant_l778_778042


namespace tan_negative_angle_l778_778748

theorem tan_negative_angle (m : ℝ) (h1 : m = Real.cos (80 * Real.pi / 180)) (h2 : m = Real.sin (10 * Real.pi / 180)) :
  Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2)) / m :=
by
  sorry

end tan_negative_angle_l778_778748


namespace chocolate_bars_in_large_box_l778_778531

theorem chocolate_bars_in_large_box:
  ∀ (n m : ℕ), n = 15 → m = 25 → n * m = 375 :=
by
  intro n m hn hm
  rw [hn, hm]
  norm_num

end chocolate_bars_in_large_box_l778_778531


namespace proof_rectangle_folds_to_hexagon_l778_778930

def rectangle_folds_to_hexagon (a b : ℝ) : Prop :=
a = sqrt 3 * b

theorem proof_rectangle_folds_to_hexagon (a b : ℝ) 
  (h : ∃ (x : ℝ), x = (a^2 + b^2) / (2 * a) ∧ 
                  x = (2 * b) / (sqrt 3)) : 
  rectangle_folds_to_hexagon a b :=
by 
  sorry

end proof_rectangle_folds_to_hexagon_l778_778930


namespace mixed_groups_count_l778_778899

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l778_778899


namespace parabola_intercepts_sum_l778_778841

noncomputable def a : ℝ := 5

noncomputable def b : ℝ := (9 + Real.sqrt 21) / 6

noncomputable def c : ℝ := (9 - Real.sqrt 21) / 6

theorem parabola_intercepts_sum : a + b + c = 8 := by
  -- definition of a
  have ha : a = 5 := rfl
  
  -- definitions of b and c from roots of 3y^2 - 9y + 5 = 0
  have hb : b = (9 + Real.sqrt 21) / 6 := rfl
  have hc : c = (9 - Real.sqrt 21) / 6 := rfl
  
  -- Vieta's formulas implies b + c = 3
  have hb_c : b + c = 3 := by
    calc
    b + c = (9 + Real.sqrt 21) / 6 + (9 - Real.sqrt 21) / 6 : by rw [hb, hc]
    ... = (9 + 9) / 6 : by ring
    ... = 18 / 6 : by norm_num
    ... = 3 : by norm_num
  
  -- Sum a + b + c
  calc
  a + b + c = 5 + (b + c) : by rw [ha]
  ... = 5 + 3 : by rw [hb_c]
  ... = 8 : by norm_num

end parabola_intercepts_sum_l778_778841


namespace ellipse_solution_geometry_solution_l778_778311

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778311


namespace trees_ratio_l778_778133

theorem trees_ratio (M T : ℕ) 
  (h1 : 30 * 3 = 90)
  (h2 : 90 - 30 = M)
  (h3 : 80 - M = T)
  (h4 : M = 60)
  (h5 : T = 20) 
  : T / M = 1 / 3 := 
by
  rw [← h2, h4] at h3
  rw [h3, h5]
  exact nat.div_eq_of_eq_mul_left zero_lt_three h3 rfl

end trees_ratio_l778_778133


namespace predicted_whales_l778_778053

theorem predicted_whales (num_last_year num_this_year num_next_year : ℕ)
  (h1 : num_this_year = 2 * num_last_year)
  (h2 : num_last_year = 4000)
  (h3 : num_next_year = 8800) :
  num_next_year - num_this_year = 800 :=
by
  sorry

end predicted_whales_l778_778053


namespace find_four_numbers_l778_778910

theorem find_four_numbers (a b c d : ℕ) (h1 : b^2 = a * c) (h2 : a * b * c = 216) (h3 : 2 * c = b + d) (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 :=
sorry

end find_four_numbers_l778_778910


namespace integral_abs_value_split_l778_778200

theorem integral_abs_value_split : 
  ∫ x in 0..2, 2 - |1 - x| = 3 :=
by
  sorry

end integral_abs_value_split_l778_778200


namespace length_PR_l778_778468

-- Defining a right triangle PQR with given conditions
def right_triangle (P Q R : Type) [right_triangle_structure : has_right_triangle P Q R] 

axiom angle_Q : ∠Q = 90
axiom sin_R : sin R = 3 / 5
axiom length_PQ : PQ = 9

-- The target is to prove PR = 15
theorem length_PR : PR = 15 :=
sorry

end length_PR_l778_778468


namespace problem_solution_l778_778648

noncomputable def solution_set : Set ℝ := {t : ℝ | ∃ X : Set ℝ, (∀ x y z ∈ X, ∀ a : ℝ, ∀ d : ℝ, 0 < d → max (|x - (a - d)|) (|y - a|) (|z - (a + d)|) > t * d) ∧ ∀ n : ℕ, ∃ x ∈ X, x > n}

theorem problem_solution :
  solution_set = {t : ℝ | t < 1 / 2} :=
sorry

end problem_solution_l778_778648


namespace HN_passes_through_fixed_point_l778_778253

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778253


namespace ellipse_solution_geometry_solution_l778_778309

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778309


namespace tan_2x_period_l778_778855

-- Define the tangent function and its properties
def tan := Real.tan

-- Define the problem
theorem tan_2x_period :
  (∃ P > 0, ∀ x, tan (2 * x) = tan (2 * x + P)) → P = π / 2 :=
by
  sorry

end tan_2x_period_l778_778855


namespace hn_passes_fixed_point_l778_778265

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778265


namespace three_digit_numbers_sum_17_l778_778939

/-- The number of three-digit numbers formed by choosing digits from the set {1, 2, ..., 9} such that 
the sum of the digits is 17 equals 57. -/
theorem three_digit_numbers_sum_17 : 
  (∃ (a b c : ℕ), a ∈ ({1, 2, ..., 9} : Set ℕ) ∧ b ∈ ({1, 2, ..., 9} : Set ℕ) ∧ c ∈ ({1, 2, ..., 9} : Set ℕ) 
  ∧ a + b + c = 17 ∧ a * 100 + b * 10 + c ∈ {100, 101, ..., 999} ) = 57 := 
sorry

end three_digit_numbers_sum_17_l778_778939


namespace max_fractional_momentum_transfer_eq_2_l778_778959

variables {m M v_0 : ℝ} (h_m : 0 < m) (h_M : 0 < M) (h_v0 : 0 < v_0)

def fractional_momentum_transfer (v_1 : ℝ) : ℝ :=
  |M * v_1 / (m * v_0)|

theorem max_fractional_momentum_transfer_eq_2 :
  let v_1 := 2 * m * v_0 / (m + M) in
  fractional_momentum_transfer m M v_0 h_v := 2 :=
sorry

end max_fractional_momentum_transfer_eq_2_l778_778959


namespace additional_interest_percentage_l778_778019

-- Define the principal amount, time period, and final amounts
def principal : ℝ := 8000
def time_period : ℝ := 3
def final_amount_original : ℝ := 9200
def final_amount_increased : ℝ := 9680

-- State the problem
theorem additional_interest_percentage :
  let SI_original := final_amount_original - principal,
      SI_increased := final_amount_increased - principal,
      additional_interest := SI_increased - SI_original,
      additional_percentage := (additional_interest * 100) / (principal * time_period)
  in additional_percentage = 2 := sorry

end additional_interest_percentage_l778_778019


namespace sqrt_sum_inequality_l778_778680

theorem sqrt_sum_inequality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  sqrt (2 * a + 1) + sqrt (2 * b + 1) ≤ 2 * sqrt 3 := 
by
  sorry

end sqrt_sum_inequality_l778_778680


namespace mixed_groups_count_l778_778895

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l778_778895


namespace rectangle_area_possibilities_l778_778550

noncomputable def area_of_rectangle (pq : ℝ) (qm : ℝ) : ℝ :=
  pq * qm

theorem rectangle_area_possibilities (PQ : ℝ) (d : ℝ) (tang_len : ℝ) (area1 area2 : ℝ) :
  PQ = 2 →
  d = sqrt 5 →
  tang_len = 1 →
  area1 = -1 + sqrt 5 →
  area2 = 1 + sqrt 5 →
  ∃ (qm : ℝ), qm = (sqrt 5 - 1) / 2 ∨ qm = (sqrt 5 + 1) / 2 ∧
  (area_of_rectangle PQ qm = area1 ∨ area_of_rectangle PQ qm = area2) :=
by { sorry }

end rectangle_area_possibilities_l778_778550


namespace minimum_period_tan_2x_l778_778860

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem minimum_period_tan_2x : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi / 2 :=
by
  sorry

end minimum_period_tan_2x_l778_778860


namespace find_quadratic_polynomial_with_root_l778_778659

noncomputable def quadratic_polynomial (a b c : ℝ) : ℝ := 3 * a^2 - 30 * b + 87

theorem find_quadratic_polynomial_with_root (x : ℂ) (h₁ : x = 5 + 2 * complex.I) 
        (h₂ : x.conj = 5 - 2 * complex.I) : 
        quadratic_polynomial x.re x.im (3) = 3 * (x^2).re - 30 * x.re + 87  :=
by
  -- Proof goes here
  sorry

end find_quadratic_polynomial_with_root_l778_778659


namespace problem_l778_778328

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778328


namespace length_of_third_side_l778_778391

theorem length_of_third_side (a b x : ℝ) (h1 : a = 6) (h2 : b = 8) (h_right_triangle : a^2 + x^2 = b^2 ∨ a^2 + b^2 = x^2) :
  x = 10 ∨ x = 2 * Real.sqrt 7 :=
by
  -- First we simplify and normalize our conditions
  cases h_right_triangle with hypotenuse leg;
  { rw [h1, h2] at *,
    exact sorry,  -- Proof for case where hypotenuse is considered
  } <|>
  { rw [h1, h2] at *,
    exact sorry,  -- Proof for case where leg is considered
  }

end length_of_third_side_l778_778391


namespace proof_alpha_beta_l778_778816

noncomputable def alpha_beta_proof (α β : ℝ) : Prop :=
  0 ≤ α ∧ α ≤ π / 2 ∧
  0 ≤ β ∧ β ≤ π / 2 ∧
  Real.tan α = 5 ∧
  Real.cot β = 2 / 3 →
  α + β = 3 * π / 4

theorem proof_alpha_beta (α β : ℝ) : alpha_beta_proof α β :=
sorry

end proof_alpha_beta_l778_778816


namespace b_alone_days_l778_778529

theorem b_alone_days {a b : ℝ} (h1 : a + b = 1/6) (h2 : a = 1/11) : b = 1/(66/5) :=
by sorry

end b_alone_days_l778_778529


namespace log_base9_of_cube_root_of_27_l778_778644

theorem log_base9_of_cube_root_of_27 : log 9 (27 ^ (1 / 3)) = 1 / 2 := by
  sorry

end log_base9_of_cube_root_of_27_l778_778644


namespace product_of_third_side_l778_778078

/-- Two sides of a right triangle have lengths 5 and 7. The product of the possible lengths of 
the third side is exactly √1776. -/
theorem product_of_third_side :
  let a := 5
  let b := 7
  (Real.sqrt (a^2 + b^2) * Real.sqrt (b^2 - a^2)) = Real.sqrt 1776 := 
by 
  let a := 5
  let b := 7
  sorry

end product_of_third_side_l778_778078


namespace problem_l778_778329

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778329


namespace finite_hedgehogs_l778_778826

structure Hedgehog :=
  (segments : fin 3 → ℝ × ℝ)
  (lengths : ∀ i, dist (0, 0) (segments i) = 1)
  (angles : ∀ i j, i ≠ j → angle (segments i) (segments j) = 120)

variable (WonderIsland : set (ℝ × ℝ))

axiom non_touching (h1 h2 : Hedgehog) (p1 p2 : ℝ × ℝ) :
  p1 ∈ WonderIsland → p2 ∈ WonderIsland →
  dist p1 p2 < 0.2 → 
  (∃ i j, dist (p1 + h1.segments i) (p2 + h2.segments j) = 0)

theorem finite_hedgehogs : 
  ∃ n : ℕ, ∀ Hs : set (Hedgehog × (ℝ × ℝ)),
  (∀ h, h ∈ Hs → (prod.snd h) ∈ WonderIsland) → 
  (∀ h1 h2, h1 ∈ Hs → h2 ∈ Hs → h1 ≠ h2 → 0.2 ≤ dist (prod.snd h1) (prod.snd h2)) →
  n ≥ fintype.card Hs :=
sorry

end finite_hedgehogs_l778_778826


namespace third_side_length_is_six_l778_778486

theorem third_side_length_is_six
  (a b : ℝ) (c : ℤ)
  (h1 : a = 6.31) 
  (h2 : b = 0.82) 
  (h3 : (a + b > c) ∧ ((b : ℝ) + (c : ℝ) > a) ∧ (c + a > b)) 
  (h4 : 5.49 < (c : ℝ)) 
  (h5 : (c : ℝ) < 7.13) : 
  c = 6 :=
by
  -- Proof goes here
  sorry

end third_side_length_is_six_l778_778486


namespace complex_div_power_simplification_l778_778027

theorem complex_div_power_simplification : 
  (3 : ℂ) / (1 - complex.I)^2 = - (3 / 2) * complex.I := 
by
  sorry

end complex_div_power_simplification_l778_778027


namespace climb_time_l778_778779

theorem climb_time (a d n : ℕ) (h₁ : a = 30) (h₂ : d = 10) (h₃ : n = 8) : 
  let l := a + (n - 1) * d in
  let S_n := n * (a + l) / 2 in
  S_n = 520 := 
by
  -- Introducing the conditions and letting Lean book-keep the rest
  rw [h₁, h₂, h₃]
  let l := 30 + (8 - 1) * 10
  let S_n := 8 * (30 + l) / 2
  have : l = 100 := by sorry -- showing l is 100 involves checking computation
  rw this
  let S_n := 8 * (30 + 100) / 2
  have : S_n = 520 := by sorry -- the final proof that S_n is 520 involves checking computation
  exact this

end climb_time_l778_778779


namespace right_isosceles_hypotenuse_angle_l778_778390

theorem right_isosceles_hypotenuse_angle (α β : ℝ) (γ : ℝ)
  (h1 : α = 45) (h2 : β = 45) (h3 : γ = 90)
  (triangle_isosceles : α = β)
  (triangle_right : γ = 90) :
  γ = 90 :=
by
  sorry

end right_isosceles_hypotenuse_angle_l778_778390


namespace shiny_pennies_prob_l778_778121

-- Define the problem statement in Lean 4.
theorem shiny_pennies_prob :
  let shiny := 4
  let dull := 5
  let total := shiny + dull
  -- Probability calculation by combination
  let p1 := (choose 5 3 * choose 4 1) / (choose 9 4)
  let p2 := (choose 5 2 * choose 4 2) / (choose 9 4)
  let p3 := (choose 5 1 * choose 4 3) / (choose 9 4)
  -- Summing all cases
  let total_prob := p1 + p2 + p3
  -- Reduced form
  let reduced_prob := 20 / 21
  -- Sum of a and b
  let a := 20
  let b := 21
  in total_prob = reduced_prob ∧ a + b = 41 :=
by {
  sorry
}

end shiny_pennies_prob_l778_778121


namespace increasing_function_on_interval_l778_778346

section
  variable (a b : ℝ)
  def f (x : ℝ) : ℝ := |x^2 - 2*a*x + b|

  theorem increasing_function_on_interval (h : a^2 - b ≤ 0) :
    ∀ x y : ℝ, a ≤ x → x ≤ y → f x ≤ f y := 
  sorry
end

end increasing_function_on_interval_l778_778346


namespace no_1968_classes_possible_l778_778775

-- Define the conditions and properties
def digits := Finset ℕ 
def class_equivalence (m n : ℕ) : Prop := 
  ∃ k : ℕ, digits.contains (m % 10^k)

theorem no_1968_classes_possible :
  ¬ ∃ (partition : Finset (Finset ℕ)), 
    (partition.card = 1968 ∧ 
    ∀ s ∈ partition, s ≠ ∅ ∧
    ∀ m n : ℕ, (digits.contains m n) ↔ (∃ s ∈ partition, m ∈ s ∧ n ∈ s)) := 
sorry

-- To be proven that the number of clusters/classes possible based on digit parity is 1024, which is less than 1968.

end no_1968_classes_possible_l778_778775


namespace ratio_of_doctors_to_nurses_l778_778991

theorem ratio_of_doctors_to_nurses (total: ℕ) (nurses: ℕ) (doctors: ℕ) (ratio: ℕ × ℕ) 
(H1: total = 200) (H2: nurses = 120) (H3: doctors = total - nurses) (H4: ratio = (Nat.gcd doctors nurses).div_gcd_nat_pair (doctors, nurses)):
ratio = (2, 3) :=
sorry

end ratio_of_doctors_to_nurses_l778_778991


namespace mixed_groups_count_l778_778905

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l778_778905


namespace books_distribution_l778_778072

-- Definitions of conditions used in the math problem
def five_books : Finset ℕ := {1, 2, 3, 4, 5}
def four_students : Finset ℕ := {1, 2, 3, 4}

-- Statement encapsulating the proof problem
theorem books_distribution :
  (∃ f : fin 5 → fin 4, 
   (∀ i : fin 4, ∃ j : fin 5, f j = i)) → 
  fintype.card {σ : (fin 5 → fin 4) // ∀ i : fin 4, ∃ j : fin 5, σ j = i} = 240 :=
by
  -- Placeholder for proof
  sorry

end books_distribution_l778_778072


namespace eclipse_falcata_area_correct_l778_778046

-- Define the radii of the circles
def big_radius : ℝ := 4
def small_radius : ℝ := 2

-- Define the areas of the respective quarter-circles
def quarter_circle_area (r : ℝ) : ℝ := (π * r^2) / 4

-- Calculate the areas of the larger and smaller quarter-circles
def big_quarter_circle_area : ℝ := quarter_circle_area big_radius
def small_quarter_circle_area : ℝ := quarter_circle_area small_radius

-- Define the area of the eclipse falcata
def eclipse_falcata_area : ℝ := big_quarter_circle_area - small_quarter_circle_area

-- Theorem to prove the area of the eclipse falcata
theorem eclipse_falcata_area_correct : eclipse_falcata_area = 3 * π :=
by
  -- We skip the proof with 'sorry'
  sorry

end eclipse_falcata_area_correct_l778_778046


namespace parities_of_A15_A16_A17_l778_778146

noncomputable def A : ℕ → ℕ 
| 0       := 1
| 1       := 1
| 2       := 0
| (n + 3) := A (n + 2) + A n

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem parities_of_A15_A16_A17 : 
  is_odd (A 15) ∧ is_even (A 16) ∧ is_odd (A 17) := 
by
  sorry

end parities_of_A15_A16_A17_l778_778146


namespace arithmetic_sequence_geometric_sequence_sum_of_c_l778_778692

theorem arithmetic_sequence (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 4 = 4) : 
  ∀ n, a n = n :=
by
  sorry

theorem geometric_sequence (b : ℕ → ℝ) (h1 : b 1 = 1) (h2 : b 1 + b 2 + b 3 = 7) : 
  ∀ n, b n = 2^(n-1) :=
by
  sorry

theorem sum_of_c (a : ℕ → ℕ) (b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ)
  (ha : ∀ n, a n = n)
  (hb : ∀ n, b n = 2^(n-1))
  (hc : ∀ n, c n = (a n) / (b n)) :
  ∀ n, S n = ∑ k in finset.range n.succ, c k → S n = 4 - (2 + n) / 2^(n-1) :=
by
  sorry

end arithmetic_sequence_geometric_sequence_sum_of_c_l778_778692


namespace product_of_intersection_coordinates_l778_778516

theorem product_of_intersection_coordinates :
  let circle1 := ∀ x y : ℝ, x^2 + 2*x + y^2 + 4*y + 4 = 0
  let circle2 := ∀ x y : ℝ, x^2 + 4*x + y^2 + 4*y + 7 = 0
  ∃ (x y : ℝ), (circle1 x y ∧ circle2 x y) ∧ (x * y = 3) :=
by
  sorry

end product_of_intersection_coordinates_l778_778516


namespace area_larger_sphere_l778_778147

noncomputable def sphere_area_relation (A1: ℝ) (R1 R2: ℝ) := R2^2 / R1^2 * A1

-- Given Conditions
def radius_smaller_sphere : ℝ := 4.0  -- R1
def radius_larger_sphere : ℝ := 6.0    -- R2
def area_smaller_sphere : ℝ := 17.0    -- A1

-- Target Area Calculation based on Proportional Relationship
theorem area_larger_sphere :
  sphere_area_relation area_smaller_sphere radius_smaller_sphere radius_larger_sphere = 38.25 :=
by
  sorry

end area_larger_sphere_l778_778147


namespace sum_of_solutions_eq_zero_l778_778638

variable (x y : ℝ)

theorem sum_of_solutions_eq_zero (h1 : y = 7) (h2 : x^2 + y^2 = 225) :
  (∑ sol in {x // sol ∈ {x | x^2 + 49 = 176}}, sol) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l778_778638


namespace total_height_correct_average_diff_correct_l778_778458

-- Define the heights of the buildings
def height_Burj_Khalifa : ℕ := 828
def height_Shanghai_Tower : ℕ := 632
def height_One_World_Trade_Center : ℕ := 541
def height_Willis_Tower : ℕ := 527

-- Define the total height of the buildings and the average height difference
def total_height : ℕ :=
  height_Burj_Khalifa + height_Shanghai_Tower + height_One_World_Trade_Center + height_Willis_Tower

def height_differences : List ℕ :=
  [height_Burj_Khalifa - height_Willis_Tower,
   height_Shanghai_Tower - height_Willis_Tower,
   height_One_World_Trade_Center - height_Willis_Tower,
   height_Willis_Tower - height_Willis_Tower]

def average_diff : ℕ :=
  (height_differences.sum / height_differences.length)

theorem total_height_correct : total_height = 2528 :=
  by
  simp [total_height, height_Burj_Khalifa, height_Shanghai_Tower, height_One_World_Trade_Center, height_Willis_Tower]
  sorry

theorem average_diff_correct : average_diff = 105 :=
  by
  simp [height_differences, average_diff, height_Burj_Khalifa, height_Shanghai_Tower, height_One_World_Trade_Center, height_Willis_Tower]
  sorry

end total_height_correct_average_diff_correct_l778_778458


namespace sufficient_but_not_necessary_condition_l778_778422

theorem sufficient_but_not_necessary_condition (a : ℝ) : 
  (a > 0) → (|2 * a + 1| > 1) ∧ ¬((|2 * a + 1| > 1) → (a > 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l778_778422


namespace sequence_geometric_proof_l778_778803

theorem sequence_geometric_proof (a : ℕ → ℕ) (h1 : a 1 = 5) (h2 : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a n = 5 * 2 ^ (n - 1) :=
by
  sorry

end sequence_geometric_proof_l778_778803


namespace coefficient_x2_expansion_l778_778401

theorem coefficient_x2_expansion : 
  (∑ n in Finset.range' 2 10, Nat.choose n 2) = 165 :=
by {
  sorry
}

end coefficient_x2_expansion_l778_778401


namespace initial_volume_of_mixture_l778_778137

/-- A mixture contains 10% water. 
5 liters of water should be added to this so that the water becomes 20% in the new mixture.
Prove that the initial volume of the mixture is 40 liters. -/
theorem initial_volume_of_mixture 
  (V : ℚ) -- Define the initial volume of the mixture
  (h1 : 0.10 * V + 5 = 0.20 * (V + 5)) -- Condition on the mixture
  : V = 40 := -- The statement to prove
by
  sorry -- Proof not required

end initial_volume_of_mixture_l778_778137


namespace smallest_cubes_to_hide_snaps_l778_778974

/-- A structure representing a cube with one protruding snap on one side and five receptacle holes on the other sides -/
structure Cube :=
(protruding_snap : Side)
(receptacle_holes : Fin 5 → Side)

/-- Given the conditions of the cube, prove that the smallest number of cubes needed such that only receptacle holes are showing is 4 -/
theorem smallest_cubes_to_hide_snaps (c : Cube) :
  ∃ (n : ℕ), n = 4 ∧ (∀ (cubes : Fin n → Cube), all_protruding_snaps_hidden cubes) :=
sorry

/-- An auxiliary definition indicating that all protruding snaps are hidden in a given arrangement of cubes -/
def all_protruding_snaps_hidden (cubes : Fin n → Cube) : Prop :=
-- Definition of how the snaps are arranged and how they hide each other
sorry

end smallest_cubes_to_hide_snaps_l778_778974


namespace sum_lent_is_10000_l778_778123

theorem sum_lent_is_10000
  (P : ℝ)
  (r : ℝ := 0.075)
  (t : ℝ := 7)
  (I : ℝ := P - 4750) 
  (H1 : I = P * r * t) :
  P = 10000 :=
sorry

end sum_lent_is_10000_l778_778123


namespace verify_total_distance_l778_778009

noncomputable def total_distance (dist_SR dist_SG dist_GF dist_FR headwind_FR tailwind_RG headwind_GS : ℝ) : ℝ :=
let adjusted_FR := dist_FR * (1 + headwind_FR / 100)
let adjusted_RG := dist_SR * (1 - tailwind_RG / 100)
let adjusted_GS := dist_SG * (1 + headwind_GS / 100)
in adjusted_FR + adjusted_RG + adjusted_GS

-- Defining the conditions given in the problem
def dist_SR : ℝ := 7019
def dist_SG : ℝ := 1615
def dist_GF : ℝ := 956
def dist_FR : ℝ := 6180
def headwind_FR : ℝ := 6
def tailwind_RG : ℝ := 4
def headwind_GS : ℝ := 2

theorem verify_total_distance : total_distance dist_SR dist_SG dist_GF dist_FR headwind_FR tailwind_RG headwind_GS = 14936.34 := by
  sorry

end verify_total_distance_l778_778009


namespace min_trips_theorem_l778_778069

noncomputable def min_trips (weights : List ℕ) (capacity : ℕ) (total_weight : ℕ) : ℕ :=
  if total_weight = 0 then 0
  else let weight_sum := List.sum weights
       if weight_sum <= capacity then 1
       else if weights.length = 1 then weights.length / capacity
       else (weight_sum + capacity - 1) / capacity

theorem min_trips_theorem (weights : List ℕ) (h_distinct : weights.nodup)
                          (h_max_weight : ∀ w ∈ weights, w ≤ 10)
                          (h_total_weight : List.sum weights = 400)
                          (h_multiple_of_quintal : ∀ w ∈ weights, w % 100 = 0) :
  min_trips weights 10 400 = 51 := 
sorry

end min_trips_theorem_l778_778069


namespace remaining_area_correct_l778_778399

noncomputable def remaining_area_ABHFGD : ℝ :=
  let area_square_ABCD := 25
  let area_square_EFGD := 16
  let side_length_ABCD := Real.sqrt area_square_ABCD
  let side_length_EFGD := Real.sqrt area_square_EFGD
  let overlap_area := 8
  area_square_ABCD + area_square_EFGD - overlap_area

theorem remaining_area_correct :
  let area := remaining_area_ABHFGD
  area = 33 :=
by
  sorry

end remaining_area_correct_l778_778399


namespace prime_factor_of_sum_of_four_consecutive_integers_l778_778912

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) :
  ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ ((n - 2) + (n - 1) + n + (n + 1)) :=
by
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_l778_778912


namespace minimum_trips_l778_778061

/-- 
Problem: There are 400 tons of cargo stored in a warehouse, each with a weight that is a multiple 
of a quintal and does not exceed 10 tons. It is known that any two pieces of cargo have different 
weights. Prove that the minimum number of trips a 10-ton truck must make to guarantee the 
transportation of all this cargo from the warehouse is 51.
-/

theorem minimum_trips (n : ℕ) (weights : fin n → ℕ)
  (h1 : ∀ i, weights i % 10 = 0)
  (h2 : ∀ i j, i ≠ j → weights i ≠ weights j)
  (h3 : ∑ i, weights i = 4000)
  (h4 : ∀ i, weights i ≤ 100) :
  ∃ t : ℕ, t = 51 ∧ ∑ i in finset.range t, weights i ≤ 10000 :=
sorry

end minimum_trips_l778_778061


namespace B_pow_101_eq_B_pow_5_l778_778411

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 0]]

theorem B_pow_101_eq_B_pow_5 : B^101 = B := 
by sorry

end B_pow_101_eq_B_pow_5_l778_778411


namespace rod_stability_l778_778596

theorem rod_stability (l : ℝ) (rho_m : ℝ) (rho_w : ℝ) (h : ℝ) :
  l = 1 → rho_m = 0.6 → rho_w = 1 → h ≤ 0.63 * l :=
by
  intros hlen hrho_m hrho_w
  rw [hlen, hrho_m, hrho_w]
  exact sorry

end rod_stability_l778_778596


namespace prime_factor_of_sum_of_four_consecutive_integers_l778_778911

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) :
  ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ ((n - 2) + (n - 1) + n + (n + 1)) :=
by
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_l778_778911


namespace factor_of_sum_of_consecutive_integers_l778_778917

theorem factor_of_sum_of_consecutive_integers :
  ∀ n : ℤ, ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by
  intro n
  use 2
  split
  exact prime_two
  split
  rfl
  calc
    2 ∣ 2 * (2 * n + 1) : by apply dvd_mul_right

end factor_of_sum_of_consecutive_integers_l778_778917


namespace interior_sum_nine_l778_778771

-- Defining the function for the sum of the interior numbers in the nth row of Pascal's Triangle
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

-- Given conditions
axiom interior_sum_4 : interior_sum 4 = 6
axiom interior_sum_5 : interior_sum 5 = 14

-- Goal to prove
theorem interior_sum_nine : interior_sum 9 = 254 := by
  sorry

end interior_sum_nine_l778_778771


namespace intercepts_sum_eq_eight_l778_778837

theorem intercepts_sum_eq_eight :
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  a + b + c = 8 :=
by
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  -- Proof will go here
  sorry

end intercepts_sum_eq_eight_l778_778837


namespace largest_number_of_acute_angles_in_convex_heptagon_l778_778514

theorem largest_number_of_acute_angles_in_convex_heptagon :
  ∀ (angles : Fin 7 → ℝ), 
  (∀ i, 0 ≤ angles i ∧ angles i < 180) → 
  (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5 + angles 6 = 900) → 
  (∃ (n : ℕ), n ≤ 4 ∧ (∃ (acute_positions : Fin n → ℝ), ∀ i, acute_positions i < 90 ∧ 
  (∃ (remaining_positions : Fin (7 - n) → ℝ), 
  ∀ j, 90 ≤ remaining_positions j ∧ remaining_positions j < 180 ∧ ∀ k, angles k = if k < n then acute_positions k else remaining_positions (k - n)))) :=
sorry

end largest_number_of_acute_angles_in_convex_heptagon_l778_778514


namespace quadratic_polynomial_real_coeff_l778_778663

theorem quadratic_polynomial_real_coeff (a b : ℂ) (h₁ : a = 5 + 2*i) 
  (h₂ : b = 5 - 2*i) (c : ℂ) (hc : c = 3) :
  3 * (X - C a) * (X - C b) = 3*X^2 - 30*X + 87 := 
by {
  sorry
}

end quadratic_polynomial_real_coeff_l778_778663


namespace sequence_614th_term_l778_778481

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (fun c => (c.toNat - '0'.toNat) ^ 2)).sum

noncomputable def sequence_term (n : ℕ) : ℕ :=
  Nat.recOn n 3107 (λ n prev_term, sum_of_squares_of_digits prev_term)

theorem sequence_614th_term : sequence_term 613 = 20 :=
sorry

end sequence_614th_term_l778_778481


namespace matrix_product_is_correct_l778_778209

noncomputable def matrix_seq_product : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range' 2 50).foldl
    (λ M n, M.mul (λ i j, if (i, j) = (0, 0) then 1 else if (i, j) = (1,1) then 1 else if (i, j) = (0, 1) then n else 0))
    (λ i j, if (i, j) = (0, 0) then 1 else if (i, j) = (1,1) then 1 else if (i, j) = (0, 1) then 2 else 0)

theorem matrix_product_is_correct :
  matrix_seq_product = (λ i j, if (i, j) = (0, 0) then 1 else if (i, j) = (1,1) then 1 else if (i, j) = (0, 1) then 2550 else 0) :=
by
  sorry

end matrix_product_is_correct_l778_778209


namespace basic_computer_price_l778_778054

theorem basic_computer_price (C P : ℝ) 
  (h1 : C + P = 2500)
  (h2 : P = 1 / 8 * ((C + 500) + P)) :
  C = 2125 :=
by
  sorry

end basic_computer_price_l778_778054


namespace super_cool_triangles_area_sum_l778_778573

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end super_cool_triangles_area_sum_l778_778573


namespace disk_area_sum_eq_82_l778_778642

noncomputable def circle_radius := 1
noncomputable def disks := 8
noncomputable def radius_of_small_disk := 2 - Real.sqrt 2
noncomputable def area_of_one_disk := Real.pi * (2 - Real.sqrt 2) ^ 2
noncomputable def total_area := disks * area_of_one_disk
noncomputable def a := 48
noncomputable def b := 32
noncomputable def c := 2

theorem disk_area_sum_eq_82 :
  (∑ i in Finset.range disks, area_of_one_disk) = Real.pi * (a - b * Real.sqrt c) →
  a + b + c = 82 :=
by
  sorry

end disk_area_sum_eq_82_l778_778642


namespace parabola_intercepts_sum_l778_778845

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end parabola_intercepts_sum_l778_778845


namespace constant_sum_of_chord_distances_l778_778500

theorem constant_sum_of_chord_distances (m : ℝ) (k : ℝ) (x₁ x₂ : ℝ) : 
  let A := (x₁, 2 * x₁ ^ 2)
      B := (x₂, 2 * x₂ ^ 2)
      C := (0, k)
      AC := Real.sqrt (x₁ ^ 2 + (2 * x₁ ^ 2 - k) ^ 2)
      BC := Real.sqrt (x₂ ^ 2 + (2 * x₂ ^ 2 - k) ^ 2)
  in k = 1 / 2 ∧
     2 * x₁ ^ 2 - x₁ * m - k = 0 ∧
     2 * x₂ ^ 2 - x₂ * m - k = 0 ∧
     (1 / AC ^ 3) + (1 / BC ^ 3) = 8 :=
begin
  sorry
end

end constant_sum_of_chord_distances_l778_778500


namespace angle_C_in_triangle_find_side_b_range_of_a_correct_statements_for_triangle_l778_778115

-- Problem 1
theorem angle_C_in_triangle 
  (a b c : ℝ) (A B C : ℝ) (h1 : ∀ Δ, Δ.angle B C A = A) 
  (h2 : ∀ Δ, Δ.angle C A B = B) 
  (h3 : ∀ Δ, Δ.angle A B C = C) 
  (h4 : (a * sin A + b * sin B - c * sin C) / (a * sin B) = 2 * sqrt 3 * sin C) :
  C = π / 6 := 
sorry

-- Problem 2
theorem find_side_b 
  (a c : ℝ) (A C B : ℝ) (h1 : a = 2) 
  (h2 : c = 2 * sqrt 3) 
  (h3 : cos A = sqrt 3 / 2) 
  (h4 : sin A = 1/2) :
  b = 2 ∨ b = 4 := 
sorry

-- Problem 3
theorem range_of_a 
  (a : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : acute (triangle.mk 1 3 a)) :
  2 * sqrt 2 < a ∧ a < sqrt 10 := 
sorry

-- Problem 4
theorem correct_statements_for_triangle (k : ℝ) (A B C a b c : ℝ) (h1 : (b + c) = 4 * k)
  (h2 : (c + a) = 5 * k) (h3 : (a + b) = 6 * k) (h4 : k > 0) 
  (h5 : ratops a b c 7 5 3) (h6 : cos A = -1/2) :
  (ratio A B C 7 5 3) ∧ (obtuse A) ∧ (b + c = 8 → area (triangle.mk a b c) = 15 * sqrt 3 / 2) := 
sorry

end angle_C_in_triangle_find_side_b_range_of_a_correct_statements_for_triangle_l778_778115


namespace sin_cos_45_degrees_l778_778619

noncomputable def Q : ℝ × ℝ := (1 / real.sqrt 2, 1 / real.sqrt 2)

def is_unit_circle (p : ℝ × ℝ) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 = 1

def is_45_45_90_triangle (h : ℝ) (a : ℝ) : Prop :=
  h = a * real.sqrt 2

theorem sin_cos_45_degrees :
  (∀ p : ℝ × ℝ, is_unit_circle p → ∃ a : ℝ, is_45_45_90_triangle 1 a ∧ p = (a, a)) →
  real.sin (real.pi / 4) = 1 / real.sqrt 2 ∧ real.cos (real.pi / 4) = 1 / real.sqrt 2 :=
by
  sorry

end sin_cos_45_degrees_l778_778619


namespace volume_pyramid_isosceles_l778_778025

-- Define the problem conditions and volume formula
def volume_of_pyramid (a φ r : ℝ) : ℝ :=
  (a^3 * sin φ * sqrt (a^2 - (a^2 / (4 * cos (φ / 2)^2)))) / 6

-- Define the theorem we want to prove
theorem volume_pyramid_isosceles (a φ r V : ℝ)
  (h_base : ∀ (ABC : Triangle), ABC.isosceles ∧ ABC.vertex_angle = φ)
  (h_lateral_edges : ∀ (D : Point) (A : Point), D.distance A = a)
  (h_inscribed_radius : ∀ (O : Point) (r := O.inscribed_circle_radius), r ∈ inscribed_radius ABC) :
  V = volume_of_pyramid a φ r :=
  sorry

end volume_pyramid_isosceles_l778_778025


namespace chess_club_mixed_groups_l778_778887

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l778_778887


namespace sin_value_l778_778700

theorem sin_value (x : ℝ) (h : Real.sin (x + π / 3) = Real.sqrt 3 / 3) :
  Real.sin (2 * π / 3 - x) = Real.sqrt 3 / 3 :=
by
  sorry

end sin_value_l778_778700


namespace ellipse_and_fixed_point_l778_778305

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778305


namespace distance_from_apex_to_A_is_zero_l778_778988

theorem distance_from_apex_to_A_is_zero :
  let A : ℝ × ℝ × ℝ := (0, 0, 0)
  let B : ℝ × ℝ × ℝ := (4 * Real.sqrt 3, 0, 0)
  let C : ℝ × ℝ × ℝ := (2 * Real.sqrt 3, 2 * Real.sqrt 3, 0)
  let P : ℝ × ℝ × ℝ := (0, 0, 10)
  let R : ℝ × ℝ × ℝ := (2 * Real.sqrt 3, 2 * Real.sqrt 3, 6)
  let t := 0 in
  let Point_on_AC := (2 * t * Real.sqrt 3, 2 * t * Real.sqrt 3, 10 - 4 * t) in
  (Point_on_AC.1 - A.1) ^ 2 + (Point_on_AC.2 - A.2) ^ 2 + 0 ^ 2 = 0 :=
by
  -- Proof goes here
  sorry

end distance_from_apex_to_A_is_zero_l778_778988


namespace common_tangents_count_l778_778729

open Real

def circle1 := { p : ℝ × ℝ | (p.fst - 1)^2 + p.snd^2 = 1 }
def circle2 := { p : ℝ × ℝ | p.fst^2 + (p.snd + 2)^2 = 4 }

theorem common_tangents_count :
  let d := sqrt ((1 - 0)^2 + (0 - (-2))^2)
  in (1 < d ∧ d < 3) → ∃ n, n = 2 := by
  sorry

end common_tangents_count_l778_778729


namespace mixed_groups_count_l778_778901

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l778_778901


namespace sin_alpha_minus_beta_find_beta_l778_778761

noncomputable theory 

variables (α β : ℝ)

def vector_a := (Real.cos α, Real.sin α)
def vector_b := (-Real.sin β, Real.cos β)
def vector_c := (-1/2, Real.sqrt 3 / 2)

def length (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
def dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2
def is_parallel (v w : ℝ × ℝ) := ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem sin_alpha_minus_beta (h : length (vector_a α β + vector_b α β) = length vector_c) :
  Real.sin (α - β) = -1/2 :=
sorry

theorem find_beta (h1 : α = 5 * Real.pi / 6) (h2 : 0 < β ∧ β < Real.pi)
  (h3 : is_parallel (vector_a α β) (vector_b α β + vector_c)) :
  β = Real.pi / 2 :=
sorry

end sin_alpha_minus_beta_find_beta_l778_778761


namespace tangent_line_existence_l778_778487

theorem tangent_line_existence :
  (∃ b1, ∃ b2, ∀ x (h1 : f x = log x) (h2 : f (x + b1) = (1/2) * x + b1), (f' x = 1 / 2)) ∧
  (∃ b3, ∃ b4, ∀ x (h3 : f x = sin x) (h4 : f (x + b3) = (1/2) * x + b3), (f' x = 1 / 2 ∨ f' x ∈ Icc (-1 : ℝ) 1)) :=
sorry

end tangent_line_existence_l778_778487


namespace mixed_groups_count_l778_778893

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l778_778893


namespace terms_before_50_of_arithmetic_sequence_l778_778369

noncomputable def arithmetic_sequence_term (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem terms_before_50_of_arithmetic_sequence :
  ∀ (a d : ℤ), a = 100 → d = -5 → ∃ n : ℕ, arithmetic_sequence_term a d n = 50 ∧ n - 1 = 10 :=
by {
  intros a d ha hd,
  use 11,
  split,
  {
    rw [ha, hd, arithmetic_sequence_term],
    norm_num,
  },
  norm_num,
}

end terms_before_50_of_arithmetic_sequence_l778_778369


namespace integral_D_equals_2_l778_778600

noncomputable def integral_A : ℝ := ∫ x in 0..5, 2 * x - 4
noncomputable def integral_B : ℝ := ∫ x in 0..Real.pi, cos x
noncomputable def integral_C : ℝ := ∫ x in 1..3, 1 / x
noncomputable def integral_D : ℝ := ∫ x in 0..Real.pi, sin x

theorem integral_D_equals_2 : integral_D = 2 := by
  sorry

end integral_D_equals_2_l778_778600


namespace smallest_integer_l778_778852

theorem smallest_integer (n : ℕ) (h : n > 0) (h1 : lcm 36 n / gcd 36 n = 24) : n = 96 :=
sorry

end smallest_integer_l778_778852


namespace necessary_condition_for_three_lines_l778_778356

theorem necessary_condition_for_three_lines (r : ℝ) (h₀ : r > 0)
  (parabola : ∀ y : ℝ, ∃ x : ℝ, y^2 = 4 * x)
  (circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = r^2)
  (h_line_intersect : ∃ (l : ℝ → ℝ) (C D A B : ℝ × ℝ),
    l 0 = 1 ∧ -- line l passes through the point (1, 0)
    (circle C.1 C.2) ∧ (circle D.1 D.2) ∧ -- intersects the circle N at points C and D
    (parabola A.2) ∧ (parabola B.2) ∧ -- intersects the parabola M at points A and B
    |C.1 - A.1| = |D.1 - B.1|) -- satisfying |AC| = |BD|
  : 1.5 ≤ r :=
sorry

end necessary_condition_for_three_lines_l778_778356


namespace part1_part2_l778_778225

variable (a b : ℝ)

-- Conditions
axiom abs_a_eq_4 : |a| = 4
axiom abs_b_eq_6 : |b| = 6

-- Part 1: If ab > 0, find the value of a - b
theorem part1 (h : a * b > 0) : a - b = 2 ∨ a - b = -2 := 
by
  -- Proof will go here
  sorry

-- Part 2: If |a + b| = -(a + b), find the value of a + b
theorem part2 (h : |a + b| = -(a + b)) : a + b = -10 ∨ a + b = -2 := 
by
  -- Proof will go here
  sorry

end part1_part2_l778_778225


namespace sin_sum_zero_cos_sum_nonneg_integer_l778_778428

variable (n : ℕ) (X : Type) [Fintype X]
variable (f : X → X) (hf : ∀ x : X, (Nat.iterate f n) x = x)
variable (m : ℕ → ℕ) (hk : ℤ)

def iterate (f : X → X) : ℕ → (X → X)
| 0 => id
| (nat.succ k) => f ∘ iterate f k

def fixed_points (f : X → X) (j : ℕ) : Finset X :=
(Fintype.elems X).filter (λ x, iterate f j x = x)

noncomputable def m_j_def (j : ℕ) : ℕ :=
(fixed_points f j).card

theorem sin_sum_zero (hf : ∀ x : X, iterate f n x = x)
    (hm : ∀ j, m j = m_j_def f j)
    (k : ℤ) :
  (1 : ℝ) / (n : ℝ) * ∑ j in Finset.range n, (m j : ℝ) * Real.sin (2 * (j + 1 : ℝ) * k * Real.pi / n) = 0 := 
sorry

theorem cos_sum_nonneg_integer (hf : ∀ x : X, iterate f n x = x)
    (hm : ∀ j, m j = m_j_def f j)
    (k : ℤ) :
  ∃ (c : ℕ), (1 : ℝ) / (n : ℝ) * ∑ j in Finset.range n, (m j : ℝ) * Real.cos (2 * (j + 1 : ℝ) * k * Real.pi / n) = c :=
sorry

end sin_sum_zero_cos_sum_nonneg_integer_l778_778428


namespace area_rectangle_ABCD_l778_778957

theorem area_rectangle_ABCD {ABCD : Type*} [rectangular ABCD]
  (A B C D E F B' C' : Point ABCD)
  (AB AD : Line ABCD)
  (BE : Segment AB) (CF : Segment CD)
  (EonAB : E ∈ AB) (FonCD : F ∈ CD)
  (BE_lt_CF : BE.length < CF.length)
  (C'_on_AD : C' ∈ AD) (B'_on_some_line : B' ∈ some_line)
  (angle_condition : ∠ AB'C' = 2 * ∠ B'EA)
  (AB'_len : AB' = 8) (BE_len : BE.length = 15) :
  ∃ (a b c : ℤ), (area ABCD = a + b * sqrt c) ∧ (a + b + c = 127) :=
  sorry

end area_rectangle_ABCD_l778_778957


namespace permutation_of_signed_vars_l778_778415

noncomputable def f (x y z : ℝ) : ℝ := x^2 + y^2 + z^2 + x * y * z

def is_permutation_of_signed_vars (p q r : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (a * b * c = 1) ∧ 
    (p = if a = 1 then (λ x y z, x) else (λ x y z, -x)) ∧
    (q = if b = 1 then (λ x y z, y) else (λ x y z, -y)) ∧
    (r = if c = 1 then (λ x y z, z) else (λ x y z, -z))

theorem permutation_of_signed_vars
  (p q r : ℝ → ℝ → ℝ → ℝ)
  (hpqrf : ∀ x y z, f (p x y z) (q x y z) (r x y z) = f x y z) :
  is_permutation_of_signed_vars p q r :=
sorry

end permutation_of_signed_vars_l778_778415


namespace max_real_part_l778_778020

open Complex

theorem max_real_part (z w : ℂ) (hz: abs z = 2) (hw: abs w = 2) (hzw : z * conj w + conj z * w = 2) :
  re (z + w) ≤ sqrt 10 :=
sorry

end max_real_part_l778_778020


namespace tangent_inclination_point_l778_778867

theorem tangent_inclination_point :
  ∃ a : ℝ, (2 * a = 1) ∧ ((a, a^2) = (1 / 2, 1 / 4)) :=
by
  sorry

end tangent_inclination_point_l778_778867


namespace greatest_n_divisible_by_all_possible_cubes_le_l778_778206

theorem greatest_n_divisible_by_all_possible_cubes_le (n : ℕ) :
  (∃ a : ℕ, n = Nat.lcm (List.range (a+1)) ∧ Nat.lcm (List.range (a+1)) ≤ a ^ 3) → n = 60 :=
by
  sorry

end greatest_n_divisible_by_all_possible_cubes_le_l778_778206


namespace kristin_annual_income_l778_778872

theorem kristin_annual_income (p : ℝ) :
  ∃ A : ℝ, 
  (0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = (0.01 * (p + 0.25) * A)) ∧
  A = 32000 :=
by
  sorry

end kristin_annual_income_l778_778872


namespace problem1_problem2_l778_778762

-- Define the point and necessary trigonometric functions
variable (α : ℝ)
def point_on_terminal_side := (-3:ℝ, -4:ℝ)
def sin_α := -4 / 5
def tan_α := 4 / 3
def cos_α := -3 / 5

-- Problem 1
theorem problem1 :
  (sin_α / tan_α) = -3 / 5 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (sin (α + π/2) * cos (9 * π / 2 - α) * tan (2 * π - α) * cos (-3 * π / 2 + α)) /
  (sin (2 * π - α) * tan (-α - π) * sin (π + α)) = 3 / 5 :=
by
  sorry

end problem1_problem2_l778_778762


namespace number_increased_by_one_fourth_l778_778925

theorem number_increased_by_one_fourth (n : ℕ) (h : 25 * 80 / 100 = 20) (h1 : 80 - 20 = 60) :
  n + n / 4 = 60 ↔ n = 48 :=
by
  -- Conditions
  have h2 : 80 - 25 * 80 / 100 = 60 := by linarith [h, h1]
  have h3 : n + n / 4 = 60 := sorry
  -- Assertion (Proof to show is omitted)
  sorry

end number_increased_by_one_fourth_l778_778925


namespace minimum_trips_proof_l778_778056

noncomputable def minimum_trips (cargo : ℕ) (weight_limit : ℕ) : ℕ :=
  by 
  have h1 : cargo = 400 := sorry -- given condition: there are 400 tons of cargo
  have h2 : weight_limit = 10 := sorry -- given condition: truck capacity is 10 tons
  have h3 : ∀ w, w % 1 = 0 ∧ w ≤ 10 ∧ (distinct_weights w) := sorry -- conditions: weights are multiples of quintals, distinct, and ≤ 10 tons
  sorry -- prove that minimum trips is 51

theorem minimum_trips_proof :
  minimum_trips 400 10 = 51 := 
  sorry

end minimum_trips_proof_l778_778056


namespace binary_calc_l778_778201

theorem binary_calc : 
  (nat.bodd 0b1011010 * nat.bodd 0b1010100) / nat.bodd 0b100 = nat.bodd 0b110001111100 :=
by {
  sorry
}

end binary_calc_l778_778201


namespace replace_mul_with_signs_to_get_2012_l778_778454

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  ∑ i in range (n // 4), 4

theorem replace_mul_with_signs_to_get_2012 : sequence_sum 2012 = 2012 :=
  sorry

end replace_mul_with_signs_to_get_2012_l778_778454


namespace largest_angle_in_triangle_l778_778052

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A + B = 126) (h₂ : A = B + 20) (h₃ : A + B + C = 180) :
  max A (max B C) = 73 := sorry

end largest_angle_in_triangle_l778_778052


namespace muffin_probability_l778_778806

theorem muffin_probability :
  let n := 7 
  let p := 3 / 4
  let k := 5
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 5103 / 16384 := by
  -- Definitions of variables n, p, k
  let n := 7 
  let p := 3 / 4
  let k := 5
  -- Calculate the binomial probability
  have h1 : (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 5103 / 16384 := sorry
  exact h1

end muffin_probability_l778_778806


namespace ellipse_equation_proof_HN_fixed_point_l778_778319

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778319


namespace factor_of_sum_of_consecutive_integers_l778_778919

theorem factor_of_sum_of_consecutive_integers :
  ∀ n : ℤ, ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by
  intro n
  use 2
  split
  exact prime_two
  split
  rfl
  calc
    2 ∣ 2 * (2 * n + 1) : by apply dvd_mul_right

end factor_of_sum_of_consecutive_integers_l778_778919


namespace ellipse_and_fixed_point_l778_778296

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778296


namespace ellipse_equation_and_fixed_point_proof_l778_778285

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778285


namespace area_of_circle_l778_778815

noncomputable def point : Type := ℝ × ℝ

def A : point := (8, 15)
def B : point := (14, 9)

def is_on_circle (P : point) (r : ℝ) (C : point) : Prop :=
  (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = r ^ 2

def tangent_intersects_x_axis (tangent_point : point) (circle_center : point) : Prop :=
  ∃ x : ℝ, ∃ C : point, C.2 = 0 ∧ tangent_point = C ∧ circle_center = (x, 0)

theorem area_of_circle :
  ∃ C : point, ∃ r : ℝ,
    is_on_circle A r C ∧ 
    is_on_circle B r C ∧ 
    tangent_intersects_x_axis A C ∧ 
    tangent_intersects_x_axis B C ∧ 
    (↑(π * r ^ 2) = (117 * π) / 8) :=
sorry

end area_of_circle_l778_778815


namespace tangent_line_equation_l778_778830

theorem tangent_line_equation (y : ℝ → ℝ) (x : ℝ) (dy_dx : ℝ → ℝ) (tangent_eq : ℝ → ℝ → Prop):
  (∀ x, y x = x^2 + Real.log x) →
  (∀ x, dy_dx x = (deriv y) x) →
  (dy_dx 1 = 3) →
  (tangent_eq x (y x) ↔ (3 * x - y x - 2 = 0)) →
  tangent_eq 1 (y 1) :=
by
  intros y_def dy_dx_def slope_at_1 tangent_line_char
  sorry

end tangent_line_equation_l778_778830


namespace ellipse_equation_proof_HN_fixed_point_l778_778320

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778320


namespace total_slices_left_is_14_l778_778631

-- Define the initial conditions
def large_pizza_slices : ℕ := 12
def small_pizza_slices : ℕ := 8
def hawaiian_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def cheese_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def pepperoni_pizza (num_small : ℕ) : ℕ := num_small * small_pizza_slices

-- Number of large pizzas ordered (Hawaiian and cheese)
def num_large_pizzas : ℕ := 2

-- Number of small pizzas received in promotion
def num_small_pizzas : ℕ := 1

-- Slices eaten by each person
def dean_slices (hawaiian_slices : ℕ) : ℕ := hawaiian_slices / 2
def frank_slices : ℕ := 3
def sammy_slices (cheese_slices : ℕ) : ℕ := cheese_slices / 3
def nancy_cheese_slices : ℕ := 2
def nancy_pepperoni_slice : ℕ := 1
def olivia_slices : ℕ := 2

-- Total slices eaten from each pizza
def total_hawaiian_slices_eaten (hawaiian_slices : ℕ) : ℕ := dean_slices hawaiian_slices + frank_slices
def total_cheese_slices_eaten (cheese_slices : ℕ) : ℕ := sammy_slices cheese_slices + nancy_cheese_slices
def total_pepperoni_slices_eaten : ℕ := nancy_pepperoni_slice + olivia_slices

-- Total slices left over
def total_slices_left (hawaiian_slices : ℕ) (cheese_slices : ℕ) (pepperoni_slices : ℕ) : ℕ := 
  (hawaiian_slices - total_hawaiian_slices_eaten hawaiian_slices) + 
  (cheese_slices - total_cheese_slices_eaten cheese_slices) + 
  (pepperoni_slices - total_pepperoni_slices_eaten)

-- The actual Lean 4 statement to be verified
theorem total_slices_left_is_14 : total_slices_left (hawaiian_pizza num_large_pizzas) (cheese_pizza num_large_pizzas) (pepperoni_pizza num_small_pizzas) = 14 := 
  sorry

end total_slices_left_is_14_l778_778631


namespace hn_passes_fixed_point_l778_778261

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778261


namespace final_speed_after_acceleration_l778_778117

-- Definitions for the given conditions
def train_length : ℝ := 500
def time_to_cross_pole : ℝ := 50
def acceleration : ℝ := 0.5
def acceleration_time : ℝ := 20

-- Initial speed when crossing the electric pole
def initial_speed := train_length / time_to_cross_pole

-- Final speed after acceleration
def final_speed := initial_speed + acceleration * acceleration_time

-- Theorem statement
theorem final_speed_after_acceleration : final_speed = 20 := by
  sorry

end final_speed_after_acceleration_l778_778117


namespace sum_of_super_cool_triangle_areas_l778_778586

noncomputable def super_cool_triangle_sum_area : ℕ :=
  let leg_pairs := [(7, 42), (8, 24), (9, 18), (10, 15), (12, 12)] in
  let areas := leg_pairs.map (λ p, (p.1 * p.2) / 2) in
  areas.sum

theorem sum_of_super_cool_triangle_areas : super_cool_triangle_sum_area = 471 :=
by
  sorry

end sum_of_super_cool_triangle_areas_l778_778586


namespace students_with_both_dog_and_cat_l778_778641

theorem students_with_both_dog_and_cat
  (n D C : ℕ)
  (h1 : n = 50)
  (h2 : D = 35)
  (h3 : C = 40) :
  ∃ (both : ℕ), both = 25 ∧ D + C - both = n := 
by
  use 25
  split
  · rfl
  · rw [h1, h2, h3]
    rfl

end students_with_both_dog_and_cat_l778_778641


namespace grant_school_students_l778_778814

theorem grant_school_students (S : ℕ) 
  (h1 : S / 3 = x) 
  (h2 : x / 4 = 15) : 
  S = 180 := 
sorry

end grant_school_students_l778_778814


namespace decreasing_interval_of_f_minimum_value_of_f_on_interval_l778_778713

noncomputable def f (a x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem decreasing_interval_of_f :
  ∃ a : ℝ, ∀ x : ℝ, (f a x').deriv < 0 → x < -1 ∨ x > 3 := sorry

theorem minimum_value_of_f_on_interval (a : ℝ) (h_max : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → f a x ≤ 20) :
  f -2 (-1) = -7 := sorry

end decreasing_interval_of_f_minimum_value_of_f_on_interval_l778_778713


namespace pool_fill_time_l778_778220

-- Definitions according to conditions
def pool_volume : ℝ := 15000  -- pool volume in gallons

def hose_rate1 : ℝ := 2       -- rate of first type of hoses in gallons per minute
def hose_rate2 : ℝ := 3       -- rate of second type of hoses in gallons per minute

def hoses_count1 : ℕ := 2     -- number of first type of hoses
def hoses_count2 : ℕ := 2     -- number of second type of hoses

-- The main theorem to be proved
theorem pool_fill_time (volume : ℝ) (rate1 rate2 : ℝ) (count1 count2 : ℕ) :
  let total_rate := (rate1 * count1) + (rate2 * count2) in
  let time_minutes := volume / total_rate in
  let time_hours := time_minutes / 60 in
  volume = pool_volume →
  rate1 = hose_rate1 →
  rate2 = hose_rate2 →
  count1 = hoses_count1 →
  count2 = hoses_count2 →
  time_hours = 25 := 
sorry

end pool_fill_time_l778_778220


namespace geometric_arithmetic_seq_unique_ratio_l778_778691

variable (d : ℚ) (q : ℚ) (k : ℤ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_pos : 0 < q) (h_q_lt_one : q < 1)
variable (h_integer : 14 / (1 + q + q^2) = k)

theorem geometric_arithmetic_seq_unique_ratio :
  q = 1 / 2 :=
by
  sorry

end geometric_arithmetic_seq_unique_ratio_l778_778691


namespace sum_of_coefficients_l778_778247

theorem sum_of_coefficients (m : ℝ) (h : ∫ x in 1..m, (2 * x - 1) = 6) :
  (1 - 2) ^ (3 * m) = -1 :=
by
  sorry

end sum_of_coefficients_l778_778247


namespace ellipse_equation_and_fixed_point_proof_l778_778282

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778282


namespace quadrilateral_area_OBEC_is_22_5_l778_778558

-- Definition of the points and lines based on the conditions
def A := (0 : ℝ, 0 : ℝ)  -- Since A is where the line intersects the positive x-axis, its y-coordinate is 0.
def B := (0 : ℝ, 9 : ℝ)  -- From the problem, B is found at (0, 9)
def C := (10 : ℝ, 0 : ℝ) -- Point C given in the problem
def E := (4.5 : ℝ, 4.5 : ℝ) -- Point E found during solution

-- Function to calculate the area of triangle given vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- Function to subtract the area of two triangles to get the quadrilateral area
def quadrilateral_area (O B E C : ℝ × ℝ) : ℝ :=
  triangle_area O B C - triangle_area O E C

-- The statement proving the quadrilateral area is 22.5 given the conditions
theorem quadrilateral_area_OBEC_is_22_5 :
  quadrilateral_area A B E C = 22.5 :=
sorry

end quadrilateral_area_OBEC_is_22_5_l778_778558


namespace required_brick_volume_l778_778968

theorem required_brick_volume :
  let height := 4 / 12 -- in feet
  let length := 6 -- in feet
  let thickness := 4 / 12 -- in feet
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  rounded_volume = 1 := 
by
  let height := 1 / 3
  let length := 6
  let thickness := 1 / 3
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  show rounded_volume = 1
  sorry

end required_brick_volume_l778_778968


namespace crystal_barrette_sets_l778_778813

-- Definitional and situational context
def cost_of_barrette : ℕ := 3
def cost_of_comb : ℕ := 1
def kristine_total_cost : ℕ := 4
def total_spent : ℕ := 14

-- The Lean 4 theorem statement to prove that Crystal bought 3 sets of barrettes
theorem crystal_barrette_sets (x : ℕ) 
  (kristine_cost : kristine_total_cost = cost_of_barrette + cost_of_comb + 1)
  (total_cost_eq : kristine_total_cost + (x * cost_of_barrette + cost_of_comb) = total_spent) 
  : x = 3 := 
sorry

end crystal_barrette_sets_l778_778813


namespace compare_cosine_ratios_l778_778183

theorem compare_cosine_ratios :
  (cos 2014 * pi / 180) / (cos 2015 * pi / 180) < (cos 2016 * pi / 180) / (cos 2017 * pi / 180) :=
by
  sorry

end compare_cosine_ratios_l778_778183


namespace night_shift_hours_l778_778967

theorem night_shift_hours
  (hours_first_guard : ℕ := 3)
  (hours_last_guard : ℕ := 2)
  (hours_each_middle_guard : ℕ := 2) :
  hours_first_guard + 2 * hours_each_middle_guard + hours_last_guard = 9 :=
by 
  sorry

end night_shift_hours_l778_778967


namespace ellipse_equation_proof_HN_fixed_point_l778_778322

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778322


namespace basketball_free_throws_l778_778828

theorem basketball_free_throws
  (a b x : ℕ)
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a)
  (h3 : 2 * a + 3 * b + x = 72)
  : x = 24 := by
  sorry

end basketball_free_throws_l778_778828


namespace measure_of_angle_f_l778_778770

theorem measure_of_angle_f (angle_D angle_E angle_F : ℝ)
  (h1 : angle_D = 75)
  (h2 : angle_E = 4 * angle_F + 30)
  (h3 : angle_D + angle_E + angle_F = 180) : 
  angle_F = 15 :=
by
  sorry

end measure_of_angle_f_l778_778770


namespace crayons_total_l778_778100

def crayons_per_child := 6
def number_of_children := 12
def total_crayons := 72

theorem crayons_total :
  crayons_per_child * number_of_children = total_crayons := by
  sorry

end crayons_total_l778_778100


namespace a_plus_b_plus_one_eq_zero_l778_778834

open Real

noncomputable def incenter_on_y_eq_x (a b : ℝ) : Prop :=
  let x1 := (-a + sqrt(a^2 - 4 * b)) / 2
  let x2 := (-a - sqrt(a^2 - 4 * b)) / 2
  let C := (0, b)
  ∃ m : ℝ, m = -a / 2 ∧ (sqrt ((m - x1)^2 + m^2) = sqrt ((m - 0)^2 + (m - b)^2))

theorem a_plus_b_plus_one_eq_zero {a b : ℝ} (h : incenter_on_y_eq_x a b) : a + b + 1 = 0 :=
sorry

end a_plus_b_plus_one_eq_zero_l778_778834


namespace points_are_collinear_l778_778711

-- Definition of the ellipse Γ
def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

-- Points A, B on the ellipse
def A := (-4 : ℝ, 0 : ℝ)
def B := (4 : ℝ, 0 : ℝ)

-- Define point C outside the ellipse
def outside (C : ℝ × ℝ) : Prop :=
  ¬ ellipse C.fst C.snd

-- Lines CA and CB intersect the ellipse at points P and Q respectively
def intersects (C A B : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  ellipse P.fst P.snd ∧ ellipse Q.fst Q.snd

-- Tangents and their intersections
def tangent_at (P : ℝ × ℝ) (Q : ℝ × ℝ) (R : ℝ × ℝ) : Prop :=
  ellipse P.fst P.snd ∧ ellipse Q.fst Q.snd ∧ ellipse R.fst R.snd

-- Collinearity of points
def collinear (R C S : ℝ × ℝ) : Prop :=
  ∃ (m b : ℝ), (R.snd = m * R.fst + b) ∧ (C.snd = m * C.fst + b) ∧ (S.snd = m * S.fst + b)

theorem points_are_collinear (C P Q R S : ℝ × ℝ) :
  ellipse A.fst A.snd ∧ ellipse B.fst B.snd ∧
  outside C ∧
  intersects C A B P Q ∧
  tangent_at A Q R ∧
  tangent_at B P S →
  collinear R C S := by
  sorry

end points_are_collinear_l778_778711


namespace range_of_m_l778_778230

-- Define the function and properties given in the problem
variable (f : ℝ → ℝ)
variable (m : ℝ)

-- Conditions
axiom f_defined : ∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≠ none   -- f is defined on [-2,2]
axiom f_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = f x   -- f is an even function
axiom f_strictly_decreasing : ∀ a b, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ a ≠ b → (f a - f b) / (a - b) < 0   -- f is strictly decreasing

-- Proof problem
theorem range_of_m : f (1 - m) < f m → -1 ≤ m ∧ m < 1 / 2 :=
sry

end range_of_m_l778_778230


namespace general_term_a_n_sum_b_n_l778_778213

-- Defining the sequence {a_n} and its associated properties
def a_n (n : ℕ) : ℝ := 2 * n - 1

def S_n (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), a_n k

-- Condition: a_n > 0
axiom a_n_pos (n : ℕ) : a_n n > 0

-- Condition: a_n = 2 * sqrt(S_n) - 1
axiom a_n_sqrt (n : ℕ) : a_n n = 2 * real.sqrt (S_n n) - 1

-- Proof Problem 1: Prove that a_n = 2n - 1
theorem general_term_a_n (n : ℕ) : a_n n = 2 * n - 1 := sorry

-- Defining the sequence {b_n}
def b_n (n : ℕ) : ℝ := a_n n / 3^n

-- Defining the sum of the first n terms of {b_n}
def T_n (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), b_n k

-- Proof Problem 2: Prove that T_n = 1 - (n + 1)/3^n
theorem sum_b_n (n : ℕ) : T_n n = 1 - (n + 1) / 3 ^ n := sorry

end general_term_a_n_sum_b_n_l778_778213


namespace range_of_m_l778_778722

theorem range_of_m (m : ℝ) :
  ( ∀ x : ℝ, |x + m| ≤ 4 → -2 ≤ x ∧ x ≤ 8) ↔ -4 ≤ m ∧ m ≤ -2 := 
by
  sorry

end range_of_m_l778_778722


namespace surface_area_of_sphere_O_l778_778238

-- Lean declaration of the problem
noncomputable def trirectangular_tetrahedron_surface_area : Prop :=
  ∃ (P A B C : ℝ → ℝ → ℝ) (O : ℝ → ℝ → ℝ → ℝ) 
    (PA PB PC : ℝ) (r : ℝ), 
    (PA = 1) ∧ (PB = 1) ∧ (PC = 1) ∧ 
    (P × A = 0) ∧ (P × B = 0) ∧ (P × C = 0) ∧
    (∀ x y z, O x y z = 0 → 
      4 * (∏ (xi ∈ [x, y, z], (xi - ∏₂(vi ∈ [x, y, z], vi) / 2))^2) = 4 * 3) ∧
    (4 * (π * (r ^ 2)) = 3 * π)

theorem surface_area_of_sphere_O :
  trirectangular_tetrahedron_surface_area :=
sorry

end surface_area_of_sphere_O_l778_778238


namespace perpendicular_condition_l778_778221

variables (a : ℝ)
def vec_a : ℝ × ℝ × ℝ := (1, 2, -3)
def vec_b : ℝ × ℝ × ℝ := (2, a - 1, a^2 - (1/3))

theorem perpendicular_condition :
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 + vec_a.3 * vec_b.3 = 0) ↔ 
  (a = 1 ∨ a = -1/3) :=
by
  sorry

end perpendicular_condition_l778_778221


namespace overlap_area_after_5_seconds_l778_778605

theorem overlap_area_after_5_seconds (a : ℝ) (P Q C : ℝ × ℝ) (t : ℝ) (speed : ℝ) (initial_area : ℝ) :
  a = 20 * Real.sqrt 2 →
  t = 5 →
  speed = 0.25 →
  initial_area = 100 →
  let P_T := (P.1 + speed * t * a, P.2)
  let Q_T := (Q.1, Q.2 + speed * t * a)
  let area := 0.125 * a^2 in
  let overlap_after_5_seconds := -- area calculation based on positions of P_T and Q_T
  overlap_after_5_seconds = 575 :=
sorry

end overlap_area_after_5_seconds_l778_778605


namespace number_of_mixed_groups_l778_778883

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l778_778883


namespace min_value_of_f_l778_778432

noncomputable def f (x : ℝ) : ℝ := x + 2 / (2 * x + 1) - 3 / 2

theorem min_value_of_f : ∀ x > 0, ∃ c : ℝ, (∀ y, f y ≥ c) ∧ (f x = c) :=
by
  assume x hx,
  use 0,
  sorry

end min_value_of_f_l778_778432


namespace ellipse_and_fixed_point_l778_778302

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778302


namespace min_delta_and_min_ab_sum_l778_778248

theorem min_delta_and_min_ab_sum:
  ∃ a b : ℕ, (1 ≤ a) ∧ (1 ≤ b) ∧ (36 * a - 81 * b = 9) ∧ 
  (∀ a' b' : ℕ, (1 ≤ a') ∧ (1 ≤ b') ∧ (36 * a' - 81 * b' = 9) → a + b ≤ a' + b') ∧ 
  (a = 7) ∧ (b = 3) :=
begin
  existsi 7,
  existsi 3,
  split, { exact nat.one_le_of_lt (by norm_num) },
  split, { exact nat.one_le_of_lt (by norm_num) },
  split, { norm_num },
  split,
  { intros a' b' h1 h2 h3, 
    sorry },
  split, { refl },
  { refl },
end

end min_delta_and_min_ab_sum_l778_778248


namespace scientific_notation_of_50000_l778_778469

theorem scientific_notation_of_50000 :
  50000 = 5 * 10^4 :=
sorry

end scientific_notation_of_50000_l778_778469


namespace BF_eq_AF_plus_CF_l778_778383

variables {A B C D E F : Type} [noncomputable] (triangle : A × A × A)
variables (AB AC BC : ℝ) (cond1 : AB < AC) (cond2 : AC < BC)
variables (D : A) (D_on_BC : D = BC)
variables (E : A) (E_on_ext_BA : E = BA)
variables (BD_BE_AC : BD = BE ∧ BE = AC)
variables (circ_BDE : circle (B, D, E)) (circ_ABC : circle (A, B, C))
variables (F : A) (inter_circles : F ∈ (circ_BDE ∩ circ_ABC))

theorem BF_eq_AF_plus_CF (AB AC BC : ℝ) (cond1 : AB < AC) (cond2 : AC < BC) 
  (D : Type) (D_on_BC : D = BC) 
  (E : Type) (E_on_ext_BA : E = BA) 
  (BD_BE_AC : BD = BE ∧ BE = AC)
  (circ_BDE : circle (B, D, E)) (circ_ABC : circle (A, B, C))
  (F : Type) (inter_circles : F ∈ (circ_BDE ∩ circ_ABC)) : 
  BF = AF + CF := 
sorry

end BF_eq_AF_plus_CF_l778_778383


namespace girls_bought_balloons_l778_778992

theorem girls_bought_balloons (initial_balloons boys_bought girls_bought remaining_balloons : ℕ)
  (h1 : initial_balloons = 36)
  (h2 : boys_bought = 3)
  (h3 : remaining_balloons = 21)
  (h4 : initial_balloons - remaining_balloons = boys_bought + girls_bought) :
  girls_bought = 12 := by
  sorry

end girls_bought_balloons_l778_778992


namespace document_sequence_count_l778_778170

theorem document_sequence_count (n : ℕ) : 
  let C_n := Mathbin.choose (2 * n) n / (n + 1) 
  in 
  let valid_paths := (C_n : ℕ)
  in
  valid_paths = Mathbin.choose (2 * n) n / (n + 1) := 
by
  sorry

end document_sequence_count_l778_778170


namespace birds_on_fence_l778_778743

theorem birds_on_fence (B x : ℝ) (h1 : sqrt B = x) (h2 : 2 * x^2 + 10 = 50) : B = 20 :=
by
  sorry

end birds_on_fence_l778_778743


namespace attend_both_reunions_l778_778505

variable (total_guests oates_guests hall_guests both_guests : ℕ)

axiom total_guests_condition : total_guests = 150
axiom oates_guests_condition : oates_guests = 70
axiom hall_guests_condition : hall_guests = 52
axiom union_condition : oates_guests + hall_guests - both_guests = total_guests

theorem attend_both_reunions :
  both_guests = 28 :=
by
  rw [total_guests_condition, oates_guests_condition, hall_guests_condition, union_condition]
  sorry

end attend_both_reunions_l778_778505


namespace digits_sum_unique_l778_778510

variable (A B C D E F G H : ℕ)

theorem digits_sum_unique :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧
  F ≠ G ∧ F ≠ H ∧
  G ≠ H ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  0 ≤ E ∧ E ≤ 9 ∧ 0 ≤ F ∧ F ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ H ∧ H ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D) + (E * 1000 + F * 100 + G * 10 + H) = 10652 ∧
  A = 9 ∧ B = 5 ∧ C = 6 ∧ D = 7 ∧
  E = 1 ∧ F = 0 ∧ G = 8 ∧ H = 5 :=
sorry

end digits_sum_unique_l778_778510


namespace find_quadratic_polynomial_l778_778655

-- Given conditions to construct a quadratic polynomial with real coefficients
noncomputable def quadratic_polynomial (a b c : ℂ) (h1 : a = 5 + 2 * complex.I) : polynomial ℂ :=
  3 * (X - C a) * (X - C (conj a))

-- The required proof problem statement
theorem find_quadratic_polynomial (x : ℂ) :
    quadratic_polynomial 5 2 0 rfl = 3 * X^2 - 30 * X + 87 :=
sorry

end find_quadratic_polynomial_l778_778655


namespace taxi_fare_l778_778493

theorem taxi_fare (x : ℝ) : 
  (2.40 + 2 * (x - 0.5) = 8) → x = 3.3 := by
  sorry

end taxi_fare_l778_778493


namespace hyperbola_eccentricity_l778_778720

theorem hyperbola_eccentricity {b : ℝ} (hb : b > 0)
  (F1 F2 : ℝ × ℝ) (hF1 : F1 = (-8 * b, 0)) (hF2 : F2 = (8 * b, 0))
  (P : ℝ × ℝ) (hP : P = (-4 * b, 4 * b ^ 2)) 
  (h_focus : ∃ m : ℝ, ∀ x y, y = m * x^2 → (0, 1) = (0, 1 / (4 * m))):
  ecc (conic_section_hyperbola b) = (8 * real.sqrt 7 / 21) :=
by 
  sorry

end hyperbola_eccentricity_l778_778720


namespace problem1_problem2_l778_778180

-- Problem 1
theorem problem1 : 5*Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 : (2*Real.sqrt 3 - 1)^2 + (Real.sqrt 24) / (Real.sqrt 2) = 13 - 2*Real.sqrt 3 := by
  sorry

end problem1_problem2_l778_778180


namespace min_value_sqrt_ab_sqrt_bc_sqrt_ca_l778_778244

theorem min_value_sqrt_ab_sqrt_bc_sqrt_ca (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : ab + bc + ca = a + b + c) (h5 : a + b + c > 0) : 
  \(\sqrt{ab} + \sqrt{bc} + \sqrt{ca} \ge 2\) := sorry

end min_value_sqrt_ab_sqrt_bc_sqrt_ca_l778_778244


namespace calculate_p_l778_778192

-- Define the function p(x, y)
def p (x y : ℝ) : ℝ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x ≥ 0 ∧ y < 0 then x^2 + y^2
  else if x < 0 ∧ y < 0 then x - 2 * y
  else 2 * x + 2 * y

-- Prove that p(p(1, -1), p(-5, -3)) = 3
theorem calculate_p : p (p 1 (-1)) (p (-5) (-3)) = 3 := by
  sorry

end calculate_p_l778_778192


namespace find_quadratic_polynomial_l778_778654

-- Given conditions to construct a quadratic polynomial with real coefficients
noncomputable def quadratic_polynomial (a b c : ℂ) (h1 : a = 5 + 2 * complex.I) : polynomial ℂ :=
  3 * (X - C a) * (X - C (conj a))

-- The required proof problem statement
theorem find_quadratic_polynomial (x : ℂ) :
    quadratic_polynomial 5 2 0 rfl = 3 * X^2 - 30 * X + 87 :=
sorry

end find_quadratic_polynomial_l778_778654


namespace remainder_is_zero_l778_778517

def f (x : ℝ) : ℝ := x^3 - 5 * x^2 + 2 * x + 8

theorem remainder_is_zero : f 2 = 0 := by
  sorry

end remainder_is_zero_l778_778517


namespace smallest_positive_multiple_l778_778088

theorem smallest_positive_multiple (x : ℕ) (h1 : 18 * 5 = 90) : 
  (∃ x, x > 0 ∧ x = 90 * 1) :=
begin
  use 1,
  split,
  { norm_num, },
  { rw h1, norm_num, },
end

end smallest_positive_multiple_l778_778088


namespace addition_problem_l778_778397

def are_digits : ℕ → Prop := λ n, n ≥ 0 ∧ n ≤ 9

theorem addition_problem (E F G H : ℕ) 
  (hE : are_digits E) (hF : are_digits F) 
  (hG : are_digits G) (hH : are_digits H) 
  (h_diff : E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H)
  (h1 : E + H = 10 ∨ E + H = 9 ∨ E + H = 11 ∨ E + H = 12 ∨ E + H = 13)
  (h2 : (H + E) % 10 = E + H - 10 * ((E + H) / 10)) : 
  E + H = 10 :=
by sorry

end addition_problem_l778_778397


namespace triangle_inequality_l778_778461

theorem triangle_inequality (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2 * x^2 + x + 1 in
  let b := 2 * x^3 + x^2 + 2 * x + 1 in
  let c := x^4 - 1 in
  a > c ∧ a > b ∧ a < b + c :=
by {
  let a := x^4 + x^3 + 2 * x^2 + x + 1,
  let b := 2 * x^3 + x^2 + 2 * x + 1,
  let c := x^4 - 1,
  sorry
}

end triangle_inequality_l778_778461


namespace correct_options_l778_778740

-- Define the function f(x)
def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

-- Statement of the required proofs
theorem correct_options :
  (∀ x, f x = Real.cos (2 * x - Real.pi / 6)) ∧ 
  (∀ x, f (x + Real.pi / 6) + f (-Real.pi / 2 - x) = 0) :=
by
  sorry

end correct_options_l778_778740


namespace tan_ratio_l778_778467

theorem tan_ratio (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2) 
  (h2 : cos x / sin y + cos y / sin x = 4) : 
  tan x / tan y + tan y / tan x = 4 := 
sorry

end tan_ratio_l778_778467


namespace intercepts_sum_eq_eight_l778_778838

theorem intercepts_sum_eq_eight :
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  a + b + c = 8 :=
by
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  -- Proof will go here
  sorry

end intercepts_sum_eq_eight_l778_778838


namespace infinite_n_integral_mean_and_stddev_l778_778456

noncomputable def arithmetic_mean (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  (n * a + d * (n * (n - 1) / 2)) / n

noncomputable def standard_deviation (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  let mean := (n * a + d * (n-1) / 2) / n in
  sqrt ((1 / n) * ∑ j in finset.range n, (d * ((j : ℤ) - (n-1) / 2))^2)

theorem infinite_n_integral_mean_and_stddev :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧
  ∀ (a : ℤ) (d : ℤ),
    let seq := (finset.range n).map (λ i, a + (i : ℤ) * d) in
    arithmetic_mean n a d ∈ ℤ ∧ standard_deviation n a d ∈ ℤ :=
begin
  sorry
end

end infinite_n_integral_mean_and_stddev_l778_778456


namespace master_bedroom_and_bath_area_l778_778922

-- Definitions of the problem conditions
def guest_bedroom_area : ℕ := 200
def two_guest_bedrooms_area : ℕ := 2 * guest_bedroom_area
def kitchen_guest_bath_living_area : ℕ := 600
def total_rent : ℕ := 3000
def cost_per_sq_ft : ℕ := 2
def total_area_of_house : ℕ := total_rent / cost_per_sq_ft
def expected_master_bedroom_and_bath_area : ℕ := 500

-- Theorem statement to prove the desired area
theorem master_bedroom_and_bath_area :
  total_area_of_house - (two_guest_bedrooms_area + kitchen_guest_bath_living_area) = expected_master_bedroom_and_bath_area :=
by
  sorry

end master_bedroom_and_bath_area_l778_778922


namespace BF_eq_AF_plus_CF_l778_778384

variables {A B C D E F : Type} [noncomputable] (triangle : A × A × A)
variables (AB AC BC : ℝ) (cond1 : AB < AC) (cond2 : AC < BC)
variables (D : A) (D_on_BC : D = BC)
variables (E : A) (E_on_ext_BA : E = BA)
variables (BD_BE_AC : BD = BE ∧ BE = AC)
variables (circ_BDE : circle (B, D, E)) (circ_ABC : circle (A, B, C))
variables (F : A) (inter_circles : F ∈ (circ_BDE ∩ circ_ABC))

theorem BF_eq_AF_plus_CF (AB AC BC : ℝ) (cond1 : AB < AC) (cond2 : AC < BC) 
  (D : Type) (D_on_BC : D = BC) 
  (E : Type) (E_on_ext_BA : E = BA) 
  (BD_BE_AC : BD = BE ∧ BE = AC)
  (circ_BDE : circle (B, D, E)) (circ_ABC : circle (A, B, C))
  (F : Type) (inter_circles : F ∈ (circ_BDE ∩ circ_ABC)) : 
  BF = AF + CF := 
sorry

end BF_eq_AF_plus_CF_l778_778384


namespace find_at_least_one_genuine_coin_l778_778562

theorem find_at_least_one_genuine_coin 
    (coins : Fin 100 → ℝ) 
    (genuine_count : ∑ i in Finset.univ.filter (λ i, is_genuine i), 1 = 30)
    (counterfeit_count : ∑ i in Finset.univ.filter (λ i, is_counterfeit i), 1 = 70)
    (genuine_mass : ∀ (i j : Fin 100), is_genuine i → is_genuine j → coins i = coins j)
    (counterfeit_mass_diff : ∀ (i j : Fin 100), is_counterfeit i → is_counterfeit j → i ≠ j → coins i ≠ coins j)
    (counterfeit_heavier : ∀ (i j : Fin 100), is_genuine i → is_counterfeit j → coins j > coins i) :
    ∃ n ≤ 70, (∃ i, is_genuine i) :=
by
  sorry

end find_at_least_one_genuine_coin_l778_778562


namespace soccer_ball_diameter_l778_778590

theorem soccer_ball_diameter 
  (h : ℝ)
  (s : ℝ)
  (d : ℝ)
  (h_eq : h = 1.25)
  (s_eq : s = 1)
  (d_eq : d = 0.23) : 2 * (d * h / (s - h)) = 0.46 :=
by
  sorry

end soccer_ball_diameter_l778_778590


namespace number_of_mixed_groups_l778_778880

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l778_778880


namespace incircle_touches_vertex_l778_778436

variable {F1 F2 M N P : Type}
variable {H : P ∈ hyperbola F1 F2 M N}

theorem incircle_touches_vertex (F1 F2 M N P : Type) (H : P ∈ hyperbola F1 F2 M N) :
  (incircle_touch_point P F1 F2 = M ∨ incircle_touch_point P F1 F2 = N) :=
sorry

end incircle_touches_vertex_l778_778436


namespace ellipse_and_fixed_point_l778_778301

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778301


namespace tan_sum_angles_l778_778613

theorem tan_sum_angles : (Real.tan (17 * Real.pi / 180) + Real.tan (28 * Real.pi / 180)) / (1 - Real.tan (17 * Real.pi / 180) * Real.tan (28 * Real.pi / 180)) = 1 := 
by sorry

end tan_sum_angles_l778_778613


namespace water_depth_tipped_l778_778130

def prism_height : ℝ := 8
def base_dim1 : ℝ := 2
def base_dim2 : ℝ := 5
def water_depth_initial : ℝ := 6
def base_dim3 : ℝ := 8

theorem water_depth_tipped :
  let volume_water := base_dim1 * base_dim2 * water_depth_initial,
      new_base_area := base_dim2 * base_dim3,
      new_water_depth := volume_water / new_base_area
  in new_water_depth = 1.5 := 
by
  sorry

end water_depth_tipped_l778_778130


namespace max_path_length_in_cuboid_l778_778554

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2).sqrt

def vertices : List (ℝ × ℝ × ℝ) :=
  [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 1), (2, 2, 0), (2, 0, 1), (0, 2, 1), (2, 2, 1)]

noncomputable def max_path_length : ℝ :=
  12 + 8 * Real.sqrt 2 + 3

theorem max_path_length_in_cuboid :
  ∃ (path : List (ℝ × ℝ × ℝ)), 
    path.length = 9 ∧
    List.Nodup path ∧
    (∀ v ∈ vertices, ∃ u ∈ path, u = v) ∧
    path.head = (0, 0, 0) ∧
    path.last = (0, 0, 0) ∧
    ∑ i in List.zipWith (λ p q => distance p q) path (List.tail path), id = max_path_length :=
sorry

end max_path_length_in_cuboid_l778_778554


namespace standard_normal_prob_gt_neg1_l778_778802

open ProbabilityTheory

noncomputable def standard_normal : ProbabilityDistribution :=
  normalPdf 0 1

theorem standard_normal_prob_gt_neg1 (p : ℝ) :
  (∀ (s : Set ℝ), standard_normal.prob s = ∫ x in s, normalPdf 0 1 x ∂volume) →
  (standard_normal.prob {x | x > 1} = p) →
  standard_normal.prob {x | x > -1} = 1 - p :=
by
  intros hProb hp
  have hSym : standard_normal.prob {x | x < -1} = p := sorry
  have hTotal : standard_normal.prob {x | x ≤ -1} = p := sorry
  have hComplement : standard_normal.prob {x | x > -1} = 1 - standard_normal.prob {x | x ≤ -1} := sorry
  exact hComplement.trans hTotal.trans hp.symm

end standard_normal_prob_gt_neg1_l778_778802


namespace factorial_divides_product_l778_778799

theorem factorial_divides_product 
  (n : ℕ) (a b : ℕ)
  (h1 : n ≥ 1)
  (h2 : ∀ p : ℕ, p.prime → p ∣ b → p > n) :
  n.factorial ∣ (List.prod (List.map (λ k, a + k * b) (List.range n))) :=
by
  sorry

end factorial_divides_product_l778_778799


namespace max_x_plus_y_l778_778695

variables {V : Type*} [inner_product_space ℝ V]
variables (OA OB OC : V)
variables (x y : ℝ)

-- Given conditions:
axiom norm_OA : ∥OA∥ = 1
axiom norm_OB : ∥OB∥ = 1
axiom norm_OC : ∥OC∥ = 1
axiom dot_product_OA_OB : inner_product_space.inner ℝ OA OB = 0
axiom OC_def : OC = x • OA + y • OB

-- Goal: Prove the maximum value of x + y is sqrt 2
theorem max_x_plus_y (h : x^2 + y^2 = 1) : x + y ≤ real.sqrt 2 :=
sorry

end max_x_plus_y_l778_778695


namespace geometric_body_with_same_views_is_sphere_l778_778082

-- Definitions for geometric bodies
inductive GeometricBody
| Sphere
| Cube
| RegularTetrahedron

def same_views (body : GeometricBody) : Prop :=
  match body with
  | GeometricBody.Sphere            => true
  | GeometricBody.Cube              => false
  | GeometricBody.RegularTetrahedron => false

-- Proof statement: The only geometric body whose three views are all the same is the Sphere.
theorem geometric_body_with_same_views_is_sphere :
  ∀ b : GeometricBody, same_views b → b = GeometricBody.Sphere :=
by
  intro b
  cases b
  case Sphere =>
    intro h
    refl
  case Cube =>
    intro h
    contradiction
  case RegularTetrahedron =>
    intro h
    contradiction

end geometric_body_with_same_views_is_sphere_l778_778082


namespace intercepts_sum_eq_eight_l778_778836

theorem intercepts_sum_eq_eight :
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  a + b + c = 8 :=
by
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  -- Proof will go here
  sorry

end intercepts_sum_eq_eight_l778_778836


namespace c_rent_share_l778_778944

-- Definitions based on conditions
def a_oxen := 10
def a_months := 7
def b_oxen := 12
def b_months := 5
def c_oxen := 15
def c_months := 3
def total_rent := 105

-- Calculate the shares in ox-months
def share_a := a_oxen * a_months
def share_b := b_oxen * b_months
def share_c := c_oxen * c_months

-- Calculate the total ox-months
def total_ox_months := share_a + share_b + share_c

-- Calculate the rent per ox-month
def rent_per_ox_month := total_rent / total_ox_months

-- Calculate the amount C should pay
def amount_c_should_pay := share_c * rent_per_ox_month

-- Prove the statement
theorem c_rent_share : amount_c_should_pay = 27 := by
  sorry

end c_rent_share_l778_778944


namespace buttons_in_fifth_box_l778_778936
-- Importing the Mathlib library

-- Malia's button sorting conditions 
def buttons_in_first_box : Nat := 1
def buttons_in_second_box : Nat := 3
def buttons_in_third_box : Nat := 9
def buttons_in_fourth_box : Nat := 27
def buttons_in_sixth_box : Nat := 243

-- Common ratio of the geometric sequence
def common_ratio : Nat := 3

-- Statement to prove that the number of buttons in the fifth box is 81
theorem buttons_in_fifth_box : 
  buttons_in_fifth_box = buttons_in_fourth_box * common_ratio :=
by
  sorry

end buttons_in_fifth_box_l778_778936


namespace count_valid_n_l778_778671

theorem count_valid_n (n : ℤ) : 
  (∃ (count : ℕ), count = 9 ∧ 
  (∀ n : ℤ, (∃ k : ℤ, 4000 * (2^n) * (5^(-n)) = k) → count = 9)) := 
by
  sorry

end count_valid_n_l778_778671


namespace alternating_binom_sum_zero_l778_778175

theorem alternating_binom_sum_zero :
  ∑ k in Finset.range (51), (-1)^k * (Nat.choose 50 k) = 0 :=
by sorry

end alternating_binom_sum_zero_l778_778175


namespace impossible_to_obtain_1998_l778_778982

-- Define the conditions and proof problem statement
theorem impossible_to_obtain_1998^7_from_7^1998
  (initial_number : ℕ := 7^1998)
  (final_number : ℕ := 1998^7) :
  ∀ operations : list (ℕ → ℕ), -- list of operations
    (∀ N,  -- each operation is of the form described
      let a := N / 10, b := N % 10 in
      N' = a + 5 * b) →
    ¬ (foldl (λ N operation, operation N) initial_number operations = final_number) := 
by
  sorry

end impossible_to_obtain_1998_l778_778982


namespace dice_product_144_probability_l778_778525

theorem dice_product_144_probability :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      events := {abc : ℕ × ℕ × ℕ | abc.1 ∈ S ∧ abc.2 ∈ S ∧ abc.3 ∈ S ∧ abc.1 * abc.2 * abc.3 = 144} in
  (|events| : ℝ) / 216 = 1 / 72 :=
by
  -- Assumption of finite set cardinality and mention of corresponding probabilities can be filled here
  sorry

end dice_product_144_probability_l778_778525


namespace rationalize_denominator_l778_778818

-- Definitions based on conditions in the problem
def expr := (7 : ℝ) / (2 * real.sqrt 98)
def rationalized_expr := real.sqrt 2 / 4

-- The proof problem: show that the given expression equals the rationalized form
theorem rationalize_denominator : expr = rationalized_expr :=
by
  -- Proof omitted
  sorry

end rationalize_denominator_l778_778818


namespace arithmetic_sequence_value_l778_778627

theorem arithmetic_sequence_value 
    (a1 : ℤ) (a2 a3 a4 : ℤ) (a1_a4 : a1 = 18) 
    (b1 b2 b3 : ℤ) 
    (b1_b3 : b3 - b2 = 6 ∧ b2 - b1 = 6 ∧ b2 = 15 ∧ b3 = 21)
    (b1_a3 : a3 = b1 - 6 ∧ a4 = a1 + (a3 - 18) / 3) 
    (c1 c2 c3 c4 : ℝ) 
    (c1_b3 : c1 = a4) 
    (c2 : c2 = -14) 
    (c4 : ∃ m, c4 = b1 - m * (6 :ℝ) + - 0.5) 
    (n : ℝ) : 
    n = -12.5 := by 
  sorry

end arithmetic_sequence_value_l778_778627


namespace chess_club_mixed_groups_l778_778886

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l778_778886


namespace find_valid_pairs_l778_778647

def divides (a b : Nat) : Prop := ∃ k, b = a * k

def valid_pair (a b : Nat) : Prop :=
  divides (a^2 * b) (b^2 + 3 * a)

theorem find_valid_pairs :
  {ab | valid_pair ab.1 ab.2} = ({(1, 1), (1, 3)} : Set (Nat × Nat)) :=
by
  sorry

end find_valid_pairs_l778_778647


namespace perpendicular_parallel_equiv_l778_778727

variables {α β : Plane} {l : Line}

def perpendicular (l : Line) (p : Plane) := sorry
def parallel (p1 p2 : Plane) := sorry

theorem perpendicular_parallel_equiv (h1 : perpendicular l α) : (parallel α β ↔ perpendicular l β) :=
sorry

end perpendicular_parallel_equiv_l778_778727


namespace total_profit_at_100_max_profit_per_set_at_140_l778_778126

-- Definitions for conditions
def sales_volume (x : ℝ) := 15 - 0.1 * x
def fixed_price := 30
def floating_price (sales_vol : ℝ) := 10 / sales_vol
def supply_price (x : ℝ) := fixed_price + floating_price (sales_volume x)
def total_profit (x : ℝ) := sales_volume x * (x - supply_price x)

-- Problem 1 statement: Verify total profit when selling price is 100
theorem total_profit_at_100 : total_profit 100 = 340 :=
by
  sorry

-- Problem 2 statement: Determine price to maximize profit per set
def profit_per_set (x : ℝ) := x - supply_price x

theorem max_profit_per_set_at_140 : ∀ x ∈ set.Ioo 0 150, profit_per_set 140 = 100 :=
by
  sorry

end total_profit_at_100_max_profit_per_set_at_140_l778_778126


namespace count_valid_two_digit_nums_eq_sixteen_l778_778676

def valid_two_digit_num (N : ℕ) : Prop :=
  let a := N / 10 in
  let b := N % 10 in
  N ≥ 10 ∧ N < 100 ∧ ∃ (k : ℕ), (k > 0) ∧ (k * k = 9 * (a - b))

theorem count_valid_two_digit_nums_eq_sixteen :
  {N : ℕ | valid_two_digit_num N}.to_finset.card = 16 :=
begin
  sorry
end

end count_valid_two_digit_nums_eq_sixteen_l778_778676


namespace gcd_45345_34534_l778_778651

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end gcd_45345_34534_l778_778651


namespace super_cool_triangles_area_sum_l778_778571

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end super_cool_triangles_area_sum_l778_778571


namespace find_f_2019_l778_778833

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2019 (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) = f(x) + f(y))
  (h2 : f (1 / 2019) = 1) :
  f 2019 = -1 := 
sorry

end find_f_2019_l778_778833


namespace square_cannot_be_on_concentric_circles_l778_778011

theorem square_cannot_be_on_concentric_circles (a d : ℝ) (ha : a > 0) (hd : d > 0) :
  ¬(∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (x1^2 + y1^2 = a^2) ∧
    (x2^2 + y2^2 = (a + d)^2) ∧
    (x3^2 + y3^2 = (a + 2d)^2) ∧
    (x4^2 + y4^2 = (a + 3d)^2) ∧
    ((x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧ (x2, y2) ≠ (x3, y3) ∧ 
    (x2, y2) ≠ (x4, y4) ∧ (x3, y3) ≠ (x4, y4)) ∧
    (dist (x1, y1) (x2, y2) = dist (x2, y2) (x3, y3) ∧
    dist (x2, y2) (x3, y3) = dist (x3, y3) (x4, y4) ∧
    dist (x3, y3) (x4, y4) = dist (x4, y4) (x1, y1))) :=
sorry

end square_cannot_be_on_concentric_circles_l778_778011


namespace circle_d_not_externally_tangent_to_circle_c_l778_778182

noncomputable def circle_c_eqn (x y r : ℝ) : ℝ := 
  x^2 + y^2 - 3 * x + 5 * y - (r^2 - 17 / 2)

def circle_d_eqn (x y : ℝ) : ℝ := 
  x^2 + y^2 - 9

def center_c : ℝ × ℝ := (3 / 2, -5 / 2)

def radius_c (r : ℝ) : ℝ := r

def center_d : ℝ × ℝ := (0, 0)

def radius_d : ℝ := 3

def distance_centers : ℝ := real.sqrt ((3 / 2)^2 + (-5 / 2)^2)

theorem circle_d_not_externally_tangent_to_circle_c (r : ℝ) (h : r > 0) : 
  distance_centers < radius_d :=
by { -- Proof is not needed as per the instructions
sorry }

end circle_d_not_externally_tangent_to_circle_c_l778_778182


namespace hn_passes_fixed_point_l778_778264

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778264


namespace rectangle_circle_area_ratio_l778_778037

theorem rectangle_circle_area_ratio (w r : ℝ) (h1 : 2 * 2 * w + 2 * w = 2 * pi * r) :
  ((2 * w) * w) / (pi * r^2) = 2 * pi / 9 :=
by
  sorry

end rectangle_circle_area_ratio_l778_778037


namespace general_term_formula_l778_778032

theorem general_term_formula :
  ∀ n : ℕ, (0 < n) → 
  (-1)^n * (2*n + 1) / (2*n) = ((-1) : ℝ)^n * ((2*n + 1) : ℝ) / (2*n) :=
by {
  sorry
}

end general_term_formula_l778_778032


namespace limit_of_sin_cos_l778_778177

theorem limit_of_sin_cos (x : ℝ) :
  filter.tendsto (fun x => (2 * x * real.sin x) / (1 - real.cos x)) (nhds 0) (nhds 4) :=
sorry

end limit_of_sin_cos_l778_778177


namespace range_of_a_l778_778709

open Real

def f (x a : ℝ) : ℝ := sin x ^ 2 + 2 * a * x

def is_never_tangent (a : ℝ) : Prop :=
  ∀ (m : ℝ), ¬(∃ x : ℝ, f' x a = -1 ∧ sin' x ≠ 0)

theorem range_of_a (a : ℝ) : is_never_tangent a ↔ (a < -1 ∨ 0 < a) := 
sorry

end range_of_a_l778_778709


namespace greatest_number_is_154_l778_778539

-- Define the set s of 50 consecutive multiples of 2 with smallest number 56
def s : Finset ℕ :=
  Finset.range 50 |>.map (λ n => 56 + n * 2)

-- Define the smallest and greatest elements in the set s
def smallest_in_s := 56
def greatest_in_s := Finset.max' s ⟨56, by simp⟩

-- State the theorem to prove the greatest number is 154
theorem greatest_number_is_154 : greatest_in_s = 154 := 
sorry

end greatest_number_is_154_l778_778539


namespace probability_calculation_l778_778122

-- Define the problem conditions
def chips : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def exceeds_seven_sum (draws : List ℕ) : Prop :=
  draws.length = 3 ∧ draws.sum > 7

-- Define and prove the probability problem
def probability_3_draws_exceeds_7 (draw : List ℕ) (h : draw ⊆ chips ∧ draw.nodup ∧ draw.length = 3) : ℚ :=
  if exceeds_seven_sum draw then 4 / 21 else 0

-- Statement of the theorem
theorem probability_calculation : 
  ∀ (draw : List ℕ), draw ⊆ chips ∧ draw.nodup ∧ draw.length = 3 → probability_3_draws_exceeds_7 draw = 4 / 21 :=
sorry

end probability_calculation_l778_778122


namespace area_ratio_l778_778040

variables (l w r : ℝ)

-- Define the conditions
def perimeter_eq_circumference : Prop := 2 * l + 2 * w = 2 * π * r
def length_eq_twice_width : Prop := l = 2 * w

-- Define the theorem to prove the ratio of the areas
theorem area_ratio (h1 : perimeter_eq_circumference l w r) (h2 : length_eq_twice_width l w) :
  (l * w) / (π * r^2) = 2 * π / 9 :=
sorry

end area_ratio_l778_778040


namespace part_I_part_II_part_III_l778_778236

noncomputable def a : ℕ → ℚ := sorry  -- We will define it as part of the proof.
axiom a_1_spec : a 1 = 1 / 2
axiom recurrence_relation (n : ℕ) (hn : n > 0) : 2 * a (n + 1) - a n = n

def b (n : ℕ) : ℚ := a (n + 1) - a n - 1

theorem part_I :
  a 2 = 3 / 4 ∧
  a 3 = 11 / 8 ∧
  a 4 = 35 / 16 := sorry

theorem part_II : geometric_sequence b :=
suffices b 1 = -3 / 4, by
  { have hb : ∀ n, b (n + 1) = (1 / 2) * b n := sorry,
    exact λ n : ℕ, 
      begin
        induction n with n ih,
        exact this,
        rw [hb, ih]
      end }

theorem part_III (n : ℕ) : a n = n - 2 + 3 / (2 ^ n) := sorry

end part_I_part_II_part_III_l778_778236


namespace shaded_area_ratio_l778_778184

theorem shaded_area_ratio (side_length : ℝ) (h1 : side_length = 60) 
    (h2 : ∀ (subdivided_rows subdivided_columns : list ℝ), 
        (subdivided_rows = [20, 20, 20]) ∧ 
        (subdivided_columns = [15, 15, 15, 15])) :
    (area_ratio : ℝ) (h3: area_ratio = (1 / 4)) := 
by 
    have large_square_area : ℝ := side_length * side_length, 
    have shaded_region_side_length : ℝ := side_length / 2, 
    have shaded_region_area : ℝ := shaded_region_side_length * shaded_region_side_length, 
    have correct_ratio : ℝ := shaded_region_area / large_square_area,
    have h : correct_ratio = (1 / 4) := by sorry,
    exact h

end shaded_area_ratio_l778_778184


namespace tan_2x_period_l778_778857

-- Define the tangent function and its standard period
def tan_period : ℝ := Real.pi

-- Define the function y = tan 2x
def f (x : ℝ) := Real.tan (2 * x)

-- State the property to be proved: The period of f is π/2
theorem tan_2x_period : ∀ x: ℝ, f(x) = f(x + π/2) := 
sorry

end tan_2x_period_l778_778857


namespace number_of_houses_on_block_l778_778559

theorem number_of_houses_on_block 
  (total_mail : ℕ) 
  (white_mailboxes : ℕ) 
  (red_mailboxes : ℕ) 
  (mail_per_house : ℕ) 
  (total_white_mail : ℕ) 
  (total_red_mail : ℕ) 
  (remaining_mail : ℕ)
  (additional_houses : ℕ)
  (total_houses : ℕ) :
  total_mail = 48 ∧ 
  white_mailboxes = 2 ∧ 
  red_mailboxes = 3 ∧ 
  mail_per_house = 6 ∧ 
  total_white_mail = white_mailboxes * mail_per_house ∧
  total_red_mail = red_mailboxes * mail_per_house ∧
  remaining_mail = total_mail - (total_white_mail + total_red_mail) ∧
  additional_houses = remaining_mail / mail_per_house ∧
  total_houses = white_mailboxes + red_mailboxes + additional_houses →
  total_houses = 8 :=
by 
  sorry

end number_of_houses_on_block_l778_778559


namespace hn_passes_fixed_point_l778_778262

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778262


namespace find_100m_10n_p_l778_778413

noncomputable def ABC_triangle_AB_AC_condition (A B C : Point) : Prop :=
  distance A B = 3 ∧ distance A C = 4

noncomputable def no_point_D_Euler_line_condition (A B C D : Point) : Prop :=
  D ≠ A ∧ ¬ collinear B C D ∧
  ∀ D, (Euler_line_triangle A B C) = (Euler_line_triangle D B C) → false

theorem find_100m_10n_p
  (A B C : Point)
  (hABC : ABC_triangle_AB_AC_condition A B C)
  (hNoPointD : no_point_D_Euler_line_condition A B C)
  (BC_square_product : ℝ) :
  (∃ (m n p : ℤ),
    BC_square_product = m + n * real.sqrt p ∧
    p > 0 ∧
    ¬ ∃ q : ℤ, q > 1 ∧ q * q ∣ p) →
  100 * m + 10 * n + p = 10782 :=
sorry

end find_100m_10n_p_l778_778413


namespace sum_of_areas_of_super_cool_triangles_l778_778575

def is_super_cool_triangle (a b : ℕ) : Prop :=
  (a * b / 2 = 3 * (a + b))

theorem sum_of_areas_of_super_cool_triangles :
  (∑ p in {p : ℕ × ℕ | is_super_cool_triangle p.1 p.2}, (p.1 * p.2) / 2) = 471 := 
by
  sorry

end sum_of_areas_of_super_cool_triangles_l778_778575


namespace find_g2_l778_778425

noncomputable def g (x : ℝ) : ℝ := sorry -- Define the polynomial g(x) later
def condition (x : ℝ) := g (x - 2) + g x + g (x + 2) = (g x) ^ 2 / (4026 * x)

theorem find_g2 (g : ℝ → ℝ)
  (h_nonconstant : ∃ a b : ℝ, a ≠ 0 ∧ g = λ x, a * x + b)
  (h_condition : ∀ x : ℝ, x ≠ 0 → condition x g) :
  g 2 = 12078 :=
sorry

end find_g2_l778_778425


namespace carpet_breadth_l778_778164

theorem carpet_breadth
  (b : ℝ)
  (h1 : ∀ b, ∃ l, l = 1.44 * b)
  (h2 : 4082.4 = 45 * ((1.40 * l) * (1.25 * b)))
  : b = 6.08 :=
by
  sorry

end carpet_breadth_l778_778164


namespace ellipse_equation_fixed_point_l778_778277

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778277


namespace problem_solution_l778_778633

def diamond (a b : ℝ) : ℝ := a ^ 2 * b - b ^ 3
def circ (a b : ℝ) : ℝ := a + Math.logBase b a - b ^ 2

theorem problem_solution : (diamond 5 3) / (circ 5 3) = -18 := 
by
  sorry

end problem_solution_l778_778633


namespace volume_of_tetrahedron_l778_778923

-- Definitions derived from the conditions
def a := (sqrt 34)
def b := (sqrt 15)
def c := (sqrt 66)

-- Proving volume of tetrahedron OABC
theorem volume_of_tetrahedron : 
  (1 / 6) * a * b * c = sqrt 5655 := 
by sorry

end volume_of_tetrahedron_l778_778923


namespace range_of_S_l778_778095

-- Definition of the condition about the starting fare.
def starting_fare := 10
-- Definition of the initial distance with the starting fare.
def initial_distance := 3
-- Definition of fare increment for every additional 0.6 km.
def increment_distance := 0.6
def increment_fare := 1

-- Definition of the total fare observed.
def total_fare := 34

-- Definition that we need to prove the range of S.
theorem range_of_S (S : ℝ) 
  (h1 : total_fare = starting_fare + ((S - initial_distance) / increment_distance) * increment_fare) : 
  17.4 ≤ S ∧ S < 18 :=
sorry

end range_of_S_l778_778095


namespace number_of_mixed_groups_l778_778881

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l778_778881


namespace parabola_intercepts_sum_l778_778840

noncomputable def a : ℝ := 5

noncomputable def b : ℝ := (9 + Real.sqrt 21) / 6

noncomputable def c : ℝ := (9 - Real.sqrt 21) / 6

theorem parabola_intercepts_sum : a + b + c = 8 := by
  -- definition of a
  have ha : a = 5 := rfl
  
  -- definitions of b and c from roots of 3y^2 - 9y + 5 = 0
  have hb : b = (9 + Real.sqrt 21) / 6 := rfl
  have hc : c = (9 - Real.sqrt 21) / 6 := rfl
  
  -- Vieta's formulas implies b + c = 3
  have hb_c : b + c = 3 := by
    calc
    b + c = (9 + Real.sqrt 21) / 6 + (9 - Real.sqrt 21) / 6 : by rw [hb, hc]
    ... = (9 + 9) / 6 : by ring
    ... = 18 / 6 : by norm_num
    ... = 3 : by norm_num
  
  -- Sum a + b + c
  calc
  a + b + c = 5 + (b + c) : by rw [ha]
  ... = 5 + 3 : by rw [hb_c]
  ... = 8 : by norm_num

end parabola_intercepts_sum_l778_778840


namespace ellipse_solution_geometry_solution_l778_778312

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778312


namespace largest_subset_non_triple_l778_778594

theorem largest_subset_non_triple (S : Set ℕ) (hS : ∀ x ∈ S, ∀ y ∈ S, x ≠ 3 * y) :
  S ⊆ {n | n ≤ 100} → S.card ≤ 76 :=
by
  sorry

end largest_subset_non_triple_l778_778594


namespace exponential_equation_solution_l778_778374

theorem exponential_equation_solution (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = (3 / 5)^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end exponential_equation_solution_l778_778374


namespace find_value_of_expression_l778_778249

-- Given conditions
variable (a : ℝ)
variable (h_root : a^2 + 2 * a - 2 = 0)

-- Mathematically equivalent proof problem
theorem find_value_of_expression : 3 * a^2 + 6 * a + 2023 = 2029 :=
by
  sorry

end find_value_of_expression_l778_778249


namespace tan_2x_period_l778_778856

-- Define the tangent function and its standard period
def tan_period : ℝ := Real.pi

-- Define the function y = tan 2x
def f (x : ℝ) := Real.tan (2 * x)

-- State the property to be proved: The period of f is π/2
theorem tan_2x_period : ∀ x: ℝ, f(x) = f(x + π/2) := 
sorry

end tan_2x_period_l778_778856


namespace visitors_inversely_proportional_l778_778822

theorem visitors_inversely_proportional (k : ℝ) (v₁ v₂ t₁ t₂ : ℝ) (h1 : v₁ * t₁ = k) (h2 : t₁ = 20) (h3 : v₁ = 150) (h4 : t₂ = 30) : v₂ = 100 :=
by
  -- This is a placeholder line; the actual proof would go here.
  sorry

end visitors_inversely_proportional_l778_778822


namespace S100_value_l778_778404

def sequence (a : ℕ → ℕ) : ℕ → ℕ
| 0       := 2
| 1       := 3
| (n + 2) := if even n then 2 * a n else a n

def sumSeq (a : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum a

theorem S100_value : sumSeq sequence 100 = 3 * 2^50 + 97 := by
  sorry

end S100_value_l778_778404


namespace new_avg_after_discarding_l778_778472

noncomputable def original_average : ℝ := 56
noncomputable def num_of_values : ℕ := 50
noncomputable def discarded_nums_sum : ℝ := 45 + 55
noncomputable def remaining_values : ℕ := num_of_values - 2
noncomputable def new_average : ℝ := 56.25

theorem new_avg_after_discarding :
  let original_sum := original_average * num_of_values in
  let new_sum := original_sum - discarded_nums_sum in
  new_sum / remaining_values = new_average :=
by
  sorry

end new_avg_after_discarding_l778_778472


namespace a18_is_3421_l778_778588

theorem a18_is_3421 :
  let four_digit_numbers := (list.permutations [1, 2, 3, 4]).map (λ l, l.foldl (λ acc d, 10 * acc + d) 0)
  let sorted_four_digit_numbers := four_digit_numbers.qsort (≤)
  sorted_four_digit_numbers.nth 17 = some 3421 :=
by sorry

end a18_is_3421_l778_778588


namespace ellipse_equation_proof_HN_fixed_point_l778_778315

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778315


namespace part_i_part_ii_l778_778784

theorem part_i (a b : ℝ) (m n : ℕ) (h₀ : a ≥ b) (h₁ : b > 0) (h₂ : a + b = 1) (h₃ : m < n) :
  a^m - a^n ≥ b^m - b^n ∧ b^m - b^n > 0 :=
sorry

theorem part_ii (a b : ℝ) (n : ℕ) (h₀ : a ≥ b) (h₁ : b > 0) (h₂ : a + b = 1) :
  ∀ x ∈ set.Icc (-1 : ℝ) 1, has_root (polynomial.C a^n + polynomial.X * polynomial.C b^n + polynomial.C (-1) * (polynomial.X ^ 2)) :=
sorry

end part_i_part_ii_l778_778784


namespace evaluate_g_l778_778187

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_l778_778187


namespace diplomats_not_speaking_Russian_l778_778001

variable (D : ℕ) (L : ℕ) (R : ℕ) (neither : ℕ) (both : ℕ)

-- Given conditions
def condition1 : D = 120 := sorry
def condition2 : L = 20 := sorry
def condition3 : neither = 24 := sorry
def condition4 : both = 12 := sorry

-- Proof statement
theorem diplomats_not_speaking_Russian :
  ∃ N : ℕ, N = D - R → N = 20 :=
by
  ext N
  have h_total : D = L + R - both + neither := by sorry
  have h_total_value : 120 = 20 + R - 12 + 24 := by sorry
  have h_R : R = 100 := by sorry
  have h_N : N = D - R := by sorry
  exact Exists.intro 20 (by rwa [h_N])

end diplomats_not_speaking_Russian_l778_778001


namespace pirate_coins_l778_778155

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem pirate_coins (x : ℕ) (h : ∃ n, n > 0 ∧ ∀ k, 1 ≤ k ∧ k ≤ 15 → x * n / 15 ^ (k-1) % k = 0) :
  x = 9009 := sorry

end pirate_coins_l778_778155


namespace trigonometric_identity_l778_778701

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) - Real.sin α * Real.cos α = -1 :=
sorry

end trigonometric_identity_l778_778701


namespace equilateral_not_all_white_possible_l778_778160
open Function

-- Define what it means to have an equilateral division with a given number of smaller triangles
structure EquilateralDivision where
  triangles : Finset (Finset ℕ)  -- representing each smaller triangle by a unique identifier
  is_black : Finset ℕ           -- a set containing exactly one black triangle
  lines_parallel_to_sides : Finset (Finset ℕ) -- each set contains triangles intersected by a parallel line

-- Define a condition that checks whether we can make all triangles white
noncomputable def all_white_possible (div: EquilateralDivision) : Prop :=
  ∃ (steps : ℕ) (lines : Vector (Finset ℕ) steps), 
    ∀ t ∈ div.triangles, t ∉ (lines.foldl (λ acc line, if acc ∈ div.is_black then acc \ line else acc) div.is_black)

-- The main statement
theorem equilateral_not_all_white_possible (div : EquilateralDivision) :
  div.is_black.card = 1 → ¬ all_white_possible div := by
  sorry

end equilateral_not_all_white_possible_l778_778160


namespace consecutive_integer_sets_l778_778866

theorem consecutive_integer_sets (S : ℕ) (hS : S = 180) : 
  ∃ n_values : Finset ℕ, 
  (∀ n ∈ n_values, (∃ a : ℕ, n * (2 * a + n - 1) = 2 * S) ∧ n >= 2) ∧ 
  n_values.card = 4 :=
by
  sorry

end consecutive_integer_sets_l778_778866


namespace minimum_trips_proof_l778_778059

noncomputable def minimum_trips (cargo : ℕ) (weight_limit : ℕ) : ℕ :=
  by 
  have h1 : cargo = 400 := sorry -- given condition: there are 400 tons of cargo
  have h2 : weight_limit = 10 := sorry -- given condition: truck capacity is 10 tons
  have h3 : ∀ w, w % 1 = 0 ∧ w ≤ 10 ∧ (distinct_weights w) := sorry -- conditions: weights are multiples of quintals, distinct, and ≤ 10 tons
  sorry -- prove that minimum trips is 51

theorem minimum_trips_proof :
  minimum_trips 400 10 = 51 := 
  sorry

end minimum_trips_proof_l778_778059


namespace simplify_expression_l778_778675

theorem simplify_expression:
    (-2.4) - (-4.7) - (+0.5) + (-3.5) = -2.4 + 4.7 - 0.5 - 3.5 :=
by
  sorry

end simplify_expression_l778_778675


namespace shift_sine_cos_graph_l778_778819

theorem shift_sine_cos_graph (m : ℝ) (m_pos: m > 0):
    (∃ k : ℤ, f (x + m) at (0) = 1) → (m = π / 4) :=
by
    -- Define the function f
    let f := λ x, 2 * sin (2 * x + π / 3)
    -- Define the shifted function g
    let g := λ x, 2 * sin (2 * (x + m) + π / 3)
    -- Apply the conditions
    sorry

end shift_sine_cos_graph_l778_778819


namespace expression_value_l778_778519

theorem expression_value : 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end expression_value_l778_778519


namespace free_donut_coupons_total_l778_778817

theorem free_donut_coupons_total :
  let books_per_coupon := 5
  let quinn_books := 5 * 5 -- (2 books + 3 books) * 5 cycles
  let taylor_books := 1 + 4 * 9
  let jordan_books := 3 * 10
  let quinn_coupons := quinn_books / books_per_coupon
  let taylor_coupons := taylor_books / books_per_coupon
  let jordan_coupons := jordan_books / books_per_coupon
  quinn_coupons + taylor_coupons + jordan_coupons = 18 :=
by {
  let books_per_coupon := 5
  let quinn_books := 5 * 5 -- (2 books + 3 books) * 5 cycles
  let taylor_books := 1 + 4 * 9
  let jordan_books := 3 * 10
  let quinn_coupons := quinn_books / books_per_coupon
  let taylor_coupons := taylor_books / books_per_coupon
  let jordan_coupons := jordan_books / books_per_coupon
  have h_quinn_coupons : quinn_coupons = 5 := by sorry
  have h_taylor_coupons : taylor_coupons = 7 := by sorry
  have h_jordan_coupons : jordan_coupons = 6 := by sorry
  show quinn_coupons + taylor_coupons + jordan_coupons = 18, from by {
    calc
      5 + 7 + 6 = 18 : by norm_num
  }
}

end free_donut_coupons_total_l778_778817


namespace math_problem_l778_778717

noncomputable def f (ω x : ℝ) := sin (ω * x + π / 5)

theorem math_problem (ω : ℝ) (hω_pos : ω > 0)
(h_zeros : ∃ (cnt : ℕ), cnt = 5 ∧ ∀ x ∈ set.Icc 0 (2 * π), f ω x = 0 ↔ (∃ n : ℤ, ω * x + π/5 = n * π)) :
(12 / 5 ≤ ω ∧ ω < 29 / 10) ∧ 
(∀ x ∈ set.Ioo 0 (π / 10), (f ω) x ∈ set.Ioi 0) ∧
(3 ∃ (cnt : ℕ), cnt = 3 ∧ ∀ x ∈ set.Ioo 0 (2 * π), differentiable_at ℝ (f ω) x ∧ critical_point (f ω) x ∧ local_max (f ω x)) ∧
(∀ x ∈ set.Ioo 0 (2 * π), differentiable_at ℝ (f ω) x ∧ critical_point (f ω) x ∧ local_min (f ω x)) :=
sorry

end math_problem_l778_778717


namespace plane_is_equidistant_set_l778_778457

variables {α : Type*} [MetricSpace α]
variables (α : Set α) (A : α) (SA' : α) (S : α)

-- Condition: plane α and point A outside it
def plane (P : Set α) (x y z : α) : Prop := sorry
def is_outside (P : Set α) (x : α) : Prop := sorry

-- Given conditions in the problem
def unique_symmetric_point (P : Set α) (A : α) (S SA' : α) : Prop :=
  ∃ (A' : α), (seg_perpendicular P A A') ∧ (seg_intersect_plane P A A' S) ∧ (opposing_vectors S A SA')

def seg_perpendicular (P : Set α) (x y : α) : Prop := sorry
def seg_intersect_plane (P : Set α) (x y z : α) : Prop := sorry
def opposing_vectors (x y z : α) : Prop := sorry

-- Statement to prove: α is the set of points equidistant from A and A'
theorem plane_is_equidistant_set (α : Set α) (A : α) (unique_sym_point : unique_symmetric_point α A S SA') :
  ∀ (M : α), M ∈ α → dist M A = dist M SA' := 
sorry

end plane_is_equidistant_set_l778_778457


namespace carol_project_workers_l778_778999

theorem carol_project_workers (total_days : ℕ) (initial_days : ℕ) (initial_workers : ℕ) (completed_percent : ℕ) : 
  total_days = 40 → initial_days = 10 → initial_workers = 10 → completed_percent = 40 → 
  let remaining_days := total_days - initial_days in
  let remaining_percent := 100 - completed_percent in
  let work_rate_per_worker := (completed_percent : ℝ) / (initial_workers : ℝ * initial_days : ℝ) in
  let required_workers := (remaining_percent : ℝ) / (remaining_days : ℝ * work_rate_per_worker) in
  required_workers ≤ 5 :=
by
  intros h_total_days h_initial_days h_initial_workers h_completed_percent
  let remaining_days := total_days - initial_days
  let remaining_percent := 100 - completed_percent
  let work_rate_per_worker : ℝ := completed_percent / (initial_workers * initial_days)
  let required_workers := remaining_percent / (remaining_days * work_rate_per_worker)
  have h_required_workers : required_workers = 5 := sorry
  exact h_required_workers.le

end carol_project_workers_l778_778999


namespace ferris_wheel_capacity_l778_778023

-- Conditions
def total_seats : ℕ := 18
def people_per_seat : ℕ := 15
def broken_seats : ℕ := 10

-- Proof problem
theorem ferris_wheel_capacity :
  let operational_seats := total_seats - broken_seats in
  operational_seats * people_per_seat = 120 :=
by sorry

end ferris_wheel_capacity_l778_778023


namespace remainder_sum_first_150_div_11300_l778_778084

theorem remainder_sum_first_150_div_11300 :
  let n := 150
  let S := n * (n + 1) / 2
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l778_778084


namespace sum_of_super_cool_triangle_areas_l778_778585

noncomputable def super_cool_triangle_sum_area : ℕ :=
  let leg_pairs := [(7, 42), (8, 24), (9, 18), (10, 15), (12, 12)] in
  let areas := leg_pairs.map (λ p, (p.1 * p.2) / 2) in
  areas.sum

theorem sum_of_super_cool_triangle_areas : super_cool_triangle_sum_area = 471 :=
by
  sorry

end sum_of_super_cool_triangle_areas_l778_778585


namespace probability_area_greater_than_diameter_l778_778148

theorem probability_area_greater_than_diameter:
  (∃ (d : ℝ), (∃ (x y : ℝ), 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 12 ∧ d = x + y) ∧ d > 1.27) →
  (∀ (d : ℝ), d ∈ {2, 3, ..., 18} → (π * (d / 2) ^ 2 > d)) →
  probability (π * (d / 2) ^ 2 > d) = 1 :=
by
  sorry

end probability_area_greater_than_diameter_l778_778148


namespace find_indistinguishable_time_l778_778142

noncomputable def indistinguishable_time : ℝ :=
  let t := 65 + 5/11 in t

theorem find_indistinguishable_time :
  ∃ t: ℝ, (0.5 * t) % 360 = (6 * t) % 360 ∧ t = indistinguishable_time :=
by sorry

end find_indistinguishable_time_l778_778142


namespace points_on_line_l778_778041

theorem points_on_line : 
  let line_equation := λ (x : ℝ), 2 * x
  (line_equation 4 = 8) ∧ (line_equation 6 = 12) ∧ (line_equation 10 = 20) :=
by
  let line_equation := λ (x : ℝ), 2 * x
  show (line_equation 4 = 8) ∧ (line_equation 6 = 12) ∧ (line_equation 10 = 20)
  sorry

end points_on_line_l778_778041


namespace candy_problem_l778_778595

-- Define the given conditions
def numberOfStudents : Nat := 43
def piecesOfCandyPerStudent : Nat := 8

-- Formulate the problem statement
theorem candy_problem : numberOfStudents * piecesOfCandyPerStudent = 344 := by
  sorry

end candy_problem_l778_778595


namespace find_point_C_l778_778587

def point : Type := ℝ × ℝ

-- Define points A and B
def A : point := (1, -1)
def B : point := (11, 3)

-- Define the vector AB
def vector_AB : point := (B.1 - A.1, B.2 - A.2)

-- Scale the vector AB by 2/3
def scaled_vector_AB : point := (2 / 3 * vector_AB.1, 2 / 3 * vector_AB.2)

-- Calculate C by adding the scaled vector to B
def C : point := (B.1 + scaled_vector_AB.1, B.2 + scaled_vector_AB.2)

-- The expected coordinates of point C
def expected_C : point := (53 / 3, 17 / 3)

theorem find_point_C : C = expected_C := by
  rw [C, expected_C, B, scaled_vector_AB, vector_AB]
  simp
  sorry

end find_point_C_l778_778587


namespace integral_solution_l778_778952

-- Define the main functions involved
def u (x : ℝ) := 2 - 3 * x
def dv (x : ℝ) := Real.sin (2 * x)

-- Statement of the indefinite integral to be proved.
theorem integral_solution :
  ∃ C : ℝ, ∫ (x : ℝ) in set.univ, (u x) * (dv x) =
    - (1/2) * (u x) * Real.cos (2 * x) - (3/4) * Real.sin (2 * x) + C :=
begin
  sorry
end

end integral_solution_l778_778952


namespace speed_ratio_l778_778545

variable (vA vB : ℝ)
variable (H1 : 3 * vA = abs (-400 + 3 * vB))
variable (H2 : 10 * vA = abs (-400 + 10 * vB))

theorem speed_ratio (vA vB : ℝ) (H1 : 3 * vA = abs (-400 + 3 * vB)) (H2 : 10 * vA = abs (-400 + 10 * vB)) : 
  vA / vB = 5 / 6 :=
  sorry

end speed_ratio_l778_778545


namespace geometric_sequence_ratio_l778_778801

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (q : ℕ) (h1 : q = 2)
  (h2 : ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 4 / S 2 = 5 :=
by
  sorry

end geometric_sequence_ratio_l778_778801


namespace grid_X_value_l778_778625

theorem grid_X_value :
  ∃ X, (∃ b d1 d2 d3 d4, 
    b = 16 ∧
    d1 = (25 - 20) ∧
    d2 = (16 - 15) / 3 ∧
    d3 = (d1 * 5) / 4 ∧
    d4 = d1 - d3 ∧
    (-12 - d4 * 4) = -30 ∧ 
    X = d4 ∧
    X = 10.5) :=
sorry

end grid_X_value_l778_778625


namespace susan_average_speed_l778_778824

noncomputable def average_speed (distances : List ℝ) (speeds : List ℝ) : ℝ :=
  let total_distance := distances.sum
  let total_time := List.zipWith (λ d s => d / s) distances speeds |>.sum
  total_distance / total_time

theorem susan_average_speed :
  average_speed [40, 20, 30, 10] [15, 60, 50, 70] ≈ 26.74 := by
  sorry

end susan_average_speed_l778_778824


namespace sequence_relation_l778_778000

theorem sequence_relation
  (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sequence_relation_l778_778000


namespace picnic_total_persons_l778_778564

-- Definitions based on given conditions
variables (W M A C : ℕ)
axiom cond1 : M = W + 80
axiom cond2 : A = C + 80
axiom cond3 : M = 120

-- Proof problem: Total persons = 240
theorem picnic_total_persons : W + M + A + C = 240 :=
by
  -- Proof will be filled here
  sorry

end picnic_total_persons_l778_778564


namespace probability_of_integer_fifth_term_is_correct_l778_778984

noncomputable def next_term (a : ℚ) (flip : Bool) : ℚ :=
if flip then 3 * a - 2 else (a - 3) / 2

def sequences : List (List ℚ) :=
let init : List (List ℚ) := [[7]]
List.foldl (λ acc i,
  acc.bind (λ seq,
    [true, false].map (next_term seq.head!))
) init (List.range 4)

def fifth_term_integers : List ℚ := 
sequences.map List.head!.filter (λ x => x.den = 1)

def probability_integer_fifth_term : Rational :=
(fifth_term_integers.length : ℚ) / (sequences.length : ℚ)

theorem probability_of_integer_fifth_term_is_correct :
  probability_integer_fifth_term = -- the correct fraction here
sorry

end probability_of_integer_fifth_term_is_correct_l778_778984


namespace infinite_series_convergence_l778_778670

def closest_int_sqrt (n : ℕ) : ℕ :=
⌊(sqrt n : ℝ)⌉

theorem infinite_series_convergence :
  ∃ L : ℝ, 
  has_sum (λ n : ℕ, (3 ^ closest_int_sqrt n + 3 ^ - closest_int_sqrt n) / 2 ^ n) L := by
sorry

end infinite_series_convergence_l778_778670


namespace conical_tank_water_percentage_l778_778163

theorem conical_tank_water_percentage (h r : ℝ) : 
  (h > 0) → 
  (r > 0) → 
  let V := (1 / 3) * π * r^2 * h in 
  let V_water := (8 / 27) * π * r^2 * h in 
  ((V_water / V) * 100).round_decimals 4 = 88.8889 := 
by
  intros h_pos r_pos
  have V_ne_zero : V ≠ 0 := sorry
  rewrite [h r, V, V_water, everything]
  -- expect (V_water / V) * 100 = 0.8888888 before rounding
  sorry


end conical_tank_water_percentage_l778_778163


namespace standard_equation_of_ellipse_l778_778869

theorem standard_equation_of_ellipse (h₁ : (∀ x y : ℝ, (x^2 / 9 + y^2 / 4 = 1) → 
        let c := sqrt (9 - 4) in 
        ∀ x y : ℝ, (x^2 / 25 + y^2 / 20 = 1)) : 
      true :=
begin
  sorry,
end

end standard_equation_of_ellipse_l778_778869


namespace tree_height_increase_l778_778090

theorem tree_height_increase :
  ∃ h : ℝ, (∀ (n m : ℕ), n = 0 ∧ m = 4 → 4 + 4 * h = 4 + 6 * h) → (4 + 6 * h = (4 + 4 * h) + (1 / 4) * (4 + 4 * h)) → h = 1 :=
begin
  sorry
end

end tree_height_increase_l778_778090


namespace geese_left_park_l778_778607

noncomputable def initial_ducks : ℕ := 25
noncomputable def initial_geese (ducks : ℕ) : ℕ := 2 * ducks - 10
noncomputable def final_ducks (ducks_added : ℕ) (ducks : ℕ) : ℕ := ducks + ducks_added
noncomputable def geese_after_leaving (geese_before : ℕ) (geese_left : ℕ) : ℕ := geese_before - geese_left

theorem geese_left_park
    (ducks : ℕ)
    (ducks_added : ℕ)
    (initial_geese : ℕ := 2 * ducks - 10)
    (final_ducks : ℕ := ducks + ducks_added)
    (geese_left : ℕ)
    (geese_remaining : ℕ := initial_geese - geese_left) :
    geese_remaining = final_ducks + 1 → geese_left = 10 := by
  sorry

end geese_left_park_l778_778607


namespace company_profit_growth_l778_778549

theorem company_profit_growth (x : ℝ) (h : 1.6 * (1 + x / 100)^2 = 2.5) : x = 25 :=
sorry

end company_profit_growth_l778_778549


namespace conditional_probability_l778_778216

-- Definitions for the given problem
def set_example := {1, 2, 3, 4, 5}
def event_A (x y : ℕ) : Prop := x * y % 2 = 0
def event_B (x y : ℕ) : Prop := (x % 2 = 0) ∧ (y % 2 = 0)

-- Definitions to find probability of events
def P (event : ℕ → ℕ → Prop) (set : Set (ℕ × ℕ)) : ℝ :=
  (set.filter (λ (xy : ℕ × ℕ), event xy.fst xy.snd)).toFinset.card / set.toFinset.card.toReal

-- Probability space
def S : Set (ℕ × ℕ) := { (x, y) | x ∈ set_example ∧ y ∈ set_example ∧ x ≠ y }

-- Probability of event A
def P_A : ℝ := P event_A S

-- Probability of event B within A
def P_A_and_B : ℝ := P (λ x y, event_B x y ∧ event_A x y) S

-- Conditional probability P(B|A)
def P_B_given_A : ℝ := P_A_and_B / P_A

theorem conditional_probability :
  P_B_given_A = (1 / 7) :=
by
  sorry

end conditional_probability_l778_778216


namespace circumference_of_circle_inscribed_in_rectangle_l778_778118

theorem circumference_of_circle_inscribed_in_rectangle :
  ∀ (a b : ℝ), a = 9 ∧ b = 12 → (by definition (sqrt (a^2 + b^2) = 15) → 2 * 15 * Real.pi) :=
begin
  -- Given conditions
  assume a b h,
  rcases h with ⟨ha, hb⟩,
  
  -- Statement of the proof omitted
  sorry,
end

end circumference_of_circle_inscribed_in_rectangle_l778_778118


namespace overall_loss_percentage_is_correct_l778_778593

def cost_price (item : String) : ℝ :=
  if item = "radio" then 1500 else if item = "calculator" then 800 else if item = "mobile phone" then 8000 else 0

def discount_percentage (item : String) : ℝ :=
  if item = "radio" then 10 else if item = "calculator" then 5 else if item = "mobile phone" then 12 else 0

def selling_price (item : String) : ℝ :=
  let cp := cost_price item
  let dp := discount_percentage item
  cp - (dp / 100 * cp)

def total_cost_price : ℝ :=
  cost_price "radio" + cost_price "calculator" + cost_price "mobile phone"

def total_selling_price : ℝ :=
  selling_price "radio" + selling_price "calculator" + selling_price "mobile phone"

def total_loss : ℝ := total_cost_price - total_selling_price

def loss_percentage : ℝ := (total_loss / total_cost_price) * 100

theorem overall_loss_percentage_is_correct : loss_percentage ≈ 11.17 := 
by
  sorry

end overall_loss_percentage_is_correct_l778_778593


namespace female_democrats_count_l778_778949

def total_participants : Nat := 870
def democrat_fraction : ℚ := 1 / 3
def female_democrat_fraction : ℚ := 1 / 2
def male_democrat_fraction : ℚ := 1 / 4
def F : ℚ := 290

theorem female_democrats_count (F M : ℚ) (h_F_M : F + M = total_participants)
  (h_democrats : (female_democrat_fraction * F) + (male_democrat_fraction * M) = democrat_fraction * total_participants) :
  (female_democrat_fraction * F) = 145 := 
by
  rw [h_F_M, h_democrats]
  -- Proof steps would go here
  sorry

end female_democrats_count_l778_778949


namespace find_m_given_conditions_l778_778333

open Complex

theorem find_m_given_conditions :
  ∃ (m : ℝ), ∀ (α β : ℂ), (α + β = 2 * sqrt 2) ∧ (|α - β| = 3) ∧ (α * β = m) → m = (17/4) :=
by
  sorry

end find_m_given_conditions_l778_778333


namespace number_of_mixed_groups_l778_778884

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l778_778884


namespace sign_up_ways_unrestricted_sign_up_ways_restricted_l778_778548

theorem sign_up_ways_unrestricted (students events : ℕ) (h_students : students = 4) (h_events : events = 3) :
  (nat.pow events students) = 81 :=
by
  have h1 : students = 4 := h_students
  have h2 : events = 3 := h_events
  calc
    (nat.pow events students)
      = (3^4) : by rw [h1, h2]
  ... = 81 : by norm_num
  sorry

theorem sign_up_ways_restricted (h1_assignments : finset (fin (4)).card = 4) (h2_events : finset (fin (3)).card = 3)
  (h2 : fact (4 + 1 - 1)! / (2! * (4 - 2)! * 3!) = 36) :
  (fact (4) / (fact (2) * fact (4 - 2))) * (fact (3)) = 36 :=
by
  calc
    (fact (4) / (fact (2) * fact (4 - 2))) * (fact (3))
      = (4.choose 2) * (fact (3)) : by
        { rw [nat.choose_eq_factorial_div_factorial] }
  ... = 36 : by
        { rw [nat.factorial, nat.factorial, nat.factorial, nat.factorial] }
  sorry

end sign_up_ways_unrestricted_sign_up_ways_restricted_l778_778548


namespace egg_price_l778_778460

theorem egg_price (num_eggs capital_remaining : ℕ) (total_cost price_per_egg : ℝ)
  (h1 : num_eggs = 30)
  (h2 : capital_remaining = 5)
  (h3 : total_cost = 5)
  (h4 : num_eggs - capital_remaining = 25)
  (h5 : 25 * price_per_egg = total_cost) :
  price_per_egg = 0.20 := sorry

end egg_price_l778_778460


namespace possible_ages_that_sum_to_a_perfect_square_l778_778553

def two_digit_number (a b : ℕ) := 10 * a + b
def reversed_number (a b : ℕ) := 10 * b + a

def sum_of_number_and_its_reversed (a b : ℕ) : ℕ := 
  two_digit_number a b + reversed_number a b

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem possible_ages_that_sum_to_a_perfect_square :
  ∃ (s : Finset ℕ), s.card = 6 ∧ 
  ∀ x ∈ s, ∃ a b : ℕ, a + b = 11 ∧ s = {two_digit_number a b} ∧ is_perfect_square (sum_of_number_and_its_reversed a b) :=
  sorry

end possible_ages_that_sum_to_a_perfect_square_l778_778553


namespace different_people_count_l778_778502

def initial_people := 9
def people_left := 6
def people_joined := 3
def total_different_people (initial_people people_left people_joined : ℕ) : ℕ :=
  initial_people + people_joined

theorem different_people_count :
  total_different_people initial_people people_left people_joined = 12 :=
by
  sorry

end different_people_count_l778_778502


namespace find_x_l778_778107

theorem find_x : ∃ x : ℤ, x + 3 * 10 = 33 → x = 3 := by
  sorry

end find_x_l778_778107


namespace quadratic_distinct_real_roots_l778_778742

-- Definitions
def is_quadratic_eq (a b c x : ℝ) (fx : ℝ) := a * x^2 + b * x + c = fx

-- Theorem statement
theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_quadratic_eq 1 (-2) m x₁ 0 ∧ is_quadratic_eq 1 (-2) m x₂ 0) → m < 1 :=
sorry -- Proof omitted

end quadratic_distinct_real_roots_l778_778742


namespace parabola_intercepts_sum_l778_778844

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end parabola_intercepts_sum_l778_778844


namespace laser_beam_distance_l778_778134

def point := (ℝ × ℝ)

def distance (p1 p2: point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def reflect_y (p: point) : point :=
  (-p.1, p.2)

def reflect_x (p: point) : point :=
  (p.1, -p.2)

def reflect_path_distance (p1 p2: point) : ℝ :=
  let p1' := reflect_y p1 in
  let p2' := reflect_y (reflect_x (reflect_y p2)) in
  distance p1 p2'

theorem laser_beam_distance : 
    let A := (4, 6) in
    let E := (8, 6) in
    reflect_path_distance A E = 4 * real.sqrt 10 := 
by
  sorry

end laser_beam_distance_l778_778134


namespace select_books_from_corner_l778_778026

def num_ways_to_select_books (n₁ n₂ k : ℕ) : ℕ :=
  if h₁ : k > n₁ ∧ k > n₂ then 0
  else if h₂ : k > n₂ then 1
  else if h₃ : k > n₁ then Nat.choose n₂ k
  else Nat.choose n₁ k + 2 * Nat.choose n₁ (k-1) * Nat.choose n₂ 1 + Nat.choose n₁ k * 0 +
    (Nat.choose n₂ 1 * Nat.choose n₂ (k-1)) + Nat.choose n₂ k * 1

theorem select_books_from_corner :
  num_ways_to_select_books 3 6 3 = 42 :=
by
  sorry

end select_books_from_corner_l778_778026


namespace ellipse_and_fixed_point_l778_778288

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778288


namespace max_intersections_of_fourth_degree_polynomials_l778_778083

theorem max_intersections_of_fourth_degree_polynomials (p q : ℝ[X]) 
  (hp_deg : p.degree = 4) (hq_deg : q.degree = 4)
  (hp_lead : p.leadingCoeff = 1) (hq_lead : q.leadingCoeff = 1) :
  p ≠ q → (∀ r : ℝ[X], r = p - q → r.degree ≤ 3) → ∃ n, n ≤ 3 :=
sorry

end max_intersections_of_fourth_degree_polynomials_l778_778083


namespace train_speed_in_kmh_l778_778097

def length_train : ℝ := 700 -- length of the train in meters
def crossing_time : ℝ := 40 -- time to cross the electric pole in seconds
def speed_m_per_s := length_train / crossing_time -- speed in meters per second
def conversion_factor := 3.6 -- conversion factor from meters per second to kilometers per hour
def speed_kmh := speed_m_per_s * conversion_factor -- speed in kilometers per hour

theorem train_speed_in_kmh (length_train : ℝ) (crossing_time : ℝ) (conversion_factor : ℝ) :
  speed_kmh = 63 :=
by
  sorry

end train_speed_in_kmh_l778_778097


namespace conjugate_of_z_squared_l778_778685

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- The statement to be proved
theorem conjugate_of_z_squared : complex.conj (z ^ 2) = -2 * complex.i :=
by
  -- Placeholder for the proof
  sorry

end conjugate_of_z_squared_l778_778685


namespace triangle_similarity_l778_778449

-- Define the geometric terms and the cyclic quadrilateral
variables {A B C D P E F G Q : Type} -- Points
variables [InnerProductSpace ℝ Ω] -- Assume real inner product space for ℝ

-- Define the conditions
def cyclic_quadrilateral (ABCD : Set Ω) := 
  ∃ (P : Ω), ConvexHull ℝ {A, B, C, D}.2Contains P ∧ 
  ∠ BPC = ∠ BAP + ∠ CDP

def feet_of_perpendiculars (P AB AD DC : Ω) (E F G : Ω) :=
  is_orthogonal (AB - A) (P - E) ∧ 
  is_orthogonal (AD - A) (P - F) ∧ 
  is_orthogonal (DC - D) (P - G)

-- Prove the triangles are similar
theorem triangle_similarity 
  (ABCD: Set Ω) (Hcyclic : cyclic_quadrilateral ABCD) 
  (Hfeet : feet_of_perpendiculars P A A D) :
  similar (triangle F E G) (triangle P B C) :=
      sorry

end triangle_similarity_l778_778449


namespace super_cool_triangles_area_sum_l778_778572

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end super_cool_triangles_area_sum_l778_778572


namespace perimeter_ratio_of_similar_triangles_l778_778546

  -- Define the given conditions in Lean 4
  variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] 
    [MetricSpace C] [MetricSpace D] [MetricSpace E]

  -- Assume ∆ ADE ∼ ∆ ABC with a similarity ratio of 1:2
  def triangles_similar_with_ratio (triangle1 triangle2 : Type) (ratio : ℚ) : Prop :=
    ∀ (a b c : triangle1) (d e f : triangle2), 
      ∃ (h : SimilarTriangles a b c d e f), 
      h.ratio = ratio

  -- The goal is to prove the ratio of the perimeters of ∆ ADE and ∆ ABC is 1:2
  theorem perimeter_ratio_of_similar_triangles 
    (similar_triangles : triangles_similar_with_ratio ΔADE ΔABC (1:2)) 
    : (perimeter ΔADE) / (perimeter ΔABC) = (1:2) := 
  sorry
  
end perimeter_ratio_of_similar_triangles_l778_778546


namespace triangle_AOB_area_l778_778396

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ := (1 + t, t - 3)

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  (2 * Real.cos θ) / (Real.sin θ)^2

/-- 
  Prove that the area of the triangle AOB is 12, 
  where A and B are points of intersection between 
  the line l and the curve C.
-/
theorem triangle_AOB_area : 
  let l (t : ℝ) := (1 + t, t - 3)
  let C (θ : ℝ) := (2 * Real.cos θ) / (Real.sin θ)^2
  ∃ (A B: ℝ × ℝ),
    A ∈ set.range l ∧ B ∈ set.range l ∧ A ≠ B ∧
    A ∈ set.range (λ θ, (C θ * Real.cos θ, C θ * Real.sin θ)) ∧
    B ∈ set.range (λ θ, (C θ * Real.cos θ, C θ * Real.sin θ)) ∧
    let base := Real.dist A B
    let height := ∥Real.sqrt (2 : ℝ)∥
    (1 / 2) * base * height = 12 := 
begin
  sorry
end

end triangle_AOB_area_l778_778396


namespace rain_ratio_l778_778198

def monday_rain := 2 + 1 -- inches of rain on Monday
def wednesday_rain := 0 -- inches of rain on Wednesday
def thursday_rain := 1 -- inches of rain on Thursday
def average_rain_per_day := 4 -- daily average rain total
def days_in_week := 5 -- days in a week
def weekly_total_rain := average_rain_per_day * days_in_week

-- Theorem statement
theorem rain_ratio (tuesday_rain : ℝ) (friday_rain : ℝ) 
  (h1 : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h2 : monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = weekly_total_rain) :
  tuesday_rain / monday_rain = 2 := 
sorry

end rain_ratio_l778_778198


namespace unique_rectangle_exists_l778_778234

theorem unique_rectangle_exists (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 4 :=
by
  sorry

end unique_rectangle_exists_l778_778234


namespace compare_abc_l778_778735

-- Definitions and conditions from the problem
def a : ℝ := Real.sqrt 2
def b : ℝ := Real.exp (1 / Real.exp 1)
def c : ℝ := Real.cbrt 6

-- Theorem statement for the problem
theorem compare_abc : a < b ∧ b < c :=
sorry

end compare_abc_l778_778735


namespace triangle_geometry_l778_778407

theorem triangle_geometry (A B C M: Type) 
  [IsTriangle A B C] 
  [IsMidpoint M B C]
  [Angle B 60] 
  [Segment AB 2] 
  [Segment AM (2*Real.sqrt 3)] :
  (Segment AC = 2*Real.sqrt 13) ∧ (CosAngle MAC = 2*Real.sqrt 39 / 13) := 
begin
  sorry
end

end triangle_geometry_l778_778407


namespace minimum_trips_l778_778060

/-- 
Problem: There are 400 tons of cargo stored in a warehouse, each with a weight that is a multiple 
of a quintal and does not exceed 10 tons. It is known that any two pieces of cargo have different 
weights. Prove that the minimum number of trips a 10-ton truck must make to guarantee the 
transportation of all this cargo from the warehouse is 51.
-/

theorem minimum_trips (n : ℕ) (weights : fin n → ℕ)
  (h1 : ∀ i, weights i % 10 = 0)
  (h2 : ∀ i j, i ≠ j → weights i ≠ weights j)
  (h3 : ∑ i, weights i = 4000)
  (h4 : ∀ i, weights i ≤ 100) :
  ∃ t : ℕ, t = 51 ∧ ∑ i in finset.range t, weights i ≤ 10000 :=
sorry

end minimum_trips_l778_778060


namespace quadratic_to_standard_form_div_l778_778639

theorem quadratic_to_standard_form_div (b c : ℤ)
  (h : ∀ x : ℤ, x^2 - 2100 * x - 8400 = (x + b)^2 + c) :
  c / b = 1058 :=
sorry

end quadratic_to_standard_form_div_l778_778639


namespace solve_given_equation_l778_778464

noncomputable def solve_equation : set ℂ :=
  { x : ℂ | (x^3 + 4 * x^2 * complex.sqrt 3 + 12 * x + 8 * complex.sqrt 3) + (x + 2 * complex.sqrt 3) = 0 }

theorem solve_given_equation :
  solve_equation = {-2 * complex.sqrt 3, -2 * complex.sqrt 3 + complex.i, -2 * complex.sqrt 3 - complex.i} := by
  sorry

end solve_given_equation_l778_778464


namespace sum_of_interior_ninth_row_l778_778774

theorem sum_of_interior_ninth_row : 
  Sum of the interior numbers of the fourth row is 6 ∧
  Sum of the interior numbers of the fifth row is 14 →
  Sum of the interior numbers of the ninth row = 254 := 
by 
  -- Assuming the conditions hold, we will prove the conclusion.
  sorry

end sum_of_interior_ninth_row_l778_778774


namespace ellipse_equation_fixed_point_l778_778272

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778272


namespace find_quadratic_polynomial_with_root_l778_778660

noncomputable def quadratic_polynomial (a b c : ℝ) : ℝ := 3 * a^2 - 30 * b + 87

theorem find_quadratic_polynomial_with_root (x : ℂ) (h₁ : x = 5 + 2 * complex.I) 
        (h₂ : x.conj = 5 - 2 * complex.I) : 
        quadratic_polynomial x.re x.im (3) = 3 * (x^2).re - 30 * x.re + 87  :=
by
  -- Proof goes here
  sorry

end find_quadratic_polynomial_with_root_l778_778660


namespace abs_expr_eval_l778_778795

theorem abs_expr_eval (x : ℝ) (h : x = -3000) : abs (abs x + x - 2 * abs x) + x = 3000 :=
by
  rw h
  sorry

end abs_expr_eval_l778_778795


namespace fruit_juice_conversion_needed_l778_778127

theorem fruit_juice_conversion_needed
  (A_milk_parts B_milk_parts A_fruit_juice_parts B_fruit_juice_parts : ℕ)
  (y : ℕ)
  (x : ℕ)
  (convert_liters : ℕ)
  (A_juice_ratio_milk A_juice_ratio_fruit : ℚ)
  (B_juice_ratio_milk B_juice_ratio_fruit : ℚ) :
  (A_milk_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_milk →
  (A_fruit_juice_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_fruit →
  (B_milk_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_milk →
  (B_fruit_juice_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_fruit →
  (A_juice_ratio_milk * x = A_juice_ratio_fruit * x + y) →
  y = 14 →
  x = 98 :=
by sorry

end fruit_juice_conversion_needed_l778_778127


namespace angle_y_supplementary_l778_778764

theorem angle_y_supplementary (m n : ℝ) (h_par : m = n) (h_angle_A : ∠ at A = 40) :
  ∠ y = 140 :=
by
  sorry

end angle_y_supplementary_l778_778764


namespace count_valid_four_digit_numbers_l778_778365

def is_valid_four_digit_number (a b c d : ℕ) : Prop :=
  b = (a + c) / 2 ∧ d = 2 * a ∧ a ≥ 1 ∧ a ≤ 4 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem count_valid_four_digit_numbers : 
  ∃! n : ℕ, n = 20 ∧ (∀ (a b c d : ℕ), is_valid_four_digit_number a b c d → 0 ≤ b ∧ b < 10) :=
begin
  sorry
end

end count_valid_four_digit_numbers_l778_778365


namespace complex_expression_simplification_l778_778800

theorem complex_expression_simplification :
  let x : ℂ := exp (complex.I * 2 * π / 9)
  (2 * x + x^3) * (2 * x^3 + 1) * (2 * x^6 + 1) * (2 * x^2 + x^6) * (2 * x^5 + x^6) * (2 * x^7 + x^3) = 557 := by
  let x : ℂ := exp (complex.I * 2 * π / 9)
  have hx9 : x^9 = 1 := sorry
  sorry

end complex_expression_simplification_l778_778800


namespace minimum_period_tan_2x_l778_778859

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem minimum_period_tan_2x : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi / 2 :=
by
  sorry

end minimum_period_tan_2x_l778_778859


namespace cos_squared_plus_tan_squared_find_length_c_l778_778421

-- Define the conditions and prove Part (1)
theorem cos_squared_plus_tan_squared (B : ℝ) (h_cosB : cos B = 1 / 3) :
  cos^2 (B / 2) + tan^2 ((π - B) / 2) = 8 / 3 :=
by
  sorry

-- Define the conditions and prove Part (2)
theorem find_length_c
  (B : ℝ)
  (b : ℝ)
  (area : ℝ)
  (h_cosB : cos B = 1 / 3)
  (h_b : b = 4)
  (h_area : area = 2 * sqrt 2) :
  c = sqrt 2 ∨ c = 3 * sqrt 2 :=
by
  sorry

end cos_squared_plus_tan_squared_find_length_c_l778_778421


namespace inlet_pipe_rate_l778_778135

theorem inlet_pipe_rate :
  ∀ (L_inlet: ℕ) (L_leak: ℕ) (tank_volume: ℕ) (empty_time_leak_only: ℕ) (empty_time_combined: ℕ)
  (rate_minutes: ℕ),
  empty_time_leak_only = 4 →
  empty_time_combined = 12 →
  tank_volume = 2160 →
  L_leak = tank_volume / empty_time_leak_only →
  L_inlet = rate_minutes * 60 →
  (L_inlet - L_leak = tank_volume / empty_time_combined) →
  rate_minutes = 12 :=
begin
  intros L_inlet L_leak tank_volume empty_time_leak_only empty_time_combined rate_minutes,
  intros h1 h2 h3 h4 h5 h6,
  sorry
end

end inlet_pipe_rate_l778_778135


namespace hyperbola_proof_l778_778205

noncomputable def hyperbola_eq : Type := 
  ∃ (a b : ℝ), (a^2 + b^2 = 4 ∧ (9 / a^2 - 2 / b^2 = 1)) → 
  ∀ (x y : ℝ), (x = 3 ∧ y = sqrt 2) → (x^2 / a^2 - y^2 / b^2 = 1)

theorem hyperbola_proof :
  hyperbola_eq := sorry

end hyperbola_proof_l778_778205


namespace ellipse_solution_geometry_solution_l778_778306

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778306


namespace possible_number_of_people_l778_778557

theorem possible_number_of_people (n : ℕ) : 
  (∃ k : ℕ, n = k * 4 + 3) → 
  n ∈ {46, 47, 48, 50} → 
  n = 47 :=
by
  intros h1 h2
  sorry

end possible_number_of_people_l778_778557


namespace mixed_groups_count_l778_778906

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l778_778906


namespace sum_of_areas_of_super_cool_triangles_l778_778577

def is_super_cool_triangle (a b : ℕ) : Prop :=
  (a * b / 2 = 3 * (a + b))

theorem sum_of_areas_of_super_cool_triangles :
  (∑ p in {p : ℕ × ℕ | is_super_cool_triangle p.1 p.2}, (p.1 * p.2) / 2) = 471 := 
by
  sorry

end sum_of_areas_of_super_cool_triangles_l778_778577


namespace main_theorem_l778_778450

structure Triangle (α : Type _) [Field α] :=
(A B C M N : α × α)
(BM AM CN AN BC : α)
(h_pos_BM : 0 < BM)
(h_pos_AM : 0 < AM)
(h_pos_CN : 0 < CN)
(h_pos_AN : 0 < AN)
(h_pos_BC : 0 < BC)
(h_BM_AM : BM = 3 * AM)
(h_CN_AN : CN = 3 * AN)
(h_BC_val : BC = 12)

noncomputable def proof_mn_parallel_bc (T : Triangle ℝ) : Prop :=
  let M := T.M
  let N := T.N
  let BC := T.BC in
  MN_parallel_to_BC M N BC

theorem main_theorem (T : Triangle ℝ) : proof_mn_parallel_bc T ∧ ∃ (MN : ℝ), MN = 3 :=
by
  sorry

end main_theorem_l778_778450


namespace problem1_problem2_problem3_l778_778345

-- 1. Prove that if f(x) is decreasing on [1,2] for f(x) = x^2 + ax - ln(x), then a ∈ (-∞, -7/2]
theorem problem1 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Icc (1 : ℝ) 2, f x = x^2 + a * x - log x) :
  (∀ x ∈ Icc (1 : ℝ) 2, deriv f x ≤ 0) → a ≤ -(7/2) := by
  sorry

-- 2. Show that there exists a real number a = e^2 such that the minimum value of g(x) = f(x) - x^2 on (0, e] is 3
theorem problem2 (a : ℝ) (f g : ℝ → ℝ)
  (h1 : ∀ x ∈ Icc (0 : ℝ) exp, f x = x^2 + a * x - log x)
  (h2 : ∀ x ∈ Icc (0 : ℝ) exp, g x = f x - x^2) :
  (∃ a : ℝ, a = exp^2 ∧ ∀ x ∈ Icc (0 : ℝ) exp, g x = 3) := by
  sorry

-- 3. Prove that for x ∈ (0, e], e^2x^2 - 5/2x > (x+1)ln(x)
theorem problem3 (x : ℝ) :
  x ∈ Icc (0 : ℝ) exp → exp^2 * x^2 - (5/2) * x > (x + 1) * log x := by
  sorry

end problem1_problem2_problem3_l778_778345


namespace no_perfect_square_with_one_digit_appending_l778_778864

def append_digit (n : Nat) (d : Fin 10) : Nat :=
  n * 10 + d.val

theorem no_perfect_square_with_one_digit_appending :
  ∀ n : Nat, (∃ k : Nat, k * k = n) → 
  (¬ (∃ d1 : Fin 10, ∃ k : Nat, k * k = append_digit n d1.val) ∧
   ¬ (∃ d2 : Fin 10, ∃ d3 : Fin 10, ∃ k : Nat, k * k = d2.val * 10 ^ (Nat.digits 10 n).length + n * 10 + d3.val)) :=
by sorry

end no_perfect_square_with_one_digit_appending_l778_778864


namespace mixed_groups_count_l778_778898

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l778_778898


namespace y_coordinate_of_third_vertex_in_equilateral_triangle_l778_778989

noncomputable def side_length (a b : ℝ) : ℝ := abs (b - a)

noncomputable def altitude (a b : ℝ) : ℝ := (real.sqrt 3 / 2) * (side_length a b)

theorem y_coordinate_of_third_vertex_in_equilateral_triangle :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  x₁ = 3 → y₁ = 6 → x₂ = 11 → y₂ = 6 →
  ∃ (y₃ : ℝ), y₃ = y₁ + altitude x₁ x₂ :=
begin
  intros x₁ y₁ x₂ y₂ hx₁ hy₁ hx₂ hy₂,
  use y₁ + altitude x₁ x₂,
  rw [hx₁, hy₁, hx₂, hy₂],
  sorry
end

end y_coordinate_of_third_vertex_in_equilateral_triangle_l778_778989


namespace sum_series_correct_l778_778667

noncomputable def sum_series : ℚ :=
  ∑ n in Finset.range 2008, 2 / ((n + 1) * (n + 2) * (n + 3))

theorem sum_series_correct : sum_series = 1009522 / 2019045 :=
by sorry

end sum_series_correct_l778_778667


namespace complete_square_solution_l778_778080

theorem complete_square_solution (x : ℝ) :
  x^2 - 2*x - 3 = 0 → (x - 1)^2 = 4 :=
by
  sorry

end complete_square_solution_l778_778080


namespace fill_pool_time_l778_778218

-- Define the conditions
def pool_volume : ℕ := 15000
def hoses1_rate : ℕ := 2
def hoses1_count : ℕ := 2
def hoses2_rate : ℕ := 3
def hoses2_count : ℕ := 2

-- Calculate the total delivery rate
def total_delivery_rate : ℕ :=
  (hoses1_rate * hoses1_count) + (hoses2_rate * hoses2_count)

-- Calculate the time to fill the pool in minutes
def time_to_fill_in_minutes : ℕ :=
  pool_volume / total_delivery_rate

-- Calculate the time to fill the pool in hours
def time_to_fill_in_hours : ℕ :=
  time_to_fill_in_minutes / 60

-- The theorem to prove
theorem fill_pool_time : time_to_fill_in_hours = 25 := by
  sorry

end fill_pool_time_l778_778218


namespace lowest_point_graph_of_y_l778_778474

theorem lowest_point_graph_of_y (x : ℝ) (h : x > -1) :
  (x, (x^2 + 2 * x + 2) / (x + 1)) = (0, 2) ∧ ∀ y > -1, ( (y^2 + 2 * y + 2) / (y + 1) >= 2) := 
sorry

end lowest_point_graph_of_y_l778_778474


namespace optimal_second_test_point_l778_778393

theorem optimal_second_test_point : 
  ∀ (lb ub : ℝ), lb = 500 → ub = 1500 → 
  let x1 := lb + 0.618 * (ub - lb) in 
  let x2 := ub + lb - x1 in 
  x2 = 882 :=
by 
  intros lb ub h_lb h_ub
  let x1 := lb + 0.618 * (ub - lb)
  let x2 := ub + lb - x1
  have : lb = 500 := h_lb
  have : ub = 1500 := h_ub
  have : x1 = 500 + 0.618 * (1500 - 500) := by 
    simp [h_lb, h_ub, x1]
  sorry

end optimal_second_test_point_l778_778393


namespace chloe_phone_numbers_count_l778_778158

def valid_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 8

theorem chloe_phone_numbers_count :
  ∃ count : ℕ, count = 28 ∧
    (count = (Nat.choose 8 6)) :=
by
  have h_comb : Nat.choose 8 6 = (8 * 7) / 2 := by
    change Nat.choose 8 6 = 28
    sorry
  use 28
  split
  . rfl
  . exact h_comb

end chloe_phone_numbers_count_l778_778158


namespace limit_an_over_n_eq_one_l778_778786

/-- Define the sequence a_n of positive integers such that:
- a_1 = 1
- for any arbitrary prime p, the set {a_1, a_2, ..., a_p} is a complete remainder system modulo p
--/
def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else sorry

/-- The set {a_1, a_2, ..., a_p} is a complete remainder system modulo p -/
axiom complete_remainder_system (p : ℕ) (prime_p : Nat.Prime p) :
  ∀ i ∈ Finset.range p, ∃ j ∈ Finset.range p, a_seq(j + 1) % p = i

theorem limit_an_over_n_eq_one :
  ∃ a_seq : ℕ → ℕ,
    (a_seq 1 = 1) ∧
    (∀ p: ℕ, Nat.Prime p → ∀ i ∈ Finset.range p, ∃ j ∈ Finset.range p, a_seq(j + 1) % p = i) →
    tendsto (λ n, (a_seq n : ℝ) / (n : ℝ)) at_top (𝓝 1) := sorry

end limit_an_over_n_eq_one_l778_778786


namespace ellipse_and_fixed_point_l778_778293

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778293


namespace ellipse_equation_proof_HN_fixed_point_l778_778316

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778316


namespace ellipse_equation_fixed_point_l778_778273

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778273


namespace calculate_expression_l778_778611

theorem calculate_expression :
  ((2000000000000 - 1234567890123) * 3 = 2296296329631) :=
by 
  sorry

end calculate_expression_l778_778611


namespace Tanya_time_to_complete_work_l778_778540

variable (Sakshi_days : ℕ)

def Sakshi_work_rate := 1 / Sakshi_days

-- Assuming Tanya is 25% more efficient than Sakshi
def efficiency_increase := 1.25

-- Tanya's work rate
def Tanya_work_rate := Sakshi_work_rate Sakshi_days * efficiency_increase

-- The time Tanya takes to complete the work
def Tanya_days := 1 / Tanya_work_rate

theorem Tanya_time_to_complete_work :
  Sakshi_days = 20 → efficiency_increase = 1.25 → Tanya_days Sakshi_days = 16 :=
by
  intros hSakshi_days hefficiency
  -- Skipping the proof steps
  sorry

end Tanya_time_to_complete_work_l778_778540


namespace evaluate_expression_l778_778615

theorem evaluate_expression :
  500 * 997 * 0.0997 * 10^2 = 5 * (997:ℝ)^2 :=
by
  sorry

end evaluate_expression_l778_778615


namespace parabola_directrix_l778_778721

variables {a b c : ℝ}
variables (F O A : ℝ × ℝ)
variables (hyperbola_line : ℝ → ℝ)
variables (triangle_area : ℝ)
variable (directrix : ℝ → Prop)

theorem parabola_directrix :
  (∀ a b : ℝ, 0 < a → 0 < b →
  (∀ c : ℝ, F = (c, 0) →
  (∀ x : ℝ, x = a^2 / c → 
  (∀ y : ℝ, y = b / a * x → 
  A = (x, y) →
  let triangle_area := 0.5 * c * (a * b / c) in
  triangle_area = a^2 / 2 →
  a = b →
  ∀ d : ℝ, y^2 = (4 * a / b) * x →
  directrix y →  x = -1)))))))
: sorry

end parabola_directrix_l778_778721


namespace no_graph_with_given_degrees_l778_778232

theorem no_graph_with_given_degrees (n : ℕ) (h : n ≥ 10) :
  ¬(∃ (G : SimpleGraph (Fin n)),
    ∀ (v : Fin n),
      (if v = ⟨n-10, sorry⟩ then 4
       else if v = ⟨n-9, sorry⟩ then 5
       else if v = ⟨n-8, sorry⟩ then 6
       else if v = ⟨n-7, sorry⟩ then 7
       else if v ∈ (Finset.univ.image (λ i, if i < 3 then Fin.of_nat n else Fin.of_nat (n-1))) then n - i
       else if v ∈ (Finset.univ.image (λ i, if i < 6 then Fin.of_nat (n-1) else if i < 3 then Fin.of_nat (n-2) else Fin.of_nat)):=

    sorry

end no_graph_with_given_degrees_l778_778232


namespace multiplication_schemes_correct_l778_778402

theorem multiplication_schemes_correct :
  ∃ A B C D E F G H I K L M N P : ℕ,
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ F = 0 ∧ G = 8 ∧ H = 3 ∧ I = 3 ∧ K = 8 ∧ L = 8 ∧ M = 0 ∧ N = 7 ∧ P = 7 ∧
    (A * 10 + B) * (C * 10 + D) * (A * 10 + B) = E * 100 + F * 10 + G ∧
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C ∧
    E * 100 + F * 10 + G / (H * 1000 + I * 100 + G * 10 + G) = (E * 100 + F * 10 + G) / (H * 1000 + I * 100 + G * 10 + G) ∧
    (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) = (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) :=
sorry

end multiplication_schemes_correct_l778_778402


namespace statement_1_statement_2_statement_3_all_statements_correct_l778_778681

-- Define the function f and the axioms/conditions given in the problem
def f : ℕ → ℕ → ℕ := sorry

-- Conditions
axiom f_initial : f 1 1 = 1
axiom f_nat : ∀ m n : ℕ, m > 0 → n > 0 → f m n > 0
axiom f_condition_1 : ∀ m n : ℕ, m > 0 → n > 0 → f m (n + 1) = f m n + 2
axiom f_condition_2 : ∀ m : ℕ, m > 0 → f (m + 1) 1 = 2 * f m 1

-- Statements to be proved
theorem statement_1 : f 1 5 = 9 := sorry
theorem statement_2 : f 5 1 = 16 := sorry
theorem statement_3 : f 5 6 = 26 := sorry

theorem all_statements_correct : (f 1 5 = 9) ∧ (f 5 1 = 16) ∧ (f 5 6 = 26) := by
  exact ⟨statement_1, statement_2, statement_3⟩

end statement_1_statement_2_statement_3_all_statements_correct_l778_778681


namespace number_of_integer_solutions_l778_778865

theorem number_of_integer_solutions : 
  (∃ (sols : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ sols ↔ (1 : ℚ)/x + (1 : ℚ)/y = 1/7) ∧ sols.length = 5) := 
sorry

end number_of_integer_solutions_l778_778865


namespace find_λ_l778_778787

noncomputable def λ_eq_zero_has_only_real_roots (n : ℕ) (λ : ℝ) (α β : Fin n → ℝ) 
  (h1 : 4 ≤ n) 
  (h2 : (∑ j, α j ^ 2) < 1)
  (h3 : (∑ j, β j ^ 2) < 1) : Prop :=
let A := Real.sqrt (1 - ∑ j, α j ^ 2);
    B := Real.sqrt (1 - ∑ j, β j ^ 2);
    W := (1 - ∑ j, α j * β j) ^ 2 / 2 in
λ = 0 →
∀ x : ℝ, (Polynomial.map Polynomial.C (Polynomial.X ^ n + λ • (Polynomial.sum (Fin.range n).tail (λ k, Polynomial.X ^ k) + W * Polynomial.X ^ 2 + A * B * Polynomial.X + 1))).roots = 0

theorem find_λ (n : ℕ) (α β : Fin n → ℝ) 
  (h1 : 4 ≤ n)
  (h2 : (∑ j, α j ^ 2) < 1)
  (h3 : (∑ j, β j ^ 2) < 1) :
  λ_eq_zero_has_only_real_roots n 0 α β :=
sorry

end find_λ_l778_778787


namespace ellipse_and_fixed_point_l778_778290

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778290


namespace ellipse_and_fixed_point_l778_778297

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778297


namespace problem_l778_778330

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778330


namespace max_b_no_lattice_points_line_l778_778969

theorem max_b_no_lattice_points_line (b : ℝ) (h : ∀ (m : ℝ), 0 < m ∧ m < b → ∀ (x : ℤ), 0 < (x : ℝ) ∧ (x : ℝ) ≤ 150 → ¬∃ (y : ℤ), y = m * x + 5) :
  b ≤ 1 / 151 :=
by sorry

end max_b_no_lattice_points_line_l778_778969


namespace interest_rate_third_year_l778_778159

theorem interest_rate_third_year :
  ∃ (r : ℝ), 
    let S := 90000 - 25000,
        S1 := S * 1.03,
        S2 := S1 * 1.04,
        S3 := S2 * (1 + r)
    in S3 = 90000 ∧ r ≈ 0.2925 :=
begin
  sorry
end

end interest_rate_third_year_l778_778159


namespace find_f_6_l778_778712

def f : ℝ → ℝ
| x => if x < 0 then x^3 - 1 else
       if -1 ≤ x ∧ x ≤ 1 then 
         if x < 0 then -(x^3 - 1) else x^3 - 1
       else 
         sorry  -- Placeholder for x > 1/2 case

lemma function_periodic_when_x_gt_half (x : ℝ) (hx : x > 1/2) : f (x + 1) = f x :=
by 
  sorry

theorem find_f_6 : f 6 = 2 :=
by
  have h1 : f (6 + 1) = f 6 := by apply function_periodic_when_x_gt_half 6 sorry
  have h2 : f 6 = f 1 := by sorry
  have h3 : f 1 = -f (-1) := by sorry
  have h4 : f (-1) = (-1)^3 - 1 := by sorry
  have h5 : f 1 = -(-2) := by simp [h4]
  exact h2

end find_f_6_l778_778712


namespace ratio_of_trout_to_catfish_l778_778937

theorem ratio_of_trout_to_catfish 
  (will_catfish : ℕ := 16)
  (will_eels : ℕ := 10)
  (total_fish : ℕ := 50)
  (henry_final_fish : ℕ := total_fish - (will_catfish + will_eels)) :
  let x := henry_final_fish / 8 in
  x = 3 :=
by
  sorry

end ratio_of_trout_to_catfish_l778_778937


namespace problem_1_problem_2_problem_3_l778_778228

-- Definitions used in Lean statements based on conditions and problems

def f (t x : ℝ) := x^3 + (3*(t-1)/2)*x^2 - 3*t*x

-- Statement for the first problem
theorem problem_1 (t : ℝ) (h : t < 0) (hne : ∀ x ∈ set.Ioo 0 2, x ∈ set.Ioo 0 2 → deriv (λ x, f t x) x ≠ 0) : t = -1 :=
sorry

-- Statement for the second problem
theorem problem_2 (t : ℝ) (h : t < 0) (hx0 : ∃ x0 ∈ set.Ioo 0 2, ∀ y ∈ set.Ico 0 2, f t x0 ≥ f t y) : t ∈ set.Iic (-5/3) :=
sorry

-- Definitions for the third problem
def g (t x : ℝ) := exp x - x^2 - (3*(t-1)/2)*x + 3*t

-- Statement for the third problem
theorem problem_3 (t : ℝ) (h : t < 0) (h_le : ∀ x ≥ 0, f t x ≤ x * exp x) (h0 : ∀ x ≥ 0, g t x ≥ 0) : 0 > t ∧ t ≥ -1/3 :=
sorry

end problem_1_problem_2_problem_3_l778_778228


namespace sum_of_super_cool_areas_l778_778581

def is_super_cool (a b : ℕ) : Prop :=
  (a - 9) * (b - 9) = 81

theorem sum_of_super_cool_areas : 
  let areas := [(90 * 10) / 2, (36 * 12) / 2, (18 * 18) / 2].erase_dup
  areas.sum = 828 :=
by
  sorry

end sum_of_super_cool_areas_l778_778581


namespace expected_value_men_shorter_than_partners_theorem_l778_778169

noncomputable def expected_value_men_shorter_than_partners (n : ℕ) : ℝ :=
∑ i in finset.range (n - 1), i

theorem expected_value_men_shorter_than_partners_theorem (n : ℕ) (h : 0 < n) :
  expected_value_men_shorter_than_partners n / n = (n - 1) / 2 := by
  sorry

end expected_value_men_shorter_than_partners_theorem_l778_778169


namespace number_of_repeating_decimals_l778_778669

def isRepeatingDecimal (n : ℕ) : Prop :=
  ∀ m, n = m * 2 + 1 ∧ gcd n (n + 1) = 1 → ¬ (∀ k, n + 1 = 2 ^ k ∨ n + 1 = 5 ^ k ∨ n + 1 = (2 ^ k) * (5 ^ k))

theorem number_of_repeating_decimals : 
  (number_of_integers (λ n, 1 ≤ n ∧ n ≤ 150 ∧ isRepeatingDecimal n) = 135) :=
  sorry

end number_of_repeating_decimals_l778_778669


namespace clock_chime_time_l778_778980

/-- The proven time it takes for a wall clock to strike 12 times at 12 o'clock -/
theorem clock_chime_time :
  (∃ (interval_time : ℝ), (interval_time = 3) ∧ (∃ (time_12_times : ℝ), (time_12_times = interval_time * (12 - 1)) ∧ (time_12_times = 33))) :=
by
  sorry

end clock_chime_time_l778_778980


namespace smallest_positive_period_f_monotonically_decreasing_interval_max_min_values_on_interval_l778_778348

noncomputable def f (x : ℝ) : ℝ := (sin x + cos x)^2 + 2 * (cos x)^2

theorem smallest_positive_period_f : 
  ∃ T > 0, ∀ x, f (x + T) = f x := 
sorry

theorem monotonically_decreasing_interval : 
  ∀ k : ℤ, is_monotonic_decreasing (λ x, f x) (set.Icc (↑k * π + π / 8) (↑k * π + 5 * π / 8)) := 
sorry

theorem max_min_values_on_interval : 
  ∃ max min, ∃ a b ∈ set.Icc (0:ℝ) (π / 2), f a = max ∧ f b = min ∧ 
  max = 2 + sqrt 2 ∧ min = 1 := 
sorry

end smallest_positive_period_f_monotonically_decreasing_interval_max_min_values_on_interval_l778_778348


namespace rectangle_area_l778_778568

theorem rectangle_area (x : ℝ) (w : ℝ) (h : ℝ) (H1 : x^2 = w^2 + h^2) (H2 : h = 3 * w) : 
  (w * h = (3 * x^2) / 10) :=
by sorry

end rectangle_area_l778_778568


namespace cookies_from_dough_l778_778941

theorem cookies_from_dough :
  ∀ (length width : ℕ), length = 24 → width = 18 →
  ∃ (side : ℕ), side = Nat.gcd length width ∧ (length / side) * (width / side) = 12 :=
by
  intros length width h_length h_width
  simp only [h_length, h_width]
  use Nat.gcd length width
  simp only [Nat.gcd_rec]
  sorry

end cookies_from_dough_l778_778941


namespace part1_part2_l778_778688

noncomputable def f : ℝ → ℝ := λ x, x^2 - 2 * x + 3

theorem part1 (x : ℝ) :
  (f 0 = 3) ∧ (∀ x, f (x + 1) - f x = 2 * x - 1) ∧ (f x = x^2 - 2 * x + 3) :=
sorry

def g (t : ℝ) : ℝ := 
if t > -2 then 
  if t ≤ 4 then 11
  else t^2 - 2 * t + 3
else 
  0  -- dummy value, g(t) is only defined for t > -2

theorem part2 (t : ℝ) (ht : t > -2) :
  g t = 
  if -2 < t ∧ t ≤ 4 then 
    11 
  else 
    t^2 - 2 * t + 3 :=
sorry

end part1_part2_l778_778688


namespace hn_passes_fixed_point_l778_778269

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778269


namespace arithmetic_geometric_progression_product_l778_778338

theorem arithmetic_geometric_progression_product : 
  ∃ (d q : ℝ), (4 = 1 + 3 * d) ∧ (4 = q^3) ∧ ((1 + 2 * d) * q^2 = 6) :=
by
  -- declare the existence of a common difference "d" and a common ratio "q"
  use 1, (3)^(1/3)
  -- the conditions for d and q from the sequences
  split
  -- condition for the arithmetic sequence
  {
    exact (by norm_num),
  },
  split
  -- condition for the geometric sequence
  {
    norm_cast,
    sorry, -- actual computation of the cube root in Lean
  }
  -- verifying the end condition (1 + 2d) * (q^2) = 6
  {
    field_simp,
    sorry,
  }

end arithmetic_geometric_progression_product_l778_778338


namespace minimum_value_f_l778_778862

def f (x : ℝ) : ℝ := Real.exp x - x

theorem minimum_value_f : ∃ x₀ : ℝ, (∀ x : ℝ, f(x₀) ≤ f(x)) ∧ f(x₀) = 1 :=
by
  sorry

end minimum_value_f_l778_778862


namespace triangle_angle_C_and_area_l778_778752

theorem triangle_angle_C_and_area (A B C : ℝ) (a b c : ℝ) 
  (h1 : 2 * c * Real.cos B = 2 * a - b)
  (h2 : c = Real.sqrt 3)
  (h3 : b - a = 1) :
  (C = Real.pi / 3) ∧
  (1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by
  sorry

end triangle_angle_C_and_area_l778_778752


namespace car_x_travel_distance_l778_778099

theorem car_x_travel_distance :
  ∀ (t : ℝ), x_speed = 35 → y_speed = 38 → t = 15.2 → 35 * t = 532 := 
by
  intro t
  intros
  rw [h1, h2, h3]
  sorry

end car_x_travel_distance_l778_778099


namespace problem_l778_778326

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778326


namespace speed_of_first_cyclist_l778_778927

-- Definitions of the conditions
def start_at_same_point : Prop := true

def opposite_directions : Prop := true

def speed_second_cyclist (v2 : ℕ) : Prop := v2 = 8

def circumference_of_circle (C : ℕ) : Prop := C = 300

def meet_after_time (t : ℕ) : Prop := t = 20

-- The final theorem we need to prove
theorem speed_of_first_cyclist (v1 v2 : ℕ) (C t: ℕ) 
    (h1 : start_at_same_point)
    (h2 : opposite_directions)
    (h3 : speed_second_cyclist v2)
    (h4 : circumference_of_circle C)
    (h5 : meet_after_time t)
    : v1 = 7 :=
begin
  sorry
end

end speed_of_first_cyclist_l778_778927


namespace chess_club_mixed_groups_l778_778888

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l778_778888


namespace divisor_is_3_l778_778445

theorem divisor_is_3 (divisor quotient remainder : ℕ) (h_dividend : 22 = (divisor * quotient) + remainder) 
  (h_quotient : quotient = 7) (h_remainder : remainder = 1) : divisor = 3 :=
by
  sorry

end divisor_is_3_l778_778445


namespace major_premise_incorrect_l778_778724

-- Definitions based on the given conditions
def is_parallel_to_plane (l : Line) (p : Plane) : Prop :=
  ∀ (P : Point), P ∈ l → P ∉ p

def is_in_plane (l : Line) (p : Plane) : Prop :=
  ∀ (P : Point), P ∈ l → P ∈ p

-- The statement to prove
theorem major_premise_incorrect 
  (α : Plane) (a b : Line)
  (h1 : is_parallel_to_plane b α)
  (h2 : is_in_plane a α) :
  ¬ (∀ l ∈ α, is_parallel_to_plane b l) :=
sorry

end major_premise_incorrect_l778_778724


namespace geometric_seq_no_arithmetic_geometric_seq_l778_778723

noncomputable def real_sequence (n : ℕ) : ℝ :=
if n = 1 then 3/5 else 3 * real_sequence (n - 1) / (2 * real_sequence (n - 1) + 1)

def geometric_term (n : ℕ) : ℝ :=
1 / real_sequence n - 1

theorem geometric_seq :
  ∃ r : ℝ, ∀ n : ℕ, geometric_term (n + 1) = r * geometric_term n := sorry

theorem no_arithmetic_geometric_seq :
  ¬∃ (m s t : ℕ), m < s ∧ s < t ∧ 2 * s = m + t ∧ 
  (a_m : ℝ) - 1 * (a_t : ℝ) - 1 = (a_s : ℝ) - 1 * (a_s : ℝ) - 1 := sorry

end geometric_seq_no_arithmetic_geometric_seq_l778_778723


namespace percent_change_in_revenue_l778_778876

-- Definitions based on given conditions
def initial_tax_rate (T : ℝ) : ℝ := T
def initial_consumption : ℝ := 150
def tax_diminishment : ℝ := 0.20
def consumption_increase : ℝ := 0.20
def high_consumption_threshold : ℝ := 200
def additional_tax_increase : ℝ := 0.05
def additional_tax_decrease : ℝ := 0.02

-- The Lean theorem statement to prove the percent change in revenue
theorem percent_change_in_revenue (T : ℝ) : 
  initial_tax_rate T > 0 ∧ initial_consumption > 0 → 
  let new_tax_rate := (1 - 0.20 - 0.02) * T in
  let new_consumption := initial_consumption * (1 + 0.20) in
  new_consumption < high_consumption_threshold →
  ((new_consumption * new_tax_rate) - (initial_consumption * initial_tax_rate T)) / (initial_consumption * initial_tax_rate T) * 100 = -6.4 :=
by {
  sorry
}

end percent_change_in_revenue_l778_778876


namespace minimum_value_of_f_l778_778478

open Real

-- Define the function
def f (x : ℝ) : ℝ := (x + 4) / sqrt x

-- Conditions
lemma domain_of_f : ∀ x, 0 < x → (f x = (x + 4) / sqrt x) :=
by
  intro x hx
  unfold f
  apply congr_fun
  apply congr_arg
  simp [hx]

theorem minimum_value_of_f : (0 < x → f x ≥ 4) ∧ (f 4 = 4) :=
by
  split
  · intro hx
    have : x > 0 := hx
    apply le_of_am_gm
    sorry -- Apply AM-GM inequality and prove f x ≥ 4
  · -- Prove f 4 = 4
    unfold f
    norm_num
    sorry

end minimum_value_of_f_l778_778478


namespace sum_of_areas_of_super_cool_triangles_l778_778576

def is_super_cool_triangle (a b : ℕ) : Prop :=
  (a * b / 2 = 3 * (a + b))

theorem sum_of_areas_of_super_cool_triangles :
  (∑ p in {p : ℕ × ℕ | is_super_cool_triangle p.1 p.2}, (p.1 * p.2) / 2) = 471 := 
by
  sorry

end sum_of_areas_of_super_cool_triangles_l778_778576


namespace ellipse_equation_proof_HN_fixed_point_l778_778318

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778318


namespace number_of_mixed_groups_l778_778879

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l778_778879


namespace triangles_are_isosceles_l778_778877

-- Define point and triangle
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define congruence of triangles
def congruent (T1 T2 : Triangle) : Prop :=
  let Triangle.mk A1 B1 C1 := T1
  let Triangle.mk A2 B2 C2 := T2
  (A1 = A2 ∧ B1 = B2 ∧ C1 = C2) ∨ 
  (A1 = B2 ∧ B1 = C2 ∧ C1 = A2) ∨ 
  (A1 = C2 ∧ B1 = A2 ∧ C1 = B2)

-- Define isosceles triangle
def isosceles (T : Triangle) : Prop :=
  let Triangle.mk A B C := T in
  (A = C ∨ B = C ∨ A = B)

-- Define the points A, B, C, and C1
variables (A B C C1 : Point)

-- Define the triangles ACC1 and BCC1
def T_ACC1 := Triangle.mk A C C1
def T_BCC1 := Triangle.mk B C C1

-- Define the conditions
axiom h1 : congruent T_ACC1 T_BCC1
axiom h2 : (A.y - C1.y) * (B.x - C1.x) < 0 -- A and B lie on opposite sides of line CC1

-- The theorem to be proved
theorem triangles_are_isosceles :
  isosceles (Triangle.mk A B C) ∧ isosceles (Triangle.mk A B C1) :=
by
  sorry

end triangles_are_isosceles_l778_778877


namespace monotonic_intervals_max_value_at_extreme_on_interval_l778_778349

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 / 3) * a * x^3 - x^2

theorem monotonic_intervals (a : ℝ) :
  (a = 1) →
  (∀ x ∈ Set.Iio(0), deriv (f a) x > 0) ∧ 
  (∀ x ∈ Set.Ioi(1), deriv (f a) x > 0) ∧
  (∀ x ∈ Set.Ioo (0, 1), deriv (f a) x < 0) :=
  by sorry

theorem max_value_at_extreme_on_interval (a : ℝ) :
  (deriv (f a) 1 = 0) →
  (f 1 1 = (2 / 3) * (3) - (1)^2) →
  a = 1 →
  ∀ x ∈ Set.Icc (2 : ℝ) (3 : ℝ),
  f 1 x ≤ f 1 3 :=
  by sorry


end monotonic_intervals_max_value_at_extreme_on_interval_l778_778349


namespace farmer_trees_l778_778966

theorem farmer_trees (x n m : ℕ) 
  (h1 : x + 20 = n^2) 
  (h2 : x - 39 = m^2) : 
  x = 880 := 
by sorry

end farmer_trees_l778_778966


namespace tan_2x_period_l778_778858

-- Define the tangent function and its standard period
def tan_period : ℝ := Real.pi

-- Define the function y = tan 2x
def f (x : ℝ) := Real.tan (2 * x)

-- State the property to be proved: The period of f is π/2
theorem tan_2x_period : ∀ x: ℝ, f(x) = f(x + π/2) := 
sorry

end tan_2x_period_l778_778858


namespace root_polynomial_h_l778_778732

theorem root_polynomial_h (h : ℤ) : (2^3 + h * 2 + 10 = 0) → h = -9 :=
by
  sorry

end root_polynomial_h_l778_778732


namespace metallic_sheet_width_l778_778560

theorem metallic_sheet_width 
  (length_of_cut_square : ℝ) (original_length_of_sheet : ℝ) (volume_of_box : ℝ) (w : ℝ)
  (h1 : length_of_cut_square = 5) 
  (h2 : original_length_of_sheet = 48) 
  (h3 : volume_of_box = 4940) : 
  (38 * (w - 10) * 5 = 4940) → w = 36 :=
by
  intros
  sorry

end metallic_sheet_width_l778_778560


namespace quadrilateral_area_l778_778626

-- Definitions based on the given conditions
structure Polygon12 :=
  (A B C D E F G H I J K L : Type) 
  (side_len : ∀ {x y : Type}, (x = (y, 5)) ∧ ((x, 5) = y))

structure IntersectionPoint :=
  (AF CD M : Type)

-- Theorem for the area of quadrilateral ABCM
theorem quadrilateral_area 
  (polygon : Polygon12) 
  (intersect : IntersectionPoint)
  (AF_len : (intersect.AF = polygon.F, 5)) 
  (CD_len : (intersect.CD = polygon.D, 5)) 
  (consecutive_right_angle : ∀ {x y z : Type}, (x = (y, z)) → ((y, z) = (x, z)))
  (meets_at_M : ∃ M : intersect.M, intersect.AF ∩ intersect.CD = M) :
  (area_quadrilateral (polygon.A, polygon.B, polygon.C, intersect.M) = 18.75) :=
begin
  -- Proof omitted
  sorry
end

end quadrilateral_area_l778_778626


namespace problem1_problem2_l778_778997

theorem problem1 :
  6 * Real.sqrt (1 / 9) - Real.cbrt 27 + (Real.sqrt 2)^2 = 1 :=
sorry

theorem problem2 :
  -1 ^ 2022 + Real.sqrt ((-2)^2) + abs (2 - Real.sqrt 3) = 3 - Real.sqrt 3 :=
sorry

end problem1_problem2_l778_778997


namespace ellipse_and_fixed_point_l778_778291

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778291


namespace complex_solutions_l778_778426

def solve_complex_equation (a : ℝ) (ha : a ≥ 0) : set ℂ := 
  {z | (z^2 + 2 * |z| = a)} 

theorem complex_solutions (a : ℝ) (ha : a ≥ 0) :
  solve_complex_equation a ha = 
  {z | z = -1 + real.sqrt (1 + a) ∨ z = -( -1 + real.sqrt (1 + a)) ∨ 
    (0 ≤ a ∧ a ≤ 1 ∧ (z = 1 + real.sqrt (1 - a) ∨ z = -(1 + real.sqrt (1 - a)) ∨ 
                     z = (1 - real.sqrt (1 - a)) * complex.I ∨ 
                     z = -(1 - real.sqrt (1 - a)) * complex.I))} := sorry

end complex_solutions_l778_778426


namespace probability_at_least_one_boy_and_one_girl_l778_778044

open Finset

theorem probability_at_least_one_boy_and_one_girl :
  let total_members := 30
  let boys := 12
  let girls := 18
  let committee_size := 5
  let total_committees := choose total_members committee_size
  let all_boy_committees := choose boys committee_size
  let all_girl_committees := choose girls committee_size
  let favorable_committees := total_committees - (all_boy_committees + all_girl_committees)
  (favorable_committees / total_committees : ℚ) = 571 / 611 := 
by {
  let total_committees := choose total_members committee_size,
  let all_boy_committees := choose boys committee_size,
  let all_girl_committees := choose girls committee_size,
  let favorable_committees := total_committees - (all_boy_committees + all_girl_committees),
  exact eq.trans (div_eq_iff (ne_of_gt (nat.cast_pos.2 _)).symm) 
                  (eq.trans _ (eq.refl 571 / 611))
}

end probability_at_least_one_boy_and_one_girl_l778_778044


namespace problem_l778_778331

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778331


namespace find_values_of_a_l778_778878

-- Define the logarithmic function and condition
def log10_eq (a : ℝ) : Prop := log 10 (a^2 - 15 * a) = 2

-- Define the solutions we expect
def a1 := 20
def a2 := -5

theorem find_values_of_a : log10_eq a1 ∧ log10_eq a2 :=
by
  sorry

end find_values_of_a_l778_778878


namespace triangle_BC_l778_778406

theorem triangle_BC {A B C : Point} {k : ℝ} 
  (hAB : dist A B = 1) 
  (hAC : dist A C = 3) 
  (hMedian : ∀ M : Point, midpoint B C M -> dist A M = k * dist M C / 2) : 
  dist B C = sqrt (20 / (4 * k^2 - 1)) :=
by
  sorry

end triangle_BC_l778_778406


namespace rational_is_integer_or_fraction_l778_778527

-- Define the conditions
def is_rational (r : ℚ) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ r = p / q

def is_integer (n : ℤ) : Prop := true

def is_fraction (f : ℚ) : Prop := 
  ∃ p q : ℤ, q ≠ 0 ∧ f = p / q ∧ ¬(∃ k : ℤ, f = k)

-- The theorem to prove
theorem rational_is_integer_or_fraction (r : ℚ) : 
  is_rational r → (is_integer r ∨ is_fraction r) :=
by
  intros
  sorry

end rational_is_integer_or_fraction_l778_778527


namespace sequence_expression_l778_778689

theorem sequence_expression {a : ℕ → ℝ} (h1 : ∀ n, a (n + 1) ^ 2 = a n ^ 2 + 4)
  (h2 : a 1 = 1) (h3 : ∀ n, a n > 0) : ∀ n, a n = Real.sqrt (4 * n - 3) := by
  sorry

end sequence_expression_l778_778689


namespace area_ratio_l778_778039

variables (l w r : ℝ)

-- Define the conditions
def perimeter_eq_circumference : Prop := 2 * l + 2 * w = 2 * π * r
def length_eq_twice_width : Prop := l = 2 * w

-- Define the theorem to prove the ratio of the areas
theorem area_ratio (h1 : perimeter_eq_circumference l w r) (h2 : length_eq_twice_width l w) :
  (l * w) / (π * r^2) = 2 * π / 9 :=
sorry

end area_ratio_l778_778039


namespace quadratic_inequality_solution_l778_778202

theorem quadratic_inequality_solution:
  {x : ℝ | x^2 - 44 * x + 392 ≤ 0} = set.Icc 12.409 31.591 :=
sorry

end quadratic_inequality_solution_l778_778202


namespace business_outcome_l778_778443

variables (C1 C2 : ℝ) 
variable (P1 : C1 ≠ 0)
variable (P2 : C2 ≠ 0)

theorem business_outcome :
  let revenue := 6.0 in
  let cost1 := (3.0 / 1.25) in
  let cost2 := (3.0 / 0.7) in
  let total_cost := cost1 + cost2 in
  revenue - total_cost = -0.69 :=
by
  sorry

end business_outcome_l778_778443


namespace ellipse_equation_and_fixed_point_proof_l778_778286

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778286


namespace mixed_groups_count_l778_778892

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l778_778892


namespace polynomial_expansion_l778_778114

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 :=
by sorry

end polynomial_expansion_l778_778114


namespace solve_for_a_days_l778_778528

-- Definitions based on the problem conditions
def work_rate_a (x : ℝ) : ℝ := 1 / x
def work_rate_b : ℝ := 1 / 10
def work_rate_together : ℝ := 6 / 35

-- Theorem statement
theorem solve_for_a_days (x : ℝ) (h : work_rate_a x + work_rate_b = work_rate_together) : x = 14 := by
  sorry

end solve_for_a_days_l778_778528


namespace part_one_part_two_l778_778433

-- Define the condition
variable (z : ℂ) (h : |2 * z + 15| = real.sqrt 3 * |conj z + 10|)

-- Define the first proof problem
theorem part_one : |z| = 5 * real.sqrt 3 :=
sorry

-- Define the second proof problem
theorem part_two (a : ℝ) : (∃ (a : ℝ), (a = 5 * real.sqrt 3 ∨ a = -5 * real.sqrt 3) ∧
    (∃ z : ℂ, |z| = 5 * real.sqrt 3 ∧ (z / a + a / z).im = 0)) :=
sorry

end part_one_part_two_l778_778433


namespace keith_missed_games_l778_778074

variable (totalGames attendedGames : ℕ)

theorem keith_missed_games (h1 : totalGames = 8) (h2 : attendedGames = 4) :
  totalGames - attendedGames = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_self 4

end keith_missed_games_l778_778074


namespace brown_dog_count_l778_778754

theorem brown_dog_count:
  ∀ (T L N : ℕ), T = 45 → L = 36 → N = 8 → (T - N - (T - L - N) = 37) :=
by
  intros T L N hT hL hN
  sorry

end brown_dog_count_l778_778754


namespace parabola_intersects_x_axis_two_points_l778_778355

theorem parabola_intersects_x_axis_two_points (m : ℝ) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ mx^2 + (m-3)*x - 1 = 0 :=
by
  sorry

end parabola_intersects_x_axis_two_points_l778_778355


namespace count_primes_50_to_70_gt_55_l778_778731

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

def condition_primes_50_to_70 (p : ℕ) : Prop :=
  p > 50 ∧ p < 70 ∧ is_prime p ∧ p > 55

theorem count_primes_50_to_70_gt_55 : (List.filter condition_primes_50_to_70 (primes_in_range 50 70)).length = 3 :=
by
  sorry

end count_primes_50_to_70_gt_55_l778_778731


namespace length_AD_l778_778394

-- Define the lengths of the segments
variables (BO OD AO OC AB AD : ℝ)
-- Define the intersection point O
variables (A B C D O : Type) [metric_space O]

-- Hypotheses based on the conditions
axiom h1 : BO = 4
axiom h2 : OD = 6
axiom h3 : AO = 8
axiom h4 : OC = 3
axiom h5 : AB = 6

-- The theorem to prove
theorem length_AD :
  AD = Real.sqrt 166 :=
sorry

end length_AD_l778_778394


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l778_778343

def f (x : ℝ) : ℝ := (2 * Real.sin (x + Real.pi / 3) + Real.sin x) * Real.cos x - Real.sqrt 3 * (Real.sin x)^2

theorem smallest_positive_period_of_f : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem max_min_values_of_f_on_interval : ∀ x ∈ Icc 0 (Real.pi / 4), 1 ≤ f x ∧ f x ≤ 2 := by
  sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l778_778343


namespace problem_l778_778332

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778332


namespace ellipse_and_fixed_point_l778_778294

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l778_778294


namespace speed_ratio_dog_to_hare_l778_778132

-- Define variables for leap distances
variables (D H : ℝ)

-- Define initial relationships
axiom dog_leap_twice_hare_leap : D = 2 * H

-- Define relationships in challenging terrain
axiom dog_reduced_leap : D' = 0.75 * D
axiom hare_reduced_leap : H' = 0.50 * H

-- Define the theorem for the speed ratio
theorem speed_ratio_dog_to_hare : (D' / H') = 3 :=
by
  -- Use the given axioms and conditions to establish the result
  sorry

end speed_ratio_dog_to_hare_l778_778132


namespace count_valid_n_l778_778672

theorem count_valid_n (n : ℤ) : 
  (∃ (count : ℕ), count = 9 ∧ 
  (∀ n : ℤ, (∃ k : ℤ, 4000 * (2^n) * (5^(-n)) = k) → count = 9)) := 
by
  sorry

end count_valid_n_l778_778672


namespace sum_inequality_l778_778227

noncomputable def S (a : ℕ → ℝ) (m : ℝ) (n : ℕ) : ℝ := (9/8) * a n - (4/3) * 3^n + m

theorem sum_inequality (a : ℕ → ℝ) (m : ℝ) (h1 : ∀ n : ℕ, a n ≥ 64 / 3)
  (h2 : m = 4 / 3) :
  ∀ n : ℕ, (∑ k in finset.range (n + 1), (3 ^ k) / (S a m k)) < 3 / 16 :=
sorry

end sum_inequality_l778_778227


namespace average_mb_per_hour_l778_778131

theorem average_mb_per_hour
  (days : ℕ)
  (original_space  : ℕ)
  (compression_rate : ℝ)
  (total_hours : ℕ := days * 24)
  (effective_space : ℝ := original_space * (1 - compression_rate))
  (space_per_hour : ℝ := effective_space / total_hours) :
  days = 20 ∧ original_space = 25000 ∧ compression_rate = 0.10 → 
  (Int.floor (space_per_hour + 0.5)) = 47 := by
  intros
  sorry

end average_mb_per_hour_l778_778131


namespace no_2014_ambiguous_integer_exists_l778_778144

theorem no_2014_ambiguous_integer_exists :
  ∀ k : ℕ, (∃ m : ℤ, k^2 - 8056 = m^2) → (∃ n : ℤ, k^2 + 8056 = n^2) → false :=
by
  -- Proof is omitted as per the instructions
  sorry

end no_2014_ambiguous_integer_exists_l778_778144


namespace HN_passes_through_fixed_point_l778_778260

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778260


namespace calculate_expression_l778_778176

theorem calculate_expression (x : ℕ) (h : x = 3) : 5 * 7 + 9 * 4 - 35 / 5 + x * 2 = 70 :=
by
  rw [h]
  norm_num
  sorry

end calculate_expression_l778_778176


namespace problem_l778_778325

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l778_778325


namespace restoration_of_axes_l778_778005

theorem restoration_of_axes (parabola : ℝ → ℝ) (h : ∀ x, parabola x = x^2) : 
  ∃ (origin : ℝ × ℝ) (x_axis y_axis : ℝ × ℝ → Prop), 
    (∀ x, x_axis (x, 0)) ∧ 
    (∀ y, y_axis (0, y)) ∧ 
    origin = (0, 0) := 
sorry

end restoration_of_axes_l778_778005


namespace total_digits_even_integers_3002_l778_778933

theorem total_digits_even_integers_3002 : 
  let count_digits_in_range (start end : ℕ) (digits_per_num : ℕ) := 
        ((end - start) / 2 + 1) * digits_per_num,
      total_digits := 
        count_digits_in_range 2 8 1 + -- count of 1-digit even numbers
        count_digits_in_range 10 98 2 + -- count of 2-digit even numbers
        count_digits_in_range 100 998 3 + -- count of 3-digit even numbers
        count_digits_in_range 1000 6004 4 -- count of 4-digit even numbers
  in
  total_digits = 11456 :=
by {
  let count_digits_in_range (start end digits_per_num : ℕ) := ((end - start) / 2 + 1) * digits_per_num,
  let total_digits := count_digits_in_range 2 8 1 + 
                      count_digits_in_range 10 98 2 + 
                      count_digits_in_range 100 998 3 + 
                      count_digits_in_range 1000 6004 4,
  have h1 : count_digits_in_range 2 8 1 = 4 := by sorry,
  have h2 : count_digits_in_range 10 98 2 = 90 := by sorry,
  have h3 : count_digits_in_range 100 998 3 = 1350 := by sorry,
  have h4 : count_digits_in_range 1000 6004 4 = 10012 := by sorry,
  have h_total : 4 + 90 + 1350 + 10012 = 11456 := by sorry,
  exact h_total
}

end total_digits_even_integers_3002_l778_778933


namespace chessboard_game_first_player_wins_l778_778565

theorem chessboard_game_first_player_wins :
  (∀ board piece : ℕ, ∀ player_move : ℕ → ℕ → ℕ, ∀ valid_move : ℕ → ℕ → Prop,
  (∀ move, valid_move board (player_move piece move) →
  ¬ visited move) →
  (∀ piece, ∃ move, valid_move board (player_move piece move)) →
  first_player_wins) :=
  sorry

end chessboard_game_first_player_wins_l778_778565


namespace smaller_number_l778_778477

theorem smaller_number (L S : ℕ) (h₁ : L - S = 2395) (h₂ : L = 6 * S + 15) : S = 476 :=
by
sorry

end smaller_number_l778_778477


namespace simplify_expression_l778_778544

noncomputable def expr := (-1 : ℝ)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1 / 8) * Real.sqrt 32

theorem simplify_expression : expr = 3 := 
by sorry

end simplify_expression_l778_778544


namespace sequence_a_formula_l778_778237

def sequence_a (a : ℕ → ℤ) := ∀ n : ℕ, n > 0 → a (n + 1) = 2 * (∑ i in finset.range n, a (i + 1)) + 1
def first_term (a : ℕ → ℤ) := a 1 = 1
def log_base_3 (x : ℤ) := log x / log 3
def sequence_b (a b : ℕ → ℤ) := ∀ n : ℕ, b n = log_base_3 (a (n + 1))
def sum_T (a b T : ℕ → ℤ) := ∀ n : ℕ, T n = (∑ i in finset.range n, a (i + 1) + b (i + 1))

theorem sequence_a_formula {a b T : ℕ → ℤ}
  (h1 : first_term a)
  (h2 : sequence_a a)
  (h3 : sequence_b a b)
  (h4 : sum_T a b T) :
  (∀ n : ℕ, n > 0 → a n = 3 ^ (n - 1)) ∧
  (∀ n : ℕ, n > 0 → T n = (3 ^ n + n ^ 2 + n - 1) / 2) :=
  sorry

end sequence_a_formula_l778_778237


namespace bug_prob_eighth_move_l778_778963

def Q : ℕ → ℚ
| 0     := 1
| (n+1) := 1 / 3 * (1 - Q n)

theorem bug_prob_eighth_move : Q 8 = 547 / 2187 := by
  sorry

end bug_prob_eighth_move_l778_778963


namespace collinearity_APQ_l778_778168

-- Assume we have points A, B, C, X, Y, P, Q
-- Assume we have a triangle ABC such that AB > AC
variables {A B C X Y P Q : Type*}
variables [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ X]
variables [affine_space ℝ Y] [affine_space ℝ P] [affine_space ℝ Q]
variables [geom: A → B → C] -- assuming geometry space

-- Definition of bisector and angles
def is_bisector (b : A → X → B → ℝ) : Prop := 
∃ (A B C X : geom), (angles_in_triangles A B X = angles_in_triangles A C X)

-- Definitions for angles as given conditions
def angles_eq (α : B → A → X → Y → ℝ) : Prop := ∠(A, B, X) = ∠(A, C, Y)

-- Conditions although one may not detail all geometry defns
axiom H1 : AB > AC
axiom H2 : is_bisector A
axiom H3 : angles_eq A (B ∠ AC)(X ∠)Y
axiom P_def : intersects (extn_line A B X) (segment C Y) P
axiom circ_def : (int_circ_of_tris geom_def: B P Y) and (int_circ_of_tris geom_def: C P X Q)

-- Main theorem to be proved
theorem collinearity_APQ : collinear {A P Q} :=
by
 sorry

end collinearity_APQ_l778_778168


namespace probability_interval_l778_778336

noncomputable def normalDist := NormalDist.mk 1 1

theorem probability_interval :
  let ξ : ℝ → ℝ := fun x => normalDist.pdf x in
  ∫ x in Icc (-1) 3, ξ x = 0.954 :=
by
  -- Given conditions
  have hξ₃ : ∫ x in Iic 3, ξ x = 0.977 := sorry,
  -- Proof of the statement
  sorry

end probability_interval_l778_778336


namespace sum_of_interior_ninth_row_l778_778773

theorem sum_of_interior_ninth_row : 
  Sum of the interior numbers of the fourth row is 6 ∧
  Sum of the interior numbers of the fifth row is 14 →
  Sum of the interior numbers of the ninth row = 254 := 
by 
  -- Assuming the conditions hold, we will prove the conclusion.
  sorry

end sum_of_interior_ninth_row_l778_778773


namespace find_f2_l778_778231

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y + 1)
  (H2 : f 8 = 15) :
  f 2 = 3 := 
sorry

end find_f2_l778_778231


namespace correct_statements_l778_778718
noncomputable def f (x : ℝ) := abs (cos (2 * x) + cos x)

theorem correct_statements:
  ( (∀ x, f (-x) = f x) ∧ 
    (∀ y, (y = abs (cos (2 * y) + cos y) → y ∈ Set.Icc (0 : ℝ) (9/8))) ∧
    (∀ x, x ∈ Set.Icc ((-5) * Real.pi / 4) (- Real.pi) → 
    (∀ y, y ∈ Set.Icc ((-5) * Real.pi / 4) (- Real.pi) → x < y → f x > f y)) ∧
    (f x = 0 → x ∈ Finset.range ((-2) * Real.pi, 2 * Real.pi) → Finset.card (Finset.filter (λ x, f x = 0) (Finset.range ((-2) * Real.pi, 2 * Real.pi))) = 8)) 
→ (correct := ([1, 3] : List Nat) sorry

end correct_statements_l778_778718


namespace roots_of_quadratic_l778_778373

theorem roots_of_quadratic (a b : ℝ) (h₁ : a + b = 2) (h₂ : a * b = -3) : a^2 + b^2 = 10 := 
by
  -- proof steps go here, but not required as per the instruction
  sorry

end roots_of_quadratic_l778_778373


namespace probability_of_slope_ge_one_l778_778342

-- Define conditions
def equation_of_line (a : ℝ) : Prop :=
  -5 ≤ a ∧ a ≤ 4 ∧ (∃ x y : ℝ, 2 * y = -a * x + 3)

-- Define the slope condition
def slope_condition (a : ℝ) : Prop :=
  -a / 2 ≥ 1

-- Define the probability calculation
def probability_of_slope_not_less_than_one : ℝ :=
  (2 - (-5)) / (4 - (-5))

-- Now state the problem
theorem probability_of_slope_ge_one (a : ℝ) (cond : equation_of_line a) : 
  slope_condition a → probability_of_slope_not_less_than_one = 1 / 3 :=
sorry

end probability_of_slope_ge_one_l778_778342


namespace general_form_of_line_l778_778970

theorem general_form_of_line (x y : ℝ) 
  (passes_through_A : ∃ y, 2 = y)          -- Condition 1: passes through A(-2, 2)
  (same_y_intercept : ∃ y, 6 = y)          -- Condition 2: same y-intercept as y = x + 6
  : 2 * x - y + 6 = 0 := 
sorry

end general_form_of_line_l778_778970


namespace find_distinct_numbers_l778_778649

def first_digit (n : ℕ) : ℕ :=
  n / 100

def S (x1 x2 x3 x4 : ℕ) : ℕ :=
  x1 + x2 + x3 + x4

def divisible_by_any_three (x1 x2 x3 x4 S : ℕ) : Prop :=
  ∀ (i j k : ℕ) (h : i ≠ j) (h' : j ≠ k) (h'' : k ≠ i), 
    let x := [x1, x2, x3, x4] in 
    (i = 0 ∧ j = 1 ∧ k = 2 → x1 + x2 + x3 ∣ S) ∧
    (i = 0 ∧ j = 1 ∧ k = 3 → x1 + x2 + x4 ∣ S) ∧
    (i = 0 ∧ j = 2 ∧ k = 3 → x1 + x3 + x4 ∣ S) ∧
    (i = 1 ∧ j = 2 ∧ k = 3 → x2 + x3 + x4 ∣ S)

theorem find_distinct_numbers :
  ∃ (x1 x2 x3 x4 : ℕ), 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧
    x3 ≠ x4 ∧
    first_digit x1 = first_digit x2 ∧
    first_digit x2 = first_digit x3 ∧
    first_digit x3 = first_digit x4 ∧
    first_digit x4 = 1 ∧
    let S := S x1 x2 x3 x4 in
    S = 540 ∧
    divisible_by_any_three x1 x2 x3 x4 S := 
begin
  use [108, 135, 180, 117],
  split, {exact 108 ≠ 135}, split, { exact 108 ≠ 180 },
  split, { exact 108 ≠ 117 }, split, { exact 135 ≠ 180 },
  split, { exact 135 ≠ 117 }, split, { exact 180 ≠ 117 },
  split, { repeat { rw first_digit, show 108 / 100 = 1, norm_num } },
  split, { repeat { rw first_digit, show 135 / 100 = 1, norm_num } },
  split, { repeat { rw first_digit, show 180 / 100 = 1, norm_num } },
  split, { repeat { rw first_digit, show 117 / 100 = 1, norm_num } },
  split, { exact 1 },
  split, { refl },
  sorry,
end

end find_distinct_numbers_l778_778649


namespace relative_errors_are_equal_l778_778161

theorem relative_errors_are_equal :
  let e1 := 0.04
  let l1 := 20.0
  let e2 := 0.3
  let l2 := 150.0
  (e1 / l1) = (e2 / l2) :=
by
  sorry

end relative_errors_are_equal_l778_778161


namespace conic_section_eccentricity_l778_778337

noncomputable def eccentricity (m : ℝ) : ℝ :=
if m = 2 then 1 / Real.sqrt 2 else
if m = -2 then Real.sqrt 3 else
0

theorem conic_section_eccentricity (m : ℝ) (h : 4 * 1 = m * m) :
  eccentricity m = 1 / Real.sqrt 2 ∨ eccentricity m = Real.sqrt 3 :=
by
  sorry

end conic_section_eccentricity_l778_778337


namespace sum_of_super_cool_triangle_areas_l778_778584

noncomputable def super_cool_triangle_sum_area : ℕ :=
  let leg_pairs := [(7, 42), (8, 24), (9, 18), (10, 15), (12, 12)] in
  let areas := leg_pairs.map (λ p, (p.1 * p.2) / 2) in
  areas.sum

theorem sum_of_super_cool_triangle_areas : super_cool_triangle_sum_area = 471 :=
by
  sorry

end sum_of_super_cool_triangle_areas_l778_778584


namespace sale_percent_saved_l778_778534

noncomputable def percent_saved (P : ℝ) : ℝ := (3 * P) / (6 * P) * 100

theorem sale_percent_saved :
  ∀ (P : ℝ), P > 0 → percent_saved P = 50 :=
by
  intros P hP
  unfold percent_saved
  have hP_nonzero : 6 * P ≠ 0 := by linarith
  field_simp [hP_nonzero]
  norm_num
  sorry

end sale_percent_saved_l778_778534


namespace shaded_area_l778_778604

-- Definitions for the problem conditions
def radius := 1
def semicircle (x : ℝ) := x -- Placeholder definition for semicircle location
def point_on_semicircle := { C := (radius : ℝ) } -- Center C
def D_on_AB (D : ℝ) := D -- Placeholder definition for point D on semicircle
def perpendicular := true -- represents CD ⟂ AB

-- Placeholder definitions for points and extensions
def E (BD_extend : ℝ) := BD_extend
def F (AD_extend : ℝ) := AD_extend

-- Semicircular arcs centered at B and A respectively
def semicircular_arc_AE (radius : ℝ) := radius
def semicircular_arc_BF (radius : ℝ) := radius

def arc_centered_at_D := D_on_AB -- Placeholder definition

-- The area of the shaded region is what needs to be proven
theorem shaded_area : 2 * π - π * sqrt 2 - 1 = sorry :=
by
  sorry

end shaded_area_l778_778604


namespace HN_passes_through_fixed_point_l778_778257

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778257


namespace miller_rabin_time_complexity_l778_778462

theorem miller_rabin_time_complexity (n : ℕ) (α : ℕ) (β : ℕ) (u : ℕ) :
  n - 1 = 2^α * β ∧ odd β ∧ 
  -- Complexity conditions for loops
  (∀ t, loop1_complexity t = ℕ → t ^ O(log(n))) ∧ 
  (∀ t k, loop2_complexity t k = ℕ → k * t = ℕ ^ (O(log(n)^2))) ∧
  (∀ t k, loop3_complexity t k = ℕ → k * t = ℕ ^ (O(log(n)^2))) →
  -- Conclusion for the overall time complexity
  O(log(n)^3) :=
by
  sorry

end miller_rabin_time_complexity_l778_778462


namespace find_p_not_geometric_if_ratios_not_equal_l778_778243

def geometric (u : ℕ → ℝ) : Prop :=
  ∃ r ∈ ℝ, ∀ n : ℕ, u (n + 1) = r * u n

def sequence_a (n : ℕ) : ℝ := 2 ^ n
def sequence_b (n : ℕ) : ℝ := 3 ^ n
def sequence_c (n : ℕ) : ℝ := sequence_a n + sequence_b n

-- Part 1
theorem find_p (p : ℝ) :
  geometric (λ n, sequence_c (n + 1) - p * sequence_c n) → (p = 2 ∨ p = 3) := sorry

-- Part 2
theorem not_geometric_if_ratios_not_equal (q1 q2 : ℝ) (h1 : q1 ≠ q2)
  (a_n : ℕ → ℝ) (b_n : ℕ → ℝ)
  (h2 : ∀ n, a_n n = (q1)^n ∧ b_n n = (q2)^n) :
  ¬ geometric (λ n, a_n n + b_n n) := sorry

end find_p_not_geometric_if_ratios_not_equal_l778_778243


namespace part1_part2_part3_l778_778509

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem part1 : determinant (-3) (-2) 4 5 = -7 := by
  sorry

theorem part2 (x: ℝ) (h: determinant 2 (-2 * x) 3 (-5 * x) = 2) : x = -1/2 := by
  sorry

theorem part3 (m n x: ℝ) 
  (h1: determinant (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = 
        determinant 6 (-1) (-n) x) : 
    m = -3/8 ∧ n = -7 := by
  sorry

end part1_part2_part3_l778_778509


namespace MaryIncomeIs64PercentOfJuanIncome_l778_778807

variable {J T M : ℝ}

-- Conditions
def TimIncome (J : ℝ) : ℝ := 0.40 * J
def MaryIncome (T : ℝ) : ℝ := 1.60 * T

-- Theorem to prove
theorem MaryIncomeIs64PercentOfJuanIncome (J : ℝ) :
  MaryIncome (TimIncome J) = 0.64 * J :=
by
  sorry

end MaryIncomeIs64PercentOfJuanIncome_l778_778807


namespace bianca_points_per_bag_l778_778172

theorem bianca_points_per_bag (total_bags : ℕ) (not_recycled : ℕ) (total_points : ℕ) 
  (h1 : total_bags = 17) 
  (h2 : not_recycled = 8) 
  (h3 : total_points = 45) : 
  total_points / (total_bags - not_recycled) = 5 :=
by
  sorry 

end bianca_points_per_bag_l778_778172


namespace find_B_max_f_A_l778_778753

namespace ProofProblem

-- Definitions
variables {A B C a b c : ℝ} -- Angles and sides in the triangle
noncomputable def givenCondition (A B C a b c : ℝ) : Prop :=
  2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 4

-- Problem Statements (to be proved)
theorem find_B (h : givenCondition A B C a b c) : B = Real.pi / 6 := sorry

theorem max_f_A (A : ℝ) (B : ℝ) (h1 : 0 < A) (h2 : A < 5 * Real.pi / 6) (h3 : B = Real.pi / 6) : (∃ (x : ℝ), f x = 1 / 2) := sorry

end ProofProblem

end find_B_max_f_A_l778_778753


namespace dice_product_144_probability_l778_778526

theorem dice_product_144_probability :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      events := {abc : ℕ × ℕ × ℕ | abc.1 ∈ S ∧ abc.2 ∈ S ∧ abc.3 ∈ S ∧ abc.1 * abc.2 * abc.3 = 144} in
  (|events| : ℝ) / 216 = 1 / 72 :=
by
  -- Assumption of finite set cardinality and mention of corresponding probabilities can be filled here
  sorry

end dice_product_144_probability_l778_778526


namespace find_f1988_11_l778_778214

-- Define a function to compute the sum of the digits of a number.
def sumOfDigits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

-- Define the function f1(k) as the square of the sum of the digits of k.
def f1 (k : ℕ) : ℕ :=
  (sumOfDigits k)^2

-- Define the recursive function fn(k) as described in the problem statement.
def f : ℕ → ℕ → ℕ
| 1, k := f1 k
| (n + 2), k := f1 (f (n + 1) k)

-- Theorem to prove the required result.
theorem find_f1988_11 : f 1988 11 = 169 :=
by {
  sorry
}

end find_f1988_11_l778_778214


namespace can_remove_one_digit_to_be_fair_l778_778931

-- Definition of digit positions being fair
def is_fair (digits : List ℕ) : Prop :=
  let even_pos := (List.range digits.length).filter (λ i, i % 2 = 1)
  let odd_pos := (List.range digits.length).filter (λ i, i % 2 = 0)
  let even_digits := even_pos.map (λ i, digits.get i)
  let odd_digits := odd_pos.map (λ i, digits.get i)
  even_digits.filter (λ d, d % 2 = 0) |>.length = odd_digits.filter (λ d, d % 2 = 0) |>.length

-- The main theorem statement
theorem can_remove_one_digit_to_be_fair (digits : List ℕ) (h_len : digits.length = 2 * m + 1) : 
  ∃ (d : ℕ), is_fair (digits.erase d) := 
sorry

end can_remove_one_digit_to_be_fair_l778_778931


namespace max_tan_A_l778_778251

theorem max_tan_A {A B C : ℝ} 
  (h₁ : A + B + C = π) 
  (h₂ : sin A + 2 * sin B * cos C = 0) 
  (h₃ : 0 < A) 
  (h₄ : 0 < B) 
  (h₅ : 0 < C) 
  (h₆ : A < π)
  (h₇ : B < π) 
  (h₈ : C < π)
  : tan A ≤ (sqrt 3 / 3) := 
sorry

end max_tan_A_l778_778251


namespace part1_part2_part3_l778_778129

section ShoppingMall

variable (x y a b : ℝ)
variable (cpaA spaA cpaB spaB : ℝ)
variable (n total_y yuan : ℝ)

-- Conditions given in the problem
def cost_price_A := 160
def selling_price_A := 220
def cost_price_B := 120
def selling_price_B := 160
def total_clothing := 100
def min_A_clothing := 60
def max_budget := 15000
def discount_diff := 4
def max_profit_with_discount := 4950

-- Definitions applied from conditions
def profit_per_piece_A := selling_price_A - cost_price_A
def profit_per_piece_B := selling_price_B - cost_price_B

-- Question 1: Functional relationship between y and x
theorem part1 : 
  (∀ (x : ℝ), x ≥ 0 → x ≤ total_clothing → 
  y = profit_per_piece_A * x + profit_per_piece_B * (total_clothing - x)) →
  y = 20 * x + 4000 := 
sorry

-- Question 2: Maximum profit under given cost constraints
theorem part2 : 
  (min_A_clothing ≤ x ∧ x ≤ 75 ∧ 
  (cost_price_A * x + cost_price_B * (total_clothing - x) ≤ max_budget)) →
  y = 20 * 75 + 4000 → 
  y = 5500 :=
sorry

-- Question 3: Determine a under max profit condition
theorem part3 : 
  (a - b = discount_diff ∧ 0 < a ∧ a < 20 ∧ 
  (20 - a) * 75 + 4000 + 100 * a - 400 = max_profit_with_discount) →
  a = 9 :=
sorry

end ShoppingMall

end part1_part2_part3_l778_778129


namespace maximize_profit_l778_778075

noncomputable def profit (x : ℝ) : ℝ :=
  16 - 4/(x+1) - x

theorem maximize_profit (a : ℝ) (h : 0 ≤ a) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ a ∧ profit x = max 13 (16 - 4/(a+1) - a) := by
  sorry

end maximize_profit_l778_778075


namespace triangle_constructible_l778_778191

/-- Given an angle α, the length of the altitude from the vertex of this angle m_a,
    and the length of the angle bisector l_α, prove that a triangle is constructible
    if l_α ≥ m_a. -/
theorem triangle_constructible (α : ℝ) (m_a l_α : ℝ) : l_α ≥ m_a → ∃ (A B C : Type), is_triangle A B C α m_a l_α :=
begin
  sorry
end

end triangle_constructible_l778_778191


namespace sin_cos_45_degrees_l778_778620

noncomputable def Q : ℝ × ℝ := (1 / real.sqrt 2, 1 / real.sqrt 2)

def is_unit_circle (p : ℝ × ℝ) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 = 1

def is_45_45_90_triangle (h : ℝ) (a : ℝ) : Prop :=
  h = a * real.sqrt 2

theorem sin_cos_45_degrees :
  (∀ p : ℝ × ℝ, is_unit_circle p → ∃ a : ℝ, is_45_45_90_triangle 1 a ∧ p = (a, a)) →
  real.sin (real.pi / 4) = 1 / real.sqrt 2 ∧ real.cos (real.pi / 4) = 1 / real.sqrt 2 :=
by
  sorry

end sin_cos_45_degrees_l778_778620


namespace sqrt_fraction_simplification_l778_778199

theorem sqrt_fraction_simplification :
  (Real.sqrt ((25 / 49) - (16 / 81)) = (Real.sqrt 1241) / 63) := by
  sorry

end sqrt_fraction_simplification_l778_778199


namespace Carlton_outfit_combinations_l778_778181

theorem Carlton_outfit_combinations :
  ∃ (svests bshirts ties shoes socks : ℕ),
    bshirts = 5 ∧
    svests = 3 * bshirts ∧
    ties = 2 * svests ∧
    shoes = 4 * ties ∧
    socks = 6 * shoes ∧
    (bshirts * svests * ties * shoes * socks = 77760000) :=
by
  let bshirts := 5
  let svests := 3 * bshirts
  let ties := 2 * svests
  let shoes := 4 * ties
  let socks := 6 * shoes
  use [svests, bshirts, ties, shoes, socks]
  simp [svests, ties, shoes, socks]
  sorry

end Carlton_outfit_combinations_l778_778181


namespace initial_distance_adrian_colton_l778_778156

variable (speed time distance_between : ℝ)

noncomputable def adrian_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem initial_distance_adrian_colton
  (initial_distance correct_distance : ℝ)
  (h1 : speed = 17)
  (h2 : time = 13)
  (h3 : distance_between = 68)
  (h4 : adrian_distance speed time = 221)
  (h5 : initial_distance = adrian_distance speed time + distance_between) :
  initial_distance = correct_distance :=
by
  rw [h4]
  sorry

end initial_distance_adrian_colton_l778_778156


namespace tetrahedron_inscribed_sphere_radius_l778_778829

theorem tetrahedron_inscribed_sphere_radius (a : ℝ) (r : ℝ) (a_pos : 0 < a) :
  (r = a * (Real.sqrt 6 + 1) / 8) ∨ 
  (r = a * (Real.sqrt 6 - 1) / 8) :=
sorry

end tetrahedron_inscribed_sphere_radius_l778_778829


namespace find_other_number_l778_778535

theorem find_other_number (B : ℕ) (hcf_cond : Nat.gcd 36 B = 14) (lcm_cond : Nat.lcm 36 B = 396) : B = 66 :=
sorry

end find_other_number_l778_778535


namespace quadratic_polynomial_real_coeff_l778_778664

theorem quadratic_polynomial_real_coeff (a b : ℂ) (h₁ : a = 5 + 2*i) 
  (h₂ : b = 5 - 2*i) (c : ℂ) (hc : c = 3) :
  3 * (X - C a) * (X - C b) = 3*X^2 - 30*X + 87 := 
by {
  sorry
}

end quadratic_polynomial_real_coeff_l778_778664


namespace initial_food_duration_l778_778501

theorem initial_food_duration
  (x : ℕ)
  (h₁ : ∀ (x : ℕ), (760 * (x - 2) = 894.11764705882354 * 17)) :
  x = 22 :=
sorry

end initial_food_duration_l778_778501


namespace limit_seq_l778_778621

open BigOperators

theorem limit_seq :
  tendsto (λ n : ℕ, (2^n + 7^n) / (2^n - 7^(n-1) : ℝ)) at_top (𝓝 (-7)) :=
begin
  sorry
end

end limit_seq_l778_778621


namespace totalGoals_l778_778760

-- Define the conditions
def louieLastMatchGoals : Nat := 4
def louiePreviousGoals : Nat := 40
def gamesPerSeason : Nat := 50
def seasons : Nat := 3
def brotherGoalsPerGame := 2 * louieLastMatchGoals

-- Define the properties derived from the conditions
def totalBrotherGoals : Nat := brotherGoalsPerGame * gamesPerSeason * seasons
def totalLouieGoals : Nat := louiePreviousGoals + louieLastMatchGoals

-- State what needs to be proved
theorem totalGoals : louiePreviousGoals + louieLastMatchGoals + brotherGoalsPerGame * gamesPerSeason * seasons = 1244 := by
  sorry

end totalGoals_l778_778760


namespace answered_both_correctly_l778_778536

variable (A B : Prop)
variable [Fintype A] [Fintype B]

-- Given conditions
def P (event : Prop) : ℝ := sorry
def complement (event : Prop) : Prop := sorry

axiom P_A : P A = 0.75
axiom P_B : P B = 0.70
axiom P_not_A_and_not_B : P (complement A ∧ complement B) = 0.20

-- Proof goal
theorem answered_both_correctly :
  P (A ∧ B) = 0.65 :=
by
  -- Prove using the principle of inclusion-exclusion
  sorry

end answered_both_correctly_l778_778536


namespace tank_capacity_l778_778073

theorem tank_capacity (rate_A rate_B : ℕ) (time_diff : ℕ) 
  (h_rate_A : rate_A = 2) (h_rate_B : rate_B = 4) (h_time_diff : time_diff = 5) :
  ∃ (t : ℕ) (C_A C_B : ℕ), 
    C_A = rate_A * (t + time_diff) ∧ 
    C_B = rate_B * t ∧ 
    C_A = C_B ∧ 
    C_A = 20 ∧ 
    C_B = 20 :=
by {
  use 5, -- t = 5
  use 20, -- C_A
  use 20, -- C_B
  split,
  { simp [h_rate_A, h_time_diff] },
  split,
  { simp [h_rate_B] },
  split,
  { simp [h_rate_A, h_rate_B, h_time_diff] },
  split,
  { refl },
  refl,
  }

end tank_capacity_l778_778073


namespace edge_properties_l778_778541

structure Polyhedron :=
  (pentagons : ℕ)
  (hexagons : ℕ)
  (stitches : ℕ)
  (intersections : ℕ)

def SoccerBall : Polyhedron :=
  { pentagons := 12,
    hexagons := 20,
    stitches := 30,
    intersections := 3 }

theorem edge_properties (P : Polyhedron) :
  (P.pentagons = 12) →
  (P.hexagons = 20) → 
  (∀ edge, ∃! parallel_edges, parallel_edges.length = 5 ∧
  ∃! perpendicular_edges, perpendicular_edges.length = 12) := 
by
  intros
  sorry

end edge_properties_l778_778541


namespace solution_to_system_l778_778359

def equations (x y : ℝ) : Prop :=
  (x - y) * x * y = 30 ∧ (x + y) * x * y = 120

theorem solution_to_system : equations 5 3 :=
by
  use 5
  use 3
  exact ⟨rfl, rfl⟩

end solution_to_system_l778_778359


namespace largest_multiple_of_7_less_than_neg_30_l778_778513

theorem largest_multiple_of_7_less_than_neg_30 (m : ℤ) (h1 : m % 7 = 0) (h2 : m < -30) : m = -35 :=
sorry

end largest_multiple_of_7_less_than_neg_30_l778_778513


namespace range_a_of_function_has_two_zeros_l778_778739

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem range_a_of_function_has_two_zeros (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) : 
  1 < a :=
sorry

end range_a_of_function_has_two_zeros_l778_778739


namespace correct_quotient_l778_778537

theorem correct_quotient :
  ∃ Q : ℕ, 21 * Q = 12 * 63 ∧ Q = 36 :=
by
  use 36
  split
  { exact Nat.mul_right_inj (by norm_num : 21 ≠ 0).mpr rfl }
  { norm_num }

end correct_quotient_l778_778537


namespace percentage_male_red_ants_proof_l778_778375

noncomputable def percentage_red_ants : ℝ := 0.85
noncomputable def percentage_female_red_ants : ℝ := 0.45
noncomputable def percentage_male_red_ants : ℝ := percentage_red_ants * (1 - percentage_female_red_ants)

theorem percentage_male_red_ants_proof : percentage_male_red_ants = 0.4675 :=
by
  -- Proof will go here
  sorry

end percentage_male_red_ants_proof_l778_778375


namespace part_a_part_b_l778_778920

-- Define the conditions
def digit5 : ℕ := 1
def digit3 : ℕ := 2
def digit2 : ℕ := 100
def total_digits : ℕ := 10

-- Define the problem statement in Lean
def ways_to_form_valid_numbers : ℕ :=
  let zero_3s := Nat.choose total_digits digit5
  let one_3 := Nat.choose total_digits digit5 * Nat.choose (total_digits - digit5) 1
  let two_3s := Nat.choose total_digits digit5 * Nat.choose (total_digits - digit5) digit3
  zero_3s + one_3 + two_3s

def number_in_position (n : ℕ) : ℕ := 5322222322

theorem part_a : ways_to_form_valid_numbers = 460 := by
  sorry

theorem part_b : number_in_position 455 = 5322222322 := by
  sorry

end part_a_part_b_l778_778920


namespace intercepts_equal_lines_parallel_l778_778354

-- Definition of the conditions: line equations
def line_l (a : ℝ) : Prop := ∀ x y : ℝ, a * x + 3 * y + 1 = 0

-- Problem (1) : The intercepts of the line on the two coordinate axes are equal
theorem intercepts_equal (a : ℝ) (h : line_l a) : a = 3 := by
  sorry

-- Problem (2): The line is parallel to x + (a-2)y + a = 0
theorem lines_parallel (a : ℝ) (h : line_l a) : (∀ x y : ℝ, x + (a-2) * y + a = 0) → a = 3 := by
  sorry

end intercepts_equal_lines_parallel_l778_778354


namespace remainder_sum_first_150_div_11300_l778_778086

theorem remainder_sum_first_150_div_11300 :
  let n := 150 in
  let S := n * (n + 1) / 2 in
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l778_778086


namespace sum_of_coefficients_eq_two_l778_778224

theorem sum_of_coefficients_eq_two {a b c : ℤ} (h : ∀ x : ℤ, x * (x + 1) = a + b * x + c * x^2) : a + b + c = 2 := 
by
  sorry

end sum_of_coefficients_eq_two_l778_778224


namespace prob_at_least_one_interested_l778_778945

noncomputable def finance_club_probability (total_members interested_members : ℕ) : ℚ :=
  let prob_not_interested_first : ℚ := (total_members - interested_members) / total_members
  let prob_not_interested_second : ℚ := (total_members - interested_members) / (total_members - 1)
  let prob_both_not_interested : ℚ := prob_not_interested_first * prob_not_interested_second
  1 - prob_both_not_interested

theorem prob_at_least_one_interested : finance_club_probability 25 20 = 23 / 24 := 
by
  -- definitions
  let total_members := 25
  let interested_members := 20
  
  -- calculations
  let prob_not_interested_first : ℚ := (total_members - interested_members) / total_members
  let prob_not_interested_second : ℚ := (total_members - interested_members) / (total_members - 1)
  let prob_both_not_interested : ℚ := prob_not_interested_first * prob_not_interested_second
  let prob_at_least_one := 1 - prob_both_not_interested
  
  -- prove the statement
  have : prob_at_least_one = 23 / 24 := by
    simp [prob_not_interested_first, prob_not_interested_second, prob_both_not_interested, prob_at_least_one]

  exact this

end prob_at_least_one_interested_l778_778945


namespace jack_pays_5445_l778_778409

def tire_costs : ℕ → ℕ
| 0 := 230
| 1 := 230
| 2 := 250
| 3 := 250
| 4 := 280
| 5 := 280
| _ := 0

def window_costs : ℕ → ℕ
| 0 := 700
| 1 := 800
| 2 := 900
| _ := 0

def paint_job_cost : ℕ := 1200

def fence_costs (planks : ℕ) : ℕ := planks * 35 + 150

def total_tire_cost : ℕ := List.sum (List.map tire_costs [0, 1, 2, 3, 4, 5])

def total_window_cost : ℕ := List.sum (List.map window_costs [0, 1, 2])

def total_fence_cost : ℕ := fence_costs 5

def total_damage_cost : ℕ := total_tire_cost + total_window_cost + paint_job_cost + total_fence_cost

theorem jack_pays_5445 : total_damage_cost = 5445 := by
  have h1 : total_tire_cost = 1520 := by sorry
  have h2 : total_window_cost = 2400 := by sorry
  have h3 : paint_job_cost = 1200 := by sorry
  have h4 : total_fence_cost = 325 := by sorry
  calc
    total_damage_cost
    _ = total_tire_cost + total_window_cost + paint_job_cost + total_fence_cost := by rfl
    _ = 1520 + 2400 + 1200 + 325 := by rw [h1, h2, h3, h4]
    _ = 5445 := by norm_num

end jack_pays_5445_l778_778409


namespace area_region_bounded_l778_778636

theorem area_region_bounded (x y : ℝ) :
  (x^2 + y^2 = 6 * abs (x - y) + 6 * abs (x + y)) →
  area_of_region_bounded_by_graph x y = 72 :=
sorry

end area_region_bounded_l778_778636


namespace trapezoid_leg_length_proof_l778_778827

noncomputable def circumscribed_trapezoid_leg_length 
  (area : ℝ) (acute_angle_base : ℝ) : ℝ :=
  -- Hypothesis: Given conditions of the problem
  if h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3 then
    -- The length of the trapezoid's leg
    8
  else
    0

-- Statement of the proof problem
theorem trapezoid_leg_length_proof 
  (area : ℝ) (acute_angle_base : ℝ)
  (h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3) :
  circumscribed_trapezoid_leg_length area acute_angle_base = 8 := 
by {
  -- skipping actual proof
  sorry
}

end trapezoid_leg_length_proof_l778_778827


namespace sin_double_angle_l778_778371

variable (α : ℝ)

-- Assumption: Given condition
def given_condition : Prop := sin (α - π / 4) = -cos (2 * α)

-- Theorem statement with the given condition and the conclusion
theorem sin_double_angle (h : given_condition α) : sin (2 * α) = -1 / 2 := 
sorry

end sin_double_angle_l778_778371


namespace minimum_trips_l778_778063

/-- 
Problem: There are 400 tons of cargo stored in a warehouse, each with a weight that is a multiple 
of a quintal and does not exceed 10 tons. It is known that any two pieces of cargo have different 
weights. Prove that the minimum number of trips a 10-ton truck must make to guarantee the 
transportation of all this cargo from the warehouse is 51.
-/

theorem minimum_trips (n : ℕ) (weights : fin n → ℕ)
  (h1 : ∀ i, weights i % 10 = 0)
  (h2 : ∀ i j, i ≠ j → weights i ≠ weights j)
  (h3 : ∑ i, weights i = 4000)
  (h4 : ∀ i, weights i ≤ 100) :
  ∃ t : ℕ, t = 51 ∧ ∑ i in finset.range t, weights i ≤ 10000 :=
sorry

end minimum_trips_l778_778063


namespace anton_has_more_cards_than_ann_l778_778165

-- Define Heike's number of cards
def heike_cards : ℕ := 60

-- Define Anton's number of cards in terms of Heike's cards
def anton_cards (H : ℕ) : ℕ := 3 * H

-- Define Ann's number of cards as equal to Heike's cards
def ann_cards (H : ℕ) : ℕ := H

-- Theorem statement
theorem anton_has_more_cards_than_ann 
  (H : ℕ) (H_equals : H = heike_cards) : 
  anton_cards H - ann_cards H = 120 :=
by
  -- At this point, the actual proof would be inserted.
  sorry

end anton_has_more_cards_than_ann_l778_778165


namespace number_of_workers_l778_778104

theorem number_of_workers (W : ℕ)
  (h1 : (8000 * W) = (12000 * 10 + 6000 * (W - 10))) :
  W = 30 :=
by {
  sorry,
}   

end number_of_workers_l778_778104


namespace number_of_mixed_groups_l778_778882

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l778_778882


namespace fair_children_count_l778_778024

theorem fair_children_count (C A : ℕ)
(h1 : C + A = 2200)
(h2 : 1.50 * C + 4.00 * A = 5050) :
  C = 1500 :=
sorry

end fair_children_count_l778_778024


namespace smallest_n_for_sum_condition_l778_778637

theorem smallest_n_for_sum_condition :
  ∃ n, n ≥ 4 ∧ (∀ S : Finset ℤ, S.card = n → ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a + b - c - d) % 20 = 0) ∧ n = 9 :=
by
  sorry

end smallest_n_for_sum_condition_l778_778637


namespace smallest_N_divisible_by_12_l778_778825

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

lemma fib_initial_conditions :
  fib 1 = 1 ∧ fib 2 = 1 := by
  split; simp [fib]
  · sorry
  · sorry

lemma fib_recursive (n : ℕ) :
  fib (n+2) = fib (n+1) + fib n := by
  simp [fib]; sorry

lemma sum_of_10_consecutive_fib_mod_11 (n : ℕ) :
  (∑ i in (finset.range 10).map (λ x, x + n), fib i) % 11 = 0 := by
  sorry

theorem smallest_N_divisible_by_12 :
  ∃ N, (∀ n, ∑ i in (finset.range N).map (λ x, x + n), fib i % 12 = 0) ∧ N = 24 := by
  use 24
  split
  · intros n
    -- Here we must demonstrate that the sum of 24 consecutive Fibonacci numbers modulo 12 equals zero
    sorry
  · rfl

end smallest_N_divisible_by_12_l778_778825


namespace sum_of_fractions_approx_l778_778932

theorem sum_of_fractions_approx :
  let S := ∑ n in Finset.range (1006 + 1), if n.bodd then (2:ℚ) / (n * (n + 2)) else 0
  abs (↑S - 0.999) < 0.001 :=
by
  unfold Finset.bodd  -- bring in the definition to clarify odd checking step
  sorry

end sum_of_fractions_approx_l778_778932


namespace sum_even_coefficients_eq_l778_778728

theorem sum_even_coefficients_eq :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ),
    (∀ x : ℝ, x^10 = a + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + 
                     a_4 * (1 - x)^4 + a_5 * (1 - x)^5 + a_6 * (1 - x)^6 + 
                     a_7 * (1 - x)^7 + a_8 * (1 - x)^8 + a_9 * (1 - x)^9 + 
                     a_{10} * (1 - x)^10) →
    a + a_2 + a_4 + a_6 + a_8 + a_{10} = 2^9 :=
by
  intro a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10},
  intro h,
  sorry

end sum_even_coefficients_eq_l778_778728


namespace ellipse_equation_and_fixed_point_proof_l778_778287

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778287


namespace ellipse_equation_and_fixed_point_proof_l778_778283

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778283


namespace sum_of_decimals_l778_778612

theorem sum_of_decimals : (5.47 + 4.96) = 10.43 :=
by
  sorry

end sum_of_decimals_l778_778612


namespace same_color_same_suit_probability_l778_778977

/-- A standard deck consists of 52 cards, made up of 13 ranks across 4 suits. -/
constant ranks : ℕ := 13
constant suits : ℕ := 4
constant cards : ℕ := 52

/-- Each suit has 13 cards. The suits are spades, hearts, diamonds, and clubs. -/
constant spades : ℕ := 13
constant hearts : ℕ := 13
constant diamonds : ℕ := 13
constant clubs : ℕ := 13

/-- Spades and clubs are black, while hearts and diamonds are red. -/
constant black_suits : Set String := { "spades", "clubs" }
constant red_suits : Set String := { "hearts", "diamonds" }

/-- The deck is shuffled randomly. -/
axiom shuffled_deck : ∀ (deck : List String), deck.length = cards → Random.shuffle deck = deck

/-- The probability of drawing three cards of the same color and same suit from a shuffled deck. -/
theorem same_color_same_suit_probability : 
  (∃ (deck : List String) (h_deck : deck.length = cards), 
    ∀ x ∈ deck, (x ∈ black_suits ∨ x ∈ red_suits)) →
  (prob_top_three_same_color_suit : ℚ) = 22 / 425 :=
  by
  sorry

end same_color_same_suit_probability_l778_778977


namespace return_speed_is_33_33_l778_778124

noncomputable def return_speed (d: ℝ) (speed_to_b: ℝ) (avg_speed: ℝ): ℝ :=
  d / (3 + (d / avg_speed))

-- Conditions
def distance := 150
def speed_to_b := 50
def avg_speed := 40

-- Prove that the return speed is 33.33 miles per hour
theorem return_speed_is_33_33:
  return_speed distance speed_to_b avg_speed = 33.33 :=
by
  unfold return_speed
  sorry

end return_speed_is_33_33_l778_778124


namespace find_lower_limit_l778_778051

-- Define the nature of non-prime numbers under 30
def is_non_prime (n : ℕ) : Prop :=
  ¬ (prime n) ∧ 2 ≤ n ∧ n < 30

-- Sum all non-prime numbers between l and 30, non-inclusive
def sum_non_primes (l : ℕ) : ℕ :=
  (Finset.range 30).filter (λ n, is_non_prime n ∧ l ≤ n).sum id

theorem find_lower_limit (l : ℕ) (h : sum_non_primes l = 173) : l = 5 := by
  sorry

end find_lower_limit_l778_778051


namespace values_of_n_l778_778673

theorem values_of_n : ∃ n_vals : Finset ℤ, n_vals.card = 9 ∧ ∀ n ∈ n_vals, (4000 * ((2:ℝ/ℚ/ℤ) / 5)^n : ℝ/ℚ/ℤ) ∈ ℤ :=
by 
  sorry

end values_of_n_l778_778673


namespace bela_wins_if_both_play_optimally_l778_778993

theorem bela_wins_if_both_play_optimally : ∀ (numbers : list ℝ),
  (∀ (x : ℝ), x ∈ numbers → 0 ≤ x ∧ x ≤ 10) ∧
  (∀ (x y : ℝ), x ∈ numbers ∧ y ∈ numbers ∧ x ≠ y → abs (x - y) > 2) ∧
  ((length numbers % 2 = 0 → last numbers = bela) → 
   (∃ (x : ℝ), (x ∉ numbers) ∧ (0 ≤ x ∧ x ≤ 10) ∧ (∀ y ∈ numbers, abs (x - y) > 2))) := sorry

end bela_wins_if_both_play_optimally_l778_778993


namespace total_snacks_l778_778781

variable (peanuts : ℝ) (raisins : ℝ)

theorem total_snacks (h1 : peanuts = 0.1) (h2 : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end total_snacks_l778_778781


namespace tan_theta_eq_cos_double_theta_pi_div_3_eq_l778_778699

noncomputable theory

open real

-- Given conditions
variables (θ : ℝ) (h1 : sin θ = 3 / 5) (h2 : π / 2 < θ ∧ θ < π)

-- Problem statements
theorem tan_theta_eq : tan θ = 3 / 4 :=
sorry

theorem cos_double_theta_pi_div_3_eq : cos (2 * θ - π / 3) = (7 - 24 * real.sqrt 3) / 50 :=
sorry

end tan_theta_eq_cos_double_theta_pi_div_3_eq_l778_778699


namespace limit_P_n_eq_e_l778_778414

noncomputable def a : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * (a n + 1)

noncomputable def P : ℕ → ℝ
| 0       := 1 + 1 / a 0
| (n + 1) := P n * (1 + 1 / a (n + 1))

theorem limit_P_n_eq_e (P_n : ℕ → ℝ) (hP : ∀ n, P_n n = (list.fin_range (n + 1)).prod (λ i, 1 + 1 / a i)) :
  tendsto P_n at_top (𝓝 real.exp) :=
by sorry

end limit_P_n_eq_e_l778_778414


namespace statement_A_statement_B_statement_C_statement_D_l778_778094

-- Statement A: If A ∩ B = A, then A ⊆ B
theorem statement_A (A B : Set ℕ) : A ∩ B = A → A ⊆ B := sorry

-- Statement B: If x ∈ A ∩ B, then x ∈ (A ∪ B)
theorem statement_B (A B : Set ℕ) (x : ℕ) : x ∈ A ∩ B → x ∈ A ∪ B := sorry

-- Statement C: Given a, b ∈ ℝ, the condition b / a < a / b is necessary but not sufficient for a < b < 0
theorem statement_C (a b : ℝ) : (b / a < a / b) → ¬ (∀ (a < b ∧ b < 0), (b / a < a / b)) := sorry

-- Statement D: The proposition ∀ x ∈ ℤ, x² > 0 is false
theorem statement_D : ¬ (∀ x : ℤ, x^2 > 0) := sorry

end statement_A_statement_B_statement_C_statement_D_l778_778094


namespace HN_passes_through_fixed_point_l778_778252

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778252


namespace zoo_children_tuesday_l778_778002

theorem zoo_children_tuesday 
  (x : ℕ) 
  (child_ticket_cost adult_ticket_cost : ℕ) 
  (children_monday adults_monday adults_tuesday : ℕ)
  (total_revenue : ℕ) : 
  child_ticket_cost = 3 → 
  adult_ticket_cost = 4 → 
  children_monday = 7 → 
  adults_monday = 5 → 
  adults_tuesday = 2 → 
  total_revenue = 61 → 
  7 * 3 + 5 * 4 + x * 3 + 2 * 4 = total_revenue → 
  x = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zoo_children_tuesday_l778_778002


namespace age_of_b_is_6_l778_778488

theorem age_of_b_is_6 (x : ℕ) (h1 : 5 * x / 3 * x = 5 / 3)
                         (h2 : (5 * x + 2) / (3 * x + 2) = 3 / 2) : 3 * x = 6 := 
by
  sorry

end age_of_b_is_6_l778_778488


namespace average_diesel_rate_l778_778640

theorem average_diesel_rate (r1 r2 r3 r4 : ℝ) (H1: (r1 + r2 + r3 + r4) / 4 = 1.52) :
    ((r1 + r2 + r3 + r4) / 4 = 1.52) :=
by
  exact H1

end average_diesel_rate_l778_778640


namespace number_of_possible_arrays_l778_778551

-- Definitions of conditions
def chairs : ℕ := 48
def min_chairs_per_row : ℕ := 2
def min_chairs_per_column : ℕ := 2

-- Theorem statement
theorem number_of_possible_arrays
  (h_chairs : ∃ r c : ℕ, r * c = chairs ∧ r ≥ min_chairs_per_row ∧ c ≥ min_chairs_per_column) :
  (number of pairs (r, c) such that (r * c = chairs) and r ≥ 2 and c ≥ 2) = 8 :=
sorry

end number_of_possible_arrays_l778_778551


namespace total_cost_l778_778410

-- Define the conditions
def dozen := 12
def cost_of_dozen_cupcakes := 10
def cost_of_dozen_cookies := 8
def cost_of_dozen_brownies := 12

def num_dozen_cupcakes := 4
def num_dozen_cookies := 3
def num_dozen_brownies := 2

-- Define the total cost for each type of treat
def total_cost_cupcakes := num_dozen_cupcakes * cost_of_dozen_cupcakes
def total_cost_cookies := num_dozen_cookies * cost_of_dozen_cookies
def total_cost_brownies := num_dozen_brownies * cost_of_dozen_brownies

-- The theorem to prove the total cost
theorem total_cost : total_cost_cupcakes + total_cost_cookies + total_cost_brownies = 88 := by
  -- Here would go the proof, but it's omitted as per the instructions
  sorry

end total_cost_l778_778410


namespace servings_in_package_l778_778139

-- Define the conditions of the problem
def package (servings : ℕ) : Prop :=
  ∃ (calories_per_serving total_calories_half : ℕ),
  calories_per_serving = 120 ∧ total_calories_half = 180 ∧ servings * calories_per_serving = 2 * total_calories_half

-- State the theorem we want to prove
theorem servings_in_package : package 3 :=
by
  unfold package
  use [120, 180]
  simp
  sorry

end servings_in_package_l778_778139


namespace problem_l778_778678

variable (α : ℝ)

def setA : Set ℝ := {Real.sin α, Real.cos α, 1}
def setB : Set ℝ := {Real.sin α ^ 2, Real.sin α + Real.cos α, 0}
theorem problem (h : setA α = setB α) : Real.sin α ^ 2009 + Real.cos α ^ 2009 = -1 := 
by 
  sorry

end problem_l778_778678


namespace centers_and_common_point_on_new_circle_l778_778504

variables {A B C D O₁ O₂ O₃ : Type}
variables [point A] [center O₁] [center O₂] [center O₃]
variables [circle (O₁, A)] [circle (O₂, A)] [circle (O₃, A)]
variables [collinear B C D]

theorem centers_and_common_point_on_new_circle :
  ∃ (α : Type) (O₁ O₂ O₃ A : α), is_circle α O₁ O₂ O₃ A :=
sorry

end centers_and_common_point_on_new_circle_l778_778504


namespace find_value_of_expression_l778_778211

theorem find_value_of_expression (θ : Float) (hθ : θ = 15) : 
  (7 / 16 - 7 / 8 * (sin (θ * Float.pi / 180)) ^ 2) = 7 * Real.sqrt 3 / 32 := 
by
  sorry

end find_value_of_expression_l778_778211


namespace hyperbola_eccentricity_l778_778233

-- Definitions for the problem conditions
variable (a b : ℝ) -- lengths related to hyperbola (a and b)
variable (e : ℝ) -- eccentricity
variable (c := Real.sqrt (a^2 + b^2)) -- relationship between foci distance and a, b

-- The condition from the problem for the hyperbola asymptotes
variable (asymptote_condition : (b / a = 3 / 4) ∨ (a / b = 3 / 4))

-- The definition of eccentricity based on a and c
def eccentricity := c / a

-- The main theorem to prove the eccentricity values
theorem hyperbola_eccentricity : 
  asymptote_condition → (eccentricity = 5/4 ∨ eccentricity = 5/3) :=
by
  sorry

end hyperbola_eccentricity_l778_778233


namespace dice_sum_ways_l778_778521

theorem dice_sum_ways : 
  let num_dice : ℕ := 8
  let target_sum : ℕ := 20
  let dice_max_value : ℕ := 6
  let adjusted_sum : ℕ := target_sum - num_dice
  ∑ (ways : ℕ) in (Finset.filter (fun (a : Fin num_dice → ℕ) => 
    (∑ i, a i = adjusted_sum) ∧ (∀ i, 0 ≤ a i ∧ a i ≤ dice_max_value - 1)) 
    (Finset.mk (Finsupp.support := num_dice) 1 adjusted_sum)), 
  ways = 50388 := 
by 
  sorry

end dice_sum_ways_l778_778521


namespace parabola_intercept_sum_l778_778849

theorem parabola_intercept_sum (a b c : ℝ) : 
  (∃ y : ℝ, a = 3 * y^2 - 9 * y + 5) ∧ (∀ x : ℝ, x = 0 → b ≠ c → 3 * b^2 - 9 * b + 5 = 0 ∧ 3 * c^2 - 9 * c + 5 = 0 ∧ b + c = 3) → 
  a + b + c = 8 :=
begin
  sorry
end

end parabola_intercept_sum_l778_778849


namespace problem_solution_l778_778358

theorem problem_solution (a : ℝ) : 
  ( ∀ x : ℝ, (ax - 1) * (x + 1) < 0 ↔ (x ∈ Set.Iio (-1) ∨ x ∈ Set.Ioi (-1 / 2)) ) →
  a = -2 :=
by
  sorry

end problem_solution_l778_778358


namespace ellipse_equation_and_fixed_point_proof_l778_778281

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l778_778281


namespace find_other_root_l778_778812

theorem find_other_root 
  (m : ℚ) 
  (h : 3 * 3^2 + m * 3 - 5 = 0) :
  (1 - 3) * (x : ℚ) = 0 :=
sorry

end find_other_root_l778_778812


namespace q_completes_work_in_4_days_l778_778103

-- Definitions of the conditions
def work_rate (total_work : ℝ) (days : ℝ) : ℝ := total_work / days

def p_and_q_work_rate (W : ℝ) : ℝ := work_rate W 20

def work_done (rate : ℝ) (days : ℝ) : ℝ := rate * days

def remaining_work (total_work : ℝ) (done_work : ℝ) : ℝ := total_work - done_work

-- The main theorem including conditions and the target proof
theorem q_completes_work_in_4_days (W : ℝ) (p_Work_in_4_days : ℝ) (remaining_Work : ℝ) 
(p_and_q_Work_in_6_days : ℝ) : 
  p_and_q_work_rate W = p_and_q_Work_in_6_days / 6 → 
  work_done (p_and_q_work_rate W) 6 = remaining_Work → 
  remaining_work W p_Work_in_4_days = remaining_Work → 
  work_done (work_rate W 4) 4 = p_Work_in_4_days → 
  (∃ x : ℝ, (work_rate W 20) + (W / x) = work_rate W 4) := 
begin 
  sorry
end

end q_completes_work_in_4_days_l778_778103


namespace count_polynomials_satisfying_condition_l778_778186

def degree_at_most_3_polynomial_with_coeffs_in_range (Q : ℚ[X]) : Prop :=
  (Q.degree ≤ 3) ∧ (∀ n, Q.coeff n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

theorem count_polynomials_satisfying_condition :
  (set_of (λ Q : ℚ[X], degree_at_most_3_polynomial_with_coeffs_in_range Q ∧ Q.eval (-1) = -15)).finite.card = 816 :=
by sorry

end count_polynomials_satisfying_condition_l778_778186


namespace total_ideal_matching_sets_l778_778419

-- Definitions based on the provided problem statement
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def is_ideal_matching_set (A B : Set ℕ) : Prop := A ∩ B = {1, 3, 5}

-- Theorem statement for the total number of ideal matching sets
theorem total_ideal_matching_sets : ∃ n, n = 27 ∧ ∀ (A B : Set ℕ), A ⊆ U ∧ B ⊆ U ∧ is_ideal_matching_set A B → n = 27 := 
sorry

end total_ideal_matching_sets_l778_778419


namespace series_sum_eq_l778_778796

theorem series_sum_eq :
  ∀ (a b : ℝ), 0 < b ∧ 0 < a ∧ a > 3 * b →
  (∑' n : ℕ, if h : n ≥ 1 then 1 / (((3 * ↑n - 6) * a - ((↑n ^ 2 - 5 * ↑n + 6) * b)) * ((3 * ↑n - 3) * a - ((↑n ^ 2 - 4 * ↑n + 3) * b))) else 0) =
  1 / (b * (a - b)) :=
by 
  intros a b conditions,
  sorry

end series_sum_eq_l778_778796


namespace ellipse_solution_geometry_solution_l778_778313

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778313


namespace triangle_third_side_length_l778_778006

theorem triangle_third_side_length
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b = 10)
  (h2 : c = 7)
  (h3 : A = 2 * B) :
  a = (50 + 5 * Real.sqrt 2) / 7 ∨ a = (50 - 5 * Real.sqrt 2) / 7 :=
sorry

end triangle_third_side_length_l778_778006


namespace vector_c_in_terms_of_a_and_b_l778_778363

noncomputable def vector_a := (3, -2)
noncomputable def vector_b := (-2, 1)
noncomputable def vector_c := (7, -4)

theorem vector_c_in_terms_of_a_and_b : 
  (7, -4) = (3, -2) + (-2) • (-2, 1) :=
by
  -- Place the detailed proof here
  sorry

end vector_c_in_terms_of_a_and_b_l778_778363


namespace train_route_length_l778_778508

theorem train_route_length (t1 t2 d1 d2 : ℝ) (D : ℝ)
  (hA : t1 = 3) 
  (hB : t2 = 2) 
  (hTotalTravel : d1 = 30)
  (hSpeedA : D / 3) 
  (hSpeedB : D / 2)
  (hMeeting : d1 + hSpeedB * (d1 / (D / 3)) = D) : 
  D = 75 :=
sorry

end train_route_length_l778_778508


namespace trapezoid_area_l778_778511

theorem trapezoid_area (x : ℝ) (y : ℝ) :
  (∀ x, y = x + 1) →
  (∀ y, y = 12) →
  (∀ y, y = 7) →
  (∀ x, x = 0) →
  ∃ area,
  area = (1/2) * (6 + 11) * 5 ∧ area = 42.5 :=
by {
  sorry
}

end trapezoid_area_l778_778511


namespace area_per_tree_l778_778750

theorem area_per_tree :
  (∃ (area per square_meter per_tree : ℕ),
   area = 10_000 ∧
   per_tree = 40 ∧
   area / 250 = per_tree ∧
   250 = 100 / 0.4) :=
sorry

end area_per_tree_l778_778750


namespace total_parents_in_auditorium_l778_778496

def num_girls : ℕ := 6
def num_boys : ℕ := 8
def parents_per_child : ℕ := 2

theorem total_parents_in_auditorium (num_girls num_boys parents_per_child : ℕ) : num_girls + num_boys = 14 → 2 * (num_girls + num_boys) = 28 := by
  assume h: num_girls + num_boys = 14
  show 2 * (num_girls + num_boys) = 28, from
    calc
      2 * (num_girls + num_boys) = 2 * 14 : by rw h
      ... = 28 : by norm_num

end total_parents_in_auditorium_l778_778496


namespace total_parents_in_auditorium_l778_778498

-- Define the conditions.
def girls : Nat := 6
def boys : Nat := 8
def total_kids : Nat := girls + boys
def parents_per_kid : Nat := 2
def total_parents : Nat := total_kids * parents_per_kid

-- The statement to prove.
theorem total_parents_in_auditorium : total_parents = 28 := by
  sorry

end total_parents_in_auditorium_l778_778498


namespace log7_18_l778_778733

open Real

noncomputable def a : ℝ := log 2 / log 10
noncomputable def b : ℝ := log 3 / log 10

theorem log7_18 (a b : ℝ) (h1 : log 10 2 = a) (h2 : log 10 3 = b) : log 7 18 = (a + 2 * b) / log 10 7 :=
by
  sorry

end log7_18_l778_778733


namespace min_value_of_m_l778_778694

/-- Given curves C₁: y = e^x and C₂: y = 1 + ln(x - m) where m > 0,
  and points A(x₁, y₁) on C₁ and B(x₂, y₂) on C₂ such that y₁ = y₂.
  For any x₁, x₂, it always holds that |AB| ≥ e.
  Prove that the minimum value of m is e - 1. -/
theorem min_value_of_m {x₁ x₂ m : ℝ} (h_gt_0 : m > 0) (h_eq : exp x₁ = 1 + log (x₂ - m))
  (h_dist : x₂ - x₁ ≥ e) : m ≥ exp 1 - 1 := sorry

end min_value_of_m_l778_778694


namespace compute_expression_l778_778792

-- Define the conditions
variables (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0)

-- State the theorem to be proved
theorem compute_expression (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := 
sorry

end compute_expression_l778_778792


namespace probability_that_both_girls_select_same_colored_marble_l778_778530

noncomputable def probability_girls_same_marble : ℚ :=
  let total_marbles := 2 + 2 in
  let white_marbles := 2 in
  let black_marbles := 2 in
  let first_girl_white := white_marbles / total_marbles in
  let second_girl_white := (white_marbles - 1) / (total_marbles - 1) in
  let first_girl_black := black_marbles / total_marbles in
  let second_girl_black := (black_marbles - 1) / (total_marbles - 1) in
  (first_girl_white * second_girl_white) + 
  (first_girl_black * second_girl_black)

theorem probability_that_both_girls_select_same_colored_marble :
  probability_girls_same_marble = 1 / 3 := 
  sorry

end probability_that_both_girls_select_same_colored_marble_l778_778530


namespace total_parents_in_auditorium_l778_778499

-- Define the conditions.
def girls : Nat := 6
def boys : Nat := 8
def total_kids : Nat := girls + boys
def parents_per_kid : Nat := 2
def total_parents : Nat := total_kids * parents_per_kid

-- The statement to prove.
theorem total_parents_in_auditorium : total_parents = 28 := by
  sorry

end total_parents_in_auditorium_l778_778499


namespace sum_of_reciprocals_of_roots_l778_778797

theorem sum_of_reciprocals_of_roots (a b c : ℝ) 
  (h_roots : (Polynomial.X^3 - 2 * Polynomial.X^2 + Polynomial.X - 1).roots = {a, b, c}) :
  (1 / (a - 2) + 1 / (b - 2) + 1 / (c - 2)) = -5 :=
sorry

end sum_of_reciprocals_of_roots_l778_778797


namespace evaluate_g_expression_l778_778189

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g_expression :
  3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_expression_l778_778189


namespace real_root_eq_l778_778210

noncomputable def root_equation (x : ℝ) : Prop :=
  sqrt (x + 9) - sqrt (x - 2) = 3

theorem real_root_eq :
  ∃ x : ℝ, x + 9 ≥ 0 ∧ x - 2 ≥ 0 ∧ root_equation x := by
  use 19/9
  sorry

end real_root_eq_l778_778210


namespace find_x_from_equation_l778_778372

/-- If (1 / 8) * 2^36 = 4^x, then x = 16.5 -/
theorem find_x_from_equation (x : ℝ) (h : (1/8) * (2:ℝ)^36 = (4:ℝ)^x) : x = 16.5 :=
by sorry

end find_x_from_equation_l778_778372


namespace simplify_complex_subtraction_l778_778820

noncomputable def a : ℂ := -5 - 3 * Complex.i
noncomputable def b : ℂ := 2 - 5 * Complex.i

theorem simplify_complex_subtraction : (a - b) = -7 + 2 * Complex.i := by
  sorry

end simplify_complex_subtraction_l778_778820


namespace problem_correct_options_l778_778793

open ProbabilityTheory

variables {Ω : Type*} [MeasureSpace Ω]
variables {A B : Set Ω} (P : Measure Ω)

theorem problem_correct_options
  (hA : 0 < P A) (hA1 : P A < 1)
  (hB : 0 < P B) (hB1 : P B < 1):
  P[B | A] + P[complement B | A] = 1 :=
begin
  sorry
end

end problem_correct_options_l778_778793


namespace quadratic_polynomial_real_coeff_l778_778665

theorem quadratic_polynomial_real_coeff (a b : ℂ) (h₁ : a = 5 + 2*i) 
  (h₂ : b = 5 - 2*i) (c : ℂ) (hc : c = 3) :
  3 * (X - C a) * (X - C b) = 3*X^2 - 30*X + 87 := 
by {
  sorry
}

end quadratic_polynomial_real_coeff_l778_778665


namespace prob_three_digit_divisible_by_5_l778_778973

theorem prob_three_digit_divisible_by_5 :
  let S := {n | 100 ≤ n ∧ n ≤ 999}
  let total := 900
  let count_div_5 := 180
  in (count_div_5 : ℚ) / (total : ℚ) = (1 / 5 : ℚ) :=
by
  sorry

end prob_three_digit_divisible_by_5_l778_778973


namespace proof_problem_l778_778687

-- Conditions
def point_on_circle (P : ℝ × ℝ) : Prop := P.1 ^ 2 + P.2 ^ 2 = 1
def foot_of_perpendicular (P Q : ℝ × ℝ) : Prop := Q.1 = 0 ∧ Q.2 = P.2
def proportional_vectors (P Q R : ℝ × ℝ) : Prop := Q.1 = P.1 * sqrt(3) ∧ Q.2 = P.2

-- Curve C' equation
def trajectory_eq (R : ℝ × ℝ) : Prop := R.1 ^ 2 / 3 + R.2 ^ 2 = 1

-- Area of triangle
def slopes_product (A M N : ℝ × ℝ) : Prop := 
  ((M.2 - A.2) / (M.1 - A.1)) * ((N.2 - A.2) / (N.1 - A.1)) = 2 / 3

def area_triangle (A M N : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * (M.2 - N.2) + M.1 * (N.2 - A.2) + N.1 * (A.2 - M.2))

-- Problem statement
theorem proof_problem 
  (P Q R A M N : ℝ × ℝ) 
  (hP : point_on_circle P) 
  (hQ : foot_of_perpendicular P Q) 
  (hR : proportional_vectors P Q R) 
  (hC : trajectory_eq R)
  (hM : trajectory_eq M) 
  (hN : trajectory_eq N) 
  (hSlopes : slopes_product A M N) :
  (
    let A := (0, 1) in 
    trajectory_eq R ∧ area_triangle A M N ≤ ((2 * sqrt 3) / 3)
  ) :=
sorry

end proof_problem_l778_778687


namespace probability_two_girls_one_is_alice_l778_778591

theorem probability_two_girls_one_is_alice
  (total_members : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (alice_is_girl : Prop)
  (total := total_members = 12)
  (boys_count := boys = 7)
  (girls_count := girls = 5)
  (alice_in_girls := alice_is_girl = true)
  (choose_two : ℕ := 12.choose 2)
  (favorable_outcomes : ℕ := 4)
  (prob : ℚ := favorable_outcomes / choose_two) :
  prob = 2 / 33 := by
  -- Proving given conditions
  have total := total, by simp[total],
  have boys_count := boys_count, by simp[boys_count],
  have girls_count := girls_count, by simp[girls_count],
  have alice_in_girls := alice_in_girls,
  have choose_two := choose_two, by simp [choose_two],
  have favorable_outcomes := favorable_outcomes, by simp[favorable_outcomes],
  have prob := prob, by simp[prob, favorable_outcomes, choose_two],
  -- Main proof
  sorry

end probability_two_girls_one_is_alice_l778_778591


namespace length_of_EK_l778_778466

theorem length_of_EK (a : ℝ) (h : a = 4)
  (E F G H K L : Point) (EK EL : Segment)
  (h1 : EK.length = EL.length)
  (h2 : Square E F G H a)
  (h3 : divides_into_two_equal_parts EK EL) : 
  EK.length = 4 * Real.sqrt 2 :=
sorry

end length_of_EK_l778_778466


namespace triangle_is_isosceles_l778_778408

theorem triangle_is_isosceles 
  (A B C : ℝ) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  ∃ A B C, A = B :=
sorry

end triangle_is_isosceles_l778_778408


namespace probability_of_sum_5_when_two_dice_are_rolled_l778_778093

/--
When two fair dice are rolled, the probability that the sum of the numbers facing up is 5 is 1/9.
-/
theorem probability_of_sum_5_when_two_dice_are_rolled : 
  let outcomes := [(1,4), (2,3), (3,2), (4,1)] in
  (list.length outcomes : ℚ) / 36 = 1 / 9 := 
by
  sorry

end probability_of_sum_5_when_two_dice_are_rolled_l778_778093


namespace total_income_200_l778_778141

theorem total_income_200 (I : ℝ) 
  (H1 : I / 16)
  (H2 : I * 15 / 16)
  (H3 : I / 16)
  (H4 : I * 14 / 16)
  (H5 : I * 7 / 8)
  (H6 : (I * 5 / 8).floor = 50) :
  I = 200 :=
  sorry

end total_income_200_l778_778141


namespace annual_income_is_32000_l778_778870

noncomputable def compute_tax (p A: ℝ) : ℝ := 
  0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000)

noncomputable def stated_tax (p A: ℝ) : ℝ := 
  0.01 * (p + 0.25) * A

theorem annual_income_is_32000 (p : ℝ) (A : ℝ) :
  compute_tax p A = stated_tax p A → A = 32000 :=
by
  intros h
  have : 0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = 0.01 * (p + 0.25) * A := h
  sorry

end annual_income_is_32000_l778_778870


namespace jerry_tax_rate_l778_778778

/-- Jerry went to a shop and bought things worth Rs. 20, out of which 30% went on sales tax on 
    taxable purchases. If the cost of the tax-free items was Rs. 14.7, the tax rate was 30%.
-/
theorem jerry_tax_rate (total_cost : ℝ) (tax_free_cost : ℝ) (tax_rate_percentage : ℝ) :
  total_cost = 20 → tax_free_cost = 14.7 → tax_rate_percentage = 30 →
  let taxable_cost := total_cost - tax_free_cost in
  let sales_tax := 0.3 * taxable_cost in
  (sales_tax / taxable_cost) * 100 = tax_rate_percentage :=
by
  intros h1 h2 h3
  dsimp
  rw [h1, h2]
  norm_num
  sorry

end jerry_tax_rate_l778_778778


namespace parallel_lines_incircle_excircles_l778_778437

-- Definitions of points and lines as stated
variables {A B C A' B' C' B* C* A'' B'' C'' B** C** A''' B''' C''' B*** C*** : Point}
variables {O O_a O_c : Point}

-- Hypothesis statements
hypothesis touching_points_inc : touching_point_incircle A' B' C' A B C
hypothesis angle_bisectors_inc: angle_bisectors_intersections A B C A' B' B* C*
hypothesis angle_bisectors_excircle_a: angle_bisectors_intersections_excircle_A O_a A' A'' B'' C'' B** C**
hypothesis angle_bisectors_excircle_c: angle_bisectors_intersections_excircle_BC O_c A''' B''' C''' B*** C***

-- Statement of the theorem
theorem parallel_lines_incircle_excircles 
  (h1 : Are_parallel B'C' BB*)
  (h2 : Are_parallel B'C' CC*)
  (h3 : Are_parallel B''C'' BB**)
  (h4 : Are_parallel B''C'' CC**)
  (h5 : Are_parallel B'''C''' BB***)
  (h6 : Are_parallel B'''C''' CC***) :
  B'C' = BB* ∧ B'C' = CC* ∧ B''C'' = BB** ∧ B''C'' = CC** ∧ B'''C''' = BB*** ∧ B'''C''' = CC*** :=
    sorry

end parallel_lines_incircle_excircles_l778_778437


namespace prime_divisor_of_sum_of_four_consecutive_integers_l778_778916

theorem prime_divisor_of_sum_of_four_consecutive_integers : 
  ∃ p : ℕ, Prime p ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) ∧ p = 2 := 
by
  sorry

end prime_divisor_of_sum_of_four_consecutive_integers_l778_778916


namespace mixed_groups_count_l778_778902

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l778_778902


namespace area_of_figure_M_l778_778763

theorem area_of_figure_M : 
  let ineq1 (x y : ℝ) := x + y + |x - y| ≤ 0,
      ineq2 (x y : ℝ) := (x^2 + 6*x + y^2 - 8*y) / (x + 3*y + 6) ≥ 0,
      M := {p : ℝ × ℝ | ineq1 p.1 p.2 ∧ ineq2 p.1 p.2} in
  ∃ (area : ℝ), area = 3 ∧ (
    ∀ q ∈ M, is_point_in_region_q q M)
:= sorry

end area_of_figure_M_l778_778763


namespace percentage_increase_cube_surface_area_l778_778108

theorem percentage_increase_cube_surface_area (L : ℝ) (h : L > 0) :
  let 
    SA_original := 6 * L^2,
    L_new := 1.10 * L,
    SA_new := 6 * (L_new^2),
    percentage_increase := ((SA_new - SA_original) / SA_original) * 100
  in percentage_increase = 21 :=
by
  sorry

end percentage_increase_cube_surface_area_l778_778108


namespace greatest_integer_less_than_PS_l778_778395

noncomputable def PS := (150 * Real.sqrt 2)

theorem greatest_integer_less_than_PS
  (PQ RS : ℝ)
  (PS : ℝ := PQ * Real.sqrt 2)
  (h₁ : PQ = 150)
  (h_midpoint : PS / 2 = PQ) :
  ∀ n : ℤ, n < PS → n = 212 :=
by
  -- Proof to be completed later
  sorry

end greatest_integer_less_than_PS_l778_778395


namespace minimum_value_w_l778_778515

noncomputable def w (x y : ℝ) : ℝ := 3 * x^2 + 4 * y^2 - 12 * x + 8 * y + 15

theorem minimum_value_w : ∀ x y : ℝ, ∃ (min_w : ℝ), min_w = -1 :=
by
  let x := 2
  let y := -1
  use w x y
  unfold w
  simp
  sorry

end minimum_value_w_l778_778515


namespace number_of_digits_base10_l778_778377

noncomputable def log_base (b x : ℝ) : ℝ :=
  real.log x / real.log b

theorem number_of_digits_base10 (x : ℝ) (hx : log_base 2 (log_base 2 (log_base 2 x)) = 2) :
  let num_digits := nat.floor (log_base 10 x) + 1 in
  num_digits = 5 :=
by
  sorry

end number_of_digits_base10_l778_778377


namespace max_has_two_nickels_l778_778473

theorem max_has_two_nickels (n : ℕ) (nickels : ℕ) (coins_value_total : ℕ) :
  (coins_value_total = 15 * n) -> (coins_value_total + 10 = 16 * (n + 1)) -> 
  coins_value_total - nickels * 5 + nickels + 25 = 90 -> 
  n = 6 -> 
  2 = nickels := 
by 
  sorry

end max_has_two_nickels_l778_778473


namespace platform_length_l778_778153

theorem platform_length (L : ℕ) : (train_length speed_pole speed_platform : ℕ)
  (train_length = 120)
  (speed_pole = 11)
  (speed_platform = 22) :
  L = 120 :=
by
  sorry

end platform_length_l778_778153


namespace range_of_a_l778_778543

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := 
by 
  sorry

end range_of_a_l778_778543


namespace sqrt_cuberoot_eq_l778_778610

theorem sqrt_cuberoot_eq (x : ℝ) (h : x = 0.000064) : Real.sqrt (Real.cbrt x) = 0.2 := 
by sorry

end sqrt_cuberoot_eq_l778_778610


namespace correct_propositions_l778_778987

/--
Two lines perpendicular to the same line are parallel.
-/
def Prop1 : Prop := ∀ (l1 l2 l : ℝ^3), (l1 ⊥ l) → (l2 ⊥ l) → (l1 ∥ l2)

/--
Two planes perpendicular to the same line are parallel.
-/
def Prop2 : Prop := ∀ (p1 p2 l : ℝ^3), (p1 ⊥ l) → (p2 ⊥ l) → (p1 ∥ p2)

/--
Two lines perpendicular to the same plane are parallel.
-/
def Prop3 : Prop := ∀ (l1 l2 p : ℝ^3), (l1 ⊥ p) → (l2 ⊥ p) → (l1 ∥ l2)

/--
Two planes perpendicular to the same plane are parallel.
-/
def Prop4 : Prop := ∀ (p1 p2 p : ℝ^3), (p1 ⊥ p) → (p2 ⊥ p) → (p1 ∥ p2)

/--
We need to prove that only Prop2 and Prop3 are true.
-/
theorem correct_propositions : (Prop1 = false) ∧ (Prop2 = true) ∧ (Prop3 = true) ∧ (Prop4 = false) :=
by
  sorry

end correct_propositions_l778_778987


namespace digit_makes_num_divisible_l778_778561

theorem digit_makes_num_divisible (d : ℕ) (h_digit: d = 4) : 
  (626840 % 8 = 0) ∧ (626840 % 5 = 0) :=
by {
  have h8 : 626840 % 8 = 0 := by sorry,
  have h5 : 626840 % 5 = 0 := by sorry,
  exact ⟨h8, h5⟩
}

end digit_makes_num_divisible_l778_778561


namespace HN_passes_through_fixed_point_l778_778255

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l778_778255


namespace cost_of_agricultural_equipment_max_units_of_type_A_l778_778113

-- Define cost equations
variables (x y : ℝ)

-- Define conditions as hypotheses
def condition1 : Prop := 2 * x + y = 4.2
def condition2 : Prop := x + 3 * y = 5.1

-- Prove the costs are respectively 1.5 and 1.2
theorem cost_of_agricultural_equipment (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 1.5 ∧ y = 1.2 := sorry

-- Define the maximum units constraint
def total_cost (m : ℕ) : ℝ := 1.5 * m + 1.2 * (2 * m - 3)

-- Prove the maximum units of type A is 3
theorem max_units_of_type_A (m : ℕ) (h : total_cost m ≤ 10) : m ≤ 3 := sorry

end cost_of_agricultural_equipment_max_units_of_type_A_l778_778113


namespace rectangle_area_WXYZ_l778_778077

theorem rectangle_area_WXYZ 
  (length_sheet : ℝ) (width_sheet : ℝ) (overlap_length : ℝ) 
  (h1 : length_sheet = 11) (h2 : width_sheet = 8) (h3 : overlap_length = 8) :
  ∃ (area_WXYZ : ℝ), area_WXYZ = 112 := by
  have h4 : length_sheet - overlap_length = 3, from by linarith [h1, h3],
  have total_length_WX : ℝ := 3 + 11, 
  have total_length_WX = 14, from by linarith [h4], 
  let area_WXYZ := 14 * 8,
  use area_WXYZ,
  show area_WXYZ = 112, from by linarith -- showing that computed area equals 112
  sorry

end rectangle_area_WXYZ_l778_778077


namespace machine_X_produces_18_widgets_per_hour_l778_778947

variable (Wx Wy Tx Ty : ℝ)

-- Definitions of conditions
def machineY_produces_more_widgets_per_hour : Prop :=
  Wy = 1.20 * Wx

def machineX_time_to_produce_1080_widgets : Prop :=
  1080 = Wx * Tx

def machineY_time_to_produce_1080_widgets : Prop :=
  1080 = Wy * Ty

def machineX_takes_10_hours_more : Prop :=
  Tx = Ty + 10

-- Theorem to state and prove the result
theorem machine_X_produces_18_widgets_per_hour
    (h1 : machineY_produces_more_widgets_per_hour)
    (h2 : machineX_time_to_produce_1080_widgets)
    (h3 : machineY_time_to_produce_1080_widgets)
    (h4 : machineX_takes_10_hours_more) :
    Wx = 18 :=
by
  sorry

end machine_X_produces_18_widgets_per_hour_l778_778947


namespace integral_bounds_l778_778643

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^3)

theorem integral_bounds :
  2 ≤ ∫ x in 0..2, f x ∧ ∫ x in 0..2, f x ≤ 6 := by
  sorry

end integral_bounds_l778_778643


namespace projectile_first_reach_height_56_l778_778030

theorem projectile_first_reach_height_56 (t : ℝ) (h1 : ∀ t, y = -16 * t^2 + 60 * t) :
    (∃ t : ℝ, y = 56 ∧ t = 1.75 ∧ (∀ t', t' < 1.75 → y ≠ 56)) :=
by
  sorry

end projectile_first_reach_height_56_l778_778030


namespace min_period_and_sym_center_l778_778350

open Real

noncomputable def func (x α β : ℝ) : ℝ :=
  sin (x - α) * cos (x - β)

theorem min_period_and_sym_center (α β : ℝ) :
  (∀ x, func (x + π) α β = func x α β) ∧ (func α 0 β = 0) :=
by
  sorry

end min_period_and_sym_center_l778_778350


namespace find_eq_line_BC_find_area_triangle_l778_778751

structure Point3D (x y : ℝ)

def pointM : Point3D := Point3D.mk 1 6

def eq_line_AB (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def eq_line_AC (x y : ℝ) : Prop := x - y + 6 = 0

-- Question 1 Lean statement:
theorem find_eq_line_BC (A B C : Point3D) :
  AM_is_altitude_AM_side_BC (Point3D.mk (-1) 5) pointM → (∃ (k : ℝ), eq_line_BC (k * x + y - 8 = 0))
sorry

-- Question 2 Lean statement:
theorem find_area_triangle (A B C : Point3D) :
  AM_is_median_on_BC (Point3D.mk (-1) 5) pointM → (6)
sorry

end find_eq_line_BC_find_area_triangle_l778_778751


namespace ellipse_and_fixed_point_l778_778300

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778300


namespace ellipse_solution_geometry_solution_l778_778314

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l778_778314


namespace problem1_problem2_l778_778998

theorem problem1 :
  6 * Real.sqrt (1 / 9) - Real.cbrt 27 + (Real.sqrt 2)^2 = 1 :=
sorry

theorem problem2 :
  -1 ^ 2022 + Real.sqrt ((-2)^2) + abs (2 - Real.sqrt 3) = 3 - Real.sqrt 3 :=
sorry

end problem1_problem2_l778_778998


namespace ellipse_equation_fixed_point_l778_778278

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778278


namespace shaded_rectangle_area_l778_778400

theorem shaded_rectangle_area (side_length : ℝ) (x y : ℝ) 
  (h1 : side_length = 42) 
  (h2 : 4 * x + 2 * y = 168 - 4 * x) 
  (h3 : 2 * (side_length - y) + 2 * x = 168 - 4 * x)
  (h4 : 2 * (2 * x + y) = 168 - 4 * x) 
  (h5 : x = 18) :
  (2 * x) * (4 * x - (side_length - y)) = 540 := 
by
  sorry

end shaded_rectangle_area_l778_778400


namespace Ben_sales_value_l778_778171

noncomputable def value_of_sale (old_salary new_salary commission_ratio sales_required : ℝ) (diff_salary: ℝ) :=
  ∃ x : ℝ, 0.15 * x * sales_required = diff_salary ∧ x = 750

theorem Ben_sales_value (old_salary new_salary commission_ratio sales_required diff_salary: ℝ)
  (h1: old_salary = 75000)
  (h2: new_salary = 45000)
  (h3: commission_ratio = 0.15)
  (h4: sales_required = 266.67)
  (h5: diff_salary = old_salary - new_salary) :
  value_of_sale old_salary new_salary commission_ratio sales_required diff_salary :=
by
  sorry

end Ben_sales_value_l778_778171


namespace domain_of_tan_function_l778_778479

theorem domain_of_tan_function :
  ∀ (x : ℝ) (k : ℤ),
  (y = tan (π / 4 - x) → y ∉ {x | ∃ k : ℤ, x = k * π + 3 * π / 4})
  ∧ (∀ x : ℝ, x ∈ {x | ∃ k : ℤ, x = k * π + 3 * π / 4} ↔ ∃ k : ℤ, x = k * π + 3 * π / 4) :=
by
  sorry

end domain_of_tan_function_l778_778479


namespace count_ways_distinguishable_to_indistinguishable_l778_778370

-- Define the conditions: 5 distinguishable balls and 2 indistinguishable boxes.
def balls : Nat := 5
def boxes : Nat := 2

-- State the problem as a theorem to be proved in Lean.
theorem count_ways_distinguishable_to_indistinguishable :
  ∃ (n : Nat), n = 17 ∧ (number_of_ways balls boxes = n) :=
sorry

end count_ways_distinguishable_to_indistinguishable_l778_778370


namespace fraction_sum_eq_l778_778178

theorem fraction_sum_eq :
  (∑ n in finset.range 2008 + 2, 4 / (n * (n + 2))) = 14165 / 6033 :=
sorry

end fraction_sum_eq_l778_778178


namespace difference_max_min_planes_l778_778079

open Set

-- Defining the regular tetrahedron and related concepts
noncomputable def tetrahedron := Unit -- Placeholder for the tetrahedron

def union_faces (T : Unit) : Set Point := sorry -- Placeholder for union of faces definition

noncomputable def simple_trace (p : Plane) (T : Unit) : Set Point := sorry -- Placeholder for planes intersecting faces

-- Calculating number of planes
def maximum_planes (T : Unit) : Nat :=
  4 -- One for each face of the tetrahedron

def minimum_planes (T : Unit) : Nat :=
  2 -- Each plane covers traces on two adjacent faces if oriented appropriately

-- Statement of the problem
theorem difference_max_min_planes (T : Unit) :
  maximum_planes T - minimum_planes T = 2 :=
by
  -- Proof skipped
  sorry

end difference_max_min_planes_l778_778079


namespace remainder_base12_div_9_l778_778522

def base12_to_decimal (a b c d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12 + d

theorem remainder_base12_div_9 :
  let n := base12_to_decimal 1 7 3 1 in
  n % 9 = 1 :=
by
  sorry

end remainder_base12_div_9_l778_778522


namespace find_quadratic_polynomial_l778_778656

-- Given conditions to construct a quadratic polynomial with real coefficients
noncomputable def quadratic_polynomial (a b c : ℂ) (h1 : a = 5 + 2 * complex.I) : polynomial ℂ :=
  3 * (X - C a) * (X - C (conj a))

-- The required proof problem statement
theorem find_quadratic_polynomial (x : ℂ) :
    quadratic_polynomial 5 2 0 rfl = 3 * X^2 - 30 * X + 87 :=
sorry

end find_quadratic_polynomial_l778_778656


namespace determine_w_l778_778628

def g (z : ℂ) : ℂ :=
  ((1 + complex.I * real.sqrt 2) * z + (4 * real.sqrt 2 - 10 * complex.I)) / 3

theorem determine_w :
  ∃ w : ℂ, g w = w ∧ w = -2 * real.sqrt 2 + 4 / 3 - (10 / 3) * complex.I :=
by
  -- The proof should go here, but we'll leave it as "sorry" for now
  sorry

end determine_w_l778_778628


namespace play_number_of_children_l778_778157

theorem play_number_of_children (A C : ℕ) (ticket_price_adult : ℕ) (ticket_price_child : ℕ)
    (total_people : ℕ) (total_money : ℕ)
    (h1 : ticket_price_adult = 8)
    (h2 : ticket_price_child = 1)
    (h3 : total_people = 22)
    (h4 : total_money = 50)
    (h5 : A + C = total_people)
    (h6 : ticket_price_adult * A + ticket_price_child * C = total_money) :
    C = 18 := sorry

end play_number_of_children_l778_778157


namespace rectangle_circle_inequality_l778_778782

variable (a b r1 r2 : ℝ)

def is_rectangle (ABCD : Prop) : Prop :=
  AB = a ∧ BC = b

def circle_radius1 (r1 : ℝ) : Prop :=
  r1 = ((b ^ 2) / (2 * b) + (a ^ 2) / (8 * b))

def circle_radius2 (r2 : ℝ) : Prop :=
  r2 = ((a ^ 2) / (2 * a) + (b ^ 2) / (8 * a))

theorem rectangle_circle_inequality
  (ABCD : Prop) (r1 r2 : ℝ) (h_rect : is_rectangle ABCD) (h_r1 : circle_radius1 r1) (h_r2 : circle_radius2 r2) :
  r1 + r2 ≥ (5 / 8) * (a + b) :=
by
  sorry

end rectangle_circle_inequality_l778_778782


namespace find_f_values_find_f_expression_l778_778424

variable (f : ℕ+ → ℤ)

-- Conditions in Lean
def is_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ {m n : ℕ+}, m < n → f m < f n

axiom h1 : is_increasing f
axiom h2 : f 4 = 5
axiom h3 : ∀ n : ℕ+, ∃ k : ℕ, f n = k
axiom h4 : ∀ m n : ℕ+, f m * f n = f (m * n) + f (m + n - 1)

-- Proof in Lean 4
theorem find_f_values : f 1 = 2 ∧ f 2 = 3 ∧ f 3 = 4 :=
by
  sorry

theorem find_f_expression : ∀ n : ℕ+, f n = n + 1 :=
by
  sorry

end find_f_values_find_f_expression_l778_778424


namespace zoe_correctness_conjecture_l778_778618

variable {t : ℝ} (ht : t > 0) -- total number of problems
variables (chloe_fraction zoe_fraction amy_fraction : ℝ)
variables (chloe_correct_alone zoe_correct_alone together_correct : ℝ)
variables (chloe_overall_correct : ℝ)

-- Given conditions
def problem_conditions : Prop :=
  chloe_fraction = 0.40 ∧
  zoe_fraction = 0.30 ∧
  amy_fraction = 0.30 ∧
  together_correct = 0.85 ∧
  chloe_correct_alone = 0.80 ∧
  chloe_overall_correct = 0.88 ∧
  zoe_correct_alone = 0.75

-- Proof goal
def zoe_correct_answer_percentage (tipping_problem_conditions : problem_conditions) : Prop :=
  let zoe_correct_total := zoe_correct_alone * zoe_fraction * t + together_correct * 0 in
  zoe_correct_total / t * 100 = 85

-- Statement in Lean 4
theorem zoe_correctness_conjecture (h : problem_conditions) : zoe_correct_answer_percentage h :=
sorry

end zoe_correctness_conjecture_l778_778618


namespace find_k_l778_778874

-- Define the sum of even integers from 2 to 2k
def sum_even_integers (k : ℕ) : ℕ :=
  2 * (k * (k + 1)) / 2

-- Define the condition that this sum equals 132
def sum_condition (t : ℕ) (k : ℕ) : Prop :=
  sum_even_integers k = t

theorem find_k (k : ℕ) (t : ℕ) (h₁ : t = 132) (h₂ : sum_condition t k) : k = 11 := by
  sorry

end find_k_l778_778874


namespace inequality_range_of_a_l778_778034

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2: ℝ) 2 :=
by
  sorry

end inequality_range_of_a_l778_778034


namespace sum_of_digits_of_gcd_l778_778102

def gcd_of_differences : ℕ := Int.gcd (Int.gcd 3360 2240) 5600

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_gcd :
  sum_of_digits gcd_of_differences = 4 :=
by
  sorry

end sum_of_digits_of_gcd_l778_778102


namespace rain_on_Tuesday_correct_l778_778776

-- Let the amount of rain on Monday be represented by m
def rain_on_Monday : ℝ := 0.9

-- Let the difference in rain between Monday and Tuesday be represented by d
def rain_difference : ℝ := 0.7

-- Define the calculated amount of rain on Tuesday
def rain_on_Tuesday : ℝ := rain_on_Monday - rain_difference

-- The statement we need to prove
theorem rain_on_Tuesday_correct : rain_on_Tuesday = 0.2 := 
by
  -- Proof omitted (to be provided)
  sorry

end rain_on_Tuesday_correct_l778_778776


namespace ac_squared_times_bd_equals_ad_squared_times_bc_l778_778490

open EuclideanGeometry

variable {A B C D : Point}

-- Definitions of tangent conditions to the circumcircle
def tangent_to_circumcircle (P : Point) (Δ : Triangle) : Prop :=
  ∃ l : Line, tangent l Δ ∧ on_line P l

def circumcircle (Δ : Triangle) : Circle :=
  let ⟨a, b, c⟩ := Δ
  Circle.mk_centered (circumcenter a b c) (circumradius a b c)

-- Main theorem to prove the required equation
theorem ac_squared_times_bd_equals_ad_squared_times_bc
  (h_tangent_1 : tangent_to_circumcircle A ⟨B, C, D⟩)
  (h_tangent_2 : tangent_to_circumcircle B ⟨A, B, D⟩) :
  distance A C ^ 2 * distance B D = distance A D ^ 2 * distance B C := 
sorry

end ac_squared_times_bd_equals_ad_squared_times_bc_l778_778490


namespace comparison_of_powers_l778_778868

theorem comparison_of_powers : 6 ^ 0.7 > 0.7 ^ 6 ∧ 0.7 ^ 6 > 0.6 ^ 7 := by
  sorry

end comparison_of_powers_l778_778868


namespace largest_k_for_log3_of_B_l778_778194

noncomputable def T' : ℕ → ℕ
| 0     := 3
| (n+1) := 3 ^ T' n

def A' := (T' 4) ^ (T' 4)
def B' := (T' 4) ^ A'

theorem largest_k_for_log3_of_B' : ∃ k : ℕ, k = 5 ∧ ∀ m : ℕ, m > 5 → False
:= by
  have T'_1 : T' 1 = 3 := rfl
  have T'_2 : T' 2 = 3 ^ 3 := rfl
  have T'_3 : T' 3 = 3 ^ (3 ^ 3) := rfl
  have T'_4 : T' 4 = 3 ^ (3 ^ (3 ^ 3)) := rfl
  sorry

end largest_k_for_log3_of_B_l778_778194


namespace mixed_groups_count_l778_778896

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l778_778896


namespace integer_solutions_l778_778420

theorem integer_solutions (x : ℝ) (n : ℤ)
  (h1 : ⌊x⌋ = n) :
  3 * x - 2 * n + 4 = 0 ↔
  x = -4 ∨ x = (-14:ℚ)/3 ∨ x = (-16:ℚ)/3 :=
by sorry

end integer_solutions_l778_778420


namespace max_cos_sum_l778_778986

theorem max_cos_sum (A B C : ℝ) (h : A + B + C = 180) : 
  ∃ (M : ℝ), M = 1 ∧ ∀ (A B C : ℝ), A + B + C = 180 → (cos A + cos B * cos C) ≤ M := 
sorry

end max_cos_sum_l778_778986


namespace luke_total_points_correct_l778_778438

/-- Luke's points per round -/
def points_per_round : ℕ := 146

/-- Luke's number of rounds played -/
def number_of_rounds : ℕ := 157

/-- Luke's total points scored -/
def total_points_scored : ℕ := points_per_round * number_of_rounds

theorem luke_total_points_correct : total_points_scored = 22822 := by
  unfold total_points_scored
  unfold points_per_round
  unfold number_of_rounds
  calc 146 * 157 = 22822 : by sorry

end luke_total_points_correct_l778_778438


namespace ellipse_equation_fixed_point_l778_778270

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778270


namespace find_quotient_l778_778380

theorem find_quotient :
  ∀ (remainder dividend divisor quotient : ℕ),
    remainder = 1 →
    dividend = 217 →
    divisor = 4 →
    quotient = (dividend - remainder) / divisor →
    quotient = 54 :=
by
  intros remainder dividend divisor quotient hr hd hdiv hq
  rw [hr, hd, hdiv] at hq
  norm_num at hq
  exact hq

end find_quotient_l778_778380


namespace sum_of_areas_of_super_cool_triangles_l778_778578

def is_super_cool_triangle (a b : ℕ) : Prop :=
  (a * b / 2 = 3 * (a + b))

theorem sum_of_areas_of_super_cool_triangles :
  (∑ p in {p : ℕ × ℕ | is_super_cool_triangle p.1 p.2}, (p.1 * p.2) / 2) = 471 := 
by
  sorry

end sum_of_areas_of_super_cool_triangles_l778_778578


namespace radius_of_wider_can_is_10_sqrt_2_l778_778076

theorem radius_of_wider_can_is_10_sqrt_2 (h : ℝ) (r : ℝ) (π : ℝ) (Vol_narrow := π * (10 ^ 2) * (2 * h)) (Vol_wide := π * (r ^ 2) * h)
  (Volume_eq : Vol_narrow = Vol_wide) : r = 10 * Real.sqrt 2 :=
by
  -- volume equation from conditions
  have volume_eqn: π * 10^2 * 2 * h = π * r^2 * h, from Volume_eq,
  -- divide both sides by h (assuming h ≠ 0) and simplify
  have simplified_eqn: 200 = r^2, by sorry,
  -- solve for r
  have r_value: r = Real.sqrt 200, by sorry,
  -- simplify sqrt 200 to 10 * sqrt 2
  have result: Real.sqrt 200 = 10 * Real.sqrt 2, by sorry,
  exact Eq.trans r_value result

end radius_of_wider_can_is_10_sqrt_2_l778_778076


namespace area_covered_by_frames_l778_778166

-- Define the given conditions as lean definitions/variables
def side_length_larger_square : ℝ := 6
def frame_width : ℝ := 1
def num_frames : ℕ := 4
def num_overlapping_squares : ℕ := 6
def side_length_smaller_overlapping_square : ℝ := frame_width -- each overlapping small square has 1 cm side

noncomputable def area_frame_covering_table : ℝ := 74

-- Now, we assert the problem statement as a theorem.
theorem area_covered_by_frames : 
  let larger_square_area := side_length_larger_square * side_length_larger_square,
      smaller_square_side_length := side_length_larger_square - 2 * frame_width,
      smaller_square_area := smaller_square_side_length * smaller_square_side_length,
      area_one_frame := larger_square_area - smaller_square_area,
      total_area_frames := num_frames * area_one_frame,
      smaller_overlapping_area := side_length_smaller_overlapping_square * side_length_smaller_overlapping_square,
      total_overlapping_area := num_overlapping_squares * smaller_overlapping_area,
      net_area_covered := total_area_frames - total_overlapping_area
  in net_area_covered = area_frame_covering_table := 
by
  sorry

end area_covered_by_frames_l778_778166


namespace reduce_weighings_l778_778958

theorem reduce_weighings {B : Type} (boxes : B) (products : ℕ) (box_count : ℕ) (missing_product : ℕ → Prop) (cannot_reopen_boxes : Prop) (normal_weighings : ℕ) :
  products = 1500 ∧ box_count = 15 ∧ missing_product 1 ∧ cannot_reopen_boxes ∧ normal_weighings = 14 →
  ∃ weighings : ℕ, weighings ≤ 4 ∧ ∀ b ∈ boxes, (missing_product b ↔ weighings < normal_weighings) :=
begin
  sorry
end

end reduce_weighings_l778_778958


namespace comparison_among_a_b_c_l778_778222

theorem comparison_among_a_b_c :
  let a := Real.logBase 2 (2 ^ -1.2) in
  let b := Real.logBase 3 6 in
  let c := Real.logBase 5 10 in
  a < c ∧ c < b :=
by
  let a := 2 ^ -1.2
  let b := Real.logBase 3 6
  let c := Real.logBase 5 10
  sorry

end comparison_among_a_b_c_l778_778222


namespace triangle_side_range_l778_778707

theorem triangle_side_range (x : ℝ) (hx1 : 8 + 10 > x) (hx2 : 10 + x > 8) (hx3 : x + 8 > 10) : 2 < x ∧ x < 18 :=
by
  sorry

end triangle_side_range_l778_778707


namespace one_div_a_plus_one_div_b_l778_778378

theorem one_div_a_plus_one_div_b (a b : ℝ) (h₀ : a ≠ b) (ha : a^2 - 3 * a + 2 = 0) (hb : b^2 - 3 * b + 2 = 0) :
  1 / a + 1 / b = 3 / 2 :=
by
  -- Proof goes here
  sorry

end one_div_a_plus_one_div_b_l778_778378


namespace annual_interest_rate_l778_778650

theorem annual_interest_rate
  (P : ℝ) (t : ℝ) (compound_interest : ℝ) (n : ℕ) :
  P = 20000 →
  t = 2 →
  compound_interest = 1648.64 →
  n = 2 →
  let A := P + compound_interest,
      r := 0.04 in
  A = P * (1 + r / (n : ℝ)) ^ (n * t) →
  r = 0.04 :=
by
  intros hP ht hci hn hA
  let A := P + compound_interest
  let r := 0.04
  have hAcalc : A = 21648.64 := by
    rw [hP, hci]
    exact rfl
  rw hAcalc at hA
  apply hA

end annual_interest_rate_l778_778650


namespace min_trips_to_transport_400_tons_l778_778066

-- Define a cargo type that has a weight
structure Cargo where
  weight : ℝ
  is_multiple_of_quintal : ∃ (k : ℕ), weight = k * 0.1
  weight_leq_10tons : weight ≤ 10

-- Define the total cargo weight and distinctness constraint
def cargo_set : set Cargo := {
  c | c.weight ∈ { w : ℝ | ∃ k : ℕ, w = k * 0.1 ∧ w ≤ 10}
} 

-- Assert the total weight constraint and uniqueness
def total_weight (cargos : set Cargo) : ℝ :=
  ∑ cargo in cargos, cargo.weight

def all_weights_distinct (cargos : set Cargo) : Prop :=
  ∀ (c1 c2 : Cargo), c1 ∈ cargos → c2 ∈ cargos → c1 ≠ c2 → c1.weight ≠ c2.weight

-- Statement of the main theorem
theorem min_trips_to_transport_400_tons (cargos : set Cargo) :
  total_weight cargos = 400 ∧ all_weights_distinct cargos →
  ∃ min_trips : ℕ, min_trips = 51 ∧ (∀ t, t < min_trips → ¬ (t = min_trips)) ∧ 
  ∑ c in (c : set Cargo), c.weight ≤ 10 * min_trips :=
sorry

end min_trips_to_transport_400_tons_l778_778066


namespace min_trips_to_transport_400_tons_l778_778067

-- Define a cargo type that has a weight
structure Cargo where
  weight : ℝ
  is_multiple_of_quintal : ∃ (k : ℕ), weight = k * 0.1
  weight_leq_10tons : weight ≤ 10

-- Define the total cargo weight and distinctness constraint
def cargo_set : set Cargo := {
  c | c.weight ∈ { w : ℝ | ∃ k : ℕ, w = k * 0.1 ∧ w ≤ 10}
} 

-- Assert the total weight constraint and uniqueness
def total_weight (cargos : set Cargo) : ℝ :=
  ∑ cargo in cargos, cargo.weight

def all_weights_distinct (cargos : set Cargo) : Prop :=
  ∀ (c1 c2 : Cargo), c1 ∈ cargos → c2 ∈ cargos → c1 ≠ c2 → c1.weight ≠ c2.weight

-- Statement of the main theorem
theorem min_trips_to_transport_400_tons (cargos : set Cargo) :
  total_weight cargos = 400 ∧ all_weights_distinct cargos →
  ∃ min_trips : ℕ, min_trips = 51 ∧ (∀ t, t < min_trips → ¬ (t = min_trips)) ∧ 
  ∑ c in (c : set Cargo), c.weight ≤ 10 * min_trips :=
sorry

end min_trips_to_transport_400_tons_l778_778067


namespace find_y_coordinate_l778_778150

theorem find_y_coordinate :
  ∀ (m b x : ℝ), m = 9.9 ∧ b = 10 ∧ x = 100 → (m * x + b = 1000) :=
by
  intros m b x
  intro H
  cases H with Hm Hb_x
  cases Hb_x with Hb Hx
  rw [Hm, Hb, Hx]
  sorry

end find_y_coordinate_l778_778150


namespace distinct_license_plates_count_l778_778552

def num_digit_choices : Nat := 10
def num_letter_choices : Nat := 26
def num_digits : Nat := 5
def num_letters : Nat := 3

theorem distinct_license_plates_count :
  (num_digit_choices ^ num_digits) * (num_letter_choices ^ num_letters) = 1757600000 := 
sorry

end distinct_license_plates_count_l778_778552


namespace parabola_intercepts_sum_l778_778839

noncomputable def a : ℝ := 5

noncomputable def b : ℝ := (9 + Real.sqrt 21) / 6

noncomputable def c : ℝ := (9 - Real.sqrt 21) / 6

theorem parabola_intercepts_sum : a + b + c = 8 := by
  -- definition of a
  have ha : a = 5 := rfl
  
  -- definitions of b and c from roots of 3y^2 - 9y + 5 = 0
  have hb : b = (9 + Real.sqrt 21) / 6 := rfl
  have hc : c = (9 - Real.sqrt 21) / 6 := rfl
  
  -- Vieta's formulas implies b + c = 3
  have hb_c : b + c = 3 := by
    calc
    b + c = (9 + Real.sqrt 21) / 6 + (9 - Real.sqrt 21) / 6 : by rw [hb, hc]
    ... = (9 + 9) / 6 : by ring
    ... = 18 / 6 : by norm_num
    ... = 3 : by norm_num
  
  -- Sum a + b + c
  calc
  a + b + c = 5 + (b + c) : by rw [ha]
  ... = 5 + 3 : by rw [hb_c]
  ... = 8 : by norm_num

end parabola_intercepts_sum_l778_778839


namespace ellipse_equation_proof_HN_fixed_point_l778_778317

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l778_778317


namespace single_colony_reaches_limit_in_21_days_l778_778125

-- Definitions and given conditions
def doubles_every_day (colony_size : ℕ → ℕ) : Prop :=
  ∀ n, colony_size (n + 1) = 2 * colony_size n

def two_colonies_reach_limit (habitat_limit : ℕ) (days : ℕ) (colony_size : ℕ → ℕ) : Prop :=
  2 * colony_size (days - 1) = habitat_limit

-- Instantiate habitat limit for single colony and double colonies
def habitat_limit : ℕ := 2^20
def single_colony_size (n : ℕ) : ℕ := 2^n

-- The math proof problem translated into a Lean 4 statement
theorem single_colony_reaches_limit_in_21_days :
  doubles_every_day single_colony_size ∧ two_colonies_reach_limit habitat_limit 20 single_colony_size
  → ∃ x, x = 21 ∧ single_colony_size x = habitat_limit :=
by
  intro h
  use 21
  split
  rfl
  have h1 : single_colony_size = λ n, 2^n := rfl
  rw [h1]
  sorry

end single_colony_reaches_limit_in_21_days_l778_778125


namespace real_number_values_purely_imaginary_values_third_quadrant_values_l778_778708

-- Given the conditions for when z is a real number
theorem real_number_values {m : ℝ} (h : m^2 - 9m + 18 = 0) : m = 3 ∨ m = 6 :=
sorry

-- Given the conditions for when z is a purely imaginary number
theorem purely_imaginary_values {m : ℝ} (h1 : m^2 - 8m + 15 = 0) (h2 : m^2 - 9m + 18 ≠ 0) : m = 5 :=
sorry

-- Given the conditions for when A is located in the third quadrant
theorem third_quadrant_values {m : ℝ} (h1 : m^2 - 8m + 15 < 0) (h2 : m^2 - 9m + 18 < 0) : 3 < m ∧ m < 5 :=
sorry

end real_number_values_purely_imaginary_values_third_quadrant_values_l778_778708


namespace general_term_proof_sum_b_seq_equal_compare_F_T_l778_778693

def arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def b_seq (b a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = 2 ^ n * a n

def a_seq_general_term (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 4 * n - 3

def T_n_eq (T : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, T n = (4 * n - 7) * 2 ^ (n + 1) + 14

def F_n_eq (F : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, F n = (4 * n - 5) * 2 ^ (n + 1)

theorem general_term_proof (a : ℕ → ℕ) (d : ℕ) (h1 : d > 0)
  (h2 : (a 1 + d) * (a 2 + 2 * d) = 45) 
  (h3 : 2 * (a 0 + d) + 3 * d = 14) :
  a_seq_general_term a := 
sorry

theorem sum_b_seq_equal (a b T : ℕ → ℕ) (h : ∀ n, b n = 2^n * a n) (h2 : a_seq_general_term a) :
  T_n_eq T a :=
sorry

theorem compare_F_T (T F : ℕ → ℕ) :
  (∀ n, F n = (4 * n - 5) * 2^(n + 1)) →
  T_n_eq T (λ n, 4 * n - 3) →
  (F 1 < T 1 ∧ ∀ n, n ≥ 2 → F n > T n) :=
sorry

end general_term_proof_sum_b_seq_equal_compare_F_T_l778_778693


namespace tunnel_length_l778_778154

/-- A train travels at 80 kmph, enters a tunnel at 5:12 am, and leaves at 5:18 am.
    The length of the train is 1 km. Prove the length of the tunnel is 7 km. -/
theorem tunnel_length 
(speed : ℕ) (enter_time leave_time : ℕ) (train_length : ℕ) 
(h_enter : enter_time = 5 * 60 + 12) 
(h_leave : leave_time = 5 * 60 + 18) 
(h_speed : speed = 80) 
(h_train_length : train_length = 1) 
: ∃ tunnel_length : ℕ, tunnel_length = 7 :=
sorry

end tunnel_length_l778_778154


namespace cubes_occupy_75_percent_of_box_l778_778569

-- Define the dimensions of the box
def box_length : ℝ := 9
def box_width : ℝ := 8
def box_height : ℝ := 12

-- Define the side length of the cube
def cube_side : ℝ := 3

-- Define the number of cubes that fit in each dimension
def cubes_in_length : ℕ := (box_length / cube_side).to_int
def cubes_in_width : ℕ := (box_width / cube_side).to_int
def cubes_in_height : ℕ := (box_height / cube_side).to_int

-- Define the total number of cubes
def total_cubes : ℕ := cubes_in_length * cubes_in_width * cubes_in_height

-- Define the volume of one cube
def cube_volume : ℝ := cube_side ^ 3

-- Define the volume occupied by the cubes
def total_cubes_volume : ℝ := total_cubes * cube_volume

-- Define the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- Define the percentage of the volume occupied by the cubes
def percentage_volume_occupied : ℝ := (total_cubes_volume / box_volume) * 100

-- The theorem to prove
theorem cubes_occupy_75_percent_of_box : percentage_volume_occupied = 75 :=
by
  sorry

end cubes_occupy_75_percent_of_box_l778_778569


namespace trig_identity_equiv_l778_778451

theorem trig_identity_equiv (α : ℝ) :
  4 * sin (π / 6 + α / 2) * sin (π / 6 - α / 2) = (cos (3 * α / 2)) / (cos (α / 2)) :=
by
  sorry

end trig_identity_equiv_l778_778451


namespace find_width_of_first_tract_l778_778364

-- Definitions based on given conditions
noncomputable def area_first_tract (W : ℝ) : ℝ := 300 * W
def area_second_tract : ℝ := 250 * 630
def combined_area : ℝ := 307500

-- The theorem we need to prove: width of the first tract is 500 meters
theorem find_width_of_first_tract (W : ℝ) (h : area_first_tract W + area_second_tract = combined_area) : W = 500 :=
by
  sorry

end find_width_of_first_tract_l778_778364


namespace fill_pool_time_l778_778217

-- Define the conditions
def pool_volume : ℕ := 15000
def hoses1_rate : ℕ := 2
def hoses1_count : ℕ := 2
def hoses2_rate : ℕ := 3
def hoses2_count : ℕ := 2

-- Calculate the total delivery rate
def total_delivery_rate : ℕ :=
  (hoses1_rate * hoses1_count) + (hoses2_rate * hoses2_count)

-- Calculate the time to fill the pool in minutes
def time_to_fill_in_minutes : ℕ :=
  pool_volume / total_delivery_rate

-- Calculate the time to fill the pool in hours
def time_to_fill_in_hours : ℕ :=
  time_to_fill_in_minutes / 60

-- The theorem to prove
theorem fill_pool_time : time_to_fill_in_hours = 25 := by
  sorry

end fill_pool_time_l778_778217


namespace problem_statement_l778_778698

noncomputable def binomial (n k : ℕ) : ℚ := (n.choose k : ℚ)

theorem problem_statement :
  ∀ (n : ℕ), 0 < n ∧ binomial 23 (3*n + 1) = binomial 23 (n + 6)
  → (let a : ℕ → ℚ := λ k, if k = 0 then 81 else if k = 1 then -108 else if k = 2 then 54 else if k = 3 then -12 else 1 in
     a 0 - a 1 + a 2 - a 3 + a 4) = 256 :=
by
  intros n hn_eq_b_eq
  let a : ℕ → ℚ := λ k,
    if k = 0 then 3^4 else
      if k = 1 then 4 * 3^3 else
        if k = 2 then 6 * 3^2 else
          if k = 3 then 4 * 3 else
            1
  apply (a 0 - a 1 + a 2 - a 3 + a 4 = 3^4)
  sorry

end problem_statement_l778_778698


namespace min_value_collinear_vectors_l778_778804

theorem min_value_collinear_vectors (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (a - 1) * 2 - 1 * (-b - 1) = 0) : 
  \frac{1}{a} + \frac{2}{b} = 8 := 
begin
  sorry
end

end min_value_collinear_vectors_l778_778804


namespace intersection_set_union_set_range_a_l778_778696

def setA : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def setB : Set ℝ := {x | 2 < x ∧ x < 9}
def setC (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem intersection_set (x : ℝ) : x ∈ setA ∩ setB ↔ 3 ≤ x ∧ x < 6 :=
by {
  sorry
}

theorem union_set (x : ℝ) : x ∈ setA ∪ setB ↔ 2 < x ∧ x < 9 :=
by {
  sorry
}

theorem range_a (a : ℝ) : setC(a) ⊆ setB ↔ 2 < a ∧ a < 8 :=
by {
  sorry
}

end intersection_set_union_set_range_a_l778_778696


namespace number_multiplied_by_9_l778_778960

theorem number_multiplied_by_9 (x : ℕ) (h : 50 = x + 26) : 9 * x = 216 := by
  sorry

end number_multiplied_by_9_l778_778960


namespace eval_first_expression_eval_second_expression_l778_778614

theorem eval_first_expression : 
  (0.64^(- (1/2)) - (-(1/8))^(0) + 8^(2/3) + (9/16)^(1/2)) = 6 := 
by 
  sorry

theorem eval_second_expression : 
  (real.log 2)^2 + real.log 2 * real.log 5 + real.log 5 = 1 := 
by 
  sorry

end eval_first_expression_eval_second_expression_l778_778614


namespace maximal_f_value_l778_778212

def starts_with_digit (m : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, 10^k ≤ m ∧ m < 10^k * (d + 1) ∧ m / 10^k = d

noncomputable def f (n d : ℕ) : ℕ :=
  Inf { k : ℕ | starts_with_digit (k * n) d }

theorem maximal_f_value : ∀ (n d : ℕ), 0 < n → 0 < d → d < 10 → f(n, d) ≤ 81 :=
by
  intros n d hn hd1 hd2
  sorry

end maximal_f_value_l778_778212


namespace lily_catches_up_mary_in_60_minutes_l778_778439

theorem lily_catches_up_mary_in_60_minutes
  (mary_speed : ℝ) (lily_speed : ℝ) (initial_distance : ℝ)
  (h_mary_speed : mary_speed = 4)
  (h_lily_speed : lily_speed = 6)
  (h_initial_distance : initial_distance = 2) :
  ∃ t : ℝ, t = 60 := by
  sorry

end lily_catches_up_mary_in_60_minutes_l778_778439


namespace sum_a_48_49_ninth_occurrence_of_5_l778_778357

def a : ℕ+ → ℕ
| n => if n % 2 = 1 then n else a (n / 2)

theorem sum_a_48_49 : a 48 + a 49 = 52 :=
sorry

theorem ninth_occurrence_of_5 : (5 * 2 ^ 8 : ℕ) = 1280 :=
sorry

end sum_a_48_49_ninth_occurrence_of_5_l778_778357


namespace problem1_solution_set_problem2_proof_l778_778351

-- Define the function f(x) with a given value of a.
def f (x : ℝ) (a : ℝ) : ℝ := |x + a|

-- Problem 1: Solve the inequality f(x) ≥ 5 - |x - 2| when a = 1.
theorem problem1_solution_set (x : ℝ) :
  f x 1 ≥ 5 - |x - 2| ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

-- Problem 2: Given the solution set of f(x) ≤ 5 is [-9, 1] and the equation 1/m + 1/(2n) = a, prove m + 2n ≥ 1
theorem problem2_proof (a m n : ℝ) (hma : a = 4) (hmpos : m > 0) (hnpos : n > 0) :
  (1 / m + 1 / (2 * n) = a) → m + 2 * n ≥ 1 :=
sorry

end problem1_solution_set_problem2_proof_l778_778351


namespace profit_growth_rate_and_expected_profit_l778_778608

theorem profit_growth_rate_and_expected_profit
  (profit_April : ℕ)
  (profit_June : ℕ)
  (months : ℕ)
  (avg_growth_rate : ℝ)
  (profit_July : ℕ) :
  profit_April = 6000 ∧ profit_June = 7260 ∧ months = 2 ∧ 
  (profit_April : ℝ) * (1 + avg_growth_rate)^months = profit_June →
  avg_growth_rate = 0.1 ∧ 
  (profit_June : ℝ) * (1 + avg_growth_rate) = profit_July →
  profit_July = 7986 := 
sorry

end profit_growth_rate_and_expected_profit_l778_778608


namespace smaller_number_is_22_l778_778875

noncomputable def smaller_number (x y : ℕ) : ℕ := 
x

theorem smaller_number_is_22 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : x = 22 :=
by
  sorry

end smaller_number_is_22_l778_778875


namespace area_of_figure_l778_778204

theorem area_of_figure (x y : ℝ) : 
  2 * (2 - x) ≥ (abs (y - x^2) + abs (y + x^2)) →
  ∃ A B C D : ℝ × ℝ,
    (A = (-2, 4)) ∧
    (B = (1, 1)) ∧
    (C = (1, -1)) ∧
    (D = (-2, -4)) ∧
    (abs (A.1 - C.1) + abs (B.1 - D.1)) = 10 ∧
    (((abs (A.2 - D.2) * 5) / 2) = 15) :=
begin
  sorry
end

end area_of_figure_l778_778204


namespace limit_seq_l778_778622

open BigOperators

theorem limit_seq :
  tendsto (λ n : ℕ, (2^n + 7^n) / (2^n - 7^(n-1) : ℝ)) at_top (𝓝 (-7)) :=
begin
  sorry
end

end limit_seq_l778_778622


namespace part1_extreme_value_part2_range_of_a_l778_778715

open Real

noncomputable def f (a x : ℝ) : ℝ := (1 - a * x) / exp x

theorem part1_extreme_value (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, is_local_min (f 1) x ∧ f 1 x = -1 / exp 2 :=
by sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 1 - 2 * x) ↔ a ≤ 1 :=
by sorry

end part1_extreme_value_part2_range_of_a_l778_778715


namespace ellipse_parametric_form_l778_778603

theorem ellipse_parametric_form :
  (∃ A B C D E F : ℤ,
    ((∀ t : ℝ, (3 * (Real.sin t - 2)) / (3 - Real.cos t) = x ∧ 
     (2 * (Real.cos t - 4)) / (3 - Real.cos t) = y) → 
    (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.gcd (Int.natAbs D) (Int.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1846)) := 
sorry

end ellipse_parametric_form_l778_778603


namespace marbles_left_l778_778440

def initial_marbles : ℝ := 9.0
def given_marbles : ℝ := 3.0

theorem marbles_left : initial_marbles - given_marbles = 6.0 := 
by
  sorry

end marbles_left_l778_778440


namespace length_C_to_C_l778_778924

-- Define the coordinates of triangle vertices
def A : EuclideanSpace ℝ (Fin 2) := ![-2, 0]
def B : EuclideanSpace ℝ (Fin 2) := ![1, 4]
def C : EuclideanSpace ℝ (Fin 2) := ![-3, 2]

-- Define the reflection operation
def reflect_y (p : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  ![-p 0, p 1]

def reflect_x (p : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  ![p 0, -p 1]

-- Define C' and C''
def C' := reflect_y C
def C'' := reflect_x C'

-- Define the distance function
def distance (p1 p2 : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.sqrt ((p1 0 - p2 0)^2 + (p1 1 - p2 1)^2)

-- State the theorem
theorem length_C_to_C'' : distance C C'' = 2 * Real.sqrt 13 := by
  sorry

end length_C_to_C_l778_778924


namespace perimeter_BDA_form_l778_778765

def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

variables {A B C D E F : Point}
variables (h_eq_triangle : is_equilateral A B C)
variables (h_AB : dist A B = 10)
variables (D_on_AE : lies_on D (segment A E))
variables (E_on_CF : lies_on E (segment C F))
variables (F_on_BD : lies_on F (segment B D))
variables (congruent_triangles : congruent (triangle A E C) (triangle B D A)
                              ∧ congruent (triangle C F B) (triangle B D A))
variables (h_DE : dist D E = 6)

theorem perimeter_BDA_form:
  ∃ (a b c d : ℕ), let P := dist B D + dist D A + dist A B
  in (d > 0 ∧ P = (a + b * sqrt c) / d ∧ Nat.gcd b d = 1 ∧ ¬ ∃ p : ℕ, p * p ∣ c) ∧ a + b + c + d = 308 := 
sorry

end perimeter_BDA_form_l778_778765


namespace max_rounds_leq_three_l778_778416

noncomputable def f (n : ℕ) : ℕ := sorry

theorem max_rounds_leq_three (n : ℕ) (hn : n > 1) :
  f(n) ≤ 3 :=
sorry

end max_rounds_leq_three_l778_778416


namespace sum_of_super_cool_areas_l778_778579

def is_super_cool (a b : ℕ) : Prop :=
  (a - 9) * (b - 9) = 81

theorem sum_of_super_cool_areas : 
  let areas := [(90 * 10) / 2, (36 * 12) / 2, (18 * 18) / 2].erase_dup
  areas.sum = 828 :=
by
  sorry

end sum_of_super_cool_areas_l778_778579


namespace rectangle_ratio_l778_778766

-- Given conditions
variable (w : ℕ) -- width is a natural number

-- Definitions based on conditions 
def length := 10
def perimeter := 30

-- Theorem to prove
theorem rectangle_ratio (h : 2 * length + 2 * w = perimeter) : w = 5 ∧ 1 = 1 ∧ 2 = 2 :=
by
  sorry

end rectangle_ratio_l778_778766


namespace total_parents_in_auditorium_l778_778497

def num_girls : ℕ := 6
def num_boys : ℕ := 8
def parents_per_child : ℕ := 2

theorem total_parents_in_auditorium (num_girls num_boys parents_per_child : ℕ) : num_girls + num_boys = 14 → 2 * (num_girls + num_boys) = 28 := by
  assume h: num_girls + num_boys = 14
  show 2 * (num_girls + num_boys) = 28, from
    calc
      2 * (num_girls + num_boys) = 2 * 14 : by rw h
      ... = 28 : by norm_num

end total_parents_in_auditorium_l778_778497


namespace arithmetic_geometric_mean_inequality_l778_778362

variables {x y z : ℝ}

theorem arithmetic_geometric_mean_inequality (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1 / 3) :=
begin
  sorry
end

end arithmetic_geometric_mean_inequality_l778_778362


namespace f_diff_l778_778250

-- Define the function f
def f (n : ℕ) : ℝ :=
  (finset.range (2 * n + 1)).filter (λ k, n + 1 ≤ k ∧ k ≤ 2 * n).sum (λ k, 1 / (k : ℝ))

-- State the theorem
theorem f_diff (n : ℕ) (hn : 0 < n) :
  f (n + 1) - f n = 1 / (2 * n + 1) - 1 / (2 * n + 2) :=
by
  sorry

end f_diff_l778_778250


namespace ratio_monkeys_camels_l778_778616

-- Definitions corresponding to conditions
variables (zebras camels monkeys giraffes : ℕ)
variables (multiple : ℕ)

-- Conditions
def condition1 := zebras = 12
def condition2 := camels = zebras / 2
def condition3 := monkeys = camels * multiple
def condition4 := giraffes = 2
def condition5 := monkeys = giraffes + 22

-- Question: What is the ratio of monkeys to camels? Prove it is 4:1 given the conditions.
theorem ratio_monkeys_camels (zebras camels monkeys giraffes multiple : ℕ) 
  (h1 : condition1 zebras) 
  (h2 : condition2 zebras camels)
  (h3 : condition3 camels monkeys multiple)
  (h4 : condition4 giraffes)
  (h5 : condition5 monkeys giraffes) :
  multiple = 4 :=
sorry

end ratio_monkeys_camels_l778_778616


namespace range_of_t_l778_778385

-- Define the problem conditions and the range of t
variables {A B C t : ℝ} {a b c : ℝ} [TriangleABC : Triangle A B C]

-- Conditions given in the problem
axiom ac_eq_one_fourth_b_squared : a * c = (1 / 4) * b ^ 2
axiom sin_A_plus_sin_C_eq_t_sin_B : Real.sin A + Real.sin C = t * Real.sin B
axiom B_is_acute : 0 < Real.cos B ∧ Real.cos B < 1

-- Statement to prove
theorem range_of_t (h : ac_eq_one_fourth_b_squared ∧ sin_A_plus_sin_C_eq_t_sin_B ∧ B_is_acute) :
  (Real.sqrt 6 / 2) < t ∧ t < Real.sqrt 2 :=
sorry

end range_of_t_l778_778385


namespace min_trips_to_transport_400_tons_l778_778064

-- Define a cargo type that has a weight
structure Cargo where
  weight : ℝ
  is_multiple_of_quintal : ∃ (k : ℕ), weight = k * 0.1
  weight_leq_10tons : weight ≤ 10

-- Define the total cargo weight and distinctness constraint
def cargo_set : set Cargo := {
  c | c.weight ∈ { w : ℝ | ∃ k : ℕ, w = k * 0.1 ∧ w ≤ 10}
} 

-- Assert the total weight constraint and uniqueness
def total_weight (cargos : set Cargo) : ℝ :=
  ∑ cargo in cargos, cargo.weight

def all_weights_distinct (cargos : set Cargo) : Prop :=
  ∀ (c1 c2 : Cargo), c1 ∈ cargos → c2 ∈ cargos → c1 ≠ c2 → c1.weight ≠ c2.weight

-- Statement of the main theorem
theorem min_trips_to_transport_400_tons (cargos : set Cargo) :
  total_weight cargos = 400 ∧ all_weights_distinct cargos →
  ∃ min_trips : ℕ, min_trips = 51 ∧ (∀ t, t < min_trips → ¬ (t = min_trips)) ∧ 
  ∑ c in (c : set Cargo), c.weight ≤ 10 * min_trips :=
sorry

end min_trips_to_transport_400_tons_l778_778064


namespace AM_less_than_BM_plus_CM_l778_778789

open_locale classical

variables {A B C M O : Type} [inner_product_space ℝ O]

-- Defining an equilateral triangle inscribed in a circle
structure equilateral_triangle (A B C : O) : Prop :=
(convex : convex_hull ℝ (finset.insert A (finset.insert B (finset.singleton C)) ⊆ metric.sphere (0 : O) (dist A 0)))

-- M is a point on the arc AC but not A or C
structure arc_point (A C M : O) : Prop :=
(on_arc : ∃ p : ℝ, 0 < p ∧ p < 1 ∧ M = p • A + (1 - p) • C)

-- Main theorem statement
theorem AM_less_than_BM_plus_CM
  (A B C M : O) (h_triangle : equilateral_triangle A B C) (h_arc : arc_point A C M) :
  dist A M < dist B M + dist C M :=
sorry

end AM_less_than_BM_plus_CM_l778_778789


namespace mms_pack_count_l778_778780

def mms_per_pack (sundaes_monday : Nat) (mms_monday : Nat) (sundaes_tuesday : Nat) (mms_tuesday : Nat) (packs : Nat) : Nat :=
  (sundaes_monday * mms_monday + sundaes_tuesday * mms_tuesday) / packs

theorem mms_pack_count 
  (sundaes_monday : Nat)
  (mms_monday : Nat)
  (sundaes_tuesday : Nat)
  (mms_tuesday : Nat)
  (packs : Nat)
  (monday_total_mms : sundaes_monday * mms_monday = 240)
  (tuesday_total_mms : sundaes_tuesday * mms_tuesday = 200)
  (total_packs : packs = 11)
  : mms_per_pack sundaes_monday mms_monday sundaes_tuesday mms_tuesday packs = 40 := by
  sorry

end mms_pack_count_l778_778780


namespace find_selling_prices_max_profit_l778_778964

-- definitions for the given conditions
def price_A := 210
def price_B := 180

def Revenue_week1 := 2150
def Revenue_week2 := 3700

def Sales_week1_A := 3
def Sales_week1_B := 5

def Sales_week2_A := 4
def Sales_week2_B := 10

def Total_units := 40
def Max_expense := 7650

-- Part (1): Finding selling prices
theorem find_selling_prices (x y : ℝ) :
  (Sales_week1_A * x + Sales_week1_B * y = Revenue_week1) ∧
  (Sales_week2_A * x + Sales_week2_B * y = Revenue_week2) →
  x = 300 ∧ y = 250 :=
sorry

-- Part (2): Maximizing the profit
theorem max_profit (a : ℝ) :
  (price_A * a + price_B * (Total_units - a) ≤ Max_expense) →
  a ≤ 15 →
  let Profit := ((300 - 210) * a + (250 - 180) * (Total_units - a)) in
  Profit ≤ 3100 :=
sorry

end find_selling_prices_max_profit_l778_778964


namespace f_is_symmetric_to_g_l778_778379

-- Conditions: Given g(x) = ln(2x) and f is symmetric to g with respect to the line x = y

def g (x : ℝ) : ℝ := Real.log (2 * x)

def symmetric (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x = y → f y = x

-- Statement: Prove that f(x) = 1/2 * e^x
theorem f_is_symmetric_to_g : ∃ f : ℝ → ℝ, symmetric f g ∧ ∀ x : ℝ, f x = (1/2) * Real.exp x :=
sorry

end f_is_symmetric_to_g_l778_778379


namespace line_distance_tangent_sum_diff_l778_778010

variables {O A B : Type} [metric_space O] [nonempty O]
noncomputable def distance (x y : O) : ℝ := sorry
noncomputable def tangent_length (x O : O) (r : ℝ) : ℝ := sorry
noncomputable def on_line (A B : O) (L : O → Prop) : Prop := sorry

theorem line_distance_tangent_sum_diff
  (O : O) (r : ℝ) (L : O → Prop)
  (not_intersect : ∀ P, ¬(L P ∧ distance P O < r))
  (A B : O) (hA : L A) (hB : L B) :
  let d := distance A B,
      tA := tangent_length A O r,
      tB := tangent_length B O r in
  (d < tA + tB ∧ d > |tA - tB|) ↔ ¬(d ∉ (|tA - tB|, tA + tB)) := sorry

end line_distance_tangent_sum_diff_l778_778010


namespace range_ab_l778_778229

-- Define any necessary variables and conditions
variables {a b : ℝ} (z : ℂ) (h : Complex.abs z = 1)

-- Define a complex number
def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

-- The main theorem statement
theorem range_ab (a b : ℝ) (h : a^2 + b^2 = 1) : a * b ∈ set.Icc (-1/2) (1/2) :=
sorry

end range_ab_l778_778229


namespace rice_total_price_rice_total_price_specific_l778_778810

theorem rice_total_price (a n : ℕ) (h_deviations : list ℕ) (H_deviations : h_deviations = [-1, -2, 1, 1, 2, 1]) :
  (6 * a + h_deviations.sum) * n = 6 * a * n + 2 * n :=
by sorry

theorem rice_total_price_specific : 
  let a := 10
  let n := 6
  let h_deviations := [-1, -2, 1, 1, 2, 1]
  in (6 * a + h_deviations.sum) * n = 372 :=
by sorry

end rice_total_price_rice_total_price_specific_l778_778810


namespace ellipse_equation_fixed_point_l778_778271

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l778_778271


namespace book_contains_300_pages_l778_778120

-- The given conditions
def total_digits : ℕ := 792
def digits_per_page_1_to_9 : ℕ := 9 * 1
def digits_per_page_10_to_99 : ℕ := 90 * 2
def remaining_digits : ℕ := total_digits - digits_per_page_1_to_9 - digits_per_page_10_to_99
def pages_with_3_digits : ℕ := remaining_digits / 3

-- The total number of pages
def total_pages : ℕ := 99 + pages_with_3_digits

theorem book_contains_300_pages : total_pages = 300 := by
  sorry

end book_contains_300_pages_l778_778120


namespace value_of_b_l778_778946

theorem value_of_b (a b : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * b) : b = 49 := by
  have fact_a : a^3 = (105:ℕ)^3 := by rw h₁
  have prime_fact : 21 * 25 * 45 = 3^3 * 5^3 * 7 := by norm_num
  rw [fact_a, prime_fact] at h₂
  sorry

end value_of_b_l778_778946


namespace trapezoid_area_l778_778597

-- Define the conditions
variables (x l : ℝ)
def base1 := 3 * x
def base2 := 4 * x
def height := x
def perimeter := base1 + base2 + 2 * l

-- Define the area expression
def area := (1 / 2) * (base1 + base2) * height

-- State the theorem to be proved
theorem trapezoid_area (h_perimeter : perimeter = 20) (h_legs : ∀ x l, 7 * x + 2 * l = 20) :
  x = 20 / 9 ∧ area = 1400 / 81 :=
begin
  -- Step 1: Given the conditions, assume legs are equal to height
  have h1 : l = x,
  { sorry },

  -- Step 2: Solve for x using the perimeter condition
  have hx : 7 * x + 2 * x = 20,
  { sorry },

  -- Step 3: Substitute x into area expression
  have ha : (1 / 2) * (7 * (20 / 9) * (20 / 9)) = 1400 / 81,
  { sorry },

  -- Combine the results
  exact ⟨hx, ha⟩
end

end trapezoid_area_l778_778597


namespace add_fractions_l778_778645

theorem add_fractions : (7 / 12) + (3 / 8) = 23 / 24 := by
  sorry

end add_fractions_l778_778645


namespace range_of_x_for_inequality_l778_778703

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_of_x_for_inequality (x : ℝ) (h1 : 0 < x) (h2 : ∀ a b, a < b → f(a) < f(b)) :
  (f(2 * x - 1) < f(1 / 3)) ↔ (1 / 2 < x ∧ x < 2 / 3) :=
by
  sorry

end range_of_x_for_inequality_l778_778703


namespace quadratic_polynomial_real_coeff_l778_778662

theorem quadratic_polynomial_real_coeff (a b : ℂ) (h₁ : a = 5 + 2*i) 
  (h₂ : b = 5 - 2*i) (c : ℂ) (hc : c = 3) :
  3 * (X - C a) * (X - C b) = 3*X^2 - 30*X + 87 := 
by {
  sorry
}

end quadratic_polynomial_real_coeff_l778_778662


namespace negation_one_zero_l778_778863

theorem negation_one_zero (a b : ℝ) (h : a ≠ 0):
  ¬ (∃! x : ℝ, a * x + b = 0) ↔ (¬ ∃ x : ℝ, a * x + b = 0 ∨ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ + b = 0 ∧ a * x₂ + b = 0) := by
sorry

end negation_one_zero_l778_778863


namespace BF_equals_AF_plus_CF_l778_778382

variables {A B C D E F : Type} [metric_space A]

-- Conditions
variables (ABC : Triangle A) (AB AC BC : Segment A)

variables (D : Point BC) (E : Point (extension BA))

axiom AB_lt_AC : AB < AC
axiom AC_lt_BC : AC < BC
axiom BD_eq_BE : BD = BE
axiom BD_eq_AC : BD = AC

-- Circumcircles intersection
axiom circumcircle_BDE_intersects_ABC : 
  intersects (circumcircle (to_tris B D E)) (circumcircle ABC) = F

-- Question to Prove
theorem BF_equals_AF_plus_CF : 
  BF = AF + CF :=
sorry

end BF_equals_AF_plus_CF_l778_778382


namespace abs_x_ge_abs_4ax_l778_778109

theorem abs_x_ge_abs_4ax (a : ℝ) (h : ∀ x : ℝ, abs x ≥ 4 * a * x) : abs a ≤ 1 / 4 :=
sorry

end abs_x_ge_abs_4ax_l778_778109


namespace parabola_intercept_sum_l778_778848

theorem parabola_intercept_sum (a b c : ℝ) : 
  (∃ y : ℝ, a = 3 * y^2 - 9 * y + 5) ∧ (∀ x : ℝ, x = 0 → b ≠ c → 3 * b^2 - 9 * b + 5 = 0 ∧ 3 * c^2 - 9 * c + 5 = 0 ∧ b + c = 3) → 
  a + b + c = 8 :=
begin
  sorry
end

end parabola_intercept_sum_l778_778848


namespace four_star_three_l778_778734

def star (a b : ℕ) : ℕ := a^2 - a * b + b^2 + 2 * a * b

theorem four_star_three : star 4 3 = 37 :=
by
  -- here we would normally provide the proof steps
  sorry

end four_star_three_l778_778734


namespace min_trips_theorem_l778_778071

noncomputable def min_trips (weights : List ℕ) (capacity : ℕ) (total_weight : ℕ) : ℕ :=
  if total_weight = 0 then 0
  else let weight_sum := List.sum weights
       if weight_sum <= capacity then 1
       else if weights.length = 1 then weights.length / capacity
       else (weight_sum + capacity - 1) / capacity

theorem min_trips_theorem (weights : List ℕ) (h_distinct : weights.nodup)
                          (h_max_weight : ∀ w ∈ weights, w ≤ 10)
                          (h_total_weight : List.sum weights = 400)
                          (h_multiple_of_quintal : ∀ w ∈ weights, w % 100 = 0) :
  min_trips weights 10 400 = 51 := 
sorry

end min_trips_theorem_l778_778071


namespace fibonacci_remainders_periodic_no_polynomial_fibonacci_prime_l778_778943

-- Definitions for Problem 1
def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_remainders_periodic (n : ℕ) : ∃ k, ∀ m, fibonacci (m + k) % n = fibonacci m % n := 
sorry

-- Definitions for Problem 2
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_polynomial_fibonacci_prime (P : ℤ → ℤ) (hP : ∀ n : ℤ, prime (int.nat_abs (P n))) : 
  ∃ k, P(int.nat_abs (fibonacci k)) = P(int.nat_abs (fibonacci (k + 1))) := 
sorry

end fibonacci_remainders_periodic_no_polynomial_fibonacci_prime_l778_778943


namespace prime_divisor_of_sum_of_four_consecutive_integers_l778_778914

theorem prime_divisor_of_sum_of_four_consecutive_integers : 
  ∃ p : ℕ, Prime p ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) ∧ p = 2 := 
by
  sorry

end prime_divisor_of_sum_of_four_consecutive_integers_l778_778914


namespace polygon_sides_l778_778746

theorem polygon_sides (h : 900 = ∑ angles in polygon ∧ exterior_sum = 360) : sides = 5 := sorry

end polygon_sides_l778_778746


namespace symmetric_point_correct_l778_778475

-- Define the point and line
def point : ℝ × ℝ := (-1, 2)
def line (x : ℝ) : ℝ := x - 1

-- Define a function that provides the symmetric point with respect to the line
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ :=
  -- Since this function is a critical part of the problem, we won't define it explicitly. Using a placeholder.
  sorry

-- The proof problem
theorem symmetric_point_correct : symmetric_point point line = (3, -2) :=
  sorry

end symmetric_point_correct_l778_778475


namespace loss_percentage_when_man_sold_car_l778_778136

def originalCostPrice : ℝ := 50000
def sellingPriceFriend : ℝ := 54000
def gainPercentageFriend : ℝ := 20 / 100  -- Convert percentage to decimal

def sellingPriceMan : ℝ := sellingPriceFriend / (1 + gainPercentageFriend)

def lossPercentage (originalCostPrice : ℝ) (sellingPriceMan : ℝ) : ℝ :=
  ((originalCostPrice - sellingPriceMan) / originalCostPrice) * 100

theorem loss_percentage_when_man_sold_car :
  lossPercentage originalCostPrice sellingPriceMan = 10 := 
by {
  sorry
}

end loss_percentage_when_man_sold_car_l778_778136


namespace problem1_problem2_l778_778045

theorem problem1 (λ : ℝ) (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) = (n^2 + n - λ) * a n) :
  a 2 = -1 → (λ = 3 ∧ a 3 = -3) :=
by
  sorry

theorem problem2 (λ : ℝ) (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) = (n^2 + n - λ) * a n) :
  ¬∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d :=
by
  sorry

end problem1_problem2_l778_778045


namespace interval_contains_root_l778_778035

noncomputable def f (x : ℝ) := -1 / x + Real.log2 x

theorem interval_contains_root : ∃ c ∈ Set.Ioo 1 2, f c = 0 := 
by {
  -- Define the points and intervals
  let a := 1,
  let b := 2,
  -- Show function values
  have f_a : f a = -1 := by sorry,
  have f_b : f b = 0.5 := by sorry,
  -- Use the intermediate value theorem
  apply IntermediateValueTheorem,
  -- Provide conditions
  sorry
}

end interval_contains_root_l778_778035


namespace solve_quadratic_eq_l778_778463

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2*x + 1 = 0) : x = 1 :=
by
  sorry

end solve_quadratic_eq_l778_778463


namespace sqrt_10_parts_l778_778741

theorem sqrt_10_parts :
  (a, b : ℝ) (h : a + b = Real.sqrt 10 ∧ a = Real.floor (Real.sqrt 10))  
  (ha : a = 3) (hb : b = Real.sqrt 10 - 3) :
  a = 3 ∧ b = Real.sqrt 10 - 3 := 
sorry

end sqrt_10_parts_l778_778741


namespace sum_series_correct_l778_778668

noncomputable def sum_series : ℚ :=
  ∑ n in Finset.range 2008, 2 / ((n + 1) * (n + 2) * (n + 3))

theorem sum_series_correct : sum_series = 1009522 / 2019045 :=
by sorry

end sum_series_correct_l778_778668


namespace mike_combinations_l778_778442

-- Definition of digits and conditions
def digits := {1, 2, 3, 4, 5, 6}
def odd_digits := {1, 3, 5}
def even_digits := {2, 4, 6}

-- Conditions in terms of propositions
def follows_conditions (seq : List ℕ) : Prop :=
  seq.length = 6 ∧
  (∀ i, i < seq.length - 1 → 
    (i % 2 = 0 → seq[i] ∈ odd_digits ∧ seq[i+1] ∈ even_digits ∧ seq[i+1] ≠ seq[(i+1)/2 * 2])
    ∧ (i % 2 = 1 → seq[i] ∈ even_digits ∧ seq[i+1] ∈ odd_digits))

-- Main proof statement
theorem mike_combinations : 
  ∃ (combinations : List (List ℕ)), 
  (∀ seq ∈ combinations, follows_conditions seq) ∧
  combinations.length = 648 :=
sorry

end mike_combinations_l778_778442


namespace sum_of_ages_l778_778101

variable (S F : ℕ)

theorem sum_of_ages (h1 : F - 18 = 3 * (S - 18)) (h2 : F = 2 * S) : S + F = 108 := by
  sorry

end sum_of_ages_l778_778101


namespace how_many_more_yellow_peaches_l778_778386

-- Definitions
def red_peaches : ℕ := 7
def yellow_peaches_initial : ℕ := 15
def green_peaches : ℕ := 8
def combined_red_green_peaches := red_peaches + green_peaches
def required_yellow_peaches := 2 * combined_red_green_peaches
def additional_yellow_peaches_needed := required_yellow_peaches - yellow_peaches_initial

-- Theorem statement
theorem how_many_more_yellow_peaches :
  additional_yellow_peaches_needed = 15 :=
by
  sorry

end how_many_more_yellow_peaches_l778_778386


namespace least_people_cheaper_second_caterer_l778_778007

noncomputable def cost_first_caterer (x : ℕ) : ℕ := 50 + 18 * x

noncomputable def cost_second_caterer (x : ℕ) : ℕ := 
  if x >= 30 then 150 + 15 * x else 180 + 15 * x

theorem least_people_cheaper_second_caterer : ∃ x : ℕ, x = 34 ∧ x >= 30 ∧ cost_second_caterer x < cost_first_caterer x :=
by
  sorry

end least_people_cheaper_second_caterer_l778_778007


namespace parabola_vertex_on_x_axis_l778_778055

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  let a := 2
  let b := -10
  (∀ x : ℝ, (y = a * x ^ 2 + b * x + c) → vertex_y := a * (vertex_x) ^ 2 + b * vertex_x + c = 0) → 
  c = 12.5 :=
by sorry

end parabola_vertex_on_x_axis_l778_778055


namespace polygon_sides_l778_778745

theorem polygon_sides (h : 900 = interior_angles_sum + 360) (interior_angles_sum = (n-2) * 180) : n = 5 :=
by {
  sorry
}

end polygon_sides_l778_778745


namespace problem1_part1_problem1_part2_problem2_l778_778435

-- Define the function f_n
def f (n : ℕ) (x : ℝ) : ℝ := (1 + x)^n

-- Problem 1, Part (i)
theorem problem1_part1 (a : ℕ → ℝ) (h : f 2013 x = ∑ i in finset.range (2013 + 1), a i * x^i) :
  (∑ i in finset.range (2013 + 1), a i) = 2^(2013) :=
sorry

-- Problem 1, Part (ii)
theorem problem1_part2 (a : ℕ → ℝ) (h : f 2013 x = ∑ i in finset.range (2013 + 1), a i * x^i) :
  (∑ i in finset.range (2013 + 1) \ \{i | even i}, a i) = 2^(2012) :=
sorry

-- Problem 2
theorem problem2 (n : ℕ) (h : n > 0) (x : ℝ) (hx : |x| ≤ 1) : 
  f n x + f n (-x) ≤ 2^n :=
sorry

end problem1_part1_problem1_part2_problem2_l778_778435


namespace interest_rate_l778_778151

theorem interest_rate (SI P : ℝ) (T : ℕ) (h₁: SI = 70) (h₂ : P = 700) (h₃ : T = 4) : 
  (SI / (P * T)) * 100 = 2.5 :=
by
  sorry

end interest_rate_l778_778151


namespace values_of_n_l778_778674

theorem values_of_n : ∃ n_vals : Finset ℤ, n_vals.card = 9 ∧ ∀ n ∈ n_vals, (4000 * ((2:ℝ/ℚ/ℤ) / 5)^n : ℝ/ℚ/ℤ) ∈ ℤ :=
by 
  sorry

end values_of_n_l778_778674


namespace equidistance_EM_MF_l778_778239

variables (A B C O D E F M : Type) [triangle : Triangle A B C]
variables (O is_circumcenter : Circumcenter A B C O)
variables (D is_midpoint : Midpoint B C D)
variables (on_AB E : On_line_segment A B E)
variables (on_AC F : On_line_segment A C F)
variables (EF_circle_with_diameter_AD_same_points : ∃ circ : Circle AD, circ.makes_contact AB E ∧ circ.makes_contact AC F)
variables (M : Point D EF (parallel AO) (intersect line D EF))

theorem equidistance_EM_MF (h1 : AcuteTriangle A B C) (h2 : AB > AC)
  (h3 : is_circumcenter O) (h4 : is_midpoint D)
  (h5 : Circle_with_diameter A D intersects AB at E and AC at F)
  (h6 : line D M parallel AO meets EF at M) :
  Distance E M = Distance M F :=
sorry

end equidistance_EM_MF_l778_778239


namespace length_of_DE_l778_778013

theorem length_of_DE (AB AD DC : ℕ) (area_eq : AB * AD = (1/2) * DC * (2 * AD / AB)) 
(h_AD : AD = 6) (h_AB : AB = 5) (h_DC : DC = 5) :
    let CE := 2 * AD / AB in
    DE = 13 :=
by
  -- setup and initialization based on provided conditions
  let CE := 12
  let DE := Math.sqrt(DC^2 + CE^2)
  sorry

end length_of_DE_l778_778013


namespace remainder_sum_first_150_div_11300_l778_778085

theorem remainder_sum_first_150_div_11300 :
  let n := 150
  let S := n * (n + 1) / 2
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l778_778085


namespace ellipse_and_fixed_point_l778_778298

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l778_778298


namespace amount_c_l778_778533

theorem amount_c (a b c d : ℝ) :
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  a + b + c + d = 750 →
  c = 225 :=
by 
  intros h1 h2 h3 h4 h5
  -- Proof omitted.
  sorry

end amount_c_l778_778533


namespace greatest_possible_hcf_l778_778961

theorem greatest_possible_hcf :
  ∃ (reds blues : Finset ℕ), reds ∪ blues = Finset.range 15 ∧
  reds.Nonempty ∧ blues.Nonempty ∧
  let a := reds.sum (λ x, 2^x) in
  let b := blues.sum (λ x, 2^x) in
  a + b = 32767 ∧ 
  ∀ d, d ∣ a ∧ d ∣ b → d ≤ 4681 := 
sorry

end greatest_possible_hcf_l778_778961


namespace n_possible_values_and_solutions_l778_778954

theorem n_possible_values_and_solutions (n : ℕ) (x : Fin n → ℤ)
  (h_eq : (4 * n + 1) * (Finset.univ.sum (λ i, (x i) ^ 2) - 1) = 4 * (Finset.univ.sum (λ i, x i)) ^ 2) :
  (n = 2 ∧ (∃ i j, i ≠ j ∧ (x i = 1 ∧ x j = 2 ∨ x i = 2 ∧ x j = 1 ∨
                            x i = -1 ∧ x j = -2 ∨ x i = -2 ∧ x j = -1))) ∨
  (n = 6 ∧ (∃ y : Fin 6 → ℤ, perm y x ∧
             (∃ a b c d e f, y = ![0, a, b, c, d, e, f] ∧ a = 1 ∧ b = 1 ∧ c = 1 ∧
                            d = 1 ∧ e = 1 ∧ f = 0 ∨
                            a = -1 ∧ b = -1 ∧ c = -1 ∧
                            d = -1 ∧ e = -1 ∧ f = 0))) :=
sorry

end n_possible_values_and_solutions_l778_778954


namespace hn_passes_fixed_point_l778_778267

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l778_778267


namespace smallest_integer_to_make_1008_perfect_square_l778_778950

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_integer_to_make_1008_perfect_square : ∃ k : ℕ, k > 0 ∧ 
  (∀ m : ℕ, m > 0 → (is_perfect_square (1008 * m) → m ≥ k)) ∧ is_perfect_square (1008 * k) :=
by
  sorry

end smallest_integer_to_make_1008_perfect_square_l778_778950


namespace sum_of_proper_divisors_864_l778_778996

theorem sum_of_proper_divisors_864 : 
  let n := 864 in
  (∑ d in (finset.filter (λ x, x ∣ n ∧ x ≠ n) (finset.range (n+1))), d) = 1656 := 
by
  sorry

end sum_of_proper_divisors_864_l778_778996


namespace limit_fn_to_e_half_l778_778482

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
begin
  if n = 1 then exact x,
  else exact 2 * x ^ (n + 1) - x ^ n + (1 / 2) * ∫ (t : ℝ) in 0..1, f (n - 1) t,
end

theorem limit_fn_to_e_half :
  (lim_{n \to ∞} (f n (1 + 1 / (2 * n)))) = real.exp (1 / 2) :=
sorry

end limit_fn_to_e_half_l778_778482


namespace calculate_breakdown_time_l778_778566

noncomputable def bus_breakdown_time (S x a : ℕ) : Prop :=
  let t := (S : ℚ) / (5 * x) + 20 - (S - x * t) / (5 * x) in
  10 + t + (S - x * t) / (5 * x) = S / (5 * x) + 30 ∧
  a + 2 * (S - x * t) / (5 * x) = 2 * S / (5 * x) + 30

theorem calculate_breakdown_time (S x a : ℕ) : 
  bus_breakdown_time S x a := 
sorry

end calculate_breakdown_time_l778_778566


namespace rectangle_circle_area_ratio_l778_778038

theorem rectangle_circle_area_ratio (w r : ℝ) (h1 : 2 * 2 * w + 2 * w = 2 * pi * r) :
  ((2 * w) * w) / (pi * r^2) = 2 * pi / 9 :=
by
  sorry

end rectangle_circle_area_ratio_l778_778038


namespace total_books_l778_778173

theorem total_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total : ℕ) :
  books_per_shelf = 56 → 
  num_shelves = 9 → 
  total = 504 →
  books_per_shelf * num_shelves = total :=
by
  intros h1 h2 h3
  rw [h1, h2]
  exact h3

end total_books_l778_778173


namespace min_trips_theorem_l778_778068

noncomputable def min_trips (weights : List ℕ) (capacity : ℕ) (total_weight : ℕ) : ℕ :=
  if total_weight = 0 then 0
  else let weight_sum := List.sum weights
       if weight_sum <= capacity then 1
       else if weights.length = 1 then weights.length / capacity
       else (weight_sum + capacity - 1) / capacity

theorem min_trips_theorem (weights : List ℕ) (h_distinct : weights.nodup)
                          (h_max_weight : ∀ w ∈ weights, w ≤ 10)
                          (h_total_weight : List.sum weights = 400)
                          (h_multiple_of_quintal : ∀ w ∈ weights, w % 100 = 0) :
  min_trips weights 10 400 = 51 := 
sorry

end min_trips_theorem_l778_778068


namespace tan_2x_period_l778_778854

-- Define the tangent function and its properties
def tan := Real.tan

-- Define the problem
theorem tan_2x_period :
  (∃ P > 0, ∀ x, tan (2 * x) = tan (2 * x + P)) → P = π / 2 :=
by
  sorry

end tan_2x_period_l778_778854


namespace polynomial_evaluation_at_3_equals_698_l778_778412

noncomputable def q : ℕ := ∑ q_i in (list.of_fn (λ i, q_i 3)), q_i(3)

theorem polynomial_evaluation_at_3_equals_698
  (q_i : ℕ → ℤ) 
  (h1 : ∀ i, irreducible (q_i : polynomial ℤ))
  (h2 : ∀ i, monic (q_i : polynomial ℤ))
  (h3 : (polynomial.eval x q_i(x)) divides (x^6 - x^3 - x - 1)) :
  q = 698 :=
by
  sorry

end polynomial_evaluation_at_3_equals_698_l778_778412


namespace card_at_position_58_l778_778195

-- Define the pattern of the sequence
def card_pattern : List String := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

-- Function to find the nth card in the sequence
def nth_card (n : ℕ) : String :=
  card_pattern[(n % 13) - 1]

theorem card_at_position_58 : nth_card 58 = "6" := by
  -- skip the proof details
  sorry

end card_at_position_58_l778_778195
