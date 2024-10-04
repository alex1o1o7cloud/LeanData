import Mathlib
import Mathlib.Algebra.ArithmeticProg
import Mathlib.Algebra.ArithmeticSeq
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Opposite
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Integration
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.CombinatorialProbability
import Mathlib.Combinatorics.Equiv.Equiv
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Perm
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Init.Data.Nat.Lemmas
import Mathlib.Init.Function
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.LinearAlgebra.Polarize
import Mathlib.NumberTheory.Prime.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.∅

namespace find_pairs_l784_784214

theorem find_pairs :
  { (a, b) : ℕ × ℕ // (a + 1) % b = 0 ∧ (b + 1) % a = 0 } =
  { (1, 1), (1, 2), (2, 3) } :=
  sorry

end find_pairs_l784_784214


namespace lucas_100_mod_9_l784_784399

def lucas_seq : ℕ → ℕ
| 0     := 1
| 1     := 3
| (n+2) := lucas_seq n + lucas_seq (n+1)

theorem lucas_100_mod_9 : lucas_seq 100 % 9 = 7 :=
by {
  sorry
}

end lucas_100_mod_9_l784_784399


namespace find_common_ratio_of_geometric_sequence_l784_784762

theorem find_common_ratio_of_geometric_sequence 
  (a1 a2 a3 : ℝ) 
  (q : ℝ) 
  (h1 : a2 = a1 * q)
  (h2 : a3 = a1 * q^2)
  (h3 : 2 * a1, (3 / 2) * a2, a3 form_arithmetic_seq) :
  q = 1 ∨ q = 2 := 
sorry

-- For defining that three numbers form an arithmetic sequence
def form_arithmetic_seq (x y z : ℝ) : Prop :=
  2 * y = x + z

end find_common_ratio_of_geometric_sequence_l784_784762


namespace systematic_sampling_first_segment_l784_784479

theorem systematic_sampling_first_segment
  (total_students : ℕ)
  (sample_size : ℕ)
  (sampling_interval : ℕ)
  (sixty_th_segment : ℕ)
  (first_segment : ℕ)
  (h1 : total_students = 300)
  (h2 : sample_size = 60)
  (h3 : sampling_interval = total_students / sample_size)
  (h4 : sixty_th_segment = 298) 
  : first_segment = sixty_th_segment - (59 * sampling_interval) := 
  by
  have hsi : sampling_interval = 5 := by
    rw [h1, h2, h3]
  rw [hsi, h4]
  sorry

end systematic_sampling_first_segment_l784_784479


namespace value_of_expression_l784_784731

theorem value_of_expression (a b : ℝ) (h : -3 * a - b = -1) : 3 - 6 * a - 2 * b = 1 :=
by
  sorry

end value_of_expression_l784_784731


namespace find_phi_l784_784735

noncomputable theory

def f (x ϕ: ℝ) : ℝ := 2 * Real.sin (x + ϕ)

theorem find_phi :
  (∃ x y : ℝ, f x ϕ = 2 * f y ϕ) →
  (0 < ϕ) →
  (ϕ < π) →
  ϕ = π / 2 :=
by
  sorry

end find_phi_l784_784735


namespace expected_yield_correct_l784_784037

-- Conditions
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def step_length_ft : ℝ := 2.5
def yield_per_sqft_pounds : ℝ := 0.75

-- Related quantities
def garden_length_ft : ℝ := garden_length_steps * step_length_ft
def garden_width_ft : ℝ := garden_width_steps * step_length_ft
def garden_area_sqft : ℝ := garden_length_ft * garden_width_ft
def expected_yield_pounds : ℝ := garden_area_sqft * yield_per_sqft_pounds

-- Statement to prove
theorem expected_yield_correct : expected_yield_pounds = 2109.375 := by
  sorry

end expected_yield_correct_l784_784037


namespace original_employees_l784_784470

theorem original_employees (x : ℕ) (reduced_percent : ℝ) (remaining_employees: ℕ)
  (h1 : reduced_percent = 0.20)
  (h2 : remaining_employees = 195)
  (h3 : remaining_employees = x - reduced_percent * x) :
  x ≈ 244 :=
by
  sorry

end original_employees_l784_784470


namespace solve_inequality_l784_784142

variable (x : ℝ)

noncomputable def u := 1 + x^2
noncomputable def v := 1 - 3*x^2 + 36*(x^4)
noncomputable def w := 1 - 27*(x^5)

theorem solve_inequality :
  (Real.logBase (u x) (w x) + Real.logBase (v x) (u x) ≤ 1 + Real.logBase (v x) (w x)) ↔
  (x ∈ ({-1/3} ∪ Ioo (-1/(2*Real.sqrt 3)) 0 ∪ Ioo 0 (1/(2*Real.sqrt 3)) ∪ Icc (1/3) (1/Real.root 27 5))) :=
sorry

end solve_inequality_l784_784142


namespace mod_neg_result_l784_784547

-- Define the hypothesis as the residue equivalence and positive range constraint.
theorem mod_neg_result : 
  ∀ (a b : ℤ), (-1277 : ℤ) % 32 = 3 := by
  sorry

end mod_neg_result_l784_784547


namespace intersection_M_N_eq_neg2_l784_784639

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784639


namespace solution_set_correct_l784_784227

noncomputable def solution (x : ℝ) : Prop :=
  |x| * (1 - 2 * x) > 0

theorem solution_set_correct : 
  {x : ℝ | solution x} = {x : ℝ | x ∈ set.Ioo (-∞) 0 ∪ set.Ioo 0 (1/2)} :=
sorry

end solution_set_correct_l784_784227


namespace trigonometric_identity_in_second_quadrant_l784_784029

variable {α : Real}
-- Conditions
def second_quadrant (α : Real) := π / 2 < α ∧ α < π

-- Statement of the theorem
theorem trigonometric_identity_in_second_quadrant (h : second_quadrant α) :
  (sin α / cos α) * sqrt (1 / (sin α) ^ 2 - 1) = -1 :=
sorry

end trigonometric_identity_in_second_quadrant_l784_784029


namespace total_workers_calculation_l784_784403

theorem total_workers_calculation :
  ∀ (N : ℕ), 
  (∀ (total_avg_salary : ℕ) (techs_salary : ℕ) (nontech_avg_salary : ℕ),
    total_avg_salary = 8000 → 
    techs_salary = 7 * 20000 → 
    nontech_avg_salary = 6000 →
    8000 * (7 + N) = 7 * 20000 + N * 6000 →
    (7 + N) = 49) :=
by
  intros
  sorry

end total_workers_calculation_l784_784403


namespace probability_of_intersecting_diagonals_l784_784446

noncomputable def intersecting_diagonals_probability : ℚ :=
let total_vertices := 8 in
let total_pairs := Nat.choose total_vertices 2 in
let total_sides := 8 in
let total_diagonals := total_pairs - total_sides in
let total_pairs_diagonals := Nat.choose total_diagonals 2 in
let intersecting_diagonals := Nat.choose total_vertices 4 in
(intersecting_diagonals : ℚ) / (total_pairs_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  intersecting_diagonals_probability = 7 / 19 :=
by
  sorry

end probability_of_intersecting_diagonals_l784_784446


namespace total_sum_alternating_sums_l784_784235

def alternating_sum (s : Finset ℕ) : ℤ :=
  let l := s.sort (by exact λ x y => x ≥ y)
  l.foldr (λ x acc => x - acc) 0

noncomputable def S_n : ℕ → ℤ :=
  λ n => ∑ s in (Finset.powerset (Finset.range n)).filter (λ x => x.nonempty),
         alternating_sum s

theorem total_sum_alternating_sums (n : ℕ) (h : n > 0) : S_n n = n * 2^(n-1) := sorry

end total_sum_alternating_sums_l784_784235


namespace ellipse_locus_and_min_distance_l784_784711

theorem ellipse_locus_and_min_distance (B C : ℝ × ℝ)
  (A : ℝ × ℝ)
  (dist_AB : ℝ) 
  (dist_AC : ℝ) 
  (hB : B = (-1, 0)) 
  (hC : C = (1, 0)) 
  (h_dist : dist_AB + dist_AC = 4) 
  (Q : ℝ × ℝ) 
  (d : ℝ) 
  (O₁ : ℝ × ℝ) 
  (hQ_on_ellipse : (Q.1^2 / 4) + (Q.2^2 / 3) = 1 ∧ Q.2 ≠ 0) 
  (h_tangent : d = abs (3 / (2 * Q.2) - Q.2 / 6)) : 
  (A.1^2 / 4) + (A.2^2 / 3) = 1 ∧ A.2 ≠ 0 ∧ d = sqrt 3 / 3 :=
sorry

end ellipse_locus_and_min_distance_l784_784711


namespace intersection_M_N_eq_neg2_l784_784637

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784637


namespace find_range_m_l784_784758

-- Define the conditions
def is_on_circle (x y : ℝ) := x^2 + y^2 = 1
def is_on_line (x y m : ℝ) := x + (real.sqrt 3) * y + m = 0
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def PA (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
def PB (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Statement of the problem
theorem find_range_m (m : ℝ) : (∃ P : ℝ × ℝ, is_on_line P.1 P.2 m ∧ PA P = 2 * PB P) ↔ (-13 / 3 ≤ m ∧ m ≤ 1) := 
sorry

end find_range_m_l784_784758


namespace f_f_neg_one_l784_784271

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + x - 1 else -((abs x)^2 + abs x - 1)

theorem f_f_neg_one : ∀ {f : ℝ → ℝ} (h_odd : ∀ x : ℝ, f (-x) = -f x)
    (h_pos : ∀ x : ℝ, 0 < x → f x = x^2 + x - 1), f (f (-1)) = -1 :=
by
  intro f h_odd h_pos
  have h_f1 : f 1 = 1 :=
    by
      exact h_pos 1 zero_lt_one
  have h_f_minus1 : f (-1) = -1 :=
    by
      calc f (-1) = -f 1 : h_odd 1
               ... = -1 : by rw [h_f1]
  calc f (f (-1)) = f (-1) : by rw [h_f_minus1]
               ... = -1 : by rw [h_f_minus1]

end f_f_neg_one_l784_784271


namespace hyperbola_standard_equation_l784_784694

-- Define the hyperbola conditions and required proof
theorem hyperbola_standard_equation
  (asymptotic : ∀ x, ∃ y : ℝ, y = x / 2 ∨ y = -x / 2)
  (focal_length : ∃ c : ℝ, ∀ c', c = real.sqrt 5 ∧ 2 * c = 2 * real.sqrt 5):
  (∃ m : ℝ, (m = 1 ∨ m = -1) ∧ (∀ x y : ℝ, (x^2 / (4 * m) - y^2 / m = 1) ∨ (y^2 / m - x^2 / (4 * m) = 1))) := 
begin
  sorry
end

end hyperbola_standard_equation_l784_784694


namespace t_minus_s_l784_784161

-- Define the number of students and teachers
def num_students : ℕ := 120
def num_teachers : ℕ := 4

-- Define the class sizes
def class_sizes : List ℕ := [60, 30, 20, 10]

-- Define t as the average number of students in a class when a teacher is randomly selected
def t : ℝ :=
  (class_sizes.map (λ n, (n : ℝ) * (1 / num_teachers))).sum

-- Define s as the average number of students in the class of a randomly selected student
def s : ℝ :=
  (class_sizes.map (λ n, (n : ℝ) * ((n : ℝ) / num_students))).sum

-- Prove that t - s = -11.663
theorem t_minus_s : t - s = -11.663 := by
  sorry

end t_minus_s_l784_784161


namespace rationalize_denominator_l784_784065

/-- Rationalizing the denominator of an expression involving cube roots -/
theorem rationalize_denominator :
  (1 : ℝ) / (real.cbrt 3 + real.cbrt 27) = real.cbrt 9 / (12 : ℝ) :=
by
  -- Define conditions
  have h1 : real.cbrt 27 = 3 * real.cbrt 3, by sorry,
  -- Proof of the equality, skipped using sorry
  sorry

end rationalize_denominator_l784_784065


namespace power_function_passes_through_point_l784_784740

theorem power_function_passes_through_point (a : ℝ) : (2 ^ a = Real.sqrt 2) → (a = 1 / 2) :=
  by
  intro h
  sorry

end power_function_passes_through_point_l784_784740


namespace intersection_M_N_l784_784648

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784648


namespace moles_of_water_formed_l784_784301

-- This is given as part of the problem conditions.
def balanced_reaction :=
  ∀ (NaHCO₃ HCl NaCl CO₂ H₂O : ℝ), NaHCO₃ + HCl = NaCl + CO₂ + H₂O

theorem moles_of_water_formed (moles_NaHCO₃ moles_HCl : ℝ) :
  -- Given conditions
  moles_NaHCO₃ = 2 → 
  moles_HCl = 2 → 
  balanced_reaction NaHCO₃ HCl NaCl CO₂ H₂O → 
  -- To prove
  H₂O = 2 :=
sorry

end moles_of_water_formed_l784_784301


namespace smallest_k_for_p_cubed_l784_784781

theorem smallest_k_for_p_cubed (p : ℕ) (h1 : nat.prime p)
  (h2 : 1007 = nat.digits 10 p) : ∃ k : ℕ, k = 1 ∧ (p^3 - k) % 24 = 0 :=
by
  sorry

end smallest_k_for_p_cubed_l784_784781


namespace side_and_diagonal_incommensurable_l784_784448

theorem side_and_diagonal_incommensurable (a : ℝ) :
  let d := a * Real.sqrt 2 in
  ∀ m n : ℕ, m • a ≠ n • d := 
begin
  sorry,
end

end side_and_diagonal_incommensurable_l784_784448


namespace bullets_balance_diamonds_l784_784878

-- Definitions for the problem
def Delta := ℝ
def Diamond := ℝ
def Bullet := ℝ

-- Conditions given in the problem
axiom condition1 (a b c : ℝ) : 2 * a + 3 * b = 12 * c
axiom condition2 (a b c : ℝ) : a = 3 * b + 2 * c

-- Existence of bullets that balance four diamonds
theorem bullets_balance_diamonds (a b c : ℝ) 
  (h1 : 2 * a + 3 * b = 12 * c) 
  (h2 : a = 3 * b + 2 * c) : 4 * b = 4 * c :=
begin
  sorry
end

end bullets_balance_diamonds_l784_784878


namespace incoming_class_student_count_l784_784939

theorem incoming_class_student_count (n : ℕ) :
  n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 → n = 418 :=
by
  sorry

end incoming_class_student_count_l784_784939


namespace range_of_f_l784_784792

def f (x : ℝ) : ℝ := |x| / (1 + |x|)

theorem range_of_f (x : ℝ) : (1 / 3) < x ∧ x < 1 → f x > f (2 * x - 1) := by
  sorry

end range_of_f_l784_784792


namespace basketball_player_possible_scores_l784_784925

/-- A basketball player made 7 baskets during a game. Each basket was worth either 2 or 3 points.
    Prove that the number of different total points scored by the player is 8. -/
theorem basketball_player_possible_scores :
  (∀ n, (14 ≤ n ∧ n ≤ 21) →
  ∃ k m, (k + m = 7) ∧ (k * 3 + m * 2 = n)) ∧
  (∀ n, (14 ≤ n ∧ n ≤ 21) → (∃! m : ℕ, m ∈ {14, 15, 16, 17, 18, 19, 20, 21}))
  :=
sorry

end basketball_player_possible_scores_l784_784925


namespace incenter_locus_l784_784971

/-- Let B and C be fixed points on a circle. A is a variable point on the circle.
    If I is the incenter of triangle ABC, the locus of I as A varies is the arc of the circle bounded by B and C. -/
theorem incenter_locus (B C : ℂ) (circle : set ℂ) (hB : B ∈ circle) (hC : C ∈ circle) :
  ∀ (A : ℂ), A ∈ circle →
  ∃ I : ℂ, I is_incenter_of_triangle A B C ∧ I ∈ locus_of_incenter B C :=
sorry

end incenter_locus_l784_784971


namespace M_inter_N_eq_neg2_l784_784661

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784661


namespace find_sum_of_squares_l784_784997

-- Define the matrix N
def N (x y w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![0, 3*y, w],
     ![x, 2*y, -w],
     ![x, -2*y, w] ]

-- Define the condition Nᵀ * N = I
noncomputable def N_transpose_mul_N (x y w : ℝ) : Prop :=
  (N x y w)ᵀ ⬝ (N x y w) = 1

-- State the theorem to prove
theorem find_sum_of_squares (x y w : ℝ) (h : N_transpose_mul_N x y w) :
  x^2 + y^2 + w^2 = 1.372549 :=
sorry

end find_sum_of_squares_l784_784997


namespace parabola_integer_solutions_l784_784098

theorem parabola_integer_solutions :
  let Q := { p : ℝ × ℝ | ∃ a, a = 5 / 9 ∧ p.2 = a * (p.1)^2 - 1 } in
  set.countable { p : ℤ × ℤ | (p.1 : ℝ, p.2 : ℝ) ∈ Q ∧ abs (3 * p.1 - 4 * p.2) ≤ 1200 } = 47 :=
begin
  sorry
end

end parabola_integer_solutions_l784_784098


namespace pool_capacity_l784_784511

theorem pool_capacity
  (pump_removes : ∀ (x : ℝ), x > 0 → (2 / 3) * x / 7.5 = (4 / 15) * x)
  (working_time : 0.15 * 60 = 9)
  (remaining_water : ∀ (x : ℝ), x > 0 → x - (0.8 * x) = 25) :
  ∃ x : ℝ, x = 125 :=
by
  sorry

end pool_capacity_l784_784511


namespace horner_rule_operations_l784_784122

noncomputable def polynomial := λ x : ℝ, 3 * x^5 + x^2 - x + 2

theorem horner_rule_operations :
  let x := -2 in
  let multiplications := 5 in
  let additions := 3 in
  (mult_operations polynomial x = multiplications) ∧ (add_operations polynomial x = additions) :=
sorry

end horner_rule_operations_l784_784122


namespace xiaoliang_steps_l784_784136

/-- 
  Xiaoping lives on the fifth floor and climbs 80 steps to get home every day.
  Xiaoliang lives on the fourth floor.
  Prove that the number of steps Xiaoliang has to climb is 60.
-/
theorem xiaoliang_steps (steps_per_floor : ℕ) (h_xiaoping : 4 * steps_per_floor = 80) : 3 * steps_per_floor = 60 :=
by {
  -- The proof is intentionally left out
  sorry
}

end xiaoliang_steps_l784_784136


namespace number_of_pairs_l784_784717

theorem number_of_pairs (m n : ℕ) (h1 : m ≤ 1000) (h2 : n ≤ 1000) :
    (∑ m n, if m ≤ 1000 ∧ n ≤ 1000 ∧ (m / (n + 1) : ℝ) < real.sqrt 2 ∧ (real.sqrt 2 < (m + 1) / n) then 1 else 0) = 1706 := sorry

end number_of_pairs_l784_784717


namespace algebraic_expression_value_l784_784254

variables (x y : ℝ)

theorem algebraic_expression_value :
  x^2 - 4 * x - 1 = 0 →
  (2 * x - 3)^2 - (x + y) * (x - y) - y^2 = 12 :=
by
  intro h
  sorry

end algebraic_expression_value_l784_784254


namespace least_num_of_cans_l784_784933

theorem least_num_of_cans (Maaza Pepsi Sprite Fanta 7UP : ℕ) 
  (h1 : Maaza = 60) 
  (h2 : Pepsi = 220) 
  (h3 : Sprite = 500) 
  (h4 : Fanta = 315) 
  (h5 : 7UP = 125) 
  : (Maaza / Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd Maaza Pepsi) Sprite) Fanta) 7UP) + 
    (Pepsi / Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd Maaza Pepsi) Sprite) Fanta) 7UP) + 
    (Sprite / Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd Maaza Pepsi) Sprite) Fanta) 7UP) + 
    (Fanta / Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd Maaza Pepsi) Sprite) Fanta) 7UP) + 
    (7UP / Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd Maaza Pepsi) Sprite) Fanta) 7UP) = 244 := 
by
  sorry

end least_num_of_cans_l784_784933


namespace number_of_boys_l784_784078

theorem number_of_boys (T : ℕ) (hT : T = 200): (0.30 * T = 60) :=
by
  -- Proof skipped
  sorry

end number_of_boys_l784_784078


namespace ratio_of_triangle_areas_l784_784466

open EuclideanGeometry

theorem ratio_of_triangle_areas 
  (ABC : Triangle)
  (A B C : Point)
  (hABC : is_right_triangle ABC A B C)
  (hA : angle A = 30)
  (O : Circle)
  (hABC_circumcircle : is_circumcircle O ABC)
  (ω1 ω2 ω3 : Circle)
  (T1 T2 T3 : Point)
  (S1 S2 S3 : Point)
  (hT1_tangent : tangent_point ω1 O T1)
  (hT2_tangent : tangent_point ω2 O T2)
  (hT3_tangent : tangent_point ω3 O T3)
  (hS1_tangent : tangent_point ω1 (line_through A B) S1)
  (hS2_tangent : tangent_point ω2 (line_through B C) S2)
  (hS3_tangent : tangent_point ω3 (line_through C A) S3)
  (T1S1 : Line := line_through T1 S1)
  (T2S2 : Line := line_through T2 S2)
  (T3S3 : Line := line_through T3 S3)
  (A' B' C' : Point)
  (hA' : T1S1 ∩ O = A')
  (hB' : T2S2 ∩ O = B')
  (hC' : T3S3 ∩ O = C') :
  let area_ABC := area_of_triangle ABC in
  let area_A'B'C' := area_of_triangle (triangle.mk A' B' C') in
  area_A'B'C' / area_ABC = (sqrt 3 + 1) / 2 :=
sorry

end ratio_of_triangle_areas_l784_784466


namespace convex_iff_mean_inequality_l784_784049

variables {α : Type*} [linear_ordered_field α] {a b : α} 
  (f : α → α) (n : ℕ) (x : fin n → α)

noncomputable def mean (xs : fin n → α) : α :=
  (finset.univ.sum xs) / n

def convex_on (f : α → α) (a b : α) :=
  ∀ ⦃x y⦄ (t : α), a ≤ x → x ≤ b → a ≤ y → y ≤ b →
  0 ≤ t → t ≤ 1 → f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

theorem convex_iff_mean_inequality :
  convex_on f a b ↔ (∀ (n : ℕ) (x : fin n → α), 
    (∀ i, a ≤ x i ∧ x i ≤ b) → 
    f (mean n x) ≤ (finset.univ.sum (λ i, f (x i))) / n) :=
sorry

end convex_iff_mean_inequality_l784_784049


namespace true_discount_is_90_l784_784106

-- Definitions from the conditions
def FV := 540 -- Face value
def BD := 108 -- Banker's discount

-- Problem Statement: Prove TD = 90 given the conditions
theorem true_discount_is_90 (TD : ℝ) : BD = TD + (TD * BD / FV) → TD = 90 :=
by
  -- Given BD = TD + (TD * BD / FV)
  assume h : BD = TD + (TD * BD / FV)
  -- We need to show TD = 90
  sorry

end true_discount_is_90_l784_784106


namespace exists_m_in_interval_l784_784544

noncomputable const x : ℕ → ℝ
def x₀ : ℝ := 5
def f (xn : ℝ) : ℝ := (xn^2 + 5 * xn + 4) / (xn + 6)
axiom x_rec : ∀ n : ℕ, x (n + 1) = f (x n)

theorem exists_m_in_interval : ∃ m : ℕ, m > 0 ∧ x m ≤ 4 + 1 / 2^10 ∧ 19 ≤ m ∧ m ≤ 60 :=
by
  -- proof goes here
  sorry

end exists_m_in_interval_l784_784544


namespace sufficient_but_not_necessary_l784_784088

variable (x : ℚ)

def is_integer (n : ℚ) : Prop := ∃ (k : ℤ), n = k

theorem sufficient_but_not_necessary :
  (is_integer x → is_integer (2 * x + 1)) ∧
  (¬ (is_integer (2 * x + 1) → is_integer x)) :=
by
  sorry

end sufficient_but_not_necessary_l784_784088


namespace complement_intersection_l784_784296

-- Definitions of the sets as given in the problem
namespace ProofProblem

def U : Set ℤ := {-2, -1, 0, 1, 2}
def M : Set ℤ := {y | y > 0}
def N : Set ℤ := {x | x = -1 ∨ x = 2}

theorem complement_intersection :
  (U \ M) ∩ N = {-1} :=
by
  sorry

end ProofProblem

end complement_intersection_l784_784296


namespace arithmetic_sequence_angles_sum_l784_784010

theorem arithmetic_sequence_angles_sum (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : 2 * B = A + C) :
  A + C = 120 :=
by
  sorry

end arithmetic_sequence_angles_sum_l784_784010


namespace letters_in_small_envelopes_l784_784175

theorem letters_in_small_envelopes (total_letters : ℕ) (large_envelopes : ℕ) (letters_per_large_envelope : ℕ) (letters_in_small_envelopes : ℕ) :
  total_letters = 80 →
  large_envelopes = 30 →
  letters_per_large_envelope = 2 →
  letters_in_small_envelopes = total_letters - (large_envelopes * letters_per_large_envelope) →
  letters_in_small_envelopes = 20 :=
by
  intros ht hl he hs
  rw [ht, hl, he] at hs
  exact hs

#check letters_in_small_envelopes

end letters_in_small_envelopes_l784_784175


namespace b_share_is_1200_l784_784905

theorem b_share_is_1200 (total_money : ℝ) (ra rb rc : ℝ) (ratio_cond : ra / rb = 2 / 3 ∧ ra / rc = 2 / 4) :
  rb * (total_money / (ra + rb + rc)) = 1200 :=
by
  -- Given values in the problem
  let ha : ℝ := 2
  let hb : ℝ := 3
  let hc : ℝ := 4
  let tm : ℝ := 3600
  -- Calculate the shares
  have total_parts_eq : ha + hb + hc = 9 := by norm_num
  have each_part_value_eq : tm / 9 = 400 := by norm_num
  have b_share_value_eq : 400 * 3 = 1200 := by norm_num
  -- Provide the final proof step
  exact b_share_value_eq

end b_share_is_1200_l784_784905


namespace bowling_average_decrease_l784_784502

-- Define the initial conditions.
def initial_average : ℝ := 12.4
def initial_wickets : ℝ := 54.99999999999995 -- approximately 55
def additional_wickets_last_match : ℝ := 4
def runs_last_match : ℝ := 26

-- Calculate the total runs given before the last match.
def total_runs_before_last_match : ℝ := initial_average * initial_wickets

-- Calculate the total runs and total wickets after the last match.
def total_wickets_after_last_match : ℝ := initial_wickets + additional_wickets_last_match
def total_runs_after_last_match : ℝ := total_runs_before_last_match + runs_last_match

-- Define the new average after the last match.
def new_average : ℝ := total_runs_after_last_match / total_wickets_after_last_match

-- Define the decrease in the average.
def average_decrease : ℝ := initial_average - new_average

-- The theorem we want to prove: the average decreased by 0.4.
theorem bowling_average_decrease : average_decrease = 0.4 := by
  -- Add the proof here
  sorry

end bowling_average_decrease_l784_784502


namespace Q_locus_is_circle_l784_784356

-- Define the conditions and the problem
variables {A B C D P E F Q : Type} [cyclic_quadrilateral A B C D] (AD_eq_BD : AD = BD) (BD_eq_AC : BD = AC)
          (on_circumcircle : ∀ P, P ∈ circumcircle A B C D)
          (AP_CD : ∀ E, line A P ∩ line C D = {E})
          (DP_AB : ∀ F, line D P ∩ line A B = {F})
          (BE_CF : ∀ Q, line B E ∩ line C F = {Q})

-- Define the locus of Q and prove it is a circle
def locus_of_Q_is_circle : Type := 
  ∃ circle, ∀ (P : circumcircle A B C D), Q ∈ circle

-- State the main theorem
theorem Q_locus_is_circle :
  locus_of_Q_is_circle A B C D P E F Q AD_eq_BD BD_eq_AC on_circumcircle AP_CD DP_AB BE_CF :=
sorry

end Q_locus_is_circle_l784_784356


namespace apple_tree_yield_l784_784336

/-- In the orchard of the Grange Poser farm, there are 30 apple trees that each give a certain 
amount of apples and 45 peach trees that each produce an average of 65 kg of fruit. The total 
mass of fruit harvested in this orchard is 7425 kg. Prove that each apple tree gives 150 kg of apples. -/
theorem apple_tree_yield :
  ∃ A : ℝ, (30 * A + 45 * 65 = 7425) ↔ (A = 150) := 
begin
  use 150,
  split,
  {
    intro h,
    calc
      30 * 150 + 45 * 65 = 4500 + 2925 : by ring
      ... = 7425 : by ring
  },
  {
    intro h,
    rw h,
    ring
  }
end

end apple_tree_yield_l784_784336


namespace train_crossing_time_l784_784720

noncomputable def length_of_train := 350 -- meters
noncomputable def length_of_bridge := 500 -- meters
noncomputable def speed_of_train_kmph := 24 -- kilometers per hour (kmph)

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600) -- conversion to meters per second (m/s)

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge -- total distance to cross

noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps -- time it takes to cross the bridge

theorem train_crossing_time : time_to_cross_bridge ≈ 127.45 :=
by
  sorry

end train_crossing_time_l784_784720


namespace circle_radius_of_complex_roots_radius_of_circle_of_roots_l784_784523

theorem circle_radius_of_complex_roots (z : ℂ) 
  (h : (z + 1)^5 = 32 * z^5) : abs (z + 1) = 2 := by
  -- We are skipping the proof here
  sorry

theorem radius_of_circle_of_roots (radius : ℝ) 
  (h : ∀ z : ℂ, (z + 1)^5 = 32 * z^5 → abs (z + 1) = 2 * abs z) : 
  radius = 2 / 3 := by
  -- Proof which relies on the given condition
  sorry

end circle_radius_of_complex_roots_radius_of_circle_of_roots_l784_784523


namespace sqrt_solution_l784_784728

theorem sqrt_solution (x : ℝ) : sqrt (5 + sqrt x) = 4 → x = 121 :=
by sorry

end sqrt_solution_l784_784728


namespace find_x1_l784_784022

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 :=
  sorry

end find_x1_l784_784022


namespace digit_in_decimal_expansion_l784_784124

theorem digit_in_decimal_expansion (n : ℕ) : 
  let repetend := [1, 1, 3, 3, 3, 3]
  let period := 6
  let decimal_repr := list.take 2 (list.repeat 0) ++ list.cycle repetend
  n = 150 → 
  decimal_repr.get_or_else (n + 2 - 1) 0 = 3 := 
by
  -- Definitions for contextual clarity
  let repetend := [1, 1, 3, 3, 3, 3]
  let period := 6
  let decimal_repr := list.take 2 (list.repeat 0) ++ list.cycle repetend
  -- Given the value of n
  intro h
  -- Simplifying the 150th digit calculation
  have hmod : (n + 2 - 1 - 2) % period = 148 % period := by simp [h]
  have hmod_val : 148 % period = 4 := by norm_num
  show decimal_repr.get_or_else 149 0 = 3
  sorry

end digit_in_decimal_expansion_l784_784124


namespace marbles_after_adjustment_l784_784436

/-- Ben has 56 marbles --/
def Ben_marbles : ℕ := 56

/-- Leo has 20 more marbles than Ben --/
def Leo_marbles : ℕ := Ben_marbles + 20

/-- Tim has 15 fewer marbles than Leo --/
def Tim_marbles : ℕ := Leo_marbles - 15

/-- Total marbles before any adjustments --/
def Total_marbles : ℕ := Ben_marbles + Leo_marbles + Tim_marbles

/-- The number of marbles after adjustment to make it divisible by 5 will be 195 --/
theorem marbles_after_adjustment : (∃ (n : ℕ), (n + Total_marbles) % 5 = 0 ∧ n ≤ 5) :=
by
  unfold Total_marbles
  unfold Ben_marbles
  unfold Leo_marbles
  unfold Tim_marbles
  exists 2
  simp
  sorry

end marbles_after_adjustment_l784_784436


namespace catriona_total_fish_eq_44_l784_784988

-- Definitions based on conditions
def goldfish : ℕ := 8
def angelfish : ℕ := goldfish + 4
def guppies : ℕ := 2 * angelfish
def total_fish : ℕ := goldfish + angelfish + guppies

-- The theorem we need to prove
theorem catriona_total_fish_eq_44 : total_fish = 44 :=
by
  -- We are skipping the proof steps with 'sorry' for now
  sorry

end catriona_total_fish_eq_44_l784_784988


namespace find_coordinates_of_point_C_l784_784766

noncomputable theory
open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 8⟩
def M : Point := ⟨4, 11⟩
def L : Point := ⟨6, 6⟩

def is_coordinates_of_point_C (C : Point) : Prop :=
  C = ⟨14, 2⟩

-- We state the theorem that given A, M, L, the coordinates of C are (14, 2)
theorem find_coordinates_of_point_C : ∃ C : Point, is_coordinates_of_point_C C :=
  sorry

end find_coordinates_of_point_C_l784_784766


namespace no_ordered_quadruples_l784_784223

open Matrix

noncomputable def invertible_matrix (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
![
  ![a, b],
  ![c, d]
]

noncomputable def inverse_matrix (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
![
  ![$(1 : ℝ) / a, $(1 : ℝ) / b],
  ![$(1 : ℝ) / c, $(1 : ℝ) / d]
]

theorem no_ordered_quadruples (a b c d : ℝ) :
  invertible_matrix a b c d ≠ inverse_matrix a b c d := by
  sorry

end no_ordered_quadruples_l784_784223


namespace man_speed_proof_l784_784167

noncomputable def train_length : ℝ := 150 
noncomputable def crossing_time : ℝ := 6 
noncomputable def train_speed_kmph : ℝ := 84.99280057595394 
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def relative_speed_mps : ℝ := train_length / crossing_time
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps
noncomputable def man_speed_kmph : ℝ := man_speed_mps * (3600 / 1000)

theorem man_speed_proof : man_speed_kmph = 5.007198224048459 := by 
  sorry

end man_speed_proof_l784_784167


namespace middle_integer_of_consecutive_odd_l784_784428

theorem middle_integer_of_consecutive_odd (n : ℕ)
  (h1 : n > 2)
  (h2 : n < 8)
  (h3 : (n-2) % 2 = 1)
  (h4 : n % 2 = 1)
  (h5 : (n+2) % 2 = 1)
  (h6 : (n-2) + n + (n+2) = (n-2) * n * (n+2) / 9) :
  n = 5 :=
by sorry

end middle_integer_of_consecutive_odd_l784_784428


namespace number_of_squirrels_l784_784077

/-
Problem: Some squirrels collected 575 acorns. If each squirrel needs 130 acorns to get through the winter, each squirrel needs to collect 15 more acorns. 
Question: How many squirrels are there?
Conditions:
 1. Some squirrels collected 575 acorns.
 2. Each squirrel needs 130 acorns to get through the winter.
 3. Each squirrel needs to collect 15 more acorns.
Answer: 5 squirrels
-/

theorem number_of_squirrels (acorns_total : ℕ) (acorns_needed : ℕ) (acorns_short : ℕ) (S : ℕ)
  (h1 : acorns_total = 575)
  (h2 : acorns_needed = 130)
  (h3 : acorns_short = 15)
  (h4 : S * (acorns_needed - acorns_short) = acorns_total) :
  S = 5 :=
by
  sorry

end number_of_squirrels_l784_784077


namespace rationalize_denominator_l784_784063

/-- Rationalizing the denominator of an expression involving cube roots -/
theorem rationalize_denominator :
  (1 : ℝ) / (real.cbrt 3 + real.cbrt 27) = real.cbrt 9 / (12 : ℝ) :=
by
  -- Define conditions
  have h1 : real.cbrt 27 = 3 * real.cbrt 3, by sorry,
  -- Proof of the equality, skipped using sorry
  sorry

end rationalize_denominator_l784_784063


namespace two_digit_numbers_l784_784172

open Finset

theorem two_digit_numbers (n : ℕ) (hn : 10 ≤ n ∧ n < 100) :
  ∃ ns : Finset ℕ, (∀ x ∈ ns, 
    let a := x / 10 in
    let b := x % 10 in
    a ≠ 0 ∧ b ≥ a) ∧ 
  ns.card = 45 :=
by
  sorry

end two_digit_numbers_l784_784172


namespace mike_total_work_hours_l784_784797

theorem mike_total_work_hours (h d total_hours : ℕ) (h_given : h = 3) (d_given : d = 5) (total_hours_given : total_hours = h * d) :
  total_hours = 15 :=
by {
  rw [h_given, d_given] at total_hours_given,
  exact total_hours_given,
  sorry
}

end mike_total_work_hours_l784_784797


namespace rain_probability_weekend_l784_784843

theorem rain_probability_weekend :
  let p_rain_F := 0.60
  let p_rain_S := 0.70
  let p_rain_U := 0.40
  let p_no_rain_F := 1 - p_rain_F
  let p_no_rain_S := 1 - p_rain_S
  let p_no_rain_U := 1 - p_rain_U
  let p_no_rain_all_days := p_no_rain_F * p_no_rain_S * p_no_rain_U
  let p_rain_at_least_one_day := 1 - p_no_rain_all_days
  (p_rain_at_least_one_day * 100 = 92.8) := sorry

end rain_probability_weekend_l784_784843


namespace train_length_l784_784168

theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ) :
  speed_kmh = 45 → cross_time_s = 30 → bridge_length_m = 227.03 →
  let speed_mps := speed_kmh * (1000 / 3600),
  let distance_crossed_m := speed_mps * cross_time_s,
  let train_length := distance_crossed_m - bridge_length_m in
  train_length = 147.97 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end train_length_l784_784168


namespace curve_C1_cartesian_curve_C2_cartesian_intersection_AB_distance_l784_784756

variables (α : ℝ) (ρ θ : ℝ)

-- Definitions from conditions
def C1_parametric_x (α : ℝ) : ℝ := cos α
def C1_parametric_y (α : ℝ) : ℝ := 1 + sin α
def C2_polar (ρ θ : ℝ) : Prop := ρ * sin (θ - π / 4) = √2

theorem curve_C1_cartesian (x y : ℝ) (α : ℝ) :
  (x = cos α) → (y = 1 + sin α) → (x^2 + (y-1)^2 = 1) :=
sorry

theorem curve_C2_cartesian (x y : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ - π / 4) = √2) → (x = ρ * cos θ) → (y = ρ * sin θ) → (x - y + 2 = 0) :=
sorry

theorem intersection_AB_distance :
  ∀ (A B : ℝ × ℝ),
    (x = cos α) ∧ (y = 1 + sin α) ∧ (ρ * sin (θ - π / 4) = √2) ∧
    (C1_parametric_x A.1 = cos α) ∧ (C1_parametric_y A.2 = 1 + sin α) ∧
    (C1_parametric_x B.1 = cos α) ∧ (C1_parametric_y B.2 = 1 + sin α) ∧
    (ρ * sin (θ - π / 4) = √2) ∧ (x = ρ * cos θ) ∧ (y = ρ * sin θ) →
  (dist A B = √2) :=
sorry

end curve_C1_cartesian_curve_C2_cartesian_intersection_AB_distance_l784_784756


namespace sum_of_extreme_T_l784_784238

theorem sum_of_extreme_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022)
  (h2 : B + M + T = 72) :
  ∃ Tmin Tmax, Tmin + Tmax = 48 ∧ Tmin ≤ T ∧ T ≤ Tmax :=
by
  sorry

end sum_of_extreme_T_l784_784238


namespace part_a_contradiction_l784_784477

theorem part_a_contradiction :
  ¬ (225 / 25 + 75 = 100 - 16 → 25 * (9 / (1 + 3)) = 84) :=
by
  sorry

end part_a_contradiction_l784_784477


namespace weaving_problem_l784_784317

theorem weaving_problem
  (a : ℕ → ℝ) -- the sequence
  (a_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0)) -- arithmetic sequence condition
  (sum_seven_days : 7 * a 0 + 21 * (a 1 - a 0) = 21) -- sum in seven days
  (sum_days_2_5_8 : 3 * a 1 + 12 * (a 1 - a 0) = 15) -- sum on 2nd, 5th, and 8th days
  : a 10 = 15 := sorry

end weaving_problem_l784_784317


namespace x_intercept_perpendicular_line_l784_784452

theorem x_intercept_perpendicular_line 
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 12)
  (h2 : y = - (3 / 4) * x + 4)
  : x = 16 / 3 := 
sorry

end x_intercept_perpendicular_line_l784_784452


namespace geometric_sequence_an_arithmetic_sequence_tn_l784_784427

-- For Question 1
theorem geometric_sequence_an (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a n = 1 * (1 / 2) ^ (n - 1))
  (h2 : ∀ n, S n = ∑ i in range n, a i)
  (arith_seq_cond : ∀ n, (1 / a 1, 1 / a 3, 1 / a 4 - 1) form_arithmetic_sequence) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
sorry

-- For Question 2
theorem arithmetic_sequence_tn (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a n = 1 + (n - 1) * 1)
  (h2 : S 1 + a 2 = 3)
  (h3 : S 2 + a 3 = 6)
  (h4 : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * d) :
  T n = ∑ i in range n, 1 / S i = 2 * n / (n + 1) :=
sorry

end geometric_sequence_an_arithmetic_sequence_tn_l784_784427


namespace problem_binom_coeffs_l784_784331

theorem problem_binom_coeffs : 
  let a := Nat.choose 10 5 in
  let b := Nat.choose 10 3 * (-2)^3 in
  a = 252 ∧ b = -960 → b / a = -80 / 21 :=
by
sorrry

end problem_binom_coeffs_l784_784331


namespace vasya_wins_l784_784801

theorem vasya_wins : 
  ∃ (s₁ s₂ : set (ℝ × ℝ)), 
    (is_square s₁ ∧ is_square s₂ ∧ side_length s₁ = 3 ∧ side_length s₂ = 3 ∧ 
    ∃ t : ℝ × ℝ, translation t s₁ = s₃ ∧ area (s₃ ∩ s₂) ≥ 7) :=
sorry

-- Definitions -- These would need to be properly specified in Mathlib or custom-defined
def is_square (s : set (ℝ × ℝ)) : Prop := -- definition to check if a set is a square
sorry

def side_length (s : set (ℝ × ℝ)) : ℝ := -- definition to get the side length of a square
sorry

def translation (t : ℝ × ℝ) (s : set (ℝ × ℝ)) : set (ℝ × ℝ) := -- definition for parallel translation
sorry

def area (s : set (ℝ × ℝ)) : ℝ := -- definition to calculate the area of a region
sorry

end vasya_wins_l784_784801


namespace necessary_but_not_sufficient_condition_l784_784251

noncomputable def mn_non_zero_condition (m n : ℝ) (h : m ≠ 0) : Prop :=
  mn ≠ 0

theorem necessary_but_not_sufficient_condition (m n : ℝ) : ¬(mn_non_zero_condition m n (λ h : m ≠ 0)) ∧ m ≠ 0 → mn ≠ 0 :=
by
  sorry

end necessary_but_not_sufficient_condition_l784_784251


namespace student_knowing_conditions_l784_784107

theorem student_knowing_conditions (n d : ℕ) :
  (∃ S : Finset (Set ℕ), S.card = n ∧
    (∀ s ∈ S, (∃ G B : Set ℕ, G ∪ B = s ∧ disjoint G B ∧
      G.card = d ∧ B.card = d ∧
      (∀ a ∈ G, ∀ b ∈ G, a ≠ b) ∧ (∀ a ∈ B, ∀ b ∈ B, a ≠ b) ∧
      (∃ F : finset (ℕ × ℕ), (F ⊆ (G × B).to_finset) ∧
        (∀ g ∈ G, F.card = d ∧ (∀ b ∈ B, (g, b) ∈ F ∨ (b, g) ∈ F) ∧
        (∀ x ∈ F, ∃ a b, x = (a, b) ∧ a ≠ b)))))) ↔
  (n % 2 = 0 ∧ n / 2 ≥ d + 1 ∧ (d * (n / 2)) % 2 = 0) := sorry

end student_knowing_conditions_l784_784107


namespace Catriona_total_fish_l784_784986

theorem Catriona_total_fish:
  ∃ (goldfish angelfish guppies : ℕ),
  goldfish = 8 ∧
  angelfish = goldfish + 4 ∧
  guppies = 2 * angelfish ∧
  goldfish + angelfish + guppies = 44 :=
by
  -- Define the number of goldfish
  let goldfish := 8

  -- Define the number of angelfish, which is 4 more than goldfish
  let angelfish := goldfish + 4

  -- Define the number of guppies, which is twice the number of angelfish
  let guppies := 2 * angelfish

  -- Prove the total number of fish is 44
  have total_fish : goldfish + angelfish + guppies = 44 := by
    rw [←nat.add_assoc, nat.add_comm 12 8, nat.add_assoc, nat.add_comm 24 12, ←nat.add_assoc]

  use [goldfish, angelfish, guppies]
  exact ⟨rfl, rfl, rfl, total_fish⟩

end Catriona_total_fish_l784_784986


namespace largest_possible_x_l784_784481

theorem largest_possible_x : ∃ (x : ℕ), let T := 100 in let W := 40 in let E := 39 in let N := 2 * x in 
T = W + E - x + N ∧ E > 38 ∧ x = 21 :=
by
  sorry

end largest_possible_x_l784_784481


namespace reflection_point_l784_784842

theorem reflection_point (m b : ℝ) 
  (h_reflected : ∀ (p r : ℝ × ℝ), p = (2, 3) → r = (10, 7) → 
    let m := - (1 / 2) in -- slope of segment (2, 3) to (10, 7)
    m * (fst r - fst p) = (snd r - snd p)  → 
    ∃ b, r = (6, 5) ∧ b = 17) : 
  m + b = 15 := 
sorry

end reflection_point_l784_784842


namespace john_has_25_roommates_l784_784773

def roommates_of_bob := 10
def roommates_of_john := 2 * roommates_of_bob + 5

theorem john_has_25_roommates : roommates_of_john = 25 := 
by
  sorry

end john_has_25_roommates_l784_784773


namespace intersection_M_N_eq_neg2_l784_784642

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784642


namespace john_marbles_problem_l784_784017

theorem john_marbles_problem :
  let total_marbles := 15
  let special_marbles := 6 -- 2 red + 2 green + 2 blue
  let other_marbles := total_marbles - special_marbles
  ∃ count : ℕ,
    (count = 9 * (Nat.choose 12 3)) ∧ count = 1980 :=
by
  have num_ways_to_choose_special : ℕ := 3 * (Nat.choose 2 2) + (3 * 2)
  have num_ways_to_choose_others : ℕ := Nat.choose 12 3
  use 9 * num_ways_to_choose_others
  split
  · rw [num_ways_to_choose_special, Nat.choose, factorial]
    sorry -- additional calculations and factor simplifications
  sorry -- ensure the final count equals 1980

end john_marbles_problem_l784_784017


namespace function_fixed_point_l784_784411

-- Given f(x) = a^x + 4
def f (a : ℝ) (x : ℝ) : ℝ := a^x + 4

-- Prove that f(a, 0) = 5
theorem function_fixed_point (a : ℝ) : f a 0 = 5 :=
by
  unfold f
  sorry

end function_fixed_point_l784_784411


namespace parallel_vectors_x_value_l784_784713

theorem parallel_vectors_x_value :
  ∀ (x : ℝ), (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (2, x) → a.1 * b.2 = a.2 * b.1) → x = -4 :=
by
  intros x h
  have h_parallel := h (1, -2) (2, x) rfl rfl
  sorry

end parallel_vectors_x_value_l784_784713


namespace unique_diff_subset_l784_784808

noncomputable def exists_unique_diff_subset : Prop :=
  ∃ S : Set ℕ, 
    (∀ n : ℕ, n > 0 → ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ n = a - b)

theorem unique_diff_subset : exists_unique_diff_subset :=
  sorry

end unique_diff_subset_l784_784808


namespace lucy_share_l784_784038

-- Define the total amount saved by Natalie's father
def total_amount : ℤ := 10000

-- Define the share percentages
def natalie_share_percent : ℤ := 50
def rick_share_percent : ℤ := 60

-- Define the actual amounts each gets
def natalie_share : ℤ := total_amount * natalie_share_percent / 100
def remaining_after_natalie : ℤ := total_amount - natalie_share
def rick_share : ℤ := remaining_after_natalie * rick_share_percent / 100
def remaining_after_rick : ℤ := remaining_after_natalie - rick_share

-- Prove that Lucy gets $2000
theorem lucy_share : remaining_after_rick = 2000 :=
begin
  -- Prove that the calculations lead to Lucy getting $2000
  sorry
end

end lucy_share_l784_784038


namespace eccentricity_of_hyperbola_l784_784704

theorem eccentricity_of_hyperbola 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola : (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)) 
  (circle : (x y : ℝ), (x^2 + y^2 - 8 * y + 15 = 0)) 
  (chord_length : ∃ (x1 y1 x2 y2 : ℝ), (x^2 + (y-4)^2 = 1) ∧ ((x1 = x2) ∧ (y1 ≠ y2) ∨ (x1 ≠ x2) ∧ (y1 = y2)) ∧ (sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 2)) 
  : (let c := sqrt (a^2 + b^2) in (e^2 = 32) ∧ (e = 4 * sqrt 2)) := by
  sorry

end eccentricity_of_hyperbola_l784_784704


namespace limit_sin_cos_tan_l784_784476

open Real

theorem limit_sin_cos_tan :
  tendsto (fun x => (sin x + cos x) ^ (1 / (tan x))) (nhds (π / 4)) (nhds (sqrt 2)) :=
sorry

end limit_sin_cos_tan_l784_784476


namespace area_of_ADE_l784_784259

theorem area_of_ADE (A B C D E : Type)
  (hABC : Triangle ABC)
  (hD_on_AB : on_edge D AB)
  (hE_on_AC : on_edge E AC)
  (hAB : length AB = 6)
  (hAC : length AC = 4)
  (hBC : length BC = 8)
  (hAD : length AD = 2)
  (hAE : length AE = 3) :
  area ADE = (3 * sqrt 15) / 4 := 
sorry

end area_of_ADE_l784_784259


namespace algebraic_expression_value_l784_784255

theorem algebraic_expression_value (x y : ℝ) (h : x^2 - 4 * x - 1 = 0) : 
  (2 * x - 3) ^ 2 - (x + y) * (x - y) - y ^ 2 = 12 := 
by {
  sorry
}

end algebraic_expression_value_l784_784255


namespace intersection_M_N_eq_neg2_l784_784634

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784634


namespace no_solution_l784_784573

theorem no_solution (x : ℝ) (h₁ : x ≠ -1/3) (h₂ : x ≠ -4/5) :
  (2 * x - 4) / (3 * x + 1) ≠ (2 * x - 10) / (5 * x + 4) := 
sorry

end no_solution_l784_784573


namespace students_history_or_statistics_or_both_l784_784321

-- Definitions of the conditions
variables (H S H_inter_S : ℕ)

-- Given conditions
def condition_1 : H = 36 := sorry
def condition_2 : S = 30 := sorry
def condition_3 : H - H_inter_S = 29 := sorry

-- The proof problem
theorem students_history_or_statistics_or_both :
  H + S - H_inter_S = 59 :=
by {
  -- Assuming the conditions
  have h1 : H = 36,    from condition_1,
  have h2 : S = 30,    from condition_2,
  have h3 : H - H_inter_S = 29, from condition_3,
  
  -- Substituting values from conditions
  sorry
}

end students_history_or_statistics_or_both_l784_784321


namespace red_face_cubes_count_l784_784487

-- Definition of the original 10 x 10 x 10 cube dimensions
def original_cube_dimensions : ℕ × ℕ × ℕ := (10, 10, 10)

-- The cube is painted red on the outside
def painted_cube := original_cube_dimensions

-- The cube is cut into 1-inch cubes (resulting in 10 x 10 x 10 smaller cubes)
def cut_into_smaller_cubes := (1, 1, 1)

-- Prove the number of 1-inch cubes with at least one red face is 488
theorem red_face_cubes_count : 
  let cube_size := original_cube_dimensions,
  let painted := painted_cube,
  let smaller_cube_size := cut_into_smaller_cubes in
  (∃ n : ℕ, n = 488) :=
begin
  use 488,
  sorry -- Proof is omitted
end

end red_face_cubes_count_l784_784487


namespace arc_length_of_sector_l784_784690

theorem arc_length_of_sector (α : ℝ) (S : ℝ) (h1 : α = 2) (h2 : S = 9) : 
  ∃ l : ℝ, l = 6 :=
begin
  sorry
end

end arc_length_of_sector_l784_784690


namespace find_number_l784_784922

theorem find_number :
  ∃ x : ℝ, 0.60 * x - 90 = 5 ∧ x ≈ 158.33 := by
  sorry

end find_number_l784_784922


namespace directrix_of_parabola_l784_784684

theorem directrix_of_parabola (a : ℝ) (P : ℝ × ℝ)
  (h1 : 3 * P.1 ^ 2 - P.2 ^ 2 = 3 * a ^ 2)
  (h2 : P.2 ^ 2 = 8 * a * P.1)
  (h3 : a > 0)
  (h4 : abs ((P.1 - 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) + abs ((P.1 + 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) = 12) :
  (a = 1) → P.1 = 6 - 3 * a → P.2 ^ 2 = 8 * a * (6 - 3 * a) → -2 * a = -2 := 
by
  sorry

end directrix_of_parabola_l784_784684


namespace xyz_cubed_sum_l784_784369

noncomputable def poly := (t : ℝ) → t^3 - 5 * t - 3

theorem xyz_cubed_sum (x y z : ℝ) 
  (hx : poly x = 0) 
  (hy : poly y = 0) 
  (hz : poly z = 0) : 
  x^3 * y^3 + x^3 * z^3 + y^3 * z^3 = -98 :=
  sorry

end xyz_cubed_sum_l784_784369


namespace harmonic_alternating_series_eq_l784_784121

theorem harmonic_alternating_series_eq (n : ℕ) (hn : n > 0) :
  (Finset.range (2 * n)).sum (λ k, ((-1)^k) / (k + 1)) =
    (Finset.Ico (n + 1) (2 * n + 1)).sum (λ k, 1 / k) :=
by
  sorry

end harmonic_alternating_series_eq_l784_784121


namespace catriona_total_fish_eq_44_l784_784989

-- Definitions based on conditions
def goldfish : ℕ := 8
def angelfish : ℕ := goldfish + 4
def guppies : ℕ := 2 * angelfish
def total_fish : ℕ := goldfish + angelfish + guppies

-- The theorem we need to prove
theorem catriona_total_fish_eq_44 : total_fish = 44 :=
by
  -- We are skipping the proof steps with 'sorry' for now
  sorry

end catriona_total_fish_eq_44_l784_784989


namespace probability_odd_sum_is_correct_l784_784375

noncomputable def spinner_A_outcomes := {1, 3, 4}
noncomputable def spinner_B_outcomes := {2, 3, 5}
noncomputable def spinner_C_outcomes := {3, 5, 8}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def probability_odd_sum : ℚ :=
  let possible_outcomes := (spinner_A_outcomes × spinner_B_outcomes × spinner_C_outcomes).to_finset in
  let odd_sum := possible_outcomes.filter (λ x, is_odd (x.1 + x.2 + x.3)) in
  (odd_sum.card : ℚ) / (possible_outcomes.card : ℚ)

theorem probability_odd_sum_is_correct : probability_odd_sum = 2 / 9 :=
  sorry

end probability_odd_sum_is_correct_l784_784375


namespace total_difference_in_fuel_cost_l784_784938

-- Define the given conditions
def number_of_vans := 6.0
def number_of_buses := 8.0

def people_per_van := 6
def people_per_bus := 18

def distance_per_van := 120
def distance_per_bus := 150

def fuel_efficiency_vans := 20
def fuel_efficiency_buses := 6

def cost_per_gallon_vans := 2.5
def cost_per_gallon_buses := 3

-- Prove the total difference in fuel cost
theorem total_difference_in_fuel_cost :
  let total_distance_vans := number_of_vans * distance_per_van,
      total_distance_buses := number_of_buses * distance_per_bus,
      fuel_consumed_vans := total_distance_vans / fuel_efficiency_vans,
      fuel_consumed_buses := total_distance_buses / fuel_efficiency_buses,
      total_fuel_cost_vans := fuel_consumed_vans * cost_per_gallon_vans,
      total_fuel_cost_buses := fuel_consumed_buses * cost_per_gallon_buses
  in total_fuel_cost_buses - total_fuel_cost_vans = 510 := by
  sorry

end total_difference_in_fuel_cost_l784_784938


namespace desk_length_l784_784174

-- Define the problem conditions
variables (d : ℝ) (n : ℕ)
axiom num_eq : ∀ (d : ℝ) (n : ℕ), n ≥ 0 ∧ 15 - (n * d + n * 1.5) = 1

-- Define the theorem to prove the length of each desk
theorem desk_length : ∃ d : ℝ, d = 5.5 :=
begin
  existsi 5.5,
  sorry -- Proof skipped
end

end desk_length_l784_784174


namespace find_value_of_a_l784_784045

theorem find_value_of_a (a : ℝ) (h : 2 - a = 0) : a = 2 :=
by {
  sorry
}

end find_value_of_a_l784_784045


namespace count_functions_l784_784233

-- Define the set {1, 2, 3}
inductive three_elements
| one
| two
| three

open three_elements

-- Define the type of functions from {1, 2, 3} to {1, 2, 3}
noncomputable def three_elements_func := three_elements → three_elements

-- Define the property f(f(x)) = f(x) for a function f
def satisfies_property (f : three_elements_func) : Prop :=
  ∀ x : three_elements, f (f x) = f x

-- Prove the total number of such functions is 10
theorem count_functions : 
  (set_of (λ f : three_elements_func, satisfies_property f)).to_finset.card = 10 := 
sorry

end count_functions_l784_784233


namespace females_in_orchestra_not_in_band_l784_784177

theorem females_in_orchestra_not_in_band 
  (females_in_band : ℤ) 
  (males_in_band : ℤ) 
  (females_in_orchestra : ℤ) 
  (males_in_orchestra : ℤ) 
  (females_in_both : ℤ) 
  (total_members : ℤ) 
  (h1 : females_in_band = 120) 
  (h2 : males_in_band = 100) 
  (h3 : females_in_orchestra = 100) 
  (h4 : males_in_orchestra = 120) 
  (h5 : females_in_both = 80) 
  (h6 : total_members = 260) : 
  (females_in_orchestra - females_in_both = 20) := 
  sorry

end females_in_orchestra_not_in_band_l784_784177


namespace question_statement_l784_784000

def line := Type
def plane := Type

-- Definitions for line lying in plane and planes being parallel 
def isIn (a : line) (α : plane) : Prop := sorry
def isParallel (α β : plane) : Prop := sorry
def isParallelLinePlane (a : line) (β : plane) : Prop := sorry

-- Conditions 
variables (a b : line) (α β : plane) 
variable (distinct_lines : a ≠ b)
variable (distinct_planes : α ≠ β)

-- Main statement to prove
theorem question_statement (h_parallel_planes : isParallel α β) (h_line_in_plane : isIn a α) : isParallelLinePlane a β := 
sorry

end question_statement_l784_784000


namespace int_sol_many_no_int_sol_l784_784539

-- Part 1: If there is one integer solution, there are at least three integer solutions
theorem int_sol_many (n : ℤ) (hn : n > 0) (x y : ℤ) 
  (hxy : x^3 - 3 * x * y^2 + y^3 = n) : 
  ∃ a b c d e f : ℤ, 
    (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (e, f) ≠ (x, y) ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f) ∧ 
    a^3 - 3 * a * b^2 + b^3 = n ∧ 
    c^3 - 3 * c * d^2 + d^3 = n ∧ 
    e^3 - 3 * e * f^2 + f^3 = n :=
sorry

-- Part 2: When n = 2891, the equation has no integer solutions
theorem no_int_sol : ¬ ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end int_sol_many_no_int_sol_l784_784539


namespace smallest_positive_integer_with_conditions_l784_784569

theorem smallest_positive_integer_with_conditions : 
  ∃ n : ℕ, (∀ n > 0, (∃ d : ℕ → Prop, (∀ k : ℕ, 1 ≤ k → k ≤ 144 → d k ∧ (1 ≤ ∃ i, d i + 9) = 144)) ∧ (n = 110880)) :=
sorry

end smallest_positive_integer_with_conditions_l784_784569


namespace factorial_condition_l784_784215

theorem factorial_condition (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ k l m : ℕ, k > 1 ∧ l > 1 ∧ m > 1 ∧ ab + 1 = k! ∧ bc + 1 = l! ∧ ca + 1 = m!) →
  (∃ k : ℕ, k > 1 ∧ (a = k! - 1 ∧ b = 1 ∧ c = 1) ∨ (b = k! - 1 ∧ c = 1 ∧ a = 1) ∨ (c = k! - 1 ∧ a = 1 ∧ b = 1)) :=
sorry

end factorial_condition_l784_784215


namespace angle_difference_in_triangle_l784_784355

theorem angle_difference_in_triangle
  {A B C P : Type}
  (triangle_ABC : is_triangle A B C)
  (P_in_interior : lies_in_interior P triangle_ABC)
  (angle_ABP : ∠ABP = 20)
  (angle_ACP : ∠ACP = 15) :
  ∠BPC - ∠BAC = 35 :=
sorry

end angle_difference_in_triangle_l784_784355


namespace tan_add_pi_over_four_sin_cos_ratio_l784_784689

-- Definition of angle α with the condition that tanα = 2
def α : ℝ := sorry -- Define α such that tan α = 2

-- The first Lean statement for proving tan(α + π/4) = -3
theorem tan_add_pi_over_four (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- The second Lean statement for proving (sinα + cosα) / (2sinα - cosα) = 1
theorem sin_cos_ratio (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 :=
sorry

end tan_add_pi_over_four_sin_cos_ratio_l784_784689


namespace number_of_common_divisors_l784_784715

def prime_factors_100 : List ℕ := [2, 2, 5, 5]
def prime_factors_150 : List ℕ := [2, 3, 5, 5]

noncomputable def divisors (n : ℕ) (factors : List ℕ) : List ℕ :=
List.foldr List.union [] (List.map (λ m, List.range (m + 1) |>.map (λ k, m ^ k)) factors)

noncomputable def divisors_100 := divisors 100 prime_factors_100
noncomputable def divisors_150 := divisors 150 prime_factors_150

def common_divisors (l1 l2 : List ℕ) : List ℕ :=
l1 |>.filter (λ x, l2.contains x)

theorem number_of_common_divisors : (common_divisors divisors_100 divisors_150).length = 12 :=
by sorry

end number_of_common_divisors_l784_784715


namespace problem_even_function_l784_784094

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x
  
theorem problem_even_function : even_function (λ x : ℝ, x^2 + cos x) :=
by
  intro x
  sorry

end problem_even_function_l784_784094


namespace measure_minor_arc_AK_l784_784007

/-- Given a circle and an inscribed angle KAT measuring 36 degrees, the measure of minor arc AK is 108 degrees. -/
theorem measure_minor_arc_AK (Q : Circle) (A K T : Point) 
  (h1 : ∠ KAT = 36) 
  (h2 : inscribed_angle Q (KAT)) : 
  measure_minor_arc Q A K = 108 :=
sorry

end measure_minor_arc_AK_l784_784007


namespace probability_correct_l784_784381

-- Defining the values on the spinner
inductive SpinnerValue
| Bankrupt
| Thousand
| EightHundred
| FiveThousand
| Thousand'

open SpinnerValue

-- Function to get value in number from SpinnerValue
def value (v : SpinnerValue) : ℕ :=
  match v with
  | Bankrupt => 0
  | Thousand => 1000
  | EightHundred => 800
  | FiveThousand => 5000
  | Thousand' => 1000

-- Total number of spins
def total_spins : ℕ := 3

-- Total possible outcomes
def total_outcomes : ℕ := (5 : ℕ) ^ total_spins

-- Number of favorable outcomes (count of permutations summing to 5800)
def favorable_outcomes : ℕ :=
  12  -- This comes from solution steps

-- The probability as a ratio of favorable outcomes to total outcomes
def probability_of_5800_in_three_spins : ℚ :=
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_5800_in_three_spins = 12 / 125 := by
  sorry

end probability_correct_l784_784381


namespace yellow_balls_count_l784_784864

theorem yellow_balls_count (R B G Y : ℕ) 
  (h1 : R = 2 * B) 
  (h2 : B = 2 * G) 
  (h3 : Y > 7) 
  (h4 : R + B + G + Y = 27) : 
  Y = 20 := by
  sorry

end yellow_balls_count_l784_784864


namespace degree_of_polynomial_l784_784201

noncomputable def p (x : ℝ) : ℝ := 5 * x^6 - 4 * x^5 + x^2 - 18
noncomputable def q (x : ℝ) : ℝ := 2 * x^12 + 6 * x^9 - 11 * x^6 + 10
noncomputable def r (x : ℝ) : ℝ := (x^3 + 4)^6

theorem degree_of_polynomial :
  polynomial.degree ((polynomial.C (p 0)).natDegree * (polynomial.C (q 0)).natDegree 
  - (polynomial.C (r 0)).natDegree) = 18 :=
sorry

end degree_of_polynomial_l784_784201


namespace moneyEarnedDuringHarvest_l784_784033

-- Define the weekly earnings, duration of harvest, and weekly rent.
def weeklyEarnings : ℕ := 403
def durationOfHarvest : ℕ := 233
def weeklyRent : ℕ := 49

-- Define total earnings and total rent.
def totalEarnings : ℕ := weeklyEarnings * durationOfHarvest
def totalRent : ℕ := weeklyRent * durationOfHarvest

-- Calculate the money earned after rent.
def moneyEarnedAfterRent : ℕ := totalEarnings - totalRent

-- The theorem to prove.
theorem moneyEarnedDuringHarvest : moneyEarnedAfterRent = 82482 :=
  by
  sorry

end moneyEarnedDuringHarvest_l784_784033


namespace total_unit_squares_in_100_rings_l784_784194

theorem total_unit_squares_in_100_rings : 
  (∑ k in Finset.range 100, 8 * (k + 1)) = 40400 :=
by
  sorry

end total_unit_squares_in_100_rings_l784_784194


namespace intersection_M_N_l784_784615

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784615


namespace max_value_of_y_l784_784845

open Real

noncomputable def y (x : ℝ) := 1 + 1 / (x^2 + 2*x + 2)

theorem max_value_of_y : ∃ x : ℝ, y x = 2 :=
sorry

end max_value_of_y_l784_784845


namespace time_to_pass_platform_l784_784169

def speed_kmhr : ℝ := 54
def speed_ms : ℝ := (speed_kmhr * 1000) / 3600
def time_pass_man : ℝ := 20
def length_platform : ℝ := 75.006
def length_train : ℝ := speed_ms * time_pass_man
def total_distance : ℝ := length_train + length_platform

theorem time_to_pass_platform :
  (total_distance / speed_ms) = 25.0004 :=
sorry

end time_to_pass_platform_l784_784169


namespace inscribed_cube_diagonal_length_l784_784406

-- Define the condition of the problem
def inscribed_sphere_radius (R : ℝ) : Prop :=
  ∃ (d : ℝ), (√3) * d / 2 = R 

-- State the theorem to be proved
theorem inscribed_cube_diagonal_length (R : ℝ) (h : inscribed_sphere_radius R) : 
  (√3) * R = 2 * R :=
sorry

end inscribed_cube_diagonal_length_l784_784406


namespace handshakes_at_gathering_l784_784074

-- Define the number of couples
def couples := 6

-- Define the total number of people
def total_people := 2 * couples

-- Each person shakes hands with 10 others (excluding their spouse)
def handshakes_per_person := 10

-- Total handshakes counted with pairs counted twice
def total_handshakes := total_people * handshakes_per_person / 2

-- The theorem to prove the number of handshakes
theorem handshakes_at_gathering : total_handshakes = 60 :=
by
  sorry

end handshakes_at_gathering_l784_784074


namespace intersection_P_Q_l784_784777

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem intersection_P_Q :
  P ∩ Q = {-1, 0, 1} :=
sorry

end intersection_P_Q_l784_784777


namespace points_on_line_l784_784099

theorem points_on_line (k : ℕ) :
  ∃ m b : ℤ, 
    (1, -4) ∈ ({p : ℤ × ℤ | p.2 = m * p.1 + b}) ∧
    (3,  2) ∈ ({p : ℤ × ℤ | p.2 = m * p.1 + b}) ∧
    (6, k / 3) ∈ ({p : ℤ × ℤ | p.2 = m * p.1 + b}) → 
  k = 33 :=
sorry

end points_on_line_l784_784099


namespace completion_days_l784_784543

theorem completion_days (D : ℝ) :
  (1 / D + 1 / 9 = 1 / 3.2142857142857144) → D = 5 := by
  sorry

end completion_days_l784_784543


namespace exists_convex_shape_l784_784761

theorem exists_convex_shape (r : ℝ) (h_r : r = 1) :
  ∃ (K : set (EuclideanSpace ℝ 2)), convex K ∧ (¬(∃ (C : set (EuclideanSpace ℝ 2)), ∃ (hC : metric.bounded C ∧ C ⊆ K), semicircle C r)) 
  ∧ (unit_circle r = K ∪ (rotation 90 K)) :=
sorry

end exists_convex_shape_l784_784761


namespace new_years_day_l784_784382

theorem new_years_day (month_has: ∃ d : ℕ → Prop, 
  d 1 ∧ d 8 ∧ d 15 ∧ d 22 ∧ d 29 ∧
  d 2 ∧ d 3 ∧ d 4 ∧ d 5 ∧ d 6 ∧ d 7 ∧
  d 9 ∧ d 10 ∧ d 11 ∧ d 12 ∧ d 13 ∧ d 14 ∧
  d 16 ∧ d 17 ∧ d 18 ∧ d 19 ∧ d 20 ∧ d 21 ∧
  d 23 ∧ d 24 ∧ d 25 ∧ d 26 ∧ d 27 ∧ d 28 ∧ d 29 ∧
  d 13) : (1, year_starts) :
  ((new_year_day : nat := 1) and (💭 ) :
  ∃ new_year_day : ℕ → Day, 
  d 1 - ∧ (∃n+52 ) new_year_day 

 


end new_years_day_l784_784382


namespace alice_probability_l784_784171

def alice_probability_starting_at_zero (n : ℕ) (p q : ℚ) : ℚ := 
  (p/q)

theorem alice_probability (a b : ℕ) (h : Nat.coprime a b) :
  alice_probability_starting_at_zero 10 15 64 = (15 / 64 : ℚ) ∧ a + b = 79 :=
by
  sorry

end alice_probability_l784_784171


namespace product_of_roots_l784_784416

theorem product_of_roots:
  (real.cbrt 4) * (real.root 8 4) = 2 * (real.root 32 12) := by
  sorry

end product_of_roots_l784_784416


namespace complex_trig_square_l784_784815

theorem complex_trig_square :
  (Complex.cos (225 * Real.pi / 180) + Complex.sin (225 * Real.pi / 180) * Complex.i)^2 =
    Complex.i :=
by
  sorry

end complex_trig_square_l784_784815


namespace nonneg_int_values_of_fraction_condition_l784_784916

theorem nonneg_int_values_of_fraction_condition (n : ℕ) : (∃ k : ℤ, 30 * n + 2 = k * (12 * n + 1)) → n = 0 := by
  sorry

end nonneg_int_values_of_fraction_condition_l784_784916


namespace complex_sum_magnitude_eq_three_l784_784364

open Complex

theorem complex_sum_magnitude_eq_three (a b c : ℂ) 
    (h1 : abs a = 1) 
    (h2 : abs b = 1) 
    (h3 : abs c = 1) 
    (h4 : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3) : 
    abs (a + b + c) = 3 := 
sorry

end complex_sum_magnitude_eq_three_l784_784364


namespace total_arrangements_l784_784802

-- Defining the problem conditions
def math_books_count := 3
def chinese_books_count := 2

-- Defining the problem statement in Lean 4
theorem total_arrangements (math_books_count = 3) (chinese_books_count = 2) :
  ∃ n, n = 48 := 
by
  -- Proof goes here
  sorry

end total_arrangements_l784_784802


namespace tangent_line_at_0_g_is_increasing_l784_784703

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp x * Real.log (x + Real.exp 1)

-- Define the point (0, f(0))
def point : ℝ × ℝ := (0, f 0)

-- Lean statement for the tangent line problem
theorem tangent_line_at_0 :
  ∀ (x y : ℝ), (e + 1) * x - e * y + e = 0 ↔ y = (f' 0 * x) + f 0 := sorry

-- Define the derivative of the function g
def g (x : ℝ) : ℝ := f'.mk' (Real.exp x * (Real.log (x + Real.exp 1) + (x + Real.exp 1)⁻¹))

-- Lean statement for the monotonicity of g
theorem g_is_increasing : ∀ x : ℝ, x ≥ 0 → ∃ δ > 0, g is_strict_mono_in_Iio (x, x + δ) := sorry

end tangent_line_at_0_g_is_increasing_l784_784703


namespace find_b2_l784_784422

theorem find_b2 (b : ℕ → ℝ) 
  (h1 : b 1 = 34) 
  (h2 : b 12 = 150) 
  (h3 : ∀ n ≥ 3, b n = (b 1 + b 2 + ∑ i in finset.range (n-2), b (i + 3)) / (n-1)) :
  b 2 = 266 :=
sorry

end find_b2_l784_784422


namespace base_conversion_and_addition_l784_784555

def C : ℕ := 12

def base9_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 9^2) + (d1 * 9^1) + (d0 * 9^0)

def base13_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 13^2) + (d1 * 13^1) + (d0 * 13^0)

def num1 := base9_to_nat 7 5 2
def num2 := base13_to_nat 6 C 3

theorem base_conversion_and_addition :
  num1 + num2 = 1787 :=
by
  sorry

end base_conversion_and_addition_l784_784555


namespace initial_chips_in_bag_l784_784376

-- Definitions based on conditions
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5
def chips_kept_by_nancy : ℕ := 10

-- Theorem statement
theorem initial_chips_in_bag (total_chips := chips_given_to_brother + chips_given_to_sister + chips_kept_by_nancy) : total_chips = 22 := 
by 
  -- we state the assertion
  sorry

end initial_chips_in_bag_l784_784376


namespace determine_fx_lg_lg2_l784_784285

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * real.sin x + 4

theorem determine_fx_lg_lg2 (a b : ℝ) (h : f a b (real.logb 10 (real.log2 10)) = 5) :
  f a b (real.logb 10 (real.log 2)) = 3 :=
  sorry

end determine_fx_lg_lg2_l784_784285


namespace hundreds_digit_25_sub_20_add_10_l784_784892

open Nat

example : Nat := 25!
example : Nat := 20!
example : Nat := 10!

theorem hundreds_digit_25_sub_20_add_10 : 
  let val := (25! - 20! + 10!) % 1000 in (val / 100) % 10 = 8 :=
by
  sorry

end hundreds_digit_25_sub_20_add_10_l784_784892


namespace ellipse_focus_smaller_y_l784_784530

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def axisLength (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p1.1 - p2.1
  let dy := p1.2 - p2.2
  real.sqrt (dx^2 + dy^2)

noncomputable def ellipseFocusLowY (c : ℝ × ℝ) (a b : ℝ) : ℝ × ℝ :=
  let cDist := real.sqrt (a^2 - b^2)
  (c.1 - cDist, c.2)

theorem ellipse_focus_smaller_y
  (major_axis_ends : ℝ × ℝ) (major_axis_end2 : ℝ × ℝ)
  (minor_axis_ends : ℝ × ℝ) (minor_axis_end2 : ℝ × ℝ) :
    let center := midpoint major_axis_ends major_axis_end2
    let major_axis_length := axisLength major_axis_ends major_axis_end2
    let minor_axis_length := axisLength minor_axis_ends minor_axis_end2
  ellipseFocusLowY center major_axis_length minor_axis_length = (5 - real.sqrt 5, 2) :=
by
  sorry

end ellipse_focus_smaller_y_l784_784530


namespace min_value_tetrahedron_abcd_l784_784398

def tetrahedron_ABCD : Type := unit

noncomputable def AD : ℝ := 30
noncomputable def BC : ℝ := 30
noncomputable def AC : ℝ := 46
noncomputable def BD : ℝ := 46
noncomputable def AB : ℝ := 54
noncomputable def CD : ℝ := 54

def g (X : ℝ) (A B C D : ℝ) := X + B + C + D
def min_g (A B C D : ℝ) := 4 * real.sqrt(731)

theorem min_value_tetrahedron_abcd (A B C D : ℝ) (X : ℝ) :
  AD = 30 → BC = 30 → AC = 46 → BD = 46 → AB = 54 → CD = 54 → g(X, A, B, C, D) = 4 * real.sqrt(731) :=
by 
  intros;
  sorry

end min_value_tetrahedron_abcd_l784_784398


namespace alex_finishes_first_l784_784812

variable (s r : ℝ)

def garden_area_samantha : ℝ := s
def garden_area_alex : ℝ := s / 3
def garden_area_nikki : ℝ := 3 * s / 2

def mow_rate_alex : ℝ := r
def mow_rate_samantha : ℝ := r / 3
def mow_rate_nikki : ℝ := r / 2

def time_to_mow (garden_area mow_rate : ℝ) : ℝ := garden_area / mow_rate

theorem alex_finishes_first 
  (h_samantha_area : garden_area_samantha s = s)
  (h_alex_area : garden_area_alex s = s / 3)
  (h_nikki_area : garden_area_nikki s = 3 * s / 2)
  (h_alex_rate : mow_rate_alex r = r)
  (h_samantha_rate : mow_rate_samantha r = r / 3)
  (h_nikki_rate : mow_rate_nikki r = r / 2) :
  time_to_mow (garden_area_alex s) (mow_rate_alex r) <
  time_to_mow (garden_area_samantha s) (mow_rate_samantha r) ∧
  time_to_mow (garden_area_alex s) (mow_rate_alex r) <
  time_to_mow (garden_area_nikki s) (mow_rate_nikki r) :=
by
  sorry

end alex_finishes_first_l784_784812


namespace AC_eq_BC_l784_784412

open_locale classical

variables {A B C A1 B1 M : Type*} [metric_space A] [metric_space B] [metric_space C]
  [metric_space A1] [metric_space B1] [metric_space M]

-- Define the condition that A A1 and B B1 are medians of triangle ABC
def is_median (A B C A1 : Type*) [metric_space A] [metric_space B]
  [metric_space C] [metric_space A1] : Prop :=
  ∃ (G : Type*) [metric_space G], centroid (G : triangle A B C) A = A1 / ∧ centroid (G : triangle A B C) B = B1 / 

-- Define the intersection at M
def intersection_at (A A1 B B1 M : Type*) [metric_space A] [metric_space B]
  [metric_space M] [metric_space A1] : Prop :=
  midpoint (A1. intersection_point A B) (B1 intersection_point B A) = M)

-- Define the quadrilateral A1 M B1 C is cyclic
def is_cyclic_quadrilateral (A1 M B1 C : Type*) [metric_space A1] [metric_space M]
  [metric_space B1] [metric_space C] : Prop :=
  ∃ (P Q R S : Type*), cyclic_quad P Q R S

-- The proof problem statement
theorem AC_eq_BC :
  is_median A A1 B1 B B1 ∧ intersection_at A A1 B B1 M ∧ is_cyclic_quadrilateral A1 M B1 C → 
  A.dist C = B.dist C :=
sorry

end AC_eq_BC_l784_784412


namespace mono_intervals_range_of_a_l784_784996

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.exp (x - 1)

theorem mono_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x, f x a > 0) ∧ 
  (a > 0 → (∀ x, x < 1 - Real.log a → f x a > 0) ∧ (∀ x, x > 1 - Real.log a → f x a < 0)) :=
sorry

theorem range_of_a (h : ∀ x, f x a ≤ 0) : a ≥ 1 :=
sorry

end mono_intervals_range_of_a_l784_784996


namespace Sn_converges_to_32π_l784_784930

-- Defining the initial side length of the square and radius of the first circle
def side_length_square : ℝ := 8
def radius_first_circle : ℝ := side_length_square / 2

-- Defining the area of the first circle
def area_first_circle : ℝ := π * radius_first_circle^2

-- Defining the total sum of the areas of first n circles
def S (n : ℕ) : ℝ := finset.sum (finset.range n) (λ k, area_first_circle * (1/2)^k)

-- Theoretical limit of S as n approaches infinity
def S_lim : ℝ := area_first_circle / (1 - (1 / 2))

-- Proof statement
theorem Sn_converges_to_32π : ∀ n, ∃ L, S n = S_lim :=
  by
  sorry

end Sn_converges_to_32π_l784_784930


namespace age_difference_l784_784942

-- Define the present age of the son.
def S : ℕ := 22

-- Define the present age of the man.
variable (M : ℕ)

-- Given condition: In two years, the man's age will be twice the age of his son.
axiom condition : M + 2 = 2 * (S + 2)

-- Prove that the difference in present ages of the man and his son is 24 years.
theorem age_difference : M - S = 24 :=
by 
  -- We will fill in the proof here
  sorry

end age_difference_l784_784942


namespace banner_scaled_height_l784_784377

theorem banner_scaled_height (width1 height1 width2 : ℝ) (h : width1 ≠ 0) (H_scale : width2 = 4 * width1) :
  let height2 := 4 * height1 in
  height2 = 8 :=
by
  -- label the knowns
  let width1 := 3
  let height1 := 2
  let width2 := 12
  let height2 := 4 * height1 -- defining the height after scaling
  have H_scale : width2 = 4 * width1 := by sorry -- scaling factor condition
  show height2 = 8 from by sorry

end banner_scaled_height_l784_784377


namespace students_only_math_is_70_l784_784096

variable (total_students : ℕ)
variable (math_students : ℕ)
variable (foreign_language_students : ℕ)
variable (science_students : ℕ)
variable (students_only_math : ℕ)

-- Defining the condition constraints
def conditions : Prop :=
  total_students = 150 ∧
  math_students = 120 ∧
  foreign_language_students = 110 ∧
  science_students = 95

-- Defining the statement we want to prove
theorem students_only_math_is_70 :
  conditions total_students math_students foreign_language_students science_students →
  students_only_math = math_students - ((math_students + foreign_language_students + science_students - total_students) / 3) + (total_students - math_students) / 6  →
  students_only_math = 70 :=
by
  intros h h1
  unfold conditions at h
  simp at h
  sorry

end students_only_math_is_70_l784_784096


namespace find_xy_l784_784534

variable (a : ℝ)

theorem find_xy (x y : ℝ) (k : ℤ) :
  x + y = a ∧ sin x ^ 2 + sin y ^ 2 = 1 - cos a ↔ 
  (x = a / 2 + k * Real.pi ∧ y = a / 2 - k * Real.pi) ∨
  (cos a = 0 ∧ x + y = (2 * k + 1) * Real.pi / 2) := 
sorry

end find_xy_l784_784534


namespace find_constant_a_l784_784229

theorem find_constant_a (x y a : ℝ) (h1 : (ax + 4 * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : a = 7 :=
sorry

end find_constant_a_l784_784229


namespace Ashutosh_completion_time_l784_784397

def Suresh_work_rate := 1 / 15
def Ashutosh_work_rate := 1 / 25
def Suresh_work_time := 9

def job_completed_by_Suresh_in_9_hours := Suresh_work_rate * Suresh_work_time
def remaining_job := 1 - job_completed_by_Suresh_in_9_hours

theorem Ashutosh_completion_time : 
  Ashutosh_work_rate * t = remaining_job -> t = 10 :=
by
  sorry

end Ashutosh_completion_time_l784_784397


namespace ants_20_feet_apart_l784_784443

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

def ant_A_position (n : ℕ) : Point3D :=
  let s := 2 * n in
  Point3D.mk s (s * (nat.cases_on (n % 3) 1 (λ _, if (n % 3) = 1 then 1 else -1) : ℕ)) s
  
def ant_B_position (n : ℕ) : Point3D :=
  let t := 3 * n in
  Point3D.mk (-t) (-t) 0

def at_distance_20 (n m : ℕ) : Prop :=
  distance (ant_A_position n) (ant_B_position m) = 20

theorem ants_20_feet_apart :
  ∃ (n m : ℕ), at_distance_20 n m ∧ 
  (nat.cases_on ((n + 1) % 3) "north" (λ _, if ((n + 1) % 3) = 1 then "east" else "up") = "up" ∧ 
  (nat.cases_on ((m + 1) % 2) "south" (λ _, "west") = "west") :=
by
  sorry

end ants_20_feet_apart_l784_784443


namespace probability_all_genuine_given_same_weight_l784_784867

def isGenuine (c : ℕ) : bool := 9 ≤ c ∧ c < 12

def pair_probability_same_weight (coin1 coin2 coin3 coin4 : ℕ) : Prop :=
  let weights := List.map (λ x, if isGenuine x then 1 else 0) [coin1, coin2, coin3, coin4];
  (weights[0] + weights[1] = weights[2] + weights[3])

theorem probability_all_genuine_given_same_weight :
  ∀ (coin1 coin2 coin3 coin4 : ℕ),
  1 ≤ coin1 ∧ coin1 ≤ 12 ∧
  1 ≤ coin2 ∧ coin2 ≤ 12 ∧
  coin1 ≠ coin2 ∧
  1 ≤ coin3 ∧ coin3 ≤ 12 ∧
  coin3 ≠ coin1 ∧ coin3 ≠ coin2 ∧
  1 ≤ coin4 ∧ coin4 ≤ 12 ∧
  coin4 ≠ coin1 ∧ coin4 ≠ coin2 ∧ coin4 ≠ coin3 ∧
  (pair_probability_same_weight coin1 coin2 coin3 coin4) →
  ((isGenuine coin1 ∧ isGenuine coin2 ∧ isGenuine coin3 ∧ isGenuine coin4) = true) →
  42 / 165 = 84 / 113 :=
by
  intros coin1 coin2 coin3 coin4 h_ranges h_distinct_pair h_genuine
  sorry

end probability_all_genuine_given_same_weight_l784_784867


namespace rectangle_perimeter_l784_784810

theorem rectangle_perimeter (JE EF JH KM : ℝ) (hJE : JE = 18) (hEF : EF = 22) (hJH : JH = 25) (hKM : KM = 35) :
  let EG := Real.sqrt (JE ^ 2 + EF ^ 2)
  let x := Real.sqrt (JH ^ 2 - JE ^ 2)
  let y := KM
  2 * (x + y) = 126 :=
by {
  let EG := Real.sqrt (18 ^ 2 + 22 ^ 2),
  let x := Real.sqrt (25 ^ 2 - 18 ^ 2),
  let y := 35,
  show 2 * (x + y) = 126,
  sorry
}

end rectangle_perimeter_l784_784810


namespace trigonometric_identity_l784_784903

variable (α : ℝ)

-- Definition of tangent
def tan (θ : ℝ) := Real.sin θ / Real.cos θ

-- Definition of cotangent
def cot (θ : ℝ) := Real.cos θ / Real.sin θ

-- Problem statement
theorem trigonometric_identity :
  (Real.cos (4 * α) * tan (2 * α) - Real.sin (4 * α)) / 
  (Real.cos (4 * α) * cot (2 * α) + Real.sin (4 * α)) = 
  -((tan (2 * α)) ^ 2) :=
by
  sorry

end trigonometric_identity_l784_784903


namespace ratio_is_correct_l784_784994

variables (r : ℝ)

def area_circle (radius : ℝ) : ℝ :=
  π * radius^2

def area_semicircle (radius : ℝ) : ℝ :=
  (1 / 2) * area_circle radius

def combined_area_two_semicircles (r : ℝ) : ℝ :=
  area_semicircle (r / 2) + area_semicircle (r / 3)

def ratio_combined_areas_to_circle (r : ℝ) : ℝ :=
  combined_area_two_semicircles r / area_circle r

theorem ratio_is_correct (r : ℝ) : ratio_combined_areas_to_circle r = 13 / 72 :=
  sorry

end ratio_is_correct_l784_784994


namespace num_distinct_paintings_l784_784332

theorem num_distinct_paintings :
  let disks := Finset.range 7
  let blue := 4
  let red := 2
  let green := 1
  let total_paintings := (disks.card.choose blue) * ((disks.card - blue).choose red) * ((disks.card - blue - red).choose green)
  let reflection_fixed := 3
  let distinct_paintings := (total_paintings + reflection_fixed) / 2
  distinct_paintings = 54 :=
by {
  let disks := Finset.range 7,
  let blue := 4,
  let red := 2,
  let green := 1,
  let total_paintings := (disks.card.choose blue) * ((disks.card - blue).choose red) * ((disks.card - blue - red).choose green),
  let reflection_fixed := 3,
  let distinct_paintings := (total_paintings + reflection_fixed) / 2,
  have h_disks : disks.card = 7 := sorry,
  have h_total : total_paintings = 105 := sorry,
  have h_reflection : reflection_fixed = 3 := sorry,
  have h_final : distinct_paintings = 54 := by {
    rw [h_total, h_reflection],
    norm_num,
  },
  exact h_final,
}

end num_distinct_paintings_l784_784332


namespace roots_of_equation_l784_784854

theorem roots_of_equation :
  {x : ℝ | -x * (x + 3) = x * (x + 3)} = {0, -3} :=
by
  sorry

end roots_of_equation_l784_784854


namespace average_first_14_even_numbers_l784_784912

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun x => 2 * (x + 1))

theorem average_first_14_even_numbers :
  let even_nums := first_n_even_numbers 14
  (even_nums.sum / even_nums.length = 15) :=
by
  sorry

end average_first_14_even_numbers_l784_784912


namespace smaller_square_side_length_l784_784079

theorem smaller_square_side_length :
  ∃ (a b c : ℕ), 
  let s := (a - (Real.sqrt b)) / c in 
  a > 0 ∧ b > 0 ∧ ¬ ∃ p : ℕ, (p > 1) ∧ (p^2 ∣ b) ∧ c > 0 ∧ s = 1 :=
sorry

end smaller_square_side_length_l784_784079


namespace mumu_identity_l784_784109

def f (m u : ℕ) : ℕ := 
  -- Assume f is correctly defined to match the number of valid Mumu words 
  -- involving m M's and u U's according to the problem's definition.
  sorry 

theorem mumu_identity (u m : ℕ) (h₁ : u ≥ 2) (h₂ : 3 ≤ m) (h₃ : m ≤ 2 * u) :
  f m u = f (2 * u - m + 1) u ↔ f m (u - 1) = f (2 * u - m + 1) (u - 1) :=
by
  sorry

end mumu_identity_l784_784109


namespace radical_simplification_l784_784532

noncomputable def simplify_radical_expression (q : ℝ) (hq : 0 ≤ q) : ℝ :=
  sqrt (11 * q) * sqrt (8 * q^3) * sqrt (9 * q^5)

theorem radical_simplification (q : ℝ) (hq : 0 ≤ q) :
  simplify_radical_expression q hq = 28 * q^4 * sqrt q := by
  sorry

end radical_simplification_l784_784532


namespace floor_sum_lemma_l784_784819

theorem floor_sum_lemma (x : Fin 1004 → ℝ) 
  (h : ∀ n : Fin 1004, x n + (n : ℝ) + 1 = ∑ i : Fin 1004, x i + 1005) 
  : ⌊|∑ i : Fin 1004, x i|⌋ = 501 :=
sorry

end floor_sum_lemma_l784_784819


namespace sum_binom_five_divisible_by_two_pow_l784_784354

theorem sum_binom_five_divisible_by_two_pow (n : ℕ) (h : 0 < n) :
  2^(n-1) ∣ ∑ k in Finset.range (n / 2), Nat.choose n (2 * k + 1) * 5^k := by
  sorry

end sum_binom_five_divisible_by_two_pow_l784_784354


namespace collinear_SKT_l784_784008

-- Definitions for the geometric figures and points
variables {A B C O I N S K M D T : Type}

-- Conditions assuming the type consists of necessary properties:
variables (circumcenter : Π (A B C : Type), O)
variables (incenter : Π (A B C : Type), I)
variables (arc_midpoint : Π (A B C : Type), N)
variables (line_intersection : Π (N I BC : Type), K)
variables (line_intersection : Π (N O BC : Type), M)
variables (second_intersection : Π (N O : Type), S)
variables (line_intersection : Π (I O : Type), D T)
variables (parallel : Π (AD : Type) (NI : Type), Prop)
variables (intersects_circle : Π (IO : Type), D T)

-- Stating the theorem
theorem collinear_SKT 
  (circumcenter_triangle : circumcenter A B C = O)
  (incenter_triangle : incenter A B C = I)
  (arc_midpoint_triangle : arc_midpoint A B C = N)
  (line_NI_BC : line_intersection N I BC = K)
  (line_NO_BC : line_intersection N O BC = M)
  (line_NO_circle : second_intersection N O = S)
  (line_IO_circle : intersects_circle IO = {D, T})
  (AD_parallel_NI : parallel AD NI) :
  collinear S K T := 
sorry

end collinear_SKT_l784_784008


namespace largest_prime_factor_of_2531_l784_784125

theorem largest_prime_factor_of_2531 :
  ∃ p, prime p ∧ p ∣ 2531 ∧ ∀ q, prime q → q ∣ 2531 → q ≤ p := sorry

end largest_prime_factor_of_2531_l784_784125


namespace arc_length_parametric_l784_784915

-- Conditions from the problem statement
noncomputable def x (t : ℝ) : ℝ := 10 * (Real.cos t) ^ 3
noncomputable def y (t : ℝ) : ℝ := 10 * (Real.sin t) ^ 3

-- Proving the arc length is equal to 15 given the conditions
theorem arc_length_parametric : 
  (∫ (t : ℝ) in 0..(Real.pi / 2), Real.sqrt ((Real.deriv x t) ^ 2 + (Real.deriv y t) ^ 2)) = 15 :=
by
  sorry

end arc_length_parametric_l784_784915


namespace floor_statements_true_l784_784572

noncomputable def floor (x : ℝ) : ℤ := 
if h : x < 0 then let n := Int.ofNat (Nat.floor (-x)) in -(n + 1)
else Int.ofNat (Nat.floor x)

theorem floor_statements_true :
  (∀ x : ℝ, floor (x + 2) = floor x + 2) ∧
  (∀ x : ℚ, ∀ y : ℝ, y ∉ ℚ → floor (x + y) = floor x + floor y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < 1 → 0 < y ∧ y < 1 → floor (x * y) = floor x * floor y) :=
by
  sorry

end floor_statements_true_l784_784572


namespace chapters_count_l784_784148

noncomputable def pages_total : ℕ := 81
noncomputable def first_chapter_pages : ℕ := 13
noncomputable def second_chapter_pages : ℕ := 68

theorem chapters_count (total pages_total) (first first_chapter_pages) (second second_chapter_pages) :
        (first + second = total) -> 2 = 2 := sorry

end chapters_count_l784_784148


namespace students_in_front_of_Seokjin_l784_784482

theorem students_in_front_of_Seokjin
  (n : ℕ)   -- Total number of students
  (b : ℕ)   -- Number of students behind Seokjin
  (h1 : n = 25)  -- Total students are 25
  (h2 : b = 13)  -- Students behind Seokjin are 13
  : ∃ f : ℕ, f = n - (b + 1) ∧ f = 11 := -- Number of students in front of Seokjin
by {
  use 11,
  rw [h1, h2],
  norm_num,
  sorry
}

end students_in_front_of_Seokjin_l784_784482


namespace perpendicular_bisector_eq_l784_784597

theorem perpendicular_bisector_eq :
  let A := (1 : ℝ, 2 : ℝ)
  let B := (3 : ℝ, 1 : ℝ)
  let M := ((1 + 3) / 2, (2 + 1) / 2)
  let slope_AB := (2 - 1) / (1 - 3)
  let slope_perpendicular := -1 / slope_AB
  (slope_perpendicular = 2) ∧ (M = (2, 3 / 2)) →
  ∀ x y : ℝ, 4 * x - 2 * y - 5 = 0 :=
by
  let A := (1, 2)
  let B := (3, 1)
  let M := ((1 + 3) / 2, (2 + 1) / 2)
  let slope_AB := (2 - 1) / (1 - 3)
  let slope_perpendicular := -1 / slope_AB
  assume h : (slope_perpendicular = 2) ∧ (M = (2, 3 / 2))
  sorry

end perpendicular_bisector_eq_l784_784597


namespace grade12_total_correct_l784_784754

variable (grade10_total grade11_total grade12_total : ℕ)
variable (sample_total sample_grade10 : ℕ)

-- Conditions
def conditions : Prop :=
  grade10_total = 1000 ∧
  grade11_total = 1200 ∧
  sample_total = 66 ∧
  sample_grade10 = 20

-- Theorem statement
theorem grade12_total_correct (h : conditions grade10_total grade11_total grade12_total sample_total sample_grade10) :
  grade12_total = 1100 :=
  sorry

end grade12_total_correct_l784_784754


namespace SA_bisects_QR_l784_784775

open Classical

noncomputable section

-- Definitions of the geometrical entities based on the conditions
variable (Ω : Type) [MetricSpace Ω] [NormedAddCommGroup Ω] [NormedSpace ℝ Ω]

-- Definitions for points, lines, and circles
variables (P A B Q R S O M: Ω) (γ : Set Ω)

-- Assumptions for the problem
variables (h1 : IsCircle γ)
variables (h2 : P ∉ γ)
variables (h3 : TangentTo PA γ P A)
variables (h4 : TangentTo PB γ P B)
variables (h5 : LineThrough P intersectsWithCircleAt γ Q R)
variables (h6 : S ∈ γ ∧ Parallel (LineThrough B S) (LineThrough Q R))

-- The theorem to be proven
theorem SA_bisects_QR :
  let M := intersectionPoint (LineThrough A S) (LineThrough Q R) in
  midpoint M Q R ↔ bisects (LineThrough A S) (LineSegment Q R) :=
sorry

end SA_bisects_QR_l784_784775


namespace sophia_transactions_l784_784378

theorem sophia_transactions : 
  let mabel_transactions := 90
  let anthony_transactions := mabel_transactions + 0.10 * mabel_transactions
  let cal_transactions := (2/3) * anthony_transactions
  let jade_transactions := cal_transactions + 19
  let sophia_transactions := jade_transactions + 0.50 * jade_transactions
  round sophia_transactions = 128 := 
by {
  let mabel_transactions := 90
  let anthony_transactions := mabel_transactions + 0.10 * mabel_transactions
  let cal_transactions := (2/3) * anthony_transactions
  let jade_transactions := cal_transactions + 19
  let sophia_transactions := jade_transactions + 0.50 * jade_transactions
  have : round sophia_transactions = 128 := sorry,
  exact this
}

end sophia_transactions_l784_784378


namespace intersection_M_N_is_correct_l784_784621

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784621


namespace base3_addition_proof_l784_784519

-- Define the base 3 numbers
def one_3 : ℕ := 1
def twelve_3 : ℕ := 1 * 3 + 2
def two_hundred_twelve_3 : ℕ := 2 * 3^2 + 1 * 3 + 2
def two_thousand_one_hundred_twenty_one_3 : ℕ := 2 * 3^3 + 1 * 3^2 + 2 * 3 + 1

-- Define the correct answer in base 3
def expected_sum_3 : ℕ := 1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3 + 0

-- The proof problem
theorem base3_addition_proof :
  one_3 + twelve_3 + two_hundred_twelve_3 + two_thousand_one_hundred_twenty_one_3 = expected_sum_3 :=
by
  -- Proof goes here
  sorry

end base3_addition_proof_l784_784519


namespace g_sqrt_50_l784_784789

noncomputable def g (x : ℝ) : ℝ :=
if x.isInt then 7 * x + 3 else ⌊x⌋ + 3 * x

theorem g_sqrt_50 : g (Real.sqrt 50) = 7 + 15 * Real.sqrt 2 :=
by
  sorry

end g_sqrt_50_l784_784789


namespace original_average_is_6_2_l784_784514

theorem original_average_is_6_2 (n : ℕ) (S : ℚ) (h1 : 6.2 = S / n) (h2 : 6.6 = (S + 4) / n) :
  6.2 = S / n :=
by
  sorry

end original_average_is_6_2_l784_784514


namespace intersection_M_N_l784_784626

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784626


namespace find_m_of_perpendicular_vectors_l784_784298

theorem find_m_of_perpendicular_vectors (m : ℝ) :
  let a := (2, m)
      b := (1, -1) in
  let c := (fst a + 2 * fst b, snd a + 2 * snd b) in
  b.1 * c.1 + b.2 * c.2 = 0 →
  m = 6 := 
by 
  intros
  let a := (2, m)
  let b := (1, -1)
  let c := (fst a + 2 * fst b, snd a + 2 * snd b)
  have H1 : b.1 * c.1 + b.2 * c.2 = 0 := ‹b.1 * c.1 + b.2 * c.2 = 0›
  dsimp [a, b, c] at H1
  sorry

end find_m_of_perpendicular_vectors_l784_784298


namespace total_glasses_l784_784180

theorem total_glasses
  (x y : ℕ)
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15) :
  12 * x + 16 * y = 480 :=
by
  sorry

end total_glasses_l784_784180


namespace find_k_l784_784340

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end find_k_l784_784340


namespace percentage_first_class_l784_784379

theorem percentage_first_class (total_passengers : ℕ) (female_percentage : ℝ) 
  (male_fraction_first_class : ℝ) (females_coach : ℕ) 
  (h_total_pos : total_passengers = 120) 
  (h_female_percentage : female_percentage = 0.4) 
  (h_male_fraction_first_class : male_fraction_first_class = 1/3) 
  (h_females_coach : females_coach = 40) : 
  let females_total : ℕ := total_passengers * (40/100)
  let females_first_class : ℕ := females_total - females_coach
  let total_first_class : ℕ := females_first_class * 3/2 in
  (total_first_class / total_passengers * 100 = 10) := 
by sorry

end percentage_first_class_l784_784379


namespace red_to_blue_ratio_l784_784970

variable (R B : ℕ)

-- Condition 1
def one_third_left_handed_red : ℕ := (1 / 3 : ℚ) * R

-- Condition 2
def two_thirds_left_handed_blue : ℕ := (2 / 3 : ℚ) * B

-- Condition 3
def fraction_left_handed (R B : ℕ) : ℚ := (one_third_left_handed_red R + two_thirds_left_handed_blue B) / (R + B)

theorem red_to_blue_ratio (h : fraction_left_handed R B = 0.431) : (R : ℚ) / B = 2.413 :=
by
  sorry

end red_to_blue_ratio_l784_784970


namespace range_of_m_for_subset_range_of_m_for_disjoint_l784_784371

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_of_m_for_subset (m : ℝ) :
  (∀ x : ℝ, B m x → A x) ↔ m ∈ set.Iic (3 : ℝ) :=
sorry

theorem range_of_m_for_disjoint (m : ℝ) :
  (∀ x : ℝ, ¬ (A x ∧ B m x)) ↔ m ∈ set.Iio 2 ∨ m ∈ set.Ioi 4 :=
sorry

end range_of_m_for_subset_range_of_m_for_disjoint_l784_784371


namespace path_of_C1_C2_l784_784749

-- Definitions of the geometric entities
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Stub definitions for incircle center and quarter circle construction
noncomputable def incircle_center (T : Triangle) : Point := sorry
noncomputable def quarter_circle (O : Point) (radius : ℝ) : Circle := sorry
noncomputable def tangent_at_point (C : Circle) (P : Point) : (Point × Point) := sorry
noncomputable def segment_contains (A B P : Point) : Prop := sorry

-- Definitions of points and circles in the problem
variable (O A B P Q1 Q2 C1 C2 F : Point)
variable (r : ℝ)

axiom right_angle_OA_OB : ∀ (O A B : Point), angle O A B = π/2

-- Setting up the problem configurations
def is_on_quarter_circle (P : Point) (C : Circle) : Prop := sorry
noncomputable def path_traced_by_C1 (A F : Point) : Set Point := sorry
noncomputable def path_traced_by_C2 (F B : Point) : Set Point := sorry

-- Main theorem to be proved
theorem path_of_C1_C2 :
  (∀ P : Point, is_on_quarter_circle P (quarter_circle O r) → 
   let (Q1, Q2) := tangent_at_point (quarter_circle O r) P 
   let C1 := incircle_center (Triangle.mk O P Q1)
   let C2 := incircle_center (Triangle.mk O P Q2) 
   ∃ F : Point,
     (segment_contains C1 A F) ∧
     (segment_contains C2 F B) ) :=
sorry

end path_of_C1_C2_l784_784749


namespace trig_expr_simplification_l784_784467

theorem trig_expr_simplification (α : ℝ) :
  2 - (sin (8 * α) / (sin (2 * α) ^ 4 - cos (2 * α) ^ 4)) = 4 * cos ((π / 4) - 2 * α) ^ 2 := 
by
  sorry

end trig_expr_simplification_l784_784467


namespace coeff_x6_in_expansion_l784_784453

theorem coeff_x6_in_expansion (x : ℝ) : 
  let C := ∑ k in finset.range (8 + 1), nat.choose 8 k * (x ^ k) * (2 ^ (8 - k))
  (coeff_in_expansion C 6) = 112 :=
by
  sorry

end coeff_x6_in_expansion_l784_784453


namespace shaded_region_area_l784_784118

noncomputable def shaded_area : ℝ :=
  let radius : ℝ := 15
  let sector_angle : ℝ := 75
  let total_area_of_two_sectors := 2 * (sector_angle / 360) * π * radius^2
  let triangle_area := radius^2 * (Real.sin (sector_angle * Real.pi / 180))
  2 * ((sector_angle / 360) * π * radius^2 - 0.5 * triangle_area)

theorem shaded_region_area :
  shaded_area = 93.37 * π - 217.16 :=
sorry

end shaded_region_area_l784_784118


namespace james_faster_than_john_l784_784014

theorem james_faster_than_john :
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds
  
  james_top_speed - john_top_speed = 2 :=
by
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds

  sorry

end james_faster_than_john_l784_784014


namespace intersection_M_N_l784_784603

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784603


namespace exists_monotonicity_b_range_l784_784284

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - 2 * a * x + Real.log x

theorem exists_monotonicity_b_range :
  ∀ (a : ℝ) (b : ℝ), 1 < a ∧ a < 2 →
  (∀ (x0 : ℝ), x0 ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2 →
   f a x0 + Real.log (a + 1) > b * (a^2 - 1) - (a + 1) + 2 * Real.log 2) →
   b ∈ Set.Iic (-1/4) :=
sorry

end exists_monotonicity_b_range_l784_784284


namespace clock_rings_in_january_l784_784963

theorem clock_rings_in_january :
  ∀ (days_in_january hours_per_day ring_interval : ℕ)
  (first_ring_time : ℕ) (january_first_hour : ℕ), 
  days_in_january = 31 →
  hours_per_day = 24 →
  ring_interval = 7 →
  january_first_hour = 2 →
  first_ring_time = 30 →
  (days_in_january * hours_per_day) / ring_interval + 1 = 107 := by
  intros days_in_january hours_per_day ring_interval first_ring_time january_first_hour
  sorry

end clock_rings_in_january_l784_784963


namespace josette_bought_three_bottles_l784_784352

/-- Define the cost of a certain number of bottles. -/
def cost_of_bottles (num_bottles : ℕ) (cost_per_four : ℝ) : ℝ :=
  (num_bottles / 4) * cost_per_four

/-- Prove that Josette bought 3 bottles for €1.50, given that four bottles cost €2. -/
theorem josette_bought_three_bottles : (∃ num_bottles : ℕ, cost_of_bottles num_bottles 2 = 1.5) :=
begin
  existsi 3,
  norm_num,
  sorry
end

end josette_bought_three_bottles_l784_784352


namespace sum_mean_median_mode_eq_78_l784_784896

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def median (l : List ℝ) : ℝ :=
  let len := l.length
  if len % 2 = 0 then
    (l.nth_le (len / 2) (Nat.div_lt_self (Nat.zero_le len) (by decide)) + l.nth_le (len / 2 - 1) (Nat.div_lt_self (by decide) (by decide))) / 2
  else
    l.nth_le (len / 2) (Nat.div_lt_self (Nat.zero_le len) (by decide))

noncomputable def mode (l : List ℝ) : ℝ :=
  l.argmax (λ x, l.count x)

theorem sum_mean_median_mode_eq_78 : sum [2, 3, 0, 3, 1, 4, 0, 3, 5, 2] / 10 + median [0, 0, 1, 2, 2, 3, 3, 3, 4, 5] + 3 = 7.8 :=
  by sorry

end sum_mean_median_mode_eq_78_l784_784896


namespace two_digit_factors_of_2_pow_18_minus_1_l784_784719

-- Define the main problem statement: 
-- How many two-digit factors does 2^18 - 1 have?

theorem two_digit_factors_of_2_pow_18_minus_1 : 
  ∃ n : ℕ, n = 5 ∧ ∀ f : ℕ, (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100) ↔ (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100 ∧ ∃ k : ℕ, (2^18 - 1) = k * f) :=
by sorry

end two_digit_factors_of_2_pow_18_minus_1_l784_784719


namespace equivalent_functions_l784_784463

-- Definitions of the functions
def f_A (x : ℝ) := real.sqrt ((x - 1)^2)
def g_A (x : ℝ) := x - 1

def f_B (x : ℝ) := x
def g_B (x : ℝ) := if x ≠ 0 then (x^2) / x else 0  -- ensuring g_B is defined piecewise to handle x = 0

def f_C (x : ℝ) := real.sqrt (x^2 - 1)
def g_C (x : ℝ) := real.sqrt (x + 1) * real.sqrt (x - 1)

def f_D (x : ℝ) := x - 1
def g_D (t : ℝ) := t - 1

-- Proposition stating which pair of functions are equivalent
theorem equivalent_functions :
  (∀ x : ℝ, f_A x = g_A x) ∧ 
  (∀ x : ℝ, f_B x = g_B x) ∧ 
  (∀ x : ℝ, f_C x = g_C x) ∧ 
  (∀ x : ℝ, f_D x = g_D x) ↔ 
  (∀ x : ℝ, f_D x = g_D x) :=
sorry  -- Proof to be filled in

end equivalent_functions_l784_784463


namespace sum_series_l784_784977

theorem sum_series : (∑ k in (Finset.range ∞), (3 ^ (2 ^ k)) / ((3 ^ 2) ^ (2 ^ k) - 1)) = 1 / 2 :=
by sorry

end sum_series_l784_784977


namespace remainder_when_expr_divided_by_9_l784_784370

theorem remainder_when_expr_divided_by_9 (n m p : ℤ)
  (h1 : n % 18 = 10)
  (h2 : m % 27 = 16)
  (h3 : p % 6 = 4) :
  (2 * n + 3 * m - p) % 9 = 1 := 
sorry

end remainder_when_expr_divided_by_9_l784_784370


namespace intersection_M_N_l784_784650

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784650


namespace hotel_made_correct_revenue_l784_784179

noncomputable def hotelRevenue : ℕ :=
  let totalRooms := 260
  let doubleRooms := 196
  let singleRoomCost := 35
  let doubleRoomCost := 60
  let singleRooms := totalRooms - doubleRooms
  let revenueSingleRooms := singleRooms * singleRoomCost
  let revenueDoubleRooms := doubleRooms * doubleRoomCost
  revenueSingleRooms + revenueDoubleRooms

theorem hotel_made_correct_revenue :
  hotelRevenue = 14000 := by
  sorry

end hotel_made_correct_revenue_l784_784179


namespace blocks_needed_l784_784926

   -- Define the dimensions of the rectangular prism
   def block_length := 6 -- inches
   def block_width := 2 -- inches
   def block_height := 1 -- inches

   -- Define the dimensions of the cylindrical sculpture
   def cylinder_diameter := 6 -- inches
   def cylinder_radius := cylinder_diameter / 2 -- inches
   def cylinder_height := 12 -- inches

   -- Calculate volumes
   def block_volume := block_length * block_width * block_height
   def cylinder_volume := Real.pi * (cylinder_radius ^ 2) * cylinder_height

   -- Calculate the number of blocks needed
   noncomputable def num_blocks := Nat.ceil (cylinder_volume / block_volume)

   -- Statement to prove
   theorem blocks_needed : num_blocks = 29 := by
     sorry
   
end blocks_needed_l784_784926


namespace sqrt_solution_l784_784727

theorem sqrt_solution (x : ℝ) : sqrt (5 + sqrt x) = 4 → x = 121 :=
by sorry

end sqrt_solution_l784_784727


namespace base_10_to_12_equivalence_l784_784542

theorem base_10_to_12_equivalence : 
  ∀ (n : ℕ), n = 144 → to_base 12 n = 100 :=
by
  sorry

end base_10_to_12_equivalence_l784_784542


namespace g_definition_l784_784998

-- Define f for n > 3 and specific definitions for initial values
def f (n : ℕ) : ℕ :=
  if n = 1 ∨ n = 2 ∨ n = 3 then 2
  else (Nat.find (λ k, ¬(k > 0 ∧ k ∣ n)))

-- Iteratively define fk
def f_iter : ℕ → (ℕ → ℕ)
| 0     := λ n, n
| (k+1) := λ n, f (f_iter k n)

-- Define g which is the smallest k such that f_iter k n = 2
def g (n : ℕ) : ℕ := Nat.find (λ k, f_iter k n = 2)

-- Prove that the determined g(n) returns the expected results
theorem g_definition (n : ℕ) : ∃ k : ℕ, f_iter k n = 2 := sorry

end g_definition_l784_784998


namespace total_workers_calculation_l784_784404

theorem total_workers_calculation :
  ∀ (N : ℕ), 
  (∀ (total_avg_salary : ℕ) (techs_salary : ℕ) (nontech_avg_salary : ℕ),
    total_avg_salary = 8000 → 
    techs_salary = 7 * 20000 → 
    nontech_avg_salary = 6000 →
    8000 * (7 + N) = 7 * 20000 + N * 6000 →
    (7 + N) = 49) :=
by
  intros
  sorry

end total_workers_calculation_l784_784404


namespace x_intercept_of_perpendicular_line_is_16_over_3_l784_784450

theorem x_intercept_of_perpendicular_line_is_16_over_3 :
  (∃ x : ℚ, (∃ y : ℚ, (4 * x - 3 * y = 12))
    ∧ (∃ x y : ℚ, (y = - (3 / 4) * x + 4 ∧ y = 0) ∧ x = 16 / 3)) :=
by {
  sorry
}

end x_intercept_of_perpendicular_line_is_16_over_3_l784_784450


namespace laura_change_l784_784353

theorem laura_change : 
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250
  (amount_given - total_cost) = 10 :=
by
  -- definitions from conditions
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250

  -- the statement we are proving
  show (amount_given - total_cost) = 10
  sorry

end laura_change_l784_784353


namespace exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l784_784937

-- Definitions based on the given conditions
def male_students := 3
def female_students := 2
def total_students := male_students + female_students

def at_least_1_male_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ male_students
def all_female_event := ∀ (n : ℕ), n ≤ female_students
def at_least_1_female_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ female_students
def all_male_event := ∀ (n : ℕ), n ≤ male_students
def exactly_1_male_event := ∃ (n : ℕ), n = 1 ∧ n ≤ male_students
def exactly_2_female_event := ∃ (n : ℕ), n = 2 ∧ n ≤ female_students

def mutually_exclusive (e1 e2 : Prop) : Prop := ¬ (e1 ∧ e2)
def complementary (e1 e2 : Prop) : Prop := e1 ∧ ¬ e2 ∨ ¬ e1 ∧ e2

-- Statement of the problem
theorem exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary :
  mutually_exclusive exactly_1_male_event exactly_2_female_event ∧ 
  ¬ complementary exactly_1_male_event exactly_2_female_event :=
by
  sorry

end exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l784_784937


namespace constant_term_in_binom_expansion_l784_784559

noncomputable theory
open Nat BigOperators

def binom_expansion_constant_term : ℤ :=
  let term := fun k => (-1 : ℤ)^k * (nat.choose 6 k) * x^(6 - 2 * k)
  let k := 3 -- We've solved 6 - 2k = 0, giving k = 3
  term k

theorem constant_term_in_binom_expansion :
  binom_expansion_constant_term = -20 :=
by
  sorry

end constant_term_in_binom_expansion_l784_784559


namespace find_a_from_expansion_l784_784691

theorem find_a_from_expansion :
  (∃ a : ℝ, ∀ x : ℝ, x ≠ 0 →
    let T_r := λ r : ℕ, (choose 5 r) * (-a) ^ r * x ^ ((5 - 2 * r) / 2)
    in T_r 1 = 30) →
  a = -6 :=
by
  sorry

end find_a_from_expansion_l784_784691


namespace intersection_M_N_is_correct_l784_784620

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784620


namespace ryan_learning_hours_l784_784554

theorem ryan_learning_hours :
  ∃ hours : ℕ, 
    (∀ e_hrs : ℕ, e_hrs = 2) → 
    (∃ c_hrs : ℕ, c_hrs = hours) → 
    (∀ s_hrs : ℕ, s_hrs = 4) → 
    hours = 4 + 1 :=
by
  sorry

end ryan_learning_hours_l784_784554


namespace days_c_worked_l784_784906

theorem days_c_worked 
    (days_a : ℕ) (days_b : ℕ) (wage_ratio_a : ℚ) (wage_ratio_b : ℚ) (wage_ratio_c : ℚ)
    (total_earnings : ℚ) (wage_c : ℚ) :
    days_a = 16 →
    days_b = 9 →
    wage_ratio_a = 3 →
    wage_ratio_b = 4 →
    wage_ratio_c = 5 →
    wage_c = 71.15384615384615 →
    total_earnings = 1480 →
    ∃ days_c : ℕ, (total_earnings = (wage_ratio_a / wage_ratio_c * wage_c * days_a) + 
                                 (wage_ratio_b / wage_ratio_c * wage_c * days_b) + 
                                 (wage_c * days_c)) ∧ days_c = 4 :=
by
  intros
  sorry

end days_c_worked_l784_784906


namespace length_of_XY_l784_784341

theorem length_of_XY
  (X Y Z : Type)
  [RightTriangle XYZ X Y Z]
  (hYZ : YZX_angle XYZ = 30)
  (hXZ : distance XZ = 15)
  : distance XY = 30 :=
sorry

end length_of_XY_l784_784341


namespace shopkeeper_profit_percentage_l784_784951

-- Definitions and conditions:
def buys_extra_weight (indicated : ℝ) : ℝ := indicated * 1.2
def sells_less_weight (claimed : ℝ) : ℝ := claimed * 0.9

-- Theorem statement:
theorem shopkeeper_profit_percentage :
  ∀ (cost_price_per_unit selling_price_per_unit : ℝ),
  let indicated := 100 in
  let bought_weight := buys_extra_weight indicated in
  let claimed_weight := 100 in
  let actual_sold_weight := sells_less_weight claimed_weight in
  let cost_price := cost_price_per_unit * indicated in
  let selling_price := selling_price_per_unit * claimed_weight in
  let profit := selling_price - cost_price in
  let profit_percentage := (profit / cost_price) * 100 in
  profit_percentage = 33.33 :=
by
  sorry

end shopkeeper_profit_percentage_l784_784951


namespace sin_alpha_value_l784_784273

theorem sin_alpha_value (α : ℝ) (h1 : sin (π / 2 + α) = -3 / 5) (h2 : 0 < α ∧ α < π) : 
  sin α = 4 / 5 := 
sorry

end sin_alpha_value_l784_784273


namespace binom_16_9_l784_784269

theorem binom_16_9 :
  (nat.choose 15 9 = 5005) →
  (nat.choose 15 8 = 6435) →
  (nat.choose 17 9 = 24310) →
  nat.choose 16 9 = 11440 :=
by
  intros h1 h2 h3
  -- Pascal's identity and other steps would follow here
  sorry

end binom_16_9_l784_784269


namespace range_of_a_l784_784289

open Real

noncomputable def f (a x : ℝ) := sqrt (x^2 - 2 * a * x + 3)

theorem range_of_a : { a : ℝ | ∀ x : ℝ, f a x ∈ ℝ } = { a : ℝ | -sqrt 3 ≤ a ∧ a ≤ sqrt 3 } :=
by
  -- Proof would go here
  sorry

end range_of_a_l784_784289


namespace exists_nat_not_divisible_by_two_but_divisible_by_others_l784_784373

theorem exists_nat_not_divisible_by_two_but_divisible_by_others :
  ∃ N : ℕ, (¬ (127 ∣ N) ∧ ¬ (128 ∣ N)) ∧ 
  (∀ k : ℕ, k ∈ (finset.range 151) → k ≠ 127 → k ≠ 128 → k > 0 → k ∣ N) :=
by
  sorry

end exists_nat_not_divisible_by_two_but_divisible_by_others_l784_784373


namespace cost_per_revision_is_correct_l784_784419

-- Define the conditions based on the problem.
def initial_typing_cost (pages: ℕ) (cost_per_page: ℕ): ℕ := pages * cost_per_page

def revision_cost (pages_once: ℕ) (pages_twice: ℕ) (cost_per_revision: ℕ): ℕ :=
  (pages_once * cost_per_revision) + (pages_twice * 2 * cost_per_revision)

def total_cost (initial_cost: ℕ) (revision_cost: ℕ): ℕ := initial_cost + revision_cost

def typing_service_problem (total_pages: ℕ) (pages_once: ℕ) (pages_twice: ℕ) (initial_cost_per_page: ℕ) (total_typing_cost: ℕ) : ℕ :=
  let pages_none := total_pages - (pages_once + pages_twice)
  let initial_cost := initial_typing_cost total_pages initial_cost_per_page
  let rev_cost := revision_cost pages_once pages_twice _
  have total := total_cost initial_cost rev_cost
  if total = total_typing_cost then rev_cost / (pages_once + (2 * pages_twice)) else _

-- A theorem that states the cost per page for each revision is $5
theorem cost_per_revision_is_correct: 
  typing_service_problem 100 30 20 10 1350 = 5 := 
by
  sorry

end cost_per_revision_is_correct_l784_784419


namespace calculation_result_l784_784535

theorem calculation_result :
  (-1:ℤ) ^ 2023 - 2 * Real.sin (Float.pi / 3) + | -Real.sqrt 3 | + 1 / (1 / 3) = 2 := by
  sorry

end calculation_result_l784_784535


namespace probability_red_before_green_l784_784499

theorem probability_red_before_green :
  let chips := ({0, 1, 2} : Finset ℕ) ∪ ({3, 4, 5} : Finset ℕ),
      red_chips := {0, 1, 2} : Finset ℕ,
      green_chips := {3, 4, 5} : Finset ℕ,
      total_arrangements := Nat.choose 6 3,
      favorable_arrangements := Nat.choose 5 3
  in
    (favorable_arrangements.toRational / total_arrangements.toRational) = 1 / 2 :=
by
  sorry

end probability_red_before_green_l784_784499


namespace number_of_students_l784_784085

theorem number_of_students 
  (N : ℕ)
  (avg_age : ℕ → ℕ)
  (h1 : avg_age N = 15)
  (h2 : avg_age 5 = 12)
  (h3 : avg_age 9 = 16)
  (h4 : N = 15 ∧ avg_age 1 = 21) : 
  N = 15 :=
by
  sorry

end number_of_students_l784_784085


namespace zeros_of_f_l784_784245

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 3 * x - 3 else (1 / 2) ^ x - 4

theorem zeros_of_f :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ ∀ x, f x = 0 → (x = 1 ∨ x = -2) :=
by
  sorry

end zeros_of_f_l784_784245


namespace intersection_M_N_l784_784656

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784656


namespace line_through_point_with_slope_l784_784090

theorem line_through_point_with_slope :
  ∃ k : ℝ, (k = real.tan (real.pi / 6)) ∧ (∀ x y : ℝ, ((y - 2) = k * (x - 1)) <-> (√3 * x - 3 * y + 6 - √3 = 0)) :=
by
  -- Setup conditions and proof
  sorry

end line_through_point_with_slope_l784_784090


namespace range_of_omega_l784_784115

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ x ∈ Ioo 0 (π/2), sin (ω * x + π/3) = 0) ↔ ω ∈ Ioo (4/3 : ℝ) (10/3 : ℝ) :=
by sorry 

end range_of_omega_l784_784115


namespace symmetric_circle_line_l784_784266

theorem symmetric_circle_line {a b : ℝ} : 
  (∃ x y : ℝ, x^2 + y^2 + 2x - 4y + 1 = 0 ∧ 2a * x - b * y + 2 = 0) → 
  a + b = 1 → 
  ab ≤ 1/4 := 
by
  sorry

end symmetric_circle_line_l784_784266


namespace log_base_3_of_0_point_375_is_negative_0_point_8927_l784_784548

-- Define the problem conditions
def log_base_3_of_0_point_375 : Real := Real.log 0.375 / Real.log 3

-- State the theorem to prove
theorem log_base_3_of_0_point_375_is_negative_0_point_8927 :
  log_base_3_of_0_point_375 = -0.8927 := 
sorry

end log_base_3_of_0_point_375_is_negative_0_point_8927_l784_784548


namespace find_special_number_exists_l784_784557

def geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

def arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

theorem find_special_number_exists :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
            (let x := n / 100, y := (n % 100) / 10, z := n % 10 in 
              geometric_progression x y z ∧
              (n - 792 = 100 * z + 10 * y + x) ∧
              (let new_x := x - 4 in
                arithmetic_progression new_x y z) ∧
              n = 931) :=
by
  sorry

end find_special_number_exists_l784_784557


namespace summer_has_150_degrees_l784_784818

-- Define the condition that Summer has five more degrees than Jolly,
-- and the combined number of degrees they both have is 295.
theorem summer_has_150_degrees (S J : ℕ) (h1 : S = J + 5) (h2 : S + J = 295) : S = 150 :=
by sorry

end summer_has_150_degrees_l784_784818


namespace total_spent_on_index_cards_l784_784981

-- Definitions for conditions
def index_cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

-- Theorem to be proven
theorem total_spent_on_index_cards :
  let total_students := students_per_class * periods_per_day
  let total_cards := total_students * index_cards_per_student
  let packs_needed := total_cards / cards_per_pack
  let total_cost := packs_needed * cost_per_pack
  total_cost = 108 :=
by
  sorry

end total_spent_on_index_cards_l784_784981


namespace jordan_rectangle_width_eq_30_l784_784985

variable (length_C : ℕ) (width_C : ℕ)
variable (length_J : ℕ) (area : ℕ)

noncomputable def carol_rect_area (length_C width_C : ℕ) := length_C * width_C
noncomputable def jordan_rect_width (length_J area : ℕ) := area / length_J

theorem jordan_rectangle_width_eq_30 :
  ∀ (length_C width_C length_J area : ℕ), 
    length_C = 8 →
    width_C = 15 →
    length_J = 4 →
    area = carol_rect_area length_C width_C →
    jordan_rect_width length_J area = 30 :=
by
  intros length_C width_C length_J area h_length_C h_width_C h_length_J h_area
  rw [h_length_C, h_width_C, h_length_J] at *
  rw [←h_area]
  sorry

end jordan_rectangle_width_eq_30_l784_784985


namespace solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l784_784817

-- For the first polynomial equation
theorem solve_cubic_eq_a (x : ℝ) : x^3 - 3 * x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

-- For the second polynomial equation
theorem solve_cubic_eq_b (x : ℝ) : x^3 - 19 * x - 30 = 0 ↔ x = 5 ∨ x = -2 ∨ x = -3 :=
by sorry

-- For the third polynomial equation
theorem solve_cubic_eq_c (x : ℝ) : x^3 + 4 * x^2 + 6 * x + 4 = 0 ↔ x = -2 :=
by sorry

end solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l784_784817


namespace num_ordered_tuples_satisfying_condition_l784_784565

theorem num_ordered_tuples_satisfying_condition : 
  let S := (a : Fin 13 → ℤ) → Σ i, a i;
  (∃ (a : Fin 13 → ℤ),
    ∀ i : Fin 13, (a i)^2 = S a - a i) ↔ 572 :=
by sorry

end num_ordered_tuples_satisfying_condition_l784_784565


namespace gcd_lcm_45_150_l784_784841

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 :=
by
  sorry

end gcd_lcm_45_150_l784_784841


namespace red_light_probability_l784_784517

theorem red_light_probability :
  let red_duration := 30
  let yellow_duration := 5
  let green_duration := 40
  let total_duration := red_duration + yellow_duration + green_duration
  let probability_of_red := (red_duration:ℝ) / total_duration
  probability_of_red = 2 / 5 := by
    sorry

end red_light_probability_l784_784517


namespace intersection_M_N_l784_784628

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784628


namespace math_problem_l784_784780

-- Define the function g with its conditions
def g : ℝ → ℝ := sorry
axiom g_condition1 : g 2 = 2
axiom g_condition2 : ∀ x y : ℝ, g (x * y + g x) = y * g x + g x

-- Define the values of m and t, and the result
def m : ℕ := 1
def t : ℝ := 1 / 2

-- Theorem stating the required result
theorem math_problem : (m * t) = 1 / 2 :=
by
  -- Skipping the proof
  sorry

end math_problem_l784_784780


namespace determine_values_l784_784204

theorem determine_values (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : (x^2 + y^2 = 697) ∧ (x + y = Real.sqrt 769) :=
by
  -- Proof goes here
  sorry

end determine_values_l784_784204


namespace distance_of_intersections_eq_four_l784_784202

noncomputable theory

open Real EuclideanSpace

def parabola (p: ℝ × ℝ) : Prop := p.snd^2 = 12 * p.fst
def circle (p: ℝ × ℝ) : Prop := p.fst^2 + p.snd^2 - 4 * p.fst - 6 * p.snd + 3 = 0

theorem distance_of_intersections_eq_four :
  ∀ (A B : ℝ × ℝ),
    parabola A ∧ circle A ∧ parabola B ∧ circle B →
    dist A B = 4 :=
sorry

end distance_of_intersections_eq_four_l784_784202


namespace proof_statement_l784_784742

def sum_of_divisors (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum id

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def problem_statement : Prop :=
  ∃ count : ℕ,
    count = ((Finset.range 2011).filter (λ i, 
      let sqrt_i := Int.sqrt i in 
      i = sqrt_i * sqrt_i ∧ is_prime sqrt_i ∧ sum_of_divisors i = 1 + sqrt_i + i)).card ∧ 
    count = 14

theorem proof_statement : problem_statement := 
sorry

end proof_statement_l784_784742


namespace equation_of_circle_l784_784582

theorem equation_of_circle :
  ∀ (C P : ℝ × ℝ), P = (-2, 1) → (∀ (x y : ℝ), C = (x + x - (-2), y + y - 1)) →
    (∀ (A B : ℝ × ℝ), (3 * A.1 + 4 * A.2 - 11 = 0) ∧ (3 * B.1 + 4 * B.2 - 11 = 0) ∧ (dist A B = 6)) →
      (∃ (x y : ℝ), circle_eq : (x^2 + (y + 1)^2 = 18)) :=
sorry

end equation_of_circle_l784_784582


namespace collinear_points_l784_784120

/-- 
Two squares and an isosceles triangle are arranged such that the vertex K of the larger square 
lies on the side of the triangle. Assume the isosceles triangle has an axis of symmetry by which 
the vertex K of the larger square transitions to the point C and the point D transitions to 
point A on reflection. Prove that the points A, B, and C are collinear. 
-/
theorem collinear_points (A B C K D : Point) 
(iso_triangle: IsIsoscelesTriangle) 
(square1 square2: IsSquare) 
(h1: K ∈ iso_triangle.side) 
(h2: ReflectsInAxis iso_triangle.symmetry_axis K C) 
(h3: ReflectsInAxis iso_triangle.symmetry_axis D A) 
: Collinear A B C := 
sorry

end collinear_points_l784_784120


namespace lowest_number_of_students_l784_784904

theorem lowest_number_of_students (n : ℕ) (h1 : n % 18 = 0) (h2 : n % 24 = 0) : n = 72 := by
  sorry

end lowest_number_of_students_l784_784904


namespace rationalize_denominator_l784_784064

/-- Rationalizing the denominator of an expression involving cube roots -/
theorem rationalize_denominator :
  (1 : ℝ) / (real.cbrt 3 + real.cbrt 27) = real.cbrt 9 / (12 : ℝ) :=
by
  -- Define conditions
  have h1 : real.cbrt 27 = 3 * real.cbrt 3, by sorry,
  -- Proof of the equality, skipped using sorry
  sorry

end rationalize_denominator_l784_784064


namespace find_x_squared_plus_y_squared_l784_784725

theorem find_x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (x - y)^2 = 49) (h2 : x * y = -12) : x^2 + y^2 = 25 := 
by 
  sorry

end find_x_squared_plus_y_squared_l784_784725


namespace herd_total_cows_l784_784934

noncomputable def total_cows (n : ℕ) : Prop :=
  let fraction_first_son := 1 / 3
  let fraction_second_son := 1 / 5
  let fraction_third_son := 1 / 9
  let fraction_combined := fraction_first_son + fraction_second_son + fraction_third_son
  let fraction_fourth_son := 1 - fraction_combined
  let cows_fourth_son := 11
  fraction_fourth_son * n = cows_fourth_son

theorem herd_total_cows : ∃ n : ℕ, total_cows n ∧ n = 31 :=
by
  existsi 31
  sorry

end herd_total_cows_l784_784934


namespace intersection_M_N_l784_784647

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784647


namespace prove_no_consistent_solution_l784_784143

noncomputable def no_consistent_solution (x y z t : ℤ) : Prop :=
  x + z + t = 200 ∧ y + z + t = 300 ∧ t = 500 ∧ z = 600 → x < 0 ∧ y < 0

theorem prove_no_consistent_solution : ∃ x y z t : ℤ, no_consistent_solution x y z t :=
begin
  use [-900, -800, 600, 500],
  sorry
end

end prove_no_consistent_solution_l784_784143


namespace cong_squares_no_cylindrical_folding_l784_784434

theorem cong_squares_no_cylindrical_folding (left_of_leftmost right_of_rightmost above_middle below_middle : Prop) :
  ¬((left_of_leftmost ∧ right_of_rightmost) ∨ (left_of_leftmost ∧ above_middle) ∨ (left_of_leftmost ∧ below_middle) ∨ 
    (right_of_rightmost ∧ above_middle) ∨ (right_of_rightmost ∧ below_middle) ∨ (above_middle ∧ below_middle)) ∧
  ¬((left_of_leftmost ∨ right_of_rightmost ∨ above_middle ∨ below_middle) ↔ true) :=
begin
  sorry
end

end cong_squares_no_cylindrical_folding_l784_784434


namespace exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l784_784211

theorem exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃ (x : ℕ), x % 14 = 0 ∧ 625 <= x ∧ x <= 640 ∧ x = 630 := 
by 
  sorry

end exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l784_784211


namespace area_of_given_sector_l784_784826

-- Define the conditions based on the problem statement
def circumference_of_sector := 6 -- cm
def central_angle_of_sector := 1 -- radian

-- Define the formula to compute the area of the sector
def area_of_sector (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

-- State the theorem we want to prove
theorem area_of_given_sector : 
  ∃ r : ℝ, (r + r + (r * central_angle_of_sector) = circumference_of_sector) ∧ 
    area_of_sector r central_angle_of_sector = 2 :=
sorry

end area_of_given_sector_l784_784826


namespace cylinder_height_l784_784093

theorem cylinder_height (r h : ℝ) (SA : ℝ) 
  (hSA : SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h) 
  (hr : r = 3) (hSA_val : SA = 36 * Real.pi) : 
  h = 3 :=
by
  sorry

end cylinder_height_l784_784093


namespace max_students_6_questions_3_options_l784_784041

theorem max_students_6_questions_3_options 
  (q : ℕ) (o : ℕ) (condition : ∀ (s1 s2 s3 : ℕ), s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 → 
    ∃ (i : ℕ), (i < q) ∧ {s1, s2, s3} ⊆ {option1, option2, option3})
  : q = 6 ∧ o = 3 → n ≤ 13 :=
sorry

end max_students_6_questions_3_options_l784_784041


namespace simplify_trig_expression_l784_784391

theorem simplify_trig_expression :
  7 * 8 * (sin 10 * pi / 180 + sin 20 * pi / 180) / (cos 10 * pi / 180 + cos 20 * pi / 180)
  = 56 * tan (15 * pi / 180) := 
by
  sorry

end simplify_trig_expression_l784_784391


namespace tortoise_age_l784_784166

-- Definitions based on the given problem conditions
variables (a b c : ℕ)

-- The conditions as provided in the problem
def condition1 (a b : ℕ) : Prop := a / 4 = 2 * a - b
def condition2 (b c : ℕ) : Prop := b / 7 = 2 * b - c
def condition3 (a b c : ℕ) : Prop := a + b + c = 264

-- The main theorem to prove
theorem tortoise_age (a b c : ℕ) (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 a b c) : b = 77 :=
sorry

end tortoise_age_l784_784166


namespace intersection_M_N_l784_784678

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784678


namespace general_formula_and_lambda_range_l784_784328

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a1 + (n - 1) * d

def Sn (n : ℕ) : ℤ := n * (n + 2)

def Tn (n : ℕ) : ℚ :=
  (1 / (2 : ℚ)) * (1 + (1 : ℚ) / 2 - (1 : ℚ) / (n + 1) - (1 : ℚ) / (n + 2))

theorem general_formula_and_lambda_range (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) (λ : ℚ) :
  a 1 + a 3 = 10 →
  a 4 = 9 →
  Tn n < 3 * λ^2 + (9 / 4) * λ →
  (∀ n : ℕ, n > 0 → a n = 2 * ↑n + 1) ∧
  (λ ∈ set.Icc (-∞ : ℚ) (-1: ℚ) ∪ set.Icc (1 / 4) (∞ : ℚ)) :=
by
  sorry

end general_formula_and_lambda_range_l784_784328


namespace range_of_c_l784_784292

theorem range_of_c (b c : ℝ) : (y : ℝ → ℝ) (x : ℝ) :=
  (y = x^2 + b * x + c) ∧
  (axis : ℝ) (quad_eq : ℝ → ℝ → ℝ → ℝ) :=
  axis = 2 ∧ quad_eq = λ x b c => -x^2 - b * x - c ∧
  (h1 : quad_eq has two equal real roots within the range -1 < x < 3) :=
  b = -4 ∧ ((-5 < c ∧ c ≤ 3) ∨ (c = 4)) :=
by intros; sorry

end range_of_c_l784_784292


namespace interest_rate_first_year_l784_784218

theorem interest_rate_first_year :
  ∃ R : ℝ, 
  ∀ (P : ℝ) (n : ℝ) (r2 : ℝ) (A : ℝ),
    P = 4000 →
    n = 2 →
    r2 = 0.05 →
    A = 4368 →
    (P + (P * R) + (P * R + P) * r2 = A) →
    R = 0.04 :=
begin
  use 0.04, sorry
end

end interest_rate_first_year_l784_784218


namespace total_workers_l784_784086

theorem total_workers (W : ℕ) (H1 : (∑ i in finset.range W, 1000) / W = 1000)
  (H2 : (∑ i in finset.range 10, 1200) / 10 = 1200)
  (H3 : (∑ i in finset.range (W - 10), 820) / (W - 10) = 820) :
  W = 21 :=
sorry

end total_workers_l784_784086


namespace solution_set_l784_784698

def f (x : ℝ) : ℝ := abs x - x + 1

theorem solution_set (x : ℝ) : f (1 - x^2) > f (1 - 2 * x) ↔ x > 2 ∨ x < -1 := by
  sorry

end solution_set_l784_784698


namespace range_of_c_div_a_l784_784682

-- Define the conditions and variables
variables (a b c : ℝ)

-- Define the given conditions
def conditions : Prop :=
  (a ≥ b ∧ b ≥ c) ∧ (a + b + c = 0)

-- Define the range of values for c / a
def range_for_c_div_a : Prop :=
  -2 ≤ c / a ∧ c / a ≤ -1/2

-- The theorem statement to prove
theorem range_of_c_div_a (h : conditions a b c) : range_for_c_div_a a c := 
  sorry

end range_of_c_div_a_l784_784682


namespace extreme_point_tangent_line_max_value_non_monotonic_l784_784702

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem extreme_point (a b : ℝ) (h_extreme : ∃ x, f' x = 0) : a = 0 ∨ a = 2 :=
sorry

theorem tangent_line_max_value (a : ℝ) (b : ℝ) (h_tangent : ∃ x y, x + y - 3 = 0 ∧ y = f(1)) : 
  (∃ x, x ∈ set.Icc (-2 : ℝ) 4 ∧ (∀ y, y ∈ set.Icc (-2 : ℝ) 4 → f y ≤ f x) ∧ f x = 8) :=
sorry

theorem non_monotonic (a : ℝ) (b : ℝ) (h_a_nonzero : a ≠ 0) (h_nonmonotonic : ∃ x y, x ≠ y → (x ∈ set.Ioo (-1 : ℝ) 1 ∧ y ∈ set.Ioo (-1 : ℝ) 1 ∧ f' x = 0 ∧ f' y = 0)) :
  a ∈ set.Ioo (-2 : ℝ) 0 ∪ set.Ioo 0 2 :=
sorry

end extreme_point_tangent_line_max_value_non_monotonic_l784_784702


namespace distance_to_other_focus_l784_784261

theorem distance_to_other_focus (x y : ℝ) (P : ℝ × ℝ) (a b c : ℝ) (F1 F2 : ℝ × ℝ) :
  (a = 4) ∧ (b = 2) ∧ (c = √(a^2 - b^2)) ∧
  (P ∈ {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 4) = 1}) ∧
  (dist P F1 = 3) ∧
  (F1 = (c, 0)) ∧ (F2 = (-c, 0)) →
  dist P F2 = 5 :=
by
  -- Proof is omitted
  sorry

end distance_to_other_focus_l784_784261


namespace smallest_possible_fourth_number_l784_784950

theorem smallest_possible_fourth_number 
  (a b : ℕ) 
  (h1 : 21 + 34 + 65 = 120)
  (h2 : 1 * (21 + 34 + 65 + 10 * a + b) = 4 * (2 + 1 + 3 + 4 + 6 + 5 + a + b)) :
  10 * a + b = 12 := 
sorry

end smallest_possible_fourth_number_l784_784950


namespace derivative_of_cos_pi_over_3_l784_784732

-- Define y
def y : ℝ := Real.cos (Real.pi / 3)

-- State and prove that the derivative of y with respect to x is 0
theorem derivative_of_cos_pi_over_3 : deriv y = 0 :=
by
  -- Since y is constant
  have h : ∀ x : ℝ, y = Real.cos (Real.pi / 3) := λ x, rfl
  -- The derivative of a constant function is 0
  exact deriv_const y sorry

end derivative_of_cos_pi_over_3_l784_784732


namespace area_KBC_l784_784334

theorem area_KBC (ABCDEF_equiangular : Prop) 
  (ABJI_square : ∃ x : ℝ, x^2 = 25) 
  (FEHG_square : ∃ y : ℝ, y^2 = 32) 
  (triangle_JBK_isosceles : ∃ z : ℝ, ∃ w : ℝ, w = z ∧ ∠ JBK = 45) 
  (FE_EQ_BC : ∃ u : ℝ, u = √32) : 
  ∃ area : ℝ, area = 5 * √2 * (√6 - √2) :=
begin
  sorry
end

end area_KBC_l784_784334


namespace function_passes_through_fixed_point_l784_784410

theorem function_passes_through_fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : (2 - a^(0 : ℝ) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l784_784410


namespace other_drain_rate_l784_784923

theorem other_drain_rate 
  (tank_capacity_kL : ℕ)
  (initial_tank_fill_kL : ℚ)
  (fill_rate_pipe_kLpm : ℚ)
  (drain_rate_first_kLpm : ℚ)
  (time_to_fill_minutes : ℕ)
  (net_fill_rate_kLpm : ℚ) :
  tank_capacity_kL = 10000 →
  fill_rate_pipe_kLpm = 0.5 →
  drain_rate_first_kLpm = 1/6 →
  time_to_fill_minutes = 60 →
  (initial_tank_fill_kL * 2 = tank_capacity_kL) →
  net_fill_rate_kLpm = fill_rate_pipe_kLpm - drain_rate_first_kLpm →
  initial_tank_fill_kL / time_to_fill_minutes = net_fill_rate_kLpm - 
  	(Net_fill_rate_needed) →
  (initial_tank_fill_kL / time_to_fill_minutes = 5 / 60) →
  net_fill_rate_needed = 5 / 60 → 
  other_drain_rate : ℚ :=
sorry

end other_drain_rate_l784_784923


namespace min_polyline_distance_between_circle_and_line_l784_784002

def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

def on_line (Q : ℝ × ℝ) : Prop :=
  2 * Q.1 + Q.2 = 2 * Real.sqrt 5

theorem min_polyline_distance_between_circle_and_line :
  ∃ P Q, on_circle P ∧ on_line Q ∧ polyline_distance P Q = (Real.sqrt 5) / 2 :=
by
  sorry

end min_polyline_distance_between_circle_and_line_l784_784002


namespace solve_for_y_l784_784392

theorem solve_for_y (y : ℕ) : (1000^4 = 10^y) → y = 12 :=
by {
  sorry
}

end solve_for_y_l784_784392


namespace error_percentage_area_l784_784910

theorem error_percentage_area (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.06 * L,
      W' := 0.95 * W,
      A := L * W,
      A' := L' * W',
      error_pct := ((A' - A) / A) * 100
  in error_pct = 0.7 :=
by
  sorry

end error_percentage_area_l784_784910


namespace episodes_per_season_l784_784504

theorem episodes_per_season (S : ℕ) (E : ℕ) (H1 : S = 12) (H2 : 2/3 * E = 160) : E / S = 20 :=
by
  sorry

end episodes_per_season_l784_784504


namespace solve_ineq_l784_784217

noncomputable def inequality (x : ℝ) : Prop :=
  (x^2 / (x+1)) ≥ (3 / (x+1) + 3)

theorem solve_ineq :
  { x : ℝ | inequality x } = { x : ℝ | x ≤ -6 ∨ (-1 < x ∧ x ≤ 3) } := sorry

end solve_ineq_l784_784217


namespace coloring_schemes_l784_784882

theorem coloring_schemes (n : ℕ) (h : n ≥ 2) : 
  ∃ a_n : ℕ, a_n = 5 * (-1)^n + 5^n  :=
by
  use 5 * (-1)^n + 5^n
  sorry

end coloring_schemes_l784_784882


namespace minimum_slope_l784_784546

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

theorem minimum_slope : ∃ x : ℝ, (f' x = 3) ∧ (∀ y : ℝ, f' y ≥ 3) := sorry

end minimum_slope_l784_784546


namespace sqrt_mul_eq_l784_784460

theorem sqrt_mul_eq {a b : ℝ} (ha: 0 ≤ a) (hb: 0 ≤ b) : Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) :=
by sorry

example : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 :=
sqrt_mul_eq (by linarith) (by linarith)

end sqrt_mul_eq_l784_784460


namespace least_cardinality_is_9_l784_784221

def satisfies_conditions (A : Set ℕ) : Prop :=
  1 ∈ A ∧ 100 ∈ A ∧ (∀ n ∈ A, n ≠ 1 → ∃ x y ∈ A, n = x + y)

theorem least_cardinality_is_9 : ∃ A : Set ℕ, satisfies_conditions A ∧ A.card = 9 :=
by
  sorry

end least_cardinality_is_9_l784_784221


namespace find_f_8_5_l784_784833

noncomputable def f : ℝ → ℝ :=
sorry

axiom periodic : ∀ x : ℝ, f(x + 4) = f(x)

axiom initial_value : f 0.5 = 9

theorem find_f_8_5 : f 8.5 = 9 :=
by
  sorry

end find_f_8_5_l784_784833


namespace disjoint_subsets_same_sum_l784_784887

/-- 
Given a set of 10 distinct integers between 1 and 100, 
there exist two disjoint subsets of this set that have the same sum.
-/
theorem disjoint_subsets_same_sum : ∃ (x : Finset ℤ), (x.card = 10) ∧ (∀ i ∈ x, 1 ≤ i ∧ i ≤ 100) → 
  ∃ (A B : Finset ℤ), (A ⊆ x) ∧ (B ⊆ x) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by
  sorry

end disjoint_subsets_same_sum_l784_784887


namespace admission_counts_l784_784326

-- Define the total number of ways to admit students under given conditions.
def ways_of_admission : Nat := 1518

-- Statement of the problem: given conditions, prove the result
theorem admission_counts (n_colleges : Nat) (n_students : Nat) (admitted_two_colleges : Bool) : 
  n_colleges = 23 → 
  n_students = 3 → 
  admitted_two_colleges = true →
  ways_of_admission = 1518 :=
by
  intros
  sorry

end admission_counts_l784_784326


namespace magnitude_a_plus_b_when_alpha_30_vectors_a_plus_b_and_a_minus_b_perpendicular_angle_between_a_and_b_is_60_implies_alpha_eq_pi_over_3_l784_784297

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vec_b : ℝ × ℝ := (-1 / 2 : ℝ, Real.sqrt 3 / 2)

theorem magnitude_a_plus_b_when_alpha_30 :
  let α := Real.pi / 6 in
  let a := vec_a α in
  let b := vec_b in
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 2 := by
  let a_alpha := vec_a (Real.pi / 6)
  let b := vec_b
  sorry

theorem vectors_a_plus_b_and_a_minus_b_perpendicular (α : ℝ) (h : 0 < α ∧ α < Real.pi / 2) :
  let a := vec_a α in
  let b := vec_b in
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0 := by
  let a := vec_a α
  let b := vec_b
  sorry

theorem angle_between_a_and_b_is_60_implies_alpha_eq_pi_over_3 (α : ℝ) 
  (hα : 0 < α ∧ α < Real.pi / 2) 
  (h_angle : (vec_a α).1 * vec_b.1 + (vec_a α).2 * vec_b.2 = 1 / 2) :
  α = Real.pi / 3 := by
  let a := vec_a α
  let b := vec_b
  sorry

end magnitude_a_plus_b_when_alpha_30_vectors_a_plus_b_and_a_minus_b_perpendicular_angle_between_a_and_b_is_60_implies_alpha_eq_pi_over_3_l784_784297


namespace combined_budget_expenses_average_l784_784541

-- Definition of project budgets and expenditures
def projectA_budget : ℝ := 42000
def projectA_spent : ℝ := 23700

def projectB_budget_eur : ℝ := 56000
def projectB_spent_eur : ℝ := 33000
def eur_to_usd : ℝ := 1.1

def projectB_budget : ℝ := projectB_budget_eur * eur_to_usd
def projectB_spent : ℝ := projectB_spent_eur * eur_to_usd

def projectC_budget_gbp : ℝ := 24000
def projectC_spent_gbp : ℝ := 11000
def gbp_to_usd : ℝ := 1.3

def projectC_budget : ℝ := projectC_budget_gbp * gbp_to_usd
def projectC_spent : ℝ := projectC_spent_gbp * gbp_to_usd

-- Calculations for remaining budget
def projectA_remaining : ℝ := projectA_budget - projectA_spent
def projectB_remaining : ℝ := projectB_budget_eur - projectB_spent_eur * eur_to_usd
def projectC_remaining : ℝ := projectC_budget_gbp - projectC_spent_gbp * gbp_to_usd

-- Combined total budget and expenses in USD
def combined_total_budget : ℝ := projectA_budget + projectB_budget + projectC_budget
def combined_total_expenses : ℝ := projectA_spent + projectB_spent + projectC_spent

-- Average percentage of budgets spent
def average_percentage_spent : ℝ := ((projectA_spent / projectA_budget) * 100 + (projectB_spent / projectB_budget) * 100 + (projectC_spent / projectC_budget) * 100) / 3

theorem combined_budget_expenses_average : 
  combined_total_budget = 134800 ∧ 
  combined_total_expenses = 74300 ∧ 
  average_percentage_spent ≈ 53.73 / 100 :=
by
  sorry

end combined_budget_expenses_average_l784_784541


namespace min_value_inequality_l784_784840

theorem min_value_inequality (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : 3 * m + n = 1) : 
  ∃ (c : ℝ), c = 5 + 2 * real.sqrt 6 ∧ (∀ x y : ℝ, (3 * x + y = 1) → (x > 0) → (y > 0) → (5 + 2 * real.sqrt 6 ≤ (1 / x) + (2 / y))) := 
by 
  sorry

end min_value_inequality_l784_784840


namespace miles_to_drive_l784_784473

theorem miles_to_drive (total_miles : ℕ) (miles_driven : ℕ) : total_miles = 1200 → miles_driven = 642 → total_miles - miles_driven = 558 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry -- Proof goes here

end miles_to_drive_l784_784473


namespace binomial_expansion_fifth_term_max_l784_784329

theorem binomial_expansion_fifth_term_max (x : ℝ) (n : ℕ)
  (h : ∀ k : ℕ, k ≠ 4 → binomial_coefficient n k < binomial_coefficient n 4)
  : n = 8 :=
sorry

end binomial_expansion_fifth_term_max_l784_784329


namespace female_officers_count_l784_784044

-- Defining variables from the conditions
variable (TotalOfficers : ℕ)
variable (FemaleOfficersOnDuty : ℕ)
variable (MaleOfficersOnDuty : ℕ)
variable (TotalFemaleOfficers : ℕ)

-- Given conditions
def condition1 : Prop := 0.40 * TotalFemaleOfficers = 120
def condition2 : Prop := TotalOfficers = 240
def condition3 : Prop := FemaleOfficersOnDuty = 0.5 * TotalOfficers

-- Theorem to prove
theorem female_officers_count (c1 : condition1) (c2 : condition2) (c3 : condition3) : TotalFemaleOfficers = 300 :=
sorry

end female_officers_count_l784_784044


namespace ned_shirts_problem_l784_784039

theorem ned_shirts_problem
  (long_sleeve_shirts : ℕ)
  (total_shirts_washed : ℕ)
  (total_shirts_had : ℕ)
  (h1 : long_sleeve_shirts = 21)
  (h2 : total_shirts_washed = 29)
  (h3 : total_shirts_had = total_shirts_washed + 1) :
  ∃ short_sleeve_shirts : ℕ, short_sleeve_shirts = total_shirts_had - total_shirts_washed - 1 :=
by
  sorry

end ned_shirts_problem_l784_784039


namespace max_value_fraction_l784_784695

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 1 = 0

theorem max_value_fraction (a b : ℝ) (H : circle_eq a b) :
  ∃ t : ℝ, -1/2 ≤ t ∧ t ≤ 1/2 ∧ b = t * (a - 3) ∧ t = 1 / 2 :=
by sorry

end max_value_fraction_l784_784695


namespace original_solution_sugar_percentage_l784_784515

theorem original_solution_sugar_percentage :
  ∃ x : ℚ, (∀ (y : ℚ), (y = 14) → (∃ (z : ℚ), (z = 26) → (3 / 4 * x + 1 / 4 * z = y))) → x = 10 := 
  sorry

end original_solution_sugar_percentage_l784_784515


namespace tangent_line_equation_l784_784834

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

def tangent_line_at_point (f : ℝ → ℝ) (x₀ y₀ : ℝ) : ℝ × ℝ → Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, y = k * (x - x₀) + y₀ → (x, y) ∈ (fun x => x, f x)

theorem tangent_line_equation : 
  tangent_line_at_point f (-2) 2 = {p : ℝ × ℝ | p.1 - p.2 + 4 = 0} := 
sorry

end tangent_line_equation_l784_784834


namespace find_coefficient_y_l784_784195

theorem find_coefficient_y (a b c : ℕ) (h1 : 100 * a + 10 * b + c - 7 * (a + b + c) = 100) (h2 : a + b + c ≠ 0) :
  100 * c + 10 * b + a = 43 * (a + b + c) :=
by
  sorry

end find_coefficient_y_l784_784195


namespace joe_starting_money_eq_240_l784_784348

theorem joe_starting_money_eq_240 :
  ∀ (M : ℕ), (∀ n, (n ≤ 12 -> M - n * (50 - 30) ≥ 0)) → M = 240 :=
by
  intro M h
  have h12 : M - 12 * 20 ≥ 0 := h 12 (le_refl 12)
  have h24 : 12 * 20 = 240 := by ring
  rw [←h24, ←Nat.sub_eq_zero_iff] at h12
  exact Nat.eq_zero_of_eq_zero_pred (le_antisymm h12 (Nat.zero_le (M - 240)))
  sorry

end joe_starting_money_eq_240_l784_784348


namespace intersection_M_N_l784_784625

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784625


namespace problem_correct_answers_l784_784493

def scores : List ℕ := [45, 48, 46, 52, 47, 49, 43, 51, 47, 45]

theorem problem_correct_answers :
  (List.Maximum scores - List.Minimum scores = 9) ∧
  (List.Median scores = 47) :=
  sorry

end problem_correct_answers_l784_784493


namespace daily_wage_of_c_l784_784907

theorem daily_wage_of_c 
  (a_days : ℕ) (b_days : ℕ) (c_days : ℕ) 
  (wage_ratio_a_b : ℚ) (wage_ratio_b_c : ℚ) 
  (total_earnings : ℚ) 
  (A : ℚ) (C : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  wage_ratio_a_b = 3 / 4 →
  wage_ratio_b_c = 4 / 5 →
  total_earnings = 1850 →
  A = 75 →
  C = 208.33 := 
sorry

end daily_wage_of_c_l784_784907


namespace trailing_zeros_30_factorial_l784_784723

-- Define the problem in Lean 4
theorem trailing_zeros_30_factorial : Nat.trailingZeroes (Nat.factorial 30) = 7 :=
by
  sorry

end trailing_zeros_30_factorial_l784_784723


namespace probability_prime_product_l784_784119

def is_prime_product (x y : ℕ) : Prop :=
  Nat.Prime (x * y)

theorem probability_prime_product :
  let outcomes1 := [4, 6]
  let outcomes2 := [2, 3, 5, 7, 11]
  let product_prime_count := ∑ x in outcomes1, ∑ y in outcomes2, if is_prime_product x y then 1 else 0
  let total_outcomes := outcomes1.length * outcomes2.length
  product_prime_count = 0 → (product_prime_count / total_outcomes : ℚ) = 0 :=
by
  sorry

end probability_prime_product_l784_784119


namespace tile_die_square_probability_l784_784876

theorem tile_die_square_probability :
  let tiles := Finset.range 12
  let dice := Finset.range 8
  let outcomes := tiles.product dice
  let favorable (pair : ℕ × ℕ) := nat.is_square (pair.fst * pair.snd)
  (∑ pair in outcomes, if favorable pair then 1 else 0) = 7 / 48 :=
begin
  sorry
end

end tile_die_square_probability_l784_784876


namespace intersection_M_N_is_correct_l784_784622

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784622


namespace series_convergence_l784_784343

theorem series_convergence (a : ℝ) (h : a > 0) :
  (∑' n : ℕ, (2 * n + 1) / a^(n + 1)) = if a > 1 then True else False := by
sorry

end series_convergence_l784_784343


namespace intersection_M_N_l784_784658

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784658


namespace corner_cell_revisit_l784_784162

theorem corner_cell_revisit
    (M N : ℕ)
    (hM : M = 101)
    (hN : N = 200)
    (initial_position : ℕ × ℕ)
    (h_initial : initial_position = (0, 0) ∨ initial_position = (0, 200) ∨ initial_position = (101, 0) ∨ initial_position = (101, 200)) :
    ∃ final_position : ℕ × ℕ, 
      final_position = initial_position ∧ (final_position = (0, 0) ∨ final_position = (0, 200) ∨ final_position = (101, 0) ∨ final_position = (101, 200)) :=
by
  sorry

end corner_cell_revisit_l784_784162


namespace find_c_k_l784_784423

noncomputable def a_n (n d : ℕ) := 1 + (n - 1) * d
noncomputable def b_n (n r : ℕ) := r ^ (n - 1)
noncomputable def c_n (n d r : ℕ) := a_n n d + b_n n r

theorem find_c_k (d r k : ℕ) (hd1 : c_n (k - 1) d r = 200) (hd2 : c_n (k + 1) d r = 2000) :
  c_n k d r = 423 :=
sorry

end find_c_k_l784_784423


namespace painters_work_days_l784_784311

theorem painters_work_days (rate_constant : ℕ) (workers1 workers2 : ℕ) (days1 days2 : ℝ) : 
  workers1 * days1 = rate_constant ∧ workers1 = 8 ∧ days1 = 0.75 ∧ workers2 = 5 → days2 = 1.2 :=
begin
  intros h,
  sorry
end

end painters_work_days_l784_784311


namespace intersection_M_N_l784_784670

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784670


namespace probability_no_friend_alone_l784_784230

theorem probability_no_friend_alone :
  (∃ (f : Fin 5 → Fin 4), (∀ (i : Fin 4), f⁻¹' {i} ≠ ∅) → 
  (ℕ.card {s : Fin 5 → Fin 4 | (∀ (i : Fin 4), f⁻¹' {i} ≠ ∅)}) / ℕ.pow 4 5 = 31 / 256) :=
begin
  sorry
end

end probability_no_friend_alone_l784_784230


namespace last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l784_784526

theorem last_digit_of_sum_1_to_5 : 
  (1 ^ 2012 + 2 ^ 2012 + 3 ^ 2012 + 4 ^ 2012 + 5 ^ 2012) % 10 = 9 :=
  sorry

theorem last_digit_of_sum_1_to_2012 : 
  (List.sum (List.map (λ k => k ^ 2012) (List.range 2012).tail)) % 10 = 0 :=
  sorry

end last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l784_784526


namespace intersection_M_N_eq_neg2_l784_784635

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784635


namespace M_inter_N_eq_neg2_l784_784662

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784662


namespace polynomial_roots_l784_784567

theorem polynomial_roots (x : ℂ) :
  8 * x ^ 5 - 45 * x ^ 4 + 84 * x ^ 3 - 84 * x ^ 2 + 45 * x - 8 = 0 →
  (x = (3 + Complex.sqrt 5) / 2 ∨ x = (3 - Complex.sqrt 5) / 2) :=
by
  sorry

end polynomial_roots_l784_784567


namespace old_machine_rate_correct_l784_784503

noncomputable def rate_old_machine (R : ℝ) : Prop := 
  let rate_new := 150
  let total_bolts := 550
  let time_hours := 132 / 60
  (R + rate_new) * time_hours = total_bolts

theorem old_machine_rate_correct : rate_old_machine 100 :=
by 
  unfold rate_old_machine
  have rate_new := 150
  have total_bolts := 550
  have time_hours := 132 / 60
  calc
    (100 + rate_new) * time_hours = (100 + 150) * 2.2 : by norm_num
    ... = 550 : by norm_num

end old_machine_rate_correct_l784_784503


namespace michael_has_more_flying_robots_l784_784036

theorem michael_has_more_flying_robots (tom_robots michael_robots : ℕ) (h_tom : tom_robots = 3) (h_michael : michael_robots = 12) :
  michael_robots / tom_robots = 4 :=
by
  sorry

end michael_has_more_flying_robots_l784_784036


namespace transformed_sine_eqn_l784_784440

theorem transformed_sine_eqn (ω : ℝ) (φ : ℝ) : 
(ω > 0) ∧ (|φ| < (Real.pi / 2)) ∧ 
(∀ x, sin (ω * (2 * (x - Real.pi / 3)) + φ) = sin x) ↔ (ω = 1/2) ∧ (φ = Real.pi / 6) := 
by sorry

end transformed_sine_eqn_l784_784440


namespace hall_mat_expenditure_l784_784748

theorem hall_mat_expenditure
  (length width height cost_per_sq_meter : ℕ)
  (H_length : length = 20)
  (H_width : width = 15)
  (H_height : height = 5)
  (H_cost_per_sq_meter : cost_per_sq_meter = 50) :
  (2 * (length * width) + 2 * (length * height) + 2 * (width * height)) * cost_per_sq_meter = 47500 :=
by
  sorry

end hall_mat_expenditure_l784_784748


namespace expression_decrease_l784_784697

-- Define the initial expression and the conditions given.
def original_expression (x y : ℝ) : ℝ := x * y^2

-- Define the new values after decreasing by 40%
def new_x (x : ℝ) : ℝ := 0.6 * x
def new_y (y : ℝ) : ℝ := 0.6 * y

-- Define the new expression with decreased values
def new_expression (x y : ℝ) : ℝ := new_x x * (new_y y)^2

-- Define the percentage decrease formula
def percentage_decrease (initial_value new_value : ℝ) : ℝ :=
  (initial_value - new_value) / initial_value * 100

-- State the theorem we need to prove
theorem expression_decrease (x y : ℝ) : percentage_decrease (original_expression x y) (new_expression x y) = 78.4 :=
by
  sorry

end expression_decrease_l784_784697


namespace coefficient_x3_eq_9_over_4_f_ge_27_for_all_x_l784_784839

variable (a : ℝ)

def f (x a : ℝ) : ℝ := (a / x + real.sqrt x) ^ 9

theorem coefficient_x3_eq_9_over_4 (h : ∀ x, (a / x + real.sqrt x) ^ 9 = (9.choose 8) * a * x^3) : a = 1 / 4 := sorry

theorem f_ge_27_for_all_x (a_pos : 0 < a) (h : ∀ x, (a / x + real.sqrt x) ^ 9 ≥ 27) : a ≥ 4 / 9 := sorry

end coefficient_x3_eq_9_over_4_f_ge_27_for_all_x_l784_784839


namespace tan_eleven_pi_over_three_l784_784980

theorem tan_eleven_pi_over_three : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := 
    sorry

end tan_eleven_pi_over_three_l784_784980


namespace line_pass_through_point_l784_784586

theorem line_pass_through_point (k b : ℝ) (x1 x2 : ℝ) (h1: b ≠ 0) (h2: x1^2 - k*x1 - b = 0) (h3: x2^2 - k*x2 - b = 0)
(h4: x1 + x2 = k) (h5: x1 * x2 = -b) 
(h6: (k^2 * (-b) + k * b * k + b^2 = b^2) = true) : 
  ∃ (x y : ℝ), (y = k * x + 1) ∧ (x, y) = (0, 1) :=
by
  sorry

end line_pass_through_point_l784_784586


namespace max_sum_of_labels_l784_784040

theorem max_sum_of_labels :
  (∃ (r : Fin 8 → Fin 8),
      ∀ i j, i ≠ j → r i ≠ r j ∧ (r i : ℕ) + 1 ≤ 8 ∧
      ∑ j, 1 / (2 * ↑(r j) + j + 1 - 1) = 0.64) :=
sorry

end max_sum_of_labels_l784_784040


namespace arithmetic_expression_l784_784536

theorem arithmetic_expression :
  let a := 2^2
  let b := abs (-3)
  let c := real.sqrt 25
  a + b - c = 2 :=
by
  -- introduce the constants
  let a := 2 ^ 2
  let b := abs (-3)
  let c := real.sqrt 25
  -- state the goal
  show a + b - c = 2
  -- use sorry to skip the proof
  sorry

end arithmetic_expression_l784_784536


namespace sum_s_of_r_range_l784_784782

def r_domain := {x : ℤ | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1}
def r_range := {-1, 0, 1, 3}
def s_domain := {-1, 0, 1, 3}
def s (x : ℤ) : ℤ := 2 * x + 1

theorem sum_s_of_r_range :
  (∀ x, x ∈ r_domain → s x ∈ s_domain) →
  (∀ x, x ∈ r_range ↔ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) →
  ∑ x in r_range, s x = 10 :=
by
  intros h1 h2
  sorry

end sum_s_of_r_range_l784_784782


namespace find_f_neg2_l784_784275

variables {f : ℝ → ℝ}

def graph_symmetric (f g : ℝ → ℝ) (line : ℝ → ℝ) : Prop :=
∀ x y, g x = y ↔ f y = -x

theorem find_f_neg2 (h : graph_symmetric f (λ x, 2^(x + 2)) (λ x, -x)) : f (-2) = 1 :=
sorry

end find_f_neg2_l784_784275


namespace intersection_M_N_is_correct_l784_784624

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784624


namespace arithmetic_sequence_sum_l784_784768

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n : ℕ, S n = 2 * n^2 - 3 * n) →
  (∀ n : ℕ, a n = S (n+1) - S n) →
  (∀ n : ℕ, a (n+1) - a n = 4) :=
by
  intros hS ha
  skip_proofs;
  sorry

end arithmetic_sequence_sum_l784_784768


namespace integral_fx_l784_784020

noncomputable def f : ℝ → ℝ := 
λ x, if (0 ≤ x ∧ x < 1) then x ^ 2 else if (1 < x ∧ x ≤ 2) then 2 - x else 0

theorem integral_fx : ∫ x in 0..2, f x = 5 / 6 :=
by {
  let g : ℝ → ℝ := λ x, if 0 ≤ x ∧ x < 1 then x ^ 2 else if 1 < x ∧ x ≤ 2 then 2 - x else 0,
  have h_g_eq_f : ∀ x, g x = f x := by simp [f, g],
  simp_rw [← h_g_eq_f],
  -- Then integrate g instead of f
  convert_to (∫ x in 0..1, x ^ 2 + ∫ x in 1..2, 2 - x = _) using 1,
  { apply interval_integrable.integral_add (interval_integrable_iff.mpr _),
    simp [g, interval_integrable, continuous_on] with integrable_simp }
  -- Then calculate the integral accordingly
  apply_dvision_partition,
  interval_integral, 
  sorry -- completes the proof accordingly
}

end integral_fx_l784_784020


namespace arithmetic_sequence_sum_square_l784_784134

theorem arithmetic_sequence_sum_square (a d : ℕ) :
  (∀ n : ℕ, ∃ k : ℕ, n * (a + (n-1) * d / 2) = k * k) ↔ (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) := 
by
  sorry

end arithmetic_sequence_sum_square_l784_784134


namespace intersection_M_N_l784_784608

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784608


namespace arithmetic_sequence_common_difference_l784_784795

variable {α : Type*} [linear_ordered_field α]

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
∑ k in finset.range n, a k

theorem arithmetic_sequence_common_difference 
  {a : ℕ → α} {S : ℕ → α} (hS2 : S 2 = 4) (hS4 : S 4 = 20)
  (hSn : ∀ n, S n = sum_first_n_terms a n) :
  ∃ d : α, d = 3 :=
sorry

end arithmetic_sequence_common_difference_l784_784795


namespace dog_roaming_area_l784_784509

theorem dog_roaming_area
  (shed_side_length : ℝ)
  (interior_angle : ℝ)
  (rope_length : ℝ)
  (midpoint : ℝ)
  (accessible_area : ℝ) :
  shed_side_length = 20 ∧
  interior_angle = 108 ∧
  rope_length = 10 ∧
  midpoint = shed_side_length / 2 →
  accessible_area = 50 * π :=
begin
  sorry
end

end dog_roaming_area_l784_784509


namespace b_arithmetic_sequence_sequence_sum_l784_784589

def a (n : ℕ) : ℝ := if n = 1 then 1/8 else a (n-1) / (1 - 2 * (a (n-1)))

def b (n : ℕ) : ℝ := 1 / a n

-- Problem 1: Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (h : n ≥ 1) :
  ∃ (d : ℝ), b (n + 1) = b n + d :=
sorry

def S (n : ℕ) : ℝ :=
if 1 ≤ n ∧ n ≤ 5 then -n^2 + 9 * n
else if n ≥ 6 then n^2 - 9 * n + 40
else 0

-- Problem 2: Prove the given sum for S_n
theorem sequence_sum (n : ℕ) (h : n ≥ 1) :
  S n = ∑ i in finset.range n, |b (i + 1)| :=
sorry

end b_arithmetic_sequence_sequence_sum_l784_784589


namespace time_per_window_l784_784158

-- Definitions of the given conditions
def total_windows : ℕ := 10
def installed_windows : ℕ := 6
def remaining_windows := total_windows - installed_windows
def total_hours : ℕ := 20
def hours_per_window := total_hours / remaining_windows

-- The theorem we need to prove
theorem time_per_window : hours_per_window = 5 := by
  -- This is where the proof would go
  sorry

end time_per_window_l784_784158


namespace least_pos_int_satisfies_conditions_l784_784894

theorem least_pos_int_satisfies_conditions :
  ∃ x : ℕ, x > 0 ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  x = 419 :=
by
  sorry

end least_pos_int_satisfies_conditions_l784_784894


namespace gcd_of_given_lcm_and_ratio_l784_784315

theorem gcd_of_given_lcm_and_ratio (C D : ℕ) (h1 : Nat.lcm C D = 200) (h2 : C * 5 = D * 2) : Nat.gcd C D = 5 :=
sorry

end gcd_of_given_lcm_and_ratio_l784_784315


namespace largest_positive_multiple_of_15_less_than_500_l784_784454

theorem largest_positive_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n < 500 ∧ 15 ∣ n ∧ ∀ m : ℕ, m < 500 ∧ 15 ∣ m → m ≤ n := 
begin
  use 495,
  split,
  { exact lt_of_lt_of_le (nat.lt_add_one_self 499) (le_of_eq rfl) },
  split,
  { unfold dvd, use 33, exact rfl },
  { intros m hm, by_contradiction h,
    have : m > 495, from lt_of_not_ge h,
    have : m ≤ 500 - 15,
    { exact le_of_lt (nat.succ_lt_succ_iff.mp (lt_of_le_of_lt (nat.le_sub_add m 500 15) (nat.lt_add_one_self 33))) },
    exact h (le_of_lt (this.trans_le (nat.sub_le_self 500 15))) }
end

end largest_positive_multiple_of_15_less_than_500_l784_784454


namespace most_suitable_athlete_l784_784104

def mean_jump_length : Type := A | B | C | D

def variance : Type := A | B | C | D

-- Definitions for the mean jump lengths in cm
def mean : mean_jump_length → ℝ
| A := 380
| B := 360
| C := 380
| D := 350

-- Definitions for the variances in cm^2
def var : variance → ℝ
| A := 12.5
| B := 13.5
| C := 2.4
| D := 2.7

-- The theorem stating that athlete C is the most suitable for the finals
theorem most_suitable_athlete :
  (mean C = 380) ∧ (var C = 2.4) :=
sorry

end most_suitable_athlete_l784_784104


namespace find_sphere_radius_l784_784866

noncomputable def sphere_radius_proof (r1 r2 r3 : ℝ) (angle1 angle2 angle3 : ℝ) (R : ℝ) : Prop :=
  r1 = 72 ∧ r2 = 28 ∧ r3 = 28 ∧ angle1 = -π/3 ∧ angle2 = 2*π/3 ∧ angle3 = 2*π/3 ∧ R = (sqrt 3 + 1) / 2

theorem find_sphere_radius (r1 r2 r3 : ℝ) (angle1 angle2 angle3 : ℝ) (R : ℝ) :
  sphere_radius_proof r1 r2 r3 angle1 angle2 angle3 R :=
by
  sorry

end find_sphere_radius_l784_784866


namespace number_of_birds_seen_l784_784071

theorem number_of_birds_seen (dozens_seen : ℕ) (birds_per_dozen : ℕ) (h₀ : dozens_seen = 8) (h₁ : birds_per_dozen = 12) : dozens_seen * birds_per_dozen = 96 :=
by sorry

end number_of_birds_seen_l784_784071


namespace integer_cubes_between_neg100_and_100_l784_784545

theorem integer_cubes_between_neg100_and_100 : 
  ∃ (count : ℕ), count = (set.finite.count (set.filter (λ n : ℤ, -100 < n^3 ∧ n^3 < 100) (set.univ : set ℤ))) ∧ count = 10 :=
by
  sorry

end integer_cubes_between_neg100_and_100_l784_784545


namespace salon_buys_hairspray_l784_784512

theorem salon_buys_hairspray
  (same_customers_each_day : ∀ (d1 d2 : ℕ), d1 ≠ d2 → 14 = 14)  -- The salon has the same number of customers each day (14).
  (hairspray_per_customer_styling : ℕ := 1)  -- Each customer needs 1 can of hairspray for styling.
  (hairspray_per_customer_home : ℕ := 1)  -- Each customer is given 1 can of hairspray to take home.
  (extra_hairspray : ℕ := 5)  -- The salon buys an extra 5 cans of hairspray each day.
  (customers_per_day : ℕ := 14)  -- The salon has 14 customers each day.
  : 14 * 2 + 5 = 33 := 
begin
  -- Calculation
  sorry,
end

end salon_buys_hairspray_l784_784512


namespace remainder_coefficient_l784_784421

def P (x : ℝ) : ℝ := x^100 - x^99 + x^98 - x^97 + x^96 - x^95 + x^94 - x^93 + x^92 - x^91 + x^90 - x^89 + x^88 - x^87 + x^86 - x^85 + x^84 - x^83 + x^82 - x^81 + x^80 - x^79 + x^78 - x^77 + x^76 - x^75 + x^74 - x^73 + x^72 - x^71 + x^70 - x^69 + x^68 - x^67 + x^66 - x^65 + x^64 - x^63 + x^62 - x^61 + x^60 - x^59 + x^58 - x^57 + x^56 - x^55 + x^54 - x^53 + x^52 - x^51 + x^50 - x^49 + x^48 - x^47 + x^46 - x^45 + x^44 - x^43 + x^42 - x^41 + x^40 - x^39 + x^38 - x^37 + x^36 - x^35 + x^34 - x^33 + x^32 - x^31 + x^30 - x^29 + x^28 - x^27 + x^26 - x^25 + x^24 - x^23 + x^22 - x^21 + x^20 - x^19 + x^18 - x^17 + x^16 - x^15 + x^14 - x^13 + x^12 - x^11 + x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_coefficient :
  let a := -50
  let b := 51 in
  2 * a + b = -49 :=
by
  let a := -50
  let b := 51
  show 2 * a + b = -49
  sorry

end remainder_coefficient_l784_784421


namespace mean_temperature_is_86_3_l784_784415

open Real

def temperatures := [82, 80, 83, 88, 84, 90, 92, 85, 89, 90]

theorem mean_temperature_is_86_3
    (temps : List ℕ := temperatures)  -- given temperatures
    (n : ℕ := temps.length)  -- number of days, which is the length of the list
    : Real :=
  (temps.sum / n : Real) = 86.3 :=
by
  sorry

end mean_temperature_is_86_3_l784_784415


namespace olya_wins_game_l784_784917

-- Definitions based on conditions
def max_dispute (connections : Finset (Fin 2009 × Fin 2009)) : Prop :=
  ∀ (i j : Fin 2009), (i, j) ∈ connections → (j, i) ∈ connections

-- The main theorem statement, using conditions and answer from the problem
theorem olya_wins_game :
  ∃ (connections : Finset (Fin 2009 × Fin 2009)), 
  max_dispute connections → 
  (∀ (start : Fin 2009), ∃ (winning_strategy : Prop), winning_strategy) :=
begin
  sorry
end

end olya_wins_game_l784_784917


namespace sum_series_equals_a_over_b_fact_minus_c_l784_784886

theorem sum_series_equals_a_over_b_fact_minus_c :
  (\sum k in Finset.range 50, (-1)^(k+1) * (k+1)^3 + (k+1)^2 + (k+1) + 1) / (k+1)! = 2551 / 49! - 1 → 
  2551 + 49 + 1 = 2601 := sorry

end sum_series_equals_a_over_b_fact_minus_c_l784_784886


namespace g_symmetric_about_075_l784_784095

-- Definitions
def floor (x : ℝ) : ℤ := int.floor x
def g (x : ℝ) : ℝ := |(floor (x + 0.5) : ℝ)| - |(floor (1.5 - x) : ℝ)|

-- Theorem statement
theorem g_symmetric_about_075 :
  ∀ x : ℝ, g(x) = g(1.5 - x) :=
sorry

end g_symmetric_about_075_l784_784095


namespace ivan_can_achieve_40_percent_solution_l784_784346

theorem ivan_can_achieve_40_percent_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a ≠ 2/3) : 
  ∃ n : ℕ, exists_concentration_40 a n :=
sorry

end ivan_can_achieve_40_percent_solution_l784_784346


namespace Catriona_total_fish_l784_784987

theorem Catriona_total_fish:
  ∃ (goldfish angelfish guppies : ℕ),
  goldfish = 8 ∧
  angelfish = goldfish + 4 ∧
  guppies = 2 * angelfish ∧
  goldfish + angelfish + guppies = 44 :=
by
  -- Define the number of goldfish
  let goldfish := 8

  -- Define the number of angelfish, which is 4 more than goldfish
  let angelfish := goldfish + 4

  -- Define the number of guppies, which is twice the number of angelfish
  let guppies := 2 * angelfish

  -- Prove the total number of fish is 44
  have total_fish : goldfish + angelfish + guppies = 44 := by
    rw [←nat.add_assoc, nat.add_comm 12 8, nat.add_assoc, nat.add_comm 24 12, ←nat.add_assoc]

  use [goldfish, angelfish, guppies]
  exact ⟨rfl, rfl, rfl, total_fish⟩

end Catriona_total_fish_l784_784987


namespace conjugate_of_complex_num_l784_784187

def complex_num := (2017 - Complex.i) / (1 - Complex.i)
def result_conjugate := Complex.conj complex_num

theorem conjugate_of_complex_num : result_conjugate = 1009 - 1008 * Complex.i := 
by sorry

end conjugate_of_complex_num_l784_784187


namespace discount_is_15_point_5_percent_l784_784165

noncomputable def wholesale_cost (W : ℝ) := W
noncomputable def retail_price (W : ℝ) := 1.5384615384615385 * W
noncomputable def selling_price (W : ℝ) := 1.3 * W
noncomputable def discount_percentage (W : ℝ) := 
  let D := retail_price W - selling_price W
  (D / retail_price W) * 100

theorem discount_is_15_point_5_percent (W : ℝ) (hW : W > 0) : 
  discount_percentage W = 15.5 := 
by 
  sorry

end discount_is_15_point_5_percent_l784_784165


namespace greatest_value_of_y_l784_784395

theorem greatest_value_of_y (x y : ℤ) (h : x * y + 5 * x + 4 * y = -5) : y ≤ 10 :=
sorry

end greatest_value_of_y_l784_784395


namespace range_of_f_l784_784566

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan (Real.sqrt ((0.5:ℝ) ^ (- Real.log (0.5) (Real.sin x / (Real.sin x + 7)))))

theorem range_of_f : set.range f = set.Ioc 0 (Real.pi / 6) :=
sorry

end range_of_f_l784_784566


namespace floor_abs_sum_eq_501_l784_784822

open Int

theorem floor_abs_sum_eq_501 (x : Fin 1004 → ℝ) (h : ∀ i, x i + (i : ℝ) + 1 = (Finset.univ.sum x) + 1005) : 
  Int.floor (abs (Finset.univ.sum x)) = 501 :=
by
  -- Proof steps will go here
  sorry

end floor_abs_sum_eq_501_l784_784822


namespace M_inter_N_eq_neg2_l784_784664

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784664


namespace hours_increase_percent_l784_784946

def original_hourly_wage : ℝ := sorry
def original_hours_per_week : ℝ := sorry
def total_sales : ℝ := sorry
def commission_percentage : ℝ := sorry

noncomputable def new_hourly_wage (W : ℝ) : ℝ :=
  0.80 * W

noncomputable def income_with_new_hours (H_new W : ℝ) : ℝ :=
  new_hourly_wage(W) * H_new

def commission_earned (C S : ℝ) : ℝ :=
  C * S

def original_income (W H : ℝ) : ℝ :=
  W * H

theorem hours_increase_percent (W H S C : ℝ) :
  (H_new : ℝ) →
  (original_income W H + commission_earned C S = income_with_new_hours H_new W + commission_earned C S) →
  (H_new = 1.25 * H) →
  ∃ E, E = 25 := sorry

end hours_increase_percent_l784_784946


namespace money_saved_l784_784242

noncomputable def total_savings :=
  let fox_price := 15
  let pony_price := 18
  let num_fox_pairs := 3
  let num_pony_pairs := 2
  let total_discount_rate := 0.22
  let pony_discount_rate := 0.10999999999999996
  let fox_discount_rate := total_discount_rate - pony_discount_rate
  let fox_savings := fox_price * fox_discount_rate * num_fox_pairs
  let pony_savings := pony_price * pony_discount_rate * num_pony_pairs
  fox_savings + pony_savings

theorem money_saved :
  total_savings = 8.91 :=
by
  -- We assume the savings calculations are correct as per the problem statement
  sorry

end money_saved_l784_784242


namespace find_a_minus_inverse_l784_784270

-- Definition for the given condition
def condition (a : ℝ) : Prop := a + a⁻¹ = 6

-- Definition for the target value to be proven
def target_value (x : ℝ) : Prop := x = 4 * Real.sqrt 2 ∨ x = -4 * Real.sqrt 2

-- Theorem statement to be proved
theorem find_a_minus_inverse (a : ℝ) (ha : condition a) : target_value (a - a⁻¹) :=
by
  sorry

end find_a_minus_inverse_l784_784270


namespace part1_part2_l784_784540

section

variable {x m : ℝ}

def M : set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N1 (m : ℝ) : set ℝ := {x | m - 6 ≤ x ∧ x ≤ 2m - 1}
def N2 (m : ℝ) : set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2m - 1}

theorem part1 (m : ℝ) : M ⊆ N1 m → 2 ≤ m ∧ m ≤ 3 :=
sorry

theorem part2 (m : ℝ) : N2 m ⊆ M → m ≤ 3 :=
sorry

end

end part1_part2_l784_784540


namespace factorization_PQ_l784_784082

theorem factorization_PQ (P Q : ℝ) (h : (λ x : ℝ, (x^2 + 2 * real.sqrt 2 * x + 5) * (x^2 + c * x + d) = x^4 + P * x^2 + Q)) : 
  P + Q = 27 := 
sorry

end factorization_PQ_l784_784082


namespace min_extractions_to_reverse_l784_784865

theorem min_extractions_to_reverse (n : ℕ) : 
  (minimum_extractions n = (n / 2) + 1) :=
sorry

end min_extractions_to_reverse_l784_784865


namespace calculate_a_minus_b_l784_784182

variable {α β : Type}
variable (f : α → β) (a b : α)
variable [invf : Function.Bijective f]

theorem calculate_a_minus_b
  (h₁ : f a = 3)
  (h₂ : f b = 1)
  (h₃ : Function.Bijective f) :
  a - b = -2 :=
by {
  sorry
}

end calculate_a_minus_b_l784_784182


namespace intersection_M_N_l784_784627

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784627


namespace equal_probabilities_l784_784497

-- Define what it means to be a fair six-sided die rolled twice and the probabilities
noncomputable def six_sided_die := {1, 2, 3, 4, 5, 6}

def sum_of_two_rolls (a b : ℕ) : ℕ := a + b

def remainder_mod_5 (n : ℕ) : ℕ := n % 5

def is_probability_of_remainder (s : set (ℕ × ℕ)) (r : ℕ) : Prop :=
  (finset.filter (λ p, remainder_mod_5 (sum_of_two_rolls p.1 p.2) = r) (finset.product six_sided_die six_sided_die)).card = 
  s.card / 36

-- Define the sets representing the possible outcomes for sums yielding remainders 0 and 4
def outcomes_with_remainder_0 : finset (ℕ × ℕ) := 
  finset.filter (λ p, remainder_mod_5 (sum_of_two_rolls p.1 p.2) = 0) (finset.product six_sided_die six_sided_die)

def outcomes_with_remainder_4 : finset (ℕ × ℕ) := 
  finset.filter (λ p, remainder_mod_5 (sum_of_two_rolls p.1 p.2) = 4) (finset.product six_sided_die six_sided_die)

theorem equal_probabilities : 
  is_probability_of_remainder outcomes_with_remainder_0 0 = 
  is_probability_of_remainder outcomes_with_remainder_4 4 := 
by sorry

end equal_probabilities_l784_784497


namespace trader_sold_meters_l784_784516

variable (x : ℕ) (SP P CP : ℕ)

theorem trader_sold_meters (h_SP : SP = 660) (h_P : P = 5) (h_CP : CP = 5) : x = 66 :=
  by
  sorry

end trader_sold_meters_l784_784516


namespace a_cards_is_1_8_9_l784_784433

def cards_of_A (a_cards b_cards c_cards d_cards : list ℕ) : Prop :=
  a_cards = [1, 8, 9] ∧ 
  b_cards.all (λ n, prime n) ∧
  (hC : ∃ p, prime p ∧ c_cards.all (λ n, n ≠ p ∧ ∃ k, k ≠ 1 ∧ k ≠ n ∧ p ∣ k ∧ k ∣ n)) ∧
  (d_cards.all (λ n, ∃ m, (m ∈ a_cards ∪ b_cards ∪ c_cards) ∧ n ≠ m)) ∧
  a_cards ∪ b_cards ∪ c_cards ∪ d_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

theorem a_cards_is_1_8_9 :
  ∃ a_cards b_cards c_cards d_cards : list ℕ,
  a_cards = [1, 8, 9] ∧ 
  b_cards.all (λ n, prime n) ∧
  (hC : ∃ p, prime p ∧ c_cards.all (λ n, n ≠ p ∧ ∃ k, k ≠ 1 ∧ k ≠ n ∧ p ∣ k ∧ k ∣ n)) ∧
  (d_cards.all (λ n, ∃ m, (m ∈ a_cards ∪ b_cards ∪ c_cards) ∧ n ≠ m)) ∧
  a_cards ∪ b_cards ∪ c_cards ∪ d_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] :=
by
  sorry

end a_cards_is_1_8_9_l784_784433


namespace dist_points_l784_784561

-- Define the points p1 and p2
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (4, 1)

-- Define the distance formula between the points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The theorem stating the distance between these points is 5
theorem dist_points : dist p1 p2 = 5 := by
  sorry

end dist_points_l784_784561


namespace average_first_19_natural_numbers_l784_784141

theorem average_first_19_natural_numbers : 
  (1 + 19) / 2 = 10 := 
by 
  sorry

end average_first_19_natural_numbers_l784_784141


namespace total_profit_l784_784908

variable (a b c p : ℝ)

-- Given conditions
def conditions :=
  a + b + c = 50000 ∧
  a = b + 4000 ∧
  b = c + 5000 ∧
  (15120 / a) = (p / 50000)

-- The theorem to prove the total profit.
theorem total_profit (h : conditions) : p = 36000 :=
  sorry

end total_profit_l784_784908


namespace roll_probability_l784_784145

theorem roll_probability (n : ℕ) (hn1 : 1 ≤ n) (hn2 : n ≤ 100) :
  (100^3: ℚ) ≠ 0 → ∑ i in finset.range 101, ∑ j in finset.range (i+1), j-1 = 3333 / 20000 * (100^3 : ℚ) :=
by sorry

end roll_probability_l784_784145


namespace stations_visited_l784_784537

theorem stations_visited
  (total_nails : ℕ)
  (nails_per_station : ℕ)
  (h1 : total_nails = 140)
  (h2 : nails_per_station = 7) :
  (total_nails / nails_per_station) = 20 := by
  rw [h1, h2]
  norm_num
  sorry

end stations_visited_l784_784537


namespace bee_distance_l784_784488

noncomputable def omega : ℂ := Complex.exp (Real.pi * Complex.I / 4)

def bee_position : ℂ :=
  1 + omega + 2 * omega^2 + 3 * omega^3 + 4 * omega^4 + 
  5 * omega^5 + 6 * omega^6 + 7 * omega^7 + 8 * omega^8 + 
  9 * omega^9 + 10 * omega^10

def final_distance_from_origin : ℝ := 
  Complex.abs (bee_position)

theorem bee_distance : 
  final_distance_from_origin = Complex.abs ((10 * omega^2 - 1 - omega) / (omega - 1)) / Real.sqrt 2 :=
sorry

end bee_distance_l784_784488


namespace x_intercept_of_perpendicular_line_is_16_over_3_l784_784449

theorem x_intercept_of_perpendicular_line_is_16_over_3 :
  (∃ x : ℚ, (∃ y : ℚ, (4 * x - 3 * y = 12))
    ∧ (∃ x y : ℚ, (y = - (3 / 4) * x + 4 ∧ y = 0) ∧ x = 16 / 3)) :=
by {
  sorry
}

end x_intercept_of_perpendicular_line_is_16_over_3_l784_784449


namespace find_n_l784_784228

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 :=
by
  sorry

end find_n_l784_784228


namespace domain_of_function_l784_784203

def domain_condition_1 (x : ℝ) : Prop := 1 - x > 0
def domain_condition_2 (x : ℝ) : Prop := x + 3 ≥ 0

def in_domain (x : ℝ) : Prop := domain_condition_1 x ∧ domain_condition_2 x

theorem domain_of_function : ∀ x : ℝ, in_domain x ↔ (-3 : ℝ) ≤ x ∧ x < 1 := 
by sorry

end domain_of_function_l784_784203


namespace existence_of_ABC_existence_of_ABC_neg_1_4_existence_of_ABC_2_5_l784_784592

-- Definitions for geometry
variables {Point : Type} [EuclideanGeometry Point]

-- Given data: triangle XYZ and rational lambda
variable (XYZ : Triangle Point) (X Y Z : Point)
variable (lambda : ℚ)

-- Require to construct a triangle ABC satisfying the given ratio conditions
axiom constructABC (ABC : Triangle Point) (A B C : Point) :
  (segment_ratio X B C = lambda) ∧ 
  (segment_ratio Y C A = lambda) ∧ 
  (segment_ratio Z A B = lambda)

-- Proof Problem: Existence of ABC for specific lambdas
theorem existence_of_ABC (XYZ : Triangle Point) (X Y Z : Point) :
  ∃ (ABC : Triangle Point) (A B C : Point), 
  constructABC XYZ X Y Z ABC A B C :=
sorry

-- Special cases for specific lambda values:
-- Case a: lambda = -1/4
theorem existence_of_ABC_neg_1_4 (XYZ : Triangle Point) (X Y Z : Point) :
  ∃ (ABC : Triangle Point) (A B C : Point), 
  constructABC XYZ X Y Z ABC A B C :=
sorry

-- Case b: lambda = 2/5
theorem existence_of_ABC_2_5 (XYZ : Triangle Point) (X Y Z : Point) :
  ∃ (ABC : Triangle Point) (A B C : Point), 
  constructABC XYZ X Y Z ABC A B C :=
sorry

end existence_of_ABC_existence_of_ABC_neg_1_4_existence_of_ABC_2_5_l784_784592


namespace proof_problem_l784_784306

variable {α : Type*} [LinearOrder α] [AddGroup α] {f : α → α}

-- Define f to be an odd function
def odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

-- Conditions
variable (h1 : odd_function f)
variable (h2 : f 3 < f 1)

-- Statement to prove
theorem proof_problem : f (-1) < f (-3) :=
by sorry

end proof_problem_l784_784306


namespace number_of_k_values_l784_784237

theorem number_of_k_values : 
  ∃ (k : ℕ) (a b c : ℕ) 
    (ha : 0 ≤ a ∧ a ≤ 72) 
    (hb : 0 ≤ b ∧ b ≤ 24) 
    (hc : 0 ≤ c), 
    [2^24 * 3^18, 2^a * 3^b * 5^c] = 2^72 * 3^24 ∧ 
    ha ∧ hb ∧ hc ∧
    (∃ (n : ℕ), n = 73 * 25 * 1 ∧ n = 1825) :=
begin
  sorry
end

end number_of_k_values_l784_784237


namespace a_n_plus_1_le_a_n_l784_784595

noncomputable def P_n (a : ℕ → ℝ) (n : ℕ) : polynomial ℝ :=
  polynomial.sum (range n) (λ i, polynomial.C (a (i+1)) * polynomial.X^(2*(n-i)))

noncomputable def a : ℕ → ℝ := sorry -- Given as a definition for Lean's context

axiom exists_min_root (n : ℕ) : ((P_n a n).roots.filter (λ r, r.is_real)).nonempty

def a_n_plus_1 (n : ℕ) : ℝ :=
  (exists_min_root n).some

theorem a_n_plus_1_le_a_n {n : ℕ} (hn : n ≥ 2021) :
  a_n_plus_1 (n+1) ≤ a n :=
sorry

end a_n_plus_1_le_a_n_l784_784595


namespace ratio_of_areas_l784_784441

noncomputable def triangle {α : Type} (a b c : α) : Type := sorry
noncomputable def area {α : Type} [field α] (t : triangle α α α) : α := sorry

def abc := triangle 5 12 13
def def := triangle 8 15 17

theorem ratio_of_areas : area abc / area def = 1 / 2 := by
  sorry

end ratio_of_areas_l784_784441


namespace circle_II_area_correct_l784_784992

noncomputable def CircleArea (r : ℝ) : ℝ := π * r^2

-- Given conditions
def circle_I_area : ℝ := 9
def circles_same_center : Prop := true  -- Sharing the same center
def circle_I_tangential_inside_circle_II : Prop := true -- Tangential from inside

-- Question: Calculate the area of Circle II (Check that Circle II area == 16)
theorem circle_II_area_correct (r R : ℝ) 
    (h1 : CircleArea r = circle_I_area)
    (h2 : R = r + 1 / Real.sqrt π) :
    CircleArea R = 16 :=
sorry

end circle_II_area_correct_l784_784992


namespace initial_interval_bisection_method_l784_784133

noncomputable def f (x : ℝ) : ℝ := 2^x - 3

theorem initial_interval_bisection_method :
  (f 1 < 0) ∧ (f 2 > 0) → ∃ (a b : ℝ), (a = 1) ∧ (b = 2) ∧ (∀ x ∈ set.Icc a b, continuous_on f (set.Icc a b)) ∧ ∀ x y, x < y → (f x < f y) :=
by
  sorry

end initial_interval_bisection_method_l784_784133


namespace range_of_b_l784_784920

theorem range_of_b (a b : ℝ) (h₁ : a ≤ -1) (h₂ : a * 2 * b - b - 3 * a ≥ 0) : b ≤ 1 := by
  sorry

end range_of_b_l784_784920


namespace intersection_M_N_l784_784671

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784671


namespace sequence_k_value_l784_784337

theorem sequence_k_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ m n : ℕ, a (m + n) = a m * a n)
  (hk1 : ∀ k : ℕ, a (k + 1) = 1024) :
  ∃ k : ℕ, k = 9 :=
by {
  sorry
}

end sequence_k_value_l784_784337


namespace office_needs_24_pencils_l784_784508

noncomputable def number_of_pencils (total_cost : ℝ) (cost_per_pencil : ℝ) (cost_per_folder : ℝ) (number_of_folders : ℕ) : ℝ :=
  (total_cost - (number_of_folders * cost_per_folder)) / cost_per_pencil

theorem office_needs_24_pencils :
  number_of_pencils 30 0.5 0.9 20 = 24 :=
by
  sorry

end office_needs_24_pencils_l784_784508


namespace no_valid_polygons_pairs_l784_784880

-- Define the interior angle formula for a polygon with n sides
def interior_angle (n : ℕ) : Real := 180 - (360 / n)

-- Define the condition that the ratio of the interior angles is 7:5
def angle_ratio_condition (r k : ℕ) : Prop :=
  (180 * r - 360) / (180 * k - 360) = 7 / 5

-- Formalize the problem to prove that there are no such pairs (r, k) that satisfy the condition
theorem no_valid_polygons_pairs : ∀ (r k : ℕ), r > 2 ∧ k > 2 → ¬ angle_ratio_condition r k :=
by
  intros r k hk
  sorry

end no_valid_polygons_pairs_l784_784880


namespace sum_of_three_numbers_l784_784459

theorem sum_of_three_numbers (a b c : ℝ) :
  a + b = 35 → b + c = 47 → c + a = 58 → a + b + c = 70 :=
by
  intros h1 h2 h3
  sorry

end sum_of_three_numbers_l784_784459


namespace circle_tangent_through_K_and_T_l784_784587

noncomputable def construct_circle (K T : Point) (C : Circle) (pT : T ∈ C) : Circle :=
sorry

theorem circle_tangent_through_K_and_T (K T : Point) (C : Circle) (pT : T ∈ C) :
  ∃ (circ : Circle), K ∈ circ ∧ T ∈ circ ∧ tangent circ C T :=
sorry

end circle_tangent_through_K_and_T_l784_784587


namespace workshop_total_workers_l784_784402

noncomputable def total_workers (total_avg_salary technicians_avg_salary non_technicians_avg_salary : ℕ) (technicians_count : ℕ) : ℕ :=
  sorry

theorem workshop_total_workers (avg_salary : ℕ) (tech_avg_salary : ℕ) (non_tech_avg_salary : ℕ) (tech_count : ℕ) :
  total_workers avg_salary tech_avg_salary non_tech_avg_salary tech_count = 49 :=
by {
  -- Given conditions
  let avg_salary := 8000,
  let tech_avg_salary := 20000,
  let non_tech_avg_salary := 6000,
  let tech_count := 7,
  -- Assertions based on these conditions would follow
  sorry
}

end workshop_total_workers_l784_784402


namespace polygon_sides_l784_784948

-- Definition of constants based on given conditions
def interior_angle : ℝ := 140
def sum_exterior_angles : ℝ := 360

-- Proof statement
theorem polygon_sides (h1 : ∀ (n : ℕ), n ≥ 3 → sum_exterior_angles / (180 - interior_angle) = n) : 
  180 - interior_angle = 40 → 360 / 40 = 9 :=
by
  intro h2
  rw h2
  norm_num
  sorry

end polygon_sides_l784_784948


namespace intersection_M_N_is_correct_l784_784619

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784619


namespace correct_ranking_proof_l784_784747

-- Define each student's position assertion
def D1 (positions : List String) := positions.nth 1 = some "Sanxi"
def D2 (positions : List String) := positions.nth 2 = some "Jianye"
def W1 (positions : List String) := positions.nth 1 = some "Meihong"
def W2 (positions : List String) := positions.nth 3 = some "Deng Qing"
def S1 (positions : List String) := positions.nth 0 = some "Deng Qing"
def S2 (positions : List String) := positions.nth 4 = some "Wu Lin"
def J1 (positions : List String) := positions.nth 2 = some "Meihong"
def J2 (positions : List String) := positions.nth 3 = some "Wu Lin"
def M1 (positions : List String) := positions.nth 1 = some "Jianye"
def M2 (positions : List String) := positions.nth 4 = some "Sanxi"

-- Teacher Zhang's condition: each student's two statements must have one true and one false
def teacherZhangCondition (positions : List String) : Prop :=
  (D1 positions ∧ ¬D2 positions) ∨ (¬D1 positions ∧ D2 positions) ∧
  (W1 positions ∧ ¬W2 positions) ∨ (¬W1 positions ∧ W2 positions) ∧
  (S1 positions ∧ ¬S2 positions) ∨ (¬S1 positions ∧ S2 positions) ∧
  (J1 positions ∧ ¬J2 positions) ∨ (¬J1 positions ∧ J2 positions) ∧
  (M1 positions ∧ ¬M2 positions) ∨ (¬M1 positions ∧ M2 positions)

-- Correct ranking order
def correctRanking : List String := ["Deng Qing", "Meihong", "Jianye", "Wu Lin", "Sanxi"]

theorem correct_ranking_proof : ∃ (positions : List String), (positions = correctRanking) ∧ teacherZhangCondition positions :=
by
  sorry

end correct_ranking_proof_l784_784747


namespace walking_speed_l784_784501

theorem walking_speed (d : ℝ) (w_speed r_speed : ℝ) (w_time r_time : ℝ)
    (h1 : d = r_speed * r_time)
    (h2 : r_speed = 24)
    (h3 : r_time = 1)
    (h4 : w_time = 3) :
    w_speed = 8 :=
by
  sorry

end walking_speed_l784_784501


namespace annual_percentage_increase_approx_four_percent_l784_784851

theorem annual_percentage_increase_approx_four_percent :
  ∀ (x : ℝ), (x < 0.1) →
  (40 * (Real.ln (1 + x)) = 1.61) →
  (100 * (1 + x)^40 = 500) →
  x ≈ 0.04 :=
by
  intro x h1 h2 h3
  sorry

end annual_percentage_increase_approx_four_percent_l784_784851


namespace ratio_volume_sphere_to_hemisphere_l784_784101

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (1/2) * volume_sphere r

theorem ratio_volume_sphere_to_hemisphere (p : ℝ) (hp : 0 < p) :
  (volume_sphere p) / (volume_hemisphere (2 * p)) = 1 / 4 :=
by
  sorry

end ratio_volume_sphere_to_hemisphere_l784_784101


namespace probability_of_intersecting_diagonals_l784_784444

def num_vertices := 8
def total_diagonals := (num_vertices * (num_vertices - 3)) / 2
def total_ways_to_choose_two_diagonals := Nat.choose total_diagonals 2
def ways_to_choose_4_vertices := Nat.choose num_vertices 4
def number_of_intersecting_pairs := ways_to_choose_4_vertices
def probability_intersecting_diagonals := (number_of_intersecting_pairs : ℚ) / (total_ways_to_choose_two_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  probability_intersecting_diagonals = 7 / 19 := by
  sorry

end probability_of_intersecting_diagonals_l784_784444


namespace sin_A_plus_cos_A_eq_l784_784314

theorem sin_A_plus_cos_A_eq (A : ℝ) (h0 : 0 < A) (h1 : A < π / 2) (h2 : sin (2 * A) = 2 / 3) : sin A + cos A = sqrt (15) / 3 := 
by 
  sorry

end sin_A_plus_cos_A_eq_l784_784314


namespace largest_integer_k_is_3_l784_784372

def T : ℕ → ℕ
| 1       := 2
| (n + 1) := 2 ^ T n

noncomputable def A : ℕ := T 4 ^ T 4

noncomputable def B : ℕ := T 4 ^ A

theorem largest_integer_k_is_3 :
  ∃ k : ℕ, ∀ m : ℕ, (k = 3 → (m > 3 → ¬(log₂^[m] B).isNat)) :=
begin
  sorry
end

end largest_integer_k_is_3_l784_784372


namespace cube_diagonal_length_l784_784862

-- Define the volume of a cube plus three times the total length of its edges equals 
-- twice its surface area, and prove that the long diagonal is 6√3

theorem cube_diagonal_length (s : ℝ) (h : s^3 + 3 * (12 * s) = 2 * (6 * s^2)) : 
  s = 6 → (s * Real.sqrt 3) = 6 * Real.sqrt 3 :=
by
  intro hs
  rw hs
  rfl


end cube_diagonal_length_l784_784862


namespace money_spent_and_left_ratio_l784_784374

noncomputable theory

-- Definition of the conditions
def initial_amount : ℝ := 65
def ice_cream_cost : ℝ := 5
def remaining_after_ice_cream (init_amt ice_cream: ℝ) := init_amt - ice_cream
def deposit_fraction : ℝ := 1 / 5
def final_cash : ℝ := 24

-- Define the money left after buying ice cream
def money_left_after_ice_cream := remaining_after_ice_cream initial_amount ice_cream_cost
def money_spent_on_tshirt (T : ℝ) := money_left_after_ice_cream - T -- T is the money spent on tshirt

-- Initializing the proof of the ratio
theorem money_spent_and_left_ratio : 
  ∃ T : ℝ, 
    money_spent_on_tshirt T - deposit_fraction * money_spent_on_tshirt T = final_cash ∧
    (T / money_left_after_ice_cream : ℝ) = 1 / 2 :=
by sorry

end money_spent_and_left_ratio_l784_784374


namespace seat_arrangement_distinct_ways_l784_784931

theorem seat_arrangement_distinct_ways :
  let n : ℕ := 12
  let k1 : ℕ := 2
  let k2 : ℕ := 4
  let k3 : ℕ := 6
  let binom (n k : ℕ) := Nat.choose n k
  binom 12 2 * binom 10 4 * binom 6 6 = 13860 :=
by 
  -- Definitions from conditions
  let n : ℕ := 12
  let k1 : ℕ := 2
  let k2 : ℕ := 4
  let k3 : ℕ := 6
  -- Combination formula
  let binom (n k : ℕ) := Nat.choose n k
  -- Calculations provided in the solution (omitting actual proof for brevity)
  have h1 : binom 12 2 = 66 := by sorry
  have h2 : binom 10 4 = 210 := by sorry
  have h3 : binom 6 6 = 1 := by sorry
  calc
    binom 12 2 * binom 10 4 * binom 6 6 = 66 * 210 * 1 : by rw [h1, h2, h3]
    ... = 13860 : by norm_num

end seat_arrangement_distinct_ways_l784_784931


namespace suitable_sampling_method_l784_784151

theorem suitable_sampling_method 
  (seniorTeachers : ℕ)
  (intermediateTeachers : ℕ)
  (juniorTeachers : ℕ)
  (totalSample : ℕ)
  (totalTeachers : ℕ)
  (prob : ℚ)
  (seniorSample : ℕ)
  (intermediateSample : ℕ)
  (juniorSample : ℕ)
  (excludeOneSenior : ℕ) :
  seniorTeachers = 28 →
  intermediateTeachers = 54 →
  juniorTeachers = 81 →
  totalSample = 36 →
  excludeOneSenior = 27 →
  totalTeachers = excludeOneSenior + intermediateTeachers + juniorTeachers →
  prob = totalSample / totalTeachers →
  seniorSample = excludeOneSenior * prob →
  intermediateSample = intermediateTeachers * prob →
  juniorSample = juniorTeachers * prob →
  seniorSample + intermediateSample + juniorSample = totalSample :=
by
  intros hsenior hins hjunior htotal hexclude htotalTeachers hprob hseniorSample hintermediateSample hjuniorSample
  sorry

end suitable_sampling_method_l784_784151


namespace total_games_played_l784_784913

theorem total_games_played (n : ℕ) (h : n = 8) : (n.choose 2) = 28 := by
  sorry

end total_games_played_l784_784913


namespace max_two_scoop_sundaes_l784_784966

theorem max_two_scoop_sundaes (n : ℕ) (h : n = 8) : (nat.choose n 2) = 28 :=
by {
  rw h,
  exact nat.choose_eq_factorial_div_factorial (8 : ℕ) (2 : ℕ),
  norm_num
}

end max_two_scoop_sundaes_l784_784966


namespace find_f_7_5_l784_784265

def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f(x) = f(-x)
axiom f_periodic : ∀ x : ℝ, f(x) = -f(x + 2)
axiom f_interval : ∀ x : ℝ, 2 < x ∧ x < 3 → f(x) = 3 - x

theorem find_f_7_5 : f(7.5) = -0.5 := 
by
  sorry

end find_f_7_5_l784_784265


namespace solve_for_a_l784_784429

theorem solve_for_a :
  ∀ (a x y : ℤ), x = 1 → y = 2 → (a * x - y = 3) → a = 5 := 
by 
  intros a x y hx hy h_eq
  rw [hx] at h_eq
  rw [hy] at h_eq
  exact eq_add_of_sub_eq h_eq

end solve_for_a_l784_784429


namespace rationalize_denominator_l784_784062

/-- Rationalizing the denominator of an expression involving cube roots -/
theorem rationalize_denominator :
  (1 : ℝ) / (real.cbrt 3 + real.cbrt 27) = real.cbrt 9 / (12 : ℝ) :=
by
  -- Define conditions
  have h1 : real.cbrt 27 = 3 * real.cbrt 3, by sorry,
  -- Proof of the equality, skipped using sorry
  sorry

end rationalize_denominator_l784_784062


namespace radius_of_circular_mat_l784_784152

theorem radius_of_circular_mat (side_len : ℝ) (fraction_covered : ℝ) (pi_approx : ℝ) :
  side_len = 24 → 
  fraction_covered = 0.545415391248228 →
  pi_approx = real.pi →
  real.sqrt ((fraction_covered * side_len^2) / pi_approx) ≈ 10 :=
by
  intros h_side h_fraction h_pi
  sorry

end radius_of_circular_mat_l784_784152


namespace sum_of_undefined_values_l784_784895

theorem sum_of_undefined_values : 
  let P := λ x : ℝ, x^2 - 7 * x + 10 = 0 in
  (∑ x in {x : ℝ | P x}, x) = 7 := by
  sorry

end sum_of_undefined_values_l784_784895


namespace tower_height_l784_784868

theorem tower_height (h : ℝ) (hd : ¬ (h ≥ 200)) (he : ¬ (h ≤ 150)) (hf : ¬ (h ≤ 180)) : 180 < h ∧ h < 200 := 
by 
  sorry

end tower_height_l784_784868


namespace intersection_M_N_l784_784651

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784651


namespace number_of_correct_propositions_l784_784032

variable (S : Set ℝ)
variable (m n : ℝ)

def condition (x : ℝ) : Prop := m ≤ x ∧ x ≤ n
def invariant_condition (x : ℝ) : Prop := condition m n x → condition m n (x^2)

-- Propositions
def prop1 : Prop := m = 1 → S = {1}
def prop2 : Prop := m = -1/2 → 1/4 ≤ n ∧ n ≤ 1
def prop3 : Prop := n = 1/2 → -real.sqrt(2)/2 ≤ m ∧ m ≤ 0

-- The final statement that needs to be proven
theorem number_of_correct_propositions : 
  (∀ x, condition m n x → invariant_condition m n x) →
  (prop1 m n S) ∧ (prop2 m n) ∧ (prop3 m n) ∧
  (∀ (p1 p2 p3 : Prop), (p1 ∧ p2 ∧ p3 → p1 ∧ p2 ∧ p3 = true → true)) → 3 :=
by sorry

end number_of_correct_propositions_l784_784032


namespace coefficient_of_q_is_3_l784_784281

variable (q : ℝ)
variable (q' : ℝ → ℝ)
variable (h1 : ∀ q, q' q = 3 * q - 3)
variable (h2 : q' (q' 6) = 210)

theorem coefficient_of_q_is_3 : ∀ q, (∃ c, ∀ q, q' q = c * q - 3) :=
by
  exists 3
  intros q
  exact h1 q
sorry

end coefficient_of_q_is_3_l784_784281


namespace algebraic_expression_interpretation_l784_784755

def donations_interpretation (m n : ℝ) : ℝ := 5 * m + 2 * n
def plazas_area_interpretation (a : ℝ) : ℝ := 6 * a^2

theorem algebraic_expression_interpretation (m n a : ℝ) :
  donations_interpretation m n = 5 * m + 2 * n ∧ plazas_area_interpretation a = 6 * a^2 :=
by
  sorry

end algebraic_expression_interpretation_l784_784755


namespace squirrel_total_time_l784_784952

-- Given conditions
def distance1 : ℝ := 0.5
def speed1 : ℝ := 6
def distance2 : ℝ := 1.5
def speed2 : ℝ := 3

-- Computation for the required total time
def total_time_hours : ℝ := (distance1 / speed1) + (distance2 / speed2)
def total_time_minutes : ℝ := total_time_hours * 60

theorem squirrel_total_time : total_time_minutes = 35 := by
  sorry

end squirrel_total_time_l784_784952


namespace calculate_pens_l784_784844

theorem calculate_pens (P : ℕ) (Students : ℕ) (Pencils : ℕ) (h1 : Students = 40) (h2 : Pencils = 920) (h3 : ∃ k : ℕ, Pencils = Students * k) 
(h4 : ∃ m : ℕ, P = Students * m) : ∃ k : ℕ, P = 40 * k := by
  sorry

end calculate_pens_l784_784844


namespace soda_amount_l784_784389

theorem soda_amount (S : ℝ) (h1 : S / 2 + 2000 = (S - (S / 2 + 2000)) / 2 + 2000) : S = 12000 :=
by
  sorry

end soda_amount_l784_784389


namespace smallest_n_Sn_lt_neg4_l784_784707

open Real

def a_n (n : ℕ) : ℝ := log 3 (n / (n + 1))

def S_n (n : ℕ) : ℝ := 
  let rec aux (k : ℕ) (acc : ℝ) : ℝ :=
    match k with
    | 0 => acc
    | (k' + 1) => aux k' (acc + a_n (k' + 1))
  aux n 0

theorem smallest_n_Sn_lt_neg4 : ∃ n : ℕ, S_n n < -4 ∧ ∀ m : ℕ, m < n → ¬(S_n m < -4) :=
sorry

end smallest_n_Sn_lt_neg4_l784_784707


namespace palindrome_divisible_by_11_probability_l784_784507

open Nat

def is_palindrome_three_digit (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  100 ≤ n ∧ n < 1000 ∧ d1 ≠ 0 ∧ d1 = d3

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem palindrome_divisible_by_11_probability :
  let palindromes := {n : ℕ | is_palindrome_three_digit n}
  let divisors := {n : ℕ | is_palindrome_three_digit n ∧ is_divisible_by_11 n}
  fintype.card divisors = 5 →
  fintype.card palindromes = 90 →
  (fintype.card divisors : ℚ) / (fintype.card palindromes : ℚ) = 1 / 18 := by
  intros palindromes divisors H1 H2
  sorry

end palindrome_divisible_by_11_probability_l784_784507


namespace rationalize_denominator_l784_784054

theorem rationalize_denominator :
  (1 / (Real.cbrt 3 + Real.cbrt (3^3))) = (Real.cbrt 9 / 12) :=
by {
  sorry
}

end rationalize_denominator_l784_784054


namespace overall_loss_percentage_l784_784500

structure Property :=
  (cost : ℝ)
  (rate_of_return : ℝ)

def first_property : Property := { cost := 675958, rate_of_return := 0.12 }
def second_property : Property := { cost := 785230, rate_of_return := -0.08 }
def third_property : Property := { cost := 924150, rate_of_return := 0.05 }
def fourth_property : Property := { cost := 1134500, rate_of_return := -0.10 }

def total_cost (properties : List Property) : ℝ :=
  properties.foldl (λ acc p => acc + p.cost) 0

def total_gain_or_loss (properties : List Property) : ℝ :=
  properties.foldl (λ acc p => acc + (p.cost * p.rate_of_return)) 0

def overall_gain_or_loss_percentage (properties : List Property) : ℝ :=
  let net_gain_or_loss := total_gain_or_loss properties
  let total_cost := total_cost properties
  (net_gain_or_loss / total_cost) * 100

theorem overall_loss_percentage : overall_gain_or_loss_percentage [
  first_property, second_property, third_property, fourth_property
] = -1.39 := by
  sorry

end overall_loss_percentage_l784_784500


namespace number_of_lines_l784_784291

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 4 = 1

def point_P : ℝ × ℝ := (1, 0)

theorem number_of_lines (L : ℝ → ℝ → Prop) :
  (∀ x y, L x y ↔ ∃ m b, y = m * x + b ∧ L 1 0) →
  (∀ L, (∃ x y, L x y ∧ hyperbola x y) → false) →
  L 1 0 →
  ∃ n, n = 3 :=
  sorry

end number_of_lines_l784_784291


namespace intersection_M_N_l784_784613

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784613


namespace great_circle_bisects_angle_l784_784465

noncomputable def north_pole : Point := sorry
noncomputable def equator_point (C : Point) : Prop := sorry
noncomputable def great_circle_through (P Q : Point) : Circle := sorry
noncomputable def equidistant_from_N (A B N : Point) : Prop := sorry
noncomputable def spherical_triangle (A B C : Point) : Triangle := sorry
noncomputable def bisects_angle (C N A B : Point) : Prop := sorry

theorem great_circle_bisects_angle
  (N A B C: Point)
  (hN: N = north_pole)
  (hA: equidistant_from_N A B N)
  (hC: equator_point C)
  (hTriangle: spherical_triangle A B C)
  : bisects_angle C N A B :=
sorry

end great_circle_bisects_angle_l784_784465


namespace prob_max_sum_l784_784363

variables {a b c : ℤ}  -- Declaring a, b, c as integers
def matrix_A := (1 / 7 : ℚ) • (Matrix.of (λ i j, ![(⟦-4, a⟧, ⟦b, c⟧) i j : ℤ]))

def is_identity (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop := A.mul A = 1

theorem prob_max_sum (hA_id: is_identity matrix_A) : ∃ a b c, a + b + c = 30 := by sorry

end prob_max_sum_l784_784363


namespace sum_of_angles_satisfying_condition_l784_784570

theorem sum_of_angles_satisfying_condition :
  (∑ x in (Finset.filter (λ x : ℝ, 0 ≤ x ∧ x ≤ 360
                              ∧ sin x ^ 3 + cos x ^ 3 = 1 / cos x + 1 / sin x 
                             ) (Finset.range 361)),
      x) = 450 := by
sorry

end sum_of_angles_satisfying_condition_l784_784570


namespace part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l784_784144

section part1
variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 2) : (x + a) * (x - 2 * a + 1) < 0 ↔ -2 < x ∧ x < 3 :=
by
  sorry
end part1

section part2
variable (x a : ℝ)

-- Case: a = 1
theorem part2_a_eq_1 (h : a = 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ False :=
by
  sorry

-- Case: a > 1
theorem part2_a_gt_1 (h : a > 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 1 < x ∧ x < 2 * a - 1 :=
by
  sorry

-- Case: a < 1
theorem part2_a_lt_1 (h : a < 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 2 * a - 1 < x ∧ x < 1 :=
by
  sorry
end part2

end part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l784_784144


namespace route_Y_saves_2_minutes_l784_784798

noncomputable def distance_X : ℝ := 8
noncomputable def speed_X : ℝ := 40

noncomputable def distance_Y1 : ℝ := 5
noncomputable def speed_Y1 : ℝ := 50
noncomputable def distance_Y2 : ℝ := 1
noncomputable def speed_Y2 : ℝ := 20
noncomputable def distance_Y3 : ℝ := 1
noncomputable def speed_Y3 : ℝ := 60

noncomputable def t_X : ℝ := (distance_X / speed_X) * 60
noncomputable def t_Y1 : ℝ := (distance_Y1 / speed_Y1) * 60
noncomputable def t_Y2 : ℝ := (distance_Y2 / speed_Y2) * 60
noncomputable def t_Y3 : ℝ := (distance_Y3 / speed_Y3) * 60
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3

noncomputable def time_saved : ℝ := t_X - t_Y

theorem route_Y_saves_2_minutes :
  time_saved = 2 := by
  sorry

end route_Y_saves_2_minutes_l784_784798


namespace intersection_M_N_l784_784612

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784612


namespace integral_solutions_count_l784_784846

theorem integral_solutions_count :
  {n : ℕ // n = (finset.count (λ (p : ℕ × ℕ × ℕ × ℕ), let ⟨a, b, c, d⟩ := p in
      a < b ∧ b < c ∧ c < d ∧  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
      1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) + 1 / (d : ℝ) = 1) 
         ({m : ℕ | 0 < m}.product {m : ℕ | 1 < m}.product {m : ℕ | 0 < m}.product {m : ℕ | 0 < m}))} :=
⟨6, sorry⟩

end integral_solutions_count_l784_784846


namespace original_cost_l784_784170

theorem original_cost (P : ℝ) (h : 0.85 * 0.76 * P = 988) : P = 1529.41 := by
  sorry

end original_cost_l784_784170


namespace smallest_value_a_b_c_l784_784883

-- Define the given summation
def T_50 : ℚ := ∑ k in finset.range 50, (-1 : ℚ)^k * (k^3 + k^2 + k + 1) / k!

-- Define our target values
def a : ℕ := 2603
def b : ℕ := 50
def c : ℕ := 1

-- State the Lean theorem to be proven
theorem smallest_value_a_b_c : T_50 = a / b! - c ∧ a + b + c = 2654 :=
by
  -- The transformation and verification steps would go here...
  sorry

end smallest_value_a_b_c_l784_784883


namespace days_x_finishes_remaining_work_l784_784914

theorem days_x_finishes_remaining_work (X Y : Type) 
  (X_rate : ℝ) (Y_rate : ℝ) (Y_days_worked : ℝ) :
  (X_rate = 1/20) →
  (Y_rate = 1/16) →
  (Y_days_worked = 12) →
  let remaining_work := 1 - (Y_days_worked * Y_rate)
  let X_days_needed := remaining_work / X_rate
  X_days_needed = 5 :=
by
  intros hX_rate hY_rate hY_days_worked 
  let remaining_work := 1 - (Y_days_worked * Y_rate)
  let X_days_needed := remaining_work / X_rate
  have : remaining_work = 1/4 := by
    rw [hY_rate, hY_days_worked]
    norm_num
  have : X_days_needed = 5 := by
    rw [hX_rate]
    norm_num
  exact this

end days_x_finishes_remaining_work_l784_784914


namespace common_chord_length_l784_784712

/-- 
Given two circles \( C_{1}: x^{2} + y^{2} + 2x + 8y - 8 = 0 \) and \( C_{2}: x^{2} + y^{2} - 4x - 4y - 2 = 0 ) that intersect,
prove that the length of their common chord is \( 2 \sqrt{5} \).
-/
theorem common_chord_length :
  let C1 := λ x y : ℝ, x^2 + y^2 + 2*x + 8*y - 8 = 0
  let C2 := λ x y : ℝ, x^2 + y^2 - 4*x - 4*y - 2 = 0
  ∃ l : ℝ, l = 2 * Real.sqrt 5 :=
sorry

end common_chord_length_l784_784712


namespace number_increase_when_reversed_l784_784505

theorem number_increase_when_reversed :
  let n := 253
  let reversed_n := 352
  reversed_n - n = 99 :=
by
  let n := 253
  let reversed_n := 352
  sorry

end number_increase_when_reversed_l784_784505


namespace largest_common_value_less_1000_l784_784400

theorem largest_common_value_less_1000 : ∃ a < 1000, (∃ n m : ℕ, a = 5 + 4 * n ∧ a = 4 + 8 * m) ∧ a = 993 := 
  sorry

end largest_common_value_less_1000_l784_784400


namespace sarah_daily_candy_consumption_l784_784232

def neighbors_candy : ℕ := 66
def sister_candy : ℕ := 15
def days : ℕ := 9

def total_candy : ℕ := neighbors_candy + sister_candy
def average_daily_consumption : ℕ := total_candy / days

theorem sarah_daily_candy_consumption : average_daily_consumption = 9 := by
  sorry

end sarah_daily_candy_consumption_l784_784232


namespace f_at_pos_eq_l784_784408

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 0 then x * (x - 1)
  else if h : x > 0 then -x * (x + 1)
  else 0

theorem f_at_pos_eq (x : ℝ) (hx : 0 < x) : f x = -x * (x + 1) :=
by
  -- Assume f is an odd function
  have h_odd : ∀ x : ℝ, f (-x) = -f x := sorry
  
  -- Given for x in (-∞, 0), f(x) = x * (x - 1)
  have h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1) := sorry
  
  -- Prove for x > 0, f(x) = -x * (x + 1)
  sorry

end f_at_pos_eq_l784_784408


namespace slope_parallel_line_l784_784456

theorem slope_parallel_line (x y : ℝ) (a b c : ℝ) (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = 1 / 2 :=
by 
  sorry

end slope_parallel_line_l784_784456


namespace Mary_avg_speed_l784_784035

def Mary_uphill_distance := 1.5 -- km
def Mary_uphill_time := 45.0 / 60.0 -- hours
def Mary_downhill_distance := 1.5 -- km
def Mary_downhill_time := 15.0 / 60.0 -- hours

def total_distance := Mary_uphill_distance + Mary_downhill_distance
def total_time := Mary_uphill_time + Mary_downhill_time

theorem Mary_avg_speed : 
  (total_distance / total_time) = 3.0 := by
  sorry

end Mary_avg_speed_l784_784035


namespace intersection_M_N_l784_784654

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784654


namespace revenue_decrease_l784_784380

noncomputable theory

variables (P Q : ℝ)

def P_new := P * 1.75
def Q_new := Q * 0.55

def original_revenue := P * Q
def new_revenue := P_new P * Q_new Q

theorem revenue_decrease : 
  (new_revenue P Q - original_revenue P Q) / original_revenue P Q * 100 = -3.75 := by
  sorry

end revenue_decrease_l784_784380


namespace min_tangent_length_l784_784327

-- Define given conditions and question in Lean

-- Point definitions
structure Point where
  x : ℝ
  y : ℝ

def F : Point := ⟨1, 0⟩
def circle_center : Point := ⟨3, 0⟩
def circle_radius_sq : ℝ := 2

-- Definition of condition: point P is on the line x = -1
def on_line_x_eq_neg1 (P : Point) : Prop :=
  P.x = -1

-- Definition of condition: Q is the midpoint of PF
def is_midpoint (P F Q : Point) : Prop :=
  Q.x = (P.x + F.x) / 2 ∧ Q.y = (P.y + F.y) / 2

-- Definition of perpendicularity condition
def is_perpendicular (P F Q M : Point) : Prop :=
  let slope_PF := if P.x ≠ F.x then (P.y - F.y) / (P.x - F.x) else 0
  let slope_QM := if Q.x ≠ M.x then (Q.y - M.y) / (Q.x - M.x) else 0
  slope_PF * slope_QM = -1

-- Definition to represent M with respect to lambda and point F
def M_eq_lambda_OF (M : Point) : Prop :=
  ∃ λ : ℝ, M.x = λ * F.x ∧ M.y = λ * F.y

-- Length of the tangent line from a point to a circle
def tangent_length (M circle_center : Point) (radius_sq : ℝ) : ℝ :=
  let d_sq := (M.x - circle_center.x)^2 + (M.y - circle_center.y)^2
  Real.sqrt (d_sq - radius_sq)

-- Theorem we want to prove
theorem min_tangent_length : ∀ (P Q M : Point),
  on_line_x_eq_neg1 P →
  is_midpoint P F Q →
  is_perpendicular P F Q M →
  M_eq_lambda_OF M →
  (∃ A B : Point, tangent_length M circle_center circle_radius_sq = Real.sqrt 6) :=
sorry

end min_tangent_length_l784_784327


namespace intersection_M_N_l784_784633

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784633


namespace sum_opposite_abs_val_eq_neg_nine_l784_784857

theorem sum_opposite_abs_val_eq_neg_nine (a b : ℤ) (h1 : a = -15) (h2 : b = 6) : a + b = -9 := 
by
  -- conditions given
  rw [h1, h2]
  -- skip the proof
  sorry

end sum_opposite_abs_val_eq_neg_nine_l784_784857


namespace largest_possible_k_l784_784855

open Finset

-- Define the set X with 1983 elements
def X : Finset ℕ := range 1983

-- Define the family of subsets
variable {S : ℕ → Finset ℕ}

-- State the conditions as hypotheses
variable (h1 : ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → (S i) ∪ (S j) ∪ (S k) = X)
variable (h2 : ∀ (i j : ℕ), i ≠ j → ((S i) ∪ (S j)).card ≤ 1979)

-- Define the proof problem
theorem largest_possible_k : ∃ k : ℕ, (∀ (S : Finset ℕ → Type), 
  (∀ i, (S i).subset X) → 
  (∀ i j k, i ≠ j → j ≠ k → k ≠ i → (S i) ∪ (S j) ∪ (S k) = X) → 
  (∀ i j, i ≠ j → ((S i) ∪ (S j)).card ≤ 1979) → 
  k ≤ 31 ∧ ¬(k + 1 ≤ 31) ) :=
sorry

end largest_possible_k_l784_784855


namespace midpoint_sum_l784_784457

theorem midpoint_sum (x1 y1 z1 x2 y2 z2 : ℝ) (h1 : x1 = 10) (h2 : y1 = -3) (h3 : z1 = 6) (h4 : x2 = 4) (h5 : y2 = 7) (h6 : z2 = -2) :
  let midpoint := ( (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2 )
  in (midpoint.1 + midpoint.2 + midpoint.3) = 11 :=
by
  sorry

end midpoint_sum_l784_784457


namespace inequality_proof_l784_784072

theorem inequality_proof (a : ℝ) : 
  2 * a^4 + 2 * a^2 - 1 ≥ (3 / 2) * (a^2 + a - 1) :=
by
  sorry

end inequality_proof_l784_784072


namespace even_dimensions_of_valid_coloring_l784_784213

def valid_coloring (m n : ℕ) (C : set ℕ) (color : ℕ → ℕ → ℕ) : Prop :=
  ∀ i ∈ C, ∀ x y : ℕ, color x y = i → 
    ((x > 0 ∧ color (x-1) y = i) + (x < m-1 ∧ color (x+1) y = i) + (y > 0 ∧ color x (y-1) = i) + (y < n-1 ∧ color x (y+1) = i) = 2)

theorem even_dimensions_of_valid_coloring (m n : ℕ) (C : set ℕ) (color : ℕ → ℕ → ℕ) 
  (h1 : 0 < m) (h2 : 0 < n) (hC : C.nonempty) (h : valid_coloring m n C color) :
  Even m ∧ Even n :=
sorry

end even_dimensions_of_valid_coloring_l784_784213


namespace defective_units_shipped_are_less_than_one_percent_l784_784335

-- Define the stages of production
def stage_one_production_defective_rate : ℝ := 0.06
def stage_two_production_defective_rate : ℝ := 0.03
def stage_three_production_defective_rate : ℝ := 0.02

-- Define the stages of shipping
def stage_one_shipping_defective_rate : ℝ := 0.04
def stage_two_shipping_defective_rate : ℝ := 0.03
def stage_three_shipping_defective_rate : ℝ := 0.02

-- Calculate the number of defective units after each production stage
def defective_units_after_stage_one (initial_units : ℝ) : ℝ :=
  initial_units * stage_one_production_defective_rate

def non_defective_units_after_stage_one (initial_units : ℝ) : ℝ :=
  initial_units - defective_units_after_stage_one initial_units

def defective_units_after_stage_two (initial_units : ℝ) : ℝ :=
  defective_units_after_stage_one initial_units +
  non_defective_units_after_stage_one initial_units * stage_two_production_defective_rate

def non_defective_units_after_stage_two (initial_units : ℝ) : ℝ :=
  non_defective_units_after_stage_one initial_units - 
  non_defective_units_after_stage_one initial_units * stage_two_production_defective_rate

def defective_units_after_stage_three (initial_units : ℝ) : ℝ :=
  defective_units_after_stage_two initial_units +
  non_defective_units_after_stage_two initial_units * stage_three_production_defective_rate

-- Calculate the number of shipped defective units after shipping stages
def defective_units_shipped_stage_one (defective_units : ℝ) : ℝ :=
  defective_units * stage_one_shipping_defective_rate

def defective_units_remaining_after_stage_one_shipping (defective_units : ℝ) : ℝ :=
  defective_units - defective_units_shipped_stage_one defective_units

def defective_units_shipped_stage_two (defective_units : ℝ) : ℝ :=
  defective_units_remaining_after_stage_one_shipping defective_units * stage_two_shipping_defective_rate

def defective_units_remaining_after_stage_two_shipping (defective_units : ℝ) : ℝ :=
  defective_units_remaining_after_stage_one_shipping defective_units - 
  defective_units_shipped_stage_two defective_units

def defective_units_shipped_stage_three (defective_units : ℝ) : ℝ :=
  defective_units_remaining_after_stage_two_shipping defective_units * stage_three_shipping_defective_rate

def total_defective_units_shipped (initial_units : ℝ) : ℝ :=
  defective_units_shipped_stage_one (defective_units_after_stage_three initial_units) + 
  defective_units_shipped_stage_two (defective_units_after_stage_three initial_units) + 
  defective_units_shipped_stage_three (defective_units_after_stage_three initial_units)

def percent_defective_units_shipped (initial_units : ℝ) : ℝ :=
  (total_defective_units_shipped initial_units) / initial_units * 100

theorem defective_units_shipped_are_less_than_one_percent : 
  ∀ initial_units : ℝ, initial_units > 0 → percent_defective_units_shipped initial_units < 1 :=
sorry

end defective_units_shipped_are_less_than_one_percent_l784_784335


namespace fixed_point_coordinates_l784_784287

-- Define the function y = log_a(x - 3) - 1
def function (a : Real) (x : Real) : Real := log a (x - 3) - 1

theorem fixed_point_coordinates (a : Real) (h : ∀ x y : Real, y = log a (x - 3) - 1 → (x, y) = (4, -1)) :
  ∃ P : Real × Real, P = (4, -1) :=
by
  use (4, -1)
  sorry

end fixed_point_coordinates_l784_784287


namespace intersection_M_N_l784_784645

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784645


namespace rationalize_denominator_correct_l784_784057

noncomputable def rationalize_denominator : Prop := 
  (1 / (Real.cbrt 3 + Real.cbrt 27) = Real.cbrt 9 / 12)

theorem rationalize_denominator_correct : rationalize_denominator := 
  sorry

end rationalize_denominator_correct_l784_784057


namespace log_base_10_of_100_l784_784918

theorem log_base_10_of_100 : log 10 100 = 2 :=
by
  sorry

end log_base_10_of_100_l784_784918


namespace sum_of_coefficients_l784_784739

theorem sum_of_coefficients (n : ℕ) (h : n = 6) 
  (coeff_largest_fourth_term : ∀ k, k ≠ 3 → binomial n k < binomial n 3) : 
  (1 / 2) ^ n = 1 / 64 :=
by
  rw h
  norm_num
  sorry

end sum_of_coefficients_l784_784739


namespace intersection_M_N_l784_784605

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784605


namespace min_value_of_function_l784_784458

noncomputable def f (a x : ℝ) : ℝ := (a^x - a)^2 + (a^(-x) - a)^2

theorem min_value_of_function (a : ℝ) (h : a > 0) : ∃ x : ℝ, f a x = 2 :=
by
  sorry

end min_value_of_function_l784_784458


namespace tan_alpha_l784_784303

variable (α : ℝ)

theorem tan_alpha (h₁ : Real.sin α = -5/13) (h₂ : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l784_784303


namespace average_percent_students_l784_784858

theorem average_percent_students : 
  let midville_students := 150 
  let easton_students := 250 
  let midville_dist := {K := 0.18, grade1 := 0.14, grade2 := 0.15, grade3 := 0.12, grade4 := 0.16, grade5 := 0.12, grade6 := 0.13}
  let easton_dist := {K := 0.10, grade1 := 0.14, grade2 := 0.17, grade3 := 0.18, grade4 := 0.13, grade5 := 0.15, grade6 := 0.13}
  let num_5th_6th_in_midville := midville_students * (midville_dist.grade5 + midville_dist.grade6)
  let num_5th_6th_in_easton := easton_students * (easton_dist.grade5 + easton_dist.grade6)
  let total_students := midville_students + easton_students
  let total_5th_6th_students := num_5th_6th_in_midville + num_5th_6th_in_easton
  average_percent : 100 * total_5th_6th_students / total_students = 27.25 :=
by
  sorry

end average_percent_students_l784_784858


namespace vodka_mixture_profit_percentage_l784_784117

theorem vodka_mixture_profit_percentage 
  (C1 C2 : ℝ) 
  (profit_ratio1 profit_ratio2 : ℝ) 
  (h1 : profit_ratio1 = 40 * 4 / 3) 
  (h2 : profit_ratio2 = 20 * 5 / 3) :
  let avg_profit_ratio : ℝ := ((profit_ratio1 + profit_ratio2) / 2) in
  avg_profit_ratio = 43.33 :=
by
  have h1 := by norm_num [h1]
  have h2 := by norm_num [h2]
  -- Calculate avg_profit_ratio
  have h3 : avg_profit_ratio = (h1 + h2) / 2 := rfl
  sorry

end vodka_mixture_profit_percentage_l784_784117


namespace expression_factorized_l784_784091

-- Definitions based on the conditions
theorem expression_factorized (C D E F : ℤ) (y : ℝ) :
  (10 * y^2 - 51 * y + 21 = (C * y - D) * (E * y - F)) → (C * E + C = 15) :=
by
  intro h
  have : C = 5 ∧ E = 2 := sorry -- Placeholder to indicate that C and E values must satisfy the equation
  simp [*]  -- Simplify based on the above assumption
  sorry  -- The final line to complete the proof

end expression_factorized_l784_784091


namespace solve_factorial_equation_l784_784076

theorem solve_factorial_equation : 
  let solutions := [(n, k, l) | n k l : ℕ, n > 1, (∀ k l, k < n ∧ l < n ∧ 2 * (k! + l!) = n! ∧ k! + l! > 1)] in
  if solutions = [] then 0
  else (solutions.map (λ (n, k, l) => n)).eraseDups.sum = 10 := 
sorry

end solve_factorial_equation_l784_784076


namespace workshop_total_workers_l784_784401

noncomputable def total_workers (total_avg_salary technicians_avg_salary non_technicians_avg_salary : ℕ) (technicians_count : ℕ) : ℕ :=
  sorry

theorem workshop_total_workers (avg_salary : ℕ) (tech_avg_salary : ℕ) (non_tech_avg_salary : ℕ) (tech_count : ℕ) :
  total_workers avg_salary tech_avg_salary non_tech_avg_salary tech_count = 49 :=
by {
  -- Given conditions
  let avg_salary := 8000,
  let tech_avg_salary := 20000,
  let non_tech_avg_salary := 6000,
  let tech_count := 7,
  -- Assertions based on these conditions would follow
  sorry
}

end workshop_total_workers_l784_784401


namespace asymptotic_lines_of_hyperbola_l784_784084

theorem asymptotic_lines_of_hyperbola : 
  ∀ x y : ℝ, (x^2 - y^2 / 3 = 1) → (y = sqrt 3 * x ∨ y = -sqrt 3 * x) :=
by
  intro x y h
  sorry

end asymptotic_lines_of_hyperbola_l784_784084


namespace equation_of_line_l784_784426

theorem equation_of_line (slope : ℝ) (A : ℝ × ℝ) (h_slope : slope = 3) (h_pass : A = (1, -2)) :
  ∃ (a b c : ℝ), a = 3 ∧ b = -1 ∧ c = -5 ∧ (∀ (x y : ℝ), y + 2 = slope * (x - 1) → a * x + b * y + c = 0) := by
  use 3, -1, -5
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  intros x y h
  linarith

end equation_of_line_l784_784426


namespace probability_point_in_region_l784_784968

/-- The vertices of the rectangle OABC are given as (0,0), (2pi,0), (2pi,2), (0,2),
and Ω is the region enclosed by the edge BC and the graph of the function y = 1 + cos x
for 0 <= x <= 2pi. Prove that the probability a point M randomly chosen inside the
rectangle OABC lies inside the region Ω is 1/2. -/
theorem probability_point_in_region :
  let length := 2 * Real.pi,
      width := 2,
      area_rectangle := length * width,
      integral := ∫ x in 0..(2 * Real.pi), (1 + Real.cos x),
      area_region := area_rectangle - integral in
  (area_region / area_rectangle) = 1 / 2 :=
by
  -- proof goes here
  sorry

end probability_point_in_region_l784_784968


namespace rationalize_denominator_l784_784061

/-- Rationalizing the denominator of an expression involving cube roots -/
theorem rationalize_denominator :
  (1 : ℝ) / (real.cbrt 3 + real.cbrt 27) = real.cbrt 9 / (12 : ℝ) :=
by
  -- Define conditions
  have h1 : real.cbrt 27 = 3 * real.cbrt 3, by sorry,
  -- Proof of the equality, skipped using sorry
  sorry

end rationalize_denominator_l784_784061


namespace mod_add_l784_784897

theorem mod_add (n : ℕ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end mod_add_l784_784897


namespace algebraic_expression_value_l784_784253

variables (x y : ℝ)

theorem algebraic_expression_value :
  x^2 - 4 * x - 1 = 0 →
  (2 * x - 3)^2 - (x + y) * (x - y) - y^2 = 12 :=
by
  intro h
  sorry

end algebraic_expression_value_l784_784253


namespace cubic_transform_l784_784805

theorem cubic_transform (A B C x z β : ℝ) (h₁ : z = x + β) (h₂ : 3 * β + A = 0) :
  z^3 + A * z^2 + B * z + C = 0 ↔ x^3 + (B - (A^2 / 3)) * x + (C - A * B / 3 + 2 * A^3 / 27) = 0 :=
sorry

end cubic_transform_l784_784805


namespace determine_n_l784_784247

theorem determine_n (n : ℕ) (hn : 0 < n) (h : ⌊⌊91 / n⌋ / n⌋ = 1) : n = 7 ∨ n = 8 ∨ n = 9 := by
  sorry

end determine_n_l784_784247


namespace quadratic_inequality_false_range_l784_784417

theorem quadratic_inequality_false_range (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
by
  sorry

end quadratic_inequality_false_range_l784_784417


namespace sufficient_but_not_necessary_to_cos2alpha_l784_784828

open Real

theorem sufficient_but_not_necessary_to_cos2alpha :
    ∀ (k : ℤ) (α : ℝ), (α = π / 6 + 2 * k * π) →
    (cos (2 * α) = 1 / 2) ∧ (¬ ∀ α, cos (2 * α) = 1 / 2 → (∃ k : ℤ, α = π / 6 + 2 * k * π)) :=
begin
  intro k,
  intro α,
  intro h1,
  split,
  { sorry },  -- Proof that α = π / 6 + 2kπ leads to cos(2α) = 1/2
  { sorry }   -- Proof that cos(2α) = 1/2 does not lead to α = π / 6 + 2kπ
end

end sufficient_but_not_necessary_to_cos2alpha_l784_784828


namespace find_m_find_range_t_l784_784596

-- Define the functions f and h
def f (x : ℝ) (m : ℝ) : ℝ := |x - m|
def h (t : ℝ) : ℝ := |t - 2| + |t + 3|

-- Define the conditions for the first part of the problem
def solution_set_f := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
def condition_f (m : ℝ) : Prop :=
  ∀ x, -1 ≤ x ∧ x ≤ 5 ↔ f(x, m) ≤ 3

-- Define the theorem for the first part
theorem find_m (m : ℝ) : condition_f m → m = 2 :=
by sorry

-- Define the conditions for the second part of the problem
def equation_has_solutions (t : ℝ) : Prop :=
  ∃ x, x^2 + 6 * x + h t = 0

def range_of_t : set ℝ := {t : ℝ | -5 ≤ t ∧ t ≤ 4}

-- Define the theorem for the second part
theorem find_range_t (t : ℝ) : equation_has_solutions t → t ∈ range_of_t :=
by sorry

end find_m_find_range_t_l784_784596


namespace b15_correct_l784_784944

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  if n = 2 then 2 else
  if n = 3 then 3 else
  2 * sequence (n - 1) + sequence (n - 2) - sequence (n - 3)

-- Define the key conditions variables
def b1 : ℕ := 1
def b2 : ℕ := 2
def b3 : ℕ := 3
def relation (n : ℕ) : Prop := 
  sequence (n + 3) = 2 * sequence (n + 2) + sequence (n + 1) - sequence n

-- The main goal
theorem b15_correct: 
  sequence 15 = 46571 := 
sorry

end b15_correct_l784_784944


namespace roots_equation_l784_784785
-- We bring in the necessary Lean libraries

-- Define the conditions as Lean definitions
variable (x1 x2 : ℝ)
variable (h1 : x1^2 + x1 - 3 = 0)
variable (h2 : x2^2 + x2 - 3 = 0)

-- Lean 4 statement we need to prove
theorem roots_equation (x1 x2 : ℝ) (h1 : x1^2 + x1 - 3 = 0) (h2 : x2^2 + x2 - 3 = 0) : 
  x1^3 - 4 * x2^2 + 19 = 0 := 
sorry

end roots_equation_l784_784785


namespace max_profit_at_x_eq_50_l784_784319

noncomputable def f (x : ℝ) : ℝ := -x^2 / 100 + 101 * x / 50 - Real.log (x / 10)
noncomputable def T (x : ℝ) : ℝ := f x - x

-- Provided conditions
axiom h1 : f 10 = 19.2
axiom h2 : f 20 = 35.7
axiom ln2 : Real.log 2 = 0.7
axiom ln3 : Real.log 3 = 1.1
axiom ln5 : Real.log 5 = 1.6

-- Proof problem statement
theorem max_profit_at_x_eq_50 : 
  f = (λ x : ℝ, -x^2 / 100 + 101 * x / 50 - Real.log (x / 10)) ∧ T 50 = 24.4 :=
sorry

end max_profit_at_x_eq_50_l784_784319


namespace min_dominos_in_2x2_l784_784486

/-- A 100 × 100 square is divided into 2 × 2 squares.
Then it is divided into dominos (rectangles 1 × 2 and 2 × 1).
Prove that the minimum number of dominos within the 2 × 2 squares is 100. -/
theorem min_dominos_in_2x2 (N : ℕ) (hN : N = 100) :
  ∃ d : ℕ, d = 100 :=
sorry

end min_dominos_in_2x2_l784_784486


namespace sum_of_coefficients_l784_784856

theorem sum_of_coefficients (x y : ℕ) :
  (x - 3 * y) ^ 19 = - (2 ^ 19) :=
by
  sorry

end sum_of_coefficients_l784_784856


namespace necessarily_positive_l784_784388

theorem necessarily_positive (a b c : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) (hc : 0 < c ∧ c < 3) :
  (b + c) > 0 :=
sorry

end necessarily_positive_l784_784388


namespace crescent_perimeter_l784_784495

def radius_outer : ℝ := 10.5
def radius_inner : ℝ := 6.7

theorem crescent_perimeter : (radius_outer + radius_inner) * Real.pi = 54.037 :=
by
  sorry

end crescent_perimeter_l784_784495


namespace disjoint_subsets_with_same_sum_l784_784889

theorem disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), S.card = 10 ∧ (∀ x ∈ S, x ∈ Finset.range 101) →
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_with_same_sum_l784_784889


namespace count_elements_of_T_l784_784590

-- Define S, n, and the simple operation
variables (S : Type) (n : ℕ)
variable [DecidableEq S]
variable (vars : Fin n → S)
variable (op : S → S → S)

-- Define the properties of the simple operation
axiom simple_operation_associative : ∀ x y z : S, op (op x y) z = op x (op y z)
axiom simple_operation_result : ∀ x y : S, op x y ∈ {x, y}

-- Define a string as a list of variables
def is_full (l : List S) : Prop := ∀ x : S, x ∈ vars → x ∈ l

-- Define equivalence of strings
def string_equiv (l1 l2 : List S) : Prop :=
  ∀ o : S → S → S, (∀ x y z : S, o (o x y) z = o x (o y z)) → (∀ x y : S, o x y ∈ {x, y}) → 
    (reduce_string o l1 = reduce_string o l2)

def reduce_string (o : S → S → S) : List S → S
| [] := vars 0 -- arbitrary choice for empty
| [x] := x
| (x :: xs) := xs.foldl o x

-- Define the set T
def T : Set (List S) := {l | is_full l ∧ ∀ l', is_full l' → string_equiv l l' → l = l'}

-- Statement of the problem
theorem count_elements_of_T : (∃T, ∀ l, l ∈ T → (∃ l', is_full l' ∧ string_equiv l l') ∧ T.finite) → cardinality T = n!^2 :=
sorry

end count_elements_of_T_l784_784590


namespace bungee_cord_extension_l784_784943

variables (m g H k h L₀ T_max : ℝ)
  (mass_nonzero : m ≠ 0)
  (gravity_positive : g > 0)
  (H_positive : H > 0)
  (k_positive : k > 0)
  (L₀_nonnegative : L₀ ≥ 0)
  (T_max_eq : T_max = 4 * m * g)
  (L_eq : L₀ + h = H)
  (hooke_eq : T_max = k * h)

theorem bungee_cord_extension :
  h = H / 2 := sorry

end bungee_cord_extension_l784_784943


namespace coefficient_xy3_l784_784220

theorem coefficient_xy3 :
  let f := (λ x y : ℂ, (sqrt x - y / 2))
  let coeff := (λ (n r : ℕ), (-1 / 2) ^ r * (nat.choose n r) * x ^ ((n - r) / 2) * y ^ r)
  (∃ (x y : ℂ), (f x y) ^ 5 = (-5/4) * x * y^3) := 
sorry

end coefficient_xy3_l784_784220


namespace union_of_A_and_B_l784_784294

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := 
by 
  sorry

end union_of_A_and_B_l784_784294


namespace frank_whack_a_mole_tickets_l784_784135

variable (W : ℕ)
variable (skee_ball_tickets : ℕ := 9)
variable (candy_cost : ℕ := 6)
variable (candies_bought : ℕ := 7)
variable (total_tickets : ℕ := W + skee_ball_tickets)
variable (required_tickets : ℕ := candy_cost * candies_bought)

theorem frank_whack_a_mole_tickets : W + skee_ball_tickets = required_tickets → W = 33 := by
  sorry

end frank_whack_a_mole_tickets_l784_784135


namespace storks_joined_l784_784484

theorem storks_joined (S : ℕ) : 
  let birds := 4 in
  let initial_storks := 3 in
  let total_storks := initial_storks + S in
  total_storks = birds + 5 → S = 6 :=
by
  sorry

end storks_joined_l784_784484


namespace problem_1_problem_2_l784_784267

structure Point2D where
  x : ℝ
  y : ℝ

def is_on_ellipse_C (A B : Point2D) : Prop :=
  (A.x^2 / 25 + A.y^2 / 9 = 1) ∧ (B.x^2 / 25 + B.y^2 / 9 = 1)

def arithmetic_sequence (d1 d2 d3 : ℝ) : Prop :=
  2 * d2 = d1 + d3

def M := Point2D.mk 4 (9 / 5)

theorem problem_1 (A B : Point2D) (h1 : is_on_ellipse_C A B)
  (h2 : arithmetic_sequence (25/4 - A.x) (25/4 - 4) (25/4 - B.x)) : A.x + B.x = 8 := 
sorry

theorem problem_2 (A B : Point2D) (h1 : is_on_ellipse_C A B)
  (h2 : arithmetic_sequence (25/4 - A.x) (25/4 - 4) (25/4 - B.x))
  (h3 : A.x + B.x = 8)
  (h4 : let D : Point2D := ⟨4, (A.y + B.y)/2⟩ in 
    let N : Point2D := ⟨64/25, 0⟩ in 
    ∀ x₀, N = ⟨x₀, 0⟩ ∧ 
            (x₀ - 4 = (A.y^2 - B.y^2) / (2 * (A.x - B.x)))) : 
  ∃ k b : ℝ, (k * M.x + b * M.y + (-k * 64/25 - b * 0)) = 0 ∧ 
             (25 = k) ∧ (b = -20) ∧ (k * k - b * b = 1600) := 
sorry

end problem_1_problem_2_l784_784267


namespace gen_formula_a_n_find_m_exists_m_l784_784013

noncomputable def a_n (n : ℕ) : ℕ :=
  if even n then 2 * 3 ^ (n / 2 - 1) else n

def S_n (n : ℕ) : ℕ :=
  ∑ i in range n, a_n i

-- Prove the general formula for the sequence
theorem gen_formula_a_n (n : ℕ) : 
  ∀ k : ℕ, a_n (2 * k - 1) = 2 * k - 1 ∧ a_n (2 * k) = 2 * 3 ^ (k - 1) :=
sorry

-- Prove the value of m
theorem find_m (m : ℕ) : 
  (a_n m) * (a_n (m + 1)) = a_n (m + 2) → m = 2 :=
sorry

-- Prove the existence of m as described
theorem exists_m (m : ℕ) : 
  ∃ (m : ℕ), (S_n (2 * m) / S_n (2 * m - 1)) = a_n m :=
sorry

end gen_formula_a_n_find_m_exists_m_l784_784013


namespace stephen_female_worker_ants_l784_784081

-- Define the conditions
def stephen_ants : ℕ := 110
def worker_ants (total_ants : ℕ) : ℕ := total_ants / 2
def male_worker_ants (workers : ℕ) : ℕ := (20 / 100) * workers

-- Define the question and correct answer
def female_worker_ants (total_ants : ℕ) : ℕ :=
  let workers := worker_ants total_ants
  workers - male_worker_ants workers

-- The theorem to prove
theorem stephen_female_worker_ants : female_worker_ants stephen_ants = 44 :=
  by sorry -- Skip the proof for now

end stephen_female_worker_ants_l784_784081


namespace probability_success_l784_784114

variable (A_success B_success C_success : ℚ)
variable (independent_events : Prop)

def P (event : Prop) :=
  ∑ e, (if event then 1 else 0) -- This defines the probability, but in real life, we use more advanced probability kernels

theorem probability_success :
  A_success = 1/2 ∧ B_success = 3/4 ∧ C_success = 2/5 ∧ independent_events → 
  P (¬ (¬ A_success ∧ ¬ B_success ∧ ¬ C_success)) = 37/40 :=
by
  sorry

end probability_success_l784_784114


namespace find_x_l784_784776

noncomputable def A (x : ℕ) := {1, 2, x}
def B := {2, 4, 5}
def union_condition (x : ℕ) := A x ∪ B = {1, 2, 3, 4, 5}

theorem find_x : ∃ x : ℕ, union_condition x :=
begin
  use 3,
  unfold union_condition A B,
  simp,
  refl,
end

end find_x_l784_784776


namespace emma_total_investment_l784_784208

theorem emma_total_investment (X : ℝ) (h : 0.09 * 6000 + 0.11 * (X - 6000) = 980) : X = 10000 :=
sorry

end emma_total_investment_l784_784208


namespace maximum_area_of_backyard_l784_784069

noncomputable def maximum_enclosed_area (fence_length : ℝ) : ℝ :=
  let r := fence_length / π in
  (1 / 2) * π * r^2

theorem maximum_area_of_backyard :
  maximum_enclosed_area 400 = 80000 / π :=
by
  sorry

end maximum_area_of_backyard_l784_784069


namespace arithmetic_sequence_sum_l784_784794

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 : ℤ) (h1 : a_1 = -2017) 
  (h2 : (S 2009) / 2009 - (S 2007) / 2007 = 2) : 
  S 2017 = -2017 :=
by
  -- definitions and steps would go here
  sorry

end arithmetic_sequence_sum_l784_784794


namespace SumOfInteriorAnglesOfTriangle_l784_784461

-- Define the conditions as propositions
def EventA : Prop := ∃ seats_available : Prop, seats_available = true
def EventB : Prop := ∃ exam_score_full : Prop, exam_score_full = true
def EventC : Prop := ∃ snow_in_xian : Prop, snow_in_xian = true

-- Event D is already defined as a mathematical fact
def EventD : Prop := ∀ (α β γ : ℝ), α + β + γ = 180 ∧ (α + β + γ = 180 → triangle α β γ)

-- Define a triangle to capture the sum of interior angles
def triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Theorem: EventD is a certain event
theorem SumOfInteriorAnglesOfTriangle : EventD := 
by 
  -- Sorry is used because the actual proof is not required.
  sorry

end SumOfInteriorAnglesOfTriangle_l784_784461


namespace sum_of_possible_a1_l784_784258

theorem sum_of_possible_a1 (k : ℚ) (a : ℕ → ℚ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = k * a n + 3 * k - 3)
  (h2 : k ≠ 0 ∧ k ≠ 1)
  (h3 : ∀ i : ℕ, 1 < i ∧ i < 6 → a i ∈ {-678, -78, -3, 22, 222, 2222}) :
  (a 1 = -3 ∨ a 1 = -34/3 ∨ a 1 = 2022) →
  ∑ x in {-3, -34/3, 2022}, x = 6023/3 :=
by
  sorry

end sum_of_possible_a1_l784_784258


namespace y_share_l784_784164

theorem y_share (total_amount : ℝ) (x_share y_share z_share : ℝ)
  (hx : x_share = 1) (hy : y_share = 0.45) (hz : z_share = 0.30)
  (h_total : total_amount = 105) :
  (60 * y_share) = 27 :=
by
  have h_cycle : 1 + y_share + z_share = 1.75 := by sorry
  have h_num_cycles : total_amount / 1.75 = 60 := by sorry
  sorry

end y_share_l784_784164


namespace negation_equiv_l784_784413

-- Define acute angle and its properties
def is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- Define the proposition and its negation
def prop (α : ℝ) : Prop := is_acute α → Real.sin α > 0
def neg_prop (α : ℝ) : Prop := ¬(is_acute α → Real.sin α > 0)

-- State the equivalence
theorem negation_equiv (α : ℝ) : (¬(is_acute α → Real.sin α > 0)) ↔ (¬is_acute α → Real.sin α ≤ 0) :=
by 
  sorry

end negation_equiv_l784_784413


namespace column_height_l784_784439

-- Definitions based on problem conditions
def parabola (p : ℝ) (x y : ℝ) := x^2 = -2 * p * y
def span := 16
def arch_height := 4
def support_distance := 4

-- Given constants for the parabola problem
def point_on_parabola := (8, -4)
def p_value := 8

theorem column_height : ∃ y : ℝ, y = -1 :=
by 
  have h : parabola p_value 4 (-1) := by 
  sorry
  use -1
  exact h

end column_height_l784_784439


namespace new_cylinder_height_percentage_l784_784139

variables (r h h_new : ℝ)

theorem new_cylinder_height_percentage :
  (7 / 8) * π * r^2 * h = (3 / 5) * π * (1.25 * r)^2 * h_new →
  (h_new / h) = 14 / 15 :=
by
  intro h_volume_eq
  sorry

end new_cylinder_height_percentage_l784_784139


namespace exists_multiple_of_2009_with_ones_l784_784814

theorem exists_multiple_of_2009_with_ones :
  ∃ n : ℕ, let a_n := (10^n - 1) / 9 in 2009 ∣ a_n := 
by
  sorry

end exists_multiple_of_2009_with_ones_l784_784814


namespace length_of_PP1P2_l784_784396

open Real

theorem length_of_PP1P2 :
  ∃ x ∈ Ioo 0 (π / 2), 4 * tan x = 6 * sin x ∧ cos x = 2 / 3 :=
by
  sorry

end length_of_PP1P2_l784_784396


namespace Bertha_daughters_and_granddaughters_l784_784531

theorem Bertha_daughters_and_granddaughters :
  ∀ (daughters granddaughters x y z : ℕ),
  daughters = 10 →
  granddaughters + daughters = 42 →
  8 * x = granddaughters →
  x + y = daughters →
  y = 6 →
  z = 32 →
  z = granddaughters →
  let females_without_daughters := y + z in
  females_without_daughters = 38 :=
by
  intros daughters granddaughters x y z h1 h2 h3 h4 h5 h6 h7
  sorry

end Bertha_daughters_and_granddaughters_l784_784531


namespace fixed_point_exists_fixed_point_value_l784_784836

noncomputable def f (z : ℂ) : ℂ :=
  (2 + complex.I * real.sqrt 2) * z / 2 + (real.sqrt 2 + 4 * complex.I) / 2 + 1 - 2 * complex.I

theorem fixed_point_exists :
  ∃ c : ℂ, f(c) = c := sorry

-- Stating explicitly the known fixed point from the solution
theorem fixed_point_value :
  ∃ c : ℂ, f(c) = c ∧ c = -complex.I * (1 + real.sqrt 2) := sorry

end fixed_point_exists_fixed_point_value_l784_784836


namespace remainder_when_divided_by_13_is_11_l784_784898

theorem remainder_when_divided_by_13_is_11 
  (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : 
  349 % 13 = 11 := 
by 
  sorry

end remainder_when_divided_by_13_is_11_l784_784898


namespace area_of_triangle_ABC_l784_784342

theorem area_of_triangle_ABC (AB BC BD BE AC E DC BD: ℝ)
  (h1 : AB = BC)
  (h2 : BD = E) -- Here, ‘E’ should correspond to the altitude condition
  (h3 : BE = 15)
  (h4 : tan (angle C BE) = angle_progression ([tan (angle DBE), tan (angle ABE)]))
  (h5 : cot (angle DBE) = geometric_progression ([cot (angle CBE), cot (angle DBC)])) :
  area ABC = 225 / 4 := by
    sorry

end area_of_triangle_ABC_l784_784342


namespace binomial_theorem_sum_l784_784310

theorem binomial_theorem_sum (a : Fin 2019 → ℝ) (x : ℝ) :
  (1 - 3*x)^2018 = ∑ i in Finset.range 2019, a i * x^i →
  a 0 = 1 →
  (3 * (∑ i in Finset.range 2018, a (i + 1) * 3^i)) = 8^2018 - 1 := 
by
  intros h1 h2
  sorry

end binomial_theorem_sum_l784_784310


namespace largest_n_binary_operation_l784_784309

-- Define the binary operation @
def binary_operation (n : ℤ) : ℤ := n - (n * 5)

-- Define the theorem stating the desired property
theorem largest_n_binary_operation (x : ℤ) (h : x > -8) :
  ∃ (n : ℤ), n = 2 ∧ binary_operation n < x :=
sorry

end largest_n_binary_operation_l784_784309


namespace mapping_preserves_set_l784_784257

def M := { p : ℝ × ℝ | p.1 * p.2 = 1 ∧ p.1 > 0 }

def f (p : ℝ × ℝ) : ℝ × ℝ := (Real.log 2 p.1, Real.log 2 p.2)

def N := { q : ℝ × ℝ | q.1 + q.2 = 0 }

theorem mapping_preserves_set : ∀ p ∈ M, f p ∈ N :=
by
  sorry

end mapping_preserves_set_l784_784257


namespace total_spent_on_index_cards_l784_784982

-- Definitions for conditions
def index_cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

-- Theorem to be proven
theorem total_spent_on_index_cards :
  let total_students := students_per_class * periods_per_day
  let total_cards := total_students * index_cards_per_student
  let packs_needed := total_cards / cards_per_pack
  let total_cost := packs_needed * cost_per_pack
  total_cost = 108 :=
by
  sorry

end total_spent_on_index_cards_l784_784982


namespace intersection_M_N_l784_784598

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784598


namespace attendance_ratio_3_to_1_l784_784156

variable (x y : ℕ)
variable (total_attendance : ℕ := 2700)
variable (second_day_attendance : ℕ := 300)

/-- 
Prove that the ratio of the number of people attending the third day to the number of people attending the first day is 3:1
-/
theorem attendance_ratio_3_to_1
  (h1 : total_attendance = 2700)
  (h2 : second_day_attendance = x / 2)
  (h3 : second_day_attendance = 300)
  (h4 : y = total_attendance - x - second_day_attendance) :
  y / x = 3 :=
by
  sorry

end attendance_ratio_3_to_1_l784_784156


namespace intersection_M_N_eq_neg2_l784_784640

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784640


namespace general_formula_geometric_sequence_sum_first_n_terms_l784_784430

noncomputable def geometric_sequence := ∀ (n : ℕ), a_n = (1 / 3) ^ n

theorem general_formula_geometric_sequence (n : ℕ) (a_1 a_2 a_3 a_6 : ℝ) 
  (h1 : 2 * a_1 + 3 * a_2 = 1) (h2 : a_3 ^ 2 = 9 * a_2 * a_6) 
  (h_geometric : ∀ m, a_m * 3 = a_(m + 1)) : 
  ∀ n, a_n = (1 / 3) ^ n := 
sorry

noncomputable def sum_first_n_terms_sequence_c (n : ℕ) : ℝ := 
  let b_n := ∑ k in finset.range n, real.log 3 (1 / 3 ^ (k + 1))
  let c_n := -1 / b_n
  ∑ k in finset.range n, c_n

theorem sum_first_n_terms (n : ℕ) (a_1 a_2 a_3 a_6 : ℝ) 
  (h1 : 2 * a_1 + 3 * a_2 = 1) (h2 : a_3 ^ 2 = 9 * a_2 * a_6) 
  (h_geometric : ∀ m, a_m * 3 = a_(m + 1)) : 
  sum_first_n_terms_sequence_c n = 2 * n / (n + 1) := 
sorry

end general_formula_geometric_sequence_sum_first_n_terms_l784_784430


namespace isabella_hair_length_end_of_year_l784_784344

theorem isabella_hair_length_end_of_year (initial_hair_length hair_growth : ℕ) (h1 : initial_hair_length = 18) (h2 : hair_growth = 6) :
  initial_hair_length + hair_growth = 24 :=
by {
  rw [h1, h2],
  sorry
}

end isabella_hair_length_end_of_year_l784_784344


namespace rationalize_denominator_l784_784052

theorem rationalize_denominator :
  (1 / (Real.cbrt 3 + Real.cbrt (3^3))) = (Real.cbrt 9 / 12) :=
by {
  sorry
}

end rationalize_denominator_l784_784052


namespace probability_of_intersecting_diagonals_l784_784445

def num_vertices := 8
def total_diagonals := (num_vertices * (num_vertices - 3)) / 2
def total_ways_to_choose_two_diagonals := Nat.choose total_diagonals 2
def ways_to_choose_4_vertices := Nat.choose num_vertices 4
def number_of_intersecting_pairs := ways_to_choose_4_vertices
def probability_intersecting_diagonals := (number_of_intersecting_pairs : ℚ) / (total_ways_to_choose_two_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  probability_intersecting_diagonals = 7 / 19 := by
  sorry

end probability_of_intersecting_diagonals_l784_784445


namespace quadratic_factorization_value_of_3d_minus_c_l784_784092

theorem quadratic_factorization :
  ∃ c d : ℕ, c > d ∧ (c = 16) ∧ (d = 4) ∧ x^2 - 20 * x + 96 = (x - c) * (x - d) :=
begin
  sorry,
end

theorem value_of_3d_minus_c :
  ∃ c d : ℕ, c > d ∧ (c = 16) ∧ (d = 4) ∧ (3 * d - c = -4) :=
begin
  sorry,
end

end quadratic_factorization_value_of_3d_minus_c_l784_784092


namespace no_solutions_in_natural_numbers_l784_784807

theorem no_solutions_in_natural_numbers (x y : ℕ) : x^2 + x * y + y^2 ≠ x^2 * y^2 :=
  sorry

end no_solutions_in_natural_numbers_l784_784807


namespace rationalize_denominator_correct_l784_784060

noncomputable def rationalize_denominator : Prop := 
  (1 / (Real.cbrt 3 + Real.cbrt 27) = Real.cbrt 9 / 12)

theorem rationalize_denominator_correct : rationalize_denominator := 
  sorry

end rationalize_denominator_correct_l784_784060


namespace max_result_of_operation_l784_784521

theorem max_result_of_operation : ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 → 3 * (300 - m) ≤ 870) ∧ 3 * (300 - n) = 870 :=
by
  sorry

end max_result_of_operation_l784_784521


namespace product_trailing_zeroes_l784_784852

theorem product_trailing_zeroes {a b : ℕ} (ha : a = 600) (hb : b = 50) : 
  (a * b) % 10000 = 0 ∧ (a * b) ≠ 0 :=
by 
  rw [ha, hb]
  have h : 600 * 50 = 30000 := by norm_num
  rw h 
  exact ⟨by norm_num, by norm_num⟩


end product_trailing_zeroes_l784_784852


namespace cookies_number_l784_784199

-- Define all conditions in the problem
def number_of_chips_per_cookie := 7
def number_of_cookies_per_dozen := 12
def number_of_uneaten_chips := 168

-- Define D as the number of dozens of cookies
variable (D : ℕ)

-- Prove the Lean theorem
theorem cookies_number (h : 7 * 6 * D = 168) : D = 4 :=
by
  sorry

end cookies_number_l784_784199


namespace translated_symmetric_function_l784_784919

theorem translated_symmetric_function (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = real.exp (-x)) → (∀ x, f x = real.exp (-x - 1)) :=
by
  intros h x
  have h1 := h (x + 1)
  simp at h1
  exact h1

end translated_symmetric_function_l784_784919


namespace rationalize_denominator_l784_784055

theorem rationalize_denominator :
  (1 / (Real.cbrt 3 + Real.cbrt (3^3))) = (Real.cbrt 9 / 12) :=
by {
  sorry
}

end rationalize_denominator_l784_784055


namespace kenneth_earnings_l784_784774

theorem kenneth_earnings (E : ℝ) 
  (h_joystick : 0.10 * E)
  (h_accessories : 0.15 * E)
  (h_snacks : 75)
  (h_utility : 80)
  (h_leftover : E - (0.10 * E) - (0.15 * E) - 75 - 80 = 405) :
  E = 746.67 :=
sorry

end kenneth_earnings_l784_784774


namespace find_constant_l784_784969

-- Definitions based on the conditions provided
variable (f : ℕ → ℕ)
variable (c : ℕ)

-- Given conditions
def f_1_eq_0 : f 1 = 0 := sorry
def functional_equation (m n : ℕ) : f (m + n) = f m + f n + c * (m * n - 1) := sorry
def f_17_eq_4832 : f 17 = 4832 := sorry

-- The mathematically equivalent proof problem
theorem find_constant : c = 4 := 
sorry

end find_constant_l784_784969


namespace algebraic_identity_l784_784252

theorem algebraic_identity (x : ℝ) (h : x = Real.sqrt 3 + 2) : x^2 - 4 * x + 3 = 2 := 
by
  -- proof steps here
  sorry

end algebraic_identity_l784_784252


namespace proportion1_proportion2_l784_784394

theorem proportion1 (x : ℚ) : (x / (5 / 9) = (1 / 20) / (1 / 3)) → x = 1 / 12 :=
sorry

theorem proportion2 (x : ℚ) : (x / 0.25 = 0.5 / 0.1) → x = 1.25 :=
sorry

end proportion1_proportion2_l784_784394


namespace abc_eq_ab_bc_ca_l784_784791

variable {u v w A B C : ℝ}
variable (Huvw : u * v * w = 1)
variable (HA : A = u * v + u + 1)
variable (HB : B = v * w + v + 1)
variable (HC : C = w * u + w + 1)

theorem abc_eq_ab_bc_ca 
  (Huvw : u * v * w = 1)
  (HA : A = u * v + u + 1)
  (HB : B = v * w + v + 1)
  (HC : C = w * u + w + 1) : 
  A * B * C = A * B + B * C + C * A := 
by
  sorry

end abc_eq_ab_bc_ca_l784_784791


namespace hyperbola_equation_standard_form_l784_784688

noncomputable def point_on_hyperbola_asymptote (A : ℝ × ℝ) (C : ℝ) : Prop :=
  let x := A.1
  let y := A.2
  (4 * y^2 - x^2 = C) ∧
  (y = (1/2) * x ∨ y = -(1/2) * x)

theorem hyperbola_equation_standard_form
  (A : ℝ × ℝ)
  (hA : A = (2 * Real.sqrt 2, 2))
  (asymptote1 asymptote2 : ℝ → ℝ)
  (hasymptote1 : ∀ x, asymptote1 x = (1/2) * x)
  (hasymptote2 : ∀ x, asymptote2 x = -(1/2) * x) :
  (∃ C : ℝ, point_on_hyperbola_asymptote A C) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (4 * (A.2)^2 - (A.1)^2 = 8) ∧ 
    (∀ x y : ℝ, (4 * y^2 - x^2 = 8) ↔ ((y^2) / a - (x^2) / b = 1))) :=
by
  sorry

end hyperbola_equation_standard_form_l784_784688


namespace andrew_age_l784_784525

theorem andrew_age (a g : ℝ) (h1 : g = 9 * a) (h2 : g - a = 63) : a = 7.875 :=
by
  sorry

end andrew_age_l784_784525


namespace modulus_of_z_l784_784583

-- Given condition
def satisfies_equation (z : ℂ) : Prop :=
  (z - 1) * complex.I = complex.I - 1

-- The theorem to prove
theorem modulus_of_z (z : ℂ) (h : satisfies_equation z) : complex.abs z = real.sqrt 5 :=
sorry

end modulus_of_z_l784_784583


namespace algebraic_expression_evaluation_l784_784250

theorem algebraic_expression_evaluation (a b c : ℝ) 
  (h1 : a^2 + b * c = 14) 
  (h2 : b^2 - 2 * b * c = -6) : 
  3 * a^2 + 4 * b^2 - 5 * b * c = 18 :=
by 
  sorry

end algebraic_expression_evaluation_l784_784250


namespace remainder_when_divided_by_11_l784_784189

theorem remainder_when_divided_by_11 :
  (7 * 7^10 + 1^10) % 11 = 8 :=
by
  have h1: 7 * 7^10 = 7^11 := by sorry
  have h2: 7^10 % 11 = 1 := by sorry
  have h3: 1^10 = 1 := by sorry
  calc
    (7 * 7^10 + 1^10) % 11
        = (7^11 + 1) % 11   : by rw [h1, h3]
    ... = (7 + 1) % 11      : by rw [h2]
    ... = 8                 : by norm_num

end remainder_when_divided_by_11_l784_784189


namespace unique_number_not_in_range_l784_784837

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_number_not_in_range (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h_g_17 : g p q r s 17 = 17) (h_g_89 : g p q r s 89 = 89) (h_involution : ∀ x : ℝ, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! (unique_irrational : ℝ), unique_irrational = 53 ∧ ¬∃ y : ℝ, g p q r s y = unique_irrational :=
begin
  sorry
end

end unique_number_not_in_range_l784_784837


namespace trigonometric_identity_l784_784578

theorem trigonometric_identity 
  (θ : ℝ) 
  (h1 : tan θ = -2) 
  (h2 : -π / 2 < θ ∧ θ < 0) :
  sin θ ^ 2 / (cos (2 * θ) + 2) = 4 / 7 :=
by
  sorry

end trigonometric_identity_l784_784578


namespace pq_fraction_of_ae_and_parallel_l784_784385

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def length (P Q : Point) : ℝ := sorry
noncomputable def parallel (A B C D : Point) : Prop := sorry

theorem pq_fraction_of_ae_and_parallel {A B C D E M K N L P Q : Point}
  (hM : M = midpoint A B) 
  (hK : K = midpoint B C) 
  (hN : N = midpoint C D) 
  (hL : L = midpoint D E)
  (hP : P = midpoint M N)
  (hQ : Q = midpoint K L) :
  length P Q = (1 / 4) * length A E ∧ parallel P Q A E := 
sorry

end pq_fraction_of_ae_and_parallel_l784_784385


namespace num_values_satisfying_eq_l784_784236

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem num_values_satisfying_eq {n : ℕ} (h_pos : n > 0) :
  let S := sum_of_digits in
  (∀ n, n + S n + S (S n) = 2023) →
  (∃ (ns : List ℕ), ns.length = 3 ∧ ∀ n ∈ ns, n + S n + S (S n) = 2023) :=
begin
  sorry
end

end num_values_satisfying_eq_l784_784236


namespace basketballs_count_l784_784973

theorem basketballs_count (x : ℕ) : 
  let num_volleyballs := x
  let num_basketballs := 2 * x
  let num_soccer_balls := x - 8
  num_volleyballs + num_basketballs + num_soccer_balls = 100 →
  num_basketballs = 54 :=
by
  intros h
  sorry

end basketballs_count_l784_784973


namespace distance_to_other_focus_is_5_l784_784263

theorem distance_to_other_focus_is_5 (P : ℝ × ℝ)
  (hP : (P.1^2 / 16) + (P.2^2 / 4) = 1)
  (h_focus_distance : real.sqrt (16 - 4) = 3) :
  ∃ d : ℝ, d = 5 :=
sorry

end distance_to_other_focus_is_5_l784_784263


namespace phoneExpences_l784_784574

structure PhonePlan where
  fixed_fee : ℝ
  free_minutes : ℕ
  excess_rate : ℝ -- rate per minute

def JanuaryUsage : ℕ := 15 * 60 + 17 -- 15 hours 17 minutes in minutes
def FebruaryUsage : ℕ := 9 * 60 + 55 -- 9 hours 55 minutes in minutes

def computeBill (plan : PhonePlan) (usage : ℕ) : ℝ :=
  let excess_minutes := (usage - plan.free_minutes).max 0
  plan.fixed_fee + (excess_minutes * plan.excess_rate)

theorem phoneExpences (plan : PhonePlan) :
  plan = { fixed_fee := 18.00, free_minutes := 600, excess_rate := 0.03 } →
  computeBill plan JanuaryUsage + computeBill plan FebruaryUsage = 45.51 := by
  sorry

end phoneExpences_l784_784574


namespace base7_perfect_square_values_l784_784510

theorem base7_perfect_square_values (a b c : ℕ) (h1 : a ≠ 0) (h2 : b < 7) :
  ∃ (n : ℕ), (343 * a + 49 * c + 28 + b = n * n) → (b = 0 ∨ b = 1 ∨ b = 4) :=
by
  sorry

end base7_perfect_square_values_l784_784510


namespace train_length_l784_784518

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

theorem train_length
  (train_speed_kmph : ℝ)
  (man_speed_kmph : ℝ)
  (opposite_directions : Bool)
  (passing_time_seconds : ℝ)
  (expected_length : ℝ) :
  let train_speed_mps := kmph_to_mps train_speed_kmph
      man_speed_mps := kmph_to_mps man_speed_kmph
  in opposite_directions = true →
     let relative_speed := train_speed_mps + man_speed_mps
     in expected_length = relative_speed * passing_time_seconds :=
by
  sorry

-- Conditions instantiated:
#eval train_length 60 6 true 6 110.04

end train_length_l784_784518


namespace circle_Γ_contains_exactly_one_l784_784006

-- Condition definitions
variables (z1 z2 : ℂ) (Γ : ℂ → ℂ → Prop)
variable (hz1z2 : z1 * z2 = 1)
variable (hΓ_passes : Γ (-1) 1)
variable (hΓ_not_passes : ¬Γ z1 z2)

-- Math proof problem
theorem circle_Γ_contains_exactly_one (hz1z2 : z1 * z2 = 1)
    (hΓ_passes : Γ (-1) 1) (hΓ_not_passes : ¬Γ z1 z2) : 
  (Γ 0 z1 ↔ ¬Γ 0 z2) ∨ (Γ 0 z2 ↔ ¬Γ 0 z1) :=
sorry

end circle_Γ_contains_exactly_one_l784_784006


namespace solve_for_x_l784_784729

theorem solve_for_x (x : ℝ) (h : real.sqrt (5 + real.sqrt x) = 4) : x = 121 := 
sorry

end solve_for_x_l784_784729


namespace number_is_45_percent_of_27_l784_784442

theorem number_is_45_percent_of_27 (x : ℝ) (h : 27 / x = 45 / 100) : x = 60 := 
by
  sorry

end number_is_45_percent_of_27_l784_784442


namespace triangle_area_l784_784796

theorem triangle_area (
  {A B C D E : Type} 
  (hAD : D ∈ segment A C) 
  (hBE: E ∈ segment B C) 
  (hmedians: median A B D ∧ median B A E)
  (angle_between_medians : ∠(vector AD) (vector BE) = π / 4)
  (AD_length : dist A D = 12)
  (BE_length : dist B E = 16)) :
  ∃ (area : ℝ), area = 64 * (sqrt 2) :=
sorry

end triangle_area_l784_784796


namespace line_intersects_circle_l784_784848

theorem line_intersects_circle (a : ℝ) : 
  let line := λ x y : ℝ, (x - 1) * a + y = 1,
      center := (0, 0),
      radius := real.sqrt 3,
      distance := real.sqrt ((1 - 0)^2 + (1 - 0)^2)
  in distance < radius :=
begin
  sorry
end

end line_intersects_circle_l784_784848


namespace correct_function_is_f1_l784_784965

noncomputable def f1 (x : ℝ) : ℝ := 1 / x
noncomputable def f2 (x : ℝ) : ℝ := x^-1^2
noncomputable def f3 (x : ℝ) : ℝ := Real.exp x
noncomputable def f4 (x : ℝ) : ℝ := Real.log (x + 1)

theorem correct_function_is_f1 :
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f1 x1 > f1 x2) ∧
  ¬ (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f2 x1 > f2 x2) ∧
  ¬ (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f3 x1 > f3 x2) ∧
  ¬ (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f4 x1 > f4 x2) :=
by
  sorry

end correct_function_is_f1_l784_784965


namespace charles_skittles_left_l784_784990

/-- 
Given that Charles initially has 250 Skittles and Diana takes away 17.5% of his Skittles, 
Charles will have 206 Skittles left. 
-/
theorem charles_skittles_left (initial_skittles : ℕ) (percentage_taken : ℚ) (final_skittles : ℕ) :
  initial_skittles = 250 → percentage_taken = 17.5 → final_skittles = 206 → 
  final_skittles = initial_skittles - (percentage_taken / 100 * initial_skittles).natAbs := by
  sorry

end charles_skittles_left_l784_784990


namespace remainder_sum_mod_14_l784_784737

theorem remainder_sum_mod_14 
  (a b c : ℕ) 
  (ha : a % 14 = 5) 
  (hb : b % 14 = 5) 
  (hc : c % 14 = 5) :
  (a + b + c) % 14 = 1 := 
by
  sorry

end remainder_sum_mod_14_l784_784737


namespace matrix_not_invertible_l784_784480

theorem matrix_not_invertible (x : ℝ) :
  let M := ![![2 + x, 5], ![4 - x, 6]]
  det M = 0 ↔ x = 8 / 11 :=
by
  let M := ![![2 + x, 5], ![4 - x, 6]]
  calc
    det M = (2 + x) * 6 - 5 * (4 - x) : by sorry
      ...  = 11 * x - 8 : by sorry
  sorry

end matrix_not_invertible_l784_784480


namespace min_output_to_avoid_losses_l784_784105

theorem min_output_to_avoid_losses (x : ℝ) (y : ℝ) (h : y = 0.1 * x - 150) : y ≥ 0 → x ≥ 1500 :=
sorry

end min_output_to_avoid_losses_l784_784105


namespace soap_duration_l784_784520

theorem soap_duration (l w h : ℝ) (h1 : l > 0) (h2 : w > 0) (h3 : h > 0) :
  let V_initial := l * w * h in
  let V_new := (l / 2) * (w / 2) * (h / 2) in
  let usage_per_hour := (V_initial - V_new) / 7 in
  let remaining_volume := V_new in
  remaining_volume / usage_per_hour = 1 := 
by {
  -- Placeholder for proof steps
  sorry
}

end soap_duration_l784_784520


namespace committee_selection_correct_l784_784753

def num_ways_to_choose_committee : ℕ :=
  let total_people := 10
  let president_ways := total_people
  let vp_ways := total_people - 1
  let remaining_people := total_people - 2
  let committee_ways := Nat.choose remaining_people 2
  president_ways * vp_ways * committee_ways

theorem committee_selection_correct :
  num_ways_to_choose_committee = 2520 :=
by
  sorry

end committee_selection_correct_l784_784753


namespace num_positive_area_triangles_l784_784722

/--
How many triangles with positive area can be formed whose vertices are points in the xy-plane, 
where the vertices have integer coordinates (x, y) satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5?
-/
theorem num_positive_area_triangles :
  let points := { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 },
      total_points := 5 * 5,
      -- Number of points in the grid
      n := total_points,
      total_combinations := nat.choose n 3,
      -- Degenerate combinations
      degenerate_rows_columns := 10 * nat.choose 5 3,
      degenerate_main_diagonals := 2 * nat.choose 5 3,
      degenerate_other_diagonals := 4 * nat.choose 4 3,
      total_degenerate := degenerate_rows_columns + degenerate_main_diagonals + degenerate_other_diagonals,
      valid_triangles := total_combinations - total_degenerate
  in valid_triangles = 2164 :=
by
  sorry

end num_positive_area_triangles_l784_784722


namespace complex_div_eq_l784_784577

theorem complex_div_eq (a b : ℝ) (h : (1 + Complex.i) / (1 - Complex.i) = a + b * Complex.i) : a * b = 0 := 
sorry

end complex_div_eq_l784_784577


namespace evening_water_usage_is_6_l784_784932

-- Define the conditions: daily water usage and total water usage over 5 days.
def daily_water_usage (E : ℕ) : ℕ := 4 + E
def total_water_usage (E : ℕ) (days : ℕ) : ℕ := days * daily_water_usage E

-- Define the condition that over 5 days the total water usage is 50 liters.
axiom water_usage_condition : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6

-- Conjecture stating the amount of water used in the evening.
theorem evening_water_usage_is_6 : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6 :=
by
  intro E
  intro h
  exact water_usage_condition E h

end evening_water_usage_is_6_l784_784932


namespace A_and_B_finish_together_in_20_days_l784_784155

noncomputable def W_B : ℝ := 1 / 30

noncomputable def W_A : ℝ := 1 / 2 * W_B

noncomputable def W_A_plus_B : ℝ := W_A + W_B

theorem A_and_B_finish_together_in_20_days :
  (1 / W_A_plus_B) = 20 :=
by
  sorry

end A_and_B_finish_together_in_20_days_l784_784155


namespace sequence_k_value_l784_784338

theorem sequence_k_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ m n : ℕ, a (m + n) = a m * a n)
  (hk1 : ∀ k : ℕ, a (k + 1) = 1024) :
  ∃ k : ℕ, k = 9 :=
by {
  sorry
}

end sequence_k_value_l784_784338


namespace number_is_prime_or_power_of_two_l784_784026

theorem number_is_prime_or_power_of_two (n : ℕ) (Hn_gt_6 : n > 6) 
  (a : { k // ∀ i, i < k → natural_numbers_lt_rel_prime n(i)} 
  (H_arith_prog : is_arithmetic_progression a k) : 
  is_prime n ∨ ∃ m : ℕ, n = 2^m :=
sorry

end number_is_prime_or_power_of_two_l784_784026


namespace find_number_l784_784130

theorem find_number (N : ℝ) (h : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N) : N = 180 :=
by 
  sorry

end find_number_l784_784130


namespace construct_triangle_ABC_l784_784835

theorem construct_triangle_ABC
  {A B C A1 B1 C1 A2 B2 C2 H : Type}
  (h_A1 : is_foot_of_altitude A B C A1)
  (h_B1 : is_foot_of_altitude B C A B1)
  (h_C1 : is_foot_of_altitude C A B C1)
  (h_A2 : orthocenter A B1 C1 A2)
  (h_B2 : orthocenter B C1 A1 B2)
  (h_C2 : orthocenter C A1 B1 C2) :
  ∃ (H : Type), is_orthocenter H A B C :=
begin
  sorry
end

end construct_triangle_ABC_l784_784835


namespace number_of_elements_in_list_l784_784200

theorem number_of_elements_in_list :
  let a := 2.5
  let d := 5.0
  let l := 62.5
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 13 :=
begin
  sorry
end

end number_of_elements_in_list_l784_784200


namespace intersection_M_N_is_correct_l784_784617

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784617


namespace m_plus_n_l784_784357

structure Ellipse :=
  (a b : ℝ)
  (c : ℝ := real.sqrt (a^2 - b^2))

structure Point :=
  (x y : ℝ)

def ellipse := Ellipse.mk 13 12
def R := Point.mk (-5) 0
def S := Point.mk 5 0
def A := Point.mk 0 12
def B := Point.mk 0 (-12)
def P := Point.mk 10 24

def distance (P Q : Point) : ℝ :=
  real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)

def perimeter (R P S Q : Point) : ℝ :=
  distance R P + distance P S + distance S Q + distance Q R

theorem m_plus_n :
  let m := 26 in
  let n := 601 in
  m + n = 627 := by
  sorry

end m_plus_n_l784_784357


namespace vectors_are_parallel_l784_784901

-- Define vector equality
def vector_eq (a b c d : ℝ → ℝ) := ∀ t, (∀ x, a t x = b t x) = (∀ x, c t x = d t x)

-- Define the condition for opposite vectors
def opposite_vectors (A B : ℝ × ℝ) := vector_eq (λ t, A.1 - B.1) (λ t, A.2 - B.2) (λ t, B.1 - A.1) (λ t, B.2 - A.2)

-- The proof statement
theorem vectors_are_parallel (A B : ℝ × ℝ) : opposite_vectors A B → (A.1 - B.1 = B.1 - A.1 ∧ A.2 - B.2 = B.2 - A.2) := by
  sorry

end vectors_are_parallel_l784_784901


namespace number_of_valid_sequences_l784_784721

noncomputable def num_valid_sequences : ℕ := 5 ^ 7

theorem number_of_valid_sequences :
  ∃ xs : vector ℕ 7, 
  (∀ i : fin 6, (xs.nth i % 2) ≠ (xs.nth (i + 1) % 2)) ∧ 
  (finset.univ.filter (λ i, xs.nth i % 2 = 0)).card ≥ 4 ∧
  (xs.val.all (λ x, x < 10)) ∧
  (xs.nth 0 % 2 = 0) → 
  num_valid_sequences = 78125 :=
sorry

end number_of_valid_sequences_l784_784721


namespace range_of_angle_of_inclination_l784_784407

theorem range_of_angle_of_inclination (α : ℝ) :
  ∃ θ : ℝ, θ ∈ (Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi) ∧
           ∀ x : ℝ, ∃ y : ℝ, y = x * Real.sin α + 1 := by
  sorry

end range_of_angle_of_inclination_l784_784407


namespace probability_of_intersecting_diagonals_l784_784447

noncomputable def intersecting_diagonals_probability : ℚ :=
let total_vertices := 8 in
let total_pairs := Nat.choose total_vertices 2 in
let total_sides := 8 in
let total_diagonals := total_pairs - total_sides in
let total_pairs_diagonals := Nat.choose total_diagonals 2 in
let intersecting_diagonals := Nat.choose total_vertices 4 in
(intersecting_diagonals : ℚ) / (total_pairs_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  intersecting_diagonals_probability = 7 / 19 :=
by
  sorry

end probability_of_intersecting_diagonals_l784_784447


namespace disjoint_subsets_with_same_sum_l784_784890

theorem disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), S.card = 10 ∧ (∀ x ∈ S, x ∈ Finset.range 101) →
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_with_same_sum_l784_784890


namespace no_mass_infection_event_not_definite_l784_784551

-- Definitions as per conditions

def median (xs : List ℕ) : ℕ := xs.sort.drop (xs.length / 2) |>.headI

def mode (xs : List ℕ) : ℕ :=
(x : ℕ in xs).groupBy id |>.maxBy λ g, xs.count g.head |>.head

def no_mass_infection (xs : List ℕ) : Prop :=
∀ x ∈ xs, x ≤ 7

-- Problem statement
theorem no_mass_infection_event_not_definite
  (xs : List ℕ)
  (h_len : xs.length = 10)
  (h_median : median xs = 2)
  (h_mode : mode xs = 3) :
  ¬ no_mass_infection xs :=
sorry

end no_mass_infection_event_not_definite_l784_784551


namespace intersection_M_N_l784_784611

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784611


namespace imperative_sentence_structure_l784_784138

theorem imperative_sentence_structure (word : String) (is_base_form : word = "Surround") :
  (word = "Surround" ∨ word = "Surrounding" ∨ word = "Surrounded" ∨ word = "Have surrounded") →
  (∃ sentence : String, sentence = word ++ " yourself with positive people, and you will keep focused on what you can do instead of what you can’t.") →
  word = "Surround" :=
by
  intros H_choice H_sentence
  cases H_choice
  case inl H1 => assumption
  case inr H2_1 =>
    cases H2_1
    case inl H2_1_1 => sorry
    case inr H2_1_2 =>
      cases H2_1_2
      case inl H2_1_2_1 => sorry
      case inr H2_1_2_2 => sorry

end imperative_sentence_structure_l784_784138


namespace line_equation_parallel_l784_784692

theorem line_equation_parallel (a b : ℝ) (h_parallel : a = 2) (h_intercept : b = 3) : 
  ∃ f : ℝ → ℝ, (∀ x, f x = a * x + b) ∧ (∀ x, f x = 2 * x + 3) :=
by
  use (λ x, a * x + b)
  split
  { intro x
    exact rfl }
  { intro x
    rw [h_parallel, h_intercept]
    exact rfl }

end line_equation_parallel_l784_784692


namespace fifteen_percent_of_x_equals_sixty_l784_784556

theorem fifteen_percent_of_x_equals_sixty (x : ℝ) (h : 0.15 * x = 60) : x = 400 :=
by
  sorry

end fifteen_percent_of_x_equals_sixty_l784_784556


namespace distance_between_skew_lines_l784_784323

-- Define the regular tetrahedron with vertices P, A, B, C, D
-- where P-ABCD has each face an equilateral triangle with side length 1

noncomputable def regular_tetrahedron := sorry

structure Point3D :=
(x : ℚ)
(y : ℚ)
(z : ℚ)

-- Assume specific points for vertices, these can be defined more rigorously in full proof.
def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨1, 0, 0⟩
def C : Point3D := ⟨0.5, sqrt (3.0/4.0), 0⟩
def P : Point3D := ⟨0.5, sqrt (3.0/4.0) / 2.0, sqrt(2.0/3.0)⟩

-- M and N are midpoints of AB and BC respectively
def M : Point3D := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2⟩
def N : Point3D := ⟨(B.x + C.x) / 2, (B.y + C.y) / 2, (B.z + C.z) / 2⟩

-- Define the proof statement
theorem distance_between_skew_lines:
  let distance := (fun (M N P : Point3D) => sqrt(2.0)/4.0) in
  distance M N P = Real.sqrt 2 / 4 :=
begin
  sorry
end

end distance_between_skew_lines_l784_784323


namespace problem_l784_784154

noncomputable def f : ℝ → ℝ := sorry

theorem problem (h1 : ∀ x : ℝ, f(-x) = -f(x))
    (h2 : ∀ x : ℝ, f(x) + f(4 - x) = 0)
    (h3 : f(1) = 8) :
    f(2010) + f(2011) + f(2012) = -8 := 
sorry

end problem_l784_784154


namespace area_difference_l784_784724

theorem area_difference (radius1 radius2 : ℝ) (pi : ℝ) (h1 : radius1 = 15) (h2 : radius2 = 14 / 2) :
  pi * radius1 ^ 2 - pi * radius2 ^ 2 = 176 * pi :=
by 
  sorry

end area_difference_l784_784724


namespace diane_15_cents_arrangement_l784_784206

def stamps : List (ℕ × ℕ) := 
  [(1, 1), 
   (2, 2), 
   (3, 3), 
   (4, 4), 
   (5, 5), 
   (6, 6), 
   (7, 7), 
   (8, 8), 
   (9, 9), 
   (10, 10), 
   (11, 11), 
   (12, 12)]

def number_of_arrangements (value : ℕ) (stamps : List (ℕ × ℕ)) : ℕ := sorry

theorem diane_15_cents_arrangement : number_of_arrangements 15 stamps = 32 := 
sorry

end diane_15_cents_arrangement_l784_784206


namespace binomial_sum_expression_l784_784050

def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem binomial_sum_expression (m n : ℕ) (h : n ∈ ℕ) :
  ((finset.range (n + 1)).sum (λ k, (-1 : ℤ)^k * (binomial n k : ℤ) / (m + k + 1))) 
  = 1 / ((m + n + 1) * (binomial (m + n) n : ℤ)) :=
sorry

end binomial_sum_expression_l784_784050


namespace general_formula_sum_b_terms_l784_784681

noncomputable def a_sequence : ℕ → ℕ
| n := n

theorem general_formula (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (a1 : a 1 = 1) 
  (cond1 : a 5 - a 3 = 2 ∨ a 2 * a 3 = 6 ∨ S 5 = 15) :
  ∀ n, a n = n := sorry

noncomputable def b_sequence : ℕ → ℕ
| n := 2 * n + 2^n

theorem sum_b_terms (a : ℕ → ℕ) 
  (b : ℕ → ℕ) 
  (T : ℕ → ℕ)
  (a_n_general : ∀ n, a n = n) :
  ∀ n, T n = n^2 + n + 2^(n+1) - 2 := sorry

end general_formula_sum_b_terms_l784_784681


namespace walking_speed_l784_784941

theorem walking_speed 
  (v : ℕ) -- v represents the man's walking speed in kmph
  (distance_formula : distance = speed * time)
  (distance_walking : distance = v * 9)
  (distance_running : distance = 24 * 3) : 
  v = 8 :=
by
  sorry

end walking_speed_l784_784941


namespace smallest_value_of_n_l784_784961

theorem smallest_value_of_n : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n + 6) % 7 = 0 ∧ (n - 9) % 4 = 0 ∧ n = 113 :=
by
  sorry

end smallest_value_of_n_l784_784961


namespace rate_in_still_water_l784_784940

theorem rate_in_still_water (with_stream_speed against_stream_speed : ℕ) 
  (h₁ : with_stream_speed = 16) 
  (h₂ : against_stream_speed = 12) : 
  (with_stream_speed + against_stream_speed) / 2 = 14 := 
by
  sorry

end rate_in_still_water_l784_784940


namespace ferris_wheel_seats_l784_784824

theorem ferris_wheel_seats (S : ℕ) (h1 : ∀ (p : ℕ), p = 9) (h2 : ∀ (r : ℕ), r = 18) (h3 : 9 * S = 18) : S = 2 :=
by
  sorry

end ferris_wheel_seats_l784_784824


namespace number_of_strawberry_cakes_l784_784958

def number_of_chocolate_cakes := 3
def price_of_chocolate_cake := 12
def price_of_strawberry_cake := 22
def total_payment := 168

theorem number_of_strawberry_cakes (S : ℕ) : 
    number_of_chocolate_cakes * price_of_chocolate_cake + S * price_of_strawberry_cake = total_payment → 
    S = 6 :=
by
  sorry

end number_of_strawberry_cakes_l784_784958


namespace maximum_value_of_m_plus_n_l784_784386

open Classical

noncomputable def proof_problem :=
  let x y z m n : ℚ in
  let x y z : ℚ := if h : (x + y + z = 1 ∧ x < y ∧ y < z ∧ (x^2 + y^2 + z^2 - 1)^3 + 8 * x * y * z = 0 ∧ ∃ (m n : ℕ), m + n < 1000 ∧ Nat.gcd m n = 1 ∧ z = (m/n)^2) then
    (x, y, z)
  else
    0 in
  m + n = 536

theorem maximum_value_of_m_plus_n : proof_problem := sorry

end maximum_value_of_m_plus_n_l784_784386


namespace probability_tile_is_blue_l784_784150

theorem probability_tile_is_blue :
  let tiles := {x ∈ Finset.range 1 (101) | x % 7 = 3}
  in P(tiles) = 7 / 50 := sorry

end probability_tile_is_blue_l784_784150


namespace problem_solution_l784_784524

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := p > 0 ∧ ∀ x, f (x + p) = f x

def problem : Prop :=
  (is_even_function (λ x => |sin x|) ∧ has_period (λ x => |sin x|) π) ∧
  ∀ f : ℝ → ℝ, 
    (is_even_function f ∧ has_period f π →
      (f = (λ x => |sin x|) ∨ f ≠ sin ∧ f ≠ (λ x => |tan x|) ∧ f ≠ cos ∧ f ≠ λ x => sin x))

theorem problem_solution : problem :=
sorry

end problem_solution_l784_784524


namespace age_of_b_l784_784469

variable (a b c : ℕ)

-- Conditions as definitions
def cond1 := a = b + 2
def cond2 := b = 2 * c
def cond3 := a + b + c = 17

-- The theorem to prove the age of b
theorem age_of_b (h1 : cond1 a b c) (h2 : cond2 b c) (h3 : cond3 a b c) : b = 6 :=
by {
  sorry
}

end age_of_b_l784_784469


namespace paths_and_sums_l784_784991

theorem paths_and_sums :
  ∃ (rolls : Fin 5 → Fin 6), 
  let moves := Finset.univ.map (λ i => match rolls i with
    | 0 => (-1, 0) -- Roll a 1: 1 km west
    | 1 => (1, 0)  -- Roll a 2: 1 km east
    | 2 => (0, 1)  -- Roll a 3: 1 km north
    | 3 => (0, -1) -- Roll a 4: 1 km south
    | 4 => (0, 0)  -- Roll a 5: no movement
    | 5 => (0, 3)  -- Roll a 6: 3 km north
    | _ => (0, 0)  -- This case is unreachable
  end),
  let position := moves.foldl (λ pos move => (pos.1 + move.1, pos.2 + move.2)) (0, 0),
  position.1 = 1 ∧ position.2 = 0 ∧ 
  let sum := Finset.univ.sum (λ i => rolls i.val + 1),
  (sum = 17 ∨ sum = 19 ∨ sum = 14) :=
sorry

end paths_and_sums_l784_784991


namespace product_equals_fraction_l784_784999

def modified_fib : ℕ → ℕ
| 1 := 2
| 2 := 1
| (n+1) := modified_fib n + modified_fib (n - 1)

theorem product_equals_fraction :
  (∏ k in (finset.range 99).filter (λ x, x ≥ 3), 
    (modified_fib k) / (modified_fib (k - 1)) - (modified_fib k) / (modified_fib (k + 1)))
  = (modified_fib 101) / (modified_fib 102) :=
by sorry

end product_equals_fraction_l784_784999


namespace common_divisors_84_90_l784_784716

theorem common_divisors_84_90 : (finset.filter (λ x, 84 ∣ x ∧ 90 ∣ x) (finset.range 85)).card * 2 = 8 := 
by
  sorry

end common_divisors_84_90_l784_784716


namespace part1_part2_l784_784046

variable (a : ℝ)

-- Proposition A
def propA (a : ℝ) := ∀ x : ℝ, ¬ (x^2 + (2*a-1)*x + a^2 ≤ 0)

-- Proposition B
def propB (a : ℝ) := 0 < a^2 - 1 ∧ a^2 - 1 < 1

theorem part1 (ha : propA a ∨ propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) ∨ (a > 1/4) :=
  sorry

theorem part2 (ha : ¬ propA a) (hb : propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) → (a^3 + 1 < a^2 + a) :=
  sorry

end part1_part2_l784_784046


namespace cube_root_of_neg8_l784_784829

-- Define the condition
def is_cube_root (x : ℝ) : Prop := x^3 = -8

-- State the problem to be proved.
theorem cube_root_of_neg8 : is_cube_root (-2) :=
by 
  sorry

end cube_root_of_neg8_l784_784829


namespace algebraic_expression_value_l784_784256

theorem algebraic_expression_value (x y : ℝ) (h : x^2 - 4 * x - 1 = 0) : 
  (2 * x - 3) ^ 2 - (x + y) * (x - y) - y ^ 2 = 12 := 
by {
  sorry
}

end algebraic_expression_value_l784_784256


namespace a7_of_arithmetic_seq_l784_784004

-- Defining the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

theorem a7_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : arithmetic_seq a d) 
  (h_a4 : a 4 = 5) 
  (h_a5_a6 : a 5 + a 6 = 11) : 
  a 7 = 6 :=
by
  sorry

end a7_of_arithmetic_seq_l784_784004


namespace intersection_M_N_l784_784659

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784659


namespace intersection_M_N_l784_784607

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784607


namespace arithmetic_sequence_mod_15_l784_784302

theorem arithmetic_sequence_mod_15 : 
  ∃ m, (∀ n : ℕ, (n > 0 ∧ n <= 21) → 5 * n - 3 < 15) ∧ m = (∑ i in finrange 21, (5 * i - 3) % 15) ∧ 0 ≤ m ∧ m < 15 ∧ m = 12 :=
begin
  -- the steps of the proof go here
  sorry
end

end arithmetic_sequence_mod_15_l784_784302


namespace exists_strictly_increasing_sequence_l784_784387

theorem exists_strictly_increasing_sequence (a1 : ℕ) (h1 : a1 > 1) :
  ∃ (a : ℕ → ℕ), (strict_mono a) ∧ (a 0 = a1) ∧ ∀ k, k ≥ 0 → (∑ i in Finset.range (k + 1), (a i) ^ 2) ∣ (∑ i in Finset.range (k + 1), a i) :=
sorry

end exists_strictly_increasing_sequence_l784_784387


namespace bridget_apples_l784_784185

theorem bridget_apples : 
  ∃ (x : ℕ), 
    let a := x / 3 in
    let c := 5 in
    let s := 2 in
    let y := x - a - c - s in
    y = 8 → x = 30 := 
by
  sorry

end bridget_apples_l784_784185


namespace probability_is_correct_l784_784087

namespace KeyProbability

variables (totalKeys : ℕ) (openableKeys : ℕ)
def probability_select_key_opening_door (totalKeys : ℕ) (openableKeys : ℕ) : ℚ :=
  openableKeys / totalKeys

theorem probability_is_correct
  (h_totalKeys : totalKeys = 5)
  (h_openableKeys : openableKeys = 2) :
  probability_select_key_opening_door totalKeys openableKeys = 2 / 5 :=
by
  unfold probability_select_key_opening_door
  rw [h_totalKeys, h_openableKeys]
  norm_num
  sorry
  
end KeyProbability

end probability_is_correct_l784_784087


namespace intersection_M_N_l784_784610

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784610


namespace find_f_neg_2_l784_784584

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * (1 - x) else (2 - x) * (1 - (2 - x))

theorem find_f_neg_2 : f(-2) = -12 := by
  -- Proof will be filled here
  sorry

end find_f_neg_2_l784_784584


namespace angle_between_Simson_lines_eq_inscribed_angle_l784_784048

theorem angle_between_Simson_lines_eq_inscribed_angle
  (ABC : Type) [triangle ABC] 
  (P1 P2 : point)
  (circumcircle : circle ABC)
  (s1 s2 : Simson_line ABC P1 s1) (simon2 : Simson_line ABC P2 s2)  :
  (P1 ∈ circumcircle) ∧ (P2 ∈ circumcircle) →
  (angle_between_Simson_lines P1 P2 s1 s2 = inscribed_angle P1 P2) :=
by
  sorry

end angle_between_Simson_lines_eq_inscribed_angle_l784_784048


namespace arithmetic_seq_value_zero_l784_784750

theorem arithmetic_seq_value_zero (a b c : ℝ) (a_seq : ℕ → ℝ)
    (l m n : ℕ) (h_arith : ∀ k, a_seq (k + 1) - a_seq k = a_seq 1 - a_seq 0)
    (h_l : a_seq l = 1 / a)
    (h_m : a_seq m = 1 / b)
    (h_n : a_seq n = 1 / c) :
    (l - m) * a * b + (m - n) * b * c + (n - l) * c * a = 0 := 
sorry

end arithmetic_seq_value_zero_l784_784750


namespace solve_factorial_equation_l784_784075

theorem solve_factorial_equation : 
  let solutions := [(n, k, l) | n k l : ℕ, n > 1, (∀ k l, k < n ∧ l < n ∧ 2 * (k! + l!) = n! ∧ k! + l! > 1)] in
  if solutions = [] then 0
  else (solutions.map (λ (n, k, l) => n)).eraseDups.sum = 10 := 
sorry

end solve_factorial_equation_l784_784075


namespace cube_long_diagonal_length_l784_784859

noncomputable def cube_side_length_satisfying_conditions : Real :=
  let s : Real := 6
  if h : s^3 + 36 * s = 12 * s^2 then
    s
  else
    0 -- this should not occur since s=6 satisfies the conditions

theorem cube_long_diagonal_length :
  let s : Real := cube_side_length_satisfying_conditions
  s ≠ 0 → (s * Real.sqrt 3) = 6 * Real.sqrt 3 :=
by
  intro s_ne_zero
  let s : Real := 6
  have : s = 6 := rfl
  calc
    (s * Real.sqrt 3)
      = 6 * Real.sqrt 3 : by rw [this]
  sorry

end cube_long_diagonal_length_l784_784859


namespace particle_distribution_configurations_l784_784869

theorem particle_distribution_configurations :
  ∀ (particles wells: ℕ) (total_energy : ℕ) (follows_fermi_dirac : Bool) (energy_levels : ℕ → ℕ) (indistinguishable_wells : Bool),
  particles = 3 →
  wells = 2 →
  total_energy = 3 →
  follows_fermi_dirac = true →
  (∀ k, 0 ≤ k ∧ k ≤ 3 → (energy_levels k) = k * \(E_0\)) →
  indistinguishable_wells = true →
  (∃ n : ℕ, n = 468) :=
by
  sorry

end particle_distribution_configurations_l784_784869


namespace sum_series_equals_a_over_b_fact_minus_c_l784_784885

theorem sum_series_equals_a_over_b_fact_minus_c :
  (\sum k in Finset.range 50, (-1)^(k+1) * (k+1)^3 + (k+1)^2 + (k+1) + 1) / (k+1)! = 2551 / 49! - 1 → 
  2551 + 49 + 1 = 2601 := sorry

end sum_series_equals_a_over_b_fact_minus_c_l784_784885


namespace probability_at_least_two_defective_probability_at_most_one_defective_l784_784899

variable (P_no_defective : ℝ)
variable (P_one_defective : ℝ)
variable (P_two_defective : ℝ)
variable (P_all_defective : ℝ)

theorem probability_at_least_two_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_two_defective + P_all_defective = 0.29 :=
  by sorry

theorem probability_at_most_one_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_no_defective + P_one_defective = 0.71 :=
  by sorry

end probability_at_least_two_defective_probability_at_most_one_defective_l784_784899


namespace limit_eq_minus_one_l784_784305

theorem limit_eq_minus_one (f : ℝ → ℝ) (x₀ : ℝ) (h : deriv f x₀ = 2) :
  tendsto (fun k : ℝ => (f (x₀ - k) - f x₀) / (2 * k)) (𝓝 0) (𝓝 (-1)) :=
sorry

end limit_eq_minus_one_l784_784305


namespace max_cyclic_permutation_sum_l784_784137

open Finset

theorem max_cyclic_permutation_sum (n : ℕ) (h : n ≥ 2) :
  ∃ (x : Fin n → ℕ), (set.univ = { x i | i < n }).perm (set.univ : Finset (Fin n))
  ∧ (∑ i : Fin n, x i * x ((i+1) % n)) = (2 * n^3 + 3 * n^2 - 11 * n + 18) / 6 :=
by sorry

end max_cyclic_permutation_sum_l784_784137


namespace cos_alpha_add_beta_l784_784679

theorem cos_alpha_add_beta (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : cos (π / 4 + α) = 1 / 3)
  (h4 : cos (π / 4 - β) = sqrt 3 / 3) :
  cos (α + β) = 5 * sqrt 3 / 9 := sorry

end cos_alpha_add_beta_l784_784679


namespace intersection_M_N_l784_784657

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784657


namespace constant_term_binomial_expansion_maximal_fifth_term_l784_784757

theorem constant_term_binomial_expansion_maximal_fifth_term :
  let n := 8
  let expansion := (λ x : ℝ, (x^(1/3) - 2/x))^n
  (choose n 2 * (-2)^2 = 112) →
  (∃ (r : ℕ), r = 2 ∧ (T_{r+1} = (choose n r * (-2)^r)))
  (∃ r : ℕ, r = 2 ∧ expansion.has_maximal_binomial_coeff r 5) :=
sorry

end constant_term_binomial_expansion_maximal_fifth_term_l784_784757


namespace find_g_values_l784_784838

theorem find_g_values
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x * y) = x * g y)
  (h2 : g 1 = 30) :
  g 50 = 1500 ∧ g 0.5 = 15 :=
by
  sorry

end find_g_values_l784_784838


namespace ratio_CD_DA_l784_784024

theorem ratio_CD_DA (A B C D : Type) 
  [Triangle ABC]
  (h1 : AB = 23) 
  (h2 : BC = 24) 
  (h3 : CA = 27) 
  (h4 : Segment D on AC)
  (h5 : Incircles_tangent BAD BCD D) :
  CD / DA = 14 / 13 := 
sorry

end ratio_CD_DA_l784_784024


namespace intersect_at_circumcircle_l784_784804

theorem intersect_at_circumcircle
  (A B C P A1 B1 C1 : Point) 
  (h1 : is_circumcircle P A B C)
  (h2 : parallel (B1 - C1) (A - P))
  (h3 : parallel (C1 - A1) (B - P))
  (h4 : parallel (A1 - B1) (C - P))
  (h5 : parallel (A1 - Circumcircle.ABC.AB))
  (h6 : parallel (B1 - Circumcircle.ABC.BC))
  (h7 : parallel (C1 - Circumcircle.ABC.CA)) :
  ∃ Q, is_circumcircle Q A1 B1 C1 := sorry

end intersect_at_circumcircle_l784_784804


namespace sum_of_sequence_l784_784293

noncomputable def sequence_sum (a : ℝ) (n : ℕ) : ℝ :=
if a = 1 then sorry else (5 * (1 - a ^ n) / (1 - a) ^ 2) - (4 + (5 * n - 4) * a ^ n) / (1 - a)

theorem sum_of_sequence (S : ℕ → ℝ) (a : ℝ) (h1 : S 1 = 1)
                       (h2 : ∀ n, S (n + 1) - S n = (5 * n + 1) * a ^ n) (h3 : |a| ≠ 1) :
  ∀ n, S n = sequence_sum a n :=
  sorry

end sum_of_sequence_l784_784293


namespace centroid_locus_hyperbola_l784_784759

theorem centroid_locus_hyperbola (θ : ℝ) (S : ℝ) 
  (hθ : 0 < θ ∧ θ < π / 2) :
  ∃ c : ℝ, ∀ (r₁ r₂ : ℝ), 
  r₁ * r₂ = 2 * S / sin(2 * θ) →
  let x := (r₁ + r₂) / 3 in
  let y := (r₁ - r₂) / 3 in
  x^2 - y^2 = c := 
begin
  sorry
end

end centroid_locus_hyperbola_l784_784759


namespace geometric_conditions_l784_784594

-- Define the ellipse with the given conditions
def ellipse (a b : ℝ) : Prop := ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions for the problem
variable {a b : ℝ} (h_condition1 : a > b > 0)
variable (h_condition2 : (sqrt 3) / 2 = 1 / 2)
axiom h_condition3 : a = 2 ∧ b = 1

-- Define the line and points conditions
def line_through (n m : ℝ) := ∀ (x y : ℝ), x = n * y + m
variable (l1 l2 : ℝ → ℝ)
variable h_l1_symm (h_l1 : ∀ y, l1 y = -l2 y)
variable (A : ℝ × ℝ) (h_A : A = (4, 0))
variable (M : ℝ × ℝ) (N : ℝ × ℝ)
variable h_M_inter (h_M : M.1^2 / a^2 + M.2^2 / b^2 = 1 ∧ (l1 M.2 = M.1 ∨ l2 M.2 = M.1))
variable h_N_inter (h_N : N.1^2 / a^2 + N.2^2 / b^2 = 1 ∧ (l1 N.2 = N.1 ∨ l2 N.2 = N.1))
variable h_MN (h_MN : M.1 ≠ N.1)

-- The statement of the problem in Lean 4: 
theorem geometric_conditions {a b : ℝ} (h_condition1 : a > b > 0)
  (h_condition2 : (sqrt 3) / 2 = 1 / 2) (h_condition3 : a = 2 ∧ b = 1)
  (l1 l2 : ℝ → ℝ) (h_l1_symm : ∀ y, l1 y = -l2 y)
  (A : ℝ × ℝ) (h_A : A = (4, 0))
  (M N : ℝ × ℝ)
  (h_M_inter : M.1^2 / a^2 + M.2^2 / b^2 = 1 ∧ (l1 M.2 = M.1 ∨ l2 M.2 = M.1))
  (h_N_inter : N.1^2 / a^2 + N.2^2 / b^2 = 1 ∧ (l1 N.2 = N.1 ∨ l2 N.2 = N.1))
  (h_MN : M.1 ≠ N.1) :
  ∃ B, B = (1, 0) ∧ (∃ S, 0 < S ∧ S < (3 * sqrt 3) / 2) :=
sorry

end geometric_conditions_l784_784594


namespace intersection_M_N_l784_784606

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784606


namespace largest_n_property_l784_784031

open Set Nat

theorem largest_n_property :
  ∀ n : ℕ,
    (∀ S : Finset ℕ, S.card = 51 → ∃ a b ∈ S, a ≠ b ∧ a + b = 101) ↔ n = 100 :=
by
  sorry

end largest_n_property_l784_784031


namespace digits_right_of_decimal_l784_784714

theorem digits_right_of_decimal (n : ℕ) (h : n = 1) : 
  let x := (5^5 : ℚ) / (10^3 * 8) in
  nat.digits 10 (x.denom % 10) = [1] :=
by
  -- Proof to be completed
  sorry

end digits_right_of_decimal_l784_784714


namespace angle_BDE_equilateral_l784_784318

theorem angle_BDE_equilateral
  (A B C D E : Type)
  [IsTriangle A B C]
  (angle_A : ∠ A = 65)
  (angle_C : ∠ C = 45)
  (midpoint_D : IsMidpoint D A B)
  (midpoint_E : IsMidpoint E B C)
  (equidistant : distance A D = distance D B ∧ distance B E = distance E C ∧ distance D B = distance B E)
  : ∠ BDE = 60 :=
begin
  sorry
end

end angle_BDE_equilateral_l784_784318


namespace chord_length_of_given_line_and_circle_l784_784188

noncomputable def chord_length_on_circle (r : ℝ) (line : ℝ → ℝ × ℝ) (circle : ℝ → ℝ × ℝ) : ℝ :=
let a := 1 in
let b := 1 in
let c := -2 in
let d := abs (a * 0 + b * 0 + c) / real.sqrt (a^2 + b^2) in
let half_chord_length := real.sqrt (r^2 - d^2) in
2 * half_chord_length

theorem chord_length_of_given_line_and_circle :
  chord_length_on_circle 3 (λ t : ℝ, (1 + 2*t, 1 - 2*t)) (λ α : ℝ, (3 * real.cos α, 3 * real.sin α)) = 2 * real.sqrt 7 :=
by
  sorry

end chord_length_of_given_line_and_circle_l784_784188


namespace solve_problem_l784_784299

-- Declare the variables involved in the problem
variables (a b c : ℕ)

-- State the conditions from the original problem
def condition1 := sqrt a = 3
def condition2 := real.cbrt (b + 1) = 2
def condition3 := sqrt c = 0

-- State the theorem to be proved with the conditions and the expected answer
theorem solve_problem (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  a = 9 ∧ b = 7 ∧ c = 0 ∧ (√ (a * b - c + 1) = 8 ∨ √ (a * b - c + 1) = -8) :=
sorry

end solve_problem_l784_784299


namespace problem_A_inter_B_empty_l784_784708

section

def set_A : Set ℝ := {x | |x| ≥ 2}
def set_B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_A_inter_B_empty : set_A ∩ set_B = ∅ := 
  sorry

end

end problem_A_inter_B_empty_l784_784708


namespace chess_tournament_draws_l784_784745

theorem chess_tournament_draws (total_participants : ℕ := 12) 
  (lists : ℕ → ℕ → Prop) 
  (h_lists_initial : ∀ p : ℕ, lists p 1 = p) 
  (h_lists_defeated: ∀ p k : ℕ, 2 ≤ k ∧ k ≤ total_participants → lists p k = lists p (k-1) ∪ {q | lists q (k-1) = p}) 
  (h_unique : ∀ p q : ℕ, lists p 12 = lists p 11 ∪ {q}) :
  ∃ draws : ℕ, draws = 54 := 
begin
  use 54,
  sorry
end

end chess_tournament_draws_l784_784745


namespace right_triangle_segments_with_integer_lengths_l784_784325

theorem right_triangle_segments_with_integer_lengths
  (A B C : Type*) [linear_ordered_field A]
  (angle_B_90 : ∠ B = 90)
  (AB BC : A)
  (AB_val : AB = 20)
  (BC_val : BC = 21) :
  let AC := real.sqrt (AB^2 + BC^2)
  in let num_segments := ∑ n in (finset.range (20 + 1)).filter (λ n, n ≥ 15 && n ≤ 21), ite (n = 21) 1 2
  in num_segments = 13 :=
begin
  sorry
end

end right_triangle_segments_with_integer_lengths_l784_784325


namespace perimeter_hexagon_PQRSTUV_l784_784995

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def P : ℝ × ℝ := (0, 8)
def Q : ℝ × ℝ := (4, 8)
def R : ℝ × ℝ := (4, 5)
def S : ℝ × ℝ := (7, 5)
def T : ℝ × ℝ := (7, 0)
def U : ℝ × ℝ := (0, 0)
def UV : ℝ := distance U (0, 8)

-- Conditions
def PQ : ℝ := 4
def QR : ℝ := 3
def RS : ℝ := 4
def ST : ℝ := 7
def TU : ℝ := 5

theorem perimeter_hexagon_PQRSTUV : 
  distance P Q + distance Q R + distance R S + distance S T + distance T U + distance U P = 27 + 4 * real.sqrt 5 + real.sqrt 74 := 
sorry

end perimeter_hexagon_PQRSTUV_l784_784995


namespace range_of_f_area_of_triangle_l784_784700

-- Variables and Functions Definitions
variable {x : ℝ}
def f (x : ℝ) : ℝ := 2 * sqrt 3 * (sin x)^2 + 2 * sin x * cos x - sqrt 3
def a : ℝ := sqrt 3
def b : ℝ := 2
def R : ℝ := 3 * sqrt 2 / 4

-- Lean Statements

-- Part (1): Range of the function f(x)
theorem range_of_f :
  ∀ x ∈ Icc (π / 3) (11 * π / 24), sqrt 3 ≤ f x ∧ f x ≤ 2 :=
sorry

-- Part (2): Area of the triangle ABC
theorem area_of_triangle :
  ∃ (S : ℝ), S = sqrt 2 ∧
    (exists A B C : ℝ, 
      sin A = a / (2 * R) ∧ sin B = b / (2 * R) ∧ sin C = sin (A + B) ∧
      S = 0.5 * a * b * sin C) :=
sorry

end range_of_f_area_of_triangle_l784_784700


namespace binary_to_decimal_and_octal_correct_l784_784197

theorem binary_to_decimal_and_octal_correct :
  ∀ b : String, b = "1010101" → 
  let decimal := 1 + 0 * 2^1 + 1 * 2^2 + 0 * 2^3 + 1 * 2^4 + 0 * 2^5 + 1 * 2^6 in
  let octal := 125 in
  decimal = 85 ∧ octal = 125 :=
by
  sorry

end binary_to_decimal_and_octal_correct_l784_784197


namespace intersection_M_N_l784_784602

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784602


namespace prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l784_784491

noncomputable def prob_first_firing_A : ℚ := 4 / 5
noncomputable def prob_first_firing_B : ℚ := 3 / 4
noncomputable def prob_first_firing_C : ℚ := 2 / 3

noncomputable def prob_second_firing : ℚ := 3 / 5

noncomputable def prob_only_one_success_first_firing :=
  prob_first_firing_A * (1 - prob_first_firing_B) * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * prob_first_firing_B * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * (1 - prob_first_firing_B) * prob_first_firing_C

theorem prob_only_one_success_first_firing_is_correct :
  prob_only_one_success_first_firing = 3 / 20 :=
by sorry

noncomputable def prob_success_after_both_firings_A := prob_first_firing_A * prob_second_firing
noncomputable def prob_success_after_both_firings_B := prob_first_firing_B * prob_second_firing
noncomputable def prob_success_after_both_firings_C := prob_first_firing_C * prob_second_firing

noncomputable def prob_all_success_after_both_firings :=
  prob_success_after_both_firings_A * prob_success_after_both_firings_B * prob_success_after_both_firings_C

theorem prob_all_success_after_both_firings_is_correct :
  prob_all_success_after_both_firings = 54 / 625 :=
by sorry

end prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l784_784491


namespace probability_square_product_l784_784877

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def favorable_outcomes : List (ℕ × ℕ) :=
  [ (1, 1), (2, 2), (3, 3), (4, 1), (4, 4), (5, 5), (6, 6), (8, 2), 
    (1, 4), (12, 3), (3, 12), (8, 8) ]

lemma prob_square_product : (12 * 6).to_rat ≠ 0 →
  ∑ p in favorable_outcomes, ite (is_perfect_square (p.1 * p.2)) 1 0 =
  12 :=
begin
  intro h_ne_zero,
  -- Expected count of favorable outcomes is 12
  sorry -- Proof omitted
end

theorem probability_square_product : 
  ∑ p in favorable_outcomes, ite (is_perfect_square (p.1 * p.2)) 1 0 / (12 * 6) = 1 / 6 :=
by 
  have h : (12 * 6).to_rat ≠ 0,
  { norm_num, },
  have favorable_count := prob_square_product h,
  -- Evaluate to get the probability
  sorry -- Proof omitted

end probability_square_product_l784_784877


namespace tan_alpha_is_one_seventh_l784_784244

-- Definitions of given conditions
def tan_half_alpha_plus_beta := (1 : Real) / 2
def tan_beta_minus_half_alpha := (1 : Real) / 3

-- Proving the main statement
theorem tan_alpha_is_one_seventh (α β : Real)
  (h1 : Real.tan (α / 2 + β) = tan_half_alpha_plus_beta)
  (h2 : Real.tan (β - α / 2) = tan_beta_minus_half_alpha) :
  Real.tan α = (1 : Real) / 7 :=
sorry

end tan_alpha_is_one_seventh_l784_784244


namespace rationalize_denominator_correct_l784_784058

noncomputable def rationalize_denominator : Prop := 
  (1 / (Real.cbrt 3 + Real.cbrt 27) = Real.cbrt 9 / 12)

theorem rationalize_denominator_correct : rationalize_denominator := 
  sorry

end rationalize_denominator_correct_l784_784058


namespace find_a10_l784_784005

noncomputable def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d a₁, a 1 = a₁ ∧ ∀ n, a (n + 1) = a n + d

theorem find_a10 (a : ℕ → ℤ) (h_seq : arithmeticSequence a) 
  (h1 : a 1 + a 3 + a 5 = 9) 
  (h2 : a 3 * (a 4) ^ 2 = 27) :
  a 10 = -39 ∨ a 10 = 30 :=
sorry

end find_a10_l784_784005


namespace nonnegative_integer_solutions_l784_784414

theorem nonnegative_integer_solutions (x : ℕ) :
  2 * x - 1 < 5 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
sorry

end nonnegative_integer_solutions_l784_784414


namespace find_x_l784_784128

theorem find_x (x : ℝ) (h : (40 / 80) = Real.sqrt (x / 80)) : x = 20 := 
by 
  sorry

end find_x_l784_784128


namespace geo_seq_a40_a60_l784_784333

theorem geo_seq_a40_a60 (a : ℕ → ℝ) (r : ℝ) (h_geo : ∀ n, a (n+1) = a n * r)
    (h_log : real.logb 2 (a 2 * a 98) = 4) : a 40 * a 60 = 16 :=
by
  sorry

end geo_seq_a40_a60_l784_784333


namespace intersection_M_N_l784_784601

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784601


namespace gcd_of_polynomials_l784_784683

theorem gcd_of_polynomials (b : ℤ) (h : b % 1620 = 0) : Int.gcd (b^2 + 11 * b + 36) (b + 6) = 6 := 
by
  sorry

end gcd_of_polynomials_l784_784683


namespace intersection_M_N_l784_784674

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784674


namespace min_sum_of_sequence_l784_784431

theorem min_sum_of_sequence (a : ℕ → ℕ)
    (h1 : ∀ n ≥ 1, a (n + 2) = (a n + 2023) / (1 + a (n + 1)))
    (h2 : ∀ n, n ≥ 1 → a n > 0) : 
    ∃ a1 a2, a1 + a2 = 136 ∧ 
    ∀ b1 b2, b1 + b2 < a1 + a2 → 
              ∃ m ≥ 1, a (m + 2) ≠ (b m + 2023) / (1 + b (m + 1)) :=
sorry

end min_sum_of_sequence_l784_784431


namespace inequality_proof_l784_784575

theorem inequality_proof (a b : Real) (h1 : (1 / a) < (1 / b)) (h2 : (1 / b) < 0) : 
  (b / a) + (a / b) > 2 :=
by
  sorry

end inequality_proof_l784_784575


namespace problem_l784_784726

noncomputable def K : ℕ := 36
noncomputable def L : ℕ := 147
noncomputable def M : ℕ := 56

theorem problem (h1 : 4 / 7 = K / 63) (h2 : 4 / 7 = 84 / L) (h3 : 4 / 7 = M / 98) :
  (K + L + M) = 239 :=
by
  sorry

end problem_l784_784726


namespace possible_division_of_delegates_l784_784176

-- Define 50 countries and 2 representatives from each country.
constant Country : Type
constant Delegate : Type
constant countries : Fin 50 → Country
constant delegates : Country → Fin 2 → Delegate

-- Assume that there are 50 countries and 2 delegates per country.
axiom distinct_countries : ∀ i j : Fin 50, i ≠ j → countries i ≠ countries j
axiom distinct_delegates : ∀ c : Country, ∀ i j : Fin 2, i ≠ j → delegates c i ≠ delegates c j

-- All 100 delegates seated around a circular table can be represented as a list.
constant seating : List Delegate
axiom seating_length : seating.length = 100

-- Each delegate is from exactly one of the 50 countries.
axiom delegate_from_country : ∀ d : Delegate, ∃ c : Country, ∃ i : Fin 2, delegates c i = d

-- Each delegate has exactly one left and one right neighbor in the seating arrangement.
constant left_neighbor : seating.length > 0 → Delegate → Delegate
constant right_neighbor : seating.length > 0 → Delegate → Delegate

-- Define the theorem that satisfies the problem's condition.
theorem possible_division_of_delegates :
  ∃ (group1 group2 : List Delegate),
    group1.length = 50 ∧
    group2.length = 50 ∧
    (∀ d ∈ group1, ∃ c : Country, delegates c 0 = d ∨ delegates c 1 = d) ∧
    (∀ d ∈ group2, ∃ c : Country, delegates c 0 = d ∨ delegates c 1 = d) ∧
    (∀ d1 d2 ∈ group1, d1 ≠ d2 → ¬ (left_neighbor seating_length_pos d1 = d2 ∨ right_neighbor seating_length_pos d1 = d2)) ∧
    (∀ d1 d2 ∈ group2, d1 ≠ d2 → ¬ (left_neighbor seating_length_pos d1 = d2 ∨ right_neighbor seating_length_pos d1 = d2)) ∧
    (∀ d ∈ group1, ∃ c : Country, d = delegates c 0 ∨ d = delegates c 1) ∧ 
    (∀ d ∈ group2, ∃ c : Country, d = delegates c 0 ∨ d = delegates c 1):=
sorry

end possible_division_of_delegates_l784_784176


namespace area_inside_S_l784_784813

-- Condition definitions
def four_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), abs w = 4 ∧ z = w - 1 / w

def S : set ℂ := {z | four_presentable z}

-- Required proof statement
theorem area_inside_S : 
  let r := 4 in
  let a := 15 / 16 in
  let b := 17 / 16 in
  let original_area := π * r^2 in
  let scaled_area := original_area * a * b in
  scaled_area = (255 / 16) * π := 
by
  sorry

end area_inside_S_l784_784813


namespace not_prime_257_1092_1092_l784_784902

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_257_1092_1092 :
  is_prime 1093 →
  ¬ is_prime (257 ^ 1092 + 1092) :=
by
  intro h_prime_1093
  -- Detailed steps are omitted, proof goes here
  sorry

end not_prime_257_1092_1092_l784_784902


namespace center_and_two_points_form_isosceles_l784_784405

theorem center_and_two_points_form_isosceles
  (O A B : Point)
  (h1 : is_center O)
  (h2 : on_circle A O)
  (h3 : on_circle B O)
  (h4 : distance O A = distance O B) :
  is_isosceles O A B :=
sorry

end center_and_two_points_form_isosceles_l784_784405


namespace f_monotonically_increasing_g_max_min_values_l784_784699

noncomputable def f (x : Real) : Real := Real.cos (2 * x + Real.pi / 3)

theorem f_monotonically_increasing (k : ℤ) :
  ∀ x : ℝ, k * Real.pi - 2 * Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi - Real.pi / 6 → 
            f x ≤ f (x + ε) := sorry

noncomputable def g (x : Real) : Real := Real.cos (x + Real.pi / 6)

theorem g_max_min_values :
  ∃ max min : Real, max = Real.sqrt 3 / 2 ∧ min = -1 ∧
                   (∀ x ∈ Icc (0 : Real) Real.pi, g x ≤ max ∧ g x ≥ min) := sorry

end f_monotonically_increasing_g_max_min_values_l784_784699


namespace intersection_M_N_l784_784673

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784673


namespace value_at_15_l784_784974

def quadratic_function (q : ℝ → ℝ) : Prop :=
  ∃ a b c, ∀ x, q x = a * x^2 + b * x + c

def passes_through (q : ℝ → ℝ) (x_val y_val : ℝ) : Prop :=
  q x_val = y_val

def symmetric_around (q : ℝ → ℝ) (sym_x : ℝ) : Prop :=
  ∀ x1 x2, x1 + x2 = 2 * sym_x → q x1 = q x2

theorem value_at_15 (q : ℝ → ℝ) (a b c : ℝ) :
  quadratic_function q →
  passes_through q 0 (-3) →
  symmetric_around q 7.5 →
  q(15) = -3 :=
by
  intros h_quad h_pass h_sym
  sorry

end value_at_15_l784_784974


namespace lowest_possible_number_of_students_l784_784928

theorem lowest_possible_number_of_students :
  ∃ N : ℕ, (N % 12 = 0) ∧ (N % 24 = 0) ∧ ∀ n : ℕ, (n % 12 = 0) ∧ (n % 24 = 0) → N ≤ n ↔ N = 24 :=
begin
  sorry
end

end lowest_possible_number_of_students_l784_784928


namespace remainder_3_pow_2000_mod_17_l784_784126

theorem remainder_3_pow_2000_mod_17 : (3^2000 % 17) = 1 := by
  sorry

end remainder_3_pow_2000_mod_17_l784_784126


namespace n_times_s_l784_784366

axiom g : ℝ → ℝ
axiom functional_eq : ∀ x y : ℝ, g (x * g y + 2 * x) = 2 * x * y + g x

theorem n_times_s :
  let n := 1 in
  let s := -4 in
  n * s = -4 :=
by
  sorry

end n_times_s_l784_784366


namespace length_of_EF_l784_784809

-- Define the rectangle and its properties
def Rectangle (A B C D : Type) [MetricSpace A] (AB BC CD DA : ℕ) : Prop :=
  dist A B = AB ∧ dist B C = BC ∧ dist C D = CD ∧ dist D A = DA ∧
  dist A C = dist B D

-- Define the property of an angle bisector
def angle_bisector (B C D : Type) (EF : Type) [MetricSpace B] : Prop :=
  ∃ E F, is_angle_bisector B D C E F

-- Define the condition of points lying on specific lines
def points_on_lines (A C D E F : Type) [MetricSpace A] : Prop :=
  lies_on A D E ∧ lies_on C D F

-- Using Rectangle and angle_bisector definitions to encode the complete problem statement
theorem length_of_EF (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (h_rect : Rectangle A B C D 8 6 8 6)
  (h_bisect : angle_bisector B D C F)
  (h_points : points_on_lines A C D E F) : dist E F = 10 := sorry

end length_of_EF_l784_784809


namespace max_cake_boxes_in_carton_l784_784471

-- Define the dimensions of the carton as constants
def carton_length := 25
def carton_width := 42
def carton_height := 60

-- Define the dimensions of the cake box as constants
def box_length := 8
def box_width := 7
def box_height := 5

-- Define the volume of the carton and the volume of the cake box
def volume_carton := carton_length * carton_width * carton_height
def volume_box := box_length * box_width * box_height

-- Define the theorem statement
theorem max_cake_boxes_in_carton : 
  (volume_carton / volume_box) = 225 :=
by
  -- The proof is omitted.
  sorry

end max_cake_boxes_in_carton_l784_784471


namespace probability_of_Xiaojia_selection_l784_784949

theorem probability_of_Xiaojia_selection : 
  let students := 2500
  let teachers := 350
  let support_staff := 150
  let total_individuals := students + teachers + support_staff
  let sampled_individuals := 300
  let student_sample := (students : ℝ)/total_individuals * sampled_individuals
  (student_sample / students) = (1 / 10) := 
by
  sorry

end probability_of_Xiaojia_selection_l784_784949


namespace rearrange_strawberries_to_plums_l784_784800

-- Definitions of the grid, strawberries, and plums
variable (n : ℕ)
def grid : Type := matrix (fin n) (fin n) (option bool)  -- None indicates no fruit, some true for strawberry, some false for plum

-- Conditions
variable (G : grid n)
variable (plum_pos : fin n × fin n) -- The position of the plum
variable (one_strawberry_per_column : ∀ j : fin n, ∃ i : fin n, G i j = some true)
variable (one_strawberry_per_row : ∀ i : fin n, ∃ j : fin n, G i j = some true)
variable (more_plums_in_any_subrectangle : ∀ (x y : fin n), ∑ i in range x, ∑ j in range y, if G i j = some false then 1 else 0 > ∑ i in range x, ∑ j in range y, if G i j = some true then 1 else 0)

-- A permissible move: swap strawberries for a given rectangle configuration
def permissible_move (G : grid n) (x1 y1 x2 y2 : fin n) (hx1x2 : x1 < x2) (hy1y2 : y1 < y2) :
  Prop := (G x1 y1 = some true) ∧ (G x2 y2 = some true) ∧ (G x1 y2 = none) ∧ (G x2 y1 = none)

-- Theorem to prove: rearrange strawberries to plums
theorem rearrange_strawberries_to_plums : ∃ (G' : grid n),
  (∀ i j, (i, j) = plum_pos → G' i j = some true) ∧
  (∀ x1 y1 x2 y2 (hx1x2 : x1 < x2) (hy1y2 : y1 < y2), permissible_move G x1 y1 x2 y2 hx1x2 hy1y2 → permissible_move G' x1 y1 x2 y2 hx1x2 hy1y2) :=
sorry

end rearrange_strawberries_to_plums_l784_784800


namespace intersection_M_N_l784_784609

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784609


namespace find_k_l784_784472

theorem find_k (m n k : ℤ) (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 0 := by
  sorry

end find_k_l784_784472


namespace farmer_eggs_per_week_l784_784935

theorem farmer_eggs_per_week (E : ℝ) (chickens : ℝ) (price_per_dozen : ℝ) (total_revenue : ℝ) (num_weeks : ℝ) (total_chickens : ℝ) (dozen : ℝ) 
    (H1 : total_chickens = 46)
    (H2 : price_per_dozen = 3)
    (H3 : total_revenue = 552)
    (H4 : num_weeks = 8)
    (H5 : dozen = 12)
    (H6 : chickens = 46)
    : E = 6 :=
by
  sorry

end farmer_eggs_per_week_l784_784935


namespace infinite_perpendicular_lines_one_perpendicular_plane_infinite_parallel_lines_one_parallel_plane_l784_784718

-- Definitions:
variables {Point : Type}
variables {Line : Type}
variables {Plane : Type}

-- Condition: Point is outside a given line
variables (p : Point) (l : Line)
axiom point_outside_line : p ∉ l

-- Proof Statements:
theorem infinite_perpendicular_lines : ∃∞ l' : Line, (p ∉ l') ∧ (l ⊥ l') := sorry

theorem one_perpendicular_plane : ∃! ℘ : Plane, (p ∈ ℘) ∧ (l ⊥ ℘) := sorry

theorem infinite_parallel_lines : ∃∞ l' : Line, (p ∉ l') ∧ (l ∥ l') := sorry

theorem one_parallel_plane : ∃! ℘ : Plane, (p ∈ ℘) ∧ (l ∥ ℘) := sorry

end infinite_perpendicular_lines_one_perpendicular_plane_infinite_parallel_lines_one_parallel_plane_l784_784718


namespace average_speed_additional_hours_l784_784490

theorem average_speed_additional_hours (v : ℝ) 
  (h1 : ∀ t ≤ 4, average_speed t = 50)
  (h2 : average_speed 8 = 65)
  (h3 : total_time = 8) :
  v = 80 := by
  sorry

end average_speed_additional_hours_l784_784490


namespace intersection_M_N_l784_784676

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784676


namespace rad_measure_of_neg_120_l784_784418

/-!
Theorem: The radian measure of -120 degrees is -2π/3
-/

/-- Convert degrees to radians given a degree value and the conversion formula.
  The conversion formula is: 1 radian = degree value × π / 180 -/
def convert_to_radians (deg : ℤ) : ℝ :=
  deg * (Real.pi / 180)

/-- Statement: The radian measure of -120 degrees equals -2π/3 -/
theorem rad_measure_of_neg_120 : 
  convert_to_radians (-120) = - (2 * Real.pi/3) :=
by
  sorry

end rad_measure_of_neg_120_l784_784418


namespace not_injective_of_gt_not_surjective_of_lt_bijective_of_eq_l784_784023

-- Definitions of vector space and linear map
variables {K : Type*} [Field K]
variables {E : Type*} [AddCommGroup E] [Module K E] [FiniteDimensional K E]
variables {F : Type*} [AddCommGroup F] [Module K F] [FiniteDimensional K F]

-- Problem 1: Prove that if n > p then f is not injective
theorem not_injective_of_gt {f : E →ₗ[K] F} (hnp : FiniteDimensional.finRank K E > FiniteDimensional.finRank K F) : ¬Function.Injective f :=
by sorry

-- Problem 2: Prove that if n < p then f is not surjective
theorem not_surjective_of_lt {f : E →ₗ[K] F} (hnp : FiniteDimensional.finRank K E < FiniteDimensional.finRank K F) : ¬Function.Surjective f :=
by sorry

-- Problem 3: Prove that if n = p then f can be bijective
theorem bijective_of_eq {f : E →ₗ[K] F} (hnp : FiniteDimensional.finRank K E = FiniteDimensional.finRank K F) : ∃ (g : E →ₗ[K] F), Function.Bijective g :=
by sorry

end not_injective_of_gt_not_surjective_of_lt_bijective_of_eq_l784_784023


namespace area_of_triangle_PQR_l784_784879

def point := (ℝ × ℝ)

def slope (m : ℝ) (pt : point) (y_intercept : ℝ) (x : ℝ) : ℝ :=
  m * (x - pt.1) + pt.2 - y_intercept

def line1 (pt : point) := slope (-1) pt 0
def line2 (pt : point) := slope 1.5 pt 0

def triangle_area (P Q R : point) : ℝ :=
  0.5 * abs ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1))

theorem area_of_triangle_PQR :
  let P : point := (2, 5)
  let Q : point := (-3, 0)
  let R : point := (5.33, 0)
  triangle_area P Q R = 20.825 := 
by 
  sorry

end area_of_triangle_PQR_l784_784879


namespace sum_of_series_eq_half_l784_784978

theorem sum_of_series_eq_half :
  (∑' k : ℕ, 3^(2^k) / (9^(2^k) - 1)) = 1 / 2 :=
by
  sorry

end sum_of_series_eq_half_l784_784978


namespace exists_line_intersecting_circle_and_passing_origin_l784_784279

theorem exists_line_intersecting_circle_and_passing_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -4) ∧ 
  ∃ (x y : ℝ), 
    ((x - 1) ^ 2 + (y + 2) ^ 2 = 9) ∧ 
    ((x - y + m = 0) ∧ 
     ∃ (x' y' : ℝ),
      ((x' - 1) ^ 2 + (y' + 2) ^ 2 = 9) ∧ 
      ((x' - y' + m = 0) ∧ ((x + x') / 2 = 0 ∧ (y + y') / 2 = 0))) :=
by 
  sorry

end exists_line_intersecting_circle_and_passing_origin_l784_784279


namespace rice_weight_per_container_in_grams_l784_784474

-- Define the initial problem conditions
def total_weight_pounds : ℚ := 35 / 6
def number_of_containers : ℕ := 5
def pound_to_grams : ℚ := 453.592

-- Define the expected answer
def expected_answer : ℚ := 529.1907

-- The statement to prove
theorem rice_weight_per_container_in_grams :
  (total_weight_pounds / number_of_containers) * pound_to_grams = expected_answer :=
by
  sorry

end rice_weight_per_container_in_grams_l784_784474


namespace quadratic_with_root_one_l784_784384

theorem quadratic_with_root_one (a b c : ℝ) (h : a ≠ 0) :
  (a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) → c = -a - b :=
by
  intro h_root
  rw [one_mul, one_mul, one_pow] at h_root
  sorry

end quadratic_with_root_one_l784_784384


namespace group_booking_cost_correct_l784_784312

-- Definitions based on the conditions of the problem
def weekday_rate_first_week : ℝ := 18.00
def weekend_rate_first_week : ℝ := 20.00
def weekday_rate_additional_weeks : ℝ := 11.00
def weekend_rate_additional_weeks : ℝ := 13.00
def security_deposit : ℝ := 50.00
def discount_rate : ℝ := 0.10
def group_size : ℝ := 5
def stay_duration : ℕ := 23

-- Computation of total cost
def total_cost (first_week_weekdays : ℕ) (first_week_weekends : ℕ) 
  (additional_week_weekdays : ℕ) (additional_week_weekends : ℕ) 
  (additional_days_weekdays : ℕ) : ℝ := 
  let cost_first_weekdays := first_week_weekdays * weekday_rate_first_week
  let cost_first_weekends := first_week_weekends * weekend_rate_first_week
  let cost_additional_weeks := 2 * (additional_week_weekdays * weekday_rate_additional_weeks + 
                                    additional_week_weekends * weekend_rate_additional_weeks)
  let cost_additional_days := additional_days_weekdays * weekday_rate_additional_weeks
  let total_before_deposit := cost_first_weekdays + cost_first_weekends + 
                              cost_additional_weeks + cost_additional_days
  let total_before_discount := total_before_deposit + security_deposit
  let total_discount := discount_rate * total_before_discount
  total_before_discount - total_discount

-- Proof setup
theorem group_booking_cost_correct :
  total_cost 5 2 5 2 2 = 327.60 :=
by
  -- Placeholder for the proof; steps not required for Lean statement
  sorry

end group_booking_cost_correct_l784_784312


namespace fruit_bowl_remaining_l784_784872

-- Define the initial conditions
def oranges : Nat := 3
def lemons : Nat := 6
def fruits_eaten : Nat := 3

-- Define the total count of fruits initially
def total_fruits : Nat := oranges + lemons

-- The goal is to prove remaining fruits == 6
theorem fruit_bowl_remaining : total_fruits - fruits_eaten = 6 := by
  sorry

end fruit_bowl_remaining_l784_784872


namespace arctan_addition_formula_l784_784563

noncomputable def arctan_add : ℝ :=
  Real.arctan (1 / 3) + Real.arctan (3 / 8)

theorem arctan_addition_formula :
  arctan_add = Real.arctan (17 / 21) :=
by
  sorry

end arctan_addition_formula_l784_784563


namespace sixth_square_area_l784_784080

noncomputable def side_length (r : ℝ) : ℝ := r * (2 / Real.sqrt 2)

theorem sixth_square_area (r : ℝ) (h : r = 10) :
  let s := side_length r,
      small_square_diagonal := r,
      small_square_side_length := small_square_diagonal / Real.sqrt 2 in
  (small_square_side_length / Real.sqrt 2) * (small_square_side_length / Real.sqrt 2) * 2 = 100 :=
by
  sorry

end sixth_square_area_l784_784080


namespace polynomial_is_constant_l784_784025

theorem polynomial_is_constant (P : Polynomial ℤ) 
  (h1 : ∀ n : ℤ, Nat.Prime (P.eval (n : ℤ))) : (∃ c : ℤ, P = Polynomial.C c) :=
by
  sorry

end polynomial_is_constant_l784_784025


namespace correct_equation_l784_784432

-- Define the conditions
variables {x : ℝ}

-- Condition 1: The unit price of a notebook is 2 yuan less than that of a water-based pen.
def notebook_price (water_pen_price : ℝ) : ℝ := water_pen_price - 2

-- Condition 2: Xiaogang bought 5 notebooks and 3 water-based pens for exactly 14 yuan.
def total_cost (notebook_price water_pen_price : ℝ) : ℝ :=
  5 * notebook_price + 3 * water_pen_price

-- Question restated as a theorem: Verify the given equation is correct
theorem correct_equation (water_pen_price : ℝ) (h : total_cost (notebook_price water_pen_price) water_pen_price = 14) :
  5 * (water_pen_price - 2) + 3 * water_pen_price = 14 :=
  by
    -- Introduce the assumption
    intros
    -- Sorry to skip the proof
    sorry

end correct_equation_l784_784432


namespace volume_ratio_of_cones_l784_784198

noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def central_angle_ratio (theta1 theta2 : ℝ) (h : 3 * theta1 = 4 * theta2) := (theta1, theta2)

theorem volume_ratio_of_cones (r θ₁ θ₂ V₁ V₂ : ℝ) 
  (h1 : 3 * θ₁ = 4 * θ₂) 
  (h2 : θ₁ + θ₂ = 2 * real.pi) 
  (h3 : V₁ = (1/3:ℝ) * real.pi * (3*r/4) ^ 2 * (sqrt (r^2 - (3*r/4)^2))) 
  (h4 : V₂ = (1/3:ℝ) * real.pi * (4*r/3) ^ 2 * (sqrt (r^2 - (4*r/3)^2))) :
  V₁ / V₂ = 27 / 64 :=
by
  sorry

end volume_ratio_of_cones_l784_784198


namespace find_b_l784_784019

noncomputable def b_value (a : ℝ) (h : a > 1) : ℝ := 1 / a + 1

theorem find_b (a : ℝ) (h : a > 1): 
  ∃ b : ℝ, b ≥ 1 ∧ 
  (∀ (x : ℝ), x > 0 → 
  tendsto (λ (x : ℝ), ∫ t in 0..x, (1 + t^a)^(-b_value a h)) at_top (𝓝 1)) := 
sorry

end find_b_l784_784019


namespace rotation_matrix_pow_eight_is_identity_l784_784993

theorem rotation_matrix_pow_eight_is_identity :
  let R := Matrix.of ![![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4)],
                       ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]]
  R^8 = 1 :=
by
  let R : Matrix (Fin 2) (Fin 2) ℝ := 
    #[#[Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4)],
      #[Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]]
  sorry

end rotation_matrix_pow_eight_is_identity_l784_784993


namespace percent_decrease_l784_784911

theorem percent_decrease (P S : ℝ) (h₀ : P = 100) (h₁ : S = 70) :
  ((P - S) / P) * 100 = 30 :=
by
  sorry

end percent_decrease_l784_784911


namespace cube_root_of_neg8_l784_784830

-- Define the condition
def is_cube_root (x : ℝ) : Prop := x^3 = -8

-- State the problem to be proved.
theorem cube_root_of_neg8 : is_cube_root (-2) :=
by 
  sorry

end cube_root_of_neg8_l784_784830


namespace part_a_part_b_part_c_part_d_part_e_l784_784954

variables {V : Type*} [inner_product_space ℝ V]

-- Definition of orthogonal_tetrahedron
structure orthogonal_tetrahedron (A B C D : V) : Prop :=
(perpendicular : ∀ (e₁ e₂ : V), ∃ (u v : V), (e₁ = B - A ∧ e₂ = D - C ∧ ⟪u, v⟫ = 0) 
                        ∨ (e₁ = C - A ∧ e₂ = D - B ∧ ⟪u, v⟫ = 0)
                        ∨ (e₁ = D - A ∧ e₂ = C - B ∧ ⟪u, v⟫ = 0))

-- Part (a)
theorem part_a (A B C D : V) (hab : ⟪B - A, D - C⟫ = 0) (hac : ⟪C - A, D - B⟫ = 0) : ⟪D - A, C - B⟫ = 0 := sorry

-- Part (b)
theorem part_b (A B C D : V) (orth : orthogonal_tetrahedron A B C D) : 
  ∃ (O : V), ∀ x ∈ {A, B, C, D}, ∃ (h : V), altitude x A B C D (h = O) := sorry

-- Part (c)
theorem part_c (A B C D : V) (h₁ : altitude x₁ A B C D (h₁ = O)) (h₂ : altitude x₂ A B C D (h₂ = O)) 
  (h₃ : altitude x₃ A B C D (h₃ = O)) : orthogonal_tetrahedron A B C D := sorry

-- Part (d)
theorem part_d (A B C D : V) (h : altitude (foot (altitude x A B C)) (face_orthocenter A B C) = O) :
  (orthogonal_tetrahedron A B C D ∧ altitude (foot (altitude x A B C)) (face_orthocenter A B C) = O) ↔ 
  orthogonal_tetrahedron A B C D := sorry

-- Part (e)
theorem part_e (A B C D : V) (orth : orthogonal_tetrahedron A B C D) : 
  ∃ (O : V), common_perpendicular A B C D (O) := sorry

end part_a_part_b_part_c_part_d_part_e_l784_784954


namespace smallest_value_of_n_l784_784962

theorem smallest_value_of_n : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n + 6) % 7 = 0 ∧ (n - 9) % 4 = 0 ∧ n = 113 :=
by
  sorry

end smallest_value_of_n_l784_784962


namespace g_increasing_in_interval_l784_784249

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + a * x + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x + a
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := 2 * x - 2 * a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f'' a x / x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 1 - a / (x^2)

theorem g_increasing_in_interval (a : ℝ) (h : a < 1) :
  ∀ x : ℝ, 1 < x → 0 < g' a x := by
  sorry

end g_increasing_in_interval_l784_784249


namespace find_f_x_squared_l784_784212

variable {C : ℝ}
variable (f : ℝ → ℝ)

def nonneg_reals := {x : ℝ // 0 ≤ x}

axiom ax1 : ∀ (x : nonneg_reals), f(f x) = x.val^4

axiom ax2 : ∀ (x : nonneg_reals), f x ≤ C * x.val^2

theorem find_f_x_squared (f : ℝ → ℝ) (C : ℝ) :
  (∀ (x : nonneg_reals), f(f(x)) = x.val^4) →
  (∀ (x : nonneg_reals), f(x) ≤ C * x.val^2) →
  f = (λ x, x^2) :=
by
  intros h1 h2
  funext x
  sorry

end find_f_x_squared_l784_784212


namespace intersection_M_N_l784_784652

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784652


namespace total_length_correct_l784_784068

-- Define the required percentages
def repaired_in_first_week (total_length : ℕ) : ℕ := total_length * 20 / 100
def repaired_in_second_week (total_length : ℕ) : ℕ := total_length * 25 / 100

-- Define the constants for the third week repairs and the remaining part
def repaired_in_third_week : ℕ := 480
def remaining_length : ℕ := 70

-- The total length of the road
def total_length_of_road : ℕ := 1000

-- The theorem statement
theorem total_length_correct (total_length : ℕ) :
  repaired_in_first_week total_length 
  + repaired_in_second_week total_length 
  + repaired_in_third_week 
  + remaining_length = total_length :=
by
  unfold repaired_in_first_week repaired_in_second_week
  simp
  sorry

end total_length_correct_l784_784068


namespace perpendicular_angles_l784_784741

theorem perpendicular_angles (α β : ℝ) (k : ℤ) : 
  (∃ k : ℤ, β - α = k * 360 + 90 ∨ β - α = k * 360 - 90) →
  β = k * 360 + α + 90 ∨ β = k * 360 + α - 90 :=
by
  sorry

end perpendicular_angles_l784_784741


namespace circle_radius_l784_784226

theorem circle_radius :
  ∃ r, (∀ x y, 4*x^2 - 8*x + 4*y^2 + 24*y + 35 = 0 → r = sqrt (5 / 4)) := by
  sorry

end circle_radius_l784_784226


namespace solution_set_of_f_gt_2x_add_4_l784_784832

theorem solution_set_of_f_gt_2x_add_4 {f : ℝ → ℝ} (h_domain : ∀ x, true) (h_f_neq : f (-1) = 2)
  (h_deriv : ∀ x, deriv f x > 2) : 
  {x : ℝ | f x > 2 * x + 4} = Ioi (-1) := 
    by 
    sorry

end solution_set_of_f_gt_2x_add_4_l784_784832


namespace distance_between_trees_l784_784909

theorem distance_between_trees (num_trees : ℕ) (length_yard : ℝ)
  (h1 : num_trees = 26) (h2 : length_yard = 800) : 
  (length_yard / (num_trees - 1)) = 32 :=
by
  sorry

end distance_between_trees_l784_784909


namespace avg_of_x_y_is_41_l784_784231

theorem avg_of_x_y_is_41 
  (x y : ℝ) 
  (h : (4 + 6 + 8 + x + y) / 5 = 20) 
  : (x + y) / 2 = 41 := 
by 
  sorry

end avg_of_x_y_is_41_l784_784231


namespace m_range_if_increasing_max_min_values_when_m_is_2_l784_784701

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * x^2 - m * Real.log x

theorem m_range_if_increasing :
  (∀ x ∈ Set.Ioi (1 / 2 : ℝ), deriv (λ x, f x m) x ≥ 0) → m ≤ 1 / 4 := by
  sorry

theorem max_min_values_when_m_is_2 :
  (m = 2) →
  (∀ x ∈ Set.Icc (1 : ℝ) Real.exp 1, f x m ≤ f Real.exp 1 m) ∧
  (∀ x ∈ Set.Icc (1 : ℝ) Real.exp 1, f x m ≥ f (Real.sqrt 2) m) := by
  sorry

end m_range_if_increasing_max_min_values_when_m_is_2_l784_784701


namespace sum_of_interior_angles_of_polygon_l784_784159

theorem sum_of_interior_angles_of_polygon (n : ℕ) (h : n - 3 = 3) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l784_784159


namespace mode_and_median_l784_784746

def scores := [80, 80, 85, 85, 90, 90, 90, 95]

theorem mode_and_median :
  (mode scores = 90) ∧ (median scores = 87.5) :=
by
  sorry

end mode_and_median_l784_784746


namespace nth_highest_price_l784_784528

theorem nth_highest_price (n : ℕ) (items : ℕ) (lowest_price_rank : ℕ) :
  items = 38 →
  lowest_price_rank = 23 →
  n = items - lowest_price_rank + 1 :=
begin
  intros h_items h_lowest_price_rank,
  rw [h_items, h_lowest_price_rank],
  exact eq.refl 17
end

end nth_highest_price_l784_784528


namespace number_of_bottles_of_milk_l784_784108

theorem number_of_bottles_of_milk (m : ℕ) :
  let loaves_of_bread := 37 in
  let total_items := 52 in
  37 + m = 52 → m = 15 :=
by
  intro h1
  sorry

end number_of_bottles_of_milk_l784_784108


namespace prove_x3_y3_z3_equals_36_l784_784028

noncomputable def sets_equality (x y z : ℕ) : Prop :=
  {3 * (x - y) * (y - z) * (z - x), x * y * z, 2 * (y^2 * z^2 + z^2 * x^2 + x^2 * y^2)} =
  {(x - y)^3 + (y - z)^3 + (z - x)^3, x + y + z, x^4 + y^4 + z^4}

theorem prove_x3_y3_z3_equals_36 :
  ∀ (x y z : ℕ),
    0 < x → 0 < y → 0 < z → sets_equality x y z →
    x^3 + y^3 + z^3 = 36 :=
by
  intros x y z x_pos y_pos z_pos sets_eq
  sorry

end prove_x3_y3_z3_equals_36_l784_784028


namespace numbers_are_integers_l784_784146

theorem numbers_are_integers : 
  ∀ (n : ℤ), n ∈ {-3, -2, -1, 0, 1, 2} -> ∃ (m : ℤ), n = m := 
by
  sorry

end numbers_are_integers_l784_784146


namespace largest_three_digit_divisible_by_l784_784893

theorem largest_three_digit_divisible_by (n : ℕ) :
  divisible (n) 6 ∧ divisible (n) 5 ∧ divisible (n) 8 ∧ divisible (n) 9 ∧ (100 ≤ n ∧ n ≤ 999) → 
  n = 720 :=
sorry

end largest_three_digit_divisible_by_l784_784893


namespace intersection_M_N_is_correct_l784_784623

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784623


namespace day50_previous_year_is_Wednesday_l784_784767

-- Given conditions
variable (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)

-- Provided conditions stating specific days are Fridays
def day250_is_Friday : Prop := dayOfWeek 250 N = 5
def day150_is_Friday_next_year : Prop := dayOfWeek 150 (N+1) = 5

-- Proving the day of week for the 50th day of year N-1
def day50_previous_year : Prop := dayOfWeek 50 (N-1) = 3

-- Main theorem tying it together
theorem day50_previous_year_is_Wednesday (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)
  (h1 : day250_is_Friday N dayOfWeek)
  (h2 : day150_is_Friday_next_year N dayOfWeek) :
  day50_previous_year N dayOfWeek :=
sorry -- Placeholder for actual proof

end day50_previous_year_is_Wednesday_l784_784767


namespace calc_repeating_fractions_l784_784533

def repeating_seventy_two : ℚ := 8 / 11
def repeating_zero_nine : ℚ := 1 / 11

lemma fraction_zero_seventy_two :
  ∀ x : ℚ, x = 0.\bar{72} -> x = repeating_seventy_two := by
  sorry

lemma fraction_two_zero_nine :
  ∀ y : ℚ, y = 2.\bar{09} -> y = 23 / 11 := by
  sorry

theorem calc_repeating_fractions :
  ∀ x y : ℚ, (x = repeating_seventy_two) -> (y = 23 / 11) -> (x / y) = 8 / 23 := by
  -- assume the results of the decimal conversions
  intros x y hx hy
  rw [hx, hy]
  norm_num
  sorry

end calc_repeating_fractions_l784_784533


namespace number_of_points_on_ellipse_with_given_conditions_l784_784576

noncomputable def ellipse_foci {a b : ℝ} (x y : ℝ) : Prop :=
  (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1

theorem number_of_points_on_ellipse_with_given_conditions :
  let a := 5 in
  let b := 4 in
  let c := 3 in
  ∃ M : ℝ × ℝ, let r := (3 : ℝ) / 2 in
  ellipse_foci 5 4 M.1 M .2 ∧
  ∃ F1 F2 : ℝ × ℝ, F1 = (-c, 0) ∧ F2 = (c, 0) ∧
  let MF1 := abs (M.1 - F1.1) in
  let MF2 := abs (M.1 - F2.1) in
  let area := (MF1 + MF2 + 2 * c) * r / 2 in
  area = 8 * r ∧
  area = c * | M.2 | ∧
  |M .2| = 4 ∧ 
  ∃! (M' : ℝ × ℝ), (ellipse_foci 5 4 M'.1 M'.2 ∧ |M'.2| = 4) :=
sorry

end number_of_points_on_ellipse_with_given_conditions_l784_784576


namespace find_f_neg_half_l784_784283

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x then 2 / 3^x else g x

axiom g (x : ℝ) : ℝ

axiom odd_f : ∀ x : ℝ, f (-x) = -f x

theorem find_f_neg_half : f (-1/2 : ℝ) = -2 * Real.sqrt 3 / 3 := by
  sorry

end find_f_neg_half_l784_784283


namespace pencil_and_pen_cost_l784_784089

theorem pencil_and_pen_cost
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 3.75)
  (h2 : 2 * p + 3 * q = 4.05) :
  p + q = 1.56 :=
by
  sorry

end pencil_and_pen_cost_l784_784089


namespace sum_of_digits_T_for_horses_l784_784863

theorem sum_of_digits_T_for_horses : 
  ∃ (T : ℕ), (T > 0) ∧ 
  (∀ (k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), ∃ n ∈ ({1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144} : set ℕ), n = k^2 ∧ T % n = 0) ∧
  (card ((λ k, (some n ∈ ({1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144} : set ℕ), n = k^2 ∧ T % n = 0))) = 6) ∧ 
  (nat.digits 10 T).sum = 9 :=
begin
  sorry -- Proof to be completed
end

end sum_of_digits_T_for_horses_l784_784863


namespace enclosed_area_eq_32_over_3_l784_784558

noncomputable def enclosed_area (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (f x - g x)

theorem enclosed_area_eq_32_over_3 :
  let f := λ x : ℝ, -x^2
  let g := λ x : ℝ, 2 * x - 3
  enclosed_area f g (-3) 1 = 32 / 3 :=
by
  sorry

end enclosed_area_eq_32_over_3_l784_784558


namespace trigonometric_range_l784_784734

open Real

theorem trigonometric_range (α : ℝ) :
  (0 < α ∧ α < 2 * π) ∧ (sin α < sqrt 3 / 2) ∧ (cos α > 1 / 2) →
  (0 < α ∧ α < π / 3) ∨ (5 * π / 3 < α ∧ α < 2 * π) :=
by
  sorry

end trigonometric_range_l784_784734


namespace investment_ratio_l784_784018

theorem investment_ratio (total_investment Jim_investment : ℕ) (h₁ : total_investment = 80000) (h₂ : Jim_investment = 36000) :
  (total_investment - Jim_investment) / Nat.gcd (total_investment - Jim_investment) Jim_investment = 11 ∧ Jim_investment / Nat.gcd (total_investment - Jim_investment) Jim_investment = 9 :=
by
  sorry

end investment_ratio_l784_784018


namespace ratio_SP_CP_l784_784420

variables (CP SP P : ℝ)
axiom ratio_profit_CP : P / CP = 2

theorem ratio_SP_CP : SP / CP = 3 :=
by
  -- Proof statement (not required as per the instruction)
  sorry

end ratio_SP_CP_l784_784420


namespace area_midpoints_geq_half_area_l784_784478

-- Let's define the 2017-gon and the statement we need to prove
open scoped Classical

/-- A type representing points -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A predicate for convex polygon -/
def convex_poly (vertices : Fin 2017 → Point) : Prop := sorry

/-- The area function for a polygon given a list of vertices -/
def area (vertices : Fin 2017 → Point) : ℝ := sorry

/-- Midpoints of the sides of a convex polygon-/
def midpoints (vertices : Fin 2017 → Point) : Fin 2017 → Point :=
  λ i, let i' := (i + 1) % 2017
       (vertices i + vertices i') / 2

theorem area_midpoints_geq_half_area (vertices : Fin 2017 → Point) (h_convex : convex_poly vertices) :
  area (midpoints vertices) ≥ (area vertices) / 2 :=
sorry

end area_midpoints_geq_half_area_l784_784478


namespace intersection_M_N_l784_784677

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784677


namespace least_subtraction_to_divisible_by_prime_l784_784129

theorem least_subtraction_to_divisible_by_prime :
  ∃ k : ℕ, (k = 46) ∧ (856324 - k) % 101 = 0 :=
by
  sorry

end least_subtraction_to_divisible_by_prime_l784_784129


namespace sum_series_l784_784976

theorem sum_series : (∑ k in (Finset.range ∞), (3 ^ (2 ^ k)) / ((3 ^ 2) ^ (2 ^ k) - 1)) = 1 / 2 :=
by sorry

end sum_series_l784_784976


namespace question2_question3_exists_value_of_m_l784_784286

def f (x : ℝ) : ℝ := x / (1 + x^2)

noncomputable def question1 (h₀ : ∀ x : ℝ, x ∈ set.Icc (-1) 1 → f x = -f (-x))
  (h₁ : f (1 / 3) = 3 / 10) :  f = λ x, x / (1 + x^2) :=
sorry

theorem question2 (x1 x2 : ℝ) (hx : x1 < x2 ∧ x1 ∈ set.Icc (-1) 1 ∧ x2 ∈ set.Icc (-1) 1) :
  f x1 < f x2 :=
sorry

theorem question3_exists_value_of_m (m : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (1/2) 1 → f (m*x - x) + f (x^2 - 1) > 0) ↔ 1 < m ∧ m ≤ 2 :=
sorry

end question2_question3_exists_value_of_m_l784_784286


namespace range_of_quadratic_l784_784581

theorem range_of_quadratic (x : ℝ) (h : x ∈ set.Ioo (-1 : ℝ) 3) : 
  set.range (λ (x : ℝ), (x - 2)^2) = set.Ico 0 9 :=
sorry

end range_of_quadratic_l784_784581


namespace sequence_term_37_l784_784921

theorem sequence_term_37 (n : ℕ) (h_pos : 0 < n) (h_eq : 3 * n + 1 = 37) : n = 12 :=
by
  sorry

end sequence_term_37_l784_784921


namespace theater_seats_l784_784485

theorem theater_seats (x y t : ℕ) (h1 : x = 532) (h2 : y = 218) (h3 : t = x + y) : t = 750 := 
by 
  rw [h1, h2] at h3
  exact h3

end theater_seats_l784_784485


namespace bob_cannot_end_with_one_28_bob_can_end_with_one_27_bob_cannot_end_with_one_29_l784_784097

-- Define the conditions as a function to check if Bob can end up with only one number
def can_bob_end_with_one (n : ℕ) : Prop :=
  ∃ f : fin n → ℕ, -- f represents the sequence of numbers initially written on the board
  ∀ a b c : fin n, -- ∀ a, b, c in the initial sequence
  (3 ∣ (f a + 2 * f b)) ∧ (3 ∣ (f a - f c)) → -- conditions for erasing a, b, c
  (∀ d : ℕ, d ≠ a + b + c) → -- the condition to transform into exactly one number
  ∃ k : ℕ, f (fin.mk 0 (nat.zero_lt_succ n)) = k

-- Prove for specific cases
theorem bob_cannot_end_with_one_28 : ¬ can_bob_end_with_one 28 := 
by {
  sorry
}

theorem bob_can_end_with_one_27 : can_bob_end_with_one 27 := 
by {
  sorry
}

theorem bob_cannot_end_with_one_29 : ¬ can_bob_end_with_one 29 := 
by {
  sorry
}

end bob_cannot_end_with_one_28_bob_can_end_with_one_27_bob_cannot_end_with_one_29_l784_784097


namespace number_of_sixes_correct_l784_784147

-- Define the conditions
def total_runs : ℕ := 120
def boundary_count : ℕ := 3
def boundary_runs : ℕ := boundary_count * 4
def runs_from_running : ℕ := total_runs / 2

-- Define the question and its verification statement
theorem number_of_sixes_correct :
  let runs_from_sixes := total_runs - boundary_runs - runs_from_running in
  runs_from_sixes / 6 = 8 :=
by
  -- Placeholder for the proof
  sorry

end number_of_sixes_correct_l784_784147


namespace distance_between_centers_l784_784778

-- Define the points P, Q, R in the plane
variable (P Q R : ℝ × ℝ)

-- Define the lengths PQ, PR, and QR
variable (PQ PR QR : ℝ)
variable (is_right_triangle : ∃ (a b c : ℝ), PQ = a ∧ PR = b ∧ QR = c ∧ a^2 + b^2 = c^2)

-- Define the inradii r1, r2, r3 for triangles PQR, RST, and QUV respectively
variable (r1 r2 r3 : ℝ)

-- Assume PQ = 90, PR = 120, and QR = 150
axiom PQ_length : PQ = 90
axiom PR_length : PR = 120
axiom QR_length : QR = 150

-- Define the centers O2 and O3 of the circles C2 and C3 respectively
variable (O2 O3 : ℝ × ℝ)

-- Assume the inradius length is 30 for the initial triangle
axiom inradius_PQR : r1 = 30

-- Assume the positions of the centers of C2 and C3
axiom O2_position : O2 = (15, 75)
axiom O3_position : O3 = (70, 10)

-- Use the distance formula to express the final result
theorem distance_between_centers : ∃ n : ℕ, dist O2 O3 = Real.sqrt (10 * n) ∧ n = 725 :=
by
  sorry

end distance_between_centers_l784_784778


namespace coefficient_x2_expansion_l784_784827

/-- The coefficient of x^2 in the expansion of (1 - 1/x^2) * (1 + x)^4 is 5 -/
theorem coefficient_x2_expansion :
  ∃ c: ℕ, c = 5 ∧ (∃ f g : polynomial ℚ, 
  f = polynomial.C (1 : ℚ) - polynomial.C (1 : ℚ) * polynomial.X ^ (-2) ∧
  g = (1 + polynomial.X)^4 ∧
  polynomial.coeff (f * g) 2 = c) :=
by
  sorry

end coefficient_x2_expansion_l784_784827


namespace hexagon_diagonals_l784_784160

theorem hexagon_diagonals : 
  let N := 6 in 
  let n := N * (N - 3) / 2 in 
  n = 9 := by
  sorry

end hexagon_diagonals_l784_784160


namespace paving_stone_proof_l784_784871

noncomputable def paving_stone_width (length_court : ℝ) (width_court : ℝ) 
                                      (num_stones: ℕ) (stone_length: ℝ) : ℝ :=
  let area_court := length_court * width_court
  let area_stone := stone_length * (area_court / (num_stones * stone_length))
  area_court / area_stone

theorem paving_stone_proof :
  paving_stone_width 50 16.5 165 2.5 = 2 :=
sorry

end paving_stone_proof_l784_784871


namespace base_addition_solution_l784_784549

def is_valid_base (b : ℕ) (digits : List ℕ) : Prop :=
  digits.All (λ d, d < b)

theorem base_addition_solution (b : ℕ) :
  b > 9 ∧ is_valid_base b [3, 0, 6, 4, 2, 9, 7] ∧
  let d306 := 3 * b^2 + 0 * b + 6 in
  let d429 := 4 * b^2 + 2 * b + 9 in
  let d743 := 7 * b^2 + 4 * b + 3 in
  d306 + d429 = d743
  → b = 12 :=
by
  sorry

end base_addition_solution_l784_784549


namespace roman_remy_gallons_l784_784070

theorem roman_remy_gallons (R : ℕ) (Remy_uses : 3 * R + 1 = 25) :
  R + (3 * R + 1) = 33 :=
by
  sorry

end roman_remy_gallons_l784_784070


namespace angle_of_elevation_proof_l784_784881

noncomputable def height_of_lighthouse : ℝ := 100

noncomputable def distance_between_ships : ℝ := 273.2050807568877

noncomputable def angle_of_elevation_second_ship : ℝ := 45

noncomputable def distance_from_second_ship := height_of_lighthouse

noncomputable def distance_from_first_ship := distance_between_ships - distance_from_second_ship

noncomputable def tanθ := height_of_lighthouse / distance_from_first_ship

noncomputable def angle_of_elevation_first_ship := Real.arctan tanθ

theorem angle_of_elevation_proof :
  angle_of_elevation_first_ship = 30 := by
    sorry

end angle_of_elevation_proof_l784_784881


namespace Frank_is_1_foot_taller_than_Pepe_l784_784975

variables (Pepe_height Frank_height : ℝ)

-- Conditions
def Pepe_height_def : Pepe_height = 4.5 := by sorry
def Frank_height_def : Frank_height = Pepe_height + 1 := by sorry

-- The theorem we need to prove
theorem Frank_is_1_foot_taller_than_Pepe : Frank_height - Pepe_height = 1 :=
by
  rw [Frank_height_def, Pepe_height_def]
  sorry

end Frank_is_1_foot_taller_than_Pepe_l784_784975


namespace car_more_miles_travel_after_modification_l784_784468

theorem car_more_miles_travel_after_modification :
  ∀ (miles_per_gallon : ℕ) (efficiency_factor : ℚ) (tank_capacity : ℕ),
  miles_per_gallon = 33 →
  efficiency_factor = 1.25 →
  tank_capacity = 16 →
  (miles_per_gallon * tank_capacity * efficiency_factor - miles_per_gallon * tank_capacity) = 132 :=
begin
  intros,
  sorry,
end

end car_more_miles_travel_after_modification_l784_784468


namespace pascal_arithmetic_l784_784806

-- Define factorial function
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Define Pascal's triangle entry in p-arithmetic
def P (a b p : ℕ) : ℕ :=
  fact a /(fact b * fact (a - b))

-- Translation of the problem into a Lean theorem statement
theorem pascal_arithmetic (p k : ℕ) (h₁ : k ≤ (p - 1) / 2) : 
  P (p - 1 - k) k p = (-1) ^ k * P (2 * k) k p := by
  sorry

end pascal_arithmetic_l784_784806


namespace inscribed_sphere_volume_l784_784163

theorem inscribed_sphere_volume (edge_length : ℝ) (h_edge : edge_length = 12) : 
  ∃ (V : ℝ), V = 288 * Real.pi :=
by
  sorry

end inscribed_sphere_volume_l784_784163


namespace problem_BD_l784_784248

variable (a b c : ℝ)

theorem problem_BD (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) :=
by
  sorry

end problem_BD_l784_784248


namespace intersection_M_N_l784_784604

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784604


namespace amber_total_cost_l784_784964

theorem amber_total_cost :
  let n_tp := 10 in
  let n_pt := 7 in
  let n_t := 3 in
  let c_tp := 1.50 in
  let c_pt := 2.00 in
  let c_t := 2.00 in
  let total_cost := n_tp * c_tp + n_pt * c_pt + n_t * c_t in
  total_cost = 35 :=
by
  rfl

end amber_total_cost_l784_784964


namespace floor_sum_lemma_l784_784820

theorem floor_sum_lemma (x : Fin 1004 → ℝ) 
  (h : ∀ n : Fin 1004, x n + (n : ℝ) + 1 = ∑ i : Fin 1004, x i + 1005) 
  : ⌊|∑ i : Fin 1004, x i|⌋ = 501 :=
sorry

end floor_sum_lemma_l784_784820


namespace rationalize_denominator_correct_l784_784059

noncomputable def rationalize_denominator : Prop := 
  (1 / (Real.cbrt 3 + Real.cbrt 27) = Real.cbrt 9 / 12)

theorem rationalize_denominator_correct : rationalize_denominator := 
  sorry

end rationalize_denominator_correct_l784_784059


namespace intersection_M_N_l784_784646

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784646


namespace find_p_q_l784_784205

theorem find_p_q (p q : ℤ)
  (h : (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) :
  p + q = 3 :=
sorry

end find_p_q_l784_784205


namespace Isabelle_ticket_cost_l784_784345

theorem Isabelle_ticket_cost :
  (∀ (week_salary : ℕ) (weeks_worked : ℕ) (brother_ticket_cost : ℕ) (brothers_saved : ℕ) (Isabelle_saved : ℕ),
  week_salary = 3 ∧ weeks_worked = 10 ∧ brother_ticket_cost = 10 ∧ brothers_saved = 5 ∧ Isabelle_saved = 5 →
  Isabelle_saved + (week_salary * weeks_worked) - ((brother_ticket_cost * 2) - brothers_saved) = 15) :=
by
  sorry

end Isabelle_ticket_cost_l784_784345


namespace syllogism_problem_l784_784693

theorem syllogism_problem
  (Class1SeniorYear2 : Type)
  (StudentsOnlyChildren : ∀ (s : Class1SeniorYear2), True)
  (AnHong : Class1SeniorYear2) :
  (AllStudentsOnlyChildrenHoward : True) ∧
  (AnHongIsAStudent : True) ∧
  (AnHongOnlyChild : True) := by
  sorry

end syllogism_problem_l784_784693


namespace sum_of_exterior_angles_of_convex_quadrilateral_l784_784127

theorem sum_of_exterior_angles_of_convex_quadrilateral:
  ∀ (α β γ δ : ℝ),
  (α + β + γ + δ = 360) → 
  (∀ (θ₁ θ₂ θ₃ θ₄ : ℝ),
    (θ₁ = 180 - α ∧ θ₂ = 180 - β ∧ θ₃ = 180 - γ ∧ θ₄ = 180 - δ) → 
    θ₁ + θ₂ + θ₃ + θ₄ = 360) := 
by 
  intros α β γ δ h1 θ₁ θ₂ θ₃ θ₄ h2
  rcases h2 with ⟨hα, hβ, hγ, hδ⟩
  rw [hα, hβ, hγ, hδ]
  linarith

end sum_of_exterior_angles_of_convex_quadrilateral_l784_784127


namespace paint_fraction_used_l784_784016

theorem paint_fraction_used (initial_paint: ℕ) (first_week_fraction: ℚ) (total_paint_used: ℕ) (remaining_paint_after_first_week: ℕ) :
  initial_paint = 360 →
  first_week_fraction = 1/3 →
  total_paint_used = 168 →
  remaining_paint_after_first_week = initial_paint - initial_paint * first_week_fraction →
  (total_paint_used - initial_paint * first_week_fraction) / remaining_paint_after_first_week = 1/5 := 
by
  sorry

end paint_fraction_used_l784_784016


namespace find_k_l784_784009

def long_distance (x y : ℝ) : ℝ :=
  max (| x |) (| y |)

def equidistant_points (p q : ℝ × ℝ) : Prop :=
  long_distance p.1 p.2 = long_distance q.1 q.2

theorem find_k (k : ℝ)
  (P : ℝ × ℝ := (-1, k + 3))
  (Q : ℝ × ℝ := (4, 4 * k - 3))
  (h : equidistant_points P Q) :
  k = 1 ∨ k = 2 :=
sorry

end find_k_l784_784009


namespace red_units_painting_l784_784588

noncomputable def min_red_unit_cubes (n : ℕ) : ℕ :=
  (n + 1) * n^2

theorem red_units_painting (n : ℕ) (h_pos : 0 < n) : 
  ∃ m : ℕ, m = (n + 1) * n^2 ∧
    (∀ w : ℕ, w < 26 * n^3 - m → ∃ r : ℕ, r ≤ m ∧ shares_vertex (w, r)) :=
  sorry

end red_units_painting_l784_784588


namespace carl_spends_108_dollars_l784_784984

theorem carl_spends_108_dollars
    (index_cards_per_student : ℕ := 10)
    (periods_per_day : ℕ := 6)
    (students_per_class : ℕ := 30)
    (cost_per_pack : ℕ := 3)
    (cards_per_pack : ℕ := 50) :
  let total_index_cards := index_cards_per_student * students_per_class * periods_per_day in
  let total_packs := total_index_cards / cards_per_pack in
  let total_cost := total_packs * cost_per_pack in
  total_cost = 108 := 
by
  sorry

end carl_spends_108_dollars_l784_784984


namespace prob_exactly_25_faces_show_six_prob_at_least_one_face_shows_one_expected_num_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_num_of_distinct_digits_on_surface_l784_784475

-- Condition: A cube made of 27 smaller dice
def num_smaller_cubes : Nat := 27
def num_visible_cubes : Nat := 26

-- a) Probability that exactly 25 of the outer faces show a six
theorem prob_exactly_25_faces_show_six:
  (26 : ℝ) = 31 / (2^(13 : ℝ) * 3^(18 : ℝ)) := sorry

-- b) Probability that there is at least one face showing a one
theorem prob_at_least_one_face_shows_one:
  1 - (5^6 / (2^2 * 3^18)) ≈ 0.99998992 := sorry

-- c) Expected number of sixes showing on the surface
theorem expected_num_of_sixes_on_surface:
  (6 * (1 / 6) + 12 * (1 / 3) + 8 * (1 / 2)) = 9 := sorry

-- d) Expected sum of numbers on the surface of the larger cube
theorem expected_sum_of_numbers_on_surface:
  (54 * 3.5) = 189 := sorry

-- e) Expected number of distinct numbers showing on the surface of the larger cube
theorem expected_num_of_distinct_digits_on_surface:
  6 - (5^6 / (2 * 3^17)) ≈ 5.99 := sorry

end prob_exactly_25_faces_show_six_prob_at_least_one_face_shows_one_expected_num_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_num_of_distinct_digits_on_surface_l784_784475


namespace max_area_AQRS_l784_784873

noncomputable def max_rectangle_area : ℝ :=
  let AB := 40
  let AC := 31
  let sinA := 1 / 5 
  let A := arcsin sinA 
  let theta := (45 - A / 2)
  620 * (1 + sinA)

theorem max_area_AQRS (AB AC : ℝ) (sinA : ℝ) (h1 : AB = 40) (h2 : AC = 31) (h3 : sinA = 1 / 5) :
  max_rectangle_area = 744 :=
  by
    sorry

end max_area_AQRS_l784_784873


namespace cone_surface_area_calculation_l784_784277

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

noncomputable def cone_surface_area (r l : ℝ) : ℝ :=
  π * r^2 + π * r * l

theorem cone_surface_area_calculation (r l h : ℝ)
  (volume_condition : cone_volume r h = 9 * real.sqrt 3 * π)
  (semicircle_condition : l = 2 * r)
  (height_condition : h = real.sqrt (4 * r^2 - r^2)) :
  cone_surface_area r l = 27 * π :=
by 
  sorry  -- Proof goes here

end cone_surface_area_calculation_l784_784277


namespace pizza_topping_combinations_l784_784947

theorem pizza_topping_combinations (T : Finset ℕ) (hT : T.card = 8) : 
  (T.card.choose 1 + T.card.choose 2 + T.card.choose 3 = 92) :=
by
  sorry

end pizza_topping_combinations_l784_784947


namespace min_minutes_to_make_B_cheaper_l784_784184

def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

def costB (x : ℕ) : ℕ := 2500 + 4 * x

theorem min_minutes_to_make_B_cheaper : ∃ (x : ℕ), x ≥ 301 ∧ costB x < costA x :=
by
  use 301
  sorry

end min_minutes_to_make_B_cheaper_l784_784184


namespace range_of_m_l784_784288

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) - m

noncomputable def two_zeros (m : ℝ) : Prop := 
  ∃ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi / 2 ∧ f x1 m = 0 ∧ f x2 m = 0

theorem range_of_m : {m : ℝ | two_zeros m} = set.Ico (1/2 : ℝ) 1 := 
  by sorry

end range_of_m_l784_784288


namespace product_of_undefined_roots_l784_784224

theorem product_of_undefined_roots :
  let f (x : ℝ) := (x^2 - 4*x + 4) / (x^2 - 5*x + 6)
  ∀ x : ℝ, (x^2 - 5*x + 6 = 0) → x = 2 ∨ x = 3 →
  (x = 2 ∨ x = 3 → x1 = 2 ∧ x2 = 3 → x1 * x2 = 6) :=
by
  sorry

end product_of_undefined_roots_l784_784224


namespace calc_inverse_l784_784361

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def c : ℚ := 1 / 12
def d : ℚ := 1 / 12

theorem calc_inverse :
  N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end calc_inverse_l784_784361


namespace proposition_2_proposition_4_correct_propositions_l784_784067

theorem proposition_2 (a : Line) (b : Line) (α : Plane) (h1 : a ∥ α) (h2 : b ⟂ α) : a ⟂ b := 
sorry

theorem proposition_4 (a : Line) (α : Plane) (β : Plane) (h1 : a ⟂ α) (h2 : a ∥ β) : α ⟂ β := 
sorry

theorem correct_propositions : Prop :=
  (proposition_2 ∧ proposition_4) := 
sorry

end proposition_2_proposition_4_correct_propositions_l784_784067


namespace cube_diagonal_length_l784_784861

-- Define the volume of a cube plus three times the total length of its edges equals 
-- twice its surface area, and prove that the long diagonal is 6√3

theorem cube_diagonal_length (s : ℝ) (h : s^3 + 3 * (12 * s) = 2 * (6 * s^2)) : 
  s = 6 → (s * Real.sqrt 3) = 6 * Real.sqrt 3 :=
by
  intro hs
  rw hs
  rfl


end cube_diagonal_length_l784_784861


namespace exists_unique_solution_l784_784216

theorem exists_unique_solution : ∀ a b : ℝ, 2 * (a ^ 2 + 1) * (b ^ 2 + 1) = (a + 1) * (b + 1) * (a * b + 1) ↔ (a, b) = (1, 1) := by
  sorry

end exists_unique_solution_l784_784216


namespace inradius_of_right_triangle_l784_784116

theorem inradius_of_right_triangle (AC BC AB : ℝ) (hAC : AC = 8) (hRatio : BC = 8 * sqrt 3 / 3) (hRightAngle : ∠ ABC = π / 2) :
  let area := (1 / 2) * AC * BC in
  let s := (AC + BC + AB) / 2 in
  let r := area / s in
  r = (96 * sqrt 3 - 8 * sqrt 3) / 141 :=
by
  sorry -- Proof to be completed

end inradius_of_right_triangle_l784_784116


namespace frog_probability_reaches_vertical_side_l784_784498

-- Define the conditions as assumptions in the Lean problem statement

variable (start : ℕ × ℕ)
variable (boundary : Set (ℕ × ℕ))
variable (jump_length : ℕ)
variable (direction_prob : ℕ → ℚ)
variable [Nontrivial ℚ] -- Ensures we work with nontrivial probabilities

-- Define the coordinates of the initial position and the boundary of the grid
noncomputable def initial_position := (2, 1)
noncomputable def grid_boundary := {(x, y) | (x = 0 ∨ x = 5) ∨ (y = 0 ∨ y = 5)}

-- Define the length of each jump and the probabilities of choosing each direction
def jump_length := 1
def direction_probability := 1 / 4

-- Define the conditions for the frog's movements and the stopping criteria
axiom initial_condition (x y : ℕ) : 
  (x, y) = initial_position → 
  ∀ (dx dy : ℕ), dx = 0 ∨ dx = 1 ∨ dx = -1 ∨ dy = 0 ∨ dy = 1 ∨ dy = -1

axiom boundary_condition (x y : ℕ) :
  (x, y) ∈ grid_boundary → 
  if x = 0 ∨ x = 5 then true 
  else false

-- Define the target mathematical statement proving the probability calculation
theorem frog_probability_reaches_vertical_side :
  let P : (ℕ × ℕ) → ℚ := λ p, if p ∈ grid_boundary then 1 else 0 in
  P initial_position = 13 / 24 := 
by
  -- Skipping the proof steps as instructed
  sorry

end frog_probability_reaches_vertical_side_l784_784498


namespace area_of_triangle_proof_l784_784743

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  0.5 * b * c * Real.sin (Real.arccos (-(b^2 + c^2 - a^2) / (2 * b * c)))

theorem area_of_triangle_proof
  (a b c : ℝ)
  (h1 : b + c = 8)
  (h2 : c + a = 10)
  (h3 : a + b = 12) :
  area_of_triangle a b c = 15 * Real.sqrt 3 / 4 :=
by
  -- In the actual proof, we'll solve b+c = 8, c+a = 10, and a+b = 12 to find a, b, and c.
  -- Then use cosine rule to find the area.
  sorry

end area_of_triangle_proof_l784_784743


namespace prob_A_prob_B_l784_784272

variable (a b : ℝ) -- Declare variables a and b as real numbers
variable (h_ab : a + b = 1) -- Declare the condition a + b = 1
variable (h_pos_a : 0 < a) -- Declare a is a positive real number
variable (h_pos_b : 0 < b) -- Declare b is a positive real number

-- Prove that 1/a + 1/b ≥ 4 under the given conditions
theorem prob_A (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Prove that a^2 + b^2 ≥ 1/2 under the given conditions
theorem prob_B (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a^2 + b^2 ≥ 1 / 2 :=
by
  sorry

end prob_A_prob_B_l784_784272


namespace sum_of_third_and_fifth_numbers_l784_784322

def sequence_b (n : ℕ) : ℚ
  | 1        := 2
  | (n + 1)  :=  ((n+1) / n : ℚ)^3

theorem sum_of_third_and_fifth_numbers : 
  (sequence_b 3 + sequence_b 5) = (341 / 64 : ℚ) :=
sorry

end sum_of_third_and_fifth_numbers_l784_784322


namespace seq_common_max_l784_784173

theorem seq_common_max : ∃ a, a ≤ 250 ∧ 1 ≤ a ∧ a % 8 = 1 ∧ a % 9 = 4 ∧ ∀ b, b ≤ 250 ∧ 1 ≤ b ∧ b % 8 = 1 ∧ b % 9 = 4 → b ≤ a :=
by 
  sorry

end seq_common_max_l784_784173


namespace describe_lines_through_D_l784_784295

-- Define a tetrahedron ABCD with points A, B, C, and D.
variables {A B C D : Type*}

-- Define the geometric conditions, assuming tetrahedron with vertices A, B, C, and D.
def is_tetrahedron (A B C D : Type*) : Prop :=
  ∃ a b c d: Type*, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Define the property of a line passing through D and intersecting the face of the tetrahedron.
def passes_through_and_intersects_face (x : Type*) (D : Type*) (face : Type*) : Prop :=
  ∃ l: Type*, l = x ∧ ∃ p: face, p ≠ D

-- Theorem statement: Determine the set of lines passing through D with the given property.
theorem describe_lines_through_D {A B C D x : Type*} :
  is_tetrahedron A B C D → 
  (passes_through_and_intersects_face x D ABC ∨
   passes_through_and_intersects_face x D ABD ∨ 
   passes_through_and_intersects_face x D ACD ∨ 
   passes_through_and_intersects_face x D BCD) → 
  ∃ l, passes_through_and_intersects_face l D ABC :=
begin
  sorry
end

end describe_lines_through_D_l784_784295


namespace inequality_proof_l784_784367

theorem inequality_proof
  (n : ℕ) (hn : n ≥ 3)
  (x : ℕ → ℝ) (hx : ∀ i, 1 ≤ i ∧ i < n → x i < x (i + 1)) :
  (n * (n - 1) / 2) * ∑ i in finset.range n, ∑ j in finset.range n, if i < j then x i * x j else 0 >
    (∑ i in finset.range (n - 1), (n - i) * x i) * (∑ j in finset.range (n - 1) \u [0], (j + 1) * x (j + 1)) :=
begin
  sorry
end

end inequality_proof_l784_784367


namespace intersection_M_N_l784_784600

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784600


namespace triangle_perimeter_l784_784011

variables (a b c : ℝ) (A : ℝ) (S : ℝ)

-- Conditions
axiom cond1 : a^2 - c^2 + 3 * b = 0
axiom cond2 : S = 5 * real.sqrt 3 / 2
axiom cond3 : A = real.pi / 3  -- 60 degrees in radians

-- Perimeter calculation statement
theorem triangle_perimeter : a^2 - c^2 + 3 * b = 0 → A = real.pi / 3 → S = 5 * real.sqrt 3 / 2 → a + b + c = 7 + real.sqrt 19 :=
begin
  intros cond1 cond2 cond3,
  sorry
end

end triangle_perimeter_l784_784011


namespace calculate_expression_l784_784186

theorem calculate_expression :
  (Int.floor ((15:ℚ)/8 * ((-34:ℚ)/4)) - Int.ceil ((15:ℚ)/8 * Int.floor ((-34:ℚ)/4))) = 0 := 
  by sorry

end calculate_expression_l784_784186


namespace dist_points_l784_784560

-- Define the points p1 and p2
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (4, 1)

-- Define the distance formula between the points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The theorem stating the distance between these points is 5
theorem dist_points : dist p1 p2 = 5 := by
  sorry

end dist_points_l784_784560


namespace smallest_n_for_divisible_sum_l784_784568

theorem smallest_n_for_divisible_sum :
  ∃ n : ℕ, (∀ (s : Finset ℕ), s.card = n → (∃ (t ⊆ s), t.card = 8 ∧ (t.sum id % 8 = 0))) ∧
  (∀ m : ℕ, (∀ (s : Finset ℕ), s.card = m → (∃ (t ⊆ s), t.card = 8 ∧ (t.sum id % 8 = 0))) → n ≤ m) :=
begin
  use 15,
  split,
  { intros s hs,
    sorry
  },
  { intros m hm,
    by_contradiction h,
    sorry
  }

end smallest_n_for_divisible_sum_l784_784568


namespace ap_contains_sixth_power_l784_784787

-- Definitions of distinct positive integers p and q, and arithmetic progression conditions
variable (p q : ℕ) (h : ℕ → ℕ) (a d : ℕ)

-- Conditions according to the problem
axiom distinct_pos_integers (h : ℕ → ℕ) (a d : ℕ) : p > 0 ∧ q > 0 ∧ p ≠ q
axiom ap_contains_p2_q3 (h : ℕ → ℕ) : ∃ n m : ℕ, h n = p^2 ∧ h m = q^3
axiom ap_is_arith_seq (h : ℕ → ℕ) (a d : ℕ) : ∀ n, h n = a + n * d ∧ a > 0 ∧ d > 0

-- Statement to prove
theorem ap_contains_sixth_power (p q : ℕ) (h : ℕ → ℕ) (a d : ℕ) :
  p > 0 ∧ q > 0 ∧ p ≠ q ∧ (∃ n m : ℕ, h n = p^2 ∧ h m = q^3) ∧ (∀ n, h n = a + n * d ∧ a > 0 ∧ d > 0) →
  ∃ z : ℕ, ∃ k : ℕ, h k = z^6 := 
begin
  sorry
end

end ap_contains_sixth_power_l784_784787


namespace red_balls_count_l784_784924

-- Lean 4 statement for proving the number of red balls in the bag is 336
theorem red_balls_count (x : ℕ) (total_balls red_balls : ℕ) 
  (h1 : total_balls = 60 + 18 * x) 
  (h2 : red_balls = 56 + 14 * x) 
  (h3 : (56 + 14 * x : ℚ) / (60 + 18 * x) = 4 / 5) : red_balls = 336 := 
by
  sorry

end red_balls_count_l784_784924


namespace coloring_problem_l784_784522

theorem coloring_problem : 
  ∃ k : ℕ, k = 4 ∧ ∀ (c : ℕ → ℕ), 
  (∀ m n, 2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 ∧ m ≠ n ∧ (m % n = 0) → c m ≠ c n) :=
begin
  sorry
end

end coloring_problem_l784_784522


namespace derivative_at_zero_l784_784972

def f (x : ℝ) : ℝ :=
  if x = 0 then 0 else x^2 * (Real.cos (11 / x))^2

theorem derivative_at_zero : 
  (by definition has_deriv_at f 0 0) :=
  by
    sorry

end derivative_at_zero_l784_784972


namespace interval_x_2x_3x_l784_784564

theorem interval_x_2x_3x (x : ℝ) :
  (2 * x > 1) ∧ (2 * x < 2) ∧ (3 * x > 1) ∧ (3 * x < 2) ↔ (x > 1 / 2) ∧ (x < 2 / 3) :=
by
  sorry

end interval_x_2x_3x_l784_784564


namespace neg_exists_eq_forall_l784_784706

theorem neg_exists_eq_forall {x : ℝ} : 
  ¬ (∃ x : ℝ, exp x - x - 1 ≤ 0) ↔ ∀ x : ℝ, exp x - x - 1 > 0 := 
by
  sorry

end neg_exists_eq_forall_l784_784706


namespace pentagon_area_l784_784330
open Real

structure Point :=
(x : ℝ)
(y : ℝ)

def line_through (P Q : Point) : Point → Prop :=
  λ R, (R.y - P.y) * (Q.x - P.x) = (R.x - P.x) * (Q.y - P.y)

def area_of_triangle (A B C : Point) : ℝ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

def area_of_pentagon (A B C D E : Point) : ℝ :=
  area_of_triangle A B C + area_of_triangle B D C - area_of_triangle B E C

theorem pentagon_area :
  let A : Point := {x := 0, y := 2}
  let B : Point := {x := 1, y := 7}
  let C : Point := {x := 10, y := 7}
  let D : Point := {x := 7, y := 1}
  let E : Point := {x := 4, y := 4}
  (line_through A C E) ∧ (line_through B D E) ∧ (area_of_pentagon A B C D E = 36) := 
begin
  sorry
end

end pentagon_area_l784_784330


namespace socks_problem_l784_784811

theorem socks_problem (a b : ℕ) (h : ∑ p, p = (207 : ℚ) / (38 : ℚ)) :
  a = 207 → b = 38 → 100 * a + b = 20738 :=
by
  intros ha hb
  rw [ha, hb]
  norm_num

# Reduce the main statement to the inputs to 207 and 38 representing a and b
example : socks_problem 207 38 :=
by
  simp only [socks_problem]
  norm_num

end socks_problem_l784_784811


namespace find_distance_between_vectors_l784_784435

noncomputable def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def cosine_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (norm v1 * norm v2)

noncomputable def u (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  norm v1 = 1 ∧ norm v2 = 1 ∧
  cosine_angle v1 ⟨3, -1, 2⟩ = real.cos (real.pi / 6) ∧
  cosine_angle v2 ⟨1, 2, 2⟩ = real.cos (real.pi / 4) ∧
  v1 ≠ v2

noncomputable def distance_between (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 + (v1.3 - v2.3)^2)

theorem find_distance_between_vectors :
  ∃(v1 v2 : ℝ × ℝ × ℝ), u v1 v2 → distance_between v1 v2 = (the specific distance calculated).
sorry

end find_distance_between_vectors_l784_784435


namespace general_term_of_sequence_l784_784793

noncomputable def a_sequence : ℕ → ℝ
| 1 := 3
| 2 := 8
| (n + 2) := 2 * a_sequence (n + 1) + 2 * a_sequence n

theorem general_term_of_sequence : 
  ∀ n : ℕ, a_sequence n = 
    (1 / 2 + real.sqrt 3 / 3) * (1 + real.sqrt 3) ^ n + 
    (1 / 2 - real.sqrt 3 / 3) * (1 - real.sqrt 3) ^ n :=
by
  sorry

end general_term_of_sequence_l784_784793


namespace gear_angular_speed_ratios_l784_784241

-- Define the problem assumptions and proof goal
theorem gear_angular_speed_ratios (p q r s : ℕ) (ω_A ω_B ω_C ω_D : ℝ)
  (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (hAB : p * ω_A = q * ω_B)
  (hBC : q * ω_B = r * ω_C)
  (hCD : r * ω_C = s * ω_D) :
  ω_A : ω_B : ω_C : ω_D = qs : ps : qs : pr := 
sorry

end gear_angular_speed_ratios_l784_784241


namespace length_of_road_l784_784823

-- Definitions based on conditions
def trees : Nat := 10
def interval : Nat := 10

-- Statement of the theorem
theorem length_of_road 
  (trees : Nat) (interval : Nat) (beginning_planting : Bool) (h_trees : trees = 10) (h_interval : interval = 10) (h_beginning : beginning_planting = true) 
  : (trees - 1) * interval = 90 := 
by 
  sorry

end length_of_road_l784_784823


namespace minimal_water_pipe_cost_l784_784111

namespace WaterPipes

-- Define the conditions in Lean

-- Number of villages
def n : ℕ := 10

-- Cost of thick pipe per kilometer
def thick_pipe_cost : ℕ := 8000

-- Cost of thin pipe per kilometer
def thin_pipe_cost : ℕ := 2000

-- Minimal cost calculated
def min_total_cost : ℕ := 414000

-- Statement of the problem in Lean
theorem minimal_water_pipe_cost
    (n_villages : ℕ)
    (thick_pipe_cost : ℕ)
    (thin_pipe_cost : ℕ)
    (min_cost : ℕ)
    (h_n : n_villages = n)
    (h_thick : thick_pipe_cost = 8000)
    (h_thin : thin_pipe_cost = 2000)
    (h_min : min_cost = min_total_cost) :
    Exists (λ cost : ℕ, cost = 414000) :=
begin
    -- Given the conditions, we want to show that the minimal cost is 414000
    sorry
end

end WaterPipes

end minimal_water_pipe_cost_l784_784111


namespace distance_between_circle_centers_l784_784324

theorem distance_between_circle_centers
  (ABCD : Type) [quadrilateral ABCD]
  (M N : ABCD)
  (circle1 center1 : Type) [circle circle1]
  (circle2 center2 : Type) [circle circle2]
  (tangent_MN_circle1 : tangent M N circle1)
  (tangent_MN_circle2 : tangent M N circle2)
  (p a r : Real)
  (h_perimeter : perimeter_ABC_D M B C N = 2 * p)
  (bc_eq_a : segment B C = a)
  (radius_diff_eq_r : |radius circle1 - radius circle2| = r) :
  distance center1 center2 = Real.sqrt (r^2 + (p - a)^2) := 
sorry

end distance_between_circle_centers_l784_784324


namespace hannah_probability_l784_784300

noncomputable def probability_exactly_x_green_marble (total_trials green_marbles purple_marbles x : ℕ) : ℚ :=
  let total_marbles := green_marbles + purple_marbles
  let choose : ℚ := nat.choose total_trials x
  let single_favorable_probability : ℚ := ((green_marbles : ℚ) / (total_marbles : ℚ))^x * 
                                          ((purple_marbles : ℚ) / (total_marbles : ℚ))^(total_trials - x)
  choose * single_favorable_probability

theorem hannah_probability : 
  probability_exactly_x_green_marble 8 6 4 3 = 154828 / 125000 :=
by
  sorry

end hannah_probability_l784_784300


namespace triangle_hypotenuse_l784_784875

noncomputable def length_hypotenuse {DE DF DP PE DQ QF QE PF : ℕ} : ℝ :=
  (DE ^ 2 + DF ^ 2 : ℕ).sqrt

theorem triangle_hypotenuse
  (DE DF : ℝ)
  (h_ratios1 : DP / (DP + PE) = 1 / 4)
  (h_ratios2 : DQ / (DQ + QF) = 1 / 4)
  (h_lengths1 : (QE : ℝ) = 18)
  (h_lengths2 : (PF : ℝ) = 30)
  (h_eq1 : ((DE / 4) ^ 2 + DF ^ 2 = 18 ^ 2))
  (h_eq2 : ((DE / 4 + DE) ^ 2 + (DF / 4) ^ 2 = 30 ^ 2)) :
  EF = 24 * (3 : ℝ).sqrt :=
begin
  sorry -- The proof is omitted as per the instruction.
end

end triangle_hypotenuse_l784_784875


namespace great_dane_more_than_triple_pitbull_l784_784494

variables (C P G : ℕ)
variables (h1 : G = 307) (h2 : P = 3 * C) (h3 : C + P + G = 439)

theorem great_dane_more_than_triple_pitbull
  : G - 3 * P = 10 :=
by
  sorry

end great_dane_more_than_triple_pitbull_l784_784494


namespace sum_of_a_to_an_eq_neg2_l784_784021

theorem sum_of_a_to_an_eq_neg2
  (a a1 a2 ... an : ℝ)
  (h : ∀ x, (x^2 + 1) * (2*x + 1)^9 = a + a1 * (x + 2) + a2 * (x + 2)^2 + ... + an * (x + 2)^n) :
  a + a1 + a2 + ... + an = -2 :=
by
  sorry

end sum_of_a_to_an_eq_neg2_l784_784021


namespace solve_for_x_l784_784730

theorem solve_for_x (x : ℝ) (h : real.sqrt (5 + real.sqrt x) = 4) : x = 121 := 
sorry

end solve_for_x_l784_784730


namespace range_of_m_l784_784696

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + (m + 2) * x + (m + 5) = 0 → 0 < x) → (-5 < m ∧ m ≤ -4) :=
by
  sorry

end range_of_m_l784_784696


namespace solve_quadratic_l784_784239

theorem solve_quadratic (k : ℝ) (x : ℝ) (hk : k = 2) : 
  ((k - 2) * x^2 + 4 * k * x - 5 = 0) ↔ x = 5 / 8 :=
by begin
  sorry
end

end solve_quadratic_l784_784239


namespace probability_A_seventh_week_l784_784492

/-
Conditions:
1. There are four different passwords: A, B, C, and D.
2. Each week, one of these passwords is used.
3. Each week, the password is chosen at random and equally likely from the three passwords that were not used in the previous week.
4. Password A is used in the first week.

Goal:
Prove that the probability that password A will be used in the seventh week is 61/243.
-/

def prob_password_A_in_seventh_week : ℚ :=
  let Pk (k : ℕ) : ℚ := 
    if k = 1 then 1
    else if k >= 2 then ((-1 / 3)^(k - 1) * (3 / 4) + 1 / 4) else 0
  Pk 7

theorem probability_A_seventh_week : prob_password_A_in_seventh_week = 61 / 243 := by
  sorry

end probability_A_seventh_week_l784_784492


namespace angle_bisector_theorem_angle_bisector_length_l784_784765

-- Question 1: Prove that AD is the angle bisector of ∠BAC in ΔABC implies BD / DC = BA / AC
theorem angle_bisector_theorem (A B C D : Type) [IsTriangle A B C]
  (h: is_angle_bisector D A B C) : 
  BD / DC = BA / AC := 
sorry

-- Question 2: Prove the length of the angle bisector AD (denoted as t_a) is ta = (2bc / (b + c)) * cos (A / 2)
theorem angle_bisector_length (A B C D : Type) [IsTriangle A B C] :
  length_AD (A B C) = (2 * length_B * length_C / (length_B + length_C)) * cos (A / 2) := 
sorry

end angle_bisector_theorem_angle_bisector_length_l784_784765


namespace min_value_expression_l784_784362

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 :=
sorry

end min_value_expression_l784_784362


namespace binary_arithmetic_l784_784210

theorem binary_arithmetic :
  let a := 0b11101
  let b := 0b10011
  let c := 0b101
  (a * b) / c = 0b11101100 :=
by
  sorry

end binary_arithmetic_l784_784210


namespace width_of_paving_stone_l784_784870

-- Given conditions as definitions
def length_of_courtyard : ℝ := 40
def width_of_courtyard : ℝ := 16.5
def number_of_stones : ℕ := 132
def length_of_stone : ℝ := 2.5

-- Define the total area of the courtyard
def area_of_courtyard := length_of_courtyard * width_of_courtyard

-- Define the equation we need to prove
theorem width_of_paving_stone :
  (length_of_stone * W * number_of_stones = area_of_courtyard) → W = 2 :=
by
  sorry

end width_of_paving_stone_l784_784870


namespace percentage_increase_chips_l784_784506

theorem percentage_increase_chips :
  ∃ P : ℕ, P = 75 ∧
  (let price_pretzel := 4
       price_chip := price_pretzel + (P / 100) * price_pretzel in
  2 * price_chip + 2 * price_pretzel = 22) :=
begin
  sorry
end

end percentage_increase_chips_l784_784506


namespace average_study_difference_is_6_l784_784320

def study_time_differences : List ℤ := [15, -5, 25, -10, 40, -30, 10]

def total_sum (lst : List ℤ) : ℤ := lst.foldr (· + ·) 0

def number_of_days : ℤ := 7

def average_difference : ℤ := total_sum study_time_differences / number_of_days

theorem average_study_difference_is_6 : average_difference = 6 :=
by
  unfold average_difference
  unfold total_sum 
  sorry

end average_study_difference_is_6_l784_784320


namespace cube_long_diagonal_length_l784_784860

noncomputable def cube_side_length_satisfying_conditions : Real :=
  let s : Real := 6
  if h : s^3 + 36 * s = 12 * s^2 then
    s
  else
    0 -- this should not occur since s=6 satisfies the conditions

theorem cube_long_diagonal_length :
  let s : Real := cube_side_length_satisfying_conditions
  s ≠ 0 → (s * Real.sqrt 3) = 6 * Real.sqrt 3 :=
by
  intro s_ne_zero
  let s : Real := 6
  have : s = 6 := rfl
  calc
    (s * Real.sqrt 3)
      = 6 * Real.sqrt 3 : by rw [this]
  sorry

end cube_long_diagonal_length_l784_784860


namespace x_intercept_perpendicular_line_l784_784451

theorem x_intercept_perpendicular_line 
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 12)
  (h2 : y = - (3 / 4) * x + 4)
  : x = 16 / 3 := 
sorry

end x_intercept_perpendicular_line_l784_784451


namespace intersection_M_N_eq_neg2_l784_784638

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784638


namespace polynomial_is_perfect_square_l784_784240

theorem polynomial_is_perfect_square (k : ℚ) :
    k = 11 / 3 →
    ∃ (p : ℚ), (λ x : ℚ, x^2 + 2*(k-9)*x + (k^2 + 3*k + 4)) = (λ x : ℚ, (x + p) * (x + p)) :=
by
  intro h
  rw h
  use (2 / 3)
  funext x
  calc
    x^2 + 2 * (11 / 3 - 9) * x + ((11 / 3)^2 + 3 * (11 / 3) + 4)
      = x^2 + 2 * (11 / 3 - 27 / 3) * x + (121 / 9 + 33 / 3 + 4) : by norm_num
  ... = x^2 + 2 * (-16 / 3) * x + (121 / 9 + 99 / 9 + 36 / 9) : by norm_num
  ... = x^2 - 32 / 3 * x + 28 : by norm_num
  ... = (x - 16 / 3) ^ 2 : by ring

end polynomial_is_perfect_square_l784_784240


namespace curve_C_cartesian_equation_curve_C_intersection_line_l_l784_784001

-- Definitions according to the provided conditions
def curve_C_polar (ρ θ : ℝ) : Prop := ρ^2 * real.cos (2 * θ) + 4 = 0
def line_l (t : ℝ) := (t, real.sqrt 5 + 2 * t)
def point_A := (0, real.sqrt 5)

-- Cartesian equation of curve C 
theorem curve_C_cartesian_equation :
  ∀ (x y : ℝ), (∃ ρ θ, x = ρ * real.cos θ ∧ y = ρ * real.sin θ ∧ curve_C_polar ρ θ) ↔ y^2 - x^2 = 4 :=
by sorry

-- Intersection of line l with curve C and the required value calculation
theorem curve_C_intersection_line_l (AM AN : ℝ → ℝ) :
  (∀ t, line_l t ∈ {p : ℝ × ℝ | p.2^2 - p.1^2 = 4}) →
  (∀ t, AM t = real.sqrt ((line_l t).1^2 + ((line_l t).2 - (point_A).2)^2)) →
  (∀ t, AN t = real.sqrt ((line_l t).1^2 + ((line_l t).2 - (point_A).2)^2)) →
  ∃ t1 t2, t1 ≠ 0 ∧ t2 ≠ 0 ∧ 1 / |AM t1| + 1 / |AN t2| = 4 :=
by sorry

end curve_C_cartesian_equation_curve_C_intersection_line_l_l784_784001


namespace unique_real_root_in_interval_l784_784304

theorem unique_real_root_in_interval (a : ℝ) (h : a > 3) :
  ∃! x ∈ Ioo 0 2, x^3 - a * x^2 + 1 = 0 :=
by
  -- sorry to skip the proof part
  sorry

end unique_real_root_in_interval_l784_784304


namespace percentage_of_water_in_dried_grapes_l784_784243

theorem percentage_of_water_in_dried_grapes 
  (weight_fresh : ℝ) 
  (weight_dried : ℝ) 
  (percentage_water_fresh : ℝ) 
  (solid_weight : ℝ)
  (water_weight_dried : ℝ) 
  (percentage_water_dried : ℝ) 
  (H1 : weight_fresh = 30) 
  (H2 : weight_dried = 15) 
  (H3 : percentage_water_fresh = 0.60) 
  (H4 : solid_weight = weight_fresh * (1 - percentage_water_fresh)) 
  (H5 : water_weight_dried = weight_dried - solid_weight) 
  (H6 : percentage_water_dried = (water_weight_dried / weight_dried) * 100) 
  : percentage_water_dried = 20 := 
  by { sorry }

end percentage_of_water_in_dried_grapes_l784_784243


namespace problem1_problem2_l784_784282

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x)
  else x - 4 / x

theorem problem1 (h : ∀ x : ℝ, f 1 x = 3 → x = 4) : ∃ x : ℝ, f 1 x = 3 ∧ x = 4 :=
sorry

theorem problem2 (h : ∀ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) →
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a ≤ -1 → 
  a = -11 / 6) : ∃ a : ℝ, a ≤ -1 ∧ (∃ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) ∧ 
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a = -11 / 6) :=
sorry

end problem1_problem2_l784_784282


namespace gallons_of_soup_l784_784209

def bowls_per_minute : ℕ := 5
def ounces_per_bowl : ℕ := 10
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem gallons_of_soup :
  (5 * 10 * 15 / 128) = 6 :=
by
  sorry

end gallons_of_soup_l784_784209


namespace difference_in_height_l784_784529

-- Define the heights of the sandcastles
def h_J : ℚ := 3.6666666666666665
def h_S : ℚ := 2.3333333333333335

-- State the theorem
theorem difference_in_height :
  h_J - h_S = 1.333333333333333 := by
  sorry

end difference_in_height_l784_784529


namespace linear_function_third_quadrant_and_origin_l784_784736

theorem linear_function_third_quadrant_and_origin (k b : ℝ) (h1 : ∀ x < 0, k * x + b ≥ 0) (h2 : k * 0 + b ≠ 0) : k < 0 ∧ b > 0 :=
sorry

end linear_function_third_quadrant_and_origin_l784_784736


namespace M_inter_N_eq_neg2_l784_784669

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784669


namespace find_y_l784_784733

def G (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y (y : ℕ) : G 3 y 6 8 = 300 → y = 5 :=
by
  intro h
  have : G 3 5 6 8 = 300 := rfl
  exact sorry

end find_y_l784_784733


namespace sum_difference_l784_784390

-- Define Set A: all odd numbers between 1 and 100 inclusive
def SetA : Set ℕ := {n | 1 ≤ n ∧ n ≤ 100 ∧ n % 2 = 1}

-- Define Set B: all odd numbers between 103 and 200 inclusive
def SetB : Set ℕ := {n | 103 ≤ n ∧ n ≤ 200 ∧ n % 2 = 1}

-- Define a function to compute the sum of elements in a finite set of natural numbers
def sumOfSet (s : Set ℕ) : ℕ := Finset.sum (s.toFinset) id

-- Prove that the difference between the sum of Set B and Set A is 4899
theorem sum_difference : sumOfSet SetB - sumOfSet SetA = 4899 := 
by
  sorry

end sum_difference_l784_784390


namespace metallic_sheet_length_l784_784157

/-- Given conditions:
  width of the metallic sheet is 36 meters,
  a square of side 3 meters is cut off from each corner,
  the volume of the resulting box is 3780 cubic meters,
  we need to prove that the length of the metallic sheet is 48 meters. -/
theorem metallic_sheet_length (width : ℕ) (cut_length : ℕ) (box_volume : ℕ) (length : ℕ) 
  (h_width : width = 36) (h_cut_length : cut_length = 3) (h_box_volume : box_volume = 3780) :
  length = 48 :=
by {
  have h_new_width := width - 2 * cut_length,
  have h_new_length := length - 2 * cut_length,
  have h_height := cut_length,
  rw [h_width, h_cut_length, h_box_volume] at *,
  have h_volume := h_new_length * h_new_width * h_height,
  rw [←h_box_volume, h_volume],
  sorry
}

end metallic_sheet_length_l784_784157


namespace triangle_identity_proof_l784_784012

noncomputable def triangle_identity (A B C α β : ℝ) (h : α + β = A) : Prop :=
  cos α * cos B + cos β * cos C = sin α * sin B + sin β * sin C

theorem triangle_identity_proof (A B C α β : ℝ) (h : α + β = A) :
  triangle_identity A B C α β h :=
by
  sorry

end triangle_identity_proof_l784_784012


namespace sum_reciprocal_greatest_power_three_l784_784358

theorem sum_reciprocal_greatest_power_three (k : ℕ) (h_pos : k > 0) :
  let b_n (n : ℕ) := Nat.gcd (3^k) (Nat.choose (3^k) n)
  in ∑ n in Finset.range (3^k), (n > 0).to_smul (1 / b_n n).to_real = (5 / 3)^k - 1 := by
  sorry

end sum_reciprocal_greatest_power_three_l784_784358


namespace unacceptable_quality_l784_784956

variable (weight : ℝ) (acceptable_range_start : ℝ) (acceptable_range_end : ℝ)

theorem unacceptable_quality (h1 : acceptable_range_start = 49.7)
                            (h2 : acceptable_range_end = 50.3)
                            (h3 : weight = 49.6) :
  weight < acceptable_range_start ∨ weight > acceptable_range_end :=
by
  sorry

# Ensure the theorem works for the given conditions
example : unacceptable_quality 49.7 50.3 49.6 := by 
  apply unacceptable_quality
  repeat { rfl }
  sorry

end unacceptable_quality_l784_784956


namespace sum_reciprocals_less_than_l784_784593

-- Definitions and conditions
def a_n (n : Nat) : ℤ := 2 * n - 1
def S_n (n : Nat) : ℤ := n * n

-- Statement to prove
theorem sum_reciprocals_less_than (n : ℕ) : ∑ k in finset.range (n + 1), (1 : ℚ) / S_n k < 5 / 3 := sorry

end sum_reciprocals_less_than_l784_784593


namespace disjoint_subsets_same_sum_l784_784888

/-- 
Given a set of 10 distinct integers between 1 and 100, 
there exist two disjoint subsets of this set that have the same sum.
-/
theorem disjoint_subsets_same_sum : ∃ (x : Finset ℤ), (x.card = 10) ∧ (∀ i ∈ x, 1 ≤ i ∧ i ≤ 100) → 
  ∃ (A B : Finset ℤ), (A ⊆ x) ∧ (B ⊆ x) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by
  sorry

end disjoint_subsets_same_sum_l784_784888


namespace cone_volume_correct_l784_784276

noncomputable def cone_volume (lateral_area : ℝ) (central_angle : ℝ) : ℝ :=
  let l := Real.sqrt 5 in
  let r := 1 in
  let h := 2 in
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_correct :
  cone_volume (Real.sqrt 5 * Real.pi) ((2 * Real.sqrt 5 * Real.pi) / 5) = (2 * Real.pi) / 3 :=
by
  sorry

end cone_volume_correct_l784_784276


namespace sum_of_series_eq_half_l784_784979

theorem sum_of_series_eq_half :
  (∑' k : ℕ, 3^(2^k) / (9^(2^k) - 1)) = 1 / 2 :=
by
  sorry

end sum_of_series_eq_half_l784_784979


namespace transformation_impossible_l784_784936

-- Definitions for the transformations and invariant quantity
def valid_transformation (x y : ℕ) : Prop :=
  (x + 1 ≤ 9 ∧ y + 1 ≤ 9) ∨ (x - 1 ≥ 0 ∧ y - 1 ≥ 0)

def invariant_quantity (a b c d : ℕ) : ℕ :=
  (d + b) - (a + c)

-- Initial and target numbers
def initial_number : ℕ × ℕ × ℕ × ℕ := (1, 2, 3, 4)
def target_number : ℕ × ℕ × ℕ × ℕ := (2, 0, 0, 2)

-- Statement of the theorem
theorem transformation_impossible :
  let (a₁, b₁, c₁, d₁) := initial_number
  let (a₂, b₂, c₂, d₂) := target_number
  invariant_quantity a₁ b₁ c₁ d₁ ≠ invariant_quantity a₂ b₂ c₂ d₂ → 
  ¬∃ (seq : list ((ℕ × ℕ) × (ℕ × ℕ))),
    (∀ (p : (ℕ × ℕ) × (ℕ × ℕ)), p ∈ seq → valid_transformation (p.1.1) (p.1.2) ∧ valid_transformation (p.2.1) (p.2.2)) ∧
    (let (a, b, c, d) := initial_number
     list.foldl (λ num trans, 
       let (a, b, c, d) := num in
       if valid_transformation a b then (a + 1, b + 1, c, d)
       else if valid_transformation b c then (a, b + 1, c + 1, d)
       else if valid_transformation c d then (a, b, c + 1, d + 1)
       else (a, b, c, d)
     ) (a₁, b₁, c₁, d₁) seq = (a₂, b₂, c₂, d₂))) :=
by
  intros h1
  let initial := initial_number
  let target := target_number
  sorry

end transformation_impossible_l784_784936


namespace extreme_point_sum_l784_784579

variable {a : ℝ}

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*a*x) * Real.log x + 2*a*x - (1/2) * x^2

theorem extreme_point_sum (h : a > 0 ∧ a ≠ 1) (x1 x2 : ℝ) (hx : x1 < x2) (hx1: f' x1 = 0) (hx2: f' x2 = 0) : 
  f x1 + f x2 < (1/2) * a^2 + 3 * a := 
by sorry

end extreme_point_sum_l784_784579


namespace smallest_n_satisfying_conditions_l784_784960

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n ≡ 1 [MOD 7] ∧ n ≡ 1 [MOD 4] ∧ n = 113 :=
by
  sorry

end smallest_n_satisfying_conditions_l784_784960


namespace min_v_value_l784_784786

def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 2 then 1
else if 2 < x ∧ x ≤ 3 then x - 1
else 0

def v (a : ℝ) : ℝ :=
  let fs := λ x, f x - a * x in
  (λ s : Set ℝ, (s.sup fs - s.inf fs)) (Set.Icc 1 3)

theorem min_v_value : (Set.range v).inf' ⟨0.5, sorry⟩ = 0.5 :=
sorry

end min_v_value_l784_784786


namespace cheetah_passed_after_deer_l784_784496

-- Definitions for speeds and times
def deer_speed : ℝ := 50
def cheetah_speed : ℝ := 60
def catchup_time : ℝ := 1 / 60 -- 1 minute expressed in hours

-- Prove that the cheetah passed the tree 4 minutes after the deer
theorem cheetah_passed_after_deer :
  ∀ t : ℝ, (deer_speed / 60 + (deer_speed / 60) * t = cheetah_speed * t) → (t = 1 / 12) → (t - catchup_time = 4 / 60) :=
by
  intros t h_eq h_t
  rw [h_t] at h_eq
  sorry

end cheetah_passed_after_deer_l784_784496


namespace equation_of_circle_unique_l784_784562

noncomputable def equation_of_circle := 
  ∃ (d e f : ℝ), 
    (4 + 4 + 2*d + 2*e + f = 0) ∧ 
    (25 + 9 + 5*d + 3*e + f = 0) ∧ 
    (9 + 1 + 3*d - e + f = 0) ∧ 
    (∀ (x y : ℝ), x^2 + y^2 + d*x + e*y + f = 0 → (x = 2 ∧ y = 2) ∨ (x = 5 ∧ y = 3) ∨ (x = 3 ∧ y = -1))

theorem equation_of_circle_unique :
  equation_of_circle := sorry

end equation_of_circle_unique_l784_784562


namespace sum_of_solutions_eq_neg3_l784_784571

noncomputable def f (x : ℝ) : ℝ := |x - 3| - 3 * |x + 1|

theorem sum_of_solutions_eq_neg3 : 
  (∃ x : ℝ, f x = 0) → (∑ x in {x | f x = 0}, x) = -3 :=
sorry

end sum_of_solutions_eq_neg3_l784_784571


namespace ellipse_problem_l784_784260

noncomputable def ellipse_equation (a b : ℝ) (h1 : 2 * a = 4) (h2 : (b^2 + 1^2 = a^2)) : Prop :=
  (a = 2) ∧ (b = Real.sqrt 3) ∧ (C1_eq : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) = (x^2 / 4 + y^2 / 3 = 1))

noncomputable def trajectory_equation : Prop :=
  ∀ x y, (y^2 = 4 * x)

noncomputable def min_area_quad (a b : ℝ) (h1 : 2 * a = 4) (h2 : (b^2 + 1^2 = a^2)) : Prop :=
  S_min : ∀ M N P Q (h3 : ∃ M N P Q, collinear (M,F2) ∧ collinear (N,F2) ∧ 
                     collinear (P,F2) ∧ collinear (Q,F2) ∧ (dot (P - F2) (M - F2) = 0)), 
                     area_PMQR = 8

theorem ellipse_problem (a b : ℝ) (h1 : 2 * a = 4) (h2 : (b^2 + 1^2 = a^2)) :
  (ellipse_equation a b h1 h2) ∧
  (trajectory_equation) ∧
  (min_area_quad a b h1 h2) := 
begin
  sorry
end

end ellipse_problem_l784_784260


namespace M_inter_N_eq_neg2_l784_784663

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784663


namespace flu_virus_diameter_in_scientific_notation_l784_784274

theorem flu_virus_diameter_in_scientific_notation :
  (0.000000815 : ℝ) = 8.15 * 10^(-7) :=
sorry

end flu_virus_diameter_in_scientific_notation_l784_784274


namespace prove_ordered_triple_l784_784784

theorem prove_ordered_triple (x y z : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : z > 2)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) : 
  (x, y, z) = (13, 11, 6) :=
sorry

end prove_ordered_triple_l784_784784


namespace circumcenter_barycentric_incenter_barycentric_orthocenter_barycentric_l784_784219

-- Definitions of the angles at vertices
variables (α β γ : ℝ)

-- Definitions of side lengths opposite to vertices
variables (a b c : ℝ)

-- Definition of circumcenter barycentric coordinates
def barycentric_circumcenter : ℝ × ℝ × ℝ :=
  (Real.sin (2 * α), Real.sin (2 * β), Real.sin (2 * γ))

-- Definition of incenter barycentric coordinates
def barycentric_incenter : ℝ × ℝ × ℝ :=
  (a, b, c)

-- Definition of orthocenter barycentric coordinates
def barycentric_orthocenter : ℝ × ℝ × ℝ :=
  (Real.tan α, Real.tan β, Real.tan γ)

-- Theorems asserting the barycentric coordinates of the points
theorem circumcenter_barycentric (α β γ : ℝ) : barycentric_circumcenter α β γ = (Real.sin (2 * α), Real.sin (2 * β), Real.sin (2 * γ)) :=
  sorry

theorem incenter_barycentric (a b c : ℝ) : barycentric_incenter a b c = (a, b, c) :=
  sorry

theorem orthocenter_barycentric (α β γ : ℝ) : barycentric_orthocenter α β γ = (Real.tan α, Real.tan β, Real.tan γ) :=
  sorry

end circumcenter_barycentric_incenter_barycentric_orthocenter_barycentric_l784_784219


namespace sum_of_x_coordinates_l784_784043

def exists_common_point (x : ℕ) : Prop :=
  (3 * x + 5) % 9 = (7 * x + 3) % 9

theorem sum_of_x_coordinates :
  ∃ x : ℕ, exists_common_point x ∧ x % 9 = 5 := 
by
  sorry

end sum_of_x_coordinates_l784_784043


namespace sum_min_max_z_l784_784368

theorem sum_min_max_z (x y : ℝ) 
  (h1 : x - y - 2 ≥ 0) 
  (h2 : x - 5 ≤ 0) 
  (h3 : y + 2 ≥ 0) :
  ∃ (z_min z_max : ℝ), z_min = 2 ∧ z_max = 34 ∧ z_min + z_max = 36 :=
by
  sorry

end sum_min_max_z_l784_784368


namespace guessing_number_is_nine_l784_784464

theorem guessing_number_is_nine :
  ∃ n : ℕ, (n ≥ 1 ∧ n ≤ 99) ∧
           (square n ∧ n < 5 → ∃ qi_true : Prop, 
             ∃ lu_true : Prop, 
             ∃ dai_true : Prop,
             ((qi_true = true ∧ lu_true = false ∧ dai_true = false) ∨ 
             (qi_true = false ∧ lu_true = false ∧ dai_true = true) ∨
             (qi_true = false ∧ lu_true = true ∧ dai_true = false)) 
             ∧ ((qi_true = true → square n ∧ n < 5) 
             ∧ (qi_true = false → ¬(square n ∧ n < 5)) 
             ∧ (lu_true = true → ¬(n < 7 ∧ (n > 9 ∧ n < 100))) 
             ∧ (lu_true = false → ¬(n > 9 ∧ n < 100)) 
             ∧ (dai_true = true → square n ∧ ¬ (n < 5) ∧ (¬ (n < 7) ∨ (¬ (n > 9 ∧ n < 100))))) := 
  sorry

end guessing_number_is_nine_l784_784464


namespace area_inside_circle_proof_l784_784083

noncomputable def triangle_area_inside_circle
  (area_ABC : ℝ)
  (angle_A : ℝ)
  (mid_AC_eq_O : ℝ → ℝ → Prop)
  (circle_center_O_tangent_BC : Prop)
  (circle_intersect_MN_AM_eq_NB : Prop) : ℝ :=
  if h : area_ABC = 1 ∧ angle_A = arctan (3/4) ∧ mid_AC_eq_O = (λ x y, 2 * x = y) 
     ∧ circle_center_O_tangent_BC 
     ∧ circle_intersect_MN_AM_eq_NB 
  then (π / 3) - (2 / 3) * arccos (3 / 4) + (sqrt 7 / 8)
  else 0

theorem area_inside_circle_proof
  (area_ABC : ℝ)
  (angle_A : ℝ)
  (mid_AC_eq_O : ℝ → ℝ → Prop)
  (circle_center_O_tangent_BC : Prop)
  (circle_intersect_MN_AM_eq_NB : Prop)
  (h : area_ABC = 1 ∧ angle_A = arctan (3/4) ∧ mid_AC_eq_O = (λ x y, 2 * x = y) 
       ∧ circle_center_O_tangent_BC 
       ∧ circle_intersect_MN_AM_eq_NB) :
  triangle_area_inside_circle area_ABC angle_A mid_AC_eq_O circle_center_O_tangent_BC circle_intersect_MN_AM_eq_NB =
  (π / 3) - (2 / 3) * arccos (3 / 4) + (sqrt 7 / 8) :=
by
  sorry  -- Proof goes here

end area_inside_circle_proof_l784_784083


namespace vector_properties_l784_784710

-- Definitions
def a := (Real.sqrt 2, 2)
def b_magnitude := 2 * Real.sqrt 3
def c_magnitude := 2 * Real.sqrt 6
def parallel (u v : ℝ × ℝ) := ∃ k : ℝ, u = (k * v.1, k * v.2)
def perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

-- Conditions
axiom h1 : a = (Real.sqrt 2, 2)
axiom h2 : ∥b∥ = b_magnitude
axiom h3 : ∥c∥ = c_magnitude
axiom h4 : parallel a c
axiom h5 : perpendicular (a - b) (3 * a + 2 * b)

-- The Math Proof Problem
theorem vector_properties :
  ∃ k : ℝ, 
        (∥c - a∥ = 2 * (Real.sqrt 6 + k * Real.sqrt 2)) 
    ∧ (a • (a + b + c) = 12 ∨ a • (a + b + c) = -12) := sorry

end vector_properties_l784_784710


namespace clock_correction_calculation_l784_784153

noncomputable def clock_correction : ℝ :=
  let daily_gain := 5/4
  let hourly_gain := daily_gain / 24
  let total_hours := (9 * 24) + 9
  let total_gain := total_hours * hourly_gain
  total_gain

theorem clock_correction_calculation : clock_correction = 11.72 := by
  sorry

end clock_correction_calculation_l784_784153


namespace percentage_books_not_sold_books_not_sold_percentage_l784_784349

-- defining the initial parameters
def initial_stock : ℕ := 700
def initial_fiction_stock : ℕ := 400
def initial_non_fiction_stock : ℕ := 300

def fiction_sold_mon : ℕ := 50
def fiction_sold_wed : ℕ := 60
def fiction_sold_fri : ℕ := 40

def non_fiction_sold_tue : ℕ := 82
def non_fiction_sold_wed : ℕ := 10
def non_fiction_sold_thu : ℕ := 48
def non_fiction_sold_fri : ℕ := 20

def fiction_returned_sat : ℕ := 30
def non_fiction_returned_sat : ℕ := 15

-- proving the final percentage of books not sold
theorem percentage_books_not_sold :
  (initial_stock - ((fiction_sold_mon + fiction_sold_wed + fiction_sold_fri - fiction_returned_sat) +
  (non_fiction_sold_tue + non_fiction_sold_wed + non_fiction_sold_thu + non_fiction_sold_fri - non_fiction_returned_sat))
  : ℕ ) = 435 :=
sorry

theorem books_not_sold_percentage : 
  ((initial_stock - ((fiction_sold_mon + fiction_sold_wed + fiction_sold_fri - fiction_returned_sat) +
  (non_fiction_sold_tue + non_fiction_sold_wed + non_fiction_sold_thu + non_fiction_sold_fri - non_fiction_returned_sat))) * 100 / initial_stock) = 62.14 :=
sorry

end percentage_books_not_sold_books_not_sold_percentage_l784_784349


namespace distance_diff_is_0_point3_l784_784351

def john_walk_distance : ℝ := 0.7
def nina_walk_distance : ℝ := 0.4
def distance_difference_john_nina : ℝ := john_walk_distance - nina_walk_distance

theorem distance_diff_is_0_point3 : distance_difference_john_nina = 0.3 :=
by
  -- proof goes here
  sorry

end distance_diff_is_0_point3_l784_784351


namespace equation_of_ellipse_range_of_m_l784_784280

-- Definitions based on the conditions
structure Ellipse where
  a b c : ℝ
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (ecc : c / a = Float.sqrt 3 / 2)

def ellipse_intersection_line (e : Ellipse) :=
  ∃ (A B : ℝ × ℝ), (line : ℝ → ℝ) (line_eq : line = λ x, x) ∧
  (mem_A : (∃ x, (line x, x) = A ∧ (x/a)^2 + (x/b)^2 = 1)) ∧
  (mem_B : (∃ x, (line x, x)= B ∧ (x/a)^2 + (x/b)^2 = 1)) ∧
  (right_vertex : ℝ × ℝ) (right_vertex_val : right_vertex = (e.a, 0)) ∧ 
  (four_norm : (2 * N.norm right_vertex) = 4)

-- Statement 1: Proving the equation of the ellipse given these conditions
theorem equation_of_ellipse (e : Ellipse) :  
  (ellipse_intersection_line e) →  (e.a = 2 ∧ e.b = 1 → ∀ x y: ℝ, (x/2)^2 + y^2 = 1) :=
sorry

-- Definitions for part II conditions
def line_ell_intersection (e : Ellipse) (k m : ℝ) := 
  ∃ M N Q : ℝ × ℝ, 
    (line : ℝ → ℝ) (line_eq : line = λ x, k*x + m) ∧
    (mem_M : ∃ x, (line x, x) = M ∧ (x/e.a)^2 + (line x/e.b)^2 = 1) ∧
    (mem_N : ∃ x, (line x, x) = N ∧ (x/e.a)^2 + (line x/e.b)^2 = 1) ∧
    (Q_val : Q = (0, -½)) ∧ 
    (equal_dists : N.norm (Q - M) = N.norm (Q - N))

-- Statement 2: Proving the range of the real number m
theorem range_of_m (e : Ellipse) (k : ℝ) : 
  (line_ell_intersection e k m) → 
  ( ∀ k : ℝk ≠ 0 ∧ m ≠ 0 ∧ 4k^2 > m^2 - 1 → 1/6 < m ∧ m < 6 ) :=
sorry

end equation_of_ellipse_range_of_m_l784_784280


namespace towel_percentage_decrease_l784_784140

theorem towel_percentage_decrease
  (L B: ℝ)
  (original_area : ℝ := L * B)
  (new_length : ℝ := 0.70 * L)
  (new_breadth : ℝ := 0.75 * B)
  (new_area : ℝ := new_length * new_breadth) :
  ((original_area - new_area) / original_area) * 100 = 47.5 := 
by 
  sorry

end towel_percentage_decrease_l784_784140


namespace find_polynomial_l784_784222

noncomputable def P (a : ℂ) (x : ℂ) : ℂ := a * x^3 - a * x^2

theorem find_polynomial (a : ℂ) (h : a ≠ 0) :
  let P := P a in P (P x) = (x^3 + x^2 + x + 1) * P x := 
by 
  sorry

end find_polynomial_l784_784222


namespace no_longer_1000_consecutive_integers_after_move_l784_784489

theorem no_longer_1000_consecutive_integers_after_move (a : ℕ → ℤ) (h_a : ∀ i, a (i + 1) = a i + 1)
    (h0 : ∀ i < 1000, a i < a (i + 1)) :
  ¬(∃ b : ℕ → ℤ, (∀ i < 1000, b (i + 1) = b i + 1) ∧ after_some_moves a b) :=
sorry

end no_longer_1000_consecutive_integers_after_move_l784_784489


namespace eccentricity_of_ellipse_l784_784705

-- Conditions: Parametric equations
def parametric_x (θ : ℝ) : ℝ := 4 * Real.cos θ
def parametric_y (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the Cartesian coordinate equation derived from parametric equations
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 4) = 1

-- Prove the eccentricity of the ellipse
theorem eccentricity_of_ellipse : 
    (∃ θ : ℝ, parametric_x θ = 4 * Real.cos θ ∧ parametric_y θ = 2 * Real.sin θ) → 
    (parametric_x θ)^2 / 16 + (parametric_y θ)^2 / 4 = 1 → 
    (∃ e : ℝ, e = (Real.sqrt 3) / 2) :=
by
  intro,
  sorry

end eccentricity_of_ellipse_l784_784705


namespace smallest_b_value_l784_784783

theorem smallest_b_value (p q : ℤ) 
  (b : ℝ)
  (h1 : ∀ x y : ℝ, (x + 3) ^ 2 + (y - 4) ^ 2 = 36)
  (h2 : ∀ x y : ℝ, (x - 3) ^ 2 + (y - 4) ^ 2 = 4)
  (h_tangent : ∀ x y : ℝ, (x - 3) ^ 2 + (y - 4) ^ 2 = (x + 3) ^ 2 + (y - 4) ^ 2)
  (h_line : y = b * x ∧ (x = 0 ∧ y = 4))
  (rel_prime : p.gcd q = 1)
  (h_b_squared : b ^ 2 = p / q)
  (min_b : b > 0) : p + q = 2 :=
by
s

end smallest_b_value_l784_784783


namespace M_inter_N_eq_neg2_l784_784666

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784666


namespace group_for_2019_is_63_l784_784850

def last_term_of_group (n : ℕ) : ℕ := (n * (n + 1)) / 2 + n

theorem group_for_2019_is_63 :
  ∃ n : ℕ, (2015 < 2019 ∧ 2019 ≤ 2079) :=
by
  sorry

end group_for_2019_is_63_l784_784850


namespace intersection_M_N_l784_784649

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784649


namespace angle_ratio_l784_784760

theorem angle_ratio (A B P Q M : Type) 
  (h1 : bisects P B (∠ ABC)) 
  (h2 : bisects Q B (∠ ABC)) 
  (h3 : bisects M B (∠ PBQ)) : 
  (measure_angle M B Q / measure_angle A B Q) = 1 / 4 := 
sorry

end angle_ratio_l784_784760


namespace clock_ticks_6_times_at_6_oclock_l784_784178

theorem clock_ticks_6_times_at_6_oclock
  (h6 : 5 * t = 25)
  (h12 : 11 * t = 55) :
  t = 5 ∧ 6 = 6 :=
by
  sorry

end clock_ticks_6_times_at_6_oclock_l784_784178


namespace complex_integer_sum_of_squares_l784_784131

theorem complex_integer_sum_of_squares (x y : ℤ) :
  (∃ a b c d e f : ℤ, x + (y:ℤ) * complex.I = (a + b * complex.I)^2 + (c + d * complex.I)^2 + (e + f * complex.I)^2)
  ↔ even y := 
sorry

end complex_integer_sum_of_squares_l784_784131


namespace john_cards_sum_l784_784350

theorem john_cards_sum :
  ∃ (g : ℕ → ℕ) (y : ℕ → ℕ),
    (∀ n, (g n) ∈ [1, 2, 3, 4, 5]) ∧
    (∀ n, (y n) ∈ [2, 3, 4, 5]) ∧
    (∀ n, (g n < g (n + 1))) ∧
    (∀ n, (y n < y (n + 1))) ∧
    (∀ n, (g n ∣ y (n + 1) ∨ y (n + 1) ∣ g n)) ∧
    (g 0 = 1 ∧ g 2 = 2 ∧ g 4 = 5) ∧
    ( y 1 = 2 ∧ y 3 = 3 ∧ y 5 = 4 ) →
  g 0 + g 2 + g 4 = 8 := by
sorry

end john_cards_sum_l784_784350


namespace triangle_leg_ratios_l784_784891

theorem triangle_leg_ratios (λ : ℝ) :
  (λ = 1 ∨ λ = sqrt ((sqrt 5 + 1) / 2)) :=
  sorry

end triangle_leg_ratios_l784_784891


namespace smallest_value_a_b_c_l784_784884

-- Define the given summation
def T_50 : ℚ := ∑ k in finset.range 50, (-1 : ℚ)^k * (k^3 + k^2 + k + 1) / k!

-- Define our target values
def a : ℕ := 2603
def b : ℕ := 50
def c : ℕ := 1

-- State the Lean theorem to be proven
theorem smallest_value_a_b_c : T_50 = a / b! - c ∧ a + b + c = 2654 :=
by
  -- The transformation and verification steps would go here...
  sorry

end smallest_value_a_b_c_l784_784884


namespace longest_sequence_positive_integer_x_l784_784538

theorem longest_sequence_positive_integer_x :
  ∃ x : ℤ, 0 < x ∧ 34 * x - 10500 > 0 ∧ 17000 - 55 * x > 0 ∧ x = 309 :=
by
  use 309
  sorry

end longest_sequence_positive_integer_x_l784_784538


namespace cost_of_seven_books_l784_784438

theorem cost_of_seven_books (h : 3 * 12 = 36) : 7 * 12 = 84 :=
sorry

end cost_of_seven_books_l784_784438


namespace problem1_tangent_line_eq_problem2_range_of_a_l784_784290

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := (1 / x) - a

-- Define the specific case of the tangent line problem at a = 2 and point (1, f(1))
theorem problem1_tangent_line_eq (x y : ℝ) (a : ℝ) (h : a = 2) (hx : x = 1) :
  f 2 1 = -2 ∧ f' 2 1 = -1 → (x + y + 1 = 0) :=
begin
  sorry
end

-- Define the condition where f(x) < 0 for all x in (0, +∞),
-- and find the range of values for a
theorem problem2_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < +∞ → f a x < 0) → a > (1 / Real.exp 1) :=
begin
  sorry
end

end problem1_tangent_line_eq_problem2_range_of_a_l784_784290


namespace sample_variance_l784_784513

noncomputable theory
open_locale classical

def sample := [1, 3, 2, 2, 3, 3, 0]

def mean (l : list ℤ) : ℚ :=
  (l.sum : ℚ) / (l.length : ℚ)

def variance (l : list ℤ) : ℚ :=
  let m := mean l in
  (l.map (λ x, (x : ℚ) - m) ^ 2).sum / (l.length : ℚ)

theorem sample_variance : variance sample = 8 / 7 :=
  by
  sorry

end sample_variance_l784_784513


namespace max_largest_int_of_avg_and_diff_l784_784738

theorem max_largest_int_of_avg_and_diff (A B C D E : ℕ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D) (h4 : D ≤ E) 
  (h_avg : (A + B + C + D + E) / 5 = 70) (h_diff : E - A = 10) : E = 340 :=
by
  sorry

end max_largest_int_of_avg_and_diff_l784_784738


namespace problem1_solution_set_l784_784313

theorem problem1_solution_set (a : ℝ) (x : ℝ) :
  let f := λ x, a * x^2 - (2 * a + 1) * x + 2 in
  ((a < 0 ∧ f x > 0 ↔ (1/a < x ∧ x < 2)) ∨
   (a = 0 ∧ f x > 0 ↔ (x < 2)) ∨
   (0 < a ∧ a < 1/2 ∧ f x > 0 ↔ ((x < 2) ∨ (1/a < x))) ∨
   (a = 1/2 ∧ f x > 0 ↔ (x ≠ 2)) ∨
   (a > 1/2 ∧ f x > 0 ↔ ((x < 1/a) ∨ (x > 2)))) := by
  sorry

end problem1_solution_set_l784_784313


namespace not_divisible_l784_784927

theorem not_divisible (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 12) : ¬∃ k : ℕ, 120 * a + 2 * b = k * (100 * a + b) := 
sorry

end not_divisible_l784_784927


namespace find_S10_l784_784268

-- Definitions of conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
-- Condition: {a_n} is a geometric sequence with a_n > 0
-- Note: In Lean, a geometric sequence can be defined using a common ratio r
variable (r : ℝ)
variable (h_geometric : ∀ n, a (n + 1) = r * a n)
variable (h_pos : ∀ n, a n > 0)

-- Definition of the sum of the first n terms of the sequence
def S (n : ℕ) := ∑ i in finset.range (n + 1), a i

-- Conditions: S_5 = 2 and S_15 = 14
variable (h_S5 : S 5 = 2)
variable (h_S15 : S 15 = 14)

-- Proof goal: Find S_10
theorem find_S10 : S 10 = 6 :=
sorry

end find_S10_l784_784268


namespace exists_increasing_sequence_l784_784047

theorem exists_increasing_sequence (a1 : ℕ) (h : a1 > 1) :
  ∃ (a : ℕ → ℕ), (∀ n, a n < a (n+1)) ∧ (∀ k, (∑ i in finset.range (k+1), (a i)^2) % (∑ i in finset.range (k+1), a i) = 0) :=
begin
  sorry
end

end exists_increasing_sequence_l784_784047


namespace distance_to_other_focus_l784_784262

theorem distance_to_other_focus (x y : ℝ) (P : ℝ × ℝ) (a b c : ℝ) (F1 F2 : ℝ × ℝ) :
  (a = 4) ∧ (b = 2) ∧ (c = √(a^2 - b^2)) ∧
  (P ∈ {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 4) = 1}) ∧
  (dist P F1 = 3) ∧
  (F1 = (c, 0)) ∧ (F2 = (-c, 0)) →
  dist P F2 = 5 :=
by
  -- Proof is omitted
  sorry

end distance_to_other_focus_l784_784262


namespace ball_placement_problem_l784_784383

noncomputable def count_valid_arrangements : ℕ := 
  let total_ways := (Finset.choose 4 2) * (Finset.perm 3 3)
  let invalid_ways := 6
  total_ways - invalid_ways

theorem ball_placement_problem : count_valid_arrangements = 30 := by
  -- This is where we would provide the proof steps if necessary
  sorry

end ball_placement_problem_l784_784383


namespace intersection_M_N_eq_neg2_l784_784636

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784636


namespace anne_carries_16point5_kg_l784_784967

theorem anne_carries_16point5_kg :
  let w1 := 2
  let w2 := 1.5 * w1
  let w3 := 2 * w1
  let w4 := w1 + w2
  let w5 := (w1 + w2) / 2
  w1 + w2 + w3 + w4 + w5 = 16.5 :=
by {
  sorry
}

end anne_carries_16point5_kg_l784_784967


namespace intersection_M_N_is_correct_l784_784616

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784616


namespace initial_books_in_bin_l784_784149

theorem initial_books_in_bin
  (x : ℝ)
  (h : x + 33.0 + 2.0 = 76) :
  x = 41.0 :=
by 
  -- Proof goes here
  sorry

end initial_books_in_bin_l784_784149


namespace greatest_five_digit_sum_l784_784360

theorem greatest_five_digit_sum (M : ℕ) (h1 : nat.digits 10 M).length = 5 (h2 : (nat.digits 10 M).prod = 180) (h3 : ∀ N, (nat.digits 10 N).length = 5 → (nat.digits 10 N).prod = 180 → N ≤ M) :
  (nat.digits 10 M).sum = 20 :=
sorry

end greatest_five_digit_sum_l784_784360


namespace maximum_area_of_equilateral_triangle_l784_784424

theorem maximum_area_of_equilateral_triangle (sides_lengths: ℝ × ℝ) (a_max_area: ℝ)
  (h_rect_len1: sides_lengths.1 = 12) 
  (h_rect_len2: sides_lengths.2 = 5) 
  (h_max_area: a_max_area = (25 * Real.sqrt 3) / 4) :
  ∃ (s: ℝ), 
  ∀ (eq_triangle_area: ℝ), 
    eq_triangle_area = (s^2 * Real.sqrt 3) / 4 → 
    s ≤ 5 → 
    eq_triangle_area ≤ a_max_area := 
begin
  sorry
end

end maximum_area_of_equilateral_triangle_l784_784424


namespace intersection_M_N_l784_784629

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784629


namespace problem_1_problem_2_l784_784580

noncomputable def f (x : ℝ) : ℝ := 2 * sin (real.pi - x) + cos (-x) - sin (5 * real.pi / 2 - x) + cos (real.pi / 2 + x)

theorem problem_1 (α : ℝ) (h1 : f(α) = 2/3) (h2 : 0 < α ∧ α < real.pi) : 
  tan α = 2 * (real.sqrt 5) / 5 := sorry

theorem problem_2 (α : ℝ) (h1 : f(α) = 2 * sin α - cos α + 3 / 4) : 
  sin α * cos α = 7 / 32 := sorry

end problem_1_problem_2_l784_784580


namespace devin_teaching_years_l784_784550

section DevinTeaching
variable (Calculus Algebra Statistics Geometry DiscreteMathematics : ℕ)

theorem devin_teaching_years :
  Calculus = 4 ∧
  Algebra = 2 * Calculus ∧
  Statistics = 5 * Algebra ∧
  Geometry = 3 * Statistics ∧
  DiscreteMathematics = Geometry / 2 ∧
  (Calculus + Algebra + Statistics + Geometry + DiscreteMathematics) = 232 :=
by
  sorry
end DevinTeaching

end devin_teaching_years_l784_784550


namespace distance_to_other_focus_is_5_l784_784264

theorem distance_to_other_focus_is_5 (P : ℝ × ℝ)
  (hP : (P.1^2 / 16) + (P.2^2 / 4) = 1)
  (h_focus_distance : real.sqrt (16 - 4) = 3) :
  ∃ d : ℝ, d = 5 :=
sorry

end distance_to_other_focus_is_5_l784_784264


namespace measure_of_angle_C_l784_784764

variable (A B C : ℕ)

theorem measure_of_angle_C :
  (A = B - 20) →
  (C = A + 40) →
  (A + B + C = 180) →
  C = 80 :=
by
  intros h1 h2 h3
  sorry

end measure_of_angle_C_l784_784764


namespace number_of_zeros_in_square_l784_784196

theorem number_of_zeros_in_square (n : ℕ) (h : n = 7) :
  let num := 10^n - 5
  let square := num * num
  ∃ (z : ℕ), z = 6 ∧ z = (square.to_digits.filter (λ x, x = 0)).length :=
by
  sorry

end number_of_zeros_in_square_l784_784196


namespace problem_l784_784307

theorem problem (k : ℕ) (h : 0 < k) : (∑ i in finset.range k, k)^k = k^(2 * k) := 
by 
  sorry

end problem_l784_784307


namespace calculate_expression_l784_784779

def f (x : ℝ) := x^2 + 3
def g (x : ℝ) := 2 * x + 4

theorem calculate_expression : f (g 2) - g (f 2) = 49 := by
  sorry

end calculate_expression_l784_784779


namespace no_integer_solutions_l784_784527

theorem no_integer_solutions :
  ∀ n m : ℤ, (n^2 + (n+1)^2 + (n+2)^2) ≠ m^2 :=
by
  intro n m
  sorry

end no_integer_solutions_l784_784527


namespace product_of_undefined_roots_l784_784225

theorem product_of_undefined_roots :
  let f (x : ℝ) := (x^2 - 4*x + 4) / (x^2 - 5*x + 6)
  ∀ x : ℝ, (x^2 - 5*x + 6 = 0) → x = 2 ∨ x = 3 →
  (x = 2 ∨ x = 3 → x1 = 2 ∧ x2 = 3 → x1 * x2 = 6) :=
by
  sorry

end product_of_undefined_roots_l784_784225


namespace simplify_logarithmic_expression_l784_784073

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem simplify_logarithmic_expression :
  simplify_expression = 4 / 3 :=
by
  sorry

end simplify_logarithmic_expression_l784_784073


namespace rationalize_denominator_l784_784053

theorem rationalize_denominator :
  (1 / (Real.cbrt 3 + Real.cbrt (3^3))) = (Real.cbrt 9 / 12) :=
by {
  sorry
}

end rationalize_denominator_l784_784053


namespace M_inter_N_eq_neg2_l784_784667

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784667


namespace james_hours_per_day_l784_784347

theorem james_hours_per_day (h : ℕ) (rental_rate : ℕ) (days_per_week : ℕ) (weekly_income : ℕ)
  (H1 : rental_rate = 20)
  (H2 : days_per_week = 4)
  (H3 : weekly_income = 640)
  (H4 : rental_rate * days_per_week * h = weekly_income) :
  h = 8 :=
sorry

end james_hours_per_day_l784_784347


namespace jasmine_stops_at_S_l784_784042

-- Definitions of the given conditions
def circumference : ℕ := 60
def total_distance : ℕ := 5400
def quadrants : ℕ := 4
def laps (distance circumference : ℕ) := distance / circumference
def isMultiple (a b : ℕ) := b ∣ a
def onSamePoint (distance circumference : ℕ) := (distance % circumference) = 0

-- The theorem to be proved: Jasmine stops at point S after running the total distance
theorem jasmine_stops_at_S 
  (circumference : ℕ) (total_distance : ℕ) (quadrants : ℕ)
  (h1 : circumference = 60) 
  (h2 : total_distance = 5400)
  (h3 : quadrants = 4)
  (h4 : laps total_distance circumference = 90)
  (h5 : isMultiple total_distance circumference)
  : onSamePoint total_distance circumference := 
  sorry

end jasmine_stops_at_S_l784_784042


namespace coordinates_of_point_B_l784_784278

theorem coordinates_of_point_B (A B : ℝ × ℝ) (AB : ℝ) :
  A = (-1, 2) ∧ B.1 = -1 ∧ AB = 3 ∧ (B.2 = 5 ∨ B.2 = -1) :=
by
  sorry

end coordinates_of_point_B_l784_784278


namespace range_f_elements_count_l784_784132

noncomputable def f (x : ℝ) : ℝ :=
  (floor x) + (floor (2 * x)) + (floor ((5/3) * x)) + (floor (3 * x)) + (floor (4 * x))

theorem range_f_elements_count :
  (set.fin_range.of (f '' (set.Icc (0 : ℝ) 100))).card = 734 :=
sorry

end range_f_elements_count_l784_784132


namespace inequality_solution_l784_784709

theorem inequality_solution (a b c : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 7 → a * x^2 + b * x + c > 0) →
  (∀ x : ℝ, (x < -1/7 ∨ x > 1/4) ↔ c * x^2 - b * x + a > 0) :=
by
  sorry

end inequality_solution_l784_784709


namespace total_roses_planted_three_days_l784_784437

-- Definitions based on conditions
def susan_roses_two_days_ago : ℕ := 10
def maria_roses_two_days_ago : ℕ := 2 * susan_roses_two_days_ago
def john_roses_two_days_ago : ℕ := susan_roses_two_days_ago + 10
def roses_two_days_ago : ℕ := susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago

def roses_yesterday : ℕ := roses_two_days_ago + 20
def susan_roses_yesterday : ℕ := susan_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def maria_roses_yesterday : ℕ := maria_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def john_roses_yesterday : ℕ := john_roses_two_days_ago * roses_yesterday / roses_two_days_ago

def roses_today : ℕ := 2 * roses_two_days_ago
def susan_roses_today : ℕ := susan_roses_two_days_ago
def maria_roses_today : ℕ := maria_roses_two_days_ago + (maria_roses_two_days_ago * 25 / 100)
def john_roses_today : ℕ := john_roses_two_days_ago - (john_roses_two_days_ago * 10 / 100)

def total_roses_planted : ℕ := 
  (susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago) +
  (susan_roses_yesterday + maria_roses_yesterday + john_roses_yesterday) +
  (susan_roses_today + maria_roses_today + john_roses_today)

-- The statement that needs to be proved
theorem total_roses_planted_three_days : total_roses_planted = 173 := by 
  sorry

end total_roses_planted_three_days_l784_784437


namespace angle_BCN_eq_angle_ADM_l784_784788

-- Given conditions
variables {ω₁ ω₂ : Circle}
variables {M N A B C D P : Point}
variables {PA PC : Line}
variables {PM : Line} -- Tangent at M
variables {PN : Line} -- Tangent at N
variables {MN : Segment}

-- Assume necessary conditions
axiom ω₁_no_common_points_ω₂ : ω₁.noCommonPoints ω₂
axiom ω₁_not_in_ω₂ : ¬ inCircle ω₁ ω₂
axiom point_on_ω₁ : M ∈ ω₁
axiom point_on_ω₂ : N ∈ ω₂
axiom tangents_intersect_P : tangent ω₁ M PM ∧ tangent ω₂ N PN ∧ intersect PM PN P
axiom isosceles_triangle_PM_PN : PM = PN
axiom ω₁_meet_MN_A : intersects ω₁ MN A
axiom ω₂_meet_MN_B : intersects ω₂ MN B
axiom PA_meets_ω₁_C : meets PA ω₁ C
axiom PB_meets_ω₂_D : meets PB ω₂ D

-- The proof problem
theorem angle_BCN_eq_angle_ADM : ∠BCN = ∠ADM :=
sorry

end angle_BCN_eq_angle_ADM_l784_784788


namespace total_area_of_figure_l784_784825

theorem total_area_of_figure :
  let base := 4
  let height := 4
  let area_of_one_triangle := (1 / 2 : ℝ) * base * height
  let number_of_triangles := 3
  let total_area := number_of_triangles * area_of_one_triangle
  total_area = 24 :=
by
  unfold base height area_of_one_triangle number_of_triangles total_area
  sorry

end total_area_of_figure_l784_784825


namespace train_speed_correct_l784_784955

-- Define the conditions as Lean definitions
def distance_train : ℝ := 1  -- in km
def distance_tunnel : ℝ := 70  -- in km
def entrance_time : ℝ := 5 * 60 + 12  -- in minutes since midnight
def exit_time : ℝ := 5 * 60 + 18  -- in minutes since midnight

-- Define time difference
def time_taken : ℝ := (exit_time - entrance_time) / 60  -- converting to hours

-- Define total distance
def total_distance : ℝ := distance_train + distance_tunnel

-- Define expected speed
def expected_speed : ℝ := 710 -- in km/h

-- Lean statement to prove the train speed
theorem train_speed_correct :
  (total_distance / time_taken) = expected_speed :=
sorry

end train_speed_correct_l784_784955


namespace beth_sister_age_l784_784183

theorem beth_sister_age :
  ∃ (x : ℕ), 18 + 8 = 2 * (x + 8) ∧ x = 5 :=
begin
  use 5,
  split,
  {
    calc 18 + 8 = 26 : by norm_num
          ... = 2 * (5 + 8) : by norm_num,
  },
  {
    refl,
  }
end

end beth_sister_age_l784_784183


namespace intersection_complement_l784_784359

def A : Set ℝ := { x | x^2 ≤ 4 * x }
def B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 3) }

theorem intersection_complement (x : ℝ) : 
  x ∈ A ∩ (Set.univ \ B) ↔ x ∈ Set.Ico 0 3 := 
sorry

end intersection_complement_l784_784359


namespace education_expenses_l784_784957

noncomputable def totalSalary (savings : ℝ) (savingsPercentage : ℝ) : ℝ :=
  savings / savingsPercentage

def totalExpenses (rent milk groceries petrol misc : ℝ) : ℝ :=
  rent + milk + groceries + petrol + misc

def amountSpentOnEducation (totalSalary totalExpenses savings : ℝ) : ℝ :=
  totalSalary - (totalExpenses + savings)

theorem education_expenses :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let petrol := 2000
  let misc := 700
  let savings := 1800
  let savingsPercentage := 0.10
  amountSpentOnEducation (totalSalary savings savingsPercentage) 
                          (totalExpenses rent milk groceries petrol misc) 
                          savings = 2500 :=
by
  sorry

end education_expenses_l784_784957


namespace find_reflection_line_l784_784874

-- Definition of the original and reflected vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := {x := 1, y := 2}
def E : Point := {x := 6, y := 7}
def F : Point := {x := -5, y := 5}
def D' : Point := {x := 1, y := -4}
def E' : Point := {x := 6, y := -9}
def F' : Point := {x := -5, y := -7}

theorem find_reflection_line (M : ℝ) :
  (D.y + D'.y) / 2 = M ∧ (E.y + E'.y) / 2 = M ∧ (F.y + F'.y) / 2 = M → M = -1 :=
by
  intros
  sorry

end find_reflection_line_l784_784874


namespace solve_x_y_l784_784066

theorem solve_x_y (x y : ℝ) (h1 : x^2 + y^2 = 16 * x - 10 * y + 14) (h2 : x - y = 6) : 
  x + y = 3 := 
by 
  sorry

end solve_x_y_l784_784066


namespace solve_equation_l784_784393

theorem solve_equation : 
  ∀ x : ℝ,
    (x + 5 ≠ 0) → 
    (x^2 + 3 * x + 4) / (x + 5) = x + 6 → 
    x = -13 / 4 :=
by 
  intro x
  intro hx
  intro h
  sorry

end solve_equation_l784_784393


namespace ratio_neha_mother_age_12_years_ago_l784_784799

variables (N : ℕ) (M : ℕ) (X : ℕ)

theorem ratio_neha_mother_age_12_years_ago 
  (hM : M = 60)
  (h_future : M + 12 = 2 * (N + 12)) :
  (12 : ℕ) * (M - 12) = (48 : ℕ) * (N - 12) :=
by
  sorry

end ratio_neha_mother_age_12_years_ago_l784_784799


namespace problem_statement_l784_784685

section

variable {f : ℝ → ℝ}

-- Conditions
axiom even_function (h : ∀ x : ℝ, f (-x) = f x) : ∀ x, f (-x) = f x 
axiom monotonically_increasing (h : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Goal
theorem problem_statement 
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  f (-Real.log 2 / Real.log 3) > f (Real.log 2 / Real.log 3) ∧ f (Real.log 2 / Real.log 3) > f 0 := 
sorry

end

end problem_statement_l784_784685


namespace quadratic_function_positive_l784_784234

theorem quadratic_function_positive (k m : ℝ) (h1 : ∀ x : ℝ, 2 * x^2 - 2 * k * x + m > 0)
  (h2 : (x^2 - 4 * x + k = 0 → (x^2 - 4 * x + k).discriminant > 0)) :
  k = 3 → m > 9 / 2 :=
by
  sorry

end quadratic_function_positive_l784_784234


namespace cube_face_covering_max_l784_784552

theorem cube_face_covering_max : 
  ∃ (k : ℕ), 
    (∀ (x y z : ℕ), x < 6 ∧ y < 6 ∧ z < 6 → 
      (∃ squares : list (ℕ × ℕ × ℕ), 
        (∀ square ∈ squares, square.1 < 2 ∧ square.2 < 2 ∧ square.3 < 2) ∧ 
        (squares.nodup) ∧ 
        (∀ (x y z : ℕ), x < 6 ∧ y < 6 ∧ z < 6 → 
          (card {s | s ∈ squares ∧ covers_sq s (x, y, z)}) = k))) 
      → k = 3 := 
sorry

end cube_face_covering_max_l784_784552


namespace intersection_M_N_l784_784653

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784653


namespace sum_of_first_ten_good_numbers_is_182_l784_784945

def is_proper_divisor (n d : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def is_good (n : ℕ) : Prop :=
  n > 1 ∧ n = List.prod (List.filter (λ d, is_proper_divisor n d) (List.range n))

def good_numbers (k : ℕ) : List ℕ :=
  (List.range 1000).filter is_good |>.take k

def sum_of_first_ten_good_numbers : ℕ :=
  List.sum (good_numbers 10)

theorem sum_of_first_ten_good_numbers_is_182 :
  sum_of_first_ten_good_numbers = 182 :=
by
  sorry

end sum_of_first_ten_good_numbers_is_182_l784_784945


namespace other_endpoint_of_diameter_eqn_l784_784192

theorem other_endpoint_of_diameter_eqn :
  ∀ (C A B : ℝ × ℝ), 
    C = (1, 2) → 
    A = (3, -1) → 
    (B.fst = C.fst - (A.fst - C.fst)) ∧ (B.snd = C.snd - (A.snd - C.snd)) → 
    B = (-1, 5) :=
by
  intros C A B hC hA hB
  rw [hC, hA] at hB
  simp at hB
  exact hB

end other_endpoint_of_diameter_eqn_l784_784192


namespace intersection_M_N_l784_784614

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l784_784614


namespace power_function_at_point_l784_784686

theorem power_function_at_point (f : ℝ → ℝ) (h : ∃ α, ∀ x, f x = x^α) (hf : f 2 = 4) : f 3 = 9 :=
sorry

end power_function_at_point_l784_784686


namespace intersection_M_N_l784_784643

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784643


namespace erasers_per_box_l784_784769

theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (erasers_per_box : ℕ) : total_erasers = 40 → num_boxes = 4 → erasers_per_box = total_erasers / num_boxes → erasers_per_box = 10 :=
by
  intros h_total h_boxes h_div
  rw [h_total, h_boxes] at h_div
  norm_num at h_div
  exact h_div

end erasers_per_box_l784_784769


namespace intersection_M_N_l784_784599

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l784_784599


namespace slope_angle_of_line_l784_784425

open Real

noncomputable def slope_angle (M N : ℝ × ℝ) : ℝ :=
let Δy := (N.snd - M.snd)
let Δx := (N.fst - M.fst)
in arctan (Δy / Δx)

theorem slope_angle_of_line :
  slope_angle (-3, 2) (-2, 3) = π / 4 := by
sorry

end slope_angle_of_line_l784_784425


namespace number_of_distinct_integer_sums_of_special_fractions_l784_784191

-- Definition of what it means to be a special fraction
def is_special_fraction (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 20

-- Define a set of fractions made from pairs that satisfy the special condition
def special_fractions : set ℚ :=
  { q : ℚ | ∃ (a b : ℕ), is_special_fraction a b ∧ q = a / b }

-- Define the sum of two special fractions
def sum_of_special_fractions : set ℚ :=
  { s : ℚ | ∃ (q1 q2 : ℚ), q1 ∈ special_fractions ∧ q2 ∈ special_fractions ∧ s = q1 + q2 }

-- The main theorem to prove
theorem number_of_distinct_integer_sums_of_special_fractions : 
  ∃ (n : ℕ), n = 7 ∧ ∀ (m : ℕ), m ∈ {i: ℤ | i ∈ sum_of_special_fractions ∧ i.denominator = 1 }.card sorry

end number_of_distinct_integer_sums_of_special_fractions_l784_784191


namespace smallest_n_satisfying_conditions_l784_784959

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n ≡ 1 [MOD 7] ∧ n ≡ 1 [MOD 4] ∧ n = 113 :=
by
  sorry

end smallest_n_satisfying_conditions_l784_784959


namespace cut_12_sided_polygon_from_square_l784_784112

theorem cut_12_sided_polygon_from_square (s : ℝ) (sides : ℕ) (side_length : ℝ) (angle_multiple : ℝ) :
  s = 2 → sides = 12 → side_length = 1 → ∃ angles : list ℝ, (∀ θ ∈ angles, θ % 45 = 0) → true :=
by {
  intro h1 h2 h3 h4,
  sorry
}

end cut_12_sided_polygon_from_square_l784_784112


namespace part1_part2_l784_784027

-- Part 1
theorem part1 (n : ℕ) (hn : n ≠ 0) (d : ℕ) (hd : d ∣ 2 * n^2) : 
  ∀ m : ℕ, ¬ (m ≠ 0 ∧ m^2 = n^2 + d) :=
by
  sorry 

-- Part 2
theorem part2 (n : ℕ) (hn : n ≠ 0) : 
  ∀ d : ℕ, (d ∣ 3 * n^2 ∧ ∃ m : ℕ, m ≠ 0 ∧ m^2 = n^2 + d) → d = 3 * n^2 :=
by
  sorry

end part1_part2_l784_784027


namespace intersection_M_N_eq_neg2_l784_784641

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l784_784641


namespace rationalize_denominator_l784_784051

theorem rationalize_denominator :
  (1 / (Real.cbrt 3 + Real.cbrt (3^3))) = (Real.cbrt 9 / 12) :=
by {
  sorry
}

end rationalize_denominator_l784_784051


namespace divide_segment_into_three_equal_parts_l784_784207

open EuclideanGeometry

variables {A B C D E P Q R S : Point}

def is_trapezoid (A B C D : Point) : Prop :=
  ∃ L1 L2,
    is_parallel (line A B) (line C D) ∧
    L1 ≠ L2 ∧
    ∀ X, X ∈ line A B → X ∈ L1 ∧ X ∈ L2 ∧
    ∀ Y, Y ∈ line C D → Y ∈ L1 ∧ Y ∈ L2

def midpoint (E C D : Point) : Prop :=
  dist E C = dist E D

def intersection_points (A B C D E : Point) (P Q : Point) : Prop :=
  is_intersection P (line A E) (line B D) ∧
  is_intersection Q (line A C) (line B E)

def intersections_with_non_parallel_sides (P Q A D B C : Point) (R S : Point) : Prop :=
  is_intersection R (line P Q) (line A D) ∧
  is_intersection S (line P Q) (line B C)

theorem divide_segment_into_three_equal_parts (A B C D E P Q R S : Point) :
  is_trapezoid A B C D →
  midpoint E C D →
  intersection_points A B C D E P Q →
  intersections_with_non_parallel_sides P Q A D B C R S →
  dist R P = dist P Q ∧ dist P Q = dist Q S
:= sorry

end divide_segment_into_three_equal_parts_l784_784207


namespace total_cost_correct_l784_784771

def total_cost (num_children num_chaperones num_teachers num_additional : ℕ)
               (num_vegetarian num_glutenfree num_both : ℕ)
               (cost_regular cost_special : ℕ → ℕ) : ℕ :=
  let total_lunches := num_children + num_chaperones + num_teachers + num_additional
  let num_vegetarian_only := num_vegetarian - num_both
  let num_glutenfree_only := num_glutenfree - num_both
  let num_regular := total_lunches - (num_both + num_vegetarian_only + num_glutenfree_only)
  cost_special 9 * num_both +
  cost_special 8 * num_vegetarian_only +
  cost_special 8 * num_glutenfree_only +
  cost_regular 7 * num_regular

theorem total_cost_correct : total_cost 35 5 1 3 10 5 2 (λ x, x) (λ x, x) = 323 :=
by
  -- Proof is not required, so this is a placeholder.
  sorry

end total_cost_correct_l784_784771


namespace flea_jumps_possible_orders_l784_784929

theorem flea_jumps_possible_orders :
  ∀ (initial_positions : List ℕ),
    initial_positions = [1, 10, 3, 12, 5] →
    ∃ (final_orders : List (List ℕ)),
    final_orders = [
      [1, 10, 3, 12, 5], 
      [10, 3, 12, 5, 1], 
      [3, 12, 5, 1, 10], 
      [12, 5, 1, 10, 3], 
      [5, 1, 10, 3, 12]
    ] :=
by
  intro initial_positions h
  have orders := [
    [1, 10, 3, 12, 5], 
    [10, 3, 12, 5, 1], 
    [3, 12, 5, 1, 10], 
    [12, 5, 1, 10, 3], 
    [5, 1, 10, 3, 12]
  ]
  use orders
  rw h
  trivial

end flea_jumps_possible_orders_l784_784929


namespace matchmakers_coexist_l784_784110

-- Define sets for boys, girls, brunette boys and blonde girls
variable (Boy : Type) (Girl : Type)
variable (Brunet : Boy → Prop) (Blond : Girl → Prop)
variable (knows : Boy → Girl → Prop)

-- Define the ability of matchmaking
variable (can_marry_off_all_brunets : (∀ b : Boy, Brunet b → ∃ g : Girl, knows b g))
variable (can_marry_off_all_blonds : (∀ g : Girl, Blond g → ∃ b : Boy, knows b g))

-- Prove that both matchmakings can be accomplished simultaneously
theorem matchmakers_coexist 
  (hb : ∀ b : Boy, Brunet b → ∃ g : Girl, knows b g)
  (hg : ∀ g : Girl, Blond g → ∃ b : Boy, knows b g) : 
  ∃ m : Boy → Option Girl, (∀ b : Boy, Brunet b → ∃ g : Girl, knows b g ∧ m b = some g) ∧ (∀ g : Girl, Blond g → ∃ b : Boy, knows b g ∧ ∃ h : Girl, knows b h ∧ m h = some g) :=
sorry

end matchmakers_coexist_l784_784110


namespace intersection_M_N_l784_784672

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784672


namespace range_of_a_l784_784246

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + 3 < 0 ∧ 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0 ) ↔ (-4 ≤ a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l784_784246


namespace suitcase_electronics_weight_l784_784853

-- Definitions from the problem conditions
variables (B C E : ℝ) -- Weights of books, clothes, electronics
variables (initial_ratio : B / C = 7 / 4) (ratio_electronics : E / B = 3 / 7)
variables (removed_clothes : 6) (new_ratio : B / (C - removed_clothes) = 14 / 4)

-- We aim to prove the weight of electronics E is 9 pounds
theorem suitcase_electronics_weight
  (h1 : B / C = 7 / 4)
  (h2 : E / B = 3 / 7)
  (h3 : B / (C - 6) = 14 / 4):
  E = 9 :=
  sorry

end suitcase_electronics_weight_l784_784853


namespace intersection_M_N_l784_784660

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784660


namespace intersection_M_N_l784_784675

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784675


namespace M_inter_N_eq_neg2_l784_784668

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784668


namespace M_inter_N_eq_neg2_l784_784665

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l784_784665


namespace bela_wins_by_8_meters_l784_784103

noncomputable def solve_race_problem (s : ℝ) (a : ℝ) (b : ℝ) (e_d : ℝ) (lead : ℝ) : Prop :=
  let ae := s - e_d in
  let be := real.sqrt (a^2 + ae^2) in
  let bf := be + e_d in
  let ant_reached := s - lead in
  a = s ∧ b = s ∧ e_d = 110 ∧ lead = 30 →
  let proportion := e_d / be in
  let antal_dist := proportion * ant_reached in
  let antal_total := lead + antal_dist in
  b ≥ s ∧ antal_dist ≤ b - s →
  a = 8

theorem bela_wins_by_8_meters :
  solve_race_problem 440 440 440 110 30 :=
  by
    sorry

end bela_wins_by_8_meters_l784_784103


namespace boards_tested_l784_784034

-- Define the initial conditions and problem
def total_thumbtacks : ℕ := 450
def thumbtacks_remaining_each_can : ℕ := 30
def initial_thumbtacks_each_can := total_thumbtacks / 3
def thumbtacks_used_each_can := initial_thumbtacks_each_can - thumbtacks_remaining_each_can
def total_thumbtacks_used := thumbtacks_used_each_can * 3
def thumbtacks_per_board := 3

-- Define the proposition to prove 
theorem boards_tested (B : ℕ) : 
  (B = total_thumbtacks_used / thumbtacks_per_board) → B = 120 :=
by
  -- Proof skipped with sorry
  sorry

end boards_tested_l784_784034


namespace intersection_M_N_l784_784631

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784631


namespace sampled_students_total_l784_784744

def total_students := 1500 + 1200 + 1000

def sample_ratio : ℝ := 60 / 1200

def total_sampled_students := total_students * sample_ratio

theorem sampled_students_total : total_sampled_students = 185 :=
by
  have h1 : total_students = 3700 := by rfl
  have h2 : sample_ratio = 1 / 20 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end sampled_students_total_l784_784744


namespace jodi_walks_days_l784_784015

section
variables {d : ℕ} -- d is the number of days Jodi walks per week

theorem jodi_walks_days (h : 1 * d + 2 * d + 3 * d + 4 * d = 60) : d = 6 := by
  sorry

end

end jodi_walks_days_l784_784015


namespace floor_abs_sum_eq_501_l784_784821

open Int

theorem floor_abs_sum_eq_501 (x : Fin 1004 → ℝ) (h : ∀ i, x i + (i : ℝ) + 1 = (Finset.univ.sum x) + 1005) : 
  Int.floor (abs (Finset.univ.sum x)) = 501 :=
by
  -- Proof steps will go here
  sorry

end floor_abs_sum_eq_501_l784_784821


namespace min_elements_in_B_l784_784591
open Set

theorem min_elements_in_B (A : Finset ℝ) (hA : A.card = 11) :
  (Finset.image (λ p : ℝ × ℝ, p.1 * p.2) (A.product (A \ {a | a = p.1}))).card ≥ 17 := 
sorry

end min_elements_in_B_l784_784591


namespace angle_DFE_l784_784113

-- Define the circles and their properties
def Point (α : Type) := α × α

structure Circle (α : Type) :=
(center : Point α)
(radius : α)

variables {α : Type} [LinearOrderedField α]

-- Conditions: Three congruent circles centered at A, B, C passing through each other's centers
def circleA := Circle α -- Circle centered at A
def circleB := Circle α -- Circle centered at B
def circleC := Circle α -- Circle centered at C

-- Assume the circles A, B, C have the same radius
axiom congruent_circles (r : α) : 
  circleA.radius = r ∧
  circleB.radius = r ∧
  circleC.radius = r ∧
  ∀ {P : Point α}, P = circleA.center ∨ P = circleB.center ∨ P = circleC.center → 
  (P = circleA.center → dist P circleB.center = r) ∧ 
  (P = circleA.center → dist P circleC.center = r) ∧
  (P = circleB.center → dist P circleA.center = r) ∧ 
  (P = circleB.center → dist P circleC.center = r) ∧
  (P = circleC.center → dist P circleA.center = r) ∧ 
  (P = circleC.center → dist P circleB.center = r)

-- Define points of intersection and angles
def D : Point α := sorry -- Point D on the circle centered at A on the line passing through A and B
def E : Point α := sorry -- Point E on the circle centered at B on the line passing through A and B
def F : Point α := sorry -- Intersection of circles centered at A and C

-- Define the angles
def ∠ : Point α → Point α → Point α → α := sorry

-- Theorem statement
theorem angle_DFE (r : α) [congruent_circles r] : ∠ D F E = 120 := 
sorry

end angle_DFE_l784_784113


namespace pencils_cost_proportion_l784_784181

/-- 
If a set of 15 pencils costs 9 dollars and the price of the set is directly 
proportional to the number of pencils it contains, then the cost of a set of 
35 pencils is 21 dollars.
--/
theorem pencils_cost_proportion :
  ∀ (p : ℕ), (∀ n : ℕ, n * 9 = p * 15) -> (35 * 9 = 21 * 15) :=
by
  intro p h1
  sorry

end pencils_cost_proportion_l784_784181


namespace simple_pairs_l784_784483

open Nat

theorem simple_pairs (n : ℕ) (h : n > 3) :
  ∃ (p1 p2 : ℕ), prime p1 ∧ prime p2 ∧ odd p1 ∧ odd p2 ∧ p2 ∣ (2 * n - p1) := sorry

end simple_pairs_l784_784483


namespace permutation_30_3_l784_784953

-- Define the number of students and the number of tasks
def n : ℕ := 30
def r : ℕ := 3

-- Define the permutation function
def P (n r : ℕ) := n.factorial / (n - r).factorial

-- Statement to prove
theorem permutation_30_3 : P 30 3 = 24360 := by
  unfold P
  norm_num
  sorry

end permutation_30_3_l784_784953


namespace inscribe_rectangle_triangle_l784_784123

theorem inscribe_rectangle_triangle (A B C : ℝ^2) (k l : ℝ) :
  ∃ (D E F G : ℝ^2),
  D ∈ line_segment A B ∧ E ∈ line_segment A C ∧
  F ∈ line_segment B C ∧ G ∈ line_segment B C ∧
  parallel (line_through D E) (line_through B C) ∧ 
  parallel (line_through F G) (line_through A B) ∧
  ‖D - E‖ / ‖F - G‖ = k / l ∧
  rectangle D E F G :=
sorry

end inscribe_rectangle_triangle_l784_784123


namespace find_section_area_l784_784102

noncomputable def side_length := (8:ℝ) / Real.sqrt 7

structure Pyramid (A B C D S : Type) :=
(base_length : ℝ)
(is_regular : Bool)

def section_area (P : Pyramid ℝ ℝ ℝ ℝ ℝ) (dist_from_apex : ℝ) : ℝ :=
  if P.is_regular = tt ∧ P.base_length = side_length 
    ∧ dist_from_apex = (2:ℝ)/3 then 6 else 0

theorem find_section_area (P : Pyramid ℝ ℝ ℝ ℝ ℝ)
  (dist_from_apex : ℝ)
  (h_base_length : P.base_length = side_length)
  (h_is_regular : P.is_regular = tt)
  (h_dist_from_apex : dist_from_apex = (2:ℝ)/3) :
  section_area P dist_from_apex = 6 :=
sorry

end find_section_area_l784_784102


namespace a6_eq_13_l784_784003

variable {α : Type} [LinearOrderedField α]

-- Definitions of the given conditions
def a (n : ℕ) : α → α → α := λ a₁ d, a₁ + (n - 1) * d
variable (a₃ : ℕ → α → α)
variable (a₅ : α → α)

-- The initial conditions given
def a3_7 : (a 3) = (7 : α) := rfl
def a5_eq_a2_plus_6 : (a 5) = (a 2) + 6 := rfl

-- The question to be proved
theorem a6_eq_13 : a 6 = 13 := sorry

end a6_eq_13_l784_784003


namespace median_score_interval_l784_784193

def intervals : List (Nat × Nat × Nat) :=
  [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]

def total_students : Nat := 100

def median_interval : Nat × Nat :=
  (70, 74)

theorem median_score_interval :
  ∃ l u n, intervals = [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]
  ∧ total_students = 100
  ∧ median_interval = (70, 74)
  ∧ ((l, u, n) ∈ intervals ∧ l ≤ 50 ∧ 50 ≤ u) :=
by
  sorry

end median_score_interval_l784_784193


namespace intersection_M_N_l784_784655

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l784_784655


namespace hexagon_coverage_percent_l784_784847

theorem hexagon_coverage_percent :
  let s := 1.0 in
  let area_hex := (3 * Real.sqrt 3) / 2 * s ^ 2 in
  let area_tri := (Real.sqrt 3) / 4 * s ^ 2 in
  let total_area := area_hex + 6 * area_tri in
  (area_hex / total_area * 100) = 50 :=
by
  let s := 1.0
  let area_hex := (3 * Real.sqrt 3) / 2 * 1 ^ 2
  let area_tri := (Real.sqrt 3) / 4 * 1 ^ 2
  let total_area := area_hex + 6 * area_tri
  have h1 : area_hex = (3 * Real.sqrt 3) / 2 := by sorry
  have h2 : area_tri = (Real.sqrt 3) / 4 := by sorry
  have h3 : total_area = 3 * Real.sqrt 3 := by sorry
  have h4 : (area_hex / total_area * 100) = 50 := by sorry
  exact h4

end hexagon_coverage_percent_l784_784847


namespace equivalent_proposition_l784_784100

theorem equivalent_proposition (a b c : ℝ) (h : a ≤ b) (hc : c^2 > 0) : 
  (a ≤ b → ac^2 ≤ bc^2) ↔ (ac^2 > bc^2 → a > b) :=
by
  sorry

end equivalent_proposition_l784_784100


namespace f_odd_function_f_decreasing_f_max_min_values_l784_784585

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x + y) = f x + f y
axiom f_neg (x : ℝ) (hx : 0 < x) : f x < 0
axiom f_value : f 3 = -2

theorem f_odd_function : ∀ (x : ℝ), f (-x) = - f x := sorry
theorem f_decreasing : ∀ (x y : ℝ), x < y → f x > f y := sorry
theorem f_max_min_values : ∀ (x : ℝ), -12 ≤ x ∧ x ≤ 12 → f x ≤ 8 ∧ f x ≥ -8 := sorry

end f_odd_function_f_decreasing_f_max_min_values_l784_784585


namespace intersection_M_N_is_correct_l784_784618

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l784_784618


namespace modulo_sum_of_99_plus_5_l784_784455

theorem modulo_sum_of_99_plus_5 : let s_n := (99 / 2) * (2 * 1 + (99 - 1) * 1)
                                 let sum_with_5 := s_n + 5
                                 sum_with_5 % 7 = 6 :=
by
  sorry

end modulo_sum_of_99_plus_5_l784_784455


namespace cube_root_approx_l784_784831

open Classical

theorem cube_root_approx (n : ℤ) (x : ℝ) (h₁ : 2^n = x^3) (h₂ : abs (x - 50) <  1) : n = 17 := by
  sorry

end cube_root_approx_l784_784831


namespace scientific_notation_l784_784763

def given_number : ℝ := 632000

theorem scientific_notation : given_number = 6.32 * 10^5 :=
by sorry

end scientific_notation_l784_784763


namespace complex_conjugate_of_z_l784_784680

theorem complex_conjugate_of_z (z : ℂ) (h : (1 - complex.I) * z = 1 + complex.I) : complex.conj z = -complex.I := 
sorry

end complex_conjugate_of_z_l784_784680


namespace evaluate_expression_l784_784190
noncomputable theory

def expression : ℤ := 7^4 - 4 * 7^3 + 6 * 7^2 - 5 * 7 + 3

theorem evaluate_expression : expression = 1553 := by
  sorry

end evaluate_expression_l784_784190


namespace a2_add_a8_l784_784751

variable (a : ℕ → ℝ) -- a_n is an arithmetic sequence
variable (d : ℝ) -- common difference

-- Condition stating that a_n is an arithmetic sequence with common difference d
axiom arithmetic_sequence : ∀ n, a (n + 1) = a n + d

-- Given condition a_3 + a_4 + a_5 + a_6 + a_7 = 450
axiom given_condition : a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem a2_add_a8 : a 2 + a 8 = 180 :=
by
  sorry

end a2_add_a8_l784_784751


namespace polynomial_coeff_fraction_eq_neg_122_div_121_l784_784790

theorem polynomial_coeff_fraction_eq_neg_122_div_121
  (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (2 - 1) ^ 5 = a0 + a1 * 1 + a2 * 1^2 + a3 * 1^3 + a4 * 1^4 + a5 * 1^5)
  (h2 : (2 - (-1)) ^ 5 = a0 + a1 * (-1) + a2 * (-1)^2 + a3 * (-1)^3 + a4 * (-1)^4 + a5 * (-1)^5)
  (h_sum1 : a0 + a1 + a2 + a3 + a4 + a5 = 1)
  (h_sum2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) :
  (a0 + a2 + a4) / (a1 + a3 + a5) = - 122 / 121 :=
sorry

end polynomial_coeff_fraction_eq_neg_122_div_121_l784_784790


namespace range_of_a_value_of_a_l784_784316

-- Definitions and conditions
def system_of_equations (a x y : ℝ) : Prop :=
  3 * x - y = 2 * a - 5 ∧ x + 2 * y = 3 * a + 3

def positive_solutions (a : ℝ) : Prop :=
  ∃ x y : ℝ, system_of_equations a x y ∧ x > 0 ∧ y > 0

def isosceles_with_perimeter_12 (a : ℝ) : Prop :=
  ∃ x y : ℝ, system_of_equations a x y ∧
  ((2 * y + x = 12 ∧ x = y) ∨ (2 * x + y = 12 ∧ y = x) ∨ (2 * y + x = 12 ∧ 2 * y + x = 2 * (a + 2) + (a - 1) = 12))

-- Theorem statement
theorem range_of_a (a : ℝ) : positive_solutions a ↔ a > 1 := sorry

theorem value_of_a (a : ℝ) : isosceles_with_perimeter_12 a ↔ a = 3 := sorry

end range_of_a_value_of_a_l784_784316


namespace part_i_part_ii_l784_784030

-- Part (i)
theorem part_i : 
  let A := fun (a b : ℕ) => (1 <= a ∧ a <= 6) ∧ (1 <= b ∧ b <= 6) ∧ (a < b * Real.sqrt 3) in
  (PMF.filter A (PMF.uniform_of_finset (Finset.product (Finset.range 7) (Finset.range 7)))).toRealProb = 7 / 12 := 
sorry

-- Part (ii)
theorem part_ii :
  let A := fun (a b : ℝ) => (a - Real.sqrt 3)^2 + (b - 1)^2 <= 4 ∧ (a < b * Real.sqrt 3) in
  (MeasureTheory.measureSpace.restrict MeasureTheory.measureSpace.volume { p : ℝ × ℝ | A p.fst p.snd } 
   (FloatingMeasureTheory.volume {p : ℝ × ℝ | (a~-Real.sqrt 3)^2 +(b-1)^2<=4 }).toRealMeasure.toProbabilitySpace = 1 / 2 := 
sorry

end part_i_part_ii_l784_784030


namespace rationalize_denominator_correct_l784_784056

noncomputable def rationalize_denominator : Prop := 
  (1 / (Real.cbrt 3 + Real.cbrt 27) = Real.cbrt 9 / 12)

theorem rationalize_denominator_correct : rationalize_denominator := 
  sorry

end rationalize_denominator_correct_l784_784056


namespace division_expression_l784_784803

theorem division_expression:
  (-1 / 30) / (2 / 3 - 1 / 10 + 1 / 6 - 2 / 5) = -1 / 10 :=
by
  calc
    (-1 / 30) / (2 / 3 - 1 / 10 + 1 / 6 - 2 / 5)
      = (-1 / 30) / (40 / 60 - 6 / 60 + 10 / 60 - 24 / 60) : by sorry
    ... = (-1 / 30) / 20 / 60 : by sorry
    ... = (-1 / 30) * (60 / 20) : by sorry
    ... = (-1 / 30) * 3 : by sorry
    ... = (-1 * 3) / 30 : by sorry
    ... = -1 / 10 : by sorry

end division_expression_l784_784803


namespace incorrect_sqrt_2_plus_sqrt_3_l784_784900

theorem incorrect_sqrt_2_plus_sqrt_3 :
  ¬ (sqrt 2 + sqrt 3 = sqrt 5) := by
  sorry

end incorrect_sqrt_2_plus_sqrt_3_l784_784900


namespace simplify_product_l784_784816

theorem simplify_product : 
  18 * (8 / 15) * (2 / 27) = 32 / 45 :=
by
  sorry

end simplify_product_l784_784816


namespace product_xy_min_value_x_plus_y_min_value_attained_l784_784308

theorem product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : x * y = 64 := 
sorry

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : 
  x + y = 18 := 
sorry

-- Additional theorem to prove that the minimum value is attained when x = 6 and y = 12
theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) :
  x = 6 ∧ y = 12 := 
sorry

end product_xy_min_value_x_plus_y_min_value_attained_l784_784308


namespace find_f_2013_l784_784409

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f(x + 2) = - (1 / f(x))
axiom initial_condition : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f(x) = x

theorem find_f_2013 : f(2013) = - (1 / 3) :=
by
  sorry

end find_f_2013_l784_784409


namespace solve_inequality_find_range_a_l784_784687

-- Problem 1
theorem solve_inequality (x : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x|) :
  f (2 * x - 3) ≤ 5 ↔ -1 ≤ x ∧ x ≤ 4 := sorry

-- Problem 2
theorem find_range_a (x a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x|) (h' : x ∈ set.Icc (-1:ℝ) 3) :
  (x^2 + 2*x + f (x - 2) + f (x + 3) ≥ a + 1) ↔ (a ≤ -3) := sorry

end solve_inequality_find_range_a_l784_784687


namespace sum_of_ages_l784_784770

-- Definitions
variable {J L : ℕ}

-- Conditions
def james_older_than_louise : Prop := J = L + 9
def future_age_relation : Prop := (J + 7) = 3 * (L - 3)

-- Theorem stating the sum of their ages
theorem sum_of_ages (h1 : james_older_than_louise) (h2 : future_age_relation) : J + L = 35 :=
by
  sorry

end sum_of_ages_l784_784770


namespace intersection_M_N_l784_784630

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784630


namespace factorization_correct_l784_784462

-- Definition of the expressions
def expr_LHS : ℤ → ℤ := λ m, 4 - m^2
def expr_RHS (m : ℤ) : ℤ := (2 + m) * (2 - m)

-- Theorem stating that the LHS equals the RHS
theorem factorization_correct (m : ℤ) : expr_LHS m = expr_RHS m := by
  sorry

end factorization_correct_l784_784462


namespace floor_sqrt_77_l784_784553

theorem floor_sqrt_77 : 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 → Int.floor (Real.sqrt 77) = 8 :=
by
  sorry

end floor_sqrt_77_l784_784553


namespace carl_spends_108_dollars_l784_784983

theorem carl_spends_108_dollars
    (index_cards_per_student : ℕ := 10)
    (periods_per_day : ℕ := 6)
    (students_per_class : ℕ := 30)
    (cost_per_pack : ℕ := 3)
    (cards_per_pack : ℕ := 50) :
  let total_index_cards := index_cards_per_student * students_per_class * periods_per_day in
  let total_packs := total_index_cards / cards_per_pack in
  let total_cost := total_packs * cost_per_pack in
  total_cost = 108 := 
by
  sorry

end carl_spends_108_dollars_l784_784983


namespace intersection_M_N_l784_784644

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l784_784644


namespace infinite_primes_dividing_polynomial_values_infinite_n_with_r_distinct_prime_factors_l784_784365

-- Define non-constant polynomial P : ℤ[X]
variable (P : Polynomial ℤ)
variable (hP : P.degree > 0)

-- Define the first proof problem: infinitely many primes divide P(n) for some n in ℤ
theorem infinite_primes_dividing_polynomial_values : 
  ∃∞ p: ℕ, ∃ n: ℤ, p.prime ∧ p ∣ P.eval n := sorry

-- Define the second proof problem: for any r in ℕ*, infinitely many n in ℕ* such that P(n) has at least r distinct prime factors
theorem infinite_n_with_r_distinct_prime_factors (r : ℕ) (hr : 0 < r) :
  ∃∞ n : ℕ, ∃ pfs : Finset ℕ, pfs.card ≥ r ∧ ∀ p ∈ pfs, p.prime ∧ p ∣ P.eval (n : ℤ) := sorry

end infinite_primes_dividing_polynomial_values_infinite_n_with_r_distinct_prime_factors_l784_784365


namespace intersection_M_N_l784_784632

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l784_784632


namespace jenny_original_amount_half_l784_784772

-- Definitions based on conditions
def original_amount (x : ℝ) := x
def spent_fraction := 3 / 7
def left_after_spending (x : ℝ) := x * (1 - spent_fraction)

theorem jenny_original_amount_half (x : ℝ) (h : left_after_spending x = 24) : original_amount x / 2 = 21 :=
by
  -- Indicate the intention to prove the statement by sorry
  sorry

end jenny_original_amount_half_l784_784772


namespace sum_of_neighbors_of_two_in_150_divisors_l784_784849

theorem sum_of_neighbors_of_two_in_150_divisors : 
  let divisors := [2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150],
      circle_condition := ∀ i j, (i ≠ j ∧ i ∈ divisors ∧ j ∈ divisors) → gcd i j > 1
  in circle_condition →
     ∃ a b, a ≠ b ∧ (a = 2 ∨ b = 2) ∧ a ∈ divisors ∧ b ∈ divisors ∧ a + b = 16 := 
sorry

end sum_of_neighbors_of_two_in_150_divisors_l784_784849


namespace find_k_l784_784339

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end find_k_l784_784339


namespace find_width_l784_784752

theorem find_width (L D V : ℕ) (hL : L = 20) (hD : D = 5) (hV : V = 1200) : 
  ∃ W : ℕ, (L * W * D = V) ∧ W = 12 :=
by 
  use 12
  split
  sorry
  sorry

end find_width_l784_784752
