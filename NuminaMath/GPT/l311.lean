import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.AdditionGroup
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Extrema
import Mathlib.Analysis.Polynomial
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Derangements
import Mathlib.Combinatorics.Probability
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Probability.Basic
import Mathlib.ProbTheory.Basic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic

namespace semi_circle_radius_l311_311332

theorem semi_circle_radius (π : ℝ) (hπ : Real.pi = π) (P : ℝ) (hP : P = 180) : 
  ∃ r : ℝ, r = 180 / (π + 2) :=
by
  sorry

end semi_circle_radius_l311_311332


namespace part1_circle_eqn_part2_line_eqn_l311_311045

noncomputable theory

-- Defining the first part of the problem
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Defining the line in the condition (tangency)
def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y - 10 = 0

-- Proving that the center of the circle C1 (0,0) and its radius results in the equation
theorem part1_circle_eqn : ∀ (x y : ℝ), circle_eqn x y ↔ (x = 0 ∧ y = 0) ∨ tangent_line x y :=
sorry

-- Defining the points and lines for second part
def point_M : (ℝ × ℝ) := (1, 2)

def line_l1 (x y : ℝ) : Prop := x = 1

def line_l2 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Proving that given conditions implies correct lines intercept chord of given length
theorem part2_line_eqn : ∀ (l : ℝ → ℝ → Prop), 
  (∀ (x y : ℝ), l x y → (x, y) = point_M) →
  (∀ (x y : ℝ), l x y → √(4 - ((2 * √3) / 2)^2) = 1) →
  (l = line_l1 ∨ l = line_l2) :=
sorry

end part1_circle_eqn_part2_line_eqn_l311_311045


namespace integer_conditions_meet_l311_311000

theorem integer_conditions_meet (n : ℤ) (hn : n ≥ 2) :
  (∃ p : ℤ, prime p ∧ n = p) ∨ (∃ p : ℤ, prime p ∧ n = p^2) ↔
  (∀ (d : ℤ), d ≥ 2 → d ∣ n → d - 1 ∣ n - 1) :=
by
  sorry

end integer_conditions_meet_l311_311000


namespace sum_of_divisors_of_29_l311_311765

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311765


namespace simplify_sqrt_l311_311195

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311195


namespace min_k_condition_l311_311501

noncomputable def a (n : ℕ) := 3 * n + 1

noncomputable def S (n : ℕ) := (finset.range n).sum (λ i, a i)

noncomputable def b (n : ℕ) := 1 / ((a n - 1) * (a (n + 1) - 1))

noncomputable def T (n : ℕ) := (finset.range n).sum b

theorem min_k_condition {k : ℝ} (h : ∀ n : ℕ, k > T n) : k ≥ 1 / 9 :=
sorry

end min_k_condition_l311_311501


namespace class_average_rounded_to_nearest_percent_l311_311516

theorem class_average_rounded_to_nearest_percent :
  let percentage1 := 0.15
  let average1 := 100
  let percentage2 := 0.50
  let average2 := 78
  let percentage3 := 0.35
  let average3 := 63
  
  let overall_average : ℝ := (percentage1 * average1) + (percentage2 * average2) + (percentage3 * average3)
  let rounded_average : ℤ := Int.round overall_average

  rounded_average = 76 := by
  sorry

end class_average_rounded_to_nearest_percent_l311_311516


namespace find_length_QS_l311_311259

theorem find_length_QS 
  (cosR : ℝ) (RS : ℝ) (QR : ℝ) (QS : ℝ)
  (h1 : cosR = 3 / 5)
  (h2 : RS = 10)
  (h3 : cosR = QR / RS) :
  QS = 8 :=
by
  sorry

end find_length_QS_l311_311259


namespace problem_part_I_problem_part_II_l311_311015

-- Conditions
def P (a : ℝ) := 80 + 4 * Real.sqrt (2 * a)
def Q (a : ℝ) := (1 / 4) * a + 120
def f (x : ℝ) := P x + Q (200 - x)

-- Assertions
theorem problem_part_I : f 50 = 277.5 := 
by sorry

theorem problem_part_II : 
  ∃ x ∈ set.Icc (20 : ℝ) 180, ∀ y ∈ set.Icc (20 : ℝ) 180, f y ≤ f x ∧ f x = 282 ∧ x = 128 := 
by sorry

end problem_part_I_problem_part_II_l311_311015


namespace find_number_eq_l311_311517

theorem find_number_eq (x : ℝ) (h : (35 / 100) * x = (20 / 100) * 40) : x = 160 / 7 :=
by
  sorry

end find_number_eq_l311_311517


namespace maximum_value_a_over_b_plus_c_l311_311150

open Real

noncomputable def max_frac_a_over_b_plus_c (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * (a + b + c) = b * c) : ℝ :=
  if (b = c) then (Real.sqrt 2 - 1) / 2 else -1 -- placeholder for irrelevant case

theorem maximum_value_a_over_b_plus_c 
  (a b c : ℝ) 
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq: a * (a + b + c) = b * c) :
  max_frac_a_over_b_plus_c a b c h_pos h_eq = (Real.sqrt 2 - 1) / 2 :=
sorry

end maximum_value_a_over_b_plus_c_l311_311150


namespace sum_of_divisors_of_29_l311_311798

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311798


namespace sum_of_divisors_prime_29_l311_311733

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311733


namespace area_of_parallelogram_l311_311609

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l311_311609


namespace sum_of_first_n_terms_l311_311456

-- Definition for the sum of the first n terms of a sequence
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (-1)^n * a n + 1/(2^n) + n - 3

-- Definition for sequence a_{2n-1}
noncomputable def a (a : ℕ → ℝ) (n : ℕ) : ℕ → ℝ
| 0       := a 0
| (k + 1) := let n' := 2 * (k + 1) - 1 in
              S a n' - S a (2 * k - 1)

-- Sum of the first n terms of the sequence
noncomputable def sum_a_2n_minus_1 (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a a (2 * (i + 1) - 1)

theorem sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) :
  sum_a_2n_minus_1 a n = 1/(2^(n-2)) - 1/(4^n) - 3 + 2 * n :=
by
  sorry

end sum_of_first_n_terms_l311_311456


namespace olivia_difference_l311_311165

-- Definitions for the conditions
def earned_from_job : ℤ := 65
def collected_from_ATM : ℤ := 195
def spent_at_supermarket : ℤ := 87
def spent_at_electronics : ℤ := 134
def spent_on_clothes : ℤ := 78

-- Statement of the theorem
theorem olivia_difference :
  let total_earned := earned_from_job + collected_from_ATM,
      total_spent := spent_at_supermarket + spent_at_electronics + spent_on_clothes
  in total_earned - total_spent = -39 :=
by
  sorry

end olivia_difference_l311_311165


namespace equation_of_C_fixed_point_max_area_l311_311071

-- Definitions
def circle1 (x y r : ℝ) : Prop := (x + real.sqrt 3)^2 + y^2 = r^2
def circle2 (x y r : ℝ) : Prop := (x - real.sqrt 3)^2 + y^2 = (4 - r)^2
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def fixed_point_N : ℝ × ℝ := (0, -3 / 5)
def point_M : ℝ × ℝ := (0, 1)
def orthogonal (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Propositions to prove
theorem equation_of_C (r : ℝ) (h1 : 0 < r) (h2 : r < 4) : 
  ∃ (x y : ℝ), circle1 x y r ∧ circle2 x y r → curve_C x y := 
sorry

theorem fixed_point (A B : ℝ × ℝ) (hA : curve_C A.1 A.2) (hB : curve_C B.1 B.2) 
  (hOrth : orthogonal (point_M, A) (point_M, B)) : 
  ∃ t, (fixed_point_N = (t, A.2)) ∨ (fixed_point_N = (t, B.2)) := 
sorry

theorem max_area (A B : ℝ × ℝ) (hA : curve_C A.1 A.2) (hB : curve_C B.1 B.2) 
  (hOrth : orthogonal (point_M, A) (point_M, B)) : 
  ∃ S, S = 64 / 25 := 
sorry

end equation_of_C_fixed_point_max_area_l311_311071


namespace simplify_cubed_roots_l311_311217

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311217


namespace smallest_p_l311_311314

theorem smallest_p (n p : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) (h3 : (n + p) % 10 = 0) : p = 1 := 
sorry

end smallest_p_l311_311314


namespace total_percentage_of_samplers_l311_311536

theorem total_percentage_of_samplers :
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  (pA + pA_not_caught + pB + pB_not_caught + pC + pC_not_caught + pD + pD_not_caught) = 54 :=
by
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  sorry

end total_percentage_of_samplers_l311_311536


namespace ellipse_with_focus_and_conditions_l311_311461

noncomputable
def ellipse_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem ellipse_with_focus_and_conditions :
  ∀ (a b c : ℝ) (P : ℝ × ℝ), 
    a > b ∧ b > 0 ∧
    P = (1, 3 / 2) ∧
    dot_product (1 - c, -1.5) (c - 1, -1.5) = 9 / 4 →
  ∃ (a b : ℝ), 
    ellipse_equation = (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 ∧
    ∀ (M N K : ℝ × ℝ),
      (some_condition_about_lines P M N K) →
      |PM| * |KN| = |PN| * |KM| ∧
      ∀ (exists geometric_sequence : Prop, ¬exists geometric_sequence) :=
sorry

end ellipse_with_focus_and_conditions_l311_311461


namespace putnam_1966_pA3_l311_311451

theorem putnam_1966_pA3 (m n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, i < j → a i < a j) (len_a : ∀ a, a ≤ m * n + 1  → 0 < a) : 
  (∃ b : ℕ → ℕ, ∃ (hb : ∀ i j, i < j → ¬ (b i ∣ b j)), ∀ k, 0 < k  → k ≤ m + 1 → b k ∈ a) 
  ∨ (∃ c : ℕ → ℕ, ∃ (hc : ∀ i j, i < j → c j ∣ c i),  ∀ k, 0 < k  → k ≤ n + 1 → c k ∈ a) :=
begin
  sorry
end

end putnam_1966_pA3_l311_311451


namespace problem_proof_l311_311592

noncomputable def g : ℝ → ℝ := sorry

axiom g_prop1 : g 2 = 2
axiom g_prop2 : ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

/-- Prove that the product of the number of possible values of g (1/3) and their sum is 2/3. -/
theorem problem_proof : 
  let m := {y : ℝ | ∃ g3, (∀ x y z : ℝ, (g (x * y + g x) = x * g y + g x)) ∧ (g 2 = 2) ∧ g3 = g (1/3) ∧ y = g3}.to_finset.card in
  let t := ∑ y in {y : ℝ | ∃ g3, (∀ x y z : ℝ, (g (x * y + g x) = x * g y + g x)) ∧ (g 2 = 2) ∧ g3 = g (1/3) ∧ y = g3}.to_finset, y in
  m * t = 2 / 3 :=
by
  let m := {y : ℝ | ∃ g3, (∀ x y z : ℝ, (g (x * y + g x) = x * g y + g x)) ∧ (g 2 = 2) ∧ g3 = g (1/3) ∧ y = g3}.to_finset.card
  let t := ∑ y in {y : ℝ | ∃ g3, (∀ x y z : ℝ, (g (x * y + g x) = x * g y + g x)) ∧ (g 2 = 2) ∧ g3 = g (1/3) ∧ y = g3}.to_finset, y
  have h_m : m = 1 := sorry
  have h_t : t = 2 / 3 := sorry
  show m * t = 2 / 3, by rw [h_m, h_t, one_mul]

end problem_proof_l311_311592


namespace time_for_train_to_pass_man_l311_311324

def bullet_train_length : ℝ := 160  -- in meters
def bullet_train_speed : ℝ := 70    -- in kmph
def man_speed : ℝ := 8              -- in kmph
def conversion_factor : ℝ := 5/18   -- factor to convert kmph to m/s

-- Define the relative speed in kmph and then convert it to m/s
def relative_speed_kmph := bullet_train_speed + man_speed
def relative_speed_mps := relative_speed_kmph * conversion_factor

-- Define the expected time in seconds
def expected_time := bullet_train_length / relative_speed_mps

-- The theorem to prove
theorem time_for_train_to_pass_man : expected_time ≈ 7.38 := 
by 
  sorry

end time_for_train_to_pass_man_l311_311324


namespace cube_root_simplification_l311_311251

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311251


namespace energy_comparison_l311_311074

noncomputable def n_butane_combustion_enthalpy : ℝ := -2878 -- kJ/mol
noncomputable def isobutane_combustion_enthalpy : ℝ := -2869 -- kJ/mol

theorem energy_comparison :
  n_butane_combustion_enthalpy > isobutane_combustion_enthalpy :=
begin
  calc
    -2869 < -2878 : by linarith,
end

end energy_comparison_l311_311074


namespace tv_horizontal_length_l311_311162

-- Conditions
def is_rectangular_tv (width height : ℝ) : Prop :=
width / height = 9 / 12

def diagonal_is (d : ℝ) : Prop :=
d = 32

-- Theorem to prove
theorem tv_horizontal_length (width height diagonal : ℝ) 
(h1 : is_rectangular_tv width height) 
(h2 : diagonal_is diagonal) : 
width = 25.6 := by 
sorry

end tv_horizontal_length_l311_311162


namespace part_I_solution_set_part_II_range_of_a_l311_311448

-- Definitions
def f (x : ℝ) (a : ℝ) := |x - 1| + |a * x + 1|
def g (x : ℝ) := |x + 1| + 2

-- Part I: Prove the solution set of the inequality f(x) < 2 when a = 1/2
theorem part_I_solution_set (x : ℝ) : f x (1/2 : ℝ) < 2 ↔ 0 < x ∧ x < (4/3 : ℝ) :=
sorry
  
-- Part II: Prove the range of a such that (0, 1] ⊆ {x | f x a ≤ g x}
theorem part_II_range_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1 → f x a ≤ g x) ↔ -5 ≤ a ∧ a ≤ 3 :=
sorry

end part_I_solution_set_part_II_range_of_a_l311_311448


namespace manager_salary_l311_311660

def average_salary_of_15_employees := 1800
def number_of_employees := 15
def increment_in_average_salary := 150

theorem manager_salary :
    let total_salary_15 := number_of_employees * average_salary_of_15_employees in
    let new_average_salary := average_salary_of_15_employees + increment_in_average_salary in
    let total_people := number_of_employees + 1 in
    let total_salary_16 := total_people * new_average_salary in
    let manager_salary := total_salary_16 - total_salary_15 in
    manager_salary = 4200 := 
  by
    sorry

end manager_salary_l311_311660


namespace _l311_311180

noncomputable theory
open Classical

/-- Statement of the theorem in Lean 4 -/
def prob_sine_floor_eq (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
  Pr(\(floor (sin x) = floor (sin y)\)) = 1 := sorry

end _l311_311180


namespace sum_of_exterior_angles_hexagon_l311_311684

theorem sum_of_exterior_angles_hexagon : 
  ∀ (n : ℕ), n = 6 → (sum_of_exterior_angles_of_n_sided_polygon n) = 360 := 
by
  intro n h
  sorry

def sum_of_exterior_angles_of_n_sided_polygon (n : ℕ) : ℕ :=
  360

end sum_of_exterior_angles_hexagon_l311_311684


namespace count_interesting_le_400_l311_311416

def f1 (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  let factors := n.factors in  -- using a list of prime factors
  factors.foldr (λ p acc, acc * (p + 1)) 1 / n + n

def f (m n : ℕ) : ℕ :=
  match m with
  | 1 => f1 n
  | m + 1 => f1 (f m n)

def interesting (n : ℕ) : Prop :=
  ∀ B : ℕ, ∃ k : ℕ, f k n > B

theorem count_interesting_le_400 : 
  (finset.range 401).filter interesting).card = 18 :=
sorry

end count_interesting_le_400_l311_311416


namespace P_2_lt_X_lt_5_l311_311095

-- Definitions capturing the given problem conditions
variable (X : ℝ → ℝ)
variable (μ σ : ℝ)

-- The normal distribution assumption
axiom X_normal : X ~ normal μ σ²

-- Given conditions
axiom P_gt_5 : ∀ x, P (X > 5) = 0.2
axiom P_lt_neg_1 : ∀ x, P (X < -1) = 0.2

-- We want to prove the following theorem
theorem P_2_lt_X_lt_5 : P (2 < X < 5) = 0.3 :=
sorry

end P_2_lt_X_lt_5_l311_311095


namespace return_to_initial_state_l311_311657

-- Define the state in terms of the distribution of golf balls and the starting box for next move.
structure State where
  boxes : Fin 10 → ℕ       -- Number of golf balls in each of the 10 boxes.
  nextBox : Fin 10         -- The box where the next move will start.

-- Define the move function which performs the move operation and returns the next state.
def move (s : State) : State := sorry

-- Define a function to check if two states are identical.
def statesEqual (s1 s2 : State) : Prop :=
  s1.boxes = s2.boxes ∧ s1.nextBox = s2.nextBox

-- Prove that after some number of moves, we get back to the initial distribution of golf balls in the boxes.
theorem return_to_initial_state (initialState : State) : ∃ n : ℕ, statesEqual (iterate move n initialState) initialState := 
sorry

end return_to_initial_state_l311_311657


namespace quadratic_reciprocity_l311_311093

theorem quadratic_reciprocity (a b : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) :
  legendre_symbol a b = (-1) ^ ((a - 1) / 2 * (b - 1) / 2) * legendre_symbol b a :=
by
  sorry

end quadratic_reciprocity_l311_311093


namespace sum_of_divisors_of_29_l311_311744

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311744


namespace fraction_of_profit_b_received_l311_311899

theorem fraction_of_profit_b_received (capital months_a_share months_b_share : ℝ) 
  (hA_contrib : capital * (1/4) * months_a_share = capital * (15/4))
  (hB_contrib : capital * (3/4) * months_b_share = capital * (30/4)) :
  (30/45) = (2/3) :=
by sorry

end fraction_of_profit_b_received_l311_311899


namespace dog_food_duration_l311_311960

-- Definitions for the given conditions
def number_of_dogs : ℕ := 4
def meals_per_day : ℕ := 2
def grams_per_meal : ℕ := 250
def sacks_of_food : ℕ := 2
def kilograms_per_sack : ℝ := 50
def grams_per_kilogram : ℝ := 1000

-- Lean statement to prove the correct answer
theorem dog_food_duration : 
  ((number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) * sacks_of_food * kilograms_per_sack) / 
  (number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) = 50 :=
by 
  simp only [number_of_dogs, meals_per_day, grams_per_meal, sacks_of_food, kilograms_per_sack, grams_per_kilogram]
  norm_num
  sorry

end dog_food_duration_l311_311960


namespace probability_of_moving_twice_vs_once_l311_311107

theorem probability_of_moving_twice_vs_once :
  let p := 1 / 4
  let q := 3 / 4
  let move_once := (4 * (p) * (q ^ 3))
  let move_twice := (6 * (p ^ 2) * (q ^ 2))
  move_twice / move_once = 1 / 2 :=
sorry

end probability_of_moving_twice_vs_once_l311_311107


namespace sum_of_divisors_of_29_l311_311740

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311740


namespace simplify_and_evaluate_l311_311650

theorem simplify_and_evaluate (x : ℝ) (h₁ : x = 3) : (x^2 - x) / (x^2 + 2 * x + 1) / (x - 1) / (x + 1) = 3 / 4 :=
by
  have t1 : (x^2 - x) / (x^2 + 2 * x + 1) / ( (x - 1) / (x + 1) ) = x / (x + 1),
  { sorry }, -- Simplification step
  rw [h₁] at t1,
  exact t1,

end simplify_and_evaluate_l311_311650


namespace cube_root_simplification_l311_311252

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311252


namespace count_four_digit_form_divisible_by_30_l311_311510

def isDigit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def isFourDigitForm (n : ℕ) : Prop :=
  ∃ a b, isDigit a ∧ isDigit b ∧ n = 1000 * a + 100 * b + 30

def isDivisibleBy30 (n : ℕ) : Prop :=
  n % 30 = 0

theorem count_four_digit_form_divisible_by_30 :
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ isFourDigitForm n ∧ isDivisibleBy30 n}.to_finset.card = 30 :=
by
  sorry

end count_four_digit_form_divisible_by_30_l311_311510


namespace sum_of_divisors_29_l311_311830

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311830


namespace CarmenBrushLengthInCentimeters_l311_311976

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters_l311_311976


namespace intersection_of_M_and_N_l311_311067

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

-- The proof statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := 
  sorry

end intersection_of_M_and_N_l311_311067


namespace sum_of_divisors_of_29_l311_311719

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311719


namespace max_ratio_MO_MF_on_parabola_l311_311546

theorem max_ratio_MO_MF_on_parabola (F M : ℝ × ℝ) : 
  let O := (0, 0)
  let focus := (1 / 2, 0)
  ∀ (M : ℝ × ℝ), (M.snd ^ 2 = 2 * M.fst) →
  F = focus →
  (∃ m > 0, M.fst = m ∧ M.snd ^ 2 = 2 * m) →
  (∃ t, t = m - (1 / 4)) →
  ∃ value, value = (2 * Real.sqrt 3) / 3 ∧
  ∃ rat, rat = dist M O / dist M F ∧
  rat = value := 
by
  admit

end max_ratio_MO_MF_on_parabola_l311_311546


namespace correct_operation_l311_311318

variables (a b : ℝ)

theorem correct_operation : 5 * a * b - 3 * a * b = 2 * a * b :=
by sorry

end correct_operation_l311_311318


namespace total_distance_traveled_l311_311638

/-- Problem Conditions -/
def DF : ℕ := 4000
def DE : ℕ := 4500

/-- Calculate FE using the Pythagorean theorem -/
def FE : ℕ := Nat.sqrt (DE^2 - DF^2)

/-- Calculate the total distance traveled -/
def total_distance : ℕ := DF + FE + DE

/-- The proof statement -/
theorem total_distance_traveled : total_distance = 10562 := by
  unfold total_distance FE
  norm_num
  sorry

end total_distance_traveled_l311_311638


namespace cube_root_multiplication_l311_311200

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311200


namespace circle_radius_range_l311_311114

theorem circle_radius_range (r : ℝ) : 
  (∃ P₁ P₂ : ℝ × ℝ, (P₁.2 = 1 ∨ P₁.2 = -1) ∧ (P₂.2 = 1 ∨ P₂.2 = -1) ∧ 
  (P₁.1 - 3) ^ 2 + (P₁.2 + 5) ^ 2 = r^2 ∧ (P₂.1 - 3) ^ 2 + (P₂.2 + 5) ^ 2 = r^2) → (4 < r ∧ r < 6) :=
by
  sorry

end circle_radius_range_l311_311114


namespace num_squares_below_line_in_first_quadrant_l311_311277

theorem num_squares_below_line_in_first_quadrant :
  let equation := 5 * x + 152 * y = 1520 
  ∧ ∀ (x y: ℤ), x ≥ 0 ∧ y ≥ 0 := 1363 :=
by sorry

end num_squares_below_line_in_first_quadrant_l311_311277


namespace sienna_average_speed_l311_311648

def minutes_to_hours (m : ℕ) : ℝ := m / 60

def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

def total_distance (d1 d2 : ℝ) : ℝ := d1 + d2

def total_time (t1 t2 : ℝ) : ℝ := t1 + t2

def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem sienna_average_speed :
  let bike_minutes := 45
  let bike_speed := 12
  let rollerblading_minutes := 75
  let rollerblading_speed := 8
  let bike_time := minutes_to_hours bike_minutes
  let rollerblading_time := minutes_to_hours rollerblading_minutes
  let bike_distance := distance bike_speed bike_time
  let rollerblading_distance := distance rollerblading_speed rollerblading_time
  let total_dist := total_distance bike_distance rollerblading_distance
  let total_travel_time := total_time bike_time rollerblading_time
  average_speed total_dist total_travel_time = 9.5 :=
by
  sorry

end sienna_average_speed_l311_311648


namespace parallelogram_area_l311_311631

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l311_311631


namespace find_BP_l311_311532

-- Given definitions and conditions
noncomputable def AB := 40
noncomputable def BC := 60
noncomputable def CA := 50

-- angle bisector of ∠A intersects circumcircle at A and P
axiom angle_bisector_intersects : ∃ P, 
  (is_angle_bisector A P) ∧ 
  (intersects_at_circumcircle A B C P)

-- The main theorem to prove
theorem find_BP : ∀ {A B C P : Point},
  (AB = 40) →
  (BC = 60) →
  (CA = 50) →
  (angle_bisector_intersects) →
  (BP = 40) :=
begin
  sorry
end

end find_BP_l311_311532


namespace oil_spent_amount_l311_311942

theorem oil_spent_amount :
  ∀ (P R M : ℝ), R = 25 → P = (R / 0.75) → ((M / R) - (M / P) = 5) → M = 500 :=
by
  intros P R M hR hP hOil
  sorry

end oil_spent_amount_l311_311942


namespace least_integer_greater_than_altitude_is_five_l311_311137

-- Define the legs of the right triangle
def leg1 : ℕ := 5
def leg2 : ℕ := 12

-- Define the area of the right triangle
def area : ℚ := (1 / 2 : ℚ) * leg1 * leg2

-- Define the hypotenuse of the right triangle using the Pythagorean theorem
def hypotenuse : ℚ := real.sqrt (leg1^2 + leg2^2)

-- Define the altitude (L) to the hypotenuse
def altitude : ℚ := (2 * area) / hypotenuse

-- Define the least integer greater than the altitude
def least_int_greater_than_altitude : ℕ := nat_ceil altitude

-- The goal is to prove that the least integer greater than the altitude is 5
theorem least_integer_greater_than_altitude_is_five : least_int_greater_than_altitude = 5 :=
by
  sorry

end least_integer_greater_than_altitude_is_five_l311_311137


namespace sum_of_divisors_prime_29_l311_311728

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311728


namespace sum_of_divisors_of_29_l311_311852

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311852


namespace odd_function_analysis_l311_311035

noncomputable def f (a b c x : ℝ) := a * x^3 + b * x^2 + c * x

theorem odd_function_analysis 
  (a b c : ℝ)
  (f_odd : ∀ x, f a b c (-x) = -f a b c x)
  (f_max_at_1 : ∀ (d : ℝ), d > 0 → ∀ x, f a b c x ≤ f a b c 1)
  (f_1_eq_2 : f a b c 1 = 2) :
  (a = -1 ∧ b = 0 ∧ c = 3)
  ∧ (∀ x ∈ Icc (-4 : ℝ) 3, f a b c x ≤ 52) 
  ∧ (∀ x ∈ Icc (-4 : ℝ) 3, f a b c x ≥ -18) :=
by
  sorry

end odd_function_analysis_l311_311035


namespace hair_cut_off_length_l311_311127

def initial_hair_length : ℕ := 18
def hair_length_after_haircut : ℕ := 9

theorem hair_cut_off_length :
  initial_hair_length - hair_length_after_haircut = 9 :=
sorry

end hair_cut_off_length_l311_311127


namespace unique_positive_integer_n_l311_311407

theorem unique_positive_integer_n :
  ∃! n : ℕ, (2 * 3^2 + 3 * 3^3 + 4 * 3^4 + ∑ i in Finset.range (n - 3), (i + 5) * 3^(i + 5)) = 3^(n + 5) ∧ n > 0 := 
sorry

end unique_positive_integer_n_l311_311407


namespace sum_of_divisors_of_prime_29_l311_311841

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311841


namespace function_increasing_on_interval_l311_311667

theorem function_increasing_on_interval : ∀ x ∈ set.Ioo (0 : ℝ) π, 
  ∀ y ∈ set.Ioo (0 : ℝ) π, x < y → (x + sin x) < (y + sin y) :=
by
  intros x hx y hy hxy
  -- Proof skipped
  sorry

end function_increasing_on_interval_l311_311667


namespace sum_of_divisors_of_prime_l311_311785

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311785


namespace race_permutations_l311_311504

theorem race_permutations (Harry Ron Neville : Type) [decidable_eq Harry] [decidable_eq Ron] [decidable_eq Neville] : 
  (finset.univ : finset (Harry × Ron × Neville)).card = 6 := 
begin
  -- Define the contestants as distinct elements
  let contestants := {Harry, Ron, Neville},

  -- Calculate the number of permutations
  have h_perms : finset.card (finset.univ : finset (perm contestants)) = 6,
  { sorry },

  -- Finish the proof
  exact h_perms,
end

end race_permutations_l311_311504


namespace sum_of_divisors_of_29_l311_311766

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311766


namespace simplify_sqrt_l311_311193

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311193


namespace blue_balls_count_l311_311535

theorem blue_balls_count:
  ∀ (T : ℕ),
  (1/4 * T) + (1/8 * T) + (1/12 * T) + 26 = T → 
  (1 / 8) * T = 6 := by
  intros T h
  sorry

end blue_balls_count_l311_311535


namespace simplify_sqrt_l311_311191

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311191


namespace sum_of_divisors_of_29_l311_311713

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311713


namespace sum_of_divisors_of_29_l311_311752

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311752


namespace fib_series_sum_l311_311593

open Nat

-- Define the Fibonacci sequence as a recursive function
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define the infinite sum of Fib_n / 2^n
def fib_series : Real :=
∑' n, (fib n : Real) / (2^n : Real)

-- The theorem stating the result of the infinite sum
theorem fib_series_sum : fib_series = 4 / 5 :=
by
  sorry

end fib_series_sum_l311_311593


namespace simplify_cuberoot_product_l311_311233

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311233


namespace area_of_parallelogram_l311_311610

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l311_311610


namespace hexagon_largest_angle_l311_311365

theorem hexagon_largest_angle (x : ℝ) 
    (h_sum : (x + 2) + (2*x + 4) + (3*x - 6) + (4*x + 8) + (5*x - 10) + (6*x + 12) = 720) :
    (6*x + 12) = 215 :=
by
  sorry

end hexagon_largest_angle_l311_311365


namespace normal_mean_is_zero_if_symmetric_l311_311675

-- Definition: A normal distribution with mean μ and standard deviation σ.
structure NormalDist where
  μ : ℝ
  σ : ℝ

-- Condition: The normal curve is symmetric about the y-axis.
def symmetric_about_y_axis (nd : NormalDist) : Prop :=
  nd.μ = 0

-- Theorem: If the normal curve is symmetric about the y-axis, then the mean μ of the corresponding normal distribution is 0.
theorem normal_mean_is_zero_if_symmetric (nd : NormalDist) (h : symmetric_about_y_axis nd) : nd.μ = 0 := 
by sorry

end normal_mean_is_zero_if_symmetric_l311_311675


namespace sum_of_torn_pages_not_1990_l311_311176

theorem sum_of_torn_pages_not_1990 (sheets : ℕ) (total_pages : ℕ) (torn_sheets : ℕ) (total_sum : ℕ) : 
  sheets = 96 → total_pages = 192 → torn_sheets = 25 → ∀ torn_page_nums : Finite 50 → total_sum ≠ 1990 :=
by
  intros h1 h2 h3 h4
  have h5 : (∑ i in torn_page_nums, (i : ℤ)) = 4 * (∑ i in (Finset.range 25), (i : ℤ)) - 25
  { sorry } -- skip proof details
  have h6 : total_sum = (∑ i in torn_page_nums, (i : ℤ))
  { sorry } -- skipping proof
  rw [h6, h5]
  have h7 : (4 * (∑ i in (Finset.range 25), (i : ℤ)) - 25) ≠ 1990
  { sorry } -- Details proving contradiction, omitted for brevity
  exact h7

end sum_of_torn_pages_not_1990_l311_311176


namespace sum_of_divisors_of_prime_29_l311_311843

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311843


namespace sophie_total_spend_l311_311255

-- Definitions based on conditions
def cost_cupcakes : ℕ := 5 * 2
def cost_doughnuts : ℕ := 6 * 1
def cost_apple_pie : ℕ := 4 * 2
def cost_cookies : ℕ := 15 * 6 / 10 -- since 0.60 = 6/10

-- Total cost
def total_cost : ℕ := cost_cupcakes + cost_doughnuts + cost_apple_pie + cost_cookies

-- Prove the total cost
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end sophie_total_spend_l311_311255


namespace chocolates_sold_l311_311524

theorem chocolates_sold (C S : ℝ) (n : ℕ) (h1 : 81 * C = n * S) (h2 : (S - C) / C * 100 = 80) : n = 45 :=
by
  have h3 : S / C = 1.8,
  { field_simp at h2, linarith },
  have eq1 : 81 * C = n * (1.8 * C),
  { rw <- h3 at h1, exact h1 },
  field_simp at eq1,
  linarith,
  sorry

end chocolates_sold_l311_311524


namespace cos_theta_equal_neg_inv_sqrt_5_l311_311892

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x - Real.cos x

theorem cos_theta_equal_neg_inv_sqrt_5 (θ : ℝ) (h_max : ∀ x : ℝ, f θ ≥ f x) : Real.cos θ = -1 / Real.sqrt 5 :=
by
  sorry

end cos_theta_equal_neg_inv_sqrt_5_l311_311892


namespace angle_NOQ_60_degrees_l311_311545

-- Definition of the equilateral triangle and the conditions
structure EquilateralTriangle (A B C : Type) :=
  (eq_side_length : ∀ (a b : A), a ≠ b → dist a b = a)

structure PointsPositioned (A M N P Q : Type) :=
  (dist_MA_AN : ∀ (a : A), dist M A + dist A N = a)
  (dist_PC_CQ : ∀ (a : A), dist P C + dist C Q = a)

-- Main theorem to prove
theorem angle_NOQ_60_degrees (A B C : Type) (a : ℝ) [equilateral : EquilateralTriangle A B C]
  (M N P Q : Type) [points : PointsPositioned (A M N P Q)] : 
  ∃ O : Type, ∠NOQ = 60 :=
begin
  sorry
end

end angle_NOQ_60_degrees_l311_311545


namespace final_volume_syrup_l311_311963

-- Definitions of the problem conditions
def initial_volume_in_quarts : ℕ := 6
def reduction_factor : ℚ := 1 / 12
def sugar_added_in_cups : ℕ := 1
def cups_per_quart : ℕ := 4

-- The statement to be proved
theorem final_volume_syrup :
  let initial_volume_in_cups := initial_volume_in_quarts * cups_per_quart in
  let reduced_volume_in_cups := initial_volume_in_cups * reduction_factor in
  let final_volume := reduced_volume_in_cups + sugar_added_in_cups in
  final_volume = 3 :=
by
  sorry

end final_volume_syrup_l311_311963


namespace hours_spent_writing_l311_311128

-- Define the rates at which Jacob and Nathan write
def Nathan_rate : ℕ := 25        -- Nathan writes 25 letters per hour
def Jacob_rate : ℕ := 2 * Nathan_rate  -- Jacob writes twice as fast as Nathan

-- Define the combined rate
def combined_rate : ℕ := Nathan_rate + Jacob_rate

-- Define the total letters written and the hours spent
def total_letters : ℕ := 750
def hours_spent : ℕ := total_letters / combined_rate

-- The theorem to prove
theorem hours_spent_writing : hours_spent = 10 :=
by 
  -- Placeholder for the proof
  sorry

end hours_spent_writing_l311_311128


namespace sum_of_coordinates_l311_311478

-- Given conditions: g(8) = 5
def g (x : ℝ) : ℝ := sorry

axiom g_8_eq_5 : g 8 = 5

-- The point to be checked
def point := (8 / 3 : ℝ, 14 / 9 : ℝ)

-- Proof problem: sum of the coordinates of point on the graph
theorem sum_of_coordinates : (point.1 + point.2) = 38 / 9 := sorry

end sum_of_coordinates_l311_311478


namespace sum_of_divisors_of_29_l311_311815

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311815


namespace f_sum_value_l311_311414

def f : ℝ → ℝ := sorry

axiom f_odd (x : ℝ) : f(x) + f(-x) = 0
axiom f_periodic (x : ℝ) : f(x) = f(x + 2)
axiom f_piecewise (x : ℝ) (h : 0 ≤ x ∧ x < 1) : f(x) = 2 * x - 1

theorem f_sum_value :
  f(1 / 2) + f(1) + f(3 / 2) + f(5 / 2) = Real.sqrt 2 - 1 :=
sorry

end f_sum_value_l311_311414


namespace sum_of_divisors_of_29_l311_311738

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311738


namespace count_of_digit_sum_11_l311_311434

-- Function to sum the digits of an integer
def digit_sum (n : ℕ) : ℕ :=
  (n.toDigits 10).sum

-- Set of integers from 1 to 2009
def integer_set : Finset ℕ := (Finset.range 2009).map (Embedding.ofStrictMono Nat.succ)

-- Condition: Counting elements with digit sum equal to 11
def count_digit_sum_11 : ℕ :=
  (integer_set.filter (λ n => digit_sum n = 11)).card

-- Theorem to be proven
theorem count_of_digit_sum_11 : count_digit_sum_11 = 133 := by
  sorry

end count_of_digit_sum_11_l311_311434


namespace deposit_percentage_is_10_l311_311362

-- Define the deposit and remaining amount
def deposit := 120
def remaining := 1080

-- Define total cost
def total_cost := deposit + remaining

-- Define deposit percentage calculation
def deposit_percentage := (deposit / total_cost) * 100

-- Theorem to prove the deposit percentage is 10%
theorem deposit_percentage_is_10 : deposit_percentage = 10 := by
  -- Since deposit, remaining and total_cost are defined explicitly,
  -- the proof verification of final result is straightforward.
  sorry

end deposit_percentage_is_10_l311_311362


namespace sum_of_divisors_of_29_l311_311757

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311757


namespace license_plate_possibilities_count_l311_311898

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

def digits : Finset Char := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

theorem license_plate_possibilities_count : 
  (vowels.card * digits.card * 2 = 100) := 
by {
  -- vowels.card = 5 because there are 5 vowels.
  -- digits.card = 10 because there are 10 digits.
  -- 2 because the middle character must match either the first vowel or the last digit.
  sorry
}

end license_plate_possibilities_count_l311_311898


namespace find_QT_sum_l311_311695

noncomputable def triangle_PQR_S_T_U := 
let PQ := 5 : ℝ;
let QR := 6 : ℝ;
let PR := 7 : ℝ;
let SU := 3 : ℝ;
let TU := 8 : ℝ;
exists (QT : ℝ), 
QT = (e : ℝ) + (f : ℝ) * sqrt(g : ℝ) / (h : ℝ) ∧
(e : ℕ ∧ f : ℕ ∧ g : ℕ ∧ h : ℕ) ∧
nat.coprime e h ∧
¬∃ (p : ℕ), (prime p) ∧ (p * p) ∣ g ∧
(e + f + g + h = ?)

/-- Define the equivalent problem that corresponds to determining QT in the given problem. -/
theorem find_QT_sum (PQ QR PR SU TU : ℝ) (e f g h : ℕ) :
PQ = 5 ∧ QR = 6 ∧ PR = 7 ∧ SU = 3 ∧ TU = 8 ∧
(∃ (QT : ℝ), QT = (e + f * sqrt (g : ℝ)) / (h : ℝ) ∧
nat.coprime e h ∧
¬∃ (p : ℕ), (nat.prime p) ∧ (p * p ∣ g) ∧
(e + f + g + h = ?)) :=
by sorry

end find_QT_sum_l311_311695


namespace max_attendance_days_l311_311947

-- Define days of the week
inductive Day : Type
  | Mon | Tues | Wed | Thurs | Fri

open Day

-- Define team member unavailability
def unavailable (member : String) (day : Day) : Prop :=
  (member = "Alice" ∧ (day = Mon ∨ day = Thurs)) ∨
  (member = "Bob" ∧ (day = Tues ∨ day = Fri)) ∨
  (member = "Charlie" ∧ (day = Mon ∨ day = Tues ∨ day = Thurs ∨ day = Fri)) ∨
  (member = "Diana" ∧ (day = Wed ∨ day = Thurs)) ∨
  (member = "Edward" ∧ day = Wed)

-- Define the number of attendees for each day
def attendees (day : Day) : Nat :=
  (if unavailable "Alice" day then 0 else 1) +
  (if unavailable "Bob" day then 0 else 1) +
  (if unavailable "Charlie" day then 0 else 1) +
  (if unavailable "Diana" day then 0 else 1) +
  (if unavailable "Edward" day then 0 else 1)

-- The theorem we want to prove
theorem max_attendance_days :
  (attendees Mon = 3 ∧ attendees Wed = 3 ∧ attendees Fri = 3) ∧
  (attendees Mon ≥ attendees Tues) ∧
  (attendees Mon ≥ attendees Thurs) ∧
  (attendees Wed ≥ attendees Tues) ∧
  (attendees Wed ≥ attendees Thurs) ∧
  (attendees Fri ≥ attendees Tues) ∧
  (attendees Fri ≥ attendees Thurs) :=
by
  sorry

end max_attendance_days_l311_311947


namespace range_of_a_l311_311057

def f (x : ℝ) : ℝ := Real.log (abs x)

theorem range_of_a (a : ℝ) : f 1 < f a → a > 1 ∨ a < -1 :=
by
  sorry

end range_of_a_l311_311057


namespace option_B_option_C_option_D_l311_311058

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  (sin x) ^ n + (tan x) ^ n

theorem option_B (n : ℕ) (h : n > 0) :
  ∀ x, 0 < x ∧ x < π / 2 → deriv (λ x : ℝ, (f n x)) x > 0 := by
sorry

theorem option_C (k : ℕ) (h : k > 0) :
  ∀ x, f (2 * k - 1) (-x) = - (f (2 * k - 1) x) := by
sorry

theorem option_D (k : ℕ) (h : k > 0) :
  ∀ x, f (2 * k) (π - x) = f (2 * k) x := by
sorry

end option_B_option_C_option_D_l311_311058


namespace log_expression_value_l311_311890

theorem log_expression_value :
  (π : ℝ) → (logBase : ℝ → ℝ → ℝ) → 
  (logBase 2 120 / logBase 60 2) - (logBase 2 240 / logBase 30 2) = 2 :=
by
  -- Mathematical definitions for convenience (these are assumed already known)
  let log₂ := λ x : ℝ, Real.log x / Real.log 2
  let log₆₀ := λ x : ℝ, Real.log x / Real.log 60
  let log₃₀ := λ x : ℝ, Real.log x / Real.log 30

  -- Rewrite the given expression using the mathematical definitions
  let expr := (log₂ 120 / log₆₀ 2) - (log₂ 240 / log₃₀ 2)
  have h: expr = 2,
  from sorry -- proof will be completed here

  exact h

end log_expression_value_l311_311890


namespace parallelogram_area_l311_311635

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l311_311635


namespace locus_of_circle_centers_l311_311986

noncomputable def distance (A B: ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem locus_of_circle_centers (A B : ℝ × ℝ) (r : ℝ)
  (h : distance A B ≤ 2 * r) :
  ∃ C : ℝ × ℝ, ∀ (P : ℝ × ℝ), 
  (distance P A = r ∧ distance P B = r) ↔ 
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0 :=
sorry

end locus_of_circle_centers_l311_311986


namespace find_triplets_find_triplets_non_negative_l311_311002

theorem find_triplets :
  ∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by
  sorry

theorem find_triplets_non_negative :
  ∀ (x y z : ℕ), x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_triplets_find_triplets_non_negative_l311_311002


namespace simplified_expression_correct_l311_311212

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311212


namespace coeff_x_31_eq_709_l311_311985

theorem coeff_x_31_eq_709 (x : ℝ) : 
  polynomial.coeff ((polynomial.sum (list.range 31) (λ n, (x ^ n))) *
                    (polynomial.sum (list.range 19) (λ n, (x ^ n)))^2) 31 = 709 := 
sorry

end coeff_x_31_eq_709_l311_311985


namespace problem_part_c_problem_part_d_l311_311551

noncomputable def binomial_expansion_sum_coefficients : ℕ :=
  (1 + (2 : ℚ)) ^ 6

theorem problem_part_c :
  binomial_expansion_sum_coefficients = 729 := 
by sorry

noncomputable def binomial_expansion_sum_binomial_coefficients : ℕ :=
  (2 : ℚ) ^ 6

theorem problem_part_d :
  binomial_expansion_sum_binomial_coefficients = 64 := 
by sorry

end problem_part_c_problem_part_d_l311_311551


namespace heejin_most_balls_is_volleyballs_l311_311076

def heejin_basketballs : ℕ := 3
def heejin_volleyballs : ℕ := 5
def heejin_baseballs : ℕ := 1

theorem heejin_most_balls_is_volleyballs :
  heejin_volleyballs > heejin_basketballs ∧ heejin_volleyballs > heejin_baseballs :=
by
  sorry

end heejin_most_balls_is_volleyballs_l311_311076


namespace sum_of_divisors_29_l311_311868

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311868


namespace find_n_from_sequence_l311_311063

theorem find_n_from_sequence (a : ℕ → ℝ) (h₁ : ∀ n : ℕ, a n = (1 / (Real.sqrt n + Real.sqrt (n + 1))))
  (h₂ : ∃ n : ℕ, a n + a (n + 1) = Real.sqrt 11 - 3) : 9 ∈ {n | a n + a (n + 1) = Real.sqrt 11 - 3} :=
by
  sorry

end find_n_from_sequence_l311_311063


namespace sum_of_divisors_of_29_l311_311708

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311708


namespace length_of_plot_l311_311941

theorem length_of_plot (total_poles : ℕ) (distance : ℕ) (one_side : ℕ) (other_side : ℕ) 
  (poles_distance_condition : total_poles = 28) 
  (fencing_condition : distance = 10) 
  (side_condition : one_side = 50) 
  (rectangular_condition : total_poles = (2 * (one_side / distance) + 2 * (other_side / distance))) :
  other_side = 120 :=
by
  sorry

end length_of_plot_l311_311941


namespace sum_of_divisors_29_l311_311875

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311875


namespace max_candies_one_student_l311_311541

theorem max_candies_one_student (num_students : ℕ) (mean_candies : ℕ) (mean_stickers : ℕ) (at_least_one_candy : ∀ i : Fin num_students, 1 ≤ candies i) (at_least_one_sticker : ∀ i : Fin num_students, 1 ≤ stickers i) (sum_candies : ∑ i, candies i = num_students * mean_candies) : 
  ∃ i : Fin num_students, candies i = num_students * mean_candies - (num_students - 1) :=
  sorry

end max_candies_one_student_l311_311541


namespace power_function_eval_l311_311049

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_eval (α : ℝ) (h : 2 ^ α = 1 / 4) : f (sqrt 2) α = 1 / 2 := by
  sorry

end power_function_eval_l311_311049


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311624

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311624


namespace smallest_integer_smaller_expansion_l311_311312

theorem smallest_integer_smaller_expansion :
  ∀ x : ℝ, (x = (5^.5 + 3^.5)) → (floor ((x)^6) = 3322) :=
by
  intros x hx
  sorry

end smallest_integer_smaller_expansion_l311_311312


namespace dot_product_OC_AB_eq_neg_nine_l311_311041

variable (O A B C : Type)
variable [inhabited O] [inhabited A] [inhabited B] [inhabited C]
variables (AC BC OC AB: ℝ)
variables (angle_ABC angle_AOC : ℝ)
variables (perpendicular_AC_BC : Prop)
variables (is_circumcenter : Prop)

noncomputable def dot_product_OC_AB : ℝ :=
  OC * AB * real.cos (2 * real.pi / 3)

theorem dot_product_OC_AB_eq_neg_nine
  (h1 : is_circumcenter)
  (h2 : AC = 3)
  (h3 : perpendicular_AC_BC)
  (h4 : angle_ABC = real.pi / 6)
  (h5 : AB = 6)
  (h6 : angle_AOC = real.pi / 3)
  (h7 : OC = 3) :
  dot_product_OC_AB O A B C AC BC OC AB angle_ABC angle_AOC perpendicular_AC_BC is_circumcenter = -9 :=
sorry

end dot_product_OC_AB_eq_neg_nine_l311_311041


namespace sum_of_divisors_of_prime_l311_311791

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311791


namespace simplify_cuberoot_product_l311_311230

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311230


namespace square_side_length_l311_311267

theorem square_side_length (d : ℝ) (sqrt_2_ne_zero : sqrt 2 ≠ 0) (h : d = 2 * sqrt 2) : 
  ∃ (s : ℝ), s = 2 ∧ d = s * sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [mul_comm, ←mul_assoc, eq_comm, mul_right_comm, mul_div_cancel, h, mul_comm]
    · exact sqrt_2_ne_zero
  sorry

end square_side_length_l311_311267


namespace cube_root_multiplication_l311_311201

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311201


namespace parallelogram_area_l311_311617

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l311_311617


namespace journey_time_l311_311924

theorem journey_time (total_distance : ℝ) (distance_40 : ℝ) (speed_40 : ℝ) (speed_60 : ℝ) 
  (h1 : total_distance = 250) (h2 : distance_40 = 124) 
  (h3 : speed_40 = 40) (h4 : speed_60 = 60) :
  let distance_60 := total_distance - distance_40 in
  let time_40 := distance_40 / speed_40 in
  let time_60 := distance_60 / speed_60 in
  time_40 + time_60 = 5.2 :=
by
  sorry

end journey_time_l311_311924


namespace line_parallel_or_contained_in_plane_l311_311596

-- Definitions based on the conditions in (a)
def direction_vector_of_line : ℝ × ℝ × ℝ := (3, -2, -1)
def normal_vector_of_plane : ℝ × ℝ × ℝ := (-1, -2, 1)

-- The primary theorem based on (c)
theorem line_parallel_or_contained_in_plane :
  let a := direction_vector_of_line,
      n := normal_vector_of_plane in
  a.1 * n.1 + a.2 * n.2 + a.3 * n.3 = 0 → (∀ β : Type, true) :=
by
  intros
  sorry

end line_parallel_or_contained_in_plane_l311_311596


namespace simplify_sqrt_l311_311194

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311194


namespace simplify_cuberoot_product_l311_311225

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311225


namespace sum_of_divisors_of_29_l311_311802

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311802


namespace sum_of_divisors_of_29_l311_311804

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311804


namespace quadrilateral_divided_into_equal_areas_l311_311378

theorem quadrilateral_divided_into_equal_areas
  (A B C D M N K L E F P : Point)
  (h_convex : convex_quad A B C D)
  (hM : midpoint A B M)
  (hN : midpoint B C N)
  (hK : midpoint C D K)
  (hL : midpoint D A L)
  (hE : midpoint A C E)
  (hF : midpoint B D F)
  (h_parallel_E : ∃ (f : Line), is_parallel f (line_through B D) ∧ is_line_through f E)
  (h_parallel_F : ∃ (g : Line), is_parallel g (line_through A C) ∧ is_line_through g F)
  (h_intersect_P : intersection (proj_lines E (line_through B D)) (proj_lines F (line_through A C)) = P)
  (h_lines_P : line_through P M ∧ line_through P N ∧ line_through P K ∧ line_through P L) :
  area (quad A L P M) = \frac{1}{4} (area (quad A B C D))
  ∧ area (quad B M P N) = \frac{1}{4} (area (quad A B C D))
  ∧ area (quad C N P K) = \frac{1}{4} (area (quad A B C D))
  ∧ area (quad D K P L) = \frac{1}{4} (area (quad A B C D)) :=
by sorry

end quadrilateral_divided_into_equal_areas_l311_311378


namespace area_of_right_triangle_l311_311303

-- Define the right triangle.
variables (A B C D : Type*)
variables (dist : B → B → ℝ)
variables [metric_space B] [normed_space ℝ B]

-- Conditions
def is_right_triangle (ABC : triangle B) : Prop :=
ABC.angle B = π / 2

def foot_of_perpendicular (D B A C : B) : Prop :=
(perpendicular (line B A) (line B C) ∧  dist A D = 5 ∧ dist D C = 12)

-- We want to prove this
theorem area_of_right_triangle (ABC : triangle B)
    (h_right : is_right_triangle ABC)
    (h_foot : foot_of_perpendicular D B A C) :
     triangle.area ABC = 17 * real.sqrt 15 :=
sorry

end area_of_right_triangle_l311_311303


namespace inscribed_quadrilateral_bound_l311_311639

theorem inscribed_quadrilateral_bound:
  ∀ (T : Type) [triangle T] (ABC : T) (A B C D E F G H I J : point T),
  (is_equilateral_triangle ABC 1) →
  (is_convex_quadrilateral_in_triangle (A B C) (D E F G)) →
  side_length D E > 1/2 →
  side_length E F > 1/2 →
  side_length F G > 1/2 →
  side_length G D > 1/2 →
  false :=
begin
  sorry
end

end inscribed_quadrilateral_bound_l311_311639


namespace ratio_surface_area_cube_to_octahedron_l311_311366

noncomputable def cube_side_length := 1

noncomputable def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

noncomputable def edge_length_octahedron := 1

-- Surface area formula for a regular octahedron with side length e is 2 * sqrt(3) * e^2
noncomputable def surface_area_octahedron (e : ℝ) : ℝ := 2 * Real.sqrt 3 * e^2

-- Finally, we want to prove that the ratio of the surface area of the cube to that of the octahedron is sqrt(3)
theorem ratio_surface_area_cube_to_octahedron :
  surface_area_cube cube_side_length / surface_area_octahedron edge_length_octahedron = Real.sqrt 3 :=
by sorry

end ratio_surface_area_cube_to_octahedron_l311_311366


namespace simplified_expression_correct_l311_311206

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311206


namespace boxes_of_erasers_donated_l311_311292

theorem boxes_of_erasers_donated (erasers_per_box : ℕ) (price_per_eraser : ℚ) (total_money_raised : ℚ) : 
  erasers_per_box = 24 → price_per_eraser = 0.75 → total_money_raised = 864 →
  (total_money_raised / price_per_eraser) / erasers_per_box = 48 :=
by
  intros h1 h2 h3 
  rw [h1, h2, h3]
  norm_num
  sorry

end boxes_of_erasers_donated_l311_311292


namespace Brady_average_hours_l311_311966

-- Definitions based on conditions
def hours_per_day_April : ℕ := 6
def hours_per_day_June : ℕ := 5
def hours_per_day_September : ℕ := 8
def days_in_April : ℕ := 30
def days_in_June : ℕ := 30
def days_in_September : ℕ := 30

-- Definition to prove
def average_hours_per_month : ℕ := 190

-- Theorem statement
theorem Brady_average_hours :
  (hours_per_day_April * days_in_April + hours_per_day_June * days_in_June + hours_per_day_September * days_in_September) / 3 = average_hours_per_month :=
sorry

end Brady_average_hours_l311_311966


namespace mean_is_not_8point4_l311_311542

theorem mean_is_not_8point4 (scores : List ℕ) (h : scores = [8, 10, 9, 7, 7, 9, 8, 9]) :
  (List.sum scores) / scores.length ≠ 8.4 := by
  sorry

end mean_is_not_8point4_l311_311542


namespace determine_a_l311_311274

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.sqrt 3 * Real.cos x

def g (a : ℝ) (x : ℝ) : ℝ := f a (x - Real.pi / 3)

theorem determine_a (a : ℝ) :
  (∀ x : ℝ, g a (Real.pi / 6 - x) = g a (Real.pi / 6 + x)) →
  a = -1 :=
sorry

end determine_a_l311_311274


namespace find_principal_amount_l311_311283

theorem find_principal_amount :
  ∃ P : ℝ, P * (1 + 0.05) ^ 4 = 9724.05 ∧ P = 8000 :=
by
  sorry

end find_principal_amount_l311_311283


namespace cody_tickets_l311_311981

-- Definitions
def children_ticket_cost : ℝ := 7.5
def adult_ticket_cost : ℝ := 12
def ticket_threshold : ℕ := 20
def total_cost : ℝ := 138

-- Variables
variables (A C : ℕ)
-- Conditions
axiom children_ticket_more : C = A + 8
axiom total_bill : 138 = (12 * A + 7.5 * C)

-- The Lean statement to prove
theorem cody_tickets (children_ticket_more : C = A + 8)
                     (total_bill : 138 = (12 * A + 7.5 * C)) :
  A = 4 ∧ C = 12 ∧ (A + C ≤ ticket_threshold) ∧ (A + C ≤ 16) :=
by {
  sorry
}

end cody_tickets_l311_311981


namespace final_number_5039_l311_311168

theorem final_number_5039 :
  ∃ x,
    x = 5039 ∧
    (∃ n (initial_numbers : Fin n → ℕ),
     initial_numbers = ![1, 2, 3, 4, 5, 6] ∧
     (∀ a b, a ∈ (Finset.image initial_numbers Finset.univ : Finset ℕ) →
             b ∈ (Finset.image initial_numbers Finset.univ : Finset ℕ) →
             ∃ i j, initial_numbers i = a ∧ initial_numbers j = b →
                    ∃ (new_nums : Fin n → ℕ), 
                     new_nums = initial_numbers.update_nth i ((a+1) * (b+1) - 1) ∧ 
                     ∀ k, k ≠ i → new_nums k = initial_numbers k) →
     (∀ remaining_numbers : Fin n → ℕ, remaining_numbers = initial_numbers → n = 1 → remaining_numbers 0 = x)) := sorry

end final_number_5039_l311_311168


namespace parabola_intersection_difference_l311_311282

noncomputable def parabola1 (x : ℝ) := 3 * x^2 - 6 * x + 6
noncomputable def parabola2 (x : ℝ) := -2 * x^2 + 2 * x + 6

theorem parabola_intersection_difference :
  let a := 0
  let c := 8 / 5
  c - a = 8 / 5 := by
  sorry

end parabola_intersection_difference_l311_311282


namespace area_of_parallelogram_l311_311607

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l311_311607


namespace sum_of_divisors_29_l311_311865

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311865


namespace train_stops_per_hour_l311_311326

-- Define the speeds
def speed_excluding_stoppages : ℝ := 45 -- kmph
def speed_including_stoppages : ℝ := 34 -- kmph

-- Calculation of the time the train stops per hour
noncomputable def time_stopped_per_hour (se ss : ℝ) : ℝ :=
  let distance_lost := se - ss in
  let speed_per_minute := se / 60 in
  distance_lost / speed_per_minute

theorem train_stops_per_hour :
  time_stopped_per_hour speed_excluding_stoppages speed_including_stoppages ≈ 14.67 := 
sorry

end train_stops_per_hour_l311_311326


namespace slope_of_line_l311_311440

variable (s : ℝ) -- real number s

def line1 (x y : ℝ) := x + 3 * y = 9 * s + 4
def line2 (x y : ℝ) := x - 2 * y = 3 * s - 3

theorem slope_of_line (s : ℝ) :
  ∀ (x y : ℝ), (line1 s x y ∧ line2 s x y) → y = (2 / 9) * x + (13 / 9) :=
sorry

end slope_of_line_l311_311440


namespace alejandro_candies_l311_311912

theorem alejandro_candies (n : ℕ) (S_n : ℕ) :
  (S_n = 2^n - 1 ∧ S_n ≥ 2007) → ((2^11 - 1 - 2007 = 40) ∧ (∃ k, k = 11)) :=
  by
    sorry

end alejandro_candies_l311_311912


namespace vector_magnitude_l311_311659

def a := (3, 0)
def b : ℝ × ℝ
def angle_ab := (2 * Real.pi / 3 : ℝ)
def norm_b := 2

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def vector_length (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

noncomputable def result : ℝ :=
  vector_length (a.1 + 2 * b.1, a.2 + 2 * b.2)

theorem vector_magnitude :
  angle_ab = (2 * Real.pi / 3 : ℝ) ∧
  a = (3, 0) ∧ 
  vector_length b = 2 → 
  result = Real.sqrt 13 :=
begin
  sorry
end

end vector_magnitude_l311_311659


namespace polar_to_cartesian_and_parametric_equiv_l311_311497

variable (ρ θ : ℝ)
variable (x y α : ℝ)

def polar_equation := ρ^2 - 4 * real.sqrt 2 * ρ * real.cos (θ - real.pi / 4) + 6 = 0

def cartesian_equation (x y : ℝ) := (x - 2)^2 + (y - 2)^2 = 2

def parametric_eqs (α : ℝ) : Prop :=
  (x = 2 + real.sqrt 2 * real.cos α) ∧ (y = 2 + real.sqrt 2 * real.sin α)

def conditions (ρ θ : ℝ): Prop :=
  polar_equation ρ θ

theorem polar_to_cartesian_and_parametric_equiv :
  ∀ (ρ θ : ℝ),
    conditions ρ θ →
    ∀ (x y α : ℝ),
      cartesian_equation x y ∧ parametric_eqs α →
      ∃ (min_val max_val : ℝ), min_val = 2 ∧ max_val = 6 :=
by intros; sorry

end polar_to_cartesian_and_parametric_equiv_l311_311497


namespace find_factor_l311_311945

-- Definitions based on the conditions
def n : ℤ := 155
def result : ℤ := 110
def constant : ℤ := 200

-- Statement to be proved
theorem find_factor (f : ℤ) (h : n * f - constant = result) : f = 2 := by
  sorry

end find_factor_l311_311945


namespace find_a_l311_311412

-- Condition: Define a * b as 2a - b^2
def star (a b : ℝ) := 2 * a - b^2

-- Proof problem: Prove the value of a given the condition and that a * 7 = 16.
theorem find_a : ∃ a : ℝ, star a 7 = 16 ∧ a = 32.5 :=
by
  sorry

end find_a_l311_311412


namespace quadratic_roots_form_l311_311436

theorem quadratic_roots_form {a b c : ℤ} (h : a = 3 ∧ b = -7 ∧ c = 1) :
  ∃ (m n p : ℤ), (∀ x, 3*x^2 - 7*x + 1 = 0 ↔ x = (m + Real.sqrt n)/p ∨ x = (m - Real.sqrt n)/p)
  ∧ Int.gcd m (Int.gcd n p) = 1 ∧ n = 37 :=
by
  sorry

end quadratic_roots_form_l311_311436


namespace sum_of_reciprocals_of_roots_l311_311157

theorem sum_of_reciprocals_of_roots : 
  let f := (6 : ℚ) * X ^ 2 - (5 : ℚ) * X + 3 in
  ∀ (a b : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ (a ≠ b) ∧ is_root f a ∧ is_root f b → (1 / a + 1 / b = 5 / 3) :=
by
  sorry

end sum_of_reciprocals_of_roots_l311_311157


namespace sum_of_divisors_of_29_l311_311884

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311884


namespace finite_set_satisfies_condition_l311_311138

def point_on_circle (θ : ℝ) : ℂ := complex.exp (2 * ↑(real.pi) * complex.I * θ)

def is_partition {α : Type*} (A B : finset α) (S : finset α) : Prop :=
  A ∪ B = S ∧ A ∩ B = ∅

def satisfies_condition (S : finset ℂ) : Prop :=
  ∃ (ε : S → (ℤ)), (∀ (x : S), (ε x = 1 ∨ epsilon x = -1)) ∧ (∑ x in S, (ε x) • x) = 0

noncomputable def satisfies_partition_condition (Γ : set ℂ) (S : finset ℂ) : Prop :=
  ∀ P ∈ Γ, ∃ (A B : finset ℂ), is_partition A B S ∧ ∑ x in A, complex.abs (P - x) = ∑ y in B, complex.abs (P - y)

theorem finite_set_satisfies_condition (Γ : set ℂ) (S : finset ℂ) :
  satisfies_condition S ↔ satisfies_partition_condition Γ S :=
sorry

end finite_set_satisfies_condition_l311_311138


namespace parallelogram_area_proof_l311_311629

noncomputable def parallelogram_area : ℝ :=
  let angle_rad := (150 * real.pi / 180)  -- converting degrees to radians
  let a := 10                              -- length of one side
  let b := 20                              -- length of another side
  let height := a * real.sqrt(3) / 2       -- height from 30-60-90 triangle properties
  b * height

theorem parallelogram_area_proof : parallelogram_area = 100 * real.sqrt(3) := by
  sorry

end parallelogram_area_proof_l311_311629


namespace cos_sub_sin_l311_311086

theorem cos_sub_sin (x : ℝ) (h : x = π / 12) : cos x - sin x = sqrt 2 / 2 :=
by
  rw [h]
  sorry

end cos_sub_sin_l311_311086


namespace horse_speed_l311_311266

theorem horse_speed (area : ℝ) (time : ℝ) (side_length : ℝ) (perimeter : ℝ) (speed : ℝ) 
  (h1 : area = 900)
  (h2 : time = 10)
  (h3 : side_length = real.sqrt area)
  (h4 : perimeter = 4 * side_length)
  (h5 : speed = perimeter / time) : 
  speed = 12 := 
sorry

end horse_speed_l311_311266


namespace parallelogram_area_l311_311613

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l311_311613


namespace probability_blue_balls_first_l311_311354

-- Define the conditions
def total_balls : ℕ := 7
def blue_balls : ℕ := 4
def yellow_balls : ℕ := 3

-- Calculate combinations
def total_arrangements : ℕ := Nat.choose total_balls yellow_balls
def favorable_arrangements : ℕ := Nat.choose (total_balls - 1) (yellow_balls)

-- Define the probability
noncomputable def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable.to_rat / total.to_rat

-- State the proof problem
theorem probability_blue_balls_first :
  probability favorable_arrangements total_arrangements = 4 / 7 := by
  sorry

end probability_blue_balls_first_l311_311354


namespace max_quizzes_lower_than_A_l311_311397

theorem max_quizzes_lower_than_A (total_quizzes : ℕ) (required_percentage : ℚ) (quizzes_taken : ℕ) (quizzes_A : ℕ) :
  total_quizzes = 60 ∧ required_percentage = 0.75 ∧ quizzes_taken = 40 ∧ quizzes_A = 30 →
  (∃ remaining_quizzes remaining_A,
    remaining_quizzes = total_quizzes - quizzes_taken ∧
    required_A = (required_percentage * total_quizzes : ℚ).toNat - quizzes_A ∧
    remaining_quizzes - required_A = 5) :=
sorry

end max_quizzes_lower_than_A_l311_311397


namespace line_parallel_not_passing_through_point_l311_311042

noncomputable def point_outside_line (A B C x0 y0 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (A * x0 + B * y0 + C = k)

theorem line_parallel_not_passing_through_point 
  (A B C x0 y0 : ℝ) (h : point_outside_line A B C x0 y0) :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x y : ℝ, Ax + By + C + k = 0 → Ax_0 + By_0 + C + k ≠ 0) :=
sorry

end line_parallel_not_passing_through_point_l311_311042


namespace sum_of_divisors_29_l311_311825

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311825


namespace sum_of_divisors_of_29_l311_311881

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311881


namespace sum_of_divisors_prime_29_l311_311732

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311732


namespace determine_mnp_l311_311983

noncomputable def volume_of_set (a b c : ℝ) (r : ℝ) : ℝ :=
  let volume_parallelepiped := a * b * c
  let volume_external := 2 * ((a * b * r) + (a * c * r) + (b * c * r))
  let volume_spheres := (8 / 3) * Real.pi * r^3
  let length_cylinders := a + b + c
  let volume_cylinders := length_cylinders * Real.pi * r^2
  (volume_parallelepiped + volume_external + volume_spheres + volume_cylinders)

theorem determine_mnp :
  let m := 228
  let n := 31
  let p := 3
  m + n + p = 262 :=
by
  let a := 2
  let b := 3
  let c := 4
  let r := 1
  have volume_set := volume_of_set a b c r
  have : volume_set = (m + n * Real.pi) / p := by sorry
  have gcd_n_p : Nat.gcd n p = 1 := by sorry
  rw [this]
  norm_num

end determine_mnp_l311_311983


namespace sum_of_divisors_of_29_l311_311886

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311886


namespace end_process_when_all_cells_one_l311_311928

theorem end_process_when_all_cells_one {n : ℕ} (h : n ≥ 3) : 
  (∃ N : ℕ, N = n ∧ ∀ (x : Fin n → ℕ), 
    ((∃ i : Fin n, x i = 0) ∧ (∀ j, x j = if j = i then 0 else 1)) →
    (∃ m : ℕ, m ≥ 1 ∧ (∃ (process_step : Fin n → ℕ → ℕ), 
      ∀ (k : ℕ) (y : Fin n → ℕ), process_step y k = x ∧ y 0 = 1 ∧
      (∀ i, x i = process_step (λ (z : ℕ), if z = i then 1 - y (z + 1) mod n else y (z mod n)) k))
  ) ↔ n % 3 ≠ 0 :=
begin
  sorry
end

end end_process_when_all_cells_one_l311_311928


namespace ab_in_sequence_l311_311987

-- Definition of the sequence v_n
def v : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := 4 * v (n+1) - v n

-- Define conditions
variables (a b : ℕ) 
@[simp] lemma odd_a : a % 2 = 1 := by sorry
@[simp] lemma odd_b : b % 2 = 1 := by sorry
@[simp] lemma cond1 : ∃ k, b^2 + 2 = k * a := by sorry 
@[simp] lemma cond2 : ∃ l, a^2 + 2 = l * b := by sorry 

-- The theorem to prove
theorem ab_in_sequence : ∃ n m, v n = a ∧ v m = b :=
by sorry

end ab_in_sequence_l311_311987


namespace parallelogram_area_proof_l311_311628

noncomputable def parallelogram_area : ℝ :=
  let angle_rad := (150 * real.pi / 180)  -- converting degrees to radians
  let a := 10                              -- length of one side
  let b := 20                              -- length of another side
  let height := a * real.sqrt(3) / 2       -- height from 30-60-90 triangle properties
  b * height

theorem parallelogram_area_proof : parallelogram_area = 100 * real.sqrt(3) := by
  sorry

end parallelogram_area_proof_l311_311628


namespace tiling_scheme_3_3_3_3_6_l311_311894

-- Definitions based on the conditions.
def angle_equilateral_triangle := 60
def angle_regular_hexagon := 120

-- The theorem states that using four equilateral triangles and one hexagon around a point forms a valid tiling.
theorem tiling_scheme_3_3_3_3_6 : 
  4 * angle_equilateral_triangle + angle_regular_hexagon = 360 := 
by
  -- Skip the proof with sorry
  sorry

end tiling_scheme_3_3_3_3_6_l311_311894


namespace numbers_unchanged_by_powers_of_n_l311_311316

-- Definitions and conditions
def unchanged_when_raised (x : ℂ) (n : ℕ) : Prop :=
  x^n = x

def modulus_one (z : ℂ) : Prop :=
  Complex.abs z = 1

-- Proof statements
theorem numbers_unchanged_by_powers_of_n :
  (∀ x : ℂ, (∀ n : ℕ, n > 0 → unchanged_when_raised x n → x = 0 ∨ x = 1)) ∧
  (∀ z : ℂ, modulus_one z → (∀ n : ℕ, n > 0 → Complex.abs (z^n) = 1)) :=
by
  sorry

end numbers_unchanged_by_powers_of_n_l311_311316


namespace excellent_sequences_exceed_suboptimal_by_product_l311_311450

variable {α : Type*} [LinearOrder α] [AddCommMonoid α]

theorem excellent_sequences_exceed_suboptimal_by_product
  (a : Fin 11 → ℕ)
  (sum_lt_2007 : (Finset.univ.sum (λ i, a i)) < 2007)
  (distinct : Function.Injective a) :
  let excellent_sequences := 1 -- Placeholder for actual definition
  let suboptimal_sequences := 1 -- Placeholder for actual definition
  excellent_sequences - suboptimal_sequences = (Finset.univ.prod (λ i, a i)) :=
sorry

end excellent_sequences_exceed_suboptimal_by_product_l311_311450


namespace sum_of_divisors_of_29_l311_311721

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311721


namespace sum_of_divisors_of_29_l311_311774

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311774


namespace sum_of_divisors_29_l311_311864

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311864


namespace sum_of_divisors_29_l311_311869

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311869


namespace correct_option_is_A_l311_311548

def second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (0, 4)
def point_D : ℝ × ℝ := (5, -6)

theorem correct_option_is_A :
  (second_quadrant point_A) ∧
  ¬(second_quadrant point_B) ∧
  ¬(second_quadrant point_C) ∧
  ¬(second_quadrant point_D) :=
by sorry

end correct_option_is_A_l311_311548


namespace correct_statement_l311_311896

-- Define statements as propositions
def statement_A : Prop := 
  let coefficient := -3 / 4
  let degree := 3
  (coefficient = (-3)) ∧ (degree = 2)

def statement_B : Prop := 
  let term1 := 5 * x^2 * y
  let term2 := -2 * y * x^2
  ¬ (term1 = term2)

def statement_C : Prop :=
  let expr := x^2 + m * x
  let is_monomial := m = 0
  is_monomial

def statement_D : Prop :=
  let zero_monomial := 0
  ¬ (is_monomial zero_monomial)

-- Main theorem to be proven
theorem correct_statement : 
  ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D := by
  sorry

end correct_statement_l311_311896


namespace sum_of_divisors_of_29_l311_311850

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311850


namespace smaller_triangles_perimeter_l311_311562

theorem smaller_triangles_perimeter
  (P_large : ℝ)
  (P_small : ℝ)
  (n : ℕ)
  (hP_large : P_large = 120)
  (hn : n = 9)
  (equal_perimeters : ∀ i j : ℕ, i < n → j < n → i ≠ j → P_small = P_small) :
  9 * P_small = P_large → P_small = 40 :=
by
  intros h
  rw [← hP_large] at h
  rw [← hn] at h
  calc
    9 * P_small = 120 : h
    P_small = 120 / 9 : by rw [mul_comm, mul_div_cancel_left _ (by norm_num : (9:ℝ) ≠ 0)]
    ... = 40 : rfl

end smaller_triangles_perimeter_l311_311562


namespace part_a_part_b_l311_311140

variables (A B C D E M F : Type) 
[h1 : parallelogram A B C D]
[h2 : ∃ E, angle_bisector_inter (ADC) E (BC)]
[h3 : ∃ M, perp_bisector_inter (AD) M (DE)]
[h4 : ∃ F, inter (AM) F (BC)]

-- a) Show that DE = AF
theorem part_a : DE = AF := 
sorry

-- b) Show that AB × AD = DE × DM
theorem part_b : AB * AD = DE * DM :=
sorry

end part_a_part_b_l311_311140


namespace value_of_a_minus_b_l311_311337

def round_to_hundredths (x : ℝ) : ℝ := (Real.floor (x * 100) + 1) / 100

theorem value_of_a_minus_b :
  let x := 13.165
  let y := 7.686
  let z := 11.545
  let a := round_to_hundredths x + round_to_hundredths y + round_to_hundredths z
  let b := round_to_hundredths (x + y + z)
  a - b = 0.01 :=
by
  -- Definitions
  let x := 13.165
  let y := 7.686
  let z := 11.545
  -- Calculation of a and b
  let a := round_to_hundredths x + round_to_hundredths y + round_to_hundredths z
  let b := round_to_hundredths (x + y + z)
  -- We are not solving it, so we just state the expected result
  have h : a - b = 0.01 := sorry
  -- Providing the proof as a return value
  show a - b = 0.01 from h

end value_of_a_minus_b_l311_311337


namespace parallelogram_area_proof_l311_311626

noncomputable def parallelogram_area : ℝ :=
  let angle_rad := (150 * real.pi / 180)  -- converting degrees to radians
  let a := 10                              -- length of one side
  let b := 20                              -- length of another side
  let height := a * real.sqrt(3) / 2       -- height from 30-60-90 triangle properties
  b * height

theorem parallelogram_area_proof : parallelogram_area = 100 * real.sqrt(3) := by
  sorry

end parallelogram_area_proof_l311_311626


namespace sum_of_divisors_of_prime_29_l311_311835

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311835


namespace distinguishable_large_triangles_l311_311301

-- Definitions of the problem's conditions
def num_colors : Nat := 8
def num_small_triangles : Nat := 6

-- The proof that the number of distinguishable large equilateral triangles is 116096
theorem distinguishable_large_triangles : num_distinguishable_large_triangles num_colors num_small_triangles = 116096 := 
sorry

end distinguishable_large_triangles_l311_311301


namespace probability_A_and_B_same_group_l311_311467

variable (A B C D : Prop)

theorem probability_A_and_B_same_group 
  (h : ∃ (S : Set (Prop × Prop)), {A, B, C, D}.card = 4 ∧ {({A, B}, {C, D}), ({A, D}, {B, C}), ({A, C}, {B, D})}.card = 3) :
  (1 / 3) :=
sorry

end probability_A_and_B_same_group_l311_311467


namespace sum_of_divisors_of_29_l311_311889

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311889


namespace area_ratio_of_triangle_quadrilateral_l311_311559

section geometric_problem

variables (A B C P Q : Type*)
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space P]
variable [metric_space Q]
variables (AB AC BC AP AQ : ℝ)
variables (h1 : AB = 30)
variables (h2 : AC = 50)
variables (h3 : BC = 45)
variables (h4 : AP = 10)
variables (h5 : AQ = 20)

theorem area_ratio_of_triangle_quadrilateral :
  let area_APQ := (real.ratios.get_area_of_triangle AP AQ) in
  let area_ABC := (real.ratios.get_area_of_triangle AB AC) in
  area_APQ / (area_ABC - area_APQ) = 1 / 8 :=
sorry

end geometric_problem

end area_ratio_of_triangle_quadrilateral_l311_311559


namespace intersection_M_N_l311_311066

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x : ℤ | 1/2 < 2^(x + 1) ∧ 2^(x + 1) < 4}

theorem intersection_M_N : M ∩ N = {-1} := by
  sorry

end intersection_M_N_l311_311066


namespace probability_multiple_of_3_l311_311334

noncomputable def count_multiples_of_3 (n : ℕ) : ℕ :=
(n / 3 : ℕ)

theorem probability_multiple_of_3 (n : ℕ) (h : n = 27) : 
  (count_multiples_of_3 n : ℚ) / n = 1 / 3 :=
by
  rw [h, count_multiples_of_3]
  norm_num
  rw div_eq_mul_inv
  norm_num
  sorry

end probability_multiple_of_3_l311_311334


namespace balls_in_boxes_l311_311082

theorem balls_in_boxes : 
  (∃ f : Fin 7 → Fin 7, (∃ (I : Finset (Fin 7)), I.card = 3 ∧ ∀ i ∈ I, f i = i) ∧ 
  (∃ (J : Finset (Fin 7)), J.card = 4 ∧ J ∩ I = ∅ ∧ 
  ∀ j ∈ J, f j ≠ j)) →
  (∃ n : ℕ, n = 315) :=
begin
  sorry -- proof goes here
end

end balls_in_boxes_l311_311082


namespace infinite_solutions_iff_c_is_5_over_2_l311_311997

theorem infinite_solutions_iff_c_is_5_over_2 (c : ℝ) :
  (∀ y : ℝ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 :=
by 
  sorry

end infinite_solutions_iff_c_is_5_over_2_l311_311997


namespace exists_k_for_circle_through_E_l311_311135

theorem exists_k_for_circle_through_E :
  ∃ k : ℝ, k ≠ 0 ∧
  let ellipse_eq := ∀ (x y : ℝ), x^2 / 3 + y^2 = 1
  let line_eq := ∀ (x y : ℝ), y = k * x + 2
  ∃ (C D : ℝ × ℝ), ellipse_eq C.1 C.2 ∧ ellipse_eq D.1 D.2 ∧
  line_eq C.1 C.2 ∧ line_eq D.1 D.2 ∧
  let E := (-1, 0)
  let diameter_C_D := dist C D
  let circle_through_E := ∃ (O R : ℝ × ℝ), dist O E = R ∧ dist O C = R ∧ dist O D = R
  circle_through_E
  :=
begin
  use 7 / 6,
  -- Further proof steps skipped with sorry
  sorry
end

end exists_k_for_circle_through_E_l311_311135


namespace probability_of_snow_at_most_three_days_in_December_l311_311287

noncomputable def prob_snow_at_most_three_days (n : ℕ) (p : ℚ) : ℚ :=
  (Finset.range (n + 1)).sum (λ k => (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)))

theorem probability_of_snow_at_most_three_days_in_December :
  prob_snow_at_most_three_days 31 (1/5:ℚ) ≈ 0.230 :=
sorry

end probability_of_snow_at_most_three_days_in_December_l311_311287


namespace sum_of_divisors_29_l311_311871

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311871


namespace number_of_odd_five_digit_numbers_l311_311391

/-- The number of odd five-digit numbers with no repeated digits 
formed from the set {1, 2, 3, 4, 5} is 72. -/
theorem number_of_odd_five_digit_numbers : 
  (finset.univ.image (λ (x : fin 5), x + 1)).filter (λ n, n % 2 = 1).card * nat.factorial 4 = 72 := 
by
  sorry

end number_of_odd_five_digit_numbers_l311_311391


namespace marks_difference_is_140_l311_311687

noncomputable def marks_difference (P C M : ℕ) : ℕ :=
  (P + C + M) - P

theorem marks_difference_is_140 (P C M : ℕ) (h1 : (C + M) / 2 = 70) :
  marks_difference P C M = 140 := by
  sorry

end marks_difference_is_140_l311_311687


namespace simplify_and_rationalize_proof_l311_311649

noncomputable def simplify_and_rationalize (a b c d e f g : ℝ) : ℝ :=
  (a * b / c * d / e) / f

theorem simplify_and_rationalize_proof :
  simplify_and_rationalize (Real.sqrt 8) (Real.sqrt 5) (Real.sqrt 10 * Real.sqrt 12) (1 : ℝ) (Real.sqrt 9) (Real.sqrt 14) = (Real.sqrt 28) / 3 :=
by
  sorry

end simplify_and_rationalize_proof_l311_311649


namespace parallelogram_area_l311_311636

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l311_311636


namespace find_circumcenter_l311_311428

-- Define a quadrilateral with vertices A, B, C, and D
structure Quadrilateral :=
  (A B C D : (ℝ × ℝ))

-- Define the coordinates of the circumcenter
def circumcenter (q : Quadrilateral) : ℝ × ℝ := (6, 1)

-- Given condition that A, B, C, and D are vertices of a quadrilateral
-- Prove that the circumcenter of the circumscribed circle is (6, 1)
theorem find_circumcenter (q : Quadrilateral) : 
  circumcenter q = (6, 1) :=
by sorry

end find_circumcenter_l311_311428


namespace sum_of_divisors_of_29_l311_311768

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311768


namespace sector_angle_l311_311050

theorem sector_angle (P A : ℝ) (hP : P = 10) (hA : A = 4) : 
  ∃ α : ℝ, α = 1 / 2 :=
by
  let l r : ℝ
  have h1 : 2 * r + l = P := sorry
  have h2 : 1/2 * l * r = A := sorry
  use 1 / 2
  sorry  

end sector_angle_l311_311050


namespace remainder_is_287_l311_311143

def is_increasing_sequence_of_7_ones (n: ℕ) : Prop :=
  (nat.popcount n = 7)

def increasing_sequence_of_7_ones : ℕ → ℕ
| 0     := 0
| (n+1) := if is_increasing_sequence_of_7_ones (increasing_sequence_of_7_ones n + 1) 
           then increasing_sequence_of_7_ones n + 1
           else increasing_sequence_of_7_ones n + 2

def T (n: ℕ) : ℕ := increasing_sequence_of_7_ones (n + 1) 

def M := T 699

theorem remainder_is_287: M % 500 = 287 := by
  sorry

end remainder_is_287_l311_311143


namespace sum_of_divisors_of_29_l311_311811

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311811


namespace profit_calculation_l311_311371

def actors_cost : ℕ := 1200
def people_count : ℕ := 50
def cost_per_person : ℕ := 3
def food_cost : ℕ := people_count * cost_per_person
def total_cost_actors_food : ℕ := actors_cost + food_cost
def equipment_rental_cost : ℕ := 2 * total_cost_actors_food
def total_movie_cost : ℕ := total_cost_actors_food + equipment_rental_cost
def movie_sale_price : ℕ := 10000
def profit : ℕ := movie_sale_price - total_movie_cost

theorem profit_calculation : profit = 5950 := by
  sorry

end profit_calculation_l311_311371


namespace identify_same_function_as_y_equals_x_l311_311392

-- Define the functions
def f1 (x : ℝ) := (sqrt x) ^ 2
def f2 (x : ℝ) := (cbrt (x ^ 3))
def f3 (x : ℝ) := sqrt (x ^ 2)
def f4 (x : ℝ) := if x ≠ 0 then x^2 / x else 0
def f (x : ℝ) := x

-- Prove that f2 is identical to f
theorem identify_same_function_as_y_equals_x : (∀ x : ℝ, f2 x = f x) ∧ (∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, abs (x - y) < ε → f2 y = f y) ∧ range f2 = range f :=
by
  sorry  

end identify_same_function_as_y_equals_x_l311_311392


namespace line_through_intersection_parallel_l311_311430

theorem line_through_intersection_parallel
  (l1 l2 l3 : ℝ → ℝ → Prop)
  (h1 : l1 = λ x y, 2 * x + 3 * y - 5 = 0)
  (h2 : l2 = λ x y, 3 * x - 2 * y - 3 = 0)
  (h_parallel : l3 = λ x y, 2 * x + y - 3 = 0) :
  ∃ c : ℝ, ∀ x y : ℝ, (2 * x + y + c = 0) ↔ (26 * x + 13 * y - 47 = 0) :=
sorry

end line_through_intersection_parallel_l311_311430


namespace log_ab_gt_one_case_l311_311023

variable (a b : ℝ)

theorem log_ab_gt_one_case (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ 1) (h4 : b ≠ 1) (h5 : log a b > 1) :
  (b - 1) * (b - a) > 0 := 
sorry

end log_ab_gt_one_case_l311_311023


namespace coefficients_sum_binomial_coefficients_sum_l311_311552

theorem coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = coeff_sum) : coeff_sum = 729 := 
sorry

theorem binomial_coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = binom_coeff_sum) : binom_coeff_sum = 64 := 
sorry

end coefficients_sum_binomial_coefficients_sum_l311_311552


namespace sum_of_divisors_of_prime_l311_311787

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311787


namespace find_y_solution_l311_311001

noncomputable def func1 (y : ℝ) := 1 / (y * (y + 2))
noncomputable def func2 (y : ℝ) := 1 / ((y + 2) * (y + 4))

theorem find_y_solution (y : ℝ) : 
  (func1 y - func2 y < 1 / 4) ↔ (y ∈ Set.Iio (-4) ∪ Set.Ioo (-2, 0) ∪ Set.Ioi 2) :=
by 
  sorry

end find_y_solution_l311_311001


namespace rate_of_current_l311_311379

-- Definitions of the conditions
def downstream_speed : ℝ := 30  -- in kmph
def upstream_speed : ℝ := 10    -- in kmph
def still_water_rate : ℝ := 20  -- in kmph

-- Calculating the rate of the current
def current_rate : ℝ := downstream_speed - still_water_rate

-- Proof statement
theorem rate_of_current :
  current_rate = 10 :=
by
  sorry

end rate_of_current_l311_311379


namespace simplify_cuberoot_product_l311_311222

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311222


namespace focal_length_of_ellipse_l311_311273

noncomputable def focal_length_of_curve_parametric (x y : ℝ → ℝ) : ℝ := 2 * real.sqrt (5^2 - 4^2)

theorem focal_length_of_ellipse :
  focal_length_of_curve_parametric (λ θ, 5 * real.cos θ) (λ θ, 4 * real.sin θ) = 6 :=
by
  calc
    focal_length_of_curve_parametric (λ θ, 5 * real.cos θ) (λ θ, 4 * real.sin θ)
    = 2 * real.sqrt (5^2 - 4^2) : rfl
    ... = 2 * real.sqrt (25 - 16) : by simp
    ... = 2 * real.sqrt 9 : by simp
    ... = 2 * 3 : by simp
    ... = 6 : by simp

end focal_length_of_ellipse_l311_311273


namespace sum_of_divisors_prime_29_l311_311731

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311731


namespace non_raining_hours_l311_311444

-- Definitions based on the conditions.
def total_hours := 9
def rained_hours := 4

-- Problem statement: Prove that the non-raining hours equals to 5 given total_hours and rained_hours.
theorem non_raining_hours : (total_hours - rained_hours = 5) :=
by
  -- The proof is omitted with "sorry" to indicate the missing proof.
  sorry

end non_raining_hours_l311_311444


namespace hospital_doctors_selection_l311_311376

open Finset

def choose (n k : ℕ) := (range n).powerset.filter (λ s, s.card = k)

theorem hospital_doctors_selection :
  let num_internal = 5
      num_surgeon = 6
      total_doctors = num_internal + num_surgeon
      selecting_doctors = 4 in
  choose total_doctors selecting_doctors.card
  - (choose num_internal selecting_doctors.card
   + choose num_surgeon selecting_doctors.card) = 310 :=
by
  sorry

end hospital_doctors_selection_l311_311376


namespace sum_of_divisors_of_prime_29_l311_311847

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311847


namespace surface_area_of_hemisphere_l311_311297

noncomputable theory

open Real

-- Define the conditions and the question
def hemisphere_volume (r : ℝ) : Prop := (2 / 3) * pi * (r ^ 3) = (500 / 3) * pi

def surface_area_hemisphere (r : ℝ) : ℝ := 3 * pi * (250 ^ (2 / 3))

-- State the theorem
theorem surface_area_of_hemisphere (r : ℝ) (h : hemisphere_volume r) :
  surface_area_hemisphere r = 3 * pi * (250 ^ (2 / 3)) := 
sorry

end surface_area_of_hemisphere_l311_311297


namespace simplify_cubed_roots_l311_311218

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311218


namespace find_m_n_l311_311438

open Nat

theorem find_m_n : ∃ m n : ℕ, m < n ∧ n < 2 * m ∧ m * n = 2013 ∧ m = 33 ∧ n = 61 := by
  use 33, 61
  simp
  sorry

end find_m_n_l311_311438


namespace CarmenBrushLengthIsCorrect_l311_311978

namespace BrushLength

def carlasBrushLengthInInches : ℤ := 12
def conversionRateInCmPerInch : ℝ := 2.5
def lengthMultiplier : ℝ := 1.5

def carmensBrushLengthInCm : ℝ :=
  carlasBrushLengthInInches * lengthMultiplier * conversionRateInCmPerInch

theorem CarmenBrushLengthIsCorrect :
  carmensBrushLengthInCm = 45 := by
  sorry

end BrushLength

end CarmenBrushLengthIsCorrect_l311_311978


namespace bottle_count_l311_311893

theorem bottle_count :
  ∃ N x : ℕ, 
    N = x^2 + 36 ∧ N = (x + 1)^2 + 3 :=
by 
  sorry

end bottle_count_l311_311893


namespace n_plus_2_and_2n_plus_2_perfect_squares_l311_311383

def is_nice (n : ℕ) : Prop :=
  ∑ d in (Finset.divisors n), d^2 = (n + 3)^2

theorem n_plus_2_and_2n_plus_2_perfect_squares
  (n p q : ℕ) (hp : p.prime) (hq : q.prime) (h : n = p * q) (hnice : is_nice n) :
  ∃ m k : ℕ, n + 2 = m^2 ∧ 2 * (n + 1) = k^2 :=
by
  sorry

end n_plus_2_and_2n_plus_2_perfect_squares_l311_311383


namespace function_even_iff_perpendicular_l311_311584

variables {α : Type*} [Field α]
variables (a b : Vector α)

def is_even_function (f : α → α) := ∀ x : α, f x = f (-x)
def perpendicular (a b : Vector α) := dot_product a b = 0

theorem function_even_iff_perpendicular (a b : Vector α) (h_a_nonzero : a ≠ 0) (h_b_nonzero : b ≠ 0) :
  is_even_function (λ x, (a * x + b)^2) ↔ perpendicular a b := sorry

end function_even_iff_perpendicular_l311_311584


namespace oranges_per_tree_A_l311_311645

-- Definitions
def num_trees : ℕ := 10
def percent_tree_A : ℝ := 0.50
def percent_tree_B : ℝ := 0.50
def oranges_tree_B : ℕ := 15
def good_orange_rate_A : ℝ := 0.60
def good_orange_rate_B : ℝ := 1/3
def total_good_oranges : ℕ := 55

-- Main theorem
theorem oranges_per_tree_A (X : ℝ) :
  let num_tree_A := percent_tree_A * num_trees in
  let num_tree_B := percent_tree_B * num_trees in
  let good_oranges_A := num_tree_A * good_orange_rate_A * X in
  let good_oranges_B := num_tree_B * good_orange_rate_B * oranges_tree_B in
  (good_oranges_A + good_oranges_B = total_good_oranges) → 
  X = 10 :=
by 
suffices h : 5 * 0.60 * X + 5 * 5 = 55
  sorry
  sorry

end oranges_per_tree_A_l311_311645


namespace scientific_notation_of_35000000_l311_311956

theorem scientific_notation_of_35000000 :
  (35_000_000 : ℕ) = 3.5 * 10^7 := by
  sorry

end scientific_notation_of_35000000_l311_311956


namespace quadratic_inequality_l311_311499

open Function

def f (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem quadratic_inequality :
  f(2) < f(3) ∧ f(3) < f(0) :=
by
  -- defining the function
  let f := λ (x : ℝ), (x - 2) ^ 2 + 1
  -- calculate the necessary values
  have h1 : f(2) = 1 := by simp [f]
  have h2 : f(3) = 2 := by simp [f]
  have h3 : f(0) = 5 := by simp [f]
  -- proving the inequalities
  exact ⟨by linarith, by linarith⟩

end quadratic_inequality_l311_311499


namespace smallest_positive_solution_exists_l311_311010

open Real

-- Define the original equation
def equation (x : ℝ) : Prop :=
  tan 4 * x + tan 3 * x = sec 3 * x

-- Define the expected smallest positive solution
def expected_solution : ℝ := π / 26

-- State the main theorem
theorem smallest_positive_solution_exists :
  ∃ x > 0, equation x ∧ x = expected_solution :=
sorry

end smallest_positive_solution_exists_l311_311010


namespace smallest_square_area_l311_311922

theorem smallest_square_area (a b c d : ℕ) (ha : a = 2) (hb : b = 4) (hc : c = 3) (hd : d = 5) :
  ∃ s : ℕ, s * s = 81 :=
by
  use 9
  norm_num
  sorry

end smallest_square_area_l311_311922


namespace find_m_l311_311064

theorem find_m {m : ℕ} (h1 : Even (m^2 - 2 * m - 3)) (h2 : m^2 - 2 * m - 3 < 0) : m = 1 :=
sorry

end find_m_l311_311064


namespace find_number_l311_311381

theorem find_number (x : ℕ) (h : x * 99999 = 65818408915) : x = 658185 :=
sorry

end find_number_l311_311381


namespace problem_equivalent_l311_311587

theorem problem_equivalent :
  ∃ n : ℕ, (n > 0) ∧ (30 ∣ n) ∧ (∃ k : ℕ, n^2 = k^3) ∧ (∃ m : ℕ, n^3 = m^2) ∧ (nat.digits 10 n − 1 + 1 = 9) :=
sorry

end problem_equivalent_l311_311587


namespace sphere_diameter_l311_311944

theorem sphere_diameter 
  (shadow_sphere : ℝ)
  (height_pole : ℝ)
  (shadow_pole : ℝ)
  (parallel_rays : Prop)
  (vertical_objects : Prop)
  (tan_theta : ℝ) :
  shadow_sphere = 12 →
  height_pole = 1.5 →
  shadow_pole = 3 →
  (tan_theta = height_pole / shadow_pole) →
  parallel_rays →
  vertical_objects →
  2 * (shadow_sphere * tan_theta) = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sphere_diameter_l311_311944


namespace function_not_monotonic_l311_311469

theorem function_not_monotonic (m : ℝ) :
  (∀ x : ℝ, f(x) = (m-1) * x^2 - 2 * m * x + 3) →
  (∀ x : ℝ, f(x) = f(-x)) →
  ¬(∀ x y : ℝ, x < y → x < 3 → y < 3 → f(x) ≤ f(y) ∨ f(x) ≥ f(y)) :=
by
  intros h1 h2
  sorry

end function_not_monotonic_l311_311469


namespace prob1_prob2_prob3_l311_311404

-- Problem 1
theorem prob1 (a b c : ℝ) : ((-8 * a^4 * b^5 * c / (4 * a * b^5)) * (3 * a^3 * b^2)) = -6 * a^6 * b^2 :=
by
  sorry

-- Problem 2
theorem prob2 (a : ℝ) : (2 * a + 1)^2 - (2 * a + 1) * (2 * a - 1) = 4 * a + 2 :=
by
  sorry

-- Problem 3
theorem prob3 (x y : ℝ) : (x - y - 2) * (x - y + 2) - (x + 2 * y) * (x - 3 * y) = 7 * y^2 - x * y - 4 :=
by
  sorry

end prob1_prob2_prob3_l311_311404


namespace quadratic_equation_factored_form_l311_311957

theorem quadratic_equation_factored_form : 
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x - 3)^2 = 15 := 
by 
  sorry

end quadratic_equation_factored_form_l311_311957


namespace calculation_101_squared_minus_99_squared_l311_311973

theorem calculation_101_squared_minus_99_squared : 101^2 - 99^2 = 400 :=
by
  sorry

end calculation_101_squared_minus_99_squared_l311_311973


namespace log_neg_inequality_l311_311096

theorem log_neg_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  Real.log (-a) > Real.log (-b) := 
sorry

end log_neg_inequality_l311_311096


namespace sum_of_divisors_of_29_l311_311772

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311772


namespace simplify_sqrt_l311_311192

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311192


namespace number_of_cows_l311_311906

theorem number_of_cows (D C : ℕ) (h1 : 2 * D + 4 * C = 40 + 2 * (D + C)) : C = 20 :=
by
  sorry

end number_of_cows_l311_311906


namespace isosceles_triangle_l311_311103

theorem isosceles_triangle (a b c : ℝ) (h : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) : 
  a = b ∨ b = c ∨ c = a :=
sorry

end isosceles_triangle_l311_311103


namespace ms_tom_investment_l311_311329

def invested_amounts (X Y : ℝ) : Prop :=
  X + Y = 100000 ∧ 0.17 * Y = 0.23 * X + 200 

theorem ms_tom_investment (X Y : ℝ) (h : invested_amounts X Y) : X = 42000 :=
by
  sorry

end ms_tom_investment_l311_311329


namespace parallel_line_distance_l311_311495

-- Definition of a line
structure Line where
  m : ℚ -- slope
  c : ℚ -- y-intercept

-- Given conditions
def given_line : Line :=
  { m := 3 / 4, c := 6 }

-- Prove that there exist lines parallel to the given line and 5 units away from it
theorem parallel_line_distance (L : Line)
  (h_parallel : L.m = given_line.m)
  (h_distance : abs (L.c - given_line.c) = 25 / 4) :
  (L.c = 12.25) ∨ (L.c = -0.25) :=
sorry

end parallel_line_distance_l311_311495


namespace maximum_value_of_f_is_sqrt2_l311_311527

def f (x : ℝ) (a : ℝ) : ℝ := sin (x / 2) + a * cos (x / 2)

theorem maximum_value_of_f_is_sqrt2 
  (a : ℝ) 
  (h : ∀ x, f x a = sin (x / 2) + cos (x / 2))
  (symmetry_condition : ∀ x, f (3 * π - x) a = -f x a) : 
  ∀ x, f x 1 ≤ sqrt 2 :=
by
  sorry

end maximum_value_of_f_is_sqrt2_l311_311527


namespace necessary_and_sufficient_condition_l311_311994

theorem necessary_and_sufficient_condition
  (a b : ℝ)
  (h1 : |a| > |b|)
  (h2 : 1/a > 1/b)
  (h3 : a^2 > b^2)
  (h4 : 2^a > 2^b) :
  a > b ↔ 2^a > 2^b :=
by
  sorry

end necessary_and_sufficient_condition_l311_311994


namespace sum_a_eq_22_l311_311139

theorem sum_a_eq_22 :
  ∃ (n : ℕ) (a : Fin n → ℕ) (c : Fin n → ℤ),
    2005 = ∑ i, c i * 3^(a i) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i, c i = 1 ∨ c i = -1) ∧
    (∑ i, a i = 22) :=
begin
  sorry

end sum_a_eq_22_l311_311139


namespace area_triangle_l311_311560

theorem area_triangle $ABC$ (A B C D : Type)
  [AC_hypotenuse : AC]
  [AD_midpoint_AC : AD = DC = 5]
  [right_triangle_C : right_triangle ABC C]
  : area_triangle ABC = 25 / 2 :=
sorry

end area_triangle_l311_311560


namespace sum_of_divisors_of_prime_29_l311_311846

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311846


namespace problem_l311_311498

open Classical

variable (A B : Set ℝ)

def p : Prop := ∃ (x : ℝ), (1 / 10 : ℝ)^x ≤ 0

theorem problem :
  ¬p ∧ (thmD ∨ thmA ∨ thmB ∨ thmC) = False → thmD = False := by
begin
  have np : ¬p := sorry,
  apply eq_false_eq,
  intros,
  split,
  { exact np },
  { apply or.inr,
    apply or.inr,
    apply or.inr,
    exact thmD_false },
end

noncomputable def p_is_false : ¬p := by
begin
  unfold p,
  apply not_exists.intro,
  intros,
  simp,
  exact pow_nonneg_of_le_zero (1 / 10) h
end

def thmA : Prop := ∀ x ∈ Ico 1 3, -2 * x^2 + x < -2 * x^2 + x
def thmB : Prop := (log 3 : ℝ) > 1
def thmC : Prop := A ∩ B = A → B ⊆ A
def thmD : Prop := (log 2 + log 3 = log 5 : ℝ)
def thmD_false : thmD → False
:=  by interpret_log_equality

#check problem

end problem_l311_311498


namespace sum_of_divisors_of_29_l311_311812

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311812


namespace area_of_parallelogram_l311_311611

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l311_311611


namespace range_of_a_l311_311047

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + 1 / 4 > 0) ↔ (Real.sqrt 5 - 3) / 2 < a ∧ a < (3 + Real.sqrt 5) / 2 :=
by
  sorry

end range_of_a_l311_311047


namespace proof_problem_l311_311415

variable {R : Type} [LinearOrderedField R]

def even_function (f : R → R) := ∀ x : R, f x = f (-x)

noncomputable def given_conditions (f : R → R) : Prop :=
  (∀ x ∈ Icc (1 : R) 2, f x < 0) ∧
  (∀ ⦃x y⦄, x ∈ Icc (1 : R) 2 → y ∈ Icc (1 : R) 2 → x < y → f x < f y)

theorem proof_problem (f : R → R) (h_even : even_function f) (h_given : given_conditions f) :
  (∀ x ∈ Icc (-2) (-1), f x < 0) ∧ (∀ ⦃x y⦄, x ∈ Icc (-2) (-1) → y ∈ Icc (-2) (-1) → x < y → f y < f x) :=
sorry

end proof_problem_l311_311415


namespace third_day_swimmers_proof_l311_311384

noncomputable def third_day_swimmers (total_people : ℕ) (first_day_people : ℕ) (difference_second_third : ℕ) (third_day_people : ℕ) : Prop :=
    total_people = first_day_people + (third_day_people + difference_second_third) + third_day_people

theorem third_day_swimmers_proof : third_day_swimmers 246 79 47 60 :=
begin
    -- Total number of people over the first 3 days should sum up to 246
    -- 79 people came on the first day
    -- second day had (x + 47) people where x is the number of people on the third day
    -- third day people is x = 60
    sorry
end

end third_day_swimmers_proof_l311_311384


namespace possible_k_values_l311_311911

theorem possible_k_values :
  (∃ k b a c : ℤ, b = 2020 + k ∧ a * (c ^ 2) = (2020 + k) ∧ 
  (k = -404 ∨ k = -1010)) :=
sorry

end possible_k_values_l311_311911


namespace quadrilateral_square_necessity_l311_311346

-- Definitions according to the given conditions
def interior_angles_equal (q : Quadrilateral) : Prop :=
  q.angle1 = q.angle2 ∧ q.angle2 = q.angle3 ∧ q.angle3 = q.angle4

def is_square (q : Quadrilateral) : Prop :=
  -- Additional properties of a square can go here
  sorry 

-- The main statement to prove
theorem quadrilateral_square_necessity (q : Quadrilateral) :
  interior_angles_equal q → (is_square q → interior_angles_equal q) :=
by
  sorry

end quadrilateral_square_necessity_l311_311346


namespace sum_of_divisors_of_29_l311_311813

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311813


namespace unique_natural_in_sequences_l311_311697

def seq_x (n : ℕ) : ℤ := if n = 0 then 10 else if n = 1 then 10 else seq_x (n - 2) * (seq_x (n - 1) + 1) + 1
def seq_y (n : ℕ) : ℤ := if n = 0 then -10 else if n = 1 then -10 else (seq_y (n - 1) + 1) * seq_y (n - 2) + 1

theorem unique_natural_in_sequences (k : ℕ) (i j : ℕ) :
  seq_x i = k → seq_y j ≠ k :=
by
  sorry

end unique_natural_in_sequences_l311_311697


namespace find_m_perpendicular_lines_l311_311160

theorem find_m_perpendicular_lines 
  (m : ℝ)
  (l1 : ∀ x y : ℝ, mx - y + 1 = 0)
  (l2 : ∀ x y : ℝ, (3*m - 2)*x + m*y - 2 = 0)
  (h_perpendicular : m * (-((3*m - 2) / m)) = -1) :
  m = 0 ∨ m = 1 := 
begin
  sorry,
end

end find_m_perpendicular_lines_l311_311160


namespace steve_dimes_count_l311_311653

theorem steve_dimes_count (D N : ℕ) (h1: D + N = 36) (h2: (0.10 * D) + (0.05 * N) = 3.10) : D = 26 :=
by
  sorry

end steve_dimes_count_l311_311653


namespace max_repeated_digits_square_l311_311308

theorem max_repeated_digits_square (n : ℕ) (h_nonzero_unit : ∃ a b : ℕ, n = 10 * a + b ∧ b ≠ 0) :
  ∃ k : ℕ, k ≤ 4 ∧ ∀ m : ℕ, (m = n^2) → ∃ d : ℕ, k = (nat_digits d m).repeat_suffix_count :=
sorry

end max_repeated_digits_square_l311_311308


namespace minimum_distance_to_line_l311_311521

-- Define the function and line 
def f (x : ℝ) := x^2 - Real.log x
def line (P : ℝ × ℝ) := P.fst - P.snd - 2 = 0

-- Define the problem statement
theorem minimum_distance_to_line :
  ∀ P : ℝ × ℝ, P ∈ (f '' (set.Ioi 0)) → line P →
  ∃ x y, x = 1 ∧ y = 1 ∧ P = (x, y) ∧ 
  dist P (1, -1, 2) = Real.sqrt 2 :=
by
  intro P hp hl
  sorry

end minimum_distance_to_line_l311_311521


namespace find_fiftieth_term_l311_311989

noncomputable def b (n : ℕ) : ℝ := ∑ k in finset.range(n+1), real.cos k

theorem find_fiftieth_term :
  ∃ n : ℕ, (b n < 0) ∧ (n = 314) := sorry

end find_fiftieth_term_l311_311989


namespace horizontal_MN_determine_angles_l311_311905

variables {g α β a b φ t : ℝ}

def conditions := 
  0 <= α ∧ α < β ∧ t = sqrt ((2 * (a * sin α - b * sin β)) / (g * (sin α)^2 - (sin β)^2))

def feasibility := 
  (sin β / sin α) ≥ (a / b) ∧ (a / b) ≥ (sin α / sin β)

theorem horizontal_MN (h_conditions : conditions) (h_feasibility : feasibility) :
  t = sqrt ((2 * (a * sin α - b * sin β)) / (g * (sin (α))^2 - (sin (β))^2)) := 
sorry

variables {α β a b: ℝ}

def part_b_conditions := 
  0 <= α ∧ α < β ∧ φ > 0 ∧ α + β + φ = 180 ∧ 
  (a / sin α) = (b / sin β) ∧ 
  (a + b) / (a - b) = (tan ((α + β) / 2)) / (tan ((α - β) / 2)) ∧
  tan ((α + β) / 2) = 1 / (tan (φ / 2))

theorem determine_angles (h : part_b_conditions):
  α = ((α + β) + (α - β)) / 2 ∧ 
  β = ((α + β) - (α - β)) / 2 :=
 sorry

end horizontal_MN_determine_angles_l311_311905


namespace evaluate_expression_l311_311972

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 :=
by 
  sorry

end evaluate_expression_l311_311972


namespace sum_of_divisors_of_29_l311_311810

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311810


namespace part1_part2_l311_311341

-- Define m as a positive integer greater than or equal to 2
def m (k : ℕ) := k ≥ 2

-- Part 1: Existential statement for x_i's
theorem part1 (m : ℕ) (h : m ≥ 2) :
  ∃ (x : ℕ → ℤ),
    ∀ i, 1 ≤ i ∧ i ≤ m →
    x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1 := by
  sorry

-- Part 2: Infinite sequence y_k
theorem part2 (x : ℕ → ℤ) (m : ℕ) (h : m ≥ 2) :
  (∀ i, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
  ∃ (y : ℤ → ℤ),
    (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
    (∀ i, 1 ≤ i ∧ i ≤ 2 * m → y i = x i) := by
  sorry

end part1_part2_l311_311341


namespace average_speed_not_exact_l311_311389

--** Conditions
variable (x : ℝ) -- time (hours)
variable (f : ℝ → ℝ) -- speed function (km/h)
variable (h1 : ∀ t, 0 ≤ t ≤ 3.5 → ((∫ s in t..t+1, f s) = 5)) -- condition: covers exactly 5 km for each 1-hour interval

--** Statement
theorem average_speed_not_exact (f : ℝ → ℝ) (x : ℝ) (h1 : ∀ t, 0 ≤ t ≤ 3.5 → ((∫ s in t..t+1, f s) = 5)) :
  (∫ s in 0..3.5, f s) ≠ 3.5 * 5 := 
sorry

end average_speed_not_exact_l311_311389


namespace mrs_quick_speed_l311_311601
noncomputable theory

-- Defining the conditions
def distance (d : ℝ) (t : ℝ) : Prop := 
  -- If driving at 50 mph and arriving 5 minutes late
  d = 50 * (t + 5/60) ∧
  -- If driving at 75 mph and arriving 5 minutes early
  d = 75 * (t - 5/60)

-- Stating the proof problem
theorem mrs_quick_speed :
  ∃ (t d : ℝ), distance d t ∧ (d / t) = 60 :=
sorry

end mrs_quick_speed_l311_311601


namespace geometric_sequence_value_l311_311554

-- Definition of the problem conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a(n+1) = q * a(n)

variables (a : ℕ → ℝ) (q : ℝ)
hypothesis h1 : is_geometric_sequence a
hypothesis h2 : a 2 = 5
hypothesis h3 : a 6 = 33

-- Proof statement of the problem
theorem geometric_sequence_value : a 3 * a 5 = 165 :=
by {
  sorry
}

end geometric_sequence_value_l311_311554


namespace expansion_identity_l311_311484

theorem expansion_identity
  (a0 a1 a2 a3 a4 : ℝ)
  (h1 : (2 + sqrt 3)^4 = a0 + a1 + a2 + a3 + a4)
  (h2 : (-2 + sqrt 3)^4 = a0 - a1 + a2 - a3 + a4) :
  (a0 + a2 + a4)^2 - (a1 + a3)^2 = 1 := by
  sorry

end expansion_identity_l311_311484


namespace max_value_S_n_l311_311120

-- Define the sequence sum S_n and conditions
noncomputable def S_n (n : ℕ) (b : ℝ) := - (n : ℝ)^2 + b * n

-- Given condition a_2 = 8
def a_2_condition := ∀ b : ℝ, (S_n 2 b - S_n 1 b) = 8

-- Define the proof problem
theorem max_value_S_n : (∃ b : ℝ, 
  a_2_condition b ∧ 
  (∃ n : ℤ, 
    S_n n.natAbs b = (30 : ℝ) 
    ∧ ∀ m : ℤ, S_n m.natAbs b ≤ (30 : ℝ))) 
:= sorry

end max_value_S_n_l311_311120


namespace triangle_third_side_l311_311306

theorem triangle_third_side (AB AC AD : ℝ) (hAB : AB = 25) (hAC : AC = 30) (hAD : AD = 24) :
  ∃ BC : ℝ, (BC = 25 ∨ BC = 11) :=
by
  sorry

end triangle_third_side_l311_311306


namespace inequality_solution_set_l311_311681

noncomputable def solution_set (a b c : ℝ) (f : ℝ → ℝ) : set ℝ := {x | f x}

theorem inequality_solution_set (a b c : ℝ) :
  set_of (λ x : ℝ, (a * x^2 + b * x + c) > 0) = { x | (-1/3) < x ∧ x < 2 } →
  set_of (λ x : ℝ, (c * x^2 + b * x + a) < 0) = { x | (-3) < x ∧ x < (1/2) } :=
by
  sorry

end inequality_solution_set_l311_311681


namespace sum_of_divisors_29_l311_311821

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311821


namespace smallest_square_area_l311_311920

theorem smallest_square_area (a b c d s : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 3) (h4 : d = 5) (h5 : s = a + c) :
  ∃ S, S * S = 81 := by
  use 9
  have h6 : 9 * 9 = 81 := by norm_num
  exact h6

end smallest_square_area_l311_311920


namespace sum_of_divisors_of_29_l311_311808

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311808


namespace people_in_the_theater_l311_311543

theorem people_in_the_theater : ∃ P : ℕ, P = 100 ∧ 
  P = 19 + (1/2 : ℚ) * P + (1/4 : ℚ) * P + 6 := by
  sorry

end people_in_the_theater_l311_311543


namespace sum_of_divisors_prime_29_l311_311730

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311730


namespace minimum_value_of_T_l311_311029

theorem minimum_value_of_T (a b c : ℝ) (h1 : a > 0) (h2 : b > a) (h3 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  T := (a + b + c) / (b - a) ≥ 3 := sorry

end minimum_value_of_T_l311_311029


namespace parabola_focus_l311_311525

theorem parabola_focus (a : ℝ) : (∀ x : ℝ, y = a * x^2) ∧ ∃ f : ℝ × ℝ, f = (0, 1) → a = (1/4) := 
sorry

end parabola_focus_l311_311525


namespace triangle_side_c_l311_311104

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the respective angles

-- Conditions given
variable (h1 : Real.tan A = 2 * Real.tan B)
variable (h2 : a^2 - b^2 = (1 / 3) * c)

-- The proof problem
theorem triangle_side_c (h1 : Real.tan A = 2 * Real.tan B) (h2 : a^2 - b^2 = (1 / 3) * c) : c = 1 :=
by sorry

end triangle_side_c_l311_311104


namespace sum_of_divisors_of_29_l311_311885

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311885


namespace simplify_cubed_roots_l311_311220

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311220


namespace sum_of_divisors_29_l311_311863

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311863


namespace frac_subtraction_simplified_l311_311970

-- Definitions of the fractions involved.
def frac1 : ℚ := 8 / 19
def frac2 : ℚ := 5 / 57

-- The primary goal is to prove the equality.
theorem frac_subtraction_simplified : frac1 - frac2 = 1 / 3 :=
by {
  -- Proof of the statement.
  sorry
}

end frac_subtraction_simplified_l311_311970


namespace decreasing_interval_l311_311486

noncomputable def f (x : ℝ) : ℝ := log (x^2 - 2*x - 3) / log 2

theorem decreasing_interval (x : ℝ) (h : x < -1) : f x.is_decreasing_on set.Iio (-1) := by
  sorry

end decreasing_interval_l311_311486


namespace probability_non_adjacent_pairs_l311_311531

theorem probability_non_adjacent_pairs : 
  let numbers := {1, 2, 3, 4, 5},
      total_pairs := 10,
      adjacent_pairs := 4,
      non_adjacent_pairs := total_pairs - adjacent_pairs, -- 6
      probability := non_adjacent_pairs / total_pairs in
  probability = 0.6 :=
by 
  let numbers := {1, 2, 3, 4, 5},
      total_pairs := 10,
      adjacent_pairs := 4,
      non_adjacent_pairs := total_pairs - adjacent_pairs, -- 6
      probability := non_adjacent_pairs / total_pairs;
  sorry

end probability_non_adjacent_pairs_l311_311531


namespace sum_approx_l311_311144

theorem sum_approx :
  (p = 100) →
  (q = 100) →
  (r = 2) →
  let T := (∑ n in Finset.range 10000, 1 / Real.sqrt (n + Real.sqrt (n^2 + 2*n))) 
  in T = p + q * Real.sqrt r →
  p + q + r = 202 :=
by
  intros
  sorry

end sum_approx_l311_311144


namespace sum_of_divisors_of_29_l311_311878

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311878


namespace sum_of_divisors_of_prime_29_l311_311845

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311845


namespace intersection_A_B_l311_311038

/-- Definition of set A: the set of integers greater than -2 and less than or equal to 1 -/
def setA : Set ℤ := {x : ℤ | -2 < x ∧ x ≤ 1}

/-- Definition of set B: the set of natural numbers greater than -2 and less than 3. Natural numbers includes zero. -/
def setB : Set ℕ := {x : ℕ | -2 < x ∧ x < 3}

/--
Theorem: The intersection of sets A and B is {0, 1}.
-/
theorem intersection_A_B :
  (setA ∩ setB : Set ℤ) = {0, 1} := 
by
  sorry

end intersection_A_B_l311_311038


namespace determine_m_l311_311019

noncomputable def function_f (m : ℝ) (x : ℝ) : ℝ := m * x - |x + 1|

def exists_constant_interval (a b c m : ℝ) : Prop :=
  a < b ∧ ∀ x, a ≤ x ∧ x ≤ b → function_f m x = c

theorem determine_m (m : ℝ) (a b c : ℝ) :
  (a < b ∧ a ≥ -2 ∧ b ≥ -2 ∧ (∀ x, a ≤ x ∧ x ≤ b → function_f m x = c)) →
  m = 1 ∨ m = -1 :=
sorry

end determine_m_l311_311019


namespace vertex_coordinates_of_parabola_l311_311688

noncomputable def parabola_vertex_coordinates : ℝ × ℝ :=
let y := λ x : ℝ, x^2 + 2*x - 3 in
(-1, -4)

theorem vertex_coordinates_of_parabola :
  ∃ x v_y : ℝ, (v_y = (x + 1)^2 - 4) ∧ v_y = -4 ∧ x = -1 :=
begin
  use [-1, -4],
  split,
  { dsimp, linarith },
  { split,
    { dsimp, ring },
    { rfl }
  }
end

end vertex_coordinates_of_parabola_l311_311688


namespace consecutive_numbers_product_l311_311288

theorem consecutive_numbers_product : 
  ∃ n : ℕ, (n + n + 1 = 11) ∧ (n * (n + 1) * (n + 2) = 210) :=
sorry

end consecutive_numbers_product_l311_311288


namespace count_divisible_by_3_5_7_in_range_l311_311508

theorem count_divisible_by_3_5_7_in_range (a b : ℕ) (h_range : a ≤ b) (h_a_def : a = 1) (h_b_def : b = 200) :
  ∃ n : ℕ, n = ∑ k in (finset.range ((b / nat.lcm 3 (nat.lcm 5 7)) + 1)), k * (nat.lcm 3 (nat.lcm 5 7)) → n = 1 :=
by
  sorry

end count_divisible_by_3_5_7_in_range_l311_311508


namespace sum_of_divisors_of_29_l311_311712

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311712


namespace micro_lesson_problem_l311_311322

noncomputable def micro_lesson_cost (x y : ℕ) : Prop :=
  (2 * x + 3 * y = 2900) ∧ (3 * x + 4 * y = 4100)

noncomputable def profit_function (a : ℕ) : ℕ :=
  50 * a + 16500

noncomputable def constraints : Prop :=
  ∀ a : ℕ, (0 < a) → (a ≤ 66 / 7) → ((33 - 1.5 * (22 - a)) ≥ 2 * a)

noncomputable def max_profit (max_profit_value : ℕ) (a_value : ℕ) : Prop :=
  max_profit_value = profit_function a_value ∧ max_profit_value = 16900 ∧ a_value = 8

theorem micro_lesson_problem :
  (∃ x y, micro_lesson_cost x y ∧ x = 700 ∧ y = 500) ∧
  (∃ max_profit_value a_value, max_profit max_profit_value a_value ∧ constraints) :=
begin
  sorry
end

end micro_lesson_problem_l311_311322


namespace Greg_PPO_Obtained_90_Percent_l311_311075

theorem Greg_PPO_Obtained_90_Percent :
  let max_procgen_reward := 240
  let max_coinrun_reward := max_procgen_reward / 2
  let greg_reward := 108
  (greg_reward / max_coinrun_reward * 100) = 90 := by
  sorry

end Greg_PPO_Obtained_90_Percent_l311_311075


namespace problem_solution_l311_311677

variables {a b c : ℝ}

theorem problem_solution
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) :
  (a + b) * (b + c) * (a + c) = 0 := 
sorry

end problem_solution_l311_311677


namespace cube_root_multiplication_l311_311202

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311202


namespace expected_value_X_l311_311455

-- Definitions and assumptions
noncomputable def X : ℕ → ℕ := sorry -- This represents the binomial random variable

-- Assumption: The random variable X follows B(6, 1/3)
axiom binomial_X : ∀ (n p : ℕ → ℕ), n = 6 ∧ p = (1 / 3) → X n = X p

-- The theorem we want to prove
theorem expected_value_X : E X = 2 := by
  sorry

end expected_value_X_l311_311455


namespace not_odd_function_l311_311054

noncomputable def f (x : ℝ) : ℝ := -Real.sin (x + Real.pi / 2)

theorem not_odd_function : ¬(∀ x : ℝ, f (-x) = - f x) :=
by
  -- Sketch of reasoning as per the solution:
  -- We know f(x) = -cos x, which is an even function.
  -- Thus, f(-x) = f(x), not -f(x).
  intro h
  -- Show contradiction using the even property of f
  have h_even : ∀ x : ℝ, f (-x) = f x := by sorry
  replace h := h_even 1
  contradiction

end not_odd_function_l311_311054


namespace julia_played_more_kids_l311_311571

variable (kidsPlayedMonday : Nat) (kidsPlayedTuesday : Nat)

theorem julia_played_more_kids :
  kidsPlayedMonday = 11 →
  kidsPlayedTuesday = 12 →
  kidsPlayedTuesday - kidsPlayedMonday = 1 :=
by
  intros hMonday hTuesday
  sorry

end julia_played_more_kids_l311_311571


namespace farm_area_l311_311385

theorem farm_area
  (b : ℕ) (l : ℕ) (d : ℕ)
  (h_b : b = 30)
  (h_cost : 15 * (l + b + d) = 1800)
  (h_pythagorean : d^2 = l^2 + b^2) :
  l * b = 1200 :=
by
  sorry

end farm_area_l311_311385


namespace part_I_part_II_l311_311036

def A := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x < a}
def M (m : ℝ) : Set ℝ := {x | x^2 - (1 + m) * x + m = 0}

theorem part_I (a : ℝ) : A ∩ B a = A → a > 2 :=
  sorry

theorem part_II (m : ℝ) (hm : m > 1) : 
  A ∪ M m = if m = 2 then {1, 2} else {1, 2, m} :=
  sorry

end part_I_part_II_l311_311036


namespace fewest_keystrokes_l311_311361

theorem fewest_keystrokes (start target : ℕ) (steps : ℕ) : start = 1 ∧ target = 200 ∧ steps = 9 →
  ∃ seq : list ℕ, (∀ n ∈ seq, n = 1 ∨ n = 2) ∧
  list.foldl (λ acc k, if k = 1 then acc + 1 else acc * 2) start seq = target ∧
  seq.length = steps :=
by
  intros h
  sorry

end fewest_keystrokes_l311_311361


namespace loaned_books_l311_311382

theorem loaned_books (initial_books : ℕ) (returned_percent : ℝ)
  (end_books : ℕ) (damaged_books : ℕ) (L : ℝ) :
  initial_books = 150 ∧
  returned_percent = 0.85 ∧
  end_books = 135 ∧
  damaged_books = 5 ∧
  0.85 * L + 5 + (initial_books - L) = end_books →
  L = 133 :=
by
  intros h
  rcases h with ⟨hb, hr, he, hd, hsum⟩
  repeat { sorry }

end loaned_books_l311_311382


namespace cube_root_simplification_l311_311245

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311245


namespace friction_coefficient_example_l311_311357

variable (α : ℝ) (mg : ℝ) (μ : ℝ)

theorem friction_coefficient_example
    (hα : α = 85 * Real.pi / 180) -- converting degrees to radians
    (hN : ∀ (N : ℝ), N = 6 * mg) -- Normal force in the vertical position
    (F : ℝ) -- Force applied horizontally by boy
    (hvert : F * Real.sin α - mg + (6 * mg) * Real.cos α = 0) -- vertical equilibrium
    (hhor : F * Real.cos α - μ * (6 * mg) - (6 * mg) * Real.sin α = 0) -- horizontal equilibrium
    : μ = 0.08 :=
by
  sorry

end friction_coefficient_example_l311_311357


namespace sum_of_divisors_of_29_l311_311743

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311743


namespace sum_of_divisors_of_prime_l311_311788

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311788


namespace product_of_45_and_360_ends_with_two_zeros_l311_311080

-- Define the conditions (factorizations)
def factor_45 : (ℕ × ℕ × ℕ) := (0, 2, 1)  -- (powers of 2, 3, and 5 respectively: 45 = 2^0 * 3^2 * 5^1)
def factor_360 : (ℕ × ℕ × ℕ) := (3, 2, 1) -- (powers of 2, 3, and 5 respectively: 360 = 2^3 * 3^2 * 5^1)

-- Define the statement
theorem product_of_45_and_360_ends_with_two_zeros :
  let (p2, p3, p5) := (factor_45.1 + factor_360.1, factor_45.2 + factor_360.2, factor_45.3 + factor_360.3)
  in min p2 p5 = 2 :=
by
  -- Theorem includes conditions expressions directly
  sorry

end product_of_45_and_360_ends_with_two_zeros_l311_311080


namespace large_monkey_doll_cost_l311_311998

theorem large_monkey_doll_cost (S L E : ℝ) 
  (h1 : S = L - 2) 
  (h2 : E = L + 1) 
  (h3 : 300 / S = 300 / L + 25) 
  (h4 : 300 / E = 300 / L - 15) : 
  L = 6 := 
sorry

end large_monkey_doll_cost_l311_311998


namespace exists_point_K_l311_311563

noncomputable def acute_angle_triangle
  (A B C : Type) [inhabited A] [inhabited B] [inhabited C]
  (triangle_ABC : Triangle A B C)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (hA : 0 < angle_A ∧ angle_A < π / 2)
  (hB : 0 < angle_B ∧ angle_B < π / 2)
  (hC : 0 < angle_C ∧ angle_C < π / 2)
  (hABC : angle_A + angle_B + angle_C = π) : Prop := sorry

theorem exists_point_K
  (A B C : Type) [inhabited A] [inhabited B] [inhabited C]
  (triangle_ABC : Triangle A B C)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (hA : 0 < angle_A ∧ angle_A < π / 2)
  (hB : 0 < angle_B ∧ angle_B < π / 2)
  (hC : 0 < angle_C ∧ angle_C < π / 2)
  (hABC : angle_A + angle_B + angle_C = π) :
  ∃ K : Point, ∠K B A = 2 * ∠K A B ∧ ∠K B C = 2 * ∠K C B := sorry

end exists_point_K_l311_311563


namespace probability_sum_even_l311_311605

theorem probability_sum_even (cards : Finset ℕ) (h : cards = {1, 2, 3, 4, 5}) :
  (let n := (cards.choose 2).card in 
   let evens := {x | x ∈ cards ∧ x % 2 = 0} in
   let odds := {x | x ∈ cards ∧ x % 2 = 1} in
   let m := (evens.choose 2).card + (odds.choose 2).card in
   m / n : ℚ) = 2 / 5 :=
by
  sorry

end probability_sum_even_l311_311605


namespace sum_of_divisors_of_29_l311_311806

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311806


namespace harmonic_mean_inequality_l311_311044

theorem harmonic_mean_inequality
  (n : ℕ)
  (x : (Fin n) → ℝ)
  (h_pos : ∀ i, 0 < x i) :
  ∑ i : Fin n, 1 / (x i) ≥ 2 * ∑ i : Fin n, 1 / (x i + x ((i + 1) % n)) :=
by
  sorry

end harmonic_mean_inequality_l311_311044


namespace scientific_notation_of_0_000815_l311_311317

theorem scientific_notation_of_0_000815 :
  (∃ (c : ℝ) (n : ℤ), 0.000815 = c * 10^n ∧ 1 ≤ c ∧ c < 10) ∧ (0.000815 = 8.15 * 10^(-4)) :=
by
  sorry

end scientific_notation_of_0_000815_l311_311317


namespace triangle_inequality_l311_311320

theorem triangle_inequality (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 → True :=
by
  intro h
  cases h with
  | intro h1 h2 h3 => sorry

end triangle_inequality_l311_311320


namespace parallelogram_area_l311_311634

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l311_311634


namespace path_length_PQ_l311_311672

theorem path_length_PQ :
  ∀ (P Q V W X Y Z : Type)
  (PQ_len part_len : ℝ),
  (PQ_len = 24) →
  (part_len = PQ_len / 6) →
  (∀ (PV VW WX XY YZ ZQ : ℝ), 
    (PV = part_len) ∧ (VW = part_len) ∧ (WX = part_len) ∧ (XY = part_len) ∧ (YZ = part_len) ∧ (ZQ = part_len)
  ) →
  (path_len : ℝ) (path_len = 6 * (3 * part_len)) →
  path_len = 72 :=
by
  intros P Q V W X Y Z PQ_len part_len h1 h2 h3 path_len h4
  sorry

end path_length_PQ_l311_311672


namespace polar_to_cartesian_l311_311678

variable (ρ α : ℝ)

def polar_eq : Prop := ρ * (sin α)^2 - 2 * cos α = 0

def cartesian_x : ℝ := ρ * cos α
def cartesian_y : ℝ := ρ * sin α

theorem polar_to_cartesian (ρ α : ℝ) (h : polar_eq ρ α) : (cartesian_y ρ α)^2 = 2 * (cartesian_x ρ α) :=
by
  sorry

end polar_to_cartesian_l311_311678


namespace movie_profit_proof_l311_311368

theorem movie_profit_proof
  (cost_actors : ℝ) 
  (num_people : ℝ)
  (cost_food_per_person : ℝ) 
  (cost_equipment_multiplier : ℝ) 
  (selling_price : ℝ) :
  cost_actors = 1200 →
  num_people = 50 →
  cost_food_per_person = 3 →
  cost_equipment_multiplier = 2 →
  selling_price = 10000 →
  let cost_food := num_people * cost_food_per_person in
  let total_cost_without_equipment := cost_actors + cost_food in
  let cost_equipment := cost_equipment_multiplier * total_cost_without_equipment in
  let total_cost := total_cost_without_equipment + cost_equipment in
  let profit := selling_price - total_cost in
  profit = 5950 :=
by
  intros h1 h2 h3 h4 h5
  have h_cost_food : cost_food = 150, by rw [h2, h3]; norm_num
  have h_total_cost_without_equipment : total_cost_without_equipment = 1350, by rw [h1, h_cost_food]; norm_num
  have h_cost_equipment : cost_equipment = 2700, by rw [h4, h_total_cost_without_equipment]; norm_num
  have h_total_cost : total_cost = 4050, by rw [h_total_cost_without_equipment, h_cost_equipment]; norm_num
  have h_profit : profit = 5950, by rw [h5, h_total_cost]; norm_num
  exact h_profit

end movie_profit_proof_l311_311368


namespace problem_f_f2_equals_16_l311_311487

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 3 then x^2 else 2^x

theorem problem_f_f2_equals_16 : f (f 2) = 16 :=
by
  sorry

end problem_f_f2_equals_16_l311_311487


namespace min_w_value_l311_311418

def w (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 45

theorem min_w_value : ∀ x y : ℝ, (w x y) ≥ 28 ∧ (∃ x y : ℝ, (w x y) = 28) :=
by
  sorry

end min_w_value_l311_311418


namespace ellipse_foci_y_axis_range_l311_311315

noncomputable def is_ellipse_with_foci_on_y_axis (k : ℝ) : Prop :=
  (k > 5) ∧ (k < 10) ∧ (10 - k > k - 5)

theorem ellipse_foci_y_axis_range (k : ℝ) :
  is_ellipse_with_foci_on_y_axis k ↔ 5 < k ∧ k < 7.5 := 
by
  sorry

end ellipse_foci_y_axis_range_l311_311315


namespace simplified_expression_correct_l311_311208

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311208


namespace solve_ellipse_eq_solve_b_value_solve_lambda_mu_l311_311482

-- Definitions based on conditions
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

def eccentricity (c a : ℝ) : Prop :=
  c / a = (Real.sqrt 6) / 3

def distance_from_origin_to_line (b : ℝ) : ℝ :=
  Real.sqrt 2

-- Given conditions and values
variables {a b c : ℝ}

-- Prove (I)
theorem solve_ellipse_eq (h₁ : a > b)
                         (h₂ : b > 0)
                         (h₃ : distance_from_origin_to_line b = Real.sqrt 2)
                         (h₄ : eccentricity c a)
                         (h₅ : a^2 - b^2 = c^2) :
  (a = Real.sqrt 12) ∧ (b = 2) := sorry

-- Prove (II)(i)
theorem solve_b_value (h₆ : |AB| = Real.sqrt 3)
                      (h₇ : eccentricity c a) :
  b = 1 := sorry

-- Prove (II)(ii)
theorem solve_lambda_mu (M A B : ℝ × ℝ)
                        (λ μ : ℝ)
                        (h₇ : ∀ (M : ℝ × ℝ), ellipse a b (⟨h₁, h₂⟩) M.1 M.2)
                        (h₈ : M = λ • A + μ • B)
                        (x1 x2 y1 y2 : ℝ)
                        (h₉ : A = (x1, y1))
                        (h₁₀ : B = (x2, y2))
                        (h₁₁ : (x1^2 + 3 * y1^2 = 3 * b^2) ∧ (x2^2 + 3 * y2^2 = 3 * b^2)) :
  λ^2 + μ^2 = 1 := sorry

end solve_ellipse_eq_solve_b_value_solve_lambda_mu_l311_311482


namespace smallest_possible_degree_l311_311279

theorem smallest_possible_degree (p : Polynomial ℝ)
  (h : ∃ l : ℝ, ∀ x : ℝ, IsLimitAtFilter ((3 * x^8 + 4 * x^7 - 2 * x^3 - 5) / p.eval x) l filter.at_top) :
  ∃ k : ℕ, k = p.degree ∧ k ≥ 8 :=
sorry

end smallest_possible_degree_l311_311279


namespace simplify_cuberoot_product_l311_311231

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311231


namespace cost_per_mile_l311_311131

theorem cost_per_mile (rental_cost gas_price_per_gallon total_expense miles_driven : ℝ) 
    (gallons_bought : ℕ) 
    (h_rental : rental_cost = 150) 
    (h_gas_price : gas_price_per_gallon = 3.50) 
    (h_total_expense : total_expense = 338) 
    (h_gallons_bought : gallons_bought = 8) 
    (h_miles_driven : miles_driven = 320) : 
    let gas_cost := gallons_bought * gas_price_per_gallon,
        total_known_cost := rental_cost + gas_cost,
        driving_cost := total_expense - total_known_cost,
        cost_per_mile := driving_cost / miles_driven 
    in cost_per_mile = 0.50 :=
by {
  sorry,
}

end cost_per_mile_l311_311131


namespace sum_of_divisors_of_prime_l311_311780

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311780


namespace largest_divisor_is_one_l311_311588

theorem largest_divisor_is_one (p q : ℤ) (hpq : p > q) (hp : p % 2 = 1) (hq : q % 2 = 0) :
  ∀ d : ℤ, (∀ p q : ℤ, p > q → p % 2 = 1 → q % 2 = 0 → d ∣ (p^2 - q^2)) → d = 1 :=
sorry

end largest_divisor_is_one_l311_311588


namespace parallelogram_area_l311_311618

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l311_311618


namespace length_segment_AB_l311_311555

def line_parametric (t : ℝ) : ℝ × ℝ := 
  ( (2 * real.sqrt 5 / 5) * t, 1 + (real.sqrt 5 / 5) * t )

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

theorem length_segment_AB :
  let t₁ t₂ := (
    λ t, (1 + (2 * real.sqrt 5 / 5) * t + (1 / 5) * t^2 = (8 * real.sqrt 5 / 5) * t)
  ) in
  let distance := real.sqrt ((t₁ + t₂)^2 - 4 * t₁ * t₂) in
  distance = 4 * real.sqrt 10 :=
sorry

end length_segment_AB_l311_311555


namespace sum_of_divisors_of_29_l311_311750

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311750


namespace meeting_time_is_correct_l311_311323

variable (track_length : ℕ) (speed_A_kmph speed_B_kmph : ℕ)

def kmph_to_mps (kmph : ℕ) : ℕ :=
  (kmph * 1000) / 3600

def time_to_complete_lap (track_length speed_mps : ℕ) : ℕ :=
  track_length / speed_mps

def lcm (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

noncomputable def time_to_meet (track_length speed_A_kmph speed_B_kmph : ℕ) : ℕ :=
  let speed_A_mps := kmph_to_mps speed_A_kmph
  let speed_B_mps := kmph_to_mps speed_B_kmph
  let time_A := time_to_complete_lap track_length speed_A_mps
  let time_B := time_to_complete_lap track_length speed_B_mps
  lcm time_A time_B

theorem meeting_time_is_correct :
  time_to_meet 900 36 54 = 180 :=
  sorry

end meeting_time_is_correct_l311_311323


namespace expression_meaningful_iff_l311_311097

theorem expression_meaningful_iff (x : ℝ) : (∃ y, y = (sqrt (x + 1) / x)) ↔ (x ≥ -1 ∧ x ≠ 0) := 
by 
  sorry

end expression_meaningful_iff_l311_311097


namespace rod_sliding_friction_l311_311355

noncomputable def coefficient_of_friction (mg : ℝ) (F : ℝ) (α : ℝ) := 
  (F * Real.cos α - 6 * mg * Real.sin α) / (6 * mg)

theorem rod_sliding_friction
  (α : ℝ)
  (hα : α = 85 * Real.pi / 180)
  (mg : ℝ)
  (hmg_pos : 0 < mg)
  (F : ℝ)
  (hF : F = (mg - 6 * mg * Real.cos 85) / Real.sin 85) :
  coefficient_of_friction mg F α = 0.08 := 
by
  simp [coefficient_of_friction, hα, hF, Real.cos, Real.sin]
  sorry

end rod_sliding_friction_l311_311355


namespace problem1_proof_problem2_proof_l311_311347

-- Problem 1: Calculation proof
noncomputable def problem1_statement : Prop :=
  (3.14 - Real.pi)^0 - (1/2)^(-2) + 2 * Real.cos (Real.pi / 3) - Real.abs (1 - Real.sqrt 3) + Real.sqrt 12 = 
  Real.sqrt 3 - 1

theorem problem1_proof : problem1_statement := 
by 
  assume h,
  sorry

-- Problem 2: Inequality system proof
def problem2_statement : Prop :=
  ∀ x : ℝ, (2 * x - 6 < 0 → x < 3) ∧ 
  ((1 - 3 * x) / 2 ≤ 5 → -3 ≤ x) ∧ 
  (-3 ≤ x → x < 3 → true)

theorem problem2_proof : problem2_statement :=
by 
  assume x,
  sorry

end problem1_proof_problem2_proof_l311_311347


namespace adjusted_sale_price_correct_l311_311289

variable (X : ℝ) (Y : ℝ)

def cost_price : ℝ := (832 + 448) / 2

def adjusted_cost_price (X : ℝ) : ℝ := cost_price + 2 * X

def sale_price (C_adj : ℝ) : ℝ := 1.55 * C_adj

def adjusted_sale_price (X : ℝ) (Y : ℝ) : ℝ :=
  (sale_price (adjusted_cost_price X)) * (1 - Y / 100)

theorem adjusted_sale_price_correct (X Y : ℝ) :
  adjusted_sale_price X Y = (992 + 3.1 * X) * (1 - Y / 100) :=
by
  sorry

end adjusted_sale_price_correct_l311_311289


namespace largest_whole_number_solution_l311_311701

theorem largest_whole_number_solution :
  ∃ x : ℕ, ∀ (h₁ : x ≤ 6) (h₂ : ∀ y : ℕ, y ≤ 6 → y ≤ x → ¬ (frac 1 / (4 : ℚ) + frac y / (5 : ℚ)) < (frac 3 / (2 : ℚ)) ) → 
  frac 1 / (4 : ℚ) + frac x / (5 : ℚ) < (frac 3 / (2 : ℚ)) :=
sorry

end largest_whole_number_solution_l311_311701


namespace length_of_O₁O₂_constant_l311_311033

variable {A B C D E : Type} [Point A] [Point B] [Point C] [Point D] [Point E]

/-- Given a trapezoid ABCD with AD ∥ BC, and E is a moving point on the side AB.
Let O₁ and O₂ be the circumcenters of triangles AED and BEC, respectively.
The length of O₁O₂ is a constant value, which is (AB / 2). -/
theorem length_of_O₁O₂_constant {A B C D E O₁ O₂ : Point} 
  (htrapezoid : is_trapezoid A B C D)
  (hAD_parallel_BC : parallel AD BC)
  (hE_on_AB : on_line E A B)
  (hO₁_circumcenter_AED : circumcenter O₁ A E D)
  (hO₂_circumcenter_BEC : circumcenter O₂ B E C) :
  distance O₁ O₂ = (distance A B) / 2 := 
sorry -- proof

end length_of_O₁O₂_constant_l311_311033


namespace scientific_notation_35_million_l311_311952

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 : Float) ^ 7 := 
by
  sorry

end scientific_notation_35_million_l311_311952


namespace f_2011_l311_311155

noncomputable def f : ℝ → ℝ
| x := sorry  -- The function is defined but not implemented.

-- Context/Assumptions
axiom f_func (x : ℝ) : f(x + 2) = f(x + 1) - f(x)
axiom f_1 : f 1 = Real.log (3 / 2)
axiom f_2 : f 2 = Real.log 15

-- Goal
theorem f_2011 : f 2011 = Real.log (3 / 2) :=
by sorry

end f_2011_l311_311155


namespace revenue_from_full_price_tickets_l311_311386

-- Let's define our variables and assumptions
variables (f h p: ℕ)

-- Total number of tickets sold
def total_tickets (f h: ℕ) : Prop := f + h = 200

-- Total revenue from tickets
def total_revenue (f h p: ℕ) : Prop := f * p + h * (p / 3) = 2500

-- Statement to prove the revenue from full-price tickets
theorem revenue_from_full_price_tickets (f h p: ℕ) (hf: total_tickets f h) 
  (hr: total_revenue f h p): f * p = 1250 :=
sorry

end revenue_from_full_price_tickets_l311_311386


namespace sum_of_divisors_of_29_l311_311715

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311715


namespace max_value_of_expression_l311_311433

theorem max_value_of_expression : 
  ∃ x : ℝ, ∀ y : ℝ, (3^x - 9^x) ≤ (3^y - 9^y) := sorry

end max_value_of_expression_l311_311433


namespace tim_age_difference_l311_311129

theorem tim_age_difference (j_turned_23_j_turned_35 : ∃ (j_age_when_james_23 : ℕ) (john_age_when_james_23 : ℕ), 
                                          j_age_when_james_23 = 23 ∧ john_age_when_james_23 = 35)
                           (tim_age : ℕ) (tim_age_eq : tim_age = 79)
                           (tim_age_twice_john_age_less_X : ∃ (X : ℕ) (john_age : ℕ), tim_age = 2 * john_age - X) :
  ∃ (X : ℕ), X = 15 :=
by
  sorry

end tim_age_difference_l311_311129


namespace rationalize_cubic_denominator_l311_311644

theorem rationalize_cubic_denominator :
  let a := real.cbrt 5
  let b := real.cbrt 4
  let numerator := real.cbrt(25) + real.cbrt(20) + real.cbrt(16)
  let denominator := 1
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  (1 / (a - b) = numerator / denominator) → X + Y + Z + W = 62 :=
by
  sorry

end rationalize_cubic_denominator_l311_311644


namespace length_of_segment_l311_311704

theorem length_of_segment (x : ℤ) (hx : |x - 3| = 4) : 
  let a := 7
  let b := -1
  a - b = 8 := by
    sorry

end length_of_segment_l311_311704


namespace focal_length_hyperbola_eq_six_l311_311028

-- Define the hyperbola and associated parameters
def hyperbola_focal_length (a : ℝ) (h : a > 0) : ℝ :=
  let c := 3 in -- derived from calculations in the solution
  2 * c -- focal length is 2 * c

-- Proof problem: Proving the focal length of the hyperbola given the conditions
theorem focal_length_hyperbola_eq_six (a : ℝ) (h : a > 0) :
  let c := 3 in
  hyperbola_focal_length a h = 6 :=
by
  sorry -- proof to be provided

end focal_length_hyperbola_eq_six_l311_311028


namespace parallel_vectors_l311_311447

variable (a b : ℝ × ℝ)
variable (m : ℝ)

theorem parallel_vectors (h₁ : a = (-6, 2)) (h₂ : b = (m, -3)) (h₃ : a.1 * b.2 = a.2 * b.1) : m = 9 :=
by
  sorry

end parallel_vectors_l311_311447


namespace sum_of_divisors_of_29_l311_311770

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311770


namespace cost_split_evenly_l311_311569

noncomputable def total_cost (num_cupcakes : ℕ) (cost_per_cupcake : ℚ) : ℚ :=
  num_cupcakes * cost_per_cupcake

noncomputable def cost_per_person (total_cost : ℚ) : ℚ :=
  total_cost / 2

theorem cost_split_evenly (num_cupcakes : ℕ) (cost_per_cupcake : ℚ) (total_cost : ℚ) :
  num_cupcakes = 12 →
  cost_per_cupcake = 3/2 →
  total_cost = total_cost num_cupcakes cost_per_cupcake →
  cost_per_person total_cost = 9 := 
by
  intros h1 h2 h3
  sorry

end cost_split_evenly_l311_311569


namespace combination_with_repetition_l311_311443

theorem combination_with_repetition {n r : ℕ} : 
  (F_n^r : ℕ) = nat.choose (n + r - 1) r := 
sorry

end combination_with_repetition_l311_311443


namespace inverse_proportion_m_value_l311_311089

theorem inverse_proportion_m_value (m : ℝ) (x : ℝ) (h : y = (m - 2) * x ^ (m^2 - 5)) : 
  y is inverse_proportional_function → m = -2 :=
sorry

end inverse_proportion_m_value_l311_311089


namespace circle_tangent_to_line_l311_311052

theorem circle_tangent_to_line :
  ∃ (a : ℝ), (a > 0) ∧ ∀ (x y : ℝ), ((3 * a + 4) / (real.sqrt (3 ^ 2 + 4 ^ 2)) = 2) →
    ((x - a) ^ 2 + y ^ 2 = 2 ^ 2) :=
by
  sorry

end circle_tangent_to_line_l311_311052


namespace speedster_convertibles_l311_311360

theorem speedster_convertibles (V : ℕ) (hV : V = 80) :
  let Speedsters := V - 50 in
  let convertibles := (4 / 5 : ℚ) * Speedsters in
  convertibles = 24 := by
  let Speedsters := V - 50
  have hS : Speedsters = 30 := by
    simp [hV, Speedsters]
  let convertibles := (4 / 5 : ℚ) * Speedsters
  calc
    convertibles = (4 / 5 : ℚ) * 30 : by simp [hS]
    ... = 24 : by norm_num

end speedster_convertibles_l311_311360


namespace necessary_and_sufficient_condition_perpendicular_lines_l311_311914

def are_perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x + y = 0) → (x - a * y = 0) → x = 0 ∧ y = 0

theorem necessary_and_sufficient_condition_perpendicular_lines :
  ∀ (a : ℝ), are_perpendicular a → a = 1 :=
sorry

end necessary_and_sufficient_condition_perpendicular_lines_l311_311914


namespace range_of_m_l311_311493

noncomputable def f (x m : ℝ) := Real.exp x + x^2 / m^2 - x

theorem range_of_m (m : ℝ) (hm : m ≠ 0) :
  (∀ a b : ℝ, a ∈ Set.Icc (-1) 1 -> b ∈ Set.Icc (-1) 1 -> |f a m - f b m| ≤ Real.exp 1) ↔
  (m ∈ Set.Iic (-Real.sqrt 2 / 2) ∪ Set.Ici (Real.sqrt 2 / 2)) :=
by
  sorry

end range_of_m_l311_311493


namespace CarmenBrushLengthInCentimeters_l311_311975

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters_l311_311975


namespace sum_of_divisors_of_29_l311_311747

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311747


namespace interval_of_monotonic_decrease_find_c_l311_311060

noncomputable theory

def f (x : ℝ) : ℝ := sqrt 3 * sin (x / 2) * cos (x / 2) - cos (x / 2) ^ 2 + 1 / 2

theorem interval_of_monotonic_decrease :
  ∀ k : ℤ, ∃ I : set ℝ, I = set.Icc (2 * real.pi / 3 + 2 * k * real.pi) (5 * real.pi / 3 + 2 * k * real.pi) ∧
  ∀ x ∈ I, ∀ ε > 0, x + ε ∈ I → f(x) ≥ f(x + ε) :=
sorry

theorem find_c (A B C a b c : ℝ) (hA : f(A) = 1 / 2) (ha : a = sqrt 3) (hsin : sin B = 2 * sin C) :
  a = sqrt 3 ∧ sin B = 2 * sin C ∧ ∃ c : ℝ, c = 1 :=
sorry

end interval_of_monotonic_decrease_find_c_l311_311060


namespace cheddar_cheese_slices_l311_311174

-- Define the conditions
def cheddar_slices (C : ℕ) := ∃ (packages : ℕ), packages * C = 84
def swiss_slices := 28
def randy_bought_same_slices (C : ℕ) := swiss_slices = 28 ∧ 84 = 84

-- Lean theorem statement to prove the number of slices per package of cheddar cheese equals 28.
theorem cheddar_cheese_slices {C : ℕ} (h1 : cheddar_slices C) (h2 : randy_bought_same_slices C) : C = 28 :=
sorry

end cheddar_cheese_slices_l311_311174


namespace extreme_values_range_of_a_inequality_of_zeros_l311_311059

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -2 * (Real.log x) - a / (x ^ 2) + 1

theorem extreme_values (a : ℝ) (h : a = 1) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≤ 0) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = 0) ∧
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≥ -3 + 2 * (Real.log 2)) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = -3 + 2 * (Real.log 2)) :=
sorry

theorem range_of_a :
  (∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 0 < a ∧ a < 1) :=
sorry

theorem inequality_of_zeros (a : ℝ) (h : 0 < a) (h1 : a < 1) (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) (hx1x2 : x1 ≠ x2) :
  1 / (x1 ^ 2) + 1 / (x2 ^ 2) > 2 / a :=
sorry

end extreme_values_range_of_a_inequality_of_zeros_l311_311059


namespace sum_divisible_by_100_l311_311640

theorem sum_divisible_by_100 (S : Finset ℤ) (hS : S.card = 200) : 
  ∃ T : Finset ℤ, T ⊆ S ∧ T.card = 100 ∧ (T.sum id) % 100 = 0 := 
  sorry

end sum_divisible_by_100_l311_311640


namespace sum_of_divisors_of_29_l311_311775

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311775


namespace lines_perpendicular_implies_parallel_l311_311556

-- Define lines l₁, l₂, l₃ and the plane P
variables {P : Type} [Plane P] {l₁ l₂ l₃ : Line P}

-- Assume l₁ is perpendicular to l₂, and l₂ is perpendicular to l₃.
axiom perp_l₁_l₂ : is_perpendicular l₁ l₂
axiom perp_l₂_l₃ : is_perpendicular l₂ l₃

-- Prove that l₁ is parallel to l₃
theorem lines_perpendicular_implies_parallel :
  is_parallel l₁ l₃ :=
by
  -- The proof goes here
  sorry

end lines_perpendicular_implies_parallel_l311_311556


namespace sum_of_digits_next_square_starting_with_two_2s_l311_311918

theorem sum_of_digits_next_square_starting_with_two_2s :
  (∃ n > 15, (n^2 ≥ 200 ∧ n^2 < 300) → ∑ d in (2500.digits 10), d = 7) :=
sorry

end sum_of_digits_next_square_starting_with_two_2s_l311_311918


namespace h1n1_diameter_in_meters_l311_311658

theorem h1n1_diameter_in_meters (diameter_nm : ℝ) (h : diameter_nm = 85) : 
  (85 * 10⁻⁹ : ℝ) = (8.5 * 10⁻⁸ : ℝ) :=
by
  have : (85 : ℝ) * (10⁻⁹ : ℝ) = 8.5 * 10⁻⁸ := sorry
  exact this

end h1n1_diameter_in_meters_l311_311658


namespace problem_part_c_problem_part_d_l311_311550

noncomputable def binomial_expansion_sum_coefficients : ℕ :=
  (1 + (2 : ℚ)) ^ 6

theorem problem_part_c :
  binomial_expansion_sum_coefficients = 729 := 
by sorry

noncomputable def binomial_expansion_sum_binomial_coefficients : ℕ :=
  (2 : ℚ) ^ 6

theorem problem_part_d :
  binomial_expansion_sum_binomial_coefficients = 64 := 
by sorry

end problem_part_c_problem_part_d_l311_311550


namespace unique_solution_eq_l311_311642

theorem unique_solution_eq (x y m n : ℕ) (hx : x > y) (hy : y > 0) (hm : m > 1) (hn : n > 1) :
  (x + y)^n = x^m + y^m → (x = 1 ∧ y = 1 ∧ n = m) :=
begin
  sorry
end

end unique_solution_eq_l311_311642


namespace relationship_among_exponentials_l311_311022

theorem relationship_among_exponentials (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : 3^b < 3^a ∧ 3^a < 4^a :=
by
  sorry

end relationship_among_exponentials_l311_311022


namespace line_symmetric_eq_l311_311429

-- Definitions:
def line_eq (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

def symmetric_point (p q : ℝ × ℝ) : ℝ × ℝ :=
  (2 * q.1 - p.1, 2 * q.2 - p.2)

-- Problem:
theorem line_symmetric_eq (a b c : ℝ) (p : ℝ × ℝ) :
  line_eq a b c (1, -1) ∧ symmetric_point p (1, -1) = (x, y) ∧ line_eq a b c (2 - x, -2 - y) →
  line_eq a b 8 (2 * x.1 + 3 * x.2 + 8) :=
sorry

end line_symmetric_eq_l311_311429


namespace new_computer_lasts_l311_311567

theorem new_computer_lasts (x : ℕ) 
  (h1 : 600 = 400 + 200)
  (h2 : ∀ y : ℕ, (2 * 200 = 400) → (2 * 3 = 6) → y = 6)
  (h3 : 200 = 600 - 400) :
  x = 6 :=
by
  sorry

end new_computer_lasts_l311_311567


namespace checkerboard_cover_question_l311_311364

noncomputable def checkerboard_squares_covered_by_disc : ℕ :=
  let checkerboard_dim := 8 in
  let square_side := 1 in
  let disc_diameter := Real.sqrt 2 in
  let disc_radius := disc_diameter / 2 in
  let center_distance := disc_radius in
  if center_distance < square_side then 1 else 0

theorem checkerboard_cover_question :
  checkerboard_squares_covered_by_disc = 1 :=
by sorry

end checkerboard_cover_question_l311_311364


namespace hexagon_area_ratio_l311_311271

theorem hexagon_area_ratio (x : ℝ) :
  let area_smaller := (3 * Real.sqrt 3 / 2) * x^2
  let area_larger := (3 * Real.sqrt 3 / 2) * (2 * x)^2
  area_smaller / area_larger = 1 / 4 :=
by
  -- Define the areas of the regular hexagons
  let area_smaller := (3 * Real.sqrt 3 / 2) * x^2
  let area_larger := (3 * Real.sqrt 3 / 2) * (2 * x)^2
  
  -- Calculate and compare their ratio
  have h : area_smaller / area_larger = ((3 * Real.sqrt 3 / 2) * x^2) / ((3 * Real.sqrt 3 / 2) * (2 * x)^2), by sorry
  exact h

end hexagon_area_ratio_l311_311271


namespace CarmenBrushLengthIsCorrect_l311_311977

namespace BrushLength

def carlasBrushLengthInInches : ℤ := 12
def conversionRateInCmPerInch : ℝ := 2.5
def lengthMultiplier : ℝ := 1.5

def carmensBrushLengthInCm : ℝ :=
  carlasBrushLengthInInches * lengthMultiplier * conversionRateInCmPerInch

theorem CarmenBrushLengthIsCorrect :
  carmensBrushLengthInCm = 45 := by
  sorry

end BrushLength

end CarmenBrushLengthIsCorrect_l311_311977


namespace sets_bounded_by_n_l311_311939

open Set

variables {n m : ℕ} (F : Fin m → Finset (Fin n))

theorem sets_bounded_by_n (h1 : ∀ i, 1 ≤ i → i ≤ m → F i ⊆ Finset.univ)
                         (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ m → (min ((F i).card - (F j).card) ((F j).card - (F i).card) = 1)) :
  m ≤ n :=
sorry

end sets_bounded_by_n_l311_311939


namespace at_least_one_defective_l311_311900

def box_contains (total defective : ℕ) := total = 24 ∧ defective = 4

def probability_at_least_one_defective : ℚ := 43 / 138

theorem at_least_one_defective (total defective : ℕ) (h : box_contains total defective) :
  @probability_at_least_one_defective = 43 / 138 :=
by
  sorry

end at_least_one_defective_l311_311900


namespace simplify_cube_roots_l311_311239

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311239


namespace complex_addition_result_l311_311585

theorem complex_addition_result (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : a + b * i = (1 - i) * (2 + i)) : a + b = 2 :=
sorry

end complex_addition_result_l311_311585


namespace sum_of_divisors_of_29_l311_311710

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311710


namespace cube_root_simplification_l311_311247

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311247


namespace angle_CRT_in_isosceles_triangle_l311_311557

theorem angle_CRT_in_isosceles_triangle
  (C A T R : Type)
  [has_inner_product C A T]
  [has_inner_product C T A]
  [has_inner_product T A C]
  [has_inner_product T R]
  (angle_ACT_eq_ATC : inner_angle C A T = inner_angle C T A)
  (angle_CAT_36 : inner_angle C A T = 36)
  (TR_bisects_ATC : ∃ (R : A), inner_angle T R T = (inner_angle C T A) / 2)
  : inner_angle C R T = 72 :=
sorry

end angle_CRT_in_isosceles_triangle_l311_311557


namespace find_length_BC_l311_311173

-- Step 1: Define the rectangle and the points with their conditions.
variables (A B C D E M : Type) [linear_ordered_field A]
variables (ED CD : A) (h1 : ED = 16) (h2 : CD = 12)

-- Step 2: Define the conditions for points E and M.
variables (AB BM AE EM : A) (h3 : AB = BM) (h4 : AE = EM)
variables (h5 : M ∈ between E C) (h6 : E ∈ between A D)

-- Step 3: State the theorem to find the length of BC.
theorem find_length_BC : BC = 20 :=
by
  have CE := sqrt (CD^2 + ED^2)
  have hCE : CE = 20 := by sorry
  exact hCE

end find_length_BC_l311_311173


namespace parallelogram_area_proof_l311_311625

noncomputable def parallelogram_area : ℝ :=
  let angle_rad := (150 * real.pi / 180)  -- converting degrees to radians
  let a := 10                              -- length of one side
  let b := 20                              -- length of another side
  let height := a * real.sqrt(3) / 2       -- height from 30-60-90 triangle properties
  b * height

theorem parallelogram_area_proof : parallelogram_area = 100 * real.sqrt(3) := by
  sorry

end parallelogram_area_proof_l311_311625


namespace avg_marks_l311_311333

theorem avg_marks (P C M : ℕ) (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  -- Proof goes here
  sorry

end avg_marks_l311_311333


namespace least_four_digit_integer_l311_311702

def is_valid_number (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧
  (∀ (d : ℕ), d ∣ n → d < 10) ∧
  (∀ (d : ℕ), (d ∈ [1,2,3,4,6,7,8,9]) → d ∣ n → d ∉ digits n) ∧
  (∀ (d1 d2 : ℕ), d1 ∈ digits n → d2 ∈ digits n → d1 ≠ d2) ∧
  (5 ∈ digits n)

theorem least_four_digit_integer : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → n ≤ m) :=
sorry

end least_four_digit_integer_l311_311702


namespace find_e_l311_311094

theorem find_e (a e : ℕ) (h1: a = 105) (h2: a ^ 3 = 21 * 25 * 45 * e) : e = 49 :=
sorry

end find_e_l311_311094


namespace triangle_area_is_32_5_l311_311310

-- Define points A, B, and C
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (1, 7)
def C : ℝ × ℝ := (4, -1)

-- Calculate the area directly using the determinant method for the area of a triangle given by coordinates
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (
    A.1 * (B.2 - C.2) +
    B.1 * (C.2 - A.2) +
    C.1 * (A.2 - B.2)
  )

-- Define the statement to be proved
theorem triangle_area_is_32_5 : area_triangle A B C = 32.5 := 
  by
  -- proof to be filled in
  sorry

end triangle_area_is_32_5_l311_311310


namespace number_of_people_today_l311_311902

theorem number_of_people_today (x : ℕ) 
  (h1 : 312 % x = 0) -- 312 can be divided by x without remainder
  (h2 : 312 % (x + 2) = 0) -- 312 can be divided by x+2 without remainder
  (h3 : 312 / (x + 2) = 312 / x - 1) -- each person receives 1 marble less if 2 people joined the group
  : x = 24 :=
by 
  have sub_eq : 312 / x - 1 = 312 / (x + 2) := by simp [h3]
  have h_x_greater_than_two : x > 2 := by
    by_contradiction
    cases x; simp at h1; linarith
    cases x; simp at h1; linarith
    cases x; simp at h1; linarith
  -- Express equation in standard form x^2 + 2x = 624
  rw [←sub_eq_zero_of_eq (h2 : 312 / (x+2) = 312 / x - 1)] at h3
  linarith

end number_of_people_today_l311_311902


namespace necessary_but_not_sufficient_l311_311048

variable {I : Set ℝ} (f : ℝ → ℝ) (M : ℝ)

theorem necessary_but_not_sufficient :
  (∀ x ∈ I, f x ≤ M) ↔
  (∀ x ∈ I, f x ≤ M ∧ (∃ x ∈ I, f x = M) → M = M ∧ ∃ x ∈ I, f x = M) :=
by
  sorry

end necessary_but_not_sufficient_l311_311048


namespace simplify_cuberoot_product_l311_311232

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311232


namespace simplify_cube_roots_l311_311243

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311243


namespace sum_of_divisors_prime_29_l311_311729

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311729


namespace matrix_vector_multiplication_l311_311145

variables (M : Matrix (Fin 2) (Fin 2) ℝ) (u z : Vector ℝ 2)

-- Conditions
def cond1 : Prop := M.mulVec u = ![4, -1]
def cond2 : Prop := M.mulVec z = ![1, 6]

theorem matrix_vector_multiplication (h1 : cond1 M u) (h2 : cond2 M z) :
  M.mulVec (2 • u - 4 • z) = ![4, -26] :=
by
  sorry

end matrix_vector_multiplication_l311_311145


namespace food_per_puppy_meal_l311_311133

-- Definitions for conditions
def mom_daily_food : ℝ := 1.5 * 3
def num_puppies : ℕ := 5
def total_food_needed : ℝ := 57
def num_days : ℕ := 6

-- Total food for the mom dog over the given period
def total_mom_food : ℝ := mom_daily_food * num_days

-- Total food for the puppies over the given period
def total_puppy_food : ℝ := total_food_needed - total_mom_food

-- Total number of puppy meals over the given period
def total_puppy_meals : ℕ := (num_puppies * 2) * num_days

theorem food_per_puppy_meal :
  total_puppy_food / total_puppy_meals = 0.5 :=
  sorry

end food_per_puppy_meal_l311_311133


namespace sum_of_divisors_of_29_l311_311748

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311748


namespace part_I_part_II_l311_311449

def f (x : ℝ) : ℝ := abs (2 * x + 3) - abs (2 * x - 1)

theorem part_I (x : ℝ) : f x < 2 ↔ x ∈ set.Ioo (neg_infty) 0 :=
sorry

theorem part_II (a : ℝ) : (∃ x : ℝ, f x > abs (3 * a - 2)) ↔ a ∈ set.Ioo (-2 / 3) 2 :=
sorry

end part_I_part_II_l311_311449


namespace square_side_length_l311_311268

theorem square_side_length (d : ℝ) (sqrt_2_ne_zero : sqrt 2 ≠ 0) (h : d = 2 * sqrt 2) : 
  ∃ (s : ℝ), s = 2 ∧ d = s * sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [mul_comm, ←mul_assoc, eq_comm, mul_right_comm, mul_div_cancel, h, mul_comm]
    · exact sqrt_2_ne_zero
  sorry

end square_side_length_l311_311268


namespace range_of_a_l311_311158

open Set Real

-- Defining the universal set
def U := ℝ

-- Defining the solution set A for the inequality |x-1| + a - 1 >= 0
def A (a : ℝ) : Set ℝ := {x | |x - 1| + a - 1 ≥ 0}

-- Defining the set B based on the trigonometric equation
def B : Set ℝ := {x | sin (π * x - π / 3) + sqrt 3 * cos (π * x - π / 3) = 0}

-- Proving the range of a such that (C_U A) ∩ B has exactly three elements
theorem range_of_a (a : ℝ) : (Cardinal.mk ((U \ A a) ∩ B) = 3) ↔ -1 < a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l311_311158


namespace largest_tile_side_length_l311_311181

theorem largest_tile_side_length (w h : ℕ) (hw : w = 17) (hh : h = 23) : Nat.gcd w h = 1 := by
  -- Proof goes here
  sorry

end largest_tile_side_length_l311_311181


namespace solution_set_of_inequality_l311_311682

theorem solution_set_of_inequality (x : ℝ) : x^2 > x ↔ x < 0 ∨ 1 < x := 
by
  sorry

end solution_set_of_inequality_l311_311682


namespace power_mean_inequality_l311_311641

noncomputable def power_mean (α : ℝ) (x : ℕ → ℝ) (n : ℕ) : ℝ :=
if α = 0 then (∏ i in finset.range n, x i) ^ (1 / n) else (∑ i in finset.range n, (x i) ^ α / n) ^ (1 / α)

theorem power_mean_inequality {α β : ℝ} {n : ℕ} (x : ℕ → ℝ) (hαβ : α < β) (hx : ∀ i, i < n → 0 < x i) :
  power_mean α x n ≤ power_mean β x n ∧ (power_mean α x n = power_mean β x n ↔ ∀ i j, i < n → j < n → x i = x j) :=
by
  sorry

end power_mean_inequality_l311_311641


namespace volume_ratio_l311_311400

theorem volume_ratio (V_L V_S : ℝ) (h1 : V_L = 125 * V_S) : V_S = V_L / 125 :=
by
suffices : 125 * V_S = V_L, from eq_div_of_mul_eq (ne_of_gt (by norm_num : (125 : ℝ) > 0)) this,
assumption

end volume_ratio_l311_311400


namespace solve_for_x_l311_311092

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 := 
by sorry

end solve_for_x_l311_311092


namespace range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l311_311046

def quadratic_has_two_distinct_positive_roots (m : ℝ) : Prop :=
  4 * m^2 - 4 * (m + 2) > 0 ∧ -2 * m > 0 ∧ m + 2 > 0

def hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 3 < 0 ∧ 1 - 2 * m > 0

theorem range_of_m_given_q (m : ℝ) :
  hyperbola_with_foci_on_y_axis m → m < -3 :=
by
  sorry

theorem range_of_m_given_p_or_q_and_not_p_and_q (m : ℝ) :
  (quadratic_has_two_distinct_positive_roots m ∨ hyperbola_with_foci_on_y_axis m) ∧
  ¬(quadratic_has_two_distinct_positive_roots m ∧ hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
by
  sorry

end range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l311_311046


namespace sum_of_divisors_prime_29_l311_311722

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311722


namespace point_p_final_position_l311_311172

theorem point_p_final_position :
  let P_start := -2
  let P_right := P_start + 5
  let P_final := P_right - 4
  P_final = -1 :=
by
  sorry

end point_p_final_position_l311_311172


namespace cost_split_evenly_l311_311570

noncomputable def total_cost (num_cupcakes : ℕ) (cost_per_cupcake : ℚ) : ℚ :=
  num_cupcakes * cost_per_cupcake

noncomputable def cost_per_person (total_cost : ℚ) : ℚ :=
  total_cost / 2

theorem cost_split_evenly (num_cupcakes : ℕ) (cost_per_cupcake : ℚ) (total_cost : ℚ) :
  num_cupcakes = 12 →
  cost_per_cupcake = 3/2 →
  total_cost = total_cost num_cupcakes cost_per_cupcake →
  cost_per_person total_cost = 9 := 
by
  intros h1 h2 h3
  sorry

end cost_split_evenly_l311_311570


namespace sum_of_divisors_of_29_l311_311760

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311760


namespace distance_AB_l311_311175

noncomputable theory 
open_locale classical

def meets_at_midpoint (S t : ℝ) :=
  ∃ (vA vB vC : ℝ),
    let vA' := vA / 2,
        vB' := vB / 2 in
      vA * t = S ∧
      vB * 2 * t = S ∧
      vA' * 2 * (t / 2) + vC * (2 * t / 2) = 2010 ∧
      (vA / 2) * t = S / 2

theorem distance_AB : 
  let S := 5360 in
  ∃ t : ℝ, meets_at_midpoint S t :=
begin
  sorry
end

end distance_AB_l311_311175


namespace all_pairs_parallel_l311_311485

-- Definitions of the vector pairs
def vec_a1 := (1:ℝ, -2, 1)
def vec_b1 := (-1:ℝ, 2, -1)

def vec_a2 := (8:ℝ, 4, 0)
def vec_b2 := (2:ℝ, 1, 0)

def vec_a3 := (1:ℝ, 0, -1)
def vec_b3 := (-3:ℝ, 0, 3)

def vec_a4 := (-4/3:ℝ, 1, -1)
def vec_b4 := (4:ℝ, -3, 3)

-- Predicate for parallel vectors
def is_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)

-- Lean statement to prove all vector pairs are parallel
theorem all_pairs_parallel :
  is_parallel vec_a1 vec_b1 ∧
  is_parallel vec_a2 vec_b2 ∧
  is_parallel vec_a3 vec_b3 ∧
  is_parallel vec_a4 vec_b4 :=
by
  sorry

end all_pairs_parallel_l311_311485


namespace find_x_l311_311457

noncomputable def sequence (x : ℕ) : ℕ → ℕ
| 1       := 1
| 2       := x
| (n + 1) := |sequence n - sequence (n - 1)|

theorem find_x (x : ℕ) (h : x ∈ [6, 7]) :
  (∃ (s : ℕ → ℕ), s 1 = 1 ∧ s 2 = x ∧ ∀ n ≥ 2, s (n + 1) = |s n - s (n - 1)| ∧
    (∃ t, t ≤ 100 ∧ ∑ i in finset.range 100, if (s i = 0) then 1 else 0 = 30)) :=
begin 
  sorry 
end

end find_x_l311_311457


namespace exponent_sum_l311_311084

variables (a : ℝ) (m n : ℝ)

theorem exponent_sum (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_sum_l311_311084


namespace proof_problem_l311_311124

variable (A B C a b c h : ℝ)

-- Given conditions in the problem
def conditions (A B C a b c h : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = π ∧
  c - a = h ∧
  h = c - a

-- The theorem that needs to be proved
theorem proof_problem (A B C a b c h : ℝ) (h_cond : conditions A B C a b c h) :
  sin ((C - A) / 2) + cos ((C + A) / 2) = 1 :=
sorry

end proof_problem_l311_311124


namespace find_intervals_and_range_of_m_l311_311490

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + 1/x + 2 * x

theorem find_intervals_and_range_of_m (a : ℝ) (h : a = 1) :
  (∀ x > 0, (f 1 x) → 
    ((0 < x ∧ x < 1/2) → (deriv (f 1) x < 0)) ∧
    ((x > 1/2) → (deriv (f 1) x > 0))) ∧
  (∀ x > 0, f 1 x ≥ 2 * x + m / x → m ≤ 1 - 1/Real.exp 1) :=
sorry

end find_intervals_and_range_of_m_l311_311490


namespace sum_of_divisors_of_29_l311_311739

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311739


namespace sum_of_divisors_of_29_l311_311773

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311773


namespace dog_food_duration_l311_311959

-- Definitions for the given conditions
def number_of_dogs : ℕ := 4
def meals_per_day : ℕ := 2
def grams_per_meal : ℕ := 250
def sacks_of_food : ℕ := 2
def kilograms_per_sack : ℝ := 50
def grams_per_kilogram : ℝ := 1000

-- Lean statement to prove the correct answer
theorem dog_food_duration : 
  ((number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) * sacks_of_food * kilograms_per_sack) / 
  (number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) = 50 :=
by 
  simp only [number_of_dogs, meals_per_day, grams_per_meal, sacks_of_food, kilograms_per_sack, grams_per_kilogram]
  norm_num
  sorry

end dog_food_duration_l311_311959


namespace sarah_bought_3_bottle_caps_l311_311646

theorem sarah_bought_3_bottle_caps
  (orig_caps : ℕ)
  (new_caps : ℕ)
  (h_orig_caps : orig_caps = 26)
  (h_new_caps : new_caps = 29) :
  new_caps - orig_caps = 3 :=
by
  sorry

end sarah_bought_3_bottle_caps_l311_311646


namespace Peter_mowing_time_l311_311164

/-- Definitions and variables for the time taken to mow the yard -/
def Nancy_time := 3
def combined_time := 1.71428571429

/-- The rate of mowing is 1 per the total time taken -/
def Nancy_rate := 1 / Nancy_time
def combined_rate := 1 / combined_time

/-- We seek to find Peter's rate such that 1/Peter_time (the rate of mowing for Peter) is equal to Peter_rate -/
def Peter_time := 4
def Peter_rate := 1 / Peter_time

theorem Peter_mowing_time :
  Nancy_rate + Peter_rate = combined_rate := 
by
  sorry

end Peter_mowing_time_l311_311164


namespace x_intercept_of_rotated_line_is_20_l311_311597

theorem x_intercept_of_rotated_line_is_20 : 
  (∃ l: Real, (∀ x y: Real, (4 * x - 3 * y + 20 = 0) → True) ∧ 
                rotated_line_intercept l (10, 10) 30 = 20) :=
sorry

noncomputable def rotated_line_intercept (l : Real) (p : Real × Real) (angle : Real) : Real :=
sorry

end x_intercept_of_rotated_line_is_20_l311_311597


namespace sum_of_divisors_29_l311_311866

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311866


namespace part_1_monotonic_intervals_part_2_range_of_a_l311_311586

-- Definitions
def f (x : ℝ) (a : ℝ) (b : ℝ) := 2 * x ^ 2 + b * x - a * Real.log x

theorem part_1_monotonic_intervals :
  (∀ a b : ℝ, a = 5 ∧ b = -1 → 
    ( ∃ (I_dec I_inc : Set ℝ), 
      I_dec = {x : ℝ | 0 < x ∧ x < 5 / 4} ∧ 
      I_inc = {x : ℝ | x > 5 / 4} ∧ 
      (∀ x ∈ I_dec, deriv (λ x, f x a b) x < 0) ∧
      (∀ x ∈ I_inc, deriv (λ x, f x a b) x > 0))) := sorry

theorem part_2_range_of_a :
  ∀ a : ℝ, (a > 2) →
    (∀ b : ℝ, b ∈ (@Set.Icc ℝ _ (-3) (-2)) →
      (∃ x : ℝ, x ∈ @Set.Ioo ℝ _ 1 (Real.exp 2) ∧ f x a b < 0)) := sorry

end part_1_monotonic_intervals_part_2_range_of_a_l311_311586


namespace max_friends_l311_311262

theorem max_friends (total_candies : ℕ) (candies_per_friend : ℕ) (h_candies : total_candies = 45) (h_per_friend : candies_per_friend = 5) : total_candies / candies_per_friend = 9 :=
by {
  rw [h_candies, h_per_friend],
  norm_num,
  -- This concludes that (45 / 5) = 9
  sorry
}

end max_friends_l311_311262


namespace volume_of_cone_formed_by_sector_l311_311372

theorem volume_of_cone_formed_by_sector :
  let radius := 6
  let sector_fraction := (5:ℝ) / 6
  let circumference := 2 * Real.pi * radius
  let cone_base_circumference := sector_fraction * circumference
  let cone_base_radius := cone_base_circumference / (2 * Real.pi)
  let slant_height := radius
  let cone_height := Real.sqrt (slant_height^2 - cone_base_radius^2)
  let volume := (1:ℝ) / 3 * Real.pi * (cone_base_radius^2) * cone_height
  volume = 25 / 3 * Real.pi * Real.sqrt 11 :=
by sorry

end volume_of_cone_formed_by_sector_l311_311372


namespace sum_of_divisors_of_prime_29_l311_311834

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311834


namespace cos_alpha_value_l311_311514

-- Given conditions
variables (α : ℝ) (h1 : sin (α - π / 6) = 3 / 5) (h2 : α ∈ Ioo 0 (π / 2))

-- Target: To prove the value of cos α given the conditions
theorem cos_alpha_value :
  cos α = (4 * real.sqrt 3 - 3) / 10 :=
sorry

end cos_alpha_value_l311_311514


namespace parallelogram_area_l311_311633

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l311_311633


namespace rectangle_problem_l311_311032

theorem rectangle_problem (x : ℝ) (h1 : 4 * x = l) (h2 : x + 7 = w) (h3 : l * w = 2 * (2 * l + 2 * w)) : x = 1 := 
by {
  sorry
}

end rectangle_problem_l311_311032


namespace c_work_rate_l311_311327

variable {W : ℝ} -- Denoting the work by W
variable {a_rate : ℝ} -- Work rate of a
variable {b_rate : ℝ} -- Work rate of b
variable {c_rate : ℝ} -- Work rate of c
variable {combined_rate : ℝ} -- Combined work rate of a, b, and c

theorem c_work_rate (W a_rate b_rate c_rate combined_rate : ℝ)
  (h1 : a_rate = W / 12)
  (h2 : b_rate = W / 24)
  (h3 : combined_rate = W / 4)
  (h4 : combined_rate = a_rate + b_rate + c_rate) :
  c_rate = W / 4.5 :=
by
  sorry

end c_work_rate_l311_311327


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311621

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311621


namespace regular_tetrahedron_distance_ratio_l311_311583

theorem regular_tetrahedron_distance_ratio
  (ABCD : Type)
  [is_regular_tetrahedron ABCD]
  (E : Point)
  (is_centroid_E : is_centroid_of_face E ABC)
  (t T : ℝ)
  (t_def : t = ∑ plane in {DAB, DBC, DCA}, distance_from E plane)
  (T_def : T = distance_from E (vertex D)
               + ∑ line in {AB, BC, CA}, distance_from E line) :
  t / T = (3 * Real.sqrt 2) / 5 :=
sorry

end regular_tetrahedron_distance_ratio_l311_311583


namespace sum_of_divisors_of_29_l311_311855

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311855


namespace sum_of_divisors_of_29_l311_311797

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311797


namespace sprinkles_remaining_l311_311406

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) 
  (h1 : initial_cans = 12) 
  (h2 : remaining_cans = (initial_cans / 2) - 3) : 
  remaining_cans = 3 := 
by
  sorry

end sprinkles_remaining_l311_311406


namespace student_number_in_sample_l311_311980

variable (n : ℕ)
variable (numbers : Finset ℕ)
variable [Fact (¬ numbers ∅)]

theorem student_number_in_sample :
  5 ∈ numbers ∧ 31 ∈ numbers ∧ 44 ∈ numbers ∧ numbers.card = 4 ∧ 
  ((∃ a d : ℕ, d ≠ 0 ∧ numbers = {a, a+d, a+2*d, a+3*d }) ∧
  (∃ b, b ∈ numbers ∧ b = 18)) :=
sorry

end student_number_in_sample_l311_311980


namespace extreme_values_unique_zero_point_l311_311062

noncomputable section

def f (x a : ℝ) := x^2 - 2 * a * Real.log x

def g (x a : ℝ) := 2 * a * x

def h (x a : ℝ) := f x a - g x a

theorem extreme_values (a : ℝ) :
  if a ≤ 0 then ∀ x : ℝ, x > 0 → ( ∀ ε : ℝ, ε > 0 → f (x - ε) a ≤ f x a ∧ f x a ≤ f (x + ε) a ) →
  if a > 0 then
    ( ∃ x : ℝ, x > 0 ∧ f x a = a - a * Real.log a ∧ ∀ y : ℝ, y > 0 → f y a ≥ f x a ) :=
sorry

theorem unique_zero_point (a : ℝ) (ha : a > 0) (h_only_one_zero : ∃ x : ℝ, x > 0 ∧ h x a = 0
  ∧ (∀ y : ℝ, y > 0 → h y a = 0 → y = x)) : a = 1/2 :=
sorry

end extreme_values_unique_zero_point_l311_311062


namespace cube_root_simplification_l311_311248

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311248


namespace sum_of_divisors_29_l311_311833

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311833


namespace sum_of_divisors_of_29_l311_311769

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311769


namespace sum_of_four_distinct_members_is_17_l311_311410

def A := {1, 5, 9, 13, 17, 21, 25, 29}
def all_sums := {x | ∃ (a b c d ∈ A), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ x = a + b + c + d}
def count_sums := all_sums.to_finset.card

theorem sum_of_four_distinct_members_is_17 :
  count_sums = 17 :=
sorry

end sum_of_four_distinct_members_is_17_l311_311410


namespace inverse_function_point_l311_311083

theorem inverse_function_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ ∀ y, (∀ x, y = a^(x-3) + 1) → (2, 3) ∈ {(y, x) | y = a^(x-3) + 1} :=
by
  sorry

end inverse_function_point_l311_311083


namespace sum_of_divisors_of_29_l311_311861

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311861


namespace sum_of_divisors_of_prime_29_l311_311837

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311837


namespace quadratic_solution_interval_l311_311098

theorem quadratic_solution_interval (a : ℝ) :
  (∃! x : ℝ, 0 < x ∧ x < 1 ∧ 2 * a * x^2 - x - 1 = 0) → a > 1 :=
by
  sorry

end quadratic_solution_interval_l311_311098


namespace simplified_expression_correct_l311_311205

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311205


namespace sum_of_valid_n_l311_311011

theorem sum_of_valid_n : 
  let S := {n : ℤ | (∃ x : ℤ, n^2 - 17 * n + 72 = x^2) ∧ 7 % n = 0}
  ∑ n in S, n = 8 :=
by sorry

end sum_of_valid_n_l311_311011


namespace calculate_final_price_l311_311923

theorem calculate_final_price (original_price : ℝ) (first_reduction_pct : ℝ) (second_reduction_pct : ℝ) : 
  original_price = 20 → 
  first_reduction_pct = 0.25 → 
  second_reduction_pct = 0.4 → 
  final_price = 9 := 
by 
  intros h_original_price h_first_reduction h_second_reduction 
  let first_reduction = original_price * first_reduction_pct
  let price_after_first_reduction = original_price - first_reduction
  let second_reduction = price_after_first_reduction * second_reduction_pct
  let final_price = price_after_first_reduction - second_reduction
  have : final_price = 9 := by {
    subst h_original_price
    subst h_first_reduction
    subst h_second_reduction
    sorry
  }
  exact this

end calculate_final_price_l311_311923


namespace solve_system_of_inequalities_l311_311652

theorem solve_system_of_inequalities (x : ℝ) :
  4*x^2 - 27*x + 18 > 0 ∧ x^2 + 4*x + 4 > 0 ↔ (x < 3/4 ∨ x > 6) ∧ x ≠ -2 :=
by
  sorry

end solve_system_of_inequalities_l311_311652


namespace cube_root_simplification_l311_311246

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311246


namespace earnings_correct_l311_311974

def phonePrice : Nat := 11
def laptopPrice : Nat := 15
def computerPrice : Nat := 18
def tabletPrice : Nat := 12
def smartwatchPrice : Nat := 8

def phoneRepairs : Nat := 9
def laptopRepairs : Nat := 5
def computerRepairs : Nat := 4
def tabletRepairs : Nat := 6
def smartwatchRepairs : Nat := 8

def totalEarnings : Nat := 
  phoneRepairs * phonePrice + 
  laptopRepairs * laptopPrice + 
  computerRepairs * computerPrice + 
  tabletRepairs * tabletPrice + 
  smartwatchRepairs * smartwatchPrice

theorem earnings_correct : totalEarnings = 382 := by
  sorry

end earnings_correct_l311_311974


namespace pipe_b_faster_than_pipe_a_l311_311177

theorem pipe_b_faster_than_pipe_a :
  (PipeAFillRate: ℝ) (TimeTogether: ℝ) (time_factor: ℝ) (x: ℝ) 
  (hA: PipeAFillRate = 1 / 24)
  (hTogether: TimeTogether = 3)
  (hCombined: TimeTogether * (PipeAFillRate + time_factor * PipeAFillRate) = 1)
  (hx: time_factor = x) :
  x = 7 := 
sorry

end pipe_b_faster_than_pipe_a_l311_311177


namespace four_digit_multiple_of_3_l311_311373

theorem four_digit_multiple_of_3 (d : ℕ) (hd : d ∈ {0, 3, 6, 9}) :
  ∃ k : ℕ, 258 * 10 + d = 3 * k :=
by 
  sorry

end four_digit_multiple_of_3_l311_311373


namespace find_a_l311_311099

section math_proof
variable (a b : ℝ)
variable (M N : ℝ × ℝ)
variable (p : ℝ)

-- Conditions
def line_eq (a : ℝ) : ℝ → ℝ := λ x, a * x + 1
def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + a * x + b * y - 4

axiom intersect_points_symmetric (a b : ℝ) (M N : ℝ × ℝ) :
  (line_eq a M.1 = M.2) ∧ (line_eq a N.1 = N.2) ∧ 
  (circle_eq M.1 M.2 = p) ∧ (circle_eq N.1 N.2 = p) ∧ 
  (M.1 = N.2) ∧ (M.2 = N.1)

-- To Prove
theorem find_a (a b : ℝ) (M N : ℝ × ℝ) (p : ℝ) 
  (h_sym : intersect_points_symmetric a b M N) : 
  a = -1 := 
begin
  sorry
end
end math_proof

end find_a_l311_311099


namespace part1_part2_l311_311492

noncomputable def f (x a : ℝ) : ℝ := x - a - Real.log x

theorem part1 :
  (∀ x > 0, f x a ≥ 0) → a ≤ 1 :=
by sorry

theorem part2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  let f := f x 1 in
  (x1 * Real.log x1 - x1 * Real.log x2 > x1 - x2) :=
by sorry

end part1_part2_l311_311492


namespace divisible_by_12_l311_311442

theorem divisible_by_12 (n : ℕ) (h1 : (5140 + n) % 4 = 0) (h2 : (5 + 1 + 4 + n) % 3 = 0) : n = 8 :=
by
  sorry

end divisible_by_12_l311_311442


namespace proj_on_line_is_constant_l311_311933

-- Define a vector on the line y = 3x - 2
def vec_on_line (a : ℝ) : ℝ × ℝ := (a, 3 * a - 2)

-- Define the vector w
def w (c d : ℝ) : ℝ × ℝ := (c, d)

-- Define the projection
def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_prod := v.1 * w.1 + v.2 * w.2
  let w_norm_sq := w.1 * w.1 + w.2 * w.2
  (dot_prod / w_norm_sq * w.1, dot_prod / w_norm_sq * w.2)

-- Define p: the projection result should be a constant vector p
def p : ℝ × ℝ := (3 / 5, -1 / 5)

-- The theorem to prove
theorem proj_on_line_is_constant (c d : ℝ) (a : ℝ) 
  (h : c + 3 * d = 0) :
  proj (vec_on_line a) (w c d) = p :=
sorry

end proj_on_line_is_constant_l311_311933


namespace problemProof_l311_311993

-- Define a monic polynomial of degree 2
def isMonicPolynomialOfDegree2 (f : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, f = λ x, x^2 + b * x + c

-- Problem: Determine the monic polynomial f(x) with degree 2 such that f(0) = 10 and f(1) = 14.
def solutionP1 : Prop :=
  ∃ f : ℝ → ℝ,
    isMonicPolynomialOfDegree2 f ∧
    f 0 = 10 ∧
    f 1 = 14 ∧
    (∀ x, f x = x^2 + 3 * x + 10)

-- Prove that the solution is given by f(x) = x^2 + 3x + 10.
theorem problemProof : solutionP1 := 
  sorry

end problemProof_l311_311993


namespace median_eq_seventyone_l311_311118

def list_of_numbers : List ℕ :=
  List.join (List.range' 1 100.erase(0))

noncomputable def median (l : List ℕ) : Real :=
if h : l.length % 2 = 1 then l.liftNth h
else (l.liftNthAsList (Nat.div l.length 2) + l.liftNthAsList (Nat.div l.length 2 - 1)) / 2

theorem median_eq_seventyone {l : List ℕ} (h : l = list_of_numbers) :
  median l = 71 := 
begin
  rw h,
  sorry
end

end median_eq_seventyone_l311_311118


namespace solve_system_of_equations_l311_311651

theorem solve_system_of_equations (x y z : ℝ) 
  (h1 : 1 / x = y + z) 
  (h2 : 1 / y = z + x) 
  (h3 : 1 / z = x + y) : 
  (x = y ∧ y = z ∧ (x = sqrt 2 / 2 ∨ x = - sqrt 2 / 2)) := by
sorry

end solve_system_of_equations_l311_311651


namespace sum_of_divisors_of_29_l311_311792

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311792


namespace quadratic_function_vertex_form_l311_311689

theorem quadratic_function_vertex_form :
  ∃ f : ℝ → ℝ, (∀ x, f x = (x - 2)^2 - 2) ∧ (f 0 = 2) ∧ (∀ x, f x = a * (x - 2)^2 - 2 → a = 1) := by
  sorry

end quadratic_function_vertex_form_l311_311689


namespace find_expression_l311_311500

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, (i+1) * a (i+1)) = (n+1) * (n+2)

theorem find_expression :
  ∃ a : ℕ → ℝ,
    sequence a ∧
    (∀ n : ℕ, a n = if n = 1 then 6 else 2 + 2 / n) :=
by
  sorry

end find_expression_l311_311500


namespace sum_of_divisors_of_29_l311_311796

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311796


namespace inverse_proportion_value_of_m_l311_311088

theorem inverse_proportion_value_of_m (m : ℤ) (x : ℝ) (y : ℝ) : 
  y = (m - 2) * x ^ (m^2 - 5) → (m = -2) := 
by
  sorry

end inverse_proportion_value_of_m_l311_311088


namespace simplify_sqrt_l311_311189

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311189


namespace angle_DPO_l311_311561

theorem angle_DPO (DOG : Triangle)
  (hne1 : DOG.∠'.DGO = DOG.∠'.DOG)
  (hne2 : DOG.∠'.DOG = 48)
  (OP_bisects_DOG : DOG.segment_OP.bisects DOG.∠'.DOG) :
  DOG.∠'.DPO = 24 :=
sorry

end angle_DPO_l311_311561


namespace harmonic_series_inequality_l311_311698

theorem harmonic_series_inequality (k : ℕ) (h : k > 1) 
  (inductive_inequality : 1 + ∑ i in finset.range k, 1/(2^i - 1) < k) : 
  ∑ i in finset.range (k + 1), 1/(2^i - 1) = ∑ i in finset.range k, 1/(2^i - 1) + ∑ i in finset.range (2^k), 1/(2^i - 1) := 
sorry

end harmonic_series_inequality_l311_311698


namespace somu_age_problem_l311_311265

-- Define the conditions and the problem
variables (S F Y : ℕ)
variables (h1 : S = 10) (h2 : S = (1 / 3) * F)

-- The goal is to prove that Y = 5
theorem somu_age_problem : Y = 5 :=
by
  -- Assuming the conditions
  have h1 : S = 10 := sorry,
  have h2 : S = (1 / 3) * F := sorry,
  -- Your proof goes here
  sorry

end somu_age_problem_l311_311265


namespace club_truncator_probability_l311_311405

theorem club_truncator_probability:
  let n := 5 in
  let total_outcomes := 3^n in
  let favorable_outcomes := (Nat.choose 5 3) * (Nat.choose 2 2) + 
                            (Nat.choose 5 4) * (Nat.choose 1 1) + 
                            (Nat.choose 5 5) * (Nat.choose 0 0) in
  let probability := favorable_outcomes / total_outcomes in
  let p := favorable_outcomes in
  let q := total_outcomes in
  Nat.coprime p q ∧ p + q = 259 :=
by 
  sorry

end club_truncator_probability_l311_311405


namespace min_triangles_to_divide_square_l311_311540

def square := { p : ℝ × ℝ // 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

def point : Type := square

def triangle (p1 p2 p3 : point) : Type := 
  { p : point // is_in_triangle p p1 p2 p3 }

def contains_point (t : triangle) (p : point) := 
  p ∈ t  

theorem min_triangles_to_divide_square (K : ℕ) (h : K > 2) (points : finset point) (h_card : points.card = K) :
  ∃ (triangles : finset (triangle)), 
    triangles.card = K + 1 ∧ 
    ∀ p ∈ points, ∃ t ∈ triangles, contains_point t p := 
sorry

def is_in_triangle (p : point) (p1 p2 p3 : point) : Prop := sorry

end min_triangles_to_divide_square_l311_311540


namespace negation_of_statement_equivalence_l311_311706

-- Definitions of the math club and enjoyment of puzzles
def member_of_math_club (x : Type) : Prop := sorry
def enjoys_puzzles (x : Type) : Prop := sorry

-- Original statement: All members of the math club enjoy puzzles
def original_statement : Prop :=
∀ x, member_of_math_club x → enjoys_puzzles x

-- Negation of the original statement
def negated_statement : Prop :=
∃ x, member_of_math_club x ∧ ¬ enjoys_puzzles x

-- Proof problem statement
theorem negation_of_statement_equivalence :
  ¬ original_statement ↔ negated_statement :=
sorry

end negation_of_statement_equivalence_l311_311706


namespace sum_of_divisors_of_prime_29_l311_311838

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311838


namespace actual_travel_time_l311_311663

noncomputable def distance : ℕ := 360
noncomputable def scheduled_time : ℕ := 9
noncomputable def speed_increase : ℕ := 5

theorem actual_travel_time (d : ℕ) (t_sched : ℕ) (Δv : ℕ) : 
  (d = distance) ∧ (t_sched = scheduled_time) ∧ (Δv = speed_increase) → 
  t_sched + Δv = 8 :=
by
  sorry

end actual_travel_time_l311_311663


namespace magnitude_of_cross_product_l311_311477

theorem magnitude_of_cross_product :
  let m := (1 : ℝ, 0 : ℝ)
  let n := (-1 : ℝ, Real.sqrt 3)
  |m.1 * n.2 - m.2 * n.1| = Real.sqrt 3
:= sorry

end magnitude_of_cross_product_l311_311477


namespace _l311_311111

noncomputable def hexagon_theorem 
  (A D B E C F : Point) 
  (angle_A angle_D angle_C angle_F angle_E angle_B : Angle) 
  (AB DE BC EF CD FA AD BE CF : Segment)
  (h1 : angle_A = 3 * angle_D)
  (h2 : angle_C = 3 * angle_F)
  (h3 : angle_E = 3 * angle_B) 
  (h4 : AB = DE) 
  (h5 : BC = EF) 
  (h6 : CD = FA) : 
  Concurrent AD BE CF := by
  sorry

end _l311_311111


namespace sum_of_divisors_of_29_l311_311816

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311816


namespace volleyball_tournament_l311_311544

theorem volleyball_tournament (n m : ℕ) (h : n = m) :
  n = m := 
by
  sorry

end volleyball_tournament_l311_311544


namespace brady_average_hours_per_month_l311_311967

noncomputable def average_hours_per_month (hours_april : ℕ) (hours_june : ℕ) (hours_september : ℕ) : ℕ :=
  (hours_april + hours_june + hours_september) / 3

theorem brady_average_hours_per_month :
  let days := 30 in
  let hours_april := 6 * days in
  let hours_june := 5 * days in
  let hours_september := 8 * days in
  average_hours_per_month hours_april hours_june hours_september = 190 :=
begin
  sorry
end

end brady_average_hours_per_month_l311_311967


namespace sum_of_divisors_of_29_l311_311848

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311848


namespace calculation_correct_l311_311969

theorem calculation_correct : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end calculation_correct_l311_311969


namespace sum_of_divisors_of_29_l311_311745

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311745


namespace probability_of_snow_at_most_3_days_l311_311285

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ := 
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def probability_at_most_3_days (p : ℝ) (n : ℕ) : ℝ :=
  (binomial_probability n 0 p) + (binomial_probability n 1 p) + (binomial_probability n 2 p) + (binomial_probability n 3 p)

theorem probability_of_snow_at_most_3_days : 
  let p : ℝ := 1 / 5,
      n : ℕ := 31 in
  abs (probability_at_most_3_days p n - 0.342) < 0.001 := 
sorry

end probability_of_snow_at_most_3_days_l311_311285


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311619

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311619


namespace smallest_prime_divisor_of_sum_l311_311707

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

theorem smallest_prime_divisor_of_sum (h1 : is_odd (3^20))
                                    (h2 : is_odd (11^14))
                                    (h3 : ∀ a b : ℕ, is_odd a → is_odd b → ∃ k, a + b = 2 * k) :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^20 + 11^14) :=
sorry

end smallest_prime_divisor_of_sum_l311_311707


namespace coloring_count_l311_311420

-- Definitions for problem conditions
def colors := {0, 1, 2}  -- Representing red, white, blue as 0, 1, 2

def valid_coloring (coloring : Fin 9 → colors) (edges : List (Fin 9 × Fin 9)) : Prop :=
  ∀ (e : Fin 9 × Fin 9), e ∈ edges → coloring e.1 ≠ coloring e.2

-- Defining the specific graph of the problem
def edges : List (Fin 9 × Fin 9) :=
  [(0, 1), (1, 2), (2, 0), -- First triangle
   (3, 4), (4, 5), (5, 3), -- Second triangle
   (6, 7), (7, 8), (8, 6), -- Third triangle
   (2, 3), (1, 7)]         -- Connecting edges between triangles

-- The theorem statement
theorem coloring_count : ∃ n : Nat, n = 54 ∧ ∀ coloring : Fin 9 → colors,
  valid_coloring coloring edges → n = 54 :=
sorry

end coloring_count_l311_311420


namespace unique_solution_abs_eq_l311_311506

theorem unique_solution_abs_eq (x : ℝ) : (|x - 9| = |x + 3| + 2) ↔ x = 2 :=
by
  sorry

end unique_solution_abs_eq_l311_311506


namespace sum_of_divisors_of_29_l311_311803

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311803


namespace proof_problem_l311_311070

def universe_set := {x : ℝ | true}

def set_A := {x : ℝ | -1 < x ∧ x < 3}

def set_B := {x : ℝ | x < 1 ∨ x ≥ 2}

def complement_R (B : set ℝ) := {x : ℝ | ¬ (x < 1 ∨ x ≥ 2)}

def intersection (A B : set ℝ) := {x : ℝ | x ∈ A ∧ x ∈ B}

noncomputable def answer := {x : ℝ | 1 ≤ x ∧ x < 2}

theorem proof_problem :
  intersection set_A (complement_R set_B) = answer :=
sorry

end proof_problem_l311_311070


namespace find_s_l311_311149

-- Define the roots of the quadratic equation
variables (a b n r s : ℝ)

-- Conditions from Vieta's formulas
def condition1 : Prop := a + b = n
def condition2 : Prop := a * b = 3

-- Roots of the second quadratic equation
def condition3 : Prop := (a + 1 / b) * (b + 1 / a) = s

-- The theorem statement
theorem find_s
  (h1 : condition1 a b n)
  (h2 : condition2 a b)
  (h3 : condition3 a b s) :
  s = 16 / 3 :=
by
  sorry

end find_s_l311_311149


namespace calculate_expression_l311_311340

theorem calculate_expression :
  (5 / 19) * ((19 / 5) * (16 / 3) + (14 / 3) * (19 / 5)) = 10 :=
by
  sorry

end calculate_expression_l311_311340


namespace part_a_part_b_l311_311349

-- Define Part (a)
theorem part_a (l : List ℤ) (h : l.length = 52) :
  ∃ (x y : ℤ), x ∈ l ∧ y ∈ l ∧ (x ≠ y) ∧ ((x + y) % 100 = 0 ∨ (x - y) % 100 = 0) :=
begin
  sorry
end

-- Define Part (b)
theorem part_b (l : List ℤ) (h : l.length = 100) :
  ∃ (s : List ℤ), s ≠ [] ∧ s ⊆ l ∧ (s.sum % 100 = 0) :=
begin
  sorry
end

end part_a_part_b_l311_311349


namespace parallelogram_area_l311_311632

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l311_311632


namespace pyramid_edges_equal_sixteen_l311_311300

theorem pyramid_edges_equal_sixteen :
  ∃ n V E F : ℕ, 
    (V = 2 * n) ∧ 
    (E = 3 * n) ∧ 
    (F = n + 2) ∧ 
    (V + E + F = 50) ∧ 
    (2 * n = 16) :=
begin
  sorry
end

end pyramid_edges_equal_sixteen_l311_311300


namespace polyhedron_volume_l311_311117

-- Define the given polygons and their properties
structure Triangle :=
  (a b : ℝ) -- two sides of the triangle
  (right : a = b) -- isosceles right triangle
  (leg_length : a = 2)

structure Square :=
  (side : ℝ)
  (length_2 : side = 2)

structure EquilateralTriangle :=
  (side : ℝ)
  (length : side = 2 * sqrt 2)

def A : Triangle := ⟨2, 2, rfl, rfl⟩
def E : Triangle := ⟨2, 2, rfl, rfl⟩
def F : Triangle := ⟨2, 2, rfl, rfl⟩

def B : Square := ⟨2, rfl⟩
def C : Square := ⟨2, rfl⟩
def D : Square := ⟨2, rfl⟩

def G : EquilateralTriangle := ⟨2 * sqrt 2, rfl⟩

-- Define the volume of the polyhedron
noncomputable def volume_polyhedron : ℝ := 22 * sqrt 2 / 3

-- The theorem to prove
theorem polyhedron_volume : volume_polyhedron = 22 * sqrt 2 / 3 :=
  by sorry

end polyhedron_volume_l311_311117


namespace sum_of_divisors_29_l311_311873

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311873


namespace sum_of_missing_digits_l311_311115

theorem sum_of_missing_digits:
  ∃ (d1 d2 : ℕ), 
    (d1 + 1 + 8 = 17) ∧  -- Condition for the tens column with carryover
    (d1 = 2) ∧
    (7 + d2 + 1 + 1 = 14) ∧  -- Condition for the hundreds column with carryover
    (d2 = 5) ∧
    (d1 + d2 = 7) := 
begin
  use [2, 5],
  simp,
sorries

end sum_of_missing_digits_l311_311115


namespace length_of_PQ_l311_311141

-- Define point structure
structure Point :=
  (x : ℚ)
  (y : ℚ)

-- Define coordinates of R
def R : Point := ⟨10, 8⟩

-- Define lines containing points P and Q
def line1 (P : Point) : Prop := 5 * P.y = 12 * P.x
def line2 (Q : Point) : Prop := 15 * Q.y = 4 * Q.x

-- Define midpoint condition
def is_midpoint (R P Q : Point) : Prop :=
  R.x = (P.x + Q.x) / 2 ∧ R.y = (P.y + Q.y) / 2

-- Define distance function
def distance (P Q : Point) : ℚ := real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- The proof problem statement in Lean 4
theorem length_of_PQ (P Q : Point) 
  (h1 : line1 P) 
  (h2 : line2 Q) 
  (h3 : is_midpoint R P Q) : 
  distance P Q = 2 * real.sqrt 41 :=
sorry

end length_of_PQ_l311_311141


namespace simplify_cube_roots_l311_311242

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311242


namespace sum_of_divisors_of_prime_l311_311784

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311784


namespace initial_savings_correct_l311_311602

-- Define the constants for ticket prices and number of tickets.
def vip_ticket_price : ℕ := 100
def vip_tickets : ℕ := 2
def regular_ticket_price : ℕ := 50
def regular_tickets : ℕ := 3
def leftover_savings : ℕ := 150

-- Define the total cost of tickets.
def total_cost : ℕ := (vip_ticket_price * vip_tickets) + (regular_ticket_price * regular_tickets)

-- Define the initial savings calculation.
def initial_savings : ℕ := total_cost + leftover_savings

-- Theorem stating the initial savings should be $500.
theorem initial_savings_correct : initial_savings = 500 :=
by
  -- Proof steps can be added here.
  sorry

end initial_savings_correct_l311_311602


namespace sum_of_digits_in_9_pow_2023_l311_311971

theorem sum_of_digits_in_9_pow_2023 : 
  let n := 9^2023,
      last_two_digits := n % 100 in
  (last_two_digits / 10) + (last_two_digits % 10) = 11 :=
by
  let n := 9^2023
  let last_two_digits := n % 100
  have h1 : (last_two_digits / 10) = 2 := sorry
  have h2 : (last_two_digits % 10) = 9 := sorry
  have h3 : 2 + 9 = 11 := by norm_num
  exact h3

end sum_of_digits_in_9_pow_2023_l311_311971


namespace sum_of_angles_of_two_right_angled_triangles_l311_311169

-- Define the concept of a right-angled triangle
structure RightAngledTriangle (A B C : Type) :=
(right_angle : ∠ BCA = 90)

-- Define the problem
theorem sum_of_angles_of_two_right_angled_triangles 
  {A B C A1 B1 C1 : Type} 
  (triangle1 : RightAngledTriangle A B C) 
  (triangle2 : RightAngledTriangle A1 B1 C1) : 
  ∠ BCA + ∠ B1C1A1 = 90 :=
sorry

end sum_of_angles_of_two_right_angled_triangles_l311_311169


namespace right_triangle_segment_ratio_l311_311102

theorem right_triangle_segment_ratio {x : ℝ} (hx : x > 0) :
  let AB := 3 * x,
      BC := x,
      AC := real.sqrt ((3 * x)^2 + x^2),
      D := AC / (real.sqrt 10) in
  let AD := (1 / 9) * D
  let CD := D - AD
  (CD / AD = 9) :=
by
  let AB := 3 * x
  let BC := x
  let AC := real.sqrt ((3 * x)^2 + x^2)
  let BD := AC / (real.sqrt 10)
  let AD := (1 / 9) * BD
  let CD := BD - AD
  show (CD / AD = 9)
  sorry

end right_triangle_segment_ratio_l311_311102


namespace sum_of_divisors_of_29_l311_311742

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311742


namespace simplify_cubed_roots_l311_311216

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311216


namespace sum_of_divisors_29_l311_311827

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311827


namespace parallelogram_area_l311_311614

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l311_311614


namespace AKS_correct_l311_311647

noncomputable def prime_factor (n : ℕ) : ℕ := sorry
noncomputable def order_modulo (n r : ℕ) : ℕ := sorry
noncomputable def cyclotomic_irreducible (r p : ℕ) (H : ℕ[X]) : Prop := sorry
noncomputable def satisfies_congruence (n r p : ℕ) (H : ℕ[X]) : Prop := sorry

theorem AKS_correct {n p r : ℕ} (H : ℕ[X]) 
  (h₁ : Nat.Prime n)
  (h₂ : n ≥ 2)
  (h₃ : p = prime_factor n)
  (h₄ : r < p)
  (h₅ : order_modulo n r > 4 * log₂ n)
  (h₆ : cyclotomic_irreducible r p H)
  (h₇ : ∀ a : ℕ, a ≤ 2 * √r * log₂ n → satisfies_congruence n r p H) :
  ∃ k : ℕ, n = p^k := 
sorry

end AKS_correct_l311_311647


namespace sum_theta_mod_2010_l311_311016

def θ (n : ℕ) : ℕ := (0 : ℕ) 

theorem sum_theta_mod_2010 :
  (∑ n in finset.range 2010, n * θ n) % 2010 = 335 :=
sorry

end sum_theta_mod_2010_l311_311016


namespace probability_Bernardo_number_larger_l311_311962

noncomputable def bernardo_and_silvia : Prop :=
∀ (a b c : ℕ), (a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) →
                (b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) →
                (c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) →
                a ≠ b ∧ a ≠ c ∧ b ≠ c →
                a ≠ 0 →
∀ (x y z : ℕ), (x ∈ {1, 2, 3, 4, 5, 6, 7, 8}) →
                (y ∈ {1, 2, 3, 4, 5, 6, 7, 8}) →
                (z ∈ {1, 2, 3, 4, 5, 6, 7, 8}) →
                x ≠ y ∧ x ≠ z ∧ y ≠ z →
(b * 10^2 + c * 10 + a) > (z * 10^2 + y * 10 + x) → 
(b * 10^2 + c * 10 + a > z * 10^2 + y * 10 + x) →
Bernardo_bigger a b c x y z = 3 / 4

theorem probability_Bernardo_number_larger :
  bernardo_and_silvia := sorry

end probability_Bernardo_number_larger_l311_311962


namespace smallest_n_with_4_prime_factors_l311_311435

open Nat

theorem smallest_n_with_4_prime_factors :
  ∃ (n : ℕ), (∀ (d : ℕ), d < n → ∃ (p q r s : ℕ), (prime p) ∧ (prime q) ∧ (prime r) ∧ (prime s) ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ (d^2 + 4 = p * q * r * s)) ∧ 
  ∃ (p1 p2 p3 p4 : ℕ), (prime p1) ∧ (prime p2) ∧ (prime p3) ∧ (prime p4) ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ (n^2 + 4 = p1 * p2 * p3 * p4) ∧ n = 179 :=
by
  sorry

end smallest_n_with_4_prime_factors_l311_311435


namespace tan_positive_iff_third_quadrant_l311_311520

theorem tan_positive_iff_third_quadrant (α : ℝ) (h : sin α < 0) :
  (tan α > 0 ↔ (π < α ∧ α < 3 * π / 2)) :=
sorry

end tan_positive_iff_third_quadrant_l311_311520


namespace inverse_proportion_m_value_l311_311090

theorem inverse_proportion_m_value (m : ℝ) (x : ℝ) (h : y = (m - 2) * x ^ (m^2 - 5)) : 
  y is inverse_proportional_function → m = -2 :=
sorry

end inverse_proportion_m_value_l311_311090


namespace simplify_cuberoot_product_l311_311228

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311228


namespace Joel_picked_approximately_11_peppers_each_day_l311_311130

-- Define the conditions
variables {total_peppers : ℕ} (non_hot_peppers : ℕ := 64) 
variables (hot_percentage : ℝ := 0.20) (days_in_week : ℕ := 7)

-- The key equation based on the given conditions
def total_peppers_equation := non_hot_peppers = (1 - hot_percentage) * total_peppers

-- Define the proof problem
theorem Joel_picked_approximately_11_peppers_each_day :
  ∃ peppers_per_day : ℕ, peppers_per_day = Nat.floor (total_peppers / days_in_week) ∧ peppers_per_day ≈ 11 :=
by
  sorry

end Joel_picked_approximately_11_peppers_each_day_l311_311130


namespace sum_of_divisors_of_29_l311_311755

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311755


namespace center_of_mass_velocity_l311_311926

-- Define the variables
variables (m : ℝ) (v₁ : ℝ := 12) (v₂ : ℝ := 0) (m₂ : ℝ := 4) (v₁' : ℝ := -6) (v₂' : ℝ)

-- Define the conditions
def elastic_collision : Prop :=
  v₂' = (9 * m) / 2 ∧
  (1 / 2) * m * (v₁)^2 = (1 / 2) * m * (v₁')^2 + (1 / 2) * m₂ * (v₂')^2

-- Define the theorem to be proved
theorem center_of_mass_velocity :
  elastic_collision m v₁ v₂ m₂ v₁' v₂' →
  m = 4 / 3 →
  let v_cm := (m * v₁ + m₂ * v₂) / (m + m₂) in
  v_cm = 3 :=
by
  sorry

end center_of_mass_velocity_l311_311926


namespace work_days_of_A_B_C_l311_311693

theorem work_days_of_A_B_C (A B C : ℚ)
  (h1 : A + B = 1/8)
  (h2 : B + C = 1/12)
  (h3 : A + C = 1/8) : A + B + C = 1/6 := 
  sorry

end work_days_of_A_B_C_l311_311693


namespace probability_of_snow_at_most_three_days_in_December_l311_311286

noncomputable def prob_snow_at_most_three_days (n : ℕ) (p : ℚ) : ℚ :=
  (Finset.range (n + 1)).sum (λ k => (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)))

theorem probability_of_snow_at_most_three_days_in_December :
  prob_snow_at_most_three_days 31 (1/5:ℚ) ≈ 0.230 :=
sorry

end probability_of_snow_at_most_three_days_in_December_l311_311286


namespace complete_half_job_in_six_days_l311_311938

theorem complete_half_job_in_six_days (x : ℕ) (h1 : 2 * x = x + 6) : x = 6 :=
  by
    sorry

end complete_half_job_in_six_days_l311_311938


namespace find_x_l311_311692

noncomputable def solution_x (m n y : ℝ) (m_gt_3n : m > 3 * n) : ℝ :=
  (n * m) / (m + n)

theorem find_x (m n y : ℝ) (m_gt_3n : m > 3 * n) :
  let initial_acid := m * (m / 100)
  let final_volume := m + (solution_x m n y m_gt_3n) + y
  let final_acid := (m - n) / 100 * final_volume
  initial_acid = final_acid → 
  solution_x m n y m_gt_3n = (n * m) / (m + n) :=
by sorry

end find_x_l311_311692


namespace sum_of_divisors_prime_29_l311_311725

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311725


namespace sum_of_divisors_of_29_l311_311711

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311711


namespace construct_n_times_segment_l311_311904

-- Definitions based on conditions
variables (A B : Point) (n : ℕ)

-- Definition of segment AB
def segment_AB : Segment := Segment.mk A B

-- Statement of the problem
theorem construct_n_times_segment (n : ℕ) : exists C : Point, Segment.length (Segment.mk A C) = n * Segment.length segment_AB :=
sorry

end construct_n_times_segment_l311_311904


namespace simplified_expression_correct_l311_311211

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311211


namespace barycentric_coordinates_exist_unique_l311_311459

variables {R : Type*} [Field R]
variables {V : Type*} [AddCommGroup V] [Module R V]
variables {P : Type*} [AffineSpace V P]

/-- We assume the existence of three points A1, A2, A3 forming the triangle on an affine space, 
    and a point X that we want to express in barycentric coordinates. -/
variables (A1 A2 A3 X : P)

namespace barycentric_coordinates
open Affine

/--
  Prove that any point X has barycentric coordinates relative to the triangle A1 A2 A3,
  and that these coordinates are uniquely determined under the condition 
  m1 + m2 + m3 = 1.
-/
theorem barycentric_coordinates_exist_unique 
  (h : ∃ (m1 m2 m3 : R), m1 + m2 + m3 = 1 ∧ X = m1 • (A1 -ᵥ A3) + m2 • (A2 -ᵥ A3) + A3) :
  ∃! (m1 m2 m3 : R), m1 + m2 + m3 = 1 ∧ X = m1 • (A1 -ᵥ A3) + m2 • (A2 -ᵥ A3) + A3 :=
sorry

end barycentric_coordinates

end barycentric_coordinates_exist_unique_l311_311459


namespace zoo_open_hours_l311_311690

theorem zoo_open_hours (h : ℕ) (visitors_per_hour : ℕ) (percent_gorilla_exhibit : ℝ) (gorilla_exhibit_visitors : ℕ) :
  visitors_per_hour = 50 →
  percent_gorilla_exhibit = 0.80 →
  gorilla_exhibit_visitors = 320 →
  (percent_gorilla_exhibit * visitors_per_hour * h = gorilla_exhibit_visitors) →
  h = 8 := 
by
  intros h visitors_per_hour percent_gorilla_exhibit gorilla_exhibit_visitors
  intros h_visitors_per_hour h_percent_gorilla_exhibit h_gorilla_exhibit_visitors h_equation
  sorry

end zoo_open_hours_l311_311690


namespace logarithm_exponent_problem_l311_311401

theorem logarithm_exponent_problem :
  ( (9 / 4)^(1 / 2) * (27 / 8)^(-1 / 3) - (Real.log 2)^2 - (Real.log 5)^2 - 2 * Real.log 2 * Real.log 5 ) = 0 :=
  sorry

end logarithm_exponent_problem_l311_311401


namespace sum_of_divisors_of_29_l311_311763

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311763


namespace simplify_cube_roots_l311_311238

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311238


namespace average_speed_l311_311184

theorem average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 50) (h2 : d2 = 20) (h3 : t1 = 50 / 20) (h4 : t2 = 20 / 40) :
  ((d1 + d2) / (t1 + t2)) = 23.33 := 
  sorry

end average_speed_l311_311184


namespace box_shiny_pennies_prob_l311_311353

theorem box_shiny_pennies_prob
    (a b : ℕ)
    (h_box : box_contains 5 6)
    (h_draws : pennies_drawn_without_replacement h_box)
    (h_prob : probability_of_fourth_shiny_after_six_draws h_box h_draws = a / b) :
    a + b = 386 := by
  sorry

end box_shiny_pennies_prob_l311_311353


namespace ellipse_circle_min_radius_l311_311408

/-- Given an ellipse defined by the equation x^2 / 9 + y^2 / 4 = 1,
and a circle that passes through both foci of this ellipse with its center along the x-axis,
prove that the minimum radius r of such a circle is sqrt(5). -/
theorem ellipse_circle_min_radius :
  ∃ r : ℝ, (∀ x y : ℝ, x^2 / 9 + y^2 / 4 = 1 → r = ℝ.sqrt(5)) :=
sorry

end ellipse_circle_min_radius_l311_311408


namespace simplify_cuberoot_product_l311_311224

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311224


namespace smallest_square_area_l311_311921

theorem smallest_square_area (a b c d : ℕ) (ha : a = 2) (hb : b = 4) (hc : c = 3) (hd : d = 5) :
  ∃ s : ℕ, s * s = 81 :=
by
  use 9
  norm_num
  sorry

end smallest_square_area_l311_311921


namespace correct_option_l311_311307
open Real

def h (t : ℝ) : ℝ := (1 / 2) * t^2 + t

def h_prime (t : ℝ) : ℝ := t + 1

def V1 : ℝ := h_prime 1

def V2 : ℝ := h_prime 4

def average_speed (initial_time end_time : ℝ) : ℝ :=
  (h end_time - h initial_time) / (end_time - initial_time)

def V : ℝ := average_speed 1 4

theorem correct_option : V = (V1 + V2) / 2 :=
by
  -- Proof skipped
  sorry

end correct_option_l311_311307


namespace sqrt_diff_lt_sqrt_diff_l311_311699

theorem sqrt_diff_lt_sqrt_diff (a b : ℝ) (h1 : a > b) (h2 : b > 0) : sqrt(a) - sqrt(b) < sqrt(a - b) :=
sorry

end sqrt_diff_lt_sqrt_diff_l311_311699


namespace find_factor_l311_311946

theorem find_factor (x f : ℕ) (hx : x = 110) (h : x * f - 220 = 110) : f = 3 :=
sorry

end find_factor_l311_311946


namespace football_team_lineup_ways_l311_311637

theorem football_team_lineup_ways :
  let members := 12
  let offensive_lineman_options := 4
  let remaining_after_linemen := members - offensive_lineman_options
  let quarterback_options := remaining_after_linemen
  let remaining_after_qb := remaining_after_linemen - 1
  let wide_receiver_options := remaining_after_qb
  let remaining_after_wr := remaining_after_qb - 1
  let tight_end_options := remaining_after_wr
  let lineup_ways := offensive_lineman_options * quarterback_options * wide_receiver_options * tight_end_options
  lineup_ways = 3960 :=
by
  sorry

end football_team_lineup_ways_l311_311637


namespace hyperbola_equation_fixed_point_l311_311027

-- Define all the initial mathematical conditions
def eccentricity := Real.sqrt 5 / 2
def center := (0 : ℝ, 0 : ℝ)
def foci := ((-2 : ℝ, 0 : ℝ), (2 : ℝ, 0 : ℝ)) -- since foci lie on the x-axis

-- Define point A and the given conditions
def point_A := (a_x : ℝ, a_y : ℝ)
def perpendicular_condition := (a_x + 2) * (a_x - 2) + a_y ^ 2 = 0  -- \overrightarrow{AF_1} \cdot \overrightarrow{AF_2} = 0
def area_condition := (1 / 2) * Real.abs(a_y * (4 : ℝ)) = 1  -- Area of \triangle F_1AF_2 is 1

-- Hyperbola equation derivation
theorem hyperbola_equation : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ eccentricity = Real.sqrt (a^2 + b^2) / a ∧ a = 2 * b ∧ b = 1) → 
  ∃ x y : ℝ, (x^2 / 4) - y^2 = 1 := 
sorry

-- Existence of fixed point
theorem fixed_point (k m : ℝ) :
  (∃ E F : ℝ × ℝ, (∃ x1 y1 x2 y2 : ℝ, E = (x1, y1) ∧ F = (x2, y2) ∧ y1 = k * x1 + m ∧ y2 = k * x2 + m ∧ 
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) / 2) = Real.sqrt ((2 - x1)^2 + (0 - y1)^2)) ∧ 
  (4 * k^2 * a^2 + 5 * b^2) = 0) → 
  ∃ fixed_pt : (ℝ × ℝ), fixed_pt = (10 / 3, 0) :=
sorry

end hyperbola_equation_fixed_point_l311_311027


namespace simplified_expression_correct_l311_311207

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311207


namespace m_plus_n_l311_311518

theorem m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m ^ n = 2^25 * 3^40) : m + n = 209957 :=
  sorry

end m_plus_n_l311_311518


namespace smallest_common_multiple_of_10_11_18_l311_311009

theorem smallest_common_multiple_of_10_11_18 : 
  ∃ (n : ℕ), (n % 10 = 0) ∧ (n % 11 = 0) ∧ (n % 18 = 0) ∧ (n = 990) :=
by
  sorry

end smallest_common_multiple_of_10_11_18_l311_311009


namespace probability_of_4_vertices_in_plane_l311_311445

-- Definition of the problem conditions
def vertices_of_cube : Nat := 8
def selecting_vertices : Nat := 4

-- Combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 4 vertices from the 8 vertices of a cube
def total_ways : Nat := combination vertices_of_cube selecting_vertices

-- Number of favorable ways that these 4 vertices lie in the same plane
def favorable_ways : Nat := 12

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

-- The ultimate proof problem
theorem probability_of_4_vertices_in_plane :
  probability = 6 / 35 :=
by
  -- Here, the proof steps would go to verify that our setup correctly leads to the given probability.
  sorry

end probability_of_4_vertices_in_plane_l311_311445


namespace cylinder_sphere_volume_ratio_l311_311290

theorem cylinder_sphere_volume_ratio (R : ℝ) (hR : 0 < R) :
  let V_sphere := (4 / 3) * Real.pi * R^3,
      V_cylinder := 2 * Real.pi * R^3 in
  V_cylinder / V_sphere = 3 / 2 :=
by
  let V_sphere := (4 / 3) * Real.pi * R^3
  let V_cylinder := 2 * Real.pi * R^3
  sorry

end cylinder_sphere_volume_ratio_l311_311290


namespace password_count_password_problem_l311_311915

-- Statement of the problem in Lean 4
theorem password_count (digits : List ℕ) (distinct_digits : List ℕ) (digit_count : ℕ) (no_adjacent_twos : Bool) : ℕ :=
 if digits = [2, 7, 1, 8, 2] ∧ distinct_digits = [7, 1, 8] ∧ digit_count = 5 ∧ no_adjacent_twos then 36
 else 0

-- state the theorem
theorem password_problem : password_count [2, 7, 1, 8, 2] [7, 1, 8] 5 true = 36 := by
  unfold password_count
  simp
  sorry

end password_count_password_problem_l311_311915


namespace monotone_decreasing_interval_sum_reciprocal_logs_minimum_integer_a_l311_311489

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) := ln x - x^2 + a * x

-- Part (1) Find the monotonically decreasing interval of the function f(x)
theorem monotone_decreasing_interval (a : ℝ) (h : f 1 a = 0) : 
  ∃ A, ∀ x : ℝ, 1 < x → f x 1 < f 1 1 :=
sorry

-- Part (2) Prove the inequality for reciprocals of logarithms
theorem sum_reciprocal_logs (n : ℕ) (h : 2 ≤ n) : 
  (∑ i in Finset.range (n - 1) + 2, 1 / ln i.succ.succ) > 1 :=
sorry

-- Part (3) Find the minimum integer value of a
theorem minimum_integer_a (a : ℝ) :
  (∀ x, 0 < x → f x a ≤ (1 / 2 * a - 1) * x^2 + (2 * a - 1) * x - 1) → 2 ≤ a :=
sorry

end monotone_decreasing_interval_sum_reciprocal_logs_minimum_integer_a_l311_311489


namespace damage_conversion_l311_311932

def usd_to_cad_conversion_rate : ℝ := 1.25
def damage_in_usd : ℝ := 60000000
def damage_in_cad : ℝ := 75000000

theorem damage_conversion :
  damage_in_usd * usd_to_cad_conversion_rate = damage_in_cad :=
sorry

end damage_conversion_l311_311932


namespace ratio_of_parallel_vectors_l311_311072

theorem ratio_of_parallel_vectors (m n : ℝ) 
  (h1 : ∃ k : ℝ, (m, 1, 3) = (k * 2, k * n, k)) : (m / n) = 18 :=
by
  sorry

end ratio_of_parallel_vectors_l311_311072


namespace min_value_is_3_plus_2_sqrt_2_l311_311465

noncomputable def minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) : ℝ :=
a + b

theorem min_value_is_3_plus_2_sqrt_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) :
  minimum_value a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_is_3_plus_2_sqrt_2_l311_311465


namespace increasing_on_neg_infinity_to_zero_l311_311101

-- Definitions based on given conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def is_quadratic_even_function (m : ℝ) : Prop := 
  is_even (λ x, (m - 1) * x^2 + 2 * m * x + 1)

def is_increasing_on_interval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x < y → x ∈ I → y ∈ I → f x ≤ f y

-- The proof problem
theorem increasing_on_neg_infinity_to_zero (f : ℝ → ℝ) (m : ℝ) :
  is_quadratic_even_function m →
  f = λ x, (m - 1) * x^2 + 2 * m * x + 1 →
  is_increasing_on_interval f (Set.Iic 0) :=
by
  sorry

end increasing_on_neg_infinity_to_zero_l311_311101


namespace parallelogram_area_proof_l311_311630

noncomputable def parallelogram_area : ℝ :=
  let angle_rad := (150 * real.pi / 180)  -- converting degrees to radians
  let a := 10                              -- length of one side
  let b := 20                              -- length of another side
  let height := a * real.sqrt(3) / 2       -- height from 30-60-90 triangle properties
  b * height

theorem parallelogram_area_proof : parallelogram_area = 100 * real.sqrt(3) := by
  sorry

end parallelogram_area_proof_l311_311630


namespace ordered_quadruples_count_l311_311079

theorem ordered_quadruples_count :
  ∃ (count : ℕ), count = 630 ∧
  ∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                   a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
                   b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
                   c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
                   d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
                   b < a ∧ b < c ∧ d < c → count = 630 := 
by 
  sorry

end ordered_quadruples_count_l311_311079


namespace student_losses_one_mark_l311_311110

def number_of_marks_lost_per_wrong_answer (correct_ans marks_attempted total_questions total_marks correct_questions : ℤ) : ℤ :=
  (correct_ans * correct_questions - total_marks) / (total_questions - correct_questions)

theorem student_losses_one_mark
  (correct_ans : ℤ)
  (marks_attempted : ℤ)
  (total_questions : ℤ)
  (total_marks : ℤ)
  (correct_questions : ℤ)
  (total_wrong : ℤ):
  correct_ans = 4 →
  total_questions = 80 →
  total_marks = 120 →
  correct_questions = 40 →
  total_wrong = total_questions - correct_questions →
  number_of_marks_lost_per_wrong_answer correct_ans marks_attempted total_questions total_marks correct_questions = 1 :=
by
  sorry

end student_losses_one_mark_l311_311110


namespace cube_root_multiplication_l311_311198

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311198


namespace sum_of_divisors_of_29_l311_311805

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311805


namespace polar_to_cartesian_equivalence_l311_311119

def circle_polar (rho theta : ℝ) : Prop := rho = Real.cos theta + Real.sin theta
def line_polar (rho theta : ℝ) : Prop := rho * Real.sin (theta - π / 4) = sqrt 2 / 2

def circle_cartesian (x y : ℝ) : Prop := x^2 + y^2 - x - y = 0
def line_cartesian (x y : ℝ) : Prop := x - y + 1 = 0

def polar_point (rho theta : ℝ) : Prop := rho = 1 ∧ theta = π / 2

theorem polar_to_cartesian_equivalence:
  (∀ rho theta, circle_polar rho theta → circle_cartesian (rho * Real.cos theta) (rho * Real.sin theta)) ∧
  (∀ rho theta, line_polar rho theta → line_cartesian (rho * Real.cos theta) (rho * Real.sin theta)) ∧
  (∃ (x y : ℝ), circle_cartesian x y ∧ line_cartesian x y ∧ polar_point (Real.sqrt (x^2 + y^2)) (Real.atan y x)) :=
sorry

end polar_to_cartesian_equivalence_l311_311119


namespace HCF_a_b_LCM_a_b_l311_311669

-- Given the HCF condition
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Given numbers
def a : ℕ := 210
def b : ℕ := 286

-- Given HCF condition
theorem HCF_a_b : HCF a b = 26 := by
  sorry

-- LCM definition based on the product and HCF
def LCM (a b : ℕ) : ℕ := (a * b) / HCF a b

-- Theorem to prove
theorem LCM_a_b : LCM a b = 2310 := by
  sorry

end HCF_a_b_LCM_a_b_l311_311669


namespace initial_fabric_l311_311411

/-!
# Proof Problem: Initial Fabric Calculation

## Given:
- Square flags are 4 feet by 4 feet.
- Wide rectangular flags are 5 feet by 3 feet.
- Tall rectangular flags are 3 feet by 5 feet.
- Darnell has already made 16 square flags, 20 wide flags, and 10 tall flags.
- Darnell has 294 square feet of fabric left.

## Prove:
- Darnell initially had 1000 square feet of fabric.
-/

theorem initial_fabric :
  let area_square := 4 * 4,
      area_wide := 5 * 3,
      area_tall := 3 * 5,
      used_square := 16 * area_square,
      used_wide := 20 * area_wide,
      used_tall := 10 * area_tall,
      total_used := used_square + used_wide + used_tall,
      fabric_left := 294
  in total_used + fabric_left = 1000 :=
by
  sorry

end initial_fabric_l311_311411


namespace probability_of_winning_l311_311380

/-- There are 4 multiple-choice questions, each with 3 choices. The contestant guesses each answer randomly.
    Prove that the probability of winning the quiz by guessing at least 3 out of 4 questions correctly is 1/9. -/
theorem probability_of_winning : 
  (let prob (n k : ℕ) := (1 / 3) ^ k * (2 / 3) ^ (n - k) in
   let term_1 := (1 / 3) ^ 4 in
   let term_2 := 4 * prob 4 3 in
   term_1 + term_2 = 1 / 9) := 
  sorry

end probability_of_winning_l311_311380


namespace find_3a_plus_4b_l311_311413

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def h (x : ℝ) : ℝ := 3 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 3 * x - 4

-- Conditions
axiom h_equals_f_inv_min_2 (x : ℝ) : h(x) = f_inv(x) - 2
axiom f_inv_is_inverse (a b : ℝ) : ∀ x : ℝ, f a b (f_inv x) = x

-- Theorem statement
theorem find_3a_plus_4b (a b : ℝ) (h_equals_f_inv_min_2 : ∀ x : ℝ, h(x) = f_inv(x) - 2)
  (f_inv_is_inverse : ∀ x : ℝ, f a b (f_inv x) = x) : 3 * a + 4 * b = 19 / 3 :=
sorry

end find_3a_plus_4b_l311_311413


namespace necessary_but_not_sufficient_l311_311348

-- Define the conditions as seen in the problem statement
def condition_x (x : ℝ) : Prop := x < 0
def condition_ln (x : ℝ) : Prop := Real.log (x + 1) < 0

-- State that the condition "x < 0" is necessary but not sufficient for "ln(x + 1) < 0"
theorem necessary_but_not_sufficient :
  ∀ (x : ℝ), (condition_ln x → condition_x x) ∧ ¬(condition_x x → condition_ln x) :=
by
  sorry

end necessary_but_not_sufficient_l311_311348


namespace sum_of_divisors_of_29_l311_311877

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311877


namespace imaginary_part_conjugate_z_l311_311481

def z : ℂ := (i - 5) / (1 + i)

theorem imaginary_part_conjugate_z : complex.im (conjugate z) = -3 := by
  sorry

end imaginary_part_conjugate_z_l311_311481


namespace ellipse_eccentricity_of_C1_l311_311464

-- Definition of point P on ellipse C1 and hyperbola C2
variables {a b : ℝ} (P F1 F2 : ℝ × ℝ)

-- Given conditions
axiom h1 : a > b ∧ b > 0
axiom h2 : ∃ P, (P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ P.1^2 - P.2^2 = 4)
axiom h3 : |P - F2| = 2
axiom h4 : F2 = (2 * √2, 0) -- Assuming right focus as (2√2, 0) for computational purposes

-- Prove the eccentricity of ellipse C1
noncomputable def ellipse_eccentricity (a c : ℝ) : ℝ :=
  c / a

theorem ellipse_eccentricity_of_C1 (a b c : ℝ) (P F1 F2 : ℝ × ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h3 : P.1^2 - P.2^2 = 4)
  (h4 : |P - F2| = 2)
  (h5 : F2 = (2 * √2, 0))
  (h6 : c = 2 * √2)
  : ellipse_eccentricity a c = (√2 / 2) :=
sorry

end ellipse_eccentricity_of_C1_l311_311464


namespace double_windows_downstairs_eq_twelve_l311_311377

theorem double_windows_downstairs_eq_twelve
  (D : ℕ)
  (H1 : ∀ d, d = D → 4 * d + 32 = 80) :
  D = 12 :=
by
  sorry

end double_windows_downstairs_eq_twelve_l311_311377


namespace solution_inequality_maximum_value_l311_311494

def f (x : ℝ) := abs (x - 1) - abs (2 * x + 3)

theorem solution_inequality :
  { x : ℝ // f x ≥ x + 1 } = { x : ℝ // x ∈ Set.Iic (-3 / 4) } := sorry

def m : ℝ := 5 / 2

theorem maximum_value (a b c : ℝ) (h : 1 / a^2 + 4 / b^2 + 9 / c^2 = 1) :
  (1 / a + 1 / b + 1 / c) ≤ 7 / 6 := sorry

end solution_inequality_maximum_value_l311_311494


namespace sum_of_divisors_prime_29_l311_311726

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311726


namespace sum_of_divisors_prime_29_l311_311734

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311734


namespace sum_of_divisors_29_l311_311820

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311820


namespace Nancy_seeds_l311_311603

def big_garden_seeds : ℕ := 28
def small_gardens : ℕ := 6
def seeds_per_small_garden : ℕ := 4

def total_seeds : ℕ := big_garden_seeds + small_gardens * seeds_per_small_garden

theorem Nancy_seeds : total_seeds = 52 :=
by
  -- Proof here...
  sorry

end Nancy_seeds_l311_311603


namespace ratio_area_rectangle_to_quarter_circles_l311_311930

-- Definitions based on conditions
def short_side : ℝ := 20
def ratio_long_to_short : ℝ := 7 / 4

-- Definition of the longer side based on the ratio
def long_side : ℝ := short_side * ratio_long_to_short

-- Area of the rectangle
def area_rectangle : ℝ := long_side * short_side

-- Radius for the quarter-circles
def radius_quarter_circle : ℝ := short_side

-- Area of the equivalent full circle from four quarter-circles
def area_full_circle : ℝ := Real.pi * radius_quarter_circle^2

-- The final theorem we aim to prove
theorem ratio_area_rectangle_to_quarter_circles : 
  area_rectangle / area_full_circle = 7 / (4 * Real.pi) :=
by 
  sorry  -- Proof is skipped as per the instructions

end ratio_area_rectangle_to_quarter_circles_l311_311930


namespace area_transformation_l311_311679

theorem area_transformation (f : ℝ → ℝ) (area_f : ℝ) (h : area_f = 10) :
  let g := λ x, 3 * f (x - 2) in
  ∫ x in 0..(upper_bound g), g(x) = 30 := 
sorry

end area_transformation_l311_311679


namespace inverse_proportion_value_of_m_l311_311087

theorem inverse_proportion_value_of_m (m : ℤ) (x : ℝ) (y : ℝ) : 
  y = (m - 2) * x ^ (m^2 - 5) → (m = -2) := 
by
  sorry

end inverse_proportion_value_of_m_l311_311087


namespace determinant_of_roots_l311_311151

noncomputable def determinant_expr (a b c d s p q r : ℝ) : ℝ :=
  by sorry

theorem determinant_of_roots (a b c d s p q r : ℝ)
    (h1 : a + b + c + d = -s)
    (h2 : abcd = r)
    (h3 : abc + abd + acd + bcd = -q)
    (h4 : ab + ac + bc = p) :
    determinant_expr a b c d s p q r = r - q + pq + p :=
  by sorry

end determinant_of_roots_l311_311151


namespace find_f_three_l311_311026

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b
axiom f_two : f 2 = 3

theorem find_f_three : f 3 = 9 / 2 :=
by
  sorry

end find_f_three_l311_311026


namespace wendy_makeup_time_l311_311309

theorem wendy_makeup_time :
  ∀ (num_products wait_time total_time makeup_time : ℕ),
    num_products = 5 →
    wait_time = 5 →
    total_time = 55 →
    makeup_time = total_time - (num_products - 1) * wait_time →
    makeup_time = 35 :=
by
  intro num_products wait_time total_time makeup_time h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_makeup_time_l311_311309


namespace alpha_necessary_not_sufficient_beta_l311_311021

variable {x a : ℝ}

theorem alpha_necessary_not_sufficient_beta
  (h₁ : ∀ x, α: x ≥ a) 
  (h₂ : ∀ x, β: |x - 1| < 1) 
  (h₃ : ∀ x, (α x → β x) ∧ ¬(β x → α x)):
  a ≤ 2 :=
sorry

end alpha_necessary_not_sufficient_beta_l311_311021


namespace problem_l311_311034

open Nat

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + n * d

theorem problem (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 5) (h_S8 : S 8 = 100)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, a n = 3 * n - 1) ∧
    (∀ b T n, (b n = 4 ^ (a n) + 2 * n) →
      T n = (∑ i in range n, b i) →
      T n = (16 / 63 * (64 ^ n - 1) + (n * (n + 1)))) :=
by
  sorry

end problem_l311_311034


namespace largest_consecutive_even_integer_l311_311685

theorem largest_consecutive_even_integer
  (sum_first_30_evens : ∑ i in (finset.range 30).map (nat.succ ∘ (•2)), id = 930)
  (sum_five_consecutive_evens : ∃ n, (n - 8) + (n - 6) + (n - 4) + (n - 2) + n = 930) :
  ∃ n, (n - 8, n - 6, n - 4, n - 2, n).2 = 190 :=
by
  sorry

end largest_consecutive_even_integer_l311_311685


namespace cube_root_simplification_l311_311250

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311250


namespace exists_three_distinct_nats_sum_prod_squares_l311_311999

theorem exists_three_distinct_nats_sum_prod_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (∃ (x : ℕ), a + b + c = x^2) ∧ 
  (∃ (y : ℕ), a * b * c = y^2) :=
sorry

end exists_three_distinct_nats_sum_prod_squares_l311_311999


namespace simplify_cuberoot_product_l311_311235

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311235


namespace solve_problem_l311_311996

noncomputable def sin_neg_390_eq : Real :=
  - (Real.sin (390 * π / 180))

noncomputable def problem_statement : Prop :=
  sin_neg_390_eq = - Real.sin (30 * π / 180)

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l311_311996


namespace monotonic_decreasing_intervals_l311_311479

theorem monotonic_decreasing_intervals (α : ℝ) (hα : α < 0) :
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → x ^ α > y ^ α) ∧ 
  (∀ x y : ℝ, x < y ∧ 0 < x ∧ 0 < y → x ^ α > y ^ α) :=
by
  sorry

end monotonic_decreasing_intervals_l311_311479


namespace april_earnings_l311_311394

theorem april_earnings (cost_per_rose : ℕ) (roses_initially : ℕ) (roses_left : ℕ) 
  (cost_set: cost_per_rose = 7) (initial_roses_set: roses_initially = 9) (remaining_roses_set: roses_left = 4) :
  (roses_initially - roses_left) * cost_per_rose = 35 :=
by
  -- Definitions from the conditions
  rw [cost_set, initial_roses_set, remaining_roses_set]
  -- Pending proof: supplied with sorry to adhere to guidelines
  sorry

end april_earnings_l311_311394


namespace total_playtime_l311_311580

-- Conditions
def lena_playtime_hours : ℝ := 3.5
def minutes_per_hour : ℝ := 60
def lena_playtime_minutes : ℝ := lena_playtime_hours * minutes_per_hour
def brother_playtime_extra_minutes : ℝ := 17
def brother_playtime_minutes : ℝ := lena_playtime_minutes + brother_playtime_extra_minutes

-- Proof problem
theorem total_playtime : lena_playtime_minutes + brother_playtime_minutes = 437 := by
  sorry

end total_playtime_l311_311580


namespace total_playtime_l311_311578

-- Conditions
def lena_playtime_hours : ℝ := 3.5
def minutes_per_hour : ℝ := 60
def lena_playtime_minutes : ℝ := lena_playtime_hours * minutes_per_hour
def brother_playtime_extra_minutes : ℝ := 17
def brother_playtime_minutes : ℝ := lena_playtime_minutes + brother_playtime_extra_minutes

-- Proof problem
theorem total_playtime : lena_playtime_minutes + brother_playtime_minutes = 437 := by
  sorry

end total_playtime_l311_311578


namespace khintchine_law_of_large_numbers_l311_311343
noncomputable section

open ProbabilityTheory MeasureTheory Filter

variable {Ω : Type*} {X : ℕ → Ω → ℝ} [MeasureSpace Ω]

/-- Khintchine's Law of Large Numbers states a result for i.i.d. random variables under certain conditions -/
theorem khintchine_law_of_large_numbers
  (X : ℕ → Ω → ℝ)
  (h_indep : ∀ n m, n ≠ m → IndepFun (X n) (X m))
  (h_iid : ∀ n, dist (X n) = dist (X 0))
  (h_condition : ∀ ε, ε > 0 → ∃ N, ∀ n > N, n * (prob (|X 1| > ennreal.of_real n)) < ε)
  (h_pos : ∀ ω, 0 < X 1 ω)
  : Tendsto (λ n, (∫ ω, (X n ω) ^ 2) / (∫ ω, finset.sum (finset.range n) (λ i, X i ω)) ^ 2) at_top (𝓝 0) :=
sorry

end khintchine_law_of_large_numbers_l311_311343


namespace smallest_square_area_l311_311919

theorem smallest_square_area (a b c d s : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 3) (h4 : d = 5) (h5 : s = a + c) :
  ∃ S, S * S = 81 := by
  use 9
  have h6 : 9 * 9 = 81 := by norm_num
  exact h6

end smallest_square_area_l311_311919


namespace speed_ratio_l311_311325

theorem speed_ratio (L tA tB : ℝ) (R : ℝ) (h1: A_speed = R * B_speed) 
  (h2: head_start = 0.35 * L) (h3: finish_margin = 0.25 * L)
  (h4: A_distance = L + head_start) (h5: B_distance = L)
  (h6: A_finish = A_distance / A_speed)
  (h7: B_finish = B_distance / B_speed)
  (h8: B_finish_time = A_finish + finish_margin / B_speed)
  : R = 1.08 :=
by
  sorry

end speed_ratio_l311_311325


namespace area_of_circle_l311_311496

noncomputable def is_circle (r : ℝ → ℝ) : Prop :=
∃ (h k : ℝ) (R : ℝ), ∀ (θ : ℝ), 
  r θ = R * (Math.cos θ - k * Math.sin θ / R) ∧ 
  (h * Math.cos θ + k * Math.sin θ = R)

theorem area_of_circle : 
  is_circle (λ θ, 3 * Math.cos θ - 4 * Math.sin θ) ∧ 
  (π * (5 / 2)^2 = 25 * π / 4) :=
by
  sorry

end area_of_circle_l311_311496


namespace outfit_count_l311_311897

section OutfitProblem

-- Define the number of each type of shirts, pants, and hats
def num_red_shirts : ℕ := 7
def num_blue_shirts : ℕ := 5
def num_green_shirts : ℕ := 8

def num_pants : ℕ := 10

def num_green_hats : ℕ := 10
def num_red_hats : ℕ := 6
def num_blue_hats : ℕ := 7

-- The main theorem to prove the number of outfits where shirt and hat are not the same color
theorem outfit_count : 
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats) +
  num_blue_shirts * num_pants * (num_green_hats + num_red_hats) +
  num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) = 3030 :=
  sorry

end OutfitProblem

end outfit_count_l311_311897


namespace graph_of_equation_l311_311393

theorem graph_of_equation (x y : ℝ) :
  x^3 * (x + y + 2) = y^3 * (x + y + 2) →
  (x + y + 2 ≠ 0 ∧ (x = y ∨ x^2 + x * y + y^2 = 0)) ∨
  (x + y + 2 = 0 ∧ y = -x - 2) →
  (y = x ∨ y = -x - 2) := 
sorry

end graph_of_equation_l311_311393


namespace proof_problem_l311_311091

theorem proof_problem :
  ∀ (X : ℝ), 213 * 16 = 3408 → (213 * 16) + (1.6 * 2.13) = X → X - (5 / 2) * 1.25 = 3408.283 :=
by
  intros X h1 h2
  sorry

end proof_problem_l311_311091


namespace sum_of_divisors_prime_29_l311_311727

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311727


namespace simplify_cuberoot_product_l311_311229

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311229


namespace cost_of_water_l311_311390

theorem cost_of_water (total_cost sandwiches_cost : ℕ) (num_sandwiches sandwich_price water_price : ℕ) 
  (h1 : total_cost = 11) 
  (h2 : sandwiches_cost = num_sandwiches * sandwich_price) 
  (h3 : num_sandwiches = 3) 
  (h4 : sandwich_price = 3) 
  (h5 : total_cost = sandwiches_cost + water_price) : 
  water_price = 2 :=
by
  sorry

end cost_of_water_l311_311390


namespace sum_of_divisors_of_prime_l311_311783

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311783


namespace sum_of_divisors_of_29_l311_311709

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311709


namespace find_first_number_l311_311668

/-- The lcm of two numbers is 2310 and hcf (gcd) is 26. One of the numbers is 286. What is the other number? --/
theorem find_first_number (A : ℕ) 
  (h_lcm : Nat.lcm A 286 = 2310) 
  (h_gcd : Nat.gcd A 286 = 26) : 
  A = 210 := 
by
  sorry

end find_first_number_l311_311668


namespace simplified_expression_correct_l311_311209

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311209


namespace largest_possible_package_size_l311_311159

theorem largest_possible_package_size :
  ∃ (p : ℕ), Nat.gcd 60 36 = p ∧ p = 12 :=
by
  use 12
  sorry -- The proof is skipped as per instructions

end largest_possible_package_size_l311_311159


namespace combined_salaries_of_B_C_D_E_l311_311680

theorem combined_salaries_of_B_C_D_E
    (A_salary : ℕ)
    (average_salary_all : ℕ)
    (total_individuals : ℕ)
    (combined_salaries_B_C_D_E : ℕ) :
    A_salary = 8000 →
    average_salary_all = 8800 →
    total_individuals = 5 →
    combined_salaries_B_C_D_E = (average_salary_all * total_individuals) - A_salary →
    combined_salaries_B_C_D_E = 36000 :=
by
  sorry

end combined_salaries_of_B_C_D_E_l311_311680


namespace hike_down_distance_l311_311901

theorem hike_down_distance :
  let rate_up := 4 -- rate going up in miles per day
  let time := 2    -- time in days
  let rate_down := 1.5 * rate_up -- rate going down in miles per day
  let distance_down := rate_down * time -- distance going down in miles
  distance_down = 12 :=
by
  sorry

end hike_down_distance_l311_311901


namespace goat_distance_max_l311_311167

def point (x y : ℝ) : Type := {p : ℝ × ℝ // p.1 = x ∧ p.2 = y}

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def max_distance_greatest (a b c x y : ℝ) (h : (x, y) = (a, b) ∧ c > 0)
  : ℝ :=
  (distance (0,0) (x,y)) + c

theorem goat_distance_max :
  max_distance_greatest 6 8 15 6 8 (by simp [point]) = 25 :=
  by sorry

end goat_distance_max_l311_311167


namespace find_expression_value_l311_311039

theorem find_expression_value 
  (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := 
by 
  sorry

end find_expression_value_l311_311039


namespace min_selections_l311_311078

theorem min_selections (p : ℝ) (n : ℕ) (P_B : ℝ) (h_p : p = 0.5) (h_P_B : P_B = 0.9) : 
  1 - p^n ≥ P_B ↔ n ≥ 4 :=
by
  rw [h_p, h_P_B]
  sorry

end min_selections_l311_311078


namespace range_of_f_on_domain_l311_311275

-- We define the function y = -x^2 + 4x - 1
def f (x : ℝ) : ℝ := -x^2 + 4 * x - 1

-- Define the domain of x
def domain (x : ℝ) := x ∈ set.Icc (-1 : ℝ) (3 : ℝ)

-- Theorem statement to prove the range of the function on the given domain
theorem range_of_f_on_domain : set.range (λ x, f x) = set.Icc (-6 : ℝ) 3 :=
by
  sorry

end range_of_f_on_domain_l311_311275


namespace sum_of_divisors_29_l311_311874

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311874


namespace students_opted_for_both_math_and_science_l311_311538

-- Definitions for the conditions
def total_students : ℕ := 40
def not_math : ℕ := 10
def not_science : ℕ := 15
def not_history : ℕ := 20
def not_geography : ℕ := 5
def not_math_or_science : ℕ := 2
def not_math_or_history : ℕ := 3
def not_math_or_geography : ℕ := 4
def not_science_or_history : ℕ := 7
def not_science_or_geography : ℕ := 8
def not_history_or_geography : ℕ := 10

-- Theorem for the number of students opted for both math and science
theorem students_opted_for_both_math_and_science :
  let M := total_students - not_math,
      S := total_students - not_science,
      MS := M + S - (total_students - not_math_or_science)
  in MS = 17 :=
by
  let M := total_students - not_math,
      S := total_students - not_science,
      MS := M + S - (total_students - not_math_or_science)
  show MS = 17
  sorry

end students_opted_for_both_math_and_science_l311_311538


namespace necessary_but_not_sufficient_l311_311655

-- Define that for all x in ℝ, x^2 - 4x + 2m ≥ 0
def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 - 4 * x + 2 * m ≥ 0

-- Main theorem statement
theorem necessary_but_not_sufficient (m : ℝ) : 
  (proposition_p m → m ≥ 2) → (m ≥ 1 → m ≥ 2) :=
by
  intros h1 h2
  sorry

end necessary_but_not_sufficient_l311_311655


namespace sum_of_divisors_29_l311_311829

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311829


namespace soccer_league_teams_l311_311299

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 :=
by
  sorry

end soccer_league_teams_l311_311299


namespace manufacturing_cost_eq_210_l311_311673

theorem manufacturing_cost_eq_210 (transport_cost : ℝ) (shoecount : ℕ) (selling_price : ℝ) (gain : ℝ) (M : ℝ) :
  transport_cost = 500 / 100 →
  shoecount = 100 →
  selling_price = 258 →
  gain = 0.20 →
  M = (selling_price / (1 + gain)) - (transport_cost) :=
by
  intros
  sorry

end manufacturing_cost_eq_210_l311_311673


namespace sum_of_divisors_of_prime_l311_311789

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311789


namespace sum_even_l311_311018

theorem sum_even (n : ℕ) (hpos : 0 < n) : 
  let a1 := n^2 - 10 * n + 23 in
  let a2 := n^2 - 9 * n + 31 in
  let a3 := n^2 - 12 * n + 46 in
  (a1 + a2 + a3) % 2 = 0 := 
by 
  sorry

end sum_even_l311_311018


namespace find_mr_balogh_with_n_minus_1_questions_minimum_questions_to_find_balogh_l311_311396

theorem find_mr_balogh_with_n_minus_1_questions (n : ℕ) :
  (∃ (find_balogh : ∀ (guests : Fin n.succ → ℕ), ℕ), 
      (∀ (journalist : ℕ → ℕ → Prop), 
        (∀ x : Fin n.succ, ∃ y : Fin n.succ, journalist x y = false)
        → (∀ x : Fin n.succ, ∀ y : Fin n.succ, x ≠ y → journalist x y = true)
        → find_balogh (λ i, i) = n - 1)
      )
    :=
sorry

theorem minimum_questions_to_find_balogh (n : ℕ) :
  (∃ (min_questions : ∀ (guests : Fin n.succ → ℕ), ℕ),
    (∀ journalist : ℕ → ℕ → Prop, 
      (∀ x : Fin n.succ, ∃ y : Fin n.succ, journalist x y = false)
      → (∀ x : Fin n.succ, ∀ y : Fin n.succ, x ≠ y → journalist x y = true)
      → min_questions (λ i, i) = n - 1)
    )
    :=
sorry

end find_mr_balogh_with_n_minus_1_questions_minimum_questions_to_find_balogh_l311_311396


namespace number_of_subsets_of_intersection_l311_311037

def A : Set ℕ := {a, b, c, d, e}
def B : Set ℕ := {b, e, f}

theorem number_of_subsets_of_intersection : 
  (A ∩ B).toFinset.card = 4 :=
sorry

end number_of_subsets_of_intersection_l311_311037


namespace poly_div_difference_l311_311147

theorem poly_div_difference (P : polynomial ℤ) (a b : ℤ) : (a - b) ∣ (P.eval a - P.eval b) :=
sorry

end poly_div_difference_l311_311147


namespace cube_root_multiplication_l311_311197

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311197


namespace gcd_k_power_eq_k_minus_one_l311_311700

noncomputable def gcd_k_power (k : ℤ) : ℤ := 
  Int.gcd (k^1024 - 1) (k^1035 - 1)

theorem gcd_k_power_eq_k_minus_one (k : ℤ) : gcd_k_power k = k - 1 := 
  sorry

end gcd_k_power_eq_k_minus_one_l311_311700


namespace trigonometric_identity_l311_311446

-- Define variables
variables (α : ℝ) (hα : α ∈ Ioc 0 π) (h_tan : Real.tan α = 2)

-- The Lean statement
theorem trigonometric_identity :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 :=
sorry

end trigonometric_identity_l311_311446


namespace sum_of_divisors_of_prime_l311_311790

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311790


namespace find_f_2_l311_311476

def f (x : ℝ) : ℝ := x² + 1

theorem find_f_2 (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 + 1) : f 2 = 2 :=
by
  have hx1 : f (1 + 1) = 1^2 + 1 := h 1
  have hx2 : f 2 = 2 := by rw [←hx1]; exact hx1
  exact hx2

-- To ensure this is complete and make sense let's end with exactly the proof requirement
#check find_f_2

end find_f_2_l311_311476


namespace geometric_sequence_value_of_b_l311_311529

theorem geometric_sequence_value_of_b :
  ∀ (a b c : ℝ), 
  (∃ q : ℝ, q ≠ 0 ∧ a = 1 * q ∧ b = 1 * q^2 ∧ c = 1 * q^3 ∧ 4 = 1 * q^4) → 
  b = 2 :=
by
  intro a b c
  intro h
  obtain ⟨q, hq0, ha, hb, hc, hd⟩ := h
  sorry

end geometric_sequence_value_of_b_l311_311529


namespace truck_toll_is_correct_l311_311296

def toll (x : ℕ) : ℝ := 2.50 + 0.50 * (x - 2)

noncomputable def truck_toll : ℝ :=
  let front_axle_wheels := 2
  let remaining_wheels := 18 - front_axle_wheels
  let other_axles := remaining_wheels / 4
  let total_axles := 1 + other_axles
  toll total_axles

theorem truck_toll_is_correct : truck_toll = 4.00 :=
by
  let x := 5 -- total axles calculation
  have htoll : toll x = 2.50 + 0.50 * (x - 2) := rfl
  show toll x = 4.00
  rw [htoll]
  norm_num
  sorry

end truck_toll_is_correct_l311_311296


namespace proof_x1_x2_squared_l311_311488

theorem proof_x1_x2_squared (x1 x2 : ℝ) (h1 : (Real.exp 1 * x1)^x2 = (Real.exp 1 * x2)^x1)
  (h2 : 0 < x1) (h3 : 0 < x2) (h4 : x1 ≠ x2) : x1^2 + x2^2 > 2 :=
sorry

end proof_x1_x2_squared_l311_311488


namespace remainder_of_x50_div_x_minus_1_cubed_l311_311007

theorem remainder_of_x50_div_x_minus_1_cubed :
  (x : ℝ) → (x ^ 50) % ((x - 1) ^ 3) = 1225 * x ^ 2 - 2400 * x + 1176 := 
by
  sorry

end remainder_of_x50_div_x_minus_1_cubed_l311_311007


namespace find_g4_l311_311276

variables (g : ℝ → ℝ)

-- Given conditions
axiom condition1 : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1
axiom condition2 : g 4 + 3 * g (-2) = 35
axiom condition3 : g (-2) + 3 * g 4 = 5

theorem find_g4 : g 4 = -5 / 2 :=
by
  sorry

end find_g4_l311_311276


namespace num_valid_integers_l311_311507

-- Define the conditions:
def is_valid_integer (n : ℕ) : Prop :=
  n >= 3050 ∧ n <= 3200 ∧
  (let digits := (n.digits 10).reverse in
   digits.length = 4 ∧
   digits = digits.nodup ∧
   digits.sort (≤) = digits)

-- State the theorem:
theorem num_valid_integers : 
  {n : ℕ | is_valid_integer n}.to_finset.card = 21 :=
by sorry

end num_valid_integers_l311_311507


namespace sum_of_divisors_of_29_l311_311741

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311741


namespace find_y_square_divisible_by_three_between_50_and_120_l311_311423

theorem find_y_square_divisible_by_three_between_50_and_120 :
  ∃ (y : ℕ), y = 81 ∧ (∃ (n : ℕ), y = n^2) ∧ (3 ∣ y) ∧ (50 < y) ∧ (y < 120) :=
by
  sorry

end find_y_square_divisible_by_three_between_50_and_120_l311_311423


namespace evaluate_s_1010_mod_500_l311_311589

def q (x : ℤ) : ℤ := x^1010 + x^1009 + x^1008 + ... + x + 1

def s (x : ℤ) := (x^2 - 1) % q(x) 

theorem evaluate_s_1010_mod_500 (x : ℤ) : s(1010) % 500 = 55 :=
by
  sorry

end evaluate_s_1010_mod_500_l311_311589


namespace sequence_comparison_l311_311480

open Real

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

noncomputable def arithmetic_sequence (b1 d : ℝ) (n : ℕ) : ℝ :=
  b1 + (n - 1) * d

theorem sequence_comparison 
  (a1 q b1 d : ℝ) (h_a1_pos : 0 < a1) (h_q_pos : 0 < q)
  (h : a1 * q ^ 5 = b1 + 6 * d) :
  a1 * q ^ 2 + a1 * q ^ 8 ≥ 2 * a1 * q ^ 5 :=
begin
  sorry
end

end sequence_comparison_l311_311480


namespace sum_of_divisors_of_29_l311_311888

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311888


namespace monotonicity_and_symmetry_l311_311666

noncomputable def f : ℝ → ℝ := sorry

theorem monotonicity_and_symmetry (a b c : ℝ) (h1 : f (2 - a) = f a) 
  (h2 : ∀ x, (x > 1) → (x - 1) * (deriv f x) < 0)
  (h3 : f (log 3 2) = a)
  (h4 : f (log 5 2) = b)
  (h5 : f (log 2 5) = c) 
  (h6 : 0 < log 5 2) 
  (h7 : log 5 2 < log 3 2) 
  (h8 : log 3 2 < 1) 
  (h9 : 1 < 2) 
  (h10 : 2 < log 2 5) : c < b ∧ b < a := 
sorry

end monotonicity_and_symmetry_l311_311666


namespace simplify_cuberoot_product_l311_311227

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311227


namespace simplify_cubed_roots_l311_311214

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311214


namespace problem1_problem2_l311_311403

-- Proof problem 1

theorem problem1 : 
  (sqrt 8 - sqrt 24) / sqrt 2 + abs (1 - sqrt 3) = 1 - sqrt 3 := 
sorry

-- Proof problem 2

theorem problem2 : 
  (1 / 2)⁻¹ - 2 * cos (real.pi / 6) + abs (2 - sqrt 3) - (2 * sqrt 2 + 1)^0 = 3 - 2 * sqrt 3 := 
sorry

end problem1_problem2_l311_311403


namespace sum_of_divisors_29_l311_311822

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311822


namespace integer_solutions_count_l311_311272

theorem integer_solutions_count : 
  {p : ℤ × ℤ // |p.1| + |p.2| = 3}.to_finset.card = 12 :=
sorry

end integer_solutions_count_l311_311272


namespace total_players_l311_311916

theorem total_players (K Kho_only Both : Nat) (hK : K = 10) (hKho_only : Kho_only = 30) (hBoth : Both = 5) : 
  (K - Both) + Kho_only + Both = 40 := by
  sorry

end total_players_l311_311916


namespace f_log3_5_l311_311471

theorem f_log3_5 :
  let f : ℝ → ℝ := λ x, if 0 < x ∧ x < 1 then 3^x - 1 else if (x + 2) % 2 == 0 then f((x + 2) % 2) else sorry,
  (∀ x, f(-x) = -f(x)) →  -- odd function condition
  (∀ x y, (f(x + y) = f x)) →  -- periodicity condition
  f (log 3 5) = -4/5 :=
by
  assume f odd periodic,
  -- complete with the necessary assumptions based on the original problem
  sorry 

end f_log3_5_l311_311471


namespace sum_of_coordinates_B_l311_311178

def midpoint (A B M : (ℝ × ℝ)) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem sum_of_coordinates_B :
  ∃ B : (ℝ × ℝ), 
  (let A : (ℝ × ℝ) := (8, 4);
       M : (ℝ × ℝ) := (4, 4) in
   midpoint A B M ∧ B.1 + B.2 = 4) :=
sorry

end sum_of_coordinates_B_l311_311178


namespace Agnes_birth_year_l311_311339

theorem Agnes_birth_year (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9)
  (h3 : (11 * x + 2 * y + x * y = 92)) : 1948 = 1900 + (10 * x + y) :=
sorry

end Agnes_birth_year_l311_311339


namespace sum_of_divisors_29_l311_311831

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311831


namespace sum_of_divisors_of_prime_29_l311_311842

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311842


namespace square_side_length_l311_311270

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ s : ℝ, s = 2 :=
by 
  sorry

end square_side_length_l311_311270


namespace modem_B_download_time_l311_311565

theorem modem_B_download_time
    (time_A : ℝ) (speed_ratio : ℝ) 
    (h1 : time_A = 25.5) 
    (h2 : speed_ratio = 0.17) : 
    ∃ t : ℝ, t = 110.5425 := 
by
  sorry

end modem_B_download_time_l311_311565


namespace longest_segment_in_cylinder_l311_311367

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧ d = Real.sqrt (h^2 + (2*r)^2) :=
by
  sorry

end longest_segment_in_cylinder_l311_311367


namespace area_of_parallelogram_l311_311608

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l311_311608


namespace power_minus_self_even_l311_311179

theorem power_minus_self_even (a n : ℕ) (ha : 0 < a) (hn : 0 < n) : Even (a^n - a) := by
  sorry

end power_minus_self_even_l311_311179


namespace all_white_possible_l311_311374

theorem all_white_possible (grid : array (fin 98) (array (fin 98) bool)) :
  (∃ operations : list (fin 98 × fin 98 × nat × nat), ∀ i j, grid[i][j] = ff)
    := by sorry

end all_white_possible_l311_311374


namespace sum_of_divisors_29_l311_311870

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311870


namespace blue_paint_needed_l311_311395

theorem blue_paint_needed (F B : ℝ) :
  (6/9 * F = 4/5 * (F * 1/3 + B) → B = 1/2 * F) :=
sorry

end blue_paint_needed_l311_311395


namespace angle_Q_measure_in_triangle_PQR_l311_311108

theorem angle_Q_measure_in_triangle_PQR (angle_R angle_Q angle_P : ℝ) (h1 : angle_P = 3 * angle_R) (h2 : angle_Q = angle_R) (h3 : angle_R + angle_Q + angle_P = 180) : angle_Q = 36 :=
by {
  -- Placeholder for the proof, which is not required as per the instructions
  sorry
}

end angle_Q_measure_in_triangle_PQR_l311_311108


namespace diagonal_intersects_l311_311345

theorem diagonal_intersects (n : ℕ) (hne : n ≥ 403) :
  ∃ d ∈ (finset.diagonals n), (finset.diagonal_intersections d) ≥ 10000 :=
sorry

end diagonal_intersects_l311_311345


namespace circle_tangent_and_radius_l311_311156

theorem circle_tangent_and_radius (R1 R2 d : ℝ) :
  let Q : Prop := ∃ (O1 O2 : ℝ), 
    (∀ (P : Prop), 
      let A := ∃ (C : Prop), true,
      let B := ∃ (D : Prop), true,
      exists circle P C D : ℝ,
      -- Proving tangency
      (circle_tangent : tangent circle P C D (circle O1 R1)), 
      (circle_tangent: tangent circle P C D (circle O2 R2)),
      -- Radius determination
      (R : ℝ), 
      (R = (d ^ 2 * (R1 + R2)) / (2 * sqrt(R1 * R2) * (d + sqrt (R1 * R2))) ∨ 
       R = (d ^ 2 * (R1 + R2)) / (2 * sqrt(R1 * R2) * |d - sqrt (R1 * R2)|))
      ) sorry

end circle_tangent_and_radius_l311_311156


namespace negation_proposition_l311_311674

-- Define the original proposition
def unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) → (x1 = x2)

-- Define the negation of the proposition
def negation_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ¬ unique_solution a b h

-- Define a proposition for "no unique solution"
def no_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∃ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) ∧ (x1 ≠ x2)

-- The Lean 4 statement
theorem negation_proposition (a b : ℝ) (h : a ≠ 0) :
  negation_unique_solution a b h :=
sorry

end negation_proposition_l311_311674


namespace h_eq_20_at_y_eq_4_l311_311654

noncomputable def k (y : ℝ) : ℝ := 40 / (y + 5)

noncomputable def h (y : ℝ) : ℝ := 4 * (k⁻¹ y)

theorem h_eq_20_at_y_eq_4 : h 4 = 20 := 
by 
  -- Insert proof here
  sorry

end h_eq_20_at_y_eq_4_l311_311654


namespace badger_hid_35_l311_311539

-- Define the variables
variables (h_b h_f x : ℕ)

-- Define the conditions based on the problem
def badger_hides : Prop := 5 * h_b = x
def fox_hides : Prop := 7 * h_f = x
def fewer_holes : Prop := h_b = h_f + 2

-- The main theorem to prove the badger hid 35 walnuts
theorem badger_hid_35 (h_b h_f x : ℕ) :
  badger_hides h_b x ∧ fox_hides h_f x ∧ fewer_holes h_b h_f → x = 35 :=
by sorry

end badger_hid_35_l311_311539


namespace triangle_EDF_is_equivalent_l311_311123

-- Defining the triangle ABC with the given conditions
variables (A B C D E F : Type) [IsTriangle A B C]
variables (AB AC BC BD BE BF : Real)
variables (angle_A : Real)

-- Given conditions
def condition1 := AB = AC
def condition2 := angle_A = 100
def point_D_on_side_BC := D ∈ LineBetween(B, C)
def point_E_on_side_AC := E ∈ LineBetween(A, C)
def point_F_on_side_AB := F ∈ LineBetween(A, B)
def condition4 := DE = DF
def condition5 := BF = BD

-- Target theorem
theorem triangle_EDF_is_equivalent :
    condition1 → condition2 → point_D_on_side_BC → point_E_on_side_AC → point_F_on_side_AB → condition4 → condition5 → 
    ∃ (angle_EDF : Real), angle_EDF = 40 :=
by
  sorry

end triangle_EDF_is_equivalent_l311_311123


namespace least_integer_square_double_condition_l311_311703

theorem least_integer_square_double_condition :
  ∃ x : ℤ, x^2 = 2 * x + 48 ∧ ∀ y : ℤ, y^2 = 2 * y + 48 → x ≤ y :=
begin
  sorry
end

end least_integer_square_double_condition_l311_311703


namespace oranges_comparison_l311_311399

theorem oranges_comparison (M B : ℕ) (h1 : M = 12) (h2 : B = 12) : M - B = 0 :=
by
  rw [h1, h2]
  norm_num
  sorry

end oranges_comparison_l311_311399


namespace distribution_of_mountaineers_l311_311298

theorem distribution_of_mountaineers :
  let total_mountaineers := 10
  let familiar_with_trails := 4
  let groups := 2
  let required_familiar_per_group := 2
  (nat.choose familiar_with_trails required_familiar_per_group) * 
  (nat.choose (total_mountaineers - familiar_with_trails) 
              (total_mountaineers / groups - required_familiar_per_group)) = 60 :=
begin
  sorry
end

end distribution_of_mountaineers_l311_311298


namespace square_side_length_l311_311269

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ s : ℝ, s = 2 :=
by 
  sorry

end square_side_length_l311_311269


namespace tagged_fish_proportion_l311_311537

def total_fish_in_pond : ℕ := 750
def tagged_fish_first_catch : ℕ := 30
def fish_second_catch : ℕ := 50
def tagged_fish_second_catch := 2

theorem tagged_fish_proportion :
  (tagged_fish_second_catch : ℤ) * (total_fish_in_pond : ℤ) = (tagged_fish_first_catch : ℤ) * (fish_second_catch : ℤ) :=
by
  -- The statement should reflect the given proportion:
  -- T * 750 = 30 * 50
  -- Given T = 2
  sorry

end tagged_fish_proportion_l311_311537


namespace sum_of_divisors_of_29_l311_311856

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311856


namespace y_axis_point_coordinates_l311_311522

theorem y_axis_point_coordinates (m : ℤ) (h : m + 2 = 0) : (m + 2 = 0) → (m + 2 = 0) → P := 
by
  assume
  rw h
  sorry

end y_axis_point_coordinates_l311_311522


namespace initial_milk_amount_l311_311656

theorem initial_milk_amount (d : ℚ) (r : ℚ) (T : ℚ) 
  (hd : d = 0.4) 
  (hr : r = 0.69) 
  (h_remaining : r = (1 - d) * T) : 
  T = 1.15 := 
  sorry

end initial_milk_amount_l311_311656


namespace range_of_a_l311_311502

noncomputable def problem_set (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 1) * (a - x) > 0}

theorem range_of_a (a : ℝ) :
  let A := problem_set a in
  (2 ∈ A) ∧ (3 ∉ A) ↔ a ∈ (Set.Icc ((1 : ℝ) / 3) (1 / 2)) ∪ (Set.Ioc 2 3) :=
by
  sorry

end range_of_a_l311_311502


namespace am_gm_inequality_l311_311591

-- Let's define the problem statement
theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end am_gm_inequality_l311_311591


namespace find_line_eq_l311_311598

noncomputable def line_eq (a : ℝ) (m : ℝ) (b : ℝ) : Prop :=
  let P : ℝ × ℝ := (-b / (1 + m^2), b - (m * b) / (1 + m^2))
  let Q : ℝ × ℝ := (b / (1 + m^2), b + (m * b) / (1 + m^2))
  let M : ℝ × ℝ := ((-b + sqrt(b^2 + a^2)) / (1 - m^2), b - (m * sqrt(b^2 + a^2)) / (1 - m^2))
  let N : ℝ × ℝ := (b / (1 - m^2), b + (m * sqrt(b^2 + a^2)) / (1 - m^2))
  abs((P.1 - Q.1)) = (1 / 3) * abs((M.1 - N.1)) ∧ 
  ((y = (2 * sqrt(5) / 5) * x) ∨ (y = -(2 * sqrt(5) / 5) * x) ∨ (y = (2 * sqrt(5) / 5) * a) ∨ (y = -(2 * sqrt(5) / 5) * a))

-- The theorem to prove the desired result.
theorem find_line_eq (a : ℝ) : 
  ∃ m b, line_eq a m b :=
sorry

end find_line_eq_l311_311598


namespace sum_of_divisors_of_prime_l311_311786

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311786


namespace sum_of_first_13_terms_l311_311549

theorem sum_of_first_13_terms (a : ℕ → ℤ) [is_arithmetic_sequence : ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d] 
  (h : 2 * (a 1 + a 4 + a 7) + 3 * (a 9 + a 11) = 24) : 
  (finset.range 13).sum (λ n, a (n+1)) = 26 :=
sorry

end sum_of_first_13_terms_l311_311549


namespace probability_one_side_is_side_of_decagon_l311_311982

theorem probability_one_side_is_side_of_decagon :
  let decagon_vertices := 10
  let total_triangles := Nat.choose decagon_vertices 3
  let favorable_one_side :=
    decagon_vertices * (decagon_vertices - 3) / 2
  let favorable_two_sides := decagon_vertices
  let favorable_outcomes := favorable_one_side + favorable_two_sides
  let probability := favorable_outcomes / total_triangles
  total_triangles = 120 ∧ favorable_outcomes = 60 ∧ probability = 1 / 2 := 
by
  sorry

end probability_one_side_is_side_of_decagon_l311_311982


namespace find_ab_l311_311594

theorem find_ab (a b c : ℕ) (H_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (H_b : b = 1) (H_ccb : (10 * a + b)^2 = 100 * c + 10 * c + b) (H_gt : 100 * c + 10 * c + b > 300) : (10 * a + b) = 21 :=
by
  sorry

end find_ab_l311_311594


namespace general_term_of_sequence_l311_311293

theorem general_term_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n, S n = 2^n - 3)
  (hSn : ∀ n, S (n + 1) = S n + a (n + 1))
  : ∀ n, a n = 
    if n = 1 then -1 
    else if n ≥ 2 then 2^(n - 1) 
    else 0 :=
by
  intro n
  split_ifs
  case h_1 =>
    sorry
  case h_2 =>
    sorry
  case h_3 =>
    sorry

end general_term_of_sequence_l311_311293


namespace minimal_degree_polynomial_l311_311006

noncomputable def minimal_poly_with_roots : Polynomial ℚ :=
  Polynomial.X^4 - 8 * Polynomial.X^3 + 24 * Polynomial.X^2 - 20 * Polynomial.X + 4

theorem minimal_degree_polynomial :
  ∃ P : Polynomial ℚ,
    (P.leadingCoeff = 1) ∧
    (P.degree = minimal_poly_with_roots.degree) ∧
    (∀ x, 
       (P.eval x = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 ∨ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6))) :=
begin
  use minimal_poly_with_roots,
  split,
  { sorry },  -- Prove leading coefficient is 1
  split,
  { sorry },  -- Prove the same degree
  { sorry }   -- Prove all roots are included
end

end minimal_degree_polynomial_l311_311006


namespace sum_of_divisors_29_l311_311872

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311872


namespace min_distance_l311_311590

noncomputable def z : ℂ := sorry
noncomputable def w : ℂ := sorry
def a : ℂ := 2 + 1*I
def b : ℂ := -3 - 4*I

-- Conditions
axiom condition1 : ∥z - a∥ = 2
axiom condition2 : ∥w - b∥ = 4

-- Correct answer
theorem min_distance : ∀ z w : ℂ, ∥z - a∥ = 2 → ∥w - b∥ = 4 → ∥z - w∥ = 5*Real.sqrt 2 - 6 :=
sorry

end min_distance_l311_311590


namespace new_person_weight_l311_311331

variable (W : ℝ)
variable (n : ℝ := 8)
variable (avg_increase : ℝ := 2.5)
variable (replaced_weight : ℝ := 60)
variable (total_increase := n * avg_increase)

/-- Prove that the weight of the new person is 80 kg --/
theorem new_person_weight : W = replaced_weight + total_increase → W = 80 := by
  intro h
  calc
    W = replaced_weight + total_increase : h
    ... = 60 + 20 : by
      simp [replaced_weight, total_increase]
    ... = 80 : by
      simp

end new_person_weight_l311_311331


namespace mangoes_per_jar_is_correct_l311_311132

-- Define the conditions
def total_mangoes : ℕ := 54
def ripe_fraction : ℚ := 1/3
def unripe_fraction : ℚ := 2/3
def kept_unripe_mangoes : ℕ := 16
def jars : ℕ := 5

-- Define the required quantities based on conditions
def ripe_mangoes : ℕ := (ripe_fraction * total_mangoes).toNat
def unripe_mangoes : ℕ := (unripe_fraction * total_mangoes).toNat
def given_unripe_mangoes : ℕ := unripe_mangoes - kept_unripe_mangoes
def mangoes_per_jar : ℕ := given_unripe_mangoes / jars

-- State the theorem and proof (incomplete, with sorry)
theorem mangoes_per_jar_is_correct : mangoes_per_jar = 4 := by
  sorry

end mangoes_per_jar_is_correct_l311_311132


namespace simplify_sqrt_l311_311190

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311190


namespace mr_sanchez_bought_less_rope_l311_311574

theorem mr_sanchez_bought_less_rope :
  (∃ last_week this_week : ℕ, last_week = 6 ∧ this_week = 96 / 12 ∧ abs (last_week - this_week) = 2) :=
by
  let last_week := 6
  let this_week := 96 / 12
  have : abs (last_week - this_week) = 2 := sorry
  exact ⟨last_week, this_week, rfl, rfl, this⟩

end mr_sanchez_bought_less_rope_l311_311574


namespace multiset_plus_equivalence_l311_311910

theorem multiset_plus_equivalence (n : ℕ) (h : n ≥ 2) :
  (∀ (A B : Finset ℚ), 
    A.card = n ∧ B.card = n ∧ A ∩ B = ∅ ∧
    (multiset.cons_map (λ (a b : ℚ), a + b) A.to_multiset A).filter (λ abab, a < b) = 
    (multiset.cons_map (λ (b b' : ℚ), b + b') B.to_multiset B).filter (λ bb'b', b < b')
  → ∃ (k : ℕ), n = 2^k) 
  ∧ 
  (∀ k : ℕ, ∃ A B : Finset ℚ, 
    n = 2^k ∧ A.card = n ∧ B.card = n ∧ A ∩ B = ∅ ∧
    (multiset.cons_map (λ (a b : ℚ), a + b) A.to_multiset A).filter (λ abab, a < b) = 
    (multiset.cons_map (λ (b b' : ℚ), b + b') B.to_multiset B).filter (λ bb'b', b < b')) :=
sorry

end multiset_plus_equivalence_l311_311910


namespace sum_of_angles_in_triangle_l311_311473

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

axiom acute_angles (A B C : ℝ) : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90
axiom tan_values : tan A = 1 ∧ tan B = 2 ∧ tan C = 3

theorem sum_of_angles_in_triangle
  (hA : 0 < A ∧ A < 90) (hB : 0 < B ∧ B < 90) (hC : 0 < C ∧ C < 90)
  (h_tan : tan A = 1 ∧ tan B = 2 ∧ tan C = 3) : A + B + C = 180 := 
begin 
  sorry 
end

end sum_of_angles_in_triangle_l311_311473


namespace parallelogram_area_proof_l311_311627

noncomputable def parallelogram_area : ℝ :=
  let angle_rad := (150 * real.pi / 180)  -- converting degrees to radians
  let a := 10                              -- length of one side
  let b := 20                              -- length of another side
  let height := a * real.sqrt(3) / 2       -- height from 30-60-90 triangle properties
  b * height

theorem parallelogram_area_proof : parallelogram_area = 100 * real.sqrt(3) := by
  sorry

end parallelogram_area_proof_l311_311627


namespace distinct_numbers_in_each_row_l311_311917

noncomputable def construct_table (a : ℕ) : ℕ → list ℕ
| 0       := [a]
| (n + 1) := (construct_table n).bind (λ x, [x^2, x + 1])

theorem distinct_numbers_in_each_row (a : ℕ) (n : ℕ) :
  (construct_table a n).nodup :=
sorry

end distinct_numbers_in_each_row_l311_311917


namespace michael_total_robots_l311_311961

variables (Tom_robots Michael_multiplier Robots_per_given Robots_received Michael_robots_original Michael_robots_total : ℕ)

def michael_multiplier := 4
def robots_per_given := 3
def tom_robots := 12

-- Calculate Michael's original robots
def michael_robots_original := michael_multiplier * tom_robots

-- Calculate Robots given to James
def robots_received := tom_robots / robots_per_given

-- Calculate total Michael's robots
def michael_robots_total := michael_robots_original + robots_received

theorem michael_total_robots : michael_robots_total = 52 :=
by
  rw [michael_robots_original, robots_received, michael_robots_total]
  sorry

end michael_total_robots_l311_311961


namespace find_a2010_l311_311943

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n, a (n + 1) = 1 - 1 / (a n)

theorem find_a2010 (a : ℕ → ℝ) (h : sequence a) : a 2010 = 2 := 
sorry

end find_a2010_l311_311943


namespace sum_of_divisors_of_29_l311_311876

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311876


namespace dali_prints_consecutive_probability_l311_311163

-- Definitions of conditions
def total_pieces_of_art : ℕ := 12
def dali_prints : ℕ := 4

-- Probability calculation formula
def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else List.product (List.range n.succ)

theorem dali_prints_consecutive_probability 
  (h1: total_pieces_of_art = 12) 
  (h2: dali_prints = 4) :
  let arrangements_with_block := factorial 9 * factorial 4
  let total_arrangements := factorial 12
  arrangements_with_block / total_arrangements = (1 : ℚ) / 55 :=
by
  sorry

end dali_prints_consecutive_probability_l311_311163


namespace mother_rides_more_than_john_l311_311964

theorem mother_rides_more_than_john:
  ∃ (X : ℕ), 
    let B := 17 in
    let J := 2 * B in
    let M := J + X in
    B + J + M = 95 ∧ X = 10 :=
by
  sorry

end mother_rides_more_than_john_l311_311964


namespace relationship_among_a_b_c_l311_311153

noncomputable def a : ℝ := Real.log (Real.tan (70 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def b : ℝ := Real.log (Real.sin (25 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def c : ℝ := (1 / 2) ^ Real.cos (25 * Real.pi / 180)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  -- proofs would go here
  sorry

end relationship_among_a_b_c_l311_311153


namespace total_minutes_last_weekend_l311_311576

-- Define the given conditions
def Lena_hours := 3.5 -- Lena played for 3.5 hours
def Brother_extra_minutes := 17 -- Brother played 17 minutes more than Lena

-- Define the conversion from hours to minutes
def hours_to_minutes (hours : ℝ) : ℕ := (hours * 60).to_nat

-- Total minutes Lena played
def Lena_minutes := hours_to_minutes Lena_hours

-- Total minutes her brother played
def Brother_minutes := Lena_minutes + Brother_extra_minutes

-- Define the total minutes played together
def total_minutes_played := Lena_minutes + Brother_minutes

-- The proof statement (with an assumed proof)
theorem total_minutes_last_weekend : total_minutes_played = 437 := 
by 
  sorry

end total_minutes_last_weekend_l311_311576


namespace even_non_friendly_vertices_l311_311454

-- Define the structures and properties
structure Polygon (n : ℕ) :=
(vertices : Fin n → (ℝ × ℝ))
(adjacent_perpendicular : ∀ i, ∀ j : Fin n, vertices (i + 1) = vertices j → (vertices i.fst = vertices (i + 1).fst ∨ vertices i.snd = vertices (i + 1).snd))

-- Define non-friendly vertices condition
def non_friendly (P : Polygon n) (v₁ v₂ : Fin n) : Prop :=
∃ l₁ l₂ : ℝ, (angle_bisector P.vertices v₁ v₂).1 l₁ = 0 ∧ (angle_bisector P.vertices v₁ v₂).2 l₂ = 0

-- Theorem statement
theorem even_non_friendly_vertices (P : Polygon n) (v : Fin n) : 
  ∃ m : ℕ, 2 * m = (Finset.filter (λ v', non_friendly P v v') (Finset.univ) ).card :=
sorry

end even_non_friendly_vertices_l311_311454


namespace find_N_l311_311512

noncomputable def sum_of_sequence : ℤ :=
  985 + 987 + 989 + 991 + 993 + 995 + 997 + 999

theorem find_N : ∃ (N : ℤ), 8000 - N = sum_of_sequence ∧ N = 64 := by
  use 64
  -- The actual proof steps will go here
  sorry

end find_N_l311_311512


namespace very_odd_power_l311_311185

def is_very_odd (A : Matrix (Fin n) (Fin n) ℤ) : Prop :=
  ∀ S : Finset (Fin n), S.Nonempty → (A.minor id id).det % 2 = 1

theorem very_odd_power {n : ℕ} (A : Matrix (Fin n) (Fin n) ℤ) (k : ℕ) (hA : is_very_odd A) (hk : 1 ≤ k) :
  is_very_odd (A^k) :=
sorry

end very_odd_power_l311_311185


namespace joan_picked_total_apples_l311_311568

theorem joan_picked_total_apples (given_apples_to_Melanie : ℕ) (apples_left_with_Joan : ℕ) :
  given_apples_to_Melanie = 27 → apples_left_with_Joan = 16 → (given_apples_to_Melanie + apples_left_with_Joan) = 43 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
sorry

end joan_picked_total_apples_l311_311568


namespace sum_of_divisors_of_29_l311_311776

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311776


namespace triangle_area_l311_311146

/-- Define the vectors a and b -/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (4, 5)

/-- Calculate the area of the triangle formed by 0, a, and b -/
theorem triangle_area : 
  let det := |(a.1 * b.2 - a.2 * b.1)| in
  (1 / 2) * det = 3.5 :=
by
  let det := abs ((a.1 * b.2) - (a.2 * b.1))
  have h_det : det = 7 := by sorry
  rw [h_det]
  norm_num

end triangle_area_l311_311146


namespace product_of_two_numbers_l311_311670

theorem product_of_two_numbers (a b : ℤ) (h1 : Int.gcd a b = 10) (h2 : Int.lcm a b = 90) : a * b = 900 := 
sorry

end product_of_two_numbers_l311_311670


namespace value_of_x_plus_y_l311_311261

theorem value_of_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x / 3 = y^2) (h2 : x / 9 = 9 * y) : 
  x + y = 2214 :=
sorry

end value_of_x_plus_y_l311_311261


namespace union_of_A_and_B_l311_311065

-- Definitions based on the conditions in the problem
def A : Set ℝ := {x | abs (x + 1) < 2}
def B : Set ℝ := {x | -x^2 + 2x + 3 ≥ 0}

-- Statement of the problem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 < x ∧ x ≤ 3} :=
by
  sorry

end union_of_A_and_B_l311_311065


namespace intersection_M_N_l311_311068

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 10 }
def N : Set ℝ := { x | x > 7 ∨ x < 1 }
def MN_intersection : Set ℝ := { x | (-1 ≤ x ∧ x < 1) ∨ (7 < x ∧ x ≤ 10) }

theorem intersection_M_N :
  M ∩ N = MN_intersection :=
by
  sorry

end intersection_M_N_l311_311068


namespace find_replaced_watermelon_weight_l311_311661

-- Define the initial conditions
def initial_average_weight : ℕ → ℝ
| 10 := 4.2

def new_average_weight : ℕ → ℝ :=
| 10 := 4.4

-- Define the total weight of the original 10 watermelons
def total_weight_initial (n : ℕ) (avg : ℝ) : ℝ :=
  n * avg

-- Define the total weight after replacing one watermelon
def total_weight_new (n : ℕ) (initial_weight : ℝ) (replaced_weight : ℝ) (new_weight : ℝ) : ℝ :=
  initial_weight - replaced_weight + new_weight

-- The weight of the newly replaced watermelon
def new_weight : ℝ := 5

-- Total weight equation to solve for the replaced watermelon's weight
theorem find_replaced_watermelon_weight (W : ℝ) :
  (total_weight_initial 10 (initial_average_weight 10) - W + new_weight) = total_weight_new 10 (total_weight_initial 10 (initial_average_weight 10)) W new_weight
  → W = 3 := 
begin
  -- The theorem setup implies the translation of the problem setup.
  sorry
end

end find_replaced_watermelon_weight_l311_311661


namespace find_b_l311_311125

-- Definitions
variables (a b c : ℝ) (B : ℝ)

-- Conditions
def is_triangle (B : ℝ) := B = real.pi / 3
def sum_sides (a b c : ℝ) := a + c = 2 * b
def product_sides (a c : ℝ) := a * c = 6

-- Proof problem statement
theorem find_b (a b c : ℝ) (B : ℝ) 
  (h1 : is_triangle B) 
  (h2 : sum_sides a b c) 
  (h3 : product_sides a c) : 
  b = real.sqrt 6 :=
sorry

end find_b_l311_311125


namespace initial_population_l311_311350

theorem initial_population (P : ℝ) (h : 0.78435 * P = 4500) : P = 5738 := 
by 
  sorry

end initial_population_l311_311350


namespace sum_a_b_l311_311055

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if x >= 0 then (sqrt x + 3)
else (a * x + b)

theorem sum_a_b (a b : ℝ)
  (h1 : ∀ x1 : ℝ, ∃! x2 : ℝ, f x1 a b = f x2 a b)
  (h2 : f (2 * a) a b = f (3 * b) a b) :
  a + b = -sqrt 6 / 2 + 3 :=
sorry

end sum_a_b_l311_311055


namespace multiple_proof_l311_311260

-- Statement and conditions
theorem multiple_proof (a b : ℕ) (h1 : ∃ k1 : ℕ, a = 4 * k1) (h2 : ∃ k2 : ℕ, b = 8 * k2) :
  (∃ k3 : ℕ, b = 4 * k3) ∧
  (∃ k4 : ℕ, a - b = 4 * k4) ∧
  ¬(∃ k5 : ℕ, a - b = 8 * k5) ∧
  (∃ k6 : ℕ, a - b = 2 * k6) :=
begin
  sorry
end

end multiple_proof_l311_311260


namespace probability_earning_700_is_7_over_125_l311_311606

noncomputable def probability_earning_700 : ℚ :=
  let outcomes := [(0:ℚ), 200, 300, 100, 1000] in
  let total_possibilities := (outcomes.product outcomes).product outcomes in
  let successful_outcomes := total_possibilities.filter (λ s,
      s.1.1 + s.1.2 + s.2 = 700) in
  (successful_outcomes.length : ℚ) / total_possibilities.length

theorem probability_earning_700_is_7_over_125 : probability_earning_700 = 7 / 125 := by
  sorry

end probability_earning_700_is_7_over_125_l311_311606


namespace strictly_increasing_5_digit_numbers_count_l311_311351

theorem strictly_increasing_5_digit_numbers_count : 
  ∃ n : ℕ, n = 126 ∧ 
  (∀ (digits : finset ℕ), digits ⊆ (finset.range 10).erase 0 ∧ digits.card = 5 → digits.card.factorial = 5! 
  ∧ (finset.card (finset.range 9).choose 5 = n)) := 
sorry

end strictly_increasing_5_digit_numbers_count_l311_311351


namespace translate_parabola_l311_311694

theorem translate_parabola (x y : ℝ) :
  (y = 2 * x^2 + 3) →
  (∃ x y, y = 2 * (x - 3)^2 + 5) :=
sorry

end translate_parabola_l311_311694


namespace calculate_exponent_l311_311515

theorem calculate_exponent (x : ℤ) (y : ℝ) (h : y = 3.456789) (d : y.digitsBeforeDec = 6) : (x = -4) → (∃ y', (10^x * y')^12 has_digits_after_decimal 24) := 
sorry

end calculate_exponent_l311_311515


namespace sum_of_divisors_29_l311_311832

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311832


namespace simplify_cuberoot_product_l311_311236

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311236


namespace profit_calculation_l311_311370

def actors_cost : ℕ := 1200
def people_count : ℕ := 50
def cost_per_person : ℕ := 3
def food_cost : ℕ := people_count * cost_per_person
def total_cost_actors_food : ℕ := actors_cost + food_cost
def equipment_rental_cost : ℕ := 2 * total_cost_actors_food
def total_movie_cost : ℕ := total_cost_actors_food + equipment_rental_cost
def movie_sale_price : ℕ := 10000
def profit : ℕ := movie_sale_price - total_movie_cost

theorem profit_calculation : profit = 5950 := by
  sorry

end profit_calculation_l311_311370


namespace tension_at_point_E_l311_311683

noncomputable def tension_force (m g CD AB : ℝ) : ℝ :=
  let T2 := m * g * (1 / 4)
  let T1 := m * g * (3 / 4)
  let T0 := 4 * (T1 * (1 / 2) - T2)
  T0

theorem tension_at_point_E :
  let m := 3
  let g := 10
  let CD := 1 -- choosing 1 as a scale factor normalization for ease of generalization
  let AB := 1 -- choosing 1 as a scale factor normalization for ease of generalization
  tension_force m g CD AB = 15 :=
by
  let m := 3
  let g := 10
  let CD := 1
  let AB := 1
  let T0 := tension_force m g CD AB
  simp [tension_force] at T0
  rw [real_div_eq_mul_one_div] at T0
  simp at T0
  sorry -- Proof that T0 can be calculated as 15 N

end tension_at_point_E_l311_311683


namespace correct_operation_l311_311319

theorem correct_operation :
  (sqrt 8) * (sqrt (9/2)) = 6 :=
by sorry

end correct_operation_l311_311319


namespace total_minutes_last_weekend_l311_311577

-- Define the given conditions
def Lena_hours := 3.5 -- Lena played for 3.5 hours
def Brother_extra_minutes := 17 -- Brother played 17 minutes more than Lena

-- Define the conversion from hours to minutes
def hours_to_minutes (hours : ℝ) : ℕ := (hours * 60).to_nat

-- Total minutes Lena played
def Lena_minutes := hours_to_minutes Lena_hours

-- Total minutes her brother played
def Brother_minutes := Lena_minutes + Brother_extra_minutes

-- Define the total minutes played together
def total_minutes_played := Lena_minutes + Brother_minutes

-- The proof statement (with an assumed proof)
theorem total_minutes_last_weekend : total_minutes_played = 437 := 
by 
  sorry

end total_minutes_last_weekend_l311_311577


namespace minimum_value_of_quadratic_l311_311705

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 + 12 * x + 5

-- Theorem that states the minimum value of the quadratic function
theorem minimum_value_of_quadratic : ∃ y : ℝ, (∀ x : ℝ, quadratic x ≥ y) ∧ y = -31 :=
by
  use -31
  split
  {
    intros x
    unfold quadratic
    calc
      x^2 + 12 * x + 5
          = (x + 6)^2 - 31 : by ring
      ... ≥ 0 - 31 : by apply sub_nonneg_of_le (pow_two_nonneg _)
      ... = -31 : by linarith
  }
  {
    refl
  }

end minimum_value_of_quadratic_l311_311705


namespace scientific_notation_of_35000000_l311_311954

theorem scientific_notation_of_35000000 :
  (35_000_000 : ℕ) = 3.5 * 10^7 := by
  sorry

end scientific_notation_of_35000000_l311_311954


namespace inequality_solution_set_l311_311291

theorem inequality_solution_set (x : ℝ) :
  ∀ x, 
  (x^2 * (x + 1) / (-x^2 - 5 * x + 6) <= 0) ↔ (-6 < x ∧ x <= -1) ∨ (x = 0) ∨ (1 < x) :=
by
  sorry

end inequality_solution_set_l311_311291


namespace sum_of_divisors_of_29_l311_311777

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311777


namespace sum_of_divisors_of_29_l311_311758

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311758


namespace scientific_notation_35_million_l311_311951

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 : Float) ^ 7 := 
by
  sorry

end scientific_notation_35_million_l311_311951


namespace exists_filled_4x4_table_l311_311564

def integer_grid (m n : ℕ) := array (array ℤ n) m

def is_3x3_subgrid_sum_negative (grid : integer_grid 4 4) (i j : ℕ) : Prop :=
  (i ≤ 1 ∧ j ≤ 1) ∧
  (((List.nth (grid to_list) i).get_to_list.take 3).sum + 
  ((List.nth (grid.to_list) (i + 1)).get_to_list.take 3).sum +
  ((List.nth (grid.to_list) (i + 2)).get_to_list.take 3).sum < 0)

def is_total_grid_sum_positive (grid : integer_grid 4 4) : Prop :=
  grid.to_list.map (λ row => row.to_list.sum).sum > 0

theorem exists_filled_4x4_table : 
  ∃ (grid : integer_grid 4 4), 
  is_total_grid_sum_positive grid ∧ 
  ∀ (i j : ℕ), is_3x3_subgrid_sum_negative grid i j := 
sorry

end exists_filled_4x4_table_l311_311564


namespace total_cost_correct_l311_311336

def sandwich_cost : ℝ := 2.44
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_correct :
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 8.36 := by
  sorry

end total_cost_correct_l311_311336


namespace optimal_start_day_for_vacation_to_maximize_sunny_days_l311_311604

-- Definitions based on conditions
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

def isSunny : Day → Prop
| Day.Monday      := False
| Day.Tuesday     := True
| Day.Wednesday   := True
| Day.Thursday    := True
| Day.Friday      := True
| Day.Saturday    := False
| Day.Sunday      := True

def vacation_days : ℕ := 30

def full_weeks (days : ℕ) : ℕ := days / 7
def remaining_days (days : ℕ) : ℕ := days % 7

def sunny_days_in_full_week : ℕ := 5 -- Tuesday, Wednesday, Thursday, Friday, Sunday

open Day

-- The main theorem
theorem optimal_start_day_for_vacation_to_maximize_sunny_days :
  ∃ d : Day, (d = Thursday) :=
by {
  -- skipping the proof for now
  sorry
}

end optimal_start_day_for_vacation_to_maximize_sunny_days_l311_311604


namespace probability_of_same_color_is_34_over_105_l311_311081

-- Define the number of each color of plates
def num_red_plates : ℕ := 7
def num_blue_plates : ℕ := 5
def num_yellow_plates : ℕ := 3

-- Define the total number of plates
def total_plates : ℕ := num_red_plates + num_blue_plates + num_yellow_plates

-- Define the total number of ways to choose 2 plates from the total plates
def total_ways_to_choose_2_plates : ℕ := Nat.choose total_plates 2

-- Define the number of ways to choose 2 red plates, 2 blue plates, and 2 yellow plates
def ways_to_choose_2_red_plates : ℕ := Nat.choose num_red_plates 2
def ways_to_choose_2_blue_plates : ℕ := Nat.choose num_blue_plates 2
def ways_to_choose_2_yellow_plates : ℕ := Nat.choose num_yellow_plates 2

-- Define the total number of favorable outcomes (same color plates)
def favorable_outcomes : ℕ :=
  ways_to_choose_2_red_plates + ways_to_choose_2_blue_plates + ways_to_choose_2_yellow_plates

-- Prove that the probability is 34/105
theorem probability_of_same_color_is_34_over_105 :
  (favorable_outcomes : ℚ) / (total_ways_to_choose_2_plates : ℚ) = 34 / 105 := by
  sorry

end probability_of_same_color_is_34_over_105_l311_311081


namespace sum_of_divisors_of_29_l311_311817

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311817


namespace satisfies_diff_eq_l311_311187

def y (x c : ℝ) := x * (c - real.log x)

theorem satisfies_diff_eq (x c : ℝ) (h₁ : x > 0) :
  let y := y x c,
      dy := deriv (λ x, y x c)
  in (x - y) + x * dy = 0 := sorry

end satisfies_diff_eq_l311_311187


namespace scientific_notation_l311_311927

theorem scientific_notation (h : 0.000000007 = 7 * 10^(-9)) : 0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_l311_311927


namespace find_m_n_l311_311439

open Nat

theorem find_m_n : ∃ m n : ℕ, m < n ∧ n < 2 * m ∧ m * n = 2013 ∧ m = 33 ∧ n = 61 := by
  use 33, 61
  simp
  sorry

end find_m_n_l311_311439


namespace sum_of_g_of_f_l311_311154

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 9
noncomputable def g (y : ℝ) : ℝ := 3*y + 1

theorem sum_of_g_of_f :
  let values := {x | f x = 9}
  ∑ x in values, g x = 20 :=
by
  sorry

end sum_of_g_of_f_l311_311154


namespace cube_root_simplification_l311_311249

theorem cube_root_simplification : (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 := 
by
  sorry

end cube_root_simplification_l311_311249


namespace students_in_class_l311_311105

theorem students_in_class (S : ℕ)
  (h₁ : S / 2 + 2 * S / 5 - S / 10 = 4 * S / 5)
  (h₂ : S / 5 = 4) :
  S = 20 :=
sorry

end students_in_class_l311_311105


namespace suff_but_not_necc_condition_l311_311466

def x_sq_minus_1_pos (x : ℝ) : Prop := x^2 - 1 > 0
def x_minus_1_pos (x : ℝ) : Prop := x - 1 > 0

theorem suff_but_not_necc_condition : 
  (∀ x : ℝ, x_minus_1_pos x → x_sq_minus_1_pos x) ∧
  (∃ x : ℝ, x_sq_minus_1_pos x ∧ ¬ x_minus_1_pos x) :=
by 
  sorry

end suff_but_not_necc_condition_l311_311466


namespace part1_part2_l311_311073

def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, -1)
def n (x : ℝ) : ℝ × ℝ := (sin x, cos^2 x)

theorem part1 (x : ℝ) (hx : x = π / 3) : 
  ((m x).1 * (n x).1 + (m x).2 * (n x).2) = 1 / 2 := by
  sorry
  
theorem part2 (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ π / 4) 
  (h : (m x).1 * (n x).1 + (m x).2 * (n x).2 = sqrt 3 / 3 - 1 / 2) : 
  cos (2*x) = (3 * sqrt 2 - sqrt 3) / 6 := by
  sorry

end part1_part2_l311_311073


namespace min_value_of_f_l311_311061

-- Defining the function f(x) = x ln(x) - a (x - 1)
def f (x a : ℝ) : ℝ := x * Real.log x - a * (x - 1)

-- The main theorem which states the conditions and the respective minimum values
theorem min_value_of_f (a : ℝ) : 
  (a ≤ 1 → ∀ x ∈ Set.Icc 1 Real.e, f x a ≥ 0 ∧ (∀ x', x' ∈ Set.Icc 1 Real.e → f x' a ≥ 0))
  ∧ (1 < a ∧ a < 2 → ∀ x ∈ Set.Icc 1 Real.e, f x a ≥ a - Real.exp (a - 1) ∧ (∀ x', x' ∈ Set.Icc 1 Real.e → f x' a ≥ a - Real.exp (a - 1)))
  ∧ (a ≥ 2 → ∀ x ∈ Set.Icc 1 Real.e, f x a ≥ a + Real.e - a*Real.e ∧ (∀ x', x' ∈ Set.Icc 1 Real.e → f x' a ≥ a + Real.e - a*Real.e)) :=
by 
  sorry -- Proof is omitted

end min_value_of_f_l311_311061


namespace correct_listed_price_l311_311891

noncomputable def listed_price 
  (initial_cost : ℝ) 
  (exchange_rate_increase : ℝ) 
  (discount_rate : ℝ) 
  (gst_rate : ℝ) 
  (profit_rate : ℝ) : ℝ :=
let increased_cost := initial_cost * (1 + exchange_rate_increase) in
let desired_profit := increased_cost * profit_rate in
let total_cost_with_profit := increased_cost + desired_profit in
let list_price := total_cost_with_profit / (1 - discount_rate) in
list_price

theorem correct_listed_price : 
  listed_price 9500 0.02 0.15 0.18 0.30 ≈ 14820.59 := 
by
  sorry

end correct_listed_price_l311_311891


namespace total_area_to_paint_is_correct_l311_311126

def length : ℕ := 15
def width : ℕ := 11
def height : ℕ := 9
def non_paintable_area_per_bedroom : ℕ := 70
def number_of_bedrooms : ℕ := 4

def wall_area_per_bedroom : ℕ := 2 * (length * height) + 2 * (width * height) - non_paintable_area_per_bedroom
def total_paintable_area : ℕ := number_of_bedrooms * wall_area_per_bedroom

theorem total_area_to_paint_is_correct : 
  total_paintable_area = 1592 := 
by 
  sorry

end total_area_to_paint_is_correct_l311_311126


namespace eastville_to_westpath_travel_time_l311_311280

theorem eastville_to_westpath_travel_time :
  ∀ (d t₁ t₂ : ℝ) (s₁ s₂ : ℝ), 
  t₁ = 6 → s₁ = 80 → s₂ = 50 → d = s₁ * t₁ → t₂ = d / s₂ → t₂ = 9.6 := 
by
  intros d t₁ t₂ s₁ s₂ ht₁ hs₁ hs₂ hd ht₂
  sorry

end eastville_to_westpath_travel_time_l311_311280


namespace cost_of_500_cookies_in_dollars_l311_311519

def cost_in_cents (cookies : Nat) (cost_per_cookie : Nat) : Nat :=
  cookies * cost_per_cookie

def cents_to_dollars (cents : Nat) : Nat :=
  cents / 100

theorem cost_of_500_cookies_in_dollars :
  cents_to_dollars (cost_in_cents 500 2) = 10
:= by
  sorry

end cost_of_500_cookies_in_dollars_l311_311519


namespace equal_segments_l311_311136

-- Define the problem conditions
variables {A B C H O1 O2 O : Point}
variables (O1_incenter_ACH : incenter O1 A C H) 
          (O2_incenter_BCH : incenter O2 B C H) 
          (O_incenter_ABC : incenter O A B C)
variables (H1 : Point) (H2 : Point) (H0 : Point)
variables (H1_proj : is_projection H1 O1 A B) -- H1 is the projection of O1 onto AB
          (H2_proj : is_projection H2 O2 A B) -- H2 is the projection of O2 onto AB
          (H0_proj : is_projection H0 O A B)  -- H0 is the projection of O onto AB
variables (CH_altitude : is_altitude C H A B) -- CH is the altitude from C to AB
variables (right_angle_ABC : right_angle ∠ACB) -- ∠ACB = 90 degrees
variables (BC_eq_2AC : BC = 2 * AC) -- BC = 2AC

-- Prove that H1H = HH0 = H0H2
theorem equal_segments (H1H_eq_HH0 : dist H1 H = dist H H0)
                       (HH0_eq_H0H2 : dist H H0 = dist H0 H2)
                       (HH1_H0H2 : dist H1 H = dist H0 H2) : 
                       H1H = HH0 ∧ HH0 = H0H2 ∧ H1H = H0H2 := 
begin
  sorry
end

end equal_segments_l311_311136


namespace tommy_needs_to_save_l311_311302

theorem tommy_needs_to_save (books : ℕ) (cost_per_book : ℕ) (money_he_has : ℕ) 
  (total_cost : ℕ) (money_needed : ℕ) 
  (h1 : books = 8)
  (h2 : cost_per_book = 5)
  (h3 : money_he_has = 13)
  (h4 : total_cost = books * cost_per_book) :
  money_needed = total_cost - money_he_has ∧ money_needed = 27 :=
by 
  sorry

end tommy_needs_to_save_l311_311302


namespace simplify_cube_roots_l311_311240

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311240


namespace wire_cut_ratio_l311_311935

/-- Given that a wire of total length 96 cm is cut into three pieces such that
    the shortest piece is 16 cm, we need to find the ratio of the lengths of 
    the three pieces. -/
theorem wire_cut_ratio (A B C : ℕ) (hA : A = 16) (hTotal : A + B + C = 96) :
  ∃ k : ℕ, (A:B:C) = (16:k:((96-16)-k)) :=
by
  sorry

end wire_cut_ratio_l311_311935


namespace auditorium_seats_l311_311109

variable (S : ℕ)

theorem auditorium_seats (h1 : 2 * S / 5 + S / 10 + 250 = S) : S = 500 :=
by
  sorry

end auditorium_seats_l311_311109


namespace movie_profit_proof_l311_311369

theorem movie_profit_proof
  (cost_actors : ℝ) 
  (num_people : ℝ)
  (cost_food_per_person : ℝ) 
  (cost_equipment_multiplier : ℝ) 
  (selling_price : ℝ) :
  cost_actors = 1200 →
  num_people = 50 →
  cost_food_per_person = 3 →
  cost_equipment_multiplier = 2 →
  selling_price = 10000 →
  let cost_food := num_people * cost_food_per_person in
  let total_cost_without_equipment := cost_actors + cost_food in
  let cost_equipment := cost_equipment_multiplier * total_cost_without_equipment in
  let total_cost := total_cost_without_equipment + cost_equipment in
  let profit := selling_price - total_cost in
  profit = 5950 :=
by
  intros h1 h2 h3 h4 h5
  have h_cost_food : cost_food = 150, by rw [h2, h3]; norm_num
  have h_total_cost_without_equipment : total_cost_without_equipment = 1350, by rw [h1, h_cost_food]; norm_num
  have h_cost_equipment : cost_equipment = 2700, by rw [h4, h_total_cost_without_equipment]; norm_num
  have h_total_cost : total_cost = 4050, by rw [h_total_cost_without_equipment, h_cost_equipment]; norm_num
  have h_profit : profit = 5950, by rw [h5, h_total_cost]; norm_num
  exact h_profit

end movie_profit_proof_l311_311369


namespace magical_castle_escape_l311_311936

theorem magical_castle_escape (n k : ℕ) (hn : 1 ≤ n) (hk : 1 ≤ k) : 
  ∃ strategy : Π (current_room : ℕ) (doors_tried : ℕ), current_room ≤ n ∧ (doors_tried ≤ k → current_room ≠ n) → (current_room = n ∧ doors_tried ≤ k) :=
by
  sorry

end magical_castle_escape_l311_311936


namespace expected_matches_result_l311_311256

noncomputable def expected_matches (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), (n - k : ℝ) * (nat.choose (n + k) k) * (1 / 2 : ℝ)^(n + k)

theorem expected_matches_result : expected_matches 60 ≈ 7.795 := 
begin
  sorry
end

end expected_matches_result_l311_311256


namespace calculate_expression_l311_311402

theorem calculate_expression :
  - (2:ℤ)^2 - real.sqrt (9:ℝ) + (- (5:ℤ))^2 * (2 / 5:ℝ) = 3 :=
by
  sorry

end calculate_expression_l311_311402


namespace sum_of_divisors_29_l311_311828

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311828


namespace tangent_line_at_one_l311_311664

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one : ∀ (x y : ℝ), y = 2 * Real.exp 1 * x - Real.exp 1 → 
  ∃ m b : ℝ, (∀ x: ℝ, f x = m * x + b) ∧ (m = 2 * Real.exp 1) ∧ (b = -Real.exp 1) :=
by
  sorry

end tangent_line_at_one_l311_311664


namespace f_odd_f_g_neg_two_l311_311470

def f (x : ℝ) : ℝ := if x < 0 then g x else 2 ^ x - 3

def g (x : ℝ) : ℝ := sorry  -- Assuming g is defined somewhere else

theorem f_odd (x : ℝ) : f (-x) = -f x := sorry  -- Given that f is odd function

theorem f_g_neg_two : f (g (-2)) = 1 := by
  have h1 : g (-2) = -f 2 := by 
    rw [f_odd (-2), f]
    sorry -- steps to derive the definition of g(-2)
  have h2 : f (-1) = -f 1 := by
    rw [f_odd 1, f]
    sorry -- steps to derive the definition of f(-1)
  have h3 : f (g (-2)) = f (-1) := by 
    rw h1
    sorry -- additional steps if necessary
  rw [h3, h2]
  exact sorry -- conclusion that it equals 1

end f_odd_f_g_neg_two_l311_311470


namespace sum_of_divisors_of_29_l311_311764

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311764


namespace sum_of_divisors_of_29_l311_311754

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311754


namespace sum_of_divisors_of_29_l311_311793

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311793


namespace sum_of_divisors_29_l311_311862

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311862


namespace scientific_notation_of_35000000_l311_311955

theorem scientific_notation_of_35000000 :
  (35_000_000 : ℕ) = 3.5 * 10^7 := by
  sorry

end scientific_notation_of_35000000_l311_311955


namespace sum_of_divisors_of_29_l311_311860

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311860


namespace solutions_case_I_no_real_solutions_case_II_l311_311253
noncomputable theory
open Real

def case_I (x : ℝ): Prop :=
  2 * 3^x * 5^(2*x) = (7^(1/(3*x))) * (11^(1/(4*x)))

def case_II (x : ℝ): Prop :=
  5 * 3^x * 2^(2*x) = ((1/7)^(1/(3*x))) * ((1/11)^(1/(4*x)))

theorem solutions_case_I :
  ∃ x, case_I x := sorry

theorem no_real_solutions_case_II :
  ¬∃ x, case_II x := sorry

end solutions_case_I_no_real_solutions_case_II_l311_311253


namespace power_of_i_l311_311472

theorem power_of_i (i : ℂ) 
  (h1: i^1 = i) 
  (h2: i^2 = -1) 
  (h3: i^3 = -i) 
  (h4: i^4 = 1)
  (h5: i^5 = i) 
  : i^2016 = 1 :=
by {
  sorry
}

end power_of_i_l311_311472


namespace tara_dad_attendance_percentage_l311_311419

theorem tara_dad_attendance_percentage
  (games_each_year : ℕ)
  (attendance_second_year : ℕ)
  (difference_attendance : ℕ)
  (attendance_first_year : ℕ)
  (percentage_attendance_first_year : ℚ) :
  games_each_year = 20 →
  attendance_second_year = 14 →
  difference_attendance = 4 →
  attendance_first_year = attendance_second_year + difference_attendance →
  percentage_attendance_first_year = (attendance_first_year / games_each_year : ℚ) * 100 →
  percentage_attendance_first_year = 90 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4] at h5 
  norm_num at h5
  exact h5


end tara_dad_attendance_percentage_l311_311419


namespace sum_of_divisors_of_29_l311_311800

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311800


namespace incorrect_statement_d_l311_311387

theorem incorrect_statement_d : 
  let profit_beginning_month (investment : ℕ) := (investment * 115 / 100) * 110 / 100 - investment
  let profit_end_month (investment : ℕ) := investment * 130 / 100 - investment - 700
  let investment_for_profit_beginning (profit : ℕ) := profit * 10000 / 1155
  let investment_for_profit_end (profit : ℕ) := (profit + 700) * 10 / 13
  (profit_beginning_month 20000 = 5300) ∧ (profit_end_month 16000 = 5300) ->
  ¬((beginning_month_investment_to_profit 5300 < end_month_investment_to_profit 5300)) :=
begin
  let incorrect_statement_d := (
    (15 / 100 + 1) * (10 / 100 + 1) - 1  * (1 + 30 / 100) - 700
  )
  sorry
end

end incorrect_statement_d_l311_311387


namespace sum_of_divisors_of_29_l311_311854

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311854


namespace distance_correct_l311_311417

def distance_center_circle_point : ℝ :=
  let circle_eq := λ x y : ℝ, x^2 + y^2 - 6*x - 8*y - 9;
  let point := (9 : ℝ, 5 : ℝ);
  let center := (3 : ℝ, 4 : ℝ);
  let distance := Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2);
  distance

theorem distance_correct : distance_center_circle_point = Real.sqrt 37 := 
by
  sorry

end distance_correct_l311_311417


namespace reciprocal_HCF_24_195_l311_311264

theorem reciprocal_HCF_24_195 : HCF 24 195 = 3 → (1 : ℚ) / 3 = (1/3 : ℚ) :=
by
  intros hcf_24_195
  -- proof will go here
  sorry

end reciprocal_HCF_24_195_l311_311264


namespace sum_of_divisors_of_29_l311_311857

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311857


namespace evaluate_f_at_one_l311_311024

def f (x : ℝ) := (x+1)/x

theorem evaluate_f_at_one : f 1 = 2 := by
  -- The proof is omitted deliberately
  sorry

end evaluate_f_at_one_l311_311024


namespace textbook_weight_difference_l311_311573

theorem textbook_weight_difference :
  let chem_weight := 7.125
  let geom_weight := 0.625
  chem_weight - geom_weight = 6.5 :=
by
  sorry

end textbook_weight_difference_l311_311573


namespace shaded_area_l311_311426

variable {Point : Type} [Inhabited Point] [linearOrder Point] [decidableEq Point]

structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

def Line.eval (L : Line) (x : ℝ) : ℝ :=
  L.m * x + L.b

-- Line passing through points (0,5) and (10,2)
def L1 : Line := {
  m := -3/10,
  b := 5
}

-- Line passing through points (2,6) and (9,1)
def L2 : Line := {
  m := -5/7,
  b := 52/7
}

theorem shaded_area (h1 : ∀ x, 0 ≤ x → L1.eval x ∈ ℝ)
                    (h2 : ∀ x, 0 ≤ x → L2.eval x ∈ ℝ) :
  ∫ x in 2..7, (L2.eval x - L1.eval x) = 85 / 14 :=
by
  -- Proof is omitted 
  sorry

end shaded_area_l311_311426


namespace simplified_expression_correct_l311_311210

noncomputable def simplify_expression : ℝ := 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27))

theorem simplified_expression_correct : simplify_expression = Real.cbrt 112 := 
by
  sorry

end simplified_expression_correct_l311_311210


namespace find_a_l311_311014

theorem find_a (a : ℝ) : 
  ((x y : ℝ) (P : ℝ → ℝ → Prop), P x y ↔ x + 2 * y - 5 = 0) → 
  ((x y : ℝ) (Q : ℝ → ℝ → Prop), Q x y ↔ 2 * x + 4 * y + a = 0) → 
  (dist : ℝ) (h_dist : dist = sqrt 5) → 
  (distance : ℝ → ℝ → Prop), 
  distance (2 * x + 4 * y - 10 = 0) (2 * x + 4 * y + a = 0) → 
  |a + 10| = 10 → 
  (a = 0 ∨ a = -20) :=
begin
  intro P,
  intro Q,
  intro dist,
  intro h_dist,
  intro distance,
  intro h_distance,
  intro h_abs,
  sorry
end

end find_a_l311_311014


namespace sum_of_odd_numbers_from_100_to_200_l311_311012

noncomputable def sum_of_odds (a b : ℕ) : ℕ :=
  (list.range' a (b - a + 1)).filter (λ x, x % 2 = 1).sum

theorem sum_of_odd_numbers_from_100_to_200 :
  sum_of_odds 100 200 = 7500 :=
by
  sorry

end sum_of_odd_numbers_from_100_to_200_l311_311012


namespace sum_of_divisors_of_29_l311_311799

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311799


namespace running_distance_difference_l311_311534

theorem running_distance_difference :
  ∀ (street_width side_length : ℕ),
  street_width = 30 →
  side_length = 500 →
  ∀ (jane_path perimeter_john jane_perimeter : ℕ),
  jane_path = 4 * side_length →
  perimeter_john = 4 * (side_length + 2 * street_width) →
  jane_perimeter = 4 * side_length →
  (perimeter_john - jane_perimeter) = 240 :=
by
  intros street_width side_length hsw hsl jane_path perimeter_john jane_perimeter
  intros hjp hjon hjane
  rw [hsl, hsw] at *
  rw [hjp, hjon, hjane]
  simp at *
  sorry

end running_distance_difference_l311_311534


namespace sum_of_divisors_of_29_l311_311737

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311737


namespace tenth_even_term_is_92_l311_311984

def arithmetic_sequence (n : ℕ) : ℕ := 5 * n - 3

def even_term_condition (n : ℕ) : Prop := (5 * n - 3) % 2 = 0

theorem tenth_even_term_is_92 :
  ∃ n : ℕ, even_term_condition n ∧ (arith_sequence n = 92) ∧ (∃ k : ℕ, (2 * k + 3) % 5 = 0 ∧ k % 3 = 1) :=
sorry

end tenth_even_term_is_92_l311_311984


namespace sum_of_terms_in_fractional_array_l311_311352

theorem sum_of_terms_in_fractional_array :
  (∑' (r : ℕ) (c : ℕ), (1 : ℝ) / ((3 * 4) ^ r) * (1 / (4 ^ c))) = (1 / 33) := sorry

end sum_of_terms_in_fractional_array_l311_311352


namespace total_pages_in_scifi_section_l311_311671

theorem total_pages_in_scifi_section : 
  let books := 8
  let pages_per_book := 478
  books * pages_per_book = 3824 := 
by
  sorry

end total_pages_in_scifi_section_l311_311671


namespace shaded_area_percentage_l311_311116

noncomputable def area_of_square (side_length : ℝ) : ℝ := side_length * side_length

/-- Each square has a side length of 1 unit -/
def side_length : ℝ := 1

/-- There are five such squares -/
def number_of_squares : ℕ := 5

/-- The shaded area consists of two squares -/
def shaded_squares : ℕ := 2

/-- The total area of the five squares -/
def total_area : ℝ := number_of_squares * area_of_square side_length

/-- The area of the shaded part -/
def shaded_area : ℝ := shaded_squares * area_of_square side_length

/-- The percentage of the total area that is shaded -/
def percentage_shaded : ℝ := (shaded_area / total_area) * 100

theorem shaded_area_percentage : percentage_shaded = 40 := by
  -- Proof will be provided here.
  sorry

end shaded_area_percentage_l311_311116


namespace bisection_of_CD_l311_311582

theorem bisection_of_CD
  (A B C D E P : Type*)
  [cyclic_quadrilateral A B C D]
  (h1 : AD^2 + BC^2 = AB^2)
  (h2 : ∃ E, intersection AC BD E)
  (h3 : ∃ P ∈ line AB, ∠APD = ∠BPC) :
  ∃ M ∈ line CD, midpoint M C D ∧ line_through P E M :=
sorry

end bisection_of_CD_l311_311582


namespace complement_P_l311_311503

def U : Set ℝ := Set.univ

def P : Set ℝ := {x | x^2 < 1}

theorem complement_P : (U \ P) = Set.Iic (-1) ∪ Set.Ici 1 := by
  sorry

end complement_P_l311_311503


namespace total_minutes_last_weekend_l311_311575

-- Define the given conditions
def Lena_hours := 3.5 -- Lena played for 3.5 hours
def Brother_extra_minutes := 17 -- Brother played 17 minutes more than Lena

-- Define the conversion from hours to minutes
def hours_to_minutes (hours : ℝ) : ℕ := (hours * 60).to_nat

-- Total minutes Lena played
def Lena_minutes := hours_to_minutes Lena_hours

-- Total minutes her brother played
def Brother_minutes := Lena_minutes + Brother_extra_minutes

-- Define the total minutes played together
def total_minutes_played := Lena_minutes + Brother_minutes

-- The proof statement (with an assumed proof)
theorem total_minutes_last_weekend : total_minutes_played = 437 := 
by 
  sorry

end total_minutes_last_weekend_l311_311575


namespace count_relatively_prime_to_24_l311_311077

-- Define the problem in Lean 4 statement
theorem count_relatively_prime_to_24 (a b n : ℕ) (h_a : a = 20) (h_b : b = 120) (h_n : n = 24) :
  (finset.card (finset.filter (λ x, nat.gcd x n = 1) (finset.range (b - a + 2)) ⊳ (λ y, y + a))) = 33 :=
begin
  sorry
end

end count_relatively_prime_to_24_l311_311077


namespace range_f_range_a_l311_311056

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Ioo (-1 : ℝ) (-0.5) then x + 1 / x
  else if x ∈ Ioo (-0.5 : ℝ) 0.5 then -2.5
  else if x ∈ Ico (0.5 : ℝ) 1 then x - 1 / x
  else 0 -- Assume f(x) = 0 for x outside [-1, 1]

noncomputable def g (a x : ℝ) : ℝ := a * x - 3

theorem range_f :
  ∀ x ∈ Set.Icc (-1:ℝ) 1, f x ∈ Set.union (Set.Icc (-2.5) (-2)) (Set.Icc (-1.5) 0) :=
sorry

theorem range_a :
  ∀ (a : ℝ),
    (∀ (x1 : ℝ), x1 ∈ Set.Icc (-1) 1 → ∃ (x0 : ℝ), x0 ∈ Set.Icc (-1) 1 ∧ g a x0 = f x1) ↔
    a ∈ Set.union (Set.Iic (-3)) (Set.Ici 3) :=
sorry

end range_f_range_a_l311_311056


namespace shortest_total_distance_piglet_by_noon_l311_311321

-- Define the distances
def distance_fs : ℕ := 1300  -- Distance through the forest (Piglet to Winnie-the-Pooh)
def distance_pr : ℕ := 600   -- Distance (Piglet to Rabbit)
def distance_rw : ℕ := 500   -- Distance (Rabbit to Winnie-the-Pooh)

-- Define the total distance via Rabbit and via forest
def total_distance_rabbit_path : ℕ := distance_pr + distance_rw + distance_rw
def total_distance_forest_path : ℕ := distance_fs + distance_rw

-- Prove that shortest distance Piglet covers by noon
theorem shortest_total_distance_piglet_by_noon : 
  min (total_distance_forest_path) (total_distance_rabbit_path) = 1600 := by
  sorry

end shortest_total_distance_piglet_by_noon_l311_311321


namespace sum_of_divisors_29_l311_311824

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311824


namespace sum_of_divisors_of_29_l311_311819

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311819


namespace coefficient_of_x5_in_expansion_l311_311427

theorem coefficient_of_x5_in_expansion :
  (∀ (f : ℕ → ℕ → ℤ), f = (λ n k, (nat.choose n k) * (-2)^(k))) →
  (∏ i in finset.range 5, (2 - 2 * i) = -160) →
  sorry

end coefficient_of_x5_in_expansion_l311_311427


namespace speed_in_still_water_l311_311937

-- Given conditions
def upstream_speed : ℝ := 60
def downstream_speed : ℝ := 90

-- Proof that the speed of the man in still water is 75 kmph
theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 75 := 
by
  sorry

end speed_in_still_water_l311_311937


namespace sum_of_products_of_two_at_a_time_l311_311294

-- Given conditions
variables (a b c : ℝ)
axiom sum_of_squares : a^2 + b^2 + c^2 = 252
axiom sum_of_numbers : a + b + c = 22

-- The goal
theorem sum_of_products_of_two_at_a_time : a * b + b * c + c * a = 116 :=
sorry

end sum_of_products_of_two_at_a_time_l311_311294


namespace gcd_problem_l311_311005

theorem gcd_problem :
  let a := 450 - 60
      b := 330 - 15
      c := 675 - 45
      d := 725 - 25
  in Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 5 :=
by
  let a := 450 - 60
  let b := 330 - 15
  let c := 675 - 45
  let d := 725 - 25
  show Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 5
  sorry

end gcd_problem_l311_311005


namespace sum_of_divisors_of_prime_l311_311782

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311782


namespace sum_of_divisors_of_29_l311_311859

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311859


namespace sum_of_divisors_of_29_l311_311814

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311814


namespace Brady_average_hours_l311_311965

-- Definitions based on conditions
def hours_per_day_April : ℕ := 6
def hours_per_day_June : ℕ := 5
def hours_per_day_September : ℕ := 8
def days_in_April : ℕ := 30
def days_in_June : ℕ := 30
def days_in_September : ℕ := 30

-- Definition to prove
def average_hours_per_month : ℕ := 190

-- Theorem statement
theorem Brady_average_hours :
  (hours_per_day_April * days_in_April + hours_per_day_June * days_in_June + hours_per_day_September * days_in_September) / 3 = average_hours_per_month :=
sorry

end Brady_average_hours_l311_311965


namespace sum_of_divisors_of_29_l311_311762

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311762


namespace distance_integral_l311_311925

variable (v : ℝ → ℝ)
variable (t : ℝ)

def speed := 3 * t + 2

theorem distance_integral : 
  ∫ (t : ℝ) in 1..2, speed t = 13 / 2 :=
by
  sorry

end distance_integral_l311_311925


namespace ellipse_equation_line_equation_l311_311462

noncomputable def ellipse_E (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

noncomputable def line_l_1 (x y : ℝ) : Prop :=
  x - real.sqrt 2 * y + real.sqrt 3 = 0

noncomputable def line_l_2 (x y : ℝ) : Prop :=
  real.sqrt 2 * x - y + real.sqrt 6 = 0

theorem ellipse_equation :
  ∀ x y : ℝ, ellipse_E x y ↔ (x = 1 ∧ y = real.sqrt 3 / 2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

theorem line_equation :
  ∀ x y : ℝ, (line_l_1 x y ∨ line_l_2 x y) ↔ 
    ∃ A B C D : ℝ × ℝ, (ellipse_E A.1 A.2) ∧ (ellipse_E B.1 B.2) ∧ (ellipse_E C.1 C.2) ∧ (ellipse_E D.1 D.2) ∧ 
      (quadrilateral_area A B C D = 4 / 3) ∧ (line_l_1 (-real.sqrt 3) 0 ∨ line_l_2 (-real.sqrt 3) 0) :=
by
  sorry

end ellipse_equation_line_equation_l311_311462


namespace extrema_of_f_l311_311581

noncomputable def f (x y : ℝ) : ℝ := (x^2 - y^2) * Real.exp (-(x^2 + y^2))

theorem extrema_of_f :
  (∃ c ∈ (Set.range (λ (x : ℝ), ∃ y: ℝ, f x y)), IsExtremePoint (λ (xy : ℝ × ℝ), f xy.1 xy.2) c) ∧
  (∀ (x y : ℝ), HasPartialDerivAt (λ (xy : ℝ × ℝ), f xy.1 xy.2) (by simp [HasPartialDerivAt]) x) →
  (∃ x y : ℝ, 
    (f (0,0) = 0) ∧ 
    (f (1,0) = Real.exp (-1)) ∧ 
    (f (-1,0) = Real.exp (-1)) ∧ 
    (f (0,1) = -Real.exp (-1)) ∧ 
    (f (0,-1) = -Real.exp (-1)) ∧
    ∀ (x y : ℝ), 
    (x, y) = (0,0) ∨ 
    (x, y) = (1,0) ∨ 
    (x, y) = (-1,0) ∨ 
    (x, y) = (0,1) ∨ 
    (x, y) = (0,-1)) :=
sorry

end extrema_of_f_l311_311581


namespace probability_of_Ace_and_King_l311_311304

noncomputable def P_Ace : ℚ := 4 / 52
noncomputable def P_King_given_Ace : ℚ := 4 / 51
noncomputable def P_Ace_and_King : ℚ := P_Ace * P_King_given_Ace

theorem probability_of_Ace_and_King :
  P_Ace_and_King = 4 / 663 :=
by
  -- Directly stating the equivalence, proof to be added
  sorry

end probability_of_Ace_and_King_l311_311304


namespace maximize_norm_c_l311_311342

variable {V : Type*} [inner_product_space ℝ V]

theorem maximize_norm_c
  (a b c : V) 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hab : ⟪a, b⟫ = 0)
  (h : ⟪3 • a - c, 4 • b - c⟫ = 0) :
  ∥c∥ = 5 := 
sorry

end maximize_norm_c_l311_311342


namespace sum_of_divisors_prime_29_l311_311724

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311724


namespace find_total_cost_l311_311662

-- Define the cost per kg for flour
def F : ℕ := 21

-- Conditions in the problem
axiom cost_eq_mangos_rice (M R : ℕ) : 10 * M = 10 * R
axiom cost_eq_flour_rice (R : ℕ) : 6 * F = 2 * R

-- Define the cost calculations
def total_cost (M R F : ℕ) : ℕ := (4 * M) + (3 * R) + (5 * F)

-- Prove the total cost given the conditions
theorem find_total_cost (M R : ℕ) (h1 : 10 * M = 10 * R) (h2 : 6 * F = 2 * R) : total_cost M R F = 546 :=
sorry

end find_total_cost_l311_311662


namespace incorrect_statements_l311_311992

noncomputable def statement1 : Prop :=
  (3 ^ 6 = 6 ^ 3) → False

noncomputable def statement2 : Prop :=
  (∑ k in Finset.range 1, Nat.choose 3 k * Nat.choose 4 (3 - k) = 60) → False

def statement3 (f g : ℝ → ℝ) [∀ x, Differentiable ℝ f x] [∀ x, Differentiable ℝ g x] : Prop :=
  (∀ x, f (-x) = -f x ∧ g (-x) = g x) →
  (∀ x, x > 0 → f' x < 0 ∧ g' x < 0) →
  (∀ x, x < 0 → f' x < 0 ∧ g' x > 0) → False

def statement4 (f : ℝ → ℝ) (a c b : ℝ) [a < c] [c < b] : Prop :=
  ∫ x in a..b, f x = ∫ x in a..c, f x + ∫ x in c..b, f x

theorem incorrect_statements :
  statement1 ∧ statement2 ∧ statement3 f g ∧ statement4 f a c b := by
  sorry

end incorrect_statements_l311_311992


namespace baby_mice_per_litter_l311_311398

def total_baby_mice (total : ℕ) (litters : ℕ) : Prop :=
  litters = 3 ∧
  ∃ x, total = x ∧ 
  let mice_to_robbie := x / 6 in
  let mice_to_pet_store := 3 * mice_to_robbie in
  let remaining_mice_before_snake_sale := x - (mice_to_robbie + mice_to_pet_store) in
  let mice_to_snake_owners := remaining_mice_before_snake_sale / 2 in
  let mice_left_with_brenda := remaining_mice_before_snake_sale - mice_to_snake_owners in
  mice_left_with_brenda = 4

theorem baby_mice_per_litter (total : ℕ) (litters : ℕ) (h : total_baby_mice total litters) : total / litters = 8 :=
by sorry

end baby_mice_per_litter_l311_311398


namespace sum_of_divisors_29_l311_311826

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311826


namespace sum_of_divisors_of_29_l311_311795

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311795


namespace parabola_line_intersection_l311_311453

theorem parabola_line_intersection (p : ℝ) (hp : p > 0) 
  (line_eq : ∃ b : ℝ, ∀ x : ℝ, 2 * x + b = 2 * x - p/2) 
  (focus := (p / 4, 0))
  (point_A := (0, -p / 2))
  (area_OAF : 1 / 2 * (p / 4) * (p / 2) = 1) : 
  p = 4 :=
sorry

end parabola_line_intersection_l311_311453


namespace sum_of_cubes_of_consecutive_integers_l311_311295

-- Define the given condition
def sum_of_squares_of_consecutive_integers (n : ℕ) : Prop :=
  (n - 1)^2 + n^2 + (n + 1)^2 = 7805

-- Define the statement we want to prove
theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : sum_of_squares_of_consecutive_integers n) : 
  (n - 1)^3 + n^3 + (n + 1)^3 = 398259 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l311_311295


namespace sum_of_divisors_prime_29_l311_311735

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311735


namespace number_of_people_l311_311375

theorem number_of_people (n k : ℕ) (h₁ : k * n * (n - 1) = 440) : n = 11 :=
sorry

end number_of_people_l311_311375


namespace find_y_for_orthogonal_vectors_l311_311424

theorem find_y_for_orthogonal_vectors : 
  (∀ y, ((3:ℝ) * y + (-4:ℝ) * 9 = 0) → y = 12) :=
by
  sorry

end find_y_for_orthogonal_vectors_l311_311424


namespace probability_of_no_two_adjacent_stands_l311_311263

/-- Define the probability of an event where no two adjacent people stand after flipping coins at a circular table -/
def probability_no_adjacent_stands (n : ℕ) : ℚ :=
  if n = 10 then 123 / 1024 else 0

/-- Theorem stating the probability calculation -/
theorem probability_of_no_two_adjacent_stands {n : ℕ} (h : n = 10) :
  probability_no_adjacent_stands n = 123 / 1024 :=
by
  rw probability_no_adjacent_stands
  split_ifs
  · exact rfl
  · contradiction

#check probability_of_no_two_adjacent_stands

end probability_of_no_two_adjacent_stands_l311_311263


namespace sum_of_divisors_of_29_l311_311767

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311767


namespace sum_of_divisors_of_29_l311_311749

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311749


namespace quadratic_completion_l311_311254

theorem quadratic_completion 
    (x : ℝ) 
    (h : 16*x^2 - 32*x - 512 = 0) : 
    ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by sorry

end quadratic_completion_l311_311254


namespace sum_of_divisors_of_29_l311_311858

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311858


namespace gross_profit_value_l311_311907

theorem gross_profit_value (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 54) 
    (h2 : gross_profit = 1.25 * cost) 
    (h3 : sales_price = cost + gross_profit): gross_profit = 30 := 
  sorry

end gross_profit_value_l311_311907


namespace friction_coefficient_example_l311_311358

variable (α : ℝ) (mg : ℝ) (μ : ℝ)

theorem friction_coefficient_example
    (hα : α = 85 * Real.pi / 180) -- converting degrees to radians
    (hN : ∀ (N : ℝ), N = 6 * mg) -- Normal force in the vertical position
    (F : ℝ) -- Force applied horizontally by boy
    (hvert : F * Real.sin α - mg + (6 * mg) * Real.cos α = 0) -- vertical equilibrium
    (hhor : F * Real.cos α - μ * (6 * mg) - (6 * mg) * Real.sin α = 0) -- horizontal equilibrium
    : μ = 0.08 :=
by
  sorry

end friction_coefficient_example_l311_311358


namespace line_equation_l311_311934

theorem line_equation :
  ∃ m b, m = 1 ∧ b = 5 ∧ (∀ x y, y = m * x + b ↔ x - y + 5 = 0) :=
by
  sorry

end line_equation_l311_311934


namespace sum_of_divisors_of_29_l311_311716

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311716


namespace speed_of_man_in_still_water_l311_311903

def upstream_speed : ℝ := 7
def downstream_speed : ℝ := 33

def speed_in_still_water : ℝ := (upstream_speed + downstream_speed) / 2

theorem speed_of_man_in_still_water : speed_in_still_water = 20 :=
by
  sorry

end speed_of_man_in_still_water_l311_311903


namespace sum_of_divisors_of_29_l311_311771

theorem sum_of_divisors_of_29 : 
  ∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311771


namespace sum_of_divisors_of_29_l311_311880

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311880


namespace cubic_function_extrema_range_l311_311043

theorem cubic_function_extrema_range (a : ℝ) :
  (∃ (f : ℝ → ℝ), f = λ x, x^3 + ax^2 + (a + 6)*x + 1 
  ∧ (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (deriv f x₁ = 0 ∧ deriv f x₂ = 0))) ↔ (a < -3 ∨ a > 6) :=
sorry

end cubic_function_extrema_range_l311_311043


namespace find_t_find_MN_length_l311_311547

noncomputable def A_polar := (√2, Real.pi / 4)

noncomputable def l_parametric (t : ℝ) : ℝ × ℝ := 
( 3 / 2 - √2 / 2 * t, 1 / 2 + √2 / 2 * t )

noncomputable def C_parametric (θ : ℝ) : ℝ × ℝ := 
( 2 * Real.cos θ, Real.sin θ )

theorem find_t (t : ℝ) :
  let A := (1 : ℝ, 1 : ℝ) in
  l_parametric t = A ↔ t = √2 / 2 :=
by
  sorry

theorem find_MN_length (θM θN: ℝ) :
  let l := { p : ℝ × ℝ | ∃ t, p = l_parametric t } in
  let C := { p : ℝ × ℝ | ∃ θ, p = C_parametric θ } in
  let intersections := (l ∩ C) in
  let M := (2 * Real.cos θM, Real.sin θM) in
  let N := (2 * Real.cos θN, Real.sin θN) in
  M ∈ intersections ∧ N ∈ intersections → 
  let x1 := 2 * Real.cos θM in
  let y1 := Real.sin θM in
  let x2 := 2 * Real.cos θN in
  let y2 := Real.sin θN in
  Real.sqrt (1 + (x1 + x2)^2 - 4 * (x1 * x2)) = 4 * Real.sqrt 2 / 5 :=
by
  sorry

end find_t_find_MN_length_l311_311547


namespace sum_of_divisors_of_prime_l311_311779

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311779


namespace brady_average_hours_per_month_l311_311968

noncomputable def average_hours_per_month (hours_april : ℕ) (hours_june : ℕ) (hours_september : ℕ) : ℕ :=
  (hours_april + hours_june + hours_september) / 3

theorem brady_average_hours_per_month :
  let days := 30 in
  let hours_april := 6 * days in
  let hours_june := 5 * days in
  let hours_september := 8 * days in
  average_hours_per_month hours_april hours_june hours_september = 190 :=
begin
  sorry
end

end brady_average_hours_per_month_l311_311968


namespace cube_surface_area_l311_311686

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : volume = 64 → side = real.cbrt 64 → surface_area = 6 * side^2 → surface_area = 96 :=
by
  intros h1 h2 h3
  rw [h2, real.cbrt_eq] at h1
  have h4 : 4 = real.cbrt 64 := by sorry
  rw [h4, pow_eq] at h1
  assumption
  sorry

end cube_surface_area_l311_311686


namespace sum_of_divisors_of_29_l311_311759

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311759


namespace scientific_notation_35_million_l311_311953

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 : Float) ^ 7 := 
by
  sorry

end scientific_notation_35_million_l311_311953


namespace perimeter_of_trapezoid_l311_311121

-- Define the given conditions
def is_trapezoid (A B C D : Type) := 
  (AB : A ≃ B) (BC : B ≃ C) (CD : C ≃ D) (DA : D ≃ A) 
  (h1 : AB = CD) (h2 : BC = BD) (h3 : height(A, B, D, 5))

-- Define the points and lengths
noncomputable def A := sorry -- Placeholder for point A
noncomputable def B := sorry -- Placeholder for point B
noncomputable def C := sorry -- Placeholder for point C
noncomputable def D := sorry -- Placeholder for point D

-- Assume the lengths and the right angle conditions
variables AB CD BC AD : ℝ
h1 : AB = CD
h2 : BC = 10
h3 : AD = 22
h4 : height(A, B, D, 5)

-- Main theorem statement to prove the perimeter
theorem perimeter_of_trapezoid (A B C D : Type) : 
  is_trapezoid A B C D → 
  perimeter A B C D = 2 * Real.sqrt 61 + 32 :=
sorry

end perimeter_of_trapezoid_l311_311121


namespace simplify_cuberoot_product_l311_311223

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311223


namespace sum_of_divisors_of_29_l311_311851

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311851


namespace sum_of_divisors_of_29_l311_311718

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311718


namespace cube_root_multiplication_l311_311203

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311203


namespace num_pos_int_less_than_201_mult_of_5_or_7_but_not_both_l311_311511

theorem num_pos_int_less_than_201_mult_of_5_or_7_but_not_both :
  (∑ k in (Finset.range 201), ((k % 5 = 0) ∨ (k % 7 = 0)) ∧ ¬((k % 5 = 0) ∧ (k % 7 = 0))) = 58 :=
by
  sorry

end num_pos_int_less_than_201_mult_of_5_or_7_but_not_both_l311_311511


namespace option_d_holds_l311_311452

theorem option_d_holds (x y : ℝ) (h : x > y) : 
  (1 / 2) ^ x < (1 / 2) ^ y :=
sorry

end option_d_holds_l311_311452


namespace area_of_parallelogram_l311_311612

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l311_311612


namespace find_locus_of_circle_center_l311_311363

noncomputable def locus_center
    (R : ℝ)
    (center : ℝ × ℝ × ℝ)
    (x y z : ℝ) : Prop :=
    (center.1^2 + center.2^2 + center.3^2 = 2 * R^2) ∧ 
    (center.1 ≤ R) ∧ 
    (center.2 ≤ R) ∧ 
    (center.3 ≤ R)

theorem find_locus_of_circle_center 
    {R : ℝ} (hR : (0 < R)):
    let center := (R * real.sqrt 2, R * real.sqrt 2, R * real.sqrt 2)
    in locus_center R center :=
begin
    sorry
end

end find_locus_of_circle_center_l311_311363


namespace negation_proof_converse_proof_l311_311281

-- Define the proposition
def prop_last_digit_zero_or_five (n : ℤ) : Prop := (n % 10 = 0) ∨ (n % 10 = 5)
def divisible_by_five (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

-- Negation of the proposition
def negation_prop : Prop :=
  ∃ n : ℤ, prop_last_digit_zero_or_five n ∧ ¬ divisible_by_five n

-- Converse of the proposition
def converse_prop : Prop :=
  ∀ n : ℤ, ¬ prop_last_digit_zero_or_five n → ¬ divisible_by_five n

theorem negation_proof : negation_prop :=
  sorry  -- to be proved

theorem converse_proof : converse_prop :=
  sorry  -- to be proved

end negation_proof_converse_proof_l311_311281


namespace karthik_average_weight_l311_311328

variable (weight : ℝ)

theorem karthik_average_weight :
  (55 < weight) ∧ (weight < 62) ∧
  (50 < weight) ∧ (weight < 60) ∧
  (weight ≤ 58) →
  (weight = 56.5) :=
by
  intro h
  -- Extract conditions from the hypothesis for clarity.
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6

  have h_lower : 55 < weight := h1
  have h_upper : weight ≤ 58 := h6

  -- Calculating the average weight satisfying all these conditions.
  have average_weight : weight = (55 + 58) / 2 := sorry

  exact average_weight

end karthik_average_weight_l311_311328


namespace satisfies_diff_eq_l311_311188

def y (x c : ℝ) := x * (c - real.log x)

theorem satisfies_diff_eq (x c : ℝ) (h₁ : x > 0) :
  let y := y x c,
      dy := deriv (λ x, y x c)
  in (x - y) + x * dy = 0 := sorry

end satisfies_diff_eq_l311_311188


namespace simplify_cuberoot_product_l311_311221

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311221


namespace sum_of_divisors_of_prime_29_l311_311836

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311836


namespace weight_of_lightest_weight_l311_311691

theorem weight_of_lightest_weight (x : ℕ) (y : ℕ) (h1 : 0 < y ∧ y < 9)
  (h2 : (10 : ℕ) * x + 45 - (x + y) = 2022) : x = 220 := by
  sorry

end weight_of_lightest_weight_l311_311691


namespace simplify_cube_roots_l311_311237

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311237


namespace find_y_square_divisible_by_three_between_50_and_120_l311_311422

theorem find_y_square_divisible_by_three_between_50_and_120 :
  ∃ (y : ℕ), y = 81 ∧ (∃ (n : ℕ), y = n^2) ∧ (3 ∣ y) ∧ (50 < y) ∧ (y < 120) :=
by
  sorry

end find_y_square_divisible_by_three_between_50_and_120_l311_311422


namespace simplify_cuberoot_product_l311_311226

theorem simplify_cuberoot_product :
  (∛(1 + 27) * ∛(1 + ∛27)) = ∛112 :=
by sorry

end simplify_cuberoot_product_l311_311226


namespace determine_n_l311_311990

def nat_phi (n : ℕ) : ℕ := Nat.totient n

theorem determine_n (n : ℕ) (h1 : n > 2) (h2 : (nat_phi(n) / 2) % 6 = 1) :
  n = 3 ∨ n = 4 ∨ n = 6 ∨ ∃ (p k : ℕ), Nat.Prime p ∧ (n = p^(2*k) ∨ n = 2 * p^(2*k)) ∧ k % 12 = 11 :=
by sorry

end determine_n_l311_311990


namespace remainder_polynomial_div_l311_311008

theorem remainder_polynomial_div (y : ℤ) :
  let f := y^5 - 3*y^3 + 4*y + 5
  let g := (y - 3)^2
  let r := 261*y - 643
  degree r < 2 ∧ f = g * (f / g) + r := 
sorry

end remainder_polynomial_div_l311_311008


namespace valid_orderings_count_l311_311170

-- Define the distinct house colors as an enumerated type
inductive HouseColor
| Green : HouseColor
| Pink : HouseColor
| Violet : HouseColor
| Cyan : HouseColor
| Another : HouseColor  -- Any other color, can be represented as a placeholder

open HouseColor

-- Define the conditions in terms of predicates and rules

-- 1. Green house (G) before Pink house (P)
def cond1 (lst : List HouseColor) : Prop :=
  lst.indexOf Green < lst.indexOf Pink

-- 2. Violet house (V) before Cyan house (C)
def cond2 (lst : List HouseColor) : Prop :=
  lst.indexOf Violet < lst.indexOf Cyan

-- 3. Violet house (V) is not next to Cyan house (C)
def cond3 (lst : List HouseColor) : Prop :=
  abs (lst.indexOf Violet - lst.indexOf Cyan) ≠ 1

-- 4. The first house is neither Violet (V) nor Cyan (C)
def cond4 (lst : List HouseColor) : Prop :=
  lst.head ≠ Violet ∧ lst.head ≠ Cyan

-- Define the problem of counting valid permutations
def count_valid_permutations : Nat :=
  List.permutations [Green, Pink, Violet, Cyan, Another]
  .filter (λ lst, cond1 lst ∧ cond2 lst ∧ cond3 lst ∧ cond4 lst)
  .length

-- Theorem statement asserting that the number of valid orderings is 6
theorem valid_orderings_count : count_valid_permutations = 6 := by
  sorry

end valid_orderings_count_l311_311170


namespace mass_percentage_C_in_C6H8Ox_undetermined_l311_311432

-- Define the molar masses of Carbon, Hydrogen, and Oxygen
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008
def molar_mass_O : ℝ := 16.00

-- Define the molecular formula
def molar_mass_C6H8O6 : ℝ := (6 * molar_mass_C) + (8 * molar_mass_H) + (6 * molar_mass_O)

-- Given the mass percentage of Carbon in C6H8O6
def mass_percentage_C_in_C6H8O6 : ℝ := 40.91

-- Problem Definition
theorem mass_percentage_C_in_C6H8Ox_undetermined (x : ℕ) : 
  x ≠ 6 → ¬ (∃ p : ℝ, p = (6 * molar_mass_C) / ((6 * molar_mass_C) + (8 * molar_mass_H) + x * molar_mass_O) * 100) :=
by
  intro h1 h2
  sorry

end mass_percentage_C_in_C6H8Ox_undetermined_l311_311432


namespace imaginary_part_div_l311_311463

open Complex

theorem imaginary_part_div (z1 z2 : ℂ) (h1 : z1 = 1 + I) (h2 : z2 = I) :
  Complex.im (z1 / z2) = -1 := by
  sorry

end imaginary_part_div_l311_311463


namespace sum_of_divisors_of_prime_l311_311781

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311781


namespace parallelepiped_volume_half_l311_311513

variables {ℝ : Type*}[inner_product_space ℝ (euclidean_space ℝ (fin 3))]

-- let a and b be unit vectors
variables (a b : euclidean_space ℝ (fin 3))
variable (h1 : ∥a∥ = 1)
variable (h2 : ∥b∥ = 1)
-- let the angle between them be π/4
variable (h_angle : real.angle a b = π / 4)

-- the function that computes the volume of the parallelepiped
noncomputable def volume_of_parallelepiped (a b : euclidean_space ℝ (fin 3)) : ℝ :=
  abs (inner_product a ((a + (crossproduct.3d ℝ _).cross a b) ⬝ b))

-- The theorem we need to prove
theorem parallelepiped_volume_half (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h_angle : real.angle a b = π / 4) :
  volume_of_parallelepiped a b = 1 / 2 := 
sorry

end parallelepiped_volume_half_l311_311513


namespace imaginary_part_of_z_l311_311523

def z : ℂ := (1 : ℂ) * I / ((1 : ℂ) - I)

theorem imaginary_part_of_z : z.im = 1 / 2 := 
by
  -- The proof goes here, we're skipping it for now.
  sorry

end imaginary_part_of_z_l311_311523


namespace print_time_325_pages_l311_311940

theorem print_time_325_pages (pages : ℕ) (rate : ℕ) (delay_pages : ℕ) (delay_time : ℕ)
  (h_pages : pages = 325) (h_rate : rate = 25) (h_delay_pages : delay_pages = 100) (h_delay_time : delay_time = 1) :
  let print_time := pages / rate
  let delays := pages / delay_pages
  let total_time := print_time + delays * delay_time
  total_time = 16 :=
by
  sorry

end print_time_325_pages_l311_311940


namespace motorcycles_meet_after_54_minutes_l311_311335

noncomputable def motorcycles_meet_time : ℕ := sorry

theorem motorcycles_meet_after_54_minutes :
  motorcycles_meet_time = 54 := sorry

end motorcycles_meet_after_54_minutes_l311_311335


namespace area_of_octagon_l311_311040

-- Define the basic geometric elements and properties
variables {A B C D E F G H : Type}
variables (isRectangle : BDEF A B C D E F G H)
variables (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2)
variables (isRightIsosceles : ABC A B C D E F G H)

-- Assumptions and known facts
def BDEF_is_rectangle : Prop := isRectangle
def AB_eq_2 : AB = 2 := hAB
def BC_eq_2 : BC = 2 := hBC
def ABC_is_right_isosceles : Prop := isRightIsosceles

-- Statement of the problem to be proved
theorem area_of_octagon : (exists (area : ℝ), area = 8 * Real.sqrt 2) :=
by {
  -- The proof details will go here, which we skip for now
  sorry
}

end area_of_octagon_l311_311040


namespace perfect_square_and_solutions_exist_l311_311474

theorem perfect_square_and_solutions_exist (m n t : ℕ)
  (h1 : t > 0) (h2 : m > 0) (h3 : n > 0)
  (h4 : t * (m^2 - n^2) + m - n^2 - n = 0) :
  ∃ (k : ℕ), m - n = k * k ∧ (∀ t > 0, ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (t * (m^2 - n^2) + m - n^2 - n = 0)) :=
by
  sorry

end perfect_square_and_solutions_exist_l311_311474


namespace problem_statement_l311_311017

theorem problem_statement (k : ℤ) (n : Fin 2008 → ℕ) :
  (∀ i, n i > 0 ∧ k = int.floor ((n i)^(1/3:ℝ))) ∧ (∀ i, k ∣ n i) →
  k = 668 :=
begin
  -- Proof omitted
  sorry,
end

end problem_statement_l311_311017


namespace sum_of_divisors_of_29_l311_311746

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311746


namespace sum_of_divisors_of_29_l311_311809

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311809


namespace perp_PQ_OI_l311_311460

-- Definitions (as per conditions)
variables {A B C O I P Q : Point}

-- Assuming the following predicates are given
axiom is_circumcenter (t : Triangle) (O : Point) : Prop
axiom is_incenter (t : Triangle) (I : Point) : Prop
axiom on_ray (A B P : Point) : Prop
axiom distance_eq (A B C : Point) : Prop

-- Conditions based on axioms
def triangle_ABC := Triangle.mk A B C
def circumcenter_O := is_circumcenter triangle_ABC O
def incenter_I := is_incenter triangle_ABC I
def point_on_ray_BA_P := on_ray B A P
def point_on_ray_CA_Q := on_ray C A Q
def BP_eq_AB := distance_eq B P A B
def CQ_eq_AB := distance_eq C Q A B

-- Required assertion statement
theorem perp_PQ_OI
  (h₁: circumcenter_O)
  (h₂: incenter_I)
  (h₃: point_on_ray_BA_P)
  (h₄: point_on_ray_CA_Q)
  (h₅: BP_eq_AB)
  (h₆: CQ_eq_AB) :
  perp (line P Q) (line O I) :=
sorry

end perp_PQ_OI_l311_311460


namespace rachel_painting_minutes_l311_311643

def painted_minutes_day1 := 100 -- 1 hour 40 minutes is 100 minutes
def painted_minutes_day2 := 120 -- 2 hours is 120 minutes
def number_of_days1 := 6
def number_of_days2 := 2
def desired_average := 110
def total_days := 10

theorem rachel_painting_minutes :
  let total_painted_8_days := number_of_days1 * painted_minutes_day1 +
                              number_of_days2 * painted_minutes_day2
  in total_days * desired_average = 1100 →
     total_painted_8_days = 840 →
     (total_days * desired_average - total_painted_8_days) = 
     (10 * 110 - (6 * 100 + 2 * 120)) = 260 :=
sorry

end rachel_painting_minutes_l311_311643


namespace prism_cutout_l311_311958

noncomputable def original_volume : ℕ := 15 * 5 * 4 -- Volume of the original prism
noncomputable def cutout_width : ℕ := 5

variables {x y : ℕ}

theorem prism_cutout:
  -- Given conditions
  (15 > 0) ∧ (5 > 0) ∧ (4 > 0) ∧ (x > 0) ∧ (y > 0) ∧ 
  -- The volume condition
  (original_volume - y * cutout_width * x = 120) →
  -- Prove that x + y = 15
  (x + y = 15) :=
sorry

end prism_cutout_l311_311958


namespace inheritance_amount_l311_311600

-- Define the conditions
variable (x : ℝ) -- Let x be the inheritance amount
variable (H1 : x * 0.25 + (x * 0.75 - 5000) * 0.15 + 5000 = 16500)

-- Define the theorem to prove the inheritance amount
theorem inheritance_amount (H1 : x * 0.25 + (0.75 * x - 5000) * 0.15 + 5000 = 16500) : x = 33794 := by
  sorry

end inheritance_amount_l311_311600


namespace mike_seashells_total_l311_311161

variable (seashells1 seashells2 seashells_total : ℝ)

axiom seashells1_def : seashells1 = 6.0
axiom seashells2_def : seashells2 = 4.0

theorem mike_seashells_total : seashells1 + seashells2 = seashells_total :=
by
  rw [seashells1_def, seashells2_def]
  exact (by norm_num : 6.0 + 4.0 = 10.0)

end mike_seashells_total_l311_311161


namespace sum_of_divisors_of_29_l311_311751

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311751


namespace seven_digit_divisible_by_eleven_l311_311530

theorem seven_digit_divisible_by_eleven (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 9) 
  (h3 : 10 - n ≡ 0 [MOD 11]) : n = 10 :=
by
  sorry

end seven_digit_divisible_by_eleven_l311_311530


namespace train_crossing_time_l311_311948

noncomputable def time_for_train_to_cross_man (train_length : ℝ) (man_speed_kmph : ℝ) (train_speed_kmph : ℝ) : ℝ :=
let relative_speed_kmph := train_speed_kmph + man_speed_kmph in
let relative_speed_mps := relative_speed_kmph * 1000 / 3600 in
train_length / relative_speed_mps

theorem train_crossing_time :
  time_for_train_to_cross_man 110 5 60.994720422366214 ≈ 6 :=
by
  sorry

end train_crossing_time_l311_311948


namespace integral_zero_polar_coordinates_l311_311988

variable (x y : ℝ)

def condition1 (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1
def condition2 (x y : ℝ) : Prop := x^2 + y^2 = 4 * y
def integrand (x y : ℝ) : ℝ := x * y^2
def region (x y : ℝ) : Prop := condition1 x y ∨ condition2 x y

theorem integral_zero_polar_coordinates :
  (∬ x y^2 dx dy).filter (λ (p : ℝ × ℝ), region p.1 p.2) = 0 := sorry

end integral_zero_polar_coordinates_l311_311988


namespace sum_of_divisors_of_29_l311_311720

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311720


namespace sum_of_divisors_of_prime_29_l311_311840

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311840


namespace variance_measures_stability_l311_311258

-- Let A and B be students with very close average scores in several math exams.
variable (A B : Type) [Inhabited A] [Inhabited B]

-- Define terms corresponding to the problem conditions
def average_scores_close (averageA averageB : ℝ) : Prop :=
  |averageA - averageB| < ε  -- for some small ε > 0

def variance (X : Type) [Inhabited X] (scores : X → ℝ) : ℝ :=
  let avg := (finset.univ : finset X).sum (λ x, scores x) / finset.card (finset.univ : finset X)
  (finset.univ : finset X).sum (λ x, (scores x - avg) ^ 2) / finset.card (finset.univ : finset X)

-- The proof statement that variance is the correct measure for stability
theorem variance_measures_stability (scoresA scoresB : List ℝ) : 
  average_scores_close A B →
  (variance A scoresA) < (variance B scoresB) → False :=
sorry

end variance_measures_stability_l311_311258


namespace bad_arrangements_count_l311_311676

theorem bad_arrangements_count :
  let numbers := [1, 2, 3, 4, 5]
  let is_bad_arrangement (arr : list ℕ) : Prop :=
    ∀ s, s ⊆ set_of arr → (subset_sum arr s < 1 ∨ 13 < subset_sum arr s)
  set.count {arr | set.arrangement arr numbers ∧ is_bad_arrangement arr} = 3 :=
sorry

end bad_arrangements_count_l311_311676


namespace simplify_cuberoot_product_l311_311234

theorem simplify_cuberoot_product :
  ( (∛(1 + 27)) * (∛(1 + (∛27))) = ∛112 ) :=
by
  -- introduce the definition of the cube root
  let cube_root x := x^(1/3)
  -- apply the definition to the problem
  have h1 : cube_root (1 + 27) = cube_root 28 :=
    by sorry -- simplify lhs
  have h2 : cube_root (1 + cube_root 27) = cube_root 4 :=
    by sorry -- equality according to the nesting of cube roots
  have h3 : cube_root 28 * cube_root 4 = cube_root (28 * 4) :=
    by sorry -- multiply the simplified terms
  have h4 : cube_root (28 * 4) = cube_root 112 :=
    by sorry -- final simplification
  -- connect the pieces together
  exact eq.trans (eq.trans h1 (eq.trans h2 h3)) h4

end simplify_cuberoot_product_l311_311234


namespace complement_U_A_l311_311069

open Set
open Classical

variable {U A : Set ℝ}
noncomputable def U : Set ℝ := {x | x ≥ -3}
noncomputable def A : Set ℝ := {x | x > 1}
theorem complement_U_A : (U \ A) = {x | -3 ≤ x ∧ x ≤ 1} :=
by sorry

end complement_U_A_l311_311069


namespace sum_of_divisors_of_29_l311_311736

theorem sum_of_divisors_of_29 :
  (∀ n : ℕ, n = 29 → Prime n) → (∑ d in (Finset.filter (λ d, 29 % d = 0) (Finset.range 30)), d) = 30 :=
by
  intros h
  sorry

end sum_of_divisors_of_29_l311_311736


namespace probability_all_quitters_from_same_group_l311_311106

theorem probability_all_quitters_from_same_group :
  let n := 20
  let k := 3
  let group_size := 10
  let total_ways_to_choose_quitters := Nat.fact n / (Nat.fact k * Nat.fact (n - k))
  let ways_to_choose_from_one_group := Nat.fact group_size / (Nat.fact k * Nat.fact (group_size - k))
  let favorable_outcomes := 2 * ways_to_choose_from_one_group
  let probability := favorable_outcomes / total_ways_to_choose_quitters
  probability = 20 / 95 :=
by
  sorry

end probability_all_quitters_from_same_group_l311_311106


namespace max_value_of_quadratic_l311_311100

theorem max_value_of_quadratic (a : ℝ) : 
  (∀ x ∈ set.Icc (1 : ℝ) (3 : ℝ), (x^2 - 2 * a * x + 3) ≤ 6) →
  (∃ x ∈ set.Icc (1 : ℝ) (3 : ℝ), (x^2 - 2 * a * x + 3) = 6) →
  a = 1 :=
by
  sorry

end max_value_of_quadratic_l311_311100


namespace smallest_odd_number_satisfying_conditions_l311_311025

theorem smallest_odd_number_satisfying_conditions : 
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d) % 9 = 8 ∧
    (1000 * c + 100 * d + 10 * a + b) - (1000 * a + 100 * b + 10 * c + d) = 5940 ∧ 
    (1000 * a + 100 * b + 10 * c + d) % 2 = 1 ∧
    (∀ n, n = 1000 * a + 100 * b + 10 * c + d → n ≠ 1978 ∧ n ≠ 1977 ∧ ... ∧ n ≠ 1981) →
    1000 * a + 100 * b + 10 * c + d = 1979 :=
sorry

end smallest_odd_number_satisfying_conditions_l311_311025


namespace average_price_proof_l311_311330

def average_price_per_book
  (books_shop1 : ℕ) (cost_shop1 : ℕ)
  (books_shop2 : ℕ) (cost_shop2 : ℕ)
  : ℝ := (cost_shop1 + cost_shop2 : ℝ) / (books_shop1 + books_shop2 : ℝ)

theorem average_price_proof :
  average_price_per_book 65 1080 55 840 = 16 := by
  sorry

end average_price_proof_l311_311330


namespace sum_divisible_by_1001_l311_311171

theorem sum_divisible_by_1001 (a : Fin 10 → ℕ) : 
  ∃ (s : Finset (Fin 10)) (f : Fin 10 → ℤ), 
    (∑ i in s, f i * a i) % 1001 = 0 :=
sorry

end sum_divisible_by_1001_l311_311171


namespace complex_number_operation_l311_311409

theorem complex_number_operation :
  let z1 := complex.mk 2 5
  let z2 := complex.mk 3 (-3)
  let z3 := complex.mk 1 2
  z1 + z2 - z3 = complex.mk 4 0 :=
by
  sorry

end complex_number_operation_l311_311409


namespace sale_in_fifth_month_l311_311931

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 : ℕ) (avg_sale total_month sales_in_fifth_month : ℕ) 
  (sale1 = 3435) (sale2 = 3920) (sale3 = 3855) (sale4 = 4230) (sale6 = 2000) (avg_sale = 3500) (total_month = 6) 
  : sales_in_fifth_month = 3560 := 
by
  have total_sales_required : ℕ := avg_sale * total_month
  have sum_of_first_four : ℕ := sale1 + sale2 + sale3 + sale4
  have sales_in_fifth_month := total_sales_required - sum_of_first_four - sale6
  exact sorry

end sale_in_fifth_month_l311_311931


namespace eight_pow_91_gt_seven_pow_92_l311_311895

theorem eight_pow_91_gt_seven_pow_92 : 8^91 > 7^92 :=
  sorry

end eight_pow_91_gt_seven_pow_92_l311_311895


namespace inequality_solution_l311_311425

noncomputable def solution_set : Set ℝ := {x : ℝ | x < 4 ∨ x > 5}

theorem inequality_solution (x : ℝ) :
  (x - 2) / (x - 4) ≤ 3 ↔ x ∈ solution_set :=
by
  sorry

end inequality_solution_l311_311425


namespace quadratic_expression_l311_311031

noncomputable def quadratic_function (a b c : ℝ) : (ℝ → ℝ) := λ x, a * x^2 + b * x + c

theorem quadratic_expression (f : ℝ → ℝ) (a b : ℝ) :
  (f 0 = f 4) →
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ (x₁^2 + x₂^2 = 10)) →
  (f 0 = 3) →
  ∃ (a : ℝ), a = 1 ∧ f = quadratic_function a (-4 * a) 3 :=
by sorry

end quadratic_expression_l311_311031


namespace equilateral_faces_sufficient_cond_regular_pyramid_l311_311388

-- Define the conditions as separate statements:
axiom lateral_faces_equilateral_triangle : Π {P : Type}, Prop
axiom square_base : Π {P : Type}, Prop
axiom apex_angle_45_degrees : Π {P : Type}, Prop
axiom projection_apex_intersection_diagonals : Π {P : Type}, Prop

-- Define the property of being a regular pyramid:
axiom regular_pyramid : Π {P : Type}, Prop

-- The property that a condition is sufficient but not necessary:
axiom sufficient_not_necessary {α : Type} (P Q : α → Prop) : (∀ x, Q x → P x) ∧ ¬ (∀ x, P x → Q x)

-- The actual theorem statement:
theorem equilateral_faces_sufficient_cond_regular_pyramid {P : Type} :
  sufficient_not_necessary (λ p : P, regular_pyramid p) (λ p : P, lateral_faces_equilateral_triangle p) :=
sorry

end equilateral_faces_sufficient_cond_regular_pyramid_l311_311388


namespace sum_of_divisors_of_29_l311_311818

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311818


namespace julia_age_after_10_years_l311_311572

-- Define the conditions
def Justin_age : Nat := 26
def Jessica_older_by : Nat := 6
def James_older_by : Nat := 7
def Julia_younger_by : Nat := 8
def years_after : Nat := 10

-- Define the ages now
def Jessica_age_now : Nat := Justin_age + Jessica_older_by
def James_age_now : Nat := Jessica_age_now + James_older_by
def Julia_age_now : Nat := Justin_age - Julia_younger_by

-- Prove that Julia's age after 10 years is 28
theorem julia_age_after_10_years : Julia_age_now + years_after = 28 := by
  sorry

end julia_age_after_10_years_l311_311572


namespace sum_of_divisors_of_prime_29_l311_311839

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311839


namespace minimum_abs_diff_l311_311441

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem minimum_abs_diff {a b : ℝ} (h₁ : f a = 0) (h₂ : f b = 0) (h₃ : a ≠ b) :
  |a - b| = Real.pi / 2 :=
sorry

end minimum_abs_diff_l311_311441


namespace enclosed_area_l311_311003

noncomputable def area_enclosed : ℝ :=
  ∫ y in -1..3, (2*y + 3 - y^2)

theorem enclosed_area :
  area_enclosed = 32 / 3 :=
by
  sorry

end enclosed_area_l311_311003


namespace integer_pairs_equation_l311_311509

theorem integer_pairs_equation :
  { (m, n) : ℤ × ℤ // (m - 1) * (n - 1) = 2 }.card = 4 :=
sorry

end integer_pairs_equation_l311_311509


namespace minimum_even_integers_l311_311305

def integers_sum_28 (x y : ℤ) : Prop :=
  x + y = 28

def integers_sum_17 (a b : ℤ) : Prop :=
  a + b = 17

def integers_sum_18 (m n : ℤ) : Prop :=
  m + n = 18

theorem minimum_even_integers (x y a b m n : ℤ) :
  integers_sum_28 x y →
  integers_sum_17 a b →
  integers_sum_18 m n →
  (∀ (e : ℕ), e = ∑ i in [x, y, a, b, m, n], (i % 2 = 0) = 1) :=
by
  intros hxy hab hmn
  sorry

end minimum_even_integers_l311_311305


namespace simplify_cube_roots_l311_311244

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311244


namespace sum_of_divisors_of_29_l311_311807

theorem sum_of_divisors_of_29 : ∑ d in ({1, 29} : Finset ℕ), d = 30 :=
by
  -- The proof would go here
  sorry

end sum_of_divisors_of_29_l311_311807


namespace least_possible_faces_two_dice_l311_311696

noncomputable def least_possible_sum_of_faces (a b : ℕ) : ℕ :=
(a + b)

theorem least_possible_faces_two_dice (a b : ℕ) (h1 : 8 ≤ a) (h2 : 8 ≤ b)
  (h3 : ∃ k, 9 * k = 2 * (11 * k)) 
  (h4 : ∃ m, 9 * m = a * b) : 
  least_possible_sum_of_faces a b = 22 :=
sorry

end least_possible_faces_two_dice_l311_311696


namespace year_2013_is_not_special_l311_311950

def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), month * day = year % 100 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

theorem year_2013_is_not_special : ¬ is_special_year 2013 := by
  sorry

end year_2013_is_not_special_l311_311950


namespace sum_of_divisors_29_l311_311823

theorem sum_of_divisors_29 : (∑ d in (finset.filter (λ d, d ∣ 29) (finset.range 30)), d) = 30 := by
  have h_prime : Nat.Prime 29 := by sorry -- 29 is prime
  sorry -- Sum of divisors calculation

end sum_of_divisors_29_l311_311823


namespace sum_of_divisors_of_29_l311_311883

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311883


namespace vasya_can_write_numbers_l311_311344

noncomputable theory

def conditions (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i < n → b i ≥ a i) ∧
  (∀ i j : ℕ, i < n → j < n → b i / b j ∈ ℤ)

theorem vasya_can_write_numbers (a b : ℕ → ℝ) (n : ℕ) (h_cond: conditions a b n) : 
  ∃ (b : ℕ → ℝ), 
    (∀ i : ℕ, i < n → b i ≥ a i) ∧ 
    (∀ i j : ℕ, i < n → j < n → b i / b j ∈ ℤ) ∧ 
    (∏ i in finset.range n, b i) ≤ 2^((n - 1) / 2) * (∏ i in finset.range n, a i) :=
sorry

end vasya_can_write_numbers_l311_311344


namespace ellipse_condition_range_k_l311_311053

theorem ellipse_condition_range_k (k : ℝ) : 
  (2 - k > 0) ∧ (3 + k > 0) ∧ (2 - k ≠ 3 + k) → -3 < k ∧ k < 2 := 
by 
  sorry

end ellipse_condition_range_k_l311_311053


namespace sum_sequence_1_sum_sequence_2_l311_311013

-- Define sequence sum for (1)
def sequence_sum_1 (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), (2 * k - 3 * (1 / 5) ^ k)

-- Define the target formula from solution for (1)
def target_formula_1 (n : ℕ) : ℝ :=
  n * (n + 1) - 3 + 3 / (5 ^ (n - 1))

-- Define sequence sum for (2)
def sequence_sum_2 (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), 1 / (k * (k + 1))

-- Define the target formula from solution for (2)
def target_formula_2 (n : ℕ) : ℝ :=
  n / (n + 1)

-- Theorems stating the equivalence

theorem sum_sequence_1 (n : ℕ) : sequence_sum_1 n = target_formula_1 n := sorry

theorem sum_sequence_2 (n : ℕ) : sequence_sum_2 n = target_formula_2 n := sorry

end sum_sequence_1_sum_sequence_2_l311_311013


namespace find_income_on_third_day_l311_311359

noncomputable def income_on_third_day 
  (income1 income2 income3 income4 income5 avg_income : ℝ)
  (h_avg : (income1 + income2 + income3 + income4 + income5) / 5 = avg_income) : Prop :=
  income3 = 650

theorem find_income_on_third_day : ∀ (income1 income2 income4 income5 : ℝ),
  income1 = 400 ∧ income2 = 250 ∧ income4 = 400 ∧ income5 = 500 → income_on_third_day income1 income2 _ income4 income5 440 :=
by {
  intros _ _ _ income5 h,
  sorry
}

end find_income_on_third_day_l311_311359


namespace find_bottles_l311_311566

-- Definitions for the conditions and calculations
def bottle_price : ℝ := 0.10  -- Price per bottle in dollars
def can_price : ℝ := 0.05     -- Price per can in dollars
def cans_recycled : ℕ := 140  -- Number of cans recycled
def total_amount_earned : ℝ := 15.0  -- Total amount earned in dollars

-- Amount Jack made from cans
def amount_from_cans : ℝ := cans_recycled * can_price

-- The equation relating bottles and earnings
theorem find_bottles (B : ℕ) : 
  let amount_from_cans := cans_recycled * can_price in
  let amount_from_bottles := total_amount_earned - amount_from_cans in
  let number_of_bottles := amount_from_bottles / bottle_price in
  number_of_bottles = 80 := 
sorry

end find_bottles_l311_311566


namespace simplify_cubed_roots_l311_311219

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311219


namespace rod_sliding_friction_l311_311356

noncomputable def coefficient_of_friction (mg : ℝ) (F : ℝ) (α : ℝ) := 
  (F * Real.cos α - 6 * mg * Real.sin α) / (6 * mg)

theorem rod_sliding_friction
  (α : ℝ)
  (hα : α = 85 * Real.pi / 180)
  (mg : ℝ)
  (hmg_pos : 0 < mg)
  (F : ℝ)
  (hF : F = (mg - 6 * mg * Real.cos 85) / Real.sin 85) :
  coefficient_of_friction mg F α = 0.08 := 
by
  simp [coefficient_of_friction, hα, hF, Real.cos, Real.sin]
  sorry

end rod_sliding_friction_l311_311356


namespace sum_of_divisors_of_29_l311_311761

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311761


namespace sum_of_divisors_of_29_l311_311756

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311756


namespace find_annual_interest_rate_l311_311004

def compound_interest_formula (P A r : ℝ) (n t : ℕ) : Prop :=
  A = P * (1 + r / n)^(n * t)

def compound_interest (P CI r : ℝ) (n t : ℕ) : Prop :=
  let A := P + CI
  in compound_interest_formula P A r n t

theorem find_annual_interest_rate : 
  ∃ r : ℝ, compound_interest 100000 8243.216 r 2 2 ∧ abs (r - 0.039794706) < 0.00001 :=
sorry

end find_annual_interest_rate_l311_311004


namespace sum_of_divisors_of_prime_l311_311778

theorem sum_of_divisors_of_prime (h_prime: Nat.prime 29) : ∑ i in ({i | i ∣ 29}) = 30 :=
by
  sorry

end sum_of_divisors_of_prime_l311_311778


namespace simplify_cube_roots_l311_311241

theorem simplify_cube_roots :
  (∛(1+27) * ∛(1+∛27) = ∛112) :=
by {
  sorry
}

end simplify_cube_roots_l311_311241


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311620

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311620


namespace measure_of_angle_C_in_triangle_l311_311558

theorem measure_of_angle_C_in_triangle 
  (A B C : ℝ)
  (h1 : 4 * Real.sin A + 2 * Real.cos B = 4)
  (h2 : (1/2) * Real.sin B + Real.cos A = (Real.sqrt 3) / 2)
  (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 :=
begin
  sorry
end

end measure_of_angle_C_in_triangle_l311_311558


namespace sum_of_divisors_of_29_l311_311879

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311879


namespace limit_of_function_l311_311909

open Real

noncomputable def limit_function := λ x : ℝ, (2 - x / 3) ^ (sin (π * x))

theorem limit_of_function : tendsto limit_function (𝓝 3) (𝓝 1) :=
by
  sorry

end limit_of_function_l311_311909


namespace sunday_deliveries_l311_311186

def base_pay(hourly_rate : ℕ) (hours : ℕ) : ℕ := hourly_rate * hours
def tips_earned(tip_per_customer : ℕ) (customers : ℕ) : ℕ := tip_per_customer * customers

variables (total_earnings weekend_earnings : ℕ) 
variables (hours_sat hours_sun customers_sat customers_sun : ℕ) 
variables (hourly_rate tip_per_customer : ℕ)

theorem sunday_deliveries :
  hourly_rate = 10 ∧
  tip_per_customer = 5 ∧
  hours_sat = 4 ∧
  customers_sat = 5 ∧
  hours_sun = 5 ∧
  total_earnings = 155 →
  customers_sun = 8 :=
begin
  sorry
end

end sunday_deliveries_l311_311186


namespace relationship_among_abc_l311_311152

noncomputable def a : ℝ := (1/4)^(1/2)
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := (1/3)^(1/2)

theorem relationship_among_abc : b > c ∧ c > a :=
by
  -- Proof will go here
  sorry

end relationship_among_abc_l311_311152


namespace sum_of_divisors_of_29_l311_311853

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311853


namespace line_intersection_midpoint_l311_311278

theorem line_intersection_midpoint (b : ℝ) :
  let midpoint := ((3 + 6) / 2, (6 + 12) / 2) in
  let (x, y) := midpoint in
  x + y = b ↔ b = 13.5 :=
by
  sorry

end line_intersection_midpoint_l311_311278


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311622

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311622


namespace semi_major_axis_ratio_l311_311166

-- We need to define all conditions as parameters and then state the main theorem.
variables (T_earth T_mars m n : ℝ)
variables (k : ℝ) -- the constant from Kepler's law

-- Given conditions
-- The ratio of the orbital periods
axiom ratio_T : T_mars / T_earth = 9 / 5
-- Kepler's third law for Earth and Mars
axiom keplers_law_earth : k = T_earth^2 / m^3
axiom keplers_law_mars : k = T_mars^2 / n^3

-- Theorem statement: Prove that m / n = (25 / 81)^(1/3)
theorem semi_major_axis_ratio : m / n = real.cbrt (25 / 81) := 
sorry

end semi_major_axis_ratio_l311_311166


namespace parallelogram_area_l311_311616

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l311_311616


namespace Chekalinsky_wins_l311_311505

-- Define the state space as a vector of 13 booleans (each representing the state of a card: face up or face down)
def State := vector Bool 13

-- Define the initial state function
def initial_state : State := vector.repeat ff 13 -- assume initial state is all face down (false)

-- Define the transition function (flipping a card)
def flip_card (s : State) (i : Fin 13) : State :=
  s.update_nth i (bnot (s.nth i))

-- Define a function to check for repetitions in the history of states
def has_repeated (history : list State) : Prop :=
  history ≠ list.nodup history

-- Define the game outcome
inductive Result
| win_Chekalinsky
| win_Hermann

-- Define the main theorem: Chekalinsky has a winning strategy
theorem Chekalinsky_wins : ∃ (strategy : (list State) → State → Fin 13), ∀ (history : list State), 
  let next_state := flip_card (history.head) (strategy history (history.head)) in
  has_repeated (next_state :: history) → Result.win_Chekalinsky := sorry

end Chekalinsky_wins_l311_311505


namespace gina_total_cost_l311_311020

-- Define the constants based on the conditions
def total_credits : ℕ := 18
def reg_credits : ℕ := 12
def reg_cost_per_credit : ℕ := 450
def lab_credits : ℕ := 6
def lab_cost_per_credit : ℕ := 550
def num_textbooks : ℕ := 3
def textbook_cost : ℕ := 150
def num_online_resources : ℕ := 4
def online_resource_cost : ℕ := 95
def facilities_fee : ℕ := 200
def lab_fee_per_credit : ℕ := 75

-- Calculating the total cost
noncomputable def total_cost : ℕ :=
  (reg_credits * reg_cost_per_credit) +
  (lab_credits * lab_cost_per_credit) +
  (num_textbooks * textbook_cost) +
  (num_online_resources * online_resource_cost) +
  facilities_fee +
  (lab_credits * lab_fee_per_credit)

-- The proof problem to show that the total cost is 10180
theorem gina_total_cost : total_cost = 10180 := by
  sorry

end gina_total_cost_l311_311020


namespace simplify_cubed_roots_l311_311213

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311213


namespace area_of_WXYZ_l311_311112

-- Definitions of the given problem
structure Parallelogram (A D E H : Point) : Prop :=
(is_parallelogram : parallelogram A D E H)
(length_AD : distance A D = 6)
(length_HE : distance H E = 6)
(trisect_AD : ∃ B C, trisect B C A D)
(trisect_HE : ∃ G F, trisect G F H E)
(length_BH : distance B H = 2 * real.sqrt 2)

-- Definition of the area of quadrilateral WXYZ
def area_WXYZ (A D E H B C G F W X Y Z : Point)
    (h_parallelogram : Parallelogram A D E H)
    (trisect_AD : ∃ B C, trisect B C A D)
    (trisect_HE : ∃ G F, trisect G F H E)
    (point_intersections : quadrilateral_intersections W X Y Z A B C D E F G H) : real :=
  3 * real.sqrt 2

-- Theorem statement
theorem area_of_WXYZ {A D E H B C G F W X Y Z: Point}
    (h_parallelogram : Parallelogram A D E H)
    (trisect_AD : ∃ B C, trisect B C A D)
    (trisect_HE : ∃ G F, trisect G F H E)
    (h_BH : distance B H = 2 * real.sqrt 2)
    (point_intersections : quadrilateral_intersections W X Y Z A B C D E F G H) :
  area_WXYZ A D E H B C G F W X Y Z h_parallelogram trisect_AD trisect_HE point_intersections = 3 * real.sqrt 2 :=
sorry

end area_of_WXYZ_l311_311112


namespace intersection_is_equilateral_triangle_l311_311665

noncomputable def circle_eq (x y : ℝ) := x^2 + (y - 1)^2 = 1
noncomputable def ellipse_eq (x y : ℝ) := 9*x^2 + (y + 1)^2 = 9

theorem intersection_is_equilateral_triangle :
  ∀ A B C : ℝ × ℝ, circle_eq A.1 A.2 ∧ ellipse_eq A.1 A.2 ∧
                 circle_eq B.1 B.2 ∧ ellipse_eq B.1 B.2 ∧
                 circle_eq C.1 C.2 ∧ ellipse_eq C.1 C.2 → 
                 (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end intersection_is_equilateral_triangle_l311_311665


namespace gcd_two_powers_l311_311311

def m : ℕ := 2 ^ 1998 - 1
def n : ℕ := 2 ^ 1989 - 1

theorem gcd_two_powers :
  Nat.gcd (2 ^ 1998 - 1) (2 ^ 1989 - 1) = 511 := 
sorry

end gcd_two_powers_l311_311311


namespace sum_of_divisors_of_29_l311_311794

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311794


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311623

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l311_311623


namespace problem_statement_l311_311051

-- Definitions based on provided conditions
def polar_curve (rho θ : ℝ) : Prop := rho - 4 * sin θ = 0

def parametric_line (t : ℝ) : ℝ × ℝ := (1 - (real.sqrt 2 / 2) * t, (real.sqrt 2 / 2) * t)

-- Prove that given the conditions, the rectangular equation of curve C and the distance 
-- difference |MA| - |MB| satisfies the given property

theorem problem_statement :
  (∀ (ρ θ : ℝ), polar_curve ρ θ -> (∃ (x y : ℝ), x^2 + y^2 - 4 * y = 0)) ∧
  (∀ (t1 t2 : ℝ), 
     t1 + t2 = 3 * real.sqrt 2 ∧ t1 * t2 = 1 -> 
     abs ((1 - (real.sqrt 2 / 2) * t1) - (1 - (real.sqrt 2 / 2) * t2)) = real.sqrt 14) := 
by sorry

end problem_statement_l311_311051


namespace max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l311_311085

noncomputable def max_xy (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then x * y else 0

noncomputable def min_y_over_x_plus_4_over_y (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then y / x + 4 / y else 0

theorem max_xy_is_2 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → max_xy x y = 2 :=
by
  intros x y hx hy hxy
  sorry

theorem min_y_over_x_plus_4_over_y_is_4 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → min_y_over_x_plus_4_over_y x y = 4 :=
by
  intros x y hx hy hxy
  sorry

end max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l311_311085


namespace distance_from_dorm_to_city_l311_311122

theorem distance_from_dorm_to_city (D : ℚ) (h1 : (1/3) * D = (1/3) * D) (h2 : (3/5) * D = (3/5) * D) (h3 : D - ((1 / 3) * D + (3 / 5) * D) = 2) :
  D = 30 := 
by sorry

end distance_from_dorm_to_city_l311_311122


namespace monthly_rent_l311_311949

theorem monthly_rent (cost : ℕ) (maintenance_percentage : ℚ) (annual_taxes : ℕ) (desired_return_rate : ℚ) (monthly_rent : ℚ) :
  cost = 20000 ∧
  maintenance_percentage = 0.10 ∧
  annual_taxes = 460 ∧
  desired_return_rate = 0.06 →
  monthly_rent = 153.70 := 
sorry

end monthly_rent_l311_311949


namespace probability_of_snow_at_most_3_days_l311_311284

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ := 
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def probability_at_most_3_days (p : ℝ) (n : ℕ) : ℝ :=
  (binomial_probability n 0 p) + (binomial_probability n 1 p) + (binomial_probability n 2 p) + (binomial_probability n 3 p)

theorem probability_of_snow_at_most_3_days : 
  let p : ℝ := 1 / 5,
      n : ℕ := 31 in
  abs (probability_at_most_3_days p n - 0.342) < 0.001 := 
sorry

end probability_of_snow_at_most_3_days_l311_311284


namespace sum_of_divisors_of_29_l311_311753

theorem sum_of_divisors_of_29 :
  let divisors := {d : ℕ | d > 0 ∧ 29 % d = 0}
  sum divisors = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311753


namespace sum_of_divisors_of_prime_29_l311_311844

theorem sum_of_divisors_of_prime_29 :
  (∀ d : Nat, d ∣ 29 → d > 0 → d = 1 ∨ d = 29) →
  let divisors := {d : Nat | d ∣ 29 ∧ d > 0}
  let sum_divisors := divisors.sum
  sum_divisors = 30 :=
by
  sorry

end sum_of_divisors_of_prime_29_l311_311844


namespace cos_A_given_conditions_l311_311533

variable (A B C a b c : ℝ)

theorem cos_A_given_conditions 
(h1 : a^2 - b^2 = sqrt 3 * b * c) 
(h2 : sin C = 2 * sqrt 3 * sin B) 
(h3 : A + B + C = π) :
cos A = sqrt 3 / 2 := 
sorry

end cos_A_given_conditions_l311_311533


namespace factorable_polynomial_l311_311995

theorem factorable_polynomial (m : ℤ) :
  (∃ A B C D E F : ℤ, 
    (A * D = 1 ∧ E + B = 4 ∧ C + F = 2 ∧ F + 3 * E + C = m + m^2 - 16)
    ∧ ((A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4 * x * y + 2 * x + m * y + m^2 - 16)) ↔
  (m = 5 ∨ m = -6) :=
by
  sorry

end factorable_polynomial_l311_311995


namespace gcd_A_C_gcd_B_C_l311_311991

def A : ℕ := 177^5 + 30621 * 173^3 - 173^5
def B : ℕ := 173^5 + 30621 * 177^3 - 177^5
def C : ℕ := 173^4 + 30621^2 + 177^4

theorem gcd_A_C : Nat.gcd A C = 30637 := sorry

theorem gcd_B_C : Nat.gcd B C = 30637 := sorry

end gcd_A_C_gcd_B_C_l311_311991


namespace coefficients_sum_binomial_coefficients_sum_l311_311553

theorem coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = coeff_sum) : coeff_sum = 729 := 
sorry

theorem binomial_coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = binom_coeff_sum) : binom_coeff_sum = 64 := 
sorry

end coefficients_sum_binomial_coefficients_sum_l311_311553


namespace find_k_l311_311483

theorem find_k (k : ℤ) :
  (-x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) → k = -16 :=
by
  sorry

end find_k_l311_311483


namespace sum_of_divisors_of_29_l311_311801

theorem sum_of_divisors_of_29 : 
  ∑ d in {1, 29}, d = 30 :=
by
  sorry

end sum_of_divisors_of_29_l311_311801


namespace consecutive_numbers_expression_l311_311338

theorem consecutive_numbers_expression (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y - 1) (h3 : z = 2) :
  2 * x + 3 * y + 3 * z = 8 * y - 1 :=
by
  -- substitute the conditions and simplify
  sorry

end consecutive_numbers_expression_l311_311338


namespace initial_oranges_correct_l311_311257

-- Define constants for the conditions
def oranges_shared : ℕ := 4
def oranges_left : ℕ := 42

-- Define the initial number of oranges
def initial_oranges : ℕ := oranges_left + oranges_shared

-- The theorem to prove
theorem initial_oranges_correct : initial_oranges = 46 :=
by 
  sorry  -- Proof to be provided

end initial_oranges_correct_l311_311257


namespace minimize_quadratic_l311_311313

-- Given quadratic equation
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The statement that we want to prove
theorem minimize_quadratic : ∃ x : ℝ, (∀ y : ℝ, f(x) ≤ f(y)) ∧ x = 3 := by
  sorry

end minimize_quadratic_l311_311313


namespace probability_of_region_A_l311_311030

def region_M (x y : ℝ) : Prop := y ≤ -x^2 + 2*x ∧ y ≥ 0
def region_A (x y : ℝ) : Prop := y ≤ x ∧ x + y ≤ 2 ∧ y ≥ 0

theorem probability_of_region_A :
  (∫ x in 0..2, (-x^2 + 2*x)) ≈ (4/3) → 
  (∫ x in 0..1, ∫ y in 0..(1-x), 1) = 1 → 
  ∃ (P : ℝ × ℝ), region_M P.1 P.2 ∧ region_A P.1 P.2 ∧ 
  P = (3/4) := 
by
  sorry

end probability_of_region_A_l311_311030


namespace unattainable_value_l311_311437

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) : 
  ¬ ∃ y : ℝ, y = (1 - x) / (3 * x + 4) ∧ y = -1/3 :=
by
  sorry

end unattainable_value_l311_311437


namespace lava_lamp_probability_l311_311182

/-

Ryan has 4 red lava lamps and 5 blue lava lamps. He arranges them in a row on a shelf randomly, and then randomly turns 4 of them on. 
Prove that the probability that the middle lamp (5th position in a row of 9) is blue and on, and the second to last lamp (8th position) is red and off is 35/143.

-/

theorem lava_lamp_probability :
  let red := 4,
      blue := 5,
      positions := 9,
      on := 4 in
  let total_ways := λ n k, nat.choose n k in
  let success_ways := λ n k b, (nat.choose (n-1) k - nat.choose (n-1-k) k + b) * nat.choose (n-1) k in
  (success_ways positions on blue / (total_ways positions on : ℝ)) = (35/143: ℝ) :=
sorry

end lava_lamp_probability_l311_311182


namespace T1_T2_T3_main_l311_311458

-- Definitions based on conditions
variable (n : ℕ) (n_ge_3 : n ≥ 3)
def maa := Prop
def pib := Set maa

-- Postulates as axioms
axiom P1 : ∀ (p : pib), ∀ (m : maa), m ∈ p
axiom P2 : ∀ (p1 p2 : pib), p1 ≠ p2 → ∃! m : maa, m ∈ p1 ∧ m ∈ p2
axiom P3 : ∀ (m : maa), ∃ p1 p2 : pib, p1 ≠ p2 ∧ m ∈ p1 ∧ m ∈ p2

-- Theorems to prove
theorem T1 : ∃ k : ℕ, k = n * (n - 1) / 2 := sorry
theorem T2 : ∀ (p : pib), ∃ k : ℕ, k = 2 * (n * (n - 1) / 2) / n := sorry
theorem T3 : ∃ k : ℕ, k = n * (2 * (n * (n - 1) / 2) / n) := sorry

-- Main theorem that all Ts are deducible from the postulates
theorem main : T1 n ∧ T2 n ∧ T3 n := sorry

end T1_T2_T3_main_l311_311458


namespace power_function_result_l311_311528
noncomputable def f (x : ℝ) (k : ℝ) (n : ℝ) : ℝ := k * x ^ n

theorem power_function_result (k n : ℝ) (h1 : f 27 k n = 3) : f 8 k (1/3) = 2 :=
by 
  sorry

end power_function_result_l311_311528


namespace sum_of_divisors_29_l311_311867

-- We define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- We define the sum_of_divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ m, m ∣ n) (finset.range (n + 1))).sum id

-- We state the theorem
theorem sum_of_divisors_29 : is_prime 29 → sum_of_divisors 29 = 30 := sorry

end sum_of_divisors_29_l311_311867


namespace extremum_at_two_intervals_of_monotonicity_max_value_condition_l311_311491

def f (x a : ℝ) : ℝ := x - (1 / 2) * a * x^2 - real.log (1 + x)

-- Problem 1
theorem extremum_at_two (a : ℝ) : 
  (deriv (λ x, f x a) 2 = 0) ↔ a = 1 / 3 := by
  sorry

-- Problem 2
theorem intervals_of_monotonicity (a : ℝ) : 
  (∀ x : ℝ, deriv (λ x, f x a) x < 0 → x ∈ intervals_of_decrease a) ∧
  (∀ x : ℝ, deriv (λ x, f x a) x > 0 → x ∈ intervals_of_increase a) := by
  sorry

-- Problem 3
theorem max_value_condition (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f x a ≤ 0) ↔ 1 ≤ a := by
  sorry

end extremum_at_two_intervals_of_monotonicity_max_value_condition_l311_311491


namespace shirts_sold_correct_l311_311183

-- Define the conditions
def shoes_sold := 6
def cost_per_shoe := 3
def earnings_per_person := 27
def total_earnings := 2 * earnings_per_person
def earnings_from_shoes := shoes_sold * cost_per_shoe
def cost_per_shirt := 2
def earnings_from_shirts := total_earnings - earnings_from_shoes

-- Define the total number of shirts sold and the target value to prove
def shirts_sold : Nat := earnings_from_shirts / cost_per_shirt

-- Prove that shirts_sold is 18
theorem shirts_sold_correct : shirts_sold = 18 := by
  sorry

end shirts_sold_correct_l311_311183


namespace simplify_sqrt_l311_311196

theorem simplify_sqrt {a b c d : ℝ} (h1 : a = 1 + 27) (h2 : b = 27) (h3 : c = 1 + 3) (h4 : d = 28 * 4) :
  (real.cbrt a) * (real.cbrt c) = real.cbrt d :=
by {
  sorry
}

end simplify_sqrt_l311_311196


namespace sum_of_divisors_of_29_l311_311849

theorem sum_of_divisors_of_29 : ∀ (n : ℕ), Prime n → n = 29 → ∑ d in (Finset.filter (∣) (Finset.range (n + 1))), d = 30 :=
by
  intro n
  intro hn_prime
  intro hn_eq_29
  rw [hn_eq_29]
  sorry

end sum_of_divisors_of_29_l311_311849


namespace sum_of_divisors_prime_29_l311_311723

theorem sum_of_divisors_prime_29 : ∑ d in (finset.filter (λ d : ℕ, 29 % d = 0) (finset.range 30)), d = 30 :=
by
  sorry

end sum_of_divisors_prime_29_l311_311723


namespace exchange_event_min_participants_l311_311421

theorem exchange_event_min_participants (A : Finset ℕ) (h : ∀ a ∈ A, 6 ∣ a) :
  ∃ n : ℕ, (∀ x ∈ A, ∃ (p : Finset (Finset ℕ)), -- p is a set of sets representing the pairs
    (∀ pair ∈ p, finset.card pair = 2) ∧ -- each pair has exactly two participants
    (∀ pair₁ pair₂ ∈ p, pair₁ ≠ pair₂ → pair₁ ∩ pair₂ = ∅) ∧ -- no two pairs share the same participant
    (∀ (p1 ∈ p) (p2 ∈ p), p1 ≠ p2 → (Finset.univ.bUnion p).card ≤ n) ∧ -- defining pair condition
    ∃ f : ℕ → ℕ, ∀ a ∈ A, f a = a) ∧ -- function f gives match count
    n = (A.max' (Finset.nonempty_of_finset_mem (A.exists_mem_of_finset_nonempty A))).div 2 + 3 :=
sorry

end exchange_event_min_participants_l311_311421


namespace quadratic_roots_relationship_l311_311595

theorem quadratic_roots_relationship 
  (a b c α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0)
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end quadratic_roots_relationship_l311_311595


namespace range_of_a_l311_311526

theorem range_of_a 
  (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = x^3 - a * x^2 + (a - 1) * x + 1)
  (h₂ : ∀ x ∈ Ioo 1 4, deriv f x < 0)
  (h₃ : ∀ x ∈ Ioi 6, deriv f x > 0) :
  5 ≤ a ∧ a ≤ 7 := sorry

end range_of_a_l311_311526


namespace earnings_from_cauliflower_correct_l311_311599

-- Define the earnings from each vegetable
def earnings_from_broccoli : ℕ := 57
def earnings_from_carrots : ℕ := 2 * earnings_from_broccoli
def earnings_from_spinach : ℕ := (earnings_from_carrots / 2) + 16
def total_earnings : ℕ := 380

-- Define the total earnings from vegetables other than cauliflower
def earnings_from_others : ℕ := earnings_from_broccoli + earnings_from_carrots + earnings_from_spinach

-- Define the earnings from cauliflower
def earnings_from_cauliflower : ℕ := total_earnings - earnings_from_others

-- Theorem to prove the earnings from cauliflower
theorem earnings_from_cauliflower_correct : earnings_from_cauliflower = 136 :=
by
  sorry

end earnings_from_cauliflower_correct_l311_311599


namespace charlie_share_l311_311908

variable (A B C : ℝ)

theorem charlie_share :
  A = (1/3) * B →
  B = (1/2) * C →
  A + B + C = 10000 →
  C = 6000 :=
by
  intros hA hB hSum
  sorry

end charlie_share_l311_311908


namespace parallelogram_height_l311_311431

theorem parallelogram_height (A B H : ℝ) (hA : A = 462) (hB : B = 22) (hArea : A = B * H) : H = 21 :=
by
  sorry

end parallelogram_height_l311_311431


namespace geometric_sequence_common_ratio_l311_311468

theorem geometric_sequence_common_ratio 
    (S_n : ℕ → ℝ) 
    (a_n : ℕ → ℝ)
    (m : ℕ) (h_m : m > 0)
    (h1 : S_n 2m / S_n m = 9)
    (h2 : a_n 2m / a_n m = (5 * m + 1) / (m - 1)) :
    ∃ q : ℝ, ∀ n : ℕ, a_n n = a_n 0 * q^(n - 1) :=
begin
  sorry
end

end geometric_sequence_common_ratio_l311_311468


namespace total_playtime_l311_311579

-- Conditions
def lena_playtime_hours : ℝ := 3.5
def minutes_per_hour : ℝ := 60
def lena_playtime_minutes : ℝ := lena_playtime_hours * minutes_per_hour
def brother_playtime_extra_minutes : ℝ := 17
def brother_playtime_minutes : ℝ := lena_playtime_minutes + brother_playtime_extra_minutes

-- Proof problem
theorem total_playtime : lena_playtime_minutes + brother_playtime_minutes = 437 := by
  sorry

end total_playtime_l311_311579


namespace sum_of_divisors_of_29_l311_311887

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311887


namespace sum_of_divisors_of_29_l311_311714

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311714


namespace sum_of_divisors_of_29_l311_311717

theorem sum_of_divisors_of_29 : (∑ d in {1, 29}, d) = 30 := by
  sorry

end sum_of_divisors_of_29_l311_311717


namespace parallelogram_area_l311_311615

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l311_311615


namespace carols_rectangle_width_l311_311979

theorem carols_rectangle_width :
  ∀ (W : ℝ),
    (∀ (A_Carol A_Jordan : ℝ),
      (5 * W = A_Carol) ∧ (4 * 30 = A_Jordan) ∧ (A_Carol = A_Jordan)
    ) → W = 24 :=
by
  intros W h
  rcases h with ⟨A_Carol, A_Jordan, h1, h2, h3⟩
  sorry

end carols_rectangle_width_l311_311979


namespace sarah_homework_problems_l311_311913

theorem sarah_homework_problems (math_pages reading_pages problems_per_page : ℕ) 
  (h1 : math_pages = 4) 
  (h2 : reading_pages = 6) 
  (h3 : problems_per_page = 4) : 
  (math_pages + reading_pages) * problems_per_page = 40 :=
by 
  sorry

end sarah_homework_problems_l311_311913


namespace cube_root_multiplication_l311_311199

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311199


namespace units_digit_S_54321_l311_311142

noncomputable def p := 2 + Real.sqrt 3
noncomputable def q := 2 - Real.sqrt 3

def S : ℕ → ℝ
| 0       => 1
| 1       => 2
| (n + 2) => 4 * S (n + 1) - S n

def units_digit (n : ℕ) : ℕ := (S n).toInt % 10

theorem units_digit_S_54321 : units_digit 54321 = 6 :=
  sorry

end units_digit_S_54321_l311_311142


namespace cube_root_multiplication_l311_311204

theorem cube_root_multiplication :
  (∛(1 + 27)) * (∛(1 + ∛27)) = ∛112 :=
by sorry

end cube_root_multiplication_l311_311204


namespace AD_length_l311_311113

-- Define the points and distances according to the conditions
variables {A B C D : Type*}
variables [EuclideanGeometry] -- Assume an appropriate Euclidean Geometry context
variables (AB BC CD AD : ℝ)
variables (right_angle_B right_angle_C : Prop)

-- Given conditions
def conditions := 
  AB = 3 ∧ 
  BC = 10 ∧ 
  CD = 25 ∧ 
  right_angle_B ∧ 
  right_angle_C

-- The proof problem: to show AD = sqrt 584 under the given conditions
theorem AD_length (h : conditions) : AD = Real.sqrt 584 :=
sorry

end AD_length_l311_311113


namespace sin_cos_sum_val_l311_311475

theorem sin_cos_sum_val (θ : ℝ) (h : sin θ ^ 3 + cos θ ^ 3 = 11 / 16) : sin θ + cos θ = 1 / 2 :=
sorry

end sin_cos_sum_val_l311_311475


namespace sum_of_divisors_of_29_l311_311882

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem sum_of_divisors_of_29 :
  is_prime 29 → sum_of_divisors 29 = 30 :=
by
  intro h_prime
  have h := h_prime
  sorry

end sum_of_divisors_of_29_l311_311882


namespace num_int_pairs_number_of_solutions_l311_311148

theorem num_int_pairs (a b : ℤ) : 
  (∃ ω : ℂ, ω^4 = 1) →
  (|a * (complex.I) + b| = 1) ↔ (a, b) ∈ {(-1, 0), (1, 0), (0, -1), (0, 1)} :=
by
  sorry

theorem number_of_solutions : 
  ∃ ω : ℂ, ω^4 = 1 → 
  ∃ (n : ℕ), n = 4 :=
by
  sorry

end num_int_pairs_number_of_solutions_l311_311148


namespace simplify_cubed_roots_l311_311215

theorem simplify_cubed_roots : 
  (Real.cbrt (1 + 27)) * (Real.cbrt (1 + Real.cbrt 27)) = Real.cbrt 28 * Real.cbrt 4 := 
by 
  sorry

end simplify_cubed_roots_l311_311215


namespace courtyard_width_l311_311929

theorem courtyard_width (length : ℕ) (brick_length brick_width : ℕ) (num_bricks : ℕ) (W : ℕ)
  (H1 : length = 25)
  (H2 : brick_length = 20)
  (H3 : brick_width = 10)
  (H4 : num_bricks = 18750)
  (H5 : 2500 * (W * 100) = num_bricks * (brick_length * brick_width)) :
  W = 15 :=
by sorry

end courtyard_width_l311_311929


namespace machinery_spent_correct_l311_311134

def raw_materials : ℝ := 3000
def total_amount : ℝ := 5714.29
def cash (total : ℝ) : ℝ := 0.30 * total
def machinery_spent (total : ℝ) (raw : ℝ) : ℝ := total - raw - cash total

theorem machinery_spent_correct :
  machinery_spent total_amount raw_materials = 1000 := 
  by
    sorry

end machinery_spent_correct_l311_311134
