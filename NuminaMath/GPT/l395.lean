import Complex
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Conics
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.Calculus.SpecificFunctions
import Mathlib.Analysis.Other.Hyperbola
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Inverse
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinations
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial.Derivative
import Mathlib.Data.Polynomial.Vieta
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Time
import Mathlib.Geometry.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Probability.MassFunction
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilitySpace
import Mathlib.SetTheory.Ordinal.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Nonempty
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Algebra.Algebra
import Mathlib.Topology.Basic
import Mathlibimport_tax +
import data.nat.basic
import data.nat.prime

namespace g_n_conjecture_range_a_induction_inequality_l395_395319
noncomputable theory
open Real

-- Define f and g
def f (x : ℝ) : ℝ := ln (1 + x)
def g (x : ℝ) : ℝ := x / (1 + x)
def g_n : ℕ → ℝ → ℝ
| 1     := g
| (n+1) := λ x, g (g_n n x)

-- (1) Conjectured expression for g_n(x).
theorem g_n_conjecture (x : ℝ) (n : ℕ) (h : x ≥ 0) : 
  g_n n x = x / (1 + n * x) := sorry

-- (2) Range of the real number a in the inequality f(x) ≥ a * g(x)
theorem range_a (a : ℝ) : (∀ x : ℝ, x ≥ 0 → f(x) ≥ a * g(x)) ↔ a ≤ 1 := sorry

-- (3) Prove the inequality by induction
theorem induction_inequality (n : ℕ) (h : 0 < n) : 
  (finset.range n).sum (λ i, g (i + 1)) > n - f n := sorry

end g_n_conjecture_range_a_induction_inequality_l395_395319


namespace find_a_b_l395_395619

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395619


namespace work_days_l395_395451

theorem work_days (W : ℝ) 
  (hx : W / 20)
  (hy : W / 12)
  (total_days : 10) : 
  ∃ d : ℝ, d = 4 ∧ 
  d * (W / 20) + (10 - d) * (W / 20 + W / 12) = W :=
by {
  use 4,
  sorry -- proof to be provided 
}

end work_days_l395_395451


namespace find_a_b_l395_395583

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395583


namespace greatest_distance_eq_3sqrt7_l395_395769

open Complex

noncomputable def A : Set ℂ :=
  {z | z^3 = 27}

noncomputable def B : Set ℂ :=
  {z | z^2 = 6 * z - 9}

theorem greatest_distance_eq_3sqrt7 :
  ∃ a ∈ A, ∃ b ∈ B, ∀ x ∈ A, ∀ y ∈ B, dist x y ≤ dist a b ∧ dist a b = 3 * Real.sqrt 7 :=
sorry

end greatest_distance_eq_3sqrt7_l395_395769


namespace urn_probability_is_4_over_21_l395_395524

theorem urn_probability_is_4_over_21 :
  let urn_after_five_operations : ℕ × ℕ := (2, 1) in
  -- After 5 operations:
  let final_urn : ℕ × ℕ := (3, 5) in
  -- Probability calculation
  (∀ (operations : list (ℕ × ℕ)),
    operations.length = 5 →
    operations.foldl
      (λ (urn : ℕ × ℕ) (draw : ℕ × ℕ),
        let (r, b) := urn in
        let (dr, db) := draw in
        if dr = 1 then (r + 1, b) else (r, b + 1))
      urn_after_five_operations = final_urn →
    -- Calculate the probability of this sequence
    let probability := (5:ℕ) * ((2:ℚ)/3 * (1/4) * (2/5) * (1/2) * (4/7)) in
    probability = (4/21:ℚ)) :=
sorry

end urn_probability_is_4_over_21_l395_395524


namespace canoe_prob_calc_l395_395474

theorem canoe_prob_calc : 
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_both_work := p_left_works * p_right_works
  let p_left_works_right_breaks := p_left_works * p_right_breaks
  let p_left_breaks_right_works := p_left_breaks * p_right_works
  let p_can_row := p_both_work + p_left_works_right_breaks + p_left_breaks_right_works
  p_left_works = 3 / 5 → 
  p_right_works = 3 / 5 → 
  p_can_row = 21 / 25 :=
by
  intros
  unfold p_left_works p_right_works p_left_breaks p_right_breaks p_both_work p_left_works_right_breaks p_left_breaks_right_works p_can_row
  sorry

end canoe_prob_calc_l395_395474


namespace chess_player_max_consecutive_win_prob_l395_395935

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l395_395935


namespace length_CD_l395_395278

noncomputable def triangle_sides (AB BC CA BD : ℝ) : Prop :=
  AB = 5 ∧ BC = 6 ∧ CA = 6 ∧ BD = 2.5

noncomputable def angle_property (α: ℝ) : Prop :=
  α = real.arccos (5 / 12)

theorem length_CD (AB BC CA BD AD CD: ℝ) (h : triangle_sides AB BC CA BD) (h_angle : angle_property (real.arccos (5 / 12))) :
  CD = 7.4 :=
by
  -- h provides: AB = 5, BC = 6, CA = 6, BD = 2.5
  -- The proof involves calculations using the Law of Cosines, which is omitted here.
  sorry

end length_CD_l395_395278


namespace fred_allowance_is_16_l395_395173

def fred_weekly_allowance (A : ℕ) : Prop :=
  (A / 2) + 6 = 14

theorem fred_allowance_is_16 : ∃ A : ℕ, fred_weekly_allowance A ∧ A = 16 := 
by
  -- Proof can be filled here
  sorry

end fred_allowance_is_16_l395_395173


namespace outfit_with_diff_colors_l395_395728

def shirts := 5
def pants := 5
def hats := 5
def colors := 5

def condition_all_diff_colors (s_color: Fin colors) (p_color: Fin colors) (h_color: Fin colors) : Prop :=
  s_color ≠ p_color ∧ p_color ≠ h_color ∧ s_color ≠ h_color

theorem outfit_with_diff_colors :
  let total_combinations := shirts * pants * hats in
  let combinations_that_are_not_valid := 3 * colors * (colors - 1) in
  let valid_combinations := total_combinations - combinations_that_are_not_valid in
  valid_combinations = 65 :=
by
  sorry

end outfit_with_diff_colors_l395_395728


namespace vectors_parallel_x_squared_eq_two_l395_395723

theorem vectors_parallel_x_squared_eq_two (x : ℝ) 
  (a : ℝ × ℝ := (x+2, 1+x)) 
  (b : ℝ × ℝ := (x-2, 1-x)) 
  (parallel : (a.1 * b.2 - a.2 * b.1) = 0) : x^2 = 2 :=
sorry

end vectors_parallel_x_squared_eq_two_l395_395723


namespace solve_for_a_b_l395_395635

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395635


namespace cookies_left_l395_395170

theorem cookies_left (days_baking : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) (frank_eats_per_day : ℕ) (ted_eats_on_sixth_day : ℕ) :
  trays_per_day * cookies_per_tray * days_baking - frank_eats_per_day * days_baking - ted_eats_on_sixth_day = 134 :=
by
  have days_baking := 6
  have trays_per_day := 2
  have cookies_per_tray := 12
  have frank_eats_per_day := 1
  have ted_eats_on_sixth_day := 4
  sorry

end cookies_left_l395_395170


namespace find_a_and_b_l395_395594

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395594


namespace train_speed_l395_395447

theorem train_speed (l t : ℕ) (h_length : l = 800) (h_time : t = 20) : 
  l / t = 40 :=
by
  rw [h_length, h_time]
  norm_num
  sorry

end train_speed_l395_395447


namespace alcohol_percentage_l395_395477

-- Define the volumes of the individual solutions and the total volume
variables (V1 V2 : ℝ) (C1 C2 : ℝ) (V : ℝ)

-- Define the percentage concentrations of the individual solutions
def final_alcohol_percentage (V1 V2 C1 C2 V : ℝ) : ℝ :=
  ((C1 * V1 + C2 * V2) / V) * 100

-- Define the conditions
def conditions : Prop :=
  V1 = 75 ∧ V = 200 ∧ V2 = (V - V1) ∧ C1 = 0.20 ∧ C2 = 0.12

-- Define the theorem that proves the final percentage of alcohol
theorem alcohol_percentage (V1 V2 : ℝ) (C1 C2 : ℝ) (V : ℝ) 
  (h: conditions) : 
  final_alcohol_percentage V1 V2 C1 C2 V = 15 := 
  by sorry

end alcohol_percentage_l395_395477


namespace product_of_five_consecutive_numbers_not_square_l395_395352

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l395_395352


namespace even_and_monotonically_decreasing_function_l395_395115

-- Definitions of the functions to examine
def f1 (x : ℝ) : ℝ := x ^ (-2)
def f2 (x : ℝ) : ℝ := x ^ 4
def f3 (x : ℝ) : ℝ := x ^ (1/2)
def f4 (x : ℝ) : ℝ := -x ^ (1/3)

-- Condition definitions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := 
  ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y

-- The main theorem statement
theorem even_and_monotonically_decreasing_function :
  (is_even f1 ∧ is_monotonically_decreasing_on f1 (Set.Ioo 0 1)) ∧ 
  ¬ (is_even f2 ∧ is_monotonically_decreasing_on f2 (Set.Ioo 0 1)) ∧
  ¬ (is_even f3 ∧ is_monotonically_decreasing_on f3 (Set.Ioo 0 1)) ∧
  ¬ (is_even f4 ∧ is_monotonically_decreasing_on f4 (Set.Ioo 0 1)) := 
by { 
  sorry 
}

end even_and_monotonically_decreasing_function_l395_395115


namespace find_p_l395_395716

def U : set ℕ := {1, 2, 3, 4}

def M (p : ℝ) : set ℝ := {x | x^2 - 5 * x + p = 0}

def complement_U_M (p : ℝ) : set ℝ := U \ M p

theorem find_p (p : ℝ) : complement_U_M p = {2, 3} → p = 4 := by
  intro h
  have hM : M p = {1, 4} := by sorry
  have s := calc 1 + 4 : by sorry
  have p_value := calc 1 * 4 : by sorry
  rw p_value
  exact rfl

end find_p_l395_395716


namespace angle_exterior_l395_395772

theorem angle_exterior (DEF_line : ∀ D E F: ℕ, collinear D E F) 
  (angle_DEG : ∠DEF G = 150) 
  (angle_EFG : ∠ EFG = 50) : 
  ∠ EGF = 100 := 
by
  sorry

end angle_exterior_l395_395772


namespace remainder_of_x50_div_x_plus_1_cubed_l395_395161

theorem remainder_of_x50_div_x_plus_1_cubed (x : ℚ) : 
  (x ^ 50) % ((x + 1) ^ 3) = 1225 * x ^ 2 + 2450 * x + 1176 :=
by sorry

end remainder_of_x50_div_x_plus_1_cubed_l395_395161


namespace ab_bisects_cd_l395_395420

-- Assuming we have points A, B, C, D and line segments as required by the conditions:
variables {A B C D O : Point}
variables (circle1 circle2 : Circle)
variables (common_tangent : Line)

-- Circle intersection conditions
axiom circles_intersect : Intersects circle1 circle2 A B
axiom tangent_touches_circles : Tangent common_tangent circle1 C ∧ Tangent common_tangent circle2 D
axiom segment_CD : LineSegment C D

-- The theorem to prove that AB bisects CD (i.e., OC = OD)
theorem ab_bisects_cd (hAB : Line A B) (hO : O ∈ hAB ∧ O ∈ segment_CD) :
  distance O C = distance O D :=
begin
  sorry
end

end ab_bisects_cd_l395_395420


namespace minimum_people_for_shared_birthday_l395_395230

theorem minimum_people_for_shared_birthday (days_in_year : ℕ) (n_people : ℕ) : 
  days_in_year = 366 → n_people = 367 → 
  ∃ (n : ℕ), n ≥ n_people ∧ ∀ (d : ℕ), d < days_in_year → ∃ (x y : ℕ), x ≠ y ∧ x < n ∧ y < n ∧ f x = f y :=
by
  sorry

end minimum_people_for_shared_birthday_l395_395230


namespace incorrect_statement_l395_395238

-- Establish basic vector properties.
variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (a b : α)

-- Assume a and b are non-zero vectors and opposite.
def opposite_vectors (a b : α) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ k * a = -b

-- The goal to prove.
theorem incorrect_statement (h : opposite_vectors a b) : 
  ¬(∥a∥ ≠ ∥b∥) :=
by 
  sorry

end incorrect_statement_l395_395238


namespace Tim_bodyguards_weekly_pay_l395_395884

theorem Tim_bodyguards_weekly_pay :
  let hourly_rate := 20
  let num_bodyguards := 2
  let daily_hours := 8
  let weekly_days := 7
  Tim pays $2240 in a week := (hourly_rate * num_bodyguards * daily_hours * weekly_days = 2240) :=
begin
  sorry
end

end Tim_bodyguards_weekly_pay_l395_395884


namespace num_non_congruent_tris_l395_395692

theorem num_non_congruent_tris (a b : ℕ) (h1 : a ≤ 2) (h2 : 2 ≤ b) (h3 : a + 2 > b) (h4 : a + b > 2) (h5 : b + 2 > a) : 
  ∃ q, q = 3 := 
by 
  use 3 
  sorry

end num_non_congruent_tris_l395_395692


namespace book_arrangement_count_l395_395871

def Book := String

def math_book : Book := "Math"
def english_books : List Book := ["English1", "English2"]
def music_books : List Book := ["Music1", "Music2", "Music3"]

def is_adjacent (a b : Book) (l : List Book) : Prop :=
  ∃ (i : ℕ), (i < l.length - 1) ∧ (l.nth i = some a) ∧ (l.nth (i + 1) = some b)

def books_not_adjacent (books : List Book) (l : List Book) : Prop :=
  ∀ (a b : Book), (a ∈ books) ∧ (b ∈ books) → ¬ is_adjacent a b l

theorem book_arrangement_count : 
  ∃ (l : List Book), (books_not_adjacent english_books l) ∧ 
                     (books_not_adjacent music_books l) ∧
                     (list.perm l [math_book] ++ english_books ++ music_books) ∧
                     (list.length l = 6) ∧  -- ensuring all books are used
                     list.perm l (math_book :: english_books ++ music_books) = 120 := 
sorry

end book_arrangement_count_l395_395871


namespace expression_value_l395_395534

theorem expression_value : 0.01 ^ (-1/2) + 8 ^ (2/3) + 2 ^ (Real.log 5 / Real.log 4) = 14 + Real.sqrt 5 := 
by
  sorry

end expression_value_l395_395534


namespace identify_conic_section_is_hyperbola_l395_395542

theorem identify_conic_section_is_hyperbola :
  ∀ x y : ℝ, x^2 - 16 * y^2 - 10 * x + 4 * y + 36 = 0 →
  (∃ a b h c d k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ h = 0 ∧ (x - c)^2 / a^2 - (y - d)^2 / b^2 = k) :=
by
  sorry

end identify_conic_section_is_hyperbola_l395_395542


namespace product_of_five_consecutive_numbers_not_square_l395_395353

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l395_395353


namespace determine_d_l395_395740

theorem determine_d (d c : ℕ) (hlcm : Nat.lcm 76 d = 456) (hhcf : Nat.gcd 76 d = c) : d = 24 :=
by
  sorry

end determine_d_l395_395740


namespace solution_of_system_l395_395367

-- Definitions of conditions
def condition1 (x y : ℝ) : Prop := x^2 + y^2 + real.sqrt 3 * x * y = 20 + 8 * real.sqrt 3 
def condition2 (y z : ℝ) : Prop := y^2 + z^2 = 13
def condition3 (x z : ℝ) : Prop := z^2 + x^2 + x * z = 37

-- All variables are positive
def positive (x y z : ℝ) : Prop := x > 0 ∧ y > 0 ∧ z > 0

-- Given system of equations
def system_equations := 
  ∃ (x y z : ℝ), positive x y z ∧ condition1 x y ∧ condition2 y z ∧ condition3 x z ∧ (x, y, z) = (4, 2, 3)

-- The Proof problem in Lean 4 statement
theorem solution_of_system : system_equations :=
sorry

end solution_of_system_l395_395367


namespace remainder_103_107_div_11_l395_395055

theorem remainder_103_107_div_11 :
  (103 * 107) % 11 = 10 :=
by
  sorry

end remainder_103_107_div_11_l395_395055


namespace train_speed_correct_l395_395044

noncomputable def speed_of_slower_train
  (l : ℕ)
  (speed_faster_train : ℕ)
  (time_to_pass : ℕ)
  (distance : ℕ)
  (relative_speed : ℕ) : Prop :=
  l = 500 ∧
  speed_faster_train = 45 ∧
  time_to_pass = 60 ∧
  distance = 1000 ∧
  relative_speed = 60 ∧
  relative_speed = speed_faster_train + v

noncomputable def question := 
  (speed_of_slower_train 500 45 60 1000 60)

#eval question -- expected to output true if the statement matches all conditions correctly.

theorem train_speed_correct : question → v = 15 := sorry

end train_speed_correct_l395_395044


namespace maximize_prob_of_consecutive_wins_l395_395950

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l395_395950


namespace votes_cast_l395_395752

theorem votes_cast (total_votes : ℕ) 
  (h1 : (3/8 : ℚ) * total_votes = 45)
  (h2 : (1/4 : ℚ) * total_votes = (1/4 : ℚ) * 120) : 
  total_votes = 120 := 
by
  sorry

end votes_cast_l395_395752


namespace Rihanna_money_left_l395_395825

theorem Rihanna_money_left (initial_money mango_count juice_count mango_price juice_price : ℕ)
  (h_initial : initial_money = 50)
  (h_mango_count : mango_count = 6)
  (h_juice_count : juice_count = 6)
  (h_mango_price : mango_price = 3)
  (h_juice_price : juice_price = 3) :
  initial_money - (mango_count * mango_price + juice_count * juice_price) = 14 :=
sorry

end Rihanna_money_left_l395_395825


namespace total_vehicles_l395_395998

-- Define the conditions
def num_trucks_per_lane := 60
def num_lanes := 4
def total_trucks := num_trucks_per_lane * num_lanes
def num_cars_per_lane := 2 * total_trucks
def total_cars := num_cars_per_lane * num_lanes

-- Prove the total number of vehicles in all lanes
theorem total_vehicles : total_trucks + total_cars = 2160 := by
  sorry

end total_vehicles_l395_395998


namespace point_of_intersection_bisects_angle_l395_395781

theorem point_of_intersection_bisects_angle {A O D B C O₁ O₂ : Point} 
  (R₁ R₂ : ℝ) (L : Point)
  (h1 : ∠ A O B = ∠ C O D)
  (h2 : CircleInAngle A O B O₁ R₁)
  (h3 : CircleInAngle C O D O₂ R₂)
  (h_tangents : TangentsIntersectAt L O₁ O₂)
  (h_sim : SimilarTriangles (OB : Triangle O O₁) (OC : Triangle O O₂)) :
  OnBisector L (∠ A O D) :=
sorry

end point_of_intersection_bisects_angle_l395_395781


namespace max_area_of_rectangular_enclosure_l395_395348

open Nat

theorem max_area_of_rectangular_enclosure : ∃ (l w : ℕ), 
  Prime l ∧ Prime w ∧ l + w = 20 ∧ l * w = 91 :=
by
  sorry

end max_area_of_rectangular_enclosure_l395_395348


namespace chloe_first_round_points_l395_395126

variable (P : ℤ)
variable (totalPoints : ℤ := 86)
variable (secondRoundPoints : ℤ := 50)
variable (lastRoundLoss : ℤ := 4)

theorem chloe_first_round_points 
  (h : P + secondRoundPoints - lastRoundLoss = totalPoints) : 
  P = 40 := by
  sorry

end chloe_first_round_points_l395_395126


namespace BN_perpendicular_to_MN_l395_395222

-- Definitions of points and midpoints in a triangle with given equilateral triangles
universe u
variables (A B C X Y Z: Type u) [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [IsEquilateralTriangle ABZ] [IsEquilateralTriangle BCX] [IsEquilateralTriangle CAZ]
variables (B' N' M N: Type u) 

-- Definition of midpoints
variables (midpoint_BC : Midpoint B C B')
variables (midpoint_CY : Midpoint C Y N')
variables (midpoint_AZ : Midpoint A Z M)
variables (midpoint_CX : Midpoint C X N)

-- Theorem to be proved
theorem BN_perpendicular_to_MN : ⊥ {
  sorry
}

end BN_perpendicular_to_MN_l395_395222


namespace circle_center_radius_l395_395376

theorem circle_center_radius :
  ∃ (h: ℝ) (k: ℝ) (r: ℝ),
    2 * (x : ℝ) ^ 2 + 2 * (y : ℝ) ^ 2 + 6 * x + -4 * y - 3 = 0 ↔ 
      (x + h) ^ 2 + (y + k) ^ 2 = r * r ∧ h = -3 / 2 ∧ k = 1 ∧ r = real.sqrt (19) / 2 :=
begin
  sorry
end

end circle_center_radius_l395_395376


namespace find_a_and_b_l395_395604

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395604


namespace parallel_lines_of_intersecting_circles_l395_395797

theorem parallel_lines_of_intersecting_circles
  (Γ₁ Γ₂ : Circle)
  (A B C D E F : Point)
  (hΓ₁A : A ∈ Γ₁)
  (hΓ₁B : B ∈ Γ₁)
  (hΓ₂A : A ∈ Γ₂)
  (hΓ₂B : B ∈ Γ₂)
  (hΓ₂C : C ∈ Γ₂)
  (hΓ₂D : D ∈ Γ₂)
  (hCAE : line CA ≠ line AE ∧ E ∈ Γ₁ ∧ E ≠ A ∧ E ∈ line CA)
  (hBDF : line BD ≠ line DF ∧ F ∈ Γ₁ ∧ F ≠ B ∧ F ∈ line BD)
  (hnotAeqB : A ≠ B) :
  parallel (line CD) (line EF) :=
sorry

end parallel_lines_of_intersecting_circles_l395_395797


namespace playground_area_l395_395397

theorem playground_area (w l : ℕ) (h1 : 2 * l + 2 * w = 72) (h2 : l = 3 * w) : l * w = 243 := by
  sorry

end playground_area_l395_395397


namespace find_AX_l395_395271

theorem find_AX
  (AB AC BC : ℚ)
  (H : AB = 80)
  (H1 : AC = 50)
  (H2 : BC = 30)
  (angle_bisector_theorem_1 : ∀ (AX XC y : ℚ), AX = 8 * y ∧ XC = 3 * y ∧ 11 * y = AC → y = 50 / 11)
  (angle_bisector_theorem_2 : ∀ (BD DC z : ℚ), BD = 8 * z ∧ DC = 5 * z ∧ 13 * z = BC → z = 30 / 13) :
  AX = 400 / 11 := 
sorry

end find_AX_l395_395271


namespace distance_B_to_AM_l395_395307

-- Define the given conditions
variables {A B C D M N : Point}
variables (angle_BAC_gt : angle A B C > 90)
variables (perpendicular_AD_BC : is_perpendicular A D B C)
variables (midpoint_M : midpoint M B C)
variables (midpoint_N : midpoint N B D)
variables (AC_len : dist A C = 2)
variables (angle_equal : angle B A N = angle M A C)
variables (equality_ABBC_AM : dist A B * dist B C = dist A M)

-- Define the goal to compute the distance from B to the line defined by A and M
theorem distance_B_to_AM : distance (B : Point) (line_through (A : Point) (M : Point)) = sqrt 285 / 38 :=
by
  sorry

end distance_B_to_AM_l395_395307


namespace max_adjacent_differently_colored_cells_l395_395186

theorem max_adjacent_differently_colored_cells 
    (n : ℕ) 
    (grid : Fin n → Fin n → Bool) 
    (cols_have_equal_black_cells : ∃ x, ∀ i, (∑ j, if grid i j then 1 else 0) = x) 
    (rows_have_different_black_cells : ∀ i j, i ≠ j → (∑ k, if grid i k then 1 else 0) ≠ (∑ k, if grid j k then 1 else 0)) 
    (h_dim : n = 100) : 
    (∃ max_adjacent : ℕ, max_adjacent = 14601) :=
by {
  sorry
}

end max_adjacent_differently_colored_cells_l395_395186


namespace problem_proof_l395_395658

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395658


namespace movie_theater_people_l395_395475

def totalSeats : ℕ := 750
def emptySeats : ℕ := 218
def peopleWatching := totalSeats - emptySeats

theorem movie_theater_people :
  peopleWatching = 532 := by
  sorry

end movie_theater_people_l395_395475


namespace ratio_of_functions_l395_395732

def f (x : ℕ) : ℕ := 3 * x + 4
def g (x : ℕ) : ℕ := 4 * x - 3

theorem ratio_of_functions :
  f (g (f 3)) * 121 = 151 * g (f (g 3)) :=
by
  sorry

end ratio_of_functions_l395_395732


namespace solve_inequality_l395_395134

theorem solve_inequality (x : ℝ) :
  (2 < (3 * x) / (4 * x - 7) ∧ (3 * x) / (4 * x - 7) ≤ 9) ↔ x ∈ set.Ioc (21/11 : ℝ) (14/5 : ℝ) := sorry

end solve_inequality_l395_395134


namespace sodium_in_salt_l395_395282

theorem sodium_in_salt (x : ℕ) (sodium_salt : 2 * x) (sodium_parmesan : 8 * 25 = 200) 
  (reduction_by_third : (2 / 3) * (2 * x + 200)) 
  (reduction_parmesan : 4 * 25 = 100) 
  (new_total_sodium : 2 * x + 100) : 2 * (2 / 3) * (2 * x + 200) = 2 * x + 100 → x =  50 := 
by
  intros h
  sorry

end sodium_in_salt_l395_395282


namespace gain_percent_correct_l395_395001

variable (MP : ℝ)

def CP : ℝ := 0.64 * MP
def SP : ℝ := 0.86 * MP

def gain : ℝ := SP MP - CP MP
def gain_percent : ℝ := (gain MP / CP MP) * 100

theorem gain_percent_correct : gain_percent MP = 34.375 := by
  sorry

end gain_percent_correct_l395_395001


namespace midpoint_coordinates_l395_395766

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  (5 * Real.cos θ, 4 * Real.sin θ)

def line (x : ℝ) : ℝ :=
  (4 / 5) * (x - 3)

theorem midpoint_coordinates (θ θ' : ℝ) (hx : 5 * Real.cos θ = (3 / 2)) (hx' : 5 * Real.cos θ' = (3 / 2)):
  let x := (5 * Real.cos θ + 5 * Real.cos θ') / 2
  let y := (line (5 * Real.cos θ) + line (5 * Real.cos θ')) / 2
  (x, y) = ((3 / 2), -(6 / 5)) :=
by
  sorry

end midpoint_coordinates_l395_395766


namespace maximize_prob_of_consecutive_wins_l395_395945

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l395_395945


namespace xiao_ming_total_evaluation_score_l395_395063

theorem xiao_ming_total_evaluation_score 
  (regular midterm final : ℤ) (weight_regular weight_midterm weight_final : ℕ)
  (h1 : regular = 80)
  (h2 : midterm = 90)
  (h3 : final = 85)
  (h_weight_regular : weight_regular = 3)
  (h_weight_midterm : weight_midterm = 3)
  (h_weight_final : weight_final = 4) :
  (regular * weight_regular + midterm * weight_midterm + final * weight_final) /
    (weight_regular + weight_midterm + weight_final) = 85 :=
by
  sorry

end xiao_ming_total_evaluation_score_l395_395063


namespace distinct_and_fraction_equal_l395_395806

variable {a b c : ℝ}

noncomputable def k := (a^3 + 12) / a

theorem distinct_and_fraction_equal (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h1 : ∀ x : ℝ, x = a ∨ x = b ∨ x = c → (x^3 + 12) / x = k) : 
  a^3 + b^3 + c^3 = -36 :=
by
  sorry

end distinct_and_fraction_equal_l395_395806


namespace man_l395_395093

-- Define the man's rowing speed in still water, the speed of the current, the downstream speed and headwind reduction.
def v : Real := 17.5
def speed_current : Real := 4.5
def speed_downstream : Real := 22
def headwind_reduction : Real := 1.5

-- Define the man's speed against the current and headwind.
def speed_against_current_headwind := v - speed_current - headwind_reduction

-- The statement to prove. 
theorem man's_speed_against_current_and_headwind :
  speed_against_current_headwind = 11.5 := by
  -- Using the conditions (which are already defined in lean expressions above), we can end the proof here.
  sorry

end man_l395_395093


namespace knights_won_34_games_l395_395393

theorem knights_won_34_games (W_K W_F W_W W_R W_M : ℕ) 
  (h1 : W_W > W_F)
  (h2 : W_K > W_R ∧ W_K < W_M)
  (h3 : W_R > 25)
  (h4 : W_F ∈ {15, 18, 30, 34, 36})
  (h5 : W_W ∈ {15, 18, 30, 34, 36})
  (h6 : W_R ∈ {15, 18, 30, 34, 36})
  (h7 : W_K ∈ {15, 18, 30, 34, 36})
  (h8 : W_M ∈ {15, 18, 30, 34, 36}) : 
  W_K = 34 := 
  sorry

end knights_won_34_games_l395_395393


namespace find_two_digit_number_l395_395754

-- Definitions based on the conditions
def is_two_digit_number (n : ℕ) : Prop :=
n >= 10 ∧ n < 100

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
let tens := n / 10 in
let units := n % 10 in
tens * tens + units * units

def product_of_digits (n : ℕ) : ℕ :=
let tens := n / 10 in
let units := n % 10 in
tens * units

def reversed_number (n : ℕ) : ℕ :=
let tens := n / 10 in
let units := n % 10 in
units * 10 + tens

-- Lean 4 statement for the proof problem
theorem find_two_digit_number (n : ℕ) :
  is_two_digit_number n →
  sum_of_squares_of_digits n = n + product_of_digits n →
  n - 36 = reversed_number (n - 36) →
  n = 48 ∨ n = 37 :=
sorry

end find_two_digit_number_l395_395754


namespace at_least_one_has_no_real_roots_l395_395131

theorem at_least_one_has_no_real_roots :
  ¬(∃ x : ℝ, 2 * x^2 - 12 = 10) ∨
  ¬(∃ x : ℝ, |3 * x - 2| = |x + 2|) ∨
  ¬(∃ x : ℝ, sqrt (4 * x^2 + 16) = sqrt (x^2 + x + 1)) :=
by 
  sorry

end at_least_one_has_no_real_roots_l395_395131


namespace intersection_on_median_l395_395758

theorem intersection_on_median
  {A B C M P Q T : Type*}
  [IsAcuteAngle A B C]
  [IsMedian BM]
  [Incenter P ABC ABM]
  [Incenter Q ABC CBM]
  [SecondIntersectionPointOnSegment ABP CBQ BM]
  : ∃ T, IsIntersectionPointOfCircumcircles ABP CBQ T ∧ LiesOnSegment T BM :=
by
  sorry

end intersection_on_median_l395_395758


namespace min_value_112_l395_395844

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d +
                                c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c)

theorem min_value_112 (a b c d : ℝ) (h : a + b + c + d = 8) : min_value_expr a b c d = 112 :=
  sorry

end min_value_112_l395_395844


namespace solve_for_a_b_l395_395630

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395630


namespace problem_proof_l395_395662

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395662


namespace prop_C_prop_D_l395_395441

theorem prop_C (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

theorem prop_D (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end prop_C_prop_D_l395_395441


namespace propositions_correct_l395_395783

variable {a : Nat → ℝ}
variable {S : Nat → ℝ}
variable {d : ℝ}

-- Definitions for the arithmetic sequence and its sum
def arithmetic_sum (a : Nat → ℝ) (n : Nat) : ℝ :=
  n * (a 0 + a (n - 1)) / 2

-- Condition S_n defined as the sum of first n terms
def S (n : Nat) := arithmetic_sum a n

-- Conditions given in the problem
axiom S6_gt_S7 : S 6 > S 7
axiom S7_gt_S5 : S 7 > S 5

theorem propositions_correct :
  (d < 0) ∧ (S 11 > 0) :=
by
  have h1 : d < 0 := sorry
  have h2 : S 11 > 0 := sorry
  exact ⟨h1, h2⟩

end propositions_correct_l395_395783


namespace problem_solution_l395_395979

def radius_formula (r1 r2 : ℝ) : ℝ := (r1 * r2) / ((Real.sqrt r1 + Real.sqrt r2) ^ 2)

def L0_r1 : ℝ := 70^2
def L0_r2 : ℝ := 73^2

def S : Set ℝ :=
  let layer_0 := {L0_r1, L0_r2}
  let rec construct_layers (k : ℕ) (layers : Set ℝ) : Set ℝ :=
    if k = 0 then layers
    else
      let new_layer := (layers.pairwise_radius_construct)
      construct_layers (k - 1) (layers ∪ new_layer)
  construct_layers 6 layer_0

noncomputable def sum_inv_sqrt_r (S : Set ℝ) : ℝ :=
  S.to_finset.sum (λ r, 1 / Real.sqrt r)

theorem problem_solution : sum_inv_sqrt_r S = 143 / 14 := sorry

end problem_solution_l395_395979


namespace angle_variance_less_than_bound_l395_395340

noncomputable def angle_variance (α β γ : ℝ) : ℝ :=
  (1/3) * ((α - (2 * Real.pi / 3))^2 + (β - (2 * Real.pi / 3))^2 + (γ - (2 * Real.pi / 3))^2)

theorem angle_variance_less_than_bound (O A B C : ℝ → ℝ) :
  ∀ α β γ : ℝ, α + β + γ = 2 * Real.pi ∧ α ≥ β ∧ β ≥ γ → angle_variance α β γ < 2 * Real.pi^2 / 9 :=
by
  sorry

end angle_variance_less_than_bound_l395_395340


namespace rihanna_money_left_l395_395823

theorem rihanna_money_left :
  ∀ (initial_amount mangoes apple_juice mango_cost juice_cost : ℕ),
    initial_amount = 50 →
    mangoes = 6 →
    apple_juice = 6 →
    mango_cost = 3 →
    juice_cost = 3 →
    initial_amount - (mangoes * mango_cost + apple_juice * juice_cost) = 14 :=
begin
  intros,
  sorry
end

end rihanna_money_left_l395_395823


namespace hyperbola_asymptote_slope_l395_395399

theorem hyperbola_asymptote_slope (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (h_hyperbola : ∀ x y : ℝ, ∀ x² / a² - y² / b² = 1) 
    (A₁ : Prod (-a) 0) (A₂ : Prod a 0) (B : Prod c (b² / a)) (C : Prod c (-b² / a))
    (h_perpendicular : ((b² / a) / (c + a)) * (-(b² / a) / (c - a)) = -1) : 
    slope_asymptote = ±1 := by
  sorry

end hyperbola_asymptote_slope_l395_395399


namespace range_of_a_ge_2_l395_395735

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ (x : ℝ), x ∈ set.Icc a (2 * a) → ∃ (y : ℝ), y ∈ set.Icc a (a ^ 2) ∧ real.log (x) / real.log (a) + real.log (y) / real.log (a) = 3

theorem range_of_a_ge_2 (a : ℝ) : range_of_a a → 2 ≤ a :=
by
  intros h
  have h2 : a ≠ 0 := sorry
  have ha : a > 0 := sorry
  sorry

end range_of_a_ge_2_l395_395735


namespace cost_price_for_A_l395_395104

variable (A B C : Type) [Field A] [Field B] [Field C]

noncomputable def cost_price (CP_A : A) := 
  let CP_B := 1.20 * CP_A
  let CP_C := 1.25 * CP_B
  CP_C = (225 : A)

theorem cost_price_for_A (CP_A : A) : cost_price CP_A -> CP_A = 150 := by
  sorry

end cost_price_for_A_l395_395104


namespace find_pairs_l395_395153

theorem find_pairs (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : (m^2 - m * n + n^2 + 1) ∣ (3^(m + n) + (m + n)!))
  (h4 : (m^2 - m * n + n^2 + 1) ∣ (3^(m^3 + n^3) + (m + n))) : 
  (m, n) = (2, 2) := 
sorry

end find_pairs_l395_395153


namespace trade_balance_problem_l395_395745

noncomputable theory

variables {x y: ℝ}

def tradeDeficit1994 (x y : ℝ) : Prop := x - y = 3.8
def export1995 (y : ℝ) : ℝ := 1.11 * y
def import1995 (x : ℝ) : ℝ := 1.03 * x
def tradeDeficit1995 (x y : ℝ) : Prop := 1.03 * x - 1.11 * y = 3

theorem trade_balance_problem :
  (∃ (x y : ℝ), tradeDeficit1994 x y ∧ tradeDeficit1995 x y) → 
  ∃ (x y : ℝ), abs (y - 11.425) < 0.001 ∧ abs (x - 15.225) < 0.001 :=
begin
  sorry
end

end trade_balance_problem_l395_395745


namespace coin_difference_l395_395789

theorem coin_difference : ∀ (p : ℕ), 1 ≤ p ∧ p ≤ 999 → (10000 - 9 * 1) - (10000 - 9 * 999) = 8982 :=
by
  intro p
  intro hp
  sorry

end coin_difference_l395_395789


namespace problem1_problem2_l395_395073

-- Problem 1
theorem problem1 (m n : ℝ)
    (H : ∀ x : ℝ, mx^2 + 3 * x - 2 > 0 ↔ n < x ∧ x < 2) :
  m = -1 ∧ n = 1 :=
by sorry

-- Problem 2
theorem problem2 (a : ℝ)
    (H : ∀ x : ℝ, x^2 + (a - 1) * x - a > 0 ↔ 
        (a < -1 ∧ (x > 1 ∨ x < -a)) ∨ 
        (a = -1 ∧ x ≠ 1)) :
  (a < -1 → ∀ x : ℝ, x^2 + (a - 1) * x - a > 0 ↔ x > 1 ∨ x < -a) ∧
  (a = -1 → ∀ x : ℝ, x^2 + (a - 1) * x - a > 0 ↔ x ≠ 1) :=
by sorry

end problem1_problem2_l395_395073


namespace product_of_five_consecutive_integers_not_square_l395_395358

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l395_395358


namespace clock_angle_12_25_is_137_point_5_l395_395433

-- Define the constants
def minute_hand_angle (minutes : ℕ) : ℝ := 6 * minutes
def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℝ := 30 * hours + 0.5 * minutes

-- Define the given time
def time_12_25 : ℕ × ℕ := (12, 25)

-- Define the angle at 12:25
def angle_between_hands (time : ℕ × ℕ) : ℝ :=
  let min_angle := minute_hand_angle time.2
  let hour_angle := hour_hand_angle time.1 time.2
  if min_angle > hour_angle then min_angle - hour_angle else hour_angle - min_angle

-- The proof statement
theorem clock_angle_12_25_is_137_point_5 :
  angle_between_hands time_12_25 = 137.5 :=
by
  sorry

end clock_angle_12_25_is_137_point_5_l395_395433


namespace brian_pencils_l395_395532

theorem brian_pencils 
  (start: ℕ) (given_away: ℕ) (bought: ℕ) 
  (h_start: start = 39) 
  (h_given_away: given_away = 18) 
  (h_bought: bought = 22) : 
  start - given_away + bought = 43 := 
by {
  rw [h_start, h_given_away, h_bought],
  norm_num, 
  sorry
}

end brian_pencils_l395_395532


namespace min_value_of_f_l395_395571

noncomputable def f (x : ℝ) : ℝ := max (2 * x + 1) (5 - x)

theorem min_value_of_f : ∃ y, (∀ x : ℝ, f x ≥ y) ∧ y = 11 / 3 :=
by 
  sorry

end min_value_of_f_l395_395571


namespace line_does_not_pass_second_quadrant_l395_395741

theorem line_does_not_pass_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0 → ¬(x < 0 ∧ y > 0)) ↔ a ≤ -1 :=
by
  sorry

end line_does_not_pass_second_quadrant_l395_395741


namespace circles_coincide_l395_395196

noncomputable def centroid (points : List (ℝ × ℝ)) : ℝ × ℝ :=
  let (sum_x, sum_y, count) := points.foldl (fun (acc : ℝ × ℝ × ℝ) (p : ℝ × ℝ) =>
    (acc.1 + p.1, acc.2 + p.2, acc.3 + 1)) (0, 0, 0)
  (sum_x / count, sum_y / count)

theorem circles_coincide (A : List (ℝ × ℝ)) (R : ℝ) (hR : R > 0) :
  ∃ N, ∀ n ≥ N, let S_n := { p : ℝ × ℝ | (p.1 - centroid (filter (λ q, (dist q (0, 0)) < R) A)).1 ^ 2
                                            + (p.2 - centroid (filter (λ q, (dist q (0, 0)) < R) A)).2 ^ 2 < R ^ 2 }
                let C_n := centroid (filter (flux S_n) A)
                C_n = C_{n+1} := sorry

end circles_coincide_l395_395196


namespace distance_from_center_to_line_is_sqrt_5_l395_395734

noncomputable def distance_from_center_to_line : ℝ :=
  let a := 2
  let center := (a, a)
  let dist := abs (2 * center.1 + center.2 - 11) / (sqrt (2^2 + 1^2))
  dist

theorem distance_from_center_to_line_is_sqrt_5 : distance_from_center_to_line = Real.sqrt 5 :=
by
  sorry

end distance_from_center_to_line_is_sqrt_5_l395_395734


namespace find_a_10_l395_395381

noncomputable def arithmetic_sequence (a : ℕ → ℕ) (r : ℕ) : Prop :=
∀ n, a (n + 1) = a n + r

theorem find_a_10 {a : ℕ → ℕ} {r : ℕ} (h1 : a 0 = 1)
  (h2 : ∃ n, 2 * n = n) -- Even number of terms
  (h3 : ∑ i in finset.range (2 * n + 1), if odd i then a i else 0 = 85)
  (h4 : ∑ i in finset.range (2 * n + 1), if even i then a i else 0 = 170) :
  a 9 = 512 :=
sorry

end find_a_10_l395_395381


namespace sqrt_17_irrational_l395_395817

theorem sqrt_17_irrational : ¬ ∃ (q : ℚ), q * q = 17 := sorry

end sqrt_17_irrational_l395_395817


namespace louis_bought_five_yards_l395_395323

/-
The cost per yard of velvet fabric is $24.
The cost of the pattern is $15.
The cost per spool of thread is $3.
Louis bought two spools of thread.
The total amount spent is $141.
Prove: Louis bought 5 yards of fabric.
-/

def cost_per_yard := 24
def pattern_cost := 15
def spool_cost := 3
def num_spools := 2
def total_spent := 141

theorem louis_bought_five_yards :
  let thread_cost := num_spools * spool_cost in
  let non_fabric_cost := pattern_cost + thread_cost in
  let fabric_cost := total_spent - non_fabric_cost in
  fabric_cost / cost_per_yard = 5 :=
by
  sorry

end louis_bought_five_yards_l395_395323


namespace geometric_progression_fourth_term_l395_395130

theorem geometric_progression_fourth_term :
  let a1 := 2^(1/2)
  let a2 := 2^(1/3)
  let a3 := 2^(1/6)
  is_geometric progression [a1, a2, a3] →
  let a4 := (a3 * (a3 / a2)) in
  a4 = 1 :=
by
  sorry

end geometric_progression_fourth_term_l395_395130


namespace polynomial_factorization_l395_395558

noncomputable def polynomial_equivalence : Prop :=
  ∀ x : ℂ, (x^12 - 3*x^9 + 3*x^3 + 1) = (x + 1)^4 * (x^2 - x + 1)^4

theorem polynomial_factorization : polynomial_equivalence := by
  sorry

end polynomial_factorization_l395_395558


namespace find_a_b_l395_395624

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395624


namespace max_value_expression_l395_395302

variables {𝕜 : Type*} [NormedField 𝕜] {E : Type*} [NormedSpace 𝕜 E]
variables (a b c : E)

-- Define the norms of a, b, and c
def norm_a := 2
def norm_b := 3
def norm_c := 4

-- Define the given conditions
def cond_a : ∥a∥ = norm_a := sorry
def cond_b : ∥b∥ = norm_b := sorry
def cond_c : ∥c∥ = norm_c := sorry

-- Main theorem statement
theorem max_value_expression : 
  ∥a - 3 • b∥^2 + ∥b - 3 • c∥^2 + ∥c - 3 • a∥^2 ≤ 429 :=
by {
  simp [cond_a, cond_b, cond_c],
  sorry
}

end max_value_expression_l395_395302


namespace maximize_probability_when_C_second_game_l395_395954

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l395_395954


namespace customer_paid_amount_l395_395019

def cost_price : Real := 7239.13
def percentage_increase : Real := 0.15
def selling_price := (1 + percentage_increase) * cost_price

theorem customer_paid_amount :
  selling_price = 8325.00 :=
by
  sorry

end customer_paid_amount_l395_395019


namespace sum_of_sequence_l395_395074

theorem sum_of_sequence : 
  (∑ n in finset.range 398, (5 * n + 1) - (5 * n + 2) - (5 * n + 3) + (5 * n + 4) + (5 * n + 5)) = 1990 :=
by
  sorry

end sum_of_sequence_l395_395074


namespace value_of_x_plus_y_l395_395706

theorem value_of_x_plus_y (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l395_395706


namespace log_expression_value_l395_395436

theorem log_expression_value :
  log 10 16 + 3 * log 10 2 + 4 * log 10 5 + log 10 32 + 2 * log 10 25 = 7.806 :=
sorry

end log_expression_value_l395_395436


namespace max_prob_win_two_consecutive_is_C_l395_395963

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l395_395963


namespace range_eight_x_plus_eight_y_l395_395689

theorem range_eight_x_plus_eight_y (x y : ℝ) (h : 2^x + 2^y = 4^x + 4^y) : 1 < 8^x + 8^y ∧ 8^x + 8^y ≤ 2 :=
sorry

end range_eight_x_plus_eight_y_l395_395689


namespace positive_number_solution_exists_l395_395047

theorem positive_number_solution_exists (x : ℝ) (h₁ : 0 < x) (h₂ : (2 / 3) * x = (64 / 216) * (1 / x)) : x = 2 / 3 :=
by sorry

end positive_number_solution_exists_l395_395047


namespace probability_of_rolling_prime_on_8_sided_die_l395_395371

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def prime_probability (total_outcomes successful_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

theorem probability_of_rolling_prime_on_8_sided_die :
  let die_faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let prime_faces := {x ∈ die_faces | is_prime x}
  prime_probability 8 prime_faces.card = 1 / 2 :=
by
  -- Definitions needed to establish context but proof is skipped with sorry.
  let die_faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let prime_faces := {x ∈ die_faces | is_prime x}
  have : prime_faces.card = 4 := sorry
  have : 4 / 8 = 1 / 2 := sorry
  sorry

end probability_of_rolling_prime_on_8_sided_die_l395_395371


namespace area_rectangle_stage6_l395_395733

theorem area_rectangle_stage6 :
  let a₁ := 1
      d := 1
      n := 6
      side_length := 3
      num_squares := a₁ + (n - 1) * d
      area_square := side_length * side_length
      total_area := num_squares * area_square
  in total_area = 54 :=
by
  sorry

end area_rectangle_stage6_l395_395733


namespace seventy_fifth_percentile_of_data_set_l395_395106

-- Define the given data set
def data_set : List ℕ := [12, 34, 15, 24, 39, 25, 31, 48, 32, 36, 36, 37, 42, 50]

-- Define the percentile function
def percentile (p : ℝ) (data : List ℕ) : ℕ :=
  let sorted_data := data.qsort (≤)
  let pos := p * (sorted_data.length.toFloat) / 100
  sorted_data.nth ((⟨(pos.ceil - 1).toNat, by simp [List.length_pos_of_mem]; exact ⟨1, (List.length sorted_data).natCast⟩⟩) : Fin (sorted_data.length))

-- Define the theorem for the 75th percentile
theorem seventy_fifth_percentile_of_data_set :
  percentile 75 data_set = 39 :=
by
  sorry

end seventy_fifth_percentile_of_data_set_l395_395106


namespace find_a_b_l395_395649

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395649


namespace find_tan_of_cos_in_4th_quadrant_l395_395199

-- Given conditions
variable (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α > 3*Real.pi/2 ∧ α < 2*Real.pi)

-- Lean statement to prove the question
theorem find_tan_of_cos_in_4th_quadrant : Real.tan α = - (4 / 3) := 
by
  sorry

end find_tan_of_cos_in_4th_quadrant_l395_395199


namespace find_a_b_l395_395668

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395668


namespace angle_AKD_75_l395_395332

-- Definitions for the geometric entities
-- Assume standard Euclidean geometry with points, lines, angles, and squares

variable {a : ℝ} -- side length of the square
variable {A B C D M K : Point}
variable [Square A B C D]

-- Given conditions
axiom angle_BAM_30 : ∠ B A M = 30
axiom angle_CKM_30 : ∠ C K M = 30
axiom point_M_on_side_BC : M ∈ lineSegment B C
axiom point_K_on_side_CD : K ∈ lineSegment C D

-- The theorem to prove
theorem angle_AKD_75 : ∠ A K D = 75 :=
by sorry -- Proof is omitted, only the statement is required

end angle_AKD_75_l395_395332


namespace non_congruent_triangle_classes_count_l395_395699

theorem non_congruent_triangle_classes_count :
  ∃ (q : ℕ),
  (∀ (a b : ℕ), a ≤ 2 ∧ 2 ≤ b ∧ a + 2 > b ∧ a + b > 2 ∧ b + 2 > a → 
  (a = 1 ∧ b = 2 ∨ a = 2 ∧ (b = 2 ∨ b = 3))) ∧ q = 3 := 
by
  use 3
  intro a b
  rintro ⟨ha, hb, h1, h2, h3⟩
  split
  { intro h,
    cases h with h1 h2,
    { exact or.inl ⟨h1, hb.eq_of_le⟩ },
    cases h2 with h2 h3,
    { exact or.inr ⟨hb.eq_of_le, or.inl rfl⟩ },
    { exact or.inr ⟨hb.eq_of_le, or.inr rfl⟩ } },
  { rintro (⟨ha1, rfl⟩ | ⟨rfl, hb1⟩),
    { refine ⟨le_refl _, le_add_of_lt hb, _, _, _⟩,
      { linarith },
      { linarith },
      { linarith } },
    cases hb1 with rfl rfl,
    { refine ⟨le_refl _, le_refl _, _, _, _⟩;
      linarith },
    { refine ⟨le_refl _, le_add_of_lt _, _, _, _⟩,
      { exact nat.zero_lt_one.trans one_lt_two },
      { linarith },
      { linarith },
      { linarith } } }
  sorry

end non_congruent_triangle_classes_count_l395_395699


namespace maximize_probability_l395_395942

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l395_395942


namespace find_a_b_l395_395584

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395584


namespace max_prob_win_two_consecutive_is_C_l395_395958

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l395_395958


namespace markers_per_box_l395_395085

theorem markers_per_box
  (students : ℕ) (boxes : ℕ) (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ) (last_group_markers : ℕ)
  (h_students : students = 30)
  (h_boxes : boxes = 22)
  (h_group1_students : group1_students = 10)
  (h_group1_markers : group1_markers = 2)
  (h_group2_students : group2_students = 15)
  (h_group2_markers : group2_markers = 4)
  (h_last_group_markers : last_group_markers = 6) :
  (110 = students * ((group1_students * group1_markers + group2_students * group2_markers + (students - group1_students - group2_students) * last_group_markers)) / boxes) :=
by
  sorry

end markers_per_box_l395_395085


namespace largest_x_divides_factorial_l395_395563

theorem largest_x_divides_factorial (n : ℕ) (p : ℕ) :
  let x := (∑ k in Finset.range (n.log p + 1), n / p^k) in
  n = 2017 ∧ p = 19 → x = 111 :=
by
  intro n p
  assume h : n = 2017 ∧ p = 19
  let x := (∑ k in Finset.range (n.log p + 1), n / p^k)
  have n_eq : n = 2017 := h.left
  have p_eq : p = 19 := h.right
  have x_calculated : x = 111 := sorry
  exact x_calculated

end largest_x_divides_factorial_l395_395563


namespace area_triangle_BMN_squared_l395_395339

theorem area_triangle_BMN_squared :
  let A := (0, 0)
  let C := (20, 0)
  let B := (16, 0)
  (∃ D E M N : prod real real,
    equilateral_triangle D A B ∧
    equilateral_triangle E B C ∧
    M = midpoint A E ∧
    N = midpoint C D ∧
    4563 = (triangle_area_squared B M N)
  ) := sorry

end area_triangle_BMN_squared_l395_395339


namespace frog_jump_l395_395013

theorem frog_jump (grasshopper_jump : ℕ) (frog_extra_jump : ℕ) (h1 : grasshopper_jump = 36) (h2 : frog_extra_jump = 17) :
  (grasshopper_jump + frog_extra_jump) = 53 :=
by
  rw [h1, h2]
  -- 36 + 17 = 53
  exact Nat.add_comm 36 17

end frog_jump_l395_395013


namespace find_a_b_l395_395614

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395614


namespace right_triangle_third_side_l395_395511

theorem right_triangle_third_side (a b : ℕ) (h₁ : a = 3) (h₂ : b = 4) : 
    (∃ c : ℝ, c^2 = ↑a^2 + ↑b^2 ∧ c = 5) ∨ (∃ d : ℝ, d^2 = ↑b^2 - ↑a^2 ∧ d = real.sqrt 7) :=
by
  sorry

end right_triangle_third_side_l395_395511


namespace sum_inverse_roots_eq_l395_395801

noncomputable def poly := ∑ i in finrange 1 11, (X^(10 - i) : ℚ[X]) + (X^0 : ℚ[X]) - 15

theorem sum_inverse_roots_eq : 
  (∑ n in finrange 1 11, 1 / (1 - (rootε (10 : ℕ) n poly))) = 11 / 3 := 
by 
  sorry

end sum_inverse_roots_eq_l395_395801


namespace waiter_total_customers_l395_395520

theorem waiter_total_customers :
  ∀ (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ), tables = 6 → women_per_table = 3 → men_per_table = 5 → tables * (women_per_table + men_per_table) = 48 :=
by
  intros tables women_per_table men_per_table h_tables h_women h_men
  rw [h_tables, h_women, h_men]
  norm_num
  rfl

end waiter_total_customers_l395_395520


namespace judy_hits_percentage_l395_395138

theorem judy_hits_percentage 
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (single_hits_percentage : ℚ) :
  total_hits = 35 →
  home_runs = 1 →
  triples = 1 →
  doubles = 5 →
  single_hits_percentage = (total_hits - (home_runs + triples + doubles)) / total_hits * 100 →
  single_hits_percentage = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end judy_hits_percentage_l395_395138


namespace heptagon_diff_sides_impossibility_l395_395089

noncomputable def heptagon_inscribed (A B C D E F G : Point) (O : Point) :=
  -- A, B, C, D, E, F, G are the vertices of heptagon inscribed in circle with center O
  Circle O A ∧ Circle O B ∧ Circle O C ∧ Circle O D ∧ Circle O E ∧ Circle O F ∧ Circle O G ∧
  -- Three of the angles are 120 degrees
  angle A B C = 120 ∧ angle C D E = 120 ∧ angle E F G = 120

theorem heptagon_diff_sides_impossibility (A B C D E F G O : Point)
    (h : heptagon_inscribed A B C D E F G O):
  ¬ (distinct_lengths A B C D E F G) :=
by
  sorry

end heptagon_diff_sides_impossibility_l395_395089


namespace problem_proof_l395_395653

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395653


namespace triangle_area_example_l395_395051

def point : Type := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example : 
  triangle_area (0, 0) (0, 6) (8, 10) = 24 :=
by
  sorry

end triangle_area_example_l395_395051


namespace possible_rectangle_areas_l395_395791

def is_valid_pair (a b : ℕ) := 
  a + b = 12 ∧ a > 0 ∧ b > 0

def rectangle_area (a b : ℕ) := a * b

theorem possible_rectangle_areas :
  {area | ∃ (a b : ℕ), is_valid_pair a b ∧ area = rectangle_area a b} 
  = {11, 20, 27, 32, 35, 36} := 
by 
  sorry

end possible_rectangle_areas_l395_395791


namespace pizza_slices_left_l395_395787

theorem pizza_slices_left (total_slices john_ate : ℕ) 
  (initial_slices : total_slices = 12) 
  (john_slices : john_ate = 3) 
  (sam_ate : ¬¬(2 * john_ate = 6)) : 
  ∃ slices_left, slices_left = 3 :=
by
  sorry

end pizza_slices_left_l395_395787


namespace distance_from_M0_to_plane_l395_395157

def point := ℝ × ℝ × ℝ

def M1: point := (2, 3, 1)
def M2: point := (4, 1, -2)
def M3: point := (6, 3, 7)
def M0: point := (-5, -4, 8)

noncomputable def distance_from_point_to_plane (p : point) (a b c : point) : ℝ :=
  let (x1, y1, z1) := a
  let (x2, y2, z2) := b
  let (x3, y3, z3) := c
  let (x0, y0, z0) := p
  
  -- Vectors
  let u := (x2 - x1, y2 - y1, z2 - z1)
  let v := (x3 - x1, y3 - y1, z3 - z1)
  
  -- Normal vector from cross product
  let n := (
    u.2 * v.3 - u.3 * v.2,
    u.3 * v.1 - u.1 * v.3,
    u.1 * v.2 - u.2 * v.1
  )
  
  -- Plane equation coefficients
  let A := n.1
  let B := n.2
  let C := n.3
  let D := - (A * x1 + B * y1 + C * z1)
  
  -- Distance calculation
  (|A * x0 + B * y0 + C * z0 + D| : ℝ) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_from_M0_to_plane : 
  distance_from_point_to_plane M0 M1 M2 M3 = 11 := 
by
  -- Proof goes here
  sorry

end distance_from_M0_to_plane_l395_395157


namespace find_theta_ratio_l395_395540

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem find_theta_ratio (θ : ℝ) 
  (h : det2x2 (Real.sin θ) 2 (Real.cos θ) 3 = 0) : 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 := 
by 
  sorry

end find_theta_ratio_l395_395540


namespace solve_quadratic_l395_395366

theorem solve_quadratic (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 → ∃ r s, (x + r)^2 = s ∧ s = 40.04 :=
by
  intro h
  sorry

end solve_quadratic_l395_395366


namespace sum_of_dihedral_angles_of_tetrahedron_gt_360_degrees_l395_395343

theorem sum_of_dihedral_angles_of_tetrahedron_gt_360_degrees
  (tetrahedron : Type) 
  (V : tetrahedron → Prop) -- V is the condition that each point forms a trihedral angle
  (trihedral_gt_180 : ∀ angle ∈ V, angle > 180) -- Sum of face angles of any trihedral angle > 180°
  (tetrahedron_vertices : ∀ t: tetrahedron, ∃ (V: fin 4 -> tetrahedron), ∀ i, V i = t) -- 4 vertices of the tetrahedron
  (face_angle_counted_twice : ∀ t: tetrahedron, face_angle_counted_twice_condition t ) -- Each face angle counted twice

  : sum_of_face_angles tetrahedron > 360 :=
by sorry

end sum_of_dihedral_angles_of_tetrahedron_gt_360_degrees_l395_395343


namespace find_a_b_l395_395674

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395674


namespace monotonic_sine_omega_l395_395244

theorem monotonic_sine_omega (ω : ℝ) (hω_gt_0 : ω > 0) :
    (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 3) →
        (sin (ω * x1 + π / 6) ≤ sin (ω * x2 + π / 6))) ↔ 0 < ω ∧ ω ≤ 1 := sorry

end monotonic_sine_omega_l395_395244


namespace savedMyLife_l395_395437

def aristocratType1 (answer : String) : Prop :=
  answer = "бал"

def aristocratType2 (answer : String) : Prop :=
  answer ≠ "бал"

noncomputable def savedMyLifeStatement (aristocrat : String → Prop) : Prop :=
  ∀ (X : Prop) (answer : String), 
  (aristocrat answer) → ((answer = "бал" ↔ (aristocrat = aristocratType1 → X)) ∧ 
                         (answer ≠ "бал" ↔ (aristocrat = aristocratType2 → X)))

theorem savedMyLife (X : Prop) (aristocrat : String → Prop) (S : Prop) (answer : String) :
  S = "You are an aristocrat of type 1" → 
  (savedMyLifeStatement aristocrat) → 
  (aristocrat answer → ((answer = "бал" ↔ (S = aristocratType1 → X)) ∧ 
                        (answer ≠ "бал" ↔ (S = aristocratType2 → X)))) :=
by
  sorry

end savedMyLife_l395_395437


namespace non_adjacent_boys_arrangements_l395_395829

-- We define the number of boys and girls
def boys := 4
def girls := 6

-- The function to compute combinations C(n, k)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- The function to compute permutations P(n, k)
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- The total arrangements where 2 selected boys are not adjacent
def total_non_adjacent_arrangements : ℕ :=
  (combinations boys 2) * (combinations girls 3) * (permutations 3 3) * (permutations (3 + 1) 2)

theorem non_adjacent_boys_arrangements :
  total_non_adjacent_arrangements = 8640 := by
  sorry

end non_adjacent_boys_arrangements_l395_395829


namespace no_roots_greater_than_3_for_all_equations_l395_395548

theorem no_roots_greater_than_3_for_all_equations :
  ∀ x : ℝ, (3 * x^2 - 2 = 25 → x ≤ 3) ∧
           ((2 * x - 1)^2 = (x - 1)^2 → x ≤ 3) ∧
           (√(x^2 - 7) = √(x - 1) → x ≤ 3) := 
by
  intros x
  split
  { intro h1
    have h1' := (by linarith : 3 * x^2 = 27) /-
    Equations solved step-by-step to get roots less than or equal to 3
    So we can conclude that x is less than or equal to 3
    -/
    sorry
  }
  split
  { intro h2
    have h2' := (by linarith : (2 * x - 1)^2 = (x - 1)^2) /-
    Equations solved step-by-step to get roots less than or equal to 3
    So we can conclude that x is less than or equal to 3
    -/
    sorry
  }
  { intro h3
    have h3' := (by linarith : √(x^2 - 7) = √(x-1)) /-
    Equations solved step-by-step to get roots less than or equal to 3
    So we can conclude that x is less than or equal to 3
    -/
    sorry
  }

end no_roots_greater_than_3_for_all_equations_l395_395548


namespace solve_for_x_l395_395835

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
sorry

end solve_for_x_l395_395835


namespace victor_lost_lives_l395_395443

-- Definition of the number of lives lost by Victor
def livesLost : Nat := 14

-- The conditions given in the problem
def currentLives : Nat := 2
def difference : Nat := 12

theorem victor_lost_lives : livesLost - currentLives = difference := by
  -- We assert the conditions
  let L := livesLost
  have h1 : L - currentLives = difference := 
    calc
      L - currentLives = 14 - 2 : by rfl
                   ... = 12     : by rfl
  exact h1

end victor_lost_lives_l395_395443


namespace max_prob_two_consecutive_wins_l395_395971

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l395_395971


namespace cos_alpha_minus_pi_div_3_l395_395209

noncomputable def alpha : ℝ := sorry

-- Given conditions
axiom α_acute : 0 < alpha ∧ alpha < π / 2
axiom cos_alpha_plus_pi_div_6 : Real.cos (alpha + π / 6) = 1 / 3

-- Proof goal
theorem cos_alpha_minus_pi_div_3 : Real.cos (alpha - π / 3) = 2 * Real.sqrt 2 / 3 := sorry

end cos_alpha_minus_pi_div_3_l395_395209


namespace angular_acceleration_solution_l395_395092

variables (v r h : ℝ)

-- Define the angular velocity based on the given linear velocity and radius
def angular_velocity (v r : ℝ) : ℝ := v / r

-- Define the rate of change of radius in terms of height, angular velocity, and radius
def rate_of_change_radius (h : ℝ) (ω : ℝ) (r : ℝ) : ℝ := - (h / (2 * π)) * ω

-- Substitute angular_velocity into the rate_of_change_radius definition
def rate_of_change_radius_instantiated (v r h : ℝ) : ℝ :=
    rate_of_change_radius h (angular_velocity v r) r

-- Define the angular acceleration
def angular_acceleration (v : ℝ) (r : ℝ) (h : ℝ) : ℝ := - (v / r^2) * (rate_of_change_radius_instantiated v r h)

-- The main theorem stating that the angular acceleration is given by the correct answer
theorem angular_acceleration_solution : angular_acceleration v r h = (h * v^2) / (2 * π * r^3) :=
begin
  -- Proof is not required, so we use "sorry" to skip it
  sorry
end

end angular_acceleration_solution_l395_395092


namespace sum_of_possible_values_nk_l395_395005

theorem sum_of_possible_values_nk
  (n k : ℕ)
  (h1 : (binom n k : ℚ) * 3 = binom n (k + 1))
  (h2 : (binom n (k + 1) : ℚ) * 2 = binom n (k + 2)) :
  n + k = 12 :=
by
  sorry

end sum_of_possible_values_nk_l395_395005


namespace pizza_slices_left_l395_395788

theorem pizza_slices_left (total_slices john_ate : ℕ) 
  (initial_slices : total_slices = 12) 
  (john_slices : john_ate = 3) 
  (sam_ate : ¬¬(2 * john_ate = 6)) : 
  ∃ slices_left, slices_left = 3 :=
by
  sorry

end pizza_slices_left_l395_395788


namespace num_non_congruent_tris_l395_395693

theorem num_non_congruent_tris (a b : ℕ) (h1 : a ≤ 2) (h2 : 2 ≤ b) (h3 : a + 2 > b) (h4 : a + b > 2) (h5 : b + 2 > a) : 
  ∃ q, q = 3 := 
by 
  use 3 
  sorry

end num_non_congruent_tris_l395_395693


namespace total_amount_paid_l395_395284

-- Definitions based on the conditions.
def cost_per_pizza : ℝ := 12
def delivery_charge : ℝ := 2
def distance_threshold : ℝ := 1000 -- distance in meters
def park_distance : ℝ := 100
def building_distance : ℝ := 2000

def pizzas_at_park : ℕ := 3
def pizzas_at_building : ℕ := 2

-- The proof problem stating the total amount paid to Jimmy.
theorem total_amount_paid :
  let total_pizzas := pizzas_at_park + pizzas_at_building
  let cost_without_delivery := total_pizzas * cost_per_pizza
  let park_charge := if park_distance > distance_threshold then pizzas_at_park * delivery_charge else 0
  let building_charge := if building_distance > distance_threshold then pizzas_at_building * delivery_charge else 0
  let total_cost := cost_without_delivery + park_charge + building_charge
  total_cost = 64 :=
by
  sorry

end total_amount_paid_l395_395284


namespace percentage_selected_A_l395_395253

variable (candidates : ℕ)
variable (A_selected : ℕ)
variable (B_selected : ℕ)

-- Given conditions
def total_candidates := 8300
def B_percentage : ℝ := 0.07
def extra_B_A := 83

-- Known values
def B_selected_candidates : ℕ := (B_percentage * total_candidates).toNat
def A_selected_candidates : ℕ := B_selected_candidates - extra_B_A

-- Theorem to prove
theorem percentage_selected_A :
  ((A_selected_candidates : ℝ) / (total_candidates : ℝ)) * 100 = 6 :=
sorry

end percentage_selected_A_l395_395253


namespace necessity_not_sufficiency_l395_395168

theorem necessity_not_sufficiency (a : ℝ) (hx : 0 < a) : 
    ∀ x : ℝ, (1 / x < a) → (x ≤ 1 / a) ∨ (x ≤ 0) :=
begin
  sorry
end

end necessity_not_sufficiency_l395_395168


namespace root_product_eq_18sqrt2_l395_395428

-- Define the roots
def fourth_root_64 := real.sqrt(2)
def cube_root_27 := 3
def square_root_9 := 3

-- Define the product of the roots
def product_roots := 2 * fourth_root_64 * cube_root_27 * square_root_9

-- Theorem statement
theorem root_product_eq_18sqrt2 : product_roots = 18 * real.sqrt(2) :=
by
  sorry

end root_product_eq_18sqrt2_l395_395428


namespace perfume_price_decrease_l395_395497

theorem perfume_price_decrease :
  let original_price := 1200
  let increased_price := original_price * (1 + 10 / 100)
  let final_price := increased_price * (1 - 15 / 100)
  original_price - final_price = 78 := by
  calc
  original_price - final_price = ...
  sorry

end perfume_price_decrease_l395_395497


namespace problem_1_problem_2_problem_3_problem_3_l395_395710

open Real

-- Problem 1: Prove that f(x) is an even function
theorem problem_1 (x : ℝ) : f x = f (-x) :=
by
  let f := λ (x : ℝ), exp x + exp (-x)
  sorry

-- Problem 2: Find the range of m such that mf(x) <= e^(-x) + m - 1 for all x in (0, +infinity)
theorem problem_2 (m : ℝ) : (∀ x : ℝ, 0 < x → m * (exp x + exp (-x)) ≤ exp (-x) + m - 1) ↔ m ≤ -1 / 3 :=
by
  let f := λ (x : ℝ), exp x + exp (-x)
  sorry

-- Problem 3: Compare sizes of e^(a-1) and a^(e-1)
theorem problem_3 (a : ℝ) (h1 : (1/2)*(exp 1 + exp (-1)) < a ∧ a < exp 1) : a^(exp 1 - 1) > exp (a - 1) :=
by
  let f := λ (x : ℝ), exp x + exp (-x)
  sorry

theorem problem_3' (a : ℝ) (h2 : a > exp 1) : a^(exp 1 - 1) < exp (a - 1) :=
by
  let f := λ (x : ℝ), exp x + exp (-x)
  sorry

end problem_1_problem_2_problem_3_problem_3_l395_395710


namespace probability_of_two_negative_real_roots_l395_395501

open Set

theorem probability_of_two_negative_real_roots :
  let f : ℝ → ℝ := λ t, Polynomial.X^2 + Polynomial.C (2*t) * Polynomial.X + Polynomial.C (3*t - 2)
  ∃ (p : ℝ), (p = (interval_integral (indicator (λ t, 4*t^2 - 4*(3*t - 2) ≥ 0 ∧ -2*t < 0 ∧ 3*t - 2 > 0) (indicator (λ t, t ∈ Icc 0 5)) (interval_integral.indicator (λ t, 1) 0 5)) / (5 - 0))
    ∧ p = 2 / 3 :=
sorry

end probability_of_two_negative_real_roots_l395_395501


namespace coefficient_x2y4_l395_395268

theorem coefficient_x2y4 (x y : ℤ) : (1 + x + y^2)^5 = ∑ k : ℕ, (λ i, (λ j, (binom 5 i * (1 + x)^(5 - i) * y^(2 * i))) (5 - k) * (x^2) * (y^4) : zmod k) := sorry

end coefficient_x2y4_l395_395268


namespace card_deal_probability_l395_395176

-- Definition of the problem conditions
def num_cards_in_deck := 52
def num_hearts := 13
def num_clubs := 13
def num_face_cards := 12
def probability_heart := num_hearts / num_cards_in_deck
def probability_club := num_clubs / (num_cards_in_deck - 1)
def probability_face_card := num_face_cards / (num_cards_in_deck - 2)

-- Combined probability calculation
def combined_probability := probability_heart * probability_club * probability_face_card

-- Statement of the theorem in Lean
theorem card_deal_probability :
  combined_probability = 39 / 2550 :=
by
  -- Definitions directly come from conditions
  have p_heart := probability_heart
  have p_club := probability_club
  have p_face_card := probability_face_card
  -- Probability calculation
  have combined := combined_probability
  -- Required equality for the theorem
  sorry

end card_deal_probability_l395_395176


namespace circumscribed_circle_area_l395_395022

theorem circumscribed_circle_area (x y c : ℝ)
  (h1 : x + y + c = 24)
  (h2 : x * y = 48)
  (h3 : x^2 + y^2 = c^2) :
  ∃ R : ℝ, (x + y + 2 * R = 24) ∧ (π * R^2 = 25 * π) := 
sorry

end circumscribed_circle_area_l395_395022


namespace solve_eq_l395_395363

def K (x t : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ t then x * (t - 1)
  else if t ≤ x ∧ x ≤ 1 then t * (x - 1)
  else 0 

noncomputable def solution (x : ℝ) (λ : ℝ) : ℝ := 
  x - (λ / π) * ∑' (n : ℕ) in (Finset.range 1), ((-1) ^ (n + 1)) / (n * (λ + (n ^ 2) * π ^ 2)) * real.sin (n * π * x)

theorem solve_eq (λ : ℝ) (φ : ℝ → ℝ) :
  (∀ x, φ x - λ * ∫ t in 0..1, K x t * φ t = x) ↔ 
  (φ = λ x, x - (λ / π) * ∑' (n : ℕ) in (Finset.range 1),
    ((-1) ^ (n + 1)) / (n * (λ + (n ^ 2) * π ^ 2)) * real.sin (n * π * x)) :=
sorry

end solve_eq_l395_395363


namespace correct_calculation_l395_395904

theorem correct_calculation (a b : ℝ) :
  ¬(a^2 + 2 * a^2 = 3 * a^4) ∧
  ¬(a^6 / a^3 = a^2) ∧
  ¬((a^2)^3 = a^5) ∧
  (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l395_395904


namespace age_of_oldest_sibling_l395_395290

theorem age_of_oldest_sibling (Kay_siblings : ℕ) (Kay_age : ℕ) (youngest_sibling_age : ℕ) (oldest_sibling_age : ℕ) 
  (h1 : Kay_siblings = 14) (h2 : Kay_age = 32) (h3 : youngest_sibling_age = Kay_age / 2 - 5) 
  (h4 : oldest_sibling_age = 4 * youngest_sibling_age) : oldest_sibling_age = 44 := 
sorry

end age_of_oldest_sibling_l395_395290


namespace independence_with_replacement_not_independence_without_replacement_l395_395747

-- Definitions of the events and the probability space
def Ω : Type := {ball : ℕ // ball < 10}
def P : MeasureTheory.ProbabilityMeasure Ω := sorry
def A : Set Ω := {ball | ball < 5}
def B_with_replacement : Set Ω := {ball | ball < 5}  -- Same setup for with replacement
def B_without_replacement (a : Ω) : Set Ω :=
  if a ∈ A then {ball | ball < 4} else {ball | ball < 5}

-- Independence definitions
def independent (P : MeasureTheory.ProbabilityMeasure Ω) (A B : Set Ω) : Prop :=
  P (A ∩ B) = P A * P B

-- Problem Statement
theorem independence_with_replacement :
  independent P A B_with_replacement := sorry

theorem not_independence_without_replacement (a : Ω) :
  ¬independent P A (B_without_replacement a) := sorry

end independence_with_replacement_not_independence_without_replacement_l395_395747


namespace negation_square_positive_negation_root_equation_negation_existence_sum_positive_negation_some_primes_odd_l395_395077

theorem negation_square_positive :
  ∃ n : ℕ, n^2 ≤ 0 := sorry

theorem negation_root_equation :
  ∃ x : ℝ, 5 * x - 12 ≠ 0 := sorry

theorem negation_existence_sum_positive :
  ∃ x : ℝ, ∀ y : ℝ, x + y ≤ 0 := sorry

theorem negation_some_primes_odd :
  ∀ p : ℕ, prime p → ¬ odd p := sorry

end negation_square_positive_negation_root_equation_negation_existence_sum_positive_negation_some_primes_odd_l395_395077


namespace sumOfTrianglesIs34_l395_395369

def triangleOp (a b c : ℕ) : ℕ := a * b - c

theorem sumOfTrianglesIs34 : 
  triangleOp 3 5 2 + triangleOp 4 6 3 = 34 := 
by
  sorry

end sumOfTrianglesIs34_l395_395369


namespace find_x_in_interval_l395_395155

noncomputable def satisfy_interval (x : ℝ) : Prop :=
  4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9

theorem find_x_in_interval :
  { x : ℝ | satisfy_interval x } = { x : ℝ | x ∈ set.Ioc (63 / 26) (28 / 11) } :=
sorry

end find_x_in_interval_l395_395155


namespace number_of_valid_pairs_l395_395576

def valid_pairs_count : ℕ :=
  ∑ y in Finset.range 150 | λ y, y > 0, sorry

theorem number_of_valid_pairs : valid_pairs_count = (∑ y in Finset.range 149 | λ y, Nat.floor ((150 - y) / (y * (y + 1)))) :=
sorry

end number_of_valid_pairs_l395_395576


namespace speed_of_second_car_l395_395071

theorem speed_of_second_car (s1 s2 s : ℕ) (v1 : ℝ) (h_s1 : s1 = 500) (h_s2 : s2 = 700) 
  (h_s : s = 100) (h_v1 : v1 = 10) : 
  (∃ v2 : ℝ, v2 = 12 ∨ v2 = 16) :=
by 
  sorry

end speed_of_second_car_l395_395071


namespace arcsin_sqrt_one_half_l395_395538

theorem arcsin_sqrt_one_half : Real.arcsin (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  -- TODO: provide proof
  sorry

end arcsin_sqrt_one_half_l395_395538


namespace bicycle_cost_calculation_l395_395101

theorem bicycle_cost_calculation 
  (CP_A CP_B CP_C : ℝ)
  (h1 : CP_B = 1.20 * CP_A)
  (h2 : CP_C = 1.25 * CP_B)
  (h3 : CP_C = 225) :
  CP_A = 150 :=
by
  sorry

end bicycle_cost_calculation_l395_395101


namespace perfume_price_decrease_l395_395499

theorem perfume_price_decrease :
  let original_price := 1200
  let increased_price := original_price * (1 + 10 / 100)
  let final_price := increased_price * (1 - 15 / 100)
  original_price - final_price = 78 := by
  calc
  original_price - final_price = ...
  sorry

end perfume_price_decrease_l395_395499


namespace percentage_markup_is_23_99_l395_395395

noncomputable def selling_price : ℝ := 8091
noncomputable def cost_price : ℝ := 6525
noncomputable def markup : ℝ := selling_price - cost_price
noncomputable def percentage_markup : ℝ := (markup / cost_price) * 100

theorem percentage_markup_is_23_99 :
  percentage_markup ≈ 23.99 := by
  sorry

end percentage_markup_is_23_99_l395_395395


namespace coin_partition_l395_395328

theorem coin_partition (coins : Fin 242 → ℕ) (h_len : (Fin 242).card = 241)
  (h_sum : (Finset.univ.sum coins) = 360) : 
  ∃ (A B C : Finset (Fin 242)), 
    (A ∪ B ∪ C = Finset.univ) ∧ 
    (A ∩ B = ∅) ∧ 
    (B ∩ C = ∅) ∧ 
    (A ∩ C = ∅) ∧ 
    (Finset.univ.card = A.card + B.card + C.card) ∧ 
    (A.sum coins = B.sum coins) ∧ 
    (B.sum coins = C.sum coins) :=
sorry

end coin_partition_l395_395328


namespace sqrt_neg9_squared_l395_395456

theorem sqrt_neg9_squared : Real.sqrt ((-9: ℝ)^2) = 9 := by
  sorry

end sqrt_neg9_squared_l395_395456


namespace three_digit_even_solutions_count_l395_395726

theorem three_digit_even_solutions_count :
  (∃ (x : ℕ), 100 ≤ x ∧ x < 1000 ∧ x % 2 = 0 ∧ 5462 * x + 729 ≡ 2689 [MOD 29]) ↔ 15 := 
sorry

end three_digit_even_solutions_count_l395_395726


namespace find_a_b_l395_395615

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395615


namespace slopes_sum_l395_395685

variables {P : ℝ × ℝ} {M : ℝ × ℝ} {y1 y2 k1 k2 : ℝ}

def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

def line_through (M : ℝ × ℝ) (m y x : ℝ) : Prop :=
  x = m * y + 3

def point (x y : ℝ) : ℝ × ℝ :=
  (x, y)

def slope (P A : ℝ × ℝ) : ℝ :=
  (A.2 - P.2) / (A.1 - P.1)

axiom P_def : P = (-3, 3)

axiom M_def : M = (3, 0)

axiom intersection1 : parabola (y1^2 / 4) y1

axiom intersection2 : parabola (y2^2 / 4) y2

axiom vieta : y1 * y2 = -12

axiom slopes : k1 = slope P (y1^2 / 4, y1) ∧ k2 = slope P (y2^2 / 4, y2)

theorem slopes_sum : k1 + k2 = -1 :=
sorry

end slopes_sum_l395_395685


namespace chess_player_max_consecutive_win_prob_l395_395932

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l395_395932


namespace speed_of_second_car_l395_395042

-- Define the distances
def distance_AB : ℝ := 300   -- Distance between points A and B in km

-- Define the speeds
def speed_car_A : ℝ := 40    -- Speed of the first car in km/h

-- Define the time interval
def time : ℝ := 2            -- Time in hours

-- Define the separation after time interval
def separation_after_time : ℝ := 100  -- Distance between the two cars after 2 hours

-- Define the possible speeds of the second car
def possible_speeds_car_B : set ℝ := {60, 140}

-- The theorem statement
theorem speed_of_second_car :
  ∃ v : ℝ, v ∈ possible_speeds_car_B ∧
           (∀ d : ℝ, d = distance_AB - (speed_car_A * time + (d - separation_after_time)/time) ∨ 
                   d = (distance_AB + separation_after_time) - d → 
                   d/time = v) :=
by
  sorry


end speed_of_second_car_l395_395042


namespace triangle_AC_length_and_circumradius_l395_395761

noncomputable def isosceles_right_triangle_hypotenuse : ℝ := 10
theorem triangle_AC_length_and_circumradius (A B C : Type*) [is_right_triangle A B C] 
  (hA : angle_A_congruent_to_angle_B A B C)
  (hAB : hypotenuse_length A B C = isosceles_right_triangle_hypotenuse) :
  (AC_length A B C = 5 * sqrt 2) ∧ (circumradius A B C = 5) :=
by
  sorry

end triangle_AC_length_and_circumradius_l395_395761


namespace solution_set_f_gt_3x_plus_4_l395_395853

open Real

variables (f : ℝ → ℝ) (x : ℝ)

-- Conditions
def dom_f : Prop := ∀ x : ℝ, true
def f_neg_one : Prop := f (-1) = 1
def f_prime_gt_three : Prop := ∀ x : ℝ, (deriv f x) > 3

-- Proof statement
theorem solution_set_f_gt_3x_plus_4 (hf_domain : dom_f f) (hf_neg_one : f_neg_one f)
  (hf_prime_gt : f_prime_gt_three f) : {x : ℝ | f x > 3 * x + 4} = Ioo (-1 : ℝ) (⊤ : ℝ) :=
sorry

end solution_set_f_gt_3x_plus_4_l395_395853


namespace perfume_price_reduction_l395_395494

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end perfume_price_reduction_l395_395494


namespace chord_length_of_line_intersecting_ellipse_l395_395768

open Real

-- Definition of the ellipse C
def ellipse (x y : ℝ) : Prop :=
  x^2 + y^2 / 4 = 1

-- Definition of the intersection points of the line y = (1/2) * x with the ellipse
def is_intersection_point (x y : ℝ) : Prop :=
  y = (1 / 2) * x ∧ ellipse x y

-- Definition of the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Main theorem to prove
theorem chord_length_of_line_intersecting_ellipse :
  let C := ellipse
  let A := is_intersection_point
  let B := is_intersection_point
  ∃ x1 y1 x2 y2 : ℝ, A x1 y1 ∧ B x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2) ∧ distance x1 y1 x2 y2 = 4 :=
begin
  sorry
end

end chord_length_of_line_intersecting_ellipse_l395_395768


namespace ellipse_incenter_divides_pd_l395_395526

noncomputable def foci_positions (a c : ℝ) : ℝ × ℝ := 
  (c, -c)

noncomputable def point_on_ellipse (a b x : ℝ) : ℝ × ℝ := 
  (x, b * (1 - ((x^2) / (a^2)))^0.5)

noncomputable def triangle_incenter (a b c : ℝ) : ℝ := 
  a / (a + b + c)

theorem ellipse_incenter_divides_pd (a c : ℝ) (P : ℝ × ℝ)
    (h1 : P.1^2/a^2 + P.2^2/b^2 = 1) (h2 : 0 < a) (h3 : 0 < c) (h4 : c < a) :
    let e := c / a,
        F1 : ℝ × ℝ := foci_positions a c,
        F2 : ℝ × ℝ := (F1.1, -F1.2),
        D : ℝ × ℝ := (0, 0),
        I : ℝ := triangle_incenter 1 e 1 in
    I = 1 / e := by 
  sorry

end ellipse_incenter_divides_pd_l395_395526


namespace max_cylinder_radius_l395_395446

-- Define the given dimensions of the crate
def crate_length := 18
def crate_width := 16
def crate_height := 12

-- Define the conditions for the radius of the cylinder to fit in the crate
def cylinder_radius : ℕ := crate_width / 2

-- Prove that the radius of the cylindrical gas tank that fits the crate and 
-- has the largest possible volume is 8 feet.
theorem max_cylinder_radius : cylinder_radius = 8 :=
by 
  have h : crate_width = 16 := rfl
  simp [cylinder_radius, h]
  sorry

end max_cylinder_radius_l395_395446


namespace percentage_of_X_in_final_mixture_l395_395068

-- Define the percentages for each mixture
def percentage_ryegrass (mix : ℕ) : ℝ :=
  if mix = 1 then 40 / 100 else if mix = 2 then 25 / 100 else 0

-- Define the percentage ryegrass of the final mixture
def final_percentage_ryegrass (P : ℝ) : ℝ :=
  0.40 * P + 0.25 * (100 - P)

-- State the theorem to prove that the percentage of X in the final mixture is 66.67%
theorem percentage_of_X_in_final_mixture :
  ∃ P : ℝ, final_percentage_ryegrass P = 35 / 100 ∧ P = 66.67 :=
by
  sorry

end percentage_of_X_in_final_mixture_l395_395068


namespace number_of_valid_seating_arrangements_l395_395413

-- Define the conditions of the problem
def seats := Finset (Fin 8)
def is_valid_seating (arrangement : Finset (Fin 8)) : Prop :=
  arrangement.card = 3 ∧ 
  ∀ p ∈ arrangement, 
    (p - 1) ∈ (Finset.range 8) \ arrangement ∧ 
    (p + 1) ∈ (Finset.range 8) \ arrangement

-- The mathematical to be stated in Lean
theorem number_of_valid_seating_arrangements : 
  (∃ (arrangements : Finset (Finset (Fin 8))), 
    (∀ a ∈ arrangements, is_valid_seating a) ∧ arrangements.card = 2880) := 
sorry

end number_of_valid_seating_arrangements_l395_395413


namespace quadratic_no_real_roots_l395_395247

theorem quadratic_no_real_roots (c : ℝ) : 
  (∀ x : ℝ, ¬(x^2 + x - c = 0)) ↔ c < -1/4 := 
sorry

end quadratic_no_real_roots_l395_395247


namespace total_doughnuts_made_l395_395078

def num_doughnuts_per_box : ℕ := 10
def num_boxes_sold : ℕ := 27
def doughnuts_given_away : ℕ := 30

theorem total_doughnuts_made :
  num_boxes_sold * num_doughnuts_per_box + doughnuts_given_away = 300 :=
by
  sorry

end total_doughnuts_made_l395_395078


namespace find_y_value_l395_395236

variable (x y k : ℝ)

-- Define direct variation condition
def direct_variation (x y k : ℝ) : Prop := y = k * x

-- Define the conditions 
def conditions : Prop := direct_variation 4 8 k ∧ direct_variation (-8) y k

-- The theorem to prove the value of y when x = -8
theorem find_y_value (h : conditions) : y = -16 :=
by
  sorry

end find_y_value_l395_395236


namespace find_a_b_l395_395616

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395616


namespace find_complex_z_l395_395459

variable (z : ℂ)

theorem find_complex_z (hz1 : (3 + z) * complex.I = 1) : z = -3 - complex.I := 
by 
  sorry

end find_complex_z_l395_395459


namespace non_congruent_triangle_classes_count_l395_395698

theorem non_congruent_triangle_classes_count :
  ∃ (q : ℕ),
  (∀ (a b : ℕ), a ≤ 2 ∧ 2 ≤ b ∧ a + 2 > b ∧ a + b > 2 ∧ b + 2 > a → 
  (a = 1 ∧ b = 2 ∨ a = 2 ∧ (b = 2 ∨ b = 3))) ∧ q = 3 := 
by
  use 3
  intro a b
  rintro ⟨ha, hb, h1, h2, h3⟩
  split
  { intro h,
    cases h with h1 h2,
    { exact or.inl ⟨h1, hb.eq_of_le⟩ },
    cases h2 with h2 h3,
    { exact or.inr ⟨hb.eq_of_le, or.inl rfl⟩ },
    { exact or.inr ⟨hb.eq_of_le, or.inr rfl⟩ } },
  { rintro (⟨ha1, rfl⟩ | ⟨rfl, hb1⟩),
    { refine ⟨le_refl _, le_add_of_lt hb, _, _, _⟩,
      { linarith },
      { linarith },
      { linarith } },
    cases hb1 with rfl rfl,
    { refine ⟨le_refl _, le_refl _, _, _, _⟩;
      linarith },
    { refine ⟨le_refl _, le_add_of_lt _, _, _, _⟩,
      { exact nat.zero_lt_one.trans one_lt_two },
      { linarith },
      { linarith },
      { linarith } } }
  sorry

end non_congruent_triangle_classes_count_l395_395698


namespace determine_dimensions_l395_395031

theorem determine_dimensions (a b : ℕ) (h : a < b) 
    (h1 : ∃ (m n : ℕ), 49 * 51 = (m * a) * (n * b))
    (h2 : ∃ (p q : ℕ), 99 * 101 = (p * a) * (q * b)) : 
    a = 1 ∧ b = 3 :=
  by 
  sorry

end determine_dimensions_l395_395031


namespace find_a_and_b_l395_395602

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395602


namespace symmetrical_character_l395_395062

def is_symmetrical (char : String) : Prop := 
  sorry  -- Here the definition for symmetry will be elaborated

theorem symmetrical_character : 
  let A : String := "坡"
  let B : String := "上"
  let C : String := "草"
  let D : String := "原"
  is_symmetrical C := 
  sorry

end symmetrical_character_l395_395062


namespace count_ways_turn_off_lamps_l395_395405

theorem count_ways_turn_off_lamps : 
  let lamps_on := 7
  let positions := 6
  nat.choose positions 3 = 20 :=
by
  let lamps_on := 7
  let positions := 6
  show nat.choose positions 3 = 20
  exact nat.choose_eq_binomial positions 3 20

end count_ways_turn_off_lamps_l395_395405


namespace rectangular_solid_no_both_spheres_l395_395509

noncomputable def rectangular_solid_has_spheres 
    (length width height : ℝ) 
    (circumscribed_sphere : Prop) 
    (inscribed_sphere : Prop) : Prop :=
    ¬(length ≠ width ∨ width ≠ height ∨ height ≠ length) → 
    (circumscribed_sphere ∧ inscribed_sphere = false)

-- Define geometric properties of the rectangular solid and spheres
axiom circumscribed_sphere_exists (length width height : ℝ) (c : ℝ) :
    circumscribed_sphere (length, width, height, c)

axiom inscribed_sphere_exists (length width height : ℝ) (r : ℝ) :
    inscribed_sphere (length, width, height, r)

-- Prove the main theorem
theorem rectangular_solid_no_both_spheres 
    (length width height : ℝ) : 
    rectangular_solid_has_spheres length width height 
        (circumscribed_sphere_exists length width height 1) 
        (inscribed_sphere_exists length width height 1) :=
sorry

end rectangular_solid_no_both_spheres_l395_395509


namespace bounded_region_area_l395_395008

/-- The area of the bounded region enclosed by the curve y^2 + 2xy + 50|x| = 500 is 2500. -/
theorem bounded_region_area :
  ∃ bounded_region, is_bounded (y^2 + 2 * x * y + 50 * |x| = 500) bounded_region ∧
  area bounded_region = 2500 :=
sorry

end bounded_region_area_l395_395008


namespace smallest_of_seven_consecutive_even_numbers_sum_406_l395_395370

theorem smallest_of_seven_consecutive_even_numbers_sum_406 :
  ∃ (n : ℤ), (406 : ℤ) = 7 * n ∧ n - 6 = 52 :=
by
  use (406 : ℤ) / 7
  split
  · linarith
  · linarith
  sorry

end smallest_of_seven_consecutive_even_numbers_sum_406_l395_395370


namespace marla_songs_probability_l395_395326

theorem marla_songs_probability
  (songs : List ℕ)
  (h_len : songs.length = 15)
  (h_durations : ∀ i < 15, songs.nth i = ↑i + 2)
  (favorite_duration : ℕ)
  (h_favorite : favorite_duration = 8)
  (total_duration : ℕ)
  (h_total : total_duration = 15) :
  let no_of_ways_favorite_planned : ℕ := 31492800 -- calculated as 14! + 2 * 13!
  let total_ways : ℕ := 1307674368000 -- calculated as 15!
  no_of_ways_favorite_planned / total_ways = 1 / 5 → 
  ∃ p : ℚ, (1 - p = 4 / 5) ∧ p = no_of_ways_favorite_planned / total_ways := 
sorry

end marla_songs_probability_l395_395326


namespace find_function_satisfying_conditions_l395_395908

noncomputable def g (x : ℝ) := (1 / 3) ^ x
noncomputable def f (x : ℝ) := g x + 1

lemma exponential_function (x : ℝ) :
  ∃ (a : ℝ), g x = a ^ x ∧ 0 < a ∧ a < 1 := by
  use 1 / 3
  split
  case hRight.left => rfl
  case hRight.right.left => norm_num
  case hRight.right.right => norm_num

lemma monotonically_decreasing (x y : ℝ) (h : x ≤ y) : f x ≥ f y := by
  simp [f, g]
  have h_g: g x ≥ g y := by
    exact pow_le_pow_of_le_one (by norm_num) (by norm_num) h
  linarith

lemma negative_one_condition : f (-1) > 3 := by
  simp [f, g]
  norm_num

theorem find_function_satisfying_conditions :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), ∃ (a : ℝ), f(x) - 1 = a ^ x ∧ 0 < a ∧ a < 1) ∧
  (∀ x y : ℝ, x ≤ y → f x ≥ f y) ∧ (f (-1) > 3) := by
  use f
  split
  case hLeft =>
    intro x
    apply exponential_function
  case hRight.left =>
    exact monotonically_decreasing
  case hRight.right =>
    exact negative_one_condition

end find_function_satisfying_conditions_l395_395908


namespace constant_term_g_eq_l395_395303

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

theorem constant_term_g_eq : 
  (h.coeff 0 = 2) ∧ (f.coeff 0 = -6) →  g.coeff 0 = -1/3 := by
  sorry

end constant_term_g_eq_l395_395303


namespace average_of_numbers_is_25_l395_395872

def numbers (A1 A2 B1 B2 B3 C1 C2 C3 : ℕ) : Prop :=
  (A1 + A2) / 2 = 20 ∧
  (B1 + B2 + B3) / 3 = 26 ∧
  C1 = C2 - 4 ∧
  C1 = C3 - 6 ∧
  C3 = 30

theorem average_of_numbers_is_25 (A1 A2 B1 B2 B3 C1 C2 C3 : ℕ) (h : numbers A1 A2 B1 B2 B3 C1 C2 C3) : 
  (A1 + A2 + B1 + B2 + B3 + C1 + C2 + C3) / 8 = 25 :=
by
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h5
  sorry

end average_of_numbers_is_25_l395_395872


namespace range_of_m_l395_395738

theorem range_of_m (m : ℝ) :
  (∃ x ∈ set.Icc (0 : ℝ) 2, x^3 - 3 * x + m = 0) → m ∈ set.Icc (-2) 2 :=
by
  sorry

end range_of_m_l395_395738


namespace find_a2_l395_395573

def sequence (A : ℕ → ℤ) (k : ℕ) : ℤ := A k

def delta (A : ℕ → ℤ) (k : ℕ) : ℤ := A (k + 1) - A k

def delta_delta (A : ℕ → ℤ) (k : ℕ) : ℤ := delta A (k + 1) - delta A k

variables (A : ℕ → ℤ)
hypothesis H1 : ∀ k : ℕ, delta_delta A k = 1
hypothesis H2 : A 12 = 0
hypothesis H3 : A 22 = 0

theorem find_a2 : A 2 = 100 := by
  sorry

end find_a2_l395_395573


namespace find_a_b_l395_395650

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395650


namespace total_earrings_l395_395750

def num_women : ℕ := 800
def percentage_one_earring : ℝ := 0.03
def num_one_earring : ℕ := (percentage_one_earring * num_women).to_nat
def num_remaining : ℕ := num_women - num_one_earring
def num_two_earrings : ℕ := num_remaining / 2
def num_no_earrings : ℕ := num_remaining / 2

theorem total_earrings : num_one_earring * 1 + num_two_earrings * 2 + num_no_earrings * 0 = 800 := by
  sorry

end total_earrings_l395_395750


namespace area_bounded_by_equation_l395_395010

theorem area_bounded_by_equation : 
  (∃ (bounded_region : ℝ) (y x : ℝ), y^2 + 2 * x * y + 50 * |x| = 500) → 
  (area_of_bounded_region bounded_region = 1250) := 
by
  sorry

end area_bounded_by_equation_l395_395010


namespace sum_even_cubes_negation_l395_395122

theorem sum_even_cubes_negation : 
  (∑ i in finset.range 25, ((2 * (i + 1)) ^ 3)) + (∑ i in finset.range 25, ((-2 * (i + 1)) ^ 3)) = 0 := by
  sorry

end sum_even_cubes_negation_l395_395122


namespace sum_of_arithmetic_sequence_l395_395435

theorem sum_of_arithmetic_sequence (a d l n : ℕ) 
  (h1 : a = 1) 
  (h2 : d = 4) 
  (h3 : n = 8) 
  (h4 : l = 29) 
  (h5 : l = a + (n - 1) * d) : 
  let S_n := n / 2 * (a + l) in
  S_n = 120 :=
by
  sorry

end sum_of_arithmetic_sequence_l395_395435


namespace inequality_to_prove_l395_395189

theorem inequality_to_prove {n : ℕ} (h_n : n ≥ 2)
    (a : ℕ → ℕ) (h_inc : ∀ i j, i < j → a i < a j)
    (h_sum_le_one : ∑ i in Finset.range n, (1 : ℝ) / a i ≤ 1)
    (x : ℝ) :
    (∑ i in Finset.range n, 1 / (a i ^ 2 + x^2)) ^ 2
    ≤ (1 / 2) * (1 / (a 0 * (a 0 - 1) + x^2)) :=
sorry

end inequality_to_prove_l395_395189


namespace cube_root_of_square_root_pm8_l395_395743

theorem cube_root_of_square_root_pm8 :
  ∀ (x : ℝ), (sqrt x = 8 ∨ sqrt x = -8) → (real.cbrt x = 4) :=
begin
  sorry
end

end cube_root_of_square_root_pm8_l395_395743


namespace total_insects_eaten_l395_395462

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end total_insects_eaten_l395_395462


namespace coordinate_plane_line_l395_395913

theorem coordinate_plane_line (m n p : ℝ) (h1 : m = n / 5 - 2 / 5) (h2 : m + p = (n + 15) / 5 - 2 / 5) : p = 3 := by
  sorry

end coordinate_plane_line_l395_395913


namespace amy_flash_drive_files_l395_395523

theorem amy_flash_drive_files 
  (music_files : ℕ) (video_files : ℕ) (deleted_files : ℕ) 
  (h_music : music_files = 4) (h_video : video_files = 21) (h_deleted : deleted_files = 23):
  music_files + video_files - deleted_files = 2 :=
by
  rw [h_music, h_video, h_deleted]
  norm_num
  sorry

end amy_flash_drive_files_l395_395523


namespace connie_needs_more_money_l395_395128

variable (cost_connie : ℕ) (cost_watch : ℕ)

theorem connie_needs_more_money 
  (h_connie : cost_connie = 39)
  (h_watch : cost_watch = 55) :
  cost_watch - cost_connie = 16 :=
by sorry

end connie_needs_more_money_l395_395128


namespace officer_ways_l395_395334

noncomputable def count_officer_ways (total_members prev_office_holders : ℕ) : ℕ :=
  -- Calculate number of ways to choose officers, following counting cases as per problem.
  let case1 := total_members * (total_members - 1) * prev_office_holders * (total_members - 3)
  let case2 := 2 * (total_members - 1) * (total_members - 2) * (prev_office_holders - 1) * (total_members - 4)
  let case3 := (prev_office_holders - 1) * (prev_office_holders - 2) * (total_members - 2) * 2 * (total_members - 5)
  case1 + case2 + case3

theorem officer_ways :
  ∃ total_members prev_office_holders,
    total_members = 10 ∧ prev_office_holders = 4 ∧ count_officer_ways total_members prev_office_holders = 5592 :=
by
  use 10, 4
  dsimp [count_officer_ways]
  split
  · rfl
  split
  · rfl
  rw [Nat.mul_sub_right_distrib, Nat.mul_sub_right_distrib, Nat.mul_sub_right_distrib]
  norm_num
  sorry

end officer_ways_l395_395334


namespace negation_of_sin_le_one_l395_395715

theorem negation_of_sin_le_one : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end negation_of_sin_le_one_l395_395715


namespace positive_number_is_nine_l395_395215

theorem positive_number_is_nine (x : ℝ) (n : ℝ) (hx : x > 0) (hn : n > 0)
  (sqrt1 : x^2 = n) (sqrt2 : (x - 6)^2 = n) : 
  n = 9 :=
by
  sorry

end positive_number_is_nine_l395_395215


namespace sum_of_prime_numbers_in_Ico_2_2004_is_2026_l395_395579

def a (n : ℕ) : ℝ := Real.logBase (n + 1) (n + 2)

def is_prime_number (n : ℕ) : Prop :=
  (∏ i in Finset.range (n + 1), a i) ∈ ℤ

def prime_numbers_sum : ℝ :=
  ∑ k in Finset.Ico 2 2004, if is_prime_number k then k else 0

theorem sum_of_prime_numbers_in_Ico_2_2004_is_2026 :
  prime_numbers_sum = 2026 :=
by
  sorry

end sum_of_prime_numbers_in_Ico_2_2004_is_2026_l395_395579


namespace bounded_region_area_l395_395009

/-- The area of the bounded region enclosed by the curve y^2 + 2xy + 50|x| = 500 is 2500. -/
theorem bounded_region_area :
  ∃ bounded_region, is_bounded (y^2 + 2 * x * y + 50 * |x| = 500) bounded_region ∧
  area bounded_region = 2500 :=
sorry

end bounded_region_area_l395_395009


namespace parallelogram_area_l395_395448

-- Define the base and height of the parallelogram
def base : ℝ := 34
def height : ℝ := 18

-- With the above definitions, state the problem of proving the area
theorem parallelogram_area : base * height = 612 := by
  sorry

end parallelogram_area_l395_395448


namespace earnings_per_widget_l395_395445

-- Defining the conditions as constants
def hours_per_week : ℝ := 40
def hourly_wage : ℝ := 12.50
def total_weekly_earnings : ℝ := 700
def widgets_produced : ℝ := 1250

-- We need to prove earnings per widget
theorem earnings_per_widget :
  (total_weekly_earnings - (hours_per_week * hourly_wage)) / widgets_produced = 0.16 := by
  sorry

end earnings_per_widget_l395_395445


namespace cookies_left_l395_395169

theorem cookies_left (days_baking : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) (frank_eats_per_day : ℕ) (ted_eats_on_sixth_day : ℕ) :
  trays_per_day * cookies_per_tray * days_baking - frank_eats_per_day * days_baking - ted_eats_on_sixth_day = 134 :=
by
  have days_baking := 6
  have trays_per_day := 2
  have cookies_per_tray := 12
  have frank_eats_per_day := 1
  have ted_eats_on_sixth_day := 4
  sorry

end cookies_left_l395_395169


namespace fifth_selected_individual_is_01_l395_395860

def population : List Nat := List.range 1 21

def random_numbers : List Nat := [
  65, 72, 08, 02, 63, 14, 07, 02, 
  43, 69, 97, 28, 01, 98, 32, 04, 
  92, 34, 49, 35, 82, 00, 36, 23, 
  48, 69, 69, 38, 74, 81
]

noncomputable def valid_selection (nums : List Nat) : List Nat :=
  nums.filter (λ x => x ≥ 1 ∧ x ≤ 20)

theorem fifth_selected_individual_is_01 :
  valid_selection (random_numbers) = [08, 02, 14, 07, 01] →
  valid_selection (random_numbers).nth 4 = some 01 :=
by
  sorry

end fifth_selected_individual_is_01_l395_395860


namespace solve_for_x_l395_395836

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
sorry

end solve_for_x_l395_395836


namespace max_self_intersection_points_l395_395897

theorem max_self_intersection_points (n : ℕ) (polygon : list (ℕ × ℕ)) 
  (h1 : polygon.length = 14) 
  (h2 : ∀ (i j : ℕ) (s1 s2 : (ℕ × ℕ)), 
      i ≠ j → polygon.nth i = some s1 → polygon.nth j = some s2 → s1 ≠ s2) :
  ∃ k : ℕ, k = 17 :=
by
  sorry

end max_self_intersection_points_l395_395897


namespace angle_YXO_proof_l395_395277

-- Definitions for the given conditions
def angle_XYZ : ℝ := 80
def angle_YZX : ℝ := 78
def angle_sum_of_triangle := λ A B C : ℝ, A + B + C = 180
def bisect_angle := λ A_DIV2 B : ℝ, B = A_DIV2 / 2

-- Proof statement to be created
theorem angle_YXO_proof
  (angle_XYZ_B : angle_XYZ = 80)
  (angle_YZX_B : angle_YZX = 78)
  (in_circle : ∃ O : Type, (∃ X Y Z : Type, True)) -- Representation of in-circle condition
  (sum_triangle : angle_sum_of_triangle angle_XYZ angle_YZX (180 - angle_XYZ - angle_YZX)) :
  bisect_angle (180 - angle_XYZ - angle_YZX) 11 :=
by {
  -- To be proved
  sorry
}

end angle_YXO_proof_l395_395277


namespace initial_pencils_l395_395335

theorem initial_pencils (P : ℕ) (h1 : 84 = P - (P - 15) / 4 + 16 - 12 + 23) : P = 71 :=
by
  sorry

end initial_pencils_l395_395335


namespace number_of_satisfying_integers_l395_395097

theorem number_of_satisfying_integers : 
  { n : ℕ | n < 50 ∧ ⌊(n : ℝ) / 2⌋ + ⌊(n : ℝ) / 3⌋ + ⌊(n : ℝ) / 6⌋ = n }.card = 8 := 
by 
  sorry

end number_of_satisfying_integers_l395_395097


namespace fish_disappeared_l395_395336

theorem fish_disappeared (g : ℕ) (c : ℕ) (left : ℕ) (disappeared : ℕ) (h₁ : g = 7) (h₂ : c = 12) (h₃ : left = 15) (h₄ : g + c - left = disappeared) : disappeared = 4 :=
by
  sorry

end fish_disappeared_l395_395336


namespace final_number_after_moves_l395_395919

theorem final_number_after_moves : 
  ∀ (n : ℕ) (numbers : Fin (n + 1) → ℚ),
    (∀ k : Fin (n + 1), numbers k = (k + 1 : ℕ) / (n + 1)) →
    let transform := λ (a b : ℚ), 3 * a * b  - 2 * a - 2 * b + 2 in
    (∀ moves (nums : list ℚ), nums.perm (list.of_fn numbers) →
      ∃ final_num : ℚ,
        (∀ m : ℕ, m < moves → 
          ∃ x y : ℚ, 
            x ∈ nums ∧ y ∈ nums ∧ x ≠ y ∧ nums.replace (nums.index_of x) (transform x y) = nums.remove y) ∧
        final_num = (2 : ℚ) / (3 : ℚ)) :=
begin
  intro n,
  intro numbers,
  intro h_numbers,
  dsimp [transform],
  sorry -- Proof goes here
end

end final_number_after_moves_l395_395919


namespace kilometers_to_chains_l395_395201

theorem kilometers_to_chains :
  (1 * 10 * 50 = 500) :=
by
  sorry

end kilometers_to_chains_l395_395201


namespace find_perfect_matching_l395_395177

-- Define the boys and girls
inductive Boy | B1 | B2 | B3
inductive Girl | G1 | G2 | G3

-- Define the knowledge relationship
def knows : Boy → Girl → Prop
| Boy.B1, Girl.G1 => true
| Boy.B1, Girl.G2 => true
| Boy.B2, Girl.G1 => true
| Boy.B2, Girl.G3 => true
| Boy.B3, Girl.G2 => true
| Boy.B3, Girl.G3 => true
| _, _ => false

-- Proposition to prove
theorem find_perfect_matching :
  ∃ (pairing : Boy → Girl), 
    (∀ b : Boy, knows b (pairing b)) ∧ 
    (∀ g : Girl, ∃ b : Boy, pairing b = g) :=
by
  sorry

end find_perfect_matching_l395_395177


namespace unique_f_form_l395_395152

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_nondecreasing : Monotone f
axiom f_at_0 : f 0 = 0
axiom f_at_1 : f 1 = 1
axiom f_functional : ∀ a b : ℝ, a < 1 → 1 < b → f(a) + f(b) = f(a) * f(b) + f(a + b - a * b)

theorem unique_f_form :
  ∃ (c k : ℝ), (c > 0) ∧ (k ≥ 0) ∧
  (∀ x : ℝ, f x = if x > 1 then c * (x - 1) ^ k else if x = 1 then 1 else 1 - (1 - x) ^ k) :=
sorry

end unique_f_form_l395_395152


namespace complex_div_symmetry_l395_395737

open Complex

-- Definitions based on conditions
def z1 : ℂ := 1 + I
def z2 : ℂ := -1 + I

-- Theorem to prove
theorem complex_div_symmetry : z2 / z1 = I := by
  sorry

end complex_div_symmetry_l395_395737


namespace find_digit_e_l395_395708

theorem find_digit_e (A B C D E F : ℕ) (h1 : A * 10 + B + (C * 10 + D) = A * 10 + E) (h2 : A * 10 + B - (D * 10 + C) = A * 10 + F) : E = 9 :=
sorry

end find_digit_e_l395_395708


namespace rhombus_angle_C_l395_395264

theorem rhombus_angle_C (A B C D : Type) [rhombus ABCD] (h1 : angle A = 120) : angle C = 120 := 
sorry

end rhombus_angle_C_l395_395264


namespace probability_X_leq_1_l395_395705

open MeasureTheory Probability

noncomputable def X : ℝ → ℝ := sorry

axiom normal_distribution (μ σ : ℝ) : probability_distribution ℝ := sorry

theorem probability_X_leq_1 (μ σ : ℝ) (hμ : μ = 2) (hσ : ∃ δ, σ^2 = δ^2) (hP : P (event (λ x, 1 < x ∧ x < 3)) = 0.4) :
  P (event (λ x, x ≤ 1)) = 0.3 :=
sorry

end probability_X_leq_1_l395_395705


namespace finite_common_terms_l395_395295

section sequences

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)

noncomputable def a' : ℕ → ℕ
| 0     => a 0
| (n+1) => n * a' n + 1

noncomputable def b' : ℕ → ℕ
| 0     => b 0
| (n+1) => n * b' n - 1

theorem finite_common_terms (a0 b0 : ℕ) (ha : ∀ n, a (n + 1) = n * a n + 1) (hb : ∀ n, b (n + 1) = n * b n - 1) :
  {n : ℕ | a n = b n}.finite :=
sorry

end sequences

end finite_common_terms_l395_395295


namespace cosine_approximation_at_2_l395_395318

theorem cosine_approximation_at_2 : 
  let f (x : ℝ) := Real.cos x 
  let f1 (x : ℝ) := -Real.sin x
  let f2 (x : ℝ) := -Real.cos x
  let f3 (x : ℝ) := Real.sin x
  let f4 (x : ℝ) := Real.cos x
  in 
  (f 2) ≈ (f 0 + (f1 0 / 1!) * 2 + (f2 0 / 2!) * (2^2) + (f3 0 / 3!) * (2^3) + (f4 0 / 4!) * (2^4)) = 
  -1 / 3 :=
by sorry

end cosine_approximation_at_2_l395_395318


namespace geometric_seq_prod_eq_sqrt10_l395_395213

noncomputable def geometric_sequence_product (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 < a n) ∧ (∃ r, ∀ n, a (n + 1) = a n * r) ∧
  (∀ x, 100 * (Real.log (a 50))^2 = Real.log (100 * x) →
       100 * (Real.log (a 51))^2 = Real.log (100 * x)) ∧
  (a 1 * a 2 * · · · * a 100 = Real.sqrt 10)

-- Now, we state our theorem
theorem geometric_seq_prod_eq_sqrt10 (a : ℕ → ℝ) : geometric_sequence_product a :=
sorry

end geometric_seq_prod_eq_sqrt10_l395_395213


namespace largest_A_l395_395113

theorem largest_A (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) : A ≤ 48 :=
  sorry

end largest_A_l395_395113


namespace solve_abs_eq_l395_395365

theorem solve_abs_eq (x : ℝ) : 
  (|x - 4| + 3 * x = 12) ↔ (x = 4) :=
by
  sorry

end solve_abs_eq_l395_395365


namespace distribution_of_items_into_boxes_l395_395760

-- Define the 5 items and 8 boxes
def items := {1, 2, 3, 4, 5}
def boxes := {1, 2, 3, 4, 5, 6, 7, 8}

theorem distribution_of_items_into_boxes :
  (∃ (f : items → boxes), function.injective f) → (Π (n : ℕ), n = 5) → 
  {f : items → boxes // function.injective f}.card = 6720 :=
by
  sorry

end distribution_of_items_into_boxes_l395_395760


namespace maximize_probability_when_C_second_game_l395_395952

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l395_395952


namespace melinda_textbooks_probability_l395_395327

theorem melinda_textbooks_probability
  (total_books : ℕ)
  (math_books : ℕ)
  (box1_capacity : ℕ)
  (box2_capacity : ℕ)
  (box3_capacity : ℕ)
  (h_total_books : total_books = 15)
  (h_math_books : math_books = 4)
  (h_box1_capacity : box1_capacity = 4)
  (h_box2_capacity : box2_capacity = 5)
  (h_box3_capacity : box3_capacity = 6)
  (h_disjoint_boxes : box1_capacity + box2_capacity + box3_capacity = total_books)
  :
  let p := 27
  let q := 1759
  in p + q = 1786 :=
by {
  sorry
}

end melinda_textbooks_probability_l395_395327


namespace sin2alpha_cos2alpha_l395_395007

noncomputable def problem (a : ℝ) (α : ℝ) (x y : ℝ) (P_x P_y : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧
  y = log a (x - 3) + 2 ∧ 
  x = P_x ∧ 
  y = P_y ∧ 
  let r := real.sqrt (P_x^2 + P_y^2) in
  let sinα := P_y / r in
  let cosα := P_x / r in
  sin 2 * α + cos 2 * α = 7 / 5

theorem sin2alpha_cos2alpha (a x y : ℝ) (P : ℝ × ℝ) (α : ℝ) (h : problem a α x y P.1 P.2) :
  sin (2 * α) + cos (2 * α) = 7 / 5 :=
by
  cases h with ha h1,
  have h2 : a > 0 := ha.1,
  have h3 : a ≠ 1 := ha.2,
  have h4 := h1.1,
  have h5 := h1.2,
  have h6 := h1.3,
  have h7 := h1.4,
  have h8 := h1.5,
  let r := real.sqrt (P.1^2 + P.2^2),
  let sinα := P.2 / r,
  let cosα := P.1 / r,
  have h_sinα : sin α = sinα,
  have h_cosα : cos α = cosα,
  sorry

end sin2alpha_cos2alpha_l395_395007


namespace worker_looms_operated_l395_395756

def is_operated_by (a : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  a i j = 1

theorem worker_looms_operated (m n : ℕ) (h_m : m > 0) (h_n : n > 0)
  (a : ℕ → ℕ → ℕ)
  (H : ∀ i j, (1 ≤ i ∧ i ≤ m) ∧ (1 ≤ j ∧ j ≤ n) → (is_operated_by a i j ∨ a i j = 0)) :
  a 4 1 + a 4 2 + a 4 3 + ⋯ + a 4 n = 3 →
  ∃ ls : List ℕ, ls.length = 3 ∧ ∀ j ∈ ls, is_operated_by a 4 j :=
by
  sorry

end worker_looms_operated_l395_395756


namespace prob_bigger_number_correct_l395_395530

def bernardo_picks := {n | 1 ≤ n ∧ n ≤ 10}
def silvia_picks := {n | 1 ≤ n ∧ n ≤ 8}

noncomputable def prob_bigger_number : ℚ :=
  let prob_bern_picks_10 : ℚ := 3 / 10
  let prob_bern_not_10_larger_silvia : ℚ := 55 / 112
  let prob_bern_not_picks_10 : ℚ := 7 / 10
  prob_bern_picks_10 + prob_bern_not_10_larger_silvia * prob_bern_not_picks_10

theorem prob_bigger_number_correct :
  prob_bigger_number = 9 / 14 := by
  sorry

end prob_bigger_number_correct_l395_395530


namespace different_denominations_count_l395_395870

theorem different_denominations_count :
  ∃ (fifty_cent_coins : Nat) (five_yuan_bills : Nat) (hundred_yuan_bills : Nat),
    fifty_cent_coins = 3 ∧ five_yuan_bills = 6 ∧ hundred_yuan_bills = 4 ∧
    (fifty_cent_coins + 1) * (five_yuan_bills + 1) * (hundred_yuan_bills + 1) = 139 :=
by {
  use 3,
  use 6,
  use 4,
  simp,
  sorry -- Proof to be completed
}

end different_denominations_count_l395_395870


namespace probability_of_region_l395_395338

theorem probability_of_region :
  let area_rect := (1000: ℝ) * 1500
  let area_polygon := 500000
  let prob := area_polygon / area_rect
  prob = (1 / 3) := sorry

end probability_of_region_l395_395338


namespace sum_of_roots_l395_395568

theorem sum_of_roots (x : ℝ) (h : x^2 = 10 * x + 16) : x = 10 :=
by 
  -- Rearrange the equation to standard form: x^2 - 10x - 16 = 0
  have eqn : x^2 - 10 * x - 16 = 0 := by sorry
  -- Use the formula for the sum of the roots of a quadratic equation
  -- Prove the sum of the roots is 10
  sorry

end sum_of_roots_l395_395568


namespace log_579_between_consec_ints_l395_395029

theorem log_579_between_consec_ints (a b : ℤ) (h₁ : 2 < Real.log 579 / Real.log 10) (h₂ : Real.log 579 / Real.log 10 < 3) : a + b = 5 :=
sorry

end log_579_between_consec_ints_l395_395029


namespace circle_distance_relation_l395_395549

theorem circle_distance_relation
  (r1 r2 d : ℝ) (six_circles : list (ℝ × ℝ)) 
  (h1 : six_circles.length = 6)
  (h2 : ∀ c ∈ six_circles, ∃ four_touching, four_touching.length = 4 ∧ (c ∈ four_touching)) :
  (∃ (R1 R2 : (ℝ × ℝ)), 
    R1 ∈ six_circles ∧ R2 ∈ six_circles ∧ 
    R1 ≠ R2 ∧ 
    ¬(∃ (P : ℝ), P ∈ [fst R1, snd R2] ∧ P ∈ [fst R2, snd R1]) ∧ 
    (d = dist (prod.fst R1, prod.snd R1) (prod.fst R2, prob.snd R2))) →
  (d^2 = r1^2 + r2^2 + 6*r1*r2 ∨ d^2 = r1^2 + r2^2 - 6*r1*r2) :=
begin
  sorry
end

end circle_distance_relation_l395_395549


namespace angle_triple_of_supplement_l395_395430

theorem angle_triple_of_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_of_supplement_l395_395430


namespace find_a_b_l395_395585

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395585


namespace valid_syllogism_l395_395906

axiom is_not_divisible_by_2 (n : ℕ) : Prop
axiom is_odd (n : ℕ) : Prop

-- Conditions based on given problem
axiom condition1 : is_not_divisible_by_2 2013
axiom condition2 : ∀ n, is_odd n → is_not_divisible_by_2 n
axiom condition3 : is_odd 2013

-- The correct sequence ②③① forms a syllogism in Lean
theorem valid_syllogism : 
  (condition2 2013 condition3) = condition1 := 
begin
  sorry
end

end valid_syllogism_l395_395906


namespace find_a_b_l395_395582

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395582


namespace find_a_b_l395_395617

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395617


namespace algebraic_expression_is_product_l395_395387

def algebraicExpressionMeaning (x : ℝ) : Prop :=
  -7 * x = -7 * x

theorem algebraic_expression_is_product (x : ℝ) :
  algebraicExpressionMeaning x :=
by
  sorry

end algebraic_expression_is_product_l395_395387


namespace probability_of_cocaptains_l395_395406

-- Define the conditions for the problem
def numberOfStudents (team : ℕ) : ℕ :=
  if team = 1 then 4 else
  if team = 2 then 6 else
  if team = 3 then 7 else
  if team = 4 then 9 else 0

def prob_cocaptains (team: ℕ) : ℚ :=
  let n := numberOfStudents team
  6 / (n * (n - 1) * (n - 2))

-- Main statement to be proved
theorem probability_of_cocaptains : 
  (1/4 : ℚ) * (prob_cocaptains 1 + prob_cocaptains 2 + prob_cocaptains 3 + prob_cocaptains 4) = 13 / 120 :=
by 
  sorry

end probability_of_cocaptains_l395_395406


namespace rowing_probability_l395_395471

open ProbabilityTheory

theorem rowing_probability
  (P_left_works : ℚ := 3 / 5)
  (P_right_works : ℚ := 3 / 5) :
  let P_left_breaks := 1 - P_left_works
  let P_right_breaks := 1 - P_right_works
  let P_left_works_and_right_works := P_left_works * P_right_works
  let P_left_works_and_right_breaks := P_left_works * P_right_breaks
  let P_left_breaks_and_right_works := P_left_breaks * P_right_works
  P_left_works_and_right_works + P_left_works_and_right_breaks + P_left_breaks_and_right_works = 21 / 25 := by
  sorry

end rowing_probability_l395_395471


namespace smallest_diff_of_primes_l395_395792

open Set

def pairwise_rel_prime (S : Finset ℕ) : Prop :=
  ∀ (a b ∈ S), a ≠ b → Nat.gcd a b = 1

theorem smallest_diff_of_primes :
  ∃ (S : Finset ℕ), S.card = 13 ∧ pairwise_rel_prime S ∧ (S.max' S.nonempty - S.min' S.nonempty = 32) :=
begin
  sorry
end

end smallest_diff_of_primes_l395_395792


namespace secant_line_slope_positive_l395_395851

theorem secant_line_slope_positive (f : ℝ → ℝ) (h_deriv : ∀ x : ℝ, 0 < (deriv f x)) :
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → 0 < (f x1 - f x2) / (x1 - x2) :=
by
  intros x1 x2 h_ne
  sorry

end secant_line_slope_positive_l395_395851


namespace stickers_after_exchange_l395_395347

-- Given conditions
def Ryan_stickers : ℕ := 30
def Steven_stickers : ℕ := 3 * Ryan_stickers
def Terry_stickers : ℕ := Steven_stickers + 20
def Emily_stickers : ℕ := Steven_stickers / 2
def Jasmine_stickers : ℕ := Terry_stickers + Terry_stickers / 10

def total_stickers_before : ℕ := 
  Ryan_stickers + Steven_stickers + Terry_stickers + Emily_stickers + Jasmine_stickers

noncomputable def total_stickers_after : ℕ := 
  total_stickers_before - 2 * 5

-- The goal is to prove that the total stickers after the exchange event is 386
theorem stickers_after_exchange : total_stickers_after = 386 := 
  by sorry

end stickers_after_exchange_l395_395347


namespace minimum_matrix_sum_l395_395453

noncomputable section

open BigOperators

def matrix_sum (M : Matrix (Fin 8) (Fin 8) ℝ) : ℝ :=
  ∑ i j, M i j

def is_valid_matrix (M : Matrix (Fin 8) (Fin 8) ℝ) : Prop :=
  ∀ i : Fin 8, (∀ j : Fin 8, M i j ≥ (i : ℝ) + 1) ∧ (∀ j : Fin 8, M j i ≥ (i : ℝ) + 1)

theorem minimum_matrix_sum :
  ∃ M : Matrix (Fin 8) (Fin 8) ℝ, is_valid_matrix M ∧ matrix_sum M = 372 := sorry

end minimum_matrix_sum_l395_395453


namespace trajectory_of_C_angle_PFQ_l395_395687

def point := ℝ × ℝ

def A : point := (-2, 0)
def B : point := (2, 0)
variable {C : point}
axiom slope_product_condition (C : point) : (C.2 / (C.1 + 2)) * (C.2 / (C.1 - 2)) = -3 / 4
def trajectory_eq (C : point) : Prop := C.2 ≠ 0 → (C.1^2 / 4) + (C.2^2 / 3) = 1

theorem trajectory_of_C : trajectory_eq C :=
sorry

def l : ℝ → ℝ := sorry
def P : point := sorry
def Q : point := (4, l 4)
def F : point := (1, 0)

theorem angle_PFQ (h₁ : C.2 ≠ 0) (h₂ : l P.1 = P.2) (h₃ : trajectory_eq P) : ∠P F Q = 90 :=
sorry

end trajectory_of_C_angle_PFQ_l395_395687


namespace compute_expression_l395_395537

theorem compute_expression :
  2 * 2^5 - 8^58 / 8^56 = 0 := by
  sorry

end compute_expression_l395_395537


namespace find_m_l395_395200

open Real

theorem find_m (x m : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -1) 
  (h2 : log 10 (sin x + cos x) = 1 / 2 * (log 10 m - 1)) : 
  m = 6 := by
  sorry

end find_m_l395_395200


namespace true_statements_identification_l395_395442

theorem true_statements_identification :
  (5 ∣ 30) ∧ (19 ∣ 209 ∧ ¬(19 ∣ 57)) ∧ (¬(30 ∣ 90) ∧ ¬(30 ∣ 65)) ∧ (17 ∣ 34 ∧ ¬(17 ∣ 68)) ∧ (9 ∣ 180) ↔
  {5 ∣ 30, 9 ∣ 180} :=
by
  sorry

end true_statements_identification_l395_395442


namespace expression_is_product_l395_395390

def not_sum (a x : Int) : Prop :=
  ¬(a + x = -7 * x)

def not_difference (a x : Int) : Prop :=
  ¬(a - x = -7 * x)

def not_quotient (a x : Int) : Prop :=
  ¬(a / x = -7 * x)

theorem expression_is_product (x : Int) : 
  not_sum (-7) x ∧ not_difference (-7) x ∧ not_quotient (-7) x → (-7 * x = -7 * x) :=
by sorry

end expression_is_product_l395_395390


namespace total_insects_eaten_l395_395466

theorem total_insects_eaten
  (geckos : ℕ)
  (insects_per_gecko : ℕ)
  (lizards : ℕ)
  (multiplier : ℕ)
  (h_geckos : geckos = 5)
  (h_insects_per_gecko : insects_per_gecko = 6)
  (h_lizards : lizards = 3)
  (h_multiplier : multiplier = 2) :
  geckos * insects_per_gecko + lizards * (insects_per_gecko * multiplier) = 66 :=
by
  rw [h_geckos, h_insects_per_gecko, h_lizards, h_multiplier]
  norm_num
  sorry

end total_insects_eaten_l395_395466


namespace colored_line_midpoint_l395_395988

theorem colored_line_midpoint (L : ℝ → Prop) (p1 p2 : ℝ) :
  (L p1 → L p2) →
  (∃ A B C : ℝ, L A = L B ∧ L B = L C ∧ 2 * B = A + C ∧ L A = L C) :=
sorry

end colored_line_midpoint_l395_395988


namespace rowing_speed_in_still_water_l395_395990

theorem rowing_speed_in_still_water (current_speed_kmph : ℝ) (time_seconds : ℝ) (distance_meters : ℝ) : 
  current_speed_kmph = 2 ∧ time_seconds = 17.998560115190788 ∧ distance_meters = 60 → 
  let current_speed_mps := (2 * (1000 / 1) * (1 / 3600) : ℝ) in
  let downstream_speed := (distance_meters / time_seconds : ℝ) in
  let rowing_speed_still_water := downstream_speed - current_speed_mps in
  rowing_speed_still_water ≈ 2.778 :=
begin
  sorry
end

end rowing_speed_in_still_water_l395_395990


namespace max_prob_two_consecutive_wins_l395_395965

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l395_395965


namespace inequality_trig_l395_395075

def a := Real.sin (33 * Real.pi / 180)
def b := Real.cos (55 * Real.pi / 180)
def c := Real.tan (55 * Real.pi / 180)

theorem inequality_trig : c > b ∧ b > a := by
  sorry

end inequality_trig_l395_395075


namespace vector_on_line_l395_395489

noncomputable def k_value (a b : Vector ℝ 3) (m : ℝ) : ℝ :=
  if h : m = 5 / 7 then
    (5 / 7 : ℝ)
  else
    0 -- This branch will never be taken because we will assume m = 5 / 7 as a hypothesis.


theorem vector_on_line (a b : Vector ℝ 3) (m k : ℝ) (h : m = 5 / 7) :
  k = k_value a b m :=
by
  sorry

end vector_on_line_l395_395489


namespace total_profit_l395_395889

-- Definitions based on the conditions
def tom_investment : ℝ := 30000
def tom_duration : ℝ := 12
def jose_investment : ℝ := 45000
def jose_duration : ℝ := 10
def jose_share_profit : ℝ := 25000

-- Theorem statement
theorem total_profit (tom_investment tom_duration jose_investment jose_duration jose_share_profit : ℝ) :
  (jose_share_profit / (jose_investment * jose_duration / (tom_investment * tom_duration + jose_investment * jose_duration)) = 5 / 9) →
  ∃ P : ℝ, P = 45000 :=
by
  sorry

end total_profit_l395_395889


namespace range_of_p_l395_395003

def p (x : ℝ) : ℝ := (x^3 + 3)^2

theorem range_of_p :
  (∀ y, ∃ x ∈ Set.Ici (-1 : ℝ), p x = y) ↔ y ∈ Set.Ici (4 : ℝ) :=
by
  sorry

end range_of_p_l395_395003


namespace average_speed_distance_div_time_l395_395928

theorem average_speed_distance_div_time (distance : ℕ) (time_minutes : ℕ) (average_speed : ℕ) : 
  distance = 8640 → time_minutes = 36 → average_speed = distance / (time_minutes * 60) → average_speed = 4 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  assumption

end average_speed_distance_div_time_l395_395928


namespace maximize_probability_when_C_second_game_l395_395955

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l395_395955


namespace price_difference_is_correct_l395_395492

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end price_difference_is_correct_l395_395492


namespace find_a_b_l395_395672

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395672


namespace problem1_problem2_l395_395016

-- Definitions
def point (a b : ℝ) := (a, b)
def line_through (P Q : ℝ × ℝ) (L : ℝ → ℝ → Prop) := 
  ∃ m b, L = λ x y, y = m * x + b ∧ (y = m * 4 + b) ∧ (y = m * (-1) + b)

def y_intercept_is_twice_x_intercept (L : ℝ → ℝ → Prop) :=
  ∃ b m, L = λ x y, y = m * x + b ∧ b = 2 * (- b / m)

-- Example Points
def P := point 4 1
def Q := point (-1) 6

-- Lean statements for each subproblem
theorem problem1 : 
  (line_through P Q (λ x y, x + y - 5 = 0)) := 
sorry

theorem problem2 : 
  (∃ L, y_intercept_is_twice_x_intercept L ∧ ((L = λ x y, y = (1 / 4) * x) ∨ (L = λ x y, y = -2 * x + 9))) := 
sorry

end problem1_problem2_l395_395016


namespace probability_of_one_radio_operator_per_group_l395_395098

-- Define the conditions
def num_soldiers : ℕ := 12
def num_radio_operators : ℕ := 3
def group_sizes : List ℕ := [3, 4, 5]

-- Define the number of ways to divide the soldiers into these groups
def total_ways_to_divide : ℕ :=
  Nat.choose 12 3 * Nat.choose 9 4 * Nat.choose 5 5

-- Define the number of ways to assign radio operators such that each group has exactly one
def favorable_ways : ℕ :=
  Nat.choose 3 1 * Nat.choose 9 2 *
  Nat.choose 2 1 * Nat.choose 7 3 *
  Nat.choose 1 1 * Nat.choose 4 4 *
  3.factorial

-- Calculate the desired probability
def desired_probability : ℚ :=
  favorable_ways / total_ways_to_divide

-- The main theorem proving the desired probability
theorem probability_of_one_radio_operator_per_group :
  desired_probability = 3 / 11 :=
by
  unfold desired_probability favorable_ways total_ways_to_divide
  sorry

end probability_of_one_radio_operator_per_group_l395_395098


namespace face_below_purple_is_violet_l395_395362

theorem face_below_purple_is_violet:
  (colors : Fin 6 → Fin 6) (hinged : ∀ i, (colors i) = i → (Fin.mk (i + 1) sorry)) 
  (config : colors 1 = 5) :
  colors (0 : Fin 6) = 3 := 
  sorry

end face_below_purple_is_violet_l395_395362


namespace solve_for_a_b_l395_395629

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395629


namespace find_a_b_l395_395622

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395622


namespace find_a_and_b_l395_395596

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395596


namespace number_of_positive_integers_for_prime_polynomial_l395_395566

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem number_of_positive_integers_for_prime_polynomial :
  {n : ℕ | isPrime (n^3 - 6 * n^2 + 15 * n - 11)}.count = 3 := sorry

end number_of_positive_integers_for_prime_polynomial_l395_395566


namespace chess_player_max_consecutive_win_prob_l395_395936

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l395_395936


namespace correct_propositions_l395_395873

-- Define the functions and variables for each proposition
def prop1_func (x: ℝ) (hx : x ≠ 0) : ℝ := x + 1 / (4 * x)
def prop2_F : ℝ × ℝ := (-2, 3)
def prop2_line (x y : ℝ) : Prop := 2 * x + y + 1 = 0

noncomputable def parabola (P: ℝ × ℝ) : Prop :=
  let (x, y) := P in (x + 2) ^ 2 + (y - 3) ^ 2 = 25 -- Just an example description

noncomputable def prop3 (α : Type) (AB CB CE CF : ℝ → ℝ × ℝ) (B C : ℝ × ℝ) : Prop :=
  ∀ x, x ∈ α → 
  ∃ (θ θ₁ θ₂ θ₃: ℝ), 
    angle (AB x) (CB x) = θ ∧ angle (AB x) (CE x) = θ₁ ∧ angle (AB x) (CF x) = θ₂ ∧ θ₁ ≠ θ₂ ∧ θ₂ ≠ θ₃

noncomputable def prop4 (f : ℝ → ℝ) (b c x₁ x₂ : ℝ) (hx : b ∈ ℝ ∧ c ∈ ℝ) : Prop := 
  f (x₁ + x₂) / 2 ≤ (f x₁ + f x₂)

-- Main theorem statements
theorem correct_propositions 
: ¬ (∀ x, x ≠ 0 → prop1_func x ‹x ≠ 0› ∈ set.Ici (1 : ℝ)) 
∧ ¬ (∃ P : ℝ × ℝ, ∀F ∈ prop2_F, P = (0, 0))
∧ ( ∃ α : Type, ∀ (x y: ℝ), prop3 α (λ x, (x, x)) (λ x, (x, x)) (λ x, (x, x)) (λ x, (x, x)) (0, 0) (0, 0)) 
∧ ( ∀x₁ x₂ : ℝ, ∀ b c : ℝ, prop4 (λx, x ^ 2 + b * x + c) b c x₁ x₂ (or.inl b.in_circle_or_ring.elim or.inr c.in_circle_or_ring.elim)) :=
by 
  sorry

end correct_propositions_l395_395873


namespace seventh_day_price_percentage_l395_395516

theorem seventh_day_price_percentage 
  (original_price : ℝ)
  (day1_reduction : original_price * 0.91)
  (day2_increase : day1_reduction * 1.05)
  (day3_reduction : day2_increase * 0.90)
  (day4_increase : day3_reduction * 1.15)
  (day5_reduction : day4_increase * 0.90)
  (day6_increase : day5_reduction * 1.08)
  (day7_reduction : day6_increase * 0.88) :
  day7_reduction / original_price * 100 = 84.59 :=
begin
  sorry
end

end seventh_day_price_percentage_l395_395516


namespace max_prob_win_two_consecutive_is_C_l395_395962

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l395_395962


namespace square_distance_from_B_is_53_l395_395500

variable (a b : ℝ)
variable (r AB BC : ℝ)
variable (C : ℝ × ℝ)
variable (A : ℝ × ℝ)
variable (B : ℝ × ℝ)
variable (center : ℝ × ℝ := (0, 0))
variable (dist_sq : ℝ)

-- Conditions
def radius (r : ℝ) := r = Real.sqrt 72
def length_AB (AB : ℝ) := AB = 8
def length_BC (BC : ℝ) := BC = 3
def angle_ABC_is_right := (Math.atan2 (B.2 - A.2) (B.1 - A.1) - Math.atan2 (C.2 - B.2) (C.1 - B.1)) = Real.pi / 2

-- Positions of points A, B, and C
def coordinates_A (A B : ℝ × ℝ) := A = (B.1, B.2 + 8)
def coordinates_C (C B : ℝ × ℝ) := C = (B.1 + 3, B.2)

-- Circle equation conditions
def circle_eq_A (A : ℝ × ℝ) := A.1 ^ 2 + (A.2) ^ 2 = 72
def circle_eq_C (C : ℝ × ℝ) := (C.1) ^ 2 + C.2 ^ 2 = 72

-- Square of the distance from B to center
def square_distance_from_B_to_center (B : ℝ × ℝ) (center : ℝ × ℝ) :=
  dist_sq = (B.1 - center.1) ^ 2 + (B.2 - center.2) ^ 2

-- The theorem we need to prove
theorem square_distance_from_B_is_53 (a b : ℝ) :
  radius r ∧
  length_AB AB ∧
  length_BC BC ∧
  angle_ABC_is_right ∧
  coordinates_A A B ∧
  coordinates_C C B ∧
  circle_eq_A A ∧
  circle_eq_C C →
  square_distance_from_B_to_center B center :=
by
  let B := (7, -2)
  let dist_sq := 7^2 + (-2)^2 = 53
  sorry

end square_distance_from_B_is_53_l395_395500


namespace calc_x_cubed_minus_3x_l395_395120

noncomputable def x : ℝ := real.cbrt (7 + 4 * real.sqrt 3) + 1 / real.cbrt (7 + 4 * real.sqrt 3)

theorem calc_x_cubed_minus_3x : x^3 - 3 * x = 14 := by
  sorry

end calc_x_cubed_minus_3x_l395_395120


namespace maximize_probability_l395_395940

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l395_395940


namespace problem_l395_395256

-- Conditions
variables (x y : ℚ)
def condition1 := 3 * x + 5 = 12
def condition2 := 10 * y - 2 = 5

-- Theorem to prove
theorem problem (h1 : condition1 x) (h2 : condition2 y) : x + y = 91 / 30 := sorry

end problem_l395_395256


namespace transformation_type_l395_395076

-- Definitions of the distribution functions

def F_G (x : ℝ) : ℝ := real.exp (-real.exp (-x))

def F_F (x : ℝ) (α : ℝ) : ℝ := 
  if x < 0 then 0 else real.exp (-x^(-α))

def F_W (x : ℝ) (α : ℝ) : ℝ :=
  if x < 0 then real.exp (-real.abs (x)^α) else 1

-- Generalized extreme value distribution
def generalized_extreme_value (x a b γ : ℝ) : ℝ :=
  if γ * (a * x + b) > -1 then real.exp (-(1 + γ * (a * x + b))^(-1 / γ)) else if γ > 0 then 1 else 0

-- Proof statement (lean term)
theorem transformation_type :
  ∀ (X Y : ℝ) (α a b γ : ℝ), a > 0 → γ ≠ 0 → 
  (F_F X α = real.exp (- (real.exp (-(a * ln X + b) )))) ∧ 
  (F_W Y α = real.exp (- (real.exp (-(a * (-ln (-Y)) + b) )))) :=
by 
  sorry

end transformation_type_l395_395076


namespace maximize_prob_of_consecutive_wins_l395_395947

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l395_395947


namespace max_prob_two_consecutive_wins_l395_395970

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l395_395970


namespace probability_of_drawing_two_black_two_white_l395_395470

noncomputable def probability_two_black_two_white : ℚ :=
  let total_ways := (Nat.choose 18 4)
  let ways_black := (Nat.choose 10 2)
  let ways_white := (Nat.choose 8 2)
  let favorable_ways := ways_black * ways_white
  favorable_ways / total_ways

theorem probability_of_drawing_two_black_two_white :
  probability_two_black_two_white = 7 / 17 := sorry

end probability_of_drawing_two_black_two_white_l395_395470


namespace ring_display_capacity_l395_395985

def necklace_capacity : ℕ := 12
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_cost : ℕ := 4
def ring_cost : ℕ := 10
def bracelet_cost : ℕ := 5
def total_cost : ℕ := 183

theorem ring_display_capacity : ring_capacity + (total_cost - ((necklace_capacity - current_necklaces) * necklace_cost + (bracelet_capacity - current_bracelets) * bracelet_cost)) / ring_cost = 30 := by
  sorry

end ring_display_capacity_l395_395985


namespace problem_proof_l395_395655

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395655


namespace middle_aged_participating_l395_395480

-- Definitions of the given conditions
def total_employees : Nat := 1200
def ratio (elderly middle_aged young : Nat) := elderly = 1 ∧ middle_aged = 5 ∧ young = 6
def selected_employees : Nat := 36

-- The stratified sampling condition implies
def stratified_sampling (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) :=
  (elderly + middle_aged + young = total) ∧
  (selected = 36)

-- The proof statement
theorem middle_aged_participating (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) 
  (h_ratio : ratio elderly middle_aged young) 
  (h_total : total = total_employees)
  (h_sampled : stratified_sampling elderly middle_aged young (elderly + middle_aged + young) selected) : 
  selected * middle_aged / (elderly + middle_aged + young) = 15 := 
by sorry

end middle_aged_participating_l395_395480


namespace correct_proposition_l395_395905

theorem correct_proposition :
  (∃ (tri : Type) (T1 T2 : tri),
    symmetric_about_line T1 T2 → congruent T1 T2) ∧
  (∀ (tri : Type) (T : tri) (m : median T),
    ¬divides_area_into_two_equal_parts T m) ∧
  (∀ (quad : Type) (Q : quad) (d1 d2 : diagonal Q),
    (equal_and_bisect_each_other d1 d2) → 
    rectangle Q ∧ not_square Q) ∧
  (∀ (quad : Type) (Q : quad) (p1 p2 : pair_of_sides Q),
    (one_pair_parallel p1 p2 ∧ another_pair_equal p1 p2) → 
    not_isosceles_trapezoid Q) :=
by
  sorry

end correct_proposition_l395_395905


namespace find_a_and_b_l395_395599

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395599


namespace find_a_b_l395_395587

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395587


namespace simplify_root_exponentiation_l395_395833

theorem simplify_root_exponentiation : (7 ^ (1 / 3) : ℝ) ^ 6 = 49 := by
  sorry

end simplify_root_exponentiation_l395_395833


namespace length_of_BC_l395_395894

-- Define the given conditions and the theorem using Lean
theorem length_of_BC 
  (A B C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hB : ∃ b : ℝ, B = (-b, -b^2)) 
  (hC : ∃ b : ℝ, C = (b, -b^2)) 
  (hBC_parallel_x_axis : ∀ b : ℝ, C.2 = B.2)
  (hArea : ∀ b : ℝ, b^3 = 72) 
  : ∀ b : ℝ, (BC : ℝ) = 2 * b := 
by
  sorry

end length_of_BC_l395_395894


namespace total_insects_eaten_l395_395463

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end total_insects_eaten_l395_395463


namespace pm_perpendicular_y_axis_area_triangle_range_l395_395210

noncomputable def parabola (p : ℝ) : ℝ → Prop := λ y : ℝ, ∃ x : ℝ, y^2 = 2 * p * x
noncomputable def semiEllipse (p : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), x^2 + y^2 / (2 * p) = 1 ∧ x < 0

theorem pm_perpendicular_y_axis (p : ℝ) (hp : 0 < p) 
  (P : ℝ × ℝ) (hP_left : P.1 < 0) (hP_not_y_axis : P.1 ≠ 0) (hP_on_parabola : parabola p P.2)
  (A B M : ℝ × ℝ) (hA_on_parabola : parabola p A.2) (hB_on_parabola : parabola p B.2)
  (M_middle : M = ((A.1 + B.1)/2, (A.2 + B.2)/2)) (M_on_parabola : parabola p M.2) : 
  P ⟂ (0, M.2 - P.2) := 
sorry

theorem area_triangle_range (p : ℝ) (hp : 0 < p) 
  (P : ℝ × ℝ) (hP_on_semiEllipse : semiEllipse p P) 
  (A B : ℝ × ℝ) (hA_on_parabola : parabola p A.2) (hB_on_parabola : parabola p B.2)
  : ∃ S : ℝ, 6 * sqrt p ≤ S ∧ S ≤ 15 * sqrt (5 * p) / 4 :=
sorry

end pm_perpendicular_y_axis_area_triangle_range_l395_395210


namespace find_a_b_l395_395607

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395607


namespace geometric_sequence_ratio_l395_395216

theorem geometric_sequence_ratio (a1 : ℝ) (S : ℝ) (h1 : a1 = 2) (h2 : S = 3)
  (h3 : ∀ (|q| < 1), Σ' n, a1 * q^n = S) : q = 1 / 3 :=
by
  sorry

end geometric_sequence_ratio_l395_395216


namespace remainder_when_12_plus_a_div_by_31_l395_395799

open Int

theorem remainder_when_12_plus_a_div_by_31 (a : ℤ) (ha : 0 < a) (h : 17 * a % 31 = 1) : (12 + a) % 31 = 23 := by
  sorry

end remainder_when_12_plus_a_div_by_31_l395_395799


namespace circle_radius_of_equal_area_l395_395265

theorem circle_radius_of_equal_area (A B C D : Type) (r : ℝ) (π : ℝ) 
  (h_rect_area : 8 * 9 = 72)
  (h_circle_area : π * r ^ 2 = 36) :
  r = 6 / Real.sqrt π :=
by
  sorry

end circle_radius_of_equal_area_l395_395265


namespace choose_three_of_nine_l395_395261

def combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_three_of_nine : combination 9 3 = 84 :=
by 
  sorry

end choose_three_of_nine_l395_395261


namespace tim_weekly_payment_l395_395881

-- Define the given conditions
def hourly_rate_bodyguard : ℕ := 20
def number_bodyguards : ℕ := 2
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7

-- Define the total weekly payment calculation
def weekly_payment : ℕ := (hourly_rate_bodyguard * number_bodyguards) * hours_per_day * days_per_week

-- The proof statement
theorem tim_weekly_payment : weekly_payment = 2240 := by
  sorry

end tim_weekly_payment_l395_395881


namespace cos_rational_deg_irrational_l395_395426

theorem cos_rational_deg_irrational (p q : ℕ) (hpq : (p : ℚ) / q ∉ (set.range (coe : ℤ → ℚ)) ∪ (set.range (λ (n : ℕ), (1 / n) * ℚ)) ∪ (set.range (λ (n : ℕ), -1 / n * ℚ))) :
  ¬ ∃ r : ℚ, cos ((p : ℝ) / q * real.pi / 180) = ↑r := sorry

end cos_rational_deg_irrational_l395_395426


namespace distance_sum_inscribed_triang_parabola_l395_395240

noncomputable def parabola := enhypot ((y: ℝ) * y = 4 * real.sqrt 2 * (x: ℝ))
def centroid_is_focus (A B C F : ℝ × ℝ) : Prop := 
  let (xA, yA) := A in
  let (xB, yB) := B in
  let (xC, yC) := C in
  let (xF, yF) := F in
  xA + xB + xC = 3 * xF ∧ yA + yB + yC = 3 * yF

theorem distance_sum_inscribed_triang_parabola (A B C F : ℝ × ℝ) 
  (h1 : parabola)
  (h2 : F = (real.sqrt 2, 0))
  (h3 : centroid_is_focus A B C F) :
  ((A.1 - F.1) ^ 2 + A.2 ^ 2) + ((B.1 - F.1) ^ 2 + B.2 ^ 2) + ((C.1 - F.1) ^ 2 + C.2 ^ 2) = 27 :=
by
  sorry

end distance_sum_inscribed_triang_parabola_l395_395240


namespace eulerian_path_odd_vertices_l395_395819

theorem eulerian_path_odd_vertices (G : Type) [graph G] :
  (∃ (path : list (vertex G)), is_eulerian_path G path) → 
  ∑ v in vertex_set G, (if is_odd_degree G v then 1 else 0) ≤ 2 := 
sorry

end eulerian_path_odd_vertices_l395_395819


namespace decreasing_by_25_l395_395910

theorem decreasing_by_25 (n : ℕ) (k : ℕ) (y : ℕ) (hy : 0 ≤ y ∧ y < 10^k) : 
  (n = 6 * 10^k + y → n / 10 = y / 25) → (∃ m, n = 625 * 10^m) := 
sorry

end decreasing_by_25_l395_395910


namespace correct_operation_l395_395440

theorem correct_operation (x y m c d : ℝ) : (5 * x * y - 4 * x * y = x * y) :=
by sorry

end correct_operation_l395_395440


namespace fish_population_after_migration_l395_395251

theorem fish_population_after_migration 
    (initial_tagged_fish : ℕ) 
    (second_catch : ℕ) 
    (second_catch_tagged : ℕ) 
    (new_fish_migrated : ℕ) :
    initial_tagged_fish = 500 →
    second_catch = 300 →
    second_catch_tagged = 6 →
    new_fish_migrated = 250 →
    let N := (initial_tagged_fish * second_catch) / second_catch_tagged in
    N + new_fish_migrated = 25000 := 
by 
    sorry

end fish_population_after_migration_l395_395251


namespace parkway_school_students_l395_395773

theorem parkway_school_students (total_boys total_soccer soccer_boys_percentage girls_not_playing_soccer : ℕ)
  (h1 : total_boys = 320)
  (h2 : total_soccer = 250)
  (h3 : soccer_boys_percentage = 86)
  (h4 : girls_not_playing_soccer = 95)
  (h5 : total_soccer * soccer_boys_percentage / 100 = 215) :
  total_boys + total_soccer - (total_soccer * soccer_boys_percentage / 100) + girls_not_playing_soccer = 450 :=
by
  sorry

end parkway_school_students_l395_395773


namespace cyrus_total_shots_attempted_l395_395748

theorem cyrus_total_shots_attempted (T : ℕ) (missed_shots : ℕ) (made_percentage : ℝ):
  missed_shots = 4 → made_percentage = 0.80 → 
  0.20 * T = missed_shots → T = 20 :=
by
  intros h_missed h_percentage h_equation
  -- sorry, proof skipped
  have h_T : T = 20, from sorry,
  exact h_T

end cyrus_total_shots_attempted_l395_395748


namespace sin_to_cos_shift_l395_395888

theorem sin_to_cos_shift (x : ℝ) : 
  let f := λ x, cos (2 * x)
      g := λ x, sin (2 * x)
      T := π in
  f x = g (x + T / 4) :=
by sorry

end sin_to_cos_shift_l395_395888


namespace imo1996_q3_l395_395311

theorem imo1996_q3
  (n p q : ℕ) (hn : n > p + q)
  (x : Fin (n+1) → ℤ)
  (h0 : x 0 = 0) (hn' : x ⟨n, by linarith⟩ = 0)
  (h_step : ∀ i : Fin n, x ⟨i.succ, by linarith⟩ - x i = p ∨ 
                          x ⟨i.succ, by linarith⟩ - x i = -q) :
  ∃ i j : Fin (n+1), i < j ∧ i ≠ 0 ∧ j ≠ ⟨n, by linarith⟩ ∧ x i = x j := 
sorry

end imo1996_q3_l395_395311


namespace chess_player_max_consecutive_win_prob_l395_395933

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l395_395933


namespace angle_B_in_isosceles_triangle_l395_395276

theorem angle_B_in_isosceles_triangle 
  (a b : ℝ) (A B C : ℝ) (h1 : a = 2) (h2 : b = 2) (h3 : A = 45) 
  (triangle_ABC : a^2 + b^2 - 2 * a * b * real.cos C = c^2) 
  (angle_sum : A + B + C = 180) :
  B = 67.5 :=
sorry

end angle_B_in_isosceles_triangle_l395_395276


namespace f_sqrt45_l395_395809

def f (x : Real) : Real :=
  if Int.floor x = x then
    7 * x + 3
  else
    Real.floor x + 7

theorem f_sqrt45 : f (Real.sqrt 45) = 13 := by
  sorry

end f_sqrt45_l395_395809


namespace promotion_price_percentage_l395_395515

noncomputable def price_on_seventh_day (p : ℝ) : ℝ :=
  let day1 := p * (1 - 0.09) in
  let day2 := day1 * (1 + 0.05) in
  let day3 := day2 * (1 - 0.10) in
  let day4 := day3 * (1 + 0.15) in
  let day5 := day4 * (1 - 0.10) in
  let day6 := day5 * (1 + 0.08) in
  let day7 := day6 * (1 - 0.12) in
  day7

theorem promotion_price_percentage (p : ℝ) (h0 : 0 < p) :
  price_on_seventh_day p = 0.8459 * p := by
  sorry

end promotion_price_percentage_l395_395515


namespace probability_of_friends_in_same_lunch_group_l395_395372

theorem probability_of_friends_in_same_lunch_group :
  let groups := 4
  let students := 720
  let group_size := students / groups
  let probability := (1 / groups) * (1 / groups) * (1 / groups)
  students % groups = 0 ->  -- Students can be evenly divided into groups
  groups > 0 ->             -- There is at least one group
  probability = (1 : ℝ) / 64 :=
by
  intros
  sorry

end probability_of_friends_in_same_lunch_group_l395_395372


namespace prove_geomSeqSumFirst3_l395_395300

noncomputable def geomSeqSumFirst3 {a₁ a₆ : ℕ} (h₁ : a₁ = 1) (h₂ : a₆ = 32) : ℕ :=
  let r := 2 -- since r^5 = 32 which means r = 2
  let S3 := a₁ * (1 - r^3) / (1 - r)
  S3

theorem prove_geomSeqSumFirst3 : 
  geomSeqSumFirst3 (h₁ : 1 = 1) (h₂ : 32 = 32) = 7 := by
  sorry

end prove_geomSeqSumFirst3_l395_395300


namespace sqrt_neg9_sq_l395_395457

theorem sqrt_neg9_sq : Real.sqrt ((-9 : Real)^2) = 9 := 
by 
  sorry

end sqrt_neg9_sq_l395_395457


namespace tangent_perpendicular_g_maximum_l395_395088

noncomputable def f (x : ℝ) (c : ℝ) := (1/3)*x^3 + c*x + 3
noncomputable def f_prime (x : ℝ) (c : ℝ) := x^2 + c

theorem tangent_perpendicular (c : ℝ) : 
  (f_prime 0 c = -1) → 
  c = -1 :=
by
  intro h
  exact h

noncomputable def f_fixed (x : ℝ) := (1/3)*x^3 - x + 3
noncomputable def g (x : ℝ) := 4 * Real.log x - (x^2 - 1)
noncomputable def g_prime (x : ℝ) := (4 / x) - 2 * x

theorem g_maximum : 
  ∀ x > 0, 
  g' x < 0 → 
  x = Real.sqrt 2 →
  g (Real.sqrt 2) = 2 * Real.log 2 - 1 :=
by
  intros x hx hprime h
  sorry

end tangent_perpendicular_g_maximum_l395_395088


namespace increasing_power_function_l395_395861

theorem increasing_power_function (m : ℝ) :
  (∀ x > 0, 0 < (m^2 - 3m + 3) * x ^ (m^2 - 2m + 1)) → m = 2 := 
by
  sorry

end increasing_power_function_l395_395861


namespace domain_and_odd_iff_inequality_and_range_of_m_comparison_n_geq_4_l395_395711

variables {a : ℝ} (ha : 0 < a ∧ a ≠ 1)

noncomputable def f (x : ℝ) : ℝ := log a ((x + 1) / (x - 1))

-- Domain and odd function proof
theorem domain_and_odd_iff : 
  ∀ (x : ℝ), 
    (x ∈ (-∞, -1) ∪ (1, +∞)) ∧ (f x = -f (-x)) :=
sorry

-- Inequality and range of m
theorem inequality_and_range_of_m (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4) : 
  ∀ m : ℝ,
    ((1 < a ∧ 0 < m ∧ m < 15) ∨ (0 < a ∧ a < 1 ∧ 45 < m)) →
      (f x > log a (m / ((x - 1)^2 * (7 - x)))) :=
sorry

-- Comparison for n ≥ 4
theorem comparison_n_geq_4 (n : ℕ) (hn : n ≥ 4) : 
  a^(sum (range n.filter (λ i, i ≥ 2)) (λ i, f i)) < 2^n - 2 :=
sorry

end domain_and_odd_iff_inequality_and_range_of_m_comparison_n_geq_4_l395_395711


namespace parallelepiped_volume_rectangular_l395_395502

theorem parallelepiped_volume_rectangular (
  (side1 side2 side3 : ℝ)
  (h₁ : side1 = 34)
  (h₂ : side2 = 30)
  (h₃ : side3 = 8 * Real.sqrt 13)
) : ∃ (a b c : ℝ),
  (a^2 + b^2 = (34 / 2)^2) ∧
  (a^2 + c^2 = (30 / 2)^2) ∧
  (b^2 + c^2 = (8 * Real.sqrt 13 / 2)^2) ∧
  (a * b * c = 1224) :=
by
  sorry

end parallelepiped_volume_rectangular_l395_395502


namespace find_a_b_l395_395643

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395643


namespace solve_for_a_b_l395_395638

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395638


namespace problem_proof_l395_395656

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395656


namespace loan_amounts_l395_395481

theorem loan_amounts (x y : ℝ) (h1 : x + y = 50) (h2 : 0.1 * x + 0.08 * y = 4.4) : x = 20 ∧ y = 30 := by
  sorry

end loan_amounts_l395_395481


namespace probability_one_die_shows_4_given_sum_7_l395_395900

def outcomes_with_sum_7 : List (ℕ × ℕ) := [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]

def outcome_has_4 (outcome : ℕ × ℕ) : Bool :=
  outcome.fst = 4 ∨ outcome.snd = 4

def favorable_outcomes : List (ℕ × ℕ) :=
  outcomes_with_sum_7.filter outcome_has_4

theorem probability_one_die_shows_4_given_sum_7 :
  (favorable_outcomes.length : ℚ) / (outcomes_with_sum_7.length : ℚ) = 1 / 3 := sorry

end probability_one_die_shows_4_given_sum_7_l395_395900


namespace collective_mowing_time_l395_395416

noncomputable def mowing_time
  (tim_time : ℝ) (linda_time : ℝ) (john_time : ℝ) : ℝ := 
  let tim_rate := 1 / tim_time
  let linda_rate := 1 / linda_time
  let john_rate := 1 / john_time
  let combined_rate := tim_rate + linda_rate + john_rate
  1 / combined_rate

theorem collective_mowing_time 
  (tim_time : ℝ) (linda_time : ℝ) (john_time : ℝ)
  (htim : tim_time = 1.5) 
  (hlinda : linda_time = 2) 
  (hjohn : john_time = 2.5) : 
  approximates (mowing_time tim_time linda_time john_time) (30/47)
  sorry

end collective_mowing_time_l395_395416


namespace least_multiple_greater_than_500_l395_395054

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 500 ∧ n % 32 = 0 := by
  let n := 512
  have h1 : n > 500 := by 
    -- proof omitted, as we're not solving the problem here
    sorry
  have h2 : n % 32 = 0 := by 
    -- proof omitted
    sorry
  exact ⟨n, h1, h2⟩

end least_multiple_greater_than_500_l395_395054


namespace sum_of_distances_is_13_l395_395083

variables {α : Type*} [LinearOrderedField α]

-- Assume d1 is the smaller distance and d2 is the bigger distance
def distances (d1 d2 : α) : Prop :=
d2 = d1 + 5

-- Let the sum of distances be the sum of these two distances
def sum_distances (d1 d2 : α) : α :=
d1 + d2

-- Define the problem statement
theorem sum_of_distances_is_13
  (d1 d2 : α) (h1 : distances d1 d2) (h2 : d1 + d2 = 13) :
  sum_distances d1 d2 = 13 :=
by
  rw [sum_distances, ←h1] at h2
  exact h2

end sum_of_distances_is_13_l395_395083


namespace prime_factor_different_from_p_sum_prime_factors_ge_half_p_sq_l395_395312

/- 
   Question 1: 
   Let \( p \) be a prime number greater than 3. 
   Prove that \( (p-1)^{p} + 1 \) has at least one prime factor different from \( p \).
-/
theorem prime_factor_different_from_p (p : ℕ) (hp : p > 3) (h_prime : Nat.Prime p) :
  ∃ q, q ≠ p ∧ Nat.Prime q ∧ q ∣ ((p-1).^p + 1) := 
sorry

/-
   Question 2: (Continuation of Question 1)
   Suppose \( (p-1)^{p} + 1 = \prod_{i=1}^{n} p_{i}^{\alpha_{i}} \),
   where \( p_{1}, p_{2}, \ldots, p_{n} \) are distinct prime numbers and
   \( \alpha_{1}, \alpha_{2}, \ldots, \alpha_{n} \) are positive integers. 
   Prove that \( \sum_{i=1}^{n} p_{i} \alpha_{i} \geq \frac{p^{2}}{2} \).
-/
theorem sum_prime_factors_ge_half_p_sq (p : ℕ) (hp : p > 3) (h_prime : Nat.Prime p) 
  (n : ℕ) (p_i : Fin n → ℕ) (α_i : Fin n → ℕ) 
  (h_distinct_primes : ∀ i j, i ≠ j → p_i i ≠ p_i j) 
  (h_prime_factors : ∀ i, Nat.Prime (p_i i))
  (h_positive_exponents : ∀ i, α_i i > 0)
  (h_prod_representation : (p-1).^p + 1 = ∏ i, (p_i i)^(α_i i)) :
  (∑ i, (p_i i) * (α_i i)) ≥ (p^2 / 2) :=
sorry

end prime_factor_different_from_p_sum_prime_factors_ge_half_p_sq_l395_395312


namespace part1_part2_l395_395764

variable (a b c t x1 x2 y1 y2 : ℝ)

namespace ProofProblem

-- Conditions
def is_on_parabola (x y : ℝ) : Prop :=
  y = a * x^2 + b * x + c

def axis_of_symmetry : Prop :=
  t = -b / (2 * a)

def a_positive : Prop :=
  a > 0

-- Part (1) Prove t = 3/2 given y1 = y2 for x1 = 1 and x2 = 2
theorem part1 (h1 : is_on_parabola 1 y1) (h2 : is_on_parabola 2 y2) (heq : y1 = y2) (apos : a_positive) : t = 3 / 2 :=
by sorry

-- Part (2) Prove t ≤ 1/2 given y1 < y2 for 0 < x1 < 1 and 1 < x2 < 2
theorem part2 (h1 : is_on_parabola x1 y1) (h2 : is_on_parabola x2 y2) (hlt : y1 < y2)
  (hx1 : 0 < x1) (hx1' : x1 < 1) (hx2 : 1 < x2) (hx2' : x2 < 2) (apos : a_positive) : t ≤ 1 / 2 :=
by sorry

end ProofProblem

end part1_part2_l395_395764


namespace length_of_AB_l395_395257

theorem length_of_AB (A B C : Type) [RightTriangle A B C]
    (angle_A : angle A B C = 90) (BC : Real := 10) (angle_B : angle B C A = 30) : 
    length A B = 5 :=
by
  sorry

end length_of_AB_l395_395257


namespace special_integers_count_zero_l395_395574

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then (n.factors.last sorry)
  else 1

def count_special_integers : ℕ :=
  finset.card (finset.filter (λ n, 2 ≤ n ∧ n < 100 ∧ greatest_prime_factor n = 5 ∧ greatest_prime_factor (n + 36) = nat.sqrt (n + 36)) (finset.range 100))

theorem special_integers_count_zero : count_special_integers = 0 :=
  sorry

end special_integers_count_zero_l395_395574


namespace regression_equation_is_correct_l395_395192

theorem regression_equation_is_correct 
  (linear_corr : ∃ (f : ℝ → ℝ), ∀ (x : ℝ), ∃ (y : ℝ), y = f x)
  (mean_b : ℝ)
  (mean_x : ℝ)
  (mean_y : ℝ)
  (mean_b_eq : mean_b = 0.51)
  (mean_x_eq : mean_x = 61.75)
  (mean_y_eq : mean_y = 38.14) : 
  mean_y = mean_b * mean_x + 6.65 :=
sorry

end regression_equation_is_correct_l395_395192


namespace red_peaches_l395_395409

theorem red_peaches (R G : ℕ) (h1 : G = 11) (h2 : G = R + 6) : R = 5 :=
by {
  sorry
}

end red_peaches_l395_395409


namespace distinct_assignment_plans_l395_395175

noncomputable def totalAssignmentPlans (students : Finset ℕ) (exclusions : Finset ℕ) :=
  let n := students.card
  let k := 4
  let r := 2 -- Number of students excluded from task A
  if h₁ : n = 6 ∧ k = 4 ∧ exclusions.card = r then
    let eligible_for_A := n - r
    eligible_for_A * ((n - 1).choose (k - 1) * factorial (k - 1))
  else
    0

theorem distinct_assignment_plans : 
  ∀ (students exclusions : Finset ℕ),
    students.card = 6 → 
    exclusions.card = 2 → 
    甲 ∈ exclusions → 
    乙 ∈ exclusions → 
    totalAssignmentPlans students exclusions = 240 :=
by
  intros
  dsimp [totalAssignmentPlans]
  rw [if_pos]
  norm_num
  sorry

end distinct_assignment_plans_l395_395175


namespace total_insects_eaten_l395_395461

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end total_insects_eaten_l395_395461


namespace incorrect_conclusion_l395_395749

noncomputable def number_of_red_balls (m : ℕ) := m
noncomputable def number_of_yellow_balls (m : ℕ) := 5 - m
def prob_A := 2 / 5
def prob_C := 1 - prob_A
def prob_B (m : ℕ) := (m / 5) * ((m - 1) / 4) + (number_of_yellow_balls m / 5) * (m / 4)
def prob_A_and_B (m : ℕ) := (m / 5) * ((m - 1) / 4)
def prob_A_or_B (m : ℕ) := prob_A + prob_B m - prob_A_and_B m

theorem incorrect_conclusion : ∀ m, prob_A_or_B m ≠ 4 / 5 :=
by sorry

end incorrect_conclusion_l395_395749


namespace reciprocals_arithmetic_sequence_l395_395306

noncomputable theory

variables (x y : ℝ)

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def distinct_points_on_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ A ≠ B

def intersects_x_axis_at (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, P = (m, 0) ∧ P = A ∨ P = B

def intersects_y_axis_at (A B : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  ∃ n : ℝ, Q = (0, n) ∧ Q = A ∨ Q = B

def point_on_line (M : ℝ × ℝ) (n : ℝ) : Prop :=
  M.2 = 3 / n

def slopes_exist_and_nonzero (A B M Q : ℝ × ℝ) : Prop :=
  (A.2 - M.2) / (A.1 - M.1) ≠ 0 ∧ (B.2 - M.2) / (B.1 - M.1) ≠ 0 ∧ (Q.2 - M.2) / (Q.1 - M.1) ≠ 0

theorem reciprocals_arithmetic_sequence
  (A B P Q M : ℝ × ℝ) (m n : ℝ)
  (h1 : distinct_points_on_ellipse A B)
  (h2 : intersects_x_axis_at A B P)
  (h3 : intersects_y_axis_at A B Q)
  (h4 : m ≠ 2 ∧ m ≠ -2 ∧ m ≠ 0)
  (h5 : n ≠ sqrt 3 ∧ n ≠ -sqrt 3 ∧ n ≠ 0)
  (h6 : point_on_line M n)
  (h7 : slopes_exist_and_nonzero A B M Q) :
  let k1 := (A.2 - M.2) / (A.1 - M.1), k2 := (B.2 - M.2) / (B.1 - M.1), k3 := (Q.2 - M.2) / (Q.1 - M.1) in
  (1 / k1 + 1 / k2 - 2 * (1 / k3)) = 0 :=
by
  sorry

end reciprocals_arithmetic_sequence_l395_395306


namespace maximize_prob_of_consecutive_wins_l395_395948

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l395_395948


namespace find_angle_ACD_l395_395262

-- Definitions and angles in the quadrilateral ABCD
variables {A B C D E : Type}
variables [angle_space A B C D]  -- Here angle_space is a hypothetical space where angles are defined

-- Given angles in the quadrilateral ABCD
def angle_ABD : ℝ := 70
def angle_CAD : ℝ := 20
def angle_BAC : ℝ := 48
def angle_CBD : ℝ := 40

-- The goal is to find angle ACD
theorem find_angle_ACD (h_ABD : angle_ABD = 70) (h_CAD : angle_CAD = 20) (h_BAC : angle_BAC = 48) (h_CBD : angle_CBD = 40) : 
  ∃ angle_ACD : ℝ, angle_ACD = 22 := 
by {
  sorry  -- proof goes here
}

end find_angle_ACD_l395_395262


namespace f_of_2_l395_395678

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end f_of_2_l395_395678


namespace ferris_wheel_cost_per_child_l395_395487

namespace AmusementPark

def num_children := 5
def daring_children := 3
def merry_go_round_cost_per_child := 3
def ice_cream_cones_per_child := 2
def ice_cream_cost_per_cone := 8
def total_spent := 110

theorem ferris_wheel_cost_per_child (F : ℝ) :
  (daring_children * F + num_children * merry_go_round_cost_per_child +
   num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone = total_spent) →
  F = 5 :=
by
  -- Here we would proceed with the proof steps, but adding sorry to skip it.
  sorry

end AmusementPark

end ferris_wheel_cost_per_child_l395_395487


namespace number_of_odd_s_n_is_12_l395_395796

def num_solutions (n : ℕ) : ℕ :=
  (finset.filter (λ (x : finset ℕ), x.sum = n)
    (finset.product (finset.of_list [2, 3, 5, 7]) (finset.of_list [2, 3, 5, 7])
      (finset.of_list [2, 3, 5, 7]) (finset.of_list [2, 3, 5, 7])
      (finset.of_list [1, 2, 3, 4]) (finset.of_list [1, 2, 3, 4])))
    .card

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem number_of_odd_s_n_is_12 :
  (finset.filter (λ n, is_odd (num_solutions n))
    (finset.range (10 + 1))).card = 12 :=
by
  sorry

end number_of_odd_s_n_is_12_l395_395796


namespace color_triangulation_l395_395033

theorem color_triangulation (n : ℕ)
  (h1 : n ≥ 4)
  (vertices : Fin n → ℕ)
  (colors : Fin n → ℕ)
  (h2 : ∀ i j, i ≠ j → (vertices i ≠ vertices j))
  (h3 : ∀ i, (vertices i) ∈ {1, 2, 3} )
  (h4 : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ {vertices i, vertices j, vertices k} = {1, 2, 3})
  (h5 : ∀ i, (vertices i) ≠ ( vertices ((i+1)%n))) :
  ∃ (tris : Fin (n - 2) → (Fin n) × (Fin n) × (Fin n)),
    (∀ i, (∃ (a b c : Fin n), tris i = (a, b, c) ∧ {vertices a, vertices b, vertices c} = {1, 2, 3})) :=
by
  sorry

end color_triangulation_l395_395033


namespace verify_Sasha_claim_l395_395272

open Real EuclideanGeometry Topology

universe u

-- Define a cube and vertices X and Y
def Cube := (Fin 3 → Fin 2 → ℝ)

def X : Cube := λ i j => if i = 0 then [0, 0] else if i = 1 then [0, 1] else [1, 1]
def Y : Cube := X

-- Define the shortest path calculation based on the conditions provided
def shortest_path_length (X Y : Cube): ℝ := 2 * Real.sqrt (1 ^ 2 + 1 ^ 2)

theorem verify_Sasha_claim :
  ¬(shortest_path_length X Y = 4.5) :=
by
  unfold shortest_path_length
  sorry

end verify_Sasha_claim_l395_395272


namespace complement_B_with_respect_to_A_l395_395712

noncomputable theory
open Set

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- The domain A of the function is the set of all real numbers
def A : Set ℝ := univ

-- The range B of the function
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, f(x) = y}

-- Prove that the complement of B with respect to A is (-∞, -4)
theorem complement_B_with_respect_to_A : (set.univ \ B) = Iio (-4) := sorry

end complement_B_with_respect_to_A_l395_395712


namespace angles_interval_l395_395151

section
variables (θ : ℝ)

/-- Given the angle θ and real number x between 0 and 1, the inequality holds:
x^2 * sin θ - x * (1 - x) + (1 - x)^2 * cos θ > 0 -/
def holds_for_all_x (θ : ℝ) : Prop :=
∀ x, (0 <= x ∧ x <= 1) → (x^2 * sin θ - x * (1 - x) + (1 - x)^2 * cos θ > 0)

/-- The angles θ that satisfy the given inequality for all x in [0, 1] are in the interval (π/12, 5π/12). -/
theorem angles_interval (θ : ℝ) (H : 0 <= θ ∧ θ <= 2 * π) :
(holds_for_all_x θ) ↔ (π / 12 < θ ∧ θ < 5 * π / 12) :=
sorry
end

end angles_interval_l395_395151


namespace find_a_and_b_l395_395600

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395600


namespace wealthiest_income_l395_395382

theorem wealthiest_income (N : ℝ) (x : ℝ) (hN : N = 5_000_000) 
  (h : N = 8 * 10^8 * x^(-3/2)) : x = 160^(2/3) :=
by
  sorry

end wealthiest_income_l395_395382


namespace recurrence_relation_1_recurrence_relation_2_recurrence_relation_3_recurrence_relation_4_recurrence_relation_5_l395_395840

-- (1)
theorem recurrence_relation_1 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a n = 2 * a (n - 1) + 2^n)
  (h0 : a 0 = 3) :
  ∀ n, a n = (n + 3) * 2^n := 
sorry

-- (2)
theorem recurrence_relation_2 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a n = n * a (n - 1) + (-1)^n)
  (h0 : a 0 = 3) :
  ∀ n, a n = nat.factorial n * (2 + ∑ k in finset.range (n + 1), (-1)^k / nat.factorial k) := 
sorry

-- (3)
theorem recurrence_relation_3 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a n = 2 * a (n - 1) - 1)
  (h0 : a 0 = 2) :
  ∀ n, a n = 2^n + 1 := 
sorry

-- (4)
theorem recurrence_relation_4 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a n = (1 / 2) * a (n - 1) + 1 / 2^n)
  (h0 : a 0 = 1) :
  ∀ n, a n = (n + 1) / 2^n := 
sorry

-- (5)
theorem recurrence_relation_5 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 2, a n = 4 * a (n - 2))
  (h0 : a 0 = 2)
  (h1 : a 1 = 1) :
  ∀ n, a n = 2^(n + (-1)^n) := 
sorry

end recurrence_relation_1_recurrence_relation_2_recurrence_relation_3_recurrence_relation_4_recurrence_relation_5_l395_395840


namespace inheritance_amount_l395_395292

theorem inheritance_amount
  (x : ℝ)
  (H1 : 0.25 * x + 0.15 * (x - 0.25 * x) = 15000) : x = 41379 := 
sorry

end inheritance_amount_l395_395292


namespace cars_each_remaining_day_l395_395755

theorem cars_each_remaining_day (total_cars : ℕ) (monday_cars : ℕ) (tuesday_cars : ℕ)
  (wednesday_cars : ℕ) (thursday_cars : ℕ) (remaining_days : ℕ)
  (h_total : total_cars = 450)
  (h_mon : monday_cars = 50)
  (h_tue : tuesday_cars = 50)
  (h_wed : wednesday_cars = 2 * monday_cars)
  (h_thu : thursday_cars = 2 * monday_cars)
  (h_remaining : remaining_days = (total_cars - (monday_cars + tuesday_cars + wednesday_cars + thursday_cars)) / 3)
  :
  remaining_days = 50 := sorry

end cars_each_remaining_day_l395_395755


namespace a2023_eq_t_plus_1_l395_395847

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Define the sum of the first n terms of the Fibonacci sequence
def fib_sum (n : ℕ) : ℕ := (finRange n).val.sum fibonacci

-- Given conditions
variable (t : ℕ) (Sn : ℕ → ℕ) (hSn : Sn 2021 = t)

-- Fibonacci sequence properties
lemma fib_prop (n : ℕ) : fibonacci (n+2) = fibonacci (n+1) + fibonacci n := by sorry

-- Sum property leading to the final term proof
lemma fib_sum_property (n : ℕ) : fibonacci (n+2) = (finRange n).val.sum fibonacci + 1 := by sorry

-- The main theorem to be proven
theorem a2023_eq_t_plus_1 (t : ℕ) (h : fib_sum 2021 = t) : fibonacci 2023 = t + 1 := by
  -- Use the given conditions and properties to show the equality
  exact sorry

end a2023_eq_t_plus_1_l395_395847


namespace expression_is_product_l395_395389

def not_sum (a x : Int) : Prop :=
  ¬(a + x = -7 * x)

def not_difference (a x : Int) : Prop :=
  ¬(a - x = -7 * x)

def not_quotient (a x : Int) : Prop :=
  ¬(a / x = -7 * x)

theorem expression_is_product (x : Int) : 
  not_sum (-7) x ∧ not_difference (-7) x ∧ not_quotient (-7) x → (-7 * x = -7 * x) :=
by sorry

end expression_is_product_l395_395389


namespace difference_representation_l395_395310

theorem difference_representation 
  (n k m : ℕ) (h1 : 2 ≤ k) (h2 : n ≤ m) (h3 : m < (2 * k - 1) * n / k) 
  (A : Finset ℕ) (hA : A.card = n) (hA_subset : ∀ x ∈ A, x ∈ Finset.range (m + 1)) :
  ∀ x ∈ Finset.range (n / (k - 1)), ∃ a a' ∈ A, x = a - a' :=
by
  sorry

end difference_representation_l395_395310


namespace knights_problem_l395_395890
noncomputable theory

-- Define the total number of knights
def total_knights : ℕ := 20

-- Define the number of chosen knights
def chosen_knights : ℕ := 4

-- Define a function to compute the binomial coefficient
def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability Q that at least two of the chosen knights are adjacent
def probability_adjacent (total chosen : ℕ) : ℚ :=
  let total_ways := choose total chosen
  -- Calculate the non-adjacent placements and other complementary steps here
  let ways_no_adjacent := 19 * 17 * 15 * 13 -- This needs the correct values
  1 - (ways_no_adjacent / total_ways)

-- Define the function to get the fraction in lowest terms
def lowest_terms_fraction (q : ℚ) : ℤ × ℤ :=
  let num := q.num
  let denom := q.denom
  let gcd := num.gcd denom
  (num / gcd, denom / gcd)

-- Compute the fraction and its lowest terms
def fraction_sum (q : ℚ) : ℤ :=
  let (m, n) := lowest_terms_fraction q
  m + n

-- Compute the final sum for the given problem setup
def knights_problem_solution : ℤ :=
  fraction_sum (probability_adjacent total_knights chosen_knights)

theorem knights_problem : knights_problem_solution = ?z := 
  sorry -- here we need to find the exact value.

end knights_problem_l395_395890


namespace right_triangle_area_l395_395350

theorem right_triangle_area (a : ℕ) : 
  (a + (24 - a) = 24) ∧ (24^2 + a^2 = (48 - a)^2) → 
  (1/2 * 18 * 24 = 216) :=
begin
  sorry
end

end right_triangle_area_l395_395350


namespace balls_minimum_needed_l395_395079

/-- A box contains 28 red balls, 20 green balls, 19 yellow balls, 13 blue balls, 11 white balls, and 9 black balls.
    Prove that to ensure at least 15 balls of the same color are selected, at least 76 balls must be drawn. -/
theorem balls_minimum_needed : 
  ∀ (red green yellow blue white black : ℕ), 
    red = 28 → green = 20 → yellow = 19 → blue = 13 → white = 11 → black = 9 → 
    (∃ n, n ≥ 76 ∧ ∀ selection : finset (nat), selection.card = n → 
      (15 ≤ selection.filter (λ x, x=red).card ∨ 
       15 ≤ selection.filter (λ x, x=green).card ∨ 
       15 ≤ selection.filter (λ x, x=yellow).card ∨ 
       15 ≤ selection.filter (λ x, x=blue).card ∨ 
       15 ≤ selection.filter (λ x, x=white).card ∨ 
       15 ≤ selection.filter (λ x, x=black).card)) := 
by {
  sorry
}

end balls_minimum_needed_l395_395079


namespace similar_triangle_rect_points_l395_395422

-- Define the problem statement and setup
variable (A B C D E F G H I J K L : Type) [MetricSpace E]
variables (P Q R S : E)
variables (ratio_P : ℝ) (ratio_Q : ℝ) (ratio_R : ℝ) (ratio_S : ℝ)

-- Assume given conditions: 2 pairs of perpendicular lines form 4 similar triangles.
-- Similar triangles share the same shape, but not necessarily the same size,
-- thus their corresponding angles are equal and corresponding sides are proportional.
variables (similar_tris_AB_CD : similar (triangle A B C) (triangle D E F))
variables (similar_tris_GH_JK : similar (triangle G H I) (triangle J K L))
variables (hypo_div_P : P ∈ segment (A, C))
variables (hypo_div_Q : Q ∈ segment (D, F))
variables (hypo_div_R : R ∈ segment (G, I))
variables (hypo_div_S : S ∈ segment (J, L))

-- Points P, Q, R, and S divide the hypotenuses in the same ratio
variables (same_ratio_PQ : P = ratio_P • A + (1 - ratio_P) • C)
variables (same_ratio_RS : Q = ratio_Q • D + (1 - ratio_Q) • F)
variables (same_ratio_GR : R = ratio_R • G + (1 - ratio_R) • I)
variables (same_ratio_JS : S = ratio_S • J + (1 - ratio_S) • L)

theorem similar_triangle_rect_points :
  ∃ rect : rect P Q R S, 
    rect = quadrilateral P Q R S ∧ 
    opposite_sides_parallel P Q R S ∧ 
    opposite_sides_equal P Q R S :=
sorry

end similar_triangle_rect_points_l395_395422


namespace problem1_problem2_l395_395220

-- Definitions based on given conditions
def f (a x : ℝ) : ℝ := a * x^2 - 1 - 2 * log x

-- Prove that for a=1, f(x) ≥ 0 for all x > 0
theorem problem1 : ∀ x : ℝ, 0 < x → f 1 x ≥ 0 :=
by
  sorry

-- Prove that f(x) has two zeros if and only if 0 < a < 1.
theorem problem2 (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 1 :=
by
  sorry

end problem1_problem2_l395_395220


namespace range_of_omega_l395_395203

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ (x1 x2 : ℝ), 
     0 ≤ x1 ∧ x1 ≤ 1 ∧ f x1 = 0 ∧ 
     0 ≤ x2 ∧ x2 ≤ 1 ∧ f x2 = 0 ∧ 
     x1 ≠ x2 ∧
     ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x = 0 → (x = x1 ∨ x = x2)) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ f x = 0) → 
  ∃ (a b : ℝ), a ≤ ω ∧ ω < b ∧ a = 3 * π / 4 ∧ b = 5 * π / 4 :=
sorry

def f (x : ℝ) : ℝ := 
  sin(ω * x - π / 4)

end range_of_omega_l395_395203


namespace find_x_y_n_l395_395304

def is_reverse_digit (x y : ℕ) : Prop := 
  x / 10 = y % 10 ∧ x % 10 = y / 10

def is_two_digit_nonzero (z : ℕ) : Prop := 
  10 ≤ z ∧ z < 100

theorem find_x_y_n : 
  ∃ (x y n : ℕ), is_two_digit_nonzero x ∧ is_two_digit_nonzero y ∧ is_reverse_digit x y ∧ (x^2 - y^2 = 44 * n) ∧ (x + y + n = 93) :=
sorry

end find_x_y_n_l395_395304


namespace order_of_a_b_c_l395_395184

variable (a b c : ℝ)

theorem order_of_a_b_c (h1 : a = 2^12) (h2 : b = (1/2)^(-0.2)) (h3 : c = 3^(-0.8)) :
  c < b ∧ b < a :=
by
  sorry

end order_of_a_b_c_l395_395184


namespace y_axis_intersection_l395_395090

-- Define the points (3, 18) and (-9, -6)
def P1 : (ℝ × ℝ) := (3, 18)
def P2 : (ℝ × ℝ) := (-9, -6)

-- Define the slope calculation function
def slope (P1 P2 : ℝ × ℝ) : ℝ :=
  (P1.2 - P2.2) / (P1.1 - P2.1)

-- Define the point-slope form function of the line passing through P1 and P2
def point_slope_line (P1 : ℝ × ℝ) (m : ℝ) : ℝ → ℝ :=
  λ x, m * (x - P1.1) + P1.2

-- Predicate to check the intersection with the y-axis
def intersects_y_axis_at (line_eq : ℝ → ℝ) (y : ℝ) : Prop :=
  line_eq 0 = y

theorem y_axis_intersection :
  intersects_y_axis_at (point_slope_line P1 (slope P1 P2)) 12 :=
by
  -- Proof part is omitted
  sorry

end y_axis_intersection_l395_395090


namespace Tim_pays_correct_amount_l395_395878

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end Tim_pays_correct_amount_l395_395878


namespace supermarket_profit_problem_l395_395082

noncomputable def profit_relationship (cost_price sell_price initial_sales incr_vol_per_decrement : ℝ) (x : ℝ) : ℝ :=
let price_after_decrement := sell_price - x in
let sales_volume := initial_sales + incr_vol_per_decrement * x in
(price_after_decrement - cost_price) * sales_volume

def max_profit_statement : Prop :=
  ∀ (x: ℝ), 0 ≤ x ∧ x ≤ 11 →
  profit_relationship 3.5 14.5 500 100 x = -100 * (x - 3)^2 + 6400

def max_profit_value : Prop :=
  profit_relationship 3.5 14.5 500 100 3 = 6400

def optimal_selling_price : Prop :=
  (14.5 - 3) = 11.5

theorem supermarket_profit_problem :
  max_profit_statement ∧ max_profit_value ∧ optimal_selling_price :=
by
  sorry

end supermarket_profit_problem_l395_395082


namespace louis_bought_5_yards_l395_395321

-- Definitions based on conditions in a)
def cost_per_yard : ℕ := 24
def pattern_cost : ℕ := 15
def thread_cost_per_spool : ℕ := 3
def num_spools : ℕ := 2
def total_cost : ℕ := 141 

-- Auxiliary definition for total thread cost
def total_thread_cost : ℕ := thread_cost_per_spool * num_spools

-- Lean theorem stating that Louis bought 5 yards of fabric
theorem louis_bought_5_yards :
  let V := 5 in 
  cost_per_yard * V + pattern_cost + total_thread_cost = total_cost :=
by 
  -- define V
  let V := 5 in 
  -- calculate intermediate values
  have h1 : total_thread_cost = 6 := by sorry,
  have h2 : cost_per_yard * V = 120 := by sorry,
  -- final proof
  calc cost_per_yard * V + pattern_cost + total_thread_cost
    = 120 + 15 + 6 : by rw [h2, h1]
    ... = 141 : by norm_num

end louis_bought_5_yards_l395_395321


namespace equal_roots_condition_l395_395266

theorem equal_roots_condition (x m : ℝ) :
  (m ≠ 0 ∧ m ≠ 2 ∧ x^2 - 2 ≠ 0) →
  (x^2 - 2x - (m^2 + 2)) / ((x^2 - 2) * (m - 2)) = x / m →
  m = -2 :=
by
  intros h h_eq
  sorry

end equal_roots_condition_l395_395266


namespace solve_for_a_b_l395_395639

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395639


namespace chess_player_max_consecutive_win_prob_l395_395930

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l395_395930


namespace non_congruent_triangles_count_l395_395694

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def count_non_congruent_triangles : ℕ :=
  let a_values := [1, 2]
  let b_values := [2, 3]
  let triangles := [(1, 2, 2), (2, 2, 2), (2, 2, 3)]
  triangles.length

theorem non_congruent_triangles_count : count_non_congruent_triangles = 3 :=
  by
    -- Proof would go here
    sorry

end non_congruent_triangles_count_l395_395694


namespace smallest_even_consecutive_sum_l395_395867

theorem smallest_even_consecutive_sum (n : ℕ) (h_even : n % 2 = 0) (h_sum : n + (n + 2) + (n + 4) = 162) : n = 52 :=
sorry

end smallest_even_consecutive_sum_l395_395867


namespace maci_red_pens_l395_395325

def cost_blue_pens (b : ℕ) (cost_blue : ℕ) : ℕ := b * cost_blue

def cost_red_pen (cost_blue : ℕ) : ℕ := 2 * cost_blue

def total_cost (cost_blue : ℕ) (n_blue : ℕ) (n_red : ℕ) : ℕ := 
  n_blue * cost_blue + n_red * (2 * cost_blue)

theorem maci_red_pens :
  ∀ (n_blue cost_blue n_red total : ℕ),
  n_blue = 10 →
  cost_blue = 10 →
  total = 400 →
  total_cost cost_blue n_blue n_red = total →
  n_red = 15 := 
by
  intros n_blue cost_blue n_red total h1 h2 h3 h4
  sorry

end maci_red_pens_l395_395325


namespace regular_even_gon_rhombus_division_l395_395535

theorem regular_even_gon_rhombus_division {n : ℕ} (h : n ≥ 1) :
  ∃ (partition : List (List (ℝ × ℝ))), 
  is_regular_2n_gon n ∧ sides_parallel partition ∧ 
  (∀ quad ∈ partition, is_rhombus quad) :=
sorry

end regular_even_gon_rhombus_division_l395_395535


namespace promotion_price_percentage_l395_395514

noncomputable def price_on_seventh_day (p : ℝ) : ℝ :=
  let day1 := p * (1 - 0.09) in
  let day2 := day1 * (1 + 0.05) in
  let day3 := day2 * (1 - 0.10) in
  let day4 := day3 * (1 + 0.15) in
  let day5 := day4 * (1 - 0.10) in
  let day6 := day5 * (1 + 0.08) in
  let day7 := day6 * (1 - 0.12) in
  day7

theorem promotion_price_percentage (p : ℝ) (h0 : 0 < p) :
  price_on_seventh_day p = 0.8459 * p := by
  sorry

end promotion_price_percentage_l395_395514


namespace cos_half_angle_neg_sqrt_l395_395233

theorem cos_half_angle_neg_sqrt (theta m : ℝ) 
  (h1 : (5 / 2) * Real.pi < theta ∧ theta < 3 * Real.pi)
  (h2 : |Real.cos theta| = m) : 
  Real.cos (theta / 2) = -Real.sqrt ((1 - m) / 2) :=
sorry

end cos_half_angle_neg_sqrt_l395_395233


namespace collinear_points_k_value_l395_395414

theorem collinear_points_k_value :
  ∃ (k : ℚ), (2 - 1) * (0 - 1) = (-1 - 1) * (k - 1) ↔ k = 3 / 2 := 
sorry

end collinear_points_k_value_l395_395414


namespace modulus_of_z_l395_395703

variable (z : ℂ) (i : ℂ)

theorem modulus_of_z (h : z * i = 1 + i) : |z| = Real.sqrt 2 := by
  sorry

end modulus_of_z_l395_395703


namespace tenth_term_l395_395550

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧
  a 2 = 4 ∧
  (∀ n, a (n+1) * a n = 12)

theorem tenth_term (a : ℕ → ℝ) (h : sequence a) : a 10 = 4 :=
by
  sorry

end tenth_term_l395_395550


namespace stewarts_theorem_ptomely_l395_395049

theorem stewarts_theorem_ptomely (A B C D P : Point) (a b c d p : ℝ)
  (Ptolemys : AC * BD + AD * BC = AB * CD)
  (Stewarts : AB^2 * CD + AC^2 * BD = AD * (BC^2 + CD * BD))
  (AB AC BC AD BD CD : ℝ) :
  CP^2 = (1/AB) * (AC^2 * BP + BC^2 * AP - AB * AP * BP) :=
by
-- sorry means proof is omitted
sorry

end stewarts_theorem_ptomely_l395_395049


namespace coefficient_h_nonzero_l395_395012

/-- 
Given a polynomial Q(x) = x^4 + fx^3 + gx^2 + hx + j 
with four distinct x-intercepts, including one at (0,0),
prove that the coefficient h cannot be zero.
-/
theorem coefficient_h_nonzero {f g h j : ℝ} :
  (∃ u v w : ℝ, u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0 ∧ u ≠ v ∧ u ≠ w ∧ v ≠ w ∧
    Q(x) = x*(x - u)*(x - v)*(x - w) ∧ Q(0) = 0) →
  h ≠ 0 :=
by
  sorry

end coefficient_h_nonzero_l395_395012


namespace sec_derivative_sum_zero_l395_395141

theorem sec_derivative_sum_zero :
  let sec := λ x : ℝ, 1 / cos x in
  let sec'' := λ x : ℝ, (differentiable_second n x) (2 * sec x ^ 3 - sec x) in
  sec'' (π / 4) + sec'' (3 * π / 4) + sec'' (5 * π / 4) + sec'' (7 * π / 4) = 0 := 
sorry

end sec_derivative_sum_zero_l395_395141


namespace total_amount_paid_l395_395286

theorem total_amount_paid : 
  (∀ (p cost_per_pizza : ℕ), cost_per_pizza = 12 → p = 3 → cost_per_pizza * p = 36) →
  (∀ (d cost_per_pizza : ℕ), cost_per_pizza = 12 → d = 2 → cost_per_pizza * d + 2 = 26) →
  36 + 26 = 62 :=
by {
  intro h1 h2,
  sorry
}

end total_amount_paid_l395_395286


namespace subtraction_example_l395_395124

theorem subtraction_example : 2 - 3 = -1 := 
by {
  -- We need to prove that 2 - 3 = -1
  -- The proof is to be filled here
  sorry
}

end subtraction_example_l395_395124


namespace Marcella_shoes_l395_395812

theorem Marcella_shoes (P : ℕ) (L : ℕ) (pairs : P = 20) (losses : L = 9) : ∃ M : ℕ, M = 11 :=
by
  -- Given
  have pairs_initial : P = 20 := pairs
  have losses_initial : L = 9 := losses
  -- To Show
  use 11
  have matching_pairs : 20 - 9 = 11 := by decide
  show 11 = 11 from Eq.refl 11
  sorry

end Marcella_shoes_l395_395812


namespace problem_solution_count_l395_395135

theorem problem_solution_count (n : ℕ) (h1 : (80 * n) ^ 40 > n ^ 80) (h2 : n ^ 80 > 3 ^ 160) : 
  ∃ s : Finset ℕ, s.card = 70 ∧ ∀ x ∈ s, 10 ≤ x ∧ x ≤ 79 :=
by
  sorry

end problem_solution_count_l395_395135


namespace probability_closer_to_6_than_to_1_l395_395096

theorem probability_closer_to_6_than_to_1:
  (λ (x : ℝ), 0 ≤ x ∧ x ≤ 7 → real.dist x 6 < real.dist x 1) → (3.5 / 7 = 0.5) :=
by
  sorry

end probability_closer_to_6_than_to_1_l395_395096


namespace num_smaller_cubes_is_twenty_l395_395981

-- Define the initial conditions 
def original_cube_edge_length : ℕ := 3 
def original_cube_volume := original_cube_edge_length ^ 3

-- Hypotheses: Smaller cubes have edges of whole number lengths in centimeters and are not all the same size
def valid_smaller_cube (e : ℕ) : Prop := e > 0 ∧ e ≤ original_cube_edge_length

-- The final theorem to be proved
theorem num_smaller_cubes_is_twenty: 
  ∃ N1 N2 : ℕ, 
    (valid_smaller_cube N1 ∧ valid_smaller_cube N2 ∧ N1 ≠ N2) →
    (N1 ^ 3) + 19 = original_cube_volume :=
begin
  sorry
end

end num_smaller_cubes_is_twenty_l395_395981


namespace find_m_l395_395722

-- Define the vector
def vec2 := (ℝ × ℝ)

-- Given vectors
def a : vec2 := (2, -1)
def c : vec2 := (-1, 2)

-- Definition of parallel vectors
def parallel (v1 v2 : vec2) := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Problem Statement
theorem find_m (m : ℝ) (b : vec2 := (-1, m)) (h : parallel (a.1 + b.1, a.2 + b.2) c) : m = -1 :=
sorry

end find_m_l395_395722


namespace joshua_skitles_friends_l395_395289

theorem joshua_skitles_friends :
  ∀ (total_skitles : ℕ) (skittles_per_friend : ℕ), total_skitles = 200 → skittles_per_friend = 40 → total_skitles / skittles_per_friend = 5 := 
by
  intros total_skitles skittles_per_friend h1 h2
  rw [h1, h2]
  norm_num

end joshua_skitles_friends_l395_395289


namespace total_insects_eaten_l395_395460

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end total_insects_eaten_l395_395460


namespace intersection_M_N_l395_395298

def M : Set ℝ := {x | Real.log10 x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_M_N : (M ∩ N) = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_M_N_l395_395298


namespace sqrt_neg9_sq_l395_395458

theorem sqrt_neg9_sq : Real.sqrt ((-9 : Real)^2) = 9 := 
by 
  sorry

end sqrt_neg9_sq_l395_395458


namespace find_a_b_l395_395610

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395610


namespace theta_condition_l395_395314

variables (a b : EuclideanSpace ℝ (Fin 2))
variable (θ : ℝ)

noncomputable def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sqrt (v ⬝ v)

axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom magnitude_b : magnitude b = 2
axiom magnitude_a : magnitude a = 1
axiom angle_condition : ∃ θ : ℝ, acos (a ⬝ b / (magnitude a * magnitude b)) = θ
axiom magnitude_diff : magnitude (b - a) = sqrt 3

theorem theta_condition : θ = π / 3 ↔ 
  magnitude (b - a) = sqrt 3 ∧ 
  magnitude b = 2 ∧ 
  magnitude a = 1 ∧
  ∃ θ' : ℝ, acos (a ⬝ b / (magnitude a * magnitude b)) = θ' :=
begin
  sorry
end

end theta_condition_l395_395314


namespace gcd_lcm_sum_l395_395901

open Nat

theorem gcd_lcm_sum :
  gcd 28 63 + lcm 18 24 = 79 :=
by
  sorry

end gcd_lcm_sum_l395_395901


namespace solve_for_a_b_l395_395637

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395637


namespace sum_first_10_terms_l395_395739

def a_n (n : ℕ) : ℤ := (-1)^n * (3 * n - 2)

theorem sum_first_10_terms :
  (∑ i in Finset.range 10, a_n (i + 1)) = 15 :=
by
  sorry

end sum_first_10_terms_l395_395739


namespace no_real_solution_for_log_equation_l395_395731

theorem no_real_solution_for_log_equation (p q : ℝ) : 
  log (p * q) = log (p^2 + q^2 + 1) → False := 
by
  sorry

end no_real_solution_for_log_equation_l395_395731


namespace packages_count_l395_395907

theorem packages_count (tshirts_per_package : ℕ) (total_tshirts : ℕ) (h : tshirts_per_package = 13 ∧ total_tshirts = 39) : total_tshirts / tshirts_per_package = 3 :=
by
  obtain ⟨h1, h2⟩ := h
  rw [h1, h2]
  norm_num
  done

end packages_count_l395_395907


namespace statement_A_statement_B_statement_D_l395_395805

-- Proof for Statement A: If f'(x) - f(x) / x > 0 for all x, then 3 * f(4) > 4 * f(3).
theorem statement_A (f : ℝ → ℝ) (h : ∀ x, f'(x) - f(x) / x > 0) : 3 * f(4) > 4 * f(3) :=
sorry

-- Proof for Statement B: If f(x) is an odd function, f(2)=0, and for x > 0, 2 * x * f(x) + x^2 * f'(x) > 0, then the range for which f(x) > 0 is (-2, 0) ∪ (2, ∞).
theorem statement_B (f : ℝ → ℝ) (h_odd : ∀ x, f(-x) = -f(x)) (h_f2 : f(2) = 0) 
  (h : ∀ x > 0, 2 * x * f(x) + x^2 * f'(x) > 0) : 
  ∀ x, x ∈ set.Ioo (-2 : ℝ) 0 ∪ set.Ioi (2 : ℝ) → f(x) > 0 :=
sorry

-- Proof for Statement D: If x * f(x) + x^2 * f'(x) = e^x for x > 0, f(1)=e, then f(x) is monotonically increasing on (0, ∞).
theorem statement_D (f : ℝ → ℝ) (h : ∀ x, x > 0 → x * f(x) + x^2 * f'(x) = real.exp(x)) (h1 : f(1) = real.exp(1)) : 
  ∀ x y, 0 < x → x < y → f(x) ≤ f(y) :=
sorry

end statement_A_statement_B_statement_D_l395_395805


namespace false_proposition_range_of_m_l395_395742

theorem false_proposition_range_of_m :
  (¬ ∀ x : ℝ, x^2 - 2 * x - m ≥ 0) ↔ m ∈ Ioi (-1) := sorry

end false_proposition_range_of_m_l395_395742


namespace solve_for_x_l395_395838

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l395_395838


namespace louis_bought_5_yards_l395_395322

-- Definitions based on conditions in a)
def cost_per_yard : ℕ := 24
def pattern_cost : ℕ := 15
def thread_cost_per_spool : ℕ := 3
def num_spools : ℕ := 2
def total_cost : ℕ := 141 

-- Auxiliary definition for total thread cost
def total_thread_cost : ℕ := thread_cost_per_spool * num_spools

-- Lean theorem stating that Louis bought 5 yards of fabric
theorem louis_bought_5_yards :
  let V := 5 in 
  cost_per_yard * V + pattern_cost + total_thread_cost = total_cost :=
by 
  -- define V
  let V := 5 in 
  -- calculate intermediate values
  have h1 : total_thread_cost = 6 := by sorry,
  have h2 : cost_per_yard * V = 120 := by sorry,
  -- final proof
  calc cost_per_yard * V + pattern_cost + total_thread_cost
    = 120 + 15 + 6 : by rw [h2, h1]
    ... = 141 : by norm_num

end louis_bought_5_yards_l395_395322


namespace normal_conversation_sound_lowest_hearable_sound_students_talking_impact_l395_395403

variable (I : ℝ) -- sound intensity in W/m²
variable (Y : ℝ) -- sound level in decibels

def sound_level (I : ℝ) : ℝ := 10 * log10 (I / 10^(-12))

-- Proof problem for question (1)
theorem normal_conversation_sound : sound_level 10^(-6) = 60 :=
by
  unfold sound_level
  rw [log10_div]
  rw [log10_pow]
  norm_num
  rfl
  sorry -- This skips the detailed proof

-- Proof problem for question (2)
theorem lowest_hearable_sound : sound_level I = 0 → I = 10^(-12) :=
by
  unfold sound_level
  intros h
  simp at h
  sorry -- This skips the detailed proof

-- Proof problem for question (3)
theorem students_talking_impact : sound_level (5 * 10^(-7)) > 50 :=
by
  unfold sound_level
  rw [log10_div, log10_mul, log10_of_real, log10_of_real]
  norm_num
  linarith
  sorry -- This skips the detailed proof

end normal_conversation_sound_lowest_hearable_sound_students_talking_impact_l395_395403


namespace max_prob_win_two_consecutive_is_C_l395_395959

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l395_395959


namespace sum_of_plane_angles_at_B_l395_395763

-- Given the tetrahedron ABCD with the specified conditions
variables {A B C D : Type} 
variable [euclidean_space A B C D]

-- Conditions
def plane_angles_at_vertex_A_are_right_angles (A B C D : Type) [euclidean_space A B C D] : Prop :=
  angle B A C = 90 ∧ angle B A D = 90 ∧ angle C A D = 90

def AB_eq_AC_plus_AD (A B C D : Type) [euclidean_space A B C D] : Prop :=
  AB = AC + AD

-- Theorem to prove
theorem sum_of_plane_angles_at_B (A B C D : Type) [euclidean_space A B C D]
  (h1 : plane_angles_at_vertex_A_are_right_angles A B C D)
  (h2 : AB_eq_AC_plus_AD A B C D) :
  angle C B D + angle D B A + angle A B C = 90 :=
sorry

end sum_of_plane_angles_at_B_l395_395763


namespace geometry_problem_l395_395452

variables {α : Type*} [EuclideanGeometry α]

-- Definitions of Points and Projections
variable (ABC P A' B' C' : α)
variable (A B C : Triangle α)

-- Assumptions: Orthogonality and Projections
def orthogonal_projections : Prop :=
  (P ⟂ BC ∧ P ⟂ CA ∧ P ⟂ AB)

-- Definitions of Points Intersection
variable (C₁ A₁ B₁ : α)
def is_parallel (ℓ₁ ℓ₂ : Line α) : Prop :=
  ∃ d : ℝ, ∀ p : α, p ∈ ℓ₁ → p + d ∈ ℓ₂

-- Assumptions: Projections Properties and Intersections
def projections_intersections : Prop :=
  (is_parallel (line_through P AB) (circumcircle PA'B' ) ∧
   is_parallel (line_through P BC) (circumcircle PB'C') ∧
   is_parallel (line_through P CA) (circumcircle PC'A'))

-- Goal 1: Intersection at a single point
def proof_goal_1 : Prop :=
  ∃ H : α, collinear ({A, A₁, H}) ∧ collinear ({B, B₁, H}) ∧ collinear ({C, C₁, H})

-- Goal 2: Similarity
def proof_goal_2 : Prop :=
  similar (triangle ABC) (triangle A₁ B₁ C₁)

-- Main Theorem Statement combining both goals
theorem geometry_problem :
  orthogonal_projections ABC P A' B' C' →
  projections_intersections ABC P A' B' C' C₁ A₁ B₁ →
  proof_goal_1 ABC P A' B' C₁ A₁ B₁ ∧
  proof_goal_2 ABC P A' B' C₁ A₁ B₁ :=
sorry

end geometry_problem_l395_395452


namespace maximize_probability_l395_395939

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l395_395939


namespace quadratic_equation_with_root_l395_395147

theorem quadratic_equation_with_root (x : ℝ) (h1 : x = sqrt 5 - 3) :
  ∃ (a b c : ℚ), a = 1 ∧ b = 6 ∧ c = 4 ∧ (a * x^2 + b * x + c = 0) :=
begin
  sorry
end

end quadratic_equation_with_root_l395_395147


namespace max_named_numbers_l395_395337

theorem max_named_numbers (P : ℕ → (ℚ[X])) :
  ∃ (n_max : ℕ), n_max = 20 ∧
    (∀ (f : ℕ → ℕ) (k : ℕ) (a b : ℤ) (seq : ℕ → ℕ),
      (∀ n, seq n = P (f n) (k + n)) →
      (∀ n, seq (n+1) - seq n = a) →
      ∃ N, N ≤ n_max) :=
by sorry

end max_named_numbers_l395_395337


namespace customers_left_l395_395521

-- Given conditions:
def initial_customers : ℕ := 21
def remaining_customers : ℕ := 12

-- Prove that the number of customers who left is 9
theorem customers_left : initial_customers - remaining_customers = 9 := by
  sorry

end customers_left_l395_395521


namespace sum_of_valid_k_l395_395032

theorem sum_of_valid_k :
  let base : ℂ := -4 + complex.i
  let a3_list := {n | 1 ≤ n ∧ n ≤ 9} -- a3 must be between 1 and 9
  let coefficients : list (ℤ × ℤ × ℤ × ℤ) := 
    [(1, 6, -56, 17), -- (a3, a2, imag1, real1 for a3 = 1)
     (9, 8, 49, -8)] -- placeholders for actual values, can be adjusted or expanded
  let valid_ks : list ℤ := 
    coefficients.map (λ ⟨a3, a2, imag1, real1⟩, 
      list.iota 10 |>
      list.map (λ a0, real1 * a3 + (17 * a2) - (4 * 1) + a0))
  let sum_k := valid_ks.sum
  sum_k = 465 := 
sorry

end sum_of_valid_k_l395_395032


namespace prob_qualified_bulb_factory_a_l395_395774

-- Define the given probability of a light bulb being produced by Factory A
def prob_factory_a : ℝ := 0.7

-- Define the given pass rate (conditional probability) of Factory A's light bulbs
def pass_rate_factory_a : ℝ := 0.95

-- The goal is to prove that the probability of getting a qualified light bulb produced by Factory A is 0.665
theorem prob_qualified_bulb_factory_a : prob_factory_a * pass_rate_factory_a = 0.665 :=
by
  -- This is where the proof would be, but we'll use sorry to skip the proof
  sorry

end prob_qualified_bulb_factory_a_l395_395774


namespace m_over_n_eq_l395_395854

variables (m n : ℝ)
variables (x y x1 y1 x2 y2 x0 y0 : ℝ)

-- Ellipse equation
axiom ellipse_eq : m * x^2 + n * y^2 = 1

-- Line equation
axiom line_eq : x + y = 1

-- Points M and N on the ellipse
axiom M_point : m * x1^2 + n * y1^2 = 1
axiom N_point : m * x2^2 + n * y2^2 = 1

-- Midpoint of MN is P
axiom P_midpoint : x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

-- Slope of OP
axiom slope_OP : y0 / x0 = (Real.sqrt 2) / 2

theorem m_over_n_eq : m / n = (Real.sqrt 2) / 2 :=
sorry

end m_over_n_eq_l395_395854


namespace lower_right_is_5_l395_395886

def initial_grid : list (list (option ℕ)) := [
  [some 1, none, some 2, none, some 3],
  [some 2, some 3, none, some 4, none],
  [none, some 4, none, none, some 1],
  [some 3, none, some 5, none, none],
  [none, none, none, none, none]
]

def correct_grid : list (list ℕ) := [
  [1, 5, 2, 4, 3],
  [2, 3, 1, 4, 5],
  [5, 4, 3, 2, 1],
  [3, 1, 5, 4, 2],
  [4, 2, 3, 1, 5]
]

theorem lower_right_is_5 : (correct_grid.nth 4).bind (list.nth 4) = some 5 := by
  sorry

end lower_right_is_5_l395_395886


namespace find_a_b_l395_395675

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395675


namespace rowing_probability_l395_395472

open ProbabilityTheory

theorem rowing_probability
  (P_left_works : ℚ := 3 / 5)
  (P_right_works : ℚ := 3 / 5) :
  let P_left_breaks := 1 - P_left_works
  let P_right_breaks := 1 - P_right_works
  let P_left_works_and_right_works := P_left_works * P_right_works
  let P_left_works_and_right_breaks := P_left_works * P_right_breaks
  let P_left_breaks_and_right_works := P_left_breaks * P_right_works
  P_left_works_and_right_works + P_left_works_and_right_breaks + P_left_breaks_and_right_works = 21 / 25 := by
  sorry

end rowing_probability_l395_395472


namespace cyclist_total_time_l395_395483

theorem cyclist_total_time (D : ℝ) (hD : D > 0) :
  let V := D / 5,
      V' := 1.25 * V,
      T_halfway := 2.5,
      T_remaining := 0.4 
  in V' = 1.25 * V ∧ T_total = T_halfway + T_remaining → 
  T_total = 2.9 :=
by
  sorry

end cyclist_total_time_l395_395483


namespace target_hit_probability_l395_395415

def random_numbers : List ℤ := [830, 3013, 7055, 7430, 7740, 4422, 7884, 2604, 3346, 0952, 6807, 9706, 5774, 5725, 6576, 5929, 9768, 6071, 9138, 6754]

def hits_condition (n : ℤ) : Bool :=
  let digits := [1, 2, 3, 4, 5, 6]
  n.digits.count (λ d => d ∈ digits) = 3

def hitting_numbers := random_numbers.filter hits_condition

theorem target_hit_probability :
  (hitting_numbers.length : ℝ) / (random_numbers.length : ℝ) = 0.25 :=
sorry

end target_hit_probability_l395_395415


namespace find_a_and_b_l395_395598

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395598


namespace arithmetic_sequence_property_l395_395193

variable {α : Type*} [AddSemigroup α] [HasSmul ℕ α]

noncomputable def seq {α : Type*} [AddSemigroup α] [HasSmul ℕ α] 
                      (a : α) (d : α) (n : ℕ) : α :=
a + n • d

lemma arithmetic_sequence_sum (a : α) (d : α) (n : ℕ) 
  [AddCommMonoid α] [HasSmul ℕ α] :
  ∑ i in finset.range n, seq a d i = n • a + (n*(n-1)/2) • d :=
by 
  sorry

noncomputable def S20 (a1 : α) (a20 : α) : α := 10 • (a1 + a20)

theorem arithmetic_sequence_property (a : α) (d : α) 
  (N : α) (h : S20 a (a + 19 • d) = 10 • N) : 
  N = (a + 8 • d) + (a + 11 • d) :=
by 
  sorry

end arithmetic_sequence_property_l395_395193


namespace lines_through_P_and_form_area_l395_395091

-- Definition of the problem conditions
def passes_through_P (k b : ℝ) : Prop :=
  b = 2 - k

def forms_area_with_axes (k b : ℝ) : Prop :=
  b^2 = 8 * |k|

-- Theorem statement
theorem lines_through_P_and_form_area :
  ∃ (k1 k2 k3 b1 b2 b3 : ℝ),
    passes_through_P k1 b1 ∧ forms_area_with_axes k1 b1 ∧
    passes_through_P k2 b2 ∧ forms_area_with_axes k2 b2 ∧
    passes_through_P k3 b3 ∧ forms_area_with_axes k3 b3 ∧
    k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 :=
sorry

end lines_through_P_and_form_area_l395_395091


namespace ratio_area_PZN_to_QZMR_l395_395762

-- Define the right triangle PQR with given side lengths and properties
variables (P Q R M N Z : Type) [euclidean_geometry P Q R M N Z]

-- Given conditions
variables (PQ QR : ℝ) (PQ_eq_8 : PQ = 8) (QR_eq_15 : QR = 15)
          (right_angle : angle P Q R = π / 2)
          (M_midpoint : midpoint P Q M) (N_midpoint : midpoint P R N)
          (Z_intersection : ∃! Z, line Q N ∩ line M R = {Z})

-- Define areas
variables (area_PQR : ℝ) (area_PZN : ℝ) (area_QZMR : ℝ)

-- Computed areas
variables (area_PQR_eq : area_PQR = 60)
variables (area_PZN_eq : area_PZN = 10)
variables (area_QZMR_eq : area_QZMR = 20)

-- Statement to prove the ratio of areas
theorem ratio_area_PZN_to_QZMR : 
  area_PZN / area_QZMR = 1 / 2 :=
sorry

end ratio_area_PZN_to_QZMR_l395_395762


namespace angle_between_vectors_l395_395195

open Real EuclideanSpace

variables {V : Type} [InnerProductSpace ℝ V]

theorem angle_between_vectors (a b : V) (h1 : ∥a∥ = ∥b∥) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : ⟪2 • a + b, b⟫ = 0) : real.angle a b = (2 * real.pi / 3) :=
by sorry

end angle_between_vectors_l395_395195


namespace time_until_next_consecutive_increasing_time_l395_395087

def is_valid_consecutive_increasing_time (h : ℕ) (m : ℕ) : Prop :=
  let digits := List.map (fin_to_int) ([h / 10, h % 10, m / 10, m % 10] : List (Fin 10))
  List.sorted Nat.lt digits ∧ List.pairwise Nat.succ digits

theorem time_until_next_consecutive_increasing_time :
  let current_hour := 4
  let current_minute := 56
  let next_valid_hour := 12
  let next_valid_minute := 34
  (next_valid_hour * 60 + next_valid_minute) - (current_hour * 60 + current_minute) = 458 :=
by sorry

end time_until_next_consecutive_increasing_time_l395_395087


namespace sqrt_inequality_iff_l395_395506

theorem sqrt_inequality_iff (y : ℝ) (hy_pos : y > 0) : (sqrt y < 3 * y) ↔ (y > 1 / 9) :=
sorry

end sqrt_inequality_iff_l395_395506


namespace find_a_b_l395_395588

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395588


namespace maximum_probability_second_game_C_l395_395978

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l395_395978


namespace largest_five_digit_integer_congruent_to_17_mod_26_l395_395898

theorem largest_five_digit_integer_congruent_to_17_mod_26 :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 26 = 17 ∧ ∀ m : ℕ, (10000 ≤ m ∧ m < 100000 ∧ m % 26 = 17) → m ≤ n :=
begin
  sorry
end

end largest_five_digit_integer_congruent_to_17_mod_26_l395_395898


namespace additional_steps_on_third_day_l395_395553

noncomputable def day1_steps := 200 + 300
noncomputable def day2_steps := 2 * day1_steps

theorem additional_steps_on_third_day : ∃ x_3 : ℕ, day1_steps + day2_steps + x_3 = 1600 ∧ x_3 = 100 :=
by
  use 100
  split
  simp [day1_steps, day2_steps]
  exact sorry

end additional_steps_on_third_day_l395_395553


namespace omega_value_l395_395179

noncomputable def f (omega x : ℝ) : ℝ := 3 * Real.sin (omega * x + Real.pi / 3)

theorem omega_value (omega : ℝ) (h_pos : omega > 0)
  (h_eq : f omega (Real.pi / 6) = f omega (Real.pi / 3))
  (h_min_no_max : ∀ x ∈ Ioo (Real.pi / 6) (Real.pi / 3), 
                  f omega x = f omega (Real.pi / 4)
                  → ¬(∃ x, x ∈ Ioo (Real.pi / 6) (Real.pi / 3) 
                      ∧ (∀ y ∈ Ioo (Real.pi / 6) (Real.pi / 3), f omega y ≤ f omega x))) : 
  omega = 14 / 3 :=
sorry

end omega_value_l395_395179


namespace fraction_is_one_half_l395_395291

def kay_age : ℕ := 32
def oldest_age : ℕ := 44
def fraction_of_kay_age (f : ℚ) : Prop :=
  let youngest_age := f * kay_age - 5 in
  oldest_age = 4 * youngest_age

theorem fraction_is_one_half : fraction_of_kay_age (1/2) :=
by
  sorry

end fraction_is_one_half_l395_395291


namespace value_of_a_l395_395250

noncomputable def find_side_a (A b S : ℝ) (hA : A = 2 * π / 3) (hb : b = sqrt 2) (hS : S = sqrt 3) : ℝ :=
  let c := 2 * sqrt 2
  let a_squared := b^2 + c^2 - 2 * b * c * cos A
  sqrt a_squared

theorem value_of_a : find_side_a (2 * π / 3) (sqrt 2) (sqrt 3) = sqrt 14 := by
  sorry

end value_of_a_l395_395250


namespace product_of_5_consecutive_numbers_not_square_l395_395356

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l395_395356


namespace sum_of_numbers_with_six_zeros_and_56_divisors_l395_395421

theorem sum_of_numbers_with_six_zeros_and_56_divisors :
  ∃ N1 N2 : ℕ, (N1 % 10^6 = 0) ∧ (N2 % 10^6 = 0) ∧ (N1_divisors = 56) ∧ (N2_divisors = 56) ∧ (N1 + N2 = 7000000) :=
by
  sorry

end sum_of_numbers_with_six_zeros_and_56_divisors_l395_395421


namespace line_through_point_max_distance_l395_395380

noncomputable def point := (2 : ℝ, 3 : ℝ)
noncomputable def origin := (0 : ℝ, 0 : ℝ)
noncomputable def line_eqn (x y : ℝ) := 2 * x + 3 * y - 13

theorem line_through_point_max_distance :
  (∃ l : ℝ → ℝ → Prop, l = line_eqn ∧ l point.1 point.2 = 0 ∧
  ∀ l' : ℝ → ℝ → Prop, l' point.1 point.2 = 0 → 
  let d := λ (l : ℝ → ℝ → Prop) p, (abs(l p.1 p.2)) / (sqrt ((l 1 0) ^ 2 + (l 0 1) ^ 2))
  in d l origin > d l' origin) :=
by
  sorry

end line_through_point_max_distance_l395_395380


namespace _l395_395331

noncomputable def youngest_age (total_sum : ℕ) (num_people : ℕ) (future_sum : ℕ) : ℕ :=
  let total := future_sum - 2 * num_people
  let part_sum := (num_people * (num_people - 1)) / 2
  (total - part_sum) / num_people

noncomputable theorem DoubleNinth_Festival :
  let num_people := 25
  let future_sum := 2000
  (25 : ℕ) * 2 + youngest_age future_sum num_people + 24 = 90 :=
by
  let num_people := 25
  let future_sum := 2000
  let total := future_sum - 2 * num_people
  let part_sum := (num_people * (num_people - 1)) / 2
  let youngest_age := (total - part_sum) / num_people
  have oldest_age := youngest_age + 24
  have h1 : (future_sum - 2 * num_people - part_sum) / num_people + 24 = oldest_age, from sorry
  show oldest_age = 90, from sorry

-- This theorem asserts that, given the conditions, the age of the oldest person this year is 90.

end _l395_395331


namespace sum_of_digits_1_to_10n_minus_1_l395_395820

theorem sum_of_digits_1_to_10n_minus_1 (n : ℕ) :
  (∑ k in Finset.range (10^n), Nat.digits k).sum = (1/2 : ℝ) * 9 * n * (10^n) :=
by
  sorry

end sum_of_digits_1_to_10n_minus_1_l395_395820


namespace max_prob_win_two_consecutive_is_C_l395_395961

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l395_395961


namespace monotonic_sine_omega_l395_395245

theorem monotonic_sine_omega (ω : ℝ) (hω_gt_0 : ω > 0) :
    (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 3) →
        (sin (ω * x1 + π / 6) ≤ sin (ω * x2 + π / 6))) ↔ 0 < ω ∧ ω ≤ 1 := sorry

end monotonic_sine_omega_l395_395245


namespace bug_position_after_2023_jumps_l395_395830

def position_after_jumps (start : ℕ) (jumps : ℕ) : ℕ :=
  let cycle := [7, 2, 5]
  cycle[(jumps % cycle.length)]

theorem bug_position_after_2023_jumps : position_after_jumps 7 2023 = 2 :=
by
  -- The exact proof goes here, but for now we use sorry to denote the skipped proof
  sorry

end bug_position_after_2023_jumps_l395_395830


namespace a₁_a₄_prod_a₂_a₃_sum_T_sum_l395_395202

-- Definitions based on the conditions provided
def geom_seq (a : ℕ → ℕ) := ∃ q, ∀ n, a (n + 1) = q * a n

def a₁ := 2^0
def a₂ := 2^1
def a₃ := 2^2
def a₄ := 2^3

theorem a₁_a₄_prod : a₁ * a₄ = 8 := by
  sorry

theorem a₂_a₃_sum : a₂ + a₃ = 6 := by
  sorry

def b (a : ℕ → ℕ) (n : ℕ) := a n + Int.log2 (a (n + 1))

def T (n : ℕ) := (∑ i in Finset.range n, b (λ n, 2^(n-1)) i)

theorem T_sum (n : ℕ) : T n = 2^n - 1 + n * (n + 1) / 2 := by
  sorry

end a₁_a₄_prod_a₂_a₃_sum_T_sum_l395_395202


namespace count_odd_x_neg_exp_eq_six_l395_395167

noncomputable def expression_x4_sub_57x2_add_56 (x : ℤ) : ℤ :=
  x^4 - 57 * x^2 + 56

def is_odd (x : ℤ) : Prop :=
  x % 2 = 1

def negative_expression (x : ℤ) : Prop :=
  expression_x4_sub_57x2_add_56 x < 0

theorem count_odd_x_neg_exp_eq_six :
  {x : ℤ // is_odd x ∧ negative_expression x}.to_finset.card = 6 := sorry

end count_odd_x_neg_exp_eq_six_l395_395167


namespace find_a_b_l395_395648

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395648


namespace ratio_5_to_1_l395_395915

theorem ratio_5_to_1 (x : ℤ) (h : x = 55) : x / 11 = 5 :=
by
  rw h
  norm_num
  sorry

end ratio_5_to_1_l395_395915


namespace perfume_price_decrease_l395_395498

theorem perfume_price_decrease :
  let original_price := 1200
  let increased_price := original_price * (1 + 10 / 100)
  let final_price := increased_price * (1 - 15 / 100)
  original_price - final_price = 78 := by
  calc
  original_price - final_price = ...
  sorry

end perfume_price_decrease_l395_395498


namespace cos_B_value_l395_395780

variable {A B C : ℝ}
variable {a b c : ℝ} (ha : a = 2) (hb : b = 4) (hc : c = 5)
variable (h1 : sin A / sin B = 2 / 4) (h2 : sin B / sin C = 4 / 5)

theorem cos_B_value (h : (2 / 4) = (sin A / sin B) ∧ (4 / 5) = (sin B / sin C)) :
  cos B = 13 / 20 := by
  sorry

end cos_B_value_l395_395780


namespace find_m_l395_395398

noncomputable def quadratic_vertex (a b x0 : ℝ) : Prop :=
  x0 = -b / (2 * a)

theorem find_m :
  ∃ (m : ℝ), quadratic_vertex (-1) (8 - m) 2 :=
by
  use 4
  rw quadratic_vertex
  simp
  sorry

end find_m_l395_395398


namespace disprove_false_by_contradiction_l395_395438

variable {P : Prop} (h : ¬ P)

theorem disprove_false_by_contradiction : ¬ (¬ P → false) → true :=
by
  intro h₁
  exact trivial

end disprove_false_by_contradiction_l395_395438


namespace find_a_b_l395_395605

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395605


namespace pens_in_shop_l395_395862

theorem pens_in_shop (P Pe E : ℕ) (h_ratio : 14 * Pe = 4 * P) (h_ratio2 : 14 * E = 14 * 3 + 11) (h_P : P = 140) (h_E : E = 30) : Pe = 40 :=
sorry

end pens_in_shop_l395_395862


namespace coefficient_x2y4_l395_395267

theorem coefficient_x2y4 (x y : ℤ) : (1 + x + y^2)^5 = ∑ k : ℕ, (λ i, (λ j, (binom 5 i * (1 + x)^(5 - i) * y^(2 * i))) (5 - k) * (x^2) * (y^4) : zmod k) := sorry

end coefficient_x2y4_l395_395267


namespace student_failed_by_89_marks_l395_395108

-- Define the conditions
def total_marks : ℕ := 800
def marks_obtained : ℕ := 175
def passing_percentage : ℝ := 0.33

-- Calculate minimum passing marks
def minimum_passing_marks : ℕ := (passing_percentage * total_marks).to_nat

-- Prove that the student failed by 89 marks
theorem student_failed_by_89_marks : minimum_passing_marks - marks_obtained = 89 := by
  -- the proof will be filled in later
  sorry

end student_failed_by_89_marks_l395_395108


namespace proof_problem_l395_395197

variables {x y : ℝ}
def p : Prop := x > y → -x < -y
def q : Prop := x < y → x^2 < y^2

theorem proof_problem (h₁ : p) (h₂ : ¬q) :
  (p ∧ q = false) ∧ (p ∨ q = true) ∧ (p ∧ ¬q = true) ∧ (¬p ∨ q = false) :=
by {
  sorry
}

end proof_problem_l395_395197


namespace subsequent_flights_requirements_l395_395450

-- Define the initial conditions
def late_flights : ℕ := 1
def on_time_flights : ℕ := 3
def total_initial_flights : ℕ := late_flights + on_time_flights

-- Define the number of subsequent flights needed
def subsequent_flights_needed (x : ℕ) : Prop :=
  let total_flights := total_initial_flights + x
  let on_time_total := on_time_flights + x
  (on_time_total : ℚ) / (total_flights : ℚ) > 0.40

-- State the theorem to prove
theorem subsequent_flights_requirements:
  ∃ x : ℕ, subsequent_flights_needed x := sorry

end subsequent_flights_requirements_l395_395450


namespace january_first_is_tuesday_l395_395427

-- Define the days of the week for convenience
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define the problem conditions
def daysInJanuary : Nat := 31
def weeksInJanuary : Nat := daysInJanuary / 7   -- This is 4 weeks
def extraDays : Nat := daysInJanuary % 7         -- This leaves 3 extra days

-- Define the problem as proving January 1st is a Tuesday
theorem january_first_is_tuesday (fridaysInJanuary : Nat) (mondaysInJanuary : Nat)
    (h_friday : fridaysInJanuary = 4) (h_monday: mondaysInJanuary = 4) : Weekday :=
  -- Avoid specific proof steps from the solution; assume conditions and directly prove the result
  sorry

end january_first_is_tuesday_l395_395427


namespace maximize_probability_when_C_second_game_l395_395953

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l395_395953


namespace min_value_of_reciprocals_l395_395305

theorem min_value_of_reciprocals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) (hz1 : z = 1) :
    1/x + 1/y + 1/z ≥ 3 :=
by
  have hz_pos : 0 < z := hz
  have hz_eq_one : z = 1 := hz1
  rw [hz_eq_one, one_div_one] at *
  sorry

end min_value_of_reciprocals_l395_395305


namespace using_five_fours_l395_395425

theorem using_five_fours (n : ℕ) (h : 1 ≤ n ∧ n ≤ 22) : ∃ (expr : ℕ), expr = n ∧ contains_five_fours expr :=
sorry

noncomputable def contains_five_fours (expr : ℕ) : Prop :=
  ∃ (fours : list ℕ), fours.length = 5 ∧ valid_expression expr fours

noncomputable def valid_expression (expr : ℕ) (fours : list ℕ) : Prop :=
  ∃ (ops : list (ℕ → ℕ → ℕ)), valid_ops ops ∧ expr = eval_expression expr ops fours

noncomputable def valid_ops (ops : list (ℕ → ℕ → ℕ)) : Prop :=
  ∀ op ∈ ops, is_arithmetic_or_exponentiation op

noncomputable def is_arithmetic_or_exponentiation (op : ℕ → ℕ → ℕ) : Prop :=
  op = (+) ∨ op = (-) ∨ op = (*) ∨ op = (/) ∨ op = (^) ∨ op = (!)

noncomputable def eval_expression (expr : ℕ) (ops : list (ℕ → ℕ → ℕ)) (fours : list ℕ) : ℕ :=
  -- implementation detail to evaluate the expression based on ops and fours
  sorry

end using_five_fours_l395_395425


namespace solve_for_a_b_l395_395633

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395633


namespace domain_log_function_is_interval_l395_395379

def domain_of_log_function (f : ℝ → ℝ) : Set ℝ :=
{ x : ℝ | 3 * x + 1 > 0 }

theorem domain_log_function_is_interval :
  domain_of_log_function (fun x => log (3 * x + 1)) = { x : ℝ | -1/3 < x } :=
by
  sorry

end domain_log_function_is_interval_l395_395379


namespace find_g_inv_f_3_l395_395235

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_g_eq : ∀ x : ℝ, f_inv (g x) = x^4 - x + 2
axiom g_has_inverse : ∀ y : ℝ, g (g_inv y) = y 

theorem find_g_inv_f_3 :
  ∃ α : ℝ, (α^4 - α - 1 = 0) ∧ g_inv (f 3) = α :=
sorry

end find_g_inv_f_3_l395_395235


namespace find_monic_cubic_polynomial_l395_395794

noncomputable def monic_cubic_polynomial (p : ℝ[X]) : Prop :=
  p.degree = 3 ∧ p.leading_coeff = 1

def satisfies_conditions (p : ℝ[X]) : Prop :=
  p.eval 0 = 1 ∧ ∀ x, p.derivative.eval x = 0 → p.eval x = 0

theorem find_monic_cubic_polynomial (p : ℝ[X]) :
  monic_cubic_polynomial p ∧ satisfies_conditions p ↔ p = (X + 1)^3 :=
sorry

end find_monic_cubic_polynomial_l395_395794


namespace no_square_number_with_ones_l395_395782

theorem no_square_number_with_ones (n : ℕ) (digits : ℕ → bool) :
  (∃ m : ℕ, (n = m^2) ∧ 
          (∀ i : ℕ, digits i = tt → (i < n.bit0)) ∧
          (∃ k : ℕ, (n.bit0).count digits = k) ∧ 
          (k = 2 ∨ k = 3) ∧ 
          (∀ j : ℕ, digits j = tt → digits j ∈ {0, 1})) → 
          false :=
by {
  sorry
}

end no_square_number_with_ones_l395_395782


namespace average_height_plants_l395_395926

theorem average_height_plants (h1 h3 : ℕ) (h1_eq : h1 = 27) (h3_eq : h3 = 9)
  (prop : ∀ (h2 h4 : ℕ), (h2 = h1 / 3 ∨ h2 = h1 * 3) ∧ (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧ (h4 = h3 / 3 ∨ h4 = h3 * 3)) : 
  ((27 + h2 + 9 + h4) / 4 = 12) :=
by 
  sorry

end average_height_plants_l395_395926


namespace price_difference_is_correct_l395_395493

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end price_difference_is_correct_l395_395493


namespace faster_train_speed_is_45_l395_395892

noncomputable def speedOfFasterTrain (V_s : ℝ) (length_train : ℝ) (time : ℝ) : ℝ :=
  let V_r : ℝ := (length_train * 2) / (time / 3600)
  V_r - V_s

theorem faster_train_speed_is_45 
  (length_train : ℝ := 0.5)
  (V_s : ℝ := 30)
  (time : ℝ := 47.99616030717543) :
  speedOfFasterTrain V_s length_train time = 45 :=
sorry

end faster_train_speed_is_45_l395_395892


namespace number_of_distinct_products_of_S_l395_395299

open Nat

def S := {n : ℕ | n > 0 ∧ ∃ (a b c : ℕ), a ≤ 4 ∧ b ≤ 2 ∧ c ≤ 2 ∧ n = 2^a * 3^b * 5^c}

theorem number_of_distinct_products_of_S:
  ∃! (count : ℕ), (count = 221) ∧ (∀ x y ∈ S, x ≠ y → ∃ z ∈ S, z = x * y) :=
sorry

end number_of_distinct_products_of_S_l395_395299


namespace converse_of_pythagorean_theorem_l395_395368

theorem converse_of_pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  ∃ (α : ℝ) (hα : α = 90), (a^2 + b^2 = c^2 ↔ cos α = 0) :=
by
  sorry

end converse_of_pythagorean_theorem_l395_395368


namespace find_angle_D_l395_395349

-- Define the conditions
variables {A B C D E : Type} [point A] [point B] [point C] [point D] [point E]
variable (BD AE : segment)
variable (intersect : ∃ C, BD ∩ AE = {C})
variable (AB_eq_BC : dist A B = dist B C)
variable (CD_eq_DE : dist C D = dist D E)
variable (DE_eq_EC : dist D E = dist E C)
variable (angleA_eq_4angleB : ∃ θ, angle ∠ A = 4 * θ ∧ angle ∠ B = θ)

-- Define the goal
theorem find_angle_D : ∃ θ : ℝ, θ = 52.5 :=
by sorry

end find_angle_D_l395_395349


namespace mrs_hilt_apples_l395_395813

theorem mrs_hilt_apples (hours : ℕ := 3) (rate : ℕ := 5) : 
  (rate * hours) = 15 := 
by sorry

end mrs_hilt_apples_l395_395813


namespace perfect_square_subsequence_exists_l395_395683

theorem perfect_square_subsequence_exists 
  {n N : ℕ} 
  (a : Fin n → ℕ) 
  (b : Fin N → ℕ)
  (h_seq : ∀ i : Fin N, ∃ j : Fin n, b i = a j)
  (h_len : N ≥ 2 ^ n) :
  ∃ (k h : Fin N), k < h ∧
  (∃ (subseq : Fin (h.val - k.val) → ℕ), 
   (∀ i : Fin (h.val - k.val), subseq i = b ⟨k.val + 1 + i.val, begin
       apply Nat.lt_of_lt_of_le i.isLt,
       apply Nat.lt_of_lt_of_le,
       apply Nat.lt_base_succ,
       exact (Nat.zero_le _),
     end⟩) ∧ 
   ∃ m : ℕ, m ^ 2 = List.prod (List.ofFn subseq)) := sorry

end perfect_square_subsequence_exists_l395_395683


namespace no_integer_n_ge_one_divides_9_l395_395547

theorem no_integer_n_ge_one_divides_9 (n : ℤ) (h : n ≥ 1) : ¬ (9 ∣ (7^n + n^3)) :=
by sorry

end no_integer_n_ge_one_divides_9_l395_395547


namespace units_digit_of_99_factorial_is_zero_l395_395902

theorem units_digit_of_99_factorial_is_zero : (99.factorial % 10) = 0 := by
  -- The proof is omitted as per instructions
  sorry

end units_digit_of_99_factorial_is_zero_l395_395902


namespace complement_union_l395_395232

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  U \ (M ∪ N) = {4} :=
by
  sorry

end complement_union_l395_395232


namespace number_of_pipes_l395_395504

theorem number_of_pipes (h : ℝ) (V : ℝ) (r1 r2 : ℝ) 
  (h_condition : h ≠ 0) (V_condition : V = pi * r1^2 * h) 
  (r1_eq : r1 = 6) (r2_eq : r2 = 1.5) :
  let V12 := pi * (r1^2) * h,
      V3 := pi * (r2^2) * h in
  (V12 = 36 * pi * h) ∧ (V3 = 2.25 * pi * h) →
  let n := (36 * pi * h) / (2.25 * pi * h) in
  n = 16 :=
by
  intro h_condition V_condition r1_eq r2_eq,
  let V12 := pi * (r1^2) * h,
  let V3 := pi * (r2^2) * h,
  assume V_props : (V12 = 36 * pi * h) ∧ (V3 = 2.25 * pi * h),
  let n := (36 * pi * h) / (2.25 * pi * h),
  have : n = 16 := by sorry,
  assumption

end number_of_pipes_l395_395504


namespace CN_equidistant_BK_AM_l395_395753

-- Define points and conditions
variables {A B C D N M K : Type}
variable [linear_ordered_field A B C D N M K]

-- Tetrahedron ABCD with provided conditions
structure Tetrahedron :=
  (A B C D : point)
  (CD_perpendicular_to_ABC : ⊥ CD (plane_of_triangle A B C))
  (N_midpoint_AB : midpoint N A B)
  (M_midpoint_BD : midpoint M B D)
  (K_divides_DC_2_1 : divides_ratio K D C (2 / 3))

-- Prove CN is equidistant from BK and AM
theorem CN_equidistant_BK_AM (T : Tetrahedron) :
  equidistant (line T.C T.N) (line T.B T.K) (line T.A T.M) :=
by
  sorry

end CN_equidistant_BK_AM_l395_395753


namespace intersecting_points_sum_l395_395856

theorem intersecting_points_sum (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1 : ∀ x y, (y = x ^ 3 - 4 * x + 3) ∧ (x + 3 * y = 3) → 
        (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :
  (x1 + x2 + x3 = 0) ∧ (y1 + y2 + y3 = 3) :=
by
  sorry

end intersecting_points_sum_l395_395856


namespace negation_of_proposition_l395_395391

-- Definitions based on given conditions
def is_not_divisible_by_2 (n : ℤ) := n % 2 ≠ 0
def is_odd (n : ℤ) := n % 2 = 1

-- The negation proposition to be proved
theorem negation_of_proposition : ∃ n : ℤ, is_not_divisible_by_2 n ∧ ¬ is_odd n := 
sorry

end negation_of_proposition_l395_395391


namespace largest_minus_smallest_49_31_76_62_l395_395407

def largest_minus_smallest (a b c d : ℕ) := max (max a b) (max c d) - min (min a b) (min c d)

theorem largest_minus_smallest_49_31_76_62 :
  largest_minus_smallest 49 31 76 62 = 45 := 
by
  dsimp [largest_minus_smallest]
  simp [Nat.max, Nat.min]
  sorry

end largest_minus_smallest_49_31_76_62_l395_395407


namespace tangent_line_equation_l395_395004

open Real

/-- Define the curve function. -/
def curve (x : ℝ) : ℝ := x^2 + 2 * x - 1

/-- Define the point of tangency. -/
def point_of_tangency : ℝ × ℝ := (1, 2)

/-- Define what it means for a line to be tangent to the curve at the point (1, 2). -/
def is_tangent_line (line_eq: ℝ → ℝ → Prop) : Prop :=
  ∃ m : ℝ, ∃ (b : ℝ), (line_eq x y ↔ y = m * x + b) ∧
  (m = 2 * 1 + 2) ∧ 
  (b = 2 - m * 1)

/-- The statement to prove. -/
theorem tangent_line_equation :
  is_tangent_line (λ x y, 4 * x - y - 2 = 0) :=
sorry

end tangent_line_equation_l395_395004


namespace max_bag_weight_is_50_l395_395030

noncomputable def max_weight_allowed (people bags_per_person more_bags_allowed total_weight : ℕ) : ℝ := 
  total_weight / ((people * bags_per_person) + more_bags_allowed)

theorem max_bag_weight_is_50 : ∀ (people bags_per_person more_bags_allowed total_weight : ℕ), 
  people = 6 → 
  bags_per_person = 5 → 
  more_bags_allowed = 90 → 
  total_weight = 6000 →
  max_weight_allowed people bags_per_person more_bags_allowed total_weight = 50 := 
by 
  sorry

end max_bag_weight_is_50_l395_395030


namespace maximize_prob_of_consecutive_wins_l395_395944

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l395_395944


namespace league_tournament_scores_l395_395986

theorem league_tournament_scores (n : ℕ) (games : ℕ) (points : ℕ → ℕ) :
  -- Conditions
  n = 15 → 
  games = (15 * 28) / 2 →
  
  -- a point rule function that returns points contributed by a game result:
  (∀ x, points x ∈ {2, 3}) → 
  
  -- Conclusion
  420 ≤ points games ∧ points games ≤ 630 :=
by
  intros
  sorry

end league_tournament_scores_l395_395986


namespace num_non_congruent_tris_l395_395691

theorem num_non_congruent_tris (a b : ℕ) (h1 : a ≤ 2) (h2 : 2 ≤ b) (h3 : a + 2 > b) (h4 : a + b > 2) (h5 : b + 2 > a) : 
  ∃ q, q = 3 := 
by 
  use 3 
  sorry

end num_non_congruent_tris_l395_395691


namespace minimum_ratio_l395_395714

-- Define the given parabola
def parabola (p : ℝ) (p_pos : 0 < p) := { (x, y) : ℝ × ℝ | x^2 = 2*p*y }

-- Define the given line
def line (x : ℝ) := 6 * x + 8

-- The dot product condition for vectors OA and OB
def orthogonality_condition (x1 x2 y1 y2 : ℝ) := x1 * x2 + y1 * y2 = 0

-- The center of the moving circle on the parabola
def center_on_parabola (a : ℝ) := (a, a^2 / 8)

-- The equation of the moving circle passing through the fixed point D(0, 4)
def circle_eq (x a : ℝ) := (x - a) ^ 2 + (4 - a^2 / 8) ^ 2 = a^2

-- Points of intersection with the x-axis and their distances from D(0, 4)
def point_E (a : ℝ) := (a - 4, 0)
def point_F (a : ℝ) := (a + 4, 0)
def dist (x y : ℝ × ℝ) := real.sqrt ((x.1 - y.1) ^ 2 + (x.2 - y.2) ^ 2)

-- Definition to prove the minimum value of the ratio of distances
theorem minimum_ratio (a : ℝ) (ha : 0 < a) :
  ∃ a, ∀ a > 0, dist (point_E a) (0, 4) / dist (point_F a) (0, 4) = real.sqrt 2 - 1 := sorry

end minimum_ratio_l395_395714


namespace cube_volume_l395_395771

theorem cube_volume (x : ℝ) (L K F : ℝ × ℝ × ℝ) :
  L = (0, x / 2, 0) ∧ K = (x / 2, 0, 0) ∧ F = (x, x, x) ∧
  ∃ M : ℝ × ℝ × ℝ, M = ((0 + x / 2) / 2, (x / 2 + 0) / 2, (0 + 0) / 2) ∧ 
  dist M F = 10 →
  (x ^ 3 ≈ 323) :=
  by
  sorry

end cube_volume_l395_395771


namespace triangle_area_correct_l395_395111

-- Define the equations of the lines
def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := -x + 4
def line3 : ℝ := 2

-- Define the vertices of the triangle
def vertex1 : (ℝ × ℝ) := (0.5, 2)
def vertex2 : (ℝ × ℝ) := (2, 2)
def vertex3 : (ℝ × ℝ) := (1, 3)

-- Calculate the area of the triangle
def triangle_area : ℝ := 1 / 2 * (2 - 0.5) * 1

theorem triangle_area_correct : triangle_area = 0.75 := by
  sorry

end triangle_area_correct_l395_395111


namespace circle_equation_l395_395707

theorem circle_equation (x y : ℝ) :
  let center := (-1, 2)
  let radius := real.sqrt 5
  let circle_eq := x^2 + y^2 + 2 * x - 4 * y = 0
  (x + 1)^2 + (y - 2)^2 = radius^2 ↔ circle_eq :=
by
  sorry

end circle_equation_l395_395707


namespace distribute_stops_between_companies_l395_395070

variable (City : Type) (Route : Type) (Stop : Type)
variable (routes : Route → Set Stop)
variable (intersects_at_exactly_one_stop : ∀ r1 r2 : Route, r1 ≠ r2 → ∃! S : Stop, S ∈ routes r1 ∧ S ∈ routes r2)
variable (at_least_four_stops : ∀ r : Route, Set.card (routes r) ≥ 4)

-- Proof goal
theorem distribute_stops_between_companies : 
  ∃ (companyA companyB : Set Stop), 
  (∀ r : Route, ∃ S1 ∈ routes r, S1 ∈ companyA ∧ ∃ S2 ∈ routes r, S2 ∈ companyB) :=
sorry

end distribute_stops_between_companies_l395_395070


namespace find_n_l395_395329

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  11 + (n - 1) * 6

-- State the problem
theorem find_n (n : ℕ) : 
  (∀ m : ℕ, m ≥ n → arithmetic_sequence m > 2017) ↔ n = 336 :=
by
  sorry

end find_n_l395_395329


namespace inscribed_circle_diameter_l395_395895

def diameter_of_inscribed_circle (DE DF EF : ℝ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  2 * r

theorem inscribed_circle_diameter (DE DF EF : ℝ) 
  (hDE : DE = 13) (hDF : DF = 7) (hEF : EF = 10) :
  diameter_of_inscribed_circle DE DF EF = (8 * Real.sqrt 3) / 3 := by
  have h1 : (DE + DF + EF) / 2 = 15 := by
    rw [hDE, hDF, hEF]
    norm_num
  have h2 : Real.sqrt (15 * (15 - 13) * (15 - 7) * (15 - 10)) = 20 * Real.sqrt 3 := by
    norm_num
  have h3 : 20 * Real.sqrt 3 / 15 = (4 * Real.sqrt 3) / 3 := by
    norm_num
  have h4 : 2 * ((4 * Real.sqrt 3) / 3) = (8 * Real.sqrt 3) / 3 := by
    norm_num
  rw [diameter_of_inscribed_circle, h1, h2, h3, h4]
  sorry

end inscribed_circle_diameter_l395_395895


namespace equal_points_distribution_l395_395412

theorem equal_points_distribution : 
  ∃ (s1 s2 s3 : List ℕ),
    s1.sum = 71 ∧ 
    s2.sum = 71 ∧ 
    s3.sum = 71 ∧ 
    (s1 ++ s2 ++ s3).Permutation [50, 25, 25, 20, 20, 20, 10, 10, 10, 5, 5, 3, 3, 2, 2, 1, 1, 1] :=
sorry

end equal_points_distribution_l395_395412


namespace Rihanna_money_left_l395_395826

theorem Rihanna_money_left (initial_money mango_count juice_count mango_price juice_price : ℕ)
  (h_initial : initial_money = 50)
  (h_mango_count : mango_count = 6)
  (h_juice_count : juice_count = 6)
  (h_mango_price : mango_price = 3)
  (h_juice_price : juice_price = 3) :
  initial_money - (mango_count * mango_price + juice_count * juice_price) = 14 :=
sorry

end Rihanna_money_left_l395_395826


namespace total_bill_l395_395846

variable (B : ℝ)
variable (h1 : 9 * (B / 10 + 3) = B)

theorem total_bill : B = 270 :=
by
  -- proof would go here
  sorry

end total_bill_l395_395846


namespace Finn_bought_index_cards_l395_395551

theorem Finn_bought_index_cards (x y : ℝ) :
  (15 * 1.85 + 7 * x = 55.40) ∧ (12 * 1.85 + y * x = 61.70) → y = 10 :=
by
  intro h
  cases h,
  sorry

end Finn_bought_index_cards_l395_395551


namespace Tim_bodyguards_weekly_pay_l395_395883

theorem Tim_bodyguards_weekly_pay :
  let hourly_rate := 20
  let num_bodyguards := 2
  let daily_hours := 8
  let weekly_days := 7
  Tim pays $2240 in a week := (hourly_rate * num_bodyguards * daily_hours * weekly_days = 2240) :=
begin
  sorry
end

end Tim_bodyguards_weekly_pay_l395_395883


namespace fraction_of_students_with_buddy_l395_395757

theorem fraction_of_students_with_buddy (s n : ℕ) (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s : ℚ) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l395_395757


namespace solve_log_equation_l395_395842

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solve_log_equation (x : ℝ) (hx : 2 * log_base 5 x - 3 * log_base 5 4 = 1) :
  x = 4 * Real.sqrt 5 ∨ x = -4 * Real.sqrt 5 :=
sorry

end solve_log_equation_l395_395842


namespace max_prob_two_consecutive_wins_l395_395966

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l395_395966


namespace remainder_of_sum_is_4_l395_395056

theorem remainder_of_sum_is_4 : 
  let seq_sum := (List.sum (List.map (fun n => 8 * n - 5) (List.range 40))),
      remainder := seq_sum % 8
  in remainder = 4 :=
by 
  sorry

end remainder_of_sum_is_4_l395_395056


namespace smallest_even_consecutive_sum_l395_395866

theorem smallest_even_consecutive_sum (n : ℕ) (h_even : n % 2 = 0) (h_sum : n + (n + 2) + (n + 4) = 162) : n = 52 :=
sorry

end smallest_even_consecutive_sum_l395_395866


namespace minimum_area_triangle_l395_395767

namespace PolarCoordinates

-- Definitions based on conditions
def curve1 (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ
def line (ρ θ : ℝ) : Prop := ρ * Real.cos (θ + Real.pi / 4) = 2 * Real.sqrt 2
def curve2 (φ : ℝ) (P : ℝ × ℝ) : Prop := P.1 = 2 * Real.cos φ ∧ P.2 = Real.sin φ

-- Stating the proof problem
theorem minimum_area_triangle 
  (A B : ℝ × ℝ) (φ : ℝ) (P : ℝ × ℝ)
  (h1: A = (4, 0)) 
  (h2: B = (2, -2)) 
  (h3: curve1 4 0) 
  (h4: line 4 0)
  (h5: curve1 (2*Real.sqrt 2) 7*Real.pi/4) 
  (h6: line (2*Real.sqrt 2) 7*Real.pi/4) 
  (h7: curve2 φ P) : 
  ∃ A B, A = (4, 0) ∨ A = (2, -2) ∨ B = (4, 0) ∨ B = (2, -2) ∧
  ∃ P : ℝ × ℝ, curve2 φ P ∧ triangle_area A B P = 4 - Real.sqrt 5 :=
sorry

end PolarCoordinates

end minimum_area_triangle_l395_395767


namespace final_cost_is_30_l395_395037

-- Define conditions as constants
def cost_of_repair : ℝ := 7
def sales_tax : ℝ := 0.50
def number_of_tires : ℕ := 4

-- Define the cost for one tire repair
def cost_one_tire : ℝ := cost_of_repair + sales_tax

-- Define the cost for all tires
def total_cost : ℝ := cost_one_tire * number_of_tires

-- Theorem stating that the total cost is $30
theorem final_cost_is_30 : total_cost = 30 :=
by
  sorry

end final_cost_is_30_l395_395037


namespace average_speed_is_correct_l395_395529

-- Define the conditions
def initial_odometer : ℕ := 2552
def final_odometer : ℕ := 2882
def time_first_day : ℕ := 5
def time_second_day : ℕ := 7

-- Calculate total time and distance
def total_time : ℕ := time_first_day + time_second_day
def total_distance : ℕ := final_odometer - initial_odometer

-- Prove that the average speed is 27.5 miles per hour
theorem average_speed_is_correct : (total_distance : ℚ) / (total_time : ℚ) = 27.5 :=
by
  sorry

end average_speed_is_correct_l395_395529


namespace find_total_value_l395_395991

noncomputable def total_value (V : ℝ) : Prop :=
  let import_tax := if V > 1000 then 0.12 * (V - 1000) else 0
  let vat := if V > 1500 then 0.05 * (V - 1500) else 0

theorem find_total_value : ∃ V : ℝ, total_value V ∧ V = 2784 :=
by
  have V := 2784
  use V
  dsimp [total_value]
  have import_tax := if V > 1000 then 0.12 * (V - 1000) else 0
  have vat := if V > 1500 then 0.05 * (V - 1500) else 0
  have combined_tax := import_tax + vat
  dsimp [combined_tax] at *
  rw [if_pos (by linarith), if_pos (by linarith)] at combined_tax
  norm_num at combined_tax
  tauto

end find_total_value_l395_395991


namespace find_a4_l395_395775

noncomputable def a : ℕ → ℕ
| 0     := 0   -- normally the index starts from 0 in Lean
| 1     := 1
| (n+2) := 2 * a (n + 1) + 1

theorem find_a4 : a 4 = 15 :=
by {
  sorry
}

end find_a4_l395_395775


namespace maximize_probability_l395_395937

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l395_395937


namespace find_a_b_l395_395647

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395647


namespace polynomial_has_factor_l395_395023

theorem polynomial_has_factor
  (a b c d : Polynomial ℂ) :
  a.comp(Polynomial.X ^ 5) + Polynomial.X * b.comp(Polynomial.X ^ 5) + Polynomial.X ^ 2 * c.comp(Polynomial.X ^ 5) = 
    (1 + Polynomial.X + Polynomial.X ^ 2 + Polynomial.X ^ 3 + Polynomial.X ^ 4) * d →
  a.eval 1 = 0 :=
by
  sorry

end polynomial_has_factor_l395_395023


namespace find_a_and_b_l395_395595

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395595


namespace largest_n_divisible_by_n_plus_11_l395_395899

theorem largest_n_divisible_by_n_plus_11 (n : ℕ) (h : n + 11 ∣ n^3 + 101) : n ≤ 302 :=
begin
  have h' : 313 ∣ (n + 11) ∨ (n + 11) ∣ 313,
  { sorry }, -- Apply Euclidean algorithm and use that 313 is prime.
  rcases h' with ⟨k, hk⟩ | ⟨d, hd⟩,
  { rw hk at h,
    sorry -- k = 1 case to be verified.
  },
  { rw hd at h,
    sorry -- d = 313 leading to n = 302 to be verified.
  }
end

end largest_n_divisible_by_n_plus_11_l395_395899


namespace range_of_a_l395_395690

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → 3 * x - a ≥ 0) → a ≤ 6 :=
by
  intros h
  sorry

end range_of_a_l395_395690


namespace tribe_leadership_organization_l395_395759

theorem tribe_leadership_organization (n : ℕ) (m : ℕ) (k : ℕ) (total : ℕ)
  (h1 : n = 11)  -- total members in the tribe
  (h2 : m = 1)   -- one chief
  (h3 : k = 2)   -- number of supporting chiefs
  (h4 : total = 11 * (Nat.choose 10 2) * (Nat.choose 8 2) * (Nat.choose 6 2)) :
  total = 207900 :=
by {
  rw [h1, h2, h3, h4],
  simp,
  sorry
}

end tribe_leadership_organization_l395_395759


namespace total_insects_eaten_l395_395464

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end total_insects_eaten_l395_395464


namespace solve_for_a_b_l395_395636

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395636


namespace perfume_price_reduction_l395_395495

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end perfume_price_reduction_l395_395495


namespace problem_proof_l395_395657

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395657


namespace AGF_larger_than_GECF_l395_395777

-- Define points, segments, and areas to translate the proof problem.
variable {Point : Type}
variable (A B C D E F G : Point)
variable (AB BC CD DA AE BF : set (Point × Point))

-- Define that ABCD is a square
variable (is_square : square A B C D)

-- Define midpoints
def is_midpoint (P Q R : Point) : Prop := midpoint P Q R

-- Define the areas of triangles and quadrilateral
variable (area : set (Point × Point) → ℝ)

-- Define the relevant conditions
variables (midpoint_E : is_midpoint B C E)
variables (midpoint_F : is_midpoint C D F)
variables (intersection_G : ∃ G, intersect AE BF G)

-- Define the areas of the respective regions
variable (S_1 : ℝ) -- area of triangle AGF
variable (S_2 : ℝ) -- area of quadrilateral GECF
variable (S : ℝ) -- area of square ABCD

-- State the condition S_1 is greater than S_2
theorem AGF_larger_than_GECF : S_1 > S_2 := sorry

end AGF_larger_than_GECF_l395_395777


namespace cube_prism_surface_area_l395_395061

theorem cube_prism_surface_area (a : ℝ) (h : a > 0) :
  2 * (6 * a^2) > 4 * a^2 + 2 * (2 * a * a) :=
by sorry

end cube_prism_surface_area_l395_395061


namespace lines_intersect_on_A_l395_395790

noncomputable def midpoint (A B : Point) : Point := sorry -- Define the midpoint function
noncomputable def symmetric_line (l bisector : Line) : Line := sorry -- Define the symmetric function
noncomputable def orthocenter (A B C : Point) : Point := sorry -- Define the orthocenter function
noncomputable def altitude (A B C : Point) : Line := sorry -- Define the altitude function
noncomputable def intersect (l1 l2 : Line) : Point := sorry -- Function to compute intersection of two lines
noncomputable def on_line (P : Point) (l : Line) : Prop := sorry -- Predicate to check if a point is on a line

-- Define an acute triangle and its properties
variables {A B C : Point} [acute_triangle A B C]

-- Define altitudes and orthocenter
def A' := intersect (altitude A B C) (straight_line B C)
def B' := intersect (altitude B A C) (straight_line A C)
def H := orthocenter A B C

-- Midpoint of the segment AB
def C0 := midpoint A B

-- Symmetric lines
def g := symmetric_line (straight_line (intersect (altitude C A B)) C0) (angle_bisector A C B)
def h := symmetric_line (straight_line H C0) (angle_bisector A H B)

-- The goal is to prove that g and h intersect on the line A'B'
theorem lines_intersect_on_A'B' : on_line (intersect g h) (straight_line A' B') :=
sorry

end lines_intersect_on_A_l395_395790


namespace total_insects_eaten_l395_395468

theorem total_insects_eaten
  (geckos : ℕ)
  (insects_per_gecko : ℕ)
  (lizards : ℕ)
  (multiplier : ℕ)
  (h_geckos : geckos = 5)
  (h_insects_per_gecko : insects_per_gecko = 6)
  (h_lizards : lizards = 3)
  (h_multiplier : multiplier = 2) :
  geckos * insects_per_gecko + lizards * (insects_per_gecko * multiplier) = 66 :=
by
  rw [h_geckos, h_insects_per_gecko, h_lizards, h_multiplier]
  norm_num
  sorry

end total_insects_eaten_l395_395468


namespace find_a_b_l395_395611

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395611


namespace find_a_b_l395_395644

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395644


namespace cookies_left_after_ted_leaves_l395_395171

theorem cookies_left_after_ted_leaves :
  let f : Nat := 2 -- trays per day
  let d : Nat := 6 -- days
  let e_f : Nat := 1 -- cookies eaten per day by Frank
  let t : Nat := 4 -- cookies eaten by Ted
  let c : Nat := 12 -- cookies per tray
  let total_cookies := f * c * d -- total cookies baked
  let cookies_eaten_by_frank := e_f * d -- total cookies eaten by Frank
  let cookies_before_ted := total_cookies - cookies_eaten_by_frank -- cookies before Ted
  let total_cookies_left := cookies_before_ted - t -- cookies left after Ted
  total_cookies_left = 134
:= by
  sorry

end cookies_left_after_ted_leaves_l395_395171


namespace smaller_volume_ratio_l395_395505

noncomputable def volume_ratio_of_smaller_part (r h : ℝ) : ℝ :=
  let V := (1/3) * Math.pi * r^2 * h in
  let V_smaller := (1/3) * Math.pi * (1/3) * r^2 * (h/2) in
  V_smaller / V

theorem smaller_volume_ratio (r h : ℝ) (hr : r = 1) (hh : h > 0) : 
  volume_ratio_of_smaller_part r h = 1 / (3 * Real.sqrt 3) :=
by 
  sorry

end smaller_volume_ratio_l395_395505


namespace chess_player_max_consecutive_win_prob_l395_395931

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l395_395931


namespace maximum_probability_second_game_C_l395_395977

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l395_395977


namespace complex_product_l395_395810

noncomputable def z1 : ℂ := 1 + 2 * complex.i
noncomputable def z2 : ℂ := 2 - complex.i

theorem complex_product : z1 * z2 = 4 + 3 * complex.i :=
by sorry

end complex_product_l395_395810


namespace distance_to_origin_l395_395770

-- Definitions based on the conditions in a)
def imaginary_unit : ℂ := complex.I
def given_complex_number : ℂ := 2 * imaginary_unit / (1 + imaginary_unit)

-- Lean 4 statement of the proof problem
theorem distance_to_origin :
  complex.abs given_complex_number = real.sqrt 2 :=
sorry

end distance_to_origin_l395_395770


namespace discard_one_point_collinear_l395_395069

noncomputable def collinear {P : Type*} [affine_plane P] (s : set P) : Prop :=
∃ (l : line P), s ⊆ l

theorem discard_one_point_collinear {P : Type*} [affine_plane P]
  (n : ℕ) (points : set P) (hn : points.size = n)
  (h : ∀ (s : set P), s.size = 4 → ∃ (p ∈ s), collinear (s \ {p})) :
  ∃ (p ∈ points), collinear (points \ {p}) :=
sorry

end discard_one_point_collinear_l395_395069


namespace length_of_AB_l395_395891

theorem length_of_AB 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 0)) 
  (hB : B = (-1/2, -sqrt 3 / 2)) 
  (hC1 : ∀ P : ℝ × ℝ, P = A ∨ P = B → P.1 ^ 2 + P.2 ^ 2 = 1) 
  (hC2 : ∀ P : ℝ × ℝ, P = A ∨ P = B → P.1 ^ 2 + P.2 ^ 2 - P.1 + sqrt 3 * P.2 = 0) : 
  dist A B = sqrt 3 :=
by
  sorry

end length_of_AB_l395_395891


namespace brick_wall_total_bricks_l395_395980

theorem brick_wall_total_bricks (x : ℝ) 
  (h1 : 1 / 12 > 0) 
  (h2 : 1 / 15 > 0) 
  (combined_rate : ∀ (b1_rate b2_rate : ℝ), b1_rate + b2_rate - 15)
  (work_complete_time : ∀ (time : ℝ), time = 6) : 
  (1 / 12 * x) + (1 / 15 * x) - 15 ≤ x / 6 → 
  x = 900 := 
by 
  sorry

end brick_wall_total_bricks_l395_395980


namespace A_l395_395808

noncomputable def length_A'B' : ℝ :=
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (0, 14)
  let C : ℝ × ℝ := (3, 6)
  let line_y_eq_x := λ (p : ℝ × ℝ), p.1 = p.2
  let A' : ℝ × ℝ := (12, 12)
  let B' : ℝ × ℝ := (42 / 11, 42 / 11) in
  real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2)

theorem A'B'_length_correct : length_A'B' = 90 * real.sqrt 2 / 11 :=
by {
  sorry
}

end A_l395_395808


namespace find_a_b_l395_395606

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395606


namespace problem_statement_l395_395744

/-
If x is equal to the sum of the even integers from 40 to 60 inclusive,
y is the number of even integers from 40 to 60 inclusive,
and z is the sum of the odd integers from 41 to 59 inclusive,
prove that x + y + z = 1061.
-/
theorem problem_statement :
  let x := (11 / 2) * (40 + 60)
  let y := 11
  let z := (10 / 2) * (41 + 59)
  x + y + z = 1061 :=
by
  sorry

end problem_statement_l395_395744


namespace find_a_b_l395_395641

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395641


namespace swapped_two_digit_number_l395_395217

variable (a : ℕ)

theorem swapped_two_digit_number (h : a < 10) (sum_digits : ∃ t : ℕ, t + a = 13) : 
    ∃ n : ℕ, n = 9 * a + 13 :=
by
  sorry

end swapped_two_digit_number_l395_395217


namespace quadrilateral_is_kite_l395_395994

noncomputable def quadrilateral_circumscribes_circle (Q : Type) [quadd Q] : Prop := sorry

noncomputable def diagonal_bisects_other (Q : Type) [quadDiag Q] : Prop := sorry

theorem quadrilateral_is_kite (Q : Type) [quadrilateral Q] 
  (circumscribes : quadrilateral_circumscribes_circle Q)
  (bisects : diagonal_bisects_other Q) : is_kite Q :=
sorry

end quadrilateral_is_kite_l395_395994


namespace train_crossing_time_l395_395227

theorem train_crossing_time
  (train_length : ℕ)
  (train_speed_kmph : ℕ)
  (bridge_length : ℕ)
  (h_train_length : train_length = 150)
  (h_train_speed : train_speed_kmph = 80)
  (h_bridge_length : bridge_length = 250) :
  let total_distance := train_length + bridge_length
  let speed_mps := (train_speed_kmph * 1000) / 3600
  let crossing_time := total_distance / speed_mps in
  crossing_time ≈ 18 := 
by
  sorry

end train_crossing_time_l395_395227


namespace find_a_b_l395_395623

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395623


namespace max_lateral_surface_area_of_tetrahedron_l395_395211

open Real

theorem max_lateral_surface_area_of_tetrahedron :
  ∀ (PA PB PC : ℝ), (PA^2 + PB^2 + PC^2 = 36) → (PA * PB + PB * PC + PA * PC ≤ 36) →
  (1/2 * (PA * PB + PB * PC + PA * PC) ≤ 18) :=
by
  intro PA PB PC hsum hineq
  sorry

end max_lateral_surface_area_of_tetrahedron_l395_395211


namespace divisors_18_product_and_sum_l395_395121

theorem divisors_18_product_and_sum :
  let divisors := {d : ℕ | d > 0 ∧ 18 % d = 0}
  ∃ (product : ℕ), (∀ d ∈ divisors, d * (18 / d) = 18) ∧ ((∏ d in divisors, d) = 5832) ∧
  ∃ (sum : ℕ), ((∑ d in divisors, d) = 39) :=
by
  let divisors := {d : ℕ | d > 0 ∧ 18 % d = 0}
  use 5832
  use 39
  sorry

end divisors_18_product_and_sum_l395_395121


namespace sum_difference_even_odd_3000_l395_395052

theorem sum_difference_even_odd_3000 : 
  let S_even := (3000 / 2) * (2 + (2 + (3000 - 1) * 2)),
      S_odd := (3000 / 2) * (3 + (3 + (3000 - 1) * 2)) in
  S_even - S_odd = -3000 :=
by
  let S_even := 1500 * 6000
  let S_odd := 1500 * 6002
  show S_even - S_odd = -3000
  sorry

end sum_difference_even_odd_3000_l395_395052


namespace find_a_b_l395_395676

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395676


namespace maximize_probability_when_C_second_game_l395_395957

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l395_395957


namespace bicycle_cost_calculation_l395_395102

theorem bicycle_cost_calculation 
  (CP_A CP_B CP_C : ℝ)
  (h1 : CP_B = 1.20 * CP_A)
  (h2 : CP_C = 1.25 * CP_B)
  (h3 : CP_C = 225) :
  CP_A = 150 :=
by
  sorry

end bicycle_cost_calculation_l395_395102


namespace domain_of_f_l395_395896

-- Define the function
def f (t : ℝ) : ℝ := 1 / ((t - 1) * (t + 1) + (t - 2)^2)

-- Prove that the function is defined for all real numbers
theorem domain_of_f : ∀ t : ℝ, (t - 1) * (t + 1) + (t - 2)^2 ≠ 0 := 
begin
  intro t,
  have h : (t - 1) * (t + 1) + (t - 2)^2 = 2 * t^2 - 4 * t + 3,
  {
    calc
    (t - 1) * (t + 1) + (t - 2)^2 = t^2 - 1 + t^2 - 4 * t + 4 : by ring
    ... = 2 * t^2 - 4 * t + 3 : by ring
  },
  rw h,
  -- A quadratic 2t^2 - 4t + 3 has no real roots and is always positive.
  -- Since this is the focus, include the corresponding lemma for clarity.
  sorry
end

end domain_of_f_l395_395896


namespace maximum_probability_second_game_C_l395_395973

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l395_395973


namespace log_expression_equality_l395_395570

noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

theorem log_expression_equality :
  (lg (4 * real.sqrt 2 / 7) - lg (2 / 3) + lg (7 * real.sqrt 5)) = (lg 6 + 1 / 2) :=
by
  sorry

end log_expression_equality_l395_395570


namespace problem_proof_l395_395654

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395654


namespace base9_sum_l395_395522

-- Definitions for base 9 numbers
def base9_256 : ℕ := 2 * 9^2 + 5 * 9^1 + 6 * 9^0
def base9_832 : ℕ := 8 * 9^2 + 3 * 9^1 + 2 * 9^0
def base9_74 : ℕ := 7 * 9^1 + 4 * 9^0

-- Sum of the numbers in base 10
def sum_base10 : ℕ := base9_256 + base9_832 + base9_74

-- Function to convert base 10 number to base 9 representation
def base10_to_base9 (n : ℕ) : ℕ :=
  let rec convert (num : ℕ) (acc : ℕ) (pow : ℕ) : ℕ :=
    if num = 0 then acc
    else convert (num / 9) (acc + (num % 9) * pow) (pow * 10)
  in convert n 0 1

-- Prove that the sum of 256_9, 832_9, and 74_9 in base 9 is 1273_9
theorem base9_sum :
  base10_to_base9 sum_base10 = 1273 := by
  sorry

end base9_sum_l395_395522


namespace non_congruent_triangle_classes_count_l395_395697

theorem non_congruent_triangle_classes_count :
  ∃ (q : ℕ),
  (∀ (a b : ℕ), a ≤ 2 ∧ 2 ≤ b ∧ a + 2 > b ∧ a + b > 2 ∧ b + 2 > a → 
  (a = 1 ∧ b = 2 ∨ a = 2 ∧ (b = 2 ∨ b = 3))) ∧ q = 3 := 
by
  use 3
  intro a b
  rintro ⟨ha, hb, h1, h2, h3⟩
  split
  { intro h,
    cases h with h1 h2,
    { exact or.inl ⟨h1, hb.eq_of_le⟩ },
    cases h2 with h2 h3,
    { exact or.inr ⟨hb.eq_of_le, or.inl rfl⟩ },
    { exact or.inr ⟨hb.eq_of_le, or.inr rfl⟩ } },
  { rintro (⟨ha1, rfl⟩ | ⟨rfl, hb1⟩),
    { refine ⟨le_refl _, le_add_of_lt hb, _, _, _⟩,
      { linarith },
      { linarith },
      { linarith } },
    cases hb1 with rfl rfl,
    { refine ⟨le_refl _, le_refl _, _, _, _⟩;
      linarith },
    { refine ⟨le_refl _, le_add_of_lt _, _, _, _⟩,
      { exact nat.zero_lt_one.trans one_lt_two },
      { linarith },
      { linarith },
      { linarith } } }
  sorry

end non_congruent_triangle_classes_count_l395_395697


namespace david_is_30_l395_395424

-- Definitions representing the conditions
def uncleBobAge : ℕ := 60
def emilyAge : ℕ := (2 * uncleBobAge) / 3
def davidAge : ℕ := emilyAge - 10

-- Statement that represents the equivalence to be proven
theorem david_is_30 : davidAge = 30 :=
by
  sorry

end david_is_30_l395_395424


namespace spanning_tree_with_exact_colors_l395_395795

variables (Γ : Type*) [graph : Graph Γ] [SpanningTree Γ]
variables (r g b : ℕ)

theorem spanning_tree_with_exact_colors {r g b : ℕ} {Γ : Graph}
  (h1 : 0 ≤ r)
  (h2 : 0 ≤ g)
  (h3 : 0 ≤ b)
  (h4 : Vertices Γ = r + g + b + 1)
  (h5 : ∃ T1 : SpanningTree Γ, (∃ T2 : SpanningTree Γ, (∃ T3 : SpanningTree Γ,
    red_edges T1 = r ∧ green_edges T2 = g ∧ blue_edges T3 = b)))
  : ∃ T : SpanningTree Γ, red_edges T = r ∧ green_edges T = g ∧ blue_edges T = b :=
sorry

end spanning_tree_with_exact_colors_l395_395795


namespace seashells_solution_l395_395035

def seashells_problem (T : ℕ) : Prop :=
  T + 13 = 50 → T = 37

theorem seashells_solution : seashells_problem 37 :=
by
  intro h
  sorry

end seashells_solution_l395_395035


namespace union_sets_is_correct_l395_395720

def M := {x : ℝ | x^2 - 4*x < 0}
def N := {x : ℝ | |x| ≤ 2}
def union_set := M ∪ N

theorem union_sets_is_correct : union_set = Set.Ico (-2 : ℝ) 4 :=
by
  intro x
  sorry

end union_sets_is_correct_l395_395720


namespace train_car_count_estimate_l395_395178

theorem train_car_count_estimate (t_pass_seconds : ℕ) (cars_count_during_period : ℕ) (time_period_seconds : ℕ) (total_pass_time_seconds : ℕ) :
  cars_count_during_period = 8 →
  time_period_seconds = 12 →
  total_pass_time_seconds = 210 →
  t_pass_seconds = (total_pass_time_seconds * cars_count_during_period) / time_period_seconds →
  t_pass_seconds = 140 := by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  have h5 : t_pass_seconds = (210 * 8) / 12, by sorry
  have h6 : t_pass_seconds = 140, by sorry
  exact h6

end train_car_count_estimate_l395_395178


namespace pair_divisibility_l395_395559

theorem pair_divisibility (m n : ℕ) : 
  (m * n ∣ m ^ 2019 + n) ↔ ((m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2 ^ 2019)) := sorry

end pair_divisibility_l395_395559


namespace smallest_integer_to_perfect_cube_l395_395058

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem smallest_integer_to_perfect_cube :
  ∃ n : ℕ, 
    n > 0 ∧ 
    is_perfect_cube (45216 * n) ∧ 
    (∀ m : ℕ, m > 0 ∧ is_perfect_cube (45216 * m) → n ≤ m) ∧ 
    n = 7 := sorry

end smallest_integer_to_perfect_cube_l395_395058


namespace ellipse_equation_and_distance_distance_sum_l395_395194

-- Define the problem conditions
variable {a b : ℝ} (ha : a > 0) (hb : b > 0) (h1 : a > b)  
variable (ecc : ℝ) (ecc_eq : ecc = (Real.sqrt 2) / 2)
variable (d : ℝ) (d_eq : d = 2 / (Real.sqrt 3))
variable {x y : ℝ} {x1 y1 x2 y2 : ℝ}
variable (P_eq : x^2 / 4 + y^2 / 2 = 1) (M_eq : x1^2 / 4 + y1^2 / 2 = 1)
variable (N_eq : x2^2 / 4 + y2^2 / 2 = 1)
variable (OP_EQ : x = λ x1 + 2 * μ x2) (oy_eq : y = λ y1 + 2 * μ y2)
variable (slope_eq : (y1 / x1) * (y2 / x2) = -1/2)
variable (E1 E2 : (ℝ × ℝ)) (he1 : E1 = (-Real.sqrt 3 / 2, 0))
variable (he2 : E2 = (Real.sqrt 3 / 2, 0))

-- Define the claim statement
theorem ellipse_equation_and_distance : 
  ∃ (a : ℝ) (b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧ (Real.sqrt (a^2 - b^2) / a = (Real.sqrt 2) / 2) ∧ 
  (a * b / Real.sqrt (a^2 + b^2) = 2 / (Real.sqrt 3)) ∧ 
  (λ x1 + 2 * μ x2)^2 / 4 + (λ y1 + 2 * μ y2)^2 / 2 = 1 :=
sorry

theorem distance_sum :
  ∃ (a b : ℝ), (λ^2 + 4*μ^2 = 1) ∧ |(λ, μ) - E1| + |(λ, μ) - E2| = 2 :=
sorry

end ellipse_equation_and_distance_distance_sum_l395_395194


namespace proved_value_l395_395208

-- Define the variables and conditions
variables {x y z : ℝ}

-- 3x, 4y, 5z form a geometric sequence
def geometric_sequence : Prop := (4 * y / 3 * x) = (5 * z / 4 * y)

-- 1/x, 1/y, 1/z form an arithmetic sequence
def arithmetic_sequence : Prop := (2 / y) = (1 / x) + (1 / z)

-- The theorem to prove
theorem proved_value (h1 : geometric_sequence) (h2 : arithmetic_sequence) : x ≠ 0 → z ≠ 0 →
  x / z + z / x = 34 / 15 :=
sorry

end proved_value_l395_395208


namespace ABCD_concyclic_l395_395821

variables {A B C D I1 I2 E F P : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space I1] [metric_space I2] [metric_space E] [metric_space F] [metric_space P]

-- Conditions
def convex (Q : Type) : Prop := sorry -- definition for convex quadrilateral
def incenter (T : Type) : Type := sorry -- definition for incenter of a triangle
def intersect_at (L1 L2 : Type) (P : Type) : Prop := sorry -- definition for lines intersecting at a point
def equal_length (L1 L2 : Type) : Prop := sorry -- definition for equal length segments

-- Given
axiom quadrilateral_ABCD_convex : convex ABCD
axiom I1_incenter_ABC : incenter ABC = I1
axiom I2_incenter_DBC : incenter DBC = I2
axiom E_intersects_AB : intersect_at (line_through I1) AB E
axiom F_intersects_DC : intersect_at (line_through I2) DC F
axiom P_intersection_of_AB_and_DC : intersect_at (extend_line AB) (extend_line DC) P
axiom PE_equal_PF : equal_length PE PF

-- To Prove
theorem ABCD_concyclic : cyclic ABCD :=
sorry

end ABCD_concyclic_l395_395821


namespace gcd_36_48_72_l395_395053

theorem gcd_36_48_72 : Int.gcd (Int.gcd 36 48) 72 = 12 := by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 48 = 2^4 * 3 := by norm_num
  have h3 : 72 = 2^3 * 3^2 := by norm_num
  sorry

end gcd_36_48_72_l395_395053


namespace circle_eq_of_diameter_l395_395721

theorem circle_eq_of_diameter {P Q : ℝ × ℝ}
  (hP : P = (3, 4)) (hQ : Q = (-5, 6)) :
  ∃ (h k r : ℝ), (h = -1) ∧ (k = 5) ∧ (r = √17) ∧ ((x + h)^2 + (y - k)^2 = r^2) :=
begin
  -- P = (3,4) and Q = (-5,6)
  rcases hP with ⟨Px, Py⟩,
  rcases hQ with ⟨Qx, Qy⟩,
  -- center = midpoint of P and Q
  let h := (3 + -5)/2,
  let k := (4 + 6)/2,
  have hc : h = -1 := by norm_num,
  have kc : k = 5 := by norm_num,
  -- radius = half the distance between P and Q
  let r := √(17),
  have r_is : r = ⟦√17⟧ := by norm_num,
  use ⟨h, k, r⟩,
  split; [exact hc| split; [exact kc| split; [exact r_is]]],
  sorry
end

end circle_eq_of_diameter_l395_395721


namespace meeting_distance_l395_395893

theorem meeting_distance (t : ℕ) (h : 0.375 * t ^ 2 + 7.625 * t = 100) : 4.5 * t = 45 :=
by
  sorry

end meeting_distance_l395_395893


namespace solve_for_a_b_l395_395631

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395631


namespace course_gender_relationship_l395_395041

-- Define given values
def sample_size : ℕ := 100
def boys_calligraphy : ℕ := 40
def boys_paper_cutting : ℕ := 10
def girls_calligraphy : ℕ := 30
def girls_paper_cutting : ℕ := 20
def total_boys : ℕ := boys_calligraphy + boys_paper_cutting
def total_girls : ℕ := girls_calligraphy + girls_paper_cutting
def total_calligraphy : ℕ := boys_calligraphy + girls_calligraphy
def total_paper_cutting : ℕ := boys_paper_cutting + girls_paper_cutting
def total_students : ℕ := total_boys + total_girls

-- Define the chi-squared statistic
def chi_squared_statistic : ℝ := (sample_size * (boys_calligraphy * girls_paper_cutting - boys_paper_cutting * girls_calligraphy) ^ 2) / (total_boys * total_girls * total_calligraphy * total_paper_cutting)

-- The critical value for 95% confidence
def critical_value_95 : ℝ := 3.841

-- Statement to prove
theorem course_gender_relationship : chi_squared_statistic > critical_value_95 :=
by
  sorry

end course_gender_relationship_l395_395041


namespace find_smallest_even_number_l395_395865

theorem find_smallest_even_number (n : ℕ) (h : n + (n + 2) + (n + 4) = 162) : n = 52 :=
by
  sorry

end find_smallest_even_number_l395_395865


namespace range_of_x_l395_395383

noncomputable def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

noncomputable def condition (f g : ℝ → ℝ) (h : ∀ x, deriv f x = g x) := 
  (odd_function f) ∧ (f 2 = 0) ∧ (∀ x > 0, x * g x - f x < 0)

theorem range_of_x (f g : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = g x)
  (h_cond : condition f g h_deriv) :
  ∀ x, f x < 0 ↔ x ∈ Set.Ioc (-2) 0 ∪ Set.Ioc 2 (real.to_nnreal ∞) := sorry

end range_of_x_l395_395383


namespace range_of_m_l395_395580

variable (x m : ℝ)

def p := |x - 4| ≤ 6
def q := x^2 - 2 * x + 1 - m^2 ≤ 0

theorem range_of_m (h1 : ¬p → ¬q) (h2 : p → q) : -3 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l395_395580


namespace part1_part2_l395_395920

-- Part 1: Inequality solution
theorem part1 (x : ℝ) :
  (1 / 3 * x - (3 * x + 4) / 6 ≤ 2 / 3) → (x ≥ -8) := 
by
  intro h
  sorry

-- Part 2: System of inequalities solution
theorem part2 (x : ℝ) :
  (4 * (x + 1) ≤ 7 * x + 13) ∧ ((x + 2) / 3 - x / 2 > 1) → (-3 ≤ x ∧ x < -2) := 
by
  intro h
  sorry

end part1_part2_l395_395920


namespace max_minus_min_value_l395_395246

def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_minus_min_value (a : ℝ) : 
  let M := max (f 0 a) (max (f 1 a) (f 3 a))
  let N := min (f 0 a) (min (f 1 a) (f 3 a))
  M - N = 18 :=
by
  -- Definitions of f and the values at points 0, 1, 3
  let f := λ x a : ℝ, x^3 - 3 * x - a
  -- Calculations are based on the critical points and endpoints
  have f_0 : f 0 a = -a := sorry
  have f_1 : f 1 a = -2 - a := sorry
  have f_3 : f 3 a = 18 - a := sorry
  -- Determine max and min values within [0,3]
  let M := max (f 0 a) (max (f 1 a) (f 3 a))
  let N := min (f 0 a) (min (f 1 a) (f 3 a))
  -- Because the question, we just calculate the difference
  -- Sorry is added to skip the proof
  have M_minus_N : M - N = 18 := sorry
  exact M_minus_N

end max_minus_min_value_l395_395246


namespace area_bounded_by_equation_l395_395011

theorem area_bounded_by_equation : 
  (∃ (bounded_region : ℝ) (y x : ℝ), y^2 + 2 * x * y + 50 * |x| = 500) → 
  (area_of_bounded_region bounded_region = 1250) := 
by
  sorry

end area_bounded_by_equation_l395_395011


namespace candidate_judge_agreement_l395_395254

-- Given conditions
variables (m n k : ℕ)
variable (h_n_ge_3 : n ≥ 3)
variable (h_n_odd : n % 2 = 1)
variable (k : ℕ)
variable (h_max_agreement : ∀ (i j : ℕ), i ≠ j → i < n → j < n → (∀ (a : ℕ), a < m → ∃! (x : Bool), ((a , i , x) = (a , i, true) ∨ (a , i , false)) → ((a , j , x) = (a , j , true) ∨ (a , j , false))) → x ≤ k)

-- Goal to prove
theorem candidate_judge_agreement (h_total_candidates : m ≠ 0) : (k : ℝ) / (m : ℝ) ≥ (n - 1 : ℝ) / (2 * n) :=
sorry

end candidate_judge_agreement_l395_395254


namespace bob_makes_weekly_profit_l395_395531

def weekly_profit (p_cost p_sell : ℝ) (m_daily d_week : ℕ) : ℝ :=
  (p_sell - p_cost) * m_daily * (d_week : ℝ)

theorem bob_makes_weekly_profit :
  weekly_profit 0.75 1.5 12 7 = 63 := 
by
  sorry

end bob_makes_weekly_profit_l395_395531


namespace problem_proof_l395_395663

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395663


namespace trig_problem_l395_395181

variable (α : ℝ)

theorem trig_problem
  (h1 : Real.sin (Real.pi + α) = -1 / 3) :
  Real.cos (α - 3 * Real.pi / 2) = -1 / 3 ∧
  (Real.sin (Real.pi / 2 + α) = 2 * Real.sqrt 2 / 3 ∨ Real.sin (Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3) ∧
  (Real.tan (5 * Real.pi - α) = -Real.sqrt 2 / 4 ∨ Real.tan (5 * Real.pi - α) = Real.sqrt 2 / 4) :=
sorry

end trig_problem_l395_395181


namespace area_ratio_trapezoid_abm_abcd_l395_395778

-- Definitions based on conditions
variables {A B C D M : Type} [Zero A] [Zero B] [Zero C] [Zero D] [Zero M]
variables (BC AD : ℝ)

-- Condition: ABCD is a trapezoid with BC parallel to AD and diagonals AC and BD intersect M
-- Given BC = b and AD = a

-- Theorem statement
theorem area_ratio_trapezoid_abm_abcd (a b : ℝ) (h1 : BC = b) (h2 : AD = a) : 
  ∃ S_ABM S_ABCD : ℝ,
  (S_ABM / S_ABCD = a * b / (a + b)^2) :=
sorry

end area_ratio_trapezoid_abm_abcd_l395_395778


namespace exists_partition_l395_395308

def M_k (k : ℤ) : Set ℤ := { m | 2 * k ^ 2 + k ≤ m ∧ m ≤ 2 * k ^ 2 + 3 * k }

theorem exists_partition (k : ℤ) :
  ∃ (A B : Set ℤ), A ⊆ M_k k ∧ B ⊆ M_k k ∧ A ∩ B = ∅ ∧ A ∪ B = M_k k ∧
  (∑ x in A, x^2) = ∑ x in B, x^2 := 
sorry

end exists_partition_l395_395308


namespace find_a_b_l395_395673

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395673


namespace minimum_value_f_l395_395165

noncomputable def f (a b c : ℝ) : ℝ :=
  a / (Real.sqrt (a^2 + 8*b*c)) + b / (Real.sqrt (b^2 + 8*a*c)) + c / (Real.sqrt (c^2 + 8*a*b))

theorem minimum_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  1 ≤ f a b c := by
  sorry

end minimum_value_f_l395_395165


namespace percentage_died_by_bombardment_l395_395476

noncomputable def initial_population : ℕ := 8515
noncomputable def final_population : ℕ := 6514

theorem percentage_died_by_bombardment :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 100) ∧
  8515 - ((x / 100) * 8515) - (15 / 100) * (8515 - ((x / 100) * 8515)) = 6514 ∧
  x = 10 :=
by
  sorry

end percentage_died_by_bombardment_l395_395476


namespace line_equations_l395_395561

theorem line_equations (x y : ℝ) 
  (h1 : (-4, 0) ∈ {p : ℝ × ℝ | p.2 = (Real.sin (Real.arctan (1 / 3))) * (p.1 + 4)}) 
  (h2 : (-2, 1) ∈ {p : ℝ × ℝ | p.2 = 0 ∨ 2 * |3 / 4| / Real.sqrt((3 / 4)^2 + 1) = 2}) : 
  (x + 2 = 0 ∨ 3*x - 4*y + 10 = 0) :=
by {
  sorry
}

end line_equations_l395_395561


namespace least_squares_solution_l395_395887

-- Define x_k and y_k arrays according to the conditions provided
def x_values : List Float := [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
def y_values : List Float := [6.01, 5.07, 4.30, 3.56, 3.07, 2.87, 2.18, 2.00, 2.14]

-- Define the sum of squared residuals function
def S (a1 a2 a3 : Float) : Float :=
  x_values.zip y_values
    |>.foldl (fun acc (p : Float × Float) => acc + (p.2 - (a1 * p.1 * p.1 + a2 * p.1 + a3))^2) 0

-- Define partial derivatives of S w.r.t a1, a2, a3
def dS_da1 (a1 a2 a3 : Float) : Float :=
  -2 * x_values.zip y_values
    |>.foldl (fun acc (p : Float × Float) => acc + (p.2 - (a1 * p.1 * p.1 + a2 * p.1 + a3)) * p.1 * p.1) 0

def dS_da2 (a1 a2 a3 : Float) : Float :=
  -2 * x_values.zip y_values
    |>.foldl (fun acc (p : Float × Float) => acc + (p.2 - (a1 * p.1 * p.1 + a2 * p.1 + a3)) * p.1) 0

def dS_da3 (a1 a2 a3 : Float) : Float :=
  -2 * x_values.zip y_values
    |>.foldl (fun acc (p : Float × Float) => acc + (p.2 - (a1 * p.1 * p.1 + a2 * p.1 + a3))) 0

-- Define the final parameters using derived sums in the solution
def a1_correct := 0.95586
def a2_correct := -1.9733
def a3_correct := 3.0684

theorem least_squares_solution :
  ∃ (a1 a2 a3 : Float),
    dS_da1 a1 a2 a3 = 0 ∧
    dS_da2 a1 a2 a3 = 0 ∧
    dS_da3 a1 a2 a3 = 0 ∧
    a1 = a1_correct ∧
    a2 = a2_correct ∧
    a3 = a3_correct :=
by
  sorry

end least_squares_solution_l395_395887


namespace set_equal_l395_395719

variable (M : Set ℝ) (P : Set ℝ)

noncomputable def setM : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}
noncomputable def setP : Set ℝ := {x | ∃ b : ℝ, x = b^2 - 4b + 5}

theorem set_equal : setM = setP := sorry

end set_equal_l395_395719


namespace must_be_three_among_integers_l395_395255

theorem must_be_three_among_integers (a b c d : ℕ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (hsums : ∃ (sums : finset ℕ), sums = {a+b, a+c, a+d, b+c, b+d, c+d} ∧ sums = {5, 6, 7, 8}) :
    a = 3 ∨ b = 3 ∨ c = 3 ∨ d = 3 := 
sorry

end must_be_three_among_integers_l395_395255


namespace original_number_of_men_l395_395488

theorem original_number_of_men 
  (x : ℕ) 
  (h1 : 17 * x > 0) 
  (h2 : 21 * (x - 8) > 0) 
  (h3 : 1 / (17 * x) = 1 / (21 * (x - 8))) :
  x = 42 :=
begin
  sorry
end

end original_number_of_men_l395_395488


namespace floor_log_sum_l395_395572

theorem floor_log_sum : (∑ n in Finset.range 2013.succ, Int.floor (Real.log10 (n + 1))) = 4932 := by
  sorry

end floor_log_sum_l395_395572


namespace problem1_problem2_l395_395718

-- Definitions for sets A and S
def setA (x : ℝ) : Prop := -7 ≤ 2 * x - 5 ∧ 2 * x - 5 ≤ 9
def setS (x k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

-- Preliminary ranges for x
lemma range_A : ∀ x, setA x ↔ -1 ≤ x ∧ x ≤ 7 := sorry

noncomputable def k_range1 (k : ℝ) : Prop := 2 ≤ k ∧ k ≤ 4
noncomputable def k_range2 (k : ℝ) : Prop := k < 2 ∨ k > 6

-- Proof problems in Lean 4

-- First problem statement
theorem problem1 (k : ℝ) : (∀ x, setS x k → setA x) ∧ (∃ x, setS x k) → k_range1 k := sorry

-- Second problem statement
theorem problem2 (k : ℝ) : (∀ x, ¬(setA x ∧ setS x k)) → k_range2 k := sorry

end problem1_problem2_l395_395718


namespace distance_from_P_to_plane_ATQ_l395_395260

-- Define the coordinates for the cube vertices
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def A' : ℝ × ℝ × ℝ := (0, 0, 1)
def B' : ℝ × ℝ × ℝ := (1, 1, 0)
def C' : ℝ × ℝ × ℝ := (1, 1, 1)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (1, 1, 1)
def D : ℝ × ℝ × ℝ := (0, 1, 0)
def D' : ℝ × ℝ × ℝ := (0, 1, 1)

-- Define the centers of the faces
def center (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1 + p4.1) / 4, (p1.2 + p2.2 + p3.2 + p4.2) / 4, (p1.3 + p2.3 + p3.3) / 4)

def T : ℝ × ℝ × ℝ := center A A' B' B
def P : ℝ × ℝ × ℝ := center A' B' C' D'
def Q : ℝ × ℝ × ℝ := center B B' C' C

-- Define the function to compute the distance from a point to a plane
def distance_point_plane (p a b c : ℝ × ℝ × ℝ) : ℝ :=
  let nx := (b.2 - a.2) * (c.3 - a.3) - (b.3 - a.3) * (c.2 - a.2)
  let ny := (b.3 - a.3) * (c.1 - a.1) - (b.1 - a.1) * (c.3 - a.3)
  let nz := (b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)
  let d := nx * a.1 + ny * a.2 + nz * a.3
  (nx * p.1 + ny * p.2 + nz * p.3 - d).abs / Math.sqrt (nx * nx + ny * ny + nz * nz)

noncomputable def distance : ℝ := distance_point_plane P A T Q

-- The problem statement in Lean 4
theorem distance_from_P_to_plane_ATQ : distance = (Real.sqrt 3 / 3) :=
by
  -- Placeholder for the actual proof
  sorry

end distance_from_P_to_plane_ATQ_l395_395260


namespace irrational_multiple_contains_0_or_9_infinitely_often_l395_395279

theorem irrational_multiple_contains_0_or_9_infinitely_often (α : ℝ) (h_irrational : irrational α) : 
  ∃ (n : ℕ), ∃ (m : ℕ), (∀k : ℕ, (fractional_part (k * α)).decimalDigits = 0 ∨ 
  (fractional_part (k * α)).decimalDigits = 9) := sorry

end irrational_multiple_contains_0_or_9_infinitely_often_l395_395279


namespace symmetric_group_elements_as_2_cycles_l395_395273

theorem symmetric_group_elements_as_2_cycles (n : ℕ) (σ : Equiv.Perm (Fin n)) :
  ∃ l : List (Equiv.Perm (Fin n)), (∀ τ ∈ l, ∃ a b, a ≠ b ∧ τ = Equiv.Perm.swap a b) ∧ l.prod = σ :=
sorry

end symmetric_group_elements_as_2_cycles_l395_395273


namespace min_value_of_squares_l395_395800

theorem min_value_of_squares (a b s t : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 :=
sorry

end min_value_of_squares_l395_395800


namespace find_vector_v_l395_395798

def vector_a : ℝ^3 := ⟨2, 3, 0⟩
def vector_b : ℝ^3 := ⟨1, 0, -2⟩

noncomputable def cross_product (v w : ℝ^3) : ℝ^3 :=
⟨v.y * w.z - v.z * w.y, 
 v.z * w.x - v.x * w.z, 
 v.x * w.y - v.y * w.x⟩

theorem find_vector_v (v : ℝ^3) : 
  cross_product v vector_a = 2 * cross_product vector_b vector_a ∧ 
  cross_product v vector_b = 3 * cross_product vector_a vector_b → 
  v = ⟨8, 9, -4⟩ :=
by 
  sorry

end find_vector_v_l395_395798


namespace amy_money_left_l395_395116

def amount_left (initial_amount doll_price board_game_price comic_book_price doll_qty board_game_qty comic_book_qty board_game_discount sales_tax_rate : ℝ) :
    ℝ :=
  let cost_dolls := doll_qty * doll_price
  let cost_board_games := board_game_qty * board_game_price
  let cost_comic_books := comic_book_qty * comic_book_price
  let discounted_cost_board_games := cost_board_games * (1 - board_game_discount)
  let total_cost_before_tax := cost_dolls + discounted_cost_board_games + cost_comic_books
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  initial_amount - total_cost_after_tax

theorem amy_money_left :
  amount_left 100 1.25 12.75 3.50 3 2 4 0.10 0.08 = 56.04 :=
by
  sorry

end amy_money_left_l395_395116


namespace Perimeter_PQR_leq_half_Perimeter_ABC_l395_395849
  
variable {α : Type}
variable (A B C : α)

-- Assume triangle and its properties are adequately defined in the context
axiom triangle_ABC : Triangle α
axiom angle_bisectors_intersect_D_E_F : IntersectAngleBisectors α triangle_ABC A B C

-- Definition of points and corresponding lengths to use for perimeter calculatons
noncomputable def perimeter_triangle_PQR : ℝ :=
  sorry -- Assuming calculation based on lengths from P, Q, R positions

noncomputable def perimeter_triangle_ABC : ℝ :=
  sorry -- Calculation using lengths of sides A, B, C

/-- Prove that the perimeter of triangle PQR is at most half the perimeter of triangle ABC -/
theorem Perimeter_PQR_leq_half_Perimeter_ABC :
  perimeter_triangle_PQR A B C ≤ 0.5 * perimeter_triangle_ABC A B C :=
sorry

end Perimeter_PQR_leq_half_Perimeter_ABC_l395_395849


namespace find_larger_number_l395_395562

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := 
by 
  sorry

end find_larger_number_l395_395562


namespace base_of_first_triangle_l395_395375

theorem base_of_first_triangle 
  (h₁ : ℕ) (h₂ : ℕ)
  (b₂ : ℕ) 
  (double_area : ℕ → ℕ) : 
  (2 * (1/2 * b₂ * h₂ : ℝ) = double_area (1/2 * (b₂ : ℝ) * h₁)) → 
  (double_area (1/2 * (b₂ : ℝ) * h₁) = 30) :=
by {
  sorry
}

end base_of_first_triangle_l395_395375


namespace price_difference_is_correct_l395_395491

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end price_difference_is_correct_l395_395491


namespace decreasing_interval_of_f_l395_395002

def f (x : ℝ) : ℝ := x^3 - 3 * x

def f' (x : ℝ) : ℝ := 3 * x^2 - 3

theorem decreasing_interval_of_f : ∀ x : ℝ, f'(x) < 0 ↔ -1 < x ∧ x < 1 :=
by
  intros x
  sorry

end decreasing_interval_of_f_l395_395002


namespace square_side_length_l395_395508

theorem square_side_length (a b : ℕ) (h : a = 9) (h' : b = 16) (A : ℕ) (h1: A = a * b) :
  ∃ (s : ℕ), s * s = A ∧ s = 12 :=
by
  sorry

end square_side_length_l395_395508


namespace quadrilateral_area_eq_l395_395507

noncomputable def area_of_quadrilateral (A B C D : (ℝ × ℝ)) : ℝ :=
  let triangle_area := λ (P Q R : (ℝ × ℝ)), (1 / 2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))
  (triangle_area A B C) + (triangle_area A C D)

theorem quadrilateral_area_eq :
  area_of_quadrilateral (0, 0) (0, 2) (3, 2) (5, 5) = 5.5 :=
by
  sorry

end quadrilateral_area_eq_l395_395507


namespace ratio_of_second_to_first_show_l395_395786

-- Definitions based on conditions
def first_show_length : ℕ := 30
def total_show_time : ℕ := 150
def second_show_length := total_show_time - first_show_length

-- Proof problem in Lean 4 statement
theorem ratio_of_second_to_first_show : 
  (second_show_length / first_show_length) = 4 := by
  sorry

end ratio_of_second_to_first_show_l395_395786


namespace quadratic_function_min_value_l395_395025

theorem quadratic_function_min_value :
  ∀ x : ℝ, x^2 - 2 * x + 3 ≥ 2 := 
by {
  sorry
}

example : (1 : ℝ)^2 - 2 * (1 : ℝ) + 3 = 2 :=
by {
  norm_num,
}

end quadratic_function_min_value_l395_395025


namespace find_a_b_l395_395586

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395586


namespace finalCostCalculation_l395_395038

-- Define the inputs
def tireRepairCost : ℝ := 7
def salesTaxPerTire : ℝ := 0.50
def numberOfTires : ℕ := 4

-- The total cost should be $30
theorem finalCostCalculation : 
  let repairTotal := tireRepairCost * numberOfTires
  let salesTaxTotal := salesTaxPerTire * numberOfTires
  repairTotal + salesTaxTotal = 30 := 
by {
  sorry
}

end finalCostCalculation_l395_395038


namespace equation_solution_l395_395839

theorem equation_solution (x y : ℕ) (h : x^3 - y^3 = x * y + 61) : x = 6 ∧ y = 5 :=
by
  sorry

end equation_solution_l395_395839


namespace area_of_union_of_triangles_l395_395110

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * |(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))|

def reflected_point (p : Point) (y_reflect_line : ℝ) : Point :=
  { x := p.x, y := 2 * y_reflect_line - p.y }

def triangle_union_area (p1 p2 p3 rp1 rp2 rp3 : Point) : ℝ :=
  if triangle_area p1 p2 p3 > triangle_area rp1 rp2 rp3 then
    triangle_area p1 p2 p3
  else
    triangle_area rp1 rp2 rp3

theorem area_of_union_of_triangles :
  let A := Point.mk 6 5,
      B := Point.mk 8 3,
      C := Point.mk 9 1,
      y_line := 1,
      A' := reflected_point A y_line,
      B' := reflected_point B y_line,
      C' := reflected_point C y_line
  in triangle_union_area A B C A' B' C' = 4 := by
  sorry

end area_of_union_of_triangles_l395_395110


namespace distance_center_to_line_l395_395578

noncomputable def distance_from_center_to_line (C C1 C2 C3 : Circle) (L : Line) : Real :=
  7

theorem distance_center_to_line (C C1 C2 C3 : Circle) (L : Line)
  (h1 : L ∩ C = ∅) -- C and L are disjoint
  (h2 : C1.touches C2) (h3 : C1.touches C3) (h4 : C2.touches C3) -- C1, C2, C3 touch each other
  (h5 : C1.touches C) (h6 : C2.touches C) (h7 : C3.touches C) -- each touches C
  (h8 : C1.touches L) (h9 : C2.touches L) (h10 : C3.touches L) -- each touches L
  (radius_C : C.radius = 1) :
  distance_from_center_to_line C C1 C2 C3 L = 7 :=
sorry

end distance_center_to_line_l395_395578


namespace pebbles_divisibility_impossibility_l395_395577

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

end pebbles_divisibility_impossibility_l395_395577


namespace horner_eval_example_l395_395048

noncomputable def polynomial_eval_at (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_eval_example : polynomial_eval_at (-0.2) ≈ 0.00427 :=
  sorry

end horner_eval_example_l395_395048


namespace coin_division_l395_395166

-- Let n be a positive integer.
def face_value (n : Nat) : ℝ := 1 / n

theorem coin_division
  (coins : List ℝ)
  (h1 : ∀ c ∈ coins, ∃ n : ℕ, n > 0 ∧ c = face_value n)
  (h2 : coins.sum ≤ 99 + 1 / 2) :
  ∃ groups : List (List ℝ), groups.length ≤ 100 ∧ (∀ g ∈ groups, g.sum ≤ 1) :=
sorry

end coin_division_l395_395166


namespace climb_stairs_l395_395040

noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def phi_bar := (1 - Real.sqrt 5) / 2

def F : ℕ → ℕ 
| 0     := 1
| 1     := 1
| (n+2) := F (n + 1) + F n

lemma varphi_property : phi^2 = phi + 1 :=
begin
  -- proof omitted
  sorry
end

lemma varphi_bar_property : phi_bar^2 = phi_bar + 1 :=
begin
  -- proof omitted
  sorry
end

theorem climb_stairs (n : ℕ) : 
  F n = (1 / Real.sqrt 5) * (phi^(n+1) - phi_bar^(n+1)) :=
begin
  -- proof omitted
  sorry
end

end climb_stairs_l395_395040


namespace d_e_f_sum_eq_903_l395_395519

noncomputable def problem_data := 
{ castle_radius : ℝ := 12,
  rope_length : ℝ := 30,
  unicorn_height : ℝ := 6,
  rope_distance_castle : ℝ := 6 }

theorem d_e_f_sum_eq_903 (d e f : ℕ) [fact (nat.prime f)] :
  let rope_touching_castle := (d - real.sqrt e) / f in 
  d + e + f = 903 :=
  
begin
  sorry
end

end d_e_f_sum_eq_903_l395_395519


namespace sum_of_cubes_of_roots_l395_395539

theorem sum_of_cubes_of_roots :
  ∀ (x : ℝ), (x * real.sqrt x - 7 * x + 8 * real.sqrt x - 1 = 0) → (real.sqrt x ≥ 0) → 
  ∑ (r : ℝ) in ({real.sqrt r | r * real.sqrt r - 7 * r + 8 * real.sqrt r - 1 = 0}.to_finset : finset ℝ), r^3 = 178 :=
sorry

end sum_of_cubes_of_roots_l395_395539


namespace vitya_can_determine_all_answers_30th_attempt_vitya_can_determine_all_answers_25th_attempt_l395_395868

theorem vitya_can_determine_all_answers_30th_attempt :
  ∃ (attempts : ℕ), attempts ≤ 30 ∧
    ( ∀ (questions : Fin 30 → Bool),
      ∃ (strategy : Fin 30 → (Fin 30 → Bool) → ℕ)
      (feedback : Π (attempt : ℕ), ℕ),
      (∀ attempt, feedback attempt = count_correct_answers (questions (strategy attempt))) →
        (knows_correct_answers (strategy) questions 30)) :=
sorry

theorem vitya_can_determine_all_answers_25th_attempt :
  ∃ (attempts : ℕ), attempts ≤ 25 ∧
    ( ∀ (questions : Fin 30 → Bool),
      ∃ (strategy : Fin 25 → (Fin 30 → Bool) → ℕ)
      (feedback : Π (attempt : ℕ), ℕ),
      (∀ attempt, feedback attempt = count_correct_answers (questions (strategy attempt))) →
        (knows_correct_answers (strategy) questions 25)) :=
sorry

end vitya_can_determine_all_answers_30th_attempt_vitya_can_determine_all_answers_25th_attempt_l395_395868


namespace permutation_of_digits_l395_395228

-- Definition of factorial
def fact : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * fact n

-- Given conditions
def n := 8
def n1 := 3
def n2 := 2
def n3 := 1
def n4 := 2

-- Statement
theorem permutation_of_digits :
  fact n / (fact n1 * fact n2 * fact n3 * fact n4) = 1680 :=
by
  sorry

end permutation_of_digits_l395_395228


namespace sin_2theta_l395_395198

theorem sin_2theta (θ : ℝ) (h : 2^(-2 + 3 * sin θ) + 1 = 2^(1 / 2 + sin θ)) : sin (2 * θ) = 4 * real.sqrt 2 / 9 :=
by sorry

end sin_2theta_l395_395198


namespace minimum_seating_l395_395084

-- Define the condition of 60 chairs arranged in a circle
def Chairs := {x // x < 60}

-- Define a configuration where N people are seated such that no new person can sit isolated
def valid_seating (N : ℕ) (seating : Finset Chairs) : Prop :=
  seating.card = N ∧ 
  ∀ i ∈ Chairs, ¬{\(seating.contains (i - 1) \lor seating.contains (i + 1) \mod 60 → i ∈ seating\)}

-- Prove the minimum N
theorem minimum_seating : 
  ∃ (N : ℕ), (N = 15) ∧ (∃ (seating : Finset Chairs), valid_seating N seating) :=
sorry

end minimum_seating_l395_395084


namespace solve_cubic_inequality_l395_395841

theorem solve_cubic_inequality : 
  ∀ x : ℝ, (-8 * x^3 - 10 * x^2 + 5 * x - 3 > 0) ↔ (x ∈ Iio (-2) ∪ Ioo (-0.5) 1) :=
by
  sorry

end solve_cubic_inequality_l395_395841


namespace cost_price_of_book_l395_395065

-- Define the variables and conditions
variable (C : ℝ)
variable (P : ℝ)
variable (S : ℝ)

-- State the conditions given in the problem
def conditions := S = 260 ∧ P = 0.20 * C ∧ S = C + P

-- State the theorem
theorem cost_price_of_book (h : conditions C P S) : C = 216.67 :=
sorry

end cost_price_of_book_l395_395065


namespace sequence_properties_l395_395190

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2 * a_n n + (-1 : ℝ) ^ n

def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
  if n = 2 then 0 else
  if n = 3 then 2 else sorry 

theorem sequence_properties (n : ℕ) (h : n > 0) :
  (a_n 1 = 1) ∧ (a_n 2 = 0) ∧ (a_n 3 = 2) ∧
  (∃ r : ℝ, ∀ n : ℕ, n > 0 →
    (a_n n + 2 / 3 * (-1 : ℝ) ^ n = r * (a_n (n - 1) + 2 / 3 * (-1 : ℝ) ^ (n - 1)))) ∧
  (∀ n : ℕ, n > 0 →
    a_n n = 1 / 3 * 2 ^ (n - 1) + 2 / 3 * (-1 : ℝ) ^ (n - 1)) :=
by
  sorry

end sequence_properties_l395_395190


namespace smallest_four_digit_number_l395_395057

theorem smallest_four_digit_number 
  (N : ℕ) 
  (h1 : 1000 ≤ N ∧ N < 10000)
  (h2 : N % 9 = 0)
  (h3 : (N.digits 10).count (λ d, d % 2 = 0) = 2)
  (h4 : (N.digits 10).count (λ d, d % 2 ≠ 0) = 2)
  (h5 : 5 ∈ N.digits 10) : 
  N = 1058 :=
sorry

end smallest_four_digit_number_l395_395057


namespace domain_sqrt_product_domain_log_fraction_l395_395158

theorem domain_sqrt_product (x : ℝ) (h1 : x - 2 ≥ 0) (h2 : x + 2 ≥ 0) : 
  2 ≤ x :=
by sorry

theorem domain_log_fraction (x : ℝ) (h1 : x + 1 > 0) (h2 : -x^2 - 3 * x + 4 > 0) : 
  -1 < x ∧ x < 1 :=
by sorry

end domain_sqrt_product_domain_log_fraction_l395_395158


namespace find_a_b_l395_395621

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395621


namespace smallest_sum_x_y_z_w_l395_395803

theorem smallest_sum_x_y_z_w :
  ∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  let A := ![![5, 0], ![0, 3]],
      B := ![![x, y], ![z, w]],
      C := ![![25, 15], ![-21, -12]] in
  A ⬝ B = B ⬝ C ∧ (x + y + z + w = 84) :=
by sorry

end smallest_sum_x_y_z_w_l395_395803


namespace counterexamples_count_l395_395565

def sum_of_digits (n : ℕ) : ℕ :=
nat.digits 10 n |>.sum

def is_non_zero_digit (n : ℕ) : Prop :=
∀ d ∈ nat.digits 10 n, d ≠ 0

def is_counterexample (n : ℕ) : Prop :=
sum_of_digits n = 5 ∧ is_non_zero_digit n ∧ ¬ nat.prime n

noncomputable def number_of_counterexamples : ℕ :=
{n | is_counterexample n}.to_finset.card

theorem counterexamples_count :
  number_of_counterexamples = 6 :=
sorry

end counterexamples_count_l395_395565


namespace cos_alpha_of_point_P_l395_395188

noncomputable def cos_of_angle (x y : ℝ) : ℝ :=
  let r := real.sqrt (x ^ 2 + y ^ 2)
  x / r

theorem cos_alpha_of_point_P (x y : ℝ) (h₁ : x = -12) (h₂ : y = 5) : 
  cos_of_angle x y = -12 / 13 :=
by
  rw [h₁, h₂]
  have hr : real.sqrt (x ^ 2 + y ^ 2) = 13 := by norm_num
  sorry

end cos_alpha_of_point_P_l395_395188


namespace find_letter_Q_l395_395814

variable (x : ℤ)

def date_A := x + 2
def date_B := x + 14
def date_sum := date_A x + date_B x
def date_C := x
def letter_difference (y : ℤ) := date_C x - y

theorem find_letter_Q :
  ∃ (Q : ℤ), letter_difference x Q = date_sum x ↔ Q = -x - 16 :=

end find_letter_Q_l395_395814


namespace max_sum_mul_table_l395_395394

-- Define the numbers
def numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Prove the maximum possible sum of the entries in the multiplication table
theorem max_sum_mul_table (a b c d e f g h : ℕ) 
  (h1 : [a, b, c, d, e, f, g, h] ~ numbers)
  : (a + b + c + d) * (e + f + g + h) ≤ 1440 :=
sorry

end max_sum_mul_table_l395_395394


namespace exists_point_in_B_connected_to_all_A_l395_395984

variables {G : Type} [Fintype G] [DecidableEq G]
variables (n k m : ℕ)
variables (A B : Finset G)
variables (conn : G → G → Prop)
variables [DecidableRel conn]

-- Definitions from the conditions
def is_graph (G : Type) := Fintype G
def subset_A := A.card = n
def subset_B := B.card = k
def connections_in_A_to_B := ∀ a ∈ A, (B.filter (conn a)).card ≥ k - m
def nm_lt_k := n * m < k

theorem exists_point_in_B_connected_to_all_A :
  is_graph G → subset_A A n → subset_B B k → connections_in_A_to_B A B conn k m → nm_lt_k n m k →
  ∃ b ∈ B, ∀ a ∈ A, conn a b :=
by {
  -- Sorry skips the proof
  sorry
}

end exists_point_in_B_connected_to_all_A_l395_395984


namespace travel_time_correct_l395_395140

-- Define the conditions of the problem
def start_time : ℕ := 7 -- Journey starts at 7am

def first_segment_distance : ℕ := 200
def first_segment_speed : ℕ := 65
def first_segment_time : ℚ := first_segment_distance / first_segment_speed

def lunch_break_time : ℚ := 1 + (15 / 60) -- 1 hour and 15 minutes

def second_segment_distance : ℕ := 350
def second_segment_speed : ℕ := 45
def second_segment_time : ℚ := second_segment_distance / second_segment_speed

def nap_break_time : ℚ := 2 -- 2 hours

def third_segment_distance : ℕ := 150
def third_segment_speed : ℕ := 60
def third_segment_time : ℚ := third_segment_distance / third_segment_speed

-- Total travel time in hours
def total_travel_time : ℚ := first_segment_time + lunch_break_time + second_segment_time + nap_break_time + third_segment_time

-- Convert total travel time to hours and minutes
def hours_part : ℕ := total_travel_time.to_int.to_nat
def minutes_part : ℚ := (total_travel_time - hours_part) * 60

theorem travel_time_correct : 
  hours_part = 16 ∧ minutes_part = 7 :=
by
  -- proof steps would go here
  sorry

end travel_time_correct_l395_395140


namespace maximize_probability_l395_395938

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l395_395938


namespace select_students_l395_395174

theorem select_students {A B C D : Prop} : -- representing students as propositions
  ∃ (f : {x // x = A ∨ x = B} → {y // y = C ∨ y = D} → Prop), 
    (∀ (a ∈ {A, B}) (c ∈ {C, D}), f ⟨a, _⟩ ⟨c, _⟩) ∧
    (∑ (a ∈ {A, B}) (c ∈ {C, D}), 1 = 4) :=
sorry

end select_students_l395_395174


namespace quadrilateral_inscribed_circle_iff_sides_sum_equal_l395_395818

-- Definitions of the conditions
variables (A B C D : Type*) [IsConvexQuadrilateral A B C D]

-- The condition involving the sides of the quadrilateral
def has_inscribed_circle (A B C D : Type*) := -- placeholder for inscribed circle property
  sorry

-- The sides' equivalence condition
def sides_sum_equal (A B C D : Type*) := 
  AB + CD = AD + BC

-- Statement of the theorem to be proven
theorem quadrilateral_inscribed_circle_iff_sides_sum_equal :
  has_inscribed_circle A B C D ↔ sides_sum_equal A B C D :=
sorry

end quadrilateral_inscribed_circle_iff_sides_sum_equal_l395_395818


namespace simplify_root_exponentiation_l395_395832

theorem simplify_root_exponentiation : (7 ^ (1 / 3) : ℝ) ^ 6 = 49 := by
  sorry

end simplify_root_exponentiation_l395_395832


namespace f_neg_1_l395_395205

-- Define the functions
variable (f : ℝ → ℝ) -- f is a real-valued function
variable (g : ℝ → ℝ) -- g is a real-valued function

-- Given conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_def : ∀ x, g x = f x + 4
axiom g_at_1 : g 1 = 2

-- Define the theorem to prove
theorem f_neg_1 : f (-1) = 2 :=
by
  -- Proof goes here
  sorry

end f_neg_1_l395_395205


namespace quadratic_eq_with_root_sqrt5_min3_l395_395149

theorem quadratic_eq_with_root_sqrt5_min3 :
  ∃ (b c : ℚ), (∀ x : ℚ, x^2 + b * x + c = 0 -> x = (√5 - 3: ℚ) ∨ x = (-√5 - 3: ℚ)) ∧ (b = 6) ∧ (c = -4) :=
by
  sorry

end quadratic_eq_with_root_sqrt5_min3_l395_395149


namespace problem_statement_l395_395237

variable {x : ℝ}
variable {z : ℝ}

theorem problem_statement (h : |2 * x - real.sqrt z| = 2 * x + real.sqrt z) : x ≥ 0 ∧ z = 0 := by
  sorry

end problem_statement_l395_395237


namespace derivative_sqrt_l395_395852

/-- The derivative of the function y = sqrt x is 1 / (2 * sqrt x) -/
theorem derivative_sqrt (x : ℝ) (h : 0 < x) : (deriv (fun x => Real.sqrt x) x) = 1 / (2 * Real.sqrt x) :=
sorry

end derivative_sqrt_l395_395852


namespace find_a_b_l395_395589

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395589


namespace maximum_pairwise_sum_is_maximal_l395_395105

noncomputable def maximum_pairwise_sum (set_sums : List ℝ) (x y z w : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), set_sums = [400, 500, 600, 700, 800, 900, x, y, z, w] ∧  
  ((2 / 5) * (400 + 500 + 600 + 700 + 800 + 900 + x + y + z + w)) = 
    (a + b + c + d + e) ∧ 
  5 * (a + b + c + d + e) - (400 + 500 + 600 + 700 + 800 + 900) = 1966.67

theorem maximum_pairwise_sum_is_maximal :
  maximum_pairwise_sum [400, 500, 600, 700, 800, 900] 1966.67 (1966.67 / 4) 
(1966.67 / 3) (1966.67 / 2) :=
sorry

end maximum_pairwise_sum_is_maximal_l395_395105


namespace selling_price_l395_395107

theorem selling_price (cost_price profit_percentage selling_price : ℝ) (h1 : cost_price = 86.95652173913044)
  (h2 : profit_percentage = 0.15) : 
  selling_price = 100 :=
by
  sorry

end selling_price_l395_395107


namespace chloe_total_profit_l395_395536

-- Define the variables according to the conditions.
def cost_per_dozen_range := [40, 45]
def average_cost_per_dozen : Float := (40 + 45) / 2
def cost_of_production (num_dozens : Nat) : Float := num_dozens * average_cost_per_dozen

def first_day_price_per_dozen : Float := 50
def second_day_price_per_dozen : Float := 50 * 1.2
def third_day_price_per_dozen : Float := 50 * 0.85

def first_day_sales : Nat := 12
def second_day_sales : Nat := 18
def third_day_sales : Nat := 20

def revenue_first_day : Float := first_day_sales * first_day_price_per_dozen
def revenue_second_day : Float := second_day_sales * second_day_price_per_dozen
def revenue_third_day : Float := third_day_sales * third_day_price_per_dozen

def total_revenue_before_discount : Float := revenue_first_day + revenue_second_day + revenue_third_day

def bulk_discount (total_revenue : Float) : Float := total_revenue * 0.1

def total_revenue_after_discount := total_revenue_before_discount - bulk_discount(total_revenue_before_discount)

def total_cost_of_production := cost_of_production 50

def total_profit := total_revenue_after_discount - total_cost_of_production

-- Prove that the total profit is $152.
theorem chloe_total_profit : total_profit = 152 := by
  have h_avg_cost : average_cost_per_dozen = 42.5 := rfl
  have h_first_day_revenue : revenue_first_day = 600 := rfl
  have h_second_day_revenue : revenue_second_day = 1080 := rfl
  have h_third_day_revenue : revenue_third_day = 850 := rfl
  have h_total_revenue_before_discount : total_revenue_before_discount = 2530 := 
    by rw [h_first_day_revenue, h_second_day_revenue, h_third_day_revenue]; norm_num
  have h_bulk_discount : bulk_discount total_revenue_before_discount = 253 := rfl
  have h_total_revenue_after_discount : total_revenue_after_discount = 2277 :=
    by rw [←h_total_revenue_before_discount, ←h_bulk_discount]; norm_num
  have h_total_cost_of_production : total_cost_of_production = 2125 :=
    by rw [←h_avg_cost]; norm_num
  have h_total_profit : total_profit = 152 := 
    by rw [←h_total_revenue_after_discount, ←h_total_cost_of_production]; norm_num
  exact h_total_profit

end chloe_total_profit_l395_395536


namespace find_a_b_l395_395671

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395671


namespace maximum_probability_second_game_C_l395_395974

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l395_395974


namespace quadratic_equation_with_root_l395_395145

theorem quadratic_equation_with_root (x : ℝ) (h1 : x = sqrt 5 - 3) :
  ∃ (a b c : ℚ), a = 1 ∧ b = 6 ∧ c = 4 ∧ (a * x^2 + b * x + c = 0) :=
begin
  sorry
end

end quadratic_equation_with_root_l395_395145


namespace simplify_polynomial_l395_395050

variable (x : ℝ)

theorem simplify_polynomial :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 =
  6*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end simplify_polynomial_l395_395050


namespace find_a_l395_395419

theorem find_a (a : ℝ) 
    (H1 : ∀ (x y : ℝ), x^2 + y^2 + 4y = 0 → x^2 + (y + 2)^2 = 4)
    (H2 : ∀ (x y : ℝ), x^2 + y^2 + 2 * (a - 1) * x + 2 * y + a^2 = 0)
    (H3 : ∀ (m n : ℝ), 
        (m^2 + n^2 + 4n = 0) ∧ 
        (2 * (a - 1) * m - 2 * n + a^2 = 0) ∧
        (((n + 2) / m) * ((n + 1) / (m - (1 - a))) = -1)) :
  a = -2 :=
sorry

end find_a_l395_395419


namespace lily_milk_remaining_l395_395811

def lilyInitialMilk : ℚ := 4
def milkGivenAway : ℚ := 7 / 3
def milkLeft : ℚ := 5 / 3

theorem lily_milk_remaining : lilyInitialMilk - milkGivenAway = milkLeft := by
  sorry

end lily_milk_remaining_l395_395811


namespace maximize_probability_l395_395943

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l395_395943


namespace slope_of_perpendicular_line_l395_395163

-- Define what it means to be the slope of a line in a certain form
def slope_of_line (a b c : ℝ) (m : ℝ) : Prop :=
  b ≠ 0 ∧ m = -a / b

-- Define what it means for two slopes to be perpendicular
def are_perpendicular_slopes (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Given conditions
def given_line : Prop := slope_of_line 4 5 20 (-4 / 5)

-- The theorem to be proved
theorem slope_of_perpendicular_line : ∃ m : ℝ, given_line ∧ are_perpendicular_slopes (-4 / 5) m ∧ m = 5 / 4 :=
  sorry

end slope_of_perpendicular_line_l395_395163


namespace probability_at_least_two_heads_in_four_tosses_l395_395828

theorem probability_at_least_two_heads_in_four_tosses :
  let p := 11 / 16 in
  (∑ k in finset.range 3, nat.choose 4 k * (1 / 2)^k * (1 / 2)^(4 - k)) = 1 - p := sorry

end probability_at_least_two_heads_in_four_tosses_l395_395828


namespace min_time_to_share_all_information_l395_395408

-- Define the basic structure and conditions of the problem
constant individual : Type
constant information : Type

-- We have 8 individuals and each has unique information initially
constant individuals : list individual
constant initial_information : individual → information
constant unique_information : ∀ i j, i ≠ j → initial_information i ≠ initial_information j

-- Each phone call takes exactly 3 minutes
constant call_duration : ℝ := 3

-- Function to model the information spread process
axiom information_spread : list (individual × individual) → individual → information

-- We need to prove that the minimum time for all to know all information is 9 minutes
theorem min_time_to_share_all_information : 
  ∀ calls : list (list (individual × individual)),
  (∀ pair_list ∈ calls, ∀ (i j) ∈ pair_list, call_duration = 3) →
  (∀ t < 9, ∃ indiv, information_spread (calls.take t) indiv ≠ information_spread (calls.take 9) indiv) →
  ∀ indiv, information_spread (calls.take 9) indiv = initial_information indiv.sorry

end min_time_to_share_all_information_l395_395408


namespace sum_sqrt_inequality_l395_395309

theorem sum_sqrt_inequality (n : ℕ) (x : Fin n.succ → ℝ)
  (h1 : ∀ i, 0 < x i)
  (h2 : (∏ i, x i) = 1) :
  (∑ i, (x i) * sqrt (∑ j in Finset.range (i + 1), (x j) ^ 2)) ≥ ((n + 1) * sqrt n.succ) / 2 :=
  sorry

end sum_sqrt_inequality_l395_395309


namespace hexagon_ratio_constant_l395_395296

open Mathlib

theorem hexagon_ratio_constant (ABCDEF : Type) [hexagon : RegularHexagon ABCDEF] (P : Point)
  (hP : OnShorterArcEF P ABCDEF) :
  ∀ A B C D E F, ABCDEF = [A, B, C, D, E, F] ->
  ∃ k, k = 3 + sqrt 3 ∧
  ∀ AP BP CP DP EP FP, 
    AP = distance A P -> BP = distance B P -> CP = distance C P -> DP = distance D P -> 
    EP = distance E P -> FP = distance F P ->
    (AP + BP + CP + DP) / (EP + FP) = k :=
sorry

end hexagon_ratio_constant_l395_395296


namespace factor_is_gcf_l395_395060

-- Define the structure for the variables and coefficients
def expr_1 (a b : ℕ) := 4 * a^2 * b^3
def expr_2 (a b : ℕ) := 6 * a^3 * b

-- Declare the variables
variables (a b : ℕ)

-- Helper functions to calculate GCF of numbers
def gcd (x y : ℕ) : ℕ := Nat.gcd x y

-- Function to determine the minimum power of a variable
def min_power (x y : ℕ) : ℕ := min x y

-- Main statement to assert that the GCF of the expression is 2a^2b
theorem factor_is_gcf :
  let gcf := 2 * a^2 * b
  in gcd (expr_1 a b) (expr_2 a b) = gcf :=
by 
  sorry

end factor_is_gcf_l395_395060


namespace maximize_profit_l395_395484

noncomputable def L (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 80 then - (1 / 3) * x^2 + 40 * x - 250
else 1200 - (x + 10000 / x)

theorem maximize_profit :
  ∀ x : ℝ, L 100 ≥ L x :=
begin
  sorry
end

end maximize_profit_l395_395484


namespace octahedron_sum_l395_395785

-- Define the properties of an octahedron
def octahedron_edges := 12
def octahedron_vertices := 6
def octahedron_faces := 8

theorem octahedron_sum : octahedron_edges + octahedron_vertices + octahedron_faces = 26 := by
  -- Here we state that the sum of edges, vertices, and faces equals 26
  sorry

end octahedron_sum_l395_395785


namespace quadratic_eq_with_root_sqrt5_min3_l395_395148

theorem quadratic_eq_with_root_sqrt5_min3 :
  ∃ (b c : ℚ), (∀ x : ℚ, x^2 + b * x + c = 0 -> x = (√5 - 3: ℚ) ∨ x = (-√5 - 3: ℚ)) ∧ (b = 6) ∧ (c = -4) :=
by
  sorry

end quadratic_eq_with_root_sqrt5_min3_l395_395148


namespace integral_exp_2x_l395_395072

theorem integral_exp_2x : ∫ x in 0..(1/2), exp (2 * x) = (1 / 2) * (Real.exp 1 - 1) := by
  sorry

end integral_exp_2x_l395_395072


namespace prime_sum_diff_l395_395154

open Nat

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The problem statement
theorem prime_sum_diff (p : ℕ) (q s r t : ℕ) :
  is_prime p → is_prime q → is_prime s → is_prime r → is_prime t →
  p = q + s → p = r - t → p = 5 :=
by
  sorry

end prime_sum_diff_l395_395154


namespace ratio_of_chords_l395_395418

theorem ratio_of_chords 
  (E F G H Q : Type)
  (EQ GQ FQ HQ : ℝ)
  (h1 : EQ = 4)
  (h2 : GQ = 10)
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 5 / 2 := 
by 
  sorry

end ratio_of_chords_l395_395418


namespace part_1_part_2_l395_395688

def p (a x : ℝ) : Prop :=
a * x - 2 ≤ 0 ∧ a * x + 1 > 0

def q (x : ℝ) : Prop :=
x^2 - x - 2 < 0

theorem part_1 (a : ℝ) :
  (∃ x : ℝ, (1/2 < x ∧ x < 3) ∧ p a x) → 
  (-2 < a ∧ a < 4) :=
sorry

theorem part_2 (a : ℝ) :
  (∀ x, p a x → q x) ∧ 
  (∃ x, q x ∧ ¬p a x) → 
  (-1/2 ≤ a ∧ a ≤ 1) :=
sorry

end part_1_part_2_l395_395688


namespace max_prob_win_two_consecutive_is_C_l395_395960

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l395_395960


namespace remainder_div_2468135790_101_l395_395434

theorem remainder_div_2468135790_101 : 2468135790 % 101 = 50 :=
by
  sorry

end remainder_div_2468135790_101_l395_395434


namespace problem_l395_395807

def y : ℝ := (sqrt 75 / 3 - 5 / 2)^(1/2)

theorem problem :
  let a := 24
  let b := 15
  let c := 27
  y^100 = 3 * y^98 + 15 * y^96 + 12 * y^94 - 2 * y^50 + a * y^46 + b * y^44 + c * y^40 :=
by
  sorry

end problem_l395_395807


namespace interchangeable_statements_l395_395239

-- Modeled conditions and relationships
def perpendicular (l p: Type) : Prop := sorry -- Definition of perpendicularity between a line and a plane
def parallel (a b: Type) : Prop := sorry -- Definition of parallelism between two objects (lines or planes)

-- Original Statements
def statement_1 := ∀ (l₁ l₂ p: Type), (perpendicular l₁ p) ∧ (perpendicular l₂ p) → parallel l₁ l₂
def statement_2 := ∀ (p₁ p₂ p: Type), (perpendicular p₁ p) ∧ (perpendicular p₂ p) → parallel p₁ p₂
def statement_3 := ∀ (l₁ l₂ l: Type), (parallel l₁ l) ∧ (parallel l₂ l) → parallel l₁ l₂
def statement_4 := ∀ (l₁ l₂ p: Type), (parallel l₁ p) ∧ (parallel l₂ p) → parallel l₁ l₂

-- Swapped Statements
def swapped_1 := ∀ (p₁ p₂ l: Type), (perpendicular p₁ l) ∧ (perpendicular p₂ l) → parallel p₁ p₂
def swapped_2 := ∀ (l₁ l₂ l: Type), (perpendicular l₁ l) ∧ (perpendicular l₂ l) → parallel l₁ l₂
def swapped_3 := ∀ (p₁ p₂ p: Type), (parallel p₁ p) ∧ (parallel p₂ p) → parallel p₁ p₂
def swapped_4 := ∀ (p₁ p₂ l: Type), (parallel p₁ l) ∧ (parallel p₂ l) → parallel p₁ p₂

-- Proof Problem: Verify which statements are interchangeable
theorem interchangeable_statements :
  (statement_1 ↔ swapped_1) ∧
  (statement_2 ↔ swapped_2) ∧
  (statement_3 ↔ swapped_3) ∧
  (statement_4 ↔ swapped_4) :=
sorry

end interchangeable_statements_l395_395239


namespace find_a_b_l395_395665

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395665


namespace range_of_omega_l395_395242

noncomputable def function_is_monotonic (ω : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (x2 < π / 3) → 
    sin (ω * x1 + π / 6) ≤ sin (ω * x2 + π / 6)

theorem range_of_omega (ω : ℝ) (hω : 0 < ω) (hmono : function_is_monotonic ω) : ω ≤ 1 :=
sorry

end range_of_omega_l395_395242


namespace biology_collections_count_l395_395330

/-- 
Proof Problem: Given the letters in "BIOLOGY", with O's and G's being indistinguishable, 
the number of distinct possible collections of 3 vowels and 2 consonants is 12.
-/
theorem biology_collections_count : 
  let vowels := ['I', 'O', 'O', 'Y'],
      consonants := ['B', 'G', 'G'],
      num_possible_collections := 4 * 3
  in num_possible_collections = 12 := 
by 
  let vowels := ['I', 'O', 'O', 'Y']
  let consonants := ['B', 'G', 'G']
  let num_vowel_groups := 4 -- from the breakdown in the solution
  let num_consonant_groups := 3 -- from the breakdown in the solution
  have distinct_collections : 4 * 3 = 12, by norm_num
  exact distinct_collections

end biology_collections_count_l395_395330


namespace canoe_prob_calc_l395_395473

theorem canoe_prob_calc : 
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_both_work := p_left_works * p_right_works
  let p_left_works_right_breaks := p_left_works * p_right_breaks
  let p_left_breaks_right_works := p_left_breaks * p_right_works
  let p_can_row := p_both_work + p_left_works_right_breaks + p_left_breaks_right_works
  p_left_works = 3 / 5 → 
  p_right_works = 3 / 5 → 
  p_can_row = 21 / 25 :=
by
  intros
  unfold p_left_works p_right_works p_left_breaks p_right_breaks p_both_work p_left_works_right_breaks p_left_breaks_right_works p_can_row
  sorry

end canoe_prob_calc_l395_395473


namespace num_mappings_from_A_to_B_l395_395316

-- Define sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set char := {'a', 'b'}

-- Define the theorem to prove
theorem num_mappings_from_A_to_B : { f : A → B // true } = 8 := 
sorry

end num_mappings_from_A_to_B_l395_395316


namespace find_a_b_l395_395618

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395618


namespace cost_to_fly_A_to_B_l395_395815

variable {distance_A_B : ℕ} -- Distance from A to B
variable {cost_per_km_fly : ℕ} -- Cost per kilometer to fly
variable {booking_fee : ℕ} -- Booking fee to fly

-- Given conditions as hypotheses
def distance_from_A_to_B := distance_A_B = 4500
def cost_per_km_to_fly := cost_per_km_fly = 12
def flight_booking_fee := booking_fee = 120

-- The proof statement to prove the total cost to fly from A to B
theorem cost_to_fly_A_to_B :
  distance_from_A_to_B →
  cost_per_km_to_fly →
  flight_booking_fee →
  (distance_A_B * cost_per_km_fly / 100 + booking_fee) = 660 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end cost_to_fly_A_to_B_l395_395815


namespace maximum_probability_second_game_C_l395_395975

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l395_395975


namespace eating_time_175_seconds_l395_395982

variable (Ponchik_time Neznaika_time : ℝ)
variable (Ponchik_rate Neznaika_rate : ℝ)

theorem eating_time_175_seconds
    (hP_rate : Ponchik_rate = 1 / Ponchik_time)
    (hP_time : Ponchik_time = 5)
    (hN_rate : Neznaika_rate = 1 / Neznaika_time)
    (hN_time : Neznaika_time = 7)
    (combined_rate := Ponchik_rate + Neznaika_rate)
    (total_minutes := 1 / combined_rate)
    (total_seconds := total_minutes * 60):
    total_seconds = 175 := by
  sorry

end eating_time_175_seconds_l395_395982


namespace arrange_number_of_ways_l395_395100

-- Define the problem setup
def num_schools : ℕ := 10
def total_days : ℕ := 30
def larger_school_days : ℕ := 2
def other_schools : ℕ := 9

-- Calculate the number of ways to arrange the visits
noncomputable def number_of_arrangements : ℕ :=
  (29.choose 1) * (28.perm 9)

-- Statement to prove
theorem arrange_number_of_ways:
  number_of_arrangements = 
  (29.choose 1) * (28.perm 9) :=
sorry

end arrange_number_of_ways_l395_395100


namespace sequence_expression_l395_395776

-- Define the sequence a_n using recursion
def a : ℕ → ℚ
| 0     := 2
| (n+1) := a n / (3 * a n + 1)

-- State the theorem about the general expression for a_n
theorem sequence_expression (n : ℕ) : a n = 2 / (6 * n - 5) :=
by
  sorry

end sequence_expression_l395_395776


namespace translate_sin_eq_cos_l395_395384

theorem translate_sin_eq_cos (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < 2 * Real.pi) :
  (∀ x, Real.cos (x - Real.pi / 6) = Real.sin (x + φ)) → φ = Real.pi / 3 :=
by
  sorry

end translate_sin_eq_cos_l395_395384


namespace find_machine_addition_l395_395729

theorem find_machine_addition :
  ∃ x : ℕ, (26 + x - 6 = 35) ∧ (x = 15) :=
begin
  use 15,
  split,
  {
    norm_num,
  },
  {
    refl,
  }
end

end find_machine_addition_l395_395729


namespace max_prob_two_consecutive_wins_l395_395967

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l395_395967


namespace find_constants_l395_395156

theorem find_constants : 
  ∃ A B C : ℚ, A = 3 ∧ B = -3 ∧ C = -6 ∧ 
  (∀ x : ℚ, x ≠ 4 → x ≠ 2 → 
    6 * x = A * (x - 2)^2 + B * (x - 4) * (x - 2) + C * (x - 4) 
  ) :=
begin
  use [3, -3, -6],
  split, { refl },
  split, { refl },
  split, { refl },
  intros x hx1 hx2,
  simp,
  sorry,
end

end find_constants_l395_395156


namespace single_rooms_booked_l395_395119

noncomputable def hotel_problem (S D : ℕ) : Prop :=
  S + D = 260 ∧ 35 * S + 60 * D = 14000

theorem single_rooms_booked (S D : ℕ) (h : hotel_problem S D) : S = 64 :=
by
  sorry

end single_rooms_booked_l395_395119


namespace geom_seq_min_value_l395_395313

theorem geom_seq_min_value (r : ℝ) : 
  (1 : ℝ) = a_1 → a_2 = r → a_3 = r^2 → ∃ r : ℝ, 6 * a_2 + 7 * a_3 = -9/7 := 
by 
  intros h1 h2 h3 
  use -3/7 
  rw [h2, h3] 
  ring 
  sorry

end geom_seq_min_value_l395_395313


namespace solve_for_a_b_l395_395632

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395632


namespace find_a_b_l395_395669

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395669


namespace factorial_divides_l395_395730

theorem factorial_divides (a : ℕ) :
  (a = 0 ∨ a = 3) ↔ (a! + (a + 2)! ∣ (a + 4)!) :=
by
  sorry

end factorial_divides_l395_395730


namespace mixed_fruit_juice_cost_correct_l395_395017

def total_volume : ℝ := 32 + 21.333333333333332
def cost_per_litre_cocktail : ℝ := 1399.45
def total_cost_cocktail : ℝ := cost_per_litre_cocktail * total_volume
def volume_acai : ℝ := 21.333333333333332
def cost_per_litre_acai : ℝ := 3104.35
def cost_acai : ℝ := cost_per_litre_acai * volume_acai
def volume_fruit_juice : ℝ := 32

-- Define the cost per litre of the mixed fruit juice to be proven
def cost_per_litre_fruit_juice : ℝ := 262.8125

theorem mixed_fruit_juice_cost_correct :
  total_cost_cocktail = (cost_per_litre_fruit_juice * volume_fruit_juice) + cost_acai :=
by
  sorry

end mixed_fruit_juice_cost_correct_l395_395017


namespace sequence_sum_wz_l395_395139

theorem sequence_sum_wz :
  ∃ s : ℝ, (s * 4 = 1) ∧ (w = 256 * s) ∧ (z = w * s) ∧ (w + z = 80) :=
begin
  -- Introduce the constant s and assume the conditions given in the problem.
  let s := (1 / 4 : ℝ),
  use s,
  split,
  { -- Prove s is the constant by the relationship 4s = 1
    exact (by norm_num : s * 4 = 1) },
  split,
  { -- Calculate w
    let w := 256 * s,
    show w = 256 * s,
    exact rfl },
  split,
  { -- Calculate z
    let z := (256 * s) * s,
    show z = (256 * s) * s,
    exact rfl },
  { -- Finally, prove w + z = 80
    let w := 256 * s,
    let z := w * s,
    have w_val : w = 64, by norm_num [w, s],
    have z_val : z = 16, by norm_num [z, w, s],
    show w + z = 80, by norm_num [w_val, z_val] }
end

end sequence_sum_wz_l395_395139


namespace find_triples_injective_functions_l395_395560

noncomputable def solution_exists (f g h : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f (x + f y) = g x + h y) ∧ (g (x + g y) = h x + f y) ∧ (h (x + h y) = f x + g y)

lemma injective (F : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ≠ b → F a ≠ F b

theorem find_triples_injective_functions (f g h : ℝ → ℝ)
  (hf : injective f) (hg : injective g) (hh : injective h)
  (h_cond : solution_exists f g h) :
  ∃ C : ℝ, ∀ x : ℝ, (f x = x + C) ∧ (g x = x + C) ∧ (h x = x + C) :=
begin
  sorry
end

end find_triples_injective_functions_l395_395560


namespace height_of_block_l395_395099

theorem height_of_block (h : ℝ) : 
  ((∃ (side : ℝ), ∃ (n : ℕ), side = 15 ∧ n = 10 ∧ 15 * 30 * h = n * side^3) → h = 75) := 
by
  intros
  sorry

end height_of_block_l395_395099


namespace maximum_probability_second_game_C_l395_395972

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l395_395972


namespace circle_ratio_is_1_l395_395510

noncomputable def regular_hexagon := 
  { ABCDEF : Type } -- Regular hexagon with side length 2
  (sides_length : ℝ)
  (side_lengths_equal : ∀ ab bc cd de ef fa : ℝ, 
    ab = bc ∧ bc = cd ∧ cd = de ∧ de = ef ∧ ef = fa ∧ fa = ab)
  (side_length : sides_length = 2)

noncomputable def circle_tangent_line 
  {ABCDEF : Type} 
  (hexagon : regular_hexagon ABCDEF) 
  := 
    ∀ (line_ab line_cd line_ef : Type), 
    (tangent : ∀ circ1 circ2 : Type, 
      tangent circ1 line_ab ∧ tangent circ2 line_cd ∧ tangent circ1 line_ef ∧ tangent circ2 line_ef)

theorem circle_ratio_is_1 
  {ABCDEF : Type} 
  (hexagon : regular_hexagon ABCDEF) 
  (tangent_results : circle_tangent_line hexagon) 
  : 
    let area_circ1 := π * (√3 / 3) ^ 2,
        area_circ2 := π * (√3 / 3) ^ 2
    in 
      (area_circ2 / area_circ1) = 1 :=
by 
  sorry

end circle_ratio_is_1_l395_395510


namespace cost_price_for_A_l395_395103

variable (A B C : Type) [Field A] [Field B] [Field C]

noncomputable def cost_price (CP_A : A) := 
  let CP_B := 1.20 * CP_A
  let CP_C := 1.25 * CP_B
  CP_C = (225 : A)

theorem cost_price_for_A (CP_A : A) : cost_price CP_A -> CP_A = 150 := by
  sorry

end cost_price_for_A_l395_395103


namespace a_n_is_perfect_square_l395_395541

def a : ℕ → ℤ
def b : ℕ → ℤ

axiom a0 : a 0 = 1
axiom a1 : a 1 = 4
axiom a2 : a 2 = 49
axiom recurrence_a : ∀ n, a (n + 1) = 7 * a n + 6 * b n - 3
axiom recurrence_b : ∀ n, b (n + 1) = 8 * a n + 7 * b n - 4

theorem a_n_is_perfect_square (n : ℕ) : ∃ k : ℤ, a n = k * k := 
sorry

end a_n_is_perfect_square_l395_395541


namespace treaty_signed_on_friday_l395_395848

def days_between (start_date : Nat) (end_date : Nat) : Nat := sorry

def day_of_week (start_day : Nat) (days_elapsed : Nat) : Nat :=
  (start_day + days_elapsed) % 7

def is_leap_year (year : Nat) : Bool :=
  if year % 4 = 0 then
    if year % 100 = 0 then
      if year % 400 = 0 then true else false
    else true
  else false

noncomputable def days_from_1802_to_1814 : Nat :=
  let leap_years := [1804, 1808, 1812]
  let normal_year_days := 365 * 9
  let leap_year_days := 366 * 3
  normal_year_days + leap_year_days

noncomputable def days_from_feb_5_to_apr_11_1814 : Nat :=
  24 + 31 + 11 -- days in February, March, and April 11

noncomputable def total_days_elapsed : Nat :=
  days_from_1802_to_1814 + days_from_feb_5_to_apr_11_1814

noncomputable def start_day : Nat := 5 -- Friday (0 = Sunday, ..., 5 = Friday, 6 = Saturday)

theorem treaty_signed_on_friday : day_of_week start_day total_days_elapsed = 5 := sorry

end treaty_signed_on_friday_l395_395848


namespace find_second_number_l395_395569

theorem find_second_number : 
  ∃ x : ℝ, 3 + x * (8 - 3) = 24.16 ∧ x = 4.232 :=
by
  use 4.232
  split
  · -- Goal 1: 3 + 4.232 * (8 - 3) = 24.16 
    sorry 
  · -- Goal 2: x = 4.232
    refl

end find_second_number_l395_395569


namespace valid_combinations_of_flowers_and_gems_l395_395112

theorem valid_combinations_of_flowers_and_gems
  (flowers : ℕ) (gems : ℕ)
  (incompatible_gem : ℕ)
  (incompatible_flowers : ℕ) :
  flowers = 4 →
  gems = 6 →
  incompatible_gem = 1 →
  incompatible_flowers = 3 →
  (flowers * gems - incompatible_gem * incompatible_flowers) = 21 :=
by
  intros h_flowers h_gems h_incompatible_gem h_incompatible_flowers
  rw [h_flowers, h_gems, h_incompatible_gem, h_incompatible_flowers]
  norm_num
  exact sorry

end valid_combinations_of_flowers_and_gems_l395_395112


namespace y_2_abs_x_valley_y_x_add_cos_x_not_valley_l395_395804

noncomputable def is_valley_function (f : ℝ → ℝ) (a b x0 : ℝ) : Prop :=
a < x0 ∧ x0 < b ∧ (∀ x, a ≤ x ∧ x ≤ x0 → f x > f (x + ε))
∧ (∀ x, x0 ≤ x ∧ x ≤ b → f x < f (x - ε))

theorem y_2_abs_x_valley :
  is_valley_function (λ x, 2 * |x|) (-1) 1 0 :=
sorry

theorem y_x_add_cos_x_not_valley :
  ¬(∃ a b x0, is_valley_function (λ x, x + Real.cos x) a b x0) :=
sorry

end y_2_abs_x_valley_y_x_add_cos_x_not_valley_l395_395804


namespace closure_of_multiplication_l395_395827

variables (S A B : Set ℝ)
variables [mul_closed : ∀ x y, x ∈ S → y ∈ S → x * y ∈ S]
variables (union_cond : S = A ∪ B) (disjoint_cond : A ∩ B = ∅)
variables (A_closed : ∀ a b c, a ∈ A → b ∈ A → c ∈ A → a * b * c ∈ A)
variables (B_closed : ∀ a b c, a ∈ B → b ∈ B → c ∈ B → a * b * c ∈ B)

theorem closure_of_multiplication :
  (∀ x y, x ∈ A → y ∈ A → x * y ∈ A) ∨ (∀ x y, x ∈ B → y ∈ B → x * y ∈ B) :=
sorry

end closure_of_multiplication_l395_395827


namespace find_a_b_l395_395670

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395670


namespace max_num_weeks_same_10_songs_l395_395555

theorem max_num_weeks_same_10_songs (n : ℕ) (songs : Fin n → ℕ) :
  (∀ i : Fin 10, ∀ j : Fin 10, i < j → songs i ≠ songs j) →
    (∀ k : ℕ, 0 < k → ∀ i : Fin 10,  ∀ j : Fin 10, i < j → songs j k < songs i k → False) →
    ∃ weeks : ℕ, weeks = 46 := 
by
  sorry

end max_num_weeks_same_10_songs_l395_395555


namespace binomial_expansion_coeff_l395_395543

theorem binomial_expansion_coeff : 
  ∀ (x y : ℝ), (coeff (expand_binomial (x + y) 10 6 4) = 210) :=
by
  sorry

end binomial_expansion_coeff_l395_395543


namespace pencils_per_box_l395_395727

theorem pencils_per_box (total_pencils : ℝ) (num_boxes : ℝ) (pencils_per_box : ℝ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4.0) 
  (h3 : pencils_per_box = total_pencils / num_boxes) : 
  pencils_per_box = 648 :=
by
  sorry

end pencils_per_box_l395_395727


namespace non_congruent_triangles_count_l395_395696

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def count_non_congruent_triangles : ℕ :=
  let a_values := [1, 2]
  let b_values := [2, 3]
  let triangles := [(1, 2, 2), (2, 2, 2), (2, 2, 3)]
  triangles.length

theorem non_congruent_triangles_count : count_non_congruent_triangles = 3 :=
  by
    -- Proof would go here
    sorry

end non_congruent_triangles_count_l395_395696


namespace leah_bought_boxes_l395_395293

def birdseed_problem : Prop :=
  ∃ (boxes_bought : ℕ),
  let boxes_in_pantry := 5,
      parrot_weekly := 100,
      cockatiel_weekly := 50,
      box_grams := 225,
      weeks := 12 in
  (parrot_weekly + cockatiel_weekly) * weeks / box_grams - boxes_in_pantry = boxes_bought ∧ 
  boxes_bought = 3

theorem leah_bought_boxes : birdseed_problem :=
begin
  sorry
end

end leah_bought_boxes_l395_395293


namespace problem_proof_l395_395664

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395664


namespace train_stop_time_per_hour_l395_395556

theorem train_stop_time_per_hour
    (v1 : ℕ) (v2 : ℕ)
    (h1 : v1 = 45)
    (h2 : v2 = 33) : ∃ (t : ℕ), t = 16 := by
  -- including the proof steps here is unnecessary, so we use sorry
  sorry

end train_stop_time_per_hour_l395_395556


namespace sum_first_50_equal_223_l395_395875

def periodic_sequence_sum (seq : ℕ → ℕ) (period : ℕ) (n : ℕ) : ℕ :=
  let sum_period := (List.sum $ List.map seq [0, 1, 2, 3, 4, 5])
  let periods := n / period
  let remaining := n % period
  periods * sum_period + (List.sum $ List.map seq $ List.range remaining)

def sequence (n : ℕ) : ℕ :=
  match n % 6 with
  | 0 => 3
  | 1 => 4
  | 2 => 5
  | 3 => 2
  | 4 => 6
  | _ => 7

theorem sum_first_50_equal_223 : periodic_sequence_sum sequence 6 50 = 223 :=
by
  sorry

end sum_first_50_equal_223_l395_395875


namespace midpoints_collinear_l395_395225

variables {Point : Type} [MetricSpace Point]

def midpoint (p1 p2 : Point) : Point := sorry 

variables {A1 B1 C1 A2 B2 C2 : Point}
variables (triangle1 : ∃ A1 B1 C1 : Point, congruent (A1, B1, C1) ∧ oriented_opposite (A1, B1, C1) (A2, B2, C2))

theorem midpoints_collinear :
  let M1 := midpoint A1 A2
  let M2 := midpoint B1 B2
  let M3 := midpoint C1 C2
  in collinear {M1, M2, M3} :=
sorry

end midpoints_collinear_l395_395225


namespace max_prob_win_two_consecutive_is_C_l395_395964

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l395_395964


namespace majority_of_votes_won_l395_395258

theorem majority_of_votes_won (p q : ℕ): 
    p = 60 → q = 6000 → (0.6 * q - 0.4 * q = 1200) :=
by
  intro h1 h2
  sorry

end majority_of_votes_won_l395_395258


namespace payment_for_C_l395_395929

/-- 
A can complete a job in 6 days with a daily wage of Rs 200.
B can complete the same job in 8 days with a daily wage of Rs 250.
A and B signed up to do the job together for Rs 2400 and completed the work in 3 days with the help of C, whose daily wage is unknown.
The total wage is divided based on each person's efficiency.
Given these conditions, prove that the payment for C is Rs 378.95.
-/
theorem payment_for_C (d1 d2 : ℝ) (w1 w2 p : ℝ) (days : ℝ) (daily_work_A daily_work_B : ℝ) (total_wage : ℝ) :
  d1 = 6 →
  w1 = 200 →
  d2 = 8 →
  w2 = 250 →
  p = 2400 →
  days = 3 →
  daily_work_A = 1 / d1 →
  daily_work_B = 1 / d2 →
  total_wage = p →
  ∃ (daily_work_C : ℝ), (daily_work_C = 1 / 8) →
    red : daily_work_C * days / (daily_work_A * days + daily_work_B * days + daily_work_C * days) * total_wage = 378.95 :=
begin
  sorry
end

end payment_for_C_l395_395929


namespace person_age_is_30_l395_395064

-- Definitions based on the conditions
def age (x : ℕ) := x
def age_5_years_hence (x : ℕ) := x + 5
def age_5_years_ago (x : ℕ) := x - 5

-- The main theorem to prove
theorem person_age_is_30 (x : ℕ) (h : 3 * age_5_years_hence x - 3 * age_5_years_ago x = age x) : x = 30 :=
by
  sorry

end person_age_is_30_l395_395064


namespace shauna_lowest_score_l395_395831

theorem shauna_lowest_score {t1 t2 t3 : ℕ} (h_t1 : t1 = 82) (h_t2 : t2 = 90) (h_t3 : t3 = 88)
    (max_points : ℕ := 100) (desired_average : ℕ := 85)
    (lower_bound : ℕ := 70) (upper_bound : ℕ := 85) :
  ∃ t4 t5 t6 : ℕ, t4 ≤ max_points ∧ t5 ≤ max_points ∧ t6 ≤ max_points ∧ 
  lower_bound ≤ t5 ∧ t5 ≤ upper_bound ∧ 
  (t1 + t2 + t3 + t4 + t5 + t6) / 6 = desired_average ∧ 
  (t4 = 65 ∨ t5 = 65 ∨ t6 = 65) :=
begin
  sorry
end

end shauna_lowest_score_l395_395831


namespace max_abs_diff_sum_proof_even_max_abs_diff_sum_proof_odd_max_arrangements_even_max_arrangements_odd_l395_395224

-- Define the vertices and the assignment of numbers
variable (n : ℕ)
variable (a : Fin n → ℕ)
variable (a_distinct : ∀ i j, a i = a j → i = j)
variable (a_range : ∀ i, 1 ≤ a i ∧ a i ≤ n)
variable (a1_eq_an1 : a 0 = a n)

noncomputable def max_abs_diff_sum (n : ℕ) (a : Fin n → ℕ) : ℕ :=
  let b (j : Fin n) := max (a j) (a (j + 1))
  let c (j : Fin n) := min (a j) (a (j + 1))
  ∑ j : Fin n, b j - ∑ j : Fin n, c j

theorem max_abs_diff_sum_proof_even (n : ℕ) [Fact (n % 2 = 0)]
  (a : Fin n → ℕ) : max_abs_diff_sum n a = n ^ 2 / 2 := by
  sorry

theorem max_abs_diff_sum_proof_odd (n : ℕ) [Fact (n % 2 = 1)]
  (a : Fin n → ℕ) : max_abs_diff_sum n a = (n ^ 2 - 1) / 2 := by
  sorry

theorem max_arrangements_even (n : ℕ) [Fact (n % 2 = 0)]
  (a : Fin n → ℕ) : 
  ∃ (k : ℕ), max_abs_diff_sum n a = n ^ 2 / 2 ∧ k = 2 * (factorial (n / 2)) ^ 2 := by
  sorry

theorem max_arrangements_odd (n : ℕ) [Fact (n % 2 = 1)]
  (a : Fin n → ℕ) : 
  ∃ (k : ℕ), max_abs_diff_sum n a = (n ^ 2 - 1) / 2 ∧ k = 2 * n * (factorial ((n - 1) / 2)) ^ 2 := by
  sorry

end max_abs_diff_sum_proof_even_max_abs_diff_sum_proof_odd_max_arrangements_even_max_arrangements_odd_l395_395224


namespace find_a_b_l395_395612

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395612


namespace units_digit_17_times_29_l395_395164

theorem units_digit_17_times_29 :
  (17 * 29) % 10 = 3 :=
by
  sorry

end units_digit_17_times_29_l395_395164


namespace yz_zx_xy_minus_2xyz_leq_7_27_l395_395207

theorem yz_zx_xy_minus_2xyz_leq_7_27 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  (y * z + z * x + x * y - 2 * x * y * z) ≤ 7 / 27 := 
by 
  sorry

end yz_zx_xy_minus_2xyz_leq_7_27_l395_395207


namespace max_AD_l395_395779

/-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
Given that ∠A = π/3 and a = 4, and point D is the midpoint of side BC, 
then the maximum value of AD is 2√3. -/
theorem max_AD (A B C D : Type)
  (a b c AD : ℝ)
  (angle_A : ℝ)
  (h1 : angle_A = π / 3)
  (h2 : a = 4)
  (h3 : D = (B + C) / 2) :
  AD ≤ 2 * real.sqrt 3 :=
sorry

end max_AD_l395_395779


namespace vector_subtraction_l395_395226

def c : ℝ × ℝ × ℝ := (5, -3, 2)
def d : ℝ × ℝ × ℝ := (-2, 1, 3)

theorem vector_subtraction : 
  let r : ℝ × ℝ × ℝ := (13, -7, -10)
  in c - 4 • d = r := by
sorry

end vector_subtraction_l395_395226


namespace ratio_LM_AH_ratio_AL_MH_l395_395479

noncomputable def ratio_area (FLM AFH : ℝ) : ℝ :=
  FLM / AFH

noncomputable def ratio_area2 (AFM FHL : ℝ) : ℝ :=
  AFM / FHL

theorem ratio_LM_AH (L M F A H : Point) (FLM AFH : ℝ) (h : ratio_area FLM AFH = 49 / 9) : 
  ratio (distance L M) (distance A H) = 7 / 3 :=
sorry

theorem ratio_AL_MH (L M F A H : Point) (AFM FHL : ℝ) (h1 : ratio_area FLM AFH = 49 / 9) (h2 : ratio_area2 AFM FHL = 1 / 4) : 
  ratio (distance L A) (distance H M) = 11 :=
sorry

end ratio_LM_AH_ratio_AL_MH_l395_395479


namespace sum_of_polynomials_zero_l395_395677

/-- We define the problem as a Lean theorem. -/
theorem sum_of_polynomials_zero (P : Fin 2019 → Polynomial ℝ) 
  (h_deg : ∀ i, (P i).degree ≤ 2018)
  (h_no_common_roots : ∀ i j (h_diff : i ≠ j), ∀ x, ¬((P i).root x ∧ (P j).root x))
  (h_common_root : ∀ i, ∃ x, (P i).root x ∧ (∑ j in Finset.univ \ {i}, P j).root x) :
  ∑ i, P i = 0 :=
by
  sorry

end sum_of_polynomials_zero_l395_395677


namespace democrat_support_for_A_l395_395751

-- Definitions of the conditions
variable (V : ℝ) -- Total number of registered voters
variable (pDem : ℝ) := 0.60 -- Percent of Democrats among total voters
variable (pRep : ℝ) := 0.40 -- Percent of Republicans among total voters
variable (pRepA : ℝ) := 0.20 -- Percent of Republicans voting for candidate A
variable (pTotalA : ℝ) := 0.59 -- Percent of total voters voting for candidate A

-- The function representing percent of Democrats voting for candidate A
def percentageDemVote (D : ℝ) : Prop :=
  (D * pDem * V + pRepA * pRep * V = pTotalA * V)

-- The proof statement
theorem democrat_support_for_A (D : ℝ) (h : percentageDemVote V D) : D = 0.85 :=
by
  -- proof. sorry as placeholder
  sorry

end democrat_support_for_A_l395_395751


namespace Tim_pays_correct_amount_l395_395879

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end Tim_pays_correct_amount_l395_395879


namespace polyhedron_space_diagonals_l395_395086

theorem polyhedron_space_diagonals (vertices edges faces : ℕ) 
  (triangular_faces quadrilateral_faces : ℕ) :
  vertices = 30 → 
  edges = 72 → 
  faces = 44 → 
  triangular_faces = 30 → 
  quadrilateral_faces = 14 → 
  (∑ i in (range vertices).filter (λ x, x ≠ i), i) - edges - 
  (triangular_faces * 0 + quadrilateral_faces * 2) = 335 := 
by 
  intros h_vertices h_edges h_faces h_tri_faces h_quad_faces 
  sorry

end polyhedron_space_diagonals_l395_395086


namespace problem_l395_395219

noncomputable def f (x θ : ℝ) : ℝ :=
  2 * sin x * (cos (θ / 2))^2 + cos x * sin θ - sin x

theorem problem
  (a b c A B C θ : ℝ)
  (h1 : ∀ x, f x θ ≥ f π θ)
  (h2 : 0 < θ ∧ θ < π)
  (h3 : a = 1)
  (h4 : b = sqrt 2)
  (h5 : f A θ = sqrt 3 / 2)
  : θ = π / 2 ∧ (C = 7 * π / 12 ∨ C = π / 12) :=
begin
  sorry
end

end problem_l395_395219


namespace isosceles_triangle_degrees_acute_l395_395478

variables {A B C D E F : Type}
variables [linear_ordered_field A] [comm_ring B] [add_comm_group C] [vector_space A B] [inner_product_space A C] [finite_dimensional A C]

-- Given definitions for isosceles triangle and incircle tangency
def isosceles_triangle (A B C : X) : Prop := 
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ dist A B = dist A C

def incircle_tangency
  (A B C D E F : X)
  (tangent_BC : X → Prop)
  (tangent_CA : X → Prop)
  (tangent_AB : X → Prop) : Prop :=
  tangent_BC D ∧ tangent_CA E ∧ tangent_AB F

-- Angles of the triangle DEF
def acute_angles (A B C : X) (angle_DEF : X → ℝ) : Prop :=
  angle_DEF D F E < 90 ∧ angle_DEF F E D < 90 ∧ angle_DEF E D F < 90

-- Lean statement to state the theorem
theorem isosceles_triangle_degrees_acute
  (A B C D E F : X)
  (tangent_BC tangent_CA tangent_AB : X → Prop)
  (angle : X × X × X → ℝ) :
  isosceles_triangle A B C →
  incircle_tangency A B C D E F tangent_BC tangent_CA tangent_AB →
  acute_angles D E F (angle D E F) :=
  sorry

end isosceles_triangle_degrees_acute_l395_395478


namespace gcd_sum_is_12_l395_395525

theorem gcd_sum_is_12 : ∑ k in {1, 2, 3, 6} : Finset ℕ, gcd k 6 = 12 := by
  sorry

end gcd_sum_is_12_l395_395525


namespace train_meets_john_l395_395518

open MeasureTheory

noncomputable def john_meets_train_probability : ℝ :=
  let train_arrival := measure_space.mk (Set.Icc 0 60) (by apply_instance)
  let john_arrival := measure_space.mk (Set.Icc 0 120) (by apply_instance)
  let train_waits := 10
  
  let total_area := (120:ℝ) * 60
  let intersection_area := 0.5 * 60 * 10
  (intersection_area / total_area)

theorem train_meets_john :
  john_meets_train_probability = 1 / 24 :=
by 
  unfold john_meets_train_probability
  simp
  norm_num
  sorry

end train_meets_john_l395_395518


namespace combined_function_is_linear_l395_395094

def original_parabola (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5

def reflected_parabola (x : ℝ) : ℝ := -original_parabola x

def translated_original_parabola (x : ℝ) : ℝ := 3 * (x - 4)^2 + 4 * (x - 4) - 5

def translated_reflected_parabola (x : ℝ) : ℝ := -3 * (x + 6)^2 - 4 * (x + 6) + 5

def combined_function (x : ℝ) : ℝ := translated_original_parabola x + translated_reflected_parabola x

theorem combined_function_is_linear : ∃ (a b : ℝ), ∀ x : ℝ, combined_function x = a * x + b := by
  sorry

end combined_function_is_linear_l395_395094


namespace least_possible_value_of_b_l395_395843

theorem least_possible_value_of_b (a b : ℕ) 
  (ha : ∃ p, (∀ q, p ∣ q ↔ q = 1 ∨ q = p ∨ q = p*p ∨ q = a))
  (hb : ∃ k, (∀ l, k ∣ l ↔ (l = 1 ∨ l = b)))
  (hdiv : a ∣ b) : 
  b = 12 :=
sorry

end least_possible_value_of_b_l395_395843


namespace problem_proof_l395_395661

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395661


namespace percent_profit_l395_395513

theorem percent_profit (CP LP SP Profit : ℝ) 
  (hCP : CP = 100) 
  (hLP : LP = CP + 0.30 * CP)
  (hSP : SP = LP - 0.10 * LP) 
  (hProfit : Profit = SP - CP) : 
  (Profit / CP) * 100 = 17 :=
by
  sorry

end percent_profit_l395_395513


namespace sine_gamma_half_leq_c_over_a_plus_b_l395_395341

variable (a b c : ℝ) (γ : ℝ)

-- Consider a triangle with sides a, b, c, and angle γ opposite to side c.
-- We need to prove that sin(γ / 2) ≤ c / (a + b).
theorem sine_gamma_half_leq_c_over_a_plus_b (h_c_pos : 0 < c) 
  (h_g_angle : 0 < γ ∧ γ < 2 * π) : 
  Real.sin (γ / 2) ≤ c / (a + b) := 
  sorry

end sine_gamma_half_leq_c_over_a_plus_b_l395_395341


namespace rationalize_sqrt_expression_l395_395822

theorem rationalize_sqrt_expression : 
    sqrt (5 / (2 + sqrt 2)) = sqrt 5 - sqrt 10 / 2 := by
  sorry

end rationalize_sqrt_expression_l395_395822


namespace math_problem_proof_l395_395922

-- Given circle C1 in rectangular coordinates
def circle_C1 : Prop := ∀ x y : ℝ, (x - 2)^2 + (y - 4)^2 = 20

-- Polar coordinate system
def polar_system := ∀ ρ θ : ℝ, circle_C1 (ρ * cos θ) (ρ * sin θ)

-- Curve C2 in polar coordinates
def curve_C2_in_polar : Prop := ∀ ρ : ℝ, θ = π / 3

-- Curve C2 in rectangular coordinates
def curve_C2_in_rectangular : Prop := ∀ x y : ℝ, y = sqrt 3 * x

-- Line C3 in polar coordinates
def line_C3_in_polar : Prop := ∀ ρ : ℝ, θ = π / 6

-- Intersection points and area calculation
def area_triangle_OMN: Prop :=
  let M := (2 + 4 * sqrt 3)
  let N := (4 + 2 * sqrt 3)
  ∀ O M N : ℝ, 
  ∠MON = π / 6 ∧ 
  (1 / 2) * |M| * |N| * sin (π / 6) = 8 + 5 * sqrt 3  

-- The Lean 4 theorem statement (all conditions)
theorem math_problem_proof :
  circle_C1 →
  polar_system →
  curve_C2_in_polar →
  curve_C2_in_rectangular →
  line_C3_in_polar →
  area_triangle_OMN :=
by sorry

end math_problem_proof_l395_395922


namespace product_of_five_consecutive_integers_not_square_l395_395357

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l395_395357


namespace down_payment_amount_l395_395281

-- Define the monthly savings per person
def monthly_savings_per_person : ℤ := 1500

-- Define the number of people
def number_of_people : ℤ := 2

-- Define the total monthly savings
def total_monthly_savings : ℤ := monthly_savings_per_person * number_of_people

-- Define the number of years they will save
def years_saving : ℤ := 3

-- Define the number of months in a year
def months_in_year : ℤ := 12

-- Define the total number of months
def total_months : ℤ := years_saving * months_in_year

-- Define the total savings needed for the down payment
def total_savings_needed : ℤ := total_monthly_savings * total_months

-- Prove that the total amount needed for the down payment is $108,000
theorem down_payment_amount : total_savings_needed = 108000 := by
  -- This part requires a proof, which we skip with sorry
  sorry

end down_payment_amount_l395_395281


namespace total_cows_in_herd_l395_395983

theorem total_cows_in_herd {n : ℚ} (h1 : 1/3 + 1/6 + 1/9 = 11/18) 
                           (h2 : (1 - 11/18) = 7/18) 
                           (h3 : 8 = (7/18) * n) : 
                           n = 144/7 :=
by sorry

end total_cows_in_herd_l395_395983


namespace quadratic_eq_with_root_sqrt5_min3_l395_395150

theorem quadratic_eq_with_root_sqrt5_min3 :
  ∃ (b c : ℚ), (∀ x : ℚ, x^2 + b * x + c = 0 -> x = (√5 - 3: ℚ) ∨ x = (-√5 - 3: ℚ)) ∧ (b = 6) ∧ (c = -4) :=
by
  sorry

end quadratic_eq_with_root_sqrt5_min3_l395_395150


namespace second_largest_element_possibilities_l395_395490

theorem second_largest_element_possibilities :
  ∃ f : ℕ → ℕ, (∀ n, 0 ≤ f n ∧ f n ≤ 11) ∧ 
    ( ∀ (a : list ℕ), a.length = 5 ∧ 
      (14 : ℚ) = (a.sum : ℚ) / 5 ∧ 
      (a.last - a.head) = 20 ∧ 
      (mode a = 10) ∧ 
      (median a = 10) → 
    (card (finset.image f (finset.range 11)) = 11)) :=
sorry

end second_largest_element_possibilities_l395_395490


namespace proof_by_contradiction_l395_395014

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

end proof_by_contradiction_l395_395014


namespace triangle_isosceles_AUV_l395_395183

theorem triangle_isosceles_AUV
  (A B C X Y U V O1 O2 : Point)
  (h1 : Collinear B X C)
  (h2 : Collinear C Y)
  (h3 : B ≠ X)
  (h4 : B ≠ Y)
  (h5 : BX * AC = CY * AB)
  (h6 : Circumcenter ACX = O1)
  (h7 : Circumcenter ABY = O2)
  (h8 : Line O1 O2 intersects Line AB at U)
  (h9 : Line O1 O2 intersects Line AC at V)
  : Isosceles_triangle A U V :=
begin
  sorry,
end

end triangle_isosceles_AUV_l395_395183


namespace problem_proof_l395_395659

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395659


namespace find_A_l395_395874

theorem find_A (A B : ℕ) (h1 : 10 * A + 7 + (30 + B) = 73) : A = 3 := by
  sorry

end find_A_l395_395874


namespace bondholder_no_win_probability_l395_395924

theorem bondholder_no_win_probability :
  ∀ (total_bonds winning_bonds draws1 draws2 draws3 : ℕ),
    total_bonds = 1000 →
    winning_bonds = 100 →
    draws1 = 3 →
    draws2 = 3 →
    draws3 = 2 →
    P(no_win) = (9 / 10) ^ 7
  :=
by
  intros total_bonds winning_bonds draws1 draws2 draws3 hv1 hv2 hv3 hv4 hv5,
  sorry

end bondholder_no_win_probability_l395_395924


namespace mod_sum_l395_395567

theorem mod_sum : 
  (5432 + 5433 + 5434 + 5435) % 7 = 2 := 
by
  sorry

end mod_sum_l395_395567


namespace sum_of_b_l395_395137

theorem sum_of_b (b1 b2 : ℝ) 
  (h1 : 3 * b1^2 + b1 * (4+ 6 * b1) + 6 * b1 + 4 = 0)
  (h2 : 3 * b2^2 + b2 * (4+ 6 * b2) + 6 * b2 + 4 = 0)
  (h3 : (b1 + 6)^2 = 48)
  (h4 : (b2 + 6)^2 = 48) :
  b1 + b2 = -12 :=
begin
  sorry,
end

end sum_of_b_l395_395137


namespace solve_abs_eq_l395_395364

theorem solve_abs_eq (x : ℝ) : 
  (|x - 4| + 3 * x = 12) ↔ (x = 4) :=
by
  sorry

end solve_abs_eq_l395_395364


namespace jamie_dimes_l395_395784

theorem jamie_dimes (y : ℕ) (h : 5 * y + 10 * y + 25 * y = 1440) : y = 36 :=
by 
  sorry

end jamie_dimes_l395_395784


namespace quadrilateral_rhombus_opposite_centers_eq_angle_l395_395333

open EuclideanGeometry -- Assuming Euclidean geometry is the relevant context.

theorem quadrilateral_rhombus_opposite_centers_eq_angle {A B C D : Point}
  (O1 O2 O3 O4 M : Point)
  (α : Angle)
  (h_conv_quad : ConvexQuadrilateral A B C D)
  (h_sim_rhombs : SimilarRhombusesOnSides A B C D O1 O2 O3 O4 α)
  (h_midpoint : Midpoint M A C) :
  Distance O1 O3 = Distance O2 O4 ∧ Angle O1 M O3 = α ∧ Angle O2 M O4 = α := 
sorry

end quadrilateral_rhombus_opposite_centers_eq_angle_l395_395333


namespace seq_is_constant_l395_395681

noncomputable def a_seq : ℕ → ℝ
| 1 := 1
| 2 := 1
| (n+1) := if n > 1 then (n^2 * (a_seq n)^2 + 5) / ((n^2 - 1) * (a_seq (n-1))) else 1  -- to handle the starting terms

def b_seq (x y : ℝ) : ℕ → ℝ
| n := (n * x + y) * (a_seq n)

theorem seq_is_constant (x y : ℝ) (h : x ≠ 0) :
  ∃ c : ℝ, ∀ n : ℕ, n ≥ 2 → (b_seq x y (n+2) + b_seq x y n) / b_seq x y (n+1) = c :=
begin
  use 2,  -- assumption of c = 2 based on the given problem
  sorry  -- proof steps to demonstrate that x = 1 and y = 0 makes the sequence constant
end

end seq_is_constant_l395_395681


namespace sum_reciprocal_squares_le_two_l395_395918

theorem sum_reciprocal_squares_le_two (n : ℕ) (hn : 1 ≤ n) : 
  (∑ k in finset.range(n+1), 1 / (k+1 : ℝ)^2) ≤ 2 := by
  sorry

end sum_reciprocal_squares_le_two_l395_395918


namespace find_m_l395_395214

theorem find_m (m : ℝ) (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : m = -3 := by
  sorry

end find_m_l395_395214


namespace sum_of_positive_integers_l395_395045

theorem sum_of_positive_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 272) : x + y = 32 := 
by 
  sorry

end sum_of_positive_integers_l395_395045


namespace find_a_b_l395_395613

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395613


namespace find_a_b_l395_395667

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395667


namespace maximize_prob_of_consecutive_wins_l395_395946

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l395_395946


namespace find_a_and_b_l395_395593

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395593


namespace linda_total_distance_l395_395320

theorem linda_total_distance :
  ∃ x : ℕ, (60 % x = 0) ∧ ((75 % (x + 3)) = 0) ∧ ((90 % (x + 6)) = 0) ∧
  (60 / x + 75 / (x + 3) + 90 / (x + 6) = 15) :=
sorry

end linda_total_distance_l395_395320


namespace division_result_l395_395903

theorem division_result (k q : ℕ) (h₁ : k % 81 = 11) (h₂ : 81 > 0) : k / 81 = q + 11 / 81 :=
  sorry

end division_result_l395_395903


namespace prism_faces_l395_395229

theorem prism_faces (E V F n : ℕ) (h1 : E + V = 30) (h2 : F + V = E + 2) (h3 : E = 3 * n) : F = 8 :=
by
  -- Actual proof omitted
  sorry

end prism_faces_l395_395229


namespace chess_player_max_consecutive_win_prob_l395_395934

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l395_395934


namespace number_of_meetings_l395_395423

-- Define the data for the problem
def pool_length : ℕ := 120
def swimmer_A_speed : ℕ := 4
def swimmer_B_speed : ℕ := 3
def total_time_seconds : ℕ := 15 * 60
def swimmer_A_turn_break_seconds : ℕ := 2
def swimmer_B_turn_break_seconds : ℕ := 0

-- Define the round trip time for each swimmer
def swimmer_A_round_trip_time : ℕ := 2 * (pool_length / swimmer_A_speed) + 2 * swimmer_A_turn_break_seconds
def swimmer_B_round_trip_time : ℕ := 2 * (pool_length / swimmer_B_speed) + 2 * swimmer_B_turn_break_seconds

-- Define the least common multiple of the round trip times
def lcm_round_trip_time : ℕ := Nat.lcm swimmer_A_round_trip_time swimmer_B_round_trip_time

-- Define the statement to prove
theorem number_of_meetings (lcm_round_trip_time : ℕ) : 
  (24 * (total_time_seconds / lcm_round_trip_time) + ((total_time_seconds % lcm_round_trip_time) / (pool_length / (swimmer_A_speed + swimmer_B_speed)))) = 51 := 
sorry

end number_of_meetings_l395_395423


namespace x_intercept_perpendicular_line_l395_395429

theorem x_intercept_perpendicular_line (a b c : ℝ) (h1 : a = 3) (h2 : b = -2) (h3 : c = 6)
  (y_intercept : ℝ) (h4 : y_intercept = 2) : 
  let slope := -((a / b)⁻¹)
  let intercept := y_intercept
  let x_intercept := (c - b * intercept) / a in 
  x_intercept = 3 :=
by
  sorry

end x_intercept_perpendicular_line_l395_395429


namespace find_inclination_angle_l395_395857

theorem find_inclination_angle :
    (∃ α : ℝ, 0 ≤ α ∧ α < π ∧ (∀ x y : ℝ, x + sqrt 3 * y + 2 = 0 → tan α = - (sqrt 3) / 3)) →
    ∃ α : ℝ, α = 5 * π / 6 := 
sorry

end find_inclination_angle_l395_395857


namespace prob_obtuse_angle_FQG_l395_395816

open Real

-- Definitions of the vertices
def F := (0:ℝ, 3:ℝ)
def G := (5:ℝ, 0:ℝ)
def H := (5 + π, 0:ℝ)
def I := (0:ℝ, 3 + π)

-- Define the function to determine the probability
noncomputable def prob_obtuse (Q : ℝ × ℝ) : ℝ :=
  if (Q.1 ≥ 0 ∧ Q.1 ≤ 5 + π) ∧ (Q.2 ≥ 0 ∧ Q.2 ≤ 3 + π) then 
    let area_rect := (5 + π) * (3 + π)
    let area_semi := (π / 4) * (π^2 + 10 * π + 34)
    area_semi / area_rect 
  else 
    0

-- Lean statement to prove the required probability
theorem prob_obtuse_angle_FQG : ∀ Q : ℝ × ℝ,
  prob_obtuse Q = if (Q.1 ≥ 0 ∧ Q.1 ≤ 5 + π) ∧ (Q.2 ≥ 0 ∧ Q.2 ≤ 3 + π) then
                    π * (π^2 + 10 * π + 34) / (60 + 32 * π + 4 * π^2)
                  else
                    0 := 
sorry

end prob_obtuse_angle_FQG_l395_395816


namespace factorial_mod_13_l395_395575

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_mod_13 : factorial 7 % 13 = 11 := by
  -- The detailed proof is skipped
  sorry

end factorial_mod_13_l395_395575


namespace number_of_true_propositions_l395_395315

variables (a b : ℝ)

def proposition_p : Prop := (a = b) → (cos a = cos b)
def proposition_converse_p : Prop := (cos a = cos b) → (a = b)
def proposition_inverse_p : Prop := (a ≠ b) → (cos a ≠ cos b)
def proposition_contrapositive_p : Prop := (cos a ≠ cos b) → (a ≠ b)

theorem number_of_true_propositions : 
  (proposition_p a b ∧ proposition_contrapositive_p a b ∧ ¬proposition_converse_p a b ∧ ¬proposition_inverse_p a b) → 
  2 = 2 :=
by sorry

end number_of_true_propositions_l395_395315


namespace find_length_VD_l395_395263

theorem find_length_VD
  (XYZW : Π (X Y Z W L M N U V : Type), Prop)
  (rect : Π (X Y Z W : Type), rect X Y Z W)
  (L_on_XW : Π (X W L: Type), L ∈ XW)
  (angle_XLZ_90 : Π (X W L Z : Type), is_right_angle ∠XLZ)
  (UV_perp_XW : Π (X W U V : Type), is_perpendicular UV XW)
  (XU_eq_UV : Π (X U V : Type), XU = UV)
  (LZ_intersects_UV_at_M : Π (L Z U V M : Type), intersects LZ UV M)
  (N_on_ZW_NX_passes_through_M : Π (N Z W M : Type), N ∈ ZW → NX passes_through M)
  (LX_eq_24 : Π (L X : Type), length L X = 24)
  (MX_eq_30 : Π (M X : Type), length M X = 30)
  (ML_eq_18 : Π (M L : Type), length M L = 18) :
  length V D = 18 / 5 :=
by sorry

end find_length_VD_l395_395263


namespace banana_cream_pie_angle_l395_395746

noncomputable def total_students := 48
def chocolate_pie_students := 15
def apple_pie_students := 9
def blueberry_pie_students := 11

theorem banana_cream_pie_angle : 
    let remaining_students := total_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students) in
    let banana_cream_pie_students := remaining_students / 2 in
    (banana_cream_pie_students / total_students: ℝ) * 360 = 45 :=
by
  let remaining_students := total_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
  let banana_cream_pie_students := remaining_students / 2
  have h: (banana_cream_pie_students: ℝ) = 6 := by sorry
  rw [h]
  simp
  sorry

end banana_cream_pie_angle_l395_395746


namespace determine_good_numbers_l395_395992

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), (∀ k : Fin n, ∃ m : ℕ, k.1 + (a k).1 + 1 = m * m)

theorem determine_good_numbers :
  is_good_number 13 ∧ is_good_number 15 ∧ is_good_number 17 ∧ is_good_number 19 ∧ ¬is_good_number 11 :=
by
  sorry

end determine_good_numbers_l395_395992


namespace find_a_b_l395_395626

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395626


namespace parallelogram_area_l395_395066

-- Definitions
def base_cm : ℕ := 22
def height_cm : ℕ := 21

-- Theorem statement
theorem parallelogram_area : base_cm * height_cm = 462 := by
  sorry

end parallelogram_area_l395_395066


namespace teacher_age_is_94_5_l395_395449

noncomputable def avg_age_students : ℝ := 18
noncomputable def num_students : ℝ := 50
noncomputable def avg_age_class_with_teacher : ℝ := 19.5
noncomputable def num_total : ℝ := 51

noncomputable def total_age_students : ℝ := num_students * avg_age_students
noncomputable def total_age_class_with_teacher : ℝ := num_total * avg_age_class_with_teacher

theorem teacher_age_is_94_5 : ∃ T : ℝ, total_age_students + T = total_age_class_with_teacher ∧ T = 94.5 := by
  sorry

end teacher_age_is_94_5_l395_395449


namespace verify_conclusions_l395_395528

noncomputable def P_ball_drawn_from_bag_B_red (P_A1 P_A2 P_A3 P_B_given_A1 P_B_given_A2 P_B_given_A3 : ℚ) : ℚ :=
P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

axiom bag_A_conditions : ∀ (P_A1 P_A2 P_A3 : ℚ),
P_A1 = 5 / 10 ∧ P_A2 = 2 / 10 ∧ P_A3 = 3 / 10

axiom bag_B_prob_given_A1 : P_ball_drawn_from_bag_B_red 5 / 10 2 / 10 3 / 10 5 / 11 4 / 11 4 / 11 = 9 / 22

axiom mutually_exclusive_events (A1 A2 A3 : Prop) : 
     (A1 ∧ ¬ A2 ∧ ¬ A3) ∨ (¬ A1 ∧ A2 ∧ ¬ A3) ∨ (¬ A1 ∧ ¬ A2 ∧ A3)

theorem verify_conclusions : 
∀ (P_A1 P_A2 P_A3 : ℚ),
P_ball_drawn_from_bag_B_red P_A1 P_A2 P_A3 5 / 11 4 / 11 4 / 11 = 9 / 22 ∧ 
mutually_exclusive_events (P_A1 > 0) (P_A2 > 0) (P_A3 > 0) :=
sorry

end verify_conclusions_l395_395528


namespace no_consecutive_beeches_probability_l395_395486

theorem no_consecutive_beeches_probability :
  let total_trees := 12
  let oaks := 3
  let holm_oaks := 4
  let beeches := 5
  let total_arrangements := (Nat.factorial total_trees) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks) * (Nat.factorial beeches))
  let favorable_arrangements :=
    let slots := oaks + holm_oaks + 1
    Nat.choose slots beeches * ((Nat.factorial (oaks + holm_oaks)) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks)))
  let probability := favorable_arrangements / total_arrangements
  probability = 7 / 99 :=
by
  sorry

end no_consecutive_beeches_probability_l395_395486


namespace vincent_total_cost_l395_395444

theorem vincent_total_cost :
  let day1_packs := 15
  let day1_pack_cost := 2.50
  let discount_percent := 0.10
  let day2_packs := 25
  let day2_pack_cost := 3.00
  let tax_percent := 0.05
  let day1_total_cost_before_discount := day1_packs * day1_pack_cost
  let day1_discount_amount := discount_percent * day1_total_cost_before_discount
  let day1_total_cost_after_discount := day1_total_cost_before_discount - day1_discount_amount
  let day2_total_cost_before_tax := day2_packs * day2_pack_cost
  let day2_tax_amount := tax_percent * day2_total_cost_before_tax
  let day2_total_cost_after_tax := day2_total_cost_before_tax + day2_tax_amount
  let total_cost := day1_total_cost_after_discount + day2_total_cost_after_tax
  total_cost = 112.50 :=
by 
  -- Mathlib can be used for floating point calculations, if needed
  -- For the purposes of this example, we assume calculations are correct.
  sorry

end vincent_total_cost_l395_395444


namespace ways_to_place_letters_l395_395144

theorem ways_to_place_letters :
  let grid_size := 4
  let letter_count := 2
  let total_ways := 3960
  (∃! (place : fin grid_size → fin grid_size → option char), 
    (∀ i j, place i j = some 'a' ∨ place i j = some 'b' ∨ place i j = none) ∧
    (∀ i, (∃! j, place i j = some 'a') ∧ (∃! j, place i j = some 'b')) ∧
    (∀ j, (∃! i, place i j = some 'a') ∧ (∃! i, place i j = some 'b'))
  ) = total_ways := sorry

end ways_to_place_letters_l395_395144


namespace tangent_line_sum_l395_395127

theorem tangent_line_sum :
  let O : Point := (0, 0)
  let omega : Circle := { center := O, radius := 7 }
  let A : Point := (15, 0) -- Using coordinates to define points
  let B := -- point of tangency of the tangent from A to the circle
  let C := -- point of tangency of the tangent from A to the circle
  let T_3 := -- point of tangency where BC touches the circle
  (distance O A = 15) -- OA = 15
  (distance B T_3 + distance T_3 C = 10) -- BC = 10
  (AB + AC = 8 * Real.sqrt 11 - 10) := sorry

end tangent_line_sum_l395_395127


namespace b_2017_equals_1_l395_395373

def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| n := fibonacci (n - 1) + fibonacci (n - 2)

def b (n : ℕ) : ℕ := (fibonacci n) % 3

def is_periodic (seq : ℕ → ℕ) (p : ℕ) : Prop :=
∀ n, seq (n + p) = seq n

theorem b_2017_equals_1 : b 2017 = 1 :=
  sorry

end b_2017_equals_1_l395_395373


namespace regular_even_polygon_dissection_l395_395342

variable {n : ℕ}

structure Lozenge where
  sides : ℝ 
  equal_sides : ∀ i j, i ≠ j → sides i = sides j

-- Defining a regular 2n-gon
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  equal_sides : ∀ i j, i ≠ j → sides i = sides j
  even_sides : even sides

-- Main proposition that needs to be proved
noncomputable def can_be_dissected_into_lozenges (P : RegularPolygon n) : Prop :=
  ∀ n, n > 1 → P.sides = 2 * n → ∃ L : list Lozenge, ∀ l ∈ L, is_lozenge l

theorem regular_even_polygon_dissection (P : RegularPolygon n) :
  can_be_dissected_into_lozenges P :=
  sorry

end regular_even_polygon_dissection_l395_395342


namespace probability_X_Y_Z_problems_l395_395909

-- Define the success probabilities for Problem A
def P_X_A : ℚ := 1 / 5
def P_Y_A : ℚ := 1 / 2

-- Define the success probabilities for Problem B
def P_Y_B : ℚ := 3 / 5

-- Define the negation of success probabilities for Problem C
def P_Y_not_C : ℚ := 5 / 8
def P_X_not_C : ℚ := 3 / 4
def P_Z_not_C : ℚ := 7 / 16

-- State the final probability theorem
theorem probability_X_Y_Z_problems :
  P_X_A * P_Y_A * P_Y_B * P_Y_not_C * P_X_not_C * P_Z_not_C = 63 / 2048 := 
sorry

end probability_X_Y_Z_problems_l395_395909


namespace find_yellow_balls_l395_395081

def total_balls := 60
def white_balls := 22
def green_balls := 18
def red_balls := 6
def purple_balls := 9
def non_red_purple_probability := 0.75
def total_non_red_purple_balls := (non_red_purple_probability * total_balls).toInt

theorem find_yellow_balls:
  ∃ Y : ℕ, total_balls = white_balls + green_balls + red_balls + purple_balls + Y ∧
    total_non_red_purple_balls = white_balls + green_balls + Y ∧
    Y = 5 :=
by
  sorry

end find_yellow_balls_l395_395081


namespace num_equidistant_points_l395_395129

-- Define the geometric setup
variables {O : Point} {r : ℝ} (circle : Circle O r)
variables {T₁ T₂ T₃ : Line}

-- Given conditions
axiom tangent_1_distance : T₁.dist O = r
axiom tangent_2_distance : T₂.dist O = r
axiom tangent_3_distance : T₃.dist O = r + 3
axiom parallel_tangents : T₁.parallel T₂ ∧ T₂.parallel T₃

-- Main statement to prove
theorem num_equidistant_points : (∃! P : Point, P.equidistant_from circle ∧ P.equidistant_from T₁ ∧ P.equidistant_from T₂ ∧ P.equidistant_from T₃) :=
  sorry

end num_equidistant_points_l395_395129


namespace simplify_fraction_l395_395454

theorem simplify_fraction (a b : ℕ) (h1 : a = 252) (h2 : b = 248) :
  (1000 ^ 2 : ℤ) / ((a ^ 2 - b ^ 2) : ℤ) = 500 := by
  sorry

end simplify_fraction_l395_395454


namespace proof_problem_l395_395006

-- Definitions of given conditions
variable (x : ℝ) -- monthly processing volume in tons
variable (y : ℝ) -- monthly processing cost in yuan

-- Conditions:
def processing_capacity (x : ℝ) : Prop := 400 ≤ x ∧ x ≤ 600
def cost_function (x : ℝ) : ℝ := x^2 - 200 * x + 80000
def product_value_per_ton : ℝ := 100
def profit (x : ℝ) : ℝ := 100 * x - cost_function x

-- Questions and correct answers:
def lowest_avg_cost_volume : Prop := x = 400
def required_subsidy : Prop := ∀ x, processing_capacity x → profit x ≤ 0 → (profit x + 40000 >= 0)

-- The main theorem that encodes the problem:
theorem proof_problem :
  (∀ x, processing_capacity x → (cost_function x / x).val  ≤ (cost_function 400 / 400).val)
  ∧ required_subsidy :=
by
  sorry

end proof_problem_l395_395006


namespace beetle_can_always_return_to_start_l395_395252

structure Cell :=
(x : Int)
(y : Int)

structure Door :=
(from : Cell)
(to : Cell)
(opened : Bool)
(direction : Bool) -- true means the door is open from 'from' to 'to'

structure Beetle :=
(position : Cell)
(travelled_positions : List Cell)
(interacted_doors : List Door)

def initial_beetle_position (b : Beetle) : Cell := b.travelled_positions.headD b.position

def can_return_to_start (b : Beetle) : Prop :=
  ∃ p ∈ b.travelled_positions, p = initial_beetle_position b

theorem beetle_can_always_return_to_start (b : Beetle) (initial : Cell)
  (h_start : b.position = initial)
  (h_doors : ∀ c1 c2, adj_by_side c1 c2 ←? Bool ⌦ Know that "adj_by_side" checks if two cells are adjacent by side.
  (h_travel : ∀ d ∈ b.interacted_doors, d.from = b.position ∨ d.to = b.position):
  can_return_to_start b :=
sorry

end beetle_can_always_return_to_start_l395_395252


namespace quadratic_root_range_l395_395000

/-- 
  Define the quadratic function y = ax^2 + bx + c for given values.
  Show that there exists x_1 in the interval (-1, 0) such that y = 0.
-/
theorem quadratic_root_range {a b c : ℝ} (h : a ≠ 0) 
  (h_minus3 : a * (-3)^2 + b * (-3) + c = -11)
  (h_minus2 : a * (-2)^2 + b * (-2) + c = -5)
  (h_minus1 : a * (-1)^2 + b * (-1) + c = -1)
  (h_0 : a * 0^2 + b * 0 + c = 1)
  (h_1 : a * 1^2 + b * 1 + c = 1) : 
  ∃ x1 : ℝ, -1 < x1 ∧ x1 < 0 ∧ a * x1^2 + b * x1 + c = 0 :=
sorry

end quadratic_root_range_l395_395000


namespace model_N_time_is_12_l395_395482

-- Define the basic conditions
def model_M_time : ℕ := 24
def num_M_computers : ℕ := 8
def num_N_computers : ℕ := 8

-- Define the rates
def rate_M (n : ℕ) : ℝ := (n : ℝ) / model_M_time
def rate_N (T : ℕ) (n : ℕ) : ℝ := (n : ℝ) / (T : ℝ)

-- Define the condition for completing the task in 1 minute
def combined_rate_condition (T : ℕ) : Prop :=
  rate_M num_M_computers + rate_N T num_N_computers = 1

-- Prove that it takes model N 12 minutes to complete the task
theorem model_N_time_is_12 : ∃ T : ℕ, combined_rate_condition T ∧ T = 12 :=
by
  use 12
  have h1 : rate_M num_M_computers = 1 / 3 := by norm_num [rate_M, num_M_computers, model_M_time]
  have h2 : rate_N 12 num_N_computers = 2 / 3 := by norm_num [rate_N, num_N_computers]
  show combined_rate_condition 12
  rw [combined_rate_condition, h1, h2]
  norm_num
  sorry

end model_N_time_is_12_l395_395482


namespace greatest_n_l395_395159

/-- 
Proof of the greatest integer n such that 2007 different positive integers 
can be chosen from [2 * 10^(n-1), 10^n) where for each pair (i < j), 
there exists a number from the chosen integers with a_j >= a_i + 2.
-/
theorem greatest_n (_ : ∃ (n : ℕ), 
                         n > 0 ∧ 
                         ∃ (S : set ℕ), 
                         S.card = 2007 ∧ 
                         ∀ (i j : ℕ), 
                         1 ≤ i ∧ i < j ∧ j ≤ n → 
                         (∃ (a : ℕ), a ∈ S ∧ digit_at_pos a j ≥ digit_at_pos a i + 2)) : 
  ∃ n, 
    (∀ (k : ℕ), k > n → ¬(∃ (S : set ℕ), 
                             S.card = 2007 ∧ 
                             ∀ (i j : ℕ), 
                             1 ≤ i ∧ i < j ∧ j ≤ k → 
                             (∃ (a : ℕ), a ∈ S ∧ digit_at_pos a j ≥ digit_at_pos a i + 2))) ∧ 
    n = 63 := 
  sorry

end greatest_n_l395_395159


namespace triangle_inequality_l395_395248

theorem triangle_inequality (A B C P : Point)
  (hAB : dist A B = 2 * real.sqrt 2)
  (hAC : dist A C = real.sqrt 2)
  (hBC : dist B C = 2)
  (hP_on_BC : lies_on P B C) :
  dist P A ^ 2 > dist P B * dist P C :=
sorry

end triangle_inequality_l395_395248


namespace find_a_b_l395_395581

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395581


namespace time_2023_minutes_after_midnight_dec_31_2020_l395_395059

theorem time_2023_minutes_after_midnight_dec_31_2020 :
  let start_time := by sorry, -- Assume some definition for midnight Dec 31, 2020
      minutes_in_day := 1440,
      hours_in_day := 24,
      minutes_in_hour := 60,
      total_minutes := 2023,
      hours := total_minutes / minutes_in_hour,
      minutes := total_minutes % minutes_in_hour,
      days := hours / hours_in_day,
      remaining_hours := hours % hours_in_day,
      time_after_days := by sorry, -- Adding days to start_time
      final_time := by sorry -- Adding remaining hours and minutes to time_after_days
  in final_time = "January 1 at 9:43 AM" :=
by
  sorry -- The actual proof is omitted

end time_2023_minutes_after_midnight_dec_31_2020_l395_395059


namespace find_a_b_l395_395652

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395652


namespace find_a_b_l395_395651

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395651


namespace solve_for_x_l395_395317

noncomputable def z1 : ℂ := 2 + Complex.i
noncomputable def z2 (x : ℝ) : ℂ := x - 2 * Complex.i

theorem solve_for_x (x : ℝ) (h : ∃ r : ℝ, z1 * z2 x = r) : x = 4 :=
by
  sorry

end solve_for_x_l395_395317


namespace selection_from_sequence_l395_395344

theorem selection_from_sequence (n : ℕ) (a : ℕ → ℝ) :
  ∃ (subseq : set ℕ), 
  (∀ k, k ∈ subseq → k + 1 ∈ subseq → k + 2 ∉ subseq) ∧
  (∀ k, k ≤ n - 2 → (k ∈ subseq ∨ k + 1 ∈ subseq ∨ k + 2 ∈ subseq)) ∧
  abs (∑ k in subseq, a k) ≥ (1 / 6) * ∑ i in finset.range n, abs (a i) := by
sorry

end selection_from_sequence_l395_395344


namespace largest_three_digit_number_l395_395431

theorem largest_three_digit_number (a b c : ℕ) (h1 : a = 8) (h2 : b = 0) (h3 : c = 7) :
  ∃ (n : ℕ), ∀ (x : ℕ), (x = a * 100 + b * 10 + c) → x = 870 :=
by
  sorry

end largest_three_digit_number_l395_395431


namespace combined_weight_of_contents_l395_395927

theorem combined_weight_of_contents
    (weight_pencil : ℝ := 28.3)
    (weight_eraser : ℝ := 15.7)
    (weight_paperclip : ℝ := 3.5)
    (weight_stapler : ℝ := 42.2)
    (num_pencils : ℕ := 5)
    (num_erasers : ℕ := 3)
    (num_paperclips : ℕ := 4)
    (num_staplers : ℕ := 2) :
    num_pencils * weight_pencil +
    num_erasers * weight_eraser +
    num_paperclips * weight_paperclip +
    num_staplers * weight_stapler = 287 := 
sorry

end combined_weight_of_contents_l395_395927


namespace solve_inequality_l395_395162

noncomputable def solution_set : set ℝ :=
  {x : ℝ | (x > -9 / 2 ∧ x < -2) ∨ (x > (1 - Real.sqrt 5) / 2 ∧ x < (1 + Real.sqrt 5) / 2)}

theorem solve_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -9 / 2) :
  (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ↔ x ∈ solution_set := 
sorry

end solve_inequality_l395_395162


namespace find_a_and_b_l395_395603

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395603


namespace income_threshold_l395_395855

theorem income_threshold (x : ℝ) (N : ℝ) 
  (h1 : N = 1.6 * 10^9 * x^(-5/3))
  (h2 : N ≤ 1600) : 
  x ≥ 10^4 := 
by
  sorry

end income_threshold_l395_395855


namespace maximize_probability_when_C_second_game_l395_395951

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l395_395951


namespace fold_condition_l395_395095

open Real

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def perpendicular_bisector (a b : ℝ × ℝ) : ℝ × ℝ → Prop :=
  let mid := midpoint a b in
  let slope := -(a.2 - b.2) / (a.1 - b.1) in
  λ p, p.2 - mid.2 = slope * (p.1 - mid.1)

theorem fold_condition : ∀ p q : ℝ,
  let a := (5, -1)
  let b := (1, 3)
  let c := (8, 2)
  let d := (p, q)
  let mid_cd := midpoint c d in
  let bisector := perpendicular_bisector a b in
  bisector mid_cd →
  q + p = 4 :=
sorry

end fold_condition_l395_395095


namespace minimum_jumps_to_visit_all_points_l395_395385

-- Define the conditions
def grid_size := 10 -- Size of each cell in the grid is 10 cm.
def jump_distance := 50 -- Grasshopper jump distance is 50 cm.
def total_points := 8 -- Total points to visit.

-- Define the problem statement in Lean
theorem minimum_jumps_to_visit_all_points
  (reachable_by_jump : ∀ (p1 p2 : ℝ × ℝ), distance p1 p2 = jump_distance → true)
  (points : ℕ → ℝ × ℝ) (h_points : ∀ i, i < total_points → (points i).fst % grid_size = 0 ∧ (points i).snd % grid_size = 0) :
  ∃ (route : ℕ → ℝ × ℝ), (∀ k < total_points - 1, distance (route k) (route (k + 1)) = jump_distance) ∧
    (∀ p ∈ range total_points, ∃ k, route k = points p) ∧
    fintype.card (range route) = total_points :=
sorry

end minimum_jumps_to_visit_all_points_l395_395385


namespace num_ordered_triples_l395_395995

theorem num_ordered_triples : 
  {n : ℕ // ∃ (a b c : ℤ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = (2 * (a * b + b * c + c * a)) / 3 ∧ n = 3} :=
sorry

end num_ordered_triples_l395_395995


namespace total_glasses_of_drinks_l395_395411

theorem total_glasses_of_drinks (pitchers : ℕ) (glasses_per_pitcher : ℕ) : pitchers = 9 → glasses_per_pitcher = 6 → pitchers * glasses_per_pitcher = 54 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end total_glasses_of_drinks_l395_395411


namespace monotonic_decreasing_interval_of_f_l395_395858

noncomputable def f : ℝ → ℝ := λ x, 2^(x^2 - 2*x + 1)

theorem monotonic_decreasing_interval_of_f :
  monotone_decreasing_on f (-∞ : ℝ) 1 :=
  by
  sorry

end monotonic_decreasing_interval_of_f_l395_395858


namespace problem_sequence_square_sum_l395_395274

theorem problem_sequence_square_sum :
  ∀ (a : ℕ → ℕ) (n : ℕ),
    (∀ (n : ℕ), 
      (a 0) + (∑ k in Finset.range n, 2^k * a (k + 1)) = 2^(2 * n) - 1) →
    (∑ k in Finset.range (n + 1), (a k)^2) = 3 * (4^n - 1) :=
by sorry

end problem_sequence_square_sum_l395_395274


namespace quadratic_equation_with_root_l395_395146

theorem quadratic_equation_with_root (x : ℝ) (h1 : x = sqrt 5 - 3) :
  ∃ (a b c : ℚ), a = 1 ∧ b = 6 ∧ c = 4 ∧ (a * x^2 + b * x + c = 0) :=
begin
  sorry
end

end quadratic_equation_with_root_l395_395146


namespace coefficient_x2_y4_in_expansion_l395_395269

theorem coefficient_x2_y4_in_expansion :
  (1 + x + y^2) ^ 5.coeff (2, 4) = 30 :=
sorry

end coefficient_x2_y4_in_expansion_l395_395269


namespace max_prob_two_consecutive_wins_l395_395968

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l395_395968


namespace area_F2AB_eq_4_3_l395_395180

-- Define the ellipse and its foci
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def F1 : (ℝ × ℝ) := (-1, 0)
def F2 : (ℝ × ℝ) := (1, 0)
def chord_AB (θ : ℝ) : Prop := θ = Real.pi / 4 ∧ ∃ A B, A ≠ B ∧ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  (A = (x₁, y₁) ∧ B = (x₂, y₂)) ∧
  (y₁ = x₁ + 1 ∧ y₂ = x₂ + 1) ∧
  (ellipse x₁ y₁ ∧ ellipse x₂ y₂) ∧
  (x₁ = 0 ∧ y₁ = 1 ∧ x₂ = -4 / 3 ∧ y₂ = -1 / 3)

-- The area of triangle F2AB
def area_of_triangle (F2 A B : (ℝ × ℝ)) : ℝ :=
  let (x₂, y₂) := F2 in
  let (x₁, y₁) := A in
  let (x₃, y₃) := B in
  1 / 2 * abs (x₂ * (y₁ - y₃) + x₁ * (y₃ - y₂) + x₃ * (y₂ - y₁))

theorem area_F2AB_eq_4_3 :
  ∃ A B (θ := Real.pi / 4), chord_AB θ →
  area_of_triangle F2 (0, 1) (-4 / 3, -1 / 3) = 4 / 3 :=
by
  sorry

end area_F2AB_eq_4_3_l395_395180


namespace arithmetic_and_geometric_mean_l395_395374
-- Importing the entire required Mathlib library.

-- The statement for the problem
theorem arithmetic_and_geometric_mean :
  (arith_mean (a b : ℝ) := (a + b) / 2) →
  (geom_mean (a b : ℝ) := real.sqrt (a * b)) →
  arith_mean 5 17 = 11 ∧ geom_mean 4 9 = 6 ∨ geom_mean 4 9 = -6 := 
by
  -- Definitions for arithmetic mean and geometric mean are introduced
  -- The goal is to show that the arithmetic mean of 5 and 17 is 11, and
  -- the geometric mean of 4 and 9 is ±6.
  sorry

end arithmetic_and_geometric_mean_l395_395374


namespace percentage_of_failed_candidates_l395_395067

theorem percentage_of_failed_candidates 
  (total_candidates : ℕ) 
  (girls : ℕ) 
  (boys : ℕ)
  (boys_passed_percentage : ℝ)
  (girls_passed_percentage : ℝ)
  (total_failed_candidates : ℝ) :
  total_candidates = 2000 →
  girls = 900 →
  boys = total_candidates - girls →
  boys_passed_percentage = 0.38 →
  girls_passed_percentage = 0.32 →
  total_failed_candidates = (real.of_nat total_candidates - (boys_passed_percentage * real.of_nat boys + girls_passed_percentage * real.of_nat girls)) →
  (total_failed_candidates / total_candidates) * 100 = 64.7 :=
begin
  intros,
  /- proof steps -/
  sorry
end

end percentage_of_failed_candidates_l395_395067


namespace pie_eating_contest_l395_395876

theorem pie_eating_contest :
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  max (max first_student second_student) third_student - 
  min (min first_student second_student) third_student = 1 / 6 :=
by
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  sorry

end pie_eating_contest_l395_395876


namespace arc_length_solution_l395_395533

open Set Filter Real

noncomputable def arc_length_parabola (x_0 : ℝ) : ℝ :=
  ∫ x in (0 : ℝ)..x_0, sqrt(1 + x^2)

theorem arc_length_solution (x_0 : ℝ) : 
  arc_length_parabola x_0 = (x_0 * sqrt(1 + x_0^2) / 2) + (1 / 2) * ln (x_0 + sqrt(1 + x_0^2)) :=
by
  sorry

end arc_length_solution_l395_395533


namespace rate_per_sq_meter_l395_395015

theorem rate_per_sq_meter (length width cost : ℝ) (h_length : length = 5.5) (h_width : width = 3.75) (h_cost : cost = 24750) :
  cost / (length * width) = 1200 :=
by
  rw [h_length, h_width, h_cost]
  norm_num
  sorry

end rate_per_sq_meter_l395_395015


namespace find_a_b_l395_395628

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395628


namespace triangle_AD_lt_AE_l395_395859

theorem triangle_AD_lt_AE (A B C D E : Point)
  (h1 : is_triangle ABC)
  (h2 : is_perpendicular_bisector D BC)
  (h3 : lies_on_intersection D AB)
  (h4 : lies_on_extension E AC)
  (h5 : E ≠ A) : 
  AD < AE := 
by
  sorry

end triangle_AD_lt_AE_l395_395859


namespace non_congruent_triangles_count_l395_395695

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def count_non_congruent_triangles : ℕ :=
  let a_values := [1, 2]
  let b_values := [2, 3]
  let triangles := [(1, 2, 2), (2, 2, 2), (2, 2, 3)]
  triangles.length

theorem non_congruent_triangles_count : count_non_congruent_triangles = 3 :=
  by
    -- Proof would go here
    sorry

end non_congruent_triangles_count_l395_395695


namespace shares_distribution_l395_395552

theorem shares_distribution :
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 16 ∧ 
                   a ≤ b + c + d ∧ b ≤ a + c + d ∧ c ≤ a + b + d ∧ d ≤ a + b + c ∧ 
                   (finset.card {s : finset (ℕ × ℕ × ℕ × ℕ) | 
                      ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
                                       a + b + c + d = 16 ∧ 
                                       a ≤ b + c + d ∧ b ≤ a + c + d ∧ 
                                       c ≤ a + b + d ∧ d ≤ a + b + c}) = 321 :=
sorry

end shares_distribution_l395_395552


namespace product_of_five_consecutive_numbers_not_square_l395_395351

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l395_395351


namespace distance_traveled_by_second_hand_l395_395028

def second_hand_length : ℝ := 8
def time_period_minutes : ℝ := 45
def rotations_per_minute : ℝ := 1

theorem distance_traveled_by_second_hand :
  let circumference := 2 * Real.pi * second_hand_length
  let rotations := time_period_minutes * rotations_per_minute
  let total_distance := rotations * circumference
  total_distance = 720 * Real.pi := by
  sorry

end distance_traveled_by_second_hand_l395_395028


namespace student_average_grade_the_year_before_l395_395109

theorem student_average_grade_the_year_before 
  (x : ℝ)
  (average_last_year : ℝ)
  (courses_last_year : ℕ)
  (courses_year_before : ℕ)
  (average_two_years : ℝ) :
  courses_last_year = 6 →
  average_last_year = 100 →
  courses_year_before = 5 →
  average_two_years = 77 →
  5 * x + 6 * 100 = 11 * 77 →
  x = 49.4 := 
by {
  intros h1 h2 h3 h4 h5,
  linarith,
}

end student_average_grade_the_year_before_l395_395109


namespace max_possible_cables_l395_395118

theorem max_possible_cables (num_employees : ℕ) (num_brand_X : ℕ) (num_brand_Y : ℕ) 
  (max_connections : ℕ) (num_cables : ℕ) :
  num_employees = 40 →
  num_brand_X = 25 →
  num_brand_Y = 15 →
  max_connections = 3 →
  (∀ x : ℕ, x < max_connections → num_cables ≤ 3 * num_brand_Y) →
  num_cables = 45 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_possible_cables_l395_395118


namespace sqrt_neg9_squared_l395_395455

theorem sqrt_neg9_squared : Real.sqrt ((-9: ℝ)^2) = 9 := by
  sorry

end sqrt_neg9_squared_l395_395455


namespace simplify_expression_l395_395361

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : x⁻² - 2 * x⁻¹ + 1 = (1 - x⁻¹) ^ 2 :=
by
  sorry

end simplify_expression_l395_395361


namespace batsman_average_l395_395923

theorem batsman_average (A : ℝ) (h1 : 24 * A < 95) 
                        (h2 : 24 * A + 95 = 25 * (A + 3.5)) : A + 3.5 = 11 :=
by
  sorry

end batsman_average_l395_395923


namespace net_percentage_gain_approx_l395_395503

noncomputable def netPercentageGain : ℝ :=
  let costGlassBowls := 250 * 18
  let costCeramicPlates := 150 * 25
  let totalCostBeforeDiscount := costGlassBowls + costCeramicPlates
  let discount := 0.05 * totalCostBeforeDiscount
  let totalCostAfterDiscount := totalCostBeforeDiscount - discount
  let revenueGlassBowls := 200 * 25
  let revenueCeramicPlates := 120 * 32
  let totalRevenue := revenueGlassBowls + revenueCeramicPlates
  let costBrokenGlassBowls := 30 * 18
  let costBrokenCeramicPlates := 10 * 25
  let totalCostBrokenItems := costBrokenGlassBowls + costBrokenCeramicPlates
  let netGain := totalRevenue - (totalCostAfterDiscount + totalCostBrokenItems)
  let netPercentageGain := (netGain / totalCostAfterDiscount) * 100
  netPercentageGain

theorem net_percentage_gain_approx :
  abs (netPercentageGain - 2.71) < 0.01 := sorry

end net_percentage_gain_approx_l395_395503


namespace find_a_b_l395_395666

namespace ComplexProof

open Complex

noncomputable def condition (a b : ℝ) : Prop :=
  let z := (1 - 2 * I : ℂ)
  z + a * conj(z) + b = 0

theorem find_a_b (a b : ℝ) (h : condition a b) :
  a = 1 ∧ b = -2 :=
sorry

end ComplexProof

end find_a_b_l395_395666


namespace two_sets_of_three_or_more_consecutive_integers_sum_18_l395_395725

def consecutive_sum (a : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a + n - 1)) / 2

def valid_set (s : List ℕ) : Prop :=
  ∀ (k : ℕ), k < s.length - 1 → s[k] + 1 = s[k + 1]

def valid_sum (s : List ℕ) : Prop :=
  s.sum = 18

def three_or_more_consecutive (s : List ℕ) : Prop :=
  s.length ≥ 3

theorem two_sets_of_three_or_more_consecutive_integers_sum_18 :
  ∃ (s1 s2 : List ℕ), three_or_more_consecutive s1 ∧
                      valid_set s1 ∧
                      valid_sum s1 ∧
                      three_or_more_consecutive s2 ∧
                      valid_set s2 ∧
                      valid_sum s2 ∧
                      s1 ≠ s2 ∧
                      ∀ (s : List ℕ), (three_or_more_consecutive s ∧ valid_set s ∧ valid_sum s) →
                      (s = s1 ∨ s = s2) :=
by
  sorry

end two_sets_of_three_or_more_consecutive_integers_sum_18_l395_395725


namespace maximum_probability_second_game_C_l395_395976

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l395_395976


namespace slope_angle_tangent_line_45_degrees_l395_395863

theorem slope_angle_tangent_line_45_degrees :
  let curve := λ x : ℝ, (1 / 3) * x^3 - 2 in
  let point := (-1 : ℝ, -7 / 3 : ℝ) in
  point.2 = curve point.1 →
  let derivative := λ x : ℝ, x^2 in
  let slope := derivative point.1 in
  real.arcsin (slope / sqrt (1 + slope^2)) = real.pi / 4 :=
begin
  intros,
  sorry
end

end slope_angle_tangent_line_45_degrees_l395_395863


namespace find_a_l395_395204

-- Define the conditions and the proof goal
theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h_eq : a + a⁻¹ = 5/2) :
  a = 1/2 :=
by
  sorry

end find_a_l395_395204


namespace coefficient_x2_y4_in_expansion_l395_395270

theorem coefficient_x2_y4_in_expansion :
  (1 + x + y^2) ^ 5.coeff (2, 4) = 30 :=
sorry

end coefficient_x2_y4_in_expansion_l395_395270


namespace find_a_b_l395_395620

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395620


namespace staffing_battle_station_l395_395125

-- Define the qualifications
def num_assistant_engineer := 3
def num_maintenance_1 := 4
def num_maintenance_2 := 4
def num_field_technician := 5
def num_radio_specialist := 5

-- Prove the total number of ways to fill the positions
theorem staffing_battle_station : 
  num_assistant_engineer * num_maintenance_1 * num_maintenance_2 * num_field_technician * num_radio_specialist = 960 := by
  sorry

end staffing_battle_station_l395_395125


namespace sqrt_three_irrational_sqrt_three_not_rational_sqrt_three_decimal_expansion_non_repeating_sqrt_three_as_decimal_is_infinite_nonrepeating_l395_395392

theorem sqrt_three_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (nat.gcd p q = 1) ∧ (↑p / ↑q = Real.sqrt 3) := sorry

theorem sqrt_three_not_rational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p / q: ℝ = Real.sqrt 3) := sorry

theorem sqrt_three_decimal_expansion_non_repeating : 
  ¬ ∃ (p q : ℤ), 
  (q ≠ 0) ∧ 
  (nat.gcd p q = 1) ∧ 
  (Real.sqrt 3 = (p: ℝ) / (q: ℝ)) ∧ 
  (fraction.summable p q) := sorry

theorem sqrt_three_as_decimal_is_infinite_nonrepeating : 
  ∀ x : ℝ, 
    (x = Real.sqrt 3) → 
    (¬(∃ (f : ℕ → ℝ), 
      (∀ n, m, n ≠ m → f(n) ≠ f(m)) ∧ 
      (summable_expansion f x)) := sorry

end sqrt_three_irrational_sqrt_three_not_rational_sqrt_three_decimal_expansion_non_repeating_sqrt_three_as_decimal_is_infinite_nonrepeating_l395_395392


namespace product_of_five_consecutive_integers_not_square_l395_395359

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l395_395359


namespace internal_bisector_ratio_external_bisector_intersection_condition_external_bisector_ratio_l395_395916

-- Definitions for tetrahedron and areas of triangles
variables {A B C D M M' : Point}
variables {t_ABC t_ABD : ℝ}

-- Internal plane bisector ratio
theorem internal_bisector_ratio 
  (h1 : IsTetrahedron A B C D)
  (h2 : IsEdge A B)
  (h3 : IsEdge C D)
  (h4 : IntersectsAt S A B C D M)
  (h5 : InternalAngleBisector S A B):
  CM / MD = t_ABC / t_ABD := 
sorry

-- Condition for the external plane bisector to intersect CD
theorem external_bisector_intersection_condition 
  (h1 : IsTetrahedron A B C D)
  (h2 : IsEdge A B)
  (h3 : IsEdge C D)
  (h4 : AreasEqual t_ABC t_ABD):
  ExternalBisectorIntersectsCD S' A B C D :=
sorry

-- External plane bisector ratio
theorem external_bisector_ratio 
  (h1 : IsTetrahedron A B C D)
  (h2 : IsEdge A B)
  (h3 : IsEdge C D)
  (h4 : IntersectsAt S' A B C D M')
  (h5 : ExternalAngleBisector S' A B):
  CM' / M'D = t_ABC / t_ABD :=
sorry

end internal_bisector_ratio_external_bisector_intersection_condition_external_bisector_ratio_l395_395916


namespace decompose_two_over_eleven_decompose_two_over_n_l395_395259

-- Problem 1: Decompose 2/11
theorem decompose_two_over_eleven : (2 : ℚ) / 11 = (1 / 6) + (1 / 66) :=
  sorry

-- Problem 2: General form for 2/n for odd n >= 5
theorem decompose_two_over_n (n : ℕ) (hn : n ≥ 5) (odd_n : n % 2 = 1) :
  (2 : ℚ) / n = (1 / ((n + 1) / 2)) + (1 / (n * (n + 1) / 2)) :=
  sorry

end decompose_two_over_eleven_decompose_two_over_n_l395_395259


namespace total_vehicles_in_lanes_l395_395999

theorem total_vehicles_in_lanes :
  ∀ (lanes : ℕ) (trucks_per_lane cars_total trucks_total : ℕ),
  lanes = 4 →
  trucks_per_lane = 60 →
  trucks_total = trucks_per_lane * lanes →
  cars_total = 2 * trucks_total →
  (trucks_total + cars_total) = 2160 :=
by intros lanes trucks_per_lane cars_total trucks_total hlanes htrucks_per_lane htrucks_total hcars_total
   -- sorry added to skip the proof
   sorry

end total_vehicles_in_lanes_l395_395999


namespace triangle_maximum_area_l395_395417

theorem triangle_maximum_area
  (D E F : ℝ × ℝ)
  (hD : D = (4, 0))
  (hE : E = (0, 4))
  (hF : F.1 + F.2 = 9) :
  ∃ area : ℝ, area = 10 :=
by
  let base := dist (4:ℝ,0) (9,0)
  let height := (0:ℝ,4).snd
  use (1/2) * base * height
  sorry

end triangle_maximum_area_l395_395417


namespace Tim_pays_correct_amount_l395_395877

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end Tim_pays_correct_amount_l395_395877


namespace maximize_prob_of_consecutive_wins_l395_395949

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l395_395949


namespace find_a_b_l395_395625

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395625


namespace range_of_a_l395_395185

-- Definitions of the conditions
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0
def q (x : ℝ) (a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0 ∧ a > 0

-- Statement of the theorem that proves the range of a
theorem range_of_a (x : ℝ) (a : ℝ) :
  (¬ (p x) → ¬ (q x a)) ∧ (¬ (q x a) → ¬ (p x)) → (a ≥ 9) :=
by
  sorry

end range_of_a_l395_395185


namespace find_m_and_union_A_B_l395_395717

variable (m : ℝ)
noncomputable def A := ({3, 4, m^2 - 3 * m - 1} : Set ℝ)
noncomputable def B := ({2 * m, -3} : Set ℝ)

theorem find_m_and_union_A_B (h : A m ∩ B m = ({-3} : Set ℝ)) :
  m = 1 ∧ A m ∪ B m = ({-3, 2, 3, 4} : Set ℝ) :=
sorry

end find_m_and_union_A_B_l395_395717


namespace num_even_integers_l395_395160

theorem num_even_integers (n : ℕ) : 
  (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n) ∧ n ≤ 1000 ∧ n % 2 = 0 →
  (∃ m : ℕ, n = 7 * m ∨ n = 7 * m + 2) →
  ∑ k in finset.range 143, (2 * k + 2) ≤ 1000 :=
by
  sorry

end num_even_integers_l395_395160


namespace round_1723_50000005_nearest_whole_l395_395346

-- Define the number in question
def num : ℝ := 1723.50000005

-- Define the condition for rounding up
def decimal_part (x : ℝ) : ℝ := x - x.floor
def should_round_up (x : ℝ) : Prop := decimal_part x ≥ 0.5

-- State the theorem
theorem round_1723_50000005_nearest_whole : should_round_up num → ⌈num⌉ = 1724 :=
by
  intros h
  sorry

end round_1723_50000005_nearest_whole_l395_395346


namespace cost_price_l395_395117

theorem cost_price (MP SP C : ℝ) (h1 : MP = 74.21875)
  (h2 : SP = MP - 0.20 * MP)
  (h3 : SP = 1.25 * C) : C = 47.5 :=
by
  sorry

end cost_price_l395_395117


namespace find_a_b_l395_395609

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395609


namespace count_a_k_divisible_by_9_l395_395802

def concat_ints (a b : ℕ) : ℕ := a * 10^(Nat.log10 b + 1) + b

def a_n_seq : ℕ → ℕ
| 1 => 1
| n+1 => concat_ints (a_n_seq n + 1) (n + 1)

def f (n : ℕ) : ℕ := n.digits 10 |>.sum

def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem count_a_k_divisible_by_9 :
  (Finset.range 100).filter (λ k => divisible_by_9 (a_n_seq (k + 1))).card = 22 := by
  sorry

end count_a_k_divisible_by_9_l395_395802


namespace geometric_sum_f_value_l395_395221

variables {f : ℝ → ℝ} (a : ℕ → ℝ)

-- Given function
def f (x : ℝ) : ℝ := 3^x / (3^x + 1)

-- Additional conditions
-- 1. f(x) + f(-x) = 1
axiom f_symm : ∀ x : ℝ, f x + f (-x) = 1

-- 2. a is a positive geometric sequence such that a_{k-1} * a_{k+1} = 1
axiom geo_seq : ∀ k : ℕ, (0 < k) ∧ (k < 100) → a (k-1) * a (k+1) = 1

-- 3. a_{50} = 1
axiom a_50 : a 50 = 1

-- Prove that: f(ln(a_1)) + f(ln(a_2)) + ... + f(ln(a_99)) = 99/2
theorem geometric_sum_f_value : (list.sum (list.map (λ n, f (real.log (a n))) (list.range 99).succ)) = 99 / 2 :=
sorry -- Proof is omitted

end geometric_sum_f_value_l395_395221


namespace systematic_sampling_interval_l395_395921

def population_size : ℕ := 2000
def sample_size : ℕ := 50
def interval (N n : ℕ) : ℕ := N / n

theorem systematic_sampling_interval :
  interval population_size sample_size = 40 := by
  sorry

end systematic_sampling_interval_l395_395921


namespace probability_interval_3_6_l395_395212

open Classical
open Probability

noncomputable def normalDistribution := Normal 0 3

theorem probability_interval_3_6 :
  let ξ := normalDistribution.random_variable in
  P (3 < ξ ∧ ξ < 6) = 0.1359 := by
sorry

end probability_interval_3_6_l395_395212


namespace total_amount_paid_l395_395287

theorem total_amount_paid : 
  (∀ (p cost_per_pizza : ℕ), cost_per_pizza = 12 → p = 3 → cost_per_pizza * p = 36) →
  (∀ (d cost_per_pizza : ℕ), cost_per_pizza = 12 → d = 2 → cost_per_pizza * d + 2 = 26) →
  36 + 26 = 62 :=
by {
  intro h1 h2,
  sorry
}

end total_amount_paid_l395_395287


namespace cut_ribbon_l395_395917

theorem cut_ribbon
    (length_ribbon : ℝ)
    (points : ℝ × ℝ × ℝ × ℝ × ℝ)
    (h_length : length_ribbon = 5)
    (h_points : points = (1, 2, 3, 4, 5)) :
    points.2.1 = (11 / 15) * length_ribbon :=
by
    sorry

end cut_ribbon_l395_395917


namespace total_insects_eaten_l395_395467

theorem total_insects_eaten
  (geckos : ℕ)
  (insects_per_gecko : ℕ)
  (lizards : ℕ)
  (multiplier : ℕ)
  (h_geckos : geckos = 5)
  (h_insects_per_gecko : insects_per_gecko = 6)
  (h_lizards : lizards = 3)
  (h_multiplier : multiplier = 2) :
  geckos * insects_per_gecko + lizards * (insects_per_gecko * multiplier) = 66 :=
by
  rw [h_geckos, h_insects_per_gecko, h_lizards, h_multiplier]
  norm_num
  sorry

end total_insects_eaten_l395_395467


namespace problem_solution_l395_395123

noncomputable def problem_statement : Prop :=
  (√3 + 2)^2023 * (√3 - 2)^2024 = -√3 + 2

theorem problem_solution : problem_statement := 
begin
  sorry
end

end problem_solution_l395_395123


namespace total_amount_paid_l395_395285

-- Definitions based on the conditions.
def cost_per_pizza : ℝ := 12
def delivery_charge : ℝ := 2
def distance_threshold : ℝ := 1000 -- distance in meters
def park_distance : ℝ := 100
def building_distance : ℝ := 2000

def pizzas_at_park : ℕ := 3
def pizzas_at_building : ℕ := 2

-- The proof problem stating the total amount paid to Jimmy.
theorem total_amount_paid :
  let total_pizzas := pizzas_at_park + pizzas_at_building
  let cost_without_delivery := total_pizzas * cost_per_pizza
  let park_charge := if park_distance > distance_threshold then pizzas_at_park * delivery_charge else 0
  let building_charge := if building_distance > distance_threshold then pizzas_at_building * delivery_charge else 0
  let total_cost := cost_without_delivery + park_charge + building_charge
  total_cost = 64 :=
by
  sorry

end total_amount_paid_l395_395285


namespace problem_statement_l395_395133

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_statement : 
  (∀ x : ℝ, f(x) + (x + 1/2) * f(1 - x) = 1) →
  (2016 / (f(0)^2 * f(1)^3) = -63) := 
by 
  intros h 
  sorry

end problem_statement_l395_395133


namespace second_chapter_length_l395_395925

variable (number_of_pages_ch2 : Nat)

-- Given condition: The second chapter is 68 pages long.
axiom page_length_cond : number_of_pages_ch2 = 68

-- The proof statement we need to show.
theorem second_chapter_length : number_of_pages_ch2 = 68 :=
by
  exact page_length_cond

end second_chapter_length_l395_395925


namespace has_root_in_interval_l395_395386

def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem has_root_in_interval (h_cont : Continuous f) (h_ivt : ∃ ξ ∈ Set.Ioo 0 1, f ξ = 0) : ∃ ξ ∈ Set.Ioo 0 1, f ξ = 0 :=
  h_ivt

end has_root_in_interval_l395_395386


namespace lottery_ticket_probability_l395_395024

theorem lottery_ticket_probability (p : ℝ) (n : ℕ) (h_prob : p = 0.002) (h_n : n = 1000) :
  "Each ticket has an equal chance of winning." :=
begin
  sorry
end

end lottery_ticket_probability_l395_395024


namespace general_term_formula_l395_395400

-- Given the sequence sum formula S_n = n^2 - n for n in ℕ*
def Sn (n : ℕ) : ℕ := n^2 - n

-- We need to prove that a_n = 2n - 2 given the conditions
theorem general_term_formula (n : ℕ) (hn : n > 0) : 
  let a (n : ℕ) := if n = 1 then Sn 1 else Sn n - Sn (n - 1)
  in a n = 2 * n - 2 :=
by
  sorry

end general_term_formula_l395_395400


namespace find_a_and_b_l395_395597

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395597


namespace sum_first_n_terms_l395_395275

def sequence_a_n (n : ℕ) : ℕ :=
  n^3 * ∏ i in finset.range 100, (n^2 - (i+1)^2)^2

def sum_S_n (n : ℕ) : ℕ :=
  finset.sum (finset.range n) (λ k, sequence_a_n (k + 1))

theorem sum_first_n_terms (n : ℕ) : 
  sum_S_n n = (n^2 / 400) * (n + 100)^2 * ∏ i in finset.range 100, (n^2 - (i + 1)^2)^2 := 
sorry

end sum_first_n_terms_l395_395275


namespace necessary_and_sufficient_l395_395700

variables (a b : ℝ^3)

-- Define unit vectors
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1

-- Define orthogonality
def orthogonal (v w : ℝ^3) : Prop := dot_product v w = 0

-- Given: a and b are unit vectors.
axiom a_is_unit_vector : is_unit_vector a
axiom b_is_unit_vector : is_unit_vector b

-- Condition: |a - 3b| = |3a + b|
axiom condition : ‖a - 3 • b‖ = ‖3 • a + b‖

-- Proof statement
theorem necessary_and_sufficient : ‖a - 3 • b‖ = ‖3 • a + b‖ ↔ orthogonal a b :=
sorry

end necessary_and_sufficient_l395_395700


namespace rearrangements_abcde_l395_395231

theorem rearrangements_abcde (S : List Char)
    (h1 : List.isPermutation S ['a', 'b', 'c', 'd', 'e'])
    (h2 : ∀ i, i < S.length - 1 → ¬((S.nth i, S.nth (i + 1)) ∈ [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')]))
    (h3 : S.head ≠ 'e' ∧ S.reverse.head ≠ 'e') : 
    ∃ (L : List (List Char)), L.length = 4 ∧ ∀ T ∈ L, 
        List.isPermutation T S ∧ 
        (∀ j, j < T.length - 1 → ¬((T.nth j, T.nth (j + 1)) ∈ [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])) ∧ 
       (T.head ≠ 'e' ∧ T.reverse.head ≠ 'e') :=
sorry

end rearrangements_abcde_l395_395231


namespace bridge_length_l395_395914

theorem bridge_length (length_of_train : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : 
  length_of_train = 110 → train_speed_kmph = 45 → time_seconds = 30 → 
  ∃ length_of_bridge : ℕ, length_of_bridge = 265 := by
  intros h1 h2 h3
  sorry

end bridge_length_l395_395914


namespace stripe_length_l395_395080

theorem stripe_length (circumference height : ℝ) (wraps : ℕ) 
  (h1 : circumference = 16) (h2 : height = 6) (h3 : wraps = 2) : 
  let rectangle_height := wraps * height
  in (rectangle_height = 12 ∧ (∃ length, length = real.sqrt (circumference^2 + rectangle_height^2) ∧ length = 20)) :=
by
  sorry

end stripe_length_l395_395080


namespace speed_of_second_car_l395_395043

theorem speed_of_second_car
  (t : ℝ) (d : ℝ) (d1 : ℝ) (d2 : ℝ) (v : ℝ)
  (h1 : t = 2.5)
  (h2 : d = 175)
  (h3 : d1 = 25 * t)
  (h4 : d2 = v * t)
  (h5 : d1 + d2 = d) :
  v = 45 := by sorry

end speed_of_second_car_l395_395043


namespace max_value_when_a_minus_1_find_a_given_max_value_l395_395680

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x + Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := a + 1 / x

theorem max_value_when_a_minus_1 (x : ℝ) (h : f' x (-1) = 0) : f x (-1) = -1 :=
by
  sorry

theorem find_a_given_max_value (x : ℝ) (h : f' x a = 0) (h_max : ∀ x ∈ set.Ioc (0:ℝ) Real.exp 1, f x a = -3) : a = - Real.exp 2 :=
by
  sorry

end max_value_when_a_minus_1_find_a_given_max_value_l395_395680


namespace eval_f_sum_l395_395546

def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log2 (2 - x) else Real.pow 2 (x - 1)

theorem eval_f_sum : f (-2) + f (Real.log2 12) = 9 :=
by
  sorry

end eval_f_sum_l395_395546


namespace find_a_b_l395_395645

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395645


namespace eccentricity_of_hyperbola_l395_395020

noncomputable def parabola := {p : ℝ} -> {F : ℝ} -> {C1 : ℝ -> ℝ -> Prop}
  := λ (p F : ℝ), λ (x y : ℝ), x^2 = 2 * p * y

noncomputable def hyperbola := {a b F1 F2 : ℝ} -> {C2 : ℝ -> ℝ -> Prop}
  := λ (a b F1 F2 : ℝ), λ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

noncomputable def point_on_both_curves := {p a b F1 F : ℝ} -> {x y : ℝ} ->
  parabola p F x y ∧ hyperbola a b F1 F2 x y

noncomputable def collinear_pff1 := {p F F1 : ℝ} -> {x y : ℝ} -> 
  point_on_both_curves p _ _ F1 _ x y -> x = y * (F1 - F) 

noncomputable def common_tangent := {p a : ℝ} -> {b F1 F2 : ℝ} -> {x y : ℝ} -> 
  point_on_both_curves p a b _ _ x y -> (x, y) on tangent {p}

theorem eccentricity_of_hyperbola (p a b F F1 F2 x y : ℝ)
  (H1 : parabola p F x y)
  (H2 : hyperbola a b F1 F2 x y)
  (H3 : point_on_both_curves p a b F1 F x y)
  (H4 : collinear_pff1 p F F1 x y)
  (H5 : common_tangent p a b F1 F2 x y) :
  sqrt (1 + (b^2 / a^2)) = sqrt 2 := sorry

end eccentricity_of_hyperbola_l395_395020


namespace chemical_solution_replacement_l395_395993

theorem chemical_solution_replacement (P : ℝ) :
  (0.5 * 0.50 + 0.5 * P = 0.55) → (P = 0.60) :=
by
  assume h : 0.5 * 0.50 + 0.5 * P = 0.55
  sorry

end chemical_solution_replacement_l395_395993


namespace time_to_cross_l395_395046

theorem time_to_cross
  (speed_faster_train_kmph : ℕ)
  (speed_slower_train_kmph : ℕ)
  (length_faster_train_m : ℕ)
  (h1 : speed_faster_train_kmph = 144)
  (h2 : speed_slower_train_kmph = 72)
  (h3 : length_faster_train_m = 380) :
  let relative_speed_mps := (speed_faster_train_kmph - speed_slower_train_kmph) * 1000 / 3600
  in (length_faster_train_m / relative_speed_mps) = 19 := by
  sorry

end time_to_cross_l395_395046


namespace find_smallest_even_number_l395_395864

theorem find_smallest_even_number (n : ℕ) (h : n + (n + 2) + (n + 4) = 162) : n = 52 :=
by
  sorry

end find_smallest_even_number_l395_395864


namespace slope_of_y_eq_x_minus_1_l395_395401

/-- Given the line equation y = x - 1, the slope of the line is 1. -/
theorem slope_of_y_eq_x_minus_1 : 
  ∀ (x y : ℝ), (y = x - 1) → (1) :=
by 
  sorry

end slope_of_y_eq_x_minus_1_l395_395401


namespace probability_X_leq_1_l395_395704

open MeasureTheory Probability

noncomputable def X : ℝ → ℝ := sorry

axiom normal_distribution (μ σ : ℝ) : probability_distribution ℝ := sorry

theorem probability_X_leq_1 (μ σ : ℝ) (hμ : μ = 2) (hσ : ∃ δ, σ^2 = δ^2) (hP : P (event (λ x, 1 < x ∧ x < 3)) = 0.4) :
  P (event (λ x, x ≤ 1)) = 0.3 :=
sorry

end probability_X_leq_1_l395_395704


namespace ratio_of_divisors_l395_395297

-- Definition of M
def M := 25 * 48 * 49 * 81

-- Function to calculate sum of odd divisors
def sum_odd_divisors (n : ℕ) : ℕ :=
  nat.divisors n |>.filter (λ x, ¬2 ∣ x) |>.sum

-- Function to calculate sum of even divisors
def sum_even_divisors (n : ℕ) : ℕ :=
  nat.divisors n |>.filter (λ x, 2 ∣ x) |>.sum

-- Theorem stating the ratio
theorem ratio_of_divisors : sum_odd_divisors M = sum_even_divisors M / 30 :=
by sorry

end ratio_of_divisors_l395_395297


namespace sum_of_values_l395_395485

noncomputable def g : ℝ → ℝ := sorry

theorem sum_of_values :
  (∀ x : ℝ, x ≠ 0 → 3 * g(x) + g(1 / x) = 7 * x + 5) →
  (∀ x : ℝ, g(x) = 3005 → True) →
  let S := {x : ℝ | g(x) = 3005} in
  let T := ∑ x in S, x in
  | T - 1144 | < 1 :=
by
  intro h1 h2
  let S := {x : ℝ | g(x) = 3005}
  let T := ∑ x in S, x
  have : T = 24025 / 21 := sorry
  sorry

end sum_of_values_l395_395485


namespace locus_of_points_P_is_ellipse_l395_395136

-- Definitions for points A and B and their distance.
noncomputable def point := ℝ × ℝ

-- Given conditions
variable (A B P: point)
variable (d : ℝ)

def distance (x y : point) : ℝ :=
  real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

-- Conditions given in the problem
axiom dist_AB : distance A B = 6
axiom sum_dist_PA_PB : distance P A + distance P B = 12

-- The proof problem translated into a formal proposition
theorem locus_of_points_P_is_ellipse (A B P : point) (d : ℝ) 
  (h1 : distance A B = 6) 
  (h2 : distance P A + distance P B = 12) : 
  ∃ c : ℝ, c > 0 ∧ (∀P, distance P A + distance P B = 12) :=
sorry

end locus_of_points_P_is_ellipse_l395_395136


namespace number_of_valid_sets_l395_395223

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8,9,10}
def valid_set (A : Set ℕ) : Prop :=
  ∃ a1 a2 a3, A = {a1, a2, a3} ∧ a3 ∈ U ∧ a2 ∈ U ∧ a1 ∈ U ∧ a3 ≥ a2 + 1 ∧ a2 ≥ a1 + 4

theorem number_of_valid_sets : ∃ (n : ℕ), n = 56 ∧ ∃ S : Finset (Set ℕ), (∀ A ∈ S, valid_set A) ∧ S.card = n := by
  sorry

end number_of_valid_sets_l395_395223


namespace student_listens_for_20_minutes_l395_395512

def listening_probability : ℝ :=
  let total_interval : ℝ := 40
  let listening_interval : ℝ := 10
  listening_interval / total_interval

theorem student_listens_for_20_minutes :
  listening_probability = 1 / 4 :=
by
  sorry

end student_listens_for_20_minutes_l395_395512


namespace range_of_fraction_l395_395026

theorem range_of_fraction : 
  ∀ y ∈ set.range (λ x : ℝ, (x + 3) / (x + 1)), 
  5 / 3 ≤ y ∧ y ≤ 3 := by
    sorry

end range_of_fraction_l395_395026


namespace find_a_b_l395_395627

theorem find_a_b (a b: ℝ) (z : ℂ) (h1: z = 1 - 2 * complex.I) 
  (h2: z + a * complex.conj(z) + b = 0) : 
  a = 1 ∧ b = -2 :=
by 
  sorry

end find_a_b_l395_395627


namespace min_value_of_m_l395_395682

def seq_satisfies (a : ℕ → ℤ) (n m : ℕ) : Prop :=
(∀ i, a i ^ 2 ≤ 1) ∧ 1 ≤ (Finset.range n).sum (λ i, a i ^ 2) ∧ (Finset.range n).sum (λ i, a i ^ 2) ≤ m

def f (n m : ℕ) : ℕ :=
Finset.card {a : ℕ → ℤ // seq_satisfies a n m}

noncomputable def g (m : ℕ) : ℕ := 3 ^ (m + 1) - 2 ^ (m + 1) - 1

theorem min_value_of_m : ∃ m : ℕ, g m > 2016 ∧ ∀ k < m, g k ≤ 2016 :=
sorry

end min_value_of_m_l395_395682


namespace ants_meet_probability_l395_395034

-- Definitions based on the conditions
variable (V : Type)
variable [Fintype V]
variable [DecidableEq V]

constant tetrahedron : SimpleGraph V
constant ants_initial_vertices : Fin 3 → V
constant move_probability : (v₁ v₂ : V) → ℝ

-- Conditions for the problem
axiom ants_start_on_different_vertices (x y : Fin 3) : x ≠ y → ants_initial_vertices x ≠ ants_initial_vertices y
axiom movement_probability (v : V) (u ∈ tetrahedron.adj v) : move_probability v u = 1 / 3

-- Main theorem statement
theorem ants_meet_probability :
  (Probability (at_stop_same_vertex ants_initial_vertices move_probability) = 1 / 16) :=
sorry

end ants_meet_probability_l395_395034


namespace largest_x_eq_neg5_l395_395545

theorem largest_x_eq_neg5 (x : ℝ) (h : x ≠ 7) : (x^2 - 5*x - 84)/(x - 7) = 2/(x + 6) → x ≤ -5 := 
sorry

end largest_x_eq_neg5_l395_395545


namespace jimmy_father_subscription_day_l395_395288

theorem jimmy_father_subscription_day : 
  ∃ x : ℕ, 
  (∀ d : ℕ, d < x → emails_per_day d = 20) ∧ 
  (∀ d : ℕ, d ≥ x → emails_per_day d = 25) ∧ 
  (∑ i in finset.range 30, emails_per_day i) = 675 ∧ 
  x = 15 :=
by
  let emails_per_day : ℕ → ℕ
  | d => if d < 15 then 20 else 25
  use 15
  split
  · intro d hd
    simp [emails_per_day, hd]
  split
  · intro d hd
    simp [emails_per_day, hd.not_lt]
  split
  · rw finset.sum_range_succ'
    sorry
  · refl

end jimmy_father_subscription_day_l395_395288


namespace perfume_price_reduction_l395_395496

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end perfume_price_reduction_l395_395496


namespace ellipse_equation_l395_395360

theorem ellipse_equation (x y : ℝ) :
  (sqrt ((x - 2) ^ 2 + y ^ 2) + sqrt ((x + 2) ^ 2 + y ^ 2) = 10) →
  (x ^ 2 / 25 + y ^ 2 / 21 = 1) :=
by
  sorry

end ellipse_equation_l395_395360


namespace identify_exponential_function_l395_395439

noncomputable def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ (∀ x : ℝ, f x = a^x)

theorem identify_exponential_function :
  let f_A := λ x : ℝ, Real.sin x;
  let f_B := λ x : ℝ, Real.log x / Real.log 2;
  let f_C := λ x : ℝ, 2^x;
  let f_D := λ x : ℝ, x
  in is_exponential f_C ∧
     ¬ is_exponential f_A ∧
     ¬ is_exponential f_B ∧
     ¬ is_exponential f_D :=
by {
  -- Proof goes here, but for now we leave it as sorry.
  sorry
}

end identify_exponential_function_l395_395439


namespace max_value_condition_l395_395709

noncomputable def f (a x : ℝ) : ℝ := -4 * x ^ 3 + a * x

theorem max_value_condition (a : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), f a x ≤ 1) ↔ a = 3 :=
sorry

end max_value_condition_l395_395709


namespace find_amount_lent_by_A_to_B_l395_395987

theorem find_amount_lent_by_A_to_B (P : ℝ) (h1 : B ↦ C, lent the sum at 11.5% p.a) (h2 : A ↦ B, lent the sum at 10% p.a) (h3 : gain_of_B (3 years) = 157.5) : P = 3500 :=
by
  have interest_from_C : ℝ := P * 0.115 * 3
  have interest_to_A : ℝ := P * 0.10 * 3
  have gain_of_B : ℝ := interest_from_C - interest_to_A
  have h_gain : gain_of_B = 157.5 := sorry
  have eq : 157.5 = P * 0.045 := sorry
  let P := 157.5 / 0.045
  exact 3500

end find_amount_lent_by_A_to_B_l395_395987


namespace Tim_bodyguards_weekly_pay_l395_395885

theorem Tim_bodyguards_weekly_pay :
  let hourly_rate := 20
  let num_bodyguards := 2
  let daily_hours := 8
  let weekly_days := 7
  Tim pays $2240 in a week := (hourly_rate * num_bodyguards * daily_hours * weekly_days = 2240) :=
begin
  sorry
end

end Tim_bodyguards_weekly_pay_l395_395885


namespace planting_methods_count_l395_395996

theorem planting_methods_count :
  (∃ (v : Fin 5 → Type) (l : Fin 4 → Type), 
    (∃ s : Finset (Fin 5), s.card = 4) ∧ 
    (∃ p : s → Finset (Fin 4), ∀ f : s, p f.card = 1)) →
    (finset.card (set.product {s // s.card = 4} {p // ∀ f, (p f).card = 1})) = 120 :=
by sorry

end planting_methods_count_l395_395996


namespace xy_product_l395_395345

theorem xy_product (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x * y = -24 := 
by {
  sorry
}

end xy_product_l395_395345


namespace total_vehicles_l395_395997

-- Define the conditions
def num_trucks_per_lane := 60
def num_lanes := 4
def total_trucks := num_trucks_per_lane * num_lanes
def num_cars_per_lane := 2 * total_trucks
def total_cars := num_cars_per_lane * num_lanes

-- Prove the total number of vehicles in all lanes
theorem total_vehicles : total_trucks + total_cars = 2160 := by
  sorry

end total_vehicles_l395_395997


namespace find_a_l395_395396

variable (θ a : ℝ) (h_a_pos : 0 < a)

def parametric_curve_x (θ a : ℝ) := a + 4 * Real.cos θ
def parametric_curve_y (θ : ℝ) := 1 + 4 * Real.sin θ

def polar_line (ρ θ : ℝ) := 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ = 5

theorem find_a
  (h_intersection : ∃ θ ρ, parametric_curve_x θ a = 3 * ρ * Real.cos θ ∧ parametric_curve_y θ = 4 * ρ * Real.sin θ ∧ polar_line ρ θ)
  (h_tangent : ∀ θ1 θ2, parametric_curve_x θ1 a + parametric_curve_y θ1 = parametric_curve_x θ2 a + parametric_curve_y θ2 → θ1 = θ2) :
  a = 7 :=
  sorry

end find_a_l395_395396


namespace f_not_necessarily_recursive_l395_395527

variable (N : Type) [Encodable N]
variable (R : N → N → Prop)
variable (ω : Ordinal)

-- Condition that R is a recursive binary relation
noncomputable def recursive_relation (R : N → N → Prop) : Prop := ∃ T, (Turing_compute T R)

-- Condition that R orders N into type ω
def orders_into_omega (R : N → N → Prop) (ω : Ordinal) : Prop := is_well_ordering R ω

-- Define f(n) as the n-th element in the order given by R
noncomputable def f (n : ℕ) : N := sorry -- Placeholder for the actual function definition

-- Statement to prove
theorem f_not_necessarily_recursive
  (R : N → N → Prop)
  (h1: recursive_relation R)
  (h2: orders_into_omega R ω) : 
  ¬ (∃ T, Turing_compute T f) :=
sorry

end f_not_necessarily_recursive_l395_395527


namespace graphs_symmetric_y_axis_l395_395234

variables {a b : ℝ}

theorem graphs_symmetric_y_axis (h₁ : log 10 a + log 10 b = 0) (h₂ : a ≠ 1) (h₃ : b ≠ 1) :
  ∀ x : ℝ, (a^(-x) = b^x) :=
by
  -- This is where the proof would be provided
  sorry

end graphs_symmetric_y_axis_l395_395234


namespace value_of_y_at_48_l395_395187

open Real

noncomputable def collinear_points (x : ℝ) : ℝ :=
  if x = 2 then 5
  else if x = 6 then 17
  else if x = 10 then 29
  else if x = 48 then 143
  else 0 -- placeholder value for other x (not used in proof)

theorem value_of_y_at_48 :
  (∀ (x1 x2 x3 : ℝ), x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → 
    ∃ (m : ℝ), m = (collinear_points x2 - collinear_points x1) / (x2 - x1) ∧ 
               m = (collinear_points x3 - collinear_points x2) / (x3 - x2)) →
  collinear_points 48 = 143 :=
by
  sorry

end value_of_y_at_48_l395_395187


namespace solve_for_a_b_l395_395640

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395640


namespace solution_set_l395_395402

-- Definitions representing the given conditions
def cond1 (x : ℝ) := x - 3 < 0
def cond2 (x : ℝ) := x + 1 ≥ 0

-- The problem: Prove the solution set is as given
theorem solution_set (x : ℝ) :
  (cond1 x) ∧ (cond2 x) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_set_l395_395402


namespace units_digit_3_pow_2005_l395_395557

theorem units_digit_3_pow_2005 : 
  let units_digit (n : ℕ) : ℕ := n % 10
  units_digit (3^2005) = 3 :=
by
  sorry

end units_digit_3_pow_2005_l395_395557


namespace jar_weight_percentage_l395_395404

theorem jar_weight_percentage (J B : ℝ) (h : 0.60 * (J + B) = J + 1 / 3 * B) :
  (J / (J + B)) = 0.403 :=
by
  sorry

end jar_weight_percentage_l395_395404


namespace algebraic_expression_is_product_l395_395388

def algebraicExpressionMeaning (x : ℝ) : Prop :=
  -7 * x = -7 * x

theorem algebraic_expression_is_product (x : ℝ) :
  algebraicExpressionMeaning x :=
by
  sorry

end algebraic_expression_is_product_l395_395388


namespace max_value_of_2sinx_l395_395432

theorem max_value_of_2sinx (x : ℝ) : 
  ∃ y, y = 2 * sin x ∧ y ≤ 2 := 
begin
  use (2 * sin x),
  split,
  { refl },
  { 
    have h_sin : -1 ≤ sin x ∧ sin x ≤ 1 := ⟨neg_one_le_sin, sin_le_one⟩,
    cases h_sin with hl hr,
    split,
    { linarith },
    { linarith }
  }
end

end max_value_of_2sinx_l395_395432


namespace seq_periodic_l395_395191

noncomputable def seq (a b : ℤ) : ℕ → ℤ
| 0     => 0
| 1     => a
| 2     => b
| (n+3) => seq a b (n+2) - seq a b (n+1)

def s (a b : ℤ) (n : ℕ) : ℤ :=
  (List.range n).sum (λ i => seq a b (i+1))

theorem seq_periodic (a b : ℤ) : seq a b 100 = -a ∧ s a b 100 = 2 * b - a :=
by sorry

end seq_periodic_l395_395191


namespace problem_proof_l395_395660

noncomputable def a := 1
noncomputable def b := -2
def z : ℂ := 1 - 2 * Complex.i

theorem problem_proof (a b : ℝ) (z : ℂ) (h₁ : z = 1 - 2 * Complex.i) (h₂ : z + a * Complex.conj z + b = 0) : a = 1 ∧ b = -2 :=
by {
  have h₃ : Complex.conj z = 1 + 2 * Complex.i,
  { rw [h₁, Complex.conj, Complex.re, Complex.im],
    dsimp,
    simp only [add_zero, zero_sub, neg_neg] },
  rw [h₁, h₃] at h₂,
  simp only [Complex.add_im, Complex.add_re, Complex.of_real_im, Complex.of_real_re] at h₂,
  split,
  { linarith },
  { linarith }
}

end problem_proof_l395_395660


namespace count_divisors_of_128_are_perfect_squares_larger_than_1_l395_395724

theorem count_divisors_of_128_are_perfect_squares_larger_than_1 :
  {d : ℕ | d ∣ 128 ∧ ∃ k : ℕ, k^2 = d ∧ d > 1}.to_finset.card = 3 :=
by sorry

end count_divisors_of_128_are_perfect_squares_larger_than_1_l395_395724


namespace find_a_b_l395_395642

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395642


namespace total_insects_eaten_l395_395465

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end total_insects_eaten_l395_395465


namespace find_a_b_l395_395592

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395592


namespace eval_expr_proof_l395_395869

noncomputable def eval_expression : ℝ :=
  6 * 36 * real.log10 (real.sqrt (3 + real.sqrt 5) + real.sqrt (3 - real.sqrt 5))

theorem eval_expr_proof : eval_expression = 1 / 2 :=
sorry

end eval_expr_proof_l395_395869


namespace min_omega_for_sine_overlap_l395_395701

theorem min_omega_for_sine_overlap (ω : ℝ) (hω : ω > 0) :
  (∃ n : ℤ, ∃ x : ℝ, y = sin (ω * x + (π / 3)) ∧ y = sin (ω * (x + 4*π / 3) + (π / 3))) →
  ω = 3 / 2 :=
by {
  sorry -- Proof to be provided
}

end min_omega_for_sine_overlap_l395_395701


namespace arc_division_inequality_l395_395911

theorem arc_division_inequality 
  (B C D A E : Point) 
  (h1 : arc_divide_four_equal_parts A E B C D) :
  area_triangle A C E < 8 * area_triangle B C D := sorry

end arc_division_inequality_l395_395911


namespace find_a_b_l395_395646

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 - 2 * complex.I) (h : z + a * complex.conj z + b = 0) : 
  a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395646


namespace matrix_transformation_exists_l395_395564

theorem matrix_transformation_exists :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), ∀ x y z w : ℝ,
    (N ⬝ (Matrix.vecCons (Matrix.vecCons x y) (Matrix.vecCons z w))) = 
      (Matrix.vecCons (Matrix.vecCons (2 * x) (2 * y)) (Matrix.vecCons (4 * z) (4 * w))) := 
by
  use Matrix.vecCons (Matrix.vecCons 2 0) (Matrix.vecCons 0 4)
  sorry

end matrix_transformation_exists_l395_395564


namespace trigonometric_expression_l395_395182

-- Definitions of the conditions
variables {θ : ℝ}

-- Given condition
def tan_theta := -3

-- Mathematical statement to prove
theorem trigonometric_expression (h : tan θ = tan_theta) :
  (sin θ - 2 * cos θ) / (cos θ + sin θ) = 5 / 2 :=
sorry

end trigonometric_expression_l395_395182


namespace cookies_left_after_ted_leaves_l395_395172

theorem cookies_left_after_ted_leaves :
  let f : Nat := 2 -- trays per day
  let d : Nat := 6 -- days
  let e_f : Nat := 1 -- cookies eaten per day by Frank
  let t : Nat := 4 -- cookies eaten by Ted
  let c : Nat := 12 -- cookies per tray
  let total_cookies := f * c * d -- total cookies baked
  let cookies_eaten_by_frank := e_f * d -- total cookies eaten by Frank
  let cookies_before_ted := total_cookies - cookies_eaten_by_frank -- cookies before Ted
  let total_cookies_left := cookies_before_ted - t -- cookies left after Ted
  total_cookies_left = 134
:= by
  sorry

end cookies_left_after_ted_leaves_l395_395172


namespace m_coins_can_collect_k_rubles_l395_395912

theorem m_coins_can_collect_k_rubles
  (a1 a2 a3 a4 a5 a6 a7 m k : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  ∃ (b1 b2 b3 b4 b5 b6 b7 : ℕ), 
    100 * (b1 + 2 * b2 + 5 * b3 + 10 * b4 + 20 * b5 + 50 * b6 + 100 * b7) = 100 * k ∧ 
    b1 + b2 + b3 + b4 + b5 + b6 + b7 = m := 
sorry

end m_coins_can_collect_k_rubles_l395_395912


namespace height_of_E_l395_395114

variable {h_E h_F h_G h_H : ℝ}

theorem height_of_E (h1 : h_E + h_F + h_G + h_H = 2 * (h_E + h_F))
                    (h2 : (h_E + h_F) / 2 = (h_E + h_G) / 2 - 4)
                    (h3 : h_H = h_E - 10)
                    (h4 : h_F + h_G = 288) :
  h_E = 139 :=
by
  sorry

end height_of_E_l395_395114


namespace xiao_zhang_payment_l395_395249

/--
In Huanggang City, a supermarket offers discounts to customers under the following conditions:
① If the purchase is less than 100 yuan, no discount is offered;
② For purchases of at least 100 yuan but not exceeding 500 yuan, a 10% discount is applied;
③ For purchases exceeding 500 yuan, the first 500 yuan receives a 10% discount, and the amount exceeding 500 yuan receives a 20% discount.
Given that Xiao Li went shopping at the supermarket twice, paying 99 yuan and 530 yuan respectively,
prove that Xiao Zhang needs to pay 609.2 yuan or 618 yuan if buying the same amount of goods in one go that Xiao Li bought in two trips.
-/
theorem xiao_zhang_payment
  (x : ℝ)
  (y : ℝ)
  (H1 : 0 ≤ x ∧ x < 100 → y = x)
  (H2 : 100 ≤ x ∧ x ≤ 500 → y = 0.9 * x)
  (H3 : x > 500 → y = 0.8 * (x - 500) + 0.9 * 500)
  (pay1 : 99) (pay2 : 530)
  (original1 : ℝ := 99)
  (original2 : ℝ := (530 - 500) / 0.8 + 500)
  (total_original : ℝ := original1 + original2) :
  (0 < total_original ∧ total_original < 100 → total_y = total_original) ∨
  (100 ≤ total_original ∧ total_original ≤ 500 → total_y = 0.9 * total_original) ∨
  (total_original > 500 → total_y = 0.8 * (total_original - 500) + 0.9 * 500) →
  (total_y = 609.2 ∨ total_y = 618) :=
sorry

end xiao_zhang_payment_l395_395249


namespace tetrahedron_volume_l395_395845

theorem tetrahedron_volume (S R V : ℝ) (h : V = (1/3) * S * R) : 
  V = (1/3) * S * R := 
by 
  sorry

end tetrahedron_volume_l395_395845


namespace common_internal_tangent_length_l395_395850

-- Definitions based on conditions
def center_distance : ℝ := 50
def radius1 : ℝ := 7
def radius2 : ℝ := 10

-- Theorem stating the length of the common internal tangent
theorem common_internal_tangent_length (A B : Type)
  (circle1 : A → ℝ) (circle2 : B → ℝ)
  (d : ℝ := center_distance)
  (r1 : ℝ := radius1)
  (r2 : ℝ := radius2) : 
  let tangent_length := Real.sqrt (d^2 - (r1 + r2)^2)
  in tangent_length = Real.sqrt 2211 := by
  sorry

end common_internal_tangent_length_l395_395850


namespace M_squared_identity_l395_395294

variable {a b c : ℂ}

def M : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, b, c], ![b, c, a], ![c, a, b]]

theorem M_squared_identity (a b c : ℂ) (h1 : M ^ 2 = Matrix.identity 3) (h2 : a * b * c = 1) :
  a ^ 3 + b ^ 3 + c ^ 3 = 2 ∨ a ^ 3 + b ^ 3 + c ^ 3 = 4 :=
by
  sorry

end M_squared_identity_l395_395294


namespace jack_paid_no_more_than_jill_l395_395283

theorem jack_paid_no_more_than_jill :
  let total_cost := 15
  let slices := 12
  let cost_per_slice := total_cost / slices
  let jack_paid := 6 * cost_per_slice
  let jill_paid := 6 * cost_per_slice
  jack_paid - jill_paid = 0 :=
by
  /(∴//s⁄*current slicing mode.)
  let total_cost := 15
  let slices := 12
  let cost_per_slice := total_cost / slices
  let jack_paid := 6 * cost_per_slice
  let jill_paid := 6 * cost_per_slice
  sorry

end jack_paid_no_more_than_jill_l395_395283


namespace min_distance_correct_l395_395679

noncomputable def min_distance_from_circle_to_line := 
  let circle_radius := λ θ : ℝ, cos θ + sin θ
  let line_eq := λ θ : ℝ, 2 * sqrt 2 / cos (θ + π / 4)
  let center_of_circle := (1 / 2, 1 / 2)
  let line_cartesian := (1, -1, -4) -- coefficients (a, b, c) for the line ax + by + c = 0
  let circle_radius_value := 1 / sqrt 2
  let distance_to_line (x : ℝ) (y : ℝ) := abs (1 * x - 1 * y - 4) / sqrt (1^2 + (-1)^2)
  let distance_center_to_line := distance_to_line (1 / 2) (1 / 2)
  let minimum_distance := distance_center_to_line - circle_radius_value
  minimum_distance

theorem min_distance_correct : 
  min_distance_from_circle_to_line = 3 * sqrt 2 / 4 := 
begin
  sorry
end

end min_distance_correct_l395_395679


namespace min_positive_period_pi_not_center_of_symmetry_not_axis_of_symmetry_monotonicity_interval_l395_395218
   
noncomputable def f (x : ℝ) := 2 * sin x * cos x + 2 * sqrt 3 * (sin x)^2

theorem min_positive_period_pi : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = π := sorry

theorem not_center_of_symmetry : ¬ (∃ (c : ℝ) (d : ℝ), c = π / 6 ∧ d = 0 ∧ ∀ x : ℝ, f (2 * c - x) = 2 * d - f x) := sorry

theorem not_axis_of_symmetry : ¬ ∃ (c : ℝ), c = π / 12 ∧ ∀ x : ℝ, f (2 * c - x) = f x := sorry

theorem monotonicity_interval : ∀ x, (π / 6) < x ∧ x < (5 * π / 12) → f' x > 0 := sorry

end min_positive_period_pi_not_center_of_symmetry_not_axis_of_symmetry_monotonicity_interval_l395_395218


namespace find_M_l395_395301

def T : Finset ℕ := Finset.range 11

def elements_in_T : Finset ℝ := T.map ⟨λ x, 3^x, by intros; apply Nat.decidableEq⟩

def positive_differences (set: Finset ℝ) : Finset ℝ :=
  set.product set |>.filter (λ pair, pair.1 > pair.2) |>.map (λ pair, pair.1 - pair.2) 

def M := (positive_differences elements_in_T).sum id

theorem find_M : M = 793168 :=
by
  sorry

end find_M_l395_395301


namespace find_a_b_l395_395608

def z : ℂ := 1 - 2 * I

def conjugate_z : ℂ := conj z

theorem find_a_b (a b : ℝ) (hz_eq : z + a * conjugate_z + b = 0) : a = 1 ∧ b = -2 :=
by
  sorry

end find_a_b_l395_395608


namespace tangent_and_parallel_l395_395021

noncomputable def parabola1 (x : ℝ) (b1 c1 : ℝ) : ℝ := -x^2 + b1 * x + c1
noncomputable def parabola2 (x : ℝ) (b2 c2 : ℝ) : ℝ := -x^2 + b2 * x + c2
noncomputable def parabola3 (x : ℝ) (b3 c3 : ℝ) : ℝ := x^2 + b3 * x + c3

theorem tangent_and_parallel (b1 b2 b3 c1 c2 c3 : ℝ) :
  (b3 - b1)^2 = 8 * (c3 - c1) → (b3 - b2)^2 = 8 * (c3 - c2) →
  ((b2^2 - b1^2 + 2 * b3 * (b2 - b1)) / (4 * (b2 - b1))) = 
  ((4 * (c1 - c2) - 2 * b3 * (b1 - b2)) / (2 * (b2 - b1))) :=
by
  intros h1 h2
  sorry

end tangent_and_parallel_l395_395021


namespace find_maximum_k_l395_395027

theorem find_maximum_k {k : ℝ} 
  (h_eq : ∀ x, x^2 + k * x + 8 = 0)
  (h_roots_diff : ∀ x₁ x₂, x₁ - x₂ = 10) :
  k = 2 * Real.sqrt 33 := 
sorry

end find_maximum_k_l395_395027


namespace find_sol_y_pct_l395_395834

-- Define the conditions
def sol_x_vol : ℕ := 200            -- Volume of solution x in milliliters
def sol_y_vol : ℕ := 600            -- Volume of solution y in milliliters
def sol_x_pct : ℕ := 10             -- Percentage of alcohol in solution x
def final_sol_pct : ℕ := 25         -- Percentage of alcohol in the final solution
def final_sol_vol := sol_x_vol + sol_y_vol -- Total volume of the final solution

-- Define the problem statement
theorem find_sol_y_pct (sol_x_vol sol_y_vol final_sol_vol : ℕ) 
  (sol_x_pct final_sol_pct : ℕ) : 
  (600 * 10 + sol_y_vol * 30) / 800 = 25 :=
by
  sorry

end find_sol_y_pct_l395_395834


namespace seventh_day_price_percentage_l395_395517

theorem seventh_day_price_percentage 
  (original_price : ℝ)
  (day1_reduction : original_price * 0.91)
  (day2_increase : day1_reduction * 1.05)
  (day3_reduction : day2_increase * 0.90)
  (day4_increase : day3_reduction * 1.15)
  (day5_reduction : day4_increase * 0.90)
  (day6_increase : day5_reduction * 1.08)
  (day7_reduction : day6_increase * 0.88) :
  day7_reduction / original_price * 100 = 84.59 :=
begin
  sorry
end

end seventh_day_price_percentage_l395_395517


namespace length_EF_eq_2_sqrt_39_div_3_l395_395765

noncomputable def length_of_segment_EF (x y : ℝ) : ℝ :=
  if (x + √3 * y - 2 = 0) ∧ ((x - 4) ^ 2 + y ^ 2 ≥ 4 * (x ^ 2 + y ^ 2))
  then 2 * sqrt (39) / 3
  else 0

theorem length_EF_eq_2_sqrt_39_div_3 (x y : ℝ) :
  (x ^ 2 + y ^ 2 = 1) → 
  ((x - 4) ^ 2 + y ^ 2 = 4) → 
  (x + √3 * y - 2 = 0) →
  ((x - 4) ^ 2 + y ^ 2 ≥ 4 * (x ^ 2 + y ^ 2)) →
  length_of_segment_EF x y = 2 * sqrt (39) / 3 :=
by {
  -- actual proof would go here
  sorry
}

end length_EF_eq_2_sqrt_39_div_3_l395_395765


namespace maximize_probability_when_C_second_game_l395_395956

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l395_395956


namespace product_divisibility_l395_395206

theorem product_divisibility 
  (k m n : ℕ ) 
  (h_prime : Nat.Prime (m + k + 1))
  (h_gt : m + k + 1 > n + 1) 
  (C : ℕ → ℕ) 
  (hC : ∀ s, C s = s * (s + 1)) : 
  ∃ d : ℕ, d > 0 ∧ (∀ (a : ℕ), a ∣ (C (m + a) - C k) ) → 
  (∏ i in Finset.range n, C (m + i + 1) - C k) % (∏ i in Finset.range n, C (i + 1)) = 0 :=
by
  sorry

end product_divisibility_l395_395206


namespace probability_safe_flight_l395_395469

theorem probability_safe_flight : 
  let edge_length := 2
      cube_volume := edge_length^3
      sphere_radius := edge_length / 2
      sphere_volume := (4/3) * Real.pi * (sphere_radius^3)
  in (sphere_volume / cube_volume) = Real.pi / 6 := by
  sorry

end probability_safe_flight_l395_395469


namespace solve_for_a_b_l395_395634

def complex_num : Type := ℂ

-- Given conditions
def z : complex_num := 1 - 2 * complex.I
def a : ℝ := 1  -- Real part solutions
def b : ℝ := -2 -- Real part solutions

def conjugate (z : complex_num) : complex_num := complex.conj z

theorem solve_for_a_b (z : complex_num) (a b : ℝ):
  z = 1 - 2 * complex.I →
  z + a * conjugate z + b = 0 →
  a = 1 ∧ b = -2 := by
  sorry

end solve_for_a_b_l395_395634


namespace find_positive_number_l395_395736

theorem find_positive_number (m : ℝ) 
  (h : (m - 1)^2 = (3 * m - 5)^2) : 
  (m - 1)^2 = 1 ∨ (m - 1)^2 = 1 / 4 :=
by sorry

end find_positive_number_l395_395736


namespace max_prob_two_consecutive_wins_l395_395969

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l395_395969


namespace rihanna_money_left_l395_395824

theorem rihanna_money_left :
  ∀ (initial_amount mangoes apple_juice mango_cost juice_cost : ℕ),
    initial_amount = 50 →
    mangoes = 6 →
    apple_juice = 6 →
    mango_cost = 3 →
    juice_cost = 3 →
    initial_amount - (mangoes * mango_cost + apple_juice * juice_cost) = 14 :=
begin
  intros,
  sorry
end

end rihanna_money_left_l395_395824


namespace maximize_probability_l395_395941

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l395_395941


namespace solve_for_x_l395_395837

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l395_395837


namespace find_a_b_l395_395590

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395590


namespace magician_inequality_l395_395989

theorem magician_inequality (N : ℕ) : 
  (N - 1) * 10^(N - 2) ≥ 10^N → N ≥ 101 :=
by
  sorry

end magician_inequality_l395_395989


namespace louis_bought_five_yards_l395_395324

/-
The cost per yard of velvet fabric is $24.
The cost of the pattern is $15.
The cost per spool of thread is $3.
Louis bought two spools of thread.
The total amount spent is $141.
Prove: Louis bought 5 yards of fabric.
-/

def cost_per_yard := 24
def pattern_cost := 15
def spool_cost := 3
def num_spools := 2
def total_spent := 141

theorem louis_bought_five_yards :
  let thread_cost := num_spools * spool_cost in
  let non_fabric_cost := pattern_cost + thread_cost in
  let fabric_cost := total_spent - non_fabric_cost in
  fabric_cost / cost_per_yard = 5 :=
by
  sorry

end louis_bought_five_yards_l395_395324


namespace find_a_b_l395_395591

noncomputable def complexExample (z : ℂ) (a b : ℝ) : Prop :=
  z = 1 - 2 * complex.I ∧ z + a * complex.conj z + b = 0 

theorem find_a_b (a b : ℝ) : complexExample (1 - 2 * complex.I) a b → a = 1 ∧ b = -2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end find_a_b_l395_395591


namespace chord_line_equation_l395_395241

theorem chord_line_equation (x y : ℝ) 
  (ellipse : ∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1)
  (bisect_point : x / 2 = 4 ∧ y / 2 = 2) : 
  x + 2 * y - 8 = 0 :=
sorry

end chord_line_equation_l395_395241


namespace tim_weekly_payment_l395_395880

-- Define the given conditions
def hourly_rate_bodyguard : ℕ := 20
def number_bodyguards : ℕ := 2
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7

-- Define the total weekly payment calculation
def weekly_payment : ℕ := (hourly_rate_bodyguard * number_bodyguards) * hours_per_day * days_per_week

-- The proof statement
theorem tim_weekly_payment : weekly_payment = 2240 := by
  sorry

end tim_weekly_payment_l395_395880


namespace fixed_constant_t_l395_395410

-- Representation of point on the Cartesian plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the parabola y = 4x^2
def parabola (p : Point) : Prop := p.y = 4 * p.x^2

-- Definition of distance squared between two points
def distance_squared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Main theorem statement
theorem fixed_constant_t :
  ∃ (c : ℝ) (C : Point), c = 1/8 ∧ C = ⟨1, c⟩ ∧ 
  (∀ (A B : Point), parabola A ∧ parabola B ∧ 
  (∃ m k : ℝ, A.y = m * A.x + k ∧ B.y = m * B.x + k ∧ k = c - m) → 
  (1 / distance_squared A C + 1 / distance_squared B C = 16)) :=
by {
  -- Proof omitted
  sorry
}

end fixed_constant_t_l395_395410


namespace find_a_and_b_l395_395601

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def z_conj : ℂ := Complex.conj z

theorem find_a_and_b 
  (a b : ℝ)
  (h : z + a * z_conj + b = 0) :
  a = 1 ∧ b = -2 := 
sorry

end find_a_and_b_l395_395601


namespace final_cost_is_30_l395_395036

-- Define conditions as constants
def cost_of_repair : ℝ := 7
def sales_tax : ℝ := 0.50
def number_of_tires : ℕ := 4

-- Define the cost for one tire repair
def cost_one_tire : ℝ := cost_of_repair + sales_tax

-- Define the cost for all tires
def total_cost : ℝ := cost_one_tire * number_of_tires

-- Theorem stating that the total cost is $30
theorem final_cost_is_30 : total_cost = 30 :=
by
  sorry

end final_cost_is_30_l395_395036


namespace sqrt_cosine_identity_l395_395544

theorem sqrt_cosine_identity :
  Real.sqrt ((3 - Real.cos (Real.pi / 8)^2) * (3 - Real.cos (3 * Real.pi / 8)^2)) = (3 * Real.sqrt 5) / 4 :=
by
  sorry

end sqrt_cosine_identity_l395_395544


namespace common_difference_of_arithmetic_sequence_l395_395684

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

theorem common_difference_of_arithmetic_sequence
  (a d : α)
  (sum_odd : α)
  (sum_even : α)
  (h_odd : sum_odd = (arithmetic_sequence a d 1) + (arithmetic_sequence a d 3)
                    + (arithmetic_sequence a d 5) + (arithmetic_sequence a d 7)
                    + (arithmetic_sequence a d 9))
  (h_even : sum_even = (arithmetic_sequence a d 2) + (arithmetic_sequence a d 4)
                     + (arithmetic_sequence a d 6) + (arithmetic_sequence a d 8)
                     + (arithmetic_sequence a d 10)) :
  sum_odd = 15 →
  sum_even = 30 →
  d = 3 :=
by
  intros h_sum_odd h_sum_even
  have h : 30 - 15 = (arithmetic_sequence a d 2 - arithmetic_sequence a d 1) 
                       + (arithmetic_sequence a d 4 - arithmetic_sequence a d 3)
                       + (arithmetic_sequence a d 6 - arithmetic_sequence a d 5)
                       + (arithmetic_sequence a d 8 - arithmetic_sequence a d 7)
                       + (arithmetic_sequence a d 10 - arithmetic_sequence a d 9), 
    sorry
  simp [h_sum_odd, h_sum_even, arithmetic_sequence] at h,
  linarith,
  sorry

end common_difference_of_arithmetic_sequence_l395_395684


namespace even_number_of_solutions_l395_395280

theorem even_number_of_solutions :
  (∃ sols : Finset (ℝ × ℝ), ∀ (x y : ℝ), ((y ^ 2 + 6) * (x - 1) = y * (x ^ 2 + 1) ∧ (x ^ 2 + 6) * (y - 1) = x * (y ^ 2 + 1)) ↔ (x, y) ∈ sols) →
  sols.card.even :=
begin
  sorry
end

end even_number_of_solutions_l395_395280


namespace lambda_sum_constant_l395_395702

-- Define the given conditions for the ellipse
def center_at_origin (C : Type) := ∀ (x y : ℝ), C = (0, 0)
def foci_on_x_axis (F : Type) := ∀ (c : ℝ), F = ((c, 0), (-c, 0))
def minor_axis_length (l : Type) := l = 2
def eccentricity (e : ℝ) := e = 2 * real.sqrt 5 / 5

-- Define the properties of the line passing through the right focus and intersections
def line_through_focus (l : Type)
  (F : (ℝ × ℝ))
  (x_1 x_2 y_1 y_2 : ℝ) : Prop :=
  -- Conditions that line passes through the right focus and intersects ellipse C
  l = F.1 ∧ ∃ (k : ℝ), ∀ (A B M : (ℝ × ℝ)), 
     A = (x_1, y_1) ∧ B = (x_2, y_2) ∧ M = (0, y_1)

-- State the final condition to prove
theorem lambda_sum_constant (l : Type) (λ_1 λ_2 m : ℝ)
  [center_at_origin C] [foci_on_x_axis F] [minor_axis_length l] [eccentricity e] :
  ∀ (x_1 x_2 : ℝ), ∃ (λ_1 λ_2 : ℝ), 
    (λ_1 + λ_2 = -10) :=
begin
  sorry,
end

end lambda_sum_constant_l395_395702


namespace finalCostCalculation_l395_395039

-- Define the inputs
def tireRepairCost : ℝ := 7
def salesTaxPerTire : ℝ := 0.50
def numberOfTires : ℕ := 4

-- The total cost should be $30
theorem finalCostCalculation : 
  let repairTotal := tireRepairCost * numberOfTires
  let salesTaxTotal := salesTaxPerTire * numberOfTires
  repairTotal + salesTaxTotal = 30 := 
by {
  sorry
}

end finalCostCalculation_l395_395039


namespace exist_alpha_beta_l395_395793

variables {a b : ℝ} {f : ℝ → ℝ}

-- Assume that f has the Intermediate Value Property (for simplicity, define it as a predicate)
def intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ k ∈ Set.Icc (min (f a) (f b)) (max (f a) (f b)),
    ∃ c ∈ Set.Ioo a b, f c = k

-- Assume the conditions from the problem
variables (h_ivp : intermediate_value_property f a b) (h_sign_change : f a * f b < 0)

-- The theorem we need to prove
theorem exist_alpha_beta (hivp : intermediate_value_property f a b) (hsign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end exist_alpha_beta_l395_395793


namespace complex_exponents_sum_l395_395554

theorem complex_exponents_sum (i : ℂ) (h_i2 : i^2 = -1) (h_i4 : i^4 = 1) : 
  i^10 + i^22 + i^{-34} = -3 :=
by
  sorry

end complex_exponents_sum_l395_395554


namespace factorize_problem1_factorize_problem2_l395_395143

-- Problem 1: Factorization of 4x^2 - 16
theorem factorize_problem1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Factorization of a^2b - 4ab + 4b
theorem factorize_problem2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2) ^ 2 :=
by
  sorry

end factorize_problem1_factorize_problem2_l395_395143


namespace perimeter_of_rhombus_l395_395377

theorem perimeter_of_rhombus (d1 d2 : ℝ) (hd1 : d1 = 8) (hd2 : d2 = 30) :
  (perimeter : ℝ) = 4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) :=
by
  simp [hd1, hd2]
  sorry

end perimeter_of_rhombus_l395_395377


namespace number_of_possible_flags_l395_395132

def colors : List String := ["purple", "gold"]

noncomputable def num_choices_per_stripe (colors : List String) : Nat := 
  colors.length

theorem number_of_possible_flags :
  (num_choices_per_stripe colors) ^ 3 = 8 := 
by
  -- Proof
  sorry

end number_of_possible_flags_l395_395132


namespace evaluate_expression_l395_395142

theorem evaluate_expression : (24 : ℕ) = 2^3 * 3 ∧ (72 : ℕ) = 2^3 * 3^2 → (24^40 / 72^20 : ℚ) = 2^60 :=
by {
  sorry
}

end evaluate_expression_l395_395142


namespace find_principal_amount_l395_395378

theorem find_principal_amount
  (r : ℝ := 0.05)  -- Interest rate (5% per annum)
  (t : ℕ := 2)    -- Time period (2 years)
  (diff : ℝ := 20) -- Given difference between CI and SI
  (P : ℝ := 8000) -- Principal amount to prove
  : P * (1 + r) ^ t - P - P * r * t = diff :=
by
  sorry

end find_principal_amount_l395_395378


namespace product_of_5_consecutive_numbers_not_square_l395_395354

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l395_395354


namespace tangent_line_computation_l395_395713

variables (f : ℝ → ℝ)

theorem tangent_line_computation (h_tangent : ∀ x, (f x = -x + 8) ∧ (∃ y, y = -x + 8 → (f y) = -x + 8 → deriv f x = -1)) :
    f 5 + deriv f 5 = 2 :=
sorry

end tangent_line_computation_l395_395713


namespace product_of_5_consecutive_numbers_not_square_l395_395355

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l395_395355


namespace star_fraction_simplified_value_l395_395018

-- Define the operation star
def star (a b : ℚ) : ℚ :=
  let ⟨m, n, _, _⟩ := a.num_denom in
  let ⟨p, q, _, _⟩ := b.num_denom in
  (m + 1) * (p - 1) * (q + 1) / (n - 1)

-- Given fractions
def frac1 : ℚ := 5 / 7
def frac2 : ℚ := 9 / 4

-- The proof statement
theorem star_fraction_simplified_value :
  star frac1 frac2 = 40 := 
sorry

end star_fraction_simplified_value_l395_395018


namespace Q_coordinates_l395_395686

def P : (ℝ × ℝ) := (2, -6)

def Q (x : ℝ) : (ℝ × ℝ) := (x, -6)

axiom PQ_parallel_to_x_axis : ∀ x, Q x = (x, -6)

axiom PQ_length : dist (Q 0) P = 2 ∨ dist (Q 4) P = 2

theorem Q_coordinates : Q 0 = (0, -6) ∨ Q 4 = (4, -6) :=
by {
  sorry
}

end Q_coordinates_l395_395686


namespace range_of_omega_l395_395243

noncomputable def function_is_monotonic (ω : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (x2 < π / 3) → 
    sin (ω * x1 + π / 6) ≤ sin (ω * x2 + π / 6)

theorem range_of_omega (ω : ℝ) (hω : 0 < ω) (hmono : function_is_monotonic ω) : ω ≤ 1 :=
sorry

end range_of_omega_l395_395243


namespace tim_weekly_payment_l395_395882

-- Define the given conditions
def hourly_rate_bodyguard : ℕ := 20
def number_bodyguards : ℕ := 2
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7

-- Define the total weekly payment calculation
def weekly_payment : ℕ := (hourly_rate_bodyguard * number_bodyguards) * hours_per_day * days_per_week

-- The proof statement
theorem tim_weekly_payment : weekly_payment = 2240 := by
  sorry

end tim_weekly_payment_l395_395882
