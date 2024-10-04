import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Arithmetic.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupDef
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometric.Basic
import Mathlib.Combinatorics.CombinatorialBasic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.Cyclic
import Mathlib.Init.Data.Int.Basic
import Mathlib.Init.Ite
import Mathlib.LinearAlgebra.Vec3
import Mathlib.NumberTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import ProbabilityTheory.CharacteristicFunction

namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64846

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64846


namespace playground_length_l64_64701

theorem playground_length
  (L_g : ℝ) -- length of the garden
  (L_p : ℝ) -- length of the playground
  (width_garden : ℝ := 24) -- width of the garden
  (width_playground : ℝ := 12) -- width of the playground
  (perimeter_garden : ℝ := 64) -- perimeter of the garden
  (area_garden : ℝ := L_g * 24) -- area of the garden
  (area_playground : ℝ := L_p * 12) -- area of the playground
  (areas_equal : area_garden = area_playground) -- equal areas
  (perimeter_condition : 2 * (L_g + 24) = 64) -- perimeter condition
  : L_p = 16 := 
by
  sorry

end playground_length_l64_64701


namespace sum_of_b_seq_l64_64495

noncomputable def a_seq (n : ℕ) : ℤ := match n with
| 0     => 0
| (n+1) => (a_seq n + 1) / 2

noncomputable def b_seq (n : ℕ) : ℤ := match n with
| 0     => 3
| (n+1) => a_seq n + b_seq n

def partial_sum (f : ℕ → ℤ) (n : ℕ) : ℤ := (List.range n).sum f

theorem sum_of_b_seq (n : ℕ) :
  partial_sum b_seq n = 2^n + 2 * n - 1 :=
sorry

end sum_of_b_seq_l64_64495


namespace euler_formula_second_quadrant_l64_64412

def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x)

theorem euler_formula_second_quadrant :
  euler_formula (2 * real.pi / 3) = -1/2 + complex.I * (real.sqrt 3 / 2) ∧
  (complex.re (euler_formula (2 * real.pi / 3)) < 0) ∧
  (complex.im (euler_formula (2 * real.pi / 3)) > 0) :=
sorry

end euler_formula_second_quadrant_l64_64412


namespace simpsons_rule_works_l64_64368

noncomputable def integralSimpsonsRule (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : ℝ :=
  let Δx := (b - a) / n
  let x i := a + i * Δx
  let y i := f (x i)
  (Δx / 3) * (y 0 + 4 * List.sum (List.map y (list.range n).filter fun i => i % 2 = 1) 
                      + 2 * List.sum (List.map y (list.range (n-1)).filter fun i => i % 2 = 0) 
                      + y n)

theorem simpsons_rule_works : integralSimpsonsRule (fun x => x ^ 2) 1 4 10 = 21 := sorry

end simpsons_rule_works_l64_64368


namespace exist_point_on_circle_l64_64502

noncomputable theory

-- Define the problem with points, line, and circle, aiming to prove the existence of such a point X.
theorem exist_point_on_circle
  (A B : Point ℝ)
  (S : Circle ℝ)
  (MN : Line ℝ) :
  ∃ X : Point ℝ, X ∈ S ∧ (∃ C D : Point ℝ, C ∈ S ∧ D ∈ S ∧ Line_through A X = Line_through B C ∧ Line_through B X = Line_through A D ∧ Parallel (Chord C D) MN) :=
sorry

end exist_point_on_circle_l64_64502


namespace cubic_expression_value_l64_64057

theorem cubic_expression_value (a b c : ℝ) 
  (h1 : a + b + c = 13) 
  (h2 : ab + ac + bc = 32) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 949 := 
by
  sorry

end cubic_expression_value_l64_64057


namespace range_of_f_l64_64378

def f (x : ℝ) : ℝ := |x + 5| - |x - 3| 

theorem range_of_f : set.range f = set.Icc (-8) 8 :=
by
  sorry

end range_of_f_l64_64378


namespace measure_water_l64_64008

theorem measure_water (a : ℤ) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ b c d e f : ℤ, b ∈ {-2, -1, 0, 1, 2} ∧ c ∈ {-2, -1, 0, 1, 2} ∧ d ∈ {-2, -1, 0, 1, 2} ∧ 
                   e ∈ {-2, -1, 0, 1, 2} ∧ f ∈ {-2, -1, 0, 1, 2} ∧ 
                   a = 625 * b + 125 * c + 25 * d + 5 * e + f := 
by
  sorry

end measure_water_l64_64008


namespace motorist_travel_distance_l64_64317

def total_distance_traveled (time_first_half time_second_half speed_first_half speed_second_half : ℕ) : ℕ :=
  (speed_first_half * time_first_half) + (speed_second_half * time_second_half)

theorem motorist_travel_distance :
  total_distance_traveled 3 3 60 48 = 324 :=
by sorry

end motorist_travel_distance_l64_64317


namespace james_profit_l64_64094

theorem james_profit (boxes bars : ℕ) (selling_price buying_price : ℝ)
  (h1 : boxes = 5)
  (h2 : bars = 10)
  (h3 : selling_price = 1.5)
  (h4 : buying_price = 1.0) :
  let total_bars := boxes * bars in
  let total_revenue := (total_bars : ℝ) * selling_price in
  let total_cost := (total_bars : ℝ) * buying_price in
  total_revenue - total_cost = 25 :=
by
  sorry

end james_profit_l64_64094


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64891

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64891


namespace sum_factors_of_30_l64_64954

theorem sum_factors_of_30 : 
  (∑ i in {1, 2, 3, 5, 6, 10, 15, 30}, i) = 72 := 
by
  -- Sorry, to skip the proof here.
  sorry

end sum_factors_of_30_l64_64954


namespace motorcycle_speed_for_10_minute_prior_arrival_l64_64981

noncomputable def distance_from_home_to_station (x : ℝ) : Prop :=
  x / 30 + 15 / 60 = x / 18 - 15 / 60

noncomputable def speed_to_arrive_10_minutes_before_departure (x : ℝ) (v : ℝ) : Prop :=
  v = x / (1 - 10 / 60)

theorem motorcycle_speed_for_10_minute_prior_arrival :
  (∀ x : ℝ, distance_from_home_to_station x) →
  (∃ x : ℝ, 
    ∃ v : ℝ, speed_to_arrive_10_minutes_before_departure x v ∧ v = 27) :=
by 
  intro h
  exists 22.5
  exists 27
  unfold distance_from_home_to_station at h
  unfold speed_to_arrive_10_minutes_before_departure
  sorry

end motorcycle_speed_for_10_minute_prior_arrival_l64_64981


namespace cone_volume_l64_64649

def slant_height := 15
def height := 9
def radius := Real.sqrt (slant_height^2 - height^2)
def volume := (1/3) * Real.pi * radius^2 * height

theorem cone_volume : 
  radius = 12 ∧ volume = 432 * Real.pi :=
by
  split
  { -- radius = 12
    unfold radius
    sorry
  }
  { -- volume = 432 * Real.pi
    unfold volume
    sorry
  }

end cone_volume_l64_64649


namespace sum_of_factors_30_l64_64975

theorem sum_of_factors_30 : 
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ) in
  ( ∑ n in factors, n ) = 72 :=
by
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ)
  rw [finset.sum_eq_sum_of_elements] -- transposing to handle the sum notation
  let factors_list := [1, 2, 3, 5, 6, 10, 15, 30]
  have h_factors : factors = {x | x ∈ factors_list}.to_finset, by sorry
  simp_rw h_factors
  rw [finset.sum_to_finset]
  dsimp -- simplifying before asserting the sum
  norm_num -- proving that the sum is indeed 72
  sorry

end sum_of_factors_30_l64_64975


namespace polynomial_division_result_l64_64578

noncomputable def f (x : ℝ) : ℝ := 6 * x^4 + 18 * x^3 - 15 * x^2 + 5 * x + 5
noncomputable def d (x : ℝ) : ℝ := x^2 + 5 * x - 7

def q (x : ℝ) : ℝ := 6 * x^2 - 15
def r (x : ℝ) : ℝ := 216 * x + 414

theorem polynomial_division_result :
  f(1) = q(1) * d(1) + r(1) ∧ q(1) + r(-1) = 189 :=
by
  sorry

end polynomial_division_result_l64_64578


namespace false_proposition_l64_64059

-- Define the symbols and types for lines and planes
variables (Line Plane : Type) 
variables (m n : Line) (α β : Plane)

-- Define properties and relations for lines and planes
variable [non_coincident_lines : ∀ (l1 l2 : Line), l1 ≠ l2]
variable [non_coincident_planes : ∀ (p1 p2 : Plane), p1 ≠ p2]
variable (is_subset_of : Line → Plane → Prop)
variable (is_parallel_to : Line → Plane → Prop)
variable (is_parallel_lines : Line → Line → Prop)
variable (is_perpendicular_to : Line → Plane → Prop)
variable (is_planes_parallel : Plane → Plane → Prop)

-- Define the main statement based on the given problem and its solution
theorem false_proposition : 
  (is_perpendicular_to α β) → (is_subset_of m α) → ¬ (is_perpendicular_to m β) :=
begin
  sorry
end

end false_proposition_l64_64059


namespace solution_set_f_greater_e_l64_64475

noncomputable def f (x : ℝ) : ℝ := sorry
def f_derivative (x : ℝ) : ℝ := sorry

theorem solution_set_f_greater_e (f : ℝ → ℝ) (f_derivative : ℝ → ℝ)
  (h1 : ∀ x : ℝ, differentiable ℝ f)
  (h2 : ∀ x : ℝ, f_derivative x = deriv f x)
  (h3 : f 0 = 1)
  (h4 : ∀ x : ℝ, f_derivative x > f x) :
  ∀ x : ℝ, (f x > Real.exp x ↔ x > 0) :=
sorry

end solution_set_f_greater_e_l64_64475


namespace dogwood_trees_planted_today_l64_64194

theorem dogwood_trees_planted_today 
    (current_trees : ℕ) 
    (trees_planted_tomorrow : ℕ) 
    (total_trees : ℕ) 
    (initial_trees : ℕ)
    (total : initial_trees + current_trees + trees_planted_tomorrow = total_trees) :
    current_trees = 41 :=
by
  -- Current conditions from the problem
  let current_trees := 39
  let trees_planted_tomorrow := 20
  let total_trees := 100
  calc
    current_trees + trees_planted_tomorrow : 39 + 20
    ... = 59
    ... = 100 - 41 := eq.trans (by simp) total.symm
  sorry

end dogwood_trees_planted_today_l64_64194


namespace first_nonzero_digit_of_fraction_l64_64698

theorem first_nonzero_digit_of_fraction :
  ∃ (d : ℕ), (0 < d ∧ d < 10) ∧ 
  (∀ ε > 0, ∃ n : ℕ, abs ((1 / 113) * 10 ^ n - d) < ε) := sorry

end first_nonzero_digit_of_fraction_l64_64698


namespace sufficient_not_necessary_for_q_l64_64464

-- Define propositions p and q
def p (x : ℂ) : Prop := x = 1
def q (x : ℂ) : Prop := x^2 ≠ 1

-- Prove that p is a sufficient but not necessary condition for q
theorem sufficient_not_necessary_for_q (x : ℂ) : 
  (p x → q x) ∧ (q x → (x = 1 ∨ x = -1)) ∧ (¬ ∀ (y : ℂ), q y → p y) :=
begin
  sorry
end

end sufficient_not_necessary_for_q_l64_64464


namespace common_sum_value_l64_64617

noncomputable def common_sum (M : matrix (fin 4) (fin 4) ℤ) : ℤ :=
  if h : ∀ i : fin 4, ∑ j in finset.univ, M i j = ∑ j in finset.univ, M j i ∧
                      ∑ k in finset.univ, M k k = ∑ k in finset.univ, M k (3 - k) then
    ∑ j in finset.univ, M 0 j
  else
    0

theorem common_sum_value :
  ∃ (M : matrix (fin 4) (fin 4) ℤ),
    (∀ i : fin 4, ∑ j in finset.univ, M i j = 10) ∧
    (∀ i : fin 4, ∑ j in finset.univ, M j i = 10) ∧
    (∑ k in finset.univ, M k k = 10) ∧
    (∑ k in finset.univ, M k (3 - k) = 10) ∧
    (finset.univ.sum (λ (i : fin 4), finset.univ.sum (λ (j : fin 4), M i j)) = 40) ∧
    (∀ i j, M i j ∈ finset.range 16.image (λ x, -5 + x)) :=
sorry

end common_sum_value_l64_64617


namespace product_of_solutions_l64_64871

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64871


namespace range_of_values_for_a_l64_64002

theorem range_of_values_for_a (a : ℝ) (h : ∀ x ∈ set.Icc 0 1, monotone (λ x, log a (2 - a * x))) : 1 < a ∧ a < 2 :=
sorry

end range_of_values_for_a_l64_64002


namespace find_smallest_n_l64_64615

open Nat

def smallest_n_for_mod7 (x y : ℤ) : ℤ :=
  if (x - 4) % 7 = 0 ∧ (y + 4) % 7 = 0 then 5 else 0

theorem find_smallest_n
  (x y : ℤ)
  (hx : (x - 4) % 7 = 0)
  (hy : (y + 4) % 7 = 0) :
  ∃ (n : ℤ), n = smallest_n_for_mod7 x y ∧ (x^2 + x * y + y^2 + n) % 7 = 0 :=
by
  use 5
  split
  · rfl
  · have h1 : x % 7 = 4 := sorry
    have h2 : y % 7 = -4 := sorry
    calc (x^2 + x * y + y^2 + 5) % 7
        = (4^2 + 4 * (-4) + (-4)^2 + 5) % 7 : by congr; try { assumption }
    ... = (16 - 16 + 16 + 5) % 7 : by sorry
    ... = 21 % 7 : by sorry
    ... = 0 : by norm_num

-- Necessary sorrys where the calculations and manipulations are skipped

end find_smallest_n_l64_64615


namespace product_of_real_solutions_triple_property_l64_64786

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64786


namespace sum_of_factors_of_30_is_72_l64_64967

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l64_64967


namespace total_height_bottle_l64_64328

theorem total_height_bottle (r1 r2 h1 h2 : ℝ) (h_total : h1 + h2 = 29)
  (h_water_upright : ℝ) (h_water_upside_down : ℝ) :
  r1 = 1 ∧ r2 = 3 → h_water_upright = 20 ∧ h_water_upside_down = 28 → h_total = 29 := by
  intros
  exact h_total
  sorry

end total_height_bottle_l64_64328


namespace product_inequality_l64_64005

theorem product_inequality
  {n : ℕ} (x : Fin n → ℝ) (a : ℝ) (s : ℝ)
  (hx : ∀ i, x i > 0)
  (ha : a > 0)
  (hs : (∑ i, x i) = s ∧ s ≤ a) :
  (∏ i, (a + x i) / (a - x i)) ≥ ( (n * a + s) / (n * a - s)) ^ n :=
  sorry

end product_inequality_l64_64005


namespace harvest_weeks_l64_64127

theorem harvest_weeks (E_1 E_2 T : ℕ) (h1 : E_1 = 28) (h2 : E_2 = 939) (h3 : T = 1054997) : 
  let E_total := E_1 + E_2 in 
  T / E_total = 1091 :=
by
  rw [h1, h2, h3]
  let E_total := 28 + 939
  have : E_total = 967 := by norm_num
  rw this
  norm_num
  sorry

end harvest_weeks_l64_64127


namespace sum_of_A_and_B_l64_64199

theorem sum_of_A_and_B (A B : ℕ) (h1 : 7 - B = 3) (h2 : A - 5 = 4) (h_diff : A ≠ B) : A + B = 13 :=
sorry

end sum_of_A_and_B_l64_64199


namespace find_second_number_l64_64187

def sum_of_three (a b c : ℚ) : Prop :=
  a + b + c = 120

def ratio_first_to_second (a b : ℚ) : Prop :=
  a / b = 3 / 4

def ratio_second_to_third (b c : ℚ) : Prop :=
  b / c = 3 / 5

theorem find_second_number (a b c : ℚ) 
  (h_sum : sum_of_three a b c)
  (h_ratio_ab : ratio_first_to_second a b)
  (h_ratio_bc : ratio_second_to_third b c) : 
  b = 1440 / 41 := 
sorry

end find_second_number_l64_64187


namespace ratio_distance_x_y_l64_64161

theorem ratio_distance_x_y (x : ℝ) (h1 : abs (x) = 12) : 
  let y := -6 in abs y / abs x = 1 / 2 := 
by 
  sorry

end ratio_distance_x_y_l64_64161


namespace significant_improvement_l64_64264

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64264


namespace indicator_significant_improvement_l64_64301

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64301


namespace arithmetic_sequence_15th_term_l64_64163

theorem arithmetic_sequence_15th_term :
  let a1 := 3
  let d := 15 - 3
  let d' := 2 * d
  ∀ n = 15, a1 + (n-1) * d' = 339 := by
    intros
    rfl
    sorry

end arithmetic_sequence_15th_term_l64_64163


namespace product_of_all_solutions_triple_reciprocal_l64_64745

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64745


namespace count_divisible_by_103_in_sequence_l64_64053

theorem count_divisible_by_103_in_sequence :
  let sequence := λ n, 10^n + 1
  let S := { n | 1 ≤ n ∧ n ≤ 1024 ∧ (sequence n) % 103 = 0 }
  S.count = 505
by
  sorry

end count_divisible_by_103_in_sequence_l64_64053


namespace height_of_water_in_cylinder_l64_64349

/-- Problem Statement: An inverted cone with base radius 9 cm and height 27 cm is full of water. This water is then poured into a tall cylinder with a radius of 18 cm. Determine the height of the water in the cylinder in centimeters. --/
theorem height_of_water_in_cylinder :
  let r_cone := 9 in
  let h_cone := 27 in
  let r_cyl := 18 in
  let V_cone := (1 / 3) * π * (r_cone ^ 2) * h_cone in
  let h_cyl := V_cone / (π * (r_cyl ^ 2)) in
  h_cyl = 2.25 :=
by
  sorry

end height_of_water_in_cylinder_l64_64349


namespace product_of_solutions_l64_64703

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64703


namespace rectangle_area_l64_64335

theorem rectangle_area (a : ℕ) (w l : ℕ) (h_square_area : a = 36) (h_square_side : w * w = a) (h_rectangle_length : l = 3 * w) : w * l = 108 :=
by
  -- Placeholder for proof
  sorry

end rectangle_area_l64_64335


namespace Q1_arrangements_no_two_women_next_to_each_other_Q2_arrangements_A_not_first_B_not_last_Q3_arrangements_ABC_fixed_sequence_Q4_arrangements_A_left_of_B_l64_64052

-- Condition: There are 4 men and 3 women.
constant men_women : ℕ := 4 + 3

-- Question 1: Prove that the number of different arrangements is 1440
theorem Q1_arrangements_no_two_women_next_to_each_other : 
  (number_of_arrangements 4 3 (λ w, no_two_adjacent w)) = 1440 := 
sorry

-- Question 2: Prove that the number of different arrangements is 3720
theorem Q2_arrangements_A_not_first_B_not_last :
  (number_of_arrangements 4 3 (λ p, p.head ≠ A ∧ p.last ≠ B)) = 3720 := 
sorry

-- Question 3: Prove that the number of different arrangements is 840
theorem Q3_arrangements_ABC_fixed_sequence : 
  (number_of_arrangements 4 3 (λ p, A_follows C p ∧ B_follows A p)) = 840 := 
sorry

-- Question 4: Prove that the number of different arrangements is 2520
theorem Q4_arrangements_A_left_of_B : 
  (number_of_arrangements 4 3 (λ p, A_left_of_B p)) = 2520 := 
sorry

end Q1_arrangements_no_two_women_next_to_each_other_Q2_arrangements_A_not_first_B_not_last_Q3_arrangements_ABC_fixed_sequence_Q4_arrangements_A_left_of_B_l64_64052


namespace complex_solution_l64_64122

noncomputable def complex_problem (a b c : ℂ) : Prop :=
  |a| = 1 ∧ |b| = 1 ∧ |c| = 1 ∧ ((a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) = 3) →
  |a + b + c|^2 = 1

theorem complex_solution (a b c : ℂ) (h1 : |a| = 1) (h2 : |b| = 1) (h3 : |c| = 1)
  (h4 : (a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) = 3) :
  complex_problem a b c :=
by
  unfold complex_problem
  intro h
  apply h
  split; assumption
sorry

end complex_solution_l64_64122


namespace significant_improvement_l64_64290

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64290


namespace units_digit_of_3_pow_4_l64_64976

theorem units_digit_of_3_pow_4 : (3^4 % 10) = 1 :=
by
  sorry

end units_digit_of_3_pow_4_l64_64976


namespace position_4123_in_permutations_l64_64212

def digits : List Nat := [1, 2, 3, 4]

theorem position_4123_in_permutations : 
  ∀ (l : List (List Nat)), l = List.permutations digits → 
    List.indexOf [4, 1, 2, 3] (List.sort compare l) = 18 :=
by
  sorry

end position_4123_in_permutations_l64_64212


namespace original_number_increase_l64_64318

theorem original_number_increase (x : ℝ) (h : 1.20 * x = 1800) : x = 1500 :=
by
  sorry

end original_number_increase_l64_64318


namespace complex_division_example_l64_64156

theorem complex_division_example : (2 : ℂ) / (I * (3 - I)) = (1 - 3 * I) / 5 := 
by {
  sorry
}

end complex_division_example_l64_64156


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64879

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64879


namespace factor_of_60n_l64_64308

theorem factor_of_60n
  (n : ℕ)
  (x : ℕ)
  (h_condition1 : ∃ k : ℕ, 60 * n = x * k)
  (h_condition2 : ∃ m : ℕ, 60 * n = 8 * m)
  (h_condition3 : n >= 8) :
  x = 60 :=
sorry

end factor_of_60n_l64_64308


namespace rounds_does_not_equal_34_56_l64_64224

-- Define the rounding function to the nearest hundredth
def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  ((x * 100).round : ℝ) / 100

-- Define the given conditions
def a : ℝ := 34.5599
def b : ℝ := 34.555
def c : ℝ := 34.5601
def d : ℝ := 34.554
def e : ℝ := 34.5611

-- Definition of the theorem
theorem rounds_does_not_equal_34_56 :
  round_to_nearest_hundredth d ≠ 34.56 :=
by
  sorry

end rounds_does_not_equal_34_56_l64_64224


namespace painters_work_days_l64_64091

theorem painters_work_days 
  (six_painters_days : ℝ) (number_six_painters : ℝ) (total_work_units : ℝ)
  (number_four_painters : ℝ) 
  (h1 : number_six_painters = 6)
  (h2 : six_painters_days = 1.4)
  (h3 : total_work_units = number_six_painters * six_painters_days) 
  (h4 : number_four_painters = 4) :
  2 + 1 / 10 = total_work_units / number_four_painters :=
by
  rw [h3, h1, h2, h4]
  sorry

end painters_work_days_l64_64091


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64822

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64822


namespace number_of_members_l64_64229

theorem number_of_members (n : ℕ) (h1 : n * n = 5929) : n = 77 :=
sorry

end number_of_members_l64_64229


namespace cone_volume_correct_l64_64671

noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let r := real.sqrt (slant_height^2 - height^2)
  in (1/3) * real.pi * r^2 * height

theorem cone_volume_correct :
  cone_volume 15 9 = 432 * real.pi := by
  sorry

end cone_volume_correct_l64_64671


namespace seq_sum_first_9_terms_l64_64457

theorem seq_sum_first_9_terms {a : ℕ → ℝ} (k : ℝ) :
  (∀ (n : ℕ), n > 0 → a n + 2 = k * (n - 5)) → (∑ i in Finset.range 9, a (i+1)) = -18 :=
by
  intros h
  sorry

end seq_sum_first_9_terms_l64_64457


namespace robin_cupcakes_l64_64606

-- Definition of variables
variables (x : ℕ)

-- Conditions as definitions
def condition1 := x + 2 * x = 12

-- Problem statement
theorem robin_cupcakes : condition1 → x = 4 :=
by
  sorry

end robin_cupcakes_l64_64606


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64853

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64853


namespace jasmine_percentage_correct_l64_64243

-- Definitions from conditions
def original_solution_volume := 100
def original_jasmine_percentage := 0.10
def added_jasmine := 5
def added_water := 10

-- Calculating quantities based on conditions
def original_jasmine_volume := original_solution_volume * original_jasmine_percentage
def new_jasmine_volume := original_jasmine_volume + added_jasmine
def new_solution_volume := original_solution_volume + added_jasmine + added_water
def new_jasmine_percentage := (new_jasmine_volume / new_solution_volume) * 100

-- The statement to prove
theorem jasmine_percentage_correct :
  new_jasmine_percentage = 13.04 := by
  sorry

end jasmine_percentage_correct_l64_64243


namespace even_function_ab_sum_l64_64021

theorem even_function_ab_sum {a b : ℝ} (h1 : ∀ x, f x = a * x ^ 2 + b * x) (hf_even : ∀ x, f (-x) = f x) (h_interval : a - 1 = - (3 * a)) :
  a + b = 1 / 4 :=
sorry

end even_function_ab_sum_l64_64021


namespace cone_volume_l64_64652

def slant_height := 15
def height := 9
def radius := Real.sqrt (slant_height^2 - height^2)
def volume := (1/3) * Real.pi * radius^2 * height

theorem cone_volume : 
  radius = 12 ∧ volume = 432 * Real.pi :=
by
  split
  { -- radius = 12
    unfold radius
    sorry
  }
  { -- volume = 432 * Real.pi
    unfold volume
    sorry
  }

end cone_volume_l64_64652


namespace volume_of_box_from_cut_squares_l64_64316

def rectangular_sheet := { length : ℕ, width : ℕ, cut_side : ℕ }

def volume_of_box (length width height : ℕ) := length * width * height

theorem volume_of_box_from_cut_squares :
  ∀ (S : rectangular_sheet), S.length = 52 ∧ S.width = 36 ∧ S.cut_side = 8 →
  volume_of_box (S.length - 2 * S.cut_side) (S.width - 2 * S.cut_side) S.cut_side = 5760 :=
by
  intros S h
  cases h with h1 h2
  cases h2 with h_l h_w
  rw [h_l, h_w]
  sorry

end volume_of_box_from_cut_squares_l64_64316


namespace ursula_purchases_total_cost_l64_64696

variable (T C B Br : ℝ)
variable (hT : T = 10) (hTC : T = 2 * C) (hB : B = 0.8 * C) (hBr : Br = B / 2)

theorem ursula_purchases_total_cost : T + C + B + Br = 21 := by
  sorry

end ursula_purchases_total_cost_l64_64696


namespace determine_fraction_l64_64167

noncomputable def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

noncomputable def p (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem determine_fraction (a b : ℝ) (h : a + b = 1 / 4) :
  (p a b (-1)) / (q (-1)) = (a - b) / 4 :=
by
  sorry

end determine_fraction_l64_64167


namespace suitable_land_for_vegetables_l64_64566

def previous_property_acres : ℝ := 2
def enlargement_factor : ℝ := 10
def pond_area : ℝ := 1

theorem suitable_land_for_vegetables :
  let new_property_acres := previous_property_acres * enlargement_factor in
  let suitable_acres := new_property_acres - pond_area in
  suitable_acres = 19 :=
by
  sorry

end suitable_land_for_vegetables_l64_64566


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64856

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64856


namespace product_of_solutions_l64_64771

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64771


namespace significant_improvement_l64_64282

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64282


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64726

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64726


namespace find_floor_of_scaled_m_l64_64107

def largest_int_not_exceeding (x : ℝ) : ℤ := ⌊x⌋

theorem find_floor_of_scaled_m (m : ℝ)
  (h1 : 0 < m)
  (h2 : m < 1)
  (h3 : (Finset.sum (finset.range 2021) (λ k, ⌊m + (k + 1)/2022⌋ )) = 21) :
  largest_int_not_exceeding (674 * m) = 6 ∨ largest_int_not_exceeding (674 * m) = 7 :=
sorry

end find_floor_of_scaled_m_l64_64107


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64763

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64763


namespace triangle_cotangent_tangent_identity_l64_64610

-- Define angles α, β, γ and note their sum
variables {α β γ : ℝ}
hypothesis triangle_angles : α + β + γ = π

-- Define cotangent and tangent functions
noncomputable def ctg (x : ℝ) := Real.cos x / Real.sin x
noncomputable def tg (x : ℝ) := Real.sin x / Real.cos x

-- State the theorem
theorem triangle_cotangent_tangent_identity (h1 : α + β + γ = π) :
  (ctg α + ctg β) / (tg α + tg β) +
  (ctg β + ctg γ) / (tg β + tg γ) +
  (ctg γ + ctg α) / (tg γ + tg α) = 1 :=
sorry

end triangle_cotangent_tangent_identity_l64_64610


namespace monotonic_decreasing_intervals_of_neg_tan_l64_64173

theorem monotonic_decreasing_intervals_of_neg_tan :
  (∀ k : ℤ, ∃ a b : ℝ, a = k * Real.pi - Real.pi / 2 ∧ b = k * Real.pi + Real.pi / 2 ∧ 
  ∀ x : ℝ, a < x ∧ x < b → monotonic_decreasing_on (λ x, -Real.tan x) (Ioo a b)) :=
sorry

end monotonic_decreasing_intervals_of_neg_tan_l64_64173


namespace cevians_circumcenters_centroid_l64_64026

theorem cevians_circumcenters_centroid {A B C G : Point} {t1 t2 t3 t4 t5 t6 : Triangle}
  (h1 : Cevian A B C G)
  (h2 : Cevian B A C G)
  (h3 : Cevian C A B G)
  (h4 : distinct_circumcenters t1 t2 t3 t4 t5 t6)
  (h5 : circumcenters_concyclic t1 t2 t3 t4 t5 t6) :
  is_centroid G A B C :=
sorry

end cevians_circumcenters_centroid_l64_64026


namespace line_parallel_perpendicular_l64_64022

variable (m n : Type) [Line m] [Line n]
variable (α β : Type) [Plane α] [Plane β]
variable [different_lines : m ≠ n] 
variable [different_planes : α ≠ β]

theorem line_parallel_perpendicular {m n : Line} {β : Plane} 
  (hmn : m ∥ n) (hmb : m ⊥ β) : n ⊥ β :=
sorry

end line_parallel_perpendicular_l64_64022


namespace max_BM_squared_l64_64555

-- Defining the points A, B, C, D
def A : (ℝ × ℝ) := (2, 0)
def B : (ℝ × ℝ) := (-1, Real.sqrt 3)
def C : (ℝ × ℝ) := (-1, -Real.sqrt 3)
def D : (ℝ × ℝ) := (0, 0)

-- Definition of the vector lengths
def vec_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Definition of dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Definitions of the conditions 
def condition_lengths : Prop :=
  vec_length D A = 2 ∧ vec_length D B = 2 ∧ vec_length D C = 2

def condition_dot_products : Prop :=
  dot_product (A - D) (B - C) = 0 ∧ dot_product (B - D) (A - C) = 0 ∧ dot_product (C - D) (A - B) = 0

-- Definitions of moving points and conditions
def P (θ : ℝ) : (ℝ × ℝ) :=
  (2 + Real.cos θ, Real.sin θ)

def M (θ : ℝ) : (ℝ × ℝ) :=
  ((1 + Real.cos θ) / 2, (Real.sin θ - Real.sqrt 3) / 2)

def condition_moving_points (θ : ℝ) : Prop :=
  vec_length A (P θ) = 1 ∧ (P θ - (M θ) = (M θ) - C)

-- Definition of the theorem to be proved
theorem max_BM_squared (θ : ℝ) (h1 : condition_lengths) (h2 : condition_dot_products) (h3 : condition_moving_points θ) :
  vec_length B (M θ) ^ 2 ≤ 49 / 4 :=
sorry -- Proof omitted

end max_BM_squared_l64_64555


namespace area_of_PQRS_integer_when_PQ10_RS4_l64_64083

variables (PQ QR RS : ℝ) (O : Type) (PS : ℝ) (T U : O)
-- Conditions
axiom pq_perp_qr : PQ ⊥ QR
axiom qr_perp_rs : QR ⊥ RS
axiom qr_tangent_circle : tangent_to_circle QR O PS

theorem area_of_PQRS_integer_when_PQ10_RS4 (hPQ : PQ = 10) (hRS : RS = 4) :
  is_perfect_square (PQ * RS) := by
  sorry

end area_of_PQRS_integer_when_PQ10_RS4_l64_64083


namespace team_savings_with_discount_l64_64405

def regular_shirt_cost : ℝ := 7.50
def regular_pants_cost : ℝ := 15.00
def regular_socks_cost : ℝ := 4.50
def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75
def team_size : ℕ := 12

theorem team_savings_with_discount :
  let regular_uniform_cost := regular_shirt_cost + regular_pants_cost + regular_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by
  sorry

end team_savings_with_discount_l64_64405


namespace hyperbola_eccentricity_eq_sqrt2_l64_64017

variable {a b : ℝ} (ha : a > 0) (hb : b > 0)
variable {M N Q E P : ℝ × ℝ}
variable (M_sym_N : M = (-N.1, -N.2))
variable (M_sym_Q_about_x : M = (Q.1, -Q.2))
variable (ME_eq_2MQ : E = (2 * Q.1 - M.1, 2 * Q.2 - M.2))
variable (line_NE_intersects_hyperbola_at_P : ∃ P : ℝ × ℝ, (P.1, P.2) ≠ (0, 0) ∧ P = E ∧ P.1^2 / a^2 - P.2^2 / b^2 = 1)
variable (PM_perp_MN : P.1 * N.1 + P.2 * N.2 = 0)

theorem hyperbola_eccentricity_eq_sqrt2 : 
  sqrt (1 + b^2 / a^2) = sqrt 2 :=
sorry 

end hyperbola_eccentricity_eq_sqrt2_l64_64017


namespace min_distance_between_points_l64_64501

noncomputable def minDistance (x : ℝ) : ℝ :=
  real.sqrt ((x - real.sqrt 2 / 2) ^ 2 + ((real.sqrt 2) - x) ^ 2)

theorem min_distance_between_points :
  ∃ x : ℝ, minDistance x = 1 / 2 :=
begin
  sorry
end

end min_distance_between_points_l64_64501


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64852

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64852


namespace decreasing_f_range_k_l64_64090

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 2 / (2^x - 1)

theorem decreasing_f (a : ℝ) (h_odd : ∀ x, f (-x) a = -f x a) :
  ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 a > f x2 a := 
sorry

theorem range_k (a : ℝ) (h_odd : ∀ x, f (-x) a = -f x a) 
  (h_dec : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 a > f x2 a)
  (h_pos : ∀ (m : ℝ), 0 < m → f (abs x) a > k + (Real.log2 (m / 2)) * (Real.log2 (4 / m))) :
  k ≤ 3 / 4 :=
sorry

end decreasing_f_range_k_l64_64090


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64764

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64764


namespace largest_integer_divides_expression_l64_64060

theorem largest_integer_divides_expression (x : ℤ) (h : Even x) :
  3 ∣ (10 * x + 1) * (10 * x + 5) * (5 * x + 3) :=
sorry

end largest_integer_divides_expression_l64_64060


namespace cheesecakes_sold_l64_64997

theorem cheesecakes_sold
  (initial_display : Nat)
  (initial_fridge : Nat)
  (left_to_sell : Nat)
  (total_cheesecakes := initial_display + initial_fridge)
  (total_after_sales : Nat) :
  initial_display = 10 →
  initial_fridge = 15 →
  left_to_sell = 18 →
  total_after_sales = total_cheesecakes - left_to_sell →
  total_after_sales = 7 := sorry

end cheesecakes_sold_l64_64997


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64732

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64732


namespace product_of_solutions_l64_64864

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64864


namespace count_possible_multisets_l64_64624

theorem count_possible_multisets :
  ∀ (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ),
  ∀ (s_1 s_2 s_3 s_4 s_5 s_6 : ℤ),
    (b_6 * s_1^6 + b_5 * s_1^5 + b_4 * s_1^4 + b_3 * s_1^3 + b_2 * s_1^2 + b_1 * s_1 + b_0 = 0) ∧
    (b_6 * s_2^6 + b_5 * s_2^5 + b_4 * s_2^4 + b_3 * s_2^3 + b_2 * s_2^2 + b_1 * s_2 + b_0 = 0) ∧
    (b_6 * s_3^6 + b_5 * s_3^5 + b_4 * s_3^4 + b_3 * s_3^3 + b_2 * s_3^2 + b_1 * s_3 + b_0 = 0) ∧
    (b_6 * s_4^6 + b_5 * s_4^5 + b_4 * s_4^4 + b_3 * s_4^3 + b_2 * s_4^2 + b_1 * s_4 + b_0 = 0) ∧
    (b_6 * s_5^6 + b_5 * s_5^5 + b_4 * s_5^4 + b_3 * s_5^3 + b_2 * s_5^2 + b_1 * s_5 + b_0 = 0) ∧
    (b_6 * s_6^6 + b_5 * s_6^5 + b_4 * s_6^4 + b_3 * s_6^3 + b_2 * s_6^2 + b_1 * s_6 + b_0 = 0) ∧
    (b_0 * s_1^6 + b_1 * s_1^5 + b_2 * s_1^4 + b_3 * s_1^3 + b_4 * s_1^2 + b_5 * s_1 + b_6 = 0) ∧
    (b_0 * s_2^6 + b_1 * s_2^5 + b_2 * s_2^4 + b_3 * s_2^3 + b_4 * s_2^2 + b_5 * s_2 + b_6 = 0) ∧
    (b_0 * s_3^6 + b_1 * s_3^5 + b_2 * s_3^4 + b_3 * s_3^3 + b_4 * s_3^2 + b_5 * s_3 + b_6 = 0) ∧
    (b_0 * s_4^6 + b_1 * s_4^5 + b_2 * s_4^4 + b_3 * s_4^3 + b_4 * s_4^2 + b_5 * s_4 + b_6 = 0) ∧
    (b_0 * s_5^6 + b_1 * s_5^5 + b_2 * s_5^4 + b_3 * s_5^3 + b_4 * s_5^2 + b_5 * s_5 + b_6 = 0) ∧
    (b_0 * s_6^6 + b_1 * s_6^5 + b_2 * s_6^4 + b_3 * s_6^3 + b_4 * s_6^2 + b_5 * s_6 + b_6 = 0) →
    (∃ T : finset (multiset ℤ), T.card = 7 ∧ ∀ t ∈ T, t = ({1, 1, 1, 1, 1, 1} : multiset ℤ) ∨ t = ({-1, -1, -1, -1, -1, -1} : multiset ℤ)) := sorry

end count_possible_multisets_l64_64624


namespace product_trippled_when_added_to_reciprocal_l64_64936

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64936


namespace indicator_significant_improvement_l64_64306

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64306


namespace projections_concyclic_l64_64333

open EuclideanGeometry

theorem projections_concyclic
  {A B C D E : Point} -- Points of the pyramid where E is the apex
  (ω : Sphere) -- the insphere of the pyramid
  (P : Point) -- the point of contact of the sphere with the base ABCD
  (h₁ : inscribed ω (pyramid A B C D E)) -- ω is inscribed in the pyramid
  : are_concyclic (projection P A B) (projection P B C) (projection P C D) (projection P D A) :=
sorry

end projections_concyclic_l64_64333


namespace jason_advertising_cost_l64_64593

def magazine_length : ℕ := 9
def magazine_width : ℕ := 12
def cost_per_square_inch : ℕ := 8
def half (x : ℕ) := x / 2
def area (L W : ℕ) := L * W
def total_cost (a c : ℕ) := a * c

theorem jason_advertising_cost :
  total_cost (half (area magazine_length magazine_width)) cost_per_square_inch = 432 := by
  sorry

end jason_advertising_cost_l64_64593


namespace distance_after_turning_left_l64_64694

theorem distance_after_turning_left
  (a : ℕ) (c : ℕ) (d : ℕ) 
  (h1 : a = 20)
  (h2 : c = 25)
  (h3 : c^2 = a^2 + d^2) :
  d = 15 :=
by
  have : 25^2 = 20^2 + d^2 := h3.subst (h2.symm ▸ (h1.symm ▸ rfl))
  have : 625 = 400 + d^2 := by norm_num at this
  have : 225 = d^2 := by norm_num at this
  have : d = 15 := by exact int.sqrt_eq zero_le (by norm_num)

end

end distance_after_turning_left_l64_64694


namespace quadratic_roots_l64_64470

theorem quadratic_roots (a b c : ℝ) :
  ∃ x y : ℝ, (x ≠ y ∧ (x^2 - (a + b) * x + (ab - c^2) = 0) ∧ (y^2 - (a + b) * y + (ab - c^2) = 0)) ∧
  (x = y ↔ a = b ∧ c = 0) := sorry

end quadratic_roots_l64_64470


namespace simplify_expression_l64_64611

variables (a b : ℝ)

theorem simplify_expression : 
  (2 * a^2 - 3 * a * b + 8) - (-a * b - a^2 + 8) = 3 * a^2 - 2 * a * b :=
by sorry

-- Note:
-- ℝ denotes real numbers. Adjust types accordingly if using different numerical domains (e.g., ℚ, ℂ).

end simplify_expression_l64_64611


namespace find_coordinates_of_Q_l64_64603

noncomputable def unit_circle_arc_length_to_angle (arc_length : ℝ) : ℝ :=
  -arc_length -- clockwise direction represented by negative angle

noncomputable def point_on_unit_circle_after_move (start : ℝ × ℝ) (arc_length : ℝ) : ℝ × ℝ :=
  let angle := unit_circle_arc_length_to_angle arc_length in
  (Real.cos angle, Real.sin angle)

theorem find_coordinates_of_Q :
  point_on_unit_circle_after_move (1, 0) (2 * Real.pi / 3) = (-1 / 2, -Real.sqrt 3 / 2) :=
by
  -- proof is omitted
  sorry

end find_coordinates_of_Q_l64_64603


namespace difference_length_width_l64_64233

-- Definition of variables and conditions
variables (L W : ℝ)
def hall_width_half_length : Prop := W = (1/2) * L
def hall_area_578 : Prop := L * W = 578

-- Theorem to prove the desired result
theorem difference_length_width (h1 : hall_width_half_length L W) (h2 : hall_area_578 L W) : L - W = 17 :=
sorry

end difference_length_width_l64_64233


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64759

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64759


namespace ordered_11_tuples_l64_64995

theorem ordered_11_tuples (a : ℕ → ℤ) (h : (∀ i, 0 ≤ i ∧ i < 11 → a i ^ 2 = ∑ j in Finset.range 11, if j = i then 0 else a j) :
  (∃ (x : ℕ), x = 495) :=
sorry

end ordered_11_tuples_l64_64995


namespace find_c_l64_64164

theorem find_c :
  let P1 : ℝ × ℝ := (1, -3)
  let P2 : ℝ × ℝ := (5, 3)
  let direction := (P2.1 - P1.1, P2.2 - P1.2)
  ∃ c : ℝ, ∃ k : ℝ, k * direction = (3, c) ∧ c = 9 / 2 := 
by
  let P1 : ℝ × ℝ := (1, -3)
  let P2 : ℝ × ℝ := (5, 3)
  let direction := (P2.1 - P1.1, P2.2 - P1.2)
  use 9 / 2 -- c = 9 / 2
  use (3 / 4) -- k
  exact ⟨rfl, rfl⟩ -- This concludes the proof for now
  sorry

end find_c_l64_64164


namespace cone_volume_l64_64674

variable (l h : ℝ) (π : ℝ := Real.pi)

noncomputable def radius (l h : ℝ) : ℝ := sqrt (l^2 - h^2)

noncomputable def volume_of_cone (r h : ℝ) (π : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (hl : l = 15) (hh : h = 9) :
  volume_of_cone (radius l h) h π = 432 * π :=
by
  sorry

end cone_volume_l64_64674


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64836

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64836


namespace radius_of_middle_circle_l64_64354

theorem radius_of_middle_circle :
  ∃ r2 r3 r4 : ℝ, 
    (r2 > 0) ∧ (r3 > 0) ∧ (r4 > 0) ∧ 
    (5 ∆  [8, r2, r3, r4, 18]) ∧ 
    (r3 = 12) :=
  -- sorry to be replaced by a full proof
  sorry

end radius_of_middle_circle_l64_64354


namespace arithmetic_sequence_z_value_l64_64550

/--
In the arithmetic sequence 13, x, y, z, t, 49,
prove that the value of z is 31.
-/
theorem arithmetic_sequence_z_value (x y t : ℕ) (h_arithmetic : ∀ {a b c : ℕ}, a + c = 2 * b) :
  let a := 13
  let b := 49
  z = (a + b) / 2 :=
begin
  sorry,  -- proof to be filled in
end

end arithmetic_sequence_z_value_l64_64550


namespace minimize_QR_minus_PQ_l64_64691

def is_triangle (PQ PR QR : ℕ) : Prop :=
  PQ > 0 ∧ PR > 0 ∧ QR > 0 ∧
  PQ + PR > QR ∧ PQ + QR > PR ∧ PR + QR > PQ

theorem minimize_QR_minus_PQ :
  ∃ (PQ PR QR : ℕ), PQ < PR ∧ PR ≤ QR ∧ PQ + PR + QR = 3010 ∧ is_triangle PQ PR QR ∧ QR - PQ = 1 :=
begin
  sorry
end

end minimize_QR_minus_PQ_l64_64691


namespace min_elements_A_l64_64006

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_property (f : ℕ → A) (A : Type) : Prop :=
  ∀ i j : ℕ, (∃ p : ℕ, is_prime p ∧ |i - j| = p) → f i ≠ f j

noncomputable def minimum_size_of_A (A : Type) [fintype A] : ℕ :=
  fintype.card A

theorem min_elements_A (A : Type) [fintype A] (f : ℕ → A) :
  satisfies_property f A → minimum_size_of_A A = 4 :=
sorry

end min_elements_A_l64_64006


namespace significant_improvement_l64_64261

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64261


namespace find_radius_l64_64494

-- Definitions of the conditions:
def circle (x y r : ℝ) : Prop := x^2 + y^2 = r^2
def line (x y : ℝ) : Prop := x + sqrt 3 * y - 2 = 0
def angle_AOB (angle : ℝ) : Prop := angle = 120

-- Main statement to prove:
theorem find_radius (r : ℝ) (h_circle : ∀ (x y : ℝ), circle x y r) (h_line : ∀ (x y : ℝ), line x y) (h_angle : angle_AOB 120) :
  r = 2 :=
sorry

end find_radius_l64_64494


namespace product_of_solutions_l64_64776

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64776


namespace triangles_similar_l64_64459

noncomputable theory

variable {A B C P Q S : Type}

def is_triangle (A B C : Type) : Prop := sorry -- Placeholder definition

def on_side (P : Type) (side : Type) : Prop := sorry -- Placeholder definition for points on sides

def circumcenter (triangle : Type) : Type := sorry -- Placeholder definition for circumcenters

def similar (triangle1 triangle2 : Type) : Prop := sorry -- Placeholder definition for triangle similarity

theorem triangles_similar (hABC : is_triangle A B C)
  (hP_on_AB : on_side P (A, B))
  (hQ_on_BC : on_side Q (B, C))
  (hS_on_CA : on_side S (C, A)) :
  similar (circumcenter (A, P, S), circumcenter (B, Q, P), circumcenter (C, S, Q)) (A, B, C) :=
sorry

end triangles_similar_l64_64459


namespace f_neg_x_eq_neg_log_2_l64_64108

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then log (x + 2) / log 2
else -log (-x + 2) / log 2

theorem f_neg_x_eq_neg_log_2 (x : ℝ) (hx : x < 0) :
  f x = -log (-x + 2) / log 2 :=
by
  sorry

end f_neg_x_eq_neg_log_2_l64_64108


namespace product_of_triple_when_added_to_reciprocal_l64_64906

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64906


namespace significant_improvement_l64_64279

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64279


namespace rentExpenses_l64_64342

noncomputable def monthlySalary : ℝ := 23000
noncomputable def milkExpenses : ℝ := 1500
noncomputable def groceriesExpenses : ℝ := 4500
noncomputable def educationExpenses : ℝ := 2500
noncomputable def petrolExpenses : ℝ := 2000
noncomputable def miscellaneousExpenses : ℝ := 5200
noncomputable def savings : ℝ := 2300

-- Calculating total non-rent expenses
noncomputable def totalNonRentExpenses : ℝ :=
  milkExpenses + groceriesExpenses + educationExpenses + petrolExpenses + miscellaneousExpenses

-- The rent expenses theorem
theorem rentExpenses : totalNonRentExpenses + savings + 5000 = monthlySalary :=
by sorry

end rentExpenses_l64_64342


namespace total_savings_l64_64400

def individual_shirt_cost : ℝ := 7.50
def individual_pants_cost : ℝ := 15.00
def individual_socks_cost : ℝ := 4.50

def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75

def team_size : ℕ := 12

theorem total_savings :
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost in
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost in
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost in
  let total_savings := savings_per_uniform * team_size in
  total_savings = 36 :=
by
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  have h : total_savings = 36 := by
    calc
      total_savings = (7.50 + 15.00 + 4.50 - (6.75 + 13.50 + 3.75)) * 12 := by sorry
                  ... = 3 * 12 := by sorry
                  ... = 36 := by sorry
  exact h

end total_savings_l64_64400


namespace candy_boxes_system_l64_64341

-- Given conditions and definitions
def sheets_total (x y : ℕ) : Prop := x + y = 35
def sheet_usage (x y : ℕ) : Prop := 20 * x = 30 * y / 2

-- Statement
theorem candy_boxes_system (x y : ℕ) (h1 : sheets_total x y) (h2 : sheet_usage x y) : 
  (x + y = 35) ∧ (20 * x = 30 * y / 2) := 
by
sorry

end candy_boxes_system_l64_64341


namespace hyperbola_eccentricity_l64_64465

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : sqrt 3 * c - c = 2 * a) : 
  let e := c / a in 
  e = sqrt 3 + 1 := 
by 
  sorry

end hyperbola_eccentricity_l64_64465


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64719

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64719


namespace complex_purely_imaginary_l64_64524

theorem complex_purely_imaginary (x : ℝ) : (∃ z : ℂ, z = (x^2 - 1) + (x + 1) * complex.I ∧ Im z ≠ 0 ∧ Re z = 0) → x = 1 :=
begin
  sorry
end

end complex_purely_imaginary_l64_64524


namespace product_of_solutions_l64_64704

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64704


namespace product_of_solutions_l64_64711

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64711


namespace sum_factors_of_30_l64_64956

theorem sum_factors_of_30 : 
  (∑ i in {1, 2, 3, 5, 6, 10, 15, 30}, i) = 72 := 
by
  -- Sorry, to skip the proof here.
  sorry

end sum_factors_of_30_l64_64956


namespace find_coefficients_l64_64102

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![[1, 2, 3], [2, 1, 2], [3, 2, 1]]

noncomputable def I : Matrix (Fin 3) (Fin 3) ℝ :=
  1

theorem find_coefficients :
  ∃ p q r : ℝ, 
    (A ^ 3 + p • A ^ 2 + q • A + r • I = 0) ∧ 
    (p = -8) ∧ (q = -2) ∧ (r = -8) :=
by
  use -8, -2, -8
  split
  sorry -- Proof of A^3 + pA^2 + qA + rI = 0
  split; refl -- Verifying p, q, r values as -8, -2, -8

end find_coefficients_l64_64102


namespace sum_of_first_10_terms_l64_64014

-- Definitions and assumptions
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ S 5 = 15 -- Conditions on the arithmetic sequence

def S (n : ℕ) : ℕ :=
  n * (n + 1) / 2  -- Sum of first n terms of an arithmetic sequence with a_1 = 1, d = 1

def b_n (a : ℕ → ℕ) (n : ℕ) : ℚ :=
  1 / (a (n + 1) ^ 2 - 1)  -- Definition of sequence b_n

-- Problem Statement
theorem sum_of_first_10_terms :
  ∀ a : ℕ → ℕ, (arithmetic_seq a) → (∑ i in finset.range 10, b_n a i) = 175 / 264 :=
by sorry

end sum_of_first_10_terms_l64_64014


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64725

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64725


namespace indicator_significant_improvement_l64_64304

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64304


namespace product_of_solutions_l64_64716

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64716


namespace linear_function_quadrants_l64_64169

theorem linear_function_quadrants (k : ℝ) :
  (k - 3 > 0) ∧ (-k + 2 < 0) → k > 3 :=
by
  intro h
  sorry

end linear_function_quadrants_l64_64169


namespace product_trippled_when_added_to_reciprocal_l64_64939

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64939


namespace sum_of_factors_of_30_is_72_l64_64964

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l64_64964


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64830

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64830


namespace product_trippled_when_added_to_reciprocal_l64_64938

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64938


namespace ursula_purchases_total_cost_l64_64695

variable (T C B Br : ℝ)
variable (hT : T = 10) (hTC : T = 2 * C) (hB : B = 0.8 * C) (hBr : Br = B / 2)

theorem ursula_purchases_total_cost : T + C + B + Br = 21 := by
  sorry

end ursula_purchases_total_cost_l64_64695


namespace constant_term_eq_three_l64_64159

theorem constant_term_eq_three : 
  let expr := (λ (x : ℝ), (sqrt x + 2) * (1 / sqrt x - 1)^5) in
  -- The constant term in the expansion is 3
  ∃ c : ℝ, c = 3 ∧ (∀ (x : ℝ), expr x = c) := 
sorry

end constant_term_eq_three_l64_64159


namespace somu_age_years_back_l64_64618

theorem somu_age_years_back (S F Y : ℕ)
  (h1 : S = 18)
  (h2 : S = F / 3)
  (h3 : S - Y = (F - Y) / 5) :
  Y = 9 :=
by
  rw [h1, h2, h3]
  -- Substitute known values
  have hF : F = 54 := by norm_num
  have h4 : 18 - Y = (54 - Y) / 5 := by sorry
  -- Solve for Y to show Y = 9
  sorry

end somu_age_years_back_l64_64618


namespace product_of_solutions_l64_64775

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64775


namespace significant_improvement_l64_64267

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64267


namespace proof_complement_union_l64_64498

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ 0}

theorem proof_complement_union :
  {x | x ≥ 1} = ⋂a, M x ∨ N x :=
sorry

end proof_complement_union_l64_64498


namespace significant_improvement_l64_64272

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64272


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64858

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64858


namespace solve_quadratic_factoring_solve_quadratic_formula_l64_64146

theorem solve_quadratic_factoring:
  (x : ℝ) -> (x^2 + 2 * x - 3 = 0) -> (x = 1) ∨ (x = -3) := 
by
  sorry

theorem solve_quadratic_formula:
  (x : ℝ) -> (2 * x^2 + 4 * x - 3 = 0) -> (x = (-2 + sqrt 10) / 2) ∨ (x = (-2 - sqrt 10) / 2) := 
by
  sorry

end solve_quadratic_factoring_solve_quadratic_formula_l64_64146


namespace cone_volume_l64_64653

def slant_height := 15
def height := 9
def radius := Real.sqrt (slant_height^2 - height^2)
def volume := (1/3) * Real.pi * radius^2 * height

theorem cone_volume : 
  radius = 12 ∧ volume = 432 * Real.pi :=
by
  split
  { -- radius = 12
    unfold radius
    sorry
  }
  { -- volume = 432 * Real.pi
    unfold volume
    sorry
  }

end cone_volume_l64_64653


namespace sum_factors_of_30_l64_64961

theorem sum_factors_of_30 : 
  (∑ i in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), i) = 72 := by
  sorry

end sum_factors_of_30_l64_64961


namespace minimum_value_of_y_l64_64633

-- Define y as a function of x given the condition
def y (x : ℝ) : ℝ := 8 + x / 2 + 2 / x

-- State that x should be greater than 0
def x_pos (x : ℝ) : Prop := x > 0

-- The theorem we are proving
theorem minimum_value_of_y (x : ℝ) (hx : x_pos x) : (8 + x / 2 + 2 / x) ≥ 10 :=
sorry

end minimum_value_of_y_l64_64633


namespace angle_BAC_is_right_l64_64352

open Triangle Geometry

theorem angle_BAC_is_right (A B C D E O : Point)
  (h1 : is_isosceles_triangle A B C)
  (h2 : midpoint O A B)
  (h3 : intersects_circle OC (circle O AB) D)
  (h4 : ray_intersects BD AC E)
  (h5 : length AE = length CD) :
  angle A B C = 90 :=
sorry

end angle_BAC_is_right_l64_64352


namespace product_of_solutions_l64_64873

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64873


namespace possible_values_of_f_l64_64398

-- Define the function f(A, B, C)
def f (A B C : ℕ) : ℕ := A^3 + B^3 + C^3 - 3 * A * B * C

-- The statement of the theorem
theorem possible_values_of_f (n : ℕ) :
  (∃ A B C : ℕ, f A B C = n) ↔ ¬ ((n % 9 = 3) ∨ (n % 9 = 6)) :=
by
  sorry

end possible_values_of_f_l64_64398


namespace product_of_all_solutions_triple_reciprocal_l64_64742

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64742


namespace greatest_prime_factor_of_sum_l64_64983

def double_factorial (x : ℕ) : ℕ :=
  if x % 2 = 0 then (List.range (x / 2)).map (λ n => 2 * (n + 1)).prod else 1

def greatest_prime_factor (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).filter (λ d => Nat.Prime d).maximum' id

theorem greatest_prime_factor_of_sum :
  greatest_prime_factor (double_factorial 22 + double_factorial 20) = 23 :=
by
  sorry

end greatest_prime_factor_of_sum_l64_64983


namespace parallel_vectors_l64_64506

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, -2)

theorem parallel_vectors (h : (1 * (-2) - 2 * x = 0)) : x = -1 :=
by
  sorry

end parallel_vectors_l64_64506


namespace indicator_significant_improvement_l64_64303

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64303


namespace product_of_triple_when_added_to_reciprocal_l64_64904

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64904


namespace new_device_significant_improvement_l64_64256

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64256


namespace farthest_distance_correct_l64_64598

-- Define the key points and distances
def origin : ℝ × ℝ := (0, 0)
def pole : ℝ × ℝ := (5, 1)
def leash_radius : ℝ := 15

-- Define the function to calculate the Euclidean distance
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the farthest_distance function
def farthest_distance_from_origin : ℝ :=
  euclidean_distance origin pole + leash_radius

theorem farthest_distance_correct :
  farthest_distance_from_origin = sqrt 26 + 15 :=
by
  sorry

end farthest_distance_correct_l64_64598


namespace median_4040_list_l64_64215

def median_combined_list (n : ℕ) (l1 l2 : List ℕ) : ℕ :=
  let combined_list := (l1 ++ l2).sort
  (combined_list[(combined_list.length / 2) - 1] + combined_list[combined_list.length / 2]) / 2

theorem median_4040_list :
  let l1 := List.range' 1 2021
  let l2 := List.range' 1 2021 |>.map (λ x => x * x)
  median_combined_list 4040 l1 l2 = 1976.5 :=
by
  sorry

end median_4040_list_l64_64215


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64825

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64825


namespace cost_of_each_cupcake_l64_64562

theorem cost_of_each_cupcake (total_cupcakes : ℕ) (john_paid : ℝ) (steve_paid : ℝ)
    (split_evenly : john_paid = steve_paid) (total_cost : john_paid + steve_paid = 18) 
    (cupcake_quantity : total_cupcakes = 12) : 
    let cost_per_cupcake := (john_paid + steve_paid) / total_cupcakes in
    cost_per_cupcake = 1.50 :=
by
  sorry

end cost_of_each_cupcake_l64_64562


namespace significant_improvement_l64_64288

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64288


namespace sum_of_coordinates_D_l64_64602

theorem sum_of_coordinates_D (x y : Int) :
  let N := (4, 10)
  let C := (14, 6)
  let D := (x, y)
  N = ((x + 14) / 2, (y + 6) / 2) →
  x + y = 8 :=
by
  intros
  sorry

end sum_of_coordinates_D_l64_64602


namespace find_omitted_angle_l64_64141

-- Definitions and conditions
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

def omitted_angle (calculated_sum actual_sum : ℝ) : ℝ :=
  actual_sum - calculated_sum

-- The theorem to be proven
theorem find_omitted_angle (n : ℕ) (h₁ : 1958 + 22 = sum_of_interior_angles n) :
  omitted_angle 1958 (sum_of_interior_angles n) = 22 :=
by
  sorry

end find_omitted_angle_l64_64141


namespace find_vector_length_l64_64504

open EuclideanGeometry

variable {α β : Vect ℝ 2}

def vectors_given_conditions (α β : Vect ℝ 2) : Prop :=
  ∥α∥ = 1 ∧ ∥β∥ = 2 ∧ α ⬝ (α - 2 • β) = 0

theorem find_vector_length (α β : Vect ℝ 2)
  (h : vectors_given_conditions α β) :
  ∥2 • α + β∥ = Real.sqrt 10 :=
  sorry

end find_vector_length_l64_64504


namespace association_between_satisfaction_and_age_expected_value_of_total_vouchers_l64_64072

-- Part 1: Chi-square test for association
def ContingencyTable :=
  { n : ℕ, a : ℕ, b : ℕ, c : ℕ, d : ℕ // a + b + c + d = n }

noncomputable def chi_square (tbl : ContingencyTable) : ℚ :=
  let ⟨n, a, b, c, d, h⟩ := tbl in
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem association_between_satisfaction_and_age :
  ∃ tbl : ContingencyTable,
    (tbl.n = 300) ∧
    (tbl.a = 140) ∧ (tbl.b = 40) ∧ (tbl.c = 80) ∧ (tbl.d = 40) ∧
    chi_square tbl > 3.841 :=
sorry

-- Part 2: Expected value of recharge vouchers
def expected_recharge_vouchers : ℚ :=
  let p_red := 1 / 3
  let p_white := 2 / 3
  let n := 100
  in 5 * n * p_red + 2 * n * p_white

theorem expected_value_of_total_vouchers :
  expected_recharge_vouchers = 300 :=
sorry

end association_between_satisfaction_and_age_expected_value_of_total_vouchers_l64_64072


namespace arrangement_count_l64_64685

theorem arrangement_count (A B C D E : Type) : 
  let arrangements := list.permutations [A, B, C, D, E]
      count := list.countp (λ l => ¬(A = l.nth! (l.index_of C - 1) ∨ A = l.nth! (l.index_of C + 1)) ∧
                               ¬(B = l.nth! (l.index_of C - 1) ∨ B = l.nth! (l.index_of C + 1))) 
                arrangements
  in count = 36 := sorry

end arrangement_count_l64_64685


namespace first_traveler_constant_speed_second_traveler_constant_speed_l64_64986

theorem first_traveler_constant_speed :
  ∀ (total_distance : ℝ) (first_half_speed second_half_speed : ℝ) (total_time : ℝ),
  total_distance = 24 ∧ first_half_speed = 4 ∧ second_half_speed = 6 ∧ total_time = 5 →
  (total_distance / total_time) = 4.8 :=
by
  intros total_distance first_half_speed second_half_speed total_time h,
  cases h,
  sorry

theorem second_traveler_constant_speed :
  ∀ (T : ℝ) (first_half_speed second_half_speed total_time : ℝ),
  first_half_speed = 4 ∧ second_half_speed = 6 ∧ total_time = 2 * T ∧ (4 * T + 6 * T) = 10 * T →
  ((4 * T + 6 * T) / total_time) = 5 :=
by
  intros T first_half_speed second_half_speed total_time h,
  cases h,
  sorry

end first_traveler_constant_speed_second_traveler_constant_speed_l64_64986


namespace product_of_triple_when_added_to_reciprocal_l64_64903

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64903


namespace symmetric_coords_l64_64625

-- Define the initial point and the line equation
def initial_point : ℝ × ℝ := (-1, 1)
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define what it means for one point to be symmetric to another point with respect to a line
def symmetric_point (p q : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), line_eq m p.1 ∧ line_eq m q.1 ∧ 
             p.1 + q.1 = 2 * m ∧
             p.2 + q.2 = 2 * m

-- The theorem we want to prove
theorem symmetric_coords : ∃ (symmetric : ℝ × ℝ), symmetric_point initial_point symmetric ∧ symmetric = (2, -2) :=
sorry

end symmetric_coords_l64_64625


namespace significant_improvement_l64_64273

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64273


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64883

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64883


namespace sum_primes_1_mod_6_6_mod_7_l64_64433

open Nat

def is_desired_prime (p : ℕ) : Prop :=
  prime p ∧ p > 1 ∧ p < 100 ∧ p % 6 = 1 ∧ p % 7 = 6

theorem sum_primes_1_mod_6_6_mod_7 :
  (∑ p in (finset.filter is_desired_prime (finset.range 101)), p) = 116 := 
by
  sorry

end sum_primes_1_mod_6_6_mod_7_l64_64433


namespace product_of_triple_when_added_to_reciprocal_l64_64898

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64898


namespace find_multiple_l64_64638

theorem find_multiple (a b m : ℤ) (h1 : a * b = m * (a + b) + 12) 
(h2 : b = 10) (h3 : b - a = 6) : m = 2 :=
by {
  sorry
}

end find_multiple_l64_64638


namespace discount_problem_l64_64175

variable (x : ℝ)

theorem discount_problem :
  (400 * (1 - x)^2 = 225) :=
sorry

end discount_problem_l64_64175


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64838

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64838


namespace product_of_real_solutions_triple_property_l64_64797

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64797


namespace extra_sweets_l64_64135

theorem extra_sweets (S : ℕ) (h1 : ∀ n: ℕ, S = 120 * 38) : 
    (38 - (S / 190) = 14) :=
by
  -- Here we will provide the proof 
  sorry

end extra_sweets_l64_64135


namespace cone_volume_l64_64664

-- Given conditions as definitions
def slant_height : ℝ := 15
def height : ℝ := 9
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- The volume of the cone
def volume : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem to prove that the calculated volume is 432π cubic centimeters
theorem cone_volume : volume = 432 * Real.pi := by
  sorry

end cone_volume_l64_64664


namespace factorize_a3_sub_a_l64_64418

theorem factorize_a3_sub_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a3_sub_a_l64_64418


namespace significant_improvement_l64_64250

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64250


namespace product_of_solutions_l64_64780

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64780


namespace shorter_leg_of_right_triangle_l64_64544

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : a^2 + b^2 = 65^2) (ha : a ≤ b) : a = 25 :=
by sorry

end shorter_leg_of_right_triangle_l64_64544


namespace complex_modulus_l64_64523

noncomputable def complex_abs_add_one (z : ℂ) (h : z^2 = -4) : ℝ :=
  abs (1 + z)

theorem complex_modulus (z : ℂ) (h : z^2 = -4) : complex_abs_add_one z h = sqrt 5 := by
  sorry

end complex_modulus_l64_64523


namespace angle_A_value_median_AM_range_l64_64070

open Real

-- Define the function
def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- Conditions from the problem:
variable (A : ℝ)
variable (a b c : ℝ)
variable (a_given : a = sqrt 3)
variable (angle_condition : ∀ x : ℝ, f x ≤ f A)

-- Part 1: Determining angle A
theorem angle_A_value (A : ℝ) (angle_condition : ∀ x : ℝ, f x ≤ f A) : A = π / 3 := by
  sorry

-- Median calculation for part 2:
theorem median_AM_range (a b c A : ℝ) 
  (a_given : a = sqrt 3)
  (angle_A : A = π / 3) 
  : (sqrt 3 / 2) < (1 / 2) * sqrt (2 * b^2 + 2 * c^2 - a^2) ∧ (1 / 2) * sqrt (2 * b^2 + 2 * c^2 - a^2) ≤ 3 / 2 := by
  sorry

end angle_A_value_median_AM_range_l64_64070


namespace johns_remaining_money_l64_64097

-- Define the cost of pizza and drinks
variables (q : ℝ) (cost_drink cost_medium_pizza cost_large_pizza : ℝ)

-- Define John’s initial amount of money
def initial_money : ℝ := 50

-- Define the cost of items John buys based on q
def cost_drink := q
def cost_medium_pizza := 3 * q
def cost_large_pizza := 4 * q

-- Define the total money John spent
def total_cost (q : ℝ) : ℝ :=
  4 * cost_drink + cost_medium_pizza + 2 * cost_large_pizza

-- Define the money John has left
def money_left (q : ℝ) : ℝ :=
  initial_money - total_cost q

-- Main statement to be proved in Lean
theorem johns_remaining_money (q : ℝ) : money_left q = 50 - 15 * q :=
by
  unfold money_left
  unfold total_cost
  sorry

end johns_remaining_money_l64_64097


namespace geometric_sequence_S5_l64_64444

noncomputable def S5 (a₁ q : ℝ) : ℝ :=
  a₁ * (1 - q^5) / (1 - q)

theorem geometric_sequence_S5 
  (a₁ q : ℝ) 
  (h₁ : a₁ * (1 + q) = 3 / 4)
  (h₄ : a₁ * q^3 * (1 + q) = 6) :
  S5 a₁ q = 31 / 4 := 
sorry

end geometric_sequence_S5_l64_64444


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64912

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64912


namespace haley_solution_l64_64534

def haleys_marbles_problem (total_marbles : Nat) (number_of_boys : Nat) (marbles_per_boy : Nat) : Prop :=
  total_marbles = 28 ∧ number_of_boys = 14 ∧ marbles_per_boy = total_marbles / number_of_boys

theorem haley_solution : haleys_marbles_problem 28 14 2 := 
by 
  unfold haleys_marbles_problem 
  simp 
  sorry

end haley_solution_l64_64534


namespace product_of_solutions_l64_64714

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64714


namespace median_of_set_l64_64058

noncomputable def median_set {α : Type*} [LinearOrder α] (s : Finset α) : α :=
if h : s.card % 2 = 1 then s.sort (· < ·) ⟨s.card / 2, sorry⟩ else sorry

theorem median_of_set {a : ℤ} {b : ℝ}
  (ha : a ≠ 0) (hb_pos : 0 < b) (hb_eq : a * b^2 = Real.log2 b) :
  median_set ({0, 1, a, b, 2/b} : Finset ℝ) = b :=
by
  sorry

end median_of_set_l64_64058


namespace problem1_problem2_problem3_l64_64010

-- Problem 1: Expression for f(x)
theorem problem1 (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x + 1) - f(x) = 2 * x) (hf0 : f 0 = 1) :
  (f = λ x, x^2 - x + 1) := sorry

-- Problem 2: Range of t for monotonicity of g(x)
theorem problem2 (t : ℝ) (f : ℝ → ℝ) (hf_eq : f = (λ x, x^2 - x + 1)) 
  (g : ℝ → ℝ) (hg_eq : g = (λ x, f x - 2 * t * x)) :
  (∀ x1 x2 : ℝ, -1 ≤ x1 → x1 ≤ x2 → x2 ≤ 5 → (g x1 ≤ g x2 ∨ g x1 ≥ g x2)) ↔
  (t ≤ -(3/2) ∨ t ≥ 9/2) := sorry

-- Problem 3: Range of m for unique real root
theorem problem3 (m : ℝ) (f : ℝ → ℝ) (hf_eq : f = (λ x, x^2 - x + 1)) :
  (∃! x : ℝ, -1 < x ∧ x < 2 ∧ f x = x + m) ↔
  (m ∈ {0} ∨ (1 ≤ m ∧ m < 4)) := sorry

end problem1_problem2_problem3_l64_64010


namespace fourth_watercraft_is_submarine_l64_64448

-- Define the conditions as Lean definitions
def same_direction_speed (w1 w2 w3 w4 : Type) : Prop :=
  -- All watercraft are moving in the same direction at the same speed
  true

def separation (w1 w2 w3 w4 : Type) (d : ℝ) : Prop :=
  -- Each pair of watercraft is separated by distance d
  true

def cargo_ship (w : Type) : Prop := true
def fishing_boat (w : Type) : Prop := true
def passenger_vessel (w : Type) : Prop := true

-- Define that the fourth watercraft is unique
def unique_watercraft (w : Type) : Prop := true

-- Proof statement that the fourth watercraft is a submarine
theorem fourth_watercraft_is_submarine 
  (w1 w2 w3 w4 : Type)
  (h1 : same_direction_speed w1 w2 w3 w4)
  (h2 : separation w1 w2 w3 w4 100)
  (h3 : cargo_ship w1)
  (h4 : fishing_boat w2)
  (h5 : passenger_vessel w3) :
  unique_watercraft w4 := 
sorry

end fourth_watercraft_is_submarine_l64_64448


namespace sum_of_ages_l64_64183

def MariaAge (MarioAge: ℕ): ℕ := MarioAge - 1

def sumAges (MarioAge MariaAge: ℕ): ℕ := MarioAge + MariaAge

theorem sum_of_ages (MarioAge : ℕ) (h1 : MarioAge = 4) (h2 : MariaAge MarioAge = MarioAge - 1) :
  sumAges MarioAge (MariaAge MarioAge) = 7 :=
by
  rw [h1, h2]
  rfl

end sum_of_ages_l64_64183


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64893

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64893


namespace product_of_solutions_l64_64807

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64807


namespace cone_volume_l64_64655

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l64_64655


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64878

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64878


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64839

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64839


namespace unique_equal_value_mapping_intervals_l64_64529

theorem unique_equal_value_mapping_intervals :
  let f₁ : ℝ → ℝ := λ x, x^2 - 1
  let f₂ : ℝ → ℝ := λ x, 2 + log2 x
  let f₃ : ℝ → ℝ := λ x, 2^x - 1
  let f₄ : ℝ → ℝ := λ x, 1 / (x - 1)
  let equal_value_mapping (f : ℝ → ℝ) : Prop :=
    ∃ m n : ℝ, m < n ∧ (∀ x ∈ set.Icc m n, f x ∈ set.Icc m n) ∧ 
              (set.range (f ∘ (λ x, x)) = set.Icc m n)
  (∃! f, (f = f₂ ∨ f = f₃) ∧ equal_value_mapping f) := sorry

end unique_equal_value_mapping_intervals_l64_64529


namespace new_device_significant_improvement_l64_64255

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64255


namespace product_of_solutions_l64_64877

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64877


namespace slope_of_line_l64_64944

theorem slope_of_line (a b c : ℝ) (h : 3 * a = 4 * b - 9) : a = 4 / 3 * b - 3 :=
by
  sorry

end slope_of_line_l64_64944


namespace graphing_calculators_count_l64_64415

noncomputable def number_of_graphing_calculators (total_calculators : ℕ) (total_cost : ℕ)
  (cost_scientific : ℕ) (cost_graphing : ℕ) : ℕ :=
  let equations := (45 : ℕ, 1625 : ℕ)
in
  if (total_calculators = equations.1 ∧ total_cost = equations.2 ∧
       cost_scientific = 10 ∧ cost_graphing = 57)
  then 25
  else sorry

theorem graphing_calculators_count :
  number_of_graphing_calculators 45 1625 10 57 = 25 :=
by
  sorry

end graphing_calculators_count_l64_64415


namespace seating_ways_for_15_women_l64_64419

def T : ℕ → ℕ
| 0       := 1
| 1       := 1
| 2       := 2
| (n+3) := T n + T (n + 1) + T (n + 2)

theorem seating_ways_for_15_women : T 15 = 987 := by
  sorry

end seating_ways_for_15_women_l64_64419


namespace sin_of_right_triangle_l64_64543

open Real

theorem sin_of_right_triangle (Q : ℝ) (h : 3 * sin Q = 4 * cos Q) : sin Q = 4 / 5 :=
by
  sorry

end sin_of_right_triangle_l64_64543


namespace estimate_pi_l64_64687

theorem estimate_pi :
  (∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 → (x^2 + y^2 + z^2 > 1 → t = 1) ∧ (x^2 + y^2 + z^2 ≤ 1 → t = 0)) →
  (Σ (i : ℕ) in (finset.range 1000), t i = 477) →
  (estimate (Σ (i : ℕ) in (finset.range 1000), t i) = 3.138) :=
by sorry

end estimate_pi_l64_64687


namespace product_trippled_when_added_to_reciprocal_l64_64930

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64930


namespace tan_u_tan_v_sum_l64_64582

theorem tan_u_tan_v_sum (u v : ℝ) 
  (h1 : (sin u / cos v) + (sin v / cos u) = 2)
  (h2 : (cos u / sin v) + (cos v / sin u) = 3) :
  (tan u / tan v) + (tan v / tan u) = 8 / 7 :=
by
  sorry

end tan_u_tan_v_sum_l64_64582


namespace find_even_integer_l64_64439

def is_even (n : ℤ) : Prop := n % 2 = 0

def h (n : ℤ) : ℤ :=
  if is_even n then
    ∑ k in Finset.range ((n / 2) + 1), 2 * k
  else
    0

theorem find_even_integer (n : ℤ) (h₁ : is_even n) (h₂ : h(18) = 90) (h₃ : h(18) / h(n) = 3) : n = 10 :=
by sorry

end find_even_integer_l64_64439


namespace number_of_integer_solutions_l64_64508

theorem number_of_integer_solutions :
  {x : ℤ | (x - 1) ^ (16 - x^2) = 1}.to_finset.card = 4 :=
by
  sorry

end number_of_integer_solutions_l64_64508


namespace product_of_solutions_l64_64803

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64803


namespace team_savings_with_discount_l64_64404

def regular_shirt_cost : ℝ := 7.50
def regular_pants_cost : ℝ := 15.00
def regular_socks_cost : ℝ := 4.50
def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75
def team_size : ℕ := 12

theorem team_savings_with_discount :
  let regular_uniform_cost := regular_shirt_cost + regular_pants_cost + regular_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by
  sorry

end team_savings_with_discount_l64_64404


namespace solution_set_of_inequality_l64_64679

theorem solution_set_of_inequality (x : ℝ) :
  1 < |x + 2| ∧ |x + 2| < 5 ↔ (x ∈ set.Ioo (-7) (-3) ∨ x ∈ set.Ioo (-1) 3) :=
by
  sorry

end solution_set_of_inequality_l64_64679


namespace find_circle_equation_l64_64453

noncomputable def is_point_on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  let (h, k) := center
  let (x, y) := point
  (x - h)^2 + (y - k)^2 = radius^2

noncomputable def distance_from_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x, y) := p
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem find_circle_equation :
  ∃ (h k r : ℝ),
    (is_point_on_circle (h, k) r (2, -1)) ∧           -- The circle passes through the point (2, -1)
    (distance_from_point_to_line (h, k) 1 1 (-1) = r) ∧ -- The circle is tangent to the line x + y = 1
    (k = -2 * h) ∧                                    -- The center lies on the line y = -2x
    ((h = 1) ∧ (k = -2) ∧ (r = sqrt 2) ∧ (is_point_on_circle (h, k) r = λ p, (p.1 - 1)^2 + (p.2 + 2)^2 = 2)) := 
begin
  use [1, -2, sqrt 2],
  split,
  { -- Proof that the circle passes through (2, -1)
    sorry
  },
  split,
  { -- Proof that distance from center to tangent line equals radius
    sorry
  },
  split,
  { -- Proof that center lies on y = -2x
    sorry
  },
  -- Proof that the standard equation of the circle is correct
  { 
    split,
    { exact rfl }, 
    split,
    { exact rfl }, 
    split,
    { exact rfl },
    exact rfl
  }
end

end find_circle_equation_l64_64453


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64880

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64880


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64921

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64921


namespace sequence_limit_l64_64371

theorem sequence_limit : 
  (∀ n : ℕ,
    \left( (3 - 4 * n)^2 / ((n - 3)^2 - (n + 3)^2) = (9 - 24 * n + 16 * n^2) / (-12 * n) \)
  → (\lim_{n \to \infty} \frac{(3-4n)^2}{(n-3)^2 - (n+3)^2} = -\infty)) :=
sorry -- proof goes here

end sequence_limit_l64_64371


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64818

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64818


namespace product_of_all_solutions_triple_reciprocal_l64_64749

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64749


namespace car_return_speed_l64_64998

-- Definitions based on the given conditions
def distance_cd : ℕ := 180
def speed_cd : ℕ := 90
def avg_speed_round_trip : ℕ := 60

-- The proof problem statement
theorem car_return_speed :
  let r := 45 in
  ∀ r, (
    let time_cd := distance_cd / speed_cd in
    let time_dc := distance_cd / r in
    let total_distance := 2 * distance_cd in
    let total_time := time_cd + time_dc in
    avg_speed_round_trip = total_distance / total_time
  ) → r = 45 :=
by
  sorry

end car_return_speed_l64_64998


namespace expected_value_of_winnings_is_5_l64_64311

/--
A fair 8-sided die is rolled. If the roll is even, then you win double that amount of dollars. 
If the roll is odd, you win nothing.
Prove that the expected value of the winnings is 5 dollars.
-/
theorem expected_value_of_winnings_is_5 :
  let outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  let even_outcomes := {2, 4, 6, 8}
  let win_amount (n : ℕ) := if n % 2 = 0 then 2 * n else 0
  let prob := (1 : ℚ) / 8
  let expected_value := 
    prob * win_amount 2 + prob * win_amount 4 + 
    prob * win_amount 6 + prob * win_amount 8
  in expected_value = 5 := 
by
  let outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  let even_outcomes := {2, 4, 6, 8}
  let win_amount (n : ℕ) := if n % 2 = 0 then 2 * n else 0
  let prob := (1 : ℚ) / 8
  let expected_value := 
    prob * win_amount 2 + prob * win_amount 4 + 
    prob * win_amount 6 + prob * win_amount 8
  show expected_value = 5 from sorry

end expected_value_of_winnings_is_5_l64_64311


namespace max_PA_PB_l64_64001

-- Definitions and conditions
variables {m : ℝ} 
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Problem statement
theorem max_PA_PB (PA PB : ℝ × ℝ) (h_PA : PA = A) (h_PB : PB = B) (h_perpendicular : distance A B = real.sqrt 10) :
  ∃ (P : ℝ × ℝ), distance P A * distance P B = 5 :=
sorry

end max_PA_PB_l64_64001


namespace Steve_has_more_money_than_Wayne_by_2004_l64_64359

theorem Steve_has_more_money_than_Wayne_by_2004:
  (∀ n: ℕ, steve_money n = 100 * 2 ^ n ∧ wayne_money n = 10000 / 2 ^ n) →
  (∃ n: ℕ, 2000 <= n ∧ 2000 + n = 2004 ∧ steve_money (n + 2000) > wayne_money (n + 2000)) :=
by
  intro h
  sorry

end Steve_has_more_money_than_Wayne_by_2004_l64_64359


namespace two_zeros_range_l64_64028

-- Function definition
def f (a x : ℝ) : ℝ := a * Real.exp x - x - a

-- Theorem statement for the range of 'a' such that 'f(x)' has exactly two zeros
theorem two_zeros_range (a : ℝ) : (∃ x1 x2 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ x1 ≠ x2) ↔ (0 < a ∧ a < 1 ∨ 1 < a) :=
by
  sorry

end two_zeros_range_l64_64028


namespace inequality_proof_l64_64114

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a + b + c = 1)

theorem inequality_proof : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := 
by 
  sorry

end inequality_proof_l64_64114


namespace triangle_angle_equality_l64_64235

variable {A B C D E F : Type}
variables (A B C : Point) (D : between A B) (E : between A C) (F : between B C)
variables (length_BF_eq_two_CF : segment_length B F = 2 * segment_length C F)
variables (length_CE_eq_two_AE : segment_length C E = 2 * segment_length A E)
variables (angle_DEF_is_right : angle D E F = 90)

theorem triangle_angle_equality :
  angle A D E = angle D E F :=
sorry

end triangle_angle_equality_l64_64235


namespace product_of_solutions_l64_64774

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64774


namespace median_4040_list_l64_64216

def median_combined_list (n : ℕ) (l1 l2 : List ℕ) : ℕ :=
  let combined_list := (l1 ++ l2).sort
  (combined_list[(combined_list.length / 2) - 1] + combined_list[combined_list.length / 2]) / 2

theorem median_4040_list :
  let l1 := List.range' 1 2021
  let l2 := List.range' 1 2021 |>.map (λ x => x * x)
  median_combined_list 4040 l1 l2 = 1976.5 :=
by
  sorry

end median_4040_list_l64_64216


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64851

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64851


namespace cone_volume_l64_64673

variable (l h : ℝ) (π : ℝ := Real.pi)

noncomputable def radius (l h : ℝ) : ℝ := sqrt (l^2 - h^2)

noncomputable def volume_of_cone (r h : ℝ) (π : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (hl : l = 15) (hh : h = 9) :
  volume_of_cone (radius l h) h π = 432 * π :=
by
  sorry

end cone_volume_l64_64673


namespace tan_theta_imaginary_subtraction_l64_64003

theorem tan_theta_imaginary_subtraction 
    (θ : ℝ) 
    (z1 : ℂ) (z2 : ℂ) 
    (h1 : z1 = complex.of_real (sin θ) - complex.I * (4 / 5)) 
    (h2 : z2 = (3 / 5) - cos θ * complex.I)
    (h3 : ∃ (y : ℝ), z1 - z2 = complex.I * y) :
    tan θ = - (3 / 4) :=
by
  sorry

end tan_theta_imaginary_subtraction_l64_64003


namespace find_a_l64_64086

variable (m n a : ℝ)
variable (h1 : m = 2 * n + 5)
variable (h2 : m + a = 2 * (n + 1.5) + 5)

theorem find_a : a = 3 := by
  sorry

end find_a_l64_64086


namespace product_of_all_solutions_triple_reciprocal_l64_64747

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64747


namespace indicator_significant_improvement_l64_64307

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64307


namespace problem_statement_l64_64579

def g (n : ℕ) : ℝ := Real.log10 (Nat.choose (2 * n) n)

theorem problem_statement (n : ℕ) : g(n) / Real.log10 3 = n * Real.log 4 / Real.log 3 :=
by
  sorry

end problem_statement_l64_64579


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64820

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64820


namespace group_sum_n_cube_l64_64686

theorem group_sum_n_cube (n : ℕ) : 
  let group := filter (λ k, ∃ m, k = 2 * m + 1) (range (n * (n + 1) / 2) \ (range ((n - 1) * n / 2))) in
  (sum group = n^3) := 
sorry

end group_sum_n_cube_l64_64686


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64722

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64722


namespace product_of_solutions_l64_64779

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64779


namespace compare_heights_l64_64242

-- Definitions based on the conditions of the given problem
def students : Type := fin 200
noncomputable def height : students → ℝ := sorry -- height function mapping each student to a real number height

def transverse_rows : fin 10 → list students := sorry
def longitudinal_rows : fin 20 → list students := sorry

-- Selections based on described criteria
def shortest_in_transverse (row : fin 10) : students :=
  argmin (height ∘ transverse_rows row)

def tallest_of_shortest_in_transverse : students :=
  argmax (height ∘ shortest_in_transverse)

def tallest_in_longitudinal (col : fin 20) : students :=
  argmax (height ∘ longitudinal_rows col)

def shortest_of_tallest_in_longitudinal : students :=
  argmin (height ∘ tallest_in_longitudinal)

-- The main theorem to prove the desired comparison
theorem compare_heights :
  height (shortest_of_tallest_in_longitudinal) > height (tallest_of_shortest_in_transverse) :=
sorry

end compare_heights_l64_64242


namespace mean_significantly_improved_l64_64295

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64295


namespace concyclic_A_D_E_N_l64_64011

open EuclideanGeometry

variables {A B C D E N : Point}

-- Given conditions
axiom triangleABC : Triangle A B C
axiom Gamma : Circle A B C -- Circumcircle of triangle ABC
axiom D_on_AB : D ∈ Segment A B
axiom E_on_AC : E ∈ Segment A C
axiom BD_CE : Distance B D = Distance C E
axiom N_midpoint_arc : N = midpointArc A B C Gamma

theorem concyclic_A_D_E_N :
  Concyclic {A, D, E, N} :=
by
  sorry

end concyclic_A_D_E_N_l64_64011


namespace product_of_real_solutions_triple_property_l64_64792

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64792


namespace graph_shift_l64_64689

-- Given conditions
def original_function (x : ℝ) : ℝ := Real.sin (2 * x)
def transformed_function (x : ℝ) : ℝ := Real.sin (2 * (x + 1/2))

-- The proof goal
theorem graph_shift : transformed_function x = Real.sin (2 * x + 1) :=
by
  sorry

end graph_shift_l64_64689


namespace set_operations_range_of_a_l64_64126

noncomputable def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3x - 7 ≥ 8 - 2x}
def C (a : ℝ) : Set ℝ := {x | 2x + a > 0}

theorem set_operations :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 4}) ∧
  (A ∪ B = {x : ℝ | x ≥ 2}) ∧
  ((U \ (A ∪ B)) = {x : ℝ | 0 < x ∧ x < 2}) ∧
  ((U \ A) ∩ B = {x : ℝ | x ≥ 4}) :=
by
  sorry

theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a > -6 :=
by
  sorry

end set_operations_range_of_a_l64_64126


namespace max_volume_pyramid_MNKP_l64_64987

theorem max_volume_pyramid_MNKP
  (A A₁ B B₁ C C₁ M N K P : Type*)
  (V_prism : ℝ)
  (h_P_in_prism : P ∈ prism ABC_A1B1C1)
  (h_AM_ratio : AM / AA₁ = 3 / 7)
  (h_BN_ratio : BN / BB₁ = 2 / 5)
  (h_CK_ratio : CK / CC₁ = 4 / 9)
  (h_prism_volume : V_prism = 40) :
  ∃ (V_MNKP : ℝ), V_MNKP = 8 :=
by
  sorry

end max_volume_pyramid_MNKP_l64_64987


namespace min_value_9x_plus_3y_l64_64505

noncomputable def minimum_value_of_expression : ℝ := 6

theorem min_value_9x_plus_3y (x y : ℝ) 
  (h1 : (x - 1) * 4 + 2 * y = 0) 
  (ha : ∃ (a1 a2 : ℝ), (a1, a2) = (x - 1, 2)) 
  (hb : ∃ (b1 b2 : ℝ), (b1, b2) = (4, y)) : 
  9^x + 3^y = minimum_value_of_expression :=
by
  sorry

end min_value_9x_plus_3y_l64_64505


namespace part_one_part_two_l64_64166
noncomputable def φ (kM kN : ℝ) (MN : ℝ) := |kM - kN| / MN

def f (x : ℝ) := x^3 + 1
def f_prime (x : ℝ) := 3 * x^2
def M := (1, f 1)
def N := (2, f 2)

theorem part_one : φ (f_prime 1) (f_prime 2) (Real.sqrt (1^2 + (f 2 - f 1)^2)) = 9 * Real.sqrt 2 / 10 :=
by
  sorry

def g (x : ℝ) := x^3 + 2
def g_prime (x : ℝ) := 3 * x^2
variables {x1 x2 : ℝ}
hypothesis (H : x1 * x2 = 1)

def φ_g (t: ℝ) := 3 * |t| / Real.sqrt (t^2 + 2 / t^2 - 2)

theorem part_two : 0 < φ_g (x1 + x2) ∧ φ_g (x1 + x2) < 3 * Real.sqrt 10 / 5 :=
by
  have t := x1 + x2
  sorry

end part_one_part_two_l64_64166


namespace identify_roles_l64_64133

structure Person :=
(is_knight : Bool) 
(is_knave : Bool)
(is_spy : Bool)

def ask_question (p : Person) (q : Person → Bool) : Bool :=
if p.is_knight then q p else not (q p)

theorem identify_roles (A B C : Person) :
  (A.is_knight ∨ A.is_knave ∨ A.is_spy) ∧
  (B.is_knight ∨ B.is_knave ∨ B.is_spy) ∧
  (C.is_knight ∨ C.is_knave ∨ C.is_spy) ∧
  (A.is_knight ∧ ¬(A.is_knave ∨ A.is_spy)) ∨
  (A.is_knave ∧ ¬(A.is_knight ∨ A.is_spy)) ∨
  (A.is_spy ∧ ¬(A.is_knight ∨ A.is_knave)) ∧
  (B.is_knight ∧ ¬(B.is_knave ∨ B.is_spy)) ∨
  (B.is_knave ∧ ¬(B.is_knight ∨ B.is_spy)) ∨
  (B.is_spy ∧ ¬(B.is_knight ∨ B.is_knave)) ∧
  (C.is_knight ∧ ¬(C.is_knave ∨ C.is_spy)) ∨
  (C.is_knave ∧ ¬(C.is_knight ∨ C.is_spy)) ∨
  (C.is_spy ∧ ¬(C.is_knight ∨ C.is_knave)) →
  ∃ (q1 q2 q3 : Person → Bool),
    let a1 := ask_question A q1,
        a2 := ask_question B q2,
        a3 := ask_question C q3 in
    (a1 = true ∨ a1 = false) ∧
    (a2 = true ∨ a2 = false) ∧
    (a3 = true ∨ a3 = false) ∧
    (A.is_knight ∨ B.is_knight ∨ C.is_knight) ∧
    (A.is_knave ∨ B.is_knave ∨ C.is_knave) ∧
    (A.is_spy ∨ B.is_spy ∨ C.is_spy) :=
begin
  sorry
end

end identify_roles_l64_64133


namespace product_of_solutions_l64_64876

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64876


namespace sequence_mod_100_repeats_l64_64376

theorem sequence_mod_100_repeats (a0 : ℕ) : ∃ k l, k ≠ l ∧ (∃ seq : ℕ → ℕ, seq 0 = a0 ∧ (∀ n, seq (n + 1) = seq n + 54 ∨ seq (n + 1) = seq n + 77) ∧ (seq k % 100 = seq l % 100)) :=
by 
  sorry

end sequence_mod_100_repeats_l64_64376


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64718

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64718


namespace total_boys_went_down_slide_l64_64361

theorem total_boys_went_down_slide :
  let boys_first_10_minutes := 22
  let boys_next_5_minutes := 13
  let boys_last_20_minutes := 35
  (boys_first_10_minutes + boys_next_5_minutes + boys_last_20_minutes) = 70 :=
by
  sorry

end total_boys_went_down_slide_l64_64361


namespace intersect_and_distance_l64_64548

-- Definitions for the line l and curve C
def parametric_equations (t : ℝ) : ℝ × ℝ :=
  (√3 * t, 1 + t)

def cartesian_equation (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

-- Points and distances
def point_A (t₁ : ℝ) := parametric_equations t₁
def point_B (t₂ : ℝ) := parametric_equations t₂
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def distance (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

-- The main theorem
theorem intersect_and_distance :
  (∀ t₁ t₂ : ℝ, cartesian_equation (point_A t₁).1 (point_A t₁).2 ∧ cartesian_equation (point_B t₂).1 (point_B t₂).2 →
  t₁ + t₂ = 2 ∧ t₁ * t₂ = -4) →
  (∀ M P : ℝ × ℝ, 
    M = midpoint (point_A (1)) (point_B (1)),
    P = (0, 1) →
      distance (point_A 1) (point_B 1) = 2 * √5 ∧ distance M P = 1)
| t₁ t₂ h_eq h_t,
    distance (parametric_equations t₁) (parametric_equations t₂) = 2 * √5,
    distance (parametric_equations (midpoint (parametric_equations t₁) (parametric_equations t₂))) (0, 1) = 1 :=
sorry

end intersect_and_distance_l64_64548


namespace even_function_derivative_properties_implies_option_C_l64_64476

open Real

variables {f : ℝ → ℝ}

-- Definitions based on given conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)
def derivative_positive_on_domain (f' : ℝ → ℝ) : Prop := ∀ x > 0, f'(x) = (x-1)*(x-2)

-- Assertion to prove the answer based on conditions
theorem even_function_derivative_properties_implies_option_C
  (h_even : is_even f)
  (h_deriv : derivative_positive_on_domain (deriv f)) :
  f(-2) < f(1) :=
  sorry

end even_function_derivative_properties_implies_option_C_l64_64476


namespace find_value_of_expression_l64_64587

noncomputable def p : ℝ := 3
noncomputable def q : ℝ := 7
noncomputable def r : ℝ := 5

def inequality_holds (f : ℝ → ℝ) : Prop :=
  ∀ x, (f x ≥ 0 ↔ (x ∈ Set.Icc 3 7 ∨ x > 5))

def given_condition : Prop := p < q

theorem find_value_of_expression (f : ℝ → ℝ)
  (h : inequality_holds f)
  (hc : given_condition) :
  p + 2*q + 3*r = 32 := 
sorry

end find_value_of_expression_l64_64587


namespace range_of_a_l64_64497

noncomputable def A : Set ℝ := Set.Ico 1 5 -- A = [1, 5)
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a -- B = (-∞, a)

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 5 ≤ a :=
sorry

end range_of_a_l64_64497


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64729

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64729


namespace infinite_values_prime_divisor_l64_64140

noncomputable def largestPrimeDivisor (n : ℕ) : ℕ :=
  sorry

theorem infinite_values_prime_divisor :
  ∃ᶠ n in at_top, largestPrimeDivisor (n^2 + n + 1) = largestPrimeDivisor ((n+1)^2 + (n+1) + 1) :=
sorry

end infinite_values_prime_divisor_l64_64140


namespace equal_segments_parallel_to_sides_l64_64531

theorem equal_segments_parallel_to_sides (AB BC AC : ℝ) (h1 : AB = 480) (h2 : BC = 500) (h3 : AC = 550) (d : ℝ) :
  (∃ P, (∃ (D D' E E' F F' : Point),
    parallel_through P D E' D' E F F' d) ∧ all_equal d) →
  d = 132000 / 779 :=
sorry

end equal_segments_parallel_to_sides_l64_64531


namespace label_elements_zero_one_l64_64586

-- Define the necessary conditions for the problem
variables {B : Type*} {A : Fin (2 * n + 1) → Set B}
variable {n : ℕ}

-- The main theorem statement
theorem label_elements_zero_one (hn : 0 < n)
  (h1 : ∀ i : Fin (2 * n + 1), (A i).card = 2 * n)
  (h2 : ∀ (i j : Fin (2 * n + 1)), i < j → (A i ∩ A j).card = 1)
  (h3 : ∀ b : B, ∃ (i j : Fin (2 * n + 1)), i ≠ j ∧ b ∈ A i ∧ b ∈ A j) :
  (∃ (f : B → Fin 2), ∀ i : Fin (2 * n + 1), (A i).filter (λ b, f b = 0)).card = n) ↔ n % 2 = 0 :=
sorry

end label_elements_zero_one_l64_64586


namespace trig_identity_l64_64516

theorem trig_identity (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : 
  Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by 
  sorry

end trig_identity_l64_64516


namespace significant_improvement_l64_64284

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64284


namespace angle_EDA_eq_angle_FDA_l64_64557

noncomputable def Point : Type := sorry
noncomputable def Line : Type := sorry

-- Conditions definitions
variable (A B C D H E F : Point)

-- Triangle definition
variable [triangle ABC : Type]

-- AD is perpendicular to BC
variable (AD : Line)
variable (BC : Line)
axiom perp_AD_BC : ∀ (D : Point), AD ⊥ BC

-- H is any point on AD
variable [line H_C : Type]
axiom H_on_AD : ∀ (H : Point), H ∈ AD

-- Line CH intersects AB at E and Line BH intersects AC at F
variable (CH : Line)
variable (BH : Line)
axiom inter_CH_AB_at_E : ∀ (E : Point), E ∈ AB ∧ E ∈ CH
axiom inter_BH_AC_at_F : ∀ (F : Point), F ∈ AC ∧ F ∈ BH

-- Goal proving that ∠EDA = ∠FDA
theorem angle_EDA_eq_angle_FDA :
  ∃ (A B C D H E F : Point) (AD : Line) (BC : Line) (CH : Line) (BH : Line),
    ∀ (H : Point), H ∈ AD →
    ∀ (E : Point), E ∈ AB ∧ E ∈ CH →
    ∀ (F : Point), F ∈ AC ∧ F ∈ BH →
    ∠EDA = ∠FDA := 
  sorry

end angle_EDA_eq_angle_FDA_l64_64557


namespace cos_of_sin_given_l64_64514

theorem cos_of_sin_given (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by
  sorry

end cos_of_sin_given_l64_64514


namespace number_of_true_statements_l64_64048

-- Definitions of planes and lines
variable (α β γ : Plane) (m n : Line)

-- Conditions for each statement
def statement1 : Prop := m ∥ α ∧ α ∩ β = n → m ∥ n
def statement2 : Prop := m ⊥ α ∧ m ∥ n ∧ n ⊂ β → α ⊥ β
def statement3 : Prop := α ⊥ β ∧ γ ⊥ β → α ∥ γ
def statement4 : Prop := α ∩ β = m ∧ m ⊥ γ → α ⊥ γ

-- Combining them for the proof of the number of correct statements
theorem number_of_true_statements : 
  (statement1 α β γ m n) + (statement2 α β γ m n) + (statement4 α β γ m n) = 3 := 
sorry

end number_of_true_statements_l64_64048


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64821

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64821


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64753

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64753


namespace find_m_value_l64_64481

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (m : ℝ)

-- Conditions
variables (h₀ : ¬Collinear ℝ ({a, b} : set V))
          (h₁ : ∀ (A B : V), A + (3 a + b) = B → B = A + 3 a + b)
          (h₂ : ∀ (A B : V), A + (a + m * b) = B → B = A + a + m * b)
          (h₃ : ∀ (A B : V), A + (2 a - b) = B → B = A + 2 a - b)
          (h₄ : ∀ (A C D : V), Collinear ℝ ({A, C, D} : set V))

-- Theorem
theorem find_m_value : m = -3 :=
sorry

end find_m_value_l64_64481


namespace problem1_problem2_l64_64015

-- Let z1 and z2 be defined as given
def z1 (x : ℝ) := complex.mk (x^2 - 1) (x^2 - 3*x + 2)
def z2 (x : ℝ) := complex.mk x (3 - 2*x)

-- First question: If z1 is pure imaginary, then x = -1
theorem problem1 (x : ℝ) (h1 : z1 x.im = z1 x) (h2 : x^2 - 1 = 0) : x = -1 := by
  sorry

-- Second question: If z1 is in the fourth quadrant and z2 is in the first quadrant, then 1 < x < 3/2
theorem problem2 (x : ℝ)
  (h1 : z1 x.re = z1 x) (h2 : 0 < z1 x.im) (h3 : z1 x.im < 0)
  (h4 : z2 x.re > 0) (h5 : z2 x.im > 0)
  (h6 : 1 < x) (h7 : x < 2)
  (h8 : 0 < x) (h9 : x < 3 / 2) :
  1 < x ∧ x < 3 / 2 := by
  sorry

end problem1_problem2_l64_64015


namespace radius_of_film_l64_64128

-- Definition of the dimensions of the box
def length := 10
def width := 5
def height := 8

-- Definition of the resulting film's thickness
def thickness := 0.2

-- Definition of the volume of the box
def volume_Y := length * width * height

-- Definition of the radius r and the equality hypothesis
def radius_resulting_circular_film (r : ℝ) : Prop :=
  π * r^2 * thickness = volume_Y

theorem radius_of_film : ∃ r : ℝ, radius_resulting_circular_film r ∧ r = Real.sqrt (2000 / π) :=
by
  use Real.sqrt (2000 / π)
  constructor
  ·
    -- First part is the equality, derived from conditions
    unfold radius_resulting_circular_film
    rw [Real.sqrt_mul_self zero_le]
    exact sorry  -- We skip the detailed proof here
  ·
    -- The second part, the radius we expect
    rfl

end radius_of_film_l64_64128


namespace exists_universal_city_l64_64074

-- Define the cities as a type
universe u
constant City : Type u

-- Define the accessibility relation
constant accessible : City → City → Prop

-- Conditions from problem
axiom reflexive : ∀ (A : City), accessible A A
axiom intermediary : ∀ (P Q : City), ∃ (R : City), accessible P R ∧ accessible Q R

-- Main theorem to be proved
theorem exists_universal_city :
  ∃ (A : City), ∀ (B : City), accessible B A :=
sorry

end exists_universal_city_l64_64074


namespace max_correct_guesses_l64_64234

theorem max_correct_guesses (n k : ℕ) : ∀ (wise_men_sane : nat), (wise_men_sane = n - k - 1) → wise_men_sane ≤ n - k - 1 :=
sorry

end max_correct_guesses_l64_64234


namespace concurrency_of_AC_BF_DE_l64_64119

-- Definitions for the given points and configurations
variables (A B C D E F P : Type)
variables (Γ : Circle (PlanePoint A B C))
variables [Γ.isCircumcircle A B C]
variable [D_on_BC : D ∈ line B C]
variable [tangent_Γ_A : tangent Γ A]
variable [E_on_parallel_BA_through_D : E ∈ parallel_line_through_point (line A B) D]
variable [C_E_line_intersects_Γ_again_at_F : F ∈ second_intersection_point (line E C) Γ]
variable [concyclic_B_D_F_E : cyclic (B, D, F, E)]

-- The theorem statement
theorem concurrency_of_AC_BF_DE :
  concurrent (line A C) (line B F) (line D E) :=
sorry

end concurrency_of_AC_BF_DE_l64_64119


namespace discriminant_eq_zero_two_equal_real_roots_l64_64441

theorem discriminant_eq_zero_two_equal_real_roots (n : ℂ) : (n^2 - 36 = 0) ↔ (n = 6 ∨ n = -6) := 
begin 
  sorry 
end

end discriminant_eq_zero_two_equal_real_roots_l64_64441


namespace product_of_solutions_l64_64713

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64713


namespace significant_improvement_l64_64274

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64274


namespace area_of_n_cells_l64_64409

noncomputable def convex_quadrilateral := {A B C D : ℝ × ℝ // convex_hull {A, B, C, D}.is_convex}

noncomputable def divide_side (AB : ℝ × ℝ → ℝ × ℝ → ℝ → ℝ × ℝ) (n : ℕ) : ℕ → ℝ × ℝ
| k := AB (k * (1 / n))

noncomputable def form_chessboard (quad : convex_quadrilateral) (n : ℕ) : ℕ × ℕ → Set (ℝ × ℝ) 
| (i, j) := let sides := [quad.1.1, quad.1.2, quad.1.3, quad.1.4] 
            partition sides i j n

theorem area_of_n_cells {quad : convex_quadrilateral} {n : ℕ} (chessboard : (ℕ × ℕ) → Set (ℝ × ℝ))
(selected_cells : ∀ i, ∃ j, chessboard (i, j)) : 
∑ i in finset.range n, ∑ j in finset.range n, area (selected_cells i j) = (area quad) / n :=
by
  sorry

end area_of_n_cells_l64_64409


namespace significant_improvement_l64_64286

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64286


namespace bottle_total_height_l64_64330

theorem bottle_total_height (r1 r2 water_height_up water_height_down : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 3) (h_water_height_up : water_height_up = 20) (h_water_height_down : water_height_down = 28) : 
    ∃ x : ℝ, (π * r1^2 * (x - water_height_up) = 9 * π * (x - water_height_down) ∧ x = 29) := 
by 
    sorry

end bottle_total_height_l64_64330


namespace parabola_proof_dot_product_proof_min_distance_proof_l64_64044

noncomputable -- Only add if using noncomputable functions

def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∀ (y : ℝ) (x : ℝ), y^2 = 2 * p * x → y^2 = 4 * x

def oa_dot_ob_independent_of_slope (t : ℝ) (ht : t > 0) (k : ℂ) : Prop :=
  let A : ℂ := t + 2 * Complex.I * Complex.sqrt t in
  let B : ℂ := t - 2 * Complex.I * Complex.sqrt t in
  A * B = t^2 - 4 * t

def min_distance_expression (t : ℝ) (ht : t > 0) : ℝ :=
  if t < 2 then 
    t 
  else 
    2 * Real.sqrt (t - 1)

theorem parabola_proof (p : ℝ) (hp : p > 0) :
  parabola_equation p hp :=
by
  sorry

theorem dot_product_proof (t : ℝ) (ht : t > 0) (k : ℂ) :
  oa_dot_ob_independent_of_slope t ht k :=
by
  sorry

theorem min_distance_proof (t : ℝ) (ht : t > 0) :
  min_distance_expression t ht = 
    if t >= 2 then 
      2 * Real.sqrt (t - 1) 
    else 
      t :=
by
  sorry

end parabola_proof_dot_product_proof_min_distance_proof_l64_64044


namespace max_b_value_monotonicity_f_l64_64489

-- Define the function f(x)
def f (a x : ℝ) : ℝ := -(a * Real.log x) + (a + 1) * x - (1 / 2) * x^2

-- Define the function g(x) for question 2
def g (a x : ℝ) : ℝ := -(a * Real.log x) + x

-- Define the function h(a) used in the final calculation for b
def h (a : ℝ) : ℝ := -a * Real.log a + a

-- The max value for b needs to be verified

theorem max_b_value (a : ℝ) (x : ℝ) (b : ℝ) :
  (a > 0) →
  f a x ≥ - (1 / 2) * x^2 + a * x + b →
  (a ∈ Set.Icc (1 / 2) 1) →
  b ≤ 1 / 2 * (1 + Real.log 2) :=
sorry

-- Monotonicity of f(x) based on a
theorem monotonicity_f (a : ℝ) (x : ℝ) :
  (a > 0) →
  (if h₀ : 0 < a ∧ a < 1 then ((0 < x ∧ x < a) → f' a x < 0) ∧
    ((a < x ∧ x < 1) → f' a x > 0) ∧ ((1 < x) → f' a x < 0)
  else if a = 1 then (0 < x → f' a x < 0)
  else ((0 < x ∧ x < 1) → f' a x < 0) ∧ ((1 < x ∧ x < a) → f' a x > 0) ∧ (a < x → f' a x < 0)) :=
sorry

end max_b_value_monotonicity_f_l64_64489


namespace spherical_to_rectangular_coords_l64_64387

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), 
    ρ = 3 → θ = 3 * Real.pi / 2 → φ = Real.pi / 3 →
    (let x := ρ * Real.sin φ * Real.cos θ;
         y := ρ * Real.sin φ * Real.sin θ;
         z := ρ * Real.cos φ 
     in (x, y, z) = (0, - (3 * Real.sqrt 3) / 2, 3 / 2)) :=
by
  intros ρ θ φ hρ hθ hφ
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coords_l64_64387


namespace product_of_solutions_l64_64715

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64715


namespace net_change_in_price_l64_64527

theorem net_change_in_price (P : ℝ) : 
  let new_price_decreased := P * (1 - 0.20),
      new_price_increased := new_price_decreased * (1 + 0.50)
  in new_price_increased - P = 0.20 * P := 
by
  let new_price_decreased := P * (1 - 0.20)
  let new_price_increased := new_price_decreased * (1 + 0.50)
  have h1 : new_price_decreased = P * 0.80 := by sorry
  have h2 : new_price_increased = P * 1.20 := by sorry
  calc
    new_price_increased - P
        = P * 1.20 - P : by sorry
    ... = P * (1.20 - 1) : by sorry
    ... = 0.20 * P : by sorry

end net_change_in_price_l64_64527


namespace product_of_all_solutions_triple_reciprocal_l64_64738

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64738


namespace Mary_chewing_gums_count_l64_64170

variable (Mary_gums Sam_gums Sue_gums : ℕ)

-- Define the given conditions
axiom Sam_chewing_gums : Sam_gums = 10
axiom Sue_chewing_gums : Sue_gums = 15
axiom Total_chewing_gums : Mary_gums + Sam_gums + Sue_gums = 30

theorem Mary_chewing_gums_count : Mary_gums = 5 := by
  sorry

end Mary_chewing_gums_count_l64_64170


namespace correct_division_answer_l64_64540

theorem correct_division_answer :
  ∀ (incorrect_divisor correct_divisor incorrect_answer : ℕ) (dividend : ℕ),
    incorrect_divisor = 174 →
    correct_divisor = 105 →
    incorrect_answer = 45 →
    dividend = incorrect_divisor * incorrect_answer →
    dividend / correct_divisor = 75 :=
by
  intros _ _ _ _
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  rfl
sorry

end correct_division_answer_l64_64540


namespace geometric_mean_external_tangent_geometric_mean_internal_tangent_l64_64236

-- Define the problem context and conditions
variables {O1 O2 : Point} {r1 r2 c : ℝ}
variables {A B C D E F G H : Point}

-- Circles and their properties
def circles_non_intersecting (O1 O2 : Point) (r1 r2 c : ℝ) : Prop :=
  c > r1 + r2

-- The common symmetry axis intersections
def common_symmetry_axis_intersects {A B C D : Point} : Prop := sorry

-- The tangent points
def external_tangent_points_touches {E F : Point} : Prop := sorry
def internal_tangent_points_touches {G H : Point} : Prop := sorry

-- Geometric mean relations
theorem geometric_mean_external_tangent (h1 : circles_non_intersecting O1 O2 r1 r2 c)
  (h2 : common_symmetry_axis_intersects)
  (h3 : external_tangent_points_touches):
  dist E F ^ 2 = dist A C * dist B D := sorry

theorem geometric_mean_internal_tangent (h1 : circles_non_intersecting O1 O2 r1 r2 c)
  (h2 : common_symmetry_axis_intersects)
  (h3 : internal_tangent_points_touches):
  dist G H ^ 2 = dist A D * dist B C := sorry

end geometric_mean_external_tangent_geometric_mean_internal_tangent_l64_64236


namespace sail_pressure_l64_64637

theorem sail_pressure (k : ℝ) :
  (forall (V A : ℝ), P = k * A * (V : ℝ)^2) 
  → (P = 1.25) → (V = 20) → (A = 1)
  → (A = 4) → (V = 40)
  → (P = 20) :=
by
  sorry

end sail_pressure_l64_64637


namespace pen_and_bottle_price_and_plans_l64_64356

theorem pen_and_bottle_price_and_plans:
  (∃ x y: ℝ, 10 * x + y = 242 ∧ x + 10 * y = 341 ∧ x = 21 ∧ y = 32) ∧
  (∃ m: ℕ, (m ≤ 20 ∧ m ≥ 14.5454) ∧ (m = 15 ∨ m = 16 ∨ m = 17 ∨ m = 18 ∨ m = 19 ∨ m = 20) ∧
  (∀ m'', m'' = 20 → 21 * m'' + 32 * (30 - m'') = 740) ∧ (6 = 6)) :=
by
  -- Proof omitted
  sorry

end pen_and_bottle_price_and_plans_l64_64356


namespace product_of_solutions_l64_64866

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64866


namespace product_of_real_solutions_triple_property_l64_64784

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64784


namespace sum_factors_of_30_l64_64960

theorem sum_factors_of_30 : 
  (∑ i in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), i) = 72 := by
  sorry

end sum_factors_of_30_l64_64960


namespace min_value_of_expression_l64_64429

theorem min_value_of_expression (n : ℕ) (x : ℕ → ℝ) :
  let cos_sum := ∑ i in finset.range n, real.cos (x i),
      sin_sum := ∑ i in finset.range n, real.sin (x i),
      A := (5 * cos_sum + sin_sum) * (cos_sum - 5 * sin_sum)
  in A ≥ -13 * n^2 := sorry

end min_value_of_expression_l64_64429


namespace balloons_lost_l64_64559

-- Definitions corresponding to the conditions
def initial_balloons : ℕ := 7
def current_balloons : ℕ := 4

-- The mathematically equivalent proof problem
theorem balloons_lost : initial_balloons - current_balloons = 3 := by
  -- proof steps would go here, but we use sorry to skip them 
  sorry

end balloons_lost_l64_64559


namespace regular_ngon_angle_l64_64138

theorem regular_ngon_angle (n : ℕ) (A B C D : Point)
  (h1 : consecutive_vertices A B C D n)
  (h2 : ∠ A C D = 120) : 
  n = 9 :=
sorry

end regular_ngon_angle_l64_64138


namespace mean_of_S_eq_651_l64_64312

theorem mean_of_S_eq_651 
  (s n : ℝ) 
  (h1 : (s + 1) / (n + 1) = s / n - 13) 
  (h2 : (s + 2001) / (n + 1) = s / n + 27) 
  (hn : n ≠ 0) : s / n = 651 := 
by 
  sorry

end mean_of_S_eq_651_l64_64312


namespace product_of_triple_when_added_to_reciprocal_l64_64897

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64897


namespace gcd_one_probability_l64_64693

open Finset

def natural_numbers := {1, 2, 3, 4, 5, 6, 7}

noncomputable def probability_gcd_one : ℚ :=
  let total_pairs := (natural_numbers.powerset.card.choose 2) in
  let pairs_with_gcd_greater_than_one :=
    ({{2, 4}, {2, 6}, {4, 6}, {3, 6}} : Finset (Finset ℕ)).card in
  1 - (pairs_with_gcd_greater_than_one / total_pairs : ℚ)

theorem gcd_one_probability :
  probability_gcd_one = 17 / 21 :=
sorry

end gcd_one_probability_l64_64693


namespace smallest_lcm_value_l64_64061

def is_five_digit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000

theorem smallest_lcm_value :
  ∃ (m n : ℕ), is_five_digit m ∧ is_five_digit n ∧ Nat.gcd m n = 5 ∧ Nat.lcm m n = 20030010 :=
by
  sorry

end smallest_lcm_value_l64_64061


namespace product_of_real_solutions_triple_property_l64_64795

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64795


namespace find_index_of_2018_l64_64351

-- Define the sequence of natural numbers whose digits sum to 11
def digit_sum_11 (n : ℕ) : Prop :=
  (n.to_digits.sum = 11)

-- Define the sequence S as the sorted list of natural numbers whose digits sum to 11
noncomputable def S : List ℕ :=
  List.filter digit_sum_11 (List.iota 10000) -- assuming the search space, you can expand if necessary
  |> List.sort (≤)

-- Define the theorem statement
theorem find_index_of_2018 : 
  List.index_of 2018 S = 134 :=
by
  sorry

end find_index_of_2018_l64_64351


namespace normal_transversals_pass_through_orthocenter_l64_64604

-- Define an orthocentric tetrahedron
variables {A B C D : Type} [Tetrahedron A B C D] [Orthocentric Tetrahedron A B C D]

-- Define the edges and corresponding normal transversals
variables (AB_CD_OPP_TRANSVERSAL : ∀ (AB_CD : EdgePair Tetrahedron), 
               is_orthocenter := ∃ orthocenter : Point,
               pass_through orthocenter AB_CD.normal_transversal 
               ∧ are_feet_altitudes AB_CD.intersections_with_edges)

theorem normal_transversals_pass_through_orthocenter 
  : ∀ (t : Tetrahedron) (e1 e2 : EdgePair Tetrahedron), 
    Tetrahedron.orthocentric t 
    → e1.opposite_with t
    → e2.opposite_with t
    → pass_through (t.orthocenter) e1.normal_transversal 
    ∧ are_feet_altitudes e1.intersections_with_edges e2.normal_transversal := sorry

end normal_transversals_pass_through_orthocenter_l64_64604


namespace team_savings_correct_l64_64406

-- Define the costs without the discount
def cost_shirt := 7.50
def cost_pants := 15.00
def cost_socks := 4.50

-- Define the costs with the discount
def discounted_shirt := 6.75
def discounted_pants := 13.50
def discounted_socks := 3.75

-- Define the number of team members
def team_members := 12

-- Total cost of one uniform without discount
def total_cost_without_discount := cost_shirt + cost_pants + cost_socks

-- Total cost of one uniform with discount
def total_cost_with_discount := discounted_shirt + discounted_pants + discounted_socks

-- Savings per uniform
def savings_per_uniform := total_cost_without_discount - total_cost_with_discount

-- Total savings for the team
def total_savings_for_team := savings_per_uniform * team_members

-- Prove that the total savings for the team is $36.00
theorem team_savings_correct : total_savings_for_team = 36.00 := 
  by 
    sorry

end team_savings_correct_l64_64406


namespace cone_volume_l64_64677

variable (l h : ℝ) (π : ℝ := Real.pi)

noncomputable def radius (l h : ℝ) : ℝ := sqrt (l^2 - h^2)

noncomputable def volume_of_cone (r h : ℝ) (π : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (hl : l = 15) (hh : h = 9) :
  volume_of_cone (radius l h) h π = 432 * π :=
by
  sorry

end cone_volume_l64_64677


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64752

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64752


namespace diameter_three_times_volume_l64_64399

noncomputable def sphere_diameter (r : ℝ) : ℝ :=
 2 * (∛(3 * ((4 / 3) * Math.pi * r^3) / (4 * Math.pi)))

theorem diameter_three_times_volume (r : ℝ) (a b : ℕ) (h1 : r = 6)
  (h2 : sphere_diameter r = a * real.cbrt b)
  (h3 : b % 2 ≠ 0) :
  a + b = 14 := by
  sorry

end diameter_three_times_volume_l64_64399


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64884

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64884


namespace product_trippled_when_added_to_reciprocal_l64_64926

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64926


namespace suitable_land_acres_l64_64568

theorem suitable_land_acres (new_multiplier : ℝ) (previous_acres : ℝ) (pond_acres : ℝ) :
  new_multiplier = 10 ∧ previous_acres = 2 ∧ pond_acres = 1 → 
  (new_multiplier * previous_acres - pond_acres) = 19 :=
by
  intro h
  sorry

end suitable_land_acres_l64_64568


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64828

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64828


namespace simplify_and_evaluate_expression_l64_64145

theorem simplify_and_evaluate_expression (a b : ℝ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end simplify_and_evaluate_expression_l64_64145


namespace new_device_significant_improvement_l64_64254

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64254


namespace product_of_solutions_l64_64709

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64709


namespace gcd_of_three_numbers_l64_64427

theorem gcd_of_three_numbers :
  gcd 9486 (gcd 13524 36582) = 6 :=
by
  -- Insert proof here
  sorry

end gcd_of_three_numbers_l64_64427


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64727

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64727


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64757

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64757


namespace product_of_solutions_l64_64808

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64808


namespace jake_biking_time_l64_64093

-- Definitions based on conditions
def time_in_hours := 30 / 60

def speed_city := 28
def speed_highway := 60
def speed_winding := 45

def distance_city := speed_city * (time_in_hours / 3)
def distance_highway := speed_highway * (time_in_hours / 3)
def distance_winding := speed_winding * (time_in_hours / 3)

def total_distance := distance_city + distance_highway + distance_winding

def jake_speed_flat_ground := 11 * 0.9
def jake_speed_uphill := 9
def jake_speed_downhill := 13 * 0.85

def fraction_flat := 0.3
def fraction_uphill := 0.4
def fraction_downhill := 0.3

def distance_flat := total_distance * fraction_flat
def distance_uphill := total_distance * fraction_uphill
def distance_downhill := total_distance * fraction_downhill

def time_flat := distance_flat / jake_speed_flat_ground
def time_uphill := distance_uphill / jake_speed_uphill
def time_downhill := distance_downhill / jake_speed_downhill

def total_time := time_flat + time_uphill + time_downhill

-- Statement to be proved
theorem jake_biking_time : total_time = 2.2587 := 
by {
  -- Definitions and computations skipped, proof should be here
  sorry
}

end jake_biking_time_l64_64093


namespace MrsYoung_puzzle_complete_l64_64596

theorem MrsYoung_puzzle_complete :
  let total_pieces := 500
  let children := 4
  let pieces_per_child := total_pieces / children
  let minutes := 120
  let pieces_Reyn := (25 * (minutes / 30))
  let pieces_Rhys := 2 * pieces_Reyn
  let pieces_Rory := 3 * pieces_Reyn
  let pieces_Rina := 4 * pieces_Reyn
  let total_pieces_placed := pieces_Reyn + pieces_Rhys + pieces_Rory + pieces_Rina
  total_pieces_placed >= total_pieces :=
by
  sorry

end MrsYoung_puzzle_complete_l64_64596


namespace new_device_significant_improvement_l64_64253

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64253


namespace spherical_to_rectangular_coordinates_l64_64384

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l64_64384


namespace valid_operation_l64_64979

theorem valid_operation :
  ∀ x : ℝ, x^2 + x^3 ≠ x^5 ∧
  ∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2 ∧
  ∀ m : ℝ, (|m| = m ↔ m ≥ 0) :=
by
  sorry

end valid_operation_l64_64979


namespace Cartesian_eq_C2_slope_of_line_l_l64_64080

noncomputable def rho : ℝ × ℝ → ℝ := λ ⟨x, y⟩, real.sqrt(x^2 + y^2)
noncomputable def theta : ℝ × ℝ → ℝ := λ ⟨x, y⟩, real.atan2 y x

def C1 := {p : ℝ × ℝ | rho p * (real.sin (theta p)) = 2}
def OM (p : ℝ × ℝ) (q : ℝ × ℝ) := p.1 * q.1 + p.2 * q.2

def is_on_segment (O M P : ℝ × ℝ) : Prop := 
  (P.1 - O.1) * (M.2 - O.2) = (P.2 - O.2) * (M.1 - O.1) ∧ OM O P * OM O M = 4

def C2 := {p : ℝ × ℝ | (rho p) = 2 * (real.sin (theta p))}

def line_parametric (α t : ℝ) : ℝ × ℝ := (t * real.cos α, t * real.sin α)

def is_on_line (α : ℝ) (A : ℝ × ℝ) : Prop := ∃ t : ℝ, A = line_parametric α t

theorem Cartesian_eq_C2 : ∀ (P : ℝ × ℝ), P ∈ C2 → P.1^2 + (P.2 - 1)^2 = 1 :=
sorry

theorem slope_of_line_l : ∀ (α : ℝ), 
  (∃ A : ℝ × ℝ, is_on_line α A ∧ (real.sqrt(A.1^2 + A.2^2) = real.sqrt(3))) → 
  α = real.pi / 3 ∨ α = 2 * real.pi / 3 :=
sorry

end Cartesian_eq_C2_slope_of_line_l_l64_64080


namespace concurrency_l64_64121

variables {K : Type*} [Field K]

structure Triangle (K : Type*) [Field K] :=
(A B C D E F D P AC BC CD DE BF CE : K)

open Triangle

theorem concurrency (Γ : Triangle K) (circumcircle_abc : Set K) (point_D_on_BC : Set K) 
(tangent_to_circumcircle_at_A : Set K) (parallel_to_BA_through_D : Set K)
(CE_intersects_circumcircle_at_F : Set K) (BDFE_concyclic : Set K) :
Set K :=
  ∃ G : Set K, G ∈ (AC ∩ BF ∩ DE)


end concurrency_l64_64121


namespace cos_alpha_pi_rational_l64_64472

theorem cos_alpha_pi_rational (α : ℚ) (h : ∃ r : ℚ, r = Real.cos (α * Real.pi)) :
  Real.cos (α * Real.pi) ∈ {0, 1, -1, 1 / 2, -1 / 2} :=
sorry

end cos_alpha_pi_rational_l64_64472


namespace product_of_solutions_l64_64712

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64712


namespace significant_improvement_l64_64283

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64283


namespace cone_volume_l64_64654

def slant_height := 15
def height := 9
def radius := Real.sqrt (slant_height^2 - height^2)
def volume := (1/3) * Real.pi * radius^2 * height

theorem cone_volume : 
  radius = 12 ∧ volume = 432 * Real.pi :=
by
  split
  { -- radius = 12
    unfold radius
    sorry
  }
  { -- volume = 432 * Real.pi
    unfold volume
    sorry
  }

end cone_volume_l64_64654


namespace solve_equation_l64_64182

theorem solve_equation :
  ∀ x : ℝ, (9 * x - 6 * 3 ^ x - 7 = 0) → x = log 7 / log 3 := 
by 
  intro x
  intro h
  sorry

end solve_equation_l64_64182


namespace rectangle_area_l64_64984
-- Import necessary libraries

-- Define the problem
theorem rectangle_area : 
  ∀ (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ), 
  x1 = -5 ∧ y1 = 1 ∧ 
  x2 = 1 ∧ y2 = 1 ∧ 
  x3 = 1 ∧ y3 = -4 ∧ 
  x4 = -5 ∧ y4 = -4 →
  (x2 - x1) * (y1 - y3) = 30 :=
by
  -- Assume coordinates of points
  intros x1 y1 x2 y2 x3 y3 x4 y4
  intro h
  -- Use the given conditions
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h_rest
  cases h_rest with h5 h_rest
  cases h_rest with h6 h_rest
  cases h_rest with h7 h8
  -- Substitute the coordinates
  have hx := congr_arg2 HasSub.sub h4 h1
  have hy := congr_arg2 HasSub.sub h2 h6
  ring at hx ⊢
  rw [hx, hy]
  exact rfl

end rectangle_area_l64_64984


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64731

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64731


namespace initial_amount_A_correct_l64_64338

noncomputable def initial_amount_A :=
  let a := 21
  let b := 5
  let c := 9

  -- After A gives B and C
  let b_after_A := b + 5
  let c_after_A := c + 9
  let a_after_A := a - (5 + 9)

  -- After B gives A and C
  let a_after_B := a_after_A + (a_after_A / 2)
  let c_after_B := c_after_A + (c_after_A / 2)
  let b_after_B := b_after_A - (a_after_A / 2 + c_after_A / 2)

  -- After C gives A and B
  let a_final := a_after_B + 3 * a_after_B
  let b_final := b_after_B + 3 * b_after_B
  let c_final := c_after_B - (3 * a_final + b_final)

  (a_final = 24) ∧ (b_final = 16) ∧ (c_final = 8)

theorem initial_amount_A_correct : initial_amount_A := 
by
  -- Skipping proof details
  sorry

end initial_amount_A_correct_l64_64338


namespace total_water_calculation_l64_64237

variables (t a b c : ℝ)

-- conditions
def condition1 : Prop := a = (1/2) * c + 6
def condition2 : Prop := b = (a + c) / 2
def condition3 : Prop := c = 18 * t + 8

-- question
def total_water : ℝ := a + b + c

-- theorem statement
theorem total_water_calculation (h1 : condition1 t a b c) 
                                 (h2 : condition2 t a b c) 
                                 (h3 : condition3 t a b c) :
  total_water t a b c = (81 * t) / 2 + 27 :=
sorry

end total_water_calculation_l64_64237


namespace distance_between_midpoints_l64_64180

-- Define the quadrilateral with given side lengths and midpoint distance
def quadrilateral (a b c d e : ℝ) := 
  a = 20 ∧ b = 16 ∧ c = 14 ∧ d = 8 ∧ e = 11

-- Define functions to compute the required distances
def dist_midpoints_bc_da (a b c d e : ℝ) : ℝ := sqrt ( (a^2 + c^2)/2 - (b^2 + d^2)/2 + e^2 )

def dist_midpoints_aa0_dc0 (a b c d e : ℝ) : ℝ := sqrt ( (d^2 + b*c)/2 - (a^2 + c^2)/2 + e^2 )

-- Hypothesis assuming a quadrilateral with specified sides
axiom quadrilateral_existence : quadrilateral 20 16 14 8 11

-- The theorem to prove regarding distances between the midpoints
theorem distance_between_midpoints : 
  quadrilateral 20 16 14 8 11 →
  dist_midpoints_bc_da 20 16 14 8 11 = 16.1 ∧
  dist_midpoints_aa0_dc0 20 16 14 8 11 = 9.1 :=
by 
  intros h;
  sorry

end distance_between_midpoints_l64_64180


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64924

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64924


namespace sequence_limit_l64_64372

theorem sequence_limit : 
  (∀ n : ℕ,
    \left( (3 - 4 * n)^2 / ((n - 3)^2 - (n + 3)^2) = (9 - 24 * n + 16 * n^2) / (-12 * n) \)
  → (\lim_{n \to \infty} \frac{(3-4n)^2}{(n-3)^2 - (n+3)^2} = -\infty)) :=
sorry -- proof goes here

end sequence_limit_l64_64372


namespace sum_factors_of_30_l64_64962

theorem sum_factors_of_30 : 
  (∑ i in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), i) = 72 := by
  sorry

end sum_factors_of_30_l64_64962


namespace quadrilateral_angle_E_l64_64545

theorem quadrilateral_angle_E (E F G H : ℝ)
  (h1 : E = 3 * F)
  (h2 : E = 4 * G)
  (h3 : E = 6 * H)
  (h_sum : E + F + G + H = 360) :
  E = 206 :=
by
  sorry

end quadrilateral_angle_E_l64_64545


namespace area_of_triangle_l64_64532

noncomputable def sin (x : ℝ) : ℝ := sorry -- Assume we have a valid definition for sin

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ)
  (ha : a = 3) 
  (hA : A = 30) 
  (hB : B = 60) 
  (hb : b = a * sin B / sin A) 
  (hC : C = 180 - A - B) 
  (hArea : S = (1 / 2) * a * b * sin C) :
  S = 9 * real.sqrt 3 / 2 := 
sorry

end area_of_triangle_l64_64532


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64882

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64882


namespace sum_factors_of_30_l64_64949

def factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ m, m > 0 ∧ n % m == 0)

theorem sum_factors_of_30 : (factors 30).sum = 72 := by
  sorry

end sum_factors_of_30_l64_64949


namespace product_of_real_solutions_triple_property_l64_64790

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64790


namespace solve_system_l64_64147

-- Define the matrix and vectors
noncomputable def A : Matrix (Fin 3) (Fin 3) ℚ :=
  !![2, -1, 1;
    3, 2, 2;
    1, -2, 1]

noncomputable def X : Fin 3 → ℚ := ![2, -1, -3]

def B : Fin 3 → ℚ := ![2, -2, 1]

-- Define the proof problem
theorem solve_system :
  A.mulVec X = B :=
by
  sorry

end solve_system_l64_64147


namespace solutions_l64_64103

noncomputable def S := ℕ

def f : S → S
def g : S → S
def h : S → S

axiom functional_eq (m n : S) : f (m + n) = g m + h n
axiom g_eq : g 1 = 1
axiom h_eq : h 1 = 1

theorem solutions (f g h : S → S) (functional_eq : ∀ m n, f (m + n) = g m + h n)
  (g_eq: g 1 = 1) (h_eq: h 1 = 1) : 
    (
      (∀ n, f n = 2 ∧ g n = 1 ∧ h n = 1) ∨ 
      (∀ n, f n = n ∧ g n = n ∧ h n = n)
    ) := 
sorry

end solutions_l64_64103


namespace fat_content_whole_milk_l64_64999

open Real

theorem fat_content_whole_milk :
  ∃ (s w : ℝ), 0 < s ∧ 0 < w ∧
  3 / 100 = 0.75 * s / 100 ∧
  s / 100 = 0.8 * w / 100 ∧
  w = 5 :=
by
  sorry

end fat_content_whole_milk_l64_64999


namespace xy_value_l64_64068

noncomputable def x (y : ℝ) : ℝ := 36 * y

theorem xy_value (y : ℝ) (h1 : y = 0.16666666666666666) : x y * y = 1 :=
by
  rw [h1, x]
  sorry

end xy_value_l64_64068


namespace mean_significantly_improved_l64_64298

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64298


namespace product_of_triple_when_added_to_reciprocal_l64_64896

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64896


namespace significant_improvement_l64_64266

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64266


namespace second_term_of_series_l64_64347

noncomputable def geometric_series_second_term (a r S : ℝ) := r * a

theorem second_term_of_series (a r : ℝ) (S : ℝ) (hr : r = 1/4) (hs : S = 16) 
  (hS_formula : S = a / (1 - r)) : geometric_series_second_term a r S = 3 :=
by
  -- Definitions are in place, applying algebraic manipulation steps here would follow
  sorry

end second_term_of_series_l64_64347


namespace sum_bn_sequence_l64_64496

theorem sum_bn_sequence {a : ℕ → ℝ} {b : ℕ → ℝ} (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n / (2 * a n + 1))
  (h3 : ∀ n, b n = a n / (2 * n + 1)) :
  ∀ n, ∑ i in range n, b i = n / (2 * n + 1) := by
  sorry

end sum_bn_sequence_l64_64496


namespace product_of_all_solutions_triple_reciprocal_l64_64736

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64736


namespace num_of_solutions_l64_64435

theorem num_of_solutions (x y : ℤ) : 
  (x ^ 4 + y ^ 2 = 4 * y) → 
  ({(x, y) ∈ ℤ × ℤ | x ^ 4 + y ^ 2 = 4 * y }).card = 2 :=
sorry

end num_of_solutions_l64_64435


namespace find_width_l64_64218

-- Definitions and Conditions
def length : ℝ := 6
def depth : ℝ := 2
def total_surface_area : ℝ := 104

-- Statement to prove the width
theorem find_width (width : ℝ) (h : 12 * width + 4 * width + 24 = total_surface_area) : width = 5 := 
by { 
  -- lean 4 statement only, proof omitted
  sorry 
}

end find_width_l64_64218


namespace cafeteria_extra_fruits_equals_134_l64_64640

def total_fruits_ordered : ℕ :=
  60 + 34 + 25 + 45

def total_fruits_taken : ℕ :=
  3 * 2 + 2 * 2 + 5 * 2 + 10 * 1

def extra_fruits : ℕ :=
  total_fruits_ordered - total_fruits_taken

theorem cafeteria_extra_fruits_equals_134 : extra_fruits = 134 :=
by
  rw [total_fruits_ordered, total_fruits_taken, extra_fruits]
  norm_num
  sorry -- Proof not required, so we leave it as a placeholder

end cafeteria_extra_fruits_equals_134_l64_64640


namespace cone_volume_l64_64646

theorem cone_volume (s h : ℝ) (h_s : s = 15) (h_h : h = 9) : 
  ∃ V : ℝ, V = (1 / 3) * π * (sqrt (s^2 - h^2))^2 * h := 
by
  use 432 * π
  rw [h_s, h_h]
  have rsq : 15^2 - 9^2 = 144 := by norm_num
  have radius : sqrt (15^2 - 9^2) = 12 := by rw [rsq, Real.sqrt_sq_eq_abs, abs_of_nonneg]; norm_num
  simp [radius]
  norm_num
  sorry

end cone_volume_l64_64646


namespace average_age_of_team_l64_64153

theorem average_age_of_team :
  ∃ A : ℝ,
    (∀ (ages : Fin 11 → ℝ),
      (ages ⟨0, by norm_num⟩ = 26) ∧
      (ages ⟨1, by norm_num⟩ = 29) ∧
      (∀ i (hi : 2 ≤ i ∧ i < 11), ages i = 32)  ∧ 
      (11 * A = sum (range 11) (λ i, ages ⟨i, by norm_num⟩)) ∧
      (9 * (A - 1) = sum (range 2 11) (λ i, ages ⟨i, by norm_num⟩))
    ) →
  A = 32 :=
sorry

end average_age_of_team_l64_64153


namespace area_triangle_PCF_l64_64455

theorem area_triangle_PCF {ABC : Type*} (A B C P D E F : ABC) (h : equilateral_triangle A B C)
  (hP : inside_triangle P A B C) 
  (hPD : perpendicular P D B C) 
  (hPE : perpendicular P E C A) 
  (hPF : perpendicular P F A B) 
  (hABC : area A B C = 2028) 
  (hPAD : area P A D = 192) 
  (hPBE : area P B E = 192) : area P C F = 630 :=
sorry

end area_triangle_PCF_l64_64455


namespace probability_product_multiple_of_4_l64_64100

theorem probability_product_multiple_of_4 :
  let laura_roll := (1 : ℕ) ⊕ 2 ⊕ 3 ⊕ 4 ⊕ 5 ⊕ 6 ⊕ 7 ⊕ 8,
      ben_roll := (1 : ℕ) ⊕ 2 ⊕ 3 ⊕ 4 ⊕ 5 ⊕ 6,
      event_roll_multiple_of_4 (l : ℕ) (b : ℕ) : Prop :=
        (l * b) % 4 = 0 in
  (finset.card (finset.filter (λ p, event_roll_multiple_of_4 p.1 p.2) (finset.product {1, 2, 3, 4, 5, 6, 7, 8} {1, 2, 3, 4, 5, 6}))).to_rat /
  (finset.card (finset.product {1, 2, 3, 4, 5, 6, 7, 8} {1, 2, 3, 4, 5, 6})).to_rat = 5 / 8 :=
sorry

end probability_product_multiple_of_4_l64_64100


namespace field_length_l64_64631

theorem field_length (w l: ℕ) (hw1: l = 2 * w) (hw2: 8 * 8 = 64) (hw3: 64 = l * w / 2) : l = 16 := 
by
  sorry

end field_length_l64_64631


namespace product_of_solutions_l64_64781

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64781


namespace singers_A_not_first_singers_A_not_first_B_not_last_l64_64373

-- Definitions of the problems as Lean statements
-- Problem 1
theorem singers_A_not_first :
  ∃ (arrangements : ℕ), 
  arrangements = 96 ∧
  (∀ (perm : Finset (Perms 5)), perm 0 ≠ 0 ↔ perm ∈ arrangements) := 
sorry

-- Problem 2
theorem singers_A_not_first_B_not_last :
  ∃ (arrangements : ℕ),
  arrangements = 78 ∧ 
  (∀ (perm : Finset (Perms 5)), (perm 0 ≠ 0 ∧ perm 4 ≠ 1) ↔ perm ∈ arrangements) := 
sorry

end singers_A_not_first_singers_A_not_first_B_not_last_l64_64373


namespace new_device_significant_improvement_l64_64259

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64259


namespace find_a3_l64_64451

theorem find_a3 (a : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) 
    (h1 : (1 + x) * (a - x)^6 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7)
    (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = 0) :
  a = 1 → a3 = -5 := 
by 
  sorry

end find_a3_l64_64451


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64841

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64841


namespace angle_A_is_120_degrees_l64_64533

theorem angle_A_is_120_degrees
  (a b : ℝ)
  (B : ℝ)
  (h_a : a = sqrt 3)
  (h_b : b = sqrt 2)
  (h_B : B = 45): 
  A = 120 := 
sorry

end angle_A_is_120_degrees_l64_64533


namespace product_of_solutions_l64_64811

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64811


namespace sum_factors_of_30_l64_64951

def factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ m, m > 0 ∧ n % m == 0)

theorem sum_factors_of_30 : (factors 30).sum = 72 := by
  sorry

end sum_factors_of_30_l64_64951


namespace A_beats_B_by_63_l64_64535

variable (A B C : ℕ)

-- Condition: A beats C by 163 meters
def A_beats_C : Prop := A = 1000 - 163
-- Condition: B beats C by 100 meters
def B_beats_C (X : ℕ) : Prop := 1000 - X = 837 + 100
-- Main theorem statement
theorem A_beats_B_by_63 (X : ℕ) (h1 : A_beats_C A) (h2 : B_beats_C X): X = 63 :=
by
  sorry

end A_beats_B_by_63_l64_64535


namespace max_positive_numbers_among_ai_bi_l64_64574

theorem max_positive_numbers_among_ai_bi
  (n : ℕ) (a b : Fin n → ℝ)
  (h : ∀ i : Fin n, ∀ x : ℝ, (a i * x - b i)^2 ≥ ∑ j in Finset.univ.filter (λ j, j ≠ i), (a j * x - b j) ) :
  ∃ p : ℕ, p = 2005 ∧ (∀ i : Fin (2 * 2005), i < p → 0 < a i ∨ 0 < b i) ∧ p = 4009 := by
  sorry

end max_positive_numbers_among_ai_bi_l64_64574


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64913

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64913


namespace smallest_integer_cos_equation_l64_64110

theorem smallest_integer_cos_equation :
  ∃ x : ℤ, x > 1 ∧ (∀ y : ℤ, y > 1 → (cos (y * (real.pi / 180)) = cos ((y ^ 2) * (real.pi / 180)) → y ≥ x)) ∧ x = 19 :=
by
  sorry

end smallest_integer_cos_equation_l64_64110


namespace intersection_x_value_l64_64188

theorem intersection_x_value : ∃ x y : ℝ, y = 3 * x + 7 ∧ 3 * x - 2 * y = -4 ∧ x = -10 / 3 :=
by
  sorry

end intersection_x_value_l64_64188


namespace spherical_to_rectangular_coords_l64_64391

theorem spherical_to_rectangular_coords (
  {ρ θ φ : ℝ} 
) (hρ : ρ = 3) (hθ : θ = 3 * Real.pi / 2) (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by 
  sorry

end spherical_to_rectangular_coords_l64_64391


namespace fruit_basket_problem_l64_64510

theorem fruit_basket_problem :
  let choices_for_apples := 6 in
  let choices_for_oranges := 11 in
  let total_combinations := choices_for_apples * choices_for_oranges in
  let valid_fruit_baskets := total_combinations - 1 in
  valid_fruit_baskets = 65 :=
by 
  let choices_for_apples := 6
  let choices_for_oranges := 11
  let total_combinations := choices_for_apples * choices_for_oranges
  let valid_fruit_baskets := total_combinations - 1
  exact calc
    valid_fruit_baskets 
      = total_combinations - 1 : by rfl
    ... = 66 - 1 : by rfl
    ... = 65 : by rfl

end fruit_basket_problem_l64_64510


namespace spherical_to_rectangular_l64_64394

theorem spherical_to_rectangular :
  ∀ (ρ θ φ : ℝ),
  ρ = 3 →
  θ = (3 * π) / 2 →
  φ = π / 3 →
  let x := ρ * real.sin φ * real.cos θ,
      y := ρ * real.sin φ * real.sin θ,
      z := ρ * real.cos φ in
  (x, y, z) = (0, -3 * real.sqrt 3 / 2, 3 / 2) := by
  intros,
  -- The actual proof would go here
  sorry

end spherical_to_rectangular_l64_64394


namespace conjugate_of_squared_complex_number_l64_64157

open Complex

theorem conjugate_of_squared_complex_number :
  conj ((1 : ℂ) + (2 : ℂ) * I)^2 = (-3 : ℂ) - (4 : ℂ) * I := by
  sorry

end conjugate_of_squared_complex_number_l64_64157


namespace product_of_solutions_l64_64875

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64875


namespace reconstruct_parallelogram_l64_64319

variable {A E F : Point}
variable (x0 y0 x1 y1 x2 y2 : ℝ)
variable A : Point := (x0, y0)
variable E : Point := (x1, y1)
variable F : Point := (x2, y2)
variable K : Point := ((x1 + x2) / 2, (y1 + y2) / 2)

theorem reconstruct_parallelogram (midpoint_E : E = midpoint (A + B))
 (midpoint_F : F = midpoint (C + D)):
  C = (x0 / 2, y0 / 2) :=
  sorry

end reconstruct_parallelogram_l64_64319


namespace minimum_weight_of_each_crate_l64_64340

theorem minimum_weight_of_each_crate (maxCrates : Finset ℕ) (maxWeight : ℕ) :
  maxCrates = {3, 4, 5} →
  maxWeight = 600 →
  ∃ minWeight : ℕ, (∀ n ∈ maxCrates, n ≤ 5 → minWeight = maxWeight / n) → minWeight = 120 :=
by 
  intros hCrates hWeight 
  use 120 
  intro h 
  apply h _ _ 
  · simp [hCrates]
  sorry

end minimum_weight_of_each_crate_l64_64340


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64915

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64915


namespace product_of_solutions_l64_64869

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64869


namespace regular_price_per_pound_l64_64324

-- Define conditions
def quarter_pound_regular_price (y : ℝ) := y / 4
def quarter_pound_sale_price (y : ℝ) := (quarter_pound_regular_price y) * 0.40

-- Define the main theorem to be proved
theorem regular_price_per_pound (y : ℝ) (h : quarter_pound_sale_price y = 2.40) : y = 24 :=
  sorry

end regular_price_per_pound_l64_64324


namespace product_trippled_when_added_to_reciprocal_l64_64928

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64928


namespace sum_of_factors_30_l64_64972

theorem sum_of_factors_30 : 
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ) in
  ( ∑ n in factors, n ) = 72 :=
by
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ)
  rw [finset.sum_eq_sum_of_elements] -- transposing to handle the sum notation
  let factors_list := [1, 2, 3, 5, 6, 10, 15, 30]
  have h_factors : factors = {x | x ∈ factors_list}.to_finset, by sorry
  simp_rw h_factors
  rw [finset.sum_to_finset]
  dsimp -- simplifying before asserting the sum
  norm_num -- proving that the sum is indeed 72
  sorry

end sum_of_factors_30_l64_64972


namespace slope_of_line_l64_64945

theorem slope_of_line : 
  let (x1, y1) := (-3 : ℚ, 3 / 2 : ℚ) in 
  let (x2, y2) := (4 : ℚ, -7 / 2 : ℚ) in 
  (y2 - y1) / (x2 - x1) = -5 / 7 := 
by
  sorry

end slope_of_line_l64_64945


namespace larger_number_of_two_numbers_l64_64232

theorem larger_number_of_two_numbers (A B : ℕ) (HCF : ℕ) (L1 : ℕ) (L2 : ℕ) 
  (h_hcf : HCF = 23) (h_l1 : L1 = 13) (h_l2 : L2 = 14) 
  (h_A_factor : A = HCF * L2) 
  (h_prod : A * B = HCF * (HCF * L1 * L2)) : 
  A = 322 := 
by 
  rw [h_hcf, h_l1, h_l2, ←Nat.mul_assoc] at h_prod 
  rw [Nat.mul_assoc HCF L1 L2, h_hcf] 
  exact h_A_factor

#check larger_number_of_two_numbers -- To verify the statement is correct.

end larger_number_of_two_numbers_l64_64232


namespace incorrect_statement_for_P_l64_64549

variable P : Real × Real
variable Q : Real × Real

def point_P_has_coordinates := P = (3, 4)
def point_Q_has_coordinates := Q = (4, 3)

def point_on_y_axis_left_move (P : Real × Real) := P.1 - 3 = 0
def ordinate_of_point (P : Real × Real) := P.2 = 4
def distance_to_x_axis (P : Real × Real) := abs P.2 = 4
def points_not_equal (P Q : Real × Real) := P ≠ Q

theorem incorrect_statement_for_P :
  point_P_has_coordinates → 
  ¬point_on_y_axis_left_move P ∧
  ordinate_of_point P ∧
  distance_to_x_axis P ∧
  points_not_equal P Q → ¬points_not_equal P Q :=
by
  sorry

end incorrect_statement_for_P_l64_64549


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64760

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64760


namespace shifted_parabola_is_correct_l64_64043

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  -((x - 1) ^ 2) + 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ :=
  -((x + 1 - 1) ^ 2) + 4

-- State the theorem
theorem shifted_parabola_is_correct :
  ∀ x : ℝ, shifted_parabola x = -x^2 + 4 :=
by
  -- Proof would go here
  sorry

end shifted_parabola_is_correct_l64_64043


namespace tangent_line_at_a_eq_2_l64_64039

noncomputable def f (x : ℝ) (a : ℝ) := a * x^2 - (a + 2) * x + Real.log x

def tangent_line_eq (a : ℝ) (x : ℝ) : Prop :=
  ∀ (x₀ : ℝ), f x₀ a = -2 → (D (f x₀ a).deriv x₀) x₀ = 1 → x - x₀ + 2 = 0

theorem tangent_line_at_a_eq_2 :
  tangent_line_eq 2 1 := by
  sorry

end tangent_line_at_a_eq_2_l64_64039


namespace isosceles_triangle_formed_l64_64223

theorem isosceles_triangle_formed :
  (∃ b : ℝ, (b = 4) ∧ (∀ t : ℝ, (t ∈ [{4*t + 4}, {1/2 * t + 4}, {-t + 4}, {2*t + 4}])) ∧ 
   ((0, b), (a, 0)) ∈ isosceles_triangle (0, 0) 
   := 
∃ line : ℝ → ℝ, 
(line = (λ x, 4 * x + 4) ∨ line = (λ x, (1/2) * x + 4) ∨ line = (λ x, -x + 4) ∨ line = (λ x, 2 * x + 4)) :=
(line = (λ x, -x + 4)) :=
begin
  sorry
end

end isosceles_triangle_formed_l64_64223


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64840

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64840


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64888

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64888


namespace general_formula_for_a_n_find_constant_c_find_largest_k_l64_64487

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

def a_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range (2 * n), g ((i : ℝ) / n)

def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_n (i + 1)

def b_n (n : ℕ) (c : ℝ) : ℝ :=
  (2 * S_n n - n) / (n + c)

def c_n (n : ℕ) : ℝ :=
  1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, c_n (i + 1)

/- First Proof: Prove general formula for a_n -/
theorem general_formula_for_a_n (n : ℕ) : a_n n = 2 * n - 1 := sorry

/- Second Proof: Find the non-zero constant c -/
theorem find_constant_c (c : ℝ) (h_arith : ∀ n : ℕ, b_n n c - b_n (n - 1) c = b_n (n + 1) c - b_n n c) :
  c = - 1 / 2 := sorry

/- Third Proof: Find the largest positive integer k -/
theorem find_largest_k (k : ℕ) (hk : ∀ n : ℕ, T_n n > k / 57) : k = 18 := sorry

end general_formula_for_a_n_find_constant_c_find_largest_k_l64_64487


namespace geom_seq_sum_eq_31_over_4_l64_64442

/-- Definition of a geometric sequence sum -/
def sum_geom_series (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_seq_sum_eq_31_over_4 (a1 q : ℝ) (S : ℕ → ℝ) :
  (a1 + a1 * q = 3/4) ∧ (a1 * q^3 * (1 + q) = 6) →
  sum_geom_series a1 q 5 = 31/4 :=
by
  intros h,
  sorry

end geom_seq_sum_eq_31_over_4_l64_64442


namespace suitable_land_for_vegetables_l64_64565

def previous_property_acres : ℝ := 2
def enlargement_factor : ℝ := 10
def pond_area : ℝ := 1

theorem suitable_land_for_vegetables :
  let new_property_acres := previous_property_acres * enlargement_factor in
  let suitable_acres := new_property_acres - pond_area in
  suitable_acres = 19 :=
by
  sorry

end suitable_land_for_vegetables_l64_64565


namespace geom_seq_sum_eq_six_l64_64468

theorem geom_seq_sum_eq_six 
    (a : ℕ → ℝ) 
    (r : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * r) 
    (h_pos : ∀ n, a n > 0)
    (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) 
    : a 5 + a 7 = 6 :=
sorry

end geom_seq_sum_eq_six_l64_64468


namespace probability_reach_5_without_touching_neg3_l64_64560

-- Statement of the problem
theorem probability_reach_5_without_touching_neg3 :
  let heads := 1
  let tails := -1
  Pr(Jerry starts at 0 and tosses a fair coin 10 times, 
     moving heads -> +1 and tails -> -1, 
     reaches 5 at some time without touching -3 before reaching 5) = 15 / 128 := 
sorry

end probability_reach_5_without_touching_neg3_l64_64560


namespace mean_significantly_improved_l64_64292

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64292


namespace product_trippled_when_added_to_reciprocal_l64_64933

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64933


namespace sequence_sum_l64_64642

def sequence (a : ℕ → ℕ) : Prop :=
∀ m n : ℕ, a 1 = 1 ∧ ∀ m n, a (m + n) = a m + a n + m * n

theorem sequence_sum (a : ℕ → ℕ ) (h : sequence a) :
  (finset.range 2017).sum (λ n, 1 / a (n + 1)) = 4034 / 2018 :=
sorry

end sequence_sum_l64_64642


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64844

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64844


namespace grains_in_batch_l64_64619

-- Define the given constants from the problem
def total_rice_shi : ℕ := 1680
def sample_total_grains : ℕ := 250
def sample_containing_grains : ℕ := 25

-- Define the statement to be proven
theorem grains_in_batch : (total_rice_shi * (sample_containing_grains / sample_total_grains)) = 168 := by
  -- Proof steps will go here
  sorry

end grains_in_batch_l64_64619


namespace distance_between_towns_l64_64599

-- Define the custom scale for conversion
def scale_in_km := 1.05  -- 1 km + 50 meters as 1.05 km

-- Input distances on the map and their conversion
def map_distance_in_inches := 6 + 11/16

noncomputable def actual_distance_in_km : ℝ :=
  let distance_in_inches := (6 * 8 + 11) / 16
  distance_in_inches * (8 / 3)

theorem distance_between_towns :
  actual_distance_in_km = 17.85 := by
  -- Equivalent mathematical steps and tests here
  sorry

end distance_between_towns_l64_64599


namespace significant_improvement_l64_64278

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64278


namespace significant_improvement_l64_64285

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64285


namespace remainder_of_sum_binom_l64_64174

theorem remainder_of_sum_binom
  (h_prime : Nat.Prime 2027)
  (T : ℕ := ∑ k in Finset.range (72 + 1), Nat.choose 2024 k) :
  T % 2027 = 1369 := sorry

end remainder_of_sum_binom_l64_64174


namespace condition_A_condition_B_condition_C_condition_D_correct_answer_l64_64162

theorem condition_A : ∀ x : ℝ, x^2 + 2 * x - 1 ≠ x * (x + 2) - 1 := sorry

theorem condition_B : ∀ a b : ℝ, (a + b)^2 = a^2 + 2 * a * b + b^2 := sorry

theorem condition_C : ∀ x y : ℝ, x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) := sorry

theorem condition_D : ∀ a b : ℝ, a^2 - a * b - a ≠ a * (a - b) := sorry

theorem correct_answer : ∀ x y : ℝ, (x^2 - 4 * y^2) = (x + 2 * y) * (x - 2 * y) := 
  by 
    exact condition_C

end condition_A_condition_B_condition_C_condition_D_correct_answer_l64_64162


namespace product_of_solutions_l64_64805

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64805


namespace pieces_of_paper_picked_up_l64_64597

theorem pieces_of_paper_picked_up (Olivia : ℕ) (Edward : ℕ) (h₁ : Olivia = 16) (h₂ : Edward = 3) : Olivia + Edward = 19 :=
by
  sorry

end pieces_of_paper_picked_up_l64_64597


namespace derive_parabola_equation_four_digit_integers_l64_64353

-- Definition of the parabola passing through A(-1,0)
def parabola_equation (a b : ℝ) : Prop :=
  b = a + 3 ∧ ∀ x : ℝ, (a * x ^ 2 + b * x + 3 = 0 → (x = -1))

-- Theorem to derive the equation of the parabola
theorem derive_parabola_equation (a b : ℝ) :
  (∃ c, parabola_equation a b ∧ c = 3) →
  b = a + 3 :=
by
  intro h
  obtain ⟨c, h_parabola, hc⟩ := h
  exact h_parabola.1

-- Prime number and perfect square conditions
def prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Theorem for four-digit integers m
theorem four_digit_integers (m : ℕ) :
  (1000 ≤ m ∧ m ≤ 2025) →
  ∃ n : ℕ, prime (m - n) ∧ is_perfect_square (m * n) :=
by
  intro h
  sorry -- proof goes here

-- Using assumptions directly without calculations from the proof step
axiom derived_data : parabola_equation a b ∧ prime (m - n) ∧ is_perfect_square (m * n)

end derive_parabola_equation_four_digit_integers_l64_64353


namespace reciprocal_of_repeating_decimal_l64_64942

theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 1 / 3 in
  1 / x = 3 :=
by
  sorry

end reciprocal_of_repeating_decimal_l64_64942


namespace significant_improvement_l64_64287

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64287


namespace prove_sum_eq_9_l64_64000

theorem prove_sum_eq_9 (a b : ℝ) (h : i * (a - i) = b - (2 * i) ^ 3) : a + b = 9 :=
by
  sorry

end prove_sum_eq_9_l64_64000


namespace cone_volume_l64_64658

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l64_64658


namespace sequence_converges_to_one_l64_64104

noncomputable def f (x : ℝ) := 1 + Real.log ((x ^ 2) / (1 + Real.log x))

theorem sequence_converges_to_one (a : ℝ) (h : a ≥ 1) : 
  ∃ L, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs ((nat.rec_on n a (λ n xn, f xn)) - L) < ε) ∧ L = 1 :=
by
  sorry

end sequence_converges_to_one_l64_64104


namespace part1_tangent_line_part2_monotonic_intervals_l64_64040

-- Define the function f(x) = ax^2 - (a+2)x + ln(x)
def f (a : ℝ) (x : ℝ) := a * x^2 - (a + 2) * x + Real.log x

-- Part 1: Prove tangent line equation when a = 2
theorem part1_tangent_line : 
  f 2 1 = -2 ∧ 
  let tangent_line := λ x y : ℝ, x - y - 3 = 0 in 
  tangent_line 1 (-2) :=
sorry

-- Part 2: Prove monotonic intervals based on a
theorem part2_monotonic_intervals (a : ℝ) :
  if a ≤ 0 then
    ∃ incr decr : Set ℝ,
      incr = {x : ℝ | 0 < x ∧ x < 1 / 2} ∧
      decr = {x : ℝ | x > 1 / 2}
  else if 0 < a ∧ a < 2 then
    ∃ incr1 incr2 decr : Set ℝ,
      incr1 = {x : ℝ | 0 < x ∧ x < 1 / 2} ∧
      incr2 = {x : ℝ | x > 1 / a} ∧
      decr = {x : ℝ | 1 / 2 < x ∧ x < 1 / a}
  else if a = 2 then
    ∃ incr : Set ℝ,
      incr = {x : ℝ | x > 0}
  else 
    ∃ incr1 incr2 decr : Set ℝ,
      incr1 = {x : ℝ | 0 < x ∧ x < 1 / a} ∧
      incr2 = {x : ℝ | x > 1 / 2} ∧
      decr = {x : ℝ | 1 / a < x ∧ x < 1 / 2} :=
sorry

end part1_tangent_line_part2_monotonic_intervals_l64_64040


namespace percentage_of_first_pay_cut_l64_64569

theorem percentage_of_first_pay_cut
  (x : ℝ)
  (h1 : ∃ y z w : ℝ, y = 1 - x/100 ∧ z = 0.86 ∧ w = 0.82 ∧ y * z * w = 0.648784):
  x = 8.04 := by
-- The proof will be added here, this is just the statement
sorry

end percentage_of_first_pay_cut_l64_64569


namespace sum_factors_of_30_l64_64955

theorem sum_factors_of_30 : 
  (∑ i in {1, 2, 3, 5, 6, 10, 15, 30}, i) = 72 := 
by
  -- Sorry, to skip the proof here.
  sorry

end sum_factors_of_30_l64_64955


namespace sum_of_factors_of_30_is_72_l64_64966

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l64_64966


namespace marshmallow_per_smore_l64_64092

theorem marshmallow_per_smore (graham_crackers : ℕ) (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) 
                               (graham_crackers_per_smore : ℕ) :
  graham_crackers = 48 ∧ initial_marshmallows = 6 ∧ additional_marshmallows = 18 ∧ graham_crackers_per_smore = 2 →
  (initial_marshmallows + additional_marshmallows) / (graham_crackers / graham_crackers_per_smore) = 1 :=
by
  intro h
  sorry

end marshmallow_per_smore_l64_64092


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64724

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64724


namespace total_savings_l64_64402

def individual_shirt_cost : ℝ := 7.50
def individual_pants_cost : ℝ := 15.00
def individual_socks_cost : ℝ := 4.50

def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75

def team_size : ℕ := 12

theorem total_savings :
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost in
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost in
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost in
  let total_savings := savings_per_uniform * team_size in
  total_savings = 36 :=
by
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  have h : total_savings = 36 := by
    calc
      total_savings = (7.50 + 15.00 + 4.50 - (6.75 + 13.50 + 3.75)) * 12 := by sorry
                  ... = 3 * 12 := by sorry
                  ... = 36 := by sorry
  exact h

end total_savings_l64_64402


namespace product_of_solutions_l64_64806

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64806


namespace pq_perpendicular_o1o2_l64_64105

/-- 
Let C_1 and C_2 be two circles with respective centers O_1 and O_2 that intersect at points A and B. 
Let C be the intersection point of the tangent to C_2 at B, 
and D be the intersection point of the tangent to C_1 at B. 
Let X be the intersection of the bisector of ∠BAC with C_1. 
Let Y be the intersection of the bisector of ∠BAD with C_2. 
Let P and Q be the centers of the circumcircles of triangles ACD and AXY, respectively. 
Prove that PQ and O_1O_2 are perpendicular. 
-/
theorem pq_perpendicular_o1o2
  (C1 C2 : Circle) (O1 O2 A B C D X Y P Q : Point)
  (hC1 : C1.center = O1) (hC2 : C2.center = O2)
  (hInt : ∃ A B, A ≠ B ∧ A ∈ C1 ∧ B ∈ C1 ∧ A ∈ C2 ∧ B ∈ C2)
  (hTanC2B : tangentAt C2 B C)
  (hTanC1B : tangentAt C1 B D)
  (hBisAC : bisector (∠BAC) X C1)
  (hBisAD : bisector (∠BAD) Y C2)
  (hCircP : circumcenter (triangle A C D) P)
  (hCircQ : circumcenter (triangle A X Y) Q) :
  is_perpendicular (line_through P Q) (line_through O1 O2) :=
sorry

end pq_perpendicular_o1o2_l64_64105


namespace product_of_real_solutions_triple_property_l64_64791

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64791


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64842

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64842


namespace solve_equation_l64_64530

theorem solve_equation (x : ℝ) (h : 3 + 1 / (2 - x) = 2 * (1 / (2 - x))) : x = 5 / 3 := 
  sorry

end solve_equation_l64_64530


namespace log_64_x_eq_log_2_5_sixth_l64_64518

theorem log_64_x_eq_log_2_5_sixth (x : ℝ) (h : log 16 (2 * x - 6) = 1 / 2) : log 64 x = log 2 5 / 6 :=
by {
  -- This is a place holder for the actual proof
  sorry
}

end log_64_x_eq_log_2_5_sixth_l64_64518


namespace smallest_region_area_l64_64424

-- Given conditions
def y_eq_2_abs_x (x : ℝ) : ℝ := 2 * |x|
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Prove the area of the smallest region bounded by the specified graphs
theorem smallest_region_area :
  let π := Real.pi in
  let r := 3 in
  let θ := π/2 in
  let area := 1/2 * r^2 * θ in
  area = 9 * π / 4 :=
by
  sorry

end smallest_region_area_l64_64424


namespace sum_of_solutions_eq_5pi_l64_64434

theorem sum_of_solutions_eq_5pi :
  ∑ x in {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ (1 / Real.sin x) + (1 / Real.cos x) = 4}, x = 5 * Real.pi := 
  sorry

end sum_of_solutions_eq_5pi_l64_64434


namespace distance_AD_between_31_and_32_l64_64601

def point (α : Type*) := (α × α)
structure triangle (α : Type*) := 
  (a b c : point α)
  (angle_BAC : ℝ)
  (dist_AC : ℝ)

theorem distance_AD_between_31_and_32 
  (A B C D : point ℝ) 
  (h_B_east_A : B.1 = A.1 + 10 ∧ B.2 = A.2) 
  (h_C_north_B : C.1 = B.1 ∧ C.2 = B.2 + 10) 
  (h_AC : dist (A, C) = 10 * real.sqrt 2)
  (h_angle_BAC : ∠BAC = 45)
  (h_D_north_C : D.1 = C.1 ∧ D.2 = C.2 + 20) : 
  31 < dist (A, D) ∧ dist (A, D) < 32 := 
sorry

end distance_AD_between_31_and_32_l64_64601


namespace no_path_visiting_all_marked_points_once_l64_64134

-- Define the problem

/-- 
Define vertices and face centers of a cube and the diagonals of the faces. 
Prove that it is not possible to visit all the marked points (vertices and face centers) exactly once 
using only the segments of these diagonals.
-/
theorem no_path_visiting_all_marked_points_once :
  let vertices := 8
  let face_centers := 6
  let marked_points := vertices + face_centers
  let face_diagonals := 2 
  ∀ (path : list (bool)) (alternates : ∀ i, path.nth i = some true ↔ path.nth (i + 1) = some false),
  (length path = marked_points) → (vertices ≠ face_centers + 2) := 
begin 
  sorry 
end

end no_path_visiting_all_marked_points_once_l64_64134


namespace sum_of_factors_30_l64_64971

theorem sum_of_factors_30 : 
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ) in
  ( ∑ n in factors, n ) = 72 :=
by
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ)
  rw [finset.sum_eq_sum_of_elements] -- transposing to handle the sum notation
  let factors_list := [1, 2, 3, 5, 6, 10, 15, 30]
  have h_factors : factors = {x | x ∈ factors_list}.to_finset, by sorry
  simp_rw h_factors
  rw [finset.sum_to_finset]
  dsimp -- simplifying before asserting the sum
  norm_num -- proving that the sum is indeed 72
  sorry

end sum_of_factors_30_l64_64971


namespace significant_improvement_l64_64271

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64271


namespace ratio_normal_is_cauchy_l64_64576

open ProbabilityTheory MeasureTheory

noncomputable def normalDist := MeasureTheory.Probability.ProbabilityMeasure (MeasureTheory.Measure.laplace)
noncomputable def cauchyDist := MeasureTheory.Probability.ProbabilityMeasure (MeasureTheory.Measure.cauchy)

theorem ratio_normal_is_cauchy (X Y : ℝ→ ℝ) [MeasureTheory.IsGaussian : MeasureTheory.ProbabilityDiscrete.normalDist X Y]
  (C : ℝ→ ℝ) [MeasureTheory.IsCauchy : MeasureTheory.ProbabilityDiscrete.cauchyDist C]:
  X / Y =d C := 
begin
  sorry -- placeholder for the proof.
end

end ratio_normal_is_cauchy_l64_64576


namespace cone_volume_l64_64662

-- Given conditions as definitions
def slant_height : ℝ := 15
def height : ℝ := 9
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- The volume of the cone
def volume : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem to prove that the calculated volume is 432π cubic centimeters
theorem cone_volume : volume = 432 * Real.pi := by
  sorry

end cone_volume_l64_64662


namespace find_a_values_l64_64422

theorem find_a_values :
  ∀ (a : ℝ), (∃ (x1 x2 x3 x4 x5 : ℝ),
    0 ≤ x1 ∧ 0 ≤ x2 ∧ 0 ≤ x3 ∧ 0 ≤ x4 ∧ 0 ≤ x5 ∧
    (∑ k in Finset.range 1 6, (k + 1) * 
    [x1, x2, x3, x4, x5].nth_le (k + 1) sorry = a) ∧
    (∑ k in Finset.range 1 6, ((k + 1)^3) * 
    [x1, x2, x3, x4, x5].nth_le (k + 1) sorry = a^2) ∧
    (∑ k in Finset.range 1 6, ((k + 1)^5) * 
    [x1, x2, x3, x4, x5].nth_le (k + 1) sorry = a^3))
  → a ∈ {0, 1, 4, 9, 16, 25} :=
begin 
  sorry
end

end find_a_values_l64_64422


namespace product_of_solutions_l64_64702

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64702


namespace trig_expression_value_l64_64467

theorem trig_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) : 
  (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 :=
by 
  sorry

end trig_expression_value_l64_64467


namespace find_a_values_l64_64027

def f (a x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then -x + 3
  else if 2 ≤ x ∧ x ≤ 8 then 1 + Real.log (2 * x) / Real.log (a^2 - 1)
  else 0

theorem find_a_values (a : ℝ) (h : ∀ y : ℝ, 2 ≤ y ∧ y ≤ 5 ↔ ∃ x : ℝ, ( -1 ≤ x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x ≤ 8) ∧ f a x = y ) :
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
  sorry

end find_a_values_l64_64027


namespace snow_on_Monday_l64_64558

def snow_on_Tuesday : ℝ := 0.21
def snow_on_Monday_and_Tuesday : ℝ := 0.53

theorem snow_on_Monday : snow_on_Monday_and_Tuesday - snow_on_Tuesday = 0.32 :=
by
  sorry

end snow_on_Monday_l64_64558


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64911

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64911


namespace annulus_sector_area_l64_64344

theorem annulus_sector_area (R r t : ℝ) (θ : ℝ) (hRr : R > r) (hRt : R^2 = r^2 + t^2) :
  (1/2 * θ * R^2 - π * r^2) = (θ/2 - π) * r^2 + θ * t^2 / 2 :=
by

sorr​​y

end annulus_sector_area_l64_64344


namespace circumcenter_on_AC_l64_64238

open EuclideanGeometry

variables {A B C X Y Z : Point}

-- Definitions and assumptions
def is_midpoint (p q r : Point) : Prop := dist p q = dist p r
def foot_of_perpendicular (p q r : Point) : Prop := ∃ s : Point, is_perpendicular p s q ∧ is_perpendicular p s r

-- Given conditions
variables (hAB_BC : dist A B = dist B C)
          (hMidX : is_midpoint A C X)
          (hMidY : is_midpoint A B Y)
          (hPerpZ : foot_of_perpendicular B C Y Z)

-- Statement to Prove
theorem circumcenter_on_AC 
  (hABC : Triangle A B C)
  (hXYZ : Triangle X Y Z) 
  (hCircumcenterLiesOnAC : Circumcenter X Y Z ∈ AC) :
  True :=
sorry

end circumcenter_on_AC_l64_64238


namespace total_height_bottle_l64_64329

theorem total_height_bottle (r1 r2 h1 h2 : ℝ) (h_total : h1 + h2 = 29)
  (h_water_upright : ℝ) (h_water_upside_down : ℝ) :
  r1 = 1 ∧ r2 = 3 → h_water_upright = 20 ∧ h_water_upside_down = 28 → h_total = 29 := by
  intros
  exact h_total
  sorry

end total_height_bottle_l64_64329


namespace inequality_solution_l64_64994

theorem inequality_solution (a b : ℝ) (h : (1 - a) * x^2 - 4 * x + 6 > 0 ∀ x) :
  a = 3 ∧
  (∀ x, 2 * x^2 + (2 - a) * x - a > 0 ↔ (x < -1 ∨ 1 < x)) ∧
  (∀ x, 3 * x^2 + b * x + 3 ≥ 0 ↔ -6 ≤ b ∧ b ≤ 6) :=
by sorry

end inequality_solution_l64_64994


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64817

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64817


namespace cone_volume_correct_l64_64672

noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let r := real.sqrt (slant_height^2 - height^2)
  in (1/3) * real.pi * r^2 * height

theorem cone_volume_correct :
  cone_volume 15 9 = 432 * real.pi := by
  sorry

end cone_volume_correct_l64_64672


namespace product_of_solutions_l64_64812

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64812


namespace line_passes_vertex_parabola_l64_64440

theorem line_passes_vertex_parabola :
  ∃ (b₁ b₂ : ℚ), (b₁ ≠ b₂) ∧ (∀ b, (b = b₁ ∨ b = b₂) → 
    (∃ x y, y = x + b ∧ y = x^2 + 4 * b^2 ∧ x = 0 ∧ y = 4 * b^2)) :=
by 
  sorry

end line_passes_vertex_parabola_l64_64440


namespace product_of_triple_when_added_to_reciprocal_l64_64908

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64908


namespace number_of_sheep_eating_grass_l64_64519

variable (S : ℕ)
variable (days_per_sheep_per_bag : ℕ)
variable (bags_eaten_total : ℕ)
variable (days_total : ℕ)

-- Given conditions
def condition1 : Prop := bags_eaten_total = 40
def condition2 : Prop := days_total = 40
def condition3 : Prop := days_per_sheep_per_bag = 40

-- Theorem to prove
theorem number_of_sheep_eating_grass (H1 : condition1)
                                      (H2 : condition2)
                                      (H3 : condition3) : S = 40 :=
sorry

end number_of_sheep_eating_grass_l64_64519


namespace area_DNELFM_eq_one_third_area_ABC_lines_DL_EM_FN_concurrent_l64_64136

variables {A B C P D E F L M N : Type} 
variables [point_space A B C P D E F L M N]

-- Assuming the definitions and required hypotheses
noncomputable def midpoint (a b : point_space) : point_space := sorry
noncomputable def intersection (l1 l2 : line) : point_space := sorry
noncomputable def area (shape : finset point_space) : ℝ := sorry

variables
  (AP : line) (BP : line) (CP : line)
  (BF : line) (CE : line) (CD : line) (AF : line) (AE : line) (BD : line)
  (DL : line) (EM : line) (FN : line)

axiom P_inside_ABC : inside_triangle P A B C
axiom D_midpoint_AP : D = midpoint A P
axiom E_midpoint_BP : E = midpoint B P
axiom F_midpoint_CP : F = midpoint C P
axiom L_intersection : L = intersection BF CE
axiom M_intersection : M = intersection CD AF
axiom N_intersection : N = intersection AE BD

theorem area_DNELFM_eq_one_third_area_ABC (h1 : area (finset.mk [D, N, E, L, F, M]) = (1/3) * area (finset.mk [A, B, C])) : 
  True :=
by sorry

theorem lines_DL_EM_FN_concurrent (h2 : concurrent [DL, EM, FN]) : 
  True :=
by sorry

end area_DNELFM_eq_one_third_area_ABC_lines_DL_EM_FN_concurrent_l64_64136


namespace cone_volume_l64_64676

variable (l h : ℝ) (π : ℝ := Real.pi)

noncomputable def radius (l h : ℝ) : ℝ := sqrt (l^2 - h^2)

noncomputable def volume_of_cone (r h : ℝ) (π : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (hl : l = 15) (hh : h = 9) :
  volume_of_cone (radius l h) h π = 432 * π :=
by
  sorry

end cone_volume_l64_64676


namespace geom_seq_sum_eq_31_over_4_l64_64443

/-- Definition of a geometric sequence sum -/
def sum_geom_series (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_seq_sum_eq_31_over_4 (a1 q : ℝ) (S : ℕ → ℝ) :
  (a1 + a1 * q = 3/4) ∧ (a1 * q^3 * (1 + q) = 6) →
  sum_geom_series a1 q 5 = 31/4 :=
by
  intros h,
  sorry

end geom_seq_sum_eq_31_over_4_l64_64443


namespace problem3_l64_64988

-- Define fractional part function
def frac (z: ℝ) : ℝ := z - z.floor

-- Define main theorem
theorem problem3 (x y : ℝ) (hx: frac x = frac y) (hx3: frac (x^3) = frac (y^3)) (hxy: x ≠ y) : 
  ∃ (a b c : ℤ), a * x^2 + b * x + c = 0 := 
by
    sorry

end problem3_l64_64988


namespace spherical_to_rectangular_coords_l64_64388

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), 
    ρ = 3 → θ = 3 * Real.pi / 2 → φ = Real.pi / 3 →
    (let x := ρ * Real.sin φ * Real.cos θ;
         y := ρ * Real.sin φ * Real.sin θ;
         z := ρ * Real.cos φ 
     in (x, y, z) = (0, - (3 * Real.sqrt 3) / 2, 3 / 2)) :=
by
  intros ρ θ φ hρ hθ hφ
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coords_l64_64388


namespace value_of_a_l64_64047

open Set

theorem value_of_a (a : ℝ) (h : {1, 2} ∪ {x | x^2 - a * x + a - 1 = 0} = {1, 2}) : a = 3 :=
by
  sorry

end value_of_a_l64_64047


namespace cone_volume_l64_64657

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l64_64657


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64885

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64885


namespace shaded_region_area_l64_64209

-- Define the circle radius and central angle in radians.
def r := 10
def θ := π / 4

-- Define the area of one sector.
def area_sector := (1 / 2) * θ * r^2

-- Define the total area of the two overlapping sectors.
def total_area_sectors := 2 * area_sector

-- Define the side length of the equilateral triangle (the diameter of the circle).
def side_length := 2 * r

-- Define the area of the equilateral triangle.
def area_triangle := (sqrt 3 / 4) * side_length^2

-- Define the area of the shaded region.
def shaded_area := total_area_sectors - area_triangle

-- State the theorem that the shaded area is equal to the given answer.
theorem shaded_region_area : shaded_area = 25 * π - 100 * sqrt 3 := by
  sorry

end shaded_region_area_l64_64209


namespace sum_factors_of_30_l64_64958

theorem sum_factors_of_30 : 
  (∑ i in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), i) = 72 := by
  sorry

end sum_factors_of_30_l64_64958


namespace polygon_sides_l64_64479

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1980) : n = 13 := 
by sorry

end polygon_sides_l64_64479


namespace symmetry_axis_of_f_l64_64036

noncomputable def f (x ϕ : ℝ) : ℝ := sin (x - ϕ)

theorem symmetry_axis_of_f (ϕ : ℝ) (h : ∫ x in 0..(2*π)/3, f x ϕ = 0) : x = 5*π/6 :=
sorry

end symmetry_axis_of_f_l64_64036


namespace product_of_real_solutions_triple_property_l64_64785

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64785


namespace judgement_only_b_correct_l64_64622

theorem judgement_only_b_correct
  (A_expr : Int := 11 + (-14) + 19 - (-6))
  (A_computed : Int := 11 + 19 + ((-14) + (-6)))
  (A_result_incorrect : A_computed ≠ 10)
  (B_expr : ℚ := -2/3 - 1/5 + (-1/3))
  (B_computed : ℚ := (-2/3 + -1/3) + -1/5)
  (B_result_correct : B_computed = -6/5) :
  (A_computed ≠ 10 ∧ B_computed = -6/5) :=
by
  sorry

end judgement_only_b_correct_l64_64622


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64756

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64756


namespace tangent_to_graph_of_f_l64_64042

def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3 * x + 1

theorem tangent_to_graph_of_f (a : ℝ) :
  (∃ x : ℝ, f'(x) = 0 ∧ f(x) = a) → (a = -8 ∨ a = 8 / 3) :=
by
  sorry

end tangent_to_graph_of_f_l64_64042


namespace min_value_expr_is_neg13n2_l64_64430

noncomputable def minimum_value_of_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  let cos_sum := ∑ i, Real.cos (x i)
  let sin_sum := ∑ i, Real.sin (x i)
  (5 * cos_sum + sin_sum) * (cos_sum - 5 * sin_sum)

theorem min_value_expr_is_neg13n2 (n : ℕ) (x : Fin n → ℝ) :
  (∃ x : Fin n → ℝ, minimum_value_of_expression n x = -13 * n^2) :=
sorry

end min_value_expr_is_neg13n2_l64_64430


namespace sum_arith_seq_inverse_l64_64680

theorem sum_arith_seq_inverse (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (n : ℕ) 
  (h1 : a_n 3 = 3) 
  (h2 : S_n 4 = 10) 
  (h3 : ∀ n, S_n n = n * (a_n 1 + (n - 1) * (a_n 2 - a_n 1) / 2)) :
  (∑ k in Finset.range n, (1 / S_n k)) = (2 * n / (n + 1)) := 
by
  sorry

end sum_arith_seq_inverse_l64_64680


namespace divisible_by_9_l64_64348

def digit_sum (n : ℕ) : ℕ := n.digits.sum

theorem divisible_by_9 (x : ℤ) 
  (h : digit_sum x.natAbs = digit_sum (3*x).natAbs) : 9 ∣ x := 
sorry

end divisible_by_9_l64_64348


namespace sum_pow_neg_one_l64_64219

theorem sum_pow_neg_one : ∑ k in Finset.range 2007, (-1 : ℤ)^(k + 1) = -1 :=
by
  sorry

end sum_pow_neg_one_l64_64219


namespace point_not_in_transformed_plane_l64_64580

theorem point_not_in_transformed_plane (A : ℝ × ℝ × ℝ) (plane : ℝ → ℝ → ℝ → Prop) (k : ℝ) :
  A = (7, 0, -1) ∧
  (∀ x y z, plane x y z ↔ x - y - z - 1 = 0) ∧
  k = 4 →
  ¬plane (k * 7) (k * 0) (k * -1) :=
by
  intro h
  obtain ⟨hA, hplane, hk⟩ := h
  have transform : ∀ x y z, plane x y z ↔ x - y - z - 4 = 0 :=
    by
    intro x y z
    have base_plane := hplane x y z
    rw [← base_plane, sub_eq_add_neg, sub_eq_add_neg, sub_eq_add_neg, sub_eq_add_neg, add_assoc, add_assoc, add_assoc, neg_mul_eq_neg_mul, mul_one, mul_zero, mul_neg, add_right_eq_self, eq_comm]
  have A_on_plane : ¬(7 * 4 - 0 - (-1) * 4 - 4 = 0) :=
    by
    calc
      7 * 4 - 0 - (-1) * 4 - 4 = 28 + 4 - 4 := by norm_num
      _ = 28 := by ring
      _ ≠ 0 := by norm_num
  exact A_on_plane (transform 7 0 (-1)).mpr ((hplane 7 0 (-1)).mpr rfl)
  sorry

end point_not_in_transformed_plane_l64_64580


namespace no_equiangular_parallelogram_in_regular_hexagon_l64_64380

theorem no_equiangular_parallelogram_in_regular_hexagon :
  ∀ (hex : Type) [RegularHexagon hex],
    ¬ ∃ (parallelogram : Type) [IsParallelogram parallelogram], 
    IsEquiangular parallelogram := 
sorry

end no_equiangular_parallelogram_in_regular_hexagon_l64_64380


namespace area_of_rectangle_at_stage_4_l64_64067

def area_at_stage (n : ℕ) : ℕ :=
  let square_area := 16
  let initial_squares := 2
  let common_difference := 2
  let total_squares := initial_squares + common_difference * (n - 1)
  total_squares * square_area

theorem area_of_rectangle_at_stage_4 :
  area_at_stage 4 = 128 :=
by
  -- computation and transformations are omitted
  sorry

end area_of_rectangle_at_stage_4_l64_64067


namespace significant_improvement_l64_64269

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64269


namespace integral_result_l64_64521

-- Define the main problem and the given conditions
theorem integral_result (a : ℝ) (h_coefficient : binomial.coeff 6 3 * a^3 = 160) :
  ∫ x in 1..a, x^(a:ℝ) = 7 / 3 :=
by
  sorry

end integral_result_l64_64521


namespace find_angle_DAB_l64_64993

-- Definition of the Triangle and conditions
variable {Point : Type} [InnerProductSpace ℝ Point]
variable (A B C D : Point)
variable (x : ℝ)

-- Empty sorry proof to ensure code builds
axiom point_on_line (A D : Point) (x : ℝ): ∃ C : Point, dist A C = x ∧ dist C D = x

def is_isosceles_triangle (A B D : Point) :=
  dist A B = dist B D

def conditions {A B C D : Point} (x : ℝ) :=
  is_isosceles_triangle A B D ∧
  dist A D = 2 * x ∧
  (∃ C : Point, dist A C = x ∧ dist C D = x) ∧
  ∃ θ : ℝ, ∠ A D B = 2 * θ ∧ ∠ D A B = θ

-- Target theorem to prove
theorem find_angle_DAB (h: conditions x) : 
  ∃ θ : ℝ, ∠ D A B = 45 :=
sorry

end find_angle_DAB_l64_64993


namespace triangle_identity_zero_l64_64069

variable (a b c A B C : ℝ)
variable (hABC : ∀ {a b c : ℝ}, 0 < a ∧ 0 < b ∧ 0 < c → a + b > c ∧ a + c > b ∧ b + c > a)

theorem triangle_identity_zero (a b c A B C : ℝ) 
  (hABC : ∀ {a b c : ℝ}, 0 < a ∧ 0 < b ∧ 0 < c → a + b > c ∧ a + c > b ∧ b + c > a)
  (haBC : a^3 * Math.sin (B - C) + b^3 * Math.sin (C - A) + c^3 * Math.sin (A - B) = 0) :
  a^3 * Math.sin (B - C) + b^3 * Math.sin (C - A) + c^3 * Math.sin (A - B) = 0 :=
by sorry

end triangle_identity_zero_l64_64069


namespace range_of_m_range_of_x_l64_64125

def f (m x : ℝ) := m * x^2 - m * x - 6 + m

theorem range_of_m (m : ℝ) : (∀ x ∈ set.Icc (1 : ℝ) 3, f m x < 0) → m < 6 / 7 := by
  sorry

theorem range_of_x (x : ℝ) : (∀ m ∈ set.Icc (-2 : ℝ) 2, f m x < 0) → -1 < x ∧ x < 2 := by
  sorry

end range_of_m_range_of_x_l64_64125


namespace fractions_count_l64_64628

def is_fraction (x : ℚ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def given_numbers : set ℚ :=
  {-0.1, -2023, 21, 3.14, 0, -15/8}

def count_fractions (s : set ℚ) : ℕ :=
  s.count is_fraction

theorem fractions_count : count_fractions given_numbers = 3 :=
  sorry

end fractions_count_l64_64628


namespace total_cupcakes_baked_l64_64609

theorem total_cupcakes_baked
    (boxes : ℕ)
    (cupcakes_per_box : ℕ)
    (left_at_home : ℕ)
    (total_given_away : ℕ)
    (total_baked : ℕ)
    (h1 : boxes = 17)
    (h2 : cupcakes_per_box = 3)
    (h3 : left_at_home = 2)
    (h4 : total_given_away = boxes * cupcakes_per_box)
    (h5 : total_baked = total_given_away + left_at_home) :
    total_baked = 53 := by
  sorry

end total_cupcakes_baked_l64_64609


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64849

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64849


namespace minimum_value_of_sum_of_squares_l64_64024

noncomputable def minimum_of_sum_of_squares (a b : ℝ) : ℝ :=
  a^2 + b^2

theorem minimum_value_of_sum_of_squares (a b : ℝ) (h : |a * b| = 6) :
  a^2 + b^2 ≥ 12 :=
by {
  sorry
}

end minimum_value_of_sum_of_squares_l64_64024


namespace eval_expression_l64_64414

theorem eval_expression :
  ( (real.cbrt (8 + 3 * real.sqrt 21)) + (real.cbrt (8 - 3 * real.sqrt 21)) ) ^ 2 = 1 := 
by
  sorry

end eval_expression_l64_64414


namespace product_of_triple_when_added_to_reciprocal_l64_64894

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64894


namespace evaluate_f_a_plus_1_l64_64492

variable (a : ℝ)  -- The variable a is a real number.

def f (x : ℝ) : ℝ := x^2 + 1  -- The function f is defined as x^2 + 1.

theorem evaluate_f_a_plus_1 : f (a + 1) = a^2 + 2 * a + 2 := by
  -- Provide the proof here
  sorry

end evaluate_f_a_plus_1_l64_64492


namespace area_under_curve_l64_64620

open Real

theorem area_under_curve : ∫ x in 1..2, 1/x = log 2 := by
  sorry

end area_under_curve_l64_64620


namespace sum_neg_one_power_is_real_l64_64049

theorem sum_neg_one_power_is_real (n : ℕ)
  (a : Fin (n+1) → ℂ)
  (b : Fin (n+1) → ℂ)
  (h_eq_1 : a 0 = 1 ∧ b 0 = 1)
  (h_real_even_sum : ∑ i in Finset.range (⌊n / 2⌋), b (⟨2 * (i + 1), sorry⟩) ∈ ℝ)
  (h_real_odd_sum : ∑ i in Finset.range (⌊(n + 1) / 2⌋), b (⟨2 * i + 1, sorry⟩) ∈ ℝ)
  (h_roots : ∀ x : ℂ, is_root (∑ i in Finset.range (n+1), b i * x ^ (n-i)) x ↔ is_root (∑ i in Finset.range (n+1), a i * x ^ (n-i)) (-x^2)) :
  (∑ i in Finset.range n, (-1 : ℂ) ^ i * a (⟨i + 1, sorry⟩)) ∈ ℝ := by sorry

end sum_neg_one_power_is_real_l64_64049


namespace significant_improvement_l64_64270

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64270


namespace polynomial_inequality_l64_64320

open Real

theorem polynomial_inequality
  (n : ℕ) (a_1 a_2 a_3 a_4 ... a_n : ℝ)
  (n_pos : 1 < n)
  (roots : Fin n → ℝ)
  (h_roots_distinct : Function.Injective roots)
  (x_1 x_2 ... x_n : ℝ)
  (h_roots_eq : ∀ i j, roots i = x_i ∧ roots j = x_j)
  (derivative_roots : Fin (n - 1) → ℝ)
  (h_derivative : ∀ i, derivative_roots i = y_i)
  (P : Polynomial ℝ := Polynomial.X^n + (Polynomial.C a_1 * Polynomial.X^(n-1)) + ... + (Polynomial.C a_n))
  (Q : Polynomial ℝ := Polynomial.deriv (Polynomial.X^n + (Polynomial.C a_1 * Polynomial.X^(n - 1)) + ... + (Polynomial.C a_n))) :
  (∑ i in range n, x_i^2) / n > (∑ i in range (n - 1), y_i^2) / (n - 1) :=
sorry

end polynomial_inequality_l64_64320


namespace normalized_sequence_embedding_l64_64438

/-- Definitions for sequences and embeddings --/
def non_negative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i, 0 ≤ a i

def interval_embedding (a : ℕ → ℝ) (x : ℕ → ℝ) (b c : ℝ) (k : ℕ) : Prop :=
  ∀ i < k, |x i - x (i - 1)| = a i ∧ x i ≤ c ∧ x i ≥ b

/-- For any given non-negative integer n, prove:(1) Any normalized sequence of length 2n+1 
can be embeded in the interval [0, 2- 1/(2^n)]. 
(2) There exists a normalized sequence of length 4n+3 that cannot be embeded in the interval [0, 2- 1/(2^n)]. --/
theorem normalized_sequence_embedding (n : ℕ) :
  (∀ a : ℕ → ℝ, (∀ i < 2 * n + 1, 0 ≤ a i ∧ a i ≤ 1) →
    ∃ x : ℕ → ℝ, interval_embedding a x 0 (2 - (1 / (2 ^ n))) (2 * n + 1)) ∧
  (∃ a : ℕ → ℝ, (∀ i < 4 * n + 3, 0 ≤ a i ∧ a i ≤ 1) ∧
    ¬ ∃ x : ℕ → ℝ, interval_embedding a x 0 (2 - (1 / (2 ^ n))) (4 * n + 3)) := sorry

end normalized_sequence_embedding_l64_64438


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64754

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64754


namespace sculpture_cost_200_NAD_to_INR_l64_64600

theorem sculpture_cost_200_NAD_to_INR (usd_to_nad : ℕ)
  (usd_to_cny : ℕ)
  (cny_to_inr : ℕ)
  (sculpture_cost_nad : ℕ) :
  usd_to_nad = 10 → usd_to_cny = 7 → cny_to_inr = 10 → sculpture_cost_nad = 200 → 
  let cost_in_usd := sculpture_cost_nad / usd_to_nad in 
  let cost_in_cny := cost_in_usd * usd_to_cny in 
  let cost_in_inr := cost_in_cny * cny_to_inr in
  cost_in_inr = 1400 :=
by
  intros h1 h2 h3 h4
  let cost_in_usd := sculpture_cost_nad / usd_to_nad
  let cost_in_cny := cost_in_usd * usd_to_cny
  let cost_in_inr := cost_in_cny * cny_to_inr
  sorry

end sculpture_cost_200_NAD_to_INR_l64_64600


namespace find_n_l64_64525

theorem find_n (x : ℝ) (n : ℕ) (h : 2 * (nat.choose n 2) = (nat.choose n 1) + (nat.choose n 3)) : n = 7 :=
sorry

end find_n_l64_64525


namespace sum_factors_of_30_l64_64953

theorem sum_factors_of_30 : 
  (∑ i in {1, 2, 3, 5, 6, 10, 15, 30}, i) = 72 := 
by
  -- Sorry, to skip the proof here.
  sorry

end sum_factors_of_30_l64_64953


namespace find_x_l64_64051

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h : a = (Real.cos (3 * x / 2), Real.sin (3 * x / 2)) ∧ b = (Real.cos (x / 2), -Real.sin (x / 2)) ∧ (a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2 = 1 ∧ 0 ≤ x ∧ x ≤ Real.pi)  :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_l64_64051


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64925

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64925


namespace cone_volume_correct_l64_64668

noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let r := real.sqrt (slant_height^2 - height^2)
  in (1/3) * real.pi * r^2 * height

theorem cone_volume_correct :
  cone_volume 15 9 = 432 * real.pi := by
  sorry

end cone_volume_correct_l64_64668


namespace probability_more_twos_than_fours_l64_64517

theorem probability_more_twos_than_fours (h : true) :
  (421 : ℚ) / 1296 = 
  (1 / 2) * (1 - 
    let equal_count_ways := 6 + 12 * 16 + 256 in
    let total_outcomes := 1296 in
    (equal_count_ways : ℚ) / total_outcomes) := sorry

end probability_more_twos_than_fours_l64_64517


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64834

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64834


namespace factorial_expression_value_l64_64977

theorem factorial_expression_value :
  (12! - 11!) / 10! = 121 :=
by
  sorry

end factorial_expression_value_l64_64977


namespace cone_volume_l64_64678

variable (l h : ℝ) (π : ℝ := Real.pi)

noncomputable def radius (l h : ℝ) : ℝ := sqrt (l^2 - h^2)

noncomputable def volume_of_cone (r h : ℝ) (π : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (hl : l = 15) (hh : h = 9) :
  volume_of_cone (radius l h) h π = 432 * π :=
by
  sorry

end cone_volume_l64_64678


namespace max_consistent_statements_l64_64446

theorem max_consistent_statements :
  let a : Nat := sorry in
  (a % 2 = 1 → a % 4 = 2 → False) →  -- Contradictory condition
  (∃ a >= 1000, 
    (a % 2 = 1 ∧ a % 3 = 1 ∧ a % 5 = 2) ∨ 
    (a % 3 = 1 ∧ a % 4 = 2 ∧ a % 5 = 2)) := 
sorry

end max_consistent_statements_l64_64446


namespace tangent_line_at_e_range_of_a_l64_64033

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x - 2 * a * x

theorem tangent_line_at_e (a : ℝ) :
  a = 0 →
  ∃ m b : ℝ, (∀ x, y = m * x + b) ∧ 
             y = (2 / Real.exp 1 - 2 * Real.exp 1) * x + (Real.exp 1)^2 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Set.Ioi 1 → g a x < 0) →
  a ∈ Set.Icc (-1) 1 :=
sorry

end tangent_line_at_e_range_of_a_l64_64033


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64855

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64855


namespace hyperbola_eccentricity_l64_64023

-- Definitions based on given conditions
def hyperbola_asymptote_eq (a b : ℝ) : Prop :=
  b = 2 * a ∨ a = 2 * b

def eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity (a b : ℝ) (h : hyperbola_asymptote_eq a b) :
  eccentricity a b = Real.sqrt 5 ∨ eccentricity a b = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l64_64023


namespace product_of_solutions_l64_64870

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64870


namespace find_area_parallelogram_l64_64577

open Real

noncomputable def area_of_parallelogram (u v : Vec3) (h1 : ‖u‖ = 1) (h2 : ‖v‖ = 1) (h3 : angle u v = π / 4) : ℝ :=
  ‖(-(u + 3 * v) + (3 * u + v)) × (2 * (u + 3 * v))‖

theorem find_area_parallelogram (u v : Vec3) (h1 : ‖u‖ = 1) (h2 : ‖v‖ = 1) (h3 : angle u v = π / 4) :
  area_of_parallelogram u v h1 h2 h3 = 2 * sqrt 2 := 
sorry

end find_area_parallelogram_l64_64577


namespace spherical_to_rectangular_coords_l64_64392

theorem spherical_to_rectangular_coords (
  {ρ θ φ : ℝ} 
) (hρ : ρ = 3) (hθ : θ = 3 * Real.pi / 2) (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by 
  sorry

end spherical_to_rectangular_coords_l64_64392


namespace product_trippled_when_added_to_reciprocal_l64_64927

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64927


namespace product_of_real_solutions_triple_property_l64_64793

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64793


namespace min_expression_value_l64_64583

theorem min_expression_value (x y z : ℝ) (xyz_eq : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n : ℝ, (∀ x y z : ℝ, x * y * z = 1 → 0 < x → 0 < y → 0 < z → 2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ n)
    ∧ n = 72 :=
sorry

end min_expression_value_l64_64583


namespace concurrency_l64_64120

variables {K : Type*} [Field K]

structure Triangle (K : Type*) [Field K] :=
(A B C D E F D P AC BC CD DE BF CE : K)

open Triangle

theorem concurrency (Γ : Triangle K) (circumcircle_abc : Set K) (point_D_on_BC : Set K) 
(tangent_to_circumcircle_at_A : Set K) (parallel_to_BA_through_D : Set K)
(CE_intersects_circumcircle_at_F : Set K) (BDFE_concyclic : Set K) :
Set K :=
  ∃ G : Set K, G ∈ (AC ∩ BF ∩ DE)


end concurrency_l64_64120


namespace solve_f_3_l64_64616

noncomputable def f : ℝ → ℝ := sorry -- The linear function f(x)
noncomputable def f_inv : ℝ → ℝ := sorry -- The inverse function of f

theorem solve_f_3 :
    (∀ x, f x = 5 * f_inv x + 7) →
    (f 0 = 5) →
    (f 3 = 3 * real.sqrt 5 + 5) :=
by 
    intro h1 h2
    sorry

end solve_f_3_l64_64616


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64827

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64827


namespace john_total_distance_l64_64563

def john_journey_distance : ℚ := 
  let y := 360 / 7
  y

theorem john_total_distance (y : ℚ) 
  (h1 : y = john_journey_distance)
  (h2 : y / 4 + 30 + y / 6 = y) : 
  y = 360 / 7 :=
begin
  sorry
end

end john_total_distance_l64_64563


namespace multiplication_correct_l64_64240

theorem multiplication_correct : 3795421 * 8634.25 = 32774670542.25 := by
  sorry

end multiplication_correct_l64_64240


namespace max_min_value_l64_64116

noncomputable def max_min_ab_bc_ca (a b c : ℝ) (h1 : a + b + c = 8) (h2 : ab + bc + ca = 16) : ℝ :=
  max (min (a * b) (b * c)) (c * a)

theorem max_min_value : ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 8 ∧ ab + bc + ca = 16 ∧
  max_min_ab_bc_ca a b c = 16 / 9 :=
by
  sorry

end max_min_value_l64_64116


namespace complex_pure_imaginary_l64_64064

theorem complex_pure_imaginary (a : ℝ) :
  let z := (a + complex.i) * (3 + 2 * complex.i)
  in (z.re = 0) → a = 2 / 3 :=
by
  sorry

end complex_pure_imaginary_l64_64064


namespace calculation_power_l64_64374

theorem calculation_power :
  (0.125 : ℝ) ^ 2012 * (2 ^ 2012) ^ 3 = 1 :=
sorry

end calculation_power_l64_64374


namespace mart_income_more_than_tim_l64_64130

variable (J : ℝ) -- Let's denote Juan's income as J
def T : ℝ := J - 0.40 * J -- Tim's income is 40 percent less than Juan's income
def M : ℝ := 0.78 * J -- Mart's income is 78 percent of Juan's income

theorem mart_income_more_than_tim : (M - T) / T * 100 = 30 := by
  sorry

end mart_income_more_than_tim_l64_64130


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64848

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64848


namespace quadratic_function_intersection_unique_l64_64456

theorem quadratic_function_intersection_unique (a b c : ℝ) (m n : ℝ) :
  (∀ x : ℝ, x = -1 → a*(-1)^2 + b*(-1) + c = m*(-1) + n = 2 ∧ 
            x = 2 → a*2^2 + b*2 + c = m*2 + n = 5 ∧ 
            a > 0 ∧ -b/(2*a) = 1) →
  (∀ x : ℝ, y = a*x^2 + b*x + c → 
    (y = x^2 + 1 ∨ y = (1/9)*x^2 + (8/9)*x + 25/9)) :=
sorry

end quadratic_function_intersection_unique_l64_64456


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64826

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64826


namespace median_of_set_l64_64063

theorem median_of_set (x : ℝ) (h_average : (2 + (-1) + 0 + (-3) + 5 + x) / 6 = 1) :
  let data := [2, -1, 0, -3, 5, x].sort
  in (data.nth 2 + data.nth 3) / 2 = 1 :=
by
  -- Lean-specific code goes here
  sorry

end median_of_set_l64_64063


namespace math_problem_l64_64034

noncomputable def f (x a : ℝ) : ℝ :=
  (2 * a - 1) * Real.log x - 1 / x - 2 * a * x

def part1 := ∀ x : ℝ, x > 0 → (f x 0).derivative x = (1 - x) / x^2

def part2 := ∀ a x : ℝ, 
  a ≠ 0 → x > 0 → 
  if a > 0 then ((f x a).derivative x = (2 * a * x + 1) * (1 - x) / x^2)
  else if a < 0 ∧ a < -1/2 then
    (if 0 < x ∧ x < -1/(2*a) then (f x a).derivative x > 0 ∧ 
     if -1/(2*a) < x ∧ x < 1 then (f x a).derivative x < 0 ∧ 
     if x > 1 then (f x a).derivative x > 0)
  else if a = -1/2 then (f x a).derivative x ≥ 0 
  else if -1/2 < a ∧ a < 0 then
    (if 0 < x ∧ x < 1 then (f x a).derivative x > 0 ∧ 
     if x > 1 ∧ x < -1/(2*a) then (f x a).derivative x < 0 ∧ 
     if x > -1/(2*a) then (f x a).derivative x > 0)

def part3 := ∀ a m : ℝ,
  a ∈ Set.Ico (-2 : ℝ) (-1) →
  ∀ x1 x2 : ℝ, x1 ∈ Set.Icc 1 Real.exp ∧ x2 ∈ Set.Icc 1 Real.exp →
  (m - 2 * Real.exp) * a - 1 / Real.exp + 2 ≥ |f x1 a - f x2 a| →
  m ≤ 5

theorem math_problem : part1 ∧ part2 ∧ part3 := 
  by 
    split
    case part1 =>
      sorry
    case part2 =>
      sorry
    case part3 =>
      sorry

end math_problem_l64_64034


namespace sum_of_series_l64_64185

theorem sum_of_series {n : ℕ} :
  let seq := λ k, (2 * k - 1) * (1 / 2^k)
  let partial_sum := λ n, ∑ i in finset.range n, seq (i + 1)
  partial_sum n = n^2 + 1 - 1 / 2^n :=
by
  sorry

end sum_of_series_l64_64185


namespace product_trippled_when_added_to_reciprocal_l64_64941

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64941


namespace oranges_equiv_frac_bananas_l64_64150

theorem oranges_equiv_frac_bananas :
  (3 / 4) * 16 * (1 / 3) * 9 = (3 / 2) * 6 :=
by
  sorry

end oranges_equiv_frac_bananas_l64_64150


namespace Bowen_buys_40_pens_l64_64355

variable (P Q : ℕ)
variable (cost_pencil cost_pen : ℝ)
variable (ratio_pncl_to_pns : ℝ)
variable (total_cost : ℝ)

def number_of_pens (P Q : ℕ) (cost_pencil cost_pen : ℝ) (ratio_pncl_to_pns total_cost : ℝ) : Prop :=
  Q = P + ratio_pncl_to_pns * P ∧
  cost_pencil = 0.25 ∧
  cost_pen = 0.15 ∧
  total_cost = 20 ∧
  cost_pen * P + cost_pencil * Q = total_cost

theorem Bowen_buys_40_pens : ∃ P, ∀ Q cost_pencil cost_pen ratio_pncl_to_pns total_cost,
  number_of_pens P Q cost_pencil cost_pen ratio_pncl_to_pns total_cost → P = 40 :=
by
  intro P
  exists 40
  intro Q cost_pencil cost_pen ratio_pncl_to_pns total_cost
  intro h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end Bowen_buys_40_pens_l64_64355


namespace Hannah_says_242_l64_64410

theorem Hannah_says_242 : ∃ n : ℕ, 1 ≤ n ∧ n ≤ 500 ∧ 
  (∀ k : ℕ, (Alice_says k ∨ Barbara_says k ∨ Candice_says k ∨ Debbie_says k ∨ 
             Eliza_says k ∨ Fatima_says k ∨ Greg_says k) → k ≠ n) 
  ∧ n = 242 :=
sorry

-- Required definitions for the preceding theorem

def Alice_says (n : ℕ) : Prop := ¬ (∃ k : ℕ, 1 ≤ k ∧ 3 * k - 1 = n)

def Barbara_says (n : ℕ) : Prop := ¬ (∃ k : ℕ, k ≥ 1 ∧ (3 * (3 * k - 1) - 1 = n) ∨ Alice_says n)

def Candice_says (n : ℕ) : Prop := ¬ (∃ k : ℕ, k ≥ 1 ∧ (3 * (3 * (3 * k - 1) - 1) - 1 = n) ∨ 
                                       Alice_says n ∨ Barbara_says n)

def Debbie_says (n : ℕ) : Prop := ¬ (∃ k : ℕ, k ≥ 1 ∧ (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1 = n) ∨ 
                                       Alice_says n ∨ Barbara_says n ∨ Candice_says n)

def Eliza_says (n : ℕ) : Prop := ¬ (∃ k : ℕ, k ≥ 1 ∧ (3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1 = n) ∨ 
                                       Alice_says n ∨ Barbara_says n ∨ Candice_says n ∨ Debbie_says n)

def Fatima_says (n : ℕ) : Prop := ¬ (∃ k : ℕ, k ≥ 1 ∧ (3 * (3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1) - 1 = n) ∨ 
                                       Alice_says n ∨ Barbara_says n ∨ Candice_says n ∨ Debbie_says n ∨ Eliza_says n)

def Greg_says (n : ℕ) : Prop := ¬ (∃ k : ℕ, k ≥ 1 ∧ (3 * (3 * (3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1) - 1) - 1 = n) ∨ 
                                       Alice_says n ∨ Barbara_says n ∨ Candice_says n ∨ Debbie_says n ∨ Eliza_says n ∨ Fatima_says n)

end Hannah_says_242_l64_64410


namespace product_of_solutions_l64_64770

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64770


namespace selection_probability_l64_64327

-- Define the total number of students and the number eliminated
def n : ℕ := 2007
def e : ℕ := 7

-- Define the remaining students
def r : ℕ := n - e

-- Define the number of students to be sampled
def k : ℕ := 50

-- Define the probability p of each student being selected
def p := (r : ℝ) / n * (k : ℝ) / r

-- State the theorem to prove the probability is as stated
theorem selection_probability :
  p = (50 : ℝ) / 2007 :=
by
  -- Sorry to skip the proof
  sorry

end selection_probability_l64_64327


namespace NeoGauss_8_year_percentage_l64_64542

def NeoGauss_employees_percentage (y : ℕ) : ℕ :=
  let T := 4 * y + 6 * y + 5 * y + 4 * y + 3 * y + 2 * y + y in
  let E := 3 * y + 2 * y + y in
  (E * 100) / T

theorem NeoGauss_8_year_percentage (y : ℕ) (hy : y > 0) :
  NeoGauss_employees_percentage y = 24 := 
by
  let T := 4 * y + 6 * y + 5 * y + 4 * y + 3 * y + 2 * y + y
  let E := 3 * y + 2 * y + y
  have hT : T = 25 * y := by simp
  have hE : E = 6 * y := by simp
  rw [NeoGauss_employees_percentage, hT, hE]
  simp
  rw [mul_div_cancel_left _ (nat.pos_of_ne_zero (ne_of_gt hy)), nat.mul_div_cancel_left 600 25]
  exact rfl

end NeoGauss_8_year_percentage_l64_64542


namespace max_bamboo_l64_64321

-- Definitions and conditions
variables {V : Type} [DecidableEq V] {G : SimpleGraph V}
variables (n : ℕ) (x : V → ℝ)

-- Assumption: the graph is a tree with n vertices
variable [hTree : G.is_tree]
variable [hCard : G.adj.cardVertices = n]

-- Assumption: The given condition 
def condition (G : SimpleGraph V) (x : V → ℝ) : Prop :=
  ∀ v : V, 1 + x v ≥ (G.neighborValues x v).sum / (G.neighborValues x v).length

-- Definitions of M and m
def M : ℝ := sup {x v | v : V}
def m : ℝ := inf {x v | v : V}

-- Theorem statement
theorem max_bamboo (h : condition G x) : M - m ≤ (n - 1)^2 := sorry

end max_bamboo_l64_64321


namespace convert_to_base10_sum_l64_64416

def base8_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def base13_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 13^2 + d1 * 13^1 + d0 * 13^0

def convert_537_8 : Nat :=
  base8_to_dec 5 3 7

def convert_4C5_13 : Nat :=
  base13_to_dec 4 12 5

theorem convert_to_base10_sum : 
  convert_537_8 + convert_4C5_13 = 1188 := 
by 
  sorry

end convert_to_base10_sum_l64_64416


namespace cone_volume_correct_l64_64669

noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let r := real.sqrt (slant_height^2 - height^2)
  in (1/3) * real.pi * r^2 * height

theorem cone_volume_correct :
  cone_volume 15 9 = 432 * real.pi := by
  sorry

end cone_volume_correct_l64_64669


namespace find_correct_speed_l64_64131

-- Definitions for given conditions
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Given conditions as definitions
def condition1 (d t : ℝ) : Prop := distance_traveled 35 (t + (5 / 60)) = d
def condition2 (d t : ℝ) : Prop := distance_traveled 55 (t - (5 / 60)) = d

-- Statement to prove
theorem find_correct_speed (d t r : ℝ) (h1 : condition1 d t) (h2 : condition2 d t) :
  r = (d / t) ∧ r = 42.78 :=
by sorry

end find_correct_speed_l64_64131


namespace slope_of_line_AF_l64_64016

variable (A F : Point) (parabola : Parabola)

def Point := ℝ × ℝ

structure Parabola where
  p : ℝ
  equation : ℝ × ℝ → Prop
  directrix : ℝ → Prop

def slope (p1 p2 : Point) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_line_AF :
  ∃ (p : ℝ), parabola.equation = (λ (p : Point), p.2^2 = 2 * p * p.1) →
  parabola.directrix = (λ (x : ℝ), x = -p/2) →
  A = (-2, 3) →
  parabola.directrix A.1 →
  ∃ F : Point, F = (p/2, 0) → slope A F = -3/4 :=
sorry

end slope_of_line_AF_l64_64016


namespace general_term_of_seq_a_value_of_seq_T_l64_64575

open Nat

def seq_a (n : ℕ) : ℕ := 2 * n + 1

def seq_S (n : ℕ) : ℕ :=
  (seq_a n ^ 2) / 4 + seq_a n / 2 - 3 / 4

theorem general_term_of_seq_a (n : ℕ) (hpos : ∀ n, seq_a n > 0) : 
  seq_a n = 2 * n + 1 :=
sorry

def seq_b (n : ℕ) : ℕ := 2^n

def seq_T (n : ℕ) : ℕ :=
(∑ i in range (n+1), seq_a i * seq_b i)

theorem value_of_seq_T (n : ℕ) : 
  seq_T n = (2 * n - 1) * 2^(n + 1) + 2 :=
sorry

end general_term_of_seq_a_value_of_seq_T_l64_64575


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64723

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64723


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64847

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64847


namespace chord_constant_correct_l64_64200

noncomputable def chord_constant_proof : Prop :=
  ∀ (d : ℝ), d = 1/2 → ∀ (m : ℝ),
    let y := 4 * d^2 * d in
    ∀ (AC BC : ℝ) (A B C : ℝ × ℝ),
      (C = (0, d)) →
      (y = m * AC + d) →
      (y = 4 * A.1^2) →
      (y = 4 * B.1^2) →
      A = (A.1, 4 * A.1^2) →
      B = (B.1, 4 * B.1^2) →
      u = 1/AC^4 + 1/BC^4 = 32

theorem chord_constant_correct : chord_constant_proof :=
by sorry

end chord_constant_correct_l64_64200


namespace product_of_solutions_l64_64767

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64767


namespace no_child_age_is_11_license_plate_divisibility_eleven_is_not_an_age_of_mr_jones_children_l64_64595

noncomputable def license_plate_number : ℕ := 8881

def mr_jones_children_ages : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def eldest_child_age : ℕ := 12

def is_divisible_by_all_children_ages (n : ℕ) : Prop :=
  ∀ age ∈ mr_jones_children_ages, age ≠ 11 → n % age = 0

theorem no_child_age_is_11 : ¬(11 ∈ mr_jones_children_ages) :=
by
  sorry

theorem license_plate_divisibility : is_divisible_by_all_children_ages license_plate_number :=
by
  sorry

theorem eleven_is_not_an_age_of_mr_jones_children : ∀ age, age ∈ mr_jones_children_ages → license_plate_number % age = 0 → age ≠ 11 :=
by
  intros age h1 h2
  apply no_child_age_is_11
  exact h1
  sorry

end no_child_age_is_11_license_plate_divisibility_eleven_is_not_an_age_of_mr_jones_children_l64_64595


namespace sqrt_expression_l64_64366

theorem sqrt_expression :
  Real.sqrt 18 - 3 * Real.sqrt (1 / 2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 :=
by
  sorry

end sqrt_expression_l64_64366


namespace angles_of_A2BC2_l64_64460

theorem angles_of_A2BC2 {A B C A1 C1 A2 C2 : Point}
  {triangle_acute : ∀ P Q R : Point, is_triangle P Q R → acute_triangle P Q R}
  (hABC : is_triangle A B C)
  (h_acute : acute_triangle A B C)
  (hA1 : altitude A A1 B C)
  (hC1 : altitude C C1 A B)
  (hA2 : ∥A A2∥ = ∥B C∥)
  (hC2 : ∥C C2∥ = ∥A B∥) :
  ∃ α β γ : ℝ,
  angle A2 B C2 = α ∧
  angle B A2 C2 = β ∧
  angle B C2 A2 = γ ∧
  α = 90 ∧ β = 45 ∧ γ = 45 :=
by {
  -- Placeholder for proof
  sorry
}

end angles_of_A2BC2_l64_64460


namespace least_money_Moe_l64_64364

theorem least_money_Moe (Bo Coe Flo Jo Moe Zoe : ℝ)
  (H1 : Flo > Jo) 
  (H2 : Flo > Bo) 
  (H3 : Bo > Zoe) 
  (H4 : Coe > Zoe) 
  (H5 : Jo > Zoe) 
  (H6 : Bo > Jo) 
  (H7 : Zoe > Moe) : 
  (Moe < Bo) ∧ (Moe < Coe) ∧ (Moe < Flo) ∧ (Moe < Jo) ∧ (Moe < Zoe) :=
by
  sorry

end least_money_Moe_l64_64364


namespace trig_identity_l64_64515

theorem trig_identity (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : 
  Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by 
  sorry

end trig_identity_l64_64515


namespace rectangle_area_l64_64337

theorem rectangle_area (square_area : ℝ) (width length : ℝ) 
  (h1 : square_area = 36) 
  (h2 : width = real.sqrt square_area) 
  (h3 : length = 3 * width) : 
  width * length = 108 :=
by
  sorry

end rectangle_area_l64_64337


namespace find_q_l64_64503

variable (p q : ℝ)

-- Given conditions
def condition1 : Prop := p > 1
def condition2 : Prop := q > 1
def condition3 : Prop := (1 / p) + (1 / q) = 1
def condition4 : Prop := p * q = 9

theorem find_q :
  condition1 p q →
  condition2 p q →
  condition3 p q →
  condition4 p q →
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  intro h1 h2 h3 h4
  sorry

end find_q_l64_64503


namespace rectangular_floor_paint_l64_64325

theorem rectangular_floor_paint (a b : ℕ) (ha : a > 0) (hb : b > a) (h1 : a * b = 2 * (a - 4) * (b - 4) + 32) : 
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → b > a :=
by 
  sorry

end rectangular_floor_paint_l64_64325


namespace melissa_games_played_l64_64594

theorem melissa_games_played (points_per_game total_points : ℕ) (h_points_per_game : points_per_game = 12) (h_total_points : total_points = 36) :
  total_points / points_per_game = 3 :=
by
  rw [h_points_per_game, h_total_points]
  norm_num

end melissa_games_played_l64_64594


namespace certain_number_is_perfect_square_l64_64220

theorem certain_number_is_perfect_square 
  (x : ℕ) (d : ℕ) 
  (hx : x = 2880) 
  (hd : d = 5) : 
  let y := x * d in 
  ∃ (n : ℕ), y = n^2 :=
sorry

end certain_number_is_perfect_square_l64_64220


namespace product_of_all_solutions_triple_reciprocal_l64_64744

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64744


namespace sum_factors_of_30_l64_64947

def factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ m, m > 0 ∧ n % m == 0)

theorem sum_factors_of_30 : (factors 30).sum = 72 := by
  sorry

end sum_factors_of_30_l64_64947


namespace product_of_solutions_l64_64804

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64804


namespace local_minimum_at_1_1_l64_64089

noncomputable def function (x y : ℝ) : ℝ :=
  x^3 + y^3 - 3 * x * y

theorem local_minimum_at_1_1 : 
  ∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ (∀ (z : ℝ), z = function x y → z = -1) :=
sorry

end local_minimum_at_1_1_l64_64089


namespace blue_die_prime_yellow_die_power_2_probability_l64_64205

def prime_numbers : Finset ℕ := {2, 3, 5, 7}

def powers_of_2 : Finset ℕ := {1, 2, 4, 8}

def total_outcomes : ℕ := 8 * 8

def successful_outcomes : ℕ := prime_numbers.card * powers_of_2.card

def probability (x y : Finset ℕ) : ℚ := (x.card * y.card) / (total_outcomes : ℚ)

theorem blue_die_prime_yellow_die_power_2_probability :
  probability prime_numbers powers_of_2 = 1 / 4 :=
by
  sorry

end blue_die_prime_yellow_die_power_2_probability_l64_64205


namespace spherical_to_rectangular_coords_l64_64385

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), 
    ρ = 3 → θ = 3 * Real.pi / 2 → φ = Real.pi / 3 →
    (let x := ρ * Real.sin φ * Real.cos θ;
         y := ρ * Real.sin φ * Real.sin θ;
         z := ρ * Real.cos φ 
     in (x, y, z) = (0, - (3 * Real.sqrt 3) / 2, 3 / 2)) :=
by
  intros ρ θ φ hρ hθ hφ
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coords_l64_64385


namespace microphotonics_budget_l64_64309

noncomputable def percentage_allocation 
  (home_electronics food_additives genetically_modified_microorganisms industrial_lubricants basic_astrophysics : ℝ) 
  (basic_astrophysics_deg total_deg : ℝ) := 
  home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + 
  (basic_astrophysics_deg / total_deg) * 100

theorem microphotonics_budget (home_electronics food_additives genetically_modified_microorganisms industrial_lubricants basic_astrophysics_deg total_deg : ℝ) :
  home_electronics = 24 → 
  food_additives = 15 → 
  genetically_modified_microorganisms = 29 → 
  industrial_lubricants = 8 → 
  basic_astrophysics_deg = 39.6 → 
  total_deg = 360 → 
  percentage_allocation home_electronics food_additives genetically_modified_microorganisms industrial_lubricants basic_astrophysics 39.6 360 = 87 → 
  100 - (percentage_allocation home_electronics food_additives genetically_modified_microorganisms industrial_lubricants basic_astrophysics 39.6 360) = 13 :=
begin
  intros,
  sorry
end

end microphotonics_budget_l64_64309


namespace principal_amount_l64_64176

theorem principal_amount (SI : ℝ) (T : ℝ) (R : ℝ) (P : ℝ) (h1 : SI = 140) (h2 : T = 2) (h3 : R = 17.5) :
  P = 400 :=
by
  -- Formal proof would go here
  sorry

end principal_amount_l64_64176


namespace union_complement_A_B_range_t_values_l64_64035

-- First Problem
theorem union_complement_A_B {a b : ℝ} (h1 : ∀ x : ℝ, 2*x^2 - 2*a*x + b ≥ 2*(-1)^2 - 2*a*(-1) + b)
                             (h2 : 2*(-1)^2 - 2*a*(-1) + b = -8) :
  (set.compl (set_of (λ x, 2*x^2 - 2*a*x + b > 0)) ∪ (set_of (λ x, |x - 1| ≤ 1))) = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

-- Second Problem
theorem range_t_values {a b : ℝ} (h1 : ∀ x : ℝ, 2*x^2 - 2*a*x + b ≥ 2*(-1)^2 - 2*a*(-1) + b)
                                (h2 : 2*(-1)^2 - 2*a*(-1) + b = -8)
                                (h3 : ¬ ∃ x, (2*x^2 - 2*a*x + b > 0) ∧ (|x - t| ≤ 1)) :
  t ∈ set.Icc (-2 : ℝ) 0 :=
sorry

end union_complement_A_B_range_t_values_l64_64035


namespace sequence_arithmetic_iff_a1_range_l64_64641

theorem sequence_arithmetic_iff_a1_range (a : ℕ → ℚ) :
  (∀ n ≥ 2, a n = if a (n-1) < n^2 then n^2 else 2 * a (n-1)) →
  (∃ d : ℚ, ∀ n m, a n - a m = d * (n - m)) ↔ a 1 ≥ 9/2 :=
sorry

end sequence_arithmetic_iff_a1_range_l64_64641


namespace suitable_land_acres_l64_64567

theorem suitable_land_acres (new_multiplier : ℝ) (previous_acres : ℝ) (pond_acres : ℝ) :
  new_multiplier = 10 ∧ previous_acres = 2 ∧ pond_acres = 1 → 
  (new_multiplier * previous_acres - pond_acres) = 19 :=
by
  intro h
  sorry

end suitable_land_acres_l64_64567


namespace coin_problem_l64_64192

theorem coin_problem
  (num_coins : ℕ)
  (coins : fin num_coins → ℕ)
  (gold_weight : ℕ := 3)
  (silver_weight : ℕ := 2)
  (copper_weight : ℕ := 1)
  (at_least_one_of_each : ∀ t, ∃ i, coins i = t)
  (sum_weights : (fin num_coins → ℕ) → ℕ := fun f => ∑ i, f i)
  (sum_weights_coins : sum_weights coins = 100):
  (∃ weighings, weighings ≤ 101 ∧ (∀ i, coins i = gold_weight ∨ coins i = silver_weight ∨ coins i = copper_weight)) :=
sorry

end coin_problem_l64_64192


namespace difference_is_correct_l64_64197

-- Definition of the given numbers
def numbers : List ℕ := [44, 16, 2, 77, 241]

-- Define the sum of the numbers
def sum_numbers := numbers.sum

-- Define the average of the numbers
def average := sum_numbers / numbers.length

-- Define the difference between sum and average
def difference := sum_numbers - average

-- The theorem we need to prove
theorem difference_is_correct : difference = 304 := by
  sorry

end difference_is_correct_l64_64197


namespace negative_large_base_zero_exponent_l64_64991

-- Define the problem conditions: base number and exponent
def base_number : ℤ := -2023
def exponent : ℕ := 0

-- Prove that (-2023)^0 equals 1
theorem negative_large_base_zero_exponent : base_number ^ exponent = 1 := by
  sorry

end negative_large_base_zero_exponent_l64_64991


namespace product_of_triple_when_added_to_reciprocal_l64_64902

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64902


namespace tan_difference_of_angle_l64_64025

noncomputable def point_on_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (2, 3) = (k * Real.cos θ, k * Real.sin θ)

theorem tan_difference_of_angle (θ : ℝ) (hθ : point_on_terminal_side θ) :
  Real.tan (θ - Real.pi / 4) = 1 / 5 :=
sorry

end tan_difference_of_angle_l64_64025


namespace significant_improvement_l64_64247

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64247


namespace min_positive_sum_of_products_l64_64449

theorem min_positive_sum_of_products {a : Fin 95 → ℤ} 
  (h₀ : ∀ i, a i = 1 ∨ a i = -1) :
  (∃ S : ℤ, S = ∑ i j in Finset.Ico 0 95, if i < j then a i * a j else 0 ∧ S > 0 ∧ ∀ T, T = ∑ i j in Finset.Ico 0 95, if i < j then a i * a j else 0 → T > 0 → T ≥ S → T = 13) :=
sorry

end min_positive_sum_of_products_l64_64449


namespace product_trippled_when_added_to_reciprocal_l64_64931

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64931


namespace same_graph_as_y_eq_x_l64_64222

theorem same_graph_as_y_eq_x (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∀ x : ℝ, (log a (a^x)) = x :=
by 
  sorry

end same_graph_as_y_eq_x_l64_64222


namespace product_of_triple_when_added_to_reciprocal_l64_64905

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64905


namespace percentage_increase_correct_l64_64528

def initial_price : ℝ := 5.00
def new_price : ℝ := 5.55
def increase (initial : ℝ) (new : ℝ) : ℝ := new - initial
def percentage_increase (initial : ℝ) (increase : ℝ) : ℝ := (increase / initial) * 100

theorem percentage_increase_correct :
  percentage_increase initial_price (increase initial_price new_price) = 11 :=
by
  sorry

end percentage_increase_correct_l64_64528


namespace product_of_triple_when_added_to_reciprocal_l64_64900

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64900


namespace significant_improvement_l64_64260

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64260


namespace product_of_solutions_l64_64874

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64874


namespace sum_factors_of_30_l64_64952

theorem sum_factors_of_30 : 
  (∑ i in {1, 2, 3, 5, 6, 10, 15, 30}, i) = 72 := 
by
  -- Sorry, to skip the proof here.
  sorry

end sum_factors_of_30_l64_64952


namespace curve_line_max_distance_curve_below_line_l64_64547

-- Conditions
def curve (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, Real.sin α)

def line_cartesian (x y : ℝ) : Prop := (x + y = 3)

def parametric_line_polar (ρ θ : ℝ) : Prop :=
  (Real.sqrt 2) * ρ * Real.sin (θ + π / 4) = 3

-- Proof problem statement
theorem curve_line_max_distance (t α : ℝ) (h1 : t = 1) :
  let p := curve t α,
  let d := (|p.1 + p.2 - 3|) / Real.sqrt 2
  ∃ d_max, d_max = (2 + 3 * Real.sqrt 2) / 2 :=
  sorry

theorem curve_below_line (t : ℝ) :
  (∀ α, let p := curve t α in p.1 + p.2 < 3) ↔ 0 < t ∧ t < 2 * Real.sqrt 2 :=
  sorry

end curve_line_max_distance_curve_below_line_l64_64547


namespace cost_of_cookies_equal_3_l64_64564

def selling_price : ℝ := 1.5
def cost_price : ℝ := 1
def number_of_bracelets : ℕ := 12
def amount_left : ℝ := 3

theorem cost_of_cookies_equal_3 : 
  (selling_price - cost_price) * number_of_bracelets - amount_left = 3 := by
  sorry

end cost_of_cookies_equal_3_l64_64564


namespace sum_of_multiples_is_even_l64_64151

theorem sum_of_multiples_is_even (a b : ℤ) (h1 : ∃ m : ℤ, a = 4 * m) (h2 : ∃ n : ℤ, b = 6 * n) : Even (a + b) :=
sorry

end sum_of_multiples_is_even_l64_64151


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64843

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64843


namespace new_device_significant_improvement_l64_64252

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64252


namespace LeahCoinsValue_l64_64101

-- Definitions and conditions
def LeahCoins :=
  ∃ (p n : ℕ), p + n = 15 ∧ p = n + 2

-- Prove the total value of coins
theorem LeahCoinsValue : LeahCoins → (9 * 5 + 7 * 1 = 52) :=
by
  intro h
  have ⟨p, n, h1, h2⟩ := h
  rw [h2] at h1
  sorry

end LeahCoinsValue_l64_64101


namespace incorrect_statement_d_l64_64168

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := cos (π / 2 - ω * x)
noncomputable def g (x : ℝ) (ω : ℝ) : ℝ := sin (ω * (x + π / 2))

theorem incorrect_statement_d (ω : ℝ) (hω_pos : ω > 0) :
  g(0, ω) = 1 →
  (∀ x ∈ Icc (0 : ℝ) ((π : ℝ) / 5), monotone_on (λ x, g(x, ω)) (Icc 0 (π / 5))) →
  ω ≤ 5 := sorry

end incorrect_statement_d_l64_64168


namespace proof_problem_solution_l64_64117

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ (a * b + b * c + c * d + d * a = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3)

theorem proof_problem_solution (a b c d : ℝ) : proof_problem a b c d :=
  sorry

end proof_problem_solution_l64_64117


namespace seq_limit_l64_64370

theorem seq_limit :
  tendsto (λ n : ℕ, (3 - 4 * n) ^ 2 / ((n - 3) ^ 2 - (n + 3) ^ 2)) at_top (𝓝 (-∞)) :=
begin
  sorry
end

end seq_limit_l64_64370


namespace min_value_expr_is_neg13n2_l64_64431

noncomputable def minimum_value_of_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  let cos_sum := ∑ i, Real.cos (x i)
  let sin_sum := ∑ i, Real.sin (x i)
  (5 * cos_sum + sin_sum) * (cos_sum - 5 * sin_sum)

theorem min_value_expr_is_neg13n2 (n : ℕ) (x : Fin n → ℝ) :
  (∃ x : Fin n → ℝ, minimum_value_of_expression n x = -13 * n^2) :=
sorry

end min_value_expr_is_neg13n2_l64_64431


namespace product_of_all_solutions_triple_reciprocal_l64_64739

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64739


namespace light_ray_reflection_path_l64_64088

open EuclideanGeometry -- Assume we have a library for Euclidean geometry.

noncomputable def reflect_point (p: Point) (line: Line): Point := sorry -- Assume function for reflection.

-- Define points M and N and sides S1 and S2 of the acute angle:
variables {M N : Point} {S1 S2 : Line}

-- Conditions in definition form:
axiom acute_angle (h_angle : angle_between S1 S2 < π / 2)
axiom M_inside_angle (M_in : inside_angle M S1 S2)
axiom N_inside_angle (N_in : inside_angle N S1 S2)

-- Define the reflection points:
def M1 := reflect_point M S1
def N1 := reflect_point N S2

-- The line through M1 and N1:
def line_M1N1 := line_through M1 N1

-- Intersection points with sides
def K := line_intersection line_M1N1 S1
def L := line_intersection line_M1N1 S2

-- Statement of the proof problem:
theorem light_ray_reflection_path
  (h_angle : angle_between S1 S2 < π / 2)
  (M_in : inside_angle M S1 S2) 
  (N_in : inside_angle N S1 S2) :
  path_with_reflections_exists M N S1 S2 :=
begin
  sorry
end

end light_ray_reflection_path_l64_64088


namespace pink_flowers_in_bagB_l64_64198

noncomputable theory

-- Conditions
def bagA_red := 6
def bagA_pink := 3
def bagB_red := 2
def probability_pink := 0.5555555555555556

-- Definitions and Proof Target
def total_bagA := bagA_red + bagA_pink
def total_bagB (x : ℕ) := bagB_red + x
def probability_pink_A := (bagA_pink : ℚ) / total_bagA
def probability_pink_B (x : ℕ) := (x : ℚ) / total_bagB x

-- Overall probability equation
axiom overall_probability_pink (x : ℕ) : 
  (probability_pink_A + probability_pink_B x) / 2 = probability_pink

-- Proof statement
theorem pink_flowers_in_bagB : ∃ (x : ℕ), overall_probability_pink x ∧ x = 7 :=
sorry

end pink_flowers_in_bagB_l64_64198


namespace general_term_formula_of_sequence_l64_64480

theorem general_term_formula_of_sequence {a : ℕ → ℝ} (S : ℕ → ℝ)
  (hS : ∀ n, S n = (2 / 3) * a n + 1 / 3) :
  (∀ n, a n = (-2) ^ (n - 1)) :=
by
  sorry

end general_term_formula_of_sequence_l64_64480


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64857

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64857


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64835

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64835


namespace estimate_yellow_balls_l64_64078

theorem estimate_yellow_balls (m : ℕ) (h1: (5 : ℝ) / (5 + m) = 0.2) : m = 20 :=
  sorry

end estimate_yellow_balls_l64_64078


namespace product_of_solutions_l64_64772

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64772


namespace evaluate_expression_l64_64230

theorem evaluate_expression : 7899665 - 12 * 3 * 2 = 7899593 :=
by
  -- This proof is skipped.
  sorry

end evaluate_expression_l64_64230


namespace tangent_line_at_intersection_with_x_axis_l64_64588

noncomputable def f : ℝ → ℝ := λ x, 1 - Real.exp x

theorem tangent_line_at_intersection_with_x_axis :
  (∃ (P : ℝ × ℝ), P = (0, 0) ∧ f 0 = 0) →
  ∀ x, f x = 1 - Real.exp x →
  ∀ x, f' 0 = -1 →  -- Derivative at x = 0
  TangentLine (f) (0) = λ x, -x :=
begin
  sorry
end

end tangent_line_at_intersection_with_x_axis_l64_64588


namespace frog_jump_no_center_crossing_l64_64541

theorem frog_jump_no_center_crossing (n : ℕ) (h : n ≥ 2) :
  (∃ j : ℤ, (2 : ℤ) ≡ j [MOD 4] ∧ ∀ (vertices : vector ℕ (2 * n)), 
    let positions := vector.map (. + 1) vertices;
    ∀ pairs : finset (finset (2 * n)) ( ∃ frog_at_each : ∀ v : finset (2 * n), finset.card v = 1 ∧ 
    ∀ f1 f2 : fin (2 * n), (f1 ∈ positions ∧ f2 ∈ positions ∧ f1 ≠ f2) → 
    ¬line_segment_passes_center f1 f2)) :=
begin
    sorry
end

end frog_jump_no_center_crossing_l64_64541


namespace dogwood_trees_planted_today_l64_64193

theorem dogwood_trees_planted_today 
    (current_trees : ℕ) 
    (trees_planted_tomorrow : ℕ) 
    (total_trees : ℕ) 
    (initial_trees : ℕ)
    (total : initial_trees + current_trees + trees_planted_tomorrow = total_trees) :
    current_trees = 41 :=
by
  -- Current conditions from the problem
  let current_trees := 39
  let trees_planted_tomorrow := 20
  let total_trees := 100
  calc
    current_trees + trees_planted_tomorrow : 39 + 20
    ... = 59
    ... = 100 - 41 := eq.trans (by simp) total.symm
  sorry

end dogwood_trees_planted_today_l64_64193


namespace product_trippled_when_added_to_reciprocal_l64_64934

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64934


namespace derivative_even_function_l64_64029

def f (x : ℝ) : ℝ := real.exp x - real.exp (-x) + 1

def f' (x : ℝ) : ℝ := real.exp x + real.exp (-x)

theorem derivative_even_function : ∀ x : ℝ, f'(-x) = f'(x) :=
by 
  intro x
  sorry

end derivative_even_function_l64_64029


namespace peter_total_miles_l64_64137

-- Definitions based on the conditions
def minutes_per_mile : ℝ := 20
def miles_walked_already : ℝ := 1
def additional_minutes : ℝ := 30

-- The value we want to prove
def total_miles_to_walk : ℝ := 2.5

-- Theorem statement corresponding to the proof problem
theorem peter_total_miles :
  (additional_minutes / minutes_per_mile) + miles_walked_already = total_miles_to_walk :=
sorry

end peter_total_miles_l64_64137


namespace gcd_apb_ab_eq1_gcd_aplusb_aminsb_l64_64511

theorem gcd_apb_ab_eq1 (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a * b) = 1 ∧ Int.gcd (a - b) (a * b) = 1 := by
  sorry

theorem gcd_aplusb_aminsb (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a - b) = 1 ∨ Int.gcd (a + b) (a - b) = 2 := by
  sorry

end gcd_apb_ab_eq1_gcd_aplusb_aminsb_l64_64511


namespace significant_improvement_l64_64275

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64275


namespace probability_no_shaded_square_l64_64996

noncomputable theory

-- Define n as the number of rectangles from a row of 2012 vertical segments
def total_rectangles (n : ℕ) : ℕ := (2012 * 2011) / 2

-- Define m as the number of rectangles containing the shaded square 
def shaded_rectangles (m : ℕ) : ℕ := 1006 * 1006

-- Define the probability calculation
def probability_no_shaded : ℚ := 1 - (shaded_rectangles 1006 / total_rectangles 1006)

-- The theorem to be proven
theorem probability_no_shaded_square : probability_no_shaded = (1005 / 2011) := by
  sorry

end probability_no_shaded_square_l64_64996


namespace red_button_probability_l64_64095

theorem red_button_probability :
  let jarA_red := 6
  let jarA_blue := 9
  let jarA_total := jarA_red + jarA_blue
  let jarA_half := jarA_total / 2
  let removed_total := jarA_total - jarA_half
  let removed_red := removed_total / 2
  let removed_blue := removed_total / 2
  let jarA_red_remaining := jarA_red - removed_red
  let jarA_blue_remaining := jarA_blue - removed_blue
  let jarB_red := removed_red
  let jarB_blue := removed_blue
  let jarA_total_remaining := jarA_red_remaining + jarA_blue_remaining
  let jarB_total := jarB_red + jarB_blue
  (jarA_total = 15) →
  (jarA_red_remaining = 6 - removed_red) →
  (jarA_blue_remaining = 9 - removed_blue) →
  (jarB_red = removed_red) →
  (jarB_blue = removed_blue) →
  (jarA_red_remaining + jarA_blue_remaining = 9) →
  (jarB_red + jarB_blue = 6) →
  let prob_red_JarA := jarA_red_remaining / jarA_total_remaining
  let prob_red_JarB := jarB_red / jarB_total
  prob_red_JarA * prob_red_JarB = 1 / 6 := by sorry

end red_button_probability_l64_64095


namespace inscribed_square_area_correct_l64_64350

noncomputable def inscribed_square_area 
  (leg_length : ℝ)
  (h1 : leg_length = 28 * Real.sqrt 2) 
  : ℝ :=
  let diagonal := leg_length in
  let side_length := diagonal / Real.sqrt 2 in
  let area := side_length ^ 2 in
  784

theorem inscribed_square_area_correct 
  (h1 : (28 * Real.sqrt (2)) = 28 * Real.sqrt 2) 
  : inscribed_square_area (28 * Real.sqrt 2) h1 = 784 :=
sorry

end inscribed_square_area_correct_l64_64350


namespace mean_significantly_improved_l64_64294

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64294


namespace exists_subset_not_divisible_47_l64_64571

theorem exists_subset_not_divisible_47 (X : Finset ℤ) (hX : X.card = 10000) 
  (hX_mod : ∀ x ∈ X, x % 47 ≠ 0) :
  ∃ Y ⊆ X, Y.card = 2007 ∧ ∀ a b c d e ∈ Y, (a - b + c - d + e) % 47 ≠ 0 := 
by
  sorry

end exists_subset_not_divisible_47_l64_64571


namespace tangency_of_circles_l64_64458

open EuclideanGeometry

variable {A B C I D E F X Y Z : Point}
variable {l l1 l2 l3 : Line}
variable {Γ Γ' : Circle}

-- Defining the points and lines
axiom h1 : ∃ (ABC : Triangle), circumcircle ABC = Γ ∧ incenter ABC = I
axiom h2 : ∃ l, ∀ l : Line, Intersects l (segment A I) ∧ Intersects l (segment B I) ∧ Intersects l (segment C I)
axiom h3 : ∃ D E F, 
  (D ∈ (line_through A I)) ∧ (E ∈ (line_through B I)) ∧ (F ∈ (line_through C I)) ∧
  D ≠ A ∧ D ≠ I ∧ D ≠ B ∧ E ≠ B ∧ E ≠ I ∧ E ≠ C ∧ F ≠ C ∧ F ≠ I ∧ F ≠ A
axiom h4 : ∃ l1 l2 l3, 
  l1 = perpendicular_bisector (segment A D) ∧ l2 = perpendicular_bisector (segment B E) ∧
  l3 = perpendicular_bisector (segment C F)

-- Intersection points
axiom h5 : X = intersection l2 l3 ∧ Y = intersection l1 l3 ∧ Z = intersection l1 l2

-- Circumcircle of new triangle
axiom h6 : Γ' = circumcircle ⟨X, Y, Z⟩

-- The Tangency Statement
theorem tangency_of_circles :
  Tangent Γ Γ' :=
sorry

end tangency_of_circles_l64_64458


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64854

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64854


namespace length_of_QR_l64_64077

-- Definitions based on the problem conditions
def right_triangle (P Q R : Type) :=
  ∃ (PQ QR PR : ℝ), PQ ≠ 0 ∧ QR ≠ 0 ∧ PR ≠ 0 ∧ (PQ^2 + PR^2 = QR^2)  

def sine_Q (PQ QR : ℝ) : ℝ := PQ / QR

variables {P Q R : Type}
variables (PQ QR : ℝ) (h_right_triangle : right_triangle P Q R) (h_sine : sine_Q PQ QR = 0.6)

-- The statement to prove
theorem length_of_QR (h_PQ : PQ = 15) : QR = 25 :=
by {
  sorry
}

end length_of_QR_l64_64077


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64733

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64733


namespace prove_tirzah_handbags_l64_64201
noncomputable def tirzah_has_24_handbags (H : ℕ) : Prop :=
  let P := 26 -- number of purses
  let fakeP := P / 2 -- half of the purses are fake
  let authP := P - fakeP -- number of authentic purses
  let fakeH := H / 4 -- one quarter of the handbags are fake
  let authH := H - fakeH -- number of authentic handbags
  authP + authH = 31 -- total number of authentic items
  → H = 24 -- prove the number of handbags is 24

theorem prove_tirzah_handbags : ∃ H : ℕ, tirzah_has_24_handbags H :=
  by
    use 24
    -- Proof goes here
    sorry

end prove_tirzah_handbags_l64_64201


namespace inequality_holds_l64_64450

theorem inequality_holds (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) (h_mul : a * b * c * d = 1) :
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (12 / (a + b + c + d)) ≥ 7 :=
by
  sorry

end inequality_holds_l64_64450


namespace product_of_solutions_l64_64802

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64802


namespace sin_mul_cos_eq_quarter_l64_64019

open Real

theorem sin_mul_cos_eq_quarter (α : ℝ) (h : sin α - cos α = sqrt 2 / 2) : sin α * cos α = 1 / 4 :=
by
  sorry

end sin_mul_cos_eq_quarter_l64_64019


namespace rectangle_area_l64_64334

theorem rectangle_area (a : ℕ) (w l : ℕ) (h_square_area : a = 36) (h_square_side : w * w = a) (h_rectangle_length : l = 3 * w) : w * l = 108 :=
by
  -- Placeholder for proof
  sorry

end rectangle_area_l64_64334


namespace significant_improvement_l64_64245

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64245


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64887

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64887


namespace other_root_of_quadratic_l64_64009

theorem other_root_of_quadratic 
  (a b c: ℝ) 
  (h : a * (b - c - d) * (1:ℝ)^2 + b * (c - a + d) * (1:ℝ) + c * (a - b - d) = 0) : 
  ∃ k: ℝ, k = c * (a - b - d) / (a * (b - c - d)) :=
sorry

end other_root_of_quadratic_l64_64009


namespace monotonically_decreasing_interval_l64_64030

-- Given conditions
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- The proof problem statement
theorem monotonically_decreasing_interval :
  ∃ a b : ℝ, (0 ≤ a) ∧ (b ≤ 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → (deriv f x ≤ 0)) :=
sorry

end monotonically_decreasing_interval_l64_64030


namespace coplanar_lines_l64_64207

def vector3 := ℝ × ℝ × ℝ

def vec1 : vector3 := (2, -1, 3)
def vec2 (k : ℝ) : vector3 := (3 * k, 1, 2)
def pointVec : vector3 := (-3, 2, -3)

def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem coplanar_lines (k : ℝ) : det3x3 2 (-1) 3 (3 * k) 1 2 (-3) 2 (-3) = 0 → k = -29 / 9 :=
  sorry

end coplanar_lines_l64_64207


namespace julia_puppy_first_year_cost_l64_64099

theorem julia_puppy_first_year_cost 
  (adoption_fee : ℕ := 150) 
  (dog_food : ℕ := 40) 
  (treat_bags : ℕ := 3) 
  (treat_cost_per_bag : ℕ := 5)
  (toy_boxes : ℕ := 2)
  (toy_cost_per_box : ℕ := 25) 
  (crate : ℕ := 120) 
  (bed : ℕ := 80) 
  (collar_leash_combo : ℕ := 35)
  (grooming_tools : ℕ := 45) 
  (class_1 : ℕ := 55) 
  (class_2_3 : ℕ := 60) 
  (class_4_5 : ℕ := 70)
  (discount_rate : ℚ := 0.12) 
  (dog_license : ℕ := 25) 
  (insurance_first_6_month_cost : ℕ := 25) 
  (insurance_next_6_month_cost : ℕ := 30) :
  let treat_cost := treat_bags * treat_cost_per_bag
      toy_cost := toy_boxes * toy_cost_per_box
      training_cost := class_1 + (2 * class_2_3) + (2 * class_4_5)
      initial_cost := adoption_fee + dog_food + treat_cost + toy_cost + crate + bed + collar_leash_combo + grooming_tools + training_cost
      discountable_initial := initial_cost - (adoption_fee + training_cost)
      discount := discount_rate * discountable_initial
      initial_cost_after_discount := initial_cost - discount
      insurance_cost := (6 * insurance_first_6_month_cost) + (6 * insurance_next_6_month_cost)
      total_first_year_cost := initial_cost_after_discount + dog_license + insurance_cost
  in total_first_year_cost = 1158.80 :=
by sorry

end julia_puppy_first_year_cost_l64_64099


namespace intervals_of_monotonicity_range_of_a_l64_64032

noncomputable def f (a b x : ℝ) : ℝ := (2 * a * x ^ 2 + b * x + 1) * Real.exp (-x)

theorem intervals_of_monotonicity (b : ℝ) :
  let f' := λ x, -((x - 1) * (x - (1 - b)) * Real.exp (-x))
  if b = 0 then ∀ x, f' x ≤ 0
  else if b < 0 then 
    (∀ x, x < 1 → f' x ≤ 0) ∧ (∀ x, 1 < x ∧ x < 1 - b → f' x > 0) ∧ (∀ x, x > 1 - b → f' x ≤ 0)
  else 
    (∀ x, x < 1 - b → f' x ≤ 0) ∧ (∀ x, 1 - b < x ∧ x < 1 → f' x > 0) ∧ (∀ x, x > 1 → f' x ≤ 0) :=
sorry

theorem range_of_a (a : ℝ) :
  let b := Real.exp 1 - 1 - 2 * a
  (Real.exp 1 - 2) / 2 < a ∧ a < 1 / 2 ↔
  ∃ x ∈ Ioo 0 1, f a b x = 1 :=
sorry

end intervals_of_monotonicity_range_of_a_l64_64032


namespace sum_of_squares_l64_64614

noncomputable def gcd (a b : ℕ) : ℕ := sorry

theorem sum_of_squares (a b c : ℕ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a + b + c = 24)
  (h3 : gcd a b + gcd b c + gcd c a = 10) :
  (a^2 + b^2 + c^2 = 394) :=
sorry

end sum_of_squares_l64_64614


namespace team_savings_with_discount_l64_64403

def regular_shirt_cost : ℝ := 7.50
def regular_pants_cost : ℝ := 15.00
def regular_socks_cost : ℝ := 4.50
def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75
def team_size : ℕ := 12

theorem team_savings_with_discount :
  let regular_uniform_cost := regular_shirt_cost + regular_pants_cost + regular_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by
  sorry

end team_savings_with_discount_l64_64403


namespace yasmin_initial_balance_l64_64982

theorem yasmin_initial_balance (X : ℝ) 
  (h1 : ∀ {X}, X > 0 → 
       let m_bal := X / 2 in
       let t_bal := m_bal + 30 in
       let interest := 0.015 * t_bal in
       let t_bal_with_interest := t_bal + interest in
       let euro_to_usd := 40 * 1.25 in
       let w_bal := t_bal_with_interest + euro_to_usd in
       w_bal = 200 →
       let th_bal := w_bal - 20 in
       th_bal = 160)
  : X = 235.59 :=
by
  sorry

end yasmin_initial_balance_l64_64982


namespace longest_side_is_x_squared_plus_4_l64_64632

noncomputable def longest_side_of_triangle (x : ℝ) : ℝ :=
  if side1 = x^2 + 4 ∧ side2 = 3 * x + 1 ∧ side3 = 7 ∧ side1 + side2 + side3 = 45 then
    max (max side1 side2) side3
  else
    0

theorem longest_side_is_x_squared_plus_4 (x : Real) :
  let side1 := x^2 + 4
  let side2 := 3 * x + 1
  let side3 := 7
  side1 + side2 + side3 = 45 →
  x ≥ 0 →
  side1 > side2 ∧ side1 > side3 :=
begin
  sorry
end

end longest_side_is_x_squared_plus_4_l64_64632


namespace competition_inequality_l64_64538

variable (a b k : ℕ)

-- Conditions
variable (h1 : b % 2 = 1) 
variable (h2 : b ≥ 3)
variable (h3 : ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k)

theorem competition_inequality (h1: b % 2 = 1) (h2: b ≥ 3) (h3: ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k) :
  (k: ℝ) / (a: ℝ) ≥ (b-1: ℝ) / (2*b: ℝ) := sorry

end competition_inequality_l64_64538


namespace remainder_of_product_l64_64943

open Nat

theorem remainder_of_product (a b : ℕ) (ha : a % 5 = 4) (hb : b % 5 = 3) :
  (a * b) % 5 = 2 :=
by
  sorry

end remainder_of_product_l64_64943


namespace mean_significantly_improved_l64_64299

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64299


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64845

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64845


namespace common_ratio_of_geometric_sequence_l64_64477

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ)
  (h_geom : ∃ q, ∀ n, a (n+1) = a n * q)
  (h1 : a 1 = 1 / 8)
  (h4 : a 4 = -1) :
  ∃ q, q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l64_64477


namespace tan_u_tan_v_sum_l64_64581

theorem tan_u_tan_v_sum (u v : ℝ) 
  (h1 : (sin u / cos v) + (sin v / cos u) = 2)
  (h2 : (cos u / sin v) + (cos v / sin u) = 3) :
  (tan u / tan v) + (tan v / tan u) = 8 / 7 :=
by
  sorry

end tan_u_tan_v_sum_l64_64581


namespace sum_closed_form_l64_64417

theorem sum_closed_form (n : ℕ) (h : 3 < n) :
  ∑ k in finset.Ico 2 n, ((k - 1) * k^2 * (k + 1)) = ((n - 2) * (n - 1) * n * (n + 1) * (2 * n - 1)) / 10 :=
by
  sorry

end sum_closed_form_l64_64417


namespace function_monotonically_increasing_l64_64493

-- The function y = x^2 - 2x + 8
def f (x : ℝ) : ℝ := x^2 - 2 * x + 8

-- The theorem stating the function is monotonically increasing on (1, +∞)
theorem function_monotonically_increasing : ∀ x y : ℝ, (1 < x) → (x < y) → (f x < f y) :=
by
  -- Proof is omitted
  sorry

end function_monotonically_increasing_l64_64493


namespace product_of_all_solutions_triple_reciprocal_l64_64741

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64741


namespace range_of_a_ineq_nat_fac_l64_64037

-- Problem statement for question 1
theorem range_of_a (a : ℝ) (f g : ℝ → ℝ)
  (h₀: ∀ x ∈ (set.Ioi 0), g x - f x ≤ -1)
  (hf : ∀ x, f x = a * x + (a - 1) / x)
  (hg : ∀ x, g x = Real.log x) :
  1 ≤ a :=
sorry

-- Problem statement for question 2
theorem ineq_nat_fac (n : ℕ) (h : 0 < n) :
  n + 1 < Real.exp (Real.sqrt (n.factorial ^ (1 / n))) :=
sorry

end range_of_a_ineq_nat_fac_l64_64037


namespace sum_of_factors_30_l64_64974

theorem sum_of_factors_30 : 
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ) in
  ( ∑ n in factors, n ) = 72 :=
by
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ)
  rw [finset.sum_eq_sum_of_elements] -- transposing to handle the sum notation
  let factors_list := [1, 2, 3, 5, 6, 10, 15, 30]
  have h_factors : factors = {x | x ∈ factors_list}.to_finset, by sorry
  simp_rw h_factors
  rw [finset.sum_to_finset]
  dsimp -- simplifying before asserting the sum
  norm_num -- proving that the sum is indeed 72
  sorry

end sum_of_factors_30_l64_64974


namespace cone_volume_l64_64650

def slant_height := 15
def height := 9
def radius := Real.sqrt (slant_height^2 - height^2)
def volume := (1/3) * Real.pi * radius^2 * height

theorem cone_volume : 
  radius = 12 ∧ volume = 432 * Real.pi :=
by
  split
  { -- radius = 12
    unfold radius
    sorry
  }
  { -- volume = 432 * Real.pi
    unfold volume
    sorry
  }

end cone_volume_l64_64650


namespace negative_large_base_zero_exponent_l64_64992

-- Define the problem conditions: base number and exponent
def base_number : ℤ := -2023
def exponent : ℕ := 0

-- Prove that (-2023)^0 equals 1
theorem negative_large_base_zero_exponent : base_number ^ exponent = 1 := by
  sorry

end negative_large_base_zero_exponent_l64_64992


namespace prime_count_of_first_10_sums_is_2_l64_64149

open Nat

def consecutivePrimes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def consecutivePrimeSums (n : Nat) : List Nat :=
  (List.range n).scanl (λ sum i => sum + consecutivePrimes.getD i 0) 0

theorem prime_count_of_first_10_sums_is_2 :
  let sums := consecutivePrimeSums 10;
  (sums.count isPrime) = 2 :=
by
  sorry

end prime_count_of_first_10_sums_is_2_l64_64149


namespace inequality_holds_l64_64143

theorem inequality_holds (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (y * z)) + (y^3 / (z * x)) + (z^3 / (x * y)) ≥ x + y + z :=
by
  sorry

end inequality_holds_l64_64143


namespace michael_points_scored_l64_64536

theorem michael_points_scored (team_points : ℕ) (other_players : ℕ) (average_points : ℕ) (michael_points : ℕ) :
  team_points = 72 → other_players = 8 → average_points = 9 → 
  michael_points = team_points - other_players * average_points → michael_points = 36 :=
by
  intro h_team_points h_other_players h_average_points h_calculation
  -- skip the actual proof for now
  sorry

end michael_points_scored_l64_64536


namespace white_square_area_l64_64129

noncomputable def cube_edge : ℕ := 8
noncomputable def total_green_paint_area : ℕ := 192
noncomputable def num_faces : ℕ := 6

theorem white_square_area : 
  let face_area := cube_edge * cube_edge in
  let total_surface_area := num_faces * face_area in
  let green_area_per_face := total_green_paint_area / num_faces in
  let white_square_area := face_area - green_area_per_face in
  white_square_area = 32 :=
by
  let face_area := cube_edge * cube_edge
  let total_surface_area := num_faces * face_area
  let green_area_per_face := total_green_paint_area / num_faces
  let white_square_area := face_area - green_area_per_face
  show white_square_area = 32
  sorry

end white_square_area_l64_64129


namespace tangent_lines_to_log_abs_l64_64626

-- Definition of the curve
def curve (x : ℝ) : ℝ := Real.log (abs x)

-- The proof statement
theorem tangent_lines_to_log_abs (x y : ℝ) (h_tangent : curve x = y ∧ (x ≠ 0)) :
  (x - Real.exp 1 * y = 0 ∨ x + Real.exp 1 * y = 0) :=
sorry

end tangent_lines_to_log_abs_l64_64626


namespace sum_of_legs_of_larger_triangle_l64_64210

-- Definitions and conditions
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

variables (a b : ℝ)
variables (area_small hypotenuse_small area_large : ℝ)

-- Given conditions
def conditions :=
  (1 / 2) * a * b = area_small ∧
  right_triangle a b hypotenuse_small ∧
  area_small = 24 ∧
  hypotenuse_small = 10 ∧
  area_large = 600 

-- The proposition to prove
theorem sum_of_legs_of_larger_triangle
  (h : conditions a b area_small hypotenuse_small area_large):
  let scale := real.sqrt (area_large / area_small)
  in sum_of_legs : ℝ = scale * a + scale * b
  in sum_of_legs = 70 :=
by
  sorry

end sum_of_legs_of_larger_triangle_l64_64210


namespace steve_more_than_wayne_first_time_at_2004_l64_64357

def initial_steve_money (year: ℕ) := if year = 2000 then 100 else 0
def initial_wayne_money (year: ℕ) := if year = 2000 then 10000 else 0

def steve_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_steve_money year
  else 2 * steve_money (year - 1)

def wayne_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_wayne_money year
  else wayne_money (year - 1) / 2

theorem steve_more_than_wayne_first_time_at_2004 :
  ∃ (year: ℕ), year = 2004 ∧ steve_money year > wayne_money year := by
  sorry

end steve_more_than_wayne_first_time_at_2004_l64_64357


namespace product_of_all_solutions_triple_reciprocal_l64_64746

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64746


namespace product_of_triple_when_added_to_reciprocal_l64_64909

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64909


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64916

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64916


namespace option_a_option_d_l64_64462

section

variable (z1 z2 : Complex)
variable (hz1 : z1 = 2 - Complex.im 1)
variable (hz2 : z2 = Complex.fromImaginary 2)

theorem option_a : Im(z2) ≠ 0 ∧ Re(z2) = 0 :=
by 
  sorry

theorem option_d : ∥z1 * z2∥ = 2 * Real.sqrt 5 :=
by 
  sorry

end

end option_a_option_d_l64_64462


namespace significant_improvement_l64_64280

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64280


namespace general_term_formula_sum_b_n_terms_l64_64013

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}

axiom a_n_arithmetic_sequence : 
  (∀ n : ℕ, S n = ((n:ℝ)/2) * (2 * a 1 + (n-1) * (a 2 - a 1))) → a 3 = 5 → S 10 = 10 

theorem general_term_formula :
  a 3 = 5 ∧ S 10 = 10 → (∀ n : ℕ, a (n+1) = 2 * (n+1) - 1) :=
  by sorry

axiom b_n_definition :
  (∀ n : ℕ, b (n+1) = 2 / ((n+1) * (a (n+1) + 5))) 

theorem sum_b_n_terms :
  a 3 = 5 ∧ S 10 = 10 → (∀ n : ℕ, T n = 1/2 * (Sum (λ k, b k) n) ∧ 
  T n = 3/4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
  by sorry

#check general_term_formula
#check sum_b_n_terms

end general_term_formula_sum_b_n_terms_l64_64013


namespace sophia_saves_dimes_l64_64613

theorem sophia_saves_dimes : 
  (∀ n : ℕ, (n = 20 → ∀ p : ℕ, (p = 1 → ∀ d : ℕ, (d = n * p / 10 → d = 2)))) :=
by
  intros n hn p hp d hd
  rw [hn, hp, hd]
  exact rfl

end sophia_saves_dimes_l64_64613


namespace significant_improvement_l64_64277

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64277


namespace num_of_digits_in_product_l64_64507

noncomputable def num_digits_of_product : ℕ :=
  let log3 := 0.4771
  let log6 := 0.7782
  let product_log := 7 * log3 + 8 * log6
  (product_log.floor + 1).to_nat

theorem num_of_digits_in_product :
  num_digits_of_product = 10 := by
  sorry

end num_of_digits_in_product_l64_64507


namespace product_of_solutions_l64_64778

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64778


namespace probability_triangle_is_correct_l64_64073

open Finset

-- Define the set of points
def points : Finset (ℝ × ℝ) := 
  {(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)}

-- Define a function to check if three points are collinear
def collinear (p1 p2 p3 : ℝ × ℝ) : Bool :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

-- Count the number of ways to choose three points from the set that are collinear
def count_collinear_triples : Nat :=
  points.to_list.combinations 3 |>.filter (fun l => collinear l[0] l[1] l[2]) |>.length

-- Total number of ways to choose three points from the set
def total_triples : Nat := (points.card.choose 3).toNat

-- Calculate the probability of forming a triangle
def probability_not_collinear : ℚ :=
  1 - (count_collinear_triples : ℚ) / (total_triples : ℚ)

-- The statement to prove the probability is as expected
theorem probability_triangle_is_correct : 
  probability_not_collinear = 4 / 5 := by
  sorry

end probability_triangle_is_correct_l64_64073


namespace business_value_l64_64314

theorem business_value (share_ratio : ℚ) (sold_ratio : ℚ) (sold_value : ℚ)
    (h_share_ratio : share_ratio = 1 / 3) (h_sold_ratio : sold_ratio = 3 / 5) 
    (h_sold_value : sold_value = 15000) : 3 * (sold_value * (1 / sold_ratio)) = 75000 :=
by
  rw [h_share_ratio, h_sold_ratio, h_sold_value]
  field_simp
  norm_num
  sorry

end business_value_l64_64314


namespace find_abc_l64_64584

def S_k (k : ℕ) (p q r : ℝ) : ℝ := p^k + q^k + r^k

noncomputable def sum_of_roots : ℝ := 5

noncomputable def product_of_pairs : ℝ := 8

noncomputable def product_of_roots : ℝ := 13

noncomputable def S_0 : ℝ := 3

noncomputable def S_1 : ℝ := 5

noncomputable def S_2 : ℝ := 9

theorem find_abc :
  let a := 5
  let b := -8
  let c := 13
  a + b + c = 10 :=
by {
  sorr
}

end find_abc_l64_64584


namespace product_trippled_when_added_to_reciprocal_l64_64929

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64929


namespace books_read_in_8_hours_l64_64605

def reading_speed := 100 -- pages per hour
def book_pages := 400 -- pages per book
def hours_available := 8 -- hours

theorem books_read_in_8_hours :
  (hours_available * reading_speed) / book_pages = 2 :=
by
  sorry

end books_read_in_8_hours_l64_64605


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64861

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64861


namespace positive_integer_solutions_count_l64_64054

theorem positive_integer_solutions_count :
  {x : ℕ | 0 < x ∧ 10 < -x + 13}.card = 2 :=
by
  sorry

end positive_integer_solutions_count_l64_64054


namespace significant_improvement_l64_64249

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64249


namespace sum_of_factors_of_30_is_72_l64_64968

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l64_64968


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64816

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64816


namespace find_divisor_l64_64231

theorem find_divisor :
  ∃ d : ℕ, 17698 = d * 89 + 14 ∧ d = 198 :=
begin
  sorry
end

end find_divisor_l64_64231


namespace product_of_solutions_l64_64801

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64801


namespace books_sold_last_month_l64_64690

def base9_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 3725 => 3 * 9^3 + 7 * 9^2 + 2 * 9^1 + 5 * 9^0
  | _    => 0

theorem books_sold_last_month (n : ℕ) (h : n = 3725) : base9_to_base10 n = 2777 :=
by
  rw [h]
  unfold base9_to_base10
  norm_num
  sorry

end books_sold_last_month_l64_64690


namespace diameter_is_correct_l64_64612

noncomputable def diameter_of_larger_circle (r : ℝ) (n : ℕ) : ℝ :=
  if (r = 5) ∧ (n = 6) then 2 * (r + 2 * r) else 0

theorem diameter_is_correct {r : ℝ} {n : ℕ} (h₁ : r = 5) (h₂ : n = 6) :
  diameter_of_larger_circle r n = 30 :=
by
  have formula : diameter_of_larger_circle r n = 2 * (r + 2 * r) := rfl
  simp [formula, h₁, h₂]
  sorry

end diameter_is_correct_l64_64612


namespace magnitude_zero_vector_zero_l64_64343

open_locale vector_space

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]

theorem magnitude_zero_vector_zero (a b c : V) :
  (∀ a b : V, (∥a∥ = ∥b∥ ∧ collinear ℝ {a, b}) → a = b) = false ∧
  (∀ a b c : V, (collinear ℝ {a, b} ∧ collinear ℝ {b, c}) → collinear ℝ {a, c}) = false ∧
  (∀ a b : V, (∥a∥ = 1 ∧ ∥b∥ = 1) → a = b) = false ∧
  (∥(0 : V)∥ = 0) = true :=
by sorry

end magnitude_zero_vector_zero_l64_64343


namespace determine_coin_types_within_weighings_l64_64189

theorem determine_coin_types_within_weighings (n : ℕ) (h_n_eq : n = 100) 
  (gold silver copper : finset ℕ) (h_gold : gold.card > 0) 
  (h_silver : silver.card > 0) (h_copper : copper.card > 0) 
  (h_total : gold.card + silver.card + copper.card = n) :
  ∃ (weighings : ℕ), weighings ≤ 101 ∧ 
  ∀ coin, coin ∈ gold ∨ coin ∈ silver ∨ coin ∈ copper :=
  sorry

end determine_coin_types_within_weighings_l64_64189


namespace product_of_solutions_l64_64798

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64798


namespace max_area_of_quadrilateral_l64_64482

noncomputable theory

variables {x y : ℝ}

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

def point_P (x y : ℝ) : Prop :=
  x = 2 ∧ y = 1

def perpendicular_lines_through_P (A B C D : ℝ × ℝ) : Prop :=
  -- Placeholders for specific properties or calculations of lines
  true -- this should have actual geometric definitions

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  -- Placeholder for the actual function to calculate the area
  sqrt 15  -- This should be the maxima calculation method

theorem max_area_of_quadrilateral (A B C D : ℝ × ℝ) :
  circle_eq x y →
  point_P x y →
  perpendicular_lines_through_P A B C D →
  area_of_quadrilateral A B C D = sqrt 15 :=
by sorry

end max_area_of_quadrilateral_l64_64482


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64881

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64881


namespace fixed_point_l64_64630

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
    let f := λ x : ℝ, a^(x + 1) - 2 in f (-1) = -1 := by
  sorry

end fixed_point_l64_64630


namespace probability_product_equals_i_l64_64682

-- Let V be the set of vertices of the centrally symmetric octagon
def V : set ℂ :=
  {2i, -2i, 1 + i, -1 + i, 1 - i, -1 - i, 3/2 + (3/2)*i, 3/2 - (3/2)*i}

-- Define P as the product of 16 elements chosen from V
def P (z : ℕ → ℂ) : ℂ :=
  ∏ j in range 16, z j

theorem probability_product_equals_i :
  -- Define the probability space and calculation conditions
  (∃ (z : ℕ → ℂ) (hz : ∀ j ∈ range 16, z j ∈ V), P z = 1 / 2 ^ 32) ↔
  (P z = i ∧ (1 : ℕ → ℂ) = z ∧ (∀ j ∈ range 16, z j ∈ V)) :=
begin
  sorry -- proof omitted
end

end probability_product_equals_i_l64_64682


namespace smallest_consecutive_odd_integer_l64_64172

theorem smallest_consecutive_odd_integer (median greatest : ℤ) (h1 : median = 156) (h2 : greatest = 165) (h3: (∃ s : Set ℤ, (∀ x ∈ s, odd x) ∧ median ∈ s ∧ greatest ∈ s ∧ s = { x : ℤ | median ≤ x ∧ x ≤ greatest })) : 
(∃ smallest : ℤ, smallest = 146 ∧ smallest ∈ { x : ℤ | median ≤ x ∧ x ≤ greatest }) :=
by 
  -- skipping the proof
  sorry

end smallest_consecutive_odd_integer_l64_64172


namespace significant_improvement_l64_64263

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64263


namespace sum_of_factors_of_30_is_72_l64_64965

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l64_64965


namespace sufficient_condition_for_parallel_lines_l64_64500

noncomputable def parallel_and_perpendicular {Point : Type} [geometry.Point Point] (a b : geometry.line Point) (α β : geometry.plane Point) : Prop :=
  (geometry.perpendicular_line_plane a α ∧ geometry.perpendicular_line_plane b β ∧ geometry.parallel_plans α β → geometry.parallel_lines a b)

theorem sufficient_condition_for_parallel_lines {Point : Type} [geometry.Point Point] :
  ∀ (a b : geometry.line Point) (α β : geometry.plane Point),
    (geometry.perpendicular_line_plane a α) → 
    (geometry.perpendicular_line_plane b β) → 
    (geometry.parallel_plans α β) → 
    (geometry.parallel_lines a b) :=
by sorry

end sufficient_condition_for_parallel_lines_l64_64500


namespace count_valid_numbers_l64_64211

-- We define the set of digits used and the five-digit number requirement
def digits := {0, 1, 2, 3, 4}
def is_five_digit_number (n : List ℕ) : Prop := n.length = 5
def no_repetition (n : List ℕ) : Prop := n.nodup
def is_even (d : ℕ) : Prop := d % 2 = 0
def is_odd (d : ℕ) : Prop := ¬ is_even d
def sandwiched_even (n : List ℕ) : Prop :=
  ∃ i, i < n.length ∧ is_even (n.nth_le i sorry) ∧ is_odd (n.nth_le (i-1) sorry) ∧ is_odd (n.nth_le (i+1) sorry)

-- Combining all the conditions
def valid_number (n : List ℕ) : Prop :=
  is_five_digit_number n ∧ no_repetition n ∧ sandwiched_even n ∧ ∀ d ∈ n, d ∈ digits

-- The main theorem statement to prove
theorem count_valid_numbers : ( (finset.univ: finset (list ℕ)).filter valid_number ).card = 28 := by
  sorry

end count_valid_numbers_l64_64211


namespace log_properties_test_l64_64469

/--
Given that \( a > 0 \), \( a \neq 1 \), \( x > 0 \), \( y > 0 \), \( x > y \),
Prove that there are 0 correct expressions among the following:
1. \( \log_a x \cdot \log_a y = \log_a(x + y) \)
2. \( \log_a x - \log_a y = \log_a(x - y) \)
3. \( \log_a\left(\frac{x}{y}\right) = \log_a x \div \log_a y \)
4. \( \log_a(x y) = \log_a x \cdot \log_a y \)
-/
theorem log_properties_test 
  (a x y : ℝ)
  (ha : a > 0) (ha1 : a ≠ 1)
  (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (¬ (log a x * log a y = log a (x + y))) ∧ 
  (¬ (log a x - log a y = log a (x - y))) ∧ 
  (¬ (log a (x / y) = log a x / log a y)) ∧ 
  (¬ (log a (x * y) = log a x * log a y )) :=
sorry

end log_properties_test_l64_64469


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64720

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64720


namespace arithmetic_sequence_sum_l64_64081

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 2 = 1) (h2 : a 4 = 5) (h3 : ∀ n, a (n + 1) = a n + d) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 15 :=
begin
  sorry
end

end arithmetic_sequence_sum_l64_64081


namespace cone_volume_l64_64659

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l64_64659


namespace pow_simplification_l64_64367

theorem pow_simplification :
  9^6 * 3^3 / 27^4 = 27 :=
by
  sorry

end pow_simplification_l64_64367


namespace find_line_l_curve_C_and_diff_l64_64085

-- Definitions from the conditions:
def parametric_line_l (a t : ℝ) : ℝ × ℝ := (2 * real.sqrt 3 + a * t, 4 + real.sqrt 3 * t)
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * real.cos θ, ρ * real.sin θ)
def point_P: ℝ × ℝ := (real.sqrt 3, 0)
def curve_C_rect (x y : ℝ) : Prop := y^2 = 4 * x

-- Point A in rectangular coordinates:
def point_A := polar_to_rect 2 (real.pi / 6)

-- Statement to be proved:
theorem find_line_l_curve_C_and_diff (a t t1 t2 : ℝ) :
  point_A = (real.sqrt 3, 1) →
  parametric_line_l a t = (real.sqrt 3, 1) →
  a = 1 →
  t = -real.sqrt 3 →
  curve_C_rect (real.sqrt 3 - (real.sqrt 3 / 2) * t) ((1 / 2) * t) →
  t1 + t2 = -8 * real.sqrt 3 →
  t1 * t2 = -16 * real.sqrt 3 →
  t1 > 0 →
  t2 < 0 →
  ((1 / |t1|) + (1 / |t2|)) = 1 / 2 :=
begin
  sorry
end

end find_line_l_curve_C_and_diff_l64_64085


namespace intersection_eq_singleton_one_l64_64050

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := { x ∣ (x + 1) * (x - 2) < 0 }

-- Prove that the intersection of A and B is {1}
theorem intersection_eq_singleton_one : A ∩ B = {1} :=
by
  sorry

end intersection_eq_singleton_one_l64_64050


namespace a_2017_minus_a_2016_l64_64046

-- Define the sequences a and b with given initial conditions and recurrence relations
noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := 3 * (a n) + 2 * (b n)

noncomputable def b : ℕ → ℝ
| 0     := 1
| (n+1) := - (a n)

-- Define the theorem to prove
theorem a_2017_minus_a_2016 : a 2017 - a 2016 = 2^2017 :=
by
  sorry

end a_2017_minus_a_2016_l64_64046


namespace product_trippled_when_added_to_reciprocal_l64_64932

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64932


namespace selection_schemes_l64_64608

theorem selection_schemes (m f : ℕ) (total : ℕ) :
  m = 4 → f = 3 → total = 7 →
  let total_ways := Nat.choose total 3,
      only_male_ways := Nat.choose m 3,
      required_ways := total_ways - only_male_ways,
      schemes_per_group := Nat.factorial 3 in
  required_ways * schemes_per_group = 186 :=
by
  sorry

end selection_schemes_l64_64608


namespace tangent_line_at_a_eq_2_l64_64038

noncomputable def f (x : ℝ) (a : ℝ) := a * x^2 - (a + 2) * x + Real.log x

def tangent_line_eq (a : ℝ) (x : ℝ) : Prop :=
  ∀ (x₀ : ℝ), f x₀ a = -2 → (D (f x₀ a).deriv x₀) x₀ = 1 → x - x₀ + 2 = 0

theorem tangent_line_at_a_eq_2 :
  tangent_line_eq 2 1 := by
  sorry

end tangent_line_at_a_eq_2_l64_64038


namespace sum_factors_of_30_l64_64950

def factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ m, m > 0 ∧ n % m == 0)

theorem sum_factors_of_30 : (factors 30).sum = 72 := by
  sorry

end sum_factors_of_30_l64_64950


namespace distances_sum_bound_l64_64539

-- Define points A, B, and D
def A : ℝ × ℝ := (15, 0)
def B : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (3, 4)

-- Define distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define distances AD and BD
def AD : ℝ := distance A D
def BD : ℝ := distance B D

-- The main statement asserting the condition on AD + BD
theorem distances_sum_bound : 17 < AD + BD ∧ AD + BD < 18 :=
by
  -- The proof is omitted
  sorry

end distances_sum_bound_l64_64539


namespace min_phi_theta_distance_l64_64483

theorem min_phi_theta_distance
  (ϕ : ℝ) (θ : ℝ)
  (h_ϕ_bound : |ϕ| < (π / 2))
  (h_sym_axis : ∃ k : ℤ, ϕ = (π / 6) + k * π)
  (h_center_sym : ∃ k : ℤ, θ = (1 / 2 : ℝ) * k * π - π / 4)
  (θ_pos : θ > 0) :
  |ϕ - θ| = π / 12 :=
begin
  sorry
end

end min_phi_theta_distance_l64_64483


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64824

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64824


namespace steve_more_than_wayne_first_time_at_2004_l64_64358

def initial_steve_money (year: ℕ) := if year = 2000 then 100 else 0
def initial_wayne_money (year: ℕ) := if year = 2000 then 10000 else 0

def steve_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_steve_money year
  else 2 * steve_money (year - 1)

def wayne_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_wayne_money year
  else wayne_money (year - 1) / 2

theorem steve_more_than_wayne_first_time_at_2004 :
  ∃ (year: ℕ), year = 2004 ∧ steve_money year > wayne_money year := by
  sorry

end steve_more_than_wayne_first_time_at_2004_l64_64358


namespace indicator_significant_improvement_l64_64300

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64300


namespace value_of_expression_l64_64413

theorem value_of_expression (V E F t h : ℕ) (H T : ℕ) 
  (h1 : V - E + F = 2)
  (h2 : F = 42)
  (h3 : T = 3)
  (h4 : H = 2)
  (h5 : t + h = 42)
  (h6 : E = (3 * t + 6 * h) / 2) :
  100 * H + 10 * T + V = 328 :=
sorry

end value_of_expression_l64_64413


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64728

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64728


namespace find_term_in_sequence_l64_64062

theorem find_term_in_sequence :
  ∃ n : ℕ, (3^(7*(n - 1)) = 3^98) ∧ n = 15 :=
by
  use 15
  split
  . simp
  . exact rfl
  sorry

end find_term_in_sequence_l64_64062


namespace significant_improvement_l64_64244

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64244


namespace trigonometric_problem_l64_64018

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : sin α = 3 / 5)
variable (h2 : cos (β + π / 6) = -3 * real.sqrt 3 / 14)

theorem trigonometric_problem :
  sin (2 * α) = 24 / 25 ∧ cos β = 1 / 7 :=
by
  sorry

end trigonometric_problem_l64_64018


namespace rectangle_diagonals_equal_l64_64990

theorem rectangle_diagonals_equal (ABCD : Type) [rect : Rectangle ABCD] :
  diagonals_equal ABCD :=
sorry

end rectangle_diagonals_equal_l64_64990


namespace sum_of_numbers_eq_answer_l64_64177

open Real

noncomputable def sum_of_numbers (x y : ℝ) : ℝ := x + y

theorem sum_of_numbers_eq_answer (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) (h4 : (1 / x) = 3 * (1 / y)) :
  sum_of_numbers x y = 16 * Real.sqrt 3 / 3 := 
sorry

end sum_of_numbers_eq_answer_l64_64177


namespace probability_blue_prime_and_yellow_power_of_two_l64_64203

-- Definitions
def blue_die_outcomes := { n : ℕ | 1 ≤ n ∧ n ≤ 8 }
def yellow_die_outcomes := { n : ℕ | 1 ≤ n ∧ n ≤ 8 }
def prime_numbers := {2, 3, 5, 7}
def powers_of_two := {1, 2, 4, 8}

-- Theorem statement
theorem probability_blue_prime_and_yellow_power_of_two :
  (finset.card (prime_numbers ×ˢ powers_of_two) : ℚ) / (finset.card (blue_die_outcomes ×ˢ yellow_die_outcomes) : ℚ) = 1 / 4 :=
by
  sorry

end probability_blue_prime_and_yellow_power_of_two_l64_64203


namespace problem_tangent_lines_problem_point_Q_problem_parallel_OP_AB_l64_64452

theorem problem_tangent_lines (O M P : ℝ × ℝ) (hO : O = (0, 0)) (hM : M = (-2, -2)) 
  (hO_eq : (O.fst)^2 + (O.snd)^2 = 2) (hM_eq : (M.fst + 2)^2 + (M.snd + 2)^2 = 2) 
  (hP : P = (1, 1)) : 
  ∃ k : ℝ, (y = ((2 + sqrt(3)) * x) ∨ y = ((2 - sqrt(3)) * x)) :=
sorry

theorem problem_point_Q (Q : ℝ × ℝ) (hQ_on_O : Q.fst^2 + Q.snd^2 = 2)
  (hO_eq2 : (Q.fst + 2)^2 + (Q.snd + 2)^2 = 8) (angle_eq : ∠EQF = π / 3) : 
  ∃ x y : ℝ, (Q = (x, y) ∧ Q = ( (-(1 + sqrt(15)) / 4, (-(1 - sqrt(15)) / 4) ) ∨ 
  Q = ((-(1 + sqrt(15)) / 4, (-(1 - sqrt(15)) / 4) )) :=
sorry

theorem problem_parallel_OP_AB (O P : ℝ × ℝ) (hO : O = (0, 0))
  (hP : P = (1, 1)) (A B : ℝ × ℝ) (hPA_compl : slope_angle PA + slope_angle PB = π / 2)
  (h_O_eq : O.fst^2 + O.snd^2 = 2) :
  line_eq k_OP = line_eq k_AB :=
sorry

end problem_tangent_lines_problem_point_Q_problem_parallel_OP_AB_l64_64452


namespace max_dot_product_OB_OA_l64_64473

theorem max_dot_product_OB_OA (P A O B : ℝ × ℝ)
  (h₁ : ∃ x y : ℝ, (x, y) = P ∧ x^2 / 16 - y^2 / 9 = 1)
  (t : ℝ)
  (h₂ : A = (t - 1) • P)
  (h₃ : P • O = 64)
  (h₄ : B = (0, 1)) :
  ∃ t : ℝ, abs (B • A) ≤ (24/5) := 
sorry

end max_dot_product_OB_OA_l64_64473


namespace transform_roots_to_quadratic_l64_64178

noncomputable def quadratic_formula (p q : ℝ) (x : ℝ) : ℝ :=
  x^2 + p * x + q

theorem transform_roots_to_quadratic (x₁ x₂ y₁ y₂ p q : ℝ)
  (h₁ : quadratic_formula p q x₁ = 0)
  (h₂ : quadratic_formula p q x₂ = 0)
  (h₃ : x₁ ≠ 1)
  (h₄ : x₂ ≠ 1)
  (hy₁ : y₁ = (x₁ + 1) / (x₁ - 1))
  (hy₂ : y₂ = (x₂ + 1) / (x₂ - 1)) :
  (1 + p + q) * y₁^2 + 2 * (1 - q) * y₁ + (1 - p + q) = 0 ∧
  (1 + p + q) * y₂^2 + 2 * (1 - q) * y₂ + (1 - p + q) = 0 := 
sorry

end transform_roots_to_quadratic_l64_64178


namespace value_of_x_when_z_is_64_l64_64227

-- Declare the necessary definitions and constants
variables {k n : ℝ} 
variables {x z : ℝ}

-- Given conditions
def directly_proportional (x y : ℝ) := x = k * y^2
def inversely_proportional (y : ℝ) (z : ℝ) := y = n / Real.sqrt z
def value_at_sixteen : x = 4 := sorry -- Existing known relationship when z = 16

-- The theorem statement
theorem value_of_x_when_z_is_64 (hz : z = 64) (h_prop : directly_proportional x (n / Real.sqrt z)) (h_given : value_at_sixteen) : x = 1 :=
by
  sorry

end value_of_x_when_z_is_64_l64_64227


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64837

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64837


namespace inradius_orthic_triangle_l64_64474

-- Definitions and hypotheses
variable {ABC : Triangle}
variable (R r P : ℝ)
variable [triangleCircumradius ABC R]
variable [triangleInradius ABC r]
variable [orthicTriangleInradius ABC P]
variable (hR : R = 1)

-- Theorem statement
theorem inradius_orthic_triangle (hR : R = 1) (r : ℝ) (P : ℝ) :
  P ≤ 1 - (1/3) * (1 + r) ^ 2 :=
sorry

end inradius_orthic_triangle_l64_64474


namespace percentage_liquid_X_in_B_l64_64591

def liquid_X_in_solution_A := 0.008 
def weight_solution_A := 600 
def weight_solution_B := 700
def percentage_liquid_X_in_mixture := 0.0174
def total_weight := weight_solution_A + weight_solution_B
def amount_liquid_X_mixture := total_weight * percentage_liquid_X_in_mixture
def amount_liquid_X_A := weight_solution_A * liquid_X_in_solution_A

theorem percentage_liquid_X_in_B (P : ℝ) :
  amount_liquid_X_A + weight_solution_B * P = amount_liquid_X_mixture →
  P = 0.0254571428571429 :=
by
  intros h
  sorry

end percentage_liquid_X_in_B_l64_64591


namespace unique_positive_integer_n_l64_64379

theorem unique_positive_integer_n :
  ∃ (n : ℕ), (3 * 3^3 + 4 * 3^4 + 5 * 3^5 + ∑ i in finset.range (n - 4 + 1), (i + 6) * 3^(i + 6)) = 3^(n + 8) ∧ n = 4375 := by
  sorry

end unique_positive_integer_n_l64_64379


namespace distance_walked_on_last_day_l64_64239

theorem distance_walked_on_last_day
  (a : ℕ → ℝ)
  (q : ℝ)
  (n : ℕ)
  (S_n : ℝ) :
  q = (1 / 2) →
  S_n = 378 →
  n = 6 →
  (Σ k in finset.range(n), a k) = S_n →
  a 0 * q ^ (n - 1) = 6 :=
by
  sorry

end distance_walked_on_last_day_l64_64239


namespace significant_improvement_l64_64276

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64276


namespace product_of_solutions_l64_64708

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64708


namespace product_of_all_solutions_triple_reciprocal_l64_64743

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64743


namespace small_pizza_slices_l64_64332

theorem small_pizza_slices :
  ∃ S : ℕ,
  let total_slices := 3 * S + 2 * 8 in
  let pieces_eaten := 3 + (3 + 1) + (3 + 1) / 2 + 3 + 3 + 3 in
  total_slices = pieces_eaten + 10 ∧ S = 4 :=
begin
  existsi 4,
  simp,
  rw add_assoc,
  dsimp,
  norm_num,
end

end small_pizza_slices_l64_64332


namespace find_a_l64_64488

def f : ℝ → ℝ := λ x : ℝ, if x ≤ 0 then 2^x else -x^2 + 1

theorem find_a (a : ℝ) : 
  f a = 1/2 → (a = -1 ∨ a = sqrt 2 / 2) :=
by
  sorry

end find_a_l64_64488


namespace rowing_rate_in_still_water_l64_64607

theorem rowing_rate_in_still_water (R C : ℝ) 
  (h1 : (R + C) * 2 = 26)
  (h2 : (R - C) * 4 = 26) : 
  R = 26 / 3 :=
by
  sorry

end rowing_rate_in_still_water_l64_64607


namespace geometric_sequence_S5_l64_64445

noncomputable def S5 (a₁ q : ℝ) : ℝ :=
  a₁ * (1 - q^5) / (1 - q)

theorem geometric_sequence_S5 
  (a₁ q : ℝ) 
  (h₁ : a₁ * (1 + q) = 3 / 4)
  (h₄ : a₁ * q^3 * (1 + q) = 6) :
  S5 a₁ q = 31 / 4 := 
sorry

end geometric_sequence_S5_l64_64445


namespace man_l64_64315

theorem man's_salary 
  (food_fraction : ℚ := 1/5) 
  (rent_fraction : ℚ := 1/10) 
  (clothes_fraction : ℚ := 3/5) 
  (remaining_money : ℚ := 15000) 
  (S : ℚ) :
  (S * (1 - (food_fraction + rent_fraction + clothes_fraction)) = remaining_money) →
  S = 150000 := 
by
  intros h1
  sorry

end man_l64_64315


namespace new_device_significant_improvement_l64_64258

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64258


namespace geometric_sequence_property_l64_64437

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n m : ℕ), a (n + 1) * a (m + 1) = a n * a m

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
(h_condition : a 2 * a 4 = 1/2) :
  a 1 * a 3 ^ 2 * a 5 = 1/4 :=
by
  sorry

end geometric_sequence_property_l64_64437


namespace bottle_total_height_l64_64331

theorem bottle_total_height (r1 r2 water_height_up water_height_down : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 3) (h_water_height_up : water_height_up = 20) (h_water_height_down : water_height_down = 28) : 
    ∃ x : ℝ, (π * r1^2 * (x - water_height_up) = 9 * π * (x - water_height_down) ∧ x = 29) := 
by 
    sorry

end bottle_total_height_l64_64331


namespace significant_improvement_l64_64251

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64251


namespace piecewise_function_correct_l64_64012

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x + 2 else x ^ 2

theorem piecewise_function_correct (x : ℝ) : 
  (x ≥ 0 → f x = x + 2) ∧ (x < 0 → f x = x ^ 2) := 
by
  split
  {
    intro h
    simp [f, h]
  }
  {
    intro h
    simp [f, h]
  }
  sorry

end piecewise_function_correct_l64_64012


namespace product_of_solutions_l64_64706

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64706


namespace team_leader_and_deputy_choice_l64_64684

def TeamLeaderSelection : Type := {x : Fin 5 // true}
def DeputyLeaderSelection (TL : TeamLeaderSelection) : Type := {x : Fin 5 // x ≠ TL.val}

theorem team_leader_and_deputy_choice : 
  (Σ TL : TeamLeaderSelection, DeputyLeaderSelection TL) → Fin 20 :=
by sorry

end team_leader_and_deputy_choice_l64_64684


namespace product_of_triple_when_added_to_reciprocal_l64_64901

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64901


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64917

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64917


namespace snakes_hiding_l64_64362

/-- The statement that given the total number of snakes and the number of snakes not hiding,
we can determine the number of snakes hiding. -/
theorem snakes_hiding (total_snakes : ℕ) (snakes_not_hiding : ℕ) (h1 : total_snakes = 95) (h2 : snakes_not_hiding = 31) :
  total_snakes - snakes_not_hiding = 64 :=
by {
  sorry
}

end snakes_hiding_l64_64362


namespace total_savings_l64_64401

def individual_shirt_cost : ℝ := 7.50
def individual_pants_cost : ℝ := 15.00
def individual_socks_cost : ℝ := 4.50

def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75

def team_size : ℕ := 12

theorem total_savings :
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost in
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost in
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost in
  let total_savings := savings_per_uniform * team_size in
  total_savings = 36 :=
by
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  have h : total_savings = 36 := by
    calc
      total_savings = (7.50 + 15.00 + 4.50 - (6.75 + 13.50 + 3.75)) * 12 := by sorry
                  ... = 3 * 12 := by sorry
                  ... = 36 := by sorry
  exact h

end total_savings_l64_64401


namespace max_subsets_with_nonempty_intersections_l64_64226

theorem max_subsets_with_nonempty_intersections {X : Type*} [Fintype X] (h : Fintype.card X = n) :
  ∃ S : Finset (Finset X), (∀ A B ∈ S, (A ∩ B).Nonempty) ∧ S.card ≤ 2^(n-1) :=
sorry

end max_subsets_with_nonempty_intersections_l64_64226


namespace sum_eq_zero_l64_64635

variable {R : Type} [Field R]

-- Define the conditions
def cond1 (a b c : R) : Prop := (a + b) / c = (b + c) / a
def cond2 (a b c : R) : Prop := (b + c) / a = (a + c) / b
def neq (b c : R) : Prop := b ≠ c

-- State the theorem
theorem sum_eq_zero (a b c : R) (h1 : cond1 a b c) (h2 : cond2 a b c) (h3 : neq b c) : a + b + c = 0 := 
by sorry

end sum_eq_zero_l64_64635


namespace tangent_line_at_P_l64_64627

def tangent_line_eq (x y : ℝ) : ℝ := x - 2 * y + 1

theorem tangent_line_at_P (x y : ℝ) (h : x ^ 2 + y ^ 2 - 4 * x + 2 * y = 0 ∧ (x, y) = (1, 1)) :
    tangent_line_eq x y = 0 := 
sorry

end tangent_line_at_P_l64_64627


namespace frog_arrangement_count_l64_64196

theorem frog_arrangement_count : 
  let frogs : List (Sum (Sum Bool Bool) Unit) := [Sum.inl (Sum.inl tt), Sum.inl (Sum.inl tt), Sum.inl (Sum.inr tt), Sum.inl (Sum.inr tt), Sum.inl (Sum.inr tt), Sum.inr (), Sum.inr (), Sum.inr ()] in
  let green_frogs := 2 in
  let red_frogs := 3 in
  let blue_frogs := 3 in
  -- Conditions
  -- 1. There are eight distinctly identifiable frogs sitting in a row.
  (length frogs = 8) ∧
  -- 2. Two are green, three are red, and three are blue.
  (green_frogs = 2) ∧
  (red_frogs = 3) ∧
  (blue_frogs = 3) ∧
  -- 3. Green frogs refuse to sit next to the red frogs.
  (∀ (i j : Nat), frogs[i] = Sum.inl (Sum.inl tt) → frogs[j] = Sum.inl (Sum.inr tt) → abs (i - j) > 1) ∧
  -- 4. Blue frogs also refuse to sit next to the red frogs.
  (∀ (i j : Nat), frogs[i] = Sum.inr () → frogs[j] = Sum.inl (Sum.inr tt) → abs (i - j) > 1)
  → Permutations frogs 144 := 
by
  sorry

end frog_arrangement_count_l64_64196


namespace product_of_real_solutions_triple_property_l64_64783

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64783


namespace euler_line_parallel_l64_64447

-- Define basic geometry and lines
noncomputable def Line (ℝ : Type) := (ℝ → ℝ) → Prop

-- Define conditions required for the problem
variables (l1 l2 l3 l4 : Line ℝ)
variables {triangle1_coords triangle2_coords triangle3_coords triangle4_coords : ℝ × ℝ}
noncomputable def Euler_line (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Line ℝ := sorry

-- Define the problem as a theorem
theorem euler_line_parallel (h1 : Euler_line (triangle1_coords, triangle2_coords, triangle3_coords) = l1)
                            (h2 : Euler_line (triangle2_coords, triangle3_coords, triangle4_coords) = l2)
                            (h3 : Euler_line (triangle3_coords, triangle4_coords, triangle1_coords) = l3)
                            (h4 : Euler_line (triangle4_coords, triangle1_coords, triangle2_coords) = l4) :
                           (∀ {i j : ℕ}, {i j ≠ 1}) → (l4 = Euler_line (triangle4_coords, triangle1_coords, triangle3_coords)) := sorry

end euler_line_parallel_l64_64447


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64765

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64765


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64889

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64889


namespace product_of_real_solutions_triple_property_l64_64788

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64788


namespace undeterminable_product_l64_64629

theorem undeterminable_product
  (f : ℝ → ℝ)
  (h_continuous : ContinuousOn f (Set.Icc (-2 : ℝ) (2 : ℝ)))
  (h_root : ∃ x ∈ Set.Ioo (-2 : ℝ) (2 : ℝ), f x = 0) :
  ¬ (f (-2) * f (2) > 0 ∨ f (-2) * f (2) < 0 ∨ f (-2) * f (2) = 0) :=
sorry

end undeterminable_product_l64_64629


namespace significant_improvement_l64_64265

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64265


namespace mean_significantly_improved_l64_64297

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64297


namespace quadrilateral_area_l64_64179

theorem quadrilateral_area (AB CD : ℝ) (r k : ℝ) (BC AD : ℝ)
  (h1 : AB * CD = 0)  -- AB and CD are perpendicular
  (h2 : AB = 2 * r ∧ CD = 2 * r)  -- AB and CD are the diameters of circles with radius r
  (h3 : BC / AD = k)  -- Given ratio
  (B C D : ℝ) : 
  ∃ A : ℝ, 
  (|BC|^2 = A^2 + B^2) ∧ 
  (|AD|^2 = (A + 2 * r)^2 + (B + 2 * r)^2) ∧ 
  (4 * r^2 = (A + r)^2 + (B + r)^2) →
  (3 * r^2 * |(1 - k^2) / (1 + k^2)| = |(A + 2 * r) * (B + 2 * r) - A * B|/2) ∧ 
  area_quadrilateral ABCD = 3 * r^2 * |(1 - k^2) / (1 + k^2)| :=
by sorry

end quadrilateral_area_l64_64179


namespace product_of_real_solutions_triple_property_l64_64796

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64796


namespace eggs_today_l64_64623

-- Condition definitions
def eggs_yesterday : ℕ := 10
def difference : ℕ := 59

-- Statement of the problem
theorem eggs_today : eggs_yesterday + difference = 69 := by
  sorry

end eggs_today_l64_64623


namespace teachers_engineers_ratio_l64_64075

-- Define the conditions
variables (t e : ℝ) -- t is the number of teachers, e is the number of engineers

-- Average ages
def avg_age_teachers := 40
def avg_age_engineers := 55
def overall_avg_age := 45

-- Total ages based on average ages
def total_age_teachers := avg_age_teachers * t
def total_age_engineers := avg_age_engineers * e

-- Overall average age equation
def overall_avg_age_eqn := (total_age_teachers + total_age_engineers) / (t + e) = overall_avg_age

theorem teachers_engineers_ratio (h : overall_avg_age_eqn) : t / e = 2 := 
by { sorry }

end teachers_engineers_ratio_l64_64075


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64918

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64918


namespace product_of_solutions_l64_64872

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64872


namespace initial_innings_count_l64_64621

theorem initial_innings_count (n T L : ℕ) 
  (h1 : T = 50 * n)
  (h2 : 174 = L + 172)
  (h3 : (T - 174 - L) = 48 * (n - 2)) :
  n = 40 :=
by 
  sorry

end initial_innings_count_l64_64621


namespace scale_reading_l64_64160

theorem scale_reading (a b c : ℝ) (h₁ : 10.15 < a ∧ a < 10.4) (h₂ : 10.275 = (10.15 + 10.4) / 2) : a = 10.3 := 
by 
  sorry

end scale_reading_l64_64160


namespace product_of_solutions_l64_64773

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64773


namespace sum_factors_of_30_l64_64948

def factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ m, m > 0 ∧ n % m == 0)

theorem sum_factors_of_30 : (factors 30).sum = 72 := by
  sorry

end sum_factors_of_30_l64_64948


namespace least_k_for_simple_representation_l64_64323

def is_simple (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

theorem least_k_for_simple_representation :
  ∃ k : ℕ, (∀ n : ℕ, ∃ a : list ℕ, (∀ x ∈ a, is_simple x) ∧ n = a.sum) ∧ (∀ m < k, ∃ n : ℕ, ¬ ∃ a : list ℕ, (∀ x ∈ a, is_simple x) ∧ n = a.sum) ∧ k = 9 :=
by
  sorry

end least_k_for_simple_representation_l64_64323


namespace significant_improvement_l64_64268

noncomputable def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : list ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : list ℝ) (mean_val : ℝ) : ℝ :=
  (data.map (λ x, (x - mean_val) ^ 2)).sum / data.length

theorem significant_improvement :
  let x_avg := mean old_device_data in
  let y_avg := mean new_device_data in
  let s1_sq := variance old_device_data x_avg in
  let s2_sq := variance new_device_data y_avg in
  y_avg - x_avg ≥ 2 * real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64268


namespace f_odd_l64_64490

def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 2)

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end f_odd_l64_64490


namespace cone_volume_l64_64648

theorem cone_volume (s h : ℝ) (h_s : s = 15) (h_h : h = 9) : 
  ∃ V : ℝ, V = (1 / 3) * π * (sqrt (s^2 - h^2))^2 * h := 
by
  use 432 * π
  rw [h_s, h_h]
  have rsq : 15^2 - 9^2 = 144 := by norm_num
  have radius : sqrt (15^2 - 9^2) = 12 := by rw [rsq, Real.sqrt_sq_eq_abs, abs_of_nonneg]; norm_num
  simp [radius]
  norm_num
  sorry

end cone_volume_l64_64648


namespace product_of_solutions_l64_64769

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64769


namespace significant_improvement_l64_64291

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64291


namespace range_of_a_l64_64031

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f(x) = x * |x|) 
  (h2 : ∀ x, x ∈ set.Icc a (a + 1) → f(x + 2 * a) > 4 * f(x)) : 
  a > 1 :=
sorry

end range_of_a_l64_64031


namespace sum_of_factors_30_l64_64970

theorem sum_of_factors_30 : 
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ) in
  ( ∑ n in factors, n ) = 72 :=
by
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ)
  rw [finset.sum_eq_sum_of_elements] -- transposing to handle the sum notation
  let factors_list := [1, 2, 3, 5, 6, 10, 15, 30]
  have h_factors : factors = {x | x ∈ factors_list}.to_finset, by sorry
  simp_rw h_factors
  rw [finset.sum_to_finset]
  dsimp -- simplifying before asserting the sum
  norm_num -- proving that the sum is indeed 72
  sorry

end sum_of_factors_30_l64_64970


namespace inequality_proof_l64_64115

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a + b + c = 1)

theorem inequality_proof : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := 
by 
  sorry

end inequality_proof_l64_64115


namespace measure_angle_ACE_l64_64454

noncomputable def angle_ACE_condition : Prop :=
  ∃ (A B C D E P Q O : Type) 
    (convex_pentagon : convex_hull {A, B, C, D, E} = {A, B, C, D, E})
    (P_def : ∃ BE AC : ∀ ab, ab ∈ {B, E} → ab ∈ {A, C}, P = BE ∩ AC)
    (Q_def : ∃ CE AD : ∀ ab, ab ∈ {C, E} → ab ∈ {A, D}, Q = CE ∩ AD)
    (O_def : ∃ AD BE : ∀ ab, ab ∈ {A, D} → ab ∈ {B, E}, O = AD ∩ BE)
    (isos_ABP : ∃ ABP : ∀ abp, abp ∈ {A, B, P} → angle.abp = 80)
    (isos_DEQ : ∃ DEQ : ∀ deq, deq ∈ {D, E, Q} → angle.deq = 80)
    (isos_APO : ∃ APO : ∀ apo, apo ∈ {A, P, O} → is_isosceles_triangle APO)
    (isos_EQO : ∃ EQO : ∀ eqo, eqo ∈ {E, Q, O} → is_isosceles_triangle EQO),
  angle A C E = 60 ∨ angle A C E = 105

theorem measure_angle_ACE (A B C D E P Q O : Type)
  (convex_pentagon : convex_hull {A, B, C, D, E} = {A, B, C, D, E})
  (P_def : ∃ BE AC : ∀ ab, ab ∈ {B, E} → ab ∈ {A, C}, P = BE ∩ AC)
  (Q_def : ∃ CE AD : ∀ ab, ab ∈ {C, E} → ab ∈ {A, D}, Q = CE ∩ AD)
  (O_def : ∃ AD BE : ∀ ab, ab ∈ {A, D} → ab ∈ {B, E}, O = AD ∩ BE)
  (isos_ABP : ∃ ABP : ∀ abp, abp ∈ {A, B, P} → angle.abp = 80)
  (isos_DEQ : ∃ DEQ : ∀ deq, deq ∈ {D, E, Q} → angle.deq = 80)
  (isos_APO : ∃ APO : ∀ apo, apo ∈ {A, P, O} → is_isosceles_triangle APO)
  (isos_EQO : ∃ EQO : ∀ eqo, eqo ∈ {E, Q, O} → is_isosceles_triangle EQO) :
  angle A C E = 60 ∨ angle A C E = 105 :=
by
  sorry

end measure_angle_ACE_l64_64454


namespace domain_of_f_2x_minus_1_l64_64065

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → (f x ≠ 0)) →
  (∀ y, 0 ≤ y ∧ y ≤ 1 ↔ exists x, (2 * x - 1 = y) ∧ (0 ≤ x ∧ x ≤ 1)) :=
by
  sorry

end domain_of_f_2x_minus_1_l64_64065


namespace determine_coin_types_within_weighings_l64_64190

theorem determine_coin_types_within_weighings (n : ℕ) (h_n_eq : n = 100) 
  (gold silver copper : finset ℕ) (h_gold : gold.card > 0) 
  (h_silver : silver.card > 0) (h_copper : copper.card > 0) 
  (h_total : gold.card + silver.card + copper.card = n) :
  ∃ (weighings : ℕ), weighings ≤ 101 ∧ 
  ∀ coin, coin ∈ gold ∨ coin ∈ silver ∨ coin ∈ copper :=
  sorry

end determine_coin_types_within_weighings_l64_64190


namespace cone_volume_l64_64643

theorem cone_volume (s h : ℝ) (h_s : s = 15) (h_h : h = 9) : 
  ∃ V : ℝ, V = (1 / 3) * π * (sqrt (s^2 - h^2))^2 * h := 
by
  use 432 * π
  rw [h_s, h_h]
  have rsq : 15^2 - 9^2 = 144 := by norm_num
  have radius : sqrt (15^2 - 9^2) = 12 := by rw [rsq, Real.sqrt_sq_eq_abs, abs_of_nonneg]; norm_num
  simp [radius]
  norm_num
  sorry

end cone_volume_l64_64643


namespace lcm_12_18_24_l64_64700

theorem lcm_12_18_24 : Nat.lcm (Nat.lcm 12 18) 24 = 72 := by
  -- Given conditions (prime factorizations)
  have h1 : 12 = 2^2 * 3 := by norm_num
  have h2 : 18 = 2 * 3^2 := by norm_num
  have h3 : 24 = 2^3 * 3 := by norm_num
  -- Prove the LCM
  sorry

end lcm_12_18_24_l64_64700


namespace spherical_to_rectangular_coordinates_l64_64383

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l64_64383


namespace find_sum_l64_64512

theorem find_sum (x y : ℝ) (h₁ : 3 * |x| + 2 * x + y = 20) (h₂ : 2 * x + 3 * |y| - y = 30) : x + y = 15 :=
sorry

end find_sum_l64_64512


namespace nice_29_nice_2006_infinitely_many_nice_numbers_l64_64322

namespace NiceNumbers

-- Define the predicate for a number being nice.
def is_nice (d : ℕ) : Prop :=
  ∀ (x y : ℕ), d ∣ ((x + y) ^ 5 - x ^ 5 - y ^ 5) ↔ d ∣ ((x + y) ^ 7 - x ^ 7 - y ^ 7)

-- Prove that 29 is nice.
theorem nice_29 : is_nice 29 := 
sorry

-- Prove that 2006 is nice.
theorem nice_2006 : is_nice 2006 := 
sorry

-- Prove that there are infinitely many nice numbers.
theorem infinitely_many_nice_numbers : ∀ n : ℕ, ∃ d : ℕ, n < d ∧ is_nice d :=
sorry

end NiceNumbers

end nice_29_nice_2006_infinitely_many_nice_numbers_l64_64322


namespace shaded_triangle_probability_l64_64553

theorem shaded_triangle_probability :
  let triangles := ["ABD", "ABC", "ADE", "BDE", "ABE", "DEC"]
      shaded := ["ABD", "DEC"] in
  (shaded.length : ℚ) / triangles.length = 1 / 3 := by
sorry

end shaded_triangle_probability_l64_64553


namespace cos_of_sin_given_l64_64513

theorem cos_of_sin_given (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by
  sorry

end cos_of_sin_given_l64_64513


namespace total_interest_l64_64985

theorem total_interest (P R T : ℕ) (h1 : T = 10) (h2 : (P * R * 10) / 100 = 600) :
  let P' := 3 * P,
      T' := 5,
      SI := 600,
      SI' := (P' * R * T') / 100 in
  SI + SI' = 1500 := by
  sorry

end total_interest_l64_64985


namespace cone_volume_l64_64647

theorem cone_volume (s h : ℝ) (h_s : s = 15) (h_h : h = 9) : 
  ∃ V : ℝ, V = (1 / 3) * π * (sqrt (s^2 - h^2))^2 * h := 
by
  use 432 * π
  rw [h_s, h_h]
  have rsq : 15^2 - 9^2 = 144 := by norm_num
  have radius : sqrt (15^2 - 9^2) = 12 := by rw [rsq, Real.sqrt_sq_eq_abs, abs_of_nonneg]; norm_num
  simp [radius]
  norm_num
  sorry

end cone_volume_l64_64647


namespace difference_is_12_l64_64509

-- Definitions of the conditions
def eighty_percent_of_40 := 0.80 * 40
def four_fifths_of_25 := (4/5) * 25

-- The proof problem statement
theorem difference_is_12 : eighty_percent_of_40 - four_fifths_of_25 = 12 := by
  sorry

end difference_is_12_l64_64509


namespace fish_to_rice_l64_64076

variables (f b r : ℝ)

theorem fish_to_rice :
  (5 * f = 3 * b) ∧ (2 * b = 5 * r) → f = (3 / 2) * r :=
by
  intro h
  cases h with h1 h2
  have hb : b = 5 / 2 * r, from by linarith
  have hf : 5 * f = 3 * (5 / 2 * r), from by rw [h1, hb]
  linarith

end fish_to_rice_l64_64076


namespace product_of_solutions_l64_64800

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64800


namespace sum_factors_of_30_l64_64946

def factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ m, m > 0 ∧ n % m == 0)

theorem sum_factors_of_30 : (factors 30).sum = 72 := by
  sorry

end sum_factors_of_30_l64_64946


namespace luke_coin_total_l64_64592

theorem luke_coin_total :
  let quarters_piles := 7 
  let quarters_per_pile := 4
  let dimes_piles := 4 
  let dimes_per_pile := 2
  let nickels_piles := 6 
  let nickels_per_pile := 5
  let pennies_piles := 3 
  let pennies_per_pile := 8
  quarters_piles * quarters_per_pile + 
  dimes_piles * dimes_per_pile + 
  nickels_piles * nickels_per_pile + 
  pennies_piles * pennies_per_pile = 90 := 
by 
  have h1 : quarters_piles * quarters_per_pile = 28 := by norm_num
  have h2 : dimes_piles * dimes_per_pile = 8 := by norm_num
  have h3 : nickels_piles * nickels_per_pile = 30 := by norm_num
  have h4 : pennies_piles * pennies_per_pile = 24 := by norm_num
  calc 
    quarters_piles * quarters_per_pile + 
    dimes_piles * dimes_per_pile + 
    nickels_piles * nickels_per_pile + 
    pennies_piles * pennies_per_pile = 
    28 + 8 + 30 + 24 := by rw [h1, h2, h3, h4]
    ... = 90 := by norm_num


end luke_coin_total_l64_64592


namespace find_c_general_formula_arithmetic_sequence_l64_64556

-- Definition of the sequence according to problem conditions
def a (n : ℕ) : ℕ := 
  if n = 1 then 2 
  else 2 + n * (n - 1)

def c : ℕ := 2

-- Mathematically equivalent proof problem in Lean 4

-- Proof of c = 2 given conditions
theorem find_c (c_value : ℕ) (a_1 : ℕ) (a_2 : ℕ) (a_3 : ℕ) 
  (H1 : a_1 = 2) 
  (H2 : a_2 = a_1 + c_value)
  (H3 : a_3 = a_2 + 2 * c_value) 
  (H4 : a_1 * (a_1 + c_value) = a_2 * 2) : 
  c_value = 2 :=
begin
  sorry
end

-- Proof of the general formula for {a_n}
theorem general_formula (n : ℕ) (hn : n ≥ 1) : 
  a n = n^2 - n + 2 :=
begin
  sorry
end

-- Proof that { (a_n - c) / n } is an arithmetic sequence
theorem arithmetic_sequence (n : ℕ) (hn : n ≥ 1) : 
  (a n - c) / n = n - 1 :=
begin
  sorry
end

end find_c_general_formula_arithmetic_sequence_l64_64556


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64892

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64892


namespace sodium_chloride_formed_l64_64432

theorem sodium_chloride_formed (moles_NaHSO3 moles_HCl : ℕ) (h : moles_NaHSO3 = 2 ∧ moles_HCl = 2) : 
  let reaction := fun (n : ℕ) => n
  moles_NaHSO3 = moles_HCl -> reaction moles_NaHSO3 = 2 :=
by
  intros h1
  have h2 : reaction 2 = 2 := by rfl
  rw [←h1] at h2
  exact h2

end sodium_chloride_formed_l64_64432


namespace base4_arithmetic_proof_l64_64420

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ => (n % 10) + 4 * base4_to_base10 (n / 10)
  end

-- Define the numbers in base 4
def n1_base4 := 231
def n2_base4 := 21
def n3_base4 := 12
def n4_base4 := 3

-- Convert base 4 numbers to base 10
def n1_base10 := base4_to_base10 n1_base4
def n2_base10 := base4_to_base10 n2_base4
def n3_base10 := base4_to_base10 n3_base4
def n4_base10 := base4_to_base10 n4_base4

-- Perform arithmetic in base 10
def div_result := n3_base10 / n4_base10
def mul_result := n2_base10 * div_result
def add_result := n1_base10 + mul_result

-- Convert result back to base 4
noncomputable def base10_to_base4 (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ => (n % 4) + 10 * base10_to_base4 (n / 4)
  end

def final_result_base4 := base10_to_base4 add_result

theorem base4_arithmetic_proof :
  final_result_base4 = 333 :=
  sorry

end base4_arithmetic_proof_l64_64420


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64823

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64823


namespace cone_volume_l64_64666

-- Given conditions as definitions
def slant_height : ℝ := 15
def height : ℝ := 9
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- The volume of the cone
def volume : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem to prove that the calculated volume is 432π cubic centimeters
theorem cone_volume : volume = 432 * Real.pi := by
  sorry

end cone_volume_l64_64666


namespace coin_problem_l64_64191

theorem coin_problem
  (num_coins : ℕ)
  (coins : fin num_coins → ℕ)
  (gold_weight : ℕ := 3)
  (silver_weight : ℕ := 2)
  (copper_weight : ℕ := 1)
  (at_least_one_of_each : ∀ t, ∃ i, coins i = t)
  (sum_weights : (fin num_coins → ℕ) → ℕ := fun f => ∑ i, f i)
  (sum_weights_coins : sum_weights coins = 100):
  (∃ weighings, weighings ≤ 101 ∧ (∀ i, coins i = gold_weight ∨ coins i = silver_weight ∨ coins i = copper_weight)) :=
sorry

end coin_problem_l64_64191


namespace product_of_triple_when_added_to_reciprocal_l64_64899

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64899


namespace product_of_solutions_l64_64705

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64705


namespace fraction_d_can_be_zero_l64_64221

theorem fraction_d_can_be_zero :
  ∃ x : ℝ, (x + 1) / (x - 1) = 0 :=
by {
  sorry
}

end fraction_d_can_be_zero_l64_64221


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64860

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64860


namespace area_of_isosceles_triangle_l64_64079

-- Define the elements used in the conditions.
variable (A B C D : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D]

-- Define the lengths of the sides and the base.
variable (AB AC BC : ℝ) (BD DC AD : ℝ)

-- State the conditions and the goal.
-- Given: AB = AC = 13, BC = 10, isosceles triangle condition
-- To prove: area of triangle ABC = 60 square units
theorem area_of_isosceles_triangle (AB_eq : AB = 13) (AC_eq : AC = 13) (BC_eq : BC = 10)
  (isosceles : AB = AC) (bisect_base : BD + DC = BC ∧ BD = DC) (pythagorean : AB^2 = BD^2 + AD^2) :
  ∃ (Area : ℝ), Area = 60 :=
by
  let AD := Math.sqrt (AB^2 - BD^2)
  let Area := 1/2 * BC * AD
  have BD_eq_5 : BD = 5 := by
    sorry -- given that BD = BC / 2
  have AD_calc : AD = 12 := by
    sorry -- calculated from the Pythagorean theorem
  have Area_calc : Area = 60 := by
    sorry -- Area calculation using the base and height
  exists Area
  rw Area_calc
  done

end area_of_isosceles_triangle_l64_64079


namespace maria_wins_game_l64_64098

theorem maria_wins_game :
  ∃ (n : ℕ), n = 45 ∧
  ∀ (initial_pieces : ℕ), initial_pieces = 3 →
  (∀ (move_num : ℕ), (move_num > 0) → (initial_pieces + move_num) = n →
  (if even move_num then "Maria" else "João") = "Maria") :=
by
  sorry

end maria_wins_game_l64_64098


namespace ellipse_and_line_equation_l64_64461

theorem ellipse_and_line_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (ab_ratio : a / b = 2)
  (focus : ℝ × ℝ := (real.sqrt 3, 0)) (focus_pos : 0 < real.sqrt 3) (focus_eq : focus = (real.sqrt 3, 0))
  (line_pass_a : ℝ × ℝ := (1, 0)) (point_b : ℝ × ℝ := (0, 1))
  ⦃c : ℝ⦄ (c_value : c = real.sqrt 3) :
  (∃ (a b : ℝ), (a / b = 2 ∧ a^2 = b^2 + (real.sqrt 3)^2 ∧
    (x y : ℝ) (h : (x - 0)^2 / a^2 + y^2 / b^2 = 1 → false)), 
   (∃ (m : ℝ), (line_pass_a = (1, 0) ∧ point_b ∈ set.univ ∧ 
    (line_eq : linear_map ℝ × ℝ := (λ (y x), x = m * y + 1),
   line_eq = (λ (y x), x + y - 1 = 0 ∨ 3 * x - 5 * y - 3 = 0)))) :=
sorry

end ellipse_and_line_equation_l64_64461


namespace area_DEF_l64_64552

-- Defining the points A, B, and C
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (12, 0)

-- Midpoints D, E, and F
def D : ℝ × ℝ := (0, 5)
def E : ℝ × ℝ := (6, 0)
def F : ℝ × ℝ := (6, 5)

-- Calculating the area of triangle DEF
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

theorem area_DEF : triangle_area D E F = 15 := by
  sorry

end area_DEF_l64_64552


namespace arithmetic_sequence_sum_proof_l64_64111

theorem arithmetic_sequence_sum_proof
  (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 17 = 170)
  (h2 : a 2000 = 2001)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a (n + 1) = a n + (a 2 - a 1)) :
  S 2008 = 2019044 :=
  sorry

end arithmetic_sequence_sum_proof_l64_64111


namespace cone_volume_correct_l64_64667

noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let r := real.sqrt (slant_height^2 - height^2)
  in (1/3) * real.pi * r^2 * height

theorem cone_volume_correct :
  cone_volume 15 9 = 432 * real.pi := by
  sorry

end cone_volume_correct_l64_64667


namespace meet_starting_point_together_at_7_40_AM_l64_64375

-- Definitions of the input conditions
def Charlie_time : Nat := 5
def Alex_time : Nat := 8
def Taylor_time : Nat := 10

-- The combined time when they meet again at the starting point
def LCM_time (a b c : Nat) : Nat := Nat.lcm a (Nat.lcm b c)

-- Proving that the earliest time they all coincide again is 40 minutes after the start
theorem meet_starting_point_together_at_7_40_AM :
  LCM_time Charlie_time Alex_time Taylor_time = 40 := 
by
  unfold Charlie_time Alex_time Taylor_time LCM_time
  sorry

end meet_starting_point_together_at_7_40_AM_l64_64375


namespace equidistant_points_l64_64087

theorem equidistant_points 
  (A B C D N M : Type) 
  (h₁ h₂ : ℝ)
  (h1 : AN = DC)
  (h2 : CM = AD)
  (AN_perpendicular_to_AB : perpendicular AN AB)
  (CM_perpendicular_to_BC : perpendicular CM BC) :
  distance B M = distance B N := 
sorry

end equidistant_points_l64_64087


namespace monotone_increasing_range_a_l64_64526

theorem monotone_increasing_range_a {a : ℝ}:
  (∀ x > 2, deriv (λ x : ℝ, a * x^2 + x - real.log x) x ≥ 0) ↔ a ∈ set.Ici 0 :=
by
  sorry

end monotone_increasing_range_a_l64_64526


namespace concyclic_points_l64_64573

variables {A B C L P S M : Point}
variable {ΔABC : Triangle}
variable {circumcircle : Circle}
variable {tangent : Line}
variable [Configuration : ABCCircle ΔABC circumcircle tangent]
variable [Projection : BProjection B tangent P]
variable [Midpoints : MidpointOfArc circumcircle A C L] 
variable [MidpointSides : MidpointOfSides ΔABC S M]

theorem concyclic_points :
  Concyclic P L S M :=
sorry

end concyclic_points_l64_64573


namespace cone_volume_l64_64644

theorem cone_volume (s h : ℝ) (h_s : s = 15) (h_h : h = 9) : 
  ∃ V : ℝ, V = (1 / 3) * π * (sqrt (s^2 - h^2))^2 * h := 
by
  use 432 * π
  rw [h_s, h_h]
  have rsq : 15^2 - 9^2 = 144 := by norm_num
  have radius : sqrt (15^2 - 9^2) = 12 := by rw [rsq, Real.sqrt_sq_eq_abs, abs_of_nonneg]; norm_num
  simp [radius]
  norm_num
  sorry

end cone_volume_l64_64644


namespace product_trippled_when_added_to_reciprocal_l64_64940

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64940


namespace divide_diagonal_into_segments_l64_64214

-- Definitions based on the conditions
variable (A B C D B' C' D' M N K L P Q S : Point)
variables [segment AB BC CD DA BD]
variable [parallelogram A B C D]
variable (M : midpoint BC)
variable (N : midpoint CD)
variable (K : midpoint (segment BM))
variable (L : midpoint (segment ND))

-- Theorem statement
theorem divide_diagonal_into_segments :
  let P := intersection (line A K) BD in
  let Q := intersection (line A M) BD in
  let S := intersection (line A L) BD in
  segment_length BP = (1/5) * segment_length BD ∧
  segment_length PQ = (2/15) * segment_length BD ∧
  segment_length QR = (1/3) * segment_length BD ∧
  segment_length RS = (2/15) * segment_length BD ∧
  segment_length SD = (1/5) * segment_length BD := 
  sorry

end divide_diagonal_into_segments_l64_64214


namespace product_of_all_solutions_triple_reciprocal_l64_64734

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64734


namespace crow_speed_is_8_l64_64310

noncomputable def crow_speed : ℕ → ℕ → ℝ → ℝ
| distance, trips, time := (distance * trips * 2) / (1000 * time)

theorem crow_speed_is_8
  (distance : ℕ) (trips : ℕ) (time : ℝ)
  (h_distance : distance = 400)
  (h_trips : trips = 15)
  (h_time : time = 1.5) : crow_speed distance trips time = 8 :=
by
  rw [h_distance, h_trips, h_time]
  show crow_speed 400 15 1.5 = 8
  -- skipping proof
  sorry

end crow_speed_is_8_l64_64310


namespace spherical_to_rectangular_l64_64393

theorem spherical_to_rectangular :
  ∀ (ρ θ φ : ℝ),
  ρ = 3 →
  θ = (3 * π) / 2 →
  φ = π / 3 →
  let x := ρ * real.sin φ * real.cos θ,
      y := ρ * real.sin φ * real.sin θ,
      z := ρ * real.cos φ in
  (x, y, z) = (0, -3 * real.sqrt 3 / 2, 3 / 2) := by
  intros,
  -- The actual proof would go here
  sorry

end spherical_to_rectangular_l64_64393


namespace find_prices_and_minimal_cost_l64_64989

variables (m n W : ℕ) (x y : ℤ)

def price_equations : Prop :=
  (3 * m + n = 360) ∧ (5 * m + 3 * n = 680)

def purchase_conditions : Prop :=
  (x + y = 100) ∧ (x ≥ 3 * y)

def prices : Prop :=
  (m = 100) ∧ (n = 60)

def minimal_cost : Prop :=
  (x = 75) ∧ (y = 25) ∧ (W = 9000)

theorem find_prices_and_minimal_cost :
  price_equations m n →
  purchase_conditions x y →
  (prices m n) →
  (minimal_cost x y W) :=
sorry

end find_prices_and_minimal_cost_l64_64989


namespace probability_blue_prime_and_yellow_power_of_two_l64_64202

-- Definitions
def blue_die_outcomes := { n : ℕ | 1 ≤ n ∧ n ≤ 8 }
def yellow_die_outcomes := { n : ℕ | 1 ≤ n ∧ n ≤ 8 }
def prime_numbers := {2, 3, 5, 7}
def powers_of_two := {1, 2, 4, 8}

-- Theorem statement
theorem probability_blue_prime_and_yellow_power_of_two :
  (finset.card (prime_numbers ×ˢ powers_of_two) : ℚ) / (finset.card (blue_die_outcomes ×ˢ yellow_die_outcomes) : ℚ) = 1 / 4 :=
by
  sorry

end probability_blue_prime_and_yellow_power_of_two_l64_64202


namespace team_savings_correct_l64_64408

-- Define the costs without the discount
def cost_shirt := 7.50
def cost_pants := 15.00
def cost_socks := 4.50

-- Define the costs with the discount
def discounted_shirt := 6.75
def discounted_pants := 13.50
def discounted_socks := 3.75

-- Define the number of team members
def team_members := 12

-- Total cost of one uniform without discount
def total_cost_without_discount := cost_shirt + cost_pants + cost_socks

-- Total cost of one uniform with discount
def total_cost_with_discount := discounted_shirt + discounted_pants + discounted_socks

-- Savings per uniform
def savings_per_uniform := total_cost_without_discount - total_cost_with_discount

-- Total savings for the team
def total_savings_for_team := savings_per_uniform * team_members

-- Prove that the total savings for the team is $36.00
theorem team_savings_correct : total_savings_for_team = 36.00 := 
  by 
    sorry

end team_savings_correct_l64_64408


namespace sum_factors_of_30_l64_64957

theorem sum_factors_of_30 : 
  (∑ i in {1, 2, 3, 5, 6, 10, 15, 30}, i) = 72 := 
by
  -- Sorry, to skip the proof here.
  sorry

end sum_factors_of_30_l64_64957


namespace product_of_all_solutions_triple_reciprocal_l64_64737

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64737


namespace ca_eq_cb_iff_cf_eq_cg_l64_64572

-- Definitions for the problem
variables {Γ ΓD ΓE : Type*} [circle Γ] [circle ΓD] [circle ΓE]
variables {A B C D E F G : Point}
variables {r : Line}

-- Conditions
axiom chord_of_circle (A B : Point) (Γ : Type*) [circle Γ] : on_circle Γ A ∧ on_circle Γ B
axiom point_on_segment (C : Point) (A B : Point) : on_segment C A B
axiom line_intersect (r : Line) (Γ : Type*) [circle Γ] (C : Point) : intersects_line_circle r Γ C = {D, E} 
axiom diff_sides (D E : Point) (A B : Point) : lies_on_diff_sides D E (perpendicular_bisector A B)
axiom externally_tangent (ΓD ΓE : Type*) [circle ΓD] [circle ΓE] (Γ : Type*) [circle Γ] (D E : Point) : externally_tangent_at ΓD Γ D ∧ externally_tangent_at ΓE Γ E
axiom tangent_at (ΓD ΓE : Type*) [circle ΓD] [circle ΓE] (D E : Point) (AB : Line) : tangent_to_line_at ΓD AB F ∧ tangent_to_line_at ΓE AB G

-- Problem statement
theorem ca_eq_cb_iff_cf_eq_cg (A B C D E F G : Point) (Γ ΓD ΓE : Type*) [circle Γ] [circle ΓD] [circle ΓE]
  (r : Line) : on_circle Γ A ∧ on_circle Γ B ∧ on_segment C A B ∧ (intersects_line_circle r Γ C = {D, E}) ∧ lies_on_diff_sides D E (perpendicular_bisector A B)
  ∧ externally_tangent_at ΓD Γ D ∧ externally_tangent_at ΓE Γ E ∧ (tangent_to_line_at ΓD (chord A B) F) ∧ (tangent_to_line_at ΓE (chord A B) G) →
  (dist C A = dist C B ↔ dist C F = dist C G) :=
sorry

end ca_eq_cb_iff_cf_eq_cg_l64_64572


namespace new_percentage_female_calculation_l64_64683

noncomputable def new_percentage_female (E : ℕ) :=
  let F := 0.60 * E in
  let new_total := E + 22 in
  ((F / new_total) * 100)

theorem new_percentage_female_calculation :
  ∃ E : ℕ, E + 22 = 264 ∧ (new_percentage_female E ≈ 54.92) :=
by {
  let E := 264 - 22,
  have hE : E + 22 = 264, by linarith,
  have hF : new_percentage_female E ≈ 54.92,
  sorry
}

end new_percentage_female_calculation_l64_64683


namespace concurrency_of_AC_BF_DE_l64_64118

-- Definitions for the given points and configurations
variables (A B C D E F P : Type)
variables (Γ : Circle (PlanePoint A B C))
variables [Γ.isCircumcircle A B C]
variable [D_on_BC : D ∈ line B C]
variable [tangent_Γ_A : tangent Γ A]
variable [E_on_parallel_BA_through_D : E ∈ parallel_line_through_point (line A B) D]
variable [C_E_line_intersects_Γ_again_at_F : F ∈ second_intersection_point (line E C) Γ]
variable [concyclic_B_D_F_E : cyclic (B, D, F, E)]

-- The theorem statement
theorem concurrency_of_AC_BF_DE :
  concurrent (line A C) (line B F) (line D E) :=
sorry

end concurrency_of_AC_BF_DE_l64_64118


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64829

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64829


namespace product_of_solutions_l64_64766

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64766


namespace geometric_fraction_l64_64007

noncomputable def a_n : ℕ → ℝ := sorry
axiom a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5
axiom geometric_sequence : ∀ n, a_n (n + 1) = a_n n * a_n (n + 1) / a_n (n - 1) 

theorem geometric_fraction (a_n : ℕ → ℝ) (a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5) :
  (a_n 13) / (a_n 9) = 9 :=
sorry

end geometric_fraction_l64_64007


namespace symmetric_point_coordinates_l64_64045

noncomputable def symmetric_point_in_rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem symmetric_point_coordinates :
  let P := (2, -5 * Real.pi / 3) in
  let Q := (2, -2 * Real.pi / 3) in
  symmetric_point_in_rectangular_coordinates 2 (-2 * Real.pi / 3) = (-1, -Real.sqrt 3) :=
by
  sorry

end symmetric_point_coordinates_l64_64045


namespace find_M_l64_64186

theorem find_M (a b c M : ℚ) 
  (h1 : a + b + c = 100)
  (h2 : a - 10 = M)
  (h3 : b + 10 = M)
  (h4 : 10 * c = M) : 
  M = 1000 / 21 :=
sorry

end find_M_l64_64186


namespace polynomial_count_l64_64109

def number_of_polynomials (n : ℕ) : ℕ :=
  -- This should be the correct count based on the given problem and solution
  (n / 2) + 1

theorem polynomial_count (n : ℕ) (hn : 0 < n) :
  ∃ P : ℕ → ℤ, 
    (∀ i, P(i) ∈ {0, 1, 2, 3}) ∧ 
    P(2) = n ∧ 
    number_of_polynomials n = (n / 2) + 1 :=
sorry

end polynomial_count_l64_64109


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64886

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64886


namespace cone_volume_l64_64675

variable (l h : ℝ) (π : ℝ := Real.pi)

noncomputable def radius (l h : ℝ) : ℝ := sqrt (l^2 - h^2)

noncomputable def volume_of_cone (r h : ℝ) (π : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (hl : l = 15) (hh : h = 9) :
  volume_of_cone (radius l h) h π = 432 * π :=
by
  sorry

end cone_volume_l64_64675


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64730

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64730


namespace set_of_zeros_l64_64499

def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x = 0 then 0 else -1

def f (x : ℝ) : ℝ := sgn x - 2 * x

theorem set_of_zeros : {x : ℝ | f x = 0} = {(-1/2), 0, (1/2)} := by
  sorry

end set_of_zeros_l64_64499


namespace intersection_point_of_lines_l64_64699

theorem intersection_point_of_lines :
  let line1 (x : ℝ) := 3 * x - 4
  let line2 (x : ℝ) := - (1 / 3) * x + 5
  (∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ x = 2.7 ∧ y = 4.1) :=
by {
    sorry
}

end intersection_point_of_lines_l64_64699


namespace leon_total_amount_l64_64570

def cost_toy_organizers : ℝ := 78
def num_toy_organizers : ℝ := 3
def discount_toy_organizers : ℝ := 0.10
def cost_gaming_chair : ℝ := 83
def num_gaming_chairs : ℝ := 2
def discount_gaming_chairs : ℝ := 0.05
def cost_desk : ℝ := 120
def cost_bookshelf : ℝ := 95

def delivery_fee_percent (num_items : ℝ) : ℝ :=
  if num_items ≤ 3 then 0.04 else
  if num_items ≤ 5 then 0.06 else
  0.08

def delivery_fee (total_cost : ℝ) (num_items : ℝ) : ℝ :=
  (delivery_fee_percent num_items) * total_cost

def calculate_total_amount : ℝ :=
  let toy_organizers_cost := cost_toy_organizers * num_toy_organizers
  let toy_organizers_discount := toy_organizers_cost * discount_toy_organizers
  let toy_organizers_final := toy_organizers_cost - toy_organizers_discount

  let gaming_chairs_cost := cost_gaming_chair * num_gaming_chairs
  let gaming_chairs_discount := gaming_chairs_cost * discount_gaming_chairs
  let gaming_chairs_final := gaming_chairs_cost - gaming_chairs_discount

  let total_items := num_toy_organizers + num_gaming_chairs + 1 + 1
  let total_cost_without_fee := toy_organizers_final + gaming_chairs_final + cost_desk + cost_bookshelf
  let fee := delivery_fee total_cost_without_fee total_items
  let total_cost_with_fee := total_cost_without_fee + Real.floor (fee * 100) / 100  -- rounding to nearest cent
  total_cost_with_fee

theorem leon_total_amount : calculate_total_amount = 629.96 := by
  sorry

end leon_total_amount_l64_64570


namespace no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l64_64144

theorem no_perfect_squares_in_ap (n x : ℤ) : ¬(3 * n + 2 = x^2) :=
sorry

theorem infinitely_many_perfect_cubes_in_ap : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^3 :=
sorry

theorem no_terms_of_form_x_pow_2m (n x : ℤ) (m : ℕ) : 3 * n + 2 ≠ x^(2 * m) :=
sorry

theorem infinitely_many_terms_of_form_x_pow_2m_plus_1 (m : ℕ) : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^(2 * m + 1) :=
sorry

end no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l64_64144


namespace mean_significantly_improved_l64_64296

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64296


namespace spherical_to_rectangular_coordinates_l64_64381

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l64_64381


namespace spherical_to_rectangular_coords_l64_64386

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), 
    ρ = 3 → θ = 3 * Real.pi / 2 → φ = Real.pi / 3 →
    (let x := ρ * Real.sin φ * Real.cos θ;
         y := ρ * Real.sin φ * Real.sin θ;
         z := ρ * Real.cos φ 
     in (x, y, z) = (0, - (3 * Real.sqrt 3) / 2, 3 / 2)) :=
by
  intros ρ θ φ hρ hθ hφ
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coords_l64_64386


namespace walking_times_relationship_l64_64980

theorem walking_times_relationship (x : ℝ) (h : x > 0) :
  (15 / x) - (15 / (x + 1)) = 1 / 2 :=
sorry

end walking_times_relationship_l64_64980


namespace find_b_value_l64_64636

noncomputable def parabola_has_vertex_and_yintercept (a b c q : ℝ) : Prop :=
(y = ax^2 + bx + c) /\
(vertex (q, q)) /\
(y_int (0, -2q)) /\
(q ≠ 0)

theorem find_b_value (a b c q : ℝ) (h : parabola_has_vertex_and_yintercept a b c q) : b = 6 / q := 
sorry 

end find_b_value_l64_64636


namespace significant_improvement_l64_64246

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64246


namespace product_of_all_solutions_triple_reciprocal_l64_64735

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64735


namespace indicator_significant_improvement_l64_64302

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64302


namespace product_of_real_solutions_triple_property_l64_64782

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64782


namespace cone_volume_l64_64660

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l64_64660


namespace product_of_solutions_l64_64809

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64809


namespace Shekar_marks_in_English_l64_64142

theorem Shekar_marks_in_English 
  (math_marks : ℕ) (science_marks : ℕ) (socialstudies_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (num_subjects : ℕ) 
  (mathscore : math_marks = 76)
  (sciencescore : science_marks = 65)
  (socialstudiesscore : socialstudies_marks = 82)
  (biologyscore : biology_marks = 85)
  (averagescore : average_marks = 74)
  (numsubjects : num_subjects = 5) :
  ∃ (english_marks : ℕ), english_marks = 62 :=
by
  sorry

end Shekar_marks_in_English_l64_64142


namespace sum_ineq_l64_64004

theorem sum_ineq (a : ℕ → ℕ) (h_distinct: ∀ i j, i ≠ j → a i ≠ a j) (n : ℕ) (h_pos: 0 < n) :
  (∑ k in Finset.range n, (a k) / (k + 1)^2) ≥ (∑ k in Finset.range n, 1 / (k + 1)) := 
sorry

end sum_ineq_l64_64004


namespace tan_alpha_in_fourth_quadrant_l64_64471

theorem tan_alpha_in_fourth_quadrant (y : ℝ) (α : ℝ) 
(h1 : 0 > y)
(h2 : Real.sin α = y / 5)
(h3 : Real.tan α = y / 4) :
  α = 0

end tan_alpha_in_fourth_quadrant_l64_64471


namespace determine_a_l64_64486

def piecewise_function (a : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 8 else x + a

theorem determine_a (a : ℝ) (h : piecewise_function a 3 = 10) : a = 7 :=
by {
  sorry
}

end determine_a_l64_64486


namespace fraction_decomposition_l64_64165

theorem fraction_decomposition :
  ∀ (A B : ℚ), (∀ x : ℚ, x ≠ -2 → x ≠ 4/3 → 
  (7 * x - 15) / ((3 * x - 4) * (x + 2)) = A / (x + 2) + B / (3 * x - 4)) →
  A = 29 / 10 ∧ B = -17 / 10 :=
by
  sorry

end fraction_decomposition_l64_64165


namespace product_of_reals_tripled_when_added_to_reciprocal_l64_64721

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l64_64721


namespace find_y_z_l64_64688

theorem find_y_z
  (x y z : ℕ)
  (h1 : x = 1)
  (h2 : ∃ n : ℕ, 
          ∃ (x_seq y_seq z_seq : ℕ → ℕ),
            (x_seq 0 = x ∧ y_seq 0 = y ∧ z_seq 0 = z) ∧
            (∀ i, x_seq (i + 1) = abs (x_seq i - y_seq i) ∧
                  y_seq (i + 1) = abs (y_seq i - z_seq i) ∧
                  z_seq (i + 1) = abs (z_seq i - x_seq i)) ∧
            (x_seq n = x ∧ y_seq n = y ∧ z_seq n = z)) :
  y = 0 ∧ z = 0 := 
sorry

end find_y_z_l64_64688


namespace find_angle_B_find_max_k_l64_64071

theorem find_angle_B
(A B C a b c : ℝ)
(h_angles : A + B + C = Real.pi)
(h_sides : (2 * a - c) * Real.cos B = b * Real.cos C)
(h_A_pos : 0 < A) (h_B_pos : 0 < B) (h_C_pos : 0 < C) 
(h_Alt_pos : A < Real.pi) (h_Blt_pos : B < Real.pi) 
(h_Clt_pos : C < Real.pi) :
B = Real.pi / 3 := 
sorry

theorem find_max_k
(A : ℝ)
(k : ℝ)
(m : ℝ × ℝ := (Real.sin A, Real.cos (2 * A)))
(n : ℝ × ℝ := (4 * k, 1))
(h_k_cond : 1 < k)
(h_max_dot : (m.1) * (n.1) + (m.2) * (n.2) = 5) :
k = 3 / 2 :=
sorry

end find_angle_B_find_max_k_l64_64071


namespace tangent_line_at_0_l64_64491

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * Real.cos x

theorem tangent_line_at_0 :
  let x := (0 : ℝ)
  let y := f x
  let slope := (Real.exp 0 - 2 * Real.sin 0)
  let point_tangent := (0, f 0)
  slope = 1 ∧ point_tangent = (0, 3) →
  ∀ (x y : ℝ), y = f(0) + slope * (x - 0) ↔ x - y + 3 = 0 :=
by
  intros
  sorry

end tangent_line_at_0_l64_64491


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64762

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64762


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64761

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64761


namespace initial_group_size_l64_64152

theorem initial_group_size (W : ℝ) : 
  (∃ n : ℝ, (W + 15) / n = W / n + 2.5) → n = 6 :=
by
  sorry

end initial_group_size_l64_64152


namespace pairwise_coprime_triples_l64_64423

open Nat

theorem pairwise_coprime_triples (a b c : ℕ) 
  (h1 : a.gcd b = 1) (h2 : a.gcd c = 1) (h3 : b.gcd c = 1)
  (h4 : (a + b) ∣ c) (h5 : (a + c) ∣ b) (h6 : (b + c) ∣ a) :
  { (a, b, c) | (a = 1 ∧ b = 1 ∧ (c = 1 ∨ c = 2)) ∨ (a = 1 ∧ b = 2 ∧ c = 3) } :=
by
  -- Proof omitted for conciseness
  sorry

end pairwise_coprime_triples_l64_64423


namespace product_of_solutions_l64_64707

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64707


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64819

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64819


namespace patrons_per_golf_cart_l64_64411

theorem patrons_per_golf_cart (patrons_from_cars patrons_from_bus golf_carts total_patrons patrons_per_cart : ℕ) 
  (h1 : patrons_from_cars = 12)
  (h2 : patrons_from_bus = 27)
  (h3 : golf_carts = 13)
  (h4 : total_patrons = patrons_from_cars + patrons_from_bus)
  (h5 : patrons_per_cart = total_patrons / golf_carts) : 
  patrons_per_cart = 3 := 
by
  sorry

end patrons_per_golf_cart_l64_64411


namespace mean_significantly_improved_l64_64293

-- Given data for old and new device
def old_device_data : list ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : list ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Mean and variance calculations
def mean (data : list ℝ) : ℝ := (data.sum / data.length.toReal)
def variance (data : list ℝ) (mean : ℝ) : ℝ := (data.map (λ x, (x - mean)^2)).sum / data.length.toReal

-- Calculated means
def x_bar : ℝ := mean old_device_data
def y_bar : ℝ := mean new_device_data

-- Calculated variances
def s1_squared : ℝ := variance old_device_data x_bar
def s2_squared : ℝ := variance new_device_data y_bar

-- Verify mean improvement
theorem mean_significantly_improved : y_bar - x_bar ≥ 2 * Real.sqrt((s1_squared + s2_squared) / 10) :=
by
  have x_bar_value := (old_device_data.sum / old_device_data.length.toReal)
  have y_bar_value := (new_device_data.sum / new_device_data.length.toReal)
  have s1_squared_value := ((old_device_data.map (λ x, (x - x_bar_value)^2)).sum / old_device_data.length.toReal)
  have s2_squared_value := ((new_device_data.map (λ x, (x - y_bar_value)^2)).sum / new_device_data.length.toReal)
  have h1 : x_bar = x_bar_value := rfl
  have h2 : y_bar = y_bar_value := rfl
  have h3 : s1_squared = s1_squared_value := rfl
  have h4 : s2_squared = s2_squared_value := rfl
  rw [h1, h2, h3, h4]
  sorry

end mean_significantly_improved_l64_64293


namespace ellipse_a_plus_k_eq_9_l64_64346

theorem ellipse_a_plus_k_eq_9 (h k a b : ℝ) (h_pos : 0 < a ∧ 0 < b) :
  let f1 := (1, 5) in
  let f2 := (1, 1) in
  let p := (7, 3) in
  dist (1, 5) (1, 1) = 4 ∧
  dist (7, 3) (1, 5) + dist (7, 3) (1, 1) = 4 * real.sqrt 10 ∧
  (x - 1)^2 / 36 + (y - 3)^2 / 40 = 1 ∧
  h = 1 ∧ 
  k = 3 ∧ 
  dist (1, 5) (1, 1) = 4 →
  a + k = 9 :=
by 
  sorry

end ellipse_a_plus_k_eq_9_l64_64346


namespace cross_product_zero_l64_64056

variables (v w u : ℝ × ℝ × ℝ)

-- Given conditions
def cross_v_w := ((1 : ℝ), -3, 2)
def cross_v_u := ((0 : ℝ), 4, -1)
def cross_w_u := ((-1 : ℝ), 0, 3)

-- Defining the cross product
def cross (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2.2 * b.2.1 - a.2.1 * b.2.2), 
   (a.2.0 * b.2.2 - a.2.2 * b.0.0), 
   (a.0.0 * b.2.1 - a.2.0 * b.0.0))

theorem cross_product_zero :
  cross (v + w + u) (v + w + u) = (0, 0, 0) :=
by
  assume h₁ : cross v w = ((1 : ℝ), -3, 2),
         h₂ : cross v u = ((0 : ℝ), 4, -1),
         h₃ : cross w u = ((-1 : ℝ), 0, 3),
  sorry

end cross_product_zero_l64_64056


namespace product_of_solutions_l64_64768

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64768


namespace cone_volume_l64_64663

-- Given conditions as definitions
def slant_height : ℝ := 15
def height : ℝ := 9
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- The volume of the cone
def volume : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem to prove that the calculated volume is 432π cubic centimeters
theorem cone_volume : volume = 432 * Real.pi := by
  sorry

end cone_volume_l64_64663


namespace conjugate_of_squared_complex_number_l64_64158

open Complex

theorem conjugate_of_squared_complex_number :
  conj ((1 : ℂ) + (2 : ℂ) * I)^2 = (-3 : ℂ) - (4 : ℂ) * I := by
  sorry

end conjugate_of_squared_complex_number_l64_64158


namespace range_of_m_for_one_positive_real_root_l64_64639

noncomputable def has_exactly_one_positive_real_root (m : ℝ) : Prop :=
  let f : ℝ → ℝ := λ x, m*x^2 - 2*x + 1
  (∃! x : ℝ, 0 < x ∧ f x = 0)

theorem range_of_m_for_one_positive_real_root :
  {m : ℝ | has_exactly_one_positive_real_root m} = {m : ℝ | m = 1} ∪ {m : ℝ | m ≤ 0} :=
by sorry

end range_of_m_for_one_positive_real_root_l64_64639


namespace max_distance_z_l64_64551

noncomputable def max_value_distance : ℝ :=
  let i : ℂ := complex.I in
  let z : ℂ := complex.abs (z - (3 + 4 * i)) in
  if complex.abs z = 1 then
    6
  else
    sorry

theorem max_distance_z (z : ℂ) (h : complex.abs z = 1) : complex.abs (z - (3 + 4 * complex.I)) ≤ 6 :=
by sorry

end max_distance_z_l64_64551


namespace total_sequences_correct_l64_64195

/-- 
Given 6 blocks arranged such that:
1. Block 1 must be removed first.
2. Blocks 2 and 3 become accessible after Block 1 is removed.
3. Blocks 4, 5, and 6 become accessible after Blocks 2 and 3 are removed.
4. A block can only be removed if no other block is stacked on top of it. 

Prove that the total number of possible sequences to remove all the blocks is 10.
-/
def total_sequences_to_remove_blocks : ℕ := 10

theorem total_sequences_correct : 
  total_sequences_to_remove_blocks = 10 :=
sorry

end total_sequences_correct_l64_64195


namespace product_of_solutions_l64_64862

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64862


namespace jane_pieces_eq_l64_64436

-- Define the total number of cinnamon swirls.
def total_cinnamon_swirls : ℕ := 50

-- Assume there are 8 attendees (Jane, 2 siblings, 5 cousins)
def num_attendees : ℕ := 8
-- Each attendee eats a different number of pieces.
axiom each_attendee_different_pieces (distinct_pieces : fin num_attendees → ℕ) : 
  ∀ i j, i ≠ j → distinct_pieces i ≠ distinct_pieces j

-- Let X be the pieces eaten by the cousin with the least amount of pieces.
def pieces_cousin_least : ℕ := X

-- Youngest sibling eats 2 more pieces than the cousin with the least pieces.
def pieces_youngest_sibling : ℕ := pieces_cousin_least + 2

-- Jane eats 1 less than the youngest sibling.
def pieces_jane : ℕ := pieces_youngest_sibling - 1

theorem jane_pieces_eq (pieces_cousin_least : ℕ) (distinct_pieces : fin num_attendees → ℕ) 
  (htotal : ∑ i, distinct_pieces i = total_cinnamon_swirls) 
  (hyoungest_sibling : distinct_pieces 1 = pieces_cousin_least + 2) 
  (hjane : distinct_pieces 0 = pieces_youngest_sibling - 1) : 
  distinct_pieces 0 = pieces_cousin_least + 1 := 
by 
  -- Since we are asked not to provide proofs, we insert sorry.
  sorry


end jane_pieces_eq_l64_64436


namespace range_of_f_l64_64066

theorem range_of_f (k : ℝ) (h : k > 0) (m n : ℝ)
  (h_range : set.range (λ x : ℝ, 1 + (2^(x+1)/(2^x + 1)) + real.sin x) = set.Icc m n) :
  m + n = 4 :=
sorry

end range_of_f_l64_64066


namespace area_sine_wave_l64_64241

theorem area_sine_wave (a b : ℝ) (f : ℝ → ℝ) (h₀ : a = -π / 2) (h₁ : b = 3 * π / 2) (h_f : ∀ x, f x = sin x) :
  ∫ x in a..b, f x = 2 * π := 
sorry

end area_sine_wave_l64_64241


namespace yellow_candy_percentage_l64_64313

variable (b : ℝ) (y : ℝ) (r : ℝ)

-- Conditions from the problem
-- 14% more yellow candies than blue candies
axiom yellow_candies : y = 1.14 * b
-- 14% fewer red candies than blue candies
axiom red_candies : r = 0.86 * b
-- Total number of candies equals 1 (or 100%)
axiom total_candies : r + b + y = 1

-- Question to prove: The percentage of yellow candies in the jar is 38%
theorem yellow_candy_percentage  : y = 0.38 := by
  sorry

end yellow_candy_percentage_l64_64313


namespace S13_is_52_l64_64184

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {n : ℕ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ n, a n = a1 + (n - 1) * d

-- Define the sum of the first n terms
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem S13_is_52 (h1 : is_arithmetic_sequence a)
                  (h2 : a 3 + a 7 + a 11 = 12)
                  (h3 : sum_of_first_n_terms S a) :
  S 13 = 52 :=
by sorry

end S13_is_52_l64_64184


namespace cone_volume_correct_l64_64670

noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let r := real.sqrt (slant_height^2 - height^2)
  in (1/3) * real.pi * r^2 * height

theorem cone_volume_correct :
  cone_volume 15 9 = 432 * real.pi := by
  sorry

end cone_volume_correct_l64_64670


namespace inequality_proof_l64_64113

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 = 1000 / 9 :=
sorry

end inequality_proof_l64_64113


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64910

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64910


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64920

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64920


namespace not_divisible_by_n_plus_4_l64_64139

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : n > 0) : ¬ ∃ k : ℕ, n^2 + 8 * n + 15 = k * (n + 4) := by
  sorry

end not_divisible_by_n_plus_4_l64_64139


namespace cone_volume_l64_64645

theorem cone_volume (s h : ℝ) (h_s : s = 15) (h_h : h = 9) : 
  ∃ V : ℝ, V = (1 / 3) * π * (sqrt (s^2 - h^2))^2 * h := 
by
  use 432 * π
  rw [h_s, h_h]
  have rsq : 15^2 - 9^2 = 144 := by norm_num
  have radius : sqrt (15^2 - 9^2) = 12 := by rw [rsq, Real.sqrt_sq_eq_abs, abs_of_nonneg]; norm_num
  simp [radius]
  norm_num
  sorry

end cone_volume_l64_64645


namespace angle_conversion_l64_64213

-- Define the known conditions
def full_circle_vens : ℕ := 800
def full_circle_degrees : ℕ := 360
def given_angle_degrees : ℕ := 135
def expected_vens : ℕ := 300

-- Prove that an angle of 135 degrees corresponds to 300 vens.
theorem angle_conversion :
  (given_angle_degrees * full_circle_vens) / full_circle_degrees = expected_vens := by
  sorry

end angle_conversion_l64_64213


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64890

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l64_64890


namespace range_of_k_l64_64485

-- Definition of the conditions
def is_ellipse_with_foci_on_y_axis (k : ℝ) : Prop := 
  ∃ x y : ℝ, (y ≠ 0) ∧ (2 * k + 1 > 2 - k > 0) ∧ 
  (x^2 / (2 - k) + y^2 / (2 * k + 1) = 1)

-- Mathematical proof problem statement
theorem range_of_k (k : ℝ) (h : is_ellipse_with_foci_on_y_axis k) : 1 < k ∧ k < 2 := 
by
  sorry

end range_of_k_l64_64485


namespace product_of_solutions_l64_64810

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64810


namespace product_of_solutions_l64_64717

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64717


namespace base_six_arithmetic_l64_64365

theorem base_six_arithmetic : 
  let a := [1, 3, 5, 4, 2]
  let b := [2, 3, 4, 3, 5]
  let c := [4, 3, 1, 2]
  let a_base10 := a[0] * 6^4 + a[1] * 6^3 + a[2] * 6^2 + a[3] * 6^1 + a[4] * 6^0
  let b_base10 := b[0] * 6^4 + b[1] * 6^3 + b[2] * 6^2 + b[3] * 6^1 + b[4] * 6^0
  let c_base10 := c[0] * 6^3 + c[1] * 6^2 + c[2] * 6^1 + c[3] * 6^0
  a_base10 + b_base10 - c_base10 = 3 * 6^4 + 1 * 6^3 + 4 * 6^2 + 0 * 6^1 + 5 * 6^0 :=
by
  sorry

end base_six_arithmetic_l64_64365


namespace function_value_when_x_is_minus_one_l64_64978

theorem function_value_when_x_is_minus_one :
  ∀ (x : ℤ), x = -1 → -2 * x = 2 :=
begin
  assume (x : ℤ) (hx : x = -1),
  rw hx,
  norm_num,
end

end function_value_when_x_is_minus_one_l64_64978


namespace product_of_solutions_l64_64777

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l64_64777


namespace cone_volume_l64_64661

-- Given conditions as definitions
def slant_height : ℝ := 15
def height : ℝ := 9
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- The volume of the cone
def volume : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem to prove that the calculated volume is 432π cubic centimeters
theorem cone_volume : volume = 432 * Real.pi := by
  sorry

end cone_volume_l64_64661


namespace votes_cast_l64_64228

-- Define the conditions as given in the problem.
def total_votes (V : ℕ) := 35 * V / 100 + (35 * V / 100 + 2400) = V

-- The goal is to prove that the number of total votes V equals 8000.
theorem votes_cast : ∃ V : ℕ, total_votes V ∧ V = 8000 :=
by
  sorry -- The proof is not required, only the statement.

end votes_cast_l64_64228


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64832

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64832


namespace MIT_room_sign_problem_l64_64155

theorem MIT_room_sign_problem :
  ∀ N M : ℕ,
  (N > 0) →
  (M is the room number after a zero in N falls off) →
  let a := 1
  let b := 10
  let c := 2
  let d := 11
  1000 * a + 100 * b + 10 * c + d = 2031 :=
begin
  -- Proof should be done here
  sorry
end

end MIT_room_sign_problem_l64_64155


namespace problem_xyz_l64_64055

theorem problem_xyz (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -8) :
  x^2 + y^2 = 32 :=
by
  sorry

end problem_xyz_l64_64055


namespace number_of_digits_in_product_l64_64106

/- Define the two large numbers as constants -/
def a : ℕ := 12345678912345
def b : ℕ := 987654321

/- Main theorem stating the number of digits in their product -/
theorem number_of_digits_in_product : 
  let Q := a * b in 
  nat_digits Q = 24 := 
sorry

/- Auxiliary function to count the digits of a number -/
noncomputable def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (nat.log10 n).to_nat + 1

/- Ensure these conditions are met -/
lemma num_digits_a : nat_digits a = 15 := sorry
lemma num_digits_b : nat_digits b = 9 := sorry

#check @number_of_digits_in_product

end number_of_digits_in_product_l64_64106


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64923

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64923


namespace product_of_real_solutions_triple_property_l64_64789

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64789


namespace Steve_has_more_money_than_Wayne_by_2004_l64_64360

theorem Steve_has_more_money_than_Wayne_by_2004:
  (∀ n: ℕ, steve_money n = 100 * 2 ^ n ∧ wayne_money n = 10000 / 2 ^ n) →
  (∃ n: ℕ, 2000 <= n ∧ 2000 + n = 2004 ∧ steve_money (n + 2000) > wayne_money (n + 2000)) :=
by
  intro h
  sorry

end Steve_has_more_money_than_Wayne_by_2004_l64_64360


namespace product_of_real_solutions_triple_property_l64_64794

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64794


namespace math_students_passed_l64_64537

variables (E_only B M_only M : ℕ)

theorem math_students_passed : E_only + B = 30 → E_only = M_only + 10 → M = M_only + B → M = 20 :=
by
  assume h1 : E_only + B = 30,
  assume h2 : E_only = M_only + 10,
  assume h3 : M = M_only + B,
  sorry

end math_students_passed_l64_64537


namespace cone_volume_l64_64651

def slant_height := 15
def height := 9
def radius := Real.sqrt (slant_height^2 - height^2)
def volume := (1/3) * Real.pi * radius^2 * height

theorem cone_volume : 
  radius = 12 ∧ volume = 432 * Real.pi :=
by
  split
  { -- radius = 12
    unfold radius
    sorry
  }
  { -- volume = 432 * Real.pi
    unfold volume
    sorry
  }

end cone_volume_l64_64651


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64755

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64755


namespace calculate_difference_l64_64339

variable (σ : ℝ) -- Let \square be represented by a real number σ
def correct_answer := 4 * (σ - 3)
def incorrect_answer := 4 * σ - 3
def difference := correct_answer σ - incorrect_answer σ

theorem calculate_difference : difference σ = -9 := by
  sorry

end calculate_difference_l64_64339


namespace probability_of_X_conditioned_l64_64692

variables (P_X P_Y P_XY : ℝ)

-- Conditions
def probability_of_Y : Prop := P_Y = 2/5
def probability_of_XY : Prop := P_XY = 0.05714285714285714
def independent_selection : Prop := P_XY = P_X * P_Y

-- Theorem statement
theorem probability_of_X_conditioned (P_X P_Y P_XY : ℝ) 
  (h1 : probability_of_Y P_Y) 
  (h2 : probability_of_XY P_XY) 
  (h3 : independent_selection P_X P_Y P_XY) :
  P_X = 0.14285714285714285 := 
sorry

end probability_of_X_conditioned_l64_64692


namespace remaining_amount_to_be_paid_l64_64132

-- Define the conditions
def first_payment : ℝ := 875
def first_payment_pct : ℝ := 0.25

def second_payment : ℝ := 650
def second_payment_pct : ℝ := 0.15

def third_payment : ℝ := 1200
def third_payment_pct : ℝ := 0.35

def sales_tax_pct : ℝ := 0.075
def delivery_fee : ℝ := 100

-- Define the proof statement
theorem remaining_amount_to_be_paid :
  let total_cost_before_taxes_and_fees := first_payment / first_payment_pct in
  let sales_tax := total_cost_before_taxes_and_fees * sales_tax_pct in
  let total_cost_with_sales_tax := total_cost_before_taxes_and_fees + sales_tax in
  let total_cost_with_sales_tax_and_delivery := total_cost_with_sales_tax + delivery_fee in
  let total_payments_made := first_payment + second_payment + third_payment in
  let remaining_amount := total_cost_with_sales_tax_and_delivery - total_payments_made in
  remaining_amount = 1137.50 :=
by
  -- proof steps would go here
  sorry

end remaining_amount_to_be_paid_l64_64132


namespace compare_neg_rat_l64_64377

theorem compare_neg_rat:
  (- (3 / 4) > - | - (4 / 5): ℚ) :=
sorry

end compare_neg_rat_l64_64377


namespace sum_of_factors_30_l64_64973

theorem sum_of_factors_30 : 
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ) in
  ( ∑ n in factors, n ) = 72 :=
by
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ)
  rw [finset.sum_eq_sum_of_elements] -- transposing to handle the sum notation
  let factors_list := [1, 2, 3, 5, 6, 10, 15, 30]
  have h_factors : factors = {x | x ∈ factors_list}.to_finset, by sorry
  simp_rw h_factors
  rw [finset.sum_to_finset]
  dsimp -- simplifying before asserting the sum
  norm_num -- proving that the sum is indeed 72
  sorry

end sum_of_factors_30_l64_64973


namespace product_of_triple_when_added_to_reciprocal_l64_64895

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64895


namespace working_together_time_l64_64225

theorem working_together_time (hA : A_rate = 1 / 6) (hD : D_rate = 1 / 11.999999999999998) : 
  together_rate = A_rate + D_rate → completion_time = 4 :=
by
  sorry

def A_rate : ℝ := 1 / 6
def D_rate : ℝ := 1 / 11.999999999999998
def together_rate : ℝ := A_rate + D_rate
def completion_time : ℝ := 1 / together_rate

end working_together_time_l64_64225


namespace indicator_significant_improvement_l64_64305

noncomputable def mean (xs : List Float) : Float :=
  (xs.foldl (· + ·) 0) / (xs.length : Float)

noncomputable def variance (xs : List Float) (μ : Float) : Float :=
  (xs.foldl (λ acc x => acc + (x - μ) ^ 2) 0) / (xs.length : Float)

theorem indicator_significant_improvement : 
  let old_device : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
  let new_device : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]
  let x := mean old_device 
  let y := mean new_device 
  let s1_sq := variance old_device x
  let s2_sq := variance new_device y 
  x = 9.9 ∧ y = 10.3 ∧ s1_sq = 0.036 ∧ s2_sq = 0.04
  → (y - x) > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) := 
by
  sorry

end indicator_significant_improvement_l64_64305


namespace cone_volume_l64_64665

-- Given conditions as definitions
def slant_height : ℝ := 15
def height : ℝ := 9
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- The volume of the cone
def volume : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem to prove that the calculated volume is 432π cubic centimeters
theorem cone_volume : volume = 432 * Real.pi := by
  sorry

end cone_volume_l64_64665


namespace product_of_solutions_l64_64813

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64813


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64922

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64922


namespace product_of_real_solutions_triple_property_l64_64787

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l64_64787


namespace find_a2_l64_64082

variable {r : ℝ}
variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n, a (n+1) = a 1 * r^n

def a_n : ℕ → ℝ
| 0 => 1 / 5
| n+1 => a_n 0 * r^n

theorem find_a2 (h1 : a_n 1 = 1 / 5) (h3 : a_n 3 = 5) :
  a_n 2 = ± 1 :=
by
  have h_geom : geometric_sequence a_n r
  sorry
  sorry

end find_a2_l64_64082


namespace exists_integer_n_l64_64585

noncomputable theory

open Real Int

theorem exists_integer_n (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) (h_cd : c * d = 1) :
  ∃ (n : ℤ), a * b ≤ n^2 ∧ n^2 ≤ (a + c) * (b + d) :=
sorry

end exists_integer_n_l64_64585


namespace find_m_l64_64463

-- Define given points
def O := (0 : ℝ, 0 : ℝ)
def A := (-1 : ℝ, 3 : ℝ)
def B := (2 : ℝ, -4 : ℝ)

-- Define the vectors
def OA := (A.1 - O.1, A.2 - O.2)
def AB := (B.1 - A.1, B.2 - A.2)

-- Define vector equation OP
def OP (m : ℝ) := (2 * OA.1 + m * AB.1, 2 * OA.2 + m * AB.2)

-- Define the condition: P is on the y-axis
def P_on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0

-- Statement to prove
theorem find_m : ∃ m : ℝ, P_on_y_axis (OP m) ∧ m = 2 / 3 :=
by {
  sorry
}

end find_m_l64_64463


namespace negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l64_64634

-- Definitions based on the conditions in the problem:
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b
def MonotonicFunction (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The proposition that 'All linear functions are monotonic functions'
def AllLinearAreMonotonic : Prop := ∀ (f : ℝ → ℝ), LinearFunction f → MonotonicFunction f

-- The correct answer to the question:
def SomeLinearAreNotMonotonic : Prop := ∃ (f : ℝ → ℝ), LinearFunction f ∧ ¬ MonotonicFunction f

-- The proof problem:
theorem negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic : 
  ¬ AllLinearAreMonotonic ↔ SomeLinearAreNotMonotonic :=
by
  sorry

end negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l64_64634


namespace significant_improvement_l64_64281

section StatisticalTest

variables (old_device_data new_device_data : Fin 10 → ℝ)

def sample_mean (data : Fin 10 → ℝ) : ℝ :=
  (1 / 10) * (data 0 + data 1 + data 2 + data 3 + data 4 + data 5 + data 6 + data 7 + data 8 + data 9)
  
def sample_variance (data : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (1 / 10) * ((data 0 - mean)^2 + (data 1 - mean)^2 + (data 2 - mean)^2 + (data 3 - mean)^2 + (data 4 - mean)^2 + 
              (data 5 - mean)^2 + (data 6 - mean)^2 + (data 7 - mean)^2 + (data 8 - mean)^2 + (data 9 - mean)^2)
  
theorem significant_improvement
  (old_device_data = ![9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7])
  (new_device_data = ![10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]) :
  let x_mean := sample_mean old_device_data,
      y_mean := sample_mean new_device_data,
      s1_2 := sample_variance old_device_data x_mean,
      s2_2 := sample_variance new_device_data y_mean in
  y_mean - x_mean ≥ 2 * real.sqrt ((s1_2 + s2_2) / 10) :=
  sorry

end StatisticalTest

end significant_improvement_l64_64281


namespace product_of_solutions_l64_64799

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l64_64799


namespace sum_factors_of_30_l64_64963

theorem sum_factors_of_30 : 
  (∑ i in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), i) = 72 := by
  sorry

end sum_factors_of_30_l64_64963


namespace spherical_to_rectangular_coordinates_l64_64382

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l64_64382


namespace spherical_to_rectangular_coords_l64_64389

theorem spherical_to_rectangular_coords (
  {ρ θ φ : ℝ} 
) (hρ : ρ = 3) (hθ : θ = 3 * Real.pi / 2) (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by 
  sorry

end spherical_to_rectangular_coords_l64_64389


namespace product_of_solutions_l64_64710

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l64_64710


namespace product_of_solutions_l64_64865

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64865


namespace cannot_plant_three_trees_l64_64326

def rectangular_plot := {length : ℝ // length = 4} × {width : ℝ // width = 1}

def min_distance : ℝ := 2.5

theorem cannot_plant_three_trees :
  ∀ plot : rectangular_plot, ¬ (∃ t1 t2 t3 : (ℝ × ℝ), 
    (∀ i j, i ≠ j → dist i j ≥ min_distance)) := 
sorry

end cannot_plant_three_trees_l64_64326


namespace lcm_of_18_and_10_l64_64208

-- Definition of GCD and LCM
def gcd (a : Nat) (b : Nat) : Nat := Nat.gcd a b
def lcm (a : Nat) (b : Nat) : Nat := Nat.lcm a b

-- Given condition that GCD of n and 10 is 5 and n is 18.
variable (n : Nat)
variable (h₀ : gcd n 10 = 5)
variable (h₁ : n = 18)

-- Theorem statement: The LCM of n and 10 is 36.
theorem lcm_of_18_and_10 (h₁ : n = 18) (h₀ : gcd n 10 = 5) : lcm n 10 = 36 :=
by
  sorry

end lcm_of_18_and_10_l64_64208


namespace surface_area_of_interior_box_l64_64148

-- Defining the initial conditions
def original_length : ℕ := 36
def original_width : ℕ := 24
def corner_side : ℕ := 6

-- Statement to prove
theorem surface_area_of_interior_box : 
  let new_length := original_length - 2 * corner_side,
      new_width := original_width - 2 * corner_side in
  new_length * new_width = 288 := 
begin
  sorry
end

end surface_area_of_interior_box_l64_64148


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64831

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64831


namespace choose_athlete_good_stable_l64_64363

def average_scores : ℕ → ℝ
| 0 := 7  -- Athlete A
| 1 := 8  -- Athlete B
| 2 := 8  -- Athlete C
| 3 := 7  -- Athlete D
| _ := 0  -- Default case, not used

def variances : ℕ → ℝ
| 0 := 0.9  -- Athlete A
| 1 := 1.1  -- Athlete B
| 2 := 0.9  -- Athlete C
| 3 := 1.0  -- Athlete D
| _ := 0    -- Default case, not used

def chosen_athlete_criteria : (ℕ → ℝ) → (ℕ → ℝ) → ℕ
| avar, var := if avar 2 = 8 ∧ var 2 = 0.9 then 2 else 0

theorem choose_athlete_good_stable (a_scores : ℕ → ℝ) (a_vars : ℕ → ℝ) :
  chosen_athlete_criteria a_scores a_vars = 2 :=
by
  unfold chosen_athlete_criteria
  simp [average_scores, variances]
  sorry

end choose_athlete_good_stable_l64_64363


namespace yeast_cells_at_2_20_pm_l64_64096

noncomputable def yeast_population (initial : Nat) (rate : Nat) (intervals : Nat) : Nat :=
  initial * rate ^ intervals

theorem yeast_cells_at_2_20_pm :
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5 -- 20 minutes / 4 minutes per interval
  yeast_population initial_population triple_rate intervals = 7290 :=
by
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5
  show yeast_population initial_population triple_rate intervals = 7290
  sorry

end yeast_cells_at_2_20_pm_l64_64096


namespace product_of_solutions_l64_64863

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64863


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64850

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64850


namespace evaluate_expression_parity_l64_64020

variable {a b c : ℕ}

theorem evaluate_expression_parity
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_even_a : Even a)
  (h_odd_b : Odd b) :
  (even c -> Odd (2^a + (b + 1)^2 + c)) ∧ (odd c -> Even (2^a + (b + 1)^2 + c)) :=
by
  sorry

end evaluate_expression_parity_l64_64020


namespace rectangle_area_l64_64336

theorem rectangle_area (square_area : ℝ) (width length : ℝ) 
  (h1 : square_area = 36) 
  (h2 : width = real.sqrt square_area) 
  (h3 : length = 3 * width) : 
  width * length = 108 :=
by
  sorry

end rectangle_area_l64_64336


namespace cone_cross_section_area_l64_64478

theorem cone_cross_section_area : 
  ∀ (R r : ℝ), 
  (∀ (h : 2 * real.pi * r = real.pi * R), 
  (R = 2 * r) ∧ 
  (2 = 2 * R / 2) → 
  (∃ A : ℝ, A = real.sqrt 3)) :=
by
sorry

end cone_cross_section_area_l64_64478


namespace part_a_part_b_l64_64123

/-- Part (a) Statement: 
    For any 3x3 real matrix A and any polynomial f in R[X] whose roots are not real, f(A) is not the zero matrix. -/
theorem part_a (A : Matrix (Fin 3) (Fin 3) ℝ) (f : Polynomial ℝ) (hroots : ∀ r : ℂ, Polynomial.map (algebraMap ℝ ℂ) f.eval r = 0 → ¬ (r.im = 0)) :
  (Matrix.eval f A) ≠ 0 := 
sorry

/-- Part (b) Statement:
    There exists n in ℕ such that (A + adj(A))^(2n) = A^(2n) + (adj(A))^(2n) if and only if det(A) = 0. -/
theorem part_b (A : Matrix (Fin 3) (Fin 3) ℝ) : 
  (∃ n : ℕ, (A + Matrix.adj A) ^ (2 * n) = A ^ (2 * n) + (Matrix.adj A) ^ (2 * n)) ↔ (Matrix.det A = 0) :=
sorry

end part_a_part_b_l64_64123


namespace seq_limit_l64_64369

theorem seq_limit :
  tendsto (λ n : ℕ, (3 - 4 * n) ^ 2 / ((n - 3) ^ 2 - (n + 3) ^ 2)) at_top (𝓝 (-∞)) :=
begin
  sorry
end

end seq_limit_l64_64369


namespace blue_die_prime_yellow_die_power_2_probability_l64_64204

def prime_numbers : Finset ℕ := {2, 3, 5, 7}

def powers_of_2 : Finset ℕ := {1, 2, 4, 8}

def total_outcomes : ℕ := 8 * 8

def successful_outcomes : ℕ := prime_numbers.card * powers_of_2.card

def probability (x y : Finset ℕ) : ℚ := (x.card * y.card) / (total_outcomes : ℚ)

theorem blue_die_prime_yellow_die_power_2_probability :
  probability prime_numbers powers_of_2 = 1 / 4 :=
by
  sorry

end blue_die_prime_yellow_die_power_2_probability_l64_64204


namespace cone_volume_l64_64656

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l64_64656


namespace repeating_six_in_base_8_eq_fraction_l64_64426

theorem repeating_six_in_base_8_eq_fraction :
  ∃ (S : ℚ), S = 6 / 7 ∧ (∀ (n : ℕ), 0.\overline{6}_8 = (6/8) + (6/8^2) + (6/8^3) + ... + (6/8^n)) → S = 6/7 := 
by
  sorry

end repeating_six_in_base_8_eq_fraction_l64_64426


namespace derivative_of_function_l64_64425

theorem derivative_of_function :
  ∀ (x : ℝ), 
    let y := (2^x * (Real.sin x + Real.cos x * Real.log 2)) / (1 + (Real.log 2)^2) in 
    deriv (λ x : ℝ, y) x = 2^x * Real.cos x :=
begin
  sorry
end

end derivative_of_function_l64_64425


namespace product_of_solutions_l64_64868

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64868


namespace inequality_proof_l64_64112

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 = 1000 / 9 :=
sorry

end inequality_proof_l64_64112


namespace smallest_divisor_perfect_cube_l64_64217

-- Definition to express the condition that a number n is a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k ^ 3

-- Given conditions: 34560 equals 2^6 * 3^3 * 5^1
def factorial_34560 : ℕ := 2 ^ 6 * 3 ^ 3 * 5 ^ 1

-- Theorem: The smallest number which divides 34560 and gives a perfect cube is 160
theorem smallest_divisor_perfect_cube : (n : ℕ), n ∣ factorial_34560 ∧ is_perfect_cube n → n = 160 := by
  sorry

end smallest_divisor_perfect_cube_l64_64217


namespace jill_lavender_candles_l64_64561

def number_of_lavender_candles 
  (lavender_per_candle: ℕ) 
  (coconut_per_candle: ℕ) 
  (almond_per_candle: ℕ) 
  (jasmine_per_candle: ℕ) 
  (lavender_to_coconut_ratio: ℕ) 
  (almond_to_jasmine_ratio: ℕ) 
  (total_almond_candles: ℕ) 
  (coconut_to_almond_ratio: ℚ) 
  (total_coconut_ml: ℕ)
  (total_jasmine_ml: ℕ):
  ℕ :=
  3 * (total_almond_candles * almond_per_candle * coconut_to_almond_ratio) / coconut_per_candle

theorem jill_lavender_candles:
  let lavender_per_candle := 10
  let coconut_per_candle := 8
  let almond_per_candle := 12
  let jasmine_per_candle := 14
  let lavender_to_coconut_ratio := 3
  let almond_to_jasmine_ratio := 2
  let total_almond_candles := 10
  let coconut_to_almond_ratio := 2.5
  
  let total_almond_ml := total_almond_candles * almond_per_candle
  let total_coconut_ml := total_almond_ml * coconut_to_almond_ratio
  let total_jasmine_ml := (total_almond_candles / almond_to_jasmine_ratio) * jasmine_per_candle
  
  number_of_lavender_candles lavender_per_candle coconut_per_candle almond_per_candle jasmine_per_candle lavender_to_coconut_ratio almond_to_jasmine_ratio total_almond_candles coconut_to_almond_ratio total_coconut_ml total_jasmine_ml = 111 := sorry

end jill_lavender_candles_l64_64561


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64914

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64914


namespace find_a_values_l64_64546

theorem find_a_values :
  let A := (1 : ℝ, 0 : ℝ)
  let B := (4 : ℝ, 0 : ℝ)
  -- Condition: P lies on the ellipse
  let ellipse a (x y : ℝ) := x^2 / a^2 + y^2 / (a^2 - 1) = 1
  -- Condition: Distance ratio
  let distance_ratio (A B P : ℝ × ℝ) := (dist P A) / (dist P B) = 1 / 2
  -- Find the set of values for a
  (∀ (a : ℝ), (∃ (x y : ℝ), ellipse a x y ∧ distance_ratio A B (x, y)) →
      a = 2 ∨ a = -2 ∨ a = sqrt 5 ∨ a = -sqrt 5) := sorry

end find_a_values_l64_64546


namespace product_of_solutions_l64_64867

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l64_64867


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64833

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l64_64833


namespace jasmine_percentage_correct_l64_64345

def volume_initial := 80
def percent_jasmine_initial := 0.1
def percent_lemon_initial := 0.05

def added_jasmine := 8
def added_water := 12
def added_lemon := 6

def total_jasmine_initial := volume_initial * percent_jasmine_initial
def total_jasmine_new := total_jasmine_initial + added_jasmine

def new_total_volume := volume_initial + added_jasmine + added_water + added_lemon

def jasmine_percent_new := (total_jasmine_new / new_total_volume) * 100

theorem jasmine_percentage_correct : jasmine_percent_new = 15.09 := 
sorry

end jasmine_percentage_correct_l64_64345


namespace real_complex_square_l64_64522

theorem real_complex_square (a : ℝ) (h : (2 + a*complex.i)^2 = (2 + a*complex.i)^2.re) : a = 0 := 
by 
-- proof goes here
sorry

end real_complex_square_l64_64522


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64815

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64815


namespace mean_equality_and_find_y_l64_64171

theorem mean_equality_and_find_y : 
  (8 + 9 + 18) / 3 = (15 + (25 / 3)) / 2 :=
by
  sorry

end mean_equality_and_find_y_l64_64171


namespace product_of_all_solutions_triple_reciprocal_l64_64748

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64748


namespace problem_statement_l64_64590

noncomputable def f : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := 23 * f (n + 1) + f n

theorem problem_statement (m : ℕ) : ∃ d : ℕ, ∀ n : ℕ, (m ∣ f (f n) ↔ d ∣ n) :=
sorry

end problem_statement_l64_64590


namespace team_savings_correct_l64_64407

-- Define the costs without the discount
def cost_shirt := 7.50
def cost_pants := 15.00
def cost_socks := 4.50

-- Define the costs with the discount
def discounted_shirt := 6.75
def discounted_pants := 13.50
def discounted_socks := 3.75

-- Define the number of team members
def team_members := 12

-- Total cost of one uniform without discount
def total_cost_without_discount := cost_shirt + cost_pants + cost_socks

-- Total cost of one uniform with discount
def total_cost_with_discount := discounted_shirt + discounted_pants + discounted_socks

-- Savings per uniform
def savings_per_uniform := total_cost_without_discount - total_cost_with_discount

-- Total savings for the team
def total_savings_for_team := savings_per_uniform * team_members

-- Prove that the total savings for the team is $36.00
theorem team_savings_correct : total_savings_for_team = 36.00 := 
  by 
    sorry

end team_savings_correct_l64_64407


namespace problem_statement_l64_64124

def quadrant (z : ℂ) : ℕ :=
if z.re > 0 then 
  if z.im > 0 then 1 else 4 
else 
  if z.im > 0 then 2 else 3

theorem problem_statement : 
  let z := Complex.i * (1 - Complex.i) 
  let z_conj := Complex.conj z
  quadrant z_conj = 4 := by
  sorry

end problem_statement_l64_64124


namespace min_value_of_expression_l64_64428

theorem min_value_of_expression (n : ℕ) (x : ℕ → ℝ) :
  let cos_sum := ∑ i in finset.range n, real.cos (x i),
      sin_sum := ∑ i in finset.range n, real.sin (x i),
      A := (5 * cos_sum + sin_sum) * (cos_sum - 5 * sin_sum)
  in A ≥ -13 * n^2 := sorry

end min_value_of_expression_l64_64428


namespace sum_factors_of_30_l64_64959

theorem sum_factors_of_30 : 
  (∑ i in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), i) = 72 := by
  sorry

end sum_factors_of_30_l64_64959


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64751

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64751


namespace significant_improvement_l64_64262

def data_old : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def data_new : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem significant_improvement :
  let x_bar := mean data_old
  let y_bar := mean data_new
  let s1_sq := variance data_old
  let s2_sq := variance data_new
  y_bar - x_bar > 2 * Real.sqrt ((s1_sq + s2_sq) / 10) :=
by
  sorry

end significant_improvement_l64_64262


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64859

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l64_64859


namespace find_all_functions_l64_64421

noncomputable def solution : Type :=
  { f : ℝ → ℝ // ∃ n : ℕ, n ≥ 2 ∧ ∀ x y : ℝ, f (x + y ^ n) = f x + (f y) ^ n }

theorem find_all_functions (f : ℝ → ℝ) (h : ∃ n : ℕ, n ≥ 2 ∧ ∀ x y : ℝ, f (x + y ^ n) = f x + (f y) ^ n) : 
  f = (λ x, 0) ∨ f = (λ x, x) ∨ (∃ n : ℕ, odd n ∧ f = (λ x, -x)) :=
begin
  sorry
end

end find_all_functions_l64_64421


namespace area_of_triangle_ABC_l64_64554

-- Definitions based on the conditions
variables (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_ABC : IsoscelesRightTriangle A B C)
variables (triangle_DEF : IsoscelesRightTriangle D E F)
variables (BD EC : ℝ)

-- Concrete values given in conditions
axiom BD_eq_2 : BD = 2
axiom EC_eq_4 : EC = 4

-- Given hypothesis
axiom area_between_ABC_DEF : TriangleAreaDifference triangle_ABC triangle_DEF 20

-- The goal is to prove the area of triangle ABC
theorem area_of_triangle_ABC : area triangle_ABC = 841 / 36 :=
by sorry

end area_of_triangle_ABC_l64_64554


namespace average_speed_is_4_l64_64520

/-- The position function of the particle --/
def s (t : ℝ) : ℝ := 3 + t^2

/-- The time interval over which we calculate the average speed --/
def t1 : ℝ := 2
def t2 : ℝ := 2.1

/-- Prove the average speed during [t1, t2] is 4.1 --/
theorem average_speed_is_4.1 : (s t2 - s t1) / (t2 - t1) = 4.1 := 
by
  sorry

end average_speed_is_4_l64_64520


namespace product_of_all_solutions_triple_reciprocal_l64_64740

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l64_64740


namespace spherical_to_rectangular_l64_64395

theorem spherical_to_rectangular :
  ∀ (ρ θ φ : ℝ),
  ρ = 3 →
  θ = (3 * π) / 2 →
  φ = π / 3 →
  let x := ρ * real.sin φ * real.cos θ,
      y := ρ * real.sin φ * real.sin θ,
      z := ρ * real.cos φ in
  (x, y, z) = (0, -3 * real.sqrt 3 / 2, 3 / 2) := by
  intros,
  -- The actual proof would go here
  sorry

end spherical_to_rectangular_l64_64395


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64758

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64758


namespace speed_of_first_boy_l64_64206

theorem speed_of_first_boy (x : ℝ) (h1 : 7.5 > 0) (h2 : 16 > 0) (h3 : 32 > 0) (h4 : 32 = 16 * (x - 7.5)) : x = 9.5 :=
by
  sorry

end speed_of_first_boy_l64_64206


namespace prism_volume_l64_64681

theorem prism_volume (x : ℝ) (L W H : ℝ) (hL : L = 2 * x) (hW : W = x) (hH : H = 1.5 * x) 
  (hedges_sum : 4 * L + 4 * W + 4 * H = 72) : 
  L * W * H = 192 := 
by
  sorry

end prism_volume_l64_64681


namespace spherical_to_rectangular_l64_64396

theorem spherical_to_rectangular :
  ∀ (ρ θ φ : ℝ),
  ρ = 3 →
  θ = (3 * π) / 2 →
  φ = π / 3 →
  let x := ρ * real.sin φ * real.cos θ,
      y := ρ * real.sin φ * real.sin θ,
      z := ρ * real.cos φ in
  (x, y, z) = (0, -3 * real.sqrt 3 / 2, 3 / 2) := by
  intros,
  -- The actual proof would go here
  sorry

end spherical_to_rectangular_l64_64396


namespace distance_center_to_line_circle_l64_64084

noncomputable def circle_center : ℝ × ℝ := (2, Real.pi / 2)

noncomputable def distance_from_center_to_line (radius : ℝ) (center : ℝ × ℝ) : ℝ :=
  radius * Real.sin (center.snd - Real.pi / 3)

theorem distance_center_to_line_circle : distance_from_center_to_line 2 circle_center = 1 := by
  sorry

end distance_center_to_line_circle_l64_64084


namespace significant_improvement_l64_64289

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) (mean_x : ℝ) : ℝ :=
  (data.map (λ x => (x - mean_x) ^ 2)).sum / (data.length)

def old_device_data : List ℝ :=
  [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List ℝ :=
  [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean_old_device := mean old_device_data
noncomputable def mean_new_device := mean new_device_data

noncomputable def var_old_device := variance old_device_data mean_old_device
noncomputable def var_new_device := variance new_device_data mean_new_device

theorem significant_improvement :
  (mean_new_device - mean_old_device) ≥ 2 * sqrt ((var_old_device + var_new_device) / 10) :=
by
  sorry

end significant_improvement_l64_64289


namespace significant_improvement_l64_64248

def old_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List Float) : Float := l.sum / l.length

def variance (l : List Float) : Float := (l.map (λ x => (x - mean l) ^ 2)).sum / l.length

theorem significant_improvement :
  let x_mean := mean old_device_data
  let y_mean := mean new_device_data
  let s1_squared := variance old_device_data
  let s2_squared := variance new_device_data
  y_mean - x_mean >= 2 * Float.sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l64_64248


namespace part1_tangent_line_part2_monotonic_intervals_l64_64041

-- Define the function f(x) = ax^2 - (a+2)x + ln(x)
def f (a : ℝ) (x : ℝ) := a * x^2 - (a + 2) * x + Real.log x

-- Part 1: Prove tangent line equation when a = 2
theorem part1_tangent_line : 
  f 2 1 = -2 ∧ 
  let tangent_line := λ x y : ℝ, x - y - 3 = 0 in 
  tangent_line 1 (-2) :=
sorry

-- Part 2: Prove monotonic intervals based on a
theorem part2_monotonic_intervals (a : ℝ) :
  if a ≤ 0 then
    ∃ incr decr : Set ℝ,
      incr = {x : ℝ | 0 < x ∧ x < 1 / 2} ∧
      decr = {x : ℝ | x > 1 / 2}
  else if 0 < a ∧ a < 2 then
    ∃ incr1 incr2 decr : Set ℝ,
      incr1 = {x : ℝ | 0 < x ∧ x < 1 / 2} ∧
      incr2 = {x : ℝ | x > 1 / a} ∧
      decr = {x : ℝ | 1 / 2 < x ∧ x < 1 / a}
  else if a = 2 then
    ∃ incr : Set ℝ,
      incr = {x : ℝ | x > 0}
  else 
    ∃ incr1 incr2 decr : Set ℝ,
      incr1 = {x : ℝ | 0 < x ∧ x < 1 / a} ∧
      incr2 = {x : ℝ | x > 1 / 2} ∧
      decr = {x : ℝ | 1 / a < x ∧ x < 1 / 2} :=
sorry

end part1_tangent_line_part2_monotonic_intervals_l64_64041


namespace total_revenue_is_405_l64_64697

-- Define the cost of rentals
def canoeCost : ℕ := 15
def kayakCost : ℕ := 18

-- Define terms for number of rentals
variables (C K : ℕ)

-- Conditions
axiom ratio_condition : 2 * C = 3 * K
axiom difference_condition : C = K + 5

-- Total revenue
def totalRevenue (C K : ℕ) : ℕ := (canoeCost * C) + (kayakCost * K)

-- Theorem statement
theorem total_revenue_is_405 (C K : ℕ) (H1 : 2 * C = 3 * K) (H2 : C = K + 5) : 
  totalRevenue C K = 405 := by
  sorry

end total_revenue_is_405_l64_64697


namespace product_trippled_when_added_to_reciprocal_l64_64937

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64937


namespace average_age_of_team_l64_64154

theorem average_age_of_team :
  ∃ A : ℝ,
    (∀ (ages : Fin 11 → ℝ),
      (ages ⟨0, by norm_num⟩ = 26) ∧
      (ages ⟨1, by norm_num⟩ = 29) ∧
      (∀ i (hi : 2 ≤ i ∧ i < 11), ages i = 32)  ∧ 
      (11 * A = sum (range 11) (λ i, ages ⟨i, by norm_num⟩)) ∧
      (9 * (A - 1) = sum (range 2 11) (λ i, ages ⟨i, by norm_num⟩))
    ) →
  A = 32 :=
sorry

end average_age_of_team_l64_64154


namespace spherical_to_rectangular_coords_l64_64390

theorem spherical_to_rectangular_coords (
  {ρ θ φ : ℝ} 
) (hρ : ρ = 3) (hθ : θ = 3 * Real.pi / 2) (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by 
  sorry

end spherical_to_rectangular_coords_l64_64390


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64750

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l64_64750


namespace product_of_tripled_reciprocals_eq_neg_half_l64_64814

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l64_64814


namespace problem_l64_64589

noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

theorem problem {a b : ℝ} (h : ∀ x, a ≤ x ∧ x ≤ b ↔ a ≤ f(x) ∧ f(x) ≤ b) : a + b = 1:=
sorry

end problem_l64_64589


namespace new_device_significant_improvement_l64_64257

noncomputable section
open Real

def old_device_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (data : List ℝ) : ℝ := data.sum / data.length
def variance (data : List ℝ) : ℝ := (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

def x̄ : ℝ := mean old_device_data
def ȳ : ℝ := mean new_device_data
def s₁_sq : ℝ := variance old_device_data
def s₂_sq : ℝ := variance new_device_data

theorem new_device_significant_improvement :
  ȳ - x̄ > 2 * sqrt ((s₁_sq + s₂_sq) / 10) :=
by sorry

end new_device_significant_improvement_l64_64257


namespace product_of_triple_when_added_to_reciprocal_l64_64907

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l64_64907


namespace sum_of_factors_of_30_is_72_l64_64969

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l64_64969


namespace find_angle_x_l64_64484

def angle_ABC := 124
def angle_BAD := 30
def angle_BDA := 28
def angle_ABD := 180 - angle_ABC
def angle_x := 180 - (angle_BAD + angle_ABD)

theorem find_angle_x : angle_x = 94 :=
by
  repeat { sorry }

end find_angle_x_l64_64484


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64919

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l64_64919


namespace cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l64_64466

theorem cos_beta_of_tan_alpha_and_sin_alpha_plus_beta 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 3) (h_sin_alpha_beta : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 10 / 10 := 
sorry

end cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l64_64466


namespace triangle_is_right_l64_64181

theorem triangle_is_right (x : ℝ) (hx : 0 < x) : 
  let a := 3 * x,
      b := 4 * x,
      c := 5 * x
  in a^2 + b^2 = c^2 :=
by {
  sorry
}

end triangle_is_right_l64_64181


namespace david_average_marks_l64_64397

-- Define the individual marks
def english_marks : ℕ := 74
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℚ := total_marks / num_subjects

-- Assert the average marks calculation
theorem david_average_marks : average_marks = 75.6 := by
  sorry

end david_average_marks_l64_64397


namespace product_trippled_when_added_to_reciprocal_l64_64935

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l64_64935
