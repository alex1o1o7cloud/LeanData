import Mathlib
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Divisibility.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Limit
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic.Proxy
import Mathlib.Topology.Instances.Real

namespace rods_in_one_mile_l573_573861

-- Definitions of the conditions
def mile_to_furlong := 10
def furlong_to_rod := 50

-- Theorem statement corresponding to the proof problem
theorem rods_in_one_mile : mile_to_furlong * furlong_to_rod = 500 := 
by sorry

end rods_in_one_mile_l573_573861


namespace washing_machine_capacity_l573_573224

theorem washing_machine_capacity 
  (shirts : ℕ) (sweaters : ℕ) (loads : ℕ) (total_clothing : ℕ) (n : ℕ)
  (h1 : shirts = 43) (h2 : sweaters = 2) (h3 : loads = 9)
  (h4 : total_clothing = shirts + sweaters)
  (h5 : total_clothing / loads = n) :
  n = 5 :=
sorry

end washing_machine_capacity_l573_573224


namespace trigonometric_identity_l573_573506

def r (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

def sin (y r : ℝ) : ℝ := y / r
def cos (x r : ℝ) : ℝ := x / r

theorem trigonometric_identity (x y : ℝ) (hx : x = 4) (hy : y = -3) :
  2 * sin y (r x y) + cos x (r x y) = -2 / 5 :=
by
  rw [hx, hy]
  -- Use previously defined sin, cos, and r
  sorry

end trigonometric_identity_l573_573506


namespace fraction_addition_l573_573642

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l573_573642


namespace problem_1_problem_2_l573_573174

-- Define the universal set U
def U : set ℤ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define sets A, B, and C
def A : set ℤ := {1, 2, 3}
def B : set ℤ := {-1, 0, 1}
def C : set ℤ := {-2, 0, 2}

-- The first proof statement
theorem problem_1 : A ∪ (B ∩ C) = {0, 1, 2, 3} :=
by
sorry

-- The second proof statement
theorem problem_2 : A ∩ (U \ (B ∪ C)) = {3} :=
by
sorry

end problem_1_problem_2_l573_573174


namespace count_positive_integers_satisfying_inequality_l573_573794

theorem count_positive_integers_satisfying_inequality :
  let S := {n : ℕ | n > 0 ∧ (n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0}
  ∃ n, S.card = 24 :=
by sorry

end count_positive_integers_satisfying_inequality_l573_573794


namespace function_passing_point_l573_573464

variable (f : ℝ → ℝ)

theorem function_passing_point (h : f 1 = 0) : f (0 + 1) + 1 = 1 := by
  calc f (0 + 1) + 1 = f 1 + 1 := by rfl
                  ... = 0 + 1 := by rw [h]
                  ... = 1 := by rfl

#check function_passing_point

end function_passing_point_l573_573464


namespace no_such_function_exists_l573_573217

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ m n : ℕ, (m + f n)^2 ≥ 3 * (f m)^2 + n^2 :=
by 
  sorry

end no_such_function_exists_l573_573217


namespace translate_point_B_l573_573507

theorem translate_point_B :
  ∀ (A B A' B' : ℤ × ℤ),
  A = (1, 2) →
  B = (7, 5) →
  A' = (-6, -3) →
  A' = (fst A - 7, snd A - 5) →
  B' = (fst B - 7, snd B - 5) →
  B' = (0, 0) := by
  intros A B A' B' hA hB hA' hTransA hTransB
  sorry

end translate_point_B_l573_573507


namespace smallest_natural_number_k_l573_573001

theorem smallest_natural_number_k :
  ∃ k : ℕ, k = 4 ∧ ∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ n → a^(k) * (1 - a)^(n) < 1 / (n + 1)^3 :=
by
  sorry

end smallest_natural_number_k_l573_573001


namespace real_solutions_eq_31_l573_573026

noncomputable def number_of_real_solutions : ℕ :=
  let zero := 0
  let fifty := 50
  let neg_fifty := -50
  let num_intervals := 8
  let num_solutions_per_interval := 2
  let total_solutions := num_intervals * num_solutions_per_interval * 2 - 1
  total_solutions

theorem real_solutions_eq_31 : number_of_real_solutions = 31 := by
  sorry

end real_solutions_eq_31_l573_573026


namespace cameron_worked_days_l573_573938

variable (x : ℝ)   -- The number of days Cameron worked alone.

-- Cameron's work rate.
def cameron_rate := (1 : ℝ) / 18

-- Combined work rate of Cameron and Sandra.
def combined_rate := (1 : ℝ) / 7

-- The remaining task completed by Cameron and Sandra in 3.5 days.
def remaining_work := 3.5 * combined_rate

-- Proof problem statement: finding the number of days Cameron worked alone.
theorem cameron_worked_days :
  (x * cameron_rate) + remaining_work = (1 : ℝ) →
  x = 9 := 
sorry

end cameron_worked_days_l573_573938


namespace relatively_prime_days_in_december_l573_573338
open Nat

def is_rel_prime_to_12 (d : ℕ) : Prop := gcd d 12 = 1

theorem relatively_prime_days_in_december : 
  ∑ d in (Finset.Icc 1 31), if is_rel_prime_to_12 d then 1 else 0 = 11 :=
by
  sorry

end relatively_prime_days_in_december_l573_573338


namespace probability_ball_2_l573_573670

theorem probability_ball_2 :
  let balls := {1, 2, 3, 4}
  let draw := {2}
  let total_events := 4
  let favorable_event := 1
  favourable_event / total_events = 1 / 4 := sorry

end probability_ball_2_l573_573670


namespace ab_ineq_ab_eq_ineq_l573_573112

theorem ab_ineq (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^2 + b^2 = 4) : 
  ab / (a + b + 2) ≤ real.sqrt 2 - 1 :=
sorry

theorem ab_eq_ineq (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^2 + b^2 = 4) :
  a = real.sqrt 2 → b = real.sqrt 2 → ab / (a + b + 2) = real.sqrt 2 - 1 :=
sorry

end ab_ineq_ab_eq_ineq_l573_573112


namespace count_positive_integers_satisfying_product_inequality_l573_573807

theorem count_positive_integers_satisfying_product_inequality :
  ∃ (k : ℕ), k = 23 ∧ 
  (n : ℕ ) → (2 ≤ n ∧ n < 100) →
  ((∃ (m: ℕ), 2 + 4 * m = n) ∧ ((n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0) = 23 :=
by
  sorry

end count_positive_integers_satisfying_product_inequality_l573_573807


namespace first_factor_lcm_of_two_numbers_l573_573256

open Nat

-- Define the given problem as a Lean 4 statement:
theorem first_factor_lcm_of_two_numbers (A B : ℕ) (hcf : ℕ) (X : ℕ) :
  hcf = 20 ∧ A = 300 ∧ B < A ∧ gcd A B = hcf ∧ A % (hcf * X * 15) = 0 → X = 1 :=
by
  intros h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2.1
  have h5 := h.2.2.2.2
  -- rest of the proof goes here
  sorry

end first_factor_lcm_of_two_numbers_l573_573256


namespace prob_exactly_four_blue_pens_l573_573493

-- Define the basic probabilities
def prob_blue : ℚ := 5 / 9
def prob_red : ℚ := 4 / 9

-- Define the probability of any specific sequence of 4 blue and 3 red pens
def prob_specific_sequence : ℚ := (prob_blue ^ 4) * (prob_red ^ 3)

-- Define the combinatorial number of ways to pick 4 blue pens out of 7
def num_ways : ℕ := Nat.choose 7 4

-- Total probability calculation
def total_prob : ℚ := num_ways * prob_specific_sequence

-- The final result should be approximately 0.294
theorem prob_exactly_four_blue_pens :
  total_prob ≈ 0.294 := sorry

end prob_exactly_four_blue_pens_l573_573493


namespace find_function_l573_573197

noncomputable def discrete_subset (C : Set ℕ) : Prop := ∀ n ∈ C, n + 1 ∉ C ∧ n - 1 ∉ C

theorem find_function (f : ℕ → ℕ) (C : Set ℕ) (hC : discrete_subset C) :
  (∀ n, n^3 - n^2 ≤ f(n) * f(f(f(n))) ∧ f(n) * f(f(f(n))) ≤ n^3 + n^2) →
  ∀ n, f(n) = if n ∈ C then n + 1 else if n - 1 ∈ C then n - 1 else n :=
sorry

end find_function_l573_573197


namespace intersection_A_B_l573_573855

variable A : Set Int
variable B : Set Int

def setA : Set Int := {-1, 1, 2, 4}
def setB : Set Int := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  let A := setA
  let B := setB
  sorry

end intersection_A_B_l573_573855


namespace sport_formulation_contains_15_ounces_of_water_l573_573932

/-- In the standard formulation, the ratio by volume of flavoring to corn syrup to water is 1 : 12 : 30.
    In the sport formulation, the ratio of flavoring to corn syrup is three times as great as in the standard formulation.
    The ratio of flavoring to water is half that of the standard formulation.
    If a large bottle of the sport formulation contains 1 ounce of corn syrup, it contains 15 ounces of water.
-/
theorem sport_formulation_contains_15_ounces_of_water :
  ∀ {f s w : ℕ},
    -- Standard formulation ratios
    f = 1 →
    s = 12 →
    w = 30 →
    -- Sport formulation conditions
    let F := f * 3 in
    let W := w * 2 in
    s = 1 →
    -- Given amount of corn syrup in sport formulation
    s = 1 →
    -- Required to prove that the bottle contains 15 ounces of water
    W / 4 = 15 :=
by
  intros f s w h_f h_s h_w
  let F := f * 3
  let W := w * 2
  assume hs : s = 1
  have h1 : 4 * (W / 4) = W * 1 / 4 * 4 := sorry
  dsimp [F, W] at h1
  exact_mod_cast h1

end sport_formulation_contains_15_ounces_of_water_l573_573932


namespace find_k_l573_573364

theorem find_k (angle_BAC : ℝ) (angle_D : ℝ)
  (h1 : 0 < angle_BAC ∧ angle_BAC < π)
  (h2 : 0 < angle_D ∧ angle_D < π)
  (h3 : (π - angle_BAC) / 2 = 3 * angle_D) :
  angle_BAC = (5 / 11) * π :=
by sorry

end find_k_l573_573364


namespace regular_pyramid_of_angles_and_base_l573_573242

theorem regular_pyramid_of_angles_and_base (n : ℕ) (V : Point) (A : Fin n → Point)
  (regular_base : regular_polygon (finset.univ.image A))
  (angles_equal : ∀ i : Fin n, ∠ (V, A i, A (i + 1 % n)) = (∠ V (A 0) (A 1))) :
  n = 3 → regular_pyramid V A :=
by
  sorry

end regular_pyramid_of_angles_and_base_l573_573242


namespace carolyn_practice_time_l573_573378

theorem carolyn_practice_time :
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  monthly_minutes_total = 1920 :=
by
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  sorry

end carolyn_practice_time_l573_573378


namespace smallest_possible_a_l573_573584

theorem smallest_possible_a {a b c : ℚ} (h_vertex : (1 / 3, -4 / 3)) 
  (h_eqn : ∀ x : ℚ, y = a * x^2 + b * x + c) (h_positive : a > 0)
  (h_integer : ∃ n : ℤ, 3 * a + 2 * b + c = n) : 
  a = 3 :=
by
  sorry

end smallest_possible_a_l573_573584


namespace total_cost_of_items_l573_573720

theorem total_cost_of_items
  (E P M : ℕ)
  (h1 : E + 3 * P + 2 * M = 240)
  (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_items_l573_573720


namespace part1_part2_part3_l573_573220

-- Part 1
theorem part1 :
  ∀ x : ℝ, (4 * x - 3 = 1) → (x = 1) ↔ 
    (¬(x - 3 > 3 * x - 1) ∧ (4 * (x - 1) ≤ 2) ∧ (x + 2 > 0 ∧ 3 * x - 3 ≤ 1)) :=
by sorry

-- Part 2
theorem part2 :
  ∀ (m n q : ℝ), (m + 2 * n = 6) → (2 * m + n = 3 * q) → (m + n > 1) → q > -1 :=
by sorry

-- Part 3
theorem part3 :
  ∀ (k m n : ℝ), (k < 3) → (∃ x : ℝ, (3 * (x - 1) = k) ∧ (4 * x + n < x + 2 * m)) → 
    (m + n ≥ 0) → (∃! n : ℝ, ∀ x : ℝ, (2 ≤ m ∧ m < 5 / 2)) :=
by sorry

end part1_part2_part3_l573_573220


namespace hannah_sold_40_cookies_l573_573108

theorem hannah_sold_40_cookies (C : ℕ) (h1 : 0.8 * C + 60 = 92) : C = 40 :=
by
  sorry

end hannah_sold_40_cookies_l573_573108


namespace even_before_odd_probability_l573_573367

/-- An unbiased 8-sided die rolls, the probability that all even numbers appear
at least once before any of the odd numbers -/
theorem even_before_odd_probability :
  ∃ P : ℝ, (P = ∑ n in (range (5 + ∞)), 
    (1 / 2 ^ n) * (1 - (choose_not_all_evens_probability (n - 1)))) :=
sorry

/-- A placeholder function for the calculation of the probability
   that not all four even numbers appeared in the first (n - 1) rolls -/
def choose_not_all_evens_probability : ℕ → ℝ := sorry

end even_before_odd_probability_l573_573367


namespace coins_in_pockets_l573_573140

theorem coins_in_pockets : (Nat.choose (5 + 3 - 1) (3 - 1)) = 21 := by
  sorry

end coins_in_pockets_l573_573140


namespace shifted_function_increasing_l573_573122

noncomputable def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x < f y

/- Given that f(x) is increasing on the interval (-2, 3), 
   prove that y = f(x + 4) is increasing on the interval (-6, -1) -/
theorem shifted_function_increasing 
  (f : ℝ → ℝ) 
  (h : is_increasing f {x | -2 < x ∧ x < 3}) : 
  is_increasing (λ x, f (x + 4)) {x | -6 < x ∧ x < -1} :=
sorry

end shifted_function_increasing_l573_573122


namespace imaginary_part_is_one_l573_573487

noncomputable def z (a : ℝ) : ℂ := (1 + complex.I) / (a - complex.I)

theorem imaginary_part_is_one (a : ℝ) (h : (z a).re = 0) : (z a).im = 1 := by
  sorry

end imaginary_part_is_one_l573_573487


namespace range_of_sin_plus_cos_l573_573452

theorem range_of_sin_plus_cos (x : ℝ) (h : 0 < x ∧ x ≤ π / 3) : 1 < sin x + cos x ∧ sin x + cos x ≤ sqrt 2 := 
sorry

end range_of_sin_plus_cos_l573_573452


namespace basketball_points_total_l573_573129

variable (Tobee_points Jay_points Sean_points Remy_points Alex_points : ℕ)

def conditions := 
  Tobee_points = 4 ∧
  Jay_points = 2 * Tobee_points + 6 ∧
  Sean_points = Jay_points / 2 ∧
  Remy_points = Tobee_points + Jay_points - 3 ∧
  Alex_points = Sean_points + Remy_points + 4

theorem basketball_points_total 
  (h : conditions Tobee_points Jay_points Sean_points Remy_points Alex_points) :
  Tobee_points + Jay_points + Sean_points + Remy_points + Alex_points = 66 :=
by sorry

end basketball_points_total_l573_573129


namespace complex_eq_sub_l573_573828

open Complex

theorem complex_eq_sub {a b : ℝ} (h : (a : ℂ) + 2 * I = I * ((b : ℂ) - I)) : a - b = -3 := by
  sorry

end complex_eq_sub_l573_573828


namespace solve_for_x_l573_573579

theorem solve_for_x (x : ℝ) : 5^(3 * x) = real.sqrt 125 → x = 1 / 2 :=
by
  sorry

end solve_for_x_l573_573579


namespace expand_product_l573_573009

theorem expand_product (x : ℝ) : (x + 2) * (x^2 + 3 * x + 4) = x^3 + 5 * x^2 + 10 * x + 8 := 
by
  sorry

end expand_product_l573_573009


namespace triangle_perimeter_comparison_l573_573061

theorem triangle_perimeter_comparison
  {A B C D E F: Type _}
  [Geometry A B C]
  (h₀: angle A B C = 30)
  (h₁: angle C B A = 30)
  (D_on_AB: on D (line A B))
  (E_on_BC: on E (line B C))
  (F_on_AC: on F (line A C))
  (BF_angle_60: angle B F D = 60)
  (BE_angle_60: angle B F E = 60)
  (perim_ABC: Real)
  (perim_DEF: Real) :
  perim_ABC ≤ 2 * perim_DEF :=
by
  sorry

end triangle_perimeter_comparison_l573_573061


namespace tripod_height_floor_sum_l573_573347

theorem tripod_height_floor_sum
  (h m n : ℝ) -- We use real numbers to account for the conditions and answer
  (h_condition :  h = 144 / Real.sqrt 262.2)
  (m_value : m = 144)
  (n_value : n = 262.2)
  : Real.floor (m + Real.sqrt n) = 160 := 
by 
  sorry

end tripod_height_floor_sum_l573_573347


namespace num_positive_integers_satisfying_l573_573800

theorem num_positive_integers_satisfying (n : ℕ) :
  (∑ k in (finset.range 25), (if (even (2 + 4 * k)) then 1 else 0) = 24) :=
sorry

end num_positive_integers_satisfying_l573_573800


namespace squares_triangles_product_l573_573246

theorem squares_triangles_product :
  let S := 7
  let T := 10
  S * T = 70 :=
by
  let S := 7
  let T := 10
  show (S * T = 70)
  sorry

end squares_triangles_product_l573_573246


namespace equivalence_of_conditions_l573_573057

theorem equivalence_of_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (sin_A sin_B sin_C : ℝ)
  (h₀ : 0 < a)
  (h₁ : 0 < b)
  (h₂ : 0 < c)
  (h₃ : 0 < sin_A)
  (h₄ : 0 < sin_B)
  (h₅ : 0 < sin_C)
  (h₆ : A + B + C = Real.pi) :
  (a / sin_A = b / sin_B ∧ a / sin_A = c / sin_C) ↔
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧
   b^2 = c^2 + a^2 - 2 * c * a * Real.cos B ∧
   c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) ↔
  (a = b * Real.cos C + c * Real.cos B ∧
   b = c * Real.cos A + a * Real.cos C ∧
   c = a * Real.cos B + b * Real.cos A) :=
sorry

end equivalence_of_conditions_l573_573057


namespace inequality_proof_l573_573440

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : x^12 - y^12 + 2 * x^6 * y^6 ≤ (Real.pi / 2) := 
by 
  sorry

end inequality_proof_l573_573440


namespace sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l573_573733

noncomputable def sin_110_degrees : ℝ := Real.sin (110 * Real.pi / 180)
noncomputable def tan_945_degrees_reduction : ℝ := Real.tan (945 * Real.pi / 180 - 5 * Real.pi)
noncomputable def cos_25pi_over_4_reduction : ℝ := Real.cos (25 * Real.pi / 4 - 6 * 2 * Real.pi)

theorem sin_110_correct : sin_110_degrees = Real.sin (110 * Real.pi / 180) :=
by
  sorry

theorem tan_945_correct : tan_945_degrees_reduction = 1 :=
by 
  sorry

theorem cos_25pi_over_4_correct : cos_25pi_over_4_reduction = Real.cos (Real.pi / 4) :=
by 
  sorry

end sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l573_573733


namespace maximum_c_magnitude_l573_573555

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem maximum_c_magnitude 
  (a b c : ℝ × ℝ)
  (ha : vector_magnitude a = 1)
  (hb : vector_magnitude b = 1)
  (hab : dot_product a b = 1 / 2)
  (habc : dot_product (a.1 - c.1, a.2 - c.2) (b.1 - c.1, b.2 - c.2) = 0) :
  vector_magnitude c ≤ (real.sqrt 3 + 1) / 2 :=
sorry -- proof to be filled in

end maximum_c_magnitude_l573_573555


namespace symmetric_circle_eq_of_given_circle_eq_l573_573250

theorem symmetric_circle_eq_of_given_circle_eq
  (x y : ℝ)
  (eq1 : (x - 1)^2 + (y - 2)^2 = 1)
  (line_eq : y = x) :
  (x - 2)^2 + (y - 1)^2 = 1 := by
  sorry

end symmetric_circle_eq_of_given_circle_eq_l573_573250


namespace necessary_and_sufficient_condition_l573_573076

theorem necessary_and_sufficient_condition (x : ℝ) : (x > 0) ↔ (1 / x > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l573_573076


namespace even_exists_in_each_row_l573_573658

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem even_exists_in_each_row (a : ℕ → ℤ → ℤ)
  (h_table : ∀ n k, a n k = if n = 1 then 1 else
    (if k < -(n:ℤ) ∨ k ≥ (n:ℤ) then 0 else 
      (a (n-1) (k-1) + a (n-1) k + a (n-1) (k+1))))
  (h_a_10 : ∀ n, a n 0 = 1) :
  ∀ n, n ≥ 3 → ∃ k, is_even (a n k) :=
begin
  intro n,
  induction n,
  { -- base case n = 0
    intro hn,
    exfalso,
    exact nat.not_lt_zero 3 hn, },
  { -- inductive step n = n + 1
    intro hn,
    cases (lt_or_ge 3 n),
    { -- case n ≥ 3
      exact sorry, }, -- you need to complete the proof here
    { --case n < 3
      replace hn : n + 1 ≥ 3 := nat.succ_le_succ h,
      cases hn,
      { exfalso, exact nat.lt_asymm h hn },
      { exfalso, exact nat.lt_asymm h (nat.lt_of_le_of_lt hn nat.lt_add_one) } 
    }
  }
end

end even_exists_in_each_row_l573_573658


namespace part1_l573_573664

theorem part1 (m : ℕ) (n : ℕ) (h1 : m = 6 * 10 ^ n + m / 25) : ∃ i : ℕ, m = 625 * 10 ^ (3 * i) := sorry

end part1_l573_573664


namespace sqrt_approximation_l573_573177

theorem sqrt_approximation (a h : ℝ) (ha_pos : 0 < a) (hh_pos : 0 < h) (h_small : h / (2 * a) < 0.1) :
  abs (sqrt (a^2 + h) - (a + h / (2 * a))) ≤ 0.025 :=
by {
  let a := 120,
  let h := 6,
  have ha_square_eq : a^2 = 14400 := by norm_num,
  let h_small := 6 / (2 * 120),
  sorry
}

end sqrt_approximation_l573_573177


namespace monotonic_intervals_range_a_nonnegative_no_extreme_value_l573_573468

-- Problem 1: Monotonic intervals when a = 1/2
theorem monotonic_intervals (f : ℝ → ℝ) (a : ℝ) (h0 : ∀ x : ℝ, f x = x * (Real.exp x - 1) - a * x^2) 
    (h1 : a = 1/2) :
  (∀ x : ℝ, (x < -1 → monotone_increasing (λ x, f x)) ∧ 
      (-1 < x ∧ x < 0 → monotone_decreasing (λ x, f x)) ∧ 
      (x > 0 → monotone_increasing (λ x, f x))) :=
sorry

-- Problem 2: Range of a for f(x) ≥ 0 when x ≥ 0
theorem range_a_nonnegative (f : ℝ → ℝ) (a : ℝ) (h0 : ∀ x : ℝ, f x = x * (Real.exp x - 1) - a * x^2) :
  (∀ x ≥ 0, f x ≥ 0) ↔ (a ∈ Iic 1) :=
sorry

-- Problem 3: Value of a for no extreme values
theorem no_extreme_value (f : ℝ → ℝ) (a : ℝ) (h0 : ∀ x : ℝ, f x = x * (Real.exp x - 1) - a * x^2) 
    (h1 : ∀ x : ℝ, ¬∃ c, is_local_extr f c) : 
  a = 1 :=
sorry

end monotonic_intervals_range_a_nonnegative_no_extreme_value_l573_573468


namespace sum_of_lengths_of_intervals_l573_573040

def floor (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℝ :=
  floor x * (2013 ^ (x - floor x) - 1)

noncomputable def intervalSum : ℝ :=
  ∑ (k : ℤ) in Finset.Icc 2 2014, Real.log (2013 : ℝ) ((k + 2) / k)

theorem sum_of_lengths_of_intervals :
  intervalSum = Real.log (2013 : ℝ) 1008 := 
  sorry

end sum_of_lengths_of_intervals_l573_573040


namespace gcf_of_36_and_54_l573_573296

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := 
by
  sorry

end gcf_of_36_and_54_l573_573296


namespace equal_values_on_plane_l573_573370

theorem equal_values_on_plane (f : ℤ × ℤ → ℕ)
    (h_avg : ∀ (i j : ℤ), f (i, j) = (f (i+1, j) + f (i-1, j) + f (i, j+1) + f (i, j-1)) / 4) :
  ∃ c : ℕ, ∀ (i j : ℤ), f (i, j) = c :=
by
  sorry

end equal_values_on_plane_l573_573370


namespace minimum_value_of_f_l573_573056

-- Define the conditions
variables {x y : ℝ}
axiom h1 : x ≥ 0
axiom h2 : y ≥ 0
axiom h3 : x + 2 * y = 1

-- Define the function to minimize
def f (x y : ℝ) : ℝ := (2 / x) + (3 / y)

-- Minimum value to prove
def min_value : ℝ := 8 + 4 * Real.sqrt 3

-- Statement of the theorem
theorem minimum_value_of_f : f x y ≥ min_value :=
by
  sorry

end minimum_value_of_f_l573_573056


namespace triangle_side_length_l573_573927

theorem triangle_side_length {ABC : Triangle} {D E : Point}
  (h_acute : ABC.isAcute) 
  (hD : D ∈ ABC.AC)
  (hE : E ∈ ABC.AB)
  (h_right_angles : ∠ADB = 90 ∧ ∠AEC = 90)
  (h_perimeter_AED : Triangle.perimeter (triangle_of_points A E D) = 9)
  (h_circumradius_AED : Triangle.circumradius (triangle_of_points A E D) = 9 / 5)
  (h_perimeter_ABC : Triangle.perimeter ABC = 15) :
  length BC = 24 / 5 := by
sorry

end triangle_side_length_l573_573927


namespace find_peter_depth_l573_573204

def depth_of_peters_pond (p : ℕ) : Prop :=
  let mark_depth := 19
  let relationship := 3 * p + 4
  relationship = mark_depth

theorem find_peter_depth : ∃ p : ℕ, depth_of_peters_pond p ∧ p = 5 :=
by {
  use 5,
  unfold depth_of_peters_pond,
  split,
  { refl },
  { refl }
}

end find_peter_depth_l573_573204


namespace avg_speed_target_l573_573322

-- Definitions from the problem
variables (D : ℝ) (S : ℝ)
def car_travelled_part1_distance := 0.4 * D
def car_travelled_part1_speed := 40

def car_travelled_remaining_distance := 0.6 * D

def total_distance := D
def avg_speed_whole_trip := 50

def time_part1 := car_travelled_part1_distance / car_travelled_part1_speed
def time_part2 := car_travelled_remaining_distance / S

def total_time := time_part1 + time_part2

def avg_speed_calculation := total_distance / total_time

theorem avg_speed_target (h : avg_speed_whole_trip = avg_speed_calculation) : S = 60 :=
by sorry

end avg_speed_target_l573_573322


namespace total_animal_eyes_l573_573925

-- Define the conditions given in the problem
def numberFrogs : Nat := 20
def numberCrocodiles : Nat := 10
def eyesEach : Nat := 2

-- Define the statement that we need to prove
theorem total_animal_eyes : (numberFrogs * eyesEach) + (numberCrocodiles * eyesEach) = 60 := by
  sorry

end total_animal_eyes_l573_573925


namespace max_fraction_value_l573_573013

theorem max_fraction_value :
  ∀ (x y : ℝ), (1/4 ≤ x ∧ x ≤ 3/5) ∧ (1/5 ≤ y ∧ y ≤ 1/2) → 
    xy / (x^2 + y^2) ≤ 2/5 :=
by
  sorry

end max_fraction_value_l573_573013


namespace cube_geometric_shapes_l573_573647

def cube_vertices := {A, B, C, D, A₁, B₁, C₁, D₁ : Point}

def is_rectangle (p1 p2 p3 p4 : Point) : Prop := 
-- Definition of a rectangle given four points
sorry

def is_isosceles_right_triangle (p1 p2 p3 : Point) : Prop := 
-- Definition of an isosceles right triangle given three points
sorry

def is_equilateral_triangle (p1 p2 p3 : Point) : Prop := 
-- Definition of an equilateral triangle given three points
sorry

def is_tetrahedron (p1 p2 p3 p4 : Point) (f1 f2 f3 f4 : Triangle) : Prop := 
-- Definition of a tetrahedron given four points and its faces
sorry

def three_faces_isosceles_one_equilateral (p1 p2 p3 p4 : Point) : Prop :=
let t1 := triangle p1 p2 p3
let t2 := triangle p1 p2 p4
let t3 := triangle p1 p3 p4
let t4 := triangle p2 p3 p4
in is_tetrahedron p1 p2 p3 p4 t1 t2 t3 t4 ∧ 
   (is_isosceles_right_triangle t1 ∧ is_isosceles_right_triangle t2 ∧ 
    is_isosceles_right_triangle t3 ∧ is_equilateral_triangle t4)

def all_faces_equilateral (p1 p2 p3 p4 : Point) : Prop :=
let t1 := triangle p1 p2 p3
let t2 := triangle p1 p2 p4
let t3 := triangle p1 p3 p4
let t4 := triangle p2 p3 p4
in is_tetrahedron p1 p2 p3 p4 t1 t2 t3 t4 ∧ 
   (is_equilateral_triangle t1 ∧ is_equilateral_triangle t2 ∧ 
    is_equilateral_triangle t3 ∧ is_equilateral_triangle t4)

def all_faces_right_triangle (p1 p2 p3 p4 : Point) : Prop :=
let t1 := triangle p1 p2 p3
let t2 := triangle p1 p2 p4
let t3 := triangle p1 p3 p4
let t4 := triangle p2 p3 p4
in is_tetrahedron p1 p2 p3 p4 t1 t2 t3 t4 ∧ 
   (is_isosceles_right_triangle t1 ∧ is_isosceles_right_triangle t2 ∧ 
    is_isosceles_right_triangle t3 ∧ is_isosceles_right_triangle t4)

theorem cube_geometric_shapes :
  ∃ (p1 p2 p3 p4 : cube_vertices),
    (is_rectangle p1 p2 p3 p4) ∧
    (three_faces_isosceles_one_equilateral p1 p2 p3 p4) ∧
    (all_faces_equilateral p1 p2 p3 p4) ∧
    (all_faces_right_triangle p1 p2 p3 p4) :=
sorry

end cube_geometric_shapes_l573_573647


namespace tape_needed_for_large_box_l573_573388

-- Definition of the problem conditions
def tape_per_large_box (L : ℕ) : Prop :=
  -- Each large box takes L feet of packing tape to seal
  -- Each medium box takes 2 feet of packing tape to seal
  -- Each small box takes 1 foot of packing tape to seal
  -- Each box also takes 1 foot of packing tape to stick the address label on
  -- Debbie packed two large boxes this afternoon
  -- Debbie packed eight medium boxes this afternoon
  -- Debbie packed five small boxes this afternoon
  -- Debbie used 44 feet of tape in total
  2 * L + 2 + 24 + 10 = 44

theorem tape_needed_for_large_box : ∃ L : ℕ, tape_per_large_box L ∧ L = 4 :=
by {
  -- Proof goes here
  sorry
}

end tape_needed_for_large_box_l573_573388


namespace simplify_expression_l573_573233

theorem simplify_expression (x : ℝ) : 3 * x + 4 - x + 8 = 2 * x + 12 :=
by
  sorry

end simplify_expression_l573_573233


namespace divisor_of_a_l573_573957

theorem divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 45) 
  (h3 : Nat.gcd c d = 75) (h4 : 80 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
  7 ∣ a :=
by
  sorry

end divisor_of_a_l573_573957


namespace find_a_of_exponential_passing_point_l573_573095

theorem find_a_of_exponential_passing_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_point : a^2 = 4) : a = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a_of_exponential_passing_point_l573_573095


namespace minimize_sum_of_squares_if_and_only_if_l573_573815

noncomputable def minimize_sum_of_squares (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) : Prop :=
  let ax_by_cz := a * x + b * y + c * z
  ax_by_cz = 2 * S ∧
  x/y = a/b ∧
  y/z = b/c ∧
  x/z = a/c

theorem minimize_sum_of_squares_if_and_only_if (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) :
  (∃ P : ℝ, minimize_sum_of_squares a b c S O x y z) ↔ (x/y = a/b ∧ y/z = b/c ∧ x/z = a/c) := sorry

end minimize_sum_of_squares_if_and_only_if_l573_573815


namespace solution_set_of_inequality_system_l573_573609

theorem solution_set_of_inequality_system (x : ℝ) : 
  (x + 5 < 4) ∧ (3 * x + 1 ≥ 2 * (2 * x - 1)) ↔ (x < -1) :=
  by
  sorry

end solution_set_of_inequality_system_l573_573609


namespace johnson_farm_acres_l573_573587

theorem johnson_farm_acres (x y : ℕ) 
  (hc_corrd_acre : x = 300) 
  (hw_wheat_acre : y = 200) 
  (cost_corn : ∀ x, 42 * x) 
  (cost_wheat : ∀ y, 30 * y) 
  (total_budget : 42 * x + 30 * y = 18600) : 
  x + y = 500 :=
by 
  -- given conditions
  have h_wheat_acre : y = 200 := hw_wheat_acre,
  have h_corn_cost  : 42 * x = 12600 := calc
     42 * x = 18600 - 30 * y : by rw [total_budget, hw_wheat_acre]
     ... = 12600 : by norm_num,
  have h_corrd_acre : x = 300 := by apply hc_corrd_acre,
  -- prove the statement
  calc 
     x + y = 300 + 200 : by rw [h_corrd_acre, h_wheat_acre]
     ... = 500 : by norm_num,
     triv    

end johnson_farm_acres_l573_573587


namespace find_p_l573_573058

-- Definitions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)
def point_on_parabola (x : ℝ) := ∃ y, parabola p x y
def distance_to_focus (p : ℝ) (x y : ℝ) : ℝ := distance x y (focus p).fst (focus p).snd
#check point_on_parabola
theorem find_p :
  ∃ (p : ℝ), (∃ (y : ℝ), parabola p 4 y) ∧ distance_to_focus p 4 (Real.sqrt (8 * p)) = 5 :=
begin
  use 2,
  split,
  { use Real.sqrt (8 * 2),
    unfold parabola,
    norm_num,
    ring, },
  { unfold distance_to_focus focus distance,
    norm_num,
    unfold Real.sqrt,
    ring,
    sorry }
end

end find_p_l573_573058


namespace count_positive_integers_satisfying_inequality_l573_573792

theorem count_positive_integers_satisfying_inequality :
  let S := {n : ℕ | n > 0 ∧ (n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0}
  ∃ n, S.card = 24 :=
by sorry

end count_positive_integers_satisfying_inequality_l573_573792


namespace parabola_intersection_l573_573091

theorem parabola_intersection
  (p : ℝ) (hp : 0 < p)
  (A : ℝ × ℝ) (hA1 : A.1 = 3 * p / 2) (hA2 : A.2 = real.sqrt(3) * (A.1 - p/2))
  (B : ℝ × ℝ) (hB1 : B.1 = p / 6) (hB2 : B.2 = real.sqrt(3) * (B.1 - p/2))
  (M : ℝ × ℝ) (hM1 : M.1 = -p / 2) (t : ℝ)
  (hBM : M.1 - B.1 = t * (A.1 - M.1)) :
  t = -1/3 :=
  sorry

end parabola_intersection_l573_573091


namespace no_arithmetic_progression_lcms_l573_573633

theorem no_arithmetic_progression_lcms (n : ℕ) (n_gt_100 : n > 100) :
  ¬ ∃ seq : Finset ℕ, (∀ a b ∈ seq, a ≠ b) ∧
  (∀ (x y : ℕ), x ∈ seq → y ∈ seq → x ≠ y → x.lcm y ∈ seq) ∧
  (∃ (d : ℕ), d > 0 ∧ ∃ m : ℕ, ∀ k : ℕ, k < (seq.card * (seq.card - 1)) / 2 → (m + k * d) ∈ seq) := 
by
  sorry

end no_arithmetic_progression_lcms_l573_573633


namespace circle_count_greater_than_pi_inv_comb_l573_573048

theorem circle_count_greater_than_pi_inv_comb (n : ℕ) (h : 1 ≤ n) :
  let points := 2 * n + 3
  in ∀ (P : Fin points → ℝ × ℝ),
  (∀ i j k : Fin points, i ≠ j → j ≠ k → i ≠ k → ¬ IsCollinear P i j k) →
  (∀ i j k l : Fin points, i ≠ j → j ≠ k → k ≠ l → l ≠ i → i ≠ k → j ≠ l → 
  ¬ IsCyclic P i j k l) →
  ∃ k : ℕ, k > (ℕ.choose (2 * n + 3) 2 / Real.pi) := 
by 
  sorry

-- Auxiliary definitions:
def IsCollinear {α : Type*} [CommRing α] (P : Fin 3 → α × α) : Prop :=
  let ⟨x1, y1⟩ := P 0
  let ⟨x2, y2⟩ := P 1
  let ⟨x3, y3⟩ := P 2
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

def IsCyclic {α : Type*} [CommRing α] [Field α] (P : Fin 4 → α × α) : Prop :=
  let ⟨x1, y1⟩ := P 0
  let ⟨x2, y2⟩ := P 1
  let ⟨x3, y3⟩ := P 2
  let ⟨x4, y4⟩ := P 3
  (x1-x2)*(x3-x4)*(y1-y3)*(y2-y4) - (x1-x3)*(x2-x4)*(y1-y2)*(y3-y4) ≠ 0

end circle_count_greater_than_pi_inv_comb_l573_573048


namespace number_of_colorings_l573_573705

open Finset

-- Definitions from problem conditions
def vertex_colors :=
  {c : Fin 12 → ℕ // ∀ {i j : Fin 12}, adjacent i j → c i ≠ c j} -- adjacent is user-defined

-- Main theorem
theorem number_of_colorings : 
  (∃ (c : Fin 12 → ℕ), 
    (∀ {i j : Fin 12}, adjacent i j → c i ≠ c j) -> 
    (coloring_count c = 384) :=
sorry

end number_of_colorings_l573_573705


namespace count_positive_integers_satisfying_inequality_l573_573795

theorem count_positive_integers_satisfying_inequality :
  let S := {n : ℕ | n > 0 ∧ (n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0}
  ∃ n, S.card = 24 :=
by sorry

end count_positive_integers_satisfying_inequality_l573_573795


namespace regular_ngon_vertex_sequence_l573_573744

-- A regular n-gon with n odd requires each vertex to occur in a defined sequence
theorem regular_ngon_vertex_sequence (n : ℕ) (h_n_odd : n % 2 = 1) :
  (∀ A : ℕ, A < n → ∃ k ≥ 3, k < n → A_k_perpendicular_bisector_sequence(n, k) = A) ↔
  ∃ m : ℕ, n = 3^m :=
begin
  sorry
end

-- Helper predicate defining A_k lies on the perpendicular bisector of A_{k-2}A_{k-1}
def A_k_perpendicular_bisector_sequence (n : ℕ) (k : ℕ) : ℕ :=
  if k < 3 then k else (A_k_perpendicular_bisector_sequence n (k - 1) + A_k_perpendicular_bisector_sequence n (k - 2)) / 2  -- This is an example; actual definition may vary

end regular_ngon_vertex_sequence_l573_573744


namespace find_Z_l573_573887

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, -2], ![3, -5]]
def b : Vector (Fin 2) ℝ := ![1, 1]
def Z : Vector (Fin 2) ℝ := ![-1, -2]

theorem find_Z (A_inv_satisfies : A⁻¹ ⬝ Z = b) : Z = ![-1, -2] :=
  sorry

end find_Z_l573_573887


namespace angle_BDE_eq_60_l573_573915

-- Definitions of angles in triangle ABC
def angle_A : ℝ := 60
def angle_C : ℝ := 60

-- Conditions for points D and E on triangle sides and segment equality
variable (D E : ℝ → ℝ)
variable (DE BE : ℝ)
variable (hD : D ∈ segment AB)
variable (hE : E ∈ segment BC)
variable (hDE_eq_BE : DE = BE)

-- Statement to prove
theorem angle_BDE_eq_60 (angle_A angle_C : ℝ) (hA: angle_A = 60) (hC: angle_C = 60) 
  (D E : ℝ → ℝ) (hD : D ∈ segment AB) (hE : E ∈ segment BC) (hDE_eq_BE : DE = BE) : 
  angle BDE = 60 := 
sorry

end angle_BDE_eq_60_l573_573915


namespace player_two_cannot_win_l573_573316

theorem player_two_cannot_win
  (grid : Matrix (Fin 3) (Fin 3) ℤ) -- 3x3 grid
  (card_numbers : Fin 9 → ℤ) -- 9 cards each with a number
  : ∀ (placement : Fin 9 → Fin 3 × Fin 3), Player1_sum placement ≥ Player2_sum placement
  :=
begin
  -- Definitions based on the problem statement
  let rows_sum := λ (assignment: Fin 9 → Fin 3 × Fin 3), 
    (grid ⟨0⟩ ∘ assignment + grid ⟨2⟩ ∘ assignment),
    
  let cols_sum := λ (assignment: Fin 9 → Fin 3 × Fin 3), 
    (grid ⟨0⟩ ∘ assignment + grid ⟨2⟩ ∘ assignment),

  let Player1_sum := rows_sum,
  let Player2_sum := cols_sum,

  -- Proof goes here
  sorry
end

end player_two_cannot_win_l573_573316


namespace number_of_three_digit_integers_l573_573094

def digits : Finset Nat := {1, 3, 5, 9}

def count_distinct_three_digit_integers (s : Finset Nat) : Nat :=
  (s.card) * (s.card - 1) * (s.card - 2)

theorem number_of_three_digit_integers :
  count_distinct_three_digit_integers digits = 24 :=
by
  -- Proof would go here
  sorry

end number_of_three_digit_integers_l573_573094


namespace coins_in_pockets_l573_573139

theorem coins_in_pockets : (Nat.choose (5 + 3 - 1) (3 - 1)) = 21 := by
  sorry

end coins_in_pockets_l573_573139


namespace find_eccentricity_l573_573474

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)
variable (asymp_cond : b / a = 1 / 2)

theorem find_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 / 2 :=
by
  let c := Real.sqrt ((a^2 + b^2) / 4)
  let e := c / a
  use e
  sorry

end find_eccentricity_l573_573474


namespace maximize_dot_product_l573_573516

-- Definitions from the problem conditions.
variables {A B C P Q : Type*}
variables (vAB vAC vBC vPQ vBP vCQ : ℝ) (a : ℝ)
variables {triangle_right : ∀ (A B C : Type*), vAB ⊥ vAC} -- Right triangle condition.

-- Lean theorem statement
theorem maximize_dot_product (h1 : triangle_right A B C) 
                             (BC_len : vBC = a) 
                             (PQ_len : vPQ = 2 * a) 
                             (PQ_midpoint : midpoint P Q = A) :
                             ∃ θ : ℝ, θ = 0 ∧ (vBP • vCQ) = 0 := 
sorry

end maximize_dot_product_l573_573516


namespace min_value_of_abs_Z_minus_1_l573_573114

noncomputable theory
open Complex

theorem min_value_of_abs_Z_minus_1 (Z : ℂ) (h : |Z - 1| = |Z + 1|) : ∃ w : ℝ, abs (Z - 1) = w ∧ ∀ z : ℂ, h_z : |z - 1| = |z + 1| → w ≤ abs (z - 1) :=
by
  use 1
  sorry

end min_value_of_abs_Z_minus_1_l573_573114


namespace animals_mistaking_l573_573499

theorem animals_mistaking (C D : ℕ) (hd : D = C + 180) : 0.2 * D = 240 :=
by
  -- condition that D = C + 180 is given in hd
  sorry

end animals_mistaking_l573_573499


namespace amc_inequality_l573_573831

theorem amc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (a / (b + c^2) + b / (c + a^2) + c / (a + b^2)) ≥ (9 / 4) :=
by
  sorry

end amc_inequality_l573_573831


namespace simplify_fraction_48_72_l573_573994

theorem simplify_fraction_48_72 : (48 : ℚ) / 72 = 2 / 3 := sorry

end simplify_fraction_48_72_l573_573994


namespace element_in_set_l573_573554

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def complement_U_M : Set ℕ := {1, 2}

-- The main statement to prove
theorem element_in_set (M : Set ℕ) (h1 : U = {1, 2, 3, 4, 5}) (h2 : U \ M = complement_U_M) : 3 ∈ M := 
sorry

end element_in_set_l573_573554


namespace sins_prayers_l573_573328

structure Sins :=
  (pride : Nat)
  (slander : Nat)
  (laziness : Nat)
  (adultery : Nat)
  (gluttony : Nat)
  (self_love : Nat)
  (jealousy : Nat)
  (malicious_gossip : Nat)

def prayer_requirements (s : Sins) : Nat × Nat × Nat :=
  ( s.pride + 2 * s.laziness + 10 * s.adultery + s.gluttony,
    2 * s.pride + 2 * s.slander + 10 * s.adultery + 3 * s.self_love + 3 * s.jealousy + 7 * s.malicious_gossip,
    7 * s.slander + 10 * s.adultery + s.self_love + 2 * s.malicious_gossip )

theorem sins_prayers (sins : Sins) :
  sins.pride = 0 ∧
  sins.slander = 1 ∧
  sins.laziness = 0 ∧
  sins.adultery = 0 ∧
  sins.gluttony = 9 ∧
  sins.self_love = 1 ∧
  sins.jealousy = 0 ∧
  sins.malicious_gossip = 2 ∧
  (sins.pride + sins.slander + sins.laziness + sins.adultery + sins.gluttony + sins.self_love + sins.jealousy + sins.malicious_gossip = 12) ∧
  prayer_requirements sins = (9, 12, 10) :=
  by
  sorry

end sins_prayers_l573_573328


namespace always_meaningful_fraction_l573_573037

theorem always_meaningful_fraction {x : ℝ} : (∀ x, ∃ option : ℕ, 
  (option = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) ∨ 
  (option = 2 ∧ True) ∨ 
  (option = 3 ∧ x ≠ 0) ∨ 
  (option = 4 ∧ x ≠ 1)) → option = 2 :=
sorry

end always_meaningful_fraction_l573_573037


namespace min_value_ineq_l573_573454

noncomputable theory
open Real

theorem min_value_ineq (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) 
  (h5 : a + 20 * b = 2) (h6 : c + 20 * d = 2) :
  ∃ L ≈ 220.5, ∀ (a b c d : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a + 20 * b = 2 ∧ c + 20 * d = 2 → 
    L = (1 / a + 1 / (b * c * d)) :=
begin
  sorry,
end

end min_value_ineq_l573_573454


namespace sum_exterior_angles_geq_360_l573_573216

theorem sum_exterior_angles_geq_360 (n : ℕ) (θ : Fin n → ℝ) (h : ∀ i, θ i < 180) :
  (∑ i, 180 - θ i) ≥ 360 :=
by
  sorry

end sum_exterior_angles_geq_360_l573_573216


namespace symmetric_point_l573_573244

theorem symmetric_point (x y : ℝ) (hx : x = -2) (hy : y = 3) (a b : ℝ) (hne : y = x + 1)
  (halfway : (a = (x + (-2)) / 2) ∧ (b = (y + 3) / 2) ∧ (2 * b = 2 * a + 2) ∧ (2 * b = 1)):
  (a, b) = (0, 1) :=
by
  sorry

end symmetric_point_l573_573244


namespace find_f_inv_of_4_l573_573465

-- Define the function and its inverse
def f (x : ℝ) : ℝ := real.sqrt x
def f_inv (x : ℝ) : ℝ := x^2

-- The main statement to be proved
theorem find_f_inv_of_4 : f_inv 4 = 16 :=
by
  sorry

end find_f_inv_of_4_l573_573465


namespace probability_no_two_adjacent_heads_l573_573821

noncomputable def probability_no_adjacent_heads : ℚ :=
  let all_positions := {seq : Fin 4 → Bool // seq 0 = false || seq 1 = false || seq 2 = false || seq 3 = false} -- Allowed sequence constraints
  let valid_positions := {seq : all_positions // 
    seq.val 0 = false || seq.val 3 = false ∧
    seq.val 0 = false || seq.val 1 = false ∧
    seq.val 1 = false || seq.val 2 = false ∧
    seq.val 2 = false || seq.val 3 = false} -- Ensure no two adjacent heads
  (valid_positions.size : ℚ) / (all_positions.size : ℚ)

theorem probability_no_two_adjacent_heads :
  probability_no_adjacent_heads = 7 / 16 := sorry

end probability_no_two_adjacent_heads_l573_573821


namespace functional_solution_l573_573392

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

theorem functional_solution (f : ℝ → ℝ) :
  functional_equation f → (∀ x, f x = 0) ∨ (∀ x, f x = x^2) :=
begin
  intros h,
  sorry, -- Placeholder for the proof
end

end functional_solution_l573_573392


namespace percentage_honda_red_l573_573919

theorem percentage_honda_red (total_cars : ℕ) (honda_cars : ℕ) (percentage_red_total : ℚ)
  (percentage_red_non_honda : ℚ) (percentage_red_honda : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  percentage_red_total = 0.60 →
  percentage_red_non_honda = 0.225 →
  percentage_red_honda = 0.90 →
  ((honda_cars * percentage_red_honda) / total_cars) * 100 = ((total_cars * percentage_red_total - (total_cars - honda_cars) * percentage_red_non_honda) / honda_cars) * 100 :=
by
  sorry

end percentage_honda_red_l573_573919


namespace xiaoming_problem_l573_573601

theorem xiaoming_problem :
  (- 1 / 24) / (1 / 3 - 1 / 6 + 3 / 8) = - 1 / 13 :=
by
  sorry

end xiaoming_problem_l573_573601


namespace range_of_f_l573_573882

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x - 1

-- Define the domain
def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

-- Define the range
def range : Set ℕ := {2, 5, 8, 11}

-- Lean 4 theorem statement
theorem range_of_f : 
  {y | ∃ x ∈ domain, y = f x} = range :=
by
  sorry

end range_of_f_l573_573882


namespace negation_of_exists_proposition_l573_573041

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end negation_of_exists_proposition_l573_573041


namespace max_value_of_vectors_l573_573106

variables {R : Type*} [normed_field R] [normed_space ℝ R]

def m : R := sorry
def n : R := sorry

axiom m_nonzero : m ≠ 0
axiom n_nonzero : n ≠ 0
axiom length_m : ∥m∥ = 2
axiom length_m_plus_2n : ∥m + 2•n∥ = 2

/-- Find the maximum value of ∥n∥ + ∥2•m + n∥ given the conditions on ∥m∥ and ∥m + 2•n∥. -/
theorem max_value_of_vectors :
  ∃ x : ℝ, x = ∥n∥ + ∥2•m + n∥ ∧
  (∀ y : ℝ, y = ∥n∥ + ∥2•m + n∥ → y ≤ x) ∧ -- Check it's a maximum
  x = (8 * real.sqrt 3) / 3 :=
begin
  sorry
end

end max_value_of_vectors_l573_573106


namespace tower_height_count_l573_573563

theorem tower_height_count (bricks : ℕ) (height1 height2 height3 : ℕ) :
  height1 = 3 → height2 = 11 → height3 = 18 → bricks = 100 →
  (∃ (h : ℕ),  h = 1404) :=
by
  sorry

end tower_height_count_l573_573563


namespace invariant_ratio_l573_573625

variables {α : Type*} [MetricSpace α]

def circle (A B : α) := { C : α | ∃ k : ℝ, Distance C A = k ∧ Distance C B = k }

variables (k1 k2 k3 : set α) (A B C D E : α)
  (secant : Π (A : α), { D : α | C ∈ k1 ∩ k2 ∩ k3}) 

theorem invariant_ratio (h1 : ∀ {x}, x ∈ k1 ↔ Distance x A = Distance x B)
                        (h2 : ∀ {x}, x ∈ k2 ↔ Distance x A = Distance x B)
                        (h3 : ∀ {x}, x ∈ k3 ↔ Distance x A = Distance x B)
                        (hC : C ∈ k1)
                        (hD : D ∈ k2)
                        (hE : E ∈ k3)
                        (hsecant : secant A = {D, C, E}) :
  let DA := Distance A D,
      DC := Distance D C,
      DE := Distance D E in
  DC / DE = DA := sorry

end invariant_ratio_l573_573625


namespace tangent_proof_l573_573228

-- Define the triangle ABC
variable (A B C K L P : Point)

-- Define the conditions
-- K is the point where the incircle touches AC
def in_tangency_condition (A C K : Point) : Prop := 
  touches_incircle (triangle.mk A B C) K AC

-- L is the point where the excircle touches AC
def ex_tangency_condition (A C L : Point) : Prop := 
  touches_excircle (triangle.mk A B C) L AC

-- P is the projection of the incenter onto the perpendicular bisector of AC
def projection_condition (I A C P : Point) : Prop := 
  projected_on_bisector (incenter (triangle.mk A B C)) A C P

-- Tangents at points K and L to the circumcircle of triangle BKL intersect on the circumcircle of triangle ABC
def tangents_intersect_condition (B K L : Point) : Prop := 
  intersects_on_circumcircle (circumcircle (triangle.mk B K L)) K L (circumcircle (triangle.mk A B C))

-- Prove that lines AB and BC are tangent to the circle PKL
theorem tangent_proof
  (h_in_tangent : in_tangency_condition A C K)
  (h_ex_tangent : ex_tangency_condition A C L)
  (h_projection : projection_condition (incenter (triangle.mk A B C)) A C P)
  (h_tangents_intersect : tangents_intersect_condition B K L)
  : are_tangent (line_through A B) (circle_of_points P K L) ∧ are_tangent (line_through B C) (circle_of_points P K L) := 
sorry

end tangent_proof_l573_573228


namespace simplify_fraction_l573_573993

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l573_573993


namespace football_cost_is_correct_l573_573354

def total_spent_on_toys : ℝ := 12.30
def spent_on_marbles : ℝ := 6.59
def spent_on_football := total_spent_on_toys - spent_on_marbles

theorem football_cost_is_correct : spent_on_football = 5.71 :=
by
  sorry

end football_cost_is_correct_l573_573354


namespace hoseok_jump_diff_l573_573896

theorem hoseok_jump_diff :
  let m : List ℕ := [88, 75, 62, 91, 80]
  in (m.maximum.getD 0 - m.minimum.getD 0 = 29) :=
by 
  -- Define the list of jumps
  have m_def : List ℕ := [88, 75, 62, 91, 80]
  -- Get the maximum and minimum values
  have max_val := m_def.maximum.getD 0
  have min_val := m_def.minimum.getD 0
  -- Assertion of the maximum and minimum difference as 29
  calc
    max_val - min_val = 91 - 62 : by sorry
                  ... = 29      : by rfl

end hoseok_jump_diff_l573_573896


namespace geometric_sequence_problem_l573_573876

open Real

noncomputable def a (n : ℕ) : ℝ := 1 / (2^(2*n - 1))
noncomputable def b (n : ℕ) : ℝ := log 2 (a n)
noncomputable def T (n : ℕ) : ℝ := (1 / 2) * (1 - (1 / (2*n + 1)))

theorem geometric_sequence_problem 
  (∀ n : ℕ, a n > 0)
  (a1_plus_4a2_one : a 1 + 4 * a 2 = 1)
  (a3_squared_eq_16a2a6 : (a 3)^2 = 16 * a 2 * a 6)
  (bn_def : ∀ n, b n = log 2 (a n))
  (t_sum_def : T n = 1/2 * (1 - 1/(2*n + 1))) : 
  (a n = 1 / (2^(2*n - 1))) ∧ 
  (∑ i in Finset.range n, 
  (1 / (b i * b (i + 1))) = T n) := 
sorry

end geometric_sequence_problem_l573_573876


namespace masha_mushrooms_l573_573044

theorem masha_mushrooms (B1 B2 B3 B4 G1 G2 G3 : ℕ) (total : B1 + B2 + B3 + B4 + G1 + G2 + G3 = 70)
  (girls_distinct : G1 ≠ G2 ∧ G1 ≠ G3 ∧ G2 ≠ G3)
  (boys_threshold : ∀ {A B C D : ℕ}, (A = B1 ∨ A = B2 ∨ A = B3 ∨ A = B4) →
                    (B = B1 ∨ B = B2 ∨ B = B3 ∨ B = B4) →
                    (C = B1 ∨ C = B2 ∨ C = B3 ∨ C = B4) → 
                    (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
                    A + B + C ≥ 43)
  (diff_no_more_than_five_times : ∀ {x y : ℕ}, (x = B1 ∨ x = B2 ∨ x = B3 ∨ x = B4 ∨ x = G1 ∨ x = G2 ∨ x = G3) →
                                  (y = B1 ∨ y = B2 ∨ y = B3 ∨ y = B4 ∨ y = G1 ∨ y = G2 ∨ y = G3) →
                                  x ≠ y → x ≤ 5 * y ∧ y ≤ 5 * x)
  (masha_max_girl : G3 = max G1 (max G2 G3))
  : G3 = 5 :=
sorry

end masha_mushrooms_l573_573044


namespace find_t_value_l573_573826

theorem find_t_value (t : ℝ) :
  let m := (1, 3)
      n := (2, t)
      add_vec := (1 + 2, 3 + t)
      sub_vec := (1 - 2, 3 - t)
  in (add_vec.1 * sub_vec.1 + add_vec.2 * sub_vec.2 = 0) → t = real.sqrt 6 ∨ t = -real.sqrt 6 :=
begin
  sorry
end

end find_t_value_l573_573826


namespace sum_of_squared_distances_range_l573_573475

theorem sum_of_squared_distances_range
  (φ : ℝ)
  (x : ℝ := 2 * Real.cos φ)
  (y : ℝ := 3 * Real.sin φ)
  (A : ℝ × ℝ := (1, Real.sqrt 3))
  (B : ℝ × ℝ := (-Real.sqrt 3, 1))
  (C : ℝ × ℝ := (-1, -Real.sqrt 3))
  (D : ℝ × ℝ := (Real.sqrt 3, -1))
  (PA := (x - A.1)^2 + (y - A.2)^2)
  (PB := (x - B.1)^2 + (y - B.2)^2)
  (PC := (x - C.1)^2 + (y - C.2)^2)
  (PD := (x - D.1)^2 + (y - D.2)^2) :
  32 ≤ PA + PB + PC + PD ∧ PA + PB + PC + PD ≤ 52 :=
  by sorry

end sum_of_squared_distances_range_l573_573475


namespace smallest_value_is_14_l573_573818

noncomputable def smallest_possible_value : ℕ :=
  Inf {d : ℕ | ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ |2011^m - 45^n| = d}

theorem smallest_value_is_14 : smallest_possible_value = 14 :=
by
  sorry

end smallest_value_is_14_l573_573818


namespace prob_at_least_seven_friends_stay_for_entire_game_l573_573762

-- Definitions of conditions
def numFriends : ℕ := 8
def numUnsureFriends : ℕ := 5
def probabilityStay (p : ℚ) : ℚ := p
def sureFriends := 3

-- The probabilities
def prob_one_third : ℚ := 1 / 3
def prob_two_thirds : ℚ := 2 / 3

-- Variables to hold binomial coefficient and power calculation
noncomputable def C (n k : ℕ) : ℚ := (Nat.choose n k)
noncomputable def probability_at_least_seven_friends_stay : ℚ :=
  C numUnsureFriends 4 * (probabilityStay prob_one_third)^4 * (probabilityStay prob_two_thirds)^1 +
  (probabilityStay prob_one_third)^5

-- Theorem statement
theorem prob_at_least_seven_friends_stay_for_entire_game :
  probability_at_least_seven_friends_stay = 11 / 243 :=
  by sorry

end prob_at_least_seven_friends_stay_for_entire_game_l573_573762


namespace valid_outfit_combinations_l573_573482

/-- Number of valid outfit combinations given constraints -/
theorem valid_outfit_combinations :
  let shirts := 9
      pants := 5
      hats := 7
      common_colors := 5 -- tan, black, blue, gray, red
  in
  let total_combinations := shirts * pants * hats
      forbidden_combinations := common_colors
  in
  total_combinations - forbidden_combinations = 310 :=
by
  sorry

end valid_outfit_combinations_l573_573482


namespace find_x_squared_plus_one_over_x_squared_l573_573829

theorem find_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end find_x_squared_plus_one_over_x_squared_l573_573829


namespace max_n_value_l573_573873

theorem max_n_value (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h₁ : ∀ n, S n = 2 * a n - a 1)
  (h₂ : ∃ k, a 3 = k ∧ a 2 + 1 = k - (a 2 - (a 1 + 1))) :
  ∀ n, (log 2 (a (n + 1)) ≤ 71) → n ≤ 70 := 
sorry

end max_n_value_l573_573873


namespace calculate_total_difference_in_miles_l573_573274

def miles_bus_a : ℝ := 1.25
def miles_walk_1 : ℝ := 0.35
def miles_bus_b : ℝ := 2.68
def miles_walk_2 : ℝ := 0.47
def miles_bus_c : ℝ := 3.27
def miles_walk_3 : ℝ := 0.21

def total_miles_on_buses : ℝ := miles_bus_a + miles_bus_b + miles_bus_c
def total_miles_walked : ℝ := miles_walk_1 + miles_walk_2 + miles_walk_3
def total_difference_in_miles : ℝ := total_miles_on_buses - total_miles_walked

theorem calculate_total_difference_in_miles :
  total_difference_in_miles = 6.17 := by
  sorry

end calculate_total_difference_in_miles_l573_573274


namespace equation_of_circle_C_equation_of_line_l_l573_573458

-- Condition: The center of the circle lies on the line y = x + 1.
def center_on_line (a b : ℝ) : Prop :=
  b = a + 1

-- Condition: The circle is tangent to the x-axis.
def tangent_to_x_axis (a b r : ℝ) : Prop :=
  r = b

-- Condition: Point P(-5, -2) lies on the circle.
def point_on_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Condition: Point Q(-4, -5) lies outside the circle.
def point_outside_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 > r^2

-- Proof (1): Find the equation of the circle.
theorem equation_of_circle_C :
  ∃ (a b r : ℝ), center_on_line a b ∧ tangent_to_x_axis a b r ∧ point_on_circle a b r (-5) (-2) ∧ point_outside_circle a b r (-4) (-5) ∧ (∀ x y, (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 3)^2 + (y + 2)^2 = 4) :=
sorry

-- Proof (2): Find the equation of the line l.
theorem equation_of_line_l (a b r : ℝ) (ha : center_on_line a b) (hb : tangent_to_x_axis a b r) (hc : point_on_circle a b r (-5) (-2)) (hd : point_outside_circle a b r (-4) (-5)) :
  ∃ (k : ℝ), ∀ x y, ((k = 0 ∧ x = -2) ∨ (k ≠ 0 ∧ y + 4 = -3/4 * (x + 2))) ↔ ((x = -2) ∨ (3 * x + 4 * y + 22 = 0)) :=
sorry

end equation_of_circle_C_equation_of_line_l_l573_573458


namespace number_of_ordered_pairs_l573_573017

theorem number_of_ordered_pairs
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 50)
  (h2 : 0 ≤ b)
  (h3 : ∃ r s : ℕ, r + s = a ∧ r * s = b) :
  ∑ a in (finset.range 51).filter (λ x, 1 ≤ x), ((a + 1) / 2).ceil = 377 :=
by {
  sorry
}

end number_of_ordered_pairs_l573_573017


namespace average_age_of_school_l573_573135

theorem average_age_of_school 
  (total_students : ℕ)
  (average_age_boys : ℕ)
  (average_age_girls : ℕ)
  (number_of_girls : ℕ)
  (number_of_boys : ℕ := total_students - number_of_girls)
  (total_age_boys : ℕ := average_age_boys * number_of_boys)
  (total_age_girls : ℕ := average_age_girls * number_of_girls)
  (total_age_students : ℕ := total_age_boys + total_age_girls) :
  total_students = 640 →
  average_age_boys = 12 →
  average_age_girls = 11 →
  number_of_girls = 160 →
  (total_age_students : ℝ) / (total_students : ℝ) = 11.75 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_of_school_l573_573135


namespace scalene_triangle_process_l573_573339

theorem scalene_triangle_process (a b c : ℝ) 
  (h1: a > 0) (h2: b > 0) (h3: c > 0) 
  (h4: a + b > c) (h5: b + c > a) (h6: a + c > b) : 
  ¬(∃ k : ℝ, (k > 0) ∧ 
    ((k * a = a + b - c) ∧ 
     (k * b = b + c - a) ∧ 
     (k * c = a + c - b))) ∧ 
  (∀ n: ℕ, n > 0 → (a + b - c)^n + (b + c - a)^n + (a + c - b)^n < 1) :=
by
  sorry

end scalene_triangle_process_l573_573339


namespace find_m_l573_573073

namespace ProofProblem

variable (x m : ℝ)
def p : Prop := x^2 + x - 2 > 0
def q : Prop := x > m

theorem find_m (h : ¬q → ¬p) : m ≥ 1 := 
sorry

end ProofProblem

end find_m_l573_573073


namespace adjacent_product_le_one_ninth_l573_573665

theorem adjacent_product_le_one_ninth
  (a b c d e : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e)
  (h_sum : a + b + c + d + e = 1) :
  ∃ (seq : list ℝ) (h_perm : seq.perm [a, b, c, d, e]),
  ∀ i, (seq.nth i) * (seq.nth ((i + 1) % 5)) ≤ 1 / 9 := 
sorry

end adjacent_product_le_one_ninth_l573_573665


namespace red_blood_cells_surface_area_l573_573266

-- Define the body surface area of an adult
def body_surface_area : ℝ := 1800

-- Define the multiplying factor for the surface areas of red blood cells
def multiplier : ℝ := 2000

-- Define the sum of the surface areas of all red blood cells
def sum_surface_area : ℝ := multiplier * body_surface_area

-- Define the expected sum in scientific notation
def expected_sum : ℝ := 3.6 * 10^6

-- The theorem that needs to be proved
theorem red_blood_cells_surface_area :
  sum_surface_area = expected_sum :=
by
  sorry

end red_blood_cells_surface_area_l573_573266


namespace sequence_a_is_n_l573_573060

variable (a : ℕ+ → ℝ) (S : ℕ+ → ℝ)
variable h_positive : ∀ n : ℕ+, 0 < a n
variable h_sum : ∀ n : ℕ+, S n = ∑ i in Finset.range n, a ⟨i+1, Nat.succ_pos i⟩
variable h_condition : ∀ n : ℕ+, 2 * S n = (a n) ^ 2 + a n

theorem sequence_a_is_n :
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ (∀ n : ℕ+, a n = n) :=
by sorry

end sequence_a_is_n_l573_573060


namespace range_of_x_l573_573416

variable {p : ℝ} {x : ℝ}

theorem range_of_x (h : 0 ≤ p ∧ p ≤ 4) : x^2 + p * x > 4 * x + p - 3 ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

end range_of_x_l573_573416


namespace sum_of_integers_with_product_5_pow_4_l573_573604

theorem sum_of_integers_with_product_5_pow_4 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 5^4 ∧
  a + b + c + d = 156 :=
by sorry

end sum_of_integers_with_product_5_pow_4_l573_573604


namespace total_number_of_birds_l573_573662

def bird_cages : Nat := 9
def parrots_per_cage : Nat := 2
def parakeets_per_cage : Nat := 6
def birds_per_cage : Nat := parrots_per_cage + parakeets_per_cage
def total_birds : Nat := bird_cages * birds_per_cage

theorem total_number_of_birds : total_birds = 72 := by
  sorry

end total_number_of_birds_l573_573662


namespace rotation_90_deg_l573_573317

theorem rotation_90_deg (z : ℂ) (r : ℂ → ℂ) (h : ∀ (x y : ℝ), r (x + y*I) = -y + x*I) :
  r (8 - 5*I) = 5 + 8*I :=
by sorry

end rotation_90_deg_l573_573317


namespace area_of_region_l573_573239

noncomputable def calcArea : ℝ :=
  ∑ n in (Finset.range 10).map Finset.natAbs ∘ (1 + ·), (4 * n - 1) * Real.pi

theorem area_of_region : calcArea = 210 * Real.pi :=
by
  sorry

end area_of_region_l573_573239


namespace problem_1_problem_2_l573_573199

-- Definitions of sets A, B, and C
def U := set ℝ
def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : set ℝ := {x | 0 < x ∧ x < 4}
def C (a : ℝ) : set ℝ := {x | x < a}

-- Lean 4 proof statement for (1)
theorem problem_1 : (A ∩ B = {x | 0 < x ∧ x ≤ 3}) ∧ (A ∪ B = {x | -1 ≤ x ∧ x < 4}) :=
by
  sorry

-- Lean 4 proof statement for (2)
theorem problem_2 (a : ℝ) (h : B ⊆ C a) : a ≥ 4 :=
by
  sorry

end problem_1_problem_2_l573_573199


namespace soccer_team_selection_l573_573708

def number_of_ways_to_choose_team (n m k : ℕ) (total_players quadruplets_players : list string) : ℕ :=
  if quadruplets_players ⊆ total_players then
    let total_combinations := nat.choose n m in
    let restricted_combinations := nat.choose (n - k) (m - k) in
    total_combinations - restricted_combinations
  else 0

theorem soccer_team_selection : number_of_ways_to_choose_team 16 11 4 ["Alex", "Ben", "Chris", "Dan"] = 3576 :=
by
  sorry

end soccer_team_selection_l573_573708


namespace magnitude_of_b_l573_573088

noncomputable def a : ℝ := sqrt 2
noncomputable def θ : ℝ := Real.pi / 4
noncomputable def a_dot_b : ℝ := 4

theorem magnitude_of_b (a b : EuclideanSpace ℝ (Fin 2)) 
    (h1 : ∥a∥ = sqrt 2) 
    (h2 : angle a b = (Real.pi / 4)) 
    (h3 : dotProduct a b = 4) : ∥b∥ = 4 :=
sorry

end magnitude_of_b_l573_573088


namespace output_value_of_y_l573_573402

/-- Define the initial conditions -/
def l : ℕ := 2
def m : ℕ := 3
def n : ℕ := 5

/-- Define the function that executes the flowchart operations -/
noncomputable def flowchart_operation (l m n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem output_value_of_y : flowchart_operation l m n = 68 := sorry

end output_value_of_y_l573_573402


namespace smallest_seating_N_l573_573326

theorem smallest_seating_N (total_chairs : ℕ) (reserved_chairs : ℕ) (occupied_distance : ℕ) : ∃ N : ℕ, 
  total_chairs = 72 ∧ reserved_chairs = 12 ∧ occupied_distance = 1 ∧ 
  (∀ n : ℕ, n < N → (total_chairs - reserved_chairs) / n > 1 ∨ n = 15) :=
begin
  use 15,
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { exact rfl },
  { intros n hn,
    rw [←nat.dvd_iff_mod_eq_zero] at hn,
    sorry
  }
end

end smallest_seating_N_l573_573326


namespace greatest_50_podpyirayushchee_X_l573_573971

noncomputable def maximal_podpyirayushchee := 0.01

theorem greatest_50_podpyirayushchee_X :
  ∀ (a : Fin 50 → ℝ), (∑ i in Finset.range 50, a i) ∈ Int → 
  (∃ i, | a i - (1/2) | ≥ maximal_podpyirayushchee) := sorry

end greatest_50_podpyirayushchee_X_l573_573971


namespace arrangement_count_correct_l573_573694

def student := unit
def teacher := unit

noncomputable def count_arrangements (students : list student) (teachers : list teacher) : ℕ :=
  if students.length = 5 ∧ teachers.length = 2 then
    960  -- from the solution, this is our result
  else
    0

theorem arrangement_count_correct (students : list student) (teachers : list teacher) :
  students.length = 5 → teachers.length = 2 →
  let valid_positions := 4 in
  count_arrangements students teachers = 960 :=
by
  intros h_students h_teachers
  -- This is where the proof would start but is omitted for now
  sorry

end arrangement_count_correct_l573_573694


namespace fraction_independence_of_n_in_triangle_l573_573521

variable {A B C : Type} [LinearOrder A] [AddGroupWithOne A] [Field B]

theorem fraction_independence_of_n_in_triangle
  (triangle : Triangle A)
  (angle_division : ∀ n : ℕ, divides_angle_into_equal_parts triangle C n)
  (intersections : (C_1 C_2 : C) → intersection_points [C_1, C_2]) :
  (∀ n > 0, (∃ (C_1 C_{n-1} : B), 
    ∃ AC_1 : B, ∃ AB : B, ∃ C_{n-1}B : B, 
    let frac := ((1 / AC_1) - (1 / AB)) / ((1 / C_{n-1}B) - (1 / AB)) in
    ∀ m : ℕ, (frac ≠ m → frac ≠ n))) := 
sorry

end fraction_independence_of_n_in_triangle_l573_573521


namespace jacket_final_price_l573_573342

/-- 
The initial price of the jacket is $20, 
the first discount is 40%, and the second discount is 25%. 
We need to prove that the final price of the jacket is $9.
-/
theorem jacket_final_price :
  let initial_price := 20
  let first_discount := 0.40
  let second_discount := 0.25
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 9 :=
by
  sorry

end jacket_final_price_l573_573342


namespace analytical_expression_range_of_f_l573_573597

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin(2 * x + π / 6)

theorem analytical_expression :
  ∀ A ω φ, (A = 2) → (ω = 2) → (φ = π / 6) →
  (f = λ x, A * Real.sin(ω * x + φ)) ∧
  ∀ k : ℤ, ∃ M : ℝ × ℝ, M = (2 * π / 3, -A) →  
  A * Real.sin(ω * (2 * π / 3) + φ) = -2 :=
by sorry

theorem range_of_f :
  ∀ x, x ∈ Set.Icc (π / 12) (π / 2) → f x ∈ Set.Icc (-1) 2 :=
by sorry

end analytical_expression_range_of_f_l573_573597


namespace simplify_fraction_48_72_l573_573996

theorem simplify_fraction_48_72 : (48 : ℚ) / 72 = 2 / 3 := sorry

end simplify_fraction_48_72_l573_573996


namespace number_of_valid_integers_l573_573193

theorem number_of_valid_integers (k : ℤ) (hk : k > 0) :
  ∃ a : ℤ, a = (4 ^ k + 2 ^ k) / 3 ∧
  ∀ n : ℕ, n ≤ 10 ^ k → (n % 3 = 0) → 
  (∀ d : ℕ, d ∈ Int.to_nat_digits 10 n → d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 7) → 
  a = count (λ n, n ≤ 10 ^ k ∧ 
                n % 3 = 0 ∧
                ∀ d : ℕ, d ∈ Int.to_nat_digits 10 n → d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 7) (finset.range (10 ^ k + 1)) :=
by {
  sorry,
}

end number_of_valid_integers_l573_573193


namespace analogical_reasoning_correct_l573_573302

-- Define the conditions as propositions
def condition_A := ∀ (a b : ℕ), a * 3 = b * 3 → a = b
def condition_B := ∀ (a b c : ℕ), (a + b) * c = a * c + b * c
def condition_C := ∀ (a b c : ℕ), (a + b) * c = a * c + b * c → c ≠ 0
def condition_D := ∀ (a b : ℕ), (a * b)^n = (a^n) * (b^n) → (a + b)^n = (a^n) + (b^n)

-- The proof problem Lean 4 statement
theorem analogical_reasoning_correct :
  (condition_C ∧ ¬ condition_A ∧ ¬ condition_B ∧ ¬ condition_D) :=
by
  sorry

end analogical_reasoning_correct_l573_573302


namespace sum_of_solutions_l573_573817

theorem sum_of_solutions : 
  let f : ℝ → ℝ := λ x, x^2 - 10 * x - 55
  let roots_sum := (λ (a b : ℝ), -b / a) in
  (roots_sum 1 (-10) = 10) :=
by
  let a := 1
  let b := -10
  let c := -55
  have h : a ≠ 0 := by norm_num
  simp [roots_sum]
  sorry

end sum_of_solutions_l573_573817


namespace greatest_50_podpyirayushchim_l573_573973

def is_podpyirayushchim (X : ℝ) : Prop :=
  ∀ (a : Fin 50 → ℝ), (∑ i in Finset.range 50, a i) ∈ ℤ → ∃ i, |a i - 0.5| ≥ X

theorem greatest_50_podpyirayushchim :
  ∀ X, is_podpyirayushchim X ↔ X ≤ 0.01 :=
sorry

end greatest_50_podpyirayushchim_l573_573973


namespace sin_eq_imp_relation_l573_573050

theorem sin_eq_imp_relation (α β : ℝ) (h : Real.sin α = Real.sin β) : 
  ∃ k : ℤ, α = k * Real.pi + (-1)^k * β := 
sorry

end sin_eq_imp_relation_l573_573050


namespace tan_alpha_in_fourth_quadrant_l573_573453

noncomputable def alpha : ℝ := sorry -- α is in the fourth quadrant and we will use ℝ as its type

theorem tan_alpha_in_fourth_quadrant (h1 : sin alpha + cos alpha = 1 / 5) 
                                     (h2 : sin alpha < 0) 
                                     (h3 : cos alpha > 0) :
  tan alpha = -3 / 4 :=
sorry

end tan_alpha_in_fourth_quadrant_l573_573453


namespace four_color_intersection_exists_l573_573353

/-- 
Given a 100 x 100 array where each point is colored in one of four colors (red, green, blue, or yellow),
and each row and each column have exactly 25 points of each color, prove that there exist two rows and
two columns such that their intersection points have four different colors.
-/
theorem four_color_intersection_exists :
  ∃ (rows: set (fin 100)) (cols: set (fin 100)),
    rows.card = 2 ∧ cols.card = 2 ∧
    ∀ (row1 row2 : fin 100) (col1 col2 : fin 100),
    row1 ≠ row2 → col1 ≠ col2 →
    (∃ (color1 color2 color3 color4 : fin 4),
    color1 ≠ color2 ∧ color1 ≠ color3 ∧ color1 ≠ color4 ∧ 
    color2 ≠ color3 ∧ color2 ≠ color4 ∧ color3 ≠ color4 ∧
    array row1 col1 = color1 ∧ 
    array row1 col2 = color2 ∧ 
    array row2 col1 = color3 ∧ 
    array row2 col2 = color4) := 
sorry

end four_color_intersection_exists_l573_573353


namespace find_third_number_x_l573_573368

variable {a b : ℝ}

theorem find_third_number_x (h : a < b) :
  (∃ x : ℝ, x = a * b / (2 * b - a) ∧ x < a) ∨ 
  (∃ x : ℝ, x = 2 * a * b / (a + b) ∧ a < x ∧ x < b) ∨ 
  (∃ x : ℝ, x = a * b / (2 * a - b) ∧ a < b ∧ b < x) :=
sorry

end find_third_number_x_l573_573368


namespace largest_fraction_of_consecutive_evens_l573_573844

theorem largest_fraction_of_consecutive_evens
  (a b c d : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b % 2 = 0)
  (h3 : c % 2 = 0)
  (h4 : d % 2 = 0)
  (h5 : 2 < a)
  (h6 : a < b)
  (h7 : b < c)
  (h8 : c < d)
  (h9 : d = b + 2)
  (h10 : b = a + 2)
  (h11 : c = a + 4) :
  (c + d) / (b + a) = 1.8 :=
by
  sorry

end largest_fraction_of_consecutive_evens_l573_573844


namespace cannot_determine_congruency_l573_573355

-- Define the congruency criteria for triangles
def SSS (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ c1 = c2
def SAS (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2
def ASA (angle1 b1 angle2 angle3 b2 angle4 : ℝ) : Prop := angle1 = angle2 ∧ b1 = b2 ∧ angle3 = angle4
def AAS (angle1 angle2 b1 angle3 angle4 b2 : ℝ) : Prop := angle1 = angle2 ∧ angle3 = angle4 ∧ b1 = b2
def HL (hyp1 leg1 hyp2 leg2 : ℝ) : Prop := hyp1 = hyp2 ∧ leg1 = leg2

-- Define the condition D, which states the equality of two corresponding sides and a non-included angle
def conditionD (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2

-- The theorem to be proven
theorem cannot_determine_congruency (a1 b1 angle1 a2 b2 angle2 : ℝ) :
  conditionD a1 b1 angle1 a2 b2 angle2 → ¬(SSS a1 b1 0 a2 b2 0 ∨ SAS a1 b1 0 a2 b2 0 ∨ ASA 0 b1 0 0 b2 0 ∨ AAS 0 0 b1 0 0 b2 ∨ HL 0 0 0 0) :=
by
  sorry

end cannot_determine_congruency_l573_573355


namespace graph_symmetry_center_graph_symmetry_line_not_max_value_odd_and_periodic_l573_573470

def f (x : ℝ) : ℝ := Real.cos x * Real.sin (2 * x)

theorem graph_symmetry_center : 
  ∀ x, f(2*Real.pi - x) + f(x) = 0 :=
by
  sorry

theorem graph_symmetry_line : 
  ∀ x, f(Real.pi - x) = f(x) :=
by
  sorry

theorem not_max_value : 
  ¬(∀ x, f(x) ≤ Real.sqrt 3 / 2) :=
by
  sorry

theorem odd_and_periodic : 
  (∀ x, f(-x) = -f(x)) ∧ (∀ x, f(x + 2 * Real.pi) = f(x)) :=
by
  sorry

end graph_symmetry_center_graph_symmetry_line_not_max_value_odd_and_periodic_l573_573470


namespace collinear_points_sum_l573_573912

variables {a b : ℝ}

/-- If the points (1, a, b), (a, b, 3), and (b, 3, a) are collinear, then b + a = 3.
-/
theorem collinear_points_sum (h : ∃ k : ℝ, 
  (a - 1, b - a, 3 - b) = k • (b - 1, 3 - a, a - b)) : b + a = 3 :=
sorry

end collinear_points_sum_l573_573912


namespace value_of_A_is_integer_l573_573227

theorem value_of_A_is_integer :
  let A := (8795689 * 8795688 * 8795687 * 8795686) / (8795688^2 + 8795686^2 + 8795684^2 + 8795682^2) -
            (8795684 * 8795683 * 8795682 * 8795681) / (8795688^2 + 8795686^2 + 8795684^2 + 8795682^2) in
  A = 43978425 :=
  sorry

end value_of_A_is_integer_l573_573227


namespace least_number_of_tiles_of_equal_size_l573_573307

theorem least_number_of_tiles_of_equal_size :
  ∀ (length width : ℕ),
  length = 624 →
  width = 432 →
  let tile_size := Nat.gcd length width in 
  (length / tile_size) * (width / tile_size) = 117 :=
by
  intros length width length_eq width_eq
  let tile_size := Nat.gcd length width
  sorry

end least_number_of_tiles_of_equal_size_l573_573307


namespace geometric_sequence_a7_l573_573512

variable {G : Type*} [LinearOrderedField G]

noncomputable def a : ℕ → G := sorry -- The definition of the sequence.

-- Given conditions:
axiom geo_seq (r : G) (init : G) : ∀ n, a n = init * r^n 
axiom cond : a 4 * a 10 = 16

-- To prove:
theorem geometric_sequence_a7 : a 7 = 4 ∨ a 7 = -4 := 
sorry

end geometric_sequence_a7_l573_573512


namespace angle_between_a_b_a_perpendicular_to_a_minus_2b_l573_573125

open Real

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
axiom a_norm : ∥a∥ = 2
axiom b_norm : ∥b∥ = 2
axiom a_add_b_norm : ∥a + b∥ = 2 * sqrt 3

-- Definition for the angle between vectors from the dot product
def angle_between (u v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  arccos ((u • v) / (∥u∥ * ∥v∥))

-- Problem 1: Prove that the angle between a and b is π/3
theorem angle_between_a_b : angle_between a b = π / 3 := by {
  sorry
}

-- Problem 2: Prove that a is perpendicular to (a - 2 * b)
theorem a_perpendicular_to_a_minus_2b : a • (a - (2 : ℝ) • b) = 0 := by {
  sorry
}

end angle_between_a_b_a_perpendicular_to_a_minus_2b_l573_573125


namespace speed_of_man_l573_573666

-- Definitions based on conditions
def length_of_train : ℝ := 400
def time_to_cross_man : ℝ := 23.998
def speed_of_train_kmph : ℝ := 63
def speed_of_train_mps : ℝ := (speed_of_train_kmph * 1000) / 3600
def relative_speed : ℝ := length_of_train / time_to_cross_man

-- Theorem statement
theorem speed_of_man :
  (speed_of_train_mps - relative_speed) * (3600 / 1000) = 2.9952 :=
by sorry

end speed_of_man_l573_573666


namespace train_time_l573_573631

-- Definitions for conditions
def speed_ratio (faster slower : ℕ) := faster = 3 * slower
def meets_after (distance speed time : ℕ) := distance = speed * time

theorem train_time (x t : ℕ) (H_speed_ratio : speed_ratio (3 * x) x)
  (H_meets : meets_after (t * x) x t)
  (H_slower_time : 36 * x = t * x)
  (H_faster_speed : 3 * x) : 
  36 / 3 = 12 :=
by
  sorry

end train_time_l573_573631


namespace real_solutions_eq_31_l573_573028

noncomputable def number_of_real_solutions : ℕ :=
  let zero := 0
  let fifty := 50
  let neg_fifty := -50
  let num_intervals := 8
  let num_solutions_per_interval := 2
  let total_solutions := num_intervals * num_solutions_per_interval * 2 - 1
  total_solutions

theorem real_solutions_eq_31 : number_of_real_solutions = 31 := by
  sorry

end real_solutions_eq_31_l573_573028


namespace range_of_m_l573_573074

variable (x m : ℝ)
def proposition_p := ∀ x, x^2 - 2 * m * x + 7 * m - 10 ≠ 0
def proposition_q := ∀ x ∈ set.Ioi 0, x^2 - m * x + 4 ≥ 0

theorem range_of_m (h1 : proposition_p m) (h2 : proposition_q m) 
  (hp_or_q : proposition_p m ∨ proposition_q m)
  (hp_and_q : proposition_p m ∧ proposition_q m) :
  2 < m ∧ m ≤ 4 :=
sorry

end range_of_m_l573_573074


namespace normal_CDF_is_correct_l573_573775

noncomputable def normal_cdf (a σ : ℝ) (x : ℝ) : ℝ :=
  0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2)

theorem normal_CDF_is_correct (a σ : ℝ) (ha : σ > 0) (x : ℝ) :
  (normal_cdf a σ x) = 0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2) :=
by
  sorry

end normal_CDF_is_correct_l573_573775


namespace find_green_towels_l573_573651

variable (G : ℕ)
variable (white_towels : ℕ := 21)
variable (towels_given : ℕ := 34)
variable (towels_left : ℕ := 22)

theorem find_green_towels :
  G + white_towels - towels_given = towels_left → G = 35 :=
by
  intro h
  have h_equiv : G - 13 = 22 := by
    rwa [add_comm, ←nat.sub_sub] at h
  exact eq_of_sub_eq_add h_equiv sorry

end find_green_towels_l573_573651


namespace amount_of_rice_distributed_in_first_5_days_l573_573654

-- Definitions from conditions
def workers_day (d : ℕ) : ℕ := if d = 1 then 64 else 64 + 7 * (d - 1)

-- The amount of rice each worker receives per day
def rice_per_worker : ℕ := 3

-- Total workers dispatched in the first 5 days
def total_workers_first_5_days : ℕ := (workers_day 1 + workers_day 2 + workers_day 3 + workers_day 4 + workers_day 5)

-- Given these definitions, we now state the theorem to prove
theorem amount_of_rice_distributed_in_first_5_days : total_workers_first_5_days * rice_per_worker = 1170 :=
by
  sorry

end amount_of_rice_distributed_in_first_5_days_l573_573654


namespace parallel_lines_determines_planes_l573_573276

theorem parallel_lines_determines_planes (l1 l2 l3 : line) (h1 : parallel l1 l2) (h2 : parallel l2 l3) :
  (∃ p : plane, (∀ x, (x = l1 ∨ x = l2 ∨ x = l3) → x ∈ p) ∧ ∀ y, y ∈ p → (y = l1 ∨ y = l2 ∨ y = l3)) ∨
  (∃ p1 p2 p3 : plane, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (∀ x, (x = l1 → x ∈ p1) ∧ (x = l2 → x ∈ p2) ∧ (x = l3 → x ∈ p3))) :=
by sorry

end parallel_lines_determines_planes_l573_573276


namespace min_value_inequality_l573_573188

theorem min_value_inequality (p q r : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) (h₂ : 0 < r) :
  ( 3 * r / (p + 2 * q) + 3 * p / (2 * r + q) + 2 * q / (p + r) ) ≥ (29 / 6) := 
sorry

end min_value_inequality_l573_573188


namespace dodecahedron_edges_l573_573272

noncomputable def regular_dodecahedron := Type

def faces : regular_dodecahedron → ℕ := λ _ => 12
def edges_per_face : regular_dodecahedron → ℕ := λ _ => 5
def shared_edges : regular_dodecahedron → ℕ := λ _ => 2

theorem dodecahedron_edges (d : regular_dodecahedron) :
  (faces d * edges_per_face d) / shared_edges d = 30 :=
by
  sorry

end dodecahedron_edges_l573_573272


namespace roger_steps_to_minutes_l573_573574

theorem roger_steps_to_minutes (h1 : ∃ t: ℕ, t = 30 ∧ ∃ s: ℕ, s = 2000)
                               (h2 : ∃ g: ℕ, g = 10000) :
  ∃ m: ℕ, m = 150 :=
by 
  sorry

end roger_steps_to_minutes_l573_573574


namespace problem_statement_l573_573509

/-
In the acute triangle \( \triangle ABC \), let \( O \) be the circumcenter, \( M \) be the midpoint of \( BC \). Point \( P \) lies on the arc \( \overparen{AO} \) of the circumcircle determined by points \( A \), \( O \), and \( M \). Points \( E \) and \( F \) lie on \( AC \) and \( AB \) respectively, and satisfy

\[ \angle APB = \angle BME \]
\[ \angle APC = \angle CMF \]

Prove that points \( B, C, E, \) and \( F \) are concyclic (lie on the same circle).
-/
noncomputable def acute_triangle_concyclic (A B C O M P : Point) (circumcircle : Circle) (E F : Point)
  [acute_triangle : Triangle ABC] (h1 : is_circumcenter O ABC) (h2 : midpoint M B C) 
  (h3 : on_arc P circumcircle (arc AO A M)) 
  (hE : on_segment E AC) (hF : on_segment F AB)
  (angle_condition1 : angle APB = angle BME) 
  (angle_condition2 : angle APC = angle CMF) : Prop :=
concyclic B C E F

/- Now we include the theorem statement which essentially states the problem -/
theorem problem_statement (A B C O M P : Point) (circumcircle : Circle) (E F : Point)
  [acute_triangle : Triangle ABC] (h1 : is_circumcenter O ABC) (h2 : midpoint M B C) 
  (h3 : on_arc P circumcircle (arc AO A M)) 
  (hE : on_segment E AC) (hF : on_segment F AB)
  (angle_condition1 : angle APB = angle BME) 
  (angle_condition2 : angle APC = angle CMF) : 
  concyclic B C E F := 
sorry

end problem_statement_l573_573509


namespace zeros_of_g_l573_573451

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def f (x : ℝ) : ℝ :=
  if x ∈ Ico 0 2 then abs (x^2 - x - 1) else sorry -- Define outside [0, 2) using periodicity and evenness, omitted for brevity

theorem zeros_of_g :
  is_even f →
  (∀ x ≥ 0, f (x + 2) = f x) →
  f = λ x, if x ∈ Ico 0 2 then abs (x^2 - x - 1) else f x →
  (set.count (λ x, f x - 1 = 0) (Icc (-2 : ℝ) 4) = 7) :=
by
  sorry

end zeros_of_g_l573_573451


namespace minimum_positive_period_of_transformed_function_is_pi_graph_of_transformed_function_is_symmetric_about_point_l573_573092

-- Conditions: Function definition and its transformation
def original_function (x : ℝ) : ℝ := 2 * sin x * cos x
def transformed_function (x : ℝ) : ℝ := sin (2 * x - π / 3)

-- Claims
theorem minimum_positive_period_of_transformed_function_is_pi :
  ((∃ T > 0, ∀ x, transformed_function (x + T) = transformed_function x) ∧ 
   (∀ T > 0, (∀ x, transformed_function (x + T) = transformed_function x) → T = π ∨ T > π)) :=
begin
  -- Proof needed here
  sorry
end

theorem graph_of_transformed_function_is_symmetric_about_point :
  (∃ x₀ y₀, (x₀ = π / 6) ∧ (y₀ = 0) ∧ 
    (∀ x, transformed_function (2 * x₀ - x) = 2 * y₀ - transformed_function x)) :=
begin
  -- Proof needed here
  sorry
end

end minimum_positive_period_of_transformed_function_is_pi_graph_of_transformed_function_is_symmetric_about_point_l573_573092


namespace FindAngleB_FindIncircleRadius_l573_573935

-- Define the problem setting
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Condition 1: a + c = 2b * sin (C + π / 6)
def Condition1 (T : Triangle) : Prop :=
  T.a + T.c = 2 * T.b * Real.sin (T.C + Real.pi / 6)

-- Condition 2: (b + c) (sin B - sin C) = (a - c) sin A
def Condition2 (T : Triangle) : Prop :=
  (T.b + T.c) * (Real.sin T.B - Real.sin T.C) = (T.a - T.c) * Real.sin T.A

-- Condition 3: (2a - c) cos B = b cos C
def Condition3 (T : Triangle) : Prop :=
  (2 * T.a - T.c) * Real.cos T.B = T.b * Real.cos T.C

-- Given: radius of incircle and dot product of vectors condition
def Given (T : Triangle) (r : ℝ) : Prop :=
  (T.a + T.c = 4 * Real.sqrt 3) ∧
  (2 * T.b * (T.a * T.c * Real.cos T.B - 3 * Real.sqrt 3 / 2) = 6)

-- Proof of B = π / 3
theorem FindAngleB (T : Triangle) :
  (Condition1 T ∨ Condition2 T ∨ Condition3 T) → T.B = Real.pi / 3 := 
sorry

-- Proof for the radius of the incircle
theorem FindIncircleRadius (T : Triangle) (r : ℝ) :
  Given T r → T.B = Real.pi / 3 → r = 1 := 
sorry


end FindAngleB_FindIncircleRadius_l573_573935


namespace car_prices_purchasing_plans_l573_573700

/-- The price of each car of model A is 150,000 yuan and the price of each car of model B is 100,000 yuan. --/
theorem car_prices 
  (A B : ℕ) 
  (x y : ℕ) 
  (h1 : 10 * x + 15 * y = 300) 
  (h2 : 8 * x + 18 * y = 300) : 
  x = 15 ∧ y = 10 := 
by
  sorry

/-- There are three purchasing plans, and Plan 3 yields the highest profit of 21 million yuan. --/
theorem purchasing_plans 
  (a b : ℕ) 
  (h1 : 15 * a + 10 * b ≤ 400) 
  (h2 : a + b = 30) 
  (h3 : 0.8 * a + 0.5 * b ≥ 204): 
  ∃ a b, a ∈ {18, 19, 20} ∧ b = 30 - a ∧ 
  [0.8 * 20 + 0.5 * 10 ≥ 21, 0.8 * 19 + 0.5 * 11 ≥ 20.7, 0.8 * 18 + 0.5 * 12 ≥ 20.4].max = [0.8 * 20 + 0.5 * 10] :=
by
  sorry

end car_prices_purchasing_plans_l573_573700


namespace root_sum_moduli_correct_l573_573966

noncomputable def sum_of_moduli_of_roots (a b : ℝ) (z1 z2 z3 z4 : ℂ) :=
  (z1.abs + z2.abs + z3.abs + z4.abs)

theorem root_sum_moduli_correct (a b : ℝ) (z1 z2 z3 z4 : ℂ) 
  (h_distinct : (z1 ≠ z2 ∧ z1 ≠ z3 ∧ z1 ≠ z4 ∧ z2 ≠ z3 ∧ z2 ≠ z4 ∧ z3 ≠ z4))
  (h_square : is_square z1 z2 z3 z4)
  (h_roots : ((polynomial.C b + polynomial.C a * polynomial.X + polynomial.X^2) *
              (polynomial.C (2*b) + polynomial.C a * polynomial.X + polynomial.X^2)).root_set ℂ 
              = {z1, z2, z3, z4}) :
  sum_of_moduli_of_roots a b z1 z2 z3 z4 = (√6 + 2 * √2) := by
  sorry

def is_square (z1 z2 z3 z4 : ℂ) : Prop :=
  let sq_dist (u v : ℂ) := (u - v).abs^2
  and
  let side_len_square := 1
  in
  sq_dist z1 z2 = side_len_square ∧
  sq_dist z2 z3 = side_len_square ∧
  sq_dist z3 z4 = side_len_square ∧
  sq_dist z4 z1 = side_len_square ∧
  sq_dist z1 z3 = (2 * side_len_square) ∧
  sq_dist z2 z4 = (2 * side_len_square)

end root_sum_moduli_correct_l573_573966


namespace greatest_50_podpyirayushchim_l573_573972

def is_podpyirayushchim (X : ℝ) : Prop :=
  ∀ (a : Fin 50 → ℝ), (∑ i in Finset.range 50, a i) ∈ ℤ → ∃ i, |a i - 0.5| ≥ X

theorem greatest_50_podpyirayushchim :
  ∀ X, is_podpyirayushchim X ↔ X ≤ 0.01 :=
sorry

end greatest_50_podpyirayushchim_l573_573972


namespace max_chips_on_grid_with_two_connected_l573_573982

-- Define the type for a 20x15 grid
structure Grid (rows : ℕ) (cols : ℕ) where
  cells : Fin rows × Fin cols → Bool -- A cell containing a chip is True, else False

def is_connected (r1 c1 r2 c2 : ℕ) (g : Grid 20 15) : Prop :=
  (r1 = r2 ∧ ∀ (c : ℕ), c1 < c ∧ c < c2 → ¬g.cells (⟨r1, by linarith⟩, ⟨c, by linarith⟩)) ∨
  (c1 = c2 ∧ ∀ (r : ℕ), r1 < r ∧ r < r2 → ¬g.cells (⟨r, by linarith⟩, ⟨c1, by linarith⟩))

-- Define the condition that each chip has at most two connected chips
def chip_has_at_most_two_connected (g : Grid 20 15) : Prop :=
  ∀ r c, g.cells (⟨r, by linarith⟩, ⟨c, by linarith⟩) → 
  (∑ r', if is_connected r c r' c g then 1 else 0) +
  (∑ c', if is_connected r c r c' g then 1 else 0) ≤ 2

-- Define the theorem stating the maximum number of chips
def max_chips (g : Grid 20 15) : ℕ :=
  ∑ r c, if g.cells (⟨r, by linarith⟩, ⟨c, by linarith⟩) then 1 else 0

theorem max_chips_on_grid_with_two_connected : ∃ g : Grid 20 15, chip_has_at_most_two_connected g ∧ max_chips g = 35 := 
sorry

end max_chips_on_grid_with_two_connected_l573_573982


namespace no_real_solution_l573_573772

theorem no_real_solution (x : ℝ) : 
  (¬ (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7) :=
sorry

end no_real_solution_l573_573772


namespace at_least_one_composite_l573_573952

theorem at_least_one_composite (a b c : ℕ) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) (h_odd_c : c % 2 = 1) 
    (h_not_perfect_square : ∀ m : ℕ, m * m ≠ a) : 
    a ^ 2 + a + 1 = 3 * (b ^ 2 + b + 1) * (c ^ 2 + c + 1) →
    (∃ p, p > 1 ∧ p ∣ (b ^ 2 + b + 1)) ∨ (∃ q, q > 1 ∧ q ∣ (c ^ 2 + c + 1)) :=
by sorry

end at_least_one_composite_l573_573952


namespace find_number_of_valid_n_l573_573812

def valid_n (n : ℕ) : Prop :=
  (2 < n) ∧ (n < 100) ∧ ((∀ k : ℕ, k >= 0 → n = 4 * k + 3) ∨ (∃ k : ℕ, n = 4 * k + 3))

theorem find_number_of_valid_n : 
  {n : ℕ | valid_n n}.card = 24 :=
by
  sorry

end find_number_of_valid_n_l573_573812


namespace determine_d_l573_573500

variables (a b c d : ℝ)

-- Conditions given in the problem
def condition1 (a b d : ℝ) : Prop := d / a = (d - 25) / b
def condition2 (b c d : ℝ) : Prop := d / b = (d - 15) / c
def condition3 (a c d : ℝ) : Prop := d / a = (d - 35) / c

-- Final statement to prove
theorem determine_d (a b c : ℝ) (d : ℝ) :
    condition1 a b d ∧ condition2 b c d ∧ condition3 a c d → d = 75 :=
by sorry

end determine_d_l573_573500


namespace competition_arrangements_l573_573213

-- Definitions of events and venues
inductive Event
| volleyball
| basketball
| soccer

inductive Venue
| venueA
| venueB
| venueC
| venueD

-- Define the main problem
theorem competition_arrangements : 
  let events := [Event.volleyball, Event.basketball, Event.soccer],
      venues := [Venue.venueA, Venue.venueB, Venue.venueC, Venue.venueD] in
  (sum_of_cases events venues) = 60 := 
sorry

-- Auxiliary function for arrangements (combining cases as described)
def sum_of_cases (events : List Event) (venues : List Venue) : ℕ :=
  let case1 := arrangements_diff_venues events venues,
      case2 := arrangements_two_same_venues events venues in
  case1 + case2

-- Case 1: Each event in a different venue
def arrangements_diff_venues (events : List Event) (venues : List Venue) : ℕ :=
  -- placeholder for the actual combinatorial calculation A_4^3
  24

-- Case 2: Two events in the same venue and the third event in a different venue
def arrangements_two_same_venues (events : List Event) (venues : List Venue) : ℕ :=
  -- placeholder for the actual combinatorial calculation C_3^2 * A_4^2
  36

-- Non-empty list lemma required for the above functions to work correctly
lemma non_empty_list : ∀ {α : Type} (l : List α), l ≠ [] → ∃ h t, l = h::t := sorry

end competition_arrangements_l573_573213


namespace sum_of_digits_n_plus_1_l573_573186

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_n_plus_1
  (n : ℕ)
  (hn : sum_of_digits n = 1274) :
  sum_of_digits (n + 1) = 1239 :=
by
  sorry

end sum_of_digits_n_plus_1_l573_573186


namespace john_payment_l573_573946

def camera_value : ℝ := 5000
def weekly_rental_percentage : ℝ := 0.10
def rental_period : ℕ := 4
def friend_contribution_percentage : ℝ := 0.40

theorem john_payment :
  let weekly_rental_fee := camera_value * weekly_rental_percentage
  let total_rental_fee := weekly_rental_fee * rental_period
  let friend_contribution := total_rental_fee * friend_contribution_percentage
  let john_payment := total_rental_fee - friend_contribution
  john_payment = 1200 :=
by
  sorry

end john_payment_l573_573946


namespace pair_without_hole_l573_573671

def color := {w | w = "white"} ∪ {b | b = "blue"} ∪ {g | g = "grey"}

structure Sock where
  sock_color : color
  has_hole : Bool

def box := 
  [ {sock_color := "white", has_hole := false},
    {sock_color := "white", has_hole := false},
    {sock_color := "blue", has_hole := false},
    {sock_color := "blue", has_hole := false},
    {sock_color := "blue", has_hole := false},
    {sock_color := "grey", has_hole := false},
    {sock_color := "grey", has_hole := false},
    {sock_color := "grey", has_hole := false},
    {sock_color := "grey", has_hole := false} ]

noncomputable def number_of_socks_to_get_pair_without_hole (socks : list Sock) : ℕ := sorry

theorem pair_without_hole : number_of_socks_to_get_pair_without_hole box ≥ 7 := sorry

end pair_without_hole_l573_573671


namespace four_edge_trips_from_A_to_C_l573_573599

-- Let's define a cube, edges, vertices and the conditions given in the problem.
structure Vertex :=
(x : ℕ)
(y : ℕ)
(z : ℕ)

structure Edge :=
(v1 : Vertex)
(v2 : Vertex)

-- Function to check if two vertices are direct neighbors i.e. differ by exactly one coordinate
def are_neighbors (v1 v2: Vertex) : Bool :=
(v1.x = v2.x ∧ v1.y = v2.y ∧ (v1.z - v2.z).abs = 1) ∨
(v1.x = v2.x ∧ (v1.y - v2.y).abs = 1 ∧ v1.z = v2.z) ∨
((v1.x - v2.x).abs = 1 ∧ v1.y = v2.y ∧ v1.z = v2.z)

-- Definition of a path in the graph
def path (vertices : List Vertex) : Bool :=
  vertices.length = 5 ∧
  ∀ i, (i < 4 → are_neighbors (vertices.nth i).getD (Vertex.mk 0 0 0) (vertices.nth (i + 1)).getD (Vertex.mk 0 0 0))

noncomputable def count_4_edge_trips (A C : Vertex) : ℕ :=
if ¬ are_neighbors A C then 
  let paths := {p | path p ∧ (p.head.getD (Vertex.mk 0 0 0) = A) ∧ (p.reverse.head.getD (Vertex.mk 0 0 0) = C)}
  paths.card
else 0

theorem four_edge_trips_from_A_to_C (A C : Vertex) (h₁ : ¬are_neighbors A C ∧ path [A, V₁, V₂, V₃, C]) :
  count_4_edge_trips A C = 24 :=
sorry

end four_edge_trips_from_A_to_C_l573_573599


namespace triangle_area_constant_circle_equation_minimum_distance_l573_573937

-- 1.  Prove that the area of triangle △AOB is a constant value (4)
theorem triangle_area_constant (t : ℝ) (ht : t ≠ 0) :
  let A := (2 * t, 0)
  let B := (0, 2 / t)
  let O := (0, 0)
  let area_triangle := 0.5 * abs (2 * t) * abs (2 / t)
  area_triangle = 4 :=
by
  sorry

-- 2. Given the line 2x + y - 4 = 0 intersects the circle at points M and N with |OM| = |ON|,
-- determine the equation of the circle (x - 2)^2 + (y - 1)^2 = 5
theorem circle_equation (t : ℝ) (ht : t ≠ 0) ( |OM| = |ON|) :
  let C := (2, 1)
  ∃h: t = 2, (x - 2)^2 + (y - 1)^2 = 5 :=
by
  sorry

-- 3. Under the same conditions, prove the minimum value of |PB| + |PQ| for moving points P
-- on x+y+2=0 and Q on circle (x-2)^2 + (y-1)^2 = 5 is 2√5, and coordinates of P at this time are
-- ( -4/3, -2/3 )
theorem minimum_distance (P Q: (ℝ × ℝ)) :
  let B := (0, 2/t)
  let B' := (-4, -2)
  let |PB| + |PQ| := abs (P - B) + abs (P - B')
  minimum_value_t_squared:= 2*sqrt (5)
  coordinates of P (P, Q)| (|PB| + |PQ|) = sorry :=
by
  sorry

end triangle_area_constant_circle_equation_minimum_distance_l573_573937


namespace min_swaps_to_descending_l573_573271

theorem min_swaps_to_descending (n : ℕ) (h : n ≥ 1) :
  ∃ (initial_arrangement final_arrangement : list ℕ),
  initial_arrangement = list.range n.succ.tail ∧
  final_arrangement = (list.range n.succ).tail.reverse →
  ((∃ f : (list ℕ → ℕ), f initial_arrangement = 0 ∧ f final_arrangement = (n * (n - 1)) / 2) →
  ((∃ swap_adj : list ℕ → list ℕ → Prop, 
    (∀ arr : list ℕ, swap_adj arr (list.swap arr _ _) → 
    (f arr = (f (list.swap arr _ _)) + 1 ∨ f arr = (f (list.swap arr _ _)) - 1)) → 
  f final_arrangement = (n * (n - 1)) / 2 → 
  (list ℕ → Prop) → 
  ∃ steps: ℕ, steps = (n * (n - 1)) / 2)
  )
sorry

end min_swaps_to_descending_l573_573271


namespace hyperbola_equation_l573_573249

noncomputable def hyperbola_params (a b c : ℝ) :=
  ∃ (k : ℝ), b = k * a ∧ a = 3 ∧ b = sqrt 7 ∧ c = 4 ∧ (c - a = 1) ∧ (c ^ 2 = a ^ 2 + b ^ 2)

theorem hyperbola_equation : 
  ∀ (a b c : ℝ), 
  hyperbola_params a b c → 
  (∃ (x y : ℝ), (x^2 / 9) - (y^2 / 7) = 1) :=
sorry

end hyperbola_equation_l573_573249


namespace acute_isosceles_inscribed_in_circle_l573_573359

noncomputable def solve_problem : ℝ := by
  -- Let x be the angle BAC
  let x : ℝ := π * 5 / 11
  -- Considering the value of k in the problem statement
  let k : ℝ := 5 / 11
  -- Providing the value of k obtained from solving the problem
  exact k

theorem acute_isosceles_inscribed_in_circle (ABC : Type)
  [inhabited ABC]
  (inscribed : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (B_tangent C_tangent : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (D : ABC)
  (angle_eq : ∀ {A B C : ABC}, ∠ABC = ∠ACB)
  (triple_angle : ∀ {A B C D : ABC}, ∠ABC = 3 * ∠BAC) :
  solve_problem = 5 / 11 := 
sorry

end acute_isosceles_inscribed_in_circle_l573_573359


namespace ratio_of_ages_l573_573899

open Real

theorem ratio_of_ages (father_age son_age : ℝ) (h1 : father_age = 45) (h2 : son_age = 15) :
  father_age / son_age = 3 :=
by
  sorry

end ratio_of_ages_l573_573899


namespace arnold_total_protein_l573_573724

-- Conditions
def protein_in_collagen_powder (scoops: ℕ) : ℕ := 9 * scoops
def protein_in_protein_powder (scoops: ℕ) : ℕ := 21 * scoops
def protein_in_steak : ℕ := 56
def protein_in_greek_yogurt : ℕ := 15
def protein_in_almonds (cups: ℕ) : ℕ := 6 * (cups * 4) / 4
def half_cup_almonds_protein : ℕ := 12

-- Statement
theorem arnold_total_protein : 
  protein_in_collagen_powder 1 + protein_in_protein_powder 2 + protein_in_steak + protein_in_greek_yogurt + half_cup_almonds_protein = 134 :=
  by
    sorry

end arnold_total_protein_l573_573724


namespace real_solutions_of_equation_l573_573023

theorem real_solutions_of_equation :
  ∃ n : ℕ, n ≈ 59 ∧ (∀ x : ℝ, x ∈ Icc (-50) 50 → (x / 50 = Real.sin x → x ∈ real_roots_of_eq))
    where
      real_roots_of_eq := {x : ℝ | x / 50 = Real.sin x} :=
sorry

end real_solutions_of_equation_l573_573023


namespace vector_addition_magnitude_l573_573830

variables {a b : ℝ}

theorem vector_addition_magnitude (ha : abs a = 1) (hb : abs b = 2)
  (angle_ab : real.angle.to_degrees (real.angle.arctan2 b a) = 60) :
  abs (a + b) = sqrt 7 :=
by sorry

end vector_addition_magnitude_l573_573830


namespace price_per_shirt_l573_573330

section TShirtFactory

variable (employees shirtsPerPerson hoursPerDay wagePerHour wagePerShirt nonEmployeeExpenses dailyProfit : ℕ)

/-- Total number of people working in the factory -/
def numEmployees := employees

/-- Average number of shirts each person makes per day -/
def avgShirtsPerPerson := shirtsPerPerson

/-- Number of hours employees work per day -/
def hoursWorked := hoursPerDay

/-- Hourly wage paid per employee -/
def hourlyWage := wagePerHour

/-- Wage paid per shirt made by employee -/
def perShirtWage := wagePerShirt

/-- Non-employee expenses per day -/
def additionalExpenses := nonEmployeeExpenses

/-- Company's profit per day -/
def dailyProfit := dailyProfit

/-- To determine the price the company sells each shirt for given the stated conditions -/
theorem price_per_shirt (h_employees : numEmployees = 20)
  (h_shirtsPerPerson : avgShirtsPerPerson = 20)
  (h_hoursWorked : hoursWorked = 8)
  (h_hourlyWage : hourlyWage = 12)
  (h_perShirtWage : perShirtWage = 5)
  (h_additionalExpenses : additionalExpenses = 1000)
  (h_dailyProfit : dailyProfit = 9080) :
  let total_shirts := numEmployees * avgShirtsPerPerson
      total_hourly_wages := numEmployees * hoursWorked * hourlyWage
      total_per_shirt_wages := total_shirts * perShirtWage
      total_wages := total_hourly_wages + total_per_shirt_wages
      total_expenses := total_wages + additionalExpenses
      total_revenue := dailyProfit + total_expenses
      price_per_shirt := total_revenue / total_shirts
  in price_per_shirt = 35 :=
by
  sorry

end TShirtFactory

end price_per_shirt_l573_573330


namespace point_B_coordinates_l573_573147

theorem point_B_coordinates :
  ∃ (B : ℝ × ℝ), (B.1 < 0) ∧ (|B.2| = 4) ∧ (|B.1| = 5) ∧ (B = (-5, 4) ∨ B = (-5, -4)) :=
sorry

end point_B_coordinates_l573_573147


namespace intersection_of_A_and_B_l573_573846

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l573_573846


namespace usual_time_proof_l573_573689

noncomputable 
def usual_time (P T : ℝ) := (P * T) / (100 - P)

theorem usual_time_proof (P T U : ℝ) (h1 : P > 0) (h2 : P < 100) (h3 : T > 0) (h4 : U = usual_time P T) : U = (P * T) / (100 - P) :=
by
    sorry

end usual_time_proof_l573_573689


namespace area_transformed_parallelogram_l573_573120

-- Define the vectors a and b
variables (a b : ℝ^3)

-- Condition: The area of the parallelogram generated by vectors a and b is 12
def area_ab : ℝ := ‖a × b‖
axiom h : area_ab a b = 12

-- The statement: Prove that the area of the parallelogram generated by 3a + 4b and 2a - 6b is 312
theorem area_transformed_parallelogram : ‖(3 • a + 4 • b) × (2 • a - 6 • b)‖ = 312 :=
sorry

end area_transformed_parallelogram_l573_573120


namespace count_positive_integers_satisfying_product_inequality_l573_573806

theorem count_positive_integers_satisfying_product_inequality :
  ∃ (k : ℕ), k = 23 ∧ 
  (n : ℕ ) → (2 ≤ n ∧ n < 100) →
  ((∃ (m: ℕ), 2 + 4 * m = n) ∧ ((n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0) = 23 :=
by
  sorry

end count_positive_integers_satisfying_product_inequality_l573_573806


namespace actual_estate_area_l573_573562

theorem actual_estate_area (map_scale : ℝ) (length_inches : ℝ) (width_inches : ℝ) 
  (actual_length : ℝ) (actual_width : ℝ) (area_square_miles : ℝ) 
  (h_scale : map_scale = 300)
  (h_length : length_inches = 4)
  (h_width : width_inches = 3)
  (h_actual_length : actual_length = length_inches * map_scale)
  (h_actual_width : actual_width = width_inches * map_scale)
  (h_area : area_square_miles = actual_length * actual_width) :
  area_square_miles = 1080000 :=
sorry

end actual_estate_area_l573_573562


namespace smallest_z_in_arithmetic_and_geometric_progression_l573_573961

theorem smallest_z_in_arithmetic_and_geometric_progression :
  ∃ x y z : ℤ, x < y ∧ y < z ∧ (2 * y = x + z) ∧ (z^2 = x * y) ∧ z = -2 :=
by
  sorry

end smallest_z_in_arithmetic_and_geometric_progression_l573_573961


namespace probability_two_red_one_blue_l573_573348

theorem probability_two_red_one_blue :
    let C_total := 512
    let C_two_red_one_blue := 48
    C_two_red_one_blue / C_total = 3 / 32 :=
by
    let C_total := 512
    let C_two_red_one_blue := 48
    rw [div_eq_mul_inv, div_eq_mul_inv]
    exact (by norm_num : (48:ℝ) = 3 * 16)
    exact (by norm_num : (512:ℝ) = 32 * 16)
    exact (by norm_num : (3:ℝ) = 3)
    exact (by norm_num : (32:ℝ) = 32)

end probability_two_red_one_blue_l573_573348


namespace probability_cos_eq_one_third_l573_573569
noncomputable def probability_cos_between (x : ℝ) (hx : x ∈ set.Icc (-1 : ℝ) (1 : ℝ)) : ℝ :=
if (0 ≤ cos (π * x / 2) ∧ cos (π * x / 2) ≤ 1/2) then 1 else 0

theorem probability_cos_eq_one_third : 
  (set_integral interval_integrable (λ x, probability_cos_between x (set.mem_Icc_of_Icc (-1 : ℝ) 1) 
   (set.interval_integrable_iff_measure_univ_of_open_interval)).to_real =
    1 / 3 :=
begin
  -- proof omitted
  sorry
end

end probability_cos_eq_one_third_l573_573569


namespace midpoint_inequality_l573_573185

open_locale classical

variables {A B C D : Point} (K L M N : Point)
variables [convex_quadrilateral A B C D]
variables (midpoint_K : midpoint K A B)
variables (midpoint_L : midpoint L B C)
variables (midpoint_M : midpoint M C D)
variables (midpoint_N : midpoint N D A)

theorem midpoint_inequality :
  KM ≤ (BC + AD) / 2 ↔ BC ∥ AD := sorry

end midpoint_inequality_l573_573185


namespace lid_circumference_l573_573763

noncomputable def π : ℝ := Real.pi

def diameter : ℝ := 2

def circumference (d : ℝ) : ℝ := π * d

theorem lid_circumference : circumference diameter ≈ 6.28318 := 
by 
  sorry

end lid_circumference_l573_573763


namespace area_of_rectangle_ABCD_l573_573153

-- Definitions of the given conditions
def BC : ℝ := 10
def EC : ℝ := 6
def area_EDF := λ area_FAB => area_FAB - 5

-- Given the above conditions, we need to prove the following statement
theorem area_of_rectangle_ABCD
  (area_FAB : ℝ)
  (area_EDF : ℝ := area_EDF area_FAB)
  (area_Δ_EBC : ℝ := 1/2 * BC * EC) :
  area_Δ_EBC + 5 = 35 := 
by 
  sorry

end area_of_rectangle_ABCD_l573_573153


namespace lipstick_cost_is_correct_l573_573557

noncomputable def cost_of_lipstick (palette_cost : ℝ) (num_palettes : ℝ) (hair_color_cost : ℝ) (num_hair_colors : ℝ) (total_paid : ℝ) (num_lipsticks : ℝ) : ℝ :=
  let total_palette_cost := num_palettes * palette_cost
  let total_hair_color_cost := num_hair_colors * hair_color_cost
  let remaining_amount := total_paid - (total_palette_cost + total_hair_color_cost)
  remaining_amount / num_lipsticks

theorem lipstick_cost_is_correct :
  cost_of_lipstick 15 3 4 3 67 4 = 2.5 :=
by
  sorry

end lipstick_cost_is_correct_l573_573557


namespace find_angle_ADE_l573_573921

variables {A B C D E : Type} [geometry A B C D E]

-- Define the conditions of the problem.
def rectangle (ABCD : Prop) := ∀ A B C D : Type, is_rectangle A B C D
lemma angle_DAC_120 (DAC : ℝ) : DAC = 120 := sorry
def intersects (D : Type) (line : Type) (E : Type) := ∀ D line, ∃ E, intersects D line E
lemma angle_EDC_150 (EDC : ℝ) : EDC = 150 := sorry

-- Define the main theorem to be proved.
theorem find_angle_ADE 
  (H1 : rectangle ABCD)
  (H2 : angle_DAC_120 120)
  (H3 : intersects D AC E)
  (H4 : angle_EDC_150 150) :
  ∃ ADE, ADE = 90 :=
by
  sorry

end find_angle_ADE_l573_573921


namespace number_divided_approximation_l573_573983

theorem number_divided_approximation :
  let divisor : ℝ := 153.75280898876406
  let quotient : ℝ := 89
  let remainder : ℝ := 14
  let number := divisor * quotient + remainder
  number ≈ 13698 :=
by
  sorry

end number_divided_approximation_l573_573983


namespace graph_transformation_point_l573_573461

theorem graph_transformation_point {f : ℝ → ℝ} (h : f 1 = 0) : f (0 + 1) + 1 = 1 :=
by
  sorry

end graph_transformation_point_l573_573461


namespace max_value_of_function_l573_573580

variables {x y : ℝ}

def ineq1 := x + 7 * y ≤ 32
def ineq2 := 2 * x + 5 * y ≤ 42
def ineq3 := 3 * x + 4 * y ≤ 62
def eqn4  := 2 * x + y = 34
def nonneg_x := x ≥ 0
def nonneg_y := y ≥ 0

def objective_function := 3 * x + 8 * y

theorem max_value_of_function : 
  (ineq1) → (ineq2) → (ineq3) → (eqn4) → (nonneg_x) → (nonneg_y) → objective_function x y = 64 := 
  sorry

end max_value_of_function_l573_573580


namespace count_positive_integers_satisfying_product_inequality_l573_573804

theorem count_positive_integers_satisfying_product_inequality :
  ∃ (k : ℕ), k = 23 ∧ 
  (n : ℕ ) → (2 ≤ n ∧ n < 100) →
  ((∃ (m: ℕ), 2 + 4 * m = n) ∧ ((n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0) = 23 :=
by
  sorry

end count_positive_integers_satisfying_product_inequality_l573_573804


namespace greatest_common_divisor_three_divisors_l573_573287

theorem greatest_common_divisor_three_divisors (m : ℕ) (h : ∃ (D : set ℕ), D = {d | d ∣ 120 ∧ d ∣ m} ∧ D.card = 3) : 
  ∃ p : ℕ, p.prime ∧ greatest_dvd_set {d | d ∣ 120 ∧ d ∣ m} = p^2 := 
sorry

end greatest_common_divisor_three_divisors_l573_573287


namespace multinomial_expansion_terms_l573_573586

theorem multinomial_expansion_terms :
  let terms := { (a, b, c) : ℕ × ℕ × ℕ // a + b + c = 10 }
  in terms.finite.to_finset.card = 66 :=
by {
  sorry
}

end multinomial_expansion_terms_l573_573586


namespace xiao_zhang_arrives_earlier_l573_573703

-- Define the uniform distribution for Xiao Zhang and Xiao Wang's arrival times
def T_Z : ℝ → Prop := λ t, 0 ≤ t ∧ t ≤ 20
def T_W : ℝ → Prop := λ t, 0 ≤ t ∧ t ≤ 20

-- Define the event that Xiao Zhang arrives at least 10 minutes earlier than Xiao Wang
def event (T_Z T_W : ℝ) : Prop := T_Z ≤ T_W - 10

-- Define the probability that Xiao Zhang arrives at least 10 minutes earlier than Xiao Wang
def probability_of_event : ℝ := 1/8

-- The theorem statement that needs to be proved
theorem xiao_zhang_arrives_earlier :
  ∀ (TZ TW : ℝ), probability_of_event = (P (event TZ TW)) := sorry

end xiao_zhang_arrives_earlier_l573_573703


namespace find_p_real_roots_l573_573771

-- Define the polynomial
def poly (p x : ℝ) := x^4 + 6*p*x^3 + 3*x^2 + 6*p*x + 9

-- Define the condition for the polynomial to have all real roots
def poly_has_real_roots (p : ℝ) : Prop :=
  ∀ x : ℝ, poly p x = 0 → ∃ (a b c d : ℝ), x = a ∨ x = b ∨ x = c ∨ x = d

-- Main theorem statement
theorem find_p_real_roots :
  ∀ p : ℝ, (poly_has_real_roots p ↔ p ∈ set.Iic (-sqrt (1/3)) ∪ set.Ici (sqrt (1/3))) :=
by sorry

end find_p_real_roots_l573_573771


namespace mike_training_hours_l573_573977

-- Define the individual conditions
def first_weekday_hours : Nat := 2
def first_weekend_hours : Nat := 1
def first_week_days : Nat := 5
def first_weekend_days : Nat := 2

def second_weekday_hours : Nat := 3
def second_weekend_hours : Nat := 2
def second_week_days : Nat := 4  -- since the first day of second week is a rest day
def second_weekend_days : Nat := 2

def first_week_hours : Nat := (first_weekday_hours * first_week_days) + (first_weekend_hours * first_weekend_days)
def second_week_hours : Nat := (second_weekday_hours * second_week_days) + (second_weekend_hours * second_weekend_days)

def total_training_hours : Nat := first_week_hours + second_week_hours

-- The final proof statement
theorem mike_training_hours : total_training_hours = 28 := by
  exact sorry

end mike_training_hours_l573_573977


namespace luzins_theorem_l573_573661

open Set MeasureTheory

theorem luzins_theorem {α : Type*} [MeasurableSpace α] [BorelSpace α] {a b : ℝ} (f : α → ℝ)
  (meas_f : Measurable f) (hab : a < b) : 
  ∀ ε > 0, ∃ (fε : ℝ → ℝ),
    Continuous fε ∧
    (measure (Icc a b ∩ {x | f x ≠ fε x}) < ε) ∧
    (∀ x ∈ Icc a b, |fε x| ≤ ∨ x ∈ Icc a b, |f x|) :=
begin
  sorry
end

end luzins_theorem_l573_573661


namespace mary_finds_eggs_l573_573976

theorem mary_finds_eggs (initial final found : ℕ) (h_initial : initial = 27) (h_final : final = 31) :
  found = final - initial → found = 4 :=
by
  intro h
  rw [h_initial, h_final] at h
  exact h

end mary_finds_eggs_l573_573976


namespace f_2016_eq_sin_x_l573_573431

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, 0
| (n + 1) := λ x, if n = 0 then cos x else (f n) ' x

theorem f_2016_eq_sin_x :
  ∀ x, (f 2016) x = sin x :=
by sorry

end f_2016_eq_sin_x_l573_573431


namespace movie_ticket_notation_l573_573902

-- Definition of movie ticket notation
def ticket_notation (row : ℕ) (seat : ℕ) : (ℕ × ℕ) :=
  (row, seat)

-- Given condition: "row 10, seat 3" is denoted as (10, 3)
def given := ticket_notation 10 3 = (10, 3)

-- Proof statement: "row 6, seat 16" is denoted as (6, 16)
theorem movie_ticket_notation : ticket_notation 6 16 = (6, 16) :=
by
  -- Proof omitted, since the theorem statement is the focus
  sorry

end movie_ticket_notation_l573_573902


namespace conjugate_in_first_quadrant_l573_573082

theorem conjugate_in_first_quadrant (i : ℂ) (z : ℂ) 
  (hi : i.im = 1 ∧ i.re = 0)
  (hz : z * (1 + i) = (-1 / 2 + (Real.sqrt 3) / 2 * i) ^ 3) : 
  (z.conj.re > 0 ∧ z.conj.im > 0) :=
by
  sorry

end conjugate_in_first_quadrant_l573_573082


namespace initial_number_of_men_l573_573581

theorem initial_number_of_men (P : ℝ) (M : ℝ) (h1 : P = 15 * M * (P / (15 * M))) (h2 : P = 12.5 * (M + 200) * (P / (12.5 * (M + 200)))) : M = 1000 :=
by
  sorry

end initial_number_of_men_l573_573581


namespace johns_age_l573_573165

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 70) : j = 20 := by
  sorry

end johns_age_l573_573165


namespace greatest_power_of_two_factor_l573_573297

theorem greatest_power_of_two_factor (n m : ℕ) (h1 : n = 12) (h2 : m = 8) :
  ∃ k, k = 1209 ∧ 2^k ∣ n^603 - m^402 :=
by
  sorry

end greatest_power_of_two_factor_l573_573297


namespace simplify_and_evaluate_l573_573232

theorem simplify_and_evaluate :
  let a := (-1: ℝ) / 3
  let b := (-3: ℝ)
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 :=
by
  have a_def : a = (-1: ℝ) / 3 := rfl
  have b_def : b = (-3: ℝ) := rfl
  sorry

end simplify_and_evaluate_l573_573232


namespace max_elements_A_l573_573103

-- Definitions of the sets and properties given in the conditions
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2020 }
def A (s : Set ℕ) : Prop := s ⊆ M ∧ ∀ x ∈ s, 4 * x ∉ s

-- Statement of the problem to be proven
theorem max_elements_A : ∃ A ⊆ M, (∀ a ∈ A, (4 * a) ∉ A) ∧ A.card = 1616 :=
sorry

end max_elements_A_l573_573103


namespace value_abs_sqrt_expr_l573_573616

theorem value_abs_sqrt_expr : |real.sqrt 2 - real.sqrt 3| + 2 * real.sqrt 2 = real.sqrt 3 + real.sqrt 2 :=
begin
  sorry
end

end value_abs_sqrt_expr_l573_573616


namespace nathan_strawberries_per_plant_l573_573209

theorem nathan_strawberries_per_plant :
  ∃ S : ℕ,
    let strawberry_plants := 5,
        tomato_plants := 7,
        tomatoes_per_plant := 16,
        tomatoes_per_basket := 7,
        strawberries_per_basket := 7,
        strawberry_basket_price := 9,
        tomato_basket_price := 6,
        total_revenue := 186 in
    ((∃ B_s B_t : ℕ, 
         B_t = (tomatoes_per_plant * tomato_plants) / tomatoes_per_basket ∧ 
         (strawberry_basket_price * B_s + tomato_basket_price * B_t = total_revenue) ∧
         B_s * strawberries_per_basket = 5 * S)
    ∧ S = 14) :=
begin
  sorry
end

end nathan_strawberries_per_plant_l573_573209


namespace floor_sqrt_150_l573_573005

theorem floor_sqrt_150 : (Real.floor (Real.sqrt 150)) = 12 := by
  have h₁ : 12^2 = 144 := by rfl
  have h₂ : 13^2 = 169 := by rfl
  have h₃ : (12 : ℝ) < Real.sqrt 150 := by 
    have : (12 : ℝ) < (Real.sqrt 144) := by sorry
    sorry
  have h₄ : Real.sqrt 150 < (13 : ℝ) := by 
    have : (Real.sqrt 169) < 13 := by sorry
    sorry
  exact sorry

end floor_sqrt_150_l573_573005


namespace exact_two_out_of_four_satisfied_l573_573713

-- Definitions based on the problem conditions
def P_S := 0.6         -- Probability that a resident is satisfied
def P_U := 1 - P_S     -- Probability that a resident is unsatisfied

def C (n k : ℕ) : ℕ := (n.choose k)  -- Choose function to calculate combinations (n choose k)

noncomputable def P_exac_two_sats : ℝ := 
  let prob_each_sequence := P_S * P_S * P_U * P_U in
  C 4 2 * prob_each_sequence  -- Probability for exactly two satisfied out of four

-- Theorem to be proven
theorem exact_two_out_of_four_satisfied : 
  P_exac_two_sats = 0.3456 :=
by
  sorry

end exact_two_out_of_four_satisfied_l573_573713


namespace find_number_of_valid_n_l573_573814

def valid_n (n : ℕ) : Prop :=
  (2 < n) ∧ (n < 100) ∧ ((∀ k : ℕ, k >= 0 → n = 4 * k + 3) ∨ (∃ k : ℕ, n = 4 * k + 3))

theorem find_number_of_valid_n : 
  {n : ℕ | valid_n n}.card = 24 :=
by
  sorry

end find_number_of_valid_n_l573_573814


namespace number_of_true_propositions_l573_573357

-- Definitions based on the problem
def proposition1 (α β : ℝ) : Prop := (α + β = 180) → (α + β = 90)
def proposition2 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)
def proposition3 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)

-- Proof problem statement
theorem number_of_true_propositions : ∃ n : ℕ, n = 2 :=
by
  let p1 := false
  let p2 := false
  let p3 := true
  existsi (if p3 then 1 else 0 + if p2 then 1 else 0 + if p1 then 1 else 0)
  simp
  sorry

end number_of_true_propositions_l573_573357


namespace find_ratio_l573_573749

variable (A B C D E F G H E1 F1 G1 H1 : Point)
variable [ConvexQuadrilateral ABCD]
variable [ConvexQuadrilateral EFGH]
variable [ConvexQuadrilateral E1F1G1H1]
variable (lambda : ℝ)

def points_on_sides (A B C D E F G H : Point) : Prop :=
  E ∈ lineSegment A B ∧
  F ∈ lineSegment B C ∧
  G ∈ lineSegment C D ∧
  H ∈ lineSegment D A

def internal_ratio_product_one (A B C D E F G H : Point) : Prop :=
  (length (lineSegment A E) / length (lineSegment E B)) *
  (length (lineSegment B F) / length (lineSegment F C)) *
  (length (lineSegment C G) / length (lineSegment G D)) *
  (length (lineSegment D H) / length (lineSegment H A)) = 1

def vertices_on_sides (E1 F1 G1 H1 A B C D : Point) : Prop :=
  A ∈ lineSegment H1 E1 ∧
  B ∈ lineSegment E1 F1 ∧
  C ∈ lineSegment F1 G1 ∧
  D ∈ lineSegment G1 H1

def parallel_segments (E1 F1 G1 H1 E F G H : Point) : Prop :=
  parallel (lineSegment E1 F1) (lineSegment E F) ∧
  parallel (lineSegment F1 G1) (lineSegment F G) ∧
  parallel (lineSegment G1 H1) (lineSegment G H) ∧
  parallel (lineSegment H1 E1) (lineSegment H E)

def given_ratio (E1 A H1 : Point) (lambda : ℝ) : Prop :=
  length (lineSegment E1 A) / length (lineSegment A H1) = lambda

theorem find_ratio (A B C D E F G H E1 F1 G1 H1 : Point)
  [ConvexQuadrilateral ABCD]
  [ConvexQuadrilateral EFGH]
  [ConvexQuadrilateral E1F1G1H1]
  (lambda : ℝ)
  (h1 : points_on_sides A B C D E F G H)
  (h2 : internal_ratio_product_one A B C D E F G H)
  (h3 : vertices_on_sides E1 F1 G1 H1 A B C D)
  (h4 : parallel_segments E1 F1 G1 H1 E F G H)
  (h5 : given_ratio E1 A H1 lambda) :
  length (lineSegment F1 C) / length (lineSegment C G1) = lambda :=
sorry

end find_ratio_l573_573749


namespace number_of_valid_subsets_l573_573421

-- Definition of ν₃ function which gives largest k such that 3^k divides n.
def ν₃ (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    Nat.find (λ k => 3^(k + 1) ∣ n ∨ ¬3^k ∣ n)

-- The main theorem statement
theorem number_of_valid_subsets :
  ∃ S : Finset (Fin 82), ∀ (a b: ℕ), a ∈ S → b ∈ S → a ≠ b → (ν₃ (a - b)).even :=
  2 ^ 81 := sorry

end number_of_valid_subsets_l573_573421


namespace geometric_sequence_a7_l573_573133

noncomputable def a (n : ℕ) : ℝ := sorry -- Definition of the sequence

theorem geometric_sequence_a7 :
  a 3 = 1 → a 11 = 25 → a 7 = 5 := 
by
  intros h3 h11
  sorry

end geometric_sequence_a7_l573_573133


namespace union_complement_eq_l573_573104

def U := {0, 1, 2, 3, 4, 5, 6}
def A := {2, 4, 5}
def B := {0, 1, 3, 5}

noncomputable def complement_U_B := U \ B

theorem union_complement_eq :
  A ∪ complement_U_B = {2, 4, 5, 6} := by
  sorry

end union_complement_eq_l573_573104


namespace distinct_rectangles_l573_573337

theorem distinct_rectangles :
  ∃! (l w : ℝ), l * w = 100 ∧ l + w = 24 :=
sorry

end distinct_rectangles_l573_573337


namespace find_common_difference_of_arithmetic_sequence_l573_573843

open BigOperators

noncomputable def arithmetic_sequence_var (a b c d e : ℝ) : ℝ :=
  (1 / 5) * ((a - c)^2 + (b - c)^2 + (c - c)^2 + (d - c)^2 + (e - c)^2)

theorem find_common_difference_of_arithmetic_sequence 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_seq : ∀ n, a (n+1) = a n + d) 
  (h_pos : d > 0) 
  (h_var : arithmetic_sequence_var (a 1) (a 2) (a 3) (a 4) (a 5) = 3) :
    d = real.sqrt 6 / 2 :=
sorry

end find_common_difference_of_arithmetic_sequence_l573_573843


namespace carlos_more_miles_than_dana_after_3_hours_l573_573247

-- Define the conditions
variable (carlos_total_distance : ℕ)
variable (carlos_advantage : ℕ)
variable (dana_total_distance : ℕ)
variable (time_hours : ℕ)

-- State the condition values that are given in the problem
def conditions : Prop :=
  carlos_total_distance = 50 ∧
  carlos_advantage = 5 ∧
  dana_total_distance = 40 ∧
  time_hours = 3

-- State the proof goal
theorem carlos_more_miles_than_dana_after_3_hours
  (h : conditions carlos_total_distance carlos_advantage dana_total_distance time_hours) :
  carlos_total_distance - dana_total_distance = 10 :=
by
  sorry

end carlos_more_miles_than_dana_after_3_hours_l573_573247


namespace reachable_or_reachable_l573_573715

variable (BusStop : Type)
variable (reachable : BusStop → BusStop → Prop)

/-- Y comes after X means:
 1. Every bus stop from which X can be reached is a bus stop from which Y can be reached.
 2. Every bus stop that can be reached from Y can also be reached from X.
-/
def comes_after (X Y : BusStop) : Prop :=
  (∀ Z, reachable Z X → reachable Z Y) ∧ (∀ Z, reachable Y Z → reachable X Z)

axiom reachable_iff_comes_after :
  ∀ X Y : BusStop, reachable X Y ↔ comes_after X Y

theorem reachable_or_reachable (A B : BusStop) : reachable A B ∨ reachable B A :=
  sorry

end reachable_or_reachable_l573_573715


namespace counting_positive_integers_satisfying_inequality_l573_573784

theorem counting_positive_integers_satisfying_inequality :
  ∃ (n : ℕ), (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 49 → (n - 2 * k) < 0) ∧ n = 47 :=
begin
  sorry
end

end counting_positive_integers_satisfying_inequality_l573_573784


namespace power_computation_l573_573741

theorem power_computation :
  16^10 * 8^6 / 4^22 = 16384 :=
by
  sorry

end power_computation_l573_573741


namespace digit_250_after_decimal_in_1_over_13_l573_573636

/-- A decimal representation helper function to get the k-th digit of a repeating sequence after the decimal point -/
def repeating_decimal_sequence (s : string) (k : ℕ) : char :=
  if s.length = 0 then ' '
  else s.get ((k - 1) % s.length)

theorem digit_250_after_decimal_in_1_over_13 :
  repeating_decimal_sequence "076923" 250 = '9' :=
by
  sorry

end digit_250_after_decimal_in_1_over_13_l573_573636


namespace second_player_win_l573_573619

-- Define the game setup
def stones_game (boxes : ℕ) (target: ℕ) (stones_per_move : ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ) → (ℕ → Prop), ∀ h, strategy h = target

-- Formalize the game conditions
def game_conditions : Prop :=
  ∀ (boxes target stones_per_move : ℕ), 
  boxes = 11 ∧ 
  stones_per_move = 10 ∧ 
  target = 21 →
  stones_game boxes target stones_per_move

-- Theorem stating the second player guarantees a win
theorem second_player_win : game_conditions → ∃ strategy, strategy = (λ _ => true) :=
by 
  intros 
  sorry

end second_player_win_l573_573619


namespace sum_fractional_series_l573_573384

theorem sum_fractional_series :
  ∑ n in finset.range 500, (1 / ((n + 1) ^ 2 + (n + 1))) = 501 / 502 := 
  sorry

end sum_fractional_series_l573_573384


namespace shaded_area_concentric_circles_l573_573252

theorem shaded_area_concentric_circles :
  (∃ (O : ℝ × ℝ) (r₁ r₂ : ℝ), r₁ = 40 ∧ r₁ < r₂ ∧ 
    (∃ (A B P : ℝ × ℝ), (dist A B = 120) ∧
      (dist A P = dist P B = 60) ∧
      (dist O P = 40) ∧
      (dist O A = sqrt(60^2 + 40^2)) ∧
      (dist O B = sqrt(60^2 + 40^2)) ∧
      (∃ (t : ℝ), set_of (λ (θ : ℝ), 
        ((A.fst = r₂ * cos θ) ∧ (A.snd = r₂ * sin θ)) ∨
        ((B.fst = r₂ * cos θ) ∧ (B.snd = r₂ * sin θ))) ∧
        P = (r₁ * cos t, r₁ * sin t) ∧
        O = (0, 0)))) →
  (π * (dist O A^2 - dist O P^2) = 3600 * π) :=
sorry

end shaded_area_concentric_circles_l573_573252


namespace A_best_strategy_l573_573928

-- Definitions for the conditions
def A_hitting_probability (x : ℝ) : ℝ := x^2
def B_hitting_probability (x : ℝ) : ℝ := x
def critical_point : ℝ := (Real.sqrt 5 - 1) / 2

-- Theorem stating A's best strategy
theorem A_best_strategy :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) →
  (A_hitting_probability x > B_hitting_probability x) → 
  x = critical_point ∨ (A_hitting_probability x < B_hitting_probability x → x < critical_point) :=
by
  sorry

end A_best_strategy_l573_573928


namespace probability_listens_to_second_class_l573_573340

theorem probability_listens_to_second_class (interval_start interval_end class_start class_end arrive_start arrive_end : ℝ)
  (h1 : class_start = 8 + 5 / 6)
  (h2 : class_end = 9 + 1 / 2)
  (h3 : arrive_start = 9 + 1 / 6)
  (h4 : arrive_end = 10)
  (h5 : interval_start = 9 + 1 / 6)
  (h6 : interval_end = 9 + 1 / 3) :
  (interval_end - interval_start) / (arrive_end - arrive_start) = 1 / 5 := 
  sorry

end probability_listens_to_second_class_l573_573340


namespace sqrt_of_4_l573_573612

theorem sqrt_of_4 : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_of_4_l573_573612


namespace problem_statement_l573_573196

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then 2 * x - 3 else if x = 0 then 0 else -(2 * (-x) - 3)

-- Defining the property that f is an odd function
lemma odd_function (x : ℝ) : f (-x) = -f x :=
  by sorry

-- Defining the given function behavior for positive inputs
lemma f_positive (x : ℝ) (h : x > 0) : f x = 2 * x - 3 :=
  by sorry

-- The main proof statement
theorem problem_statement : f (-2) + f 0 = -1 :=
  by sorry

end problem_statement_l573_573196


namespace sum_of_repeating_digits_eq_20_l573_573261

noncomputable def repeating_seq_digits := (3, 8, 4, 5) -- Representing (c, d, e, f) as (3, 8, 4, 5)

theorem sum_of_repeating_digits_eq_20 :
    let (c, d, e, f) := repeating_seq_digits in
    c + d + e + f = 20 :=
by
    sorry

end sum_of_repeating_digits_eq_20_l573_573261


namespace sum_of_n_square_eq_perfect_square_l573_573033

theorem sum_of_n_square_eq_perfect_square :
  ∑ n in (Finset.filter (λ n, ∃ x : ℕ, n^2 - 17 * n + 72 = x^2) 
  (Finset.range 1 100)), n = 17 := 
sorry

end sum_of_n_square_eq_perfect_square_l573_573033


namespace evaluate_difference_of_squares_l573_573769

theorem evaluate_difference_of_squares :
  (64^2 - 36^2 = 2800) :=
by
  -- Using the difference of squares formula
  have h : (64^2 - 36^2) = (64 + 36) * (64 - 36), by
  {
    exact (eq.symm (Nat.sub_eq_mul (sq 64) (sq 36)))
  }
  -- Simplifying the expression
  calc
    64^2 - 36^2 = (64 + 36) * (64 - 36) : by rw h
            ... = 100 * 28           : by norm_num
            ... = 2800               : by norm_num

end evaluate_difference_of_squares_l573_573769


namespace FrankFilled3BagsOnSunday_l573_573426

variable (BagsSaturday BagsPerBag TotalCans : ℕ)
variable (CansSaturday CansSunday BagsSunday : ℕ)

def BagsSaturday := 5
def BagsPerBag := 5
def TotalCans := 40
def CansSaturday := BagsSaturday * BagsPerBag
def CansSunday := TotalCans - CansSaturday
def BagsSunday := CansSunday / BagsPerBag

theorem FrankFilled3BagsOnSunday :
    BagsSunday = 3 :=
by
  unfold BagsSaturday BagsPerBag TotalCans
  unfold CansSaturday CansSunday BagsSunday
  sorry

end FrankFilled3BagsOnSunday_l573_573426


namespace roots_k_m_l573_573545

theorem roots_k_m (k m : ℝ) 
  (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 11 ∧ a * b + b * c + c * a = k ∧ a * b * c = m)
  : k + m = 52 :=
sorry

end roots_k_m_l573_573545


namespace inequality_proof_l573_573727

noncomputable def x1 (a b : ℝ) : ℝ := b / a
noncomputable def x2 (b c : ℝ) : ℝ := c / b
noncomputable def x3 (c d : ℝ) : ℝ := d / c
noncomputable def x4 (d e : ℝ) : ℝ := e / d
noncomputable def x5 (e a : ℝ) : ℝ := a / e

theorem inequality_proof (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  let x1 := b / a, x2 := c / b, x3 := d / c, x4 := e / d, x5 := a / e in
  x1 * x2 * x3 * x4 * x5 = 1 →
  (x1 + x1 * x2 * x3) / (1 + x1 * x2 + x1 * x2 * x3 * x4) + 
  (x2 + x2 * x3 * x4) / (1 + x2 * x3 + x2 * x3 * x4 * x5) + 
  (x3 + x3 * x4 * x5) / (1 + x3 * x4 + x3 * x4 * x5 * x1) + 
  (x4 + x4 * x5 * x1) / (1 + x4 * x5 + x4 * x5 * x1 * x2) + 
  (x5 + x5 * x1 * x2) / (1 + x5 * x1 + x5 * x1 * x2 * x3) ≥ 10 / 3 := 
begin
  sorry
end

end inequality_proof_l573_573727


namespace Sabrina_initial_cookies_l573_573223

-- Definitions of the initial conditions
variable (cookies_at_start : ℕ)
variable (cookies_to_brother : ℕ := 10)
variable (cookies_from_mother : ℕ := cookies_to_brother / 2)
variable (cookies_after_giving_to_sister : ℕ := 5)
variable (cookies_total_after_mother_help : ℕ := cookies_after_giving_to_sister * 3)

-- Conditions as hypotheses
variables (cond1 : cookies_after_giving_to_sister = cookies_total_after_mother_help * (1 / 3))
variables (cond2 : cookies_from_mother = cookies_to_brother / 2)
variables (cond3 : cookies_total_before_sister := cookies_total_after_mother_help - cookies_from_mother)
variables (cond4 : cookies_at_start = cookies_total_before_sister + cookies_to_brother)

-- Goal: To prove that the number of cookies at the start is 20
theorem Sabrina_initial_cookies : cookies_at_start = 20 := by
  sorry

end Sabrina_initial_cookies_l573_573223


namespace weight_of_berries_l573_573989

theorem weight_of_berries (total_weight : ℝ) (melon_weight : ℝ) : total_weight = 0.63 → melon_weight = 0.25 → total_weight - melon_weight = 0.38 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end weight_of_berries_l573_573989


namespace find_a8_l573_573064

variable {a : ℕ → ℝ} -- Assuming the sequence is real-valued for generality

-- Defining the necessary properties and conditions of the arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions as hypothesis
variable (h_seq : arithmetic_sequence a) 
variable (h_sum : a 3 + a 6 + a 10 + a 13 = 32)

-- The proof statement
theorem find_a8 : a 8 = 8 :=
by
  sorry -- The proof itself

end find_a8_l573_573064


namespace compare_magnitude_l573_573968

def f (x : ℝ) : ℝ := Real.log (1 - x)
def g (x : ℝ) : ℝ := Real.log (1 + x)

theorem compare_magnitude (x : ℝ) (h : -1 < x ∧ x < 1) :
  (0 < x ∧ x < 1 → |f x| > |g x|) ∧ 
  (x = 0 → |f x| = |g x|) ∧ 
  (-1 < x ∧ x < 0 → |f x| < |g x|) :=
by
  sorry

end compare_magnitude_l573_573968


namespace solution_set_f_x_minus_2_ge_zero_l573_573870

-- Define the necessary conditions and prove the statement
noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_minus_2_ge_zero (f_even : ∀ x, f x = f (-x))
  (f_mono : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_zero : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x | x ≥ 3 ∨ x ≤ 1} :=
by {
  sorry
}

end solution_set_f_x_minus_2_ge_zero_l573_573870


namespace number_of_solutions_l573_573788
-- Importing the necessary library

-- Define the condition for the product of the sequence to be negative
noncomputable def condition (n : ℕ) :=
  (n > 0) ∧ (Finset.prod (Finset.range 49) (λ k, (n - 2 * (k + 1)) : ℤ) < 0)

-- State theorem corresponding to the mathematically equivalent proof problem
theorem number_of_solutions : 
  finset.filter condition (finset.range (99)).card = 24 :=
sorry

end number_of_solutions_l573_573788


namespace real_solutions_of_equation_l573_573024

theorem real_solutions_of_equation :
  ∃ n : ℕ, n ≈ 59 ∧ (∀ x : ℝ, x ∈ Icc (-50) 50 → (x / 50 = Real.sin x → x ∈ real_roots_of_eq))
    where
      real_roots_of_eq := {x : ℝ | x / 50 = Real.sin x} :=
sorry

end real_solutions_of_equation_l573_573024


namespace prob_sum_of_three_dice_is_18_l573_573649

theorem prob_sum_of_three_dice_is_18 :
  let die_values := {1, 2, 3, 4, 5, 6}
  let outcomes := list.product (list.product die_values die_values) die_values
  let count_events := list.filter (fun (abc : ℕ × ℕ × ℕ) => abc.1.1 + abc.1.2 + abc.2 = 18) outcomes
  let total_events := outcomes.length
  (count_events.length : ℚ) / total_events = 1 / 216 := by
  sorry

end prob_sum_of_three_dice_is_18_l573_573649


namespace find_a_l573_573880

noncomputable def f (x a : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a (a : ℝ) :
  f (1 - a) a = f (1 + a) a ↔ a = -3 / 4 :=
by
  intro h
  sorry

end find_a_l573_573880


namespace weight_of_substance_l573_573617

variable (k W1 W2 : ℝ)

theorem weight_of_substance (h1 : ∃ (k : ℝ), ∀ (V W : ℝ), V = k * W)
  (h2 : 48 = k * W1) (h3 : 36 = k * 84) : 
  (∃ (W2 : ℝ), 48 = (36 / 84) * W2) → W2 = 112 := 
by
  sorry

end weight_of_substance_l573_573617


namespace isosceles_triangle_l573_573916

theorem isosceles_triangle (a c : ℝ) (A C : ℝ) (h : a * Real.sin A = c * Real.sin C) : a = c → Isosceles :=
sorry

end isosceles_triangle_l573_573916


namespace moles_of_ammonium_chloride_combined_l573_573778

-- Define the conditions
def balanced_equation (nh4cl naoh nh4oh nacl: ℕ) : Prop :=
  nh4cl = 1 ∧ naoh = 1 ∧ nh4oh = 1 ∧ nacl = 1

-- State the problem in Lean 4
theorem moles_of_ammonium_chloride_combined (nh4cl naoh nh4oh nacl : ℕ) :
  balanced_equation nh4cl naoh nh4oh nacl → nh4oh = 1 → nh4cl = 1 :=
by
  intro h1 h2,
  cases h1,
  cases h1_right,
  sorry

end moles_of_ammonium_chloride_combined_l573_573778


namespace cos_double_angle_l573_573860

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 3/5) : Real.cos (2 * a) = 7/25 :=
by
  sorry

end cos_double_angle_l573_573860


namespace hyperbola_eccentricity_l573_573069

-- Definitions based on conditions
variables {a b : ℝ} (ha : 0 < a) (hb : 0 < b)

def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def circle (x y : ℝ) : Prop := x^2 + y^2 = a^2 + b^2

-- Hypothesis: point P that intersects both the hyperbola and the circle
variables {x y : ℝ}

def point_P := hyperbola x y ∧ circle x y ∧ y = a

-- Target: Prove the eccentricity e of the hyperbola equals (1 + sqrt 5) / 2
def eccentricity (e : ℝ) : Prop := e = (1 + Real.sqrt 5) / 2

theorem hyperbola_eccentricity : 
  point_P → ∃ e, eccentricity e :=
by sorry

end hyperbola_eccentricity_l573_573069


namespace y_coordinate_equidistant_l573_573635

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ ptC ptD : ℝ × ℝ, ptC = (-3, 0) → ptD = (4, 5) → 
    dist (0, y) ptC = dist (0, y) ptD) ∧ y = 16 / 5 :=
by
  sorry

end y_coordinate_equidistant_l573_573635


namespace number_of_students_l573_573309

theorem number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = 80 * N)
  (h2 : (T - 160) / (N - 8) = 90) :
  N = 56 :=
sorry

end number_of_students_l573_573309


namespace minimum_value_of_D_l573_573049

noncomputable def D (x a : ℝ) : ℝ :=
  sqrt ((x - a)^2 + (log x - a^2 / 4)^2) + a^2 / 4 + 1

theorem minimum_value_of_D (a : ℝ) : ∃ x : ℝ, D x a = sqrt 2 :=
sorry

end minimum_value_of_D_l573_573049


namespace range_of_a_l573_573080

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x > 1, f x = a^x) ∧ 
  (∀ x ≤ 1, f x = (4 - (a / 2)) * x + 2) → 
  4 ≤ a ∧ a < 8 :=
by
  sorry

end range_of_a_l573_573080


namespace evaluate_seventy_two_square_minus_twenty_four_square_l573_573008

theorem evaluate_seventy_two_square_minus_twenty_four_square :
  72 ^ 2 - 24 ^ 2 = 4608 := 
by {
  sorry
}

end evaluate_seventy_two_square_minus_twenty_four_square_l573_573008


namespace determine_a_l573_573596

theorem determine_a (a b c : ℤ)
  (vertex_condition : ∀ x : ℝ, x = 2 → ∀ y : ℝ, y = -3 → y = a * (x - 2) ^ 2 - 3)
  (point_condition : ∀ x : ℝ, x = 1 → ∀ y : ℝ, y = -2 → y = a * (x - 2) ^ 2 - 3) :
  a = 1 :=
by
  sorry

end determine_a_l573_573596


namespace product_simplification_l573_573234

theorem product_simplification :
  ∏ k in Finset.range 101, (4 * k + 6) / (4 * k + 2) = 203 := by
sorry

end product_simplification_l573_573234


namespace problem_ab_value_l573_573055

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x^2 - 4 * x else a * x^2 + b * x

theorem problem_ab_value (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) → a * b = 12 :=
by
  intro h
  let f_eqn := h 1 -- Checking the function equality for x = 1
  sorry

end problem_ab_value_l573_573055


namespace binary_rep_of_17_l573_573403

theorem binary_rep_of_17 : Nat.toDigits 2 17 = [1, 0, 0, 0, 1] :=
by
  sorry

end binary_rep_of_17_l573_573403


namespace number_of_valid_sets_l573_573341

noncomputable def num_sets_of_integers_satisfying_properties : Nat :=
  24

theorem number_of_valid_sets :
  ∃ (S : Set (Finset ℕ)), 
    (∀ s ∈ S, s.card = 5 ∧ 
      (∀ n ∈ s, odd n ∧ 2 < n) ∧ 
      (∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧
        let prod := Finset.prod s id in 
        prod ≥ 10000 ∧ prod < 100000 ∧ 
        (prod % 10000) / 1000 = A ∧ 
        (prod % 100) / 10 = 0 ∧ 
        (prod / 10000) = A ∧ 
        (prod % 10) = B)) ∧
    card S = num_sets_of_integers_satisfying_properties := 
  sorry

end number_of_valid_sets_l573_573341


namespace trig_identities_l573_573218

theorem trig_identities :
  (sin 15 = (sqrt 6 - sqrt 2) / 4) ∧ (cos 15 = (sqrt 6 + sqrt 2) / 4) ∧ 
  (sin 18 = (-1 + sqrt 5) / 4) ∧ (cos 18 = (sqrt (10 + 2 * sqrt 5)) / 4) :=
by
  sorry

end trig_identities_l573_573218


namespace Geraldo_drank_pints_l573_573405

variable (gallons_of_tea : ℝ)
variable (num_containers : ℝ)
variable (containers_drank : ℝ)
variable (consumption_rate : ℝ)
variable (gallons_to_pints : ℝ)
variable (foam_percentage : ℝ)

def tea_problem_conditions : Prop :=
  gallons_of_tea = 50 ∧
  num_containers = 200 ∧
  containers_drank = 44 ∧
  consumption_rate = 0.75 ∧
  gallons_to_pints = 8 ∧
  foam_percentage = 0.05

theorem Geraldo_drank_pints (h : tea_problem_conditions) : 
  let tea_per_container := gallons_of_tea / num_containers,
      pints_per_container := tea_per_container * gallons_to_pints,
      liquid_content_per_container := pints_per_container * (1 - foam_percentage),
      total_liquid_consumed := containers_drank * liquid_content_per_container,
      actual_liquid_drank := total_liquid_consumed * consumption_rate 
  in actual_liquid_drank = 62.7 := 
by
  -- Proof goes here
  sorry

end Geraldo_drank_pints_l573_573405


namespace collinear_vectors_m_n_sum_l573_573484

theorem collinear_vectors_m_n_sum (m n : ℝ) 
  (h_a : (2, 3, m) = λ * (2n, 6, 8))
  (h_collinear : ∃ λ : ℝ, (2, 3, m) = λ • (2n, 6, 8)) :
  m + n = 6 :=
by
  sorry

end collinear_vectors_m_n_sum_l573_573484


namespace find_a_l573_573553

theorem find_a (a : ℤ) :
  let U := {1, 3, 5, 7}
  let M := {1, Int.abs (a - 5)}
  M ⊆ U ∧ (U \ M) = {5, 7} → (a = 2 ∨ a = 8) :=
by
  sorry

end find_a_l573_573553


namespace total_cost_of_items_l573_573721

theorem total_cost_of_items
  (E P M : ℕ)
  (h1 : E + 3 * P + 2 * M = 240)
  (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_items_l573_573721


namespace odd_consecutive_nums_divisibility_l573_573987

theorem odd_consecutive_nums_divisibility (a b : ℕ) (h_consecutive : b = a + 2) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) : (a^b + b^a) % (a + b) = 0 := by
  sorry

end odd_consecutive_nums_divisibility_l573_573987


namespace find_p_from_fraction_l573_573747

theorem find_p_from_fraction : 
  let a := 2022 
  (let expr :=  (a + 1) / a - a / (a + 1), 
  let p := 2 * a + 1, 
  let q := a * (a + 1)) 
  in (p, q).gcd = 1 -> p = 4045 := 
by
  sorry

end find_p_from_fraction_l573_573747


namespace intersection_of_domain_and_range_l573_573430

def f (x : ℝ) : ℝ := Real.log (1 + x)
def g (x : ℝ) : ℝ := 2 * x + 1

noncomputable def domain_of_ln (f : ℝ → ℝ) : Set ℝ := {x | 1 + x > 0}
noncomputable def range_of_linear (g : ℝ → ℝ) : Set ℝ := {y | ∃ x : ℝ, y = 2 * x + 1}

theorem intersection_of_domain_and_range :
  Set.inter (domain_of_ln f) (range_of_linear g) = {y | 1 < y} :=
by
  sorry

end intersection_of_domain_and_range_l573_573430


namespace accident_rate_is_100_million_l573_573948

theorem accident_rate_is_100_million (X : ℕ) (h1 : 96 * 3000000000 = 2880 * X) : X = 100000000 :=
by
  sorry

end accident_rate_is_100_million_l573_573948


namespace measure_minor_arc_MB_l573_573130

variable {P : Type} [is_circle P]
variable {M B C : ↑P} -- Points M, B, C are on the circle P
variable (angle_MBC : angle M B C = 60)

theorem measure_minor_arc_MB : measure_minor_arc M B = 60 := 
sorry

end measure_minor_arc_MB_l573_573130


namespace problem1_problem2_l573_573467

-- Conditions and definitions
def f (x : ℝ) := 2 * sqrt 3 * sin (π + x) * cos (-3 * π - x) - 2 * sin (π / 2 - x) * cos (π - x)

def condition1 : Prop :=
  ∀ k : ℤ, ∀ x : ℝ, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → 
    (2 * sin (2 * x + π / 6) + 1) = f x 

def condition2 (α : ℝ) : Prop :=
  (π / 2 < α ∧ α < π ∧ f (α / 2 - π / 12) = 3 / 2) →
    cos (2 * α + π / 3) = (7 + 3 * sqrt 5) / 16

-- Lean 4 theorem statements
theorem problem1 : condition1 :=
by 
  sorry

theorem problem2 : ∀ α : ℝ, condition2 α :=
by 
  sorry

end problem1_problem2_l573_573467


namespace max_profit_l573_573327

variables (x y : ℝ)

def profit (x y : ℝ) : ℝ := 50000 * x + 30000 * y

theorem max_profit :
  (3 * x + y ≤ 13) ∧ (2 * x + 3 * y ≤ 18) ∧ (x ≥ 0) ∧ (y ≥ 0) →
  (∃ x y, profit x y = 390000) :=
by
  sorry

end max_profit_l573_573327


namespace fraction_of_sums_l573_573546

-- Definitions required for the problem
variables (a_1 d : ℝ)

-- Defining a_4 and a_2 for the arithmetic sequence
def a_4 := a_1 + 3 * d
def a_2 := a_1 + d

-- Defining the sum of the first n terms of the arithmetic sequence
def S : ℕ → ℝ
| 0     := 0
| (n+1) := (a_1 + n * d) + S n

-- Problem statement in Lean 4
theorem fraction_of_sums (h : a_4 / a_2 = 5 / 3) : 
  (S 4 / S 2) = 14 / 5 := by sorry

end fraction_of_sums_l573_573546


namespace ryan_chinese_learning_hours_l573_573770

theorem ryan_chinese_learning_hours : 
    ∀ (h_english : ℕ) (diff : ℕ), 
    h_english = 7 → 
    h_english = 2 + (h_english - diff) → 
    diff = 5 := by
  intros h_english diff h_english_eq h_english_diff_eq
  sorry

end ryan_chinese_learning_hours_l573_573770


namespace real_solutions_eq59_l573_573021

theorem real_solutions_eq59 :
  (∃ (x: ℝ), -50 ≤ x ∧ x ≤ 50 ∧ (x / 50) = sin x) ∧
  (∃! (S: ℕ), S = 59) :=
sorry

end real_solutions_eq59_l573_573021


namespace sphere_surface_area_l573_573565

-- Let A, B, C, D be distinct points on the same sphere
variables (A B C D : ℝ)

-- Defining edges AB, AC, AD and their lengths
variables (AB AC AD : ℝ)
variable (is_perpendicular : AB * AC = 0 ∧ AB * AD = 0 ∧ AC * AD = 0)

-- Setting specific edge lengths
variables (AB_length : AB = 1) (AC_length : AC = 2) (AD_length : AD = 3)

-- The proof problem: Prove that the surface area of the sphere is 14π
theorem sphere_surface_area : 4 * Real.pi * ((1 + 4 + 9) / 4) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l573_573565


namespace hyperbola_asymptotes_l573_573443

theorem hyperbola_asymptotes (a b c : ℝ) (A B : ℝ × ℝ) :
  (a > 0) →
  (b > 0) →
  (A.1 ^ 2 / a ^ 2 - A.2 ^ 2 / b ^ 2 = 1) →
  (B.1 ^ 2 / a ^ 2 - B.2 ^ 2 / b ^ 2 = 1) →
  (c ^ 2 = a ^ 2 + b ^ 2) →
  (A.1^2 + A.2^2 = c^2 / 2) →
  (B.1^2 + B.2^2 = c^2 / 2) →
  ((A.1, A.2), (B.1, B.2) ≠ (0,0)) → (y = A.2) → (y = B.2) → ∃ m, (A.2/A.1 = m) ∧ (A.2/A.1 = B.2/B.1) ∧ m^2 = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end hyperbola_asymptotes_l573_573443


namespace log_c_a_lt_log_c_b_l573_573863

theorem log_c_a_lt_log_c_b (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < c) (h4 : c < 1) :
  log c a < log c b := 
sorry

end log_c_a_lt_log_c_b_l573_573863


namespace least_k_cover_l573_573531

-- Define the main set M
def M := {1, 2, ..., 19}

-- Define the condition for A
def covers (A : Set ℕ) : Prop :=
  ∀ b ∈ M, ∃ a_i a_j ∈ A, b = a_i ∨ b = a_i + a_j ∨ b = a_i - a_j

-- The final statement of the problem
theorem least_k_cover :
  ∃ A : Set ℕ, (A ⊆ M) ∧ covers A ∧ (∀ B : Set ℕ, (B ⊆ M) ∧ covers B → B.card ≥ A.card) ∧ A.card = 5 :=
sorry

end least_k_cover_l573_573531


namespace abc_order_l573_573052

open Real

noncomputable def a : ℝ := exp 0.11
noncomputable def b : ℝ := (1.1)^(1.1)
def c : ℝ := 1.11

theorem abc_order :
  a > b ∧ b > c :=
by
  -- Proof steps will be filled here
  sorry

end abc_order_l573_573052


namespace incircle_radius_l573_573501

-- Define the elements of the problem.
variables (D E F : Type) [euclidean_geom D E F] (DE DF EF : ℝ)

-- Given conditions
def DEF_right_triangle := angle D 90
def DEF_angle_E_45 := angle E 45
def DEF_DE_8 := side DE 8

-- Target statement
theorem incircle_radius (D E F : Type) [euclidean_geom D E F] (DE DF EF : ℝ)
  (h1 : DEF_right_triangle D E F)
  (h2 : DEF_angle_E_45 D E F)
  (h3 : DEF_DE_8 D E F) :
  incircle_radius D E F = 4 - 2 * sqrt 2 := 
sorry

end incircle_radius_l573_573501


namespace division_remainder_is_7_l573_573248

theorem division_remainder_is_7 (d q D r : ℕ) (hd : d = 21) (hq : q = 14) (hD : D = 301) (h_eq : D = d * q + r) : r = 7 :=
by
  sorry

end division_remainder_is_7_l573_573248


namespace river_ratio_l573_573257

theorem river_ratio (total_length straight_length crooked_length : ℕ) 
  (h1 : total_length = 80) (h2 : straight_length = 20) 
  (h3 : crooked_length = total_length - straight_length) : 
  (straight_length / Nat.gcd straight_length crooked_length) = 1 ∧ (crooked_length / Nat.gcd straight_length crooked_length) = 3 := 
by
  sorry

end river_ratio_l573_573257


namespace quadratic_no_real_roots_l573_573488

theorem quadratic_no_real_roots (k : ℝ) : 
  (let Δ := (-5)^2 - 4 * 1 * k in Δ < 0) ↔ k > 25 / 4 :=
by sorry

end quadratic_no_real_roots_l573_573488


namespace log_base_2_3_l573_573079

theorem log_base_2_3 (a : ℝ) (h : a = Real.log 3 / Real.log 2) : 4^a + 4^(-a) = 82 / 9 := by
  sorry

end log_base_2_3_l573_573079


namespace maximize_sum_arithmetic_sequence_l573_573839

variable (a : ℕ → ℝ) (d : ℝ) (n : ℕ)

-- Given:
-- ∀ n, a_(n+1) = a_n + d
-- d < 0 
-- a 1 ^ 2 = a 10 ^ 2
-- We need to prove that the sum of the first n terms is maximized for n = 5

theorem maximize_sum_arithmetic_sequence (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h_neg_d : d < 0) 
  (h_condition : (a 1) ^ 2 = (a 10) ^ 2) : 
  ∀ n, (0 < n) → (n < 6) → 
      ∑ i in finset.range n, a (i + 1) < ∑ i in finset.range 5, a (i + 1) :=
sorry

end maximize_sum_arithmetic_sequence_l573_573839


namespace floor_area_cannot_exceed_10_square_meters_l573_573598

theorem floor_area_cannot_exceed_10_square_meters
  (a b : ℝ)
  (h : 3 > 0)
  (floor_lt_wall1 : a * b < 3 * a)
  (floor_lt_wall2 : a * b < 3 * b) :
  a * b ≤ 9 :=
by
  -- This is where the proof would go
  sorry

end floor_area_cannot_exceed_10_square_meters_l573_573598


namespace simplify_fraction_l573_573999

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l573_573999


namespace triangle_area_zero_l573_573382

theorem triangle_area_zero
  (rA rB rC : ℝ)
  (hA : rA = 2)
  (hB : rB = 3)
  (hC : rC = 4)
  (angle_m : ℝ)
  (h_angle : angle_m = π / 6) -- 30 degrees in radians
  (tangent_points : ℕ → ℝ × ℝ)
  (tangent_point_A : tangent_points 1 = (-5, sqrt 3))
  (tangent_point_B : tangent_points 2 = (0, 3 * sqrt 3 / 2))
  (tangent_point_C : tangent_points 3 = (7, 2 * sqrt 3)) :
  let A := (-5 : ℝ, 0),
      B := (0 : ℝ, 0),
      C := (7 : ℝ, 0) in
  (1 / 2 : ℝ) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 0 :=
by
  sorry

end triangle_area_zero_l573_573382


namespace regular_tetrahedron_distance_sum_ratio_l573_573550

variable (a : ℝ) -- edge length of the tetrahedron
variable (E F : ℝ × ℝ × ℝ) -- points inside faces ABC and BCD respectively

-- distances from E to planes
variable (d1 d2 d3 : ℝ) -- distances from E to planes ABD, ACD, BCD
-- distances from F to planes
variable (e1 e2 e3 : ℝ) -- distances from F to planes ABC, ABD, ACD

-- distances from E to edges
variable (s1 s2 s3 : ℝ) -- distances from E to edges AB, BC, CA
-- distances from F to edges
variable (t1 t2 t3 : ℝ) -- distances from F to edges BC, CD, DB

def s1_sum : ℝ := d1 + d2 + d3
def s2_sum : ℝ := e1 + e2 + e3
def S1_sum : ℝ := s1 + s2 + s3
def S2_sum : ℝ := t1 + t2 + t3

theorem regular_tetrahedron_distance_sum_ratio 
(h₁ : d1 = d2) (h₂ : d2 = d3) (h₃ : e1 = e2) (h₄ : e2 = e3)
(h₅ : s1 = s2) (h₆ : s2 = s3) (h₇ : t1 = t2) (h₈ : t2 = t3)
(h₉ : ∃ (hf : ℝ), s1 + s2 + s3 = 2 * hf) :
  (s1_sum + s2_sum) / (S1_sum + S2_sum) = sqrt 2 := 
by
  sorry

end regular_tetrahedron_distance_sum_ratio_l573_573550


namespace train_speed_l573_573716

theorem train_speed (time_sec : ℝ) (distance_m : ℝ) (h_time : time_sec = 15) (h_distance : distance_m = 250) : 
  let time_hr := time_sec / 3600
      distance_km := distance_m / 1000
  in (distance_km / time_hr = 60) :=
by
  let time_hr := time_sec / 3600
  let distance_km := distance_m / 1000
  show distance_km / time_hr = 60
  sorry

end train_speed_l573_573716


namespace total_area_ratio_l573_573251

theorem total_area_ratio (s : ℝ) (h : s > 0) :
  let small_area := (sqrt 3 / 4) * s^2
  let total_small_area := 3 * small_area
  let S := 3 * s
  let large_area := (sqrt 3 / 4) * S^2
  (total_small_area / large_area) = (1 / 3) :=
by
  sorry

end total_area_ratio_l573_573251


namespace greatest_common_divisor_of_three_divisors_l573_573290

theorem greatest_common_divisor_of_three_divisors (m : ℕ) (h1 : ∃ x, x ∣ 120 ∧ x ∣ m ∧ x > 0 ∧ (∀ d, d ∣ x → d = 1 ∨ d = 2 ∨ d = 4))
  : gcd 120 m = 4 :=
begin
  sorry,
end

end greatest_common_divisor_of_three_divisors_l573_573290


namespace inequality_proof_l573_573832

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
    (b^2 / a + a^2 / b) ≥ (a + b) := 
    sorry

end inequality_proof_l573_573832


namespace rod_length_of_weight_l573_573914

theorem rod_length_of_weight (w10 : ℝ) (wL : ℝ) (L : ℝ) (h1 : w10 = 23.4) (h2 : wL = 14.04) : L = 6 :=
by
  sorry

end rod_length_of_weight_l573_573914


namespace measure_of_angle_A_thm_perimeter_range_thm_l573_573128

-- Define the conditions for the problem
variables {A B C : ℝ} {a b c : ℝ}

-- Condition definitions
def triangle_condition_1 (a b c A C : ℝ) : Prop :=
  a / (sqrt 3 * (cos A)) = c / (sin C)

-- Answer calculations
def measure_of_angle_A (A : ℝ) : Prop :=
  A = π / 3

def perimeter_range (a : ℝ) : Prop :=
  12 < a + b + c ∧ a + b + c ≤ 18

-- Theorem statements
theorem measure_of_angle_A_thm (hc1 : triangle_condition_1 a b c A C) : measure_of_angle_A A :=
  sorry

theorem perimeter_range_thm (a_val : a = 6) : perimeter_range a :=
  sorry

end measure_of_angle_A_thm_perimeter_range_thm_l573_573128


namespace true_propositions_l573_573418

-- Definitions based on conditions
def replacement_interval (D : Set ℝ) (f g : ℝ → ℝ) : Prop :=
  ∀ x ∈ D, abs (f x - g x) ≤ 1

-- Proposition ① proof
def prop1 (x : ℝ) : Prop :=
  f1 x = x^2 + 1 ∧ g1 x = x^2 + 1/2 →
  replacement_interval set.univ f1 g1

-- Proposition ② proof
def interval_2 : Set ℝ := set.Icc (1/4) (3/2)

def prop2 (x : ℝ) : Prop :=
  f2 x = x ∧ g2 x = 1 - 1/(4*x) →
  replacement_interval interval_2 f2 g2

-- Proposition ③ proof
def interval_3 : Set ℝ := set.Icc 1 real.sqrt2

def prop3 (b : ℝ) : Prop :=
  (∀ x ∈ interval_3, abs (real.log x - (1/x - b)) ≤ 1) →
  0 ≤ b ∧ b ≤ 1/exp 1

theorem true_propositions :
  (∀ x, prop1 x) ∧ (∀ x, prop2 x) ∧ (∀ b, prop3 b) := 
by {
  sorry,
}

end true_propositions_l573_573418


namespace trace_of_C_in_interval_2017_2018_l573_573610

theorem trace_of_C_in_interval_2017_2018 (x : ℝ) (h : 2017 ≤ x ∧ x ≤ 2018) :
    f(x) = real.sqrt(1 + 2*(x - 2016) - (x - 2016)^2) := sorry

end trace_of_C_in_interval_2017_2018_l573_573610


namespace solution_set_abs_inequality_l573_573263

theorem solution_set_abs_inequality (x : ℝ) :
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  sorry

end solution_set_abs_inequality_l573_573263


namespace problem1_proof_problem2_proof_l573_573659

noncomputable def problem1 := ¬ ∃ (x y : ℝ),
  (2^(x) + 2^(x+1) + 2^(x+2) + 2^(x+3) = 4^(y) + 4^(y+1) + 4^(y+2) + 4^(y+3)) ∧
  (3^(x) + 3^(x+1) + 3^(x+2) + 3^(x+3) = 9^(y) + 9^(y+1) + 9^(y+2) + 9^(y+3))

noncomputable def problem2 : Prop :=
  ∃ (x y : ℝ),
    (x ≈ -2.323) ∧ (y ≈ -2.536) ∧
    (2^(x) + 2^(x+1) + 2^(x+2) + 2^(x+3) = 8^(y) + 8^(y+1) + 8^(y+2) + 8^(y+3))

theorem problem1_proof : problem1 := sorry
theorem problem2_proof : problem2 := sorry

end problem1_proof_problem2_proof_l573_573659


namespace count_nonzero_complex_numbers_forming_square_l573_573481

noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

theorem count_nonzero_complex_numbers_forming_square : 
  (∃ z : ℂ, z ≠ 0 ∧ 
          ∃ θ : ℝ, z = cis θ ∧ 
          ((4 * θ - θ) ≡ 90 [MOD 360] ∨ (4 * θ - θ) ≡ 270 [MOD 360])) → 
  ∃! n, n = 4 :=
by sorry

end count_nonzero_complex_numbers_forming_square_l573_573481


namespace point_in_circle_count_l573_573419

theorem point_in_circle_count :
  let center := (3, 3)
  let radius := 5
  ∃ (count: ℕ), 
    count = (finset.card (finset.filter (λ x: ℤ, (x - 3)^2 + ((-x) - 3)^2 ≤ radius^2) (finset.Icc (-2) 2))) ∧
    count = 3 := 
by
  sorry

end point_in_circle_count_l573_573419


namespace simplify_fraction_48_72_l573_573997

theorem simplify_fraction_48_72 : (48 : ℚ) / 72 = 2 / 3 := sorry

end simplify_fraction_48_72_l573_573997


namespace find_acute_angle_parallel_vectors_l573_573825

-- Step c): Create definitions for the problem conditions
def vector_a (α : ℝ) : ℝ × ℝ := (3, Real.sin α)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.cos α)

-- Define the parallel condition for two vectors
def are_parallel {α β : ℝ × ℝ} : Prop :=
  ∀ t : ℝ, vector_a t = (vector_b t)

-- Define the acute angle condition
def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- State the theorem
theorem find_acute_angle_parallel_vectors (α : ℝ) 
  (h_parallel : are_parallel α) 
  (h_acute : is_acute_angle α) : α = π / 3 :=
sorry

end find_acute_angle_parallel_vectors_l573_573825


namespace trapezoid_FG_squared_l573_573157

theorem trapezoid_FG_squared (EF GH EH FG FH EG : ℝ) 
  (h1 : EF = 3)
  (h2 : EH = Real.sqrt 2001)
  (h3 : FG ^ 2 + EF ^ 2 = EH ^ 2)
  (h4 : GH = Real.sqrt (3 * FG ^ 2 + (FG + Real.sqrt (EH ^ 2 - GH ^ 2 - FG ^ 2)))))
  (h5 : EG * FH = 0)
  (h6 : GH = FG ^ 2 / 3) :
  FG ^ 2 = (9 + 3 * Real.sqrt 7977) / 2 :=
by
  sorry

end trapezoid_FG_squared_l573_573157


namespace measure_of_angle_BCD_l573_573151

-- Define angles and sides as given in the problem
variables (α β : ℝ)

-- Conditions: angles and side equalities
axiom angle_ABD_eq_BDC : α = β
axiom angle_DAB_eq_80 : α = 80
axiom side_AB_eq_AD : ∀ AB AD : ℝ, AB = AD
axiom side_DB_eq_DC : ∀ DB DC : ℝ, DB = DC

-- Prove that the measure of angle BCD is 65 degrees
theorem measure_of_angle_BCD : β = 65 :=
sorry

end measure_of_angle_BCD_l573_573151


namespace graph_transformation_point_l573_573462

theorem graph_transformation_point {f : ℝ → ℝ} (h : f 1 = 0) : f (0 + 1) + 1 = 1 :=
by
  sorry

end graph_transformation_point_l573_573462


namespace find_ellipse_eq_existence_of_B_l573_573842

-- Define the ellipse C with semi-major axis a and semi-minor axis b
variables (a b : ℝ) 
variables (a_gt_b : a > b) (b_gt_0 : b > 0) 

-- The given point on the ellipse
variables (P : ℝ × ℝ)
variables (P_on_ellipse : P = (sqrt 2, 1)) 
variables (ellipse_spec_a : a > b > 0)
variables (ellipse_def : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1)

-- Point A through which line l passes
variables (A : ℝ × ℝ) (A_def : A = (0, 1))

-- Left focus of the ellipse
variables (c : ℝ) 
variables (c_def : c = sqrt (a ^ 2 - b ^ 2)) 
variables (c_val : c = sqrt 2)

-- Slope of line l when passing through left focus
variables (slope_f_left : ℝ) 
variables (slope_f_left_def : slope_f_left = sqrt 2 / 2)

-- Theorem (1): Finding the equation of the ellipse
theorem find_ellipse_eq : 
  ∃ a b : ℝ, a > b > 0 ∧ (sqrt 2) ^ 2 / a ^ 2 + 1 ^ 2 / b ^ 2 = 1 ∧ a ^ 2 - b ^ 2 = 2 ∧ c = sqrt 2 ∧ c = sqrt (a ^ 2 - b ^ 2) ∧ (4 : ℚ) = a ^ 2 ∧ (2 : ℚ) = b ^ 2 :=
sorry

-- Theorem (2): Existence of another fixed point B
theorem existence_of_B :
  ¬ ∃ B : ℝ × ℝ, B ≠ A ∧ (∀ M N : ℝ × ℝ, on_line l A M ∧ on_line l A N ∧ M ≠ N → ∠ ABM = ∠ ABN) :=
sorry

end find_ellipse_eq_existence_of_B_l573_573842


namespace find_symmetric_line_l573_573910

theorem find_symmetric_line :
  (∀ x y : ℝ, x^2 + y^2 = 1 → x^2 + y^2 + 4 * x - 4 * y + 7 = 0 → (x - y + 2 = 0)) := 
begin
  sorry,
end

end find_symmetric_line_l573_573910


namespace cut_50cm_without_measuring_tools_l573_573479

/-

Given a string that is \( \frac{2}{3} \) meters long, prove that you can cut 50 cm from it
without any measuring tools resulting in exactly 50 cm remaining.

-/

theorem cut_50cm_without_measuring_tools (l : ℝ) (h : l = 2 / 3 * 100) : 
  ∃ p, p = 50 :=
by
  -- Convert the given length in meters to centimeters
  have h_cm : l = 66.6667 := by sorry
  -- Find the fraction p such that cutting p * l cm leaves 50 cm
  have h_frac : p = 0.75 := by sorry
  -- Show that cutting 0.75 * 66.6667 cm leaves exactly 50 cm
  existsi p
  -- Conclusion: verify that after cutting 0.75 * l, 50 cm remains
  sorry

end cut_50cm_without_measuring_tools_l573_573479


namespace first_sequence_general_term_second_sequence_general_term_l573_573042

-- For the first sequence
def first_sequence_sum : ℕ → ℚ
| n => n^2 + 1/2 * n

theorem first_sequence_general_term (n : ℕ) : 
  (first_sequence_sum (n+1) - first_sequence_sum n) = (2 * (n+1) - 1/2) := 
sorry

-- For the second sequence
def second_sequence_sum : ℕ → ℚ
| n => 1/4 * n^2 + 2/3 * n + 3

theorem second_sequence_general_term (n : ℕ) : 
  (second_sequence_sum (n+1) - second_sequence_sum n) = 
  if n = 0 then 47/12 
  else (6 * (n+1) + 5)/12 := 
sorry

end first_sequence_general_term_second_sequence_general_term_l573_573042


namespace sum_of_a_l573_573039

def a (k : ℕ) : ℕ :=
  Nat.trailingZeroes k

theorem sum_of_a (n : ℕ) (hn : n > 0) :
  (∑ k in Finset.range (2^n + 1), a k) = 2^n - 1 :=
by
  sorry

end sum_of_a_l573_573039


namespace min_value_expression_l573_573439

theorem min_value_expression (cards : Multiset ℕ)
  (h : cards = {5, 5, 6, 6, 6, 7, 8, 8, 9}) :
  ∃ A B C : ℕ, 
    (∃ a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ, 
      Multiset.of_list [a1, a2, a3, b1, b2, b3, c1, c2, c3] = cards ∧
      a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 ∧ 
      b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3 ∧ 
      c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 ∧
      A = a1 * 100 + a2 * 10 + a3 ∧ 
      B = b1 * 100 + b2 * 10 + b3 ∧ 
      C = c1 * 100 + c2 * 10 + c3) ∧ 
  A + B - C = 149 :=
begin
  sorry
end

end min_value_expression_l573_573439


namespace initial_percentage_glycerin_l573_573709

theorem initial_percentage_glycerin
    (initial_volume : ℝ)
    (added_water : ℝ)
    (final_percentage : ℝ)
    (P : ℝ) : 
    initial_volume = 4 →
    added_water = 0.8 →
    final_percentage = 0.75 →
    4 * P = 3.6 →
    P = 0.9 :=
begin
    intros h_initial_volume h_added_water h_final_percentage h_equation,
    have eq1 : initial_volume * P = final_percentage * (initial_volume + added_water),
    { rw [h_initial_volume, h_added_water, h_final_percentage],
      ring_nf, },
    rw h_equation at eq1,
    have hP : P = 0.9,
    { exact eq1.symm },
    exact hP,
end

end initial_percentage_glycerin_l573_573709


namespace cos_angle_A_eq_0_l573_573920

noncomputable theory

variables {A C : ℝ} {AB CD AD BC: ℝ}
def quadrilateral_ABCD (angle_A_cong_angle_C : A = C) 
    (AB_eq_CD : AB = 200) 
    (AD_ne_BC : AD ≠ BC) 
    (perimeter_eq_720 : AB + BC + CD + AD = 720) : ℝ :=
  floor (1000 * 0.7) = 700  

theorem cos_angle_A_eq_0.7 : 
  quadrilateral_ABCD (A = C) (AB = 200) (AD ≠ BC) (AB + BC + CD + AD = 720) :=
  by 
    sorry

end cos_angle_A_eq_0_l573_573920


namespace f_of_72_l573_573448

theorem f_of_72 (f : ℕ → ℝ) (p q : ℝ) (h1 : ∀ a b : ℕ, f (a * b) = f a + f b)
  (h2 : f 2 = p) (h3 : f 3 = q) : f 72 = 3 * p + 2 * q := 
sorry

end f_of_72_l573_573448


namespace count_positive_integers_satisfying_inequality_l573_573793

theorem count_positive_integers_satisfying_inequality :
  let S := {n : ℕ | n > 0 ∧ (n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0}
  ∃ n, S.card = 24 :=
by sorry

end count_positive_integers_satisfying_inequality_l573_573793


namespace intersection_correct_l573_573851

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_correct : A ∩ B = {1, 2} := 
sorry

end intersection_correct_l573_573851


namespace property_holds_for_n_l573_573455

noncomputable def holds_for_n (n : ℕ) (x : Fin n → ℝ) : Prop :=
  Σ (∀ (i : Fin n), ∑ x = 0) ∧
  ∃ (3 ≤ n → x (0) * x (1) + x (1) * x (2) + x (2) * x (0) ≤ 0) 

theorem property_holds_for_n :
  ∀ (n : ℕ) (x : Fin n → ℝ),
  holds_for_n n x → (if n = 3 then x 0 * x 1 + x 1 * x 2 + x 2 * x 0 ≤ 0 else 
    if n = 4 then x 0 * x 1 + x 1 * x 2 + x 2 * x 3 + x 3 * x 0 ≤ 0 else 
    ∃ x (Fin 5) → x 0 + x 1 + x 2 + x 3 + x 4 = 0 → x 0 * x 1 + x 1 * x (Fin 5)  + x (Fin 5) * x (Fin 5) > 0) := sorry

end property_holds_for_n_l573_573455


namespace right_triangle_CD_length_l573_573956

theorem right_triangle_CD_length
  (A B C D : Type)
  (h_triangle : triangle ABC)
  (h_right_angle : angle B = pi / 2)
  (h_circle : circle (BC / 2))
  (h_diameter : diameter BC)
  (h_AC_intersect : segment AC ∩ circle_points = D)
  (h_AD : AD = 1)
  (h_BD : BD = 4) :
  CD = 16 :=
sorry

end right_triangle_CD_length_l573_573956


namespace angle_DFE_is_70_l573_573931

theorem angle_DFE_is_70
  (A B C D E F : Point)
  (ABCD_square : is_square A B C D)
  (E_on_extension_of_CD : lies_on_extension E C D)
  (angle_CDE_140 : angle C D E = 140)
  (F_on_AB : lies_on F A B)
  (DE_eq_DF : dist D E = dist D F) :
  angle D F E = 70 := by
  sorry

end angle_DFE_is_70_l573_573931


namespace balance_scale_l573_573622

theorem balance_scale :
  (∃ (weights : list ℝ), weights = list.map real.log (list.range' 3 77) ∧
    (∀ (distribute : list ℝ → list (list ℝ) × list (list ℝ)),
      (∀ l r, distribute weights = (l, r) → |(l.sum - r.sum)| ≤ 1) 
    )
  ) :=
sorry

end balance_scale_l573_573622


namespace array_product_constant_l573_573225

open Real

def r := [2, 3, 5, 8, 9, 10]
def c := [2, 3, 5, 7, 11, 13]

def Element (i j : Nat) : Real :=
  r.getD i 0 * c.getD j 0

theorem array_product_constant :
  (∏ i in Finset.range 6, r[i]) * (∏ j in Finset.range 6, c[j]) = 648648000 := by
  sorry

end array_product_constant_l573_573225


namespace complement_union_l573_573198

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union : 
  U = {0, 1, 2, 3, 4} →
  (U \ A = {1, 2}) →
  B = {1, 3} →
  (A ∪ B = {0, 1, 3, 4}) :=
by
  intros hU hA hB
  sorry

end complement_union_l573_573198


namespace minimize_cost_at_4_l573_573677

-- Given definitions and conditions
def surface_area : ℝ := 12
def max_side_length : ℝ := 5
def front_face_cost_per_sqm : ℝ := 400
def sides_cost_per_sqm : ℝ := 150
def roof_ground_cost : ℝ := 5800
def wall_height : ℝ := 3

-- Definition of the total cost function
noncomputable def total_cost (x : ℝ) : ℝ :=
  900 * (x + 16 / x) + 5800

-- The main theorem to be proven
theorem minimize_cost_at_4 (h : 0 < x ∧ x ≤ max_side_length) : 
  (∀ x, total_cost x ≥ total_cost 4) ∧ total_cost 4 = 13000 :=
sorry

end minimize_cost_at_4_l573_573677


namespace root_of_sqrt_eq_l573_573410

theorem root_of_sqrt_eq : 
  ∃ x : ℝ, sqrt x + sqrt (x + 6) = 12 ∧ x = 529 / 16 := 
by
  use 529 / 16
  split
  {  -- the first part is to show the equation holds
    have H1 : sqrt (529 / 16) + sqrt ((529 / 16) + 6) = 12,
    sorry  -- skipped proof for simplification
  }
  {  -- the second part is to confirm the value of x
    refl
  }

end root_of_sqrt_eq_l573_573410


namespace acute_isosceles_inscribed_in_circle_l573_573360

noncomputable def solve_problem : ℝ := by
  -- Let x be the angle BAC
  let x : ℝ := π * 5 / 11
  -- Considering the value of k in the problem statement
  let k : ℝ := 5 / 11
  -- Providing the value of k obtained from solving the problem
  exact k

theorem acute_isosceles_inscribed_in_circle (ABC : Type)
  [inhabited ABC]
  (inscribed : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (B_tangent C_tangent : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (D : ABC)
  (angle_eq : ∀ {A B C : ABC}, ∠ABC = ∠ACB)
  (triple_angle : ∀ {A B C D : ABC}, ∠ABC = 3 * ∠BAC) :
  solve_problem = 5 / 11 := 
sorry

end acute_isosceles_inscribed_in_circle_l573_573360


namespace parallel_min_value_non_obtuse_angle_l573_573107

variables {α : Type*} [linear_ordered_field α]

def m_vector (a b : α) : α × α :=
  (a, b^2 - b + (7 / 3))

def n_vector (a b : α) : α × α :=
  (a + b + 2, 1)

def mu_vector : α × α :=
  (2, 1)

theorem parallel_min_value (a b : α) (h : m_vector a b = (2 * (b^2 - b + 7 / 3), _)) :
  a = 25 / 6 := sorry

theorem non_obtuse_angle (a b : α) :
  let m := m_vector a b in
  let n := n_vector a b in
  m.1 * n.1 + m.2 * n.2 ≥ 0 := sorry

end parallel_min_value_non_obtuse_angle_l573_573107


namespace arithmetic_sequence_a3_l573_573063

variable {α : Type*} [add_comm_group α] [module ℝ α]

/-- Given an arithmetic sequence {a_n} whose sum of the first n terms is S_n.
    The sequence has the property that S4 - S1 = 3. Then a3 is 1. -/
theorem arithmetic_sequence_a3 (a : ℕ → α) (S : ℕ → α)
  (a_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (sum_arith : ∀ n : ℕ, S n = (n.succ * a 0 + n * (n - 1) / 2 * (a 1 - a 0)))
  (h : S 4 - S 1 = 3) :
  a 3 = 1 := 
sorry

end arithmetic_sequence_a3_l573_573063


namespace circle_equation_l573_573868

theorem circle_equation :
  ∃ r : ℝ, ∀ x y : ℝ,
  ((x - 2) * (x - 2) + (y - 1) * (y - 1) = r * r) ∧
  ((5 - 2) * (5 - 2) + (-2 - 1) * (-2 - 1) = r * r) ∧
  (5 + 2 * -2 - 5 + r * r = 0) :=
sorry

end circle_equation_l573_573868


namespace lcm_of_ratio_4_5_l573_573630

def ratio (a b : ℕ) (r : ℕ × ℕ) : Prop := a * r.2 = b * r.1

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem lcm_of_ratio_4_5 (a b : ℕ) (h_ratio : ratio a b (4, 5)) (h_a : a = 36) : lcm a b = 1620 := by
  sorry

end lcm_of_ratio_4_5_l573_573630


namespace total_boys_across_grades_is_692_l573_573726

theorem total_boys_across_grades_is_692 (ga_girls gb_girls gc_girls : ℕ) (ga_boys : ℕ) :
  ga_girls = 256 →
  ga_girls = ga_boys + 52 →
  gb_girls = 360 →
  gb_boys = gb_girls - 40 →
  gc_girls = 168 →
  gc_girls = gc_boys →
  ga_boys + gb_boys + gc_boys = 692 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_boys_across_grades_is_692_l573_573726


namespace conditional_probability_l573_573494

-- Define the number of products and their classes
def total_products := 4
def first_class_products := 3
def second_class_products := 1

-- Define events A and B
def event_A := "the first product drawn is a first-class product"
def event_B := "the second product drawn is a first-class product"

-- Define the required conditional probability
def P_A : ℚ := 3 / 4
def P_AB : ℚ := 1 / 2
def P_B_given_A : ℚ := 2 / 3

theorem conditional_probability (h : P(B | A) = P_AB / P_A := 
    P(B | A) = 2 / 3: P(B | A | event_A ∧ event_B) = 
by
    sorry

end conditional_probability_l573_573494


namespace max_value_fraction_l573_573083

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) <= (2 / 3) := 
sorry

end max_value_fraction_l573_573083


namespace constant_expression_l573_573954

variable {R a b : ℝ}

-- Define the conditions
variable (ABC : Type*) (H P : ABC)
variable [IsoscelesTriangle ABC]
variable [Orthocenter H ABC]
variable [Circumcircle P ABC]

theorem constant_expression (P : ABC) (R a b : ℝ)                       -- Given variables and types
  (ht_iso: IsIsoscelesTriangle ABC b)                                  -- Condition 1: Isosceles triangle with sides AB = AC = b
  (ht_h: IsOrthocenter H ABC)                                          -- Condition 2: H is the orthocenter
  (hp_circumcircle: IsOnCircumcircle P ABC R)                          -- Condition 3: P is on the circumcircle with radius R
  : PA P A ^ 2 + PB P B ^ 2 - PH P H ^ 2 = R ^ 2 - a ^ 2 :=            -- Statement: this expression is equal to R^2 - a^2
sorry

end constant_expression_l573_573954


namespace pairs_sum_non_negative_l573_573265

theorem pairs_sum_non_negative {a : Fin 100 → ℝ} (h_sum : ∑ i, a i = 0) :
  ∃ s : Finset (Fin 100 × Fin 100), s.card ≥ 99 ∧ (∀ p ∈ s, a p.1 + a p.2 ≥ 0) :=
by
  sorry

end pairs_sum_non_negative_l573_573265


namespace polynomial_no_linear_term_l573_573627

theorem polynomial_no_linear_term (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x - n) = x^2 + mn → n + m = 0) :=
sorry

end polynomial_no_linear_term_l573_573627


namespace find_tunnel_length_l573_573346

variable (length_of_train speed_of_train total_time : ℝ)
variable (crossing_time : ℤ)
variable (speed_conversion_factor : ℝ)
variable (time_conversion_factor : ℝ)

-- Conditions from step a:
def length_of_train := 800
def speed_of_train := 78 / 3.6 -- converts km/hr to m/s
def total_time := 60 -- seconds
def speed_conversion_factor := 1000 / 3600 -- conversion factor from km/hr to m/s
def time_conversion_factor := 60 -- conversion factor from minutes to seconds

-- Define the length of the tunnel
noncomputable def length_of_tunnel : ℝ :=
  let total_distance := (speed_of_train * total_time : ℝ)
  total_distance - length_of_train

theorem find_tunnel_length (length_of_train = 800) (speed_of_train = 78 / 3.6) (total_time = 60) : 
  length_of_tunnel = 500.2 := 
by
  intros
  sorry

end find_tunnel_length_l573_573346


namespace find_m_l573_573260

-- Mathematical definitions from the given conditions
def condition1 (m : ℝ) : Prop := m^2 - 2 * m - 2 = 1
def condition2 (m : ℝ) : Prop := m + 1/2 * m^2 > 0

-- The proof problem summary
theorem find_m (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = 3 :=
by
  sorry

end find_m_l573_573260


namespace population_growth_l573_573618

theorem population_growth (P : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : P = 5.48) 
  (h₂ : y = P * (1 + x / 100)^8) : 
  y = 5.48 * (1 + x / 100)^8 := 
by
  sorry

end population_growth_l573_573618


namespace num_positive_integers_satisfying_l573_573801

theorem num_positive_integers_satisfying (n : ℕ) :
  (∑ k in (finset.range 25), (if (even (2 + 4 * k)) then 1 else 0) = 24) :=
sorry

end num_positive_integers_satisfying_l573_573801


namespace percentage_y_less_than_x_l573_573331

variable (x y : ℝ)

-- given condition
axiom hyp : x = 11 * y

-- proof problem: Prove that the percentage y is less than x is (10/11) * 100
theorem percentage_y_less_than_x (x y : ℝ) (hyp : x = 11 * y) : 
  (x - y) / x * 100 = (10 / 11) * 100 :=
by
  sorry

end percentage_y_less_than_x_l573_573331


namespace alice_bob_sum_is_42_l573_573578

theorem alice_bob_sum_is_42 :
  ∃ (A B : ℕ), 
    (1 ≤ A ∧ A ≤ 60) ∧ 
    (1 ≤ B ∧ B ≤ 60) ∧ 
    Nat.Prime B ∧ B > 10 ∧ 
    (∀ n : ℕ, n < 5 → (A + B) % n ≠ 0) ∧ 
    ∃ k : ℕ, 150 * B + A = k * k ∧ 
    A + B = 42 :=
by 
  sorry

end alice_bob_sum_is_42_l573_573578


namespace sqrt_0_54_in_terms_of_a_b_l573_573955

variable (a b : ℝ)

-- Conditions
def sqrt_two_eq_a : Prop := a = Real.sqrt 2
def sqrt_three_eq_b : Prop := b = Real.sqrt 3

-- The main statement to prove
theorem sqrt_0_54_in_terms_of_a_b (h1 : sqrt_two_eq_a a) (h2 : sqrt_three_eq_b b) :
  Real.sqrt 0.54 = 0.3 * a * b := sorry

end sqrt_0_54_in_terms_of_a_b_l573_573955


namespace calculate_ggg1_l573_573180

def g (x : ℕ) : ℕ := 7 * x + 3

theorem calculate_ggg1 : g (g (g 1)) = 514 := 
by
  sorry

end calculate_ggg1_l573_573180


namespace probability_event_A_probability_event_B_probability_event_C_l573_573275

-- Define the total number of basic events for three dice
def total_basic_events : ℕ := 6 * 6 * 6

-- Define events and their associated basic events
def event_A_basic_events : ℕ := 2 * 3 * 3
def event_B_basic_events : ℕ := 2 * 3 * 6
def event_C_basic_events : ℕ := 6 * 6 * 3

-- Define probabilities for each event
def P_A : ℚ := event_A_basic_events / total_basic_events
def P_B : ℚ := event_B_basic_events / total_basic_events
def P_C : ℚ := event_C_basic_events / total_basic_events

-- Statement to be proven
theorem probability_event_A : P_A = 1 / 12 := by
  sorry

theorem probability_event_B : P_B = 1 / 6 := by
  sorry

theorem probability_event_C : P_C = 1 / 2 := by
  sorry

end probability_event_A_probability_event_B_probability_event_C_l573_573275


namespace locus_equation_of_points_distance_2_l573_573012

-- Define the distance between two parallel lines
def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  (abs (c2 - c1)) / (sqrt (a^2 + b^2))

-- The given line and the distance
def line1 : ℝ → ℝ → ℝ := λ x y, 3 * x - 4 * y - 1
def d : ℝ := 2

-- Given the lines' equations, we need to find c such that the distance is 2
theorem locus_equation_of_points_distance_2 (x y : ℝ) :
  (3 * x - 4 * y - 11 = 0 ∨ 3 * x - 4 * y + 9 = 0) ↔
  distance_between_parallel_lines 3 (-4) (-1) (-c) = 2 :=
by
  sorry

end locus_equation_of_points_distance_2_l573_573012


namespace same_color_rectangle_exists_l573_573924

-- Define the structure for strips in the coordinate plane
def strip (n : ℤ) : set (ℝ × ℝ) := {p : ℝ × ℝ | n ≤ p.fst ∧ p.fst < n + 1}

-- The main theorem to prove
theorem same_color_rectangle_exists (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (color : ℤ → bool) : 
  ∃ (A B C D : ℝ × ℝ), (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) ∧ 
  (strip (⌊A.fst⌋) = strip (⌊B.fst⌋) ∧ strip (⌊B.fst⌋) = strip (⌊C.fst⌋) ∧ strip (⌊C.fst⌋) = strip (⌊D.fst⌋)) ∧ 
  (color ⌊A.fst⌋ = color ⌊B.fst⌋ ∧ color ⌊B.fst⌋ = color ⌊C.fst⌋ ∧ color ⌊C.fst⌋ = color ⌊D.fst⌋) ∧
  ((A.snd = B.snd ∧ B.snd = C.snd ∧ C.snd = D.snd) ∨ (A.fst = D.fst ∧ B.fst = C.fst) ∧
  (abs (B.fst - A.fst) = a ∧ abs (D.snd - A.snd) = b)) :=
sorry

end same_color_rectangle_exists_l573_573924


namespace boat_speed_l573_573699

theorem boat_speed (v : ℝ) :
  (AvgSpeed : ℝ) := 5.090909090909091 →
  (ReturnSpeed : ℝ) := 7 →
  5.090909090909091 = ((2 * v * 7) / (v + 7)) →
  v = 4 :=
by
  sorry

end boat_speed_l573_573699


namespace menelaus_theorem_ceva_theorem_l573_573660

variables {A B C A₁ B₁ C₁ : Point}

-- Define the lengths as ratios of directed segments
def directed_segment_length (P Q R: Point) : ℝ := dist P Q / dist P R

-- Menelaus' theorem statement
theorem menelaus_theorem 
    (hA₁ : Collinear {A, A₁, B})
    (hB₁ : Collinear {B, B₁, C})
    (hC₁ : Collinear {C, C₁, A}) :
    Collinear {A₁, B₁, C₁} ↔
    directed_segment_length A₁ B C * 
    directed_segment_length B₁ C A * 
    directed_segment_length C₁ A B = 1 := 
sorry

-- Ceva's theorem statement
theorem ceva_theorem
    (hNotParallel : ¬(Parallel (LineThrough A A₁) (LineThrough B B₁) ∧
                      Parallel (LineThrough A A₁) (LineThrough C C₁) ∧
                      Parallel (LineThrough B B₁) (LineThrough C C₁))) :
    Concurrent {LineThrough A A₁, LineThrough B B₁, LineThrough C C₁} ↔
    directed_segment_length A₁ B C * 
    directed_segment_length B₁ C A * 
    directed_segment_length C₁ A B = -1 := 
sorry

end menelaus_theorem_ceva_theorem_l573_573660


namespace sum_of_frac_parts_l573_573187

open Int

def fractional_part (x : ℚ) : ℚ :=
  x - x.floor

theorem sum_of_frac_parts (p : ℕ) (hp : p > 1) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) :
  ∑ k in Finset.range (p - 1), fractional_part (↑(k^2) / p) = (p - 1) / 4 := by
  sorry

end sum_of_frac_parts_l573_573187


namespace sum_of_excluded_values_l573_573758

theorem sum_of_excluded_values : 
  (∑ r in ({ x | 3 * x ^ 2 - 9 * x + 6 = 0 }) (fun x => x) = 3) :=
begin
  sorry
end

end sum_of_excluded_values_l573_573758


namespace find_baseball_deck_price_l573_573981

variables (numberOfBasketballPacks : ℕ) (pricePerBasketballPack : ℝ) (numberOfBaseballDecks : ℕ)
           (totalMoney : ℝ) (changeReceived : ℝ) (totalSpent : ℝ) (spentOnBasketball : ℝ) (baseballDeckPrice : ℝ)

noncomputable def problem_conditions : Prop :=
  numberOfBasketballPacks = 2 ∧
  pricePerBasketballPack = 3 ∧
  numberOfBaseballDecks = 5 ∧
  totalMoney = 50 ∧
  changeReceived = 24 ∧
  totalSpent = totalMoney - changeReceived ∧
  spentOnBasketball = numberOfBasketballPacks * pricePerBasketballPack ∧
  totalSpent = spentOnBasketball + (numberOfBaseballDecks * baseballDeckPrice)

theorem find_baseball_deck_price (h : problem_conditions numberOfBasketballPacks pricePerBasketballPack numberOfBaseballDecks totalMoney changeReceived totalSpent spentOnBasketball baseballDeckPrice) :
  baseballDeckPrice = 4 :=
sorry

end find_baseball_deck_price_l573_573981


namespace total_cost_l573_573722

variable (E P M : ℝ)

axiom condition1 : E + 3 * P + 2 * M = 240
axiom condition2 : 2 * E + 5 * P + 4 * M = 440

theorem total_cost : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_l573_573722


namespace exists_subset_B_l573_573963

theorem exists_subset_B (Y : Set ℕ) (n : ℕ) (hY_len : Y.card = n) (hY_subset : Y ⊆ { k : ℕ | 0 < k }) : 
  ∃ (B : Set ℕ), B ⊆ Y ∧ B.card > n / 3 ∧ ∀ (u v : ℕ), u ∈ B → v ∈ B → u + v ∉ B := 
sorry

end exists_subset_B_l573_573963


namespace lineup_restriction_count_l573_573503

theorem lineup_restriction_count :
  (∃ (people : Fin 5 → Prop), ∀ (first : Fin 5), 
    first ∉ ({0, 1} : Finset (Fin 5)) →
    ∏ (i : Fin (5 - 1)), (5 - i : ℕ) = 72) := sorry

end lineup_restriction_count_l573_573503


namespace greatest_common_divisor_three_divisors_l573_573286

theorem greatest_common_divisor_three_divisors (m : ℕ) (h : ∃ (D : set ℕ), D = {d | d ∣ 120 ∧ d ∣ m} ∧ D.card = 3) : 
  ∃ p : ℕ, p.prime ∧ greatest_dvd_set {d | d ∣ 120 ∧ d ∣ m} = p^2 := 
sorry

end greatest_common_divisor_three_divisors_l573_573286


namespace propositions_correctness_l573_573356

noncomputable def correctPropositions : set ℕ :=
  {1, 4, 5}

theorem propositions_correctness :
  (∀ x : ℝ, ∀ x_increase : ℝ, y_increase : ℝ,
    (λ (x : ℝ), 3 + 2 * x) (x + x_increase) - (λ (x : ℝ), 3 + 2 * x) x = y_increase
    ↔ x_increase = 2 ∧ y_increase = 4) ∧
  (∀ r : ℝ, (r ≠ 0) →
    (|r| > 1 → false) ∧ (|r| < 1 → true)) ∧
  (∀ ξ : ℝ, ∀ σ : ℝ, σ > 0 →
    (∫ ξ in set.interval 0 (2 * σ), NormalDist 0 σ ξ) = 0.5 →
    (∫ ξ in set.interval (-2 * σ) 0, NormalDist 0 σ ξ) = 0.4 →
    (∫ ξ in set.interval 2 (∞), NormalDist 0 σ ξ) = 0.1) ∧
  (∫ x in 0..π, sin x dx = 2) ∧
  (∀ n : ℤ, (n ≠ 4) →
    (n / (n-4) + (8-n) / ((8-n)-4) = 2)) →
  correctPropositions = {1, 4, 5} := 
begin
  sorry
end

end propositions_correctness_l573_573356


namespace sum_lent_1840point62_l573_573335

theorem sum_lent_1840point62 :
  ∃ P : ℝ, 
    let r := 0.06 in
    let t := 10 in
    let I := P - 385 in
    let A := P * (1 + r) ^ t in
    A - P = I ∧ P = 1840.62 := 
by
  sorry

end sum_lent_1840point62_l573_573335


namespace Masha_gathers_5_mushrooms_l573_573046

def mushrooms_collected (B G : list ℕ) : ℕ :=
  B.sum + G.sum

def unique_girls (G : list ℕ) : Prop :=
  ∀ i j, i ≠ j → G.get_or_else i 0 ≠ G.get_or_else j 0

def at_least_43_mushrooms (B : list ℕ) : Prop :=
  ∀ i j k, B.get_or_else i 0 + B.get_or_else j 0 + B.get_or_else k 0 ≥ 43 

def within_5_times (A : list ℕ) : Prop :=
  ∀ i j, (max (A.get_or_else i 0) (A.get_or_else j 0)) ≤ 5 * (min (A.get_or_else i 0) (A.get_or_else j 0))

noncomputable def Masha_collects_most_mushrooms (G : list ℕ) : ℕ :=
  G.maximum_def 0

theorem Masha_gathers_5_mushrooms (B G : list ℕ) (h_size_B : B.length = 4) (h_size_G : G.length = 3) 
  (h_total : mushrooms_collected B G = 70) 
  (h_unique : unique_girls G) 
  (h_at_least_43 : at_least_43_mushrooms B) 
  (h_within_5times : within_5_times (B ++ G)) : 
  Masha_collects_most_mushrooms G = 5 := 
sorry

end Masha_gathers_5_mushrooms_l573_573046


namespace count_positive_integers_satisfying_product_inequality_l573_573803

theorem count_positive_integers_satisfying_product_inequality :
  ∃ (k : ℕ), k = 23 ∧ 
  (n : ℕ ) → (2 ≤ n ∧ n < 100) →
  ((∃ (m: ℕ), 2 + 4 * m = n) ∧ ((n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0) = 23 :=
by
  sorry

end count_positive_integers_satisfying_product_inequality_l573_573803


namespace g_symmetry_solutions_l573_573544

noncomputable def g : ℝ → ℝ := sorry

theorem g_symmetry_solutions (g_def: ∀ (x : ℝ), x ≠ 0 → g x + 3 * g (1 / x) = 6 * x^2) :
  ∀ (x : ℝ), g x = g (-x) → x = 1 ∨ x = -1 :=
by
  sorry

end g_symmetry_solutions_l573_573544


namespace coefficient_of_square_term_l573_573085

theorem coefficient_of_square_term 
  (T : ℕ → ℝ)
  (h1 : ( ∑ k in Finset.range (n + 1), T k * x ^ (n - k) * (-sqrt 2) ^ k = (x - sqrt 2) ^ n))
  (h2 : (T 1 / T 3 = 1 / 2)) : 
  (third_term_coefficient (T 2 * x^2) = -12) := 
sorry

end coefficient_of_square_term_l573_573085


namespace arithmetic_sequence_general_term_l573_573840

noncomputable def arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a n = a 1 + (n - 1) * d

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℕ) 
  (h : ∀ n, a (n + 1) - a n = a 2 - a 1) 
  (h1 : a 1 + a 2 + a 3 = 12) 
  (h2 : a 1 * a 2 * a 3 = 48): 
  ∀ n, a n = 2 * n :=
begin
  sorry
end

end arithmetic_sequence_general_term_l573_573840


namespace sum_sequence_proof_l573_573535

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def sequence_x (n : ℕ) : ℤ := floor (n / 5)

def sum_sequence (n : ℕ) : ℤ := 
  (list.range (5 * n)).sum (λ i, sequence_x (i + 1))

theorem sum_sequence_proof (n : ℕ) :
  sum_sequence n = (5 * n * (n - 1) / 2) + n := 
sorry

end sum_sequence_proof_l573_573535


namespace xiaoming_problem_l573_573600

theorem xiaoming_problem :
  (- 1 / 24) / (1 / 3 - 1 / 6 + 3 / 8) = - 1 / 13 :=
by
  sorry

end xiaoming_problem_l573_573600


namespace exists_coprime_sequence_l573_573761

theorem exists_coprime_sequence :
  ∃ (a : ℕ → ℕ), (∀ i j p q r : ℕ, i ≠ j ∧ {i, j} ≠ {p, q, r} → i ≠ p ∧ i ≠ q ∧ i ≠ r ∧ j ≠ p ∧ j ≠ q ∧ j ≠ r → (a i < a j ∧ a j < a (j+1) 
    ∧ a (j+1) < a (j+2)) → Nat.gcd ((a i) + (a j)) ((a p) + (a q) + (a r)) = 1) :=
sorry

end exists_coprime_sequence_l573_573761


namespace return_percentage_is_6_5_l573_573717

def investment1 : ℤ := 16250
def investment2 : ℤ := 16250
def profit_percentage1 : ℚ := 0.15
def loss_percentage2 : ℚ := 0.05
def total_investment : ℤ := 25000
def net_income : ℚ := investment1 * profit_percentage1 - investment2 * loss_percentage2
def return_percentage : ℚ := (net_income / total_investment) * 100

theorem return_percentage_is_6_5 : return_percentage = 6.5 := by
  sorry

end return_percentage_is_6_5_l573_573717


namespace evaluate_polynomial_l573_573768

theorem evaluate_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) (hx : x > 0) :
  x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = (65 + 81 * real.sqrt 5) / 2 :=
sorry

end evaluate_polynomial_l573_573768


namespace checkerboard_no_identical_numbers_l573_573325

theorem checkerboard_no_identical_numbers :
  ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 11 ∧ 1 ≤ j ∧ j ≤ 19 → 19 * (i - 1) + j = 11 * (j - 1) + i → false :=
by
  sorry

end checkerboard_no_identical_numbers_l573_573325


namespace sequence_of_circles_in_triangle_l573_573745

/-- Given a sequence of circles within triangle ABC, and incircle radius r,
    prove the given relations involving tangents and radii -/
theorem sequence_of_circles_in_triangle
    (A B C : ℝ)
    (r : ℝ) -- radius of the incircle
    (r1 r2 : ℝ) -- radii of the first and second circles in the sequence
    (K1 K2 K3 K4 : ℝ) -- representation of circles
    :
    (r1 * cot (A / 2) + 2 * sqrt (r1 * r2) + r2 * cot (B / 2) = r * (cot (A / 2) + cot (B / 2)))
    ∧ (∃ t1 : ℝ, r1 = r * cot (B / 2) * cot (C / 2) * (sin t1) ^ 2)
    ∧ (K1 = K7) :=
    sorry

end sequence_of_circles_in_triangle_l573_573745


namespace terms_required_to_sum_identical_digits_l573_573111

theorem terms_required_to_sum_identical_digits :
  ∃ n : ℕ, (∃ a : ℕ, a * 111 = (n * (n + 1)) / 2) ∧ (100 ≤ (n * (n + 1)) / 2) ∧ ((n * (n + 1)) / 2) < 1000 := 
begin
  use 36,
  have ha : 6 * 111 = (36 * 37) / 2 := by norm_num,
  have hsum : (36 * 37) / 2 = 666 := by norm_num,  
  split,
  { 
    use 6,
    exact ha,
  },
  split,
  {
    norm_num,
  },
  {
    norm_num,
  },
end

end terms_required_to_sum_identical_digits_l573_573111


namespace min_value_f_l573_573390

open Real

noncomputable def f (t : ℝ) : ℝ :=
  ∫ x in 0..1, abs (exp x - t) + abs (exp (2 * x) - t)

theorem min_value_f : ∃ t, 1 ≤ t ∧ t ≤ exp 1 ∧ ∀ u, 1 ≤ u → u ≤ exp 1 → f u ≥ f (exp (2 / 3)) :=
begin
  use exp (2 / 3),
  split,
  { exact exp_pos (2 / 3) },
  split,
  { rw [← exp 1],
    apply exp_le_exp.mpr,
    norm_num },
  { intros u hu1 hu2,
    sorry -- Proof omitted
  }
end

end min_value_f_l573_573390


namespace count_four_digit_numbers_with_digit_sum_12_divisible_by_9_l573_573480

def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  (n.digits.sum = s)

def divisible_by (n m : ℕ) : Prop :=
  n % m = 0

theorem count_four_digit_numbers_with_digit_sum_12_divisible_by_9 :
  {n : ℕ | is_four_digit n ∧ digits_sum_to n 12 ∧ divisible_by (n.digits.sum) 9}.card = 220 :=
sorry

end count_four_digit_numbers_with_digit_sum_12_divisible_by_9_l573_573480


namespace determine_b_l573_573471

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem determine_b (a b c m1 m2 : ℝ) (h1 : a > b) (h2 : b > c) (h3 : f a b c 1 = 0)
  (h4 : a^2 + (f a b c m1 + f a b c m2) * a + (f a b c m1) * (f a b c m2) = 0) : 
  b ≥ 0 := 
by
  -- Proof logic goes here
  sorry

end determine_b_l573_573471


namespace real_solutions_eq59_l573_573022

theorem real_solutions_eq59 :
  (∃ (x: ℝ), -50 ≤ x ∧ x ≤ 50 ∧ (x / 50) = sin x) ∧
  (∃! (S: ℕ), S = 59) :=
sorry

end real_solutions_eq59_l573_573022


namespace multiple_of_denominator_l573_573591

def denominator := 5
def numerator := denominator + 4

theorem multiple_of_denominator:
  (numerator + 6) = 3 * denominator :=
by
  -- Proof steps go here
  sorry

end multiple_of_denominator_l573_573591


namespace find_number_of_valid_n_l573_573811

def valid_n (n : ℕ) : Prop :=
  (2 < n) ∧ (n < 100) ∧ ((∀ k : ℕ, k >= 0 → n = 4 * k + 3) ∨ (∃ k : ℕ, n = 4 * k + 3))

theorem find_number_of_valid_n : 
  {n : ℕ | valid_n n}.card = 24 :=
by
  sorry

end find_number_of_valid_n_l573_573811


namespace intersection_A_B_l573_573857

variable A : Set Int
variable B : Set Int

def setA : Set Int := {-1, 1, 2, 4}
def setB : Set Int := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  let A := setA
  let B := setB
  sorry

end intersection_A_B_l573_573857


namespace counting_positive_integers_satisfying_inequality_l573_573783

theorem counting_positive_integers_satisfying_inequality :
  ∃ (n : ℕ), (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 49 → (n - 2 * k) < 0) ∧ n = 47 :=
begin
  sorry
end

end counting_positive_integers_satisfying_inequality_l573_573783


namespace find_lambda_l573_573142

noncomputable def lambda_value (OA OB OC : ℝ) (λ : ℝ) : Prop := 
  let OM := (1 / 2) * OA + (1 / 6) * OB + λ * OC
  ∃ (MA MB MC : ℝ), (MA = OA - OM) ∧ (MB = OB - OM) ∧ (MC = OC - OM) ∧ 
  (MA * MB * MC = 0)

theorem find_lambda (OA OB OC : ℝ) :
  (λ : ℝ) → lambda_value OA OB OC λ → λ = 1 / 3 :=
  by sorry

end find_lambda_l573_573142


namespace no_integer_solutions_l573_573567

theorem no_integer_solutions (x y : ℤ) : ¬ (x^2 + 4 * x - 11 = 8 * y) := 
by
  sorry

end no_integer_solutions_l573_573567


namespace dodecagon_triangle_count_l573_573898

theorem dodecagon_triangle_count : 
  let n := 12 in
  nat.choose n 3 = 220 :=
by
  sorry

end dodecagon_triangle_count_l573_573898


namespace shallow_depth_of_pool_l573_573714

theorem shallow_depth_of_pool (w l D V : ℝ) (h₀ : w = 9) (h₁ : l = 12) (h₂ : D = 4) (h₃ : V = 270) :
  (0.5 * (d + D) * w * l = V) → d = 1 :=
by
  intros h_equiv
  sorry

end shallow_depth_of_pool_l573_573714


namespace find_MN_length_l573_573502

-- Definitions and conditions:
variables {α : Type*} [LinearOrderedField α]

def is_isosceles (a b : α) := ∀ {A B C : α}, B = A + a ∧ C = A + a ∧ B ≠ C ∧ b = A ∧ ∀ M N : α, AM_is_bisector A B ∧ CN_is_bisector C B

def AM_is_bisector (A B : α) {M : α} : Prop :=
∃ (x y z : α), A = B + x ∧ M = B + y ∧ C = B + z

def CN_is_bisector (C B : α) {N : α} : Prop :=
∃ (x y z : α), C = B + x ∧ N = B + y ∧ A = B + z

-- Proof statement:
theorem find_MN_length (a b : α) (h_iso : is_isosceles a b) :
  ∃ MN : α, MN = (a * b) / (a + b) :=
sorry

end find_MN_length_l573_573502


namespace triangle_ratio_l573_573508

variable (A B C P1 P2 D E M : Type)
variable [Field A] [Field B] [Field C] [Field P1] [Field P2] [Field D] [Field E] [Field M]

variable (triangle_ABC : A → B → C → Prop)
variable (midpointM : M → B → C → Prop)
variable (symmetric_lines : P1 → P2 → M → Prop)
variable (second_intersection : D → E → M → P1 → P2 → Prop)

variable (eq_sines : sin (angle D M C) / sin (angle E M B) = DP1 / EM)
variable (AP_equal : AP1 = AP2)

theorem triangle_ratio :
  (∃ (A B C P1 P2 D E M : Type) 
     [Field A] [Field B] 
     [Field C] [Field P1] [Field P2] 
     [Field D] [Field E] [Field M]
     (triangle_ABC : A → B → C → Prop)
     (midpointM : M → B → C → Prop)
     (symmetric_lines : P1 → P2 → M → Prop)
     (second_intersection : D → E → M → P1 → P2 → Prop)
     (eq_sines : sin (angle D M C) / sin (angle E M B) = DP1 / EM)
     (AP_equal : AP1 = AP2)),
  ∀ (BP1 BC DP1 EM: ℝ), 
  \frac{BP1}{BC} = \frac{1}{2} \cdot \frac{DP1}{EM} :=
sorry

end triangle_ratio_l573_573508


namespace paperboy_delivery_count_l573_573394

def no_miss_four_consecutive (n : ℕ) (E : ℕ → ℕ) : Prop :=
  ∀ k > 3, E k = E (k - 1) + E (k - 2) + E (k - 3)

def base_conditions (E : ℕ → ℕ) : Prop :=
  E 1 = 2 ∧ E 2 = 4 ∧ E 3 = 8

theorem paperboy_delivery_count : ∃ (E : ℕ → ℕ), 
  base_conditions E ∧ no_miss_four_consecutive 12 E ∧ E 12 = 1854 :=
by
  sorry

end paperboy_delivery_count_l573_573394


namespace max_perfect_square_area_l573_573697

theorem max_perfect_square_area (l w : ℕ) (h1 : 2 * l + 2 * w = 34) (h2 : 0 < l) (h3 : 0 < w) : ∃ (lw : ℕ), lw = l * w ∧ nat.sqrt lw * nat.sqrt lw = lw ∧ ∀ (l' w' : ℕ), (2 * l' + 2 * w' = 34) → (nat.sqrt (l' * w') * nat.sqrt (l' * w') = l' * w' → l * w ≥ l' * w') :=
by
  sorry

end max_perfect_square_area_l573_573697


namespace possible_same_tallest_dwarf_shortest_giant_no_shortest_giant_less_than_tallest_dwarf_l573_573497

variable {α : Type*} [LinearOrder α]
variable (matrix : List (List α))

noncomputable def tallestDwarf (matrix : List (List α)) : α :=
  matrix.foldl (λ acc row => max acc (row.foldl min (row.head!))) (matrix.head!.head!)

noncomputable def shortestGiant (matrix : List (List α)) : α :=
  List.foldr (λ idx acc => min acc (matrix.foldr (λ row acc' => max acc' (List.get! row idx)) (List.head! (List.head! matrix)))) (List.head! (List.head! matrix)) (List.range (List.length (List.head! matrix)))

-- Proposition 1: It is possible that the tallest dwarf is equal to the shortest giant
theorem possible_same_tallest_dwarf_shortest_giant (mat : List (List α)) :
  ∃ (student : α), student = tallestDwarf mat ∧ student = shortestGiant mat := sorry

-- Proposition 2: There are no situations where the shortest giant is shorter than the tallest dwarf
theorem no_shortest_giant_less_than_tallest_dwarf (mat : List (List α)) :
  ∀ (td : α) (sg : α), sg = shortestGiant mat → td = tallestDwarf mat → sg < td → False := sorry

end possible_same_tallest_dwarf_shortest_giant_no_shortest_giant_less_than_tallest_dwarf_l573_573497


namespace next_June_8_Sunday_l573_573206

theorem next_June_8_Sunday (H1 : ∃ y : ℕ, y = 2024 ∧ is_leap_year y)
(H2 : day_of_week 2024 1 1 = day_of_week.monday) :  
∃ y : ℕ, y > 2024 ∧ day_of_week y 6 8 = day_of_week.sunday ∧ y = 2030 :=
sorry

end next_June_8_Sunday_l573_573206


namespace C_and_D_mutually_exclusive_B_and_D_independent_l573_573917

-- Define events A, B, C, D in the context of the probability space
def eventA (s : List ℕ) : Prop := s.headD 0 = 2
def eventB (s : List ℕ) : Prop := s.tail.headD 0 = 3
def eventC (s : List ℕ) : Prop := s.sum = 4
def eventD (s : List ℕ) : Prop := s.sum = 5

-- Define the finite sample space of drawing two balls without replacement from {1, 2, 3, 4}
def sample_space : List (List ℕ) :=
  { [1, 2], [1, 3], [1, 4], [2, 1], [2, 3], [2, 4], [3, 1], [3, 2], [3, 4], [4, 1], [4, 2], [4, 3] }

-- Define the probability measure on the sample space based on uniform distribution
def prob (p : List ℕ → Prop) : ℝ :=
  (sample_space.filter p).length.toReal / sample_space.length.toReal

-- Statement for mutual exclusivity of events C and D
theorem C_and_D_mutually_exclusive : ∀ s ∈ sample_space, ¬ (eventC s ∧ eventD s) :=
begin
  intros s hs,
  simp [eventC, eventD],
  sorry
end

-- Statement for independence of events B and D
theorem B_and_D_independent : prob eventB * prob eventD = prob (λ s, eventB s ∧ eventD s) :=
begin
  simp [prob, eventB, eventD],
  sorry
end

end C_and_D_mutually_exclusive_B_and_D_independent_l573_573917


namespace num_arts_students_l573_573495

-- Definitions for the conditions
def locals_arts (A : ℕ) := 0.5 * A
def locals_science := 25
def locals_commerce := 102
def total_locals := 327

-- The theorem to be proved
theorem num_arts_students (A : ℕ) (h1 : locals_arts A + locals_science + locals_commerce = total_locals) : A = 400 :=
by
  sorry

end num_arts_students_l573_573495


namespace ratio_of_gilled_to_spotted_l573_573324

theorem ratio_of_gilled_to_spotted (total_mushrooms gilled_mushrooms spotted_mushrooms : ℕ) 
  (h1 : total_mushrooms = 30) 
  (h2 : gilled_mushrooms = 3) 
  (h3 : spotted_mushrooms = total_mushrooms - gilled_mushrooms) :
  gilled_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 1 ∧ 
  spotted_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 9 := 
by
  sorry

end ratio_of_gilled_to_spotted_l573_573324


namespace volume_of_region_is_36_l573_573034

open Set

-- Definitions of the constraints
def region (x y z : ℝ) : Prop :=
  abs (x + y + z) + abs (x + y - z) ≤ 12 ∧
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x + y ≥ 2 * z

-- The statement to prove the volume of the region
theorem volume_of_region_is_36 :
  (∫∫∫ r in Region, 1) = 36 :=
by
  sorry

end volume_of_region_is_36_l573_573034


namespace eval_integral_l573_573006

theorem eval_integral : ∀ (C : ℝ), ∫ (x : ℝ) in 0..x, (x * cos x) / (sin x) ^ 3 = - (x + cos x * sin x) / (2 * sin x ^ 2) + C :=
by
  sorry

end eval_integral_l573_573006


namespace calculate_division_of_powers_l573_573375

theorem calculate_division_of_powers (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end calculate_division_of_powers_l573_573375


namespace negation_exists_implies_forall_l573_573259

theorem negation_exists_implies_forall : 
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by
  sorry

end negation_exists_implies_forall_l573_573259


namespace roger_steps_time_l573_573573

theorem roger_steps_time (steps_per_30_min : ℕ := 2000) (time_for_2000_steps : ℕ := 30) (goal_steps : ℕ := 10000) : 
  (goal_steps * time_for_2000_steps) / steps_per_30_min = 150 :=
by 
  -- This is the statement. Proof is omitted as per instruction.
  sorry

end roger_steps_time_l573_573573


namespace fiona_observe_pairs_l573_573131

def classroom_pairs (n : ℕ) : ℕ :=
  if n > 1 then n - 1 else 0

theorem fiona_observe_pairs :
  classroom_pairs 12 = 11 :=
by
  sorry

end fiona_observe_pairs_l573_573131


namespace fokker_planck_equation_l573_573646

noncomputable def Pt (t x μ σ : ℝ) : ℝ :=
  (sqrt(2) / (sqrt(π * t))) * exp (- (x - μ * t) ^ 2 / (2 * t * σ^2))

theorem fokker_planck_equation (t x μ σ : ℝ) (ht : 0 < t) (hσ : 0 < σ) :
  deriv (λ t, Pt t x μ σ) t = 
  (-μ) * deriv (λ x, Pt t x μ σ) x + (1 / 2) * σ^2 * (deriv (λ x, deriv (λ x, Pt t x μ σ) x) x) :=
sorry

end fokker_planck_equation_l573_573646


namespace Freddy_calling_cost_l573_573427

theorem Freddy_calling_cost :
  let cost_dad := 45 * 0.05 in
  let cost_brother := 31 * 0.25 in
  let cost_cousin := 20 * 0.10 in
  let cost_grandparents := 15 * 0.30 in
  let total_cost := cost_dad + cost_brother + cost_cousin + cost_grandparents in
  total_cost = 16.50 :=
by
  sorry

end Freddy_calling_cost_l573_573427


namespace complement_U_A_is_correct_l573_573893

open Set

/-- Define the universal set U -/
def U : Set ℝ := {x | x^2 > 1}

/-- Define the set A -/
def A : Set ℝ := {x | x^2 - 4x + 3 < 0}

/-- Define the complement of A in U -/
def complement_U_A := U \ A

/-- Prove that the complement of A in U is equal to (-∞, -1) ∪ [3, +∞) -/
theorem complement_U_A_is_correct :
  complement_U_A = {x | x < -1} ∪ {x | 3 ≤ x} :=
by
  sorry

end complement_U_A_is_correct_l573_573893


namespace hyperbola_foci_asymptote_distance_l573_573594

theorem hyperbola_foci_asymptote_distance : 
  ∀ x y : ℝ, (x^2 / 8 - y^2 = 1) → 
  let c:= sqrt (8 + 1) 
  let foci : Set (ℝ × ℝ) := { (3,0), (-3,0) } 
  let asymptotes : Set (ℝ × ℝ) := { (x, y) | y = sqrt(2)/4 * x ∨ y = -sqrt(2)/4 * x }
  ∀ f : (ℝ × ℝ), f ∈ foci → 
  ∀ a b k : ℝ, a x + b y = k → 
  ∃ d : ℝ, d = abs(3) / sqrt(1 + 8) ∧ d = 1 :=
sorry

end hyperbola_foci_asymptote_distance_l573_573594


namespace _l573_573934

noncomputable def angle_ACB_is_45_degrees (A B C D E F : Type) [LinearOrderedField A]
  (angle : A → A → A → A) (AB AC : A) (h1 : AB = 3 * AC)
  (BAE ACD : A) (h2 : BAE = ACD)
  (BCA : A) (h3 : BAE = 2 * BCA)
  (CF FE : A) (h4 : CF = FE)
  (is_isosceles : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a = b → b = c → a = c)
  (triangle_sum : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a + b + c = 180) :
  ∃ (angle_ACB : A), angle_ACB = 45 := 
by
  -- Here we assume we have the appropriate conditions from geometry
  -- Then you'd prove the theorem based on given hypotheses
  sorry

end _l573_573934


namespace jane_output_increase_with_assistant_l573_573655

-- Definitions of conditions
variables (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

def withAssistantBears : ℝ := 1.80 * B
def withAssistantHours : ℝ := 0.90 * H

-- Calculate output per hour with and without assistant
def originalOutputPerHour : ℝ := B / H
def assistantOutputPerHour : ℝ := withAssistantBears B / withAssistantHours H

-- The theorem to prove
theorem jane_output_increase_with_assistant (B H : ℝ) (h_B_pos : B > 0) (h_H_pos : H > 0) :
  100 * ((assistantOutputPerHour B H - originalOutputPerHour B H) / originalOutputPerHour B H) = 100 :=
by
  sorry

end jane_output_increase_with_assistant_l573_573655


namespace matrix_BA_l573_573536

-- Define matrices A and B
variables (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- Define the given conditions
noncomputable def condition1 : Prop := A + B = A * B
noncomputable def condition2 : Prop := A * B = ![{#[[1, 2], [3, 4]]}]
noncomputable def condition3 : Prop := A * B = B * A

-- Prove that B * A is equal to the given matrix
theorem matrix_BA :
  condition1 A B →
  condition2 A B →
  condition3 A B →
  B * A = ![{#[[1, 2], [3, 4]]}] :=
by
  -- Proof is omitted
  sorry

end matrix_BA_l573_573536


namespace glasses_per_pitcher_l573_573381

theorem glasses_per_pitcher (total_glasses : ℕ) (num_pitchers : ℕ) (h_total : total_glasses = 30) (h_pitchers : num_pitchers = 6) :
  total_glasses / num_pitchers = 5 :=
by
  have h_div : total_glasses / num_pitchers = 30 / 6, from congr_arg2 (/) h_total h_pitchers
  rw div_eq_of_lt
  exact h_div

end glasses_per_pitcher_l573_573381


namespace find_b_l573_573293

-- Definitions from the conditions
variables (a b : ℝ)

-- Theorem statement using the conditions and the correct answer
theorem find_b (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
by
  sorry

end find_b_l573_573293


namespace CE_perpendicular_BD_l573_573150

-- Definitions and conditions
variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D A₁ B₁ C₁ D₁ E : V)
variables (cube : set V) (midpoint : V → V → V) (midpE : E = midpoint A₁ C₁)

-- The relationship to be proved
theorem CE_perpendicular_BD (h1 : midpoint A₁ C₁ = E) (h2 : ∃ cube, cube = ({A, B, C, D, A₁, B₁, C₁, D₁} : set V)):
  ⟪E - C, D - B⟫ = 0 := by sorry

end CE_perpendicular_BD_l573_573150


namespace shaded_area_l573_573152

theorem shaded_area (area_circle : ℝ) (h1 : area_circle = 100 * Real.pi)
    (center_in_circle : ∀ r: ℝ, r = 10)
    (diagonal_is_diameter : ∀ d: ℝ, d = 20) : 
    (shaded_area : ℝ) = 50 * Real.pi :=
by
  sorry

end shaded_area_l573_573152


namespace M_subset_P_l573_573892

open Set

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}
def P : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 2)}

theorem M_subset_P : M ⊆ P := 
by
  sorry

end M_subset_P_l573_573892


namespace count_positive_integers_satisfying_inequality_l573_573796

theorem count_positive_integers_satisfying_inequality :
  let S := {n : ℕ | n > 0 ∧ (n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0}
  ∃ n, S.card = 24 :=
by sorry

end count_positive_integers_satisfying_inequality_l573_573796


namespace diameter_increase_factor_l573_573672

noncomputable def original_time_per_round := 40 / 8
noncomputable def new_time_per_round := 50

def increased_factor (k : ℝ) (d : ℝ) :=
  ∀ (C_original C_new : ℝ), 
  C_original = π * d →
  C_new = π * (k * d) →
  (original_time_per_round / new_time_per_round = C_original / C_new) →
  k = 10

theorem diameter_increase_factor (k : ℝ) (d : ℝ) : 
  increased_factor k d :=
by
  unfold increased_factor
  intros C_original C_new hC_orig hC_new h_ratio
  sorry

end diameter_increase_factor_l573_573672


namespace range_of_g_l573_573179

noncomputable def f (x : ℝ) : ℝ := 2 * x - 3

noncomputable def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → -29 ≤ g x ∧ g x ≤ 3) :=
sorry

end range_of_g_l573_573179


namespace negation_at_most_three_l573_573258

theorem negation_at_most_three :
  ¬ (∀ n : ℕ, n ≤ 3) ↔ (∃ n : ℕ, n ≥ 4) :=
by
  sorry

end negation_at_most_three_l573_573258


namespace current_temperature_l573_573269

theorem current_temperature (required_temp increase_needed : ℕ) (h1 : required_temp = 546) (h2 : increase_needed = 396) :
  required_temp - increase_needed = 150 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add (eq.symm (Nat.add_sub_of_le (le_add_of_nonneg_right (le_refl _))))

end current_temperature_l573_573269


namespace initial_rows_of_chairs_l573_573415

theorem initial_rows_of_chairs (x : ℕ) (h1 : 12 * x + 11 = 95) : x = 7 := 
by
  sorry

end initial_rows_of_chairs_l573_573415


namespace age_problem_l573_573551

theorem age_problem (a b c d : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : b = 3 * d)
  (h4 : a + b + c + d = 87) : 
  b = 30 :=
by sorry

end age_problem_l573_573551


namespace intersection_of_A_and_B_l573_573849

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l573_573849


namespace ball_total_distance_l573_573366

noncomputable def total_distance (initial_height : ℕ) (bounces : ℕ) : ℕ :=
  let rec aux (n : ℕ) (height : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else
      aux (n - 1) (height / 2) (acc + height + (height / 2))
  aux bounces initial_height initial_height

theorem ball_total_distance (initial_height : ℕ) (bounces : ℕ) :
  initial_height = 16 ∧ bounces = 4 → total_distance initial_height bounces = 45 :=
by
  intro h
  cases' h with h_initial h_bounces
  rw [h_initial, h_bounces]
  -- Here using nat type so we need to convert it accordingly
  have : total_distance 16 4 = 16 + 8 + 8 + 4 + 4 + 2 + 2 + 1 := by 
  -- You can add more steps/descriptions if necessary
  sorry
  rw [this]
  norm_num
  exact rfl

end ball_total_distance_l573_573366


namespace inequality_am_gm_l573_573441

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3 * a * b * c :=
sorry

end inequality_am_gm_l573_573441


namespace find_peter_depth_l573_573205

def depth_of_peters_pond (p : ℕ) : Prop :=
  let mark_depth := 19
  let relationship := 3 * p + 4
  relationship = mark_depth

theorem find_peter_depth : ∃ p : ℕ, depth_of_peters_pond p ∧ p = 5 :=
by {
  use 5,
  unfold depth_of_peters_pond,
  split,
  { refl },
  { refl }
}

end find_peter_depth_l573_573205


namespace triangle_inequality_l573_573160

theorem triangle_inequality (a b c : ℝ)
  (R : ℝ := 1)
  (S_triangle : ℝ := 1 / 4)
  (h1 : a * b * c = 4 * R * S_triangle) :
  (sqrt a + sqrt b + sqrt c < 1 / a + 1 / b + 1 / c) :=
by
  sorry

end triangle_inequality_l573_573160


namespace arithmetic_c_sequence_bounded_sum_l573_573862

noncomputable def a (n : ℕ) := d + n * d
noncomputable def b (n : ℕ) := Real.sqrt (a n * a (n + 1))
noncomputable def c (n : ℕ) := (b (n + 1))^2 - (b n)^2
noncomputable def T (n : ℕ) := ∑ k in (1:2n).toFinset, (if even k then 1 else -1) * (b k)^2

theorem arithmetic_c_sequence (d : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)
  (h_an : ∀ n, a (n+1) = a_n + d) 
  (h_b : ∀ n, b n = Real.sqrt (a n * a (n + 1)))
  (h_c : ∀ n, c n = (b (n + 1))^2 - (b n)^2) : 
  ∀ n, c_{n+1} - c_n = 2d^2 := 
by sorry

theorem bounded_sum (d : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ)
  (h_an : ∀ n, a (n+1) = a_n + d) 
  (h_b : ∀ n, b n = Real.sqrt (a n * a (n + 1)))
  (h_c : ∀ n, c n = (b (n + 1))^2 - (b n)^2)
  (h_T : ∀ n, T n = ∑ k in (1:2n).toFinset, (if even k then 1 else -1) * (b k)^2) : 
  ∑ k in (1:n).toFinset, 1 / T k < 1 / (2 * d^2) := 
by sorry

end arithmetic_c_sequence_bounded_sum_l573_573862


namespace gcd_lcm_inequality_gcd_lcm_equality_condition_l573_573532

theorem gcd_lcm_inequality (a b : ℕ) (h1 : a * b > 2)
  (h2 : (gcd a b + lcm a b) % (a + b) = 0) :
  (gcd a b + lcm a b) / (a + b) ≤ (a + b) / 4 :=
begin
  sorry
end

theorem gcd_lcm_equality_condition (a b : ℕ) (h1 : a * b > 2)
  (h2 : (gcd a b + lcm a b) % (a + b) = 0) :
  (gcd a b + lcm a b) / (a + b) = (a + b) / 4 ↔ ∃ x y : ℕ, gcd x y = 1 ∧ a = gcd a b * x ∧ b = gcd a b * y ∧ x - y = 2 :=
begin
  sorry
end

end gcd_lcm_inequality_gcd_lcm_equality_condition_l573_573532


namespace abc_relationship_l573_573442

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := ∫ x in 1..2, x + 1/x
noncomputable def c : ℝ := -Real.log 30 / Real.log 3

theorem abc_relationship : a < b ∧ b < c :=
by
  have ha : 1 < a ∧ a < 2 := sorry
  have hb : 2 < b ∧ b < 5/2 := sorry
  have hc : c > 3 := sorry
  exact ⟨ha.2, hb.2⟩

end abc_relationship_l573_573442


namespace add_fractions_l573_573640

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l573_573640


namespace bisector_line_intersection_area_l573_573547

theorem bisector_line_intersection_area (A B C : Point)
    (h_eq_triangle : triangle A B C)
    (h_ratio : (1 : ℝ) / 2 < dist A B / dist A C ∧ dist A B / dist A C < 1) :
    ∃ ℓ : Line, (intersection_area (reflection (triangle A B C) ℓ) (triangle_interior A B C)) > (2 / 3) * area (triangle A B C) :=
by
  sorry

end bisector_line_intersection_area_l573_573547


namespace divisor_count_of_8n3_l573_573960

def is_odd (n : ℕ) : Prop := n % 2 = 1

def num_divisors (n : ℕ) : ℕ := 
  (factors n).product.map (λ p, p.count + 1).prod

theorem divisor_count_of_8n3 (n : ℕ) (hn_odd : is_odd n) (hn_divisor : num_divisors n = 11) : 
  num_divisors (8 * n^3) = 124 :=
sorry

end divisor_count_of_8n3_l573_573960


namespace ellipse_equation_and_circle_l573_573066

-- Given conditions
variable (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = √6) (h4 : b = √2)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the points of intersection
def line (x y : ℝ) : Prop := y = x + 2

-- Variables for points A and B
variable (A B : ℝ × ℝ) (hA : ellipse a b A.1 A.2) (hB : ellipse a b B.1 B.2) (hA_line : line A.1 A.2) (hB_line : line B.1 B.2)

-- Proof problem combining all conditions
theorem ellipse_equation_and_circle :
  let a := √(2 + c^2) in
  ellipse (x : ℝ) (y : ℝ) ↔ (x^2 / 8 + y^2 / 2 = 1) ∧ ∀ A B : ℝ × ℝ, 
  ellipse A.1 A.2 → ellipse B.1 B.2 → line A.1 A.2 → line B.1 B.2 → 
  ∃ x0 y0 r, 
  (x + x0)^2 + (y + y0)^2 = r^2 ∧ x0 = -8/5 ∧ y0 = 2/5 ∧ r^2 = 48/25 := 
by
  sorry

end ellipse_equation_and_circle_l573_573066


namespace intersection_correct_l573_573850

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_correct : A ∩ B = {1, 2} := 
sorry

end intersection_correct_l573_573850


namespace max_daily_profit_l573_573676

def cost_price : ℕ := 30
def selling_price₁ (x : ℕ) : ℕ := 0.5 * x + 35
def quantity_sold₁ (x : ℕ) : ℕ := 128 - 2 * x
def selling_price₂ : ℕ := 50
def quantity_sold₂ (x : ℕ) : ℕ := 128 - 2 * x

def profit₁ (x : ℕ) : ℕ := (0.5 * x + 5) * (128 - 2 * x)
def profit₂ (x : ℕ) : ℕ := 20 * (128 - 2 * x)

noncomputable
def daily_profit (x : ℕ) : ℕ := 
  if 1 ≤ x ∧ x ≤ 30 then -x^2 + 54 * x + 640
  else if 31 ≤ x ∧ x ≤ 60 then -40 * x + 2560
  else 0

theorem max_daily_profit : ∃ x, 1 ≤ x ∧ x ≤ 60 ∧ daily_profit x = 1369 := sorry

end max_daily_profit_l573_573676


namespace intersection_M_N_eq_set_l573_573891

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- The theorem to be proved
theorem intersection_M_N_eq_set : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_set_l573_573891


namespace min_value_18_solve_inequality_l573_573864

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (1/a^3) + (1/b^3) + (1/c^3) + 27 * a * b * c

theorem min_value_18 (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  min_value a b c ≥ 18 :=
by sorry

theorem solve_inequality (x : ℝ) :
  abs (x + 1) - 2 * x < 18 ↔ x > -(19/3) :=
by sorry

end min_value_18_solve_inequality_l573_573864


namespace intersection_A_B_l573_573854

variable A : Set Int
variable B : Set Int

def setA : Set Int := {-1, 1, 2, 4}
def setB : Set Int := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  let A := setA
  let B := setB
  sorry

end intersection_A_B_l573_573854


namespace find_OP_l573_573576

variable (a b c d e f : ℝ)
variable (P : ℝ)

-- Given conditions
axiom AP_PD_ratio : (a - P) / (P - d) = 2 / 3
axiom BP_PC_ratio : (b - P) / (P - c) = 3 / 4

-- Conclusion to prove
theorem find_OP : P = (3 * a + 2 * d) / 5 :=
by
  sorry

end find_OP_l573_573576


namespace simplify_and_evaluate_l573_573230

theorem simplify_and_evaluate 
  (a b : ℚ) (h_a : a = -1/3) (h_b : b = -3) : 
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := 
  by 
    rw [h_a, h_b]
    sorry

end simplify_and_evaluate_l573_573230


namespace max_value_sqrt43_l573_573964

noncomputable def max_value_expr (x y z : ℝ) : ℝ :=
  3 * x * z * Real.sqrt 2 + 5 * x * y

theorem max_value_sqrt43 (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x^2 + y^2 + z^2 = 1) :
  max_value_expr x y z ≤ Real.sqrt 43 :=
sorry

end max_value_sqrt43_l573_573964


namespace ammonium_chloride_reaction_l573_573774

/-- 
  Given the reaction NH4Cl + H2O → NH4OH + HCl, 
  if 1 mole of NH4Cl reacts with 1 mole of H2O to produce 1 mole of NH4OH, 
  then 1 mole of HCl is formed.
-/
theorem ammonium_chloride_reaction :
  (∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl = 1 ∧ H2O = 1 ∧ NH4OH = 1 → HCl = 1) :=
by
  sorry

end ammonium_chloride_reaction_l573_573774


namespace skittles_per_friend_l573_573528

theorem skittles_per_friend (ts : ℕ) (nf : ℕ) (h1 : ts = 200) (h2 : nf = 5) : (ts / nf = 40) :=
by sorry

end skittles_per_friend_l573_573528


namespace jason_retirement_age_l573_573939

variable (join_age : ℕ) (years_to_chief : ℕ) (percent_longer : ℕ) (additional_years : ℕ)

def time_to_master_chief := years_to_chief + (years_to_chief * percent_longer / 100)

def total_time_in_military := years_to_chief + time_to_master_chief years_to_chief percent_longer + additional_years

def retirement_age := join_age + total_time_in_military join_age years_to_chief percent_longer additional_years

theorem jason_retirement_age :
  join_age = 18 →
  years_to_chief = 8 →
  percent_longer = 25 →
  additional_years = 10 →
  retirement_age join_age years_to_chief percent_longer additional_years = 46 :=
by
  intros h1 h2 h3 h4
  simp [retirement_age, total_time_in_military, time_to_master_chief, h1, h2, h3, h4]
  sorry

end jason_retirement_age_l573_573939


namespace counting_positive_integers_satisfying_inequality_l573_573782

theorem counting_positive_integers_satisfying_inequality :
  ∃ (n : ℕ), (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 49 → (n - 2 * k) < 0) ∧ n = 47 :=
begin
  sorry
end

end counting_positive_integers_satisfying_inequality_l573_573782


namespace number_of_ordered_pairs_l573_573016

theorem number_of_ordered_pairs : 
  let f : ℕ → ℕ := λ n, (if 2 ∣ n then n/2 else (n+1)/2)
  in (∑ a in (finset.range 50).filter (λ n, 1 ≤ n), f a) = 800 :=
by
  sorry

end number_of_ordered_pairs_l573_573016


namespace find_bxy_l573_573194

theorem find_bxy (x y b : ℤ) (hx : x = 20) (hy : y = 1) (hb : b = 1) :
  ((2 : ℝ)^(0.15 * x)) ^ (b * y) = 8 :=
by
  rw [hx, hy, hb]
  norm_num
  sorry

end find_bxy_l573_573194


namespace congruence_mod_10_l573_573003

theorem congruence_mod_10 :
  let a := ∑ k in finset.range 21, nat.choose 20 k * (2 ^ k)
  in a % 10 = 2011 % 10 :=
by
  sorry

end congruence_mod_10_l573_573003


namespace negation_of_universal_proposition_l573_573602

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x < 0 :=
by sorry

end negation_of_universal_proposition_l573_573602


namespace equal_area_division_l573_573035

theorem equal_area_division (d : ℝ) (x y : ℝ) : 
  let area : ℝ := 5,
      half_area : ℝ := area / 2,
      line_eq : (x = d ∧ y = 0) ∨ (x = 2 ∧ y = 2) → (2 - d) * (2 + (2 * d) / (2 - d)) = 2 * 2.5 in
  d = 1 / 3 :=
by
  sorry

end equal_area_division_l573_573035


namespace range_of_m_l573_573432

noncomputable def p (m : ℝ) : Prop :=
  ∀ x ∈ set.Icc (1 / 4 : ℝ) (1 / 2 : ℝ), 2 * x < m * (x^2 + 1)

noncomputable def q (m : ℝ) : Prop :=
  ∃ x : ℝ, 4^x + 2^(x + 1) + m - 1 = 0

theorem range_of_m (m : ℝ) : p m ∧ q m ↔ m ∈ set.Ioo (4 / 5 : ℝ) 1 :=
sorry

end range_of_m_l573_573432


namespace max_C_n_l573_573417

noncomputable def LCM (s : Set ℕ) : ℕ := sorry

open Set

theorem max_C_n (n : ℕ) (hn : n > 0) : 
  ∃ C : ℝ, 
  (∀ f : ℤ → ℤ, ∀ a b : ℤ, a ≠ b ∧ f a ≠ f b → C ≤ | ((f a - f b) : ℚ) / (a - b) |)
  ∧ (C = 1 / (LCM {x | 1 ≤ x ∧ x ≤ n}))
:= sorry

end max_C_n_l573_573417


namespace work_done_to_stretch_is_0_18J_l573_573906

-- The given conditions as parameters and definitions
def spring_constant (F : ℝ) (x : ℝ) : ℝ := F / x
def work_done (k : ℝ) (x : ℝ) : ℝ := 1/2 * k * x^2

constant F : ℝ := 10 -- Force in Newton
constant x1 : ℝ := 0.1 -- Compression distance in meters
constant x2 : ℝ := 0.06 -- Stretching distance in meters

-- Compute the spring constant
noncomputable def k : ℝ := spring_constant F x1

-- The main proof statement
theorem work_done_to_stretch_is_0_18J : work_done k x2 = 0.18 := by
  sorry

end work_done_to_stretch_is_0_18J_l573_573906


namespace range_of_m_if_p_true_range_of_m_if_p_and_q_true_l573_573075

variable (m : ℝ)

def eccentricity_hyperbola (m : ℝ) : ℝ :=
  let b_sq := 5
  let a_sq := m
  real.sqrt (1 + a_sq / b_sq)

def represents_ellipse (m : ℝ) : Prop :=
  0 < m ∧ m < 9 - m

def proposition_p : Prop :=
  ecc = eccentricity_hyperbola m
  ecc > real.sqrt(6) / 2 ∧ ecc < real.sqrt 2

def proposition_q : Prop :=
  represents_ellipse m

theorem range_of_m_if_p_true {m : ℝ} (hp : proposition_p m) : 5/2 < m ∧ m < 5 := 
  sorry

theorem range_of_m_if_p_and_q_true {m : ℝ} (hp : proposition_p m) (hq : proposition_q m) : 5/2 < m ∧ m < 3 :=
  sorry

end range_of_m_if_p_true_range_of_m_if_p_and_q_true_l573_573075


namespace expected_value_Y_l573_573889

-- Define the probability mass function P(X = k)
noncomputable def P_X (k : ℕ) : ℚ := if k = 0 then 0 else 1 / (2 * k)

-- Define the random variable Y as a function of X mod 3
noncomputable def Y (X : ℕ) : ℕ := 3 % X

-- Define the expected value function computation for Y
noncomputable def E_Y : ℚ :=
  ∑ i in (finset.range 1000), (if Y i = 0 then 0 else real.to_rat ((Y i : ℝ) / (1000 : ℝ))) * P_X i

-- The theorem statement
theorem expected_value_Y : E_Y = 8 / 7 := 
by sorry

end expected_value_Y_l573_573889


namespace sufficient_not_necessary_condition_not_necessary_condition_l573_573865

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  a^2 + b^2 = 1 → (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) :=
by
  sorry

theorem not_necessary_condition (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) → ¬(a^2 + b^2 = 1) :=
by
  sorry

end sufficient_not_necessary_condition_not_necessary_condition_l573_573865


namespace ratio_of_areas_l573_573909

theorem ratio_of_areas (R_A R_B : ℝ) 
  (h1 : (1 / 6) * 2 * Real.pi * R_A = (1 / 9) * 2 * Real.pi * R_B) :
  (Real.pi * R_A^2) / (Real.pi * R_B^2) = (4 : ℝ) / 9 :=
by 
  sorry

end ratio_of_areas_l573_573909


namespace pencils_per_student_l573_573308

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ) 
  (h_total : total_pencils = 125) 
  (h_students : students = 25) 
  (h_div : pencils_per_student = total_pencils / students) : 
  pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l573_573308


namespace gas_cost_problem_l573_573820

theorem gas_cost_problem (x : ℝ) (h : x / 4 - 15 = x / 7) : x = 140 :=
sorry

end gas_cost_problem_l573_573820


namespace percentage_students_passed_l573_573136

theorem percentage_students_passed
    (total_students : ℕ)
    (students_failed : ℕ)
    (students_passed : ℕ)
    (percentage_passed : ℕ)
    (h1 : total_students = 840)
    (h2 : students_failed = 546)
    (h3 : students_passed = total_students - students_failed)
    (h4 : percentage_passed = (students_passed * 100) / total_students) :
    percentage_passed = 35 := by
  sorry

end percentage_students_passed_l573_573136


namespace square_root_of_4_is_pm2_l573_573613

theorem square_root_of_4_is_pm2 : ∃ (x : ℤ), x * x = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end square_root_of_4_is_pm2_l573_573613


namespace number_of_solutions_l573_573790
-- Importing the necessary library

-- Define the condition for the product of the sequence to be negative
noncomputable def condition (n : ℕ) :=
  (n > 0) ∧ (Finset.prod (Finset.range 49) (λ k, (n - 2 * (k + 1)) : ℤ) < 0)

-- State theorem corresponding to the mathematically equivalent proof problem
theorem number_of_solutions : 
  finset.filter condition (finset.range (99)).card = 24 :=
sorry

end number_of_solutions_l573_573790


namespace fraction_addition_l573_573645

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l573_573645


namespace intersection_correct_l573_573852

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_correct : A ∩ B = {1, 2} := 
sorry

end intersection_correct_l573_573852


namespace simplify_fraction_l573_573990

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l573_573990


namespace lunch_cost_before_tip_l573_573626

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.2 * C = 60.6) : C = 50.5 :=
sorry

end lunch_cost_before_tip_l573_573626


namespace three_digit_numbers_with_2_or_5_and_odd_end_count_l573_573897

theorem three_digit_numbers_with_2_or_5_and_odd_end_count :
  ∃ count : ℕ, count = 676 ∧
    (∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 →
      (∃ d : ℕ, d ∈ [2, 5] ∧ (n = d ∨ n / 10 = d ∨ n / 100 = d)) ∧
      (∃ m : ℕ, m ∈ [1, 3, 7, 9] ∧ n % 10 = m)) :=
begin
  use 676,
  split,
  { refl },
  { intros n h1 h2,
    sorry
  }
end

end three_digit_numbers_with_2_or_5_and_odd_end_count_l573_573897


namespace collinear_condition_perpendicular_condition_l573_573478

namespace Vectors

-- Definitions for vectors a and b
def a : ℝ × ℝ := (4, -2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinear condition
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

-- Perpendicular condition
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Proof statement for collinear condition
theorem collinear_condition (x : ℝ) (h : collinear a (b x)) : x = -2 := sorry

-- Proof statement for perpendicular condition
theorem perpendicular_condition (x : ℝ) (h : perpendicular a (b x)) : x = 1 / 2 := sorry

end Vectors

end collinear_condition_perpendicular_condition_l573_573478


namespace num_positive_integers_satisfying_l573_573799

theorem num_positive_integers_satisfying (n : ℕ) :
  (∑ k in (finset.range 25), (if (even (2 + 4 * k)) then 1 else 0) = 24) :=
sorry

end num_positive_integers_satisfying_l573_573799


namespace sequence_inequality_l573_573434

def sequence_r : ℕ → ℕ
| 0 => 2
| (n+1) => (sequence_r n).recOn (λ m IH, sequence_r 0 * b.recOn (λ m IH, IH * sequence_r m) + 1)

theorem sequence_inequality (a : ∀ n, ℕ) (h : ∑ i in finrange n, (1/(a i : ℝ)) < 1) :
  ∑ i in finrange n, (1/(a i : ℝ)) ≤ ∑ i in finrange n, (1/(sequence_r i : ℝ)) := by
  sorry

end sequence_inequality_l573_573434


namespace length_of_AB_on_hyperbola_l573_573885

theorem length_of_AB_on_hyperbola :
  ∀ (x y: ℝ),
  (∀ A B: ℝ × ℝ,
  (A.2 = (Real.sqrt 3 / 3) * (A.1 - 3)) ∧
  (B.2 = (Real.sqrt 3 / 3) * (B.1 - 3)) ∧
  (A.1 ^ 2 / 3 - A.2 ^ 2 / 6 = 1) ∧
  (B.1 ^ 2 / 3 - B.2 ^ 2 / 6 = 1) →
  Real.dist A B = (16 / 5) * Real.sqrt 3) :=
begin
  sorry 
end

end length_of_AB_on_hyperbola_l573_573885


namespace add_fractions_l573_573639

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l573_573639


namespace inverse_of_A_l573_573407

def A : Matrix (Fin 2) (Fin 2) ℚ := ![![4, -3], ![-2, 1]]
def A_inv_expected : Matrix (Fin 2) (Fin 2) ℚ := ![[-1/2, -3/2], [-1, -2]]

theorem inverse_of_A : (A.det ≠ 0) → (A⁻¹ = A_inv_expected) :=
by
  intro h
  have det_A : A.det = -2 :=
    by sorry  -- The calculation step is skipped
  exact sorry  -- The proof that A⁻¹ actually equals A_inv_expected

end inverse_of_A_l573_573407


namespace tangent_circles_proof_l573_573950

noncomputable def acute_non_isosceles_triangle := 
∀ (A B C O T D E M K : Point), 
  is_acute △ABC → 
  circ.center △ABC = O → 
  tangent.line (circumcircle △ABC) B ∩ tangent.line (circumcircle △ABC) C = T → 
  (line.through T ∩ segment AB = D) ∧ (line.through T ∩ ray CA = E) → 
  midpoint D E = M → 
  line.intersect MA (circumcircle △ABC) = K → 
  tangent.to (circular_arc MKT) (circumcircle △ABC).

theorem tangent_circles_proof :
  acute_non_isosceles_triangle := 
by
  sorry

end tangent_circles_proof_l573_573950


namespace swordtail_food_l573_573949

def num_goldfish : ℕ := 2
def food_per_goldfish : ℝ := 1
def num_guppies : ℕ := 8
def food_per_guppy : ℝ := 0.5
def total_food : ℝ := 12
def num_swordtails : ℕ := 3

theorem swordtail_food : 
  let total_goldfish_food := num_goldfish * food_per_goldfish in
  let total_guppy_food := num_guppies * food_per_guppy in
  let remaining_food := total_food - (total_goldfish_food + total_guppy_food) in
  remaining_food / num_swordtails = 2 :=
by
  sorry

end swordtail_food_l573_573949


namespace circumcenter_of_AMN_on_AC_l573_573214

open Real EuclideanGeometry

-- Define the points and the square
structure Square (A B C D : Point) : Prop :=
  (AB_eq_CD : dist A B = dist C D)
  (BC_eq_DA : dist B C = dist D A)
  (AB_perpendicular_BC : ∠ABC = π / 2)
  (BC_perpendicular_CD : ∠BCD = π / 2)
  (CD_perpendicular_DA : ∠CDA = π / 2)
  (DA_perpendicular_AB : ∠DAB = π / 2)

-- Define the given conditions
variables {A B C D M N : Point}
variable (sq : Square A B C D)
variable (M_on_BC : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = B + t • (C - B))
variable (N_on_CD : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ N = C + t • (D - C))
variable (angle_MAN_eq_45 : ∠ A M N = π / 4)

-- Prove the goal
theorem circumcenter_of_AMN_on_AC :
  let O := circumcenter A M N 
  in ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ O = A + t • (C - A) :=
sorry

end circumcenter_of_AMN_on_AC_l573_573214


namespace product_of_consecutive_even_numbers_divisible_by_8_l573_573215

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 
  8 ∣ (2 * n) * (2 * n + 2) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l573_573215


namespace batsman_avg_l573_573667

variable (A : ℕ) -- The batting average in 46 innings

-- Given conditions
variables (highest lowest : ℕ)
variables (diff : ℕ) (avg_excl : ℕ) (num_excl : ℕ)

namespace cricket

-- Define the given values
def highest_score := 225
def difference := 150
def avg_excluding := 58
def num_excluding := 44

-- Calculate the lowest score
def lowest_score := highest_score - difference

-- Calculate the total runs in 44 innings excluding highest and lowest scores
def total_run_excluded := avg_excluding * num_excluding

-- Calculate the total runs in 46 innings
def total_runs := total_run_excluded + highest_score + lowest_score

-- Define the equation relating the average to everything else
def batting_avg_eq : Prop :=
  total_runs = 46 * A

-- Prove that the batting average A is 62 given the conditions
theorem batsman_avg :
  A = 62 :=
  by
    sorry

end cricket

end batsman_avg_l573_573667


namespace fraction_addition_l573_573644

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l573_573644


namespace mass_percentage_H_in_NH4I_is_correct_l573_573777

noncomputable def molar_mass_NH4I : ℝ := 1 * 14.01 + 4 * 1.01 + 1 * 126.90

noncomputable def mass_H_in_NH4I : ℝ := 4 * 1.01

noncomputable def mass_percentage_H_in_NH4I : ℝ := (mass_H_in_NH4I / molar_mass_NH4I) * 100

theorem mass_percentage_H_in_NH4I_is_correct :
  abs (mass_percentage_H_in_NH4I - 2.79) < 0.01 := by
  sorry

end mass_percentage_H_in_NH4I_is_correct_l573_573777


namespace values_for_a_l573_573595

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2 * x else 1 / x

theorem values_for_a (a : ℝ) : f 1 + f a = -2 ↔ a = -1 ∨ a = 1 := by
  sorry

#eval values_for_a 1 -- should be true since 1 is in the set
#eval values_for_a (-1) -- should be true since -1 is in the set
#eval values_for_a 2 -- should be false since 2 is not in the set

end values_for_a_l573_573595


namespace sum_of_integers_l573_573411

theorem sum_of_integers (n : ℕ) (h1 : 1.5 * n - 3 > 7.5) (h2 : n <= 20) : 
  ∃ s, s = ∑ i in finset.range ((20 - 8) + 1), (i + 8) ∧ s = 182 :=
by
  sorry

end sum_of_integers_l573_573411


namespace number_of_solutions_l573_573789
-- Importing the necessary library

-- Define the condition for the product of the sequence to be negative
noncomputable def condition (n : ℕ) :=
  (n > 0) ∧ (Finset.prod (Finset.range 49) (λ k, (n - 2 * (k + 1)) : ℤ) < 0)

-- State theorem corresponding to the mathematically equivalent proof problem
theorem number_of_solutions : 
  finset.filter condition (finset.range (99)).card = 24 :=
sorry

end number_of_solutions_l573_573789


namespace find_fx_l573_573904

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = 2 * x^2 + 1) : ∀ x : ℝ, f x = 2 * x - 1 := 
sorry

end find_fx_l573_573904


namespace avg_speed_is_correct_inst_speed_at_t_is_correct_l573_573332

-- Define the displacement function.
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the interval
def interval := (1, 1.1)

-- Define the average speed over the interval
def avg_speed : ℝ :=
  (displacement interval.2 - displacement interval.1) / (interval.2 - interval.1)

-- Define the instantaneous speed as the derivative of the displacement function
def inst_speed (t : ℝ) : ℝ := 6 * t^2

-- Define the times we are interested in
def t := 1

-- Prove that the average speed over [1, 1.1] is 6.62 m/s
theorem avg_speed_is_correct : avg_speed = 6.62 := by sorry

-- Prove that the instantaneous speed at t=1 is 6 m/s
theorem inst_speed_at_t_is_correct : inst_speed t = 6 := by sorry

end avg_speed_is_correct_inst_speed_at_t_is_correct_l573_573332


namespace monotonic_decreasing_range_of_a_l573_573489

-- Define the given function
def f (a x : ℝ) := a * x^2 - 3 * x + 4

-- State the proof problem
theorem monotonic_decreasing_range_of_a (a : ℝ) : (∀ x : ℝ, x < 6 → deriv (f a) x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
sorry

end monotonic_decreasing_range_of_a_l573_573489


namespace function_passing_point_l573_573463

variable (f : ℝ → ℝ)

theorem function_passing_point (h : f 1 = 0) : f (0 + 1) + 1 = 1 := by
  calc f (0 + 1) + 1 = f 1 + 1 := by rfl
                  ... = 0 + 1 := by rw [h]
                  ... = 1 := by rfl

#check function_passing_point

end function_passing_point_l573_573463


namespace range_of_m_l573_573123

noncomputable def unique_zero_point (m : ℝ) : Prop :=
  ∀ x : ℝ, m * (1/4)^x - (1/2)^x + 1 = 0 → ∀ x' : ℝ, m * (1/4)^x' - (1/2)^x' + 1 = 0 → x = x'

theorem range_of_m (m : ℝ) : unique_zero_point m → (m ≤ 0 ∨ m = 1/4) :=
sorry

end range_of_m_l573_573123


namespace nth_term_of_sequence_l573_573423

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (n : ℕ)

-- Given condition
def sum_of_first_terms := ∀ n : ℕ, S n = n^2

-- Definition of sequence terms in terms of sum
def sequence_term := ∀ n : ℕ, n > 0 → a n = S n - S (n - 1)

-- To prove: the nth term of the sequence
theorem nth_term_of_sequence (h : sum_of_first_terms S) (h_term : sequence_term a S) : 
  ∀ n : ℕ, n > 0 → a n = 2 * n - 1 := 
by 
  intros n hn
  cases n
  case.zero =>
    contradiction
  case.succ n =>
    sorry

end nth_term_of_sequence_l573_573423


namespace oliver_spent_amount_l573_573210

theorem oliver_spent_amount :
  ∀ (S : ℕ), (33 - S + 32 = 61) → S = 4 :=
by
  sorry

end oliver_spent_amount_l573_573210


namespace expected_value_after_50_centuries_l573_573529

-- Define the initial conditions
def initial_amount : ℝ := 0.50
def probability_double : ℝ := 0.5
def probability_reset : ℝ := 0.5
def reset_amount : ℝ := 0.50

-- Define the recurrence relation for the expected value
noncomputable def expected_value : ℕ → ℝ
| 0       := initial_amount
| (n + 1) := (probability_double * 2 * expected_value n) + (probability_reset * reset_amount)

-- Statement of the problem
theorem expected_value_after_50_centuries : expected_value 50 = 13.00 := 
by {
  -- The proof would go here, but we use sorry to indicate the proof is omitted.
  sorry
}

end expected_value_after_50_centuries_l573_573529


namespace sin_sum_ge_sin_sum_l573_573219

-- Define the conditions with appropriate constraints
variables {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxπ : x ≤ π) (hyπ : y ≤ π) (hzπ : z ≤ π)

-- State the theorem to prove
theorem sin_sum_ge_sin_sum (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (hxπ : x ≤ π) (hyπ : y ≤ π) (hzπ : z ≤ π) : 
  (sin x) + (sin y) + (sin z) ≥ sin (x + y + z) := 
sorry

end sin_sum_ge_sin_sum_l573_573219


namespace calculate_ratio_l573_573878

namespace Mathproof

theorem calculate_ratio (a x y : ℝ) (h1 : a ≠ x) (h2 : a ≠ y) (h3 : x ≠ y) 
(h4 : sqrt(a * (x - a)) + sqrt(a * (y - a)) = sqrt(x - a) - sqrt(a - y)) : 
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1 / 3 :=
by 
  sorry

end Mathproof

end calculate_ratio_l573_573878


namespace probability_calculations_l573_573822

-- Define the number of students
def total_students : ℕ := 2006

-- Number of students eliminated in the first step
def eliminated_students : ℕ := 6

-- Number of students remaining after elimination
def remaining_students : ℕ := total_students - eliminated_students

-- Number of students to be selected in the second step
def selected_students : ℕ := 50

-- Calculate the probability of a specific student being eliminated
def elimination_probability := (6 : ℚ) / total_students

-- Calculate the probability of a specific student being selected from the remaining students
def selection_probability := (50 : ℚ) / remaining_students

-- The theorem to prove our equivalent proof problem
theorem probability_calculations :
  elimination_probability = (3 : ℚ) / 1003 ∧
  selection_probability = (25 : ℚ) / 1003 :=
by
  sorry

end probability_calculations_l573_573822


namespace man_l573_573690

theorem man's_present_age (P : ℝ) 
  (h1 : P = (4/5) * P + 10)
  (h2 : P = (3/2.5) * P - 10) :
  P = 50 :=
sorry

end man_l573_573690


namespace mul_binom_expansion_l573_573376

variable (a : ℝ)

theorem mul_binom_expansion : (a + 1) * (a - 1) = a^2 - 1 :=
by
  sorry

end mul_binom_expansion_l573_573376


namespace length_of_A_l573_573172

structure Point := (x : ℝ) (y : ℝ)

noncomputable def length (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

theorem length_of_A'B' (A A' B B' C : Point) 
    (hA : A = ⟨0, 6⟩)
    (hB : B = ⟨0, 10⟩)
    (hC : C = ⟨3, 6⟩)
    (hA'_line : A'.y = A'.x)
    (hB'_line : B'.y = B'.x) 
    (hA'C : ∃ m b, ((C.y = m * C.x + b) ∧ (C.y = b) ∧ (A.y = b))) 
    (hB'C : ∃ m b, ((C.y = m * C.x + b) ∧ (B.y = m * B.x + b)))
    : length A' B' = (12 / 7) * Real.sqrt 2 :=
by
  sorry

end length_of_A_l573_573172


namespace smallest_lcm_for_triplets_l573_573556

def triplet_count_27000 (a b c : ℕ) : Prop :=
  (nat.gcd (nat.gcd a b) c = 91) ∧ 
  (nat.lcm (nat.lcm a b) c = 17836000)

theorem smallest_lcm_for_triplets :
  ∃ (a b c : ℕ), triplet_count_27000 a b c ∧ 
  ∀ (n : ℕ), (∀ (x y z : ℕ), (nat.gcd (nat.gcd x y) z = 91 ∧ nat.lcm (nat.lcm x y) z = n) → n ≥ 17836000) :=
begin
  sorry
end

end smallest_lcm_for_triplets_l573_573556


namespace gcd_lcm_product_l573_573732

theorem gcd_lcm_product (a b: ℕ) (h1 : a = 36) (h2 : b = 210) :
  Nat.gcd a b * Nat.lcm a b = 7560 := 
by 
  sorry

end gcd_lcm_product_l573_573732


namespace alex_silver_tokens_l573_573352

open Nat

theorem alex_silver_tokens :
  ∃ (x y : ℕ), 100 - 3 * x + 2 * y < 3 ∧ 65 + 2 * x - 4 * y < 4 ∧ x + y = 65 :=
begin
  sorry
end

end alex_silver_tokens_l573_573352


namespace neither_4_nice_nor_6_nice_less_or_equal_500_l573_573752

def is_4_nice (N : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (a ^ 4).nat_divisors.length = N

def is_6_nice (N : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (a ^ 6).nat_divisors.length = N

def count_not_4_or_6_nice : ℕ :=
  let upto_500 := finset.range 500
  let count_4_nice := finset.filter (λn, is_4_nice n) upto_500
  let count_6_nice := finset.filter (λn, is_6_nice n) upto_500
  let count_both := finset.filter (λn, is_4_nice n ∧ is_6_nice n) upto_500
  500 - (count_4_nice.card + count_6_nice.card - count_both.card)

theorem neither_4_nice_nor_6_nice_less_or_equal_500 :
  count_not_4_or_6_nice = 333 :=
by {
  sorry
}

end neither_4_nice_nor_6_nice_less_or_equal_500_l573_573752


namespace add_fractions_l573_573638

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l573_573638


namespace container_initial_percentage_l573_573668

noncomputable def initial_percentage (P : ℝ) : Prop :=
  (P / 100) * 40 + 18 = 30

theorem container_initial_percentage :
  ∃ P : ℝ, initial_percentage P ∧ P = 30 :=
by
  use 30
  split
  · unfold initial_percentage -- This uses the definition and checks the stated condition
    sorry -- Here, we would normally provide the proof, but it is skipped as instructed.
  · rfl -- Reflexivity shows P = 30 based on the value used

end container_initial_percentage_l573_573668


namespace initial_butterfat_percentage_l573_573984

theorem initial_butterfat_percentage (P : ℝ) :
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  initial_butterfat - removed_butterfat = desired_butterfat
→ P = 4 :=
by
  intros
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  sorry

end initial_butterfat_percentage_l573_573984


namespace complex_argument_problem_l573_573608

theorem complex_argument_problem (z : ℂ) (α : ℝ) (h : arg z = α) :
  (∃ w : ℂ, w ∈ {z : ℂ | ∃ z, arg z = α ∧ z = w^2} ∧ (arg w = 2*α ∨ arg w = -2*α ∨ arg w = α)) -> false :=
by
  sorry

end complex_argument_problem_l573_573608


namespace product_of_cosines_value_l573_573386

noncomputable def product_of_cosines : ℝ :=
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) *
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12))

theorem product_of_cosines_value :
  product_of_cosines = 1 / 16 :=
by
  sorry

end product_of_cosines_value_l573_573386


namespace sqrt_combination_l573_573719

theorem sqrt_combination :
  ∃ (k : ℝ), sqrt (1 / 8) = k * sqrt 2 :=
sorry

end sqrt_combination_l573_573719


namespace find_solutions_l573_573967

theorem find_solutions {a : ℕ → ℕ} {n : ℕ} (h0 : a 0 > 1) (hn : ∀ i, i ≤ n → a i > 1)
  (hdesc : ∀ i j, i < j → a i > a j) 
  (heqn : (∑ i in Finset.range n, 1 - 1 / a (i + 1)) = 2 * (1 - 1 / a 0)) : 
  (a 0 = 24 ∧ a 1 = 4 ∧ a 2 = 3 ∧ a 3 = 2) ∨ 
  (a 0 = 60 ∧ a 1 = 5 ∧ a 2 = 3 ∧ a 3 = 2) := 
sorry

end find_solutions_l573_573967


namespace compound_ratio_l573_573383

theorem compound_ratio (total_weight A_weight B_weight : ℕ) (h_total_weight: total_weight = 222) (h_B_weight: B_weight = 185) (h_A_weight: A_weight = total_weight - B_weight) : (A_weight : B_weight) = (1 : 5) :=
  sorry

end compound_ratio_l573_573383


namespace binomial_sum_pattern_l573_573561

theorem binomial_sum_pattern (n : ℕ) (h : 0 < n) : 
  (finset.sum (finset.range n) (λ k, nat.choose (2*n - 1) k)) = 4^(n - 1) :=
by 
  sorry

end binomial_sum_pattern_l573_573561


namespace choose_agency_l573_573684

variables (a : ℝ) (x : ℕ)

def cost_agency_A (a : ℝ) (x : ℕ) : ℝ :=
  a + 0.55 * a * x

def cost_agency_B (a : ℝ) (x : ℕ) : ℝ :=
  0.75 * (x + 1) * a

theorem choose_agency (a : ℝ) (x : ℕ) : if (x = 1) then 
                                            (cost_agency_B a x ≤ cost_agency_A a x)
                                         else if (x ≥ 2) then 
                                            (cost_agency_A a x ≤ cost_agency_B a x)
                                         else
                                            true :=
by
  sorry

end choose_agency_l573_573684


namespace roots_magnitude_order_l573_573476

theorem roots_magnitude_order (m : ℝ) (a b c d : ℝ)
  (h1 : m > 0)
  (h2 : a ^ 2 - m * a - 1 = 0)
  (h3 : b ^ 2 - m * b - 1 = 0)
  (h4 : c ^ 2 + m * c - 1 = 0)
  (h5 : d ^ 2 + m * d - 1 = 0)
  (ha_pos : a > 0) (hb_neg : b < 0)
  (hc_pos : c > 0) (hd_neg : d < 0) :
  |a| > |c| ∧ |c| > |b| ∧ |b| > |d| :=
sorry

end roots_magnitude_order_l573_573476


namespace percentage_pure_ghee_l573_573930

theorem percentage_pure_ghee 
  (Q : ℝ) 
  (initial_ghee : Q = 10) 
  (vanaspati_percentage : 0.40 * Q) 
  (added_ghee : 10) 
  (new_total : Q + 10) 
  (new_vanaspati_percentage : 0.20 * (Q + 10) = 0.40 * Q) : 
  (0.60 * Q / Q * 100 = 60) := 
by 
  sorry

end percentage_pure_ghee_l573_573930


namespace joan_apples_l573_573944

theorem joan_apples (initial_apples : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : 
  initial_apples = 43 ∧ given_to_melanie = 27 ∧ given_to_sarah = 11 → (initial_apples - given_to_melanie - given_to_sarah) = 5 := 
by
  sorry

end joan_apples_l573_573944


namespace minimum_ribbon_length_l573_573623

def side_length : ℚ := 13 / 12

def perimeter_of_equilateral_triangle (a : ℚ) : ℚ := 3 * a

theorem minimum_ribbon_length :
  perimeter_of_equilateral_triangle side_length = 3.25 := 
by
  sorry

end minimum_ribbon_length_l573_573623


namespace tangent_slope_at_1_3_l573_573262

def curve (x : ℝ) : ℝ := x^3 - 2 * x + 4

def slope_of_tangent_at (x : ℝ) : ℝ := deriv curve x

theorem tangent_slope_at_1_3 : slope_of_tangent_at 1 = 1 :=
by
  sorry

end tangent_slope_at_1_3_l573_573262


namespace high_school_sampling_problem_l573_573323

theorem high_school_sampling_problem :
  let first_year_classes := 20
  let first_year_students_per_class := 50
  let first_year_total_students := first_year_classes * first_year_students_per_class
  let second_year_classes := 24
  let second_year_students_per_class := 45
  let second_year_total_students := second_year_classes * second_year_students_per_class
  let total_students := first_year_total_students + second_year_total_students
  let survey_students := 208
  let first_year_sample := (first_year_total_students * survey_students) / total_students
  let second_year_sample := (second_year_total_students * survey_students) / total_students
  let A_selected_probability := first_year_sample / first_year_total_students
  let B_selected_probability := second_year_sample / second_year_total_students
  (survey_students = 208) →
  (first_year_sample = 100) →
  (second_year_sample = 108) →
  (A_selected_probability = 1 / 10) →
  (B_selected_probability = 1 / 10) →
  (A_selected_probability = B_selected_probability) →
  (student_A_in_first_year : true) →
  (student_B_in_second_year : true) →
  true :=
  by sorry

end high_school_sampling_problem_l573_573323


namespace number_of_ways_to_place_coins_l573_573137

theorem number_of_ways_to_place_coins :
  (nat.choose 7 2) = 21 :=
by
  sorry

end number_of_ways_to_place_coins_l573_573137


namespace greatest_50_podpyirayushchee_X_l573_573970

noncomputable def maximal_podpyirayushchee := 0.01

theorem greatest_50_podpyirayushchee_X :
  ∀ (a : Fin 50 → ℝ), (∑ i in Finset.range 50, a i) ∈ Int → 
  (∃ i, | a i - (1/2) | ≥ maximal_podpyirayushchee) := sorry

end greatest_50_podpyirayushchee_X_l573_573970


namespace jordan_purchase_total_rounded_l573_573167

theorem jordan_purchase_total_rounded :
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2 -- rounded value of p1
  let r2 := 7 -- rounded value of p2
  let r3 := 11 -- rounded value of p3
  r1 + r2 + r3 = 20 :=
by
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2
  let r2 := 7
  let r3 := 11
  show r1 + r2 + r3 = 20
  sorry

end jordan_purchase_total_rounded_l573_573167


namespace altitude_eq_line_eq_with_equal_intercepts_l573_573663

open Real

-- First proof problem: Line containing the altitude from point C to side BC
theorem altitude_eq {A B C : Point} (hA : A = (0, 5)) (hB : B = (1, -2)) (hC : C = (-6, 4)) :
  ∃ (l : Line), l.equation = 7 * x - 6 * y + 30 = 0 := 
sorry

-- Second proof problem: Line with equal intercepts
theorem line_eq_with_equal_intercepts (a : ℝ) 
  (l : Line) (hl : l.equation = (a - 1) * x + y - 2 - a = 0)
  (hx_intercept : x_intercept l = intercept)
  (hy_intercept : y_intercept l = intercept) :
  ((a = -2 ∧ l.equation = 3 * x - y = 0) ∨ (a = 2 ∧ l.equation = x + y - 4 = 0)) :=
sorry

end altitude_eq_line_eq_with_equal_intercepts_l573_573663


namespace original_paint_intensity_l573_573235

theorem original_paint_intensity
  (I : ℝ) -- Original intensity of the red paint
  (f : ℝ) -- Fraction of the original paint replaced
  (new_intensity : ℝ) -- Intensity of the new paint
  (replacement_intensity : ℝ) -- Intensity of the replacement red paint
  (hf : f = 2 / 3)
  (hreplacement_intensity : replacement_intensity = 0.30)
  (hnew_intensity : new_intensity = 0.40)
  : I = 0.60 := 
sorry

end original_paint_intensity_l573_573235


namespace Xiaofang_English_score_l573_573926

/-- Given the conditions about the average scores of Xiaofang's subjects:
  1. The average score for 4 subjects is 88.
  2. The average score for the first 2 subjects is 93.
  3. The average score for the last 3 subjects is 87.
Prove that Xiaofang's English test score is 95. -/
theorem Xiaofang_English_score
    (L M E S : ℝ)
    (h1 : (L + M + E + S) / 4 = 88)
    (h2 : (L + M) / 2 = 93)
    (h3 : (M + E + S) / 3 = 87) :
    E = 95 :=
by
  sorry

end Xiaofang_English_score_l573_573926


namespace even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l573_573582

-- Definitions for the conditions in Lean 4
def regular_n_gon (n : ℕ) := true -- Dummy definition; actual geometric properties not needed for statement

def connected_path_visits_each_vertex_once (n : ℕ) := true -- Dummy definition; actual path properties not needed for statement

def parallel_pair (i j p q : ℕ) (n : ℕ) : Prop := (i + j) % n = (p + q) % n

-- Statements for part (a) and (b)

theorem even_n_has_parallel_pair (n : ℕ) (h_even : n % 2 = 0) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n := 
sorry

theorem odd_n_cannot_have_exactly_one_parallel_pair (n : ℕ) (h_odd : n % 2 = 1) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ¬∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n ∧ 
  (∀ (i' j' p' q' : ℕ), (i' ≠ p' ∨ j' ≠ q') → ¬parallel_pair i' j' p' q' n) := 
sorry

end even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l573_573582


namespace determine_occupations_l573_573922

structure person :=
(name : String)

def Juci : person := ⟨"Juci"⟩
def Magda : person := ⟨"Magda"⟩
def Mária : person := ⟨"Mária"⟩
def Margit : person := ⟨"Margit"⟩

def János : person := ⟨"János"⟩
def Jenő : person := ⟨"Jenő"⟩
def József : person := ⟨"József"⟩
def Mihály : person := ⟨"Mihály"⟩

inductive occupation
| carpenter
| judge
| locksmith
| doctor

open occupation

structure group :=
(members : list person)
(rel : person → person → Prop)
(occu : person → occupation)

def example_group : group :=
{ members := [Juci, Magda, Mária, Margit, János, Jenő, József, Mihály],
  rel := sorry,  -- Relations among persons based on the clues
  occu := sorry  -- Occupations of the persons based on the clues
}

theorem determine_occupations (g : group) : 
  g.occu János = judge ∧ 
  g.occu Mihály = doctor ∧ 
  g.occu József = locksmith ∧ 
  g.occu Jenő = carpenter :=
by {
  sorry  -- Proof steps according to the conditions provided
}

end determine_occupations_l573_573922


namespace binary_rep_of_17_l573_573404

theorem binary_rep_of_17 : Nat.toDigits 2 17 = [1, 0, 0, 0, 1] :=
by
  sorry

end binary_rep_of_17_l573_573404


namespace maggie_fraction_caught_l573_573203

theorem maggie_fraction_caught :
  let total_goldfish := 100
  let allowed_to_take_home := total_goldfish / 2
  let remaining_goldfish_to_catch := 20
  let goldfish_caught := allowed_to_take_home - remaining_goldfish_to_catch
  (goldfish_caught / allowed_to_take_home : ℚ) = 3 / 5 :=
by
  sorry

end maggie_fraction_caught_l573_573203


namespace man_swims_upstream_l573_573688

def swimming_problem (distance_downstream time_hours speed_still_water down_dist up_time : ℝ) (result: ℝ) : Prop :=
  ∃ (v : ℝ), 
    5 + v = distance_downstream / time_hours ∧ 
    result = (speed_still_water - v) * up_time

theorem man_swims_upstream :
    swimming_problem 18 3 5 18 3 12 :=
by
  unfold swimming_problem
  use 1
  split
  · calc 5 + 1 = 6 : by norm_num
         ... = 18 / 3 : by norm_num
  · calc  (5 - 1) * 3 = 4 * 3 : by norm_num
                ...  = 12 : by norm_num

end man_swims_upstream_l573_573688


namespace find_k_l573_573399

theorem find_k : ∃ k : ℕ, 7! * 13! = 20 * k! ∧ k = 14 :=
by 
  exists 14
  split
  {
    calc 7! * 13! = 20 * 14! : sorry
  }
  {
    reflexivity
  }

end find_k_l573_573399


namespace simplify_and_evaluate_l573_573231

theorem simplify_and_evaluate :
  let a := (-1: ℝ) / 3
  let b := (-3: ℝ)
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 :=
by
  have a_def : a = (-1: ℝ) / 3 := rfl
  have b_def : b = (-3: ℝ) := rfl
  sorry

end simplify_and_evaluate_l573_573231


namespace minimum_value_y_l573_573710

theorem minimum_value_y (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 200) : y = 8 :=
by
  sorry

end minimum_value_y_l573_573710


namespace smallest_positive_period_l573_573469

def f (x : ℝ) : ℝ := 2 * Real.sin (-2 * x + Real.pi / 4)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

end smallest_positive_period_l573_573469


namespace total_flour_l573_573207

def cups_of_flour (flour_added : ℕ) (flour_needed : ℕ) : ℕ :=
  flour_added + flour_needed

theorem total_flour :
  ∀ (flour_added flour_needed : ℕ), flour_added = 3 → flour_needed = 6 → cups_of_flour flour_added flour_needed = 9 :=
by 
  intros flour_added flour_needed h_added h_needed
  rw [h_added, h_needed]
  rfl

end total_flour_l573_573207


namespace sum_ge_six_l573_573566

theorem sum_ge_six (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 :=
by
  sorry

end sum_ge_six_l573_573566


namespace circles_intersect_l573_573395

def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

theorem circles_intersect : 
  (∀ x y : ℝ, circle1 x y ↔ (x + 1)^2 + (y + 1)^2 = 4) →
  (∀ x y : ℝ, circle2 x y ↔ (x - 2)^2 + (y - 1)^2 = 4) →
  let d := Real.sqrt 13 in
  let r1 := 2 in
  let r2 := 2 in
  r1 - r2 < d ∧ d < r1 + r2 →
  "intersect" = "intersect" :=
by
  sorry

end circles_intersect_l573_573395


namespace max_value_of_sequence_l573_573477

def sequence (n : ℕ) : ℤ :=
  -2 * n^2 + 29 * n + 3

theorem max_value_of_sequence : ∃ n : ℕ, sequence n = 108 ∧ ∀ m : ℕ, sequence m ≤ 108 :=
by
  sorry

end max_value_of_sequence_l573_573477


namespace part1_part2_l573_573834

namespace Proof

variables {f : ℝ → ℝ}
variables {x y t : ℝ}

-- Conditions
hypothesis (H1 : ∀ x y ∈ ℝ, f(x) + f(y) = 2 + f(x+y))
hypothesis (H2 : f(3) = 5)
hypothesis (H3 : ∀ x y ∈ ℝ, x < y → f(x) < f(y))

-- Part 1: Prove that f(1) + f(-1) = 4
theorem part1 : f(1) + f(-1) = 4 :=
sorry

-- Part 2: Prove that there exists a unique real number t = 0 such that for any x ∈ (0, 1), f(x^2 + 2t^2 * x) < 3
theorem part2 : ∃! t, (t = 0) ∧ (∀ x ∈ Ioo 0 1, f(x^2 + 2*t^2*x) < 3) :=
sorry

end Proof

end part1_part2_l573_573834


namespace bert_total_stamps_l573_573728

theorem bert_total_stamps (bought_stamps : ℕ) (half_stamps_before : ℕ) (total_stamps_after : ℕ) :
  (bought_stamps = 300) ∧ (half_stamps_before = bought_stamps / 2) → (total_stamps_after = half_stamps_before + bought_stamps) → (total_stamps_after = 450) :=
by
  sorry

end bert_total_stamps_l573_573728


namespace trajectory_equation_of_P_l573_573071

theorem trajectory_equation_of_P :
  ∀ (x y : ℝ),
  (x ≠ -2) → (x ≠ 2) → (x ≠ 0) → (y ≠ 0) →
  (let k_AP := y / (x + 2) in let k_BP := y / (x - 2) in k_AP * k_BP = -1 / 4) →
  (x^2 / 4 + y^2 = 1) :=
by
  intros x y hx1 hx2 hx3 hy cond
  sorry

end trajectory_equation_of_P_l573_573071


namespace Robin_needs_to_buy_more_bottles_l573_573988

/-- Robin wants to drink exactly nine bottles of water each day.
    She initially bought six hundred seventeen bottles.
    Prove that she will need to buy 4 more bottles on the last day
    to meet her goal of drinking exactly nine bottles each day. -/
theorem Robin_needs_to_buy_more_bottles :
  ∀ total_bottles bottles_per_day : ℕ, total_bottles = 617 → bottles_per_day = 9 → 
  ∃ extra_bottles : ℕ, (617 % 9) + extra_bottles = 9 ∧ extra_bottles = 4 :=
by
  sorry

end Robin_needs_to_buy_more_bottles_l573_573988


namespace ratio_of_areas_l573_573159

theorem ratio_of_areas
  (XY XZ YZ : ℝ) (h1 : XY = 20) (h2 : XZ = 30) (h3 : YZ = 35)
  (P : Type) (X Y Z : P) (XP : P → P → Prop) (angle_bisector : XP X P ∧ XP P Z) :
  let YP := XY * (YZ - ZP) / YZ,
      ZP := XZ * (YZ - YP) / YZ
  in
  ∃ r : ℝ, r = 2 / 3 :=
by
  sorry

end ratio_of_areas_l573_573159


namespace find_x_l573_573389

def bin_op (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  (p1.1 - 2 * p2.1, p1.2 + 2 * p2.2)

theorem find_x :
  ∃ x y : ℤ, 
  bin_op (2, -4) (1, -3) = bin_op (x, y) (2, 1) ∧ x = 4 :=
by
  sorry

end find_x_l573_573389


namespace number_of_ordered_pairs_l573_573015

theorem number_of_ordered_pairs : 
  let f : ℕ → ℕ := λ n, (if 2 ∣ n then n/2 else (n+1)/2)
  in (∑ a in (finset.range 50).filter (λ n, 1 ≤ n), f a) = 800 :=
by
  sorry

end number_of_ordered_pairs_l573_573015


namespace real_solutions_of_equation_l573_573025

theorem real_solutions_of_equation :
  ∃ n : ℕ, n ≈ 59 ∧ (∀ x : ℝ, x ∈ Icc (-50) 50 → (x / 50 = Real.sin x → x ∈ real_roots_of_eq))
    where
      real_roots_of_eq := {x : ℝ | x / 50 = Real.sin x} :=
sorry

end real_solutions_of_equation_l573_573025


namespace problem_abc_is_isosceles_l573_573087

open EuclideanGeometry
open scoped Real

variables {A B C O : Point}

theorem problem_abc_is_isosceles
    (h1 : (vectorFrom O B - vectorFrom O C) • (vectorFrom O B + vectorFrom O C - 2 • vectorFrom O A) = 0) :
  (dist A B = dist A C) :=
sorry

end problem_abc_is_isosceles_l573_573087


namespace find_S10_l573_573969

def sequence_sums (S : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 3 * S n - S (n + 1) - 1)

theorem find_S10 (S a : ℕ → ℚ) (h : sequence_sums S a) : S 10 = 513 / 2 :=
  sorry

end find_S10_l573_573969


namespace hexagon_triangle_area_l573_573530

theorem hexagon_triangle_area (ABCDEF : convex_hexagon) (h1 : area ABCDEF = 1) 
  (h2 : opposite_sides_parallel ABCDEF) :
  ∃ Δ1 Δ2 : triangle,
    (intersect_lines ABCDEF Δ1 Δ2) ∧ 
    (area Δ1 ≥ 3/2 ∨ area Δ2 ≥ 3/2) := 
begin 
  sorry
end

end hexagon_triangle_area_l573_573530


namespace find_angle_B_find_a_plus_c_l573_573127

variable (A B C a b c S : Real)

-- Conditions
axiom h1 : a = (1 / 2) * c + b * Real.cos C
axiom h2 : S = Real.sqrt 3
axiom h3 : b = Real.sqrt 13

-- Questions (Proving the answers from the problem)
theorem find_angle_B (hA : A = Real.pi - (B + C)) : 
  B = Real.pi / 3 := by
  sorry

theorem find_a_plus_c (hac : (1 / 2) * a * c * Real.sin (Real.pi / 3) = Real.sqrt 3) : 
  a + c = 5 := by
  sorry

end find_angle_B_find_a_plus_c_l573_573127


namespace positive_root_of_cubic_eq_l573_573029

theorem positive_root_of_cubic_eq : ∃ (x : ℝ), x > 0 ∧ x^3 - 4 * x^2 + x - 2 * real.sqrt 2 = 0 :=
begin
  use 2 + real.sqrt 2,
  split,
  { -- Prove that x is positive
    linarith [real.sqrt_pos.2 (by norm_num : (0 : ℝ) < 2)], },
  { -- Prove that x satisfies the equation
    field_simp [real.sqrt_two_mul_self],
    norm_num,
    ring_nf,
    simp [real.sqrt_two_mul_self],
    exact eq.refl 0 }
end

end positive_root_of_cubic_eq_l573_573029


namespace root_sum_greater_than_one_l573_573883

noncomputable def f (x a : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (x a : ℝ) : ℝ := (x^2 - x) * f x a

theorem root_sum_greater_than_one {a m x1 x2 : ℝ} (ha : a < 0)
  (h_eq_m : ∀ x, h x a = m) (hx1_root : h x1 a = m) (hx2_root : h x2 a = m)
  (hx1x2_distinct : x1 ≠ x2) :
  x1 + x2 > 1 := 
sorry

end root_sum_greater_than_one_l573_573883


namespace inequality_comparison_l573_573072

theorem inequality_comparison 
  (a b : ℝ) (x y : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) :
  a^2 + b^2 ≥ (x + y)^2 :=
sorry

end inequality_comparison_l573_573072


namespace scrap_cookie_radius_l573_573686

theorem scrap_cookie_radius : 
  let r_large : ℝ := 5
  let r_small : ℝ := 1
  let num_small : ℕ := 9
  let total_area_large := π * r_large^2
  let area_one_small := π * r_small^2
  let total_area_small := num_small * area_one_small
  let scrap_area := total_area_large - total_area_small
  let r_scrap := Real.sqrt(scrap_area / π)
  r_scrap = 4 :=
by
  let r_large : ℝ := 5
  let r_small : ℝ := 1
  let num_small : ℕ := 9
  let total_area_large := π * r_large^2
  let area_one_small := π * r_small^2
  let total_area_small := num_small * area_one_small
  let scrap_area := total_area_large - total_area_small
  let r_scrap := Real.sqrt(scrap_area / π)
  have h1 : total_area_large = 25 * π := by sorry
  have h2 : area_one_small = π := by sorry
  have h3 : total_area_small = 9 * π := by sorry
  have h4 : scrap_area = 16 * π := by sorry
  have h5 : r_scrap = 4 := by sorry
  exact h5

end scrap_cookie_radius_l573_573686


namespace f_2013_value_l573_573054

def f (a b : ℝ) : ℝ → ℝ := λ x, a * x ^ 3 + b * Real.sin x + 9

theorem f_2013_value (a b : ℝ) (h : f a b (-2013) = 7) : f a b 2013 = 11 :=
by 
  -- Add the proof here
  sorry

end f_2013_value_l573_573054


namespace Masha_gathers_5_mushrooms_l573_573045

def mushrooms_collected (B G : list ℕ) : ℕ :=
  B.sum + G.sum

def unique_girls (G : list ℕ) : Prop :=
  ∀ i j, i ≠ j → G.get_or_else i 0 ≠ G.get_or_else j 0

def at_least_43_mushrooms (B : list ℕ) : Prop :=
  ∀ i j k, B.get_or_else i 0 + B.get_or_else j 0 + B.get_or_else k 0 ≥ 43 

def within_5_times (A : list ℕ) : Prop :=
  ∀ i j, (max (A.get_or_else i 0) (A.get_or_else j 0)) ≤ 5 * (min (A.get_or_else i 0) (A.get_or_else j 0))

noncomputable def Masha_collects_most_mushrooms (G : list ℕ) : ℕ :=
  G.maximum_def 0

theorem Masha_gathers_5_mushrooms (B G : list ℕ) (h_size_B : B.length = 4) (h_size_G : G.length = 3) 
  (h_total : mushrooms_collected B G = 70) 
  (h_unique : unique_girls G) 
  (h_at_least_43 : at_least_43_mushrooms B) 
  (h_within_5times : within_5_times (B ++ G)) : 
  Masha_collects_most_mushrooms G = 5 := 
sorry

end Masha_gathers_5_mushrooms_l573_573045


namespace cut_square_into_rectangles_l573_573751

theorem cut_square_into_rectangles :
  ∃ x y : ℕ, 3 * x + 4 * y = 25 :=
by
  -- Given that the total area is 25 and we are using rectangles of areas 3 and 4
  -- we need to verify the existence of integers x and y such that 3x + 4y = 25
  existsi 7
  existsi 1
  sorry

end cut_square_into_rectangles_l573_573751


namespace geometric_progression_product_eq_l573_573743

noncomputable def geometric_progression_Q (b k : ℝ) (n : ℕ) : ℝ :=
  (b * (1 - k^n) / (1 - k)) * ((k^n - 1) / (b * (k - 1)))

theorem geometric_progression_product_eq (b k : ℝ) (n : ℕ) :
  n = 5 →
  let Q := b^5 * k^(5*2) in
  Q = (geometric_progression_Q b k n) ^ (n - 1) / 2 := 
by
  intros
  sorry

end geometric_progression_product_eq_l573_573743


namespace fraction_addition_l573_573643

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l573_573643


namespace complex_conjugate_problem_l573_573869

theorem complex_conjugate_problem (z : ℂ) (h1 : z + conj z = 2) (h2 : z * conj z = 2) :
  (conj z / z) ^ 2017 = ↑(pm i) := sorry

end complex_conjugate_problem_l573_573869


namespace square_root_of_4_is_pm2_l573_573614

theorem square_root_of_4_is_pm2 : ∃ (x : ℤ), x * x = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end square_root_of_4_is_pm2_l573_573614


namespace angle_between_vectors_l573_573895

variables {V : Type*} [inner_product_space ℝ V]

def vector_a : V := sorry
def vector_b : V := sorry

axiom norm_a : ∥vector_a∥ = real.sqrt 3
axiom norm_b : ∥vector_b∥ = 1
axiom norm_sum_ab : ∥vector_a + vector_b∥ = 2

theorem angle_between_vectors :
  let θ := real.angle vector_a vector_b in
  real.cos (real.angle (vector_a + vector_b) (vector_a - vector_b)) = 1 / 2 → θ = real.pi / 3 :=
sorry

end angle_between_vectors_l573_573895


namespace seq_ineq_l573_573836

noncomputable def seq (n : ℕ) : ℕ → ℝ
| 1 => 2
| n + 1 => (seq n * n + 1) / (n + 1)

theorem seq_ineq (t : ℝ) :
  (∀ a ∈ Set.Icc (-2 : ℝ) 2, ∀ n : ℕ, n > 0 →
    seq n * (n + 1) / (n + 1) < 2 * t^2 + a * t - 1) ↔
  t ∈ Set.Iic (-2) ∪ Set.Ici (2) :=
sorry

end seq_ineq_l573_573836


namespace percent_preferred_apples_l573_573241

def frequencies : List ℕ := [75, 80, 45, 100, 50]
def frequency_apples : ℕ := 75
def total_frequency : ℕ := frequency_apples + frequencies[1] + frequencies[2] + frequencies[3] + frequencies[4]

theorem percent_preferred_apples :
  (frequency_apples * 100) / total_frequency = 21 := by
  -- Proof steps go here
  sorry

end percent_preferred_apples_l573_573241


namespace cookies_second_round_l573_573202

theorem cookies_second_round 
  (cookies_first_round : ℕ) 
  (cookies_total : ℕ)
  (h_first_round : cookies_first_round = 34)
  (h_total : cookies_total = 61) : 
  cookies_total - cookies_first_round = 27 :=
by
  rw [h_first_round, h_total]
  norm_num

end cookies_second_round_l573_573202


namespace appropriate_sampling_methods_l573_573680

def sales_outlets := {A := 150, B := 120, C := 180, D := 150}
def total_sales_outlets := 600
def investigation_1_sample : Nat := 100
def region_C_large_outlets : Nat := 10
def investigation_2_sample : Nat := 7

def stratified_sampling : Π {α : Type}, (α → Prop) → Prop := sorry
def simple_random_sampling : Π {α : Type}, (α → Prop) → Prop := sorry

theorem appropriate_sampling_methods :
  (stratified_sampling (λ outlet, outlet ∈ sales_outlets) ∧
   simple_random_sampling (λ outlet, outlet ∈ region_C_large_outlets)) :=
by
  sorry

end appropriate_sampling_methods_l573_573680


namespace real_solutions_eq59_l573_573020

theorem real_solutions_eq59 :
  (∃ (x: ℝ), -50 ≤ x ∧ x ≤ 50 ∧ (x / 50) = sin x) ∧
  (∃! (S: ℕ), S = 59) :=
sorry

end real_solutions_eq59_l573_573020


namespace propositionD_l573_573866

variables (m n : Type) [line m] [line n]
variables (α β : Type) [plane α] [plane β]

-- Definitions based on given conditions
def diff_lines : Prop := m ≠ n
def diff_planes : Prop := α ≠ β
def perp_line_plane (l : Type) [line l] (p : Type) [plane p] : Prop := sorry -- definition of line perpendicular to plane
def subset_line_plane (l : Type) [line l] (p : Type) [plane p] : Prop := sorry -- definition of line subset of plane
def parallel_planes (p1 p2 : Type) [plane p1] [plane p2] : Prop := sorry -- definition of parallel planes
def perp_lines (l1 l2 : Type) [line l1] [line l2] : Prop := sorry -- definition of perpendicular lines
def parallel_lines (l1 l2 : Type) [line l1] [line l2] : Prop := sorry -- definition of parallel lines

-- The theorem we need to prove
theorem propositionD (diff_lines : diff_lines m n) (diff_planes : diff_planes α β)
  (h1 : parallel_planes α β) (h2 : perp_line_plane m α) (h3 : parallel_lines n β) :
  perp_lines m n :=
sorry

end propositionD_l573_573866


namespace sequences_relation_l573_573526

theorem sequences_relation (a b : ℕ → ℝ) (h : ∀ n, b n = ∑ i in Finset.range (n+1), (Nat.choose n i) * (a i)) :
  ∀ n, a n = ∑ i in Finset.range (n+1), ((-1) ^ (n - i)) * (Nat.choose n i) * (b i) :=
by
  sorry

end sequences_relation_l573_573526


namespace area_of_cross_l573_573245

-- Definitions based on the conditions
def congruent_squares (n : ℕ) := n = 5
def perimeter_of_cross (p : ℕ) := p = 72

-- Targeting the proof that the area of the cross formed by the squares is 180 square units
theorem area_of_cross (n p : ℕ) (h1 : congruent_squares n) (h2 : perimeter_of_cross p) : 
  5 * (p / 12) ^ 2 = 180 := 
by 
  sorry

end area_of_cross_l573_573245


namespace min_g_x1_minus_g_x2_l573_573472

/-- Define the function f(x) = x - 1/x - a * ln(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x - 1/x - a * log x

/-- Define the derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + 1) / x

/-- Define the function g(x) = f(x) + 2a * ln(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := f(a, x) + 2*a * log x

/-- Define the function h(x) = 2(x - 1/x) - 2(x + 1/x)ln(x) -/
def h (x : ℝ) : ℝ := 2 * (x - 1/x) - 2 * (x + 1/x) * log x

/-- Validate that for x1 ∈ (0, e], the minimum value of g(x1) - g(x2) is -4/e -/
theorem min_g_x1_minus_g_x2 (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 ≤ Real.exp 1) (hx2 : x2 = 1 / x1) 
  : g 1 x1 - g 1 x2 = -4 / Real.exp 1 :=
  by
  sorry

end min_g_x1_minus_g_x2_l573_573472


namespace ellipse_and_triangle_area_proof_l573_573841

variables {x y a b c k m : ℝ}

def ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop := (p.1^2 / a^2) + (p.2^2 / b^2) = 1

-- Given conditions
def eccentricity (a b c : ℝ) : Prop := c = (sqrt 3) / 2 * a
def max_area_OAB : Prop := (1 / 2 * a * b = 1)

-- Conditions for the points M and N on line l intersecting ellipse
def intersect_points (C : ℝ × ℝ → Prop) (l : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  C p1 ∧ C p2 ∧ l p1.1 = p1.2 ∧ l p2.1 = p2.2

-- Given condition for slopes
def slopes_condition (p1 p2 : ℝ × ℝ) : Prop :=
  ((p1.2) / (p1.1)) * ((p2.2) / (p2.1)) = 5 / 4

-- The goal to prove
theorem ellipse_and_triangle_area_proof :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∃ c : ℝ, eccentricity a b c ∧ max_area_OAB ∧ ellipse a b) ∧
  (∀ (k m : ℝ), ∃ p1 p2 : ℝ × ℝ, 
    intersect_points (ellipse 2 1) (λ x, k * x + m) p1 p2 →
    slopes_condition p1 p2 → 
    1 / 2 * ((-m / k) * (p1.2 - p2.2)) = 1)) :=
sorry

end ellipse_and_triangle_area_proof_l573_573841


namespace add_fractions_l573_573729

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = 5 / 8 :=
by
  sorry

end add_fractions_l573_573729


namespace final_expression_l573_573237

noncomputable def f (x : ℝ) : ℝ := sorry

lemma functional_form (t x : ℝ) (ht : x = (t + 2) ^ 2 - 4) (hdom : ∀ x, -4 ≤ x → x ≤ f x) : 
  f x = |t + 2| := sorry

theorem final_expression (x : ℝ) (hdom : x ≥ -4) : 
  f x = real.sqrt (x + 4) := sorry

end final_expression_l573_573237


namespace min_pigs_condition_l573_573621

theorem min_pigs_condition (P H : ℕ) (T := P + H) :
  (0.54 * T).ceil ≤ P ∧ P ≤ (0.57 * T).floor → T = 9 → P = 5 :=
by sorry

end min_pigs_condition_l573_573621


namespace min_value_reciprocal_sum_l573_573429

theorem min_value_reciprocal_sum
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Real.sqrt 5 = Real.geom_mean (5 ^ a) (5 ^ b)) :
  4 ≤ (1 / a + 1 / b) :=
sorry

end min_value_reciprocal_sum_l573_573429


namespace number_of_divisors_with_units_digit_1_of_2121_l573_573385

theorem number_of_divisors_with_units_digit_1_of_2121 : 
  (finset.card {d : ℕ | d ∣ 2121 ∧ (d % 10 = 1)}.to_finset) = 4 := 
by 
  -- conditions
  have h2121_factorization: 2121 = 3 * 7 * 101 := by norm_num,
  -- sorry used to skip all the proof steps
  sorry

end number_of_divisors_with_units_digit_1_of_2121_l573_573385


namespace total_cars_counted_l573_573527

noncomputable def jaredCars : ℕ := 300
noncomputable def annCars : ℕ := 345
noncomputable def alfredInitialCars : ℕ := 338
noncomputable def alfredRecountedCars: ℕ := 379
noncomputable def bellaCars: ℕ := 341

theorem total_cars_counted :
  jaredCars + annCars + alfredRecountedCars + bellaCars = 1365 :=
  by
    -- Specify each car count
    have jared_count : jaredCars = 300 := rfl
    have ann_count : annCars = 345 := rfl
    have alfred_recount : alfredRecountedCars = 379 := rfl
    have bella_count : bellaCars = 341 := rfl
    -- Aggregate total sum
    calc
    300 + 345 + 379 + 341
      = 1365 : by sorry

end total_cars_counted_l573_573527


namespace disproves_proposition_l573_573422

theorem disproves_proposition : ∃ (a b : ℝ), (a = -3) ∧ (b = 2) ∧ (a^2 > b^2) ∧ (a ≤ b) := 
by
  exists (-3 : ℝ)
  exists (2 : ℝ)
  repeat { split }
  · rfl
  · rfl
  · norm_num
  · norm_num
  · sorry

end disproves_proposition_l573_573422


namespace total_donations_correct_l573_573702

def num_basketball_hoops : Nat := 60

def num_hoops_with_balls : Nat := num_basketball_hoops / 2

def num_pool_floats : Nat := 120
def num_damaged_floats : Nat := num_pool_floats / 4
def num_remaining_floats : Nat := num_pool_floats - num_damaged_floats

def num_footballs : Nat := 50
def num_tennis_balls : Nat := 40

def num_hoops_without_balls : Nat := num_basketball_hoops - num_hoops_with_balls

def total_donations : Nat := 
  num_hoops_without_balls + num_hoops_with_balls + num_remaining_floats + num_footballs + num_tennis_balls

theorem total_donations_correct : total_donations = 240 := by
  sorry

end total_donations_correct_l573_573702


namespace no_such_x_exists_l573_573398

theorem no_such_x_exists : ¬ ∃ x : ℝ, 
  (∃ x1 : ℤ, x - 1/x = x1) ∧ 
  (∃ x2 : ℤ, 1/x - 1/(x^2 + 1) = x2) ∧ 
  (∃ x3 : ℤ, 1/(x^2 + 1) - 2*x = x3) :=
by
  sorry

end no_such_x_exists_l573_573398


namespace carol_age_l573_573264

theorem carol_age (B C : ℕ) (h1 : B + C = 66) (h2 : C = 3 * B + 2) : C = 50 :=
sorry

end carol_age_l573_573264


namespace find_k_l573_573363

theorem find_k (angle_BAC : ℝ) (angle_D : ℝ)
  (h1 : 0 < angle_BAC ∧ angle_BAC < π)
  (h2 : 0 < angle_D ∧ angle_D < π)
  (h3 : (π - angle_BAC) / 2 = 3 * angle_D) :
  angle_BAC = (5 / 11) * π :=
by sorry

end find_k_l573_573363


namespace probability_triangle_area_l573_573336

theorem probability_triangle_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let P := set.Icc (0 : ℝ) 1 ×ˢ set.Icc (0 : ℝ) 1 in
  let A : set (ℝ × ℝ) := {p | ∃ u v : ℝ, (u, v) ∈ P ∧ u > v} in
  let area := ∫⁻ p in A, 1 in
  (area / ∫⁻ p in P, 1) = (1 / 2) :=
sorry

end probability_triangle_area_l573_573336


namespace third_pipe_empty_time_l573_573292

theorem third_pipe_empty_time :
  (let A := 1 / 60 in 
  let B := 1 / 80 in 
  let rate_when_all_open := 1 / 40 in 
  let rate_third_pipe_only := rate_when_all_open - (A + B) in
  let time_to_empty := 1 / rate_third_pipe_only in
  time_to_empty = 240) := 
by 
  sorry

end third_pipe_empty_time_l573_573292


namespace find_m_l573_573519

-- Definitions of points and triangle sides
variables (A B C : ℝ × ℝ)
variables (M : ℝ × ℝ)
variable (m : ℝ)
variable (AB AC BC : ℝ)

-- Conditions
axiom h₁ : B = (0, 0)
axiom h₂ : A = (0, 5)
axiom h₃ : C = (12, 0)
axiom h₄ : AB = 5
axiom h₅ : BC = 12
axiom h₆ : AC = 13
axiom h₇ : M = ((12 / 2), (5 / 2))
axiom h₈ : BM = m * Real.sqrt 2
axiom BM : ℝ

-- Definition of distance BM
def BM_distance (B M : ℝ × ℝ) : ℝ := Real.sqrt ((fst M - fst B) ^ 2 + (snd M - snd B) ^ 2)

-- Median from vertex B to side AC
def median_from_B_to_AC : ℝ := BM_distance B M

-- Main theorem to prove
theorem find_m : m = 13 / 2 := by
  sorry

end find_m_l573_573519


namespace sin_A_eq_sides_b_c_l573_573456

-- Conditions
variables {A B C : Angle}
variables {a b c : ℝ}
variable (s_ABC : ℝ)

axiom sides_eq : a = 2
axiom cosB_eq : cos B = 3 / 5

-- Part I: Proof that sin A = 2 / 5
theorem sin_A_eq : b = 4 → sin A = 2 / 5 :=
by
  intros h1
  sorry

-- Part II: Proof that b = sqrt 17 and c = 5
theorem sides_b_c (s_abc_eq : s_ABC = 4) : b = Real.sqrt 17 ∧ c = 5 :=
by
  sorry

end sin_A_eq_sides_b_c_l573_573456


namespace bowl_capacity_l573_573560

theorem bowl_capacity (C : ℝ) (h1 : (2/3) * C * 5 + (1/3) * C * 4 = 700) : C = 150 := 
by
  sorry

end bowl_capacity_l573_573560


namespace tetrahedron_triangle_area_l573_573533

noncomputable def area_of_triangle_QRS (a b c u v w : ℝ) : ℝ :=
  sqrt (u^2 + v^2 + w^2)

theorem tetrahedron_triangle_area (a b c u v w : ℝ) 
  (h1 : u = 1 / 2 * a * b)
  (h2 : v = 1 / 2 * b * c)
  (h3 : w = 1 / 2 * c * a) 
  : area_of_triangle_QRS a b c u v w = sqrt (u^2 + v^2 + w^2) :=
  by
    sorry

end tetrahedron_triangle_area_l573_573533


namespace center_of_mass_of_triangular_system_eq_l573_573634

variables {A B C : (ℝ × ℝ)} (mA mB mC : ℝ)

-- Define the masses and coordinates of the vertices
def mass_1 := 1
def mass_2 := 2
def mass_3 := 3

def vertex_A := (0, 0)
def vertex_B := (1, 0)
def vertex_C := (1/2, real.sqrt 3 / 2)

-- Define the total mass
def total_mass : ℝ := mass_1 + mass_2 + mass_3

-- Prove the center of mass
theorem center_of_mass_of_triangular_system_eq :
  ((mass_1 * vertex_A.1 + mass_2 * vertex_B.1 + mass_3 * vertex_C.1) / total_mass,
   (mass_1 * vertex_A.2 + mass_2 * vertex_B.2 + mass_3 * vertex_C.2) / total_mass)
  = (7 / 12, real.sqrt 3 / 4) :=
by
  -- The proof would be placed here
  sorry

end center_of_mass_of_triangular_system_eq_l573_573634


namespace ce_length_l573_573588

-- Definitions for the problem
structure Point (α : Type) := (x y z : α)

def A : Point ℝ := ⟨0, 0, 0⟩
def B : Point ℝ := ⟨10, 0, 0⟩
def C : Point ℝ := ⟨0, 10, 0⟩

def A1 : Point ℝ := ⟨0, 0, 12⟩
def B1 : Point ℝ := ⟨10, 0, 12⟩
def C1 : Point ℝ := ⟨0, 10, 12⟩

def M : Point ℝ := ⟨0, 0, 6⟩ -- Midpoint of AA1

-- The statement of the problem
theorem ce_length (E : Point ℝ) (hE : ∃ (λ t : ℝ, (1 - t) • A1 + t • C1 = E) 
  ∧ (∃ (h : ℝ), (M.x * h + B1.x) * (cos (45 * (π / 180))) = 1)) : 
  dist C E = 4 := 
sorry

end ce_length_l573_573588


namespace probability_of_stable_number_l573_573908

-- Definition of a "stable number"
def is_stable_number (n : ℕ) : Prop :=
  let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
  (digits.nth 0 = some 1 ∨ digits.nth 0 = some 2 ∨ digits.nth 0 = some 3) ∧
  (digits.nth 1 = some 1 ∨ digits.nth 1 = some 2 ∨ digits.nth 1 = some 3) ∧
  (digits.nth 2 = some 1 ∨ digits.nth 2 = some 2 ∨ digits.nth 2 = some 3) ∧
  (|digits.nth 0.get_or_else 0 - digits.nth 1.get_or_else 0| ≤ 1) ∧
  (|digits.nth 1.get_or_else 0 - digits.nth 2.get_or_else 0| ≤ 1)

-- List of all possible three-digit numbers made from 1, 2, 3 without repetition
def all_possible_numbers : list ℕ := [123, 132, 213, 231, 312, 321]

-- Number of stable numbers
def count_stable_numbers : ℕ :=
  list.length (all_possible_numbers.filter is_stable_number)

-- Total number of possible three-digit numbers
def total_possible_numbers : ℕ := list.length all_possible_numbers

-- Probability of a number being stable
def stable_number_probability : ℚ :=
  count_stable_numbers / total_possible_numbers

-- Statement of the problem
theorem probability_of_stable_number : stable_number_probability = 1/3 :=
by {
  -- Indicate this part is not be filled, hence "sorry".
  sorry
}

end probability_of_stable_number_l573_573908


namespace geometric_sequence_sum_range_l573_573446

noncomputable def a_n (n : ℕ) : ℝ :=
  if (n = 0) then (4 : ℝ) else (4 / (2 ^ n))

def sequence_sum (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (a_n i) * (a_n (i + 1))

theorem geometric_sequence_sum_range (n : ℕ) :
  8 ≤ sequence_sum n ∧ sequence_sum n < 32 / 3 := by
  sorry

end geometric_sequence_sum_range_l573_573446


namespace target_runs_l573_573510

theorem target_runs (r1 r2 : ℝ) (o1 o2 : ℕ) (target : ℝ) :
  r1 = 3.6 ∧ o1 = 10 ∧ r2 = 6.15 ∧ o2 = 40 → target = (r1 * o1) + (r2 * o2) := by
  sorry

end target_runs_l573_573510


namespace simplify_and_evaluate_l573_573229

theorem simplify_and_evaluate 
  (a b : ℚ) (h_a : a = -1/3) (h_b : b = -3) : 
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := 
  by 
    rw [h_a, h_b]
    sorry

end simplify_and_evaluate_l573_573229


namespace exists_function_f_l573_573760

-- Define the golden ratio.
def phi : ℝ := (1 + Real.sqrt 5) / 2

-- Define the function f using the floor function.
def f (n : ℕ) : ℕ := Int.toNat ⌊phi * n + 1/2⌋

-- Define the conditions as hypotheses.
theorem exists_function_f :
  (∀ n, 0 < n → f (f n) = f n + n) ∧
  (∀ n, 0 < n → f n < f (n + 1)) ∧
  f 1 = 2 :=
by
  -- The proof is omitted.
  sorry

end exists_function_f_l573_573760


namespace remainder_of_2001st_term_l573_573607

-- Definitions
def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    let k := (n - 1) / 2 in 9 * k^2 + 12 * k + 3
  else 
    let k := n / 2 in 9 * k^2 - 1

def a2001 := a_n 2001

-- The target theorem statement
theorem remainder_of_2001st_term : a2001 % 1000 = 3 :=
  by {
    sorry
  }

end remainder_of_2001st_term_l573_573607


namespace find_divisor_l573_573305

theorem find_divisor (D N : ℕ) (k l : ℤ)
  (h1 : N % D = 255)
  (h2 : (2 * N) % D = 112) :
  D = 398 := by
  -- Proof here
  sorry

end find_divisor_l573_573305


namespace know_number_eventually_l573_573279

theorem know_number_eventually (a b : ℕ) (h : |a - b| = 1) : 
  ∃ N : ℕ, (∀ k < N, ¬ (mathematician_a_knows_number ∨ mathematician_b_knows_number)) → (mathematician_a_knows_number ∨ mathematician_b_knows_number) := 
sorry

-- The following are necessary auxiliary definitions assuming some interpretational freedom for the problem's context.
def mathematician_a_knows_number : Prop := sorry
def mathematician_b_knows_number : Prop := sorry

end know_number_eventually_l573_573279


namespace arithmetic_sequences_l573_573149

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

def b_n (n : ℕ) : ℕ :=
  if n = 1 then 4 else 2 * n + 1

def S_n (n : ℕ) : ℕ := n^2 + a_n n

noncomputable def T_n (n : ℕ) : ℚ :=
  if n = 1 then 1 / 20 
  else (6 * n - 1) / (20 * (2 * n + 3))

theorem arithmetic_sequences :
  (∀ n, a_n n = 2 * n + 1) ∧
  (∀ n, b_n n = (if n = 1 then 4 else 2 * n + 1)) ∧
  (∀ n, T_n n = if n = 1 then 1 / 20 else (6 * n - 1) / (20 * (2 * n + 3))) := 
  sorry

end arithmetic_sequences_l573_573149


namespace find_divisor_l573_573656

theorem find_divisor (D : ℕ) : 
  let dividend := 109
  let quotient := 9
  let remainder := 1
  (dividend = D * quotient + remainder) → D = 12 :=
by
  sorry

end find_divisor_l573_573656


namespace non_isolated_5_element_subsets_l573_573735

theorem non_isolated_5_element_subsets (n : ℕ) (S : finset ℕ) (h : S = finset.range (n + 1)) :
  (finset.card {A : finset ℕ | A ⊆ S ∧ finset.card A = 5 ∧ (∀ a ∈ A, (a - 1) ∈ A ∨ (a + 1) ∈ A)}) = (n - 4) ^ 2 :=
by sorry

end non_isolated_5_element_subsets_l573_573735


namespace sum_inverse_S_ge_one_third_l573_573096

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 4

def sequence (d : ℝ) (n : ℕ) : ℝ := 2 * n + 1

def S (n : ℕ) : ℝ := n * (n + 2)

theorem sum_inverse_S_ge_one_third (n : ℕ) :
  (∑ i in Finset.range n, (1 / S (i + 1))) ≥ 1 / 3 := by
  sorry

end sum_inverse_S_ge_one_third_l573_573096


namespace equal_values_on_plane_l573_573369

theorem equal_values_on_plane (f : ℤ × ℤ → ℕ)
    (h_avg : ∀ (i j : ℤ), f (i, j) = (f (i+1, j) + f (i-1, j) + f (i, j+1) + f (i, j-1)) / 4) :
  ∃ c : ℕ, ∀ (i j : ℤ), f (i, j) = c :=
by
  sorry

end equal_values_on_plane_l573_573369


namespace frequency_of_rolling_six_is_0_point_19_l573_573693

theorem frequency_of_rolling_six_is_0_point_19 :
  ∀ (total_rolls number_six_appeared : ℕ), total_rolls = 100 → number_six_appeared = 19 → 
  (number_six_appeared : ℝ) / (total_rolls : ℝ) = 0.19 := 
by 
  intros total_rolls number_six_appeared h_total_rolls h_number_six_appeared
  sorry

end frequency_of_rolling_six_is_0_point_19_l573_573693


namespace movie_ticket_notation_l573_573903

-- Definition of movie ticket notation
def ticket_notation (row : ℕ) (seat : ℕ) : (ℕ × ℕ) :=
  (row, seat)

-- Given condition: "row 10, seat 3" is denoted as (10, 3)
def given := ticket_notation 10 3 = (10, 3)

-- Proof statement: "row 6, seat 16" is denoted as (6, 16)
theorem movie_ticket_notation : ticket_notation 6 16 = (6, 16) :=
by
  -- Proof omitted, since the theorem statement is the focus
  sorry

end movie_ticket_notation_l573_573903


namespace incorrect_scientific_statement_is_D_l573_573002

-- Define the number of colonies screened by Student A and other students
def studentA_colonies := 150
def other_students_colonies := 50

-- Define the descriptions
def descriptionA := "The reason Student A had such results could be due to different soil samples or problems in the experimental operation."
def descriptionB := "Student A's prepared culture medium could be cultured without adding soil as a blank control, to demonstrate whether the culture medium is contaminated."
def descriptionC := "If other students use the same soil as Student A for the experiment and get consistent results with Student A, it can be proven that Student A's operation was without error."
def descriptionD := "Both experimental approaches described in options B and C follow the principle of control in the experiment."

-- The incorrect scientific statement identified
def incorrect_statement := descriptionD

-- The main theorem statement
theorem incorrect_scientific_statement_is_D : incorrect_statement = descriptionD := by
  sorry

end incorrect_scientific_statement_is_D_l573_573002


namespace partA_partB_l573_573593

-- Part (a)
theorem partA
  (ABCD_convex : convex_quadrilateral ABCD)
  (diagonals_intersect : diagonals_intersect O ABCD)
  (BOC_eq_equi : equilateral BO C O)
  (AOD_eq_equi : equilateral AOD O)
  (T_symm_to_O : symmetric_O_midpoint_CD T O ABCD) :
  equilateral_triangle AB T:= sorry

-- Part (b)
theorem partB
  (ABCD_convex : convex_quadrilateral ABCD)
  (diagonals_intersect : diagonals_intersect O ABCD)
  (BOC_eq_equi : equilateral BO C O)
  (AOD_eq_equi : equilateral AOD O)
  (T_symm_to_O : symmetric_O_midpoint_CD T O ABCD)
  (BC_len : length BC = 3)
  (AD_len : length AD = 5) :
  area_ratio (triangle AB T) (quadrilateral ABCD) = 49 / 64 := sorry

end partA_partB_l573_573593


namespace part1_range_m_minus1_part2_max_value_l573_573884

-- Define the function
def y (x m : ℝ) : ℝ := 2 * (Real.sin x)^2 + m * (Real.cos x) - 1/8

-- Part 1: Proving the range for specified m and x
theorem part1_range_m_minus1 :
  ∀ x : ℝ, (-Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3) →
  -9/8 ≤ y x (-1) ∧ y x (-1) ≤ 2 :=
sorry

-- Part 2: Proving the max value for any x in R for different ranges of m
theorem part2_max_value :
  ∀ m : ℝ,
    (m < -4 → ∀ x : ℝ, y x m ≤ -m - 1/8) ∧
    (-4 ≤ m ∧ m ≤ 4 → ∀ x : ℝ, y x m ≤ (m^2 + 15)/8) ∧
    (m > 4 → ∀ x : ℝ, y x m ≤ m - 1/8) :=
sorry

end part1_range_m_minus1_part2_max_value_l573_573884


namespace segment_equality_l573_573070

variables {Point : Type} [AddGroup Point]

-- Define the points A, B, C, D, E, F
variables (A B C D E F : Point)

-- Given conditions
variables (AC CE BD DF AD CF : Point)
variable (h1 : AC = CE)
variable (h2 : BD = DF)
variable (h3 : AD = CF)

-- Theorem statement
theorem segment_equality (h1 : A - C = C - E)
                         (h2 : B - D = D - F)
                         (h3 : A - D = C - F) :
  (C - D) = (A - B) ∧ (C - D) = (E - F) :=
by
  sorry

end segment_equality_l573_573070


namespace total_weight_of_three_new_people_l573_573240
-- Import necessary libraries

-- Define the problem statement
theorem total_weight_of_three_new_people :
  ∀ (W X : ℝ),
    (∀ w1 w2 w3 : ℝ, w1 = 35 → w2 = 45 → w3 = 56 → W = 14 * ((W - w1 - w2 - w3 + X) / 14 + 2.7)) →
    W = (14 * (W / 14)) →
    X = 173.8 :=
by
  -- assume initial weight W and weight of new people X
  intros W X h_condition h_init_weight
  have h_weight_loss : (W - 35 - 45 - 56 + X) / 14 = (W / 14) + 2.7, from sorry
  have h_X_eqn : X = 173.8, from sorry
  exact h_X_eqn

end total_weight_of_three_new_people_l573_573240


namespace intersection_correct_l573_573853

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_correct : A ∩ B = {1, 2} := 
sorry

end intersection_correct_l573_573853


namespace sum_of_fractions_eq_approx_l573_573637

theorem sum_of_fractions_eq_approx : 
  (∑ n in Finset.range 2023, (2 : ℝ) / (n + 1) / (n + 1 + 2)) ≈ 1.499 (by norm_num) := 
sorry

end sum_of_fractions_eq_approx_l573_573637


namespace train_speed_l573_573345

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (conversion_factor : ℝ) :
  length_of_train = 200 → 
  time_to_cross = 24 → 
  conversion_factor = 3600 → 
  (length_of_train / 1000) / (time_to_cross / conversion_factor) = 30 := 
by
  sorry

end train_speed_l573_573345


namespace min_value_expr_l573_573408

noncomputable def expr (θ : Real) : Real :=
  3 * (Real.cos θ) + 2 / (Real.sin θ) + 2 * Real.sqrt 2 * (Real.tan θ)

theorem min_value_expr :
  ∃ (θ : Real), 0 < θ ∧ θ < Real.pi / 2 ∧ expr θ = (7 * Real.sqrt 2) / 2 := 
by
  sorry

end min_value_expr_l573_573408


namespace parents_gave_money_l573_573380

def money_before_birthday : ℕ := 159
def money_from_grandmother : ℕ := 25
def money_from_aunt_uncle : ℕ := 20
def total_money_after_birthday : ℕ := 279

theorem parents_gave_money :
  total_money_after_birthday = money_before_birthday + money_from_grandmother + money_from_aunt_uncle + 75 :=
by
  sorry

end parents_gave_money_l573_573380


namespace Tadd_250th_l573_573132

-- Define Tadd's sequence
def Tadd_sequence (n : ℕ) : ℕ :=
  let blocks := λ k, 6 * k - 5 in
  let sum_blocks := λ k, 3 * k * (k - 1) in
  let find_k := λ m, Nat.find (λ k, sum_blocks (k - 1) < m ∧ m <= sum_blocks k) in
  let k := find_k n in
  let block_start := sum_blocks (k - 1) + 1 in
  block_start + (n - block_start)

-- Statement to prove
theorem Tadd_250th : 
  Tadd_sequence 250 = 250 :=
sorry

end Tadd_250th_l573_573132


namespace mathlib_problem_l573_573170

/-- Given positive integers a, b, c, define d = gcd(a, b, c) and a = dx, b = dy, c = dz.
Prove that there exists a positive integer N such that
  a ∣ Nbc + b + c,
  b ∣ Nca + c + a,
  c ∣ Nab + a + b
if and only if 
  x, y, z are pairwise coprime, 
  gcd(d, xyz) ∣ x + y + z. 
-/
theorem mathlib_problem (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let d := Nat.gcd (Nat.gcd a b) c in
  let x := a / d in
  let y := b / d in
  let z := c / d in
  (∃ (N : ℕ), a ∣ N * b * c + b + c ∧ b ∣ N * c * a + c + a ∧ c ∣ N * a * b + a + b) ↔ 
  Nat.coprime x y ∧ Nat.coprime y z ∧ Nat.coprime z x ∧ Nat.gcd d (x * y * z) ∣ (x + y + z) :=
by
  sorry

end mathlib_problem_l573_573170


namespace parabola_equation_l573_573845

theorem parabola_equation (M : ℝ × ℝ) (hM : M = (5, 3))
    (h_dist : ∀ a : ℝ, |5 + 1/(4*a)| = 6) :
    (y = (1/12)*x^2) ∨ (y = -(1/36)*x^2) :=
sorry

end parabola_equation_l573_573845


namespace rotated_intersection_point_l573_573445

theorem rotated_intersection_point (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
  ∃ P : ℝ × ℝ, P = (-Real.sin θ, Real.cos θ) ∧ 
    ∃ φ : ℝ, φ = θ + π / 2 ∧ 
      P = (Real.cos φ, Real.sin φ) := 
by
  sorry

end rotated_intersection_point_l573_573445


namespace add_base6_l573_573351

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  6 * d1 + d0

theorem add_base6 (a b : Nat) (ha : base6_to_base10 a = 23) (hb : base6_to_base10 b = 10) : 
  base6_to_base10 (53 : Nat) = 33 :=
by
  sorry

end add_base6_l573_573351


namespace number_of_solutions_l573_573773

def condition1 (x y z : ℤ) : Prop := |x + y| + z = 23
def condition2 (x y z : ℤ) : Prop := x * y + |z| = 119
theorem number_of_solutions : {p : ℤ × ℤ × ℤ | condition1 p.1 p.2.1 p.2.2 ∧ condition2 p.1 p.2.1 p.2.2}.to_finset.card = 4 := 
by 
  sorry

end number_of_solutions_l573_573773


namespace non_overlapping_squares_in_20th_figure_l573_573253

theorem non_overlapping_squares_in_20th_figure :
  let T := λ n, 2 * n^2 - 2 * n + 1 in
  T 20 = 761 :=
by
  let T := λ n, 2 * n^2 - 2 * n + 1
  have h : T 20 = 761 := by sorry
  exact h

end non_overlapping_squares_in_20th_figure_l573_573253


namespace solve_system_of_inequalities_l573_573755

variable {R : Type*} [LinearOrderedField R]

theorem solve_system_of_inequalities (x1 x2 x3 x4 x5 : R)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h5 : x5 > 0) :
  (x1^2 - x3^2) * (x2^2 - x3^2) ≤ 0 ∧ 
  (x3^2 - x1^2) * (x3^2 - x1^2) ≤ 0 ∧ 
  (x3^2 - x3 * x2) * (x1^2 - x3 * x2) ≤ 0 ∧ 
  (x1^2 - x1 * x3) * (x3^2 - x1 * x3) ≤ 0 ∧ 
  (x3^2 - x2 * x1) * (x1^2 - x2 * x1) ≤ 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 :=
sorry

end solve_system_of_inequalities_l573_573755


namespace tens_digit_11_pow_2045_l573_573396

theorem tens_digit_11_pow_2045 : 
    ((11 ^ 2045) % 100) / 10 % 10 = 5 :=
by
    sorry

end tens_digit_11_pow_2045_l573_573396


namespace probability_no_three_consecutive_A_l573_573682

def total_strings : ℕ :=
  3^6

def count_strings_with_three_consecutive_A : ℕ :=
  32 + 12 + 4 + 1

def count_strings_without_three_consecutive_A : ℕ :=
  total_strings - count_strings_with_three_consecutive_A

theorem probability_no_three_consecutive_A : (count_strings_without_three_consecutive_A : ℚ) / total_strings = 680/729 := by sorry

end probability_no_three_consecutive_A_l573_573682


namespace series_sum_evaluation_l573_573007

def series_sum : ℝ := ∑' k : ℕ, (k^3:ℝ)/2^(k:ℝ)

theorem series_sum_evaluation : series_sum = 26 := by
  sorry

end series_sum_evaluation_l573_573007


namespace concurrency_of_l573_573962

open Set

-- Definitions based on the problem's conditions
variable {A B C P P1 P2 Q1 Q2 : Point}
variable {AC BC AP BP : Line}

-- Points P1 and P2 as the feet of perpendiculars from P to AC and BC
def perp_foot_P1_to_AC (P AC : Line) (P1 : Point) : Prop := P.perpendicular AC ∧ P1 ∈ AC ∧ P1 ∈ P
def perp_foot_P2_to_BC (P BC : Line) (P2 : Point) : Prop := P.perpendicular BC ∧ P2 ∈ BC ∧ P2 ∈ P

-- Lines AP and BP
def line_AP (A P : Point) : Line := line_through_points A P
def line_BP (B P : Point) : Line := line_through_points B P

-- Points Q1 and Q2 as the feet of perpendiculars from C to AP and BP
def perp_foot_Q1_to_AP (C AP : Line) (Q1 : Point) : Prop := C.perpendicular AP ∧ Q1 ∈ AP ∧ Q1 ∈ C
def perp_foot_Q2_to_BP (C BP : Line) (Q2 : Point) : Prop := C.perpendicular BP ∧ Q2 ∈ BP ∧ Q2 ∈ C

-- Prove that these lines are concurrent
theorem concurrency_of Q1P2_Q2P1_AB :
  perp_foot_P1_to_AC P AC P1 →
  perp_foot_P2_to_BC P BC P2 →
  perp_foot_Q1_to_AP C (line_AP A P) Q1 →
  perp_foot_Q2_to_BP C (line_BP B P) Q2 →
  are_concurrent (line_through_points Q1 P2) (line_through_points Q2 P1) (line_through_points A B) :=
by
  sorry

end concurrency_of_l573_573962


namespace ratio_simplified_l573_573725

theorem ratio_simplified (kids_meals : ℕ) (adult_meals : ℕ) (h1 : kids_meals = 70) (h2 : adult_meals = 49) : 
  ∃ (k a : ℕ), k = 10 ∧ a = 7 ∧ kids_meals / Nat.gcd kids_meals adult_meals = k ∧ adult_meals / Nat.gcd kids_meals adult_meals = a :=
by
  sorry

end ratio_simplified_l573_573725


namespace limit_problem_l573_573731

theorem limit_problem (h : ∀ x, x ≠ -3):
  (∀ x, (x^2 + 2*x - 3)^2 = (x + 3)^2 * (x - 1)^2) →
  (∀ x, x^3 + 4*x^2 + 3*x = x * (x + 1) * (x + 3)) →
  tendsto (λ x, ((x^2 + 2*x - 3)^2) / (x^3 + 4*x^2 + 3*x)) (𝓝[-] (-3)) (𝓝 0) :=
by
  intros numerator_factor denominator_factor
  sorry

end limit_problem_l573_573731


namespace problem_statement_l573_573542

def f0 (x : ℝ) := Real.cos x
def f' (f : ℝ → ℝ) := λ x, Real.deriv f x

noncomputable def fn : ℕ → (ℝ → ℝ)
| 0     := f0
| (n+1) := f' (fn n)

theorem problem_statement (x : ℝ) : fn 2011 x = Real.sin x :=
sorry

end problem_statement_l573_573542


namespace prob_one_product_successful_expected_company_profit_l573_573675

theorem prob_one_product_successful 
  (P_A : ℝ) (P_B : ℝ) (independent : Prop)
  (P_A_success : P_A = 4/5) (P_B_success : P_B = 3/4) (independent_events : independent):
  (1 - P_A) * P_B + P_A * (1 - P_B) = 7/20 :=
by
  -- Placeholder for the actual use of conditions and computation of probabilities
  sorry

theorem expected_company_profit 
  (P_A : ℝ) (P_B : ℝ) 
  (P_A_success : P_A = 4/5) (P_B_success : P_B = 3/4):
  let ξ := λ (A_success B_success : Prop), 
    if A_success then 
      if B_success then 270 else 110 
    else 
      if B_success then 60 else -100;
  let P := λ (p : ℝ), 
    if p = 7/20 then (ξ false true) * (1 - P_A) * P_B + (ξ true false) * P_A * (1 - P_B) in
  E(ξ) = -100 * (1 - P_A) * (1 - P_B) 
    + 60 * (1 - P_A) * P_B 
    + 110 * P_A * (1 - P_B)
    + 270 * P_A * P_B = 188 :=
by
  -- Placeholder for the actual computation of the expected value
  sorry

end prob_one_product_successful_expected_company_profit_l573_573675


namespace balloon_air_volume_l573_573164

theorem balloon_air_volume (n t k : ℕ) (h1 : n = 1000) (h2 : t = 500) (h3 : k = 20) :
  10000 / n = 10 :=
by
  have h4 : 10000 = k * t,
  { rw [h3, h2], 
    sorry },
  rw h4,
  rw h1,
  norm_num,
  norm_num,
  sorry

end balloon_air_volume_l573_573164


namespace sqrt_of_4_l573_573611

theorem sqrt_of_4 : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_of_4_l573_573611


namespace min_value_of_z_l573_573183

-- Define the conditions as separate hypotheses.
variable (x y : ℝ)

def condition1 : Prop := x - y + 1 ≥ 0
def condition2 : Prop := x + y - 1 ≥ 0
def condition3 : Prop := x ≤ 3

-- Define the objective function.
def z : ℝ := 2 * x - 3 * y

-- State the theorem to prove the minimum value of z given the conditions.
theorem min_value_of_z (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x) :
  ∃ x y, condition1 x y ∧ condition2 x y ∧ condition3 x ∧ z x y = -6 :=
sorry

end min_value_of_z_l573_573183


namespace largest_multiple_of_7_l573_573888

def repeated_188 (k : Nat) : ℕ := (List.replicate k 188).foldr (λ x acc => x * 1000 + acc) 0

theorem largest_multiple_of_7 :
  ∃ n, n = repeated_188 100 ∧ ∃ m, m ≤ 303 ∧ m ≥ 0 ∧ m ≠ 300 ∧ (repeated_188 m % 7 = 0 → n ≥ repeated_188 m) :=
by
  sorry

end largest_multiple_of_7_l573_573888


namespace geometric_sequence_problem_l573_573437

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (a 1 * (1 - (r : ℝ) ^ n)) / (1 - r)

theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) (r : ℝ) 
  (ha1 : a 1 = 2) 
  (ha2a8 : a 2 * a 8 = 1024)
  (ham : a m = 32)
  (hg : geometric_sequence a)
  (hs : sum_of_first_n_terms a S)
  : S m = 62 := 
sorry

end geometric_sequence_problem_l573_573437


namespace evaluate_expression_l573_573764

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^a + (b^a)^b = 793) := by
  -- The following lines skip the proof but outline the structure:
  sorry

end evaluate_expression_l573_573764


namespace mode_of_list_is_five_l573_573155

def list := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def occurrence_count (l : List ℕ) (x : ℕ) : ℕ :=
  l.count x

def is_mode (l : List ℕ) (x : ℕ) : Prop :=
  ∀ y : ℕ, occurrence_count l x ≥ occurrence_count l y

theorem mode_of_list_is_five : is_mode list 5 := by
  sorry

end mode_of_list_is_five_l573_573155


namespace g1_plus_gneg1_eq_l573_573438

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom cond1 : ∀ x y : ℝ, f x y = f x g y - g x f y
axiom cond2 : f 1 = f 2 ∧ f 1 ≠ 0

theorem g1_plus_gneg1_eq :
  g 1 + g (-1) = 1 :=
sorry

end g1_plus_gneg1_eq_l573_573438


namespace number_of_ways_to_purchase_magazines_l573_573669

/-
Conditions:
1. The bookstore sells 11 different magazines.
2. 8 of these magazines are priced at 2 yuan each.
3. 3 of these magazines are priced at 1 yuan each.
4. Xiao Zhang has 10 yuan to buy magazines.
5. Xiao Zhang can buy at most one copy of each magazine.
6. Xiao Zhang wants to spend all 10 yuan.

Question:
The number of different ways Xiao Zhang can purchase magazines with 10 yuan.

Answer:
266
-/

theorem number_of_ways_to_purchase_magazines : ∀ (magazines_1_yuan magazines_2_yuan : ℕ),
  magazines_1_yuan = 3 →
  magazines_2_yuan = 8 →
  (∃ (ways : ℕ), ways = 266) :=
by
  intros
  sorry

end number_of_ways_to_purchase_magazines_l573_573669


namespace tangent_line_at_a1_lower_bound_for_f_l573_573097

-- Define the function f(x) as given
def f (a x : ℝ) : ℝ := (1/2) * a * x ^ 2 + (2 * a - 1) * x - 2 * real.log x

theorem tangent_line_at_a1 :
  let f (x : ℝ) := (1/2) * x ^ 2 + x - 2 * real.log x in
  tangent_line f 2 = (λ x, 2 * x - 2 * real.log 2) :=
by
  sorry

theorem lower_bound_for_f (a : ℝ) (ha : 0 < a) :
  ∀ x > 0, (1/2) * a * x ^ 2 + (2 * a - 1) * x - 2 * real.log x ≥ 4 - (5/(2 * a)) :=
by
  sorry

end tangent_line_at_a1_lower_bound_for_f_l573_573097


namespace sixth_root_of_binomial_expansion_l573_573753

theorem sixth_root_of_binomial_expansion :
  (\left( \sum_{k=0}^{6} \binom{6}{k} * 100^(6-k) * 1^k \right) ^ (1 / 6)) = 101 :=
by
  sorry

end sixth_root_of_binomial_expansion_l573_573753


namespace eval_expression_pow_i_l573_573400

theorem eval_expression_pow_i :
  i^(12345 : ℤ) + i^(12346 : ℤ) + i^(12347 : ℤ) + i^(12348 : ℤ) = (0 : ℂ) :=
by
  -- Since this statement doesn't need the full proof, we use sorry to leave it open 
  sorry

end eval_expression_pow_i_l573_573400


namespace value_of_x_plus_y_l573_573102

theorem value_of_x_plus_y (x y : ℝ) (hx : 2^x + 4 * x + 12 = 0) (hy : log 2 ((y - 1)^3) + 3 * y + 12 = 0) : x + y = -2 := 
sorry

end value_of_x_plus_y_l573_573102


namespace possible_values_of_f_l573_573539

def x (k : ℕ) : ℝ := (-1) ^ k

def f (n : ℕ) (h : n > 0) : ℝ := (∑ k in finset.range n, x (k + 1)) / n

theorem possible_values_of_f (n : ℕ) (h : n > 0) :
  {f n h | n > 0} = {0, -1 / n} :=
sorry

end possible_values_of_f_l573_573539


namespace find_arith_seq_an_l573_573084

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ k in finset.range n, a (k + 1)

theorem find_arith_seq_an (a : ℕ → ℝ) (d : ℝ) (h_arith : is_arithmetic_sequence a)
(h_nonzero : ∀ n : ℕ, a n ≠ 0)
(h_sum : ∀ n : ℕ, (a n)^2 = sum_of_first_n_terms a (2 * n - 1)) :
∀ n : ℕ, a n = 2 * n - 1 :=
by
  sorry

end find_arith_seq_an_l573_573084


namespace f_of_72_l573_573447

theorem f_of_72 (f : ℕ → ℝ) (p q : ℝ) (h1 : ∀ a b : ℕ, f (a * b) = f a + f b)
  (h2 : f 2 = p) (h3 : f 3 = q) : f 72 = 3 * p + 2 * q := 
sorry

end f_of_72_l573_573447


namespace turtle_foot_sphere_area_l573_573505

theorem turtle_foot_sphere_area
  (PA AB AC : ℝ)
  (PA_perpendicular_ABC : PA ⊥ plane ABC)
  (P_lies_on_sphere : ∀ point ∈ {P, A, B, C}, lies_on_sphere point O)
  (PA_eq : PA = 3)
  (AB_eq : AB = 4)
  (AC_eq : AC = 5) :
  surface_area_of_sphere O = 34 * π :=
by sorry

-- Definitions of all the geometric and measurement terms used:
def plane (pts : set ℝ) : Type := sorry
def lies_on_sphere (point : ℝ) (sphere : ℝ) : Prop := sorry
def surface_area_of_sphere (sphere : ℝ) : ℝ := sorry 

end turtle_foot_sphere_area_l573_573505


namespace find_a_l573_573823

theorem find_a (a : ℝ) :
    let A := {x : ℝ | x^2 + 3 * x + 2 = 0},
        B := {x : ℝ | a * x - 2 = 0}
    in A ∪ B = A → a = 0 ∨ a = -1 ∨ a = -2 := 
by
  sorry

end find_a_l573_573823


namespace max_score_is_43_l573_573738

def scores : List ℤ := [10, 7, 6, 8, 5, 9, 8, 8, 5, 6]
def times : List ℚ := [2/3, 1/2, 1/3, 2/3, 1/4, 2/3, 1/2, 2/5, 1/5, 1/4]
def costs : List ℤ := [1000, 700, 300, 800, 200, 900, 900, 600, 400, 600]

def is_correct_vector (v : List ℤ) : Prop :=
  (List.dotProd v times ≤ 3) ∧ (List.dotProd v costs ≤ 3500)

def score_of_vector (v : List ℤ) : ℤ :=
  List.dotProd v scores

def maximum_score (vectors : List (List ℤ)) : ℤ :=
  vectors.filter is_correct_vector |>.map score_of_vector |>.maximum

theorem max_score_is_43 :
  maximum_score (List.replicate 10 [0, 1]) = 43 :=
sorry

end max_score_is_43_l573_573738


namespace P_is_linear_and_leading_coefficient_l573_573294

def isAngelic (n : ℕ) : Prop :=
  n > 10^100 ∧ (∀ k, nat.digits 10 n).all (λ k, k ∈ [1, 3, 5, 7, 8])

def sumOfDigits (n : ℕ) : ℕ :=
  (nat.digits 10 n).sum

noncomputable def P : polynomial ℕ := sorry -- P is a polynomial with nonnegative integer coefficients

theorem P_is_linear_and_leading_coefficient (P : ℕ → ℕ) 
  (hP : ∀ n : ℕ, n > 0 → isAngelic n → 
    let Pn := P (sumOfDigits n) 
    let sn := sumOfDigits (P n)
    in Pn.toDigits.containsAll sn.toDigits) :
    ∃ a₁ a₀ : ℕ, (a₁ = 1 ∨ (∃ k : ℕ, a₁ = 10^k)) ∧ (∀ x : ℕ, P x = a₁ * x + a₀) :=
begin
  sorry
end

end P_is_linear_and_leading_coefficient_l573_573294


namespace triangle_length_inradius_l573_573086

variable {A B C : ℝ} -- Angles in triangle ABC
variable {a b c R : ℝ} -- Sides opposite the angles and the circumcircle radius
variable {r : ℝ} -- Inradius

def sides_opposite_angles (A B C a b c : ℝ) : Prop :=
  ∃ a b c, true

def circumcircle_radius (R : ℝ) : Prop :=
  R = √3

def sine_conditions (A B C : ℝ) : Prop :=
  sin B ^ 2 + sin C ^ 2 - sin B * sin C = sin A ^ 2 

def length_of_side_a (a : ℝ) : Prop :=
  a = 3

def inradius_range (r : ℝ) : Prop :=
  0 < r ∧ r ≤ √3 / 2

theorem triangle_length_inradius
  (a b c A B C R : ℝ)
  (h_sides : sides_opposite_angles A B C a b c)
  (h_circumcircle : circumcircle_radius R)
  (h_sine : sine_conditions A B C) :
  length_of_side_a a ∧ inradius_range r :=
sorry

end triangle_length_inradius_l573_573086


namespace sqrt_defined_value_l573_573518

theorem sqrt_defined_value (x : ℝ) (h : x ≥ 4) : x = 5 → true := 
by 
  intro hx
  sorry

end sqrt_defined_value_l573_573518


namespace probability_one_true_one_false_l573_573657

theorem probability_one_true_one_false :
  let propositions := [
    false,  -- Statement 1
    true,   -- Statement 2
    true,   -- Statement 3
    false   -- Statement 4
  ] 
  ∃ P : ℚ, P = 2/3 :=
by {
  have total_combinations := (4.choose 2).to_nat,
  have favorable_combinations := (2.choose 1).to_nat * (2.choose 1).to_nat,
  have P := (favorable_combinations : ℚ) / total_combinations,
  use P,
  exact P = 2 / 3,
  sorry
}

end probability_one_true_one_false_l573_573657


namespace find_CM_dot_CB_l573_573126

variables {A B C M : Type} [AddCommGroup A] [VectorSpace ℝ A]
variables (C_angle : ℝ) (CA CB : ℝ) (BM MA : A) (CM CB_v : A)

-- Given conditions
def is_right_triangle (C_angle : ℝ) : Prop := C_angle = 90
def length_CA (CA : ℝ) : Prop := CA = 3
def length_CB (CB : ℝ) : Prop := CB = 3
def BM_equals_2MA (BM MA : A) : Prop := BM = 2 • MA

-- Theorem to prove
theorem find_CM_dot_CB (h1 : is_right_triangle C_angle) (h2 : length_CA CA) (h3 : length_CB CB)
  (h4 : BM_equals_2MA BM MA) :
  CM ⬝ CB_v = 3 :=
sorry

end find_CM_dot_CB_l573_573126


namespace solution_set_inequality_l573_573032

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 :=
sorry

end solution_set_inequality_l573_573032


namespace equal_angles_l573_573444

-- Given that BD is the angle bisector of triangle ABC
def is_angle_bisector (BD : Line) (ABC : Triangle) : Prop := sorry

-- E is the foot of the perpendicular from A to BD
def is_perpendicular_foot (E : Point) (A : Point) (BD : Line) : Prop := sorry

-- F is the foot of the perpendicular from C to BD
def is_perpendicular_foot_F (F : Point) (C : Point) (BD : Line) : Prop := sorry

-- P is the foot of the perpendicular from D to BC
def is_perpendicular_foot_P (P : Point) (D : Point) (BC : Line) : Prop := sorry

-- Prove that ∠DPE = ∠DPF
theorem equal_angles (ABC : Triangle) (A B C D E F P : Point) (BD BC : Line) :
  is_angle_bisector BD ABC →
  is_perpendicular_foot E A BD →
  is_perpendicular_foot_F F C BD →
  is_perpendicular_foot_P P D BC →
  ∠ D P E = ∠ D P F :=
by
  sorry

end equal_angles_l573_573444


namespace expand_product_l573_573010

theorem expand_product : ∀ (x : ℝ), (3 * x - 4) * (2 * x + 9) = 6 * x^2 + 19 * x - 36 :=
by
  intro x
  sorry

end expand_product_l573_573010


namespace hyperbola_eccentricity_hyperbola_equation_exists_fixed_point_G_l573_573985

open Real

def hyperbola (x y a b : ℝ) : Prop := x^2 / (a^2) - y^2 / (b^2) = 1
def dist (x1 y1 x2 y2 : ℝ) := sqrt((x2 - x1)^2 + (y2 - y1)^2)

-- Assuming (x1, y1) := (P’s coordinates), (x2, y2) := (F1’s coordinates), (x3, y3) := (F2’s coordinates)
variables (x1 y1 x2 y2 x3 y3 a b m: ℝ)
variable (P_hyperbola : hyperbola x1 y1 a b)
variable (dist_PF1_eq_2dist_PF2 : dist x1 y1 x2 y2 = 2 * dist x1 y1 x3 y3)
variable (F1_ortho_F2: (* some orthogonal condition involving PF1 and PF2*) sorry)
variable (asymptote_cond : (* conditions on vector involving P1 and P2 *) sorry)

-- Part (Ⅰ) 
def eccentricity (a c : ℝ) := c / a

theorem hyperbola_eccentricity : 
  ∃ e, e = sqrt 5 ∧ ∃ c, eccentricity a c = e :=
by sorry

-- Part (Ⅱ)
theorem hyperbola_equation :
  ∃ a, ∃ b, hyperbola x1 y1 a b ∧ asymptote_cond :=
by sorry

-- Part (Ⅲ) 
def vector_condition (Gx Gy : ℝ) : Prop := 
  ∃ (M N : ℝ × ℝ), (Gx, Gy) = (2 / (m), 0) ∧ (F1_ortho_F2 ∨ (some other vector condition))

theorem exists_fixed_point_G : 
  ∃ Gx Gy, vector_condition Gx Gy ∧ Gy = 0 :=
by sorry

end hyperbola_eccentricity_hyperbola_equation_exists_fixed_point_G_l573_573985


namespace evaluate_expression_l573_573766

theorem evaluate_expression (d : ℕ) (h : d = 4) : 
  (d^d - d*(d-2)^d + d^2)^(d-1) = 9004736 :=
by {
  subst h, -- Substitute d = 4
  -- Original expression: (4^4 - 4*(4-2)^4 + 4^2)^(4-1)
  -- Calculation: (256 - 64 + 16)^3 = 208^3 = 9004736
  sorry    
}

end evaluate_expression_l573_573766


namespace probability_dart_lands_inner_hexagon_l573_573683

theorem probability_dart_lands_inner_hexagon (s : ℝ) (h : s > 0) :
  let A_outer := (3 * Real.sqrt 3 / 2) * s^2 in
  let A_inner := (3 * Real.sqrt 3 / 8) * s^2 in
  let P := A_inner / A_outer in
  P = 1 / 4 :=
by
  sorry

end probability_dart_lands_inner_hexagon_l573_573683


namespace dividend_percentage_correct_l573_573681

-- Definitions based on given conditions
def face_value : ℝ := 50
def roi_percentage : ℝ := 25
def purchase_price : ℝ := 25

-- Dividend received per share
def dividend_per_share : ℝ := (roi_percentage / 100) * purchase_price

-- The dividend percentage paid by the company
def dividend_percentage := (dividend_per_share / face_value) * 100

-- Statement to prove
theorem dividend_percentage_correct : dividend_percentage = 12.5 := by
  -- Proof will go here
  sorry

end dividend_percentage_correct_l573_573681


namespace probability_is_one_fourth_l573_573154

noncomputable def probability_condition_met : ℝ :=
  (∫ (x : ℝ) in 0..2, ∫ (y : ℝ) in 2 * x..2, 1) / (∫ (x : ℝ) in 0..2, ∫ (y : ℝ) in 0..2, 1)

theorem probability_is_one_fourth :
  probability_condition_met = 1 / 4 :=
by
  sorry

end probability_is_one_fourth_l573_573154


namespace exists_infinitely_many_not_useful_but_not_optimized_l573_573734

-- Definition for "useful but not optimized" numbers
def useful_but_not_optimized (n : ℕ) : Prop :=
  ∃ (s t : Finset ℕ), n = s.sum (λ i, 3^i) + t.sum (λ j, 5^j)

-- Main theorem statement
theorem exists_infinitely_many_not_useful_but_not_optimized : 
  ∃ᶠ (n : ℕ) in Filter.atTop, ¬useful_but_not_optimized n :=
sorry

end exists_infinitely_many_not_useful_but_not_optimized_l573_573734


namespace profit_calculation_correct_l573_573692

-- Given conditions
def cost : ℝ := 80
def discount_rate : ℝ := 0.20
def discounted_price : ℝ := 130
def selling_price : ℝ := discounted_price / (1 - discount_rate)

-- Correct answer (Profit Percentage on retailer's cost)
def profit_percentage (cost selling_price : ℝ) : ℝ := ((selling_price - cost) / cost) * 100

theorem profit_calculation_correct :
  profit_percentage cost selling_price = 103.125 := by
  simp [cost, discount_rate, discounted_price, selling_price, profit_percentage]
  sorry

end profit_calculation_correct_l573_573692


namespace parabola_sequence_l573_573267

theorem parabola_sequence (m: ℝ) (n: ℕ):
  (∀ t s: ℝ, t * s = -1/4) →
  (∀ x y: ℝ, y^2 = (1/(3^n)) * m * (x - (m / 4) * (1 - (1/(3^n))))) :=
sorry

end parabola_sequence_l573_573267


namespace internet_usage_minutes_l573_573583

-- Define the given conditions
variables (M P E : ℕ)

-- Problem statement
theorem internet_usage_minutes (h : P ≠ 0) : 
  (∀ M P E : ℕ, ∃ y : ℕ, y = (100 * E * M) / P) :=
by {
  sorry
}

end internet_usage_minutes_l573_573583


namespace range_of_m_l573_573098

theorem range_of_m (a m : ℝ) (h_a : a > 0)
  (h_f : ∀ (x : ℝ), f x = a * x - (2 * a + 1) / x)
  (h_f_pos : ∀ (x : ℝ), 0 < x → f (m^2 + 1) > f (m^2 - m + 3))  :
  (m > 2) := 
  begin
    sorry
  end

end range_of_m_l573_573098


namespace periodic_sequence_l573_573953

theorem periodic_sequence 
  (r : ℕ) 
  (h_r : 0 < r)
  (a : ℕ → ℝ)
  (h : ∀ (m s : ℕ), ∃ (n : ℕ), m + 1 ≤ n ∧ n ≤ m + r ∧ 
       (∑ i in finset.range (s + 1), a (m + i) = ∑ i in finset.range (s + 1), a (n + i))
  ) : ∃ p ≥ 1, ∀ n ≥ 0, a (n + p) = a n :=
by
  sorry

end periodic_sequence_l573_573953


namespace derivative_at_pi_over_3_l573_573592

noncomputable def f (x : ℝ) : ℝ := (sin x - cos x) / (2 * cos x)
noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem derivative_at_pi_over_3 : f' (π / 3) = 2 :=
sorry

end derivative_at_pi_over_3_l573_573592


namespace find_x_y_l573_573113

theorem find_x_y (x y : ℝ) 
  (h1 : 3 * x = 0.75 * y)
  (h2 : x + y = 30) : x = 6 ∧ y = 24 := 
by
  sorry  -- Proof is omitted

end find_x_y_l573_573113


namespace telephone_call_duration_l573_573304

theorem telephone_call_duration (x : ℝ) :
  (0.60 + 0.06 * (x - 4) = 0.08 * x) → x = 18 :=
by
  sorry

end telephone_call_duration_l573_573304


namespace evaluate_polynomial_at_three_l573_573632

theorem evaluate_polynomial_at_three :
  let f : ℕ → ℕ := λ x, x^5 + x^3 + x^2 + x + 1
  in f 3 = 283 := by
  let f : ℕ → ℕ := λ x, x^5 + x^3 + x^2 + x + 1
  have h : f(3) = 283 := by {
    -- Calculation steps for our proof. We'll not define these here.
    -- It will be done in the proof part, so let's leave it as sorry.
    sorry
  }
  exact h

end evaluate_polynomial_at_three_l573_573632


namespace find_y_when_x_is_1_l573_573585

theorem find_y_when_x_is_1 (t : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = 1) : 
  y = 11 :=
by
  sorry

end find_y_when_x_is_1_l573_573585


namespace number_of_ways_to_place_coins_l573_573138

theorem number_of_ways_to_place_coins :
  (nat.choose 7 2) = 21 :=
by
  sorry

end number_of_ways_to_place_coins_l573_573138


namespace find_missing_number_l573_573004

theorem find_missing_number (square boxplus boxtimes boxminus : ℕ) :
  square = 423 / 47 ∧
  1448 = 282 * boxminus + (boxminus * 10 + boxtimes) ∧
  423 * (boxplus / 3) = 282 →
  square = 9 ∧
  boxminus = 5 ∧
  boxtimes = 8 ∧
  boxplus = 2 ∧
  9 = 9 :=
by
  intro h
  sorry

end find_missing_number_l573_573004


namespace quadratic_has_two_distinct_real_roots_l573_573571

/-- Given a quadratic equation with specific coefficients, prove it has two distinct real roots. -/
theorem quadratic_has_two_distinct_real_roots (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -2) (h₃ : c = -1) :
  let Δ := b^2 - 4 * a * c in
  Δ > 0 :=
by
  have a := h₁
  have b := h₂
  have c := h₃
  let Δ := b^2 - 4 * a * c
  -- Prove that Δ > 0 based on the given a, b, and c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l573_573571


namespace find_line_and_a_l573_573871

variable (l1 : ℝ → ℝ → Prop) (l2 : ℝ → ℝ → Prop) (a : ℝ)

-- Assuming the midpoint condition
def midpoint_condition : Prop :=
  ∃ P Q : ℝ × ℝ, l1 P.1 P.2 ∧ l2 Q.1 Q.2 ∧ (P.1 + Q.1) / 2 = 0 ∧ (P.2 + Q.2) / 2 = 0

-- Lines equations
def line1 (x y : ℝ) : Prop := 2 * x + y + 1 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y - 3 = 0

-- Parabola equation
def parabola (x : ℝ) : ℝ := a * x^2 - 1

-- Symmetry condition
def symmetric_points (x1 x2 : ℝ) : Prop :=
  ∀ t : ℝ, Δ (x1 x2 : Roots) > 0 ∧ x1 + x2 = 1/a ∧ x1 * x2 = -1/a

-- Discriminant for quadratic equation in the solution
def Δ (x1 x2 : ℝ) (a t : ℝ) : ℝ := 1 + 4 * a * (t + 1)

-- Main theorem
theorem find_line_and_a (a_ne_zero : a ≠ 0) :
  midpoint_condition line1 line2 →
  ∀ l, ( ∃ x y : ℝ, l = x + y ∧
    ( ∀ x1 x2 : ℝ, 
      ( ¬ symmetric_points (parabola x1) (parabola x2) l → a ≤ 3 / 4)
    )
  ) sorry

end find_line_and_a_l573_573871


namespace scatter_plot_correlation_l573_573358

-- Define the conditions
variable (SampleData : Type)
variable (scatterPlot : SampleData → Type)
variable (orderedPairs : SampleData → Type)
variable (positiveCorrelation : Type)
variable (negativeCorrelation : Type)

-- State the proof problem
theorem scatter_plot_correlation (data : SampleData) :
  ∃ plot : scatterPlot data, -- there exists a scatter plot for the given data
  (∃ posCorr : positiveCorrelation, ∃ negCorr : negativeCorrelation, 
    (posCorr ≠ negCorr ∧ ∃ sc : scatterPlot data, True)) := -- through this plot, one can see the distinction
sorry

end scatter_plot_correlation_l573_573358


namespace alice_can_find_town_with_one_outgoing_road_l573_573718

theorem alice_can_find_town_with_one_outgoing_road (n : ℕ) (h : 2 ≤ n) :
  ∃ m ≤ 4 * n, ∃ town_with_one_outgoing : (fin n → fin n) → Prop, town_with_one_outgoing :=
by
  sorry

end alice_can_find_town_with_one_outgoing_road_l573_573718


namespace train_cross_pole_time_l573_573936

def km_per_hr_to_m_per_s (v : ℝ) : ℝ := v * 1000 / 3600 

theorem train_cross_pole_time :
  ∀ (length : ℝ) (speed_km_per_hr : ℝ),
  length = 450 →
  speed_km_per_hr = 180 →
  let speed_m_per_s := km_per_hr_to_m_per_s speed_km_per_hr in
  let time := length / speed_m_per_s in
  time = 9 :=
by
  intros length speed_km_per_hr h_length h_speed
  let speed_m_per_s := km_per_hr_to_m_per_s speed_km_per_hr
  let time := length / speed_m_per_s
  have h_speed_m_per_s : speed_m_per_s = 50 :=
    by simp [km_per_hr_to_m_per_s, h_speed]
  rw [h_length, h_speed_m_per_s]
  norm_num
  exact rfl
  sorry

end train_cross_pole_time_l573_573936


namespace problem_solution_l573_573548

-- Definition: Sets Ai and S
def S := {1, 2, 3, ..., 2002}
def A (i : Fin 4) := Set S

-- Definition: Set of all quadruples F
def F := Set (A 0 × A 1 × A 2 × A 3)

noncomputable def union_cardinality_sum : ℕ :=
  ∑ (A1 A2 A3 A4 : A 0), (A1 ∪ A2 ∪ A3 ∪ A4).card

-- Theorem: Prove the sum of cardinalities of the union is as given
theorem problem_solution :
  union_cardinality_sum = 2^(8004) * 30030 :=
by
  sorry

end problem_solution_l573_573548


namespace wrapping_paper_needs_l573_573379

theorem wrapping_paper_needs :
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  first_present + second_present + third_present = 7 := by
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  sorry

end wrapping_paper_needs_l573_573379


namespace correct_statements_l573_573648

open ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω}
variable (A_rolls_6 B_rolls_5 both_roll_6 neither_roll_6 : Set Ω)

def independence_events (E F : Set Ω) : Prop :=
  μ (E ∩ F) = μ E * μ F

def complementary_events (E F : Set Ω) : Prop :=
  E = Fᶜ

axiom roll_equiprobable (p : ℝ) : 
  0 < p ∧ p < 1 ∧ p = 1 / 6

axiom independent_rolls :
  independence_events A_rolls_6 B_rolls_5

axiom complementary_rolls :
  complementary_events both_roll_6 neither_roll_6

theorem correct_statements :
  independence_events A_rolls_6 B_rolls_5 ∧
  complementary_events both_roll_6 neither_roll_6 :=
begin
  split;
  { sorry }
end

end correct_statements_l573_573648


namespace a_n_formula_T_n_bound_l573_573838

noncomputable def a_n (n : ℕ) : ℝ :=
if n = 0 then 0 else 3^(n - 1)

def S_n (n : ℕ) : ℝ :=
if n = 0 then 0 else (3/2) * a_n n - 1/2

def b_n (n : ℕ) : ℝ :=
if n = 0 then 0 else n / 3^n

noncomputable def T_n (n : ℕ) : ℝ :=
(finset.range n).sum (λ k, b_n (k + 1))

theorem a_n_formula (n : ℕ) (hn : 0 < n) : a_n n = 3^(n - 1) :=
sorry

theorem T_n_bound (n : ℕ) : ∃ k : ℝ, ∀ n : ℕ, T_n n < k :=
∃ (k : ℝ), k = 3 / 4 ∧ ∀ n : ℕ, T_n n < k :=
sorry

end a_n_formula_T_n_bound_l573_573838


namespace k_tuple_problem_l573_573414

theorem k_tuple_problem (k : ℕ) (h_k : k ≥ 1) 
(n : fin k → ℕ) (h_pos : ∀ i, 0 < n i) 
(h_gcd : nat.gcd (list.of_fn n).product = 1)
(h_div1 : ∀ i : fin (k - 1), (n i + 1) ^ (n i) - 1 ∣ n (i + 1))
(h_div2 : (n (k - 1) + 1) ^ (n (k - 1)) - 1 ∣ n 0) : 
  ∀ i : fin k, n i = 1 :=
begin
  sorry
end

end k_tuple_problem_l573_573414


namespace greatest_common_divisor_of_120_and_m_l573_573283

theorem greatest_common_divisor_of_120_and_m (m : ℕ) (h : (∀ d, d ∣ 120 ∧ d ∣ m → d = 1 ∨ d = 2 ∨ d = 4)) : gcd 120 m = 4 :=
by
  sorry

end greatest_common_divisor_of_120_and_m_l573_573283


namespace geometric_sequence_sum_l573_573134

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_seq_1 : a 6 - a 4 = 24)
  (h_seq_2 : a 3 * a 5 = 64) :
  (∑ i in Finset.range 8, a i) = 255 := 
by
  sorry

end geometric_sequence_sum_l573_573134


namespace min_value_f_min_value_exists_l573_573101

open set

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + a * x + (a^2) / 4 + 1

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -2 then
    (a^2) / 4 + a + 2
  else if -2 < a ∧ a < 2 then
    1
  else
    (a^2) / 4 - a + 2

theorem min_value_f (a : ℝ) :
  ∀ x ∈ Icc (-1:ℝ) 1, f x a ≥ g a :=
begin
  sorry
end

theorem min_value_exists (a : ℝ) :
  ∃ x ∈ Icc (-1:ℝ) 1, f x a = g a :=
begin
  sorry
end

end min_value_f_min_value_exists_l573_573101


namespace simplify_fraction_l573_573992

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l573_573992


namespace triangle_cot_identity_l573_573525

theorem triangle_cot_identity (a b c : ℝ) (h : 9 * a^2 + 9 * b^2 - 19 * c^2 = 0) :
  (Real.cot (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) / 
  ((Real.cot (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) + 
   (Real.cot (Real.arccos ((c^2 + a^2 - b^2) / (2 * c * a))))) = 5 / 9 := by
  sorry

end triangle_cot_identity_l573_573525


namespace find_complex_number_l573_573014

theorem find_complex_number 
  (z : ℂ) 
  (h : 2 * z + (5 - 3 * complex.I) = 6 + 11 * complex.I): 
  z = 1 / 2 + 7 * complex.I :=
by 
  sorry

end find_complex_number_l573_573014


namespace sin_cos_monotonically_increasing_intervals_l573_573406

theorem sin_cos_monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x y : ℝ,
    (2 * k * Real.pi - 3 * Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ 2 * k * Real.pi + Real.pi / 4) →
    (sin x + cos x ≤ sin y + cos y) :=
by
  intros k x y h
  sorry

end sin_cos_monotonically_increasing_intervals_l573_573406


namespace all_n_energetic_triplets_no_energetic_2004_2005_not_2007_triplet_l573_573067

noncomputable def is_n_energetic (a b c n : ℕ) : Prop := 
  (a^n + b^n + c^n) % (a + b + c) = 0

noncomputable def is_triplet_energetic_for_all_n (a b c : ℕ) : Prop :=
  ∀ n ≥ 1, is_n_energetic a b c n

noncomputable def is_triplet_energetic_for_specific_ns (a b c : ℕ) : Prop :=
  is_n_energetic a b c 2004 ∧ is_n_energetic a b c 2005 ∧ ¬is_n_energetic a b c 2007

theorem all_n_energetic_triplets :
  {t : ℕ × ℕ × ℕ | is_triplet_energetic_for_all_n t.1 t.2 t.3} = {(1, 1, 1), (1, 2, 3)} :=
sorry

theorem no_energetic_2004_2005_not_2007_triplet :
  ¬∃ t : ℕ × ℕ × ℕ, is_triplet_energetic_for_specific_ns t.1 t.2 t.3 :=
sorry

end all_n_energetic_triplets_no_energetic_2004_2005_not_2007_triplet_l573_573067


namespace semicircle_triang_BCF_area_m_plus_n_l573_573704

-- Definitions:
-- The total diameter AD of the semicircle is 30 units.
-- There exist specific points B, C, E, and F satisfying given conditions.

theorem semicircle_triang_BCF_area_m_plus_n : 
  ∀ (AD BC CD AB BF m n : ℝ), 
  AD = 30 → 
  AB = 10 → 
  BC = 10 → 
  CD = 10 → 
  ∃ (m n : ℕ), 
  m * sqrt n = 50 * sqrt 2 ∧ 
  ¬ (∃ p : ℕ, p^2 ∣ n) ∧ 
  m + n = 52 :=
by
  -- Any proof will be added here.
  sorry

end semicircle_triang_BCF_area_m_plus_n_l573_573704


namespace problem_l573_573540

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := λ x, Real.cos x
| (n + 1) := λ x, (f n)' x

theorem problem (x : ℝ) : f 2011 x = Real.sin x :=
by sorry

end problem_l573_573540


namespace find_n_from_combos_l573_573077

theorem find_n_from_combos (n : ℕ) (h1 : n ≥ 2) (h2 : C n 2 = C (n-1) 2 + C (n-1) 3) : n = 5 := 
sorry

end find_n_from_combos_l573_573077


namespace intersection_of_A_and_B_l573_573848

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l573_573848


namespace area_ABC_is_84_square_units_l573_573833

-- Define points A, B, C, D and distance function
variables {Point : Type*} [metric_space Point] (A B C D : Point)

-- Define conditions
def coplanar : Prop := ∃ plane : set Point, ∀ p ∈ {A, B, C, D}, p ∈ plane
def right_angle_D : Prop := ∠ (A - D) (C - D) = π / 2
def AC_eq_10 : dist A C = 10
def AB_eq_17 : dist A B = 17
def DC_eq_6 : dist D C = 6

theorem area_ABC_is_84_square_units (h1 : coplanar A B C D) (h2 : right_angle_D A C D)
(h3 : AC_eq_10 A C) (h4 : AB_eq_17 A B) (h5 : DC_eq_6 D C) : 
  ∃ area : ℝ, area = 84 := 
begin
  sorry
end

end area_ABC_is_84_square_units_l573_573833


namespace solve_for_x_l573_573299

theorem solve_for_x : (∃ x : ℝ, ((10 - 2 * x) ^ 2 = 4 * x ^ 2 + 16) ∧ x = 2.1) :=
by
  sorry

end solve_for_x_l573_573299


namespace fred_found_28_more_seashells_l573_573628

theorem fred_found_28_more_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (h_tom : tom_seashells = 15) (h_fred : fred_seashells = 43) : 
  fred_seashells - tom_seashells = 28 := 
by 
  sorry

end fred_found_28_more_seashells_l573_573628


namespace find_k_l573_573515

theorem find_k (x y k x1 y1 x2 y2 : ℝ) 
  (h1: dist (0, -√3) (x, y) + dist (0, √3) (x, y) = 4)
  (h2: x^2 + y^2 / 4 = 1)
  (h3: y1 = k * x1 + 1)
  (h4: y2 = k * x2 + 1)
  (h5: x_1^2 + y1^2 / 4 = 1)
  (h6: x_2^2 + y2^2 / 4 = 1)
  (h7: x1 * x2 + y1 * y2 = 0) :
  k = 1/2 ∨ k = -1/2 :=
sorry

end find_k_l573_573515


namespace find_A1B_l573_573156

variables (V : Type*) [AddCommGroup V] [Module ℝ V] 
variables (a b c : V) -- these correspond to vectors ↔ a, b, c.
variables (A B C A1 B1 C1 C1' : V)
variables (CA CB CC1 : V)

-- Define the relationships given in the conditions:
def def_CA : CA = a := rfl
def def_CB : CB = b := rfl
def def_CC1 : CC1 = c := rfl

-- Define the vector we need to prove is equal to the given expression
def A1B := (-a - c + b)

-- The theorem statement encapsulating the required proof
theorem find_A1B
  (h1 : CA = a)
  (h2 : CB = b)
  (h3 : CC1 = c) :
  A1B = (-a - c + b) :=
  by
    sorry

end find_A1B_l573_573156


namespace number_of_triplets_l573_573959

open Int

theorem number_of_triplets (k : ℤ) (h_pos : 0 < k) :
  (∃ (n : ℤ), n = (∑ x in (Finset.Icc 0 k).image (λ x, 3 * (k + 1)) ∪ (Finset.Icc (-k) (-1)).image (λ y, 3 * (k - 1)), 6 * k)) :=
sorry

end number_of_triplets_l573_573959


namespace houses_with_dogs_l573_573923

theorem houses_with_dogs (C B Total : ℕ) (hC : C = 30) (hB : B = 10) (hTotal : Total = 60) :
  ∃ D, D = 40 :=
by
  -- The overall proof would go here
  sorry

end houses_with_dogs_l573_573923


namespace sum_of_terms_l573_573176

noncomputable def fraction_series : ℚ :=
  ∑' n, if n % 2 = 0 then (↑(n/2 + 1)) / (2^(n + 2)) else (↑((n + 1) / 2 + 1)) / (3^(n + 2 / 2))

theorem sum_of_terms {a b : ℕ} (hp : a.gcd b = 1) (h : (a : ℚ) / b = fraction_series) : a + b = 721 :=
sorry

end sum_of_terms_l573_573176


namespace greatest_common_divisor_of_120_and_m_l573_573285

theorem greatest_common_divisor_of_120_and_m (m : ℕ) (h : (∀ d, d ∣ 120 ∧ d ∣ m → d = 1 ∨ d = 2 ∨ d = 4)) : gcd 120 m = 4 :=
by
  sorry

end greatest_common_divisor_of_120_and_m_l573_573285


namespace extreme_value_at_x_eq_neg3_l573_573254

theorem extreme_value_at_x_eq_neg3 (a : ℝ) :
  (∃ x : ℝ, (x = -3) ∧ (derivative (λ x : ℝ, x^3 + a * x^2 + 3 * x - 9) x = 0)) → a = 5 :=
by
  -- We state the condition that there is an extreme value at x = -3
  sorry

end extreme_value_at_x_eq_neg3_l573_573254


namespace birdhouse_distance_l573_573268

theorem birdhouse_distance (car_distance : ℕ) (lawnchair_distance : ℕ) (birdhouse_distance : ℕ) 
  (h1 : car_distance = 200) 
  (h2 : lawnchair_distance = 2 * car_distance) 
  (h3 : birdhouse_distance = 3 * lawnchair_distance) : 
  birdhouse_distance = 1200 :=
by
  sorry

end birdhouse_distance_l573_573268


namespace trigonometric_expression_value_l573_573413

noncomputable def trigonometric_expression (α : ℝ) : ℝ :=
  sin(α)^2 + sec(α)^2 + (1 / sec(α)^2) - (1 / cot(α)^2)

theorem trigonometric_expression_value (α : ℝ) : log 2 (trigonometric_expression α) = 1 :=
sorry

end trigonometric_expression_value_l573_573413


namespace smallest_area_of_ellipse_l573_573365

theorem smallest_area_of_ellipse 
    (a b : ℝ)
    (h1 : ∀ x y, (x - 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1)
    (h2 : ∀ x y, (x + 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1) :
    π * a * b = π :=
sorry

end smallest_area_of_ellipse_l573_573365


namespace investment_ratio_correct_l573_573605

variable (P Q : ℝ)
variable (investment_ratio: ℝ := 7 / 5)
variable (profit_ratio: ℝ := 7 / 10)
variable (time_p: ℝ := 7)
variable (time_q: ℝ := 14)

theorem investment_ratio_correct :
  (P * time_p) / (Q * time_q) = profit_ratio → (P / Q) = investment_ratio := 
by
  sorry

end investment_ratio_correct_l573_573605


namespace seat_notation_format_l573_573900

theorem seat_notation_format (r1 r2 s1 s2 : ℕ) : 
  (r1, s1) = (10, 3) → (r2, s2) = (6, 16) :=
by
  intro h
  rw h
  sorry

end seat_notation_format_l573_573900


namespace point_on_line_k_value_l573_573387

theorem point_on_line_k_value :
  ∀ (x0 y0 x1 y1 x2 y2 x3 k : ℕ),
    (x0 = -1) →
    (y0 = 1) →
    (x1 = 2) →
    (y1 = 5) →
    (x2 = 5) →
    (y2 = 9) →
    (x3 = 50) →
    (y1 - y0) * (x2 - x0) = (y2 - y0) * (x1 - x0) →
    (y2 - y1) * (x0 - x1) = (y0 - y1) * (x2 - x1) →
    (y2 - y0) * (x1 - x0) = (y1 - y0) * (x2 - x0) →
    (y3 = k) →
    (51 * (y2 - y1) / 3) + y0 = 69 :=
by
  intros
  sorry

end point_on_line_k_value_l573_573387


namespace sqrt_equation_solution_l573_573485

theorem sqrt_equation_solution (t : ℝ) : (sqrt (2 * sqrt (t - 2))) = (real.root 4 (7 - t)) → t = 3 :=
by
  sorry

end sqrt_equation_solution_l573_573485


namespace hyperbola_equation_l573_573090

theorem hyperbola_equation (a b : ℝ) (a_sq b_sq : ℝ)
  (ha_sq : a_sq = a^2) (hb_sq : b_sq = b^2)
  (asymptotes : ∀ x, y = ±(2/3) x)
  (passes_through : ∃ x y, (x = 3) ∧ (y = 4)
  (hyperbola_eq : ∀ x y, (y^2 / b_sq - x^2 / a_sq = 1)) :
  (y^2 / 12 - x^2 / 27) = 1 :=
begin
  sorry
end

end hyperbola_equation_l573_573090


namespace probability_of_stable_number_l573_573907

-- Definition of a "stable number"
def is_stable_number (n : ℕ) : Prop :=
  let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
  (digits.nth 0 = some 1 ∨ digits.nth 0 = some 2 ∨ digits.nth 0 = some 3) ∧
  (digits.nth 1 = some 1 ∨ digits.nth 1 = some 2 ∨ digits.nth 1 = some 3) ∧
  (digits.nth 2 = some 1 ∨ digits.nth 2 = some 2 ∨ digits.nth 2 = some 3) ∧
  (|digits.nth 0.get_or_else 0 - digits.nth 1.get_or_else 0| ≤ 1) ∧
  (|digits.nth 1.get_or_else 0 - digits.nth 2.get_or_else 0| ≤ 1)

-- List of all possible three-digit numbers made from 1, 2, 3 without repetition
def all_possible_numbers : list ℕ := [123, 132, 213, 231, 312, 321]

-- Number of stable numbers
def count_stable_numbers : ℕ :=
  list.length (all_possible_numbers.filter is_stable_number)

-- Total number of possible three-digit numbers
def total_possible_numbers : ℕ := list.length all_possible_numbers

-- Probability of a number being stable
def stable_number_probability : ℚ :=
  count_stable_numbers / total_possible_numbers

-- Statement of the problem
theorem probability_of_stable_number : stable_number_probability = 1/3 :=
by {
  -- Indicate this part is not be filled, hence "sorry".
  sorry
}

end probability_of_stable_number_l573_573907


namespace angle_bisector_segment_conditional_equality_l573_573062

theorem angle_bisector_segment_conditional_equality
  (a1 b1 a2 b2 : ℝ)
  (h1 : ∃ (P : ℝ), ∃ (e1 e2 : ℝ → ℝ), (e1 P = a1 ∧ e2 P = b1) ∧ (e1 P = a2 ∧ e2 P = b2)) :
  (1 / a1 + 1 / b1 = 1 / a2 + 1 / b2) :=
by 
  sorry

end angle_bisector_segment_conditional_equality_l573_573062


namespace number_of_solutions_l573_573787
-- Importing the necessary library

-- Define the condition for the product of the sequence to be negative
noncomputable def condition (n : ℕ) :=
  (n > 0) ∧ (Finset.prod (Finset.range 49) (λ k, (n - 2 * (k + 1)) : ℤ) < 0)

-- State theorem corresponding to the mathematically equivalent proof problem
theorem number_of_solutions : 
  finset.filter condition (finset.range (99)).card = 24 :=
sorry

end number_of_solutions_l573_573787


namespace nat_condition_l573_573754

theorem nat_condition (n : ℕ) : (∀ d ∣ n, d + 1 ∣ n + 1) → (n = 1 ∨ (nat.prime n ∧ n % 2 = 1)) :=
by
  sorry

end nat_condition_l573_573754


namespace number_of_diagonals_octagon_heptagon_diff_l573_573119

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_octagon_heptagon_diff :
  let A := number_of_diagonals 8
  let B := number_of_diagonals 7
  A - B = 6 :=
by
  sorry

end number_of_diagonals_octagon_heptagon_diff_l573_573119


namespace intersection_point_on_line_BC_l573_573161

open EuclideanGeometry

variables {A B C M N X : Point}
variables (Γ_B Γ_C : Circle)

/- Given Conditions -/
def midpoint (A B M : Point) := dist A M = dist M B /\ ∃ L, B = midpoint_coordinates M L
def tangent (AX : Line) (circumcircle_ABC : Circle) := tangent_line AX circumcircle_ABC
def circle_tangent (MX : Line) (Γ : Circle) := circle_tangent_line MX Γ

/- Main Theorem -/
theorem intersection_point_on_line_BC 
  (h_midpoint_M : midpoint A B M)
  (h_midpoint_N : midpoint A C N)
  (h_tangent_AX_X : tangent X (circumcircle A B C))
  (h_circle_tangent_ΓB : circle_tangent MX Γ_B)
  (h_circle_tangent_ΓC : circle_tangent NX Γ_C) 
  : Line.passes_through (intersection_points Γ_B Γ_C) B C :=
sorry

end intersection_point_on_line_BC_l573_573161


namespace smallest_number_with_conditions_l573_573031

theorem smallest_number_with_conditions :
  ∃ x : ℕ, x ≡ 2 [MOD 5] ∧ x ≡ 2 [MOD 4] ∧ x ≡ 3 [MOD 6] ∧ x = 102 :=
begin
  sorry
end

end smallest_number_with_conditions_l573_573031


namespace num_positive_integers_satisfying_l573_573802

theorem num_positive_integers_satisfying (n : ℕ) :
  (∑ k in (finset.range 25), (if (even (2 + 4 * k)) then 1 else 0) = 24) :=
sorry

end num_positive_integers_satisfying_l573_573802


namespace range_of_a_l573_573238

variables (a : ℝ)

def p : Prop := a > 1
def q : Prop := ∀ x : ℝ, -x^2 + 2 * x - 2 ≤ a

theorem range_of_a (h1 : p ∨ q) (h2 : ¬ (p ∧ q)) : -1 ≤ a ∧ a ≤ 1 := 
sorry

end range_of_a_l573_573238


namespace count_equilateral_triangles_l573_573886

noncomputable def number_of_equilateral_triangles : ℕ :=
  let lines := { m : ℤ | -12 ≤ m ∧ m ≤ 12 }.image (λ m, ({(x, y) : ℝ × ℝ | y = m} ∪ {(x, y) : ℝ × ℝ | y = sqrt 3 * x + 3 * m} ∪ {(x, y) : ℝ × ℝ | y = -sqrt 3 * x + 3 * m})) in
  1224

theorem count_equilateral_triangles : number_of_equilateral_triangles = 1224 :=
sorry

end count_equilateral_triangles_l573_573886


namespace find_closest_s_l573_573397

def u (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3 * s, -1 - 2 * s, 2 + 4 * s)

def b : ℝ × ℝ × ℝ := (-3, 2, -1)

def direction : ℝ × ℝ × ℝ := (3, -2, 4)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def closest_s (s : ℝ) : Prop :=
  dot_product ⟨(u s).1 - b.1, (u s).2 - b.2, (u s).3 - b.3⟩ direction = 0

theorem find_closest_s : ∃ s : ℝ, closest_s s ∧ s = -30 / 29 :=
sorry

end find_closest_s_l573_573397


namespace correlation_coefficient_linear_regression_l573_573504

-- Defining the given conditions
def sum_x_variances : ℝ := 10
def sum_y_variances : ℝ := 430
def sum_xy_covariances : ℝ := 65
def sqrt_4300_approx : ℝ := 65.57
def mean_x : ℝ := 5
def mean_y : ℝ := 58

-- Proving the calculation of correlation coefficient r
theorem correlation_coefficient :
  let r := sum_xy_covariances / (Real.sqrt sum_x_variances * Real.sqrt sum_y_variances)
  r ≈ 0.99 :=
by
  let r := 65 / (Real.sqrt 10 * Real.sqrt 430)
  have hsqrt : Real.sqrt 4300 ≈ 65.57 := sorry
  have hsq_eq : r ≈ 65 / 65.57 := sorry
  exact sorry

-- Proving the linear regression equation
theorem linear_regression :
  let b := sum_xy_covariances / sum_x_variances
  let a := mean_y - b * mean_x
  ∀ x : ℝ, y = b * x + a :=
by
  let b := 65 / 10
  let a := 58 - b * 5
  have hb : b ≈ 6.5 := sorry
  have ha : a ≈ 25.5 := sorry
  show ∀ x, y = b * x + a from
    assume x,
    exact sorry

end correlation_coefficient_linear_regression_l573_573504


namespace grid_values_constant_l573_573372

open Int

theorem grid_values_constant (f : ℤ × ℤ → ℕ)
  (h : ∀ x y, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ c : ℕ, ∀ x y, f (x, y) = c :=
begin
  -- proof goes here
  sorry
end

end grid_values_constant_l573_573372


namespace money_out_of_pocket_l573_573945

theorem money_out_of_pocket
  (old_system_cost : ℝ)
  (trade_in_percent : ℝ)
  (new_system_cost : ℝ)
  (discount_percent : ℝ)
  (trade_in_value : ℝ)
  (discount_value : ℝ)
  (discounted_price : ℝ)
  (money_out_of_pocket : ℝ) :
  old_system_cost = 250 →
  trade_in_percent = 80 / 100 →
  new_system_cost = 600 →
  discount_percent = 25 / 100 →
  trade_in_value = old_system_cost * trade_in_percent →
  discount_value = new_system_cost * discount_percent →
  discounted_price = new_system_cost - discount_value →
  money_out_of_pocket = discounted_price - trade_in_value →
  money_out_of_pocket = 250 := by
  intros
  sorry

end money_out_of_pocket_l573_573945


namespace part1_part2_l573_573059

noncomputable def point_M (m : ℝ) : ℝ × ℝ := (2 * m + 1, m - 4)
def point_N : ℝ × ℝ := (5, 2)

theorem part1 (m : ℝ) (h : m - 4 = 2) : point_M m = (13, 2) := by
  sorry

theorem part2 (m : ℝ) (h : 2 * m + 1 = 3) : point_M m = (3, -3) := by
  sorry

end part1_part2_l573_573059


namespace removed_numbers_correct_l573_573819

namespace FourEvenNumbers

def four_removed_numbers (n : ℕ) (mean : ℚ) : list ℕ :=
  if n = 100 ∧ mean = 51.5625 then [22, 24, 26, 28] else []

theorem removed_numbers_correct (n : ℕ) (mean : ℚ) :
  n = 100 → mean = 51.5625 → four_removed_numbers n mean = [22, 24, 26, 28] :=
by
  intros hn hmean
  simp [four_removed_numbers, hn, hmean]

end FourEvenNumbers

end removed_numbers_correct_l573_573819


namespace sample_of_population_l573_573929

-- Definitions and conditions
def n : ℕ := 200
def is_population (batch : Type) := ∀ x : batch, true

-- The batch of products is the population
variable (batch : Type)
variable [pop: is_population batch]

-- Prove that the 200 products represent a sample of the population
theorem sample_of_population : n = 200 → ∃ (sample : Set batch), sample.size = n :=
by
  intro h
  sorry

end sample_of_population_l573_573929


namespace count_positive_integers_satisfying_product_inequality_l573_573808

theorem count_positive_integers_satisfying_product_inequality :
  ∃ (k : ℕ), k = 23 ∧ 
  (n : ℕ ) → (2 ≤ n ∧ n < 100) →
  ((∃ (m: ℕ), 2 + 4 * m = n) ∧ ((n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0) = 23 :=
by
  sorry

end count_positive_integers_satisfying_product_inequality_l573_573808


namespace range_of_x_logarithmic_inequality_l573_573000

theorem range_of_x_logarithmic_inequality :
  { x : ℝ // x < 0 } → { x : ℝ // x ∈ Iio 0 } → 
    (log 2 (-x) < x + 1) ↔ x ∈ Ioo (-1 : ℝ) (-0.28879 : ℝ)
:= by
  sorry

end range_of_x_logarithmic_inequality_l573_573000


namespace range_of_f_cos_A_minus_B_l573_573881

-- Define the function f and its properties
def f (x : ℝ) : ℝ := 2 * sin (x + π / 3) * cos x

-- First proof statement: Range of f(x) for 0 ≤ x ≤ π / 2
theorem range_of_f : set.Icc 0 (π / 2) ⊆ f '' set.Icc (0 : ℝ) (π / 2) := sorry

-- Define the given conditions for the triangle
variables {a b c : ℝ} {A B C : ℝ}

-- Conditions in the triangle
hypothesis (h1 : A ∈ set.Ioi 0 ∩ set.Iio (π / 2)) -- A is an acute angle
hypothesis (h2 : f A = √3 / 2)
hypothesis (hb : b = 2)
hypothesis (hc : c = 3)

-- Second proof statement: Finding cos(A - B)
theorem cos_A_minus_B : ∃ B : ℝ, cos (A - B) = (5 * sqrt 7) / 14 :=
by
  -- Use the Law of Cosines and other trigonometric properties to prove the required theorem
  sorry

end range_of_f_cos_A_minus_B_l573_573881


namespace roger_steps_to_minutes_l573_573575

theorem roger_steps_to_minutes (h1 : ∃ t: ℕ, t = 30 ∧ ∃ s: ℕ, s = 2000)
                               (h2 : ∃ g: ℕ, g = 10000) :
  ∃ m: ℕ, m = 150 :=
by 
  sorry

end roger_steps_to_minutes_l573_573575


namespace evaluate_expression_l573_573765

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^a + (b^a)^b = 793) := by
  -- The following lines skip the proof but outline the structure:
  sorry

end evaluate_expression_l573_573765


namespace maximum_weight_allowed_l573_573278

noncomputable theory

def weight_of_socks := 2
def weight_of_underwear := 4
def weight_of_shirt := 5
def weight_of_shorts := 8
def weight_of_pants := 10

def count_pants := 1
def count_shirts := 2
def count_shorts := 1
def count_socks := 3
def count_underwear := 4

def weight_pants := count_pants * weight_of_pants
def weight_shirts := count_shirts * weight_of_shirt
def weight_shorts := count_shorts * weight_of_shorts
def weight_socks := count_socks * weight_of_socks
def weight_underwear := count_underwear * weight_of_underwear

def total_weight := weight_pants + weight_shirts + weight_shorts + weight_socks + weight_underwear

theorem maximum_weight_allowed : total_weight = 50 := by
  sorry

end maximum_weight_allowed_l573_573278


namespace new_supervisor_salary_correct_l573_573310

noncomputable def salary_new_supervisor
  (avg_salary_old : ℝ)
  (old_supervisor_salary : ℝ)
  (avg_salary_new : ℝ)
  (workers_count : ℝ)
  (total_salary_workers : ℝ := (avg_salary_old * (workers_count + 1)) - old_supervisor_salary)
  (new_supervisor_salary : ℝ := (avg_salary_new * (workers_count + 1)) - total_salary_workers)
  : ℝ :=
  new_supervisor_salary

theorem new_supervisor_salary_correct :
  salary_new_supervisor 430 870 420 8 = 780 :=
by
  simp [salary_new_supervisor]
  sorry

end new_supervisor_salary_correct_l573_573310


namespace no_intersections_and_sum_is_zero_l573_573255

def g (x : ℝ) : ℝ :=
  if x ≤ -2 then 2 * x + 2
  else if x ≤ -1 then -x - 1
  else if x ≤ 1 then 2 * x - 1
  else if x ≤ 2 then -x + 2
  else 2 * x - 4

theorem no_intersections_and_sum_is_zero :
  (∀ x, g x ≠ x + 2) → 
  (∑ x in {x : ℝ | g x = x + 2}, x) = 0 := 
by
  intro h
  simp
  sorry

end no_intersections_and_sum_is_zero_l573_573255


namespace red_beads_cost_l573_573570

theorem red_beads_cost (R : ℝ) (H : 4 * R + 4 * 2 = 10 * 1.72) : R = 2.30 :=
by
  sorry

end red_beads_cost_l573_573570


namespace angle_A_is_two_pi_over_three_ratio_b_over_c_l573_573491

-- Define the vectors and conditions
variables {A B C : ℝ}
variables {a b c : ℝ}
variables (vector_a : ℝ × ℝ := (sqrt 3, 1))
variables (vector_b : ℝ × ℝ := (sin A, cos A))
variables (angle_ab : ℝ := π / 3)

-- Conditions
axiom triangle_ABC : a > 0 ∧ b > 0 ∧ c > 0
axiom angle_between_a_b : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 
                          sqrt 3 * sin A + cos A = abs_vector_a * abs_vector_b * cos(angle_ab)
axiom sin_B_minus_C_eq : sin (B - C) = 2 * cos B * sin C
axiom sum_of_angles : A + B + C = π

-- Prove the measure of angle A
theorem angle_A_is_two_pi_over_three (h1 : triangle_ABC)
                                     (h2 : angle_between_a_b) :
  A = 2 * π / 3 := by
  sorry

-- Prove the value of b/c
theorem ratio_b_over_c (h1 : triangle_ABC)
                       (h2 : sin_B_minus_C_eq)
                       (h3 : angle_A_is_two_pi_over_three) :
  b / c = (3 * sqrt 13 - 3) / 2 := by
  sorry

end angle_A_is_two_pi_over_three_ratio_b_over_c_l573_573491


namespace polynomial_remainder_l573_573696

variable (p : ℚ[X]) -- Assume p(x) is a polynomial with rational coefficients.

theorem polynomial_remainder :
  p.eval 3 = 7 → p.eval (-2) = -3 → ∃ q : ℚ[X], p = (X - 3) * (X + 2) * q + 2 * X + 1 :=
by
  intro h1 h2
  let a := 2
  let b := 1
  use (p - (2 * X + 1)) / ((X - 3) * (X + 2))
  sorry

end polynomial_remainder_l573_573696


namespace sum_a3_a7_l573_573837

noncomputable def a_n (n : ℕ) := 4n + 1

def S_n (n : ℕ) := 2n^2 + 3n

theorem sum_a3_a7 : a_n 3 + a_n 7 = 42 :=
by {
  sorry
}

end sum_a3_a7_l573_573837


namespace distance_between_parallel_lines_l573_573742

noncomputable def distance_of_parallel_lines : ℝ :=
  let a := ⟨5, 1⟩ : ℝ × ℝ
  let b := ⟨6, -2⟩ : ℝ × ℝ
  let d := ⟨2, -4⟩ : ℝ × ℝ
  let v := (b.1 - a.1, b.2 - a.2)
  let dot_v_d := v.1 * d.1 + v.2 * d.2
  let dot_d_d := d.1 * d.1 + d.2 * d.2
  let proj_v_d := (dot_v_d / dot_d_d) * d.1, (dot_v_d / dot_d_d) * d.2
  let p := (v.1 - proj_v_d.1, v.2 - proj_v_d.2)
  Real.sqrt (p.1 * p.1 + p.2 * p.2)

theorem distance_between_parallel_lines :
  distance_of_parallel_lines = Real.sqrt 5 / 5 := sorry

end distance_between_parallel_lines_l573_573742


namespace value_of_r_when_n_is_three_l573_573182

theorem value_of_r_when_n_is_three :
  ∀ (r : ℤ), r = 3^s - 3*s ∧ s = 5^3 + 1 → r = 3^(5^3 + 1) - 3 * (5^3 + 1) :=
by
  intro r
  intro h
  cases h with h1 h2
  rw h2 at h1
  exact h1

end value_of_r_when_n_is_three_l573_573182


namespace coordinates_of_2000th_point_in_spiral_l573_573511

theorem coordinates_of_2000th_point_in_spiral :
    ∃ (x y : ℕ), spiral_enumeration (2000) = (x, y) ∧ (x, y) = (44, 19) := 
sorry

def spiral_enumeration : ℕ → ℕ × ℕ :=
sorry

end coordinates_of_2000th_point_in_spiral_l573_573511


namespace tan_double_phi_sin_cos_expression_l573_573858

noncomputable def phi : ℝ := sorry
axiom phi_in_bound (h1: 0 < phi) (h2: phi < Real.pi) : True
axiom tan_phi_plus_pi_over_4 : Real.tan (phi + Real.pi / 4) = -1 / 3

theorem tan_double_phi : Real.tan (2 * phi) = 4 / 3 := by
  sorry

theorem sin_cos_expression : (Real.sin phi + Real.cos phi) / (2 * Real.cos phi - Real.sin phi) = -1 / 4 := by
  sorry

end tan_double_phi_sin_cos_expression_l573_573858


namespace slope_of_intersection_line_l573_573746

theorem slope_of_intersection_line 
    (h1 : ∀ x y : ℝ, x^2 + y^2 - 6 * x + 4 * y - 20 = 0)
    (h2 : ∀ x y : ℝ, x^2 + y^2 - 8 * x + 6 * y + 12 = 0) :
    slope (line (intersection_points h1 h2)) = 1 :=
sorry

end slope_of_intersection_line_l573_573746


namespace b_first_12_digits_are_9_l573_573606

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions
axiom condition_1 : 0 < a ∧ a < a + (1 / 2) ∧ a + (1 / 2) ≤ b
axiom condition_2 : a^40 + b^40 = 1

-- State the theorem
theorem b_first_12_digits_are_9 : 
  ∀ (b : ℝ), (0 < a ∧ a < a + 1 / 2 ∧ a + 1 / 2 ≤ b) ∧ (a^40 + b^40 = 1) → 
  string.take 12 (string.drop 2 (toString b)) = "999999999999" :=
by
  intros,
  sorry

end b_first_12_digits_are_9_l573_573606


namespace mrs_thomson_savings_l573_573978

variables (X : ℝ)

def amount_saved (X : ℝ) : ℝ :=
  let after_food := X * 0.625 in
  let after_clothes := after_food * 0.78 in
  let after_household := after_clothes * 0.85 in
  let after_stocks := after_household * 0.70 in
  let after_tuition := after_stocks * 0.60 in
  after_tuition

theorem mrs_thomson_savings (X : ℝ) : amount_saved X = 0.1740375 * X := by sorry

end mrs_thomson_savings_l573_573978


namespace count_positive_integers_satisfying_product_inequality_l573_573805

theorem count_positive_integers_satisfying_product_inequality :
  ∃ (k : ℕ), k = 23 ∧ 
  (n : ℕ ) → (2 ≤ n ∧ n < 100) →
  ((∃ (m: ℕ), 2 + 4 * m = n) ∧ ((n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0) = 23 :=
by
  sorry

end count_positive_integers_satisfying_product_inequality_l573_573805


namespace polygons_red_vs_green_l573_573311

theorem polygons_red_vs_green :
  let n := 2006
  let k := 2007
  let R := ∑ i in finset.range (n + 1), if i ≥ 3 then (nat.choose n i) else 0
  let G := ∑ i in finset.range (k + 1), if i ≥ 3 then (nat.choose n (i - 1)) else 0
  G > R := by sorry

end polygons_red_vs_green_l573_573311


namespace car_average_speed_l573_573673

theorem car_average_speed 
  (d1 d2 d3 d5 d6 d7 d8 : ℝ) 
  (t_total : ℝ) 
  (avg_speed : ℝ)
  (h1 : d1 = 90)
  (h2 : d2 = 50)
  (h3 : d3 = 70)
  (h5 : d5 = 80)
  (h6 : d6 = 60)
  (h7 : d7 = -40)
  (h8 : d8 = -55)
  (h_t_total : t_total = 8)
  (h_avg_speed : avg_speed = (d1 + d2 + d3 + d5 + d6 + d7 + d8) / t_total) :
  avg_speed = 31.875 := 
by sorry

end car_average_speed_l573_573673


namespace jason_retirement_age_l573_573942

def age_at_retirement (initial_age years_to_chief extra_years_ratio years_after_masterchief : ℕ) : ℕ :=
  initial_age + years_to_chief + (years_to_chief * extra_years_ratio / 100) + years_after_masterchief

theorem jason_retirement_age :
  age_at_retirement 18 8 25 10 = 46 :=
by
  sorry

end jason_retirement_age_l573_573942


namespace dodecagon_squares_count_l573_573514

noncomputable def num_squares_in_dodecagon : ℕ := 183

theorem dodecagon_squares_count (A : Fin 12 → Point) (h : regular_dodecagon A):
  num_squares_in_dodecagon = 183 :=
by
  sorry

end dodecagon_squares_count_l573_573514


namespace min_value_frac_sum_l573_573051

theorem min_value_frac_sum (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
    (habc : a * b + b * c + c * a = 1) : 
    ∃ (x : ℝ), x = (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ∧ x = (9 + 3 * Real.sqrt 3) / 2 :=
by {
  use (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)),
  split,
  sorry,
}

end min_value_frac_sum_l573_573051


namespace shooting_probability_l573_573334

def hitting_prob (p : ℝ) : ℕ → ℝ := λ k, (nat.choose 3 k) * (p^k) * ((1-p)^(3-k))

theorem shooting_probability :
  (hitting_prob 0.6 2 + hitting_prob 0.6 3) = 81 / 125 :=
by
  sorry

end shooting_probability_l573_573334


namespace general_term_sequence_l573_573890

noncomputable def sequence (n : ℕ) : ℕ := 
  if n = 1 then 3 
  else 4 * sequence (n - 1) + 3

theorem general_term_sequence (n : ℕ) (h : n > 0) : sequence n = 4 ^ n - 1 := by
  sorry

end general_term_sequence_l573_573890


namespace tan_theta_value_l573_573124

theorem tan_theta_value (theta : ℝ)
  (x y : ℝ)
  (h1 : x = - (sqrt 3) / 2)
  (h2 : y = 1 / 2)
  (h3 : cos theta = x)
  (h4 : sin theta = y) :
  tan theta = - (sqrt 3) / 3 :=
by sorry

end tan_theta_value_l573_573124


namespace number_of_solutions_l573_573785
-- Importing the necessary library

-- Define the condition for the product of the sequence to be negative
noncomputable def condition (n : ℕ) :=
  (n > 0) ∧ (Finset.prod (Finset.range 49) (λ k, (n - 2 * (k + 1)) : ℤ) < 0)

-- State theorem corresponding to the mathematically equivalent proof problem
theorem number_of_solutions : 
  finset.filter condition (finset.range (99)).card = 24 :=
sorry

end number_of_solutions_l573_573785


namespace find_a_from_expansion_l573_573911

theorem find_a_from_expansion (a : ℝ) (h : expand_of_x3 (1 + a * x)^5 = -80) : a = -2 := 
sorry

noncomputable def expand_of_x3 (s : polynomial ℝ) : ℝ :=
s.coeff 3

end find_a_from_expansion_l573_573911


namespace part1_part2_l573_573038

variable {x : ℝ} 

-- Prove that for all real numbers x, |x + 6| + |x - 1| ≥ m implies m ≤ 7
theorem part1 (m : ℝ) (h : ∀ x : ℝ, |x + 6| + |x - 1| ≥ m) : m ≤ 7 :=
sorry

-- Given m = 7, solve the inequality |x - 4| - 3x ≤ 5
theorem part2 (m : ℝ) (hx : m = 7) : ∀ x : ℝ, |x - 4| - 3x ≤ 2 * m - 9 ↔ x ∈ Set.Ici (-1/4) :=
sorry

end part1_part2_l573_573038


namespace probability_snow_at_least_once_l573_573212

-- Define the probabilities given in the conditions
def p_day_1_3 : ℚ := 1 / 3
def p_day_4_7 : ℚ := 1 / 4
def p_day_8_10 : ℚ := 1 / 2

-- Define the complementary no-snow probabilities
def p_no_snow_day_1_3 : ℚ := 2 / 3
def p_no_snow_day_4_7 : ℚ := 3 / 4
def p_no_snow_day_8_10 : ℚ := 1 / 2

-- Compute the total probability of no snow for all ten days
def p_no_snow_all_days : ℚ :=
  (p_no_snow_day_1_3 ^ 3) * (p_no_snow_day_4_7 ^ 4) * (p_no_snow_day_8_10 ^ 3)

-- Define the proof problem: Calculate probability of at least one snow day
theorem probability_snow_at_least_once : (1 - p_no_snow_all_days) = 2277 / 2304 := by
  sorry

end probability_snow_at_least_once_l573_573212


namespace problem_l573_573171

theorem problem (a : ℕ → ℝ) (h0 : a 1 = 0) (h9 : a 9 = 0)
  (h2_8 : ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i > 0) (h_nonneg : ∀ n, 1 ≤ n ∧ n ≤ 9 → a n ≥ 0) : 
  (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 2 * a i) ∧ (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 1.9 * a i) := 
sorry

end problem_l573_573171


namespace tan_angle_ACQ_l573_573951

-- Definitions of the conditions
variables (ω ω1 ω2 : Circle)
variables (A B P Q C : Point)
variable (AB : Line)
variable [semicircle ω AB]

-- Hypotheses based on the conditions
hypotheses
  (h1 : is_diameter A B ω)
  (h2 : tangent ω1 ω ∧ tangent ω1 AB ∧ tangent ω2 ω ∧ tangent ω2 AB)
  (h3 : tangent ω1 ω2)
  (h4 : tangent_point ω1 AB P ∧ tangent_point ω2 AB Q)
  (h5 : P_between_A_Q A P Q)
  (h6 : tangent_point ω1 ω C)

-- The main theorem statement
theorem tan_angle_ACQ : 
  ∃ tanACQ : ℝ, tanACQ = 3 + 2 * real.sqrt 2 := 
by 
  use 3 + 2 * real.sqrt 2
  sorry

end tan_angle_ACQ_l573_573951


namespace evaluate_odd_sines_product_l573_573767

theorem evaluate_odd_sines_product :
  (∏ n in Finset.range 45, sin ((2*n + 1 : ℕ) * (Real.pi / 180))) = real.sqrt 2 / 2^45 := sorry

end evaluate_odd_sines_product_l573_573767


namespace tetrahedron_minimum_g_value_l573_573195

noncomputable def minimum_value_of_g (PQ RS PR QS PS QR : ℝ) : ℝ :=
  let m := 4
  let n := 884
  m * Real.sqrt n

theorem tetrahedron_minimum_g_value :
  PQ = 30 → RS = 30 → PR = 48 → QS = 48 → PS = 58 → QR = 58 →
  (let g (Y : Point) := dist Y P + dist Y Q + dist Y R + dist Y S in
  ∃ (Y : Point), g(Y) = minimum_value_of_g PQ RS PR QS PS QR) ∧
  (4 + 884 = 888) := 
by
  intros hPQ hRS hPR hQS hPS hQR
  sorry

end tetrahedron_minimum_g_value_l573_573195


namespace lean_proof_l573_573450

noncomputable def proof_problem (f : ℕ → ℤ) (p q : ℤ) : Prop :=
  f(2) = p ∧ f(3) = q ∧ (∀ a b : ℕ, f(a * b) = f(a) + f(b)) → f(72) = 3 * p + 2 * q

-- Here is the statement without the proof
theorem lean_proof (f : ℕ → ℤ) (p q : ℤ) (h1 : f(2) = p) (h2 : f(3) = q)
  (h3 : ∀ a b : ℕ, f(a * b) = f(a) + f(b)) : f(72) = 3 * p + 2 * q :=
by
  sorry

end lean_proof_l573_573450


namespace pizza_topping_count_l573_573695

theorem pizza_topping_count (n : ℕ) (h : n = 6) : n + n * (n - 1) / 2 = 21 :=
by
  rw [h]
  sorry

end pizza_topping_count_l573_573695


namespace find_functions_l573_573312

variable (f : ℝ → ℝ)

def isFunctionPositiveReal := ∀ x : ℝ, x > 0 → f x > 0

axiom functional_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) : f (x ^ y) = f x ^ f y

theorem find_functions (hf : isFunctionPositiveReal f) :
  (∀ x : ℝ, x > 0 → f x = 1) ∨ (∀ x : ℝ, x > 0 → f x = x) := sorry

end find_functions_l573_573312


namespace ratio_of_volumes_l573_573816

theorem ratio_of_volumes (h r : ℝ) (hcylinder : ℝ := π * r^2 * h) (hcone : ℝ := (1/3:ℝ) * π * r^2 * (h / 3)) :
  (hcone / hcylinder) = 1 / 9 :=
by
  -- Define the volumes based on the given conditions
  let vcylinder := π * r^2 * h
  let vcone := (1/3:ℝ) * π * r^2 * (h / 3)
  -- Calculate the ratio and simplify
  calc
    vcone / vcylinder
        = ((1/3:ℝ) * π * r^2 * (h / 3)) / (π * r^2 * h) : rfl
    ... = (1/9:ℝ) : sorry

end ratio_of_volumes_l573_573816


namespace sum_of_long_vectors_is_zero_l573_573436

-- Definition of a "long" vector
def is_long {α : Type*} [InnerProductSpace ℝ α] (v : α) (s : multiset α) : Prop :=
  ∥v∥ ≥ ∥(s.erase v).sum∥

theorem sum_of_long_vectors_is_zero {α : Type*} [InnerProductSpace ℝ α] (s : multiset α) (h : ∀ v ∈ s, is_long v s) (hn : 2 < s.card) :
  s.sum = 0 :=
by
  sorry

end sum_of_long_vectors_is_zero_l573_573436


namespace digit_80_after_decimal_of_one_seventh_l573_573295

theorem digit_80_after_decimal_of_one_seventh : 
  let seq := "142857"
  let n := 80
  let period := seq.length
  let pos := n % period
  seq.get ⟨pos, sorry⟩ = '4' := 
by
  let seq := "142857"
  let n := 80
  let period := seq.length
  let pos := n % period
  -- The position needs to be within the string bounds
  have : pos < seq.length := sorry
  exact sorry

end digit_80_after_decimal_of_one_seventh_l573_573295


namespace intersection_of_A_and_B_l573_573847

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l573_573847


namespace number_of_solutions_l573_573786
-- Importing the necessary library

-- Define the condition for the product of the sequence to be negative
noncomputable def condition (n : ℕ) :=
  (n > 0) ∧ (Finset.prod (Finset.range 49) (λ k, (n - 2 * (k + 1)) : ℤ) < 0)

-- State theorem corresponding to the mathematically equivalent proof problem
theorem number_of_solutions : 
  finset.filter condition (finset.range (99)).card = 24 :=
sorry

end number_of_solutions_l573_573786


namespace num_permutations_l573_573019

def permutations_count (n : ℕ) : ℕ := 
  if n % 2 = 0 then 1 else 2^((n-1)/2)

theorem num_permutations {a : ℕ → ℕ} (h : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ (a : ℕ), 1 ≤ a ∧ a ≤ n ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ n) → |a k - k| ≥ (n-1)/2) : permutations_count n = if n % 2 = 0 then 1 else 2^((n-1)/2) :=
by
  sorry

end num_permutations_l573_573019


namespace sum_exp_series_l573_573740

noncomputable def omega : ℂ := complex.exp (2 * real.pi * complex.I / 15)

theorem sum_exp_series : 
  ∑ i in (finset.range 14).map (add_left_embedding 1), complex.exp (2 * real.pi * complex.I * i / 15) = -1 :=
by
  sorry

end sum_exp_series_l573_573740


namespace find_blue_beads_per_row_l573_573377

-- Given the conditions of the problem:
def number_of_purple_beads : ℕ := 50 * 20
def number_of_gold_beads : ℕ := 80
def total_cost : ℕ := 180

-- Define the main theorem to solve for the number of blue beads per row.
theorem find_blue_beads_per_row (x : ℕ) :
  (number_of_purple_beads + 40 * x + number_of_gold_beads = total_cost) → x = (total_cost - (number_of_purple_beads + number_of_gold_beads)) / 40 := 
by {
  -- Proof steps would go here
  sorry
}

end find_blue_beads_per_row_l573_573377


namespace jason_retirement_age_l573_573940

variable (join_age : ℕ) (years_to_chief : ℕ) (percent_longer : ℕ) (additional_years : ℕ)

def time_to_master_chief := years_to_chief + (years_to_chief * percent_longer / 100)

def total_time_in_military := years_to_chief + time_to_master_chief years_to_chief percent_longer + additional_years

def retirement_age := join_age + total_time_in_military join_age years_to_chief percent_longer additional_years

theorem jason_retirement_age :
  join_age = 18 →
  years_to_chief = 8 →
  percent_longer = 25 →
  additional_years = 10 →
  retirement_age join_age years_to_chief percent_longer additional_years = 46 :=
by
  intros h1 h2 h3 h4
  simp [retirement_age, total_time_in_military, time_to_master_chief, h1, h2, h3, h4]
  sorry

end jason_retirement_age_l573_573940


namespace water_strider_probability_l573_573318

-- Defining the events A and B, and their probabilities
def A : Prop := true -- Event {the taken water contains water strider a}
def B : Prop := true -- Event {the taken water contains water strider b}

axiom prob_A : Prob A = 0.1
axiom prob_B : Prob B = 0.1

-- Axiom for independence
axiom independence : independent A B

-- Prove that the probability of finding a water strider in the taken water is 0.19
theorem water_strider_probability : Prob (A ∪ B) = 0.19 :=
by
  -- Using the formula for the probability of the union of two independent events
  have h_union : Prob (A ∪ B) = Prob A + Prob B - Prob (A ∩ B) := by sorry
  -- Using the given probabilities and the fact that A and B are independent
  have h_intersection : Prob (A ∩ B) = Prob A * Prob B := by sorry
  rw [h_union, h_intersection]
  exact sorry

end water_strider_probability_l573_573318


namespace g_satisfies_functional_equation_and_g4_is_16_l573_573958

noncomputable def g : ℝ → ℝ := λ x, x ^ 2

theorem g_satisfies_functional_equation_and_g4_is_16 :
  (∀ x y : ℝ, g (g x + y) = g (x ^ 2 - y) + 3 * g x * y + 1) →
  g 4 = 16 :=
begin
  intros h,
  -- proof of the theorem goes here
  sorry
end

end g_satisfies_functional_equation_and_g4_is_16_l573_573958


namespace celsius_to_fahrenheit_conversion_l573_573208

theorem celsius_to_fahrenheit_conversion (k b : ℝ) :
  (∀ C : ℝ, (C * k + b = C * 1.8 + 32)) → (k = 1.8 ∧ b = 32) :=
by
  intro h
  sorry

end celsius_to_fahrenheit_conversion_l573_573208


namespace kerosene_costs_as_much_as_eggs_l573_573498

theorem kerosene_costs_as_much_as_eggs :
  let cost_of_dozen_eggs : ℝ := 0.33
  let cost_of_liter_kerosene : ℝ := 0.22
  let cost_of_one_egg := cost_of_dozen_eggs / 12
  let cost_of_four_eggs := 4 * cost_of_one_egg
in cost_of_four_eggs = cost_of_liter_kerosene / 2 :=
by
  sorry

end kerosene_costs_as_much_as_eggs_l573_573498


namespace screen_width_l573_573350

theorem screen_width
  (A : ℝ) -- Area of the screen
  (h : ℝ) -- Height of the screen
  (w : ℝ) -- Width of the screen
  (area_eq : A = 21) -- Condition 1: Area is 21 sq ft
  (height_eq : h = 7) -- Condition 2: Height is 7 ft
  (area_formula : A = w * h) -- Condition 3: Area formula
  : w = 3 := -- Conclusion: Width is 3 ft
sorry

end screen_width_l573_573350


namespace distribute_candies_l573_573315

theorem distribute_candies (n : ℕ) (h : ∃ m : ℕ, n = 2^m) : 
  ∀ k : ℕ, ∃ i : ℕ, (1 / 2) * i * (i + 1) % n = k :=
sorry

end distribute_candies_l573_573315


namespace olivia_initial_money_l573_573273

theorem olivia_initial_money :
  ∃ x : ℕ, (x + 148 = 248) ∧ (x = 248 - 148) :=
by
  use 100
  constructor
  · sorry
  · sorry

end olivia_initial_money_l573_573273


namespace find_lambda_l573_573143

noncomputable def lambda_value (OA OB OC : ℝ) (λ : ℝ) : Prop := 
  let OM := (1 / 2) * OA + (1 / 6) * OB + λ * OC
  ∃ (MA MB MC : ℝ), (MA = OA - OM) ∧ (MB = OB - OM) ∧ (MC = OC - OM) ∧ 
  (MA * MB * MC = 0)

theorem find_lambda (OA OB OC : ℝ) :
  (λ : ℝ) → lambda_value OA OB OC λ → λ = 1 / 3 :=
  by sorry

end find_lambda_l573_573143


namespace rectangle_perimeter_l573_573313
-- Refined definitions and setup
variables (AB BC AE BE CF : ℝ)
-- Conditions provided in the problem
def conditions := AB = 2 * BC ∧ AE = 10 ∧ BE = 26 ∧ CF = 5
-- Perimeter calculation based on the conditions
def perimeter (AB BC : ℝ) : ℝ := 2 * (AB + BC)
-- Main theorem stating the conditions and required result
theorem rectangle_perimeter {m n : ℕ} (h: conditions AB BC AE BE CF) :
  m + n = 105 ∧ Int.gcd m n = 1 ∧ perimeter AB BC = m / n := sorry

end rectangle_perimeter_l573_573313


namespace num_positive_integers_satisfying_l573_573797

theorem num_positive_integers_satisfying (n : ℕ) :
  (∑ k in (finset.range 25), (if (even (2 + 4 * k)) then 1 else 0) = 24) :=
sorry

end num_positive_integers_satisfying_l573_573797


namespace arithmetic_sequence_a100_l573_573874

noncomputable def a_n : ℕ → ℝ := sorry

axiom sum_first_nine_terms : ∀ n, ∑ i in finset.range 9, a_n i = 27

axiom a_tenth_term : a_n 10 = 8

theorem arithmetic_sequence_a100 : a_n 100 = 98 := sorry

end arithmetic_sequence_a100_l573_573874


namespace sin_eq_tan_of_triangle_ABC_range_of_c_l573_573522

-- Conditions in a) are formalized as Lean definitions
variables {A B C : ℝ} {a b c : ℝ}
variables [triangle_abc : Triangle a b c] -- Hypothetical typeclass for triangle
notation "triangle_ABC" => (triangle_abc)
variable (h : a - b = b * Real.cos C)

-- Task (1): Prove that sin(C) = tan(B)
theorem sin_eq_tan_of_triangle_ABC (h : a - b = b * Real.cos C) : Real.sin C = Real.tan B :=
by sorry

-- Hypothesis for Task (2): Given a = 1 and C is acute,
variable (ha : a = 1)
variable (hC : 0 < C ∧ C < π / 2)

-- Task (2): Prove the range of c
theorem range_of_c (ha : a = 1) (hC : 0 < C ∧ C < π / 2) (hb : b = 1 / (1 + Real.cos C)) : (1 / 2) < c ∧ c < Real.sqrt 2 :=
by sorry

end sin_eq_tan_of_triangle_ABC_range_of_c_l573_573522


namespace find_x_l573_573192

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 - q.2)

theorem find_x : ∃ x : ℤ, ∃ y : ℤ, star (4, 5) (1, 3) = star (x, y) (2, 1) ∧ x = 3 :=
by 
  sorry

end find_x_l573_573192


namespace quadratic_non_real_roots_b_interval_l573_573486

theorem quadratic_non_real_roots_b_interval (b : ℝ) :
  (∀ x : ℂ, ¬ is_root (λ x : ℂ, x^2 + (b : ℂ) * x + (16 : ℂ)) x) -> (b ∈ set.Ioo (-8 : ℝ) 8) :=
by
  sorry

end quadratic_non_real_roots_b_interval_l573_573486


namespace possible_winning_scores_count_l573_573496

def total_runners := 15
def total_score := (total_runners * (total_runners + 1)) / 2

def min_score := 15
def max_potential_score := 39

def is_valid_winning_score (score : ℕ) : Prop :=
  min_score ≤ score ∧ score ≤ max_potential_score

theorem possible_winning_scores_count : 
  ∃ scores : Finset ℕ, ∀ score ∈ scores, is_valid_winning_score score ∧ Finset.card scores = 25 := 
sorry

end possible_winning_scores_count_l573_573496


namespace sin_squared_identity_l573_573577

theorem sin_squared_identity (x y : ℝ) : 
  sin x ^ 2 + sin (x + y) ^ 2 - 2 * sin x * sin y * sin (x + y) = 
  1 / 2 * (1 - cos x ^ 2 - cos y ^ 2) := 
by
  sorry

end sin_squared_identity_l573_573577


namespace courtyard_length_l573_573162

theorem courtyard_length (L : ℕ) (H : 3 * (0.4 * 4 * L * 25) + 1.5 * (0.6 * 4 * L * 25) = 2100) : L = 10 := 
by
  -- Proof omitted
  sorry

end courtyard_length_l573_573162


namespace complex_number_solution_l573_573877

theorem complex_number_solution (z i : ℂ) (h : z * (i - i^2) = 1 + i^3) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : 
  z = -i := 
by 
  sorry

end complex_number_solution_l573_573877


namespace counting_positive_integers_satisfying_inequality_l573_573781

theorem counting_positive_integers_satisfying_inequality :
  ∃ (n : ℕ), (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 49 → (n - 2 * k) < 0) ∧ n = 47 :=
begin
  sorry
end

end counting_positive_integers_satisfying_inequality_l573_573781


namespace seat_notation_format_l573_573901

theorem seat_notation_format (r1 r2 s1 s2 : ℕ) : 
  (r1, s1) = (10, 3) → (r2, s2) = (6, 16) :=
by
  intro h
  rw h
  sorry

end seat_notation_format_l573_573901


namespace fifth_term_sum_first_10_terms_l573_573980

-- Define the general term a_n
def a (n : ℕ) : ℚ :=
  if n > 0 then 1 / ((2 * n - 1) * (2 * n + 1)) else 0

-- Prove the 5th equation
theorem fifth_term : a 5 = 1 / 99 ∧ a 5 = (1 / 2) * (1 / 9 - 1 / 11) :=
  sorry

-- Define the sum of the first 10 terms
def sum_a (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, a (k + 1)

-- Prove the sum of the first 10 terms
theorem sum_first_10_terms : sum_a 10 = 10 / 21 :=
  sorry

end fifth_term_sum_first_10_terms_l573_573980


namespace f_2020_eq_neg_1_l573_573459

noncomputable def f: ℝ → ℝ :=
sorry

axiom f_2_x_eq_neg_f_x : ∀ x: ℝ, f (2 - x) = -f x
axiom f_x_minus_2_eq_f_neg_x : ∀ x: ℝ, f (x - 2) = f (-x)
axiom f_specific : ∀ x : ℝ, -1 < x ∧ x < 1 -> f x = x^2 + 1

theorem f_2020_eq_neg_1 : f 2020 = -1 :=
sorry

end f_2020_eq_neg_1_l573_573459


namespace count_valid_numbers_l573_573110

def is_geometric (ab bc cd : ℕ) : Prop :=
  10 * bc ^ 2 = (10 * bc + cd) * ab ∧ ab < bc ∧ bc < cd

def digit_range (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 9

def valid_number (abcd : ℕ) : Prop :=
  let a := abcd / 1000
  let b := (abcd / 100) % 10
  let c := (abcd / 10) % 10
  let d := abcd % 10
  a ≠ 0 ∧ 
  digit_range b ∧ digit_range c ∧ digit_range d ∧
  is_geometric (10 * a + b) (10 * b + c) (10 * c + d)

theorem count_valid_numbers : 
  (finset.range 10000).filter valid_number).card = 15 :=
sorry

end count_valid_numbers_l573_573110


namespace correct_memorization_order_l573_573425

-- Define the students
inductive Student
| A | B | C | D
deriving DecidableEq, Inhabited

open Student

-- Define the number of poems each student memorizes
variable (num_poems : Student → ℕ)

-- Conditions given in the problem
def condition_A := num_poems B < num_poems D
def condition_B := num_poems A > num_poems C
def condition_C := num_poems C > num_poems D
def condition_D := num_poems C > num_poems B

-- Unique memorization condition
def unique_memorization :
  num_poems A ≠ num_poems B ∧
  num_poems A ≠ num_poems C ∧
  num_poems A ≠ num_poems D ∧
  num_poems B ≠ num_poems C ∧
  num_poems B ≠ num_poems D ∧
  num_poems C ≠ num_poems D

-- Student with the least memorization tells the truth
def least_memorizes_truth (truth_teller : Student) : Prop :=
  ∀ s, (num_poems s ≥ num_poems truth_teller) ↔ (s = A ∧ condition_A) ∨
                                                         (s = B ∧ condition_B) ∨
                                                         (s = C ∧ condition_C) ∨
                                                         (s = D ∧ condition_D)

-- Ordering assertion
def correct_order :=
  num_poems D > num_poems B ∧
  num_poems B > num_poems C ∧
  num_poems C > num_poems A

-- Correct order theorem
theorem correct_memorization_order
  (h_unique_memorization : unique_memorization num_poems)
  (h_least_memorizes_truth : least_memorizes_truth num_poems A):
  correct_order num_poems :=
by {
  sorry
}

end correct_memorization_order_l573_573425


namespace distance_to_grocery_store_l573_573975

-- Definitions of given conditions
def miles_to_mall := 6
def miles_to_pet_store := 5
def miles_back_home := 9
def miles_per_gallon := 15
def cost_per_gallon := 3.5
def total_cost := 7

-- The Lean statement to prove the distance driven to the grocery store.
theorem distance_to_grocery_store (miles_to_mall miles_to_pet_store miles_back_home miles_per_gallon cost_per_gallon total_cost : ℝ) :
(total_cost / cost_per_gallon) * miles_per_gallon - (miles_to_mall + miles_to_pet_store + miles_back_home) = 10 := by
  sorry

end distance_to_grocery_store_l573_573975


namespace lean_proof_l573_573449

noncomputable def proof_problem (f : ℕ → ℤ) (p q : ℤ) : Prop :=
  f(2) = p ∧ f(3) = q ∧ (∀ a b : ℕ, f(a * b) = f(a) + f(b)) → f(72) = 3 * p + 2 * q

-- Here is the statement without the proof
theorem lean_proof (f : ℕ → ℤ) (p q : ℤ) (h1 : f(2) = p) (h2 : f(3) = q)
  (h3 : ∀ a b : ℕ, f(a * b) = f(a) + f(b)) : f(72) = 3 * p + 2 * q :=
by
  sorry

end lean_proof_l573_573449


namespace num_heterogeneous_towers_l573_573270

open Function

-- Define the set of acrobat numbers
def acrobats := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} 

-- Define what constitutes a Tower based on the problem's conditions
structure Tower :=
(circleA : set ℕ)
(circleB : set ℕ)
(h_circleA_size : circleA.card = 6)
(h_circleB_size : circleB.card = 6)
(h_in_circleA : ∀ n ∈ circleA, n ∈ acrobats)
(h_in_circleB : ∀ n ∈ circleB, n ∈ acrobats)
(h_b_sum : ∀ b ∈ circleB, ∃ a1 a2 ∈ circleA, b = a1 + a2)

-- Define heterogeneous towers
def isHeterogeneous (t1 t2 : Tower) : Prop :=
¬ (t1 = t2 ∨ ∃ (f : Perm ℕ), t1.circleA = (f '' t2.circleA) ∧ t1.circleB = (f '' t2.circleB))

-- Prove the number of heterogeneous towers
theorem num_heterogeneous_towers : ∃ towers : finset Tower, towers.card = 6 :=
begin
  sorry
end

end num_heterogeneous_towers_l573_573270


namespace min_friend_pairs_l573_573314

-- Define conditions
def n : ℕ := 2000
def invitations_per_person : ℕ := 1000
def total_invitations : ℕ := n * invitations_per_person

-- Mathematical problem statement
theorem min_friend_pairs : (total_invitations / 2) = 1000000 := 
by sorry

end min_friend_pairs_l573_573314


namespace greatest_common_divisor_of_three_divisors_l573_573289

theorem greatest_common_divisor_of_three_divisors (m : ℕ) (h1 : ∃ x, x ∣ 120 ∧ x ∣ m ∧ x > 0 ∧ (∀ d, d ∣ x → d = 1 ∨ d = 2 ∨ d = 4))
  : gcd 120 m = 4 :=
begin
  sorry,
end

end greatest_common_divisor_of_three_divisors_l573_573289


namespace sum_of_solutions_eq_seven_l573_573412

theorem sum_of_solutions_eq_seven :
  ∑ x in { x : ℝ | 2^(x^2 - 4 * x - 3) = 8^(x - 5) }, x = 7 :=
by
  sorry

end sum_of_solutions_eq_seven_l573_573412


namespace greatest_common_divisor_of_three_divisors_l573_573291

theorem greatest_common_divisor_of_three_divisors (m : ℕ) (h1 : ∃ x, x ∣ 120 ∧ x ∣ m ∧ x > 0 ∧ (∀ d, d ∣ x → d = 1 ∨ d = 2 ∨ d = 4))
  : gcd 120 m = 4 :=
begin
  sorry,
end

end greatest_common_divisor_of_three_divisors_l573_573291


namespace tin_capacity_difference_l573_573306

theorem tin_capacity_difference (h1 : π ≈ 3.14159):
  let volume_square_tin := 64 * 14,
      volume_circular_tin := π * (4 ^ 2) * 14 in
  abs (volume_circular_tin - volume_square_tin) ≈ 192.28384 :=
sorry

end tin_capacity_difference_l573_573306


namespace sin_2A_value_l573_573105

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h₁ : a / (2 * Real.cos A) = b / (3 * Real.cos B))
variable (h₂ : b / (3 * Real.cos B) = c / (6 * Real.cos C))

theorem sin_2A_value (h₃ : a / (2 * Real.cos A) = c / (6 * Real.cos C)) :
  Real.sin (2 * A) = 3 * Real.sqrt 11 / 10 := sorry

end sin_2A_value_l573_573105


namespace acute_isosceles_inscribed_in_circle_l573_573361

noncomputable def solve_problem : ℝ := by
  -- Let x be the angle BAC
  let x : ℝ := π * 5 / 11
  -- Considering the value of k in the problem statement
  let k : ℝ := 5 / 11
  -- Providing the value of k obtained from solving the problem
  exact k

theorem acute_isosceles_inscribed_in_circle (ABC : Type)
  [inhabited ABC]
  (inscribed : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (B_tangent C_tangent : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (D : ABC)
  (angle_eq : ∀ {A B C : ABC}, ∠ABC = ∠ACB)
  (triple_angle : ∀ {A B C D : ABC}, ∠ABC = 3 * ∠BAC) :
  solve_problem = 5 / 11 := 
sorry

end acute_isosceles_inscribed_in_circle_l573_573361


namespace find_number_of_valid_n_l573_573810

def valid_n (n : ℕ) : Prop :=
  (2 < n) ∧ (n < 100) ∧ ((∀ k : ℕ, k >= 0 → n = 4 * k + 3) ∨ (∃ k : ℕ, n = 4 * k + 3))

theorem find_number_of_valid_n : 
  {n : ℕ | valid_n n}.card = 24 :=
by
  sorry

end find_number_of_valid_n_l573_573810


namespace intersection_A_B_l573_573856

variable A : Set Int
variable B : Set Int

def setA : Set Int := {-1, 1, 2, 4}
def setB : Set Int := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  let A := setA
  let B := setB
  sorry

end intersection_A_B_l573_573856


namespace discount_percentage_l573_573558

theorem discount_percentage (price_paid : ℕ) (amount_saved : ℕ) (original_price : ℕ) : 
(price_paid = 120) → (amount_saved = 46) → (original_price = price_paid + amount_saved) → 
(46 / original_price) * 100 = 27.71 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end discount_percentage_l573_573558


namespace yellow_candies_range_l573_573163

-- Definitions for conditions
def total_candies : ℕ := 22

def num_colors : ℕ := 4

def is_most_yellow (yellow : ℕ) (other_colors : ℕ) : Prop :=
  yellow > other_colors -- since yellow is the most

-- Prove minimum and maximum yellow candies
theorem yellow_candies_range
  (yellow : ℕ)
  (other_colors : list ℕ)
  (h_sum : yellow + other_colors.sum = total_candies)
  (h_most : ∀ c ∈ other_colors, is_most_yellow yellow c)
  (h_turns : ∀ candies_being_taken (remaining : list ℕ),
             (remaining.length = 0 ∨ remaining.length = 1 ∨
              (∀ c ∈ remaining, c ≤ total_candies - yellow - remaining.sum))) :
  9 ≤ yellow ∧ yellow ≤ 11 :=
by {
  sorry
}

end yellow_candies_range_l573_573163


namespace biscuits_afternoon_eq_40_l573_573653

-- Define the initial conditions given in the problem.
def butter_cookies_afternoon : Nat := 10
def additional_biscuits : Nat := 30

-- Define the number of biscuits based on the initial conditions.
def biscuits_afternoon : Nat := butter_cookies_afternoon + additional_biscuits

-- The statement to prove according to the problem.
theorem biscuits_afternoon_eq_40 : biscuits_afternoon = 40 := by
  -- The proof is to be done, hence we use 'sorry'.
  sorry

end biscuits_afternoon_eq_40_l573_573653


namespace cubic_identity_l573_573116

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := 
by
  sorry

end cubic_identity_l573_573116


namespace rectangle_definition_l573_573391

variable (a b : ℝ)

def rectangle_as_solutions (x y : ℝ) : Prop :=
  (0 ≤ x ∧ x ≤ a) ∧ (0 ≤ y ∧ y ≤ b) ↔
  ∃ (u v: ℝ), u * (a - u) * v * (b - v) = 0 ∧
  (u * (a - u) = 0 ∨ v * (b - v) = 0)

theorem rectangle_definition :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ a) ∧ (0 ≤ y ∧ y ≤ b) →
    (∃ (w z: ℝ), sqrt w * sqrt (a - w) * sqrt z * sqrt (b - z) = 0 ∧
    (sqrt w * sqrt (a - w) = 0 ∨ sqrt z * sqrt (b - z) = 0)) :=
by
  sorry

end rectangle_definition_l573_573391


namespace percentage_within_one_std_dev_l573_573674

-- Define the conditions
variables {α : Type*} [measure_space α] [is_probability_measure α]
variables (m h : ℝ) (f : α → ℝ)

-- State these conditions
def symmetric_about_mean (m : ℝ) (f : α → ℝ) : Prop := 
  ∀ x, f (m + x) = f (m - x)

def less_than_m_plus_h (m h : ℝ) (f : α → ℝ) (p : ℝ) : Prop :=
  ∫ x in (-∞, m + h], f x = p

-- The main theorem
theorem percentage_within_one_std_dev (pmf : probability_mass_function α) (m h : ℝ) 
  (Hsym : symmetric_about_mean m pmf)
  (H84 : less_than_m_plus_h m h pmf 0.84) :
  ∫ x in (m - h, m + h), pmf x = 0.68 := 
sorry

end percentage_within_one_std_dev_l573_573674


namespace triangle_XYZ_r_s_max_sum_l573_573629

theorem triangle_XYZ_r_s_max_sum
  (r s : ℝ)
  (h_area : 1/2 * abs (r * (15 - 18) + 10 * (18 - s) + 20 * (s - 15)) = 90)
  (h_slope : s = -3 * r + 61.5) :
  r + s ≤ 42.91 :=
sorry

end triangle_XYZ_r_s_max_sum_l573_573629


namespace minimum_area_l573_573894

-- Define point A
def A : ℝ × ℝ := (-4, 0)

-- Define point B
def B : ℝ × ℝ := (0, 4)

-- Define the circle
def on_circle (C : ℝ × ℝ) : Prop := (C.1 - 2)^2 + C.2^2 = 2

-- Instantiating the proof of the minimum area of △ABC = 8
theorem minimum_area (C : ℝ × ℝ) (h : on_circle C) : 
  ∃ C : ℝ × ℝ, on_circle C ∧ 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) = 8 := 
sorry

end minimum_area_l573_573894


namespace greatest_common_divisor_of_120_and_m_l573_573284

theorem greatest_common_divisor_of_120_and_m (m : ℕ) (h : (∀ d, d ∣ 120 ∧ d ∣ m → d = 1 ∨ d = 2 ∨ d = 4)) : gcd 120 m = 4 :=
by
  sorry

end greatest_common_divisor_of_120_and_m_l573_573284


namespace correct_option_D_l573_573650

variable (a b : ℕ)

theorem correct_option_D : a^4 / a^3 = a :=
by {
  -- using exponent rules: a^m / a^n = a^(m-n)
  calc
    a^4 / a^3 = a^(4 - 3) : by rw [nat.pow_sub₀ (zero_le _) (le_refl 4)]
            ... = a^1    : by rw [sub_self]
            ... = a      : by rw [nat.pow_one]
}

end correct_option_D_l573_573650


namespace number_of_ordered_pairs_l573_573018

theorem number_of_ordered_pairs
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 50)
  (h2 : 0 ≤ b)
  (h3 : ∃ r s : ℕ, r + s = a ∧ r * s = b) :
  ∑ a in (finset.range 51).filter (λ x, 1 ≤ x), ((a + 1) / 2).ceil = 377 :=
by {
  sorry
}

end number_of_ordered_pairs_l573_573018


namespace last_two_digits_of_fraction_l573_573965

def greatest_integer_less_equal (x : ℝ) : ℤ := floor x

theorem last_two_digits_of_fraction :
  let f := greatest_integer_less_equal (10^93 / (10^31 + 3)) in
  f % 100 = 8 :=
by
  sorry

end last_two_digits_of_fraction_l573_573965


namespace real_solution_count_l573_573393

noncomputable def f (x : ℝ) : ℝ := ∑ k in (finset.range 50).image (λ k, k + 1), (k : ℝ) / (x - k)

theorem real_solution_count : ∃! x : ℝ, f(x) = x := sorry

end real_solution_count_l573_573393


namespace correct_answer_l573_573424

-- Define languages as types
inductive Language
| Jap | Fre | Eng | Ger

open Language

-- Define who speaks which language
structure Person :=
  (speaks : Language → Prop)

-- Given people
variable (A B C D : Person)

-- Conditions
variable (H1 : speaks A Jap ∧ ¬ speaks D Jap ∧ ∀ l, speaks A l ∧ speaks D l)
variable (H2 : ∀ p : Person, ¬ (speaks p Jap ∧ speaks p Fre))
variable (H3 : ¬ ∃ l, speaks A l ∧ speaks B l ∧ speaks C l ∧ speaks D l)
variable (H4 : ¬ speaks B Eng ∧ ∀ l, speaks A l ∧ speaks C l → speaks B l)

-- Expected languages each person speaks
def solution_correct :=
  speaks A Jap ∧ speaks A Ger ∧
  speaks B Fre ∧ speaks B Ger ∧
  speaks C Eng ∧ speaks C Fre ∧
  speaks D Eng ∧ speaks D Ger

theorem correct_answer : solution_correct A B C D :=
by
  sorry

end correct_answer_l573_573424


namespace geometric_coloring_l573_573933

/-- Given a geometric figure with eleven dots that can each be colored either red, white, or blue,
where each connected pair of dots (by a segment with no other dots between) must not be of the same color,
prove that the number of valid colorings of the dots is 72. -/
theorem geometric_coloring : 
  let colors := {red, white, blue} in 
  ∃ (ways : ℕ), 
  let dots : list color := 
    { (color_0, color_1, color_2), (color_3, color_4, color_2), (color_5, color_6), 
      (color_7, color_8, color_9, color_10)} in 
  -- Constraints for different colors for connected dots:
  ∀ i j, (dots[i], dots[j] is connected) → dots[i] ≠ dots[j]
  ways = 72 :=
  sorry

end geometric_coloring_l573_573933


namespace crystal_discount_is_50_percent_l573_573750

noncomputable def discount_percentage_original_prices_and_revenue
  (original_price_cupcake : ℝ)
  (original_price_cookie : ℝ)
  (total_cupcakes_sold : ℕ)
  (total_cookies_sold : ℕ)
  (total_revenue : ℝ)
  (percentage_discount : ℝ) :
  Prop :=
  total_cupcakes_sold * (original_price_cupcake * (1 - percentage_discount / 100)) +
  total_cookies_sold * (original_price_cookie * (1 - percentage_discount / 100)) = total_revenue

theorem crystal_discount_is_50_percent :
  discount_percentage_original_prices_and_revenue 3 2 16 8 32 50 :=
by sorry

end crystal_discount_is_50_percent_l573_573750


namespace regular_tetrahedron_surface_area_l573_573913

noncomputable def tetrahedron_volume (a : ℝ) : ℝ := (real.sqrt 2 / 12) * a^3

noncomputable def tetrahedron_surface_area (a : ℝ) : ℝ := real.sqrt 3 * a^2

theorem regular_tetrahedron_surface_area (V : ℝ) (S : ℝ) (h : V = (16 / 3) * real.sqrt 2) : 
  S = 16 * real.sqrt 3 :=
begin
  -- Hypothesis: The volume V is given and equals (16 / 3) * sqrt 2
  let a := 4,
  -- Hence, the edge length of the tetrahedron must be a = 4
  have ha : tetrahedron_volume a = V,
  { sorry },
  -- Then calculate the surface area using edge length a
  have hs : tetrahedron_surface_area a = S,
  { sorry },
  -- We should have S = tetrahedron_surface_area 4
  show S = tetrahedron_surface_area 4,
  from hs,
end

end regular_tetrahedron_surface_area_l573_573913


namespace students_not_making_the_cut_l573_573624

-- Define the total number of girls, boys, and the number of students called back
def number_of_girls : ℕ := 39
def number_of_boys : ℕ := 4
def students_called_back : ℕ := 26

-- Define the total number of students trying out
def total_students : ℕ := number_of_girls + number_of_boys

-- Formulate the problem statement as a theorem
theorem students_not_making_the_cut : total_students - students_called_back = 17 := 
by 
  -- Omitted proof, just the statement
  sorry

end students_not_making_the_cut_l573_573624


namespace age_of_john_l573_573166

theorem age_of_john (J S : ℕ) 
  (h1 : S = 2 * J)
  (h2 : S + (50 - J) = 60) :
  J = 10 :=
sorry

end age_of_john_l573_573166


namespace sum_of_distinct_prime_factors_156000_l573_573298

theorem sum_of_distinct_prime_factors_156000 : 
  let n := 156000 in
  (∑ p in { p : ℕ | p.prime ∧ p ∣ n }, p) = 23 :=
by
  sorry

end sum_of_distinct_prime_factors_156000_l573_573298


namespace general_formula_and_sum_l573_573065

noncomputable def a_sequence (a : ℕ → ℚ) : Prop :=
  a 1 + a 5 = 7 ∧ a 6 = 13 / 2 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

theorem general_formula_and_sum (a : ℕ → ℚ) 
  (h : a_sequence a) 
  (n : ℕ) : 
  (a n = (2 * n + 1) / 2) ∧ 
  (let a_inv_seq := λ n, 4 / ((2 * n + 1) * (2 * n + 3)) in
   let S_n := ∑ i in (finset.range n), a_inv_seq i in
   S_n = (4 * n) / (6 * n + 9)) :=
by 
  sorry

end general_formula_and_sum_l573_573065


namespace man_savings_percentage_l573_573687

theorem man_savings_percentage
  (salary expenses : ℝ)
  (increase_percentage : ℝ)
  (current_savings : ℝ)
  (P : ℝ)
  (h1 : salary = 7272.727272727273)
  (h2 : increase_percentage = 0.05)
  (h3 : current_savings = 400)
  (h4 : current_savings + (increase_percentage * salary) = (P / 100) * salary) :
  P = 10.5 := 
sorry

end man_savings_percentage_l573_573687


namespace line_perpendicular_to_intersection_l573_573827

variable {α β : Type}
variable [plane α] [plane β]
variable (a l : line)
variable (h1 : a ⊆ α)
variable (h2 : a ⊥ β)
variable (h3 : α ⊥ β)
variable (h4 : α ∩ β = l)

theorem line_perpendicular_to_intersection :
  a ⊆ α ∧ a ⊥ l :=
by
  sorry

end line_perpendicular_to_intersection_l573_573827


namespace find_g_expression_l573_573473

theorem find_g_expression (g : ℝ → ℝ) (h : ∀ x : ℝ, g(x + 2) = 2 * x - 3) : ∀ x : ℝ, g(x) = 2 * x - 11 :=
by
  sorry

end find_g_expression_l573_573473


namespace fixed_point_exists_l573_573979

theorem fixed_point_exists : ∃ (x y : ℝ), (∀ k : ℝ, (2 * k - 1) * x - (k + 3) * y - (k - 11) = 0) ∧ x = 2 ∧ y = 3 := 
by
  -- Placeholder for proof
  sorry

end fixed_point_exists_l573_573979


namespace constant_term_in_binomial_expansion_l573_573590

theorem constant_term_in_binomial_expansion :
  let x := (1/2: ℚ) in
  let n := 6 in
  let f (r: ℕ) := (1 / 2) ^ r * nat.choose n r * x ^ (n - 2 * r) in
  ∃ r, n - 2 * r = 0 ∧ f r = 5 / 2 :=
by
  sorry

end constant_term_in_binomial_expansion_l573_573590


namespace problem_statement_l573_573543

def f0 (x : ℝ) := Real.cos x
def f' (f : ℝ → ℝ) := λ x, Real.deriv f x

noncomputable def fn : ℕ → (ℝ → ℝ)
| 0     := f0
| (n+1) := f' (fn n)

theorem problem_statement (x : ℝ) : fn 2011 x = Real.sin x :=
sorry

end problem_statement_l573_573543


namespace point_outside_circle_l573_573872

open Real

-- Define the conditions
def O := (0, 0 : ℝ × ℝ)
def r := 5
def P := (8, 0 : ℝ × ℝ)

-- Define the distance function
def dist (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the property we want to prove
theorem point_outside_circle :
  dist P O > r := by
  -- Proof goes here
  sorry

end point_outside_circle_l573_573872


namespace number_of_same_family_functions_l573_573490

def sameFamily (f g : ℝ → ℝ) (R : set ℝ) : Prop :=
  ∀ x, g x ∈ R → f x = g x

theorem number_of_same_family_functions :
  let f : ℝ → ℝ := λ x, x^2;
      R : set ℝ := {1, 4};
      D₁ : set ℝ := {1, 2};
  (f ∈ D₁ → f '' D₁ = R) →
  (∀ g, sameFamily f g R → ∃ D : set ℝ, g ∈ D ) →
  ∃ domains, 8 = number_of_valid_domains domains R f :=
sorry

end number_of_same_family_functions_l573_573490


namespace sin_pi_six_minus_two_alpha_l573_573428

theorem sin_pi_six_minus_two_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = - 7 / 9 :=
by
  sorry

end sin_pi_six_minus_two_alpha_l573_573428


namespace find_A_l573_573301

def A : ℕ := 7 * 5 + 3

theorem find_A : A = 38 :=
by
  sorry

end find_A_l573_573301


namespace quadratic_no_discriminant_23_l573_573737

theorem quadratic_no_discriminant_23 (a b c : ℤ) (h_eq : b^2 - 4 * a * c = 23) : False := sorry

end quadratic_no_discriminant_23_l573_573737


namespace range_a_plus_b_l573_573053

def f (x a b : ℝ) : ℝ := a * 2^x + x^2 + b * x

theorem range_a_plus_b (a b : ℝ) (h : {x | f x a b = 0} = {x | f (f x a b) a b = 0} ∧ {x | f x a b = 0} ≠ ∅) :
  0 ≤ a + b ∧ a + b < 4 := sorry

end range_a_plus_b_l573_573053


namespace soap_ratio_l573_573691

/-- A marketing firm determined that, of 200 households surveyed, 
    80 used neither brand R nor brand B soap, 60 used only brand R soap, 
    and there were 40 households that used both brands of soap. 
    Prove that the ratio of households that used only brand B soap to 
    those that used both brands of soap is 1:2. -/
theorem soap_ratio : 
  let total_households := 200
  let neither_rb := 80
  let only_r := 60
  let both_rb := 40
  let used_soap := total_households - neither_rb
  let only_b := used_soap - only_r - both_rb
  in ratio only_b both_rb = (1 : ℕ) / (2 : ℕ) :=
by {
  sorry
}

end soap_ratio_l573_573691


namespace inverse_at_2_l573_573081

noncomputable def f : ℝ → ℝ := λ x, Real.log x / Real.log 2

noncomputable def f_inv (y : ℝ) : ℝ := 2^y - 1

theorem inverse_at_2 : f_inv 2 = 3 := by
  unfold f_inv
  simp
  norm_num
  rfl

end inverse_at_2_l573_573081


namespace minimum_expression_value_l573_573537

variables (a b c : ℝ^3)

-- Assume that a, b, and c are unit vectors
def is_unit_vector (v : ℝ^3) : Prop :=
  ∥v∥ = 1

-- Assume the dot product of a and b is 0
def orthogonal (u v : ℝ^3) : Prop :=
  u ⬝ v = 0

-- Define the expression we want to minimize
def expression (a b c : ℝ^3) : ℝ :=
  (a - c) ⬝ (b - c)

theorem minimum_expression_value :
  is_unit_vector a →
  is_unit_vector b →
  is_unit_vector c →
  orthogonal a b →
  expression a b c ≥ 1 - real.sqrt 2 :=
sorry

end minimum_expression_value_l573_573537


namespace swimming_pool_width_l573_573343

theorem swimming_pool_width 
  (V : ℝ) (L : ℝ) (B1 : ℝ) (B2 : ℝ) (h : ℝ)
  (h_volume : V = (h / 2) * (B1 + B2) * L) 
  (h_V : V = 270) 
  (h_L : L = 12) 
  (h_B1 : B1 = 1) 
  (h_B2 : B2 = 4) : 
  h = 9 :=
  sorry

end swimming_pool_width_l573_573343


namespace triangle_side_length_l573_573520

-- Definitions of the conditions
variables {A B C M N : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N]
variables [HasMedianTriangle A B C] (medAM : IsMedian A M) (medBN : IsMedian B N)
variables (perpendicular : IsPerpendicular medAM medBN)

-- Lengths of medians
def length_AM : ℝ := 15
def length_BN : ℝ := 20

-- Theorem statement
theorem triangle_side_length (hAM : ∃ M, length_AM = 15) (hBN : ∃ N, length_BN = 20)
  (h_perpendicular : IsPerpendicular medAM medBN) : 
  ∃ AB, AB = (10/3) * Real.sqrt 13 := 
sorry

end triangle_side_length_l573_573520


namespace complex_expression_equality_l573_573457

noncomputable def complex := ℂ

variable (x y : complex)
variable (hx : x ≠ 0) (hy : y ≠ 0)
variable (h : x^2 + x*y + y^2 = 0)

theorem complex_expression_equality : (x / (x + y)) ^ 1990 + (y / (x + y)) ^ 1990 = -1 :=
by
  sorry

end complex_expression_equality_l573_573457


namespace limit_problem_l573_573730

theorem limit_problem (h : ∀ x, x ≠ -3):
  (∀ x, (x^2 + 2*x - 3)^2 = (x + 3)^2 * (x - 1)^2) →
  (∀ x, x^3 + 4*x^2 + 3*x = x * (x + 1) * (x + 3)) →
  tendsto (λ x, ((x^2 + 2*x - 3)^2) / (x^3 + 4*x^2 + 3*x)) (𝓝[-] (-3)) (𝓝 0) :=
by
  intros numerator_factor denominator_factor
  sorry

end limit_problem_l573_573730


namespace ratio_AD_AB_l573_573524

noncomputable def angle_A := 60
noncomputable def angle_B := 45
noncomputable def angle_ADF := 45

constant AD DF AB : ℝ
constant area_equal : Prop

def area_divided_equal := area_equal
-- Given conditions of triangle and line creating equal areas
axiom triangle_ABC : ∠A = angle_A ∧ ∠B = angle_B
axiom line_DF : D ∈ line_AB ∧ ∠ADF = angle_ADF
axiom equal_area_partition : area_divided_equal

theorem ratio_AD_AB :
  (∠A = angle_A ∧ ∠B = angle_B ∧ D ∈ line_AB ∧ ∠ADF = angle_ADF ∧ area_divided_equal) →
  AD / AB = (real.sqrt 3 + 1) / 2 := 
begin
  sorry
end

end ratio_AD_AB_l573_573524


namespace prime_ge_7_div_30_l573_573118

theorem prime_ge_7_div_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end prime_ge_7_div_30_l573_573118


namespace tan_alpha_value_l573_573859

noncomputable def alpha : ℝ := sorry

theorem tan_alpha_value (h1 : sin alpha + cos alpha = (1 - sqrt 3) / 2)
  (h2 : 0 < alpha) (h3 : alpha < π) : tan alpha = -sqrt 3 / 3 :=
sorry

end tan_alpha_value_l573_573859


namespace find_number_of_valid_n_l573_573809

def valid_n (n : ℕ) : Prop :=
  (2 < n) ∧ (n < 100) ∧ ((∀ k : ℕ, k >= 0 → n = 4 * k + 3) ∨ (∃ k : ℕ, n = 4 * k + 3))

theorem find_number_of_valid_n : 
  {n : ℕ | valid_n n}.card = 24 :=
by
  sorry

end find_number_of_valid_n_l573_573809


namespace range_of_k_l573_573099

theorem range_of_k (k : ℝ) :
  (∃ a b : ℝ, -2 ≤ a ∧ b ≤ a ∧ ∀ x ∈ set.Icc a b, f x = k + sqrt (x + 2) ∧ f x ∈ set.Icc a b) ↔ k ∈ Ioo (-9 / 4 : ℝ) (-2 : ℝ) :=
sorry

end range_of_k_l573_573099


namespace compare_a_b_c_l573_573538

noncomputable def a : ℝ := (3/4)^(0.5)
noncomputable def b : ℝ := (4/3)^(0.4)
noncomputable def c : ℝ := Real.logBase (3/4) (Real.log 3 4)

theorem compare_a_b_c : c < a ∧ a < b :=
by
  sorry

end compare_a_b_c_l573_573538


namespace equally_spaced_sum_of_squares_is_rational_multiple_of_triangle_sides_l573_573986

variables {A B C : Point}
variables {A_1 A_2 A_3 ... A_n B_1 B_2 B_3 ... B_n C_1 C_2 C_3 ... C_n : Point}

def is_equally_spaced_on (points : list Point) (segment : Segment) : Prop :=
  ∀ i, (i < points.length - 1) → (distance (points.nth i) (points.nth (i+1)) = 
  distance (segment.start) (segment.end) / (points.length - 1))

theorem equally_spaced_sum_of_squares_is_rational_multiple_of_triangle_sides 
  (hA : is_equally_spaced_on [A_1, A_2, ..., A_n] ⟨B, C⟩)
  (hB : is_equally_spaced_on [B_1, B_2, ..., B_n] ⟨C, A⟩)
  (hC : is_equally_spaced_on [C_1, C_2, ..., C_n] ⟨A, B⟩) :
  ∃ (r : ℚ), 
  (∑ i in finset.range n, (distance A (A_i))^2 + 
   ∑ i in finset.range n, (distance B (B_i))^2 + 
   ∑ i in finset.range n, (distance C (C_i))^2) 
  = r * ((distance A B)^2 + (distance B C)^2 + (distance C A)^2) := 
sorry

end equally_spaced_sum_of_squares_is_rational_multiple_of_triangle_sides_l573_573986


namespace equal_angles_l573_573517

noncomputable theory

open_locale classical

variables {α : Type}

-- Definition for points A, B, C, D, E, F, G and lines in a scalene triangle
variables (A B C D E F G : α)

-- Predicate for points being collinear
def collinear (P Q R : α) : Prop := 
  -- Define collinearity based on geometry theory
  sorry

-- Predicate for angle, tangent to circle, and bisector properties
def angle_bisector (A B C D : α) : Prop := 
  -- Define angle bisector properties
  sorry

def tangent_to_circumcircle (A B D E : α) : Prop :=
  -- Define tangent properties of circumcircle
  sorry

def intersect (P Q R : α) : Prop := 
  -- Define intersection of lines PQ and PR
  sorry

-- Main theorem
theorem equal_angles
  (triangle_scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (D_property : angle_bisector A B C D)
  (tangent_ABD : tangent_to_circumcircle A B D E)
  (tangent_ACD : tangent_to_circumcircle A C D F)
  (G_property : intersect B E G ∧ intersect C F G) :
  angle A D F = angle E D G :=
begin
  sorry
end

end equal_angles_l573_573517


namespace determinant_squared_eq_zero_l573_573175

-- Let u, v, w be vectors in ℝ^3 such that u + v + w = 0.
variables (u v w : ℝ^3)
-- Assume the condition
axiom h : u + v + w = 0

-- Define the cross products for the columns of the matrix
def col1 := u × v
def col2 := v × w
def col3 := w × u

-- Define the matrix A assembled from these columns
noncomputable def A : Matrix ℝ 3 3 := ![![col1.x, col2.x, col3.x], ![col1.y, col2.y, col3.y], ![col1.z, col2.z, col3.z]]

-- Define the determinant K of matrix A and K' (K^2)
noncomputable def K : ℝ := Matrix.det A
noncomputable def K' : ℝ := K ^ 2

-- Prove that K' is 0 given the initial condition.
theorem determinant_squared_eq_zero : K' = 0 :=
sorry

end determinant_squared_eq_zero_l573_573175


namespace expression_value_l573_573905

theorem expression_value (x : ℝ) (h : x = -2) : (x * x^2 * (1/x) = 4) :=
by
  rw [h]
  sorry

end expression_value_l573_573905


namespace sum_binomial_coeff_l573_573568

open BigOperators

theorem sum_binomial_coeff (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n ≥ m) :
  ∑ k in Finset.range (m + 1), (-1 : ℤ)^k * (n.choose k) = (-1 : ℤ)^m * (n - 1).choose m :=
sorry

end sum_binomial_coeff_l573_573568


namespace green_square_area_percentage_l573_573711

noncomputable def flag_side_length (k: ℝ) : ℝ := k
noncomputable def cross_area_fraction : ℝ := 0.49
noncomputable def cross_area (k: ℝ) : ℝ := cross_area_fraction * k^2
noncomputable def cross_width (t: ℝ) : ℝ := t
noncomputable def green_square_side (x: ℝ) : ℝ := x
noncomputable def green_square_area (x: ℝ) : ℝ := x^2

theorem green_square_area_percentage (k: ℝ) (t: ℝ) (x: ℝ)
  (h1: x = 2 * t)
  (h2: 4 * t * (k - t) + x^2 = cross_area k)
  : green_square_area x / (k^2) * 100 = 6.01 :=
by
  sorry

end green_square_area_percentage_l573_573711


namespace problem_l573_573541

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := λ x, Real.cos x
| (n + 1) := λ x, (f n)' x

theorem problem (x : ℝ) : f 2011 x = Real.sin x :=
by sorry

end problem_l573_573541


namespace cloth_cost_price_per_metre_l573_573706

theorem cloth_cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) :
  total_metres = 300 → total_price = 18000 → loss_per_metre = 5 → (total_price / total_metres + loss_per_metre) = 65 :=
by
  intros
  sorry

end cloth_cost_price_per_metre_l573_573706


namespace domain_of_function_l573_573757

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 3 * x + 1 > 0
def condition2 (x : ℝ) : Prop := 2 - x ≠ 0

-- Define the domain of the function
def domain (x : ℝ) : Prop := x > -1 / 3 ∧ x ≠ 2

theorem domain_of_function : 
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ domain x := 
by
  sorry

end domain_of_function_l573_573757


namespace measure_angle_QRS_l573_573523

-- Define the angles in triangle PQR
def angle_PQR : ℝ := 42
def angle_PRQ : ℝ := 42

-- Define that PS bisects angle PQR
def bisects (P Q R S : Point) : Prop :=
  ∡ Q P S = ∡ Q P R / 2

theorem measure_angle_QRS
  (P Q R S : Point)
  (h1 : ∡ P Q R = angle_PQR) 
  (h2 : ∡ P R Q = angle_PRQ)
  (h3 : bisects P Q R S) :
  ∡ Q R S = 63 :=
by
  sorry

end measure_angle_QRS_l573_573523


namespace complex_number_solution_l573_573089

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) * (2 - I) = 5) : z = 2 + 3 * I :=
  sorry

end complex_number_solution_l573_573089


namespace find_angles_and_area_l573_573093

noncomputable def angles_in_arithmetic_progression (A B C : ℝ) : Prop :=
  A + C = 2 * B ∧ A + B + C = 180

noncomputable def side_ratios (a b : ℝ) : Prop :=
  a / b = Real.sqrt 2 / Real.sqrt 3

noncomputable def triangle_area (a b c A B C : ℝ) : ℝ :=
  (1/2) * a * c * Real.sin B

theorem find_angles_and_area :
  ∃ (A B C a b c : ℝ), 
    angles_in_arithmetic_progression A B C ∧ 
    side_ratios a b ∧ 
    c = 2 ∧ 
    A = 45 ∧ 
    B = 60 ∧ 
    C = 75 ∧ 
    triangle_area a b c A B C = 3 - Real.sqrt 3 :=
sorry

end find_angles_and_area_l573_573093


namespace claire_price_is_correct_l573_573974

open Real

-- Defining the conditions
def total_savings : ℝ := 86
def liam_oranges : ℝ := 40
def liam_price_per_two_oranges : ℝ := 2.5
def claire_oranges : ℝ := 30
def liam_price_per_orange : ℝ := liam_price_per_two_oranges / 2
def liam_earnings : ℝ := liam_oranges * liam_price_per_orange
def claire_earnings : ℝ := total_savings - liam_earnings
def claire_price_per_orange : ℝ := claire_earnings / claire_oranges

-- Stating the theorem
theorem claire_price_is_correct : claire_price_per_orange = 1.20 := 
by
  sorry

end claire_price_is_correct_l573_573974


namespace chocolates_for_charlie_l573_573947

theorem chocolates_for_charlie (
    chocolates_per_saturday_herself: ℕ,
    chocolates_per_saturday_sister: ℕ,
    total_chocolates: ℕ,
    saturdays_in_month: ℕ
) (h1: chocolates_per_saturday_herself = 2)
  (h2: chocolates_per_saturday_sister = 1)
  (h3: total_chocolates = 22)
  (h4: saturdays_in_month = 4) : 
  ∃ chocolates_for_charlie : ℕ, chocolates_for_charlie = 10 :=
by
  sorry

end chocolates_for_charlie_l573_573947


namespace triangle_side_ratio_range_l573_573158

theorem triangle_side_ratio_range (A B C a b c : ℝ) (h1 : A + 4 * B = 180) (h2 : C = 3 * B) (h3 : 0 < B ∧ B < 45) 
  (h4 : a / b = Real.sin (4 * B) / Real.sin B) : 
  1 < a / b ∧ a / b < 3 := 
sorry

end triangle_side_ratio_range_l573_573158


namespace part1_part2_l573_573824

variables {x m : ℝ}
def A : ℝ := -3 * x^2 - 2 * m * x + 3 * x + 1
def B : ℝ := 2 * x^2 + 2 * m * x - 1

theorem part1 : 2 * A + 3 * B = 2 * m * x + 6 * x - 1 := by
  sorry

theorem part2 (h : 2 * A + 3 * B = 2 * m * x + 6 * x - 1) : m = -3 := by
  have h_indep : 2 * m * x + 6 * x = 0 := by sorry
  have h_eqn : 2 * m + 6 = 0 := by sorry
  sorry

end part1_part2_l573_573824


namespace masha_mushrooms_l573_573043

theorem masha_mushrooms (B1 B2 B3 B4 G1 G2 G3 : ℕ) (total : B1 + B2 + B3 + B4 + G1 + G2 + G3 = 70)
  (girls_distinct : G1 ≠ G2 ∧ G1 ≠ G3 ∧ G2 ≠ G3)
  (boys_threshold : ∀ {A B C D : ℕ}, (A = B1 ∨ A = B2 ∨ A = B3 ∨ A = B4) →
                    (B = B1 ∨ B = B2 ∨ B = B3 ∨ B = B4) →
                    (C = B1 ∨ C = B2 ∨ C = B3 ∨ C = B4) → 
                    (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
                    A + B + C ≥ 43)
  (diff_no_more_than_five_times : ∀ {x y : ℕ}, (x = B1 ∨ x = B2 ∨ x = B3 ∨ x = B4 ∨ x = G1 ∨ x = G2 ∨ x = G3) →
                                  (y = B1 ∨ y = B2 ∨ y = B3 ∨ y = B4 ∨ y = G1 ∨ y = G2 ∨ y = G3) →
                                  x ≠ y → x ≤ 5 * y ∧ y ≤ 5 * x)
  (masha_max_girl : G3 = max G1 (max G2 G3))
  : G3 = 5 :=
sorry

end masha_mushrooms_l573_573043


namespace jason_retirement_age_l573_573941

def age_at_retirement (initial_age years_to_chief extra_years_ratio years_after_masterchief : ℕ) : ℕ :=
  initial_age + years_to_chief + (years_to_chief * extra_years_ratio / 100) + years_after_masterchief

theorem jason_retirement_age :
  age_at_retirement 18 8 25 10 = 46 :=
by
  sorry

end jason_retirement_age_l573_573941


namespace cube_root_of_64_l573_573374

theorem cube_root_of_64 : ∃ x : ℝ, x^3 = 64 ∧ x = 4 :=
by
  sorry

end cube_root_of_64_l573_573374


namespace angle_BAC_right_angle_l573_573513

variables {A B C O D E : Type*} [ordered_comm_ring A]

def isosceles_triangle (AB AC : A) : Prop := AB = AC
def midpoint (A B O : A) : Prop := O = (A + B) / 2
def OC_intersects_circle (O C AB D : A) : Prop := (O ≤ AB) ∧ (C = OC) ∧ (D ∈ OC)
def BD_intersects_AC (B D A C E : A) : Prop := (B = BD) ∧ (E ∈ AC)
def AE_eq_CD (AE CD : A) : Prop := AE = CD

theorem angle_BAC_right_angle 
(AB AC AE CD : A) 
(h1: isosceles_triangle AB AC)
(h2: midpoint A B O)
(h3: OC_intersects_circle O C AB D)
(h4: BD_intersects_AC B D A C E)
(h5: AE_eq_CD AE CD) : 
(∠BAC = 90) :=
sorry

end angle_BAC_right_angle_l573_573513


namespace fraction_equivalency_and_decimal_l573_573879

theorem fraction_equivalency_and_decimal :
  ∀ (a b c d : ℕ), a = 2 → b = 4 → c = 6 → d = 40 →
  (a / b = c / 12) ∧ (a / b = 20 / d) ∧ (20 / d = 0.5) :=
by
  intros a b c d ha hb hc hd
  -- Use the given information
  sorry

end fraction_equivalency_and_decimal_l573_573879


namespace X_completes_work_in_5_days_l573_573333

theorem X_completes_work_in_5_days (W : ℝ) 
  (Y_work_rate : ℝ := W / 20) 
  (Z_work_rate : ℝ := W / 30) 
  (X_work_rate : ℝ) 
  (work_completed_together : 2 * X_work_rate + 2 * Y_work_rate + 2 * Z_work_rate)
  (work_completed_by_Z : 13 * Z_work_rate)
  (total_work : work_completed_together + work_completed_by_Z = W) :
  W / X_work_rate = 5 :=
by
  sorry

end X_completes_work_in_5_days_l573_573333


namespace no_obtuse_angle_condition_l573_573756

open Function

theorem no_obtuse_angle_condition (n : ℕ) (S : Finset (Fin n → ℝ)) :
  (∀ (x y z : Fin n → ℝ), 
    x ≠ y → y ≠ z → x ≠ z → 
    (¬collinear {x, y, z}) →
    (∀ (c₁ c₂ c₃ : ℕ), 
      (c₁ = c₂ ∨ c₁ = c₃ ∨ c₂ = c₃ → ¬obtuse_angle x y z))) →
  (∀ (x y z : Fin n → ℝ), 
    c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ → 
    ¬obtuse_angle x y z)) →
  n ≤ 12 := 
sorry

/-- 
Definitions:
- collinear: A set of three points is collinear if they lie on a single straight line.
- obtuse_angle: A determined angle between three points is obtuse if it is greater than 90 degrees.

Conditions:
- n is the number of points in set S.
- The set S consists of n points in ℝ space.
- No three points in S are collinear.
- Any three points in S of the same color or with three different colors do not form an obtuse angle.
- An unlimited number of colors are available.

Goal:
Given the above conditions, prove that n must be less than or equal to 12.
-/

end no_obtuse_angle_condition_l573_573756


namespace total_cost_l573_573723

variable (E P M : ℝ)

axiom condition1 : E + 3 * P + 2 * M = 240
axiom condition2 : 2 * E + 5 * P + 4 * M = 440

theorem total_cost : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_l573_573723


namespace sum_of_first_ten_terms_l573_573615

theorem sum_of_first_ten_terms (a1 d : ℝ) (h1 : 3 * (a1 + d) = 15) 
  (h2 : (a1 + d - 1) ^ 2 = (a1 - 1) * (a1 + 2 * d + 1)) : 
  (10 / 2) * (2 * a1 + (10 - 1) * d) = 120 := 
by 
  sorry

end sum_of_first_ten_terms_l573_573615


namespace equilateral_triangle_area_unique_l573_573226

theorem equilateral_triangle_area_unique (x y : ℝ) (A B C : ℝ × ℝ)
  (h_curve : ∀ (x y : ℝ), x^3 + 3 * x * y + y^3 = 1)
  (h_equilateral : equilateral A B C)
  (h_on_curve_A : curve A)
  (h_on_curve_B : curve B)
  (h_on_curve_C : curve C)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  area A B C = (3 * real.sqrt 3) / 2 := 
sorry

end equilateral_triangle_area_unique_l573_573226


namespace integral_with_coefficient_l573_573121

theorem integral_with_coefficient (a : ℝ) 
  (h : ∃ (k : ℝ), (k = 30 ∧ (x^2 - a)*(x + 1/x)^10).coeff 6 = k) : 
  (a = 2) → ∫ x in 0..a, (3 * x^2 + 1) = 10 :=
by
  sorry

end integral_with_coefficient_l573_573121


namespace logarithmic_sum_l573_573867

noncomputable def geometric_sequence_log_sum (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) -- All terms are positive
  ∧ (a 10 * a 11 + a 9 * a 12 = 2 * real.exp 2) -- Given condition

theorem logarithmic_sum (a : ℕ → ℝ) (h : geometric_sequence_log_sum a) :
  ∑ i in finset.range 20, real.log (a (i + 1)) = 20 :=
sorry

end logarithmic_sum_l573_573867


namespace Jimin_scabs_l573_573943

theorem Jimin_scabs (total_scabs : ℕ) (days_in_week : ℕ) (daily_scabs: ℕ)
  (h₁ : total_scabs = 220) (h₂ : days_in_week = 7) 
  (h₃ : daily_scabs = (total_scabs + days_in_week - 1) / days_in_week) : 
  daily_scabs ≥ 32 := by
  sorry

end Jimin_scabs_l573_573943


namespace find_function_value_l573_573184

variable {α : ℝ} (f : ℝ → ℝ)

-- Assume functions as per conditions given
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def period_condition (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f (4 - x)
def domain_condition (f : ℝ → ℝ) : Prop := ∀ x ∈ set.Icc (0:ℝ) 4, f x = x
def sin_alpha : ℝ := (sqrt 2) / 3

-- Main proof statement
theorem find_function_value (α : ℝ)
  (h1 : even_function f)
  (h2 : period_condition f)
  (h3 : domain_condition f)
  (h4 : sin α = sin_alpha) :
  f (2016 + (sin (α - 2 * π)) * (sin (π + α)) - 2 * (cos (-α)) ^ 2) = 5 / 9 := by
  sorry

end find_function_value_l573_573184


namespace train_speed_approx_l573_573344

noncomputable def train_speed_in_kmph (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_approx (distance time : ℝ) (h_distance : distance = 180) (h_time : time = 11.999040076793857) :
  train_speed_in_kmph distance time ≈ 54.00450093607801 :=
  sorry

end train_speed_approx_l573_573344


namespace count_positive_integers_satisfying_inequality_l573_573791

theorem count_positive_integers_satisfying_inequality :
  let S := {n : ℕ | n > 0 ∧ (n - 2) * (n - 4) * (n - 6) * ... * (n - 98) < 0}
  ∃ n, S.card = 24 :=
by sorry

end count_positive_integers_satisfying_inequality_l573_573791


namespace angle_30_degrees_l573_573282

variables (a b : ℝ^3)
-- Placeholder for non-zero condition (e.g., a ≠ 0 and b ≠ 0)
axiom a_nonzero : ∥a∥ ≠ 0
axiom b_nonzero : ∥b∥ ≠ 0

-- Conditions from the problem
axiom magnitudes : ∥a∥ = ∥b∥ ∧ ∥a∥ = ∥a - b∥

-- Definition of the angle between a and a + b
def angle_between (u v : ℝ^3) : ℝ := 
  real.arccos ((u • v) / (∥u∥ * ∥v∥))

-- Statement of the theorem
theorem angle_30_degrees : angle_between a (a + b) = real.pi / 6 :=
  by sorry

end angle_30_degrees_l573_573282


namespace grid_values_constant_l573_573371

open Int

theorem grid_values_constant (f : ℤ × ℤ → ℕ)
  (h : ∀ x y, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ c : ℕ, ∀ x y, f (x, y) = c :=
begin
  -- proof goes here
  sorry
end

end grid_values_constant_l573_573371


namespace product_of_possible_values_l573_573483

theorem product_of_possible_values (x : ℝ) (h : (x + 2) * (x - 3) = -10) : 
  let a := 1,
      b := -1,
      c := 4 in
  let roots_product := c / a in
  roots_product = 4 := 
by
  sorry

end product_of_possible_values_l573_573483


namespace add_fractions_l573_573641

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l573_573641


namespace speed_on_local_roads_l573_573321

theorem speed_on_local_roads (v : ℝ) (h1 : 60 + 120 = 180) (h2 : (60 + 120) / (60 / v + 120 / 60) = 36) : v = 20 :=
by
  sorry

end speed_on_local_roads_l573_573321


namespace orthic_triangle_of_excenters_l573_573748

open EuclideanGeometry

noncomputable theory

def construct_triangle (K O_A O_B : Point) : Triangle :=
sorry

theorem orthic_triangle_of_excenters 
  (K O_A O_B : Point) 
  (A B C : Point) 
  (hK : is_circumcenter K A B C) 
  (hO_A : is_excenter O_A A B C)
  (hO_B : is_excenter O_B B A C) :
  is_orthic_triangle (Triangle.mk O_A O_B (exc_center K O_A O_B)) (Triangle.mk A B C) :=
sorry

end orthic_triangle_of_excenters_l573_573748


namespace Tn_less_than_9_over_4_l573_573435

-- Define the sequence and conditions
def a_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * (finset.range (n + 1)).sum a + 1

def b_sequence (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → b n = real.log (a (n + 1)) / real.log 3

def T_sum (a b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = (finset.range n).sum (λ i, b (i + 1) / a (i + 1))

-- The theorem to prove
theorem Tn_less_than_9_over_4 (a b : ℕ → ℝ) (T : ℕ → ℝ) 
    (ha : a_sequence a) (hb : b_sequence a b) (hT : T_sum a b T) :
    ∀ n : ℕ, T n < 9 / 4 :=
sorry

end Tn_less_than_9_over_4_l573_573435


namespace bug_travel_distance_l573_573835

-- Define the Euclidean distance function
def euclidean_distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the vertices A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (x, y)

-- Statement of the proof problem in Lean 4
theorem bug_travel_distance (x y : ℝ) :
  let AB := euclidean_distance A B in
  AB ≥ 0 ∧ (∃ dir : ℝ × ℝ, dir = (1, 0) ∨ dir = (1, real.sqrt 3 / 2) ∨ dir = (0, real.sqrt 3) ∧
  ∃ n : ℕ, n ≥ 1 ∧ n * real.sqrt dir.1^2 + dir.2^2 = AB / 2) →
  (n * AB / 2 ≥ (AB / 2 → n = 3)) :=
sorry

end bug_travel_distance_l573_573835


namespace greatest_common_divisor_three_divisors_l573_573288

theorem greatest_common_divisor_three_divisors (m : ℕ) (h : ∃ (D : set ℕ), D = {d | d ∣ 120 ∧ d ∣ m} ∧ D.card = 3) : 
  ∃ p : ℕ, p.prime ∧ greatest_dvd_set {d | d ∣ 120 ∧ d ∣ m} = p^2 := 
sorry

end greatest_common_divisor_three_divisors_l573_573288


namespace circle_center_and_radius_l573_573243

theorem circle_center_and_radius (x y: ℝ) :
  x^2 + y^2 - 4 * x = 0 → (∃ r: ℝ, (x - 2)^2 + y^2 = r^2 ∧ r = 2 ∧ 2 = 2) :=
by 
  intro h,
  have : ∃ r: ℝ, (x - 2)^2 + y^2 = r^2,
  from sorry,
  use 2,
  exact ⟨sorry, sorry, sorry⟩

end circle_center_and_radius_l573_573243


namespace find_length_CD_l573_573141

noncomputable def find_CD
  (BO OD AO OC AB : ℝ)
  (hBO : BO = 5)
  (hOD : OD = 7)
  (hAO : AO = 9)
  (hOC : OC = 4)
  (hAB : AB = 7) : ℝ :=
  let cosAOB := (AB^2 - AO^2 - BO^2) / (-2 * AO * BO) in
  let cosCOD := -cosAOB in
  let CD2 := OC^2 + OD^2 - 2 * OC * OD * cosCOD in
  Real.sqrt CD2

theorem find_length_CD : find_CD 5 7 9 4 7 = 10 :=
by
  rw [find_CD, ← hBO, ← hOD, ← hAO, ← hOC, ← hAB]
  simp
  sorry

end find_length_CD_l573_573141


namespace simplify_fraction_l573_573998

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l573_573998


namespace total_flour_needed_l573_573168

noncomputable def katie_flour : ℝ := 3

noncomputable def sheila_flour : ℝ := katie_flour + 2

noncomputable def john_flour : ℝ := 1.5 * sheila_flour

theorem total_flour_needed :
  katie_flour + sheila_flour + john_flour = 15.5 :=
by
  sorry

end total_flour_needed_l573_573168


namespace tomas_first_month_distance_l573_573277

theorem tomas_first_month_distance 
  (distance_n_5 : ℝ := 26.3)
  (double_distance_each_month : ∀ (n : ℕ), n ≥ 1 → (distance_n : ℝ) = distance_n_5 / (2 ^ (5 - n)))
  : distance_n_5 / (2 ^ (5 - 1)) = 1.64375 :=
by
  sorry

end tomas_first_month_distance_l573_573277


namespace quadratic_function_expression_l573_573466

variable (a x : ℝ)

def quadratic_function_with_vertex (a : ℝ) : ℝ → ℝ
| x => a * (x + 1)^2 + 2

theorem quadratic_function_expression :
  ∃ a,  (quadratic_function_with_vertex a 1 = -3) ∧ for (y : ℝ) (x : ℝ), y = quadratic_function_with_vertex a x ↔ y = -5/4 * x^2 - 5/2 * x + 3/4 := 
  sorry

end quadratic_function_expression_l573_573466


namespace total_sheep_l573_573349

-- Define the conditions as hypotheses
variables (Aaron_sheep Beth_sheep : ℕ)
def condition1 := Aaron_sheep = 7 * Beth_sheep
def condition2 := Aaron_sheep = 532
def condition3 := Beth_sheep = 76

-- Assert that under these conditions, the total number of sheep is 608.
theorem total_sheep
  (h1 : condition1 Aaron_sheep Beth_sheep)
  (h2 : condition2 Aaron_sheep)
  (h3 : condition3 Beth_sheep) :
  Aaron_sheep + Beth_sheep = 608 :=
by sorry

end total_sheep_l573_573349


namespace part1_part2_l573_573178

variable (a b c : ℝ)

-- Conditions
axiom h1 : a + b + c = 0
axiom h2 : a * b * c = 1

-- Part (1)
theorem part1 : a * b + b * c + c * a < 0 := by
  sorry

-- Part (2)
theorem part2 : max a (max b c) ≥ real.cbrt 4 := by
  sorry

end part1_part2_l573_573178


namespace inscribed_square_side_length_l573_573222

theorem inscribed_square_side_length :
  let ABC : Type := euclidean_geometry.triangle (-1,0) (0,12) (0,13) in
  let AB : ℝ := 5
  let BC : ℝ := 12
  let AC : ℝ := 13
  let areaABC := 0.5 * AB * BC
  let h := 2 * areaABC / AC
  let s := 13 * h / (13 + h)
  in s = 780 / 229 := by
  sorry

end inscribed_square_side_length_l573_573222


namespace counting_positive_integers_satisfying_inequality_l573_573779

theorem counting_positive_integers_satisfying_inequality :
  ∃ (n : ℕ), (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 49 → (n - 2 * k) < 0) ∧ n = 47 :=
begin
  sorry
end

end counting_positive_integers_satisfying_inequality_l573_573779


namespace max_elements_divisibility_condition_l573_573549

theorem max_elements_divisibility_condition (M : set ℕ) (hM : M ⊆ (set.Icc 1 2007)) 
  (h_div: ∀ a b c ∈ M, ∃ x y ∈ {a, b, c}, (x ∣ y ∨ y ∣ x) ∧ x ≠ y): 
  M.card ≤ 21 := 
sorry

end max_elements_divisibility_condition_l573_573549


namespace olya_candies_l573_573564

theorem olya_candies (P M T O : ℕ) (h1 : P + M + T + O = 88) (h2 : 1 ≤ P) (h3 : 1 ≤ M) (h4 : 1 ≤ T) (h5 : 1 ≤ O) (h6 : M + T = 57) (h7 : P > M) (h8 : P > T) (h9 : P > O) : O = 1 :=
by
  sorry

end olya_candies_l573_573564


namespace sum_of_squares_l573_573736

theorem sum_of_squares (n m : ℕ) : 
  let x := (17^2 + 19^2) / 2
  in x^2 = n^2 + m^2 :=
by
sorry

end sum_of_squares_l573_573736


namespace meeting_point_correct_l573_573559

-- Define the initial coordinates for Mark and Sandy
def Mark : (ℕ × ℕ) := (0, 7)
def Sandy : (ℕ × ℕ) := (2, -3)

-- Define the midpoint calculation as a function
def midpoint (p1 p2 : (ℕ × ℕ)) : (ℕ × ℕ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the adjustment function to move 1 unit north
def move_north (p : (ℕ × ℕ)) : (ℕ × ℕ) :=
  (p.1, p.2 + 1)

-- Define the final position after adjusting from the midpoint
def final_meeting_point : (ℕ × ℕ) :=
  move_north (midpoint Mark Sandy)

-- The theorem we aim to prove
theorem meeting_point_correct : final_meeting_point = (1, 3) :=
by
  sorry

end meeting_point_correct_l573_573559


namespace excluded_twins_lineup_l573_573319

/-- 
  Prove that the number of ways to choose 5 starters from 15 players,
  such that both Alice and Bob (twins) are not included together in the lineup, is 2717.
-/
theorem excluded_twins_lineup (n : ℕ) (k : ℕ) (t : ℕ) (u : ℕ) (h_n : n = 15) (h_k : k = 5) (h_t : t = 2) (h_u : u = 3) :
  ((n.choose k) - ((n - t).choose u)) = 2717 :=
by {
  sorry
}

end excluded_twins_lineup_l573_573319


namespace sampling_is_systematic_l573_573701

structure auditorium :=
  (rows : ℕ)
  (seats_per_row : ℕ)

def sample_method (a : auditorium) : option string :=
  if a.rows = 30 ∧ a.seats_per_row = 20 then some "systematic sampling" else none

theorem sampling_is_systematic :
  sample_method ⟨30, 20⟩ = some "systematic sampling" :=
by {
  -- Proof of systematic sampling follows directly from the definition and conditions
  sorry
}

end sampling_is_systematic_l573_573701


namespace curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l573_573146

noncomputable def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = 3 * Real.sqrt 2

theorem curve_C1_general_equation (x y : ℝ) (α : ℝ) :
  (2 * Real.cos α = x) ∧ (Real.sqrt 2 * Real.sin α = y) →
  x^2 / 4 + y^2 / 2 = 1 :=
sorry

theorem curve_C2_cartesian_equation (ρ θ : ℝ) (x y : ℝ) :
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ polar_curve_C2 ρ θ →
  x + y = 6 :=
sorry

theorem minimum_distance_P1P2 (P1 P2 : ℝ × ℝ) (d : ℝ) :
  (∃ α, P1 = parametric_curve_C1 α) ∧ (∃ x y, P2 = (x, y) ∧ x + y = 6) →
  d = (3 * Real.sqrt 2 - Real.sqrt 3) :=
sorry

end curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l573_573146


namespace roger_steps_time_l573_573572

theorem roger_steps_time (steps_per_30_min : ℕ := 2000) (time_for_2000_steps : ℕ := 30) (goal_steps : ℕ := 10000) : 
  (goal_steps * time_for_2000_steps) / steps_per_30_min = 150 :=
by 
  -- This is the statement. Proof is omitted as per instruction.
  sorry

end roger_steps_time_l573_573572


namespace product_of_real_parts_of_roots_l573_573181

def roots_real_product (a b c : ℂ) : ℂ :=
  let delta := b^2 - 4 * a * c
  let r1 := (-b + Complex.sqrt delta) / (2 * a)
  let r2 := (-b - Complex.sqrt delta) / (2 * a)
  (r1.re * r2.re)

theorem product_of_real_parts_of_roots : 
  roots_real_product 1 2 (-(10 - 8 * Complex.I)) = -15 := 
  by
  sorry

end product_of_real_parts_of_roots_l573_573181


namespace asymptotes_of_hyperbola_l573_573078

theorem asymptotes_of_hyperbola 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (c : ℝ) (F1 F2 P : ℝ × ℝ)
  (F1_def : F1 = (-c, 0)) (F2_def : F2 = (c, 0)) (P_def : P = (0, 2 * b))
  (right_triangle_cond : ∃ right_angle, (right_angle = |F1 - (0, 2 * b)|) ∧ (right_angle = 2 * c))
  (b_eq_sqrt : b = sqrt (c^2 - a^2)) :
  -- The statement we want to prove:
  (∀ x y : ℝ, (y = sqrt 3 * x) ∨ (y = - sqrt 3 * x)) :=
sorry

end asymptotes_of_hyperbola_l573_573078


namespace base5_odd_digit_count_l573_573409

theorem base5_odd_digit_count (n : ℕ) (h : n = 365) : 
  let b5 := (2 * 5^3 + 4 * 5^2 + 3 * 5^1 + 0 * 5^0) in (λ d, d % 2 = 1) '' finset.coe (finset.filter (λ d, true) (finset.range 5)).count = 1 :=
by
  let b5 := (2 * 5^3 + 4 * 5^2 + 3 * 5^1 + 0 * 5^0)
  have h1 : b5 = 2430 := by norm_num
  sorry

end base5_odd_digit_count_l573_573409


namespace height_is_seven_given_surface_area_condition_l573_573200

noncomputable def height_of_box (x : ℝ) : ℝ := x + 5

theorem height_is_seven_given_surface_area_condition 
  (x : ℝ) (h : 6 * x^2 + 20 * x ≥ 120) (h_pos : x ≥ 2) : 
  height_of_box x = 7 :=
by {
  have h_eq : x = 2 := sorry,
  simp [height_of_box, h_eq],
}

end height_is_seven_given_surface_area_condition_l573_573200


namespace ratio_of_arithmetic_sequence_sums_l573_573373

theorem ratio_of_arithmetic_sequence_sums :
  let a1 := 2
  let d1 := 3
  let l1 := 41
  let n1 := (l1 - a1) / d1 + 1
  let sum1 := n1 / 2 * (a1 + l1)

  let a2 := 4
  let d2 := 4
  let l2 := 60
  let n2 := (l2 - a2) / d2 + 1
  let sum2 := n2 / 2 * (a2 + l2)
  sum1 / sum2 = 301 / 480 :=
by
  sorry

end ratio_of_arithmetic_sequence_sums_l573_573373


namespace smallest_solution_of_equation_l573_573030

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (3 * x / (x - 2) + (3 * x^2 - 36) / x = 13) ∧ (x = (2 - real.sqrt 58) / 3) :=
by {
  sorry
}

end smallest_solution_of_equation_l573_573030


namespace find_lambda_l573_573144

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {O A B C M : V}
variables (λ : ℝ)

-- Conditions
def condition1 : Prop :=
  M = (1/2)•A + (1/6)•B + λ•C

def vectors_coplanar : Prop :=
  ∃ (m n : ℝ), 
    (A - M) = m • (B - M) + n • (C - M)

-- Correct answer
theorem find_lambda (h1 : condition1) (h2 : vectors_coplanar) : λ = 1/3 := sorry

end find_lambda_l573_573144


namespace smallest_nineteen_people_l573_573679

theorem smallest_nineteen_people (N : ℕ) (chairs : ℕ) (circular : chairs = 75) : 
  (∀ M < 19, ¬ seating_possible M) ∧ seating_possible 19 :=
by
  sorry

noncomputable def seating_possible (M : ℕ) : Prop := 
  ∀ (chair_number : ℕ), chair_number < 75 → occupied (chair_number + 1) 75

/-- The hypothesis occupied needs to be defined correctly
    This definition ensures whether a person is seated on a given chair.
  -/
def occupied (chair_number placement : ℕ) : Prop :=
  sorry

end smallest_nineteen_people_l573_573679


namespace equidistant_lines_l573_573776

structure Point := (x : ℝ) (y : ℝ)

def is_equidistant (P A B : Point) (line : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line x y → (abs ((B.y - A.y) * x + (A.x - B.x) * y + (B.x * A.y - A.x * B.y)) / real.sqrt ((B.y - A.y) ^ 2 + (A.x - B.x) ^ 2)) = 
         (abs ((B.y - A.y) * (P.x) + (A.x - B.x) * (P.y) + (B.x * A.y - A.x * B.y)) / real.sqrt ((B.y - A.y) ^ 2 + (A.x - B.x) ^ 2))

def line1 (x y : ℝ) : Prop := y = 4 * x - 2
def line2 (x y : ℝ) : Prop := x = 1

theorem equidistant_lines :
  ∃ (line : ℝ → ℝ → Prop), 
    (line = line1 ∨ line = line2) ∧
    is_equidistant (Point.mk 1 2) (Point.mk 2 3) (Point.mk 0 (-5)) line := 
sorry

end equidistant_lines_l573_573776


namespace polynomial_with_rational_values_is_rational_l573_573011

-- Definition of a polynomial with complex coefficients
variable {R : Type*} [CommRing R] [IsDomain R] [CharZero R]

noncomputable def is_rational_valued_polynomial (P : R[X]) : Prop :=
  ∀ (q : ℚ), (P.eval (algebraMap ℚ R q) ∈ ℚ)

-- Statement of the proof problem
theorem polynomial_with_rational_values_is_rational (P : ℂ[X])
  (h : is_rational_valued_polynomial P) : ∀ i, (P.coeff i ∈ ℚ) :=
sorry

end polynomial_with_rational_values_is_rational_l573_573011


namespace find_r_cubed_and_reciprocal_cubed_l573_573115

variable (r : ℝ)
variable (h : (r + 1 / r) ^ 2 = 5)

theorem find_r_cubed_and_reciprocal_cubed (r : ℝ) (h : (r + 1 / r) ^ 2 = 5) : r ^ 3 + 1 / r ^ 3 = 2 * Real.sqrt 5 := by
  sorry

end find_r_cubed_and_reciprocal_cubed_l573_573115


namespace find_a_over_b_l573_573036

noncomputable def a := sorry
noncomputable def b := sorry

axiom hyperbola_condition : (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → a > b)
axiom asymptotes_angle_condition : (∀ θ : ℝ, θ = 45 → tan θ = 1)

theorem find_a_over_b (a b : ℝ) (h₁ : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → a > b)
                          (h₂ : ∀ θ : ℝ, θ = 45 → tan θ = 1) :
    a / b = 1 + Real.sqrt 2 := 
begin
  sorry
end

end find_a_over_b_l573_573036


namespace simplify_fraction_48_72_l573_573995

theorem simplify_fraction_48_72 : (48 : ℚ) / 72 = 2 / 3 := sorry

end simplify_fraction_48_72_l573_573995


namespace sticks_picked_up_l573_573303

variable (original_sticks left_sticks picked_sticks : ℕ)

theorem sticks_picked_up :
  original_sticks = 99 → left_sticks = 61 → picked_sticks = original_sticks - left_sticks → picked_sticks = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sticks_picked_up_l573_573303


namespace choose_1010_numbers_l573_573620

theorem choose_1010_numbers :
  let numbers : List ℕ := List.range (2021 + 1) -- numbers from 0 to 2021
  let valid_selection (s : Finset ℕ) : Prop :=
    s.card = 1010 ∧ (∀ x y ∈ s, x ≠ y → x + y ≠ 2021 ∧ x + y ≠ 2022)
  in
  (Finset (Finset ℕ)).card {s ∈ Finset.powerset (Finset.ofList numbers) | valid_selection s} = 511566 :=
by
  sorry

end choose_1010_numbers_l573_573620


namespace avg_retail_price_l573_573739

theorem avg_retail_price (n : ℕ) (products : Fin n → ℕ) 
  (h_len : n = 20)
  (h_min : ∀ i, products i ≥ 400)
  (h_ten : (Finset.filter (λ i, products i < 1000) Finset.univ).card = 10)
  (h_max : ∃ i, products i = 11000) :
  (Finset.univ.sum products) / n = 930 := 
sorry

end avg_retail_price_l573_573739


namespace rebecca_eggs_l573_573221

theorem rebecca_eggs (groups : ℕ) (eggs_per_group : ℕ) (total_eggs : ℕ) 
  (h1 : groups = 3) (h2 : eggs_per_group = 3) : total_eggs = 9 :=
by
  sorry

end rebecca_eggs_l573_573221


namespace time_for_trains_to_cross_each_other_l573_573280

-- Definitions based on the conditions
def length_of_train : ℕ := 120
def time_to_cross_post_train1 : ℕ := 10
def time_to_cross_post_train2 : ℕ := 12

-- Speeds calculated from the conditions
def speed_train1 : ℝ := length_of_train / time_to_cross_post_train1
def speed_train2 : ℝ := length_of_train / time_to_cross_post_train2

-- Relative speed when trains move in opposite directions
def relative_speed : ℝ := speed_train1 + speed_train2

-- Total distance when trains cross each other
def total_distance : ℝ := 2 * length_of_train

-- Time to cross
def time_to_cross : ℝ := total_distance / relative_speed

-- Statement to prove
theorem time_for_trains_to_cross_each_other : time_to_cross ≈ 10.91 := by
  sorry

end time_for_trains_to_cross_each_other_l573_573280


namespace part1_solution_set_part2_f_le_g_l573_573100

open Real

noncomputable def f (x b : ℝ) : ℝ := abs (x + b^2) - abs (-x + 1)
noncomputable def g (x a b c : ℝ) : ℝ := abs (x + a^2 + c^2) + abs (x - 2 * b^2)

theorem part1_solution_set (x : ℝ) : 
  ∀ b : ℝ, b = 1 → 
  (∃ sol_set : Set ℝ, sol_set = { x | 1 / 2 ≤ x } ∧ ∀ x ∈ sol_set, f x b ≥ 1) := 
begin
  intros b hb,
  rw hb,
  sorry
end

theorem part2_f_le_g (x a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ a * b + b * c + a * c = 1 → 
  f x b ≤ g x a b c :=
begin
  intros h,
  sorry
end

end part1_solution_set_part2_f_le_g_l573_573100


namespace find_k_l573_573362

theorem find_k (angle_BAC : ℝ) (angle_D : ℝ)
  (h1 : 0 < angle_BAC ∧ angle_BAC < π)
  (h2 : 0 < angle_D ∧ angle_D < π)
  (h3 : (π - angle_BAC) / 2 = 3 * angle_D) :
  angle_BAC = (5 / 11) * π :=
by sorry

end find_k_l573_573362


namespace sequence_properties_l573_573875

variable {Seq : Nat → ℕ}
-- Given conditions: Sn = an(an + 3) / 6
def Sn (n : ℕ) := Seq n * (Seq n + 3) / 6

theorem sequence_properties :
  (Seq 1 = 3) ∧ (Seq 2 = 9) ∧ (∀ n : ℕ, Seq (n+1) = 3 * (n + 1)) :=
by 
  have h1 : Sn 1 = (Seq 1 * (Seq 1 + 3)) / 6 := rfl
  have h2 : Sn 2 = (Seq 2 * (Seq 2 + 3)) / 6 := rfl
  sorry

end sequence_properties_l573_573875


namespace book_distribution_l573_573047

theorem book_distribution :
  ∃ (ways : ℕ), ways = 34 ∧
  ∀ (books friends : ℕ), books = 7 → friends = 4 →
    (∀ f : fin friends, f.val ≥ 1) → 
    (number_of_ways books friends = ways) :=
by
  sorry

end book_distribution_l573_573047


namespace angle_OMX_eq_90_degrees_l573_573678

open Point

structure Circle (P : Type) :=
  (center : P)
  (radius : Real)

structure Triangle (P : Type) :=
  (A B C : P)

variables {P : Type}
  [MetricSpace P]
  [NormedAddCommGroup P]
  [NormedSpace ℝ P]

def is_tangent (line : AffineSubspace ℝ P) (circle : Circle P) (point : P) : Prop :=
  circle.center ∈ line ∧ dist circle.center point = circle.radius

def is_circumscribed_by (circle : Circle P) (triangle : Triangle P) : Prop :=
  dist circle.center triangle.A = circle.radius ∧
  dist circle.center triangle.B = circle.radius ∧
  dist circle.center triangle.C = circle.radius

noncomputable def circle_through_points (center : P) (A B : P) : Circle P :=
  ⟨center, dist center A⟩ -- assuming center equals to the circumcenter of A, B, and its radius being the circle passing through A, B

theorem angle_OMX_eq_90_degrees
  {O A B C X Y M : P}
  (ω : Circle P)
  (ω1 : Circle P)
  (ω2 : Circle P)
  (h_circum : is_circumscribed_by ω ⟨A, B, C⟩)
  (h_tangent_ω1 : is_tangent (AffineSubspace.span ℝ {A, B}) ω1 A)
  (h_tangent_ω2 : is_tangent (AffineSubspace.span ℝ {A, C}) ω2 A)
  (h_XY_distinct : X ≠ A ∧ Y ≠ A)
  (h_X_on_ω1 : X ∈ ω1)
  (h_Y_on_ω2 : Y ∈ ω2)
  (h_M_mid : dist M X = dist M Y) :
  ∠OMX = π / 2 := sorry

end angle_OMX_eq_90_degrees_l573_573678


namespace windows_per_floor_is_3_l573_573201

-- Given conditions
variables (W : ℕ)
def windows_each_floor (W : ℕ) : Prop :=
  (3 * 2 * W) - 2 = 16

-- Correct answer
theorem windows_per_floor_is_3 : windows_each_floor 3 :=
by 
  sorry

end windows_per_floor_is_3_l573_573201


namespace max_volume_48cm_square_l573_573712

def volume_of_box (x : ℝ) := x * (48 - 2 * x)^2

theorem max_volume_48cm_square : 
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → volume_of_box x ≥ volume_of_box y) ∧ x = 8 :=
sorry

end max_volume_48cm_square_l573_573712


namespace circumcenter_of_IHO_on_Omega_l573_573173

variables {A B C H I O M : Type}
variables [Triangle A B C] [Orthocenter H A B C] [Incenter I A B C] [Circumcenter O A B C]
variables [Scalene A B C] [Acute A B C] [AngleBAC60 : Angle A B C = 60]
variables [MidpointArc M B C]

theorem circumcenter_of_IHO_on_Omega :
  Concyclic {I, H, O, M} :=
begin
  sorry
end

end circumcenter_of_IHO_on_Omega_l573_573173


namespace coats_collected_elem_schools_correct_l573_573236

-- Conditions
def total_coats_collected : ℕ := 9437
def coats_collected_high_schools : ℕ := 6922

-- Definition to find coats collected from elementary schools
def coats_collected_elementary_schools : ℕ := total_coats_collected - coats_collected_high_schools

-- Theorem statement
theorem coats_collected_elem_schools_correct : 
  coats_collected_elementary_schools = 2515 := sorry

end coats_collected_elem_schools_correct_l573_573236


namespace map_distance_representation_l573_573211

theorem map_distance_representation
    (L : ℝ)
    (scale_factor : ℝ)
    (unit_length : ℝ)
    (map_distance : ℝ)
    (map_length : ℝ)
    (physical_distance : ℝ)
    (map_length_eq_nine : map_length = 9)
    (physical_distance_eq_fifty_four : physical_distance = 54)
    (scale_factor_def : scale_factor = physical_distance / map_length)
    (unit_length_eq_one : unit_length = 1)
    (map_distance_eq_twenty : map_distance = 20) :
    L = map_distance * scale_factor :=
begin
  sorry
end

end map_distance_representation_l573_573211


namespace find_lambda_l573_573145

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {O A B C M : V}
variables (λ : ℝ)

-- Conditions
def condition1 : Prop :=
  M = (1/2)•A + (1/6)•B + λ•C

def vectors_coplanar : Prop :=
  ∃ (m n : ℝ), 
    (A - M) = m • (B - M) + n • (C - M)

-- Correct answer
theorem find_lambda (h1 : condition1) (h2 : vectors_coplanar) : λ = 1/3 := sorry

end find_lambda_l573_573145


namespace distinct_real_roots_range_root_sum_reciprocal_condition_l573_573433

-- Define the quadratic equation conditions
def quadratic_eq (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Part 1: Proving the range for m
theorem distinct_real_roots_range (m : ℝ) :
  let Δ := (-2 * (m - 1))^2 - 4 * 1 * m^2 in Δ > 0 ↔ m < 1 / 2 :=
by
  sorry

-- Part 2: Given roots conditions and solving for m
theorem root_sum_reciprocal_condition (m : ℝ) (x1 x2 : ℝ) (h1 : quadratic_eq 1 (-2 * (m - 1)) m^2 x1)
  (h2 : quadratic_eq 1 (-2 * (m - 1)) m^2 x2) (h3 : x1 ≠ x2) :
  (1 / x1 + 1 / x2 = -2) ↔ m = (-1 - Real.sqrt 5) / 2 :=
by
  sorry

end distinct_real_roots_range_root_sum_reciprocal_condition_l573_573433


namespace number_of_solutions_l573_573420

theorem number_of_solutions : ∃ (s : Finset ℕ), (∀ x ∈ s, 100 ≤ x^2 ∧ x^2 ≤ 200) ∧ s.card = 5 :=
by
  sorry

end number_of_solutions_l573_573420


namespace ripe_oranges_harvest_l573_573109

theorem ripe_oranges_harvest (daily_ripe_oranges : ℕ) (days_of_harvest : ℕ) : 
  daily_ripe_oranges = 5 → days_of_harvest = 73 → daily_ripe_oranges * days_of_harvest = 365 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end ripe_oranges_harvest_l573_573109


namespace buckets_left_l573_573329

theorem buckets_left (initial : ℝ) (sowed : ℝ) (remaining : ℝ) 
  (h_initial : initial = 8.75) 
  (h_sowed : sowed = 2.75) 
  (h_remaining : remaining = initial - sowed) : 
  remaining = 6 := 
by 
  rw [h_initial, h_sowed, sub_eq_add_neg] 
  norm_num

end buckets_left_l573_573329


namespace curve_is_non_square_rhombus_l573_573148

-- The definition of the curve equation
def curve (x y a b : ℝ) : Prop :=
  (abs (x + y) / (2 * a)) + (abs (x - y) / (2 * b)) = 1

-- The geometric interpretation of the curve
def isRhombus (a b : ℝ) : Prop :=
  a ≠ b ∧ a > 0 ∧ b > 0 ∧ 
  ∀ {x y : ℝ}, curve x y a b → ∃ k l : ℝ, k ≠ l ∧ 
    (∃ A B C D : ℝ × ℝ, 
      A = (sqrt (2 : ℝ) * a, 0) ∧ B = (0, sqrt (2 : ℝ) * b) ∧
      C = (-sqrt (2 : ℝ) * a, 0) ∧ D = (0, -sqrt (2 : ℝ) * b) ∧
      dist A B = dist B C ∧ dist C D = dist D A)

-- The main theorem stating that the curve is a non-square rhombus
theorem curve_is_non_square_rhombus (a b : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) : 
  isRhombus a b :=
sorry

end curve_is_non_square_rhombus_l573_573148


namespace find_number_l573_573300

theorem find_number (x : ℝ) : 14 * x + 15 * x + 18 * x + 11 = 152 → x = 3 := by
  sorry

end find_number_l573_573300


namespace problem_statement_l573_573534

open Set

-- Definition of the set S'
def S' : Set (ℤ × ℤ × ℤ) := 
  { p | (0 ≤ p.1 ∧ p.1 ≤ 3) ∧ (0 ≤ p.2.1 ∧ p.2.1 ≤ 4) ∧ (0 ≤ p.2.2 ∧ p.2.2 ≤ 5) }

-- The Lean statement for the problem condition and required proof
theorem problem_statement :
  let valid_midpoint (p1 p2 : ℤ × ℤ × ℤ) := 
    ((p1.1 + p2.1) / 2, (p1.2.1 + p2.2.1) / 2, (p1.2.2 + p2.2.2) / 2) ∈ S' ∧ 
    (∃ p, p ∈ S' ∧ p ≠ p1 ∧ p ≠ p2 ∧ (even p1.1 ∨ even p1.2.1 ∨ even p1.2.2) ∧ (even p2.1 ∨ even p2.2.1 ∨ even p2.2.2)) in
  let m := 13 in
  let n := 25 in
  valid_midpoint ⟶ (m + n = 38) := sorry

end problem_statement_l573_573534


namespace simplify_fraction_l573_573991

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l573_573991


namespace length_AM_l573_573552

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C M : V)
variables (AB AC : V)

-- Define the given conditions
def midpoint (M B C : V) : Prop := (M = 0.5 • (B + C))
def outside_line (A B C : V) : Prop := (¬ collinear ℝ ({A, B, C} : set V))

-- Main theorem
theorem length_AM 
  (h_mid : midpoint M B C) 
  (h_out : outside_line A B C) 
  (h_BC : ∥B - C∥ = 4) 
  (h_equal_lengths : ∥(A - B) + (A - C)∥ = ∥(A - B) - (A - C)∥)
  : ∥A - M∥ = 2 :=
sorry

end length_AM_l573_573552


namespace group_of_ten_exists_l573_573918

noncomputable def exists_group_of_ten_with_connection (people : Type) [Fintype people] [DecidableEq people] (knows : people → people → Prop) : Prop :=
  (∀ S : Finset people, S.card = 11 → ∃ x y ∈ S, x ≠ y ∧ knows x y) →
  ∃ S : Finset people, S.card = 10 ∧ ∀ p ∉ S, ∃ q ∈ S, knows q p

theorem group_of_ten_exists {people : Type} [Fintype people] [DecidableEq people] (knows : people → people → Prop)
  (h : ∀ S : Finset people, S.card = 11 → ∃ x y ∈ S, x ≠ y ∧ knows x y) :
  ∃ S : Finset people, S.card = 10 ∧ ∀ p ∉ S, ∃ q ∈ S, knows q p :=
begin
  sorry
end

end group_of_ten_exists_l573_573918


namespace sum_series_eq_three_l573_573401

theorem sum_series_eq_three : 
  ∑' (k : ℕ), (k^2 : ℝ) / (2^k : ℝ) = 3 := sorry

end sum_series_eq_three_l573_573401


namespace angle_measure_l573_573759

theorem angle_measure (y : ℝ) (hyp : 45 + 3 * y + y = 180) : y = 33.75 :=
by
  sorry

end angle_measure_l573_573759


namespace counting_positive_integers_satisfying_inequality_l573_573780

theorem counting_positive_integers_satisfying_inequality :
  ∃ (n : ℕ), (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 49 → (n - 2 * k) < 0) ∧ n = 47 :=
begin
  sorry
end

end counting_positive_integers_satisfying_inequality_l573_573780


namespace constant_term_in_expansion_of_binomial_l573_573589

theorem constant_term_in_expansion_of_binomial :
  (∀ (x : ℝ), x ≠ 0 → (let t := (x + (1 / x) + 1) in t^4 = 19)) :=
begin
  intro x,
  intro hx,
  -- Mathematical equivalent conditions and necessary properties
  have h1 : (x + (1 / x) + 1)^4 = ∑ k in {0, 1, 2, 3, 4}, 
    (∑ j in {0, 1, 2, 3, 4}, ite (4 - 2 * k = 0) ((nat.choose 4 k * nat.choose k j * 1) : ℝ) 0) :=
    sorry,

  -- Prove that the constant term calculated as in the solution equals 19
  have h2 : (1 + 6 * 2 + 1 = 19) :=
    by norm_num,

  -- Use h2 to finalize the theorem
  rw h1,
  exact h2
end

end constant_term_in_expansion_of_binomial_l573_573589


namespace group_size_is_seven_l573_573603

theorem group_size_is_seven :
  ∃ (k N : ℕ), 10 < N ∧ N < 40 ∧ (N - 3) % 30 = 0 ∧ (N % k = 5) ∧ k = 7 :=
by
  -- Declare variables
  let N := 33
  let k := 7
  -- Conditions according to the problem
  have condition1 : 10 < N ∧ N < 40 := by
    simp [N],
    exact ⟨by linarith, by linarith⟩
  have condition2 : (N - 3) % 30 = 0 := by
    simp [N],
    exact mod_eq_zero_of_dvd (by norm_num)
  have condition3 : N % k = 5 := by
    simp [N, k],
    exact mod_eq_of_lt (by norm_num)
  -- Combine all conditions
  exact ⟨k, N, condition1.1, condition1.2, condition2, condition3, rfl⟩

end group_size_is_seven_l573_573603


namespace buffet_dishes_l573_573698

-- To facilitate the whole proof context, but skipping proof parts with 'sorry'

-- Oliver will eat if there is no mango in the dishes

variables (D : ℕ) -- Total number of dishes

-- Conditions:
variables (h1 : 3 <= D) -- there are at least 3 dishes with mango salsa
variables (h2 : 1 ≤ D / 6) -- one-sixth of dishes have fresh mango
variables (h3 : 1 ≤ D) -- there's at least one dish with mango jelly
variables (h4 : D / 6 ≥ 2) -- Oliver can pick out the mangoes from 2 of dishes with fresh mango
variables (h5 : D - (3 + (D / 6 - 2) + 1) = 28) -- there are 28 dishes Oliver can eat

theorem buffet_dishes : D = 36 :=
by
  sorry -- Skip the actual proof

end buffet_dishes_l573_573698


namespace num_positive_integers_satisfying_l573_573798

theorem num_positive_integers_satisfying (n : ℕ) :
  (∑ k in (finset.range 25), (if (even (2 + 4 * k)) then 1 else 0) = 24) :=
sorry

end num_positive_integers_satisfying_l573_573798


namespace bisect_rectangle_l573_573068

-- Definitions based on the problem's conditions
def point (x y : ℕ) : Type := { x := x, y := y }

def R : point := point.mk 0 0 
def S (a : ℕ) : point := point.mk a 0
def T (a : ℕ) : point := point.mk a 6
def U : point := point.mk 0 6

def line (b : ℚ) (x : ℚ) : ℚ := b * (x - 7) + 4

-- The main statement to be proved
theorem bisect_rectangle (a : ℕ) (b : ℚ) (h₁ : a = 20) : 
  (line b (a/2 : ℚ) = 3) -> b = -1/3 := by   -- area bisected means line passes through the midpoint (a/2, 3)
  sorry

end bisect_rectangle_l573_573068


namespace machines_produce_x_units_l573_573652

variable (x : ℕ) (d : ℕ)

-- Define the conditions
def four_machines_produce_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  4 * (x / d) = x / d

def twelve_machines_produce_three_x_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  12 * (x / d) = 3 * (x / d)

-- Given the conditions, prove the number of days for 4 machines to produce x units
theorem machines_produce_x_units (x : ℕ) (d : ℕ) 
  (H1 : four_machines_produce_in_d_days x d)
  (H2 : twelve_machines_produce_three_x_in_d_days x d) : 
  x / d = x / d := 
by 
  sorry

end machines_produce_x_units_l573_573652


namespace probability_two_red_crayons_l573_573685

def num_crayons : ℕ := 6
def num_red : ℕ := 3
def num_blue : ℕ := 2
def num_green : ℕ := 1
def num_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_two_red_crayons :
  let total_pairs := num_choose num_crayons 2
  let red_pairs := num_choose num_red 2
  (red_pairs : ℚ) / (total_pairs : ℚ) = 1 / 5 :=
by
  sorry

end probability_two_red_crayons_l573_573685


namespace find_number_of_valid_n_l573_573813

def valid_n (n : ℕ) : Prop :=
  (2 < n) ∧ (n < 100) ∧ ((∀ k : ℕ, k >= 0 → n = 4 * k + 3) ∨ (∃ k : ℕ, n = 4 * k + 3))

theorem find_number_of_valid_n : 
  {n : ℕ | valid_n n}.card = 24 :=
by
  sorry

end find_number_of_valid_n_l573_573813


namespace real_solutions_eq_31_l573_573027

noncomputable def number_of_real_solutions : ℕ :=
  let zero := 0
  let fifty := 50
  let neg_fifty := -50
  let num_intervals := 8
  let num_solutions_per_interval := 2
  let total_solutions := num_intervals * num_solutions_per_interval * 2 - 1
  total_solutions

theorem real_solutions_eq_31 : number_of_real_solutions = 31 := by
  sorry

end real_solutions_eq_31_l573_573027


namespace problem_statement_l573_573189

variable {R : Type*} [LinearOrderedField R]

theorem problem_statement
  (x1 x2 x3 y1 y2 y3 : R)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : y1 + y2 + y3 = 0)
  (h3 : x1 * y1 + x2 * y2 + x3 * y3 = 0)
  (h4 : (x1^2 + x2^2 + x3^2) * (y1^2 + y2^2 + y3^2) > 0) :
  (x1^2 / (x1^2 + x2^2 + x3^2) + y1^2 / (y1^2 + y2^2 + y3^2) = 2 / 3) := 
sorry

end problem_statement_l573_573189


namespace five_guys_meals_sets_l573_573707

theorem five_guys_meals_sets :
  ∃ (meal_sets : Finset (Finset ℕ)), -- Defining a finite set of finite sets of natural numbers
  meal_sets.card = 7 ∧ -- There are exactly 7 such sets
  ∀ (meals ∈ meal_sets), -- For every set in meal_sets
  meals.card = 5 ∧ -- It should contain exactly 5 elements
  (∀ (m ∈ meals), m ∈ {1, 2, ..., 20}.finset) ∧ -- Each element should be in the range [1, 20]
  (∑ m in meals, m) = 20 := -- The elements should sum to 20
sorry

end five_guys_meals_sets_l573_573707


namespace g_increasing_on_one_inf_x_one_local_min_g_g_at_most_two_zeros_l573_573460

variable {R : Type*} [Real R]
variable (f : R → R)
variable (g : R → R := λ x, f x / exp x)

-- Conditions
variable (h_diff : Differentiable R f)
variable (h_f_zero : f 0 = 1)
variable (h_derivative : ∀ x, (f' x - f x) / (x - 1) > 0)

-- Statements
theorem g_increasing_on_one_inf :
  MonotoneOn g (Ioi 1) := sorry

theorem x_one_local_min_g :
  IsLocalMin g 1 := sorry

theorem g_at_most_two_zeros :
  ∃ x1 x2, (x1 ≠ x2) ∧ 
            (g x1 = 0) ∧ 
            (g x2 = 0) ∧ 
            ∀ x, 
              (g x = 0 → (x = x1 ∨ x = x2)) := sorry

end g_increasing_on_one_inf_x_one_local_min_g_g_at_most_two_zeros_l573_573460


namespace sine_theorem_trihedral_l573_573190

theorem sine_theorem_trihedral 
  (α β γ A B C : ℝ) 
  (h1 : sin α ≠ 0)
  (h2 : sin β ≠ 0)
  (h3 : sin γ ≠ 0)
  (h4 : sin A ≠ 0)
  (h5 : sin B ≠ 0)
  (h6 : sin C ≠ 0) :
  (sin α / sin A = sin β / sin B) ∧ (sin β / sin B = sin γ / sin C) :=
by
  sorry

end sine_theorem_trihedral_l573_573190


namespace find_c_plus_d_l573_573191

def f (x : ℝ) (c d : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if x >= -1 then 2 * x - 4
  else 3 * x - d

theorem find_c_plus_d (c d : ℝ) :
  (∀ x : ℝ, continuous_at (f x c d) 1) → (∀ x : ℝ, continuous_at (f x c d) (-1)) → c + d = -7 :=
by
  sorry

end find_c_plus_d_l573_573191


namespace solve_for_x_l573_573117

noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (x + 2) / 5) ^ (1 / 4)

theorem solve_for_x : 
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -404 / 201 := 
by {
  sorry
}

end solve_for_x_l573_573117


namespace distance_to_karasuk_is_140_l573_573320

noncomputable def distance_to_karasuk (v_b v_c : ℝ) (t : ℝ) : ℝ :=
  let distance_bus := 70 + v_b * t
  let distance_car := v_c * t
  in distance_bus

theorem distance_to_karasuk_is_140 : 
  ∀ (v_b v_c t : ℝ), v_c = 2 * v_b → (v_b * (t - 70 / v_b) = 20) → (v_c * (t - 70 / v_c) = 40) → distance_to_karasuk v_b v_c t = 140 :=
by 
  intros v_b v_c t h_speed h_bus h_car
  sorry

end distance_to_karasuk_is_140_l573_573320


namespace kolya_can_determine_l573_573492

-- Define properties of coins
inductive Coin
| genuine
| counterfeit

open Coin

-- All four coins are in a set
def coins : Finset Coin := {genuine, genuine, counterfeit, counterfeit}

-- Define the balance action
def balanced (left right : Finset Coin) : Prop :=
  left.sum_weight = right.sum_weight

noncomputable def sum_weight (co : Coin) : ℕ :=
  match co with
  | genuine => 2
  | counterfeit => 1

theorem kolya_can_determine :
  ∃ (w1 w2 : Finset Coin),
  (w1 ∪ w2 = coins) ∧
  (balanced w1 w2 ∨ ¬balanced w1 w2) ∧ 
  (∀ (x y : Finset Coin), 
     (balanced x y → balanced (Finset.filter Coin.genuine x ∩ Finset.filter Coin.counterfeit y) (Finset.filter Coin.genuine y ∩ Finset.filter Coin.counterfeit x))) →
  (∃ (result : bool), result = true) :=
begin
    sorry
end

end kolya_can_determine_l573_573492


namespace kermit_unique_positions_l573_573169

theorem kermit_unique_positions : 
  ∃ (positions : Finset (ℤ × ℤ)), 
    positions.card = 10201 ∧ 
    ∀ (x y : ℤ),
      (x, y) ∈ positions ↔ abs x ≤ 50 ∧ abs y ≤ 50 :=
begin
  sorry
end

end kermit_unique_positions_l573_573169


namespace circumference_of_tangent_circle_l573_573281

noncomputable def r1 : ℝ := 37.5 / Real.pi
noncomputable def r2 : ℝ := 45 / Real.pi

theorem circumference_of_tangent_circle : ∃ (r3 : ℝ), 2 * Real.pi * r3 =
  2 * Real.pi * (
    let r_avg := (r1 + r2) / 2 in
    sqrt ((r1 - r3) * (r2 - r3))
  ) := sorry

end circumference_of_tangent_circle_l573_573281
