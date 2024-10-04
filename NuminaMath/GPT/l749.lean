import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Graph
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finsupp.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Pi
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Function
import Mathlib.Probability
import Mathlib.RingTheory.Base
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Omega
import Mathlib.Topology.Basic
import Mathlib.Topology.Continuous_Function.Basic

namespace angle_BMC_eq_one_third_angle_DMC_l749_749469

noncomputable section

open EuclideanGeometry

variables {O A B C M D : Point ℝ} -- Points involved
variable {r : ℝ} -- Radius

-- Given conditions
axiom h1 : CircleCenter O A B  -- O is the center of the circle with diameter AB
axiom h2 : Distance O A = r    -- OA is the radius and equal to r, so is OB hence AB = 2r
axiom h3 : OnLineExtends A B C  -- AB extended to C, so BC = r / 2
axiom h4 : TangentAt B S        -- S is the tangent at point B
axiom h5 : OnTangent M S        -- M is any point on tangent S
axiom h6 : TangentFrom M D      -- MD is a tangent from M to the circle at point D

-- To show
theorem angle_BMC_eq_one_third_angle_DMC :
  ∠B M C = (1 / 3) * ∠D M C :=
sorry

end angle_BMC_eq_one_third_angle_DMC_l749_749469


namespace find_lambda_l749_749047

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749047


namespace value_of_a_l749_749917

-- We state f(x) as given in the problem.
def f (x : ℝ) (a : ℝ) := a - 2 / (2^x + 1)

-- We state the problem: proving that if f is odd and defined as above, then a = 1.
theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) a = -f x a) → a = 1 :=
by
  -- To be proved.
  sorry

end value_of_a_l749_749917


namespace find_lambda_l749_749084

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749084


namespace jacob_fifth_test_score_l749_749493

theorem jacob_fifth_test_score (s1 s2 s3 s4 s5 : ℕ) :
  s1 = 85 ∧ s2 = 79 ∧ s3 = 92 ∧ s4 = 84 ∧ ((s1 + s2 + s3 + s4 + s5) / 5 = 85) →
  s5 = 85 :=
sorry

end jacob_fifth_test_score_l749_749493


namespace find_lambda_l749_749007

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749007


namespace smallest_integer_with_18_divisors_l749_749680

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749680


namespace number_of_lattice_points_l749_749844

theorem number_of_lattice_points :
  {p : ℤ × ℤ // p.1^2 + p.2^2 < 25 ∧ p.1^2 + p.2^2 < 10 * p.1 ∧ p.1^2 + p.2^2 < 10 * p.2}.card = 6 := by
  sorry

end number_of_lattice_points_l749_749844


namespace base_121_is_perfect_square_l749_749460

theorem base_121_is_perfect_square (b : ℕ) (hb : b > 2) : 
  let n := (1 * b^2 + 2 * b + 1) in 
  ∃ k : ℕ, n = k^2 := 
by
  sorry

end base_121_is_perfect_square_l749_749460


namespace algebraic_expression_value_l749_749887

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 6 - Real.sqrt 2) : 2 * x^2 + 4 * Real.sqrt 2 * x = 8 :=
sorry

end algebraic_expression_value_l749_749887


namespace sum_mod_15_l749_749857

theorem sum_mod_15 :
  let seq := list.range' 2 ((102 - 2) / 5 + 1)
  let sum_seq := seq.sum
  (sum_seq % 15) = 6 :=
by
  sorry

end sum_mod_15_l749_749857


namespace equilateral_triangle_ratio_correct_l749_749614

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l749_749614


namespace digits_0_and_1_multiple_digits_1_multiple_if_coprime_10_l749_749158

theorem digits_0_and_1_multiple (N : ℕ) (hN : N > 0) : ∃ m : ℕ, (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) ∧ N ∣ m :=
sorry

theorem digits_1_multiple_if_coprime_10 (N : ℕ) (hN : N > 0) (h_coprime : Nat.coprime N 10) : ∃ m : ℕ, (∀ d ∈ m.digits 10, d = 1) ∧ N ∣ m :=
sorry

end digits_0_and_1_multiple_digits_1_multiple_if_coprime_10_l749_749158


namespace count_positive_integers_in_range_sq_le_count_positive_integers_in_range_sq_l749_749384

theorem count_positive_integers_in_range_sq_le (x : ℕ) : 
  225 ≤ x^2 ∧ x^2 ≤ 400 → x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20 :=
by { sorry }

theorem count_positive_integers_in_range_sq : 
  { x : ℕ | 225 ≤ x^2 ∧ x^2 ≤ 400 }.to_finset.card = 6 :=
by { sorry }

end count_positive_integers_in_range_sq_le_count_positive_integers_in_range_sq_l749_749384


namespace points_lie_on_ellipse_l749_749874

noncomputable def curve_points (t : ℝ) : ℝ × ℝ :=
  (cos t + sin t, 4 * (cos t - sin t))

theorem points_lie_on_ellipse :
  ∀ t : ℝ, let (x, y) := curve_points t in
  (x^2 / 2) + (y^2 / 32) = 1 :=
by sorry

end points_lie_on_ellipse_l749_749874


namespace greatest_possible_points_l749_749118

theorem greatest_possible_points (n_teams games_per_pair total_points award_points : ℕ) 
  (h1 : n_teams = 8) 
  (h2 : games_per_pair = 2) 
  (h3 : total_points = 3 * (binom n_teams 2) * games_per_pair)
  (h4 : award_points = 3) 
  : ∃ points, points = 36 ∧ 
              ∀ t1 t2 t3, t1 ∈ top_teams ∧ t2 ∈ top_teams ∧ t3 ∈ top_teams → 
              points_earned t1 = points_earned t2 ∧ points_earned t2 = points_earned t3 ∧ 
              points_earned t1 = 36 := 
sorry

end greatest_possible_points_l749_749118


namespace resulting_chemical_percentage_l749_749767

theorem resulting_chemical_percentage 
  (init_solution_pct : ℝ) (replacement_frac : ℝ) (replacing_solution_pct : ℝ) (resulting_solution_pct : ℝ) : 
  init_solution_pct = 0.85 →
  replacement_frac = 0.8181818181818182 →
  replacing_solution_pct = 0.30 →
  resulting_solution_pct = 0.40 :=
by
  intros h1 h2 h3
  sorry

end resulting_chemical_percentage_l749_749767


namespace a_range_iff_l749_749107

theorem a_range_iff (a x : ℝ) (h1 : x < 3) (h2 : (a - 1) * x < a + 3) : 
  1 ≤ a ∧ a < 3 := 
by
  sorry

end a_range_iff_l749_749107


namespace floor_sqrt_245_l749_749354

theorem floor_sqrt_245 : (Int.floor (Real.sqrt 245)) = 15 :=
by
  sorry

end floor_sqrt_245_l749_749354


namespace positive_integers_count_count_positive_integers_l749_749386

open Nat

theorem positive_integers_count (x : ℕ) :
  (225 ≤ x * x ∧ x * x ≤ 400) ↔ (x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20) :=
by sorry

theorem count_positive_integers : 
  Finset.card { x : ℕ | 225 ≤ x * x ∧ x * x ≤ 400 } = 6 :=
by
  sorry

end positive_integers_count_count_positive_integers_l749_749386


namespace arcsin_one_half_eq_pi_six_l749_749811

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l749_749811


namespace t_in_base_c_l749_749346

theorem t_in_base_c (c : ℕ) (t : ℕ) :
  ((c + 4) * (c + 7) * (c + 9) = 5 * c^3 + 2 * c^2 + 4 * c + 3) →
  t = (14 + 17 + 19 : ℕ) →
  c = 11 →
  (t : ℕ) = 49 :=
by
  intros h_product h_sum h_c
  rw [h_c, h_sum]
  sorry

end t_in_base_c_l749_749346


namespace cans_for_credit_l749_749553

theorem cans_for_credit (P C R : ℕ) : 
  (3 * P = 2 * C) → (C ≠ 0) → (R ≠ 0) → P * R / C = (P * R / C : ℕ) :=
by
  intros h1 h2 h3
  -- proof required here
  sorry

end cans_for_credit_l749_749553


namespace monotonicity_a_eq_1_range_of_a_l749_749933

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_a_eq_1 :
  (∀ x : ℝ, 0 < x → (exp x + 2 * x - 1 > 0)) ∧
  (∀ x : ℝ, x < 0 → (exp x + 2 * x - 1 < 0)) := sorry

-- Part 2: Range of a for f(x) ≥ 1/2 * x ^ 3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 <= x → (exp x + a * x^2 - x >= 1/2 * x^3 + 1)) ↔
  (a ≥ (7 - exp 2) / 4) := sorry

end monotonicity_a_eq_1_range_of_a_l749_749933


namespace sarah_calculate_profit_l749_749200

noncomputable def sarah_total_profit (hot_day_price : ℚ) (regular_day_price : ℚ) (cost_per_cup : ℚ) (cups_per_day : ℕ) (hot_days : ℕ) (total_days : ℕ) : ℚ := 
  let hot_day_revenue := hot_day_price * cups_per_day * hot_days
  let regular_day_revenue := regular_day_price * cups_per_day * (total_days - hot_days)
  let total_revenue := hot_day_revenue + regular_day_revenue
  let total_cost := cost_per_cup * cups_per_day * total_days
  total_revenue - total_cost

theorem sarah_calculate_profit : 
  let hot_day_price := (20951704545454546 : ℚ) / 10000000000000000
  let regular_day_price := hot_day_price / 1.25
  let cost_per_cup := 75 / 100
  let cups_per_day := 32
  let hot_days := 4
  let total_days := 10
  sarah_total_profit hot_day_price regular_day_price cost_per_cup cups_per_day hot_days total_days = (34935102 : ℚ) / 10000000 :=
by
  sorry

end sarah_calculate_profit_l749_749200


namespace smallest_positive_integer_with_18_divisors_l749_749694

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749694


namespace find_s_l749_749424

theorem find_s (t p s : ℝ) (h1 : t = 3 * s^3 + 2 * p) (h2 : t = 29) (h3 : p = 3) :
  s = real.cbrt (23 / 3) :=
by
  sorry

end find_s_l749_749424


namespace find_lambda_l749_749013

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749013


namespace river_joe_orders_l749_749198

theorem river_joe_orders :
  ∃ c s : ℕ, c + s = 26 ∧ (6 * c + 35 * s = 1335) ∧ s = 9 :=
by
  let c := 26 - s
  have h1 : c + s = 26 := sorry
  have h2 : 6 * c + 35 * s = 1335 := sorry
  have final : s = 9 := sorry
  use c
  use s
  exact ⟨ h1, h2, final ⟩

end river_joe_orders_l749_749198


namespace isosceles_right_triangle_circumcircle_side_length_l749_749743

theorem isosceles_right_triangle_circumcircle_side_length
  (ABC : Type) [triangle ABC]
  (isosceles_right_triangle : ∀ {A B C : ABC}, ∠ C = 90 → BC = AC)
  (circumscribing_circle_radius : ∀ {A B C : ABC}, circumscribing_circle ABC radius = 8)
  : ∃ BC, BC = 8 * real.sqrt 2 :=
begin
  sorry
end

end isosceles_right_triangle_circumcircle_side_length_l749_749743


namespace product_sequence_equals_fraction_l749_749780

theorem product_sequence_equals_fraction :
  (∏ n in Finset.range 48, (n + 1) / (n + 6)) = (1 / 56) :=
by
  sorry

end product_sequence_equals_fraction_l749_749780


namespace sum_cos_positive_negative_l749_749539

/-- 
  The sum \( \cos 32 x + a_{31} \cos 31 x + a_{30} \cos 30 x + \cdots + a_{1} \cos x \)
  takes both positive and negative values.
-/
theorem sum_cos_positive_negative (a : ℕ → ℝ) : 
  ∃ x : ℝ, (cos (32 * x) + a 31 * cos (31 * x) + a 30 * cos (30 * x) + 
              a 29 * cos (29 * x) + a 28 * cos (28 * x) + a 27 * cos (27 * x) +
              a 26 * cos (26 * x) + a 25 * cos (25 * x) + a 24 * cos (24 * x) + 
              a 23 * cos (23 * x) + a 22 * cos (22 * x) + a 21 * cos (21 * x) + 
              a 20 * cos (20 * x) + a 19 * cos (19 * x) + a 18 * cos (18 * x) + 
              a 17 * cos (17 * x) + a 16 * cos (16 * x) + a 15 * cos (15 * x) + 
              a 14 * cos (14 * x) + a 13 * cos (13 * x) + a 12 * cos (12 * x) + 
              a 11 * cos (11 * x) + a 10 * cos (10 * x) + a 9 * cos (9 * x) + 
              a 8 * cos (8 * x) + a 7 * cos (7 * x) + a 6 * cos (6 * x) + 
              a 5 * cos (5 * x) + a 4 * cos (4 * x) + a 3 * cos (3 * x) + 
              a 2 * cos (2 * x) + a 1 * cos (x)) > 0 ∧
  ∃ y : ℝ, (cos (32 * y) + a 31 * cos (31 * y) + a 30 * cos (30 * y) + 
              a 29 * cos (29 * y) + a 28 * cos (28 * y) + a 27 * cos (27 * y) +
              a 26 * cos (26 * y) + a 25 * cos (25 * y) + a 24 * cos (24 * y) + 
              a 23 * cos (23 * y) + a 22 * cos (22 * y) + a 21 * cos (21 * y) + 
              a 20 * cos (20 * y) + a 19 * cos (19 * y) + a 18 * cos (18 * y) + 
              a 17 * cos (17 * y) + a 16 * cos (16 * y) + a 15 * cos (15 * y) + 
              a 14 * cos (14 * y) + a 13 * cos (13 * y) + a 12 * cos (12 * y) + 
              a 11 * cos (11 * y) + a 10 * cos (10 * y) + a 9 * cos (9 * y) + 
              a 8 * cos (8 * y) + a 7 * cos (7 * y) + a 6 * cos (6 * y) + 
              a 5 * cos (5 * y) + a 4 * cos (4 * y) + a 3 * cos (3 * y) + 
              a 2 * cos (2 * y) + a 1 * cos (y)) < 0 :=
sorry

end sum_cos_positive_negative_l749_749539


namespace distance_from_M_to_focus_l749_749906

noncomputable def parabola_focus_distance (M : (ℝ × ℝ)) (C : ℝ → ℝ → Prop) : ℝ :=
{ x := 2, y := 4 }

def parabola (y x p : ℝ) : Prop :=
y^2 = 2 * p * x

def parabola_focus (p : ℝ) : ℝ × ℝ :=
(if p = (2 : ℝ) then (2, 0) else (0, 0))

def distance (A B : ℝ × ℝ) : ℝ :=
Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem distance_from_M_to_focus (M : (ℝ × ℝ)) (p : ℝ) (C : ℝ → ℝ → Prop)
  (h₁ : M = (2, 4))
  (h₂ : parabola (M.2) (M.1) p)
  (h₃ : p = 4) :
  distance M (parabola_focus p) = 4 :=
sorry

end distance_from_M_to_focus_l749_749906


namespace license_plates_count_l749_749449

theorem license_plates_count :
  let letters := 26^3 in
  let first_digit_choices := 5 in
  let second_digit_choices := 10 in
  let third_digit_choices := 4 in
  letters * first_digit_choices * second_digit_choices * third_digit_choices = 3514400 :=
by
  let letters := 26^3
  let first_digit_choices := 5
  let second_digit_choices := 10
  let third_digit_choices := 4
  calc
    letters * first_digit_choices * second_digit_choices * third_digit_choices
    = 17576 * 5 * 10 * 4 : by sorry
    ... = 3514400 : by sorry

end license_plates_count_l749_749449


namespace dot_product_range_l749_749423

   noncomputable section

   open Real

   -- Definitions of the ellipse parameters and solving the conditions for a and b
   def ellipse_params : {a b c : ℕ} × ℝ × ℝ × ℝ :=
     let a := 5
     let b := 4
     let c := 3
     (⟨a, b, c⟩, a^2 - b^2 - c^2, 2 * b - (a + c), c)

   -- Point P and the focus F in the coordinate space with the conditions provided
   def point_P_in_ellipse (x₀ y₀ : ℝ) : Prop :=
     (x₀ > 0) ∧ (x₀ < 5) ∧ (y₀ > 0) ∧ (y₀ < 4) ∧ ( (x₀^2 / 25) + (y₀^2 / 16) = 1 )

   -- The final dot product expression as per given conditions transformed for proof
   def dot_product_OP_PF (x₀ : ℝ) : ℝ :=
     - (9/25) * x₀^2 + 3 * x₀ - 16

   -- Prove the given range for the dot product of vectors OP and PF
   theorem dot_product_range (x₀ : ℝ) (y₀ : ℝ) (h : point_P_in_ellipse x₀ y₀) :
     -16 < dot_product_OP_PF x₀ ∧ dot_product_OP_PF x₀ ≤ -39/4 :=
   sorry
   
end dot_product_range_l749_749423


namespace total_ways_to_choose_gifts_l749_749213

/-- The 6 pairs of zodiac signs -/
def zodiac_pairs : Set (Set String) :=
  {{"Rat", "Ox"}, {"Tiger", "Rabbit"}, {"Dragon", "Snake"}, {"Horse", "Sheep"}, {"Monkey", "Rooster"}, {"Dog", "Pig"}}

/-- The preferences of Students A, B, and C -/
def A_likes : Set String := {"Ox", "Horse"}
def B_likes : Set String := {"Ox", "Dog", "Sheep"}
def C_likes : Set String := {"Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", "Horse", "Sheep", "Monkey", "Rooster", "Dog", "Pig"}

theorem total_ways_to_choose_gifts : 
  True := 
by
  -- We prove that the number of ways is 16
  sorry

end total_ways_to_choose_gifts_l749_749213


namespace compare_f_l749_749904

noncomputable def f : ℝ → ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x ≠ 0
axiom f_deriv : ∀ x : ℝ, deriv f x < f x

theorem compare_f (h : ∀ x : ℝ, deriv f x < f x) : 
  f 2 < exp 2 * f 0 ∧ f 2023 < exp 2023 * f 0 :=
by
  sorry

end compare_f_l749_749904


namespace number_of_divisors_l749_749446

theorem number_of_divisors (n : ℕ) (h1: n = 70) (h2: ∀ k, k ∣ n → k > 3 ↔ (k = 5 ∨ k = 7 ∨ k = 10 ∨ k = 14 ∨ k = 35 ∨ k = 70)) : 
  {k : ℕ | k ∣ n ∧ k > 3}.to_finset.card = 6 :=
by {
  simp [h1, h2],
  sorry
}

end number_of_divisors_l749_749446


namespace maria_zoo_ticket_discount_percentage_l749_749532

theorem maria_zoo_ticket_discount_percentage 
  (regular_price : ℝ) (paid_price : ℝ) (discount_percentage : ℝ)
  (h1 : regular_price = 15) (h2 : paid_price = 9) :
  discount_percentage = 40 :=
by
  sorry

end maria_zoo_ticket_discount_percentage_l749_749532


namespace book_page_count_l749_749966

theorem book_page_count (x : ℝ) : 
    (x - (1 / 4 * x + 20)) - ((1 / 3 * (x - (1 / 4 * x + 20)) + 25)) - (1 / 2 * ((x - (1 / 4 * x + 20)) - (1 / 3 * (x - (1 / 4 * x + 20)) + 25)) + 30) = 70 →
    x = 480 :=
by
  sorry

end book_page_count_l749_749966


namespace increasing_interval_of_f_l749_749572

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x

theorem increasing_interval_of_f :
  ∀ x : ℝ, 3 ≤ x → ∀ y : ℝ, 3 ≤ y → x < y → f x < f y := 
sorry

end increasing_interval_of_f_l749_749572


namespace radius_of_inscribed_semicircle_l749_749308

/-
  Given a rectangle formed by reflecting an isosceles triangle with a base of 24 and height 10 over its base,
  and a semicircle inscribed along the base of the triangle, prove that the radius of the semicircle is 60/11.
-/

theorem radius_of_inscribed_semicircle :
  let base := 24
  let height := 10
  let semiperimeter := (base + 2 * height) / 2
  let area := base * (2 * height)
  let r := area / (2 * semiperimeter)
  r = 60 / 11 :=
by {
  exact base, 
  exact height,
  exact semiperimeter,
  exact area,
  exact r,
  have h1 : semiperimeter = (24 + 2 * 10) / 2 := rfl,
  have h2 : area = 24 * (2 * 10) := rfl,
  have h3 : r = area / (2 * semiperimeter) := rfl,
  show r = 60 / 11,
  calc
    r = 480 / (2 * 44) : by { rw h2, rw h1, sorry }
    ... = 60 / 11 : by { sorry }
}

end radius_of_inscribed_semicircle_l749_749308


namespace man_l749_749300

theorem man's_speed_downstream (v : ℕ) (h1 : v - 3 = 8) (s : ℕ := 3) : v + s = 14 :=
by
  sorry

end man_l749_749300


namespace find_lambda_l749_749079

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749079


namespace range_of_a_l749_749886

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, e^x + 1/e^x > a) ∧ (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ↔ (-4 ≤ a ∧ a < 2) :=
by
  sorry

end range_of_a_l749_749886


namespace sum_of_b_for_one_solution_l749_749848

theorem sum_of_b_for_one_solution :
  let f (b : ℝ) := 3 * x^2 + (b + 12) * x + 16
  let discriminant condition : ℝ → Prop := λ b, (b + 12) ^ 2 = 192
  (∑ b in {b : ℝ | discriminant condition b}, b) = -24 :=
by
  sorry

end sum_of_b_for_one_solution_l749_749848


namespace train_speed_calculation_l749_749318

def train_speed (train_length bridge_length time_seconds : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := total_distance / time_seconds
  speed_mps * 36

theorem train_speed_calculation :
  train_speed 100 140 23.998080153587715 = 36 :=
by sorry

end train_speed_calculation_l749_749318


namespace smallest_integer_with_18_divisors_l749_749643

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749643


namespace lambda_solution_l749_749067

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749067


namespace calculate_expression_l749_749783

noncomputable def cube_root (x : ℝ) := x^(1/3:ℝ)
noncomputable def abs (x : ℝ) := if x < 0 then -x else x
noncomputable def square_root (x : ℝ) := x^(1/2:ℝ)
noncomputable def power (x : ℝ) (n : ℤ) := x^n

theorem calculate_expression : 
  cube_root (-8) + abs (1 - real.pi) + square_root (9) - power (-1) 2 = real.pi - 1 :=
by
  sorry

end calculate_expression_l749_749783


namespace sum_of_solutions_l749_749712

theorem sum_of_solutions : 
  (∑ x in {n | |3 * n - 8| = 4}, x) = 16 / 3 :=
by
  sorry

end sum_of_solutions_l749_749712


namespace range_of_b_l749_749918

theorem range_of_b (b : ℝ) (h : ∃ x ∈ set.Icc (1/2 : ℝ) 2, x^2 + 2 * x - b * (x + 1) > 0) : b < 8 / 3 :=
sorry

end range_of_b_l749_749918


namespace smallest_integer_with_18_divisors_l749_749707

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749707


namespace remaining_money_l749_749187

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end remaining_money_l749_749187


namespace flower_bee_relationship_l749_749587

def numberOfBees (flowers : ℕ) (fewer_bees : ℕ) : ℕ :=
  flowers - fewer_bees

theorem flower_bee_relationship :
  numberOfBees 5 2 = 3 := by
  sorry

end flower_bee_relationship_l749_749587


namespace at_least_four_2x2_squares_with_sum_greater_than_100_l749_749251

theorem at_least_four_2x2_squares_with_sum_greater_than_100 :
  ∀ (board : Fin 8 → Fin 8 → ℕ),
  (∀ i j, board i j ∈ Finset.range 64) →
  (Finset.univ.bUnion (λ i, Finset.univ.bUnion (λ j, ({board i j} : Finset ℕ))) = Finset.range 64) →
  ∃ (S : Finset (Fin 8 × Fin 8)),
  S.card ≥ 4 ∧
  ∀ (p : (Fin 8 × Fin 8)), p ∈ S →
  let sum2x2 := (board p.1 p.2) + (board (p.1 + 1) p.2) + (board p.1 (p.2 + 1)) + (board (p.1 + 1) (p.2 + 1))
  in sum2x2 > 100 := 
begin 
  sorry
end

end at_least_four_2x2_squares_with_sum_greater_than_100_l749_749251


namespace find_lambda_l749_749076

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749076


namespace asymptote_equations_l749_749948

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) (e : ℝ) (x y : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (e = sqrt 3) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

theorem asymptote_equations (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : sqrt (a^2 + b^2) / a = sqrt 3) :
  ∀ (x : ℝ), ∃ (y : ℝ), y = sqrt 2 * x ∨ y = -sqrt 2 * x :=
sorry

end asymptote_equations_l749_749948


namespace smallest_integer_with_18_divisors_l749_749676

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749676


namespace problem1_problem2_l749_749397

-- Given function
def f (α : ℝ) : ℝ :=
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.cos ((3 * Real.pi / 2) + α)) /
  (Real.cos ((Real.pi / 2) + α) * Real.sin (Real.pi + α))

-- Problem 1
theorem problem1 : f (-Real.pi / 3) = 1 / 2 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) (h3 : Real.cos (α - Real.pi / 2) = 3 / 5) : f α = -4 / 5 :=
by sorry

end problem1_problem2_l749_749397


namespace base_5_to_base_10_conversion_l749_749329

/-- An alien creature communicated that it produced 263_5 units of a resource. 
    Convert this quantity to base 10. -/
theorem base_5_to_base_10_conversion : ∀ (n : ℕ), n = 2 * 5^2 + 6 * 5^1 + 3 * 5^0 → n = 83 :=
by
  intros n h
  rw [h]
  sorry

end base_5_to_base_10_conversion_l749_749329


namespace intersection_complement_eq_one_l749_749166

def U : set ℕ := {0, 1, 2, 3, 4}
def A : set ℕ := {1, 2, 3}
def B : set ℕ := {2, 3, 4}

theorem intersection_complement_eq_one : (A ∩ (U \ B)) = {1} := 
by 
  sorry

end intersection_complement_eq_one_l749_749166


namespace no_sum_14_l749_749981

theorem no_sum_14 (x y : ℤ) (h : x * y + 4 = 40) : x + y ≠ 14 :=
by sorry

end no_sum_14_l749_749981


namespace monotonicity_f_when_a_eq_1_range_of_a_l749_749927

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_f_when_a_eq_1 :
  (∀ x > 0, deriv (λ x, f x 1) x > 0) ∧ (∀ x < 0, deriv (λ x, f x 1) x < 0) :=
sorry

-- Part 2: Range of a such that f(x) ≥ 1/2 * x^3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x a ≥ 1/2 * x^3 + 1) ↔ a ≥ (7 - Real.exp 2) / 4 :=
sorry

end monotonicity_f_when_a_eq_1_range_of_a_l749_749927


namespace lambda_value_l749_749030

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749030


namespace abc_sum_eq_sixteen_l749_749500

theorem abc_sum_eq_sixteen (a b c : ℤ) (h1 : a ≠ b ∨ a ≠ c ∨ b ≠ c) (h2 : a ≥ 4 ∧ b ≥ 4 ∧ c ≥ 4) (h3 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by 
  sorry

end abc_sum_eq_sixteen_l749_749500


namespace no_complete_cover_possible_l749_749590

theorem no_complete_cover_possible :
  ∀ (board_size : ℕ) (piece1 : ℕ × ℕ) (piece2 : ℕ × ℕ), 
  board_size = 2003 → 
  piece1 = (1, 2) → 
  piece2 = (1, 3) → 
  ¬ (∃ (placement: list (ℕ × ℕ) × ℕ), 
  ∀ (pos : ℕ × ℕ), 
  pos.1 < board_size ∧ pos.2 < board_size → 
  (placement.1 = [(1, 2)] ∨ placement.1 = [(1, 3)]) ∧ 
  pos.2 % 3 = 0 → 
  (placement.snd = placement.snd + 3)
  ) 
  :=
begin
  intros,
  sorry
end

end no_complete_cover_possible_l749_749590


namespace subtract3_from_M_l749_749969

def M_binary := "101010" -- Define the binary representation of M as a string.

def M_decimal : ℕ := 42 -- Convert M to its decimal representation.

def expected_result := "100111" -- Define the expected result as a string.

theorem subtract3_from_M : ("101010".to_nat 2) - 3 = "100111".to_nat 2 :=
by
  sorry

end subtract3_from_M_l749_749969


namespace find_a_max_area_triangle_min_tangent_distance_l749_749400

variables {a : ℝ}
noncomputable def circle_eq : (x y : ℝ) → Prop := 
  λ x y, x^2 + y^2 + a * x - 4 * y + 1 = 0

noncomputable def midpoint_A_B := P : point → (x y : ℝ) → circle_eq x y → circle_eq x y → Prop
  | P : point, A : point, B : point, circle_eq x₁₁ y₁₁, circle_eq x₂₂ y₂₂ :=
    (P.1 = (x₁₁ + x₂₂) / 2) ∧ (P.2 = (y₁₁ + y₂₂) / 2)

theorem find_a (x y : ℝ) (P : ℝ × ℝ) : midpoint_A_B P A B → midpoint_A_B (0, 1) A B → a = 2 :=
sorry

theorem max_area_triangle (x y : ℝ) (xA yA xB yB : ℝ) (E : ℝ × ℝ) (K : ℝ) 
    (P : ℝ × ℝ) (A B : ℝ × ℝ) :
    circle_eq x y → circle_eq xA yA → circle_eq xB yB → circle_eq E.1 E.2 →
    midpoint_A_B P A B → 
    |(2+2*sqrt(2))| :=
sorry

theorem min_tangent_distance (x y : ℝ) (xM yM : ℝ) (P : ℝ × ℝ) 
    (M : ℝ × ℝ) (N : ℝ × ℝ) :
    circle_eq x y → circle_eq xM yM → 
    |(sqrt(2)/2)| ∧ M = (1/2, 1/2) :=
sorry

end find_a_max_area_triangle_min_tangent_distance_l749_749400


namespace new_polyhedron_edges_l749_749290

theorem new_polyhedron_edges (n : ℕ) (Q : ConvexPolyhedron) (V : Fin n → Vertex) (P : Fin n → Plane) 
  (hQ_edges : Q.edges = 120) 
  (hVk_cuts : ∀ k, cuts_edge (P k) (emits_from (V k))) 
  (hPk_adds_edges : ∀ k, adds_edges (P k) 2 (emits_from (V k))) 
  (hP_intersect : ∀ i j, i ≠ j → ¬intersects (P i) (P j) (surface Q)) :
  ∃ R : ConvexPolyhedron, R.edges = 840 :=
sorry

end new_polyhedron_edges_l749_749290


namespace plan_y_cost_effective_l749_749325

theorem plan_y_cost_effective (m : ℕ) (h1 : ∀ minutes, cost_plan_x = 15 * minutes)
(h2 : ∀ minutes, cost_plan_y = 3000 + 10 * minutes) :
m ≥ 601 → 3000 + 10 * m < 15 * m :=
by
sorry

end plan_y_cost_effective_l749_749325


namespace laura_annual_income_l749_749993

variable (p : ℝ) -- percentage p
variable (A T : ℝ) -- annual income A and total income tax T

def tax1 : ℝ := 0.01 * p * 35000
def tax2 : ℝ := 0.01 * (p + 3) * (A - 35000)
def tax3 : ℝ := 0.01 * (p + 5) * (A - 55000)

theorem laura_annual_income (h_cond1 : A > 55000)
  (h_tax : T = 350 * p + 600 + 0.01 * (p + 5) * (A - 55000))
  (h_paid_tax : T = (0.01 * (p + 0.45)) * A):
  A = 75000 := by
  sorry

end laura_annual_income_l749_749993


namespace rosa_bonheur_birth_day_l749_749212

/--
Given that Rosa Bonheur's 210th birthday was celebrated on a Wednesday,
prove that she was born on a Sunday.
-/
theorem rosa_bonheur_birth_day :
  let anniversary_year := 2022
  let birth_year := 1812
  let total_years := anniversary_year - birth_year
  let leap_years := (total_years / 4) - (total_years / 100) + (total_years / 400)
  let regular_years := total_years - leap_years
  let day_shifts := regular_years + 2 * leap_years
  (3 - day_shifts % 7) % 7 = 0 := 
sorry

end rosa_bonheur_birth_day_l749_749212


namespace equilateral_triangle_ratio_correct_l749_749616

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l749_749616


namespace find_lambda_l749_749055

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749055


namespace day_of_week_after_6_pow_2023_l749_749255

def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem day_of_week_after_6_pow_2023 :
  day_of_week_after_days 4 (6^2023) = 3 :=
by
  sorry

end day_of_week_after_6_pow_2023_l749_749255


namespace min_Q_value_l749_749872

open Real Nat

def closest_integer_div (m k : ℤ) : ℤ := round (m.to_rat / k.to_rat)

def Q (k : ℤ) : ℚ :=
  let valid_ns := { n | 1 ≤ n ∧ n ≤ 149 ∧ closest_integer_div n k + closest_integer_div (150 - n) k = closest_integer_div 150 k }
  (valid_ns.to_finset.card : ℚ) / 149

theorem min_Q_value :
  ∃ k : ℤ, is_odd_prime k ∧ 1 ≤ k ∧ k ≤ 149 ∧ Q k = 50 / 101 :=
sorry

end min_Q_value_l749_749872


namespace arcsin_one_half_l749_749825

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749825


namespace simplify_expression_l749_749204

theorem simplify_expression :
  8 * (18 / 5) * (-40 / 27) = - (128 / 3) := 
by
  sorry

end simplify_expression_l749_749204


namespace max_area_bpc_correct_sum_a_b_c_correct_l749_749139

open Real

structure Triangle :=
  (A B C : Point)
  (AB BC CA : ℝ)
  (hAB : AB = 8)
  (hBC : BC = 15)
  (hCA : CA = 17)

structure Midpoint :=
  (D : Point)
  (BD DC : ℝ)
  (hMid : BD = BC / 2 ∧ DC = BC / 2)

structure Incenter :=
  (I_B I_C : Point)

structure CircumcircleIntersection :=
  (P : Point)

def maxTriangleArea (T : Triangle) (D : Midpoint) (I : Incenter) (Circ : CircumcircleIntersection) : ℝ :=
  let AD := ⟨D.D⟩
  let cosBAC := (BC * BC + CA * CA - AB * AB) / (2 * BC * CA)
  let sinBAC := sqrt (1 - cosBAC ^ 2)
  let BD := D.BD
  let DC := D.DC
  (1 / 2) * BD * DC * sinBAC

theorem max_area_bpc_correct (T : Triangle) (D : Midpoint) (I : Incenter) (Circ : CircumcircleIntersection) :
  maxTriangleArea T D I Circ = 6.25 * sqrt 13 :=
sorry

theorem sum_a_b_c_correct (T : Triangle) (D : Midpoint) (I : Incenter) (Circ : CircumcircleIntersection) :
  let area := maxTriangleArea T D I Circ
  let a := 6
  let b := 1
  let c := 13
  a + b + c = 20 :=
by
  have h := sum_bpc T D I Circ
  exact h

def sum_bpc (T : Triangle) (D : Midpoint) (I : Incenter) (Circ : CircumcircleIntersection) : ℝ := 
  6 + 1 + 13


end max_area_bpc_correct_sum_a_b_c_correct_l749_749139


namespace option_C_is_true_option_A_is_false_option_B_is_false_option_D_is_false_l749_749716

theorem option_C_is_true (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x ≠ y) :
  (xy / (x^2 - xy)) = (y / (x - y)) :=
begin
  sorry
end

theorem option_A_is_false (x y : ℝ) :
  (-(x) + y) / 2 ≠ (x + y) / 2 := 
begin
  sorry
end

theorem option_B_is_false (x : ℝ) (h : x ≠ 3) :
  (x + 3) / (x^2 + 9) ≠ 1 / (x - 3) :=
begin
  sorry
end

theorem option_D_is_false (x : ℝ) (h : x ≠ 0) :
  (x + 2) / (x^2 + 2x) ≠ x :=
begin
  sorry
end

end option_C_is_true_option_A_is_false_option_B_is_false_option_D_is_false_l749_749716


namespace constant_term_is_ninth_term_l749_749915

def f (x : ℝ) : ℝ := -x^3 + 2 * (f' 2) * x

def n : ℝ := f' 2

theorem constant_term_is_ninth_term (f' : ℝ → ℝ) (h : ∀ x, f' x = -3 * x^2 + 2 * (f' 2)) : 
  (x + 2 / (sqrt x)) ^ n = 
  ((((x + 2 / (sqrt x)) ^ 12).terms.find_index (λ t, t = 12 - 3/2 * 8))) + 1 := 
sorry

end constant_term_is_ninth_term_l749_749915


namespace lambda_value_l749_749024

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749024


namespace monotonicity_a_eq_1_range_of_a_l749_749932

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_a_eq_1 :
  (∀ x : ℝ, 0 < x → (exp x + 2 * x - 1 > 0)) ∧
  (∀ x : ℝ, x < 0 → (exp x + 2 * x - 1 < 0)) := sorry

-- Part 2: Range of a for f(x) ≥ 1/2 * x ^ 3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 <= x → (exp x + a * x^2 - x >= 1/2 * x^3 + 1)) ↔
  (a ≥ (7 - exp 2) / 4) := sorry

end monotonicity_a_eq_1_range_of_a_l749_749932


namespace smallest_positive_integer_with_18_divisors_l749_749692

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749692


namespace smallest_integer_with_18_divisors_l749_749701

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749701


namespace subcommittee_count_l749_749298

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let select_republicans := 4
  let select_democrats := 3
  let num_ways_republicans := Nat.choose republicans select_republicans
  let num_ways_democrats := Nat.choose democrats select_democrats
  let num_ways := num_ways_republicans * num_ways_democrats
  num_ways = 11760 :=
by
  sorry

end subcommittee_count_l749_749298


namespace find_x_orthogonal_vectors_l749_749359

theorem find_x_orthogonal_vectors : ∀ (x : ℝ), 
  let v1 := (3 : ℝ, 4 : ℝ),
      v2 := (x, -6 : ℝ) in
  (3 * x + 4 * (-6) = 0) → x = 8 :=
begin
  intro x,
  intro h,
  sorry
end

end find_x_orthogonal_vectors_l749_749359


namespace lambda_value_l749_749023

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749023


namespace probability_of_desired_event_is_67_over_663_l749_749257

open Classical

def probability_two_cards_sum_to_fifteen_or_both_are_fives (deck : Finset (Fin 52)) : ℚ :=
  let numbers := {n : Fin 52 | n.val % 13 ≤ 10 ∧ n.val % 13 ≥ 2 }
  let valid_pairs := { (a, b) : Fin 52 × Fin 52 | (a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b ∧ 
                                (a.val % 13 + b.val % 13 = 15 ∨ (a.val % 13 = 5 ∧ b.val % 13 = 5))) }
  valid_pairs.card.toRat / ((deck.card.toRat * (deck.card.toRat - 1)))

theorem probability_of_desired_event_is_67_over_663 (deck : Finset (Fin 52)) 
  (hdeck : deck.card = 52) : 
  probability_two_cards_sum_to_fifteen_or_both_are_fives deck = 67 / 663 :=
by 
  sorry

end probability_of_desired_event_is_67_over_663_l749_749257


namespace amitabh_avg_expenditure_feb_to_jul_l749_749273

variable (expenditure_avg_jan_to_jun expenditure_jan expenditure_jul : ℕ)

theorem amitabh_avg_expenditure_feb_to_jul (h1 : expenditure_avg_jan_to_jun = 4200) 
  (h2 : expenditure_jan = 1200) (h3 : expenditure_jul = 1500) :
  (expenditure_avg_jan_to_jun * 6 - expenditure_jan + expenditure_jul) / 6 = 4250 := by
  -- Using the given conditions
  sorry

end amitabh_avg_expenditure_feb_to_jul_l749_749273


namespace arcsin_half_eq_pi_six_l749_749805

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l749_749805


namespace arcsin_half_eq_pi_six_l749_749807

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l749_749807


namespace minimum_keys_needed_l749_749998

theorem minimum_keys_needed (total_cabinets : ℕ) (boxes_per_cabinet : ℕ)
(boxes_needed : ℕ) (boxes_per_cabinet : ℕ) 
(warehouse_key : ℕ) (boxes_per_cabinet: ℕ)
(h1 : total_cabinets = 8)
(h2 : boxes_per_cabinet = 4)
(h3 : (boxes_needed = 52))
(h4 : boxes_per_cabinet = 4)
(h5 : warehouse_key = 1):
    6 + 2 + 1 = 9 := 
    sorry

end minimum_keys_needed_l749_749998


namespace find_angle_A_find_area_l749_749902

noncomputable def m := (sin (A : ℝ), 1)
noncomputable def n := (cos (A : ℝ), sqrt 3)

theorem find_angle_A (A : ℝ) 
  (h_parallel: m = n) : A = π / 6 :=
sorry

theorem find_area (a b : ℝ) 
  (ha: a = 2) (hb: b = 2 * sqrt 2) (A : ℝ)
  (h_A: A = π / 6) : 
  area_of_triangle = 1 + sqrt 3 ∨ area_of_triangle = sqrt 3 - 1 :=
sorry

end find_angle_A_find_area_l749_749902


namespace range_of_m_l749_749225

theorem range_of_m (m : ℝ) (h : m > 0) :
  (∀ x1 ∈ Icc (0 : ℝ) (Real.pi / 4), ∃ x2 ∈ Icc (0 : ℝ) (Real.pi / 4), 
    m * Real.cos (2 * x1 - Real.pi / 6) - 2 * m + 3 = 2 * Real.sin (2 * x2 + Real.pi / 3)) →
  1 ≤ m ∧ m ≤ (4 / 3) :=
by
  sorry

end range_of_m_l749_749225


namespace monotonicity_a1_range_of_a_l749_749935

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- 1. Monotonicity when \( a = 1 \)
theorem monotonicity_a1 :
  (∀ x > 0, (f x 1)' > 0) ∧ (∀ x < 0, (f x 1)' < 0) :=
by
  sorry

-- 2. Range of \( a \) for \( f(x) \geq \frac{1}{2} x^3 + 1 \) for \( x \geq 0 \)
theorem range_of_a (a : ℝ) (x : ℝ) (hx : x ≥ 0) (hf : f x a ≥ (1 / 2) * x^3 + 1) :
  a ≥ (7 - Real.exp 2) / 4 :=
by
  sorry

end monotonicity_a1_range_of_a_l749_749935


namespace lambda_value_l749_749037

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749037


namespace sqrt_x_eq_0_123_l749_749882

theorem sqrt_x_eq_0_123 (x : ℝ) (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  -- proof goes here, but it is omitted
  sorry

end sqrt_x_eq_0_123_l749_749882


namespace lambda_value_l749_749036

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749036


namespace example_theorem_l749_749471

noncomputable def P (A : Set ℕ) : ℝ := sorry

variable (A1 A2 A3 : Set ℕ)

axiom prob_A1 : P A1 = 0.2
axiom prob_A2 : P A2 = 0.3
axiom prob_A3 : P A3 = 0.5

theorem example_theorem : P (A1 ∪ A2) ≤ 0.5 := 
by {
  sorry
}

end example_theorem_l749_749471


namespace covers_all_except_a_series_l749_749952

def a_series (n : ℕ) := 3 * n ^ 2 - 2 * n

def f (n : ℕ) : ℕ := ⌊n + sqrt (n / 3) + 1 / 2⌋

theorem covers_all_except_a_series :
  ∀ m, ∃ n, f n = m ∧ (∀ k, m ≠ a_series k) :=
sorry

end covers_all_except_a_series_l749_749952


namespace range_of_a_l749_749920

-- Definitions from conditions
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- Given the conditions
theorem range_of_a (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 3) :
  (∀ x : ℝ, x ∈ set.Icc 1 a → f(x) ≥ f(a)) :=
by
  sorry

end range_of_a_l749_749920


namespace eq_sol_unit_circle_l749_749548

theorem eq_sol_unit_circle {n : ℕ} :
  (∃ z : ℂ, |z| = 1 ∧ z^n + z + 1 = 0) ↔ (n - 2) % 3 = 0 :=
by
  sorry

end eq_sol_unit_circle_l749_749548


namespace find_lambda_l749_749070

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749070


namespace p_or_q_iff_not_p_and_not_q_false_l749_749972

variables (p q : Prop)

theorem p_or_q_iff_not_p_and_not_q_false : (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
by sorry

end p_or_q_iff_not_p_and_not_q_false_l749_749972


namespace at_least_60_percent_speak_same_language_l749_749333

noncomputable def language_participants (n : ℕ) (A B C D X Y : ℕ) : Prop :=
  2 * X + 3 * Y ≤ A + B + C + D ∧ A + B + C + D = n

theorem at_least_60_percent_speak_same_language
  (n : ℕ) (A B C D X Y : ℕ)
  (h : language_participants n A B C D X Y) :
  max (max A (max B C)) D ≥ 60 * n / 100 :=
begin
  sorry
end

end at_least_60_percent_speak_same_language_l749_749333


namespace smallest_integer_with_18_divisors_l749_749687

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749687


namespace probability_individual_selected_stratified_l749_749871

variable (m : ℕ)
variable (hm : m ≥ 3)

def probability_of_individual_selected_systematic : ℚ := 1 / 3

theorem probability_individual_selected_stratified :
  probability_of_individual_selected_systematic m hm = 1 / 3 := 
sorry

end probability_individual_selected_stratified_l749_749871


namespace sale_price_of_trouser_l749_749142

theorem sale_price_of_trouser (original_price : ℝ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) (h2 : discount_percentage = 0.5) : sale_price = 50 :=
by
  sorry

end sale_price_of_trouser_l749_749142


namespace MH_length_l749_749122

/-
In rectangle $EFGH$, point $X$ is on $FG$ such that $\angle EXH = 90^\circ$. $LM$ is perpendicular to $FG$ with $FX=XL$. $XH$ intersects $LM$ at $N$. Point $Z$ is on $GH$ such that $ZE$ passes through $N$. In $\triangle XNE$, $XE=15$, $EN=20$ and $XN=9$. Find $MH$.
-/

theorem MH_length
  (X L M N Z E F G H : Point)
  (rect_EFGH : Rectangle E F G H)
  (X_on_FG : LineSegment F G X)
  (angle_EXH : angle E X H = 90)
  (LM_perpendicular_FG : Perpendicular L M F G)
  (FX_is_XL : distance F X = distance X L)
  (XH_intersects_LM_at_N : Intersects X H L M N)
  (Z_on_GH : LineSegment G H Z)
  (ZE_passes_through_N : PassesThrough Z E N)
  (dist_XE : distance X E = 15)
  (dist_EN : distance E N = 20)
  (dist_XN : distance X N = 9) :
  let MH := find_MH E F G H X L M N Z in
  MH = 12.343 :=
sorry

end MH_length_l749_749122


namespace number_of_years_borrowed_l749_749302

theorem number_of_years_borrowed (n : ℕ)
  (H1 : ∃ (p : ℕ), 5000 = p ∧ 4 = 4 ∧ n * 200 = 150)
  (H2 : ∃ (q : ℕ), 5000 = q ∧ 7 = 7 ∧ n * 350 = 150)
  : n = 1 :=
by
  sorry

end number_of_years_borrowed_l749_749302


namespace simplify_fraction_multiplication_l749_749205

theorem simplify_fraction_multiplication:
  (101 / 5050) * 50 = 1 := by
  sorry

end simplify_fraction_multiplication_l749_749205


namespace value_of_a6_l749_749488

-- Define the conditions of the problem
variable {a_n : ℕ → ℝ}

-- Definition of the sequence being geometric
def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) / a_n n = a_n (n + 2) / a_n (n + 1)

-- Conditions for the problem
axiom a4_a8_roots : ∃ a_4 a_8 : ℝ, a_4 + a_8 = 3 ∧ a_4 * a_8 = 2
axiom seq_geometric : geometric_sequence a_n

-- Prove the required result
theorem value_of_a6 : ∃ a_6 : ℝ, a_6 = sqrt 2 :=
by
  sorry

end value_of_a6_l749_749488


namespace proof_problem_l749_749957

-- Definitions based on the given conditions
def cond1 : Prop := 1 * 9 + 2 = 11
def cond2 : Prop := 12 * 9 + 3 = 111
def cond3 : Prop := 123 * 9 + 4 = 1111
def cond4 : Prop := 1234 * 9 + 5 = 11111
def cond5 : Prop := 12345 * 9 + 6 = 111111

-- Main statement to prove
theorem proof_problem (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) : 
  123456 * 9 + 7 = 1111111 :=
sorry

end proof_problem_l749_749957


namespace divisors_count_of_108n5_l749_749392

theorem divisors_count_of_108n5 {n : ℕ} (hn_pos : 0 < n) (h_divisors_150n3 : (150 * n^3).divisors.card = 150) : 
(108 * n^5).divisors.card = 432 :=
sorry

end divisors_count_of_108n5_l749_749392


namespace x_is_4286_percent_less_than_y_l749_749466

theorem x_is_4286_percent_less_than_y (x y : ℝ) (h : y = 1.75 * x) : 
  ((y - x) / y) * 100 = 42.86 :=
by
  sorry

end x_is_4286_percent_less_than_y_l749_749466


namespace pierre_nathalie_ratio_l749_749285

-- Define the cake weight
def cake_weight : ℝ := 400

-- Define the number of parts the cake is divided into
def parts : ℕ := 8

-- Define the weight of each part
def part_weight : ℝ := cake_weight / parts

-- Define the weight Nathalie ate (one part)
def nathalie_ate : ℝ := part_weight

-- Define the weight Pierre ate
def pierre_ate : ℝ := 100

-- Define the ratio of the amount Pierre ate to the amount Nathalie ate
def ratio : ℝ := pierre_ate / nathalie_ate

-- Statement of the theorem to prove the ratio is 2:1
theorem pierre_nathalie_ratio : ratio = 2 := by
  -- skip the proof
  sorry

end pierre_nathalie_ratio_l749_749285


namespace find_lambda_l749_749072

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749072


namespace find_r_divisibility_l749_749393

theorem find_r_divisibility :
  ∃ r : ℝ, (10 * r ^ 2 - 4 * r - 26 = 0 ∧ (r = (19 / 10) ∨ r = (-3 / 2))) ∧ (r = -3 / 2) ∧ (10 * r ^ 3 - 5 * r ^ 2 - 52 * r + 60 = 0) :=
by
  sorry

end find_r_divisibility_l749_749393


namespace three_to_the_sum_l749_749100

theorem three_to_the_sum {a b : ℝ} (h1 : 3^a = 2) (h2 : 3^b = 5) : 3^(a + b) = 10 := 
by 
  sorry

end three_to_the_sum_l749_749100


namespace smallest_positive_integer_with_18_divisors_l749_749664

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749664


namespace solve_for_x_l749_749549

theorem solve_for_x : 
  (let x := \frac{\sqrt{8^2 + 15^2}}{\sqrt{25 + 16}} 
   in x = \frac{17}{\sqrt{41}}) :=
by
  sorry

end solve_for_x_l749_749549


namespace initial_ratio_of_milk_to_water_l749_749991

variable (M W : ℕ) -- M represents the amount of milk, W represents the amount of water

theorem initial_ratio_of_milk_to_water (h1 : M + W = 45) (h2 : 8 * M = 9 * (W + 23)) :
  M / W = 4 :=
by
  sorry

end initial_ratio_of_milk_to_water_l749_749991


namespace solution_set_of_f_eq_half_l749_749916

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2
  else Real.log x / Real.log 2

theorem solution_set_of_f_eq_half : 
  {x : ℝ | f x = 1 / 2} = {Real.sqrt 2, -Real.sqrt 2 / 2} := 
by
  sorry

end solution_set_of_f_eq_half_l749_749916


namespace AY_not_eq_BX_l749_749472

section RightTrapezoid

variables {A B C D X Y T : Point}
variables (AB CD : Line)
variables (CT_circle TD_circle : Circle)
variables (M N : Point)

-- Definitions and Conditions
def RightTrapezoid (A B C D : Point) (AB CD : Line) := true -- Formal definition omitted for simplicity.
def OnLine (P Q : Point) (L : Line) := LineContains L P ∧ LineContains L Q
def PointOnCircle (P : Point) (C : Circle) := CircleContains C P
def TouchesLateral (L : Line) (C : Circle) := true -- Represents that L touches C.

-- Assuming the conditions of the problem in the context.
axiom TrapezoidABCD : RightTrapezoid A B C D AB CD
axiom PointT_on_CD : OnLine T CD
axiom CirclesWithDiameters : PointOnCircle C CT_circle ∧ PointOnCircle T CT_circle ∧ PointOnCircle T TD_circle ∧ PointOnCircle D TD_circle
axiom CirclesTouchAB : TouchesLateral AB CT_circle ∧ TouchesLateral AB TD_circle
axiom CirclesTouchPoints : PointOnCircle X CT_circle ∧ PointOnCircle Y TD_circle

theorem AY_not_eq_BX :
  ¬ (distance A Y = distance B X) :=
sorry

end RightTrapezoid

end AY_not_eq_BX_l749_749472


namespace solution_l749_749327

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)

def has_period_pi (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x - π) = f(x)

def candidate1 (x : ℝ) : ℝ := Real.sin x
def candidate2 (x : ℝ) : ℝ := Real.sin (2 * x)
def candidate3 (x : ℝ) : ℝ := Real.cos x
def candidate4 (x : ℝ) : ℝ := Real.cos (2 * x)

theorem solution :
  (is_even candidate4 ∧ has_period_pi candidate4) ∧
  ¬ (is_even candidate1 ∧ has_period_pi candidate1) ∧
  ¬ (is_even candidate2 ∧ has_period_pi candidate2) ∧
  ¬ (is_even candidate3 ∧ has_period_pi candidate3) :=
by
  sorry

end solution_l749_749327


namespace classify_numbers_l749_749342

def a : ℤ := -2
def b : ℚ := 3 / 10
def c : ℤ := 0
def d : ℝ := Real.sqrt 7
def e : ℝ := 0.3030030003 -- Assume this is defined appropriately in Lean
def f : ℚ := -3 / 7

theorem classify_numbers :
  (Int √(-8)) ≡ a ∧ 
  (Int c) ∧ 
  (Rat b) ∧ 
  (Rat f) ∧ 
  (Irr d) ∧ 
  (Irr e) :=
by
  sorry

end classify_numbers_l749_749342


namespace exactly_one_pit_no_replanting_l749_749247

noncomputable def pit_prob : ℚ := 1/4
noncomputable def no_replanting_prob : ℚ := 1 - pit_prob

theorem exactly_one_pit_no_replanting :
  (3.choose 1) * no_replanting_prob * (pit_prob ^ 2) = 9/64 := by
  sorry

end exactly_one_pit_no_replanting_l749_749247


namespace part1_monotonicity_part2_find_range_l749_749940

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end part1_monotonicity_part2_find_range_l749_749940


namespace count_valid_numbers_l749_749096

def is_positive (n : ℕ) : Prop := n > 0
def is_less_than_10000 (n : ℕ) : Prop := n < 10000
def has_at_most_three_digits (n : ℕ) : Prop := (n.to_digits.length ≤ 3)

def valid_number (n : ℕ) : Prop := 
  is_positive n ∧ is_less_than_10000 n ∧ has_at_most_three_digits n

theorem count_valid_numbers : (finset.range 10000).filter (λ n, valid_number n).card = 3231 := by 
  sorry -- Proof to be filled.

end count_valid_numbers_l749_749096


namespace part_1_part_2_l749_749854

def f (x a : ℝ) : ℝ := |x - a| + 5 * x

theorem part_1 (x : ℝ) : (|x + 1| + 5 * x ≤ 5 * x + 3) ↔ (x ∈ Set.Icc (-4 : ℝ) 2) :=
by
  sorry

theorem part_2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) :=
by
  sorry

end part_1_part_2_l749_749854


namespace green_or_yellow_probability_l749_749609

-- Given the number of marbles of each color
def green_marbles : ℕ := 4
def yellow_marbles : ℕ := 3
def white_marbles : ℕ := 6

-- The total number of marbles
def total_marbles : ℕ := green_marbles + yellow_marbles + white_marbles

-- The number of favorable outcomes (green or yellow marbles)
def favorable_marbles : ℕ := green_marbles + yellow_marbles

-- The probability of drawing a green or yellow marble as a fraction
def probability_of_green_or_yellow : Rat := favorable_marbles / total_marbles

theorem green_or_yellow_probability :
  probability_of_green_or_yellow = 7 / 13 :=
by
  sorry

end green_or_yellow_probability_l749_749609


namespace a6_is_sqrt5_to_6_b6_l749_749841

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := (7/4) * a n + (9/4) * real.sqrt (5^n - (a n)^2)

def b (n : ℕ) : ℝ := a n / real.sqrt (5^n)

theorem a6_is_sqrt5_to_6_b6 : 
  a 6 = real.sqrt 5 ^ 6 * b 6 := 
sorry

end a6_is_sqrt5_to_6_b6_l749_749841


namespace find_lambda_l749_749052

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749052


namespace smallest_int_with_18_divisors_l749_749661

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749661


namespace lambda_solution_l749_749060

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749060


namespace pages_left_to_read_correct_l749_749279

def total_pages : Nat := 563
def pages_read : Nat := 147
def pages_left_to_read : Nat := 416

theorem pages_left_to_read_correct : total_pages - pages_read = pages_left_to_read := by
  sorry

end pages_left_to_read_correct_l749_749279


namespace sequence_general_term_l749_749579

theorem sequence_general_term (n : ℕ) : 
  let a := λ n : ℕ, n + 1 / 2^n in
  a n = n + 1 / 2^n := by
  sorry

end sequence_general_term_l749_749579


namespace positive_integers_count_count_positive_integers_l749_749387

open Nat

theorem positive_integers_count (x : ℕ) :
  (225 ≤ x * x ∧ x * x ≤ 400) ↔ (x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20) :=
by sorry

theorem count_positive_integers : 
  Finset.card { x : ℕ | 225 ≤ x * x ∧ x * x ≤ 400 } = 6 :=
by
  sorry

end positive_integers_count_count_positive_integers_l749_749387


namespace smallest_area_of_2020th_square_l749_749769

theorem smallest_area_of_2020th_square (n : ℕ) (A : ℕ) :
  (∃ m : ℕ, (n ∈ ℕ) ∧ (A ∈ ℕ) ∧ n^2 = 2019 + A ∧ n^2 - 2019 = m^2 ∧ A ≠ 1 ∧ m^2 = (n - m) * (n + m) ∧ (n - m = 3 ∨ n - m = 1 ∨ n + m = 3 ∨ n + m = 673)) → 
  A = 112225 :=
begin
  sorry
end

end smallest_area_of_2020th_square_l749_749769


namespace arcsin_half_eq_pi_six_l749_749808

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l749_749808


namespace part_one_part_two_l749_749283

variable (a b : ℝ) (C : ℝ)

def triangle_area (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

theorem part_one (hC : Real.sin C ≤ 1) : 
  triangle_area a b C ≤ 0.5 * (a^2 - a * b + b^2) :=
by 
  sorry

theorem part_two (hC : Real.sin C ≤ 1) : 
  triangle_area a b C ≤ ( (a + b) / (2 * Real.sqrt 2) )^2 :=
by
  sorry

end part_one_part_two_l749_749283


namespace problem_l749_749900

variables {a b c : ℝ}

noncomputable def f (x : ℝ) : ℝ := |x + a| + |x - b| + c

theorem problem (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) 
    (h_min_f : ∀ x, f(x) ≥ 2): (a + b + c = 2) ∧ (Real.sqrt (1 / a) + Real.sqrt (1 / b) + Real.sqrt (1 / c)) = 9 / 2 := 
begin
  sorry,
end

end problem_l749_749900


namespace pizza_topping_combinations_l749_749304

-- Given condition
def num_toppings : Nat := 8

-- Problem statement
theorem pizza_topping_combinations :
  ∑ k in {1, 2, 3}, Nat.choose num_toppings k = 92 := 
  by
  sorry

end pizza_topping_combinations_l749_749304


namespace lambda_solution_l749_749064

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749064


namespace min_a2_plus_2b2_min_4_over_a_minus_b_plus_1_over_2b_l749_749903

noncomputable def min_value_1 (a b : ℝ) (h1: a > b) (h2: b > 0) (h3: a + b = 1) : ℝ := a^2 + 2 * (b^2)
noncomputable def min_value_2 (a b : ℝ) (h1: a > b) (h2: b > 0) (h3: a + b = 1) : ℝ := 4 / (a - b) + 1 / (2 * b)

theorem min_a2_plus_2b2 : Inf {x : ℝ | ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a + b = 1 ∧ x = a^2 + 2 * (b^2)} = 2 / 3 := sorry

theorem min_4_over_a_minus_b_plus_1_over_2b : Inf {x : ℝ | ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a + b = 1 ∧ x = 4 / (a - b) + 1 / (2 * b)} = 9 := sorry

end min_a2_plus_2b2_min_4_over_a_minus_b_plus_1_over_2b_l749_749903


namespace triangle_congruence_l749_749201

theorem triangle_congruence (A B C D O : Type) 
  [inst : Inhabited A] [inhB : Inhabited B] [inhC : Inhabited C] [inhD : Inhabited D] [inhO : Inhabited O]
  (AB_CD_intersect_O : A ≠ B ∧ C ≠ D ∧ (∃ (O : Type), O ∈ LineThrough A B ∧ O ∈ LineThrough C D))
  (angle_ACO_EQ_angle_DBO : ∃ (a b : ℝ), ∠ACO = a ∧ ∠DBO = b ∧ a = b)
  (BO_EQ_OC : ∃ (o : ℝ), BO = o ∧ OC = o) : 
  ∃ (tri_ACO tri_DBO : Triangle), tri_ACO ≅ tri_DBO :=
sorry

end triangle_congruence_l749_749201


namespace zeros_of_f_on_interval_l749_749963

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem zeros_of_f_on_interval : ∃ (S : Set ℝ), S ⊆ (Set.Ioo 0 1) ∧ S.Infinite ∧ ∀ x ∈ S, f x = 0 := by
  sorry

end zeros_of_f_on_interval_l749_749963


namespace problem1_problem2_problem3_l749_749209

-- Problem 1
theorem problem1 (x : ℝ) : (3 * (x - 1)^2 = 12) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (3 * x^2 - 6 * x - 2 = 0) ↔ (x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (3 * x * (2 * x + 1) = 4 * x + 2) ↔ (x = -1 / 2 ∨ x = 2 / 3) :=
by
  sorry

end problem1_problem2_problem3_l749_749209


namespace store_price_reduction_l749_749741

theorem store_price_reduction 
    (initial_price : ℝ) (initial_sales : ℕ) (price_reduction : ℝ)
    (sales_increase_factor : ℝ) (target_profit : ℝ)
    (x : ℝ) : (initial_price, initial_price - price_reduction, x) = (80, 50, 12) →
    sales_increase_factor = 20 →
    target_profit = 7920 →
    (30 - x) * (200 + sales_increase_factor * x / 2) = 7920 →
    x = 12 ∧ (initial_price - x) = 68 :=
by 
    intros h₁ h₂ h₃ h₄
    sorry

end store_price_reduction_l749_749741


namespace midpoint_in_polar_coordinates_l749_749994

-- Define the problem as a theorem in Lean 4
theorem midpoint_in_polar_coordinates :
  let A := (10, Real.pi / 4)
  let B := (10, 3 * Real.pi / 4)
  ∃ r θ, (r = 5 * Real.sqrt 2) ∧ (θ = Real.pi / 2) ∧
         0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

end midpoint_in_polar_coordinates_l749_749994


namespace angle_quadrant_l749_749452

theorem angle_quadrant 
  (θ : Real) 
  (h1 : Real.cos θ > 0) 
  (h2 : Real.sin (2 * θ) < 0) : 
  3 * π / 2 < θ ∧ θ < 2 * π := 
by
  sorry

end angle_quadrant_l749_749452


namespace units_digit_of_fraction_l749_749264

theorem units_digit_of_fraction :
  ((30 * 31 * 32 * 33 * 34) / 400) % 10 = 4 :=
by
  sorry

end units_digit_of_fraction_l749_749264


namespace johnny_yellow_picks_l749_749143

variable (total_picks red_picks blue_picks yellow_picks : ℕ)

theorem johnny_yellow_picks
    (h_total_picks : total_picks = 3 * blue_picks)
    (h_half_red_picks : red_picks = total_picks / 2)
    (h_blue_picks : blue_picks = 12)
    (h_pick_sum : total_picks = red_picks + blue_picks + yellow_picks) :
    yellow_picks = 6 := by
  sorry

end johnny_yellow_picks_l749_749143


namespace circle_line_intersection_range_k_l749_749421

theorem circle_line_intersection_range_k :
  let C := {p : ℝ × ℝ | p.1^2 + p.2^2 + 4 * p.1 + 3 = 0} in
  ∃ k : ℝ, (-4/3 ≤ k ∧ k ≤ 0) ∧
  (∃ p : ℝ × ℝ, p ∈ C ∧ (∃ q : ℝ × ℝ, (q.snd = k * q.fst - 1)
  ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1)) :=
sorry

end circle_line_intersection_range_k_l749_749421


namespace calc_WZ_length_l749_749470

-- Variables and constants
variables (O C D W Z : Type) [elementary_geometry.has_circle O]

-- Circle radius
constant r : ℝ := 10

-- Angle COD is 45 degrees
constant angle_COD : ℝ := 45

-- CZ is perpendicular to CD intersecting at W
constant perp_CZ_CD : elementary_geometry.perpendicular C Z CD

-- Length of WZ
theorem calc_WZ_length (h : elementary_geometry.has_angle O C D angle_COD)
  (h1 : elementary_geometry.is_radius O C r)
  (h2 : elementary_geometry.is_radius O D r)
  (h3 : elementary_geometry.perpendicular C Z CD.and elementary_geometry.intersect_at C Z CD W)
  : elementary_geometry.distance_between_points W Z = 3.82 := 
sorry

end calc_WZ_length_l749_749470


namespace sequence_sum_proof_l749_749127

-- We define the arithmetic sequence and its sum as given conditions in the problem
variables {a : ℕ → ℕ} {S : ℕ → ℕ}

-- Condition: The sum of an arithmetic sequence \( S \)
def sum_arithmetic_sequence (n : ℕ) : Prop :=
  S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

-- Condition: The difference in sums given in the problem
def sum_diff_condition : Prop :=
  S 16 - S 5 = 165

-- Main theorem: Prove, given the conditions above, that \( a_9 + a_8 + a_{16} = 45 \)
theorem sequence_sum_proof : 
  (∀ n, sum_arithmetic_sequence n) → sum_diff_condition → (a 9 + a 8 + a 16 = 45) :=
by
  intros
  sorry

end sequence_sum_proof_l749_749127


namespace triangle_trig_identity_sin_triangle_trig_identity_cos_l749_749140

-- Problem statements with the given conditions

theorem triangle_trig_identity_sin (A B C : ℝ) (h : A + B + C = π) :
  sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 2 * (1 + cos A * cos B * cos C) :=
sorry

theorem triangle_trig_identity_cos (A B C : ℝ) (h : A + B + C = π) :
  cos A ^ 2 + cos B ^ 2 + cos C ^ 2 = 1 - 2 * cos A * cos B * cos C :=
sorry

end triangle_trig_identity_sin_triangle_trig_identity_cos_l749_749140


namespace find_lambda_l749_749087

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749087


namespace escalator_length_is_120_l749_749190

variable (x : ℝ) -- Speed of escalator in steps/unit time.

constant steps_while_ascending : ℕ := 75
constant steps_while_descending : ℕ := 150
constant speed_ascending : ℝ := 1.0
constant speed_descending : ℝ := 3.0
constant walking_speed_ratio : ℝ := 3.0

theorem escalator_length_is_120 :
  let t_ascending := (steps_while_ascending / (speed_ascending + x))
      t_descending := (steps_while_descending / (speed_descending - x)) * (1 / walking_speed_ratio) in
  steps_while_ascending * (speed_ascending + x) = steps_while_descending * (speed_descending - x) / walking_speed_ratio →
  75 * (1 + 0.6) = 120 :=
by
  intro h
  sorry

end escalator_length_is_120_l749_749190


namespace am_value_l749_749923

noncomputable def f (x : ℝ) (m a : ℝ) : ℝ := x^m - a * x

theorem am_value :
  ∀ (m a : ℝ), deriv (λ x : ℝ, f x m a) = (λ x : ℝ, 2 * x + 1) → a * m = -2 :=
by
  intros m a h
  sorry

end am_value_l749_749923


namespace sum_of_roots_equation_l749_749220

noncomputable def sum_of_roots (a b c : ℝ) : ℝ :=
  (-b) / a

theorem sum_of_roots_equation :
  let a := 3
  let b := -15
  let c := 20
  sum_of_roots a b c = 5 := 
  by {
    sorry
  }

end sum_of_roots_equation_l749_749220


namespace min_circles_l749_749476

noncomputable def segments_intersecting_circles (N : ℕ) : Prop :=
  ∀ seg : (ℝ × ℝ) × ℝ, (seg.fst.fst ≥ 0 ∧ seg.fst.fst + seg.snd ≤ 100 ∧ seg.fst.snd ≥ 0 ∧ seg.fst.snd ≤ 100 ∧ seg.snd = 10) →
    ∃ c : ℝ × ℝ, (dist c seg.fst < 1 ∧ c.fst ≥ 0 ∧ c.fst ≤ 100 ∧ c.snd ≥ 0 ∧ c.snd ≤ 100) 

theorem min_circles (N : ℕ) (h : segments_intersecting_circles N) : N ≥ 400 :=
sorry

end min_circles_l749_749476


namespace find_max_value_l749_749926

noncomputable def f (x θ : ℝ) : ℝ := Real.cos (2 * x + θ)

theorem find_max_value (θ : ℝ) (hθ : |θ| < π / 2) : 
  let x := π / 16 in
  -3 * π / 8 <= x ∧ x <= -π / 6 →
  ∃ M, M = 1 ∧ ∀ y, f y θ ≤ M := 
begin
  sorry
end

end find_max_value_l749_749926


namespace smallest_integer_with_18_divisors_l749_749699

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749699


namespace lim_condition_expectation_F_l749_749277

variables {Ω : Type*} {F : Type*} [MeasureTheory.MeasureSpace Ω] [MeasureTheory.ProbabilitySpace Ω]

noncomputable theory

open MeasureTheory

-- Definitions of σ-subalgebras as sequences
variables {𝓔 𝓕 𝓖 : ℕ → MeasureTheory.Measure ℕ}

-- Random variable ξ such that E[ξ^2] < ∞
variable {ξ : Ω → ℝ}
variable (h1 : Integrable ξ) -- This implies E[ξ^2] < ∞

-- σ-algebra sequences condition
variable (h2 : ∀ n, 𝓔 n ≤ 𝓕 n ∧ 𝓕 n ≤ 𝓖 n)

-- Limit conditions
variable (h3 : ∀ᵐ w ∂(MeasureTheory.ProbabilityMeasure), ∀ n, ConditionalExpectation (𝓔 n) ξ w = η w)
variable (h4 : ∀ᵐ w ∂(MeasureTheory.ProbabilityMeasure), ∀ n, ConditionalExpectation (𝓖 n) ξ w = η w)

-- The theorem to prove
theorem lim_condition_expectation_F {η : Ω → ℝ}
  (h3 : ∀ᵐ w ∂(MeasureTheory.ProbabilityMeasure), ∀ n, ConditionalExpectation (𝓔 n) ξ w = η w)
  (h4 : ∀ᵐ w ∂(MeasureTheory.ProbabilityMeasure), ∀ n, ConditionalExpectation (𝓖 n) ξ w = η w)
  : ∀ᵐ w ∂(MeasureTheory.ProbabilityMeasure), ∀ n, ConditionalExpectation (𝓕 n) ξ w = η w :=
sorry

end lim_condition_expectation_F_l749_749277


namespace largest_circle_area_l749_749763

theorem largest_circle_area (w l: ℝ) (h_w : 2 * w = l) (h_area : w * l = 200) : 
  let perimeter := 2 * (w + l) in
  let r := perimeter / (2 * Real.pi) in
  let circle_area := Real.pi * r ^ 2 in
  Int.nearest_ne (circle_area / Real.pi) = 287 :=
by
  -- Definitions and conditions
  let w := Real.sqrt 100
  let l := 2 * w
  have h_w : 2 * w = l := by sorry
  have h_area : w * l = 200 := by sorry
  -- Calculations
  let perimeter := 2 * (w + l)
  let r := perimeter / (2 * Real.pi)
  let circle_area := Real.pi * r ^ 2
  -- Prove the final area equals approximately 287
  sorry

end largest_circle_area_l749_749763


namespace focal_length_asymptote_l749_749222

def hyperbola_asymptotes (b : ℝ) (hb : b > 0) : Prop :=
  (differentiable_at ℝ (λ x : ℝ, x / 2)) ∧ 
  (differentiable_at ℝ (λ x : ℝ, -x / 2))

theorem focal_length_asymptote :
  ∀ (b : ℝ) (hb : b > 0), (4 + b^2 = 9) → hyperbola_asymptotes b hb :=
by
  intro b hb h
  sorry

end focal_length_asymptote_l749_749222


namespace find_lambda_l749_749018

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749018


namespace z_on_ok_and_ratio_l749_749504

open EuclideanGeometry

variables {A B C O Z : Point}
variables {R r : ℝ}
variables (K : Point)

/-- Let O be the center and R be the radius of the circumcircle of triangle ABC, 
    Z be the center and r be the radius of the incircle of triangle ABC, 
    and K be the centroid of the triangle formed by the points of tangency 
    of the incircle with the sides of triangle ABC.
    Then Z lies on segment OK and OZ: ZK = 3R: r. -/
theorem z_on_ok_and_ratio (hO : circumcenter O A B C R)
                          (hZ : incirclecenter Z A B C r)
                          (hK : is_centroid K (tangency_triangle A B C)) :
  collinear3 O Z K ∧ dist O Z / dist Z K = 3 * R / r :=
sorry

end z_on_ok_and_ratio_l749_749504


namespace compute_expression_l749_749833

theorem compute_expression : (-3) * 2 + 4 = -2 := 
by
  sorry

end compute_expression_l749_749833


namespace lambda_value_l749_749029

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749029


namespace intersection_A_B_eq_C_l749_749895

def A : Set ℝ := {1, 3, 5, 7}
def B : Set ℝ := {x | -x^2 + 4 * x ≥ 0}
def C : Set ℝ := {1, 3}

theorem intersection_A_B_eq_C : A ∩ B = C := 
by sorry

end intersection_A_B_eq_C_l749_749895


namespace each_dog_food_intake_l749_749855

theorem each_dog_food_intake (total_food : ℝ) (dog_count : ℕ) (equal_amount : ℝ) : total_food = 0.25 → dog_count = 2 → (total_food / dog_count) = equal_amount → equal_amount = 0.125 :=
by
  intros h1 h2 h3
  sorry

end each_dog_food_intake_l749_749855


namespace circles_intersect_at_single_point_l749_749594

-- Defining a triangle with vertices A, B, and C
variables (A B C : Point)

-- Defining circles passing through pairs of vertices
def circle_O1 := {p : Point | p = A ∨ p = B}
def circle_O2 := {p : Point | p = B ∨ p = C}
def circle_O3 := {p : Point | p = C ∨ p = A}

-- Given angles for arcs subtended outside the triangle
variables (θ1 θ2 θ3 : ℝ)

-- Condition: Sum of the angles is 180 degrees
hypothesis h_angle_sum : θ1 + θ2 + θ3 = 180

-- To be proved statement: these circles intersect at a common point P
theorem circles_intersect_at_single_point :
  ∃ P : Point, P ∈ circle_O1 ∧ P ∈ circle_O2 ∧ P ∈ circle_O3 :=
sorry

end circles_intersect_at_single_point_l749_749594


namespace Perelmans_conjecture_name_l749_749467

theorem Perelmans_conjecture_name : 
  (∃ conjecture : String, 
    (conjecture = "The sphere is the only type of bounded three-dimensional surface without holes") ∧
    (proof_completed_by conjecture "Grigori Perelman" 2003) ∧
    (one_of_millennium_prize_problems conjecture) ∧
    (fields_medal_awarded conjecture "Grigori Perelman" 2006)) →
  (conjecture_name = "Poincaré Conjecture") :=
sorry

end Perelmans_conjecture_name_l749_749467


namespace arcsin_of_half_l749_749792

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l749_749792


namespace smallest_positive_integer_with_18_divisors_l749_749691

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749691


namespace lower_limit_of_total_people_l749_749858

noncomputable def total_people_under_21 (T : ℕ) : Prop := (3 / 7) * T = 33
def people_over_65 (T : ℕ) : Prop := ∃ k, (5 / 11) * T = k
def total_people_bounds (T : ℕ) : Prop := T < 100

theorem lower_limit_of_total_people (T : ℕ) :
  total_people_under_21 T ∧ people_over_65 T ∧ total_people_bounds T → T = 77 :=
by
  intros,
  sorry

end lower_limit_of_total_people_l749_749858


namespace find_lambda_l749_749015

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749015


namespace least_edges_after_operations_l749_749568

-- Define the complete graph and the operation
def complete_graph (n : ℕ) := {V : finset (fin n) // ∀ (x y : fin n), x ≠ y ↔ (x, y) ∈ V}

def operation (G : Type) [Graph G] (c : Cycle G) (e : Edge G) : Graph G := sorry 

-- Define the theorem statement
theorem least_edges_after_operations (n : ℕ) (h : n ≥ 4) :
  ∃ (G : Graph (complete_graph n)), 
    (∀ (G' : Graph (complete_graph n)), operation G c e G' → G'.numEdges ≥ n) :=
sorry

end least_edges_after_operations_l749_749568


namespace max_sum_recip_distances_l749_749136

section Problem

variable (θ α ρ1 ρ2 : ℝ)
variable (x y k ρ : ℝ)

/-- Given the parametric equations of the curve C -/
def parametric_eqn (θ : ℝ) : Prop :=
  x = 3 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ

/-- Given the line l : y = kx (x ≥ 0) -/
def line_eqn (k: ℝ) : Prop :=
  y = k * x ∧ x ≥ 0

/-- Polar equation of the curve C -/
def polar_eqn (ρ θ : ℝ) : Prop :=
  ρ^2 - 6 * ρ * Real.cos θ + 5 = 0

/-- Sum of reciprocals of distances from the origin to points A and B -/
noncomputable def reciprocal_distances_sum (ρ1 ρ2 α : ℝ) : ℝ :=
  (ρ1 + ρ2) / (ρ1 * ρ2)

/-- Polar line equation -/
def polar_line_eqn (θ α : ℝ) : Prop :=
  θ = α ∧ ρ ∈ Set.univ

theorem max_sum_recip_distances (h : cos α = 1) :
  ∃ θ, parametric_eqn θ →
  polar_eqn ρ θ →
  polar_line_eqn θ α →
  reciprocals_distances_sum ρ1 ρ2 α = 6 / 5 := sorry

end Problem

end max_sum_recip_distances_l749_749136


namespace hyperbola_asymptotes_l749_749431

theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) (h_ecc : ∀ c : ℝ, c = 2 * a) : 
∀ x : ℝ, (y = ±√3 * x) := by
  sorry

end hyperbola_asymptotes_l749_749431


namespace smallest_integer_with_18_divisors_l749_749635

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749635


namespace peter_remaining_money_l749_749179

theorem peter_remaining_money (initial_money : ℕ) 
                             (potato_cost_per_kilo : ℕ) (potato_kilos : ℕ)
                             (tomato_cost_per_kilo : ℕ) (tomato_kilos : ℕ)
                             (cucumber_cost_per_kilo : ℕ) (cucumber_kilos : ℕ)
                             (banana_cost_per_kilo : ℕ) (banana_kilos : ℕ) :
  initial_money = 500 →
  potato_cost_per_kilo = 2 → potato_kilos = 6 →
  tomato_cost_per_kilo = 3 → tomato_kilos = 9 →
  cucumber_cost_per_kilo = 4 → cucumber_kilos = 5 →
  banana_cost_per_kilo = 5 → banana_kilos = 3 →
  initial_money - (potato_cost_per_kilo * potato_kilos + 
                   tomato_cost_per_kilo * tomato_kilos +
                   cucumber_cost_per_kilo * cucumber_kilos +
                   banana_cost_per_kilo * banana_kilos) = 426 := by
  sorry

end peter_remaining_money_l749_749179


namespace find_lambda_l749_749044

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749044


namespace num_positive_integers_in_square_range_l749_749390

theorem num_positive_integers_in_square_range :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), 225 ≤ x^2 ∧ x^2 ≤ 400 → (x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20) :=
by
  existsi 6
  split
  sorry

end num_positive_integers_in_square_range_l749_749390


namespace sequence_exists_l749_749540

def is_prime_power (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), (Nat.prime p) ∧ (n = p ^ k)

noncomputable def exists_infinite_sequence : Prop :=
  ∃ (a : ℕ → ℕ), ∀ (i : ℕ), 0 < i → 
  (a (i + 1) % a i = 0) ∧
  (a i % 3 ≠ 0) ∧
  (2^(i + 2) ∣ a i) ∧ ¬(2^(i + 3) ∣ a i) ∧
  (is_prime_power (6 * a i + 1)) ∧
  (∃ (x y : ℕ), a i = x^2 + y^2)

-- A proof would be provided here
theorem sequence_exists : exists_infinite_sequence :=
by sorry

end sequence_exists_l749_749540


namespace new_cost_percent_l749_749724

variables (t b : ℝ)

theorem new_cost_percent (t b : ℝ) : 
  let C := t * b^4 in
  let E := t * (2 * b)^4 in
  (E / C) * 100 = 1600 := by
  let C := t * b^4
  let E := t * (2 * b)^4
  sorry

end new_cost_percent_l749_749724


namespace geometric_sequence_seventh_term_l749_749754

-- Define the initial conditions
def geometric_sequence_first_term := 3
def geometric_sequence_fifth_term (r : ℝ) := geometric_sequence_first_term * r^4 = 243

-- Statement for the seventh term problem
theorem geometric_sequence_seventh_term (r : ℝ) 
  (h1 : geometric_sequence_first_term = 3) 
  (h2 : geometric_sequence_fifth_term r) : 
  3 * r^6 = 2187 :=
sorry

end geometric_sequence_seventh_term_l749_749754


namespace find_lambda_l749_749088

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749088


namespace lambda_value_l749_749038

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749038


namespace arcsin_one_half_eq_pi_six_l749_749814

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l749_749814


namespace length_PF1_l749_749777

open Real

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

def a : ℝ := 2
def b : ℝ := 1
def c : ℝ := sqrt (a^2 - b^2)

noncomputable def F1 : (ℝ×ℝ) := (-c, 0)
noncomputable def F2 : (ℝ×ℝ) := (c, 0)

noncomputable def P (y : ℝ) : (ℝ × ℝ) := (sqrt 3, y)

theorem length_PF1 :
  ∃ y : ℝ, ellipse (sqrt 3) y ∧
  ∃ PF2 : ℝ, PF2 = abs (P y).snd ∧
  abs (dist (P y) F1) = 2 * a - PF2 :=
sorry

end length_PF1_l749_749777


namespace angle_determines_magnitude_of_a_l749_749525

variables {𝕜 : Type*} [IsROrC 𝕜] {V : Type*} [InnerProductSpace 𝕜 V]

-- Definitions from conditions
variables {a b : V} (θ : Real.Angle) [NonZero a] [NonZero b]

-- Given Conditions
axiom condition_min_val : ∀ t : 𝕜, ∥a + t • b∥ ≥ 1

-- The theorem to prove
theorem angle_determines_magnitude_of_a 
  (hθ : θ = Real.Angle.ofVectorAngle a b) :
  ∥a∥ = 1 / Real.sin θ := sorry

end angle_determines_magnitude_of_a_l749_749525


namespace percent_carnations_l749_749294

theorem percent_carnations
  (pink_roses_ratio : ℚ)
  (yellow_orchids_ratio : ℚ)
  (pink_or_yellow_ratio : ℚ)
  (total_ratio : ℚ) :
  pink_roses_ratio = 1/5 →
  yellow_orchids_ratio = 4/5 →
  pink_or_yellow_ratio = 7/10 →
  total_ratio = 3/10 →
  total_ratio * 100 = 30 := 
by
  intros h1 h2 h3 h4
  rw h4
  exact eq.refl (30 : ℚ)

end percent_carnations_l749_749294


namespace peter_remaining_money_l749_749181

theorem peter_remaining_money (initial_money : ℕ) 
                             (potato_cost_per_kilo : ℕ) (potato_kilos : ℕ)
                             (tomato_cost_per_kilo : ℕ) (tomato_kilos : ℕ)
                             (cucumber_cost_per_kilo : ℕ) (cucumber_kilos : ℕ)
                             (banana_cost_per_kilo : ℕ) (banana_kilos : ℕ) :
  initial_money = 500 →
  potato_cost_per_kilo = 2 → potato_kilos = 6 →
  tomato_cost_per_kilo = 3 → tomato_kilos = 9 →
  cucumber_cost_per_kilo = 4 → cucumber_kilos = 5 →
  banana_cost_per_kilo = 5 → banana_kilos = 3 →
  initial_money - (potato_cost_per_kilo * potato_kilos + 
                   tomato_cost_per_kilo * tomato_kilos +
                   cucumber_cost_per_kilo * cucumber_kilos +
                   banana_cost_per_kilo * banana_kilos) = 426 := by
  sorry

end peter_remaining_money_l749_749181


namespace circle_equation_l749_749244

theorem circle_equation (x y : ℝ) :
  (∀ (C P : ℝ × ℝ), C = (8, -3) ∧ P = (5, 1) →
    ∃ R : ℝ, (x - 8)^2 + (y + 3)^2 = R^2 ∧ R^2 = 25) :=
sorry

end circle_equation_l749_749244


namespace probability_is_correct_l749_749376

/-- Representation of a standard six-sided die. -/
def is_standard_die (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6

/-- Number of ways to roll five standard dice such that the product of their values is even. -/
def even_product_configuration : Finset (Fin 6 → ℕ) :=
  (Finset.univ.filter (λ v, ∃ i, is_standard_die (v i) ∧ v i % 2 = 0))

/-- Number of ways to roll five standard dice such that the sum of their values is even and product is even. -/
def even_sum_given_even_product_configuration : Finset (Fin 6 → ℕ) :=
  even_product_configuration.filter (λ v, (Finset.univ.sum (λ i, v i)) % 2 = 0)

/-- The number of valid outcomes where the product of dice values is even. -/
def valid_even_product_outcomes : ℕ := 6^5 - 3^5

/-- The number of valid outcomes where the sum of dice values is even given that their product is even. -/
def valid_even_sum_given_even_product_outcomes : ℕ :=
  even_sum_given_even_product_configuration.card

/-- The probability that the sum of the dice values is even given that the product of their values is even. -/
def probability_even_sum_given_even_product : ℚ :=
  valid_even_sum_given_even_product_outcomes / valid_even_product_outcomes

theorem probability_is_correct :
  probability_even_sum_given_even_product = 1296 / 2511 := 
sorry

end probability_is_correct_l749_749376


namespace average_percentage_for_all_students_l749_749726

-- Definitions of the variables
def students1 : Nat := 15
def average1 : Nat := 75
def students2 : Nat := 10
def average2 : Nat := 90
def total_students : Nat := students1 + students2
def total_percentage1 : Nat := students1 * average1
def total_percentage2 : Nat := students2 * average2
def total_percentage : Nat := total_percentage1 + total_percentage2

-- Main theorem stating the average percentage for all students.
theorem average_percentage_for_all_students :
  total_percentage / total_students = 81 := by
  sorry

end average_percentage_for_all_students_l749_749726


namespace sum_of_n_values_abs_eq_4_l749_749709

theorem sum_of_n_values_abs_eq_4 :
  (∀ n : ℚ, abs (3 * n - 8) = 4 → n = 4 ∨ n = 4 / 3) →
  ((∑ x in ({4, 4/3}: finset ℚ), x) = 16 / 3) :=
by
  intro h
  sorry

end sum_of_n_values_abs_eq_4_l749_749709


namespace range_of_a_l749_749980

theorem range_of_a (a : ℝ) (h : ∀ x, 1 < x → |a * x + 1| > 2) : a ∈ [1, +∞) ∪ (-∞, -3] :=
sorry

end range_of_a_l749_749980


namespace path_length_abc_div_4_l749_749116

noncomputable def totalPathLength (AB BC : ℕ) : ℝ :=
  let AC := Real.sqrt (AB^2 + BC^2)
  let S1 := AC / (1 - 1/2)
  let S2 := Real.sqrt ((BC / 2)^2 + AB^2) / (1 - 1/2)
  S1 + S2

theorem path_length_abc_div_4 (AB BC : ℕ) (hAB : AB = 5) (hBC : BC = 12) :
  ∃ a b c : ℕ, let S := totalPathLength AB BC in 
  S = a + b * Real.sqrt c ∧ (a * b * c) / 4 = 793 :=
by
  have hAC : totalPathLength AB BC = 26 + 2 * Real.sqrt 61 :=
      sorry
  use 26, 2, 61
  split
  { exact hAC }
  { simp only [Nat.cast_mul, Rat.cast_div]
    norm_num }

end path_length_abc_div_4_l749_749116


namespace solution_set_of_inequality_l749_749398

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem solution_set_of_inequality :
  { x : ℝ | f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 } = 
    Ioo (1 / Real.exp 1) (Real.exp 1) := 
by
  sorry

end solution_set_of_inequality_l749_749398


namespace sin_log_infinite_zeros_in_01_l749_749964

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end sin_log_infinite_zeros_in_01_l749_749964


namespace exists_uv_for_interesting_triplet_l749_749260

noncomputable def interesting_triplet (a b c : ℕ) : Prop :=
  ((a^2 + 1) * (b^2 + 1)) % (c^2 + 1) = 0 ∧ (a^2 + 1) % (c^2 + 1) ≠ 0 ∧ (b^2 + 1) % (c^2 + 1) ≠ 0

theorem exists_uv_for_interesting_triplet (a b c : ℕ) (h : interesting_triplet a b c) :
  ∃ u v : ℕ, interesting_triplet u v c ∧ u * v < c^3 :=
begin
  sorry
end

end exists_uv_for_interesting_triplet_l749_749260


namespace problem_statement_l749_749732

theorem problem_statement (a : Fin k → ℝ) (u : ℕ → ℝ)
  (h : ∀ n, u (n + k) = ∑ i in Finset.range k, a i * u (n + k - i - 1)) :
  ∃ P_k_minus_1 : Polynomial ℝ, 
    (u 1 + ∑ i in Finset.range ∞, (u (i + 2) * (Polynomial.X ^ i))) =
      P_k_minus_1 / (1 - ∑ i in Finset.range k, a i * (Polynomial.X ^ (i + 1))) :=
by
  sorry

end problem_statement_l749_749732


namespace fraction_of_area_of_larger_square_covered_by_shaded_square_l749_749536

def side_length_of_unit_square_diagonal : ℝ := Real.sqrt 2

def area_of_shaded_square : ℝ := (side_length_of_unit_square_diagonal) ^ 2

def area_of_larger_square : ℝ := 6 ^ 2

def fraction_area_inside_shaded_square := area_of_shaded_square / area_of_larger_square

theorem fraction_of_area_of_larger_square_covered_by_shaded_square :
  fraction_area_inside_shaded_square = 1 / 18 := by
  sorry

end fraction_of_area_of_larger_square_covered_by_shaded_square_l749_749536


namespace stratified_sampling_A_l749_749305

theorem stratified_sampling_A (ratio_A ratio_B : ℕ) (sample_size : ℕ) 
  (h_ratio : ratio_A = 5) (h_ratio_B : ratio_B = 3) (h_sample_size : sample_size = 120) :
  (ratio_A * sample_size) / (ratio_A + ratio_B) = 75 :=
by
  rw [h_ratio, h_ratio_B, h_sample_size]
  norm_num
  sorry

end stratified_sampling_A_l749_749305


namespace num_seven_digit_numbers_l749_749097

theorem num_seven_digit_numbers (a b c d e f g : ℕ)
  (h1 : a * b * c = 30)
  (h2 : c * d * e = 7)
  (h3 : e * f * g = 15) :
  ∃ n : ℕ, n = 4 := 
sorry

end num_seven_digit_numbers_l749_749097


namespace smallest_integer_with_18_divisors_l749_749649

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749649


namespace minimum_value_of_f_l749_749979

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (x - 1) * (x + 2) * (x^2 + a * x + b)

theorem minimum_value_of_f : 
  ∀ (a b : ℝ),
  ( ∀ x : ℝ, f x a b = f (-x) a b ) → 
  ∃ x : ℝ, f x (-1) (-2) = -9 / 4 :=
by
  intro a b h_sym
  use [classical *] (-real.sqrt 10 / 2)
  unfold f
  have ha : a = -1, from sorry
  have hb : b = -2, from sorry
  simp [ha, hb]
  sorry

end minimum_value_of_f_l749_749979


namespace find_lambda_l749_749053

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749053


namespace lemming_average_distance_5_l749_749758

structure Rectangle :=
  (length : ℝ)
  (width : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

def lemming_distance_average (rect : Rectangle) (init_pos : Point) (d1 d2 d3 : ℝ) (move1 move2 : ℝ) : ℝ :=
  let diagonal := Real.sqrt (rect.length ^ 2 + rect.width ^ 2)
  let scaling_factor := d1 / diagonal
  let x1 := rect.length * scaling_factor
  let y1 := rect.width * scaling_factor
  let x2 := x1 + move2
  let y2 := y1
  let x_final := x2
  let y_final := y2 - move3
  let dist_left := x_final
  let dist_bottom := y_final
  let dist_right := rect.length - x_final
  let dist_top := rect.width - y_final
  (dist_left + dist_bottom + dist_right + dist_top) / 4

theorem lemming_average_distance_5 :
  lemming_distance_average {length := 12, width := 8} {x := 0, y := 0} 7.5 3 2 = 5 :=
by sorry

end lemming_average_distance_5_l749_749758


namespace find_lambda_l749_749078

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749078


namespace circumcenter_fixed_circle_l749_749439

-- Define the conditions under which the problem is stated.
variables {S1 S2 : Type} [metric_space S1] [metric_space S2]
variables (P Q : S1) (A1 B1 : S1) (A2 B2 : S2) (C : Type)
variables (circumcenter : S1 → S1 → S2 → S1) -- Assume a circumcenter function for triangles

-- Given conditions in Lean
variable (circle1 : S1)
variable (circle2 : S2)
variable (intersect_points : ∃ P Q, P ≠ Q ∧ P ∈ circle1 ∧ P ∈ circle2 ∧ Q ∈ circle1 ∧ Q ∈ circle2)
variable (distinct_A1B1 : A1 ≠ B1 ∧ A1 ∈ circle1 ∧ B1 ∈ circle1 ∧ A1 ≠ P ∧ A1 ≠ Q ∧ B1 ≠ Q ∧ B1 ≠ P)
variable (line_A1P : exists A2, (A1 ≠ A2) ∧ (A1 ≠ P) ∧ (A1 ∈ circle1) ∧ (A2 ∈ circle2) ∧ P ∈ S1)
variable (line_B1P : exists B2, (B1 ≠ B2) ∧ (B1 ≠ P) ∧ (B1 ∈ circle1) ∧ (B2 ∈ circle2) ∧ P ∈ S1)
variable (intersect_AB_C : exists C, (C ∈ S1) ∧ (C ∈ S2) ∧ (C ≠ A1) ∧ (C ≠ B1) ∧ (C ≠ A2) ∧ (C ≠ B2))

-- The statement to be proven
theorem circumcenter_fixed_circle :
  ∀ (A1 A2 C : S1), circumcenter A1 A2 C ∈ S1 := sorry

end circumcenter_fixed_circle_l749_749439


namespace max_integer_value_l749_749976

theorem max_integer_value (x : ℝ) : 
  ∃ M : ℤ, ∀ y : ℝ, (M = ⌊ 1 + 10 / (4 * y^2 + 12 * y + 9) ⌋ ∧ M ≤ 11) := 
sorry

end max_integer_value_l749_749976


namespace smallest_integer_with_18_divisors_l749_749634

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749634


namespace range_of_a_l749_749459

noncomputable def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def f (a x : ℝ) : ℝ := 2 * x^2 - 4 * (1 - a) * x + 1

theorem range_of_a (a : ℝ) :
  (is_increasing_on (f a) (set.Ici 3)) ↔ a ≥ -2 :=
sorry

end range_of_a_l749_749459


namespace ratio_of_cube_volumes_l749_749347

theorem ratio_of_cube_volumes (a b : ℕ) (ha : a = 10) (hb : b = 25) :
  (a^3 : ℚ) / (b^3 : ℚ) = 8 / 125 := by
  sorry

end ratio_of_cube_volumes_l749_749347


namespace smallest_integer_with_18_divisors_l749_749685

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749685


namespace smallest_integer_with_18_divisors_l749_749688

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749688


namespace smallest_n_for_terminating_fraction_l749_749626

theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), (n + 101).factorization.keys ⊆ {2, 5} ∧ ∀ m : ℕ, (m < n) → (¬(m + 101).factorization.keys ⊆ {2, 5}) :=
sorry

end smallest_n_for_terminating_fraction_l749_749626


namespace smallest_positive_integer_with_18_divisors_l749_749693

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749693


namespace companyB_cheaper_l749_749246

-- Define costs for Company A
def costA (x : ℝ) : ℝ :=
  if x ≤ 1 then 22 * x else 15 * x + 7

-- Define cost for Company B
def costB (x : ℝ) : ℝ := 16 * x + 3

-- Prove that Company B is cheaper than Company A for 0.5 < x < 4
theorem companyB_cheaper (x : ℝ) (hx1 : 0.5 < x) (hx2 : x < 4) : 
  costB x < costA x :=
by
  sorry

end companyB_cheaper_l749_749246


namespace total_flowers_sold_l749_749881

/-
Ginger owns a flower shop, where she sells roses, lilacs, and gardenias.
On Tuesday, she sold three times more roses than lilacs, and half as many gardenias as lilacs.
If she sold 10 lilacs, prove that the total number of flowers sold on Tuesday is 45.
-/

theorem total_flowers_sold
    (lilacs roses gardenias : ℕ)
    (h_lilacs : lilacs = 10)
    (h_roses : roses = 3 * lilacs)
    (h_gardenias : gardenias = lilacs / 2)
    (ht : lilacs + roses + gardenias = 45) :
    lilacs + roses + gardenias = 45 :=
by sorry

end total_flowers_sold_l749_749881


namespace find_natural_numbers_l749_749583

def LCM (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

theorem find_natural_numbers :
  ∃ a b : ℕ, a + b = 54 ∧ LCM a b - Nat.gcd a b = 114 ∧ (a = 24 ∧ b = 30 ∨ a = 30 ∧ b = 24) := by {
  sorry
}

end find_natural_numbers_l749_749583


namespace correct_statements_about_f_l749_749876

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem correct_statements_about_f : 
  (∀ x, (f x) ≤ (f e)) ∧ (f e = 1 / e) ∧ 
  (∀ x, (f x = 0) → x = 1) ∧ 
  (f 2 < f π ∧ f π < f 3) :=
by
  sorry

end correct_statements_about_f_l749_749876


namespace minimum_value_function_sequence_term_formula_find_angle_C_collinear_points_arithmetic_no_max_term_geometric_props_sequence_is_geometric_l749_749735

-- Problem 1
theorem minimum_value_function (x : ℝ) (h : x > 0) : (x + 4 / x ≥ 4) :=
sorry

-- Problem 2
theorem sequence_term_formula (n : ℕ) (h : n ≥ 1) : (a_n = 2n - 1) :=
sorry

-- Problem 3
theorem find_angle_C (a b c : ℝ) (S : ℝ) (h : S = (a^2 + b^2 - c^2) / 4) : C = 45 :=
sorry

-- Problem 4
-- Proposition ①
theorem collinear_points_arithmetic (a_n : ℕ → ℤ) (n : ℕ) 
(h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d) (S : ℕ → ℝ) 
(h_sums : ∀ n, S n = n * (a_n 1 + (n-1) * d / 2)) :
collinear [(10, S 10 / 10), (100, S 100 / 100), (110, S 110 / 110)] :=
sorry

-- Proposition ②
theorem no_max_term (a_n : ℕ → ℤ) (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d)
(a_1 : ℤ) (h_a1 : a_n 1 = -11) (h_a3a7 : a_n 3 + a_n 7 = -6) : 
∃ max (S : ℕ → ℝ), false :=
sorry

-- Proposition ③
theorem geometric_props (a_n : ℕ → ℤ) (S : ℕ → ℝ) (m : ℕ) (h_geometric : ∀ n, a_n (n + 1) = q * a_n n) 
(h_sum_fn : ∀ n, S n = ∑ k in range n, a_n k) (h_m : m > 0) :
is_geometric_seq [S m, S (2 * m) - S m, S (3 * m) - S (2 * m)] :=
sorry

-- Proposition ④
theorem sequence_is_geometric (a_n : ℕ → ℤ) (S : ℕ → ℝ) (a_1 : ℝ) (q : ℝ)
(h_sum_relation : ∀ n, S (n + 1) = a_1 + q * S n) (h_a1_ne0 : a_1 ≠ 0) (h_q_ne0 : q ≠ 0)
(h_geometric : ∀ n, a_n (n + 1) = q * a_n n) :
is_geometric_seq a_n :=
sorry

end minimum_value_function_sequence_term_formula_find_angle_C_collinear_points_arithmetic_no_max_term_geometric_props_sequence_is_geometric_l749_749735


namespace log_sin_x_eq_a_minus_half_log_1_b_2a_l749_749971

variable {b x a : Real}

theorem log_sin_x_eq_a_minus_half_log_1_b_2a (hb : b > 1) (htan : 0 < Real.tan x)
  (hlog : Real.log b (Real.tan x) = a) (hcos : 0 < Real.cos x) :
  Real.log b (Real.sin x) = a - 1 / 2 * Real.log b (1 + b^(2 * a)) := by
  sorry

end log_sin_x_eq_a_minus_half_log_1_b_2a_l749_749971


namespace range_of_a_l749_749919

-- Definitions from conditions
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- Given the conditions
theorem range_of_a (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 3) :
  (∀ x : ℝ, x ∈ set.Icc 1 a → f(x) ≥ f(a)) :=
by
  sorry

end range_of_a_l749_749919


namespace floor_m_plus_half_l749_749161

noncomputable def m : ℝ :=
  ∑ k in Finset.range 2009, ((k:ℕ) * 2^k / (k:ℕ)!)

theorem floor_m_plus_half :
  ⌊m + 1 / 2⌋ = 2 :=
sorry

end floor_m_plus_half_l749_749161


namespace arcsin_one_half_eq_pi_six_l749_749812

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l749_749812


namespace find_lambda_l749_749012

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749012


namespace total_items_for_children_l749_749851

theorem total_items_for_children :
  ∀ (children : ℕ) (pencils_per_child erasers_per_child skittles_per_child crayons_per_child : ℕ),
  pencils_per_child = 5 →
  erasers_per_child = 3 →
  skittles_per_child = 13 →
  crayons_per_child = 7 →
  children = 12 →
  (pencils_per_child * children = 60) ∧
  (erasers_per_child * children = 36) ∧
  (skittles_per_child * children = 156) ∧
  (crayons_per_child * children = 84) :=
by
  intros children pencils_per_child erasers_per_child skittles_per_child crayons_per_child
  repeat {intro}
  { split }
  { rfl }
  { split }
  { rfl }
  { split }
  { rfl }
  { rfl }

# where
# rfl is the reflexivity of equality: a proof that a term is equal to itself.

end total_items_for_children_l749_749851


namespace smallest_integer_with_18_divisors_l749_749627

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749627


namespace arcsin_one_half_l749_749795

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749795


namespace number_of_even_ones_matrices_l749_749836

noncomputable def count_even_ones_matrices (m n : ℕ) : ℕ :=
if m = 0 ∨ n = 0 then 1 else 2^((m-1)*(n-1))

theorem number_of_even_ones_matrices (m n : ℕ) : 
  count_even_ones_matrices m n = 2^((m-1)*(n-1)) := sorry

end number_of_even_ones_matrices_l749_749836


namespace tree_height_at_end_of_3_years_l749_749773

def tree_height (n : ℕ) : ℝ := 3^n 

theorem tree_height_at_end_of_3_years : tree_height 3 = 27 :=
by
  -- Assuming tree_height(5) is 243
  have h1 : tree_height 5 = 243 := by
    unfold tree_height
    norm_num
    
  -- Given initial height as 1 and growth rate triples each year.
  unfold tree_height
  norm_num
  sorry

end tree_height_at_end_of_3_years_l749_749773


namespace range_OA_OB_l749_749235

-- Let α be the parameter for curve C1
variable (α : ℝ)

-- Curve C1 parametric form
def curve_C1_parametric : ℝ × ℝ :=
(2 + 2 * Real.cos α, 2 * Real.sin α)

-- Curve C2 polar form
def curve_C2_polar (ρ θ : ℝ) := ρ * (Real.cos θ)^2 = Real.sin θ

-- Polar equation of curve C1
def curve_C1_polar_equation (θ : ℝ) : Prop :=
∃ ρ, curve_C1_parametric ρ θ = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ ρ = 4 * Real.cos θ

-- Cartesian equation of curve C2
def curve_C2_cartesian_equation (x y : ℝ) : Prop :=
y = x^2

-- Angle α constraint for the ray l
axiom α_cond : π / 6 < α ∧ α ≤ π / 4

-- Parametric equation of the ray l
def ray_l_parametric (t : ℝ) : ℝ × ℝ :=
(t * Real.cos α, t * Real.sin α)

-- Intersection of ray l with curve C1
def intersection_ray_C1 (t : ℝ) : Prop :=
let (x, y) := ray_l_parametric α t in (x - 2)^2 + y^2 = 4

-- Intersection of ray l with curve C2
def intersection_ray_C2 (t : ℝ) : Prop :=
let (x, y) := ray_l_parametric α t in y = x^2

-- Proof goal: Range of |OA| * |OB|
theorem range_OA_OB : ∀ α, α_cond α →
  let t₁ := 4 * Real.cos α
  let t₂ := Real.sin α / (Real.cos α)^2
  4 * Real.tan α ∈ Set.Icc (4 * (Real.sqrt 3 / 3)) 4 := sorry

end range_OA_OB_l749_749235


namespace probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l749_749337

def probability_of_at_least_one_head (p : ℚ) (n : ℕ) : ℚ := 
  1 - (1 - p)^n

theorem probability_of_at_least_one_head_in_three_tosses_is_7_over_8 :
  probability_of_at_least_one_head (1/2) 3 = 7/8 :=
by 
  sorry

end probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l749_749337


namespace unoccupied_volume_l749_749747

def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h 
def volume_sphere (r : ℝ) : ℝ := (4/3) * π * r^3 

theorem unoccupied_volume :
  let r_cylinder := 5
  let h_cylinder := 10
  let r_sphere := 1
  let num_spheres := 15
  let cyl_volume := volume_cylinder r_cylinder h_cylinder 
  let water_volume := 0.4 * cyl_volume
  let sphere_volume := volume_sphere r_sphere * num_spheres
  cyl_volume - (water_volume + sphere_volume) = 130 * π :=
by
  sorry

end unoccupied_volume_l749_749747


namespace largest_possible_average_l749_749531

noncomputable def ten_test_scores (a b c d e f g h i j : ℤ) : ℤ :=
  a + b + c + d + e + f + g + h + i + j

theorem largest_possible_average
  (a b c d e f g h i j : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 100)
  (h2 : 0 ≤ b ∧ b ≤ 100)
  (h3 : 0 ≤ c ∧ c ≤ 100)
  (h4 : 0 ≤ d ∧ d ≤ 100)
  (h5 : 0 ≤ e ∧ e ≤ 100)
  (h6 : 0 ≤ f ∧ f ≤ 100)
  (h7 : 0 ≤ g ∧ g ≤ 100)
  (h8 : 0 ≤ h ∧ h ≤ 100)
  (h9 : 0 ≤ i ∧ i ≤ 100)
  (h10 : 0 ≤ j ∧ j ≤ 100)
  (h11 : a + b + c + d ≤ 190)
  (h12 : b + c + d + e ≤ 190)
  (h13 : c + d + e + f ≤ 190)
  (h14 : d + e + f + g ≤ 190)
  (h15 : e + f + g + h ≤ 190)
  (h16 : f + g + h + i ≤ 190)
  (h17 : g + h + i + j ≤ 190)
  : ((ten_test_scores a b c d e f g h i j : ℚ) / 10) ≤ 44.33 := sorry

end largest_possible_average_l749_749531


namespace monotonicity_a1_range_of_a_l749_749938

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- 1. Monotonicity when \( a = 1 \)
theorem monotonicity_a1 :
  (∀ x > 0, (f x 1)' > 0) ∧ (∀ x < 0, (f x 1)' < 0) :=
by
  sorry

-- 2. Range of \( a \) for \( f(x) \geq \frac{1}{2} x^3 + 1 \) for \( x \geq 0 \)
theorem range_of_a (a : ℝ) (x : ℝ) (hx : x ≥ 0) (hf : f x a ≥ (1 / 2) * x^3 + 1) :
  a ≥ (7 - Real.exp 2) / 4 :=
by
  sorry

end monotonicity_a1_range_of_a_l749_749938


namespace smallest_positive_integer_l749_749835

-- Define the complex vertices
def vertex_1 (n : ℤ) : ℂ := n + complex.I
def vertex_2 (n : ℤ) : ℂ := (n + complex.I) ^ 2
def vertex_3 (n : ℤ) : ℂ := (n + complex.I) ^ 3

-- Area calculation using Shoelace theorem
def shoelace_area (n : ℤ) : ℚ :=
  let x1 := n : ℚ,
      y1 := 1,
      x2 := n^2 - 1,
      y2 := 2n,
      x3 := n^3 - 3n,
      y3 := 3n^2 - 1 in
  (1/2) * | 2n^4 - n^3 + 4n^2 - 4n + 1 |

-- Problem statement
theorem smallest_positive_integer (n : ℤ) (h₁ : n > 0) (h₂ : 2 * n^4 - n^3 + 4 * n^2 - 4 * n + 1 > 8060) : n = 10 :=
by
  sorry

end smallest_positive_integer_l749_749835


namespace find_t_from_trig_conditions_l749_749125

noncomputable theory
open Real

variable {θ t : ℝ}

def terminal_side_passes_through (θ t : ℝ) : Prop :=
  let sint := t / sqrt (4 + t^2)
  let cost := -2 / sqrt (4 + t^2)
  sint + cost = sqrt 5 / 5

theorem find_t_from_trig_conditions (h : terminal_side_passes_through θ t) : t = 4 :=
sorry

end find_t_from_trig_conditions_l749_749125


namespace Ramu_spent_on_repairs_l749_749541

theorem Ramu_spent_on_repairs (purchase_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : selling_price = 61900) 
  (h3 : profit_percent = 12.545454545454545) 
  (h4 : selling_price = purchase_price + R + (profit_percent / 100) * (purchase_price + R)) : 
  R = 13000 :=
by
  sorry

end Ramu_spent_on_repairs_l749_749541


namespace polar_eq_C1_intersection_line_eq_l749_749950

section Problem

-- Definitions of parametric equations of curve C1
def C1_x (theta : ℝ) : ℝ := 3 + 4 * Real.cos theta
def C1_y (theta : ℝ) : ℝ := 4 + 4 * Real.sin theta

-- Polar equation of curve C2
def C2_polar (theta : ℝ) : ℝ := 4 * Real.sin theta

-- Prove that the polar equation of C1 is as given
theorem polar_eq_C1 (rho phi : ℝ) :
  (rho^2 - 6 * rho * Real.cos phi - 8 * rho * Real.sin phi + 9 = 0) ↔
  (∃ theta, rho = Real.sqrt ((C1_x theta)^2 + (C1_y theta)^2) ∧ 
    phi = Real.atan2 (C1_y theta) (C1_x theta)) :=
sorry

-- Prove the polar equation of the line where C1 and C2 intersect
theorem intersection_line_eq (rho theta : ℝ) :
  (6 * rho * Real.cos theta + 4 * rho * Real.sin θ - 9 = 0) ↔
  (∃ (x y : ℝ), x = 3 + 4 * Real.cos theta ∧ y = 4 + 4 * Real.sin θ ∧ 
   x^2 + y^2 - 6 * x - 8 * y + 9 = 0 ∧ x^2 + y^2 - 4 * y = 0) :=
sorry

end Problem

end polar_eq_C1_intersection_line_eq_l749_749950


namespace average_cost_per_pencil_l749_749284

theorem average_cost_per_pencil :
  let pencils_cost_dollars := 30.75
  let shipping_cost_dollars := 8.25
  let discount := 0.10
  let pencils_count := 300
  let total_cost_dollars := (pencils_cost_dollars * (1 - discount)) + shipping_cost_dollars
  let total_cost_cents := total_cost_dollars * 100
  let average_cost_per_pencil_cents := total_cost_cents / pencils_count
  Nat.round average_cost_per_pencil_cents = 12 := by
  sorry

end average_cost_per_pencil_l749_749284


namespace a2_eq_1_a3_eq_13_b_arithmetic_sum_Sn_l749_749241

-- Defining the sequence a_n
def a : ℕ → ℤ
| 0       := -3  -- Note: Adjusted to be 0-indexed in Lean
| (n + 1) := 2 * a n + 2^(n + 2) + 3

-- Defining the sequence b_n
def b (n : ℕ) : ℤ := (a (n + 1) + 3) / 2^(n + 1)

-- Prove the values of a_2 and a_3
theorem a2_eq_1 : a 1 = 1 :=
  sorry

theorem a3_eq_13 : a 2 = 13 :=
  sorry

-- Prove b_n is an arithmetic sequence with common difference 1
theorem b_arithmetic (n : ℕ) : b (n + 1) - b n = 1 :=
  sorry

-- Prove the sum of first n terms of a_n
def S (n : ℕ) : ℤ :=
  (n - 2) * 2^(n + 1) - 3 * n + 4

theorem sum_Sn (n : ℕ) : (∑ i in finset.range n, a i) = S n :=
  sorry

end a2_eq_1_a3_eq_13_b_arithmetic_sum_Sn_l749_749241


namespace smallest_integer_with_18_divisors_l749_749645

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749645


namespace find_angle_BEC_l749_749114

-- Constants and assumptions
def angle_A : ℝ := 45
def angle_D : ℝ := 50
def angle_F : ℝ := 55
def E_above_C : Prop := true  -- This is a placeholder to represent the condition that E is directly above C.

-- Definition of the problem
theorem find_angle_BEC (angle_A_eq : angle_A = 45) 
                      (angle_D_eq : angle_D = 50) 
                      (angle_F_eq : angle_F = 55)
                      (triangle_BEC_formed : Prop)
                      (E_directly_above_C : E_above_C) 
                      : ∃ (BEC : ℝ), BEC = 10 :=
by sorry

end find_angle_BEC_l749_749114


namespace total_number_of_bricks_l749_749987

theorem total_number_of_bricks (rows : ℕ) (bricks_in_bottom_row : ℕ) (condition : ∀ n, n < rows → (bricks_in_bottom_row - n) > 0)
  (rows_eq : rows = 5) (bricks_in_bottom_row_eq : bricks_in_bottom_row = 8) :
  (List.sum (List.map (λ n, bricks_in_bottom_row - n) (List.range rows))) = 30 := by
  sorry

end total_number_of_bricks_l749_749987


namespace P_10_is_4_l749_749150

def T (n : ℕ) : ℕ := if n ≥ 2 then (n * (n + 1)) / 2 - 1 else 0

def P (n : ℕ) : ℚ := (List.prod (List.map (λ k, (T k : ℚ) / ((T k : ℚ) - 1)) (List.range' 3 (n - 2))))

theorem P_10_is_4 : P 10 = 4 := 
by
  -- The proof goes here
  sorry

end P_10_is_4_l749_749150


namespace smallest_integer_with_18_divisors_l749_749652

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749652


namespace food_bank_remaining_after_four_weeks_l749_749110

def week1_donated : ℝ := 40
def week1_given_out : ℝ := 0.6 * week1_donated
def week1_remaining : ℝ := week1_donated - week1_given_out

def week2_donated : ℝ := 1.5 * week1_donated
def week2_given_out : ℝ := 0.7 * week2_donated
def week2_remaining : ℝ := week2_donated - week2_given_out
def total_remaining_after_week2 : ℝ := week1_remaining + week2_remaining

def week3_donated : ℝ := 1.25 * week2_donated
def week3_given_out : ℝ := 0.8 * week3_donated
def week3_remaining : ℝ := week3_donated - week3_given_out
def total_remaining_after_week3 : ℝ := total_remaining_after_week2 + week3_remaining

def week4_donated : ℝ := 0.9 * week3_donated
def week4_given_out : ℝ := 0.5 * week4_donated
def week4_remaining : ℝ := week4_donated - week4_given_out
def total_remaining_after_week4 : ℝ := total_remaining_after_week3 + week4_remaining

theorem food_bank_remaining_after_four_weeks : total_remaining_after_week4 = 82.75 := by
  sorry

end food_bank_remaining_after_four_weeks_l749_749110


namespace coefficient_x20_expansion_l749_749837

theorem coefficient_x20_expansion :
  let f1 := (∑ k in finset.range 20, x^k)
  let f2 := (∑ k in finset.range 12, x^k)^3 
  (coeff (20 : ℕ) (f1 * f2)) = 494 :=
by sorry

end coefficient_x20_expansion_l749_749837


namespace locus_of_P_is_hyperbola_l749_749174

noncomputable def circle_locus_is_hyperbola (O F : Point) (r : ℝ) (P : Point) : Prop :=
  let C : set Point := { M : Point | dist M O = r }
  let fold_locus (M : Point) (hM : M ∈ C) : Prop :=
    let P := (fun x => x) -- function to determine P given M and F
    dist P O - dist P F = c
  ∃ M : Point, (M ∈ C) ∧ (fold_locus M M.property) → hyperbola P O F

theorem locus_of_P_is_hyperbola (O F : Point) (r : ℝ) (h₁ : dist O F > r) : 
  ∃ P : Point, circle_locus_is_hyperbola O F r P :=
begin
  sorry -- Proof to be done.
end

end locus_of_P_is_hyperbola_l749_749174


namespace smallest_integer_with_18_divisors_l749_749677

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749677


namespace range_of_a_l749_749884

def f (a x : Real) : Real :=
  a * x + a / x

def g (a x : Real) : Real :=
  Real.exp x - 3 * a * x

theorem range_of_a (a : Real) :
  (∀ x1 : Real, 0 < x1 ∧ x1 < 1 → ∃ x2 : Real, 1 < x2 ∧ f a x1 = g a x2) →
  a > 0 →
  ∃ R : Set Real, R = Ici (Real.exp 1 / 5) ∧ a ∈ R :=
by
  intro h1 h2
  set R := Ici (Real.exp 1 / 5)
  use R
  constructor
  · refl
  · sorry

end range_of_a_l749_749884


namespace maximum_number_of_kids_l749_749379

variable (k n: ℕ)

-- Assume k and n are positive integers
variables (hk : k > 0) (hn : n > 0)

-- Define the conditions of the problem
def conditions (k n : ℕ) : Prop :=
  ∀ (kids : List (List ℕ)), kids.length = 2 * n ∧          
  (∀ kid_pair : Fin (k + 1) → Prop, 
    ∃ i j, i ≠ j ∧ kid_pair i ∩ kid_pair j ≠ ∅)

-- Define the maximum number of kids possible
def max_kids (n k : ℕ) : ℕ := 3 * n * k

-- Statement that under the given conditions, the maximum number of kids is 3nk
theorem maximum_number_of_kids (k n : ℕ)(hk : k > 0)(hn : n > 0)(h : conditions k n):
  ∀ kids : ℕ, kids ≤ max_kids n k :=
sorry

end maximum_number_of_kids_l749_749379


namespace tenth_line_is_correct_l749_749138

def next_line (s : String) : String :=
  let rec aux (s : List Char) (prev : Char) (count : Nat) (acc : List Char) : List Char :=
    match s with
    | [] => (acc ++ [Char.ofNat (count + '0'.toNat)] ++ [prev])
    | hd :: tl =>
      if hd = prev then
        aux tl prev (count + 1) acc
      else
        aux tl hd 1 (acc ++ [Char.ofNat (count + '0'.toNat)] ++ [prev])
  (aux s.toList s.head! 1 []).asString

def nth_line (n : Nat) (s : String) : String :=
  match n with
  | 0 => s
  | n + 1 => nth_line n (next_line s)

def ninth_line := "311311222113111231131112322211231231131112"
def tenth_line := "13211321322111312211"

theorem tenth_line_is_correct : nth_line 1 (nth_line 0 ninth_line) = tenth_line := by
  -- Proof will need to be filled in later
  sorry

end tenth_line_is_correct_l749_749138


namespace probability_of_green_or_yellow_marble_l749_749606

theorem probability_of_green_or_yellow_marble :
  let total_marbles := 4 + 3 + 6 in
  let favorable_marbles := 4 + 3 in
  favorable_marbles / total_marbles = 7 / 13 :=
by
  sorry

end probability_of_green_or_yellow_marble_l749_749606


namespace conic_section_is_parabola_l749_749845

theorem conic_section_is_parabola :
  (∀ x y : ℝ, x^2 - 4 * x * y + 4 * y^2 - 6 * x - 8 * y + 9 = 0) → "parabola" :=
  by
  sorry

end conic_section_is_parabola_l749_749845


namespace smallest_integer_with_18_divisors_l749_749703

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749703


namespace lambda_solution_l749_749061

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749061


namespace slope_parallelogram_l749_749344

def Point : Type := (ℝ × ℝ)

def is_parallelogram (A B C D : Point) : Prop :=
  ∃ M : Point, 
    (M.1 = ((A.1 + C.1) / 2)) ∧ (M.2 = ((A.2 + C.2) / 2)) ∧ 
    (M.1 = ((B.1 + D.1) / 2)) ∧ (M.2 = ((B.2 + D.2) / 2))

def slope_through_origin (P : Point) : ℝ := P.2 / P.1

def is_congruent_polygons (P Q R S : Point) (Line : ℝ) : Prop := sorry -- Details of congruence not provided

noncomputable def find_slope (A B C D : Point) : ℝ :=
  slope_through_origin (10, 45 + 135 / 19) -- Derived point (10, 45 + a) where a = 135 / 19

theorem slope_parallelogram :
  ∀ (A B C D : Point),
    is_parallelogram A B C D →
    (A = (10, 45)) → (B = (10, 114)) → (C = (28, 153)) → (D = (28, 84)) →
    find_slope A B C D = 99 / 19 ∧ 99 + 19 = 118 := by
  intros A B C D h_parallelogram hA hB hC hD
  have h_slope : find_slope A B C D = 99 / 19 := by sorry
  have h_sum : 99 + 19 = 118 := rfl
  exact ⟨h_slope, h_sum⟩

end slope_parallelogram_l749_749344


namespace samia_walked_distance_l749_749545

-- Define the given data as constants
constant distance_jogged_fraction : ℝ := 1 / 3
constant distance_walked_fraction : ℝ := 2 / 3
constant speed_jogging : ℝ := 8
constant speed_walking : ℝ := 4
constant total_time_hours : ℝ := 105 / 60

-- Define the total distance
def total_distance (x : ℝ) := 3 * x

-- Define the jogging distance
def jogging_distance (x : ℝ) := x

-- Define the walking distance
def walking_distance (x : ℝ) := 2 * x

-- Define the time taken to jog
def time_jogging (x : ℝ) := jogging_distance(x) / speed_jogging

-- Define the time taken to walk
def time_walking (x : ℝ) := walking_distance(x) / speed_walking

-- Define the proof problem
theorem samia_walked_distance : 
  (∃ x : ℝ, total_time_hours = time_jogging(x) + time_walking(x) ∧ walking_distance(x) = 5.6) :=
begin
  sorry
end

end samia_walked_distance_l749_749545


namespace smallest_integer_with_18_divisors_l749_749672

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749672


namespace DeMoivreTheorem_solution_l749_749784

noncomputable def cos_210 := Real.cos (210 * Real.pi / 180)
noncomputable def sin_210 := Real.sin (210 * Real.pi / 180)
noncomputable def cos_120 := Real.cos (120 * Real.pi / 180)
noncomputable def sin_120 := Real.sin (120 * Real.pi / 180)

theorem DeMoivreTheorem (n : ℕ) (θ : ℝ) : 
    (Real.cos θ + complex.i * Real.sin θ)^n = Real.cos (n * θ) + complex.i * Real.sin (n * θ) := sorry

theorem solution : (cos_210 + complex.i * sin_210)^60 = -1/2 + complex.i * (Real.sqrt 3 / 2) :=
by
  apply DeMoivreTheorem
  sorry

end DeMoivreTheorem_solution_l749_749784


namespace quad_composition_even_l749_749513

theorem quad_composition_even {g : ℝ → ℝ} (h : ∀ x, g(-x) = g(x)) :
  ∀ x, g(g(g(g(-x)))) = g(g(g(g(x)))) :=
by
  intros x
  sorry

end quad_composition_even_l749_749513


namespace angle_EAD_is_10_degrees_l749_749492

theorem angle_EAD_is_10_degrees
  (A D E B C : Type) 
  [AddGroup A] [AddGroup D] [AddGroup E] [AddGroup B] [AddGroup C]
  (\angle ADE : ℝ) -- Angle ADE
  (h1 : \angle ADE = 140)
  (h2 : Point B ∈ LineSegment A D) -- Point B is on side AD
  (h3 : Point C ∈ LineSegment A E) -- Point C is on side AE
  (h4 : Distance A B = Distance B C ∧ Distance B C = Distance C D ∧ Distance C D = Distance D E) -- AB = BC = CD = DE
  : \angle EAD = 10 :=
by
  sorry

end angle_EAD_is_10_degrees_l749_749492


namespace find_lambda_l749_749056

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749056


namespace lambda_solution_l749_749062

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749062


namespace determinant_of_matrix_l749_749832

def matrix := 𝕄[7, -2; -3, 6]

theorem determinant_of_matrix : det(matrix) = 36 :=
by 
  sorry

end determinant_of_matrix_l749_749832


namespace total_weight_apples_l749_749170

variable (Minjae_weight : ℝ) (Father_weight : ℝ)

theorem total_weight_apples (h1 : Minjae_weight = 2.6) (h2 : Father_weight = 5.98) :
  Minjae_weight + Father_weight = 8.58 :=
by 
  sorry

end total_weight_apples_l749_749170


namespace opp_sqrt3_minus_2_eq_2_minus_sqrt3_abs_sqrt3_minus_2_eq_2_minus_sqrt3_l749_749234

theorem opp_sqrt3_minus_2_eq_2_minus_sqrt3 : - (sqrt 3 - 2) = 2 - sqrt 3 :=
by
  sorry

theorem abs_sqrt3_minus_2_eq_2_minus_sqrt3 : abs (sqrt 3 - 2) = 2 - sqrt 3 :=
by
  sorry

end opp_sqrt3_minus_2_eq_2_minus_sqrt3_abs_sqrt3_minus_2_eq_2_minus_sqrt3_l749_749234


namespace smallest_number_satisfies_conditions_l749_749370

-- Define the number we are looking for
def number : ℕ := 391410

theorem smallest_number_satisfies_conditions :
  (number % 7 = 2) ∧
  (number % 11 = 2) ∧
  (number % 13 = 2) ∧
  (number % 17 = 3) ∧
  (number % 23 = 0) ∧
  (number % 5 = 0) :=
by
  -- We need to prove that 391410 satisfies all the given conditions.
  -- This proof will include detailed steps to verify each condition
  sorry

end smallest_number_satisfies_conditions_l749_749370


namespace distance_between_Q_and_R_l749_749473

noncomputable def distance_QR : ℝ :=
  let DE : ℝ := 9
  let EF : ℝ := 12
  let DF : ℝ := 15
  let N : ℝ := 7.5
  let QF : ℝ := (N * DF) / EF
  let QD : ℝ := DF - QF
  let QR : ℝ := (QD * DF) / EF
  QR

theorem distance_between_Q_and_R 
  (DE EF DF N QF QD QR : ℝ )
  (h1 : DE = 9)
  (h2 : EF = 12)
  (h3 : DF = 15)
  (h4 : N = DF / 2)
  (h5 : QF = N * DF / EF)
  (h6 : QD = DF - QF)
  (h7 : QR = QD * DF / EF) :
  QR = 7.03125 :=
by
  sorry

end distance_between_Q_and_R_l749_749473


namespace distinct_triangles_count_l749_749959

-- Define the 2x4 grid of points.
def grid_points : List (ℤ × ℤ) := [(0,0), (1,0), (2,0), (3,0),
                                    (0,1), (1,1), (2,1), (3,1)]

-- Define a function that checks if three points are collinear.
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define a function that counts the number of combinations of three points from the grid.
def combinations_of_three (l : List (ℤ × ℤ)) : List (ℤ × ℤ × ℤ) :=
  l.bind (λ p1, l.bind (λ p2, l.bind (λ p3, if p1 < p2 ∧ p2 < p3 then [(p1, p2, p3)] else [])))

-- Define a function that checks if three points form a non-degenerate triangle.
def non_degenerate_triangle (p1 p2 p3 : ℤ × ℤ) : Prop := ¬ collinear p1 p2 p3

-- Main theorem statement: there are 44 non-degenerate triangles.
theorem distinct_triangles_count : ∑ (t : ℤ × ℤ × ℤ) in (combinations_of_three grid_points),
  (if non_degenerate_triangle t.1 t.2 t.3 then 1 else 0) = 44 :=
by
  sorry

end distinct_triangles_count_l749_749959


namespace pie_selling_days_l749_749310

theorem pie_selling_days (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end pie_selling_days_l749_749310


namespace range_of_a_l749_749921

def f (x : ℝ) : ℝ := x^2 - 6 * x + 8

theorem range_of_a (a : ℝ) (h : 1 < a ∧ a ≤ 3) :
  (∀ x ∈ set.Icc 1 a, f x ≥ f a) :=
sorry

end range_of_a_l749_749921


namespace find_lambda_l749_749006

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749006


namespace jack_flyers_count_l749_749141

-- Definitions based on the given conditions
def total_flyers : ℕ := 1236
def rose_flyers : ℕ := 320
def flyers_left : ℕ := 796

-- Statement to prove
theorem jack_flyers_count : total_flyers - (rose_flyers + flyers_left) = 120 := by
  sorry

end jack_flyers_count_l749_749141


namespace simplify_expression_l749_749733

-- Define the variables x and y
variables (x y : ℝ)

-- State the theorem
theorem simplify_expression (x y : ℝ) (hy : y ≠ 0) :
  ((x + 3 * y)^2 - (x + y) * (x - y)) / (2 * y) = 3 * x + 5 * y := 
by 
  -- skip the proof
  sorry

end simplify_expression_l749_749733


namespace sufficient_not_necessary_l749_749563

theorem sufficient_not_necessary (x : ℝ) (h1 : -1 < x) (h2 : x < 3) :
    x^2 - 2*x < 8 :=
by
    -- Proof to be filled in.
    sorry

end sufficient_not_necessary_l749_749563


namespace quadratic_intersects_at_two_points_l749_749880

variable {k : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_intersects_at_two_points (k : ℝ) :
  (let a := k - 2 in
   let b := -(2 * k - 1) in
   let c := k in
   discriminant a b c > 0 ∧ a ≠ 0) ↔ k > -1 / 4 ∧ k ≠ 2 :=
by
  let a := k - 2
  let b := -(2 * k - 1)
  let c := k
  have D : discriminant a b c = 4 * k + 1 := by
    rw [discriminant, sq, neg_mul_eq_mul_neg, neg_neg, mul_assoc, mul_assoc, mul_comm (4 * k),
        mul_comm 4 1, ←mul_sub, add_comm, sq_sub, mul_sub, mul_comm]
  sorry

end quadratic_intersects_at_two_points_l749_749880


namespace smallest_four_digits_valid_remainder_l749_749864

def isFourDigit (x : ℕ) : Prop := 1000 ≤ x ∧ x ≤ 9999 

def validRemainder (x : ℕ) : Prop := 
  ∀ k ∈ [2, 3, 4, 5, 6], x % k = 1

theorem smallest_four_digits_valid_remainder :
  ∃ x1 x2 x3 x4 : ℕ,
    isFourDigit x1 ∧ validRemainder x1 ∧
    isFourDigit x2 ∧ validRemainder x2 ∧
    isFourDigit x3 ∧ validRemainder x3 ∧
    isFourDigit x4 ∧ validRemainder x4 ∧
    x1 = 1021 ∧ x2 = 1081 ∧ x3 = 1141 ∧ x4 = 1201 := 
sorry

end smallest_four_digits_valid_remainder_l749_749864


namespace marbles_given_to_joan_l749_749534

def mary_original_marbles : ℝ := 9.0
def mary_marbles_left : ℝ := 6.0

theorem marbles_given_to_joan :
  mary_original_marbles - mary_marbles_left = 3 := 
by
  sorry

end marbles_given_to_joan_l749_749534


namespace determine_phi_l749_749945

-- Define the given function y = sin(x + ϕ)
def given_function (x ϕ : ℝ) : ℝ :=
  Real.sin (x + ϕ)

-- Define the transformed function y = sin(2x - (2π/3) + ϕ)
def transformed_function (x ϕ : ℝ) : ℝ :=
  Real.sin (2*x - (2*Real.pi/3) + ϕ)

-- Define symmetry about the y-axis condition
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- The main theorem to prove that if the transformed function is symmetric about the y-axis,
-- then ϕ = π/6
theorem determine_phi (ϕ : ℝ) (h : symmetric_about_y_axis (transformed_function ϕ)) : ϕ = Real.pi/6 :=
by
  sorry

end determine_phi_l749_749945


namespace largest_value_fraction_l749_749552

noncomputable def largest_value (x y : ℝ) : ℝ := (x + y) / x

theorem largest_value_fraction
  (x y : ℝ)
  (hx1 : -5 ≤ x)
  (hx2 : x ≤ -3)
  (hy1 : 3 ≤ y)
  (hy2 : y ≤ 5)
  (hy_odd : ∃ k : ℤ, y = 2 * k + 1) :
  largest_value x y = 0.4 :=
sorry

end largest_value_fraction_l749_749552


namespace find_general_term_of_arithmetic_sequence_l749_749433

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def increasing_arithmetic_sequence (a : ℕ → ℤ) :=
  arithmetic_sequence a ∧ ∀ n : ℕ, a (n + 1) > a n

-- The sequence a_n where a_1 = 1, a_3 = a_2 ^ 2 - 4, and it's increasing
def a : ℕ → ℤ := λ n, 2 * (n + 1) - 1 -- Initial guess of the formula

-- The proof problem statement 
theorem find_general_term_of_arithmetic_sequence 
  (h0 : a 0 = 1)
  (h1 : a 2 = a 1 ^ 2 - 4)
  (h_inc : increasing_arithmetic_sequence a) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  sorry

end find_general_term_of_arithmetic_sequence_l749_749433


namespace speed_of_boat_still_water_l749_749582

variable (x : ℝ) -- Define the speed of the boat in still water as x

-- Conditions:
variables (current_speed : ℝ) (travel_time : ℝ) (travel_distance : ℝ)
variable (effective_speed : ℝ)

-- Assign values to the conditions given in the problem:
def current_speed_value : current_speed = 3
def travel_time_value : travel_time = 0.4
def travel_distance_value : travel_distance = 7.2

-- Define the effective speed downstream in terms of x and current_speed:
def effective_speed_value : effective_speed = x + current_speed

-- State the main proof goal under the conditions provided:
theorem speed_of_boat_still_water : 
  current_speed_value → 
  travel_time_value → 
  travel_distance_value → 
  effective_speed_value →
  effective_speed * travel_time = travel_distance → 
  x = 15 :=
by
  intros
  sorry

end speed_of_boat_still_water_l749_749582


namespace mn_condition_l749_749978

theorem mn_condition {m n : ℕ} (h : m * n = 121) : (m + 1) * (n + 1) = 144 :=
sorry

end mn_condition_l749_749978


namespace smallest_integer_with_18_divisors_l749_749632

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749632


namespace f_a_proof_l749_749426

def f (x : ℝ) : ℝ := 
  if x >= 0 then x^3 - 1 else 2^x

theorem f_a_proof (a : ℝ) (h : f 0 = a) : 
  f a = 1 / 2 :=
begin
  have ha : a = -1,
  { simp [f] at h,
    exact h },
  rw ha,
  simp [f]
end

end f_a_proof_l749_749426


namespace divisibility_proof_l749_749501

theorem divisibility_proof (n : ℕ) (hn : 0 < n) (h : n ∣ (10^n - 1)) : 
  n ∣ ((10^n - 1) / 9) :=
  sorry

end divisibility_proof_l749_749501


namespace find_lambda_l749_749089

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749089


namespace smallest_positive_integer_with_18_divisors_l749_749670

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749670


namespace smallest_integer_with_18_divisors_l749_749642

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749642


namespace reflection_symmetry_l749_749299

theorem reflection_symmetry 
  {A A' B L P Q : Point} 
  (hA' : symmetric_about_line PQ A A')
  (hL : reflection_point L PQ A B)
  (h_incidence_reflection : ∀ X Y Z M, incident_angle PQ X Y = reflected_angle PQ Y M Z) 
  : collinear B L A' := 
sorry

end reflection_symmetry_l749_749299


namespace smallest_s_is_4_l749_749581

noncomputable def smallest_s (a b : ℝ) (s : ℕ) : Prop :=
  a + s > b ∧ a + b > s ∧ b + s > a

theorem smallest_s_is_4 : smallest_s 7.5 11 4 :=
by
  unfold smallest_s
  apply And.intro
  .
  .
  .
  sorry

end smallest_s_is_4_l749_749581


namespace complement_of_A_in_U_is_2_l749_749530

open Set

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }

theorem complement_of_A_in_U_is_2 : compl A ∩ U = {2} :=
by
  sorry

end complement_of_A_in_U_is_2_l749_749530


namespace proof_example_l749_749358

noncomputable def problem_statement : Prop :=
  (0 ∈ ℕ) ∧ (π ∉ ℚ) ∧ (-1 ∉ ℕ)

theorem proof_example : problem_statement := by
  split
  . exact Nat.zero_mem
  . exact Real.not_rat_pi
  . exact Nat.neg_one_not_nat

end proof_example_l749_749358


namespace player_B_more_than_18_guesses_strategy_within_24_guesses_strategy_using_not_more_than_22_guesses_l749_749598

noncomputable def playerA_picked (N : ℕ) (h : 10 ≤ N ∧ N < 100) : Prop :=
h

noncomputable def playerA_response (N guess : ℕ) (h : 10 ≤ N ∧ N < 100) : Prop :=
(N = guess) ∨
(N / 10 = guess / 10 ∧ |N % 10 - guess % 10| = 1) ∨
(N % 10 = guess % 10 ∧ |N / 10 - guess / 10| = 1)

theorem player_B_more_than_18_guesses :
  ∀ guess_limit : ℕ, guess_limit ≤ 18 →
  ∃ N : ℕ, 10 ≤ N ∧ N < 100 ∧
  ∀ guesses : list ℕ, (∀ g ∈ guesses, 10 ≤ g ∧ g < 100) →
  (∀ guess ∈ guesses, playerA_response N guess (and.intro (by linarith) (by linarith)) = false) →
  guess_limit < |guesses| :=
sorry

theorem strategy_within_24_guesses :
  ∃ s : (ℕ → ℕ) → list ℕ, ∀ N : ℕ, 10 ≤ N ∧ N < 100 →
  ∃ guesses : list ℕ, (∀ g ∈ guesses, 10 ≤ g ∧ g < 100) ∧
  |guesses| ≤ 24 ∧
  (∃ guess ∈ guesses, playerA_response N guess (and.intro (by linarith) (by linarith))) :=
sorry

theorem strategy_using_not_more_than_22_guesses :
  ∃ s : (ℕ → ℕ) → list ℕ, ∀ N : ℕ, 10 ≤ N ∧ N < 100 →
  ∃ guesses : list ℕ, (∀ g ∈ guesses, 10 ≤ g ∧ g < 100) ∧
  |guesses| ≤ 22 ∧
  (∃ guess ∈ guesses, playerA_response N guess (and.intro (by linarith) (by linarith))) :=
sorry

end player_B_more_than_18_guesses_strategy_within_24_guesses_strategy_using_not_more_than_22_guesses_l749_749598


namespace compare_points_on_line_l749_749410

theorem compare_points_on_line (m n : ℝ) 
  (hA : ∃ (x : ℝ), x = -3 ∧ m = -2 * x + 1) 
  (hB : ∃ (x : ℝ), x = 2 ∧ n = -2 * x + 1) : 
  m > n :=
by sorry

end compare_points_on_line_l749_749410


namespace evaluate_fraction_l749_749356

theorem evaluate_fraction (a b : ℝ) (h1 : a = 5) (h2 : b = 3) : 3 / (a + b) = 3 / 8 :=
by
  rw [h1, h2]
  sorry

end evaluate_fraction_l749_749356


namespace arcsin_one_half_l749_749826

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749826


namespace evaluate_poly_using_horner_at_2_l749_749782

def poly (x : ℕ) : ℕ := 8 * x ^ 4 + 5 * x ^ 3 + 3 * x ^ 2 + 2 * x + 1

-- Using Horner's method for evaluation

def horner_eval (a b : ℕ) : ℕ := a * b

theorem evaluate_poly_using_horner_at_2 :
  let x := 2 in
  let v0 := 8 in
  let v1 := horner_eval v0 x + 5 in
  let v2 := horner_eval v1 x + 3 in
  v2 = 45 :=
by
  sorry

end evaluate_poly_using_horner_at_2_l749_749782


namespace count_positive_integers_in_range_sq_le_count_positive_integers_in_range_sq_l749_749382

theorem count_positive_integers_in_range_sq_le (x : ℕ) : 
  225 ≤ x^2 ∧ x^2 ≤ 400 → x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20 :=
by { sorry }

theorem count_positive_integers_in_range_sq : 
  { x : ℕ | 225 ≤ x^2 ∧ x^2 ≤ 400 }.to_finset.card = 6 :=
by { sorry }

end count_positive_integers_in_range_sq_le_count_positive_integers_in_range_sq_l749_749382


namespace expected_pairs_of_adjacent_hearts_l749_749560

noncomputable def expected_adjacent_heart_pairs (num_cards : ℕ) (num_hearts : ℕ) : ℚ :=
  let prob_same_suit := (num_hearts - 1) / (num_cards - 1)
  in num_hearts * prob_same_suit

theorem expected_pairs_of_adjacent_hearts :
  expected_adjacent_heart_pairs 40 10 = 30 / 13 :=
by 
  sorry

end expected_pairs_of_adjacent_hearts_l749_749560


namespace initial_number_of_men_l749_749210

theorem initial_number_of_men (M : ℕ) 
  (h1 : M * 8 * 40 = (M + 30) * 6 * 50) 
  : M = 450 :=
by 
  sorry

end initial_number_of_men_l749_749210


namespace min_fraction_value_l749_749442

noncomputable theory

open Real

theorem min_fraction_value 
  (m n : ℝ)
  (h_pos_m : m > 0)
  (h_pos_n : n > 0)
  (h_parallel : (2, 1) ∥ (4 - n, m)) :
  (n / m + 8 / n) = 6 :=
  sorry

end min_fraction_value_l749_749442


namespace smallest_integer_with_18_divisors_l749_749682

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749682


namespace quad_composition_even_l749_749512

theorem quad_composition_even {g : ℝ → ℝ} (h : ∀ x, g(-x) = g(x)) :
  ∀ x, g(g(g(g(-x)))) = g(g(g(g(x)))) :=
by
  intros x
  sorry

end quad_composition_even_l749_749512


namespace smallest_int_with_18_divisors_l749_749657

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749657


namespace transport_capacity_min_vehicles_needed_min_transport_cost_l749_749475

section TransportProblem

variables (x y α : ℕ)

-- Condition definitions
def condition1 : Prop := 5 * x + 3 * y = 370
def condition2 : Prop := 4 * x + 7 * y = 480
def condition3 : Prop := 50 * α + 40 * (20 - α) ≥ 955
def condition4 : Prop := 3000 * α + 2000 * (20 - α) ≤ 58800

-- Correct answers
def answer1 : ℕ := 50
def answer2 : ℕ := 40
def answer3 : ℕ := 16
def answer4 : ℕ := 56000

-- The proof problem statements
theorem transport_capacity :
  condition1 ∧ condition2 → x = answer1 ∧ y = answer2 :=
begin
  sorry,
end

theorem min_vehicles_needed :
  condition3 → α ≥ answer3 :=
begin
  sorry,
end

theorem min_transport_cost :
  condition3 ∧ condition4 → 3000 * α + 2000 * (20 - α) = answer4 :=
begin
  sorry,
end

end TransportProblem

end transport_capacity_min_vehicles_needed_min_transport_cost_l749_749475


namespace fixed_point_inequality_l749_749399

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * a^((x + 1) / 2) - 4

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = -1 :=
sorry

theorem inequality (a : ℝ) (x : ℝ) (h : a > 1) :
  f a (x - 3 / 4) ≥ 3 / (a^(x^2 / 2)) - 4 :=
sorry

end fixed_point_inequality_l749_749399


namespace smallest_int_with_18_divisors_l749_749660

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749660


namespace cos_sq_sub_sin_sq_15_eq_l749_749352

-- Auxiliary function to convert degrees to radians
def deg_to_rad (d : ℝ) : ℝ := d * (π / 180)

-- Definition of cosine and sine for specific degree
def cos_deg (d : ℝ) : ℝ := Real.cos (deg_to_rad d)
def sin_deg (d : ℝ) : ℝ := Real.sin (deg_to_rad d)

theorem cos_sq_sub_sin_sq_15_eq : 
  cos_deg 15 ^ 2 - sin_deg 15 ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end cos_sq_sub_sin_sq_15_eq_l749_749352


namespace find_lambda_l749_749058

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749058


namespace find_constants_l749_749529

open Nat

variables {n : ℕ} (b c : ℤ)
def S (n : ℕ) := n^2 + b * n + c
def a (n : ℕ) := S n - S (n - 1)

theorem find_constants (a2a3_sum_eq_4 : a 2 + a 3 = 4) : 
  c = 0 ∧ b = -2 := 
by 
  sorry

end find_constants_l749_749529


namespace rational_numbers_set_l749_749519

def R := ℝ
def F := {f : R → R // ∀ x y : R, f(x + f(y)) = f(x) + f(y)}

theorem rational_numbers_set :
  ∀ q : ℚ, (∀ f : F, ∃ z : R, f.val z = q * z) ↔
    q ∈ { q : ℚ | ∃ n : ℤ, n ≠ 0 ∧ q = (n + 1) / n } :=
by
  sorry

end rational_numbers_set_l749_749519


namespace necessary_and_sufficient_cond_eq_conjugate_necessary_and_sufficient_cond_eq_reciprocal_of_conjugate_l749_749551

-- Definitions for conditions
def complex_eq_conjugate (z : ℂ) : Prop :=
  ∃ (a b : ℝ), z = a + b * complex.I ∧ z = a - b * complex.I

def complex_eq_reciprocal_of_conjugate (z : ℂ) : Prop :=
  ∃ (a b : ℝ), z = a + b * complex.I ∧ z = (a + b * complex.I) / (a ^ 2 + b ^ 2)

-- Proof statements
theorem necessary_and_sufficient_cond_eq_conjugate (a b : ℝ) :
  complex_eq_conjugate (a + b * complex.I) ↔ b = 0 := 
sorry

theorem necessary_and_sufficient_cond_eq_reciprocal_of_conjugate (a b : ℝ) :
  complex_eq_reciprocal_of_conjugate (a + b * complex.I) ↔ a ^ 2 + b ^ 2 = 1 := 
sorry

end necessary_and_sufficient_cond_eq_conjugate_necessary_and_sufficient_cond_eq_reciprocal_of_conjugate_l749_749551


namespace find_missing_number_l749_749117

def median (s : List ℕ) : ℕ := 
  let sorted := s.sort
  let len := sorted.length
  if len % 2 = 1 then sorted[len / 2] 
  else (sorted[len / 2 - 1] + sorted[len / 2]) / 2

variable {s : List ℕ} (h_s : s = [5, 6, 3, 10, 4])
variable {missing_number : ℕ} (h_miss : missing_number = 10)

theorem find_missing_number (s : List ℕ) (missing_number : ℕ) (h_s : s = [5, 6, 3, 10, 4]) (h_miss : missing_number = 10) :
  median [5, 6, 3, missing_number, 4] = 10 := 
by {
  sorry
}

end find_missing_number_l749_749117


namespace sum_of_solutions_l749_749711

theorem sum_of_solutions : 
  (∑ x in {n | |3 * n - 8| = 4}, x) = 16 / 3 :=
by
  sorry

end sum_of_solutions_l749_749711


namespace evaluate_expression_l749_749856

theorem evaluate_expression (x y : ℚ) (hx : x = 4 / 3) (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 40 :=
by
  rw [hx, hy]
  sorry

end evaluate_expression_l749_749856


namespace projection_correct_l749_749907

variables (a b : ℝ^(2))
noncomputable def projection_of_a_on_b (a b : ℝ^(2)) (θ : ℝ) := sqrt 2 * Real.cos θ

theorem projection_correct 
  (θ : ℝ) (hθ : θ = Real.arccos ((a ⬝ b) / ((Real.sqrt (a ⬝ a)) * (Real.sqrt (b ⬝ b))))) 
  (ha : (Real.sqrt (a ⬝ a)) = sqrt 2) 
  (hθ_val : θ = π / 3) : 
  projection_of_a_on_b a b θ = sqrt 2 * Real.cos (π / 3) :=
by
  rw [projection_of_a_on_b, hθ_val, Real.cos_pi_div_three]
  sorry

end projection_correct_l749_749907


namespace correct_conditions_l749_749223

namespace PowerFunctionConditions

def power_function (n : ℝ) (x : ℝ) : ℝ := x^n

def condition_1 (n : ℝ) : Prop := power_function n 1 = 1 ∧ power_function n 0 = 0
def condition_2 (n : ℝ) : Prop := ∀ x : ℝ, x < 0 → power_function n x ≥ 0
def condition_3 (n : ℝ) : Prop := n = 0 → ∀ x : ℝ, power_function n x = 1
def condition_4 (n : ℝ) : Prop := n > 0 → ∀ x y : ℝ, x < y → power_function n x < power_function n y
def condition_5 (n : ℝ) : Prop := n < 0 → ∀ x y : ℝ, 0 < x ∧ x < y → power_function n x > power_function n y

theorem correct_conditions : 
  (condition_2 2 ∧ condition_5 (-1)) ∧ 
  ¬(condition_1 (-1) ∧ condition_4 2 ∧ condition_3 0) := 
by sorry

end PowerFunctionConditions

end correct_conditions_l749_749223


namespace smallest_positive_period_l749_749371

noncomputable def function_min_period (f : ℝ → ℝ) : ℝ :=
  Inf {T | T > 0 ∧ ∀ x, f (x + T) = f x}

theorem smallest_positive_period : function_min_period (λ x : ℝ, sin x * sin (π / 2 + x)) = π := 
by 
  sorry

end smallest_positive_period_l749_749371


namespace ratio_area_perimeter_eq_sqrt3_l749_749625

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l749_749625


namespace variance_transformation_l749_749465

theorem variance_transformation (x : Fin 10 → ℝ) (h : variance x = 8) :
  variance (λ i => 2 * x i - 1) = 32 :=
sorry

end variance_transformation_l749_749465


namespace quadrilateral_with_all_right_angles_is_rectangle_l749_749269

theorem quadrilateral_with_all_right_angles_is_rectangle (Q : Type) [quadrilateral Q]
  (all_right_angles : ∀ (a b c d : Q), right_angle a b ∧ right_angle b c ∧ right_angle c d ∧ right_angle d a) :
  rectangle Q :=
sorry

end quadrilateral_with_all_right_angles_is_rectangle_l749_749269


namespace product_of_roots_l749_749834

theorem product_of_roots :
  let a := 1
  let b := -15
  let c := 75
  let d := -125
  ∀ (x: ℝ), (a * x^3 + b * x^2 + c * x + d = 0) →
    (d ≠ 0) →
    (a ≠ 0) →
    (Product_Roots (a * x^3 + b * x^2 + c * x + d) = 125) := 
by
  -- Define the necessary assumptions and root finding mechanism
  sorry

end product_of_roots_l749_749834


namespace equilateral_triangle_ratio_l749_749613

theorem equilateral_triangle_ratio (s : ℝ) (h_s : s = 12) : 
  (let A := (√3 * s^2) / 4 in let P := 3 * s in A / P = √3) :=
by
  sorry

end equilateral_triangle_ratio_l749_749613


namespace work_completion_l749_749720

theorem work_completion (a b : ℕ) (hab : a = 2 * b) (hwork_together : (1/a + 1/b) = 1/8) : b = 24 := by
  sorry

end work_completion_l749_749720


namespace counting_ways_to_arrange_balls_l749_749332

/-- Prove that there are 1728 ways to arrange 10 balls labeled with numbers 1 to 10 in a row such that the sum of the numbers on any three consecutive balls is a multiple of 3. -/
theorem counting_ways_to_arrange_balls : 
  ∃ (arrangements : ℕ), arrangements = 1728 ∧ 
  (∀ (balls : list ℕ), (balls.length = 10 ∧ (∀ i : ℕ, i + 2 < balls.length → (balls.nth i).getD 0 + (balls.nth (i+1)).getD 0 + (balls.nth (i+2)).getD 0) % 3 = 0) → arrangements) := 
  sorry

end counting_ways_to_arrange_balls_l749_749332


namespace max_stamps_l749_749105

theorem max_stamps (cents_per_stamp : ℕ) (budget_cents : ℕ) (h_stamp_price : cents_per_stamp = 33) (h_budget : budget_cents = 3200) :
  ∃ n : ℕ, n * cents_per_stamp ≤ budget_cents ∧ ∀ m : ℕ, m > n → m * cents_per_stamp > budget_cents :=
by
  have cents_per_stamp := 33
  have budget_cents := 3200
  use 96
  split
  {
    -- Proof that 96 * 33 ≤ 3200
    sorry
  }
  {
    -- Proof of the upper bound restriction
    sorry
  }

end max_stamps_l749_749105


namespace determine_m_l749_749419

noncomputable def has_equal_real_roots (m : ℝ) : Prop :=
  m ≠ 0 ∧ (m^2 - 8 * m = 0)

theorem determine_m (m : ℝ) (h : has_equal_real_roots m) : m = 8 :=
  sorry

end determine_m_l749_749419


namespace peter_remaining_money_l749_749182

theorem peter_remaining_money :
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 426 :=
by
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  show remaining_money = 426 from sorry

end peter_remaining_money_l749_749182


namespace arithmetic_sequence_a_general_term_b_smallest_n_condition_l749_749905

open Function

noncomputable def a_seq (n : ℕ) : ℚ := 
  if n = 0 then 1 else 1 / 3 * n + 2 / 3

noncomputable def S_seq (n : ℕ) : ℚ := 
  if n = 0 then 0 else n / 6 * (n + 5)

noncomputable def b_seq (n : ℕ) : ℕ :=
  3 * 2^(n-1) - 2

theorem arithmetic_sequence_a (n : ℕ) (hn : n > 0) : 
  2 * S_seq n = 3 * a_seq n^2 + a_seq n - 2 :=
sorry

theorem general_term_b (n : ℕ) : 
  b_seq n = 3 * 2^(n-1) - 2 :=
sorry

theorem smallest_n_condition (n : ℕ) (hn : n > 0) : 
  (S_seq n / (b_seq n + 2) < 1 / 4) ↔ n ≥ 5 :=
sorry

end arithmetic_sequence_a_general_term_b_smallest_n_condition_l749_749905


namespace monotonicity_a1_range_of_a_l749_749937

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- 1. Monotonicity when \( a = 1 \)
theorem monotonicity_a1 :
  (∀ x > 0, (f x 1)' > 0) ∧ (∀ x < 0, (f x 1)' < 0) :=
by
  sorry

-- 2. Range of \( a \) for \( f(x) \geq \frac{1}{2} x^3 + 1 \) for \( x \geq 0 \)
theorem range_of_a (a : ℝ) (x : ℝ) (hx : x ≥ 0) (hf : f x a ≥ (1 / 2) * x^3 + 1) :
  a ≥ (7 - Real.exp 2) / 4 :=
by
  sorry

end monotonicity_a1_range_of_a_l749_749937


namespace number_of_divisors_l749_749445

theorem number_of_divisors (n : ℕ) (h1: n = 70) (h2: ∀ k, k ∣ n → k > 3 ↔ (k = 5 ∨ k = 7 ∨ k = 10 ∨ k = 14 ∨ k = 35 ∨ k = 70)) : 
  {k : ℕ | k ∣ n ∧ k > 3}.to_finset.card = 6 :=
by {
  simp [h1, h2],
  sorry
}

end number_of_divisors_l749_749445


namespace student_sums_l749_749771

theorem student_sums (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 48) : y = 36 :=
by
  sorry

end student_sums_l749_749771


namespace isosceles_trapezoid_midline_l749_749557

noncomputable def midline (S : ℝ) (α : ℝ) : ℝ := 
sqrt (S / sin α)

theorem isosceles_trapezoid_midline (S : ℝ) (α : ℝ) (h : 0 < sin α) :
  ∃ x : ℝ, x = sqrt (S / sin α) :=
begin
  use sqrt (S / sin α),
  exact rfl,
end

end isosceles_trapezoid_midline_l749_749557


namespace special_sale_day_price_l749_749779

-- Define the original price
def original_price : ℝ := 250

-- Define the first discount rate
def first_discount_rate : ℝ := 0.40

-- Calculate the price after the first discount
def price_after_first_discount (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- Define the second discount rate (special sale day)
def second_discount_rate : ℝ := 0.10

-- Calculate the price after the second discount
def price_after_second_discount (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price * (1 - discount_rate)

-- Theorem statement
theorem special_sale_day_price :
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 := by
  sorry

end special_sale_day_price_l749_749779


namespace max_value_on_interval_l749_749925

def f (x : ℝ) : ℝ := cos (2 * x) + sqrt 3 * sin (2 * x)

theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 12) ∧
  f x = sqrt 3 := sorry

end max_value_on_interval_l749_749925


namespace arcsin_of_half_l749_749791

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l749_749791


namespace vectors_combination_l749_749090

theorem vectors_combination (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h : ∃ m n : ℝ, (9, 4) = m • (2, -3) + n • (1, 2)) : (1 / m + 1 / n) = 7 / 10 :=
sorry

end vectors_combination_l749_749090


namespace common_difference_is_neg_one_l749_749899

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 : ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Conditions given in the problem
axiom arithmetic_sequence : ∀ n, a n = arithmetic_seq a1 n d
axiom geometric_mean_condition : (a 4 + 4) ^ 2 = (a 2 + 2) * (a 6 + 6)

-- Prove that the common difference is -1
theorem common_difference_is_neg_one : d = -1 := by
  sorry

end common_difference_is_neg_one_l749_749899


namespace arithmetic_sequence_common_difference_l749_749485

theorem arithmetic_sequence_common_difference  (a_n : ℕ → ℝ)
  (h1 : a_n 1 + a_n 6 = 12)
  (h2 : a_n 4 = 7) :
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 1 + (n - 1) * d ∧ d = 2 := 
sorry

end arithmetic_sequence_common_difference_l749_749485


namespace candy_sister_gave_l749_749869

theorem candy_sister_gave (initial_candies eaten_candies current_candies : ℕ) (h1 : initial_candies = 47) (h2 : eaten_candies = 25) (h3 : current_candies = 62) : 
  current_candies - (initial_candies - eaten_candies) = 40 :=
by
  have left_candies : ℕ := initial_candies - eaten_candies
  have given_candies : ℕ := current_candies - left_candies
  have h : given_candies = 40
  sorry

end candy_sister_gave_l749_749869


namespace min_value_x_4_over_x_min_value_x_4_over_x_eq_l749_749454

theorem min_value_x_4_over_x (x : ℝ) (h : x > 0) : x + 4 / x ≥ 4 :=
sorry

theorem min_value_x_4_over_x_eq (x : ℝ) (h : x > 0) : (x + 4 / x = 4) ↔ (x = 2) :=
sorry

end min_value_x_4_over_x_min_value_x_4_over_x_eq_l749_749454


namespace number_of_right_triangles_l749_749518

-- Conditions
variables (p : ℕ) (hp : p > 0)
def M := (p * 1994, 7 * p * 1994)

-- Lean statement of the proof problem
theorem number_of_right_triangles (p : ℕ) (hp : p > 0) :
  let num_right_triangles :=
    if p = 2 then 18
    else if p = 997 then 20
    else 36
  in num_right_triangles = 
    if p = 2 then 18
    else if p = 997 then 20
    else 36 :=
by {
  sorry
}

end number_of_right_triangles_l749_749518


namespace find_lambda_l749_749074

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749074


namespace smallest_integer_with_18_divisors_l749_749675

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749675


namespace min_value_sin_cos_expression_l749_749847

theorem min_value_sin_cos_expression : 
  ∀ x : ℝ, 
  let f := (sin x)^5 + (cos x)^5 + 1
  let g := (sin x)^3 + (cos x)^3 + 1
  (f / g) ≥ 1 
:= 
by
  sorry -- proof to be completed

end min_value_sin_cos_expression_l749_749847


namespace five_dice_probability_l749_749377

noncomputable def probability_even_sum_given_even_product : ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let even_product_outcomes := total_outcomes - odd_outcomes
  let even_sum_outcomes :=
    3^5 + (Nat.choose 5 2 * 3^5) + (Nat.choose 5 4 * 3^5)
  even_sum_outcomes / even_product_outcomes

theorem five_dice_probability :
  probability_even_sum_given_even_product = 3888 / 7533 := by sorry

end five_dice_probability_l749_749377


namespace find_lambda_l749_749019

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749019


namespace set_union_complement_l749_749281

open Set

variable (U A B : Set ℤ)

-- Define the universal set U, set A, and set B
def U := {-2, -1, 0, 1, 2}
def A := {1, 2}
def B := {-2, 1, 2}

theorem set_union_complement :
  A ∪ ((U \ B) : Set ℤ) = {-1, 0, 1, 2} :=
by
  rw [U, A, B]
  sorry

end set_union_complement_l749_749281


namespace fixed_point_l749_749569

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) + 2

-- Prove that the function f passes through the fixed point (0, 2)
theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 0 = 2 :=
by
  sorry

end fixed_point_l749_749569


namespace ratio_area_perimeter_eq_sqrt3_l749_749623

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l749_749623


namespace smallest_integer_with_18_divisors_l749_749651

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749651


namespace general_formula_sum_S_n_less_than_one_l749_749734

-- Define the arithmetic sequence and extract necessary roots and common difference
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) := ∀ n, a (n + 1) = a n + d

-- Define the quadratic equation and extract its roots
def quadratic_roots (a b c : ℤ) (r1 r2 : ℤ) :=
  a * r1 * r1 + b * r1 + c = 0 ∧ a * r2 * r2 + b * r2 + c = 0

-- Given conditions
axiom a_n_is_arithmetic_sequence : is_arithmetic_sequence (λ n, 4 * n + 1) 4
axiom a1_a2_roots_of_equation : quadratic_roots 1 (-14) 45 5 9
axiom d_positive: 0 < 4

-- General formula for the sequence
theorem general_formula :
  ∀ n, (λ n, 4 * n + 1) n = 4 * n + 1 :=
by
  intro n
  sorry

-- Define b_n and S_n
def b_n (n : ℕ) : ℚ := 2 / ((4 * n + 1) * (4 * (n + 1) + 1))
def S_n (n : ℕ) : ℚ := ∑ k in Finset.range n, b_n k

-- Prove that S_n < 1
theorem sum_S_n_less_than_one :
  ∀ n, S_n n < 1 :=
by
  intro n
  sorry

end general_formula_sum_S_n_less_than_one_l749_749734


namespace maximum_projection_of_OP_on_x_axis_l749_749123

variable {x y λ : ℝ}

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 9) = 1

def projection_len (x y : ℝ) (λ : ℝ) (m : ℝ) : Prop :=
    m = (72 / (9 / x + (16 / 25) * x)) * x

theorem maximum_projection_of_OP_on_x_axis
  (x y : ℝ) (A_on_ellipse : ellipse_eq x y)
  (m : ℝ) (A_perpendicular : projection_len x y λ m)
  (H : ∀ x ∈ (0 : ℝ) <.. 5, (72 / (9 / x + (16 / 25) * x)) <= 15):
  ∃ (x : ℝ), (x = 15 / 4) ∧ (72 / (9 / x + (16 / 25) * x)) = 15 :=
sorry

end maximum_projection_of_OP_on_x_axis_l749_749123


namespace fractal_seq_2000_fractal_sum_2000_l749_749953

/-- Define the fractal sequence based on the problem conditions. --/
def fractal_seq : ℕ → ℕ 
| 0       := 1
| (n + 1) := fractal_seq ((n + 1) / 2) + (if (n + 1) % 2 = 1 then 0 else 1)

/-- The value of the 2000th term in the fractal sequence is 2. --/
theorem fractal_seq_2000 : fractal_seq 2000 = 2 := 
  sorry

/-- Summation helper function for the fractal sequence. --/
def fractal_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum fractal_seq

/-- The sum of the first 2000 terms of the fractal sequence is 4004. --/
theorem fractal_sum_2000 : fractal_sum 2000 = 4004 := 
  sorry

end fractal_seq_2000_fractal_sum_2000_l749_749953


namespace find_common_ratio_l749_749554

variable (a b : ℕ → ℝ)    -- Declare sequences a_n and b_n
variable (A B : ℕ → ℝ)    -- Declare partial sums A_n and B_n
variable (q : ℝ)          -- Declare the common ratio of the geometric sequence

-- Conditions given in the problem
variable (h1 : a 3 = b 3)
variable (h2 : a 4 = b 4)
variable (h3 : (A 5 - A 3) / (B 4 - B 2) = 7)

-- Axiom for arithmetic sequence's partial sum
axiom A_n (n : ℕ) : A n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

-- Axiom for geometric sequence's partial sum
axiom B_n (n : ℕ) : B n = b 1 * (1 - q^n) / (1 - q)

-- Statement of the proof problem
theorem find_common_ratio :
  ∃ (q : ℝ), q = -sqrt(7 / 2) ∧
         a 3 = b 3 ∧
         a 4 = b 4 ∧
         (A 5 - A 3) / (B 4 - B 2) = 7 :=
sorry

end find_common_ratio_l749_749554


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749618

def equilateral_triangle (s : ℝ) : Prop :=
  s = 12

theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  ∀ (s : ℝ), equilateral_triangle s → (let P := 3 * s in
  let A := (sqrt 3 / 4) * s ^ 2 in
  A / P = sqrt 3) :=
by
  intro s hs
  rw [equilateral_triangle, hs]
  let P := 3 * 12
  let A := (sqrt 3 / 4) * 12 ^ 2
  have A_eq : A = 36 * sqrt 3 := by
    calc
      (sqrt 3 / 4) * 12 ^ 2 = (sqrt 3 / 4) * 144  : by norm_num
                      ... = (sqrt 3 * 144) / 4  : by rw div_mul_cancel
                      ... = 36 * sqrt 3         : by norm_num
  have P_eq : P = 36 := by norm_num
  rw [A_eq, P_eq]
  norm_num
  rfl

end ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749618


namespace passes_through_point_l749_749226

theorem passes_through_point (a : ℝ) (h : 0 < a :=1) : y = a^(x-1) + 1 :=
by sorry

end passes_through_point_l749_749226


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749621

def equilateral_triangle (s : ℝ) : Prop :=
  s = 12

theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  ∀ (s : ℝ), equilateral_triangle s → (let P := 3 * s in
  let A := (sqrt 3 / 4) * s ^ 2 in
  A / P = sqrt 3) :=
by
  intro s hs
  rw [equilateral_triangle, hs]
  let P := 3 * 12
  let A := (sqrt 3 / 4) * 12 ^ 2
  have A_eq : A = 36 * sqrt 3 := by
    calc
      (sqrt 3 / 4) * 12 ^ 2 = (sqrt 3 / 4) * 144  : by norm_num
                      ... = (sqrt 3 * 144) / 4  : by rw div_mul_cancel
                      ... = 36 * sqrt 3         : by norm_num
  have P_eq : P = 36 := by norm_num
  rw [A_eq, P_eq]
  norm_num
  rfl

end ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749621


namespace interest_rate_is_six_percent_l749_749369

noncomputable def amount : ℝ := 1120
noncomputable def principal : ℝ := 979.0209790209791
noncomputable def time_years : ℝ := 2 + 2 / 5

noncomputable def total_interest (A P: ℝ) : ℝ := A - P

noncomputable def interest_rate_per_annum (I P T: ℝ) : ℝ := I / (P * T) * 100

theorem interest_rate_is_six_percent :
  interest_rate_per_annum (total_interest amount principal) principal time_years = 6 := 
by
  sorry

end interest_rate_is_six_percent_l749_749369


namespace greatest_prime_factor_of_143_is_13_l749_749262

theorem greatest_prime_factor_of_143_is_13 :
  13 = (argmax_factor 143) :=
by
  have h1 : ¬ (143 % 2 = 0) := by norm_num
  have h2 : ¬ (143 % 3 = 0) := by norm_num
  have h3 : ¬ (143 % 5 = 0) := by norm_num
  have h4 : ¬ (143 % 7 = 0) := by norm_num
  have h5 : 143 % 11 = 0 :=
    by norm_num
  have h6 : nat.prime 11 :=
    by norm_num
  have h7 : nat.prime 13 :=
    by norm_num
  have factors_143 : 143 = 11 * 13 := by norm_num
  simp [greatest_prime_factor_of_143_is_13]
  exact eq.symm rfl

end greatest_prime_factor_of_143_is_13_l749_749262


namespace smallest_integer_with_18_divisors_l749_749646

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749646


namespace complex_power_cos_sin_l749_749786

theorem complex_power_cos_sin (θ : ℝ) (hθ : θ = 210) : 
  (complex.of_real (cos θ) + complex.i * complex.of_real (sin θ))^60 = 1 :=
by
  sorry

end complex_power_cos_sin_l749_749786


namespace lambda_value_l749_749022

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749022


namespace arcsin_of_half_l749_749816

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l749_749816


namespace factorial_ends_in_zeros_l749_749231

theorem factorial_ends_in_zeros :
  let count_factors (n k : ℕ) := n / k
in
let n := 625
in count_factors n 5 + count_factors n 25 + count_factors n 125 + count_factors n 625 = 156 :=
by sorry

end factorial_ends_in_zeros_l749_749231


namespace initial_green_marbles_l749_749838

theorem initial_green_marbles (m g' : ℕ) (h_m : m = 23) (h_g' : g' = 9) : (g' + m = 32) :=
by
  subst h_m
  subst h_g'
  rfl

end initial_green_marbles_l749_749838


namespace lambda_value_l749_749035

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749035


namespace geometric_sequence_seventh_term_l749_749753

-- Define the initial conditions
def geometric_sequence_first_term := 3
def geometric_sequence_fifth_term (r : ℝ) := geometric_sequence_first_term * r^4 = 243

-- Statement for the seventh term problem
theorem geometric_sequence_seventh_term (r : ℝ) 
  (h1 : geometric_sequence_first_term = 3) 
  (h2 : geometric_sequence_fifth_term r) : 
  3 * r^6 = 2187 :=
sorry

end geometric_sequence_seventh_term_l749_749753


namespace sum_seq_4032_l749_749436

-- Define sequence a_n for n in 1 to 2016
def seq (n : ℕ) : ℕ → ℕ := λ n, n

-- Hypotheses given in the problem
variable (a : ℕ → ℕ)
variable (h1 : ∀ n, 1 ≤ n ∧ n ≤ 2016 → (a n) + (a (2017 - n)) = 4)

-- Target statement
theorem sum_seq_4032 : (∑ n in Finset.range 2016, a (n + 1)) = 4032 :=
by
  sorry

end sum_seq_4032_l749_749436


namespace lambda_solution_l749_749065

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749065


namespace mode_is_correct_l749_749240

noncomputable def scores : List ℕ :=
  [60, 60, 60, 60, 72, 75, 80, 83, 85, 85, 88, 91, 91, 91, 96, 97, 97, 97, 102, 102, 102, 104, 106, 109, 110, 110, 111]

noncomputable def mode : List ℕ := [91, 97, 102]

theorem mode_is_correct :
  (∀ m ∈ mode, (List.count scores m = 3 ∧ List.count scores m ≥ List.count scores x) for all x ∈ scores) := by
  sorry

end mode_is_correct_l749_749240


namespace find_lambda_l749_749077

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749077


namespace matrix_vector_solution_l749_749840

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![0, 1], ![4, 0]]

def I : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.eye (Fin 2)

def v : Vector (Fin 2) ℚ :=
  ![-(5 : ℚ) / 28899, 1700 / 28899]

def b : Vector (Fin 2) ℚ :=
  ![5, 0]

theorem matrix_vector_solution :
  (85 • A + I) ⬝ v = b :=
by
  sorry

end matrix_vector_solution_l749_749840


namespace min_value_x1_x2_l749_749949

theorem min_value_x1_x2 (a x_1 x_2 : ℝ) (h_a_pos : 0 < a) (h_sol_set : x_1 + x_2 = 4 * a) (h_prod_set : x_1 * x_2 = 3 * a^2) : 
  x_1 + x_2 + a / (x_1 * x_2) = 4 * a + 1 / (3 * a) :=
sorry

end min_value_x1_x2_l749_749949


namespace evaluate_polynomial_at_3_l749_749600

def f (x : ℕ) : ℕ := 3 * x ^ 3 + x - 3

theorem evaluate_polynomial_at_3 : f 3 = 28 :=
by
  sorry

end evaluate_polynomial_at_3_l749_749600


namespace arcsin_one_half_l749_749827

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749827


namespace range_of_squared_function_l749_749577

theorem range_of_squared_function (x : ℝ) (hx : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end range_of_squared_function_l749_749577


namespace smallest_integer_with_18_divisors_l749_749629

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749629


namespace cardinality_inequality_l749_749159

variables (S : Finset (ℝ × ℝ × ℝ))
noncomputable def S_x : Finset (ℝ × ℝ) := S.image (λ p, (p.1, p.2.2))
noncomputable def S_y : Finset (ℝ × ℝ) := S.image (λ p, (p.1, p.2.1))
noncomputable def S_z : Finset (ℝ × ℝ) := S.image (λ p, (p.2.1, p.2.2))

theorem cardinality_inequality : 
  (S.card)^2 ≤ (S_x S).card * (S_y S).card * (S_z S).card :=
sorry

end cardinality_inequality_l749_749159


namespace symmetry_about_origin_l749_749227

noncomputable def f (x : ℝ) : ℝ := x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -g (-x) :=
by
  sorry

end symmetry_about_origin_l749_749227


namespace find_lambda_l749_749011

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749011


namespace arcsin_one_half_eq_pi_six_l749_749809

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l749_749809


namespace find_x_values_l749_749366

theorem find_x_values (x : ℝ) :
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x ∈ Set.Ici 2 ∨ x ∈ Set.Iic (-4)) := by
sorry

end find_x_values_l749_749366


namespace x1_x2_gt_one_l749_749428

noncomputable def f (x : ℝ) : ℝ := log x - x

def g (m x : ℝ) : ℝ := log x + 1/(2*x) - m

theorem x1_x2_gt_one {m x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (h_x1_lt_x2 : x1 < x2)
  (hx1_zero : g m x1 = 0) (hx2_zero : g m x2 = 0) : x1 + x2 > 1 := by
  sorry

end x1_x2_gt_one_l749_749428


namespace count_valid_three_digit_numbers_l749_749961

def valid_digit (d : ℕ) : Prop :=
  d ≠ 5 ∧ d ≠ 6 ∧ d ≠ 7 ∧ d ≠ 9

def three_digit_valid (n : ℕ) : Prop :=
  let hundreds := n / 100 in
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  (hundreds ≠ 0) ∧ valid_digit hundreds ∧ valid_digit tens ∧ valid_digit units

theorem count_valid_three_digit_numbers : 
  (Fin₃.cardinal { n // three_digit_valid n } = 180) :=
sorry

end count_valid_three_digit_numbers_l749_749961


namespace BDE_not_equilateral_l749_749491

open EuclideanGeometry

-- Definitions to be used
variables {P A B C D E : Point}

-- let tetrahedron P ABC be a regular tetrahedron
def regular_tetrahedron (P A B C : Point) : Prop :=
  tetrahedron P A B C ∧ ∀ (X Y : Point), X ∈ {P, A, B, C} → Y ∈ {P, A, B, C} → (X = Y ∨ dist X Y = dist P A)

-- D is the midpoint of PA
def midpoint_PA (P A D : Point) : Prop :=
  D = midpoint P A

-- E is the midpoint of AC
def midpoint_AC (A C E : Point) : Prop :=
  E = midpoint A C

-- The theorem statement
theorem BDE_not_equilateral (hT : regular_tetrahedron P A B C)
  (hD : midpoint_PA P A D)
  (hE : midpoint_AC A C E) : ¬ is_equilateral_triangle B D E :=
sorry

end BDE_not_equilateral_l749_749491


namespace tangent_line_to_circle_l749_749913

-- Definitions of the problem conditions.
def circle (x y : ℝ) := x^2 + y^2 = 1
def external_point := (1 : ℝ, 2 : ℝ)

-- Statement of the problem to be proved in Lean.
theorem tangent_line_to_circle (x y : ℝ) (tangent : x^2 + y^2 = 1) (P : ℝ × ℝ := (1, 2)) : 
  (3 * x - 4 * y + 5 = 0) ∨ (x = 1) :=
by sorry

end tangent_line_to_circle_l749_749913


namespace smallest_integer_with_18_divisors_l749_749679

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749679


namespace probability_below_x_axis_l749_749456

open Real

theorem probability_below_x_axis :
  let center : ℝ × ℝ := (2, sqrt 3),
      radius : ℝ := 2,
      circle_area : ℝ := π * radius ^ 2,
      sector_area : ℝ := 1 / 6 * circle_area,
      triangle_area : ℝ := 2 * sqrt 3,
      segment_area : ℝ := sector_area - triangle_area,
      probability : ℝ := segment_area / circle_area
  in probability = (1 / 6 - sqrt 3 / (2 * π)) := by
  sorry

end probability_below_x_axis_l749_749456


namespace chess_piece_paths_195_l749_749489

theorem chess_piece_paths_195 :
  let steps := 6
  ∃ k m, k + 2 * m = 6 ∧ 
  (∑ j in range(steps+1), if (steps - j) % 2 = 0 then combin (steps - j) (j / 2) else 0) *
  combin steps 2 = 195 :=
by {
  let steps := 6,
  use [0, 3],
  sorry
}

end chess_piece_paths_195_l749_749489


namespace finite_sets_l749_749268

open Set

theorem finite_sets : 
  ¬Finite (range Nat.succ) → ¬Finite (univ : Set ℚ) → ¬Finite (univ : Set ℝ) → Finite (complement (range Nat.succ)) :=
by
  intro h1 h2 h3
  sorry

end finite_sets_l749_749268


namespace tan_eq_860_l749_749368

theorem tan_eq_860 (n : ℤ) (hn : -180 < n ∧ n < 180) : 
  n = -40 ↔ (Real.tan (n * Real.pi / 180) = Real.tan (860 * Real.pi / 180)) := 
sorry

end tan_eq_860_l749_749368


namespace triangle_area_difference_l749_749131

-- Definitions per conditions
def right_angle (A B C : Type) (angle_EAB : Prop) : Prop := angle_EAB
def angle_ABC_eq_30 (A B C : Type) (angle_ABC : ℝ) : Prop := angle_ABC = 30
def length_AB_eq_5 (A B : Type) (AB : ℝ) : Prop := AB = 5
def length_BC_eq_7 (B C : Type) (BC : ℝ) : Prop := BC = 7
def length_AE_eq_10 (A E : Type) (AE : ℝ) : Prop := AE = 10
def lines_intersect_at_D (A B C E D : Type) (intersects : Prop) : Prop := intersects

-- Main theorem statement
theorem triangle_area_difference
  (A B C E D : Type)
  (angle_EAB : Prop)
  (right_EAB : right_angle A E B angle_EAB)
  (angle_ABC : ℝ)
  (angle_ABC_is_30 : angle_ABC_eq_30 A B C angle_ABC)
  (AB : ℝ)
  (AB_is_5 : length_AB_eq_5 A B AB)
  (BC : ℝ)
  (BC_is_7 : length_BC_eq_7 B C BC)
  (AE : ℝ)
  (AE_is_10 : length_AE_eq_10 A E AE)
  (intersects : Prop)
  (intersects_at_D : lines_intersect_at_D A B C E D intersects) :
  (area_ADE - area_BDC) = 16.25 := sorry

end triangle_area_difference_l749_749131


namespace kids_bike_wheels_l749_749334

theorem kids_bike_wheels
  (x : ℕ) 
  (h1 : 7 * 2 + 11 * x = 58) :
  x = 4 :=
sorry

end kids_bike_wheels_l749_749334


namespace angle_ABD_is_75_l749_749516

theorem angle_ABD_is_75
  (A B C D K L : Point)
  (parallelogram_ABCD : parallelogram A B C D)
  (angle_BAD_60 : ∠ B A D = 60)
  (midpoint_K : midpoint B C K)
  (midpoint_L : midpoint C D L)
  (cyclic_ABKL : cyclicQuadrilateral A B K L) :
  ∠ A B D = 75 := 
  sorry

end angle_ABD_is_75_l749_749516


namespace smallest_integer_with_18_divisors_l749_749674

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749674


namespace arcsin_one_half_l749_749824

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749824


namespace biology_class_grades_l749_749112

theorem biology_class_grades (total_students : ℕ)
  (PA PB PC PD : ℕ)
  (h1 : PA = 12 * PB / 10)
  (h2 : PC = PB)
  (h3 : PD = 5 * PB / 10)
  (h4 : PA + PB + PC + PD = total_students) :
  total_students = 40 → PB = 11 := 
by
  sorry

end biology_class_grades_l749_749112


namespace equilateral_triangle_ratio_l749_749610

theorem equilateral_triangle_ratio (s : ℝ) (h_s : s = 12) : 
  (let A := (√3 * s^2) / 4 in let P := 3 * s in A / P = √3) :=
by
  sorry

end equilateral_triangle_ratio_l749_749610


namespace johnny_yellow_picks_l749_749144

variable (total_picks red_picks blue_picks yellow_picks : ℕ)

theorem johnny_yellow_picks
    (h_total_picks : total_picks = 3 * blue_picks)
    (h_half_red_picks : red_picks = total_picks / 2)
    (h_blue_picks : blue_picks = 12)
    (h_pick_sum : total_picks = red_picks + blue_picks + yellow_picks) :
    yellow_picks = 6 := by
  sorry

end johnny_yellow_picks_l749_749144


namespace electric_sharpener_time_l749_749296

theorem electric_sharpener_time
  (time_hand_crank : ℕ) (time_in_seconds : ℕ) (extra_pencils : ℕ)
  (total_pencils_hand_crank : ℕ) (total_pencils_electric : ℕ) : 
  time_hand_crank = 45 →
  time_in_seconds = 360 →
  extra_pencils = 10 →
  total_pencils_hand_crank = time_in_seconds / time_hand_crank →
  total_pencils_electric = total_pencils_hand_crank + extra_pencils →
  ∃ (time_electric : ℕ), total_pencils_electric = time_in_seconds / time_electric ∧ time_electric = 20 :=
begin
  sorry
end

end electric_sharpener_time_l749_749296


namespace chinese_horses_problem_l749_749214

variables (x y : ℕ)

theorem chinese_horses_problem (h1 : x + y = 100) (h2 : 3 * x + (y / 3) = 100) :
  (x + y = 100) ∧ (3 * x + (y / 3) = 100) :=
by
  sorry

end chinese_horses_problem_l749_749214


namespace pyramid_volume_correct_l749_749115

-- Define the dimensions and points
variable (AB BC CG : ℝ) (M E : ℝ × ℝ × ℝ)

-- Define the conditions
def dimensions_conditions : Prop := (AB = 4) ∧ (BC = 2) ∧ (CG = 3)
def midpoint_M_condition : Prop := M = (∣BC / 2,  ∣CG / 2,  3 / 2)
def point_E_condition : Prop := E = (1, 0, 0)

-- Complete condition combining all
def all_conditions : Prop := dimensions_conditions AB BC CG ∧ midpoint_M_condition M ∧ point_E_condition E

-- Define the base area and height
def base_area (BC CF : ℝ) : ℝ := BC * CF
def height (CG : ℝ) : ℝ := CG / 2

-- The pyramid volume
def pyramid_volume (AB BC CG : ℝ) (M : ℝ × ℝ × ℝ) (E : ℝ × ℝ × ℝ) : ℝ :=
  if all_conditions AB BC CG M E then
    1 / 3 * base_area BC CG * height CG
  else 0

-- The theorem statement to prove
theorem pyramid_volume_correct (AB BC CG : ℝ) (M E : ℝ × ℝ × ℝ) :
  all_conditions AB BC CG M E → pyramid_volume AB BC CG M E = 3 := by
  sorry

end pyramid_volume_correct_l749_749115


namespace minimum_odd_integers_l749_749258

theorem minimum_odd_integers {a b c d e f : ℤ} 
  (h1 : a + b = 26) 
  (h2 : a + b + c + d = 41) 
  (h3 : a + b + c + d + e + f = 57) : 
  let odd_count := (ite (a % 2 = 1) 1 0) +
                   (ite (b % 2 = 1) 1 0) +
                   (ite (c % 2 = 1) 1 0) +
                   (ite (d % 2 = 1) 1 0) +
                   (ite (e % 2 = 1) 1 0) +
                   (ite (f % 2 = 1) 1 0) 
  in odd_count ≥ 1 :=
by {
  sorry
}

end minimum_odd_integers_l749_749258


namespace count_positive_integers_in_range_sq_le_count_positive_integers_in_range_sq_l749_749383

theorem count_positive_integers_in_range_sq_le (x : ℕ) : 
  225 ≤ x^2 ∧ x^2 ≤ 400 → x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20 :=
by { sorry }

theorem count_positive_integers_in_range_sq : 
  { x : ℕ | 225 ≤ x^2 ∧ x^2 ≤ 400 }.to_finset.card = 6 :=
by { sorry }

end count_positive_integers_in_range_sq_le_count_positive_integers_in_range_sq_l749_749383


namespace equilateral_triangle_ratio_correct_l749_749615

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l749_749615


namespace how_many_more_choc_chip_cookies_l749_749444

-- Define the given conditions
def choc_chip_cookies_yesterday := 19
def raisin_cookies_this_morning := 231
def choc_chip_cookies_this_morning := 237

-- Define the total chocolate chip cookies
def total_choc_chip_cookies : ℕ := choc_chip_cookies_this_morning + choc_chip_cookies_yesterday

-- Define the proof statement
theorem how_many_more_choc_chip_cookies :
  total_choc_chip_cookies - raisin_cookies_this_morning = 25 :=
by
  -- Proof will go here
  sorry

end how_many_more_choc_chip_cookies_l749_749444


namespace selling_price_41_l749_749288

-- Purchase price per item
def purchase_price : ℝ := 30

-- Government restriction on pice increase: selling price cannot be more than 40% increase of the purchase price
def price_increase_restriction (a : ℝ) : Prop :=
  a <= purchase_price * 1.4

-- Profit condition equation
def profit_condition (a : ℝ) : Prop :=
  (a - purchase_price) * (112 - 2 * a) = 330

-- The selling price of each item that satisfies all conditions is 41 yuan  
theorem selling_price_41 (a : ℝ) (h1 : profit_condition a) (h2 : price_increase_restriction a) :
  a = 41 := sorry

end selling_price_41_l749_749288


namespace time_to_empty_tank_l749_749774

-- Conditions
constant four_fifths_full : ℚ := 4/5
constant rate_pipe_a : ℚ := 1/10
constant rate_pipe_b : ℚ := 1/6

-- Question stated as a goal to prove
theorem time_to_empty_tank :
  let combined_rate := rate_pipe_a - rate_pipe_b in
  let time_to_empty := four_fifths_full / (- combined_rate) in
  time_to_empty = 12 := 
by
  sorry

end time_to_empty_tank_l749_749774


namespace parallel_lines_l749_749280

-- Definitions of lines and plane
variable {Line : Type}
variable {Plane : Type}
variable (a b c : Line)
variable (α : Plane)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlane : Line → Plane → Prop)

-- Given conditions
variable (h1 : parallel a c)
variable (h2 : parallel b c)

-- Theorem statement
theorem parallel_lines (a b c : Line) 
                       (α : Plane) 
                       (parallel : Line → Line → Prop) 
                       (perpendicular : Line → Line → Prop) 
                       (parallelPlane : Line → Plane → Prop)
                       (h1 : parallel a c) 
                       (h2 : parallel b c) : 
                       parallel a b :=
sorry

end parallel_lines_l749_749280


namespace car_fuel_efficiency_l749_749287

theorem car_fuel_efficiency (x : ℝ) (h1 : 0.8 * (15 * (x / 0.8)) = 0.8 * (15 * x + 105)) : x = 28 :=
by
  have h2 : 15 * (x / 0.8) = 15 * x + 105,
    from (h1).trans (by simp [mul_add, mul_comm, mul_assoc]),
  sorry

end car_fuel_efficiency_l749_749287


namespace multiples_of_three_l749_749589

theorem multiples_of_three : ∃ n : ℕ, (∀ k, 1 ≤ k ∧ k ≤ 67 → 3 * k + 99 = (list.range (n - 100 + 1)).count (λ x, x % 3 = 0)) → n = 300 :=
sorry

end multiples_of_three_l749_749589


namespace cosine_of_angle_l749_749898

theorem cosine_of_angle (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) : 
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_of_angle_l749_749898


namespace quadrilateral_area_l749_749291

theorem quadrilateral_area 
  (h1 : ∀ (x y : ℝ), x^4 + y^4 = 100 → xy = 4) :
  ∃ (A : ℝ), A = 4 * real.sqrt 17 :=
sorry

end quadrilateral_area_l749_749291


namespace find_m_l749_749951

noncomputable def sqrt_2 := Real.sqrt 2

def circle_eq (x y : ℝ) := x^2 + y^2 = 4 * x

def line_eq (x y m t : ℝ) := x = sqrt_2 / 2 * t + m ∧ y = sqrt_2 / 2 * t

def is_tangent (c_x c_y : ℝ) (r : ℝ) (A B C : ℝ) : Prop :=
  abs (A * c_x + B * c_y + C) = r * sqrt (A^2 + B^2)

theorem find_m :
  ∀ (m : ℝ),
    (∃ t x y, line_eq x y m t) →
    (∃ x y, circle_eq x y) →
    ∀ c_x c_y r A B C,
      is_tangent c_x c_y r A B C →
      (m = 2 + 2 * sqrt_2 ∨ m = 2 - 2 * sqrt_2) :=
begin
  sorry
end

end find_m_l749_749951


namespace volume_of_displaced_water_l749_749746

-- Defining the conditions of the problem
def cube_side_length : ℝ := 6
def cyl_radius : ℝ := 5
def cyl_height : ℝ := 12
def cube_volume (s : ℝ) : ℝ := s^3

-- Statement: The volume of water displaced by the cube when it is fully submerged in the barrel
theorem volume_of_displaced_water :
  cube_volume cube_side_length = 216 := by
  sorry

end volume_of_displaced_water_l749_749746


namespace find_lambda_l749_749001

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749001


namespace popsicle_stick_count_l749_749544

variable (Sam Sid Steve : ℕ)

def number_of_sticks (Sam Sid Steve : ℕ) : ℕ :=
  Sam + Sid + Steve

theorem popsicle_stick_count 
  (h1 : Sam = 3 * Sid)
  (h2 : Sid = 2 * Steve)
  (h3 : Steve = 12) :
  number_of_sticks Sam Sid Steve = 108 :=
by
  sorry

end popsicle_stick_count_l749_749544


namespace area_of_AMDN_eq_ABC_l749_749778

open EuclideanGeometry Real

-- Definitions for the constructed points and perpendiculars
def is_acute_triangle (A B C : Point) : Prop := 
  ∠BAC < π / 2 ∧ ∠ABC < π / 2 ∧ ∠BCA < π / 2

def is_perpendicular (P Q R : Point) : Prop := 
  ∠(QPA) = π / 2

-- In any Euclidean geometry context
theorem area_of_AMDN_eq_ABC
  (A B C E F D M N : Point)
  (h_acute : is_acute_triangle A B C)
  (h_angle_eq : ∠BAE = ∠CAF)
  (h_perpendicular_FM : is_perpendicular F M A B)
  (h_perpendicular_FN : is_perpendicular F N A C)
  (h_D_on_circumcircle : cyclic A B C D) :
  area (quadrilateral A M D N) = area (triangle A B C) :=
sorry

end area_of_AMDN_eq_ABC_l749_749778


namespace smallest_positive_integer_with_18_divisors_l749_749671

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749671


namespace smallest_integer_with_18_divisors_l749_749689

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749689


namespace roman_numeral_calculation_l749_749199

def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end roman_numeral_calculation_l749_749199


namespace brothers_selection_probability_l749_749113

theorem brothers_selection_probability :
  let P_A := 1 / 7 : ℚ
  let P_B := 2 / 5 : ℚ
  let P_Xi := 3 / 4 : ℚ
  let P_Xt := 4 / 9 : ℚ
  let P_Yi := 5 / 8 : ℚ
  let P_Yt := 7 / 10 : ℚ
  let P_X := P_A * P_Xi * P_Xt
  let P_Y := P_B * P_Yi * P_Yt
  P_X * P_Y = 7 / 840 :=
  by
    let P_A := 1 / 7 : ℚ
    let P_B := 2 / 5 : ℚ
    let P_Xi := 3 / 4 : ℚ
    let P_Xt := 4 / 9 : ℚ
    let P_Yi := 5 / 8 : ℚ
    let P_Yt := 7 / 10 : ℚ
    let P_X := P_A * P_Xi * P_Xt
    let P_Y := P_B * P_Yi * P_Yt
    have : P_X = 1 / 21 := by
      sorry
    have : P_Y = 7 / 40 := by
      sorry
    show P_X * P_Y = 7 / 840
    sorry

end brothers_selection_probability_l749_749113


namespace part_one_part_two_l749_749526

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem part_one (h : f x ≤ 5) : -2 ≤ x ∧ x ≤ 3 :=
  sorry

theorem part_two {x m : ℝ} (hx : 0 ≤ x ∧ x ≤ 2) (hf : f x = 3) (h_ge : f x ≥ -x^2 + 2x + m) : m ≤ 2 :=
  sorry

end part_one_part_two_l749_749526


namespace operation_X_value_l749_749461

def operation_X (a b : ℤ) : ℤ := b + 7 * a - a^3 + 2 * b

theorem operation_X_value : operation_X 4 3 = -27 := by
  sorry

end operation_X_value_l749_749461


namespace smallest_integer_with_18_divisors_l749_749637

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749637


namespace slope_angle_of_l1_distance_to_l2_is_sqrt3_div2_l749_749440

-- Define line l1 and line l2
def line_l1 (x y : ℝ) : Prop := √3 * x + y - 1 = 0
def line_l2 (a x y : ℝ) : Prop := a * x + y = 1

-- Define the property of perpendicular lines
def are_perpendicular (a b : ℝ) : Prop := a * b = -1

-- Define the property of distance from a point to the line
def distance_from_origin (a : ℝ) : ℝ := abs(0 + 0 - 1) / sqrt((a^2) + 1)

-- Prove the slope angle of l1 is 2π/3
theorem slope_angle_of_l1 : 
  ∀ (α : ℝ), (α = atan(-√3) ∨ α = atan(-√3) + π) → α = 2 * π / 3 := 
by sorry

-- Prove the distance from the origin to l2 is sqrt(3)/2
theorem distance_to_l2_is_sqrt3_div2 :
  ∀ a : ℝ, are_perpendicular (√3) a → distance_from_origin a = √3 / 2 :=
by sorry

end slope_angle_of_l1_distance_to_l2_is_sqrt3_div2_l749_749440


namespace smallest_positive_integer_with_18_divisors_l749_749696

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749696


namespace deer_families_initial_count_l749_749250

theorem deer_families_initial_count (stayed moved_out : ℕ) (h_stayed : stayed = 45) (h_moved_out : moved_out = 34) :
  stayed + moved_out = 79 :=
by
  sorry

end deer_families_initial_count_l749_749250


namespace angle_AMB_eq_72_l749_749561

-- Given conditions
variables (A B C D M : Type)
           [is_circle A B C D]  -- A to D are points on a circle
           (AB BC CD DA : ℝ)    -- arc lengths
           (ratio_cond : AB / BC = 3 / 2 ∧ BC / CD = 2 / 13 ∧ CD / DA = 13 / 7)

-- Definition of the specific arc lengths following the given ratio
def arc_AB := 3 * (360 / 25)
def arc_BC := 2 * (360 / 25)
def arc_CD := 13 * (360 / 25)
def arc_DA := 7 * (360 / 25)

-- Total circle condition
lemma total_circle_sum : arc_AB + arc_BC + arc_CD + arc_DA = 360 :=
by
  -- simplifying to prove the total sum is 360°
  unfold arc_AB arc_BC arc_CD arc_DA
  simp [mul_div_cancel', add_assoc, eq_sub_iff_add_eq]

-- The main theorem to prove the specific angle
theorem angle_AMB_eq_72 
  (extens_AD_BC_interact_at_M : extends_to_intersect A D B C M) :
  ∠AMB = 72 :=
by
  sorry

end angle_AMB_eq_72_l749_749561


namespace lambda_solution_l749_749068

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749068


namespace polar_to_cartesian_equiv_l749_749348

noncomputable def polar_to_cartesian (rho theta : ℝ) : Prop :=
  let x := rho * Real.cos theta
  let y := rho * Real.sin theta
  (Real.sqrt 3 * x + y = 2) ↔ (rho * Real.cos (theta - Real.pi / 6) = 1)

theorem polar_to_cartesian_equiv (rho theta : ℝ) : polar_to_cartesian rho theta :=
by
  sorry

end polar_to_cartesian_equiv_l749_749348


namespace cranberry_parts_l749_749499

theorem cranberry_parts (L C : ℕ) :
  L = 3 →
  L + C = 72 →
  C = L + 18 →
  C = 21 :=
by
  intros hL hSum hDiff
  sorry

end cranberry_parts_l749_749499


namespace second_swimmer_longer_l749_749599

-- Define the lengths of the sides of the rectangle
variables (a b : ℝ) (h_a : a > 0) (h_b : b > 0)

-- Define the starting point of the second swimmer that divides a side in the ratio 2018:2019
variables (N : ℝ) (h_N : N > 0) (ratio_division : N = a * 2018 / (2018 + 2019))

-- Define the points K, M, L on other sides
variables (K M L : ℝ)

-- Calculate the diagonal length
noncomputable def diagonal_length := 2 * (Real.sqrt (a^2 + b^2))

-- Define the lengths of edges of the quadrilateral path
variables (NK KL LM MN : ℝ)

-- Assuming the swimmer swims along these paths
variables (path_length : NK + KL + LM + MN = NK + LM + MN + NK)

-- Theorem statement to prove that the second swimmer's path cannot be shorter than first swimmer's path
theorem second_swimmer_longer (h_path_eq : NK + KL + LM + MN ≥ diagonal_length) : 
  (NK + KL + LM + MN ≥ diagonal_length) :=
sorry

end second_swimmer_longer_l749_749599


namespace unique_function_solution_l749_749843

noncomputable def f (x : ℝ) : ℝ := 1 - x^2 / 2

theorem unique_function_solution :
  ∀ (f : ℝ → ℝ),
  (∀ (x y : ℝ), f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1) →
  f = (λ x, 1 - x^2 / 2) :=
begin
  intros f h,
  ext x,
  have : f 0 = 1 := sorry,
  have : f (-1) = -1 := sorry,
  sorry
end

end unique_function_solution_l749_749843


namespace smallest_integer_with_18_divisors_l749_749653

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749653


namespace question_1_question_2_question_3_l749_749404

section sequence_proof

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {c : ℕ → ℝ} {d : ℕ → ℝ} {T : ℕ → ℝ}

-- Preconditions
axiom a_pos : ∀ n : ℕ, a n > 0
axiom a_1 : a 1 = 1
axiom a_induction : ∀ n : ℕ, a (n + 1) ^ 2 - 1 = 4 * a n * (a n + 1)
axiom b_def : ∀ n : ℕ, b n = 2 * log (2, 1 + a n) - 1

-- Questions
theorem question_1 : (∀ n : ℕ, a n = 2 ^ n - 1) :=
sorry

theorem question_2 : (∑ i in Finset.range 100, c i) = 11116 :=
sorry

theorem question_3 (n m : ℕ) (h1 : 1 < m < n)
  (h2: T 1 = 1 / 3) (h3 : T n = n / (2 * n + 1))
  : (T 1, T m, T n) in (geometric_sequence T n) :=
b := sorry

end sequence_proof

end question_1_question_2_question_3_l749_749404


namespace escalator_steps_l749_749189

theorem escalator_steps
  (steps_ascending : ℤ)
  (steps_descending : ℤ)
  (ascend_units_time : ℤ)
  (descend_units_time : ℤ)
  (speed_ratio : ℤ)
  (equation : ((steps_ascending : ℚ) / (1 + (ascend_units_time : ℚ))) = ((steps_descending : ℚ) / ((descend_units_time : ℚ) * speed_ratio)) )
  (solution_x : (125 * 0.6 = 75)) : 
  (steps_ascending * (1 + 0.6 : ℚ) = 120) :=
by
  sorry

end escalator_steps_l749_749189


namespace intersection_x_val_l749_749245

theorem intersection_x_val (x y : ℝ) (h1 : y = 3 * x - 24) (h2 : 5 * x + 2 * y = 102) : x = 150 / 11 :=
by
  sorry

end intersection_x_val_l749_749245


namespace ratio_area_perimeter_eq_sqrt3_l749_749622

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l749_749622


namespace largest_circle_area_l749_749761

noncomputable def rectangle_to_circle_area (w : ℝ) (h : ℝ) (P : ℝ) (A_rect : ℝ) (C : ℝ) : ℝ :=
  let r := C / (2 * Real.pi)
  in Real.pi * r^2

theorem largest_circle_area (w h : ℝ) (H : 2 * w = h) (A_rect : w * h = 200) (P : 2 * (w + h)) : 
  Int.round (rectangle_to_circle_area w h P (w * h) P) = 287 :=
by
  sorry

end largest_circle_area_l749_749761


namespace smallest_integer_with_18_divisors_l749_749650

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749650


namespace unique_ordered_triples_count_l749_749403

theorem unique_ordered_triples_count :
  ∃ (n : ℕ), n = 1 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  abc = 4 * (ab + bc + ca) ∧ a = c / 4 -> False :=
sorry

end unique_ordered_triples_count_l749_749403


namespace locus_of_M_midpoint_line_l749_749995

theorem locus_of_M_midpoint_line 
  {A B C M : Point} (right_triangle: right_angle A B C) 
  (H: dist_sq M B + dist_sq M C = 2 * dist_sq M A) : ∃ line : Set Point, midpoint_line_through A B C line ∧ M ∈ line := 
sorry

def right_angle (A B C : Point) : Prop := sorry
def dist_sq (P Q : Point) : ℝ := sorry
def midpoint (P Q : Point) (M : Point) : Prop := sorry
def midpoint_line_through (A B C : Point) (line : Set Point) : Prop := sorry

structure Point :=
  (x : ℝ)
  (y : ℝ)

end locus_of_M_midpoint_line_l749_749995


namespace max_digits_product_l749_749737

def digitsProduct (A B : ℕ) : ℕ := A * B

theorem max_digits_product 
  (A B : ℕ) 
  (h1 : A + B + 5 ≡ 0 [MOD 9]) 
  (h2 : 0 ≤ A ∧ A ≤ 9) 
  (h3 : 0 ≤ B ∧ B ≤ 9) 
  : digitsProduct A B = 42 := 
sorry

end max_digits_product_l749_749737


namespace probability_three_heads_in_tosses_l749_749719

noncomputable def binomial_probability (n k : ℕ) (p q : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * (q^(n-k))

theorem probability_three_heads_in_tosses (n k : ℕ) (p q : ℝ) (h_n : n = 4) (h_k : k = 3) (h_p : p = 0.5) (h_q : q = 0.5) :
  binomial_probability n k p q = 0.25 :=
by
  sorry

end probability_three_heads_in_tosses_l749_749719


namespace remainder_of_power_mod_l749_749865

theorem remainder_of_power_mod (a b n : ℕ) (h_prime : Nat.Prime n) (h_a_not_div : ¬ (n ∣ a)) :
  a ^ b % n = 82 :=
by
  have : n = 379 := sorry
  have : a = 6 := sorry
  have : b = 97 := sorry
  sorry

end remainder_of_power_mod_l749_749865


namespace arithmetic_sequence_a3_l749_749508

variable {a : ℕ → ℝ}  -- Define the sequence as a function from natural numbers to real numbers.

-- Definition that the sequence is arithmetic.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- The given condition in the problem
axiom h1 : a 1 + a 5 = 6

-- The statement to prove
theorem arithmetic_sequence_a3 (h : is_arithmetic_sequence a) : a 3 = 3 :=
by {
  -- The proof is omitted.
  sorry
}

end arithmetic_sequence_a3_l749_749508


namespace find_lambda_l749_749040

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749040


namespace five_digit_palindrome_div_by_11_probability_l749_749750

theorem five_digit_palindrome_div_by_11_probability :
  let palindrome_count := 9 * 10 * 10
  let favorable_palindrome_count := 90
  let probability := (favorable_palindrome_count : ℚ) / palindrome_count
  in probability = (1 : ℚ) / 10 := by
    sorry

end five_digit_palindrome_div_by_11_probability_l749_749750


namespace count_sets_without_perfect_square_l749_749506

-- Define the set Ti
def T (i : ℕ) : set ℕ := {n : ℕ | 50*i ≤ n ∧ n < 50*(i + 1)}

-- Define a predicate to check if a set contains a perfect square
def contains_perfect_square (s : set ℕ) : Prop :=
  ∃ x : ℕ, x*x ∈ s

-- Define the main theorem
theorem count_sets_without_perfect_square :
  let sets := {i : ℕ | i ≤ 1999} in
  let sets_with_perfect_square := {i | contains_perfect_square (T i)} in
  let total_sets := 2000 in -- Total sets from T_0 to T_1999
  let sets_with_no_perfect_square := total_sets - (finset.card sets_with_perfect_square) in
  sets_with_no_perfect_square = 1733 :=
by
  sorry

end count_sets_without_perfect_square_l749_749506


namespace arcsin_one_half_l749_749798

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749798


namespace max_expected_usable_balloons_l749_749533

theorem max_expected_usable_balloons :
  let price_small := 4
  let count_small := 50
  let pop_prob_small := 0.1
  let price_medium := 6
  let count_medium := 75
  let pop_prob_medium := 0.05
  let price_large := 12
  let count_large := 200
  let pop_prob_large := 0.01
  let budget := 24
  let max_bags := 5 in
  let expected_balloons (count: ℕ) (pop_prob: ℝ) : ℝ := count * (1 - pop_prob) in
  let expected_small := expected_balloons count_small pop_prob_small
  let expected_medium := expected_balloons count_medium pop_prob_medium
  let expected_large := expected_balloons count_large pop_prob_large in
  ∃ (n_small n_medium n_large: ℕ),
    n_small + n_medium + n_large ≤ max_bags ∧
    n_small * price_small + n_medium * price_medium + n_large * price_large ≤ budget ∧
    n_large = 2 ∧
    n_medium = 0 ∧
    n_small = 0 →
    n_large * expected_large + n_medium * expected_medium + n_small * expected_small = 396 :=
by
  let price_small := 4
  let count_small := 50
  let pop_prob_small := 0.1
  let price_medium := 6
  let count_medium := 75
  let pop_prob_medium := 0.05
  let price_large := 12
  let count_large := 200
  let pop_prob_large := 0.01
  let budget := 24
  let max_bags := 5
  let expected_balloons (count: ℕ) (pop_prob: ℝ) : ℝ := count * (1 - pop_prob)
  let expected_small := expected_balloons count_small pop_prob_small
  let expected_medium := expected_balloons count_medium pop_prob_medium
  let expected_large := expected_balloons count_large pop_prob_large
  use [0, 0, 2]
  simp
  rfl

end max_expected_usable_balloons_l749_749533


namespace min_n_for_factorization_l749_749372

theorem min_n_for_factorization (n : ℤ) :
  (∃ A B : ℤ, 6 * A * B = 60 ∧ n = 6 * B + A) → n = 66 :=
sorry

end min_n_for_factorization_l749_749372


namespace pentagon_PT_length_l749_749120

theorem pentagon_PT_length (QR RS ST : ℝ) (angle_T right_angle_QRS T : Prop) (length_PT := (fun (a b : ℝ) => a + 3 * Real.sqrt b)) :
  QR = 3 →
  RS = 3 →
  ST = 3 →
  angle_T →
  right_angle_QRS →
  (angle_Q angle_R angle_S : ℝ) →
  angle_Q = 135 →
  angle_R = 135 →
  angle_S = 135 →
  ∃ (a b : ℝ), length_PT a b = 6 * Real.sqrt 2 ∧ a + b = 2 :=
by
  sorry

end pentagon_PT_length_l749_749120


namespace LarryTerryCafe_l749_749498

variable (total_money : ℝ) (cost_sandwich : ℝ) (cost_coffee : ℝ)
variable (max_sandwich : ℕ) (remaining_money : ℝ) (max_coffee : ℕ)
variable (total_items : ℕ)

theorem LarryTerryCafe : 
  total_money = 50.0 ∧ 
  cost_sandwich = 3.5 ∧ 
  cost_coffee = 1.5 ∧ 
  max_sandwich = floor (total_money / cost_sandwich).toNat ∧ 
  remaining_money = total_money - (max_sandwich * cost_sandwich) ∧ 
  max_coffee = floor (remaining_money / cost_coffee).toNat ∧ 
  total_items = max_sandwich + max_coffee 
  → total_items = 14 :=
by
  intros
  sorry

end LarryTerryCafe_l749_749498


namespace max_value_of_f_l749_749414

-- Define the function f(x) = x * (5 - 4x)
def f (x : ℝ) : ℝ := x * (5 - 4 * x)

theorem max_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 5 / 4) :
  ∃ (x_max : ℝ), f(x_max) = 25 / 16 ∧ (∀ y, 0 < y → y < 5 / 4 → f(y) ≤ f(x_max)) :=
by
  sorry

end max_value_of_f_l749_749414


namespace driver_total_miles_per_week_l749_749748

theorem driver_total_miles_per_week :
  let distance_monday_to_saturday := (30 * 3 + 25 * 4 + 40 * 2) * 6
  let distance_sunday := 35 * (5 - 1)
  distance_monday_to_saturday + distance_sunday = 1760 := by
  sorry

end driver_total_miles_per_week_l749_749748


namespace lambda_value_l749_749034

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749034


namespace incorrect_statement_l749_749973

theorem incorrect_statement (a : ℝ) (x : ℝ) (h : a > 1) :
  ¬((x = 0 → a^x = 1) ∧
    (x = 1 → a^x = a) ∧
    (x = -1 → a^x = 1/a) ∧
    (x < 0 → 0 < a^x ∧ ∀ ε > 0, ∃ x' < x, a^x' < ε)) :=
sorry

end incorrect_statement_l749_749973


namespace not_possible_results_l749_749429

variable (a b : ℝ) (c : ℤ)

def f (x : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + c

theorem not_possible_results (a b : ℝ) (c : ℤ) :
  ¬ (f a b c 2 = 3 ∧ f a b c (-2) = 6) :=
by 
  sorry

end not_possible_results_l749_749429


namespace cone_cylinder_volume_ratio_l749_749313

noncomputable def volume_ratio_cone_to_cylinder (r h : ℝ) (hr : r > 0) (hh : h > 0) : ℝ :=
  let volume_cone := (1 / 3) * π * r^2 * h
  let volume_cylinder := π * r^2 * h
  volume_cone / volume_cylinder

theorem cone_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  volume_ratio_cone_to_cylinder r h hr hh = (1 / 3) := by
  sorry

end cone_cylinder_volume_ratio_l749_749313


namespace smallest_degree_of_polynomial_with_roots_l749_749555

def polynomial_with_roots_in_Q (f : ℚ[X]) (r : ℝ) : Prop :=
  is_root f r ∧ is_root f r.conj

theorem smallest_degree_of_polynomial_with_roots :
  ∀ (f : ℚ[X]),
  (polynomial_with_roots_in_Q f (3 - sqrt 8) ∧ polynomial_with_roots_in_Q f (5 + sqrt 11) ∧
   polynomial_with_roots_in_Q f (15 - sqrt 28) ∧ polynomial_with_roots_in_Q f (-2 - sqrt 3)) →
  nat_degree f = 8 :=
sorry

end smallest_degree_of_polynomial_with_roots_l749_749555


namespace loss_denomination_l749_749457

theorem loss_denomination (profit_pos : ∀ (profit : ℕ), "+" profit = profit) :
  (lost : ℕ), "-" lost = -lost :=
begin
  assume lost,
  sorry
end

end loss_denomination_l749_749457


namespace golden_ratio_sum_l749_749178

noncomputable def golden_ratio_a : ℝ := (Real.sqrt 5 - 1) / 2
noncomputable def golden_ratio_b : ℝ := (Real.sqrt 5 + 1) / 2

noncomputable def S (n : ℕ) := 
  n / (1 + golden_ratio_a^n) + n / (1 + golden_ratio_b^n)

theorem golden_ratio_sum :
  ∑ i in Finset.range 100, S (i + 1) = 5050 :=
begin
  -- Proof steps go here
  sorry
end

end golden_ratio_sum_l749_749178


namespace cube_rolls_probability_l749_749292

theorem cube_rolls_probability :
  ∃ n, (∀ (a b : ℕ), a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} →
  (a + b).prime → r = 15 / 36) → n = 2 :=
sorry

end cube_rolls_probability_l749_749292


namespace flower_bee_relationship_l749_749588

def numberOfBees (flowers : ℕ) (fewer_bees : ℕ) : ℕ :=
  flowers - fewer_bees

theorem flower_bee_relationship :
  numberOfBees 5 2 = 3 := by
  sorry

end flower_bee_relationship_l749_749588


namespace exists_student_with_conspirators_l749_749315

noncomputable def student_with_conspirators (n : ℕ) (hn : n ≥ 4) : Prop :=
  ∃ k : ℕ, k ≥ n.root 3 * ((n - 1) * (n - 2)).root 3

theorem exists_student_with_conspirators {n : ℕ} (hn : n ≥ 4) :
  student_with_conspirators n hn :=
sorry

end exists_student_with_conspirators_l749_749315


namespace log_a_inequality_l749_749883

theorem log_a_inequality (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) : 
  (log a (3 / 5) < 1) ↔ (0 < a ∧ a < 3 / 5) ∨ (1 < a) :=
sorry

end log_a_inequality_l749_749883


namespace conjugate_quadrant_l749_749422

noncomputable def z (θ : ℝ) : ℂ := complex.of_real (real.cos θ) + complex.i * complex.of_real (real.cos (θ + real.pi / 2))

theorem conjugate_quadrant {θ : ℝ} (h1 : θ ∈ set.Ioo (real.pi / 2) real.pi) :
  ∃ q : ℕ, q = 2 ∧ 
    (complex.re (complex.conj (z θ)) < 0) ∧ 
    (complex.im (complex.conj (z θ)) > 0) :=
by
  sorry

end conjugate_quadrant_l749_749422


namespace quadratic_eq2_is_quadratic_l749_749267

-- Conditions
def eq1 := (x : ℝ) → x^3 + 2 * x = 0
def eq2 := (x : ℝ) → x * (x - 3) = 0
def eq3 := (x : ℝ) → 1/x - x^2 = 1
def eq4 := (x y : ℝ) → y - x^2 = 4

-- Statement: proving that eq2 is a quadratic equation
theorem quadratic_eq2_is_quadratic (x : ℝ) : eq2 x ↔ (∃ (a b c : ℝ), a ≠ 0 ∧ eq2 x = a * x^2 + b * x + c := 0) := by
  sorry

end quadratic_eq2_is_quadratic_l749_749267


namespace smallest_integer_with_18_divisors_l749_749630

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749630


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749620

def equilateral_triangle (s : ℝ) : Prop :=
  s = 12

theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  ∀ (s : ℝ), equilateral_triangle s → (let P := 3 * s in
  let A := (sqrt 3 / 4) * s ^ 2 in
  A / P = sqrt 3) :=
by
  intro s hs
  rw [equilateral_triangle, hs]
  let P := 3 * 12
  let A := (sqrt 3 / 4) * 12 ^ 2
  have A_eq : A = 36 * sqrt 3 := by
    calc
      (sqrt 3 / 4) * 12 ^ 2 = (sqrt 3 / 4) * 144  : by norm_num
                      ... = (sqrt 3 * 144) / 4  : by rw div_mul_cancel
                      ... = 36 * sqrt 3         : by norm_num
  have P_eq : P = 36 := by norm_num
  rw [A_eq, P_eq]
  norm_num
  rfl

end ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749620


namespace equilateral_triangle_ratio_l749_749612

theorem equilateral_triangle_ratio (s : ℝ) (h_s : s = 12) : 
  (let A := (√3 * s^2) / 4 in let P := 3 * s in A / P = √3) :=
by
  sorry

end equilateral_triangle_ratio_l749_749612


namespace monotonicity_a_eq_1_range_of_a_l749_749931

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_a_eq_1 :
  (∀ x : ℝ, 0 < x → (exp x + 2 * x - 1 > 0)) ∧
  (∀ x : ℝ, x < 0 → (exp x + 2 * x - 1 < 0)) := sorry

-- Part 2: Range of a for f(x) ≥ 1/2 * x ^ 3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 <= x → (exp x + a * x^2 - x >= 1/2 * x^3 + 1)) ↔
  (a ≥ (7 - exp 2) / 4) := sorry

end monotonicity_a_eq_1_range_of_a_l749_749931


namespace max_altitudes_product_l749_749147

theorem max_altitudes_product (A B C : Type) (AB : ℝ) (altitude_from_C : ℝ) (h_b h_c : ℝ) :  
  (AB = 3) ∧ (altitude_from_C = 2) →
  (∃ S, S = (1/2) * AB * altitude_from_C) →
  ∃ bc, bc = (S ↔ h_b * h_c = (4 * S^2 / bc)) →
  ∃ k max_bcc, max_bcc = (144 / 25) :=
by
  sorry

end max_altitudes_product_l749_749147


namespace find_principal_l749_749721

theorem find_principal 
  (SI : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (h_SI : SI = 4052.25) 
  (h_R : R = 9) 
  (h_T : T = 5) : 
  (SI * 100) / (R * T) = 9005 := 
by 
  rw [h_SI, h_R, h_T]
  sorry

end find_principal_l749_749721


namespace tallest_is_jie_l749_749982

variable (Igor Jie Faye Goa Han : Type)
variable (Shorter : Type → Type → Prop) -- Shorter relation

axiom igor_jie : Shorter Igor Jie
axiom faye_goa : Shorter Goa Faye
axiom jie_faye : Shorter Faye Jie
axiom han_goa : Shorter Han Goa

theorem tallest_is_jie : ∀ p, p = Jie :=
by
  sorry

end tallest_is_jie_l749_749982


namespace quadratic_intersects_at_two_points_l749_749879

variable {k : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_intersects_at_two_points (k : ℝ) :
  (let a := k - 2 in
   let b := -(2 * k - 1) in
   let c := k in
   discriminant a b c > 0 ∧ a ≠ 0) ↔ k > -1 / 4 ∧ k ≠ 2 :=
by
  let a := k - 2
  let b := -(2 * k - 1)
  let c := k
  have D : discriminant a b c = 4 * k + 1 := by
    rw [discriminant, sq, neg_mul_eq_mul_neg, neg_neg, mul_assoc, mul_assoc, mul_comm (4 * k),
        mul_comm 4 1, ←mul_sub, add_comm, sq_sub, mul_sub, mul_comm]
  sorry

end quadratic_intersects_at_two_points_l749_749879


namespace arcsin_one_half_l749_749801

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749801


namespace divide_athletes_into_two_teams_l749_749479

theorem divide_athletes_into_two_teams : 
  ∑ C (10,5) / 2 = 126 :=
by sorry

end divide_athletes_into_two_teams_l749_749479


namespace sin_log_infinite_zeros_in_01_l749_749965

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end sin_log_infinite_zeros_in_01_l749_749965


namespace find_lambda_l749_749014

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749014


namespace max_area_triangle_ACD_l749_749133

-- Define vectors in the plane (as variables, axioms, or properties)
axiom V : Type
axiom zero_v : V
axiom add_v : V → V → V
axiom smul_v : ℝ → V → V
axiom dot_product_v : V → V → ℝ

-- Assume conditions as axioms
axiom AC DO AB BA BD : V
axiom condition1 : add_v AC (smul_v 2 DO) = smul_v 4 AB
axiom condition2 : 3 * dot_product_v BA BA ^ (1 / 2) = 2 * dot_product_v AB (add_v AB AC) ^ (1 / 2)
axiom condition3 : dot_product_v (add_v zero_v (smul_v (-1) BA)) BD = 9

-- Define the area of triangle ACD
noncomputable def area_triangle_ACD : ℝ :=
  let side_AB := dot_product_v AB AB ^ (1 / 2)
  let side_BO := 9 / (side_AB * dot_product_v BA BD ^ (1 / 2))
  let angle_theta := acos (dot_product_v BA BA / dot_product_v BA BD ^ (dot_product_v AB BA ^ (1 / 2)))
  4 * side_AB * side_BO * sin angle_theta

-- Define maximum area theorem
theorem max_area_triangle_ACD : area_triangle_ACD = 12 * (3^(1/2)) := by
  apply sorry

end max_area_triangle_ACD_l749_749133


namespace distance_from_E_to_AB_l749_749486

theorem distance_from_E_to_AB (AC BC BD DE : ℝ) (angle_CAB_right angle_CBD_right angle_CDE_right : Prop)
  (h_AC : AC = 3) (h_BC : BC = 5) (h_BD : BD = 12) (h_DE : DE = 84) :
  let E := (5328 / 65, -2044 / 65)
  let distance := (5328 : ℝ) / 65
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ distance = m / n ∧ (m + n) = 5393 := 
begin
  sorry
end

end distance_from_E_to_AB_l749_749486


namespace AM_MB_ratio_l749_749849

variables {A B C P M : Point}
variables (circumcircle : Circle) (tangentA tangentB : Line)

-- Conditions
axiom on_circumcircle : A ∈ circumcircle ∧ B ∈ circumcircle ∧ C ∈ circumcircle
axiom tangents : (tangentA ∩ circumcircle) = {A} ∧ (tangentB ∩ circumcircle) = {B}
axiom intersection_or_parallel : (∃ P, P ∈ tangentA ∧ P ∈ tangentB) ∨ (tangentA ∥ tangentB)
axiom M_on_AB : M ∈ Line(A, B)
axiom CM_intersects_AB : (Line(C, M) ∩ Line(A, B)) = {M}

-- Statement to prove
theorem AM_MB_ratio :
  ∃ A B C M : Point, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ M ∈ Line(A, B) ∧ C ∉ Line(A, B) ∧ (
    (Line.perpendicular (Line(C, M)) (Line(A, B)) → Line.tangent circumcircle A → Line.tangent circumcircle B → CAM_BM_ratio)) ∨
    (Line.intersect tangentA tangentB P → Line.perpendicular (Line(C, M)) (Line(A, B)) → CAM_BM_ratio) → 
    ∀ C M , Line(C, M) ∩ Line(A, B) = {M} → CAM_BM_ratio

def CAM_BM_ratio : Prop :=
  AM / BM = (AC * AC) / (BC * BC)

end AM_MB_ratio_l749_749849


namespace smallest_integer_with_18_divisors_l749_749631

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749631


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749619

def equilateral_triangle (s : ℝ) : Prop :=
  s = 12

theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  ∀ (s : ℝ), equilateral_triangle s → (let P := 3 * s in
  let A := (sqrt 3 / 4) * s ^ 2 in
  A / P = sqrt 3) :=
by
  intro s hs
  rw [equilateral_triangle, hs]
  let P := 3 * 12
  let A := (sqrt 3 / 4) * 12 ^ 2
  have A_eq : A = 36 * sqrt 3 := by
    calc
      (sqrt 3 / 4) * 12 ^ 2 = (sqrt 3 / 4) * 144  : by norm_num
                      ... = (sqrt 3 * 144) / 4  : by rw div_mul_cancel
                      ... = 36 * sqrt 3         : by norm_num
  have P_eq : P = 36 := by norm_num
  rw [A_eq, P_eq]
  norm_num
  rfl

end ratio_of_area_to_perimeter_of_equilateral_triangle_l749_749619


namespace total_rainfall_2004_l749_749468

theorem total_rainfall_2004 (average_rainfall_2003 : ℝ) (increase_percentage : ℝ) (months : ℝ) :
  average_rainfall_2003 = 36 →
  increase_percentage = 0.10 →
  months = 12 →
  (average_rainfall_2003 * (1 + increase_percentage) * months) = 475.2 :=
by
  -- The proof is left as an exercise
  sorry

end total_rainfall_2004_l749_749468


namespace unique_rectangle_with_one_non_right_angle_l749_749507

theorem unique_rectangle_with_one_non_right_angle 
    (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
    (AB_AC : dist A B = dist A C)
    (BAC_right : ∠ B A C = 90) 
    (isosceles : ∠ B C A = ∠ C B A)
    (plane : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
    ∃! R : Type, is_rectangle R 
    ∧ (∃ D E F G : R, (D = A ∨ D = B ∨ D = C ∨ D = point_across_AB A B C) 
    ∧ (E = A ∨ E = B ∨ E = C ∨ E = point_across_AB A B C) 
    ∧ (F = A ∨ F = B ∨ F = C ∨ F = point_across_AB A B C) 
    ∧ (G = A ∨ G = B ∨ G = C ∨ G = point_across_AB A B C)) 
    ∧ (angle_condition : ∃! (D E F G : R), ∠ E F G ≠ 90 ∧ ∠ G D F ≠ 90)) :=
sorry

end unique_rectangle_with_one_non_right_angle_l749_749507


namespace jose_is_12_years_older_l749_749497

theorem jose_is_12_years_older (J M : ℕ) (h1 : M = 14) (h2 : J + M = 40) : J - M = 12 :=
by
  sorry

end jose_is_12_years_older_l749_749497


namespace min_ab_l749_749396

variable (a b : ℝ)

theorem min_ab (h1 : a > 1) (h2 : b > 2) (h3 : a * b = 2 * a + b) : a + b ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end min_ab_l749_749396


namespace f_value_1982_l749_749224

-- Noncomputable definition of the function f
noncomputable def f : ℕ → ℕ

-- Axioms based on given conditions
axiom f_def1 : f 2 = 0
axiom f_def2 : f 3 > 0
axiom f_def3 : f 9999 = 3333
axiom f_prop : ∀ m n : ℕ, f (m + n) - f m - f n ∈ {0, 1}

-- The goal statement
theorem f_value_1982 : f 1982 = 660 :=
sorry

end f_value_1982_l749_749224


namespace segment_ratio_eq_l749_749168

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables (C D Q : Point)

theorem segment_ratio_eq (h : ∃ k : ℝ, k = 2 / 5 ∧ Q = -k * C + (1 + k) * D) :
  ∃ x y : ℝ, Q = x • C + y • D ∧ x = -2 / 5 ∧ y = 7 / 5 :=
by 
  obtain ⟨k, hleft, hright⟩ := h
  use [ -2 / 5, 7 / 5 ]
  rw [hright]
  exact ⟨rfl, rfl⟩

end segment_ratio_eq_l749_749168


namespace monotonicity_f_when_a_eq_1_range_of_a_l749_749930

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_f_when_a_eq_1 :
  (∀ x > 0, deriv (λ x, f x 1) x > 0) ∧ (∀ x < 0, deriv (λ x, f x 1) x < 0) :=
sorry

-- Part 2: Range of a such that f(x) ≥ 1/2 * x^3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x a ≥ 1/2 * x^3 + 1) ↔ a ≥ (7 - Real.exp 2) / 4 :=
sorry

end monotonicity_f_when_a_eq_1_range_of_a_l749_749930


namespace perimeter_of_figure_composed_of_squares_l749_749130

theorem perimeter_of_figure_composed_of_squares
  (n : ℕ)
  (side_length : ℝ)
  (square_perimeter : ℝ := 4 * side_length)
  (total_squares : ℕ := 7)
  (total_perimeter_if_independent : ℝ := square_perimeter * total_squares)
  (meet_at_vertices : ∀ i j : ℕ, i ≠ j → ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ¬(s1 = s2))
  : total_perimeter_if_independent = 28 :=
by sorry

end perimeter_of_figure_composed_of_squares_l749_749130


namespace find_lambda_l749_749085

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749085


namespace total_ingredients_used_l749_749578

theorem total_ingredients_used (water oliveOil salt : ℕ) 
  (h_ratio : water / oliveOil = 3 / 2) 
  (h_salt : water / salt = 3 / 1)
  (h_water_cups : water = 15) : 
  water + oliveOil + salt = 30 :=
sorry

end total_ingredients_used_l749_749578


namespace equilateral_triangle_ratio_correct_l749_749617

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l749_749617


namespace handshakes_between_boys_l749_749974

theorem handshakes_between_boys (n : Nat) (k : Nat) (h1 : n = 7) (h2 : k = 2) : Nat.choose n k = 21 :=
by
  rw [h1, h2]
  constructor
  exact rfl

end handshakes_between_boys_l749_749974


namespace irrational_element_in_sequence_l749_749239

theorem irrational_element_in_sequence :
  ∃ (i : ℕ), ¬ ∃ (p q : ℕ), 0 < q ∧ gcd p q = 1 ∧ a i = (p : ℝ) / q :=
sorry

end irrational_element_in_sequence_l749_749239


namespace gcd_490_910_l749_749571

theorem gcd_490_910 : Nat.gcd 490 910 = 70 :=
by
  sorry

end gcd_490_910_l749_749571


namespace pies_per_day_l749_749312

theorem pies_per_day (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by sorry

end pies_per_day_l749_749312


namespace quadratic_equation_l749_749715

-- Define the given equations
def optionA := 3 * x - 1 = 0
def optionB := x^3 - 4 * x = 3
def optionC := x^2 + 2 * x - 1 = 0
def optionD := (1 / x^2) - x + 1 = 0

-- Define a predicate to check if an equation is quadratic
def is_quadratic (eq : x^2 + b * x + c = 0) := ∃ a b c, a ≠ 0

theorem quadratic_equation : is_quadratic optionC ∧ ¬is_quadratic optionA ∧ ¬is_quadratic optionB ∧ ¬is_quadratic optionD :=
sorry

end quadratic_equation_l749_749715


namespace smallest_int_with_18_divisors_l749_749655

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749655


namespace flower_total_l749_749249

theorem flower_total (H C D : ℕ) (h1 : H = 34) (h2 : H = C - 13) (h3 : C = D + 23) : 
  H + C + D = 105 :=
by 
  sorry  -- Placeholder for the proof

end flower_total_l749_749249


namespace permutations_without_HMMT_l749_749960

noncomputable def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem permutations_without_HMMT :
  let total_permutations := multinomial 8 2 2 4
  let block_permutations := multinomial 5 1 1 2
  (total_permutations - block_permutations + 1) = 361 :=
by
  sorry

end permutations_without_HMMT_l749_749960


namespace avg_first_12_even_is_13_l749_749728

-- Definition of the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- The sum of the first 12 even numbers
def sum_first_12_even_numbers : ℕ := first_12_even_numbers.sum

-- Number of first 12 even numbers
def count_12_even_numbers : ℕ := first_12_even_numbers.length

-- The average of the first 12 even numbers
def average_12_even_numbers : ℕ := sum_first_12_even_numbers / count_12_even_numbers

-- Proof statement that the average of the first 12 even numbers is 13
theorem avg_first_12_even_is_13 : average_12_even_numbers = 13 := by
  sorry

end avg_first_12_even_is_13_l749_749728


namespace average_speed_approx_l749_749320

-- Definitions for each segment as given in the conditions
def segment_A_distance := D : ℝ
def segment_A_speed := S : ℝ
def segment_B_distance := 3 * D : ℝ
def segment_B_speed := 2 * S : ℝ
def segment_C_distance := D / 2 : ℝ
def segment_C_speed := 0.75 * S : ℝ
def segment_D_distance := 1.5 * D : ℝ
def segment_D_speed := 0.8 * S : ℝ
def segment_E_distance := 2.5 * D : ℝ
def segment_E_average_speed := (0.5 * S + S) / 2 : ℝ 
def segment_F_distance := 4 * D : ℝ
def segment_F_effective_speed := 1.2 * S * 0.88 : ℝ
def segment_G_distance := 1.2 * D : ℝ
def segment_G_speed := 0.5 * S : ℝ

-- Total distance
def total_distance := 14.2 * D

-- Total time calculation
def total_time :=
  segment_A_distance / segment_A_speed +
  segment_B_distance / segment_B_speed +
  segment_C_distance / segment_C_speed +
  segment_D_distance / segment_D_speed +
  segment_E_distance / segment_E_average_speed +
  segment_F_distance / segment_F_effective_speed +
  segment_G_distance / segment_G_speed

-- Average speed calculation
def average_speed := total_distance / total_time

-- Proof statement
theorem average_speed_approx : 
  average_speed = S / 1.002 := by sorry

end average_speed_approx_l749_749320


namespace smallest_n_product_exceeds_l749_749947

theorem smallest_n_product_exceeds (n : ℕ) : (5 : ℝ) ^ (n * (n + 1) / 14) > 1000 ↔ n = 7 :=
by sorry

end smallest_n_product_exceeds_l749_749947


namespace smallest_integer_with_18_divisors_l749_749704

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749704


namespace repeating_decimal_ratio_eq_4_l749_749603

-- Definitions for repeating decimals
def rep_dec_36 := 0.36 -- 0.\overline{36}
def rep_dec_09 := 0.09 -- 0.\overline{09}

-- Lean 4 statement of proof problem
theorem repeating_decimal_ratio_eq_4 :
  (rep_dec_36 / rep_dec_09) = 4 :=
sorry

end repeating_decimal_ratio_eq_4_l749_749603


namespace min_dist_points_on_curves_l749_749192

def polar_to_cartesian (ρ θ : Real) : (Real × Real) := (ρ * cos θ, ρ * sin θ)

def curve1 (p : Real × Real) : Prop := p.snd = 2
def curve2 (p : Real × Real) : Prop := p.fst^2 + p.snd^2 = p.fst^2 / 4

def dist (p1 p2 : Real × Real) : Real := Real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2)

theorem min_dist_points_on_curves
  (M N : Real × Real)
  (hM : curve1 M)
  (hN : curve2 N) :
  ∃ min_dist : Real, ∀ (M' N' : Real × Real), curve1 M' → curve2 N' → dist M' N' ≥ min_dist := by
  sorry

end min_dist_points_on_curves_l749_749192


namespace calculate_Y_l749_749450

theorem calculate_Y : 
  let P := 208 / 4 
  let Q := P / 2 
  let Y := P - Q * 0.10 
  in Y = 49.4 := 
by 
  let P := 208 / 4 
  let Q := P / 2 
  let Y := P - Q * 0.10 
  sorry

end calculate_Y_l749_749450


namespace calculate_platform_length_l749_749271

theorem calculate_platform_length :
  let train_length := 120 -- meters
  let train_speed_kmph := 60 -- kmph
  let train_speed_mps := (train_speed_kmph * 1000) / 3600 -- converting kmph to mps
  let crossing_time := 20 -- seconds
  let total_distance := train_speed_mps * crossing_time -- used for the total distance covered
  let platform_length := total_distance - train_length -- platform length calculation
  platform_length = 213.4 :=
by
  let train_length := 120
  let train_speed_kmph := 60
  let train_speed_mps := (train_speed_kmph * 1000) / 3600
  let crossing_time := 20
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  show platform_length = 213.4 from sorry

end calculate_platform_length_l749_749271


namespace Jie_is_tallest_l749_749985

variables (Person : Type) (Igor Jie Faye Goa Han : Person)
variable (taller : Person → Person → Prop)

-- Given conditions
axiom h1 : taller Jie Igor
axiom h2 : taller Faye Goa
axiom h3 : taller Jie Faye
axiom h4 : taller Goa Han

-- Problem statement
theorem Jie_is_tallest : ∀ x : Person, x ∈ {Igor, Faye, Goa, Han} → taller Jie x :=
by
  intros x hx
  cases hx <;> try { assumption } <;> try { exact_taller.trans h3
  |>
.javascript() { sorry }

end Jie_is_tallest_l749_749985


namespace range_of_a_l749_749527

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then (x - a) ^ 2 + Real.exp 1 else x / Real.log x + a + 10

theorem range_of_a (a : ℝ) :
    (∀ x, f x a ≥ f 2 a) → (2 ≤ a ∧ a ≤ 6) :=
by
  sorry

end range_of_a_l749_749527


namespace top_layer_lamps_l749_749483

theorem top_layer_lamps (a : ℕ) :
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a = 381) → a = 3 := 
by
  intro h
  sorry

end top_layer_lamps_l749_749483


namespace combined_ratio_l749_749850

namespace ArtShow

-- Define the conditions
def painted_1 := 175
def sold_1 := 82
def painted_2 := 242
def sold_2 := 163
def painted_3 := 198
def sold_3 := 135

-- Define the unsold and sold pictures
def unsold_1 := painted_1 - sold_1
def unsold_2 := painted_2 - sold_2
def unsold_3 := painted_3 - sold_3

def total_unsold := unsold_1 + unsold_2 + unsold_3
def total_sold := sold_1 + sold_2 + sold_3

-- Define the target ratio
def ratio_unsold_to_sold := (total_unsold, total_sold)

theorem combined_ratio : ratio_unsold_to_sold = (235, 380) :=
  by
  unfold painted_1
  unfold painted_2
  unfold painted_3
  unfold sold_1
  unfold sold_2
  unfold sold_3
  unfold unsold_1
  unfold unsold_2
  unfold unsold_3
  unfold total_unsold
  unfold total_sold
  unfold ratio_unsold_to_sold
  sorry

end ArtShow

end combined_ratio_l749_749850


namespace find_lambda_l749_749086

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749086


namespace complex_number_a_real_l749_749407

theorem complex_number_a_real (a : ℝ) :
  let z1 := complex.mk 3 a,
      z2 := complex.mk a (-3) in
  (z1 * z2).im = 0 → a = 3 ∨ a = -3 := by
  sorry

end complex_number_a_real_l749_749407


namespace find_m_condition_l749_749955

structure vector (α : Type*) :=
  (x y : α)

def collinear {α : Type*} [field α] (u v : vector α) : Prop :=
  ∃ k : α, u.x = k * v.x ∧ u.y = k * v.y

def find_m (m : ℝ) : vector ℝ :=
  let a : vector ℝ := ⟨1, 2⟩ 
  let b : vector ℝ := ⟨2, -3⟩ 
  collinear ⟨m + 2, 2 * m - 3⟩ ⟨1, 9⟩

theorem find_m_condition  (m : ℝ) : find_m m → m = -3 :=
  sorry

end find_m_condition_l749_749955


namespace inclination_angle_30_degrees_l749_749458

noncomputable def inclination_angle (M N : ℝ × ℝ) : ℝ :=
  let k := (N.2 - M.2) / (N.1 - M.1) in
  Real.arctan k * 180 / Real.pi

theorem inclination_angle_30_degrees :
  inclination_angle (1, 2) (4, 2 + Real.sqrt 3) = 30 :=
by
  sorry

end inclination_angle_30_degrees_l749_749458


namespace quadratic_complete_square_r_plus_s_l749_749169

theorem quadratic_complete_square_r_plus_s :
  ∃ r s : ℚ, (∀ x : ℚ, 7 * x^2 - 21 * x - 56 = 0 → (x + r)^2 = s) ∧ r + s = 35 / 4 := sorry

end quadratic_complete_square_r_plus_s_l749_749169


namespace billy_sandwiches_count_l749_749335

/-- 
Billy made some sandwiches; Katelyn made 47 more than that. Chloe made a quarter of the amount that Katelyn made. They made 169 sandwiches in all. This theorem proves the number of sandwiches Billy made.
-/
theorem billy_sandwiches_count (B K C : ℕ)  
  (h1 : K = B + 47) 
  (h2 : C = (1 / 4 : ℝ) * (B + 47)) 
  (h3 : (B + K + C : ℝ) = 169) 
  : B = 49 :=
begin
  have h4 : (4 : ℝ) * B + (4 : ℝ) * K + (4 : ℝ) * C = 4 * 169, sorry,
  have h5 : (4 : ℝ) * B + (4 : ℝ) * (B + 47) + (B + 47) = 676, sorry,
  have h6 : (4 : ℝ) * B + (4 : ℝ) * B + 188 + B + 47 = 676, sorry,
  have h7 : (9 : ℝ) * B = 441, sorry,
  have h8 : B = 49, sorry,
  exact h8,
end

end billy_sandwiches_count_l749_749335


namespace inequality_abc_l749_749164

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
    a^2 + b^2 + c^2 + 3 ≥ (1 / a) + (1 / b) + (1 / c) + a + b + c :=
sorry

end inequality_abc_l749_749164


namespace arcsin_one_half_l749_749828

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749828


namespace arcsin_one_half_l749_749829

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749829


namespace range_of_a_l749_749437

def A : set ℝ := {x | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : set ℝ := {x | x > a}

theorem range_of_a (a : ℝ) (h : (A ∩ B a).nonempty) : a < 2 :=
sorry

end range_of_a_l749_749437


namespace find_A_from_AB9_l749_749360

theorem find_A_from_AB9 (A B : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3 : 100 * A + 10 * B + 9 = 459) : A = 4 :=
sorry

end find_A_from_AB9_l749_749360


namespace single_elimination_games_l749_749172

theorem single_elimination_games (n : ℕ) (h : n = 32) : 
  ∀ k : ℕ, k = n - 1 → 
  (∀ (m : ℕ), (m = 32) → ∀ t, t = m - 1 → t + sum_range 1 5 = 31) :=
by 
  intros, 
  sorry

end single_elimination_games_l749_749172


namespace find_lambda_l749_749059

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749059


namespace log_cos_range_l749_749944

theorem log_cos_range :
  ∀ (x : ℝ), x ∈ Icc (-real.pi / 2) (real.pi / 2) →
  log 2 (3 * real.cos x + 1) ∈ Icc 0 2 :=
begin
  sorry
end

end log_cos_range_l749_749944


namespace probability_king_of_hearts_l749_749316

noncomputable def probability_top_card_king_of_hearts (deck : List (ULift {n : ℕ // n < 52})) : ℝ := 
  1 / 52

theorem probability_king_of_hearts 
  (cards : List (ULift {n : ℕ // n < 52}))
  (h1 : cards.length = 52) 
  (h2 : ∃ k ∈ cards, k = 1) /* assuming ULift 1 stands for King of Hearts */
  (h3 : ∀ i (h : i < cards.length), ∃ k ∈ cards, k = i ) : /* each card is unique and present */
  probability_top_card_king_of_hearts cards = 1/52 :=
by
  sorry

end probability_king_of_hearts_l749_749316


namespace initial_percentage_alcohol_l749_749739

-- Define the initial conditions
variables (P : ℚ) -- percentage of alcohol in the initial solution
variables (V1 V2 : ℚ) -- volumes of the initial solution and added alcohol
variables (C2 : ℚ) -- concentration of the resulting solution

-- Given the initial conditions and additional parameters
def initial_solution_volume : ℚ := 6
def added_alcohol_volume : ℚ := 1.8
def final_solution_volume : ℚ := initial_solution_volume + added_alcohol_volume
def final_solution_concentration : ℚ := 0.5 -- 50%

-- The amount of alcohol initially = (P / 100) * V1
-- New amount of alcohol after adding pure alcohol
-- This should equal to the final concentration of the new volume

theorem initial_percentage_alcohol : 
  (P / 100 * initial_solution_volume) + added_alcohol_volume = final_solution_concentration * final_solution_volume → 
  P = 35 :=
sorry

end initial_percentage_alcohol_l749_749739


namespace algebraic_identity_example_l749_749736

-- Define the variables a and b
def a : ℕ := 287
def b : ℕ := 269

-- State the problem and the expected result
theorem algebraic_identity_example :
  a * a + b * b - 2 * a * b = 324 :=
by
  -- Since the proof is not required, we insert sorry here
  sorry

end algebraic_identity_example_l749_749736


namespace lambda_value_l749_749025

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749025


namespace father_has_nine_children_l749_749749

noncomputable def number_of_children (x : ℕ) where
  distribute : ∀ i : ℕ, x ≡ 1000 * (i + 1) + (x / 10) * (9 - i)
  equal_distribution : ∀ i j, (i ≠ j) → distribute i = distribute j

theorem father_has_nine_children : ∃ n : ℕ, number_of_children n ∧ n = 9 := by
  sorry

end father_has_nine_children_l749_749749


namespace functional_equation_solution_l749_749364

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, a + b + c ≥ 0 → f(a^3) + f(b^3) + f(c^3) ≥ 3 * f(abc)) ∧
  (∀ a b c : ℝ, a + b + c ≤ 0 → f(a^3) + f(b^3) + f(c^3) ≤ 3 * f(abc)) →
  ∃ m : ℝ, 0 ≤ m ∧ ∀ x : ℝ, f(x) = m * x :=
by sorry

end functional_equation_solution_l749_749364


namespace arcsin_of_half_l749_749789

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l749_749789


namespace coloring_books_got_rid_of_l749_749770

def coloringBooks 
    (initial_books : ℝ) 
    (coupons_total : ℝ) 
    (coupons_per_book : ℝ) : ℝ :=
    initial_books - (coupons_total / coupons_per_book)

theorem coloring_books_got_rid_of 
    (initial_books : ℝ)
    (coupons_total : ℝ)
    (coupons_per_book : ℝ)
    (h_initial : initial_books = 40.0)
    (h_total : coupons_total = 80)
    (h_per_book : coupons_per_book = 4.0) : coloringBooks initial_books coupons_total coupons_per_book = 20 :=
by
    sorry

end coloring_books_got_rid_of_l749_749770


namespace find_x_l749_749177

theorem find_x (x : ℝ) : 0.20 * x - (1 / 3) * (0.20 * x) = 24 → x = 180 :=
by
  intro h
  sorry

end find_x_l749_749177


namespace range_of_a_l749_749408

-- Define the conditions and the problem
def neg_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def neg_q (x : ℝ) (a : ℝ) : Prop := x > a
def p (x : ℝ) : Prop := x ≤ -3 ∨ x ≥ 0
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, neg_p x → ¬ p x) ∧
  (∀ x : ℝ, neg_q x a → ¬ q x a) ∧
  (∀ x : ℝ, q x a → p x) ∧
  (∃ x : ℝ, ¬ (q x a → p x)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l749_749408


namespace sum_of_areas_of_triangles_l749_749502

/-- Definition of a unit square and associated points and intersections as described in the problem. -/
structure Square := (A B C D : Point)
                   (unit_length : dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1)

structure ProblemPoints := (Q : ℕ → Point)
                          (P : ℕ → Point)
                          (Q1_mid_BC : midpoint Q 1 B C)

def triangle_area (B Q P : Point) : ℝ :=
  0.5 * dist B Q * height_from_point P BC

theorem sum_of_areas_of_triangles (s : Square) (pp : ProblemPoints) :
  ∑' i, triangle_area s.B (pp.Q i) (pp.P i) = 1 / 18 :=
by
  sorry

end sum_of_areas_of_triangles_l749_749502


namespace arcsin_of_half_l749_749790

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l749_749790


namespace problem_l749_749380

theorem problem (a b c d : ℤ) (ha : 0 ≤ a ∧ a ≤ 99) (hb : 0 ≤ b ∧ b ≤ 99) 
  (hc : 0 ≤ c ∧ c ≤ 99) (hd : 0 ≤ d ∧ d ≤ 99) :
  let n (x : ℤ) := 101 * x - 100 * 2 ^ x in
  n a + n b ≡ n c + n d [ZMOD 10100] →
  ({a, b} = {c, d} : multiset ℤ) := 
by sorry

end problem_l749_749380


namespace number_of_possible_multisets_l749_749562

def polynomial1 (x : ℤ) : ℤ := a₁₀ * x^10 + a₉ * x^9 + a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

def polynomial2 (x : ℤ) : ℤ := a₀ * x^10 + a₁ * x^9 + a₂ * x^8 + a₃ * x^7 + a₄ * x^6 + a₅ * x^5 + a₆ * x^4 + a₇ * x^3 + a₈ * x^2 + a₉ * x + a₁₀

theorem number_of_possible_multisets 
(a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ)
(h1 : a₁₀ ≠ 0) 
(h2 : a₀ ≠ 0)
(h3 : ∀ r : ℤ, polynomial1 r = 0 → polynomial2 r = 0)
: ∃ S : finset ℤ, S.card = 11 := 
sorry

end number_of_possible_multisets_l749_749562


namespace alternating_draws_probability_l749_749740

theorem alternating_draws_probability :
  let W := 6
  let B := 6
  (W + B) = 12 →
  (W / (W + B)) * ((B - 1) / (W + B - 1)) * (W - 1) / (B + W - 2) * 
  ((B - 2) / (W + B - 3)) * ((W - 2) / (B + W - 4)) * 
  ((B - 3) / (W + B - 5)) * ((W - 3) / (B + W - 6)) * 
  ((B - 4) / (W + B - 7)) * ((W - 4) / (B + W - 8)) * 
  ((B - 5) / (W + B - 9)) * ((W - 5) / (B + W - 10)) * 
  ((B - 6) / (W + B - 11)) * ((W - 6) / (W + B - 12)) = (1 / 924) :=
by
  let W := 6
  let B := 6
  have h1 : (W + B) = 12 := rfl
  have h2 : (W / (W + B)) * ((B - 1) / (W + B - 1)) * (W - 1) / (B + W - 2) * 
            ((B - 2) / (W + B - 3)) * ((W - 2) / (B + W - 4)) * 
            ((B - 3) / (W + B - 5)) * ((W - 3) / (B + W - 6)) * 
            ((B - 4) / (W + B - 7)) * ((W - 4) / (B + W - 8)) * 
            ((B - 5) / (W + B - 9)) * ((W - 5) / (B + W - 10)) * 
            ((B - 6) / (W + B - 11)) * ((W - 6) / (W + B - 12)) = (1 / 924) :=
  sorry
  exact h2

end alternating_draws_probability_l749_749740


namespace lambda_value_l749_749020

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749020


namespace num_positive_integers_in_square_range_l749_749388

theorem num_positive_integers_in_square_range :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), 225 ≤ x^2 ∧ x^2 ≤ 400 → (x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20) :=
by
  existsi 6
  split
  sorry

end num_positive_integers_in_square_range_l749_749388


namespace monotonicity_a1_range_of_a_l749_749936

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- 1. Monotonicity when \( a = 1 \)
theorem monotonicity_a1 :
  (∀ x > 0, (f x 1)' > 0) ∧ (∀ x < 0, (f x 1)' < 0) :=
by
  sorry

-- 2. Range of \( a \) for \( f(x) \geq \frac{1}{2} x^3 + 1 \) for \( x \geq 0 \)
theorem range_of_a (a : ℝ) (x : ℝ) (hx : x ≥ 0) (hf : f x a ≥ (1 / 2) * x^3 + 1) :
  a ≥ (7 - Real.exp 2) / 4 :=
by
  sorry

end monotonicity_a1_range_of_a_l749_749936


namespace quadratic_intersect_condition_l749_749878

theorem quadratic_intersect_condition (k : ℝ) :
  (k > -1/4) ∧ (k ≠ 2) ↔ ((2*k - 1)^2 - 4*k*(k - 2) > 0) ∧ (k - 2 ≠ 0) :=
begin
  sorry
end

end quadratic_intersect_condition_l749_749878


namespace lambda_value_l749_749026

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749026


namespace max_integer_solutions_l749_749766

noncomputable def is_super_centered (p : ℤ[X]) : Prop :=
  p.eval 50 = 50

theorem max_integer_solutions (p : ℤ[X]) (h : is_super_centered p) :
  ∃ k_set : Finset ℤ, k_set.card ≤ 7 ∧ ∀ k ∈ k_set, p.eval k = k^4 :=
sorry

end max_integer_solutions_l749_749766


namespace lambda_solution_l749_749069

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749069


namespace factorize_expression_l749_749357

variable (m n : ℤ)

theorem factorize_expression : 2 * m * n^2 - 12 * m * n + 18 * m = 2 * m * (n - 3)^2 := by
  sorry

end factorize_expression_l749_749357


namespace unique_paths_count_l749_749760

-- Definitions of initial and final points.
def start : (ℕ × ℕ) := (0, 0)
def end : (ℕ × ℕ) := (4, 4)

-- Defining the conditions as per the problem.
def valid_move (p q : (ℕ × ℕ)) : Prop :=
  (q.1 = p.1 + 1 ∧ q.2 = p.2) ∨
  (q.1 = p.1 ∧ q.2 = p.2 + 1) ∨
  (q.1 = p.1 + 1 ∧ q.2 = p.2 + 1)

def no_right_angle_turns (path : List (ℕ × ℕ)) : Prop :=
  ∀ i < path.length - 2,
    ¬((path.get i).1 = (path.get (i+1)).1 ∨ (path.get i).2 = (path.get (i+1)).2) ∨
    ¬((path.get (i+1)).1 = (path.get (i+2)).1 ∨ (path.get (i+1)).2 = (path.get (i+2)).2)

def is_valid_path (path : List (ℕ × ℕ)) : Prop :=
  path.head? = some start ∧
  (∀ i < path.length - 1, valid_move (path.get i) (path.get (i+1))) ∧
  path.get (path.length-1) = end ∧
  no_right_angle_turns path

-- The main statement to prove.
theorem unique_paths_count : 
  {path : List (ℕ × ℕ) // is_valid_path path}.to_finset.card = 27 := 
sorry

end unique_paths_count_l749_749760


namespace find_lattice_points_l749_749537

-- Define the quadratic function
def quadratic_function (x : ℤ) : ℚ :=
  (x^2 / 10) - (x / 10) + 9 / 5

-- The statement of the theorem to prove
theorem find_lattice_points :
  ∃ (S : set (ℤ × ℤ)),
    S = {(2, 2), (4, 3), (7, 6), (9, 9), (-6, 6), (-3, 3)} ∧
    ∀ (x y : ℤ), (x, y) ∈ S ↔ (y = quadratic_function x ∧ y ≤ |x|) := 
  sorry

end find_lattice_points_l749_749537


namespace three_digit_largest_fill_four_digit_smallest_fill_l749_749596

theorem three_digit_largest_fill (n : ℕ) (h1 : n * 1000 + 28 * 4 < 1000) : n ≤ 2 := sorry

theorem four_digit_smallest_fill (n : ℕ) (h2 : n * 1000 + 28 * 4 ≥ 1000) : 3 ≤ n := sorry

end three_digit_largest_fill_four_digit_smallest_fill_l749_749596


namespace smallest_integer_with_18_divisors_l749_749639

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749639


namespace coins_division_possible_l749_749723

theorem coins_division_possible :
  (∃ A B C : Multiset ℕ, A.sum = B.sum ∧ B.sum = C.sum ∧ (A + B + C = {1, 2, 3, ..., 20}.val)) :=
by sorry

end coins_division_possible_l749_749723


namespace num_positive_integers_in_square_range_l749_749389

theorem num_positive_integers_in_square_range :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), 225 ≤ x^2 ∧ x^2 ≤ 400 → (x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20) :=
by
  existsi 6
  split
  sorry

end num_positive_integers_in_square_range_l749_749389


namespace three_digit_numbers_without_5_or_6_l749_749098

theorem three_digit_numbers_without_5_or_6 : 
  { n | 100 ≤ n ∧ n < 1000 ∧ ∀ d ∈ [n % 10, (n / 10) % 10, n / 100], d ≠ 5 ∧ d ≠ 6 }.card = 343 :=
by
  sorry

end three_digit_numbers_without_5_or_6_l749_749098


namespace ratio_prob_l749_749853

-- Definitions based on the conditions provided in the problem
def num_ways_distribute (total_balls bins : ℕ) : ℕ := bins ^ total_balls

def count_A (total_balls : ℕ) (a b x y : ℕ) : ℕ :=
finset.card (finset.range_total_balls).powerset.filter (λ s, s.card = a).card

def count_B (total_balls b x y : ℕ) : ℕ :=
finset.card (finset.range total_balls).powerset.len_eq_card (λ s, s.card = b).card

-- Probability p where bins have the 3-5-6-6 distribution
def prob_p (total_balls : ℕ) (distribution : List ℕ) : ℕ := count_A total_balls 3 5 6 6

-- Probability q where bins have the 5-5-5-5 distribution
def prob_q (total_balls : ℕ) (distribution : List ℕ) : ℕ := count_B total_balls 5 5 5 5

-- Proof statement
theorem ratio_prob (total_balls bins : ℕ) (h : total_balls = 20) (full_bins : bins = 4) :
  (prob_p total_balls [3, 5, 6, 6]) / (prob_q total_balls [5, 5, 5, 5]) = sorry := 
sorry

end ratio_prob_l749_749853


namespace find_natural_number_l749_749713

variable {A : ℕ}

theorem find_natural_number (h1 : A = 8 * 2 + 7) : A = 23 :=
sorry

end find_natural_number_l749_749713


namespace find_x_given_y_64_l749_749586

-- Define the main variables and properties.
variables (x y : ℝ) (k : ℝ)

-- The conditions
def positive {z : ℝ} (h : z > 0) := true
axiom positive_x : positive x (by sorry)
axiom positive_y : positive y (by sorry)
axiom inverse_proportional : x^3 * y = k
axiom initial_condition : x = 2 ∧ y = 8

-- The theorem to prove
theorem find_x_given_y_64 : (y = 64) → (x = 1) := by
  sorry

end find_x_given_y_64_l749_749586


namespace find_son_age_l749_749584

theorem find_son_age (F S : ℕ) (h1 : F + S = 55)
  (h2 : ∃ Y, S + Y = F ∧ (F + Y) + (S + Y) = 93)
  (h3 : F = 18 ∨ S = 18) : S = 18 :=
by
  sorry  -- Proof to be filled in

end find_son_age_l749_749584


namespace computation_correct_l749_749830

noncomputable def compute_expression : Float :=
  let inner := (625681 + 1000 : Float)
  let sqrt_inner := Real.sqrt inner
  let sqrt_1000 := Real.sqrt 1000
  (sqrt_inner - sqrt_1000)^2

theorem computation_correct : compute_expression = 626681 - 2 * Real.sqrt 626681 * 31.622776601683793 + 1000 :=
  sorry

end computation_correct_l749_749830


namespace graduation_messages_total_l749_749295

/-- Define the number of students in the class -/
def num_students : ℕ := 40

/-- Define the combination formula C(n, 2) for choosing 2 out of n -/
def combination (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Prove that the total number of graduation messages written is 1560 -/
theorem graduation_messages_total : combination num_students = 1560 :=
by
  sorry

end graduation_messages_total_l749_749295


namespace smallest_positive_integer_with_18_divisors_l749_749698

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749698


namespace x_cannot_be_zero_l749_749977

def f (x : ℝ) : ℝ := 1 / x

theorem x_cannot_be_zero (x : ℝ) (hx : f(f(x)) ≠ 0.14285714285714285) : x ≠ 0 :=
by
  have h1 : f x = 1 / x := rfl
  have h2 : f (f x) = x := 
    calc
      f (f x) = f (1 / x)   : by rw h1
            ... = 1 / (1 / x) : by rw h1
            ... = x          : by field_simp
  sorry

end x_cannot_be_zero_l749_749977


namespace min_AP_plus_BP_l749_749156

-- Definitions of points A, B, and the parabola
def A := (1 : ℝ, 0 : ℝ)
def B := (7 : ℝ, 6 : ℝ)
def parabola (P : ℝ × ℝ) : Prop := P.snd ^ 2 = 4 * P.fst

-- Definition to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((Q.fst - P.fst) ^ 2 + (Q.snd - P.snd) ^ 2)

-- Define the function to minimize
def AP_plus_BP (P : ℝ × ℝ) : ℝ := dist A P + dist B P

-- The theorem statement
theorem min_AP_plus_BP : ∃ (P : ℝ × ℝ), parabola P ∧ AP_plus_BP P = 8 :=
by
  sorry

end min_AP_plus_BP_l749_749156


namespace lambda_value_l749_749039

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749039


namespace smallest_int_with_18_divisors_l749_749658

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749658


namespace angle_bisector_length_ratio_l749_749126

theorem angle_bisector_length_ratio 
  {A B C L M K N : Type*} 
  [Triangle A B C] 
  (AL_bisector : is_angle_bisector A L)
  (CM_median : is_median C M)
  (K_projected : is_orthogonal_projection L K AC)
  (N_projected : is_orthogonal_projection M N AC)
  (AK_ratio : Rat := 4)
  (KC_ratio : Rat := 1)
  (AN_ratio : Rat := 3)
  (NC_ratio : Rat := 7)
  (h1 : ratio AK KC = AK_ratio / KC_ratio)
  (h2 : ratio AN NC = AN_ratio / NC_ratio) :
  (length AL) / (sqrt 13) = 4 := 
sorry

end angle_bisector_length_ratio_l749_749126


namespace cupboard_cost_price_l749_749730

-- Define a hypothesis for selling price condition
def selling_price (C : ℝ) : ℝ := C * 0.86

-- Define a hypothesis for profitable selling price condition
def profitable_selling_price (C : ℝ) : ℝ := C * 1.14

-- Given conditions
axiom condition1 (C : ℝ) : 
  profitable_selling_price C = selling_price C + 2086

-- The theorem to prove 
theorem cupboard_cost_price : 
  ∃ C : ℝ, C = 7450 ∧ condition1 C :=
sorry

end cupboard_cost_price_l749_749730


namespace arcsin_one_half_eq_pi_six_l749_749815

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l749_749815


namespace num_parallelograms_4x6_grid_l749_749111

noncomputable def numberOfParallelograms (m n : ℕ) : ℕ :=
  let numberOfRectangles := (Nat.choose (m + 1) 2) * (Nat.choose (n + 1) 2)
  let numberOfSquares := (m * n) + ((m - 1) * (n - 1)) + ((m - 2) * (n - 2)) + ((m - 3) * (n - 3))
  let numberOfRectanglesWithUnequalSides := numberOfRectangles - numberOfSquares
  2 * numberOfRectanglesWithUnequalSides

theorem num_parallelograms_4x6_grid : numberOfParallelograms 4 6 = 320 := by
  sorry

end num_parallelograms_4x6_grid_l749_749111


namespace reservoir_full_percentage_after_storm_l749_749321

theorem reservoir_full_percentage_after_storm 
  (original_contents water_added : ℤ) 
  (percentage_full_before_storm: ℚ) 
  (total_capacity new_contents : ℚ) 
  (H1 : original_contents = 220 * 10^9) 
  (H2 : water_added = 110 * 10^9) 
  (H3 : percentage_full_before_storm = 0.40)
  (H4 : total_capacity = original_contents / percentage_full_before_storm)
  (H5 : new_contents = original_contents + water_added) :
  (new_contents / total_capacity) = 0.60 := 
by 
  sorry

end reservoir_full_percentage_after_storm_l749_749321


namespace opposite_face_to_x_is_E_l749_749889

variables (Face : Type) (x A B C D E : Face)

-- Define the conditions
variable (NetCanBeFoldedIntoCube : Prop)
variable (AdjacentToX : ∀ (face : Face), face = A ∨ face = B ∨ face = D → True)
variable (C_AdjacentToB : Prop)
variable (E_AdjacentToA : Prop)

-- Define the theorem
theorem opposite_face_to_x_is_E : NetCanBeFoldedIntoCube → AdjacentToX x A ∧ AdjacentToX x B ∧ AdjacentToX x D ∧ C_AdjacentToB ∧ E_AdjacentToA → ∃ (face : Face), face = E :=
sorry

end opposite_face_to_x_is_E_l749_749889


namespace min_value_of_function_l749_749351

theorem min_value_of_function : 
  ∃ (c : ℝ), (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2) ≥ c) ∧
             (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2 = c) → c = 1) := 
sorry

end min_value_of_function_l749_749351


namespace count_valid_numbers_l749_749095

def is_positive (n : ℕ) : Prop := n > 0
def is_less_than_10000 (n : ℕ) : Prop := n < 10000
def has_at_most_three_digits (n : ℕ) : Prop := (n.to_digits.length ≤ 3)

def valid_number (n : ℕ) : Prop := 
  is_positive n ∧ is_less_than_10000 n ∧ has_at_most_three_digits n

theorem count_valid_numbers : (finset.range 10000).filter (λ n, valid_number n).card = 3231 := by 
  sorry -- Proof to be filled.

end count_valid_numbers_l749_749095


namespace zeros_of_f_on_interval_l749_749962

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem zeros_of_f_on_interval : ∃ (S : Set ℝ), S ⊆ (Set.Ioo 0 1) ∧ S.Infinite ∧ ∀ x ∈ S, f x = 0 := by
  sorry

end zeros_of_f_on_interval_l749_749962


namespace smallest_k_is_10_l749_749345

noncomputable def smallest_k : ℕ := 10

theorem smallest_k_is_10 :
  ∃ (k : ℕ), ∀ (cover_function : ℕ → set ℝ),
    (∀ n, n ≥ 1 → cover_function n = {x : ℝ | ∃ i, (i ≤ 5 ∧ (x ∈ set.Icc (i * (1 / ↑k)) (i * (1 / ↑k + 1))}) →
    (∀ i, i > 0 → ∃ j, j > 0 ∧ x = (1 / j)) →
    (k = smallest_k) := 
by sorry

end smallest_k_is_10_l749_749345


namespace delta4_zero_l749_749381

def seq (n : ℕ) : ℕ := n^3 + 2 * n

def delta1 (u : ℕ → ℕ) (n : ℕ) : ℕ :=
  u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℕ) : ℕ → ℕ
  | 0, n => u n
  | 1, n => delta1 u n
  | (k + 1), n => delta1 (delta k u) n

theorem delta4_zero (u : ℕ → ℕ) (h : u = seq) :
  ∀ n, delta 4 u n = 0 ∧ delta 3 u n ≠ 0 :=
by
  sorry

end delta4_zero_l749_749381


namespace smallest_integer_with_18_divisors_l749_749628

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749628


namespace hyperbola_eccentricity_eq_l749_749503

variables {a b a1 b1 c : ℝ}
variables {F1 F2 M : ℝ × ℝ}

def ellipse_eq : Prop := ∀ (x y : ℝ), (x, y) ∈ ell := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola_eq : Prop := ∀ (x y : ℝ), (x, y) ∈ hyp := x^2 / a1^2 - y^2 / b1^2 = 1
def intersect_eq : Prop := ∃ (x y : ℝ), (x, y) ∈ ell ∧ (x, y) ∈ hyp ∧ 0 < x ∧ 0 < y
def angle_ninety_deg_eq : Prop := ∠(F1, M, F2) = π / 2
def eccentricity_ellipse : Prop := (1 - (b / a)^2)^0.5 = 3 / 4

theorem hyperbola_eccentricity_eq (a b a1 e1 : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a1 > 0) 
    (h4 : intersect_eq) (h5 : angle_ninety_deg_eq) (h6 : eccentricity_ellipse) :
    (1 + (b1 / a1)^2)^0.5 = 3 * 2^0.5 / 2 :=
sorry

end hyperbola_eccentricity_eq_l749_749503


namespace sequence_comparison_l749_749911

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)
noncomputable def arith_seq (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n-1) * d

theorem sequence_comparison
  (a₁ b₁ q d : ℝ)
  (h₃ : geom_seq a₁ q 3 = arith_seq b₁ d 3)
  (h₇ : geom_seq a₁ q 7 = arith_seq b₁ d 7)
  (q_pos : 0 < q)
  (d_pos : 0 < d) :
  geom_seq a₁ q 5 < arith_seq b₁ d 5 ∧
  geom_seq a₁ q 1 > arith_seq b₁ d 1 ∧
  geom_seq a₁ q 9 > arith_seq b₁ d 9 :=
by
  sorry

end sequence_comparison_l749_749911


namespace m_mobile_cheaper_than_t_mobile_l749_749535

-- Definitions based on the conditions
def t_mobile_cost (n : ℕ) : ℝ :=
  if n ≤ 2 then 50 else 50 + 16 * (n - 2)

def t_mobile_total_cost (n : ℕ) (autopay : bool) : ℝ :=
  let base_cost := t_mobile_cost n
  let data_cost := 3 * n
  let total_cost := base_cost + data_cost
  if autopay then total_cost * 0.9 else total_cost

def m_mobile_cost (n : ℕ) : ℝ :=
  if n ≤ 2 then 45 else 45 + 14 * (n - 2)

def m_mobile_total_cost (n : ℕ) : ℝ :=
  let base_cost := m_mobile_cost n
  let activation_fee := 20 * n
  base_cost + activation_fee / 12

-- The proof problem statement
theorem m_mobile_cheaper_than_t_mobile :
  let n := 5
  let yearly_t_mobile_cost := t_mobile_total_cost n true * 12
  let yearly_m_mobile_cost := m_mobile_total_cost n * 12
  yearly_t_mobile_cost - yearly_m_mobile_cost = 76.40 := by
  sorry

end m_mobile_cheaper_than_t_mobile_l749_749535


namespace maximum_value_l749_749415

theorem maximum_value (a b c : ℝ) (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) :
  ∃ M : ℝ, (∀ a b c : ℝ, ¬ (a = 0 ∧ b = 0 ∧ c = 0) → (ab + 2 * bc) / (a^2 + b^2 + c^2) ≤ M) ∧ M = sqrt 5 / 2 :=
by sorry

end maximum_value_l749_749415


namespace find_lambda_l749_749005

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749005


namespace total_sampled_students_l749_749757

-- Define the total number of students in each grade
def students_in_grade12 : ℕ := 700
def students_in_grade11 : ℕ := 700
def students_in_grade10 : ℕ := 800

-- Define the number of students sampled from grade 10
def sampled_from_grade10 : ℕ := 80

-- Define the total number of students in the school
def total_students : ℕ := students_in_grade12 + students_in_grade11 + students_in_grade10

-- Prove that the total number of students sampled (x) is equal to 220
theorem total_sampled_students : 
  (sampled_from_grade10 : ℚ) / (students_in_grade10 : ℚ) * (total_students : ℚ) = 220 := 
by
  sorry

end total_sampled_students_l749_749757


namespace exists_zero_point_of_continuous_l749_749909

theorem exists_zero_point_of_continuous (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_sign : f a * f b < 0) :
  ∃ c ∈ Set.Icc a b, f c = 0 :=
sorry

end exists_zero_point_of_continuous_l749_749909


namespace find_lambda_l749_749008

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749008


namespace smallest_int_with_18_divisors_l749_749662

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749662


namespace unique_subset_empty_set_l749_749106

def discriminant (a : ℝ) : ℝ := 4 - 4 * a^2

theorem unique_subset_empty_set (a : ℝ) :
  (∀ (x : ℝ), ¬(a * x^2 + 2 * x + a = 0)) ↔ (a > 1 ∨ a < -1) :=
by
  sorry

end unique_subset_empty_set_l749_749106


namespace Namjoon_gave_Yoongi_9_pencils_l749_749538

theorem Namjoon_gave_Yoongi_9_pencils
  (stroke_pencils : ℕ)
  (strokes : ℕ)
  (pencils_left : ℕ)
  (total_pencils : ℕ := stroke_pencils * strokes)
  (given_pencils : ℕ := total_pencils - pencils_left) :
  stroke_pencils = 12 →
  strokes = 2 →
  pencils_left = 15 →
  given_pencils = 9 := by
  sorry

end Namjoon_gave_Yoongi_9_pencils_l749_749538


namespace hyperbola_circle_tangent_l749_749135

theorem hyperbola_circle_tangent
    (b : ℝ)
    (h : b > 0)
    (F1 F2 : ℝ × ℝ)
    (c : ℝ)
    (A B : ℝ × ℝ)
    (h1 : F1 = (- c, 0))
    (h2 : F2 = (c, 0))
    (h3 : c = real.sqrt (1 + b ^ 2))
    (h4 : tangent_to_circle_through_point F1 Circle (1,0) A B)
    (h5 : distance F2 B = distance A B) :
  b = 1 + real.sqrt 3 :=
by
  sorry

end hyperbola_circle_tangent_l749_749135


namespace find_lambda_l749_749054

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749054


namespace num_7_digit_integers_correct_l749_749091

-- Define the number of choices for each digit
def first_digit_choices : ℕ := 9
def other_digit_choices : ℕ := 10

-- Define the number of 7-digit positive integers
def num_7_digit_integers : ℕ := first_digit_choices * other_digit_choices^6

-- State the theorem to prove
theorem num_7_digit_integers_correct : num_7_digit_integers = 9000000 :=
by
  sorry

end num_7_digit_integers_correct_l749_749091


namespace difference_of_squares_divisible_by_9_l749_749349

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end difference_of_squares_divisible_by_9_l749_749349


namespace factorial_fraction_l749_749355

theorem factorial_fraction :
  (factorial (factorial 4)) / (factorial 4) = 25852016738884976640000 := by
  sorry

end factorial_fraction_l749_749355


namespace nested_even_function_l749_749511

-- Defining an even function
def even_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- The property that we want to prove
theorem nested_even_function (g : ℝ → ℝ) (h : even_function g) : even_function (λ x, g (g (g (g x)))) :=
by
  -- Placeholder for the proof
  sorry

end nested_even_function_l749_749511


namespace tom_average_speed_is_60_l749_749145

def karen_wins_race (v : ℝ) : Prop :=
  ∀ t x : ℝ,
    (t = 1 / 3) →
    (x = 20) →
    (60 * t = x) →
    (v * (t + 4 / 60) = x + 4) →
    (x + 4 = 24) →
    v = 60

theorem tom_average_speed_is_60 : ∀ v : ℝ, karen_wins_race v → v = 60 :=
by
  intros v h
  have t := h 1/3 20 rfl rfl (by ring) (by ring)
  assumption

end tom_average_speed_is_60_l749_749145


namespace lambda_value_l749_749032

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749032


namespace Jie_is_tallest_l749_749984

variables (Person : Type) (Igor Jie Faye Goa Han : Person)
variable (taller : Person → Person → Prop)

-- Given conditions
axiom h1 : taller Jie Igor
axiom h2 : taller Faye Goa
axiom h3 : taller Jie Faye
axiom h4 : taller Goa Han

-- Problem statement
theorem Jie_is_tallest : ∀ x : Person, x ∈ {Igor, Faye, Goa, Han} → taller Jie x :=
by
  intros x hx
  cases hx <;> try { assumption } <;> try { exact_taller.trans h3
  |>
.javascript() { sorry }

end Jie_is_tallest_l749_749984


namespace find_value_l749_749990

-- Defining the sequence a_n, assuming all terms are positive
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

-- Definition to capture the given condition a_2 * a_4 = 4
def condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 4

-- The main statement
theorem find_value (a : ℕ → ℝ) (h_seq : is_geometric_sequence a) (h_cond : condition a) : 
  a 1 * a 5 + a 3 = 6 := 
by 
  sorry

end find_value_l749_749990


namespace hyperbola_standard_equations_l749_749373

-- Definitions derived from conditions
def focal_distance (c : ℝ) : Prop := c = 8
def eccentricity (e : ℝ) : Prop := e = 4 / 3
def equilateral_focus (c : ℝ) : Prop := c^2 = 36

-- Theorem stating the standard equations given the conditions
noncomputable def hyperbola_equation1 (y2 : ℝ) (x2 : ℝ) : Prop :=
y2 / 36 - x2 / 28 = 1

noncomputable def hyperbola_equation2 (x2 : ℝ) (y2 : ℝ) : Prop :=
x2 / 18 - y2 / 18 = 1

theorem hyperbola_standard_equations
  (c y2 x2 : ℝ)
  (c_focus : focal_distance c)
  (e_value : eccentricity (4 / 3))
  (equi_focus : equilateral_focus c) :
  hyperbola_equation1 y2 x2 ∧ hyperbola_equation2 x2 y2 :=
by
  sorry

end hyperbola_standard_equations_l749_749373


namespace log_problem_l749_749101

-- Define the problem's conditions
variables (x : ℝ) (h_log : real.log 343 / real.log (3 * x) = x)

-- Prove that x equals 7/3 and is a non-square, non-cube, non-integral rational number
theorem log_problem : x = 7 / 3 ∧ ¬(∃ n: ℕ, x = n * n) ∧ ¬(∃ m: ℕ, x = m * m * m) ∧ ∃ p q : ℤ, x = p / q ∧ ¬ ∃ r : ℤ, x = r :=
sorry

end log_problem_l749_749101


namespace min_a2_plus_a9_l749_749402

-- A positive geometric sequence
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ (a1 r : ℝ), r > 0 ∧ ∀ n, a n = a1 * r^(n - 1)

-- The goal is to prove the minimum value of a_2 + a_9
theorem min_a2_plus_a9 {a : ℕ → ℝ} (h1 : is_geometric_seq a) (h2 : a 5 * a 6 = 16) :
  ∃ (min_val : ℝ), min_val = 8 ∧ ∀ (x y : ℝ), x = a 2 → y = a 9 → x + y ≥ min_val :=
begin
  sorry
end

end min_a2_plus_a9_l749_749402


namespace part1_monotonicity_part2_find_range_l749_749941

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end part1_monotonicity_part2_find_range_l749_749941


namespace arcsin_half_eq_pi_six_l749_749802

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l749_749802


namespace exists_consecutive_divisible_by_cube_l749_749522

theorem exists_consecutive_divisible_by_cube (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ j : ℕ, j < k → ∃ m : ℕ, 1 < m ∧ (n + j) % (m^3) = 0 := 
sorry

end exists_consecutive_divisible_by_cube_l749_749522


namespace smallest_integer_with_18_divisors_l749_749644

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749644


namespace sum_of_solutions_l749_749509

-- Defining the function f
def f (x : ℝ) : ℝ := 3 * x - 2

-- Defining the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ := (y + 2) / 3

-- The statement asserting the problem's conditions and the desired outcome
theorem sum_of_solutions :
  (∑ x in ({x | f_inv x = f (x⁻¹)} : set ℝ), x) = -8 :=
by
  sorry

end sum_of_solutions_l749_749509


namespace problem_statement_l749_749573

variables (a b : ℝ)

-- Conditions: The lines \(x = \frac{1}{3}y + a\) and \(y = \frac{1}{3}x + b\) intersect at \((3, 1)\).
def lines_intersect_at (a b : ℝ) : Prop :=
  (3 = (1/3) * 1 + a) ∧ (1 = (1/3) * 3 + b)

-- Goal: Prove that \(a + b = \frac{8}{3}\)
theorem problem_statement (H : lines_intersect_at a b) : a + b = 8 / 3 :=
by
  sorry

end problem_statement_l749_749573


namespace find_lambda_l749_749009

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749009


namespace pedestrian_distances_l749_749395

theorem pedestrian_distances (s : ℝ) (x y : ℝ) :
  (∀ t1 t2 : ℝ, (0 < t1 ∧ ∀ f1 f2 : ℝ, (0 < f1 ∧ f1 < f2 ∧ t1 * 2 = x) →
  (t2 = t1 - 10) ∧ 2 * (f1 * 60) / ((120 / f1) - 10) = x) ∧ 
  (0 < t2 ∧ ∀ g1 g2 : ℝ, (0 < g1 ∧ g1 < g2 ∧ t2 * 2 = y) → 
  2 * (g1 * 60) / ((120 / g1) - 10) + x = s)
  ) → 
  (x + y = 2 * (t1 + t2) ∧ t1 ≠ t2) →
  x = (24 - s - real.sqrt (s^2 + 288)) / 2 ∧ y = (s + 24 - real.sqrt (s^2 + 288)) / 2 :=
by
  sorry

end pedestrian_distances_l749_749395


namespace hyperbola_eccentricity_correct_l749_749482

noncomputable def parabola_focus : ℝ × ℝ := (2, 0)

def hyperbola_eccentricity (a : ℝ) (h : a > 0) : ℝ :=
  let c := Real.sqrt (a^2 + 1) in
  c / a

theorem hyperbola_eccentricity_correct :
  ∀ (a : ℝ) (h : a > 0), (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2 - p.2^2 = 1)} →
  hyperbola_eccentricity a h = Real.sqrt 5 / 2 :=
by
  intros a h hyp
  have ha2 : a^2 = 4 := by sorry
  have hc : Real.sqrt (a^2 + 1) = Real.sqrt 5 := by sorry
  rw [ha2, hc]
  calc Real.sqrt 5 / a = Real.sqrt 5 / 2
    : by sorry

end hyperbola_eccentricity_correct_l749_749482


namespace value_at_4_value_of_x_when_y_is_0_l749_749430

-- Problem statement
def f (x : ℝ) : ℝ := 2 * x - 3

-- Proof statement 1: When x = 4, y = 5
theorem value_at_4 : f 4 = 5 := sorry

-- Proof statement 2: When y = 0, x = 3/2
theorem value_of_x_when_y_is_0 : (∃ x : ℝ, f x = 0) → (∃ x : ℝ, x = 3 / 2) := sorry

end value_at_4_value_of_x_when_y_is_0_l749_749430


namespace smallest_positive_integer_with_18_divisors_l749_749665

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749665


namespace lateral_surface_area_of_cylinder_l749_749765

variable (m n : ℝ) (S : ℝ)

theorem lateral_surface_area_of_cylinder (h1 : S > 0) (h2 : m > 0) (h3 : n > 0) :
  ∃ (lateral_surface_area : ℝ),
    lateral_surface_area = (π * S) / (Real.sin (π * n / (m + n))) :=
sorry

end lateral_surface_area_of_cylinder_l749_749765


namespace x_divisible_by_5_l749_749894

theorem x_divisible_by_5
  (x y : ℕ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_gt_1 : 1 < x)
  (h_eq : 2 * x^2 - 1 = y^15) : x % 5 = 0 :=
sorry

end x_divisible_by_5_l749_749894


namespace area_of_triangle_A1B1C1_l749_749891

-- Definitions from the conditions
def right_triangle (A B C : Type) [EuclideanGeometry A] (BC : ℝ) (AC : ℝ) (AB : ℝ) : Prop :=
  BC = 30 ∧ AC = 40 ∧ AB = Real.sqrt (BC ^ 2 + AC ^ 2)

def points_on_sides (C1 A1 B1 : Point) (ABC : Triangle) : Prop :=
  (AC1 = 1 ∧ BA1 = 1 ∧ CB1 = 1)

-- Main theorem statement based on the problem transformation
theorem area_of_triangle_A1B1C1 (ABC : Triangle) (A1 B1 C1 : Point) 
  (h_right_triangle : right_triangle A B C)
  (h_points_on_sides : points_on_sides C1 A1 B1 ABC) :
  ∃ S : ℝ, S = 554.2 :=
sorry

end area_of_triangle_A1B1C1_l749_749891


namespace p_sufficient_for_not_q_l749_749409

variable (x : ℝ)
def p : Prop := 0 < x ∧ x ≤ 1
def q : Prop := 1 / x < 1

theorem p_sufficient_for_not_q : p x → ¬q x :=
by
  sorry

end p_sufficient_for_not_q_l749_749409


namespace closest_point_on_parabola_l749_749218

/-- The coordinates of the point on the parabola y^2 = x that is closest to the line x - 2y + 4 = 0 are (1,1). -/
theorem closest_point_on_parabola (y : ℝ) (x : ℝ) (h_parabola : y^2 = x) (h_line : x - 2*y + 4 = 0) :
  (x = 1 ∧ y = 1) :=
sorry

end closest_point_on_parabola_l749_749218


namespace tangent_circles_length_l749_749597

theorem tangent_circles_length (A B C : Point)
  (rA rB : ℝ) 
  (h1 : rA = 7)
  (h2 : rB = 4)
  (h3 : distance A B = rA + rB)
  (tangent_to_circles : tangent AB C)
  : distance B C = 44 / 3 := 
sorry

end tangent_circles_length_l749_749597


namespace interval_length_t_subset_interval_t_l749_749954

-- Statement (1)
theorem interval_length_t (t : ℝ) (h : (Real.log t / Real.log 2) - 2 = 3) : t = 32 :=
  sorry

-- Statement (2)
theorem subset_interval_t (t : ℝ) (h : 2 ≤ Real.log t / Real.log 2 ∧ Real.log t / Real.log 2 ≤ 5) :
  0 < t ∧ t ≤ 32 :=
  sorry

end interval_length_t_subset_interval_t_l749_749954


namespace find_other_endpoint_l749_749230

theorem find_other_endpoint :
  ∃ (B : ℝ × ℝ), let (Mx, My) := (-1 : ℝ, 3 : ℝ) in
                  let (Ax, Ay) := (2 : ℝ, -4 : ℝ) in
                  let (Bx, By) := B in
                  (Mx = (Ax + Bx) / 2) ∧ (My = (Ay + By) / 2) ∧ B = (-4, 10) :=
by {
    -- Proof will be provided here
    sorry
}

end find_other_endpoint_l749_749230


namespace solve_for_x_l749_749208

theorem solve_for_x : (42 / (7 - 3 / 7) = 147 / 23) :=
by
  sorry

end solve_for_x_l749_749208


namespace bug_final_position_after_2000_jumps_l749_749207

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def move (point : ℕ) : ℕ :=
if is_odd point then (point % 6) + 1
else if is_divisible_by_4 point then (point + 3) % 6
else (point + 2) % 6

def bug_position (initial_point : ℕ) (jumps : ℕ) : ℕ :=
  nat.iterate move jumps initial_point

theorem bug_final_position_after_2000_jumps : bug_position 6 2000 = 2 :=
by sorry

end bug_final_position_after_2000_jumps_l749_749207


namespace relationship_of_sequence_l749_749420

theorem relationship_of_sequence (n : ℕ) (h : n ≥ 2) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n : ℕ, S n = -2 * n^2 + 3 * n) (ha1 : a 1 = 1) (ha : ∀ n : ℕ, n ≥ 2 → a n = -4 * n + 5) :
  S n > n * a n ∧ n * a 1 > S n := by
  sorry

end relationship_of_sequence_l749_749420


namespace equilateral_triangle_on_curve_midpoint_distance_range_l749_749134

-- Part (1)
/--
Parametric equations of curve (E): x = 2 * cos(t), y = 2 * sin(t)
A, B, C are points on (E) such that triangle ABC is an equilateral triangle.
Point A has polar coordinates (2, π/4).
Prove the polar coordinates of points B and C.
-/
theorem equilateral_triangle_on_curve (t : ℝ) (A B C : ℝ × ℝ)
  (hA : A = (2, Real.pi / 4))
  (hB : B.1 = 2 ∧ B.2 = (Real.pi / 4 + 2 * Real.pi / 3))
  (hC : C.1 = 2 ∧ C.2 = (Real.pi / 4 + 4 * Real.pi / 3)) :
  ∃ B C, A = (2, Real.pi / 4) ∧ B = (2, 11 * Real.pi / 12) ∧ C = (2, 19 * Real.pi / 12) :=
sorry

-- Part (2)
/--
For points P and Q on curve (E) with parameters t = α and t = 2α,
where 0 < α < 2π, M is the midpoint of PQ.
Prove the range of |MO| is [0, 2).
-/
theorem midpoint_distance_range (α : ℝ) (P Q M : ℝ × ℝ)
  (hα : 0 < α ∧ α < 2 * Real.pi)
  (hP : P = (2 * Real.cos α, 2 * Real.sin α))
  (hQ : Q = (2 * Real.cos (2 * α), 2 * Real.sin (2 * α)))
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  ∃ d, d = Real.sqrt ((M.1)^2 + (M.2)^2) ∧ d ∈ set.Ico 0 2 :=
sorry

end equilateral_triangle_on_curve_midpoint_distance_range_l749_749134


namespace smallest_integer_with_18_divisors_l749_749640

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749640


namespace smallest_positive_integer_with_18_divisors_l749_749663

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749663


namespace verify_problem_l749_749278

def square : Type := {sq : Nat // sq ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}}

noncomputable def arrows : square → square := sorry -- Arrow configuration function details

def initial_assignment : square → Nat
| ⟨1, h⟩ := 1
| ⟨9, h⟩ := 9
| _ := 0

def proof_problem : Prop :=
  initial_assignment ⟨A, sorry⟩ = 6 ∧
  initial_assignment ⟨B, sorry⟩ = 2 ∧
  initial_assignment ⟨C, sorry⟩ = 4 ∧
  initial_assignment ⟨D, sorry⟩ = 5 ∧
  initial_assignment ⟨E, sorry⟩ = 3 ∧
  initial_assignment ⟨F, sorry⟩ = 8 ∧
  initial_assignment ⟨G, sorry⟩ = 7

theorem verify_problem : proof_problem :=
by
  sorry

end verify_problem_l749_749278


namespace five_lines_regions_l749_749868

theorem five_lines_regions (L : Fin 5 → AffinePlane ℝ) 
  (h1 : ∀ i j : Fin 5, i ≠ j → ¬ Parallel (L i) (L j))
  (h2 : ∀ i j k : Fin 5, pairwise (≠) [i, j, k] → ¬ Concurrent (L i) (L j) (L k)) :
  count_regions L = 16 :=
sorry

end five_lines_regions_l749_749868


namespace peter_remaining_money_l749_749183

theorem peter_remaining_money :
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 426 :=
by
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  show remaining_money = 426 from sorry

end peter_remaining_money_l749_749183


namespace find_lambda_l749_749003

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749003


namespace num_suitable_n_l749_749162

theorem num_suitable_n : let n : ℤ, 1 ≤ n ∧ n ≤ 1990 ∧ ∃ a b : ℤ, (a + b = 1) ∧ (ab = -3 * n) in 
  ncount numberof_n = 50 :=
sorry

end num_suitable_n_l749_749162


namespace quadratic_has_real_root_m_value_l749_749908

theorem quadratic_has_real_root_m_value (x m : ℝ) :
  (x^2 + (1 - 2*complex.I : ℂ)*x + (3*m - complex.I : ℂ) = 0) → x = -1/2 → m = 1/12 :=
by
  sorry

end quadratic_has_real_root_m_value_l749_749908


namespace find_b1_l749_749243

theorem find_b1 (b : ℕ → ℝ) (h : ∀ n ≥ 2, (∑ i in finset.range (n+1), b i) = n^2 * b n) (h_50 : b 50 = 2) : b 1 = 2550 :=
sorry

end find_b1_l749_749243


namespace flower_pots_count_l749_749550

noncomputable def total_flower_pots (x : ℕ) : ℕ :=
  if h : ((x / 2) + (x / 4) + (x / 7) ≤ x - 1) then x else 0

theorem flower_pots_count : total_flower_pots 28 = 28 :=
by
  sorry

end flower_pots_count_l749_749550


namespace sum_of_n_values_abs_eq_4_l749_749710

theorem sum_of_n_values_abs_eq_4 :
  (∀ n : ℚ, abs (3 * n - 8) = 4 → n = 4 ∨ n = 4 / 3) →
  ((∑ x in ({4, 4/3}: finset ℚ), x) = 16 / 3) :=
by
  intro h
  sorry

end sum_of_n_values_abs_eq_4_l749_749710


namespace count_numbers_with_at_most_three_digits_l749_749094

theorem count_numbers_with_at_most_three_digits :
  let numbers := {n : ℕ | n < 10000 ∧ 
                           ((∃ d1, ∀ i ∈ n.digits 10, i = d1) ∨
                            (∃ d1 d2, d1 ≠ d2 ∧ ∀ i ∈ n.digits 10, i = d1 ∨ i = d2) ∨
                            (∃ d1 d2 d3, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ ∀ i ∈ n.digits 10, i = d1 ∨ i = d2 ∨ i = d3)) } in
  numbers.card = 4119 :=
by admit

end count_numbers_with_at_most_three_digits_l749_749094


namespace number_of_primes_in_interval_35_to_44_l749_749580

/--
The number of prime numbers in the interval [35, 44] is 3.
-/
theorem number_of_primes_in_interval_35_to_44 : 
  (Finset.filter Nat.Prime (Finset.Icc 35 44)).card = 3 := 
by
  sorry

end number_of_primes_in_interval_35_to_44_l749_749580


namespace find_larger_number_l749_749725

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
sorry

end find_larger_number_l749_749725


namespace find_v_1010_l749_749515

def sequence (k n : ℕ) : ℕ :=
  k * n + (k * (k - 1)) / 2

noncomputable def find_v_n (a b : ℕ) : ℕ :=
  if h : a ≤ b then
    let sqrt_term := int.to_nat (int.floor (real.sqrt (2 * b))) in
    sorry -- This is a placeholder for the actual logic to find the term based on our condition.
  else 0

theorem find_v_1010 : find_v_n 5 1010 = 4991 :=
  sorry

end find_v_1010_l749_749515


namespace paint_faces_count_l749_749331

-- We define the set of all faces on a die
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- We define a function to check if two faces are adjacent
def adjacent_faces (a b : ℕ) : Bool :=
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨
  (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨
  (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨
  (a = 5 ∧ b = 6) ∨ (a = 6 ∧ b = 5) ∨
  (a = 6 ∧ b = 1) ∨ (a = 1 ∧ b = 6)

-- We define a function to check if two faces are opposite
def opposite_faces (a b : ℕ) : Bool :=
  (a = 1 ∧ b = 6) ∨ (a = 6 ∧ b = 1) ∨
  (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) ∨
  (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3)

-- We define a predicate to check if two faces can be painted red
def valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧
  ¬adjacent_faces a b ∧
  ¬opposite_faces a b ∧
  (a + b ≠ 9)

-- We obtain all valid pairs
def valid_pairs : Finset (ℕ × ℕ) :=
  faces.product faces |>.filter (λ p, valid_pair p.1 p.2)

-- The theorem statement: The number of valid pairs is 10
theorem paint_faces_count : valid_pairs.card = 10 :=
  sorry

end paint_faces_count_l749_749331


namespace find_function_expression_l749_749416

noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 7

theorem find_function_expression (x : ℝ) :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  f x = x^2 - 5*x + 7 :=
by
  intro h
  sorry

end find_function_expression_l749_749416


namespace geometric_sequence_seventh_term_l749_749752

theorem geometric_sequence_seventh_term :
  ∃ (r : ℕ), (3 : ℕ) * r^4 = 243 ∧ (3 : ℕ) * r^6 = 2187 :=
by
  use 3
  split
  · calc
      (3 : ℕ) * 3^4 = 3 * 81 := by refl
      ... = 243 := by norm_num
  · calc
      (3 : ℕ) * 3^6 = 3 * 729 := by refl
      ... = 2187 := by norm_num

end geometric_sequence_seventh_term_l749_749752


namespace arithmetic_sequence_sum_l749_749484

noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem arithmetic_sequence_sum :
  (a1 d : ℤ) (h_a1 : a1 = -11) (h_d : d = 2) (n : ℕ) (h_n : n = 11) :
  S 11 a1 d = -11 := by
  rw [S, h_a1, h_d, h_n]
  -- working out the intermediate steps leads to the final value
  sorry

end arithmetic_sequence_sum_l749_749484


namespace original_cost_l749_749775

theorem original_cost (A : ℝ) (discount : ℝ) (sale_price : ℝ) (original_price : ℝ) (h1 : discount = 0.30) (h2 : sale_price = 35) (h3 : sale_price = (1 - discount) * original_price) : 
  original_price = 50 := by
  sorry

end original_cost_l749_749775


namespace smallest_integer_with_18_divisors_l749_749647

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749647


namespace num_ways_to_replace_stars_to_div_12_l749_749490

def star_digits := {0, 2, 4, 5, 7, 9}

def is_divisible_by_4 (n : Nat) : Prop :=
  (n % 4) = 0

def is_divisible_by_3 (digits : List Nat) : Prop :=
  (digits.sum % 3) = 0

def valid_replacements (digits : List Nat) : Nat :=
  let num_ways := 6^5
  let valid_last_digits := ["00", "04"] -- Representing last two digits options.
  2 * num_ways -- Multiplying the two valid end digits with the 5 positions freedom

theorem num_ways_to_replace_stars_to_div_12 :
  valid_replacements [2, 0, 1, 6, 0, 2] = 5184 :=
sorry

end num_ways_to_replace_stars_to_div_12_l749_749490


namespace constant_distance_sum_fixed_midpoint_DE_l749_749517

-- Define the setup conditions
structure Point (α : Type) := (x : α) (y : α)
structure LineSegment (α : Type) := (a b : Point α)

-- Definitions for Squares and Distances
def square {α : Type} [HasAdd α] [HasSub α] (A C : Point α) (B : Point α) : LineSegment α :=
sorry -- Placeholder for exact square definition between points

def distance_to_line {α : Type} [LinearOrderedField α] (P : Point α) (AB : LineSegment α) : α :=
sorry -- Placeholder for distance calculation from point to line

-- Main theorem statements
theorem constant_distance_sum {α : Type} [LinearOrderedField α] :
  ∀ (A B C D E D' E' : Point α), 
  let AB := LineSegment.mk A B in
  square A C D C ∧ square B C E C → 
  distance_to_line D AB + distance_to_line E AB = distance_to_line A B :=
sorry

theorem fixed_midpoint_DE {α : Type} [LinearOrderedField α] :
  ∀ (A B C D E : Point α),
  let AB := LineSegment.mk A B in
  square A C D C ∧ square B C E C → 
  ∃ M : Point α, M.x = (D.x + E.x) / 2 ∧ M.y = (D.y + E.y) / 2 ∧ -- Midpoint condition
  M.x = (A.x + B.x) / 2 ∧ M.y > 0 := -- Fixed point condition (Assuming AB on x-axis)
sorry

end constant_distance_sum_fixed_midpoint_DE_l749_749517


namespace angle_DEC_l749_749109

open Triangle

namespace Geometry

-- Definition of the right triangle with specific angle measures
def triangle_ABC : Triangle := {
  A := (0, 0),
  B := (1, 0),
  C := (0, 1),
  angleA := 90,
  angleB := 45,
  angleC := 45
}

-- Assume that AD is an altitude, and BE is a median and also an angle bisector
def altitude_AD (ABC : Triangle) (D : Point) : Prop :=
  is_right_triangle ABC ∧ 
  altitude (ABC) AD

def median_BE (ABC : Triangle) (E : Point) : Prop :=
  is_isosceles_right_triangle ABC ∧
  midpoint B E C ∧ 
  angle_bisector BE ABC

theorem angle_DEC {ABC : Triangle} (D E : Point) :
  altitude_AD ABC D →
  median_BE ABC E →
  angle DEC = 67.5 :=
by sorry

end Geometry

end angle_DEC_l749_749109


namespace arithmetic_sequence_6000th_term_l749_749567

theorem arithmetic_sequence_6000th_term :
  ∀ (p r : ℕ), 
  (2 * p) = 2 * p → 
  (2 * p + 2 * r = 14) → 
  (14 + 2 * r = 4 * p - r) → 
  (2 * p + (6000 - 1) * 4 = 24006) :=
by 
  intros p r h h1 h2
  sorry

end arithmetic_sequence_6000th_term_l749_749567


namespace smallest_integer_with_18_divisors_l749_749678

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749678


namespace length_AD_equals_seventeen_length_BK_equals_240_over_17_perimeter_EBM_equals_552_over_17_l749_749401

variable (A B C D M E K : Point)
variable (parallelogram_ABCD : Parallelogram A B C D)
variable (circle_omega : Circle)
variable (diameter_omega : Diameter circle_omega = 17)
variable (circumscribed_ABM : Circumscribed circle_omega (Triangle A B M))
variable (intersection_M : Diagonals_Intersection A B C D = M)
variable (arc_length_AE : Arc_Length circle_omega A E = 2 * Arc_Length circle_omega B M)
variable (segment_length_MK : Segment_Length M K = 8)
variable (point_E_intersects_CB : Intersects_Ray circle_omega C B E)
variable (point_K_intersects_AD : Intersects_Segment circle_omega A D K)

-- Prove that AD = 17
theorem length_AD_equals_seventeen : 
  length (segment A D) = 17 := sorry

-- Prove that BK = 240/17
theorem length_BK_equals_240_over_17 : 
  length (segment B K) = 240 / 17 := sorry

-- Prove that the perimeter of triangle EBM is 552/17
theorem perimeter_EBM_equals_552_over_17 :
  perimeter (triangle E B M) = 552 / 17 := sorry

end length_AD_equals_seventeen_length_BK_equals_240_over_17_perimeter_EBM_equals_552_over_17_l749_749401


namespace exists_k_digit_in_A_l749_749157

noncomputable def S (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

def A (x : ℕ) : Prop :=
  x > 0 ∧ (∀ d ∈ x.digits 10, d ≠ 0) ∧ (S x ∣ x)

theorem exists_k_digit_in_A (k : ℕ) (hk : k > 0) : ∃ x : ℕ, A x ∧ x.digits 10 = List.repeat 1 k :=
  sorry

end exists_k_digit_in_A_l749_749157


namespace line_equation_l749_749566

theorem line_equation (A : (ℝ × ℝ)) (hA_x : A.1 = 2) (hA_y : A.2 = 0)
  (h_intercept : ∀ B : (ℝ × ℝ), B.1 = 0 → 2 * B.1 + B.2 + 2 = 0 → B = (0, -2)) :
  ∃ (l : ℝ × ℝ → Prop), (l A ∧ l (0, -2)) ∧ 
    (∀ x y : ℝ, l (x, y) ↔ x - y - 2 = 0) :=
by
  sorry

end line_equation_l749_749566


namespace smallest_integer_with_18_divisors_l749_749702

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749702


namespace find_lambda_l749_749010

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749010


namespace hexagon_perimeter_l749_749729

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 4) (h2 : num_sides = 6) : 
    num_sides * side_length = 24 :=
by
  rw [h1, h2]
  rfl

end hexagon_perimeter_l749_749729


namespace prime_coprime_pairs_420_l749_749119

def is_prime (n : ℕ) : Prop := nat.prime n

def are_coprime (a b : ℕ) : Prop := nat.gcd a b = 1

noncomputable def count_prime_coprime_pairs (sum : ℕ) : ℕ :=
  let pairs := {uv : ℕ × ℕ | uv.1 < uv.2 ∧ uv.1 + uv.2 = sum 
                              ∧ is_prime uv.1 ∧ is_prime uv.2 ∧ are_coprime uv.1 uv.2}
  in pairs.to_finset.card

theorem prime_coprime_pairs_420 : count_prime_coprime_pairs 420 = 30 :=
begin
  sorry
end

end prime_coprime_pairs_420_l749_749119


namespace escalator_length_is_120_l749_749191

variable (x : ℝ) -- Speed of escalator in steps/unit time.

constant steps_while_ascending : ℕ := 75
constant steps_while_descending : ℕ := 150
constant speed_ascending : ℝ := 1.0
constant speed_descending : ℝ := 3.0
constant walking_speed_ratio : ℝ := 3.0

theorem escalator_length_is_120 :
  let t_ascending := (steps_while_ascending / (speed_ascending + x))
      t_descending := (steps_while_descending / (speed_descending - x)) * (1 / walking_speed_ratio) in
  steps_while_ascending * (speed_ascending + x) = steps_while_descending * (speed_descending - x) / walking_speed_ratio →
  75 * (1 + 0.6) = 120 :=
by
  intro h
  sorry

end escalator_length_is_120_l749_749191


namespace find_lambda_l749_749043

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749043


namespace max_value_of_z_l749_749986

theorem max_value_of_z (x y : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) :
  x^2 + y^2 ≤ 2 :=
by {
  sorry
}

end max_value_of_z_l749_749986


namespace equilateral_triangle_ratio_l749_749611

theorem equilateral_triangle_ratio (s : ℝ) (h_s : s = 12) : 
  (let A := (√3 * s^2) / 4 in let P := 3 * s in A / P = √3) :=
by
  sorry

end equilateral_triangle_ratio_l749_749611


namespace total_steps_to_times_square_l749_749322

-- Define the conditions
def steps_to_rockefeller : ℕ := 354
def steps_to_times_square_from_rockefeller : ℕ := 228

-- State the theorem using the conditions
theorem total_steps_to_times_square : 
  steps_to_rockefeller + steps_to_times_square_from_rockefeller = 582 := 
  by 
    -- We skip the proof for now
    sorry

end total_steps_to_times_square_l749_749322


namespace largest_circle_area_l749_749764

theorem largest_circle_area (w l: ℝ) (h_w : 2 * w = l) (h_area : w * l = 200) : 
  let perimeter := 2 * (w + l) in
  let r := perimeter / (2 * Real.pi) in
  let circle_area := Real.pi * r ^ 2 in
  Int.nearest_ne (circle_area / Real.pi) = 287 :=
by
  -- Definitions and conditions
  let w := Real.sqrt 100
  let l := 2 * w
  have h_w : 2 * w = l := by sorry
  have h_area : w * l = 200 := by sorry
  -- Calculations
  let perimeter := 2 * (w + l)
  let r := perimeter / (2 * Real.pi)
  let circle_area := Real.pi * r ^ 2
  -- Prove the final area equals approximately 287
  sorry

end largest_circle_area_l749_749764


namespace platform_length_equals_train_length_l749_749228

theorem platform_length_equals_train_length
  (train_speed_kmh : ℕ)
  (crossing_time_min : ℕ)
  (train_length_m : ℕ)
  (platform_length_m : ℕ) :
  train_speed_kmh = 72 →
  crossing_time_min = 1 →
  train_length_m = 600 →
  platform_length_m = 600 :=
by
  intros h1 h2 h3
  have speed_mpm : ℕ := 72 * 1000 / 60
  have total_distance_m := 600 + platform_length_m
  have distance_covered := speed_mpm * crossing_time_min
  rw [h1, h2] at distance_covered
  calc
    72000 / 60 = 1200   : by norm_num
    ... = 600 + platform_length_m : by simp [distance_covered, h3]
    ... = 600 + 600     : by rewrite add_eq_self_iff at distance_covered; assumption
    ... = 1200          : by norm_num
  exact relation_eq_self_of_eq platform_length_m

end platform_length_equals_train_length_l749_749228


namespace red_ball_count_l749_749477

theorem red_ball_count (w : ℕ) (f : ℝ) (total : ℕ) (r : ℕ) 
  (hw : w = 60)
  (hf : f = 0.25)
  (ht : total = w / (1 - f))
  (hr : r = total * f) : 
  r = 20 :=
by 
  -- Lean doesn't require a proof for the problem statement
  sorry

end red_ball_count_l749_749477


namespace equilateral_triangle_area_l749_749330

theorem equilateral_triangle_area (perimeter : ℝ) (h1 : perimeter = 120) :
  ∃ A : ℝ, A = 400 * Real.sqrt 3 ∧
    (∃ s : ℝ, s = perimeter / 3 ∧ A = (Real.sqrt 3 / 4) * (s ^ 2)) :=
by
  sorry

end equilateral_triangle_area_l749_749330


namespace arcsin_half_eq_pi_six_l749_749806

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l749_749806


namespace slope_angle_of_line_l749_749464

theorem slope_angle_of_line (m : ℝ) :
  ∀ θ : ℝ, θ = 30 → θ = real.arctan (-1 / m) → m = -real.sqrt 3 :=
by
  intro θ h1 h2
  sorry

end slope_angle_of_line_l749_749464


namespace three_digit_numbers_with_repeated_digits_l749_749601

theorem three_digit_numbers_with_repeated_digits :
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  total_three_digit_numbers - without_repeats = 252 := by
{
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  show total_three_digit_numbers - without_repeats = 252
  sorry
}

end three_digit_numbers_with_repeated_digits_l749_749601


namespace losing_team_higher_scores_possible_l749_749256

theorem losing_team_higher_scores_possible : 
  ∃ (scores_team1 : list (list ℕ)) (scores_team2 : list (list ℕ)),
  (∀ contest ∈ scores_team1 ++ scores_team2, ∀ score ∈ contest, 1 ≤ score ∧ score ≤ 5) ∧
  scores_team1.length = 4 ∧ scores_team2.length = 4 ∧
  (∀ contest ∈ scores_team1, contest.length = 6) ∧
  (∀ contest ∈ scores_team2, contest.length = 6) ∧
  let avg := λ l : list ℕ, (l.sum : ℚ) / l.length in
  let round_nearest_tenth := λ q : ℚ, (q * 10).round / 10 in
  let rounded_avg_team1 := scores_team1.map (λ c, round_nearest_tenth (avg c)) in
  let rounded_avg_team2 := scores_team2.map (λ c, round_nearest_tenth (avg c)) in
  let total_team1 := rounded_avg_team1.sum in
  let total_team2 := rounded_avg_team2.sum in
  total_team1 > total_team2 ∧ scores_team2.sum (λ l, l.sum) > scores_team1.sum (λ l, l.sum) :=
begin
  -- Proof that it's possible for the losing team to have higher raw scores than the winning team.
  sorry
end

end losing_team_higher_scores_possible_l749_749256


namespace bisector_theorem_l749_749163

-- Define the points and midpoints
variables {A B C D M N : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited M] [Inhabited N]

-- Define the properties of being midpoints
def is_midpoint (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R] (M : Type) [Inhabited M] : Prop := sorry

-- Define angle bisector property
def angle_bisector (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R] (B : Type) [Inhabited B] : Prop := sorry

-- Define cyclic quadrilateral property
def is_cyclic_quadrilateral (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] : Prop := sorry

-- The main theorem statement
theorem bisector_theorem {A B C D M N : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  [Inhabited M] [Inhabited N] 
  (h1: is_cyclic_quadrilateral A B C D) 
  (h2: is_midpoint A C M) 
  (h3: is_midpoint B D N) 
  (h4: angle_bisector A N C B D) 
  : angle_bisector B M D A C :=
sorry

end bisector_theorem_l749_749163


namespace g_is_correct_l749_749842

-- Define the given polynomial equation
def poly_lhs (x : ℝ) : ℝ := 2 * x^5 - x^3 + 4 * x^2 + 3 * x - 5
def poly_rhs (x : ℝ) : ℝ := 7 * x^3 - 4 * x + 2

-- Define the function g(x)
def g (x : ℝ) : ℝ := -2 * x^5 + 6 * x^3 - 4 * x^2 - x + 7

-- The theorem to be proven
theorem g_is_correct : ∀ x : ℝ, poly_lhs x + g x = poly_rhs x :=
by
  intro x
  unfold poly_lhs poly_rhs g
  sorry

end g_is_correct_l749_749842


namespace arcsin_of_half_l749_749820

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l749_749820


namespace lambda_value_l749_749031

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749031


namespace bob_homework_time_l749_749326

variable (T_Alice T_Bob : ℕ)

theorem bob_homework_time (h_Alice : T_Alice = 40) (h_Bob : T_Bob = (3 * T_Alice) / 8) : T_Bob = 15 :=
by
  rw [h_Alice] at h_Bob
  norm_num at h_Bob
  exact h_Bob

-- Assuming T_Alice represents the time taken by Alice to complete her homework
-- and T_Bob represents the time taken by Bob to complete his homework,
-- we prove that T_Bob is 15 minutes given the conditions.

end bob_homework_time_l749_749326


namespace arcsin_one_half_l749_749796

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749796


namespace sin_240_deg_l749_749831

open Real

theorem sin_240_deg :
  sin (240 * pi / 180) = -sqrt 3 / 2 :=
by
  -- Conversion from degrees to radians
  let a := 180 * pi / 180
  let b := 60 * pi / 180
  -- Known values
  have sin180 : sin a = 0 := by sorry
  have cos180 : cos a = -1 := by sorry
  have sin60 : sin b = sqrt 3 / 2 := by sorry
  have cos60 : cos b = 1 / 2 := by sorry
  -- Use sine addition formula
  calc
    sin (a + b) = sin a * cos b + cos a * sin b : by sorry
            ... = 0 * (1 / 2) + (-1) * (sqrt 3 / 2) : by sorry
            ... = -sqrt 3 / 2 : by sorry

end sin_240_deg_l749_749831


namespace distinct_triangles_in_grid_l749_749447

theorem distinct_triangles_in_grid 
  (n m : ℕ) (h_n : n = 2) (h_m : m = 4) :
  let points := n * m in
  points = 8 →
  (∃ valid_triangles : ℕ, valid_triangles = 48) :=
by
  intro h_points
  let total_points := 8
  let points_set := finset.range total_points
  let total_combinations := nat.choose total_points 3
  let degenerate_rows := 2 * nat.choose 4 3
  have h_total_combinations : total_combinations = 56 := 
    by norm_num
  have h_degenerate_cases : degenerate_rows = 8 := 
    by norm_num
  let valid_triangles := total_combinations - degenerate_rows
  use valid_triangles
  have h_valid_triangles : valid_triangles = 48 := 
    by norm_num
  exact h_valid_triangles

end distinct_triangles_in_grid_l749_749447


namespace find_lambda_l749_749075

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749075


namespace problem1_problem2_l749_749108

theorem problem1 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 2 * x ^ 2 + (2 - a) * x - a > 0 ↔ x < -1 ∨ x > 3 / 2) :=
by
  sorry

theorem problem2 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 3 ≥ 0) ↔ (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end problem1_problem2_l749_749108


namespace circle_equation_correct_l749_749870

def line_through_fixed_point (a : ℝ) :=
  ∀ x y : ℝ, (x + y - 1) - a * (x + 1) = 0 → x = -1 ∧ y = 2

def equation_of_circle (x y: ℝ) :=
  (x + 1)^2 + (y - 2)^2 = 5

theorem circle_equation_correct (a : ℝ) (h : line_through_fixed_point a) :
  ∀ x y : ℝ, equation_of_circle x y ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
sorry

end circle_equation_correct_l749_749870


namespace convex_ngon_max_vertices_l749_749195

-- Defining the problem conditions
def convex_ngon_in_grid (n : ℕ) : Prop :=
  ∀ (vertices : Fin n → (ℤ × ℤ)), 
  convex_hull ℝ (range vertices) ∈ lattice_grid ∧
  no_other_grid_vertices (convex_hull ℝ (range vertices))

-- What we need to prove
theorem convex_ngon_max_vertices (n : ℕ) : convex_ngon_in_grid n → n ≤ 4 :=
by { sorry }

end convex_ngon_max_vertices_l749_749195


namespace cost_price_l749_749759

theorem cost_price (sell_price : ℝ) (profit_rate : ℝ) (cost_price : ℝ) : 
  sell_price = 48 ∧ profit_rate = 0.20 → cost_price = 40 :=
begin
  intros h,
  cases h with sell_eq profit_eq,
  have : 1.20 * cost_price = 48,
  { rw [mul_comm, profit_eq, ←sell_eq],
    norm_num,
    sorry }, -- fill in the steps of exact proof here
  linarith,
end

end cost_price_l749_749759


namespace pies_per_day_l749_749311

theorem pies_per_day (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by sorry

end pies_per_day_l749_749311


namespace find_lambda_l749_749016

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749016


namespace excess_percentage_l749_749480

theorem excess_percentage (x : ℝ) 
  (L W : ℝ) (hL : L > 0) (hW : W > 0) 
  (h1 : L * (1 + x / 100) * W * 0.96 = L * W * 1.008) : 
  x = 5 :=
by sorry

end excess_percentage_l749_749480


namespace nat_numbers_in_segment_l749_749211

theorem nat_numbers_in_segment (a : ℕ → ℕ) (blue_index red_index : Set ℕ)
  (cond1 : ∀ i ∈ blue_index, i ≤ 200 → a (i - 1) = i)
  (cond2 : ∀ i ∈ red_index, i ≤ 200 → a (i - 1) = 201 - i) :
    ∀ i, 1 ≤ i ∧ i ≤ 100 → ∃ j, j < 100 ∧ a j = i := 
by
  sorry

end nat_numbers_in_segment_l749_749211


namespace smallest_integer_with_18_divisors_l749_749641

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749641


namespace problem1_problem2_problem3_l749_749154

-- Given increasing function f satisfying a specific functional equation
variable {f : ℝ → ℝ}

-- Define the increasing property on (0, +∞)
def is_increasing (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 < x → 0 < y → x < y → f(x) < f(y)

-- Given condition on the functional equation of f
def satisfies_functional_equation (f : ℝ → ℝ) := ∀ {x y : ℝ}, 0 < x → 0 < y →
  f(x / y) = f(x) - f(y)

-- Problem 1: Prove f(1) = 0
theorem problem1 (h_increasing : is_increasing f) (h_fun_eq : satisfies_functional_equation f) : 
  f 1 = 0 := sorry

-- Problem 2: Prove that f(x-1) < 0 implies x ∈ (1, 2) given f(1) = 0
theorem problem2 (h_increasing : is_increasing f) (h_fun_eq : satisfies_functional_equation f) (h_f_1 : f 1 = 0) : 
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | 1 < x ∧ x < 2} := sorry

-- Problem 3: Prove that f(x+3) - f(1/x) < 2 implies x ∈ (0, 1) given f(2) = 1
theorem problem3 (h_increasing : is_increasing f) (h_fun_eq : satisfies_functional_equation f) (h_f_2 : f 2 = 1) : 
  {x : ℝ | f (x + 3) - f (1 / x) < 2} = {x : ℝ | 0 < x ∧ x < 1} := sorry

end problem1_problem2_problem3_l749_749154


namespace smallest_integer_with_18_divisors_l749_749681

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749681


namespace smallest_positive_integer_with_18_divisors_l749_749668

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749668


namespace lambda_value_l749_749033

def vector (α : Type) := (α × α)

def dot_product (v w : vector ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector ℝ) : Prop :=
  dot_product v w = 0

theorem lambda_value (λ : ℝ) :
  let a := (1,3) : vector ℝ
  let b := (3,4) : vector ℝ
  perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) b → λ = 3 / 5 :=
by
  intros
  sorry

end lambda_value_l749_749033


namespace neq_necessary_not_sufficient_for_sin_l749_749217

theorem neq_necessary_not_sufficient_for_sin (x y : ℝ) (h₁ : x ≠ y) : 
¬ (sin x ≠ sin y ↔ x ≠ y) :=
sorry

end neq_necessary_not_sufficient_for_sin_l749_749217


namespace KO_eq_PL_l749_749175

variables (A B C D K L M N O P : Type*) [euclidean_geometry α]
variables (AD BC AB CD AC BD KM KL : α)
variables (P_on_diag_O : on_diagonal_point O AC)
variables (P_on_diag_P : on_diagonal_point P BD)
variables (P_on_side_M : on_side_point M AB)
variables (P_on_side_N : on_side_point N CD)
variables (KtAK : extends A K AD)
variables (LtCL : extends C L BC)
variables (is_trapezoid : is_trapezoid ABCD)
variables (intersects_M : intersects KL AB M)
variables (intersects_N : intersects KL CD N)
variables (intersects_O : intersects AC BD O)
variables (intersects_P : intersects BD KL P)
variables (KM_eq_NL : KM = NL)

theorem KO_eq_PL : KO = PL := sorry

end KO_eq_PL_l749_749175


namespace find_lambda_l749_749048

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749048


namespace sum_of_remainders_l749_749714

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 3 + n % 6) = 7 :=
by
  sorry

end sum_of_remainders_l749_749714


namespace distance_between_foci_of_hyperbola_l749_749215

theorem distance_between_foci_of_hyperbola :
  let asymptote1 := fun x : ℝ => x + 1
  let asymptote2 := fun x : ℝ => 3 - x
  let hyperbola (x y : ℝ) := (x - 1)^2 / 3 - (y - 2)^2 / 3 = 1
  in hyperbola 3 3 → (∀ x : ℝ, asymptote1 x = 3 - x)
  → ∃ c : ℝ, 2 * c = 2 * Real.sqrt 6 := 
sorry

end distance_between_foci_of_hyperbola_l749_749215


namespace cos_product_pi_over_8_l749_749781

theorem cos_product_pi_over_8 : 
  cos (Real.pi / 8) * cos (5 * Real.pi / 8) = - (Real.sqrt 2 / 4) :=
by 
  have h1 : cos (5 * Real.pi / 8) = cos (Real.pi / 2 + Real.pi / 8) := sorry
  have h2 : cos (Real.pi / 2 + Real.pi / 8) = -sin (Real.pi / 8) := sorry
  have h3 : - sin (Real.pi / 8) * cos (Real.pi / 8) = - (1 / 2) * sin (Real.pi / 4) := sorry
  have h4 : sin (Real.pi / 4) = Real.sqrt 2 / 2 := sorry
  sorry -- complete the proof based on the conditions and the correct answer

end cos_product_pi_over_8_l749_749781


namespace green_or_yellow_probability_l749_749608

-- Given the number of marbles of each color
def green_marbles : ℕ := 4
def yellow_marbles : ℕ := 3
def white_marbles : ℕ := 6

-- The total number of marbles
def total_marbles : ℕ := green_marbles + yellow_marbles + white_marbles

-- The number of favorable outcomes (green or yellow marbles)
def favorable_marbles : ℕ := green_marbles + yellow_marbles

-- The probability of drawing a green or yellow marble as a fraction
def probability_of_green_or_yellow : Rat := favorable_marbles / total_marbles

theorem green_or_yellow_probability :
  probability_of_green_or_yellow = 7 / 13 :=
by
  sorry

end green_or_yellow_probability_l749_749608


namespace arcsin_of_half_l749_749794

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l749_749794


namespace find_lambda_l749_749051

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749051


namespace total_whipped_cream_l749_749875

theorem total_whipped_cream (cream_from_farm : ℕ) (cream_to_buy : ℕ) (total_cream : ℕ) 
  (h1 : cream_from_farm = 149) 
  (h2 : cream_to_buy = 151) 
  (h3 : total_cream = cream_from_farm + cream_to_buy) : 
  total_cream = 300 :=
sorry

end total_whipped_cream_l749_749875


namespace smallest_integer_with_18_divisors_l749_749683

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749683


namespace tangent_lines_at_angle_tangent_lines_through_point_l749_749261

/-- Define the circle and the given point -/
def circle (x y : ℝ) := x^2 + y^2 = 4
def pointP := (0 : ℝ, -10 / 3 : ℝ)

-- prove tangent lines for the given angle and point
theorem tangent_lines_at_angle (x y : ℝ) (α : ℝ) : 
    α = 143 + (7 / 60) + (48.2 / 3600) → 
    (circle x y → (3 * x + 4 * y = 10 ∨ 3 * x + 4 * y = -10)) := 
by
  sorry

theorem tangent_lines_through_point (x y : ℝ) : 
    circle x y → pointP.snd = y →
    (4 * x - 3 * y = 10 ∨ 4 * x + 3 * y = -10) := 
by
  sorry

end tangent_lines_at_angle_tangent_lines_through_point_l749_749261


namespace derivative_at_zero_l749_749453

def f (x : ℝ) : ℝ := sin x * cos x

theorem derivative_at_zero :
  deriv f 0 = 1 :=
sorry

end derivative_at_zero_l749_749453


namespace number_of_members_l749_749756

theorem number_of_members (n : ℕ) (h : n^2 = 9216) : n = 96 :=
sorry

end number_of_members_l749_749756


namespace geometric_sum_formula_sum_series_bound_l749_749153

variable {n : ℕ}
variable {a : ℕ → ℕ} 
variable {b : ℕ → ℕ} 
variable {T : ℕ → ℝ}

noncomputable def geometric_series (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * (2 ^ n) - 2

theorem geometric_sum_formula :
  a 1 = 2 → (∀ n, a (n + 1) = 2 ^ (n + 1)) →
  geometric_series a n = 2^(n+1) - 2 := by
  sorry

def arithmetic_sequence (b : ℕ → ℕ) (n : ℕ) : ℕ := 3 * n - 1

noncomputable def sum_series (T : ℕ → ℝ) (n : ℕ) : ℝ := 
  ∑ k in finset.range n, (2 * (k + 1) + 1) / ((k + 1)^2 * (arithmetic_sequence b (k + 1) + 4)^2)

theorem sum_series_bound :
  (∀ n, T n = sum_series T n) →
  ∀ n, T n ≤ 1 / 9 := by
  sorry

end geometric_sum_formula_sum_series_bound_l749_749153


namespace linear_function_of_additivity_l749_749203

theorem linear_function_of_additivity (f : ℝ → ℝ) 
  (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end linear_function_of_additivity_l749_749203


namespace max_min_modulus_m_l749_749435

theorem max_min_modulus_m (z1 z2 m α β : ℂ) 
    (h1 : z1^2 - 4 * z2 = 16 + 20 * complex.i)
    (h2 : ∃ α β, α ≠ β ∧ |α - β| = 2 * real.sqrt 7 ∧ α + β = -z1 ∧ α * β = z2 + m) :
    real.sqrt (41) + 7 = real.abs m ∨ 7 - real.sqrt (41) = real.abs m :=
begin
  sorry
end

end max_min_modulus_m_l749_749435


namespace complex_power_cos_sin_l749_749787

theorem complex_power_cos_sin (θ : ℝ) (hθ : θ = 210) : 
  (complex.of_real (cos θ) + complex.i * complex.of_real (sin θ))^60 = 1 :=
by
  sorry

end complex_power_cos_sin_l749_749787


namespace probability_of_green_or_yellow_marble_l749_749607

theorem probability_of_green_or_yellow_marble :
  let total_marbles := 4 + 3 + 6 in
  let favorable_marbles := 4 + 3 in
  favorable_marbles / total_marbles = 7 / 13 :=
by
  sorry

end probability_of_green_or_yellow_marble_l749_749607


namespace tan_frac_eq_one_l749_749099

open Real

-- Conditions given in the problem
def sin_frac_cond (x y : ℝ) : Prop := (sin x / sin y) + (sin y / sin x) = 4
def cos_frac_cond (x y : ℝ) : Prop := (cos x / cos y) + (cos y / cos x) = 3

-- Statement of the theorem to be proved
theorem tan_frac_eq_one (x y : ℝ) (h1 : sin_frac_cond x y) (h2 : cos_frac_cond x y) : (tan x / tan y) + (tan y / tan x) = 1 :=
by
  sorry

end tan_frac_eq_one_l749_749099


namespace pupils_like_only_maths_l749_749248

noncomputable def number_pupils_like_only_maths (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) 
(neither_lovers: ℕ) (both_lovers: ℕ) : ℕ :=
maths_lovers - both_lovers

theorem pupils_like_only_maths : 
∀ (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) (neither_lovers: ℕ) (both_lovers: ℕ),
total = 30 →
maths_lovers = 20 →
english_lovers = 18 →
both_lovers = 2 * neither_lovers →
neither_lovers + maths_lovers + english_lovers - both_lovers - both_lovers = total →
number_pupils_like_only_maths total maths_lovers english_lovers neither_lovers both_lovers = 4 :=
by
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end pupils_like_only_maths_l749_749248


namespace smallest_int_with_18_divisors_l749_749656

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749656


namespace find_k_l749_749238

theorem find_k (k : ℕ) (h : (9 * (10^k - 1) / 9: ℤ).digits.sum = 1009) : k = 112 :=
by 
  sorry

end find_k_l749_749238


namespace solution_set_l749_749363

theorem solution_set :
  {x : ℝ | (x / 4 ≤ 3 + 2 * x) ∧ (3 + 2 * x < -3 * (1 + x))} =
  set.Ico (-12 / 7 : ℝ) (-6 / 5 : ℝ) :=
by {
  sorry
}

end solution_set_l749_749363


namespace cost_of_football_and_basketball_max_number_of_basketballs_l749_749859

-- Problem 1: Cost of one football and one basketball
theorem cost_of_football_and_basketball (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 310) 
  (h2 : 2 * x + 5 * y = 500) : 
  x = 50 ∧ y = 80 :=
sorry

-- Problem 2: Maximum number of basketballs
theorem max_number_of_basketballs (x : ℝ) 
  (h1 : 50 * (96 - x) + 80 * x ≤ 5800) 
  (h2 : x ≥ 0) 
  (h3 : x ≤ 96) : 
  x ≤ 33 :=
sorry

end cost_of_football_and_basketball_max_number_of_basketballs_l749_749859


namespace abs_diff_eq_two_l749_749574

theorem abs_diff_eq_two {a1 a2 b1 b2 : ℕ} (ha1a2 : a1 ≥ a2) (hb1b2 : b1 ≥ b2) (hpos : 0 < a1 ∧ 0 < a2 ∧ 0 < b1 ∧ 0 < b2)
    (h_eq : 2100 = (Nat.factorial a1 * Nat.factorial a2) / (Nat.factorial b1 * Nat.factorial b2))
    (h_min : ∀ x y, (2100 = (Nat.factorial x * Nat.factorial a2) / (Nat.factorial y * Nat.factorial b2)) → x + y ≥ a1 + b1) :
    |a1 - b1| = 2 := by
  sorry

end abs_diff_eq_two_l749_749574


namespace proof_P_and_Q_l749_749576

/-!
Proposition P: The line y=2x is perpendicular to the line x+2y=0.
Proposition Q: The projections of skew lines in the same plane could be parallel lines.
Prove: P ∧ Q is true.
-/

def proposition_P : Prop := 
  let slope1 := 2
  let slope2 := -1 / 2
  slope1 * slope2 = -1

def proposition_Q : Prop :=
  ∃ (a b : ℝ), (∃ (p q r s : ℝ),
    (a * r + b * p = 0) ∧ (a * s + b * q = 0)) ∧
    (a ≠ 0 ∨ b ≠ 0)

theorem proof_P_and_Q : proposition_P ∧ proposition_Q :=
  by
  -- We need to prove the conjunction of both propositions is true.
  sorry

end proof_P_and_Q_l749_749576


namespace number_of_unique_three_digit_integers_l749_749425

theorem number_of_unique_three_digit_integers : 
  ∃ n, n = 24 ∧
  (∀ (digits : Finset ℕ), digits = {2, 4, 7, 9} → 
    (∀ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a →
      ∃ l : list ℕ, l.perm [a, b, c] ∧ l.length = 3 ∧ to_nat (from_digits 10 l) ≠ 0)) :=
begin
  sorry
end

end number_of_unique_three_digit_integers_l749_749425


namespace arcsin_one_half_eq_pi_six_l749_749813

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l749_749813


namespace mistaken_divisor_is_12_l749_749989

-- Definitions based on conditions
def correct_divisor : ℕ := 21
def correct_quotient : ℕ := 36
def mistaken_quotient : ℕ := 63

-- The mistaken divisor  is computed as:
def mistaken_divisor : ℕ := correct_quotient * correct_divisor / mistaken_quotient

-- The theorem to prove the mistaken divisor is 12
theorem mistaken_divisor_is_12 : mistaken_divisor = 12 := by
  sorry

end mistaken_divisor_is_12_l749_749989


namespace largest_n_sum_of_digits_l749_749514

def is_prime (n : ℕ) : Prop := nat.prime n

def single_digit_primes : set ℕ := {2, 3, 5, 7}

def valid_n (d e n : ℕ) : Prop :=
  d ∈ single_digit_primes ∧
  e ∈ single_digit_primes ∧
  d ≠ e ∧
  is_prime (12 * d + e) ∧
  n = d * e * (12 * d + e)

def sum_of_digits (n : ℕ) : ℕ :=
  n.to_string.foldr (λ c sum, sum + (c.to_nat - '0'.to_nat)) 0

theorem largest_n_sum_of_digits : ∃ n, (∃ d e, valid_n d e n) ∧ 
  n = 2345 ∧ sum_of_digits n = 14 :=
by
  sorry

end largest_n_sum_of_digits_l749_749514


namespace tan_alpha_l749_749897

noncomputable def tan_of_angle (α : ℝ) (m : ℝ) (h₁ : sin α = m) (h₂ : (π / 2) < α) (h₃ : α < π) (h₄ : |m| < 1) : ℝ :=
  -m / sqrt (1 - m ^ 2)

theorem tan_alpha (α : ℝ) (m : ℝ) (h₁ : sin α = m) (h₂ : (π / 2) < α) (h₃ : α < π) (h₄ : |m| < 1) :
  tan α = -m / sqrt (1 - m ^ 2) :=
  sorry

end tan_alpha_l749_749897


namespace exists_infinitely_many_m_with_2017_successful_nums_l749_749167

/-- A number is called successful if it can be expressed in the form x^3 + y^2 with natural numbers x and y. -/
def is_successful (n : ℕ) : Prop := ∃ x y : ℕ, n = x ^ 3 + y ^ 2

theorem exists_infinitely_many_m_with_2017_successful_nums :
  ∃ᶠ m in at_top, (∃ count : ℕ, count = (Finset.range (2016 ^ 2)).filter (λ k, is_successful (m + k)).card ∧ count = 2017) :=
by
  sorry

end exists_infinitely_many_m_with_2017_successful_nums_l749_749167


namespace part1_monotonicity_part2_find_range_l749_749939

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end part1_monotonicity_part2_find_range_l749_749939


namespace arcsin_of_half_l749_749817

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l749_749817


namespace strip_division_total_parts_l749_749591

def divide_strip (n : ℕ) (L : ℕ) : set ℕ :=
  { k | ∃ i : ℕ, 1 ≤ i ∧ i < n ∧ k = i * (L / n) }

theorem strip_division_total_parts :
  let L := 60 in  -- LCM of 6, 10, and 12
  let cuts := (divide_strip 6 L) ∪ (divide_strip 10 L) ∪ (divide_strip 12 L) in
  cuts.card + 1 = 20 :=
by
  sorry

end strip_division_total_parts_l749_749591


namespace smallest_positive_integer_with_18_divisors_l749_749666

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749666


namespace six_digit_numbers_divisible_by_7_l749_749367

noncomputable def is_valid_number (n : ℕ) : Prop :=
  let A := n / 100000
  let B := (n % 100000) / 10000
  let C := (n % 10000) / 1000
  let num_ABC := 100 * A + 10 * B + C in
  let num_ABC_CBA := 100001 * A + 10010 * B + 1100 * C in
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  A ≠ 0 ∧
  num_ABC % 7 = 0 ∧
  num_ABC_CBA % 7 = 0 ∧
  n = num_ABC_CBA

theorem six_digit_numbers_divisible_by_7 :
  { n : ℕ | is_valid_number n } = {168861, 259952, 861168, 952259} :=
sorry

end six_digit_numbers_divisible_by_7_l749_749367


namespace smallest_int_with_18_divisors_l749_749654

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749654


namespace find_smallest_S_l749_749266

noncomputable def smallest_S (n : ℕ) (dice_sum : ℕ) : ℤ :=
  5 * n - dice_sum

theorem find_smallest_S :
  let n_min := Nat.ceil(980.0 / 6.0) in
  smallest_S n_min 980 = 5 := 
by
  sorry

end find_smallest_S_l749_749266


namespace angle_AOD_is_135_l749_749276

noncomputable def angle_AOD_is_135_degrees : Prop := 
  let A := ⟨-1, 2⟩
  let O := ⟨0, 0⟩
  let D := ⟨3, -1⟩
  let OA := (A.1, A.2)
  let OD := (D.1, D.2)
  let dot_product := OA.fst * OD.fst + OA.snd * OD.snd
  let magnitude_OA := Real.sqrt (OA.fst ^ 2 + OA.snd ^ 2)
  let magnitude_OD := Real.sqrt (OD.fst ^ 2 + OD.snd ^ 2)
  let cos_theta := dot_product / (magnitude_OA * magnitude_OD)
  let angle_in_radians := Real.arccos cos_theta
  let angle_in_degrees := angle_in_radians * (180 / Real.pi)
  angle_in_degrees = 135

theorem angle_AOD_is_135 : angle_AOD_is_135_degrees :=
by sorry

end angle_AOD_is_135_l749_749276


namespace five_dice_probability_l749_749378

noncomputable def probability_even_sum_given_even_product : ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let even_product_outcomes := total_outcomes - odd_outcomes
  let even_sum_outcomes :=
    3^5 + (Nat.choose 5 2 * 3^5) + (Nat.choose 5 4 * 3^5)
  even_sum_outcomes / even_product_outcomes

theorem five_dice_probability :
  probability_even_sum_given_even_product = 3888 / 7533 := by sorry

end five_dice_probability_l749_749378


namespace sin_390_eq_one_half_l749_749353

theorem sin_390_eq_one_half : sin (390 * (real.pi / 180)) = 1 / 2 :=
by 
  sorry

end sin_390_eq_one_half_l749_749353


namespace max_min_sum_l749_749543

theorem max_min_sum (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  let N := max (xy + xz + yz) (fun A => 0 ≤ A ∧ A ≤ 9) 
  let n := min (xy + xz + yz) (fun A => 0 ≤ A ∧ A ≤ 9) 
  in N + 10 * n = 27 :=
sorry

end max_min_sum_l749_749543


namespace smallest_integer_with_18_divisors_l749_749705

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749705


namespace no_nat_numbers_satisfy_l749_749197

theorem no_nat_numbers_satisfy (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k := 
sorry

end no_nat_numbers_satisfy_l749_749197


namespace max_students_l749_749996

-- Definitions for the conditions
noncomputable def courses := ["Mathematics", "Physics", "Biology", "Music", "History", "Geography"]

def most_preferred (ranking : List String) : Prop :=
  "Mathematics" ∈ (ranking.take 2) ∨ "Mathematics" ∈ (ranking.take 3)

def least_preferred (ranking : List String) : Prop :=
  "Music" ∉ ranking.drop (ranking.length - 2)

def preference_constraints (ranking : List String) : Prop :=
  ranking.indexOf "History" < ranking.indexOf "Geography" ∧
  ranking.indexOf "Physics" < ranking.indexOf "Biology"

def all_rankings_unique (rankings : List (List String)) : Prop :=
  ∀ (r₁ r₂ : List String), r₁ ≠ r₂ → r₁ ∈ rankings → r₂ ∈ rankings → r₁ ≠ r₂

-- The goal statement
theorem max_students : 
  ∃ (rankings : List (List String)), 
  (∀ r ∈ rankings, most_preferred r) ∧
  (∀ r ∈ rankings, least_preferred r) ∧
  (∀ r ∈ rankings, preference_constraints r) ∧
  all_rankings_unique rankings ∧
  rankings.length = 44 :=
sorry

end max_students_l749_749996


namespace smallest_integer_with_18_divisors_l749_749633

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≤ 131072) → 
  (∃ d : ℕ, d = 18 → (associated (factors.count d) (num_factors m)) → m = 131072)) :=
sorry

end smallest_integer_with_18_divisors_l749_749633


namespace chord_length_l749_749742

/-- In a circle with a radius of length 15, let CD be a chord that is the perpendicular bisector of the radius. 
The length of the chord CD is 26√3. -/
theorem chord_length (r : ℝ) (h : r = 15) (OC OD : ℝ) (h2 : OC = OD) (h3 : OC = r / 2):
  let CD := 2 * (sqrt (r^2 - (r / 2)^2)) in 
  CD = 26 * sqrt 3 :=
by
  intro r h OC OD h2 h3 CD
  sorry

end chord_length_l749_749742


namespace sequence_periodicity_l749_749242

theorem sequence_periodicity :
  ∃ (a : ℕ → ℝ), 
  (∀ n, (0 ≤ a n ∧ a n < 1) ∧
    (a (n + 1) = if 0 ≤ a n ∧ a n < 1/2 then 2 * a n else 2 * a n - 1)) ∧
  (a 1 = 3/5) ∧
  (a 2015 = 2/5) :=
begin
  sorry
end

end sequence_periodicity_l749_749242


namespace count_ways_to_write_2010_l749_749148

theorem count_ways_to_write_2010 : ∃ N : ℕ, 
  (∀ (a_3 a_2 a_1 a_0 : ℕ), a_0 ≤ 99 ∧ a_1 ≤ 99 ∧ a_2 ≤ 99 ∧ a_3 ≤ 99 → 
    2010 = a_3 * 10^3 + a_2 * 10^2 + a_1 * 10 + a_0) ∧ 
    N = 202 :=
sorry

end count_ways_to_write_2010_l749_749148


namespace cook_remaining_potatoes_l749_749289

def total_time_to_cook_remaining_potatoes (total_potatoes cooked_potatoes time_per_potato : ℕ) : ℕ :=
  (total_potatoes - cooked_potatoes) * time_per_potato

theorem cook_remaining_potatoes 
  (total_potatoes cooked_potatoes time_per_potato : ℕ) 
  (h_total_potatoes : total_potatoes = 13)
  (h_cooked_potatoes : cooked_potatoes = 5)
  (h_time_per_potato : time_per_potato = 6) : 
  total_time_to_cook_remaining_potatoes total_potatoes cooked_potatoes time_per_potato = 48 :=
by
  -- Proof not required
  sorry

end cook_remaining_potatoes_l749_749289


namespace geometric_sequence_seventh_term_l749_749751

theorem geometric_sequence_seventh_term :
  ∃ (r : ℕ), (3 : ℕ) * r^4 = 243 ∧ (3 : ℕ) * r^6 = 2187 :=
by
  use 3
  split
  · calc
      (3 : ℕ) * 3^4 = 3 * 81 := by refl
      ... = 243 := by norm_num
  · calc
      (3 : ℕ) * 3^6 = 3 * 729 := by refl
      ... = 2187 := by norm_num

end geometric_sequence_seventh_term_l749_749751


namespace arcsin_one_half_l749_749823

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749823


namespace smallest_positive_integer_with_18_divisors_l749_749667

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749667


namespace find_lambda_l749_749049

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749049


namespace nested_even_function_l749_749510

-- Defining an even function
def even_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- The property that we want to prove
theorem nested_even_function (g : ℝ → ℝ) (h : even_function g) : even_function (λ x, g (g (g (g x)))) :=
by
  -- Placeholder for the proof
  sorry

end nested_even_function_l749_749510


namespace sum_of_squares_l749_749418

theorem sum_of_squares (a b c : ℝ) (h₁ : a + b + c = 31) (h₂ : ab + bc + ca = 10) :
  a^2 + b^2 + c^2 = 941 :=
by
  sorry

end sum_of_squares_l749_749418


namespace value_less_than_mean_by_std_dev_l749_749558

theorem value_less_than_mean_by_std_dev :
  ∀ (mean value std_dev : ℝ), mean = 16.2 → std_dev = 2.3 → value = 11.6 → 
  (mean - value) / std_dev = 2 :=
by
  intros mean value std_dev h_mean h_std_dev h_value
  -- The proof goes here, but per instructions, it is skipped
  -- So we put 'sorry' to indicate that the proof is intentionally left incomplete
  sorry

end value_less_than_mean_by_std_dev_l749_749558


namespace probability_is_correct_l749_749375

/-- Representation of a standard six-sided die. -/
def is_standard_die (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6

/-- Number of ways to roll five standard dice such that the product of their values is even. -/
def even_product_configuration : Finset (Fin 6 → ℕ) :=
  (Finset.univ.filter (λ v, ∃ i, is_standard_die (v i) ∧ v i % 2 = 0))

/-- Number of ways to roll five standard dice such that the sum of their values is even and product is even. -/
def even_sum_given_even_product_configuration : Finset (Fin 6 → ℕ) :=
  even_product_configuration.filter (λ v, (Finset.univ.sum (λ i, v i)) % 2 = 0)

/-- The number of valid outcomes where the product of dice values is even. -/
def valid_even_product_outcomes : ℕ := 6^5 - 3^5

/-- The number of valid outcomes where the sum of dice values is even given that their product is even. -/
def valid_even_sum_given_even_product_outcomes : ℕ :=
  even_sum_given_even_product_configuration.card

/-- The probability that the sum of the dice values is even given that the product of their values is even. -/
def probability_even_sum_given_even_product : ℚ :=
  valid_even_sum_given_even_product_outcomes / valid_even_product_outcomes

theorem probability_is_correct :
  probability_even_sum_given_even_product = 1296 / 2511 := 
sorry

end probability_is_correct_l749_749375


namespace arcsin_of_half_l749_749788

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l749_749788


namespace count_numbers_with_at_most_three_digits_l749_749093

theorem count_numbers_with_at_most_three_digits :
  let numbers := {n : ℕ | n < 10000 ∧ 
                           ((∃ d1, ∀ i ∈ n.digits 10, i = d1) ∨
                            (∃ d1 d2, d1 ≠ d2 ∧ ∀ i ∈ n.digits 10, i = d1 ∨ i = d2) ∨
                            (∃ d1 d2 d3, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ ∀ i ∈ n.digits 10, i = d1 ∨ i = d2 ∨ i = d3)) } in
  numbers.card = 4119 :=
by admit

end count_numbers_with_at_most_three_digits_l749_749093


namespace midpoint_of_AB_l749_749411

variables (A B : ℝ × ℝ)
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_of_AB :
  ∀ (A B : ℝ × ℝ), A = (1, 2) → B = (3, -2) → midpoint A B = (2, 0) :=
by
  intro A B hA hB
  rw [hA, hB]
  unfold midpoint
  sorry

end midpoint_of_AB_l749_749411


namespace part1_monotonicity_part2_find_range_l749_749942

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end part1_monotonicity_part2_find_range_l749_749942


namespace smallest_int_with_18_divisors_l749_749659

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l749_749659


namespace arithmetic_progression_sum_l749_749405

theorem arithmetic_progression_sum (a : ℕ → ℕ) (h_positive : ∀ n, a n > 0)
  (h_a1 : a 1 = 1) (h_inc : ∀ n, a (n + 1) > a n) 
  (h_recur : ∀ n, a (n + 2) + a n = 2 * a (n + 1)) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) :
  ∑ k in finset.range n, a k = n * (n + 1) / 2 :=
by
  sorry

end arithmetic_progression_sum_l749_749405


namespace no_3_digit_even_numbers_with_digit_sum_27_divisible_by_3_l749_749958

theorem no_3_digit_even_numbers_with_digit_sum_27_divisible_by_3 :
  ¬ ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n.digits.sum = 27) :=
sorry

end no_3_digit_even_numbers_with_digit_sum_27_divisible_by_3_l749_749958


namespace total_weight_proof_l749_749595

-- Definitions of the conditions in the problem.
def bags_on_first_trip : ℕ := 10
def common_ratio : ℕ := 2
def number_of_trips : ℕ := 20
def weight_per_bag_kg : ℕ := 50

-- Function to compute the total number of bags transported.
noncomputable def total_number_of_bags : ℕ :=
  bags_on_first_trip * (1 - common_ratio^number_of_trips) / (1 - common_ratio)

-- Function to compute the total weight of onions harvested.
noncomputable def total_weight_of_onions : ℕ :=
  total_number_of_bags * weight_per_bag_kg

-- Theorem stating that the total weight of onions harvested is 524,287,500 kgs.
theorem total_weight_proof : total_weight_of_onions = 524287500 := by
  sorry

end total_weight_proof_l749_749595


namespace find_lambda_l749_749071

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749071


namespace sin_minus_cos_third_quadrant_l749_749912

theorem sin_minus_cos_third_quadrant (α : ℝ) (h_tan : Real.tan α = 2) (h_quadrant : π < α ∧ α < 3 * π / 2) : 
  Real.sin α - Real.cos α = -Real.sqrt 5 / 5 := 
by 
  sorry

end sin_minus_cos_third_quadrant_l749_749912


namespace sum_of_valid_m_l749_749867

def is_valid_m (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 300 ∧ (∀ n : ℕ, n ≥ 2 → 2013 * m ∣ (n^n - 1) → 2013 * m ∣ (n - 1))

theorem sum_of_valid_m : (∑ m in Finset.filter (is_valid_m) (Finset.range 301), m) = 4650 :=
sorry

end sum_of_valid_m_l749_749867


namespace squirrel_paths_l749_749443

/-- A type representing the points on the grid. -/
inductive Point
| P | A | B | C | D | E | F | G | H | I | K | L | Q

open Point

/-- A function representing the valid movements of the squirrel, which is always getting closer to Q and farther from P. -/
def valid_move : Point → Point → Prop
| P A := true
| P B := true
| A C := true
| A D := true
| B D := true
| C F := true
| C G := true
| D E := true
| D G := true
| E L := true
| F H := true
| G K := true
| H Q := true
| K Q := true
| L Q := true
| _ _ := false

/-- A function that recursively counts the number of paths from a starting point to the endpoint Q. -/
def count_paths : Point → Nat 
| Q := 1
| start := (list.filter (λ (p : Point), valid_move start p) [P, A, B, C, D, E, F, G, H, I, K, L, Q]).map count_paths |>.sum

theorem squirrel_paths : count_paths P = 14 := by
  sorry

end squirrel_paths_l749_749443


namespace arcsin_of_half_l749_749793

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l749_749793


namespace max_visible_unit_cubes_12_cube_l749_749738

theorem max_visible_unit_cubes_12_cube : 
  let n := 12 in 
  let unit_cubes := n^3 in 
  let face_cubes := n^2 in 
  (3 * face_cubes) - (3 * (n - 1)) + 1 = 400 := 
by 
  let n := 12
  let unit_cubes := n ^ 3
  let face_cubes := n ^ 2
  have visible_cubes := (3 * face_cubes) - (3 * (n - 1)) + 1
  have h_eq : visible_cubes = 400 := by sorry
  exact h_eq

end max_visible_unit_cubes_12_cube_l749_749738


namespace smallest_positive_integer_with_18_divisors_l749_749695

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749695


namespace part1_part2_l749_749427

variable (x : ℝ)

def f (x : ℝ) : ℝ := 2 * sin (x + π / 6)

theorem part1 (h : sin x = 4 / 5) (hx : π / 2 ≤ x ∧ x ≤ π) : f x = (4 * sqrt 3 + 3) / 5 := sorry

theorem part2 (hx : π / 2 ≤ x ∧ x ≤ π) : (1 : ℝ) ≤ f x ∧ f x ≤ 2 := sorry

end part1_part2_l749_749427


namespace sum_primes_below_20_without_eleven_l749_749708

def primes_below_20 := [2, 3, 5, 7, 11, 13, 17, 19]

def without_eleven := primes_below_20.filter (≠ 11)

def expected_sum := 66

theorem sum_primes_below_20_without_eleven :
  (without_eleven.sum = expected_sum) :=
by
  sorry

end sum_primes_below_20_without_eleven_l749_749708


namespace problem1_problem2_l749_749885

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 2) * x + 4

theorem problem1 (a : ℝ) :
  (∀ x, f a x > 0) → 0 < a ∧ a < 4 :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x, -3 <= x ∧ x <= 1 → f a x > 0) → (-1/2 < a ∧ a < 4) :=
sorry

end problem1_problem2_l749_749885


namespace polynomial_not_divisible_by_x_minus_5_l749_749967

theorem polynomial_not_divisible_by_x_minus_5 (m : ℝ) :
  (∀ x, x = 4 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) →
  ¬(∀ x, x = 5 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) :=
by
  sorry

end polynomial_not_divisible_by_x_minus_5_l749_749967


namespace count_isosceles_numbers_l749_749165

theorem count_isosceles_numbers : 
  let n := fin 9 in
  (card { n : fin 900 // 
           let (a, b, c) := (n / 100 + 1, (n % 100) / 10 + 1, n % 10 + 1) in
           (a = b ∨ a = c ∨ b = c) ∧
           (a, b, c).all (λ x => 1 ≤ x ∧ x ≤ 9) 
          }) = 165 :=
by
  sorry

end count_isosceles_numbers_l749_749165


namespace angle_ECA_is_100_l749_749129

-- Define the angles
variables (A B C D E : Point)
variables (α β γ δ ε : ℝ) -- angles in degrees

-- Conditions
axiom h1 : parallel DC AB
axiom h2 : ∠DCA = 50
axiom h3 : ∠ABC = 60
axiom h4 : on_line AB E
axiom h5 : ∠BAE = 20

-- Theorem to prove
theorem angle_ECA_is_100 : ∠ECA = 100 :=
by
  sorry

end angle_ECA_is_100_l749_749129


namespace necessary_and_sufficient_condition_l749_749438

open Set

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 ^ 2}

noncomputable def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 ^ 2 + (p.2 - a) ^ 2 ≤ 1}

theorem necessary_and_sufficient_condition (a : ℝ) :
  N a ⊆ M ↔ a ≥ 5 / 4 := sorry

end necessary_and_sufficient_condition_l749_749438


namespace statue_original_cost_l749_749727

theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) 
    (h1 : selling_price = 670) (h2 : profit_percentage = 0.25) :
    let cost_profit_ratio := 1 + profit_percentage in
    original_cost = selling_price / cost_profit_ratio →
    original_cost = 536 :=
begin
  intros,
  sorry
end

end statue_original_cost_l749_749727


namespace part_1_part_2_l749_749317

noncomputable def question_1 (x : ℝ) : Prop := 
  (x ≤ 60) ∧ (49 * (50 - 3 * (x - 50) / 0.5) ≤ 686)

theorem part_1 (x : ℝ) : 
  (question_1 x) → (56 ≤ x ∧ x ≤ 60) := 
sorry

noncomputable def question_2 (a : ℝ) : Prop := 
  let t := a / 100 in
  (32 * t^2 - 12 * t - 1 = 0) ∧ 
  ¬ (91 * (1 - 2 * t) ∉ ℕ)

theorem part_2 (a : ℝ) : 
  (question_2 a) → (a = 25) := 
sorry

end part_1_part_2_l749_749317


namespace integer_points_on_line_in_region_l749_749092

theorem integer_points_on_line_in_region :
  let line := λ x : ℤ, 4 * x + 3
  ∃! n : ℕ, n = 32 ∧ (∀ x y : ℤ, 25 ≤ x ∧ x ≤ 75 ∧ 120 ≤ y ∧ y ≤ 250 ∧ y = line x) :=
begin
  sorry
end

end integer_points_on_line_in_region_l749_749092


namespace polynomial_degree_pow_l749_749604

noncomputable def f : ℝ[X] := 5 * X^3 + 7 * X^2 + 4

theorem polynomial_degree_pow (n : ℕ) (h : n = 10) :
  degree (f ^ n) = 30 :=
by
  have h_deg : degree f = 3 := sorry -- Based on our given condition
  rw [h, pow_eq_pow, nat_degree_mul, nat_degree_add_eq_right_of_nat_degree_lt (nat_degree_add_eq_left_of_nat_degree_lt (nat_degree_C_mul_X_pow _
    (nat_degree_add_eq_right_of_nat_degree_lt nat_degree_C_mul_X_pow_prime _ _ _)), nat_degree_l_eq_of_nat_degree_lt _)],
  exact sorry

end polynomial_degree_pow_l749_749604


namespace gammas_wins_prob_l749_749487

-- Define probabilities and conditions
def prob_SF_home_win := 0.5
def prob_OAK_home_win := 0.6
def earthquake_chance_SF := 0.5

-- Define the functions F and A
-- F(x) is the probability Gammas will win if they are ahead by x games and next game is in SF
-- A(x) is the probability Gammas will win if they are ahead by x games and next game is in Oakland
def F : ℤ → ℝ
def A : ℤ → ℝ


-- Conditions and recursive definitions given in problem
axiom F_2_def: F 2 = 3/4 + A 1 / 4
axiom A_1_def: A 1 = 6 * F 0 / 10 + 4 * F 2 / 10
axiom F_0_def: F 0 = 1/4 + A 1 / 4 + A (-1) / 4
axiom A_neg1_def: A (-1) = 6 * F (-2) / 10 + 4 * F 0 / 10
axiom F_neg2_def: F (-2) = A (-1) / 4

-- Final statement we aim to prove
theorem gammas_wins_prob : F 0 = 34 / 73 :=
by sorry

end gammas_wins_prob_l749_749487


namespace angle_C_in_parallelogram_l749_749481

theorem angle_C_in_parallelogram (ABCD : Quadrilateral)
  (h_parallelogram : is_parallelogram ABCD)
  (h_angle_A : angle_A ABCD = 150) : angle_C ABCD = 150 :=
sorry

end angle_C_in_parallelogram_l749_749481


namespace non_negative_real_solutions_to_system_l749_749731

noncomputable def k : ℝ := Real.sqrt (Real.sqrt 2 - 1)

theorem non_negative_real_solutions_to_system :
  ∀ x : Fin 1999 → ℝ,
    (∀ i, 0 ≤ x i) →
    (x 1 ^ 2 + x 0 * x 1 + x 0 ^ 4 = 1) →
    (∃ x : Fin 1999 → ℝ, ∀ i, x ((i + 1) % 1999) ^ 2 + x i * x ((i + 1) % 1999) + x i ^ 4 = 1) →
    ∀ i, x i = Real.sqrt (Real.sqrt 2 - 1) :=
begin
  sorry
end

end non_negative_real_solutions_to_system_l749_749731


namespace train_crossing_time_l749_749319

theorem train_crossing_time {
  (length_train : ℕ) (speed_kmph : ℕ) (length_bridge : ℕ) : length_train = 120 → speed_kmph = 45 → length_bridge = 255 → 
  let total_distance := length_train + length_bridge in 
  let speed_mps := (speed_kmph * 1000) / 3600 in 
  let time_seconds := total_distance / speed_mps in 
  time_seconds = 30 :=
begin
  intros h1 h2 h3,
  rw h1,
  rw h2,
  rw h3,
  -- total_distance = 375
  have h_total_distance : total_distance = 120 + 255 := rfl,
  rw h_total_distance,
  -- speed in m/s = 12.5
  have h_speed_mps : speed_mps = (45 * 1000) / 3600 := rfl,
  rw h_speed_mps,
  -- proving the final time_seconds = 30
  have h_time_seconds : time_seconds = 375 / 12.5 := rfl,
  rw h_time_seconds,
  norm_num,
end

end train_crossing_time_l749_749319


namespace sum_of_interior_angles_l749_749196

theorem sum_of_interior_angles (n : ℕ) (h : n ≥ 3) : 
  ∑ (i : fin n), interior_angle i = (n - 2) * 180 := 
sorry

end sum_of_interior_angles_l749_749196


namespace lambda_solution_l749_749063

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749063


namespace min_additional_cells_l749_749406

-- Definitions based on conditions
def num_cells_shape : Nat := 32
def side_length_square : Nat := 9
def area_square : Nat := side_length_square * side_length_square

-- The statement to prove
theorem min_additional_cells (num_cells_given : Nat := num_cells_shape) 
(side_length : Nat := side_length_square)
(area : Nat := area_square) :
  area - num_cells_given = 49 :=
by
  sorry

end min_additional_cells_l749_749406


namespace minimum_keys_needed_l749_749997

theorem minimum_keys_needed (total_cabinets : ℕ) (boxes_per_cabinet : ℕ)
(boxes_needed : ℕ) (boxes_per_cabinet : ℕ) 
(warehouse_key : ℕ) (boxes_per_cabinet: ℕ)
(h1 : total_cabinets = 8)
(h2 : boxes_per_cabinet = 4)
(h3 : (boxes_needed = 52))
(h4 : boxes_per_cabinet = 4)
(h5 : warehouse_key = 1):
    6 + 2 + 1 = 9 := 
    sorry

end minimum_keys_needed_l749_749997


namespace count_numbers_containing_6_and_7_l749_749448

theorem count_numbers_containing_6_and_7 : 
  -- Define our set of interest
  let S := { n : ℕ | 800 ≤ n ∧ n < 1500 } in
  -- Capture numbers in S that contain both digits 6 and 7
  let contains_6_and_7 (n : ℕ) : Prop := 
    (n / 10 % 10 = 6 ∨ n % 10 = 6) ∧ (n / 10 % 10 = 7 ∨ n % 10 = 7) in
  -- Prove the number of such numbers is 12
  finset.card (finset.filter contains_6_and_7 (finset.filter (λ n : ℕ, n ∈ S) (finset.range 1500))) = 12 :=
begin
  sorry
end

end count_numbers_containing_6_and_7_l749_749448


namespace MissAisha_height_l749_749171

theorem MissAisha_height (H : ℝ)
  (legs_length : ℝ := H / 3)
  (head_length : ℝ := H / 4)
  (rest_body_length : ℝ := 25) :
  H = 60 :=
by sorry

end MissAisha_height_l749_749171


namespace find_number_l749_749362

theorem find_number (l m : ℕ) (n : ℕ) [Fact (n = 2^l * 3^m)] :
    (∑ d in (finset.divisors n), d) = 403 → n = 144 :=
by
  sorry

end find_number_l749_749362


namespace average_age_of_remaining_people_l749_749559

theorem average_age_of_remaining_people (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ)
  (h1 : avg_age = 28) (h2 : num_people = 7) (h3 : leaving_age = 20) :
  (avg_age * num_people - leaving_age) / (num_people - 1) = 29.33 :=
by
  sorry

end average_age_of_remaining_people_l749_749559


namespace ratio_area_perimeter_eq_sqrt3_l749_749624

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l749_749624


namespace greatest_power_of_3_in_factorial_product_l749_749102

theorem greatest_power_of_3_in_factorial_product :
  ∃ k : ℕ, (3^k ∣ (∏ i in Finset.range 101, i + 1)) ∧ k = 48 :=
sorry

end greatest_power_of_3_in_factorial_product_l749_749102


namespace find_x_l749_749128

-- Definitions and given conditions
variables {A B C D G E F : Type}
variables {y DE EF : ℝ}
variables (collinear_A_B_G : collinear ℝ ![A, B, G])
variables (square_ABCD : square ℝ ![A, B, C, D])
variables (intersect_AC_DG_E : intersection_point ℝ ![A, C, D, G] E)
variables (intersect_DG_BC_F : intersection_point ℝ ![D, G, B, C] F)
variables (DE_len : DE = 15)
variables (EF_len : EF = 9)
variables (y_len : ∀ (a b c d : ℝ), side_length ℝ ![a, b, c, d] = y)

-- target statement
theorem find_x {FG : ℝ} (find_x : FG = 16) :
  ∀ (A B C D G E F : Type) (collinear_A_B_G : collinear ℝ ![A, B, G])
  (square_ABCD : square ℝ ![A, B, C, D])
  (intersect_AC_DG_E : intersection_point ℝ ![A, C, D, G] E)
  (intersect_DG_BC_F : intersection_point ℝ ![D, G, B, C] F)
  (DE_len : DE = 15) (EF_len : EF = 9) (y_len : ∀ (a b c d : ℝ), side_length ℝ ![a, b, c, d] = y),
  FG = 16 := 
begin
  sorry
end

end find_x_l749_749128


namespace lcm_triples_count_l749_749151

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem lcm_triples_count : 
  let valid_triplet_count : ℕ := 
    { t : ℕ × ℕ × ℕ // 
          let a := t.1 
          let b := t.2.1
          let c := t.2.2
          lcm a b = 800 ∧ lcm b c = 1600 ∧ lcm c a = 1600 
    }.card 
  in valid_triplet_count = 12 := by
  sorry

end lcm_triples_count_l749_749151


namespace abigail_score_l749_749992

theorem abigail_score (sum_20 : ℕ) (sum_21 : ℕ) (h1 : sum_20 = 1700) (h2 : sum_21 = 1806) : (sum_21 - sum_20) = 106 :=
by
  sorry

end abigail_score_l749_749992


namespace tallest_is_jie_l749_749983

variable (Igor Jie Faye Goa Han : Type)
variable (Shorter : Type → Type → Prop) -- Shorter relation

axiom igor_jie : Shorter Igor Jie
axiom faye_goa : Shorter Goa Faye
axiom jie_faye : Shorter Faye Jie
axiom han_goa : Shorter Han Goa

theorem tallest_is_jie : ∀ p, p = Jie :=
by
  sorry

end tallest_is_jie_l749_749983


namespace find_lambda_l749_749000

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749000


namespace false_same_side_interior_angles_no_parallel_l749_749270

theorem false_same_side_interior_angles_no_parallel (h_parallel: ∀ a b : Line, ¬ (a ∥ b)) : ¬ (∀ α β : Angle, same_side_interior α β → supplementary α β) :=
by
  sorry

end false_same_side_interior_angles_no_parallel_l749_749270


namespace monotonicity_a_eq_1_range_of_a_l749_749934

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_a_eq_1 :
  (∀ x : ℝ, 0 < x → (exp x + 2 * x - 1 > 0)) ∧
  (∀ x : ℝ, x < 0 → (exp x + 2 * x - 1 < 0)) := sorry

-- Part 2: Range of a for f(x) ≥ 1/2 * x ^ 3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 <= x → (exp x + a * x^2 - x >= 1/2 * x^3 + 1)) ↔
  (a ≥ (7 - exp 2) / 4) := sorry

end monotonicity_a_eq_1_range_of_a_l749_749934


namespace trailing_zeros_in_hundred_factorial_to_the_hundred_l749_749336

theorem trailing_zeros_in_hundred_factorial_to_the_hundred:
  nat.trailing_zeros ((nat.factorial 100) ^ 100) = 2400 :=
sorry

end trailing_zeros_in_hundred_factorial_to_the_hundred_l749_749336


namespace find_lambda_l749_749041

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749041


namespace bus_commutes_three_times_a_week_l749_749253

-- Define the commuting times
def bike_time := 30
def bus_time := bike_time + 10
def friend_time := bike_time * (1 - (2/3))
def total_weekly_time := 160

-- Define the number of times taking the bus as a variable
variable (b : ℕ)

-- The equation for total commuting time
def commuting_time_eq := bike_time + bus_time * b + friend_time = total_weekly_time

-- The proof statement: b should be equal to 3
theorem bus_commutes_three_times_a_week (h : commuting_time_eq b) : b = 3 := sorry

end bus_commutes_three_times_a_week_l749_749253


namespace common_ratio_of_geometric_sequence_l749_749132

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 2)
  (h2 : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) :
  (∃ a1 : ℝ, ∃ q : ℝ,
    (∀ n, a n = a1 * q ^ (n - 1)) ∧ 
    q = 2) := 
by 
  sorry

end common_ratio_of_geometric_sequence_l749_749132


namespace positive_integers_count_count_positive_integers_l749_749385

open Nat

theorem positive_integers_count (x : ℕ) :
  (225 ≤ x * x ∧ x * x ≤ 400) ↔ (x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19 ∨ x = 20) :=
by sorry

theorem count_positive_integers : 
  Finset.card { x : ℕ | 225 ≤ x * x ∧ x * x ≤ 400 } = 6 :=
by
  sorry

end positive_integers_count_count_positive_integers_l749_749385


namespace sin_double_angle_neg_one_l749_749896

open Real

theorem sin_double_angle_neg_one (α : ℝ) (h1 : sin α - cos α = √2) (h2 : 0 < α ∧ α < π) : sin (2 * α) = -1 := 
sorry

end sin_double_angle_neg_one_l749_749896


namespace non_neg_int_solutions_l749_749365

theorem non_neg_int_solutions : 
  ∀ (x y : ℕ), 2 * x ^ 2 + 2 * x * y - x + y = 2020 → 
               (x = 0 ∧ y = 2020) ∨ (x = 1 ∧ y = 673) :=
by
  sorry

end non_neg_int_solutions_l749_749365


namespace find_a_b_and_water_usage_l749_749252

noncomputable def water_usage_april (a : ℝ) :=
  (15 * (a + 0.8) = 45)

noncomputable def water_usage_may (a b : ℝ) :=
  (17 * (a + 0.8) + 8 * (b + 0.8) = 91)

noncomputable def water_usage_june (a b x : ℝ) :=
  (17 * (a + 0.8) + 13 * (b + 0.8) + (x - 30) * 6.8 = 150)

theorem find_a_b_and_water_usage :
  ∃ (a b x : ℝ), water_usage_april a ∧ water_usage_may a b ∧ water_usage_june a b x ∧ a = 2.2 ∧ b = 4.2 ∧ x = 35 :=
by {
  sorry
}

end find_a_b_and_water_usage_l749_749252


namespace no_solutions_l749_749863

theorem no_solutions : ¬ ∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end no_solutions_l749_749863


namespace triangle_formation_conditions_l749_749263

theorem triangle_formation_conditions (a b c : ℝ) :
  (a + b > c ∧ |a - b| < c) ↔ (a + b > c ∧ b + c > a ∧ c + a > b ∧ |a - b| < c ∧ |b - c| < a ∧ |c - a| < b) :=
sorry

end triangle_formation_conditions_l749_749263


namespace quadrilateral_is_rectangle_l749_749523

variable {A B C D : Type}
variables [OrderedCommRing A] [HasArea A B C D]

noncomputable def is_rectangle (ABCD : ConvexQuadrilateral) : Prop :=
  ∃ (AB CD AD BC : ℝ), 
    ABCD.area = (AB + CD) / 2 * (AD + BC) / 2 → 
    ABCD.isRectangle

theorem quadrilateral_is_rectangle
  (ABCD : ConvexQuadrilateral)
  (h : ABCD.area = (ABCD.AB + ABCD.CD) / 2 * (ABCD.AD + ABCD.BC) / 2) :
  ABCD.isRectangle :=
by
  -- proof here
  sorry

end quadrilateral_is_rectangle_l749_749523


namespace remaining_money_l749_749185

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end remaining_money_l749_749185


namespace problem_p3_l749_749282

noncomputable def count_perfect_squares (n : ℕ) : ℕ :=
  Nat.floor (Real.sqrt n)

noncomputable def count_perfect_cubes (n : ℕ) : ℕ :=
  Nat.floor (Real.cbrt n)

noncomputable def count_perfect_sixth_powers (n : ℕ) : ℕ :=
  Nat.floor (Real.root 6 n)

noncomputable def count_either_perfect_squares_or_cubes_not_both (n : ℕ) : ℕ :=
  let squares := count_perfect_squares n
  let cubes := count_perfect_cubes n
  let sixth_powers := count_perfect_sixth_powers n
  squares + cubes - 2 * sixth_powers

theorem problem_p3 : count_either_perfect_squares_or_cubes_not_both 2019 = 47 := by
  sorry

end problem_p3_l749_749282


namespace find_lambda_l749_749073

noncomputable def a : ℝ × ℝ := (1, 3)
noncomputable def b : ℝ × ℝ := (3, 4)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749073


namespace arcsin_of_half_l749_749819

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l749_749819


namespace arcsin_one_half_l749_749800

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749800


namespace smallest_integer_with_18_divisors_l749_749700

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749700


namespace recurrence_relation_solution_l749_749155

theorem recurrence_relation_solution (f : ℕ → ℕ)
  (h1 : f 1 = 1)
  (h2 : f 2 = 3)
  (h3 : ∀ n, 2 * f (n - 2) + f (n - 1) = f n) :
  ∀ n, f n = (2^(n+1) + (-1)^n)/3 :=
by
  sorry

end recurrence_relation_solution_l749_749155


namespace avg_weight_of_13_children_l749_749216

-- Definitions based on conditions:
def boys_avg_weight := 160
def boys_count := 8
def girls_avg_weight := 130
def girls_count := 5

-- Calculation to determine the total weights
def boys_total_weight := boys_avg_weight * boys_count
def girls_total_weight := girls_avg_weight * girls_count

-- Combined total weight
def total_weight := boys_total_weight + girls_total_weight

-- Average weight calculation
def children_count := boys_count + girls_count
def avg_weight := total_weight / children_count

-- The theorem to prove:
theorem avg_weight_of_13_children : avg_weight = 148 := by
  sorry

end avg_weight_of_13_children_l749_749216


namespace functions_equal_to_f_l749_749924

def f (x : ℝ) := |x|
def h (x : ℝ) := real.sqrt (x^2)
def p (x : ℝ) : ℝ := if x >= 0 then x else -x

theorem functions_equal_to_f :
  (∀ x, f x = h x) ∧ (∀ x, f x = p x) :=
by
  sorry

end functions_equal_to_f_l749_749924


namespace three_digit_number_mul_seven_results_638_l749_749772

theorem three_digit_number_mul_seven_results_638 (N : ℕ) 
  (hN1 : 100 ≤ N) 
  (hN2 : N < 1000)
  (hN3 : ∃ (x : ℕ), 7 * N = 1000 * x + 638) : N = 234 := 
sorry

end three_digit_number_mul_seven_results_638_l749_749772


namespace f_is_odd_max_f_prime_l749_749478

noncomputable def f (x : ℝ) : ℝ :=
  (0).sum (λ n, if n.odd ∧ n ≤ 13 then (Real.sin (n * x)) / n else 0)

theorem f_is_odd :
  ∀ x : ℝ, f (-x) = -f (x) := by
  sorry

theorem max_f_prime :
  ∃ (x : ℝ), (f' x) = 7 := by
  sorry

end f_is_odd_max_f_prime_l749_749478


namespace jim_needs_more_miles_l749_749275

-- Define the conditions
def totalMiles : ℕ := 1200
def drivenMiles : ℕ := 923

-- Define the question and the correct answer
def remainingMiles : ℕ := totalMiles - drivenMiles

-- The theorem statement
theorem jim_needs_more_miles : remainingMiles = 277 :=
by
  -- This will contain the proof which is to be done later
  sorry

end jim_needs_more_miles_l749_749275


namespace a_seq_general_term_l749_749893

noncomputable def a_seq : ℕ → ℕ
| 0       := 1 -- Note: Lean index starts from 0, so we adjust accordingly
| (n + 1) := 2 * a_seq n + 1

theorem a_seq_general_term (n : ℕ) : a_seq n = 2^n - 1 :=
sorry

end a_seq_general_term_l749_749893


namespace part_I_monotonicity_part_II_f_minimum_l749_749528

-- Define the function f(x) for given a and x > -1
def f (a x : ℝ) : ℝ := exp x - (a * x) / (x + 1)

-- Define the condition where x > -1
axiom x_gt_neg_one (x : ℝ) : x > -1

-- Part (I): Show monotonicity of f(x) when a = 1
theorem part_I_monotonicity (x : ℝ) (h : x > -1) :
  (f 1 x ≤ f 1 0) ∨ (f 1 x ≥ f 1 0) := sorry

-- Part (II): Prove that f(x0) ≤ 1 when a > 0 and f(x) attains its minimum at x = x0
theorem part_II_f_minimum (x₀ a : ℝ) (h₁ : a > 0) (h₂ : ∀ (x : ℝ), f a x ≥ f a x₀) :
  f a x₀ ≤ 1 := sorry

end part_I_monotonicity_part_II_f_minimum_l749_749528


namespace lambda_value_l749_749027

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749027


namespace unique_function_satisfying_equation_l749_749862

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, (∀ x y : ℝ, f(x + f(y)) = 2 * x + 2 * f(y + 1)) ∧ (∀ x : ℝ, f(x) = 2 * x + 4) :=
by
  sorry

end unique_function_satisfying_equation_l749_749862


namespace find_lambda_l749_749050

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749050


namespace obtuse_triangle_third_vertex_l749_749259

noncomputable def area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem obtuse_triangle_third_vertex :
  ∃ (x : ℝ), x < 0 ∧ (∃ (p3 : ℝ × ℝ), p3 = (x, 0) ∧ 
  area (4, -5) (0, 0) p3 = 40) :=
begin
  use -16,
  split,
  { linarith },
  { use (-16, 0),
    split,
    { refl },
    { simp only [area],
      norm_num, sorry } }
end

end obtuse_triangle_third_vertex_l749_749259


namespace find_lambda_l749_749057

def vec := (ℝ × ℝ)
def a : vec := (1, 3)
def b : vec := (3, 4)
def orthogonal (u v : vec) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) : λ = 3 / 5 := 
by
  -- proof not required
  sorry

end find_lambda_l749_749057


namespace geometry_problem_z_eq_87_deg_l749_749343

noncomputable def measure_angle_z (ABC ABD ADB : Real) : Real :=
  43 -- \angle ADB

theorem geometry_problem_z_eq_87_deg
  (ABC : Real)
  (h1 : ABC = 130)
  (ABD : Real)
  (h2 : ABD = 50)
  (ADB : Real)
  (h3 : ADB = 43) :
  measure_angle_z ABC ABD ADB = 87 :=
by
  unfold measure_angle_z
  sorry

end geometry_problem_z_eq_87_deg_l749_749343


namespace hyperbola_eccentricity_ratio_hyperbola_condition_l749_749272

-- Part (a)
theorem hyperbola_eccentricity_ratio
  (a b c : ℝ) (h1 : c^2 = a^2 + b^2)
  (x0 y0 : ℝ) 
  (P : ℝ × ℝ) (h2 : P = (x0, y0))
  (F : ℝ × ℝ) (h3 : F = (c, 0))
  (D : ℝ) (h4 : D = a^2 / c)
  (d_PF : ℝ) (h5 : d_PF = ( (x0 - c)^2 + y0^2 )^(1/2))
  (d_PD : ℝ) (h6 : d_PD = |x0 - a^2 / c|)
  (e : ℝ) (h7 : e = c / a) :
  d_PF / d_PD = e :=
sorry

-- Part (b)
theorem hyperbola_condition
  (F_l : ℝ × ℝ) (h1 : F_l = (0, k))
  (X_l : ℝ × ℝ) (h2 : X_l = (x, l))
  (d_XF : ℝ) (h3 : d_XF = (x^2 + y^2)^(1/2))
  (d_Xl : ℝ) (h4 : d_Xl = |x - k|)
  (e : ℝ) (h5 : e > 1)
  (h6 : d_XF / d_Xl = e) :
  ∃ a b : ℝ, (x / a)^2 - (y / b)^2 = 1 :=
sorry

end hyperbola_eccentricity_ratio_hyperbola_condition_l749_749272


namespace time_difference_between_car_and_minivan_arrival_l749_749286

variable (car_speed : ℝ := 40)
variable (minivan_speed : ℝ := 50)
variable (pass_time : ℝ := 1 / 6) -- in hours

theorem time_difference_between_car_and_minivan_arrival :
  (60 * (1 / 6 - (20 / 3 / 50))) = 2 := sorry

end time_difference_between_car_and_minivan_arrival_l749_749286


namespace smallest_prime_factor_in_C_l749_749546

def set_C := {54, 56, 59, 63, 65}

theorem smallest_prime_factor_in_C (x : ℕ) (hx : x ∈ set_C) : ∃ p, p.prime ∧ p ≤ 2 :=
by
  sorry

end smallest_prime_factor_in_C_l749_749546


namespace sum_expression_polar_form_l749_749338

theorem sum_expression_polar_form :
  ∃ r θ, (r = 30 * |Real.cos (7 * Real.pi / 26)| ∧ (θ = Real.pi / 2 ∨ θ = -Real.pi / 2)) ∧
  15 * Complex.exp (3 * Real.pi * Complex.I / 13) + 15 * Complex.exp (10 * Real.pi * Complex.I / 13) = r * Complex.exp (θ * Complex.I) :=
begin
  sorry
end

end sum_expression_polar_form_l749_749338


namespace first_is_20_percent_of_second_l749_749104

theorem first_is_20_percent_of_second (X : ℝ) : 
  let first_number := (6 / 100) * X
  let second_number := (30 / 100) * X
  first_number / second_number * 100 = 20 :=
by
  let first_number := (6 / 100) * X
  let second_number := (30 / 100) * X
  have h : first_number / second_number * 100 = (6 / 30) * 100 := by sorry
  have h_simplify : (6 / 30) * 100 = 20 := by sorry
  exact eq.trans h h_simplify

end first_is_20_percent_of_second_l749_749104


namespace monotonicity_f_when_a_eq_1_range_of_a_l749_749928

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_f_when_a_eq_1 :
  (∀ x > 0, deriv (λ x, f x 1) x > 0) ∧ (∀ x < 0, deriv (λ x, f x 1) x < 0) :=
sorry

-- Part 2: Range of a such that f(x) ≥ 1/2 * x^3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x a ≥ 1/2 * x^3 + 1) ↔ a ≥ (7 - Real.exp 2) / 4 :=
sorry

end monotonicity_f_when_a_eq_1_range_of_a_l749_749928


namespace Alice_has_3_more_dimes_than_quarters_l749_749324

-- Definitions of the conditions given in the problem
variable (n d : ℕ) -- number of 5-cent and 10-cent coins
def q : ℕ := 10
def total_coins : ℕ := 30
def total_value : ℕ := 435
def extra_dimes : ℕ := 6

-- Conditions translated to Lean
axiom total_coin_count : n + d + q = total_coins
axiom total_value_count : 5 * n + 10 * d + 25 * q = total_value
axiom dime_difference : d = n + extra_dimes

-- The theorem that needs to be proven: Alice has 3 more 10-cent coins than 25-cent coins.
theorem Alice_has_3_more_dimes_than_quarters :
  d - q = 3 :=
sorry

end Alice_has_3_more_dimes_than_quarters_l749_749324


namespace find_lambda_l749_749002

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749002


namespace smallest_integer_with_18_divisors_l749_749638

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749638


namespace greatest_four_digit_divisible_l749_749350

theorem greatest_four_digit_divisible :
  ∃ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧
          (∃ n : ℕ, n = reverse m ∧ m % 55 = 0 ∧ n % 55 = 0) ∧
          m % 11 = 0 ∧
          ∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧
                   (∃ l : ℕ, l = reverse k ∧ k % 55 = 0 ∧ l % 55 = 0) ∧
                   k % 11 = 0 →
                   m ≥ k :=
  let rev (x : ℕ) : ℕ := nat.digits 10 x |>.reverse |>.foldl (λ a b => a * 10 + b) 0 in
    ∃ m, 1000 ≤ m ∧ m < 10000 ∧
         m % 55 = 0 ∧
         rev m % 55 = 0 ∧
         m % 11 = 0 ∧
         ∀ k, 1000 ≤ k ∧ k < 10000 ∧
              k % 55 = 0 ∧
              rev k % 55 = 0 ∧
              k % 11 = 0 →
              m ≥ k

end greatest_four_digit_divisible_l749_749350


namespace probability_multiple_2_or_3_l749_749233

theorem probability_multiple_2_or_3 : 
  let cards := Finset.range 31 \{0} in
  let multiples_of_2 := cards.filter (λ n, n % 2 = 0) in
  let multiples_of_3 := cards.filter (λ n, n % 3 = 0) in
  let multiples_of_6 := cards.filter (λ n, n % 6 = 0) in
  (multiples_of_2.card + multiples_of_3.card - multiples_of_6.card) / cards.card = 2 / 3 := 
by
  sorry

end probability_multiple_2_or_3_l749_749233


namespace problem_statement_l749_749592

-- Define the conditions as Lean predicates
def is_odd (n : ℕ) : Prop := n % 2 = 1
def between_400_and_600 (n : ℕ) : Prop := 400 < n ∧ n < 600
def divisible_by_55 (n : ℕ) : Prop := n % 55 = 0

-- Define a function to calculate the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Main theorem to prove
theorem problem_statement (N : ℕ)
  (h_odd : is_odd N)
  (h_range : between_400_and_600 N)
  (h_divisible : divisible_by_55 N) :
  sum_of_digits N = 18 :=
sorry

end problem_statement_l749_749592


namespace problem1_problem2i_problem2ii_l749_749892

-- Define the sequence \( \{a_n\} \) with given conditions
def sequence (λ μ : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = (λ * (a n)^2 + μ * (a n) + 4) / (a n + 2)  

-- Problem (1): Given λ=3 and μ=8, prove that \{a_n\} is a geometric sequence and find its general formula
theorem problem1 : ∀ (a : ℕ → ℝ), 
  sequence 3 8 a → 
  (∃ (r : ℝ), ∃ (b : ℕ → ℝ), (∀ n, a n = b n * r^(n-1)) ) :=
sorry

-- Problem (2i): Determine λ and μ for the arithmetic sequence {a_n}
theorem problem2i : ∀ (λ μ : ℝ) (a : ℕ → ℝ) (d : ℝ), 
  sequence λ μ a → 
  (∀ n, a n = a 1 + (n - 1) * d) → 
  λ = 1 ∧ μ = 4 :=
sorry

-- Problem (2ii): Check for the existence of a 4-term subsequence sum of {S_n} from arithmetic sequence {a_n}
theorem problem2ii : ∀ (a : ℕ → ℝ), 
  sequence 1 4 a → 
  (∀ n, a n = 2 * n - 1) → 
  (∃ (S : ℕ → ℝ), S 1 = 1 ∧ ∀ n, S (n + 1) = S n + a n ∧ 
  (∃ (x y z : ℕ), S 1 + S (2 * x) + S (2 * y) + S (2 * z) = 2017)) :=
sorry

end problem1_problem2i_problem2ii_l749_749892


namespace conjugate_expression_l749_749914

open Complex

theorem conjugate_expression (z : ℂ) (hz : z = 1 - I) :
  conj (2 / z - z^2) = 1 - 3 * I :=
by
  rw [hz]
  sorry

end conjugate_expression_l749_749914


namespace x_plus_q_eq_five_l749_749975

theorem x_plus_q_eq_five (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x < 5) : x + q = 5 :=
by
  sorry

end x_plus_q_eq_five_l749_749975


namespace conversion_rates_l749_749254

noncomputable def teamADailyConversionRate (a b : ℝ) := 1.2 * b
noncomputable def teamBDailyConversionRate (a b : ℝ) := b

theorem conversion_rates (total_area : ℝ) (b : ℝ) (h1 : total_area = 1500) (h2 : b = 50) 
    (h3 : teamADailyConversionRate 1500 b * b = 1.2) 
    (h4 : teamBDailyConversionRate 1500 b = b) 
    (h5 : (1500 / teamBDailyConversionRate 1500 b) - 5 = 1500 / teamADailyConversionRate 1500 b) :
  teamADailyConversionRate 1500 b = 60 ∧ teamBDailyConversionRate 1500 b = 50 := 
by
  sorry

end conversion_rates_l749_749254


namespace angle_subtraction_result_l749_749451

-- Definitions of angles in degrees and minutes
structure Angle where
  degrees : Int
  minutes : Int

-- 20 degrees 18 minutes
def angle1 : Angle := { degrees := 20, minutes := 18 }

-- 69 degrees 42 minutes
def expectedResult : Angle := { degrees := 69, minutes := 42 }

-- Lean statement to prove the equivalence
theorem angle_subtraction_result : Angle → Angle
| ⟨20, 18⟩ := ⟨69, 42⟩
| _ := sorry

example : angle_subtraction_result angle1 = expectedResult := by
  simp [angle_subtraction_result, angle1, expectedResult]
  sorry

end angle_subtraction_result_l749_749451


namespace smallest_positive_integer_with_18_divisors_l749_749697

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749697


namespace least_possible_value_in_S_l749_749505

theorem least_possible_value_in_S :
  ∃ S : Set ℕ, S ⊆ { x | 2 ≤ x ∧ x ≤ 13 } ∧ S.card = 5 ∧
  (∀ a b ∈ S, a < b → ¬ (b = a * a ∨ b = 2 * a)) ∧ S.min = 4 :=
by sorry

end least_possible_value_in_S_l749_749505


namespace train_cross_time_l749_749274

noncomputable theory

open Real

def length_of_train : ℝ := 100 -- Length of train in meters
def speed_of_train_kmh : ℝ := 144 -- Speed of train in km/hr

def speed_of_train_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600) -- Convert km/hr to m/s

def time_to_cross_pole (length : ℝ) (speed_mps : ℝ) : ℝ :=
  length / speed_mps -- Time = Distance / Speed

theorem train_cross_time : time_to_cross_pole length_of_train (speed_of_train_mps speed_of_train_kmh) = 2.5 :=
by 
  sorry

end train_cross_time_l749_749274


namespace smallest_integer_with_18_divisors_l749_749673

theorem smallest_integer_with_18_divisors : 
  ∃ n : ℕ, (n > 0 ∧ (nat.divisors_count n = 18) ∧ (∀ m : ℕ, m > 0 ∧ (nat.divisors_count m = 18) → n ≤ m)) := 
sorry

end smallest_integer_with_18_divisors_l749_749673


namespace tan_alpha_second_quadrant_l749_749152

-- Declare the variables involved in the problem
variables {α : ℝ} {x : ℝ}

-- Define the conditions
def in_second_quadrant (α : ℝ) : Prop := π/2 < α ∧ α < π
def point_on_terminal_side (α : ℝ) (x : ℝ) : Prop := cos α = (1 / 3) * x

-- State the theorem
theorem tan_alpha_second_quadrant (h1: in_second_quadrant α) (h2: point_on_terminal_side α x) : 
  tan α = - (sqrt 2) / 4 := by
  sorry

end tan_alpha_second_quadrant_l749_749152


namespace find_w_l749_749861

theorem find_w (w : ℤ) : 3^8 * 3^w = 81 → w = -4 :=
by
  intro h
  have h1 : 3^(8 + w) = 81 := by
    rw [←pow_add, add_comm, h]
  have h2 : 81 = 3^4 := by rfl -- since 81 = 3^4
  have h3 : 3^(8 + w) = 3^4 := by
    rw [h2] at h1
    exact h1
  have h4 : 8 + w = 4 := by
    apply (pow_inj (nat.succ_pos' 2)).mpr h3
  linarith
  sorry

end find_w_l749_749861


namespace peter_remaining_money_l749_749184

theorem peter_remaining_money :
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 426 :=
by
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  show remaining_money = 426 from sorry

end peter_remaining_money_l749_749184


namespace smallest_integer_with_18_divisors_l749_749706

def num_divisors (n : ℕ) : ℕ := 
  (factors n).freq.map (λ x => x.2 + 1).prod

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, num_divisors n = 18 ∧ ∀ m : ℕ, num_divisors m = 18 → n ≤ m := 
by
  let n := 180
  use n
  constructor
  ...
  sorry

end smallest_integer_with_18_divisors_l749_749706


namespace sequence_a2010_l749_749137

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 2 = 3 ∧ ∀ n ≥ 2, a (n + 1) = (a n * a (n - 1)) % 10

theorem sequence_a2010 (a : ℕ → ℕ) (h : sequence a) : a 2010 = 4 :=
  sorry

end sequence_a2010_l749_749137


namespace solve_for_y_l749_749968

theorem solve_for_y (y : ℝ) : 8^log 8 15 = 6 * y + 7 → y = 4 / 3 :=
by
  intro h
  sorry

end solve_for_y_l749_749968


namespace quadratic_intersect_condition_l749_749877

theorem quadratic_intersect_condition (k : ℝ) :
  (k > -1/4) ∧ (k ≠ 2) ↔ ((2*k - 1)^2 - 4*k*(k - 2) > 0) ∧ (k - 2 ≠ 0) :=
begin
  sorry
end

end quadratic_intersect_condition_l749_749877


namespace find_lambda_l749_749004

open EuclideanSpace

section

variable (λ : ℝ)
variable (a b : ℝ × ℝ)

def is_orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (ha : a = (1, 3))
  (hb : b = (3, 4))
  (h_orth : is_orthogonal (a.1 - λ * b.1, a.2 - λ * b.2) b) :
  λ = 3 / 5 := by
    sorry

end

end find_lambda_l749_749004


namespace number_of_valid_pairs_l749_749873

theorem number_of_valid_pairs : 
  let valid_int_range := {x | -2015 ≤ x ∧ x ≤ 2015 ∧ x ≠ 0}
  in (∃ valid_pairs (c d : ℤ) : valid_int_range.contains c ∧ valid_int_range.contains d, 
      (∃ x : ℤ, c * x = d) ∧ (∃ x : ℤ, d * x = c) ∧ valid_pairs.card = 8060) :=
sorry

end number_of_valid_pairs_l749_749873


namespace bird_count_l749_749303

theorem bird_count (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : parakeets_per_cage = 7) 
  (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) : 
  total_birds = 72 := 
  by
  sorry

end bird_count_l749_749303


namespace range_of_a_l749_749922

def f (x : ℝ) : ℝ := x^2 - 6 * x + 8

theorem range_of_a (a : ℝ) (h : 1 < a ∧ a ≤ 3) :
  (∀ x ∈ set.Icc 1 a, f x ≥ f a) :=
sorry

end range_of_a_l749_749922


namespace equilateral_triangle_altitude_l749_749462

theorem equilateral_triangle_altitude (ABC : Triangle) (h_eq : ABC.isEquilateral) (h_perimeter : ABC.perimeter = 24) : 
  ABC.altitude = 4 * Real.sqrt 3 := 
sorry

end equilateral_triangle_altitude_l749_749462


namespace calculate_calories_in_250g_l749_749602

def calories_in_250g_of_lemonade 
  (lemon_juice_calories_per_100g : ℕ)
  (lemon_juice_grams : ℕ)
  (honey_calories_per_100g : ℕ)
  (honey_grams : ℕ)
  (water_grams : ℕ)
  (total_grams : ℕ) 
  (total_calories : ℕ)
  : ℕ :=
  lemon_juice_calories_per_100g * lemon_juice_grams / 100 +
  honey_calories_per_100g * honey_grams / 100

theorem calculate_calories_in_250g:
  let lemon_juice_calories_per_100g := 30 in
  let lemon_juice_grams := 150 in
  let honey_calories_per_100g := 304 in
  let honey_grams := 200 in
  let water_grams := 500 in
  let total_grams := 850 in
  let total_calories := calories_in_250g_of_lemonade lemon_juice_calories_per_100g lemon_juice_grams honey_calories_per_100g honey_grams water_grams total_grams in
  total_grams = lemon_juice_grams + honey_grams + water_grams →
  total_calories = 653 →
  (total_calories * 250 / total_grams : ℕ) = 192 := by
  intros _ _ 
  sorry

end calculate_calories_in_250g_l749_749602


namespace arcsin_one_half_eq_pi_six_l749_749810

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l749_749810


namespace interval_departures_is_4_min_l749_749718

-- Define the entities and conditions from the mathematical problem
structure Speed :=
  (x : ℝ)  -- speed of the bus in meters per minute
  (y : ℝ)  -- walking speed of Xiao Wang in meters per minute

noncomputable def interval_between_departures (s : Speed) (f : ℝ) : ℝ :=
  sorry

theorem interval_departures_is_4_min (s : Speed) (s_eq : 6 * s.x - 6 * s.y = 3 * s.x + 3 * s.y) :
  interval_between_departures s 4 :=
begin
  sorry
end

end interval_departures_is_4_min_l749_749718


namespace find_lambda_l749_749082

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749082


namespace Calen_pencils_proof_l749_749340
/-- Import necessary library --/

/-- Noncomputable definitions for the conditions given in the problem --/

def Candy_pencils : ℕ := 9
def Caleb_pencils : ℕ := 2 * Candy_pencils - 3
def Calen_original : ℕ := Caleb_pencils + 5
def Calen_final : ℕ := Calen_original - 10

/-- The theorem to prove that Calen's final number of pencils is 10 --/
theorem Calen_pencils_proof : Calen_final = 10 :=
  by
  -- this is where the proof steps would go
  sorry

end Calen_pencils_proof_l749_749340


namespace volume_removed_percentage_l749_749314
  
-- Define the dimensions of the box and the side of the cube removed from each corner
def box_length : ℝ := 18
def box_width : ℝ := 12
def box_height : ℝ := 10
def cube_side : ℝ := 4

-- Define the volume calculation functions
def volume_box (l w h : ℝ) : ℝ := l * w * h
def volume_cube (s : ℝ) : ℝ := s ^ 3

-- Define the number of cubes removed from the box
def num_cubes_removed : ℝ := 8

-- Define the formula to calculate the percentage of volume removed
def percentage_volume_removed (v_box v_cubes : ℝ) : ℝ := (v_cubes / v_box) * 100

-- Translate the proof problem
theorem volume_removed_percentage :
  percentage_volume_removed (volume_box box_length box_width box_height)
                            (num_cubes_removed * volume_cube cube_side) = 23.7 := by
  sorry

end volume_removed_percentage_l749_749314


namespace DeMoivreTheorem_solution_l749_749785

noncomputable def cos_210 := Real.cos (210 * Real.pi / 180)
noncomputable def sin_210 := Real.sin (210 * Real.pi / 180)
noncomputable def cos_120 := Real.cos (120 * Real.pi / 180)
noncomputable def sin_120 := Real.sin (120 * Real.pi / 180)

theorem DeMoivreTheorem (n : ℕ) (θ : ℝ) : 
    (Real.cos θ + complex.i * Real.sin θ)^n = Real.cos (n * θ) + complex.i * Real.sin (n * θ) := sorry

theorem solution : (cos_210 + complex.i * sin_210)^60 = -1/2 + complex.i * (Real.sqrt 3 / 2) :=
by
  apply DeMoivreTheorem
  sorry

end DeMoivreTheorem_solution_l749_749785


namespace lambda_value_l749_749028

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749028


namespace arcsin_one_half_l749_749799

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749799


namespace true_statement_l749_749173

-- Definitions/Conditions
variables {l m n : Line}
variables {α β γ : Plane}
variable {x y z : Point}

-- Assume the conditions given in the problem
axiom intersect_alpha_beta : α ∩ β = l
axiom intersect_beta_gamma : β ∩ γ = m
axiom intersect_gamma_alpha : γ ∩ α = n
axiom line_parallel_gamma : l ∥ γ

-- Statement to prove
theorem true_statement :
  m ∥ n :=
sorry

end true_statement_l749_749173


namespace tithe_percentage_l749_749323

-- Definitions based on conditions
def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def water_bill : ℝ := 55
def remaining_income : ℝ := 345

-- The statement to prove 
theorem tithe_percentage : 
  let tax_deduction := tax_rate * weekly_income in
  let income_after_tax := weekly_income - tax_deduction in
  let income_after_water_bill := income_after_tax - water_bill in
  let tithe := income_after_water_bill - remaining_income in
  (tithe / weekly_income) * 100 = 10 :=
by
  sorry

end tithe_percentage_l749_749323


namespace jenny_kenny_reunion_time_l749_749494

/-- Define initial conditions given in the problem --/
def jenny_initial_pos : ℝ × ℝ := (-60, 100)
def kenny_initial_pos : ℝ × ℝ := (-60, -100)
def building_radius : ℝ := 60
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def distance_apa : ℝ := 200
def initial_distance : ℝ := 200

theorem jenny_kenny_reunion_time : ∃ t : ℚ, 
  (t = (10 * (Real.sqrt 35)) / 7) ∧ 
  (17 = (10 + 7)) :=
by
  -- conditions to be used
  let jenny_pos (t : ℝ) := (-60 + 2 * t, 100)
  let kenny_pos (t : ℝ) := (-60 + 4 * t, -100)
  let circle_eq (x y : ℝ) := (x^2 + y^2 = building_radius^2)
  
  sorry

end jenny_kenny_reunion_time_l749_749494


namespace find_lambda_l749_749045

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749045


namespace regular_2003_gon_l749_749149

-- Definitions for the conditions
variables {n : ℕ} (P : set ℤ)
variable [fintype P]

-- Assume necessary conditions
def convex (P : set ℤ) : Prop := sorry  -- example placeholder for convexity definition
def is_polygon (P : set ℤ) (n : ℕ) : Prop := sorry  -- example placeholder for n-gon definition

def angle_division_condition (P : set ℤ) : Prop :=
∀ (A ∈ P), ∃ (angles : ℕ → ℝ), (∀ i, angles i = _) -- placeholder angle division


-- The main statement to complete the proof
theorem regular_2003_gon (h1 : is_polygon P 2003) (h2 : convex P) (h3 : angle_division_condition P) :
  ∃ (R : ℝ), ∀ (vertices : ℕ → ℤ) (i j k :ℕ), (vertices i) ≠ (vertices j) ≠ (vertices k) → P 
:= sorry

end regular_2003_gon_l749_749149


namespace monotonicity_f_when_a_eq_1_range_of_a_l749_749929

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_f_when_a_eq_1 :
  (∀ x > 0, deriv (λ x, f x 1) x > 0) ∧ (∀ x < 0, deriv (λ x, f x 1) x < 0) :=
sorry

-- Part 2: Range of a such that f(x) ≥ 1/2 * x^3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x a ≥ 1/2 * x^3 + 1) ↔ a ≥ (7 - Real.exp 2) / 4 :=
sorry

end monotonicity_f_when_a_eq_1_range_of_a_l749_749929


namespace factor_expression_l749_749860

theorem factor_expression (x : ℝ) :
  4 * x * (x - 5) + 7 * (x - 5) + 12 * (x - 5) = (4 * x + 19) * (x - 5) :=
by
  sorry

end factor_expression_l749_749860


namespace line_integral_path_independence_l749_749547

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (x * y^2 * z, (1 + z^2) * y * z, (1 / 2) * x^2 * y^2)

-- Define the problem statement
theorem line_integral_path_independence :
  (∀ (x y z : ℝ), continuous (λ t : ℝ, vector_field x y z)) ∧
  (∀ (x y z : ℝ), differentiable ℝ (λ t : ℝ, vector_field x y z)) ∧
  (domain_is_simply_connected : ∀ (x y z : ℝ), ∃ u, vector_field x y z = vector_field u (x + y) (x + z)) ∧
  (∀ (x y z : ℝ), ∇ × vector_field x y z = (0, 0, 0)) →
  ∀ L : ℝ → ℝ^3, path_independent (vector_field) L :=
sorry

end line_integral_path_independence_l749_749547


namespace correct_conclusions_l749_749901

variable (a b : ℚ)

-- Formalize the conclusions
def conclusion_2 : Prop :=
  ∀ a : ℚ, a^2 = (-a)^2

def conclusion_4 : Prop :=
  ∀ a b : ℚ, a * b < 0 → |a + b| = | |a| - |b| |

-- Prove that conclusions ② and ④ are correct
theorem correct_conclusions (a b : ℚ) : conclusion_2 a ∧ conclusion_4 a b := by
  sorry

end correct_conclusions_l749_749901


namespace lambda_solution_l749_749066

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b : ℝ × ℝ := (3, 4)
noncomputable def λ_val := (3 : ℝ) / (5 : ℝ)

theorem lambda_solution (λ : ℝ) 
  (h₁ : vector_a = (1, 3)) 
  (h₂ : vector_b = (3, 4)) 
  (h₃ : let v := (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) 
         in v.1 * vector_b.1 + v.2 * vector_b.2 = 0) : 
  λ = λ_val :=
sorry

end lambda_solution_l749_749066


namespace ratio_PR_QS_l749_749193

noncomputable def PR_QS_ratio (PQ QR PS : ℝ) : ℝ :=
  (PQ + QR) / (PS - (PQ + QR))

theorem ratio_PR_QS
  (P Q R S : Point)
  (PQ QR PS : ℝ)
  (hPQ : PQ = dist P Q)
  (hQR : QR = dist Q R)
  (hPS : PS = dist P S)
  : PR_QS_ratio PQ QR PS = 10 / 7 :=
by
  -- Point P, Q, R, S lie on a line in that order and distances are as given.
  -- So, we use the given conditions to prove the desired ratio.
  sorry

end ratio_PR_QS_l749_749193


namespace arcsin_half_eq_pi_six_l749_749804

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l749_749804


namespace sum_odd_prob_l749_749232

theorem sum_odd_prob (grid : Fin 4 × Fin 4 -> Fin 16)
  (injective_grid : Function.Injective grid):
  ∃ (config : Fin 4 × Fin 4 -> Fin 16), 
       (∀ i : Fin 4, 
           (grid i |>.fst).sum % 2 = 1 ∧ 
           (grid i |>.snd).sum % 2 = 1)
        sorry

end sum_odd_prob_l749_749232


namespace smallest_integer_with_18_divisors_l749_749684

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749684


namespace minimum_keys_needed_l749_749999

def cabinets : ℕ := 8
def boxes_per_cabinet : ℕ := 4
def phones_per_box : ℕ := 10
def total_phones_needed : ℕ := 52

theorem minimum_keys_needed : 
  ∀ (cabinets boxes_per_cabinet phones_per_box total_phones_needed: ℕ), 
  cabinets = 8 →
  boxes_per_cabinet = 4 →
  phones_per_box = 10 →
  total_phones_needed = 52 →
  exists (keys_needed : ℕ), keys_needed = 9 :=
by
  intros _ _ _ _ hc hb hp ht
  have h1 : nat.ceil (52 / 10) = 6 := sorry -- detail of calculation
  have h2 : nat.ceil (6 / 4) = 2 := sorry -- detail of calculation
  use 9
  sorry

end minimum_keys_needed_l749_749999


namespace F_one_eq_one_l749_749570

noncomputable def F : ℝ → ℝ := sorry

axiom F_continuous : Continuous F

axiom exists_n_for_every_x (x : ℝ) : ∃ (n : ℕ), (nat.iterate F n x) = 1

theorem F_one_eq_one : F 1 = 1 :=
sorry

end F_one_eq_one_l749_749570


namespace probability_of_selection_l749_749394

-- Problem setup
def number_of_students : ℕ := 54
def number_of_students_eliminated : ℕ := 4
def number_of_remaining_students : ℕ := number_of_students - number_of_students_eliminated
def number_of_students_selected : ℕ := 5

-- Statement to be proved
theorem probability_of_selection :
  (number_of_students_selected : ℚ) / (number_of_students : ℚ) = 5 / 54 :=
sorry

end probability_of_selection_l749_749394


namespace pie_selling_days_l749_749309

theorem pie_selling_days (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end pie_selling_days_l749_749309


namespace range_of_function_l749_749219

def is_range (f : ℝ → ℝ) (dom : set ℝ) (r : set ℝ) : Prop :=
  ∀ y, y ∈ r ↔ ∃ x, x ∈ dom ∧ f x = y

noncomputable def example_function : ℝ → ℝ := λ x, 2 / (x - 1)

theorem range_of_function : 
  is_range example_function ((set.Ioo (-∞) 1) ∪ (set.Ico 2 5)) ((set.Ioo (-∞) 0) ∪ (set.Ioo (1 / 2) 2]) :=
sorry

end range_of_function_l749_749219


namespace escalator_steps_l749_749188

theorem escalator_steps
  (steps_ascending : ℤ)
  (steps_descending : ℤ)
  (ascend_units_time : ℤ)
  (descend_units_time : ℤ)
  (speed_ratio : ℤ)
  (equation : ((steps_ascending : ℚ) / (1 + (ascend_units_time : ℚ))) = ((steps_descending : ℚ) / ((descend_units_time : ℚ) * speed_ratio)) )
  (solution_x : (125 * 0.6 = 75)) : 
  (steps_ascending * (1 + 0.6 : ℚ) = 120) :=
by
  sorry

end escalator_steps_l749_749188


namespace smallest_next_divisor_l749_749496

def isOddFourDigitNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1000 ≤ n ∧ n < 10000

noncomputable def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => d > 0 ∧ n % d = 0)

theorem smallest_next_divisor (m : ℕ) (h₁ : isOddFourDigitNumber m) (h₂ : 437 ∈ divisors m) :
  ∃ k, k > 437 ∧ k ∈ divisors m ∧ k % 2 = 1 ∧ ∀ n, n > 437 ∧ n < k → n ∉ divisors m := by
  sorry

end smallest_next_divisor_l749_749496


namespace systematic_sampling_correct_l749_749202

open Nat

def systematic_sampling (n m k : ℕ) (seq : List ℕ) : Prop :=
  seq.length = m ∧
  (∀ i, i < m - 1 → seq.nth (i + 1) = some (seq.nthLe i (Nat.lt_of_lt_of_le i.lt_succ_self (m-1).le_self) + k))

-- Given problem
theorem systematic_sampling_correct :
  let n := 55
  let m := 5
  let k := n / m
  let seq := [5, 16, 27, 38, 49]
  systematic_sampling n m k seq :=
by
  sorry

end systematic_sampling_correct_l749_749202


namespace airline_odd_landings_l749_749229

/-- Given 1983 localities with direct service between any two of them and 10 international airlines
providing round-trip flights, prove that at least one of these airlines has a round trip with an odd
number of landings. -/
theorem airline_odd_landings (P : Fin 1983 → Type) (A : Fin 10 → Set (Σ i j, P i × P j))
  (h1 : ∀ (i j : Fin 1983), ∃ (a : Fin 10), ∃ (p : P i × P j), (⟨i, j, p⟩ ∈ A a))
  (h2 : ∀ (a : Fin 10) (i : Fin 1983) (j : Fin 1983) (p : P i × P j), (⟨i, j, p⟩ ∈ A a) ↔ (⟨j, i, p.swap⟩ ∈ A a)) :
  ∃ (a : Fin 10), ¬ bipartite (A a) := 
sorry

end airline_odd_landings_l749_749229


namespace remaining_money_l749_749186

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end remaining_money_l749_749186


namespace largest_common_divisor_of_408_and_330_is_6_l749_749605

theorem largest_common_divisor_of_408_and_330_is_6 :
  let n1 := 408;
  let n2 := 330;
  is_divisor (λ x y : ℕ, x ∣ y) n1 ∧ is_divisor (λ x y : ℕ, x ∣ y) n2 → gcd n1 n2 = 6 :=
by
  let n1 := 408;
  let n2 := 330;
  sorry

end largest_common_divisor_of_408_and_330_is_6_l749_749605


namespace arcsin_of_half_l749_749822

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l749_749822


namespace simplifyTrigExpr_correct_l749_749206

noncomputable def simplifyTrigExpr (α : ℝ) : Prop :=
  (cos (π + α) * cos ((11 * π / 2) - α)) / (cos (π - α) * sin ((9 * π / 2) + α)) = -tan α

theorem simplifyTrigExpr_correct (α : ℝ) : simplifyTrigExpr α :=
  sorry

end simplifyTrigExpr_correct_l749_749206


namespace logical_equivalence_l749_749717

theorem logical_equivalence (P Q R : Prop) :
  ((¬ P ∧ ¬ Q) → ¬ R) ↔ (R → (P ∨ Q)) :=
by sorry

end logical_equivalence_l749_749717


namespace enclosure_blocks_count_l749_749839

theorem enclosure_blocks_count (length width height : ℕ) (wall_thickness floor_thickness : ℕ)
  (h_length : length = 15) 
  (h_width : width = 8) 
  (h_height : height = 7)
  (h_wall : wall_thickness = 1)
  (h_floor : floor_thickness = 1):
  let original_volume := length * width * height,
      interior_length := length - 2 * wall_thickness,
      interior_width := width - 2 * wall_thickness,
      interior_height := height - floor_thickness,
      interior_volume := interior_length * interior_width * interior_height,
      blocks_count := original_volume - interior_volume
  in blocks_count = 372 := 
by 
  sorry

end enclosure_blocks_count_l749_749839


namespace perfectNumberSumOfPowers_l749_749306

-- Define what it means to be a perfect number
def isPerfect (n : Nat) : Prop :=
  ∑ i in (Finset.range n).filter (fun x => x ∣ n && x ≠ n), i = n

-- Prove that 8128 can be written as the sum of consecutive powers of 2
theorem perfectNumberSumOfPowers (n : Nat) (h_prime : Nat.Prime (2^n -1)) : 
  isPerfect (2^(n-1) * (2^n - 1)) → 
  2^(n-1) * (2^n - 1) = ∑ i in Finset.range (n+5), 2^(i-1) :=
by
  sorry

end perfectNumberSumOfPowers_l749_749306


namespace lambda_value_l749_749021

theorem lambda_value (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, 4)) 
  (h_perp : ((1 - 3 * λ, 3 - 4 * λ) · (3, 4)) = 0) : 
  λ = 3 / 5 :=
by 
  sorry

end lambda_value_l749_749021


namespace peter_remaining_money_l749_749180

theorem peter_remaining_money (initial_money : ℕ) 
                             (potato_cost_per_kilo : ℕ) (potato_kilos : ℕ)
                             (tomato_cost_per_kilo : ℕ) (tomato_kilos : ℕ)
                             (cucumber_cost_per_kilo : ℕ) (cucumber_kilos : ℕ)
                             (banana_cost_per_kilo : ℕ) (banana_kilos : ℕ) :
  initial_money = 500 →
  potato_cost_per_kilo = 2 → potato_kilos = 6 →
  tomato_cost_per_kilo = 3 → tomato_kilos = 9 →
  cucumber_cost_per_kilo = 4 → cucumber_kilos = 5 →
  banana_cost_per_kilo = 5 → banana_kilos = 3 →
  initial_money - (potato_cost_per_kilo * potato_kilos + 
                   tomato_cost_per_kilo * tomato_kilos +
                   cucumber_cost_per_kilo * cucumber_kilos +
                   banana_cost_per_kilo * banana_kilos) = 426 := by
  sorry

end peter_remaining_money_l749_749180


namespace hyperbola_equation_shared_foci_asymptotes_l749_749221

noncomputable def ellipse_foci (a b : ℝ) (h : a > b) : ℝ :=
Real.sqrt (a^2 - b^2)

theorem hyperbola_equation_shared_foci_asymptotes :
  ∀ (x y : ℝ), 
  (ellipse_foci 7 2√6 (by norm_num) = 5) →
  (∀ x y : ℝ, (x^2 / 36 - y^2 / 64 = 1) → asymptotes x y = 0 ) →
  (∃ (k : ℝ), k < 0 ∧ y^2 / (-64 * k) - x^2 / (-36 * k) = 1)
  (∃ k xy : ℝ, equation = y^2 / 16 - x^2 / 9 = 1 : Prop :=
begin
  intro x,
  intro y,
  assume h,
  exact sorry,
end

end hyperbola_equation_shared_foci_asymptotes_l749_749221


namespace sequence_closed_form_l749_749463

variable {nat : Type} [nat.Semiring nat]

def a : ℕ → ℕ
| 1         := 1
| (n+1) := a n + 1 + 2^n

theorem sequence_closed_form (n : ℕ) : a n = 2^n + n - 2 :=
by
  sorry

end sequence_closed_form_l749_749463


namespace brain_info_scientific_notation_l749_749297

theorem brain_info_scientific_notation :
  ∃ (n : ℝ), n = 86 ∧ (86 * 10^6 = 8.6 * 10^7) :=
by
  use 86
  split
  exact rfl
  norm_num
  sorry

end brain_info_scientific_notation_l749_749297


namespace inequality_proof_max_value_expression_l749_749956

-- Define the vectors u and v and establish the given conditions
variables (a b c d : ℝ) (u v : ℝ × ℝ)
variables (x : ℝ)

-- Assumptions: a, b, c, d are all positive real numbers
-- Condition: Given |u · v| = |u||v|
axiom vector_condition : abs (a * c + b * d) = sqrt (a^2 + b^2) * sqrt (c^2 + d^2)

-- To prove: (a^2 + b^2)(c^2 + d^2) ≥ (ac + bd)^2
theorem inequality_proof : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := sorry

-- To prove the maximum value of the expression 
-- sqrt(4*x + 13) + 2*sqrt(3 - x) is 5 * sqrt(2) when x = -1/8
noncomputable def expression_value := real.sqrt(4 * x + 13) + 2 * real.sqrt(3 - x)
theorem max_value_expression : 
  ∃ x₀, (x₀ = -1 / 8) ∧ (expression_value x₀ = 5 * sqrt 2) := sorry

end inequality_proof_max_value_expression_l749_749956


namespace area_of_transformed_region_l749_749524

noncomputable def area_transformed_region (T : Set (ℝ × ℝ)) (A : Matrix (Fin 2) (Fin 2) ℝ) :=
  (Matrix.det A).abs * 8

theorem area_of_transformed_region :
  let A := Matrix.of 2 2 (λ i j, ![(3 : ℝ), 2, 4, -1].get (i * 2 + j)) in
  area_transformed_region {p : ℝ × ℝ | True} A = 88 :=
by
  let A : Matrix (Fin 2) (Fin 2) ℝ := Matrix.of 2 2 (λ i j, ![(3 : ℝ), 2, 4, -1].get (i * 2 + j))
  have det_A : Matrix.det A = -11 :=
    by
      -- Computation of the determinant
      unfold Matrix.det
      sorry
  calc
    (Matrix.det A).abs * 8 = (11 : ℝ) * 8 := by rw [det_A, Real.abs_neg, abs_of_pos zero_lt_eleven] 
                        ... = 88 := by norm_num

end area_of_transformed_region_l749_749524


namespace hyperbola_eccentricity_is_2_l749_749432

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) : ℝ :=
c / a

theorem hyperbola_eccentricity_is_2 (a b c : ℝ)
  (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) :
  hyperbola_eccentricity a b c h H1 H2 = 2 :=
sorry

end hyperbola_eccentricity_is_2_l749_749432


namespace trig_identity_sum_l749_749846

theorem trig_identity_sum :
  ∃ (a b c d : ℤ), 
    (∀ x : ℝ, cos 2 * x + cos 4 * x + cos 8 * x + cos 10 * x = 
              a * cos b * x * cos c * x * cos d * x) 
    ∧ a + b + c + d = 14 := 
by
  use [4, 6, 3, 1]
  split
  { -- prove the trigonometric identity
    intro x
    sorry
  }
  { -- prove the sum of coefficients
    norm_num
  }

end trig_identity_sum_l749_749846


namespace place_mat_length_l749_749768

theorem place_mat_length 
    (radius : ℝ)
    (width : ℝ)
    (n : ℕ)
    (edge_on_table : ℝ) 
    (inner_touches_adjacent : ℝ)
    (h_radius : radius = 5)
    (h_width : width = 1)
    (h_n : n = 8)
    (h_inner_touches : inner_touches_adjacent = 1.951)
    (arc_length : ℝ := (2 * real.pi * radius) / n)
    (theta : ℝ := 360 / (2 * n) := 22.5)
    (sin_eval : ℝ := real.sin (theta / 2))
    (chord_formula : ℝ := 2 * radius * sin_eval)
  : edge_on_table = inner_touches_adjacent := sorry

end place_mat_length_l749_749768


namespace smallest_integer_with_18_divisors_l749_749648

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → (∀ d, d ∣ n → d ∣ m) → n = 288) ∧ (nat.divisors_count 288 = 18) := by
  sorry

end smallest_integer_with_18_divisors_l749_749648


namespace translation_cosine_graph_l749_749124

noncomputable def g (x : ℝ) : ℝ := cos (2 * (x - (π / 6)))

theorem translation_cosine_graph : g (π / 2) = -1 / 2 :=
by
  sorry

end translation_cosine_graph_l749_749124


namespace carrot_sticks_leftover_l749_749556

theorem carrot_sticks_leftover (total_carrots : ℕ) (people : ℕ) (h1 : total_carrots = 74) (h2 : people = 12) :
  total_carrots % people = 2 :=
by
  rw [h1, h2]
  exact Nat.mod_eq_of_lt (Nat.zero_le 2) (show 2 < 12 by norm_num)

end carrot_sticks_leftover_l749_749556


namespace wheel_moves_distance_in_one_hour_l749_749722

-- Definition of the given conditions
def rotations_per_minute : ℕ := 10
def distance_per_rotation : ℕ := 20
def minutes_per_hour : ℕ := 60

-- Theorem statement to prove the wheel moves 12000 cm in one hour
theorem wheel_moves_distance_in_one_hour : 
  rotations_per_minute * minutes_per_hour * distance_per_rotation = 12000 := 
by
  sorry

end wheel_moves_distance_in_one_hour_l749_749722


namespace max_profit_at_12_l749_749121

def total_cost (x : ℝ) : ℝ := 12 + 10 * x

def sales_revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 16 then -0.5 * x^2 + 22 * x
  else 224

def profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 16 then -0.5 * x^2 + 12 * x - 12
  else 212 - 10 * x

theorem max_profit_at_12 :
  profit 12 = 60 := 
sorry

end max_profit_at_12_l749_749121


namespace smallest_positive_integer_with_18_divisors_l749_749669

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l749_749669


namespace cannot_arrange_51_to_150_l749_749575

def has_integral_roots (a b : ℤ) : Prop :=
∃ x y : ℤ, x + y = a ∧ x * y = b

theorem cannot_arrange_51_to_150 :
  ¬(∃ M : matrix (fin 10) (fin 10) ℤ, 
      (∀ i j, 51 ≤ M i j ∧ M i j ≤ 150) ∧
      (∀ i j, (i < 9 → has_integral_roots (M i j) (M (i+1) j)) ∧
               (j < 9 → has_integral_roots (M i j) (M i (j+1))))): 
by
  -- Proof omitted
  sorry

end cannot_arrange_51_to_150_l749_749575


namespace find_lambda_l749_749083

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749083


namespace compare_abc_l749_749413

noncomputable def a : ℝ := - Real.logb 2 (1/5)
noncomputable def b : ℝ := Real.logb 8 27
noncomputable def c : ℝ := Real.exp (-3)

theorem compare_abc : a = Real.logb 2 5 ∧ 1 < b ∧ b < 2 ∧ c = Real.exp (-3) → a > b ∧ b > c :=
by
  sorry

end compare_abc_l749_749413


namespace problem_l749_749412

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}
def C := (Aᶜ) ∩ B

theorem problem : C = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end problem_l749_749412


namespace digits_of_sum_l749_749970

noncomputable def sum_of_numbers (A B : ℕ) :=
  19876 + (10^3 * A + 10^2 * B + 32) + (10^2 * 2 + 10 * B + 1)

theorem digits_of_sum (A B : ℕ) (hA : 1 ≤ A) (hA' : A ≤ 9) (hB : 1 ≤ B) (hB' : B ≤ 9) : 
  (sum_of_numbers A B).digits.length = 5 :=
sorry

end digits_of_sum_l749_749970


namespace find_lambda_l749_749046

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749046


namespace infinite_coprime_terms_l749_749160

theorem infinite_coprime_terms 
  (a b m : ℕ) 
  (coprime_ab : Nat.coprime a b) 
  (hm : m > 0) 
  : ∃ᶠ n in at_top, Nat.coprime (a + n * b) m :=
sorry

end infinite_coprime_terms_l749_749160


namespace farmers_acres_to_clean_l749_749755

-- Definitions of the main quantities
variables (A D : ℕ)

-- Conditions
axiom condition1 : A = 80 * D
axiom condition2 : 90 * (D - 1) + 30 = A

-- Theorem asserting the total number of acres to be cleaned
theorem farmers_acres_to_clean : A = 480 :=
by
  -- The proof would go here, but is omitted as per instructions
  sorry

end farmers_acres_to_clean_l749_749755


namespace largest_circle_area_l749_749762

noncomputable def rectangle_to_circle_area (w : ℝ) (h : ℝ) (P : ℝ) (A_rect : ℝ) (C : ℝ) : ℝ :=
  let r := C / (2 * Real.pi)
  in Real.pi * r^2

theorem largest_circle_area (w h : ℝ) (H : 2 * w = h) (A_rect : w * h = 200) (P : 2 * (w + h)) : 
  Int.round (rectangle_to_circle_area w h P (w * h) P) = 287 :=
by
  sorry

end largest_circle_area_l749_749762


namespace smallest_n_inequality_l749_749866

theorem smallest_n_inequality : ∃ (n : ℕ), (∀ (x y : ℝ), (x^2 + y^2)^2 ≤ n * (x^4 + y^4)) ∧
  (∀ m : ℕ, (∀ (x y : ℝ), (x^2 + y^2)^2 ≤ m * (x^4 + y^4)) → m ≥ 2) :=
begin
  use 2,
  split,
  { intros x y,
    calc (x^2 + y^2)^2 ≤ 2 * (x^4 + y^4) : by apply sorry },
  { intros m h,
    apply sorry }
end

end smallest_n_inequality_l749_749866


namespace rectangle_area_l749_749307

-- Definitions of the conditions
def side_of_square (area_sq : ℝ) : ℝ := real.sqrt area_sq
def radius_of_circle (side_sq : ℝ) : ℝ := side_sq
def length_of_rectangle (radius_circle : ℝ) : ℝ := radius_circle / 6
def breadth_of_rectangle : ℝ := 10

-- The tuple containing question, conditions, and correct answer
theorem rectangle_area (area_sq : ℝ) (h_area : area_sq = 1296) :
  let side_sq := side_of_square area_sq in
  let radius_circle := radius_of_circle side_sq in
  let length_rect := length_of_rectangle radius_circle in
  breadth_of_rectangle * length_rect = 60 :=
by
  intros
  sorry

end rectangle_area_l749_749307


namespace find_k_find_theta_l749_749441

variables {a b : ℝ} (k : ℝ)
variables (vec_a vec_b : ℝ → ℝ) -- Assuming they represent some vector space.
open_locale real_inner_product_space -- For dot product and other vector space operations.

-- Given conditions
def angle_between (vec_a vec_b : ℝ → ℝ) (θ : ℝ) : Prop :=
inner_product_space.angle vec_a vec_b = θ

def magnitude (vec : ℝ → ℝ) (value : ℝ) : Prop :=
sqrt (real_inner_product_space.inner vec vec) = value

def vector_m (vec_a vec_b : ℝ → ℝ) : ℝ → ℝ :=
3 • vec_a - 2 • vec_b

def vector_n (vec_a vec_b : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
2 • vec_a + k • vec_b

-- (I) Determine k such that vector_m is perpendicular to vector_n
theorem find_k (h_angle : angle_between vec_a vec_b (2 * π / 3))
  (h_mag_a : magnitude vec_a 2) (h_mag_b : magnitude vec_b 3)
  (h_perp : real_inner_product_space.inner (vector_m vec_a vec_b) (vector_n vec_a vec_b k) = 0) :
  k = 4 / 3 :=
sorry

-- (II) Determine the angle between vector_m and vector_n when k = -4 / 3
theorem find_theta (h_angle : angle_between vec_a vec_b (2 * π / 3))
  (h_mag_a : magnitude vec_a 2) (h_mag_b : magnitude vec_b 3)
  (h_perp : real_inner_product_space.inner (vector_m vec_a vec_b) (vector_n vec_a vec_b (-4 / 3)) = 0)
  (k_eq : k = -4 / 3) :
  let θ := real_inner_product_space.angle (vector_m vec_a vec_b) (vector_n vec_a vec_b k) in θ = 0 :=
sorry

end find_k_find_theta_l749_749441


namespace compute_expression_l749_749339

theorem compute_expression : 
  (cbrt (-8 : ℝ) + abs (2 - real.sqrt 5) + 4 * (real.sqrt 5 / 2)) = 3 * real.sqrt 5 - 4 :=
by
  sorry

end compute_expression_l749_749339


namespace cicely_100th_birthday_l749_749341

-- Definition of the conditions
def birth_year (birthday_year : ℕ) (birthday_age : ℕ) : ℕ :=
  birthday_year - birthday_age

def birthday (birth_year : ℕ) (age : ℕ) : ℕ :=
  birth_year + age

-- The problem restatement in Lean 4
theorem cicely_100th_birthday (birthday_year : ℕ) (birthday_age : ℕ) (expected_year : ℕ) :
  birthday_year = 1939 → birthday_age = 21 → expected_year = 2018 → birthday (birth_year birthday_year birthday_age) 100 = expected_year :=
by
  intros h1 h2 h3
  rw [birthday, birth_year]
  rw [h1, h2]
  sorry

end cicely_100th_birthday_l749_749341


namespace digit_seven_occurrences_in_sum_l749_749146

noncomputable def calculate_N (n : ℕ) : Nat := 
  (0:nat) -- A placeholder implementation for the sequence sum 
  sorry

theorem digit_seven_occurrences_in_sum : 
  let N := calculate_N 100 in 
  (count_digit 7 N) = 6 :=
sorry

end digit_seven_occurrences_in_sum_l749_749146


namespace find_length_PT_l749_749988

noncomputable def length_PT : ℝ :=
  let PQ := 2 * Real.sqrt 13 in
  6 * PQ / 11

theorem find_length_PT :
  length_PT = 12 * Real.sqrt 13 / 11 :=
by
  -- here would be the full proof based on the geometric arguments
  sorry

end find_length_PT_l749_749988


namespace inequality_5positives_l749_749194

variable {x1 x2 x3 x4 x5 : ℝ}

theorem inequality_5positives (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end inequality_5positives_l749_749194


namespace limit_tan_div_tan3x_l749_749361

noncomputable def limit_of_tan_div_tan3x : Real :=
  limit (n:=Real) (f: ℝ → ℝ) (x → y) := f x = y → 
  (∀ δ, (0 < δ) → (∃ ε, (0 < ε) ∧ (x - ε < x ∧ x < x + ε)) →
    ∀ x, (x - ε < x ∧ x < x + ε) → |f x - y| < δ)

theorem limit_tan_div_tan3x :
  limit_of_tan_div_tan3x (λ x, tan x / tan(3 * x)) (real.pi / 2) 3 :=
begin
  sorry
end

end limit_tan_div_tan3x_l749_749361


namespace range_of_a_l749_749434

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) : -4 < a ∧ a ≤ 0 := 
sorry

end range_of_a_l749_749434


namespace right_triangle_angle_bisector_l749_749474

theorem right_triangle_angle_bisector
  (m n : ℝ) (h : m > n)
  (other_leg hyp : ℝ)
  (triangle_condition : ∃(a b c : ℝ), a^2 + b^2 = c^2 ∧ a = n * hyp ∧ b = m * other_leg ∧ c = m * hyp) :
  other_leg = n * sqrt((m + n) / (m - n)) ∧ hyp = m * sqrt((m + n) / (m - n)) := 
by
sorry

end right_triangle_angle_bisector_l749_749474


namespace arcsin_of_half_l749_749821

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l749_749821


namespace combinations_for_ten_dollars_l749_749852

theorem combinations_for_ten_dollars :
  ∃ (x y z : ℕ), (x + 2 * y + 5 * z = 36 ∧ 5 * y + 4 * z = 0) :=
begin
  sorry
end

end combinations_for_ten_dollars_l749_749852


namespace purely_imaginary_m_value_fourth_quadrant_m_range_l749_749888

variable (m : ℝ)

-- Defining the complex number z
def z (m : ℝ) : Complex := Complex.mk (2*m^2 - 7*m + 6) (m^2 - m - 2)

-- Prove (1) if z is purely imaginary, then m = 3/2
theorem purely_imaginary_m_value {m : ℝ} (h1 : z m).re = 0 : m = 3/2 :=
by sorry

-- Prove (2) if z is in the fourth quadrant, then m is in (-1, 3/2)
theorem fourth_quadrant_m_range {m : ℝ} (h2 : (z m).re > 0) (h3 : (z m).im < 0) : -1 < m ∧ m < 3/2 :=
by sorry

end purely_imaginary_m_value_fourth_quadrant_m_range_l749_749888


namespace range_of_f_l749_749943

noncomputable def g (x : ℝ) : ℝ := x^2 - 2
noncomputable def f (x : ℝ) : ℝ :=
  if x < g x then g x + x + 4 else g x - x

theorem range_of_f : set.range f = (set.Icc (-9/4) 0 ∪ { x : ℝ | x > 2 }) :=
sorry

end range_of_f_l749_749943


namespace correct_statement_about_meiosis_and_fertilization_l749_749328

def statement_A : Prop := 
  ∃ oogonia spermatogonia zygotes : ℕ, 
    oogonia = 20 ∧ spermatogonia = 8 ∧ zygotes = 32 ∧ 
    (oogonia + spermatogonia = zygotes)

def statement_B : Prop := 
  ∀ zygote_dna mother_half father_half : ℕ,
    zygote_dna = mother_half + father_half ∧ 
    mother_half = father_half

def statement_C : Prop := 
  ∀ (meiosis stabilizes : Prop) (chromosome_count : ℕ),
    (meiosis → stabilizes) ∧ 
    (stabilizes → chromosome_count = (chromosome_count / 2 + chromosome_count / 2))

def statement_D : Prop := 
  ∀ (diversity : Prop) (gene_mutations chromosomal_variations : Prop),
    (diversity → ¬ (gene_mutations ∨ chromosomal_variations))

theorem correct_statement_about_meiosis_and_fertilization :
  ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D :=
by
  sorry

end correct_statement_about_meiosis_and_fertilization_l749_749328


namespace arcsin_of_half_l749_749818

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l749_749818


namespace percentage_of_sum_is_14_l749_749455

-- Define variables x, y as real numbers
variables (x y P : ℝ)

-- Define condition 1: y is 17.647058823529413% of x
def y_is_percentage_of_x : Prop := y = 0.17647058823529413 * x

-- Define condition 2: 20% of (x - y) is equal to P% of (x + y)
def percentage_equation : Prop := 0.20 * (x - y) = (P / 100) * (x + y)

-- Define the statement to be proved: P is 14
theorem percentage_of_sum_is_14 (h1 : y_is_percentage_of_x x y) (h2 : percentage_equation x y P) : 
  P = 14 :=
by
  sorry

end percentage_of_sum_is_14_l749_749455


namespace intersection_point_interval_l749_749946

theorem intersection_point_interval (x₀ : ℝ) (h : x₀^3 = 2^x₀ + 1) : 
  1 < x₀ ∧ x₀ < 2 :=
by
  sorry

end intersection_point_interval_l749_749946


namespace cone_height_l749_749744

theorem cone_height (V : ℝ) (h r : ℝ) (pi : ℝ) (vertex_angle : ℝ)
  (h_eq_r : h = r)
  (volume_eq : V = 1728 * pi)
  (volume_formula : V = 1 / 3 * pi * r^2 * h)
  (pi_def : pi = real.pi) :
  h = 12 :=
by
  sorry

end cone_height_l749_749744


namespace distance_between_parallel_lines_l749_749565

-- Definitions of the line equations based on the conditions
def line1_eq (x y : ℝ) : Prop := x + 3 * y - 4 = 0
def line2_eq (x y : ℝ) : Prop := 2 * x + 6 * y - 13 = 0

-- The hypothesis stating that the lines are given by the above equations
variable (x y : ℝ)

-- The main theorem that the distance between the parallel lines is as stated
theorem distance_between_parallel_lines :
  let A := 1
  let B := 3
  let C1 := -4
  let C2 := -(13 / 2)
  abs ((-13 / 2) - (-4)) / sqrt (1^2 + 3^2) = sqrt 10 / 4 := 
by {
  -- Insert the proof here
  sorry
}

end distance_between_parallel_lines_l749_749565


namespace smallest_integer_with_18_divisors_l749_749686

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, has_exactly_18_divisors n ∧ (∀ m : ℕ, has_exactly_18_divisors m → n ≤ m) ∧ n = 540 := sorry

end smallest_integer_with_18_divisors_l749_749686


namespace value_of_a_l749_749910

-- Define the lines with their equations
def line1 (a : ℝ) := ∀ (x y : ℝ), 2 * x + (a - 1) * y + a = 0
def line2 (a : ℝ) := ∀ (x y : ℝ), a * x + y + 2 = 0

-- The slope of line1: 2x + (a-1)y + a = 0 is -2 / (a-1)
def slope_line1 (a : ℝ) : ℝ := -2 / (a - 1)

-- The slope of line2: ax + y + 2 = 0 is -a
def slope_line2 (a : ℝ) : ℝ := -a

-- Proof problem: Given the lines are parallel, prove a = -1
theorem value_of_a (a : ℝ) : (slope_line1 a = slope_line2 a) → a = -1 :=
by
  sorry

end value_of_a_l749_749910


namespace arcsin_half_eq_pi_six_l749_749803

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l749_749803


namespace smallest_positive_integer_with_18_divisors_l749_749690

def has_exactly_divisors (n d : ℕ) : Prop :=
  (finset.filter (λ k, n % k = 0) (finset.range (n + 1))).card = d

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ has_exactly_divisors n 18 ∧ ∀ m : ℕ, 
  0 < m ∧ has_exactly_divisors m 18 → n ≤ m :=
begin
  use 90,
  split,
  { norm_num },
  split,
  { sorry },  -- Proof that 90 has exactly 18 positive divisors
  { intros m hm h_div,
    sorry }  -- Proof that for any m with exactly 18 divisors, m ≥ 90
end

end smallest_positive_integer_with_18_divisors_l749_749690


namespace value_of_expression_l749_749265

theorem value_of_expression :
  (81 ^ (Real.log 2023 / Real.log 3)) ^ (1 / 4) = 2023 :=
by
  sorry

end value_of_expression_l749_749265


namespace Romina_bought_20_boxes_l749_749237

theorem Romina_bought_20_boxes 
    (initial_price : ℕ)
    (price_reduction : ℕ)
    (total_paid : ℕ)
    (reduced_price := initial_price - price_reduction)
    (num_boxes := total_paid / reduced_price) :
    initial_price = 104 → 
    price_reduction = 24 → 
    total_paid = 1600 → 
    num_boxes = 20 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  compute_num_boxes


end Romina_bought_20_boxes_l749_749237


namespace find_sum_of_digits_l749_749585

theorem find_sum_of_digits (a c : ℕ) (h1 : 200 + 10 * a + 3 + 427 = 600 + 10 * c + 9) (h2 : (600 + 10 * c + 9) % 3 = 0) : a + c = 4 :=
sorry

end find_sum_of_digits_l749_749585


namespace car_has_traveled_2964_l749_749293

def actual_miles_traveled (odometer_reading : ℕ) : ℕ :=
  let skipped_digits := [4, 6]
  let count_skipped (n : ℕ) : ℕ := 
    (toString n).foldl (λ (acc : ℕ) (d : Char), 
      if d.toNat - '0'.toNat ∈ skipped_digits then acc + 1 else acc) 0
  odometer_reading - count_skipped odometer_reading

theorem car_has_traveled_2964 (odometer_reading : ℕ) (h : odometer_reading = 3509) : 
  actual_miles_traveled odometer_reading = 2964 :=
  by
  rw [h]
  sorry

end car_has_traveled_2964_l749_749293


namespace asymptote2_eq_l749_749176

noncomputable def hyperbola_center : ℝ × ℝ := (-3, -12)

noncomputable def asymptote1 (x : ℝ) : ℝ := 4 * x

theorem asymptote2_eq :
  ∀ x, asymptote2 x = -4 * x - 36 :=
sorry

end asymptote2_eq_l749_749176


namespace find_lambda_l749_749080

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749080


namespace probability_ace_king_queen_l749_749593

theorem probability_ace_king_queen :
  let prob_ace : ℚ := 4 / 52,
      prob_king : ℚ := 4 / 51,
      prob_queen : ℚ := 4 / 50
  in prob_ace * prob_king * (2 / 25) = 8 / 16575 :=
by sorry

end probability_ace_king_queen_l749_749593


namespace unbounded_set_bounded_series_impossible_l749_749520

noncomputable def sequence_a : ℕ+ → ℝ := sorry -- ℝ refers to real numbers, ℕ+ to positive natural numbers
def b (n : ℕ+) : ℝ := (sequence_a n) ^ 2
def angle_series : ℕ+ → ℝ := λ n, real.arctan $ (b n) / (∑ i in finset.range n, b i)

theorem unbounded_set_bounded_series_impossible (h1 : ∀ n : ℕ+, sequence_a n > 0) 
  (h_unbounded : tsum b = ∞) 
  (h_converge : tsum angle_series < ∞) : 
  false :=
begin
  sorry -- proof goes here
end

end unbounded_set_bounded_series_impossible_l749_749520


namespace project_completion_time_eq_ten_l749_749103

-- Define constants for the conditions
def project_time_A := 18  -- A can complete the project in 18 days
def project_time_B := 15  -- B can complete the project in 15 days
def break_time := 4       -- A takes a 4-day break halfway through

-- Define efficiencies
def efficiency_A := (1 / project_time_A)
def efficiency_B := (1 / project_time_B)
def combined_efficiency := efficiency_A + efficiency_B

-- Define total time to complete the project
def total_days : ℝ :=
  let work_done_by_B := efficiency_B * break_time in
  let remaining_work := 1 - work_done_by_B in
  let time_to_complete_remaining_work := remaining_work / combined_efficiency in
  time_to_complete_remaining_work + break_time

theorem project_completion_time_eq_ten : total_days = 10 := by
  sorry

end project_completion_time_eq_ten_l749_749103


namespace rationalization_correct_l749_749542

theorem rationalization_correct :
  ∀ A B C D E F : ℤ,
  A = -1 ∧ B = -3 ∧ C = 1 ∧ D = 2 ∧ E = 33 ∧ F = 17 →
  A + B + C + D + E + F = 49 ∧
  (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / (F : ℝ) :=
begin
  intros,
  split,
  { rcases h with ⟨_, _, _, _, _, _⟩,
    ring_nf,
    sorry },
  { sorry }
end

end rationalization_correct_l749_749542


namespace area_ratio_l749_749890

noncomputable def point := ℝ × ℝ

structure Triangle :=
(A B C : point)

def centroid (A B C : point) : point :=
(((A.1 + B.1 + C.1) / 3), ((A.2 + B.2 + C.2) / 3))

def vector_eq (O A B C : point) :=
(O.1 - A.1) + 2 * (O.1 - B.1) + 3 * (O.1 - C.1) = 0 ∧
(O.2 - A.2) + 2 * (O.2 - B.2) + 3 * (O.2 - C.2) = 0

noncomputable def area (A B C : point) : ℝ :=
abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem area_ratio (T : Triangle) (O : point)
  (h : vector_eq O T.A T.B T.C) :
  area T.A T.B T.C / area T.A O T.C = 3 := by
sorry

end area_ratio_l749_749890


namespace find_lambda_l749_749081

def vec (x y : ℝ) := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def condition (a b : ℝ × ℝ) (λ : ℝ) :=
  dot_product (vec (a.1 - λ * b.1) (a.2 - λ * b.2)) b = 0

theorem find_lambda
(a b : ℝ × ℝ)
(λ : ℝ)
(h₁ : a = (1, 3))
(h₂ : b = (3, 4))
(h₃ : condition a b λ) :
  λ = 3 / 5 :=
sorry

end find_lambda_l749_749081


namespace car_final_price_l749_749236

theorem car_final_price (initial_price : ℝ) (discount1 discount2 discount3 discount4 : ℝ) :
  initial_price = 20000 → discount1 = 0.25 → discount2 = 0.20 → discount3 = 0.15 → discount4 = 0.10 →
  let price1 := initial_price * (1 - discount1) in
  let price2 := price1 * (1 - discount2) in
  let price3 := price2 * (1 - discount3) in
  let final_price := price3 * (1 - discount4) in
  final_price = 9180 :=
begin
  intros,
  sorry
end

end car_final_price_l749_749236


namespace john_spent_fraction_on_snacks_l749_749495

theorem john_spent_fraction_on_snacks (x : ℚ) :
  (∀ (x : ℚ), (1 - x) * 20 - (3 / 4) * (1 - x) * 20 = 4) → (x = 1 / 5) :=
by sorry

end john_spent_fraction_on_snacks_l749_749495


namespace degree_of_minus_5x4y_l749_749564

def degree_of_monomial (coeff : Int) (x_exp y_exp : Nat) : Nat :=
  x_exp + y_exp

theorem degree_of_minus_5x4y : degree_of_monomial (-5) 4 1 = 5 :=
by
  sorry

end degree_of_minus_5x4y_l749_749564


namespace approx_subtraction_l749_749374

noncomputable def value1 := Real.sqrt (49 + 16)
noncomputable def value2 := Real.sqrt (36 - 9)

theorem approx_subtraction : (value1 - value2) ≈ 2.8661 := by
  sorry

end approx_subtraction_l749_749374


namespace g_neg_10_value_l749_749417

theorem g_neg_10_value
  (f : ℝ → ℝ)
  (y : ℝ → ℝ)
  (h₀ : ∀ x, y x = f x + x^3)
  (h₁ : ∀ x, y(-x) = -y x)
  (h₂ : f 10 = 10)
  : (f (-10) + 5) = -5 := by
  sorry

end g_neg_10_value_l749_749417


namespace find_lambda_l749_749017

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, 4)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- λ that satisfies the condition
theorem find_lambda (λ : ℝ) (h : dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b = 0) : λ = 3 / 5 := 
sorry

end find_lambda_l749_749017


namespace find_lambda_l749_749042

open Real

variable (a b : (Fin 2) → ℝ)

def dot_product (x y : (Fin 2) → ℝ) :=
  x 0 * y 0 + x 1 * y 1

theorem find_lambda 
  (a := fun i => if i = 0 then (1 : ℝ) else (3 : ℝ))
  (b := fun i => if i = 0 then (3 : ℝ) else (4 : ℝ))
  (h : dot_product (fun i => a i - λ * b i) b = 0) :
  λ = 3 / 5 :=
by
  sorry

end find_lambda_l749_749042


namespace arcsin_one_half_l749_749797

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l749_749797


namespace sum_g_0_to_2013_l749_749521

variable {ℝ : Type} [Real ℝ]

theorem sum_g_0_to_2013
  (f g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (f (x + y)) = f x + (x + y) * g y) :
  (Finset.range 2014).sum (λ n, g n) = 0 :=
by
  sorry

end sum_g_0_to_2013_l749_749521


namespace min_technicians_to_profit_l749_749301

theorem min_technicians_to_profit
  (daily_operational_cost : ℤ)
  (technician_hourly_wage : ℤ)
  (machines_per_hour : ℤ)
  (selling_price_per_machine : ℤ)
  (workday_hours : ℤ)
  (h_daily_operational_cost : daily_operational_cost = 1000)
  (h_technician_hourly_wage : technician_hourly_wage = 20)
  (h_machines_per_hour : machines_per_hour = 4)
  (h_selling_price_per_machine : selling_price_per_machine = 4.50)
  (h_workday_hours : workday_hours = 10) :
  ∃ n : ℤ, n >= 51 
  ∧ 180 * n > daily_operational_cost + 200 * n :=
by {
  -- given the assumptions, derive the statement
  use 51,
  have h1 : 180 * 51 > 180 * 50 := sorry,
  have h2 : daily_operational_cost + 200 * 51 - 1000 - 200 * 50 = 0 := sorry,
  exact ⟨h1, h2⟩
}

end min_technicians_to_profit_l749_749301


namespace remaining_volume_correct_l749_749745

-- Define the side length of the cube
def side_length : ℝ := 6

-- Define the radius of the cylindrical section
def cylinder_radius : ℝ := 3

-- Define the height of the cylindrical section (which is equal to the side length of the cube)
def cylinder_height : ℝ := side_length

-- Define the volume of the cube
def volume_cube : ℝ := side_length^3

-- Define the volume of the cylindrical section
def volume_cylinder : ℝ := Real.pi * cylinder_radius^2 * cylinder_height

-- Define the remaining volume after removing the cylindrical section from the cube
def remaining_volume : ℝ := volume_cube - volume_cylinder

-- Theorem stating the remaining volume is 216 - 54π cubic feet
theorem remaining_volume_correct : remaining_volume = 216 - 54 * Real.pi :=
by
  -- Proof will go here
  sorry

end remaining_volume_correct_l749_749745


namespace condition_for_complex_transformation_l749_749391

noncomputable def complexPair := let a := (1/2 : ℝ) in let b := (Real.sqrt 3 / 2 : ℝ) 
noncomputable def potentialPairs : list (ℝ × ℝ) := [(1/2, Real.sqrt 3 / 2), (1/2, -Real.sqrt 3 / 2)]

theorem condition_for_complex_transformation (a b : ℝ) :
    let P0 := (1 : ℝ, 0 : ℝ) in
    let next_point := fun (x y : ℝ) => (a * x - b * y, b * x + a * y) in
    let (P1x, P1y) := next_point P0.1 P0.2 in
    let (P2x, P2y) := next_point P1x P1y in
    let (P3x, P3y) := next_point P2x P2y in
    let (P4x, P4y) := next_point P3x P3y in
    let (P5x, P5y) := next_point P4x P4y in
    let (P6x, P6y) := next_point P5x P5y in
    P0 = (P6x, P6y) ∧
    list.distinct [(P0.1, P0.2), (P1x, P1y), (P2x, P2y), (P3x, P3y), (P4x, P4y), (P5x, P5y)] 
    → (a, b) ∈ potentialPairs :=
sorry

end condition_for_complex_transformation_l749_749391


namespace smallest_integer_with_18_divisors_l749_749636

theorem smallest_integer_with_18_divisors : ∃ n : ℕ, (∃ (p q r : ℕ), n = p^5 * q^2 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^5 ∧ prime p ∧ prime q)
  ∨ (∃ (p q r : ℕ), n = p^2 * q^2 * r^2 ∧ prime p ∧ prime q ∧ prime r)
  ∨ (∃ (p q : ℕ), n = p^8 * q ∧ prime p ∧ prime q)
  ∨ (∃ (p q : ℕ), n = p^17 ∧ prime p)
  ∧ n = 288 := sorry

end smallest_integer_with_18_divisors_l749_749636


namespace shapes_234_are_centrally_and_axisymmetric_l749_749776

-- Define each shape and their properties
inductive Shape
| parallelogram
| rectangle
| square
| rhombus
| isosceles_trapezoid

-- Define central symmetry and axisymmetry
def centrally_symmetric (s : Shape) : Prop :=
  s = Shape.parallelogram ∨ s = Shape.rectangle ∨ s = Shape.square ∨ s = Shape.rhombus

def axisymmetric (s : Shape) : Prop :=
  s = Shape.rectangle ∨ s = Shape.square ∨ s = Shape.rhombus ∨ s = Shape.isosceles_trapezoid

-- Proof problem: shapes ②, ③, and ④ are both axisymmetric and centrally symmetric
theorem shapes_234_are_centrally_and_axisymmetric :
  (centrally_symmetric Shape.rectangle ∧ axisymmetric Shape.rectangle) ∧
  (centrally_symmetric Shape.square ∧ axisymmetric Shape.square) ∧
  (centrally_symmetric Shape.rhombus ∧ axisymmetric Shape.rhombus) :=
by
  split; sorry

end shapes_234_are_centrally_and_axisymmetric_l749_749776
