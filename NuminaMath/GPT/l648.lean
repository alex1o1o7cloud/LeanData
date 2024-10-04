import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace problem_I_problem_II_range_m_l648_648732

def f (x : ℝ) : ℝ := 2^x - 1 / 2^|x|

theorem problem_I (x : ℝ) (hx : f(x) = 2) : x = Real.log 2 (1 + Real.sqrt 2) :=
sorry

theorem problem_II_range_m (m : ℝ) : (∀ t : ℝ, 1 ≤ t ∧ t ≤ 2 → 2^t * f(2*t) + m * f(t) ≥ 0) ↔ m ≥ -5 :=
sorry

end problem_I_problem_II_range_m_l648_648732


namespace coffee_shop_hours_l648_648584

noncomputable def coffee_shop_hours_per_day (h : ℕ) : Prop :=
  let weekday_cups := 5 * 10 * h
  let total_week_cups := weekday_cups + 120
  total_week_cups = 370 → h = 5

theorem coffee_shop_hours : ∃ h : ℕ, coffee_shop_hours_per_day h := 
begin
  existsi 5,
  intro h_eq,
  simp [h_eq],
  sorry
end

end coffee_shop_hours_l648_648584


namespace balls_distribution_l648_648321

theorem balls_distribution (balls boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 3) :
  let x := {n // 0 <= n}, y := {n // 0 <= n}, z := {n // 0 <= n} in
  ∑ x y z, (x + y + z = balls ∧ y >= z) = 12 :=
by {
  cases h_balls with | rfl,
  cases h_boxes with | rfl,
  sorry
}

end balls_distribution_l648_648321


namespace simplify_complex_subtraction_l648_648072

theorem simplify_complex_subtraction : (7 - 3 * complex.i) - (2 + 5 * complex.i) = 5 - 8 * complex.i := 
by
  sorry

end simplify_complex_subtraction_l648_648072


namespace sum_of_lengths_of_edges_geometric_progression_l648_648450

theorem sum_of_lengths_of_edges_geometric_progression :
  ∃ (a r : ℝ), (a / r) * a * (a * r) = 8 ∧ 2 * (a / r * a + a * a * r + a * r * a / r) = 32 ∧ 
  4 * ((a / r) + a + (a * r)) = 32 :=
by
  sorry

end sum_of_lengths_of_edges_geometric_progression_l648_648450


namespace exists_two_pairs_with_sum_ge_8_l648_648847

-- Define the conditions for the problem.
variable (circle : Fin 2009 → ℕ)
variable (Hsum : ∑ i : Fin 2009, circle i = 7036)

-- State the proof goal
theorem exists_two_pairs_with_sum_ge_8 :
  ∃ i j k l : Fin 2009, (abs (i.val - j.val) = 1 ∨ abs (i.val - j.val) = 2008) ∧ 
                      (abs (k.val - l.val) = 1 ∨ abs (k.val - l.val) = 2008) ∧ 
                      i ≠ k ∧ (circle i + circle j ≥ 8) ∧ (circle k + circle l ≥ 8) :=
by
  sorry

end exists_two_pairs_with_sum_ge_8_l648_648847


namespace problem_1_l648_648345

-- Define the line l as given in the conditions.
def line_l (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the point A(2,1) as given in the conditions.
def point_A (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Define the perpendicular relationship for line m passing through A
def perpendicular_line_m (m : ℝ) (A : ℝ × ℝ) : Prop :=
  ∃ b : ℝ, (A.1, A.2) = (2, 1) ∧ A.2 = (1 / 2) * A.1 + b ∧ b = -1

-- Define the parallel relationship for line n 
def parallel_line_n (x y b : ℝ) : Prop := y = -2 * x + b

-- Define the intercepts condition
def intercepts_condition (b : ℝ) : Prop := b + (b / 2) = 9 

-- Statement of the proof problem in Lean 4

theorem problem_1 :
  (∀ (x y : ℝ), line_l x y → ∃ m b A, point_A x y ∧ perpendicular_line_m m A → (x - 2 * y = 0)) ∧
  (∀ (x y b : ℝ), line_l x y → parallel_line_n x y b ∧ intercepts_condition b → (2 * x + y - 6 = 0)) :=
by
  -- Proof will go here
  sorry

end problem_1_l648_648345


namespace max_cookies_based_on_storage_l648_648371

def cookies_per_ingredient (milk_per_cookie sugar_per_cookie flour_per_cookie : ℝ) : ℝ :=
  min (80 / milk_per_cookie) (min (200 / sugar_per_cookie) (220 / flour_per_cookie))

theorem max_cookies_based_on_storage :
  ∀ (milk_per_cookie sugar_per_cookie flour_per_cookie : ℝ),
  milk_per_cookie = 10 / 40 ∧ sugar_per_cookie = 5 / 40 ∧ flour_per_cookie = 15 / 40 →
  cookies_per_ingredient milk_per_cookie sugar_per_cookie flour_per_cookie = 320 :=
by intros milk_per_cookie sugar_per_cookie flour_per_cookie h
   cases h with h₁ h2
   cases h2 with h₂ h₃
   simp [cookies_per_ingredient, h₁, h₂, h₃]
   sorry

end max_cookies_based_on_storage_l648_648371


namespace parametric_equations_curveC2_minimum_distance_M_to_curveC_l648_648727

noncomputable def curveC1_param (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α)

def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem parametric_equations_curveC2 (θ : ℝ) :
  scaling_transform (Real.cos θ) (Real.sin θ) = (3 * Real.cos θ, 2 * Real.sin θ) :=
sorry

noncomputable def curveC (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin θ + ρ * Real.cos θ = 10

noncomputable def distance_to_curveC (θ : ℝ) : ℝ :=
  abs (3 * Real.cos θ + 4 * Real.sin θ - 10) / Real.sqrt 5

theorem minimum_distance_M_to_curveC : 
  ∀ θ, distance_to_curveC θ >= Real.sqrt 5 :=
sorry

end parametric_equations_curveC2_minimum_distance_M_to_curveC_l648_648727


namespace sum_of_primes_between_1_and_20_l648_648495

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648495


namespace total_cost_is_eight_times_short_cost_l648_648031

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l648_648031


namespace jason_fires_weapon_every_15_seconds_l648_648564

theorem jason_fires_weapon_every_15_seconds
    (flame_duration_per_fire : ℕ)
    (total_flame_duration_per_minute : ℕ)
    (seconds_per_minute : ℕ)
    (h1 : flame_duration_per_fire = 5)
    (h2 : total_flame_duration_per_minute = 20)
    (h3 : seconds_per_minute = 60) :
    seconds_per_minute / (total_flame_duration_per_minute / flame_duration_per_fire) = 15 := 
by
  sorry

end jason_fires_weapon_every_15_seconds_l648_648564


namespace smallest_four_digit_divisible_five_smallest_primes_l648_648661

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l648_648661


namespace slope_sum_l648_648391

def f (x : ℝ) : ℝ := x / (x^2 + 3*x + 2)

def k (n : ℕ) (h : n > 0) : ℝ := 1 / (n^2 + 3*n + 2)

def sum_slopes (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, k (i + 1) (Nat.succ_pos i))

theorem slope_sum (n : ℕ) : sum_slopes n = n / (2 * n + 4) := sorry

end slope_sum_l648_648391


namespace inner_cube_surface_area_l648_648955

theorem inner_cube_surface_area (S : ℝ) (hc : S = 54) : 
  ∃ (s_inner : ℝ), s_inner ^ 2 * 6 = 54 :=
by 
  have h_side_length := sqrt (S / 6) 
  have h_diameter := sqrt (S / 6) 
  have h_height := sqrt (S / 6) 
  have h_radius := (sqrt (S / 6)) / 2 
  use sqrt (S / 6)
  sorry

end inner_cube_surface_area_l648_648955


namespace articles_count_l648_648766

noncomputable def cost_price_per_article : ℝ := 1
noncomputable def selling_price_per_article (x : ℝ) : ℝ := x / 16
noncomputable def profit : ℝ := 0.50

theorem articles_count (x : ℝ) (h1 : cost_price_per_article * x = selling_price_per_article x * 16)
                       (h2 : selling_price_per_article 16 = cost_price_per_article * (1 + profit)) :
  x = 24 :=
by
  sorry

end articles_count_l648_648766


namespace bob_salary_march_l648_648619

theorem bob_salary_march : 
  let january_salary := 3000
  let february_raise := 0.30
  let march_cut := 0.10
  let february_salary := january_salary * (1 + february_raise)
  let march_salary := february_salary * (1 - march_cut)
  march_salary = 3510 :=
by {
  let january_salary := 3000
  let february_raise := 0.30
  let march_cut := 0.10
  let february_salary := january_salary * (1 + february_raise)
  let march_salary := february_salary * (1 - march_cut)
  show march_salary = 3510,
  sorry
}


end bob_salary_march_l648_648619


namespace sum_primes_between_1_and_20_l648_648501

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648501


namespace minimum_value_is_138_l648_648372

-- Definition of problem conditions and question
def is_digit (n : ℕ) : Prop := n < 10
def digits (A : ℕ) : List ℕ := A.digits 10

def multiple_of_3_not_9 (A : ℕ) : Prop :=
  A % 3 = 0 ∧ A % 9 ≠ 0

def product_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· * ·) 1

def sum_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· + ·) 0

def given_condition (A : ℕ) : Prop :=
  A % 9 = 0 → False ∧
  (A + product_of_digits A) % 9 = 0

-- Main goal: Prove that the minimum value A == 138 satisfies the given conditions
theorem minimum_value_is_138 : ∃ A, A = 138 ∧
  multiple_of_3_not_9 A ∧
  given_condition A :=
sorry

end minimum_value_is_138_l648_648372


namespace largest_students_exceeding_median_l648_648398

-- Define the number of students
def num_students : ℕ := 101

-- Define what it means for a student to exceed the median score
def exceeds_median (scores : list ℕ) (student_index : ℕ) : Prop :=
  scores.nth student_index > scores.nth (num_students / 2)

-- Theorem: the largest number of students who can exceed the median score
theorem largest_students_exceeding_median (scores : list ℕ) (h_len : scores.length = num_students)
  (h_sorted : list.sorted scores) : 
  ∃ n, n = 50 ∧ ∀ i, i ≥ num_students / 2 → i < num_students → exceeds_median scores i :=
sorry

end largest_students_exceeding_median_l648_648398


namespace next_term_geometric_sequence_l648_648487

theorem next_term_geometric_sequence (y : ℝ) (h0 : y ≠ 0) :
  let r := 3 * y in
  let term := 81 * y^3 in
  term * r = 243 * y^4 :=
by
  let r := 3 * y
  let term := 81 * y^3
  have h : term * r = 243 * y^4 := sorry
  exact h

end next_term_geometric_sequence_l648_648487


namespace smallest_real_mu_l648_648254

theorem smallest_real_mu :
  ∃ μ : ℝ, μ = 2 ∧ ∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d →
  a^2 + b^2 + c^2 + d^2 ≥ 2 * a * b + μ * b * c + 2 * c * d :=
by
  let μ := 2
  use μ
  split
  · exact rfl
  · intros a b c d ha hb hc hd
    calc
      a^2 + b^2 + c^2 + d^2
          ≥ 2 * a * b + μ * b * c + 2 * c * d := sorry

end smallest_real_mu_l648_648254


namespace sum_of_primes_between_1_and_20_l648_648491

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648491


namespace no_such_ab_l648_648236

theorem no_such_ab (a b : ℤ) : ¬ (2006^2 ∣ a^2006 + b^2006 + 1) :=
sorry

end no_such_ab_l648_648236


namespace simplify_A_condition_1_condition_2_l648_648317

variable {a : ℝ}

def A : ℝ := (2 / (a^2 - 4)) - (1 / (a * (a - 2)))

theorem simplify_A (h : a ≠ 2 ∧ a ≠ -2 ∧ a ≠ 0) : A = 1 / (a * (a + 2)) :=
  sorry

theorem condition_1 (h1 : a * (a + 2) = 8) (h : a ≠ 2 ∧ a ≠ -2 ∧ a ≠ 0) : A = 1 / 8 :=
  by
  have hA := simplify_A h
  rw [h1] at hA
  exact hA

theorem condition_2 (h2 : a^2 + a = 8 - a) (h : a ≠ 2 ∧ a ≠ -2 ∧ a ≠ 0) : A = 1 / 8 :=
  by
  have ha : a * (a + 2) = 8 :=
    by linarith
  exact condition_1 ha h

end simplify_A_condition_1_condition_2_l648_648317


namespace arithmetic_sqrt_of_16_l648_648427

theorem arithmetic_sqrt_of_16 : sqrt 16 = 4 :=
by 
  sorry

end arithmetic_sqrt_of_16_l648_648427


namespace find_fx_when_x_positive_l648_648282

def isOddFunction {α : Type} [AddGroup α] [Neg α] (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)
variable (h_odd : isOddFunction f)
variable (h_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + x)

theorem find_fx_when_x_positive : ∀ x : ℝ, x > 0 → f x = x^2 + x :=
by
  sorry

end find_fx_when_x_positive_l648_648282


namespace tangent_line_circle_intersection_l648_648739

open Real

noncomputable def is_tangent (θ : ℝ) : Prop :=
  abs (4 * tan θ) / sqrt ((tan θ) ^ 2 + 1) = 2

theorem tangent_line_circle_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < π) :
  is_tangent θ ↔ θ = π / 6 ∨ θ = 5 * π / 6 :=
sorry

end tangent_line_circle_intersection_l648_648739


namespace chandler_bike_purchase_l648_648623

theorem chandler_bike_purchase : 
    ∀ (x : ℕ), (200 + 20 * x = 800) → (x = 30) :=
by
  intros x h
  sorry

end chandler_bike_purchase_l648_648623


namespace graph_self_intersections_l648_648594

noncomputable def parametric_x (t : ℝ) : ℝ := Real.cos t + t / 3
noncomputable def parametric_y (t : ℝ) : ℝ := Real.sin t

theorem graph_self_intersections :
  let intersections := {t1 t2 : ℝ // t1 ≠ t2 ∧ parametric_x t1 = parametric_x t2 ∧ parametric_y t1 = parametric_y t2 ∧ 3 ≤ parametric_x t1 ∧ parametric_x t1 ≤ 45}
  finset.card (finset.univ.filter (λ n, ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ parametric_x t1 = parametric_x t2 ∧ parametric_y t1 = parametric_y t2 ∧ 3 ≤ parametric_x t1 ∧ parametric_x t1 ≤ 45)) = 12 :=
begin
  sorry
end

end graph_self_intersections_l648_648594


namespace MN_bisected_by_PQ_l648_648010

-- Define an isosceles triangle with vertices A, B, C and AC = BC
variable (A B C M N : ℝ × ℝ)
variable (hAC_eq_BC : dist A C = dist B C)
variable (hM_on_AC : M ∈ segment ℝ A C)
variable (hN_on_BC : N ∈ segment ℝ B C)
variable (hCM_CN_eq_AC : dist C M + dist C N = dist A C)

-- Define P and Q as the midpoints of AC and BC
def P : ℝ × ℝ := midpoint ℝ A C
def Q : ℝ × ℝ := midpoint ℝ B C

-- Prove the segment MN is bisected by the line connecting P and Q
theorem MN_bisected_by_PQ :
  ∃ R : ℝ × ℝ, R ∈ line ℝ P Q ∧ is_midpoint ℝ R M N :=
sorry

end MN_bisected_by_PQ_l648_648010


namespace smallest_number_divisible_l648_648490

theorem smallest_number_divisible
  (n : ℕ)
  (h1 : ∀ m, m % 3 = 0 → m % 5 = 0 → m % 7 = 0 → m % 13 = 0 → 1000 < m → n ≤ m) :
  n = 1365 :=
by
  let n := 1365
  trivial
  sorry

end smallest_number_divisible_l648_648490


namespace channel_depth_l648_648930

theorem channel_depth (top_width bottom_width : ℝ) (cross_section_area : ℝ)
  (htop : top_width = 14)
  (hbottom : bottom_width = 8)
  (harea : cross_section_area = 880) :
  ∃ depth : ℝ, depth = 80 :=
begin
  sorry
end

end channel_depth_l648_648930


namespace find_lambda_mu_values_l648_648316

noncomputable def vector_lambda_mu (λ μ : ℝ) : Prop :=
  let a := (λ + 1, 0, 2 * λ)
  let b := (6, 2 * μ - 1, 2)
  ∃ M : ℝ, a = (M * b.1, M * b.2, M * b.3)

theorem find_lambda_mu_values (λ μ : ℝ) :
  vector_lambda_mu λ μ → λ = 1/5 ∧ μ = 1/2 :=
by
  sorry

end find_lambda_mu_values_l648_648316


namespace petya_purchase_cost_l648_648042

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l648_648042


namespace limit_of_S_n_l648_648950

-- Definition of the side length of the initial square
def side_length_first_square : ℝ := 10

-- Definition of the radius of the first circle
def radius_first_circle : ℝ := side_length_first_square / 2

-- Definition of the area of the first circle
def area_first_circle : ℝ := π * radius_first_circle ^ 2

-- Definition of the areas of subsequent circles
noncomputable def area_circle (n : ℕ) : ℝ :=
  let side_n := side_length_first_square / (2 ^ n)
  π * (side_n / 2) ^ 2

-- Sum of areas of the first n circles
noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, area_circle (i + 1)

-- The limit we want to prove
theorem limit_of_S_n : tendsto (λ n, S_n n) at_top (𝓝 (100 * π / 3)) :=
sorry

end limit_of_S_n_l648_648950


namespace max_ordered_triples_l648_648288

-- Given positive integers a1, a2, a3, a4 satisfying the following condition:
-- In any circular arrangement of these four numbers, there are two adjacent elements that are not coprime.

def circular_condition (a1 a2 a3 a4 : ℕ) : Prop :=
  (¬ coprime a1 a2 ∨ ¬ coprime a2 a3 ∨ ¬ coprime a3 a4 ∨ ¬ coprime a4 a1) ∧
  (¬ coprime a2 a1 ∨ ¬ coprime a1 a4 ∨ ¬ coprime a4 a3 ∨ ¬ coprime a3 a2)

theorem max_ordered_triples (a1 a2 a3 a4 : ℕ) :
  circular_condition a1 a2 a3 a4 →
  ∃ n : ℕ, n = 16 ∧
  ∀ i j k : fin 4, i ≠ j ∧ j ≠ k ∧ k ≠ i → (nat.gcd a1 a2) ^ 2 ∣ a1 ∨ -- This part should be adjusted in loop
  sorry

end max_ordered_triples_l648_648288


namespace lower_limit_of_range_l648_648115

-- Definitions and conditions based on part a)
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def not_divisible_by_self_factorial (n : ℕ) : Prop := ¬ (n ∣ factorial (n - 1))

def prime_numbers_up_to_40 : list ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Statement to be proved
theorem lower_limit_of_range : ∃ l : ℕ, (∀ n : ℕ, 18 ≤ n ∧ n ≤ 40 → not_divisible_by_self_factorial n) ∧ list.length (list.filter is_prime (list.range' l 23)) = 7 := 
sorry

end lower_limit_of_range_l648_648115


namespace area_of_rectangular_field_l648_648147

def length (L : ℝ) : Prop := L > 0
def breadth (L : ℝ) (B : ℝ) : Prop := B = 0.6 * L
def perimeter (L : ℝ) (B : ℝ) : Prop := 2 * L + 2 * B = 800
def area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

theorem area_of_rectangular_field (L B A : ℝ) 
  (h1 : breadth L B) 
  (h2 : perimeter L B) : 
  area L B 37500 :=
sorry

end area_of_rectangular_field_l648_648147


namespace count_paths_outside_square_l648_648577

theorem count_paths_outside_square :
  let paths := {p : list (int × int) | 
    ∀ i < p.length, (p.get i).fst ∈ {x | x ≤ -3 ∨ x ≥ 3} ∧ (p.get i).snd ∈ {y | y ≤ -3 ∨ y ≥ 3} ∧ 
    (∀ j < 20, (p.get j).fst ≤ (p.get (j+1)).fst + 1 ∧ (p.get j).snd ≤ (p.get (j+1)).snd + 1) ∧
    ((p.head = (-5, -5)) ∧ (p.last = (5, 5))) ∧ 
    p.length = 21 }
  in paths.card = 4252 :=
by sorry

end count_paths_outside_square_l648_648577


namespace symmetry_about_pi_over_12_maximum_value_l648_648304

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (1/2) * Math.cos x * Math.sin (x + Real.pi / 3)

-- Proposition: The graph of f(x) is symmetric about x = π/12
theorem symmetry_about_pi_over_12 : 
  ∀ x : ℝ, f (Real.pi / 12 - x) = f (Real.pi / 12 + x) := 
sorry

-- Proposition: The maximum value of f(x) is 1/4 + sqrt(3)/8
theorem maximum_value : 
  ∃ x : ℝ, f x = 1/4 + Real.sqrt 3 / 8 := 
sorry

end symmetry_about_pi_over_12_maximum_value_l648_648304


namespace sum_primes_between_1_and_20_l648_648506

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648506


namespace janet_dresses_total_pockets_l648_648365

theorem janet_dresses_total_pockets :
  let dresses := 24
  let with_pockets := dresses / 2
  let with_two_pockets := with_pockets / 3
  let with_three_pockets := with_pockets - with_two_pockets
  let total_two_pockets := with_two_pockets * 2
  let total_three_pockets := with_three_pockets * 3
  total_two_pockets + total_three_pockets = 32 := by
  sorry

end janet_dresses_total_pockets_l648_648365


namespace number_of_people_in_partnership_l648_648152

-- Defining the conditions
def surplus_condition (people : ℕ) : Prop :=
  5 * people + 45 = 7 * people + 3

-- Proving the number of people in the partnership
theorem number_of_people_in_partnership : ∃ x : ℕ, surplus_condition x ∧ x = 21 :=
by
  use 21
  unfold surplus_condition
  simp
  sorry -- The proof that 5 * 21 + 45 = 7 * 21 + 3 satisfies this is skipped

end number_of_people_in_partnership_l648_648152


namespace minuend_is_12_point_5_l648_648449

theorem minuend_is_12_point_5 (x y : ℝ) (h : x + y + (x - y) = 25) : x = 12.5 := by
  sorry

end minuend_is_12_point_5_l648_648449


namespace period_and_max_value_f_l648_648876

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem period_and_max_value_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ y > 0, y < T → ∃ x, f (x + y) ≠ f x) ∧
  (∃ M, ∀ x, f x ≤ M ∧ (∃ x, f x = M)) :=
begin
  sorry
end

end period_and_max_value_f_l648_648876


namespace specific_values_exist_l648_648635

def expr_equal_for_specific_values (a b c : ℝ) : Prop :=
  a + b^2 * c = (a^2 + b) * (a + c)

theorem specific_values_exist :
  ∃ a b c : ℝ, expr_equal_for_specific_values a b c :=
sorry

end specific_values_exist_l648_648635


namespace union_intersection_l648_648375

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C : Set ℝ := {x | x^2 - 3 * x + 2 > 0}
def D : Set ℕ := {0, 3, 4}

theorem union_intersection:
  (A ∪ B).inter (C ∩ Set.Ioo (-∞:ℝ) (∞:ℝ)) = D := by
  sorry

end union_intersection_l648_648375


namespace polynomial_properties_l648_648802

theorem polynomial_properties :
  let poly := (λ x : ℚ, (x^2 - 2/x)^6) in
  (poly 1 = 1) ∧
  (∑ i in finset.range 7, nat.choose 6 i * (2 : ℚ) ^ i = 64) ∧
  (nat.choose 6 4 * (-2)^4 = 240) ∧
  (nat.choose 6 2 * 4 = 60) :=
by {
  let poly := (λ x : ℚ, (x^2 - 2/x)^6),
  have h_sum_coeffs : poly 1 = 1 := by {
    calc poly 1 = (1^2 - 2 / 1)^6 : by simp
         ... = (-1)^6 : by norm_num
         ... = 1 : by norm_num
  },
  have h_binom_sum : ∑ i in finset.range 7, nat.choose 6 i * (2 : ℚ) ^ i = 64 := by {
    calc ∑ i in finset.range 7, nat.choose 6 i * (2 : ℚ) ^ i
         = (2 : ℚ)^6 : by { rw finset.sum_binomial, simp }
         ... = 64 : by norm_num
  },
  have h_const_term : nat.choose 6 4 * (-2)^4 = 240 := by {
    calc nat.choose 6 4 * (-2)^4
         = 15 * 16 : by { norm_num, exact nat.choose_succ_self 5 4 }
         ... = 240 : by norm_num
  },
  have h_coeff_x6 : nat.choose 6 2 * 4 = 60 := by {
    calc nat.choose 6 2 * 4
         = 15 * 4 : by { norm_num, exact nat.choose_succ_self 5 2 }
         ... = 60 : by norm_num
  },
  exact ⟨h_sum_coeffs, h_binom_sum, h_const_term, h_coeff_x6⟩
}

end polynomial_properties_l648_648802


namespace right_triangle_segments_ratio_l648_648340

theorem right_triangle_segments_ratio (x : ℝ) (h : x > 0) :
  ∀ (A B C D: Type) (AB AC: ℝ),
  BC = x ∧ AB = 3 * x ∧ AC = (BC^2 + AB^2)^(1/2) ∧ is_right_triangle A B C
  → ∃ (ratio : ℝ), ratio = 9 :=
by
  intro A B C D AB AC
  sorry

end right_triangle_segments_ratio_l648_648340


namespace product_of_roots_l648_648216

noncomputable def a := Real.root 16 4
noncomputable def b := Real.root 9 6

theorem product_of_roots : a * b = 2 * Real.root 3 3 := by
  sorry

end product_of_roots_l648_648216


namespace problem_1_problem_2_l648_648943

-- Proof Problem 1 statement
theorem problem_1 (α : ℝ) (h1 : tan α = 2) :
  (3 * sin α + 2 * cos α) / (sin α - cos α) = -8 :=
sorry

-- Proof Problem 2 statement
theorem problem_2 (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : sin α + cos α = (1 / 5)) :
  tan α = - (4 / 3) :=
sorry

end problem_1_problem_2_l648_648943


namespace minimum_number_of_circles_l648_648130

-- Define the problem conditions
def conditions_of_problem (circles : ℕ) (n : ℕ) (highlighted_lines : ℕ) (sides_of_regular_2011_gon : ℕ) : Prop :=
  circles ≥ n ∧ highlighted_lines = sides_of_regular_2011_gon

-- The main theorem we need to prove
theorem minimum_number_of_circles :
  ∀ (n circles highlighted_lines sides_of_regular_2011_gon : ℕ),
    sides_of_regular_2011_gon = 2011 ∧ (highlighted_lines = sides_of_regular_2011_gon * 2) ∧ conditions_of_problem circles n highlighted_lines sides_of_regular_2011_gon → n = 504 :=
by
  sorry

end minimum_number_of_circles_l648_648130


namespace jogger_distance_ahead_l648_648177

def speed_jogger_kmph : ℕ := 9
def speed_train_kmph : ℕ := 45
def length_train_m : ℕ := 120
def time_to_pass_jogger_s : ℕ := 36

theorem jogger_distance_ahead :
  let relative_speed_mps := (speed_train_kmph - speed_jogger_kmph) * 1000 / 3600
  let distance_covered_m := relative_speed_mps * time_to_pass_jogger_s
  let jogger_distance_ahead : ℕ := distance_covered_m - length_train_m
  jogger_distance_ahead = 240 :=
by
  sorry

end jogger_distance_ahead_l648_648177


namespace total_cost_shorts_tshirt_boots_shinguards_l648_648021

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l648_648021


namespace var_power_l648_648758

theorem var_power {a b c x y z : ℝ} (h1 : x = a * y^4) (h2 : y = b * z^(1/3)) :
  ∃ n : ℝ, x = c * z^n ∧ n = 4/3 := by
  sorry

end var_power_l648_648758


namespace repeating_decimal_to_fraction_l648_648641

theorem repeating_decimal_to_fraction : (0.7 + 23 / 99 / 10) = (62519 / 66000) := by
  sorry

end repeating_decimal_to_fraction_l648_648641


namespace log_sum_eq_neg_one_exists_largest_m_l648_648628

-- Part 1: Prove the sum of logarithms
theorem log_sum_eq_neg_one : 
  (∑ n in Finset.range 9, Real.log (n.succ / (n.succ + 1))) = -1 := 
by 
  sorry

-- Part 2: Prove the existence and value of the largest positive integer m
theorem exists_largest_m : 
  ∃ m : ℕ, 
  m = 11 ∧ ∀ n : ℕ, (n > 0) → (∑ k in Finset.range (n + 1), ∑ i in Finset.range (n + k), (1 : ℝ)/(i + n + 1)) > 11/24 :=
by
  sorry

end log_sum_eq_neg_one_exists_largest_m_l648_648628


namespace find_10a_l648_648765

theorem find_10a (a b : ℕ) (h₁ : a = 6) (h₂ : b = 20) : 10 * a = 60 :=
by
  -- Since h₁ states that a = 6, we can substitute a to get 10 * 6 = 60.
  have step1 : 10 * 6 = 60 := by norm_num
  -- Substituting h₁ into 10 * a gives us our theorem.
  rw [h₁]
  exact step1

end find_10a_l648_648765


namespace parade_team_people_count_min_l648_648971

theorem parade_team_people_count_min (n : ℕ) :
  n ≥ 1000 ∧ n % 5 = 0 ∧ n % 4 = 3 ∧ n % 3 = 2 ∧ n % 2 = 1 → n = 1045 :=
by
  sorry

end parade_team_people_count_min_l648_648971


namespace sum_primes_between_1_and_20_l648_648503

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648503


namespace identify_odd_monotonic_function_l648_648609

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem identify_odd_monotonic_function : 
  (is_odd_function (λ x : ℝ, x - 1) ∧ is_monotonically_increasing (λ x : ℝ, x - 1)) ∨ 
  (is_odd_function (λ x : ℝ, Real.tan x) ∧ is_monotonically_increasing (λ x : ℝ, Real.tan x)) ∨ 
  (is_odd_function (λ x : ℝ, x^3) ∧ is_monotonically_increasing (λ x : ℝ, x^3)) ∨ 
  (is_odd_function (λ x : ℝ, Real.log x) ∧ is_monotonically_increasing (λ x : ℝ, Real.log x)) → 
  ( ∀ f : ℝ → ℝ, (is_odd_function f ∧ is_monotonically_increasing f) → f = (λ x : ℝ, x^3)) := 
by
  sorry

end identify_odd_monotonic_function_l648_648609


namespace probability_at_least_one_passes_l648_648471
open Classical

variable (P_A P_B : ℝ)
axiom prob_A : P_A = 3/4
axiom prob_B : P_B = 2/3
axiom independent : P_A * (1 - P_B) = (1 - P_A) * P_B

theorem probability_at_least_one_passes : 
  let prob_none_pass := (1 - P_A) * (1 - P_B) in
  let prob_at_least_one_pass := 1 - prob_none_pass in
  prob_at_least_one_pass = 11/12 := 
by 
  have h1 : 1 - P_A = 1/4 := by rw [prob_A]; norm_num,
  have h2 : 1 - P_B = 1/3 := by rw [prob_B]; norm_num,
  have h3 : (1 - P_A) * (1 - P_B) = 1/12 := by rw [h1, h2]; norm_num,
  have prob_none_pass := (1 - P_A) * (1 - P_B),
  have prob_at_least_one_pass := 1 - prob_none_pass,
  calc
    prob_at_least_one_pass = 1 - (1/12) : by rw [h3]
    ... = 11/12 : by norm_num

end probability_at_least_one_passes_l648_648471


namespace smallest_four_digit_number_divisible_by_smallest_primes_l648_648657

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l648_648657


namespace total_cost_is_eight_times_l648_648051

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l648_648051


namespace radius_third_circle_l648_648469

/-- Two circles are centered at points A and B respectively with radii 2 and 5.
    These circles are externally tangent to each other.
    Determine the radius of the third circle that is tangent to both circles
    and one of their common external tangents. -/
theorem radius_third_circle 
  (A B : Type) [normed_add_tors_group A] [normed_add_tors_group B]
  (distance_AB : ℝ) 
  (radius_A : ℝ) (radius_B : ℝ)
  (tangent_distance : distance_AB = radius_A + radius_B)
  (radius_A_eq_2 : radius_A = 2) 
  (radius_B_eq_5 : radius_B = 5) 
  (radius_third : ℝ) :
  (radius_third = (3 + real.sqrt 51) / 2) :=
sorry

end radius_third_circle_l648_648469


namespace petya_purchase_cost_l648_648045

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l648_648045


namespace andy_last_problem_l648_648206

theorem andy_last_problem (start_num : ℕ) (num_solved : ℕ) (result : ℕ) : 
  start_num = 78 → 
  num_solved = 48 → 
  result = start_num + num_solved - 1 → 
  result = 125 :=
by
  sorry

end andy_last_problem_l648_648206


namespace value_of_f_at_5_l648_648382

theorem value_of_f_at_5 (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = - f x) 
  (h_period : ∀ x, f (x + 4) = f x)
  (h_func : ∀ x, -2 ≤ x ∧ x < 0 → f x = 3 * x + 1) : 
  f 5 = 2 :=
  sorry

end value_of_f_at_5_l648_648382


namespace marathon_distance_l648_648125

theorem marathon_distance (d_1 : ℕ) (n : ℕ) (h1 : d_1 = 3) (h2 : n = 5): 
  (2 ^ (n - 1)) * d_1 = 48 :=
by
  sorry

end marathon_distance_l648_648125


namespace lengths_of_trains_combined_l648_648463

-- Define constants and conditions
def speed_train1 : ℝ := 80
def speed_train2 : ℝ := 60
def overtake_time : ℝ := 18
def kmhr_to_ms : ℝ := 5 / 18

-- Calculate relative speed in m/s
def relative_speed : ℝ := (speed_train1 - speed_train2) * kmhr_to_ms

-- Define combined length of trains
def combined_length_of_trains : ℝ := relative_speed * overtake_time

-- Problem statement: Prove that the length of both trains combined is 100 meters
theorem lengths_of_trains_combined : combined_length_of_trains = 100 := by
  sorry

end lengths_of_trains_combined_l648_648463


namespace factory_temporary_employee_percentage_l648_648792

theorem factory_temporary_employee_percentage :
  (t n p_t p_n : ℕ) (h_tot : t + n = 100)
  (h_t : 2 * t = 100) (h_n : 2 * n = 100) 
  (h_p_t : 2 * p_t = t) (h_p_n : 2 * p_n = n) :
  2 * (100 - (p_t + p_n)) = 100 := by
sory

end factory_temporary_employee_percentage_l648_648792


namespace range_of_x_l648_648756

noncomputable def log_range (x : ℝ) : Prop :=
  log x < 0

theorem range_of_x (x : ℝ) (h : log_range x) : 0 < x ∧ x < 1 :=
by
  sorry

end range_of_x_l648_648756


namespace find_star_value_l648_648947

theorem find_star_value (a b c d e : ℤ) (h1 : 2 + 0 + 1 + a = 11)
                                 (h2 : 0 + 1 + a + b = 11)
                                 (h3 : 1 + a + b + 2 = 11)
                                 (h4 : a + b + 2 + 0 = 11)
                                 (h5 : 2 + 0 + 1 + c = 11)
                                 (h6 : 0 + 1 + c + d = 11)
                                 (h7 : 1 + c + d + 2 = 11)
                                 (h8 : c + d + 2 + 0 = 11)
                                 (h9 : 2 + 0 + 1 + e = 11)
                                 (h10 : 0 + 1 + e + a = 11)
                                 (h11 : 1 + e + a + 2 = 11)
                                 (h12 : e + a + 2 + 0 = 11) :
  a = 8 :=
by {
  -- Using the first equation: h1
  have ha : a = 8,
  { linarith [h1] },
  -- All subsequent equations must satisfy this value since the numbers repeat
  exact ha,
}

end find_star_value_l648_648947


namespace annie_spent_on_candies_l648_648207

theorem annie_spent_on_candies : 
  ∀ (num_classmates : ℕ) (candies_per_classmate : ℕ) (candies_left : ℕ) (cost_per_candy : ℚ),
  num_classmates = 35 →
  candies_per_classmate = 2 →
  candies_left = 12 →
  cost_per_candy = 0.1 →
  (num_classmates * candies_per_classmate + candies_left) * cost_per_candy = 8.2 :=
by
  intros num_classmates candies_per_classmate candies_left cost_per_candy
         h_classmates h_candies_per_classmate h_candies_left h_cost_per_candy
  simp [h_classmates, h_candies_per_classmate, h_candies_left, h_cost_per_candy]
  sorry

end annie_spent_on_candies_l648_648207


namespace cat_speed_theorem_l648_648124

noncomputable def cat_speed_proof (rabbit_speed : ℕ) (head_start : ℕ) (catch_up_time : ℕ) : ℕ :=
  let v := (rabbit_speed * catch_up_time) / (head_start + catch_up_time) in
  v

theorem cat_speed_theorem : cat_speed_proof 25 1 1 = 20 :=
by
  sorry

end cat_speed_theorem_l648_648124


namespace remaining_amounts_to_pay_l648_648171

theorem remaining_amounts_to_pay (priceA priceB priceC : ℝ) (depositA depositB depositC : ℝ)
  (discountA discountB : ℝ) (taxA taxB taxC : ℝ) :
  priceA = 1200 → depositA = 120 → discountA = 0.15 → taxA = 0.07 →
  priceB = 2500 → depositB = 250 → discountB = 0.10 → taxB = 0.08 →
  priceC = 3800 → depositC = 380 → taxC = 0.05 →
  let priceA_after_discount := priceA * (1 - discountA) in
  let totalA_after_tax := priceA_after_discount * (1 + taxA) in
  let remainingA := totalA_after_tax - depositA in
  let priceB_after_discount := priceB * (1 - discountB) in
  let totalB_after_tax := priceB_after_discount * (1 + taxB) in
  let remainingB := totalB_after_tax - depositB in
  let totalC_after_tax := priceC * (1 + taxC) in
  let remainingC := totalC_after_tax - depositC in
  remainingA = 971.40 ∧ remainingB = 2180 ∧ remainingC = 3610 :=
by
  intros
  unfold priceA_after_discount totalA_after_tax remainingA
  unfold priceB_after_discount totalB_after_tax remainingB
  unfold totalC_after_tax remainingC
  sorry

end remaining_amounts_to_pay_l648_648171


namespace max_a_value_l648_648747

theorem max_a_value : ∃ a b : ℕ, 1 < a ∧ a < b ∧
  (∀ x y : ℝ, y = -2 * x + 4033 ∧ y = |x - 1| + |x + a| + |x - b| → 
  a = 4031) := sorry

end max_a_value_l648_648747


namespace sum_of_primes_1_to_20_l648_648562

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648562


namespace min_frac_inv_proof_l648_648389

noncomputable def min_value_frac_inv (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 15) : ℝ :=
  ∃ (value : ℝ), (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 15 → value ≤ (1/x + 1/y)) ∧ 
    value = 4/15

theorem min_frac_inv_proof : min_value_frac_inv := by
  sorry

end min_frac_inv_proof_l648_648389


namespace arthur_day1_cost_l648_648613

-- Define the relevant parameters
variables (H D : ℝ) -- H represents the cost of a hamburger, D represents the cost of a hot dog
variables (day1_hamburgers day1_hotdogs day2_hamburgers day2_hotdogs : ℝ) 
variables (day2_total_cost : ℝ)

-- Provide the specific values given in the problem
def day1_hamburgers := 3
def day1_hotdogs := 4
def day2_hamburgers := 2
def day2_hotdogs := 3
def day2_total_cost := 7
def D := 1

theorem arthur_day1_cost :
  ∃ H : ℝ, (day2_hamburgers * H + day2_hotdogs * D = day2_total_cost) ∧ (day1_hamburgers * H + day1_hotdogs * D = 10) :=
sorry

end arthur_day1_cost_l648_648613


namespace cans_needed_eq_l648_648405

axiom Paula_initial_rooms : ℕ
axiom Paula_lost_cans : ℕ
axiom Paula_after_loss_rooms : ℕ
axiom cans_for_25_rooms : ℕ

theorem cans_needed_eq :
  Paula_initial_rooms = 30 →
  Paula_lost_cans = 3 →
  Paula_after_loss_rooms = 25 →
  cans_for_25_rooms = 15 :=
by
  intros
  sorry

end cans_needed_eq_l648_648405


namespace distance_AB_is_8_l648_648708

-- Define points A and B
def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (4, 5)

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Statement to prove that the distance between points A and B is 8.
theorem distance_AB_is_8 : distance A B = 8 := 
by
  sorry

end distance_AB_is_8_l648_648708


namespace total_cost_shorts_tshirt_boots_shinguards_l648_648024

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l648_648024


namespace sum_of_primes_1_to_20_l648_648532

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648532


namespace min_period_and_max_value_l648_648873

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∃ T > 0, T = 6 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x0, f x0 = sqrt 2)) :=
by
  sorry

end min_period_and_max_value_l648_648873


namespace roger_total_coins_l648_648571

theorem roger_total_coins :
  let piles_quarters := 3 in
  let piles_dimes := 3 in
  let coins_per_pile := 7 in
  let quarters := piles_quarters * coins_per_pile in
  let dimes := piles_dimes * coins_per_pile in
  quarters + dimes = 42 :=
by
  let piles_quarters := 3
  let piles_dimes := 3
  let coins_per_pile := 7
  let quarters := piles_quarters * coins_per_pile
  let dimes := piles_dimes * coins_per_pile
  show quarters + dimes = 42
  sorry

end roger_total_coins_l648_648571


namespace cost_of_each_television_l648_648208

-- Define the conditions
def number_of_televisions : Nat := 5
def number_of_figurines : Nat := 10
def cost_per_figurine : Nat := 1
def total_spent : Nat := 260

-- Define the proof problem
theorem cost_of_each_television (T : Nat) :
  (number_of_televisions * T + number_of_figurines * cost_per_figurine = total_spent) → (T = 50) :=
by
  sorry

end cost_of_each_television_l648_648208


namespace log_eq_solution_l648_648421

theorem log_eq_solution (x : ℝ) : 
  (log 2 (9^x - 5) = 2 + log 2 (3^x - 2)) → x = 1 :=
by 
  sorry 

end log_eq_solution_l648_648421


namespace total_cost_l648_648103

variable (m n : ℕ)

theorem total_cost (h1 : ∀ x, x * 2.5 = 2.5 * x)
                   (h2 : ∀ y, y * 4 = 4 * y) :
  2.5 * m + 4 * n = 2.5 * m + 4 * n :=
by
  sorry

end total_cost_l648_648103


namespace rectangle_perimeter_l648_648456

def long_side : ℝ := 1
def short_side : ℝ := long_side - (2 / 8)
def perimeter (a b : ℝ) : ℝ := 2 * a + 2 * b

theorem rectangle_perimeter : perimeter long_side short_side = 3.5 := by
  -- Proof steps go here
  sorry

end rectangle_perimeter_l648_648456


namespace star_k_l648_648229

def star (x y : ℝ) : ℝ := 2^x - y

theorem star_k (k : ℝ) : star k (star k k) = k :=
by 
  -- proof will go here, skipped with sorry for now
  sorry

end star_k_l648_648229


namespace find_m_l648_648767

theorem find_m (m : ℤ) (h : (m - 2) * x ^ (m^2 - 2) + 4 * x - 7 = 0) (is_quadratic : (m^2 - 2) = 2) (coeff_nonzero : m ≠ 2) : 
  m = -2 := 
sorry

end find_m_l648_648767


namespace measure_dihedral_angle_l648_648352

noncomputable def dihedral_angle_measure (cube_side_length : ℝ) : ℝ :=
  let d1 := (cube_side_length ^ 2 + cube_side_length ^ 2).sqrt in
  let bd1 := (cube_side_length ^ 2 + d1 ^ 2).sqrt in
  let ae := (cube_side_length * d1 / bd1) in
  let ce := (cube_side_length ^ 2 - ae ^ 2).sqrt in
  let ac := (cube_side_length ^ 2 + cube_side_length ^ 2).sqrt in
  let cos_aec := (ae ^ 2 + ce ^ 2 - ac ^ 2) / (2 * ae * ce) in
  let angle_aec := real.arccos cos_aec in
  180 - angle_aec

theorem measure_dihedral_angle : 
  dihedral_angle_measure 1 = 60 :=
sorry

end measure_dihedral_angle_l648_648352


namespace distance_covered_downstream_l648_648965

noncomputable def speed_boat_still_water : ℝ := 15 -- in kmph
noncomputable def speed_current : ℝ := 3 -- in kmph
noncomputable def time_to_cover_distance : ℝ := 23.998080153587715 -- in seconds

theorem distance_covered_downstream :
  let effective_speed_kmph := speed_boat_still_water + speed_current in
  let effective_speed_mps := (effective_speed_kmph * 1000) / 3600 in
  let distance := effective_speed_mps * time_to_cover_distance in
  distance ≈ 120 := -- using approximately equal ≈ to denote rounding
by
  sorry

end distance_covered_downstream_l648_648965


namespace find_function_expression_and_monotonic_intervals_l648_648303

def f (x : ℝ) : ℝ := x^3 + bx^2 + cx + d

theorem find_function_expression_and_monotonic_intervals:
  (f 0 = 2) ∧ (6 * (-1) - f (-1) + 7 = 0) ∧
  (∀ x, f x = x^3 - 3x^2 - 3x + 2) ∧
  (∀ x, x < 1 - sqrt 2 ∨ x > 1 + sqrt 2 → f' x > 0) ∧
  (∀ x, 1 - sqrt 2 < x ∧ x < 1 + sqrt 2 → f' x < 0) :=
sorry

end find_function_expression_and_monotonic_intervals_l648_648303


namespace petya_purchase_cost_l648_648041

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l648_648041


namespace sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l648_648714

variable {a b : ℝ}

theorem sufficient_condition_for_reciprocal_square :
  (b > a ∧ a > 0) → (1 / a^2 > 1 / b^2) :=
sorry

theorem not_necessary_condition_for_reciprocal_square :
  ¬((1 / a^2 > 1 / b^2) → (b > a ∧ a > 0)) :=
sorry

end sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l648_648714


namespace GreenfieldHighMeanScoreCorrect_l648_648211

noncomputable def GreenfieldHighMeanScore : Prop :=
  ∃ (s n : ℕ) (m_s m_n : ℝ),
  s + n = 120 ∧
  n = 1.75 * s ∧
  120 * 150 = 18000 ∧
  m_s = 2 * m_n ∧
  (s : ℝ) * m_s + (n: ℝ) * m_n = 18000 ∧
  m_s = 220

theorem GreenfieldHighMeanScoreCorrect : GreenfieldHighMeanScore :=
  sorry

end GreenfieldHighMeanScoreCorrect_l648_648211


namespace arithmetic_mean_log_l648_648083

-- Definitions of the given logarithmic values
def a := log (sqrt 3 - sqrt 2)
def b := log (sqrt 3 + sqrt 2)

-- The theorem to be proved
theorem arithmetic_mean_log (a := log (sqrt 3 - sqrt 2)) (b := log (sqrt 3 + sqrt 2)) :
  (a + b) / 2 = 0 :=
by
  sorry

end arithmetic_mean_log_l648_648083


namespace total_cost_of_books_l648_648322

theorem total_cost_of_books
  (C1 : ℝ) (C2 : ℝ)
  (h1 : C1 = 315)
  (h2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2565 :=
by 
  sorry

end total_cost_of_books_l648_648322


namespace find_a_l648_648803

theorem find_a (a : ℝ) (h_pos : 0 < a) 
  (prob : (2 / a) = (1 / 3)) : a = 6 :=
by sorry

end find_a_l648_648803


namespace find_fraction_l648_648862

noncomputable def fraction_of_eighths (N : ℝ) (a b : ℝ) : Prop :=
  (3/8) * N * (a/b) = 24

noncomputable def two_fifty_percent (N : ℝ) : Prop :=
  2.5 * N = 199.99999999999997

theorem find_fraction {N a b : ℝ} (h1 : fraction_of_eighths N a b) (h2 : two_fifty_percent N) :
  a/b = 4/5 :=
sorry

end find_fraction_l648_648862


namespace collinear_points_cube_l648_648994

-- Define a function that counts the sets of three collinear points in the described structure.
def count_collinear_points : Nat :=
  -- Placeholders for the points (vertices, edge midpoints, face centers, center of the cube) and the count logic
  -- The calculation logic will be implemented as the proof
  49

theorem collinear_points_cube : count_collinear_points = 49 :=
  sorry

end collinear_points_cube_l648_648994


namespace proof_problem_l648_648684

open Real

def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, (∀ h : ℝ, 0 < h ∧ h < ε → f(x + h) ≤ f(x)) ∧ (∀ h : ℝ, 0 < h ∧ h < ε → f(x - h) ≤ f(x))

def is_decreasing_function (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f y ≤ f x

theorem proof_problem (a : ℝ) (h₁ : a ≠ 0) :
  ¬ is_extreme_point (λ x, a * x^3) 0 ∧ (a < 0 → is_decreasing_function (λ x, a * x^3)) :=
  sorry

end proof_problem_l648_648684


namespace fraction_of_smart_integers_divisible_by_10_l648_648841

def is_smart_integer (n : ℕ) : Prop := 
  even n ∧ 40 < n ∧ n < 150 ∧ (n.digits 10).sum = 10

def smart_integers : list ℕ :=
  (list.range' 41 109).filter is_smart_integer

def smart_integers_divisible_by_10 : list ℕ :=
  smart_integers.filter (λ n, n % 10 = 0)

theorem fraction_of_smart_integers_divisible_by_10 :
  (smart_integers_divisible_by_10.length : ℚ) / smart_integers.length = 1 / 6 :=
by 
  sorry

end fraction_of_smart_integers_divisible_by_10_l648_648841


namespace prime_N_k_iff_k_eq_2_l648_648264

-- Define the function to generate the number N_k based on k
def N_k (k : ℕ) : ℕ := (10^(2 * k) - 1) / 99

-- Define the main theorem to prove
theorem prime_N_k_iff_k_eq_2 (k : ℕ) : Nat.Prime (N_k k) ↔ k = 2 :=
by
  sorry

end prime_N_k_iff_k_eq_2_l648_648264


namespace volume_of_pyramid_l648_648973

noncomputable def pyramid_volume (s l : ℝ) : ℝ :=
  let base_area := s^2
  let height := Real.sqrt (l^2 - (s^2 / 2))
  (1/3) * base_area * height

theorem volume_of_pyramid
  (s : ℝ) (h₁ : s = 10)
  (l : ℝ) (h₂ : l = 17) :
  pyramid_volume s l = 100 * Real.sqrt 239 / 3 :=
by
  unfold pyramid_volume
  rw [h₁, h₂]
  have h_base_area : 10^2 = 100 := by norm_num
  rw [h_base_area]
  have h_sqrt : Real.sqrt (17^2 - (10^2 / 2)) = Real.sqrt 239 :=
    by norm_num; simp [pow_two, Real.sqrt_sub_self, Real.sqrt]
  rw [h_sqrt]
  norm_num
  sorry

end volume_of_pyramid_l648_648973


namespace triangle_area_l648_648899

noncomputable def inscribed_circle_area : ℝ := 16 * Real.pi

theorem triangle_area (ABC : Triangle) (O : Point) (D : Point) 
  (h1 : ABC.isEquilateral)
  (h2 : O = ABC.inscribedCircleCenter)
  (h3 : ABC.inscribedCircle.area = inscribed_circle_area)
  (h4 : ∠ A O D = 45) 
  (h5 : Triangle.isRightTriangle A O D) :
  ABC.area = 48 * Real.sqrt 3 :=
  sorry

end triangle_area_l648_648899


namespace train_length_l648_648568

-- Definitions based on the conditions
def speed_kmh : ℝ := 30
def conversion_factor : ℝ := 5 / 18
def speed_ms : ℝ := speed_kmh * conversion_factor
def time_s : ℝ := 9

-- Define the length of the train as per the given conditions
def length_of_train : ℝ := speed_ms * time_s

-- The theorem statement to be proved
theorem train_length : length_of_train = 74.97 := by
  simp only [speed_kmh, conversion_factor, speed_ms, time_s, length_of_train]
  sorry

end train_length_l648_648568


namespace perp_planes_from_lines_perp_lines_from_planes_l648_648153

variable (m n : Line)
variable (α β : Plane)

theorem perp_planes_from_lines
  (h₁ : m ⊥ n)
  (h₃ : n ⊥ β) 
  (h₄ : m ⊥ α) :
  α ⊥ β := 
sorry

theorem perp_lines_from_planes
  (h₂ : α ⊥ β)
  (h₃ : n ⊥ β)
  (h₄ : m ⊥ α) :
  m ⊥ n :=
sorry

end perp_planes_from_lines_perp_lines_from_planes_l648_648153


namespace sum_of_primes_1_to_20_l648_648548

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648548


namespace minimum_area_of_triangle_ABC_l648_648283

-- Definition of points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 2)

-- Definition of the circle function
def on_circle (C : ℝ × ℝ) : Prop :=
  let (x, y) := C in
  (x^2 - 2 * x + y^2 = 0)

-- The main theorem
theorem minimum_area_of_triangle_ABC
  (HA : A = (-2, 0))
  (HB : B = (0, 2))
  (HC : ∃ C : ℝ × ℝ, on_circle C) :
  ∃ C : ℝ × ℝ, on_circle C ∧ (area_ABC A B C = 3 - real.sqrt 2) :=
sorry

-- Assuming the area_ABC function calculates the area of triangle ABC
noncomputable def area_ABC (A B C : ℝ × ℝ) : ℝ := sorry

end minimum_area_of_triangle_ABC_l648_648283


namespace total_cookies_l648_648151

-- Define the number of bags and the number of cookies per bag
def bags : ℕ := 37
def cookies_per_bag : ℕ := 19

-- State the theorem
theorem total_cookies : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l648_648151


namespace complex_square_subtract_l648_648431

theorem complex_square_subtract (a b : ℝ) (i : ℂ) (h1 : i^2 = -1)
  (h2 : (1 - i : ℂ)/real.sqrt 2 ^ 2 = a + b * i) : a^2 - b^2 = -1 :=
sorry

end complex_square_subtract_l648_648431


namespace sum_of_primes_1_to_20_l648_648554

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648554


namespace quadratic_roots_real_distinct_l648_648772

theorem quadratic_roots_real_distinct (k : ℝ) :
  (k > (1/2)) ∧ (k ≠ 1) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k-1) * x1^2 + 2 * x1 - 2 = 0) ∧ ((k-1) * x2^2 + 2 * x2 - 2 = 0)) :=
by
  sorry

end quadratic_roots_real_distinct_l648_648772


namespace correct_expression_for_algebraic_notation_l648_648917

constant ab3 : Prop
constant two_third_xy2 : Prop
constant three_pi_m_over_4 : Prop
constant x_plus_3_grams : Prop

axiom ab3_not_algebraic : ¬ ab3
axiom two_third_not_algebraic : ¬ two_third_xy2
axiom three_pi_m_algebraic : three_pi_m_over_4
axiom x_plus_3_not_algebraic : ¬ x_plus_3_grams

theorem correct_expression_for_algebraic_notation :
  three_pi_m_over_4 := 
by 
  exact three_pi_m_algebraic

end correct_expression_for_algebraic_notation_l648_648917


namespace sum_primes_between_1_and_20_l648_648502

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648502


namespace smallest_four_digit_divisible_by_primes_l648_648652

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l648_648652


namespace perimeter_of_quadrilateral_APMQ_l648_648424

/-!
## APMQ Perimeter Problem

Given:
- Altitudes from vertices B and C of the acute-angled triangle ABC are 7 and 9 respectively.
- The median AM equals 8.
- Points P and Q are symmetric to point M with respect to sides AC and AB respectively.

We need to prove that the perimeter of the quadrilateral APMQ is 32.
-/

variables {A B C M P Q : Type}
-- Assuming the conditions
constant (altitude_B : ℝ)
constant (altitude_C : ℝ)
constant (median_AM : ℝ)
constant (symmetric_M_AC : P) -- Symmetric point P with respect to AC
constant (symmetric_M_AB : Q) -- Symmetric point Q with respect to AB

def perimeter_APMQ : ℝ :=
  2 * median_AM + altitude_B + altitude_C 

theorem perimeter_of_quadrilateral_APMQ
  (h1 : altitude_B = 7)
  (h2 : altitude_C = 9)
  (h3 : median_AM = 8)
  (h4 : P = symmetric_M_AC)
  (h5 : Q = symmetric_M_AB) :
  perimeter_APMQ = 32 :=
by
  sorry

end perimeter_of_quadrilateral_APMQ_l648_648424


namespace sum_of_primes_between_1_and_20_l648_648544

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648544


namespace next_term_geometric_sequence_l648_648485

theorem next_term_geometric_sequence (y : ℝ) (h0 : y ≠ 0) :
  let r := 3 * y in
  let term := 81 * y^3 in
  term * r = 243 * y^4 :=
by
  let r := 3 * y
  let term := 81 * y^3
  have h : term * r = 243 * y^4 := sorry
  exact h

end next_term_geometric_sequence_l648_648485


namespace scientific_notation_of_twenty_eight_billion_l648_648606

-- Definitions based on given conditions
def billion: ℕ := 1000000000 -- One billion
def twenty_eight_billion: ℕ := 28 * billion -- 28 billion

-- Proof statement equivalence to the mathematical problem
theorem scientific_notation_of_twenty_eight_billion : twenty_eight_billion = 2.8 * 10^9 := 
sorry

end scientific_notation_of_twenty_eight_billion_l648_648606


namespace petya_purchase_cost_l648_648039

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l648_648039


namespace cylinder_radius_range_l648_648778

theorem cylinder_radius_range :
  (V : ℝ) → (h : ℝ) → (r : ℝ) →
  V = 20 * Real.pi →
  h = 2 →
  (V = Real.pi * r^2 * h) →
  3 < r ∧ r < 4 :=
by
  -- Placeholder for the proof
  intro V h r hV hh hV_eq
  sorry

end cylinder_radius_range_l648_648778


namespace average_additional_minutes_l648_648210

theorem average_additional_minutes (differences : List ℤ) (h_length : differences.length = 7) :
  List.sum differences / differences.length = 8 :=
by
  have h_diff_values : differences = [12, -15, 25, 10, -5, 15, 15] := by rfl
  have h_sum : List.sum differences = 57 := by
    calc
      List.sum [12, -15, 25, 10, -5, 15, 15]
      = 12 - 15 + 25 + 10 - 5 + 15 + 15 : by rfl
      ... = 57 : by norm_num
  rw [h_diff_values] at h_length,
  have h_len : (7 : ℤ) = (differences.length : ℤ) := by
    norm_cast
    exact h_length
  rw [h_sum, h_len]
  norm_num

end average_additional_minutes_l648_648210


namespace total_cost_is_eight_x_l648_648053

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l648_648053


namespace total_cost_shorts_tshirt_boots_shinguards_l648_648022

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l648_648022


namespace total_cost_is_eight_times_l648_648048

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l648_648048


namespace tan_alpha_two_alpha_minus_beta_l648_648716

variable (α β : ℝ)

-- Conditions
noncomputable def α_range := α ∈ set.Ioo 0 (Real.pi / 4)
noncomputable def β_range := β ∈ set.Ioo 0 Real.pi
noncomputable def tan_alpha_minus_beta := Real.tan (α - β) = 1 / 2
noncomputable def tan_beta := Real.tan β = - (1 / 7)

-- Statements to prove
theorem tan_alpha (h1: α_range) (h2: β_range) (h3: tan_alpha_minus_beta) (h4: tan_beta) : 
  Real.tan α = 1 / 3 := 
sorry

theorem two_alpha_minus_beta (h1: α_range) (h2: β_range) (h3: tan_alpha_minus_beta) (h4: tan_beta) : 
  2 * α - β = - (3 * Real.pi) / 4 := 
sorry

end tan_alpha_two_alpha_minus_beta_l648_648716


namespace sum_of_primes_1_to_20_l648_648557

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648557


namespace birdseed_mixture_l648_648567

theorem birdseed_mixture (x : ℝ) (h1 : 0.40 * x + 0.65 * (100 - x) = 50) : x = 60 :=
by
  sorry

end birdseed_mixture_l648_648567


namespace dave_spent_102_dollars_l648_648228

noncomputable def total_cost (books_animals books_space books_trains cost_per_book : ℕ) : ℕ :=
  (books_animals + books_space + books_trains) * cost_per_book

theorem dave_spent_102_dollars :
  total_cost 8 6 3 6 = 102 := by
  sorry

end dave_spent_102_dollars_l648_648228


namespace total_cost_is_eight_times_short_cost_l648_648030

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l648_648030


namespace sum_of_primes_1_to_20_l648_648547

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648547


namespace minimum_disks_for_files_l648_648241

def file_sizes : List Float := [1.0, 1.0, 1.0, 1.0] ++ List.repeat 0.9 16 ++ List.repeat 0.5 20

def disk_capacity : Float := 2.0

theorem minimum_disks_for_files :
  let total_files := 40
  4 = List.count (· = 1.0) file_sizes ∧ 
  16 = List.count (· = 0.9) file_sizes ∧ 
  20 = List.count (· = 0.5) file_sizes - 4 ∧ 
  disk_capacity = 2.0 ∧ 
  no_splitting : ¬ ∃ f1 f2, f1 + f2 > disk_capacity → f1 ∈ file_sizes ∧ f2 ∈ file_sizes :=
  List.foldr (λ x acc, acc + x) 0 (List.map (λ x, ceiling (x / disk_capacity)) file_sizes) = 16 :=
by
  sorry

end minimum_disks_for_files_l648_648241


namespace Madeline_hours_left_over_l648_648004

theorem Madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  total_hours_per_week - total_busy_hours = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  have : total_hours_per_week - total_busy_hours = 168 - 122 := by rfl
  have : 168 - 122 = 46 := by rfl
  exact this

end Madeline_hours_left_over_l648_648004


namespace det_transformed_matrix_l648_648326

variables {p q r s : ℝ} -- Defining the variables over the real numbers

-- Defining the first determinant condition as an axiom
axiom det_initial_matrix : (p * s - q * r) = 10

-- Stating the theorem to be proved
theorem det_transformed_matrix : 
  (p + 2 * r) * s - (q + 2 * s) * r = 10 :=
by
  sorry -- Placeholder for the actual proof

end det_transformed_matrix_l648_648326


namespace triangle_area_ratio_l648_648356

theorem triangle_area_ratio (p q r : ℚ) (hpqrsum : p + q + r = 1) (hpqrsum_sq : p^2 + q^2 + r^2 = 1 / 3) : 
  (let ratio := 1 / 3 in 
  let m := 1 in 
  let n := 3 in 
  m + n = 4) := 
begin
  sorry
end

end triangle_area_ratio_l648_648356


namespace total_cost_is_eight_times_short_cost_l648_648027

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l648_648027


namespace sum_primes_between_1_and_20_l648_648524

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648524


namespace max_distance_from_point_to_circle_l648_648438

theorem max_distance_from_point_to_circle 
    (A : EuclideanSpace ℝ (Fin 2)) (xA yA : ℝ)
    (C : EuclideanSpace ℝ (Fin 2)) (xC yC r : ℝ) 
    (eqn_C : ∀ x y : ℝ, x^2 + (y - yC)^2 = r^2) :
    A = ![2, 1] ∧ C = ![0, 1] ∧ r = 1 → 
    ∃ d : ℝ, d = 3 := 
by
  have h1 : ⁅d₁, d₂⁆ :=
      dist (![2, 1] : EuclideanSpace ℝ (Fin 2)) (![0, 1] : EuclideanSpace ℝ (Fin 2))
  sorry

end max_distance_from_point_to_circle_l648_648438


namespace hyperbola_eccentricity_l648_648307

theorem hyperbola_eccentricity {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0)
    (h₂ : ∀ x y : ℝ, (x, y) ∈ (λ x y, x^2 / a^2 - y^2 / b^2 = 1)) 
    (h₃ : ∀ F : ℝ × ℝ, F = (c, 0) ∧ c^2 = a^2 + b^2)
    (h₄ : ∀ Q : ℝ × ℝ, (bx + ay = 0) ∧ is_perpendicular (Q, (F, asymptote))) 
    (h₅ : ∀ M N : ℝ × ℝ, 
            M ∈ (left_branch_of_hyperbola) ∧ 
            N ∈ (right_branch_of_hyperbola) ∧ 
            ((F, Q) intersects (line_passing_through M N)) ∧ 
            |MQ| = 3 * |QN|) :
    eccentricity (h₂) = √5 := 
sorry

end hyperbola_eccentricity_l648_648307


namespace points_contained_in_circle_l648_648343

noncomputable def points_in_square (points : set (ℝ × ℝ)) : Prop :=
  ∃ x y, (x, y) ∈ points ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

theorem points_contained_in_circle (points : set (ℝ × ℝ)) : 
  points_in_square points ∧ (|points| = 51) → 
  ∃ (c : ℝ × ℝ) (r : ℝ), r = (1 / 7) ∧ ∃ p1 p2 p3 ∈ points, dist c p1 ≤ r ∧ dist c p2 ≤ r ∧ dist c p3 ≤ r :=
sorry

end points_contained_in_circle_l648_648343


namespace sequence_contains_prime_l648_648253

-- Define the conditions for being square-free and relatively prime
def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Statement of the problem
theorem sequence_contains_prime :
  ∀ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ 14 → 2 ≤ a i ∧ a i ≤ 1995 ∧ is_square_free (a i)) →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 14 → are_relatively_prime (a i) (a j)) →
  ∃ i, 1 ≤ i ∧ i ≤ 14 ∧ is_prime (a i) :=
sorry

end sequence_contains_prime_l648_648253


namespace teacher_B_works_in_Baoji_and_teaches_C_l648_648865

noncomputable theory

-- Definitions of teachers, cities, and subjects
inductive Teacher
| A | B | C

inductive City
| Yanan | Xianyang | Baoji

inductive Subject
| A | B | C

-- Conditions as given in the problem
def condition1 : Prop :=
  ∀ (t : Teacher), (t = Teacher.A → ∀ (c : City), c ≠ City.Yanan) ∧
                  (t = Teacher.B → ∀ (c : City), c ≠ City.Xianyang)

def condition2 : Prop :=
  ∀ (t : Teacher) (c : City), c = City.Yanan → (t = Teacher.A ∨ t = Teacher.B)

def condition3 : Prop :=
  ∀ (t : Teacher) (c : City), c = City.Xianyang → t = Teacher.A

def condition4 : Prop :=
  ∀ t : Teacher, t = Teacher.B → ∀ s : Subject, s ≠ Subject.B

-- We want to demonstrate that Teacher B works in Baoji and teaches subject C
def problem : Prop :=
  ∀ t : Teacher, t = Teacher.B → ∃ (c : City) (s : Subject), c = City.Baoji ∧ s = Subject.C

-- Main theorem statement
theorem teacher_B_works_in_Baoji_and_teaches_C :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 → problem :=
by 
  intros h t hB,
  sorry

end teacher_B_works_in_Baoji_and_teaches_C_l648_648865


namespace simplify_expression_l648_648857

theorem simplify_expression (α : ℝ) :
  (sin (π / 2 + α) * sin (π + α) * tan (3 * π + α)) / (cos (3 * π / 2 + α) * sin (-α)) = 1 := 
by
  sorry

end simplify_expression_l648_648857


namespace find_side_a_of_triangle_l648_648358

noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ

theorem find_side_a_of_triangle
  (c : ℝ) (A : ℝ) (C : ℝ)
  (h1 : c = Real.sqrt 3)
  (h2 : A = Real.pi / 4)
  (h3 : C = Real.pi / 3) :
  ∃ (a : ℝ), a = Real.sqrt 2 :=
by
  -- Law of Sines application
  have h4 : sin A ≠ 0 := Real.sin_ne_zero_of_ne_zero (by linarith [h2])
  have h5 : sin C ≠ 0 := Real.sin_ne_zero_of_ne_zero (by linarith [h3])
  let a := (c * sin A) / sin C
  use a
  have h6 : a = Real.sqrt 2 := by
    -- Substitute the given values into the expression and simplify
    rw [h1, h2, h3]
    simp [sin, Real.sin_pi_div_four, Real.sin_pi_div_three]
    field_simp
  exact h6

end find_side_a_of_triangle_l648_648358


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l648_648675

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l648_648675


namespace gcd_lcm_sum_l648_648910

theorem gcd_lcm_sum (GCD : ℕ → ℕ → ℕ) (LCM : ℕ → ℕ → ℕ) (gcd_42_98 : GCD 42 98 = 14) (lcm_60_15 : LCM 60 15 = 60) : 
  GCD 42 98 + LCM 60 15 = 74 := 
by
  rw [gcd_42_98, lcm_60_15]
  exact rfl

end gcd_lcm_sum_l648_648910


namespace find_side_length_of_equilateral_triangle_l648_648455

noncomputable def equilateral_triangle_side_length (a b c : ℝ) (PA PB PC : ℝ) : ℝ :=
  if PA = 3 ∧ PB = 4 ∧ PC = 5 then √(25 + 12 * Real.sqrt 3) else 0

theorem find_side_length_of_equilateral_triangle (d PA PB PC : ℝ) :
  PA = 3 ∧ PB = 4 ∧ PC = 5 → d = √(25 + 12 * Real.sqrt 3) :=
by
  intros hPA hPB hPC,
  sorry

end find_side_length_of_equilateral_triangle_l648_648455


namespace arithmetic_sequence_sum_l648_648706

noncomputable def integral_val : ℝ :=
  ∫ x in 0..2, |1 - x^2|

-- Define the arithmetic sequence and the key property
axiom arithmetic_sequence (a : ℕ → ℝ) : Prop
axiom key_property (a : ℕ → ℝ) : a 5 + a 7 = integral_val

-- Lean 4 theorem statement
theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hk : key_property a) :
  a 4 + a 6 + a 8 = 3 := sorry

end arithmetic_sequence_sum_l648_648706


namespace max_n_Tn_lt_2019_l648_648724

def a (n : ℕ) : ℕ := 2 * n - 1
def b (n : ℕ) : ℕ := 2 ^ (n - 1)
def c (n : ℕ) : ℕ := a (b n)
def T (n : ℕ) : ℕ := (Finset.range n).sum (λ k, c (k + 1))

theorem max_n_Tn_lt_2019 : ∀ n, T n < 2019 → n ≤ 9 :=
by 
  sorry

end max_n_Tn_lt_2019_l648_648724


namespace share_of_B_l648_648991

noncomputable def B_share (B_investment A_investment C_investment D_investment total_profit : ℝ) : ℝ :=
  (B_investment / (A_investment + B_investment + C_investment + D_investment)) * total_profit

theorem share_of_B (B_investment total_profit : ℝ) (hA : A_investment = 3 * B_investment) 
  (hC : C_investment = (3 / 2) * B_investment) 
  (hD : D_investment = (3 / 2) * B_investment) 
  (h_profit : total_profit = 19900) :
  B_share B_investment A_investment C_investment D_investment total_profit = 2842.86 :=
by
  rw [B_share, hA, hC, hD, h_profit]
  sorry

end share_of_B_l648_648991


namespace sum_primes_upto_20_l648_648514

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648514


namespace num_assignments_l648_648616

def students := { "A", "B", "C", "D" }
def classes := { "A", "B", "C", "D" }

def valid_assignments := { f : students → classes // ∀ s, f s ≠ s }

theorem num_assignments : (set.finite valid_assignments) ∧ (card valid_assignments = 18) := 
sorry

end num_assignments_l648_648616


namespace distance_from_point_to_line_l648_648353

def polarToCartesian (r θ : ℝ) : (ℝ × ℝ) :=
  (r * Real.cos θ, r * Real.sin θ)

def distancePointToLine (p : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  Real.abs (y_line - p.2)

noncomputable def polar_point : ℝ × ℝ := polarToCartesian 2 (Real.pi / 6)

theorem distance_from_point_to_line :
  distancePointToLine polar_point 2 = 1 := by
  sorry

end distance_from_point_to_line_l648_648353


namespace area_between_parabola_and_line_l648_648944

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..1, (2 * x - 2 * x^2)

theorem area_between_parabola_and_line : enclosed_area = 1 / 3 := 
by
  -- proof goes here
  sorry

end area_between_parabola_and_line_l648_648944


namespace area_of_triangle_ABC_l648_648799

noncomputable def area_of_isosceles_right_triangle (AB BC AC : ℝ) (B : Angle) (isosceles_right: Bool) : ℝ :=
  if isosceles_right ∧ B = 90 ∧ AC = 12 then 36 else 0

theorem area_of_triangle_ABC :
  let AB := 6 * Real.sqrt 2
  let BC := 6 * Real.sqrt 2
  let AC := 12
  let B : Angle := 90
  let isosceles_right := true
  area_of_isosceles_right_triangle AB BC AC B isosceles_right = 36 := 
by 
  sorry

end area_of_triangle_ABC_l648_648799


namespace lighting_candles_correct_l648_648468

noncomputable def time_to_light_candles (initial_length : ℝ) : ℝ :=
  let burn_rate_1 := initial_length / 300
  let burn_rate_2 := initial_length / 240
  let t := (5 * 60 + 43) - (5 * 60) -- 11:17 AM is 342.857 minutes before 5 PM
  if ((initial_length - burn_rate_2 * t) = 3 * (initial_length - burn_rate_1 * t)) then 11 + 17 / 60 else 0 -- Check if the condition is met

theorem lighting_candles_correct :
  ∀ (initial_length : ℝ), time_to_light_candles initial_length = 11 + 17 / 60 :=
by
  intros initial_length
  sorry  -- Proof goes here

end lighting_candles_correct_l648_648468


namespace not_island_native_l648_648185

inductive Native
| Knight
| Liar

def speaks (n : Native) : String :=
match n with
| Native.Knight => "I am a knight"
| Native.Liar => "I am a knight"

theorem not_island_native (s : String) (hs : s = "I am a liar") : ∀ n : Native, speaks n ≠ s :=
by
  intro n
  cases n
  case Knight =>
    -- A knight would say "I am a knight", hence cannot say "I am a liar"
    rw [hs] at hs
    contradiction
  case Liar =>
    -- A liar would also say "I am a knight", hence cannot say "I am a liar"
    rw [hs] at hs
    contradiction

end not_island_native_l648_648185


namespace value_of_b_minus_a_div_d_l648_648717

-- Given conditions
variable (a b c d e : ℤ)
variable (h1 : ∀ a b c : ℤ, -1, a, b, -4 forms an arithmetic sequence)
variable (h2 : ∀ a b c d: ℤ, -1, c, d, e, -4 forms a geometric sequence)

-- The main theorem to prove
theorem value_of_b_minus_a_div_d (h1 : b = a + m ∧ -4 = -1 + 3 * m → m = -1 → b - a = -1)
                                 (h2 : -4 = -1 * q^4 ∧ q^2 = 2 → d = -1 * q^2 → d = -2)
                                 : (b - a) / d = 1 / 2 := 
by sorry

end value_of_b_minus_a_div_d_l648_648717


namespace sin_squared_equiv_cosine_l648_648633

theorem sin_squared_equiv_cosine :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = (Real.sqrt 2) / 2 :=
by sorry

end sin_squared_equiv_cosine_l648_648633


namespace intersection_empty_set_l648_648313

def M : Set ℝ := { y | ∃ x, x > 0 ∧ y = 2^x }
def N : Set ℝ := { y | ∃ x, y = Real.sqrt (2*x - x^2) }

theorem intersection_empty_set :
  M ∩ N = ∅ :=
by
  sorry

end intersection_empty_set_l648_648313


namespace max_clocks_is_11_l648_648787

-- Define the problem context
noncomputable def max_num_clocks (hours minutes : List ℕ) : ℕ :=
  if hours.length = minutes.length 
    ∧ hours.length ≥ 2 
    ∧ hours.length ≤ 12 
    ∧ hours.nodup 
    ∧ minutes.nodup 
    ∧ (all h in hours, 1 ≤ h ∧ h ≤ 12) 
    ∧ (all m in minutes, 0 ≤ m ∧ m < 60) 
    ∧ (sum hours + sum minutes).pred = sum hours.pred + sum minutes.pred
    then hours.length
    else 0

/-- Given the conditions on the clocks in the workshop, prove the maximum number of clocks is 11 --/
theorem max_clocks_is_11 : ∀ (hours minutes : List ℕ), 
  (hours.length = minutes.length 
  ∧ hours.length ≥ 2 
  ∧ hours.length ≤ 12 
  ∧ hours.nodup 
  ∧ minutes.nodup 
  ∧ (all h in hours, 1 ≤ h ∧ h ≤ 12) 
  ∧ (all m in minutes, 0 ≤ m ∧ m < 60) 
  ∧ (sum hours + sum minutes).pred = sum hours.pred + sum minutes.pred) → 
  hours.length ≤ 11 := 
by 
  sorry

end max_clocks_is_11_l648_648787


namespace quadratic_distinct_roots_k_range_l648_648773

theorem quadratic_distinct_roots_k_range (k : ℝ) :
  (k - 1) * x^2 + 2 * x - 2 = 0 ∧ 
  ∀ Δ, Δ = 2^2 - 4*(k-1)*(-2) ∧ Δ > 0 ∧ (k ≠ 1) ↔ k > 1/2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_distinct_roots_k_range_l648_648773


namespace water_jugs_problem_l648_648570

-- Definitions based on the conditions
variables (m n : ℕ) (relatively_prime_m_n : Nat.gcd m n = 1)
variables (k : ℕ) (hk : 1 ≤ k ∧ k ≤ m + n)

-- Statement of the theorem
theorem water_jugs_problem : 
    ∃ (x y z : ℕ), 
    (x = m ∨ x = n ∨ x = m + n) ∧ 
    (y = m ∨ y = n ∨ y = m + n) ∧ 
    (z = m ∨ z = n ∨ z = m + n) ∧ 
    (x ≤ m + n) ∧ 
    (y ≤ m + n) ∧ 
    (z ≤ m + n) ∧ 
    x + y + z = m + n ∧ 
    (x = k ∨ y = k ∨ z = k) :=
sorry

end water_jugs_problem_l648_648570


namespace Q_sum_of_squares_and_one_zero_l648_648309

-- Define Q
def Q (x₁ x₂ x₃ x₄ : ℝ) : ℝ := 
  4 * (x₁^2 + x₂^2 + x₃^2 + x₄^2) - (x₁ + x₂ + x₃ + x₄)^2

-- Prove the statements
theorem Q_sum_of_squares_and_one_zero :
  ∃ (P₁ P₂ P₃ P₄ : ℝ → ℝ → ℝ → ℝ), 
    (Q = λ x₁ x₂ x₃ x₄, P₁ x₁ x₂ x₃ x₄ ^ 2 + P₂ x₁ x₂ x₃ x₄ ^ 2 + P₃ x₁ x₂ x₃ x₄ ^ 2 + P₄ x₁ x₂ x₃ x₄ ^ 2) ∧
    (∀ P₁ P₂ P₃ P₄, Q = λ x₁ x₂ x₃ x₄, P₁ x₁ x₂ x₃ x₄ ^ 2 + P₂ x₁ x₂ x₃ x₄ ^ 2 + P₃ x₁ x₂ x₃ x₄ ^ 2 + P₄ x₁ x₂ x₃ x₄ ^ 2 → 
      P₁ = λ _ _ _ _, 0 ∨ P₂ = λ _ _ _ _, 0 ∨ P₃ = λ _ _ _ _, 0 ∨ P₄ = λ _ _ _ _, 0)
:= sorry

end Q_sum_of_squares_and_one_zero_l648_648309


namespace scrap_cookie_radius_correct_l648_648590

-- Define the given problem
def large_cookie_radius : ℝ := 6
def small_cookie_radius : ℝ := 1.5
def num_small_cookies : ℕ := 9

-- Define the areas
def large_cookie_area : ℝ := π * large_cookie_radius^2
def small_cookie_area (r : ℝ) : ℝ := π * r^2
def total_small_cookie_area : ℝ := num_small_cookies * small_cookie_area small_cookie_radius

-- Leftover dough area
def scrap_area : ℝ := large_cookie_area - total_small_cookie_area

-- Scrap cookie radius
def scrap_cookie_radius : ℝ := Real.sqrt (scrap_area / π)

-- Prove that the scrap cookie radius is approximately 3.97 inches
theorem scrap_cookie_radius_correct : scrap_cookie_radius ≈ 3.97 := by
  sorry

end scrap_cookie_radius_correct_l648_648590


namespace unique_two_digit_number_l648_648320

-- Definition of the problem in Lean
def is_valid_number (n : ℕ) : Prop :=
  n % 4 = 1 ∧ n % 17 = 1 ∧ 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 69 :=
by
  sorry

end unique_two_digit_number_l648_648320


namespace sieve_of_eratosthenes_time_complexity_l648_648417

noncomputable def sieve_time_complexity (n : ℕ) : BigO :=
  BigO (n * (log n)^2)

theorem sieve_of_eratosthenes_time_complexity (n : ℕ) :
  time_complexity (sieve_of_eratosthenes n) = sieve_time_complexity n :=
sorry

end sieve_of_eratosthenes_time_complexity_l648_648417


namespace second_printer_cost_l648_648586

theorem second_printer_cost (p1_cost : ℕ) (num_units : ℕ) (total_spent : ℕ) (x : ℕ) 
  (h1 : p1_cost = 375) 
  (h2 : num_units = 7) 
  (h3 : total_spent = p1_cost * num_units) 
  (h4 : total_spent = x * num_units) : 
  x = 375 := 
sorry

end second_printer_cost_l648_648586


namespace stella_annual_income_after_tax_l648_648864

-- Definitions of the conditions
def base_salary_per_month : ℝ := 3500
def bonuses : List ℝ := [1200, 600, 1500, 900, 1200]
def months_paid : ℝ := 10
def tax_rate : ℝ := 0.05

-- Calculations derived from the conditions
def total_base_salary : ℝ := base_salary_per_month * months_paid
def total_bonuses : ℝ := bonuses.sum
def total_income_before_tax : ℝ := total_base_salary + total_bonuses
def tax_deduction : ℝ := total_income_before_tax * tax_rate
def annual_income_after_tax : ℝ := total_income_before_tax - tax_deduction

-- The theorem to prove
theorem stella_annual_income_after_tax :
  annual_income_after_tax = 38380 := by
  sorry

end stella_annual_income_after_tax_l648_648864


namespace rotate_parabola_180deg_l648_648066

theorem rotate_parabola_180deg (x y : ℝ) :
  (∀ x, y = 2 * x^2 - 12 * x + 16) →
  (∀ x, y = -2 * x^2 + 12 * x - 20) :=
sorry

end rotate_parabola_180deg_l648_648066


namespace solve_equation_l648_648112

theorem solve_equation (x : ℝ) : 4 * (x - 1) ^ 2 = 9 ↔ x = 5 / 2 ∨ x = -1 / 2 := 
by 
  sorry

end solve_equation_l648_648112


namespace part1_part2_l648_648702

-- The quadratic equation of interest
def quadratic_eq (k x : ℝ) : ℝ :=
  x^2 + (2 * k - 1) * x + k^2 - k

-- Part 1: Proof that the equation has two distinct real roots
theorem part1 (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic_eq k x1 = 0 ∧ quadratic_eq k x2 = 0) := 
  sorry

-- Part 2: Given x = 2 is a root, prove the value of the expression
theorem part2 (k : ℝ) (h : quadratic_eq k 2 = 0) : -2 * k^2 - 6 * k - 5 = -1 :=
  sorry

end part1_part2_l648_648702


namespace sum_of_primes_1_to_20_l648_648546

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648546


namespace area_increase_37point5_percent_l648_648148

theorem area_increase_37point5_percent (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A := L * B,
      L' := 1.10 * L,
      B' := 1.25 * B,
      A' := L' * B'
  in ((A' / A - 1) * 100) = 37.5 :=
by
  sorry

end area_increase_37point5_percent_l648_648148


namespace infinite_lines_intersecting_l648_648779

open Set

structure Line (α : Type) :=
(points : Set α)
(is_line : ∀ (p₁ p₂ : α), p₁ ∈ points → p₂ ∈ points → ∃ l, ∀ p ∈ l, p ∈ points)

axiom skew_lines (α : Type) [MetricSpace α] (a b c : Line α) :
  ¬ ∃ p, p ∈ a.points ∧ p ∈ b.points ∧ 
  ¬ ∃ q, q ∈ b.points ∧ q ∈ c.points ∧ 
  ¬ ∃ r, r ∈ c.points ∧ r ∈ a.points

theorem infinite_lines_intersecting (α : Type) [MetricSpace α] (a b c : Line α) (h : skew_lines α a b c) :
  ∃ (l : Set (Line α)), Infinite l ∧ ∀ (l' : Line α) (H : l' ∈ l), ∃ p₁ p₂ p₃, p₁ ∈ a.points ∧ p₂ ∈ b.points ∧ p₃ ∈ c.points ∧ p₁ = p₂ ∧ p₂ = p₃ :=
by
  sorry

end infinite_lines_intersecting_l648_648779


namespace max_value_is_one_eighth_l648_648682

noncomputable def find_max_value (a b c : ℝ) : ℝ :=
  a^2 * b^2 * c^2 * (a + b + c) / ((a + b)^3 * (b + c)^3)

theorem max_value_is_one_eighth (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  find_max_value a b c ≤ 1 / 8 :=
by
  sorry

end max_value_is_one_eighth_l648_648682


namespace squared_sum_l648_648780

theorem squared_sum (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 :=
by
  sorry

end squared_sum_l648_648780


namespace basketball_campers_l648_648196

theorem basketball_campers (total_campers soccer_campers football_campers : ℕ)
  (h_total : total_campers = 88)
  (h_soccer : soccer_campers = 32)
  (h_football : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 :=
by
  sorry

end basketball_campers_l648_648196


namespace students_exceed_guinea_pigs_l648_648638

theorem students_exceed_guinea_pigs :
  ∀ (students_per_classroom guinea_pigs_per_classroom classrooms : ℕ),
    students_per_classroom = 20 →
    guinea_pigs_per_classroom = 3 →
    classrooms = 5 →
    (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 85 :=
by
  intros students_per_classroom guinea_pigs_per_classroom classrooms
  intros h_students h_guinea_pigs h_classrooms
  rw [h_students, h_guinea_pigs, h_classrooms]
  calc
    (20 * 5) - (3 * 5) = 100 - 15 : by norm_num
                    ... = 85     : by norm_num

end students_exceed_guinea_pigs_l648_648638


namespace no_partition_k_plus_2020_partition_k_squared_plus_2020_l648_648362

theorem no_partition_k_plus_2020 : 
  ¬ ∃ (A : ℕ → set ℕ), (∀ n m : ℕ, n ≠ m → disjoint (A n) (A m)) ∧
  (∀ k : ℕ, ∑ x in (A k), x = k + 2020) := 
  sorry

theorem partition_k_squared_plus_2020 : 
  ∃ (A : ℕ → set ℕ), (∀ n m : ℕ, n ≠ m → disjoint (A n) (A m)) ∧
  (∀ k : ℕ, ∑ x in (A k), x = k^2 + 2020) := 
  sorry

end no_partition_k_plus_2020_partition_k_squared_plus_2020_l648_648362


namespace geometric_bodies_that_appear_as_circle_l648_648328

theorem geometric_bodies_that_appear_as_circle
  (solid : Type)
  (is_circle_projection : solid → Type → Prop)
  (cone : Type)
  (cylinder : Type)
  (sphere : Type)
  (h_cone : is_circle_projection cone solid)
  (h_cylinder : is_circle_projection cylinder solid)
  (h_sphere : is_circle_projection sphere solid) :
  ∃ b : solid, b = cone ∨ b = cylinder ∨ b = sphere :=
sorry

end geometric_bodies_that_appear_as_circle_l648_648328


namespace constant_term_in_expansion_l648_648333

theorem constant_term_in_expansion (n : ℕ) (h : 2^n = 64) : 
  let constant_term := (-2)^4 * Nat.choose 6 4
  constant_term = 240 :=
by
  -- proof placeholder
  sorry

end constant_term_in_expansion_l648_648333


namespace convex_polyhedron_similar_faces_two_pairs_equal_l648_648169

theorem convex_polyhedron_similar_faces_two_pairs_equal {P : Type*} [convex_polyhedron P] 
  (h_similar_faces : ∀ f1 f2 : face P, similar f1 f2) : 
  ∃ (f1 f2 f3 f4 : face P), f1 ≠ f2 ∧ f3 ≠ f4 ∧ f1 = f2 ∧ f3 = f4 :=
by 
  sorry

end convex_polyhedron_similar_faces_two_pairs_equal_l648_648169


namespace angle_bisector_D_l648_648791

variables (A B C D : Point) (E : Point)
variable [convex_quad : ConvexQuadrilateral A B C D]
variables (AD AB CD : Real)
variable (angle_bisector_A_passes_through_E : AngleBisector A E)

-- Condition statements
axiom condition1 : AD = AB + CD
axiom condition2 : angle_bisector_A_passes_through_E

-- The theorem to be proven
theorem angle_bisector_D (h : AD = AB + CD) (h_angle : AngleBisector A E) : AngleBisector D E :=
sorry

end angle_bisector_D_l648_648791


namespace probability_at_least_one_heart_l648_648602

theorem probability_at_least_one_heart (total_cards hearts : ℕ) 
  (top_card_positions : Π n : ℕ, n = 3) 
  (non_hearts_cards : Π n : ℕ, n = total_cards - hearts) 
  (h_total_cards : total_cards = 52) (h_hearts : hearts = 13) 
  : (1 - ((39 * 38 * 37 : ℚ) / (52 * 51 * 50))) = (325 / 425) := 
by {
  sorry
}

end probability_at_least_one_heart_l648_648602


namespace polynomial_division_quotient_l648_648252

theorem polynomial_division_quotient : 
  (Polynomial.div' (8 * X ^ 3 - 2 * X ^ 2 + 4 * X - 7 : Polynomial ℤ) (X - 1)).1 = 8 * X ^ 2 + 6 * X + 10 :=
by
  sorry

end polynomial_division_quotient_l648_648252


namespace minimize_travel_time_l648_648694

-- Define the speeds for road and field travel
def road_speed := 4 -- km/h
def field_speed := 3 -- km/h

-- Define the total distance to be traveled
variable (total_distance : ℝ)

-- Define the function for travel time on the road
def road_time (distance_on_road : ℝ) : ℝ :=
  distance_on_road / road_speed

-- Define the function for travel time on the field
def field_time (distance_on_field : ℝ) : ℝ :=
  distance_on_field / field_speed

-- Define the total travel time as the sum of road travel time and field travel time
def total_travel_time (distance_on_road distance_on_field : ℝ) : ℝ :=
  road_time distance_on_road + field_time distance_on_field

-- Problem statement: given the speed on road and field, determine the minimal travel time
theorem minimize_travel_time (d_road d_field : ℝ) :
  (d_road + d_field = total_distance) → 
  ∃ optimal_path : ℝ × ℝ, 
    optimal_path = (d_road, d_field) ∧ 
    ∀ path : ℝ × ℝ, 
      total_travel_time path.1 path.2 ≥ total_travel_time optimal_path.1 optimal_path.2 :=
sorry

end minimize_travel_time_l648_648694


namespace next_term_geometric_sequence_l648_648477

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l648_648477


namespace cost_of_purchase_l648_648034

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l648_648034


namespace sequence_an_l648_648392

theorem sequence_an (a : ℕ → ℚ) (h₁ : a 1 = 1/2) (h₂ : ∀ n : ℕ, 1 ≤ n → ∑ i in finset.range n.succ, a (i+1) = n^2) :
  ∀ n ≥ 1, a n = 1 / (n * (n + 1)) :=
by sorry

end sequence_an_l648_648392


namespace median_implies_equal_areas_l648_648816

noncomputable def part_of_triangle (M A B C : Point) : Real := sorry

theorem median_implies_equal_areas (M A B C At Bt Ct : Point) :
  let S := part_of_triangle;
  S M A Ct + S M Bt A + S M C At = S M At C + S M Bt A + S M Ct B →
  (is_median A A1 B C ∨ is_median B B1 A C ∨ is_median C C1 A B) :=
sorry

end median_implies_equal_areas_l648_648816


namespace logarithm_base_l648_648885

theorem logarithm_base (x : ℝ) (b : ℝ) : (9 : ℝ)^(x + 5) = (16 : ℝ)^x → b = 16 / 9 → x = Real.log 9^5 / Real.log b := by sorry

end logarithm_base_l648_648885


namespace max_value_expression_l648_648290

theorem max_value_expression (x y : ℝ) (h : (x^2 / 4) + y^2 = 1) :
  ∃ (M : ℝ), M = (1 + Real.sqrt 2) / 2 ∧
    ∀ (x y), ((x^2 / 4) + y^2 = 1) → 
      (xy / (x + 2y - 2)) ≤ M := 
sorry

end max_value_expression_l648_648290


namespace trig_identity_orthocenter_coordinates_l648_648819

variable {R α β γ : ℝ}

noncomputable def orthocenter (α β γ : ℝ) (R : ℝ) : ℝ × ℝ :=
  (R * (Real.cos α + Real.cos β + Real.cos γ), R * (Real.sin α + Real.sin β + Real.sin γ))

theorem trig_identity (α β γ : ℝ) :
  Real.tan ((β + γ) / 2) * (Real.cos β + Real.cos γ) - Real.tan ((γ + α) / 2) * (Real.cos γ + Real.cos α) = Real.sin β - Real.sin α :=
begin
  sorry
end

theorem orthocenter_coordinates (α β γ : ℝ) (R : ℝ) :
  orthocenter α β γ R = (R * (Real.cos α + Real.cos β + Real.cos γ), R * (Real.sin α + Real.sin β + Real.sin γ)) :=
begin
  sorry
end

end trig_identity_orthocenter_coordinates_l648_648819


namespace parabola_line_intersect_at_one_point_l648_648233

theorem parabola_line_intersect_at_one_point :
  ∃ a : ℝ, (∀ x : ℝ, (ax^2 + 5 * x + 2 = -2 * x + 1)) ↔ a = 49 / 4 :=
by sorry

end parabola_line_intersect_at_one_point_l648_648233


namespace horizontal_asymptote_x_value_l648_648265

def g (x : ℝ) := (3 * x^2 - 12 * x - 9) / (x^2 - 5 * x + 2)

theorem horizontal_asymptote_x_value : ∃ x : ℝ, g(x) = 3 ∧ x = 5 :=
by
  exact ⟨5, by sorry, by sorry⟩

end horizontal_asymptote_x_value_l648_648265


namespace area_of_region_l648_648131

theorem area_of_region : 
  (∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y = 0 → 
     let a := (x - 4)^2 + (y + 3)^2 
     (a = 25) ∧ ∃ r : ℝ, r = 5 ∧ (π * r^2 = 25 * π)) := 
sorry

end area_of_region_l648_648131


namespace isosceles_triangle_sin_l648_648111

theorem isosceles_triangle_sin (x : ℝ) (hx : 0 < x ∧ x < π / 2)
  (h1 : sin x = sin (π / 2 - x)) 
  (h2 : sin x = sin (π - 9 * x)) :
  x = π / 7 ∨ x = 3 * π / 7 ∨ x = π / 11 ∨ x = 3 * π / 11 :=
sorry

end isosceles_triangle_sin_l648_648111


namespace total_cost_is_eight_x_l648_648056

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l648_648056


namespace distinct_ordered_pairs_count_l648_648879

theorem distinct_ordered_pairs_count :
  let pairs := {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ x^4 * y^4 - 20 * x^2 * y^2 + 64 = 0} in
  pairs.to_finset.card = 5 :=
by
  sorry

end distinct_ordered_pairs_count_l648_648879


namespace total_students_correct_l648_648972

noncomputable def total_students : ℕ :=
  let x := 522
  let y := 549
  99 + x

theorem total_students_correct : total_students = 621 :=
by
  -- Assume the conditions
  have h1 : 99 + 522 = 549 + 72 := by sorry
  have h2 : 122 * 99 + 155 * 522 = 148 * 549 + 163 * 72 := by sorry
  
  -- Calculate total students using conditions
  calc
    total_students
    = 99 + 522 : by rfl
    ... = 621 : by rfl

end total_students_correct_l648_648972


namespace find_number_l648_648076

theorem find_number (x : ℤ) (h : x - 27 = 49) : x = 76 := by
  sorry

end find_number_l648_648076


namespace num_imaginary_complex_numbers_l648_648067

def distinct (a b : ℕ) : Prop := a ≠ b
def imaginary (z : ℂ) : Prop := z.re = 0

theorem num_imaginary_complex_numbers :
  {z : ℂ // ∃ (a b : ℕ), a ∈ {0, 1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ distinct a b ∧ imaginary (a + b * complex.I)}.to_finset.card = 36 := 
by
  sorry

end num_imaginary_complex_numbers_l648_648067


namespace ROI_difference_is_correct_l648_648640

noncomputable def compound_interest (P : ℝ) (rates : List ℝ) : ℝ :=
rates.foldl (λ acc rate => acc * (1 + rate)) P

noncomputable def Emma_investment := compound_interest 300 [0.15, 0.12, 0.18]

noncomputable def Briana_investment := compound_interest 500 [0.10, 0.08, 0.14]

noncomputable def ROI_difference := Briana_investment - Emma_investment

theorem ROI_difference_is_correct : ROI_difference = 220.808 := 
sorry

end ROI_difference_is_correct_l648_648640


namespace geometric_sequence_S12_l648_648258

theorem geometric_sequence_S12 (S : ℕ → ℝ) (S_4_eq : S 4 = 20) (S_8_eq : S 8 = 30) :
  S 12 = 35 :=
by
  sorry

end geometric_sequence_S12_l648_648258


namespace number_of_pencil_cartons_l648_648848

theorem number_of_pencil_cartons
  (P E : ℕ) 
  (h1 : P + E = 100)
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_pencil_cartons_l648_648848


namespace solution_set_of_xf_x_minus_1_gt_0_l648_648680

variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = - f x)
variable (h_pos : ∀ x : ℝ, x > 0 → f x = x^2 - 3x - 4)

theorem solution_set_of_xf_x_minus_1_gt_0 :
  {x : ℝ | x * f (x - 1) > 0} = 
  {x | x < -3} ∪ {x | 0 < x ∧ x < 1} ∪ {x | x > 5} :=
sorry

end solution_set_of_xf_x_minus_1_gt_0_l648_648680


namespace distance_le_radius_l648_648962

variable (L : Line) (O : Circle)
variable (d r : ℝ)

-- Condition: Line L intersects with circle O
def intersects (L : Line) (O : Circle) : Prop := sorry -- Sketch: define what it means for a line to intersect a circle

axiom intersection_condition : intersects L O

-- Problem: Prove that if a line L intersects a circle O, then the distance d from the center of the circle to the line is less than or equal to the radius r of the circle.
theorem distance_le_radius (L : Line) (O : Circle) (d r : ℝ) :
  intersects L O → d ≤ r := by
  sorry

end distance_le_radius_l648_648962


namespace total_cost_is_eight_times_l648_648013

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l648_648013


namespace cars_meeting_distance_from_midpoint_l648_648690

def distance_AB : ℝ := 240
def speed_car1 : ℝ := 60
def speed_car2 : ℝ := 80
def midpoint_distance : ℝ := distance_AB / 2
def meeting_distance : ℝ := 17.14

theorem cars_meeting_distance_from_midpoint :
  ∀ (d_AB : ℝ) (s_c1 : ℝ) (s_c2 : ℝ) (mid_d : ℝ),
  d_AB = distance_AB →
  s_c1 = speed_car1 →
  s_c2 = speed_car2 →
  mid_d = midpoint_distance →
  (abs ((s_c1 / (s_c1 + s_c2)) * d_AB - mid_d) = meeting_distance) :=
by
  intros d_AB s_c1 s_c2 mid_d h_d_AB h_sc1 h_sc2 h_mid_d
  rw [h_d_AB, h_sc1, h_sc2, h_mid_d]
  sorry

end cars_meeting_distance_from_midpoint_l648_648690


namespace no_injective_homomorphism_GL2_to_Sp_l648_648678

open scoped Classical

variables {p : ℕ} [fact (Nat.prime p)]

-- Definition of GL_2(Z/pZ), the group of invertible 2x2 matrices mod p
def GL_2 (p : ℕ) [fact (Nat.prime p)] := 
  {A : Matrix (Fin 2) (Fin 2) (ZMod p) // A.det ≠ 0}

def S_p (p : ℕ) := Equiv.Perm (Fin p)

-- Statement that there is no injective homomorphism from GL_2(Z/pZ) to S_p
theorem no_injective_homomorphism_GL2_to_Sp {p : ℕ} [fact (Nat.prime p)] :
  ¬ ∃ (φ : GL_2 p →* S_p p), Function.Injective φ := sorry

end no_injective_homomorphism_GL2_to_Sp_l648_648678


namespace range_of_m_l648_648269

noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then 3 + 3 * x
  else if x <= 3 then -1
  else x + 5

theorem range_of_m (m : ℝ) (x : ℝ) (hx : f x ≥ 1 / m - 4) :
  m < 0 ∨ m = 1 :=
sorry

end range_of_m_l648_648269


namespace magnitude_of_z_l648_648297

-- Define the complex number z
def z : ℂ := (2 + complex.i) / (1 - complex.i)

-- Theorem: The magnitude of z is sqrt(10) / 2
theorem magnitude_of_z : complex.abs z = (real.sqrt 10) / 2 := 
sorry

end magnitude_of_z_l648_648297


namespace total_cost_is_eight_times_l648_648049

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l648_648049


namespace least_number_to_add_to_2496_l648_648135

theorem least_number_to_add_to_2496 :
  ∃ n, (∃ k : ℕ, 2496 + n = k * (Nat.lcm (Nat.lcm 5 7) 13)) ∧ n = 234 :=
begin
  sorry
end

end least_number_to_add_to_2496_l648_648135


namespace positive_root_l648_648251

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 - 2 * x - real.sqrt 3

theorem positive_root (a b : ℝ) (h₁ : a = 2) (h₂ : b = 1) :
  f (a + b * real.sqrt 3) = 0 :=
by 
  -- Introduce definitions and necessary computations
  have x : ℝ := a + b * real.sqrt 3,
  -- We need to show f(x) = 0 which we'll approach by simplifying the expression 
  sorry

end positive_root_l648_648251


namespace equal_segments_OI_OJ_l648_648615

/--
Given quadrilateral ABCD with diagonals intersecting at point O, 
extensions of opposite sides intersecting at points E and F, 
a line through O parallel to EF intersecting BC and AD at points I and J, 
we are to prove that OI = OJ.
-/
theorem equal_segments_OI_OJ
  {A B C D E F O I J : Type}
  (h1 : intersect_diag ABCD O)
  (h2 : intersects E (extend_opposite_sides ABCD))
  (h3 : intersects F (extend_opposite_sides ABCD))
  (h4 : parallel (line_through O) (line_through EF) (intersect BC I) (intersect AD J)) :
  segment_length O I = segment_length O J :=
sorry

end equal_segments_OI_OJ_l648_648615


namespace sum_of_primes_1_to_20_l648_648549

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648549


namespace min_students_same_place_l648_648221

-- Define the context of the problem
def classSize := 45
def numberOfChoices := 6

-- The proof statement
theorem min_students_same_place : 
  ∃ (n : ℕ), 8 ≤ n ∧ n = Nat.ceil (classSize / numberOfChoices) :=
by
  sorry

end min_students_same_place_l648_648221


namespace similar_triangles_l648_648341

variable {A B C D : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variable [Ord A] [Ord B] [Ord C] [Ord D]

-- Define the right triangle ΔABC with right angle at C
variable {ΔABC : Type}
variables (a b c : ℕ) (D : ℕ)

-- Conditionally define the structure of the triangle and the altitude
axiom right_triangle_abc (ΔABC : Type) (C : Type) : A → B → C → Prop
axiom altitude_cd (C D: Type) : C → D → Prop

-- Definition of the three triangles: the original and the two smaller triangles
noncomputable def triangle_ABC := right_triangle_abc ΔABC C
noncomputable def triangle_ACD := altitude_cd C D
noncomputable def triangle_BCD := altitude_cd C D

-- Stating the similarity of the three triangles
theorem similar_triangles : 
  ∀ (ΔABC : Type) (ΔACD : Type) (ΔBCD : Type), 
    right_triangle_abc ΔABC C ∧ altitude_cd C D →
    (ΔABC ≈ ΔACD ∧ ΔABC ≈ ΔBCD ∧ ΔACD ≈ ΔBCD) := 
by sorry

end similar_triangles_l648_648341


namespace sum_primes_between_1_and_20_l648_648519

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648519


namespace jared_current_age_jared_current_age_l648_648905

theorem jared_current_age (jared_age_two_years_ago : ℕ) (tom_age_in_five_years : ℕ) : 
  (jared_age_two_years_ago + 2) = 48 :=
by
  sorry

variables (tom_curr_age : ℕ) (tom_age_two_years_ago : ℕ) (jared_age_two_years_ago : ℕ)
(h1 : tom_curr_age = tom_age_in_five_years - 5)
(h2 : tom_age_two_years_ago = tom_curr_age - 2)
(h3 : jared_age_two_years_ago = 2 * tom_age_two_years_ago)

include h1 h2 h3

theorem jared_current_age (jared_age_two_years_ago : ℕ) (tom_age_in_five_years : ℕ) : 
  (jared_age_two_years_ago + 2) = 48 :=
by
  let tom_curr_age := tom_age_in_five_years - 5
  let tom_age_two_years_ago := tom_curr_age - 2
  let jared_age_two_years_ago := 2 * tom_age_two_years_ago
  exact Eq.refl 48

#check @jared_current_age

end jared_current_age_jared_current_age_l648_648905


namespace probability_at_least_4_consecutive_heads_l648_648172

def fair_coin_flip : Type := Bool
def outcome_7_flips : Type := vector fair_coin_flip 7

def at_least_4_consecutive_heads (v : outcome_7_flips) : Prop :=
  (v.to_list.take 4 = [true, true, true, true]) ∨
  (v.to_list.drop 1).take 4 = [true, true, true, true] ∨
  (v.to_list.drop 2).take 4 = [true, true, true, true] ∨
  (v.to_list.drop 3).take 4 = [true, true, true, true] ∨
  (v.to_list.drop 4).take 4 = [true, true, true, true]

theorem probability_at_least_4_consecutive_heads :
  let outcomes := vector.finset_enum fair_coin_flip 7
  in (finset.filter at_least_4_consecutive_heads outcomes).card.to_rat / outcomes.card.to_rat = 1 / 4 :=
by sorry

end probability_at_least_4_consecutive_heads_l648_648172


namespace total_carrots_l648_648845

def initial_carrots_day1 : Nat := 125
def damaged_carrots_day1 : Nat := 30
def initial_carrots_day2 : Nat := 250
def percentage_spoiled_day2 : Float := 0.15

theorem total_carrots (day1_carrots day2_carrots spoiled_carrots : Float) :
  day1_carrots = initial_carrots_day1 - damaged_carrots_day1 →
  day2_carrots = initial_carrots_day2 - spoiled_carrots →
  spoiled_carrots = Float.floor (initial_carrots_day2 * percentage_spoiled_day2) →
  day1_carrots + day2_carrots = 308 :=
by
  sorry

end total_carrots_l648_648845


namespace find_divisors_of_zks_l648_648386

-- Definitions to match conditions
def z : ℤ := sorry -- Assume z is given
def z_k (k : ℕ) : ℤ := (1 + z) + (List.range k).sum (λ i, z^(i + 1))
def M : Set ℤ := { z_k k | k : ℕ }

-- Define the set T based on the problem statement
def T : Set ℕ := { n | Nat.coprime n (Int.natAbs z) }

-- The theorem statement
theorem find_divisors_of_zks (h : z > 1) :
  T = { n | Nat.coprime n (Int.natAbs z) } :=
sorry

end find_divisors_of_zks_l648_648386


namespace find_coordinates_of_H_l648_648896

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def midpoint (p1 p2 : Point3D) : Point3D :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2⟩

theorem find_coordinates_of_H (E F G H : Point3D)
  (hE : E = ⟨2, 3, -1⟩)
  (hF : F = ⟨0, -1, 3⟩)
  (hG : G = ⟨4, -3, 1⟩)
  (h_midpoints : midpoint E G = midpoint F H) :
  H = ⟨6, 1, -3⟩ :=
sorry

end find_coordinates_of_H_l648_648896


namespace color_and_painting_method_combinations_l648_648395

theorem color_and_painting_method_combinations
  (colors : ℕ) (methods : ℕ)
  (h1 : colors = 5)
  (h2 : methods = 4) :
  colors * methods = 20 :=
by
  rw [h1, h2]
  norm_num

end color_and_painting_method_combinations_l648_648395


namespace circle_condition1_circle_condition2_l648_648154

open Real

-- Define points and conditions
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)
def line (p : ℝ × ℝ) : Prop := p.1 - 2 * p.2 - 3 = 0

-- Define the circle equation for part (1)
def circle_eq1 (p : ℝ × ℝ) : Prop := (p.1 + 1)^2 + (p.2 + 2)^2 = 10

-- Define the circle equation for part (2)
def circle_eq2 (p : ℝ × ℝ) : Prop := p.1^2 + (p.2 + 4)^2 = 5

-- Part (1) proof goal
theorem circle_condition1 (center : ℝ × ℝ) (x y : ℝ) :
  (circle_eq1 A) ∧ (circle_eq1 B) ∧ (line center) → (x + 1)^2 + (y + 2)^2 = 10 :=
sorry

-- Part (2) proof goal
theorem circle_condition2 (center : ℝ × ℝ) (x y : ℝ) :
  (circle_eq2 A) ∧ (circle_eq2 B) ∧ (line center) ∧ minimized_area x y → x^2 + (y + 4)^2 = 5 :=
sorry

end circle_condition1_circle_condition2_l648_648154


namespace verify_parabola_D_l648_648295

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D_l648_648295


namespace number_of_roots_l648_648768

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 2 * b * x + 3 * c

theorem number_of_roots (a b c x₁ x₂ : ℝ) (h_extreme : x₁ ≠ x₂)
    (h_fx1 : f a b c x₁ = x₁) :
    (∃ (r : ℝ), 3 * (f a b c r)^2 + 4 * a * (f a b c r) + 2 * b = 0) :=
sorry

end number_of_roots_l648_648768


namespace inverse_function_property_l648_648301

noncomputable def f (a x : ℝ) : ℝ := (x - a) * |x|

theorem inverse_function_property (a : ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, f a (g x) = x) ↔ a = 0 :=
by sorry

end inverse_function_property_l648_648301


namespace initial_amount_l648_648923

theorem initial_amount (bread_price : ℝ) (bread_qty : ℝ) (pb_price : ℝ) (leftover : ℝ) :
  bread_price = 2.25 → bread_qty = 3 → pb_price = 2 → leftover = 5.25 →
  bread_qty * bread_price + pb_price + leftover = 14 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num


end initial_amount_l648_648923


namespace next_term_geometric_sequence_l648_648483

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l648_648483


namespace smallest_four_digit_divisible_five_smallest_primes_l648_648664

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l648_648664


namespace range_of_independent_variable_l648_648107

theorem range_of_independent_variable (x : ℝ) : 2 < x → ∃ y, y = 2 / (real.sqrt (x - 2)) :=
by
  sorry

end range_of_independent_variable_l648_648107


namespace wood_length_equation_l648_648939

-- Define the conditions as hypotheses
def length_of_wood_problem (x : ℝ) :=
  (1 / 2) * (x + 4.5) = x - 1

-- Now we state the theorem we want to prove, which is equivalent to the question == answer
theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l648_648939


namespace not_divisible_by_3_or_11_below_1200_l648_648630

theorem not_divisible_by_3_or_11_below_1200 : 
  {n : ℕ | n < 1200 ∧ ¬ (n % 3 = 0 ∨ n % 11 = 0)}.card = 727 := 
sorry

end not_divisible_by_3_or_11_below_1200_l648_648630


namespace triangle_ABC_angles_l648_648801

variables {A B C H O I : Type}
variables [triangle A B C] [acute_angle_60 (angle BAC 60)] 

theorem triangle_ABC_angles 
  (triangle_ABC : Triangle A B C)
  (angle_BAC_60 : angle BAC = 60)
  (H_OC_I : ∃ H O I, orthocenter H A B C ∧ circumcenter O A B C ∧ incenter I A B C)
  (BH_eq_OI : BH = OI) :
  angle ABC = 80 ∧ angle ACB = 40 :=
sorry

end triangle_ABC_angles_l648_648801


namespace problem_part1_problem_part2_l648_648744

variable (a : ℝ)

def quadratic_solution_set_1 := {x : ℝ | x^2 + 2*x + a = 0}
def quadratic_solution_set_2 := {x : ℝ | a*x^2 + 2*x + 2 = 0}

theorem problem_part1 :
  (quadratic_solution_set_1 a = ∅ ∨ quadratic_solution_set_2 a = ∅) ∧ ¬ (quadratic_solution_set_1 a = ∅ ∧ quadratic_solution_set_2 a = ∅) →
  (1/2 < a ∧ a ≤ 1) :=
sorry

theorem problem_part2 :
  quadratic_solution_set_1 a ∪ quadratic_solution_set_2 a ≠ ∅ →
  a ≤ 1 :=
sorry

end problem_part1_problem_part2_l648_648744


namespace part1_maximum_value_part1_minimum_value_part2_maximum_value_part2_minimum_value_l648_648696

-- Define the function f(x) with parameter m
noncomputable def f (x m : ℝ) : ℝ := 2*sin x^2 + m*cos x + 1

-- Part (1): Maximum and minimum values when m = 1
theorem part1_maximum_value : 
  ∃ x : ℝ, f x 1 = 17 / 8 :=
sorry

theorem part1_minimum_value :
  ∃ x : ℝ, f x 1 = -1 :=
sorry

-- Part (2): Maximum and minimum values for general m
theorem part2_maximum_value (m : ℝ) :
  (∀ x : ℝ, -4 ≤ m ∧ m ≤ 4 → f x m ≤ m^2 / 8 + 2) ∧ 
  (∀ x : ℝ, m < -4 → f x m ≤ -m) ∧
  (∀ x : ℝ, m > 4 → f x m ≤ m) :=
sorry

theorem part2_minimum_value (m : ℝ) :
  (∀ x : ℝ, m < 0 → f x m ≥ m) ∧ 
  (∀ x : ℝ, 0 ≤ m → f x m ≥ -m) :=
sorry

end part1_maximum_value_part1_minimum_value_part2_maximum_value_part2_minimum_value_l648_648696


namespace arithmetic_sequence_fifth_term_l648_648092

theorem arithmetic_sequence_fifth_term (x y : ℚ) 
  (h1 : a₁ = x + y) 
  (h2 : a₂ = x - y) 
  (h3 : a₃ = x * y) 
  (h4 : a₄ = x / y) 
  (h5 : a₂ - a₁ = -2 * y) 
  (h6 : a₃ - a₂ = -2 * y) 
  (h7 : a₄ - a₃ = -2 * y) 
  (hx : x = -9 / 8)
  (hy : y = -3 / 5) : 
  a₅ = 123 / 40 :=
by
  sorry

end arithmetic_sequence_fifth_term_l648_648092


namespace cost_comparison_l648_648585

theorem cost_comparison (n : ℕ) (x : ℝ) (h : n > 0) : 
  let y1 := x + (3 / 4) * x * (n - 1)
  let y2 := (4 / 5) * n * x
  in (n > 5 → y1 < y2) ∧ (n < 5 → y1 > y2) ∧ (n = 5 → y1 = y2) :=
by
  sorry

end cost_comparison_l648_648585


namespace tetrahedron_sum_of_faces_l648_648687

theorem tetrahedron_sum_of_faces (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum_vertices : b * c * d + a * c * d + a * b * d + a * b * c = 770) :
  a + b + c + d = 57 :=
sorry

end tetrahedron_sum_of_faces_l648_648687


namespace no_315_segment_polygonal_line_l648_648808

def segment := ℝ × ℝ -- Representing a segment as a pair of real numbers (for simplicity)

-- Defining a closed polygonal line
structure polygonal_line (n : ℕ) :=
(segments : fin n → segment)
(intersect : Π (i j : fin n), i ≠ j → exists p : ℝ × ℝ, (p ∈ segments i) ∧ (p ∈ segments j))
(perpendicular : Π (i j : fin n), i ≠ j → ∀ p : ℝ × ℝ, (p ∈ segments i) ∧ (p ∈ segments j) → is_perpendicular (segments i) (segments j)) -- Assuming the existence of is_perpendicular check
(symmetric_about_axis : bool)

-- Defining the property of mutual perpendicularity for simplicity
def is_perpendicular (s1 s2 : segment) : Prop := sorry

-- The main statement to prove
theorem no_315_segment_polygonal_line : ¬ ∃ (pl : polygonal_line 315), pl.symmetric_about_axis = true :=
sorry

end no_315_segment_polygonal_line_l648_648808


namespace assembly_possible_l648_648271

-- Definition of the color type
inductive Color
| Red
| Blue
| White

-- Given cubes: 9 red, 9 blue, 9 white
def num_cubes : Color → ℕ
| Color.Red := 9
| Color.Blue := 9
| Color.White := 9

-- A function representing a 3x3x3 cube with colors assigned to each position:
def cube (i j k : Fin 3) : Color

-- The main statement that needs to be proved
theorem assembly_possible : 
  (∃ f : (Fin 3) × (Fin 3) × (Fin 3) → Color, 
    (∀ i j k, f (i, j, k) = cube i j k) ∧
    ∀ i j, (∃ c1 c2 : Color, c1 ≠ c2 ∧ 
          (∃ a b c, 
            (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
              (f (i, j, a) = c1 ∧ f (i, j, b) = c2 ∧ f (i, j, c) = c1 ∨
               f (i, j, a) = c2 ∧ f (i, j, b) = c1 ∧ f (i, j, c) = c2))
          )
          ∧ ∀ l, ∃ c1 c2 : Color, c1 ≠ c2 ∧
                (cube i j k = c1 ∨ cube i j k = c2)
            )
  ) :=
sorry

end assembly_possible_l648_648271


namespace arithmetic_sqrt_16_l648_648426

theorem arithmetic_sqrt_16 : real.sqrt 16 = 4 := by
  sorry

end arithmetic_sqrt_16_l648_648426


namespace games_needed_to_find_winner_l648_648158

theorem games_needed_to_find_winner (n : ℕ) (h : n = 16) : 
  let total_games := n - 1 in total_games = 15 :=
by
  have x : total_games = n - 1 := rfl
  simp [x, h]
  sorry

end games_needed_to_find_winner_l648_648158


namespace student_arrangement_count_l648_648118

theorem student_arrangement_count :
  let students := ["A", "B", "C", "D", "E"] in
  let possible_arrangements := students.permutations.to_list in
  let valid_arrangements := possible_arrangements.filter (λ arrangement, (arrangement.head ≠ "A") ∧ (arrangement.tail.head ≠ "B")) in
  valid_arrangements.length = 114 :=
by
  sorry

end student_arrangement_count_l648_648118


namespace tank_filling_rate_l648_648907

/-- Water is poured into a tank so that the tank is being filled at a certain rate. The empty rectangular tank is 6 feet long, 4 feet wide, and 3 feet deep. 
It takes approximately 18 hours to fill the tank. Prove that the rate at which the tank is being filled is 4 cubic feet per hour. -/
theorem tank_filling_rate :
  let length := 6
  let width := 4
  let depth := 3
  let time := 18
  let volume := length * width * depth
  volume / time = 4 :=
by
  let length := 6
  let width := 4
  let depth := 3
  let time := 18
  let volume := length * width * depth
  have h1 : volume = 6 * 4 * 3 := rfl
  have h2 : volume = 72 := by rw [h1]
  have h3 : 72 / 18 = 4 := rfl
  rw [h2, h3]
  sorry

end tank_filling_rate_l648_648907


namespace smallest_even_number_of_sum_1194_l648_648334

-- Defining the given condition
def sum_of_three_consecutive_even_numbers (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) = 1194

-- Stating the theorem to prove the smallest even number
theorem smallest_even_number_of_sum_1194 :
  ∃ x : ℕ, sum_of_three_consecutive_even_numbers x ∧ x = 396 :=
by
  sorry

end smallest_even_number_of_sum_1194_l648_648334


namespace four_digit_even_numbers_l648_648102

theorem four_digit_even_numbers : 
  ∃ n : ℕ, n = 10 ∧
    let digits := {2, 0, 1, 7} in
    let four_digit_even (x : ℕ) := even x ∧ 1000 ≤ x ∧ x < 10000 in
    ∀ (x : ℕ), four_digit_even x → ∀ d ∈ digits, 
      (d / 1000) ∈ digits ∧ (d % 1000 / 100) ∈ digits ∧ 
      (d % 100 / 10) ∈ digits ∧ (d % 10) ∈ digits → 
      cardinality {y : ℕ | four_digit_even y ∧ y ∈ digits} = n 
:= 
begin
  sorry
end

end four_digit_even_numbers_l648_648102


namespace triangle_area_from_linear_function_l648_648719

noncomputable def linear_function_expression (A B : ℝ × ℝ) : ℝ → ℝ :=
  λ x, -x + 2

theorem triangle_area_from_linear_function (A B : ℝ × ℝ)
  (hA : A = (-1, 3)) (hB : B = (3, -1)) : 
  (0, 2), (2, 0) and (0, 0) are vertices of the triangle formed by y = -x + 2 and axes
    and area is 2 :=
begin
  sorry
end

end triangle_area_from_linear_function_l648_648719


namespace janet_dresses_total_pockets_l648_648364

theorem janet_dresses_total_pockets :
  let dresses := 24
  let with_pockets := dresses / 2
  let with_two_pockets := with_pockets / 3
  let with_three_pockets := with_pockets - with_two_pockets
  let total_two_pockets := with_two_pockets * 2
  let total_three_pockets := with_three_pockets * 3
  total_two_pockets + total_three_pockets = 32 := by
  sorry

end janet_dresses_total_pockets_l648_648364


namespace calculation_l648_648330

def operation_e (x y z : ℕ) : ℕ := 3 * x * y * z

theorem calculation :
  operation_e 3 (operation_e 4 5 6) 1 = 3240 :=
by
  sorry

end calculation_l648_648330


namespace root_bounds_inequality_smallest_k_satisfying_inequality_l648_648069

theorem root_bounds_inequality (n : ℕ) (hn : n > 0) : 1 ≤ n ^ (1 / n : ℝ) ∧ n ^ (1 / n : ℝ) ≤ 2 :=
by sorry

theorem smallest_k_satisfying_inequality :
  ∀ n : ℕ, n > 0 → 1 ≤ n ^ (1 / n : ℝ) ∧ n ^ (1 / n : ℝ) ≤ (real.sqrt 3) :=
by sorry

end root_bounds_inequality_smallest_k_satisfying_inequality_l648_648069


namespace solution_set_of_inequality_l648_648886

variable (a b x : ℝ)
variable (h1 : ∀ x, ax + b > 0 ↔ 1 < x)

theorem solution_set_of_inequality : ∀ x, (ax + b) * (x - 2) < 0 ↔ (1 < x ∧ x < 2) :=
by sorry

end solution_set_of_inequality_l648_648886


namespace number_of_self_inverses_l648_648230

def is_involution (a d : ℝ) : Prop :=
  let M := Matrix 2 2 ℝ 
  in M.det == 1 ∧ M.trace == 0
  -- Example for writing out the full conditions
  ∧ (M 0 0 == a * a + 4 * -7)
  ∧ (M 0 1 == 0)
  ∧ (M 1 0 == 0)
  ∧ (M 1 1 == d * d + 4 * -7)

theorem number_of_self_inverses : 
  (finset.univ.filter (λ a : ℝ, finset.univ.filter (λ d : ℝ, is_involution a d).card).card) = 2 :=
begin
  sorry
end

end number_of_self_inverses_l648_648230


namespace division_of_exponents_l648_648920

-- Define the conditions as constants and statements that we are concerned with
variables (x : ℝ)

-- The Lean 4 statement of the equivalent proof problem
theorem division_of_exponents (h₁ : x ≠ 0) : x^8 / x^2 = x^6 := 
sorry

end division_of_exponents_l648_648920


namespace true_propositions_count_l648_648443

noncomputable def point := ℝ × ℝ

def segment (A B : point) : set point := 
  {P | P.1 = (A.1 + B.1) / 2}

def equidistant (A B P : point) : Prop :=
  dist A P = dist B P

def perpendicular_bisector (A B : point) : set point :=
  {P | equidistant A B P}

def original_proposition (A B : point) : Prop :=
  ∀ P, P ∈ (perpendicular_bisector A B) → equidistant A B P

def inverse_proposition (A B : point) : Prop :=
  ∀ P, equidistant A B P → P ∈ (perpendicular_bisector A B)

def negation_proposition (A B : point) : Prop :=
  ∀ P, P ∉ (perpendicular_bisector A B) → ¬ (equidistant A B P)

def contrapositive_proposition (A B : point) : Prop :=
  ∀ P, ¬ (equidistant A B P) → P ∉ (perpendicular_bisector A B)

theorem true_propositions_count (A B : point) :
  [original_proposition A B, inverse_proposition A B, 
   negation_proposition A B, contrapositive_proposition A B]
  .count(λ p, p=true) = 3 := 
sorry

end true_propositions_count_l648_648443


namespace marked_price_l648_648966

theorem marked_price (purchase_price marked_price : ℝ) (cost_discount target_gain discount_on_marked : ℝ)
  (purchase : purchase_price = 30 * (1 - cost_discount))
  (selling_price : marked_price * (1 - discount_on_marked) = purchase_price * (1 + target_gain)) :
  marked_price = 35.42 :=
by
  -- Definitions and initial assumptions
  have cost_discount_def : cost_discount = 0.15 := sorry,
  have target_gain_def : target_gain = 0.25 := sorry,
  have discount_on_marked_def : discount_on_marked = 0.10 := sorry,
  
  -- Calculating the purchase price
  rw [cost_discount_def] at purchase,
  have purchase_price_def : purchase_price = 30 * 0.85 := by rw purchase,
  
  -- Setting up the selling price equation
  rw [target_gain_def, purchase_price_def] at selling_price,
  have expected_selling_price : marked_price * 0.90 = 31.875 := by rw selling_price,
  
  -- Calculating the marked price
  have marked_price_calc : marked_price = 31.875 / 0.90 := sorry,

  -- Conclusion
  exact eq.trans marked_price_calc (_ : 31.875 / 0.90 = 35.42),
  norm_num,

end marked_price_l648_648966


namespace sequence_product_n_eq_million_l648_648742

theorem sequence_product_n_eq_million (n : ℕ) :
  (∏ k in Finset.range n, 10 ^ (k + 1) / 11) = 1000000 ↔ n = 11 := by
sorry

end sequence_product_n_eq_million_l648_648742


namespace wood_length_equation_l648_648941

-- Define the conditions as hypotheses
def length_of_wood_problem (x : ℝ) :=
  (1 / 2) * (x + 4.5) = x - 1

-- Now we state the theorem we want to prove, which is equivalent to the question == answer
theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l648_648941


namespace orthocenter_AXY_moves_on_circle_l648_648629

variable {K : Type*} [EuclideanGeometry K]

-- Define the points on the circle ω
variables (A B C M H X Y : K)
-- Define the circle ω, with B and C on it, and A varying on it
variable (ω : Circle K)

-- Define the midpoint of BC
def midpoint (B C : K) : K := (B + C) / 2

#check EuclideanGeometry.Circle

-- Prove that the orthocenter of triangle AXY moves on a circle as A varies on ω
theorem orthocenter_AXY_moves_on_circle
  (B C : K)
  (ω : Circle K)
  (M := midpoint B C)
  (H := EuclideanGeometry.orthocenter (triangle A B C))
  (X := EuclideanGeometry.perpendicular_from H (line_through M H))
  (Y := EuclideanGeometry.perpendicular_from H (line_through A C)) :
  ∀ A ∈ ω, ∃ (H' : K), 
  H' = EuclideanGeometry.orthocenter (triangle A X Y) ∧ H' ∈ some_circle :=
sorry

end orthocenter_AXY_moves_on_circle_l648_648629


namespace beetle_total_distance_l648_648895

theorem beetle_total_distance 
  (r_outer : ℝ) (r_middle : ℝ) (r_inner : ℝ)
  (r_outer_eq : r_outer = 25)
  (r_middle_eq : r_middle = 15)
  (r_inner_eq : r_inner = 5)
  : (1/3 * 2 * Real.pi * r_middle + (r_outer - r_middle) + 1/2 * 2 * Real.pi * r_inner + 2 * r_outer + (r_middle - r_inner)) = (15 * Real.pi + 70) :=
by
  rw [r_outer_eq, r_middle_eq, r_inner_eq]
  have := Real.pi
  sorry

end beetle_total_distance_l648_648895


namespace sum_of_primes_between_1_and_20_l648_648494

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648494


namespace find_alpha_l648_648286

theorem find_alpha (P : Real × Real) (h: P = (Real.sin (50 * Real.pi / 180), 1 + Real.cos (50 * Real.pi / 180))) :
  ∃ α : Real, α = 65 * Real.pi / 180 := by
  sorry

end find_alpha_l648_648286


namespace length_of_one_overlapping_segment_l648_648137

theorem length_of_one_overlapping_segment 
  (tape_length : ℕ)
  (total_length : ℕ)
  (num_tapes : ℕ)
  (actual_total_length : ℕ)
  (overlapping_segments : ℕ)
  (total_non_overlapping_length : ℕ := num_tapes * tape_length)
  (total_overlap_length : ℕ := total_non_overlapping_length - actual_total_length) :
  total_length = total_non_overlapping_length - overlapping_segments * (total_overlap_length / overlapping_segments) →
  (total_overlap_length / overlapping_segments) = 25 :=
by
  intro h
  have h1 : total_non_overlapping_length = 1000 := by rw [total_non_overlapping_length]
  have h2 : total_overlap_length = 75 := by rw [total_overlap_length, h1, h]
  have h3 : overlapping_segments = 3 := by decide
  have h4 : total_overlap_length / overlapping_segments = 25 := by norm_num [h2, h3]
  rw [h4]
  exact h4
  sorry

end length_of_one_overlapping_segment_l648_648137


namespace cobbler_works_fri_hours_l648_648168

-- Conditions
def mending_rate : ℕ := 3  -- Pairs of shoes per hour
def mon_to_thu_days : ℕ := 4
def hours_per_day : ℕ := 8
def weekly_mended_pairs : ℕ := 105

-- Translate the conditions
def hours_mended_mon_to_thu : ℕ := mon_to_thu_days * hours_per_day
def pairs_mended_mon_to_thu : ℕ := mending_rate * hours_mended_mon_to_thu
def pairs_mended_fri : ℕ := weekly_mended_pairs - pairs_mended_mon_to_thu

-- Theorem statement to prove the desired question
theorem cobbler_works_fri_hours : (pairs_mended_fri / mending_rate) = 3 := by
  sorry

end cobbler_works_fri_hours_l648_648168


namespace sequence_solutions_l648_648278

theorem sequence_solutions:
  (∃ a : ℕ → ℕ, 
    (∀ n : ℕ, 0 < n → a (n + 1) - a n = 3) ∧ 
    (∑ i in finset.range 5, a (i + 1) = 45) ∧ 
    (∀ n : ℕ, a n = 3 * n)) 
  ∧
  (∃ b : ℕ → ℕ, 
    (b 1 = 2) ∧ 
    (∃ q : ℕ, ∀ n : ℕ, b (n + 1) = 2 * b n) ∧ 
    (∀ n : ℕ, b n = 2^n))
  ∧
  (∃ S : ℕ → ℕ, 
    (S n = (n - 1) * 2^(n + 1) + 2) := sorry

end sequence_solutions_l648_648278


namespace fraction_by_foot_l648_648805

theorem fraction_by_foot (D distance_by_bus distance_by_car distance_by_foot : ℕ) (h1 : D = 24) 
  (h2 : distance_by_bus = D / 4) (h3 : distance_by_car = 6) 
  (h4 : distance_by_foot = D - (distance_by_bus + distance_by_car)) : 
  (distance_by_foot : ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_by_foot_l648_648805


namespace max_cards_divisible_l648_648117

def max_arranged_cards (card_numbers : List ℕ) : ℕ :=
  if h : card_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9] then
    8
  else
    sorry

theorem max_cards_divisible :
  max_arranged_cards [1, 2, 3, 4, 5, 6, 7, 8, 9] = 8 := 
by 
  unfold max_arranged_cards
  exact if_pos rfl

end max_cards_divisible_l648_648117


namespace solution_set_inequality_l648_648291

variable {a b c x x1 x2 : ℝ}

theorem solution_set_inequality(ax2_plus_bx_plus_c_0_roots: 
  ∀ (x : ℝ), (ax*x^2 + b*x + c = 0) ↔ (x = x1 ∨ x = x2),
  roots_order: x1 < x2, a_lt_zero: a < 0):
  { x : ℝ | ax * x^2 + b * x + c > 0 } = { x : ℝ | x1 < x ∧ x < x2 } :=
by 
  sorry

end solution_set_inequality_l648_648291


namespace set_intersection_M_N_l648_648710

theorem set_intersection_M_N (x : ℝ) :
  let M := {x | -4 < x ∧ x < -2}
  let N := {x | x^2 + 5*x + 6 < 0}
  M ∩ N = {x | -3 < x ∧ x < -2} :=
by
  sorry

end set_intersection_M_N_l648_648710


namespace sum_of_primes_1_to_20_l648_648553

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648553


namespace next_term_geometric_sequence_l648_648482

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l648_648482


namespace sqrt_difference_l648_648133

theorem sqrt_difference : sqrt (49 + 16) - sqrt (64 - 49) = sqrt 65 - sqrt 15 :=
by sorry

end sqrt_difference_l648_648133


namespace binom_18_10_l648_648622

theorem binom_18_10 :
  nat.choose 18 10 = 47190 :=
by
  have h₁ : nat.choose 16 7 = 11440 := rfl,
  have h₂ : nat.choose 16 9 = 11440 := rfl,
  sorry

end binom_18_10_l648_648622


namespace actual_distance_between_towns_l648_648432

theorem actual_distance_between_towns
  (d_map : ℝ)
  (miles_per_scale : ℝ)
  (scale : ℝ)
  (h1 : d_map = 18)
  (h2 : scale = 0.3)
  (h3 : miles_per_scale = 5)
  : (18 * (5 / 0.3)) = 300 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end actual_distance_between_towns_l648_648432


namespace max_value_sum_l648_648422

theorem max_value_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 10) :
  ∃ (a b c d : ℕ), x^2 + 3 * x * y + 2 * y^2 = (a + b * real.sqrt c) / d ∧ a + b + c + d = 154 := 
sorry

end max_value_sum_l648_648422


namespace compare_abc_l648_648825

noncomputable def a : ℝ := Real.exp 0.25
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := -4 * Real.log 0.75

theorem compare_abc : b < c ∧ c < a := by
  -- Additional proof steps would follow here
  sorry

end compare_abc_l648_648825


namespace find_c_l648_648701

   variable {a b c : ℝ}
   
   theorem find_c (h1 : 4 * a - 3 * b + c = 0)
     (h2 : (a - 1)^2 + (b - 1)^2 = 4) :
     c = 9 ∨ c = -11 := 
   by
     sorry
   
end find_c_l648_648701


namespace probability_of_vertex_and_integer_solution_l648_648453

def balls : Finset ℤ := { -5, -4, -3, -2, 2, 1 }

def parabola_vertex_third_quadrant (a : ℤ) : Prop :=
  let vertex := (-1, a + 1)
  vertex.2 < 0 -- Condition for third quadrant

def equation_has_integer_solution (a : ℤ) : Prop :=
  ∃ x : ℤ, (a * x) / (x - 2) = (3 * x + 2) / (2 - x) + 2

theorem probability_of_vertex_and_integer_solution :
  (let favorable_counts := balls.filter (λ a, parabola_vertex_third_quadrant a ∧ equation_has_integer_solution a).card
   let total_counts := balls.card
   favorable_counts / total_counts.toFloat = 2 / 3) :=
sorry

end probability_of_vertex_and_integer_solution_l648_648453


namespace total_weight_is_correct_l648_648160

/-- Define the input conditions --/
def weight_of_potatoes_kg : ℝ := 36
def weight_of_conversion_factor : ℝ := 2.20462
def weight_of_potatoes_pounds := weight_of_potatoes_kg * weight_of_conversion_factor
def weight_of_carrots_kg := weight_of_potatoes_kg / 1.5
def weight_of_carrots_pounds := weight_of_carrots_kg * weight_of_conversion_factor
def weight_of_tomatoes_pounds : ℝ := 16

/-- The total weight of all produce in pounds --/
def total_weight_of_produce_pounds := weight_of_potatoes_pounds + weight_of_carrots_pounds + weight_of_tomatoes_pounds

/-- Problem statement: Given the conditions, prove that the total weight of the produce is approximately 148.2772 pounds --/
theorem total_weight_is_correct :
  |total_weight_of_produce_pounds - 148.2772| < 0.01 := sorry

end total_weight_is_correct_l648_648160


namespace jared_current_age_condition_l648_648904

variable (t j: ℕ)

-- Conditions
def tom_current_age := 25
def tom_future_age_condition := t + 5 = 30
def jared_past_age_condition := j - 2 = 2 * (t - 2)

-- Question
theorem jared_current_age_condition : 
  (t + 5 = 30) ∧ (j - 2 = 2 * (t - 2)) → j = 48 :=
by
  sorry

end jared_current_age_condition_l648_648904


namespace y_works_in_40_days_l648_648931

theorem y_works_in_40_days :
  ∃ d, (d > 0) ∧ 
  (1/20 + 1/d = 3/40) ∧ 
  d = 40 :=
by
  use 40
  sorry

end y_works_in_40_days_l648_648931


namespace parabola_vertex_expression_l648_648293

theorem parabola_vertex_expression (h k : ℝ) :
  (h = 2 ∧ k = 3) →
  ∃ (a : ℝ), (a ≠ 0) ∧
    (∀ x y : ℝ, y = a * (x - h)^2 + k ↔ y = -(x - 2)^2 + 3) :=
by
  sorry

end parabola_vertex_expression_l648_648293


namespace next_term_geometric_sequence_l648_648484

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l648_648484


namespace height_of_minimal_material_l648_648842

-- Define the variables and conditions
variables (x h : ℝ)
variable (V : ℝ) -- Volume
variable (S : ℝ) -- Surface area

-- Conditions: Volume constraint
def volume_constraint (x h : ℝ) := x^2 * h = 4

-- Surface area of the tank (without lid)
def surface_area (x h : ℝ) := x^2 + 4 * x * h

-- The main theorem to prove: height of the tank when minimizing material
theorem height_of_minimal_material {x h : ℝ} (h_vol : volume_constraint x h) :
  (∃ x, surface_area x (4 / x^2) = min (surface_area x (4 / x^2)) → h = 1) := 
sorry

end height_of_minimal_material_l648_648842


namespace greatest_possible_xy_value_l648_648601

-- Define the conditions
variables (a b c d x y : ℕ)
variables (h1 : a < b) (h2 : b < c) (h3 : c < d)
variables (sums : Finset ℕ) (hsums : sums = {189, 320, 287, 234, x, y})

-- Define the goal statement to prove
theorem greatest_possible_xy_value : x + y = 791 :=
sorry

end greatest_possible_xy_value_l648_648601


namespace z_share_per_rupee_x_l648_648983

-- Definitions according to the conditions
def x_gets (r : ℝ) : ℝ := r
def y_gets_for_x (r : ℝ) : ℝ := 0.45 * r
def y_share : ℝ := 18
def total_amount : ℝ := 78

-- Problem statement to prove z gets 0.5 rupees for each rupee x gets.
theorem z_share_per_rupee_x (r : ℝ) (hx : x_gets r = 40) (hy : y_gets_for_x r = 18) (ht : total_amount = 78) :
  (total_amount - (x_gets r + y_share)) / x_gets r = 0.5 := by
  sorry

end z_share_per_rupee_x_l648_648983


namespace hyperbola_eccentricity_l648_648249

theorem hyperbola_eccentricity : 
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e = Real.sqrt 5 / 2 := 
by
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  sorry

end hyperbola_eccentricity_l648_648249


namespace population_doubles_in_approximately_34_66_years_l648_648928

theorem population_doubles_in_approximately_34_66_years : 
  let birth_rate_per_1000 := 39.4 
  let death_rate_per_1000 := 19.4
  let r := (birth_rate_per_1000 - death_rate_per_1000) / 1000 
  r = 0.02 →
  ∀ P₀, 
    ∃ t, 
      (t ≈ 34.66) ∧ 
      (2 * P₀ = P₀ * Real.exp (r * t)) :=
by
  intros birth_rate_per_1000 death_rate_per_1000 r hr P₀
  use Real.log 2 / r
  constructor
  · sorry -- Approximation proof
  · sorry -- Main proof


end population_doubles_in_approximately_34_66_years_l648_648928


namespace no_polynomial_exists_l648_648745

open Polynomial

theorem no_polynomial_exists (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ¬ ∃ (P : ℤ[X]), P.eval a = b ∧ P.eval b = c ∧ P.eval c = a :=
sorry

end no_polynomial_exists_l648_648745


namespace sum_of_primes_1_to_20_l648_648556

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648556


namespace correlation_height_weight_l648_648998

theorem correlation_height_weight : (∃ (h w : ℝ), (h > 0 ∧ w > 0 ∧ correlated h w)) ∧
  (∀ (a : ℝ), ∃ (V : ℝ), ¬correlated V a) ∧
  (∀ (t s : ℝ), ¬correlated s t) ∧
  (∀ (h e : ℝ), ¬correlated h e) :=
by
  -- Define some properties related to correlation
  def correlated (x y : ℝ) : Prop := 
    -- Placeholder definition for correlation (to be defined properly)
    sorry

  -- Conditions:
  have condA : ∀ (a : ℝ), ∃ (V : ℝ), V = a ^ 3 := by
    sorry
  have condB : ∀ (t s : ℝ), ∃ (v : ℝ), s = v * t := by
    sorry
  have condC : ∀ (h e : ℝ), ¬correlated h e := by
    sorry

  -- D: Height and weight correlation is possible
  have condD : ∃ (h w : ℝ), (h > 0 ∧ w > 0 ∧ correlated h w) := by
    sorry

  -- Combining all conditions to the theorem.
  exact ⟨condD, condA, condB, condC⟩

end correlation_height_weight_l648_648998


namespace sum_of_primes_1_to_20_l648_648545

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648545


namespace chocolate_bars_per_box_l648_648462

-- Definitions for the given conditions
def total_chocolate_bars : ℕ := 849
def total_boxes : ℕ := 170

-- The statement to prove
theorem chocolate_bars_per_box : total_chocolate_bars / total_boxes = 5 :=
by 
  -- Proof is omitted here
  sorry

end chocolate_bars_per_box_l648_648462


namespace number_added_is_10_l648_648472

theorem number_added_is_10 (x y a : ℕ) (h1 : y = 40) 
  (h2 : x * 4 = 3 * y) 
  (h3 : (x + a) * 5 = 4 * (y + a)) : a = 10 := 
by
  sorry

end number_added_is_10_l648_648472


namespace list_price_is_30_l648_648608

-- Define the list price x
variable (x : ℝ)

-- Define Alice's and Bob's selling prices and commissions
def Alice_selling_price : ℝ := x - 15
def Alice_commission : ℝ := 0.10 * Alice_selling_price

def Bob_selling_price : ℝ := x - 20
def Bob_commission : ℝ := 0.15 * Bob_selling_price

-- Define the hypothesis that Alice and Bob earn the same amount in commission
def equal_commission (h : Alice_commission = Bob_commission) : Prop := 
  Alice_commission = Bob_commission

-- Define the proof problem
theorem list_price_is_30 (h : equal_commission x) : x = 30 :=
sorry

end list_price_is_30_l648_648608


namespace min_value_squared_sum_l648_648830

variable (p q r s t u v w : ℤ)
variable (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : p ≠ t) 
variable (h5 : p ≠ u) (h6 : p ≠ v) (h7 : p ≠ w)
variable (h8 : q ≠ r) (h9 : q ≠ s) (h10 : q ≠ t) 
variable (h11 : q ≠ u) (h12 : q ≠ v) (h13 : q ≠ w)
variable (h14 : r ≠ s) (h15 : r ≠ t) (h16 : r ≠ u) 
variable (h17 : r ≠ v) (h18 : r ≠ w)
variable (h19 : s ≠ t) (h20 : s ≠ u) (h21 : s ≠ v) 
variable (h22 : s ≠ w)
variable (h23 : t ≠ u) (h24 : t ≠ v) (h25 : t ≠ w)
variable (h26 : u ≠ v) (h27 : u ≠ w)
variable (h28 : v ≠ w)
variable (hs : {p, q, r, s, t, u, v, w} = {-8, -6, -4, -1, 1, 3, 5, 14})

theorem min_value_squared_sum : 
  (∃ p q r s t u v w : ℤ, 
    {p, q, r, s, t, u, v, w} = {-8, -6, -4, -1, 1, 3, 5, 14} ∧
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
     q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
     r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
     s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
     t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
     u ≠ v ∧ u ≠ w ∧ v ≠ w) → 
    let y := p + q + r + s in 
    ∃ (y = 2), (y - 2) ^ 2 * 2 + 8 = 8) 
    sorry

end min_value_squared_sum_l648_648830


namespace circumscribed_sphere_radius_l648_648097

-- Conditions (Lean definitions of the variables and conditions)
variables (a b c : ℝ)

-- Mathematically equivalent proof problem (Lean statement)
theorem circumscribed_sphere_radius (a b c : ℝ) :
  (∃ D A B C : ℝ → Prop, (D A ∧ D B ∧ D C) ∧ (D = a ∧ B = b ∧ C = c) ∧ 
  ((a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)) ∧ 
  (D A ∧ D B ∧ D C ∧ (D A ∧ D B ∧ D C) ∧ 
  (D B ∧ D A ∧ (D C)) ∧ (D C ∧ D A ∧ (D B)) ∧ 
  D A = a ∧ D B = c ∧ D C = b)) → 
  let R : ℝ := (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) 
  in R = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) := 
sorry

end circumscribed_sphere_radius_l648_648097


namespace inequality_solution_l648_648234

theorem inequality_solution (x : ℝ) : (x^2 - 4 * x > 44) ↔ (x ∈ set.Iio (-4) ∪ set.Ioi 11) := 
by 
  sorry

end inequality_solution_l648_648234


namespace beans_problem_l648_648423

theorem beans_problem :
  let total_beans := 572
  let red_beans := total_beans / 4
  let remaining_after_red := total_beans - red_beans
  let white_beans := remaining_after_red / 3
  let remaining_after_white := remaining_after_red - white_beans
  let blue_beans := remaining_after_white / 5
  let remaining_after_blue := remaining_after_white - blue_beans.toNat
  let yellow_beans := remaining_after_blue / 6
  let remaining_after_yellow := remaining_after_blue - yellow_beans.toNat
  let green_beans := remaining_after_yellow / 2
  green_beans.toNat = 95 :=
by
  let total_beans := 572
  let red_beans := total_beans / 4
  let remaining_after_red := total_beans - red_beans
  let white_beans := remaining_after_red / 3
  let remaining_after_white := remaining_after_red - white_beans
  let blue_beans := remaining_after_white / 5
  let remaining_after_blue := remaining_after_white - blue_beans.toNat
  let yellow_beans := remaining_after_blue / 6
  let remaining_after_yellow := remaining_after_blue - yellow_beans.toNat
  let green_beans := remaining_after_yellow / 2
  exact Eq.refl (green_beans.toNat)

end beans_problem_l648_648423


namespace water_left_after_3_hours_l648_648985

-- Define the conditions
def drain_time := 4 -- hours
def hose_time := 6 -- hours
def pool_capacity := 120 -- liters
def starting_volume := pool_capacity
def time_elapsed := 3 -- hours

-- Define the rates based on conditions
def drain_rate := pool_capacity / drain_time
def hose_rate := pool_capacity / hose_time

-- Proof statement
theorem water_left_after_3_hours : 
  starting_volume - drain_rate * time_elapsed + hose_rate * time_elapsed = 90 := 
by 
  -- Drain rate is 30 liters/hour
  have h1: drain_rate = 30, by sorry,
  -- Hose rate is 20 liters/hour
  have h2: hose_rate = 20, by sorry,
  -- Water removed by drain in 3 hours
  have water_removed := drain_rate * time_elapsed,
  -- Water added by hose in 3 hours
  have water_added := hose_rate * time_elapsed,
  -- Net water calculation
  calc
    starting_volume - water_removed + water_added = starting_volume - (drain_rate * time_elapsed) + (hose_rate * time_elapsed) : by sorry 
    ... = 120 - 90 + 60 : by sorry 
    ... = 90 : by sorry

end water_left_after_3_hours_l648_648985


namespace correct_statement_3_correct_answer_l648_648821

variable {α : Type*} [Plane α]
variable {l m n : Line α}

theorem correct_statement_3 (h1 : l ∥ m) (h2 : m ⊥ α) (h3 : n ⊥ α) : l ∥ n := by
  -- Proof yet to be provided
  sorry

theorem correct_answer :
  (∀ (m n : Line α), (m ⊆ α) ∧ (n ⊆ α) → (l ⊥ m) ∧ (l ⊥ n) → ¬ (l ⊥ α)) ∧
  (∀ (m n : Line α), (m ⊆ α) ∧ (n ⊆ α) → ¬ (l ∥ m)) ∧
  (∀ (m n : Line α), (l ∥ m) ∧ (m ⊥ α) ∧ (n ⊥ α) → (l ∥ n)) ∧
  (∀ (m n : Line α), (l ⊥ m) ∧ (l ⊥ n) → ¬ (n ∥ m)) :=
by
  -- Proof yet to be provided
  sorry

end correct_statement_3_correct_answer_l648_648821


namespace inversion_tangent_properties_l648_648851

-- Definitions required from the problem
constant Point : Type
constant Circle : Type
constant Line : Type
constant inversion : Point → Point → Point -- Inversion centered at a point
constant touches : Circle → Line → Point → Prop
constant touchesCircles : Circle → Circle → Point → Prop
constant transformedIntoParallelLines : Circle → Line → Point → Prop
constant transformedIntoParallelLinesCircles : Circle → Circle → Point → Prop

-- Problem statement
theorem inversion_tangent_properties
  (O M : Point)
  (h : M ≠ O)
  (C₁ : Circle) (L : Line)
  (C₂ : Circle)

  (h_touches_CL : touches C₁ L M)
  (h_touches_CC : touchesCircles C₁ C₂ M)
:
  (touches (inversion O (C₁)) (inversion O (L)) (inversion O (M)))
  ∧ (touchesCircles (inversion O (C₁)) (inversion O (C₂)) (inversion O (M)))
  ∧ (transformedIntoParallelLines C₁ L M)
  ∧ (transformedIntoParallelLinesCircles C₁ C₂ M) :=
by
  sorry

end inversion_tangent_properties_l648_648851


namespace axis_of_symmetry_parabola_l648_648429

theorem axis_of_symmetry_parabola :
  ∀ (a b c : ℝ), (a = 1) → (b = -2) → (c = 3) → (-b / (2 * a) = 1) :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  norm_num
  sorry

end axis_of_symmetry_parabola_l648_648429


namespace probability_of_Y_l648_648466

variable (P_X : ℝ) (P_X_and_Y : ℝ) (P_Y : ℝ)

theorem probability_of_Y (h1 : P_X = 1 / 7)
                         (h2 : P_X_and_Y = 0.031746031746031744) :
  P_Y = 0.2222222222222222 :=
sorry

end probability_of_Y_l648_648466


namespace count_squares_below_line_l648_648434

/-!
  Given the line equation: 7 * x + 253 * y = 1771,
  Prove that the number of 1 by 1 squares, entirely within the first quadrant and lying completely below this line, is 756.
-/

theorem count_squares_below_line :
  let line_equation := λ (x y : ℕ), 7 * x + 253 * y = 1771
  ∃ count : ℕ, count = 756 ∧ ∀ (x y : ℕ), line_equation x y → 1 * x * y < 756 := sorry

end count_squares_below_line_l648_648434


namespace exists_same_color_points_at_distance_one_l648_648188

theorem exists_same_color_points_at_distance_one (coloring : ℝ × ℝ → Fin 3) :
  ∃ (p q : ℝ × ℝ), (coloring p = coloring q) ∧ (dist p q = 1) := sorry

end exists_same_color_points_at_distance_one_l648_648188


namespace f_2009_equals_4_l648_648095

open Real

def f (x α β : ℝ) : ℝ := sin (π * x + α) + cos (π * x + β) + 3

theorem f_2009_equals_4 (α β : ℝ) (h : f 2008 α β = 2) : f 2009 α β = 4 :=
  sorry

end f_2009_equals_4_l648_648095


namespace time_for_A_to_complete_work_l648_648163

-- Defining the work rates and the condition
def workRateA (a : ℕ) : ℚ := 1 / a
def workRateB : ℚ := 1 / 12
def workRateC : ℚ := 1 / 24
def combinedWorkRate (a : ℕ) : ℚ := workRateA a + workRateB + workRateC
def togetherWorkRate : ℚ := 1 / 4

-- Stating the theorem
theorem time_for_A_to_complete_work : 
  ∃ (a : ℕ), combinedWorkRate a = togetherWorkRate ∧ a = 8 :=
by
  sorry

end time_for_A_to_complete_work_l648_648163


namespace quadratic_equation_real_roots_l648_648410

theorem quadratic_equation_real_roots (m : ℝ) (h : m ≠ 0) :
  let a := m
  let b := 1 - 5 * m
  let c := -5
  (b^2 - 4 * a * c) ≥ 0 :=
by
  have a := m
  have b := 1 - 5 * m
  have c := -5
  calc
    b^2 - 4 * a * c = (1 - 5 * m)^2 - 4 * m * (-5) : by sorry
               ...  = (5 * m + 1)^2 : by sorry
               ...  ≥ 0 : by sorry

end quadratic_equation_real_roots_l648_648410


namespace sum_of_primes_1_to_20_l648_648552

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648552


namespace range_g_minus_2x_l648_648869

variable (g : ℝ → ℝ)
variable (x : ℝ)

axiom g_values : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → 
  (g x = x ∨ g x = x - 1 ∨ g x = x - 2 ∨ g x = x - 3 ∨ g x = x - 4)

axiom g_le_2x : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → g x ≤ 2 * x

theorem range_g_minus_2x : 
  Set.range (fun x => g x - 2 * x) = Set.Icc (-5 : ℝ) 0 :=
sorry

end range_g_minus_2x_l648_648869


namespace parabola_vertex_expression_l648_648294

theorem parabola_vertex_expression (h k : ℝ) :
  (h = 2 ∧ k = 3) →
  ∃ (a : ℝ), (a ≠ 0) ∧
    (∀ x y : ℝ, y = a * (x - h)^2 + k ↔ y = -(x - 2)^2 + 3) :=
by
  sorry

end parabola_vertex_expression_l648_648294


namespace find_a_minus_b_l648_648440

-- Definition of rotation by 180 degrees around a point (h, k)
def rotate_180 (h k x y : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

-- Definition of reflection about the line y = x
def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

-- The main theorem statement
theorem find_a_minus_b (a b : ℝ) (h k : ℝ) 
  (H : reflect_y_eq_x (rotate_180 h k a b) = (3, -7)) : 
  a - b = 8 :=
sorry

end find_a_minus_b_l648_648440


namespace volume_truncated_triangular_pyramid_l648_648870

def volume_of_truncated_pyramid (H : ℝ) ( α : ℝ) : ℝ :=
  H^3 * (Real.sqrt 3) / (4 * (Real.sin α)^2)

theorem volume_truncated_triangular_pyramid (a b H α : ℝ)
  (H_geometric_mean : H = Real.sqrt (a * b))
  (angle_condition : a - b = H * Real.cot α) :
  volume_of_truncated_pyramid H α = H^3 * (Real.sqrt 3) / (4 * (Real.sin α)^2) :=
sorry

end volume_truncated_triangular_pyramid_l648_648870


namespace canonical_form_lines_l648_648464

theorem canonical_form_lines (x y z : ℝ) :
  (2 * x - y + 3 * z - 1 = 0) →
  (5 * x + 4 * y - z - 7 = 0) →
  (∃ (k : ℝ), x = -11 * k ∧ y = 17 * k + 2 ∧ z = 13 * k + 1) :=
by
  intros h1 h2
  sorry

end canonical_form_lines_l648_648464


namespace tank_capacity_l648_648986

variable (C : ℕ) (t : ℕ)
variable (hC_nonzero : C > 0)
variable (ht_nonzero : t > 0)
variable (h_rate_pipe_A : t = C / 5)
variable (h_rate_pipe_B : t = C / 8)
variable (h_rate_inlet : t = 4 * 60)
variable (h_combined_time : t = 5 + 3)

theorem tank_capacity (C : ℕ) (h1 : C / 5 + C / 8 - 4 * 60 = 8) : C = 1200 := 
by
  sorry

end tank_capacity_l648_648986


namespace xiaohong_distance_l648_648921

variable (t : ℝ) -- Time in minutes
variable (d : ℝ) -- Distance in meters

-- Conditions
def condition1 := d = 120 * (t - 5)
def condition2 := d = 90 * (t + 3)

-- Theorem statement that needs to be proved
theorem xiaohong_distance 
  (h1 : condition1)
  (h2 : condition2) :
  d = 2880 :=
  sorry

end xiaohong_distance_l648_648921


namespace train_speed_l648_648159

/-- Define the length of the train in meters. -/
def train_length : ℝ := 240

/-- Define the time taken to cross the man in seconds. -/
def crossing_time : ℝ := 4

/-- Convert the distance from meters to kilometers. -/
def distance_km : ℝ := train_length / 1000

/-- Convert the time from seconds to hours. -/
def time_hr : ℝ := crossing_time / 3600

/-- Define the expected speed in km/h. -/
def expected_speed : ℝ := 216

/-- Theorem stating the speed of the train given the conditions. -/
theorem train_speed : distance_km / time_hr = expected_speed :=
by
  sorry

end train_speed_l648_648159


namespace pyarelal_loss_l648_648925

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio pyarelal_ratio : ℝ)
  (h1 : ashok_ratio = 1/9) (h2 : pyarelal_ratio = 1)
  (h3 : total_loss = 2000) : (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss = 1800 :=
by
  sorry

end pyarelal_loss_l648_648925


namespace minimum_value_of_ex_4e_negx_l648_648996

theorem minimum_value_of_ex_4e_negx : 
  ∃ (x : ℝ), (∀ (y : ℝ), y = Real.exp x + 4 * Real.exp (-x) → y ≥ 4) ∧ (Real.exp x + 4 * Real.exp (-x) = 4) :=
sorry

end minimum_value_of_ex_4e_negx_l648_648996


namespace exists_island_with_2008_roads_sum_of_roads_l648_648209

-- Define the number of roads per island
def a : ℕ → ℕ
| 1     := 55
| (k+1) := a k + k

-- Function to prove the existence of an island with 2008 roads
theorem exists_island_with_2008_roads : ∃ n : ℕ, a n = 2008 := sorry

-- Function to prove the sum of the number of roads across all islands
theorem sum_of_roads (n : ℕ) : (∑ k in Finset.range (n+1), a k) = 55 * n + choose (n+1) 3 := sorry

end exists_island_with_2008_roads_sum_of_roads_l648_648209


namespace sum_primes_between_1_and_20_l648_648523

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648523


namespace f_cos_x_l648_648324

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (hx : f (Real.sin x) = 2 - Real.cos (2 * x)) :
  f (Real.cos x) = 2 + (Real.cos x)^2 :=
sorry

end f_cos_x_l648_648324


namespace total_samples_correct_l648_648620

-- Define the conditions as constants
def samples_per_shelf : ℕ := 65
def number_of_shelves : ℕ := 7

-- Define the total number of samples and the expected result
def total_samples : ℕ := samples_per_shelf * number_of_shelves
def expected_samples : ℕ := 455

-- State the theorem to be proved
theorem total_samples_correct : total_samples = expected_samples := by
  -- Proof to be filled in
  sorry

end total_samples_correct_l648_648620


namespace probability_of_prime_choice_l648_648901

-- Defining the set of numbers from 1 to 20
def set_20 : set ℕ := {n | 1 ≤ n ∧ n ≤ 20}

-- Defining the set of prime numbers from 1 to 20
def primes_upto_20 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Defining the combination formula
noncomputable def combination (n r : ℕ) : ℕ :=
  (nat.factorial n) / ((nat.factorial r) * (nat.factorial (n - r)))

-- The total number of ways to choose 2 different integers from 1 to 20
noncomputable def total_combinations : ℕ := combination 20 2

-- The number of ways to choose 2 prime numbers from the identified prime numbers
noncomputable def prime_combinations : ℕ := combination 8 2

-- The probability that both numbers chosen are prime
noncomputable def probability_both_prime : ℚ :=
  (prime_combinations : ℚ) / (total_combinations : ℚ)

-- The proof statement
theorem probability_of_prime_choice :
  probability_both_prime = 14 / 95 :=
by
  sorry

end probability_of_prime_choice_l648_648901


namespace total_cost_is_eight_times_short_cost_l648_648029

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l648_648029


namespace non_negative_row_col_sums_l648_648704

variable {m n : ℕ}
variable {A : Matrix (Fin m) (Fin n) ℝ}

theorem non_negative_row_col_sums :
  ∃ A' : Matrix (Fin m) (Fin n) ℝ,
    (∀ i : Fin m, 0 ≤ ∑ j : Fin n, A' i j) ∧
    (∀ j : Fin n, 0 ≤ ∑ i : Fin m, A' i j) :=
begin
  sorry
end

end non_negative_row_col_sums_l648_648704


namespace equations_solution_l648_648861

-- Definition of the conditions
def equation1 := ∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)
def equation2 := ∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)

-- The main statement combining both problems
theorem equations_solution :
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)) := by
  sorry

end equations_solution_l648_648861


namespace smallest_four_digit_divisible_five_smallest_primes_l648_648665

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l648_648665


namespace area_union_of_rotated_triangles_l648_648359

noncomputable def triangle := Type

variables (A B C A' B' C' G : triangle)
variables (AB BC AC : ℝ)
variables (G_centroid : is_centroid G A B C)
variables (A'_image : is_image_of_rotation G A A' (π))
variables (B'_image : is_image_of_rotation G B B' (π))
variables (C'_image : is_image_of_rotation G C C' (π))

theorem area_union_of_rotated_triangles : 
  AB = 7 → BC = 24 → AC = 25 → 
  area (union (triangle_area A B C) (triangle_area A' B' C')) = 84 :=
by
  intros hAB hBC hAC
  sorry

end area_union_of_rotated_triangles_l648_648359


namespace smallest_divisible_four_digit_number_l648_648670

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l648_648670


namespace simplify_expression_l648_648218

theorem simplify_expression :
  sqrt ((-2 : ℝ)^2) + abs (sqrt 2 - sqrt 3) - abs (sqrt 3 - 1) = 3 - sqrt 2 :=
by sorry

end simplify_expression_l648_648218


namespace units_digit_2008_pow_2008_l648_648813

theorem units_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := 
by
  -- The units digits of powers of 8 repeat in a cycle: 8, 4, 2, 6
  -- 2008 mod 4 = 0 which implies it falls on the 4th position in the pattern cycle
  sorry

end units_digit_2008_pow_2008_l648_648813


namespace smallest_value_w3_z3_l648_648697

-- Define the complex numbers w and z and the given conditions.
variables (w z : ℂ)
hypothesis h1 : abs (w + z) = 2
hypothesis h2 : abs (w^2 + z^2) = 15

-- State the theorem for the smallest possible value of |w^3 + z^3|
theorem smallest_value_w3_z3 :
  abs (w^3 + z^3) = 41 :=
sorry

end smallest_value_w3_z3_l648_648697


namespace correct_factorization_l648_648202

theorem correct_factorization : 
  (∃ (t : ℕ → ℕ → Prop), t (m^2 + 2m) (m * (m + 2))) → 
  ¬(t (m^2 - 9) ((m - 3)^2)) ∧ 
  ¬(t (m^2 - m + 1) (m * (m - 1) + 1)) ∧ 
  ¬(t ((m + 1)^2) (m^2 + 2m + 1)) :=
by 
  sorry

end correct_factorization_l648_648202


namespace complement_intersection_eq_l648_648835

open Finset

variable (U M N : Finset ℕ)

def U : Finset ℕ := {0, 1, 2, 3}
def M : Finset ℕ := {0, 1, 2}
def N : Finset ℕ := {1, 2, 3}

theorem complement_intersection_eq {M N U : Finset ℕ} (hU : U = {0, 1, 2, 3}) (hM : M = {0, 1, 2}) (hN : N = {1, 2, 3}) : compl (M ∩ N) U = {0, 3} :=
by
  -- lean proof placeholder
  sorry

end complement_intersection_eq_l648_648835


namespace integral_result_l648_648213

noncomputable def calculate_integral : ℝ := ∫ x in 0..2, x / 2

theorem integral_result : calculate_integral = 1 := by sorry

end integral_result_l648_648213


namespace accommodation_arrangements_l648_648960

-- Given conditions
def triple_room_capacity : Nat := 3
def double_room_capacity : Nat := 2
def single_room_capacity : Nat := 1
def num_adult_men : Nat := 4
def num_little_boys : Nat := 2

-- Ensuring little boys are always accompanied by an adult and all rooms are occupied
def is_valid_arrangement (triple double single : Nat × Nat) : Prop :=
  let (triple_adults, triple_boys) := triple
  let (double_adults, double_boys) := double
  let (single_adults, single_boys) := single
  triple_adults + double_adults + single_adults = num_adult_men ∧
  triple_boys + double_boys + single_boys = num_little_boys ∧
  triple = (triple_room_capacity, num_little_boys) ∨
  (triple = (triple_room_capacity, 1) ∧ double = (double_room_capacity, 1)) ∧
  triple_adults + triple_boys = triple_room_capacity ∧
  double_adults + double_boys = double_room_capacity ∧
  single_adults + single_boys = single_room_capacity

-- Main theorem statement
theorem accommodation_arrangements : ∃ (triple double single : Nat × Nat),
  is_valid_arrangement triple double single ∧
  -- The number 36 comes from the correct answer in the solution steps part b)
  (triple.1 + double.1 + single.1 = 4 ∧ triple.2 + double.2 + single.2 = 2) :=
sorry

end accommodation_arrangements_l648_648960


namespace sum_primes_upto_20_l648_648511

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648511


namespace driver_speed_l648_648587

theorem driver_speed (v t : ℝ) (h1 : t > 0) (h2 : v > 0) (h3 : v * t = (v + 37.5) * (3 / 8) * t) : v = 22.5 :=
by
  sorry

end driver_speed_l648_648587


namespace no_valid_f_values_l648_648829

theorem no_valid_f_values :
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x) * f(y) - f(x * y) = 2 * x + 2 * y) (n s : ℕ) :
  let n := 0 in
  let s := 0 in
  n * s = 0 := 
by
  sorry

end no_valid_f_values_l648_648829


namespace sector_triangle_radii_l648_648600

theorem sector_triangle_radii 
  (r : ℝ) (theta : ℝ) (radius : ℝ) 
  (h_theta_eq: theta = 60)
  (h_radius_eq: radius = 10) :
  let R := (radius * Real.sqrt 3) / 3
  let r_in := (radius * Real.sqrt 3) / 6
  R = 10 * (Real.sqrt 3) / 3 ∧ r_in = 10 * (Real.sqrt 3) / 6 := 
by
  sorry

end sector_triangle_radii_l648_648600


namespace product_of_numbers_is_86_l648_648335

-- Definitions of the two conditions
def sum_eq_24 (x y : ℝ) : Prop := x + y = 24
def sum_of_squares_eq_404 (x y : ℝ) : Prop := x^2 + y^2 = 404

-- The theorem to prove the product of the two numbers
theorem product_of_numbers_is_86 (x y : ℝ) (h1 : sum_eq_24 x y) (h2 : sum_of_squares_eq_404 x y) : x * y = 86 :=
  sorry

end product_of_numbers_is_86_l648_648335


namespace smallest_nonzero_y_satisfying_eq_l648_648632

theorem smallest_nonzero_y_satisfying_eq :
  ∃ (y : ℝ), y ≠ 0 ∧ sqrt (6 * y + 3) = 3 * y + 1 ∧ y = real.sqrt 2 / 3 :=
sorry

end smallest_nonzero_y_satisfying_eq_l648_648632


namespace cans_needed_eq_l648_648404

axiom Paula_initial_rooms : ℕ
axiom Paula_lost_cans : ℕ
axiom Paula_after_loss_rooms : ℕ
axiom cans_for_25_rooms : ℕ

theorem cans_needed_eq :
  Paula_initial_rooms = 30 →
  Paula_lost_cans = 3 →
  Paula_after_loss_rooms = 25 →
  cans_for_25_rooms = 15 :=
by
  intros
  sorry

end cans_needed_eq_l648_648404


namespace max_clocks_l648_648789

theorem max_clocks (n : ℕ) (h m : Fin n → ℕ) (cond1 : ∀ i j, i ≠ j → h i ≠ h j ∧ m i ≠ m j)
  (cond2 : ∀ i, 1 ≤ h i ∧ h i ≤ 12)
  (cond3 : ∀ i, 0 ≤ m i ∧ m i < 60)
  (cond4 : ∑ i, h i - 1 = ∑ i, h i - n)
  (cond5 : ∑ i, m i - 1 = ∑ i, m i - 59) :
  n ≤ 11 :=
by sorry

end max_clocks_l648_648789


namespace metal_roofing_l648_648397

variable (X : ℕ) 

theorem metal_roofing (cond1 : ∀ (r : ℕ), cost_per_foot r = 8)
  (cond2 : supplied_roofing = 250)
  (cond3 : total_payment = 400)
  (H : total_payment = (X - supplied_roofing) * cost_per_foot X) : 
  X = 300 :=
sorry

end metal_roofing_l648_648397


namespace marcel_corn_l648_648394

theorem marcel_corn (C : ℕ) (H1 : ∃ D, D = C / 2) (H2 : 27 = C + C / 2 + 8 + 4) : C = 10 :=
sorry

end marcel_corn_l648_648394


namespace triangle_tangent_relation_l648_648807

variable {A B C : ℝ}

theorem triangle_tangent_relation (h_triangle: ∀x y z : ℝ, x + y + z = π) (h_tangent_relation: tan A * tan B = tan A * tan C + tan C * tan B) :
  ∃ a b c : ℝ, (a^2 + b^2) / c^2 = 3 :=
by
  sorry

end triangle_tangent_relation_l648_648807


namespace second_friend_shells_l648_648814

theorem second_friend_shells (initial_shells : ℕ) (first_friend_shells : ℕ) (total_shells : ℕ) (second_friend_shells : ℕ) :
  initial_shells = 5 → first_friend_shells = 15 → total_shells = 37 → initial_shells + first_friend_shells + second_friend_shells = total_shells → second_friend_shells = 17 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end second_friend_shells_l648_648814


namespace triangle_inequality_l648_648063

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := 
sorry

end triangle_inequality_l648_648063


namespace sum_of_primes_1_to_20_l648_648530

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648530


namespace smallest_four_digit_divisible_by_primes_l648_648653

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l648_648653


namespace total_weight_of_new_individuals_l648_648957

theorem total_weight_of_new_individuals (avg_weight_original : ℝ)
    (num_people_original : ℕ) (weight_leave1 weight_leave2 weight_leave3 : ℝ)
    (new_avg_weight : ℝ) (num_new_people : ℕ) :
    avg_weight_original = 70 → num_people_original = 15 →
    weight_leave1 = 60 → weight_leave2 = 65 → weight_leave3 = 68 →
    new_avg_weight = avg_weight_original + 1.5 →
    num_new_people = 15 →
    ∃ (weight_new_people : ℝ), weight_new_people = 215.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  have h_sum : avg_weight_original * num_people_original = 1050 := by linarith [h1, h2]
  have h_leave : weight_leave1 + weight_leave2 + weight_leave3 = 193 := by linarith [h3, h4, h5]
  have h_remain : 1050 - 193 = 857 := by linarith [h_sum, h_leave]
  have h_new_total : new_avg_weight * num_new_people = 1072.5 := by linarith [h6, h7]
  have h_new_people : 1072.5 - 857 = 215.5 := by linarith [h_new_total, h_remain]
  exact ⟨215.5, h_new_people⟩

end total_weight_of_new_individuals_l648_648957


namespace allocation_schemes_l648_648461

theorem allocation_schemes (A B : Type) [fintype A] [fintype B] (hAB : fintype.card A = 5) :
  (C (fintype.card A) 2 - 4) * 2 = 12 := 
sorry

end allocation_schemes_l648_648461


namespace flatville_additional_plates_max_count_l648_648677

noncomputable def flatville_initial_plate_count : Nat :=
  6 * 4 * 5

noncomputable def flatville_max_plate_count : Nat :=
  6 * 6 * 6

theorem flatville_additional_plates_max_count : flatville_max_plate_count - flatville_initial_plate_count = 96 :=
by
  sorry

end flatville_additional_plates_max_count_l648_648677


namespace no_integer_points_strictly_between_C_and_D_l648_648963

def points_strictly_between_integer_coordinates (C D : ℤ × ℤ) : ℕ :=
  let m := ((D.2 - C.2) : ℚ) / ((D.1 - C.1) : ℚ)
  let b := C.2 - m * C.1
  by
    let x_min := min C.1 D.1
    let x_max := max C.1 D.1
    have key_eq : ∀ x : ℤ, (x > x_min ∧ x < x_max) → (∃ y : ℤ, y = m * x + b) → false
    { 
      sorry
    }
    exact 0


theorem no_integer_points_strictly_between_C_and_D :
    points_strictly_between_integer_coordinates (2, 3) (101, 200) = 0 :=
  by
    rw [points_strictly_between_integer_coordinates]
    sorry

end no_integer_points_strictly_between_C_and_D_l648_648963


namespace cos_theta_minus_pi_over_6_l648_648713

variable (θ : ℝ)

theorem cos_theta_minus_pi_over_6 (h1 : cos θ = -5/13) (h2 : θ ∈ Ioo π (3 * π / 2)) :
  cos (θ - π / 6) = -((5 * Real.sqrt 3 + 12) / 26) :=
by
  sorry

end cos_theta_minus_pi_over_6_l648_648713


namespace maria_fraction_of_remaining_distance_l648_648238

theorem maria_fraction_of_remaining_distance (total_distance remaining_distance distance_travelled : ℕ) 
(h_total : total_distance = 480) 
(h_first_stop : distance_travelled = total_distance / 2) 
(h_remaining : remaining_distance = total_distance - distance_travelled)
(h_final_leg : remaining_distance - distance_travelled = 180) : 
(distance_travelled / remaining_distance) = (1 / 4) := 
by
  sorry

end maria_fraction_of_remaining_distance_l648_648238


namespace midpoint_equidistant_l648_648266

variable {Point : Type} [EuclideanGeometry Point] 

variables (A B C D M : Point)
-- Define the conditions as given in the problem
variable (Angle : Angle Point)
variable (AngleBisector : Line Point)
variable (PerpendicularA : Line Point)
variable (PerpendicularB : Line Point)
variable (MidpointCD : Point)

-- Assume the necessary conditions are defined correctly
axiom h1 : PerpendicularTo A AngleBisector PerpendicularA
axiom h2 : PerpendicularTo B AngleBisector PerpendicularB
axiom h3 : Intersection PerpendicularA AngleBisector C
axiom h4 : Intersection PerpendicularB AngleBisector D
axiom h5 : MidpointOfSegment M C D

theorem midpoint_equidistant (h1 : PerpendicularTo A AngleBisector PerpendicularA)
                           (h2 : PerpendicularTo B AngleBisector PerpendicularB)
                           (h3 : Intersection PerpendicularA AngleBisector C)
                           (h4 : Intersection PerpendicularB AngleBisector D)
                           (h5 : MidpointOfSegment M C D) :
  dist M A = dist M B :=
sorry

end midpoint_equidistant_l648_648266


namespace value_of_s_l648_648634

theorem value_of_s (s : ℝ) (h : 9 = 3^(2 * s + 2)) : s = 0 :=
sorry

end value_of_s_l648_648634


namespace range_of_a1_l648_648075

theorem range_of_a1 (a1 : ℝ) :
  (∃ (a2 a3 : ℝ), 
    ((a2 = 2 * a1 - 12) ∨ (a2 = a1 / 2 + 12)) ∧
    ((a3 = 2 * a2 - 12) ∨ (a3 = a2 / 2 + 12)) ) →
  ((a3 > a1) ↔ ((a1 ≤ 12) ∨ (24 ≤ a1))) :=
by
  sorry

end range_of_a1_l648_648075


namespace total_cost_is_eight_x_l648_648059

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l648_648059


namespace grocery_store_kinds_of_jelly_l648_648174

noncomputable def grape_jelly_sold := 36
noncomputable def strawberry_jelly_sold := 18
noncomputable def raspberry_jelly_sold := 12
noncomputable def plum_jelly_sold := 6

def condition_1 := grape_jelly_sold = 2 * strawberry_jelly_sold
def condition_2 := raspberry_jelly_sold = 2 * plum_jelly_sold
def condition_3 := raspberry_jelly_sold = grape_jelly_sold / 3

theorem grocery_store_kinds_of_jelly (h1: condition_1) (h2: condition_2) (h3: condition_3) : 
    ∃ kinds_of_jelly, kinds_of_jelly = 4 :=
begin
    use 4,
    sorry,
end

end grocery_store_kinds_of_jelly_l648_648174


namespace sum_of_primes_between_1_and_20_l648_648496

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648496


namespace square_geometry_problem_l648_648849

theorem square_geometry_problem
  (F : ℝ) (AD DF CF CE BE : ℝ)
  (H1 : DF = (1 / 3) * AD)
  (H2 : AD = √144)
  (H3 : CF = AD - DF)
  (H4 : 90 = (1/2) * CF * CE)
  (H5 : BE = √(12^2 + CE^2)) :
  BE = 25.5 :=
sorry

end square_geometry_problem_l648_648849


namespace eval_polynomial_at_3_l648_648244

theorem eval_polynomial_at_3 : (3 : ℤ) ^ 3 + (3 : ℤ) ^ 2 + 3 + 1 = 40 := by
  sorry

end eval_polynomial_at_3_l648_648244


namespace probability_different_colors_l648_648578

/-- A bag contains 7 blue chips, 5 yellow chips, and 2 red chips. A chip is 
drawn from the bag and then placed back into the bag. A second chip is then drawn.
What is the probability that the two selected chips are of different colors?
We aim to prove that this probability is \(\dfrac{59}{98}\). -/
theorem probability_different_colors :
  let blue_chips := 7
  let yellow_chips := 5
  let red_chips := 2
  let total_chips := blue_chips + yellow_chips + red_chips
  let prob_diff_colors := 
    (blue_chips / total_chips * yellow_chips / total_chips) +
    (blue_chips / total_chips * red_chips / total_chips) +
    (yellow_chips / total_chips * blue_chips / total_chips) +
    (yellow_chips / total_chips * red_chips / total_chips) +
    (red_chips / total_chips * blue_chips / total_chips) +
    (red_chips / total_chips * yellow_chips / total_chips)
  in
  prob_diff_colors = 59 / 98 :=
by
  let blue_chips := 7
  let yellow_chips := 5
  let red_chips := 2
  let total_chips := blue_chips + yellow_chips + red_chips
  let prob_diff_colors :=
    (blue_chips / total_chips * yellow_chips / total_chips) +
    (blue_chips / total_chips * red_chips / total_chips) +
    (yellow_chips / total_chips * blue_chips / total_chips) +
    (yellow_chips / total_chips * red_chips / total_chips) +
    (red_chips / total_chips * blue_chips / total_chips) +
    (red_chips / total_chips * yellow_chips / total_chips)
  have: prob_diff_colors = 59 / 98 := sorry
  exact this

end probability_different_colors_l648_648578


namespace diameter_of_outer_boundary_l648_648952

theorem diameter_of_outer_boundary
  (pond_diameter : ℝ)
  (seating_area_width : ℝ)
  (garden_width : ℝ)
  (walking_path_width : ℝ) :
  pond_diameter = 20 ∧ seating_area_width = 4 ∧ garden_width = 10 ∧ walking_path_width = 8 →
  let pond_radius := pond_diameter / 2 in
  let total_additional_width := seating_area_width + garden_width + walking_path_width in
  let total_radius := pond_radius + total_additional_width in
  let outer_diameter := 2 * total_radius in
  outer_diameter = 64 :=
begin
  intros h,
  rcases h with ⟨hp, hs, hg, hw⟩,
  sorry
end

end diameter_of_outer_boundary_l648_648952


namespace trader_sold_40_meters_l648_648988

noncomputable def meters_of_cloth_sold (profit_per_meter total_profit : ℕ) : ℕ :=
  total_profit / profit_per_meter

theorem trader_sold_40_meters (profit_per_meter total_profit : ℕ) (h1 : profit_per_meter = 35) (h2 : total_profit = 1400) :
  meters_of_cloth_sold profit_per_meter total_profit = 40 :=
by
  sorry

end trader_sold_40_meters_l648_648988


namespace floor_of_ceiling_of_fraction_squared_plus_another_fraction_l648_648243

theorem floor_of_ceiling_of_fraction_squared_plus_another_fraction :
  (Real.floor (Real.ceil ((15 / 8)^2) + (20 / 3))) = 10 := by
  sorry

end floor_of_ceiling_of_fraction_squared_plus_another_fraction_l648_648243


namespace point_2008_in_third_quadrant_l648_648441

noncomputable def point_in_third_quadrant (θ : ℝ) : Prop :=
  let P := (Real.cos θ, Real.sin θ) in
  P.1 < 0 ∧ P.2 < 0

theorem point_2008_in_third_quadrant : point_in_third_quadrant 2008 :=
sorry

end point_2008_in_third_quadrant_l648_648441


namespace place_squares_area_two_l648_648284

theorem place_squares_area_two (squares : list ℝ) (h : ∀ s ∈ squares, s > 0) (total_area : squares.sum (λ s, s^2) = 1) :
  ∃ (x : ℝ), x > 0 ∧ x^2 = 2 ∧ nonempty (set (ℝ × ℝ) → set (ℝ × ℝ)) :=
sorry

end place_squares_area_two_l648_648284


namespace no_valid_7x7_table_l648_648932

def valid_table (f : ℕ → ℕ → ℕ) (n : ℕ) := 
  (∀ i j, 1 ≤ i ∧ i ≤ n - 1 ∧ 1 ≤ j ∧ j ≤ n - 1 → (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 2 = 1) ∧
  (∀ i j, 1 ≤ i ∧ i ≤ n - 2 ∧ 1 ≤ j ∧ j ≤ n - 2 → (list.sum [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)].map (λ (p: ℕ × ℕ), f (i + p.1) (j + p.2))) % 2 = 1)

theorem no_valid_7x7_table : ¬ (∃ f : ℕ → ℕ → ℕ, valid_table f 7) :=
sorry

end no_valid_7x7_table_l648_648932


namespace solve_congruence_l648_648419

theorem solve_congruence : ∃ (a m : ℕ), 10 * x + 3 ≡ 7 [MOD 18] ∧ x ≡ a [MOD m] ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := 
sorry

end solve_congruence_l648_648419


namespace find_age_of_15th_student_l648_648146
-- Importing the necessary library

noncomputable def age_of_15th_student
  (avg_age_all : ℝ) (n_all : ℕ)
  (avg_age_8 : ℝ) (n_8 : ℕ)
  (avg_age_6 : ℝ) (n_6 : ℕ) : ℝ :=
  let total_age_all := n_all * avg_age_all in
  let total_age_8 := n_8 * avg_age_8 in
  let total_age_6 := n_6 * avg_age_6 in
  total_age_all - (total_age_8 + total_age_6)

theorem find_age_of_15th_student
  (avg_age_all : ℝ) (h1 : avg_age_all = 15)
  (n_all : ℕ) (h2 : n_all = 15)
  (avg_age_8 : ℝ) (h3 : avg_age_8 = 14)
  (n_8 : ℕ) (h4 : n_8 = 8)
  (avg_age_6 : ℝ) (h5 : avg_age_6 = 16)
  (n_6 : ℕ) (h6 : n_6 = 6) :
  age_of_15th_student avg_age_all n_all avg_age_8 n_8 avg_age_6 n_6 = 17 :=
by
  have total_age_all := 15 * 15
  have total_age_8 := 8 * 14
  have total_age_6 := 6 * 16
  have total_age_others := total_age_all - (total_age_8 + total_age_6)
  simp [age_of_15th_student, total_age_all, total_age_8, total_age_6]
  exact calc
    total_age_all - (total_age_8 + total_age_6)
        = 225 - (112 + 96) : by norm_num
    ... = 225 - 208       : by norm_num
    ... = 17              : by norm_num

end find_age_of_15th_student_l648_648146


namespace largest_vertex_sum_l648_648681

noncomputable def vertex_sum (a T : ℤ) (hT : T ≠ 0) : ℤ :=
  let x_vertex := (3 * T) / 2
  let y_vertex := -((3 * a * (T ^ 2)) / 4)
  x_vertex + y_vertex

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0) (h_passing_A : (0,0) ∈ set (λ x, a * x * (x - 3 * T))) 
                          (h_passing_B : (3 * T, 0) ∈ set (λ x, a * x * (x - 3 * T))) 
                          (h_passing_C : (3 * T + 1, 35) ∈ set (λ x, a * x * (x - 3 * T))) :
  ∃ T a, vertex_sum a T hT = 3 :=
by sorry

end largest_vertex_sum_l648_648681


namespace circumcircle_contains_midpoints_l648_648442

variable {Point : Type}
variables {A B C I I_a I_b I_c : Point}
variables [triangle : IsTriangle A B C] [incenter : IsIncenter I A B C]
variables [excenterA : IsExcenter I_a B C] [excenterB : IsExcenter I_b A C] [excenterC : IsExcenter I_c A B]

theorem circumcircle_contains_midpoints :
  circumcircle A B C ∈
    { M : Point |
      (is_midpoint_of_side M I_a I_b I_c) ∨
      (is_midpoint_of_segment M I I_a) ∨
      (is_midpoint_of_segment M I I_b) ∨
      (is_midpoint_of_segment M I I_c) } :=
sorry

end circumcircle_contains_midpoints_l648_648442


namespace train_pass_time_is_38_seconds_l648_648176

noncomputable def speed_of_jogger_kmhr : ℝ := 9
noncomputable def speed_of_train_kmhr : ℝ := 45
noncomputable def lead_distance_m : ℝ := 260
noncomputable def train_length_m : ℝ := 120

noncomputable def speed_of_jogger_ms : ℝ := speed_of_jogger_kmhr * (1000 / 3600)
noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmhr * (1000 / 3600)

noncomputable def relative_speed_ms : ℝ := speed_of_train_ms - speed_of_jogger_ms
noncomputable def total_distance_m : ℝ := lead_distance_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_m / relative_speed_ms

theorem train_pass_time_is_38_seconds :
  time_to_pass_jogger_s = 38 := 
sorry

end train_pass_time_is_38_seconds_l648_648176


namespace sum_primes_upto_20_l648_648513

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648513


namespace cosine_product_formula_l648_648225

theorem cosine_product_formula (α : ℝ) (n : ℕ)
  (h : ∀ β : ℝ, sin (2 * β) = 2 * sin β * cos β) :
  ∏ k in finset.range(n + 1), cos (2^k * α) = (1 / 2^n) * (sin (2^(n + 1) * α) / sin α) :=
by
  sorry

end cosine_product_formula_l648_648225


namespace trigonometric_identity_proof_l648_648074

-- Definitions for given conditions
def sin_45 : ℝ := Real.sin (Real.pi / 4)
def sin_105 : ℝ := Real.sin (105 * Real.pi / 180)
def sin_15 : ℝ := Real.sin (15 * Real.pi / 180)
-- Using trigonometric identity to find the value
def lhs : ℝ := sin_45 * sin_105 + sin_45 * sin_15
def rhs : ℝ := Real.sqrt 3 / 2

-- Proof statement
theorem trigonometric_identity_proof : lhs = rhs := 
  sorry

end trigonometric_identity_proof_l648_648074


namespace average_rate_of_change_x_squared_l648_648085

theorem average_rate_of_change_x_squared :
  let f : ℝ → ℝ := λ x, x^2,
      a : ℝ := 1,
      b : ℝ := 2 in
  (f b - f a) / (b - a) = 3 :=
by 
  let f : ℝ → ℝ := λ x, x^2,
      a : ℝ := 1,
      b : ℝ := 2 in
  sorry

end average_rate_of_change_x_squared_l648_648085


namespace intersection_M_P_l648_648312

noncomputable def M : Set ℝ := {y | ∃ x: ℝ, y = 2^x ∧ x < -1}
noncomputable def P : Set ℝ := {y | ∃ x: ℝ, y = Real.log2 x ∧ x ≥ 1}

theorem intersection_M_P :
  M ∩ P = {y | 0 < y ∧ y < 1/2} :=
by
  sorry

end intersection_M_P_l648_648312


namespace maximum_value_of_a_squared_b_l648_648709

theorem maximum_value_of_a_squared_b {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * (a + b) = 27) : 
  a^2 * b ≤ 54 :=
sorry

end maximum_value_of_a_squared_b_l648_648709


namespace sum_of_coefficients_l648_648349

theorem sum_of_coefficients (n : ℕ) : 
  let P := (1 - 2 * (1 : ℝ))^n 
  in P = if even n then 1 else -1 :=
by {
  unfold even,
  unfold odd,
  sorry
}

end sum_of_coefficients_l648_648349


namespace smallest_divisible_four_digit_number_l648_648669

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l648_648669


namespace bob_distance_walked_l648_648145

theorem bob_distance_walked
    (dist : ℕ)
    (yolanda_rate : ℕ)
    (bob_rate : ℕ)
    (hour_diff : ℕ)
    (meet_time_bob: ℕ) :

    dist = 31 → yolanda_rate = 1 → bob_rate = 2 → hour_diff = 1 → meet_time_bob = 10 →
    (bob_rate * meet_time_bob) = 20 :=
by
  intros
  sorry

end bob_distance_walked_l648_648145


namespace tangent_line_at_origin_range_of_k_for_monotonicity_l648_648737

noncomputable def f (x k : ℝ) : ℝ := x * Real.exp (k * x)

theorem tangent_line_at_origin (k : ℝ) (h : k ≠ 0) : 
  let f := f
  f 0 k = 0 ∧
  ∀ x : ℝ, deriv (λ x, f x k) x = Real.exp (k * x) * (1 + k * x) ∧ deriv (λ x, f x k) 0 = 1 ∧
  y = x := 
sorry

theorem range_of_k_for_monotonicity (k : ℝ) (h : k ≠ 0) :
  (k ∈ Set.Icc (-1 : ℝ) 1 \ Set.singleton 0) ↔ 
  ∀ x y ∈ Set.Ioc (-1 : ℝ) 1, x < y → f x k < f y k :=
sorry

end tangent_line_at_origin_range_of_k_for_monotonicity_l648_648737


namespace eccentricity_e_tangent_line_l648_648281

variables {a b c : ℝ}

-- Conditions
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def ellipse_major_minor_relation : Prop := a = √3 * b
def a_gt_b : Prop := a > b
def b_gt_0 : Prop := b > 0
def focal_distance : ℝ := 2 * √2
def focal_relation : Prop := c = √(a^2 - b^2)

-- Questions formulated as Lean statements
theorem eccentricity_e {a b : ℝ} (h1 : ellipse_major_minor_relation)
                       (h2 : a_gt_b) (h3 : b_gt_0) : 
                       (c/a = √6/3) :=
sorry

theorem tangent_line {a b : ℝ} (h1 : ellipse_major_minor_relation)
                     (h2 : a_gt_b) (h3 : b_gt_0) (h4 : focal_relation) (h5 : focal_distance) :
                     (∀ l, (∃ P Q, ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ l_intersects PQ ∧ OP_perp OQ) →
                      (l_tangent_to_circle l x^2+y^2=3/4)) :=
sorry

-- Helper functions to define properties of lines, intersections, perpendicularity, and tangency
def l_intersects (P Q : ℝ × ℝ) (l : Prop) : Prop := sorry
def OP_perp (O : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop := sorry
def l_tangent_to_circle (l : Prop) (circle_eq : Prop) : Prop := sorry

end eccentricity_e_tangent_line_l648_648281


namespace determine_alpha_l648_648267

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x > f y

def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem determine_alpha
  (α : ℝ)
  (hα : α ∈ ({-2, -1, -1/2, 1/2, 1, 2, 3} : set ℝ))
  (hf_odd : is_odd_function (power_function α))
  (hf_decreasing : is_decreasing_on (power_function α) (set.Ioi 0)) :
  α = -1 :=
sorry

end determine_alpha_l648_648267


namespace marble_selection_l648_648752

/-- 
The number of ways to choose 3 marbles from a bag of 8 (numbered from 1 to 8) 
such that their sum equals the number on a marble chosen from a bag of 10 (numbered from 1 to 10) 
is exactly 90.
-/
theorem marble_selection (bag1 : Finset ℕ) (bag2 : Finset ℕ) (choose3: Finset ℕ → Finset (Finset ℕ)) (choose1: Finset ℕ → Finset ℕ):
  bag1 = {1, 2, 3, 4, 5, 6, 7, 8} →
  bag2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} →
  (∑ s in choose3 bag1, (sum s = 1) →
  (s ∈ choose1 bag2)) = 90 := 
sorry

end marble_selection_l648_648752


namespace b_2023_eq_l648_648385

noncomputable def b : ℕ → ℚ
| 0     := 5  -- Using 0-based indexing here, equivalent to b_1 in problem
| 1     := 7  -- Equivalent to b_2
| n + 2 := b (n + 1) / b n

theorem b_2023_eq : b 2022 = 1 / 7 :=  -- b_2023 corresponds to b 2022 in 0-based indexing
by sorry

end b_2023_eq_l648_648385


namespace sum_primes_between_1_and_20_l648_648504

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648504


namespace isosceles_triangle_circumcircle_area_l648_648165

-- Define the conditions of the problem
structure IsoscelesTriangle :=
(a b c : ℝ)
(h : a = 6 ∧ b = 6 ∧ c = 4)

-- Define the function that computes the area of the circle
def area_of_circumcircle (Δ : IsoscelesTriangle) : ℝ :=
  π * (4 * 4)

-- Define the theorem that relates to the given problem
theorem isosceles_triangle_circumcircle_area
    (Δ : IsoscelesTriangle)
    (h : Δ.a = 6 ∧ Δ.b = 6 ∧ Δ.c = 4) :
  area_of_circumcircle Δ = 16 * π :=
by sorry

end isosceles_triangle_circumcircle_area_l648_648165


namespace find_a_l648_648728

variable (a x : ℝ)

noncomputable def curve1 (x : ℝ) := x + Real.log x
noncomputable def curve2 (a x : ℝ) := a * x^2 + (a + 2) * x + 1

theorem find_a : (curve1 1 = 1 ∧ curve1 1 = curve2 a 1) → a = 8 :=
by
  sorry

end find_a_l648_648728


namespace exists_close_points_l648_648783

-- Definitions for the rectangle and points conditions
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  width : ℝ
  height : ℝ
  points : List Point

-- Provided a rectangle and points conditions
def exampleRectangle : Rectangle := 
  { width := 3, height := 4, points := [ 
      { x := 0.0, y := 0.0 },
      { x := 1.0, y := 1.0 },
      { x := 2.0, y := 2.0 },
      { x := 2.5, y := 3.5 },
      { x := 1.5, y := 1.5 },
      { x := 3.0, y := 3.0 }
    ] }

-- The theorem statement
theorem exists_close_points (r : Rectangle) (h : r.width = 3 ∧ r.height = 4 ∧ r.points.length = 6) :
  ∃ p1 p2 ∈ r.points, p1 ≠ p2 ∧ dist p1 p2 ≤ √2 := sorry

end exists_close_points_l648_648783


namespace wood_length_equation_l648_648940

-- Define the conditions as hypotheses
def length_of_wood_problem (x : ℝ) :=
  (1 / 2) * (x + 4.5) = x - 1

-- Now we state the theorem we want to prove, which is equivalent to the question == answer
theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l648_648940


namespace midpoint_is_correct_l648_648700

noncomputable def midpoint_of_intersection : (ℝ × ℝ) :=
  let line := λ t : ℝ, (1 + 1/2 * t, -3 * Real.sqrt 3 + Real.sqrt 3 / 2 * t)
  let circle := λ p : ℝ × ℝ, p.1 ^ 2 + p.2 ^ 2 = 16
  let standard_line := λ x : ℝ, Real.sqrt 3 * x - 4 * Real.sqrt 3
  let intersection_x := {x : ℝ // x^2 - 6*x + 8 = 0}
  let midpoint_x := (Sum.elim (λ h : intersection_x, h.val) (λ h : intersection_x, h.val) : ℝ) / 2
  (midpoint_x, standard_line midpoint_x)

theorem midpoint_is_correct : midpoint_of_intersection = (3, -Real.sqrt 3) := 
by 
  sorry

end midpoint_is_correct_l648_648700


namespace number_of_symmetric_scanning_codes_l648_648978

def is_symmetric (code : Matrix (Fin 5) (Fin 5) Bool) : Prop :=
  ∀ (rot : {n // n < 4}), rotate90 code n = code ∧
  code = reflect_diagonals code ∧
  code = reflect_vertical_midpoints code ∧
  code = reflect_horizontal_midpoints code

theorem number_of_symmetric_scanning_codes : 
  (∃ code : Matrix (Fin 5) (Fin 5) Bool, 
      0 < ∑ i j, if code i j then 1 else 0 ∧ 
      ∑ i j, if code i j then 1 else 0 < 25 ∧ 
      is_symmetric code) = 30 := 
sorry

end number_of_symmetric_scanning_codes_l648_648978


namespace value_of_m_l648_648393

variable {U A : Set ℕ}

theorem value_of_m (m : ℕ) (U : Set ℕ) (A : Set ℕ) (hU : U = {2, 4, 6, 8}) (hA : A = {2, |m - 6|}) (hSubset : A ⊆ U) (hComp : U \ A = {6, 8}) :
  m = 10 ∨ m = 2 :=
by
  sorry

end value_of_m_l648_648393


namespace find_x_angle_l648_648348

theorem find_x_angle
  (A B C : Type)
  (angle_ABC : Real) 
  (isosceles_ABC : A = C)
  (angle_BAC_x : 180 - x)
  (angle_ABC_val : angle_ABC = 30) :
  x = 105 :=
by
  sorry

end find_x_angle_l648_648348


namespace find_a3_and_a37_l648_648065

noncomputable def sequence_condition (a : Fin 37 → Nat) : Prop :=
  a 0 = 37 ∧ a 1 = 1 ∧
  (∀ k : Fin 36, (∑ i in Fin.range (k + 1), a i) % a (k + 1) = 0)

theorem find_a3_and_a37 (a : Fin 37 → Nat) (h : sequence_condition a) : 
  a 2 = 2 ∧ a 36 = 19 := 
sorry

end find_a3_and_a37_l648_648065


namespace tax_percentage_first_40000_l648_648781

theorem tax_percentage_first_40000 (P : ℝ) :
  (0 < P) → 
  (P / 100) * 40000 + 0.20 * 10000 = 8000 →
  P = 15 :=
by
  intros hP h
  sorry

end tax_percentage_first_40000_l648_648781


namespace quadratic_function_is_explicit_form_l648_648741

-- Conditions
variable {f : ℝ → ℝ}
variable (H1 : f (-1) = 0)
variable (H2 : ∀ x : ℝ, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2)

-- The quadratic function we aim to prove
def quadratic_function_form_proof (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (1/4) * x^2 + (1/2) * x + (1/4)

-- Main theorem statement
theorem quadratic_function_is_explicit_form : quadratic_function_form_proof f :=
by
  -- Placeholder for the proof
  sorry

end quadratic_function_is_explicit_form_l648_648741


namespace sum_primes_between_1_and_20_l648_648521

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648521


namespace number_of_digits_s99_l648_648927

def s (n : ℕ) : ℕ :=
  let squares := List.map (fun x => x * x) (List.range n.succ)
  List.foldl (fun acc x => acc * 10^(Nat.numDigits 10 x) + x) 0 squares

example : s(1) = 1 := by rfl
example : s(2) = 14 := by rfl
example : s(3) = 149 := by rfl
example : s(4) = 14916 := by rfl
example : s(5) = 1491625 := by rfl

theorem number_of_digits_s99 : ∃ y, y = 189 ∧ Nat.numDigits 10 (s 99) = y := 
by
  sorry

end number_of_digits_s99_l648_648927


namespace solution_set_of_inequality_l648_648227

def f (x : ℝ) : ℝ :=
  if x < 0 then x + 6 else x^2 - 4 * x + 6

theorem solution_set_of_inequality :
  (∀ x : ℝ, (f 1 = 3) → (f x > 3 ↔ ((-3 < x ∧ x < 1) ∨ x > 3))) :=
by
  intro x h
  sorry

end solution_set_of_inequality_l648_648227


namespace acute_triangle_reconstruction_impossible_l648_648400

theorem acute_triangle_reconstruction_impossible
  (A B C A1 B1 C1 : Type) [A1 A B]
  (acute_angle : ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 180)
  (altitudes_feet : ∃ (A1 B1 : Type), B1 = B ∧ A1 = A)
  (midpoint_C1 : ∃ (C1 : Type), C1 = (A + B) / 2) :
  ¬ (∃ (A' B' C' : Type), A' = A ∧ B' = B ∧ C' = C) := 
sorry

end acute_triangle_reconstruction_impossible_l648_648400


namespace necessary_but_not_sufficient_l648_648062

variable (p q : Prop)
-- Condition p: The base of a right prism is a rhombus.
def base_of_right_prism_is_rhombus := p
-- Condition q: A prism is a right rectangular prism.
def prism_is_right_rectangular := q

-- Proof: p is a necessary but not sufficient condition for q.
theorem necessary_but_not_sufficient (p q : Prop) 
  (h1 : base_of_right_prism_is_rhombus p)
  (h2 : prism_is_right_rectangular q) : 
  (q → p) ∧ ¬ (p → q) :=
sorry

end necessary_but_not_sufficient_l648_648062


namespace find_a5_l648_648339

variable {a_n : ℕ → ℝ}
variable [GeometricSequence a_n]
variable {n : ℕ}
variable (a_pos : ∀ n, 0 < a_n)
variable (H : a_n 2 * a_n 8 + a_n 3 * a_n 7 = 32)

theorem find_a5 : a_n 5 = 4 :=
by
  sorry

end find_a5_l648_648339


namespace sum_of_primes_1_to_20_l648_648534

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648534


namespace essentially_different_proportions_l648_648319

theorem essentially_different_proportions (x y z t : α) [DecidableEq α] 
  (h1 : x ≠ y) (h2 : x ≠ z) (h3 : x ≠ t) (h4 : y ≠ z) (h5 : y ≠ t) (h6 : z ≠ t) : 
  ∃ n : ℕ, n = 3 := by
  sorry

end essentially_different_proportions_l648_648319


namespace matrix_product_l648_648222

def A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![1, 2, 0], 
    ![4, -1, 3], 
    ![5, 0, 6]]

def B : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![1, 0, 4], 
    ![0, 3, 0], 
    ![2, 1, 5]]

def C : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![1, 6, 4], 
    ![10, -3, 29], 
    ![17, 6, 50]]

theorem matrix_product : A ⬝ B = C := 
  by sorry

end matrix_product_l648_648222


namespace birds_joined_l648_648575

variable (initialBirds : ℕ) (totalBirds : ℕ)

theorem birds_joined (h1 : initialBirds = 2) (h2 : totalBirds = 6) : (totalBirds - initialBirds) = 4 :=
by
  sorry

end birds_joined_l648_648575


namespace log_sum_max_value_l648_648722

open Real

theorem log_sum_max_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 4) :
   log 2 x + log 2 y ≤ 2 := 
sorry

end log_sum_max_value_l648_648722


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l648_648672

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l648_648672


namespace correct_calculation_l648_648914

theorem correct_calculation (x a : Real) :
  (3 * x^2 - x^2 ≠ 3) → 
  (-3 * a^2 - 2 * a^2 ≠ -a^2) →
  (x^3 / x ≠ 3) → 
  ((-x)^3 = -x^3) → 
  true :=
by
  intros _ _ _ _
  trivial

end correct_calculation_l648_648914


namespace total_cost_is_eight_times_short_cost_l648_648025

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l648_648025


namespace boys_girls_total_l648_648195

-- Definitions for conditions
def total_money : ℝ := 312
def money_per_boy : ℝ := 3.60
def money_per_girl : ℝ := 2.40
def approx_boys : ℕ := 60

-- Define the unknowns
variable (B G : ℕ)

-- Statement of the theorem
theorem boys_girls_total :
  (money_per_boy * B + money_per_girl * G = total_money) → B = approx_boys → B + G = 100 :=
begin
  intros h1 h2,
  have h3 : money_per_girl * G = total_money - money_per_boy * B,
  { rw h2 at h1, exact (sub_eq_of_eq_add' h1).symm, },
  have h4 : G = (total_money - money_per_boy * B) / money_per_girl,
  { symmetry, exact (div_eq_iff_mul_eq money_per_girl_ne_zero).2 h3, },
  have h5 : G = 40,
  { rw [h2, sub_mul, div_eq_of_eq_mul_right money_per_girl_ne_zero 96],
    exact (eq_div_iff money_per_girl_ne_zero).1 rfl, },
  rw [h2, h5],
end

-- Leaves the proof as sorry since we do not need to provide the actual proof steps

end boys_girls_total_l648_648195


namespace total_cost_is_eight_times_short_cost_l648_648026

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l648_648026


namespace arithmetic_sqrt_16_l648_648425

theorem arithmetic_sqrt_16 : real.sqrt 16 = 4 := by
  sorry

end arithmetic_sqrt_16_l648_648425


namespace suitable_lotus_plant_model_l648_648162

theorem suitable_lotus_plant_model :
  ∃ k a : ℝ, (k > 0) ∧ (a > 1) ∧
  (∀ (x : ℕ), (x = 2 → y = 24) ∧ (x = 3 → y = 36) → y = k * a^x) ∧
  (∀ (x : ℕ), (y = k * a^x) → (y x) > (10 * y 0) → x ≥ 6) := 
sorry

end suitable_lotus_plant_model_l648_648162


namespace shaded_area_eq_l648_648890

theorem shaded_area_eq : 
  let side := 8 
  let radius := 3 
  let square_area := side * side
  let sector_area := (1 / 4) * Real.pi * (radius * radius)
  let four_sectors_area := 4 * sector_area
  let triangle_area := (1 / 2) * radius * radius
  let four_triangles_area := 4 * triangle_area
  let shaded_area := square_area - four_sectors_area - four_triangles_area
  shaded_area = 64 - 9 * Real.pi - 18 :=
by
  sorry

end shaded_area_eq_l648_648890


namespace cos_angle_AND_of_regular_tetrahedron_l648_648572

-- Definitions based on the conditions:
structure Tetrahedron :=
(vertex : Fin 4 → Point)

def is_regular_tetrahedron (T : Tetrahedron) : Prop :=
  ∀ i j : Fin 4, i ≠ j → (distance (T.vertex i) (T.vertex j)) = (distance (T.vertex 0) (T.vertex 1))

def is_midpoint (P Q R : Point) (M : Point) : Prop :=
  distance P M = distance Q M ∧ M = midpoint P Q R

-- Statement of the proof problem:
theorem cos_angle_AND_of_regular_tetrahedron
  {A B C D N : Point}
  (T : Tetrahedron)
  (h_regular : is_regular_tetrahedron T)
  (h_midpoint: is_midpoint B C N)
  (h_tetrahedron_structure: T.vertex 0 = A ∧ T.vertex 1 = B ∧ T.vertex 2 = C ∧ T.vertex 3 = D) :
  cos (angle A N D) = 1 / 3 :=
sorry

end cos_angle_AND_of_regular_tetrahedron_l648_648572


namespace matchsticks_10th_stage_l648_648981

theorem matchsticks_10th_stage :
  let a_1 := 4
  ∧ ∀ n: ℕ, n > 0 →  ( (∀ k ∈ {2, 3, ..., n}, a_k = a_1 + ∑i in finset.range(k-1).image(λ x, x + 2), 3 * i) 
  → a_10 = 181 := sorry

end matchsticks_10th_stage_l648_648981


namespace exists_f_satisfies_conditions_l648_648412

open Set Ordinal

noncomputable def f (α β : Ordinal) : Ordinal :=
  if hα : α > 0 then sup {γ | γ < α ∧ ω_limit γ} else 0

theorem exists_f_satisfies_conditions :
  ∃ f : Set (Ordinal × Ordinal) → Ordinal,
    (∀ α β : Ordinal, α > 0 → β > 0 → f (α, β) < min α β) ∧
    (∀ (α_i : ℕ → Ordinal),
      strict_mono α_i →
      α_i 0 < ω₁ →
      sup (λ i, α_i i) = sup (λ ⟨i, j⟩, f (α_i i, α_i j))) :=
by
  let f (α β : Ordinal) := if hα : α > 0 then sup {γ | γ < α ∧ ω_limit γ} else 0
  use f
  sorry

end exists_f_satisfies_conditions_l648_648412


namespace annual_increase_Q_eq_15_paise_l648_648104

noncomputable def price_P_in_2001 : ℝ := 4.20
noncomputable def price_Q_in_2001 : ℝ := 6.30
noncomputable def annual_increase_P : ℝ := 0.40  -- 40 paise in Rs

def price_P (year : ℕ) : ℝ := price_P_in_2001 + annual_increase_P * (year - 2001)
def price_Q (year : ℕ) (annual_increase_Q : ℝ) : ℝ := price_Q_in_2001 + annual_increase_Q * (year - 2001)

theorem annual_increase_Q_eq_15_paise :
  ∃ x : ℝ, (price_P 2011 - price_Q 2011 x = 0.40) → (x = 0.15) :=
begin
  sorry
end

end annual_increase_Q_eq_15_paise_l648_648104


namespace smallest_four_digit_divisible_by_primes_l648_648655

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l648_648655


namespace solve_trigonometric_eq_l648_648073

theorem solve_trigonometric_eq (x : ℝ) (h1 : sqrt (1 + tan x) = sin x + cos x) (h2 : abs (2 * x - 5) < 2) : 
  x = 3 * π / 4 :=
sorry

end solve_trigonometric_eq_l648_648073


namespace total_cost_is_eight_times_l648_648016

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l648_648016


namespace largest_digit_A_l648_648354

theorem largest_digit_A (A : ℕ) (h1 : (31 + A) % 3 = 0) (h2 : 96 % 4 = 0) : 
  A ≤ 7 ∧ (∀ a, a > 7 → ¬((31 + a) % 3 = 0 ∧ 96 % 4 = 0)) :=
by
  sorry

end largest_digit_A_l648_648354


namespace angle_A_CB_is_90_l648_648465

-- Define the problem conditions
variables (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variables (triangle_ABC : Triangle)
variables (circle : Circle)
variables (equilateral_triangle_CFE : EquilateralTriangle)
variables (AB AC : Length)
variables (angle_DAC angle_EAC angle_ECF angle_FCE angle_ABC angle_AEB angle_ACB : Angle)

-- Assume the conditions
axiom AB_eq_3AC : AB = 3 * AC
axiom circle_through_A_C : circle.passes_through A C 
axiom center_circle_on_AB : circle.center ∈ AB
axiom circle_intersects_AB_at_D : circle.intersects AB D
axiom circle_intersects_BC_at_E : circle.intersects BC E
axiom intersection_AE_CD_at_F : LineThrough AE ∩ LineThrough CD = F
axiom equilateral_CFE_angles : angle_ECF = 60 ∧ angle_FCE = 60 ∧ angle_CFE = 60
axiom angle_DAC_right : angle_DAC = 90
axiom angle_EAC_right : angle_EAC = 90
axiom angle_sum_triangle_ABC : angle_ABC + angle_BCA + angle_ACB = 180

-- Prove the target angle
theorem angle_A_CB_is_90 : angle_ACB = 90 := by
  sorry

end angle_A_CB_is_90_l648_648465


namespace distance_between_blue_lights_l648_648121

theorem distance_between_blue_lights (n m : ℕ) (pattern_length blue_lights yellow_lights spacing_inches spacing_feet : ℕ)
  (h1 : pattern_length = blue_lights + yellow_lights)
  (h2 : blue_lights = 3) (h3 : yellow_lights = 4) (h4 : spacing_inches = 7) (h5 : spacing_feet = 12) :
  let position_n := ((n - 1) / blue_lights) * pattern_length + (n - 1) % blue_lights + 1,
      position_m := ((m - 1) / blue_lights) * pattern_length + (m - 1) % blue_lights + 1 in
  m > n →
  (position_m - position_n) * spacing_inches / spacing_feet = 28 :=
by
  intros position_n position_m hmn
  sorry

end distance_between_blue_lights_l648_648121


namespace gcd_subtraction_result_l648_648217

theorem gcd_subtraction_result : gcd 8100 270 - 8 = 262 := by
  sorry

end gcd_subtraction_result_l648_648217


namespace exists_happy_configuration_l648_648374

noncomputable def all_children_happy (n: ℕ) (a: Fin n → ℕ) (xA xB xC: ℕ) : Prop :=
  let total_raisins := (n * (n + 1)) / 2
  let child_raises_at (x: ℕ) : ℕ := 
    (Finset.range n).sum (λ k => if xA = k ∨ xB = k ∨ xC = k then a k else 0)
  let unhappy (x: ℕ) (r: ℕ) : Prop :=
    ∃ y ∈ Finset.range n, y ≠ x ∧ child_raises_at y > r
  ∀ x ∈ {xA, xB, xC}, ¬ unhappy x (child_raises_at x)

theorem exists_happy_configuration : ∀ n: ℕ, (2 ≤ n ∧ n ≤ 8) ↔ ∃ (a: Fin n → ℕ) (xA xB xC: ℕ), all_children_happy n a xA xB xC := 
  by
    sorry

end exists_happy_configuration_l648_648374


namespace number_of_students_l648_648785

-- Definitions based on problem conditions
def age_condition (a n : ℕ) : Prop :=
  7 * (a - 1) + 2 * (a + 2) + (n - 9) * a = 330

-- Main theorem to prove the correct number of students
theorem number_of_students (a n : ℕ) (h : age_condition a n) : n = 37 :=
  sorry

end number_of_students_l648_648785


namespace new_length_maintains_area_l648_648597

noncomputable def new_length_for_doubled_width (A W : ℝ) : ℝ := A / (2 * W)

theorem new_length_maintains_area (A W : ℝ) (hA : A = 35.7) (hW : W = 3.8) :
  new_length_for_doubled_width A W = 4.69736842 :=
by
  rw [new_length_for_doubled_width, hA, hW]
  norm_num
  sorry

end new_length_maintains_area_l648_648597


namespace sum_of_primes_1_to_20_l648_648535

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648535


namespace sum_of_primes_1_to_20_l648_648558

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648558


namespace gcd_polynomials_l648_648285

noncomputable def b : ℤ := sorry -- since b is given as an odd multiple of 997

theorem gcd_polynomials (h : ∃ k : ℤ, b = 997 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 41 * b + 101) (b + 17) = 1 :=
sorry

end gcd_polynomials_l648_648285


namespace work_completion_days_l648_648761

theorem work_completion_days (A B C : ℝ) (h1 : A + B + C = 1/4) (h2 : B = 1/18) (h3 : C = 1/6) : A = 1/36 :=
by
  sorry

end work_completion_days_l648_648761


namespace people_left_at_table_l648_648959

theorem people_left_at_table {initial_people left_people remaining_people : ℕ} 
  (h_initial : initial_people = 11) 
  (h_left : left_people = 6) 
  (h_calculation : remaining_people = initial_people - left_people) : 
  remaining_people = 5 := 
by
  rw [h_initial, h_left] at h_calculation
  rw [h_calculation]
  sorry

end people_left_at_table_l648_648959


namespace total_time_taken_l648_648407

theorem total_time_taken 
  (R : ℝ) -- Rickey's speed
  (T_R : ℝ := 40) -- Rickey's time
  (T_P : ℝ := (40 * (4 / 3))) -- Prejean's time derived from given conditions
  (P : ℝ := (3 / 4) * R) -- Prejean's speed
  (k : ℝ := 40 * R) -- constant k for distance
 
  (h1 : T_R = 40)
  (h2 : T_P = 40 * (4 / 3))
  -- Main goal: Prove total time taken equals 93.33 minutes
  : (T_R + T_P) = 93.33 := 
  sorry

end total_time_taken_l648_648407


namespace like_terms_monomials_l648_648770

variable (a b : ℕ)
variable (m n : ℕ)

theorem like_terms_monomials : 
  (n + 1 = 3) → 
  (m = 1) → 
  (m - n = -1) :=
by
  sorry

end like_terms_monomials_l648_648770


namespace percentage_permanent_technicians_l648_648793

theorem percentage_permanent_technicians 
  (W : ℝ)  -- total number of workers
  (H1 : 0.90 * W = T)  -- number of technicians
  (H2 : 0.10 * W = N)  -- number of non-technicians
  (H3 : 0.10 * N = Pn)  -- number of permanent non-technicians
  (H4 : 0.82 * W = TA)  -- number of temporary workers
  : (P_t : ℝ)
  (H5 : P_t * T + Pn = 0.18 * W)  -- total number of permanent workers
  : P_t = 0.17 / 0.90 :=
begin
  sorry
end

end percentage_permanent_technicians_l648_648793


namespace area_of_triangle_COD_l648_648831

theorem area_of_triangle_COD (x p : ℕ) (hx : 0 < x) (hx' : x < 12) (hp : 0 < p) :
  (∃ A : ℚ, A = (x * p : ℚ) / 2) :=
sorry

end area_of_triangle_COD_l648_648831


namespace total_cost_is_eight_times_l648_648012

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l648_648012


namespace probability_all_same_house_l648_648576

noncomputable def apply_probability : ℚ :=
let p_house : ℚ := 1 / 3 in
let p_all_choose_same_house := p_house * p_house * p_house in
(p_all_choose_same_house) * 3

theorem probability_all_same_house : apply_probability = 1 / 9 := by
  sorry

end probability_all_same_house_l648_648576


namespace sobhas_parents_age_difference_l648_648859

def difference_in_ages (F M : ℕ) : ℕ := F - M

theorem sobhas_parents_age_difference
  (S F M : ℕ)
  (h1 : F = S + 38)
  (h2 : M = S + 32) :
  difference_in_ages F M = 6 := by
  sorry

end sobhas_parents_age_difference_l648_648859


namespace solution_is_constant_l648_648070

-- Definitions
variable (M N : ℝ → ℝ → ℝ)
variable (ψ : ℝ → ℝ → ℝ)

-- Conditions
def is_exact : Prop :=
  ∀ x y, (∂/∂x ψ(x,y) = M x y) ∧ (∂/∂y ψ(x,y) = N x y)

def is_homogeneous (n : ℝ) : Prop :=
  ∀ t x y, M (t * x) (t * y) = t^n * (M x y) ∧ N (t * x) (t * y) = t^n * (N x y)

-- Main Theorem
theorem solution_is_constant (n : ℝ)
  (exact : is_exact M N)
  (homogeneous : is_homogeneous M N n) :
  ∀ x y, x * (M x y) + y * (N x y) = const := 
sorry

end solution_is_constant_l648_648070


namespace last_two_digits_base7_l648_648908

theorem last_two_digits_base7 (n : ℕ) (k : ℕ) :
  n = 2018 → k = 2018 → (n ^ k % 49) = 32 ∧ 
  let digits_in_base_7 := nat.digits 7 32 in 
  (digits_in_base_7.reverse.take 2 = [4, 4]) :=
by
  intros
  have h1 : n ^ k % 49 = 32 := sorry
  have h2 : nat.digits 7 32 = [4, 4] := sorry
  split
  exact h1
  exact h2

end last_two_digits_base7_l648_648908


namespace sum_of_primes_between_1_and_20_l648_648537

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648537


namespace find_mn_l648_648837

open_locale classical

noncomputable def vector_a (m n : ℝ) : ℝ × ℝ × ℝ := (2, 2 * m - 3, n + 2)
noncomputable def vector_b (m n : ℝ) : ℝ × ℝ × ℝ := (4, 2 * m + 1, 3 * n - 2)

def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2 ∧ a.3 = k * b.3

theorem find_mn (m n : ℝ) : 
  parallel (vector_a m n) (vector_b m n) → m * n = 21 :=
by
  sorry

end find_mn_l648_648837


namespace total_cost_is_eight_times_l648_648015

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l648_648015


namespace parliament_bipartition_l648_648794

-- Definitions and conditions
variable (Members : Type)
variable (enemies : Members → Members → Prop)

-- Conditions
variable [finite Members]
variable h_enemies_symmetric : ∀ m₁ m₂ : Members, enemies m₁ m₂ → enemies m₂ m₁
variable h_max_three_enemies : ∀ m : Members, (finset.univ.filter (enemies m)).card ≤ 3

-- Theorem statement to be proved
theorem parliament_bipartition : 
  ∃ (partition : Members → bool), 
  ∀ m : Members, (finset.univ.filter (λ x, partition x = partition m ∧ enemies x m)).card ≤ 1 :=
by
  sorry

end parliament_bipartition_l648_648794


namespace quadratic_real_roots_condition_l648_648775

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m-1) * x₁^2 - 4 * x₁ + 1 = 0 ∧ (m-1) * x₂^2 - 4 * x₂ + 1 = 0) ↔ (m < 5 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_condition_l648_648775


namespace minimize_segment_length_l648_648090

-- Definition of line as a set of points in a 3D space
structure Line (ℝ : Type) [Real ℝ] :=
  (points : set (ℝ × ℝ × ℝ))
  (is_line : ∃ a b c : ℝ, ∀ (x y z : ℝ), (x, y, z) ∈ points → a * x + b * y + c * z = 0)

-- Definition of a segment with endpoints lying on given lines
structure Segment (ℝ : Type) [Real ℝ] :=
  (A B : ℝ × ℝ × ℝ)
  (A_on_a : ∀ (a : Line ℝ), A ∈ a.points)
  (B_on_b : ∀ (b : Line ℝ), B ∈ b.points)

-- Theorem to prove that the segment length AB is minimized when it is perpendicular to both lines a and b.
theorem minimize_segment_length (a b : Line ℝ) (seg : Segment ℝ) :
  (seg.A ≠ seg.B) →
  (min ∃ k l : ℝ, (∀ (A B : ℝ × ℝ × ℝ), A ∈ a.points → B ∈ b.points → seg.A ≠ seg.B → 
    (seg.A.1 - seg.B.1)^2 + (seg.A.2 - seg.B.2)^2 + (seg.A.3 - seg.B.3)^2 = k ∧ 
    k is minimized when the segment AB is perpendicular to both lines a and b)) :=
by
  sorry

end minimize_segment_length_l648_648090


namespace sum_denominators_lower_bound_l648_648409

theorem sum_denominators_lower_bound (n : ℕ) (f : Fin n → ℚ) 
    (h_n : n ≥ 2) 
    (h_f : ∀ (i j : Fin n), i ≠ j → f i ≠ f j ∧ 0 < f i ∧ f i < 1) : 
    (∑ i in Finset.univ, (f i).den) ≥ (1 / 3) * n ^ (3 / 2) := 
by
  sorry

end sum_denominators_lower_bound_l648_648409


namespace invariant_configuration_count_l648_648954

-- Definitions of the problem conditions
def cube_size := (3, 3, 3)
def red_cubes := 14
def yellow_cubes := 13
def total_cubes := 27

-- The question reformulated as a theorem
theorem invariant_configuration_count :
  (number_of_ways_to_construct_cube_invariant_under_180_rotation 
    cube_size red_cubes yellow_cubes = 16) :=
sorry

end invariant_configuration_count_l648_648954


namespace enrollment_difference_l648_648128

theorem enrollment_difference 
  (Varsity_enrollment : ℕ)
  (Northwest_enrollment : ℕ)
  (Central_enrollment : ℕ)
  (Greenbriar_enrollment : ℕ) 
  (h1 : Varsity_enrollment = 1300) 
  (h2 : Northwest_enrollment = 1500)
  (h3 : Central_enrollment = 1800)
  (h4 : Greenbriar_enrollment = 1600) : 
  Varsity_enrollment < Northwest_enrollment ∧ 
  Northwest_enrollment < Greenbriar_enrollment ∧ 
  Greenbriar_enrollment < Central_enrollment → 
    (Greenbriar_enrollment - Varsity_enrollment = 300) :=
by
  sorry

end enrollment_difference_l648_648128


namespace cost_of_purchase_l648_648032

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l648_648032


namespace probability_of_different_colors_l648_648892

noncomputable def total_chips := 6 + 5 + 4

noncomputable def prob_diff_color : ℚ :=
  let pr_blue := 6 / total_chips
  let pr_red := 5 / total_chips
  let pr_yellow := 4 / total_chips

  let pr_not_blue := (5 + 4) / total_chips
  let pr_not_red := (6 + 4) / total_chips
  let pr_not_yellow := (6 + 5) / total_chips

  pr_blue * pr_not_blue + pr_red * pr_not_red + pr_yellow * pr_not_yellow

theorem probability_of_different_colors :
  prob_diff_color = 148 / 225 :=
sorry

end probability_of_different_colors_l648_648892


namespace smallest_divisible_four_digit_number_l648_648667

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l648_648667


namespace total_cost_is_eight_times_l648_648047

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l648_648047


namespace regular_tetrahedron_properties_l648_648621

/-- A regular tetrahedron is a polyhedron with four equilateral triangle faces. -/
structure RegularTetrahedron :=
  (edges_equal_length : ∀ e1 e2, length e1 = length e2)
  (faces_congruent : ∀ f1 f2, congruent_equilateral_triangle f1 f2)
  (angle_between_edges_at_vertex_equal : ∀ v e1 e2 e3 e4, angle v e1 e2 = angle v e3 e4)
  (dihedral_angle_between_faces_equal : ∀ f1 f2 f3 f4, dihedral_angle f1 f2 = dihedral_angle f3 f4)

/-- The properties of a regular tetrahedron inferred from the properties of an equilateral triangle in the plane. -/
theorem regular_tetrahedron_properties (T : RegularTetrahedron) :
  T.edges_equal_length ∧ T.faces_congruent ∧ 
  T.angle_between_edges_at_vertex_equal ∧ T.dihedral_angle_between_faces_equal :=
by sorry

end regular_tetrahedron_properties_l648_648621


namespace cartesian_eqn_C1_cartesian_eqn_C2_intersection_MA_MB_l648_648346

-- Definitions based on conditions
def parametric_eqn_C1 (t : ℝ) : ℝ × ℝ :=
  (2 - real.sqrt 2 * t, -1 + real.sqrt 2 * t)

def polar_eqn_C2 (θ : ℝ) : ℝ :=
  2 / real.sqrt (1 + 3 * real.sin θ ^ 2)

def point_M : ℝ × ℝ := (2, -1)

-- Theorem statements based on generated proof problems
theorem cartesian_eqn_C1 (x y t : ℝ) (h_param : (x, y) = parametric_eqn_C1 t) :
  y = 1 - x := sorry

theorem cartesian_eqn_C2 (x y ρ θ : ℝ) (h_polar : ρ = polar_eqn_C2 θ)
  (h_x : x = ρ * real.cos θ) (h_y : y = ρ * real.sin θ) : 
  x^2 + 4 * y^2 = 4 := sorry

theorem intersection_MA_MB (A B : ℝ × ℝ) (h_inter_A : ∃ t1, A = parametric_eqn_C1 t1 ∧ 
  x^2 + 4 * (1 - x)^2 = 4 \ t = t1) 
  (h_inter_B : ∃ t2, B = parametric_eqn_C1 t2 ∧ 
  (2 - real.sqrt 2 * t, -1 + real.sqrt 2 * t) t = t2) :
  ∥ point_M - A ∥ * ∥ point_M - B ∥ = 8 / 5 := sorry

end cartesian_eqn_C1_cartesian_eqn_C2_intersection_MA_MB_l648_648346


namespace induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l648_648408

open Nat

theorem induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25 :
  ∀ n : ℕ, n > 0 → 25 ∣ (2^(n+2) * 3^n + 5*n - 4) :=
by
  intro n hn
  sorry

end induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l648_648408


namespace sum_of_interior_angles_increases_l648_648912

theorem sum_of_interior_angles_increases (n : ℕ) (h : n ≥ 3) : (n-2) * 180 > (n-3) * 180 :=
by
  sorry

end sum_of_interior_angles_increases_l648_648912


namespace juan_marbles_l648_648624

-- Conditions
def connie_marbles : ℕ := 39
def extra_marbles_juan : ℕ := 25

-- Theorem statement: Total marbles Juan has
theorem juan_marbles : connie_marbles + extra_marbles_juan = 64 :=
by
  sorry

end juan_marbles_l648_648624


namespace max_revenue_at_300_l648_648589

/-- Define the number of rooms in the hotel -/
def num_rooms : ℕ := 100

/-- Define the initial room rate in yuan per day -/
def initial_rate : ℕ := 400

/-- Define the initial occupancy rate as a fraction -/
def initial_occupancy_rate : ℚ := 0.50

/-- Define the revenue function as a function of room rate -/
def revenue (rate : ℕ) : ℕ :=
  let occupied_rooms := 50 + (initial_rate - rate) / 20 * 5 in
  rate * occupied_rooms

/-- The statement to be proven: The room rate which maximizes hotel's room revenue is 300 yuan/day. -/
theorem max_revenue_at_300 : revenue 300 = max (revenue <$> [380, 360, 340, 320, 300, 280, 260]) := 
  sorry

end max_revenue_at_300_l648_648589


namespace range_of_2a_plus_3b_inequality_between_expressions_l648_648695

-- First proof problem
theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) (h3 : -1 ≤ a - b) (h4 : a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
sorry

-- Second proof problem
theorem inequality_between_expressions (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (1 / (a^2 + 1) + 1 / (b^2 + 2)) > (1 / 2 - 1 / (c^2 + 3)) :=
sorry

end range_of_2a_plus_3b_inequality_between_expressions_l648_648695


namespace wood_length_equation_l648_648933

theorem wood_length_equation (x : ℝ) : 
  (∃ r : ℝ, r - x = 4.5 ∧ r/2 + 1 = x) → 1/2 * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l648_648933


namespace sum_of_primes_1_to_20_l648_648528

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648528


namespace age_difference_is_18_l648_648838

def sum_of_digits (n : ℕ) : ℕ :=
  Nat.digits 10 n |>.sum

def interesting_year (birth_year current_year : ℕ) : Prop :=
  current_year - birth_year = sum_of_digits current_year

theorem age_difference_is_18 (x y z t : ℕ) (hI : 0 ≤ x ∧ x ≤ 9) (hI' : 0 ≤ y ∧ y ≤ 9)
  (hV : 0 ≤ z ∧ z ≤ 9) (hV' : 0 ≤ t ∧ t ≤ 9)
  (hIvan : ∃ (current_year : ℕ), interesting_year (1900 + 10 * x + y) current_year)
  (hVovochka : ∃ (current_year : ℕ), interesting_year (2000 + 10 * z + t) current_year) :
  abs ((2000 + 10 * z + t) - (1900 + 10 * x + y)) = 18 :=
sorry

end age_difference_is_18_l648_648838


namespace matrixN_correct_l648_648631

noncomputable def N : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![-1, 3, 6],
  ![4, -2, 0],
  ![9, 5, -3]
]

def i : Fin 3 → ℝ := ![1, 0, 0]
def j : Fin 3 → ℝ := ![0, 1, 0]
def k : Fin 3 → ℝ := ![0, 0, 1]

def v1 : Fin 3 → ℝ := ![-1, 4, 9]
def v2 : Fin 3 → ℝ := ![3, -2, 5]
def v3 : Fin 3 → ℝ := ![6, 0, -3]
def v4 : Fin 3 → ℝ := ![8, 2, 11]

theorem matrixN_correct :
  (N ⬝ i = v1) ∧
  (N ⬝ j = v2) ∧
  (N ⬝ k = v3) ∧
  (N ⬝ (i + j + k) = v4) :=
by
  sorry

end matrixN_correct_l648_648631


namespace proof_m_n_sum_l648_648126

noncomputable def m : ℕ := 2
noncomputable def n : ℕ := 125

theorem proof_m_n_sum : 
  (m = 2 ∧ (∀ k : ℕ, k < 2 → (k = 1 ∨ ∃ d : ℕ, d ∣ k) → false)) → 
  (n < 200 ∧ ∀ k : ℕ, k < 200 → (∀ d : ℕ, d ∣ k → d = 1 ∨ d = k ∨ d = p ∨ d = q) → n ≥ k) → 
  m + n = 127 :=
by
  intro h1 h2
  rw [h1.1, h2.1]
  exact rfl

end proof_m_n_sum_l648_648126


namespace determine_rhombus_l648_648995

def QuadrilateralWithFourEqualSides (Q : Type) [Quadrilateral Q] : Prop :=
  ∀ (a b c d : Q), a = b ∧ b = c ∧ c = d

def QuadrilateralWithPerpendicularBisectingDiagonals (Q : Type) [Quadrilateral Q] : Prop :=
  ∀ (p1 p2 : Q), (p1.perpendicular p2) ∧ (p1.bisects p2)

def QuadrilateralWithPairOfAdjacentEqualSides (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (a b : Q), a.adjacent_to b ∧ a = b 

def ParallelogramWithDiagonalBisectsOppositeAngles (Q : Type) [Quadrilateral Q] [Parallelogram Q] : Prop :=
  ∃ (d : Q), d.diagonal_bisects_opposite_angles

theorem determine_rhombus :
  (QuadrilateralWithFourEqualSides Q) ∨ (QuadrilateralWithPerpendicularBisectingDiagonals Q)
  ∨ (ParallelogramWithDiagonalBisectsOppositeAngles Q) ↔ Rhombus Q :=
sorry

end determine_rhombus_l648_648995


namespace sample_size_student_congress_l648_648583

-- Definitions based on the conditions provided in the problem
def num_classes := 40
def students_per_class := 3

-- Theorem statement for the mathematically equivalent proof problem
theorem sample_size_student_congress : 
  (num_classes * students_per_class) = 120 := 
by 
  sorry

end sample_size_student_congress_l648_648583


namespace max_clocks_is_11_l648_648788

-- Define the problem context
noncomputable def max_num_clocks (hours minutes : List ℕ) : ℕ :=
  if hours.length = minutes.length 
    ∧ hours.length ≥ 2 
    ∧ hours.length ≤ 12 
    ∧ hours.nodup 
    ∧ minutes.nodup 
    ∧ (all h in hours, 1 ≤ h ∧ h ≤ 12) 
    ∧ (all m in minutes, 0 ≤ m ∧ m < 60) 
    ∧ (sum hours + sum minutes).pred = sum hours.pred + sum minutes.pred
    then hours.length
    else 0

/-- Given the conditions on the clocks in the workshop, prove the maximum number of clocks is 11 --/
theorem max_clocks_is_11 : ∀ (hours minutes : List ℕ), 
  (hours.length = minutes.length 
  ∧ hours.length ≥ 2 
  ∧ hours.length ≤ 12 
  ∧ hours.nodup 
  ∧ minutes.nodup 
  ∧ (all h in hours, 1 ≤ h ∧ h ≤ 12) 
  ∧ (all m in minutes, 0 ≤ m ∧ m < 60) 
  ∧ (sum hours + sum minutes).pred = sum hours.pred + sum minutes.pred) → 
  hours.length ≤ 11 := 
by 
  sorry

end max_clocks_is_11_l648_648788


namespace sunzi_wood_problem_l648_648936

theorem sunzi_wood_problem (x : ℝ) :
  (∃ (length_of_rope : ℝ), length_of_rope = x + 4.5 ∧
    ∃ (half_length_of_rope : ℝ), half_length_of_rope = length_of_rope / 2 ∧ 
      (half_length_of_rope + 1 = x)) ↔ 
  (1 / 2 * (x + 4.5) = x - 1) :=
by
  sorry

end sunzi_wood_problem_l648_648936


namespace aligned_point_partition_l648_648817

def is_aligned (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∨ p1.2 = p2.2

def are_collinear (ps : set (ℝ × ℝ)) : Prop :=
  ∃ x, ∀ p ∈ ps, p.1 = x ∨ ∃ y, ∀ p ∈ ps, p.2 = y

theorem aligned_point_partition (S : set (ℝ × ℝ)) (n : ℕ)
  (hS : fintype.card S = n) :
  ∃ (partition : finset (set (ℝ × ℝ))),
    (∀ subset ∈ partition, are_collinear subset) ∧
    (∀ pair : (ℝ × ℝ) × (ℝ × ℝ), pair.fst ≠ pair.snd → pair.fst ∈ S → pair.snd ∈ S → 
      is_aligned pair.fst pair.snd → ¬∃ subset ∈ partition, pair.fst ∈ subset ∧ pair.snd ∈ subset) ∧
    finset.card ((partition.bind (λ subset, (@finset.pairwise_disjoint (ℝ × ℝ) _ _ _ 
      (hefin subset.to_finset)))).to_finset) ≤ n ^ (3/2) :=
sorry

end aligned_point_partition_l648_648817


namespace sum_primes_upto_20_l648_648515

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648515


namespace speed_of_current_l648_648592

-- Given conditions
def speed_in_still_water : ℝ := 15 -- kmph
def distance_covered : ℝ := 0.06 -- kilometers
def time_taken : ℝ := 11.999040076793857 / 3600 -- hours

-- Definition for the downstream speed
def downstream_speed : ℝ := distance_covered / time_taken

-- Proof statement
theorem speed_of_current : downstream_speed - speed_in_still_water = 3 :=
by 
    have h1 : downstream_speed = 0.06 / (11.999040076793857 / 3600)
    have h2 : downstream_speed = 18
    have h3 : speed_of_current = 18 - 15 -- Vd - Vm = Vc
    sorry

end speed_of_current_l648_648592


namespace total_cost_is_eight_times_l648_648017

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l648_648017


namespace rationalization_correctness_l648_648064

-- Assuming necessary definitions and imports
noncomputable def rationalizeDenom : ℚ :=
  (3 * Real.sqrt 3 + 9 * Real.sqrt 5 + 3 * Real.sqrt 11 - 2 * Real.sqrt 33) / 51

theorem rationalization_correctness :
  let x := 1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11)
  x = rationalizeDenom ∧ (3 + 9 + 3 - 2 + 33 + 51 = 97) :=
by
  sorry

end rationalization_correctness_l648_648064


namespace find_f2_l648_648754

theorem find_f2 (a b c : ℝ) (h₀ : f(1) = 7) (h₁ : f(3) = 19) (h₂ : f(x) = ax^2 + bx + c) : f(2) = 12 :=
by
  -- Placeholders for working through the provided conditions
  have eq1 : a + b + c = 7 := sorry
  have eq2 : 9 * a + 3 * b + c = 19 := sorry
  have eq3 : 8 * a + 2 * b = 12 := sorry
  have eq4 : 4 * a + b = 6 := sorry
  have eq5 : b = 6 - 4 * a := sorry
  have eq6 : c = 1 + 3 * a := sorry
  -- Substitute back into the polynomial and find f(2)
  calc f(2) = a * 2 ^ 2 + b * 2 + c : sorry
          ... = 12 : sorry

end find_f2_l648_648754


namespace distance_from_starting_point_l648_648595

-- Define the side length of the hexagon
def side_length : ℕ := 4

-- Define the total distance traveled by Bob
def distance_traveled : ℕ := 10

-- Prove the final distance from the starting point
theorem distance_from_starting_point
  (s : ℕ := side_length) (d : ℕ := distance_traveled) :
  sqrt (s^2 * 6 + 3) = sqrt 103 :=
by
  sorry

end distance_from_starting_point_l648_648595


namespace polynomial_roots_p_lt_zero_l648_648881

theorem polynomial_roots_p_lt_zero
  (a b c p q : ℝ) 
  (h_roots : (∃ (f : ℝ → ℝ), ∀ x, f x = x^3 + p * x + q ∧ ∀ r, f r = 0 → r = a ∨ r = b ∨ r = c)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a))
  (h_vieta1 : a + b + c = 0)
  (h_vieta2 : ab + bc + ca = p) :
  p < 0 := 
sorry

end polynomial_roots_p_lt_zero_l648_648881


namespace sum_primes_upto_20_l648_648510

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648510


namespace MI_through_midpoint_arc_l648_648999

-- Assume the necessary geometrical entities and their properties
variables (ω : circle) (A B C D I Q P M : Point)

-- Conditions
variables (hABC : IsoscelesTriangle A B C)
variables (hInscribed : InscribedInCircle A B C ω)
variables (hTangent : TangentAt ω C D)
variables (hIncenter : Incenter I A B C)
variables (hIntersect1 : IntersectAIQ AI Q angleCDB)
variables (hIntersect2 : IntersectBIP BI P angleCDB)
variables (hMidpoint : Midpoint M P Q)
variables (L : Point)
variables (hArcMidpoint : MidpointArc L ω A C B)

-- The theorem to prove
theorem MI_through_midpoint_arc : LineThrough M I (ArcMidpoint L ω A C B) := sorry

end MI_through_midpoint_arc_l648_648999


namespace upper_limit_for_n_l648_648764

theorem upper_limit_for_n : 
  ∀ n : ℕ, (0 < n) ∧ ((14 * n) % 60 = 0) ∧ (number_of_prime_factors n = 3) → n ≤ 210 :=
sorry

end upper_limit_for_n_l648_648764


namespace boat_downstream_travel_time_l648_648967

theorem boat_downstream_travel_time (D : ℝ) (V_b : ℝ) (T_u : ℝ) (V_c : ℝ) (T_d : ℝ) : 
  D = 300 ∧ V_b = 105 ∧ T_u = 5 ∧ (300 = (105 - V_c) * 5) ∧ (300 = (105 + V_c) * T_d) → T_d = 2 :=
by
  sorry

end boat_downstream_travel_time_l648_648967


namespace true_propositions_count_l648_648094

variables {V : Type*} [inner_product_space ℝ V]

def is_parallelogram (A B C D : V) : Prop :=
  (A - B) = (D - C)

theorem true_propositions_count
    (prop1 : ∀ (a b : V), ∥a∥ = ∥b∥ → a = b)
    (prop2 : ∀ (a b c : V), a = b ∧ b = c → a = c)
    (prop3 : ∀ (a b : V), a = b ↔ (∥a∥ = ∥b∥ ∧ (a = b ∨ a = -b)))
    (prop4 : ∀ (a b c : V), (a = b ∨ a = -b) ∧ (b = c ∨ b = -c) → (a = c ∨ a = -c))
    (prop5 : ∀ (A B C D : V), ¬ collinear ({A, B, C, D} : set V) →
      ((A - B) = (D - C) ↔ is_parallelogram A B C D)) :
    (2 = (if (∃ x : V × V, ∥x.1∥ = ∥x.2∥ ∧ x.1 ≠ x.2) then 0 else 1) +
      (if (∀ (a b c : V), a = b ∧ b = c → a = c) then 1 else 0) +
      (if (∃ x : V × V, ∥x.1∥ = ∥x.2∥ ∧ x.1 ≠ x.2 ∧ (x.1 = -x.2)) then 0 else 1) +
      (if (∃ x : V × V × V, ((x.1 = x.2 ∨ x.1 = -x.2) ∧ (x.2 = x.3 ∨ x.2 = -x.3)) ∧ (x.1 ≠ x.3 ∧ x.1 ≠ -x.3)) then 0 else 1) +
      (if (∀ (A B C D : V), ¬ collinear ({A, B, C, D}) → ((A - B) = (D - C) ↔ is_parallelogram A B C D)) then 1 else 0)) :=
by
  -- The proof body is omitted
  sorry

end true_propositions_count_l648_648094


namespace paint_cans_for_25_rooms_l648_648403

theorem paint_cans_for_25_rooms (cans rooms : ℕ) (H1 : cans * 30 = rooms) (H2 : cans * 25 = rooms - 5 * cans) :
  cans = 15 :=
by
  sorry

end paint_cans_for_25_rooms_l648_648403


namespace selected_number_in_14th_group_is_272_l648_648127

-- Definitions based on conditions
def total_students : ℕ := 400
def sample_size : ℕ := 20
def first_selected_number : ℕ := 12
def sampling_interval : ℕ := total_students / sample_size
def target_group : ℕ := 14

-- Correct answer definition
def selected_number_in_14th_group : ℕ := first_selected_number + (target_group - 1) * sampling_interval

-- Theorem stating the correct answer is 272
theorem selected_number_in_14th_group_is_272 :
  selected_number_in_14th_group = 272 :=
sorry

end selected_number_in_14th_group_is_272_l648_648127


namespace prove_S7_l648_648889

variable {a_n : ℕ → ℝ} -- Define the arithmetic sequence
variable {S_n : ℕ → ℝ} -- Define the sum of the first n terms of the sequence
variable (d : ℝ)       -- Define the common difference of the arithmetic sequence
variables (a1 : ℝ)

-- Define the conditions as Lean 4 definitions
def condition1 : Prop := S_n 7 - S_n 5 = 24
def condition2 : Prop := a_n 3 = 5
def condition3 : Prop := forall (n : ℕ), S_n n = n * (a1 + (n - 1) * d / 2)

-- Lean 4 statement to prove S_7 = 49 given the conditions
theorem prove_S7 (h1 : condition1) (h2 : condition2) (h3 : condition3) : S_n 7 = 49 :=
  sorry

end prove_S7_l648_648889


namespace sum_largest_second_smallest_l648_648116

theorem sum_largest_second_smallest :
  let numbers := {75, 91, 83, 72}
  let largest := 91
  let second_smallest := 75
  largest + second_smallest = 166 := 
by
  let numbers := {75, 91, 83, 72}
  let largest := 91
  let second_smallest := 75
  show largest + second_smallest = 166
  sorry

end sum_largest_second_smallest_l648_648116


namespace propositions_truth_l648_648119

theorem propositions_truth :
  (∀ (x y : ℝ), (x + y = 0 → x = -y) → (x = -y → x + y = 0)) ∧
  (¬ (∀ (Δ₁ Δ₂ : Triangle), congruent Δ₁ Δ₂ → area Δ₁ = area Δ₂) → 
   ¬ (∀ (Δ₁ Δ₂ : Triangle), ¬ congruent Δ₁ Δ₂ → ¬ area Δ₁ = area Δ₂)) ∧
  (∀ (a : ℝ), (¬ is_rational (a + 5) → ¬ is_rational a) → 
   (is_rational a → is_rational (a + 5))) ∧
  (∀ (a b : ℝ), (even a ∧ even b → even (a + b)) → 
   (even (a + b) → even a ∧ even b)) :=
begin
  -- Provide proofs for each of the propositions
  -- ① The converse of "If x + y = 0, then x and y are opposite numbers" is true.
  split,
  {
    intros x y hxy hcon.
    exact hxy hcon,
  },

  -- ② The negation of "The areas of congruent triangles are equal" is false.
  split,
  {
    intros hneg hcon.
    sorry,  -- this requires geometric proof which is not the focus here
  },
  
  -- ③ The contrapositive of "If a + 5 is an irrational number, then a is an irrational number" is true.
  split,
  {
    intros a hcon.
    exact hcon,
  },
  
  -- ④ The converse of "If a and b are both even numbers, then a + b is an even number" is false.
  {
    intros a b hcon.
    intro heven.
    sorry,  -- this also requires more detailed proof
  },
end

end propositions_truth_l648_648119


namespace common_ratio_q_l648_648726

variable {a₁ q : ℝ}

def S (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem common_ratio_q (h : S 6 / S 3 = 9) : q = 2 :=
by
  sorry

end common_ratio_q_l648_648726


namespace mod_inverse_of_40_l648_648711

/- Define the problem conditions and the proof goal -/
theorem mod_inverse_of_40 (h : 21⁻¹ ≡ 15 [MOD 61]) : 40⁻¹ ≡ 46 [MOD 61] := 
sorry

end mod_inverse_of_40_l648_648711


namespace find_n_l648_648646

theorem find_n :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4573 [ZMOD 8] ∧ n = 3 :=
begin
  use 3,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { refl },
end

end find_n_l648_648646


namespace general_term_formula_sum_b_terms_l648_648380

variable {a : ℕ → ℤ} -- Defines the sequence {a_n}
variable {S : ℕ → ℤ} -- Defines the sum S_n of the first n terms of the sequence {a_n}
variable {a2 a6 : ℤ} -- Variables for a_2 and a_6
variable {n : ℕ} -- General term index n

-- Conditions on the problem
def arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  ∀ n, a (n + 1) - a n = d

def given_conditions : Prop := 
  a 2 + a 6 = 2 ∧ S 15 = 75

-- Define the sequence S_n as sum of the first n terms
def sum_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := 
  ∀ n, S n = ∑ i in finset.range n, a i

-- Prove the general term formula
theorem general_term_formula (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : given_conditions) 
  (h3 : sum_sequence S a) : 
  ∀ n, a n = n - 3 := 
by 
  sorry

-- Define new sequence b_n
def b (a : ℕ → ℤ) : ℕ → ℤ := 
  λ n, 2 * a n - (2 * n - 1)

-- Define T_n as sum of the first n terms of sequence {b_n}
def sum_b_sequence (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop := 
  ∀ n, T n = ∑ i in finset.range n, b i

-- Prove the sum of the first n terms of sequence {b_n}
theorem sum_b_terms (a : ℕ → ℤ) (b : ℕ → ℤ) (T : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : given_conditions) 
  (h3 : sum_sequence S a) 
  (h4 : ∀ n, a n = n - 3) 
  (h5 : sum_b_sequence T (b a)) : 
  ∀ n, T n = (1 / 4) * (2 ^ n - 1) - n ^ 2 := 
by 
  sorry

end general_term_formula_sum_b_terms_l648_648380


namespace confidence_k_squared_l648_648336

-- Define the condition for 95% confidence relation between events A and B
def confidence_95 (A B : Prop) : Prop := 
  -- Placeholder for the actual definition, assume 95% confidence implies a specific condition
  True

-- Define the data value and critical value condition
def K_squared : ℝ := sorry  -- Placeholder for the actual K² value

theorem confidence_k_squared (A B : Prop) (h : confidence_95 A B) : K_squared > 3.841 := 
by
  sorry  -- Proof is not required, only the statement

end confidence_k_squared_l648_648336


namespace choose_one_basketball_and_one_volleyball_l648_648164

theorem choose_one_basketball_and_one_volleyball : 
  (number_of_basketballs number_of_volleyballs: ℕ) (h1 : number_of_basketballs = 5) (h2 : number_of_volleyballs = 4) :
  (number_of_basketballs * number_of_volleyballs = 20) :=
by
  simp [h1, h2]
  sorry

end choose_one_basketball_and_one_volleyball_l648_648164


namespace max_volume_prism_l648_648795

noncomputable def isosceles_triangle_area (a : ℝ) : ℝ :=
  (sqrt 3 / 4) * a^2

noncomputable def prism_volume (a h : ℝ) : ℝ :=
  (sqrt 3 / 4) * a^2 * h

noncomputable def surface_area_constraint (a h : ℝ) : ℝ :=
  2 * a * h + (sqrt 3 / 4) * a^2

theorem max_volume_prism (a h : ℝ) 
  (h_pos : h > 0) 
  (h_area : surface_area_constraint a h = 36) :
  prism_volume a h ≤ 432 := sorry

end max_volume_prism_l648_648795


namespace cost_of_purchase_l648_648036

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l648_648036


namespace checkerboard_sum_l648_648949

/-- A function to number cells row-wise. -/
def f (i j : ℕ) : ℕ := 15 * (i - 1) + j

/-- A function to number cells column-wise. -/
def g (i j : ℕ) : ℕ := 15 * (j - 1) + i

/-- The sum of the numbers in the cells with the same number in both numbering systems on a 15x15 grid. -/
theorem checkerboard_sum :
  (∑ i in Finset.range 15, f (i + 1) (i + 1)) = 1695 :=
by
  sorry

end checkerboard_sum_l648_648949


namespace convex_polytope_with_1989_vertices_assignment_l648_648810

theorem convex_polytope_with_1989_vertices_assignment :
  ∃ (f : Fin 1989 → ℤ), (∀ i : Fin 1989, ∏ j in (neighbors i), f j = -1) :=
sorry

end convex_polytope_with_1989_vertices_assignment_l648_648810


namespace range_abs_plus_one_l648_648231

 theorem range_abs_plus_one : 
   ∀ y : ℝ, (∃ x : ℝ, y = |x| + 1) ↔ y ≥ 1 := 
 by
   sorry
 
end range_abs_plus_one_l648_648231


namespace problem_a_proof_l648_648142

variables {A B C D M K : Point}
variables {triangle_ABC : Triangle A B C}
variables {incircle : Circle} (ht : touches incircle AC D) 
variables (hdm : diameter incircle D M) 
variables (bm_line : Line B M) (intersect_bm_ac : intersects bm_line AC K)

theorem problem_a_proof : 
  AK = DC :=
sorry

end problem_a_proof_l648_648142


namespace ellipse_equation_maximum_area_triangle_l648_648280

section EllipseEquation

variables {a b c : ℝ} (h1 : 0 < b) (h2 : b < a) (h3 : a * 1/2 = c) (h4 : a = 2) (h5 : b = real.sqrt 3) (h6 : c = 1)

theorem ellipse_equation :
  ∀ (x y : ℝ), (x^2) / 4 + (y^2) / 3 = 1 ↔ (x^2) / a^2 + (y^2) / b^2 = 1 :=
by
  intro x y
  sorry

end EllipseEquation

section MaximumArea

variables {a b c : ℝ} (h1 : a = 2) (h2 : b = real.sqrt 3) (h3 : c = 1) (h4 : ∀ (x y : ℝ), (x^2) / 4 + (y^2) / 3 = 1)

theorem maximum_area_triangle :
  ∀ (M F1 F2 : ℝ → ℝ → Prop), let area := real.sqrt 3 in
      α(Triangle M F1 F2) = area :=
by
  intro M F1 F2 area
  sorry

end MaximumArea

end ellipse_equation_maximum_area_triangle_l648_648280


namespace sum_of_primes_1_to_20_l648_648527

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648527


namespace compute_g_of_g_of_2_l648_648715

def g (x : ℚ) : ℚ := x⁻² + x⁻² / (1 - x⁻²)

theorem compute_g_of_g_of_2 : g (g 2) = 6624 / 4655 := by
  sorry

end compute_g_of_g_of_2_l648_648715


namespace difference_between_roots_l648_648645

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := -7
noncomputable def c : ℝ := 11

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b ^ 2 - 4 * a * c
  ((-b + Real.sqrt discriminant) / (2 * a), (-b - Real.sqrt discriminant) / (2 * a))

-- Extract the roots from the equation
noncomputable def r1_r2 := quadratic_roots a b c

noncomputable def r1 : ℝ := r1_r2.1
noncomputable def r2 : ℝ := r1_r2.2

-- Theorem statement: the difference between the roots is sqrt(5)
theorem difference_between_roots :
  |r1 - r2| = Real.sqrt 5 :=
  sorry

end difference_between_roots_l648_648645


namespace minimum_product_sum_dominoes_l648_648782

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

noncomputable def min_sum_product_dominoes : ℕ :=
  1 / 2 * sum_of_squares 100 - 1 / 2 * 5000

theorem minimum_product_sum_dominoes :
  let square_length := 10
  let num_dominoes := square_length * square_length / 2
  let total_sum := 100 * 101 * 201 / 6
  let correct_arrangement :=  - 5000 / 2
  num_dominoes = 50 ∧ correct_arrangement = -2500 ∧ 1 / 2 * total_sum - correct_arrangement = 166675 :=
begin
  sorry
end

end minimum_product_sum_dominoes_l648_648782


namespace elena_pens_l648_648639

theorem elena_pens (X Y : ℝ) 
  (h1 : X + Y = 12) 
  (h2 : 4 * X + 2.80 * Y = 40) :
  X = 5 :=
by
  sorry

end elena_pens_l648_648639


namespace total_cost_is_eight_times_l648_648052

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l648_648052


namespace sum_of_primes_between_1_and_20_l648_648499

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648499


namespace frog_problem_l648_648120

theorem frog_problem 
  (N : ℕ) 
  (h1 : N < 50) 
  (h2 : N % 2 = 1) 
  (h3 : N % 3 = 1) 
  (h4 : N % 4 = 1) 
  (h5 : N % 5 = 0) : 
  N = 25 := 
  sorry

end frog_problem_l648_648120


namespace find_buying_price_per_gram_l648_648843

variable (side_length : ℝ) (density : ℝ) (profit : ℝ) (selling_multiplier : ℝ)

def volume := side_length ^ 3
def mass := density * volume
def buying_price_per_gram := profit / (selling_multiplier * mass - mass)

theorem find_buying_price_per_gram (h1 : side_length = 6) (h2 : density = 19)
  (h3 : profit = 123120) (h4 : selling_multiplier = 1.5) :
  buying_price_per_gram side_length density profit selling_multiplier = 60 := by
  sorry

end find_buying_price_per_gram_l648_648843


namespace binomial_constant_term_l648_648350

theorem binomial_constant_term :
  ∃ (r : ℕ), (6 - 2 * r = 0) ∧ (binomial 6 r * 6^r = 4320) :=
sorry

end binomial_constant_term_l648_648350


namespace quadratic_distinct_roots_k_range_l648_648774

theorem quadratic_distinct_roots_k_range (k : ℝ) :
  (k - 1) * x^2 + 2 * x - 2 = 0 ∧ 
  ∀ Δ, Δ = 2^2 - 4*(k-1)*(-2) ∧ Δ > 0 ∧ (k ≠ 1) ↔ k > 1/2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_distinct_roots_k_range_l648_648774


namespace triangle_inequality_l648_648997

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality :
  ∃ (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 1 ∧ b = 2 ∧ c = 3) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 2 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 3 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) :=
by
  sorry

end triangle_inequality_l648_648997


namespace sum_of_primes_1_to_20_l648_648550

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648550


namespace certain_event_is_A_l648_648916

def conditions (option_A option_B option_C option_D : Prop) : Prop :=
  option_A ∧ ¬option_B ∧ ¬option_C ∧ ¬option_D

theorem certain_event_is_A 
  (option_A option_B option_C option_D : Prop)
  (hconditions : conditions option_A option_B option_C option_D) : 
  ∀ e, (e = option_A) := 
by
  sorry

end certain_event_is_A_l648_648916


namespace harly_adopts_percentage_l648_648751

/-- Definitions for the conditions -/
def initial_dogs : ℝ := 80
def dogs_taken_back : ℝ := 5
def dogs_left : ℝ := 53

/-- Define the percentage of dogs adopted out -/
def percentage_adopted (P : ℝ) := P

/-- Lean 4 statement where we prove that if the given conditions are met, then the percentage of dogs initially adopted out is 40 -/
theorem harly_adopts_percentage : 
  ∃ P : ℝ, 
    (initial_dogs - (percentage_adopted P / 100 * initial_dogs) + dogs_taken_back = dogs_left) 
    ∧ P = 40 :=
by
  sorry

end harly_adopts_percentage_l648_648751


namespace next_term_geometric_sequence_l648_648478

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l648_648478


namespace books_in_special_collection_l648_648596

theorem books_in_special_collection (B : ℕ) :
  (∃ returned not_returned loaned_out_end  : ℝ, 
    loaned_out_end = 54 ∧ 
    returned = 0.65 * 60.00000000000001 ∧ 
    not_returned = 60.00000000000001 - returned ∧ 
    B = loaned_out_end + not_returned) → 
  B = 75 :=
by 
  intro h
  sorry

end books_in_special_collection_l648_648596


namespace find_ab_find_max_min_l648_648574

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 5

-- Tangent line condition at x = 1
def tangent_line_condition (a b : ℝ) : Prop :=
  let f' := λ x, 3 * x^2 + 2 * a * x + b in
  f'(1) = 3 ∧ f(1 a b) = 4

-- Prove that given the tangent line condition, a = 2 and b = -4
theorem find_ab : 
  ∃ (a b : ℝ), tangent_line_condition a b ∧ a = 2 ∧ b = -4 :=
sorry

-- Define the intervals for maxima and minima
def interval := Set.Icc (-3 : ℝ) 1

-- Prove the maxima and minima in the interval [-3,1]
theorem find_max_min (a b : ℝ) (ha : a = 2) (hb : b = -4) : 
  tangent_line_condition a b →
  let f_x := f x a b in
  let fx_values := [f_x (-2), f_x (2/3), f_x (-3), f_x 1] in
  (Set.max fx_values = 13) ∧ (Set.min fx_values = 95 / 27) :=
sorry

end find_ab_find_max_min_l648_648574


namespace car_average_speed_l648_648888

-- Define speeds and times
def speed1 : ℝ := 90
def speed2 : ℝ := 55
def time1 : ℝ := 1
def time2 : ℝ := 1

-- Definition for the total distance
def total_distance : ℝ := speed1 + speed2

-- Definition for the total time
def total_time : ℝ := time1 + time2

-- Definition for average speed
def average_speed (total_distance total_time : ℝ) : ℝ := total_distance / total_time

theorem car_average_speed :
  average_speed total_distance total_time = 72.5 := by
  sorry

end car_average_speed_l648_648888


namespace math_proof_problem_l648_648383

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def problem_statement (M : Set ℕ) (A : Set ℕ) : Prop :=
  ∀ a ∈ A, a ∈ M ∧
  (∀ x y ∈ A, x ≠ y → is_rel_prime x y) →
  ∃ B ⊆ A, B.card ≥ 15 ∧ ∀ b ∈ B, is_prime b

theorem math_proof_problem :
  let M := {n | 1 ≤ n ∧ n ≤ 2008}
  let A := {a | a ∈ M ∧ (∃ S ⊆ M, S.card = 30 ∧ ∀ x y ∈ S, x ≠ y → is_rel_prime x y)}
  problem_statement M A :=
sorry

end math_proof_problem_l648_648383


namespace bummed_out_numbers_l648_648969

def is_bummed_out (n : ℕ) : Prop :=
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ ( ( (x^2 : ℝ) / y ).floor + ( (y^2 : ℝ) / x ).floor = n )

theorem bummed_out_numbers :
  {n : ℕ | is_bummed_out n} = {2, 6, 8, 10} :=
by sorry

end bummed_out_numbers_l648_648969


namespace joey_age_sum_l648_648368

theorem joey_age_sum :
  ∃ (C J Z n : ℤ), Z = 2 ∧
  J = C + 4 ∧
  (∀ m : ℤ, 0 ≤ m ∧ m < 10 → (C + m) % (Z + m) = 0) ∧
  (J + n = 2 * (Z + n)) ∧
  (J = 54) ∧
  (n = 50) ∧
  (C - 2 = 48) ∧
  (J + n = 104) ∧
  (104.digitSum = 5) :=
by
  sorry

end joey_age_sum_l648_648368


namespace distance_ratio_l648_648199

-- Define the distances as given in the conditions
def distance_from_city_sky_falls := 8 -- Distance in miles
def distance_from_city_rocky_mist := 400 -- Distance in miles

theorem distance_ratio : distance_from_city_rocky_mist / distance_from_city_sky_falls = 50 := 
by
  -- Proof skipped
  sorry

end distance_ratio_l648_648199


namespace largest_possible_red_socks_l648_648961

theorem largest_possible_red_socks (t r g : ℕ) (h1 : t = r + g) (h2 : t ≤ 3000)
    (h3 : (r * (r - 1) + g * (g - 1)) * 5 = 3 * t * (t - 1)) :
    r ≤ 1199 :=
sorry

end largest_possible_red_socks_l648_648961


namespace find_f3_l648_648868

theorem find_f3 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f(x) + 2 * f(1 - x) = 3 * x^2) : f(3) = -1 :=
by
  sorry

end find_f3_l648_648868


namespace total_cost_shorts_tshirt_boots_shinguards_l648_648018

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l648_648018


namespace max_clocks_l648_648790

theorem max_clocks (n : ℕ) (h m : Fin n → ℕ) (cond1 : ∀ i j, i ≠ j → h i ≠ h j ∧ m i ≠ m j)
  (cond2 : ∀ i, 1 ≤ h i ∧ h i ≤ 12)
  (cond3 : ∀ i, 0 ≤ m i ∧ m i < 60)
  (cond4 : ∑ i, h i - 1 = ∑ i, h i - n)
  (cond5 : ∑ i, m i - 1 = ∑ i, m i - 59) :
  n ≤ 11 :=
by sorry

end max_clocks_l648_648790


namespace number_of_distinct_f_values_l648_648259

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def f (x : ℝ) : ℤ := (floor (3 * x)) + (floor (4 * x)) + (floor (5 * x)) + (floor (6 * x))

theorem number_of_distinct_f_values :
  (finset.univ.image (λ x : ℝ, f x)).card = 1201 :=
sorry

end number_of_distinct_f_values_l648_648259


namespace min_value_f_l648_648100

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 1/2 * Real.cos (2 * x) - 1

theorem min_value_f : ∃ x : ℝ, f x = -5/2 := sorry

end min_value_f_l648_648100


namespace Ofelia_savings_December_l648_648007

/-- Define a function that models Ofelia's savings. -/
def savings (n : ℕ) : ℕ :=
  if n = 1 then 10
  else 3 * savings (n - 1)

/-- Ofelia's savings in December (month 12) are $1,771,470. 
  Given the conditions on her savings:
  1. Ofelia saves three times the amount she saved in the previous month.
  2. In January, she saved $10.
 -/
theorem Ofelia_savings_December : savings 12 = 1771470 :=
by
  sorry

end Ofelia_savings_December_l648_648007


namespace total_distance_traveled_l648_648582

-- Definitions based on the conditions
def speed1 := 65 -- miles per hour
def time1 := 3 -- hours

def speed2 := 45 -- miles per hour
def time2 := 2 -- hours

def rest_stop := 0.5 -- 30 minutes in hours, does not affect distance

def speed3 := 55 -- miles per hour
def time3 := 4 -- hours

-- Total distance
theorem total_distance_traveled :
  let distance1 := speed1 * time1 in
  let distance2 := speed2 * time2 in
  let distance3 := speed3 * time3 in
  distance1 + distance2 + distance3 = 505 :=
by
  sorry

end total_distance_traveled_l648_648582


namespace min_value_36_l648_648827

noncomputable def minimum_value (γ δ : ℝ) : ℝ :=
  (3 * real.cos γ + 4 * real.sin δ - 7)^2 + (3 * real.sin γ + 4 * real.cos δ - 12)^2

theorem min_value_36 (γ δ : ℝ) : minimum_value γ δ ≥ 36 :=
sorry

end min_value_36_l648_648827


namespace trig_identity_proof_l648_648245

theorem trig_identity_proof
  (h1: Float.sin 50 = Float.cos 40)
  (h2: Float.tan 45 = 1)
  (h3: Float.tan 10 = Float.sin 10 / Float.cos 10)
  (h4: Float.sin 80 = Float.cos 10) :
  Float.sin 50 * (Float.tan 45 + Float.sqrt 3 * Float.tan 10) = 1 :=
by
  sorry

end trig_identity_proof_l648_648245


namespace pyramid_surface_area_l648_648082

theorem pyramid_surface_area
  (S1 S2 S : ℝ)
  (h1 : 0 ≤ S1)
  (h2 : 0 ≤ S2)
  (h3 : ∃ S0 : ℝ, (S = (Real.sqrt S1 + Real.sqrt S2) * (Real.sqrt (Real.sqrt S1) + Real.sqrt (Real.sqrt S2))^2)) : 
  S = (Real.sqrt S1 + Real.sqrt S2) * (Real.sqrt (Real.sqrt S1) + Real.sqrt (Real.sqrt S2))^2 :=
by
  sorry

end pyramid_surface_area_l648_648082


namespace gcd_sum_abcde_edcba_l648_648093

-- Definition to check if digits are consecutive
def consecutive_digits (a b c d e : ℤ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4

-- Definition of the five-digit number in the form abcde
def abcde (a b c d e : ℤ) : ℤ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- Definition of the five-digit number in the form edcba
def edcba (a b c d e : ℤ) : ℤ :=
  10000 * e + 1000 * d + 100 * c + 10 * b + a

-- Definition which sums both abcde and edcba
def sum_abcde_edcba (a b c d e : ℤ) : ℤ :=
  abcde a b c d e + edcba a b c d e

-- Lean theorem statement for the problem
theorem gcd_sum_abcde_edcba (a b c d e : ℤ) (h : consecutive_digits a b c d e) :
  Int.gcd (sum_abcde_edcba a b c d e) 11211 = 11211 :=
by
  sorry

end gcd_sum_abcde_edcba_l648_648093


namespace length_of_uncovered_side_l648_648598

-- Define the conditions of the problem
def area_condition (L W : ℝ) : Prop := L * W = 210
def fencing_condition (L W : ℝ) : Prop := L + 2 * W = 41

-- Define the proof statement
theorem length_of_uncovered_side (L W : ℝ) (h_area : area_condition L W) (h_fence : fencing_condition L W) : 
  L = 21 :=
  sorry

end length_of_uncovered_side_l648_648598


namespace wood_length_equation_l648_648935

theorem wood_length_equation (x : ℝ) : 
  (∃ r : ℝ, r - x = 4.5 ∧ r/2 + 1 = x) → 1/2 * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l648_648935


namespace true_proposition_is_D_l648_648610

theorem true_proposition_is_D :
  ∃ (P : Prop), (P = (∀ {a b : ℝ}, a^2 = b^2 → a = b) → false) ∧
                (P = (∃ (Δ : Type) [is_triangle Δ] [H : orthocenter Δ], True) → true) ∧
                (P = (∀ {L1 L2 L3 : set Point}, intersecting L1 L2 L3 → equal_corresponding_angles L1 L2 L3) → false) ∧
                (P = (∀ {L1 L2 : set Point}, parallel L1 L2 → equal_or_supplementary_angles L1 L2) → true) :=
by
  sorry

end true_proposition_is_D_l648_648610


namespace collinearity_of_XC_and_B_l648_648820

noncomputable def are_collinear (A B C : Point) : Prop := ∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l

variables (Γ₁ Γ₂ : Circle)
variables (A B P Q X C : Point)
variables (PQ : Line)

-- Conditions
axiom intersecting_circles : Γ₁ ∩ Γ₂ = {A, B}
axiom tangent_points : is_tangent PQ Γ₁ P ∧ is_tangent PQ Γ₂ Q
axiom intersection_of_tangents : X = tangents_intersection (circumcircle P Q A) P Q
axiom reflection_point : C = reflect B PQ

-- Theorem
theorem collinearity_of_XC_and_B : are_collinear X C B :=
by sorry

end collinearity_of_XC_and_B_l648_648820


namespace coefficient_x2_in_binomial_expansion_l648_648430

theorem coefficient_x2_in_binomial_expansion : 
  let n := 7 in 
  let k := 2 in 
  (binomial n k) = 21 := 
by 
  sorry

end coefficient_x2_in_binomial_expansion_l648_648430


namespace max_diff_mass_l648_648399

theorem max_diff_mass (m ε : ℝ) (h_m : m = 20) (h_ε : ε = 0.2) : 
    ∃ Δ, Δ = 0.4 ∧
    ∀ (x y : ℝ), (x ∈ set.Icc (m - ε) (m + ε)) → (y ∈ set.Icc (m - ε) (m + ε)) → |x - y| ≤ Δ :=
by
    sorry

end max_diff_mass_l648_648399


namespace parallel_lines_circle_chords_l648_648459

/-- Three equally spaced parallel lines intersect a circle, forming three chords of lengths 40, 40, and 36. 
    The distance between two adjacent parallel lines is √1188 and the radius of the circle is √357. -/
theorem parallel_lines_circle_chords : 
  ∃ (d r : ℝ), 
    (d ^ 2 = 1188) ∧ 
    (r ^ 2 = 357) ∧ 
    (∃ (A B C D E F P Q : point) (O : point) (radius : ℝ),
      ∥A - B∥ = 40 ∧ 
      ∥E - F∥ = 40 ∧ 
      ∥C - D∥ = 36 ∧
      ∥O - A∥ = radius ∧ 
      ∥O - B∥ = radius ∧ 
      ∥O - C∥ = radius ∧ 
      ∥O - D∥ = radius ∧ 
      distance_between_lines (line_of_points A B) (line_of_points E F) = d) := 
sorry

end parallel_lines_circle_chords_l648_648459


namespace quadratic_function_count_l648_648693

open Finset

def is_even (n : ℕ) := ∃ k, n = 2 * k

theorem quadratic_function_count :
  let S := (range 10).filter (λ x, x > 0) in
  (card (S.powerset.filter (λ t, t.card = 3 ∧ is_even (t.sum)))) = 264 :=
by
  sorry

end quadratic_function_count_l648_648693


namespace period_and_max_value_f_l648_648875

def f (x : ℝ) : ℝ := sin (x / 3) + cos (x / 3)

theorem period_and_max_value_f :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ y > 0, y < T → ∃ x, f (x + y) ≠ f x) ∧
  (∃ M, ∀ x, f x ≤ M ∧ (∃ x, f x = M)) :=
begin
  sorry
end

end period_and_max_value_f_l648_648875


namespace minimize_average_cost_l648_648990

noncomputable def average_comprehensive_cost (x : ℝ) : ℝ :=
  560 + 48 * x + 2160 * 10^6 / (2000 * x)

theorem minimize_average_cost : 
  ∃ x_min : ℝ, x_min ≥ 10 ∧ 
  ∀ x ≥ 10, average_comprehensive_cost x ≥ average_comprehensive_cost x_min :=
sorry

end minimize_average_cost_l648_648990


namespace find_girls_attending_l648_648686

-- Define the number of students and attendees
def total_students := 1500
def total_attendees := 975

-- Define the relations given in the problem
def girls_attend_ratio := 3 / 4
def boys_attend_ratio := 2 / 5

-- The equation conditions
def total_condition (g b : ℕ) := g + b = total_students
def attendance_condition (g b : ℕ) := (girls_attend_ratio * g + boys_attend_ratio * b : ℚ) = total_attendees

-- Statement of the proof problem
theorem find_girls_attending (g b : ℕ) (hg : total_condition g b) (ha : attendance_condition g b) : 
  (girls_attend_ratio * g = 803 : ℚ) :=
by
  -- Exact mathematical steps or proof would go here
  sorry

end find_girls_attending_l648_648686


namespace max_value_of_m_l648_648305

noncomputable def f (x m n : ℝ) : ℝ := x^2 + m*x + n^2
noncomputable def g (x m n : ℝ) : ℝ := x^2 + (m+2)*x + n^2 + m + 1

theorem max_value_of_m (m n t : ℝ) :
  (∀(t : ℝ), f t m n ≥ 0 ∨ g t m n ≥ 0) → m ≤ 1 :=
by
  intro h
  sorry

end max_value_of_m_l648_648305


namespace sum_of_primes_1_to_20_l648_648533

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648533


namespace unique_solution_condition_l648_648753

theorem unique_solution_condition (a b : ℝ) : 
  (a ≠ 10) ∧ (b ≠ 5 / 2) → ∃! x : ℝ, 5 * x - 7 + a = 2 * b * x + 3 :=
begin
  sorry
end

end unique_solution_condition_l648_648753


namespace all_lines_pass_through_single_point_l648_648274

variable {α : Type*} [LinearOrder α] [MetricSpace α] [CompleteMetricSpace α]

/-- Given a finite number of pairwise non-parallel lines on a plane, 
such that through the intersection point of any two of them passes another one of the given lines, 
prove that all these lines pass through a single point. -/

theorem all_lines_pass_through_single_point 
  (P : Finset (Line α)) 
  (h_non_par : ∀ l₁ l₂ ∈ P, l₁ ≠ l₂ → l₁ ∩ l₂ ≠ ∅)
  (h_inter : ∀ l₁ l₂ l₃ ∈ P, l₁ ≠ l₂ → l₂ ≠ l₃ → l₃ ≠ l₁ → (l₁ ∩ l₂ ∩ l₃ ≠ ∅)) :
  ∃ p, ∀ l ∈ P, p ∈ l :=
sorry

end all_lines_pass_through_single_point_l648_648274


namespace translation_transform_l648_648139

theorem translation_transform :
  ∀ (x : ℝ), 5 * sin (2 * (x + π / 12)) = 5 * sin (2 * x + π / 6) :=
by
  sorry

end translation_transform_l648_648139


namespace even_function_f4_l648_648201

noncomputable def f1 (x : ℝ) : ℝ := -3 * x + 2
noncomputable def f2 (x : ℝ) : ℝ := Real.log 2 x
noncomputable def f3 (x : ℝ) : ℝ := x ^ 3
noncomputable def f4 (x : ℝ) : ℝ := abs x

theorem even_function_f4 : ∀ (x : ℝ), f4 x = f4 (-x) :=
by {
  intro x,
  unfold f4,
  simp only [abs_neg],
}

end even_function_f4_l648_648201


namespace percentage_discount_l648_648858

theorem percentage_discount (discounted_price original_price : ℝ) (h1 : discounted_price = 560) (h2 : original_price = 700) :
  (original_price - discounted_price) / original_price * 100 = 20 :=
by
  simp [h1, h2]
  sorry

end percentage_discount_l648_648858


namespace Daisy_vs_Bess_l648_648246

-- Define the conditions
def Bess_daily : ℕ := 2
def Brownie_multiple : ℕ := 3
def total_pails_per_week : ℕ := 77
def days_per_week : ℕ := 7

-- Define the weekly production for Bess
def Bess_weekly : ℕ := Bess_daily * days_per_week

-- Define the weekly production for Brownie
def Brownie_weekly : ℕ := Brownie_multiple * Bess_weekly

-- Farmer Red's total weekly milk production is the sum of Bess, Brownie, and Daisy's production
-- We need to prove the difference in weekly production between Daisy and Bess is 7 pails.
theorem Daisy_vs_Bess (Daisy_weekly : ℕ) (h : Bess_weekly + Brownie_weekly + Daisy_weekly = total_pails_per_week) :
  Daisy_weekly - Bess_weekly = 7 :=
by
  sorry

end Daisy_vs_Bess_l648_648246


namespace base_of_isosceles_triangle_l648_648167

theorem base_of_isosceles_triangle
  (O A B C : Type*)
  [MetricSpace O]
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (circ_center : O)
  (area_circle : ℝ)
  (isosceles_triangle : Prop)
  (chord : Prop)
  (distance_OA : ℝ) 
  (point_O_inside : Prop) 
  (area_circle_def : area_circle = 100 * Real.pi)
  (OA_eq : distance_OA = 6)
  (O_inside_def : point_O_inside = True)
  (isosceles_triangle_def : isosceles_triangle = (AB = AC))
  (BC_chord_def : chord = (BC.is_chord))
: length BC = 16 := sorry

end base_of_isosceles_triangle_l648_648167


namespace bill_sun_vs_sat_l648_648008

theorem bill_sun_vs_sat (B_Sat B_Sun J_Sun : ℕ) 
  (h1 : B_Sun = 6)
  (h2 : J_Sun = 2 * B_Sun)
  (h3 : B_Sat + B_Sun + J_Sun = 20) : 
  B_Sun - B_Sat = 4 :=
by
  sorry

end bill_sun_vs_sat_l648_648008


namespace heptagon_angle_sum_lt_450_l648_648175

open Real EuclideanGeometry

-- Define the problem conditions
def inscribed_heptagon (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : Point) (O : Point) : Prop :=
  Concyclic [A₁, A₂, A₃, A₄, A₅, A₆, A₇] ∧
  CenterOf (CircleCircumscribedTo [A₁, A₂, A₃, A₄, A₅, A₆, A₇]) = O ∧
  Inside (Polygon [A₁, A₂, A₃, A₄, A₅, A₆, A₇]) O

-- Define the angles sum condition to be proved
def angle_sum_condition (A₁ A₃ A₅ : Point) : Prop :=
  ∠ A₁ + ∠ A₃ + ∠ A₅ < 450

-- The main proof problem statement
theorem heptagon_angle_sum_lt_450 
  (A₁ A₂ A₃ A₄ A₅ A₆ A₇ O : Point)
  (h : inscribed_heptagon A₁ A₂ A₃ A₄ A₅ A₆ A₇ O) :
  angle_sum_condition A₁ A₃ A₅ := 
sorry

end heptagon_angle_sum_lt_450_l648_648175


namespace find_k_to_ensure_collinearity_l648_648379

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]

theorem find_k_to_ensure_collinearity 
  (O A B C D : V)
  (h : 5 • (A - O) - 3 • (B - O) + 4 • (C - O) + k • (D - O) = 0)
  (h_collinear : collinear ℝ {A, B, C, D}) : 
  k = -6 :=
sorry

end find_k_to_ensure_collinearity_l648_648379


namespace min_value_of_A_abs_x1_sub_x2_l648_648720

theorem min_value_of_A_abs_x1_sub_x2 :
  let f (x : ℝ) := Real.sin(2017 * x + π / 6) + Real.cos(2017 * x + π / 3) in
  (∀ x : ℝ, f x ≤ 2) ∧ 
  (∃ x1 x2 : ℝ, ∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) →
  ∃ x1 x2 : ℝ, 2 * |x1 - x2| = 2 * (π / 2017) :=
by
  sorry

end min_value_of_A_abs_x1_sub_x2_l648_648720


namespace value_of_expression_l648_648256

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := 
by
  sorry

end value_of_expression_l648_648256


namespace sum_of_primes_between_1_and_20_l648_648492

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648492


namespace droid_weekly_coffee_consumption_l648_648637

noncomputable def weekly_consumption_A : ℕ :=
  (3 * 5) + 4 + 2 + 1 -- Weekdays + Saturday + Sunday + Monday increase

noncomputable def weekly_consumption_B : ℕ :=
  (2 * 5) + 3 + (1 - 1 / 2) -- Weekdays + Saturday + Sunday decrease

noncomputable def weekly_consumption_C : ℕ :=
  (1 * 5) + 2 + 1 -- Weekdays + Saturday + Sunday

theorem droid_weekly_coffee_consumption :
  weekly_consumption_A = 22 ∧ weekly_consumption_B = 14 ∧ weekly_consumption_C = 8 :=
by 
  sorry

end droid_weekly_coffee_consumption_l648_648637


namespace particle_return_to_A_l648_648183

theorem particle_return_to_A (n : ℕ) (h : n = 10) : 
  let moves := 10,
  let start := 'A' : Char,
  let vertices := ['A', 'B', 'C', 'D', 'E'] : List Char,
  PentagonalPaths start moves start vertices = 254 :=
sorry

end particle_return_to_A_l648_648183


namespace tommys_profit_l648_648897

-- Definitions of the conditions
def crateA_cost : ℕ := 220
def crateB_cost : ℕ := 375
def crateC_cost : ℕ := 180

def crateA_count : ℕ := 2
def crateB_count : ℕ := 3
def crateC_count : ℕ := 1

def crateA_capacity : ℕ := 20
def crateB_capacity : ℕ := 25
def crateC_capacity : ℕ := 30

def crateA_rotten : ℕ := 4
def crateB_rotten : ℕ := 5
def crateC_rotten : ℕ := 3

def crateA_price_per_kg : ℕ := 5
def crateB_price_per_kg : ℕ := 6
def crateC_price_per_kg : ℕ := 7

-- Calculations based on the conditions
def total_cost : ℕ := crateA_cost + crateB_cost + crateC_cost

def sellable_weightA : ℕ := crateA_count * crateA_capacity - crateA_rotten
def sellable_weightB : ℕ := crateB_count * crateB_capacity - crateB_rotten
def sellable_weightC : ℕ := crateC_count * crateC_capacity - crateC_rotten

def revenueA : ℕ := sellable_weightA * crateA_price_per_kg
def revenueB : ℕ := sellable_weightB * crateB_price_per_kg
def revenueC : ℕ := sellable_weightC * crateC_price_per_kg

def total_revenue : ℕ := revenueA + revenueB + revenueC

def profit : ℕ := total_revenue - total_cost

-- The theorem we want to verify
theorem tommys_profit : profit = 14 := by
  sorry

end tommys_profit_l648_648897


namespace sum_first_three_prime_values_of_f_l648_648157

noncomputable def f (n : ℕ) : ℕ := n^2 + 6*n + 11

def is_prime (p : ℕ) : Prop := Nat.Prime p

def first_three_primes_of_f (f : ℕ → ℕ) : List ℕ :=
  List.filter (λ x => is_prime (f x)) (List.range (1000)) |>.take 3
  |>.map f

theorem sum_first_three_prime_values_of_f :
  let primes := first_three_primes_of_f f in
  primes.sum = 753 :=
by
  let primes := first_three_primes_of_f f
  sorry

end sum_first_three_prime_values_of_f_l648_648157


namespace sums_not_all_different_l648_648809

theorem sums_not_all_different (n : ℕ) : 
  ¬(∃ (table : Fin n → Fin n → ℤ), 
    (∀ (i j : Fin n), table i j = -1 ∨ table i j = 0 ∨ table i j = 1) ∧
    (∀ (i j : Fin n), ∑ k, table i k ≠ ∑ l, table i l ∨ k ≠ l ) ∧
    (∀ (i j : Fin n), ∑ k, table k j ≠ ∑ l, table l j ∨ k ≠ l ) ∧
    ∑ i, table i i ≠ ∑ j, table j (n - 1 - j) ∧
    (∀ m p, m < 2n + 2 → ∃! q, (∑ i, table i i = q ∨ ∑ j, table j (n - 1 - j) = q))) :=
sorry

end sums_not_all_different_l648_648809


namespace sale_price_is_207_l648_648161

-- Define a namespace for our problem
namespace BicyclePrice

-- Define the conditions as constants
def priceAtStoreP : ℝ := 200
def regularPriceIncreasePercentage : ℝ := 0.15
def salePriceDecreasePercentage : ℝ := 0.10

-- Define the regular price at Store Q
def regularPriceAtStoreQ : ℝ := priceAtStoreP * (1 + regularPriceIncreasePercentage)

-- Define the sale price at Store Q
def salePriceAtStoreQ : ℝ := regularPriceAtStoreQ * (1 - salePriceDecreasePercentage)

-- The final theorem we need to prove
theorem sale_price_is_207 : salePriceAtStoreQ = 207 := by
  sorry

end BicyclePrice

end sale_price_is_207_l648_648161


namespace problem_proof_l648_648279

variable (P : Set ℤ)

-- Conditions
axiom condition1 : ∃ a ∈ P, a > 0 ∧ ∃ b ∈ P, b < 0
axiom condition2 : ∃ c ∈ P, Odd c ∧ ∃ d ∈ P, Even d
axiom condition3 : -1 ∉ P
axiom condition4 : ∀ x y ∈ P, x + y ∈ P

-- Proposition statements
theorem problem_proof : (1 ∉ P) ∧ (2 ∉ P) ∧ (0 ∈ P) := by
  sorry

end problem_proof_l648_648279


namespace matthew_and_zac_strawberries_l648_648370

theorem matthew_and_zac_strawberries (total_strawberries jonathan_and_matthew_strawberries zac_strawberries : ℕ) (h1 : total_strawberries = 550) (h2 : jonathan_and_matthew_strawberries = 350) (h3 : zac_strawberries = 200) : (total_strawberries - (jonathan_and_matthew_strawberries - zac_strawberries) = 400) :=
by { sorry }

end matthew_and_zac_strawberries_l648_648370


namespace zero_vectors_count_l648_648418

variable (A B C D E F O P Q : Type)
variables (AB BC CA AC BD CD FQ QP EF EP OA OB : E → E)

theorem zero_vectors_count :
  (AB + BC + CA = 0) ∧
  (AB - AC + BD - CD = 0) ∧
  (FQ + QP + EF - EP = 0) ∧
  (OA - OB + AB = 0) →
  4 = 4 :=
by
  intros h
  sorry

end zero_vectors_count_l648_648418


namespace ball_picking_problem_proof_l648_648451

-- Define the conditions
def red_balls : ℕ := 8
def white_balls : ℕ := 7

-- Define the questions
def num_ways_to_pick_one_ball : ℕ :=
  red_balls + white_balls

def num_ways_to_pick_two_different_color_balls : ℕ :=
  red_balls * white_balls

-- Define the correct answers
def correct_answer_to_pick_one_ball : ℕ := 15
def correct_answer_to_pick_two_different_color_balls : ℕ := 56

-- State the theorem to be proved
theorem ball_picking_problem_proof :
  (num_ways_to_pick_one_ball = correct_answer_to_pick_one_ball) ∧
  (num_ways_to_pick_two_different_color_balls = correct_answer_to_pick_two_different_color_balls) :=
by
  sorry

end ball_picking_problem_proof_l648_648451


namespace sum_of_primes_between_1_and_20_l648_648540

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648540


namespace cars_meet_distance_from_midpoint_l648_648692

theorem cars_meet_distance_from_midpoint :
  ∀ (A B C : ℝ) (d_AB : ℝ) (s1 s2 : ℝ)
  (dA_C dB_C time when_meet : ℝ),
  d_AB = 240 →
  s1 = 60 → s2 = 80 →
  dA_C = d_AB / 2 →
  dB_C = d_AB / 2 →
  time = d_AB / (s1 + s2) →
  when_meet = (s1 * time) →
abs((when_meet - dA_C / 1) : ℝ) *
  = 17.14 := by
 sorry

end cars_meet_distance_from_midpoint_l648_648692


namespace total_cost_is_eight_times_l648_648011

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l648_648011


namespace find_original_money_sandy_took_l648_648415

noncomputable def originalMoney (remainingMoney : ℝ) (clothingPercent electronicsPercent foodPercent additionalSpendPercent salesTaxPercent : ℝ) : Prop :=
  let X := (remainingMoney / (1 - ((clothingPercent + electronicsPercent + foodPercent) + additionalSpendPercent) * (1 + salesTaxPercent)))
  abs (X - 397.73) < 0.01

theorem find_original_money_sandy_took :
  originalMoney 140 0.25 0.15 0.10 0.20 0.08 :=
sorry

end find_original_money_sandy_took_l648_648415


namespace find_cosine_lateral_face_angle_l648_648460

structure RegularTriangularPyramid where
  side_length : ℝ

noncomputable def cos_lateral_face_angle (P : RegularTriangularPyramid)
  (h : ∀ (C : P.side_length) (α : ℝ), α = acos (2/3)) : ℝ :=
  1/7

theorem find_cosine_lateral_face_angle (P : RegularTriangularPyramid)
  (h : ∀ (C : P.side_length) (α : ℝ), α = acos (2/3)) : 
  cos_lateral_face_angle P h = 1/7 := 
sorry

end find_cosine_lateral_face_angle_l648_648460


namespace shop_discount_percentage_l648_648448

-- Definitions based on conditions
def original_price := 800
def price_paid := 560
def discount_amount := original_price - price_paid
def percentage_discount := (discount_amount / original_price) * 100

-- Proposition to prove
theorem shop_discount_percentage : percentage_discount = 30 := by
  sorry

end shop_discount_percentage_l648_648448


namespace sunzi_wood_problem_l648_648938

theorem sunzi_wood_problem (x : ℝ) :
  (∃ (length_of_rope : ℝ), length_of_rope = x + 4.5 ∧
    ∃ (half_length_of_rope : ℝ), half_length_of_rope = length_of_rope / 2 ∧ 
      (half_length_of_rope + 1 = x)) ↔ 
  (1 / 2 * (x + 4.5) = x - 1) :=
by
  sorry

end sunzi_wood_problem_l648_648938


namespace problem_number_of_true_propositions_l648_648439

open Set

variable {α : Type*} {A B : Set α}

def card (s : Set α) : ℕ := sorry -- The actual definition of cardinality is complex and in LF (not imperative here).

-- Statement of the problem translated into a Lean statement
theorem problem_number_of_true_propositions :
  (∀ {A B : Set ℕ}, A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B) ∧
  (∀ {A B : Set ℕ}, A ⊆ B → card A ≤ card B) ∧
  (∀ {A B : Set ℕ}, A ⊂ B → card A < card B) →
   (3 = 3) :=
by 
  sorry


end problem_number_of_true_propositions_l648_648439


namespace y_intercept_of_line_l648_648911

theorem y_intercept_of_line (y : ℝ) : (∃ x : ℝ, x = 0 ∧ (x - y + 3 = 0)) → y = 3 :=
by
  intro h
  rcases h with ⟨x, hx, hline⟩
  rw [hx, sub_zero] at hline
  linarith

end y_intercept_of_line_l648_648911


namespace find_students_who_like_basketball_not_table_tennis_l648_648786

open set

variable (U A B : Finset ℕ)
variable (students_total like_basketball like_tabletennis neither_sport : ℕ)

-- Define the conditions
def conditions :
  students_total = 42 ∧          -- Total number of students in the class
  like_basketball = 20 ∧         -- Number of students who like basketball
  like_tabletennis = 25 ∧        -- Number of students who like table tennis
  neither_sport = 12             -- Number of students who do not like either sport
: Prop :=
students_total = 42 ∧
like_basketball = 20 ∧
like_tabletennis = 25 ∧
neither_sport = 12

-- Define the problem statement
theorem find_students_who_like_basketball_not_table_tennis
  (h_cond : conditions students_total like_basketball like_tabletennis neither_sport) :
  let U := (Finset.range students_total).erase 0 in
  let A := (Finset.range like_basketball).erase 0 in
  let B := (Finset.range like_tabletennis).erase 0 in
  |A \ B| = 5 := sorry

end find_students_who_like_basketball_not_table_tennis_l648_648786


namespace domain_range_sum_l648_648088

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem domain_range_sum (m n : ℝ) (hmn : ∀ x, m ≤ x ∧ x ≤ n → (f x = 3 * x)) : m + n = -1 :=
by
  sorry

end domain_range_sum_l648_648088


namespace P_sufficient_not_necessary_for_Q_l648_648390

def P : set ℝ := {1, 2, 3, 4}
def Q : set ℝ := {x | 0 < x ∧ x < 5}

theorem P_sufficient_not_necessary_for_Q (x : ℝ) :
  (x ∈ P → x ∈ Q) ∧ (¬(x ∈ Q → x ∈ P)) :=
sorry

end P_sufficient_not_necessary_for_Q_l648_648390


namespace digit_assignment_count_is_correct_l648_648798

open Finset

/-
In how many different ways can the digits 0, 1, 2, ..., 9 be placed into the following scheme so that a correct addition is obtained?

A
BC
DEF
———
CHJK
-/
noncomputable def count_valid_digit_assignments : ℕ := 60

theorem digit_assignment_count_is_correct :
  ∃ (A B C D E F H J K : ℕ), 
    {A, B, C, D, E, F, H, J, K}.card = 9 ∧
    A ∈ (range 10) ∧ B ∈ (range 10) ∧ C ∈ (range 10) ∧ D ∈ (range 10) ∧
    E ∈ (range 10) ∧ F ∈ (range 10) ∧ H ∈ (range 10) ∧ J ∈ (range 10) ∧ K ∈ (range 10) ∧
    0 < A ∧ 0 < B ∧ 0 < D ∧
    1000 * C + 100 * H + 10 * J + K = A + 10 * B + C + 100 * D + 10 * E + F ∧
    count_valid_digit_assignments = 60 :=
by
  sorry

end digit_assignment_count_is_correct_l648_648798


namespace sum_primes_between_1_and_20_l648_648526

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648526


namespace num_solutions_system_eqns_l648_648828

theorem num_solutions_system_eqns :
  ∃ (c : ℕ), 
    (∀ (a1 a2 a3 a4 a5 a6 : ℕ), 
       a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 = 26 ∧ 
       a1 + a2 + a3 + a4 + a5 + a6 = 5 → 
       (a1, a2, a3, a4, a5, a6) ∈ (solutions : Finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
    solutions.card = 5 := sorry

end num_solutions_system_eqns_l648_648828


namespace number_of_pink_cubes_l648_648924

theorem number_of_pink_cubes :
  let large_cube_side := 125
  let small_cubes_count := 125
  let small_cube_side := Math.sqrt (Math.sqrt (large_cube_side : ℝ))
  (small_cube_side = 5) → (small_cubes_count = 125) → (∃ n : ℕ, n = 46) :=
by
  -- Proof steps would go here
  sorry

end number_of_pink_cubes_l648_648924


namespace smallest_distance_between_plants_l648_648193

theorem smallest_distance_between_plants (side_length : ℝ) (num_plants : ℕ) 
  (h1 : side_length = 2) (h2 : num_plants = 9) :
  ∃ (d : ℝ), d = (sqrt 2) / 2 :=
by
  sorry

end smallest_distance_between_plants_l648_648193


namespace factor_poly_l648_648644

noncomputable theory

def p (x : ℝ) : ℝ := 3*x^2 + 7*x - 2

theorem factor_poly (t : ℝ) : p t = 0 → t = 1 ∨ t = -8/3 :=
by
  sorry

end factor_poly_l648_648644


namespace largest_difference_from_set_l648_648476

def largest_difference (s : Set ℤ) : ℤ :=
  s.sup id - s.inf id

theorem largest_difference_from_set {s : Set ℤ} (h : s = { -20, -5, 1, 3, 7, 15 }) :
  largest_difference s = 35 :=
by
  subst h
  rw [largest_difference, Finite.sup_eq_sup, Finite.inf_eq_inf]
  dsimp
  -- additional computation details would be simplified in the actual proof
  sorry

end largest_difference_from_set_l648_648476


namespace f_of_f1_l648_648833

def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 2 * x - 3
  else 2^x - 1

theorem f_of_f1 : f (f 1) = -1 / 2 :=
by
  sorry

end f_of_f1_l648_648833


namespace interest_rate_increase_l648_648984

-- Define the conditions
def principal (P : ℕ) := P = 1000
def time (t : ℕ) := t = 5
def original_amount (A : ℕ) := A = 1500
def new_amount (A' : ℕ) := A' = 1750

-- Prove that the interest rate increase is 50%
theorem interest_rate_increase
  (P : ℕ) (t : ℕ) (A A' : ℕ)
  (hP : principal P)
  (ht : time t)
  (hA : original_amount A)
  (hA' : new_amount A') :
  (((((A' - P) / (P * t)) - ((A - P) / (P * t))) / ((A - P) / (P * t))) * 100) = 50 := by
  sorry

end interest_rate_increase_l648_648984


namespace complex_number_solution_l648_648298

theorem complex_number_solution (z : ℂ) (h : |z - 1| = |z - complex.I|) :
  (z - (z - 6) / (z - 1)).im = 0 ∧ (z - (z - 6) / (z - 1)).re > 0 → z = 2 + 2 * complex.I :=
sorry

end complex_number_solution_l648_648298


namespace repeating_decimals_fraction_l648_648214

theorem repeating_decimals_fraction :
  (0.81:ℚ) / (0.36:ℚ) = 9 / 4 :=
by
  have h₁ : (0.81:ℚ) = 81 / 99 := sorry
  have h₂ : (0.36:ℚ) = 36 / 99 := sorry
  sorry

end repeating_decimals_fraction_l648_648214


namespace sum_of_primes_1_to_20_l648_648561

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648561


namespace sqrt_eq_fourth_root_l648_648134

-- Define the given condition as a Lean theorem
theorem sqrt_eq_fourth_root (x : ℝ) : 
    sqrt(72 / 25) = real.root 4 (x / 25) → x = 207.36 :=
by
  intro h
  sorry

end sqrt_eq_fourth_root_l648_648134


namespace sum_of_primes_between_1_and_20_l648_648541

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648541


namespace xiao_ming_sales_difference_xiao_ming_earnings_l648_648079

def planned_sales_per_day : ℕ := 100
def daily_excess_or_shortfall : List ℤ := [+4, -5, -3, +9, -7, +12, +5]
def selling_price_per_kg : ℕ := 10
def shipping_cost_per_kg : ℕ := 4

theorem xiao_ming_sales_difference :
  let highest_sales := daily_excess_or_shortfall.maximum
  let lowest_sales := daily_excess_or_shortfall.minimum
  highest_sales = some 12 ∧ lowest_sales = some (-7) →
  highest_sales.get_or_else 0 - lowest_sales.get_or_else 0 = 19 :=
by
  intros
  sorry

theorem xiao_ming_earnings :
  let total_planned_sales := planned_sales_per_day * 7
  let total_adjustments := daily_excess_or_shortfall.foldl (.+.) 0
  let total_actual_sales := total_planned_sales + total_adjustments
  let net_earning_per_kg := selling_price_per_kg - shipping_cost_per_kg
  let total_revenue := total_actual_sales * net_earning_per_kg
  net_earning_per_kg = 6 → total_actual_sales = 715 →
  total_revenue = 4290 :=
by
  intros
  sorry

end xiao_ming_sales_difference_xiao_ming_earnings_l648_648079


namespace girl_wins_against_boy_l648_648457

theorem girl_wins_against_boy
  (n m : ℕ)
  (played_match : ∀ i j, i ≠ j → (g_i played g_j) ∨ (g_j played g_i))
  (no_draws : ∀ i j, i ≠ j → (won g_i against g_j) ∨ (won g_j against g_i))
  (everyone_lost : ∀ i, ∃ j, i ≠ j ∧ (won g_j against g_i))
  (boy_losses_different : ∀ (b1 b2 : ℕ), b1 ≠ b2 → (calculate_boy_losses b1) ≠ (calculate_boy_losses b2)) :
  ∃ girl boy, (won girl against boy) :=
by
  sorry

end girl_wins_against_boy_l648_648457


namespace chord_length_from_polar_eqns_l648_648155

theorem chord_length_from_polar_eqns 
  (rho : ℝ → ℝ)
  (theta : ℝ)
  (h_line : ∀ theta, rho(theta) * (Real.sin(theta) - Real.cos(theta)) = 2)
  (h_circle : ∀ theta, rho(theta) = 4 * Real.sin(theta)) :
  chord_length rho rho h_line h_circle = 4 := 
sorry

end chord_length_from_polar_eqns_l648_648155


namespace speed_of_first_bus_l648_648467

theorem speed_of_first_bus (v : ℕ) (h : (v + 60) * 4 = 460) : v = 55 :=
by
  sorry

end speed_of_first_bus_l648_648467


namespace trivia_team_l648_648603

theorem trivia_team (total_members showed_up_points total_points : ℕ) 
(h_total : total_members = 5) 
(h_points : showed_up_points = 6) 
(h_total_points : total_points = 18) :
  (total_members - (total_points / showed_up_points)) = 2 :=
by
  rw [h_total, h_points, h_total_points]
  norm_num
  sorry

end trivia_team_l648_648603


namespace integral_evaluation_l648_648323

variable {f : ℝ → ℝ}

theorem integral_evaluation (h₁ : ∫ x in 0..1, f x = 1) (h₂ : ∫ x in 0..2, f x = -1) :
  ∫ x in 1..2, f x = -2 :=
by
  sorry

end integral_evaluation_l648_648323


namespace rectangle_side_lengths_l648_648475

theorem rectangle_side_lengths (AB BC : ℝ) (h_ratio : AB / BC = 7 / 5) 
  (grey_square_area : 72) 
  (h_ABX : isosceles_right_triangle_with_right_angle_at_X AB BC grey_square_area)
  (h_CDY : isosceles_right_triangle_with_right_angle_at_Y AB BC grey_square_area):
  AB = 42 ∧ BC = 30 := 
sorry

end rectangle_side_lengths_l648_648475


namespace starting_number_l648_648215

theorem starting_number (x : ℝ) (h : (x + 26) / 2 = 19) : x = 12 :=
by
  sorry

end starting_number_l648_648215


namespace reduced_price_equals_50_l648_648976

noncomputable def reduced_price (P : ℝ) : ℝ := 0.75 * P

theorem reduced_price_equals_50 (P : ℝ) (X : ℝ) 
  (h1 : 1000 = X * P)
  (h2 : 1000 = (X + 5) * 0.75 * P) : reduced_price P = 50 :=
sorry

end reduced_price_equals_50_l648_648976


namespace sum_of_primes_between_1_and_20_l648_648538

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648538


namespace distance_between_ports_l648_648184

/-- A passenger ship travels at a speed of 26 km/h in still water between ports A and B.
    The river's current speed is 6 km/h.
    If the ship makes 4 round trips between ports A and B in a total of 13 hours 
    (excluding the time for the ship to turn around),
    then the distance between ports A and B is 40 km. -/
theorem distance_between_ports (speed_boat : ℕ) (speed_current : ℕ) (round_trips : ℕ) (total_time : ℝ) :
  speed_boat = 26 →
  speed_current = 6 →
  round_trips = 4 →
  total_time = 13 →
  let effective_speed_down = speed_boat + speed_current,
      effective_speed_up = speed_boat - speed_current,
      time_per_trip = total_time / round_trips,
      t_down = (5 / 13) * time_per_trip,
      t_up = time_per_trip - t_down
  in 
  effective_speed_down * t_down = 40 :=
by
  intros
  sorry

end distance_between_ports_l648_648184


namespace literature_more_than_science_science_less_than_literature_percent_l648_648446

theorem literature_more_than_science (l s : ℕ) (h : 8 * s = 5 * l) : (l - s) / s = 3 / 5 :=
by {
  -- definition and given condition will be provided
  sorry
}

theorem science_less_than_literature_percent (l s : ℕ) (h : 8 * s = 5 * l) : ((l - s : ℚ) / l) * 100 = 37.5 :=
by {
  -- definition and given condition will be provided
  sorry
}

end literature_more_than_science_science_less_than_literature_percent_l648_648446


namespace largest_whole_number_l648_648096

theorem largest_whole_number (x : ℕ) (h : 7 * x < 150) : x ≤ 21 :=
begin
  assume h : 7 * x < 150,
  show x ≤ 21,
  sorry
end

end largest_whole_number_l648_648096


namespace total_cost_is_eight_x_l648_648055

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l648_648055


namespace sum_primes_upto_20_l648_648517

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648517


namespace sequence_a_6_is_8_l648_648310

def sequence_a : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n+2) := sequence_a n + sequence_a (n + 1)

theorem sequence_a_6_is_8 : sequence_a 6 = 8 := by
  sorry

end sequence_a_6_is_8_l648_648310


namespace valid_sentence_count_is_208_l648_648081

def four_words := ["splargh", "glumph", "amr", "flark"]

def valid_sentence (sentence : List String) : Prop :=
  ¬(sentence.contains "glumph amr")

def count_valid_sentences : Nat :=
  let total_sentences := 4^4
  let invalid_sentences := 3 * 4 * 4
  total_sentences - invalid_sentences

theorem valid_sentence_count_is_208 :
  count_valid_sentences = 208 := by
  sorry

end valid_sentence_count_is_208_l648_648081


namespace highest_score_not_necessarily_at_least_12_l648_648591

section

-- Define the number of teams
def teams : ℕ := 12

-- Define the number of games each team plays
def games_per_team : ℕ := teams - 1

-- Define the total number of games
def total_games : ℕ := (teams * games_per_team) / 2

-- Define the points system
def points_for_win : ℕ := 2
def points_for_draw : ℕ := 1

-- Define the total points in the tournament
def total_points : ℕ := total_games * points_for_win

-- The highest score possible statement
def highest_score_must_be_at_least_12_statement : Prop :=
  ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)

-- Theorem stating that the statement "The highest score must be at least 12" is false
theorem highest_score_not_necessarily_at_least_12 (h : ∀ (scores : Fin teams → ℕ), (∃ i, scores i ≥ 12)) : False :=
  sorry

end

end highest_score_not_necessarily_at_least_12_l648_648591


namespace constant_term_expansion_l648_648823

noncomputable def integral_a : ℝ := ∫ x in 0..real.pi, real.sin x

theorem constant_term_expansion (a : ℝ) (h : a = integral_a) : 
  constant_term ((a * real.sqrt x - 1 / real.sqrt x) ^ 6) = -160 :=
by 
  -- Proof goes here...
  sorry

end constant_term_expansion_l648_648823


namespace ratio_p_q_l648_648856

theorem ratio_p_q 
  (total_amount : ℕ) 
  (amount_r : ℕ) 
  (ratio_q_r : ℕ × ℕ) 
  (total_amount_eq : total_amount = 1210) 
  (amount_r_eq : amount_r = 400) 
  (ratio_q_r_eq : ratio_q_r = (9, 10)) :
  ∃ (amount_p amount_q : ℕ), 
    total_amount = amount_p + amount_q + amount_r ∧ 
    (amount_q : ℕ) = 9 * (amount_r / 10) ∧ 
    (amount_p : ℕ) / (amount_q : ℕ) = 5 / 4 := 
by sorry

end ratio_p_q_l648_648856


namespace number_of_points_on_parabola_l648_648224

-- Defining the parabola with focus at (0,0)
def parabola_with_focus (x y : ℝ) : Prop :=
  -- Placeholder for the equation derived from the problem conditions
  sorry

-- Condition for points on the parabola satisfying the given inequality
def condition (x y : ℤ) : Prop :=
  (3 * x + 4 * y).abs ≤ 1250

-- The main theorem statement
theorem number_of_points_on_parabola :
  -- Counting the integer solutions (x, y) on the parabola Q under given condition
  let points := {p ∈ ℤ × ℤ | parabola_with_focus p.1 p.2 ∧ condition p.1 p.2 } in
  points.to_finset.card = 502 :=
sorry

end number_of_points_on_parabola_l648_648224


namespace janet_total_pockets_l648_648367

theorem janet_total_pockets
  (total_dresses : ℕ)
  (dresses_with_pockets : ℕ)
  (dresses_with_2_pockets : ℕ)
  (dresses_with_3_pockets : ℕ)
  (pockets_from_2 : ℕ)
  (pockets_from_3 : ℕ)
  (total_pockets : ℕ)
  (h1 : total_dresses = 24)
  (h2 : dresses_with_pockets = total_dresses / 2)
  (h3 : dresses_with_2_pockets = dresses_with_pockets / 3)
  (h4 : dresses_with_3_pockets = dresses_with_pockets - dresses_with_2_pockets)
  (h5 : pockets_from_2 = 2 * dresses_with_2_pockets)
  (h6 : pockets_from_3 = 3 * dresses_with_3_pockets)
  (h7 : total_pockets = pockets_from_2 + pockets_from_3)
  : total_pockets = 32 := 
by
  sorry

end janet_total_pockets_l648_648367


namespace smallest_four_digit_divisible_five_smallest_primes_l648_648663

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l648_648663


namespace sum_g_eq_2010_l648_648263

noncomputable def g (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12)

theorem sum_g_eq_2010 :
  ∑ i in Finset.range 2010, g ((i + 1) / 2011 : ℝ) = 2010 :=
sorry

end sum_g_eq_2010_l648_648263


namespace math_problem_l648_648863

-- Define the sum of even squared integers from 2 to 10
def x : ℕ := (∑ i in {2, 4, 6, 8, 10}.toFinset, i^2)
-- Define the number of prime numbers from 1 to 10
def y : ℕ := {2, 3, 5, 7}.card
-- Define the product of odd integers from 1 to 10
def z : ℕ := (∏ i in {1, 3, 5, 7, 9}.toFinset, i)

-- Prove the final result
theorem math_problem : x - y + z = 1161 := by
  sorry

end math_problem_l648_648863


namespace quadratic_complete_the_square_l648_648091

theorem quadratic_complete_the_square (y : ℝ) : ∃ k : ℝ, (k = -1) ∧ (y^2 + 14 * y + 48 = (y + 7)^2 + k) :=
by 
  let k := -1
  use k
  split
  sorry
  sorry

end quadratic_complete_the_square_l648_648091


namespace value_of_b_l648_648776

variable (a b c y1 y2 : ℝ)

def equation1 := (y1 = 4 * a + 2 * b + c)
def equation2 := (y2 = 4 * a - 2 * b + c)
def difference := (y1 - y2 = 8)

theorem value_of_b 
  (h1 : equation1 a b c y1)
  (h2 : equation2 a b c y2)
  (h3 : difference y1 y2) : 
  b = 2 := 
by 
  sorry

end value_of_b_l648_648776


namespace sum_of_primes_between_1_and_20_l648_648493

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648493


namespace Madeline_hours_left_over_l648_648005

theorem Madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  total_hours_per_week - total_busy_hours = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  have : total_hours_per_week - total_busy_hours = 168 - 122 := by rfl
  have : 168 - 122 = 46 := by rfl
  exact this

end Madeline_hours_left_over_l648_648005


namespace cross_section_area_l648_648086

noncomputable def area_of_cross_section : ℝ := 
  let side_length := sqrt 3 in
  let height := side_length in
  let angle := 60 in
  let distance_from_A_to_plane := 0.25 in
  -- Placeholder for the actual calculation
  (11 * sqrt 3) / 30

theorem cross_section_area (side_length height distance_from_A_to_plane : ℝ) 
  (angle : ℝ) (MB AM : ℝ) 
  (h1 : side_length = sqrt 3)
  (h2 : height = side_length)
  (h3 : angle = 60)
  (h4 : distance_from_A_to_plane = 0.25)
  (h5 : MB = 2 * AM)
  : area_of_cross_section side_length height distance_from_A_to_plane angle MB AM = (11 * sqrt 3) / 30 :=
sorry

end cross_section_area_l648_648086


namespace problem_solution_l648_648968

theorem problem_solution : 
  ∀ (x y S : ℝ), 
  x + y = 56 → 
  y = 37.66666666666667 → 
  S = 10 * x + 22 * y → 
  S = 1012 := 
by {
  intros x y S h1 h2 h3,
  sorry
}

end problem_solution_l648_648968


namespace december_sales_fraction_l648_648926

noncomputable theory
open_locale classical

-- Define the average sales for January through November
def avg_sales_Jan_to_Nov : ℝ := sorry

-- Define the total sales for January through November
def total_sales_Jan_to_Nov : ℝ := 11 * avg_sales_Jan_to_Nov

-- Define the sales for December
def sales_Dec : ℝ := 3 * avg_sales_Jan_to_Nov

-- Define the total sales for the year
def total_sales_year : ℝ := total_sales_Jan_to_Nov + sales_Dec

-- Prove that the fraction of the sales total for December out of the total sales for the year is 3/14
theorem december_sales_fraction : sales_Dec / total_sales_year = 3 / 14 :=
by sorry

end december_sales_fraction_l648_648926


namespace isosceles_triangle_perimeter_l648_648325

theorem isosceles_triangle_perimeter (x y : ℝ) (h : 4 * x ^ 2 + 17 * y ^ 2 - 16 * x * y - 4 * y + 4 = 0):
  x = 4 ∧ y = 2 → 2 * x + y = 10 :=
by
  intros
  sorry

end isosceles_triangle_perimeter_l648_648325


namespace next_term_geometric_sequence_l648_648488

theorem next_term_geometric_sequence (y : ℝ) (h0 : y ≠ 0) :
  let r := 3 * y in
  let term := 81 * y^3 in
  term * r = 243 * y^4 :=
by
  let r := 3 * y
  let term := 81 * y^3
  have h : term * r = 243 * y^4 := sorry
  exact h

end next_term_geometric_sequence_l648_648488


namespace jared_current_age_condition_l648_648903

variable (t j: ℕ)

-- Conditions
def tom_current_age := 25
def tom_future_age_condition := t + 5 = 30
def jared_past_age_condition := j - 2 = 2 * (t - 2)

-- Question
theorem jared_current_age_condition : 
  (t + 5 = 30) ∧ (j - 2 = 2 * (t - 2)) → j = 48 :=
by
  sorry

end jared_current_age_condition_l648_648903


namespace incorrect_operation_found_l648_648566

variable (a : ℝ)
open Lean

theorem incorrect_operation_found 
  (opA : (-a) ^ 4 = a ^ 4)
  (opB : -a + 3 * a = 2 * a)
  (opC : (2 * a ^ 2) ^ 3 = 6 * a ^ 5)
  (opD : a ^ 6 / a ^ 2 = a ^ 4) : 
  opC = false :=
sorry

end incorrect_operation_found_l648_648566


namespace smallest_integer_l648_648650

theorem smallest_integer (x : ℤ) (h : 3 * (Int.natAbs x)^3 + 5 < 56) : x = -2 :=
sorry

end smallest_integer_l648_648650


namespace number_of_true_statements_l648_648311

open Set

theorem number_of_true_statements (A : Set ℤ) (h : A = {x | x^2 - 1 = 0}) :
  let s : Set ℤ := {-1, 1}
  in (1 ∈ s) ∧ (¬ ({-1} ∈ s)) ∧ (∅ ⊆ s) ∧ ({1, -1} ⊆ s) -> 3 :=
by 
  intros s h1 h2 h3 h4
  have s_eq : s = {-1, 1} := by rfl
  split; split; exact (rfl : s = {-1, 1})
  exfalso; exact (Set.not_mem_singleton.2 h2 rfl)
  exact rfl; exact rfl;
  have h5 : Finset.card s = 3 := by rfl
  sorry

end number_of_true_statements_l648_648311


namespace sum_of_primes_between_1_and_20_l648_648539

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648539


namespace quadratic_roots_real_distinct_l648_648771

theorem quadratic_roots_real_distinct (k : ℝ) :
  (k > (1/2)) ∧ (k ≠ 1) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k-1) * x1^2 + 2 * x1 - 2 = 0) ∧ ((k-1) * x2^2 + 2 * x2 - 2 = 0)) :=
by
  sorry

end quadratic_roots_real_distinct_l648_648771


namespace solve_inequality_1_solve_inequality_2_l648_648740

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem solve_inequality_1 :
  {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 2} :=
by sorry

theorem solve_inequality_2 (t : ℝ) :
  (∀ x ∈ Icc (2 : ℝ) (4 : ℝ), f (x - t) ≤ x - 2) → 3 ≤ t ∧ t ≤ 3 + Real.sqrt 2 :=
by sorry

end solve_inequality_1_solve_inequality_2_l648_648740


namespace sum_of_primes_1_to_20_l648_648560

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648560


namespace statements_valid_l648_648226

variable (b x y : ℝ)

theorem statements_valid :
  (∀ b x y, b * (x + y) = b * x + b * y) ∧
  (∀ b x y, log b (x * y) = log b x + log b y) ∧
  (∀ b x y, b * (x - y) = b * x - b * y) :=
by
  sorry

end statements_valid_l648_648226


namespace find_min_dist_point_l648_648061

/-- Define points A and B, and line l. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨1, 2⟩
def B : Point := ⟨-2, 0⟩

def line_l (P : Point) : Prop := P.x - P.y + 3 = 0

/-- Define function for distance calculation. -/
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- Define function to check if a point P makes PA + PB the smallest. -/
def is_min_dist_point (P : Point) : Prop :=
  line_l P ∧ ∀ Q : Point, line_l Q → distance P A + distance P B ≤ distance Q A + distance Q B

/-- Main theorem stating the coordinates of the point P. -/
theorem find_min_dist_point : ∃ P : Point, is_min_dist_point P ∧ P = Point.mk (-5/3 : ℝ) (4/3 : ℝ) :=
  sorry

end find_min_dist_point_l648_648061


namespace dimes_turned_in_l648_648212

theorem dimes_turned_in (total_coins nickels quarters : ℕ) (h1 : total_coins = 11) (h2 : nickels = 2) (h3 : quarters = 7) : 
  ∃ dimes : ℕ, dimes + nickels + quarters = total_coins ∧ dimes = 2 :=
by
  sorry

end dimes_turned_in_l648_648212


namespace assignment_count_l648_648893

theorem assignment_count (students : Finset α) (A B : α) (subjects : Finset β)
  (non_biology : subjects ≠ ∅) (biology : β) :
  students.card = 6 →
  subjects.card = 4 →
  A ∈ students →
  B ∈ students →
  (A ≠ B) →
  biology ∈ subjects →
  (∀ s ∈ subjects, s ≠ biology → s ≠ A ∧ s ≠ B) →
  (students.erase A).erase B ≠ ∅ →
  (students.erase A).erase B.card = 4 →
  ∃ assignment : Finset (α × β), 
  assignment.card = 240 := 
sorry

end assignment_count_l648_648893


namespace find_A_l648_648099

theorem find_A (J : ℤ := 15)
  (JAVA_pts : ℤ := 50)
  (AJAX_pts : ℤ := 53)
  (AXLE_pts : ℤ := 40)
  (L : ℤ := 12)
  (JAVA_eq : ∀ A V : ℤ, 2 * A + V + J = JAVA_pts)
  (AJAX_eq : ∀ A X : ℤ, 2 * A + X + J = AJAX_pts)
  (AXLE_eq : ∀ A X E : ℤ, A + X + L + E = AXLE_pts) : A = 21 :=
sorry

end find_A_l648_648099


namespace second_player_wins_bishop_game_l648_648902

theorem second_player_wins_bishop_game :
  ∀ (board : list (list ℕ)), (∀ r c, board.nth r != none → board.nth c != none → (abs (r - c) ≠ abs (board.nth r - board.nth c))) → 
  ∃ f : (ℕ × ℕ) → (ℕ × ℕ), (∀ p, f p ≠ p) → 
  ∀ (turn : ℕ), (turn % 2 = 1) → false :=
by
  sorry

end second_player_wins_bishop_game_l648_648902


namespace tan_A_eq_two_length_BC_eq_five_l648_648357

-- Define the properties of triangle and angles
variables {A B C : ℝ} -- Angles in the triangle
variables {a b c : ℝ} -- Sides opposite to the angles A, B, and C respectively

-- Given conditions
variable (h1 : b * Math.cos A = 2)
variable (h2 : (1/2) * b * c * Math.sin A = 2)

-- Part 1: Prove tan A = 2
theorem tan_A_eq_two : Math.tan A = 2 :=
by
  -- To simplify the proof process
  sorry

-- Additional condition for Part 2
variable (h3 : Math.sin B = 2 * Math.cos A * Math.sin C)

-- Part 2: Find the length of BC (side c)
theorem length_BC_eq_five : c = 5 :=
by
  -- Interaction between different known values
  have h_cos : Math.cos A = 1 / Math.sqrt (1 + Math.tan A ^ 2),
  -- Continue to match the derived steps from mathematic equivalent proof problem
  sorry

end tan_A_eq_two_length_BC_eq_five_l648_648357


namespace smallest_number_of_hikers_l648_648958

theorem smallest_number_of_hikers (n : ℕ) :
  (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 4) ↔ n = 154 :=
by sorry

end smallest_number_of_hikers_l648_648958


namespace part_a_l648_648945

theorem part_a (n : ℕ) (h_condition : n < 135) : ∃ r, r = 239 % n ∧ r ≤ 119 := 
sorry

end part_a_l648_648945


namespace isosceles_triangle_perimeter_l648_648762

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 3 ∨ a = 7)) (h2 : (b = 3 ∨ b = 7)) (h3 : a ≠ b) : 
  ∃ (c : ℕ), (a = 7 ∧ b = 3 ∧ c = 17) ∨ (a = 3 ∧ b = 7 ∧ c = 17) := 
by
  sorry

end isosceles_triangle_perimeter_l648_648762


namespace sum_of_primes_1_to_20_l648_648529

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648529


namespace parallelogram_area_sum_lt_half_triangle_l648_648703

variables {A_0 B_0 C_0 : Point}
variables {A : ℕ → Point} {C : ℕ → Point}
variables {n : ℕ}

-- Given conditions
axiom condition_A (i : ℕ) (h : i < n) : segment A_0 B_0 (A i)
axiom condition_C (i : ℕ) (h : i < n) : segment B_0 C_0 (C i)
axiom condition_parallel_AC (i : ℕ) (h : i < n - 1) : parallel (A i) (C (i + 1))
axiom condition_parallel_CA (i : ℕ) (h : i < n - 1) : parallel (C i) (A (i + 1))

-- Proof problem
theorem parallelogram_area_sum_lt_half_triangle :
  sum (λ i, area_parallelogram (D i) (A (i + 1)) (D (i + 1)) (C (i + 1))) (range (n - 1)) <
  (1 / 2) * area_triangle A_0 B_0 C_0 :=
sorry

end parallelogram_area_sum_lt_half_triangle_l648_648703


namespace wood_length_equation_l648_648934

theorem wood_length_equation (x : ℝ) : 
  (∃ r : ℝ, r - x = 4.5 ∧ r/2 + 1 = x) → 1/2 * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l648_648934


namespace option_B_correct_l648_648268

variables {l m : Type} {α β γ : Type}

-- Conditions
variables [L_perpendicular_to_m : l ⊥ m] [L_perpendicular_to_alpha : l ⊥ α]
variables [M_perpendicular_to_beta : m ⊥ β] [Alpha_parallel_to_beta : α ∥ β]
variables [Beta_parallel_to_gamma : β ∥ γ] [L_perpendicular_to_alpha_2 : l ⊥ α]
variables [M_perpendicular_to_alpha : m ⊥ α]

-- Correct Answer
theorem option_B_correct :
  (l ⊥ m ∧ l ⊥ α ∧ m ⊥ β → α ⊥ β) ∧
  (α ∥ β ∧ β ∥ γ → α ∥ γ) ∧
  (l ⊥ α ∧ α ∥ β → l ⊥ β) ∧
  (l ⊥ α ∧ m ⊥ α → l ∥ m) :=
by { sorry }

end option_B_correct_l648_648268


namespace least_distinct_values_l648_648178

variable (L : List Nat) (h_len : L.length = 2023) (mode : Nat) 
variable (h_mode_unique : ∀ x ∈ L, L.count x ≤ 15 → x = mode)
variable (h_mode_count : L.count mode = 15)

theorem least_distinct_values : ∃ k, k = 145 ∧ (∀ d ∈ L, List.count d L ≤ 15) :=
by
  sorry

end least_distinct_values_l648_648178


namespace monkey_reaches_top_l648_648182

def monkey_climb_time (tree_height : ℕ) (climb_per_hour : ℕ) (slip_per_hour : ℕ) 
  (rest_hours : ℕ) (cycle_hours : ℕ) : ℕ :=
  if (tree_height % (climb_per_hour - slip_per_hour) > climb_per_hour) 
    then (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours
    else (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours - 1

theorem monkey_reaches_top :
  monkey_climb_time 253 7 4 1 4 = 109 := 
sorry

end monkey_reaches_top_l648_648182


namespace rational_terms_count_l648_648723

theorem rational_terms_count (x n : ℕ) (h : (4^n) / (2^n) = 64) : 
  ∃ n : ℕ, n = 6 ∧ ∀ r, r = 0 ∨ r = 3 ∨ r = 6 :=
begin
  sorry
end

end rational_terms_count_l648_648723


namespace sum_of_consecutive_even_numbers_l648_648113

theorem sum_of_consecutive_even_numbers (n : ℤ) 
  (h : n + 4 = 14) : n + (n + 2) + (n + 4) + (n + 6) = 52 :=
by
  sorry

end sum_of_consecutive_even_numbers_l648_648113


namespace floor_width_l648_648413

theorem floor_width
  (widthX lengthX : ℝ) (widthY lengthY : ℝ)
  (hX : widthX = 10) (lX : lengthX = 18) (lY : lengthY = 20)
  (h : lengthX * widthX = lengthY * widthY) :
  widthY = 9 := 
by
  -- proof goes here
  sorry

end floor_width_l648_648413


namespace smallest_four_digit_number_divisible_by_smallest_primes_l648_648658

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l648_648658


namespace sum_of_primes_1_to_20_l648_648551

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648551


namespace performance_schedule_arrangements_l648_648123

-- Definitions for the problem
def solo_count : ℕ := 5
def chorus_count : ℕ := 3
def is_not_first (position : ℕ) : Prop := position ≠ 0
def not_adjacent (positions : list ℕ) : Prop := list.pairwise (≠) (list.map nat.succ positions)

-- The function to compute A_n^r
def A (n r : ℕ) : ℕ := nat.factorial n / nat.factorial (n - r)

-- The theorem statement
theorem performance_schedule_arrangements :
  ∃ (solo_positions : list ℕ) (chorus_positions : list ℕ), length solo_positions = solo_count ∧ length chorus_positions = chorus_count ∧
  is_not_first (list.head chorus_positions) ∧ not_adjacent chorus_positions →
  A solo_count solo_count * A solo_count chorus_count = A 5 5 * A 5 3 :=
sorry

end performance_schedule_arrangements_l648_648123


namespace trapezoid_perimeter_is_correct_l648_648355

noncomputable def trapezoid_perimeter_proof : ℝ :=
  let EF := 60
  let θ := Real.pi / 4 -- 45 degrees in radians
  let h := 30 * Real.sqrt 2
  let GH := EF + 2 * h / Real.tan θ
  let EG := h / Real.tan θ
  EF + GH + 2 * EG -- Perimeter calculation

theorem trapezoid_perimeter_is_correct :
  trapezoid_perimeter_proof = 180 + 60 * Real.sqrt 2 := 
by
  sorry

end trapezoid_perimeter_is_correct_l648_648355


namespace sum_of_roots_of_quadratic_l648_648132

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h : a ≠ 0) (h_eq : ∀ x, a * x^2 + b * x + c = 0 → x^2 - (b / a) * x + (c / a) = 0) : 
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = -b / a :=
by {
  sorry
}

example : ∑ x in {x | x^2 - 7 * x + 12 = 0}, x = 7 :=
by {
  have h : 1 ≠ 0 := by norm_num,
  have h_eq : ∀ x, 1 * x^2 + (-7) * x + 12 = 0 → x^2 - (-7 / 1) * x + 12 / 1 = 0 := by {
    intro x,
    intro hx,
    simp at *,
    exact hx,
  },
  exact sum_of_roots_of_quadratic 1 (-7) 12 h h_eq,
  sorry
}

end sum_of_roots_of_quadratic_l648_648132


namespace exists_same_color_points_at_distance_one_l648_648189

theorem exists_same_color_points_at_distance_one (coloring : ℝ × ℝ → Fin 3) :
  ∃ (p q : ℝ × ℝ), (coloring p = coloring q) ∧ (dist p q = 1) := sorry

end exists_same_color_points_at_distance_one_l648_648189


namespace triangle_is_right_triangle_l648_648806

theorem triangle_is_right_triangle 
  (A B C : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = 180)
  (h3 : A / B = 2 / 3)
  (h4 : A / C = 2 / 5) : 
  A = 36 ∧ B = 54 ∧ C = 90 := 
sorry

end triangle_is_right_triangle_l648_648806


namespace part1_k_value_part2_effective_removal_time_l648_648593

-- Condition: f(x) function definition
def f (x : ℝ) : ℝ :=
  if x ≤ 5 then 16 / (9 - x) - 1 else 11 - (2/45) * x^2

-- Part (1) proof statement
theorem part1_k_value {k : ℝ} (h_k_range : 1 ≤ k ∧ k ≤ 4) (h_y : k * (16 / (9 - 3) - 1) = 4) : 
  k = 12 / 5 :=
by sorry

-- Part (2) proof statement
theorem part2_effective_removal_time {k : ℝ} (h_k_eq : k = 4) (h_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 16) :
  1 ≤ x ∧ x ≤ 15 → 14 := sorry

end part1_k_value_part2_effective_removal_time_l648_648593


namespace min_P2_sub_Q2_l648_648381

variable (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
def P := Real.sqrt (a + 3) + Real.sqrt (b + 6) + Real.sqrt (c + 12)
def Q := Real.sqrt (a + 2) + Real.sqrt (b + 2) + Real.sqrt (c + 2)

theorem min_P2_sub_Q2 : (P^2 - Q^2) = 36 :=
by
  sorry

end min_P2_sub_Q2_l648_648381


namespace math_proof_problem_l648_648648

open Nat

noncomputable def number_of_pairs := 
  let N := 20^19
  let num_divisors := (38 + 1) * (19 + 1)
  let total_pairs := num_divisors * num_divisors
  let ab_dividing_pairs := 780 * 210
  total_pairs - ab_dividing_pairs

theorem math_proof_problem : number_of_pairs = 444600 := 
  by exact sorry

end math_proof_problem_l648_648648


namespace volume_increase_l648_648975

theorem volume_increase (l w h: ℕ) 
(h1: l * w * h = 4320) 
(h2: l * w + w * h + h * l = 852) 
(h3: l + w + h = 52) : 
(l + 1) * (w + 1) * (h + 1) = 5225 := 
by 
  sorry

end volume_increase_l648_648975


namespace rational_tangents_l648_648071

theorem rational_tangents 
  (n : ℕ) (a : Fin n → ℚ) (h_odd : n % 2 = 1) 
  (h_convex : convex_polygon n a) 
  (h_circumscribed : circumscribed n a) : 
  ∀ i, is_rational (tangent_segment_length i n a) :=
by
  sorry

end rational_tangents_l648_648071


namespace arithmetic_sqrt_one_four_l648_648084

theorem arithmetic_sqrt_one_four : real.sqrt (1 / 4) = 1 / 2 := 
by {
  sorry
}

end arithmetic_sqrt_one_four_l648_648084


namespace Vasya_has_larger_amount_l648_648129

-- Defining the conditions and given data
variables (V P : ℝ)

-- Vasya's profit calculation
def Vasya_profit (V : ℝ) : ℝ := 0.20 * V

-- Petya's profit calculation considering exchange rate increase
def Petya_profit (P : ℝ) : ℝ := 0.2045 * P

-- Proof statement
theorem Vasya_has_larger_amount (h : Vasya_profit V = Petya_profit P) : V > P :=
sorry

end Vasya_has_larger_amount_l648_648129


namespace total_cost_shorts_tshirt_boots_shinguards_l648_648019

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l648_648019


namespace ellipse_equation_constant_m_squared_l648_648707

theorem ellipse_equation
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (eccentricity : ℝ) (hec : eccentricity = (Real.sqrt 3) / 2)
  (point : ℝ × ℝ) (hp : point = (Real.sqrt 2, Real.sqrt 2 / 2))
  (hx_ellipse : (point.1)^2 / a^2 + (point.2)^2 / b^2 = 1)
  (h_eccentricity : eccentricity = Real.sqrt (1 - (b/a)^2)) :
  a = 2 ∧ b = 1 := sorry

theorem constant_m_squared
  (k m : ℝ) (h_k : k ≠ 0)
  (h_slope : ∀ P Q : ℝ × ℝ, (m * P.fst) / P.snd + (m * Q.fst) / Q.snd = 4 * k)
  (ellipse_eq : ∀ x y : ℝ, y = k * x + m → (x^2)/4 + y^2 = 1) :
  m^2 = 1/2 := sorry

end ellipse_equation_constant_m_squared_l648_648707


namespace triangle_arithmetic_progression_l648_648411

-- Define the conditions and prove statements.
theorem triangle_arithmetic_progression
  (a b c : ℝ) (h : b = (a + c) / 2) :
  ∃ (r h_b : ℝ),
    (r = (1 / 3) * h_b) ∧
    (∃ G I, 
      (is_centroid_of G (a, b, c)) ∧ 
      (is_incenter_of I (a, b, c)) ∧ 
      (line_parallel_to (line G I) b)) ∧ 
    (∃ angle_bisector,
      (is_angle_bisector_of angle_bisector (a, b, c)) ∧
      (angle_bisector_perpendicular_to_line (angle_bisector) (line (incenter (a, b, c)) (circumcenter (a, b, c))))) ∧
    (∀ pt, 
      (on_angle_bisector pt (a, b, c) → 
      sum_distances_to_sides_constant pt (a, b, c))) ∧
    (∃ L,
      midpoint_of L (longest_side (a, b, c)) (shortest_side (a, b, c)) ∧
      (cyclic_quad (incenter (a, b, c)) L (midpoint (shortest_side (a, b, c))) (opp_vertex (longest_side (a, b, c) shortest_side (a, b, c))))) :=
sorry

end triangle_arithmetic_progression_l648_648411


namespace homothety_parabola_transformation_l648_648854

theorem homothety_parabola_transformation (p k X Y : ℝ) (h : k = 2 * p) :
  (2 * p) * (k * Y) = (k * X)^2 → Y = X^2 :=
by
  intro h_parabola
  have h1 : (2 * p) * (k * Y) = k^2 * X^2, by rwa h at h_parabola
  rw [mul_assoc, mul_comm k, mul_assoc] at h1
  rw [←mul_assoc, mul_assoc 2 p k, mul_comm p k, mul_assoc 2 k p] at h1
  have h2 : (2 * k * p) * Y / k = k^2 * X^2 / k, from congr_arg (/ k) h1
  simp only [mul_assoc, mul_div_right_comm] at h2
  have h3 : 2 * p * Y = k * X^2, from h2
  rwa h at h3
  simp only [mul_comm, mul_div_cancel_left]


end homothety_parabola_transformation_l648_648854


namespace coefficient_x3_expansion_l648_648725

theorem coefficient_x3_expansion (a : ℕ) (h : ∑ n in finset.range 6, if odd n then (a + (1 : ℕ)) * (1 + (1 : ℕ)^n) else 0 = 32) :
  (finset.range 6).filter (λ n, n = 3).sum (λ n, (a + (1 : ℕ)) * (1 + (1 : ℕ)^n)) = 18 :=
begin
  sorry
end

end coefficient_x3_expansion_l648_648725


namespace birds_never_gather_44_l648_648141

theorem birds_never_gather_44 :
    ∀ (position : Fin 44 → Nat), 
    (∀ (i : Fin 44), position i ≤ 44) →
    (∀ (i j : Fin 44), position i ≠ position j) →
    ∃ (S : Nat), S % 4 = 2 →
    ∀ (moves : (Fin 44 → Fin 44) → (Fin 44 → Fin 44)),
    ¬(∃ (tree : Nat), ∀ (i : Fin 44), position i = tree) := 
sorry

end birds_never_gather_44_l648_648141


namespace extreme_value_of_g_l648_648822

noncomputable def g (x : ℝ) : ℝ := (4/3) * x^3 + 2 * x^2 - 3 * x - 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else Real.log x / Real.log a

theorem extreme_value_of_g (a : ℝ) (h : ∃ x, g' x = 0 ∧ x = a) :
  f a (1 / 4) + f a (Real.log 2 (1 / 6)) = 8 :=
by
  sorry

end extreme_value_of_g_l648_648822


namespace sunzi_wood_problem_l648_648937

theorem sunzi_wood_problem (x : ℝ) :
  (∃ (length_of_rope : ℝ), length_of_rope = x + 4.5 ∧
    ∃ (half_length_of_rope : ℝ), half_length_of_rope = length_of_rope / 2 ∧ 
      (half_length_of_rope + 1 = x)) ↔ 
  (1 / 2 * (x + 4.5) = x - 1) :=
by
  sorry

end sunzi_wood_problem_l648_648937


namespace change_back_l648_648979

theorem change_back (price_laptop : ℤ) (price_smartphone : ℤ) (qty_laptops : ℤ) (qty_smartphones : ℤ) (initial_amount : ℤ) (total_cost : ℤ) (change : ℤ) :
  price_laptop = 600 →
  price_smartphone = 400 →
  qty_laptops = 2 →
  qty_smartphones = 4 →
  initial_amount = 3000 →
  total_cost = (price_laptop * qty_laptops) + (price_smartphone * qty_smartphones) →
  change = initial_amount - total_cost →
  change = 200 := by
  sorry

end change_back_l648_648979


namespace sequence_existence_l648_648649

theorem sequence_existence (n : ℕ) : 
  (∃ (x : ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ i + j ≤ n ∧ ((x i - x j) % 3 = 0) → (x (i + j) + x i + x j + 1) % 3 = 0)) ↔ (n = 8) := 
by 
  sorry

end sequence_existence_l648_648649


namespace odd_function_neg_value_l648_648755

theorem odd_function_neg_value
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  ∀ x, x < 0 → f x = -x^2 + 2 * x :=
by
  intros x hx
  -- The proof would go here
  sorry

end odd_function_neg_value_l648_648755


namespace total_cost_is_eight_x_l648_648057

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l648_648057


namespace sum_of_primes_between_1_and_20_l648_648497

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648497


namespace isosceles_and_angle_A_l648_648242

open EuclideanGeometry

-- Define the points and triangles
variables {A B C A' B' M : Point}
variables (hA'BC : Equilateral A' B C) 
          (hB'AC : Equilateral B' A C) 
          (hC'AB : Equilateral C' A B)
          (hM_center : CenterOfTriangle M C' A B)

-- The proof statement
theorem isosceles_and_angle_A'MB' (hA'BC : Equilateral A' B C) 
                                   (hB'AC : Equilateral B' A C) 
                                   (hC'AB : Equilateral C' A B)
                                   (hM_center : CenterOfTriangle M C' A B) : 
  Isosceles A' B' M ∧ ∠ A' M B' = 120 := 
sorry

end isosceles_and_angle_A_l648_648242


namespace problem_I_problem_II_problem_III_l648_648302

-- (I)
theorem problem_I (x : ℝ) (h : x > 0) : 
  (x / (1 + x)) < Real.log(1 + x) := 
sorry

-- (II)
theorem problem_II : 
  2015^2013 < 2014^2014 := 
sorry

-- (III)
theorem problem_III (n : ℕ) (h1 : n > 2015) (x : Fin n → ℝ) (h2 : ∑ i, x i = 1) :
  (∑ i, (x i)^2 / (1 + x i))^2015 > (1 / 2016)^n := 
sorry

end problem_I_problem_II_problem_III_l648_648302


namespace same_asymptotes_hyperbolas_l648_648746

theorem same_asymptotes_hyperbolas (M : ℝ) :
  (∀ x y : ℝ, ((x^2 / 9) - (y^2 / 16) = 1) ↔ ((y^2 / 32) - (x^2 / M) = 1)) →
  M = 18 :=
by
  sorry

end same_asymptotes_hyperbolas_l648_648746


namespace sum_primes_between_1_and_20_l648_648518

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648518


namespace fraction_sum_numerator_denominator_l648_648454

theorem fraction_sum_numerator_denominator (x : ℚ) (h : sqrt (x^2 + 5) - x = 1 / 3) :
  x.num + x.denom = 25 :=
sorry

end fraction_sum_numerator_denominator_l648_648454


namespace total_bricks_required_l648_648953

def courtyard_length : ℝ := 37  -- meters
def courtyard_width : ℝ := 21   -- meters
def garden_length : ℝ := 3      -- meters
def garden_width : ℝ := 2       -- meters
def brick_length : ℕ := 18      -- centimeters
def brick_width : ℕ := 12       -- centimeters

theorem total_bricks_required : 
  let courtyard_area := courtyard_length * courtyard_width in
  let garden_area := garden_length * garden_width in
  let area_to_pave := courtyard_area - garden_area in
  let area_to_pave_cm := area_to_pave * 10000 in
  let brick_area := brick_length * brick_width in
  let total_bricks := nat.ceil (area_to_pave_cm / brick_area) in
  total_bricks = 35695 :=
by
  sorry

end total_bricks_required_l648_648953


namespace prob_chair_theorem_l648_648191

def numAvailableChairs : ℕ := 10 - 1

def totalWaysToChooseTwoChairs : ℕ := Nat.choose numAvailableChairs 2

def adjacentPairs : ℕ :=
  let pairs := [(1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]
  pairs.length

def probNextToEachOther : ℚ := adjacentPairs / totalWaysToChooseTwoChairs

def probNotNextToEachOther : ℚ := 1 - probNextToEachOther

theorem prob_chair_theorem : probNotNextToEachOther = 5/6 :=
by
  sorry

end prob_chair_theorem_l648_648191


namespace charge_per_kilo_l648_648846

variable (x : ℝ)

theorem charge_per_kilo (h : 5 * x + 10 * x + 20 * x = 70) : x = 2 := by
  -- Proof goes here
  sorry

end charge_per_kilo_l648_648846


namespace sum_primes_between_1_and_20_l648_648505

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648505


namespace triangle_ratio_problem_l648_648898

theorem triangle_ratio_problem :
  let AC := 6
  let BC := 8
  let AD := 15
  let AB := Real.sqrt (AC^2 + BC^2)
  (AB - AD) < 0 → -- ensures DB = -(AD - AB) is considered on the extension of AB past B
  ∃ m n, (Real.relatively_prime m n) ∧ m + n = 14 :=
by
  let AC := 6
  let BC := 8
  let AD := 15
  let AB := Real.sqrt (AC^2 + BC^2)
  Intro h
  have h1 : AB = 10 := by { exact Real.sqrt_eq_rfl.mpr (by linarith) }
  have h2 : DB := AD - AB
  use 9, 5
  constructor
  . exact Nat.rel_prime_9_5
  . linarith[sorry]

-- Note: sorry is added because the proof validation part is omitted as per requirements.

end triangle_ratio_problem_l648_648898


namespace mrs_sheridan_initial_cats_l648_648844

theorem mrs_sheridan_initial_cats (bought_cats total_cats : ℝ) (h_bought : bought_cats = 43.0) (h_total : total_cats = 54) : total_cats - bought_cats = 11 :=
by
  rw [h_bought, h_total]
  norm_num

end mrs_sheridan_initial_cats_l648_648844


namespace parallel_vectors_orthogonal_vectors_projection_of_a_onto_b_l648_648750

variables (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (-1, 3)) 
          (h₂ : b = (m, m - 3))

-- 1. If \overrightarrow{a} ∥ \overrightarrow{b}, then m = 3/4
theorem parallel_vectors (h₃ : (a.1 / b.1 = a.2 / b.2)) : m = 3 / 4 :=
by sorry

-- 2. If (\overrightarrow{a} + \overrightarrow{b}) ⊥ (\overrightarrow{a} - \overrightarrow{b}), then |\overrightarrow{b}| = sqrt(10)
theorem orthogonal_vectors (h₄ : (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0) : 
  |(b.1, b.2)| = real.sqrt 10 :=
by sorry

-- 3. If m = 3, the projection of vector \overrightarrow{a} onto the direction of \overrightarrow{b} is -1 
theorem projection_of_a_onto_b (h₅ : m = 3) : 
  (a.1 * b.1 + a.2 * b.2) / real.sqrt (b.1 ^ 2 + b.2 ^ 2) = -1 :=
by sorry

end parallel_vectors_orthogonal_vectors_projection_of_a_onto_b_l648_648750


namespace convex_polygons_from_12_points_on_circle_l648_648900

theorem convex_polygons_from_12_points_on_circle :
  let n := 12 in
  let total_subsets := 2^n in
  let subsets_with_less_than_4_points := Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3 in
  let valid_subsets := total_subsets - subsets_with_less_than_4_points in
  valid_subsets = 3797 := by
{
  let n := 12
  let total_subsets := 2^n
  let subsets_with_less_than_4_points := Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3
  let valid_subsets := total_subsets - subsets_with_less_than_4_points
  have : subsets_with_less_than_4_points = 1 + 12 + 66 + 220
  norm_num1
  have : valid_subsets = 4096 - 299 := by norm_num1
  exact valid_subsets
  sorry
}

end convex_polygons_from_12_points_on_circle_l648_648900


namespace weekly_rent_is_correct_l648_648839

/-
   Conditions:
   1. Lewis earns $491 every week.
   2. Harvest season lasts 1181 weeks.
   3. At the end of the harvest season, Lewis will have $324,775.

   Question:
   How much rent does Lewis pay every week?
-/

noncomputable def weeklyEarnings : ℕ := 491
noncomputable def weeks : ℕ := 1181
noncomputable def finalAmount : ℕ := 324775

def totalEarnings : ℕ := weeklyEarnings * weeks
def totalRentSpent : ℕ := totalEarnings - finalAmount
def weeklyRent : ℕ := totalRentSpent / weeks

theorem weekly_rent_is_correct : weeklyRent = 216.08 := by
  sorry

end weekly_rent_is_correct_l648_648839


namespace age_difference_l648_648179

theorem age_difference (S M : ℕ) 
  (h1 : S = 35)
  (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 37 :=
by
  sorry

end age_difference_l648_648179


namespace no_rel_prime_a_b_c_div_conditions_l648_648237

open Nat

theorem no_rel_prime_a_b_c_div_conditions :
  ∀ a b c : ℕ, (Nat.coprime a b) ∧ (Nat.coprime b c) ∧ (Nat.coprime c a) →
  (a + b) ∣ (c^2) → (b + c) ∣ (a^2) → (c + a) ∣ (b^2) → False :=
by
  intro a b c h_coprime h1 h2 h3
  sorry

end no_rel_prime_a_b_c_div_conditions_l648_648237


namespace greatest_number_of_large_chips_l648_648894

theorem greatest_number_of_large_chips (s l p : ℕ) (h1 : s + l = 60) (h2 : s = l + p) 
  (hp_prime : Nat.Prime p) (hp_div : p ∣ l) : l ≤ 29 :=
by
  sorry

end greatest_number_of_large_chips_l648_648894


namespace hannah_age_double_july_age_20_years_ago_l648_648563

/-- Define the current ages of July (J) and her husband (H) -/
def current_age_july : ℕ := 23
def current_age_husband : ℕ := 25

/-- Assertion that July's husband is 2 years older than her -/
axiom husband_older : current_age_husband = current_age_july + 2

/-- We denote the ages 20 years ago -/
def age_july_20_years_ago := current_age_july - 20
def age_hannah_20_years_ago := current_age_husband - 20 - 2 * (current_age_july - 20)

theorem hannah_age_double_july_age_20_years_ago :
  age_hannah_20_years_ago = 6 :=
by sorry

end hannah_age_double_july_age_20_years_ago_l648_648563


namespace exists_same_color_points_distance_one_l648_648186

theorem exists_same_color_points_distance_one
    (color : ℝ × ℝ → Fin 3)
    (h : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_distance_one_l648_648186


namespace cost_first_10_kgs_l648_648612

variable (l q : ℝ)
variable (h1 : 30 * l + 3 * q = 11.67)
variable (h2 : 30 * l + 6 * q = 12.48)

theorem cost_first_10_kgs : 10 * l = 3.62 := by
  have hq : q = 0.27 := by
    linarith [h1, h2]
  have hl : l = 0.362 := by
    linarith [h1, hq]
  rw [hl]
  ring

end cost_first_10_kgs_l648_648612


namespace AndrewAge_l648_648205

variable (a f g : ℚ)
axiom h1 : f = 8 * a
axiom h2 : g = 3 * f
axiom h3 : g - a = 72

theorem AndrewAge : a = 72 / 23 :=
by
  sorry

end AndrewAge_l648_648205


namespace acute_triangle_l648_648106

theorem acute_triangle (r R : ℝ) (h : R < r * (Real.sqrt 2 + 1)) : 
  ∃ (α β γ : ℝ), α + β + γ = π ∧ (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ (α < π / 2) ∧ (β < π / 2) ∧ (γ < π / 2) := 
sorry

end acute_triangle_l648_648106


namespace H3C_bisects_converse_H3C_bisects_l648_648705

noncomputable def point := ℝ × ℝ

variables (A B C H3 P A1 B1 : point)

-- Conditions
variables
  (h_altitude : ∃ (H3 : point), ∀ X : point, X ∈ line(A,B) → dist(X, C) = dist(X, H3))
  (h_between : between H3 A B)
  (h_intersections : ∃ (P : point), ∃ (A1 B1 : point),
    lies_on P (altitude C H3) ∧ intersect (line A P) (line B P) BC AC A1 B1)

-- Theorem to prove
theorem H3C_bisects (h_altitude h_between h_intersections)
  : angle_bisector (H3, C) (A1, H3, B1) :=
sorry

-- Converse theorem
theorem converse_H3C_bisects (h_altitude h_between h_intersections)
  : converse_angle_bisector (H3, C) (A1, H3, B1) :=
sorry

end H3C_bisects_converse_H3C_bisects_l648_648705


namespace constant_product_of_areas_l648_648626

open Locale.Classical
open EuclideanGeometry Affine FGeometry

variable (Ω : Circle) (A P : Point)
variable {B C Q R S : Point}

-- Definitions and assumptions
def tangent_points (A : Point) (Ω : Circle) : Point × Point := sorry -- Points B, C
def is_tangent (p v : Point) (Ω : Circle) : Prop := sorry -- Tangency condition
def parallel (p1 p2 p3 p4 : Point) : Prop := sorry -- Parallelism condition

noncomputable def area_triangle (a b c : Point) : Real := sorry -- Area of the triangle

axiom tangency_points (B C : Point) (h1 : is_tangent A B Ω) (h2 : is_tangent A C Ω) : Prop
axiom existence_PQ (Q : Point) (h : is_tangent P Q Ω) : Prop
axiom existence_RS (R S : Point) (h1 : P ≠ R) (h2 : parallel P Q R S) (h3 : is_tangent R S Ω) : Prop

-- Main theorem
theorem constant_product_of_areas
  (AO_ne_OB : A ≠ B)
  (h_tangency_AB : tangency_points B C)
  (h_point_P : between A B P)
  (h_tangency_PQ : existence_PQ Q)
  (h_RS : existence_RS R S)
  : (area_triangle A P Q) * (area_triangle A R S) = (area_triangle A B C) ^ 2 := sorry

end constant_product_of_areas_l648_648626


namespace orthocenter_circumcenter_distance_l648_648852

theorem orthocenter_circumcenter_distance 
  (H O : Type*)
  (R : ℝ)
  (A B C : ℝ)
  (H_orthocenter : is_orthocenter H)
  (O_circumcenter : is_circumcenter O)
  (R_circumradius : is_circumradius R)
  (internal_angles : is_internal_angles A B C) :
  OH^2 = R^2 * (1 - 8 * cos A * cos B * cos C) :=
sorry

end orthocenter_circumcenter_distance_l648_648852


namespace share_of_B_l648_648992

noncomputable def B_share (B_investment A_investment C_investment D_investment total_profit : ℝ) : ℝ :=
  (B_investment / (A_investment + B_investment + C_investment + D_investment)) * total_profit

theorem share_of_B (B_investment total_profit : ℝ) (hA : A_investment = 3 * B_investment) 
  (hC : C_investment = (3 / 2) * B_investment) 
  (hD : D_investment = (3 / 2) * B_investment) 
  (h_profit : total_profit = 19900) :
  B_share B_investment A_investment C_investment D_investment total_profit = 2842.86 :=
by
  rw [B_share, hA, hC, hD, h_profit]
  sorry

end share_of_B_l648_648992


namespace arithmetic_sequence_sum_l648_648883

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end arithmetic_sequence_sum_l648_648883


namespace percentage_not_voted_correct_l648_648796

def percentage_of_yes_votes := (3 / 5 : ℝ) * 100
def percentage_of_no_votes := 28
def total_participation := percentage_of_yes_votes + percentage_of_no_votes
def percentage_did_not_vote := 100 - total_participation

theorem percentage_not_voted_correct :
  percentage_did_not_vote = 12 := by
  sorry

end percentage_not_voted_correct_l648_648796


namespace cars_meet_distance_from_midpoint_l648_648691

theorem cars_meet_distance_from_midpoint :
  ∀ (A B C : ℝ) (d_AB : ℝ) (s1 s2 : ℝ)
  (dA_C dB_C time when_meet : ℝ),
  d_AB = 240 →
  s1 = 60 → s2 = 80 →
  dA_C = d_AB / 2 →
  dB_C = d_AB / 2 →
  time = d_AB / (s1 + s2) →
  when_meet = (s1 * time) →
abs((when_meet - dA_C / 1) : ℝ) *
  = 17.14 := by
 sorry

end cars_meet_distance_from_midpoint_l648_648691


namespace smallest_divisible_four_digit_number_l648_648666

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l648_648666


namespace correct_answer_l648_648289

noncomputable def phi (x a b : ℝ) : ℝ := (1 / (Real.sqrt (2 * Real.pi) * a)) * Real.exp (-(x - b)^2 / (2 * a^2))

variable (a b : ℝ)
variable (X : Type) [MeasureSpace X] [Probability X]

axiom a_pos : a > 0
axiom b_pos : b > 0

def f (k : ℝ) : ℝ := Prob X (λ x, x < k)
def g (k : ℝ) : ℝ := Prob X (λ x, x > k + a)

theorem correct_answer :
  (∀ x, phi x a b = (1 / (Real.sqrt (2 * Real.pi) * a)) * Real.exp (-(x - b)^2 / (2 * a^2))) →
  (∃ μ σ², μ = 2 * a ∧ σ² = a^2 ∧ X ~ Normal 2*a a^2) ∧
  (f a = g (2 * a)) ∧ 
  (f (2 * a) + g (2 * a) = f a + g a) :=
by
  sorry

end correct_answer_l648_648289


namespace coeff_friction_of_rod_l648_648579

theorem coeff_friction_of_rod (W : ℝ) (alpha : ℝ) (N : ℝ) (deg_alpha : alpha = 70)
     (reaction_force_eq : N = 21 * W) : ∃ μ : ℝ, ( μ ≈ 0.05 ) := by
  sorry

end coeff_friction_of_rod_l648_648579


namespace sum_of_first_45_terms_l648_648306

theorem sum_of_first_45_terms :
  (∑ i in finset.range 45, (λ n, if n % 2 = 0 then (2 : ℕ)^(n/2) else 3*(n/2+1)-1) i) = 2^45 - 3017 := sorry

end sum_of_first_45_terms_l648_648306


namespace max_revenue_90_l648_648956

noncomputable def maximum_revenue : ℝ :=
  let x₀ := 0 in
  let y₀ := 0 in
  let x₁ := 0 in
  let y₁ := 200 in
  let x₂ := 100 in
  let y₂ := 300 in
  let revenue (x y : ℝ) := 0.3 * x + 0.2 * y in
  max (max (revenue x₀ y₀) (revenue x₁ y₁)) (revenue x₂ y₂)

theorem max_revenue_90 :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x + 2 * y ≤ 400 → 2 * x + y ≤ 500 → revenue x y ≤ 90) →
  maximum_revenue = 90 := by
  sorry

end max_revenue_90_l648_648956


namespace derivative_log2_plus_cos_const_l648_648867

def y (x : ℝ) : ℝ := log x (2 : ℝ) + cos (π / 4)

theorem derivative_log2_plus_cos_const (x : ℝ) (h : x ≠ 0) : 
  deriv (λ x : ℝ, log x (2 : ℝ) + cos (π / 4)) x = 1 / (x * log 2) :=
by
  sorry

end derivative_log2_plus_cos_const_l648_648867


namespace typing_time_l648_648855

theorem typing_time (t_randy t_candy t_sandy : ℕ) (h1 : t_randy = 30) (h2 : t_candy = 45) (h3 : t_sandy = 60) : 
  let total_time := 1 / (1/t_randy + 1/t_candy + 1/t_sandy) in
  abs (total_time - 180/13) < 0.01 :=
by
  sorry

end typing_time_l648_648855


namespace car_rent_per_day_leq_30_l648_648581

variable (D : ℝ) -- daily rental rate
variable (cost_per_mile : ℝ := 0.23) -- cost per mile
variable (daily_budget : ℝ := 76) -- daily budget
variable (distance : ℝ := 200) -- distance driven

theorem car_rent_per_day_leq_30 :
  D + cost_per_mile * distance ≤ daily_budget → D ≤ 30 :=
sorry

end car_rent_per_day_leq_30_l648_648581


namespace interval_of_monotonic_increase_l648_648136

noncomputable def log_a (a x : ℝ) : ℝ := real.log x / real.log a

theorem interval_of_monotonic_increase (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∃ S : set ℝ, (∀ x ∈ S, -∞ < x ∧ x < 1) ∧ (∀ x y ∈ S, x < y → log_a a (x^2 - 4 * x + 3) < log_a a (y^2 - 4 * y + 3)) :=
sorry

end interval_of_monotonic_increase_l648_648136


namespace sum_primes_between_1_and_20_l648_648500

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648500


namespace verify_parabola_D_l648_648296

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D_l648_648296


namespace minimum_value_l648_648108

theorem minimum_value (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 2) : ∃ (a : ℝ), x^2 + 2*y^2 = 4 - 2*real.sqrt 2 := by
  sorry

end minimum_value_l648_648108


namespace sum_product_distinct_zero_l648_648388

open BigOperators

theorem sum_product_distinct_zero {n : ℕ} (h : n ≥ 3) (a : Fin n → ℝ) (ha : Function.Injective a) :
  (∑ i, (a i) * ∏ j in Finset.univ \ {i}, (1 / (a i - a j))) = 0 := 
by
  sorry

end sum_product_distinct_zero_l648_648388


namespace standard_deviation_of_applicants_ages_l648_648866

theorem standard_deviation_of_applicants_ages
  (average_age : ℝ)
  (s : ℝ)
  (hiring_standard : ∀ x : ℝ, x ∈ Icc (average_age - s) (average_age + s) → True)
  (max_different_ages : ℕ)
  (h_max_different_ages : max_different_ages = 11)
  (h_average_age : average_age = 31) :
  s = 5 :=
by
  -- Proof goes here.
  sorry

end standard_deviation_of_applicants_ages_l648_648866


namespace find_distinct_integers_l648_648260

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := filter is_prime (List.range (n + 1)) in
    List.maximum (filter (λ p, p ∣ n) factors)
  else n

theorem find_distinct_integers (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : z = 2 * y - x) (h4 : largest_prime_factor (x * y * z) ≤ 3) :
  ∃ h a b c,
  (x = h * 1 ∧ y = h * 2 ∧ z = h * 3 ∧ h = 2 ^ a * 3 ^ b ∧ is_prime (a + b + c)) ∨
  (x = h * 2 ∧ y = h * 3 ∧ z = h * 4 ∧ h = 2 ^ a * 3 ^ b ∧ is_prime (a + b + c)) ∨
  (x = h * 2 ∧ y = h * 9 ∧ z = h * 16 ∧ h = 2 ^ a * 3 ^ b ∧ is_prime (a + b + c)) :=
sorry

end find_distinct_integers_l648_648260


namespace polynomial_coefficients_identity_l648_648327

theorem polynomial_coefficients_identity :
  let a : ℕ → ℝ := λ n, (1 - 2 * (x : ℝ))^2018.nat_coeff x;
  let sum := (λ (n : ℕ), a n / 2 ^ n);
  ∑ i in (range 2018), sum i = -1 :=
by
  sorry

end polynomial_coefficients_identity_l648_648327


namespace sum_of_remainders_mod_53_l648_648913

theorem sum_of_remainders_mod_53 (x y z : ℕ) (h1 : x % 53 = 31) (h2 : y % 53 = 17) (h3 : z % 53 = 8) : 
  (x + y + z) % 53 = 3 :=
by {
  sorry
}

end sum_of_remainders_mod_53_l648_648913


namespace seq_sum_S2016_l648_648277

theorem seq_sum_S2016 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 3)
  (h_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) - a n)
  (h_S : ∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1)) :
  S 2016 = 0 :=
sorry

end seq_sum_S2016_l648_648277


namespace sum_of_angles_in_regular_ngon_l648_648373

-- Definition of points and geometric properties
variables {n : ℕ} (A : fin n → ℝ × ℝ) -- Regular n-gon vertices
variable (B : fin (n-1) → ℝ × ℝ) -- Midpoints and bisectors intersections

-- Midpoints and points on angle bisectors
def is_midpoint (P Q R : ℝ × ℝ) := 2 * Q.1 = P.1 + R.1 ∧ 2 * Q.2 = P.2 + R.2
def is_intersection_of_angle_bisector (P Q R S T : ℝ × ℝ) := sorry

-- Define endpoints
variables (A1 A2 An Anm : ℝ × ℝ) -- Specific vertices and points

-- Sum of angles property
def sum_of_angles (A1 : ℝ × ℝ) (An : ℝ × ℝ) (B : fin (n-1) → ℝ × ℝ) : Prop :=
  sorry

-- Main theorem statement
theorem sum_of_angles_in_regular_ngon (A : fin n → ℝ × ℝ) (B : fin (n-1) → ℝ × ℝ)
  (h_midpoints : is_midpoint (A 0) (B 0) (A 1) ∧ is_midpoint (A (n-1)) (B (n-2)) (A n))
  (h_bisectors : ∀ i : fin (n-3), is_intersection_of_angle_bisector (A i) (A (i+1)) (B (i+1)) sorry sorry) :
  sum_of_angles (A 0) (A n) B := 
sorry

end sum_of_angles_in_regular_ngon_l648_648373


namespace rectangle_ratio_l648_648818

theorem rectangle_ratio (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a)
  (ha₁ : 1 / 2 * a * (6 / a) = 3) (hb₁ : 1 / 2 * b * (8 / b) = 4)
  (ha₂ : 1 / 2 * a * (10 / a) = 5) (hb₂ : 1 / 2 * b * (12 / b) = 6)
  (area : a * b = 18) : a / b = 2 :=
begin
  sorry
end

end rectangle_ratio_l648_648818


namespace boys_contributions_l648_648122

theorem boys_contributions (x y z : ℝ) (h1 : z = x + 6.4) (h2 : (1 / 2) * x = (1 / 3) * y) (h3 : (1 / 2) * x = (1 / 4) * z) :
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  -- This is where the proof would go
  sorry

end boys_contributions_l648_648122


namespace number_of_bikes_l648_648812

theorem number_of_bikes (total_wheels : ℕ) (car_wheels : ℕ) (tricycle_wheels : ℕ) (roller_skate_wheels : ℕ) (trash_can_wheels : ℕ) (bike_wheels : ℕ) (num_bikes : ℕ) :
  total_wheels = 25 →
  car_wheels = 2 * 4 →
  tricycle_wheels = 3 →
  roller_skate_wheels = 4 →
  trash_can_wheels = 2 →
  bike_wheels = 2 →
  (total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels + trash_can_wheels)) = bike_wheels * num_bikes →
  num_bikes = 4 := 
by
  intros total_wheels_eq total_car_wheels_eq tricycle_wheels_eq roller_skate_wheels_eq trash_can_wheels_eq bike_wheels_eq remaining_wheels_eq
  sorry

end number_of_bikes_l648_648812


namespace max_ratio_midpoint_projection_l648_648880

-- Definition of the parabola, focus, directrix, and points A and B.
variables {p : ℝ}
variables (hp : p > 0)

def parabola_equation (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus : (ℝ × ℝ) := (p / 2, 0)
def directrix (x : ℝ) : Prop := x = -p / 2
variables (A B : ℝ × ℝ)
def on_parabola (P : ℝ × ℝ) : Prop := parabola_equation P.1 P.2

-- Angle AFB = π/3 condition
variable (angle_AFB : ℝ)
def angle_condition : Prop := angle_AFB = π / 3

-- Midpoint definition
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Projection definition
def projection (M : ℝ × ℝ) : ℝ × ℝ := (-(p/2), M.2)

-- Distance 
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main statement
theorem max_ratio_midpoint_projection (A B : ℝ × ℝ) (hA : on_parabola A) (hB : on_parabola B) (h_angle : angle_condition angle_AFB) :
  ∃ M N, M = midpoint A B ∧ N = projection M ∧ ∀ AB, AB = dist A B → max ((dist M N) / AB) 1 :=
sorry

end max_ratio_midpoint_projection_l648_648880


namespace tangents_and_line_through_points_of_tangency_l648_648688

noncomputable def P : Point := (-2, 3)

def C : Circle := {
  center := (1, 1),
  radius := 1
}

theorem tangents_and_line_through_points_of_tangency (P : Point) (C : Circle) :
 (P = (-2, 3) ∧ C = { center := (1, 1), radius := 1 }) →
   ( ∃ k : ℝ, 
      (kx - y + 2k + 3 = 0 ∧ 
      ( (kx - y + 2k + 3 = (√3 + 6)x - 4y + 2√3 - 3) ∨ 
        (kx - y + 2k + 3 = (3 + √3)x + 4y - 6 + 2√3)
      )) ∧ 
      (3x - 2y - 3 = 0)) := 
by 
  sorry

end tangents_and_line_through_points_of_tangency_l648_648688


namespace total_crayons_l648_648763

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
      (h1 : crayons_per_child = 18) (h2 : num_children = 36) : 
        crayons_per_child * num_children = 648 := by
  sorry

end total_crayons_l648_648763


namespace complex_conjugate_equiv_l648_648699

theorem complex_conjugate_equiv (z : ℂ) (h : z / (z - complex.I) = complex.I) :
    complex.conj z = (1 - complex.I) / 2 := by
  sorry

end complex_conjugate_equiv_l648_648699


namespace find_a3_a4_a5_l648_648884

variable (a : ℕ → ℝ)

-- Recurrence relation for the sequence (condition for n ≥ 2)
axiom rec_relation (n : ℕ) (h : n ≥ 2) : 2 * a n = a (n - 1) + a (n + 1)

-- Additional conditions
axiom cond1 : a 1 + a 3 + a 5 = 9
axiom cond2 : a 3 + a 5 + a 7 = 15

-- Statement to prove
theorem find_a3_a4_a5 : a 3 + a 4 + a 5 = 12 :=
  sorry

end find_a3_a4_a5_l648_648884


namespace probability_of_one_pair_one_triplet_l648_648239

-- Define the necessary conditions
def six_sided_die_rolls (n : ℕ) : ℕ := 6 ^ n

def successful_outcomes : ℕ :=
  6 * 20 * 5 * 3 * 4

def total_outcomes : ℕ :=
  six_sided_die_rolls 6

def probability_success : ℚ :=
  successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_one_pair_one_triplet :
  probability_success = 25/162 :=
sorry

end probability_of_one_pair_one_triplet_l648_648239


namespace sum_of_primes_between_1_and_20_l648_648536

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648536


namespace diff_quotient_remainder_n_75_l648_648970

theorem diff_quotient_remainder_n_75 :
  ∃ n q r p : ℕ,  n = 75 ∧ n = 5 * q ∧ n = 34 * p + r ∧ q > r ∧ (q - r = 8) :=
by
  sorry

end diff_quotient_remainder_n_75_l648_648970


namespace find_triples_l648_648247

theorem find_triples (a m n : ℕ) (h1 : a ≥ 2) (h2 : m ≥ 2) :
  a^n + 203 ∣ a^(m * n) + 1 → ∃ (k : ℕ), (k ≥ 1) := 
sorry

end find_triples_l648_648247


namespace next_term_geometric_sequence_l648_648479

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l648_648479


namespace not_cone_if_view_is_square_l648_648804

-- Define geometric figures
inductive GeometricFigure
| Cylinder
| Cone
| Pyramid
| Prism

-- Define a property that one of the three views is a square
def view_is_square (fig : GeometricFigure) : Prop :=
match fig with
| GeometricFigure.Cone => False
| _ => True
end

-- Statement to prove that a solid geometric figure whose view is a square cannot be a cone
theorem not_cone_if_view_is_square (fig : GeometricFigure) (h : view_is_square fig) : fig ≠ GeometricFigure.Cone :=
by
  sorry

end not_cone_if_view_is_square_l648_648804


namespace combine_terms_l648_648915

theorem combine_terms (a b : ℕ) : 
  let lhs := (2 * a * b ^ 3)
  let rhs := (- a * b ^ 3)
  lhs + rhs = (2 - 1) * a * b ^ 3 :=
by sorry

end combine_terms_l648_648915


namespace intersection_A_B_l648_648743

def A : set ℕ := {0, 1, 2}
def B : set ℕ := {x | 1 < x ∧ x < 4}

theorem intersection_A_B : A ∩ B = {2} :=
by {
    sorry
}

end intersection_A_B_l648_648743


namespace variance_y_eq_16_l648_648292

variables {n : ℕ} (x y : ℕ → ℝ)

-- Condition: variance of x_1, x_2, ..., x_2017 is 4
def variance_x := (1 / (2017 : ℝ)) * (∑ i in finset.range 2017, (x i - (∑ i in finset.range 2017, x i) / 2017) ^ 2) = 4

-- Condition: y_i = 2x_i - 1
def y_def (i : ℕ) := y i = 2 * x i - 1

-- The proof goal: variance of y_1, y_2, ..., y_2017 is 16
theorem variance_y_eq_16 
  (h1 : variance_x x) 
  (h2 : ∀ i, i < 2017 → y_def x y i) :
  (1 / (2017 : ℝ)) * (∑ i in finset.range 2017, (y i - (∑ i in finset.range 2017, y i) / 2017) ^ 2) = 16 := 
sorry

end variance_y_eq_16_l648_648292


namespace union_intersection_l648_648376

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C : Set ℝ := {x | x^2 - 3 * x + 2 > 0}
def D : Set ℕ := {0, 3, 4}

theorem union_intersection:
  (A ∪ B).inter (C ∩ Set.Ioo (-∞:ℝ) (∞:ℝ)) = D := by
  sorry

end union_intersection_l648_648376


namespace calculate_spade_l648_648262

def spade (x y : ℝ) : ℝ := x - 1 / y

theorem calculate_spade : spade 3 (spade 3 3) = 21 / 8 := 
by {
  sorry
}

end calculate_spade_l648_648262


namespace find_F1C_CG1_l648_648406

variable {A B C D E F G H E1 F1 G1 H1 : Type*}
variables (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ) (a : ℝ)

axiom convex_quadrilateral (AE EB BF FC CG GD DH HA : ℝ) : 
  AE / EB * BF / FC * CG / GD * DH / HA = 1 

axiom quadrilaterals_similar 
  (E1F1 EF F1G1 FG G1H1 GH H1E1 HE : Prop) :
  E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True)

axiom given_ratio (E1A AH1 : ℝ) (a : ℝ) :
  E1A / AH1 = a

theorem find_F1C_CG1
  (conv : AE / EB * BF / FC * CG / GD * DH / HA = 1)
  (parallel_lines : E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True))
  (ratio : E1A / AH1 = a) :
  F1C / CG1 = a := 
sorry

end find_F1C_CG1_l648_648406


namespace intersection_of_sets_l648_648377

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x | x > 2 ∨ x < 1}

theorem intersection_of_sets :
  (A ∪ B) ∩ C = {0, 3, 4} :=
by
  sorry

end intersection_of_sets_l648_648377


namespace smallest_four_digit_number_divisible_by_smallest_primes_l648_648659

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l648_648659


namespace g_neither_even_nor_odd_l648_648363

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def g (x : ℝ) : ℝ := (x.floor : ℝ) + Real.cos (π * x)

theorem g_neither_even_nor_odd : ¬is_even_function g ∧ ¬is_odd_function g := by
  sorry

end g_neither_even_nor_odd_l648_648363


namespace cos_2x_plus_2y_l648_648698

theorem cos_2x_plus_2y (x y : ℝ) (h1 : sin x + sin y = log 2 2) (h2 : cos x + cos y = log 7 4) :
  cos (2 * (x + y)) = -7 / 25 :=
by sorry

end cos_2x_plus_2y_l648_648698


namespace probability_of_interval_l648_648087

-- Define the random variable ξ and its probability distribution P(ξ = k)
variables (ξ : ℕ → ℝ) (P : ℕ → ℝ)

-- Define a constant a
noncomputable def a : ℝ := 5/4

-- Given conditions
axiom condition1 : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 → P k = a / (k * (k + 1))
axiom condition2 : P 1 + P 2 + P 3 + P 4 = 1

-- Statement to prove
theorem probability_of_interval : P 1 + P 2 = 5/6 :=
by sorry

end probability_of_interval_l648_648087


namespace cylinder_volume_sq_lateral_area_l648_648769

theorem cylinder_volume_sq_lateral_area (h : ℝ) 
  (lateral_area : ℝ := 4 * π)
  (square_cross_section : true) 
  (cylinder_volume : ℝ) : 
  lateral_area = 4 * π → square_cross_section → cylinder_volume = 2 * π :=
begin
  intros,
  sorry,
end

end cylinder_volume_sq_lateral_area_l648_648769


namespace percentage_saved_is_2667_l648_648617

-- Define the regular price of one hat
def regular_price : ℝ := 40

-- Define the discount for the second and third hat
def second_hat_discount : ℝ := 0.5
def third_hat_discount : ℝ := 0.3

-- Define the total cost for three hats without discount
def total_regular_cost : ℝ := 3 * regular_price

-- Calculate the cost of the hats with the festival discounts
def discounted_price : ℝ := 
  (regular_price) + 
  (regular_price * (1 - second_hat_discount)) + 
  (regular_price * (1 - third_hat_discount))

-- Calculate the savings made by Maria
def savings : ℝ := total_regular_cost - discounted_price

-- Calculate the percentage savings
def percentage_savings : ℝ := (savings / total_regular_cost) * 100

theorem percentage_saved_is_2667 :
  percentage_savings = 26.67 :=
by
  sorry

end percentage_saved_is_2667_l648_648617


namespace first_digit_after_erasing_l648_648850

theorem first_digit_after_erasing (digits_erased : ℕ) (concat_numbers : ℕ → list ℕ) (n : ℕ) (remaining_digit : ℕ) :
  (concat_numbers n).take digits_erased = (concat_numbers 500).take 500 →
  remaining_digit = 3 :=
  sorry

end first_digit_after_erasing_l648_648850


namespace smallest_four_digit_divisible_by_primes_l648_648651

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l648_648651


namespace digits_base_d_l648_648077

theorem digits_base_d (d A B : ℕ) (h₀ : d > 7) (h₁ : A < d) (h₂ : B < d) 
  (h₃ : A * d + B + B * d + A = 2 * d^2 + 2) : A - B = 2 :=
by
  sorry

end digits_base_d_l648_648077


namespace Antoine_wins_when_n_is_odd_Pierre_wins_when_n_is_even_l648_648150

-- Definitions for the grid and the rules of the game
def grid (n : ℕ) := fin n → fin n → Prop
def initial_grid (n : ℕ) : grid n := λ _ _, false

-- Definitions for the game rule
def valid_move (g : grid n) (x y : fin n) : Prop :=
  ¬ g x y

def third_cross_in_2x2 (g : grid n) : Prop :=
  ∃ i j : fin (n-1), g i j ∧ g (i+1) j ∧ g i (j+1) ∧ g (i+1) (j+1)

-- Define the players
inductive player
| Antoine
| Pierre

-- Statement for n = 2011^2012 
theorem Antoine_wins_when_n_is_odd (n : ℕ) (h : n % 2 = 1) : 
  player :=
  player.Antoine
sorry

-- Statement for n = 2012^2011 
theorem Pierre_wins_when_n_is_even (n : ℕ) (h : n % 2 = 0) : 
  player :=
  player.Pierre
sorry

-- Applying the definitions for specific power cases
example : Antoine_wins_when_n_is_odd (2011 ^ 2012) (by norm_num) = player.Antoine := by sorry
example : Pierre_wins_when_n_is_even (2012 ^ 2011) (by norm_num) = player.Pierre := by sorry

end Antoine_wins_when_n_is_odd_Pierre_wins_when_n_is_even_l648_648150


namespace sum_of_primes_between_1_and_20_l648_648498

theorem sum_of_primes_between_1_and_20:
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p) = 77 :=
by
  -- Sum of primes within the given range
  have prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
  have sum_primes := ∑ p in prime_set, p
  -- Performing the actual summation and comparison
  calc sum_primes
    = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 : by sorry
    = 77 : by sorry

end sum_of_primes_between_1_and_20_l648_648498


namespace triangle_altitude_ratio_1_2_3_does_not_exist_l648_648993

def ratio_inv (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  let common_denominator := a * b * c
  (common_denominator / a, common_denominator / b, common_denominator / c)

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_altitude_ratio_1_2_3_does_not_exist :
  ∀ (a b c : ℕ), ratio_inv a b c = (6, 3, 2) → ¬ satisfies_triangle_inequality 6 3 2 :=
by
  intros a b c h
  cases h
  refine ⟨_, _, _⟩;
  sorry

end triangle_altitude_ratio_1_2_3_does_not_exist_l648_648993


namespace smallest_four_digit_number_divisible_by_smallest_primes_l648_648660

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l648_648660


namespace ellipse_equation_l648_648729

theorem ellipse_equation 
  (a b : ℝ)
  (h1 : a > b > 0)
  (h2 : ∀ (x y : ℝ), x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1)
  (h3 : (∃ A B : ℝ × ℝ, ∃ (line : ℝ), A.1 ≠ B.1 ∧ A.2 ≠ B.2 ∧ 
    (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1 ∧ 
    ∀ (x : ℝ), (∃ (y : ℝ), (x, y) ∈ {p : ℝ × ℝ | A = (3, 0) ∧ B = (x, y)})))
  : (a ^ 2 = 18) ∧ (b ^ 2 = 9) :=
sorry

end ellipse_equation_l648_648729


namespace next_term_geometric_sequence_l648_648486

theorem next_term_geometric_sequence (y : ℝ) (h0 : y ≠ 0) :
  let r := 3 * y in
  let term := 81 * y^3 in
  term * r = 243 * y^4 :=
by
  let r := 3 * y
  let term := 81 * y^3
  have h : term * r = 243 * y^4 := sorry
  exact h

end next_term_geometric_sequence_l648_648486


namespace sum_primes_between_1_and_20_l648_648508

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648508


namespace area_le_one_over_sqrt_three_l648_648437

theorem area_le_one_over_sqrt_three 
  (α β γ : ℝ) 
  (α_pos : α > 0) (β_pos : β > 0) (γ_pos : γ > 0) 
  (angle_sum : α + β + γ = Real.pi)
  (AD BE CF : ℝ)
  (AD_le_1 : AD ≤ 1) 
  (BE_le_1 : BE ≤ 1) 
  (CF_le_1 : CF ≤ 1) 
  (angle_bisectors : are_angle_bisectors α β γ AD BE CF) :
  area_of_triangle α β γ ≤ 1 / Real.sqrt 3 :=
sorry

end area_le_one_over_sqrt_three_l648_648437


namespace max_cos_of_angles_l648_648361

theorem max_cos_of_angles
  (A B C : ℝ)
  (h_triangle : A + B + C = real.pi)
  (h_sin : real.sin A ^ 2 + real.sin B ^ 2 + real.sin C ^ 2 = 2) :
  ∃ (max_value : ℝ), max_value = real.sqrt 5 ∧
  ∀ a b c, a + b + c = real.pi → real.sin a ^ 2 + real.sin b ^ 2 + real.sin c ^ 2 = 2 → 
  (real.cos a + real.cos b + 2 * real.cos c ≤ max_value) :=
begin
  sorry
end

end max_cos_of_angles_l648_648361


namespace calculate_supplementary_angle_l648_648712

def degrees := ℝ
def minutes := ℝ

structure Angle where 
  deg : degrees
  min : minutes

def supplementaryAngle (a : Angle) : Angle :=
  let totalDegrees := 180
  let totalMinutes := 0
  if totalMinutes >= a.min then 
    { deg := totalDegrees - a.deg, min := totalMinutes - a.min }
  else 
    { deg := (totalDegrees - 1) - a.deg, min := 60 + totalMinutes - a.min }

def given_angle : Angle := { deg := 39, min := 23 }

theorem calculate_supplementary_angle :
  supplementaryAngle given_angle = { deg := 140, min := 37 } :=
sorry

end calculate_supplementary_angle_l648_648712


namespace sum_of_primes_1_to_20_l648_648555

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648555


namespace foci_on_x_axis_l648_648760

def hyperbola_foci_on_x_axis (m n t : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : m^2 - n^2 = t^2): Prop :=
  n^2 - m^2 < 0

theorem foci_on_x_axis (m n t : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : m^2 - n^2 = t^2) : 
  hyperbola_foci_on_x_axis m n t h1 h2 h3 :=
begin
  sorry
end

end foci_on_x_axis_l648_648760


namespace fractions_expressible_iff_prime_l648_648276

noncomputable def is_good_fraction (a b n : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem fractions_expressible_iff_prime (n : ℕ) (hn : n > 1) :
  (∀ (a b : ℕ), b < n → ∃ (k l : ℤ), k * a + l * n = b) ↔ Prime n :=
sorry

end fractions_expressible_iff_prime_l648_648276


namespace mean_median_sum_is_11_l648_648261

theorem mean_median_sum_is_11 (m n : ℕ) (h1 : m + 5 < n)
  (h2 : (m + (m + 3) + (m + 5) + n + (n + 1) + (2 * n - 1)) / 6 = n)
  (h3 : (m + 5 + n) / 2 = n) : m + n = 11 := by
  sorry

end mean_median_sum_is_11_l648_648261


namespace probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l648_648891

noncomputable section

-- Problem 1: Probability of drawing a white ball on the third draw without replacement is 1/3.
theorem probability_third_white_no_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let totalWaysToDraw3 := Nat.choose totalBalls 3
  let favorableWays := Nat.choose (totalBalls - 1) 2 * Nat.choose white 1
  let probability := favorableWays / totalWaysToDraw3
  probability = 1 / 3 :=
by
  sorry

-- Problem 2: Probability of drawing red balls no more than 4 times in 6 draws with replacement is 441/729.
theorem probability_red_no_more_than_4_in_6_draws_with_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let p_red := red / totalBalls
  let p_X5 := Nat.choose 6 5 * p_red^5 * (1 - p_red)
  let p_X6 := Nat.choose 6 6 * p_red^6
  let probability := 1 - p_X5 - p_X6
  probability = 441 / 729 :=
by
  sorry

end probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l648_648891


namespace volume_of_pyramid_l648_648109

-- Define the geometric conditions
def rectangle_EFGH (EF GH : ℕ) : ℕ × ℕ := (EF, GH)
def point_Q (x y z : ℚ) : ℚ × ℚ × ℚ := (x, y, z)
def is_tetrahedron_with_isosceles_faces (base EF: ℚ) (altitude z: ℚ) : Prop :=
  true

-- Define the theorem (volume calculation)
theorem volume_of_pyramid (EF GH : ℚ) (x y z : ℚ)
  (hEF : EF = 15 * real.sqrt 2)
  (hGH : GH = 8 * real.sqrt 2)
  (h_ispyramid : is_tetrahedron_with_isosceles_faces (real.sqrt (EF^2 + GH^2)) z) :
  ∃ (V : ℚ), V = (578 / 4) * z :=
by
  sorry

end volume_of_pyramid_l648_648109


namespace trace_angle_of_rotating_square_and_sphere_l648_648149

theorem trace_angle_of_rotating_square_and_sphere 
  (a r : ℝ)
  (ABCD : square_plate)
  (G : sphere)
  (S : plane)
  (E : point)
  (BE : ℝ) 
  (90 : ℝ)
  (alpha : ℝ) 
  (trace_angle : ℝ) 
  (AB_is_4 : a = 4)
  (radius_is_1 : r = 1) 
  (E_intersection : intersects_at_diagonals_diagonal E ABCD)
  (G_touch_E : sphere_touches_point_at_plane G E S)
  (BE_length : BE = 2 * real.sqrt 2) : 
  alpha = 114.58 :=
sorry

end trace_angle_of_rotating_square_and_sphere_l648_648149


namespace find_side_AB_l648_648360

theorem find_side_AB 
  (B C : ℝ) (BC : ℝ) (hB : B = 45) (hC : C = 45) (hBC : BC = 10) : 
  ∃ AB : ℝ, AB = 5 * Real.sqrt 2 :=
by
  -- We add 'sorry' here to indicate that the proof is not provided.
  sorry

end find_side_AB_l648_648360


namespace madeline_hours_left_over_l648_648003

theorem madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleep_hours_per_day := 8
  let sleep_hours_per_week := sleep_hours_per_day * 7
  let part_time_hours := 20
  let total_hours_in_week := 168
  total_hours_in_week - (class_hours + homework_hours_per_week + sleep_hours_per_week + part_time_hours) = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleep_hours_per_day := 8
  let sleep_hours_per_week := sleep_hours_per_day * 7
  let part_time_hours := 20
  let total_hours_in_week := 168
  calc
    total_hours_in_week - (class_hours + homework_hours_per_week + sleep_hours_per_week + part_time_hours)
        = 168 - (18 + 4 * 7 + 8 * 7 + 20) : by rfl
    ... = 168 - (18 + 28 + 56 + 20) : by rfl
    ... = 168 - 122 : by rfl
    ... = 46 : by rfl

end madeline_hours_left_over_l648_648003


namespace intersection_of_sets_l648_648378

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x | x > 2 ∨ x < 1}

theorem intersection_of_sets :
  (A ∪ B) ∩ C = {0, 3, 4} :=
by
  sorry

end intersection_of_sets_l648_648378


namespace total_tickets_l648_648618

theorem total_tickets (n_friends : ℕ) (tickets_per_friend : ℕ) (h1 : n_friends = 6) (h2 : tickets_per_friend = 39) : n_friends * tickets_per_friend = 234 :=
by
  -- Place for proof, to be constructed
  sorry

end total_tickets_l648_648618


namespace TV_cost_l648_648840

theorem TV_cost (savings_furniture_fraction : ℚ)
                (original_savings : ℝ)
                (spent_on_furniture : ℝ)
                (spent_on_TV : ℝ)
                (hfurniture : savings_furniture_fraction = 2/4)
                (hsavings : original_savings = 600)
                (hspent_furniture : spent_on_furniture = original_savings * savings_furniture_fraction) :
                spent_on_TV = 300 := 
sorry

end TV_cost_l648_648840


namespace sum_of_primes_1_to_20_l648_648559

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l648_648559


namespace perimeter_addition_l648_648080

/-- 
  Ten 1x1 square tiles form a polygon with a perimeter of 16 units. 
  If three additional 1x1 square tiles are added such that at least one side of each new tile 
  is shared with a side of one of the original tiles, then there exists a configuration 
  where the perimeter of the new figure is 19 units.
-/
theorem perimeter_addition (initial_perimeter : ℕ) (additional_tiles : ℕ) (final_perimeter : ℕ) :
  initial_perimeter = 16 → additional_tiles = 3 → final_perimeter = 19 → 
  ∃(config : Type), config.initial_perimeter = initial_perimeter ∧ 
                    config.additional_tiles = additional_tiles ∧ 
                    config.final_perimeter = final_perimeter :=
by
  sorry

end perimeter_addition_l648_648080


namespace find_a_values_for_eccentricity_l648_648089

theorem find_a_values_for_eccentricity (a : ℝ) : 
  ( ∃ a : ℝ, ((∀ x y : ℝ, (x^2 / (a+8) + y^2 / 9 = 1)) ∧ (e = 1/2) ) 
  → (a = 4 ∨ a = -5/4)) := 
sorry

end find_a_values_for_eccentricity_l648_648089


namespace probability_top_card_is_king_l648_648170

theorem probability_top_card_is_king :
  let num_suits := 5
  let num_ranks := 13
  let total_cards := num_suits * num_ranks
  let num_kings := 5
  let probability := num_kings / total_cards
  probability = 1 / 13 :=
by
  let num_suits := 5
  let num_ranks := 13
  let total_cards := num_suits * num_ranks
  let num_kings := 5
  let probability := (num_kings : ℚ) / total_cards
  show probability = 1 / 13
  sorry  -- Proof will be provided here.

end probability_top_card_is_king_l648_648170


namespace total_weekly_reading_time_l648_648318

def morning_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def morning_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

def evening_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def evening_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

theorem total_weekly_reading_time :
  let morning_minutes := 30
  let evening_minutes := 60
  let weekdays := 5
  let weekend_days := 2
  morning_reading_weekdays morning_minutes weekdays +
  morning_reading_weekends morning_minutes +
  evening_reading_weekdays evening_minutes weekdays +
  evening_reading_weekends evening_minutes = 810 :=
by
  sorry

end total_weekly_reading_time_l648_648318


namespace polynomial_remainder_l648_648565

theorem polynomial_remainder (p : Polynomial ℝ) :
  (p.eval 2 = 3) → (p.eval 3 = 9) → ∃ q : Polynomial ℝ, p = (Polynomial.X - 2) * (Polynomial.X - 3) * q + (6 * Polynomial.X - 9) :=
by
  sorry

end polynomial_remainder_l648_648565


namespace polynomial_all_values_prime_impossible_l648_648384

theorem polynomial_all_values_prime_impossible (P : Polynomial ℤ) (h_deg : P.degree ≠ 0) :
  ¬ (∀ n : ℕ, Prime (P.eval n)) :=
sorry

end polynomial_all_values_prime_impossible_l648_648384


namespace range_f_l648_648197

-- Definition of ⊗ operation
def op (a b : ℝ) : ℝ :=
  if a ≤ b then a else b

-- Definition of f(x)
def f (x : ℝ) : ℝ :=
  op (Real.sin x) (Real.cos x)

-- Theorem statement
theorem range_f : set.range f = (set.Icc (-1 : ℝ) (Real.sqrt 2 / 2)) := 
by
  -- Proof goes here
  sorry

end range_f_l648_648197


namespace determine_a_l648_648734

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 - 2 * a * x + 1 

theorem determine_a (a : ℝ) (h : ¬ (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0)) : a > 1 :=
sorry

end determine_a_l648_648734


namespace sum_of_every_second_term_l648_648192

-- Define the problem-specific terms and conditions
def sequence_sum (a : ℕ) (n : ℕ) := ∑ i in finset.range n, (a + i)

def every_second_sum (a : ℕ) (n : ℕ) := ∑ i in finset.range ((n + 1) / 2), (a + 2 * i)

theorem sum_of_every_second_term : 
  ∀ (a n : ℕ), n = 2020 → sequence_sum a n = 5410 → every_second_sum a n = 2200 :=
by intros a n h1 h2 sorry

end sum_of_every_second_term_l648_648192


namespace jared_current_age_jared_current_age_l648_648906

theorem jared_current_age (jared_age_two_years_ago : ℕ) (tom_age_in_five_years : ℕ) : 
  (jared_age_two_years_ago + 2) = 48 :=
by
  sorry

variables (tom_curr_age : ℕ) (tom_age_two_years_ago : ℕ) (jared_age_two_years_ago : ℕ)
(h1 : tom_curr_age = tom_age_in_five_years - 5)
(h2 : tom_age_two_years_ago = tom_curr_age - 2)
(h3 : jared_age_two_years_ago = 2 * tom_age_two_years_ago)

include h1 h2 h3

theorem jared_current_age (jared_age_two_years_ago : ℕ) (tom_age_in_five_years : ℕ) : 
  (jared_age_two_years_ago + 2) = 48 :=
by
  let tom_curr_age := tom_age_in_five_years - 5
  let tom_age_two_years_ago := tom_curr_age - 2
  let jared_age_two_years_ago := 2 * tom_age_two_years_ago
  exact Eq.refl 48

#check @jared_current_age

end jared_current_age_jared_current_age_l648_648906


namespace proof_num_ordered_arrays_l648_648331

noncomputable def num_ordered_arrays (s : Finset ℕ) (conditions_correct : (s = {1, 2, 3, 4}) ∧ (∀ a b c d, (a = 1 ∨ b ≠ 1 ∨ c = 2 ∨ d ≠ 4) ∧  (a = 1 → ¬(b ≠ 1 ∧ c = 2 ∧ d ≠ 4))
    ∧ (b ≠ 1 → ¬(a = 1 ∧ c = 2 ∧ d ≠ 4))
    ∧ (c = 2 → ¬(a = 1 ∧ b ≠ 1 ∧ d ≠ 4))
    ∧ (d ≠ 4 → ¬(a = 1 ∧ b ≠ 1 ∧ c = 2)))) : ℕ :=
6

theorem proof_num_ordered_arrays : num_ordered_arrays {1, 2, 3, 4} (by {
  dsimp;
  split;
  { simp, },
  { intros a b c d,
    split,
    { 
      intro h,
      cases h;
      try {right; simp; done},
      assumption,
    },
    { intros h1 h2 h3 h4,
      split;
      intro h;
      cases h;
      try {right; simp; done},
      assumption,
    },
    { intro ha, intro hb, intro hc, intro hd,
      cases hb; cases hc; cases hd; contradiction, },
    { intro ha, intro hb, intro hc, intro hd,
      cases ha; cases hc; cases hd; contradiction, },
    { intro ha, intro hb, intro hc, intro hd,
      cases ha; cases hb; cases hd; contradiction, },
    { intro ha, intro hb, intro hc, intro hd,
      cases ha; cases hb; cases hc; contradiction, }
  },
}) = 6 := sorry

end proof_num_ordered_arrays_l648_648331


namespace sector_central_angle_l648_648287

-- Definitions and constants
def arc_length := 4 -- arc length of the sector in cm
def area := 2       -- area of the sector in cm²

-- The central angle of the sector we want to prove
def theta := 4      -- radian measure of the central angle

-- Main statement to prove
theorem sector_central_angle : 
  ∃ (r : ℝ), (1 / 2) * theta * r^2 = area ∧ theta * r = arc_length :=
by
  -- No proof is required as per the instruction
  sorry

end sector_central_angle_l648_648287


namespace monotonic_increasing_interval_l648_648101

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (x^2 - 4)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 2 < x → (f x < f (x + 1)) :=
by
  intros x h
  sorry

end monotonic_increasing_interval_l648_648101


namespace total_cost_is_eight_times_l648_648050

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l648_648050


namespace older_brother_catches_up_in_half_hour_l648_648198

-- Defining the parameters according to the conditions
def speed_younger_brother := 4 -- kilometers per hour
def speed_older_brother := 20 -- kilometers per hour
def initial_distance := 8 -- kilometers

-- Calculate the relative speed difference
def speed_difference := speed_older_brother - speed_younger_brother

theorem older_brother_catches_up_in_half_hour:
  ∃ t : ℝ, initial_distance = speed_difference * t ∧ t = 0.5 := by
  use 0.5
  sorry

end older_brother_catches_up_in_half_hour_l648_648198


namespace coloring_lattice_points_l648_648275

/-- 
Given any finite set of lattice points in a plane, 
it is always possible to color some of these points red and the others white 
in such a way that, on every line parallel to one of the coordinate axes, 
the number of red points differs from the number of white points by at most 1. 
-/
theorem coloring_lattice_points (S : Finset (ℤ × ℤ)) :
  ∃ (red : (ℤ × ℤ) → Prop) (white : (ℤ × ℤ) → Prop),
    (∀ p ∈ S, red p ∨ white p) ∧
    (∀ p ∈ S, ¬(red p ∧ white p)) ∧
    (∀ y : ℤ, ∥((S.filter (λ p, p.2 = y)).filter red).card - ((S.filter (λ p, p.2 = y)).filter white).card∥ ≤ 1) ∧
    (∀ x : ℤ, ∥((S.filter (λ p, p.1 = x)).filter red).card - ((S.filter (λ p, p.1 = x)).filter white).card∥ ≤ 1) :=
sorry

end coloring_lattice_points_l648_648275


namespace shaded_area_l648_648470

-- Define the points P and Q with the given properties
structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def P := point.mk 0 0
noncomputable def Q := point.mk (6 * real.sqrt 3) 0

-- Define the circles centered at P and Q with radius 3
def radius := 3

noncomputable def CircleP := {M: point | dist P M = radius}
noncomputable def CircleQ := {M: point | dist Q M = radius}

-- Define point R as the midpoint of PQ
noncomputable def R : point := point.mk (3 * real.sqrt 3) 0

-- Define the tangents RS and RT and the common tangent UV
-- (Details of S, T, U, V need to be defined based on geometric properties, omitted for brevity)

-- Calculate the area and prove the relationship asked
theorem shaded_area : 
  let area_ReactanglePQ := 18 * (real.sqrt 3) in
  let area_triangles := 9 in
  let area_sectors := (9 * real.pi) / 4 in
  area_ReactanglePQ - area_triangles - area_sectors = 18 * (real.sqrt 3) - 9 - (9 * real.pi) / 4 :=
begin
  sorry
end

end shaded_area_l648_648470


namespace sequence_50th_group_sum_l648_648636

theorem sequence_50th_group_sum :
  let sequence := list.concat [[(2*i - 1)] | i in list.range 1 (50 / 2 + 1)] ++ [[(2*j - 1), (2*j + 1 - 1), (2*j + 2 - 1)] | j in list.range 1 (50 / 2 + 1)];
  let group_50 := (list.drop ((25 * 1) + (25 * 3 - 3)) sequence).take 3;
  (group_50.sum = 591) :=
by
  sorry

end sequence_50th_group_sum_l648_648636


namespace angle_between_vectors_l648_648759

open Real

variables {a b : ℝ^3} -- Define a and b as vectors in ℝ^3
variables {theta : ℝ} -- Define theta as a real number representing the angle

theorem angle_between_vectors 
  (ha : ∥a∥ = 2) -- Condition 1: |a| = 2
  (hb : ∥b∥ = 4) -- Condition 2: |b| = 4
  (h_perp : inner (a + b) a = 0) -- Condition 3: (a + b) ∙ a = 0
  : theta = (2 * π) / 3 := -- Prove that theta = 2π / 3
sorry -- Proof placeholder

end angle_between_vectors_l648_648759


namespace sum_of_primes_between_1_and_20_l648_648543

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648543


namespace cross_section_area_ratio_correct_l648_648342

variable (α : ℝ)
noncomputable def cross_section_area_ratio : ℝ := 2 * (Real.cos α)

theorem cross_section_area_ratio_correct (α : ℝ) : 
  cross_section_area_ratio α = 2 * Real.cos α :=
by
  unfold cross_section_area_ratio
  sorry

end cross_section_area_ratio_correct_l648_648342


namespace product_eq_one_l648_648811

theorem product_eq_one (a b c : ℝ) (h1 : a^2 + 2 = b^4) (h2 : b^2 + 2 = c^4) (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 :=
sorry

end product_eq_one_l648_648811


namespace converse_and_inverse_false_l648_648314

def triangle_is_equilateral (T : Type) [triangle T] : Prop := sorry
def triangle_is_isosceles (T : Type) [triangle T] : Prop := sorry

theorem converse_and_inverse_false (T : Type) [triangle T]:
  (triangle_is_equilateral T → triangle_is_isosceles T) →
  ¬ (triangle_is_isosceles T → triangle_is_equilateral T) ∧ ¬ (¬ triangle_is_equilateral T → ¬ triangle_is_isosceles T) :=
by
  sorry

end converse_and_inverse_false_l648_648314


namespace sibling_pair_probability_l648_648948

theorem sibling_pair_probability
  (business_students : ℕ)
  (law_students : ℕ)
  (sibling_pairs : ℕ)
  (h_business : business_students = 500)
  (h_law : law_students = 800)
  (h_sibling : sibling_pairs = 30) :
  let total_pairs := business_students * law_students in
  let probability := sibling_pairs / (total_pairs : ℝ) in
  probability = 0.000075 := 
by 
  sorry

end sibling_pair_probability_l648_648948


namespace difference_quotient_correct_l648_648834

theorem difference_quotient_correct (a b : ℝ) :
  abs (3 * a - b) / abs (a + 2 * b) = abs (3 * a - b) / abs (a + 2 * b) :=
by
  sorry

end difference_quotient_correct_l648_648834


namespace total_cost_is_eight_x_l648_648054

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l648_648054


namespace paint_cans_for_25_rooms_l648_648402

theorem paint_cans_for_25_rooms (cans rooms : ℕ) (H1 : cans * 30 = rooms) (H2 : cans * 25 = rooms - 5 * cans) :
  cans = 15 :=
by
  sorry

end paint_cans_for_25_rooms_l648_648402


namespace problem_proof_l648_648836

noncomputable def BH_length (p q r s : ℕ) : Prop :=
  let AB := 3
  let BC := 4
  let CA := 5
  let GI := 3
  let HI := 8
  let GH := Real.sqrt 73
  let AG := 3 + (3 * Real.sqrt 73) / 8
  let AH := AG + (5 * (3 * Real.sqrt 73) / 8) / 3
  let BH := AH - AB
  p = 13 ∧ q = 1 ∧ r = 73 ∧ s = 8 ∧ BH = (p + q * Real.sqrt r) / s

theorem problem_proof : 
  ∃ (p q r s : ℕ), BH_length p q r s ∧ p + q + r + s = 95 := 
by
  have p := 13
  have q := 1
  have r := 73
  have s := 8
  use [p, q, r, s]
  sorry

end problem_proof_l648_648836


namespace inequality_holds_for_all_y_l648_648257

theorem inequality_holds_for_all_y (x : ℝ) :
  (∀ y : ℝ, y^2 - (5^x - 1) * (y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end inequality_holds_for_all_y_l648_648257


namespace dimension_sum_l648_648987

-- Define the dimensions A, B, C and areas AB, AC, BC
variables (A B C : ℝ) (AB AC BC : ℝ)

-- Conditions
def conditions := AB = 40 ∧ AC = 90 ∧ BC = 100 ∧ A * B = AB ∧ A * C = AC ∧ B * C = BC

-- Theorem statement
theorem dimension_sum : conditions A B C AB AC BC → A + B + C = (83 : ℝ) / 3 :=
by
  intro h
  sorry

end dimension_sum_l648_648987


namespace range_of_k_l648_648738

open Real

-- Definitions of the line and hyperbola
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 1
def hyperbola (x y : ℝ) : Bool := x^2 - y^2 = 4

-- Definition of having two intersection points
def has_two_intersection_points (k : ℝ) : Prop :=
  ∃ x1 x2 y1 y2, x1 ≠ x2 ∧ line k x1 = y1 ∧ line k x2 = y2 ∧ hyperbola x1 y1 ∧ hyperbola x2 y2

-- The theorem for the range of k satisfying the intersection condition
theorem range_of_k (k : ℝ) (h : has_two_intersection_points k) : 1 < k ∧ k < sqrt 5 / 2 :=
sorry

end range_of_k_l648_648738


namespace maximum_mass_difference_l648_648009

theorem maximum_mass_difference (m1 m2 : ℝ) (h1 : 19.7 ≤ m1 ∧ m1 ≤ 20.3) (h2 : 19.7 ≤ m2 ∧ m2 ≤ 20.3) :
  abs (m1 - m2) ≤ 0.6 :=
by
  sorry

end maximum_mass_difference_l648_648009


namespace find_vector_v_l648_648387

noncomputable def vec : Type := ℝ × ℝ × ℝ

def cross_product (u v : vec) : vec :=
  (u.2.2 * v.3 - u.3 * v.2.2,
   u.3 * v.1.1 - u.1.1 * v.3,
   u.1.1 * v.2.2 - u.2.2 * v.1.1)

theorem find_vector_v :
  let a : vec := (1, 2, 1)
  let b : vec := (2, -1, 0)
  let c : vec := (0, 0, 1)
  let v : vec := (3/4, 3/2, 7/4)
  cross_product v a = cross_product b a ∧
  cross_product v b = cross_product c b :=
by
  sorry

end find_vector_v_l648_648387


namespace problem_2011_Mentougou_l648_648573

theorem problem_2011_Mentougou 
  (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (H2 : ∀ x : ℝ, 0 < x → 0 < f x) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
sorry

end problem_2011_Mentougou_l648_648573


namespace point_on_y_axis_l648_648800

theorem point_on_y_axis {P : ℝ × ℝ} (h : P = (0, -4)) : P.1 = 0 :=
by {
  rw h,
  exact rfl,
}

end point_on_y_axis_l648_648800


namespace find_combinatorial_types_l648_648642

-- Define a structure for topologically regular polyhedron.
structure TopologicallyRegularPolyhedron (F : ℕ) where
  faces : List (List ℕ)
  edge_count : ℕ
  vertex_figure : List ℕ
  (h1 : ∀ f ∈ faces, f.length = edge_count)
  (h2 : ∀ v ∈ List.bind faces id, List.count v (List.bind vertex_figure id) = edge_count)

-- Define the result sets for each type of polyhedron.
def tetrahedron_combinatorial_types : List String := ["Tetrahedron"]
def pentahedron_combinatorial_types : List String := ["Quadrilateral Pyramid", "Triangular Prism"]
def hexahedron_combinatorial_types : List String := ["Triangular Bipyramid", "Square Pyramid"]

-- Problem Statement
theorem find_combinatorial_types (F : ℕ) :
  ∀ poly : TopologicallyRegularPolyhedron F,
    (F = 4 → poly.faces.map (λ x => x.length) = tetrahedron_combinatorial_types) ∧
    (F = 5 → poly.faces.map (λ x => x.length) = pentahedron_combinatorial_types) ∧
    (F = 6 → poly.faces.map (λ x => x.length) = hexahedron_combinatorial_types) :=
sorry

end find_combinatorial_types_l648_648642


namespace shaded_to_white_ratio_l648_648909

-- Define the condition: the vertices of all squares, except for the largest one,
-- are located at the midpoints of their corresponding sides.
def vertices_at_midpoints (n : ℕ) : Prop :=
  ∀ (k : ℕ), (1 ≤ k) → (k < n) → 
    ( ∀ (v : ℝ × ℝ), v ∈ vertices_of_square k → 
      v ∈ midpoints_of_sides (k + 1) )

-- Given the condition, prove the ratio of the area of the shaded part to the white part
theorem shaded_to_white_ratio (n : ℕ) (H : vertices_at_midpoints n) : 
  ratio_of_areas shaded_area white_area = 5 / 3 :=
sorry

end shaded_to_white_ratio_l648_648909


namespace total_cost_is_eight_times_l648_648046

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l648_648046


namespace calculate_boys_count_l648_648588

-- Definitions based on conditions
def total_girls := 200
def boys_attendance_rate := 0.75
def girls_nonattendance_rate := 0.40
def class_attendance_rate := 2/3

-- Definitions to compute girls_attendance_rate and total attendance
def girls_attendance_rate := 1 - girls_nonattendance_rate
def total_students (B : ℕ) := B + total_girls
def total_college_attendees (B : ℕ) := boys_attendance_rate * B + girls_attendance_rate * total_girls

theorem calculate_boys_count : 
  ∃ B : ℕ, total_college_attendees B = class_attendance_rate * total_students B ∧ B = 160 := 
by
  sorry

end calculate_boys_count_l648_648588


namespace sum_odd_digit_two_digit_numbers_l648_648676

theorem sum_odd_digit_two_digit_numbers :
  (∑ x in ({11, 13, 15, 17, 19, 31, 33, 35, 37, 39, 51, 53, 55, 57, 59,
             71, 73, 75, 77, 79, 91, 93, 95, 97, 99} : Finset ℕ), x) = 1375 :=
by
  sorry

end sum_odd_digit_two_digit_numbers_l648_648676


namespace greatest_real_part_of_cube_l648_648138

theorem greatest_real_part_of_cube :
  let w1 : ℂ := -3
  let w2 : ℂ := -2 + I
  let w3 : ℂ := 1 + 2 * I
  let w4 : ℂ := 2 + I
  let w5 : ℂ := -I
  Re((w1)^3) ≤ Re((w4)^3) ∧
  Re((w2)^3) ≤ Re((w4)^3) ∧
  Re((w3)^3) ≤ Re((w4)^3) ∧
  Re((w5)^3) ≤ Re((w4)^3) :=
by
  sorry

end greatest_real_part_of_cube_l648_648138


namespace cost_price_of_article_l648_648611

theorem cost_price_of_article (M : ℝ) (SP : ℝ) (C : ℝ) 
  (hM : M = 65)
  (hSP : SP = 0.95 * M)
  (hProfit : SP = 1.30 * C) : 
  C = 47.50 :=
by 
  sorry

end cost_price_of_article_l648_648611


namespace moving_last_digit_divisible_by_3_example_857142_example_428571_l648_648401

theorem moving_last_digit_divisible_by_3 (x : ℕ) (d1 d2 d3 d4 d5 d6 : ℕ) : 
  x = 100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6 →
  d1 + d2 + d3 + d4 + d5 + d6 = 27 → 
  (100000 * d6 + 10000 * d1 + 1000 * d2 + 100 * d3 + 10 * d4 + d5) % 3 = 0 :=
by
  intros h1 h2
  sorry

theorem example_857142 : moving_last_digit_divisible_by_3 857142 8 5 7 1 4 2 :=
by
  sorry

theorem example_428571 : moving_last_digit_divisible_by_3 428571 4 2 8 5 7 1 :=
by
  sorry

end moving_last_digit_divisible_by_3_example_857142_example_428571_l648_648401


namespace volume_of_regular_quadrangular_prism_l648_648235

noncomputable def volume_of_prism (a : ℝ) : ℝ :=
  let b := a * (real.sqrt 3)
  let h := real.sqrt (b^2 - a^2)
  a^2 * h

theorem volume_of_regular_quadrangular_prism (a : ℝ) : 
  (∃ V, V = a^3 * real.sqrt 2) :=
begin
  use volume_of_prism a,
  sorry,
end

end volume_of_regular_quadrangular_prism_l648_648235


namespace sum_primes_between_1_and_20_l648_648525

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648525


namespace number_of_ways_to_pick_three_numbers_meeting_conditions_l648_648060

theorem number_of_ways_to_pick_three_numbers_meeting_conditions :
  let S := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let T (a b c : ℕ) := a + b + c ≥ 10 ∧ (a + b + c) % 2 = 0
  { (a, b, c) ∈ (S × S × S) | T a b c }.toFinset.card = 51 :=
by
  sorry

end number_of_ways_to_pick_three_numbers_meeting_conditions_l648_648060


namespace smallest_four_digit_divisible_five_smallest_primes_l648_648662

theorem smallest_four_digit_divisible_five_smallest_primes :
  let lcm_5_min_primes := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 (Nat.lcm 7 11))
  ∀ m : ℕ, (m ≥ 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ m)) → m = 2310 :=
by
  let lcm_5_min_primes := 2310
  intro m h
  have h1 : m = 2310 := sorry
  exact h1

end smallest_four_digit_divisible_five_smallest_primes_l648_648662


namespace cost_of_purchase_l648_648033

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l648_648033


namespace smallest_divisible_four_digit_number_l648_648668

theorem smallest_divisible_four_digit_number :
  let n := 2310
  in ∃ k : ℕ, n * k >= 1000 ∧ n = 2 * 3 * 5 * 7 * 11 :=
begin
  let n := 2310,
  use 1,
  split,
  { norm_num },
  { norm_num },
  sorry
end

end smallest_divisible_four_digit_number_l648_648668


namespace subset_weight_range_l648_648922

-- Define the problem parameters and conditions as Lean definitions
variables {n : ℕ} (blocks : Fin n → ℝ)

-- Conditions: 
-- Each block weighs at least 1
def block_weights : Prop := ∀ i, blocks i ≥ 1

-- The total weight of all blocks is 2n
def total_weight : Prop := ∑ i, blocks i = 2 * n

-- The main theorem statement
theorem subset_weight_range (r : ℝ) (h : 0 ≤ r ∧ r ≤ 2*n - 2) :
  block_weights blocks → total_weight blocks → 
  ∃ (A : Finset (Fin n)), r ≤ ∑ i in A, blocks i ∧ ∑ i in A, blocks i ≤ r + 2 :=
sorry

end subset_weight_range_l648_648922


namespace smallest_four_digit_divisible_by_primes_l648_648654

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l648_648654


namespace ratio_of_triangle_areas_l648_648974

-- Defining the conditions
variables (a b : ℝ) (h_a : a > 0) (h_b : b > 0)

-- The corresponding Lean statement for the proof
theorem ratio_of_triangle_areas
  (h₁ : ∀ C D, C = (1/2) * (a / 3) * (b / 2) ∧ D = (1/2) * (b / 4) * (a / 2)) :
  ∃ r, r = (4 / 3) :=
by
  -- Using the given areas of triangles C and D to find the ratio
  let C : ℝ := (1/2) * (a / 3) * (b / 2)
  let D : ℝ := (1/2) * (b / 4) * (a / 2)
  have h_C : C = (ab / 12), from sorry
  have h_D : D = (ab / 16), from sorry
  let r := C / D
  have h_ratio : r = (4 / 3), from sorry
  exact ⟨r, h_ratio⟩

end ratio_of_triangle_areas_l648_648974


namespace sum_primes_upto_20_l648_648512

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648512


namespace final_number_geq_one_over_l648_648452

theorem final_number_geq_one_over (n : ℕ) (hn : 0 < n) : ∃ x : ℝ, (∀ (list : List ℝ), 
(list.length = n → (∀ (i : ℕ), i < n → list.nth i = some 1) → 
(pick_and_replace list (n-1) = [x])) → x ≥ 1 / n) :=
by sorry

end final_number_geq_one_over_l648_648452


namespace f_neither_odd_nor_even_l648_648433

def f (x : ℝ) : ℝ := (1 / 3^(x - 1)) - 3

theorem f_neither_odd_nor_even : 
  ¬ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∀ x : ℝ, f (-x) = -f x) :=
by 
  sorry

end f_neither_odd_nor_even_l648_648433


namespace problem_statement_l648_648736

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Math.sin (2 * x - Real.pi / 3)

theorem problem_statement:
  (∀ x, x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 2 ≤ f x ∧ f x ≤ 3) ∧
  (∀ (m : ℝ), (∀ x, x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → |f x - m| < 2) → 1 < m ∧ m < 4) :=
by
  sorry

end problem_statement_l648_648736


namespace power_inequality_l648_648272

theorem power_inequality (a b c d : ℝ) (ha : 0 < a) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end power_inequality_l648_648272


namespace probability_s_less_than_0_l648_648143

theorem probability_s_less_than_0 :
  let s (p : ℤ) := p^2 - 13 * p + 40 in
  let p_values := {p | 1 ≤ p ∧ p ≤ 10} in
  let p_satisfy := {p | 6 ≤ p ∧ p ≤ 7} in
  p_satisfy.card / p_values.card = 1 / 5 :=
by
  sorry

end probability_s_less_than_0_l648_648143


namespace line_shift_passes_point_l648_648200

theorem line_shift_passes_point (m : ℝ) (h : m > 0) :
  (∃ (y x : ℝ), (y = -x + 1) → x = 1 → y = -3 → y = -(x + m) + 1) → m = 3 :=
by
  intro h1
  have h2 : -3 = -(1 + m) + 1 := by
  cases h1 with y h1
  cases h1 with x h1
  have hy := h1.1 rfl
  have hx := h1.2
  rw[hx, hy]
  linarith
  linarith

end line_shift_passes_point_l648_648200


namespace negation_of_proposition_is_false_l648_648877

open_locale classical

theorem negation_of_proposition_is_false (x y : ℝ) :
  ¬ (x > 2 ∧ y > 3 ∧ x + y ≤ 5) :=
by
  sorry

end negation_of_proposition_is_false_l648_648877


namespace madeline_hours_left_over_l648_648002

theorem madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleep_hours_per_day := 8
  let sleep_hours_per_week := sleep_hours_per_day * 7
  let part_time_hours := 20
  let total_hours_in_week := 168
  total_hours_in_week - (class_hours + homework_hours_per_week + sleep_hours_per_week + part_time_hours) = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleep_hours_per_day := 8
  let sleep_hours_per_week := sleep_hours_per_day * 7
  let part_time_hours := 20
  let total_hours_in_week := 168
  calc
    total_hours_in_week - (class_hours + homework_hours_per_week + sleep_hours_per_week + part_time_hours)
        = 168 - (18 + 4 * 7 + 8 * 7 + 20) : by rfl
    ... = 168 - (18 + 28 + 56 + 20) : by rfl
    ... = 168 - 122 : by rfl
    ... = 46 : by rfl

end madeline_hours_left_over_l648_648002


namespace circle_tangent_to_line_l648_648250

theorem circle_tangent_to_line :
  ∃ r : ℝ, (x - 4)^2 + y^2 = r^2 ∧ r = 2 * Real.sqrt 3 ∧ ∀ x y, Real.abs (√3 * x - y) / Real.sqrt (3 + 1) = 2 * Real.sqrt 3 :=
sorry

end circle_tangent_to_line_l648_648250


namespace new_expression_decreases_71_2_percent_l648_648351

theorem new_expression_decreases_71_2_percent (x y : ℝ) :
  let original_expr := x * y^2
  let new_x := 0.8 * x
  let new_y := 0.6 * y
  let new_expr := new_x * new_y^2
  (original_expr - new_expr) / original_expr = 0.712 :=
by { 
  let original_expr := x * y^2,
  let new_x := 0.8 * x,
  let new_y := 0.6 * y,
  let new_expr := new_x * new_y^2,
  sorry
}

end new_expression_decreases_71_2_percent_l648_648351


namespace next_term_geometric_sequence_l648_648480

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l648_648480


namespace trigonometric_identity_solution_l648_648140

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 ↔
  ∃ (k : ℤ), x = Real.pi + 2 * Real.pi * k := by
  sorry

end trigonometric_identity_solution_l648_648140


namespace functions_are_equal_l648_648300

-- Define the functions as described in the problem conditions
def f4 (x : ℝ) : ℝ := 2^(-x)
def g4 (x : ℝ) : ℝ := (1/2)^x

-- State the proof obligation
theorem functions_are_equal : ∀ x : ℝ, f4 x = g4 x :=
by
  -- Placeholder for the actual proof
  sorry

end functions_are_equal_l648_648300


namespace points_lie_on_circle_l648_648683

theorem points_lie_on_circle (s : ℝ) :
  let x := (2 - s^2) / (2 + s^2)
  let y := 2 * s / (2 + s^2)
  in x^2 + y^2 = 1 :=
by { sorry }

end points_lie_on_circle_l648_648683


namespace number_of_boys_l648_648110

noncomputable def numGirls : Nat := 46
noncomputable def numGroups : Nat := 8
noncomputable def groupSize : Nat := 9
noncomputable def totalMembers : Nat := numGroups * groupSize
noncomputable def numBoys : Nat := totalMembers - numGirls

theorem number_of_boys :
  numBoys = 26 := by
  sorry

end number_of_boys_l648_648110


namespace sum_primes_between_1_and_20_l648_648507

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l648_648507


namespace arc_length_of_sector_l648_648105

theorem arc_length_of_sector : 
  ∀ (r : ℝ) (theta: ℝ), r = 1 ∧ theta = 30 * (Real.pi / 180) → (theta * r = Real.pi / 6) :=
by
  sorry

end arc_length_of_sector_l648_648105


namespace max_value_of_quadratic_function_l648_648347

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_value_of_quadratic_function (a b c m : ℝ) (h1 : a < 0) (h2 : b = -a*m) 
  (h3 : b ≥ a) (h4 : -1 ≤ m ∧ m < 0) (h5 : 1 ≤ (m + 2)^2 < 4) 
  (h6 : ∃ x, quadratic_function a b c x ≤ quadratic_function a b c x ∧ quadratic_function a b c x = -2) :
  a ≤ -8/3 := 
sorry

end max_value_of_quadratic_function_l648_648347


namespace profit_without_discount_l648_648980

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 0.05
noncomputable def profit_with_discount_percentage : ℝ := 0.387
noncomputable def selling_price_with_discount : ℝ := cost_price * (1 + profit_with_discount_percentage)

noncomputable def profit_without_discount_percentage : ℝ :=
  let selling_price_without_discount := selling_price_with_discount / (1 - discount_percentage)
  ((selling_price_without_discount - cost_price) / cost_price) * 100

theorem profit_without_discount :
  profit_without_discount_percentage = 45.635 := by
  sorry

end profit_without_discount_l648_648980


namespace police_emergency_number_has_prime_factor_gt_7_l648_648190

theorem police_emergency_number_has_prime_factor_gt_7 (k : ℕ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) := 
sorry

end police_emergency_number_has_prime_factor_gt_7_l648_648190


namespace sam_change_amount_l648_648414

theorem sam_change_amount 
  (num_cans : ℕ)
  (cost_per_can : ℕ)
  (num_coupons : ℕ)
  (discount_per_coupon : ℕ)
  (payment_dollars : ℕ)
  : num_cans = 9 → 
    cost_per_can = 175 → 
    num_coupons = 5 → 
    discount_per_coupon = 25 → 
    payment_dollars = 20 → 
    ((payment_dollars * 100) - (num_cans * cost_per_can - num_coupons * discount_per_coupon)) / 100 = 5.50 :=
by
  sorry

end sam_change_amount_l648_648414


namespace maximum_volume_prism_l648_648797

-- Define the conditions
variables {l w h : ℝ}
axiom area_sum_eq : 2 * h * l + l * w = 30

-- Define the volume of the prism
def volume (l w h : ℝ) : ℝ := l * w * h

-- Statement to be proved
theorem maximum_volume_prism : 
  (∃ l w h : ℝ, 2 * h * l + l * w = 30 ∧ 
  ∀ u v t : ℝ, 2 * t * u + u * v = 30 → l * w * h ≥ u * v * t) → volume l w h = 112.5 :=
by
  sorry

end maximum_volume_prism_l648_648797


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l648_648674

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l648_648674


namespace solve_for_integers_l648_648643

theorem solve_for_integers (x y : ℤ) : 
  (2 * x + 5 * y + 1) * (2 ^ |x| + x ^ 2 + x + y) = 105 → (x, y) = (0, 4) :=
by
  sorry

end solve_for_integers_l648_648643


namespace sum_of_primes_1_to_20_l648_648531

open Nat

theorem sum_of_primes_1_to_20 : 
  ∑ n in {n | nat.prime n ∧ n < 20 }.to_finset = 77 :=
by
  sorry

end sum_of_primes_1_to_20_l648_648531


namespace least_k_property_l648_648068

def sequence (u : ℕ → ℝ) : Prop :=
  u 0 = 1/6 ∧ ∀ k, u (k + 1) = 2 * u k - 2 * (u k) ^ 2 + 1/3

theorem least_k_property :
  (∃ u : ℕ → ℝ, sequence u ∧ tends_to (λ k, u k) (5/6) ∧
  (∀ k, |u k - 5/6| ≤ 1/2^500 → k = 9)) :=
sorry

end least_k_property_l648_648068


namespace tangent_length_l648_648098

-- Definitions and conditions
def circle_center : ℝ × ℝ := (0, 0)
def circle_radius : ℝ := 1
def pointP : ℝ × ℝ := (2, 0)

-- Distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating the length of the tangent line
theorem tangent_length : dist circle_center pointP = 2 → sqrt ((dist circle_center pointP)^2 - circle_radius^2) = sqrt 3 := 
by 
  intros h
  sorry

end tangent_length_l648_648098


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l648_648673

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l648_648673


namespace sum_of_x_on_circle_l648_648220

theorem sum_of_x_on_circle :
  let center_x := 3
  let center_y := -4
  let radius := 5 in
  let h := center_x
  let k := center_y
  let r := radius in
  let circle_eq (x y : ℝ) := (x - h)^2 + (y + 4) ^ 2 = r^2 in
  (∃ x₁ x₂ : ℝ, circle_eq x₁ 0 ∧ circle_eq x₂ 0 ∧ x₁ + x₂ = 6) :=
  sorry

end sum_of_x_on_circle_l648_648220


namespace problem_l648_648338

variables {A B C D E F P : Type}
variables {triangle : A → B → C → Prop}
variables (on_side : A → B → C → D → Prop)
variables (concurrent : A → B → C → D → E → F → P → Prop)
variables {AF FB : ℕ}
variables (area_APB : ℝ) (area_APC : ℝ)

-- Given:
-- 1. \(AF: FB = 4: 5\)
-- 2. The ratio of the area of \(\triangle APB\) to \(\triangle APC\) is \(1: 2\).

axiom h₀ : on_side A B C D
axiom h₁ : on_side B C A E
axiom h₂ : on_side C A B F
axiom h₃ : concurrent A B C D E F P
axiom h₄ : AF / FB = 4 / 5
axiom h₅ : area_APB / area_APC = 1 / 2

-- To Prove: AE : AC = 2 / 7
theorem problem : (AE / AC = 2 / 7) :=
sorry

end problem_l648_648338


namespace probability_each_player_has_five_coins_l648_648605

noncomputable def probability_each_has_5_coins : ℝ :=
  1 / 207360000

theorem probability_each_player_has_five_coins :
  ∀ (players : Fin 4 → ℕ) (rounds : ℕ) (urn : List String) (draws : Fin 5 → (Fin 4 × String)),
  (∀ i, players i = 5) →
  rounds = 5 →
  urn = ["green", "red", "blue", "white", "white"] →
  (∀ r, 
    let ⟨p1, b1⟩ := draws r,
        ⟨p2, b2⟩ := draws (r + 1 mod 5)
    in 
    (if b1 = "green" then players p1 := players p1 - 1 else players p1) =
    (if b2 = "red" then players p2 := players p2 + 1 else players p2) ∧
    (if b1 = "blue" then players p1 := players p1 - 2 else players p1) =
    (if b2 = "green" ∧ b2 = "red" then players p2 := players p2 + 1 else players p2)
  ) →
  real.ext_iff.mp (pmf.mass_of players = pmf.mass_of draws) = probability_each_has_5_coins :=
sorry

end probability_each_player_has_five_coins_l648_648605


namespace complex_expression_value_l648_648730

-- Definitions related to the problem conditions
def i := Complex.I

-- Conditions
lemma i_sq : i^2 = -1 :=
  sorry

lemma i_pow_4 : i^4 = 1 :=
  sorry

-- Problem statement to prove
theorem complex_expression_value :
  (2 * i / (1 + i)) * (2 * i - i^2016) = -3 + i :=
by
  -- Making use of the conditions
  apply i_sq,
  apply i_pow_4,
  sorry

end complex_expression_value_l648_648730


namespace sum_primes_between_1_and_20_l648_648520

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648520


namespace village_current_population_l648_648344

def initial_population : ℕ := 4675
def died_by_bombardment : ℕ := (5*initial_population + 99) / 100 -- Equivalent to rounding (5/100) * 4675
def remaining_after_bombardment : ℕ := initial_population - died_by_bombardment
def left_due_to_fear : ℕ := (20*remaining_after_bombardment + 99) / 100 -- Equivalent to rounding (20/100) * remaining
def current_population : ℕ := remaining_after_bombardment - left_due_to_fear

theorem village_current_population : current_population = 3553 := by
  sorry

end village_current_population_l648_648344


namespace product_inequality_l648_648273

theorem product_inequality (n : ℕ) (x : Fin n → ℝ) 
  (hx_pos : ∀ k, 0 < x k) 
  (hx_sum : ∑ k, x k = 1) : 
  (∏ k, (1 + x k) / x k) ≥ (∏ k, (n - x k) / (1 - x k)) :=
sorry

end product_inequality_l648_648273


namespace union_A_B_compl_inter_A_B_l648_648315

-- Definitions based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

def B : Set ℝ := {x | 2 * x - 9 ≥ 6 - 3 * x}

-- The first proof statement
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ 2} := by
  sorry

-- The second proof statement
theorem compl_inter_A_B : U \ (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 4} := by
  sorry

end union_A_B_compl_inter_A_B_l648_648315


namespace coefficients_balance_l648_648826

noncomputable def num_positive_coeffs (n : ℕ) : ℕ :=
  n + 1

noncomputable def num_negative_coeffs (n : ℕ) : ℕ :=
  n + 1

theorem coefficients_balance (n : ℕ) (h_odd: Odd n) (x : ℝ) :
  num_positive_coeffs n = num_negative_coeffs n :=
by
  sorry

end coefficients_balance_l648_648826


namespace combined_transformation_matrix_l648_648647

open Matrix

-- Definitions of the matrices
def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 0],
  ![0, 2]
]

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, 1],
  ![-1, 0]
]

-- The proof statement
theorem combined_transformation_matrix :
  (rotation_matrix ⬝ dilation_matrix) = ![
    ![0, 2],
    ![-2, 0]
  ] :=
by
  simp [rotation_matrix, dilation_matrix]
  sorry

end combined_transformation_matrix_l648_648647


namespace average_pages_per_book_l648_648396

theorem average_pages_per_book :
  let pages := [120, 150, 180, 210, 240]
  let num_books := 5
  let total_pages := pages.sum
  total_pages / num_books = 180 := by
  sorry

end average_pages_per_book_l648_648396


namespace find_base_b_l648_648784

theorem find_base_b : ∃ b : ℕ, b > 4 ∧ (b + 2)^2 = b^2 + 4 * b + 4 ∧ b = 5 := 
sorry

end find_base_b_l648_648784


namespace sector_angle_l648_648721

theorem sector_angle (r l : ℝ) (h1 : l + 2 * r = 6) (h2 : 1/2 * l * r = 2) : 
  l / r = 1 ∨ l / r = 4 := 
sorry

end sector_angle_l648_648721


namespace dot_product_value_sum_norm_value_l648_648749

variables (a b : ℝ) (v_a v_b : EuclideanSpace ℝ (Fin 2))
noncomputable def norm_eq_two : Prop := ∥v_a∥ = 2
noncomputable def norm_eq_one : Prop := ∥v_b∥ = 1
noncomputable def norm_diff_eq_two : Prop := ∥v_a - v_b∥ = 2

theorem dot_product_value (ha : norm_eq_two v_a) (hb : norm_eq_one v_b) (hd : norm_diff_eq_two v_a v_b) :
  (2 • v_a + v_b) ⋅ (v_a - 2 • v_b) = 9 / 2 :=
sorry

theorem sum_norm_value (ha : norm_eq_two v_a) (hb : norm_eq_one v_b) (hd : norm_diff_eq_two v_a v_b) :
  ∥v_a + v_b∥ = real.sqrt 6 :=
sorry

end dot_product_value_sum_norm_value_l648_648749


namespace problem1_problem2_problem3_problem4_problem5_l648_648156

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + 4 * y^2 = 64

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 47) = 1

noncomputable def foci_equiv (x y : ℝ) : Prop :=
  x^2 + (4 * y^2) = (64)

noncomputable def distance_property (x y : ℝ) : Prop :=
  abs (sqrt (x^2 + (4 * y^2))) - abs (sqrt (x^2 + (4 * y^2))) = 2

theorem problem1 (x y : ℝ) (h1 : ellipse_equation x y) (h2 : foci_equiv x y) (h3 : distance_property x y) : hyperbola_equation x y :=
  sorry

noncomputable def function1 (x : ℝ) : ℝ :=
  f'(1) * x^3 - 2 * x^2 + 3

theorem problem2 (h1 : function1 (1) = f'(1) * (1)^3 - 2 * (1)^2 + 3) : f'(1) = 2 :=
  sorry

noncomputable def function2 (x : ℝ) : ℝ :=
  ln x - (1 / 2) * a * x^2 - 2 * x

theorem problem3 (h1 : function2 2 = 0) : a = - (3 / 4) :=
  sorry

noncomputable def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def line45 (x y : ℝ) : Prop :=
  y = x - 1

theorem problem4 (x1 x2 y1 y2: ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) (h3 : line45 x1 y1) (h4 : line45 x2 y2) : |(y2 - y1)| = 8 :=
  sorry

noncomputable def parabola2 (x y : ℝ) : Prop :=
  y^2 = 2 * x

noncomputable def point_A (x y : ℝ) : Prop :=
  x = 4 ∧ y = 2

noncomputable def min_distance (MF MA : ℝ) : Prop :=
  MF + MA is minimized

theorem problem5 (x y : ℝ) (h1 : point_A x y) (h2 : parabola2 x y) (h3 : min_distance|stances MF MA) : (x = 2 ∧ y = 2) :=
  sorry

end problem1_problem2_problem3_problem4_problem5_l648_648156


namespace sum_of_primes_between_1_and_20_l648_648542

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l648_648542


namespace sculptures_not_on_display_eq_1200_l648_648203

-- Define the number of pieces of art in the gallery
def total_pieces_art := 2700

-- Define the number of pieces on display (1/3 of total pieces)
def pieces_on_display := total_pieces_art / 3

-- Define the number of pieces not on display
def pieces_not_on_display := total_pieces_art - pieces_on_display

-- Define the number of sculptures on display (1/6 of pieces on display)
def sculptures_on_display := pieces_on_display / 6

-- Define the number of paintings not on display (1/3 of pieces not on display)
def paintings_not_on_display := pieces_not_on_display / 3

-- Prove the number of sculptures not on display
theorem sculptures_not_on_display_eq_1200 :
  total_pieces_art = 2700 →
  pieces_on_display = total_pieces_art / 3 →
  pieces_not_on_display = total_pieces_art - pieces_on_display →
  sculptures_on_display = pieces_on_display / 6 →
  paintings_not_on_display = pieces_not_on_display / 3 →
  pieces_not_on_display - paintings_not_on_display = 1200 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sculptures_not_on_display_eq_1200_l648_648203


namespace percent_games_lost_l648_648445

theorem percent_games_lost (won_lost_ratio_num won_lost_ratio_den total_games : ℕ)
  (h_ratio : won_lost_ratio_num = 13) (h_ratio_den : won_lost_ratio_den = 3)
  (h_total : total_games = 64) : 
  (19 : ℕ) = Int.round ((won_lost_ratio_den / (won_lost_ratio_num + won_lost_ratio_den) * total_games : ℝ) * 100 / total_games) :=
by
  sorry

end percent_games_lost_l648_648445


namespace cost_of_purchase_l648_648035

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l648_648035


namespace sum_not_divisible_by_10_iff_l648_648757

theorem sum_not_divisible_by_10_iff (n : ℕ) :
  ¬ (1981^n + 1982^n + 1983^n + 1984^n) % 10 = 0 ↔ n % 4 = 0 :=
sorry

end sum_not_divisible_by_10_iff_l648_648757


namespace coefficient_of_x4_in_expansion_l648_648248

open Real

-- Definitions of the expressions involved
noncomputable def expr1 := x * (x - (2 / x))^7
noncomputable def coeff_x4 := (84 : ℤ)  -- The coefficient we aim to prove

-- Statement of the theorem
theorem coefficient_of_x4_in_expansion (x : ℝ) (hx : x ≠ 0) :
  coeff_x4 = 84 :=
sorry

end coefficient_of_x4_in_expansion_l648_648248


namespace find_f_3003_l648_648447

-- Define function f
variable (f : ℕ → ℝ)

-- Define the given condition on f
def func_property (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a + b = 3^n → f(a) + f(b) = 2 * n^2

-- State the main theorem
theorem find_f_3003 (H : func_property f) : f(3003) = 44 :=
by
  sorry -- proof is omitted

end find_f_3003_l648_648447


namespace num_arrangements_l648_648458
-- Import the necessary library from Lean 4

-- Define the proof problem in Lean 4 statements
open Nat
open Fintype

theorem num_arrangements :
  (∃(A B C : finset (Σ(i : fin 3), fin 3)) 
    [decidable_pred (λ x, x ∈ A)] 
    [decidable_pred (λ x, x ∈ B)] 
    [decidable_pred (λ x, x ∈ C)],
    -- Define each letter appears exactly once in each row and column
    A.card = 3 ∧ B.card = 3 ∧ C.card = 3 ∧
    -- Each letter must appear exactly once per row
    (∀i j, (⟨i, j⟩ : Σ(i : fin 3), fin 3) ∈ A ∨ 
            (⟨i, j⟩ : Σ(i : fin 3), fin 3) ∈ B ∨ 
            (⟨i, j⟩ : Σ(i : fin 3), fin 3) ∈ C) ∧
    -- Each letter must appear exactly once per column
    (∀j i, (⟨i, j⟩ : Σ(i : fin 3), fin 3) ∈ A ∨ 
            (⟨i, j⟩ : Σ(i : fin 3), fin 3) ∈ B ∨ 
            (⟨i, j⟩ : Σ(i : fin 3), fin 3) ∈ C) ∧
    -- Fix B in the upper left corner
    (⟨0, 0⟩ : Σ(i, fin 3)) ∈ B
  ) ∧ (∃(count : ℕ), count = 4) :=
sorry

end num_arrangements_l648_648458


namespace max_value_f_on_interval_l648_648872

def f (x : ℝ) : ℝ := -4 * x^3 + 3 * x + 2

theorem max_value_f_on_interval : ∃ x ∈ set.Icc (0 : ℝ) (1 : ℝ), ∀ y ∈ set.Icc (0 : ℝ) (1 : ℝ), f y ≤ f x ∧ f x = 3 :=
by
  sorry

end max_value_f_on_interval_l648_648872


namespace cost_of_purchase_l648_648037

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l648_648037


namespace finn_boxes_of_paper_clips_l648_648240

theorem finn_boxes_of_paper_clips (cost_per_box_paper_clips cost_per_pkg_index_cards : ℝ)
  (cost_eldora : ℝ) (boxes_eldora : ℕ) (pkgs_index_cards_eldora : ℕ)
  (cost_finn : ℝ) (pkgs_index_cards_finn : ℕ) (y : ℕ) :
  boxes_eldora = 15 ∧ pkgs_index_cards_eldora = 7 ∧ cost_eldora = 55.40 ∧
  pkgs_index_cards_finn = 10 ∧ cost_finn = 61.70 ∧ cost_per_box_paper_clips = 1.85 ∧
  (15 * cost_per_box_paper_clips + 7 * cost_per_pkg_index_cards = cost_eldora) ∧
  (y * cost_per_box_paper_clips + 10 * cost_per_pkg_index_cards = cost_finn) →
  y = 12 :=
begin
  sorry
end

end finn_boxes_of_paper_clips_l648_648240


namespace min_period_and_max_value_l648_648874

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem min_period_and_max_value :
  (∃ T > 0, T = 6 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧
  (∀ x, f x ≤ sqrt 2 ∧ (∃ x0, f x0 = sqrt 2)) :=
by
  sorry

end min_period_and_max_value_l648_648874


namespace petya_purchase_cost_l648_648040

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l648_648040


namespace not_perfect_square_7_pow_2025_all_others_perfect_squares_l648_648918

theorem not_perfect_square_7_pow_2025 :
  ¬ (∃ x : ℕ, x^2 = 7^2025) :=
sorry

theorem all_others_perfect_squares :
  (∃ x : ℕ, x^2 = 6^2024) ∧
  (∃ x : ℕ, x^2 = 8^2026) ∧
  (∃ x : ℕ, x^2 = 9^2027) ∧
  (∃ x : ℕ, x^2 = 10^2028) :=
sorry

end not_perfect_square_7_pow_2025_all_others_perfect_squares_l648_648918


namespace calculate_expression_l648_648679

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 :=
by
  sorry

end calculate_expression_l648_648679


namespace B_took_18_more_boxes_than_D_l648_648604

noncomputable def A_boxes : ℕ := sorry
noncomputable def B_boxes : ℕ := A_boxes + 4
noncomputable def C_boxes : ℕ := sorry
noncomputable def D_boxes : ℕ := C_boxes + 8
noncomputable def A_owes_C : ℕ := 112
noncomputable def B_owes_D : ℕ := 72

theorem B_took_18_more_boxes_than_D : (B_boxes - D_boxes) = 18 :=
sorry

end B_took_18_more_boxes_than_D_l648_648604


namespace sum_positive_integers_condition_l648_648255

theorem sum_positive_integers_condition (h : ∀ n : Nat, 1.5 * (n : Real)^2 - 6 < 10 → n = 1 ∨ n = 2 ∨ n = 3) :
  ∑ k in {1, 2, 3} : Finset Nat, k = 6 :=
by
  have : (∑ k in {1, 2, 3} : Finset Nat, k) = 6 := by
    simp [Finset.sum]
    norm_num
  exact this

end sum_positive_integers_condition_l648_648255


namespace exists_same_color_points_distance_one_l648_648187

theorem exists_same_color_points_distance_one
    (color : ℝ × ℝ → Fin 3)
    (h : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_distance_one_l648_648187


namespace parallelogram_area_complex_solutions_l648_648860

theorem parallelogram_area_complex_solutions :
  (∃ (z1 z2 w1 w2 : ℂ), z1^2 = (3 + 3*sqrt(7)*complex.I) ∧ z2^2 = (3 + 3*sqrt(7)*complex.I) ∧ w1^2 = (5 + 5*sqrt(3)*complex.I) ∧ w2^2 = (5 + 5*sqrt(3)*complex.I) ∧ 
  abs ((z1 * conj w1).im + (z2 * conj w1).im + (z1 * conj w2).im + (z2 * conj w2).im) = 2 * sqrt(6) + 6 * sqrt(14)) :=
sorry

end parallelogram_area_complex_solutions_l648_648860


namespace length_of_other_train_l648_648473

-- Given conditions
def speed_first_train := 210  -- in km/hr
def speed_second_train := 90  -- in km/hr
def length_first_train := 1.10  -- in km
def crossing_time := 24  -- in seconds

-- Proof problem: length of the other train
theorem length_of_other_train : 
  let relative_speed_km_per_hr := speed_first_train + speed_second_train
  let relative_speed_m_per_s := (relative_speed_km_per_hr * 5) / 18
  let total_distance_m := relative_speed_m_per_s * crossing_time
  let length_first_train_m := length_first_train * 1000 in
  total_distance_m - length_first_train_m = 899.92 := sorry

end length_of_other_train_l648_648473


namespace x_in_set_implies_value_l648_648270

theorem x_in_set_implies_value (x : ℝ) (h : x ∈ ({1, 2, x^2 - x} : set ℝ)) : x = 0 ∨ x = 1 :=
by sorry

end x_in_set_implies_value_l648_648270


namespace petya_purchase_cost_l648_648043

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l648_648043


namespace number_of_students_l648_648929

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : 95 * (N - 5) = T - 100) : N = 25 :=
by
  sorry

end number_of_students_l648_648929


namespace sum_reciprocals_l648_648299

-- Assumptions
variables {a b c : ℝ}
variables (h_poly : (40 * a^3 - 70 * a^2 + 32 * a - 2 = 0) 
                ∧ (40 * b^3 - 70 * b^2 + 32 * b - 2 = 0) 
                ∧ (40 * c^3 - 70 * c^2 + 32 * c - 2 = 0))
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_between : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1)

-- Theorem to prove
theorem sum_reciprocals (h_poly h_distinct h_between) : 
  (1/(1-a)) + (1/(1-b)) + (1/(1-c)) = 11 / 20 := 
sorry

end sum_reciprocals_l648_648299


namespace monkey_slip_distance_l648_648181

theorem monkey_slip_distance :
  ∃ s : ℝ, 
    (∀ (c T H : ℝ), c = 6 → T = 37 → H = 60 →
      let net_climb := H - c in
      let climb_time := (T - 1) / 2 in
      let slip_time := (T - 1) / 2 in
      let total_climb := climb_time * c in
      total_climb - slip_time * s = net_climb → s = 3) :=
sorry

end monkey_slip_distance_l648_648181


namespace sum_of_nonnegative_reals_l648_648777

theorem sum_of_nonnegative_reals (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 52) (h2 : a * b + b * c + c * a = 24) (h3 : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  a + b + c = 10 :=
sorry

end sum_of_nonnegative_reals_l648_648777


namespace total_cost_shorts_tshirt_boots_shinguards_l648_648020

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l648_648020


namespace gym_hours_per_week_l648_648369

-- Definitions for conditions
def timesAtGymEachWeek : ℕ := 3
def weightliftingTimeEachDay : ℕ := 1
def warmupCardioFraction : ℚ := 1 / 3

-- The theorem to prove
theorem gym_hours_per_week : (timesAtGymEachWeek * (weightliftingTimeEachDay + weightliftingTimeEachDay * warmupCardioFraction) = 4) := 
by
  sorry

end gym_hours_per_week_l648_648369


namespace dodecagon_diagonals_intersections_and_regions_l648_648114

theorem dodecagon_diagonals_intersections_and_regions
  (dodecagon : Type)
  (vertices : Fin 12 → dodecagon)
  (even_indices_diagonals : ∀ i : Fin 12, i.even → ∀ j : Fin 12, i ≠ j → {line : dodecagon | line = segment (vertices i) (vertices j)} ) :
  ∃ (intersection_points regions : ℕ), intersection_points = 193 ∧ regions = 264 :=
by
  sorry

end dodecagon_diagonals_intersections_and_regions_l648_648114


namespace margo_total_distance_l648_648006

noncomputable def total_distance_walked (t1 t2 : ℝ) (v1 : ℝ) (v2 : ℝ) : ℝ :=
  (t1 * v1) + (t2 * v2)

def conditions (t1 t2 v1 d : ℝ) : Prop :=
  d = t1 * v1 ∧ d = t2 * v2

theorem margo_total_distance : conditions (15.0 / 60.0) (25.0 / 60.0) 4 (1) ∧
    total_distance_walked (15.0 / 60.0) (25.0 / 60.0) 4 (2.4) = 2 :=
by
  sorry

end margo_total_distance_l648_648006


namespace positive_diagonal_difference_l648_648625

def initial_matrix : Matrix (Fin 5) (Fin 5) ℕ :=
  ![![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![11, 12, 13, 14, 15],
    ![16, 17, 18, 19, 20],
    ![21, 22, 23, 24, 25]]

def transformed_matrix : Matrix (Fin 5) (Fin 5) ℕ :=
  ![![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![15, 14, 13, 12, 11],
    ![16, 17, 18, 19, 20],
    ![25, 24, 23, 22, 21]]

def main_diagonal (m : Matrix (Fin 5) (Fin 5) ℕ) : Fin 5 → ℕ
  | ⟨0, _⟩ => m 0 0
  | ⟨1, _⟩ => m 1 1
  | ⟨2, _⟩ => m 2 2
  | ⟨3, _⟩ => m 3 3
  | ⟨4, _⟩ => m 4 4
  | _ => 0

def secondary_diagonal (m : Matrix (Fin 5) (Fin 5) ℕ) : Fin 5 → ℕ
  | ⟨0, _⟩ => m 0 4
  | ⟨1, _⟩ => m 1 3
  | ⟨2, _⟩ => m 2 2
  | ⟨3, _⟩ => m 3 1
  | ⟨4, _⟩ => m 4 0
  | _ => 0

noncomputable def diff_diagonal_sums : ℕ :=
  (List.sum (List.ofFn (main_diagonal transformed_matrix)) 
   - List.sum (List.ofFn (secondary_diagonal transformed_matrix))).natAbs

theorem positive_diagonal_difference :
  diff_diagonal_sums = 8 := by
  sorry

end positive_diagonal_difference_l648_648625


namespace cars_meeting_distance_from_midpoint_l648_648689

def distance_AB : ℝ := 240
def speed_car1 : ℝ := 60
def speed_car2 : ℝ := 80
def midpoint_distance : ℝ := distance_AB / 2
def meeting_distance : ℝ := 17.14

theorem cars_meeting_distance_from_midpoint :
  ∀ (d_AB : ℝ) (s_c1 : ℝ) (s_c2 : ℝ) (mid_d : ℝ),
  d_AB = distance_AB →
  s_c1 = speed_car1 →
  s_c2 = speed_car2 →
  mid_d = midpoint_distance →
  (abs ((s_c1 / (s_c1 + s_c2)) * d_AB - mid_d) = meeting_distance) :=
by
  intros d_AB s_c1 s_c2 mid_d h_d_AB h_sc1 h_sc2 h_mid_d
  rw [h_d_AB, h_sc1, h_sc2, h_mid_d]
  sorry

end cars_meeting_distance_from_midpoint_l648_648689


namespace locus_is_parallel_to_m_l648_648474

open Set

variables {r R : ℝ} (m : Set (ℝ × ℝ)) -- assuming the line m is represented in the Euclidean plane ℝ²

noncomputable def locus_of_tangents : Set (ℝ × ℝ) :=
{M | ∃ (O₁ O₂ : ℝ × ℝ), dist O₁ O₂ = r + R ∧ ∀ x ∈ M, 
        dist x m = (2 * r * R) / (R + r) }

theorem locus_is_parallel_to_m (r R : ℝ) (m : Set (ℝ × ℝ)) :
  ∃ (p : ℝ), locus_of_tangents m = { x | ∃ y, y ∈ m ∧ x = (y.1, y.2 + p) } :=
sorry

end locus_is_parallel_to_m_l648_648474


namespace area_of_triangle_value_of_a_l648_648337

-- Definitions and conditions
variable (A B C : Type)
variable [triangle A B C]
variable (a b c : ℝ) -- sides opposite to angles A, B, and C
variable (cosA : ℝ) (cosA_eq : cosA = 3 / 5)
variable (b_plus_c : ℝ) (b_plus_c_eq : b_plus_c = 6)
variable (bc : ℝ) (bc_eq : bc = 5)

-- Question 1: Prove the area of triangle ABC is 2
theorem area_of_triangle :
  let sinA := real.sqrt (1 - (3 / 5) ^ 2) in
  sinA = 4 / 5 →
  let S := (1 / 2) * bc * sinA in
  S = 2 :=
by sorry

-- Question 2: Prove the value of a is 2√5
theorem value_of_a : 
  let cosA := 3 / 5 in 
  let b_plus_c := 6 in 
  let bc := 5 in
  let a_sq := (b_plus_c) ^ 2 - 2 * bc * cosA in
  sqrt a_sq = 2 * real.sqrt 5 :=
by sorry

end area_of_triangle_value_of_a_l648_648337


namespace Naomi_l648_648614

def Katherine.time_per_website := 20
def Naomi.extra_time_fraction := 1 / 4
def Naomi.total_time_spent := 750

theorem Naomi.developed_websites :
  let k_time := Katherine.time_per_website
      n_extra_frac := Naomi.extra_time_fraction
      total_time_naomi := Naomi.total_time_spent
      n_time_per_website := k_time + n_extra_frac * k_time
  in total_time_naomi / n_time_per_website = 30 :=
by
  sorry

end Naomi_l648_648614


namespace find_other_number_l648_648436

-- Define LCM and HCF conditions
def lcm_a_b := 2310
def hcf_a_b := 83
def number_a := 210

-- Define the problem to find the other number
def number_b : ℕ :=
  lcm_a_b * hcf_a_b / number_a

-- Statement: Prove that the other number is 913
theorem find_other_number : number_b = 913 := by
  -- Placeholder for proof
  sorry

end find_other_number_l648_648436


namespace sum_primes_between_1_and_20_l648_648522

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l648_648522


namespace price_per_foot_correct_l648_648489

-- Defining the area and total cost
def area : ℝ := 289
def total_cost : ℝ := 3495.28

-- Defining the radius using the area of the circle
def radius : ℝ := real.sqrt (area / real.pi)

-- Defining the circumference
def circumference : ℝ := 2 * real.pi * radius

-- Defining the price per foot
def price_per_foot : ℝ := total_cost / circumference

-- Stating the theorem that needs to be proved
theorem price_per_foot_correct : abs (price_per_foot - 102.80) < 0.01 :=
by sorry

end price_per_foot_correct_l648_648489


namespace coeff_friction_of_rod_l648_648580

theorem coeff_friction_of_rod (W : ℝ) (alpha : ℝ) (N : ℝ) (deg_alpha : alpha = 70)
     (reaction_force_eq : N = 21 * W) : ∃ μ : ℝ, ( μ ≈ 0.05 ) := by
  sorry

end coeff_friction_of_rod_l648_648580


namespace total_cost_shorts_tshirt_boots_shinguards_l648_648023

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l648_648023


namespace train_speed_is_54_kmph_l648_648989

def train_length : ℝ := 180  -- in meters
def bridge_length : ℝ := 660  -- in meters
def time_to_cross_bridge : ℝ := 55.99552035837134  -- in seconds

def total_distance := train_length + bridge_length
def speed_in_m_per_s := total_distance / time_to_cross_bridge
def speed_in_km_per_h := speed_in_m_per_s * 3.6

theorem train_speed_is_54_kmph : speed_in_km_per_h ≈ 54 :=
by
  have total_distance_calc : total_distance = 840 := by norm_num
  have speed_in_m_per_s_calc : speed_in_m_per_s ≈ 15 := by
    unfold speed_in_m_per_s total_distance
    rw [total_distance_calc]
    norm_num
  have speed_in_km_per_h_calc : speed_in_km_per_h ≈ 54 := by
    unfold speed_in_km_per_h
    rw [speed_in_m_per_s_calc]
    norm_num
  exact speed_in_km_per_h_calc

end train_speed_is_54_kmph_l648_648989


namespace range_of_a_l648_648832

def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1
def q (x a : ℝ) : Prop := x^2 + (2*a + 1)*x + a*(a + 1) ≥ 0

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, p x → q x a) ∧ (¬ ∀ x : ℝ, q x a → p x) → a ∈ Iic (-4) ∪ Ici (-1) := by
  sorry

end range_of_a_l648_648832


namespace closest_whole_number_area_shaded_l648_648166

noncomputable def area_shaded_closest_integer (rect_width rect_height circle_diameter : ℝ) : ℤ :=
  let rect_area := rect_width * rect_height
  let circle_radius := circle_diameter / 2
  let circle_area := Real.pi * circle_radius^2
  let shaded_area := rect_area - circle_area
  Int.round shaded_area

theorem closest_whole_number_area_shaded :
  area_shaded_closest_integer 4 5 2 = 17 :=
by
  sorry

end closest_whole_number_area_shaded_l648_648166


namespace jane_wins_probability_l648_648982

def num_sectors := 7

def jane_wins (x y : ℕ) : Prop := |x - y| < 4

def total_outcomes := num_sectors * num_sectors

noncomputable def probability_jane_wins : ℚ :=
  1 - (12 : ℚ) / total_outcomes  -- 12 is the number of losing outcomes calculated manually

theorem jane_wins_probability : probability_jane_wins = 37 / 49 := by sorry

end jane_wins_probability_l648_648982


namespace slope_of_parallelogram_line_l648_648627

noncomputable def p_plus_q (p q : ℕ) (coprime : pnat.coprime p q) (slope_frac : rat) : ℕ :=
if slope_frac.num = p ∧ slope_frac.denom = q then p + q else 0

theorem slope_of_parallelogram_line :
  let p := 3
  let q := 1
  let coprime := pnat.coprime p q
  let vertices := [(20, 50), (20, 150), (60, 190), (60, 90)]
  let slope_frac := (3 : ℚ)
  p_plus_q p q coprime slope_frac = 4 :=
by sorry

end slope_of_parallelogram_line_l648_648627


namespace solve_for_m_l648_648420

theorem solve_for_m (m : ℤ) : 3^m * 9^m = 81^(m - 24) ↔ m = 96 := by
  sorry

end solve_for_m_l648_648420


namespace perimeter_of_rectangle_l648_648882

theorem perimeter_of_rectangle (area : ℝ) (num_squares : ℕ) (square_side : ℝ) (width : ℝ) (height : ℝ) 
  (h1 : area = 216) (h2 : num_squares = 6) (h3 : area / num_squares = square_side^2)
  (h4 : width = 3 * square_side) (h5 : height = 2 * square_side) : 
  2 * (width + height) = 60 :=
by
  sorry

end perimeter_of_rectangle_l648_648882


namespace value_f_half_l648_648731

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

-- State the theorem
theorem value_f_half : f (1 / 2) = 0 := by
  sorry

end value_f_half_l648_648731


namespace distinct_collections_l648_648001

-- Define the letters and conditions
def vowels := ['A', 'O', 'U']
def consonants := ['M', 'H', 'C', 'S', 'T', 'T', 'N', 'N']

-- The condition of the number of letters falling off
def num_vowels_fall_off := 3
def num_consonants_fall_off := 2

-- The correct answer that needs to be proven
theorem distinct_collections:
  (pick 3 vowels from vowels) *
  (pick 2 consonants from consonants) = 16 := by
  sorry

end distinct_collections_l648_648001


namespace number_problem_l648_648144

theorem number_problem (x : ℤ) (h1 : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := by
  sorry

end number_problem_l648_648144


namespace chris_sick_weeks_l648_648219

theorem chris_sick_weeks :
  ∀ (h1 : ∀ w : ℕ, w = 4 → 2 * w = 8),
    ∀ (h2 : ∀ h w : ℕ, h = 20 → ∀ m : ℕ, 2 * (w * m) = 160),
    ∀ (h3 : ∀ h : ℕ, h = 180 → 180 - 160 = 20),
    ∀ (h4 : ∀ h w : ℕ, h = 20 → w = 20 → 20 / 20 = 1),
    180 - 160 = (20 / 20) * 20 :=
by
  intros
  sorry

end chris_sick_weeks_l648_648219


namespace correct_operation_B_l648_648919

theorem correct_operation_B (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 :=
sorry

end correct_operation_B_l648_648919


namespace circle_area_ratio_l648_648946

variables (rK rM : ℝ)

def radius_ratio (rK rM : ℕ) : ℝ := (rK / rM) ^ 2

theorem circle_area_ratio (rK rM : ℝ) (h1 : rK > 0) (h2 : rM > 0) :
  radius_ratio rK rM = (rK ^ 2 / rM ^ 2) :=
by sorry

end circle_area_ratio_l648_648946


namespace find_x_plus_y_l648_648718

def direction_vector_l1 (x : ℝ) : Vector ℝ := {2, 4, x}
def direction_vector_l2 (y : ℝ) : Vector ℝ := {2, y, 2}
def magnitude (v : Vector ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)
def dot_product (v1 v2 : Vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_x_plus_y (x y : ℝ)
  (h1 : magnitude (direction_vector_l1 x) = 6)
  (h2 : dot_product (direction_vector_l1 x) (direction_vector_l2 y) = 0) :
  x + y = -3 ∨ x + y = 1 :=
by
  sorry

end find_x_plus_y_l648_648718


namespace sum_primes_upto_20_l648_648509

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648509


namespace problem_statement_l648_648329

noncomputable def decimalPartSqrtFive : ℝ := Real.sqrt 5 - 2
def integerPartSqrtThirteen : ℕ := 3

theorem problem_statement :
  decimalPartSqrtFive + integerPartSqrtThirteen - Real.sqrt 5 = 1 :=
by
  sorry

end problem_statement_l648_648329


namespace units_digit_quotient_l648_648878

theorem units_digit_quotient (n : ℕ) :
  (2^1993 + 3^1993) % 5 = 0 →
  ((2^1993 + 3^1993) / 5) % 10 = 3 := by
  sorry

end units_digit_quotient_l648_648878


namespace imaginary_part_of_complex_number_l648_648435

theorem imaginary_part_of_complex_number : 
  let z : ℂ := (1 - 2 * complex.i) / (1 + complex.i ^ 3)
  in complex.im z = -1 / 2 :=
by
  let z : ℂ := (1 - 2 * complex.i) / (1 + complex.i ^ 3)
  show complex.im z = -1 / 2
  sorry

end imaginary_part_of_complex_number_l648_648435


namespace smallest_four_digit_number_divisible_by_smallest_primes_l648_648656

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l648_648656


namespace relationship_among_a_b_c_l648_648735

noncomputable def f : ℝ → ℝ := λ x, (exp (-x)) - (exp x)

def a : ℝ := - f (Real.log 3 / Real.log (1 / 2))
def b : ℝ := f (Real.log (1 / 5) / Real.log (1 / 2))
def c : ℝ := f (2 ^ (-0.2))

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  -- We'll provide the precise proof here
  sorry

end relationship_among_a_b_c_l648_648735


namespace distance_AC_l648_648000

variables (m n : ℕ) (hm : m = 6) (hn : n = 30)

def distance_AB := 100 -- distance between A and B is 100 kilometers
def time_meeting := 5 -- they meet after 5 hours

noncomputable def speed_A := distance_AB / m
noncomputable def speed_B := distance_AB / n

theorem distance_AC :
  speed_A * time_meeting = 83 + 1 / 3 :=
by
-- All the necessary conditions and given values are encapsulated here,
-- which tie back to proving the distance from A to C (83 1/3 km).
s sorry

end distance_AC_l648_648000


namespace fraction_received_l648_648942

theorem fraction_received (total_money : ℝ) (spent_ratio : ℝ) (spent_amount : ℝ) (remaining_amount : ℝ) (fraction_received : ℝ) :
  total_money = 240 ∧ spent_ratio = 1/5 ∧ spent_amount = spent_ratio * total_money ∧ remaining_amount = 132 ∧ spent_amount + remaining_amount = fraction_received * total_money →
  fraction_received = 3 / 4 :=
by {
  sorry
}

end fraction_received_l648_648942


namespace number_of_green_fish_and_carp_drawn_is_6_l648_648173

-- Definitions/parameters from the problem
def total_fish := 80 + 20 + 40 + 40 + 20
def sample_size := 20
def number_of_green_fish := 20
def number_of_carp := 40
def probability_of_being_drawn := sample_size / total_fish

-- Theorem to prove the combined number of green fish and carp drawn is 6
theorem number_of_green_fish_and_carp_drawn_is_6 :
  (number_of_green_fish + number_of_carp) * probability_of_being_drawn = 6 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_green_fish_and_carp_drawn_is_6_l648_648173


namespace chemistry_vs_geometry_weight_l648_648815

theorem chemistry_vs_geometry_weight :
  ∀ (chemistry_weight geometry_weight : ℝ),
    chemistry_weight = 7.125 ∧ geometry_weight = 0.625 →
    chemistry_weight - geometry_weight = 6.5 :=
begin
  intros chemistry_weight geometry_weight,
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num,
end

end chemistry_vs_geometry_weight_l648_648815


namespace quadratic_ineq_solution_set_l648_648332

theorem quadratic_ineq_solution_set {m : ℝ} :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
sorry

end quadratic_ineq_solution_set_l648_648332


namespace agatha_initial_money_l648_648607

/-
Agatha has some money to spend on a new bike. She spends $15 on the frame, and $25 on the front wheel.
If she has $20 left to spend on a seat and handlebar tape, prove that she had $60 initially.
-/

theorem agatha_initial_money (frame_cost wheel_cost remaining_money initial_money: ℕ) 
  (h1 : frame_cost = 15) 
  (h2 : wheel_cost = 25) 
  (h3 : remaining_money = 20) 
  (h4 : initial_money = frame_cost + wheel_cost + remaining_money) : 
  initial_money = 60 :=
by {
  -- We state explicitly that initial_money should be 60
  sorry
}

end agatha_initial_money_l648_648607


namespace hyperbola_asymptotes_proof_l648_648308

noncomputable def hyperbola_asymptotes {a b : ℝ} (h_a : a > 0) (h_b : b > 0) (h_ecc : 2 * a = real.sqrt (a ^ 2 + b ^ 2)) : Prop :=
  ∀ x, ∀ y, y = (x * sqrt (a^2 + b^2 - a^2)/a) → y = x * sqrt 3 ∨ y = -x * sqrt 3

theorem hyperbola_asymptotes_proof {a b : ℝ} (h_a : a > 0) (h_b : b > 0) (h_ecc : 2 * a = real.sqrt (a ^ 2 + b ^ 2)) : 
  hyperbola_asymptotes h_a h_b h_ecc :=
sorry

end hyperbola_asymptotes_proof_l648_648308


namespace SarahsNumber_is_2880_l648_648416

def SarahsNumber (n : ℕ) : Prop :=
  (144 ∣ n) ∧ (45 ∣ n) ∧ (1000 ≤ n ∧ n ≤ 3000)

theorem SarahsNumber_is_2880 : SarahsNumber 2880 :=
  by
  sorry

end SarahsNumber_is_2880_l648_648416


namespace arithmetic_sqrt_of_16_l648_648428

theorem arithmetic_sqrt_of_16 : sqrt 16 = 4 :=
by 
  sorry

end arithmetic_sqrt_of_16_l648_648428


namespace equilateral_triangle_side_length_correct_l648_648204

noncomputable def equilateral_triangle_side_length (AB CD EF : ℕ) (hAB : AB = 50) (hCD : CD = 50) (hEF : EF = 50) : ℝ :=
  50 * Real.sqrt 3

theorem equilateral_triangle_side_length_correct (AB CD EF : ℕ) (hAB : AB = 50) (hCD : CD = 50) (hEF : EF = 50) :
  equilateral_triangle_side_length AB CD EF hAB hCD hEF = 50 * Real.sqrt 3 :=
by
  unfold equilateral_triangle_side_length
  simp

#eval equilateral_triangle_side_length_correct 50 50 50 rfl rfl rfl

end equilateral_triangle_side_length_correct_l648_648204


namespace cost_of_purchase_l648_648038

theorem cost_of_purchase (x : ℝ) (T_shirt boots shin_guards : ℝ) 
  (h1 : x + T_shirt = 2 * x)
  (h2 : x + boots = 5 * x)
  (h3 : x + shin_guards = 3 * x) :
  x + T_shirt + boots + shin_guards = 8 * x :=
begin
  sorry
end

end cost_of_purchase_l648_648038


namespace volume_ratio_parallelepiped_cylinder_l648_648599

-- Defining the conditions
variables (α β H R d : ℝ)
variable (a : ℝ := d * Real.cos α )
variable (b : ℝ := d * Real.cos β )
variable (V1 : ℝ := a * b * H)
variable (V2 : ℝ := (Math.pi * R^2 * H))
variable (cos_sq_alpha_beta_sum : ℝ := Real.cos α ^ 2 + Real.cos β ^ 2)
variable (R_sq : ℝ := d^2 * cos_sq_alpha_beta_sum / 4)

-- Statement of the theorem we want to prove
theorem volume_ratio_parallelepiped_cylinder :
  (V1 / V2) = 4 * (Real.cos α * Real.cos β) / (Math.pi * cos_sq_alpha_beta_sum) :=
by
  sorry

end volume_ratio_parallelepiped_cylinder_l648_648599


namespace fill_the_remaining_half_tank_l648_648964

noncomputable def F : ℝ := 1 / 25 -- fill rate (tanks per minute)
noncomputable def E : ℝ := 1 / 50 -- empty rate (tanks per minute)
noncomputable def combined_rate : ℝ := F - E

theorem fill_the_remaining_half_tank : 
  (combined_rate * 25 = 1 / 2) :=
begin
  sorry
end

end fill_the_remaining_half_tank_l648_648964


namespace smallest_four_digit_divisible_by_2_3_5_7_11_l648_648671

theorem smallest_four_digit_divisible_by_2_3_5_7_11 :
  ∃ n : ℕ, n = 2310 ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ (n % 11 = 0) ∧ 1000 ≤ n ∧ n < 10000 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_2_3_5_7_11_l648_648671


namespace petya_purchase_cost_l648_648044

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l648_648044


namespace inequality_solution_l648_648887

theorem inequality_solution (x : ℝ) : x + 8 < 4x - 1 ↔ x > 3 :=
begin
  sorry
end

end inequality_solution_l648_648887


namespace next_term_geometric_sequence_l648_648481

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l648_648481


namespace total_sales_february_l648_648194

-- We define the given conditions
def last_year_sales : ℝ := 320
def percent_increase : ℝ := 30
def increase_in_sales : ℝ := (percent_increase / 100) * last_year_sales
def this_year_sales : ℝ := last_year_sales + increase_in_sales

-- We state the theorem to prove the question equals the correct answer given the conditions
theorem total_sales_february : this_year_sales = 416 := by
  sorry

end total_sales_february_l648_648194


namespace domain_m_1_range_m_if_f_R_range_m_increasing_l648_648733

-- Definitions and Conditions for the functions and the domain of the problem
def f (x : ℝ) (m : ℝ) : ℝ := (Real.log (x^2 - m*x - m))^(1 / 2)

-- 1. Prove the domain for m = 1
theorem domain_m_1 :
  (∀ x : ℝ, f x 1 = (Real.log (x^2 - x - 1))^(1 / 2) → 
  (x < (1 - Real.sqrt 5) / 2 ∨ x > (1 + Real.sqrt 5) / 2)) := sorry

-- 2. Prove the range of m if Range of f(x) is ℝ
theorem range_m_if_f_R :
  (∀ m : ℝ, (range (λ x, f x m) = Set.univ(ℝ)) → 
  (m ≤ -4 ∨ m > 0)) := sorry

-- 3. Prove the range of m if f(x) is increasing in (-∞, 1 - sqrt(3))
theorem range_m_increasing :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Iic (1 - Real.sqrt 3) → (∀ ε > 0, 
  f (x + ε) m > f x m)) → 
  (2 - 2 * Real.sqrt 3 ≤ m ∧ m ≤ 2)) := sorry

end domain_m_1_range_m_if_f_R_range_m_increasing_l648_648733


namespace sum_primes_upto_20_l648_648516

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l648_648516


namespace sequence_sum_value_l648_648685

theorem sequence_sum_value :
  (∃ a : ℕ → ℝ, (∀ n, a n > 0 ∧ a n ^ 2 + (1 - n) * a n - n = 0)) →
  (∃ b : ℕ → ℝ, (∀ n, b n = 1 / ((n + 1) * a n))) →
  (∃ T : ℕ → ℝ, (∀ n, T n = ∑ i in Finset.range(n), b i)) →
  T 2016 = 2016 / 2017 :=
by
  sorry

end sequence_sum_value_l648_648685


namespace profit_percentage_l648_648977

/--
A retailer purchases 250 pens at the market price of 200 pens.
If he sells these pens giving a discount of 3.5%, the profit percentage is 20.625%.
-/
theorem profit_percentage
  (market_price_per_pen : ℝ)
  (pens_bought : ℕ)
  (pens_cost : ℕ)
  (discount_percentage : ℝ) :
  pens_bought = 250 ∧ pens_cost = 200 ∧ discount_percentage = 0.035 ∧ market_price_per_pen = 1 →
  let CP := pens_cost * market_price_per_pen in
  let market_price := pens_bought * market_price_per_pen in
  let discount := discount_percentage * market_price in
  let SP := market_price - discount in
  let profit := SP - CP in
  let profit_percent := (profit / CP) * 100 in
  profit_percent = 20.625 :=
by
  intros h
  sorry

end profit_percentage_l648_648977


namespace vector_BC_correct_l648_648748

-- Define the conditions
def vector_AB : ℝ × ℝ := (-3, 2)
def vector_AC : ℝ × ℝ := (1, -2)

-- Define the problem to be proved
theorem vector_BC_correct :
  let vector_BC := (vector_AC.1 - vector_AB.1, vector_AC.2 - vector_AB.2)
  vector_BC = (4, -4) :=
by
  sorry -- The proof is not required, but the structure indicates where it would go

end vector_BC_correct_l648_648748


namespace janet_total_pockets_l648_648366

theorem janet_total_pockets
  (total_dresses : ℕ)
  (dresses_with_pockets : ℕ)
  (dresses_with_2_pockets : ℕ)
  (dresses_with_3_pockets : ℕ)
  (pockets_from_2 : ℕ)
  (pockets_from_3 : ℕ)
  (total_pockets : ℕ)
  (h1 : total_dresses = 24)
  (h2 : dresses_with_pockets = total_dresses / 2)
  (h3 : dresses_with_2_pockets = dresses_with_pockets / 3)
  (h4 : dresses_with_3_pockets = dresses_with_pockets - dresses_with_2_pockets)
  (h5 : pockets_from_2 = 2 * dresses_with_2_pockets)
  (h6 : pockets_from_3 = 3 * dresses_with_3_pockets)
  (h7 : total_pockets = pockets_from_2 + pockets_from_3)
  : total_pockets = 32 := 
by
  sorry

end janet_total_pockets_l648_648366


namespace total_cost_is_eight_times_l648_648014

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l648_648014


namespace ratio_of_w_y_l648_648444

variable (w x y z : ℚ)

theorem ratio_of_w_y (h1 : w / x = 4 / 3)
                     (h2 : y / z = 3 / 2)
                     (h3 : z / x = 1 / 3) :
                     w / y = 8 / 3 := by
  sorry

end ratio_of_w_y_l648_648444


namespace total_cost_is_eight_x_l648_648058

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l648_648058


namespace abs_inequality_solution_l648_648232

theorem abs_inequality_solution (x : ℝ) : |2 * x + 1| - 2 * |x - 1| > 0 ↔ x > 1 / 4 :=
sorry

end abs_inequality_solution_l648_648232


namespace new_percentage_water_is_correct_l648_648180

def initial_volume : ℕ := 120
def initial_percentage_water : ℚ := 20 / 100
def added_water : ℕ := 8

def initial_volume_water : ℚ := initial_percentage_water * initial_volume
def initial_volume_wine : ℚ := initial_volume - initial_volume_water
def new_volume_water : ℚ := initial_volume_water + added_water
def new_total_volume : ℚ := initial_volume + added_water

def calculate_new_percentage_water : ℚ :=
  (new_volume_water / new_total_volume) * 100

theorem new_percentage_water_is_correct :
  calculate_new_percentage_water = 25 := 
by
  sorry

end new_percentage_water_is_correct_l648_648180


namespace compute_m_l648_648223

def alternating_sequence_sum (n : ℕ) : ℕ := 
  (if n % 4 = 0 then (n^2 + (n-1)^2 - (n-2)^2 - (n-3)^2)
   else if n % 4 = 2 then (n^2 + (n+1)^2 - (n-1)^2 - (n-2)^2)
   else 0) -- gives differences (0 outside pairs)

theorem compute_m :
  let M := (Σ k in finset.range 61, alternating_sequence_sum (2 * k)) in
  M = 14520 :=
by 
  let M := (Σ k in finset.range 61, alternating_sequence_sum (2 * k)),
  sorry

end compute_m_l648_648223


namespace total_cost_is_eight_times_short_cost_l648_648028

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l648_648028


namespace trigonometric_comparison_l648_648824

noncomputable def a : ℝ := Real.sin (3 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (2 * Real.pi / 5)
noncomputable def c : ℝ := Real.tan (2 * Real.pi / 5)

theorem trigonometric_comparison :
  b < a ∧ a < c :=
by {
  -- Use necessary steps to demonstrate b < a and a < c
  sorry
}

end trigonometric_comparison_l648_648824


namespace smallest_polynomial_degree_l648_648078

noncomputable def roots := [3 - Real.sqrt 7, 5 + Real.sqrt 11, 17 - 2 * Real.sqrt 10, -Real.sqrt 3]
def polynomial_has_rational_coefficients : Prop := 
  ∃ (p : Polynomial ℚ), roots.all (λ r, p.eval r = 0)

theorem smallest_polynomial_degree
  (h : polynomial_has_rational_coefficients) : 
  ∃ (p : Polynomial ℚ), p.degree = 8 :=
sorry

end smallest_polynomial_degree_l648_648078


namespace prob_of_xi_between_90_and_100_l648_648871

noncomputable def normalDistribution (mean variance : ℝ) := sorry

def ξ := normalDistribution 100 (5^2)

axiom condition1 : ∀ x, P (ξ < x) = sorry

axiom condition2 : P (ξ < 110) = 0.98

theorem prob_of_xi_between_90_and_100 : P (90 < ξ < 100) = 0.48 :=
by
  sorry

end prob_of_xi_between_90_and_100_l648_648871


namespace toll_for_18_wheel_truck_l648_648569

theorem toll_for_18_wheel_truck : 
  ∀ (x t : ℝ), (t = 3.50 + 0.50 * (x - 2)) → 
    (18 = 2 + 4 * (x - 1)) → 
    (t = 5.00) :=
begin
  intros x t ht hx,
  sorry
end

end toll_for_18_wheel_truck_l648_648569


namespace area_difference_l648_648951

open Real

-- Define the radius and side of the triangle
def radius : ℝ := 3
def side_length : ℝ := 9

-- Define the area of the circle
def area_circle : ℝ := π * radius^2

-- Define the area of the equilateral triangle
def area_triangle : ℝ := (sqrt 3 / 4) * side_length^2

-- State the theorem
theorem area_difference :
  (area_triangle - area_circle) = (81 * sqrt 3 / 4 - 9 * π) :=
by sorry

end area_difference_l648_648951


namespace tangency_lines_intersect_at_diagonal_intersection_point_l648_648853

noncomputable def point := Type
noncomputable def line := Type

noncomputable def tangency (C : point) (l : line) : Prop := sorry
noncomputable def circumscribed (Q : point × point × point × point) (C : point) : Prop := sorry
noncomputable def intersects (l1 l2 : line) (P : point) : Prop := sorry
noncomputable def connects_opposite_tangency (Q : point × point × point × point) (l1 l2 : line) : Prop := sorry
noncomputable def diagonals_intersect_at (Q : point × point × point × point) (P : point) : Prop := sorry

theorem tangency_lines_intersect_at_diagonal_intersection_point :
  ∀ (Q : point × point × point × point) (C P : point), 
  circumscribed Q C →
  diagonals_intersect_at Q P →
  ∀ (l1 l2 : line), connects_opposite_tangency Q l1 l2 →
  intersects l1 l2 P :=
sorry

end tangency_lines_intersect_at_diagonal_intersection_point_l648_648853
