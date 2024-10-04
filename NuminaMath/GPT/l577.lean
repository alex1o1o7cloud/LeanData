import Algebra.Parabola
import Mathbin.Combinatorics.Basic
import Mathlib
import Mathlib.Algebra.ArithmeticSeq
import Mathlib.Algebra.Cubic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Analytic.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Integral.IntervalIntegral
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Permutations
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Bool.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.OrderOfElement
import Mathlib.GroupTheory.Subgroup.Defs
import Mathlib.Integral
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Digits
import Mathlib.NumberTheory.Divisors
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import data.nat.basic

namespace polynomial_simplification_l577_577318

theorem polynomial_simplification (y : ℤ) : 
  (2 * y - 1) * (4 * y ^ 10 + 2 * y ^ 9 + 4 * y ^ 8 + 2 * y ^ 7) = 8 * y ^ 11 + 6 * y ^ 9 - 2 * y ^ 7 :=
by 
  sorry

end polynomial_simplification_l577_577318


namespace Mike_limes_picked_l577_577634

theorem Mike_limes_picked (Alyssa_ate : ℝ) (limes_left : ℝ) (total_limes_picked : ℝ) : 
  Alyssa_ate = 25.0 → 
  limes_left = 7.0 → 
  total_limes_picked = Alyssa_ate + limes_left → 
  total_limes_picked = 32 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end Mike_limes_picked_l577_577634


namespace trapezoid_area_is_correct_l577_577336

-- Define the conditions and the problem
def trapezoid_area (AC BD sum_bases : ℝ) : ℝ :=
  if AC = 12 ∧ BD = 6 ∧ sum_bases = 14 then 16 * real.sqrt 5 else 0

-- Statement to prove
theorem trapezoid_area_is_correct :
  trapezoid_area 12 6 14 = 16 * real.sqrt 5 :=
by sorry

end trapezoid_area_is_correct_l577_577336


namespace dot_product_example_l577_577198

theorem dot_product_example : 
  let a := (5, -7)
  let b := (-6, -4)
  a.1 * b.1 + a.2 * b.2 = -2 :=
by
  let a := (5, -7)
  let b := (-6, -4)
  calc
    a.1 * b.1 + a.2 * b.2 = 5 * (-6) + (-7) * (-4) : by rfl
                       ... = -30 + 28            : by rfl
                       ... = -2                  : by rfl

end dot_product_example_l577_577198


namespace figure_can_be_reassembled_into_square_l577_577887

/-
  Problem Statement:
  Given a non-rectangular figure as shown, it can be segmented into three parts such that these parts can be reassembled to form a square.
-/

noncomputable def figure : Type := sorry

def cut1 (f : figure) : figure × figure := sorry
def cut2 (f1 : figure) : figure × figure := sorry

-- The reassemble function takes three parts and arranges them into a square.
def reassemble_to_square (f1 f2 f3 : figure) : Prop := sorry

theorem figure_can_be_reassembled_into_square :
  ∃ (f1 f2 f3 : figure), 
    (f1, f2) = cut1 figure ∧
    (f2, f3) = cut2 f2 ∧ 
    reassemble_to_square f1 f2 f3 := 
sorry

end figure_can_be_reassembled_into_square_l577_577887


namespace Nina_money_before_tax_l577_577295

theorem Nina_money_before_tax :
  ∃ (M P : ℝ), M = 6 * P ∧ M = 8 * 0.9 * P ∧ M = 5 :=
by 
  sorry

end Nina_money_before_tax_l577_577295


namespace triangle_area_ratio_l577_577579

/-
In triangle XYZ, XY=12, YZ=16, and XZ=20. Point D is on XY,
E is on YZ, and F is on XZ. Let XD=p*XY, YE=q*YZ, and ZF=r*XZ,
where p, q, r are positive and satisfy p+q+r=0.9 and p^2+q^2+r^2=0.29.
Prove that the ratio of the area of triangle DEF to the area of triangle XYZ 
can be written in the form m/n where m, n are relatively prime positive 
integers and m+n=137.
-/

theorem triangle_area_ratio :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m + n = 137 ∧ 
  ∃ (p q r : ℝ), p + q + r = 0.9 ∧ p^2 + q^2 + r^2 = 0.29 ∧ 
                  ∀ (XY YZ XZ : ℝ), XY = 12 ∧ YZ = 16 ∧ XZ = 20 → 
                  (1 - (p * (1 - r) + q * (1 - p) + r * (1 - q))) = (37 / 100) :=
by
   sorry

end triangle_area_ratio_l577_577579


namespace Ryan_hours_learning_Spanish_is_4_l577_577470

-- Definitions based on conditions
def hoursLearningChinese : ℕ := 5
def hoursLearningSpanish := ∃ x : ℕ, hoursLearningChinese = x + 1

-- Proof Statement
theorem Ryan_hours_learning_Spanish_is_4 : ∃ x : ℕ, hoursLearningSpanish ∧ x = 4 :=
by
  sorry

end Ryan_hours_learning_Spanish_is_4_l577_577470


namespace simplify_expr_l577_577316

theorem simplify_expr : ((256 : ℝ) ^ (1 / 4)) * ((144 : ℝ) ^ (1 / 2)) = 48 := by
  have h1 : (256 : ℝ) = 2^8 := by
    norm_num,
  have h2 : (144 : ℝ) = 12^2 := by
    norm_num,
  have h3 : (2^8 : ℝ) ^ (1 / 4) = 4 := by
    norm_num,
  have h4 : (12^2 : ℝ) ^ (1 / 2) = 12 := by
    norm_num,
  sorry

end simplify_expr_l577_577316


namespace congruent_triangles_partition_l577_577196

theorem congruent_triangles_partition (T1 T2 : Triangle) 
  (h_congruent : Congruent T1 T2) (h_reflection : IsReflection T1 T2) : 
  (is_right_triangle T1 ∧ is_right_triangle T2 → (∃ (parts : List Triangle), parts.length = 2 ∧ rearrange T1 parts T2)) ∧
  (is_obtuse_triangle T1 ∧ is_obtuse_triangle T2 → (∃ (parts : List Triangle), parts.length = 4 ∧ rearrange T1 parts T2)) ∧
  (is_acute_triangle T1 ∧ is_acute_triangle T2 → (∃ (parts : List Triangle), parts.length = 3 ∧ rearrange T1 parts T2)) :=
by
sorry

end congruent_triangles_partition_l577_577196


namespace skylar_starting_age_l577_577674

-- Conditions of the problem
def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_amount_donated : ℕ := 440000

-- Question and proof statement
theorem skylar_starting_age :
  (current_age - total_amount_donated / annual_donation) = 16 := 
by
  sorry

end skylar_starting_age_l577_577674


namespace area_triangle_OCD_l577_577753

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l577_577753


namespace least_days_to_repay_l577_577257

theorem least_days_to_repay (borrowed_amount : ℕ) (daily_rate : ℚ) (required_multiple : ℕ)
    (h_borrowed_amount : borrowed_amount = 20)
    (h_daily_rate : daily_rate = 0.10) 
    (h_required_multiple : required_multiple = 3) : 
    ∃ x : ℕ, x = 20 ∧ borrowed_amount + x * (daily_rate * borrowed_amount).nat_abs ≥ required_multiple * borrowed_amount :=
by 
  sorry

end least_days_to_repay_l577_577257


namespace find_a_l577_577953

theorem find_a (x y z a : ℝ) (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) (h2 : a > 0) (h3 : ∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + 6 * z^2 = a → (x + y + z) ≤ 1) :
  a = 1 := 
sorry

end find_a_l577_577953


namespace distance_AB_250_l577_577657

variables (A B C D : Point)
variable (distance : Point → Point → ℕ)

-- Conditions as lean definitions
def distance_AC_100_m_from_B : Prop := distance C B = 100
def distance_A_doubles_speed_after_meeting : Prop := distance D A = 50
def persons_meet_C_and_D_points :
  ∀ {A B C D : Point}, distance C B = 100 ∧ distance D A = 50 ∧ C ≠ D

-- The proof goal, proving AB = 250
theorem distance_AB_250 {A B C D : Point} (h1: distance C B = 100)
                      (h2: distance D A = 50)
                      (h3: persons_meet_C_and_D_points):
                    distance A B = 250 :=
sorry

end distance_AB_250_l577_577657


namespace flagpole_height_l577_577838

-- Define the conditions and required values
variables (AB AC AD DE DC: ℝ)

-- Given conditions
noncomputable def problem_conditions : Prop :=
  AC = 4 ∧
  AD = 3 ∧
  DE = 1.8 ∧
  DC = AC - AD

-- Define the question which is the height of the flagpole
noncomputable def height_of_flagpole (AB : ℝ) :=
  (AB / AC) = (DE / DC)

-- Lean 4 proof statement 
theorem flagpole_height : 
  problem_conditions → height_of_flagpole AB := 
by
  intros h
  sorry

end flagpole_height_l577_577838


namespace sum_of_perimeters_of_triangles_l577_577430

theorem sum_of_perimeters_of_triangles (S1 : ℝ) (hS1 : S1 = 50) : 
  let S : ℕ → ℝ := λ n, S1 / 2 ^ (n - 1)
  let P : ℕ → ℝ := λ n, 3 * S n
  let inf_sum := ∑' n, P n
  inf_sum = 300 := 
by
  have hP1 : P 1 = 150 := by sorry
  have h_inf_sum : inf_sum = 2 * P 1 := by sorry
  have h_psum : 2 * P 1 = 300 := by sorry
  exact h_psum

end sum_of_perimeters_of_triangles_l577_577430


namespace mongoliaTST2011_test1_number2_l577_577635

theorem mongoliaTST2011_test1_number2 (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 2) :
    (∑ k in Finset.range (p + 1), (-1 : ℤ)^k * (Nat.choose p k) * (Nat.choose (p + k) k)) % (p^3) = -1 := 
begin
  sorry
end

end mongoliaTST2011_test1_number2_l577_577635


namespace digit_in_thousandths_place_of_7_over_32_is_8_l577_577786

theorem digit_in_thousandths_place_of_7_over_32_is_8 :
  (decimal_expansion_digit (7 / 32) 3) = 8 :=
sorry

end digit_in_thousandths_place_of_7_over_32_is_8_l577_577786


namespace count_safe_numbers_l577_577141

def p_safe (p n : ℕ) : Prop :=
  ∀ m : ℕ, |n - m * p| > 3

def seven_safe (n : ℕ) : Prop := p_safe 7 n
def eleven_safe (n : ℕ) : Prop := p_safe 11 n
def thirteen_safe (n : ℕ) : Prop := p_safe 13 n

theorem count_safe_numbers : 
  {n : ℕ | n ≤ 20000 ∧ seven_safe n ∧ eleven_safe n ∧ thirteen_safe n}.to_finset.card = 1200 := 
sorry

end count_safe_numbers_l577_577141


namespace no_real_solution_arctan_eqn_l577_577675

theorem no_real_solution_arctan_eqn :
  ¬∃ x : ℝ, 0 < x ∧ (Real.arctan (1 / x ^ 2) + Real.arctan (1 / x ^ 4) = (Real.pi / 4)) :=
by
  sorry

end no_real_solution_arctan_eqn_l577_577675


namespace triangle_area_is_64_l577_577769

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l577_577769


namespace angle_between_unit_vectors_l577_577997

-- Definitions from conditions
variables (a b : EuclideanSpace ℝ (Fin 2))
variables (h_unit_a : ‖a‖ = 1) (h_unit_b : ‖b‖ = 1)
variables (h_perp : (2 • a + b) ⬝ b = 0)

-- Theorem to prove the angle between a and b is 2π / 3
theorem angle_between_unit_vectors (a b : EuclideanSpace ℝ (Fin 2))
  (h_unit_a : ‖a‖ = 1) (h_unit_b : ‖b‖ = 1)
  (h_perp : (2 • a + b) ⬝ b = 0) : real.angle a b = 2 * real.pi / 3 :=
sorry

end angle_between_unit_vectors_l577_577997


namespace problem_inequality_l577_577278

variable (a : Fin 2023 → ℝ)
variable (h_pos : ∀ i, 0 < a i)
variable (h_sum : ∑ i, a i^(i.1+1) = 2023)

theorem problem_inequality : 
  ∑ i, a i^(2023 - i.1) > 1 + 1 / 2023 := sorry

end problem_inequality_l577_577278


namespace line_EF_bisects_AC_l577_577490
-- Import the entire Mathlib library to ensure necessary components are present

-- Define the problem statement in Lean 4
theorem line_EF_bisects_AC 
  {A B C K E F : Type} -- Assume generic types for points
  [IsTriangle : Triangle A B C] -- Given there exists a triangle ABC
  (h_right_angle : RightAngle ∠ ACB) -- C is the right angle
  (height_CK : LineSegment C K) -- CK is the height from C to AB
  (angle_bisector_CE : AngleBisector ∠ ACK CE) -- CE is the angle bisector of angle ACK
  (line_BF_parallel_to_CE : Parallel (LineThrough B F) (LineThrough C E)) -- BF is parallel to CE and intersects CK at F
  (inters_F_on_CK : Intersection F (LineSegment C K)) -- F is on CK
  (EF : LineSegment E F) -- Line segment EF
  : Bisects EF A C := 
sorry  -- Proof is omitted, only the statement is required.

end line_EF_bisects_AC_l577_577490


namespace max_sqrt_sum_is_sqrt_30_l577_577532

noncomputable def max_of_sqrt_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h : a + b = 10) : ℝ := max (sqrt (a + 2) + sqrt (b + 3)) sorry

theorem max_sqrt_sum_is_sqrt_30 (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h : a + b = 10) :
  max_of_sqrt_sum a b h₁ h₂ h = sqrt 30 :=
sorry

end max_sqrt_sum_is_sqrt_30_l577_577532


namespace total_invested_amount_l577_577370

theorem total_invested_amount :
  ∃ (A B : ℝ), (A = 3000 ∧ B = 5000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000)
  ∨ 
  (A = 5000 ∧ B = 3000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000) :=
sorry

end total_invested_amount_l577_577370


namespace extra_discount_percentage_l577_577054

theorem extra_discount_percentage 
  (initial_price : ℝ)
  (first_discount : ℝ)
  (new_price : ℝ)
  (final_price : ℝ)
  (extra_discount_amount : ℝ)
  (x : ℝ)
  (discount_formula : x = (extra_discount_amount * 100) / new_price) :
  initial_price = 50 ∧ 
  first_discount = 2.08 ∧ 
  new_price = 47.92 ∧ 
  final_price = 46 ∧ 
  extra_discount_amount = new_price - final_price → 
  x = 4 :=
by
  -- The proof will go here
  sorry

end extra_discount_percentage_l577_577054


namespace max_additional_hours_l577_577661

/-- Define the additional hours of studying given the investments in dorms, food, and parties -/
def additional_hours (a b c : ℝ) : ℝ :=
  5 * a + 3 * b + (11 * c - c^2)

/-- Define the total investment constraint -/
def investment_constraint (a b c : ℝ) : Prop :=
  a + b + c = 5

/-- Prove the maximal additional hours of studying -/
theorem max_additional_hours : ∃ (a b c : ℝ), investment_constraint a b c ∧ additional_hours a b c = 34 :=
by
  sorry

end max_additional_hours_l577_577661


namespace total_cookies_correct_l577_577066

noncomputable def cookies_monday : ℕ := 5
def cookies_tuesday := 2 * cookies_monday
def cookies_wednesday := cookies_tuesday + (40 * cookies_tuesday / 100)
def total_cookies := cookies_monday + cookies_tuesday + cookies_wednesday

theorem total_cookies_correct : total_cookies = 29 := by
  sorry

end total_cookies_correct_l577_577066


namespace blue_hat_cost_is_6_l577_577746

-- Total number of hats is 85
def total_hats : ℕ := 85

-- Number of green hats
def green_hats : ℕ := 20

-- Number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Cost of each green hat
def cost_per_green_hat : ℕ := 7

-- Total cost for all hats
def total_cost : ℕ := 530

-- Total cost of green hats
def total_cost_green_hats : ℕ := green_hats * cost_per_green_hat

-- Total cost of blue hats
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats

-- Cost per blue hat
def cost_per_blue_hat : ℕ := total_cost_blue_hats / blue_hats 

-- Prove that the cost of each blue hat is $6
theorem blue_hat_cost_is_6 : cost_per_blue_hat = 6 :=
by
  sorry

end blue_hat_cost_is_6_l577_577746


namespace incorrect_statement_l577_577807

noncomputable def algorithm_properties (A B C: Prop) :=
  (A = ∀ steps : list (ℕ → ℕ), steps.length < ∞)
  ∧ (B = ∀ step : ℕ → ℕ, clear_executable step)
  ∧ (C = ∀ alg : list (ℕ → ℕ), ∃ result, determinant alg result)

theorem incorrect_statement (A B C D : Prop) (h : algorithm_properties A B C) :
  D = ¬(∀ problem, ∃! algorithm, solves algorithm problem) :=
sorry

end incorrect_statement_l577_577807


namespace maria_needs_nuts_l577_577629

theorem maria_needs_nuts (total_cookies nuts_per_cookie : ℕ) 
  (nuts_fraction : ℚ) (chocolate_fraction : ℚ) 
  (H1 : nuts_fraction = 1 / 4) 
  (H2 : chocolate_fraction = 0.4) 
  (H3 : total_cookies = 60) 
  (H4 : nuts_per_cookie = 2) :
  (total_cookies * nuts_fraction + (total_cookies - total_cookies * nuts_fraction - total_cookies * chocolate_fraction) * nuts_per_cookie) = 72 := 
by
  sorry

end maria_needs_nuts_l577_577629


namespace number_of_valid_sets_l577_577501

def is_odd (n : Nat) : Prop := n % 2 = 1

def is_valid_set (A : Set Nat) : Prop :=
  A ⊆ {2, 3, 9} ∧ ∃ n ∈ A, is_odd n

theorem number_of_valid_sets : {A : Set Nat // is_valid_set A}.card = 6 := by
  sorry

end number_of_valid_sets_l577_577501


namespace number_of_proper_subsets_of_P_l577_577289

theorem number_of_proper_subsets_of_P (P : Set ℝ) (hP : P = {x | x^2 = 1}) : 
  (∃ n, n = 2 ∧ ∃ k, k = 2 ^ n - 1 ∧ k = 3) :=
by
  sorry

end number_of_proper_subsets_of_P_l577_577289


namespace solution_to_problem_l577_577896

def distinct_remainders (n : ℕ) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i ∧ i < n → 1 ≤ j ∧ j < n → i ≠ j → (fact i % n) ≠ (fact j % n)

theorem solution_to_problem : {n : ℕ // n ≥ 2 ∧ distinct_remainders n} = {n | n = 2 ∨ n = 3} :=
  by
    sorry

end solution_to_problem_l577_577896


namespace integral_value_l577_577874

noncomputable def integrate_using_trapezoidal_rule_with_5_parts : ℝ :=
  let f := λ (x : ℝ), 1 / Real.sqrt (x + 4)
  let a : ℝ := 0
  let b : ℝ := 5
  let n : ℝ := 5
  let Δx := (b - a) / n
  let x := λ (i : ℕ), a + i * Δx
  let y := λ i, f (x i)
  (Δx / 2) * (y 0 + 2 * (y 1 + y 2 + y 3 + y 4) + y 5)

theorem integral_value :
  integrate_using_trapezoidal_rule_with_5_parts = 2.0035 :=
sorry

end integral_value_l577_577874


namespace tens_digit_of_M_l577_577605

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

theorem tens_digit_of_M {M : ℕ} (h : 10 ≤ M ∧ M < 100) (h_eq : M = P M + S M + 6) :
  M / 10 = 1 ∨ M / 10 = 2 :=
sorry

end tens_digit_of_M_l577_577605


namespace sin_interval_prob_l577_577514

theorem sin_interval_prob {θ : ℝ} (hθ : 0 ≤ θ ∧ θ ≤ π) : 
  let interval := {θ | 0 ≤ θ ∧ θ ≤ π}
      transformed_interval := {θ | θ > π / 2 ∧ θ ≤ π} in
  (set.countable transformed_interval : ℝ) / (set.countable interval : ℝ) = 1 / 2 := 
    sorry

end sin_interval_prob_l577_577514


namespace determine_f_at_2_l577_577445

noncomputable def monic_quartic_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d e, a = 1 ∧ ∀ x, f(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e

theorem determine_f_at_2 (f : ℝ → ℝ) 
  (hf_monic : monic_quartic_polynomial f)
  (h1 : f (-2) = -4)
  (h2 : f (1) = -1)
  (h3 : f (-3) = -9)
  (h4 : f (5) = -25) : 
  f(2) = -64 := 
sorry

end determine_f_at_2_l577_577445


namespace square_angle_bisector_l577_577150

-- Definition of the square and its vertices
structure Square :=
  (A B C D : Point)
  (is_square : is_square A B C D)

-- Define the main statement
theorem square_angle_bisector (A B C D M N : Point) (s : Square) (hM : M ≠ C) (h1 : M ∈ (segment A D) ∨ M ∈ (segment A C))
  (h2 : ∃ (l : Line), (l ∈ (segment A M)) ∧ ¬(l ∋ C) ∧ (l intersects (segment B C) ∨ l intersects (segment D C))) : 
  (BN_distance A B N + DM_distance D M = AM_distance A M) ∨ (DN_distance D N + BM_distance B M = AM_distance A M) :=
sorry

end square_angle_bisector_l577_577150


namespace sixth_bar_placement_l577_577725

theorem sixth_bar_placement (f : ℕ → ℕ) (h1 : f 1 = 1) (h2 : f 2 = 121) :
  (∃ n, f 6 = n ∧ (n = 16 ∨ n = 46 ∨ n = 76 ∨ n = 106)) :=
sorry

end sixth_bar_placement_l577_577725


namespace trapezoid_in_27_gon_l577_577311

theorem trapezoid_in_27_gon (S : Finset (EuclideanSpace ℝ (Fin 2))) 
  (hS : ∃ P : Finset (EuclideanSpace ℝ (Fin 2)), P.card = 27 ∧ S ⊆ P ∧ S.card = 7)
  (regular : ∃ (C : submodule ℝ (EuclideanSpace ℝ (Fin 2)))), 
    ∀ p ∈ S, geometry_is_regular_gon P 27 C):
  ∃ (A B C D : EuclideanSpace ℝ (Fin 2)), 
    A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ quadrilateral_is_trapezoid A B C D :=
by
  sorry

end trapezoid_in_27_gon_l577_577311


namespace unique_12_tuple_l577_577135

theorem unique_12_tuple :
  ∃! (t : Fin 12 → ℝ), 
    let x := fun (i : Fin 12) => t i in
    (1 - x 0)^2 + ∑ i in Fin.range 11, 2 * (x i - x (i + 1))^2 + x 11^2 = 1/13 :=
begin
  sorry
end

end unique_12_tuple_l577_577135


namespace total_marks_of_all_candidates_l577_577327

theorem total_marks_of_all_candidates 
  (average_marks : ℕ) 
  (num_candidates : ℕ) 
  (average : average_marks = 35) 
  (candidates : num_candidates = 120) : 
  average_marks * num_candidates = 4200 :=
by
  -- The proof will be written here
  sorry

end total_marks_of_all_candidates_l577_577327


namespace trader_loss_percent_l577_577545

/-- Given the selling prices of two cars and the percentage gain and loss on each, prove that the 
overall loss percent on the whole transaction is approximately 1.438%. -/
theorem trader_loss_percent 
  (SP1 SP2 : ℝ) 
  (gain_percentage loss_percentage : ℝ) 
  (CP1 CP2 TCP TSP : ℝ) 
  (profit_or_loss_percent : ℝ) 
  (hSP1 : SP1 = 325475) 
  (hSP2 : SP2 = 325475) 
  (h_gain_percentage : gain_percentage = 0.12)
  (h_loss_percentage : loss_percentage = 0.12)
  (hCP1 : CP1 = SP1 / (1 + gain_percentage)) 
  (hCP2 : CP2 = SP2 / (1 - loss_percentage)) 
  (hTCP : TCP = CP1 + CP2) 
  (hTSP : TSP = SP1 + SP2) 
  (h_profit_or_loss_percent : profit_or_loss_percent = ((TSP - TCP) / TCP) * 100) : 
  profit_or_loss_percent ≈ -1.438 := 
by {
  sorry
}

end trader_loss_percent_l577_577545


namespace yard_length_l577_577428

theorem yard_length (n : ℕ) (d : ℕ) (trees : ℕ) : 
  n = 23 → d = 18 → trees = 24 → (n * d) + d = 414 :=
by
  intros hn hd htrees
  rw [hn, hd, htrees]
  sorry

end yard_length_l577_577428


namespace role_assignment_count_l577_577029

theorem role_assignment_count :
  let men := 6
  let women := 7
  let male_role := men.choose 1 * (men - 1).choose 0
  let female_role := women.choose 1 * (women - 1).choose 0
  let remaining := 11 // i.e., (men + women) - 2
  let either_gender_roles := remaining.choose 4 * Nat.factorial 4
  let total_assignments := male_role * female_role * either_gender_roles
  total_assignments = 33120 :=
by
  sorry

end role_assignment_count_l577_577029


namespace digit_in_thousandths_place_of_7_over_32_is_8_l577_577785

theorem digit_in_thousandths_place_of_7_over_32_is_8 :
  (decimal_expansion_digit (7 / 32) 3) = 8 :=
sorry

end digit_in_thousandths_place_of_7_over_32_is_8_l577_577785


namespace triangle_incenter_perpendicular_l577_577390

open_locale real_inner_product_space

variables {A B C I D : Type*} [metric_space A] 
  [metric_space B] [metric_space C] [metric_space I] [metric_space D]
  [normed_add_torsor ℝ A] [normed_add_torsor ℝ B]
  [normed_add_torsor ℝ C] [normed_add_torsor ℝ I] [normed_add_torsor ℝ D]
  {angle_A : real_angle} {angle_B : real_angle} {angle_C : real_angle} 
  {angle_I : real_angle} {angle_D : real_angle}

instance : is_triangle A B C := sorry
instance : metric_space (triangle A B C) := sorry

constant incenter : triangle A B C → point I
constant incircle : triangle A B C → metric_sphere C real_radius

def angle_equiv : angle = 30 := sorry
def metric_is_right_angle : dist C B = dist C A := sorry
def incircle_intersection (B : line_segment B I) (I : incenter) : incircle = D :=
  sorry

theorem triangle_incenter_perpendicular 
  (h1: incenter (triangle A B C) = I)
  (h2: dist C B = dist C A)
  (h3: angle BAC = 30)
  (h4: incircle_intersection (line_segment B I) I = D) : 
  ⊥ (line_segment A I) (line_segment C D) :=
sorry

end triangle_incenter_perpendicular_l577_577390


namespace distinct_real_numbers_inequality_l577_577621

theorem distinct_real_numbers_inequality
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ( (2 * a - b) / (a - b) )^2 + ( (2 * b - c) / (b - c) )^2 + ( (2 * c - a) / (c - a) )^2 ≥ 5 :=
by {
    sorry
}

end distinct_real_numbers_inequality_l577_577621


namespace remainder_product_l577_577291

theorem remainder_product (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := 
by 
  sorry

end remainder_product_l577_577291


namespace solve_trigonometric_sum_eq_l577_577676

theorem solve_trigonometric_sum_eq (x : ℝ) :
  (\sum k in (finset.range 1007), real.sin ((2 * k + 1) * x)) =
  (\sum k in (finset.range 1007), real.cos ((2 * k + 1) * x)) →
  (∃ k : ℤ, x = ↑k * real.pi / 1007 ∧ k % 1007 ≠ 0) ∨
  (∃ k : ℤ, x = (real.pi + 4 * ↑k * real.pi) / 4028) :=
by
  sorry

end solve_trigonometric_sum_eq_l577_577676


namespace perpendicular_line_plane_l577_577518

variable (a : ℝ × ℝ × ℝ) (m : ℝ × ℝ × ℝ)

def direction_vector (t : ℝ) : ℝ × ℝ × ℝ := (-2, 1, t)
def normal_vector : ℝ × ℝ × ℝ := (4, -2, -2)

theorem perpendicular_line_plane (t : ℝ) (h : direction_vector t = normal_vector) : t = 1 := 
by {
  unfold direction_vector at h,
  unfold normal_vector at h,
  sorry
}

end perpendicular_line_plane_l577_577518


namespace pascal_triangle_seventh_element_l577_577231

noncomputable def binom : ℕ → ℕ → ℕ
| n, k := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem pascal_triangle_seventh_element :
  binom 20 6 = 38760 ∧ (binom 20 6) = 204 * (binom 20 2) := by
  have h1 : binom 20 6 = 38760 := sorry
  have h2 : (binom 20 6) = 204 * (binom 20 2) := sorry
  exact ⟨h1, h2⟩

end pascal_triangle_seventh_element_l577_577231


namespace excircle_selection_count_l577_577261

theorem excircle_selection_count :
  ∃ (a b c : ℕ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (-a + b + c) % 2 = 0 ∧ (-a + b + c) ≤ 100 ∧
    (a - b + c) % 2 = 0 ∧ (a - b + c) ≤ 100 ∧
    (a + b - c) % 2 = 0 ∧ (a + b - c) ≤ 100 ∧
    (let m := (a + a) / 2; n := (b + b) / 2; o := (c + c) / 2;
       -- Midpoints properties, assuming geom condition) in
    true ∧ -- Assuming geometric condition holds
    sorry -- Proof of the number of ways to select (a, b, c)
proof_size = 125000 := sorry

end excircle_selection_count_l577_577261


namespace fruit_seller_original_apples_l577_577020

variable (x : ℝ)

theorem fruit_seller_original_apples (h : 0.60 * x = 420) : x = 700 := by
  sorry

end fruit_seller_original_apples_l577_577020


namespace S_is_infinite_l577_577274

-- Define the set S and the condition that every point in S is the midpoint of two other points in S
variables {Point : Type*} [inhabited Point]

def is_midpoint (P A B : Point) [has_add Point] [has_scalar ℕ Point] : Prop :=
  P = (A + B) / 2

def is_midpoint_of_two_others (S : set Point) [inhabited S] : Prop :=
  ∀ P ∈ S, ∃ A B ∈ S, is_midpoint P A B

-- State the theorem that S is infinite under the given conditions
theorem S_is_infinite (S : set Point) [inhabited S] [has_add Point] [has_scalar ℕ Point]
  (h : is_midpoint_of_two_others S) : ∃ (inf : S.infinite), true :=
sorry

end S_is_infinite_l577_577274


namespace sum_log2_floor_l577_577089

theorem sum_log2_floor (N : ℕ) (hN : 1 ≤ N ∧ N ≤ 2048) :
  ∑ N in finset.range 2048, nat.log N = 6157 := sorry

end sum_log2_floor_l577_577089


namespace rate_of_current_l577_577353

theorem rate_of_current (S_boat D : ℚ) (T : ℚ) : 
  S_boat = 20 → D = 8.75 → T = 21 → 
  ∃ c : ℚ, c ≈ 5 := 
by
  intros h1 h2 h3
  let T_h := T / 60
  have eqn1 : D = (S_boat + c) * T_h := sorry
  sorry

end rate_of_current_l577_577353


namespace sara_quarters_eq_l577_577669

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 21
def dad_quarters : ℕ := 49
def spent_quarters : ℕ := 15
def mom_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4
def amy_quarters (x : ℕ) := x

-- Define the function to compute total quarters
noncomputable def total_quarters (x : ℕ) : ℕ :=
initial_quarters + dad_quarters - spent_quarters + mom_dollars * quarters_per_dollar + amy_quarters x

-- Prove that the total number of quarters matches the expected value
theorem sara_quarters_eq (x : ℕ) : total_quarters x = 63 + x :=
by
  sorry

end sara_quarters_eq_l577_577669


namespace area_of_triangle_l577_577761

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l577_577761


namespace expression_evaluates_to_0_181_l577_577392

def expr : ℝ := (0.5 ^ 3) - (0.1 ^ 3) / (0.5 ^ 2) + 0.05 + (0.1 ^ 2)

theorem expression_evaluates_to_0_181 : expr = 0.181 :=
by
  sorry

end expression_evaluates_to_0_181_l577_577392


namespace probability_diana_larger_than_apollo_l577_577455

theorem probability_diana_larger_than_apollo :
  (∃ (d a : ℕ), (1 ≤ d ∧ d ≤ 8) ∧ (1 ≤ a ∧ a ≤ 6) ∧ d > a) →
  (card { p : ℕ × ℕ // (1 ≤ p.1 ∧ p.1 ≤ 8) ∧ (1 ≤ p.2 ∧ p.2 ≤ 6) ∧ p.1 > p.2 } / 48 = 9 / 16) :=
by
  sorry

end probability_diana_larger_than_apollo_l577_577455


namespace arithmetic_question_l577_577437

theorem arithmetic_question :
  ((3.25 - 1.57) * 2) = 3.36 :=
by 
  sorry

end arithmetic_question_l577_577437


namespace jellybean_count_l577_577731

variable (initial_count removed1 added_back removed2 : ℕ)

theorem jellybean_count :
  initial_count = 37 →
  removed1 = 15 →
  added_back = 5 →
  removed2 = 4 →
  initial_count - removed1 + added_back - removed2 = 23 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jellybean_count_l577_577731


namespace knights_count_is_two_l577_577643

def inhabitant (i : Fin 5) : Prop :=
  (i = 0 → inhabitants_truth 1) ∧
  (i = 1 → inhabitants_truth 2) ∧
  (i = 2 → inhabitants_truth 3) ∧
  (i = 3 → ∀ k, k ≠ 3 → ¬inhabitants_truth k) ∧
  (i = 4 → inhabitants_truth 4 ≠ inhabitants_truth 3)

def inhabitants_truth (n : ℕ) : Prop := sorry

theorem knights_count_is_two : ∃ (knights : Fin 5 → Prop), (∃! i, knights i) ∧ (∃! j, knights j) ∧ (i ≠ j) :=
sorry

end knights_count_is_two_l577_577643


namespace simplify_fraction_l577_577317

theorem simplify_fraction : (270 / 18) * (7 / 140) * (9 / 4) = 27 / 16 :=
by sorry

end simplify_fraction_l577_577317


namespace find_a6_l577_577947

variable (a : ℕ → ℝ)

-- condition: a_2 + a_8 = 16
axiom h1 : a 2 + a 8 = 16

-- condition: a_4 = 1
axiom h2 : a 4 = 1

-- question: Prove that a_6 = 15
theorem find_a6 : a 6 = 15 :=
sorry

end find_a6_l577_577947


namespace math_problem_l577_577175

-- Condition 1: The solution set of the inequality \(\frac{x-2}{ax+b} > 0\) is \((-1,2)\)
def solution_set_condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x > -1 ∧ x < 2) ↔ ((x - 2) * (a * x + b) > 0)

-- Condition 2: \(m\) is the geometric mean of \(a\) and \(b\)
def geometric_mean_condition (a b m : ℝ) : Prop :=
  a * b = m^2

-- The mathematical statement to prove: \(\frac{3m^{2}a}{a^{3}+2b^{3}} = 1\)
theorem math_problem (a b m : ℝ) (h1 : solution_set_condition a b) (h2 : geometric_mean_condition a b m) :
  3 * m^2 * a / (a^3 + 2 * b^3) = 1 :=
sorry

end math_problem_l577_577175


namespace f_is_odd_and_increasing_l577_577284

def f (x : ℝ) : ℝ := x - Real.sin x

theorem f_is_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x ≤ y → f x ≤ f y) := 
by
  sorry

end f_is_odd_and_increasing_l577_577284


namespace sin_beta_correct_l577_577491

noncomputable def sin_beta : Prop :=
  ∀ (α β : ℝ), 
    0 < α ∧ α < π / 2 ∧
    -π / 2 < β ∧ β < 0 ∧
    cos (α - β) = -3 / 5 ∧ 
    tan α = 4 / 3 → 
  sin β = -24 / 25

theorem sin_beta_correct : sin_beta := 
by
  sorry

end sin_beta_correct_l577_577491


namespace number_of_knights_l577_577655

-- Definition of knight and liar
inductive Inhabitant
| knight : Inhabitant
| liar : Inhabitant

open Inhabitant

-- Statements of the inhabitants
def statement_1 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i1 with
  | knight => 1
  | liar => true

def statement_2 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i2 with
  | knight => 2
  | liar => true

def statement_3 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i3 with
  | knight => 3
  | liar => true

def statement_4 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  (i1 = liar ∧ i2 = liar ∧ i3 = liar ∧ i4 = liar ∧ i5 = liar)

def statement_5 (i4 : Inhabitant) : Prop := 
  match i5 with
  | knight => i4 = liar
  | liar => true

-- Problem statement in Lean 4
theorem number_of_knights (i1 i2 i3 i4 i5 : Inhabitant) 
  (h1 : statement_1 i1 i2 i3 i4 i5)
  (h2 : statement_2 i1 i2 i3 i4 i5)
  (h3 : statement_3 i1 i2 i3 i4 i5)
  (h4 : statement_4 i1 i2 i3 i4 i5)
  (h5 : statement_5 i4) :
  (i1 = knight ∨ i2 = knight ∨ i3 = knight ∨ i4 = knight ∨ i5 = knight) → 
  (i1 = knight ∧ i2 = knight ∧ i3 = knight ∧ i4 = knight ∧ i5 = knight) → 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end number_of_knights_l577_577655


namespace largest_divisor_of_n_squared_sub_n_squared_l577_577112

theorem largest_divisor_of_n_squared_sub_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n_squared_sub_n_squared_l577_577112


namespace earnings_yesterday_correct_minimum_selling_price_cabbage_today_correct_l577_577373

-- Definitions from given conditions
def purchase_price_cabbage := 2.8
def purchase_price_broccoli := 3.2
def selling_price_cabbage_yesterday := 4.0
def selling_price_broccoli_yesterday := 4.5
def total_weight := 200
def total_cost := 600
def weight_cabbage := 100  -- from solution part
def weight_broccoli := 100 -- from solution part
def earnings_yesterday := 250

def damaged_fraction := 0.1
def selling_price_broccoli := selling_price_broccoli_yesterday  -- unchanged price
def minimum_earnings := earnings_yesterday

-- Placeholder for today’s selling price of cabbage to be calculated
noncomputable def selling_price_cabbage_today : ℝ := sorry

-- Proofs
theorem earnings_yesterday_correct :
  (selling_price_cabbage_yesterday - purchase_price_cabbage) * weight_cabbage + 
  (selling_price_broccoli_yesterday - purchase_price_broccoli) * weight_broccoli = earnings_yesterday := by
  -- Proof goes here, skipped
  sorry

theorem minimum_selling_price_cabbage_today_correct :
  ∃ (selling_price_cabbage_today : ℝ), 
  (selling_price_cabbage_today - purchase_price_cabbage) * ((1.0 - damaged_fraction) * weight_cabbage) + 
  (selling_price_broccoli - purchase_price_broccoli) * weight_broccoli ≥ minimum_earnings
  ∧ selling_price_cabbage_today ≥ 4.1 := by
  -- Proof goes here, skipped
  sorry

end earnings_yesterday_correct_minimum_selling_price_cabbage_today_correct_l577_577373


namespace digit_in_thousandths_place_of_7_over_32_is_8_l577_577787

theorem digit_in_thousandths_place_of_7_over_32_is_8 :
  (decimal_expansion_digit (7 / 32) 3) = 8 :=
sorry

end digit_in_thousandths_place_of_7_over_32_is_8_l577_577787


namespace matrix_multiplication_correct_l577_577881

def mat1 : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [2, -1, 3],
      [1, 3, -2],
      [-2, 3, 2] ]

def mat2 : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [1, -3, 0],
      [0, 2, -3],
      [5, 1, 0] ]

def mat_product : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [17, -5, 3],
      [-9, 1, -9],
      [8, 14, -9] ]

theorem matrix_multiplication_correct : mat1 ⬝ mat2 = mat_product :=
by
  sorry

end matrix_multiplication_correct_l577_577881


namespace log_expression_eval_f_periodic_and_odd_l577_577827

-- Statement for the first part of the problem
theorem log_expression_eval : 
  real.log 4 + 2 * real.log 5 + (0.25 : ℝ)^(-1/2) - 8^(2/3) = 0 := 
sorry

-- Definitions and theorem for the second part of the problem
def f (x : ℝ) : ℝ :=
if x ∈ set.Ioo 0 2 then 2 * x^2 else 
if x ∈ set.Ioo (-2) 0 then -2 * (-x)^2 else 0

theorem f_periodic_and_odd (x : ℝ) : 
  (f x = 2 * x^2 ∧ x ∈ set.Ioo 0 2) → 
  (f (x + 2) = -f x) → 
  (f (-x) = -f x) → 
  f 2015 = -2 := 
sorry

end log_expression_eval_f_periodic_and_odd_l577_577827


namespace number_of_knights_l577_577653

-- Definition of knight and liar
inductive Inhabitant
| knight : Inhabitant
| liar : Inhabitant

open Inhabitant

-- Statements of the inhabitants
def statement_1 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i1 with
  | knight => 1
  | liar => true

def statement_2 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i2 with
  | knight => 2
  | liar => true

def statement_3 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i3 with
  | knight => 3
  | liar => true

def statement_4 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  (i1 = liar ∧ i2 = liar ∧ i3 = liar ∧ i4 = liar ∧ i5 = liar)

def statement_5 (i4 : Inhabitant) : Prop := 
  match i5 with
  | knight => i4 = liar
  | liar => true

-- Problem statement in Lean 4
theorem number_of_knights (i1 i2 i3 i4 i5 : Inhabitant) 
  (h1 : statement_1 i1 i2 i3 i4 i5)
  (h2 : statement_2 i1 i2 i3 i4 i5)
  (h3 : statement_3 i1 i2 i3 i4 i5)
  (h4 : statement_4 i1 i2 i3 i4 i5)
  (h5 : statement_5 i4) :
  (i1 = knight ∨ i2 = knight ∨ i3 = knight ∨ i4 = knight ∨ i5 = knight) → 
  (i1 = knight ∧ i2 = knight ∧ i3 = knight ∧ i4 = knight ∧ i5 = knight) → 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end number_of_knights_l577_577653


namespace minimum_beacons_unique_determination_l577_577249

-- Define the type representing Rooms and Beacons
structure Maze :=
  (rooms : Type)
  (corridors : rooms → rooms → Prop)
  (beacons : rooms → Prop)
  (signal_distance : rooms → rooms → ℕ)

-- Define the conditions
axiom room_A1 : Maze.rooms
axiom room_B3 : Maze.rooms
axiom room_D4 : Maze.rooms
axiom signal_from_A1 : Maze.signal_distance room_A1
axiom signal_from_B3 : Maze.signal_distance room_B3
axiom signal_from_D4 : Maze.signal_distance room_D4

-- Theorem: Minimum number of beacons for unique room determination is 3
theorem minimum_beacons_unique_determination (M : Maze) : 
  (∃ r₁, M.beacons r₁ ∧ M.signal_distance r₁ = signal_from_A1) ∧
  (∃ r₂, M.beacons r₂ ∧ M.signal_distance r₂ = signal_from_B3) ∧
  (∃ r₃, M.beacons r₃ ∧ M.signal_distance r₃ = signal_from_D4) → 
  (∀ (r : Maze.rooms), ∃! (r₁ r₂ r₃ : Maze.rooms), 
    M.signal_distance r r₁ ≠ M.signal_distance r r₂ ∨ 
    M.signal_distance r r₁ ≠ M.signal_distance r r₃ ∨ 
    M.signal_distance r r₂ ≠ M.signal_distance r r₃ ) :=
begin
  sorry
end

end minimum_beacons_unique_determination_l577_577249


namespace jeanne_should_buy_more_tickets_l577_577586

theorem jeanne_should_buy_more_tickets :
  let cost_ferris_wheel := 5
  let cost_roller_coaster := 4
  let cost_bumper_cars := 4
  let jeanne_current_tickets := 5
  let total_tickets_needed := cost_ferris_wheel + cost_roller_coaster + cost_bumper_cars
  let tickets_needed_to_buy := total_tickets_needed - jeanne_current_tickets
  tickets_needed_to_buy = 8 :=
by
  sorry

end jeanne_should_buy_more_tickets_l577_577586


namespace area_of_triangle_bounded_by_lines_l577_577774

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l577_577774


namespace sahil_selling_price_l577_577668

noncomputable def exchange_rate_eur_to_usd : ℝ := 1 / 0.85
noncomputable def exchange_rate_gbp_to_usd : ℝ := 1 / 0.75

def purchase_price_usd : ℝ := 12000
noncomputable def repair_costs_usd : ℝ := 4000 * exchange_rate_eur_to_usd
noncomputable def transportation_costs_usd : ℝ := 1000 * exchange_rate_gbp_to_usd

noncomputable def total_expenses_usd : ℝ := purchase_price_usd + repair_costs_usd + transportation_costs_usd
def profit_margin : ℝ := 0.50
noncomputable def profit_usd : ℝ := profit_margin * total_expenses_usd
noncomputable def selling_price_usd : ℝ := total_expenses_usd + profit_usd

theorem sahil_selling_price : selling_price_usd = 27058.82 := by
  have h_repair_costs_usd : repair_costs_usd = 4705.88 := by sorry
  have h_transportation_costs_usd : transportation_costs_usd = 1333.33 := by sorry
  have h_total_expenses_usd : total_expenses_usd = 18039.21 := by sorry
  have h_profit_usd : profit_usd = 9019.61 := by sorry
  show selling_price_usd = 27058.82 from sorry

end sahil_selling_price_l577_577668


namespace cortney_downloads_all_files_in_2_hours_l577_577105

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l577_577105


namespace triangle_area_l577_577226

noncomputable def is_orthocenter (O A B : Point) (H : Point) : Prop :=
  ... -- Definition of orthocenter
  
noncomputable def parabola (a : ℝ) : (ℝ × ℝ) → Prop :=
  λ (x y : ℝ), y^2 = 4 * a * x

namespace Triangle
  
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (O : Point)
  (A : Point)
  (B : Point)

def area (Δ : Triangle) : ℝ :=
  0.5 * abs (Δ.A.x * Δ.B.y + Δ.B.x * Δ.O.y + Δ.O.x * Δ.A.y - Δ.A.y * Δ.B.x - Δ.O.y * Δ.A.x - Δ.O.x * Δ.B.y)

end Triangle

open Triangle

theorem triangle_area :
  ∀ (O A B : Point),
    is_orthocenter O A B (Point.mk 1 0) →
    parabola 1 (A.x, A.y) →
    parabola 1 (B.x, B.y) →
    O = (Point.mk 0 0) →
    area
      ⟨O, A, B⟩ = 10 * real.sqrt 5 :=
by
  sorry

end triangle_area_l577_577226


namespace problem1_problem2_l577_577005

-- Problem (1)
variable {f g : ℝ → ℝ}
variable (hf6 : f 6 = 5) (hg6 : g 6 = 4)
variable (hf'6 : deriv f 6 = 3) (hg'6 : deriv g 6 = 1)
def h (x : ℝ) : ℝ := f x * g x - 1

theorem problem1 : deriv h 6 = 17 :=
by 
  sorry

-- Problem (2)
def f2 (x : ℝ) : ℝ := Real.sin x
def tangent_line (x : ℝ) (b : ℝ) : ℝ := 1/2 * x + b

theorem problem2 (k : ℤ) (b : ℝ) : 
            (tangent_line (2 * Real.pi * (k : ℝ) + Real.pi / 3) b = f2 (2 * Real.pi * (k : ℝ) + Real.pi / 3)) ∧ 
            (tangent_line (2 * Real.pi * (k : ℝ) - Real.pi / 3) b = f2 (2 * Real.pi * (k : ℝ) - Real.pi / 3)) :=
by 
  sorry

end problem1_problem2_l577_577005


namespace car_b_speed_l577_577092

theorem car_b_speed :
  (∃ x : ℕ, (50 * 6 - 6 * x = 80 * 2 - 2 * x) ∧ x = 35) :=
by {
  let x := 35,
  use x,
  split,
  {
    -- Condition created from the first scenario
    have h1 : 50 * 6 = 300 := by norm_num,
    have h2 : 6 * x = 6 * 35 := by norm_num,
    have h3 : 300 - 6 * x = 300 - 6 * 35 := by rw [h2],
    have eq1 : 300 - 6 * 35 = 300 - 210 := by norm_num,
    -- Condition created from the second scenario
    have h4 : 80 * 2 = 160 := by norm_num,
    have h5 : 2 * x = 2 * 35 := by norm_num,
    have h6 : 160 - 2 * x = 160 - 2 * 35 := by rw [h5],
    have eq2 : 160 - 2 * 35 = 160 - 70 := by norm_num,
    -- Equate the two conditions
    rw [eq1, eq2],
    norm_num,
  },
  -- the solution step
  norm_num,
}

end car_b_speed_l577_577092


namespace max_difference_intersection_ycoords_l577_577132

theorem max_difference_intersection_ycoords :
  let f₁ (x : ℝ) := 5 - 2 * x^2 + x^3
  let f₂ (x : ℝ) := 1 + x^2 + x^3
  let x1 := (2 : ℝ) / Real.sqrt 3
  let x2 := - (2 : ℝ) / Real.sqrt 3
  let y1 := f₁ x1
  let y2 := f₂ x2
  (f₁ = f₂)
  → abs (y1 - y2) = (16 * Real.sqrt 3 / 9) :=
by
  sorry

end max_difference_intersection_ycoords_l577_577132


namespace distance_between_lines_l577_577529

-- Definitions of the lines
def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y - 1 = 0

-- The distance between the two parallel lines
theorem distance_between_lines : 
  let l1 := line1 
  let l2 := line2 
  ∀ (a b : ℝ), 
    l1 a b → l2 a b → 
    ∃ (d : ℝ), d = real.sqrt 2 :=
  by
  intros a b l1_def l2_def
  use real.sqrt 2
  sorry

end distance_between_lines_l577_577529


namespace area_of_triangle_bounded_by_lines_l577_577776

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l577_577776


namespace ellipse_foci_and_eccentricity_and_hyperbola_l577_577967

-- Define the ellipse
def ellipse_eq (x y : ℝ) := 3 * x^2 + y^2 = 18

noncomputable def foci_of_ellipse : ℝ × ℝ := (0, 2)

def eccentricity_of_ellipse : ℝ := (6 : ℝ).sqrt / 3

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) := y^2 / 6 - x^2 / 12 = 1

theorem ellipse_foci_and_eccentricity_and_hyperbola :
  (∀ x y : ℝ, ellipse_eq x y → (x, y) = foci_of_ellipse) ∧
  (∀ x y : ℝ, ellipse_eq x y → eccentricity_of_ellipse = (6 : ℝ).sqrt / 3) ∧
  (∀ x y : ℝ, hyperbola_eq x y) :=
by
  sorry

end ellipse_foci_and_eccentricity_and_hyperbola_l577_577967


namespace count_ints_in_interval_l577_577712

theorem count_ints_in_interval : 
  let S := {x : ℕ | 5 < Real.sqrt x ∧ Real.sqrt x < 6},
  let T := {x : ℕ | 25 < x ∧ x < 36}
  in S = T ∧ ∃ (n : ℕ), n = 10 ∧ ∀ x ∈ T, x ∈ S :=
by
  sorry

end count_ints_in_interval_l577_577712


namespace rational_function_identity_l577_577981

theorem rational_function_identity
  (r s : ℤ[X])
  (h_r_quad : degree r = 2)
  (h_s_cubic : degree s = 3)
  (h_r_value : r.eval 2 = 2)
  (h_s_value : s.eval (-1) = 3)
  (h_asymptote : ∃ a k : ℤ, s = X * (X - 3)^2 * (X - k))
  (h_hole : ∃ b h : ℤ, r = b * (X - 1) * (X - h)) :
  r + s = X^3 + (49/3) * X - 48 := by
  sorry

end rational_function_identity_l577_577981


namespace download_time_l577_577102

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l577_577102


namespace correct_profit_equation_l577_577024

def total_rooms : ℕ := 50
def initial_price : ℕ := 180
def price_increase_step : ℕ := 10
def cost_per_occupied_room : ℕ := 20
def desired_profit : ℕ := 10890

theorem correct_profit_equation (x : ℕ) : 
  (x - cost_per_occupied_room : ℤ) * (total_rooms - (x - initial_price : ℤ) / price_increase_step) = desired_profit :=
by sorry

end correct_profit_equation_l577_577024


namespace solution_l577_577534

noncomputable def math_problem (a b : ℝ) : ℝ :=
if {1, a, b / a} = {0, a^2, a + b} ∧ a ≠ 0 then a ^ 2016 + b ^ 2016 else 0

theorem solution (a b : ℝ) (h1 : {1, a, b / a} = {0, a^2, a + b}) (h2 : a ≠ 0) : 
  math_problem a b = 1 := 
by
  sorry

end solution_l577_577534


namespace jellybeans_final_count_l577_577729

-- Defining the initial number of jellybeans and operations
def initial_jellybeans : ℕ := 37
def removed_first : ℕ := 15
def added_back : ℕ := 5
def removed_second : ℕ := 4

-- Defining the final number of jellybeans to prove it equals 23
def final_jellybeans : ℕ := (initial_jellybeans - removed_first) + added_back - removed_second

-- The theorem that states the final number of jellybeans is 23
theorem jellybeans_final_count : final_jellybeans = 23 :=
by
  -- The proof will be provided here if needed
  sorry

end jellybeans_final_count_l577_577729


namespace triangle_area_is_64_l577_577766

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l577_577766


namespace product_of_b_values_is_neg_12_l577_577697

theorem product_of_b_values_is_neg_12 (b : ℝ) (y1 y2 x1 : ℝ) (h1 : y1 = 3) (h2 : y2 = 7) (h3 : x1 = 2) (h4 : y2 - y1 = 4) (h5 : ∃ b1 b2, b1 = x1 - 4 ∧ b2 = x1 + 4) : 
  (b1 * b2 = -12) :=
by
  sorry

end product_of_b_values_is_neg_12_l577_577697


namespace no_positive_integer_solutions_to_equation_l577_577123

theorem no_positive_integer_solutions_to_equation (p m n k : ℕ) (hp : nat.prime p) (hodd : p % 2 = 1) 
  (hnm1 : n ≤ m) (hnm2 : m ≤ 3 * n) : ¬ (p^m + p^n + 1 = k^2) :=
sorry

end no_positive_integer_solutions_to_equation_l577_577123


namespace exists_unique_nonzero_integer_m_l577_577488

noncomputable def A : set (ℕ+ × ℤ) := { p | p.snd = -3 * (p.fst : ℤ) + 2 }

noncomputable def B (m : ℤ) : set (ℕ+ × ℤ) := { p | p.snd = m * (↑p.fst * ↑p.fst - ↑p.fst + 1) }

theorem exists_unique_nonzero_integer_m :
  ∃! m : ℤ, m ≠ 0 ∧ (A ∩ B m).nonempty :=
sorry

end exists_unique_nonzero_integer_m_l577_577488


namespace division_remainder_l577_577562

theorem division_remainder :
  ∃ (R D Q : ℕ), D = 3 * Q ∧ D = 3 * R + 3 ∧ 251 = D * Q + R ∧ R = 8 := by
  sorry

end division_remainder_l577_577562


namespace cortney_downloads_all_files_in_2_hours_l577_577104

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l577_577104


namespace increasing_sequence_a8_l577_577950

theorem increasing_sequence_a8 :
  ∃ (a : ℕ → ℕ), 
    (∀ n, a (n+2) = a (n+1) + a n) ∧ 
    (∀ n, a n < a (n+1)) ∧ 
    a 7 = 120 ∧ 
    a 1 > 0 ∧ a 2 > 0 → 
    a 8 = 194 :=
begin
  sorry
end

end increasing_sequence_a8_l577_577950


namespace probability_all_primes_appearing_before_first_non_prime_l577_577413

-- Define basic properties and outcomes for the 10-sided die
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_non_prime (n : ℕ) : Prop :=
  n = 1 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 9 ∨ n = 10

-- Calculate the probability .(TODO: implement probability calculations in proofs)
def probability_all_primes_before_non_prime : ℚ :=
  1/30

-- Final theorem statement
theorem probability_all_primes_appearing_before_first_non_prime :
  probability_all_primes_before_non_prime = 1/30 :=
begin
  sorry
end

end probability_all_primes_appearing_before_first_non_prime_l577_577413


namespace probability_of_hitting_target_l577_577418

def is_hit (n : Nat) : Bool :=
  n >= 2  -- 2 to 9 represents hitting the target

def count_hits (group : List Nat) : Nat :=
  group.countp is_hit

def at_least_3_hits (group : List Nat) : Bool :=
  count_hits group >= 3

def simulated_groups : List (List Nat) :=
  [[7, 5, 2, 7], [0, 2, 9, 3], [7, 1, 4, 0], [9, 8, 5, 7], [0, 3, 4, 7],
   [4, 3, 7, 3], [8, 6, 3, 6], [6, 9, 4, 7], [1, 4, 1, 7], [4, 6, 9, 8],
   [0, 3, 7, 1], [6, 2, 3, 3], [2, 6, 1, 6], [8, 0, 4, 5], [6, 0, 1, 1],
   [3, 6, 6, 1], [9, 5, 9, 7], [7, 4, 2, 4], [7, 6, 1, 0], [4, 2, 8, 1]]

noncomputable def probability_at_least_3_hits : Real :=
  (simulated_groups.countp at_least_3_hits).toReal / simulated_groups.length.toReal

theorem probability_of_hitting_target :
  probability_at_least_3_hits = 0.75 :=
  sorry

end probability_of_hitting_target_l577_577418


namespace avg_row_col_ratio_l577_577439

/-- Carmen and Dani have a rectangular array of numbers with 50 rows and 40 columns.
Carmen adds the numbers in each row, and the average of her 50 sums is denoted as C.
Dani adds the numbers in each column, and the average of his 40 sums is denoted as D.
We are to prove that the value of C / D is 4 / 5. -/
theorem avg_row_col_ratio (arr : ℕ → ℕ → ℝ) (C D : ℝ) 
 (hC : C = (∑ i in Finset.range 50, ∑ j in Finset.range 40, arr i j) / 50)
 (hD : D = (∑ j in Finset.range 40, ∑ i in Finset.range 50, arr i j) / 40) :
  C / D = 4 / 5 :=
by
  sorry

end avg_row_col_ratio_l577_577439


namespace altitude_length_l577_577420

variables (l w : ℝ)
def rectangle_area (l w : ℝ) : ℝ := l * w
def triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height
def diagonal_length (l w : ℝ) : ℝ := Real.sqrt (l^2 + w^2)

theorem altitude_length (h : ℝ) (hl : w = l * Real.sqrt 2 / 2): 
  h = l * Real.sqrt 3 / 3 :=
  let rect_area := rectangle_area l w
  let diag := diagonal_length l w
  let tri_area := triangle_area diag h
  show h = l * Real.sqrt 3 / 3 from sorry

end altitude_length_l577_577420


namespace monotonically_increasing_interval_l577_577184

noncomputable def f (x : ℝ) : ℝ := Real.logb 2⁻¹ (-x^2 + 2 * x + 15)

theorem monotonically_increasing_interval : 
  ∀ x ∈ Ioo (1 : ℝ) (5 : ℝ), ∀ y ∈ Ioo (1 : ℝ) (5 : ℝ), x < y → f(x) < f(y) := 
sorry

end monotonically_increasing_interval_l577_577184


namespace equation_of_line_OF_l577_577571

variables {a b c p : ℝ}

theorem equation_of_line_OF (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : p ≠ 0) : 
  let A := (0, a),
      B := (b, 0),
      C := (c, 0),
      P := (0, p),
      E := (A.1 * (1 - B.1 / AB.1) + B.1 * (B.1 / AB.1), A.2 * (1 - B.2 / AB.2) + B.2 * (B.2 / AB.2)) -- Point on AC
      F := (A.1 * (1 - C.1 / AC.1) + C.1 * (C.1 / AC.1), A.2 * (1 - C.2 / AC.2) + C.2 * (C.2 / AC.2)) -- Point on AB
    in (AC.1 * P.2 - AB.1 * B.1) * x - (AB.1 * E.1 - P.1 * B.1) * y = 0 ↔
       \(\left(\frac{1}{b} - \frac{1}{c}\right)x - \left(\frac{1}{p} - \frac{1}{a}\right)y = 0\) := 
  sorry

end equation_of_line_OF_l577_577571


namespace angle_B_value_side_b_length_l577_577517

-- Definitions of angles and sides
variables (A B C a b c : ℝ)

-- Assumptions
axiom angles_arithmetic_sequence : 2 * B = A + C
axiom angles_sum : A + B + C = 180
axiom sides_geometric_sequence : b^2 = a * c
axiom area_triangle : real.sqrt 3 = 1 / 2 * a * c / 2 * (sin A) * (sin B) / (sin A + sin B + sin C)
axiom cos_rule : cos B = 1/2

-- Proof statement for angle B
theorem angle_B_value : B = 60 :=
by {
  sorry
}

-- Proof statement for side b length
theorem side_b_length : b = 2 :=
by {
  sorry
}

end angle_B_value_side_b_length_l577_577517


namespace kh_parallel_to_bc_l577_577074

-- Given
variables {A B C P Q H K : Point}

-- Define the notion of angle bisectors and perpendiculars
def is_angle_bisector (P : Point) (A B P : Point) : Prop := sorry
def is_perpendicular (A H B : Point) : Prop := sorry

-- Conditions
axiom condition1 : is_angle_bisector P A B P
axiom condition2 : is_angle_bisector Q A C Q
axiom condition3 : is_perpendicular A H P
axiom condition4 : is_perpendicular A K Q

-- Theorem to prove
theorem kh_parallel_to_bc : KH ∥ BC :=
sorry

end kh_parallel_to_bc_l577_577074


namespace sum_radii_invariant_l577_577568

theorem sum_radii_invariant {n : ℕ} (polygon : Polygon n) (triangulation1 triangulation2 : triangulation polygon) :
  (sum_inscribed_circle_radii triangulation1) = (sum_inscribed_circle_radii triangulation2) := 
sorry

end sum_radii_invariant_l577_577568


namespace sum_of_abs_squared_series_correct_l577_577883

noncomputable def sum_of_abs_squared_series (a r : ℝ) (h : |r| < 1) : ℝ :=
  a^2 / (1 - |r|^2)

theorem sum_of_abs_squared_series_correct (a r : ℝ) (h : |r| < 1) :
  sum_of_abs_squared_series a r h = a^2 / (1 - |r|^2) :=
by
  sorry

end sum_of_abs_squared_series_correct_l577_577883


namespace no_function_f_sin_identity_l577_577115

theorem no_function_f_sin_identity :
  ¬ (∃ f : ℝ → ℝ, ∀ x : ℝ, sin (4 * x) = f (sin x)) :=
sorry

end no_function_f_sin_identity_l577_577115


namespace area_of_triangle_bounded_by_lines_l577_577777

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l577_577777


namespace anne_speed_l577_577432

-- Definition of distance and time
def distance : ℝ := 6
def time : ℝ := 3

-- Statement to prove
theorem anne_speed : distance / time = 2 := by
  sorry

end anne_speed_l577_577432


namespace jeremy_gifted_37_goats_l577_577559

def initial_horses := 100
def initial_sheep := 29
def initial_chickens := 9

def total_initial_animals := initial_horses + initial_sheep + initial_chickens
def animals_bought_by_brian := total_initial_animals / 2
def animals_left_after_brian := total_initial_animals - animals_bought_by_brian

def total_male_animals := 53
def total_female_animals := 53
def total_remaining_animals := total_male_animals + total_female_animals

def goats_gifted_by_jeremy := total_remaining_animals - animals_left_after_brian

theorem jeremy_gifted_37_goats :
  goats_gifted_by_jeremy = 37 := 
by 
  sorry

end jeremy_gifted_37_goats_l577_577559


namespace arithmetic_sequence_properties_l577_577720

variable (a_n : ℕ → ℚ)
variable (a_3 a_11 : ℚ)

notation "a₃" => a_3
notation "a₁₁" => a_11

theorem arithmetic_sequence_properties :
  a₃ = a_n 3 → a₁₁ = a_n 11 → 
  (∃ (a₁ d : ℚ), a_n n = a₁ + (n - 1) * d ∧ a₁ = 0 ∧ d = 3/2) := sorry

end arithmetic_sequence_properties_l577_577720


namespace hyperbola_eccentricity_range_valid_l577_577624

def hyperbola_eccentricity_range (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) : Prop :=
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let Q := (c, 3 * a / 2)
  let A := (c, sqrt (c^2 - a^2) * b / a) in
  let hyperbola_eq (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1 in
  hyperbola_eq c (sqrt (c^2 - a^2) * b / a) ∧
  (dist F2 Q > dist F2 A) ∧
  ∀ P : ℝ × ℝ, (hyperbola_eq P.1 P.2) →
    (dist P F1 + dist P Q > (3 / 2) * 2 * c) →
    (1 < c / a ∧ c / a < 7 / 6)

theorem hyperbola_eccentricity_range_valid (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) :
  hyperbola_eccentricity_range a b c h_a_pos h_b_pos h_c_pos :=
begin
  sorry -- Proof is omitted as per the instructions.
end

end hyperbola_eccentricity_range_valid_l577_577624


namespace revenue_percentage_l577_577698

theorem revenue_percentage (R C : ℝ) (hR_pos : R > 0) (hC_pos : C > 0) :
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 62.5 := by
  sorry

end revenue_percentage_l577_577698


namespace two_digit_number_is_9_l577_577038

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l577_577038


namespace factorial_sum_eq_n_to_k_l577_577128

theorem factorial_sum_eq_n_to_k (m n k : ℕ) (hm : m > 1) (hn : n > 1) (hk : k > 1) :
  (∑ i in Finset.range m, i.factorial) = n ^ k ↔ (m = 3 ∧ n = 3 ∧ k = 2) :=
by
  sorry

end factorial_sum_eq_n_to_k_l577_577128


namespace determine_a_and_f_neg_t_l577_577978

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 * Real.sin x + a

theorem determine_a_and_f_neg_t (t a : ℝ) :
  (f π a = 1) ∧ (f t a = 2) → (a = 1) ∧ (f (-t) a = 0) :=
begin
  intros h,
  sorry
end

end determine_a_and_f_neg_t_l577_577978


namespace convex_quadrilaterals_count_l577_577910

theorem convex_quadrilaterals_count :
  let n := 15 in
  let central_point := 1 in 
  (n.choose 3) = 455 :=
by sorry

end convex_quadrilaterals_count_l577_577910


namespace arc_length_of_y_eq_exp_x_plus_26_l577_577003

noncomputable def arc_length : ℝ :=
∫ x in Real.log (Real.sqrt 8) .. Real.log (Real.sqrt 24), Real.sqrt (1 + Real.exp (2 * x))

theorem arc_length_of_y_eq_exp_x_plus_26 : 
  arc_length = 2 + 1 / 2 * Real.log (4 / 3) :=
sorry

end arc_length_of_y_eq_exp_x_plus_26_l577_577003


namespace number_of_digits_base10_of_x_l577_577536

theorem number_of_digits_base10_of_x (x : ℝ) (h : log 3 (log 3 (log 3 x)) = 3) : 
  ⌊(log 10 (3^(27)) + 1)⌋ = 14 :=
by
  sorry

end number_of_digits_base10_of_x_l577_577536


namespace equation_of_curve_t_circle_through_fixed_point_l577_577242

noncomputable def problem (x y : ℝ) : Prop :=
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -1)
  let O : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (x, y)
  let N : ℝ × ℝ := (0, y)
  (x + 1) * (x - 1) + y * y = y * (y + 1)

noncomputable def curve_t_equation (x : ℝ) : ℝ :=
  x^2 - 1

theorem equation_of_curve_t (x y : ℝ) 
  (h : problem x y) :
  y = curve_t_equation x := 
sorry

noncomputable def passing_through_fixed_point (x y : ℝ) : Prop :=
  let y := x^2 - 1
  let y' := 2 * x
  let P : ℝ × ℝ := (x, y)
  let Q_x := (4 * x^2 - 1) / (8 * x)
  let Q : ℝ × ℝ := (Q_x, -5 / 4)
  let H : ℝ × ℝ := (0, -3 / 4)
  (x * Q_x + (-3 / 4 - y) * ( -3 / 4 + 5 / 4)) = 0

theorem circle_through_fixed_point (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y = curve_t_equation x)
  (h : passing_through_fixed_point x y) :
  ∃ t : ℝ, passing_through_fixed_point x t ∧ t = -3 / 4 :=
sorry

end equation_of_curve_t_circle_through_fixed_point_l577_577242


namespace remainder_of_S_mod_5_is_3_l577_577875

noncomputable def S : ℚ :=
  ∑ k in Finset.range 401, (2015 : ℚ) / (((5 * (k + 1) : ℚ) - 2) * (((5 * (k + 1) : ℚ) + 3)))

theorem remainder_of_S_mod_5_is_3 : (Int.round S) % 5 = 3 := sorry

end remainder_of_S_mod_5_is_3_l577_577875


namespace no_integer_greater_than_2008_eq_sum_of_squares_digits_l577_577665

theorem no_integer_greater_than_2008_eq_sum_of_squares_digits :
  ∀ n, n > 2008 → (∀ (k : ℕ) (a : ℕ → ℕ), n ≠ (finset.range (k + 1)).sum (λ i, (a i)^2)) :=
by
  sorry

end no_integer_greater_than_2008_eq_sum_of_squares_digits_l577_577665


namespace cars_in_race_l577_577363

theorem cars_in_race (C : ℕ)
  (h1 : ∀ car, car_has C car (2 + 1)) -- Definition that each car starts with 3 people (2 passengers + 1 driver)
  (h2 : ∀ car, halfway_gains car 1) -- Definition that each car gains 1 additional passenger after halfway
  (h3 : (4 * C = 80)) -- Total number of people in the cars at the end of the race is 80
  : C = 20 := -- Prove that the number of cars is 20
sorry

end cars_in_race_l577_577363


namespace value_of_b1b7b10_l577_577515

noncomputable theory

-- Define the non-zero arithmetic sequence {a_n}
def isArithmeticSeq (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + 1) = a n + a 1

-- Define the non-zero arithmetic sequence {a_n}, satisfying the given equation condition
def satisfiesCondition (a : ℕ → ℝ) : Prop :=
a 3 - 2 * (a 6)^2 + 3 * a 7 = 0

-- Define the geometric sequence {b_n}
def isGeometricSeq (b : ℕ → ℝ) : Prop :=
∀ n m : ℕ, b (n + m) = b n * (b m)

-- Given {b_n}, b_6 = a_6
def b6_equals_a6 (b a : ℕ → ℝ) : Prop :=
b 6 = a 6

-- The main statement to prove
theorem value_of_b1b7b10 (a b : ℕ → ℝ) 
  (ha_arith : isArithmeticSeq a)
  (ha_cond : satisfiesCondition a)
  (hb_geom : isGeometricSeq b)
  (hb_eq : b6_equals_a6 b a) :
  b 1 * b 7 * b 10 = 8 :=
by sorry

end value_of_b1b7b10_l577_577515


namespace max_area_equilateral_triangle_inscribed_in_rectangle_l577_577352

theorem max_area_equilateral_triangle_inscribed_in_rectangle (AB BC : ℝ) :
  AB = 15 → BC = 8 →
  ∃ (p q r : ℕ), (q > 0) ∧ (∀ prime x, x^2 ∣ q → false) ∧
  let A := (p * Real.sqrt q - r) in
  (∃ (s : ℝ), s ≤ BC ∧ (Real.sqrt 3 * s^2)/4 = A) ∧ p + q + r = 19 :=
by
  intros hAB hBC
  exists 16, 3, 0
  split
  repeat {split}
  intros x x_prime
  -- proof of this prime condition is needed here but we will use sorry
  sorry
  exists 8
  split
  apply le_refl
  field_simp
  ring
  -- proof is not needed because of the problem was specified not to include proof steps
  sorry

end max_area_equilateral_triangle_inscribed_in_rectangle_l577_577352


namespace continuous_stripe_probability_l577_577459

noncomputable def probability_continuous_stripe_encircle_cube : ℚ :=
  let total_combinations : ℕ := 2^6
  let favor_combinations : ℕ := 3 * 4 -- 3 pairs of parallel faces, with 4 valid combinations each
  favor_combinations / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe_encircle_cube = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l577_577459


namespace optimistic_annual_reports_infinite_l577_577262

theorem optimistic_annual_reports_infinite {a : ℕ → ℕ} (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) 
  (h2 : ∑ (i : ℕ) in finset.range n, 1 / (a i : ℝ) ≤ 1 / 2) : 
  ∃ (f : ℕ → ℕ → ℕ),
  (∀ k, ∀ i, 1 ≤ i ∧ i ≤ n → 1 ≤ f k i ∧ f k i ≤ a i) ∧
  (∀ k, ∃ j, 1 ≤ j ∧ j ≤ n ∧ 
    (∀ i, (1 ≤ i ∧ i ≤ n ∧ i ≠ j) → f (k + 1) i > f k i)) ∧
  (∀ k, ∃ j, 1 ≤ j ∧ j ≤ n ∧ 
    (∃ i, (1 ≤ i ∧ i ≤ n ∧ i ≠ j) → f (k + 1) i = f k i)) := sorry

end optimistic_annual_reports_infinite_l577_577262


namespace vasya_guaranteed_win_l577_577658

theorem vasya_guaranteed_win :
  ∃ strategy : (ℕ → ℕ), (∀ n ∈ set.range strategy, n ≤ 2018) → 
  (∀ n m, strategy n = strategy m → n = m) → 
  (∀ S : finset ℕ, S.card = 3 → ∃ a b c ∈ S, b - a = c - b) → 
  Vasya_wins :=
sorry

end vasya_guaranteed_win_l577_577658


namespace boat_round_trip_ratio_l577_577830

theorem boat_round_trip_ratio
  (speed_boat_still_water : ℕ)
  (speed_current : ℕ)
  (distance_downstream : ℕ)
  (distance_upstream : ℕ)
  (h1 : speed_boat_still_water = 20)
  (h2 : speed_current = 4)
  (h3 : distance_downstream = 2)
  (h4 : distance_upstream = 2) :
  let downstream_speed := speed_boat_still_water + speed_current,
      upstream_speed := speed_boat_still_water - speed_current,
      time_downstream := distance_downstream / downstream_speed,
      time_upstream := distance_upstream / upstream_speed,
      total_time := time_downstream + time_upstream,
      total_distance := distance_downstream + distance_upstream,
      average_speed := total_distance / total_time,
      ratio := average_speed / speed_boat_still_water
  in ratio = 24 / 25 :=
by {
  sorry
}

end boat_round_trip_ratio_l577_577830


namespace part1_extremum_part2i_range_part2ii_inequality_l577_577523

-- Definitions and conditions based on the problem statement
def f (x a : ℝ) := x - 1 / x - a * (Real.log x)

-- Part 1: Prove that if f(x) has an extremum at x = 2, then a = 5/2 and find the extremum values
theorem part1_extremum (a : ℝ) (h_ext : ∀ x : ℝ, x − 1 / x - a * Real.log x = 0 → x = 2) :
  a = 5 / 2 ∧ (∀ x, x < 1 / 2 → f x (5 / 2) < f (1 / 2) (5 / 2)) ∧
      (∀ x, 1 / 2 < x < 2 → f x (5 / 2) < f (2) (5 / 2)) ∧
      (∀ x, x > 2 → f x (5 / 2) > f (2) (5 / 2)) :=
by sorry

-- Part 2(i): Prove if f(x) ≥ 0 for x ≥ 1, then the range of a is (-∞, 2]
theorem part2i_range (a : ℝ) (h_nonneg : ∀ x : ℝ, x ≥ 1 → f x a ≥ 0) :
  a ≤ 2 :=
by sorry

-- Part 2(ii): Prove the given logarithmic inequality for n ∈ ℕ+
theorem part2ii_inequality (n : ℕ) (h_pos : 0 < n) :
  (Finset.range n).sum (λ k, Real.log (1 + 1 / (k + 1)))^2 < n / (n + 1) :=
by sorry

end part1_extremum_part2i_range_part2ii_inequality_l577_577523


namespace AJ_length_l577_577604

-- Definitions of points and relationships
noncomputable theory
open_locale classical

def Point : Type := ℝ × ℝ -- Assuming a point in 2D plane defined by coordinates

-- Placeholder definitions for is_parallelogram and intersects
def is_parallelogram (A B C D : Point) : Prop := sorry
def intersects (line1 line2 : Point × Point) (P : Point) : Prop := sorry

-- The theorem to be stated and proved
theorem AJ_length (A B C D H J I : Point) :
  is_parallelogram A B C D →
  dist B H = 3 * dist B C →
  intersects (A, H) (B, D) I →
  intersects (A, D) (A, J) →
  dist J I = 30 →
  dist H I = 45 →
  dist A J = 45 :=
by {
  -- Omitted proof
  sorry
}

end AJ_length_l577_577604


namespace cricket_team_average_age_l577_577330

theorem cricket_team_average_age (A : ℕ) (captain_age : ℕ) (wicket_keeper_age : ℕ) (remainings_avg : ℕ) (remainings_total_age : ℕ) :
  captain_age = 24 →
  wicket_keeper_age = 27 →
  remainings_avg = A - 1 →
  remainings_total_age = 9 * remainings_avg →
  11 * A - 51 = remainings_total_age →
  A = 21 :=
by {
  intros h1 h2 h3 h4 h5,
  sorry
}

end cricket_team_average_age_l577_577330


namespace function_properties_l577_577803

noncomputable def f (x : ℝ) (A ω ϕ : ℝ) := A * sin (ω * x + ϕ)

theorem function_properties
  (A ω ϕ : ℝ)
  (hf₁ : ω * (-(π / 3)) + ϕ = 0)
  (hf₂ : A * sin (π / 2) = 2)
  (hf₃ : A * sin (π) = 0)
  (hf₄ : A * sin (3 * π / 2) = -2)
  (hf₅ : A * sin (2 * π) = 0) :
  A = 2 ∧ ∀ x : ℝ, f (x + π / 3) A ω ϕ = f (-x + π / 3) A ω ϕ :=
by
  sorry

end function_properties_l577_577803


namespace total_cost_correct_l577_577059

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_cost : ℝ := 12.30

theorem total_cost_correct : football_cost + marbles_cost = total_cost := 
by
  sorry

end total_cost_correct_l577_577059


namespace symmetric_point_is_correct_l577_577481

def symmetric_point (A : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let ⟨a, b, c, d⟩ := plane
  let ⟨xA, yA, zA⟩ := A
  let t := (a * xA + b * yA + c * zA + d) / (a^2 + b^2 + c^2)
  (xA - 2 * a * t, yA - 2 * b * t, zA - 2 * c * t)

theorem symmetric_point_is_correct :
  symmetric_point (2, 0, 2) (4, 6, 4, -50) = (6, 6, 6) :=
by
  sorry

end symmetric_point_is_correct_l577_577481


namespace range_of_b_l577_577341

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

theorem range_of_b (b : ℝ) : 
  (∃ (x1 x2 x3 : ℝ), f x1 = -b ∧ f x2 = -b ∧ f x3 = -b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ (-1 < b ∧ b < 0) :=
by
  sorry

end range_of_b_l577_577341


namespace find_increasing_function_l577_577060

def increases_as_x_increases (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (0 < x ∧ x < y) → f x < f y

theorem find_increasing_function :
  let f1 := λ x : ℝ, -2 * x + 1
  let f2 := λ x : ℝ, (x + 1)^2 + 1
  let f3 := λ x : ℝ, -x^2 - 1
  let f4 := λ x : ℝ, 1 / x
  increases_as_x_increases f2 ∧ 
  ¬ (increases_as_x_increases f1) ∧ 
  ¬ (increases_as_x_increases f3) ∧ 
  ¬ (increases_as_x_increases f4) := 
by
  sorry

end find_increasing_function_l577_577060


namespace cos_double_angle_l577_577213

-- Define the hypothesis
def cos_alpha (α : ℝ) : Prop := Real.cos α = 1 / 2

-- State the theorem
theorem cos_double_angle (α : ℝ) (h : cos_alpha α) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l577_577213


namespace find_larger_number_l577_577741

theorem find_larger_number :
  ∃ x y : ℤ, x + y = 30 ∧ 2 * y - x = 6 ∧ x > y ∧ x = 18 :=
by
  sorry

end find_larger_number_l577_577741


namespace Intersect_A_B_l577_577508

-- Defining the sets A and B according to the problem's conditions
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x ∈ Set.univ | x^2 - 5*x + 4 < 0}

-- Prove that the intersection of A and B is {2}
theorem Intersect_A_B : A ∩ B = {2} := by
  sorry

end Intersect_A_B_l577_577508


namespace number_of_knights_l577_577654

-- Definition of knight and liar
inductive Inhabitant
| knight : Inhabitant
| liar : Inhabitant

open Inhabitant

-- Statements of the inhabitants
def statement_1 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i1 with
  | knight => 1
  | liar => true

def statement_2 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i2 with
  | knight => 2
  | liar => true

def statement_3 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i3 with
  | knight => 3
  | liar => true

def statement_4 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  (i1 = liar ∧ i2 = liar ∧ i3 = liar ∧ i4 = liar ∧ i5 = liar)

def statement_5 (i4 : Inhabitant) : Prop := 
  match i5 with
  | knight => i4 = liar
  | liar => true

-- Problem statement in Lean 4
theorem number_of_knights (i1 i2 i3 i4 i5 : Inhabitant) 
  (h1 : statement_1 i1 i2 i3 i4 i5)
  (h2 : statement_2 i1 i2 i3 i4 i5)
  (h3 : statement_3 i1 i2 i3 i4 i5)
  (h4 : statement_4 i1 i2 i3 i4 i5)
  (h5 : statement_5 i4) :
  (i1 = knight ∨ i2 = knight ∨ i3 = knight ∨ i4 = knight ∨ i5 = knight) → 
  (i1 = knight ∧ i2 = knight ∧ i3 = knight ∧ i4 = knight ∧ i5 = knight) → 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end number_of_knights_l577_577654


namespace area_of_triangle_l577_577763

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l577_577763


namespace sum_floor_log2_to_2048_l577_577082

theorem sum_floor_log2_to_2048 :
  (Finset.sum (Finset.range 2048.succ) (λ N : ℕ, Int.toNat ⌊Real.logb 2 (N : ℝ)⌋) = 14349) :=
by
  sorry

end sum_floor_log2_to_2048_l577_577082


namespace archer_shots_and_hits_l577_577869

theorem archer_shots_and_hits :
  ∃ n m : ℕ, -- declare natural numbers n and m
    (∀ (h : 10 < n ∧ n < 20),  -- condition: 10 < n < 20
     8 * m = 3 * n ∧  -- condition: 8m = 3n
     5 * m - 3 * (n - m) = 0 ∧  -- condition: 5m - 3(n - m) = 0
     n = 16 ∧  -- conclusion: n = 16
     m = 6). -- conclusion: m = 6
Proof
{
  sorry -- proof to be completed
}

end archer_shots_and_hits_l577_577869


namespace length_of_AM_l577_577308

-- Define the given conditions
def rectangle_ABCD : Prop := ∃ A B C D M : Type,
  let AB := 4
  let BC := 8
  -- Area condition
  ∃ a b c d m : ℝ, a = (4 : ℝ) ∧ b = (8 : ℝ) ∧
  ∃ M, 
  area (ABM : ℝ) = 2 * area (MBDC : ℝ)

-- Prove that the length of AM is 38/3
theorem length_of_AM {A B C D M : Type} (h : rectangle_ABCD) : AM = 38 / 3 :=
sorry

end length_of_AM_l577_577308


namespace cost_of_book_first_sold_at_loss_l577_577208

theorem cost_of_book_first_sold_at_loss (C1 C2 C3 : ℝ) (h1 : C1 + C2 + C3 = 810)
    (h2 : 0.88 * C1 = 1.18 * C2) (h3 : 0.88 * C1 = 1.27 * C3) : 
    C1 = 333.9 := 
by
  -- Conditions given
  have h4 : C2 = 0.88 * C1 / 1.18 := by sorry
  have h5 : C3 = 0.88 * C1 / 1.27 := by sorry

  -- Substituting back into the total cost equation
  have h6 : C1 + 0.88 * C1 / 1.18 + 0.88 * C1 / 1.27 = 810 := by sorry

  -- Simplifying and solving for C1
  have h7 : C1 = 333.9 := by sorry

  -- Conclusion
  exact h7

end cost_of_book_first_sold_at_loss_l577_577208


namespace angle_between_unit_vectors_l577_577996

-- Definitions from conditions
variables (a b : EuclideanSpace ℝ (Fin 2))
variables (h_unit_a : ‖a‖ = 1) (h_unit_b : ‖b‖ = 1)
variables (h_perp : (2 • a + b) ⬝ b = 0)

-- Theorem to prove the angle between a and b is 2π / 3
theorem angle_between_unit_vectors (a b : EuclideanSpace ℝ (Fin 2))
  (h_unit_a : ‖a‖ = 1) (h_unit_b : ‖b‖ = 1)
  (h_perp : (2 • a + b) ⬝ b = 0) : real.angle a b = 2 * real.pi / 3 :=
sorry

end angle_between_unit_vectors_l577_577996


namespace area_triangle_OCD_l577_577752

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l577_577752


namespace polynomial_decomposition_l577_577555

theorem polynomial_decomposition :
  let f := (λ x : ℝ, (2 * x ^ 2 - 3 * x + 5) * (5 - x))
  ∃ a b c d : ℝ,
  (∀ x : ℝ, f x = a * x^3 + b * x^2 + c * x + d) ∧ (27 * a + 9 * b + 3 * c + d = 28) :=
by
  sorry

end polynomial_decomposition_l577_577555


namespace triangle_identity_l577_577580

variables {α : Type*} [real_division_ring α]

theorem triangle_identity
  (a b c : α)
  (A B C : α)
  (s : α)
  (cos_A cos_B cos_C : α)
  (h_s : s = (a + b + c) / 2)
  (h_cos_A : cos_A = cos A)
  (h_cos_B : cos_B = cos B)
  (h_cos_C : cos_C = cos C) :
  (a^2 * (s - a) / (1 + cos_A) + b^2 * (s - b) / (1 + cos_B) + c^2 * (s - c) / (1 + cos_C)) = a * b * c :=
by {
  sorry
}

end triangle_identity_l577_577580


namespace pair_not_product_48_l577_577808

theorem pair_not_product_48:
  (∀(a b : ℤ), (a, b) = (-6, -8)                    → a * b = 48) ∧
  (∀(a b : ℤ), (a, b) = (-4, -12)                   → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (3/4, -64)                  → a * b ≠ 48) ∧
  (∀(a b : ℤ), (a, b) = (3, 16)                     → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (4/3, 36)                   → a * b = 48)
  :=
by
  sorry

end pair_not_product_48_l577_577808


namespace mark_midpoints_of_square_l577_577489

theorem mark_midpoints_of_square {A B C D E F G H : Type*}
  (hAB : segment A B)
  (hBC : segment B C)
  (hCD : segment C D)
  (hDA : segment D A)
  (hE : midpoint A D E)
  (hF : midpoint C D F)
  (hG : midpoint C B G)
  (hH : midpoint B A H)
  : 
  (∀ M N : Point, M ≠ N → ∃ K, perpendicular_bisector (segment M N) K ∧ K = 2)
  := sorry

end mark_midpoints_of_square_l577_577489


namespace angle_between_unit_vectors_correct_l577_577995

open Real

noncomputable def angle_between_unit_vectors
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (h : (2 • a + b) ⬝ b = 0) : Real :=
  acos (- (1 / 2))

theorem angle_between_unit_vectors_correct
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (h : (2 • a + b) ⬝ b = 0) : 
  angle_between_unit_vectors a b ha hb h = 2 * π / 3 :=
by
  sorry

end angle_between_unit_vectors_correct_l577_577995


namespace inscribed_circle_radius_l577_577797

noncomputable def triangle_radius_proof (AB AC BC : ℕ) (h1 : AB = 15) (h2 : AC = 20) (h3 : BC = 25) : Real :=
let s := (AB + AC + BC) / 2
let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
let r := K / s
r

theorem inscribed_circle_radius :
  ∀ (AB AC BC : ℕ), AB = 15 → AC = 20 → BC = 25 → triangle_radius_proof AB AC BC = 5 :=
  by intros AB AC BC h1 h2 h3
     unfold triangle_radius_proof
     sorry

end inscribed_circle_radius_l577_577797


namespace abs_diff_expr_l577_577819

theorem abs_diff_expr :
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  |a| - |b| = 4 :=
by
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  sorry

end abs_diff_expr_l577_577819


namespace triangle_area_bounded_by_lines_l577_577782

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l577_577782


namespace vertex_of_parabola_l577_577335

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

-- Define the vertex point
def vertex : ℝ × ℝ := (-2, -1)

-- The theorem we need to prove
theorem vertex_of_parabola : ∀ x : ℝ, parabola x = (x + 2)^2 - 1 → vertex = (-2, -1) := 
by
  sorry

end vertex_of_parabola_l577_577335


namespace number_of_valid_numbers_l577_577840

-- We define the conditions given
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def hundred_place : ℕ := 5

def count_valid_numbers : ℕ :=
  let possible_thousands := {1, 2, 3, 4}
  let remaining_digits := digits.erase 5
  let choices_for_hundreds_thousands := possible_thousands.card
  let permutations_of_two := (remaining_digits.card.choose 2) * 2
  choices_for_hundreds_thousands * permutations_of_two

-- Assert the problem question is equivalent to the correct answer
theorem number_of_valid_numbers : count_valid_numbers = 48 := by
  sorry

end number_of_valid_numbers_l577_577840


namespace find_working_hours_for_y_l577_577001

theorem find_working_hours_for_y (Wx Wy Wz Ww : ℝ) (h1 : Wx = 1/8)
  (h2 : Wy + Wz = 1/6) (h3 : Wx + Wz = 1/4) (h4 : Wx + Wy + Ww = 1/5)
  (h5 : Wx + Ww + Wz = 1/3) : 1 / Wy = 24 :=
by
  -- Given the conditions
  -- Wx = 1/8
  -- Wy + Wz = 1/6
  -- Wx + Wz = 1/4
  -- Wx + Wy + Ww = 1/5
  -- Wx + Ww + Wz = 1/3
  -- We need to prove that 1 / Wy = 24
  sorry

end find_working_hours_for_y_l577_577001


namespace knights_count_l577_577647

-- Define the inhabitants and their nature
inductive Inhabitant : Type
| first : Inhabitant
| second : Inhabitant
| third : Inhabitant
| fourth : Inhabitant
| fifth : Inhabitant

open Inhabitant

-- Define whether an inhabitant is a knight or a liar
inductive Nature : Type
| knight : Nature
| liar : Nature

open Nature

-- Assume each individual is either a knight or a liar (truth value function)
def truth_value : Inhabitant → Nature → Prop
| first, knight  => sorry -- To be proven
| second, knight => sorry -- To be proven
| third, knight  => sorry -- To be proven
| fourth, knight => sorry -- To be proven
| fifth, knight  => sorry -- To be proven

-- Define the statements made by inhabitants as logical conditions
def statements : Prop :=
  (truth_value first knight → (1 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value second knight → (2 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value third knight → (3 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value fourth knight → ¬ (truth_value first knight) ∧ ¬ (truth_value second knight) ∧ ¬ (truth_value third knight) ∧ ¬ (truth_value fifth knight))
  ∧ (truth_value fifth knight → ¬ (truth_value fourth knight))

-- The goal is to prove that there are exactly 2 knights
theorem knights_count : (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0) = 2 :=
by
  sorry

end knights_count_l577_577647


namespace preceding_zeros_2009_pow_2011_zeros_precede_last_digit_2009_pow_2011_l577_577110

theorem preceding_zeros_2009_pow_2011 : 
  (2009 ^ 2011) % 1000 = 609 :=
by
  sorry

theorem zeros_precede_last_digit_2009_pow_2011 :
  (2009 ^ 2011) % 1000 = 609 → preceding_zeros_2009_pow_2011 = 1 :=
by
  sorry

end preceding_zeros_2009_pow_2011_zeros_precede_last_digit_2009_pow_2011_l577_577110


namespace eccentricity_of_ellipse_l577_577173

variables {E F1 F2 P Q : Type}
variables (a c : ℝ) 

-- Define the foci and intersection conditions
def is_right_foci (F1 F2 : Type) (E : Type) : Prop := sorry
def line_intersects_ellipse (E : Type) (P Q : Type) (slope : ℝ) : Prop := sorry
def is_right_triangle (P F2 : Type) : Prop := sorry

-- Prove the eccentricity condition
theorem eccentricity_of_ellipse
  (h_foci : is_right_foci F1 F2 E)
  (h_line : line_intersects_ellipse E P Q (4 / 3))
  (h_triangle : is_right_triangle P F2) :
  (c / a) = (5 / 7) :=
sorry

end eccentricity_of_ellipse_l577_577173


namespace number_of_knights_is_two_number_of_knights_is_two_l577_577649

-- Define types for inhabitants (knights or liars)
inductive Inhabitant
| knight
| liar

open Inhabitant

-- Define the statements given by the inhabitants
def statements (i : ℕ) : String :=
  match i with
  | 1 => "One knight"
  | 2 => "Two knights"
  | 3 => "Three knights"
  | 4 => "Don't believe them, they are all liars"
  | 5 => "You're the liar!"
  | _ => ""

-- Define the truth-telling property of knights
def tells_truth (i : ℕ) (s : String) : Prop :=
  match i with
  | 1 => (s = "One knight") ↔ (count_knights = 1)
  | 2 => (s = "Two knights") ↔ (count_knights = 2)
  | 3 => (s = "Three knights") ↔ (count_knights = 3)
  | 4 => (s = "Don't believe them, they are all liars") ↔ (inhabitant 1 = liar ∧ inhabitant 2 = liar ∧ inhabitant 3 = liar)
  | 5 => (s = "You're the liar!") ↔ (inhabitant 4 = liar)
  | _ => false

-- Define the main theorem to be proven
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

-- Noncomputable definition to avoid computational problems
noncomputable def count_knights : ℕ :=
  sorry

-- Noncomputable to define each inhabitant's type
noncomputable def inhabitant (i : ℕ) : Inhabitant :=
  match i with
  | 1 => liar
  | 2 => knight
  | 3 => liar
  | 4 => liar
  | 5 => knight
  | _ => liar -- Default to liar, although there are only 5 inhabitants

-- Additional properties used
def is_knight (i : ℕ) : Prop := inhabitant i = knight
def is_liar (i : ℕ) : Prop := inhabitant i = liar

-- Count the number of knights
noncomputable def count_knights : ℕ :=
  List.length (List.filter (λ i => is_knight i) [1, 2, 3, 4, 5])

-- Main theorem that states there are exactly two knights according to the statements
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

end number_of_knights_is_two_number_of_knights_is_two_l577_577649


namespace all_terms_perfect_squares_l577_577983

open Nat

variable {a : ℕ → ℤ}

/-- Given conditions on the sequence -/
axiom cond1 : ∀ n, 2 ≤ n → a (n+1) = 3 * a n - 3 * a (n-1) + a (n-2)
axiom cond2 : 2 * a 1 = a 0 + a 2 - 2
axiom cond3 : ∀ m, ∃ k, ∀ i, i < m → is_square (a (k + i))

/-- Prove that all terms of the sequence are perfect squares -/
theorem all_terms_perfect_squares : ∀ n, ∃ p, a n = p ^ 2 :=
by
  sorry

end all_terms_perfect_squares_l577_577983


namespace triangle_area_is_64_l577_577771

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l577_577771


namespace expectation_between_min_max_l577_577305

variable {α : Type*} [MeasureTheory.MeasurableSpace α] [MeasureTheory.ProbabilityMeasure (MeasureTheory.Measure α)]

-- Given conditions from step a)
variable (a b : ℝ)
variable (X : α → ℝ)
variable (f : ℝ → ℝ) 
variable (μ : MeasureTheory.Measure α)

-- The density function condition
variable (h₀ : ∀ x, a ≤ x ∧ x ≤ b → f(x) ≥ 0)
variable (h₁ : ∀ x, ¬(a ≤ x ∧ x ≤ b) → f(x) = 0)

-- Normalizing the probability density function
variable (h₂ : MeasureTheory.Integrable f (μ.restrict (MeasureTheory.Icc a b)))
variable (h_norm : ∫ x in MeasureTheory.Icc a b, f x ∂μ = 1)

-- Expected value definition
noncomputable def expected_value (X : α → ℝ) : ℝ :=
  MeasureTheory.Integral X μ

-- The proof statement
theorem expectation_between_min_max :
  a ≤ expected_value X ∧ expected_value X ≤ b := sorry

end expectation_between_min_max_l577_577305


namespace descending_sequence_l577_577381

variables (a b c d e : ℝ)

def descending_order (x y z w v : ℝ) : Prop :=
  x > y ∧ y > z ∧ z > w ∧ w > v

theorem descending_sequence :
  descending_order (3^1) (3^(1/5)) (3^(-2/3)) (3^(-2)) (3^(-3)) :=
by
  sorry

end descending_sequence_l577_577381


namespace find_c_find_cos_A_l577_577503

-- Define the vertices of the triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (0, 0)
def C (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the vectors AB and AC
def vectorAB : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vectorAC (c : ℝ) : ℝ × ℝ := (C(c).1 - A.1, C(c).2 - A.2)

-- Define the dot product condition
def dot_product_is_zero (c : ℝ) : Prop := 
  vectorAB.1 * vectorAC(c).1 + vectorAB.2 * vectorAC(c).2 = 0

-- The value of c such that dot product is zero
theorem find_c : ∃ c, dot_product_is_zero c ∧ c = 25 / 3 :=
by
  use 25 / 3
  sorry

-- Given c = 5, find the value of cos ∠A
def c : ℝ := 5

-- Distances
def dAB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
def dAC : ℝ := Real.sqrt ((A.1 - C(c).1)^2 + (A.2 - C(c).2)^2)
def dBC : ℝ := Real.sqrt ((B.1 - C(c).1)^2 + (B.2 - C(c).2)^2)

-- Cosine of angle A using the cosine rule
def cos_A : ℝ := (dAB^2 + dAC^2 - dBC^2) / (2 * dAB * dAC)

-- The value of cos ∠A
theorem find_cos_A : c = 5 → cos_A = Real.sqrt 5 / 5 :=
by
  intros h
  rw [h]
  sorry

end find_c_find_cos_A_l577_577503


namespace find_prime_p_l577_577124

theorem find_prime_p (p : ℕ) (hp : Prime p) (hdiv : (Nat.divisors (p^2 + 11)).length = 6) : p = 3 :=
sorry

end find_prime_p_l577_577124


namespace horizontal_asymptote_f_l577_577096

-- Define the rational function f
def f (x : ℝ) : ℝ := (8 * x^2 - 12) / (4 * x^2 + 6 * x - 3)

-- Define the horizontal asymptote
def horizontal_asymptote (y : ℝ) : Prop :=
  ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f(x) - y| < ε

-- State the proof problem
theorem horizontal_asymptote_f : horizontal_asymptote 2 :=
sorry

end horizontal_asymptote_f_l577_577096


namespace knights_count_is_two_l577_577641

def inhabitant (i : Fin 5) : Prop :=
  (i = 0 → inhabitants_truth 1) ∧
  (i = 1 → inhabitants_truth 2) ∧
  (i = 2 → inhabitants_truth 3) ∧
  (i = 3 → ∀ k, k ≠ 3 → ¬inhabitants_truth k) ∧
  (i = 4 → inhabitants_truth 4 ≠ inhabitants_truth 3)

def inhabitants_truth (n : ℕ) : Prop := sorry

theorem knights_count_is_two : ∃ (knights : Fin 5 → Prop), (∃! i, knights i) ∧ (∃! j, knights j) ∧ (i ≠ j) :=
sorry

end knights_count_is_two_l577_577641


namespace percentage_saved_each_month_l577_577449

-- Define the annual salary and monthly savings as constants
def annualSalary : ℝ := 48000
def monthlySavings : ℝ := 400

-- Define the monthly salary as being the annual salary divided by 12
def monthlySalary : ℝ := annualSalary / 12

-- State the theorem that the percentage of salary saved each month is 10%
theorem percentage_saved_each_month (h1 : annualSalary = 48000) (h2 : monthlySavings = 400) :
  (monthlySavings / monthlySalary) * 100 = 10 :=
by
  sorry

end percentage_saved_each_month_l577_577449


namespace seating_arrangements_around_round_table_l577_577570

theorem seating_arrangements_around_round_table (n : ℕ) (h : n = 12) : 
  (12! / 12) = 11! :=
by sorry

end seating_arrangements_around_round_table_l577_577570


namespace max_subset_property_l577_577273

def I (n : ℕ) : set ℕ := { i | 1 ≤ i ∧ i ≤ n }

noncomputable def max_subset_card (n : ℕ) : ℕ :=
  ⌊ (n + 2) / 4 ⌋

theorem max_subset_property (n : ℕ) :
  ∃ (S : set ℕ), S ⊆ I n ∧ (∀ a b ∈ S, ¬ (a ≠ b ∧ a ∣ b)) ∧ S.card = max_subset_card n :=
by
  sorry

end max_subset_property_l577_577273


namespace grants_test_score_l577_577202

theorem grants_test_score :
  ∀ (hunter_score : ℕ) (john_score : ℕ) (grant_score : ℕ), hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 :=
by
  intro hunter_score john_score grant_score
  intro hunter_eq john_eq grant_eq
  rw [hunter_eq, john_eq, grant_eq]
  sorry

end grants_test_score_l577_577202


namespace sufficient_but_not_necessary_condition_l577_577540

theorem sufficient_but_not_necessary_condition 
  (α : ℝ) 
  (h : sin (α + π / 2) = 1 / 4) : |cos α| = 1 / 4 :=
by 
  -- We don't need the proof steps, just the statement
  sorry

end sufficient_but_not_necessary_condition_l577_577540


namespace sum_of_digits_of_x_l577_577279

-- Condition: x is a three-digit palindrome
def is_palindrome (x : ℕ) : Prop := 
  x ≥ 100 ∧ x < 1000 ∧ toString x = toString x.reverse

-- Condition: x + 54 is a four-digit palindrome
def is_four_digit_palindrome (x : ℕ) : Prop :=
  x + 54 ≥ 1000 ∧ x + 54 < 10000 ∧ 
  let y := x + 54 in toString y = toString y.reverse

-- Proof statement: The sum of the digits of x is 20
theorem sum_of_digits_of_x (x : ℕ) (h1 : is_palindrome x) (h2 : is_four_digit_palindrome x) : 
  let digits_sum (n : ℕ) : ℕ := (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) 
  digits_sum x = 20 := 
sorry

end sum_of_digits_of_x_l577_577279


namespace gcd_of_A_B_l577_577225

noncomputable def A (k : ℕ) := 2 * k
noncomputable def B (k : ℕ) := 5 * k

theorem gcd_of_A_B (k : ℕ) (h_lcm : Nat.lcm (A k) (B k) = 180) : Nat.gcd (A k) (B k) = 18 :=
by
  sorry

end gcd_of_A_B_l577_577225


namespace complement_of_B_in_A_l577_577992

def complement (A B : Set Int) := { x ∈ A | x ∉ B }

theorem complement_of_B_in_A (A B : Set Int) (a : Int) (h1 : A = {2, 3, 4}) (h2 : B = {a + 2, a}) (h3 : A ∩ B = B)
: complement A B = {3} :=
  sorry

end complement_of_B_in_A_l577_577992


namespace total_distance_maria_l577_577901

theorem total_distance_maria (D : ℝ)
  (half_dist : D/2 + (D/2 - D/8) + 180 = D) :
  3 * D / 8 = 180 → 
  D = 480 :=
by
  sorry

end total_distance_maria_l577_577901


namespace find_number_l577_577403

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l577_577403


namespace no_real_solutions_l577_577447

def equation_has_no_roots : Prop :=
  ∀ x : ℝ, ¬ (x + 2 * (sqrt (x - 5)) = 6)

theorem no_real_solutions : equation_has_no_roots := 
begin
  sorry
end

end no_real_solutions_l577_577447


namespace area_of_triangle_bounded_by_lines_l577_577775

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l577_577775


namespace campers_afternoon_l577_577396

def morning_campers : ℕ := 52
def additional_campers : ℕ := 9
def total_campers_afternoon : ℕ := morning_campers + additional_campers

theorem campers_afternoon : total_campers_afternoon = 61 :=
by
  sorry

end campers_afternoon_l577_577396


namespace theta_value_l577_577493

theorem theta_value (θ : ℝ) (h1 : sin (π + θ) = -sqrt 3 * cos (2 * π - θ)) (h2 : abs θ < π / 2) : 
  θ = π / 3 :=
by
  sorry

end theta_value_l577_577493


namespace number_of_zeros_of_g_l577_577969

noncomputable def f (x a : ℝ) := Real.exp x * (x + a)

noncomputable def g (x a : ℝ) := f (x - a) a - x^2

theorem number_of_zeros_of_g (a : ℝ) :
  (if a < 1 then ∃! x, g x a = 0
   else if a = 1 then ∃! x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0
   else ∃! x1 x2 x3, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0) := sorry

end number_of_zeros_of_g_l577_577969


namespace find_two_digit_number_l577_577044

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l577_577044


namespace knights_count_is_two_l577_577644

def inhabitant (i : Fin 5) : Prop :=
  (i = 0 → inhabitants_truth 1) ∧
  (i = 1 → inhabitants_truth 2) ∧
  (i = 2 → inhabitants_truth 3) ∧
  (i = 3 → ∀ k, k ≠ 3 → ¬inhabitants_truth k) ∧
  (i = 4 → inhabitants_truth 4 ≠ inhabitants_truth 3)

def inhabitants_truth (n : ℕ) : Prop := sorry

theorem knights_count_is_two : ∃ (knights : Fin 5 → Prop), (∃! i, knights i) ∧ (∃! j, knights j) ∧ (i ≠ j) :=
sorry

end knights_count_is_two_l577_577644


namespace find_x_from_area_l577_577927

theorem find_x_from_area (x : ℝ) (hx : x > 0)
  (harea : 1 / 2 * x * (3 * x) = 120) : x = 4 * real.sqrt 5 := by
  sorry

end find_x_from_area_l577_577927


namespace find_m_minus_n_l577_577961

noncomputable def poly : ℕ → ℕ → Polynomial ℝ :=
  λ m n, 4 * Polynomial.monomial 2 1 
    - 3 * Polynomial.monomial (m + 1) 1 * Polynomial.monomial 1 1
    - Polynomial.monomial 1 1

theorem find_m_minus_n (m n : ℕ) (hdeg : Polynomial.total_degree (poly m n) = 5)
  (hcoeff : n = -3) : m - n = 6 :=
by sorry

end find_m_minus_n_l577_577961


namespace continuous_stripe_probability_l577_577456

noncomputable def probability_continuous_stripe : ℚ :=
let total_combinations := 2^6 in
-- 3 pairs of parallel faces; for each pair, 4 favorable configurations.
let favorable_outcomes := 3 * 4 in
favorable_outcomes / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l577_577456


namespace gopi_salary_turbans_l577_577199

-- Define the question and conditions as statements
def total_salary (turbans : ℕ) : ℕ := 90 + 30 * turbans
def servant_receives : ℕ := 60 + 30
def fraction_annual_salary : ℚ := 3 / 4

-- The theorem statement capturing the equivalent proof problem
theorem gopi_salary_turbans (T : ℕ) 
  (salary_eq : total_salary T = 90 + 30 * T)
  (servant_eq : servant_receives = 60 + 30)
  (fraction_eq : fraction_annual_salary = 3 / 4)
  (received_after_9_months : ℚ) :
  fraction_annual_salary * (90 + 30 * T : ℚ) = received_after_9_months → 
  received_after_9_months = 90 →
  T = 1 :=
sorry

end gopi_salary_turbans_l577_577199


namespace remainder_of_55_power_55_plus_55_l577_577541

-- Define the problem statement using Lean

theorem remainder_of_55_power_55_plus_55 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  sorry

end remainder_of_55_power_55_plus_55_l577_577541


namespace area_under_arccos_cos_eq_l577_577129

noncomputable def area_under_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..(2 * Real.pi), Real.arccos (Real.cos x)

theorem area_under_arccos_cos_eq :
  area_under_arccos_cos = Real.pi^2 := by
  sorry

end area_under_arccos_cos_eq_l577_577129


namespace triangle_area_bounded_by_lines_l577_577780

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l577_577780


namespace triangle_area_bounded_by_lines_l577_577778

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l577_577778


namespace triangle_area_bounded_by_lines_l577_577755

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l577_577755


namespace basic_astrophysics_degrees_l577_577404

def percentages : List ℚ := [12, 22, 14, 27, 7, 5, 3, 4]

def total_budget_percentage : ℚ := 100

def degrees_in_circle : ℚ := 360

def remaining_percentage (lst : List ℚ) (total : ℚ) : ℚ :=
  total - lst.sum / 100  -- convert sum to percentage

def degrees_of_percentage (percent : ℚ) (circle_degrees : ℚ) : ℚ :=
  percent * (circle_degrees / total_budget_percentage) -- conversion rate per percentage point

theorem basic_astrophysics_degrees :
  degrees_of_percentage (remaining_percentage percentages total_budget_percentage) degrees_in_circle = 21.6 :=
by
  sorry

end basic_astrophysics_degrees_l577_577404


namespace first_term_of_arithmetic_seq_l577_577606

theorem first_term_of_arithmetic_seq (S : ℕ → ℚ) (a : ℚ) (d : ℚ) (h_d : d = 4)
  (hS_n : ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2)
  (h_ratio_const : ∀ n, (S (2 * n)) / (S n) = 4)
  : a = 2 :=
by
  sorry

end first_term_of_arithmetic_seq_l577_577606


namespace find_two_digit_number_l577_577032

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l577_577032


namespace sequence_evaluation_l577_577852

noncomputable def a (i : ℕ) : ℤ :=
if h : i ≤ 4 then (i ^ 2 : ℤ)
else (finset.prod (finset.range (i - 1)) a - i : ℤ)

theorem sequence_evaluation : 
  (finset.prod (finset.range 100) a) - (finset.sum (finset.range 100) (λ i, a i ^ 2)) = -5388 := 
by
  sorry

end sequence_evaluation_l577_577852


namespace number_of_three_digit_integers_l577_577968

-- Defining the set of available digits
def digits : List ℕ := [3, 5, 8, 9]

-- Defining the property for selecting a digit without repetition
def no_repetition (l : List ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ l → l.filter (fun x => x = d) = [d]

-- The main theorem stating the number of three-digit integers that can be formed
theorem number_of_three_digit_integers (h : no_repetition digits) : 
  ∃ n : ℕ, n = 24 :=
by
  sorry

end number_of_three_digit_integers_l577_577968


namespace a1_b1_sum_l577_577993

-- Definitions from the conditions:
def strict_inc_seq (s : ℕ → ℕ) : Prop := ∀ n, s n < s (n + 1)

def positive_int_seq (s : ℕ → ℕ) : Prop := ∀ n, s n > 0

def a : ℕ → ℕ := sorry -- Define the sequence 'a' (details skipped).

def b : ℕ → ℕ := sorry -- Define the sequence 'b' (details skipped).

-- Conditions given:
axiom cond_a_inc : strict_inc_seq a

axiom cond_b_inc : strict_inc_seq b

axiom cond_a_pos : positive_int_seq a

axiom cond_b_pos : positive_int_seq b

axiom cond_a10_b10_lt_2017 : a 10 = b 10 ∧ a 10 < 2017

axiom cond_a_rec : ∀ n, a (n + 2) = a (n + 1) + a n

axiom cond_b_rec : ∀ n, b (n + 1) = 2 * b n

-- The theorem to prove:
theorem a1_b1_sum : a 1 + b 1 = 5 :=
sorry

end a1_b1_sum_l577_577993


namespace S_nk_less_S_n_plus_S_k_l577_577933

def S (m : ℕ) : ℝ :=
  if h : m < 3 then 0
  else 1 + ∑ i in (finset.range (m + 1)).filter (λ x, x ≠ 2), (1 : ℝ) / i

theorem S_nk_less_S_n_plus_S_k (n k : ℕ) (hn : n ≥ 3) (hk : k ≥ 3) :
  S (n * k) < S n + S k :=
by { sorry }

end S_nk_less_S_n_plus_S_k_l577_577933


namespace abs_w_of_square_l577_577323

theorem abs_w_of_square (w : ℂ) (h : w^2 = -48 + 14 * complex.i) : complex.abs w = 5 * real.sqrt 2 :=
sorry

end abs_w_of_square_l577_577323


namespace jeanne_additional_tickets_l577_577588

-- Define the costs
def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def jeanne_tickets : ℕ := 5

-- Calculate the total cost
def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

-- Define the proof problem
theorem jeanne_additional_tickets : total_cost - jeanne_tickets = 8 :=
by sorry

end jeanne_additional_tickets_l577_577588


namespace greg_ate_4_halves_l577_577999

def greg_ate_halves (total_cookies : ℕ) (brad_halves : ℕ) (left_halves : ℕ) : ℕ :=
  2 * total_cookies - (brad_halves + left_halves)

theorem greg_ate_4_halves : greg_ate_halves 14 6 18 = 4 := by
  sorry

end greg_ate_4_halves_l577_577999


namespace Owen_spent_720_dollars_on_burgers_l577_577142

def days_in_June : ℕ := 30
def burgers_per_day : ℕ := 2
def cost_per_burger : ℕ := 12

def total_burgers (days : ℕ) (burgers_per_day : ℕ) : ℕ :=
  days * burgers_per_day

def total_cost (burgers : ℕ) (cost_per_burger : ℕ) : ℕ :=
  burgers * cost_per_burger

theorem Owen_spent_720_dollars_on_burgers :
  total_cost (total_burgers days_in_June burgers_per_day) cost_per_burger = 720 := by
  sorry

end Owen_spent_720_dollars_on_burgers_l577_577142


namespace no_positive_integers_sum_to_one_l577_577306

theorem no_positive_integers_sum_to_one (m : ℕ) (h_m : 2 ≤ m) :
  ¬ ∃ (x : Fin m → ℕ), (∀ i j : Fin m, i < j → x i < x j) ∧
  (∑ i, (1 / (x i)^3 : ℚ) = 1) :=
begin
  sorry
end

end no_positive_integers_sum_to_one_l577_577306


namespace max_k_property_l577_577134

-- Define the set S as the set of integers from 1 to 2017
def S : Finset ℕ := (Finset.range 2018).filter (λ x, x > 0)

-- Define the property P that no number in the subset can be a power of any other number
def P (k : ℕ) (subset : Finset ℕ) : Prop :=
  (∀ (x ∈ subset) (y ∈ subset), x ≠ y → ∀ (n : ℕ), y ≠ x ^ n)

-- Define the main theorem statement
theorem max_k_property :
  ∃ k : ℕ, k = 1974 ∧ ∃ subset : Finset ℕ, subset.card = k ∧ subset ⊆ S ∧ P k subset :=
by
  sorry

end max_k_property_l577_577134


namespace solution_statement_l577_577177

open Complex

noncomputable def problem_statement : Prop :=
  ∃ z : ℂ, (z * Complex.i = 2 - Complex.i ∧ 
            ∀ part : ℂ, part = conj(z) →
                         (part.re < 0 ∧ part.im > 0))

theorem solution_statement : problem_statement :=
by
  -- provide logical steps here
  sorry

end solution_statement_l577_577177


namespace ratio_S1_S2_l577_577476

theorem ratio_S1_S2 : 
  let S1 := ∑ k in finset.range 2019, (-1) ^ (k + 1) * (1 / 3 ^ (k + 1))
  let S2 := ∑ k in finset.range 2019, (-1) ^ (k + 1) * (1 / 3 ^ (k + 1))
  S1 / S2 = 1 := by 
  sorry

end ratio_S1_S2_l577_577476


namespace area_of_triangle_l577_577762

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l577_577762


namespace range_of_a_l577_577006

-- Defining the sets A and B based on the conditions
def setA (a : ℝ) : set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def setB (a : ℝ) : set ℝ := {x | x ≥ a - 1}

-- Statement of the theorem in Lean 4
theorem range_of_a (a : ℝ) : (setA a ∪ setB a = set.univ) ↔ a ∈ set.Iic 2 := 
by sorry

end range_of_a_l577_577006


namespace find_width_of_bobs_tv_l577_577436

def area (w h : ℕ) : ℕ := w * h

def weight_in_oz (area : ℕ) : ℕ := area * 4

def weight_in_lb (weight_in_oz : ℕ) : ℕ := weight_in_oz / 16

def width_of_bobs_tv (x : ℕ) : Prop :=
  area 48 100 = 4800 ∧
  weight_in_lb (weight_in_oz (area 48 100)) = 1200 ∧
  weight_in_lb (weight_in_oz (area x 60)) = 15 * x ∧
  15 * x = 1350

theorem find_width_of_bobs_tv : ∃ x : ℕ, width_of_bobs_tv x := sorry

end find_width_of_bobs_tv_l577_577436


namespace water_bottles_needed_l577_577461

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l577_577461


namespace maria_needs_nuts_l577_577628

theorem maria_needs_nuts (total_cookies nuts_per_cookie : ℕ) 
  (nuts_fraction : ℚ) (chocolate_fraction : ℚ) 
  (H1 : nuts_fraction = 1 / 4) 
  (H2 : chocolate_fraction = 0.4) 
  (H3 : total_cookies = 60) 
  (H4 : nuts_per_cookie = 2) :
  (total_cookies * nuts_fraction + (total_cookies - total_cookies * nuts_fraction - total_cookies * chocolate_fraction) * nuts_per_cookie) = 72 := 
by
  sorry

end maria_needs_nuts_l577_577628


namespace brandon_textbooks_weight_l577_577259

-- Define the weights of Jon's textbooks
def jon_textbooks : List ℕ := [2, 8, 5, 9]

-- Define the weight ratio between Jon's and Brandon's textbooks
def weight_ratio : ℕ := 3

-- Define the total weight of Jon's textbooks
def weight_jon : ℕ := jon_textbooks.sum

-- Define the weight of Brandon's textbooks to be proven
def weight_brandon : ℕ := weight_jon / weight_ratio

-- The theorem to be proven
theorem brandon_textbooks_weight : weight_brandon = 8 :=
by sorry

end brandon_textbooks_weight_l577_577259


namespace total_water_bottles_needed_l577_577468

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l577_577468


namespace value_of_x_l577_577557

variable {x y z : ℤ}

theorem value_of_x
  (h1 : x + y = 31)
  (h2 : y + z = 47)
  (h3 : x + z = 52)
  (h4 : y + z = x + 16) :
  x = 31 := by
  sorry

end value_of_x_l577_577557


namespace positive_integer_sum_form_l577_577312

noncomputable def sum_of_form (n : ℕ) : Prop :=
  ∃ (terms : List ℕ), (∀ t ∈ terms, ∃ (a b : ℕ), t = 2^a * 3^b) ∧
  ∑ t in terms, t = n ∧
  ∀ i j ∈ terms, i ≠ j → ¬ (i ∣ j)

theorem positive_integer_sum_form (n : ℕ) (h : n > 0) : sum_of_form n :=
sorry

end positive_integer_sum_form_l577_577312


namespace lemonade_price_on_hot_day_l577_577309

theorem lemonade_price_on_hot_day
  (profit : ℝ)
  (cost_per_cup : ℝ)
  (cups_per_day : ℕ)
  (hot_days : ℕ)
  (total_days : ℕ)
  (hot_day_multiplier : ℝ)
  (final_profit : ℝ) :
  profit = 2.18 →
  cost_per_cup = 0.75 →
  cups_per_day = 32 →
  hot_days = 4 →
  total_days = 10 →
  hot_day_multiplier = 1.25 →
  final_profit = 350 →
  let P := (final_profit + (total_days - hot_days) * cups_per_day * cost_per_cup) / (total_days * cups_per_day) in
  let P_hot := hot_day_multiplier * P in
  P_hot ≈ profit :=
by
  sorry

end lemonade_price_on_hot_day_l577_577309


namespace symmetrical_point_wrt_x_axis_l577_577243

theorem symmetrical_point_wrt_x_axis (x y : ℝ) (P_symmetrical : (ℝ × ℝ)) (hx : x = -1) (hy : y = 2) : 
  P_symmetrical = (x, -y) → P_symmetrical = (-1, -2) :=
by
  intros h
  rw [hx, hy] at h
  exact h

end symmetrical_point_wrt_x_axis_l577_577243


namespace white_tshirts_per_package_l577_577380

theorem white_tshirts_per_package (p t : ℕ) (h1 : p = 28) (h2 : t = 56) :
  t / p = 2 :=
by 
  sorry

end white_tshirts_per_package_l577_577380


namespace magnitude_of_z5_l577_577107

def z (n : ℕ) : ℂ :=
  if h : n = 1 then 1
  else if m : 1 ≤ n then (z (n-1))^2 * (1 + complex.i)
  else 0

theorem magnitude_of_z5 : complex.abs (z 5) = 128 * real.sqrt 2 := 
by 
  sorry

end magnitude_of_z5_l577_577107


namespace Annie_cookies_sum_l577_577072

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end Annie_cookies_sum_l577_577072


namespace Dave_deleted_apps_l577_577889

theorem Dave_deleted_apps : 
  ∀ (a b d : ℝ), 
    a = 300.5 → 
    b = 129.5 → 
    d = a - b → 
    d = 171 :=
by
  intros a b d h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end Dave_deleted_apps_l577_577889


namespace g_neither_even_nor_odd_l577_577450

noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (4 + x^2))

theorem g_neither_even_nor_odd : ∀ x : ℝ, g (-x) ≠ g x ∧ g (-x) ≠ -g x := by
  intro x
  let gx := Real.log (x + Real.sqrt (4 + x^2))
  let g_minus_x := Real.log (-x + Real.sqrt (4 + x^2))
  have h1 : g_minus_x ≠ gx := by sorry
  have h2 : g_minus_x ≠ -gx := by sorry
  exact ⟨h1, h2⟩

end g_neither_even_nor_odd_l577_577450


namespace solve_eq1_solve_eq2_l577_577677

-- Define the first problem statement and the correct answers
theorem solve_eq1 (x : ℝ) (h : (x - 2) ^ 2 = 169) : x = 15 ∨ x = -11 := 
  by sorry

-- Define the second problem statement and the correct answer
theorem solve_eq2 (x : ℝ) (h : 3 * (x - 3) ^ 3 - 24 = 0) : x = 5 := 
  by sorry

end solve_eq1_solve_eq2_l577_577677


namespace find_A_l577_577355

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 :=
by
  sorry

end find_A_l577_577355


namespace index_commutator_subgroup_is_even_l577_577575

open Finite FiniteGroup Subgroup

variable (G : Type*) [Group G] [Fintype G] [DecidableEq G]

/-- Let G' be the commutator subgroup of G. Assume |G'| = 2. 
    Prove that the index |G : G'| is even. -/
theorem index_commutator_subgroup_is_even (G' : Subgroup G) [IsNormalSubgroup G'] (hG' : Fintype.card G' = 2) :
  Even (Fintype.card G / Fintype.card G') :=
sorry

end index_commutator_subgroup_is_even_l577_577575


namespace kite_diagonals_perpendicular_l577_577809

-- Define what it means to be a kite
structure IsKite (a b c d : ℝ) : Prop :=
(equiv1 : a = b)
(equiv2 : c = d)
(diagonals_perpendicular : ⊥ (a + b, c + d))

-- State the problem
theorem kite_diagonals_perpendicular (a b c d : ℝ) (h1 : IsKite a b c d) :
  ⊥ (a + b, c + d) :=
  h1.diagonals_perpendicular

end kite_diagonals_perpendicular_l577_577809


namespace part_a_part_b_l577_577026

def is_good (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k ∧ ∃ (s : finset ℕ), s.card.even ∧ s.prod = x

def m (x a b : ℕ) := (x + a) * (x + b)

theorem part_a : 
  ∃ (a b : ℕ), ∀ x, 1 ≤ x ∧ x ≤ 2010 → is_good (m x a b) := 
sorry

theorem part_b :
  (∀ x : ℕ, is_good (m x a a)) → ∀ (a b : ℕ), (∀ x : ℕ, is_good (m x a b)) → a = b := 
sorry

end part_a_part_b_l577_577026


namespace line_passes_through_fixed_point_l577_577692

theorem line_passes_through_fixed_point (k : ℝ) : ∀ k : ℝ, (∃ (x y : ℝ), y = k * (x - 2) + 3 ∧ x = 2 ∧ y = 3) :=
by {
  intros k,
  use (2 : ℝ),
  use (3 : ℝ),
  split,
  {
    rw [mul_sub, mul_zero, zero_add],
    exact rfl,
  },
  {
    split, 
    { exact rfl }, 
    { exact rfl }
  }
}

end line_passes_through_fixed_point_l577_577692


namespace sqrt_product_simplification_l577_577673

theorem sqrt_product_simplification :
  Real.sqrt 18 * Real.sqrt 32 = 24 := by
  have h1 : 18 = 2 * 3^2 := by sorry
  have h2 : 32 = 2^5 := by sorry
  have h3 : Real.sqrt (2 * 3^2) = Real.sqrt 18 := by sorry
  have h4 : Real.sqrt (2^5) = Real.sqrt 32 := by sorry
  have h5 : Real.sqrt 18 * Real.sqrt 32 = Real.sqrt (18 * 32) := Real.sqrt_mul (by norm_num) (by norm_num)
  rw [mul_comm] at h5
  rw [←h5]
  have h6 : 18 * 32 = 2^6 * 3^2 := by
    rw [h1, h2]
    ring
  rw [h6]
  have h7 : Real.sqrt (2^6 * 3^2) = Real.sqrt (8^2 * 3^2) := by
    norm_num
  rw [h7]
  have h8 : Real.sqrt (8^2 * 3^2) = 8 * 3 := Real.sqrt_mul (by norm_num) (by norm_num)
  rw [h8]
  norm_num

end sqrt_product_simplification_l577_577673


namespace solve_equation_l577_577940

noncomputable def maxRational (x y : ℚ) : ℚ :=
  if x > y then x else y

theorem solve_equation (x : ℚ) : maxRational x (-x) = 2 * x + 9 ↔ x = -3 :=
begin
  sorry
end

end solve_equation_l577_577940


namespace number_of_action_figures_bought_l577_577903

-- Definitions of conditions
def cost_of_board_game : ℕ := 2
def cost_per_action_figure : ℕ := 7
def total_spent : ℕ := 30

-- The problem to prove
theorem number_of_action_figures_bought : 
  ∃ (n : ℕ), total_spent - cost_of_board_game = n * cost_per_action_figure ∧ n = 4 :=
by
  sorry

end number_of_action_figures_bought_l577_577903


namespace teresa_ahmad_equation_l577_577679

theorem teresa_ahmad_equation (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 7 ∨ x = 1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = 1) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end teresa_ahmad_equation_l577_577679


namespace evaluate_powers_of_i_l577_577907

theorem evaluate_powers_of_i :
  (Complex.I ^ 50) + (Complex.I ^ 105) = -1 + Complex.I :=
by 
  sorry

end evaluate_powers_of_i_l577_577907


namespace download_time_is_2_hours_l577_577100

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l577_577100


namespace find_first_term_and_difference_l577_577718

-- Define the two conditions given in the problem.
def a3 : ℕ → ℝ := λ n, 3
def a11 : ℕ → ℝ := λ n, 15

-- Define the arithmetic sequence using the formula a_n = a_1 + (n-1) * d
def arithmetic_seq (a1 d : ℝ) : ℕ → ℝ := λ n, a1 + (n - 1) * d

-- State the theorem with the conditions and the goal (proof not provided, using sorry).
theorem find_first_term_and_difference :
  (∃ a1 d : ℝ, arithmetic_seq a1 d 3 = 3 ∧ arithmetic_seq a1 d 11 = 15 ∧ a1 = 0 ∧ d = 3 / 2) := 
sorry

end find_first_term_and_difference_l577_577718


namespace vertices_form_rectangle_l577_577506

noncomputable def rectangle_condition (t1 t2 t3 t4 : ℂ) : Prop :=
  (|t1| = |t2| ∧ |t2| = |t3| ∧ |t3| = |t4| ∧ |t4| > 0) ∧ (t1 + t2 + t3 + t4 = 0)

theorem vertices_form_rectangle 
  (t1 t2 t3 t4 : ℂ)
  (h : rectangle_condition t1 t2 t3 t4) : 
  is_rectangle {A B C D : ℝ}
  sorry

end vertices_form_rectangle_l577_577506


namespace value_range_cosine_function_l577_577359

open Real

theorem value_range_cosine_function (x : ℝ) (hx : cos x ∈ Icc (-1 : ℝ) 1) :
  ∃ (y_range : set ℝ), y_range = Icc (-7) 9 ∧ ∀ y, y = cos (2 * x) - 8 * cos x → y ∈ y_range := by sorry

end value_range_cosine_function_l577_577359


namespace triangle_area_bounded_by_lines_l577_577756

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l577_577756


namespace speed_of_train_l577_577388

-- Define the given conditions
def length_of_bridge : ℝ := 200
def length_of_train : ℝ := 100
def time_to_cross_bridge : ℝ := 60

-- Define the speed conversion factor
def m_per_s_to_km_per_h : ℝ := 3.6

-- Prove that the speed of the train is 18 km/h
theorem speed_of_train :
  (length_of_bridge + length_of_train) / time_to_cross_bridge * m_per_s_to_km_per_h = 18 :=
by
  sorry

end speed_of_train_l577_577388


namespace lower_upper_bound_f_l577_577314

-- definition of the function f(n, d) as given in the problem
def func_f (n : ℕ) (d : ℕ) : ℕ :=
  -- placeholder definition; actual definition would rely on the described properties
  sorry

theorem lower_upper_bound_f (n d : ℕ) (hn : 0 < n) (hd : 0 < d) :
  (n-1) * 2^d + 1 ≤ func_f n d ∧ func_f n d ≤ (n-1) * n^d + 1 :=
by
  sorry

end lower_upper_bound_f_l577_577314


namespace compare_means_l577_577678

theorem compare_means (a b c : ℝ) (h_pos_abc : 0 < a ∧ 0 < b ∧ 0 < c) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (real.sqrt (a * b)) > (real.sqrt (a * b * c))^(1/3) ∧ (real.sqrt (a * b * c))^(1/3) > (2 * b * c / (b + c)) := by
  sorry

end compare_means_l577_577678


namespace unique_zero_of_f_max_integer_a_for_increasing_g_l577_577181

noncomputable def f (x : ℝ) := (x-2) * Real.log x + 2 * x - 3

noncomputable def g (a x : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem unique_zero_of_f :
  ∃! x : ℝ, 1 ≤ x ∧ f x = 0 :=
sorry

theorem max_integer_a_for_increasing_g :
  ∀ a : ℕ, (∀ x : ℝ, 1 ≤ x → (1:ℝ) ≤ g' a x) → a ≤ 6 :=
sorry

noncomputable def g' (a x : ℝ) := Real.log x + 1 - a / x + a / (x ^ 2)

end unique_zero_of_f_max_integer_a_for_increasing_g_l577_577181


namespace Q_plus_S_value_l577_577615

noncomputable def complex_conjugate (z : ℂ) : ℂ := z.conj

noncomputable def g (z : ℂ) : ℂ := -3 * complex.I * (complex_conjugate z)

-- Polynomial R(z) = z^4 + 3z^3 + 5z^2 + z + 2
def R : Polynomial ℂ := polynomial.C 2 + polynomial.X + polynomial.C 5 * polynomial.X^2 + polynomial.C 3 * polynomial.X^3 + polynomial.X^4

-- Roots of R(z)
variable (z1 z2 z3 z4 : ℂ)
-- Assume z1, z2, z3, z4 are roots of R(z)
axiom root_of_R1 : polynomial.eval z1 R = 0
axiom root_of_R2 : polynomial.eval z2 R = 0
axiom root_of_R3 : polynomial.eval z3 R = 0
axiom root_of_R4 : polynomial.eval z4 R = 0

-- Polynomial S(z) = z^4 + Pz^3 + Qz^2 + Rz + S with roots g(z1), g(z2), g(z3), g(z4)
noncomputable def S : Polynomial ℂ := polynomial.C 0 + polynomial.X

theorem Q_plus_S_value : ∃ (Q S : ℂ), (polynomial.X^4 + polynomial.C Q * polynomial.X^2 + polynomial.C S).roots = [g z1, g z2, g z3, g z4] ∧ Q + S = 117 := sorry

end Q_plus_S_value_l577_577615


namespace find_d_l577_577270

noncomputable def polynomial_roots_neg_int (g : Polynomial ℝ) : Prop :=
  ∃ s1 s2 s3 s4 : ℝ, s1 > 0 ∧ s2 > 0 ∧ s3 > 0 ∧ s4 > 0 ∧
    g = Polynomial.C (s1 * s2 * s3 * s4) * (Polynomial.X + (-s1)) *
          (Polynomial.X + (-s2)) * (Polynomial.X + (-s3)) * (Polynomial.X + (-s4))

theorem find_d 
  (g : Polynomial ℝ) 
  (a b c d : ℝ)
  (h_g : g = Polynomial.mk [d, c, b, a, 1]) 
  (h_roots : polynomial_roots_neg_int g)
  (h_sum : a + b + c + d = 2003) :
  d = 1992 :=
  sorry

end find_d_l577_577270


namespace rolls_sold_to_grandmother_l577_577932

theorem rolls_sold_to_grandmother (t u n s g : ℕ) 
  (h1 : t = 45)
  (h2 : u = 10)
  (h3 : n = 6)
  (h4 : s = 28)
  (total_sold : t - s = g + u + n) : 
  g = 1 := 
  sorry

end rolls_sold_to_grandmother_l577_577932


namespace conjugate_in_fourth_quadrant_l577_577703

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Given complex number
def z : ℂ := ⟨5, 3⟩

-- Conjugate of z
def z_conjugate : ℂ := complex_conjugate z

-- Cartesian coordinates of the conjugate
def z_conjugate_coordinates : ℝ × ℝ := (z_conjugate.re, z_conjugate.im)

-- Definition of the Fourth Quadrant
def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem conjugate_in_fourth_quadrant :
  is_in_fourth_quadrant z_conjugate_coordinates :=
by sorry

end conjugate_in_fourth_quadrant_l577_577703


namespace maximum_value_exists_l577_577507

noncomputable def polynomial_max_value 
  (a b c x1 x2 x3 : ℝ) (λ : ℝ) 
  (λ_pos : 0 < λ)
  (poly_roots : (x - x1) * (x - x2) * (x - x3) = x^3 + a * x^2 + b * x + c)
  (condition1 : x2 - x1 = λ)
  (condition2 : x3 > (1 / 2) * (x1 + x2)) : 
  Prop := (∃ a b c x1 x2 x3 λ, 0 < λ ∧
  (x - x1) * (x - x2) * (x - x3) = x^3 + a * x^2 + b * x + c ∧
  x2 - x1 = λ ∧
  x3 > (1 / 2) * (x1 + x2) ∧
  2 * a^3 + 27 * c - 9 * a * b = 3 * sqrt(3) * λ^3)

theorem maximum_value_exists 
  (a b c x1 x2 x3 : ℝ) (λ : ℝ) 
  (λ_pos : 0 < λ)
  (poly_roots : (x - x1) * (x - x2) * (x - x3) = x^3 + a * x^2 + b * x + c)
  (condition1 : x2 - x1 = λ)
  (condition2 : x3 > (1 / 2) * (x1 + x2)) : 
  polynomial_max_value a b c x1 x2 x3 λ :=
sorry

end maximum_value_exists_l577_577507


namespace max_value_l577_577616

noncomputable def max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x + y + z = 8) : ℝ :=
  \sqrt{3x + 1} + \sqrt{3y + 1} + \sqrt{3z + 1}

theorem max_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x + y + z = 8) :
  max_value_of_expression x y z hx hy hz h ≤ 3 * \sqrt{3} := sorry

end max_value_l577_577616


namespace area_A_l577_577886

variables (A B C D A' B' C' D' : Type)
variables [convex_quadrilateral ABCD]
variables [extended_point A B' (6: ℝ)]
variables [extended_point B C' (7: ℝ)]
variables [extended_point C D' (8: ℝ)]
variables [extended_point D A' (9: ℝ)]
variables (area_ABCD : ℝ)
variables (area_A'B'C'D' : ℝ)

axiom quadrilateral_area (h₁ : convex_quadrilateral ABCD) 
                         (h₂ : extended_point A B' 6)
                         (h₃ : extended_point B C' 7)
                         (h₄ : extended_point C D' 8)
                         (h₅ : extended_point D A' 9)
                         (area_ABCD : area_of_quadrilateral ABCD = 10) :
                         area_of_quadrilateral A'B'C'D' = 50

theorem area_A'B'C'D'_correct : area_of_quadrilateral A'B'C'D' = 50 := sorry

end area_A_l577_577886


namespace sophie_marbles_probability_l577_577321

theorem sophie_marbles_probability :
  let blue_marbles := 10
  let red_marbles := 5
  let total_marbles := blue_marbles + red_marbles
  let withdraws := 8
  let exact_four_blue : ℝ :=
    (nat.choose withdraws 4).to_real *
    ((blue_marbles / total_marbles) ^ 4 * (red_marbles / total_marbles) ^ 4)
  let no_red : ℝ := (blue_marbles / total_marbles) ^ withdraws
  ((exact_four_blue * 1000.0).round / 1000.0) - ((no_red * 1000.0).round / 1000.0) = 0.131 :=
by
  sorry

end sophie_marbles_probability_l577_577321


namespace simplify_expression_l577_577453

theorem simplify_expression :
  (-2) ^ 2006 + (-1) ^ 3007 + 1 ^ 3010 - (-2) ^ 2007 = -2 ^ 2006 := 
sorry

end simplify_expression_l577_577453


namespace num_red_balls_l577_577569

theorem num_red_balls (x : ℕ) (h1 : 60 = 60) (h2 : (x : ℝ) / (x + 60) = 0.25) : x = 20 :=
sorry

end num_red_balls_l577_577569


namespace part1_part2_l577_577287

section
variable (a x : ℝ)

def f (a x : ℝ) : ℝ := |x + 1/a| + |x - a|
def g (a : ℝ) : Prop := ∀ x, f a x ≥ 2

-- Proof for (1)
theorem part1 (h : a > 0) : g a :=
sorry

-- Proof for (2)
theorem part2 (h : f a 2 < 4) : 1 < a ∧ a < 2 + Real.sqrt 3 :=
sorry

end

end part1_part2_l577_577287


namespace total_earnings_l577_577902

def num_members : ℕ := 20
def candy_bars_per_member : ℕ := 8
def cost_per_candy_bar : ℝ := 0.5

theorem total_earnings :
  (num_members * candy_bars_per_member * cost_per_candy_bar) = 80 :=
by
  sorry

end total_earnings_l577_577902


namespace total_turnips_l577_577260

-- Conditions
def turnips_keith : ℕ := 6
def turnips_alyssa : ℕ := 9

-- Statement to be proved
theorem total_turnips : turnips_keith + turnips_alyssa = 15 := by
  -- Proof is not required for this prompt, so we use sorry
  sorry

end total_turnips_l577_577260


namespace dog_treats_cost_l577_577595

theorem dog_treats_cost
  (treats_per_day : ℕ)
  (cost_per_treat : ℚ)
  (days_in_month : ℕ)
  (H1 : treats_per_day = 2)
  (H2 : cost_per_treat = 0.1)
  (H3 : days_in_month = 30) :
  treats_per_day * days_in_month * cost_per_treat = 6 :=
by sorry

end dog_treats_cost_l577_577595


namespace largest_N_l577_577487

noncomputable def largest_vertex_sum (a T : ℤ) (hT : T ≠ 0)
    (hA : ∃ (a b c : ℤ), ∀ x, a * x * (x - 3 * T) = ax^2 + bx + c)
    (hB : (3 * T, 0) ∈ set_of (λ '(x, y), y = a * x * (x - 3 * T)))
    (hC : (3 * T + 1, 36) ∈ set_of (λ '(x, y), y = a * x * (x - 3 * T))) : ℤ :=
begin
  exact 62,
end

theorem largest_N {a T : ℤ} (hT : T ≠ 0)
    (hA : (0, 0) ∈ set_of (λ '(x, y), y = a * x * (x - 3 * T)))
    (hB : (3 * T, 0) ∈ set_of (λ '(x, y), y = a * x * (x - 3 * T)))
    (hC : (3 * T + 1, 36) ∈ set_of (λ '(x, y), y = a * x * (x - 3 * T))) :
    largest_vertex_sum a T hT hA hB hC = 62 :=
sorry

end largest_N_l577_577487


namespace area_of_triangle_l577_577764

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l577_577764


namespace cans_collected_l577_577938

theorem cans_collected (C : ℕ) 
  (h1 : (20 / 5) * 1.5 = 6)
  (h2 : (C / 12) * 0.5 + 6 = 12) : 
  C = 144 :=
by {
-- Assume the proof to be here
sorry
}

end cans_collected_l577_577938


namespace carol_sold_cupcakes_l577_577147

variable (initial_cupcakes := 30) (additional_cupcakes := 28) (final_cupcakes := 49)

theorem carol_sold_cupcakes : (initial_cupcakes + additional_cupcakes - final_cupcakes = 9) :=
by sorry

end carol_sold_cupcakes_l577_577147


namespace inlet_rate_is_1800_l577_577848

-- Define the parameters
def tank_volume : ℝ := 8640
def leak_empty_time : ℝ := 8
def combined_empty_time : ℝ := 12

-- Translate the given conditions
def leak_rate : ℝ := tank_volume / leak_empty_time
def combined_rate : ℝ := tank_volume / combined_empty_time

-- Prove that the inlet rate is 1800 liters per hour
theorem inlet_rate_is_1800 : 
  ∃ R : ℝ, (R - leak_rate = combined_rate) ∧ (R = 1800) :=
by
  -- Definitions used in the problem
  let leak_rate := tank_volume / leak_empty_time
  let combined_rate := tank_volume / combined_empty_time
  sorry

end inlet_rate_is_1800_l577_577848


namespace first_player_min_score_55_l577_577371

theorem first_player_min_score_55 :
  ∃ a b : ℕ, 1 ≤ a ∧ a < b ∧ b ≤ 101 ∧ (∀ s t : finset ℕ, s.card = 99 → s ⊆ finset.range 102 → t = (finset.range 102).filter (λ x, ¬ x ∈ s) → ∃ x y, x ∈ t ∧ y ∈ t ∧ |x - y| ≥ 55) :=
begin
  sorry
end

end first_player_min_score_55_l577_577371


namespace microbrewery_increase_l577_577402

theorem microbrewery_increase (B H B' : ℝ) (approx : ℝ) :
  (B' = 1.89901 * B) → (approx = 90) → 
  (output_per_hour := B / H) → 
  (new_hours := 0.7 * H) → 
  (new_output_per_hour := B' / new_hours = 2.7143 * (B / H)) → 
  (percent_increase := 100 * (B' - B) / B) → 
  percent_increase ≈ approx :=
sorry

end microbrewery_increase_l577_577402


namespace regular_18gon_symmetries_l577_577423

theorem regular_18gon_symmetries :
  let L := 18
  let R := 20
  L + R = 38 := by
sorry

end regular_18gon_symmetries_l577_577423


namespace positive_integers_solution_l577_577913

theorem positive_integers_solution (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (⌊(a ^ 2) / b⌋ + ⌊(b ^ 2) / a⌋ = ⌊((a ^ 2 + b ^ 2) / (a * b))⌋ + a * b) ↔ (a = b ^ 2 + 1) :=
by
  sorry

end positive_integers_solution_l577_577913


namespace dihedral_angles_perpendicular_l577_577239

theorem dihedral_angles_perpendicular (A B : ℝ) (h : A ≠ B) 
  (H : ∀ x y : ℝ, (∠ x y ⊥ ∠A -> ∠ x y ⊥ ∠B) ∨ (∠ x y = ∠A) ∨ (∠ x y = ∠B)) :
  (dihedral_angle A) ⊥ (dihedral_angle B) ↔ (dihedral_angle A = dihedral_angle B) ∨ (dihedral_angle A + dihedral_angle B = 180°) :=
sorry

end dihedral_angles_perpendicular_l577_577239


namespace triangle_area_is_64_l577_577767

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l577_577767


namespace value_of_a_5_l577_577987

-- Define the sequence with the general term formula
def a (n : ℕ) : ℕ := 4 * n - 3

-- Prove that the value of a_5 is 17
theorem value_of_a_5 : a 5 = 17 := by
  sorry

end value_of_a_5_l577_577987


namespace triangulation_inequality_l577_577934

-- Given A as a finite set of points in a plane.
variable (A : Finset (ℝ × ℝ))

-- v(A) is the number of triangles in a triangulation of the set A
-- Let's assume v is a function mapping a finite set of points to the number of triangles in a triangulation
noncomputable def v (S : Finset (ℝ × ℝ)) : ℕ := sorry

-- A + A = { x + y | x, y ∈ A }
noncomputable def A_add_A (A : Finset (ℝ × ℝ)) : Finset (ℝ × ℝ) := Finset.image (λ p : (ℝ × ℝ) × (ℝ × ℝ), (p.1.1 + p.2.1, p.1.2 + p.2.2)) (A.product A)

-- The goal is to prove that v(A + A) ≥ 4 * v(A)
theorem triangulation_inequality (A : Finset (ℝ × ℝ)) :
  v (A_add_A A) ≥ 4 * v A :=
sorry

end triangulation_inequality_l577_577934


namespace complex_modulus_l577_577497

theorem complex_modulus (z : ℂ) (h : (z - complex.i) * complex.i = 2 + 3 * complex.i) : complex.abs z = real.sqrt 10 :=
by
  sorry

end complex_modulus_l577_577497


namespace area_ratio_sum_eq_three_halves_l577_577253

open EuclideanGeometry

variable {A B C M G P Q : Point}

-- Definitions of conditions
variable (hMidpoint : midpoint B C M)
variable (hCentroid : centroid A B C G)
variable (hLine : collinear G P Q)
variable (hIntAB : collinear A B P)
variable (hIntAC : collinear A C Q)
variable (hNePB : P ≠ B)
variable (hNeQC : Q ≠ C)

-- Target theorem statement
theorem area_ratio_sum_eq_three_halves
  (hMidpoint : midpoint B C M)
  (hCentroid : centroid A B C G)
  (hLine : collinear G P Q)
  (hIntAB : collinear A B P)
  (hIntAC : collinear A C Q)
  (hNePB : P ≠ B)
  (hNeQC : Q ≠ C)
  : area_ratio B G M P A G + area_ratio C M G Q A G = 3 / 2 :=
sorry

end area_ratio_sum_eq_three_halves_l577_577253


namespace digit_in_thousandths_place_l577_577790

theorem digit_in_thousandths_place :
  (decimals (7 / 32)).nth 3 = some 8 := 
sorry

end digit_in_thousandths_place_l577_577790


namespace opposite_pairs_l577_577804

theorem opposite_pairs :
  (3^2 = 9) ∧ (-3^2 = -9) ∧
  ¬ ((3^2 = 9 ∧ -2^3 = -8) ∧ 9 = -(-8)) ∧
  ¬ ((3^2 = 9 ∧ (-3)^2 = 9) ∧ 9 = -9) ∧
  ¬ ((-3^2 = -9 ∧ -(-3)^2 = -9) ∧ -9 = -(-9)) :=
by
  sorry

end opposite_pairs_l577_577804


namespace download_time_is_2_hours_l577_577098

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l577_577098


namespace solve_number_l577_577039

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l577_577039


namespace select_donors_l577_577835

-- We define the number of people with each blood type
def O : Nat := 10
def A : Nat := 5
def B : Nat := 8
def AB : Nat := 3

-- Statement that computes the number of ways to select one person of each blood type
theorem select_donors :
  O * A * B * AB = 1200 :=
by
  simp [O, A, B, AB]
  sorry

end select_donors_l577_577835


namespace triangle_is_isosceles_right_l577_577230

theorem triangle_is_isosceles_right (a b c : ℝ) (B : ℝ)
  (h1 : log a - log c = log (sin B))
  (h2 : log (sin B) = - log (sqrt 2))
  (h3 : B < π / 2) :
  a = b ∧ B = π / 4 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end triangle_is_isosceles_right_l577_577230


namespace probability_white_ball_from_first_urn_correct_l577_577728

noncomputable def probability_white_ball_from_first_urn : ℝ :=
  let p_H1 : ℝ := 0.5
  let p_H2 : ℝ := 0.5
  let p_A_given_H1 : ℝ := 0.7
  let p_A_given_H2 : ℝ := 0.6
  let p_A : ℝ := p_H1 * p_A_given_H1 + p_H2 * p_A_given_H2
  p_H1 * p_A_given_H1 / p_A

theorem probability_white_ball_from_first_urn_correct :
  probability_white_ball_from_first_urn = 0.538 :=
sorry

end probability_white_ball_from_first_urn_correct_l577_577728


namespace problem_sum_sin_l577_577138

theorem problem_sum_sin (a x: ℝ) (n: ℕ) :
  (∑ k in Finset.range (n + 1), (-1 : ℝ)^(n - k) * Nat.choose n k * real.sin (k * x + a)) = 
  (2 : ℝ)^n * (real.sin (x / 2))^n * real.sin (a + (n * (x + real.pi)) / 2) :=
by sorry

end problem_sum_sin_l577_577138


namespace download_time_is_2_hours_l577_577099

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l577_577099


namespace Ned_earning_money_l577_577826

def total_games : Nat := 15
def non_working_games : Nat := 6
def price_per_game : Nat := 7
def working_games : Nat := total_games - non_working_games
def total_money : Nat := working_games * price_per_game

theorem Ned_earning_money : total_money = 63 := by
  sorry

end Ned_earning_money_l577_577826


namespace hockey_season_games_l577_577362

theorem hockey_season_games (n_teams : ℕ) (n_faces : ℕ) (h1 : n_teams = 18) (h2 : n_faces = 10) :
  let total_games := (n_teams * (n_teams - 1) / 2) * n_faces
  total_games = 1530 :=
by
  sorry

end hockey_season_games_l577_577362


namespace sum_floor_log2_to_2048_l577_577081

theorem sum_floor_log2_to_2048 :
  (Finset.sum (Finset.range 2048.succ) (λ N : ℕ, Int.toNat ⌊Real.logb 2 (N : ℝ)⌋) = 14349) :=
by
  sorry

end sum_floor_log2_to_2048_l577_577081


namespace solution_set_of_inequality_l577_577271

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

noncomputable def g (x : ℝ) : ℝ := f x / Real.exp x

theorem solution_set_of_inequality :
  (∀ x : ℝ, Differentiable ℝ f) →
  (∀ x : ℝ, f' x = (fun x => (f' x - f x) / Real.exp x)) →
  (∀ x : ℝ, f x < f' x) →
  (f 0 = 3) →
  {x | f x < 3 * Real.exp x} = {x | 0 < x} := sorry

end solution_set_of_inequality_l577_577271


namespace parallelepiped_volume_l577_577685

theorem parallelepiped_volume (a : ℝ) (h₁ : ∀ (A B C D : Prop), base_is_rhombus A B C D ∧ side_length A B = a ∧ ∠ A B C = 60) 
(h₂ : ∀ (A A1 : Prop), edge_length A A1 = a) 
(h₃ : ∀ (A A1 B D : Prop), ∠ A A1 B = 45 ∧ ∠ A A1 D = 45) : 
  volume_of_parallelepiped a = a^3 / 2 := 
sorry

end parallelepiped_volume_l577_577685


namespace number_of_knights_is_two_number_of_knights_is_two_l577_577652

-- Define types for inhabitants (knights or liars)
inductive Inhabitant
| knight
| liar

open Inhabitant

-- Define the statements given by the inhabitants
def statements (i : ℕ) : String :=
  match i with
  | 1 => "One knight"
  | 2 => "Two knights"
  | 3 => "Three knights"
  | 4 => "Don't believe them, they are all liars"
  | 5 => "You're the liar!"
  | _ => ""

-- Define the truth-telling property of knights
def tells_truth (i : ℕ) (s : String) : Prop :=
  match i with
  | 1 => (s = "One knight") ↔ (count_knights = 1)
  | 2 => (s = "Two knights") ↔ (count_knights = 2)
  | 3 => (s = "Three knights") ↔ (count_knights = 3)
  | 4 => (s = "Don't believe them, they are all liars") ↔ (inhabitant 1 = liar ∧ inhabitant 2 = liar ∧ inhabitant 3 = liar)
  | 5 => (s = "You're the liar!") ↔ (inhabitant 4 = liar)
  | _ => false

-- Define the main theorem to be proven
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

-- Noncomputable definition to avoid computational problems
noncomputable def count_knights : ℕ :=
  sorry

-- Noncomputable to define each inhabitant's type
noncomputable def inhabitant (i : ℕ) : Inhabitant :=
  match i with
  | 1 => liar
  | 2 => knight
  | 3 => liar
  | 4 => liar
  | 5 => knight
  | _ => liar -- Default to liar, although there are only 5 inhabitants

-- Additional properties used
def is_knight (i : ℕ) : Prop := inhabitant i = knight
def is_liar (i : ℕ) : Prop := inhabitant i = liar

-- Count the number of knights
noncomputable def count_knights : ℕ :=
  List.length (List.filter (λ i => is_knight i) [1, 2, 3, 4, 5])

-- Main theorem that states there are exactly two knights according to the statements
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

end number_of_knights_is_two_number_of_knights_is_two_l577_577652


namespace probability_drawing_black_piece_l577_577724

/-- There are 7 Go pieces in a pocket, 3 white and 4 black.
    The probability of drawing a black piece is 4/7. -/
theorem probability_drawing_black_piece (total_pieces: ℕ) (white_pieces: ℕ) (black_pieces: ℕ) :
  total_pieces = 7 → white_pieces = 3 → black_pieces = 4 → (black_pieces : ℚ) / total_pieces = 4 / 7 :=
by
  intros h1 h2 h3
  rw [h1, h3]
  norm_cast
  norm_num
  sorry

end probability_drawing_black_piece_l577_577724


namespace find_angle4_l577_577966

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ)
                    (h1 : angle1 + angle2 = 180)
                    (h2 : angle3 = 2 * angle4)
                    (h3 : angle1 = 50)
                    (h4 : angle3 + angle4 = 130) : 
                    angle4 = 130 / 3 := by 
    sorry

end find_angle4_l577_577966


namespace fg_sqrt3_l577_577538

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 - 1

theorem fg_sqrt3 : f (g (real.sqrt 3)) = 1 :=
by sorry

end fg_sqrt3_l577_577538


namespace point_C_is_in_region_l577_577866

variable (x y : ℝ)

def point_in_region := ∃ (x y : ℝ), (x + 2 * y - 1 > 0) ∧ (x - y + 3 < 0)

theorem point_C_is_in_region : point_in_region 0 4 :=
by {
  let x := 0
  let y := 4
  unfold point_in_region
  split,
  {
    -- First inequality
    rw add_mul,
    linarith,
  },
  {
    -- Second inequality
    linarith,
  },
  sorry
}

end point_C_is_in_region_l577_577866


namespace find_m_plus_n_l577_577607

def is_valid_point (x y z : ℕ) : Prop :=
  (0 ≤ x ∧ x ≤ 3) ∧ (0 ≤ y ∧ y ≤ 4) ∧ (0 ≤ z ∧ z ≤ 5)

def is_midpoint_in_T (p1 p2 : ℕ × ℕ × ℕ) : Prop :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  let (xm, ym, zm) := (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
  is_valid_point xm ym zm

theorem find_m_plus_n : 
  let T := {p : ℕ × ℕ × ℕ | is_valid_point p.1 p.2.1 p.2.2}
  let num_points := Finset.card (Finset.filter (λ p, p ∈ T) Finset.univ)
  let num_pairs := (num_points * (num_points - 1)) / 2
  let num_valid_pairs := (8 * 13 * 20) - num_points
  let prob := Rat.mk num_valid_pairs num_pairs in
  prob.num = 205 ∧ prob.den = 714 ∧ (prob.num + prob.den = 919) :=
by
  sorry

end find_m_plus_n_l577_577607


namespace curve_cartesian_eq_and_line_eq_l577_577572

theorem curve_cartesian_eq_and_line_eq :
  (∀ θ : ℝ, x = 4 * Real.cos θ ∧ y = 3 * Real.sin θ) →
  (∃ α : ℝ, ∀ t : ℝ, M = (0, 1) →
      ∃ A B : ℝ × ℝ, A = (4 * cos θ, 3 * sin θ) ∧ B = (4 * cos (2 * θ + π), 3 * sin (2 * θ + π)) ∧
      BM = 2 * AM) →
  (∃ x y : ℝ, (x^2 / 16 + y^2 / 9 = 1) ∧ (line x y = 0)) :=
begin
  sorry
end

end curve_cartesian_eq_and_line_eq_l577_577572


namespace distance_between_city_centers_l577_577688

theorem distance_between_city_centers (d_map : ℝ) (scale : ℝ) (d_real : ℝ) (h1 : d_map = 112) (h2 : scale = 10) (h3 : d_real = d_map * scale) : d_real = 1120 := by
  sorry

end distance_between_city_centers_l577_577688


namespace solution_l577_577898

def problem_statement : Prop :=
  (3025 - 2880) ^ 2 / 225 = 93

theorem solution : problem_statement :=
by {
  sorry
}

end solution_l577_577898


namespace angle_same_terminal_side_l577_577212

theorem angle_same_terminal_side (α θ : ℝ) (hα : α = 1690) (hθ : 0 < θ) (hθ2 : θ < 360) (h_terminal_side : ∃ k : ℤ, α = k * 360 + θ) : θ = 250 :=
by
  sorry

end angle_same_terminal_side_l577_577212


namespace number_of_knights_l577_577656

-- Definition of knight and liar
inductive Inhabitant
| knight : Inhabitant
| liar : Inhabitant

open Inhabitant

-- Statements of the inhabitants
def statement_1 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i1 with
  | knight => 1
  | liar => true

def statement_2 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i2 with
  | knight => 2
  | liar => true

def statement_3 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  match i3 with
  | knight => 3
  | liar => true

def statement_4 (i1 i2 i3 i4 i5 : Inhabitant) : Prop := 
  (i1 = liar ∧ i2 = liar ∧ i3 = liar ∧ i4 = liar ∧ i5 = liar)

def statement_5 (i4 : Inhabitant) : Prop := 
  match i5 with
  | knight => i4 = liar
  | liar => true

-- Problem statement in Lean 4
theorem number_of_knights (i1 i2 i3 i4 i5 : Inhabitant) 
  (h1 : statement_1 i1 i2 i3 i4 i5)
  (h2 : statement_2 i1 i2 i3 i4 i5)
  (h3 : statement_3 i1 i2 i3 i4 i5)
  (h4 : statement_4 i1 i2 i3 i4 i5)
  (h5 : statement_5 i4) :
  (i1 = knight ∨ i2 = knight ∨ i3 = knight ∨ i4 = knight ∨ i5 = knight) → 
  (i1 = knight ∧ i2 = knight ∧ i3 = knight ∧ i4 = knight ∧ i5 = knight) → 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end number_of_knights_l577_577656


namespace part1_part2_l577_577188

-- Definition of the function f
def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

-- Part 1: For m = 1, the solution set of f(x) >= 6
theorem part1 (x : ℝ) : f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := 
by 
  sorry

-- Part 2: If the inequality f(x) ≤ 2m - 5 has a solution with respect to x, then m ≥ 8
theorem part2 (m : ℝ) (h : ∃ x, f x m ≤ 2 * m - 5) : m ≥ 8 :=
by
  sorry

end part1_part2_l577_577188


namespace number_of_knights_is_two_number_of_knights_is_two_l577_577650

-- Define types for inhabitants (knights or liars)
inductive Inhabitant
| knight
| liar

open Inhabitant

-- Define the statements given by the inhabitants
def statements (i : ℕ) : String :=
  match i with
  | 1 => "One knight"
  | 2 => "Two knights"
  | 3 => "Three knights"
  | 4 => "Don't believe them, they are all liars"
  | 5 => "You're the liar!"
  | _ => ""

-- Define the truth-telling property of knights
def tells_truth (i : ℕ) (s : String) : Prop :=
  match i with
  | 1 => (s = "One knight") ↔ (count_knights = 1)
  | 2 => (s = "Two knights") ↔ (count_knights = 2)
  | 3 => (s = "Three knights") ↔ (count_knights = 3)
  | 4 => (s = "Don't believe them, they are all liars") ↔ (inhabitant 1 = liar ∧ inhabitant 2 = liar ∧ inhabitant 3 = liar)
  | 5 => (s = "You're the liar!") ↔ (inhabitant 4 = liar)
  | _ => false

-- Define the main theorem to be proven
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

-- Noncomputable definition to avoid computational problems
noncomputable def count_knights : ℕ :=
  sorry

-- Noncomputable to define each inhabitant's type
noncomputable def inhabitant (i : ℕ) : Inhabitant :=
  match i with
  | 1 => liar
  | 2 => knight
  | 3 => liar
  | 4 => liar
  | 5 => knight
  | _ => liar -- Default to liar, although there are only 5 inhabitants

-- Additional properties used
def is_knight (i : ℕ) : Prop := inhabitant i = knight
def is_liar (i : ℕ) : Prop := inhabitant i = liar

-- Count the number of knights
noncomputable def count_knights : ℕ :=
  List.length (List.filter (λ i => is_knight i) [1, 2, 3, 4, 5])

-- Main theorem that states there are exactly two knights according to the statements
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

end number_of_knights_is_two_number_of_knights_is_two_l577_577650


namespace lcm_of_lap_times_l577_577063

theorem lcm_of_lap_times :
  Nat.lcm (Nat.lcm 5 8) 10 = 40 := by
  sorry

end lcm_of_lap_times_l577_577063


namespace number_of_integer_multisets_l577_577452

-- Define the conditions given in the problem
variable (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ)
axiom b_6_nonzero : b_6 ≠ 0
axiom b_0_nonzero : b_0 ≠ 0

-- Define the polynomials
def p (x : ℝ) : ℝ := b_6 * x^6 + b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
def q (x : ℝ) : ℝ := b_0 * x^6 + b_1 * x^5 + b_2 * x^4 + b_3 * x^3 + b_4 * x^2 + b_5 * x + b_6

-- Hypotheses regarding the roots
variable (r : Fin 6 → ℝ)
axiom roots_p : ∀ i, p (r i) = 0
axiom roots_q : ∀ i, q (r i) = 0

-- The proof problem
theorem number_of_integer_multisets (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ) 
  (b_6_nonzero : b_6 ≠ 0) (b_0_nonzero : b_0 ≠ 0) 
  (r : Fin 6 → ℝ) 
  (roots_p : ∀ i, p b_6 b_5 b_4 b_3 b_2 b_1 b_0 (r i) = 0) 
  (roots_q : ∀ i, q b_0 b_1 b_2 b_3 b_4 b_5 b_6 (r i) = 0) : 
  ∃ S : Finset (ℤ), S.card = 7 ∧ (∀ i, r i ∈ S) :=
sorry

end number_of_integer_multisets_l577_577452


namespace jeanne_should_buy_more_tickets_l577_577585

theorem jeanne_should_buy_more_tickets :
  let cost_ferris_wheel := 5
  let cost_roller_coaster := 4
  let cost_bumper_cars := 4
  let jeanne_current_tickets := 5
  let total_tickets_needed := cost_ferris_wheel + cost_roller_coaster + cost_bumper_cars
  let tickets_needed_to_buy := total_tickets_needed - jeanne_current_tickets
  tickets_needed_to_buy = 8 :=
by
  sorry

end jeanne_should_buy_more_tickets_l577_577585


namespace inner_circle_radius_l577_577941

theorem inner_circle_radius (side_length : ℝ) (r : ℝ) : 
  side_length = 4 ∧ 
  ∀ (x y : ℝ), (x, y) ∈ set.univ → dist (x, y) (0, 0) = 2 * real.sqrt 2 → ∃ r, r = 1 + real.sqrt 3 → 
  ∃ inner_circle_radius, inner_circle_radius = (1 + real.sqrt 3) :=
by sorry

end inner_circle_radius_l577_577941


namespace parallel_lines_m_values_l577_577218

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (m-2) * x - y - 1 = 0) ∧ (∀ x y : ℝ, 3 * x - m * y = 0) → 
  (m = -1 ∨ m = 3) :=
by
  sorry

end parallel_lines_m_values_l577_577218


namespace probability_same_color_l577_577209

theorem probability_same_color (red blue green : ℕ) (total plates chosen : ℕ) :
  red = 6 ∧ blue = 5 ∧ green = 3 ∧ plates = 14 ∧ chosen = 3 → 
  ((choose red chosen + choose blue chosen + choose green chosen).to_float / choose plates chosen.to_float = 31 / 364) :=
by
  intro h
  cases h with h_red h_blue
  cases h_blue with h_blue h_green
  cases h_green with h_green h_plates
  cases h_plates with h_plates h_chosen
  skip

end probability_same_color_l577_577209


namespace smallest_value_geq_4_l577_577610

noncomputable def smallest_value (a b c d : ℝ) : ℝ :=
  (a + b + c + d) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem smallest_value_geq_4 (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_value a b c d ≥ 4 :=
by
  sorry

end smallest_value_geq_4_l577_577610


namespace minimum_value_alpha_beta_l577_577960

-- Considering the setup for the problem
variables {O A B C : Type}
variables (a : ℝ) (α β : ℝ)
variables (AB AC : ℝ ) -- lengths of AB and AC
variables (angle_BAC : ℝ) -- angle BAC in radians

-- Set the conditions
def circumcenter (O : Type) (A B C : Type) : Prop := -- to be defined: the property which ensures O is the circumcenter
AB = 2 * a ∧ AC = 2 / a ∧ angle_BAC = 2 * Real.pi / 3 -- 120 degrees = 2π/3 radians

-- Problem statement
theorem minimum_value_alpha_beta 
  (circumcenter O A B C) 
  (h : ∃ R AO, AO = α * AB + β * AC):
  ∃ α β, α + β = 2 :=
sorry

end minimum_value_alpha_beta_l577_577960


namespace renu_completion_time_l577_577667

-- Define the conditions
variable (R : ℝ) -- time Renu needs to complete the work alone
variable (S : ℝ := 12) -- time Suma needs to complete the work alone
variable (C : ℝ := 4) -- time together they need to complete the work

-- Define their rates
def renu_rate := 1 / R
def suma_rate := 1 / S
def together_rate := 1 / C

-- The main theorem to prove
theorem renu_completion_time :
  (renu_rate + suma_rate = together_rate) → R = 24 := by
  sorry

end renu_completion_time_l577_577667


namespace problem_solution_l577_577166

/-- Define proposition p: ∀α∈ℝ, sin(π-α) ≠ -sin(α) -/
def p := ∀ α : ℝ, Real.sin (Real.pi - α) ≠ -Real.sin α

/-- Define proposition q: ∃x∈[0,+∞), sin(x) > x -/
def q := ∃ x : ℝ, 0 ≤ x ∧ Real.sin x > x

/-- Prove that ¬p ∨ q is a true proposition -/
theorem problem_solution : ¬p ∨ q :=
by
  sorry

end problem_solution_l577_577166


namespace sum_floor_log2_l577_577079

theorem sum_floor_log2 :
  (∑ N in Finset.range 2048, Int.floor (Real.log N / Real.log 2)) = 20445 :=
by
  sorry

end sum_floor_log2_l577_577079


namespace calculate_expression_l577_577877

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 :=
by {
  -- hint to the Lean prover to consider associative property
  sorry
}

end calculate_expression_l577_577877


namespace find_positive_integer_n_l577_577911

theorem find_positive_integer_n : 
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, 2 * 10^(k + 1) + 10 * n + 1 = 33 * n :=
by
  use 87
  exists 2
  -- now, we fill the details to show:
  -- 2 * 10^(2 + 1) + 10 * 87 + 1 = 33 * 87
  sorry

end find_positive_integer_n_l577_577911


namespace three_digit_diff_l577_577686

theorem three_digit_diff (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) :
  ∃ d : ℕ, d = a - b ∧ (d < 10 ∨ (10 ≤ d ∧ d < 100) ∨ (100 ≤ d ∧ d < 1000)) :=
sorry

end three_digit_diff_l577_577686


namespace cube_diagonals_l577_577837

def is_diagonal (cube : list (ℕ × ℕ)) (edge : ℕ × ℕ) : Prop := 
  ∀ (v₁ v₂ : ℕ), (v₁, v₂) = edge → v₁ ≠ v₂ ∧ ¬((v₁, v₂) ∈ cube)

theorem cube_diagonals : 
  ∀ (cube : list (ℕ × ℕ)), 
    (cube.length = 12 ∧ (∀ (v : ℕ), ∃! e, e ∈ cube ∧ e.fst = v ∧ e.snd = v.succ)) →
    ∃ (diagonals : list (ℕ × ℕ)), 
      (∀ d ∈ diagonals, is_diagonal cube d) ∧ diagonals.length = 16 := 
by
  sorry

end cube_diagonals_l577_577837


namespace gcd_of_lcm_ratio_l577_577222

theorem gcd_of_lcm_ratio {A B k : ℕ} (h1 : Nat.lcm A B = 180) (h2 : A * 5 = B * 2) :
  Nat.gcd A B = 18 := 
by
  sorry

end gcd_of_lcm_ratio_l577_577222


namespace B_completes_work_in_n_days_l577_577013

-- Define the conditions
def can_complete_work_A_in_d_days (d : ℕ) : Prop := d = 15
def fraction_of_work_left_after_working_together (t : ℕ) (fraction : ℝ) : Prop :=
  t = 5 ∧ fraction = 0.41666666666666663

-- Define the theorem to be proven
theorem B_completes_work_in_n_days (d t : ℕ) (fraction : ℝ) (x : ℕ) 
  (hA : can_complete_work_A_in_d_days d) 
  (hB : fraction_of_work_left_after_working_together t fraction) : x = 20 :=
sorry

end B_completes_work_in_n_days_l577_577013


namespace large_duck_cost_l577_577331

theorem large_duck_cost
  (price_regular : ℝ)
  (price_large : ℝ)
  (num_regular : ℕ)
  (num_large : ℕ)
  (total_amount : ℝ)
  (h_price_regular : price_regular = 3)
  (h_num_regular : num_regular = 221)
  (h_num_large : num_large = 185)
  (h_total_amount : total_amount = 1588) :
  price_large = 5 :=
by
  -- Step 1: Write the total amount equation based on conditions
  have h1: total_amount = num_regular * price_regular + num_large * price_large, sorry
  
  -- Step 2: Substitute given quantities into the equation
  rw [h_price_regular, h_num_regular, h_num_large] at h1, sorry
  
  -- Step 3: Simplify and solve for price_large
  linarith, sorry

end large_duck_cost_l577_577331


namespace seq_val_a7_l577_577853

theorem seq_val_a7 {a : ℕ} {b : ℕ} 
  (h1 : a < b) 
  (h2 : a_6 = a_1 + 3 * (a_1 + 2 * a_2) + a_5) 
  (h3 : a_n = if a_n = 6 then 74 else a_n) 
  (h4 : ∀ n ≥ 1, a_{n+2} = a_{n+1} + a_n) : 
  a_7 = 119 ∨ a_7 = 120 := 
sorry

end seq_val_a7_l577_577853


namespace waiters_hired_correct_l577_577873

noncomputable def waiters_hired (W H : ℕ) : Prop :=
  let cooks := 9
  (cooks / W = 3 / 8) ∧ (cooks / (W + H) = 1 / 4) ∧ (H = 12)

theorem waiters_hired_correct (W H : ℕ) : waiters_hired W H :=
  sorry

end waiters_hired_correct_l577_577873


namespace sum_of_sequence_l577_577162

def a : ℕ → ℤ
| 0     := 1
| (n+1) := 3 * a n + 1

def S : ℕ → ℤ
| 0     := 0
| (n+1) := S n + a n

theorem sum_of_sequence (n : ℕ) :
  S n = (1 / 4 : ℚ) * (3 ^ (n + 1) - 2 * n - 3) := 
sorry

end sum_of_sequence_l577_577162


namespace larger_number_is_450_l577_577818

-- Given conditions
def HCF := 30
def Factor1 := 10
def Factor2 := 15

-- Derived definitions needed for the proof
def LCM := HCF * Factor1 * Factor2

def Number1 := LCM / Factor1
def Number2 := LCM / Factor2

-- The goal is to prove the larger of the two numbers is 450
theorem larger_number_is_450 : max Number1 Number2 = 450 :=
by
  sorry

end larger_number_is_450_l577_577818


namespace infinite_set_of_midpoints_l577_577277

variable {Point : Type*} [AddCommGroup Point] [Module ℝ Point]

def is_midpoint (S : set Point) : Prop :=
∀ (p ∈ S), ∃ (a b ∈ S), p = (a + b) / 2

theorem infinite_set_of_midpoints (S : set Point) (h : is_midpoint S) : set.infinite S :=
sorry

end infinite_set_of_midpoints_l577_577277


namespace add_congruence_mul_congruence_l577_577304

namespace ModularArithmetic

-- Define the congruence relation mod m
def is_congruent_mod (a b m : ℤ) : Prop := ∃ k : ℤ, a - b = k * m

-- Part (a): Proving a + c ≡ b + d (mod m)
theorem add_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a + c) (b + d) m :=
  sorry

-- Part (b): Proving a ⋅ c ≡ b ⋅ d (mod m)
theorem mul_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a * c) (b * d) m :=
  sorry

end ModularArithmetic

end add_congruence_mul_congruence_l577_577304


namespace quadratic_function_relation_l577_577986

theorem quadratic_function_relation
  (a : ℝ) (h_a : a < 0)
  (x₁ x₂ : ℝ) (h_x : x₁ < x₂)
  (h_sum : x₁ + x₂ = 0) :
  let f := λ x, 2 * a * x^2 - a * x + 1 in
  f x₁ < f x₂ :=
by
  let f := λ x, 2 * a * x^2 - a * x + 1
  sorry

end quadratic_function_relation_l577_577986


namespace total_distance_covered_l577_577801

theorem total_distance_covered :
  ∀ (r j w total : ℝ),
    r = 40 →
    j = (3 / 5) * r →
    w = 5 * j →
    total = r + j + w →
    total = 184 := by
  sorry

end total_distance_covered_l577_577801


namespace trajectory_of_point_l577_577245

/-- In the coordinate plane, if there are two fixed points A and B, and a moving point P,
    such that the product of the slopes of the lines PA and PB is a constant value m,
    then the possible trajectory of the point P could be an ellipse, hyperbola, circle, or straight line. -/
theorem trajectory_of_point 
  (A B : ℝ × ℝ) 
  (P : ℝ × ℝ → Prop) 
  (m : ℝ) 
  (h : ∀ x y, P (x, y) → ((y - A.2) / (x - A.1)) * ((y - B.2) / (x - B.1)) = m) :
    ∃ t : ℝ, P t ∈ {ellipse, hyperbola, circle, line} :=
sorry

end trajectory_of_point_l577_577245


namespace profit_percentage_for_unspecified_weight_l577_577414

-- Definitions to align with the conditions
def total_sugar : ℝ := 1000
def profit_400_kg : ℝ := 0.08
def unspecified_weight : ℝ := 600
def overall_profit : ℝ := 0.14
def total_400_kg := total_sugar - unspecified_weight
def total_overall_profit := total_sugar * overall_profit
def total_400_kg_profit := total_400_kg * profit_400_kg
def total_unspecified_weight_profit (profit_percentage : ℝ) := unspecified_weight * profit_percentage

-- The theorem statement
theorem profit_percentage_for_unspecified_weight : 
  ∃ (profit_percentage : ℝ), total_400_kg_profit + total_unspecified_weight_profit profit_percentage = total_overall_profit ∧ profit_percentage = 0.18 := by
  sorry

end profit_percentage_for_unspecified_weight_l577_577414


namespace min_floor_sum_l577_577611

-- Definitions of the conditions
variables (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24)

-- Our main theorem statement
theorem min_floor_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24) :
  (Nat.floor ((a+b) / c) + Nat.floor ((b+c) / a) + Nat.floor ((c+a) / b)) = 6 := 
sorry

end min_floor_sum_l577_577611


namespace smallest_12_digit_number_divisible_by_36_with_all_digits_l577_577823

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  n ≥ 10^11 ∧ n < 10^12

noncomputable def contains_all_digits (n : ℕ) : Prop :=
  (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → d ∈ n.digits 10)

noncomputable def divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

theorem smallest_12_digit_number_divisible_by_36_with_all_digits : 
  ∃ n : ℕ, is_12_digit_number n ∧ contains_all_digits n ∧ divisible_by_36 n ∧ 
  (∀ m : ℕ, is_12_digit_number m ∧ contains_all_digits m ∧ divisible_by_36 m → n ≤ m) :=
sorry

end smallest_12_digit_number_divisible_by_36_with_all_digits_l577_577823


namespace remainder_when_K_divided_by_1000_l577_577935

def S (n : ℕ) : ℕ := n.digits_base 10 |>.sum

def satisfies_condition (n : ℕ) : Prop :=
  S(n) = (S(S(n)))^2

def count_valid_numbers (max_n : ℕ) : ℕ :=
  (List.range (max_n + 1)).countP satisfies_condition

theorem remainder_when_K_divided_by_1000 : 
  (count_valid_numbers (10^10) % 1000) = 632 :=
sorry

end remainder_when_K_divided_by_1000_l577_577935


namespace baseball_opponents_total_score_l577_577829

theorem baseball_opponents_total_score :
  -- Defining the conditions
  (∃ (team_scores : List ℕ) (opponent_scores : List ℚ),
    team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] ∧
    (∃ (lost_games_indices : List ℕ) (won_games_indices : List ℕ),
      lost_games_indices = [0, 2, 4, 6, 8, 10, 12] ∧
      won_games_indices = [1, 3, 5, 7, 9, 11, 13, 14] ∧
      opponent_scores = List.map (λ i, if i ∈ lost_games_indices then team_scores[i] + 2 else team_scores[i] / 3) (List.range 15) ∧
      List.sum opponent_scores = 87
    )
  )
  :=
  sorry

end baseball_opponents_total_score_l577_577829


namespace mean_difference_incorrect_correct_l577_577682

theorem mean_difference_incorrect_correct (S' : ℝ) :
  (S' + 1_480_000) / 1500 - (S' + 98_000) / 1500 = 921.333333 :=
by sorry

end mean_difference_incorrect_correct_l577_577682


namespace complete_square_l577_577377

theorem complete_square (x : ℝ) : (x^2 + 4*x - 1 = 0) → ((x + 2)^2 = 5) :=
by
  intro h
  sorry

end complete_square_l577_577377


namespace integral_log_two_l577_577908

theorem integral_log_two :
  ∫ (x : ℝ) in 1..2, (1 / x) = Real.log 2 := 
by
  sorry

end integral_log_two_l577_577908


namespace projection_calculation_l577_577521

open Real

noncomputable def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ × ℝ) := a.1 * b.1 + a.2 * b.2 + a.3 * b.3 in
  let scalar := dot_product u v / dot_product v v in
  (scalar * v.1, scalar * v.2, scalar * v.3)

theorem projection_calculation :
  let u := (2, 2, 6)
  let proj_u := (1, -0.5, 0.5)
  let w := (5, 5, 0)
  ∃ v : ℝ × ℝ × ℝ, projection u v = proj_u ∧ projection w v = (1.667, -0.833, 0.833) :=
by
  sorry

end projection_calculation_l577_577521


namespace combined_sleep_time_l577_577408

variables (cougar_night_sleep zebra_night_sleep total_sleep_cougar total_sleep_zebra total_weekly_sleep : ℕ)

theorem combined_sleep_time :
  (cougar_night_sleep = 4) →
  (zebra_night_sleep = cougar_night_sleep + 2) →
  (total_sleep_cougar = cougar_night_sleep * 7) →
  (total_sleep_zebra = zebra_night_sleep * 7) →
  (total_weekly_sleep = total_sleep_cougar + total_sleep_zebra) →
  total_weekly_sleep = 70 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end combined_sleep_time_l577_577408


namespace distinct_roots_of_quadratic_l577_577959

theorem distinct_roots_of_quadratic (m : ℝ) : 
  let Δ := m^2 + 8 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - m * x1 - 2 = 0 ∧ x2^2 - m * x2 - 2 = 0) :=
by {
  sorry
}

end distinct_roots_of_quadratic_l577_577959


namespace find_B_days_l577_577011

noncomputable def work_rate_A := 1 / 15
noncomputable def work_rate_B (x : ℝ) := 1 / x

theorem find_B_days (x : ℝ) : 
  (5 * (work_rate_A + work_rate_B x) = 0.5833333333333334) →
  (x = 20) := 
by 
  intro h,
  sorry

end find_B_days_l577_577011


namespace max_ab_l577_577215

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x - 2

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (deriv (λ x, f x a b)) 1 = 0) : ab ≤ 9 :=
by
  have h4 : a + b = 6 := sorry
  apply le_of_eq
  sorry

end max_ab_l577_577215


namespace triangle_area_bounded_by_lines_l577_577781

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l577_577781


namespace fruit_seller_apples_l577_577019

theorem fruit_seller_apples : 
  ∃ (x : ℝ), (x * 0.6 = 420) → x = 700 :=
sorry

end fruit_seller_apples_l577_577019


namespace win_sector_area_l577_577407

theorem win_sector_area {r : ℝ} {p : ℝ} (hr : r = 8) (hp : p = 3/8) :
  let A_circle := π * r ^ 2 in
  p * A_circle = 24 * π :=
by 
  sorry

end win_sector_area_l577_577407


namespace hose_drain_rate_l577_577680

-- Define the pool dimensions
def length : ℝ := 150
def width : ℝ := 50
def depth : ℝ := 10

-- Define the capacity percentage
def capacity_percentage : ℝ := 0.80

-- Define the time to drain the pool
def time_to_drain : ℝ := 1000

-- Prove the rate at which the hose removes water per minute
theorem hose_drain_rate : 
  let pool_volume := length * width * depth in
  let volume_at_80 := pool_volume * capacity_percentage in
  let rate := volume_at_80 / time_to_drain in
  rate = 60 := 
by
  sorry

end hose_drain_rate_l577_577680


namespace find_a_plus_b_l577_577345

theorem find_a_plus_b (a b : ℤ) (h1 : 2 * a = 0) (h2 : a^2 - b = 25) : a + b = -25 :=
by 
  sorry

end find_a_plus_b_l577_577345


namespace area_triangle_OCD_l577_577751

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l577_577751


namespace find_OP_dot_OQ_and_general_equation_l577_577244

-- Define the parametric equation of line l.
def line_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * cos α, 1 + t * sin α)

-- Define the polar equation of curve C.
def curve_polar (ρ θ : ℝ) : Prop :=
  ρ * sin θ ^ 2 + 2 * sin θ = ρ

-- Define the Cartesian form of curve C i.e., x^2 = 2y.
def curve_cartesian (x y : ℝ) : Prop :=
  x^2 = 2 * y

-- Given point A(0,1)
def point_A : (ℝ × ℝ) :=
  (0, 1)

-- The proof problem: find the value of OP ⋅ OQ for intersection points P and Q, and find the general equation of line l.
theorem find_OP_dot_OQ_and_general_equation (α t1 t2 : ℝ) (hα : α ∈ Ioo 0 (π / 2))
  (h1 : curve_cartesian (t1 * cos α) (1 + t1 * sin α))
  (h2 : curve_cartesian (t2 * cos α) (1 + t2 * sin α))
  (h3 : t1 + t2 = 2 * sin α / cos α ^ 2)
  (h4 : t1 * t2 = -2 / cos α ^ 2)
  (h5 : |1 + t1 * sin α - 1| = 2 * |1 + t2 * sin α - 1|):
  (t1 * cos α) * (t2 * cos α) + (1 + t1 * sin α) * (1 + t2 * sin α) = -1 ∧
  (∃ (m b : ℝ), (∀ (x y : ℝ), y = m * x + b ↔ (y = (1/2)*x + 1 ∨ y = (x - 2)/(-2)))) := sorry

end find_OP_dot_OQ_and_general_equation_l577_577244


namespace units_digit_7_pow_6pow5_l577_577140
open Int

theorem units_digit_7_pow_6pow5 : 
  let unit_cycle := [7, 9, 3, 1] in
  (6 ^ 5) % 4 = 0 →
  let n := ((6 ^ 5) % 4) in
  let units_digit := unit_cycle[n] in
  units_digit = 1 := 
by
  intro h
  have H : unit_cycle[(6 ^ 5) % 4 = 0] := by sorry
  let units_digit := unit_cycle[0]
  show 1 from rfl

end units_digit_7_pow_6pow5_l577_577140


namespace prob_C_prob_B_prob_at_least_two_l577_577015

variable {P : Set ℕ → ℝ}

-- Given Conditions
def probability_A := 3 / 4
def probability_AC_incorrect := 1 / 12
def probability_BC_correct := 1 / 4

-- Definitions used in conditions
def probability_C := 2 / 3
def probability_B := 3 / 8

-- Proof Statements
theorem prob_C : P {3} = probability_C := sorry
theorem prob_B : P {2} = probability_B := sorry
theorem prob_at_least_two :
  (P {1, 2, 3} + P {1, 2} (1 - probability_C) + P {1, 3} (1 - probability_B) + P {2, 3} (1 - probability_A)) = 21 / 32 := sorry

end prob_C_prob_B_prob_at_least_two_l577_577015


namespace not_divisible_by_11599_l577_577256

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem not_divisible_by_11599 :
  ¬ (factorial 3400 / (factorial 1700 * factorial 1700) % 11599 = 0) :=
by
  sorry

end not_divisible_by_11599_l577_577256


namespace red_marbles_count_l577_577367

noncomputable def total_marbles (R : ℕ) : ℕ := R + 16

noncomputable def P_blue (R : ℕ) : ℚ := 10 / (total_marbles R)

noncomputable def P_neither_blue (R : ℕ) : ℚ := (1 - P_blue R) * (1 - P_blue R)

noncomputable def P_either_blue (R : ℕ) : ℚ := 1 - P_neither_blue R

theorem red_marbles_count
  (R : ℕ) 
  (h1 : P_either_blue R = 0.75) :
  R = 4 :=
by
  sorry

end red_marbles_count_l577_577367


namespace bubble_gum_cost_l577_577395

theorem bubble_gum_cost (pieces : ℕ) (total_cost : ℕ) (h : pieces = 136) (k : total_cost = 2448) : total_cost / pieces = 18 :=
by
  rw [k, h]
  norm_num
  sorry

end bubble_gum_cost_l577_577395


namespace tangent_line_at_zero_f_ge_neg_x2_plus_x_range_k_l577_577179

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 - 1

theorem tangent_line_at_zero :
  TangentLine f 0 = (λ x => x) :=
sorry

theorem f_ge_neg_x2_plus_x :
  ∀ x : ℝ, f x ≥ -x^2 + x :=
sorry

theorem range_k :
  ∀ k : ℝ, (∀ x > 0, f x > k * x) → k < Real.exp 1 - 2 :=
sorry

end tangent_line_at_zero_f_ge_neg_x2_plus_x_range_k_l577_577179


namespace Maria_needs_72_nuts_l577_577630

theorem Maria_needs_72_nuts
    (fraction_nuts : ℚ := 1 / 4)
    (percentage_chocolate_chips : ℚ := 40 / 100)
    (nuts_per_cookie : ℕ := 2)
    (total_cookies : ℕ := 60) :
    (total_cookies * ((fraction_nuts + (1 - fraction_nuts - percentage_chocolate_chips)) * nuts_per_cookie).toRat) = 72 :=
by
    sorry

end Maria_needs_72_nuts_l577_577630


namespace Grant_score_is_100_l577_577201

/-- Definition of scores --/
def Hunter_score : ℕ := 45

def John_score (H : ℕ) : ℕ := 2 * H

def Grant_score (J : ℕ) : ℕ := J + 10

/-- Theorem to prove Grant's score --/
theorem Grant_score_is_100 : Grant_score (John_score Hunter_score) = 100 := 
  sorry

end Grant_score_is_100_l577_577201


namespace tank_weight_when_full_l577_577424

theorem tank_weight_when_full (p q : ℝ) (x y : ℝ)
  (h1 : x + (3/4) * y = p)
  (h2 : x + (1/3) * y = q) :
  x + y = (8/5) * p - (8/5) * q :=
by
  sorry

end tank_weight_when_full_l577_577424


namespace max_sin4_cos6_l577_577633

theorem max_sin4_cos6 : ∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ 2 * Real.pi) → sin(θ)^4 + cos(θ)^6 ≤ 1 := by
  sorry

end max_sin4_cos6_l577_577633


namespace constant_sequence_l577_577892

theorem constant_sequence (a : ℕ → ℕ) (h : ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → (i + j) ∣ (i * a i + j * a j)) :
  ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → a i = a j :=
by
  sorry

end constant_sequence_l577_577892


namespace distance_AC_in_terms_of_M_l577_577681

-- Define the given constants and the relevant equations
variables (M x : ℝ) (AB BC AC : ℝ)
axiom distance_eq_add : AB = M + BC
axiom time_AB : (M + x) / 7 = x / 5
axiom time_BC : BC = x
axiom time_S : (M + x + x) = AC

theorem distance_AC_in_terms_of_M : AC = 6 * M :=
by
  sorry

end distance_AC_in_terms_of_M_l577_577681


namespace planes_parallel_imp_lines_not_intersect_l577_577958

-- Define the conditions
variables (m n : Set Point) (α β : Set Point)
variables [IsLine m] [IsLine n] [IsPlane α] [IsPlane β]
variable [SubOf m α]
variable [SubOf n β]

-- Define the statement to be proven
theorem planes_parallel_imp_lines_not_intersect (h : α ∥ β) : ∀ p, p ∈ m → ∀ q, q ∈ n → p ≠ q :=
by
  sorry

end planes_parallel_imp_lines_not_intersect_l577_577958


namespace find_two_digit_number_l577_577033

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l577_577033


namespace leap_year_1996_l577_577433

def divisible_by (n m : ℕ) : Prop := m % n = 0

def is_leap_year (y : ℕ) : Prop :=
  (divisible_by 4 y ∧ ¬divisible_by 100 y) ∨ divisible_by 400 y

theorem leap_year_1996 : is_leap_year 1996 :=
by
  sorry

end leap_year_1996_l577_577433


namespace initial_sheep_count_l577_577671

theorem initial_sheep_count (S : ℕ) :
  let S1 := S - (S / 3 + 1 / 3)
  let S2 := S1 - (S1 / 4 + 1 / 4)
  let S3 := S2 - (S2 / 5 + 3 / 5)
  S3 = 409
  → S = 1025 := 
by 
  sorry

end initial_sheep_count_l577_577671


namespace least_divisor_of_1050_is_1050_l577_577376

theorem least_divisor_of_1050_is_1050 : ∃ d, (d > 1049) ∧ (1049 + 1) % d = 0 ∧ d = 1050 :=
by {
  use 1050,
  apply and.intro,
  { linarith },
  apply and.intro,
  { exact Nat.mod_self 1050 },
  { refl }
}

end least_divisor_of_1050_is_1050_l577_577376


namespace binomial_ratio_sum_l577_577340

theorem binomial_ratio_sum (n k : ℕ) 
  (h1 : binomial n k = binomial (n - 1) k + binomial (n - 1) (k - 1))
  (h2 : binomial n (k + 1) = binomial n k + binomial (n - 1) k)
  (h3 : binomial n (k + 2) = binomial n (k + 1) + binomial (n - 1) (k + 1)) 
  (ratio1 : binomial n k = 1)
  (ratio2 : binomial (n) (k + 1) = 3)
  (ratio3 : binomial (n) (k + 2) = 6):
  (n = 11) ∧ (k = 2) ∧ (n + k = 13) := by
  sorry

end binomial_ratio_sum_l577_577340


namespace pencils_per_row_l577_577909

theorem pencils_per_row (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) 
    (total_pencils : ℕ) (pencils_per_row : ℕ) :
    packs = 28 →
    pencils_per_pack = 24 →
    rows = 42 →
    total_pencils = packs * pencils_per_pack →
    pencils_per_row = total_pencils / rows →
    pencils_per_row = 16 := by
    intros h1 h2 h3 h4 h5
    rw [h1, h2] at h4
    have h6 : total_pencils = 672 := by
        rw [h1, h2]
        exact h4
    rw [h6, h3] at h5
    exact h5

end pencils_per_row_l577_577909


namespace max_pages_l577_577075

/-- Prove that the maximum number of pages the book has is 208 -/
theorem max_pages (pages: ℕ) (h1: pages ≥ 16 * 12 + 1) (h2: pages ≤ 13 * 16) 
(h3: pages ≥ 20 * 10 + 1) (h4: pages ≤ 11 * 20) : 
  pages ≤ 208 :=
by
  -- proof to be filled in
  sorry

end max_pages_l577_577075


namespace smallest_12_digit_natural_number_with_all_digits_divisible_by_36_l577_577820

theorem smallest_12_digit_natural_number_with_all_digits_divisible_by_36 :
  ∃ (n : ℕ), 
    (nat.digits 10 n).length = 12 ∧
    ∀ d ∈ list.range 10, d ∈ nat.digits 10 n ∧
    n % 36 = 0 ∧
    n = 100023457896 :=
by
  -- Proof omitted
  sorry

end smallest_12_digit_natural_number_with_all_digits_divisible_by_36_l577_577820


namespace sequences_of_length_10_l577_577206

noncomputable def a : ℕ → ℕ
| 2 := 1
| (n+1) := b n + c n

noncomputable def b : ℕ → ℕ
| 2 := 1
| (n+1) := a n

noncomputable def c : ℕ → ℕ
| 2 := 1
| (n+1) := b n

theorem sequences_of_length_10 : a 10 + b 10 + c 10 = 28 :=
by
  sorry

end sequences_of_length_10_l577_577206


namespace complement_M_in_U_l577_577989

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem to prove that the complement of M in U is (1, +∞)
theorem complement_M_in_U :
  (U \ M) = {x | 1 < x} :=
by
  sorry

end complement_M_in_U_l577_577989


namespace triangle_area_bounded_by_lines_l577_577783

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l577_577783


namespace exists_triplet_l577_577704

noncomputable def rad (N : ℕ) : ℕ :=
  (N.factors.erase_dup.prod : Multiset ℕ).prod

theorem exists_triplet (exists A B C : ℕ) (h1 : Nat.Coprime A B) (h2 : Nat.Coprime A C) (h3 : Nat.Coprime B C) (h4 : A + B = C) :
  (C > 1000 * rad (A * B * C)) := 
  sorry

end exists_triplet_l577_577704


namespace count_numbers_with_exactly_one_non_divisor_l577_577486

theorem count_numbers_with_exactly_one_non_divisor : 
  let S := {2, 3, 4, 5, 6, 7, 8, 9}
  in ∃ (numbers : Set ℕ), 
     numbers ⊆ Fin 1000 \ {0} ∧ 
     ∀ n ∈ numbers, (S.filter (λd, ¬ ∣ d n) = 1) ∧
     numbers.card = 4 :=
sorry

end count_numbers_with_exactly_one_non_divisor_l577_577486


namespace total_eggs_collected_l577_577904

theorem total_eggs_collected :
  let number_of_hens := 28.0
  let eggs_per_hen := 10.82142857
  let total_eggs := number_of_hens * eggs_per_hen
  round total_eggs = 303 :=
by 
  sorry

end total_eggs_collected_l577_577904


namespace find_k_values_l577_577109

def a_seq (k : ℕ) : ℕ → ℕ
| 0       := k
| (n + 1) := if a_seq n % 2 = 0 then a_seq n / 2 else a_seq n - b_seq n / 2 - c_seq n

def b_seq : ℕ → ℕ
| 0       := 4
| (n + 1) := if a_seq n % 2 = 0 then 2 * b_seq n else b_seq n

def c_seq : ℕ → ℕ
| 0       := 1
| (n + 1) := if a_seq n % 2 = 0 then c_seq n else b_seq n + c_seq n

def invariant (n : ℕ) : ℚ :=
  a_seq n * b_seq n + (c_seq n ^ 2) / 2

noncomputable def u_sequence (u : ℕ) : ℕ := u * (2 * u + 1)

theorem find_k_values : ∀ k < 1995, (∃ u, k = u_sequence u) → (∃ n, a_seq k n = 0) := 
by
  sorry


end find_k_values_l577_577109


namespace locus_of_intersection_is_circle_l577_577740

-- Definitions based on the conditions
variables {A B C B1 C1 : Point}
variables {O1 O2 Q1 Q2 D : Point} -- Centers and midpoint as stated in the solution

-- Conditions
def circles_touch_at_A (O1 O2 A : Point) : Prop := sorry
def line_through_A_intersects_at_BC (A B C : Point) : Prop := sorry
def another_line_intersects_at_B1C1 (B1 C1 : Point) : Prop := sorry
def B_on_first_circle (B B1 : Point) : Prop := sorry

-- Equivalent proof problem
theorem locus_of_intersection_is_circle :
  circles_touch_at_A O1 O2 A →
  line_through_A_intersects_at_BC A B C →
  another_line_intersects_at_B1C1 B1 C1 →
  B_on_first_circle B B1 →
  ∃ D : Point, midpoint O1 O2 D ∧ locus_of_intersections A B C B1 C1 D :=
sorry

end locus_of_intersection_is_circle_l577_577740


namespace sum_log2_floor_l577_577091

theorem sum_log2_floor (N : ℕ) (hN : 1 ≤ N ∧ N ≤ 2048) :
  ∑ N in finset.range 2048, nat.log N = 6157 := sorry

end sum_log2_floor_l577_577091


namespace batsman_average_after_17th_inning_l577_577399

theorem batsman_average_after_17th_inning (A : ℚ):
  (A + 3 = (16 * A + 66) / 17) → (A + 3 = 18) :=
by
  intro h
  let prior_avg := 15
  have h1 : A = prior_avg := by linarith
  rw [h1] at h
  have h2 : prior_avg + 3 = 18 := by norm_num
  exact h2

end batsman_average_after_17th_inning_l577_577399


namespace shorter_leg_of_right_triangle_l577_577565

theorem shorter_leg_of_right_triangle (a b : ℕ) (h1 : a < b)
    (h2 : a^2 + b^2 = 65^2) : a = 16 :=
sorry

end shorter_leg_of_right_triangle_l577_577565


namespace vanessa_savings_weeks_l577_577372

-- Define the conditions as constants
def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def weekly_arcade_spending : ℕ := 15
def weekly_snack_spending : ℕ := 5

-- The theorem statement based on the problem
theorem vanessa_savings_weeks : 
  ∃ (n : ℕ), (n * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings) ≥ dress_cost ∧ 
             (n - 1) * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings < dress_cost := by
  sorry

end vanessa_savings_weeks_l577_577372


namespace ratio_S1_S2_l577_577475

theorem ratio_S1_S2 : 
  let S1 := ∑ k in finset.range 2019, (-1) ^ (k + 1) * (1 / 3 ^ (k + 1))
  let S2 := ∑ k in finset.range 2019, (-1) ^ (k + 1) * (1 / 3 ^ (k + 1))
  S1 / S2 = 1 := by 
  sorry

end ratio_S1_S2_l577_577475


namespace range_of_a_l577_577531

theorem range_of_a 
    (a : ℝ)
    (ha : 0 < a)
    (P : ℝ × ℝ)
    (on_curve : P.1 ^ 2 + P.2 ^ 2 - 4 * sqrt 3 * P.1 - 4 * P.2 + 7 = 0)
    (angle_APB : ∠ (a, 0) P (-a, 0) = π / 2) :
    1 ≤ a ∧ a ≤ 7 := by
  sorry

end range_of_a_l577_577531


namespace total_area_of_smaller_squares_l577_577405

-- Definitions from the problem statement
def larger_square_side_length : ℝ := 2
def radius_of_inscribed_circle : ℝ := larger_square_side_length / 2
def side_length_of_smaller_square : ℝ :=
  2 - Real.sqrt 2

-- Based on the problem setup and calculated in the solution
def area_of_one_smaller_square : ℝ := side_length_of_smaller_square ^ 2
def total_area_of_four_smaller_squares : ℝ := 4 * area_of_one_smaller_square

-- The assertion or theorem to be proven
theorem total_area_of_smaller_squares :
  total_area_of_four_smaller_squares = (48 - 32 * Real.sqrt 2) / 9 :=
by
  -- This is where the proof would go
  -- Including the various steps taken in the original problem solution
  sorry

end total_area_of_smaller_squares_l577_577405


namespace greatest_k_for_3_l577_577384

-- Define the product of integers from 1 to 34
def prod_upto_34 : ℕ := (Finset.range 34).product (λ n, n + 1)

-- Define the function to count the multiplicity of a prime factor in a given number
def prime_multiplicity (p n : ℕ) : ℕ :=
  if p = 1 then 0 else Nat.iterate_on_div n (λ k b, (b + 1) * (k % p = 0)) / p

-- Theorem statement: The greatest k such that 3^k divides the product of integers from 1 to 34
theorem greatest_k_for_3 (p : ℕ) (h : p = prod_upto_34) : 
  ∃ k, 3^k ∣ p ∧ ∀ l, 3^l ∣ p → l ≤ 16 := 
by
  sorry

end greatest_k_for_3_l577_577384


namespace smallest_naive_number_max_naive_number_divisible_by_ten_l577_577931

/-- Definition of naive number -/
def is_naive_number (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  (a = d + 6) ∧ (b = c + 2)

/-- Definition of P function -/
def P (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  3 * (a + b) + c + d

/-- Definition of Q function -/
def Q (M : ℕ) : ℤ :=
  let a := M / 1000
  a - 5

namespace NaiveNumber

/-- Smallest naive number is 6200 -/
theorem smallest_naive_number : ∃ M : ℕ, is_naive_number M ∧ M = 6200 :=
  sorry

/-- Maximum naive number such that P(M)/Q(M) is divisible by 10 is 9313 -/
theorem max_naive_number_divisible_by_ten : ∃ M : ℕ, is_naive_number M ∧ (P(M) / Q(M)) % 10 = 0 ∧ M = 9313 :=
  sorry

end NaiveNumber

end smallest_naive_number_max_naive_number_divisible_by_ten_l577_577931


namespace evaluate_expression_l577_577119

theorem evaluate_expression : 2 + (3 / (4 + (5 / 6))) = 76 / 29 := 
by
  sorry

end evaluate_expression_l577_577119


namespace Annie_cookies_sum_l577_577071

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end Annie_cookies_sum_l577_577071


namespace sequence_terms_and_sum_l577_577251

theorem sequence_terms_and_sum (a : ℕ → ℕ) (a₀ : ℕ) (a₁ : ℕ)
    (h₁ : a 0 = 1)
    (h₂ : ∀ n, a (n + 1) - 3 * a n = 9 * 3 ^ (n - 1)) :
    (a 1 = 12 ∧ a 2 = 63) ∧
    (let S_n := ∑ i in finset.range n, (a i : ℚ) / 3 ^ i in S_n = (3 * n * n - n) / 6) :=
by
  sorry

end sequence_terms_and_sum_l577_577251


namespace infinite_grid_can_be_tiled_with_crosses_l577_577204

/--
  A cross consists of one central cell and four adjacent cells (up, down, left, right).
  We aim to prove that the infinite grid plane can be entirely tiled with such crosses.
-/
theorem infinite_grid_can_be_tiled_with_crosses :
  ∃ (tiling : ℤ × ℤ → option (ℤ × ℤ)), 
    (∀ x y, ∃ c, tiling (x, y) = some c ∧ 
                  ((c = (x, y)) ∨ (c = (x + 1, y)) ∨ (c = (x - 1, y)) ∨ 
                   (c = (x, y + 1)) ∨ (c = (x, y - 1)))) :=
sorry

end infinite_grid_can_be_tiled_with_crosses_l577_577204


namespace B_completes_work_in_n_days_l577_577012

-- Define the conditions
def can_complete_work_A_in_d_days (d : ℕ) : Prop := d = 15
def fraction_of_work_left_after_working_together (t : ℕ) (fraction : ℝ) : Prop :=
  t = 5 ∧ fraction = 0.41666666666666663

-- Define the theorem to be proven
theorem B_completes_work_in_n_days (d t : ℕ) (fraction : ℝ) (x : ℕ) 
  (hA : can_complete_work_A_in_d_days d) 
  (hB : fraction_of_work_left_after_working_together t fraction) : x = 20 :=
sorry

end B_completes_work_in_n_days_l577_577012


namespace find_first_term_and_difference_l577_577717

-- Define the two conditions given in the problem.
def a3 : ℕ → ℝ := λ n, 3
def a11 : ℕ → ℝ := λ n, 15

-- Define the arithmetic sequence using the formula a_n = a_1 + (n-1) * d
def arithmetic_seq (a1 d : ℝ) : ℕ → ℝ := λ n, a1 + (n - 1) * d

-- State the theorem with the conditions and the goal (proof not provided, using sorry).
theorem find_first_term_and_difference :
  (∃ a1 d : ℝ, arithmetic_seq a1 d 3 = 3 ∧ arithmetic_seq a1 d 11 = 15 ∧ a1 = 0 ∧ d = 3 / 2) := 
sorry

end find_first_term_and_difference_l577_577717


namespace litres_from_vessel_a_is_four_l577_577333

theorem litres_from_vessel_a_is_four
    (concentration_a : ℝ) (concentration_b : ℝ) (concentration_c : ℝ)
    (volume_b : ℝ) (volume_c : ℝ) (resultant_concentration : ℝ) :
    concentration_a = 0.45 → 
    concentration_b = 0.3 → 
    concentration_c = 0.1 → 
    volume_b = 5 → 
    volume_c = 6 → 
    resultant_concentration = 0.26 →
    (∃ x : ℝ, 0.45 * x + 0.3 * 5 + 0.1 * 6 = 0.26 * (x + 5 + 6) ∧ x = 4) :=
begin
  intros h1 h2 h3 h4 h5 h6,
  use 4,
  split,
  { rw [h1, h2, h3, h4, h5, h6],
    norm_num },
  { refl },
end

end litres_from_vessel_a_is_four_l577_577333


namespace triangle_median_inequality_l577_577946

variable (a b c m_a m_b m_c D : ℝ)

-- Assuming the conditions are required to make the proof valid
axiom median_formula_m_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2
axiom median_formula_m_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2
axiom median_formula_m_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2

theorem triangle_median_inequality : 
  a^2 + b^2 <= m_c * 6 * D ∧ b^2 + c^2 <= m_a * 6 * D ∧ c^2 + a^2 <= m_b * 6 * D → 
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b <= 6 * D := 
by
  sorry

end triangle_median_inequality_l577_577946


namespace length_of_YW_l577_577608

theorem length_of_YW
  (X Y Z W : Point)
  (hYZ_right_triangle : is_right_triangle Y Z X)
  (hY_angle : angle Y = 90)
  (h_circle_diameter : diameter Y Z circle.diameter)
  (h_intersect : circle.intersects X Z W)
  (h_area : triangle_area Y Z X = 98)
  (h_base : length X Z = 14) :
  length Y W = 14 :=
sorry

end length_of_YW_l577_577608


namespace prism_volume_l577_577328

variable (a : ℝ)

theorem prism_volume 
  (h_base_eq_tri : ∃ u v w : ℝ, equilateral_triangle u v w
                         ∧ base_area = (√3 / 4) * (a ^ 2)
                         ∧ projection_area = 2 * base_area)
  (h_sphere_passing : ∃ A B A₁ C₁ : Point, radius (sphere_passing A B A₁ C₁) = a) :
  prism_volume = (√3 / 4) * (a ^ 3) :=
sorry  -- Proof to be provided.

end prism_volume_l577_577328


namespace solve_vector_constants_l577_577893

theorem solve_vector_constants : 
    ∃ (a b: ℝ), 
        3 * a + b = 5 ∧ 4 * a - 2 * b = 0 ∧ a = 1 ∧ b = 2 :=
by
  use 1, 2
  split; linarith
  split; linarith
  split; refl
  refl

end solve_vector_constants_l577_577893


namespace head_start_distance_l577_577637

noncomputable def distance (speed time : ℕ) : ℕ := speed * time

theorem head_start_distance :
  ∀ (speed_cristina speed_nicky time : ℕ),
    speed_cristina = 6 →
    speed_nicky = 3 →
    time = 12 →
    (distance speed_cristina time) - (distance speed_nicky time) = 36 :=
by
  intros speed_cristina speed_nicky time h1 h2 h3
  rw [h1, h2, h3]
  simp [distance]
  sorry

end head_start_distance_l577_577637


namespace find_n_from_lcms_l577_577171

theorem find_n_from_lcms (n : ℕ) (h_pos : n > 0) (h_lcm1 : Nat.lcm 40 n = 200) (h_lcm2 : Nat.lcm n 45 = 180) : n = 100 := 
by
  sorry

end find_n_from_lcms_l577_577171


namespace asymptote_of_hyperbola_l577_577339

theorem asymptote_of_hyperbola :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) → y = x / 2 ∨ y = - x / 2 :=
sorry

end asymptote_of_hyperbola_l577_577339


namespace product_of_common_divisors_l577_577136

theorem product_of_common_divisors:
  ∀ d : ℤ, d ∣ 180 → d ∣ 30 → d ≠ 0 → (∏ x in {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} , x) = 1 :=
by {
  sorry
}

end product_of_common_divisors_l577_577136


namespace circular_table_permutations_l577_577117

/--
Given 8 chairs evenly spaced around a circular table, and 8 people initially seated in each chair, 
each person gets up and sits down in a different, non-adjacent chair, 
so that again each chair is occupied by one person.
Prove that there are exactly 80 valid permutations of seating.
-/
theorem circular_table_permutations : 
  let chairs := {1, 2, 3, 4, 5, 6, 7, 8}
  let permutations := {σ : Equiv.Perm chairs | ∀ i, σ i ≠ i ∧ σ i ≠ (i + 1) % 8 ∧ σ i ≠ (i - 1) % 8}
  permutations.card = 80 := 
by
  sorry

end circular_table_permutations_l577_577117


namespace circle_center_sum_l577_577897

theorem circle_center_sum {x y : ℝ} (h : x^2 + y^2 - 10*x + 4*y + 15 = 0) :
  (x, y) = (5, -2) ∧ x + y = 3 :=
by
  sorry

end circle_center_sum_l577_577897


namespace total_amount_correct_l577_577831

namespace ProofExample

def initial_amount : ℝ := 3

def additional_amount : ℝ := 6.8

def total_amount (initial : ℝ) (additional : ℝ) : ℝ := initial + additional

theorem total_amount_correct : total_amount initial_amount additional_amount = 9.8 :=
by
  sorry

end ProofExample

end total_amount_correct_l577_577831


namespace tangent_line_eq_at_origin_max_min_values_l577_577186

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

-- Problem 1: Equation of tangent line at (0, f(0)) = (0, 1)
theorem tangent_line_eq_at_origin :
  let f' (x : ℝ) := Real.exp x * (Real.cos x - Real.sin x) - 1
  in f' 0 = 0 ∧ f 0 = 1 ∧ ∀ x, f x = 1 → x = 0 :=
sorry

-- Problem 2: Maximum and minimum values on [0, π/2]
theorem max_min_values :
  let p : ℝ := Real.pi / 2
  in (∀ x ∈ Set.Icc (0 : ℝ) p, f x ≤ 1) ∧ (∃ x ∈ Set.Icc (0 : ℝ) p, f x = 1) ∧
     (∀ x ∈ Set.Icc (0 : ℝ) p, f x ≥ -p / 2) ∧ (∃ x ∈ Set.Icc (0 : ℝ) p, f x = -p / 2) :=
sorry

end tangent_line_eq_at_origin_max_min_values_l577_577186


namespace sum_numerator_denominator_cos_gamma_l577_577560

noncomputable def radius_of_chord (length : ℝ) (cos_theta : ℝ) : ℝ :=
  Real.sqrt (length^2 / (2 * (1 - cos_theta)))

noncomputable def cos_from_length (length : ℝ) (r : ℝ) : ℝ :=
  1 - length^2 / (2 * r^2)

theorem sum_numerator_denominator_cos_gamma :
  ∃ (p q : ℕ), 
  let γ := Real.acos ((radius_of_chord 5 γ)),
  let δ := Real.acos ((radius_of_chord 7 δ)),
  let r := radius_of_chord 12 (γ + δ),
  let cos_γ := cos_from_length 5 r,
  let cos_γ_rat := Rat.ofReal cos_γ,
  cos_γ_rat.num.natAbs + cos_γ_rat.denom = p + q :=
begin
  sorry
end

end sum_numerator_denominator_cos_gamma_l577_577560


namespace train_length_is_sixteenth_mile_l577_577425

theorem train_length_is_sixteenth_mile
  (train_speed : ℕ)
  (bridge_length : ℕ)
  (man_speed : ℕ)
  (cross_time : ℚ)
  (man_distance : ℚ)
  (length_of_train : ℚ)
  (h1 : train_speed = 80)
  (h2 : bridge_length = 1)
  (h3 : man_speed = 5)
  (h4 : cross_time = bridge_length / train_speed)
  (h5 : man_distance = man_speed * cross_time)
  (h6 : length_of_train = man_distance) :
  length_of_train = 1 / 16 :=
by sorry

end train_length_is_sixteenth_mile_l577_577425


namespace original_value_of_gift_card_l577_577632

theorem original_value_of_gift_card (x : ℝ) (h1 : x / 2) (h2 : 3 * x / 8 = 75) : x = 200 := 
by
  sorry

end original_value_of_gift_card_l577_577632


namespace count_operations_to_one_tile_l577_577050

theorem count_operations_to_one_tile :
  let remove_perfect_squares_and_cubes (n : ℕ) := 
    n - ((finset.range (n + 1)).filter (λ k, ∃ m, k = m * m ∨ k = m * m * m)).card in
  (nat.iterate remove_perfect_squares_and_cubes 7 121) = 1 :=
sorry

end count_operations_to_one_tile_l577_577050


namespace find_first_term_and_difference_l577_577716

-- Define the two conditions given in the problem.
def a3 : ℕ → ℝ := λ n, 3
def a11 : ℕ → ℝ := λ n, 15

-- Define the arithmetic sequence using the formula a_n = a_1 + (n-1) * d
def arithmetic_seq (a1 d : ℝ) : ℕ → ℝ := λ n, a1 + (n - 1) * d

-- State the theorem with the conditions and the goal (proof not provided, using sorry).
theorem find_first_term_and_difference :
  (∃ a1 d : ℝ, arithmetic_seq a1 d 3 = 3 ∧ arithmetic_seq a1 d 11 = 15 ∧ a1 = 0 ∧ d = 3 / 2) := 
sorry

end find_first_term_and_difference_l577_577716


namespace albaszu_machine_productivity_l577_577348

-- Definitions associated with the problem conditions.
def initial_trees : ℕ := 10
def initial_hours : ℕ := 8
def increased_hours : ℕ := 10
def repair_factor : ℝ := 1.5
def additional_worker1_productivity : ℝ := 0.8
def additional_worker2_productivity : ℝ := 0.6
def diminishing_return : ℝ := 0.10

-- Calculations based on the problem conditions.
def new_productivity := initial_trees * repair_factor
def increased_work_hours_factor := (increased_hours:ℝ) / (initial_hours:ℝ)
def productivity_with_increased_hours := new_productivity * increased_work_hours_factor
def new_workers_productivity_factor := 1 + additional_worker1_productivity + additional_worker2_productivity
def productivity_with_new_workers := productivity_with_increased_hours * new_workers_productivity_factor
def final_productivity := productivity_with_new_workers * (1 - diminishing_return)

-- Round down the final productivity to the nearest whole number.
def final_daily_trees := final_productivity.toNat

-- The theorem stating the expected outcome.
theorem albaszu_machine_productivity : final_daily_trees = 35 := by
  sorry  -- Proof steps to be completed as required

end albaszu_machine_productivity_l577_577348


namespace jellybeans_final_count_l577_577730

-- Defining the initial number of jellybeans and operations
def initial_jellybeans : ℕ := 37
def removed_first : ℕ := 15
def added_back : ℕ := 5
def removed_second : ℕ := 4

-- Defining the final number of jellybeans to prove it equals 23
def final_jellybeans : ℕ := (initial_jellybeans - removed_first) + added_back - removed_second

-- The theorem that states the final number of jellybeans is 23
theorem jellybeans_final_count : final_jellybeans = 23 :=
by
  -- The proof will be provided here if needed
  sorry

end jellybeans_final_count_l577_577730


namespace evaluate_ff_l577_577970

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then x^2 + 4 else Real.log x / Real.log (1 / 8)

theorem evaluate_ff (h : ∀ x, f x = if x ≤ 1 then x^2 + 4 else Real.log x / Real.log (1 / 8)) : 
  f (f (-2)) = -1 :=
by
  sorry

end evaluate_ff_l577_577970


namespace arithmetic_geometric_sequences_l577_577955

variables {a b : ℝ}
variable {a_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}
variable {d q : ℝ}

-- Definitions for arithmetic sequence and geometric sequence
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d
def geometric_seq (a q : ℝ) (n : ℕ) : ℝ := a * q ^ n

-- Given conditions
definition cond_a3_b3_a (a_n b_n : ℕ → ℝ) (a : ℝ) : Prop :=
  a_n 3 = a ∧ b_n 3 = a

definition cond_a6_b6_b (a_n b_n : ℕ → ℝ) (b : ℝ) : Prop :=
  a_n 6 = b ∧ b_n 6 = b

definition cond_a_greater_b (a b : ℝ) : Prop :=
  a > b

-- Given property of sequences
definition arithmetic_seq_property (a b : ℝ) : Prop :=
  ∃ d, arithmetic_seq a d 3 = a ∧ arithmetic_seq a d 6 = b

definition geometric_seq_property (a b : ℝ) : Prop :=
  ∃ q, geometric_seq a q 3 = a ∧ geometric_seq a q 6 = b

-- Function to compute a_4 and b_4
def a_4 (a d : ℝ) : ℝ := arithmetic_seq a d 4
def b_4 (a q : ℝ) : ℝ := geometric_seq a q 4

-- Function to compute a_5 and b_5
def a_5 (a d : ℝ) : ℝ := arithmetic_seq a d 5
def b_5 (a q : ℝ) : ℝ := geometric_seq a q 5

-- The main theorem we wish to prove
theorem arithmetic_geometric_sequences (a_n b_n : ℕ → ℝ) (a b : ℝ) (d q : ℝ)
  (h1 : cond_a3_b3_a a_n b_n a)
  (h2 : cond_a6_b6_b a_n b_n b)
  (h3 : cond_a_greater_b a b)
  (h4 : arithmetic_seq_property a b)
  (h5 : geometric_seq_property a b)
  (h6 : (a_4 a d - b_4 a q) * (a_5 a d - b_5 a q) < 0) :
  a * b < 0 := sorry

end arithmetic_geometric_sequences_l577_577955


namespace car_average_speed_approx_l577_577900

-- Define the conditions
def first_hour_distance : ℕ := 145
def second_hour_distance : ℕ := 60
def stop_time_minutes : ℕ := 20
def fluctuation_speeds : list ℕ := [45, 100]

-- Define the function to calculate average speed during fluctuations
def average_speed (speeds : list ℕ) : ℝ :=
  (speeds.foldr (+) 0) / (speeds.length : ℝ)

-- Define the total distance traveled
def total_distance : ℝ :=
  first_hour_distance + second_hour_distance + 0 + 
  (average_speed fluctuation_speeds * 1)

-- Define the total time including the stop in hours
def total_time : ℝ :=
  4 + (stop_time_minutes.to_real / 60)

-- Define the average speed for the trip
def average_speed_trip : ℝ := total_distance / total_time

-- Prove that the average speed is approximately 64.06 km/hr
theorem car_average_speed_approx : abs (average_speed_trip - 64.06) < 0.01 :=
by sorry

end car_average_speed_approx_l577_577900


namespace download_time_l577_577103

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l577_577103


namespace solve_equation_l577_577351

open Real

theorem solve_equation (x : ℝ) (h : x > 0) :
  x ^ log10 x = (x^4) / 1000 ↔ x = 10 ∨ x = 1000 :=
sorry

end solve_equation_l577_577351


namespace fitness_club_alpha_is_more_advantageous_l577_577744

-- Define the costs and attendance pattern constants
def yearly_cost_alpha : ℕ := 11988
def monthly_cost_beta : ℕ := 1299
def weeks_per_month : ℕ := 4

-- Define the attendance pattern
def attendance_pattern : List ℕ := [3 * weeks_per_month, 2 * weeks_per_month, 1 * weeks_per_month, 0 * weeks_per_month]

-- Compute the total visits in a year for regular attendance
def total_visits (patterns : List ℕ) : ℕ :=
  patterns.sum * 3

-- Compute the total yearly cost for Beta when considering regular attendance
def yearly_cost_beta (monthly_cost : ℕ) : ℕ :=
  monthly_cost * 12

-- Calculate cost per visit for each club with given attendance
def cost_per_visit (total_cost : ℕ) (total_visits : ℕ) : ℚ :=
  total_cost / total_visits

theorem fitness_club_alpha_is_more_advantageous :
  cost_per_visit yearly_cost_alpha (total_visits attendance_pattern) <
  cost_per_visit (yearly_cost_beta monthly_cost_beta) (total_visits attendance_pattern) :=
by
  sorry

end fitness_club_alpha_is_more_advantageous_l577_577744


namespace triangle_area_bounded_by_lines_l577_577779

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l577_577779


namespace combined_height_is_9_l577_577599

def barrys_reach : ℝ := 5 -- Barry can reach apples that are 5 feet high

def larrys_full_height : ℝ := 5 -- Larry's full height is 5 feet

def larrys_shoulder_height : ℝ := larrys_full_height * 0.8 -- Larry's shoulder height is 20% less than his full height

def combined_reach (b_reach : ℝ) (l_shoulder : ℝ) : ℝ := b_reach + l_shoulder

theorem combined_height_is_9 : combined_reach barrys_reach larrys_shoulder_height = 9 := by
  sorry

end combined_height_is_9_l577_577599


namespace vector_magnitude_given_conditions_l577_577170

variable (a b : ℝ^3)
variable (theta : ℝ)

-- Definitions based on given conditions
def mag_a := ∥a∥ -- magnitude of vector a
def mag_b := ∥b∥ -- magnitude of vector b
def angle := θ = 120 * (π / 180) -- Convert degrees to radians

-- Stating the problem to be proven
theorem vector_magnitude_given_conditions :
  ∥2 • a + b∥ = sqrt 13 :=
by
  sorry

end vector_magnitude_given_conditions_l577_577170


namespace find_m_plus_n_l577_577368

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

noncomputable def height (area : ℝ) (base : ℝ) : ℝ :=
  2 * area / base

theorem find_m_plus_n :
  let ab : ℝ := 12
  let bc : ℝ := 25
  let ca : ℝ := 17
  let area_abc : ℝ := triangle_area ab bc ca
  let height_ad : ℝ := height area_abc bc
  -- β is computed here
  let β : ℝ := 36 / 125
  let m := 36
  let n := 125
  in m + n = 161 :=
by
  sorry

end find_m_plus_n_l577_577368


namespace range_of_a_l577_577526

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 2 → y = 3 * (x - a) ^ 2 → y increases as x increases) → a ≤ 2 :=
  sorry

end range_of_a_l577_577526


namespace find_a_l577_577549

noncomputable def coeff_of_x_pow_six (a : ℝ) := 
  let c := (x - (1:x)⁻¹)^(10:ℤ)
  let p := (x^2 + a) * c
  p.coeff (6:ℤ)

theorem find_a (a : ℝ) (h : coeff_of_x_pow_six a = -30) : a = 2 :=
by
  sorry

end find_a_l577_577549


namespace alpha_beta_sum_two_l577_577193

theorem alpha_beta_sum_two (α β : ℝ) 
  (hα : α^3 - 3 * α^2 + 5 * α - 17 = 0)
  (hβ : β^3 - 3 * β^2 + 5 * β + 11 = 0) : 
  α + β = 2 :=
by
  sorry

end alpha_beta_sum_two_l577_577193


namespace number_of_possible_values_l577_577343

-- Definitions for the conditions
def log₁₀ (x : ℝ) := Real.log x / Real.log 10
def isTriangle (a b c : ℝ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

-- Theorem statement for the proof problem
theorem number_of_possible_values (n : ℕ) :
  let a := log₁₀ 12
  let b := log₁₀ 75 in
  (∃ n : ℕ, 7 ≤ n ∧ n < 900 ∧ isTriangle a b (log₁₀ n)) →
  set.size {n : ℕ | 7 ≤ n ∧ n < 900 ∧ isTriangle a b (log₁₀ n)} = 893 := 
sorry

end number_of_possible_values_l577_577343


namespace sum_of_squares_three_not_square_sum_of_squares_six_not_square_sum_of_squares_eleven_is_square_l577_577356

noncomputable def sum_of_squares (n : ℕ) (x : ℤ) : ℤ :=
  ∑ i in finset.range n, (x - (n / 2 : ℕ) + i) ^ 2

-- Prove that the sum of squares of 3 consecutive integers cannot be a perfect square
theorem sum_of_squares_three_not_square (x : ℤ) :
  ¬ ∃ k : ℤ, sum_of_squares 3 x = k ^ 2 :=
sorry

-- Prove that the sum of squares of 6 consecutive integers cannot be a perfect square
theorem sum_of_squares_six_not_square (x : ℤ) :
  ¬ ∃ k : ℤ, sum_of_squares 6 x = k ^ 2 :=
sorry

-- Example: sums of squares of 11 consecutive integers can be a perfect square
def specific_example := 23
theorem sum_of_squares_eleven_is_square (x : ℤ) :
  sum_of_squares 11 specific_example = 11 * specific_example ^ 2 + 110 :=
sorry

end sum_of_squares_three_not_square_sum_of_squares_six_not_square_sum_of_squares_eleven_is_square_l577_577356


namespace area_of_triangle_bounded_by_lines_l577_577773

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l577_577773


namespace equal_roots_of_quadratic_eq_l577_577550

theorem equal_roots_of_quadratic_eq (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * real.sqrt 3 * x + k = 0) → k = 3 :=
by
  sorry

end equal_roots_of_quadratic_eq_l577_577550


namespace a5_b5_sum_l577_577609

-- Definitions of arithmetic sequences
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable
def a : ℕ → ℝ := sorry -- defining the arithmetic sequences
noncomputable
def b : ℕ → ℝ := sorry

-- Common differences for the sequences
noncomputable
def d_a : ℝ := sorry
noncomputable
def d_b : ℝ := sorry

-- Conditions given in the problem
axiom a1_b1_sum : a 1 + b 1 = 7
axiom a3_b3_sum : a 3 + b 3 = 21
axiom a_is_arithmetic : arithmetic_seq a d_a
axiom b_is_arithmetic : arithmetic_seq b d_b

-- Theorem to be proved
theorem a5_b5_sum : a 5 + b 5 = 35 := 
by sorry

end a5_b5_sum_l577_577609


namespace miles_per_gallon_correct_l577_577598

def total_distance : ℕ := 15 + 6 + 2 + 4 + 11

def total_gallons : ℕ := 2

def miles_per_gallon : ℕ := total_distance / total_gallons

theorem miles_per_gallon_correct : miles_per_gallon = 19 := by
  have h : total_distance = 38 := by rfl
  have h2 : total_gallons = 2 := by rfl
  show miles_per_gallon = 19
  calc
    miles_per_gallon
    = total_distance / total_gallons := rfl
    ... = 38 / 2 := by rw [h, h2]
    ... = 19 := by norm_num

end miles_per_gallon_correct_l577_577598


namespace points_on_same_circle_l577_577165

theorem points_on_same_circle (a : ℕ → ℂ) (s : ℝ) (q : ℂ) (hne: ∀ i, a i ≠ 0) (hc : ∀ i, a (i + 1) = a i * q)
  (h_sum_a : (∑ i in finset.range 5, a i) = 4)
  (h_sum_reciprocal : (∑ i in finset.range 5, (a i)⁻¹) = s)
  (hs : s ∈ set.Icc (-2:ℝ) 2) :
  |q| = 1 := sorry

end points_on_same_circle_l577_577165


namespace cannot_form_figure_C_l577_577248

-- Assume definitions for pieces and figures
def pieces : List (Set (Int × Int)) -- A list of sets representing the pieces
def figures : List (Set (Int × Int)) -- A list of sets representing the figures

-- Definitions
def area (shape : Set (Int × Int)) : Int := shape.card

theorem cannot_form_figure_C (h_pieces : pieces.card = 6)
  (h_figures : figures.card = 4)
  (h_total_area : (pieces.map area).sum = 18)
  (h_FigureA : ∃ s ∈ pieces, s = figures.nthLe 0 sorry)
  (h_FigureB : ∃ s ∈ pieces, s = figures.nthLe 1 sorry)
  (h_FigureC : ¬∃ s ∈ pieces, s = figures.nthLe 2 sorry)
  (h_FigureD : ∃ s ∈ pieces, s = figures.nthLe 3 sorry) : 
  ¬∃ s ∈ pieces, s = figures.nthLe 2 sorry :=
by
  -- Proof here, using sorry to skip
  sorry

end cannot_form_figure_C_l577_577248


namespace book_arrangement_l577_577811

theorem book_arrangement (m_books h_books : ℕ) (m_conditions : m_books = 4) (h_conditions : h_books = 6) : 
  (∃! total_ways, total_ways = (4 * 3 * (7!))) :=
by {
  have books_count : ℕ := 4 * 3 * (fact 7),
  use books_count,
  simp,
  exact books_count,
}

end book_arrangement_l577_577811


namespace value_of_f_for_positive_x_l577_577495

-- Definition of even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Hypotheses
variables (f : ℝ → ℝ)
variable h_even : is_even_function f
variable h_neg : ∀ x : ℝ, x < 0 → f x = x * (x + 1)

-- Goal
theorem value_of_f_for_positive_x :
  ∀ x : ℝ, x > 0 → f x = x * (x - 1) :=
sorry

end value_of_f_for_positive_x_l577_577495


namespace garden_feet_count_l577_577247

theorem garden_feet_count 
  (dogs : ℕ) (ducks : ℕ) (cats : ℕ) (birds : ℕ) (insects : ℕ) 
  (special_dog_legs : ℕ) (special_cat_legs : ℕ) (mutant_bird_legs : ℕ)
  (hdogs : dogs = 6) (hducks : ducks = 2) (hcats : cats = 4) 
  (hbirds : birds = 7) (hinsects : insects = 10) 
  (hspecial_dog_legs : special_dog_legs = 3 = 1) 
  (hspecial_cat_legs : special_cat_legs = 3 = 1) 
  (hmutant_bird_legs : mutant_bird_legs = 3 = 1) :
  let normal_dogl_legs := (dogs - special_dog_legs) * 4
  let special_dog_legs_sum := special_dog_legs * 3
  let total_dog_legs := normal_dogl_legs + special_dog_legs_sum
  let total_duck_legs := ducks * 2
  let normal_cat_legs := (cats - special_cat_legs) * 4
  let special_cat_legs_sum := special_cat_legs * 3
  let total_cat_legs := normal_cat_legs + special_cat_legs_sum
  let normal_bird_legs := (birds - mutant_bird_legs) * 2
  let total_mutant_bird_legs := mutant_bird_legs * 3
  let total_bird_legs := normal_bird_legs + total_mutant_bird_legs
  let total_insect_legs := insects * 6
  let total_feet := total_dog_legs + total_duck_legs + total_cat_legs + total_bird_legs + total_insect_legs
  in total_feet = 117 := by
  sorry

end garden_feet_count_l577_577247


namespace sum_log2_floor_l577_577090

theorem sum_log2_floor (N : ℕ) (hN : 1 ≤ N ∧ N ≤ 2048) :
  ∑ N in finset.range 2048, nat.log N = 6157 := sorry

end sum_log2_floor_l577_577090


namespace ellipse_standard_eq_max_inradius_of_triangle_l577_577948

variable (a b : ℝ)
variable (e : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : b = sqrt 3)
variable (h4 : a * e = b)
variable (h5 : e = 1 / 2)

theorem ellipse_standard_eq : (a = 2 ∧ b = sqrt 3) → (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ ((x, y) ∈ set_of (λ p, (p.1^2 / a^2 + p.2^2 / b^2) = 1))) :=
by intros; simp [a, b]

theorem max_inradius_of_triangle : 
∀ F1 A B : (ℝ × ℝ),
  let F2 := (a * e, 0) in
  let line_l := {x : ℝ × ℝ | x.2 = ( (A.1 - 1) / A.2 ) * (x.1 - 1) + B.2} in
  (∀ x y : ℝ, x ∈ line_l → y ∈ line_l → 
    ∃ R : ℝ, R ≤ 3 / 4) :=
sorry

end ellipse_standard_eq_max_inradius_of_triangle_l577_577948


namespace arithmetic_evaluation_l577_577441

theorem arithmetic_evaluation :
  -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 :=
by
  sorry

end arithmetic_evaluation_l577_577441


namespace area_of_smallest_region_l577_577451

theorem area_of_smallest_region :
  let circle_equation := ∀ (x y : ℝ), x^2 + y^2 = 16
  let v_shape := ∀ (x y : ℝ), y = |x|
  ( ∃ (x1 x2 y1 y2 : ℝ), circle_equation x1 y1 ∧ v_shape x1 y1 ∧ circle_equation x2 y2 ∧ v_shape x2 y2 ∧ 
    ∃ (r θ : ℝ), r = 4 ∧ θ = π / 2 ∧ x1 = r * cos θ ∧ y1 = r * sin θ ∧ 
      area_smallest_region x1 y1 x2 y2 r θ = 4 * π)

end area_of_smallest_region_l577_577451


namespace box_dimensions_l577_577825

theorem box_dimensions (x : ℝ) (bow_length_top bow_length_side : ℝ)
  (h1 : bow_length_top = 156 - 6 * x)
  (h2 : bow_length_side = 178 - 7 * x)
  (h_eq : bow_length_top = bow_length_side) :
  x = 22 :=
by sorry

end box_dimensions_l577_577825


namespace fitness_club_alpha_is_more_advantageous_l577_577745

-- Define the costs and attendance pattern constants
def yearly_cost_alpha : ℕ := 11988
def monthly_cost_beta : ℕ := 1299
def weeks_per_month : ℕ := 4

-- Define the attendance pattern
def attendance_pattern : List ℕ := [3 * weeks_per_month, 2 * weeks_per_month, 1 * weeks_per_month, 0 * weeks_per_month]

-- Compute the total visits in a year for regular attendance
def total_visits (patterns : List ℕ) : ℕ :=
  patterns.sum * 3

-- Compute the total yearly cost for Beta when considering regular attendance
def yearly_cost_beta (monthly_cost : ℕ) : ℕ :=
  monthly_cost * 12

-- Calculate cost per visit for each club with given attendance
def cost_per_visit (total_cost : ℕ) (total_visits : ℕ) : ℚ :=
  total_cost / total_visits

theorem fitness_club_alpha_is_more_advantageous :
  cost_per_visit yearly_cost_alpha (total_visits attendance_pattern) <
  cost_per_visit (yearly_cost_beta monthly_cost_beta) (total_visits attendance_pattern) :=
by
  sorry

end fitness_club_alpha_is_more_advantageous_l577_577745


namespace tan_150_sin_cos_identity_l577_577444

noncomputable def sin_30 := 1 / 2
noncomputable def cos_30 := real.sqrt 3 / 2

theorem tan_150 (h1 : 150 = 180 - 30)
                (h2 : sin 30 = sin_30)
                (h3 : cos 30 = cos_30) : 
                tan 150 = - (real.sqrt 3 / 3) := 
by
  have sin_150 : sin 150 = sin 30 := by rw [sin_sub, sin_30, sub_self]
  have cos_150 : cos 150 = - cos 30 := by rw [cos_sub, cos_30, neg_self]
  show tan 150 = sin 150 / cos 150
  rw [sin_150, cos_150, sin_30, cos_30]
  sorry

theorem sin_cos_identity (h1 : 150 = 180 - 30)
                         (h2 : sin 30 = sin_30)
                         (h3 : cos 30 = cos_30): 
                         sin^2 150 + cos^2 150 = 1 :=
by
  have sin_150 : sin 150 = sin 30 := by rw [sin_sub, sin_30, sub_self]
  have cos_150 : cos 150 = - cos 30 := by rw [cos_sub, cos_30, neg_self]
  rw [pow_two, pow_two]
  rw [sin_150, cos_150, sin_30, cos_30]
  sorry

end tan_150_sin_cos_identity_l577_577444


namespace pyramid_volume_l577_577851

noncomputable def volume_of_pyramid (s : ℝ) (A_abe : ℝ) (A_cde : ℝ) :=
  let h_abe := (2 * A_abe) / s
  let h_cde := (2 * A_cde) / s
  let eq1 := (16 - (h_abe - h_cde)) / 2
  let h := Math.sqrt (h_abe * h_cde) 
  (1 / 3) * s^2 * h

theorem pyramid_volume :
  volume_of_pyramid 16 120 104 = 1163 := 
by {
  sorry
}

end pyramid_volume_l577_577851


namespace similar_triangle_division_iff_right_triangle_l577_577662

noncomputable def is_right_triangle (ABC : Type) := sorry

theorem similar_triangle_division_iff_right_triangle (ABC : Type) (h : ∃ e, divides_into_two_similar_triangles ABC e) :
  is_right_triangle ABC ↔ (∃ e, divides_into_two_similar_triangles ABC e) :=
sorry

end similar_triangle_division_iff_right_triangle_l577_577662


namespace count_squares_in_G_l577_577890

def G : set (ℤ × ℤ) := {p | (1 ≤ |p.1| ∧ |p.1| ≤ 7) ∧ (1 ≤ |p.2| ∧ |p.2| ≤ 7)}

def is_square_of_side (length : ℤ) (p1 p2 p3 p4 : (ℤ × ℤ)) : Prop :=
  (p1, p2, p3, p4) forms a square with side length length -- Placeholder for actual implementation

def count_squares_of_side_at_least (side_length : ℤ) : ℕ :=
  -- Implementation that counts all squares in G of given side length or more

theorem count_squares_in_G : count_squares_of_side_at_least 6 = 20 :=
sorry

end count_squares_in_G_l577_577890


namespace find_value_l577_577971

noncomputable def F (x a : ℝ) : ℝ := ( (Real.ln x / x) ^ 2 ) + ( (a - 1) * (Real.ln x / x) ) + (1 - a)

theorem find_value (a x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) 
  (h3 : F x1 a = 0) (h4 : F x2 a = 0) (h5 : F x3 a = 0) :
  (1 - Real.ln x1 / x1) ^ 2 * (1 - Real.ln x2 / x2) * (1 - Real.ln x3 / x3) = 1 :=
sorry

end find_value_l577_577971


namespace water_needed_quarts_l577_577736

-- Definitions from conditions
def ratio_water : ℕ := 8
def ratio_lemon : ℕ := 1
def total_gallons : ℚ := 1.5
def gallons_to_quarts : ℚ := 4

-- State what needs to be proven
theorem water_needed_quarts : 
  (total_gallons * gallons_to_quarts * (ratio_water / (ratio_water + ratio_lemon))) = 16 / 3 :=
by
  sorry

end water_needed_quarts_l577_577736


namespace right_triangle_with_integer_sides_l577_577125

theorem right_triangle_with_integer_sides (k : ℤ) :
  ∃ (a b c : ℤ), a = 2*k+1 ∧ b = 2*k*(k+1) ∧ c = 2*k^2+2*k+1 ∧ (a^2 + b^2 = c^2) ∧ (c = a + 1) := by
  sorry

end right_triangle_with_integer_sides_l577_577125


namespace circle_center_on_line_and_tangent_x_axis_l577_577857

noncomputable def line_passing_through (x1 y1 x2 y2 : ℝ) : ℝ → ℝ :=
  λ x, y1 + (y2 - y1) / (x2 - x1) * (x - x1)

theorem circle_center_on_line_and_tangent_x_axis
  (x1 y1 x2 y2 : ℝ) 
  (h k : ℝ)
  (hx : x1 = 2) (hy1 : y1 = 1) (hx2 : x2 = 6) (hy2 : y2 = 3)
  (h_eq : h = 2) (k_eq : k = 1)
  (tangent_x : ∀ (x : ℝ), line_passing_through x1 y1 x2 y2 x = y1 + (y2 - y1) / (x2 - x1) * (x - x1))
  (circle_eq : ∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = 0) :
  (h, k) = (2, 1) ∧ (x ↦ (2, 1))
:=
by {
  sorry
}

end circle_center_on_line_and_tangent_x_axis_l577_577857


namespace balance_triangle_with_squares_l577_577574

noncomputable theory

def triangle := Type
def circle := Type
def square := Type

constant Δ : triangle
constant ∘ : circle
constant □ : square

constant balance1 : 3 * Δ = 5 * ∘
constant balance2 : ∘ = 2 * □

theorem balance_triangle_with_squares :
  ∃ n : ℕ, (n = 4 ∧ 
            Δ = (10 / 3) * □) := sorry

end balance_triangle_with_squares_l577_577574


namespace value_of_f_pi_area_enclosed_l577_577269

section
  variable (f : ℝ → ℝ)
  variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
  variable (h_periodic : ∀ x : ℝ, f (x + 2) = -f x)
  variable (h_restricted : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x)

  -- Proof Problem 1: value of f(pi) is -1.14
  theorem value_of_f_pi : f π = -1.14 :=
  by
    sorry

  -- Proof Problem 2: area enclosed by the graph of f(x) and the x-axis for -4 ≤ x ≤ 4 is 4
  theorem area_enclosed : 
    let area := ∫ x in -4..4, abs (f x) 
    area = 4 :=
  by
    sorry
end

end value_of_f_pi_area_enclosed_l577_577269


namespace parallelogram_K1L1M1N1_l577_577307

open EuclideanGeometry

-- We define the quadrilateral ABCD and the circle O around which it is circumscribed.
variables {A B C D O K L M N K1 L1 M1 N1 : Point}
variables (circumscribed : IsCircumscribedQuadrilateral A B C D O)
variables (angle_bisectors : 
  ExternalAngleBisectorsIntersect A B K ∧
  ExternalAngleBisectorsIntersect B C L ∧
  ExternalAngleBisectorsIntersect C D M ∧
  ExternalAngleBisectorsIntersect D A N)
variables (orthocenters : 
  OrthocenterOfTriangle A B K K1 ∧ 
  OrthocenterOfTriangle B C L L1 ∧ 
  OrthocenterOfTriangle C D M M1 ∧ 
  OrthocenterOfTriangle D A N N1)

theorem parallelogram_K1L1M1N1 : IsParallelogram K1 L1 M1 N1 :=
by
  sorry

end parallelogram_K1L1M1N1_l577_577307


namespace trigonometric_inequality_l577_577600

theorem trigonometric_inequality (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) :
  sqrt ((sin x ^ 2) / (1 + cos x ^ 2)) + sqrt ((cos x ^ 2) / (1 + sin x ^ 2)) ≥ 1 := 
sorry

end trigonometric_inequality_l577_577600


namespace smallest_distance_AB_ge_2_l577_577263

noncomputable def A (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9
noncomputable def B (x y : ℝ) : Prop := y^2 = -8 * x

theorem smallest_distance_AB_ge_2 :
  ∀ (x1 y1 x2 y2 : ℝ), A x1 y1 → B x2 y2 → dist (x1, y1) (x2, y2) ≥ 2 := by
  sorry

end smallest_distance_AB_ge_2_l577_577263


namespace ping_pong_tournament_four_clique_l577_577726

-- Define the variables and conditions
variables {G : SimpleGraph (Fin 9)}
variables (h_edges : G.edgeFinset.card = 28)

-- State the theorem
theorem ping_pong_tournament_four_clique :
  ∃ (H : Finset (Fin 9)), H.card = 4 ∧ G.IsClique H :=
by
  -- Proof goes here
  sorry

end ping_pong_tournament_four_clique_l577_577726


namespace value_not_uniquely_determined_l577_577884

variables (v : Fin 9 → ℤ) (s : Fin 9 → ℤ)

-- Given conditions
axiom announced_sums : ∀ i, s i = v ((i - 1) % 9) + v ((i + 1) % 9)
axiom sums_sequence : s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 12 ∧ s 3 = 18 ∧ s 4 = 24 ∧ s 5 = 31 ∧ s 6 = 40 ∧ s 7 = 48 ∧ s 8 = 53

-- Statement asserting the indeterminacy of v_5
theorem value_not_uniquely_determined (h: s 3 = 18) : 
  ∃ v : Fin 9 → ℤ, sorry :=
sorry

end value_not_uniquely_determined_l577_577884


namespace population_projection_l577_577832

theorem population_projection (P_current : ℝ) (rate : ℝ) (P_future : ℝ) : 
  P_current = 1.75 ∧ rate = 3.25 ∧ P_future = P_current + P_current * rate → P_future = 7.4375 :=
by {
  intros h,
  sorry
}

end population_projection_l577_577832


namespace number_of_combinations_l577_577210

theorem number_of_combinations (n k : ℕ) (h_n : n = 14) (h_k : k = 3) : 
  nat.choose n k = 364 :=
by
  rw [h_n, h_k]
  rw nat.choose_eq_factorial_div_factorial
  calc
    14! / (3! * (14 - 3)!) = 14! / (3! * 11!) : by rw nat.sub_self
    ... = (14 * 13 * 12 * 11!) / (6 * 11!) : by simp [factorial]
    ... = (14 * 13 * 12) / 6 : by rw nat.mul_div_right
    ... = 364 : by norm_num

end number_of_combinations_l577_577210


namespace river_bank_depth_l577_577817

-- Definitions related to the problem
def is_trapezium (top_width bottom_width height area : ℝ) :=
  area = 1 / 2 * (top_width + bottom_width) * height

-- The theorem we want to prove
theorem river_bank_depth :
  ∀ (top_width bottom_width area : ℝ), 
    top_width = 12 → 
    bottom_width = 8 → 
    area = 500 → 
    ∃ h : ℝ, is_trapezium top_width bottom_width h area ∧ h = 50 :=
by
  intros top_width bottom_width area ht hb ha
  sorry

end river_bank_depth_l577_577817


namespace infinite_set_k_l577_577255

theorem infinite_set_k (C : ℝ) : ∃ᶠ k : ℤ in at_top, (k : ℝ) * Real.sin k > C :=
sorry

end infinite_set_k_l577_577255


namespace lincoln_one_way_fare_l577_577292

-- Define the given conditions as assumptions
variables (x : ℝ) (days : ℝ) (total_cost : ℝ) (trips_per_day : ℝ)

-- State the conditions
axiom condition1 : days = 9
axiom condition2 : total_cost = 288
axiom condition3 : trips_per_day = 2

-- The theorem we want to prove based on the conditions
theorem lincoln_one_way_fare (h1 : total_cost = days * trips_per_day * x) : x = 16 :=
by
  -- We skip the proof for the sake of this exercise
  sorry

end lincoln_one_way_fare_l577_577292


namespace sum_of_corners_is_16_l577_577246

theorem sum_of_corners_is_16 (a b c d e f g h i : ℕ)
    (h1 : a + b + c = 12)
    (h2 : d + e + f = 12)
    (h3 : g + h + i = 12)
    (h4 : a + d + g = 12)
    (h5 : b + e + h = 12)
    (h6 : c + f + i = 12)
    (h7 : a + e + i = 12)
    (h8 : c + e + g = 12)
    (ha : a = 4)
    (hc : c = 3)
    (hg : g = 5)
    (hi : i = 4) :
  a + c + g + i = 16 :=
by
  calc
    4 + 3 + 5 + 4 = 16 := sorry

end sum_of_corners_is_16_l577_577246


namespace plain_b_area_l577_577706

theorem plain_b_area : 
  ∃ x : ℕ, (x + (x - 50) = 350) ∧ x = 200 :=
by
  sorry

end plain_b_area_l577_577706


namespace problem_1_problem_2_l577_577977

noncomputable def f (x : ℝ) : ℝ := x^2

sequence a : ℕ → ℝ
| 0       => 3
| (n+1) => 2 * f (a n - 1) + 1

def b (n : ℕ) : ℝ :=
  Real.log (a n - 1) / Real.log 2

theorem problem_1 (n : ℕ) :
  ∃ (r : ℝ), ∀ n, b (n+1) + 1 = r * (b n + 1) :=
by
  use 2
  sorry

theorem problem_2 (n : ℕ) :
  (finset.range n).sum (λ k, b k) = 2^(n+1) - 2 - n :=
by
  sorry

end problem_1_problem_2_l577_577977


namespace orthocenter_is_lattice_point_l577_577847

-- Definition of a lattice point point in 2D space
def lattice_point : Type := { p : ℝ × ℝ // p.1 ∈ ℤ ∧ p.2 ∈ ℤ }

-- Vertices of the triangle are lattice points
variables {A B C : lattice_point}

-- Coordinates of the vertices
def x1 := (A : ℝ × ℝ).1
def y1 := (A : ℝ × ℝ).2
def x2 := (B : ℝ × ℝ).1
def y2 := (B : ℝ × ℝ).2
def x3 := (C : ℝ × ℝ).1
def y3 := (C : ℝ × ℝ).2

-- The area condition
def area_condition : Prop :=
  let area := 0.5 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)| in
  area = 0.5

-- Orthocenter of the triangle is a lattice point
theorem orthocenter_is_lattice_point
  (h_area : area_condition) : ∃ H : lattice_point, ∀ H, H ∈ ℝ × ℝ :=
sorry

end orthocenter_is_lattice_point_l577_577847


namespace parallelogram_law_tetrahedron_edge_length_equation_l577_577007

-- Part (a): Parallelogram law
theorem parallelogram_law (a b d1 d2 : ℝ)
    (h1 : d1 = real.sqrt (a^2 + b^2 - 2 * a * b * real.cos 0)) 
    (h2 : d2 = real.sqrt (a^2 + b^2 + 2 * a * b * real.cos 0)) :
    2 * a^2 + 2 * b^2 = d1^2 + d2^2 := sorry

-- Part (b): Tetrahedron edge length equation
theorem tetrahedron_edge_length_equation (a b c d e f x y z : ℝ) :
    4 * (x^2 + y^2 + z^2) = a^2 + b^2 + c^2 + d^2 + e^2 + f^2 := sorry

end parallelogram_law_tetrahedron_edge_length_equation_l577_577007


namespace find_y_l577_577504

theorem find_y (x : ℝ) (h : x^2 + (1 / x)^2 = 7) : x + 1 / x = 3 :=
by
  sorry

end find_y_l577_577504


namespace probability_at_least_two_same_l577_577298

theorem probability_at_least_two_same (n : ℕ) (d : ℕ) (dice : Fin d → ℕ) 
  (fair_dice : (∀ i, 1 ≤ dice i ∧ dice i ≤ d))
  (num_dice : dice = 5) (num_sides : d = 6) : 
  (let all_diff := (Finset.univ : Finset (Fin d)).card = d  in 
   1 - (720 / 7776) = 7056 / 7776) := 
sorry

end probability_at_least_two_same_l577_577298


namespace principal_invested_years_l577_577146

-- Define the given conditions
def principal : ℕ := 9200
def rate : ℕ := 12
def interest_deficit : ℤ := 5888

-- Define the time to be proved
def time_invested : ℤ := 3

-- Define the simple interest formula
def simple_interest (P R t : ℕ) : ℕ :=
  (P * R * t) / 100

-- Define the problem statement
theorem principal_invested_years :
  ∃ t : ℕ, principal - interest_deficit = simple_interest principal rate t ∧ t = time_invested := 
by
  sorry

end principal_invested_years_l577_577146


namespace f_scaling_l577_577623

-- Define the function and conditions in terms of Lean's notation and logic
variable (f : ℝ → ℝ)

axiom f_condition : ∀ (x y : ℝ), f(x^3 + y^3) = (x + y) * ((f(x))^2 - f(x) * f(y) + f(f(y))^2)

-- The theorem we want to prove
theorem f_scaling (x : ℝ) : f (1996 * x) = 1996 * f x := sorry

end f_scaling_l577_577623


namespace num_cats_l577_577025

-- Definitions based on conditions
variables (C S K Cap : ℕ)
variable (heads : ℕ) (legs : ℕ)

-- Conditions as equations
axiom heads_eq : C + S + K + Cap = 16
axiom legs_eq : 4 * C + 2 * S + 2 * K + 1 * Cap = 41

-- Given values from the problem
axiom K_val : K = 1
axiom Cap_val : Cap = 1

-- The proof goal in terms of satisfying the number of cats
theorem num_cats : C = 5 :=
by
  sorry

end num_cats_l577_577025


namespace max_sqrt_expr_l577_577133

theorem max_sqrt_expr (x : ℝ) (hx : 0 ≤ x) (hx20 : x ≤ 20) :
  (sqrt (x + 16) + sqrt (20 - x) + 2 * sqrt x) ≤ 18 :=
sorry

end max_sqrt_expr_l577_577133


namespace smallest_divisor_l577_577591

noncomputable def even_four_digit_number (m : ℕ) : Prop :=
  1000 ≤ m ∧ m < 10000 ∧ m % 2 = 0

def divisor_ordered (m : ℕ) (d : ℕ) : Prop :=
  d ∣ m

theorem smallest_divisor (m : ℕ) (h1 : even_four_digit_number m) (h2 : divisor_ordered m 437) :
  ∃ d,  d > 437 ∧ divisor_ordered m d ∧ (∀ e, e > 437 → divisor_ordered m e → d ≤ e) ∧ d = 874 :=
sorry

end smallest_divisor_l577_577591


namespace evaluate_fraction_l577_577120

theorem evaluate_fraction (a b : ℤ) (h₁ : a = 7) (h₂ : b = -3) : (3 : ℚ) / (a - b : ℚ) = 3 / 10 := by
  subst h₁
  subst h₂
  rw [sub_neg_eq_add, (show 7 + 3 = 10 by rfl)]
  norm_num
  sorry

end evaluate_fraction_l577_577120


namespace part1_part2_l577_577180

noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2

theorem part1 (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ f x a = 0 ∧ f y a = 0) : a ∈ Ioo 0 (1 / Real.exp 1) :=
sorry

noncomputable def g (x a : ℝ) := f x a - x + a

theorem part2 (a x1 x2 λ : ℝ) (hx1_lt_x2 : x1 < x2) (hλ : λ ≥ 1)
  (h_extreme : g x1 a = g x2 a ∧ g x1 a = 0 ∧ g x2 a = 0) :
  x1 / Real.exp 1 > (Real.exp 1 / x2) ^ λ :=
sorry

end part1_part2_l577_577180


namespace total_cost_after_discount_l577_577816

noncomputable def mango_cost : ℝ := sorry
noncomputable def rice_cost : ℝ := sorry
noncomputable def flour_cost : ℝ := 21

theorem total_cost_after_discount :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (flour_cost = 21) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost) * 0.9 = 808.92 :=
by
  intros h1 h2 h3
  -- sorry as placeholder for actual proof
  sorry

end total_cost_after_discount_l577_577816


namespace proof_problem_l577_577691

noncomputable def equation_of_complement_inclination_line (m : ℝ) (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∀ (p : ℝ × ℝ), (p = P) → l (p.1, p.2) →

  let m := 1 in
  let l_eq := λ p, (p.1 - p.2 - 1 = 0) in
  let complementary_slope := -1 in
  ∃ (q : ℝ × ℝ), ((q.1 + q.2 - 3 = 0) ∧ l_eq p)

theorem proof_problem : equation_of_complement_inclination_line 1 (2, 1) (λ p : ℝ × ℝ, p.1 - p.2 - 1 = 0) :=
  sorry

end proof_problem_l577_577691


namespace max_area_garden_l577_577842

/-- Given a rectangular garden with a total perimeter of 480 feet and one side twice as long as another,
    prove that the maximum area of the garden is 12800 square feet. -/
theorem max_area_garden (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 480) : l * w = 12800 := 
sorry

end max_area_garden_l577_577842


namespace dog_treats_cost_l577_577594

theorem dog_treats_cost
  (treats_per_day : ℕ)
  (cost_per_treat : ℚ)
  (days_in_month : ℕ)
  (H1 : treats_per_day = 2)
  (H2 : cost_per_treat = 0.1)
  (H3 : days_in_month = 30) :
  treats_per_day * days_in_month * cost_per_treat = 6 :=
by sorry

end dog_treats_cost_l577_577594


namespace rectangle_area_error_l577_577386

theorem rectangle_area_error (L W : ℝ) :
  let actual_area := L * W
  let measured_length := 1.10 * L
  let measured_width := 0.95 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  let error_percentage := (error / actual_area) * 100
  in error_percentage = 4.5 :=
by sorry

end rectangle_area_error_l577_577386


namespace largest_four_digit_div_by_4_l577_577375

theorem largest_four_digit_div_by_4 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ ∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 → n ≥ m :=
by
  use 9996
  have h1 : 1000 ≤ 9996 := by sorry
  have h2 : 9996 < 10000 := by sorry
  have h3 : 9996 % 4 = 0 := by sorry
  have h4 : ∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 → 9996 ≥ m := by sorry
  exact ⟨h1, h2, h3, h4⟩

end largest_four_digit_div_by_4_l577_577375


namespace border_area_correct_l577_577422

-- Define the dimensions of the photograph
def photograph_height : ℕ := 12
def photograph_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the photograph
def area_photograph : ℕ := photograph_height * photograph_width

-- Define the total dimensions including the frame
def total_height : ℕ := photograph_height + 2 * border_width
def total_width : ℕ := photograph_width + 2 * border_width

-- Define the area of the framed area
def area_framed : ℕ := total_height * total_width

-- Define the area of the border
def area_border : ℕ := area_framed - area_photograph

theorem border_area_correct : area_border = 198 := by
  sorry

end border_area_correct_l577_577422


namespace cos_alpha_sin_beta_range_l577_577169

variable {α β : ℝ}

theorem cos_alpha_sin_beta_range (h : sin α * cos β = -1 / 2) :
  -1 / 2 ≤ cos α * sin β ∧ cos α * sin β ≤ 1 / 2 :=
sorry

end cos_alpha_sin_beta_range_l577_577169


namespace prime_solutions_l577_577917

theorem prime_solutions (p : ℕ) (n : ℕ) (hp : p.prime) :
  p^2 + n^2 = 3 * p * n + 1 ↔ (p, n) = (3, 1) ∨ (p, n) = (3, 8) :=
by sorry

end prime_solutions_l577_577917


namespace infinite_set_of_midpoints_l577_577276

variable {Point : Type*} [AddCommGroup Point] [Module ℝ Point]

def is_midpoint (S : set Point) : Prop :=
∀ (p ∈ S), ∃ (a b ∈ S), p = (a + b) / 2

theorem infinite_set_of_midpoints (S : set Point) (h : is_midpoint S) : set.infinite S :=
sorry

end infinite_set_of_midpoints_l577_577276


namespace bridesmaids_count_l577_577310

theorem bridesmaids_count
  (hours_per_dress : ℕ)
  (hours_per_week : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (dresses : ℕ) :
  hours_per_dress = 12 →
  hours_per_week = 4 →
  weeks = 15 →
  total_hours = hours_per_week * weeks →
  dresses = total_hours / hours_per_dress →
  dresses = 5 := by
  sorry

end bridesmaids_count_l577_577310


namespace leopards_arrangement_l577_577293

theorem leopards_arrangement :
  let total_leopards := 9
  let positions := total_leopards
  let shortest_leopards_ends := 2
  let tallest_leopard_middle := 1
  let remaining_leopards_fact := fact (total_leopards - shortest_leopards_ends - tallest_leopard_middle)
  let total_ways := shortest_leopards_ends * tallest_leopard_middle * remaining_leopards_fact
  total_ways = 1440 := 
by
  sorry

end leopards_arrangement_l577_577293


namespace find_mean_of_normal_distribution_l577_577153

-- Define the necessary variables and assumptions
variables {μ σ : ℝ} (X : ℝ → ℝ)

-- Assume X follows a normal distribution N(μ, σ²)
axiom normal_dist : ∀ x, X x = normal μ σ

-- State the given condition
axiom condition : P (λ x, X x ≤ 0) = P (λ x, X x ≥ 2)

-- Formalize the proof problem
theorem find_mean_of_normal_distribution (h : condition) : μ = 1 :=
  sorry

end find_mean_of_normal_distribution_l577_577153


namespace range_of_expression_l577_577148

theorem range_of_expression (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) :
  (∃ (y x : ℝ), y = 2 * sqrt (1 - t) ∧ x = sqrt t ∧ (2 / 3 ≤ (y + 2) / (x + 2) ∧ (y + 2) / (x + 2) ≤ 2)) :=
by
  use (2 * sqrt (1 - t))
  use (sqrt t)
  split
  { exact rfl }
  split
  { exact rfl }
  split
  {  -- Admissible range proof to show start ≥ 2 / 3
    sorry }
  {  -- Admissible range proof to show end ≤ 2
    sorry }

end range_of_expression_l577_577148


namespace solve_z_l577_577176

open Complex

theorem solve_z (z : ℂ) (i_unit : Complex.Im = 1) :
    (1 + Complex.i) * z = -2 * Complex.i -> 
    z = -1 - Complex.i :=
by sorry

end solve_z_l577_577176


namespace blue_yellow_percentages_l577_577237

def responses : Nat := 75 + 90 + 55 + 80 + 50

def percentage (part total : Nat) : Float :=
  (part.toFloat / total.toFloat) * 100

theorem blue_yellow_percentages :
  percentage 90 responses ≈ 25.71 ∧ percentage 80 responses ≈ 22.86 := by
  sorry

end blue_yellow_percentages_l577_577237


namespace line_through_circle_center_perpendicular_to_given_line_l577_577158

theorem line_through_circle_center_perpendicular_to_given_line
  (circle : ∀ x y : ℝ, x^2 + (y - 3)^2 = 4)
  (line_passing : ∃ l : ℝ × ℝ, l = (0, 3)) 
  (line_perpendicular : ∃ m : ℝ, m = 1 ∧ is_perpendicular_via_slope (x + y + 1 = 0) m) :
  ∃ l_eq : ∀ x y : ℝ, x - y + 3 = 0 :=
sorry

end line_through_circle_center_perpendicular_to_given_line_l577_577158


namespace primes_solution_l577_577915

theorem primes_solution (p : ℕ) (n : ℕ) (h_prime : Prime p) (h_nat : 0 < n) : 
  (p^2 + n^2 = 3 * p * n + 1) ↔ (p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8) := sorry

end primes_solution_l577_577915


namespace coin_split_exists_l577_577446

noncomputable def smallest_C := (50 : ℝ) / 51

theorem coin_split_exists
  (coins : Fin 100 → ℝ)
  (hpositive : ∀ i, 0 < coins i)
  (hbounded : ∀ i, coins i ≤ 1)
  (htotal : (∑ i, coins i) = 50) :
  ∃ (stack1 stack2 : Finset (Fin 100)), 50 ∈ stack1.card ∧ 50 ∈ stack2.card ∧
  |(∑ i in stack1, coins i) - (∑ i in stack2, coins i)| ≤ smallest_C :=
sorry

end coin_split_exists_l577_577446


namespace proof_problem_l577_577689

noncomputable def p (x : ℝ) := k * (x - 5) * (x - 1)
noncomputable def q (x : ℝ) := (x - 5) * (x + 2)

theorem proof_problem (k : ℝ) (hk : ∀ x, 2 * (x - 1) = k * (x - 1)) :
  (p 3) / (q 3) = (4 / 5) :=
by
  have h1 : q 3 = (3 - 5) * (3 + 2), from sorry,
  have h2 : p 3 = k * (3 - 5) * (3 - 1), from sorry,
  sorry

end proof_problem_l577_577689


namespace sum_floor_log2_l577_577087

open Int

theorem sum_floor_log2 (S: ℕ) (hS : S = ∑ N in (finset.range 2048).map (λ x, x + 1), ⌊log (N : ℝ) / log 2⌋) : S = 45055 :=
sorry

end sum_floor_log2_l577_577087


namespace stock_yield_is_8_percent_l577_577397

def stock_yield_percentage (annual_dividend market_value : ℝ) : ℝ :=
  (annual_dividend / market_value) * 100

theorem stock_yield_is_8_percent (market_value : ℝ) (par_value : ℝ) (dividend_rate : ℝ) :
  market_value = 137.5 →
  par_value = 100 →
  dividend_rate = 0.11 →
  stock_yield_percentage (par_value * dividend_rate) market_value ≈ 8 := by
    sorry

end stock_yield_is_8_percent_l577_577397


namespace range_of_trig_function_l577_577211

theorem range_of_trig_function (x : ℝ) (h : 0 < x ∧ x ≤ π / 3) :
  set_of (λ y, y = sin x + cos x + sin x * cos x) = set.Ioc 1 (2 + 2 * real.sqrt 2) :=
by
  sorry

end range_of_trig_function_l577_577211


namespace vector_subtraction_l577_577442

-- Lean definitions for the problem conditions
def v₁ : ℝ × ℝ := (3, -5)
def v₂ : ℝ × ℝ := (-2, 6)
def s₁ : ℝ := 4
def s₂ : ℝ := 3

-- The theorem statement
theorem vector_subtraction :
  s₁ • v₁ - s₂ • v₂ = (18, -38) :=
by
  sorry

end vector_subtraction_l577_577442


namespace repeating_decimal_as_fraction_l577_577471

def repeating_decimal_37 : ℝ := 37 / 99

theorem repeating_decimal_as_fraction :
  (.37 : ℝ) = repeating_decimal_37 :=
by
  sorry

end repeating_decimal_as_fraction_l577_577471


namespace find_k_l577_577160

noncomputable def polynomial1 : Polynomial Int := sorry

theorem find_k :
  ∃ P : Polynomial Int,
  (P.eval 1 = 2013) ∧
  (P.eval 2013 = 1) ∧
  (∃ k : Int, P.eval k = k) →
  ∃ k : Int, P.eval k = k ∧ k = 1007 :=
by
  sorry

end find_k_l577_577160


namespace smallest_naive_number_max_naive_number_divisible_by_ten_l577_577930

/-- Definition of naive number -/
def is_naive_number (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  (a = d + 6) ∧ (b = c + 2)

/-- Definition of P function -/
def P (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  3 * (a + b) + c + d

/-- Definition of Q function -/
def Q (M : ℕ) : ℤ :=
  let a := M / 1000
  a - 5

namespace NaiveNumber

/-- Smallest naive number is 6200 -/
theorem smallest_naive_number : ∃ M : ℕ, is_naive_number M ∧ M = 6200 :=
  sorry

/-- Maximum naive number such that P(M)/Q(M) is divisible by 10 is 9313 -/
theorem max_naive_number_divisible_by_ten : ∃ M : ℕ, is_naive_number M ∧ (P(M) / Q(M)) % 10 = 0 ∧ M = 9313 :=
  sorry

end NaiveNumber

end smallest_naive_number_max_naive_number_divisible_by_ten_l577_577930


namespace safe_unlockable_by_five_l577_577690

def min_total_keys (num_locks : ℕ) (num_people : ℕ) (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) : ℕ :=
  num_locks * ((num_people + 1) / 2)

theorem safe_unlockable_by_five (num_locks : ℕ) (num_people : ℕ) 
  (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) :
  (∀ (P : Finset (Fin num_people)), P.card = 5 → (∀ k : Fin num_locks, ∃ p ∈ P, key_distribution k p)) →
  min_total_keys num_locks num_people key_distribution = 20 := 
by
  sorry

end safe_unlockable_by_five_l577_577690


namespace actual_avg_height_correct_l577_577683

-- Given conditions
constant num_boys : ℕ := 35
constant initial_avg_height : ℝ := 182
constant incorrect_height : ℝ := 166
constant actual_height : ℝ := 106

-- Proof goal
theorem actual_avg_height_correct :
  (initial_avg_height * num_boys - (incorrect_height - actual_height)) / num_boys = 180.29 :=
by
  sorry

end actual_avg_height_correct_l577_577683


namespace cube_volume_multiple_of_6_l577_577053

theorem cube_volume_multiple_of_6 (n : ℕ) (h : ∃ m : ℕ, n^3 = 24 * m) : ∃ k : ℕ, n = 6 * k :=
by
  sorry

end cube_volume_multiple_of_6_l577_577053


namespace factor_expression_l577_577473

variable (y : ℝ)

theorem factor_expression : 
  6*y*(y + 2) + 15*(y + 2) + 12 = 3*(2*y + 5)*(y + 2) :=
sorry

end factor_expression_l577_577473


namespace tangent_line_perpendicular_y_axis_l577_577551

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + Real.log x

def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * x - a + (1 / x)

theorem tangent_line_perpendicular_y_axis (a : ℝ) :
  (∃ x : ℝ, f_deriv a x = 0) → a > 2 :=
sorry

end tangent_line_perpendicular_y_axis_l577_577551


namespace animals_total_sleep_in_one_week_l577_577411

-- Define the conditions
def cougar_sleep_per_night := 4 -- Cougar sleeps 4 hours per night
def zebra_extra_sleep := 2 -- Zebra sleeps 2 hours more than cougar

-- Calculate the sleep duration for the zebra
def zebra_sleep_per_night := cougar_sleep_per_night + zebra_extra_sleep

-- Total sleep duration per week
def week_nights := 7

-- Total weekly sleep durations
def cougar_weekly_sleep := cougar_sleep_per_night * week_nights
def zebra_weekly_sleep := zebra_sleep_per_night * week_nights

-- Total sleep time for both animals in one week
def total_weekly_sleep := cougar_weekly_sleep + zebra_weekly_sleep

-- The target theorem
theorem animals_total_sleep_in_one_week : total_weekly_sleep = 70 := by
  sorry

end animals_total_sleep_in_one_week_l577_577411


namespace sum_of_first_2017_terms_l577_577954

noncomputable theory

variable {a : ℕ → ℝ}

-- Definitions and conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 0 = 1 ∧ ∀ n, a (n + 1) = q * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

def condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  9 * sum_of_first_n_terms a 3 = sum_of_first_n_terms a 6

-- Statement of the problem
theorem sum_of_first_2017_terms (q : ℝ) (h_geometric : is_geometric_sequence a q) (h_condition : condition a q) :
  ∑ i in Finset.range 2017, a i * a (i + 1) = (2 / 3) * (4 ^ 2017 - 1) :=
sorry

end sum_of_first_2017_terms_l577_577954


namespace hyperbola_problem_statement_l577_577159

noncomputable theory

-- Define the hyperbola with given asymptotes and focal length
def hyperbola_eq₁ (a b : ℝ) : Prop :=
  (a / b = (Real.sqrt 3) / 3) ∧ ((2:ℝ)^2 = a^2 + b^2)

def hyperbola_standard_eq :=
  ∃ (a b : ℝ), a = 1 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, y^2 - x^2 / 3 = 1)

-- Define the hyperbola's equation and the properties of L passing through A
def line_through_midpoint (M N A : Point) (L : Line) : Prop :=
  A.x = 1 ∧ A.y = 1/2 ∧
  L.contains A ∧ L.contains M ∧ L.contains N ∧
  midpoint M N = A

def hyperbola_eq₂ (x₁ y₁ x₂ y₂ : ℝ) :=
  y₁^2 - x₁^2 / 3 = 1 ∧ y₁ - y₂ - 2 * (x₁ - x₂) / 3 = 0

-- Theorem stating the two proofs required
theorem hyperbola_problem_statement :
  (∀ a b : ℝ, hyperbola_eq₁ a b → hyperbola_standard_eq) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, hyperbola_eq₂ x₁ y₁ x₂ y₂ → 
    ∃ L : Line, L.equation = 4*x₁ - 6*y₁ - 1 = 0) :=
by
  sorry

end hyperbola_problem_statement_l577_577159


namespace evaluate_M_l577_577469

def a : ℝ := (√(√7 + 3) + √(√7 - 3)) / √(√7 + 2)
def b : ℝ := √(4 - 2√3)
def M : ℝ := a + b

theorem evaluate_M : M = 7 / 4 := 
by {
  sorry
}

end evaluate_M_l577_577469


namespace tangent_line_at_pi_over_4_l577_577975

noncomputable def f (x : ℝ) := Real.cos (2 * x)

theorem tangent_line_at_pi_over_4 :
  let x₀ := Real.pi / 4
  let y₀ := f x₀
  fderiv ℝ f x₀ = (-2 : ℝ) • (1 : ℝ) ∧ y₀ = 0 → 
  ∀ x, f x₀ + (fderiv ℝ f x₀) * (x - x₀) = -2 * (x - x₀) := sorry

end tangent_line_at_pi_over_4_l577_577975


namespace fruit_seller_original_apples_l577_577021

variable (x : ℝ)

theorem fruit_seller_original_apples (h : 0.60 * x = 420) : x = 700 := by
  sorry

end fruit_seller_original_apples_l577_577021


namespace pyramid_x_value_l577_577700

theorem pyramid_x_value (x y : ℝ) 
  (h1 : 150 = 10 * x)
  (h2 : 225 = x * 15)
  (h3 : 1800 = 150 * y * 225) :
  x = 15 :=
sorry

end pyramid_x_value_l577_577700


namespace value_of_question_l577_577393

noncomputable def value_of_approx : ℝ := 0.2127541038062284

theorem value_of_question :
  ((0.76^3 - 0.1^3) / (0.76^2) + value_of_approx + 0.1^2) = 0.66 :=
by
  sorry

end value_of_question_l577_577393


namespace p_bound_roots_l577_577149

-- Define the conditions
def equation (x p : ℝ) : Prop := sqrt(x^2 - p) + 2 * sqrt(x^2 - 1) = x

-- Prove the range of p for which the equation has real roots
theorem p_bound (p : ℝ) : (∃ x : ℝ, equation x p) → 0 ≤ p ∧ p ≤ 4 / 3 :=
by sorry

-- Find the roots of the equation under the constraints on p
theorem roots (x p : ℝ) : (equation x p ∧ 0 ≤ p ∧ p ≤ 4 / 3) → x = 1 :=
by sorry

end p_bound_roots_l577_577149


namespace smallest_12_digit_number_divisible_by_36_with_all_digits_l577_577822

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  n ≥ 10^11 ∧ n < 10^12

noncomputable def contains_all_digits (n : ℕ) : Prop :=
  (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → d ∈ n.digits 10)

noncomputable def divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

theorem smallest_12_digit_number_divisible_by_36_with_all_digits : 
  ∃ n : ℕ, is_12_digit_number n ∧ contains_all_digits n ∧ divisible_by_36 n ∧ 
  (∀ m : ℕ, is_12_digit_number m ∧ contains_all_digits m ∧ divisible_by_36 m → n ≤ m) :=
sorry

end smallest_12_digit_number_divisible_by_36_with_all_digits_l577_577822


namespace PQ_perp_MN_l577_577406

theorem PQ_perp_MN
  (A B C D O K L M N P Q : Point)
  (hCircle : InscribedCircle ABCD O)
  (hA : K ∈ Line OA)
  (hB : L ∈ Line OB)
  (hC : M ∈ Line OC)
  (hD : N ∈ Line OD)
  (hP : P = Intersection (Line KL) (Line MN))
  (hQ : Q = Midpoint K L) :
  Perpendicular (Line PQ) (Line MN) :=
  sorry

end PQ_perp_MN_l577_577406


namespace water_bottles_needed_l577_577465

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l577_577465


namespace download_time_l577_577101

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l577_577101


namespace original_distance_between_Stacy_and_Heather_l577_577322

theorem original_distance_between_Stacy_and_Heather
  (H_speed : ℝ := 5)  -- Heather's speed in miles per hour
  (S_speed : ℝ := 6)  -- Stacy's speed in miles per hour
  (delay : ℝ := 0.4)  -- Heather's start delay in hours
  (H_distance : ℝ := 1.1818181818181817)  -- Distance Heather walked when they meet
  : H_speed * (H_distance / H_speed) + S_speed * ((H_distance / H_speed) + delay) = 5 := by
  sorry

end original_distance_between_Stacy_and_Heather_l577_577322


namespace janet_initial_stickers_l577_577583

variable (x : ℕ)

theorem janet_initial_stickers (h : x + 53 = 56) : x = 3 := by
  sorry

end janet_initial_stickers_l577_577583


namespace sphere_radius_twice_cone_volume_l577_577845

theorem sphere_radius_twice_cone_volume :
  ∀ (r_cone h_cone : ℝ) (r_sphere : ℝ), 
    r_cone = 2 → h_cone = 8 → 2 * (1 / 3 * Real.pi * r_cone^2 * h_cone) = (4/3 * Real.pi * r_sphere^3) → 
    r_sphere = 2^(4/3) :=
by
  intros r_cone h_cone r_sphere h_r_cone h_h_cone h_volume_equiv
  sorry

end sphere_radius_twice_cone_volume_l577_577845


namespace percentage_of_carnations_is_44_percent_l577_577839

noncomputable def total_flowers : ℕ := sorry
def pink_percentage : ℚ := 2 / 5
def red_percentage : ℚ := 2 / 5
def yellow_percentage : ℚ := 1 / 5
def pink_roses_fraction : ℚ := 2 / 5
def red_carnations_fraction : ℚ := 1 / 2

theorem percentage_of_carnations_is_44_percent
  (F : ℕ)
  (h_pink : pink_percentage * F = 2 / 5 * F)
  (h_red : red_percentage * F = 2 / 5 * F)
  (h_yellow : yellow_percentage * F = 1 / 5 * F)
  (h_pink_roses : pink_roses_fraction * (pink_percentage * F) = 2 / 25 * F)
  (h_red_carnations : red_carnations_fraction * (red_percentage * F) = 1 / 5 * F) :
  ((6 / 25 * F + 5 / 25 * F) / F) * 100 = 44 := sorry

end percentage_of_carnations_is_44_percent_l577_577839


namespace find_k_l577_577288

theorem find_k 
  (k b : ℝ)
  (h1 : ∃ A B C : ℝ × ℝ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    (A = (a, f(a)) ∧ B = (0, 0) ∧ C = (-a, -f(a))) ∧
    f(x) = k * x ∧ f(x) = x^3 - x ∧
    abs((fst B) - (fst A)) = 2 ∧ abs((fst C) - (fst B)) = 2)
  (h2 : b = 0) :
  k = 1 :=
by
  sorry

end find_k_l577_577288


namespace find_two_digit_number_l577_577045

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l577_577045


namespace smallest_a_for_sin_eq_l577_577267

noncomputable theory

open Real

theorem smallest_a_for_sin_eq (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, sin (a * x + b + π/4) = sin (15 * x + π/4)) : a = 15 :=
by
  sorry

end smallest_a_for_sin_eq_l577_577267


namespace max_u_min_fraction_l577_577513

noncomputable def maxValueOfU (x y : ℝ) : ℝ :=
  if (x > 0 ∧ y > 0 ∧ 2*x + 5*y = 20) then
    log x / log 10 + log y / log 10
  else
    0

noncomputable def minValueOfFraction (x y : ℝ) : ℝ :=
  if (x > 0 ∧ y > 0 ∧ 2*x + 5*y = 20) then
    1/x + 1/y
  else
    0

theorem max_u {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2*x + 5*y = 20) : 
  maxValueOfU x y <= 1 :=
by
  sorry

theorem min_fraction {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2*x + 5*y = 20) : 
  minValueOfFraction x y >= (7 + 2*sqrt 10) / 20 :=
by
  sorry

end max_u_min_fraction_l577_577513


namespace min_sum_ab_72_l577_577618

theorem min_sum_ab_72 (a b : ℤ) (h : a * b = 72) : a + b ≥ -17 := sorry

end min_sum_ab_72_l577_577618


namespace oil_amount_to_add_l577_577364

variable (a b : ℝ)
variable (h1 : a = 0.16666666666666666)
variable (h2 : b = 0.8333333333333334)

theorem oil_amount_to_add (a b : ℝ) (h1 : a = 0.16666666666666666) (h2 : b = 0.8333333333333334) : 
  b - a = 0.6666666666666667 := by
  rw [h1, h2]
  norm_num
  sorry

end oil_amount_to_add_l577_577364


namespace complement_of_A_in_U_l577_577527

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}
def complement : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l577_577527


namespace ellipse_C2_standard_equation_S1_div_S2_range_l577_577164

open Real

def C1 (x y : ℝ) : Prop := (x^2 / 8) + (y^2 / 4) = 1

def e1 : ℝ := sqrt 2 / 2

def C2 (x y : ℝ) : Prop := (y^2 / 4) + (x^2 / 2) = 1

def A : Point := ⟨0, 2⟩

def S1 (M N : Point) : ℝ := sorry  -- Function to calculate area of triangle AMN

def S2 (P Q : Point) : ℝ := sorry  -- Function to calculate area of triangle APQ

-- Prove that C2 correctly describes the ellipse based on given conditions
theorem ellipse_C2_standard_equation : 
  (∃ x y : ℝ, C2 x y ∧ (x = 1) ∧ (y = sqrt 2)) ∧
  (∃ e : ℝ, e = e1) ∧ 
  (∀ x y : ℝ, C2 x y → (y^2 / 4 + x^2 / 2 = 1)) :=
sorry

-- Find the range of S1 / S2
theorem S1_div_S2_range (P Q M N : Point) (h1 : C2 P.x P.y) (h2: C2 Q.x Q.y) (h3 : C1 M.x M.y) (h4 : C1 N.x N.y) :
  let ratio := S1 M N / S2 P Q in
  ratio ∈ Ico (64 / 25) 4 :=
sorry

end ellipse_C2_standard_equation_S1_div_S2_range_l577_577164


namespace sum_of_possible_values_for_T_area_l577_577846

-- Definitions based on the given conditions
def larger_square_side (s : ℝ) : Prop :=
  s ≥ 3.5

def smaller_square_area := 4 -- Area of the smaller square (2*2)

def rectangle_area := 1.5 * 2 -- Area of the rectangle (1.5*2)

def right_triangle_area := 0.5 * 1 * 1 -- Area of the right triangle (1/2 * 1 * 1)

-- The total area of the larger square
def total_area_larger_square (s : ℝ) : ℝ :=
  s * s

-- The area occupied by the smaller figures
def occupied_area := smaller_square_area + rectangle_area + right_triangle_area

-- The total area left for rectangle T
def area_left_for_T (s : ℝ) : ℝ :=
  total_area_larger_square s - occupied_area

-- The main theorem to be proved
theorem sum_of_possible_values_for_T_area (s : ℝ) (h₁ : larger_square_side s) :
  area_left_for_T s = 4.75 :=
by
  sorry

end sum_of_possible_values_for_T_area_l577_577846


namespace odd_function_neg_value_l577_577519

noncomputable def f (x : ℝ) : ℝ := if x > 0 then x^3 - 2 * x^2 else -if -x > 0 then f (-x) else 0

theorem odd_function_neg_value (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x, 0 < x → f x = x^3 - 2 * x^2) :
  ∀ x, x < 0 → f x = x^3 + 2 * x^2 :=
begin
  intros x hx,
  have h_neg : f (-x) = (-x)^3 - 2 * (-x)^2, from h_pos (-x) (neg_pos_of_neg hx),
  rw [←h_odd x, neg_neg x, h_neg],
  simp,
end

end odd_function_neg_value_l577_577519


namespace proof_problem_statement_l577_577660

noncomputable def problem_statement : Prop :=
  ∃ (A B : ℝ × ℝ),
    dist A B = 8 ∧
    (∃ (rₐ r_b : ℝ), rₐ = 3 ∧ r_b = 4 ∧ Circle A rₐ ∧ Circle B r_b) ∧
    (∃ (L : set (ℝ × ℝ)),
      (line_contains_point_dist L A 3) ∧
      (line_contains_point_dist L B 4) ∧
      card L = 2)

theorem proof_problem_statement : problem_statement :=
sorry

end proof_problem_statement_l577_577660


namespace two_digit_number_is_9_l577_577035

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l577_577035


namespace simplify_log_expression_l577_577319

theorem simplify_log_expression : (log 2)^2 + log 2 * log 5 + log 5 = 1 :=
by
  sorry

end simplify_log_expression_l577_577319


namespace line_equation_l2_l577_577964

theorem line_equation_l2 :
  ∀ (θ : ℝ), let l1 := ∀ x y : ℝ, x - 2 * y - 2 = 0
    → ∀ (l2 : ℝ), l2 = 2 * θ
    → ∃ (a b c : ℝ), a * x + b * y + c = 0
      → 4 * x - 3 * y + 9 = 0 := sorry

end line_equation_l2_l577_577964


namespace correct_parallel_statement_l577_577429

-- Define the necessary properties and conditions 
def parallel_projection_of_parallel_lines_coincides : Prop :=
  ∀ (L1 L2 : ℝ → ℝ), ∀ (P : ℝ → ℝ → ℝ),
    (L1 ∥ L2) → (P (L1 x) = L1 x ∧ P (L2 x) = L2 x) → (L1 ∥ L2)

def planes_parallel_to_same_line_are_parallel (L : ℝ → ℝ) (P1 P2 : ℝ → ℝ → ℝ) : Prop :=
  (P1 ∥ L ∧ P2 ∥ L) → (P1 ∥ P2)

def planes_perpendicular_to_same_plane_are_parallel (P0 P1 P2 : ℝ → ℝ → ℝ) : Prop :=
  (P1 ⟂ P0 ∧ P2 ⟂ P0) → (P1 ∥ P2)

def lines_perpendicular_to_same_plane_are_parallel (L1 L2 : ℝ → ℝ) (P : ℝ → ℝ → ℝ) : Prop :=
  (L1 ⟂ P ∧ L2 ⟂ P) → (L1 ∥ L2)

-- Problem: The statement we want to prove is that two lines perpendicular to the same plane are parallel
theorem correct_parallel_statement :
  lines_perpendicular_to_same_plane_are_parallel :=
sorry

end correct_parallel_statement_l577_577429


namespace sum_a_n_l577_577189

noncomputable def f (n : ℕ) : ℝ := n^2 * Real.cos (n * Real.pi)

noncomputable def a (n : ℕ) : ℝ := f n + f (n + 1)

-- We want to prove that the sum of the first 20 terms a₁ + a₂ + ... + a₂₀ is -20
theorem sum_a_n : (∑ n in finset.range 20, a (n + 1)) = -20 := sorry

end sum_a_n_l577_577189


namespace polynomial_value_l577_577228

theorem polynomial_value (y : ℝ) (h : 4 * y^2 - 2 * y + 5 = 7) : 2 * y^2 - y + 1 = 2 :=
by
  sorry

end polynomial_value_l577_577228


namespace opposite_numbers_add_l577_577537

theorem opposite_numbers_add : ∀ {a b : ℤ}, a + b = 0 → a + b + 3 = 3 :=
by
  intros
  sorry

end opposite_numbers_add_l577_577537


namespace part_one_part_two_l577_577625

-- Define the parabola and its properties
variables {p : ℝ} (h₀ : p > 0)
def parabola : set (ℝ × ℝ) := {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line l
def line_l (x y : ℝ) := y = (ℝ.sqrt 2 / 2) * (x + p / 2)

-- Define points F and Q
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def directrix_intersection_x (p : ℝ) : ℝ := -p / 2
def point_Q (p : ℝ) : ℝ × ℝ := (directrix_intersection_x p, 0)

-- Define points A and B
def point_A_B (x₁ x₂ y₁ y₂ : ℝ) :=
  x₁ + x₂ = 3 * p ∧ x₁ * x₂ = p^2 / 4 ∧
  y₁ = ℝ.sqrt 2 / 2 * (x₁ + p / 2) ∧ y₂ = ℝ.sqrt 2 / 2 * (x₂ + p / 2)

-- Prove that -> FA ⋅ -> FB = 0
theorem part_one (x₁ x₂ y₁ y₂ : ℝ) (h : point_A_B p x₁ x₂ y₁ y₂) :
  (x₁ - p / 2, y₁) • (x₂ - p / 2, y₂) = 0 := sorry

-- Define slopes k₁ and k₂
def k₁ (x₁ y₁ : ℝ) := y₁ / (x₁ - p / 2)
def k₂ (x₂ y₂ : ℝ) := y₂ / (x₂ - p / 2)

-- Prove that k₁ + k₂ = 0
theorem part_two (x₁ x₂ y₁ y₂ : ℝ) (h : point_A_B p x₁ x₂ y₁ y₂) :
  k₁ p x₁ y₁ + k₂ p x₂ y₂ = 0 := sorry

end part_one_part_two_l577_577625


namespace integer_a_conditions_l577_577121

theorem integer_a_conditions (a : ℤ) :
  (∃ (x y : ℕ), x ≠ y ∧ (a * x * y + 1) ∣ (a * x^2 + 1) ^ 2) → a ≥ -1 :=
sorry

end integer_a_conditions_l577_577121


namespace cannot_determine_right_triangle_l577_577378

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def two_angles_complementary (α β : ℝ) : Prop :=
  α + β = 90

def exterior_angle_is_right (γ : ℝ) : Prop :=
  γ = 90

theorem cannot_determine_right_triangle :
  ¬ (∃ (a b c : ℝ), a = 1 ∧ b = 1 ∧ c = 2 ∧ is_right_triangle a b c) :=
by sorry

end cannot_determine_right_triangle_l577_577378


namespace f_divisible_by_8_l577_577157

def f (n : ℕ) : ℕ := 5 * n + 2 * (-1) ^ n + 1

theorem f_divisible_by_8 (n : ℕ) (h : n = 1 ∨ n = 2 ∨ n = 3) : 8 ∣ f n :=
by
  cases h with
  | inl h1 =>
    rw [h1]
    exact dvd.intro 1 rfl
  | inr h23 =>
    cases h23 with
    | inl h2 =>
      rw [h2]
      exact dvd.intro 4 rfl
    | inr h3 =>
      rw [h3]
      exact dvd.intro 18 rfl

end f_divisible_by_8_l577_577157


namespace min_max_sum_eq_one_l577_577943

theorem min_max_sum_eq_one 
  (x : ℕ → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_eq_one : (x 1 + x 2 + x 3 + x 4 + x 5) = 1) :
  (min (max (x 1 + x 2) (max (x 2 + x 3) (max (x 3 + x 4) (x 4 + x 5)))) = (1 / 3)) :=
by
  sorry

end min_max_sum_eq_one_l577_577943


namespace two_digit_number_is_9_l577_577036

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l577_577036


namespace profit_and_combined_cost_l577_577856

variable (SA_SP : ℝ) (SA_profit : ℝ)
variable (SB_SP : ℝ) (SB_profit : ℝ)
variable (SC_SP : ℝ) (SC_loss : ℝ)
variable (SD_SP : ℝ) (SD_loss : ℝ)
variable (SE_SP : ℝ) (SE_profit : ℝ)

def calculateCP (SP : ℝ) (percentage : ℝ) (isLoss : Bool) : ℝ :=
  if isLoss then SP / (1 - percentage / 100) else SP / (1 + percentage / 100)

def CP_A := calculateCP SA_SP SA_profit false
def CP_B := calculateCP SB_SP SB_profit false
def CP_C := calculateCP SC_SP SC_loss true
def CP_D := calculateCP SD_SP SD_loss true
def CP_E := calculateCP SE_SP SE_profit false

def combined_CP := CP_A + CP_B + CP_C + CP_D + CP_E

def combined_SP := SA_SP + SB_SP + SC_SP + SD_SP + SE_SP

def total_profit := combined_SP - combined_CP

def total_profit_percentage := (total_profit / combined_CP) * 100

theorem profit_and_combined_cost :
  SA_SP = 120 → SA_profit = 20 →
  SB_SP = 100 → SB_profit = 25 →
  SC_SP = 90 → SC_loss = 10 →
  SD_SP = 85 → SD_loss = 15 →
  SE_SP = 130 → SE_profit = 30 →
  combined_CP = 480 ∧ total_profit_percentage = 9.375 := by
  sorry

end profit_and_combined_cost_l577_577856


namespace circle_equation_diameter_circle_passes_through_spec_point_l577_577241

-- Definition of the quadratic function f(x) = x^2 + 2x - 1
def f (x : ℝ) := x^2 + 2 * x - 1

-- Definition of the circle C that passes through the intersection points of f and coordinate axes
def isCircleC (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 1 = 0

-- Coordinate of point A and B found from solving f(x) = 0
def pointA := (-1 - Real.sqrt 2, 0)
def pointB := (Real.sqrt 2 - 1, 0)

-- Prove that the equation of circle C is x^2 + y^2 + 2x - 1 = 0
theorem circle_equation :
  ∀ (x y : ℝ), isCircleC x y ↔ x^2 + y^2 + 2 * x - 1 = 0 := by
  sorry

-- Definition of points M and N with respect to a point P on the circle C
def isPointOnC (P : ℝ × ℝ) := x^2 + y^2 + 2 * P.1 - 1 = 0

-- Definition of the circle that passes through a certain point
def specificCircle (x y : ℝ) : Prop := (x - 2)^2 + y^2 - 7 = 0

-- Prove that the circle with diameter MN passes through a specific point on segment AB
theorem diameter_circle_passes_through_spec_point:
  ∀ P : ℝ × ℝ, isPointOnC P → specificCircle (2 - Real.sqrt 7) 0 :=
  by
  sorry

end circle_equation_diameter_circle_passes_through_spec_point_l577_577241


namespace simplify_expr_perimeter_triangle_l577_577957

-- Part (1)
theorem simplify_expr (a b c : ℤ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (|a - b + c| + |c - a - b| - |a + b|  = a - b) := sorry

-- Part (2)
theorem perimeter_triangle (a b c : ℤ) (h_eq : a^2 + b^2 - 2a - 8b + 17 = 0) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a + b + c = 9 := sorry

end simplify_expr_perimeter_triangle_l577_577957


namespace find_k_l577_577197

-- Define the lines as given in the problem
def line1 (k : ℝ) (x y : ℝ) : Prop := k * x + (1 - k) * y - 3 = 0
def line2 (k : ℝ) (x y : ℝ) : Prop := (k - 1) * x + (2 * k + 3) * y - 2 = 0

-- Define the condition for perpendicular lines
def perpendicular (k : ℝ) : Prop :=
  let slope1 := -k / (1 - k)
  let slope2 := -(k - 1) / (2 * k + 3)
  slope1 * slope2 = -1

-- Problem statement: Prove that the lines are perpendicular implies k == 1 or k == -3
theorem find_k (k : ℝ) : perpendicular k → (k = 1 ∨ k = -3) :=
sorry

end find_k_l577_577197


namespace hyperbola_eccentricity_l577_577982

variable (a b : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b)

def hyperbola_eq (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

def left_focus := (-√(a^2 + b^2), 0)

def perp_foot (c : ℝ) := (-a^2 / c, ab / c)

def vector_condition := ∃ (x y c : ℝ), (x + c, y) = 3 * (-a^2 / c + c, ab / c)

theorem hyperbola_eccentricity (x y c : ℝ)
  (h1 : hyperbola_eq x y)
  (h2 : vector_condition)
  : ∃ e : ℝ, e = √13 / 2 :=
sorry

end hyperbola_eccentricity_l577_577982


namespace limit_of_seq_l577_577620

noncomputable def seq (x₁ a : ℝ) : ℕ → ℝ
| 0     := x₁
| (n+1) := seq n + a / seq n

theorem limit_of_seq (x₁ a : ℝ) (h₀ : 0 < x₁) (h₁ : 0 < a) :
  filter.tendsto (seq x₁ a) filter.at_top (nhds (Real.sqrt a)) :=
sorry

end limit_of_seq_l577_577620


namespace Grant_score_is_100_l577_577200

/-- Definition of scores --/
def Hunter_score : ℕ := 45

def John_score (H : ℕ) : ℕ := 2 * H

def Grant_score (J : ℕ) : ℕ := J + 10

/-- Theorem to prove Grant's score --/
theorem Grant_score_is_100 : Grant_score (John_score Hunter_score) = 100 := 
  sorry

end Grant_score_is_100_l577_577200


namespace problem_solution_l577_577250

-- Definitions for the conditions
def parametric_equation_line (t : ℝ) : ℝ × ℝ :=
  (3 - (√3) / 2 * t, √3 - 1 / 2 * t)

def polar_equation_curve (θ : ℝ) : ℝ :=
  2 * sin (θ + π / 6)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- The problem as a Lean statement
theorem problem_solution :
  (∀ t : ℝ, parametric_equation_line t = (x, y) → x - √3 * y = 0) ∧
  (∀ θ : ℝ, let ρ := polar_equation_curve θ in
    let (x, y) := polar_to_cartesian ρ θ in
    x^2 + y^2 - x - √3 * y = 0) ∧
  ∀ t1 t2 : ℝ, let (x1, y1) := parametric_equation_line t1 in
    let (x2, y2) := parametric_equation_line t2 in
    x1^2 + y1^2 - x1 - √3 * y1 = 0 ∧ x2^2 + y2^2 - x2 - √3 * y2 = 0 →
    (1 / (dist (3, √3) (x1, y1)) + 1 / (dist (3, √3) (x2, y2))) = √3 / 2 :=
sorry

end problem_solution_l577_577250


namespace fraction_of_capital_contributed_by_a_l577_577812

theorem fraction_of_capital_contributed_by_a
  (A B : ℚ)
  (h1 : 15 * A / (10 * B) = 1 / 2)
  (h2 : A + B = 1) :
  A = 1 / 4 :=
sorry

end fraction_of_capital_contributed_by_a_l577_577812


namespace sum_xi_bound_l577_577602

noncomputable def x_seq : ℕ → ℝ
| 0       := x_1
| (n + 1) := 1 + x_seq n - 1 / 2 * (x_seq n)^2

theorem sum_xi_bound (x_1 : ℝ) (m : ℕ) (h1 : 1 < x_1) (h2 : x_1 < 2) (h3 : 3 ≤ m) :
  (∑ i in finset.range (m + 1) \ finset.range 3, |x_seq x_1 i - real.sqrt 2|) < 1 / 4 :=
sorry

end sum_xi_bound_l577_577602


namespace find_45th_digit_l577_577219

theorem find_45th_digit :
  let sequence := (List.range 21).reverse.map (λ n, if n < 10 then (n + 10).toString else (n - 10).toString) in
  let concatenated_sequence := String.join sequence in
  concatenated_sequence.get! 44 = '7' :=
by
  sorry

end find_45th_digit_l577_577219


namespace standard_equation_of_ellipse_midpoint_of_chord_l577_577949

variables (a b c : ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (A B : ℝ × ℝ)

axiom conditions :
  a > b ∧ b > 0 ∧
  (c / a = (Real.sqrt 6) / 3) ∧
  a = Real.sqrt 3 ∧
  a^2 = b^2 + c^2 ∧
  (A = (-1, 0)) ∧ (B = (x2, y2)) ∧
  A ≠ B ∧
  (∃ l : ℝ -> ℝ, l (-1) = 0 ∧ ∀ x, l x = x + 1) ∧
  (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = -3 / 2)

theorem standard_equation_of_ellipse :
  ∃ (e : ℝ), e = 1 ∧ (x1 / 3) + y1 = 1 := sorry

theorem midpoint_of_chord :
  ∃ (m : ℝ × ℝ), m = (-(3 / 4), 1 / 4) := sorry

end standard_equation_of_ellipse_midpoint_of_chord_l577_577949


namespace floor_neg_seven_four_cubed_l577_577905

theorem floor_neg_seven_four_cubed : 
  let a := (-7 / 4) ^ 3 in ⌊a⌋ = -6 :=
by
  let a := (-7 / 4 : ℝ) ^ 3
  show floor a = -6
  sorry

end floor_neg_seven_four_cubed_l577_577905


namespace emily_and_eli_probability_l577_577118

noncomputable def probability_same_number : ℚ :=
  let count_multiples (n k : ℕ) := (k - 1) / n
  let emily_count := count_multiples 20 250
  let eli_count := count_multiples 30 250
  let common_lcm := Nat.lcm 20 30
  let common_count := count_multiples common_lcm 250
  common_count / (emily_count * eli_count : ℚ)

theorem emily_and_eli_probability :
  let probability := probability_same_number
  probability = 1 / 24 :=
by
  sorry

end emily_and_eli_probability_l577_577118


namespace total_cookies_correct_l577_577065

noncomputable def cookies_monday : ℕ := 5
def cookies_tuesday := 2 * cookies_monday
def cookies_wednesday := cookies_tuesday + (40 * cookies_tuesday / 100)
def total_cookies := cookies_monday + cookies_tuesday + cookies_wednesday

theorem total_cookies_correct : total_cookies = 29 := by
  sorry

end total_cookies_correct_l577_577065


namespace principal_value_of_argument_l577_577496

noncomputable def theta := Real.arctan (5/12)

def z := (Complex.cos (2 * theta) + Complex.sin (2 * theta) * Complex.I) / (Complex.mk 239 1)

theorem principal_value_of_argument :
  Complex.arg z = Real.pi / 4 :=
by
  sorry

end principal_value_of_argument_l577_577496


namespace arithmetic_sequence_properties_l577_577719

variable (a_n : ℕ → ℚ)
variable (a_3 a_11 : ℚ)

notation "a₃" => a_3
notation "a₁₁" => a_11

theorem arithmetic_sequence_properties :
  a₃ = a_n 3 → a₁₁ = a_n 11 → 
  (∃ (a₁ d : ℚ), a_n n = a₁ + (n - 1) * d ∧ a₁ = 0 ∧ d = 3/2) := sorry

end arithmetic_sequence_properties_l577_577719


namespace imaginary_unit_sum_l577_577484

theorem imaginary_unit_sum (i : ℂ) (H : i^4 = 1) : i^1234 + i^1235 + i^1236 + i^1237 = 0 :=
by
  sorry

end imaginary_unit_sum_l577_577484


namespace probability_eq_3_div_10_l577_577511

noncomputable def a : ℝ := ∫ x in 0..Real.pi, Real.sin x

theorem probability_eq_3_div_10 :
  let P : ℝ := 3 / 10 in
  P = 3 / 10 :=
by
  sorry

end probability_eq_3_div_10_l577_577511


namespace sequence_integer_count_l577_577448

theorem sequence_integer_count (n : ℕ) (h : n = 9720) : 
  (Nat.floor (log 2 (n : ℝ))) + 1 = 4 :=
by
  sorry

end sequence_integer_count_l577_577448


namespace intersection_of_segments_l577_577670

variables (A B C D P Q K L M N : Point)
variables (AB CD AQ BQ CP DP KL MN PQ : Segment)
variables (midpoint : Segment → Point)

-- Conditions
axiom ab_nonparallel_cd : ¬ parallel AB CD
axiom ab_cd_not_intersect : ¬ intersect AB CD
axiom p_on_ab : lies_on P AB
axiom q_on_cd : lies_on Q CD
axiom k_midpoint : midpoint AQ = K
axiom l_midpoint : midpoint BQ = L
axiom m_midpoint : midpoint CP = M
axiom n_midpoint : midpoint DP = N

-- Question: Prove that segments KL, MN, and PQ intersect at a single point.
theorem intersection_of_segments :
  ∃ M_pq : Point, lies_on M_pq PQ ∧ 
                lies_on M_pq KL ∧ 
                lies_on M_pq MN :=
sorry

end intersection_of_segments_l577_577670


namespace perpendicular_bisector_eq_l577_577195

noncomputable def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + 4 * p.2 = 0 }
noncomputable def circle2 := { p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 = 0 }

theorem perpendicular_bisector_eq :
  ∃ (A B : ℝ × ℝ), A ∈ circle1 ∧ A ∈ circle2 ∧ B ∈ circle1 ∧ B ∈ circle2 ∧
  (∀ (x y : ℝ), (2 * x - y - 4 = 0 ↔ ((x, y) = midpoint A B))) :=
sorry

end perpendicular_bisector_eq_l577_577195


namespace prime_solutions_l577_577916

theorem prime_solutions (p : ℕ) (n : ℕ) (hp : p.prime) :
  p^2 + n^2 = 3 * p * n + 1 ↔ (p, n) = (3, 1) ∨ (p, n) = (3, 8) :=
by sorry

end prime_solutions_l577_577916


namespace exp_product_correct_l577_577798

def exp_1 := (2 : ℕ) ^ 4
def exp_2 := (3 : ℕ) ^ 2
def exp_3 := (5 : ℕ) ^ 2
def exp_4 := (7 : ℕ)
def exp_5 := (11 : ℕ)
def final_value := exp_1 * exp_2 * exp_3 * exp_4 * exp_5

theorem exp_product_correct : final_value = 277200 := by
  sorry

end exp_product_correct_l577_577798


namespace max_sum_squares_in_circle_is_equilateral_l577_577862

noncomputable def maximize_sum_squares_of_lengths (r : ℝ) : Triangle :=
  sorry

theorem max_sum_squares_in_circle_is_equilateral (r : ℝ) :
  maximize_sum_squares_of_lengths r = equilateral_triangle r :=
sorry

end max_sum_squares_in_circle_is_equilateral_l577_577862


namespace exist_distinct_indices_l577_577613

theorem exist_distinct_indices (n : ℕ) (h1 : n > 3)
  (a : Fin n.succ → ℕ) 
  (h2 : StrictMono a) 
  (h3 : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n.succ), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ 
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
    k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ 
    a i + a j = a k + a l ∧ 
    a k + a l = a m := 
sorry

end exist_distinct_indices_l577_577613


namespace solve_quadratic_eq_l577_577320

theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - (x^2 - 2 * x + 1) = 0 ↔ x = 1 / 3 ∨ x = -1 := by
  sorry

end solve_quadratic_eq_l577_577320


namespace profit_percentage_calculation_l577_577349

theorem profit_percentage_calculation (sale_price_incl_tax : ℝ) (sales_tax_rate : ℝ) (cost_price : ℝ) 
  (h1 : sale_price_incl_tax = 616) (h2 : sales_tax_rate = 0.10) (h3 : cost_price = 540.35) : 
  (sale_price_incl_tax / (1 + sales_tax_rate) - cost_price) / cost_price * 100 ≈ 3.64 := 
by
  -- Adding proof placeholder
  sorry

end profit_percentage_calculation_l577_577349


namespace chelsea_victory_l577_577232

def chelsea_leading_points : ℕ := 60
def total_shots : ℕ := 120
def points_per_shot (score: ℕ) : Prop := score ∈ {0, 3, 7, 10}
def chelsea_min_score_per_shot : ℕ := 3
def opponent_max_score_per_shot : ℕ := 10

theorem chelsea_victory (k n : ℕ) 
  (h₀ : 60 ≤ total_shots)
  (h₁ : points_per_shot 10)
  (h₂ : points_per_shot 7)
  (h₃ : points_per_shot 3)
  (h₄ : points_per_shot 0)
  (h₅ : ∀ m, ∀ s ≤ m, s ≥ chelsea_min_score_per_shot)
  (h₆ : k ≥ chelsea_leading_points)
  (h₇ : (k + 7 * n + 180 > k + (opponent_max_score_per_shot * 60))
  (hk : ∀ n ≥ 52, k + 7 * n + 180 > k + 540) :
  n ≥ 52 :=
begin
  sorry
end

end chelsea_victory_l577_577232


namespace n_gon_has_diagonal_inside_min_diagonals_in_n_gon_l577_577814

-- Given conditions for a polygon.
variables {n : ℕ}

-- Define the problem for part (a)
theorem n_gon_has_diagonal_inside (h : n ≥ 4) : 
  ∃ d, d.is_diagonal ∧ d.lies_entirely_inside :=
sorry

-- Define the problem for part (b)
theorem min_diagonals_in_n_gon (h : n ≥ 4) :
  ∃ d, d.length = n - 3 ∧ d.lies_entirely_inside :=
sorry

end n_gon_has_diagonal_inside_min_diagonals_in_n_gon_l577_577814


namespace sum_sequence_equal_square_l577_577296

theorem sum_sequence_equal_square (n : ℕ) (h : 0 < n) : 
  (finset.range (n + 1)).sum + (finset.range n).sum = n^2 := 
begin
  sorry
end

end sum_sequence_equal_square_l577_577296


namespace pants_cut_amount_l577_577858

theorem pants_cut_amount :
  ∀ (pants_cut skirt_cut : ℝ), skirt_cut = 0.75 → skirt_cut = pants_cut + 0.25 → pants_cut = 0.50 :=
by
  intros pants_cut skirt_cut h₀ h₁
  rw [h₁, add_comm] at h₀
  linarith


end pants_cut_amount_l577_577858


namespace problem_l577_577806

def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

def CannotFormRightTriangle (lst : List ℝ) : Prop :=
  ¬isRightTriangle lst.head! lst.tail.head! lst.tail.tail.head!

theorem problem :
  (¬isRightTriangle 3 4 5 ∧ ¬isRightTriangle 5 12 13 ∧ ¬isRightTriangle 2 3 (Real.sqrt 13)) ∧ CannotFormRightTriangle [4, 6, 8] :=
by
  sorry

end problem_l577_577806


namespace equal_cubes_l577_577990

theorem equal_cubes (r s : ℤ) (hr : 0 ≤ r) (hs : 0 ≤ s)
  (h : |r^3 - s^3| = |6 * r^2 - 6 * s^2|) : r = s :=
by
  sorry

end equal_cubes_l577_577990


namespace triangle_area_is_64_l577_577770

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l577_577770


namespace arithmetic_mean_of_arithmetic_progression_l577_577303

variable (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ)

/-- General term of an arithmetic progression -/
def arithmetic_progression (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem arithmetic_mean_of_arithmetic_progression (k p : ℕ) (hk : 1 < k) :
  a k = (a (k - p) + a (k + p)) / 2 := by
  sorry

end arithmetic_mean_of_arithmetic_progression_l577_577303


namespace cookies_to_milk_l577_577735

theorem cookies_to_milk (Q3 : ℕ) (Qc : ℕ) (Cookies_needed_for : ℕ) :
  Q3 = 3 →
  Qc = 4 →
  Cookies_needed_for = 6 →
  (Q3 * Qc * 6) / 18 = 4 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold_coes
  norm_num
  sorry

end cookies_to_milk_l577_577735


namespace range_g_eq_arctan2_neg_pi_over_2_l577_577912

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_g_eq_arctan2_neg_pi_over_2 :
  set.range g = {y : ℝ | y = -real.pi / 2 ∨ y = real.arctan 2} :=
sorry

end range_g_eq_arctan2_neg_pi_over_2_l577_577912


namespace sum_floor_log2_l577_577086

open Int

theorem sum_floor_log2 (S: ℕ) (hS : S = ∑ N in (finset.range 2048).map (λ x, x + 1), ⌊log (N : ℝ) / log 2⌋) : S = 45055 :=
sorry

end sum_floor_log2_l577_577086


namespace w_function_relationship_minimum_value_of_w_l577_577427

def f (t : ℕ) (ht : 1 ≤ t ∧ t ≤ 30) : ℝ := 4 + (1 / t)
def g (t : ℕ) (ht : 1 ≤ t ∧ t ≤ 30) : ℝ := 115 - abs (t - 15)

def w (t : ℕ) (ht : 1 ≤ t ∧ t ≤ 30 ∧ t ≠ 0) : ℝ :=
if h : 1 ≤ t ∧ t < 15 then f t ht * (t + 100)
else f t ht * (130 - t)

theorem w_function_relationship (t : ℕ) (ht : 1 ≤ t ∧ t ≤ 30) (h₀ : t ≠ 0) :
  w t (And.intro ht h₀) = 
  if h : 1 ≤ t ∧ t < 15 then (4 + (1 / t)) * (t + 100)
  else (4 + (1 / t)) * (130 - t) :=
by
  sorry

theorem minimum_value_of_w :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ w t ⟨and.intro (and.intro (by simp) (by simp)) (by simp)⟩ = 403 + 1 / 3 :=
by
  sorry

end w_function_relationship_minimum_value_of_w_l577_577427


namespace expectation_of_poisson_variance_of_poisson_l577_577002

variables {X : Type} (λ : ℝ) [hλ : λ > 0]

noncomputable def poisson_pmf (m : ℕ) : ℝ := (λ^m * (Real.exp (-λ))) / (Nat.factorial m)

def expectation_poisson (X : ℕ → ℝ) : ℝ := ∑' m, m * X m

def variance_poisson (X : ℕ → ℝ) : ℝ := ∑' m, m^2 * X m - (expectation_poisson X)^2

theorem expectation_of_poisson (λ : ℝ) (hλ : 0 < λ) :
  expectation_poisson (poisson_pmf λ) = λ :=
sorry

theorem variance_of_poisson (λ : ℝ) (hλ : 0 < λ) :
  variance_poisson (poisson_pmf λ) = λ :=
sorry

end expectation_of_poisson_variance_of_poisson_l577_577002


namespace plane_divides_BC_l577_577415

-- Define the points and midpoints in the pyramid
structure Point := (x y z : ℝ)
structure Pyramid := (A B C D : Point)

-- The plane passes through the midpoints of AB and CD
def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2,
    z := (p1.z + p2.z) / 2 }

-- Define the points in the pyramid ABCD
variables (A B C D : Point)
def K := midpoint A B
def M := midpoint C D

-- The plane divides edge AD in the ratio 3:1 from A
def divides_in_ratio (p1 p2 : Point) (r1 r2 : ℕ) : Prop :=
  ∃ P : Point, (P = { x := (r1 * p2.x + r2 * p1.x) / (r1 + r2), 
                      y := (r1 * p2.y + r2 * p1.y) / (r1 + r2), 
                      z := (r1 * p2.z + r2 * p1.z) / (r1 + r2) })

-- The condition that the plane divides AD in the prescribed ratio
axiom plane_divides_AD : divides_in_ratio A D 3 1

-- The proof to show the plane divides BC in the ratio 3:1 from B
theorem plane_divides_BC (A B C D : Point) :
  divides_in_ratio B C 3 1 :=
sorry

end plane_divides_BC_l577_577415


namespace arithmetic_sequence_sum_l577_577505

noncomputable def sum_of_first_13_terms (a : ℕ → ℝ) : ℝ := 
  let a3 := (a 3)
    a11 := (a 11)
  (13 / 2) * (a3 + a11)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : a 1 + a 3 + a 5 = 12) (h2 : a 10 + a 11 + a 12 = 24) :
  sum_of_first_13_terms a = 78 :=
sorry

end arithmetic_sequence_sum_l577_577505


namespace surjective_function_unique_l577_577127

-- Assuming the definition for a function from ℕ to ℕ
variable (f : ℕ → ℕ)

-- Adding the required condition as a hypothesis
axiom surjective (h : f : ℕ → ℕ) : ∀ y : ℕ, ∃ x : ℕ, f x = y

-- Adding the second condition related to primes
axiom condition (h : f : ℕ → ℕ) : ∀ m n : ℕ, ∀ p : ℕ, prime p → (p ∣ f (m + n) ↔ p ∣ (f m + f n))

-- Theorem statement we want to prove
theorem surjective_function_unique {f : ℕ → ℕ} (hsurj : surjective f) (hcond : condition f) : f = id :=
by sorry

end surjective_function_unique_l577_577127


namespace square_line_product_l577_577695

theorem square_line_product (b : ℝ) (h1 : y = 3) (h2 : y = 7) (h3 := x = 2) (h4 : x = b) : 
  (b = 6 ∨ b = -2) → (6 * -2 = -12) :=
by
  sorry

end square_line_product_l577_577695


namespace cost_price_of_one_ball_l577_577815

theorem cost_price_of_one_ball (x : ℝ) (h : 11 * x - 720 = 5 * x) : x = 120 :=
sorry

end cost_price_of_one_ball_l577_577815


namespace area_ratio_triangle_EFP_EPG_l577_577240

noncomputable def square_EFGH (EF : ℝ) (N : (ℝ × ℝ)) (C : ℝ × ℝ) (P : (ℝ × ℝ)) : Prop :=
  let E := (0, 0) in
  let F := (EF, 0) in
  let G := (EF, EF) in
  let H := (0, EF) in
  EF = 8 ∧
  N = ((EF + 0) / 2, EF) ∧
  C = (6, EF) ∧
  P = ( /* coordinates of intersection point of EC and FN */ ) ∧
  -- Function to compute the areas of triangles EFP and EPQ to prove the area ratio
  let area_EFP := /-- function to compute area of triangle EFP --/ in
  let area_EPG := /-- function to compute area of triangle EPG --/ in
  area_EFP / area_EPG = 2 / 3

-- Theorem statement
theorem area_ratio_triangle_EFP_EPG :
  ∀ (EF : ℝ) (N : (ℝ × ℝ)) (C : (ℝ × ℝ)) (P : (ℝ × ℝ)), 
    square_EFGH EF N C P → 
    (let area_EFP := /-- function to compute area of triangle EFP --/ in
     let area_EPG := /-- function to compute area of triangle EPG --/ in
     area_EFP / area_EPG = 2 / 3) :=
begin
  -- Proof omitted
  sorry
end

end area_ratio_triangle_EFP_EPG_l577_577240


namespace angle_A_is_pi_div_3_length_b_l577_577510

open Real

theorem angle_A_is_pi_div_3
  (A B C : ℝ) (a b c : ℝ)
  (hABC : A + B + C = π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (hm : m = (sqrt 3, cos (π - A) - 1))
  (hn : n = (cos (π / 2 - A), 1))
  (horthogonal : m.1 * n.1 + m.2 * n.2 = 0) :
  A = π / 3 := 
sorry

theorem length_b 
  (A B : ℝ) (a b : ℝ)
  (hA : A = π / 3)
  (ha : a = 2)
  (hcosB : cos B = sqrt 3 / 3) :
  b = 4 * sqrt 2 / 3 :=
sorry

end angle_A_is_pi_div_3_length_b_l577_577510


namespace jorge_spent_amount_l577_577596

theorem jorge_spent_amount
  (num_tickets : ℕ)
  (price_per_ticket : ℕ)
  (discount_percentage : ℚ)
  (h1 : num_tickets = 24)
  (h2 : price_per_ticket = 7)
  (h3 : discount_percentage = 0.5) :
  num_tickets * price_per_ticket * (1 - discount_percentage) = 84 := 
by
  simp [h1, h2, h3]
  sorry

end jorge_spent_amount_l577_577596


namespace max_sum_squares_in_circle_is_equilateral_l577_577863

noncomputable def maximize_sum_squares_of_lengths (r : ℝ) : Triangle :=
  sorry

theorem max_sum_squares_in_circle_is_equilateral (r : ℝ) :
  maximize_sum_squares_of_lengths r = equilateral_triangle r :=
sorry

end max_sum_squares_in_circle_is_equilateral_l577_577863


namespace ratio_pq_equilateral_triangle_l577_577374

theorem ratio_pq_equilateral_triangle (p q : ℝ) (a : ℝ) 
  (h : 0 < p ∧ 0 < q ∧ 0 < a) 
  (area_relation : (19 / 64) * a^2 = p^2 + q^2 - p * q) : 
  p / q = 5 / 3 ∨ p / q = 3 / 5 := 
sorry

end ratio_pq_equilateral_triangle_l577_577374


namespace hyperbola_h_k_a_b_l577_577566

theorem hyperbola_h_k_a_b : 
    let h := -3
    let k := 1
    let a := 4
    let c := Real.sqrt 41
    let b := Real.sqrt (c ^ 2 - a ^ 2)
in
h + k + a + b = 7 :=
by
  let h := -3
  let k := 1
  let a := 4
  let c := Real.sqrt 41
  let b := Real.sqrt (c ^ 2 - a ^ 2)
  have h_eq : h = -3 := rfl
  have k_eq : k = 1 := rfl
  have a_eq : a = 4 := rfl
  have b_eq : b = 5 := by
    unfold b
    rw [c, a]
    norm_num
  rw [h_eq, k_eq, a_eq, b_eq]
  norm_num
  sorry

end hyperbola_h_k_a_b_l577_577566


namespace length_of_bridge_l577_577860

def train_length : ℝ := 200
def time_cross_lamp_post : ℝ := 5
def time_cross_bridge : ℝ := 10
def speed (d t : ℝ) : ℝ := d / t

theorem length_of_bridge :
    let v := speed train_length time_cross_lamp_post in
    v = 40 →
    let total_distance := train_length + v * time_cross_bridge in
    total_distance - train_length = 200 :=
begin
    intros v v_eq,
    have h1 : v = 40, from v_eq,
    have h2 : total_distance = train_length + v * time_cross_bridge,
    { rw [v_eq], simp [train_length, time_cross_bridge] },
    rw [h2, h1], simp [train_length],
    linarith,
end

end length_of_bridge_l577_577860


namespace smallest_sum_pentagon_sides_l577_577710

theorem smallest_sum_pentagon_sides : 
  ∃ (S : ℕ), 
  (∀ (a b c : ℕ), a + b + c = S) ∧ 
  (∑ i in finset.range 10, i + 1 = 55) ∧ 
  (S = 14) :=
by
  sorry

end smallest_sum_pentagon_sides_l577_577710


namespace four_consecutive_integers_divisible_by_24_l577_577794

noncomputable def product_of_consecutive_integers (n : ℤ) : ℤ :=
  n * (n + 1) * (n + 2) * (n + 3)

theorem four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ product_of_consecutive_integers n :=
by
  sorry

end four_consecutive_integers_divisible_by_24_l577_577794


namespace find_min_a_l577_577544

theorem find_min_a (a : ℕ) (h1 : (3150 * a) = x^2) (h2 : a > 0) :
  a = 14 := by
  sorry

end find_min_a_l577_577544


namespace p_even_and_p_pi_periodic_q_even_and_q_pi_periodic_l577_577824

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := (f x + f (-x)) / 2
def h (x : ℝ) : ℝ := (f x - f (-x)) / 2
def p (x : ℝ) : ℝ :=
  if ∃ k : ℤ, x = ↑k * π + π / 2 then 0
  else (g x - g (x + π)) / (2 * Real.cos x)
def q (x : ℝ) : ℝ :=
  if ∃ k : ℤ, x = ↑k * π / 2 then 0
  else (h x + h (x + π)) / (2 * Real.sin (2 * x))

theorem p_even_and_p_pi_periodic :
  ∀ x : ℝ, p (-x) = p x ∧ p (x + π) = p x := sorry

theorem q_even_and_q_pi_periodic :
  ∀ x : ℝ, q (-x) = q x ∧ q (x + π) = q x := sorry

end p_even_and_p_pi_periodic_q_even_and_q_pi_periodic_l577_577824


namespace abs_w_of_square_l577_577324

theorem abs_w_of_square (w : ℂ) (h : w^2 = -48 + 14 * complex.i) : complex.abs w = 5 * real.sqrt 2 :=
sorry

end abs_w_of_square_l577_577324


namespace original_smallest_element_l577_577699

theorem original_smallest_element (x : ℤ) 
  (h1 : x < -1) 
  (h2 : x + 14 + 0 + 6 + 9 = 2 * (2 + 3 + 0 + 6 + 9)) : 
  x = -4 :=
by sorry

end original_smallest_element_l577_577699


namespace new_person_age_l577_577387

theorem new_person_age (T A : ℕ) : 
  let original_average := T / 10 in
  let new_total_age := T - 48 + A in 
  let new_average := new_total_age / 10 in 
  new_average = original_average - 3 → A = 18 := 
sorry

end new_person_age_l577_577387


namespace sum_of_reciprocal_S_n_l577_577265

theorem sum_of_reciprocal_S_n (S : ℕ → ℚ) (a_n : ℕ → ℚ) (a1 : ℚ) (d : ℚ):
    (∀ n, S n = n * a1 + (n * (n-1) / 2) * d) → 
    a1 = 1 →
    (S 2017 / 2017 - S 2015 / 2015 = 1) →
    ∑ k in finset.range 2017, (1 / S (k + 1)) = 2017 / 1009 := sorry

end sum_of_reciprocal_S_n_l577_577265


namespace incircle_median_ratios_l577_577426

theorem incircle_median_ratios (ABC : Triangle) (AD : Median ABC) (k : Circle) :
  (incircle_divides_median_three_parts k AD) →
  (divides_other_medians ABC k AD = {(2, 16, 9), (38, 16, 3)}) :=
by 
  sorry

end incircle_median_ratios_l577_577426


namespace star_star_value_l577_577485

-- Define the operation v * = v - v / 3
def star (v : ℝ) : ℝ := v - (v / 3)

-- Given condition: v = 35.99999999999999
def v : ℝ := 35.99999999999999

-- Define the next application of star operation: (v *) *
def star_star (v : ℝ) : ℝ := star (star v)

-- The theorem to prove
theorem star_star_value : star_star v = 15.999999999999996 := 
by sorry

end star_star_value_l577_577485


namespace min_rows_512_l577_577116

theorem min_rows_512 (n : ℕ) (table : ℕ → ℕ → ℕ) 
  (H : ∀ A (i j : ℕ), i < 10 → j < 10 → i ≠ j → ∃ B, B < n ∧ (table B i ≠ table A i) ∧ (table B j ≠ table A j) ∧ ∀ k, k ≠ i ∧ k ≠ j → table B k = table A k) : 
  n ≥ 512 :=
sorry

end min_rows_512_l577_577116


namespace ordered_triple_unique_l577_577612

theorem ordered_triple_unique (a b c : ℝ) (h2 : a > 2) (h3 : b > 2) (h4 : c > 2)
    (h : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
    a = 7 ∧ b = 5 ∧ c = 3 :=
sorry

end ordered_triple_unique_l577_577612


namespace integer_roots_of_polynomial_l577_577918

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 6 * x^2 - 4 * x + 24 = 0} = {2, -2} :=
by
  sorry

end integer_roots_of_polynomial_l577_577918


namespace max_sum_squares_polygon_circle_l577_577865

/-- Given that a polygon is inscribed in a circle, prove that the inscribed polygon
with the maximum sum of the squares of the lengths of its sides is an equilateral triangle. -/
theorem max_sum_squares_polygon_circle (R : ℝ)
  (P : Type) [Geometry.Polygon P] [Geometry.Inscribed P R] :
  ∃ (E : Geometry.Triangle), 
  Geometry.Equilateral E ∧ Geometry.Inscribed E R ∧
  ∀ (Q : Type) [Geometry.Polygon Q] [Geometry.Inscribed Q R],
  ∑ (s : ℝ) in Geometry.sides_lengths Q, s^2 ≤ ∑ (t : ℝ) in Geometry.sides_lengths E, t^2 :=
sorry

end max_sum_squares_polygon_circle_l577_577865


namespace range_m_union_A_B_l577_577168

open Set

section problem_statement

variable {R : Type} [LinearOrderedField R]

def A (x : R) : Prop := x^2 + x - 2 < 0
def B (x : R) (m : R) : Prop := x^2 + 2 * m * x + m^2 - 1 < 0

--- 1. Define the range of m such that A' ∩ B = ∅
theorem range_m (m : R) : (∀ x, (¬(A x) ∧ B x m) → False) ↔ 0 ≤ m ∧ m ≤ 1 :=
begin
  sorry
end

--- 2. Define the union of A and B given A ∩ B contains exactly one integer element
theorem union_A_B (m : R) :
  (∃ x ∈ Icc (-2 : R) (1 : R), A x ∧ B x m ∧ ∀ y, A y ∧ B y m → y = x) →
  (1 ≤ m ∧ m < 2 → A ∪ B = λ x, -1 - m < x ∧ x < 1) ∧ 
  (-1 < m ∧ m ≤ 0 → A ∪ B = λ x, -2 < x ∧ x < 1 - m) :=
begin
  sorry
end

end problem_statement

end range_m_union_A_B_l577_577168


namespace garden_not_taken_by_pond_l577_577052

theorem garden_not_taken_by_pond (perimeter_garden : ℝ) (area_pond : ℝ)
  (h1 : perimeter_garden = 48) (h2 : area_pond = 20) : 
  (let side_length := perimeter_garden / 4 in 
  let area_garden := side_length * side_length in 
  area_garden - area_pond) = 124 :=
by {
  sorry
}

end garden_not_taken_by_pond_l577_577052


namespace quadrilateral_is_parallelogram_l577_577581

variables {A B C D : Type}
variables (angle_DAB angle_ABC angle_BAD angle_BCD : ℕ)
variables (is_supplementary : angle_DAB + angle_ABC = 180)
variables (angle_BAD_45 : angle_BAD = 45)
variables (angle_BCD_45 : angle_BCD = 45)

def is_parallel (line1 line2 : Type) : Prop := sorry

def is_parallelogram {A B C D : Type} (AB BC CD DA : Type) : Prop :=
  is_parallel AB CD ∧ is_parallel BC DA

theorem quadrilateral_is_parallelogram (A B C D : Type)
  (line_AB : Type) (line_BC : Type) (line_CD : Type) (line_DA : Type)
  (h1 : is_supplementary)
  (h2 : angle_BAD_45)
  (h3 : angle_BCD_45) :
  is_parallelogram line_AB line_BC line_CD line_DA :=
sorry

end quadrilateral_is_parallelogram_l577_577581


namespace pieces_form_parallelogram_l577_577567

theorem pieces_form_parallelogram (n : ℕ) (h1 : n ≥ 2) (pieces : set (fin n × fin n)) (h2 : pieces.size = 2 * n) :
  ∃ (a b c d : fin n × fin n),
    a ∈ pieces ∧ b ∈ pieces ∧ c ∈ pieces ∧ d ∈ pieces ∧
    (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 = c.2 ∧ b.2 = d.2) ∨
    (a.1 = c.1 ∧ b.1 = d.1 ∧ a.2 = b.2 ∧ c.2 = d.2) ∨
    (a.1 = d.1 ∧ b.1 = c.1 ∧ a.2 = b.2 ∧ c.2 = d.2) ∨
    (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 = b.2 ∧ c.2 = d.2) :=
by sorry

end pieces_form_parallelogram_l577_577567


namespace find_x_l577_577925

noncomputable def x : ℝ :=
  if h : (∃ x : ℝ, 0 < x ∧ x * ⌊x⌋ = 24) 
  then classical.some h 
  else 0

theorem find_x (h_x : ∃ x : ℝ, 0 < x ∧ x * ⌊x⌋ = 24) :
  x = 6 := 
sorry

end find_x_l577_577925


namespace max_naive_number_l577_577928

-- Define the digits and conditions for a naive number
variable (a b c d : ℕ)
variable (M : ℕ)
variable (h1 : b = c + 2)
variable (h2 : a = d + 6)
variable (h3 : M = 1000 * a + 100 * b + 10 * c + d)

-- Define P(M) and Q(M)
def P (a b c d : ℕ) : ℕ := 3 * (a + b) + c + d
def Q (a : ℕ) : ℕ := a - 5

-- Problem statement: Prove the maximum value of M satisfying the divisibility condition
theorem max_naive_number (div_cond : (P a b c d) % (Q a) = 0) (hq : Q a % 10 = 0) : M = 9313 := 
sorry

end max_naive_number_l577_577928


namespace total_water_bottles_needed_l577_577466

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l577_577466


namespace range_of_k_l577_577434

/-- 
Given the ellipse equation \( \frac{x^2}{4} + \frac{y^2}{k} = 1 \),
and the angle condition \( \angle APB = 120^\circ \) for some point \( P \) on the ellipse where \( A \) and \( B \) are the foci,
prove that \( k \in \left(0,1\right] \cup \left[16,+\infty \right) \).
-/
theorem range_of_k (k : ℝ) (x y : ℝ) (A B P : ℝ × ℝ)
  (h1 : A = (-2, 0)) (h2 : B = (2, 0)) (h3 : (P.1 ^ 2) / 4 + (P.2 ^ 2) / k = 1)
  (h4 : (A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1) ≠ 0)
  (h5 : ∃ P, (P.1 ^ 2) / 4 + (P.2 ^ 2) / k = 1 ∧ (A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1) ≠ 0) :
  k ∈ set.Icc 0 1 ∪ set.Ici 16 := 
sorry

end range_of_k_l577_577434


namespace linear_function_val_difference_l577_577281

theorem linear_function_val_difference (g : ℝ → ℝ) (h_lin : ∀ x y, g(x) - g(y) = (x - y) * ((g(10) - g(4)) / (10 - 4)))
  (h_cond : g(10) - g(4) = 18) : g(16) - g(4) = 36 :=
by
  sorry

end linear_function_val_difference_l577_577281


namespace vertex_of_parabola_l577_577334

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

-- Define the vertex point
def vertex : ℝ × ℝ := (-2, -1)

-- The theorem we need to prove
theorem vertex_of_parabola : ∀ x : ℝ, parabola x = (x + 2)^2 - 1 → vertex = (-2, -1) := 
by
  sorry

end vertex_of_parabola_l577_577334


namespace smallest_base_for_200_l577_577113

-- Define the conditions
def condition (b : ℕ) : Prop := b^5 ≤ 200 ∧ 200 < b^6

-- Define the statement to be proven
theorem smallest_base_for_200 : ∃ b : ℕ, condition b ∧ ∀ b', condition b' → b ≤ b' :=
begin
  sorry, -- The proof goes here
end

end smallest_base_for_200_l577_577113


namespace volume_of_inequality_region_l577_577926

noncomputable def region_volume : ℝ :=
  let region (x y z : ℝ) := |x + 2*y + z| + |x + y - z| + |x - y + 2*z| + |-x + y + z| ≤ 8
  volume { p : ℝ × ℝ × ℝ | region p.1 p.2 p.3 }

theorem volume_of_inequality_region : region_volume = 832 / 15 :=
sorry

end volume_of_inequality_region_l577_577926


namespace arithmetic_sequence_length_l577_577894

theorem arithmetic_sequence_length :
  ∃ n : ℕ, 
    (∀ k : ℕ, 1 ≤ k → k ≤ n → -3 + (k - 1) * 4 ≠ 45) ∧ -3 + (n - 1) * 4 = 45 :=
by {
  sorry
}

end arithmetic_sequence_length_l577_577894


namespace find_x0_and_m_l577_577191

theorem find_x0_and_m (x : ℝ) (m : ℝ) (x0 : ℝ) :
  (abs (x + 3) - 2 * x - 1 < 0 ↔ x > 2) ∧ 
  (∃ x, abs (x - m) + abs (x + 1 / m) - 2 = 0) → 
  (x0 = 2 ∧ m = 1) := 
by
  sorry

end find_x0_and_m_l577_577191


namespace pirate_total_dollar_amount_l577_577028

def base_5_to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨p, d⟩ => d * base^p) |>.sum

def jewelry_base5 := [3, 1, 2, 4]
def gold_coins_base5 := [3, 1, 2, 2]
def alcohol_base5 := [1, 2, 4]

def jewelry_base10 := base_5_to_base_10 jewelry_base5 5
def gold_coins_base10 := base_5_to_base_10 gold_coins_base5 5
def alcohol_base10 := base_5_to_base_10 alcohol_base5 5

def total_base10 := jewelry_base10 + gold_coins_base10 + alcohol_base10

theorem pirate_total_dollar_amount :
  total_base10 = 865 :=
by
  unfold total_base10 jewelry_base10 gold_coins_base10 alcohol_base10 base_5_to_base_10
  simp
  sorry

end pirate_total_dollar_amount_l577_577028


namespace solve_number_l577_577040

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l577_577040


namespace largest_k_divides_factorial_l577_577880

theorem largest_k_divides_factorial (k : ℕ) :
  (∀ n : ℕ, (2024^k ∣ n! ↔ k ≤ ∑ i in finset.range (n + 1), (n / 2024^i))) → k = 91 :=
by
  sorry

end largest_k_divides_factorial_l577_577880


namespace product_of_common_divisors_l577_577137

theorem product_of_common_divisors:
  ∀ d : ℤ, d ∣ 180 → d ∣ 30 → d ≠ 0 → (∏ x in {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} , x) = 1 :=
by {
  sorry
}

end product_of_common_divisors_l577_577137


namespace area_of_triangle_l577_577760

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l577_577760


namespace sum_floor_log2_l577_577078

theorem sum_floor_log2 :
  (∑ N in Finset.range 2048, Int.floor (Real.log N / Real.log 2)) = 20445 :=
by
  sorry

end sum_floor_log2_l577_577078


namespace triangle_area_is_64_l577_577768

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l577_577768


namespace pow_tower_seq_monotone_and_bounded_l577_577747

noncomputable def pow_tower_seq : ℕ → ℝ
| 0 => 1
| (n + 1) => (real.sqrt 2) ^ pow_tower_seq n

theorem pow_tower_seq_monotone_and_bounded :
  (∀ n : ℕ, pow_tower_seq n < pow_tower_seq (n + 1)) ∧ (∀ n : ℕ, pow_tower_seq n ≤ 2) :=
by
  sorry

end pow_tower_seq_monotone_and_bounded_l577_577747


namespace knights_count_l577_577645

-- Define the inhabitants and their nature
inductive Inhabitant : Type
| first : Inhabitant
| second : Inhabitant
| third : Inhabitant
| fourth : Inhabitant
| fifth : Inhabitant

open Inhabitant

-- Define whether an inhabitant is a knight or a liar
inductive Nature : Type
| knight : Nature
| liar : Nature

open Nature

-- Assume each individual is either a knight or a liar (truth value function)
def truth_value : Inhabitant → Nature → Prop
| first, knight  => sorry -- To be proven
| second, knight => sorry -- To be proven
| third, knight  => sorry -- To be proven
| fourth, knight => sorry -- To be proven
| fifth, knight  => sorry -- To be proven

-- Define the statements made by inhabitants as logical conditions
def statements : Prop :=
  (truth_value first knight → (1 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value second knight → (2 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value third knight → (3 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value fourth knight → ¬ (truth_value first knight) ∧ ¬ (truth_value second knight) ∧ ¬ (truth_value third knight) ∧ ¬ (truth_value fifth knight))
  ∧ (truth_value fifth knight → ¬ (truth_value fourth knight))

-- The goal is to prove that there are exactly 2 knights
theorem knights_count : (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0) = 2 :=
by
  sorry

end knights_count_l577_577645


namespace exists_two_digit_pair_product_l577_577344

theorem exists_two_digit_pair_product (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (hprod : a * b = 8670) : a * b = 8670 :=
by
  exact hprod

end exists_two_digit_pair_product_l577_577344


namespace h2o_formed_l577_577920

-- Definition of the given conditions
def NH4Cl : Type := ℕ  -- Number of NH4Cl in moles
def NaOH : Type := ℕ   -- Number of NaOH in moles
def H2O : Type := ℕ    -- Number of H2O in moles

-- Reaction ratio property
axiom reaction_ratio (n_NH4Cl n_NaOH : NH4Cl) : n_NH4Cl = n_NaOH → n_NH4Cl = n_NaOH

-- Given quantities
def given_NH4Cl : NH4Cl := 3
def given_NaOH : NaOH := 3

-- Proof statement: Amount of H2O formed
theorem h2o_formed :
  ∀ (NH4Cl NaOH H2O : Type) (n_NH4Cl : NH4Cl) (n_NaOH : NaOH),
    given_NH4Cl = 3 ∧ given_NaOH = 3 → (reaction_ratio n_NH4Cl n_NaOH → H2O = 3) := by
  sorry

end h2o_formed_l577_577920


namespace water_bottles_needed_l577_577464

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l577_577464


namespace ratio_of_distances_l577_577266

noncomputable def VABCD (V A B C D P: Point) : Prop := 
  is_right_square_pyramid V A B C D ∧
  P_inside_square_base A B C D P

noncomputable def distance_from_point_to_faces (V A B C D P: Point) : ℝ := 
  sum_of_distances_from_point_to_faces V A B C D P

noncomputable def distance_from_point_to_sides (A B C D P: Point) : ℝ := 
  sum_of_distances_from_point_to_sides A B C D P

theorem ratio_of_distances 
  (V A B C D P: Point)
  (h1: is_right_square_pyramid V A B C D)
  (h2: P_inside_square_base A B C D P) :
  (distance_from_point_to_faces V A B C D P) / (distance_from_point_to_sides A B C D P) = (2 * (sqrt 2) / 3) :=
 by
  sorry

end ratio_of_distances_l577_577266


namespace abby_area_l577_577882

noncomputable def abbyRoamingArea : ℝ :=
  let radius := 5 -- leash length
  let fractionOfCircle := 3 / 4
  let additionalRadius := 1 -- extra length beyond the barn at the bottom side
  let mainArea := fractionOfCircle * π * radius^2
  let additionalArea := (1 / 4) * π * additionalRadius^2
  mainArea + additionalArea

theorem abby_area : abbyRoamingArea = 19 * π := by
  let radius := 5 : ℝ
  let fractionOfCircle := 3 / 4 : ℝ
  let additionalRadius := 1 : ℝ
  let mainArea := fractionOfCircle * π * radius^2
  let additionalArea := (1 / 4) * π * additionalRadius^2
  have mainArea_calc : mainArea = (75 / 4) * π :=
    by simp [mainArea, fractionOfCircle, pow_two, mul_assoc]; linarith
  have additionalArea_calc : additionalArea = (1 / 4) * π :=
    by simp [additionalArea, additionalRadius, pow_two, mul_assoc]; linarith
  have total_area_calc : mainArea + additionalArea = (76 / 4) * π :=
    by rw [mainArea_calc, additionalArea_calc]; linarith
  show abbyRoamingArea = 19 * π, by
    simp [abbyRoamingArea, mainArea, additionalArea, total_area_calc]; linarith

end abby_area_l577_577882


namespace locus_of_point_M_max_area_triangle_AOB_l577_577850

theorem locus_of_point_M :
  (∀ (x0 y0 : ℝ), x0^2 + y0^2 = 3 →
  (∃ (x y : ℝ), x0 = x ∧ y0 = √3 * y ∧ (P x0 y0))) →
  ∃ M : ℝ × ℝ, (let (x, y) := M in x^2 / 3 + y^2 = 1) ∧ 
  eccentricity (ellipse_eqn x y) = √6 / 3 :=
sorry

theorem max_area_triangle_AOB :
  (∀ (l : ℝ), l = √3 / 2 →
  (∃ (A B : ℝ × ℝ), 
    A ∈ ellipse_locus ∧ B ∈ ellipse_locus ∧ 
    maximum_area_triangle_origin A B  = √3 / 2)) :=
sorry

end locus_of_point_M_max_area_triangle_AOB_l577_577850


namespace max_value_of_linear_expression_l577_577172

theorem max_value_of_linear_expression (x y : ℝ) (h : x^2 + y^2 = 16x + 8y + 20) : 
  ∃ z, z = 4 * x + 3 * y ∧ z ≤ 40 :=
by
  sorry

end max_value_of_linear_expression_l577_577172


namespace water_bottles_needed_l577_577463

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l577_577463


namespace factorial_mod_10_13_l577_577937

theorem factorial_mod_10_13 : 10! % 13 = 7 := by
  sorry

end factorial_mod_10_13_l577_577937


namespace john_spends_on_dog_treats_l577_577593

def number_of_days_in_month := 30
def treats_per_day := 2
def cost_per_treat := 0.1

theorem john_spends_on_dog_treats : 
  (number_of_days_in_month * treats_per_day * cost_per_treat) = 6 := 
by 
  sorry

end john_spends_on_dog_treats_l577_577593


namespace simplest_fraction_C_l577_577868

def FractionA (a : ℚ) : ℚ := (a + 1) / (a^2 - 1)
def FractionB (a b c : ℚ) : ℚ := 4 * a / (6 * b * c^2)
def FractionC (a : ℚ) : ℚ := 2 * a / (2 - a)
def FractionD (a b : ℚ) : ℚ := (a + b) / (a^2 + a * b)

theorem simplest_fraction_C
  (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (condA : ∀ a, FractionA a = (1 : ℚ)) 
  (condB : ∀ a b c, FractionB a b c = (1 : ℚ))
  (condC : ∀ a, ¬ ∃ d, d*a = 2 - a)
  (condD : ∀ a b, FractionD a b = (1 : ℚ)) :
  (∀ x, x ≠ a ≠ 1 → x ≠ b ≠ 1 → x ≠ c ≠ 1 → x ≠ 0) → FractionC a = 2 * a / (2 - a) :=
begin
  sorry
end

end simplest_fraction_C_l577_577868


namespace value_of_g_at_pi_div_3_l577_577183

theorem value_of_g_at_pi_div_3
  (ω φ : ℝ)
  (h_f_symm : ∀ x : ℝ, (1 / 2) * Real.cos (ω * (π / 3 - x) + φ) = (1 / 2) * Real.cos (ω * (π / 3 + x) + φ))
  (g : ℝ → ℝ := λ x, 3 * Real.sin (ω * x + φ) - 2) :
  g (π / 3) = -5 := 
sorry

end value_of_g_at_pi_div_3_l577_577183


namespace sixty_six_dips_eq_twenty_two_point_five_daps_l577_577542

-- Definitions of the equivalence relationships
def five_daps_eq_four_dops := 5 * daps = 4 * dops
def three_dops_eq_eleven_dips := 3 * dops = 11 * dips

-- Theorem to prove the number of daps equivalent to 66 dips
theorem sixty_six_dips_eq_twenty_two_point_five_daps 
  (five_daps_eq_four_dops : 5 * daps = 4 * dops) 
  (three_dops_eq_eleven_dips : 3 * dops = 11 * dips) : 
  66 * dips = 22.5 * daps :=
sorry

end sixty_six_dips_eq_twenty_two_point_five_daps_l577_577542


namespace mushroom_finding_chocolate_division_l577_577733

section MushroomProblem

variables (total_mushrooms : ℕ) (Liba_contribution Maruska_contribution Sarka_contribution : ℕ)
variable (total_chocolates : ℕ)

-- Total mushrooms found
def totalMushrooms := 55

-- Contributions to the dish
def LibaContribution := 6
def MaruskaContribution := 8
def SarkaContribution := 5

-- Chocolates to divide
def totalChocolates := 38

-- Remaining mushrooms (should be equal among friends)
def remainingMushrooms (total_found contributed : ℕ) : ℕ :=
total_found - contributed

theorem mushroom_finding :
  let total_used := LibaContribution + MaruskaContribution + SarkaContribution in
  let remaining_per_person := (totalMushrooms - total_used) / 3 in
    remaining_per_person + LibaContribution = 18 ∧
    remaining_per_person + MaruskaContribution = 20 ∧
    remaining_per_person + SarkaContribution = 17 :=
by
  sorry

theorem chocolate_division :
  let liba_portion := LibaContribution - 4.75 in
  let maruska_portion := MaruskaContribution - 4.75 in
  let sarka_portion := SarkaContribution - 4.75 in
  let total_portion := liba_portion + maruska_portion + sarka_portion in
  let ratio_liba := liba_portion / total_portion in
  let ratio_maruska := maruska_portion / total_portion in
  let ratio_sarka := sarka_portion / total_portion in
  ratio_liba * totalChocolates = 10 ∧
  ratio_maruska * totalChocolates = 26 ∧
  ratio_sarka * totalChocolates = 2 :=
by
  sorry

end MushroomProblem

end mushroom_finding_chocolate_division_l577_577733


namespace possible_values_of_sum_of_digits_of_squares_l577_577919

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
(n.digits 10).sum

noncomputable def f (n : ℕ) : ℕ :=
sum_of_digits (n * n)

theorem possible_values_of_sum_of_digits_of_squares :
  {x : ℕ | ∃ n : ℕ, f n = x } = {0, 1, 4, 7} ∩ (λ (m : ℕ), m % 9 = 0 ∨ m % 9 = 1 ∨ m % 9 = 4 ∨ m % 9 = 7) :=
by
  sorry

end possible_values_of_sum_of_digits_of_squares_l577_577919


namespace find_perpendicular_line_l577_577055

-- Definitions related to the conditions
def point := ℝ × ℝ
def is_perpendicular_to (m1 m2 : ℝ) : Prop := m1 * m2 = -1
def slope_of_line (a b : ℝ) (c : ℝ) : ℝ := -a / b  -- Assumes b ≠ 0

-- Given conditions
def point1 : point := (1, 3)
def line1_coeffs : ℝ × ℝ × ℝ := (2, -6, -8)
def line1_slope : ℝ := slope_of_line 2 (-6) (-8)
def perpendicular_slope : ℝ := -1 / line1_slope

-- Define the desired line passing through (1, 3) and having the perpendicular slope
def desired_line (x y : ℝ) : Prop := y + 3*x - 6 = 0

-- Prove that the equation of the line passing through point1 and perpendicular to line1 is desired_line
theorem find_perpendicular_line : ∃ k, ∀ (x y : ℝ), (y - 3 = -3 * (x - 1)) ↔ desired_line x y :=
by
  sorry

end find_perpendicular_line_l577_577055


namespace number_of_pages_to_copy_l577_577097

-- Definitions based on the given conditions
def total_budget : ℕ := 5000
def service_charge : ℕ := 500
def copy_cost : ℕ := 3

-- Derived definition based on the conditions
def remaining_budget : ℕ := total_budget - service_charge

-- The statement we need to prove
theorem number_of_pages_to_copy : (remaining_budget / copy_cost) = 1500 :=
by {
  sorry
}

end number_of_pages_to_copy_l577_577097


namespace perimeter_triangle_DPQ_l577_577739

-- Define the side lengths of the triangle DEF
variables (DE EF DF : ℝ)
-- Define points I (incenter), P, and Q
variables (I P Q : Type*)

-- Given conditions
axiom DE_length : DE = 15
axiom EF_length : EF = 30
axiom DF_length : DF = 22.5

-- Perimeter of triangle DPQ calculation
noncomputable def perimeter_DPQ : ℝ :=
  DE + DF

-- Prove that the perimeter of triangle DPQ is 37.5
theorem perimeter_triangle_DPQ : perimeter_DPQ DE DF = 37.5 :=
by
  simp [perimeter_DPQ, DE_length, DF_length]
  exact add_assoc _ _ _
  sorry

end perimeter_triangle_DPQ_l577_577739


namespace sum_floor_log2_l577_577077

theorem sum_floor_log2 :
  (∑ N in Finset.range 2048, Int.floor (Real.log N / Real.log 2)) = 20445 :=
by
  sorry

end sum_floor_log2_l577_577077


namespace rabbit_fraction_l577_577547

theorem rabbit_fraction
  (initial_rabbits : ℕ) (added_rabbits : ℕ) (total_rabbits_seen : ℕ)
  (h_initial : initial_rabbits = 13)
  (h_added : added_rabbits = 7)
  (h_seen : total_rabbits_seen = 60) :
  (initial_rabbits + added_rabbits) / total_rabbits_seen = 1 / 3 :=
by
  -- we will prove this
  sorry

end rabbit_fraction_l577_577547


namespace fg_x_eq_4_l577_577528

def f (x : ℝ) := x^2 + 3
def g (x : ℝ) := 3x + 2
def x := -1

theorem fg_x_eq_4 : f (g x) = 4 := by sorry

end fg_x_eq_4_l577_577528


namespace knights_count_is_two_l577_577642

def inhabitant (i : Fin 5) : Prop :=
  (i = 0 → inhabitants_truth 1) ∧
  (i = 1 → inhabitants_truth 2) ∧
  (i = 2 → inhabitants_truth 3) ∧
  (i = 3 → ∀ k, k ≠ 3 → ¬inhabitants_truth k) ∧
  (i = 4 → inhabitants_truth 4 ≠ inhabitants_truth 3)

def inhabitants_truth (n : ℕ) : Prop := sorry

theorem knights_count_is_two : ∃ (knights : Fin 5 → Prop), (∃! i, knights i) ∧ (∃! j, knights j) ∧ (i ≠ j) :=
sorry

end knights_count_is_two_l577_577642


namespace range_of_k_l577_577416

noncomputable def point_eq_distance (p : ℝ × ℝ) (F : ℝ × ℝ) (Lx : ℝ) : Prop :=
  dist p F = abs (p.1 - Lx)

noncomputable def parabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

noncomputable def line_through (P : ℝ × ℝ) (k : ℝ) : set (ℝ × ℝ) :=
  {q | q.2 = k * (q.1 + 2)}

theorem range_of_k (k : ℝ) :
  (∀ p : ℝ × ℝ, point_eq_distance p (1, 0) (-1) → parabola p) →
  ¬ ∃ p : ℝ × ℝ, point_eq_distance p (1, 0) (-1) ∧ p ∈ line_through (-2, 0) k →
  k < -real.sqrt 2 / 2 ∨ k > real.sqrt 2 / 2 :=
by
  sorry

end range_of_k_l577_577416


namespace side_length_equality_l577_577991

open EuclideanGeometry

noncomputable def rectangle := Π {R: Type}, R → R → Prop

variable (R1 R2 : Type) [rectangle R1] [rectangle R2]

-- Definitions for the circumcircles and touching diagonals
variable (diag_R1 : R1 → line)
variable (diag_R2 : R2 → line)
variable (circ_R1 : circle)
variable (circ_R2 : circle)

-- Conditions
variable (h1 : ∀ diag_R1 circ_R2, touches diag_R1 circ_R2)
variable (h2 : ∀ diag_R2 circ_R1, touches diag_R2 circ_R1)

-- The proof goal
theorem side_length_equality : 
  ∃ (s1 : ℝ) (s2 : ℝ), (s1 ∈ side_lengths R1) ∧ (s2 ∈ side_lengths R2) ∧ (s1 = s2) := 
sorry

end side_length_equality_l577_577991


namespace total_distance_covered_l577_577799

-- Definitions based on conditions
def distance_ran : ℝ := 40
def distance_walked : ℝ := (3 / 5) * distance_ran
def distance_jogged : ℝ := distance_walked / 5

-- Theorem stating the total distance covered
theorem total_distance_covered : distance_ran + distance_walked + distance_jogged = 64.8 := by
  -- You can place the formal proof steps here
  sorry

end total_distance_covered_l577_577799


namespace A_share_calculation_l577_577073

def investment_A := 6300
def investment_B := 4200
def investment_C := 10500

def profit_percentage_A := 0.45 
def profit_percentage_B := 0.30
def profit_percentage_C := 0.25

def total_profit := 12200

def A_share : ℤ := (profit_percentage_A * total_profit).toInt

theorem A_share_calculation :
  A_share = 5490 := 
by
  sorry

end A_share_calculation_l577_577073


namespace range_of_a_l577_577182

theorem range_of_a 
  (f : ℝ → ℝ)
  (h_def : ∀ x, f x = 2^x - 2^(-x))
  (h_ineq : ∀ x, f(x^2 - a * x + a) + f 3 > 0) :
  -2 < a ∧ a < 6 :=
sorry

end range_of_a_l577_577182


namespace semicircle_radius_inscribed_in_triangle_l577_577048

/-- Given an isosceles triangle with base 24 and height 18,
    where a semicircle is inscribed with its diameter along the base,
    the radius of the semicircle is 36 * sqrt(13) / 13. -/
theorem semicircle_radius_inscribed_in_triangle :
  ∃ (r : ℝ), 
    (∃ (base : ℝ), base = 24) ∧
    (∃ (height : ℝ), height = 18) ∧
    (∃ (diameter_base_relation : bool), diameter_base_relation = true) ∧
    r = (36 * Real.sqrt 13) / 13 :=
begin
    sorry
end

end semicircle_radius_inscribed_in_triangle_l577_577048


namespace y_intercept_of_line_l577_577360

theorem y_intercept_of_line : ∀ (x : ℝ), y = 3 * x + 2 → y_intercept y = 2 := 
sorry

end y_intercept_of_line_l577_577360


namespace find_two_digit_number_l577_577043

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l577_577043


namespace digit_in_thousandths_place_l577_577788

theorem digit_in_thousandths_place :
  (decimals (7 / 32)).nth 3 = some 8 := 
sorry

end digit_in_thousandths_place_l577_577788


namespace monotonically_increasing_a_ge_1_l577_577552

noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (Real.sin x + a)

theorem monotonically_increasing_a_ge_1 (a : ℝ) : 
  (∀ x ∈ (set.Ioo (-Real.pi / 2) (Real.pi / 2)), 0 ≤ Real.exp x * (Real.sin x + a + Real.cos x)) ↔ a ≥ 1 :=
by 
  sorry

end monotonically_increasing_a_ge_1_l577_577552


namespace train_speed_l577_577859

theorem train_speed (L : ℝ) (man_speed : ℝ) (t : ℝ) (V_train : ℝ) : 
  L = 120 → man_speed = 6 → t = 6 → V_train = (L / t * 3.6 - man_speed) → V_train = 66 :=
by
  intros hL hM ht hV
  rw [hL, hM, ht] at hV
  linarith

end train_speed_l577_577859


namespace children_count_l577_577023

variable (M W C : ℕ)

theorem children_count (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : M + W + C = 300) : C = 30 := by
  sorry

end children_count_l577_577023


namespace triangle_problems_l577_577578

open Real

noncomputable def solve_triangle_problem (a b: ℝ) (cos_B: ℝ) (area: ℝ) : Prop :=
  (sin B = sqrt (1 - cos_B ^ 2)) ∧
  (sin A = (a * sin B) / b) ∧
  (c = 5) ∧
  (b = sqrt (a ^ 2 + c ^ 2 - 2 * a * c * cos_B))

theorem triangle_problems
  (a : ℝ) (b1 b2: ℝ) (cos_B : ℝ) (area : ℝ) 
  (a_eq : a = 2)
  (cos_B_eq : cos_B = 4 / 5)
  (b1_eq: b1 = 3)
  (area_eq : area = 3):
  solve_triangle_problem a b1 cos_B area ∨ solve_triangle_problem a b2 cos_B area :=
by 
  have h := and.intro sorry (and.intro sorry (and.intro sorry sorry)); -- Fill sorrys with details as needed.
  exact h

end triangle_problems_l577_577578


namespace new_rectangle_area_gt_twice_original_l577_577622

variables (a : ℝ)

-- Given the side lengths of the original rectangle
def side1 := 2 * a
def side2 := 3 * a

-- Calculate the diagonal of the original rectangle
def diagonal := real.sqrt ((side1 ^ 2) + (side2 ^ 2))

-- Dimensions of the new rectangle
def new_base := diagonal + side2
def new_height := (3 * side2) - (1 / 2 * diagonal)

-- Area calculations
def original_area := side1 * side2
def new_area := new_base * new_height

-- Prove the inequality relation
theorem new_rectangle_area_gt_twice_original :
  new_area > 2 * original_area :=
by sorry

end new_rectangle_area_gt_twice_original_l577_577622


namespace john_spends_on_dog_treats_l577_577592

def number_of_days_in_month := 30
def treats_per_day := 2
def cost_per_treat := 0.1

theorem john_spends_on_dog_treats : 
  (number_of_days_in_month * treats_per_day * cost_per_treat) = 6 := 
by 
  sorry

end john_spends_on_dog_treats_l577_577592


namespace draws_to_exceeding_sum_probability_l577_577400

open ProbabilityTheory

noncomputable def chips := {1, 2, 3, 4, 5}

theorem draws_to_exceeding_sum_probability : 
  ∀ draws : list ℕ, (draws.nodup ∧ ∀ x ∈ draws, x ∈ chips ∧ draws.length = 3) →
  (draws.sum > 4) →
  (∃ n : ℕ, n = 3 ∧ (Probability (draws.length = 3 ∧ draws.sum > 4 | sum_of_values <= 4) = 1 / 5)) :=
sorry

end draws_to_exceeding_sum_probability_l577_577400


namespace listed_price_percentage_above_cost_price_is_correct_l577_577412

def cost_price_per_article : ℝ := 1
def num_articles : ℕ := 45
def cost_price_total : ℝ := cost_price_per_article * num_articles
def selling_price_total : ℝ := (cost_price_per_article * 40)
def profit : ℝ := (0.20 * cost_price_total)
def actual_selling_price : ℝ := cost_price_total + profit
def discount_percentage : ℝ := 0.10
def discount : ℝ := discount_percentage * actual_selling_price
def listed_price : ℝ := actual_selling_price / (1 - discount_percentage)
def listed_price_per_article : ℝ := listed_price / num_articles
def percentage_above_cost_price : ℝ := ((listed_price_per_article - cost_price_per_article) / cost_price_per_article) * 100

theorem listed_price_percentage_above_cost_price_is_correct :
  percentage_above_cost_price = 211 := 
sorry

end listed_price_percentage_above_cost_price_is_correct_l577_577412


namespace solve_for_k_l577_577998

open Classical

theorem solve_for_k
  (k : ℚ)
  (OA : ℚ × ℚ := (k, 12))
  (OB : ℚ × ℚ := (4, 5))
  (OC : ℚ × ℚ := (-k, 10))
  (collinear : ∃ (c : ℚ), (4 - k, -7) = c • (-k - 4, 5)) :
  k = -2 / 3 :=
begin
  sorry
end

end solve_for_k_l577_577998


namespace mod_17_residue_l577_577795

theorem mod_17_residue : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := 
  by sorry

end mod_17_residue_l577_577795


namespace pair_integers_2xy_perfect_square_x2_y2_prime_l577_577891

theorem pair_integers_2xy_perfect_square_x2_y2_prime (x y : ℤ) :
  (Nat.perfectSquare (2 * x * y) ∧ Nat.prime (x^2 + y^2)) ↔ ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2)) := 
sorry

end pair_integers_2xy_perfect_square_x2_y2_prime_l577_577891


namespace average_annual_growth_rate_equation_l577_577236

variable (x : ℝ)
axiom seventh_to_ninth_reading_increase : (1 : ℝ) * (1 + x) * (1 + x) = 1.21

theorem average_annual_growth_rate_equation :
  100 * (1 + x) ^ 2 = 121 :=
by
  have h : (1 : ℝ) * (1 + x) * (1 + x) = 1.21 := seventh_to_ninth_reading_increase x
  sorry

end average_annual_growth_rate_equation_l577_577236


namespace arithmetic_sequence_properties_l577_577721

variable (a_n : ℕ → ℚ)
variable (a_3 a_11 : ℚ)

notation "a₃" => a_3
notation "a₁₁" => a_11

theorem arithmetic_sequence_properties :
  a₃ = a_n 3 → a₁₁ = a_n 11 → 
  (∃ (a₁ d : ℚ), a_n n = a₁ + (n - 1) * d ∧ a₁ = 0 ∧ d = 3/2) := sorry

end arithmetic_sequence_properties_l577_577721


namespace solution_set_l577_577187

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.sin x + Real.cos x + x^2

theorem solution_set (x : ℝ) :
  f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 ↔ (1 / Real.exp 1 < x ∧ x < Real.exp 1) :=
by {
  sorry
}

end solution_set_l577_577187


namespace matrix_identity_l577_577394

variable {A B : Matrix (Fin 3) (Fin 3) ℝ}

-- Assume all required inverses exist
def invertibleA [Invertible A] : Prop := true
def invertibleBI [Invertible B] : Prop := (B^-1 - A).Invertible

theorem matrix_identity :
  invertibleA → invertibleBI →
  A - (A^-1 + (B^-1 - A)^-1)^-1 = A * B * A :=
by
  intro h1 h2
  sorry

end matrix_identity_l577_577394


namespace staircase_perimeter_is_correct_l577_577252

def area_of_rectangle (width height : ℝ) : ℝ := width * height
def area_of_staircase (num_squares : ℝ) (side_length : ℝ) : ℝ := num_squares * (side_length ^ 2)
def perimeter_of_staircase (base_height : ℝ) (base_width : ℝ) (num_sides : ℕ) (side_length : ℝ) : ℝ :=
base_height + base_width + 3 + 7 + (num_sides * side_length)

theorem staircase_perimeter_is_correct :
  ∀ (base_height base_width tick_side_length : ℝ) (num_squares : ℝ) (area_staircase area_total : ℝ)
    (num_sides : ℕ),
  (base_height = 7.9091)
  → (base_width = 11)
  → (tick_side_length = 1)
  → (num_squares = 13) 
  → (area_staircase = area_of_staircase num_squares tick_side_length)
  → (area_total = 74)
  → (area(total) = area_of_rectangle base_width base_height - area_staircase)
  → perimeter_of_staircase(base_height, base_width, num_sides, tick_side_length) = 39 := 
begin 
  intros,
  sorry
end

end staircase_perimeter_is_correct_l577_577252


namespace correct_book_borrowing_sequence_l577_577805

def StorageEntry : Prop := sorry
def LocatingBook : Prop := sorry
def Reading : Prop := sorry
def Borrowing : Prop := sorry
def StorageExit : Prop := sorry
def Returning : Prop := sorry

def seq_A := [StorageEntry, Reading, Borrowing, LocatingBook, StorageExit, Returning]
def seq_B := [StorageEntry, LocatingBook, Reading, Borrowing, StorageExit, Returning]
def seq_C := [StorageEntry, Reading, Borrowing, LocatingBook, Returning, StorageExit]
def seq_D := [StorageEntry, LocatingBook, Reading, Borrowing, Returning, StorageExit]

axiom storage_entry_first : ∀ s, s.head = StorageEntry

axiom locating_book_before_reading_or_borrowing : ∀ s, LocatingBook ∈ s -> (Reading ∈ s ∨ Borrowing ∈ s -> s.indexOf LocatingBook < s.indexOf Reading ∨ s.indexOf LocatingBook < s.indexOf Borrowing)

axiom borrowing_after_reading : ∀ s, Reading ∈ s -> Borrowing ∈ s -> s.indexOf Reading < s.indexOf Borrowing

axiom storage_exit_after_borrowing : ∀ s, Borrowing ∈ s -> StorageExit ∈ s -> s.indexOf Borrowing < s.indexOf StorageExit

axiom returning_after_storage_exit : ∀ s, StorageExit ∈ s -> Returning ∈ s -> s.indexOf StorageExit < s.indexOf Returning

theorem correct_book_borrowing_sequence : seq_B = [StorageEntry, LocatingBook, Reading, Borrowing, StorageExit, Returning] :=
sorry

end correct_book_borrowing_sequence_l577_577805


namespace a_value_if_fx_geq_zero_fx_gt_xlnx_minus_sinx_l577_577973

noncomputable def f (a x : ℝ) : ℝ := a * exp x - x - a

-- Part (1)
theorem a_value_if_fx_geq_zero :
  (∀ x : ℝ, f a x ≥ 0) → a = 1 :=
sorry

-- Part (2)
theorem fx_gt_xlnx_minus_sinx (a : ℝ) :
  (a ≥ 1) → (∀ x : ℝ, x > 0 → f a x > x * log x - sin x) :=
sorry

end a_value_if_fx_geq_zero_fx_gt_xlnx_minus_sinx_l577_577973


namespace combined_percentage_basketball_l577_577872

theorem combined_percentage_basketball (N_students : ℕ) (S_students : ℕ) 
  (N_percent_basketball : ℚ) (S_percent_basketball : ℚ) :
  N_students = 1800 → S_students = 3000 →
  N_percent_basketball = 0.25 → S_percent_basketball = 0.35 →
  ((N_students * N_percent_basketball) + (S_students * S_percent_basketball)) / (N_students + S_students) * 100 = 31 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end combined_percentage_basketball_l577_577872


namespace find_area_l577_577326

variables (u v : ℝ^3) (A : ℝ)

def area_of_parallelgram (a b : ℝ^3) : ℝ := ‖a × b‖

theorem find_area (h : area_of_parallelgram u v = 10) : 
  area_of_parallelgram (3 • u + 4 • v) (2 • u - 6 • v) = 260 :=
by
  -- proof, to be filled later
  sorry

end find_area_l577_577326


namespace total_distance_covered_l577_577800

-- Definitions based on conditions
def distance_ran : ℝ := 40
def distance_walked : ℝ := (3 / 5) * distance_ran
def distance_jogged : ℝ := distance_walked / 5

-- Theorem stating the total distance covered
theorem total_distance_covered : distance_ran + distance_walked + distance_jogged = 64.8 := by
  -- You can place the formal proof steps here
  sorry

end total_distance_covered_l577_577800


namespace tangent_intersection_angle_l577_577743

theorem tangent_intersection_angle (P D E : Point) (c : Circle) (h_tangent_PD : tangent P D c) (h_tangent_PE : tangent P E c) (h_ratio : arc_length c D E / arc_length c E D = 3 / 5) : 
  ∠ D P E = 67.5° := by
  sorry

end tangent_intersection_angle_l577_577743


namespace john_paid_insurance_l577_577590

def vet_appointments : ℕ := 3
def cost_per_appointment : ℕ := 400
def insurance_coverage : ℕ := 80
def total_paid : ℕ := 660

noncomputable def insurance_paid : ℕ := 
  total_paid - cost_per_appointment - (2 * cost_per_appointment * (100 - insurance_coverage) / 100)

theorem john_paid_insurance :
  insurance_paid = 100 :=
by
  unfold insurance_paid
  simp
  sorry

end john_paid_insurance_l577_577590


namespace total_peaches_l577_577022

variable {n m : ℕ}

-- conditions
def equal_subgroups (n : ℕ) := (n % 3 = 0)

def condition_1 (n m : ℕ) := (m - 27) % n = 0 ∧ (m - 27) / n = 5

def condition_2 (n m : ℕ) : Prop := 
  ∃ x : ℕ, 0 < x ∧ x < 7 ∧ (m - x) % n = 0 ∧ ((m - x) / n = 7) 

-- theorem to be proved
theorem total_peaches (n m : ℕ) (h1 : equal_subgroups n) (h2 : condition_1 n m) (h3 : condition_2 n m) : m = 102 := 
sorry

end total_peaches_l577_577022


namespace tan_interval_increasing_l577_577921

theorem tan_interval_increasing (k : ℤ) : 
  ∀ x, x ∈ set.Ioo (k * π - π / 4) (k * π + 3 * π / 4) → strict_mono (λ x, tan (x - π / 4)) := 
by 
  sorry

end tan_interval_increasing_l577_577921


namespace symmetric_line_with_respect_to_point_symmetric_line_with_respect_to_line_l577_577984

-- Definition of line l
def l := λ x : ℝ, 2 * x + 1

-- Definition of point M(3, 2)
def M := (3, 2)

-- First proof statement: Symmetric line with respect to point M
theorem symmetric_line_with_respect_to_point : 
  ( ∃ b : ℝ, ∀ x : ℝ, x ≠ 6 → l x = 2 * x + b ) ∧ ( l 6 = 2 * 6 - 9 ) :=
sorry

-- Definition of line x - y - 2 = 0
def line := λ x y : ℝ, x - y - 2

-- Second proof statement: Symmetric line with respect to line l
theorem symmetric_line_with_respect_to_line :
  ( ∃ a c : ℝ, ∀ x y : ℝ, line x y = 0 → line a c = 0 ∧ ( 7 * x - y + 16 = 0 ) ) :=
sorry

end symmetric_line_with_respect_to_point_symmetric_line_with_respect_to_line_l577_577984


namespace smallest_y_l577_577027

theorem smallest_y (x : ℕ) (hx : x = 7 * 24 * 48) : ∃ y : ℕ, y = 588 ∧ (∃ n : ℕ, is_perfect_cube (x * y)) := by
  sorry

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

end smallest_y_l577_577027


namespace largest_of_20_consecutive_even_integers_l577_577713

theorem largest_of_20_consecutive_even_integers (x : ℕ) 
  (h : 20 * (x + 19) = 8000) : (x + 38) = 419 :=
  sorry

end largest_of_20_consecutive_even_integers_l577_577713


namespace max_naive_number_l577_577929

-- Define the digits and conditions for a naive number
variable (a b c d : ℕ)
variable (M : ℕ)
variable (h1 : b = c + 2)
variable (h2 : a = d + 6)
variable (h3 : M = 1000 * a + 100 * b + 10 * c + d)

-- Define P(M) and Q(M)
def P (a b c d : ℕ) : ℕ := 3 * (a + b) + c + d
def Q (a : ℕ) : ℕ := a - 5

-- Problem statement: Prove the maximum value of M satisfying the divisibility condition
theorem max_naive_number (div_cond : (P a b c d) % (Q a) = 0) (hq : Q a % 10 = 0) : M = 9313 := 
sorry

end max_naive_number_l577_577929


namespace wilson_fraction_l577_577810

theorem wilson_fraction (N : ℝ) (result : ℝ) (F : ℝ) (h1 : N = 8) (h2 : result = 16 / 3) (h3 : N - F * N = result) : F = 1 / 3 := 
by
  sorry

end wilson_fraction_l577_577810


namespace intersection_set_l577_577194

def M : Set ℤ := {1, 2, 3, 5, 7}
def N : Set ℤ := {x | ∃ k ∈ M, x = 2 * k - 1}
def I : Set ℤ := {1, 3, 5}

theorem intersection_set :
  M ∩ N = I :=
by sorry

end intersection_set_l577_577194


namespace sum_f_eq_l577_577283

def f (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.range n).sum (λ k, 1 / (k + 1 : ℚ))

theorem sum_f_eq (n : ℕ) (h : 1 < n) : 
  (Finset.range (n - 1)).sum (λ k, f (k + 1)) = n * (f n - 1) := 
by
  sorry

end sum_f_eq_l577_577283


namespace find_n_l577_577062

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n, a (n + 1) = a n + d

theorem find_n (a : ℕ → ℝ) (d : ℝ) (n : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : (finset.range (n + 1)).sum (λ i, a (2 * i + 1)) = 4)
  (h3 : (finset.range n).sum (λ i, a (2 * i + 2)) = 3) :
  n = 3 :=
sorry

end find_n_l577_577062


namespace range_of_a_l577_577952

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 → x > a) ∧ (∃ x : ℝ, x > a ∧ ¬(x^2 - 2 * x - 3 < 0)) → a ≤ -1 :=
by
  sorry

end range_of_a_l577_577952


namespace agreed_price_l577_577382

theorem agreed_price (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ∃ x : ℝ, (x = (b - a) / (a + b)) ∧ (P = (2 * a * b) / (a + b)) :=
by
  let x := (b - a) / (a + b)
  let P := a * (1 + x)
  have : P = (2 * a * b) / (a + b) := sorry
  use x
  split
  exact this

end agreed_price_l577_577382


namespace square_line_product_l577_577694

theorem square_line_product (b : ℝ) (h1 : y = 3) (h2 : y = 7) (h3 := x = 2) (h4 : x = b) : 
  (b = 6 ∨ b = -2) → (6 * -2 = -12) :=
by
  sorry

end square_line_product_l577_577694


namespace ceil_mul_eq_225_l577_577474

theorem ceil_mul_eq_225 {x : ℝ} (h₁ : ⌈x⌉ * x = 225) (h₂ : x > 0) : x = 15 :=
sorry

end ceil_mul_eq_225_l577_577474


namespace determine_c_l577_577342

theorem determine_c (c : ℝ) : 
  (∃ (x y : ℝ), 2 * x - y = c ∧ x = 5 ∧ y = 11) →
  c = -1 := 
by
  intro h,
  cases h with x hx,
  cases hx with y hy,
  cases hy with heq1 heq2,
  cases heq2 with hx5 hy11,
  rw [hx5, hy11] at heq1,
  linarith

end determine_c_l577_577342


namespace number_of_eight_digit_numbers_with_product_3375_l577_577922

-- Define the condition that an eight-digit number has a product of its digits equal to 3375
def has_product_of_digits_3375 (n : ℕ) : Prop :=
  ∃ digits : List ℕ, digits.length = 8 ∧ (digits.product = 3375) ∧ n = digits.toNat

-- Define the final statement to be proved
theorem number_of_eight_digit_numbers_with_product_3375 : 
  (∃ (S : Finset ℕ), S.card = 1680 ∧ ∀ n ∈ S, has_product_of_digits_3375 n) :=
sorry

end number_of_eight_digit_numbers_with_product_3375_l577_577922


namespace dorothy_age_relation_l577_577899

theorem dorothy_age_relation (D S : ℕ) (h1: S = 5) (h2: D + 5 = 2 * (S + 5)) : D = 3 * S :=
by
  -- implement the proof here
  sorry

end dorothy_age_relation_l577_577899


namespace sum_of_coordinates_B_l577_577300

theorem sum_of_coordinates_B 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hM_def : M = (-3, 2))
  (hA_def : A = (-8, 5))
  (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  B.1 + B.2 = 1 := 
sorry

end sum_of_coordinates_B_l577_577300


namespace ratio_of_volumes_of_octahedron_and_tetrahedron_l577_577564

noncomputable def vertices_of_tetrahedron : List (ℝ × ℝ × ℝ) := [
  (1, 1, 1),
  (1, -1, -1),
  (-1, 1, -1),
  (-1, -1, 1)
]

noncomputable def midpoints_of_edges (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := v1;
  let (x2, y2, z2) := v2;
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

noncomputable def vertices_of_octahedron : List (ℝ × ℝ × ℝ) :=
  List.map (λ i, midpoints_of_edges (vertices_of_tetrahedron.nthLe i.1 i.2)
                                     (vertices_of_tetrahedron.nthLe i.3 i.4))
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

theorem ratio_of_volumes_of_octahedron_and_tetrahedron
  (V_T V_O : ℝ) (h_T : V_T = (1 / 6) * (2 * sqrt 3)^3 * sqrt 2 := 8 * sqrt 6)
  (h_O : V_O = (1 / 3) * 2 * sqrt 2 * 1^2 := 4 * sqrt 2 / 3) :
  V_O / V_T = 1 / 18 :=
by
  rw [h_T, h_O]
  sorry

end ratio_of_volumes_of_octahedron_and_tetrahedron_l577_577564


namespace ratio_AR_AU_l577_577254

-- Define the conditions in the problem as variables and constraints
variables (A B C P Q U R : Type)
variables (AP PB AQ QC : ℝ)
variables (angle_bisector_AU : A -> U)
variables (intersect_AU_PQ_at_R : A -> U -> P -> Q -> R)

-- Assuming the given distances
def conditions (AP PB AQ QC : ℝ) : Prop :=
  AP = 2 ∧ PB = 6 ∧ AQ = 4 ∧ QC = 5

-- The statement to prove
theorem ratio_AR_AU (h : conditions AP PB AQ QC) : 
  (AR / AU) = 108 / 289 :=
sorry

end ratio_AR_AU_l577_577254


namespace geom_seq_c_seq_gen_formulas_l577_577709

noncomputable def a_seq : ℕ → ℤ
| 0     := 2
| (n+1) := sorry

noncomputable def b_seq : ℕ → ℤ
| 0     := 4
| (n+1) := sorry

noncomputable def c_seq (a_seq : ℕ → ℤ) : ℕ → ℤ :=
λ n, a_seq (n+1) - 2*a_seq n

theorem geom_seq_c_seq (a_seq : ℕ → ℤ) (b_seq : ℕ → ℤ) (h1 : a_seq 1 = 2) (h2 : b_seq 1 = 4) 
(h3 : ∀ n, a_seq (n+1) = -a_seq n - 2*b_seq n) (h4 : ∀ n, b_seq (n+1) = 6*a_seq n + 6*b_seq n) :
  ∀ n, ∃ r : ℤ, c_seq a_seq (n+1) = -14 * 3^n :=
sorry

theorem gen_formulas (a_seq : ℕ → ℤ) (b_seq : ℕ → ℤ) 
  (h1 : a_seq 1 = 2) (h2 : b_seq 1 = 4) 
  (h3 : ∀ n, a_seq (n+1) = -a_seq n - 2*b_seq n) 
  (h4 : ∀ n, b_seq (n+1) = 6*a_seq n + 6*b_seq n) :
  (∀ n, a_seq n = 2^(n+3) - 14 * 3^(n-1)) ∧ (∀ n, b_seq n = 28 * 3^(n-1) - 3 * 2^(n+2)) :=
sorry

end geom_seq_c_seq_gen_formulas_l577_577709


namespace normal_dist_mean_l577_577152

theorem normal_dist_mean (μ σ : ℝ) (h : ∀ X : ℝ, X ~ Normal μ σ → P(X ≤ 0) = P(X ≥ 2)) : μ = 1 :=
sorry

end normal_dist_mean_l577_577152


namespace land_to_water_time_ratio_l577_577325

-- Define the conditions
def distance_water : ℕ := 50
def distance_land : ℕ := 300
def speed_ratio : ℕ := 3

-- Define the Lean theorem statement
theorem land_to_water_time_ratio (x : ℝ) (hx : x > 0) : 
  (distance_land / (speed_ratio * x)) / (distance_water / x) = 2 := by
  sorry

end land_to_water_time_ratio_l577_577325


namespace sum_log2_floor_l577_577088

theorem sum_log2_floor (N : ℕ) (hN : 1 ≤ N ∧ N ≤ 2048) :
  ∑ N in finset.range 2048, nat.log N = 6157 := sorry

end sum_log2_floor_l577_577088


namespace exam_results_l577_577563

variable (E F G H : Prop)

def emma_statement : Prop := E → F
def frank_statement : Prop := F → ¬G
def george_statement : Prop := G → H
def exactly_two_asing : Prop :=
  (E ∧ F ∧ ¬G ∧ ¬H) ∨ (¬E ∧ F ∧ G ∧ ¬H) ∨
  (¬E ∧ ¬F ∧ G ∧ H) ∨ (¬E ∧ F ∧ ¬G ∧ H) ∨
  (E ∧ ¬F ∧ ¬G ∧ H)

theorem exam_results :
  (E ∧ F) ∨ (G ∧ H) :=
by {
  sorry
}

end exam_results_l577_577563


namespace fraction_to_decimal_l577_577472

theorem fraction_to_decimal (numer: ℚ) (denom: ℕ) (h_denom: denom = 2^5 * 5^1) :
  numer.den = 160 → numer.num = 59 → numer == 0.36875 :=
by
  intros
  sorry  

end fraction_to_decimal_l577_577472


namespace sum_of_areas_greater_factor_l577_577235

noncomputable def sum_areas_factor (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ) (r5 : ℝ) : ℝ := 
π * r1^2 + π * r2^2 + π * r3^2 + π * r4^2 + π * r5^2

theorem sum_of_areas_greater_factor
    (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ) (r5 : ℝ)
    (condition1 : r2 = 3 * r1)
    (condition2 : r3 = 3 * r2)
    (condition3 : r4 = 3 * r3)
    (condition4 : r5 = 3 * r4) :
    sum_areas_factor r1 r2 r3 r4 r5 = 7381 * (π * r1^2) :=
by
  sorry

end sum_of_areas_greater_factor_l577_577235


namespace area_triangle_OCD_l577_577750

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l577_577750


namespace pizza_toppings_l577_577057

theorem pizza_toppings :
  ∀ (F V T : ℕ), F = 4 → V = 16 → F * (1 + T) = V → T = 3 :=
by
  intros F V T hF hV h
  sorry

end pizza_toppings_l577_577057


namespace evaluate_propositions_l577_577985

-- Define the propositions
def p : Prop := 0 = 0
def q : Prop := ∀ (P : Prop), P → P

-- The proof problem statement
theorem evaluate_propositions :
  (p = False) → (q = True) →
  (p ∨ q = True) ∧ (p ∧ q = False) ∧ (¬p = True) :=
by
  intros
  split
  { -- Prove p ∨ q = True
    sorry }
  split
  { -- Prove p ∧ q = False
    sorry }
  { -- Prove ¬p = True
    sorry }

end evaluate_propositions_l577_577985


namespace find_f_of_neg5_l577_577282

def f (x : ℝ) : ℝ :=
if x <= -3 then 3*x + 7 else 6 - 3*x

theorem find_f_of_neg5 : f (-5) = -8 :=
by
  sorry

end find_f_of_neg5_l577_577282


namespace obtuse_subset_of_n_points_l577_577049

def is_obtuse_triangle (A B C : Point) : Prop :=
  ∠ABC > 90 ∨ ∠BCA > 90 ∨ ∠CAB > 90

def obtuse_set (points : set Point) : Prop :=
  ∀ A B C ∈ points, is_obtuse_triangle A B C

theorem obtuse_subset_of_n_points (n : ℕ) (points : set Point)
  (h₁ : points.card = n)
  (h₂ : ∀ A B C ∈ points, ¬collinear A B C) :
  ∃ (subset : set Point), subset.card ≥ nat.floor (real.sqrt n) ∧ obtuse_set subset :=
sorry

end obtuse_subset_of_n_points_l577_577049


namespace fg_at_2_eq_l577_577619

noncomputable def f (x : ℝ) : ℝ :=
  3 * real.sqrt x + 9 / real.sqrt x

noncomputable def g (x : ℝ) : ℝ :=
  3 * x ^ 2 - 3 * x - 4

theorem fg_at_2_eq : f (g 2) = 15 * real.sqrt 2 / 2 :=
by
  sorry

end fg_at_2_eq_l577_577619


namespace min_value_a_l577_577167

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, (3 * x - 5 * y) ≥ 0 → x > 0 → y > 0 → (1 - a) * x ^ 2 + 2 * x * y - a * y ^ 2 ≤ 0) ↔ a ≥ 55 / 34 := 
by 
  sorry

end min_value_a_l577_577167


namespace composite_polynomial_l577_577313

-- Definition that checks whether a number is composite
def is_composite (a : ℕ) : Prop := ∃ (b c : ℕ), b > 1 ∧ c > 1 ∧ a = b * c

-- Problem translated into a Lean 4 statement
theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  is_composite (n ^ (5 * n - 1) + n ^ (5 * n - 2) + n ^ (5 * n - 3) + n + 1) :=
sorry

end composite_polynomial_l577_577313


namespace sequence_not_arithmetic_nor_geometric_l577_577290

noncomputable def S : ℕ → ℕ
| 0       => 1  -- Assuming S_0 = 1 as a base case to make sense of ℕ to ℕ
| (n + 1) => 2 * (n + 1) * n + 1

def a (n : ℕ) : ℕ :=
if n = 0 then S 0 else S n - S (n - 1)

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
∃ d : ℕ, ∀ n : ℕ, n > 0 → a (n + 1) = a n + d

def is_geometric_seq (a : ℕ → ℕ) : Prop :=
∃ r : ℕ, ∀ n : ℕ, n > 0 → a (n + 1) = a n * r

theorem sequence_not_arithmetic_nor_geometric :
  ¬is_arithmetic_seq a ∧ ¬is_geometric_seq a :=
by
  sorry

end sequence_not_arithmetic_nor_geometric_l577_577290


namespace digit_in_thousandths_place_l577_577789

theorem digit_in_thousandths_place :
  (decimals (7 / 32)).nth 3 = some 8 := 
sorry

end digit_in_thousandths_place_l577_577789


namespace sqrt_sum_inequality_l577_577714

variable {a b c : ℝ}

theorem sqrt_sum_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  sqrt a + sqrt b + sqrt c ≥ a * b + b * c + c * a :=
sorry

end sqrt_sum_inequality_l577_577714


namespace smallest_H_polygon_diagonals_l577_577923

theorem smallest_H_polygon_diagonals :
  ∃ (H : ℕ), 
    (∀ (polygon : Type) [convex_polygon polygon H],
      ∀ (S : set (diagonal polygon)),
      ∑ (d in S) (length d) ≥ (∑ (d in (diagonals polygon \ S)) (length d)))
    ∧ H = 499000 :=
begin
  sorry
end

end smallest_H_polygon_diagonals_l577_577923


namespace even_function_has_zero_coefficient_l577_577220

theorem even_function_has_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (x^2 + a*x) = (x^2 + a*(-x))) → a = 0 :=
by
  intro h
  -- the proof part is omitted as requested
  sorry

end even_function_has_zero_coefficient_l577_577220


namespace continuous_stripe_probability_l577_577457

noncomputable def probability_continuous_stripe : ℚ :=
let total_combinations := 2^6 in
-- 3 pairs of parallel faces; for each pair, 4 favorable configurations.
let favorable_outcomes := 3 * 4 in
favorable_outcomes / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l577_577457


namespace Maria_needs_72_nuts_l577_577631

theorem Maria_needs_72_nuts
    (fraction_nuts : ℚ := 1 / 4)
    (percentage_chocolate_chips : ℚ := 40 / 100)
    (nuts_per_cookie : ℕ := 2)
    (total_cookies : ℕ := 60) :
    (total_cookies * ((fraction_nuts + (1 - fraction_nuts - percentage_chocolate_chips)) * nuts_per_cookie).toRat) = 72 :=
by
    sorry

end Maria_needs_72_nuts_l577_577631


namespace problem_inequality_l577_577302

theorem problem_inequality 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_le_a : a ≤ 1)
  (h_pos_b : 0 < b) (h_le_b : b ≤ 1)
  (h_pos_c : 0 < c) (h_le_c : c ≤ 1)
  (h_pos_d : 0 < d) (h_le_d : d ≤ 1) :
  (1 / (a^2 + b^2 + c^2 + d^2)) ≥ (1 / 4) + (1 - a) * (1 - b) * (1 - c) * (1 - d) :=
by
  sorry

end problem_inequality_l577_577302


namespace Annie_cookies_sum_l577_577070

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end Annie_cookies_sum_l577_577070


namespace total_volume_of_all_cubes_l577_577438

def volume_of_cube (s : ℕ) : ℕ := s^3

def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_of_cube s

theorem total_volume_of_all_cubes (V_Carl V_Kate : ℕ) : 
  (4 * (3^3)) + (6 * (1^3)) = 114 := by
  calc
  4 * (3^3) = 4 * 27 : by rfl
  ... = 108 : by norm_num
  6 * (1^3) = 6 * 1 : by rfl
  ... = 6 : by norm_num
  108 + 6 = 114 : by norm_num

end total_volume_of_all_cubes_l577_577438


namespace max_concentration_at_2_l577_577737

noncomputable def concentration (t : ℝ) : ℝ := (20 * t) / (t^2 + 4)

theorem max_concentration_at_2 : ∃ t : ℝ, 0 ≤ t ∧ ∀ s : ℝ, (0 ≤ s → concentration s ≤ concentration t) ∧ t = 2 := 
by 
  sorry -- we add sorry to skip the actual proof

end max_concentration_at_2_l577_577737


namespace sum_abs_diff_fixed_l577_577936

theorem sum_abs_diff_fixed {n : ℕ} (hn : 0 < n)
  (c : Fin 2n → ℝ) (h_distinct : ∀ i j : Fin 2n, i ≠ j → c i ≠ c j) :
  let a := Array.zip (Fin.range n) (Fin.range (n + 1, 2n)).toList in
  let b := (Fin.range (n + 1, 2n)).toList.reverse in
  (a, b) = List.unzip (List.sort (compare on c a) a ++ List.sort (compare on b) b) →
  (List.foldl (λ acc (p : Fin n × Fin (n + 1, 2n)), acc + (c p.1 - c p.2).abs) 0 (List.zip a b)) =
  (List.sum (List.map c (Fin.range (n + 1, 2n))) - List.sum (List.map c (Fin.range n))) :=
by
   intros
   sorry

end sum_abs_diff_fixed_l577_577936


namespace continuous_stripe_probability_l577_577458

noncomputable def probability_continuous_stripe_encircle_cube : ℚ :=
  let total_combinations : ℕ := 2^6
  let favor_combinations : ℕ := 3 * 4 -- 3 pairs of parallel faces, with 4 valid combinations each
  favor_combinations / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe_encircle_cube = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l577_577458


namespace product_of_b_values_is_neg_12_l577_577696

theorem product_of_b_values_is_neg_12 (b : ℝ) (y1 y2 x1 : ℝ) (h1 : y1 = 3) (h2 : y2 = 7) (h3 : x1 = 2) (h4 : y2 - y1 = 4) (h5 : ∃ b1 b2, b1 = x1 - 4 ∧ b2 = x1 + 4) : 
  (b1 * b2 = -12) :=
by
  sorry

end product_of_b_values_is_neg_12_l577_577696


namespace S_is_infinite_l577_577275

-- Define the set S and the condition that every point in S is the midpoint of two other points in S
variables {Point : Type*} [inhabited Point]

def is_midpoint (P A B : Point) [has_add Point] [has_scalar ℕ Point] : Prop :=
  P = (A + B) / 2

def is_midpoint_of_two_others (S : set Point) [inhabited S] : Prop :=
  ∀ P ∈ S, ∃ A B ∈ S, is_midpoint P A B

-- State the theorem that S is infinite under the given conditions
theorem S_is_infinite (S : set Point) [inhabited S] [has_add Point] [has_scalar ℕ Point]
  (h : is_midpoint_of_two_others S) : ∃ (inf : S.infinite), true :=
sorry

end S_is_infinite_l577_577275


namespace min_value_f_eq_zero_range_of_a_prod_leq_one_l577_577976

-- Problem 1
theorem min_value_f_eq_zero (a : ℝ) (h : a = 1) :
  ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 0 :=
by sorry

-- Problem 2
theorem range_of_a (a : ℝ) (h : ∃ x ∈ Icc (0 : ℝ) 2, f x = 0) :
  1 < a ∧ a ≤ (Real.exp 2 - 1) / 2 :=
by sorry

-- Problem 3
theorem prod_leq_one (a b : ℕ → ℝ) (n : ℕ) (h1 : ∀ k, 1 ≤ k ∧ k ≤ n → a k > 0 ∧ b k > 0)
  (h2 : ∑ k in Finset.range n, a k * b k ≤ ∑ k in Finset.range n, b k) :
  (∏ k in Finset.range n, a k ^ b k) ≤ 1 :=
by sorry

end min_value_f_eq_zero_range_of_a_prod_leq_one_l577_577976


namespace seq_solution_l577_577350

open Nat

noncomputable def a_n (n : ℕ) : ℚ := (2^n - 1) / (2^(n-1))

theorem seq_solution (n : ℕ) (hn : n > 0) :
  ∃ a : ℕ → ℚ, (∀ n > 0, 2 * n - a n = S n) ∧ (a n = (2^n - 1) / 2^(n-1)) := 
by
  sorry


end seq_solution_l577_577350


namespace union_of_sets_l577_577264

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (hA : A = {x, y}) (hB : B = {x + 1, 5}) (h_inter : A ∩ B = {2}) :
  A ∪ B = {1, 2, 5} :=
by
  sorry

end union_of_sets_l577_577264


namespace impossible_to_select_7_weights_with_unique_sums_l577_577942

theorem impossible_to_select_7_weights_with_unique_sums (S : Set ℕ) (h1 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 26) (h2 : S.card = 7) :
  ∃ (A B : Finset ℕ), A ≠ B ∧ A ⊆ S ∧ B ⊆ S ∧ 0 < A.card ∧ 0 < B.card ∧ A.sum = B.sum :=
by
  sorry

end impossible_to_select_7_weights_with_unique_sums_l577_577942


namespace largest_real_lambda_l577_577131

theorem largest_real_lambda (λ : ℝ) :
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ ab + λ * bd + cd) ↔ λ ≤ 3 / 2 :=
  sorry

end largest_real_lambda_l577_577131


namespace eight_applications_of_g_l577_577216

theorem eight_applications_of_g (g : ℝ → ℝ) (x : ℝ) (h : g x = -1 / x) : g (g (g (g (g (g (g (g x)))))))) = x := by
  sorry

example : eight_applications_of_g (λ x, -1 / x) 8 (by simp) = 8 := by
  sorry

end eight_applications_of_g_l577_577216


namespace false_statements_l577_577956

variable (a b c : ℝ)

theorem false_statements (a b c : ℝ) :
  ¬(a > b → a^2 > b^2) ∧ ¬((a^2 > b^2) → a > b) ∧ ¬(a > b → a * c^2 > b * c^2) ∧ ¬(a > b ↔ |a| > |b|) :=
by
  sorry

end false_statements_l577_577956


namespace find_principal_l577_577636

/-- Define all conditions on the Lean 4 statement --/
def rate : ℝ := 0.06  -- 6% as a decimal
def time : ℕ := 9  -- 9 years
def total_amount : ℝ := 8410  -- Rs. 8410

/-- Define the equation for simple interest --/
def simple_interest (P : ℝ) : ℝ :=
  P * rate * time / 100

/-- Define the equation for the total amount --/
def total_amount_formula (P : ℝ) : ℝ :=
  P + simple_interest P

/-- The theorem statement proving the principal borrowed amount --/
theorem find_principal :
  ∃ P : ℝ, total_amount_formula P = total_amount ∧ P = 5461 :=
by
  use 5461
  sorry

end find_principal_l577_577636


namespace calculate_f_value_l577_577885

def f (x y : ℚ) : ℚ := x - y * ⌈x / y⌉

theorem calculate_f_value :
  f (1/3) (-3/7) = -2/21 := by
  sorry

end calculate_f_value_l577_577885


namespace percent_of_students_in_range_l577_577841

theorem percent_of_students_in_range
    (f90_100 : ℕ) (f80_89 : ℕ) (f70_79 : ℕ) (f60_69 : ℕ) (f50_59 : ℕ) (f_below_50 : ℕ)
    (total_students : ℕ)
    (freq_90_100 : f90_100 = 3)
    (freq_80_89 : f80_89 = 7)
    (freq_70_79 : f70_79 = 10)
    (freq_60_69 : f60_69 = 5)
    (freq_50_59 : f50_59 = 6)
    (freq_below_50 : f_below_50 = 4)
    (total : total_students = f90_100 + f80_89 + f70_79 + f60_69 + f50_59 + f_below_50) :
    (6:ℝ) / 35 * 100 ≈ 17.14 :=
  by
  sorry

end percent_of_students_in_range_l577_577841


namespace Annie_total_cookies_l577_577068

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end Annie_total_cookies_l577_577068


namespace julia_song_download_l577_577597

theorem julia_song_download : 
  let internet_speed := 20 -- in MBps
  let half_hour_in_minutes := 30
  let size_per_song := 5 -- in MB
  (internet_speed * 60 * half_hour_in_minutes) / size_per_song = 7200 :=
by
  sorry

end julia_song_download_l577_577597


namespace probability_correct_l577_577576

noncomputable def probability_perpendicular_slopes : ℚ := 
  let set := {-3, -5 / 4, -1 / 2, 0, 1 / 3, 1, 4 / 5, 2}
  let total_pairs := (set.card * (set.card - 1)) / 2
  let successful_pairs := {( -3, 1 / 3), (-5 / 4, 4 / 5), (-1 / 2, 2)}
  let num_successful_pairs := successful_pairs.to_finset.card
  num_successful_pairs / total_pairs

theorem probability_correct :
  probability_perpendicular_slopes = 3 / 28 :=
by
  sorry

end probability_correct_l577_577576


namespace find_mean_of_normal_distribution_l577_577154

-- Define the necessary variables and assumptions
variables {μ σ : ℝ} (X : ℝ → ℝ)

-- Assume X follows a normal distribution N(μ, σ²)
axiom normal_dist : ∀ x, X x = normal μ σ

-- State the given condition
axiom condition : P (λ x, X x ≤ 0) = P (λ x, X x ≥ 2)

-- Formalize the proof problem
theorem find_mean_of_normal_distribution (h : condition) : μ = 1 :=
  sorry

end find_mean_of_normal_distribution_l577_577154


namespace interest_credited_cents_l577_577849

-- Definitions based on conditions
def principal_savings : ℝ := 308.08 -- Derived from P calculation.
def total_amount : ℝ := 310.45
def annual_interest_rate : ℝ := 0.03
def time_in_years : ℝ := 3 / 12

-- Lean statement proving that the interest credited is 37 cents
theorem interest_credited_cents (P A r t : ℝ) (hP : P = principal_savings) (hA : A = total_amount) (hr : r = annual_interest_rate) (ht : t = time_in_years) :
  let interest := A - P in
  let cents := interest.fract * 100 in
  cents = 37 := by sorry

end interest_credited_cents_l577_577849


namespace andy_basketball_team_members_l577_577431

/--
Andy had a platter of chocolate cookies. He ate 3 of them then gave his little brother 5 because he was behaving.
He then handed the platter to his basketball team of some members. The first player to arrive took 1, the second player to arrive took 3,
the third player took 5, and so on. When the last player took his share, the platter was empty. Andy had 72 chocolate cookies from the start.
Prove that there are 8 members in Andy's basketball team.
-/
theorem andy_basketball_team_members :
  let initial_cookies := 72
  let eaten_by_andy := 3
  let given_to_brother := 5
  let cookies_left := initial_cookies - (eaten_by_andy + given_to_brother)
  let sum_arithmetic (n : ℕ) := n / 2 * (1 + (1 + (n - 1) * 2))
  cookies_left = 64 -> sum_arithmetic 8 = cookies_left :=
by
  let initial_cookies := 72
  let eaten_by_andy := 3
  let given_to_brother := 5
  let cookies_left := initial_cookies - (eaten_by_andy + given_to_brother)
  let sum_arithmetic (n : ℕ) := n / 2 * (1 + (1 + (n - 1) * 2))
  assume h : cookies_left = 64
  show sum_arithmetic 8 = cookies_left
  from sorry

end andy_basketball_team_members_l577_577431


namespace sufficient_not_necessary_condition_not_necessary_condition_for_problem_statement_l577_577354

theorem sufficient_not_necessary_condition (a : ℝ) : (a > 4) → (a^2 > 16) :=
by
  intro h
  calc
    a > 4 : h
    a^2 > 4^2 : by exact (pow_lt_pow_of_lt_left h zero_lt_four (zero_le_one.trans zero_le_two))
    a^2 > 16 : by rwa pow_two

theorem not_necessary_condition_for (a : ℝ) : (a^2 > 16) → (a > 4) := 
by
  intro h
  have h₁ := (sqrt_lt h).mp zero_lt_sixteen
  sorry

theorem problem_statement (a : ℝ) : 
  ((a > 4) → (a^2 > 16)) ∧ ¬((a^2 > 16) → (a > 4)) :=
by
  exact ⟨sufficient_not_necessary_condition a, not_necessary_condition_for a⟩

end sufficient_not_necessary_condition_not_necessary_condition_for_problem_statement_l577_577354


namespace not_perfect_square_l577_577672

open Nat

theorem not_perfect_square (m n : ℕ) : ¬∃ k : ℕ, k^2 = 1 + 3^m + 3^n :=
by
  sorry

end not_perfect_square_l577_577672


namespace find_function_l577_577582

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1) :
  ∀ x : ℝ, f x = if x ≠ 0.5 then 1 / (0.5 - x) else 0.5 :=
by
  sorry

end find_function_l577_577582


namespace radius_of_sphere_l577_577854

theorem radius_of_sphere (r : ℝ) (x : ℝ) (d : ℝ) (hole_radius : ℝ) :
  hole_radius = 15 ∧ d = 10 ∧ (hole_radius^2 + x^2 = (x + d)^2) →
  r = real.sqrt (x^2 + hole_radius^2) → 
  r = 16.25 :=
by
  intro h1 h2
  cases h1 with h_radius h_depth
  cases h_depth with h_d h_eq
  sorry

end radius_of_sphere_l577_577854


namespace imaginary_part_of_conjugate_l577_577522

def z : ℂ := 10 / (2 - I)

theorem imaginary_part_of_conjugate :
  (complex.conj z).im = -2 :=
sorry

end imaginary_part_of_conjugate_l577_577522


namespace functional_identity_l577_577617

-- Define the set of non-negative integers
def S : Set ℕ := {n | n ≥ 0}

-- Define the function f with the required domain and codomain
def f (n : ℕ) : ℕ := n

-- The hypothesis: the functional equation satisfied by f
axiom functional_equation :
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

-- The theorem we want to prove
theorem functional_identity (n : ℕ) : f n = n :=
  sorry

end functional_identity_l577_577617


namespace sequence_periodic_l577_577108

-- Definitions based on the provided conditions
def sequence (a : ℕ → ℕ) (p q : ℕ) (hp : p.prime) (hq : q.prime) (hpq : p < q) : Prop :=
  a 1 = p ∧ a 2 = q ∧
  (∀ n, ∃ m, a (n+2) = if ∃ m, a n + a (n+1) = 2^m then 2 else min_fac (a n + a (n+1)))

-- Main statement to prove
theorem sequence_periodic (a : ℕ → ℕ) (p q : ℕ) (hp : p.prime) (hq : q.prime) (hpq : p < q) :
  sequence a p q hp hq hpq →
  ∃ M, ∀ n > M, a (n) = 2 ∧ a (n + 1) = 7 ∧ a (n + 2) = 3 ∧ a (n + 3) = 5 :=
sorry

end sequence_periodic_l577_577108


namespace thousandths_place_digit_of_7_div_32_l577_577793

noncomputable def decimal_thousandths_digit : ℚ := 7 / 32

theorem thousandths_place_digit_of_7_div_32 :
  (decimal_thousandths_digit * 1000) % 10 = 8 :=
sorry

end thousandths_place_digit_of_7_div_32_l577_577793


namespace sequence_square_sum_l577_577479

theorem sequence_square_sum (n : ℕ) (hn : n ≥ 7) :
  (∃ (a : Fin n → {-1, 1}), ∑ i in Finset.range n, a ⟨i, sorry⟩ * (i + 1)^2 = 0) ↔ (n % 4 = 0 ∨ n % 4 = 3) :=
sorry

end sequence_square_sum_l577_577479


namespace probability_same_tribe_quitters_l577_577705

theorem probability_same_tribe_quitters :
  let P := λ (n k : ℕ), (n.choose k) → ℚ
  (totalWays : ℕ := (Nat.choose 16 2))
  (specificTribeWays : ℕ := (Nat.choose 4 2))
  (sameTribeWays : ℕ := 4 * specificTribeWays) in
  totalWays ≠ 0 →
  (sameTribeWays : ℚ) / (totalWays : ℚ) = (1 : ℚ) / 5 :=
by
  intros
  exact sorry

end probability_same_tribe_quitters_l577_577705


namespace transformed_line_theorem_l577_577693

theorem transformed_line_theorem (k b : ℝ) (h₁ : k = 1) (h₂ : b = 1) (x : ℝ) :
  (k * x + b > 0) ↔ (x > -1) :=
by sorry

end transformed_line_theorem_l577_577693


namespace median_attendance_is_six_l577_577357

-- Definitions based on the conditions
def total_students : ℕ := 20

def attendance_distribution : List (ℕ × ℕ) :=
  [(4, 1), (5, 5), (6, 7), (7, 4), (8, 3)]

-- Proof problem statement
theorem median_attendance_is_six 
  (total : ℕ)
  (distribution : List (ℕ × ℕ))
  (h_total : total = 20)
  (h_distribution : distribution = [(4, 1), (5, 5), (6, 7), (7, 4), (8, 3)]) :
  median_of_distribution distribution = 6 :=
sorry

end median_attendance_is_six_l577_577357


namespace total_sum_alternating_sums_l577_577144

noncomputable def alternating_sum (s : Finset ℕ) : ℤ :=
  s.1.sort (≥) |> List.enumFrom 1 |> List.map (λ ⟨i, n⟩, if i % 2 = 1 then n else -n) |> List.sum

def sum_of_alternating_sums (n : ℕ) : ℤ :=
  (Finset.range n).powerset.filter (λ s, ¬s.isEmpty)
  |> Finset.fold (λ acc s, acc + alternating_sum s) 0

theorem total_sum_alternating_sums (m : ℕ) (h : m = 7) :
  sum_of_alternating_sums m = 448 := by
  sorry

end total_sum_alternating_sums_l577_577144


namespace S1_div_S2_eq_one_l577_577478

-- Defining S1 and S2
def S1 : ℝ := ∑ k in finset.range 1 2020, (-1)^(k+1) * (1/3)^k
def S2 : ℝ := ∑ k in finset.range 1 2020, (-1)^(k+1) * (1/3)^k

-- Prove the main statement
theorem S1_div_S2_eq_one : S1 / S2 = 1 := 
  by sorry

end S1_div_S2_eq_one_l577_577478


namespace angle_EMN_eq_70_l577_577440

/- Definitions of the angles and conditions -/
def angleD : ℝ := 50
def angleE : ℝ := 70
def angleF : ℝ := 60

/- Main theorem statement to prove that ∠EMN is 70° -/
theorem angle_EMN_eq_70 :
  let Ω : ℝ → Prop := λ Ω, 
    (Ω is_incirlce_of_Δ(angleD, angleE, angleF) ∧ Ω is_circumcircle_of_ΔEMN) ->
    (∀ M N P : Point, M on_line AB ∧ N on_line BC ∧ P on_line CA ∧
    ∠D = angleD ∧ ∠E = angleE ∧ ∠F = angleF),
  angle(EMN) = 70 :=
sorry

end angle_EMN_eq_70_l577_577440


namespace water_bottles_needed_l577_577462

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l577_577462


namespace area_of_triangle_bounded_by_lines_l577_577772

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l577_577772


namespace quadratic_no_real_roots_m_eq_1_quadratic_root_intervals_l577_577192

-- Problem 1: Proving no real roots for m = 1
theorem quadratic_no_real_roots_m_eq_1 :
  ∀ (x : ℝ), (x^2 + 2 * x + 3 = 0) → false :=
by {
  intros x h,
  sorry  -- Proof would go here
}

-- Problem 2: Prove the range of m given the conditions on the roots
theorem quadratic_root_intervals (m : ℝ) :
  (let f (x : ℝ) := x^2 + 2 * m * x + 2 * m + 1 in
  (f (-1) * f 0 < 0 ∧ f 1 * f 2 < 0)) → -5/6 < m ∧ m < -1/2 :=
by {
  intros h,
  sorry  -- Proof would go here
}

end quadratic_no_real_roots_m_eq_1_quadratic_root_intervals_l577_577192


namespace determine_constants_l577_577111

def vec_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def vec_smul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

theorem determine_constants :
  ∃ c1 c2 : ℝ,
    vec_add (vec_smul c1 ⟨1, 4⟩) (vec_smul c2 ⟨-3, 6⟩) = ⟨2, 1⟩ ∧
    c1 = 5 / 6 ∧ c2 = -7 / 18 :=
begin
  sorry
end

end determine_constants_l577_577111


namespace range_of_f_l577_577974

def f (x : ℤ) : ℤ := x^2 + 2 * x

theorem range_of_f : 
  let xs := {1, 2, -3}
  ∃ r, r = {3, 8} ∧ ∀ x ∈ xs, f x ∈ r :=
by
  sorry

end range_of_f_l577_577974


namespace find_theta_l577_577494

theorem find_theta (θ : ℝ) (h1 : sin (π + θ) = - (sqrt 3) * cos (2 * π - θ)) (h2 : abs θ < π / 2) : θ = π / 3 :=
begin
  sorry,
end

end find_theta_l577_577494


namespace probability_of_six_and_queen_l577_577369

variable {deck : Finset (ℕ × String)}
variable (sixes : Finset (ℕ × String))
variable (queens : Finset (ℕ × String))

def standard_deck : Finset (ℕ × String) := sorry

-- Condition: the deck contains 52 cards (13 hearts, 13 clubs, 13 spades, 13 diamonds)
-- and it has 4 sixes and 4 Queens.
axiom h_deck_size : standard_deck.card = 52
axiom h_sixes : ∀ c ∈ standard_deck, c.1 = 6 → c ∈ sixes
axiom h_queens : ∀ c ∈ standard_deck, c.1 = 12 → c ∈ queens

-- Define the probability function for dealing cards
noncomputable def prob_first_six_and_second_queen : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_six_and_queen :
  prob_first_six_and_second_queen = 4 / 663 :=
by
  sorry

end probability_of_six_and_queen_l577_577369


namespace probability_x_y_less_than_3_l577_577030

theorem probability_x_y_less_than_3 :
  let A := 6 * 2
  let triangle_area := (1 / 2) * 3 * 2
  let P := triangle_area / A
  P = 1 / 4 := by sorry

end probability_x_y_less_than_3_l577_577030


namespace sum_floor_log2_to_2048_l577_577080

theorem sum_floor_log2_to_2048 :
  (Finset.sum (Finset.range 2048.succ) (λ N : ℕ, Int.toNat ⌊Real.logb 2 (N : ℝ)⌋) = 14349) :=
by
  sorry

end sum_floor_log2_to_2048_l577_577080


namespace count_numbers_without_digit_1_l577_577207

theorem count_numbers_without_digit_1 :
  let count := Finset.filter (λ n : ℕ, 
        n < 1001 ∧ (n.toString.toList.all (λ d, d ≠ '1'))
      ) (Finset.range 1001)
  in count.card = 728 :=
by
  sorry

end count_numbers_without_digit_1_l577_577207


namespace find_m_range_of_g_l577_577979

-- Define the power function h(x)
def h (m : ℝ) (x : ℝ) : ℝ := (m^2 - 5 * m + 1) * x^(m + 1)

-- Define the function g(x)
def g (x : ℝ) : ℝ := x + sqrt (1 - 2 * x)

-- Proving the value of m
theorem find_m : ∃ m : ℝ, (∀ x : ℝ, h m x = x) ∧ h m x = x ∧ 0 = 0 := sorry

-- Proving the range of the function g(x)
theorem range_of_g : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1/2) → 1/2 ≤ g x ∧ g x ≤ 1 := sorry

end find_m_range_of_g_l577_577979


namespace thousandths_place_digit_of_7_div_32_l577_577792

noncomputable def decimal_thousandths_digit : ℚ := 7 / 32

theorem thousandths_place_digit_of_7_div_32 :
  (decimal_thousandths_digit * 1000) % 10 = 8 :=
sorry

end thousandths_place_digit_of_7_div_32_l577_577792


namespace area_ratio_proof_l577_577329

noncomputable def area_ratio (AB CD m : ℝ) (isosceles_trapezoid : Type)
  (is_parallel : isosceles_trapezoid → Prop)
  (height : isosceles_trapezoid → ℝ)
  (area_trapezoid : isosceles_trapezoid → ℝ)
  (area_triangle : isosceles_trapezoid → ℝ) : Prop :=
  AB = 3 * m ∧ CD = 2 * m ∧ height isosceles_trapezoid = m ∧ is_parallel isosceles_trapezoid →
  area_trapezoid isosceles_trapezoid = 25 * area_triangle isosceles_trapezoid

-- Define the isosceles_trapezoid, is_parallel, height, area_trapezoid, and area_triangle as per problem's requirements.
variable {isosceles_trapezoid : Type}

axiom is_parallel (t : isosceles_trapezoid) : Prop

axiom height (t : isosceles_trapezoid) : ℝ

axiom area_trapezoid (t : isosceles_trapezoid) : ℝ

axiom area_triangle (t : isosceles_trapezoid) : ℝ

theorem area_ratio_proof (AB CD m : ℝ) (t : isosceles_trapezoid)
  (ht : height t = m) (hyp_parallel : is_parallel t)
  (hyp_ab : AB = 3 * m) (hyp_cd : CD = 2 * m) :
  area_trapezoid t = 25 * area_triangle t :=
begin
  sorry
end

end area_ratio_proof_l577_577329


namespace bus_length_proof_l577_577009

noncomputable def length_of_bus (speed_kmph : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) :=
  let speed_mps := (speed_kmph * 1000) / 3600 in
  (speed_mps * time_s) - bridge_length_m

theorem bus_length_proof : length_of_bus 50 18 150 = 100 :=
by
  unfold length_of_bus
  norm_num   -- this simplifies the arithmetic
  sorry      -- replace this with a detailed proof if needed

end bus_length_proof_l577_577009


namespace jellybean_count_l577_577732

variable (initial_count removed1 added_back removed2 : ℕ)

theorem jellybean_count :
  initial_count = 37 →
  removed1 = 15 →
  added_back = 5 →
  removed2 = 4 →
  initial_count - removed1 + added_back - removed2 = 23 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jellybean_count_l577_577732


namespace mother_age_is_correct_l577_577828

variable (D M : ℕ)

theorem mother_age_is_correct:
  (D + 3 = 26) → (M - 5 = 2 * (D - 5)) → M = 41 := by
  intros h1 h2
  sorry

end mother_age_is_correct_l577_577828


namespace center_of_symmetry_l577_577553

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + (Real.pi / 6))

theorem center_of_symmetry : 
  (∃ k : ℤ, (\exists (x : ℝ), 2 * x + Real.pi / 6 = k * Real.pi ∧ x = 5 * Real.pi / 12))
:=
begin
  sorry
end

end center_of_symmetry_l577_577553


namespace find_g_of_f_and_sum_l577_577539

theorem find_g_of_f_and_sum (f g : ℝ[X]) (h₁ : f + g = -3 * X^2 + 2 * X - 4)
    (h₂ : f = X^4 - X^3 + 3 * X - 1) :
    g = -X^4 + X^3 - 3 * X^2 - X - 3 :=
sorry

end find_g_of_f_and_sum_l577_577539


namespace sum_floor_log2_l577_577076

theorem sum_floor_log2 :
  (∑ N in Finset.range 2048, Int.floor (Real.log N / Real.log 2)) = 20445 :=
by
  sorry

end sum_floor_log2_l577_577076


namespace projectile_height_reaches_49_l577_577338

theorem projectile_height_reaches_49 (t : ℝ) :
  (∃ t : ℝ, 49 = -20 * t^2 + 100 * t) → t = 0.7 :=
by
  sorry

end projectile_height_reaches_49_l577_577338


namespace gcd_of_lcm_ratio_l577_577223

theorem gcd_of_lcm_ratio {A B k : ℕ} (h1 : Nat.lcm A B = 180) (h2 : A * 5 = B * 2) :
  Nat.gcd A B = 18 := 
by
  sorry

end gcd_of_lcm_ratio_l577_577223


namespace total_promotional_items_l577_577419

def num_calendars : ℕ := 300
def num_date_books : ℕ := 200

theorem total_promotional_items : num_calendars + num_date_books = 500 := by
  sorry

end total_promotional_items_l577_577419


namespace tangent_line_eq_l577_577301

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- Define the value of m for the point M(1, m)
def m : ℝ := f 1

/-- The equation of the tangent line to the function f at point (1, m) is y = 3x - 2. -/
theorem tangent_line_eq : 
  let M := (1, m) in
  ∀ x y : ℝ, 
    y = 3 * x - 2 → 
    y - 1 = 3 * (x - 1) :=
by 
  -- proof goes here
  sorry

end tangent_line_eq_l577_577301


namespace two_friends_visit_45_times_in_365_days_l577_577888

theorem two_friends_visit_45_times_in_365_days :
  (∀ n, n % 4 = 0 → visits_alex n) →
  (∀ n, n % 6 = 0 → visits_bella n) →
  (∀ n, n % 8 = 0 → visits_casey n) →
  (count_2_visits_a_b_c 365) = 45 :=
by sorry

-- Definitions required:
def visits_alex (n : ℕ) : Prop := n % 4 = 0
def visits_bella (n : ℕ) : Prop := n % 6 = 0
def visits_casey (n : ℕ) : Prop := n % 8 = 0

def count_2_visits_a_b_c (days: ℕ): ℕ := 
  (finset.filter 
     (λ n, 
        (∃ d ∈ finset.range days, 
           (visits_alex d ∧ visits_bella d ∧ ¬ visits_casey d) ∨
           (visits_alex d ∧ visits_casey d ∧ ¬ visits_bella d) ∨
           (visits_bella d ∧ visits_casey d ∧ ¬ visits_alex d)
        )
     )
     (finset.range days)).card

#print two_friends_visit_45_times_in_365_days

end two_friends_visit_45_times_in_365_days_l577_577888


namespace volume_of_regular_triangular_pyramid_l577_577047

theorem volume_of_regular_triangular_pyramid (base_edge side_edge : ℝ)
  (h_base_edge : base_edge = 6) (h_side_edge : side_edge = Real.sqrt 15) :
  let base_area := (Real.sqrt 3 / 4) * base_edge^2,
      pyramid_height := Real.sqrt (side_edge^2 - (base_edge / 2)^2),
      volume := (1 / 3) * base_area * pyramid_height in
  volume = 9 := by
  sorry

end volume_of_regular_triangular_pyramid_l577_577047


namespace correct_system_of_equations_l577_577833

theorem correct_system_of_equations :
  ∃ (x y : ℕ), 
    x + y = 38 
    ∧ 26 * x + 20 * y = 952 := 
by
  sorry

end correct_system_of_equations_l577_577833


namespace all_primes_in_a_seq_l577_577499

noncomputable def a_seq (k : ℕ) : ℕ → ℕ
| 0     := 1
| (n+1) := Nat.find (λ x, x > a_seq k n ∧ x = 1 + ∑ i in Finset.range (n + 1), ∑ j in Finset.range x, (if j % (a_seq k i) == 0 then 1 else 0))

theorem all_primes_in_a_seq (k : ℕ) (hk : k ≥ 2) : ∀ p : ℕ, Nat.Prime p → ∃ n, a_seq k n = p :=
sorry

end all_primes_in_a_seq_l577_577499


namespace longest_side_is_48_l577_577711

noncomputable def longest_side_of_triangle (a b c : ℝ) (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : ℝ :=
  a

theorem longest_side_is_48 {a b c : ℝ} (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : 
  longest_side_of_triangle a b c ha hb hc hp = 48 :=
sorry

end longest_side_is_48_l577_577711


namespace median_passes_fixed_point_l577_577285

noncomputable def midpoint (A B: Point) : Point := sorry  -- Define midpoint function appropriately

variables (O A B : Point) (circle_O : Circle O AB) (M N P Q R S E F : Point)
  (h1 : diameter circle_O A B)
  (h2 : M ∈ circle_O)
  (h3 : bisects_angle M N A B)
  (h4 : external_bisector M P Q N A B)
  (h5 : intersects_AM circle_O M Q R)
  (h6 : intersects_BM circle_O M P S)
  (hA : AE_perpendicular MN)
  (hF : AF_perpendicular MN)

theorem median_passes_fixed_point :
  let O_fixed := midpoint E F in
  median_line (Triangle.mk N R S) N ∩ O_fixed :=
sorry

end median_passes_fixed_point_l577_577285


namespace trigonometric_identity_l577_577155

theorem trigonometric_identity (θ : ℝ) (h : cos (θ + π / 2) = -1 / 2) :
  (cos (θ + π) / (sin (π / 2 - θ) * (cos (3 * π - θ) - 1)) +
   cos (θ - 2 * π) / (cos (-θ) * cos (π - θ) + sin (θ + 5 * π / 2))) = 8 :=
by
  sorry

end trigonometric_identity_l577_577155


namespace comparison_M_N_l577_577492

def M (x : ℝ) : ℝ := x^2 - 3*x + 7
def N (x : ℝ) : ℝ := -x^2 + x + 1

theorem comparison_M_N (x : ℝ) : M x > N x :=
  by sorry

end comparison_M_N_l577_577492


namespace geometric_seq_problem_l577_577962

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r

theorem geometric_seq_problem (h_geom : geometric_sequence a) 
  (h_cond : a 8 * a 9 * a 10 = -a 13 ^ 2 ∧ -a 13 ^ 2 = -1000) :
  a 10 * a 12 = 100 * Real.sqrt 10 :=
by
  sorry

end geometric_seq_problem_l577_577962


namespace amusement_park_total_cost_l577_577844

def rides_cost_ferris_wheel : ℕ := 5 * 6
def rides_cost_roller_coaster : ℕ := 7 * 4
def rides_cost_merry_go_round : ℕ := 3 * 10
def rides_cost_bumper_cars : ℕ := 4 * 7
def rides_cost_haunted_house : ℕ := 6 * 5
def rides_cost_log_flume : ℕ := 8 * 3

def snacks_cost_ice_cream : ℕ := 8 * 4
def snacks_cost_hot_dog : ℕ := 6 * 5
def snacks_cost_pizza : ℕ := 4 * 3
def snacks_cost_pretzel : ℕ := 5 * 2
def snacks_cost_cotton_candy : ℕ := 3 * 6
def snacks_cost_soda : ℕ := 2 * 7

def total_rides_cost : ℕ := 
  rides_cost_ferris_wheel + 
  rides_cost_roller_coaster + 
  rides_cost_merry_go_round + 
  rides_cost_bumper_cars + 
  rides_cost_haunted_house + 
  rides_cost_log_flume

def total_snacks_cost : ℕ := 
  snacks_cost_ice_cream + 
  snacks_cost_hot_dog + 
  snacks_cost_pizza + 
  snacks_cost_pretzel + 
  snacks_cost_cotton_candy + 
  snacks_cost_soda

def total_cost : ℕ :=
  total_rides_cost + total_snacks_cost

theorem amusement_park_total_cost :
  total_cost = 286 :=
by
  unfold total_cost total_rides_cost total_snacks_cost
  unfold rides_cost_ferris_wheel 
         rides_cost_roller_coaster 
         rides_cost_merry_go_round 
         rides_cost_bumper_cars 
         rides_cost_haunted_house 
         rides_cost_log_flume
         snacks_cost_ice_cream 
         snacks_cost_hot_dog 
         snacks_cost_pizza 
         snacks_cost_pretzel 
         snacks_cost_cotton_candy 
         snacks_cost_soda
  sorry

end amusement_park_total_cost_l577_577844


namespace harriet_trip_l577_577389

noncomputable def time_from_A_to_B (D : ℝ) := D / 110
noncomputable def time_from_B_to_A (D : ℝ) := D / 140

theorem harriet_trip :
  ∃ (T_AB : ℝ), (time_from_A_to_B 30.8 = T_AB) ∧ (T_AB * 60 = 16.8) :=
by
  let D := 30.8
  let total_time := (time_from_A_to_B D) + (time_from_B_to_A D)
  have h_total_time : total_time = 5 := sorry
  have h_time_AB := time_from_A_to_B D
  have h_time_AB_in_minutes : h_time_AB * 60 = 16.8 := sorry
  use h_time_AB
  exact ⟨by rfl, h_time_AB_in_minutes⟩

end harriet_trip_l577_577389


namespace animals_total_sleep_in_one_week_l577_577410

-- Define the conditions
def cougar_sleep_per_night := 4 -- Cougar sleeps 4 hours per night
def zebra_extra_sleep := 2 -- Zebra sleeps 2 hours more than cougar

-- Calculate the sleep duration for the zebra
def zebra_sleep_per_night := cougar_sleep_per_night + zebra_extra_sleep

-- Total sleep duration per week
def week_nights := 7

-- Total weekly sleep durations
def cougar_weekly_sleep := cougar_sleep_per_night * week_nights
def zebra_weekly_sleep := zebra_sleep_per_night * week_nights

-- Total sleep time for both animals in one week
def total_weekly_sleep := cougar_weekly_sleep + zebra_weekly_sleep

-- The target theorem
theorem animals_total_sleep_in_one_week : total_weekly_sleep = 70 := by
  sorry

end animals_total_sleep_in_one_week_l577_577410


namespace series_sum_correct_l577_577454

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem series_sum_correct :
  geometric_series_sum (1 / 2) (-1 / 3) 6 = 91 / 243 :=
by
  -- Proof goes here
  sorry

end series_sum_correct_l577_577454


namespace quadrilateral_perpendicular_l577_577577

theorem quadrilateral_perpendicular
    {A B C D : Type}
    [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
    (AC BD AD BC AB CD : A)

    (h1 : AC ⊥ BD)
    (h2 : AD ⊥ BC) :
    AB ⊥ CD :=
begin
  sorry
end

end quadrilateral_perpendicular_l577_577577


namespace number_of_expressible_integers_l577_577205

def floor_sum (x : ℝ) : ℤ :=
  Int.floor (3 * x) + Int.floor (6 * x) + Int.floor (9 * x) + Int.floor (12 * x)

theorem number_of_expressible_integers :
  (card {n : ℕ | n ≤ 2000 ∧ ∃ x : ℝ, floor_sum x = n}) = 990 :=
sorry

end number_of_expressible_integers_l577_577205


namespace digit_in_tens_place_l577_577784

theorem digit_in_tens_place (n : ℕ) (cycle : List ℕ) (h_cycle : cycle = [16, 96, 76, 56]) (hk : n % 4 = 3) :
  (6 ^ n % 100) / 10 % 10 = 7 := by
  sorry

end digit_in_tens_place_l577_577784


namespace min_value_arith_seq_l577_577963

theorem min_value_arith_seq (a : ℕ → ℝ) 
  (arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h : a 2015 + a 2017 = ∫ x in 0..2, sqrt (4 - x^2)) : 
  a 2016 * (a 2014 + a 2018) = π^2 / 2 :=
by 
  sorry

end min_value_arith_seq_l577_577963


namespace primes_solution_l577_577914

theorem primes_solution (p : ℕ) (n : ℕ) (h_prime : Prime p) (h_nat : 0 < n) : 
  (p^2 + n^2 = 3 * p * n + 1) ↔ (p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8) := sorry

end primes_solution_l577_577914


namespace relation_among_a_b_c_l577_577094

noncomputable def f : ℝ → ℝ := sorry
def a : ℝ := f (Real.log 3 / Real.log 0.5)
def b : ℝ := f (Real.log 5 / Real.log 2)
def c : ℝ := f (Real.log 2 / Real.log 3)

-- Conditions given in the problem
axiom even_function (x : ℝ) : f x = f (-x)
axiom deriv_exists_not_zero (x : ℝ) (hx : x ≠ 0) : ∃ f', f = (λ x, f' x) 
axiom x_deriv_pos (x : ℝ) (hx : x ≠ 0) : x * deriv f x > 0

theorem relation_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relation_among_a_b_c_l577_577094


namespace negation_of_proposition_l577_577525

theorem negation_of_proposition :
  (¬ ∃ m : ℝ, 1 / (m^2 + m - 6) > 0) ↔ (∀ m : ℝ, (1 / (m^2 + m - 6) < 0) ∨ (m^2 + m - 6 = 0)) :=
by
  sorry

end negation_of_proposition_l577_577525


namespace num_integer_points_on_parabola_l577_577480

open Real

def focus : ℝ × ℝ := (1, 1)
def point1 : ℝ × ℝ := (5, 4)
def point2 : ℝ × ℝ := (-3, -2)
def constraint (x y : ℝ) : Prop := |3 * x + 4 * y| ≤ 900
noncomputable def parabola (x y : ℝ) : Prop := -- This should be the parabola equation derived
  sorry -- Replace with the actual equation derived

theorem num_integer_points_on_parabola :
  (finset.filter (λ (p : ℤ × ℤ), parabola p.1 p.2 ∧ constraint p.1 p.2)
    (finset.Icc (-(900 / 4) : ℤ) (900 / 3 : ℤ) (-900 / 4 : ℤ) (900 / 3 : ℤ))).card = 73 := by
  sorry

end num_integer_points_on_parabola_l577_577480


namespace count_values_of_a_l577_577145

noncomputable def vertex_x (a : ℝ) : ℝ := -a / 2
noncomputable def vertex_y (a : ℝ) : ℝ := (vertex_x a)^2 + a * (vertex_x a)

theorem count_values_of_a :
  {a : ℝ | let x := vertex_x a; let y := vertex_y a in y = a * x + a}.card = 2 :=
by
  sorry

end count_values_of_a_l577_577145


namespace exists_function_f_l577_577004

noncomputable def my_function : ℝ → ℝ := sorry

theorem exists_function_f : 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f(f(x)) = -x := 
begin
  use my_function,
  intros x,
  sorry
end

end exists_function_f_l577_577004


namespace area_triangle_OCD_l577_577749

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l577_577749


namespace air_conditioner_days_used_l577_577061

noncomputable def running_8_hours_consumption : ℝ := 7.2
noncomputable def running_8_hours : ℝ := 8
noncomputable def daily_hours : ℝ := 6
noncomputable def total_consumption : ℝ := 27

theorem air_conditioner_days_used :
  let hourly_consumption := running_8_hours_consumption / running_8_hours in
  let daily_consumption := hourly_consumption * daily_hours in
  let days_used := total_consumption / daily_consumption in
  days_used = 5 := sorry

end air_conditioner_days_used_l577_577061


namespace value_of_y_l577_577227

theorem value_of_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) : y = 1 / 2 :=
by
  sorry

end value_of_y_l577_577227


namespace calculate_k_l577_577346

theorem calculate_k :
  ∃ n k : ℕ, 0 ≤ k ∧ k < 2013 ∧ 
  (let pairs := ((Finset.range 2015).sum (λ x, x + 1))^3 
    (pairs % 2013 = k)) ∧ k = 27 :=
by
  sorry

end calculate_k_l577_577346


namespace Annie_total_cookies_l577_577069

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end Annie_total_cookies_l577_577069


namespace green_socks_count_l577_577589

theorem green_socks_count: 
  ∀ (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) (red_socks : ℕ) (green_socks : ℕ),
  total_socks = 900 →
  white_socks = total_socks / 3 →
  blue_socks = total_socks / 4 →
  red_socks = total_socks / 5 →
  green_socks = total_socks - (white_socks + blue_socks + red_socks) →
  green_socks = 195 :=
by
  intros total_socks white_socks blue_socks red_socks green_socks
  sorry

end green_socks_count_l577_577589


namespace polynomial_real_root_l577_577530

def monic_polynomial_of_degree_2016 (P : Polynomial ℝ) : Prop :=
  P.degree = 2016 ∧ P.leading_coeff = 1

theorem polynomial_real_root 
  (P Q : Polynomial ℝ) 
  (hP : monic_polynomial_of_degree_2016 P)
  (hQ : monic_polynomial_of_degree_2016 Q)
  (hPQ : ¬ ∃ x : ℝ, P.eval x = Q.eval x) :
  ∃ x : ℝ, P.eval x = Q.eval (x + 1) :=
begin
  sorry
end

end polynomial_real_root_l577_577530


namespace triangle_area_bounded_by_lines_l577_577754

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l577_577754


namespace percentage_of_students_studying_languages_l577_577435

theorem percentage_of_students_studying_languages (
  p_eng_lang : ℝ := 0.35    -- 35% of students that study a language study English
  p_non_eng_all : ℝ := 0.13 -- 13% of all the university students study a language other than English
) : (∃ p_lang_all : ℝ, p_lang_all = 0.20) := 
sorry

end percentage_of_students_studying_languages_l577_577435


namespace convex_parallelogram_faces_1992_l577_577058

theorem convex_parallelogram_faces_1992 (n : ℕ) (h : n > 0) : (n * (n - 1) ≠ 1992) := 
by
  sorry

end convex_parallelogram_faces_1992_l577_577058


namespace total_water_bottles_needed_l577_577467

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l577_577467


namespace right_triangle_from_square_diagonal_l577_577502

-- Define a square
structure Square (A B C D : Type) :=
  (AB : Line A B)
  (BC : Line B C)
  (CD : Line C D)
  (DA : Line D A)
  (AC : Diagonal A C)
  (BD : Diagonal B D)
  (is_square : AB.len = BC.len ∧ BC.len = CD.len ∧ CD.len = DA.len 
               ∧ AB.is_perpendicular BC ∧ BC.is_perpendicular CD 
               ∧ CD.is_perpendicular DA ∧ DA.is_perpendicular AB)

-- Define the point E on the diagonal AC
structure PointOnDiagonal (A C E : Type) (AC : Diagonal A C) :=
  (point_on : E ∈ AC.points)

-- Define the sides of the potential right triangle
structure RightTriangle (A E C F : Type) :=
  (right_angle_at : E ∧ AE ⊥ EC)

theorem right_triangle_from_square_diagonal (A B C D E F : Type)
  [Square A B C D]
  [PointOnDiagonal A C E (Square.AC A B C D)]
  (EB_perpendicular_BF : E × B = F × B ∧ BE ⊥ BF)
  (BF_len_EQ_BE_len : BE = BF) :
  RightTriangle A E C F := by
  sorry -- proof skipped

end right_triangle_from_square_diagonal_l577_577502


namespace total_kittens_proof_l577_577734

-- Define the initial number of kittens Tim had
def total_kittens_initial (kittens_given_jessica kittens_given_sara kittens_left : ℕ) : ℕ :=
  kittens_given_jessica + kittens_given_sara + kittens_left

-- Prove the total number of kittens initially is 18
theorem total_kittens_proof : total_kittens_initial 3 6 9 = 18 :=
by {
  sorry,
}

end total_kittens_proof_l577_577734


namespace matrix_vector_dot_product_l577_577443

-- Define the given matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 2], ![-1, 5]]

-- Define the given vector v
def v : Vector ℝ 2 := ![4, -2]

-- Define the given vector w
def w : Vector ℝ 2 := ![-1, 3]

-- Define the problem statement as a Lean theorem
theorem matrix_vector_dot_product :
  let Av := A.mulVec v in
  (Av ⬝ w : ℝ) = -50 :=
by
  sorry

end matrix_vector_dot_product_l577_577443


namespace absent_children_l577_577640

theorem absent_children (total_children bananas_per_child_if_present bananas_per_child_if_absent children_present absent_children : ℕ) 
  (H1 : total_children = 740)
  (H2 : bananas_per_child_if_present = 2)
  (H3 : bananas_per_child_if_absent = 4)
  (H4 : children_present * bananas_per_child_if_absent = total_children * bananas_per_child_if_present)
  (H5 : children_present = total_children - absent_children) : 
  absent_children = 370 :=
sorry

end absent_children_l577_577640


namespace sum_of_104th_parenthesis_is_correct_l577_577659

def b (n : ℕ) : ℕ := 2 * n + 1

def sumOf104thParenthesis : ℕ :=
  let cycleCount := 104 / 4
  let numbersBefore104 := 260
  let firstNumIndex := numbersBefore104 + 1
  let firstNum := b firstNumIndex
  let secondNum := b (firstNumIndex + 1)
  let thirdNum := b (firstNumIndex + 2)
  let fourthNum := b (firstNumIndex + 3)
  firstNum + secondNum + thirdNum + fourthNum

theorem sum_of_104th_parenthesis_is_correct : sumOf104thParenthesis = 2104 :=
  by
    sorry

end sum_of_104th_parenthesis_is_correct_l577_577659


namespace triangle_area_bounded_by_lines_l577_577757

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l577_577757


namespace jeanne_additional_tickets_l577_577587

-- Define the costs
def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def jeanne_tickets : ℕ := 5

-- Calculate the total cost
def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

-- Define the proof problem
theorem jeanne_additional_tickets : total_cost - jeanne_tickets = 8 :=
by sorry

end jeanne_additional_tickets_l577_577587


namespace smallest_lambda_l577_577601

-- Define the sets and functions as per the problem conditions
def Am (m : ℕ) : set (ℚ × ℚ) := 
  { p | p.1 ≠ 0 ∧ p.2 ≠ 0 ∧ (p.1 * p.2 / m) ∈ ℤ }

def fm_segment (m : ℕ) (p1 p2 : ℚ × ℚ) : ℕ := 
  -- Assume a function definition that counts points on segment p1p2 in Am
  sorry

-- Define the smallest lambda and conditions under the proof
theorem smallest_lambda (l : ℚ × ℚ) (β : ℚ × ℚ → ℝ) : 
  ∃ (beta : ℝ), 
    (∀ (M N: ℚ × ℚ), 
      f_2016 (M N) ≤ (2015/6 : ℝ) * f_2015 (M N) + beta) :=
begin
  sorry
end

end smallest_lambda_l577_577601


namespace area_triangle_OCD_l577_577748

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l577_577748


namespace sum_sequence_2023_l577_577174

def sequence (n : ℕ) : ℕ
| 0 := 1
| 1 := 1
| 2 := 2
| k+3 := if k = 0 then 4 else 1

lemma sequence_recursive (n : ℕ) (h : n ≥ 0) : 
  sequence n * sequence (n + 1) * sequence (n + 2) * sequence (n + 3) = 
  sequence n + sequence (n + 1) + sequence (n + 2) + sequence (n + 3) :=
sorry

theorem sum_sequence_2023 : ∑ i in finset.range 2023, sequence i = 4044 :=
sorry

end sum_sequence_2023_l577_577174


namespace tan_neg4095_eq_one_l577_577878

theorem tan_neg4095_eq_one : Real.tan (Real.pi / 180 * -4095) = 1 := by
  sorry

end tan_neg4095_eq_one_l577_577878


namespace sin_neg_nine_pi_div_two_l577_577723

theorem sin_neg_nine_pi_div_two : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end sin_neg_nine_pi_div_two_l577_577723


namespace range_of_a_l577_577972

def f : ℝ → ℝ
| x := if x ≥ 0 then real.exp x + x^2 else real.exp (-x) + x^2

theorem range_of_a (a : ℝ) (h : f (-a) + f (a) ≤ 2 * f 1) : -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l577_577972


namespace slope_of_l_l577_577516

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def is_on_parabola (p : ℝ × ℝ) : Prop := (p.2 ^ 2 = 4 * p.1)

def line_through_focus (k : ℝ) : ℝ → ℝ × ℝ := λ x, (x, k * (x - 1))

def intersects_parabola (k : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ is_on_parabola A ∧ is_on_parabola B ∧ 
                ∃ F: ℝ × ℝ, F = parabola_focus ∧ 
                (∃ t : ℝ, A.2 = t * k ∧ B.2 = -4 * t * k)

theorem slope_of_l :
  ∀ (k : ℝ), (∃ A B : ℝ × ℝ, A ≠ B ∧ is_on_parabola A ∧ is_on_parabola B ∧
                ∃ F : ℝ × ℝ, F = parabola_focus ∧
                (∃ t : ℝ, A.2 = t * k ∧ B.2 = -4 * t * k)) →
               (k = 4/3 ∨ k = -4/3) :=
by
  sorry

end slope_of_l_l577_577516


namespace count_Omega_functions_l577_577498

def is_Omega_function (f : ℝ → ℝ) : Prop := 
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

def f1 (x : ℝ) : ℝ := Real.cos x
def f2 (x : ℝ) : ℝ := 2 ^ x
def f3 (x : ℝ) : ℝ := x * abs x
def f4 (x : ℝ) : ℝ := Real.log (x^2 + 1)

theorem count_Omega_functions :
  let functions := [f1, f2, f3, f4]
  let Omega_functions := functions.filter is_Omega_function
  Omega_functions.length = 2 := 
by
  sorry

end count_Omega_functions_l577_577498


namespace smallest_possible_N_l577_577051

theorem smallest_possible_N (l m n : ℕ) (h_visible : (l - 1) * (m - 1) * (n - 1) = 252) : l * m * n = 392 :=
sorry

end smallest_possible_N_l577_577051


namespace log_expression_value_l577_577482

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log_expression_value :
  (log2 8 * (log2 2 / log2 8)) + log2 4 = 3 :=
by
  sorry

end log_expression_value_l577_577482


namespace find_two_digit_number_l577_577034

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l577_577034


namespace find_valid_pairs_l577_577122

-- Defining the conditions and target answer set.
def valid_pairs : List (Nat × Nat) := [(2,2), (3,3), (1,2), (2,1), (2,3), (3,2)]

theorem find_valid_pairs (a b : Nat) :
  (∃ n m : Int, (a^2 + b = n * (b^2 - a)) ∧ (b^2 + a = m * (a^2 - b)))
  ↔ (a, b) ∈ valid_pairs :=
by sorry

end find_valid_pairs_l577_577122


namespace no_real_solutions_of_quadratic_eq_l577_577663

theorem no_real_solutions_of_quadratic_eq
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  ∀ x : ℝ, ¬ (b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 = 0) :=
by
  sorry

end no_real_solutions_of_quadratic_eq_l577_577663


namespace johns_long_distance_bill_is_correct_l577_577939

def monthly_fee : ℝ := 5
def cost_per_minute : ℝ := 0.25
def minutes_billed : ℝ := 28.08
def cost_of_minutes : ℝ := minutes_billed * cost_per_minute
def total_bill : ℝ := monthly_fee + cost_of_minutes

theorem johns_long_distance_bill_is_correct :
  total_bill = 12.02 :=
by
  unfold monthly_fee cost_per_minute minutes_billed cost_of_minutes total_bill
  -- Uncomment the sorry statement and complete the proof here.
  sorry

end johns_long_distance_bill_is_correct_l577_577939


namespace find_two_digit_number_l577_577031

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l577_577031


namespace probability_gather_information_both_classes_l577_577707

theorem probability_gather_information_both_classes (n g j b: ℕ)
    (total_enrolled : n = 30) 
    (taking_german : g = 22) 
    (taking_japanese : j = 19)
    (taking_both : b = g + j - n) :
    let total_pairs := (nat.choose n 2) in
    let german_only := g - b in
    let japanese_only := j - b in
    let pairs_german_only := (nat.choose german_only 2) in
    let pairs_japanese_only := (nat.choose japanese_only 2) in
    let pairs_single_class := pairs_german_only + pairs_japanese_only in
    let probability_single_class := pairs_single_class / total_pairs in
    let probability_both_classes := 1 - probability_single_class in
    probability_both_classes = (16 : ℚ) / 21 :=
by
  sorry

end probability_gather_information_both_classes_l577_577707


namespace f_log_inv3_12_eq_neg_one_third_l577_577512

noncomputable def f : ℝ → ℝ :=
  fun x => if h : 0 ≤ x ∧ x < 1 then 3 ^ x - 1 else sorry

theorem f_log_inv3_12_eq_neg_one_third :
  (∀ x : ℝ, f(x) + f(-x) = 0) ∧
  (∀ x : ℝ, f(x - 1) = f(x + 1)) ∧
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) → f(x) = 3 ^ x - 1) →
  f(log 3 12 / log 3 (1 / 3)) = -1 / 3 :=
by
  sorry

end f_log_inv3_12_eq_neg_one_third_l577_577512


namespace circle_center_and_radius_l577_577965

theorem circle_center_and_radius :
  ∀ (x y : ℝ), (x + 1)^2 + y^2 = 2 →
  (∃ a b r, a = -1 ∧ b = 0 ∧ r = sqrt 2 ∧ (x - a)^2 + (y - b)^2 = r^2) :=
by {
  sorry
}

end circle_center_and_radius_l577_577965


namespace thousandths_place_digit_of_7_div_32_l577_577791

noncomputable def decimal_thousandths_digit : ℚ := 7 / 32

theorem thousandths_place_digit_of_7_div_32 :
  (decimal_thousandths_digit * 1000) % 10 = 8 :=
sorry

end thousandths_place_digit_of_7_div_32_l577_577791


namespace jane_runs_more_than_daniel_l577_577573

def street_width : ℝ := 30
def block_side_length : ℝ := 500
def radius_increase : ℝ := street_width / 2
def daniel_lap_length : ℝ := 4 * block_side_length
def jane_lap_length : ℝ := 4 * (block_side_length + 2 * radius_increase)

theorem jane_runs_more_than_daniel :
  jane_lap_length - daniel_lap_length = 120 := by
  sorry

end jane_runs_more_than_daniel_l577_577573


namespace roots_cubic_identity_l577_577217

theorem roots_cubic_identity (p q r s : ℝ) (h1 : r + s = p) (h2 : r * s = -q) (h3 : ∀ x : ℝ, x^2 - p*x - q = 0 → (x = r ∨ x = s)) :
  r^3 + s^3 = p^3 + 3*p*q := by
  sorry

end roots_cubic_identity_l577_577217


namespace wedge_volume_correct_l577_577017

-- Define the given conditions
def radius : ℝ := 5
def height : ℝ := 10
def percentage : ℝ := 0.3

-- Calculate the volume of the cylinder
def volume_cylinder : ℝ := π * radius^2 * height

-- Calculate the volume of the wedge
def volume_wedge : ℝ := percentage * volume_cylinder

-- The expected volume of the wedge
def expected_volume : ℝ := 235.5

-- Prove that the volume of the wedge is 235.5 cubic centimeters
theorem wedge_volume_correct : volume_wedge = expected_volume := by
  -- Sorry means we omit the actual proof implementation here
  sorry

end wedge_volume_correct_l577_577017


namespace PQRS_rectangle_l577_577603

variables {A B C D E P Q R S : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables [metric_space P] [metric_space Q] [metric_space R] [metric_space S]

-- Definitions of points and midpoints in the quadrilateral ABCD
def is_midpoint (E A B : Type*) [metric_space E] [metric_space A] [metric_space B] : Prop := 
  dist E A = dist E B

-- Midpoints for diagonals
def E_midpt_of_AC_BD (E A C B D : Type*) [metric_space E] [metric_space A] [metric_space C] [metric_space B] [metric_space D] : Prop := 
  is_midpoint E A C ∧ is_midpoint E B D ∧ dist (E : Type*) = dist E B

-- Perpendicular diagonals
def diagonals_perpendicular (A B E : Type*) [metric_space A] [metric_space B] [metric_space E] : Prop :=
  dist A B ^ 2 + dist E A ^ 2 = dist A B ^ 2 

-- Define Midpoints of each side
def quadrilateral_midpoints (P Q R S A B C D : Type*) [metric_space P] [metric_space Q] [metric_space R] [metric_space S] [metric_space A] [metric_space B] [metric_space C] [metric_space D] : 
  Prop :=
    is_midpoint P A B ∧ is_midpoint Q B C ∧ is_midpoint R C D ∧ is_midpoint S D A

-- Prove PQRS is a rectangle
theorem PQRS_rectangle (ABCD : Type*)
  [metric_space ABCD] [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (h1 : E_midpt_of_AC_BD E A C B D)
  (h2 : diagonals_perpendicular A B E C) : 
  quadrilateral_midpoints P Q R S A B C D → 
  is_rectangle P Q R S :=
sorry

end PQRS_rectangle_l577_577603


namespace equivalence_gcd_prime_power_l577_577143

theorem equivalence_gcd_prime_power (a b n : ℕ) :
  (∀ m, 0 < m ∧ m < n → Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔ 
  (∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k) :=
by
  sorry

end equivalence_gcd_prime_power_l577_577143


namespace symmetric_point_line_eq_l577_577715

theorem symmetric_point_line_eq (A B : ℝ × ℝ) (l : ℝ → ℝ) (x1 y1 x2 y2 : ℝ)
  (hA : A = (4, 5))
  (hB : B = (-2, 7))
  (hSymmetric : ∀ x y, B = (2 * l x - A.1, 2 * l y - A.2)) :
  ∀ x y, l x = 3 * x - 5 ∧ l y = 3 * y + 6 :=
by
  sorry

end symmetric_point_line_eq_l577_577715


namespace wrapping_paper_area_correct_l577_577834

-- Define the conditions
variables (s h : ℝ) -- s: side length of the base, h: height of the box

-- Box's properties
def box : Prop := s > 0 ∧ h > 0

-- Dimensions of the wrapping paper
def wrapping_paper_side : ℝ := 2 * s

-- Area of the wrapping paper
def wrapping_paper_area : ℝ := wrapping_paper_side ^ 2

-- The theorem statement, which encapsulates the proof problem
theorem wrapping_paper_area_correct (s h : ℝ) (hs : s > 0) (hh : h > 0) :
  wrapping_paper_area s h = 4 * s^2 :=
by
  unfold wrapping_paper_area wrapping_paper_side
  sorry

end wrapping_paper_area_correct_l577_577834


namespace actual_price_of_food_l577_577016

theorem actual_price_of_food (P : ℝ) : 
  (∃ P, (let total_with_discount := (0.6 * P + (0.4 * P - 0.06 * P)),
            price_with_tax := total_with_discount * 1.10,
            final_price := price_with_tax * 1.20
         in final_price = 198.60)
         → P ≈ 160.06) :=
by
  sorry -- No proof required

end actual_price_of_food_l577_577016


namespace valid_sequences_count_l577_577095

noncomputable def isValidSequence (seq : List ℕ) : Prop :=
  (∀ i, 2 ≤ i → i < seq.length → (seq[i] + 1 ∈ seq[:i] ∨ seq[i] - 1 ∈ seq[:i])) ∧
  (∀ i, 2 ≤ i → i < seq.length → (seq[i] ≠ seq[i-2]) ∧ (seq[i] ≠ seq[i+2]))

theorem valid_sequences_count : 
  (number_of_valid_sequences : ℕ) = 512 :=
sorry

end valid_sequences_count_l577_577095


namespace molecular_weight_proof_l577_577796

/-- Atomic weights in atomic mass units (amu) --/
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_P : ℝ := 30.97

/-- Number of atoms in the compound --/
def num_Al : ℝ := 2
def num_O : ℝ := 4
def num_H : ℝ := 6
def num_N : ℝ := 3
def num_P : ℝ := 1

/-- calculating the molecular weight --/
def molecular_weight : ℝ := 
  (num_Al * atomic_weight_Al) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_N * atomic_weight_N) +
  (num_P * atomic_weight_P)

-- The proof statement
theorem molecular_weight_proof : molecular_weight = 197.02 := 
by
  sorry

end molecular_weight_proof_l577_577796


namespace perfect_cubes_count_l577_577895

theorem perfect_cubes_count (x y : ℕ) (h1: x = 2^9 + 1!) (h2: y = 2^{17} + 1!) : 
  ∃ c : ℕ, c = 42 ∧ 
  let lower_bound := (nat.cbrt x).succ in
  let upper_bound := (nat.cbrt y) in
  ∀ n, lower_bound ≤ n ∧ n ≤ upper_bound → n^3 ≥ x ∧ n^3 ≤ y :=
sorry

end perfect_cubes_count_l577_577895


namespace problem_proof_l577_577980

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^x - 1 else log 3 x + 1

theorem problem_proof : f (f (sqrt 3 / 9)) = sqrt 2 / 2 - 1 := by
  sorry

end problem_proof_l577_577980


namespace probability_in_rectangular_region_l577_577299

noncomputable def rectangular_region : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3009 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3010}

def probability_x_gt_6y {x y : ℝ} (h : x > 6 * y) : Prop :=
  (x, y) ∈ rectangular_region

theorem probability_in_rectangular_region :
  ∃ p : ℚ,
    (∀ x y : ℝ, (x, y) ∈ rectangular_region → x > 6 * y → 
      p = (502835 / 604006 : ℚ)) sorry

end probability_in_rectangular_region_l577_577299


namespace distance_to_x_axis_l577_577722

theorem distance_to_x_axis (x y : ℝ) :
  (x^2 / 9 - y^2 / 16 = 1) →
  (x^2 + y^2 = 25) →
  abs y = 16 / 5 :=
by
  -- Conditions: x^2 / 9 - y^2 / 16 = 1, x^2 + y^2 = 25
  -- Conclusion: abs y = 16 / 5 
  intro h1 h2
  sorry

end distance_to_x_axis_l577_577722


namespace find_a_plus_b_l577_577483

theorem find_a_plus_b (a b : ℚ) (y : ℚ) (x : ℚ) :
  (y = a + b / x) →
  (2 = a + b / (-2 : ℚ)) →
  (3 = a + b / (-6 : ℚ)) →
  a + b = 13 / 2 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_plus_b_l577_577483


namespace smallest_12_digit_natural_number_with_all_digits_divisible_by_36_l577_577821

theorem smallest_12_digit_natural_number_with_all_digits_divisible_by_36 :
  ∃ (n : ℕ), 
    (nat.digits 10 n).length = 12 ∧
    ∀ d ∈ list.range 10, d ∈ nat.digits 10 n ∧
    n % 36 = 0 ∧
    n = 100023457896 :=
by
  -- Proof omitted
  sorry

end smallest_12_digit_natural_number_with_all_digits_divisible_by_36_l577_577821


namespace sum_red_numbers_is_zero_l577_577666

variable (n : ℕ)
variable (colors : Fin n.succ → Bool) -- true is red, false is blue
variable (values : Fin n.succ → ℝ)

-- Condition 1: Each red number is equal to the sum of its neighboring numbers.
def red_condition : Prop :=
  ∀ i, colors i → values i = values ((i : Fin n.succ) - 1) + values ((i : Fin n.succ) + 1)

-- Condition 2: Each blue number is equal to half the sum of its neighboring numbers.
def blue_condition : Prop :=
  ∀ i, ¬colors i → 2 * values i = values ((i : Fin n.succ) - 1) + values ((i : Fin n.succ) + 1)

-- Prove that the sum of the red numbers is zero.
theorem sum_red_numbers_is_zero
  (h_red : red_condition n colors values)
  (h_blue : blue_condition n colors values) :
  (∑ i, if colors i then values i else 0) = 0 :=
sorry

end sum_red_numbers_is_zero_l577_577666


namespace red_marbles_count_l577_577398

variable (n : ℕ)

-- Conditions
def ratio_green_yellow_red := (3 * n, 4 * n, 2 * n)
def not_red_marbles := 3 * n + 4 * n = 63

-- Goal
theorem red_marbles_count (hn : not_red_marbles n) : 2 * n = 18 :=
by
  sorry

end red_marbles_count_l577_577398


namespace find_p_q_r_s_a_b_c_l577_577008

-- defining the setup and conditions
def probability_shot := 0.4
def total_shots := 10
def made_shots := 4
def max_ratio := 0.4

-- statement of the problem in Lean
theorem find_p_q_r_s_a_b_c :
  (∃ p q r s a b c : ℕ, 
    (p, q, r, s).all prime
    ∧ (p + q + r + s) = 10 
    ∧ (a + b + c) = 20 
    ∧ (p + q + r + s) * (a + b + c) = 200) :=
sorry

end find_p_q_r_s_a_b_c_l577_577008


namespace person_speed_l577_577383

-- Definitions
def distance_m : ℝ := 600  -- distance in meters
def time_min : ℝ := 2      -- time in minutes

-- Convert distance to kilometers
def distance_km : ℝ := distance_m / 1000

-- Convert time to hours
def time_hr : ℝ := time_min / 60

-- Calculate speed in km/hr
def speed_km_per_hr : ℝ := distance_km / time_hr

-- Statement to prove the speed is 18 km/hr
theorem person_speed : speed_km_per_hr = 18 := by
sorry

end person_speed_l577_577383


namespace platform_length_l577_577056

theorem platform_length 
  (train_length : ℝ) (train_speed_kmph : ℝ) (time_s : ℝ) (platform_length : ℝ)
  (H1 : train_length = 360) 
  (H2 : train_speed_kmph = 45) 
  (H3 : time_s = 40)
  (H4 : platform_length = (train_speed_kmph * 1000 / 3600 * time_s) - train_length ) :
  platform_length = 140 :=
by {
 sorry
}

end platform_length_l577_577056


namespace sufficient_not_necessary_condition_l577_577286

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

theorem sufficient_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) :=
by
  sorry

end sufficient_not_necessary_condition_l577_577286


namespace stick_length_l577_577584

theorem stick_length (x : ℕ) (h1 : 2 * x + (2 * x - 1) = 14) : x = 3 := sorry

end stick_length_l577_577584


namespace cooking_competition_probability_l577_577561

theorem cooking_competition_probability :
  let n := 8
  let f := 4
  let k := 2
  let total_pairs := Nat.choose n k
  let female_pairs := Nat.choose f k
  (female_pairs / total_pairs : ℚ) = 3 / 14 := by
  -- just the statement
  sorry

end cooking_competition_probability_l577_577561


namespace number_of_knights_is_two_number_of_knights_is_two_l577_577651

-- Define types for inhabitants (knights or liars)
inductive Inhabitant
| knight
| liar

open Inhabitant

-- Define the statements given by the inhabitants
def statements (i : ℕ) : String :=
  match i with
  | 1 => "One knight"
  | 2 => "Two knights"
  | 3 => "Three knights"
  | 4 => "Don't believe them, they are all liars"
  | 5 => "You're the liar!"
  | _ => ""

-- Define the truth-telling property of knights
def tells_truth (i : ℕ) (s : String) : Prop :=
  match i with
  | 1 => (s = "One knight") ↔ (count_knights = 1)
  | 2 => (s = "Two knights") ↔ (count_knights = 2)
  | 3 => (s = "Three knights") ↔ (count_knights = 3)
  | 4 => (s = "Don't believe them, they are all liars") ↔ (inhabitant 1 = liar ∧ inhabitant 2 = liar ∧ inhabitant 3 = liar)
  | 5 => (s = "You're the liar!") ↔ (inhabitant 4 = liar)
  | _ => false

-- Define the main theorem to be proven
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

-- Noncomputable definition to avoid computational problems
noncomputable def count_knights : ℕ :=
  sorry

-- Noncomputable to define each inhabitant's type
noncomputable def inhabitant (i : ℕ) : Inhabitant :=
  match i with
  | 1 => liar
  | 2 => knight
  | 3 => liar
  | 4 => liar
  | 5 => knight
  | _ => liar -- Default to liar, although there are only 5 inhabitants

-- Additional properties used
def is_knight (i : ℕ) : Prop := inhabitant i = knight
def is_liar (i : ℕ) : Prop := inhabitant i = liar

-- Count the number of knights
noncomputable def count_knights : ℕ :=
  List.length (List.filter (λ i => is_knight i) [1, 2, 3, 4, 5])

-- Main theorem that states there are exactly two knights according to the statements
theorem number_of_knights_is_two : count_knights = 2 :=
by
  sorry

end number_of_knights_is_two_number_of_knights_is_two_l577_577651


namespace averaging_associative_averaging_commutative_addition_distributes_over_averaging_avg_properties_l577_577638

def avg (x y : ℝ) : ℝ := (x + y) / 2

theorem averaging_associative (x y z : ℝ) : avg (avg x y) z = avg x (avg y z) := 
by sorry

theorem averaging_commutative (x y : ℝ) : avg x y = avg y x := 
by sorry

theorem addition_distributes_over_averaging (a b c : ℝ) : a + avg b c = avg (a + b) (a + c) := 
by sorry

theorem avg_properties (x y z : ℝ) :
  (¬ averaging_associative x y z) ∧ 
  averaging_commutative x y ∧ 
  addition_distributes_over_averaging x y z 
:=
by sorry

end averaging_associative_averaging_commutative_addition_distributes_over_averaging_avg_properties_l577_577638


namespace two_digit_number_is_9_l577_577037

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l577_577037


namespace inscribed_quad_area_bounds_l577_577280

variable {α β a : ℝ}

structure InscribedQuadrilateral (ABCD : Type) : Prop :=
  (diagonal_AC : ℝ)
  (angle_α : ℝ)
  (angle_β : ℝ)
  (AC_eq_a : diagonal_AC = a)
  (form_angles : diagonal_AC ∠ α ∧ diagonal_AC ∠ β)

theorem inscribed_quad_area_bounds
  {ABCD : Type} [InscribedQuadrilateral ABCD]
  (S_ABCD : ℝ)
  (h_inscribed : ∀ (a α β : ℝ), InscribedQuadrilateral ABCD → 
    ∀ (S_ABCD : ℝ), 
    (S_ABCD ≥ (a^2 * (Real.sin (α + β)) * (Real.sin β)) / (2 * (Real.sin α))
    ∧
    S_ABCD ≤ (a^2 * (Real.sin (α + β)) * (Real.sin α)) / (2 * (Real.sin β)))) :
  (a^2 * (Real.sin (α + β)) * (Real.sin β)) / (2 * (Real.sin α))
  ≤ S_ABCD
  ∧
  S_ABCD ≤ (a^2 * (Real.sin (α + β)) * (Real.sin α)) / (2 * (Real.sin β)) :=
sorry

end inscribed_quad_area_bounds_l577_577280


namespace cortney_downloads_all_files_in_2_hours_l577_577106

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l577_577106


namespace a_999_eq_499500_l577_577708

noncomputable def a : ℕ → ℕ
| 1       := 1
| (n + 1) := a n + 2 * a n / n

theorem a_999_eq_499500 : a 999 = 499500 :=
  sorry

end a_999_eq_499500_l577_577708


namespace solve_number_l577_577042

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l577_577042


namespace distance_traveled_is_22_l577_577871

-- Define the velocity function
def velocity (t : ℝ) : ℝ := 2 * t + 3

-- The distance traveled by the object between 3 and 5 seconds
def distance_traveled : ℝ := ∫ t in (3 : ℝ)..(5 : ℝ), velocity t

-- The proof statement we aim to prove
theorem distance_traveled_is_22 : distance_traveled = 22 :=
by
  -- Proof goes here
  sorry

end distance_traveled_is_22_l577_577871


namespace unique_integer_solution_l577_577535

theorem unique_integer_solution (n : ℤ) :
  (⟦ n^2 / 3 ⟧ - ⟦ n / 2 ⟧ ^ 2 = 3) ↔ (n = 6) :=
sorry

end unique_integer_solution_l577_577535


namespace simplest_fraction_C_l577_577867

def FractionA (a : ℚ) : ℚ := (a + 1) / (a^2 - 1)
def FractionB (a b c : ℚ) : ℚ := 4 * a / (6 * b * c^2)
def FractionC (a : ℚ) : ℚ := 2 * a / (2 - a)
def FractionD (a b : ℚ) : ℚ := (a + b) / (a^2 + a * b)

theorem simplest_fraction_C
  (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (condA : ∀ a, FractionA a = (1 : ℚ)) 
  (condB : ∀ a b c, FractionB a b c = (1 : ℚ))
  (condC : ∀ a, ¬ ∃ d, d*a = 2 - a)
  (condD : ∀ a b, FractionD a b = (1 : ℚ)) :
  (∀ x, x ≠ a ≠ 1 → x ≠ b ≠ 1 → x ≠ c ≠ 1 → x ≠ 0) → FractionC a = 2 * a / (2 - a) :=
begin
  sorry
end

end simplest_fraction_C_l577_577867


namespace Jim_fits_into_average_l577_577543

variable (Jim_apples : ℕ) (Jane_apples : ℕ) (Jerry_apples : ℕ) (Jack_apples : ℕ) (Jill_apples : ℕ)

def Jack_gives_25_percent : ℕ := (Jack_apples / 4)

def Jack_new_apples : ℕ := (Jack_apples - Jack_gives_25_percent)
def Jill_new_apples : ℕ := (Jill_apples + Jack_gives_25_percent)
def total_apples : ℕ := (Jim_apples + Jane_apples + Jerry_apples + Jack_new_apples + Jill_new_apples)
def average_apples_per_person : ℕ := (total_apples / 5)

theorem Jim_fits_into_average :
  Jim_apples = 20 → Jane_apples = 60 → Jerry_apples = 40 →
  Jack_apples = 80 → Jill_apples = 50 → 
  (average_apples_per_person / Jim_apples) = 2.5 :=
by
  intros hjim hjane hjerry hjack hjill
  sorry

end Jim_fits_into_average_l577_577543


namespace geometric_sequence_sum_l577_577234

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n)
    (h1 : a 0 + a 1 = 324) (h2 : a 2 + a 3 = 36) : a 4 + a 5 = 4 :=
by
  sorry

end geometric_sequence_sum_l577_577234


namespace distance_between_vertices_l577_577130

theorem distance_between_vertices (x y : ℝ) :
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0 → 
  let a^2 := 1/4 in 2 * real.sqrt a^2 = 1 :=
by
  sorry

end distance_between_vertices_l577_577130


namespace circumcenter_of_excenters_l577_577272

open EuclideanGeometry

structure Triangle (α : Type) [EuclideanSpace α] :=
(A B C : α)

structure PerpendicularFeet (α : Type) [EuclideanSpace α] (P : α) (T : Triangle α) :=
(D E F : α)
(hD : isPerpendicular P T.B C D)
(hE : isPerpendicular P T.C A E)
(hF : isPerpendicular P T.A B F)

structure Excenters (α : Type) [EuclideanSpace α] (T : Triangle α) :=
(Ia Ib Ic : α)
(hIa : isExcenter T.A T.B T.C Ia)
(hIb : isExcenter T.B T.C T.A Ib)
(hIc : isExcenter T.C T.A T.B Ic)

noncomputable def isCircumcenter {α : Type} [EuclideanSpace α]
  (P : α) (T : Triangle α) (Ex : Excenters α T) : Prop :=
  dist P Ex.Ia = dist P Ex.Ib ∧ dist P Ex.Ib = dist P Ex.Ic

theorem circumcenter_of_excenters {α : Type} [EuclideanSpace α]
  (T : Triangle α) (P : α) 
  (Feet : PerpendicularFeet α P T)
  (Ex : Excenters α T)
  (h_eq : dist P T.A ^ 2 + dist P Feet.D ^ 2 =
          dist P T.B ^ 2 + dist P Feet.E ^ 2 ∧
          dist P T.B ^ 2 + dist P Feet.E ^ 2 =
          dist P T.C ^ 2 + dist P Feet.F ^ 2) :
  isCircumcenter P T Ex := by
  sorry

end circumcenter_of_excenters_l577_577272


namespace bikes_per_gym_l577_577843

noncomputable def num_gyms : ℕ := 20
noncomputable def bike_cost : ℝ := 700
noncomputable def treadmill_cost : ℝ := 1.5 * bike_cost
noncomputable def elliptical_cost : ℝ := 2 * treadmill_cost
noncomputable def total_cost : ℝ := 455000
noncomputable def treadmill_and_elliptical_cost (num_gyms : ℕ) : ℝ := (5 * treadmill_cost + 5 * elliptical_cost) * num_gyms

theorem bikes_per_gym :
  let remaining_cost := total_cost - treadmill_and_elliptical_cost num_gyms in
  let total_bikes := remaining_cost / bike_cost in
  total_bikes / num_gyms = 10 := 
by
  sorry

end bikes_per_gym_l577_577843


namespace bike_ride_distance_l577_577861

theorem bike_ride_distance (D : ℝ) (h : D / 10 = D / 15 + 0.5) : D = 15 :=
  sorry

end bike_ride_distance_l577_577861


namespace angle_DAB_is_45_degrees_l577_577229

theorem angle_DAB_is_45_degrees
  (A B C D E: Type*)
  (ACB_is_right : ∀ (P Q R : Type*), ∠PQR = 90 → angle_eq A B C 90)
  (CA_eq_CB : ∀ (P Q : Type*), segment_eq P C Q → segment_eq A C B)
  (BCDE_is_square : ∀ (P Q R S : Type*), is_square P Q R S inside (triangle A B C))
  : ∠DAB = 45 :=
sorry

end angle_DAB_is_45_degrees_l577_577229


namespace percent_more_than_l577_577000

variable (Erica Robin Charles : ℕ)
variable (w1 : Robin = Erica + (30 * Erica) / 100)
variable (w2 : Charles = Erica + (60 * Erica) / 100)

theorem percent_more_than :
  (Charles - Robin) * 100 / Robin ≈ 23.08 := sorry

end percent_more_than_l577_577000


namespace quadratic_decreasing_interval_l577_577417

variable {a b c : ℝ}

theorem quadratic_decreasing_interval (h_a_gt_0 : a > 0) (h_roots : ∃ C : ℝ, (x + 5) * (x - 3) * C = a * x^2 + b * x + c) :
  (∀ x : ℝ, x ∈ Icc (-∞) (-1 : ℝ) → derivative (λ x, a * x^2 + b * x + c) x < 0) :=
sorry

end quadratic_decreasing_interval_l577_577417


namespace remainder_div_mod_l577_577727

theorem remainder_div_mod (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) 
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : a ∈ {1, 2, 4, 8}) (h8 : b ∈ {1, 2, 4, 8}) 
  (h9 : c ∈ {1, 2, 4, 8}) (h10 : d ∈ {1, 2, 4, 8}) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (a * b * c * d)⁻¹) % 9 = 3 :=
by 
  sorry

end remainder_div_mod_l577_577727


namespace best_illustration_of_inflation_l577_577379

-- Definitions related to the problem
def isInflationIllustration (diagram: ℕ → ℝ) : Prop :=
  ∀ t1 t2, t1 < t2 → diagram t1 < diagram t2

variable (Diagram1 Diagram2 Diagram3 Diagram4 : ℕ → ℝ)

-- Conditions as definitions in Lean
def Diagram1Condition : Prop := isInflationIllustration Diagram1
def Diagram2Condition : Prop := isInflationIllustration Diagram2
def Diagram3Condition : Prop := isInflationIllustration Diagram3
def Diagram4Condition : Prop := isInflationIllustration Diagram4

-- Statement of the problem
theorem best_illustration_of_inflation : Diagram3Condition :=
by
  -- Specific conditions for this problem (could be added)
  sorry

end best_illustration_of_inflation_l577_577379


namespace floor_sum_correct_l577_577879

def floor_sum_1_to_24 := 
  let sum := (3 * 1) + (5 * 2) + (7 * 3) + (9 * 4)
  sum

theorem floor_sum_correct : floor_sum_1_to_24 = 70 := by
  sorry

end floor_sum_correct_l577_577879


namespace measured_centimeters_on_map_l577_577639

def inches_per_mile : ℝ := 1.5 / 24
def centimeters_per_inch : ℕ := 254 / 100 -- 2.54 as a ratio of whole numbers
def distance_measured_miles : ℝ := 283.46456692913387
def approximate_centimeters : ℝ := 45

theorem measured_centimeters_on_map :
  (inches_per_mile * distance_measured_miles : ℝ) * (centimeters_per_inch : ℝ) ≈ approximate_centimeters := by
  sorry

end measured_centimeters_on_map_l577_577639


namespace inequality_must_hold_l577_577951

-- Problem Statement
variable {f : ℝ → ℝ} (h_odd : ∀ x, f(-x) = -f(x)) 
  (h_diff : Continuous (f')) 
  (h_bound : ∀ x, 0 < x → f'' x < 2 * f x)

theorem inequality_must_hold : e^2 * f (-1) < - f 2 :=
  by sorry

end inequality_must_hold_l577_577951


namespace smallest_part_of_120_divided_into_3_5_7_is_24_l577_577533

-- Definitions for the conditions
def proportional_parts (p1 p2 p3 : ℕ) (total : ℕ) : Prop :=
  ∃ x : ℕ, total = p1 * x + p2 * x + p3 * x

def smallest_part (p : ℕ) (r : ℕ) : Prop :=
  p = 3 * r

-- The math proof problem
theorem smallest_part_of_120_divided_into_3_5_7_is_24 :
  ∀ (p1 p2 p3 total : ℕ),
    p1 = 3 → p2 = 5 → p3 = 7 → total = 120 →
    proportional_parts p1 p2 p3 total →
    smallest_part (min (3 * (total / (p1 + p2 + p3))) (min (5 * (total / (p1 + p2 + p3))) (7 * (total / (p1 + p2 + p3))))) 8 :=
by
  intros p1 p2 p3 total h1 h2 h3 h4 h5
  have x_val : ℕ := total / (p1 + p2 + p3)
  have : p1 * x_val = 24 := by sorry
  rw [min_eq_left (by sorry)]
  use 8
  exact this


end smallest_part_of_120_divided_into_3_5_7_is_24_l577_577533


namespace max_sum_squares_polygon_circle_l577_577864

/-- Given that a polygon is inscribed in a circle, prove that the inscribed polygon
with the maximum sum of the squares of the lengths of its sides is an equilateral triangle. -/
theorem max_sum_squares_polygon_circle (R : ℝ)
  (P : Type) [Geometry.Polygon P] [Geometry.Inscribed P R] :
  ∃ (E : Geometry.Triangle), 
  Geometry.Equilateral E ∧ Geometry.Inscribed E R ∧
  ∀ (Q : Type) [Geometry.Polygon Q] [Geometry.Inscribed Q R],
  ∑ (s : ℝ) in Geometry.sides_lengths Q, s^2 ≤ ∑ (t : ℝ) in Geometry.sides_lengths E, t^2 :=
sorry

end max_sum_squares_polygon_circle_l577_577864


namespace sum_of_reciprocals_l577_577114

noncomputable def S (n : ℕ) : ℚ :=
∑ p q in finset.filter (λ pq : ℕ × ℕ, 0 < pq.1 ∧ pq.1 < pq.2 ∧ pq.2 ≤ n ∧ pq.1 + pq.2 > n ∧ Nat.gcd pq.1 pq.2 = 1)
  ((finset.range (n + 1)).product (finset.range (n + 1))),
  (1 : ℚ) / (pq.1 * pq.2)

theorem sum_of_reciprocals (n : ℕ) (hn : n ≥ 2) : S n = 1 / 2 := 
sorry

end sum_of_reciprocals_l577_577114


namespace range_of_y_l577_577093

noncomputable def y (x : ℝ) : ℝ := abs(x + 10) - abs(3 * x - 1)

theorem range_of_y : ∀ (y' : ℝ), (∃ (x : ℝ), y x = y') ↔ y' ∈ Set.Iic 31 := 
by
  sorry

end range_of_y_l577_577093


namespace combined_sleep_time_l577_577409

variables (cougar_night_sleep zebra_night_sleep total_sleep_cougar total_sleep_zebra total_weekly_sleep : ℕ)

theorem combined_sleep_time :
  (cougar_night_sleep = 4) →
  (zebra_night_sleep = cougar_night_sleep + 2) →
  (total_sleep_cougar = cougar_night_sleep * 7) →
  (total_sleep_zebra = zebra_night_sleep * 7) →
  (total_weekly_sleep = total_sleep_cougar + total_sleep_zebra) →
  total_weekly_sleep = 70 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end combined_sleep_time_l577_577409


namespace opposite_of_2022_l577_577347

theorem opposite_of_2022 (n : ℤ) (h : n = 2022) : -n = -2022 :=
by {
  rw h,
  simp,
}

end opposite_of_2022_l577_577347


namespace range_of_a_l577_577156

noncomputable def f (x a : ℝ) := x * Real.log x - a * x
noncomputable def g (x : ℝ) := x^3 - x + 6
noncomputable def g' (x : ℝ) := 3 * x^2 - 1

theorem range_of_a (x : ℝ) (a : ℝ) (hx_pos : 0 < x) (h_condition : 2 * (f x a) ≤ g' x + 2) :
  a ∈ set.Ici (-2) :=
by
  sorry

end range_of_a_l577_577156


namespace sum_of_squares_of_solutions_l577_577139

theorem sum_of_squares_of_solutions : 
  (∑ x in ({0, 1} : Finset ℝ) ∪ ({a, b} : Finset ℝ), x^2) = 1003 / 502 :=
sorry

end sum_of_squares_of_solutions_l577_577139


namespace calculate_expression_l577_577876

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 :=
by {
  -- hint to the Lean prover to consider associative property
  sorry
}

end calculate_expression_l577_577876


namespace convex_polygon_segment_length_l577_577836

theorem convex_polygon_segment_length (P: Type) [convex P] 
  (h_area : area P > 0.5) (h_bound: bounded_by_square P 1) :
  ∃ (l : ℝ), l ≥ 0.5 ∧ segment_parallel_to_side P l :=
sorry

end convex_polygon_segment_length_l577_577836


namespace a_total_money_received_l577_577813

def total_profit : ℝ := 9600
def a_investment : ℝ := 5000
def b_investment : ℝ := 1000
def a_management_share_percentage : ℝ := 10 / 100

def management_share (total_profit : ℝ) (percentage : ℝ) : ℝ := total_profit * percentage
def remaining_profit (total_profit management_share : ℝ) : ℝ := total_profit - management_share
def capital_ratio (a_investment b_investment : ℝ) : ℝ := a_investment / (a_investment + b_investment)
def a_share_of_remaining (remaining_profit capital_ratio : ℝ) : ℝ := remaining_profit * capital_ratio

theorem a_total_money_received 
  (total_profit a_investment b_investment a_management_share_percentage : ℝ) 
  (h_total_profit : total_profit = 9600) 
  (h_a_investment : a_investment = 5000) 
  (h_b_investment : b_investment = 1000) 
  (h_a_management_share_percentage : a_management_share_percentage = 10 / 100) :
  let a_management_share := management_share total_profit a_management_share_percentage in
  let remaining := remaining_profit total_profit a_management_share in
  let ratio := capital_ratio a_investment b_investment in
  let a_share := a_share_of_remaining remaining ratio in
  (a_management_share + a_share) = 8160 := 
by
  sorry

end a_total_money_received_l577_577813


namespace find_m_l577_577944

theorem find_m (m A B : ℝ) (h1 : (2 - m * complex.I) / (1 + 2 * complex.I) = A + B * complex.I) (h2 : A + B = 0) : m = -2 := by
  sorry

end find_m_l577_577944


namespace regression_lines_intersect_at_sample_center_l577_577738

variable {ℝ} -- Real numbers

/-- Regression lines -/
def regression_line (s: Set (ℝ × ℝ)) (l: ℝ → ℝ) :=
  ∀ (x: ℝ) (y: ℝ), (x, y) ∈ s → y = l x

variables {l₁ l₂: ℝ → ℝ}
variables {s: Set (ℝ × ℝ)}
variables {x̄ ȳ : ℝ}

axiom regression_line_l1: regression_line s l₁
axiom regression_line_l2: regression_line s l₂
axiom point_in_lines: ∃ x y, (x, y) ∈ s ∧ x = x̄ ∧ y = ȳ

theorem regression_lines_intersect_at_sample_center:
  l₁ x̄ = ȳ ∧ l₂ x̄ = ȳ :=
sorry

end regression_lines_intersect_at_sample_center_l577_577738


namespace range_of_a_l577_577626

-- Definitions of the two curves
def curve1 (a x : ℝ) := (a * x - 1) * Real.exp x
def curve2 (x : ℝ) := (1 - x) * Real.exp (-x)

-- Derivatives of the two curves
def deriv_curve1 (a x : ℝ) := (a * x + a - 1) * Real.exp x
def deriv_curve2 (x : ℝ) := (x - 2) * Real.exp (-x)

-- Condition for perpendicular tangents
def perp_condition (a x0 : ℝ) := 
  deriv_curve1 a x0 * deriv_curve2 x0 = -1

-- The range of the real number a given the condition
theorem range_of_a
  (x0 : ℝ) 
  (h1 : 0 <= x0) 
  (h2 : x0 <= 3 / 2) 
  (h3 : perp_condition a x0) :
  1 <= a ∧ a <= 3 / 2 := by
  sorry

end range_of_a_l577_577626


namespace total_cookies_correct_l577_577064

noncomputable def cookies_monday : ℕ := 5
def cookies_tuesday := 2 * cookies_monday
def cookies_wednesday := cookies_tuesday + (40 * cookies_tuesday / 100)
def total_cookies := cookies_monday + cookies_tuesday + cookies_wednesday

theorem total_cookies_correct : total_cookies = 29 := by
  sorry

end total_cookies_correct_l577_577064


namespace sum_of_degrees_eq_twice_number_of_edges_number_of_odd_degree_vertices_is_even_l577_577664

-- Definition of a graph in Lean as an undirected graph with set of vertices and set of edges
structure Graph (V : Type) :=
  (E : set (V × V))
  (no_loops : ∀ ⦃v⦄, (v, v) ∉ E)
  (undirected : ∀ ⦃u v⦄, (u, v) ∈ E → (v, u) ∈ E)

namespace Graph

variables {V : Type} (G : Graph V)

-- Definition of the degree of a vertex
def degree (v : V) : ℕ :=
  (G.E.to_finset.filter (λ e, e.1 = v ∨ e.2 = v)).card

-- Part (a): Sum of degrees of all vertices is twice the number of edges
theorem sum_of_degrees_eq_twice_number_of_edges : 
  (∑ v in G.E.to_finset.bUnion (λ e, {e.1, e.2}), G.degree v) = 2 * G.E.to_finset.card :=
sorry

-- Part (b): Number of vertices with odd degree is even
theorem number_of_odd_degree_vertices_is_even : 
  even (G.E.to_finset.bUnion (λ e, {e.1, e.2}).filter (λ v, odd (G.degree v)).card) :=
sorry

end Graph

end sum_of_degrees_eq_twice_number_of_edges_number_of_odd_degree_vertices_is_even_l577_577664


namespace prob_C_prob_B_prob_at_least_two_l577_577014

variable {P : Set ℕ → ℝ}

-- Given Conditions
def probability_A := 3 / 4
def probability_AC_incorrect := 1 / 12
def probability_BC_correct := 1 / 4

-- Definitions used in conditions
def probability_C := 2 / 3
def probability_B := 3 / 8

-- Proof Statements
theorem prob_C : P {3} = probability_C := sorry
theorem prob_B : P {2} = probability_B := sorry
theorem prob_at_least_two :
  (P {1, 2, 3} + P {1, 2} (1 - probability_C) + P {1, 3} (1 - probability_B) + P {2, 3} (1 - probability_A)) = 21 / 32 := sorry

end prob_C_prob_B_prob_at_least_two_l577_577014


namespace gcd_of_A_B_l577_577224

noncomputable def A (k : ℕ) := 2 * k
noncomputable def B (k : ℕ) := 5 * k

theorem gcd_of_A_B (k : ℕ) (h_lcm : Nat.lcm (A k) (B k) = 180) : Nat.gcd (A k) (B k) = 18 :=
by
  sorry

end gcd_of_A_B_l577_577224


namespace sum_floor_log2_to_2048_l577_577083

theorem sum_floor_log2_to_2048 :
  (Finset.sum (Finset.range 2048.succ) (λ N : ℕ, Int.toNat ⌊Real.logb 2 (N : ℝ)⌋) = 14349) :=
by
  sorry

end sum_floor_log2_to_2048_l577_577083


namespace fruit_seller_apples_l577_577018

theorem fruit_seller_apples : 
  ∃ (x : ℝ), (x * 0.6 = 420) → x = 700 :=
sorry

end fruit_seller_apples_l577_577018


namespace find_B_days_l577_577010

noncomputable def work_rate_A := 1 / 15
noncomputable def work_rate_B (x : ℝ) := 1 / x

theorem find_B_days (x : ℝ) : 
  (5 * (work_rate_A + work_rate_B x) = 0.5833333333333334) →
  (x = 20) := 
by 
  intro h,
  sorry

end find_B_days_l577_577010


namespace domain_f_value_of_a_l577_577178

open Real

-- Define the function f : ℝ → ℝ for given a
def f (a : ℝ) (x : ℝ) : ℝ := log a (1 - x) + log a (3 + x)

-- Hypothesis: 0 < a < 1
axiom a_condition (a : ℝ) : 0 < a ∧ a < 1

-- Proof goals
theorem domain_f (a : ℝ) : ∀ x : ℝ, f a x ∈ ℝ → x ∈ Ioo (-3 : ℝ) 1 := sorry

theorem value_of_a (a : ℝ) : (∃ x : ℝ, f a x = -4) → a = real.sqrt 2 / 2 := sorry

end domain_f_value_of_a_l577_577178


namespace sum_mod_30_l577_577366

theorem sum_mod_30 (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 7) 
  (h3 : c % 30 = 18) : 
  (a + 2 * b + c) % 30 = 17 := 
by
  sorry

end sum_mod_30_l577_577366


namespace find_k_parallel_vectors_l577_577558

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem find_k_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (-2, 6)
  vector_parallel a b → k = -3 :=
by
  sorry

end find_k_parallel_vectors_l577_577558


namespace cos_2theta_l577_577214

theorem cos_2theta (θ : ℝ) (h : cos θ + sin θ = 3 / 2) : cos (2 * θ) = 7 / 32 := by
  sorry

end cos_2theta_l577_577214


namespace Jose_got_5_questions_wrong_l577_577233

def Jose_questions_wrong (M J A : ℕ) : Prop :=
  M = J - 20 ∧
  J = A + 40 ∧
  M + J + A = 210 ∧
  (50 * 2 = 100) ∧
  (100 - J) / 2 = 5

theorem Jose_got_5_questions_wrong (M J A : ℕ) (h1 : M = J - 20) (h2 : J = A + 40) (h3 : M + J + A = 210) : 
  Jose_questions_wrong M J A :=
by
  sorry

end Jose_got_5_questions_wrong_l577_577233


namespace standard_equation_hyperbola_line_slope_intersect_hyperbola_exactly_one_point_l577_577945

variables (x y k : ℝ)
noncomputable def hyperbola : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let (x, y) := p in (x^2) / 16 - (y^2) / 9 = 1 }

theorem standard_equation_hyperbola :
  (∀ (x y : ℝ), let p := (x, y) in p ∈ hyperbola → (x^2) / 16 - (y^2) / 9 = 1) :=
sorry

theorem line_slope_intersect_hyperbola_exactly_one_point :
  (∀ k : ℝ, let line := λ (x: ℝ), k * (x - 3) in
   ((λ (x y : ℝ), let p := (x, y) in p ∈ hyperbola ∧ y = line x) → False)
    ↔ (k = 3 / 4 ∨ k = -3 / 4 ∨ k = 3 * sqrt 7 / 7 ∨ k = -3 * sqrt 7 / 7)) :=
sorry

end standard_equation_hyperbola_line_slope_intersect_hyperbola_exactly_one_point_l577_577945


namespace knights_count_l577_577646

-- Define the inhabitants and their nature
inductive Inhabitant : Type
| first : Inhabitant
| second : Inhabitant
| third : Inhabitant
| fourth : Inhabitant
| fifth : Inhabitant

open Inhabitant

-- Define whether an inhabitant is a knight or a liar
inductive Nature : Type
| knight : Nature
| liar : Nature

open Nature

-- Assume each individual is either a knight or a liar (truth value function)
def truth_value : Inhabitant → Nature → Prop
| first, knight  => sorry -- To be proven
| second, knight => sorry -- To be proven
| third, knight  => sorry -- To be proven
| fourth, knight => sorry -- To be proven
| fifth, knight  => sorry -- To be proven

-- Define the statements made by inhabitants as logical conditions
def statements : Prop :=
  (truth_value first knight → (1 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value second knight → (2 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value third knight → (3 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value fourth knight → ¬ (truth_value first knight) ∧ ¬ (truth_value second knight) ∧ ¬ (truth_value third knight) ∧ ¬ (truth_value fifth knight))
  ∧ (truth_value fifth knight → ¬ (truth_value fourth knight))

-- The goal is to prove that there are exactly 2 knights
theorem knights_count : (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0) = 2 :=
by
  sorry

end knights_count_l577_577646


namespace value_of_a_range_of_b_inequality_solution_set_l577_577268

section

variables {a b x : ℝ}
def f (x : ℝ) : ℝ := log ((1 + a * x) / (1 + x))

-- Prove that a = -1 given f(x) is odd
theorem value_of_a (h : ∀ x, f (-x) = -f x) : a = -1 :=
sorry

-- Prove the range of values for b is (0, 1] for the domain of f(x) = log ( (1 - x) / (1 + x) )
theorem range_of_b (h : ∀ x, x ∈ Ioo (-b) b → log ((1 - x) / (1 + x)) ∈ Real.logr) : 0 < b ∧ b ≤ 1 :=
sorry

-- Prove that the solution set of f(x) > 0 is (-1, 0)
theorem inequality_solution_set (h : a = -1) : {x : ℝ | f x > 0} = Ioo (-1) 0 :=
sorry

end

end value_of_a_range_of_b_inequality_solution_set_l577_577268


namespace simplify_expr_l577_577315

theorem simplify_expr : ((256 : ℝ) ^ (1 / 4)) * ((144 : ℝ) ^ (1 / 2)) = 48 := by
  have h1 : (256 : ℝ) = 2^8 := by
    norm_num,
  have h2 : (144 : ℝ) = 12^2 := by
    norm_num,
  have h3 : (2^8 : ℝ) ^ (1 / 4) = 4 := by
    norm_num,
  have h4 : (12^2 : ℝ) ^ (1 / 2) = 12 := by
    norm_num,
  sorry

end simplify_expr_l577_577315


namespace eval_floor_ceiling_expression_l577_577906

theorem eval_floor_ceiling_expression :
  (floor (ceil (((15 / 8 : ℚ)^2 - 1 / 2 : ℚ)) + 19 / 5) : ℤ) = 7 := by
  sorry

end eval_floor_ceiling_expression_l577_577906


namespace problem1_problem2_l577_577988

-- Definitions of sets A and B
def setA : Set ℝ := { x | x^2 - 8 * x + 15 = 0 }
def setB (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

-- Problem 1: If a = 1/5, B is a subset of A.
theorem problem1 : setB (1 / 5) ⊆ setA := sorry

-- Problem 2: If A ∩ B = B, then C = {0, 1/3, 1/5}.
def setC : Set ℝ := { a | a = 0 ∨ a = 1 / 3 ∨ a = 1 / 5 }

theorem problem2 (a : ℝ) : (setA ∩ setB a = setB a) ↔ (a ∈ setC) := sorry

end problem1_problem2_l577_577988


namespace average_class_is_45_6_l577_577385

noncomputable def average_class_score (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) 
  (zero_scorers : ℕ) (remaining_students_avg : ℕ) : ℚ :=
  let total_top_score := top_scorers * top_score
  let total_zero_score := zero_scorers * 0
  let remaining_students := total_students - top_scorers - zero_scorers
  let total_remaining_score := remaining_students * remaining_students_avg
  let total_score := total_top_score + total_zero_score + total_remaining_score
  total_score / total_students

theorem average_class_is_45_6 : average_class_score 25 3 95 3 45 = 45.6 := 
by
  -- sorry is used here to skip the proof. Lean will expect a proof here.
  sorry

end average_class_is_45_6_l577_577385


namespace grants_test_score_l577_577203

theorem grants_test_score :
  ∀ (hunter_score : ℕ) (john_score : ℕ) (grant_score : ℕ), hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 :=
by
  intro hunter_score john_score grant_score
  intro hunter_eq john_eq grant_eq
  rw [hunter_eq, john_eq, grant_eq]
  sorry

end grants_test_score_l577_577203


namespace area_of_triangle_l577_577765

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l577_577765


namespace part_I_part_II_l577_577524

def f (x : ℝ) : ℝ :=
  |x - 2|

theorem part_I (a : ℝ) (h : a < 3) :
  (∀ x, f (x - a + 2) + f (x - 1) ≥ 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2) →
  a = 2 :=
by sorry

theorem part_II (a : ℝ) :
  (∀ x, f (x - a + 2) + 2 * f (x - 1) ≥ 1) →
  a ∈ set.Iic 2 ∪ set.Ici 4 :=
by sorry

end part_I_part_II_l577_577524


namespace probability_each_book_read_by_one_student_l577_577238

theorem probability_each_book_read_by_one_student :
  let books := {bookA, bookB} in
  let students := {studentA, studentB} in
  let outcomes := {(readA, readA), (readB, readB), (readA, readB), (readB, readA), (readAB, readA), (readAB, readB), (readA, readAB), (readB, readAB), (readAB, readAB)} in
  let favorable_outcomes := {(readA, readB), (readB, readA), (readAB, readA), (readAB, readB), (readA, readAB), (readB, readAB), (readAB, readAB)} in
  (favorable_outcomes.size : ℚ) / (outcomes.size : ℚ) = 7 / 9 :=
by sorry

end probability_each_book_read_by_one_student_l577_577238


namespace solution_of_inequality_l577_577556

theorem solution_of_inequality (a b : ℝ) (h : ∀ x : ℝ, (1 < x ∧ x < 3) ↔ (x^2 < a * x + b)) :
  b^a = 81 := 
sorry

end solution_of_inequality_l577_577556


namespace quadratic_inequality_solution_set_l577_577924

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3 * x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end quadratic_inequality_solution_set_l577_577924


namespace y_decreasing_odd_l577_577221
-- Required import

-- Noncomputable context for real number functions
noncomputable theory

-- Given conditions definitions
def f (x : ℝ) : ℝ := x^3
def y (x : ℝ) : ℝ := f (-x)

-- Proof statement
theorem y_decreasing_odd : 
  (∀ x : ℝ, f x = x^3) → 
  (∀ x : ℝ, y x = f (-x)) → 
  (∀ x : ℝ, y x = -f x ∧ 
  ∀ x1 x2 : ℝ, x1 < x2 → y x1 > y x2 ∧ 
  ∀ x : ℝ, y (-x) = -y x) :=
by
  intros h1 h2
  sorry

end y_decreasing_odd_l577_577221


namespace area_ratio_square_circle_l577_577855

theorem area_ratio_square_circle (P : ℝ) (hP : 0 < P) :
  let A := (P/Pow 4) ^ 2,
      B := (P/(2 * π) ^ 2),
  (A/B) = (π/4) :=
by
  let s := P / Pow 4
  have h_sq_area : A = Pow (P / 4) 2 := sorry
  have h_circle_area : B = Pow (P / (2 * π)) 2 := sorry
  have h_perimeter : 2 * π * (P / (2 * π)) = P := sorry
  sorry

end area_ratio_square_circle_l577_577855


namespace find_two_digit_number_l577_577046

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l577_577046


namespace water_bottles_needed_l577_577460

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l577_577460


namespace total_distance_covered_l577_577802

theorem total_distance_covered :
  ∀ (r j w total : ℝ),
    r = 40 →
    j = (3 / 5) * r →
    w = 5 * j →
    total = r + j + w →
    total = 184 := by
  sorry

end total_distance_covered_l577_577802


namespace hyperbola_eccentricity_l577_577190

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h : ∀ x y : ℝ, distance (1, 0) (line_through (λ t : ℝ, (t, t * (a / b))) x y) = (sqrt 2) / 2) : 
    (a = b) → (sqrt((2:ℝ)) * a / a) = sqrt 2 := 
begin
  sorry
end

end hyperbola_eccentricity_l577_577190


namespace num_common_points_l577_577701

theorem num_common_points :
  let eq1 := (x y : ℝ) → (x - y + 3) * (4 * x + y - 5) = 0
  let eq2 := (x y : ℝ) → (x + y - 3) * (3 * x - 4 * y + 6) = 0
  ∃ (s : Finset (ℝ × ℝ)),   
    (∀ p ∈ s, eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧
    s.card = 4 := 
by
  sorry

end num_common_points_l577_577701


namespace solve_number_l577_577041

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l577_577041


namespace min_omega_l577_577614

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega (ω φ T : ℝ) (hω : ω > 0) (hφ1 : 0 < φ) (hφ2 : φ < Real.pi / 2)
  (hT : f ω φ T = Real.sqrt 3 / 2)
  (hx : f ω φ (Real.pi / 6) = 0) :
  ω = 4 := by
  sorry

end min_omega_l577_577614


namespace angle_sub_scalar_mul_l577_577548

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem angle_sub_scalar_mul (h : Real.angle_between a b = Real.pi / 3) :
  Real.angle_between (a - b) (2 • b) = Real.pi / 3 :=
sorry

end angle_sub_scalar_mul_l577_577548


namespace price_each_puppy_l577_577258

def puppies_initial : ℕ := 8
def puppies_given_away : ℕ := puppies_initial / 2
def puppies_remaining_after_giveaway : ℕ := puppies_initial - puppies_given_away
def puppies_kept : ℕ := 1
def puppies_to_sell : ℕ := puppies_remaining_after_giveaway - puppies_kept
def stud_fee : ℕ := 300
def profit : ℕ := 1500
def total_amount_made : ℕ := profit + stud_fee
def price_per_puppy : ℕ := total_amount_made / puppies_to_sell

theorem price_each_puppy :
  price_per_puppy = 600 :=
sorry

end price_each_puppy_l577_577258


namespace isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l577_577554

def isosceles_right_triangle_initial_leg_length (x : ℝ) (h : ℝ) : Prop :=
  x + 4 * ((x + 4) / 2) ^ 2 = x * x / 2 + 112 

def isosceles_right_triangle_legs_correct (a b : ℝ) (h : ℝ) : Prop :=
  a = 26 ∧ b = 26 * Real.sqrt 2

theorem isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm :
  ∃ (x : ℝ) (h : ℝ), isosceles_right_triangle_initial_leg_length x h ∧ 
                       isosceles_right_triangle_legs_correct x (x * Real.sqrt 2) h := 
by
  sorry

end isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l577_577554


namespace find_a5_l577_577163

-- Definitions related to the conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a5 (a : ℕ → ℕ) (h_arith : arithmetic_sequence a) (h_a3 : a 3 = 3)
  (h_geo : geometric_sequence (a 1) (a 2) (a 4)) :
  a 5 = 5 ∨ a 5 = 3 :=
  sorry

end find_a5_l577_577163


namespace triangle_area_bounded_by_lines_l577_577758

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l577_577758


namespace arcs_length_division_l577_577687

theorem arcs_length_division (D : ℝ) (n : ℕ) :
  let seg_len := D / (2 * n) in
  let semicircle_length := (n : ℝ) * (π * D) / (4 * n) in
  let quartercircle_length := (2 * n : ℝ) * (π * D) / (8 * n) in
  (semicircle_length + quartercircle_length) = (π * D / 2) :=
begin
  -- Proof is omitted
  sorry
end

end arcs_length_division_l577_577687


namespace volume_ratio_of_trapezoidal_pyramids_l577_577684

theorem volume_ratio_of_trapezoidal_pyramids 
  (V U : ℝ) (m n m₁ n₁ : ℝ)
  (hV : V > 0) (hU : U > 0) (hm : m > 0) (hn : n > 0) (hm₁ : m₁ > 0) (hn₁ : n₁ > 0)
  (h_ratio : U / V = (m₁ + n₁)^2 / (m + n)^2) :
  U / V = (m₁ + n₁)^2 / (m + n)^2 :=
sorry

end volume_ratio_of_trapezoidal_pyramids_l577_577684


namespace Bryan_deposited_312_l577_577294

-- Definitions based on conditions
def MarkDeposit : ℕ := 88
def TotalDeposit : ℕ := 400
def MaxBryanDeposit (MarkDeposit : ℕ) : ℕ := 5 * MarkDeposit 

def BryanDeposit (B : ℕ) : Prop := B < MaxBryanDeposit MarkDeposit ∧ MarkDeposit + B = TotalDeposit

theorem Bryan_deposited_312 : BryanDeposit 312 :=
by
   -- Proof steps go here
   sorry

end Bryan_deposited_312_l577_577294


namespace equation_of_hyperbola_l577_577421

-- Define the given conditions as hypotheses
variables {x y : ℝ}

-- Define the fact that the hyperbola passes through (2, sqrt(2))
def HyperbolaThroughPoint (h : ℝ → ℝ → Prop) : Prop :=
  h 2 (Real.sqrt 2)

-- Prove the equation of the hyperbola
theorem equation_of_hyperbola :
  ∃ λ : ℝ, λ = 2 ∧ (∀ x y : ℝ, HyperbolaThroughPoint (λ x y = x^2 - y^2) → x^2 - y^2 = 2) :=
sorry

end equation_of_hyperbola_l577_577421


namespace arrangement_of_chairs_and_stools_l577_577365

theorem arrangement_of_chairs_and_stools :
  (Nat.choose 10 3) = 120 :=
by
  -- Proof goes here
  sorry

end arrangement_of_chairs_and_stools_l577_577365


namespace S1_div_S2_eq_one_l577_577477

-- Defining S1 and S2
def S1 : ℝ := ∑ k in finset.range 1 2020, (-1)^(k+1) * (1/3)^k
def S2 : ℝ := ∑ k in finset.range 1 2020, (-1)^(k+1) * (1/3)^k

-- Prove the main statement
theorem S1_div_S2_eq_one : S1 / S2 = 1 := 
  by sorry

end S1_div_S2_eq_one_l577_577477


namespace complex_number_in_second_quadrant_l577_577702

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Given conditions
lemma i_squared_eq_neg_one : i^2 = -1 := by
  simp [Complex.I_sq]

-- Problem statement
theorem complex_number_in_second_quadrant :
  let z := (3 + 4 * i) * i in
  z.re < 0 ∧ z.im > 0 :=
by
  let z := (3 + 4 * i) * i
  sorry

end complex_number_in_second_quadrant_l577_577702


namespace elongation_rate_significantly_improved_l577_577401

noncomputable def x : List ℕ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
noncomputable def y : List ℕ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z : List ℤ := List.zipWith (λ a b => a - b) x y

def mean_z : ℚ := (z.sum : ℚ) / 10
def variance_z : ℚ :=
  (z.map (λ zi => (zi - mean_z) ^ 2).sum : ℚ) / 10

theorem elongation_rate_significantly_improved :
  mean_z = 11 ∧ variance_z = 61 ∧ mean_z ≥ 2 * Real.sqrt (variance_z / 10) :=
by
  have z_vals : z = [9, 6, 8, -8, 15, 11, 19, 18, 20, 12] := by sorry
  have mean_z_val : mean_z = 11 := by sorry
  have variance_z_val : variance_z = 61 := by sorry
  have condition : 2 * Real.sqrt (variance_z / 10) < 11 := by sorry
  exact ⟨mean_z_val, variance_z_val, le_of_lt condition⟩

end elongation_rate_significantly_improved_l577_577401


namespace sum_floor_log2_l577_577084

open Int

theorem sum_floor_log2 (S: ℕ) (hS : S = ∑ N in (finset.range 2048).map (λ x, x + 1), ⌊log (N : ℝ) / log 2⌋) : S = 45055 :=
sorry

end sum_floor_log2_l577_577084


namespace number_of_proper_subsets_of_12_l577_577546

theorem number_of_proper_subsets_of_12 : 
  ∃ M : finset (fin 3), M ⊂ {1, 2} → M.card = 3 :=
by
  sorry

end number_of_proper_subsets_of_12_l577_577546


namespace normal_dist_mean_l577_577151

theorem normal_dist_mean (μ σ : ℝ) (h : ∀ X : ℝ, X ~ Normal μ σ → P(X ≤ 0) = P(X ≥ 2)) : μ = 1 :=
sorry

end normal_dist_mean_l577_577151


namespace cone_volume_l577_577520

theorem cone_volume (r l: ℝ) (r_eq : r = 2) (l_eq : l = 4) (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) :
  (1 / 3) * π * r^2 * h = (8 * Real.sqrt 3 * π) / 3 :=
by
  -- Sorry to skip the proof
  sorry

end cone_volume_l577_577520


namespace electricity_price_per_kWh_l577_577337

theorem electricity_price_per_kWh (consumption_rate : ℝ) (hours_used : ℝ) (total_cost : ℝ) :
  consumption_rate = 2.4 → hours_used = 25 → total_cost = 6 →
  total_cost / (consumption_rate * hours_used) = 0.10 :=
by
  intros hc hh ht
  have h_energy : consumption_rate * hours_used = 60 :=
    by rw [hc, hh]; norm_num
  rw [ht, h_energy]
  norm_num

end electricity_price_per_kWh_l577_577337


namespace triangle_area_bounded_by_lines_l577_577759

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l577_577759


namespace coefficient_sqrt_x_in_expansion_l577_577332

open BigOperators

-- Define the expression
def expr (x : ℝ) := (x^(3/2) - 1/x)^7

-- Theorem to prove the coefficient of sqrt(x) term
theorem coefficient_sqrt_x_in_expansion : 
  ∀ (x : ℝ), 
  ∃ c : ℝ, 
  (∃ r : ℕ, (21 - 5 * r) / 2 = 1 / 2 ∧ c = (-1)^r * Nat.choose 7 r) ∧ c = 35 := 
by 
  sorry

end coefficient_sqrt_x_in_expansion_l577_577332


namespace larger_triangle_perimeter_l577_577742

def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

def scale_factor (a b : ℝ) (shortest_side : ℝ) : ℝ := shortest_side / a

def similar_triangle_side (original_side : ℝ) (factor : ℝ) : ℝ := original_side * factor

def perimeter (sides : List ℝ) : ℝ := sides.foldl (· + ·) 0

theorem larger_triangle_perimeter :
  ∀ (a b shortest_side : ℝ), 
  a = 8 → b = 15 → shortest_side = 20 → 
  perimeter [similar_triangle_side a (scale_factor a b shortest_side), 
             similar_triangle_side b (scale_factor a b shortest_side), 
             similar_triangle_side (hypotenuse a b) (scale_factor a b shortest_side)] = 100 :=
by
  intros 
  rw [←Real.sqrt_inj]
  unfold perimeter similar_triangle_side scale_factor hypotenuse 
  sorry

end larger_triangle_perimeter_l577_577742


namespace value_of_S₁₂_l577_577500

def sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0 ∧ (a n * (a (n + 1) + ∑ i in finset.range (n + 1), a i)
                 - a (n + 1) * ∑ i in finset.range (n + 1), a i
                 + a n - a (n + 1)
                 = (1 / 2) * a n * a (n + 1))

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range (n + 1), a i

theorem value_of_S₁₂ (a : ℕ → ℝ) (h₀ : a 0 = 1) (seq_cond : sequence a) :
  (3 / 34) * S a 11 = 3 :=
sorry

end value_of_S₁₂_l577_577500


namespace ratio_y_to_x_l577_577870

-- Definitions based on conditions
variable (c : ℝ) -- Cost price
def x : ℝ := 0.8 * c -- Selling price for a loss of 20%
def y : ℝ := 1.25 * c -- Selling price for a gain of 25%

-- Statement to prove the ratio of y to x
theorem ratio_y_to_x : y / x = 25 / 16 := by
  -- skip the proof
  sorry

end ratio_y_to_x_l577_577870


namespace largest_fraction_l577_577509

theorem largest_fraction (a b c d e : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max ((a + b + e) / (c + d))
                        (max ((a + d) / (b + e))
                            (max ((b + c) / (a + e)) ((c + e) / (a + b + d)))) := 
sorry

end largest_fraction_l577_577509


namespace math_proof_l577_577185

noncomputable def f (ω x : ℝ) := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem math_proof (h1 : ∀ x, f ω x = f ω (x + π)) (h2 : 0 < ω) :
  (ω = 2) ∧ (f 2 (-5 * Real.pi / 6) = 0) ∧ ¬∀ x : ℝ, x ∈ Set.Ioo (Real.pi / 3) (11 * Real.pi / 12) → 
  (∃ x₁ x₂ : ℝ, f 2 x₁ < f 2 x₂) ∧ (∀ x : ℝ, f 2 (x - Real.pi / 3) ≠ Real.cos (2 * x - Real.pi / 6)) := 
by
  sorry

end math_proof_l577_577185


namespace Annie_total_cookies_l577_577067

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end Annie_total_cookies_l577_577067


namespace lines_intersect_at_x_value_l577_577358

theorem lines_intersect_at_x_value :
  (∃ x y : ℝ, y = 3 * x + 1 ∧ 5 * x + y = 100) →
  ∃ x : ℝ, x = 99 / 8 :=
by
  intro h
  cases h with x hx
  exists x
  sorry

end lines_intersect_at_x_value_l577_577358


namespace Alex_total_marbles_value_l577_577627

noncomputable def Lorin_initial_black := 4
noncomputable def Lorin_initial_green := 8.5
noncomputable def Lorin_initial_red   := 3

noncomputable def Lorin_final_black := Lorin_initial_black
noncomputable def Lorin_final_green := Lorin_initial_green - 2.5
noncomputable def Lorin_final_red   := Lorin_initial_red + 1.5

noncomputable def Jimmy_initial_yellow := 22
noncomputable def Jimmy_initial_green  := 6.2
noncomputable def Jimmy_initial_red    := 2 * Lorin_initial_red

noncomputable def Jimmy_final_yellow := Jimmy_initial_yellow - 3
noncomputable def Jimmy_final_green  := Jimmy_initial_green - 1
noncomputable def Jimmy_final_red    := Jimmy_initial_red

noncomputable def Alex_black := 2 * Lorin_initial_black
noncomputable def Alex_yellow := 1.5 * Jimmy_final_yellow
noncomputable def Alex_green := 3.5 * Lorin_initial_green
noncomputable def Alex_red := Jimmy_final_red

noncomputable def Alex_total_marbles := Alex_black + Alex_yellow + Alex_green + Alex_red

theorem Alex_total_marbles_value : Alex_total_marbles = 72.25 :=
by
  calc 
  Alex_total_marbles = 8 + 28.5 + 29.75 + 6 : by sorry
                      ... = 72.25 : by sorry

end Alex_total_marbles_value_l577_577627


namespace knights_count_l577_577648

-- Define the inhabitants and their nature
inductive Inhabitant : Type
| first : Inhabitant
| second : Inhabitant
| third : Inhabitant
| fourth : Inhabitant
| fifth : Inhabitant

open Inhabitant

-- Define whether an inhabitant is a knight or a liar
inductive Nature : Type
| knight : Nature
| liar : Nature

open Nature

-- Assume each individual is either a knight or a liar (truth value function)
def truth_value : Inhabitant → Nature → Prop
| first, knight  => sorry -- To be proven
| second, knight => sorry -- To be proven
| third, knight  => sorry -- To be proven
| fourth, knight => sorry -- To be proven
| fifth, knight  => sorry -- To be proven

-- Define the statements made by inhabitants as logical conditions
def statements : Prop :=
  (truth_value first knight → (1 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value second knight → (2 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value third knight → (3 = (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0)))
  ∧ (truth_value fourth knight → ¬ (truth_value first knight) ∧ ¬ (truth_value second knight) ∧ ¬ (truth_value third knight) ∧ ¬ (truth_value fifth knight))
  ∧ (truth_value fifth knight → ¬ (truth_value fourth knight))

-- The goal is to prove that there are exactly 2 knights
theorem knights_count : (if (truth_value first knight) then 1 else 0 + if (truth_value second knight) then 1 else 0 + if (truth_value third knight) then 1 else 0 + if (truth_value fourth knight) then 1 else 0 + if (truth_value fifth knight) then 1 else 0) = 2 :=
by
  sorry

end knights_count_l577_577648


namespace sum_floor_log2_l577_577085

open Int

theorem sum_floor_log2 (S: ℕ) (hS : S = ∑ N in (finset.range 2048).map (λ x, x + 1), ⌊log (N : ℝ) / log 2⌋) : S = 45055 :=
sorry

end sum_floor_log2_l577_577085


namespace volume_of_prism_equals_Q_l577_577391

variables {a φ : ℝ} (Q : ℝ)

-- Definition: Equilateral triangle base
def base_area (a : ℝ) : ℝ := (sqrt 3 / 4) * (a ^ 2)

-- Definition: Height of the prism based on cross-section angle
def prism_height (a φ : ℝ) : ℝ := (a * sqrt 3 / 2) * tan(φ)

-- Definition: Volume of the prism
def prism_volume (a φ : ℝ) : ℝ := base_area a * prism_height a φ

-- Given: Area of the cross-section plane is Q
axiom given_cross_section (h : base_area a * prism_height a φ = Q) : True

-- Proof: Volume of the prism is equal to Q
theorem volume_of_prism_equals_Q : prism_volume a φ = Q :=
by 
  rw [prism_volume, base_area, prism_height]
  exact given_cross_section sorry

end volume_of_prism_equals_Q_l577_577391


namespace probability_of_two_slate_rocks_l577_577361

theorem probability_of_two_slate_rocks :
  ∀ (slate pumice granite : ℕ), slate = 10 → pumice = 11 → granite = 4 → 
  (let total := slate + pumice + granite in
  ( slaterocks := slate - 1 in
  P := (slate / total) * ((slaterocks / (total - 1)) in
  P = 3 / 20 )) :=
begin
  intros slate pumice granite hslate hpumice hgranite,
  rw [hslate, hpumice, hgranite],
  let total := 10 + 11 + 4,
  let P1 := 10 / total,
  let P2 := 9 / (total - 1),
  have hP1: P1 = 2 / 5 := by sorry,
  have hP2: P2 = 3 / 8 := by sorry,
  calc (2 / 5) * (3 / 8) = 6 / 40 := by sorry,
                   ... = 3 / 20 := by sorry,
end

end probability_of_two_slate_rocks_l577_577361


namespace crayons_end_of_school_year_l577_577297

-- Definitions based on conditions
def crayons_after_birthday : Float := 479.0
def total_crayons_now : Float := 613.0

-- The mathematically equivalent proof problem statement
theorem crayons_end_of_school_year : (total_crayons_now - crayons_after_birthday = 134.0) :=
by
  sorry

end crayons_end_of_school_year_l577_577297


namespace solve_sqrt_equation_l577_577126

theorem solve_sqrt_equation :
  ∃ x : ℝ, sqrt x + sqrt (x + 9) + 3 * sqrt (x^2 + 9 * x) + sqrt (3 * x + 27) = 45 - 3 * x ∧
           x = 729 / 144 :=
by
  sorry

end solve_sqrt_equation_l577_577126


namespace angle_between_unit_vectors_correct_l577_577994

open Real

noncomputable def angle_between_unit_vectors
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (h : (2 • a + b) ⬝ b = 0) : Real :=
  acos (- (1 / 2))

theorem angle_between_unit_vectors_correct
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (h : (2 • a + b) ⬝ b = 0) : 
  angle_between_unit_vectors a b ha hb h = 2 * π / 3 :=
by
  sorry

end angle_between_unit_vectors_correct_l577_577994


namespace probability_individual_selected_l577_577161

/-- Given a population of 8 individuals, the probability that each 
individual is selected in a simple random sample of size 4 is 1/2. -/
theorem probability_individual_selected :
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  probability = (1 : ℚ) / 2 :=
by
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  sorry

end probability_individual_selected_l577_577161
