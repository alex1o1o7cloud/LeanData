import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parity
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Quotient
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.InnerProductSpace.basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Tetrahedron
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Integral
import Mathlib.NumberTheory.PythagoreanTriples.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.SolveByElim
import Mathlib.Tactic.Sorry
import Mathlib.Topology.MetricSpace.Basic

namespace question_cos_inequality_l78_78603

theorem question_cos_inequality (alpha beta gamma : ℝ) 
  (h_sin_sum : sin alpha + sin beta + sin gamma ≥ 2) : 
  cos alpha + cos beta + cos gamma ≤ sqrt 5 :=
sorry

end question_cos_inequality_l78_78603


namespace num_sides_polygon_l78_78006

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l78_78006


namespace roof_length_width_difference_l78_78401

theorem roof_length_width_difference (w l : ℝ) 
  (h1 : l = 5 * w) 
  (h2 : l * w = 720) : l - w = 48 := 
sorry

end roof_length_width_difference_l78_78401


namespace correctNumberOfProps_l78_78781

def prop1 : Prop :=
  ∀ (L1 L2 : Line) (P : Point),
    L1 ≠ L2 ∧ L1.intersect L2 = P → (L1.angle P = L2.angle P) ↔ (L1.isPerpendicularTo L2)

def prop2 : Prop :=
  ∀ (L1 L2 : Line) (P : Point) (A B : Angle),
    L1 ≠ L2 ∧ L1.angle P = A ∧ L2.angle P = B ∧ A = B ↔ L1.isPerpendicularTo L2

def prop3 : Prop :=
  ∀ (L1 L2 : Line) (A B : Angle),
    L1.parallelTo L2 ∧ (A = L1.angle) ∧ (B = L2.angle) ∧ A = B ↔ (L1.angleBisector.isPerpendicularTo L2.angleBisector)

def prop4 : Prop :=
  ∀ (L1 L2 : Line) (A B : Angle),
    (A = L1.angle) ∧ (B = L2.angle) ∧ A + B = 90 ↔ L1.angleBisector.isPerpendicularTo L2.angleBisector

def numberOfCorrectProps : Nat :=
  if prop1 = false then 0 else 1
    + if prop2 = true then 1 else 0
    + if prop3 = false then 0 else 1
    + if prop4 = true then 1 else 0

theorem correctNumberOfProps : numberOfCorrectProps = 2 := by
  -- Placeholder for steps that verify each of the propositions
  sorry

end correctNumberOfProps_l78_78781


namespace smallest_distance_proof_l78_78332

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  Complex.abs (z - w)

theorem smallest_distance_proof (z w : ℂ) 
  (h1 : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (h2 : Complex.abs (w - (-5 + 6*Complex.I)) = 4) :
  smallest_distance z w ≥ Real.sqrt 149 - 6 :=
by
  sorry

end smallest_distance_proof_l78_78332


namespace g_two_gt_one_third_g_n_gt_one_third_l78_78231

def seq_a (n : ℕ) : ℕ := 3 * n - 2
noncomputable def f (n : ℕ) : ℝ := (Finset.range n).sum (λ i => 1 / (seq_a (i + 1) : ℝ))
noncomputable def g (n : ℕ) : ℝ := f (n^2) - f (n - 1)

theorem g_two_gt_one_third : g 2 > 1 / 3 :=
sorry

theorem g_n_gt_one_third (n : ℕ) (h : n ≥ 3) : g n > 1 / 3 :=
sorry

end g_two_gt_one_third_g_n_gt_one_third_l78_78231


namespace correct_equation_l78_78842

theorem correct_equation (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by
  sorry

end correct_equation_l78_78842


namespace fraction_satisfactory_is_two_thirds_l78_78287

-- Total number of students with satisfactory grades
def satisfactory_grades : ℕ := 3 + 7 + 4 + 2

-- Total number of students with unsatisfactory grades
def unsatisfactory_grades : ℕ := 4

-- Total number of students
def total_students : ℕ := satisfactory_grades + unsatisfactory_grades

-- Fraction of satisfactory grades
def fraction_satisfactory : ℚ := satisfactory_grades / total_students

theorem fraction_satisfactory_is_two_thirds :
  fraction_satisfactory = 2 / 3 := by
  sorry

end fraction_satisfactory_is_two_thirds_l78_78287


namespace max_snowmen_l78_78525

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78525


namespace min_coins_needed_l78_78827

-- Definitions for coins
def coins (pennies nickels dimes quarters : Nat) : Nat :=
  pennies + nickels + dimes + quarters

-- Condition: minimum number of coins to pay any amount less than a dollar
def can_pay_any_amount (pennies nickels dimes quarters : Nat) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount < 100 →
  ∃ (p n d q : Nat), p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧
  p + 5 * n + 10 * d + 25 * q = amount

-- The main Lean 4 statement
theorem min_coins_needed :
  ∃ (pennies nickels dimes quarters : Nat),
    coins pennies nickels dimes quarters = 11 ∧
    can_pay_any_amount pennies nickels dimes quarters :=
sorry

end min_coins_needed_l78_78827


namespace find_other_number_l78_78840

theorem find_other_number (x : ℕ) (hx : x > 0) (h : 100 % x = 4) : ∃ Y : ℕ, Y % x = 4 ∧ Y = 100 + x :=
by
  use 100 + x
  split
  focus
    · exact Nat.mod_add_mod 100 x
  focus
    · exact rfl

end find_other_number_l78_78840


namespace skateboard_distance_l78_78145

theorem skateboard_distance (
    a_1 : ℕ := 8,
    d : ℕ := 10,
    n : ℕ := 20
) : let a_n := λ (n : ℕ), a_1 + (n - 1) * d in
    let S_n := (n * (a_1 + (a_1 + (n - 1) * d))) / 2 in
    S_n = 2060 :=
by
  sorry

end skateboard_distance_l78_78145


namespace three_points_same_color_l78_78891

theorem three_points_same_color (line : ℕ → Prop) (colored : line → Prop) :
  ∃ (A B C : ℕ), colored A = colored B ∧ colored B = colored C ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (B - A) = (C - B) :=
sorry

end three_points_same_color_l78_78891


namespace rice_containers_l78_78762

theorem rice_containers (pound_to_ounce : ℕ) (total_rice_lb : ℚ) (container_oz : ℕ) : 
  pound_to_ounce = 16 → 
  total_rice_lb = 33 / 4 → 
  container_oz = 33 → 
  (total_rice_lb * pound_to_ounce) / container_oz = 4 :=
by sorry

end rice_containers_l78_78762


namespace largest_whole_x_l78_78424

theorem largest_whole_x : 
  ∃ x : ℕ, (∀ y : ℕ, y ≤ x → (1 / 4 : ℝ) + y / 5 < 9 / 10) ∧ 
           (∀ z : ℕ, (1 / 4 : ℝ) + ((x + 1 : ℕ) : ℝ) / 5 ≥ 9 / 10) :=
begin
  sorry
end

end largest_whole_x_l78_78424


namespace neg_prop_p_l78_78257

theorem neg_prop_p :
  (¬ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end neg_prop_p_l78_78257


namespace exist_indices_for_sequences_l78_78261

open Nat

theorem exist_indices_for_sequences 
  (a b c : ℕ → ℕ) : 
  ∃ p q, p ≠ q ∧ p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
  sorry

end exist_indices_for_sequences_l78_78261


namespace question_I_question_II_l78_78994

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := f x a + (1 / 2) * x^2 - b * x

theorem question_I (a : ℝ) : 
  (f⟮1, a⟯ = x + a * Real.log x .1 = 1 + a = 2) → a = 1 := sorry

theorem question_II (a : ℝ) (b : ℝ) (x₁ x₂ : ℝ) (hx : x₁ < x₂) : 
  b ≥ 7 / 2 → (g x₁ a b - g x₂ a b) = (Real.log (x₁ / x₂) - (1 / 2) * (x₁^2 - x₂^2)) = ((b - 1) * (x₁ - x₂)) →
  x₁ + x₂ = b - 1 → x₁ * x₂ = 1 → x₂ > x₁ → h x₁ x₂ ≥ h (1 / 4) = 15 / 8 - 2 * Real.log 2 := sorry
  where 
    h := λ t, Real.log t - (1 / 2) * (t - 1 / t)

end question_I_question_II_l78_78994


namespace find_number_l78_78041

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l78_78041


namespace negative_reciprocal_smallest_root_l78_78397

theorem negative_reciprocal_smallest_root (x : ℝ) (h : x^2 - 3 * real.abs x - 2 = 0) :
  -1 / (min (x, -x)) = (real.sqrt 17 - 3) / 4 := 
sorry

end negative_reciprocal_smallest_root_l78_78397


namespace perp_bisector_b_value_l78_78393

theorem perp_bisector_b_value : ∃ b : ℝ, (∀ (x y : ℝ), x + y = b) ∧ (x + y = b) ∧ (x = (-1) ∧ y = 2) ∧ (x = 3 ∧ y = 8) := sorry

end perp_bisector_b_value_l78_78393


namespace true_discount_l78_78440

variables (PW BG TD : ℝ)

theorem true_discount (h1 : PW = 576) (h2 : BG = 16) : TD = 96 :=
by 
  -- these are the assumptions directly from conditions
  have h_gain : BG = (TD * TD) / PW, from sorry
  -- 16 = TD² / 576 ⇒ TD² = 16 * 576 ⇒ TD = √(16 * 576) = 96
  sorry

end true_discount_l78_78440


namespace max_distinct_fans_l78_78079

-- Define the problem conditions and the main statement
theorem max_distinct_fans : 
  let n := 6  -- Number of sectors per fan
  let configurations := 2^n  -- Total configurations without considering symmetry
  let unchanged_flips := 8  -- Number of configurations unchanged by flipping
  let distinct_configurations := (configurations - unchanged_flips) / 2 + unchanged_flips 
  in 
  distinct_configurations = 36 := by sorry  # We state the answer based on the provided steps


end max_distinct_fans_l78_78079


namespace division_problem_l78_78045

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l78_78045


namespace minimum_vertices_l78_78149

-- Defining the problem conditions
def division_of_triangle (T : Type) [triangle T] (small_triangles : list (triangle T)) : Prop :=
  small_triangles.length = 1000

-- Theorem stating the minimum number of distinct points for the given conditions
theorem minimum_vertices (T : Type) [triangle T] (small_triangles : list (triangle T))
  (h_div: division_of_triangle T small_triangles) :
  ∃ points : set T, points.card = 503 ∧ ∀ t ∈ small_triangles, ∃ (a b c ∈ points), 
  triangle_vertices t = {a, b, c} :=
sorry

end minimum_vertices_l78_78149


namespace equation_of_line_through_center_and_perpendicular_l78_78957

theorem equation_of_line_through_center_and_perpendicular (l : ℝ → ℝ → Prop) :
  (∀ P, (P.1)^2 + (P.2 - 3)^2 = 4) →
  (∀ Q, (Q.1 + Q.2 + 1 = 0) → (l = fun x y => x - y + 3 = 0))
:= by
  sorry

end equation_of_line_through_center_and_perpendicular_l78_78957


namespace caterer_ordered_225_ice_cream_bars_l78_78105

theorem caterer_ordered_225_ice_cream_bars (x : ℕ) : 
  let total_cost := 200
  let cost_per_ice_cream_bar := 0.60
  let num_sundaes := 125
  let cost_per_sundae := 0.52
  let cost_of_sundaes := num_sundaes * cost_per_sundae
  let cost_of_ice_cream_bars := x * cost_per_ice_cream_bar
  in cost_of_ice_cream_bars + cost_of_sundaes = total_cost → x = 225 :=
by 
  sorry

end caterer_ordered_225_ice_cream_bars_l78_78105


namespace wage_difference_l78_78059

variable (P Q H : ℝ)

-- Condition 1: Candidate P earns 1.5 times of Candidate Q
def condition1 : Prop := P = 1.5 * Q

-- Condition 2: Total pay equations
def condition2 : Prop := P * H = 300
def condition3 : Prop := Q * (H + 10) = 300

-- Theorem: Candidate P’s hourly wage is $5 greater than candidate Q’s hourly wage
theorem wage_difference (h1 : condition1) (h2 : condition2) (h3 : condition3) : P - Q = 5 :=
by
  sorry

end wage_difference_l78_78059


namespace h_eq_zero_iff_b_l78_78926

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x - 7

-- Prove that h(b) = 0 if and only if b = 7/5
theorem h_eq_zero_iff_b (b : ℝ) : h(b) = 0 ↔ b = 7 / 5 := by
  sorry

end h_eq_zero_iff_b_l78_78926


namespace max_snowmen_l78_78486

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78486


namespace vector_OE_l78_78693

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (O A B C D E : V)

-- Definitions of points and midpoints
def midpoint (X Y : V) : V := 1/2 • (X + Y)

-- Conditions
def is_midpoint_D : Prop := D = midpoint B C
def is_midpoint_E : Prop := E = midpoint A D

-- Theorem statement
theorem vector_OE (hD : is_midpoint_D D) (hE : is_midpoint_E E) : 
  E = (1/2 • A) + (1/4 • (B - A)) + (1/4 • (C - A)) := 
sorry

end vector_OE_l78_78693


namespace jessie_lowest_score_to_achieve_average_l78_78717

def jessies_tests : ℕ := 4
def max_points_per_test : ℕ := 120
def scores_first_three_tests : list ℕ := [88, 105, 96]
def desired_average : ℕ := 90
def total_tests : ℕ := 6
def total_points_needed : ℕ := desired_average * total_tests
def points_scored_first_three_tests : ℕ := scores_first_three_tests.sum
def points_needed_last_three_tests : ℕ := total_points_needed - points_scored_first_three_tests
def scores_remaining_tests : list ℕ := [some_test_score_1, some_test_score_2, some_test_score_3]
def max_possible_score : ℕ := max_points_per_test

-- Create a theorem stating the lowest possible score for Jessie to achieve his goal
theorem jessie_lowest_score_to_achieve_average :
  ∃ (some_test_score_1 some_test_score_2 some_test_score_3 : ℕ), 
    some_test_score_1 ≤ max_possible_score ∧ 
    some_test_score_2 ≤ max_possible_score ∧ 
    some_test_score_3 ≤ max_possible_score ∧ 
    some_test_score_1 + some_test_score_2 + some_test_score_3 = points_needed_last_three_tests ∧ 
    min some_test_score_1 (min some_test_score_2 some_test_score_3) = 11 :=
sorry

end jessie_lowest_score_to_achieve_average_l78_78717


namespace profit_percentage_for_A_l78_78144

-- Define the initial constants and conditions
def CP_A : ℝ := 150    -- Cost Price for A
def SP_C : ℝ := 225    -- Selling Price for C
def Profit_B_percentage : ℝ := 0.25  -- Profit percentage for B (25%)

-- Define B's cost price
def CP_B : ℝ := SP_C / (1 + Profit_B_percentage)

-- Define the selling price of B which is same as CP_B
def SP_B : ℝ := CP_B

-- Define profit for A
def Profit_A : ℝ := SP_B - CP_A

-- Define profit percentage for A
def Profit_Percentage_A : ℝ := (Profit_A / CP_A) * 100

-- The proof statement in Lean
theorem profit_percentage_for_A : Profit_Percentage_A = 20 := by
  sorry

end profit_percentage_for_A_l78_78144


namespace max_snowmen_l78_78521

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78521


namespace lisa_goal_l78_78743

theorem lisa_goal: 
  let total_quizzes := 60 in
  let goal_percentage := 0.80 in
  let mid_year_quizzes := 40 in
  let mid_year_a_grades := 28 in
  let remaining_quizzes := total_quizzes - mid_year_quizzes in
  let total_a_needed := goal_percentage * total_quizzes in
  let additional_a_needed := total_a_needed - mid_year_a_grades in
  let max_non_a := remaining_quizzes - additional_a_needed in
  max_non_a = 0 :=
by {
  sorry
}

end lisa_goal_l78_78743


namespace largest_even_digit_multiple_of_9_below_10000_l78_78833

theorem largest_even_digit_multiple_of_9_below_10000 : 
  ∃ (n : ℕ), n < 10000 ∧ (∀ d ∈ (to_digits 10 n), d % 2 = 0) ∧ n % 9 = 0 ∧ (∀ m : ℕ, m < 10000 ∧ (∀ d ∈ (to_digits 10 m), d % 2 = 0) ∧ m % 9 = 0 → m ≤ n) :=
sorry

def to_digits (b n : ℕ) : List ℕ :=
n.digits b

end largest_even_digit_multiple_of_9_below_10000_l78_78833


namespace polar_equation_of_line_chord_length_l78_78641

-- Define the parametric line equations and the polar circle equation.
def line_parametric (t : ℝ) : ℝ × ℝ := (1/2 * t, (Real.sqrt 3) / 2 * t)
def polar_circle (θ : ℝ) : ℝ := 4 * Real.sin θ

-- The polar equation of the line (I)
theorem polar_equation_of_line :
  ∃ θ, θ = Real.pi / 3 :=
begin
  have h1 : ∀ t, line_parametric t = (1/2 * t, (Real.sqrt 3) / 2 * t), from λ t, rfl,
  have h2 : ∀ θ, θ = Real.pi / 3 → line_parametric (2 * Real.tan θ) = (Real.cos θ, Real.sin θ),
  { intro θ,
    intro hθ,
    unfold line_parametric,
    rw [hθ, Real.cos_pi_div_three, Real.sin_pi_div_three, mul_assoc, Real.tan_pi_div_three],
    simp },
  use Real.pi / 3,
  exact eq.refl _,
end

-- The length of the chord cut off by the line (II)
theorem chord_length :
  let ρ := polar_circle,
      l := line_parametric in
  2 * Real.sqrt 3 :=
by sorry

end polar_equation_of_line_chord_length_l78_78641


namespace hypotenuse_square_is_720_l78_78569

theorem hypotenuse_square_is_720 (u v w : ℂ) (s t : ℂ)
  (h_zeroes : ∀ z : ℂ, Polynomial.eval z (Polynomial.Coeff (Polynomial.coeff Q 3) s + Polynomial.Coeff (Polynomial.coeff Q 1) t) = 0 → (z = u ∨ z = v ∨ z = w))
  (h_sum_squares : |u|^2 + |v|^2 + |w|^2 = 400)
  (h_right_triangle : (u - v) * (w - v) = 0) : 
  let k : ℝ := Complex.abs (u - w)
  in k^2 = 720 :=
begin
  sorry
end

end hypotenuse_square_is_720_l78_78569


namespace f_strictly_increasing_solve_inequality_l78_78988

variable (f : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 3

-- Prove monotonicity
theorem f_strictly_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Solve the inequality
theorem solve_inequality (m : ℝ) : -2/3 < m ∧ m < 2 ↔ f (3 * m^2 - m - 2) < 2 := by
  sorry

end f_strictly_increasing_solve_inequality_l78_78988


namespace complex_modulus_multiplication_l78_78565

theorem complex_modulus_multiplication 
  (z1 z2 : ℂ)
  (h1 : z1 = 4 - 3 * complex.I)
  (h2 : z2 = 4 + 3 * complex.I) :
  complex.abs z1 * complex.abs z2 = 25 := by
  sorry

end complex_modulus_multiplication_l78_78565


namespace minimum_area_of_triangle_l78_78787

noncomputable def area_of_triangle (k : ℝ) : ℝ :=
  0.5 * (k + (1/k) + 2)

theorem minimum_area_of_triangle : ∀ k > 0, area_of_triangle k = 2 ↔ k = 1 := by
begin
  intros k hk,
  split,
  { 
    intro h,
    rw [area_of_triangle, ← h],
    apply sorry
  },
  {
      intro hk,
      rw [hk],
      simp [area_of_triangle],
      apply sorry
  }
end

end minimum_area_of_triangle_l78_78787


namespace length_of_segment_AB_value_of_m_l78_78616

-- Definitions from the conditions in the problem
def quadratic_function (a m x : ℝ) : ℝ := a * (x - m)^2 + a * (x - m)

-- Condition: a > 0 and m > 0
variables (a m : ℝ)
variables (h_a_pos : a > 0)
variables (h_m_pos : m > 0)

-- Theorem for part (1): The length of segment AB is 1
theorem length_of_segment_AB 
  (x1 x2 : ℝ) 
  (hx1 : x1 = m) 
  (hx2 : x2 = m - 1) : 
  (abs (x1 - x2) = 1) :=
sorry

-- Given vertex C and intersection at positive y-axis, and given the area condition
-- Prove that m = 3/2
theorem value_of_m 
  (h_vertex_C : (quadratic_function a m (- (2 * m + 1) / (2 * a)) = - (m * (m + 1))^2 / (4 * a)))
  (h_intersect_positive_y : (quadratic_function a m 0 = a * m * (m - 1))) 
  (h_area_condition : S_triang ABC = 1/3 * S_triang ABD) :
  m = 3 / 2 :=
sorry

end length_of_segment_AB_value_of_m_l78_78616


namespace fractional_equation_solution_l78_78805

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l78_78805


namespace prime_divisors_1050_l78_78651

theorem prime_divisors_1050 : ∃ (n : ℕ), n = 4 ∧ ∀ p : ℕ, prime p → p ∣ 1050 → p ∈ {2, 3, 5, 7} :=
by
  sorry

end prime_divisors_1050_l78_78651


namespace apples_per_pie_l78_78560

-- Conditions
def initial_apples : ℕ := 50
def apples_per_teacher_per_child : ℕ := 3
def number_of_teachers : ℕ := 2
def number_of_children : ℕ := 2
def remaining_apples : ℕ := 24

-- Proof goal: the number of apples Jill uses per pie
theorem apples_per_pie : 
  initial_apples 
  - (apples_per_teacher_per_child * number_of_teachers * number_of_children)  - remaining_apples = 14 -> 14 / 2 = 7 := 
by
  sorry

end apples_per_pie_l78_78560


namespace milk_replacement_problem_l78_78894

theorem milk_replacement_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 90)
  (h3 : (90 - x) - ((90 - x) * x / 90) = 72.9) : x = 9 :=
sorry

end milk_replacement_problem_l78_78894


namespace log_diff_l78_78370

theorem log_diff (N M : ℕ) (n m : ℤ) 
  (hN : ∃ k, 10^n ≤ k ∧ k < 10^(n+1) ∧ N = 9 * 10^n) 
  (hM : ∃ k, 10^m ≤ k ∧ k < 10^(m+1) ∧ M = 9 * 10^(m-1)) :
  (log 10 N - log 10 M = n - m + 1) :=
sorry

end log_diff_l78_78370


namespace sin_240_eq_neg_sqrt3_div_2_l78_78580

open Real

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 * π / 180) = -sqrt 3 / 2 :=
by
  -- using the known trigonometric identity
  have h1 : ∀ α, sin (180 * π / 180 + α) = -sin α :=
    by -- identity proof would go here
   sorry,
  -- using the known value of sin 60 degrees
  have h2 : sin (60 * π / 180) = sqrt 3 / 2 :=
    by -- value proof would go here
   sorry,
  -- now, applying these to prove the original statement
  calc
    sin (240 * π / 180)
        = sin (180 * π / 180 + 60 * π / 180) : by ring
    ... = -sin (60 * π / 180)                : by rw [h1]
    ... = -sqrt 3 / 2                        : by rw [h2]

end sin_240_eq_neg_sqrt3_div_2_l78_78580


namespace parallel_condition_l78_78646

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4 * x)

-- Define the condition for parallelism for two-dimensional vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Define the theorem to prove
theorem parallel_condition (x : ℝ) :
  parallel (vector_a x) (vector_b x) ↔ |x| = 2 :=
by {
  sorry
}

end parallel_condition_l78_78646


namespace sum_of_radii_eq_radius_l78_78919

variables {S S₁ S₂ : Type}
variables {O O₁ O₂ A B C : Type}
variables {r R : ℝ}

-- Assume we have circles S, S₁, S₂ with radii R, r₁, r₂ respectively.
-- Let O, O₁, O₂ be the centers of S, S₁ and S₂ respectively.
-- Let A and B be the points where S₁ and S₂ touch S respectively.
-- Let C be an intersection point of S₁ and S₂ lying on segment AB.

structure Circle (center : Type) (radius : ℝ) :=
(center : Type)
(radius : ℝ)

variable (O O₁ O₂)

def Circle_S := Circle O R
def Circle_S₁ := Circle O₁ r₁
def Circle_S₂ := Circle O₂ r₂

variables (A B : Type)
variable (C : Type)

-- Assume the tangency conditions and the intersection condition.
axiom circle_tangency_S₁ (t1 : ∀ A, Circle_S₁.center = A ) : ∀ A B, A = B
axiom circle_tangency_S₂ (t2 : ∀ B, Circle_S₂.center = B) : ∀ B A, B = A
axiom intersection_condition (ic : C ∈ [A, B])

theorem sum_of_radii_eq_radius : R = r₁ + r₂ :=
sorry

end sum_of_radii_eq_radius_l78_78919


namespace ivan_income_tax_l78_78712

-- Define the salary schedule
def first_two_months_salary: ℕ := 20000
def post_probation_salary: ℕ := 25000
def bonus_in_december: ℕ := 10000
def income_tax_rate: ℝ := 0.13

-- Define the total taxable income
def total_taxable_income: ℕ :=
  (first_two_months_salary * 2) + (post_probation_salary * 8) + bonus_in_december

-- Define the expected tax amount
def expected_tax: ℕ := 32500

-- Define the personal income tax calculation function
def calculate_tax (income: ℕ) (rate: ℝ): ℕ :=
  (income * rate).toInt

-- The statement which shows that the calculated tax is equal to the expected tax
theorem ivan_income_tax: calculate_tax total_taxable_income income_tax_rate = expected_tax := by
  -- Skip the actual proof
  sorry

end ivan_income_tax_l78_78712


namespace eccentricity_of_hyperbola_l78_78639

variable {a b c e : ℝ}
variable (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h_vertices : A1 = (-a, 0) ∧ A2 = (a, 0))
variable (h_imaginary_axis : B1 = (0, b) ∧ B2 = (0, -b))
variable (h_foci : F1 = (-c, 0) ∧ F2 = (c, 0))
variable (h_relation : a^2 + b^2 = c^2)
variable (h_tangent_circle : ∀ d, (d = 2*a) → (tangent (circle d) (rhombus F1 B1 F2 B2)))

theorem eccentricity_of_hyperbola : e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l78_78639


namespace distance_inequality_l78_78721

variables {A B C H : Type} [triangle : Triangle A B C] 
variables (d_a d_b d_c R r : ℝ)
variables [orthocenter H A B C] [distances_from_orthocenter H A B C to_rays d_a d_b d_c]
variables [circumradius R A B C] [inradius r A B C]

theorem distance_inequality :
  d_a + d_b + d_c ≤ (3 * R^2) / (4 * r) :=
sorry

end distance_inequality_l78_78721


namespace area_enclosed_y_eq_kx_y_eq_xsq_l78_78629
-- Additional imports as required by the Lean system to work seamlessly

noncomputable def k_slope : ℝ :=
  deriv (λ x, real.exp (2 * x)) 0

theorem area_enclosed_y_eq_kx_y_eq_xsq :
  (k_slope = 2) → ∫ x in 0..2, (2 * x - x ^ 2) = 4 / 3 :=
begin
  -- Proof not included, only statement as per the requirements
  sorry
end

end area_enclosed_y_eq_kx_y_eq_xsq_l78_78629


namespace ellipse_eqn_max_area_triangle_l78_78972

/-- Given that O is the coordinate origin, C is an ellipse with the equation
    x^2/a^2 + y^2/b^2 = 1 (a > b > 0), with left and right foci F₁, F₂, upper vertex P, and right vertex Q.
    A circle at O with diameter F₁F₂ is tangent to the ellipse C, and the chord length of
    the intersection between the line PQ and the circle is 2sqrt(3)/3. -/
variables {a b : ℝ} (F₁ F₂ P Q : ℝ × ℝ)

/-- Prove (I): The standard equation of the ellipse C is x^2/2 + y^2 = 1. -/
theorem ellipse_eqn 
  (h1: a > b)
  (h2: b > 0)
  (h3: a = sqrt 2)
  (h4: b = 1)
  (h5: ∀ x y, (x - F₁.1)^2 + y^2 = b^2 → (x - F₂.1)^2 + y^2 = b^2)
  (h6: (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = (2*b)^2)
  : (∀ x y, x^2 / 2 + y^2 = 1) := sorry

variables {O : ℝ × ℝ} {A B : ℝ × ℝ} {m n : ℝ}

/-- Prove (II): If a line l is tangent to the circle O with diameter F₁F₂ and intersects the ellipse C 
    at two distinct points A, B, then the maximum area of triangle OAB is sqrt(2)/2.
    Also, the equation of the line l when the area is maximum is x = ±1. -/
theorem max_area_triangle
  (h1: ∀ x y, x^2 + y^2 = 1)
  (h2: ∀ x y, x/my + n = 1)
  (h3: ∀ x y, x^2 / 2 + y^2 = 1)
  (h4: ∀ x y, (x - F₁.1)^2 + y^2 = b^2 → (x - F₂.1)^2 + y^2 = b^2)
  (h5: (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = (2*b)^2)
  : (∃ m n, S = sqrt(2)/2 ∧ ∀ x y, [x = 1] ∨ [x = -1]) := sorry

end ellipse_eqn_max_area_triangle_l78_78972


namespace average_score_for_boys_combined_l78_78559

theorem average_score_for_boys_combined (L l M m : ℕ) 
(h1 : 65 * L + 70 * l = 68 * (L + l))
(h2 : 75 * M + 85 * m = 78 * (M + m))
(h3 : 70 * l + 85 * m = 80 * (l + m)) :
  (65 * L + 75 * M) / (L + M) = 24.58 := 
sorry

end average_score_for_boys_combined_l78_78559


namespace coefficient_x2_term_l78_78198

theorem coefficient_x2_term : 
  (Let e := (2 + x) * (1 - 2 * x)^5 in 
   Let e_exp := e.expand in
   coefficient e_exp 2 = 70) :=
sorry

end coefficient_x2_term_l78_78198


namespace speed_of_B_l78_78450

theorem speed_of_B {v : ℝ} (hv : v = 4) :
  ∃ A B : ℝ, (A = 6) ∧ (B = v) ∧ (A * 5 + B * 5 = 50) :=
by
  use [6, 4]
  split
  repeat { sorry }

end speed_of_B_l78_78450


namespace player_paired_with_one_is_fifteen_l78_78100

-- Define the problem conditions
structure PlayerPairs :=
(n : ℕ)
(players : Fin n → Fin n)
(sum_is_square : ∀ i, i < n → 
  ∃ k, ((players i).val + (Fin.ofNat i).val = k^2))

-- Define the main problem
theorem player_paired_with_one_is_fifteen (pairs : PlayerPairs) 
  (h_pairs : pairs.n = 9) 
  (h_players : players (Fin.ofNat 1) = Fin.ofNat 15) 
  : true := by
  sorry

end player_paired_with_one_is_fifteen_l78_78100


namespace geric_initial_bills_l78_78608

theorem geric_initial_bills :
  ∀ (bills_jessa : ℕ) (bills_kylan: ℕ) (bills_geric : ℕ),
  bills_jessa = 10 →
  bills_kylan = bills_jessa - 2 →
  bills_geric = 2 * bills_kylan →
  bills_geric = 16 :=
by
  intros bills_jessa bills_kylan bills_geric h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  rw [h3]
  sorry

end geric_initial_bills_l78_78608


namespace find_f_2017_l78_78784

noncomputable def f (x : ℝ) (a b α β : ℝ) : ℝ := a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 1008

theorem find_f_2017 (a b α β : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0) (h_nonzero_alpha : α ≠ 0) (h_nonzero_beta : β ≠ 0) (h_f_2016 : f 2016 a b α β = 16) : f 2017 a b α β = 2000 :=
by
  sorry

end find_f_2017_l78_78784


namespace selection_with_surgical_and_psychological_therapist_selection_with_at_least_two_surgical_experts_and_constraint_l78_78677

-- Given definitions:
def nursing_experts := {A1, A2, A3}
def surgical_experts := {B1, B2, B3, B4, B5}
def psychological_therapy_experts := {C1, C2}

-- Problem (1)
theorem selection_with_surgical_and_psychological_therapist (h1 : true) :
  {n : ℕ // n = 30} :=
by {
  -- proof will go here
  sorry,
}

-- Problem (2)
theorem selection_with_at_least_two_surgical_experts_and_constraint (h2 : true) :
  {n : ℕ // n = 133} :=
by {
  -- proof will go here
  sorry,
}

end selection_with_surgical_and_psychological_therapist_selection_with_at_least_two_surgical_experts_and_constraint_l78_78677


namespace waterDepthWhenUprightIsCorrect_l78_78886

-- Define the problem data
def waterTankHeight : Float := 20.0
def diameter : Float := 5.0
def waterDepthWhenHorizontal : Float := 2.0

-- Define the radius
def radius := diameter / 2

-- Define the volume of water when tank is horizontal
noncomputable def horizontalSegmentArea :=
  let theta := 2 * Real.acos ((radius - waterDepthWhenHorizontal) / radius)
  (radius ^ 2 / 2) * (theta - Real.sin theta)

noncomputable def volumeOfWater :=
  horizontalSegmentArea * waterTankHeight

-- Define the depth of the water when tank is upright
noncomputable def uprightWaterDepth :=
  volumeOfWater / (Real.pi * radius ^ 2)

-- Proof statement
theorem waterDepthWhenUprightIsCorrect :
  uprightWaterDepth ≈ 12.1 := sorry

end waterDepthWhenUprightIsCorrect_l78_78886


namespace largest_c_for_minus3_in_range_of_quadratic_l78_78201

theorem largest_c_for_minus3_in_range_of_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 5*x + c = -3) ↔ c ≤ 13/4 :=
sorry

end largest_c_for_minus3_in_range_of_quadratic_l78_78201


namespace find_fraction_value_l78_78219

theorem find_fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : (m * n) / (m - n) = -1/6 :=
sorry

end find_fraction_value_l78_78219


namespace number_of_divisors_of_N_cubed_l78_78053

-- Define distinct prime numbers and exponents
variables {p q : ℕ} [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (λ₁ λ₂ : ℕ)

-- Conditions
def N := p^λ₁ * q^λ₂
def N_sq_divisors := (2 * λ₁ + 1) * (2 * λ₂ + 1) = 15

-- Proof statement
theorem number_of_divisors_of_N_cubed (h1 : N_sq_divisors λ₁ λ₂) : 
  (1 + 3 * λ₁) * (1 + 3 * λ₂) = 28 := 
by 
  sorry

end number_of_divisors_of_N_cubed_l78_78053


namespace max_snowmen_l78_78510

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78510


namespace find_m_l78_78245

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 2 * x

noncomputable def line_L (x y m: ℝ) : Prop := x = sqrt 3 * y + m

noncomputable def point_P (m: ℝ) : ℝ × ℝ := (m, 0)

theorem find_m (m : ℝ) (HA HB : ℝ × ℝ)
  (hA : curve_C HA.1 HA.2)
  (hB : curve_C HB.1 HB.2)
  (hL1 : line_L HA.1 HA.2 m)
  (hL2 : line_L HB.1 HB.2 m)
  (hP : point_P m)
  (hPA_PB : abs ((HA.1 - hP.1) * (HB.1 - hP.1) + 
                  (HA.2 - hP.2) * (HB.2 - hP.2)) = 1) :
  m = 1 + sqrt 2 ∨ m = 1 - sqrt 2 :=
sorry

end find_m_l78_78245


namespace remainder_when_divided_by_13_l78_78454

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) : (N = 39 * k + 17) → (N % 13 = 4) := by
  sorry

end remainder_when_divided_by_13_l78_78454


namespace length_of_AC_l78_78455

-- Define the lengths of the sides of the cyclic quadrilateral ABCD
def AB := 3
def BC := 4
def CD := 5
def AD := 2

-- Define the problem: Find the length of AC
def find_AC : ℝ := Real.sqrt (299 / 11)

-- Lean statement for the problem
theorem length_of_AC (AB BC CD AD : ℝ) (hAB : AB = 3) (hBC : BC = 4) (hCD : CD = 5) (hAD : AD = 2) :
  (∃ AC : ℝ, AC = find_AC) :=
by
  sorry

end length_of_AC_l78_78455


namespace circle_cos_max_intersections_l78_78269

noncomputable def max_circle_cos_intersections : ℕ :=
  2

theorem circle_cos_max_intersections :
  let f := λ x, (x - 1) ^ 2 + (Real.cos x) ^ 2
  ∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 * Real.pi →
  (x1, Real.cos x1) ≠ (x2, Real.cos x2) ∧
  f x1 = 4 ∧ f x2 = 4 :=
by sorry

end circle_cos_max_intersections_l78_78269


namespace age_ratio_l78_78577

theorem age_ratio (darcie_age : ℕ) (father_age : ℕ) (mother_ratio : ℚ) (mother_fraction : ℚ)
  (h1 : darcie_age = 4)
  (h2 : father_age = 30)
  (h3 : mother_ratio = 4/5)
  (h4 : mother_fraction = mother_ratio * father_age)
  (h5 : mother_fraction = 24) :
  (darcie_age : ℚ) / mother_fraction = 1 / 6 :=
by
  sorry

end age_ratio_l78_78577


namespace find_constants_for_matrix_condition_l78_78320

noncomputable section

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3], ![0, 1, 2], ![1, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℝ :=
  1

theorem find_constants_for_matrix_condition :
  ∃ p q r : ℝ, B^3 + p • B^2 + q • B + r • I = 0 :=
by
  use -5, 3, -6
  sorry

end find_constants_for_matrix_condition_l78_78320


namespace largest_25_supporting_X_l78_78122

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i : ℝ) ∈ Set.Ico (- (1 / 2) : ℝ) ((25 : ℝ) / 2 + (1 / 2)) → 
  ∃ i, |a i - 1 / 2| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ 
  (∀ Y : ℝ, Y > X → ¬is_25_supporting Y) ∧ X = 0.02 :=
sorry

end largest_25_supporting_X_l78_78122


namespace select_twins_in_grid_l78_78818

theorem select_twins_in_grid (persons : Fin 8 × Fin 8 → Fin 2) :
  ∃ (selection : Fin 8 × Fin 8 → Bool), 
    (∀ i : Fin 8, ∃ j : Fin 8, selection (i, j) = true) ∧ 
    (∀ j : Fin 8, ∃ i : Fin 8, selection (i, j) = true) :=
sorry

end select_twins_in_grid_l78_78818


namespace infinite_primes_composite_sum_l78_78340

theorem infinite_primes_composite_sum (m : ℕ) (hm : 0 < m) : 
  ∃ᶠ p in filter (λ p, Prime p) (Ico 1 1000000), ¬ Prime (m + p^3) :=
by sorry

end infinite_primes_composite_sum_l78_78340


namespace quadrilateral_is_rectangle_l78_78617

-- Definition of a quadrilateral with equal incircle radii
structure Quadrilateral (A B C D : Type) :=
(equal_incircle_radii : rABC = rBCD ∧ rBCD = rCDA ∧ rCDA = rDAB)
  [Triangle : Type]
  (incircle_radius : Triangle → ℝ)
  (rABC rBCD rCDA rDAB : ℝ)
  (ABC BCD CDA DAB: Triangle)

-- Theorem statement
theorem quadrilateral_is_rectangle 
    (A B C D : Type) 
    [Quadrilateral A B C D]
    (equal_incircle_radii : Quadrilateral.equal_incircle_radii A B C D) 
    : is_rectangle A B C D := 
by
  sorry

end quadrilateral_is_rectangle_l78_78617


namespace cistern_fill_time_l78_78055

/-- A tap can fill the cistern in 5 hours. -/
def R_fill : ℝ := 1 / 5

/-- Another tap can empty the cistern in 6 hours. -/
def R_empty : ℝ := 1 / 6

/-- The net rate of filling the cistern when both taps are opened. -/
def R_net : ℝ := R_fill - R_empty

/-- The time it takes to fill the cistern when both taps are opened. -/
def time_to_fill : ℝ := 1 / R_net

/-- Proof that if both taps are opened simultaneously, the cistern will get filled in 30 hours. -/
theorem cistern_fill_time : time_to_fill = 30 := sorry

end cistern_fill_time_l78_78055


namespace find_p_l78_78147

-- Defining the points A, B, C, D, and E
def A := (0, 0, 0)
def B := (4, 0, 0)
def C := (4, 4, 0)
def D := (0, 4, 0)
def E := (2, 2, 2 * Real.sqrt 2)

-- Defining the midpoints R, S, and T
def R := ((0 + 2) / 2, (0 + 2) / 2, (0 + 2 * Real.sqrt 2) / 2)
def S := ((4 + 4) / 2, (0 + 4) / 2, (0 + 0) / 2)
def T := ((4 + 0) / 2, (4 + 4) / 2, (0 + 0) / 2)

-- Area of the intersection in the plane is expressed as sqrt p
def plane_intersection_area := Real.sqrt 80

theorem find_p : 
  let square_pyramid := (A, B, C, D, E)
  ∧ (midpoints := (R, S, T))
  ∧ (plane := { P : ℝ × ℝ × ℝ | P.1 + P.2 + 2 * Real.sqrt 2 * P.3 = 6 }) 
  ∧ (intersection := { P : ℝ × ℝ × ℝ | plane P ∧ (in_pyramid P square_pyramid) })  -- need in_pyramid predicate
  ∧ (area intersection = Real.sqrt p)
  ⇒ p = 80 := 
by 
  sorry

end find_p_l78_78147


namespace log_series_identity_l78_78852

theorem log_series_identity (n : ℕ) : 
  ln (n + 1) = ln n + 2 * ∑ i in (Set.Ioi 0).filter (λ x, odd x), (1 : ℝ) / (i * (2 * n + 1) ^ i) := 
sorry

end log_series_identity_l78_78852


namespace distinct_fan_count_l78_78086

def max_distinct_fans : Nat :=
  36

theorem distinct_fan_count (n : Nat) (r b : S) (paint_scheme : Fin n → bool) :
  (∀i, r ≠ b → (paint_scheme i = b ∨ paint_scheme i = r)) ∧ 
  (∀i, paint_scheme i ≠ paint_scheme (i + n / 2 % n)) →
  n = 6 →
  max_distinct_fans = 36 :=
by
  sorry

end distinct_fan_count_l78_78086


namespace factor_quadratic_l78_78584

theorem factor_quadratic (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := 
by 
  sorry

end factor_quadratic_l78_78584


namespace correct_propositions_l78_78342

-- Definitions of planes and lines assuming some basic properties and relations
variable {α β γ : Plane}
variable {a b : Line}

def is_parallel (l : Line) (p : Plane) : Prop :=
sorry   -- logic to define when a line is parallel to a plane

def is_perpendicular (l : Line) (p : Plane) : Prop :=
sorry   -- logic to define when a line is perpendicular to a plane

def is_parallel_lines (l1 l2 : Line) : Prop :=
sorry   -- logic to define when two lines are parallel

def is_perpendicular_lines (l1 l2 : Line) : Prop :=
sorry   -- logic to define when two lines are perpendicular

def projections_perpendicular_on_plane (l1 l2 : Line) (p : Plane) : Prop :=
sorry   -- logic to define when projections of two lines on a plane are perpendicular

-- Propositions as given in the problem

def proposition_1 : Prop :=
∀ (a b : Line) (α : Plane),
is_parallel a α → is_parallel b α → is_parallel_lines a b

def proposition_2 : Prop :=
∀ (a b : Line) (α β : Plane),
is_parallel a α → is_parallel b β → is_parallel_lines a b → is_parallel α β

def proposition_3 : Prop :=
∀ (a b : Line) (α β : Plane),
is_perpendicular a α → is_perpendicular b β → is_perpendicular_lines a b → is_perpendicular α β

def proposition_4 : Prop :=
∀ (a b : Line) (α : Plane),
projections_perpendicular_on_plane a b α → is_perpendicular_lines a b

-- Main theorem to state which propositions are correct

theorem correct_propositions :
  proposition_3 ∧ ¬proposition_1 ∧ ¬proposition_2 ∧ ¬proposition_4 :=
by {
  -- Proof should be provided later
  sorry
}

end correct_propositions_l78_78342


namespace determine_constants_l78_78727

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B P : V}

theorem determine_constants (h_on_segment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B)
  (h_ratio : ∀ P : V, (P - A) = 5 • (B - P) → (P - A) = (3 : ℝ) • (P - B)) :
  ∃ t u : ℝ, t = 3/8 ∧ u = 5/8 ∧ P = t • A + u • B :=
by
  use [3/8, 5/8]
  split
  case left => rfl
  case right => split
  case left => rfl
  case right =>
    -- This is where we would normally prove the equality P = (3/8) • A + (5/8) • B.
    sorry

end determine_constants_l78_78727


namespace complex_conjugate_problem_l78_78588

theorem complex_conjugate_problem (z : ℂ) (h : (1 + complex.I) * z = 3 + complex.I) : 
  complex.conj z = 2 + complex.I :=
begin
  sorry,
end

end complex_conjugate_problem_l78_78588


namespace max_snowmen_l78_78523

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78523


namespace largest_square_area_l78_78385

theorem largest_square_area (a b c : ℝ) 
  (h1 : c^2 = a^2 + b^2) 
  (h2 : a = b - 5) 
  (h3 : a^2 + b^2 + c^2 = 450) : 
  c^2 = 225 :=
by 
  sorry

end largest_square_area_l78_78385


namespace obtuse_angle_inclination_l78_78700

/-- Prove that the line with equation (x/2) + (y/3) = 1 has an obtuse angle of inclination,
given the equations of the other lines. -/
theorem obtuse_angle_inclination (A : ℝ → ℝ) (B : ∀ x, x = -2) 
(C : ℝ → ℝ) (D : ℝ → ℝ) : 
 (C = λ x, - (3 / 2) * x + 3) → 
 ((∃ x, A x = 3 * x - 1) ∧ (∀ x, B x → x = -2) ∧ (∃ x, C x = - (3 / 2) * x + 3) ∧ (∃ x, D x = 2 * x - 1))
→ C = λ x, - (3 / 2) * x + 3 :=
begin
  sorry
end

end obtuse_angle_inclination_l78_78700


namespace james_spends_on_pistachios_per_week_l78_78713

theorem james_spends_on_pistachios_per_week :
  let cost_per_can := 10
  let ounces_per_can := 5
  let total_ounces_per_5_days := 30
  let days_per_week := 7
  let cost_per_ounce := cost_per_can / ounces_per_can
  let daily_ounces := total_ounces_per_5_days / 5
  let daily_cost := daily_ounces * cost_per_ounce
  daily_cost * days_per_week = 84 :=
by
  sorry

end james_spends_on_pistachios_per_week_l78_78713


namespace remainder_when_squared_l78_78857

theorem remainder_when_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end remainder_when_squared_l78_78857


namespace new_class_mean_l78_78284

theorem new_class_mean 
  (n1 n2 : ℕ) (mean1 mean2 : ℚ) 
  (h1 : n1 = 24) (h2 : n2 = 8) 
  (h3 : mean1 = 85/100) (h4 : mean2 = 90/100) :
  (n1 * mean1 + n2 * mean2) / (n1 + n2) = 345/400 :=
by
  rw [h1, h2, h3, h4]
  sorry

end new_class_mean_l78_78284


namespace maximize_area_l78_78019

noncomputable def max_area : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area (l w : ℝ) (h1 : 2 * l + 2 * w = 400) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  (l * w ≤ 10000) :=
sorry

end maximize_area_l78_78019


namespace sum_cosine_roots_of_unity_l78_78209

theorem sum_cosine_roots_of_unity (n : ℕ) (α : ℝ) :
  ∑ k in finset.range (2 * n + 1), real.cos (α + (2 * ↑k * real.pi / (2 * n + 1))) = 0 := 
sorry

end sum_cosine_roots_of_unity_l78_78209


namespace evaluate_at_minus_two_l78_78632

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x + 3)
  else x * (x - 3)

theorem evaluate_at_minus_two : f (-2) = 10 := by
  sorry

end evaluate_at_minus_two_l78_78632


namespace number_of_valid_three_digit_numbers_l78_78270

def is_digit (n : ℕ) : Prop := n < 10

def is_valid_set (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  is_digit a ∧ is_digit b ∧ is_digit c ∧
  a + b = c ∧ 
  a ≠ 0 ∧ b ≠ 0

theorem number_of_valid_three_digit_numbers : 
  (finset.univ.filter (λ n : ℕ × ℕ × ℕ, is_valid_set n.1 n.2 n.3)).card = 90 :=
by 
  sorry

end number_of_valid_three_digit_numbers_l78_78270


namespace distinct_fan_count_l78_78088

def max_distinct_fans : Nat :=
  36

theorem distinct_fan_count (n : Nat) (r b : S) (paint_scheme : Fin n → bool) :
  (∀i, r ≠ b → (paint_scheme i = b ∨ paint_scheme i = r)) ∧ 
  (∀i, paint_scheme i ≠ paint_scheme (i + n / 2 % n)) →
  n = 6 →
  max_distinct_fans = 36 :=
by
  sorry

end distinct_fan_count_l78_78088


namespace monotonic_decreasing_interval_l78_78395

def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∃ a b : ℝ, 0 < a ∧ a ≤ x ∧ x ≤ b ∧ b = 1) ↔ (f' x ≤ 0) := sorry

end monotonic_decreasing_interval_l78_78395


namespace problem_statement_l78_78732

noncomputable def relatively_prime_positive_integers (m n : ℕ) : Prop :=
Nat.Coprime m n ∧ m > 0 ∧ n > 0

theorem problem_statement :
  ∃ (m n : ℕ), relatively_prime_positive_integers m n ∧
  ((probability (λ b : ℝ, b ∈ set.Icc (-9 : ℝ) 9 ∧ has_two_distinct_real_solutions (b : ℝ)) = (m : ℚ) / (n : ℚ)) ∧
  m + n = 152) :=
sorry

/-- Helper predicate to express when the equation x^3 + 45b^2 = (9b^2 - 12b)x^2 has at least two distinct real solutions -/
def has_two_distinct_real_solutions (b : ℝ) : Prop :=
  let x := 9 * b^2 - 12 * b in
  let discriminant := x^2 - 4 * 45 * b^2 in
  ∃ (y1 y2 : ℝ), y1 ≠ y2 ∧ y1 * y2 = 45 * b^2 ∧ y1 + y2 = x

end problem_statement_l78_78732


namespace integer_subset_property_l78_78233

theorem integer_subset_property (M : Set ℤ) (h1 : ∃ a ∈ M, a > 0) (h2 : ∃ b ∈ M, b < 0)
(h3 : ∀ {a b : ℤ}, a ∈ M → b ∈ M → 2 * a ∈ M ∧ a + b ∈ M)
: ∀ a b : ℤ, a ∈ M → b ∈ M → a - b ∈ M :=
by
  sorry

end integer_subset_property_l78_78233


namespace sum_of_powers_of_i_l78_78447

theorem sum_of_powers_of_i :
  (∑ n in Finset.range 8 | n + 1 • (Complex.I ^ (n + 1))) = 4 - 4 * Complex.I :=
by sorry

end sum_of_powers_of_i_l78_78447


namespace arrangement_meeting_ways_l78_78108

-- For convenience, define the number of members per school and the combination function.
def num_members_per_school : ℕ := 6
def num_schools : ℕ :=  4
def combination (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem arrangement_meeting_ways : 
  let host_ways := num_schools
  let host_reps_ways := combination num_members_per_school 2
  let non_host_schools := num_schools - 1
  let non_host_reps_ways := combination num_members_per_school 2
  let total_non_host_reps_ways := non_host_reps_ways ^ non_host_schools
  let total_ways := host_ways * host_reps_ways * total_non_host_reps_ways
  total_ways = 202500 :=
by 
  -- Definitions and computation is deferred to the steps,
  -- which are to be filled during the proof.
  sorry

end arrangement_meeting_ways_l78_78108


namespace more_philosophers_than_mathematicians_l78_78756

theorem more_philosophers_than_mathematicians
  (m p : ℕ)
  (h1 : p / 9 = m / 7)
  (h2 : p = Int.nat_div (9 * m) 7) :
  p > m :=
by
  sorry

end more_philosophers_than_mathematicians_l78_78756


namespace color_regions_of_circles_l78_78335

theorem color_regions_of_circles (n : ℕ) (h : n ≥ 1) :
  ∃ f : ℝ² → bool, (∀ c : ℝ² → Prop, is_circle c →
    ∀ x y ∈ region_separated_by_circle c, f x ≠ f y) :=
sorry

end color_regions_of_circles_l78_78335


namespace red_crayons_count_l78_78111

variables (total_crayons blue_crayons green_crayons pink_crayons red_crayons : ℕ)

def num_crayons := 24
def num_blue := 6
def num_pink := 6
def num_green := (2 / 3 : ℚ) * num_blue

-- Statement to prove
theorem red_crayons_count (h1 : total_crayons = num_crayons) 
                           (h2 : blue_crayons = num_blue) 
                           (h3 : green_crayons = num_green.to_nat)
                           (h4 : pink_crayons = num_pink) :
    red_crayons = total_crayons - (blue_crayons + green_crayons + pink_crayons) :=
by
    sorry

end red_crayons_count_l78_78111


namespace oliver_final_amount_l78_78355

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l78_78355


namespace sum_of_five_shaded_cells_l78_78399

-- Define the problem conditions
def cells (n : ℕ) := { x // 1 ≤ x ∧ x ≤ 9 }

-- Given conditions
def diagonal1_sum : ℕ :=
  1 + 2 + 4 -- sum of the numbers on one diagonal

def diagonal2_sum : ℕ :=
  9 + 8 + 4 -- sum of the numbers on the other diagonal

def table (σ : matrix (fin 3) (fin 3) ℕ)
  (h1 : diagonal1_sum = 7)
  (h2 : diagonal2_sum = 21)
  (h3 : ∀ i j, σ i j ∈ cells 9)

-- The question to prove
theorem sum_of_five_shaded_cells (σ : matrix (fin 3) (fin 3) ℕ)
  (h1 : diagonal1_sum σ = 7)
  (h2 : diagonal2_sum σ = 21)
  (h3 : ∀ i j, σ i j ∈ cells 9) :
  (σ 0 0) + (σ 0 2) + (σ 1 1) + (σ 2 0) + (σ 2 2) = 25 := 
sorry

end sum_of_five_shaded_cells_l78_78399


namespace intersection_A_B_l78_78970

-- Definitions for sets A and B
def A : Set ℝ := { x | ∃ y : ℝ, x + y^2 = 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

-- The proof goal to show the intersection of sets A and B
theorem intersection_A_B : A ∩ B = { z | -1 ≤ z ∧ z ≤ 1 } :=
by
  sorry

end intersection_A_B_l78_78970


namespace find_point_C_l78_78542

theorem find_point_C :
  ∃ C : ℝ × ℝ, let A : ℝ × ℝ := (-3, 5) in
                 let B : ℝ × ℝ := (9, -1) in
                 let AB := (B.1 - A.1, B.2 - A.2) in
                 C = (B.1 + 0.5 * AB.1, B.2 + 0.5 * AB.2) ∧ 
                 C = (15, -4) :=
by
  sorry

end find_point_C_l78_78542


namespace geometric_sequence_sum_l78_78973

open Nat

-- Conditions of the problem
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q, ∀ n, a (n + 1) = q * a n

def a_2 := 2
def a_5 := 1 / 4

-- Define the sequence {a_n} as a function of n
noncomputable def a : ℕ → ℝ
| 0       := 4            -- a₁ = 4
| 1       := a_2          -- a₂ = 2
| 4       := a_5          -- a₅ = 1 / 4
| (n + 1) := a n * (1 / 2) -- Using q = 1/2 inferred from the solution

-- Prove the required sum equals the correct answer
theorem geometric_sequence_sum :
  ∑ i in Finset.range n, a i * a (i + 1) = (32 / 3) * (1 - 4 ^ (-n)) :=
begin
  sorry
end

end geometric_sequence_sum_l78_78973


namespace common_divisor_of_differences_l78_78276

theorem common_divisor_of_differences 
  (a1 a2 b1 b2 c1 c2 d : ℤ) 
  (h1: d ∣ (a1 - a2)) 
  (h2: d ∣ (b1 - b2)) 
  (h3: d ∣ (c1 - c2)) : 
  d ∣ (a1 * b1 * c1 - a2 * b2 * c2) := 
by sorry

end common_divisor_of_differences_l78_78276


namespace largest_supporting_25_X_l78_78124

def is_supporting_25 (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ X

theorem largest_supporting_25_X :
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ 0.02 :=
by {
  sorry
}

end largest_supporting_25_X_l78_78124


namespace intersection_of_lines_l78_78832

noncomputable def intersection_point : ℚ × ℚ :=
  let x := 1 / 8 in
  let y := -6 * x in
  (x, y)

theorem intersection_of_lines :
  let y1 := -6 * (1 / 8) in
  let y2 := 18 * (1 / 8) - 3 in
  y1 = y2 ∧ intersection_point = (1 / 8, -3 / 4) := by
  sorry

end intersection_of_lines_l78_78832


namespace total_red_cards_l78_78889

theorem total_red_cards (d : ℕ) (h : ∃ k : ℕ, d = 52 * k) (n : ℕ) : 
(∃ m : ℕ, m = 2 * (13 * n)) :=
begin
  sorry
end

def shopkeeper_red_cards (decks : ℕ) (h : decks = 8) : nat :=
  if decks = 8 then 208 else 0

-- Please note that to ‘prove’ the number of red cards, 
-- we assume the existence of a specific count for the particular conditions given.

end total_red_cards_l78_78889


namespace area_of_ABCD_l78_78171

noncomputable def area_ABCD (side_length : ℕ) : ℚ :=
  let circumradius := (side_length : ℚ) / real.sqrt 3 in
  let d1 := 2 * circumradius in
  let d2 := 2 * circumradius in
  (1 / 2) * d1 * d2

theorem area_of_ABCD {A B C D : ℝ × ℝ} (side_length : ℕ) (h : side_length = 1)
  (equilateral_triangles : ∀ (P Q R : ℝ × ℝ), (∃ (center : ℝ × ℝ), 
  center = (0, -(side_length : ℚ) / real.sqrt 3) ∨ center = ((side_length : ℚ) / real.sqrt 3, 0)
  ∨ center = (0, (side_length : ℚ) / real.sqrt 3) ∨ center = (-(side_length : ℚ) / real.sqrt 3, 0))) :
  area_ABCD side_length = (3 + real.sqrt 3) / 6 := by  
  sorry

end area_of_ABCD_l78_78171


namespace fish_added_l78_78653

theorem fish_added (x : ℕ) (hx : x + (x - 4) = 20) : x - 4 = 8 := by
  sorry

end fish_added_l78_78653


namespace question1_question2_question3_l78_78232

-- Define the sequence {a_n}
noncomputable def a : ℕ → ℝ
| 0 := 1
| n+1 := n / a n
  
-- Define the conditions and questions as Lean theorems.
theorem question1 : a 2 = 1 ∧ a 3 = 2 ∧ a 4 = 3 / 2 := sorry

theorem question2 (n : ℕ) : a (n+2) = 1 / a (n+1) + a n := sorry

theorem question3 (n : ℕ) : 2 * real.sqrt n - 1 ≤ ∑ i in finset.range (n + 1), 1 / a i ∧ 
                            ∑ i in finset.range (n + 1), 1 / a i < 3 * real.sqrt n - 1 := sorry

end question1_question2_question3_l78_78232


namespace pentagon_regular_l78_78289

theorem pentagon_regular
  (ABCDE : Type)
  [convex_pentagon ABCDE]
  (eq_sides : ∀ {a b : ℝ}, a ∈ sides ABCDE → b ∈ sides ABCDE → a = b)
  (eq_diagonals : ∃ d₁ d₂ d₃ d₄ d₅ ∈ diagonals ABCDE, d₁ = d₂ ∧ d₂ = d₃ ∧ d₃ = d₄ ∧ d₄ ≠ d₅) :
  regular_pentagon ABCDE :=
sorry

end pentagon_regular_l78_78289


namespace S8_value_l78_78243

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def geometric_sequence (a b c : ℝ) : Prop :=
  a * c = b * b

def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem S8_value 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 1) 
  (h3 : S = sum_arithmetic_sequence a) 
  (h4 : geometric_sequence (a 1) (a 2) (a 5)) : 
  S 8 = 8 ∨ S 8 = 64 :=
by
  sorry

end S8_value_l78_78243


namespace min_value_x_plus_4y_l78_78612

-- Variables and conditions
variables (x y : ℝ)
hypothesis hx : x > 0
hypothesis hy : y > 0
hypothesis hcond : (1/x) + (1/(2*y)) = 2

theorem min_value_x_plus_4y : ∃ (x y : ℝ), (x > 0) ∧ (y > 0) ∧ ((1/x) + (1/(2*y)) = 2) ∧ (x + 4*y = 2 + real.sqrt 2) :=
by
  sorry

end min_value_x_plus_4y_l78_78612


namespace arrangement_six_people_l78_78553

theorem arrangement_six_people :
  let Grade1 := {A : Type} 
  let Grade2 := {B : Type} 
  let Grade3 := {C : Type} 
  let people := {A, B, C, D, E, F} : Set Type 
  ∃ (arrangements : Finset (Finset Type)), 
  arrangements.card = 6 ∧
  ∀(G1 G2 G3 : Finset Type)(hG1 : G1 ∈ arrangements)(hG2 : G2 ∈ arrangements)(hG3 : G3 ∈ arrangements),
  (A ∈ G1) ∧ (B ∉ G3) ∧ (C ∉ G3) ∧ (G1.card = 2) ∧ (G2.card = 2) ∧ (G3.card = 2) 
  := by {sorry}

end arrangement_six_people_l78_78553


namespace tangent_line_slope_3_l78_78631

-- Defining the function f(x)
def f (x : ℝ) : ℝ := 2 * x + Real.log x

-- The point through which the tangent line passes
def point : ℝ × ℝ := (0, -1)

-- The slope we need to prove
def tangent_slope (p : ℝ × ℝ) (m : ℝ) : ℝ := 2 + 1 / m

-- Lean statement for the proof problem
theorem tangent_line_slope_3 : 
  (∃ m : ℝ, f m = 2 * m + Real.log m ∧ (0, -1) = ((2 * m + Real.log m - (-1)) / (m - 0)) ∧ (2 + 1 / m = 3)) :=
sorry

end tangent_line_slope_3_l78_78631


namespace find_a_b_c_l78_78195

variable (a b c : ℚ)

def parabola (x : ℚ) : ℚ := a * x^2 + b * x + c

def vertex_condition := ∀ x, parabola a b c x = a * (x - 3)^2 - 2
def contains_point := parabola a b c 0 = 5

theorem find_a_b_c : vertex_condition a b c ∧ contains_point a b c → a + b + c = 10 / 9 :=
by
sorry

end find_a_b_c_l78_78195


namespace tangent_line_equation_l78_78278

theorem tangent_line_equation {a b : ℝ} (h_curve_tangent : b = sqrt a ∧ (a > 0) ∧ l = (1 / (2 * sqrt a)) * x + (sqrt a / 2))
  (h_circle_tangent : ∀ {x y : ℝ}, x^2 + y^2 = 1 / 5 -> distance (0, 0) (x, (1 / (2 * sqrt a)) * x + (sqrt a / 2)) = sqrt (1 / 5)) :
  l = (1 / 2) * x + (1 / 2) :=
by 
  sorry

end tangent_line_equation_l78_78278


namespace range_of_f_l78_78279

def log_op (a b : ℝ) : ℝ :=
  if a < b then b else a

def f (x : ℝ) : ℝ :=
  log_op (Real.log 2 x) (Real.log 1/2 x)

theorem range_of_f : (∀ x > 0, f x ∈ [0, ∞)) :=
  sorry

end range_of_f_l78_78279


namespace find_number_l78_78040

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l78_78040


namespace seq_a4_equals_15_l78_78618

-- Define the sequence according to the given conditions
def seq : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * seq n + 1

-- Prove the specific value for a_4
theorem seq_a4_equals_15 : seq 3 = 15 := by
  sorry

end seq_a4_equals_15_l78_78618


namespace min_value_expr_l78_78204

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4 * x + 1 / x^2 ≥ 5 :=
by
  sorry

end min_value_expr_l78_78204


namespace perpendiculars_form_square_l78_78879

-- Definitions of points and geometric entities
variables {Point : Type} [AffineGeometry Point]
variables (A B C D : Point) (A1 B1 C1 D1 : Point)

-- Conditions: A, B, C, D form a square; A1, B1, C1, D1 form a parallelogram circumscribed around the square
def is_square (A B C D : Point) : Prop := /* Definition for points forming a square */
def is_parallelogram (A1 B1 C1 D1 : Point) : Prop := /* Definition for points forming a parallelogram */
def circumscribed (A B C D A1 B1 C1 D1 : Point) : Prop := /* Definition for the parallelogram A1B1C1D1 circumscribed around square ABCD */

-- Question: The perpendiculars dropped from A1, B1, C1, D1 to sides of ABCD form a square
theorem perpendiculars_form_square
  (h_square : is_square A B C D)
  (h_parallelogram : is_parallelogram A1 B1 C1 D1)
  (h_circumscribed : circumscribed A B C D A1 B1 C1 D1) :
  ∃ l₁ l₂ l₃ l₄ : Point → Prop,
    (∀ p, l₁ p → ⟪p, _⟫) ∧  -- perpendicular from vertices of parallelogram
    (∀ p, l₂ p → ⟪p, _⟫) ∧
    (∀ p, l₃ p → ⟪p, _⟫) ∧
    (∀ p, l₄ p → ⟪p, _⟫) ∧
    -- Showing l1, l2, l3, l4 form a square
    /* formulation for l₁, l₂, l₃, and l₄ forming a square */
    sorry

end perpendiculars_form_square_l78_78879


namespace height_of_pole_l78_78030

theorem height_of_pole (pole_shadow tree_shadow tree_height : ℝ) 
                       (ratio_equal : pole_shadow = 84 ∧ tree_shadow = 32 ∧ tree_height = 28) : 
                       round (tree_height * (pole_shadow / tree_shadow)) = 74 :=
by
  sorry

end height_of_pole_l78_78030


namespace total_exercise_time_l78_78312

-- Definitions based on given conditions
def javier_daily : ℕ := 50
def javier_days : ℕ := 7
def sanda_daily : ℕ := 90
def sanda_days : ℕ := 3

-- Proof problem to verify the total exercise time for both Javier and Sanda
theorem total_exercise_time : javier_daily * javier_days + sanda_daily * sanda_days = 620 := by
  sorry

end total_exercise_time_l78_78312


namespace value_of_y_l78_78034

theorem value_of_y (y : ℝ) : (40 / 60 = real.sqrt (y / 60 - 10 / 60)) → y = 110 / 3 :=
by
  sorry

end value_of_y_l78_78034


namespace max_value_of_exp_sum_l78_78735

theorem max_value_of_exp_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_pos : 0 < a * b) :
    ∃ θ : ℝ, a * Real.exp θ + b * Real.exp (-θ) = 2 * Real.sqrt (a * b) :=
by
  sorry

end max_value_of_exp_sum_l78_78735


namespace range_of_zeros_of_g_l78_78242

variable {a b x : ℝ}

def f (x : ℝ) : ℝ := x^2 - 2 * a * x + b
def f' (x : ℝ) : ℝ := 2 * x - 2 * a
def g (x : ℝ) : ℝ := f' x + b

theorem range_of_zeros_of_g (h : ∀ (y : ℝ), y ≥ -1 ↔ ∃ x : ℝ, f x = y) : 
  (∃ x : ℝ, g x = 0) → (x ≤ 1) :=
by
  sorry

end range_of_zeros_of_g_l78_78242


namespace recommendation_plans_correct_l78_78410

def students : List String := ["A", "B", "C", "D"]

def schools : List String := ["SchoolA", "SchoolB", "SchoolC"]

def must_at_least_one_student (assignments : String → String) : Prop :=
  ∃ a b c, a ∈ assignments.values ∧ b ∈ assignments.values ∧ c ∈ assignments.values 

def A_not_in_SchoolA (assignments : String → String) : Prop :=
  assignments "A" ≠ "SchoolA"

noncomputable def num_recommendation_plans : Nat := 24

theorem recommendation_plans_correct :
  ∃ assignments : String → String, 
    (must_at_least_one_student assignments) ∧ 
    (A_not_in_SchoolA assignments) ∧ 
    (assignments.values.to_finset.card = 4) ∧
    (num_recommendation_plans = 24) :=
sorry

end recommendation_plans_correct_l78_78410


namespace infinite_composite_not_complete_residue_infinite_composite_complete_residue_l78_78597

open Nat

-- Definitions as specified in the problem
def S (n : ℕ) : Set ℕ := 
  {k | n ≤ k ∧ k ≤ n^2 ∧ choose k n ≠ 0}

-- Part (a) statement
theorem infinite_composite_not_complete_residue (h : ∀ n : ℕ, n ≥ 2 → ∃ p : ℕ, p < n ∧ n = 2 * p) : 
  ∃∞ n : ℕ, n ≥ 2 ∧ Comp n ∧ ¬ (S n).Complete n := 
sorry

-- Part (b) statement
theorem infinite_composite_complete_residue (h : ∀ n : ℕ, n ≥ 2 → ∃ p : ℕ, p < n ∧ n = p^2) : 
  ∃∞ n : ℕ, n ≥ 2 ∧ Comp n ∧ (S n).Complete n :=
sorry


end infinite_composite_not_complete_residue_infinite_composite_complete_residue_l78_78597


namespace no_solution_exists_l78_78679

theorem no_solution_exists :
  ¬ ∃ m n : ℕ, 
    m + n = 2009 ∧ 
    (m * (m - 1) + n * (n - 1) = 2009 * 2008 / 2) := by
  sorry

end no_solution_exists_l78_78679


namespace fraction_relationships_l78_78659

variable (p r s u : ℚ)

theorem fraction_relationships (h1 : p / r = 8) (h2 : s / r = 5) (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 :=
sorry

end fraction_relationships_l78_78659


namespace Maya_takes_longer_l78_78849

-- Define the constants according to the conditions
def Xavier_reading_speed : ℕ := 120
def Maya_reading_speed : ℕ := 60
def novel_pages : ℕ := 360
def minutes_per_hour : ℕ := 60

-- Define the times it takes for Xavier and Maya to read the novel
def Xavier_time : ℕ := novel_pages / Xavier_reading_speed
def Maya_time : ℕ := novel_pages / Maya_reading_speed

-- Define the time difference in hours and then in minutes
def time_difference_hours : ℕ := Maya_time - Xavier_time
def time_difference_minutes : ℕ := time_difference_hours * minutes_per_hour

-- The statement to prove
theorem Maya_takes_longer :
  time_difference_minutes = 180 :=
by
  sorry

end Maya_takes_longer_l78_78849


namespace minimum_varphi_l78_78763

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)

theorem minimum_varphi (φ : ℝ) (φ_pos : φ > 0) :
  (∀ x : ℝ, f (x) = f (-x)) → φ = π / 12 :=
sorry

end minimum_varphi_l78_78763


namespace find_d_over_a1_l78_78774

-- Define the arithmetic sequence conditions
variables {α : Type*} [LinearOrderedField α]
variables (a1 d : α)

-- Define the necessary terms from the sequence
def a2 : α := a1 + d
def S3 : α := (3 * (a1 + (a1 + 2 * d))) / 2
def S5 : α := (5 * (a1 + (a1 + 4 * d))) / 2

-- Define the geometric sequence condition
def geometric_condition : Prop := S3^2 = a2 * (a2 + S5)

-- State the theorem to prove
theorem find_d_over_a1 (h : geometric_condition) (h_pos : ∀ n, a1 + (n - 1) * d > 0) :
  d / a1 = 3 / 2 :=
by sorry

end find_d_over_a1_l78_78774


namespace ivan_income_tax_l78_78711

-- Define the salary schedule
def first_two_months_salary: ℕ := 20000
def post_probation_salary: ℕ := 25000
def bonus_in_december: ℕ := 10000
def income_tax_rate: ℝ := 0.13

-- Define the total taxable income
def total_taxable_income: ℕ :=
  (first_two_months_salary * 2) + (post_probation_salary * 8) + bonus_in_december

-- Define the expected tax amount
def expected_tax: ℕ := 32500

-- Define the personal income tax calculation function
def calculate_tax (income: ℕ) (rate: ℝ): ℕ :=
  (income * rate).toInt

-- The statement which shows that the calculated tax is equal to the expected tax
theorem ivan_income_tax: calculate_tax total_taxable_income income_tax_rate = expected_tax := by
  -- Skip the actual proof
  sorry

end ivan_income_tax_l78_78711


namespace parabola_focus_coordinates_l78_78589

theorem parabola_focus_coordinates (h : ∀ x : ℝ, (2 * x^2 + 8 * x - 1 : ℝ) = (2 * (x+2)^2 - 9 : ℝ)) :
  ∃ p : ℝ × ℝ, p = (-2, -8.875) ∧ ∀ (x : ℝ), (2 * (x+2)^2 - 9 : ℝ) = (two_mul x + 4 * x + (-1 : ℝ)) :=
begin
  sorry
end

end parabola_focus_coordinates_l78_78589


namespace min_value_of_m_l78_78271

theorem min_value_of_m : (2 ∈ {x | ∃ (m : ℤ), x * (x - m) < 0}) → ∃ (m : ℤ), m = 3 :=
by
  sorry

end min_value_of_m_l78_78271


namespace average_balance_is_correct_l78_78929

def balance_at_end (month : ℕ) : ℝ :=
  match month with
  | 1 => 100
  | 2 => 200
  | 3 => 150
  | 4 => 150
  | 5 => 200 * 1.05
  | 6 => 250 * 1.05
  | _ => 0 -- assume zero for other months

def average_balance : ℝ :=
  (balance_at_end 1 + balance_at_end 2 + balance_at_end 3 +
   balance_at_end 4 + balance_at_end 5 + balance_at_end 6) / 6

theorem average_balance_is_correct :
  average_balance = 178.75 :=
sorry

end average_balance_is_correct_l78_78929


namespace problem_I_problem_II_problem_III_l78_78991

-- The function f(x)
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1/2) * x^2 - a * Real.log x + b

-- Tangent line at x = 1
def tangent_condition (a : ℝ) (b : ℝ) :=
  1 - a = 3 ∧ f 1 a b = 0

-- Extreme point at x = 1
def extreme_condition (a : ℝ) :=
  1 - a = 0 

-- Monotonicity and minimum m
def inequality_condition (a m : ℝ) :=
  -2 ≤ a ∧ a < 0 ∧ ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 ≤ 2 ∧ 0 < x2 ∧ x2 ≤ 2 → 
  |f x1 a (0 : ℝ) - f x2 a 0| ≤ m * |1 / x1 - 1 / x2|

-- Proof problem 1
theorem problem_I : ∃ (a b : ℝ), tangent_condition a b → a = -2 ∧ b = -0.5 := sorry

-- Proof problem 2
theorem problem_II : ∃ (a : ℝ), extreme_condition a → a = 1 := sorry

-- Proof problem 3
theorem problem_III : ∃ (m : ℝ), inequality_condition (-2 : ℝ) m → m = 12 := sorry

end problem_I_problem_II_problem_III_l78_78991


namespace geric_initial_bills_l78_78609

theorem geric_initial_bills (G K J : ℕ) 
  (h1: G = 2 * K)
  (h2: K = J - 2)
  (h3: J - 3 = 7) : G = 16 := 
  by 
  sorry

end geric_initial_bills_l78_78609


namespace product_of_repeating_decimal_l78_78906

theorem product_of_repeating_decimal :
  let q := (1 : ℚ) / 3 in
  q * 9 = 3 :=
by
  sorry

end product_of_repeating_decimal_l78_78906


namespace lisa_punch_l78_78744

theorem lisa_punch (x : ℝ) (H : x = 0.125) :
  (0.3 + x) / (2 + x) = 0.20 :=
by
  sorry

end lisa_punch_l78_78744


namespace distance_between_stations_l78_78060

theorem distance_between_stations :
  ∀ (x t : ℕ), 
    (20 * t = x) ∧ 
    (25 * t = x + 70) →
    (2 * x + 70 = 630) :=
by
  sorry

end distance_between_stations_l78_78060


namespace cos2_alpha_plus_2sin2_alpha_l78_78237

variable {α : ℝ}

-- Given condition 
def tan_alpha : ℝ := 3 / 4

theorem cos2_alpha_plus_2sin2_alpha (h : Real.tan α = tan_alpha) :
    Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
sorry

end cos2_alpha_plus_2sin2_alpha_l78_78237


namespace largest_25_supporting_X_l78_78130

def is_25_supporting (X : ℝ) : Prop :=
∀ (a : fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : Sup {X : ℝ | is_25_supporting X} = 0.02 :=
sorry

end largest_25_supporting_X_l78_78130


namespace combined_share_a_c_l78_78137

-- Define the conditions
def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

-- The total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

-- The value of each part
def value_per_part : ℕ := total_money / total_parts

-- The shares of a and c
def share_a : ℕ := ratio_a * value_per_part
def share_c : ℕ := ratio_c * value_per_part

-- Prove that the combined share of a + c equals 6952
theorem combined_share_a_c : share_a + share_c = 6952 :=
by
  -- This is the proof placeholder
  sorry

end combined_share_a_c_l78_78137


namespace lambda_system_equiv_l78_78904

universe u

variables {Ω : Type u} (I : set (set Ω))

-- Condition definitions
def condition_a := Ω ∈ I
def condition_b := ∀ {A B : set Ω}, A ∈ I → B ∈ I → A ⊆ B → B \ A ∈ I
def condition_c := ∀ {A : set (set Ω)}, (∀ n, A n ∈ I) → ∀ A, (∀ n, A n ⊆ A n.succ) → (⋃ n, A n = A) → A ∈ I
def condition_b' := ∀ {A : set Ω}, A ∈ I → Ω \ A ∈ I
def condition_c' := ∀ {A : set (set Ω)}, (∀ n, A n ∈ I) → (∀ n m, n ≠ m → A n ∩ A m = ∅) → (⋃ n, A n) ∈ I

-- Statement of equivalence
theorem lambda_system_equiv :
  (condition_a I ∧ condition_b I ∧ condition_c I) ↔ (condition_a I ∧ condition_b' I ∧ condition_c' I) :=
begin
  sorry
end

end lambda_system_equiv_l78_78904


namespace max_snowmen_l78_78501

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78501


namespace roots_geometric_progression_two_complex_conjugates_l78_78586

theorem roots_geometric_progression_two_complex_conjugates (a : ℝ) :
  (∃ b k : ℝ, b ≠ 0 ∧ k ≠ 0 ∧ (k + 1/ k = 2) ∧ 
    (b * (1 + k + 1/k) = 9) ∧ (b^2 * (k + 1 + 1/k) = 27) ∧ (b^3 = -a)) →
  a = -27 :=
by sorry

end roots_geometric_progression_two_complex_conjugates_l78_78586


namespace total_area_of_squares_l78_78823

-- Condition 1: Definition of the side length
def side_length (s : ℝ) : Prop := s = 12

-- Condition 2: Definition of the center of one square coinciding with the vertex of another
-- Here, we assume the positions are fixed so this condition is given
def coincide_center_vertex (s₁ s₂ : ℝ) : Prop := s₁ = s₂ 

-- The main theorem statement
theorem total_area_of_squares
  (s₁ s₂ : ℝ) 
  (h₁ : side_length s₁)
  (h₂ : side_length s₂)
  (h₃ : coincide_center_vertex s₁ s₂) :
  (2 * s₁^2) - (s₁^2 / 4) = 252 :=
by
  sorry

end total_area_of_squares_l78_78823


namespace minValue_expression_l78_78274

theorem minValue_expression (x y : ℝ) (h : x + 2 * y = 4) : ∃ (v : ℝ), v = 2^x + 4^y ∧ ∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ v :=
by 
  sorry

end minValue_expression_l78_78274


namespace ratio_in_set_rational_l78_78754

variable {S : Set ℝ}
hypothesis hS : S.card = 100
hypothesis h_diff : ∀ (x y : ℝ), x ∈ S → y ∈ S → x ≠ y → x ≠ y
hypothesis h_int : ∀ (a b c : ℝ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → a ≠ c → b ≠ c → ∃ n : ℤ, a^2 + b * c = n

theorem ratio_in_set_rational (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) : x ≠ y → Rat.cast (x / y) :=
by
  sorry

end ratio_in_set_rational_l78_78754


namespace line_l_equation_line_l2_equation_l78_78627

theorem line_l_equation
    (l_passes_through : ∀ p : ℝ × ℝ, p = (0, 5) → Prop)
    (sum_of_intercepts : ∀ x_int y_int : ℝ, x_int + y_int = 2 ∧ ∃ (p : ℝ × ℝ), p = (0, 5) → Prop)
    : ∀ (x y : ℝ), 5 * x - 3 * y + 15 = 0 :=
by
  sorry

theorem line_l2_equation
    (l1_passes_through : ∀ p : ℝ × ℝ, p = (8 / 3, -1) → Prop)
    (perpendicular_to_l : ∀ l1 l : ℝ → ℝ, l1 = (fun x => -1 / (l x)) ∧ l_passes_through (0, 5) → Prop)
    (symmetric_about_x_axis : ∀ l1 l2 : ℝ → ℝ, l2 = (fun x => -l1 x) ∧ l1_passes_through (8 / 3, -1) → Prop)
    : ∀ (x y : ℝ), 3 * x - 5 * y - 3 = 0 :=
by
  sorry

end line_l_equation_line_l2_equation_l78_78627


namespace base_for_195₁₀_four_digit_even_final_digit_l78_78951

theorem base_for_195₁₀_four_digit_even_final_digit :
  ∃ b : ℕ, (b^3 ≤ 195 ∧ 195 < b^4) ∧ (∃ d : ℕ, 195 % b = d ∧ d % 2 = 0) ∧ b = 5 :=
by {
  sorry
}

end base_for_195₁₀_four_digit_even_final_digit_l78_78951


namespace total_number_of_people_l78_78816

theorem total_number_of_people (c a : ℕ) (h1 : c = 2 * a) (h2 : c = 28) : c + a = 42 :=
by
  sorry

end total_number_of_people_l78_78816


namespace sum_of_largest_and_smallest_prime_factors_of_2730_l78_78427

theorem sum_of_largest_and_smallest_prime_factors_of_2730 : 
  ∃ (smallest largest : ℕ), smallest ∈ {2, 3, 5, 7, 13} ∧ 
  largest ∈ {2, 3, 5, 7, 13} ∧ 
  smallest < largest ∧ 
  smallest + largest = 15 :=
by
  sorry

end sum_of_largest_and_smallest_prime_factors_of_2730_l78_78427


namespace least_positive_integer_n_l78_78591

noncomputable def P (n : ℕ) (X : ℂ) : ℂ :=
  (real.sqrt 3) * X^(n + 1) - X^n - 1

theorem least_positive_integer_n (n : ℕ) :
  (∃ (X : ℂ), abs X = 1 ∧ P n X = 0) ↔ n = 10 := by
  sorry

end least_positive_integer_n_l78_78591


namespace product_m_t_l78_78733

noncomputable def g : ℝ → ℝ := sorry

axiom g_property : ∀ (x y z : ℝ), g (x^2 + y * g z) = x * g x + z * g y 

def possible_values_of_g3 : set ℝ := {y | ∃ c : ℝ, (c = 0 ∨ c = 1) ∧ (g = λ x, c * x) ∧ g 3 = y}

def m : ℕ := possible_values_of_g3.to_finset.card
def t : ℝ := possible_values_of_g3.to_finset.sum id

theorem product_m_t : m * t = 6 := 
sorry

end product_m_t_l78_78733


namespace smallest_five_digit_divisible_by_2_3_5_7_11_is_11550_l78_78208

theorem smallest_five_digit_divisible_by_2_3_5_7_11_is_11550 :
  ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ n = 11550 :=
by {
  existsi 11550,
  split; try split; try split; try split; try split; try split; try split, 
  sorry,
}

end smallest_five_digit_divisible_by_2_3_5_7_11_is_11550_l78_78208


namespace sum_of_powers_l78_78729

-- Here is the statement in Lean 4
theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68) = (ω^2 - 1) / (ω^4 - 1) :=
sorry -- Proof is omitted as per instructions.

end sum_of_powers_l78_78729


namespace area_KDC_is_25sqrt3_l78_78305

noncomputable def area_of_triangle_KDC : ℝ :=
  let r := 10 in
  let CD := 10 in
  let KA := 20 in
  let height := r * sqrt 3 in
  0.5 * CD * height

theorem area_KDC_is_25sqrt3 :
  area_of_triangle_KDC = 25 * sqrt 3 :=
by
  sorry

end area_KDC_is_25sqrt3_l78_78305


namespace sufficient_but_not_necessary_l78_78899

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > b + 1) → (a > b) ∧ ¬(a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_l78_78899


namespace max_snowmen_l78_78524

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78524


namespace major_axis_length_4_ellipse_max_area_difference_l78_78620

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) := 
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)

theorem major_axis_length_4_ellipse (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) (major_axis_eq_4 : 2 * a = 4) (equilateral_condition : a = 2 ∧ b = sqrt 3) :
  ellipse_equation a b h1 h2 :=
sorry

theorem max_area_difference (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) (major_axis_eq_4 : 2 * a = 4) 
  (equilateral_condition : a = 2 ∧ b = sqrt 3) :
  ∃ k : ℝ, ∀ (O A D C : ℝ) (line_l : ℝ → ℝ → Prop) (left_focus : ℝ) (C_intersect : ℝ)
  (D_intersect : ℝ), 
  |area (triangle O A D) - area (triangle O A C)| ≤  
  (6 * |k| / (3 * k^2 + 4)) <= sqrt 3 / 2 :=
sorry

end major_axis_length_4_ellipse_max_area_difference_l78_78620


namespace business_executive_emails_l78_78870

def daily_emails (n : ℕ) : ℕ :=
  match n with
  | 1 => 20
  | 2 => 15
  | 3 => 11
  | 4 => 6
  | 5 => 6
  | 6 => 8
  | 7 => 6
  | 8 => 4
  | 9 => 2
  | 10 => 3
  | _ => 0

def total_emails_received : ℕ :=
  (List.range 10).map (λ n => daily_emails (n + 1)).sum

theorem business_executive_emails : total_emails_received = 81 := by
  sorry

end business_executive_emails_l78_78870


namespace area_of_circle_is_130_pi_l78_78918

open Real

noncomputable def area_of_circle
  (AP BP CD : ℝ)
  (h_AP : AP = 6)
  (h_BP : BP = 12)
  (h_CD : CD = 22)
  (h_perpendicular : ∃ (O : ℝ × ℝ), True) -- Placeholder for perpendicular intersection property
: ℝ :=
let AB := AP + BP in
let AM := AB / 2 in
let MP := AM - AP in
let CD_half := CD / 2 in
let OC := sqrt ((MP)^2 + (CD_half)^2) in
π * (OC)^2

theorem area_of_circle_is_130_pi
  : area_of_circle 6 12 22 6 rfl 12 rfl 22 rfl (by trivial) = 130 * π :=
by
  sorry

end area_of_circle_is_130_pi_l78_78918


namespace stratified_sampling_undergraduate_students_l78_78150

theorem stratified_sampling_undergraduate_students
  (total_students : ℕ) (junior_college_students : ℕ) (undergraduate_students : ℕ) (graduate_students : ℕ) (sample_size : ℕ) :
  total_students = 5600 →
  junior_college_students = 1300 →
  undergraduate_students = 3000 →
  graduate_students = 1300 →
  sample_size = 280 →
  undergraduate_students * sample_size / total_students = 150 :=
by
  intros h_total h_junior h_undergrad h_graduate h_sample
  rw [h_total, h_junior, h_undergrad, h_graduate, h_sample]
  norm_num
  sorry

end stratified_sampling_undergraduate_students_l78_78150


namespace num_green_hats_l78_78061

-- Definitions
def total_hats : ℕ := 85
def blue_hat_cost : ℕ := 6
def green_hat_cost : ℕ := 7
def total_cost : ℕ := 548

-- Prove the number of green hats (g) is 38 given the conditions
theorem num_green_hats (b g : ℕ) 
  (h₁ : b + g = total_hats)
  (h₂ : blue_hat_cost * b + green_hat_cost * g = total_cost) : 
  g = 38 := by
  sorry

end num_green_hats_l78_78061


namespace complement_intersection_l78_78644

def U : set ℤ := {x | -4 < x ∧ x < 4}
def A : set ℤ := {-1, 0, 2, 3}
def B : set ℤ := {-2, 0, 1, 2}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {-3} :=
sorry

end complement_intersection_l78_78644


namespace min_lcm_leq_six_floor_l78_78065

theorem min_lcm_leq_six_floor (n : ℕ) (h : n ≠ 4) (a : Fin n → ℕ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 2 * n) : 
  ∃ i j, i < j ∧ Nat.lcm (a i) (a j) ≤ 6 * (n / 2 + 1) :=
by
  sorry

end min_lcm_leq_six_floor_l78_78065


namespace eccentricity_of_hyperbola_l78_78236

-- Definitions and assumptions based on the problem
variable (a b c : ℝ)

-- The hyperbola with a > 0 and b > 0
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Definitions of the foci coordinates
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)

-- Condition given by the right triangle with vertices F1, F2, and P
axiom right_triangle_condition : real.sqrt (c^2 + 4 * b^2) = 2 * c

-- Identity to simplify foci condition
def simplify_foci_condition (c a : ℝ) : Prop := c^2 = 4 * a^2

-- Main statement to be proven
theorem eccentricity_of_hyperbola : simplify_foci_condition c a → (c / a = 2) :=
by sorry

end eccentricity_of_hyperbola_l78_78236


namespace ivan_income_tax_l78_78705

noncomputable def personalIncomeTax (monthly_salary: ℕ → ℕ) (bonus: ℕ) (tax_rate: ℚ) : ℕ :=
  let taxable_income := (monthly_salary 3 + monthly_salary 4) +
                       (List.sum (List.map monthly_salary [5, 6, 7, 8, 9, 10, 11, 12])) +
                       bonus
  in taxable_income * tax_rate

theorem ivan_income_tax :
  personalIncomeTax
    (λ m, if m ∈ [3, 4] then 20000 else if m ∈ [5, 6, 7, 8, 9, 10, 11, 12] then 25000 else 0)
    10000 0.13 = 32500 :=
  sorry

end ivan_income_tax_l78_78705


namespace non_obtuse_triangle_perimeter_gt_4R_l78_78764

-- Define the conditions for a non-obtuse triangle with a circumcircle radius R
structure non_obtuse_triangle (ABC : Type*) :=
(a b c R : ℝ)
(is_non_obtuse : ∀ (α β γ : ℝ), α ≤ 90 ∧ β ≤ 90 ∧ γ ≤ 90)
(is_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
(circumradius : R = √((a * b * c) / (4 * (a + b + c))))

-- Define the theorem to prove the perimeter is greater than 4 times the circumradius
theorem non_obtuse_triangle_perimeter_gt_4R (T : non_obtuse_triangle ℝ) :
  T.a + T.b + T.c > 4 * T.R :=
sorry

end non_obtuse_triangle_perimeter_gt_4R_l78_78764


namespace train_A_reaches_destination_in_6_hours_l78_78419

/-- Given two trains A and B traveling in opposite directions, meeting at a certain point. Train A
travels at a speed of 70 km/h, and train B travels at a speed of 105 km/h. Train B reaches its
destination 4 hours after meeting train A. Prove that train A takes 6 hours to reach its
destination after meeting train B. -/
theorem train_A_reaches_destination_in_6_hours :
  ∀ (speed_A speed_B : ℝ) (time_B : ℝ),
  speed_A = 70 →
  speed_B = 105 →
  time_B = 4 →
  let distance_B := speed_B * time_B in
  let time_A := distance_B / speed_A in
  time_A = 6 :=
by
  intros speed_A speed_B time_B h_speed_A h_speed_B h_time_B
  let distance_B := speed_B * time_B
  let time_A := distance_B / speed_A
  have h_distance_B : distance_B = 420, from calc
    distance_B = 105 * 4 : by rw [h_speed_B, h_time_B]
    ... = 420 : by norm_num
  have h_time_A : time_A = 420 / 70, from calc
    time_A = distance_B / speed_A : by rfl
    ... = 420 / 70 : by rw [h_distance_B, h_speed_A]
  exact h_time_A.symm.trans (by norm_num)


end train_A_reaches_destination_in_6_hours_l78_78419


namespace volume_of_emmas_can_l78_78188

noncomputable def volume_of_cylinder (diameter height : ℝ) : ℝ :=
  let r := diameter / 2
  π * r^2 * height

theorem volume_of_emmas_can :
  volume_of_cylinder 2.75 4.8 ≈ 28.513 :=
begin
  -- Introduce the calculation of radius and volume
  let r := 2.75 / 2,
  let V := π * r^2 * 4.8,
  -- Show that the calculated volume is approximately 28.513 cubic inches
  have h : V ≈ 28.513,
  { sorry },
  exact h,
end

end volume_of_emmas_can_l78_78188


namespace maximum_snowmen_count_l78_78490

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78490


namespace equation_of_parallel_line_final_equation_l78_78199

def passes_through (x y : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
line x y

def is_parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
∃ k : ℝ, ∀ x y : ℝ, l1 x y ↔ l2 (k * x) (k * y)

def line_eq (c : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * x - y + c = 0

theorem equation_of_parallel_line (c : ℝ) : 
  (∀ x y : ℝ, passes_through 1 2 (line_eq c) ∧ 
             is_parallel (line_eq c) (λ x y, 2 * x - y - 1 = 0)) → 
  c = 0 :=
by
  intro h
  sorry

theorem final_equation : ∃ c : ℝ, line_eq c 1 2 ∧ is_parallel (line_eq c) (λ x y, 2 * x - y - 1 = 0) → 
  ∀ x y : ℝ, 2 * x - y = 0 :=
by
  intro h
  obtain ⟨c, hc1, hc2⟩ := h
  have h_c := equation_of_parallel_line c (by
    split
    · exact hc1
    · exact hc2)
  rw h_c
  sorry

end equation_of_parallel_line_final_equation_l78_78199


namespace freddie_can_form_all_amounts_under_dollar_l78_78023

-- Define the types of coins
inductive Coin
| penny     -- 1 cent
| nickel    -- 5 cents
| halfdollar -- 50 cents

-- Define the value of each coin
def coin_value : Coin → ℕ
| Coin.penny     := 1
| Coin.nickel    := 5
| Coin.halfdollar := 50

-- The set of coins Freddie possesses
def freddie_coins : List (Coin × Nat) := [(Coin.penny, 5), (Coin.nickel, 5), (Coin.halfdollar, 1)]

-- Function to calculate the total value of a set of coins
def total_value (coins : List (Coin × Nat)) : ℕ :=
  coins.foldl (λ acc (coin : Coin × Nat), acc + (coin_value coin.fst * coin.snd)) 0

-- Predicate to check if any amount from 1 to 99 cents can be formed
def can_form_amounts (coins : List (Coin × Nat)) (max_amount : ℕ) : Prop :=
  ∀ amount ∈ List.range max_amount.succ.tail, ∃ combination : List (Coin × Nat),
    total_value combination = amount ∧ combination.all (λ c, c ∈ coins ∧ c.snd ≤ coins.lookup c.fst.get_or_else 0)

-- Statement of the problem
theorem freddie_can_form_all_amounts_under_dollar :
  can_form_amounts freddie_coins 99 :=
sorry

end freddie_can_form_all_amounts_under_dollar_l78_78023


namespace concentric_circles_probability_l78_78920

theorem concentric_circles_probability (r : ℝ) (h : r > 0) : 
  let R := 6 * r in
  let area_y := π * r^2 in
  let area_x := π * R^2 in
  let area_outside_y := area_x - area_y in
  (area_outside_y / area_x) = (35 / 36) :=
by
  sorry

end concentric_circles_probability_l78_78920


namespace min_sum_arithmetic_sequence_l78_78619

theorem min_sum_arithmetic_sequence : 
  (let a_n := λ n : ℕ, 2 * n - 19 in
   let S_n := λ n : ℕ, n * (a_n(1) + a_n(n)) / 2 in
   S_n 9 = -81) :=
by
   sorry

end min_sum_arithmetic_sequence_l78_78619


namespace checkerboard_squares_containing_at_least_7_black_squares_l78_78865

theorem checkerboard_squares_containing_at_least_7_black_squares :
  let n := 10 in
  let contains_at_least_7_black (i j : ℕ) (size : ℕ) : Prop :=
    size ≥ 4 ∧ size ≤ n ∧ size * size ≥ 7 in
  (∑ size in finset.range (n + 1), (n - size + 1) * (n - size + 1) * ite (contains_at_least_7_black 0 0 size) 1 0) = 140 :=
by
  sorry

end checkerboard_squares_containing_at_least_7_black_squares_l78_78865


namespace relationship_P_N_M_l78_78621

variable (a b c : ℝ)

-- Condition definitions
def M := 2^a
def N := 5^(-b)
def P := Real.log c

-- Given the conditions 0 < a < b < c < 1
-- Prove the relationship among M, N, P is P < N < M
theorem relationship_P_N_M (h : 0 < a ∧ a < b ∧ b < c ∧ c < 1) :
  P < N ∧ N < M := by
  sorry

end relationship_P_N_M_l78_78621


namespace boy_scouts_signed_slips_l78_78116

-- Definitions for the problem conditions have only been used; solution steps are excluded.

theorem boy_scouts_signed_slips (total_scouts : ℕ) (signed_slips : ℕ) (boy_scouts : ℕ) (girl_scouts : ℕ)
  (boy_scouts_signed : ℕ) (girl_scouts_signed : ℕ)
  (h1 : signed_slips = 4 * total_scouts / 5)  -- 80% of the scouts arrived with signed permission slips
  (h2 : boy_scouts = 2 * total_scouts / 5)  -- 40% of the scouts were boy scouts
  (h3 : girl_scouts = total_scouts - boy_scouts)  -- Rest are girl scouts
  (h4 : girl_scouts_signed = 8333 * girl_scouts / 10000)  -- 83.33% of girl scouts with permission slips
  (h5 : signed_slips = boy_scouts_signed + girl_scouts_signed)  -- Total signed slips by both boy and girl scouts
  : (boy_scouts_signed * 100 / boy_scouts = 75) :=    -- 75% of boy scouts with permission slips
by
  -- Proof to be filled in.
  sorry

end boy_scouts_signed_slips_l78_78116


namespace oliver_total_money_l78_78354

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l78_78354


namespace largest_25_supporting_X_l78_78128

def is_25_supporting (X : ℝ) : Prop :=
∀ (a : fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : Sup {X : ℝ | is_25_supporting X} = 0.02 :=
sorry

end largest_25_supporting_X_l78_78128


namespace find_fraction_l78_78196

noncomputable def some_fraction_of_number_is (N f : ℝ) : Prop :=
  1 + f * N = 0.75 * N

theorem find_fraction (N : ℝ) (hN : N = 12.0) :
  ∃ f : ℝ, some_fraction_of_number_is N f ∧ f = 2 / 3 :=
by
  sorry

end find_fraction_l78_78196


namespace total_fence_cost_l78_78142

def length : ℕ := 500
def width : ℕ := 150
def gate_width : ℕ := 1_25
def num_gates : ℕ := 4
def barbed_wire_cost_per_meter : ℕ := 120
def picket_fence_cost_per_meter : ℕ := 250

def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

def barbed_wire_length (p g w : ℕ) : ℕ := p - (g * w)

def barbed_wire_fencing_cost (barbed_wire_length cost_per_meter : ℕ) : ℕ :=
  barbed_wire_length * cost_per_meter

def picket_fence_cost (p cost_per_meter : ℕ) : ℕ := p * cost_per_meter

def total_cost (barbed_wire_cost picket_fence_cost : ℕ) : ℕ :=
  barbed_wire_cost + picket_fence_cost

theorem total_fence_cost :
  total_cost (barbed_wire_fencing_cost (barbed_wire_length (perimeter length width) num_gates gate_width) barbed_wire_cost_per_meter)
             (picket_fence_cost (perimeter length width) picket_fence_cost_per_meter) = 4804 :=
by
  calc total_cost (barbed_wire_fencing_cost (barbed_wire_length (perimeter length width) num_gates gate_width) barbed_wire_cost_per_meter)
                  (picket_fence_cost (perimeter length width) picket_fence_cost_per_meter)
      = 4804 : sorry

end total_fence_cost_l78_78142


namespace geometric_series_remainder_l78_78428

theorem geometric_series_remainder :
  (∑ i in Finset.range 1005, (3^i : ℕ)) % 500 = 121 := 
sorry

end geometric_series_remainder_l78_78428


namespace sum_binomial_identity_l78_78376

open Nat

theorem sum_binomial_identity (n : ℕ) (h : n > 0) :
  (∑ k in finset.range (n + 1), 2^k * (nat.choose n k) * (nat.choose (n - k) (floor ((n - k) / 2)))) = nat.choose (2 * n + 1) n := 
sorry

end sum_binomial_identity_l78_78376


namespace find_roots_of_polynomial_l78_78594

noncomputable def polynomial_roots : Set ℝ :=
  {x | (6 * x^4 + 25 * x^3 - 59 * x^2 + 28 * x) = 0 }

theorem find_roots_of_polynomial :
  polynomial_roots = {0, 1, (-31 + Real.sqrt 1633) / 12, (-31 - Real.sqrt 1633) / 12} :=
by
  sorry

end find_roots_of_polynomial_l78_78594


namespace time_for_faster_train_to_pass_slower_l78_78826

noncomputable def relative_speed (speed_fast : ℝ) (speed_slow : ℝ) : ℝ :=
  speed_fast - speed_slow

noncomputable def relative_speed_meters_per_second (relative_speed_kmh : ℝ) : ℝ :=
  (relative_speed_kmh * 5) / 18

noncomputable def total_distance (length_train : ℝ) : ℝ :=
  length_train * 2

noncomputable def time_to_pass (total_distance : ℝ) (relative_speed_meters_per_second : ℝ) : ℝ :=
  total_distance / relative_speed_meters_per_second

theorem time_for_faster_train_to_pass_slower :
  let speed_fast := 47.0
  let speed_slow := 36.0
  let length_train := 55.0
  let relative_speed_kmh := relative_speed speed_fast speed_slow
  let relative_speed_mps := relative_speed_meters_per_second relative_speed_kmh
  let total_dist := total_distance length_train
  let time := time_to_pass total_dist relative_speed_mps
  time ≈ 36 :=
sorry

end time_for_faster_train_to_pass_slower_l78_78826


namespace fatima_total_donation_l78_78193

theorem fatima_total_donation :
  let cloth1 := 100
  let cloth1_piece1 := 0.40 * cloth1
  let cloth1_piece2 := 0.30 * cloth1
  let cloth1_piece3 := 0.30 * cloth1
  let donation1 := cloth1_piece2 + cloth1_piece3

  let cloth2 := 65
  let cloth2_piece1 := 0.55 * cloth2
  let cloth2_piece2 := 0.45 * cloth2
  let donation2 := cloth2_piece2

  let cloth3 := 48
  let cloth3_piece1 := 0.60 * cloth3
  let cloth3_piece2 := 0.40 * cloth3
  let donation3 := cloth3_piece2

  donation1 + donation2 + donation3 = 108.45 :=
by
  sorry

end fatima_total_donation_l78_78193


namespace area_of_R2_l78_78533

-- Define rectangles and their properties
structure Rectangle where
  length : ℝ
  width : ℝ

-- Conditions
def R1 : Rectangle := { length := 3, width := 8 }
def area_R1 : ℝ := 24
def similar (r1 r2 : Rectangle) : Prop := 
  (r1.length / r1.width) = (r2.length / r2.width)
def diagonal (r : Rectangle) : ℝ := 
  (r.length ^ 2 + r.width ^ 2) ^ 0.5
def R2 : Rectangle := { length := (15 / (73 / 9) ^ 0.5), width := (40 / (73 / 9) ^ 0.5) }
def diagonal_R2 : ℝ := 20

-- Theorem to prove
theorem area_of_R2 : similar R1 R2 ∧ diagonal R2 = diagonal_R2 → R2.length * R2.width = 3200 / 73 := 
by
  sorry

end area_of_R2_l78_78533


namespace loss_percent_example_l78_78555

theorem loss_percent_example :
  ∀ (CP SP : ℝ), CP = 560 ∧ SP = 340 → (CP - SP) / CP * 100 ≈ 39.29 := by
  sorry

end loss_percent_example_l78_78555


namespace log_diff_decreases_l78_78388

-- Define the natural number n
variable (n : ℕ)

-- Proof statement
theorem log_diff_decreases (hn : 0 < n) : 
  (Real.log (n + 1) - Real.log n) = Real.log (1 + 1 / n) ∧ 
  ∀ m : ℕ, ∀ hn' : 0 < m, m > n → Real.log (m + 1) - Real.log m < Real.log (n + 1) - Real.log n := by
  sorry

end log_diff_decreases_l78_78388


namespace fish_added_l78_78655

theorem fish_added (T C : ℕ) (h1 : T + C = 20) (h2 : C = T - 4) : C = 8 :=
by
  sorry

end fish_added_l78_78655


namespace integer_values_satisfying_sqrt_condition_l78_78403

theorem integer_values_satisfying_sqrt_condition : ∃! n : Nat, 2.5 < Real.sqrt n ∧ Real.sqrt n < 3.5 :=
by {
  sorry -- Proof to be filled in
}

end integer_values_satisfying_sqrt_condition_l78_78403


namespace train_length_is_250_l78_78547

noncomputable def train_length : ℝ :=
  let speed_kmh : ℝ := 55
  let speed : ℝ := speed_kmh * (1 / 3.6)
  let time : ℝ := 50.395968322534195
  let distance : ℝ := speed * time
  let platform_length : ℝ := 520
  distance - platform_length

theorem train_length_is_250 : train_length ≈ 250 :=
by sorry

end train_length_is_250_l78_78547


namespace weave_fifth_day_length_l78_78770

theorem weave_fifth_day_length :
  ∃ d : ℝ, (∀ n ∈ ℕ, 1 ≤ n ∧ n ≤ 30 → 
    let nth_term := 5 + (n - 1) * d in
    ∑ i in range 30, (5 + i * d) = 390 → 
    nth_term 5 = 5 + 4 * d) ∧
  let nth_term := 5 + 4 * d in
  nth_term = 209 / 29 :=
by sorry

end weave_fifth_day_length_l78_78770


namespace max_snowmen_l78_78509

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78509


namespace overall_average_marks_l78_78205

theorem overall_average_marks (n1 n2 n3 : ℕ) (a1 a2 a3 : ℕ) 
  (h_n1 : n1 = 55) (h_n2 : n2 = 48) (h_n3 : n3 = 40) 
  (h_a1 : a1 = 60) (h_a2 : a2 = 58) (h_a3 : a3 = 65) : 
  (n1 * a1 + n2 * a2 + n3 * a3) / (n1 + n2 + n3) = 60.73 :=
by
  sorry

end overall_average_marks_l78_78205


namespace range_of_m_l78_78614

theorem range_of_m (m : ℝ) : 
  (∃ (x y : ℝ), x - m * y + (m * real.sqrt 3) = 0 ∧ (y / (x + 1)) * (y / (x - 1)) = 3) →
  m ≤ -real.sqrt 6 / 6 ∨ m ≥ real.sqrt 6 / 6 :=
begin
  sorry
end

end range_of_m_l78_78614


namespace factorize_1_factorize_2_factorize_3_solve_system_l78_78164

-- Proving the factorization identities
theorem factorize_1 (y : ℝ) : 5 * y - 10 * y^2 = 5 * y * (1 - 2 * y) :=
by
  sorry

theorem factorize_2 (m : ℝ) : (3 * m - 1)^2 - 9 = (3 * m + 2) * (3 * m - 4) :=
by
  sorry

theorem factorize_3 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 :=
by
  sorry

-- Proving the solution to the system of equations
theorem solve_system (x y : ℝ) (h1 : x - y = 3) (h2 : x - 3 * y = -1) : x = 5 ∧ y = 2 :=
by
  sorry

end factorize_1_factorize_2_factorize_3_solve_system_l78_78164


namespace max_snowmen_l78_78526

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78526


namespace arithmetic_sequence_solution_l78_78304

-- Declare the arithmetic sequence {a_n} with conditions
def is_arithmetic_sequence (a : ℕ → ℕ) := ∃ d, ∀ n, a (n + 1) = a n + d

-- The sum of the first n terms of the sequence
def S (a : ℕ → ℕ) (n : ℕ) := ∑ i in finset.range (n + 1), a i

-- The given condition for the sum of terms
def sum_condition (a : ℕ → ℕ) := ∀ n, S a (2 * n) / S a n = (4 * n + 2) / (n + 1)

-- The sequence b_n = a_n * p^a_n
def b (a : ℕ → ℕ) (p : ℕ) (n : ℕ) := a n * p ^ a n

-- The sum of the first n terms of the sequence b_n
def T (a : ℕ → ℕ) (p : ℕ) (n : ℕ) := ∑ i in finset.range (n + 1), b a p i

-- The final theorem to prove the two parts
theorem arithmetic_sequence_solution (a : ℕ → ℕ) (p : ℕ) :
  is_arithmetic_sequence a →
  a 1 = 1 →
  sum_condition a →
  (∀ n, a n = n) ∧
  (∀ n,
    T a p n = if p = 1 then (n * (n + 1)) / 2
              else p * (1 - p ^ n) / (1 - p) ^ 2 - n * p ^ (n + 1) / (1 - p)) :=
by
  sorry

end arithmetic_sequence_solution_l78_78304


namespace max_distinct_fans_l78_78066

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l78_78066


namespace max_number_of_snowmen_l78_78516

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78516


namespace m_range_l78_78666

-- Defining the initial condition
def condition (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  (2 - x / (Real.exp 1 * y)) * (Real.log x - Real.log y) - 1 / m ≤ 0

-- The theorem statement to prove
theorem m_range (m : ℝ) :
  (∀ (x y : ℝ), (0 < x) → (0 < y) → condition x y m (by linarith) (by linarith)) → m ∈ Ioc 0 1 := sorry

end m_range_l78_78666


namespace find_lambda_l78_78018

-- Definition of the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 2) = 1

-- Definition of lambda
def lambda : ℝ := 4

-- Proof statement
theorem find_lambda (l : ℝ → ℝ) :
  (∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ hyperbola_equation p1.1 p1.2 ∧ hyperbola_equation p2.1 p2.2 ∧ (dist p1 p2 = lambda)) →
  λ = 4 :=
by
  sorry

end find_lambda_l78_78018


namespace positive_difference_between_sums_l78_78062

def sum_squares_first_six : ℕ := ∑ i in Finset.range 7, i^2
def prime_numbers_between (a b : ℕ) : Finset ℕ := Finset.filter Nat.Prime (Finset.range (b + 1)) \ Finset.range a
def sum_prime_numbers_between_one_sixteen : ℕ := ∑ p in prime_numbers_between 1 16, p

theorem positive_difference_between_sums :
  (sum_squares_first_six - sum_prime_numbers_between_one_sixteen) = 50 := by
  sorry

end positive_difference_between_sums_l78_78062


namespace race_ordering_l78_78291

theorem race_ordering
  (Lotar Manfred Jan Victor Eddy : ℕ) 
  (h1 : Lotar < Manfred) 
  (h2 : Manfred < Jan) 
  (h3 : Jan < Victor) 
  (h4 : Eddy < Victor) : 
  ∀ x, x = Victor ↔ ∀ y, (y = Lotar ∨ y = Manfred ∨ y = Jan ∨ y = Eddy) → y < x :=
by
  sorry

end race_ordering_l78_78291


namespace find_starting_number_sum_to_78_l78_78811

-- Definitions for conditions used in the problem
def is_arithmetic_series (x : ℕ) (sum n : ℕ) : Prop :=
  sum = (n * (x + (x + n - 1))) / 2

def num_terms (start last : ℕ) : ℕ :=
  last - start + 1

-- Proposition stating the problem
theorem find_starting_number_sum_to_78 : 
  ∃ x, is_arithmetic_series x 78 (num_terms x 12) ∧ x < 12 ∧ x = 1 :=
begin
  sorry
end

end find_starting_number_sum_to_78_l78_78811


namespace range_of_m_l78_78634

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x - 1

theorem range_of_m (m : ℝ) (h : ∀ x, m ≤ x ∧ x ≤ m + 1 → f x m < 0) :
  -real.sqrt 2 / 2 < m ∧ m < 0 :=
sorry

end range_of_m_l78_78634


namespace thirty_percent_less_than_80_equals_half_more_than_l78_78817

-- Define the conditions
def thirty_percent_less_than_80 : ℝ := 0.7 * 80 -- Calculate 30% less than 80
def half_more_than (n : ℝ) : ℝ := (3 / 2) * n   -- Define half more than a number

-- State the proof problem
theorem thirty_percent_less_than_80_equals_half_more_than (n : ℝ) :
  thirty_percent_less_than_80 = half_more_than n ↔ n = 112 / 3 :=
by
  sorry

end thirty_percent_less_than_80_equals_half_more_than_l78_78817


namespace sum_lent_is_1050_l78_78462

-- Define the variables for the problem
variable (P : ℝ) -- Sum lent
variable (r : ℝ) -- Interest rate
variable (t : ℝ) -- Time period
variable (I : ℝ) -- Interest

-- Define the conditions
def conditions := 
  r = 0.06 ∧ 
  t = 6 ∧ 
  I = P - 672 ∧ 
  I = P * (r * t)

-- Define the main theorem
theorem sum_lent_is_1050 (P r t I : ℝ) (h : conditions P r t I) : P = 1050 :=
  sorry

end sum_lent_is_1050_l78_78462


namespace subproblem_B_l78_78819

theorem subproblem_B :
  let A : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 1], ![2, 1]]
  let beta : Fin 2 → ℚ := ![1, 2]
  let alpha : Fin 2 → ℚ := ![-1, 2]
  A.mul_vec (A.mul_vec alpha) = beta :=
sorry

end subproblem_B_l78_78819


namespace circumcircles_concurrent_l78_78737

variables (A B C D E F S T : Type) [linear_ordered_field R]
variables [affine_space ℝ] [metric_space ℝ]

-- Conditions
variables (A B C D : Type) [affine_space ℝ] (E F : Type) 
variables [segment A D E] [segment B C F] 
variables (ratio_condition : (AE / ED) = (BF / FC))
variables (S_def : S = (line_segment E F) ∩ (line_segment A B))
variables (T_def : T = (line_segment E F) ∩ (line_segment C D))

-- Statement to prove
theorem circumcircles_concurrent :
  ∃ P : Type, ∃ (P ∈ (circumcircle S A E)) 
             (P ∈ (circumcircle S B F)) 
             (P ∈ (circumcircle T C F)) 
             (P ∈ (circumcircle T D E)), True :=
sorry

end circumcircles_concurrent_l78_78737


namespace largest_25_supporting_X_l78_78131

def is_25_supporting (X : ℝ) : Prop :=
∀ (a : fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : Sup {X : ℝ | is_25_supporting X} = 0.02 :=
sorry

end largest_25_supporting_X_l78_78131


namespace chicken_bucket_feeds_l78_78471

theorem chicken_bucket_feeds :
  ∀ (cost_per_bucket : ℝ) (total_cost : ℝ) (total_people : ℕ),
  cost_per_bucket = 12 →
  total_cost = 72 →
  total_people = 36 →
  (total_people / (total_cost / cost_per_bucket)) = 6 :=
by
  intros cost_per_bucket total_cost total_people h1 h2 h3
  sorry

end chicken_bucket_feeds_l78_78471


namespace avg_growth_rate_l78_78860

def initial_area_2020 := 100
def area_2022 := 196
def time_period_years := 2

theorem avg_growth_rate :
  ∃ x, (initial_area_2020 * (1 + x) ^ time_period_years = area_2022) ∧ x = 0.4 :=
sorry

end avg_growth_rate_l78_78860


namespace geric_initial_bills_l78_78610

theorem geric_initial_bills (G K J : ℕ) 
  (h1: G = 2 * K)
  (h2: K = J - 2)
  (h3: J - 3 = 7) : G = 16 := 
  by 
  sorry

end geric_initial_bills_l78_78610


namespace enclosed_area_formula_l78_78416

noncomputable def enclosed_area (r : ℝ) : ℝ :=
  (4 * Real.sqrt 3 - 11 / 6 * Real.pi) * r ^ 2

theorem enclosed_area_formula {r : ℝ} (hr : 0 < r) :
  let small_circle_radius := r,
      large_circle_radius := 3 * r in
  ∀ a b : ℝ, a = small_circle_radius ∧ b = large_circle_radius →
  enclosed_area r = (4 * Real.sqrt 3 - 11 / 6 * Real.pi) * r ^ 2 :=
by
  intros small_circle_radius large_circle_radius h
  sorry

end enclosed_area_formula_l78_78416


namespace num_correct_propositions_eq_2_l78_78779

-- Declare the propositions as variables.
variables (p1 p2 p3 p4 : Prop)

-- Define each condition from part a)
def condition_1 : Prop := 
  ∀ (a b : ℝ), p1 ↔ (a + b = 180 ∧ a = b) → (a = 90)

def condition_2 : Prop := 
  ∀ (a : ℝ), p2 ↔ (a + a = 180) ∧ (a = 90)

def condition_3 : Prop := 
  ∀ (a b : ℝ), p3 ↔ (a = b) → (some intersection or alignment property, e.g. parallel) ∧ (error inferred)

def condition_4 : Prop := 
  ∀ (a b : ℝ), p4 ↔ (a + b = 90) → (45*2 = 90, so perpendicular)

-- Definition of the resulting proof problem
theorem num_correct_propositions_eq_2
  (h1 : condition_1 p1)
  (h2 : condition_2 p2)
  (h3 : condition_3 p3)
  (h4 : condition_4 p4) :
  (ite p1 1 0 + ite p2 1 0 + ite p3 1 0 + ite p4 1 0) = 2 :=
by
  -- Proof steps would go here, but are omitted by the guidelines
  sorry

end num_correct_propositions_eq_2_l78_78779


namespace steiner_l78_78091

def points_on_conic (A B C D E F : Point) (conic : Set Point) : Prop :=
  A ∈ conic ∧ B ∈ conic ∧ C ∈ conic ∧ D ∈ conic ∧ E ∈ conic ∧ F ∈ conic

theorem steiner (A B C D E F : Point) (conic : Set Point) :
  points_on_conic A B C D E F conic →
  ∃ P : Point, 
  collinear {intersection (line_through A F) (line_through B E),
             intersection (line_through E D) (line_through C F),
             intersection (line_through A D) (line_through B C)} P ∧
  collinear {intersection (line_through A D) (line_through E C),
             intersection (line_through E B) (line_through C F),
             intersection (line_through D E) (line_through B F)} P ∧
  collinear {intersection (line_through A D) (line_through C F),
             intersection (line_through C E) (line_through B F),
             intersection (line_through D C) (line_through E B)} P :=
sorry

end steiner_l78_78091


namespace angle_B_and_max_area_of_triangle_l78_78687

theorem angle_B_and_max_area_of_triangle {A B C a b c : ℝ}
  (h_acute : ∀ {θ : ℝ}, θ ∈ {A, B, C} → 0 < θ ∧ θ < π / 2)
  (h_sides : a = sin A ∧ b = sin B ∧ c = sin C)
  (h_m : ℝ × ℝ := (2 * sin (A + C), sqrt 3))
  (h_n : ℝ × ℝ := (cos (2 * B), 2 * cos_sq (B / 2) - 1))
  (h_collinear : 2 * sin (A + C) * (2 * cos_sq (B / 2) - 1) - sqrt 3 * cos (2 * B) = 0)
  (h_b : b = 1)
  : (B = π / 6) ∧ 
    (S_max : ℝ = (2 + sqrt 3) / 4 → 
    S_max = (1/2) * a * c * sin B) := by
  sorry

end angle_B_and_max_area_of_triangle_l78_78687


namespace steel_mill_production_2010_l78_78890

noncomputable def steel_mill_production (P : ℕ → ℕ) : Prop :=
  (P 1990 = 400000) ∧ (P 2000 = 500000) ∧ ∀ n, (P n) = (P (n-1)) + (500000 - 400000) / 10

theorem steel_mill_production_2010 (P : ℕ → ℕ) (h : steel_mill_production P) : P 2010 = 630000 :=
by
  sorry -- proof omitted

end steel_mill_production_2010_l78_78890


namespace max_snowmen_l78_78483

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78483


namespace tangential_circle_exists_l78_78322

-- Given problem setup
variable {C1 C2 : Circle}
variable {O1 O2 : Point}
variable {l : Line}
variable {P1 P2 : Point}
variable {R : ℝ}

-- Hypotheses
hypothesis (h1 : Circle.radius C1 = R)
hypothesis (h2 : Circle.radius C2 = R)
hypothesis (h3 : C1 ∩ C2 = ∅)
hypothesis (h4 : O1.is_left_of O2)
hypothesis (h5 : ∀ p ∈ (line_points l), p.is_secant_to (C1 ∪ C2))
hypothesis (h6 : P1 ∈ line_points l ∧ P1.is_left_of C1)
hypothesis (h7 : P2 ∈ line_points l ∧ P2.is_right_of C2)
hypothesis (h8 : Quadrilateral (tangent P1 C1) (tangent P2 C2))

-- The statement to prove: there exists a circle tangent to the four sides of the quadrilateral
theorem tangential_circle_exists : 
  ∃ Γ : Circle, Γ.is_tangent_to (tangent P1 C1) ∧ Γ.is_tangent_to (tangent P1 side 2)
                 ∧ Γ.is_tangent_to (tangent P2 C2) ∧ Γ.is_tangent_to (tangent P2 side 2) :=
sorry

end tangential_circle_exists_l78_78322


namespace polygon_sides_sum_l78_78004

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l78_78004


namespace problem_l78_78259

-- Definitions and conditions
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n, 2 ≤ n → 2 * a n / (a n * (Finset.sum (Finset.range n) a) - (Finset.sum (Finset.range n) a) ^ 2) = 1)

-- Sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := Finset.sum (Finset.range n) a

-- The proof statement
theorem problem (a : ℕ → ℚ) (h : seq a) : S a 2017 = 1 / 1009 := sorry

end problem_l78_78259


namespace initial_candies_is_720_l78_78876

-- Definitions according to the conditions
def candies_remaining_after_day_n (initial_candies : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 1 => initial_candies / 2
  | 2 => (initial_candies / 2) / 3
  | 3 => (initial_candies / 2) / 3 / 4
  | 4 => (initial_candies / 2) / 3 / 4 / 5
  | 5 => (initial_candies / 2) / 3 / 4 / 5 / 6
  | _ => 0 -- For days beyond the fifth, this is nonsensical

-- Proof statement
theorem initial_candies_is_720 : ∀ (initial_candies : ℕ), candies_remaining_after_day_n initial_candies 5 = 1 → initial_candies = 720 :=
by
  intros initial_candies h
  sorry

end initial_candies_is_720_l78_78876


namespace sequence_a10_gt_10_l78_78974

noncomputable def sequence (a b : ℝ) : ℕ → ℝ
| 0     := a
| (n+1) := (sequence a b n)^2 + b

theorem sequence_a10_gt_10 (a b : ℝ) (h : b = 0.5) : sequence a b 10 > 10 :=
sorry

end sequence_a10_gt_10_l78_78974


namespace dot_product_is_quarter_l78_78225

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 2))]
variables (a b : (euclidean_space ℝ (fin 2)))

noncomputable def find_dot_product (a b : (euclidean_space ℝ (fin 2))) : ℝ :=
  if (∥a∥ = real.sqrt 3) ∧ (∥b∥ = 1) ∧ (∥a - 2 • b∥ = real.sqrt 6) then
    (a ⬝ b)
  else 0 -- Fallback case, will not be used as per our conditions

theorem dot_product_is_quarter (a b : (euclidean_space ℝ (fin 2))) 
  (ha : ∥a∥ = real.sqrt 3)
  (hb : ∥b∥ = 1)
  (hab : ∥a - 2 • b∥ = real.sqrt 6) :
  (a ⬝ b) = 1 / 4 :=
by
  sorry -- Proof steps go here.

end dot_product_is_quarter_l78_78225


namespace subcommittee_count_l78_78775

theorem subcommittee_count (total_members council_officers : ℕ) (subcommittee_size : ℕ) 
  (h1 : total_members = 12) (h2 : council_officers = 5) (h3 : subcommittee_size = 5) :
  (∑ k in (finset.range (council_officers + 1)), if k >= 2 then nat.choose council_officers k * nat.choose (total_members - council_officers) (subcommittee_size - k) else 0) = 596 :=
by
  sorry

end subcommittee_count_l78_78775


namespace min_mnp_l78_78674

theorem min_mnp (m n p : ℕ) (hm : m.prime) (hn : n.prime) (hp : p.prime) (hdiff : m ≠ n ∧ m ≠ p ∧ n ≠ p) (h_sum : m + n = p) :
  m * n * p = 30 :=
sorry

end min_mnp_l78_78674


namespace regular_polygon_perimeter_l78_78534

theorem regular_polygon_perimeter (s : ℝ) (exterior_angle : ℝ) 
  (h1 : s = 7) (h2 : exterior_angle = 45) : 
  8 * s = 56 :=
by
  sorry

end regular_polygon_perimeter_l78_78534


namespace polygon_inscribed_circle_radius_l78_78866

theorem polygon_inscribed_circle_radius :
  (∀ n : ℕ, n = 12) →
  (∀ m : ℕ, m = 6) →
  (∀ a b : ℝ, a = sqrt 2 ∧ b = sqrt 24) →
  (∃ r : ℝ, r = 4 * sqrt 2) :=
by
  intro n m a b hn hm hab
  use 4 * sqrt 2
  sorry

end polygon_inscribed_circle_radius_l78_78866


namespace proof_negation_l78_78791

-- Definitions of rational and real numbers
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- Proposition stating the existence of an irrational number that is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational x

-- Negation of the original proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬ is_rational x

theorem proof_negation : ¬ original_proposition = negated_proposition := 
sorry

end proof_negation_l78_78791


namespace count_numbers_in_sequence_l78_78648

theorem count_numbers_in_sequence : 
  let sequence : List ℕ := List.range' 45 166
    |>.filter (fun n => (n % 3 = 0) && (n <= 165))
  in sequence.length = 41 := by
  sorry

end count_numbers_in_sequence_l78_78648


namespace area_of_inscribed_pentagon_l78_78883

def inscribed_pentagon (A B C D E : ℝ) (R : ℝ) (AB AE BE BD : ℝ) :=
  R = 1 ∧
  AB = real.sqrt 2 ∧
  ∃ angle_ABE : ℝ, angle_ABE = 45 ∧
  ∃ angle_EBD : ℝ, angle_EBD = 30 ∧
  BD = BE * real.cos (angle_EBD * real.pi / 180) ∧
  BC = CD

theorem area_of_inscribed_pentagon (A B C D E : ℝ) (R : ℝ) (AB AE BE BD : ℝ) (AB_sqrt AE_sqrt BD_sqrt : ℝ):
  inscribed_pentagon A B C D E R AB AE BE BD →
  BD_sqrt = (BD / 2) * real.tan (120 * real.pi / 180) →
  let S_ABC : ℝ := 1/2 * AB * AE in
  let S_BDE : ℝ := 1/2 * BD * real.sqrt (BD) in
  let S_DCB : ℝ := 1/2 * BD * (BD_sqrt) in
  S_ABC + S_BDE + S_DCB = 1 + 3 * real.sqrt(3) / 4 :=
sorry

end area_of_inscribed_pentagon_l78_78883


namespace range_of_f_ffn_is_1278_l78_78950

noncomputable def e : ℝ := 2.71828182845 -- Assume this is an approximation of e for simplicity

noncomputable def digit_of_e (n : ℕ) : ℕ :=
  -- This function needs to be defined such that it returns the n-th digit after the decimal point of e
  sorry

noncomputable def f (n : ℕ) : ℕ :=
  if n = 0 then 2 else digit_of_e n

theorem range_of_f_ffn_is_1278 : (set.range (λ n : ℕ, f (f (f n)))) = {1, 2, 7, 8} :=
by sorry

end range_of_f_ffn_is_1278_l78_78950


namespace fair_hair_percentage_l78_78680

theorem fair_hair_percentage
    (women_fair_hair_percent : ℝ)
    (fair_hair_women_long_hair_percent : ℝ)
    (fair_hair_women_short_hair_percent : ℝ)
    (women_fair_hair_ratio : ℝ)
    (fair_hair_blue_eyes_percent : ℝ)
    (fair_hair_green_eyes_percent : ℝ)
    (senior_position_blue_eyes_fair_hair_percent : ℝ) :
    women_fair_hair_percent = 0.28 →
    fair_hair_women_long_hair_percent = 0.60 →
    fair_hair_women_short_hair_percent = 0.40 →
    women_fair_hair_ratio = 0.40 →
    fair_hair_blue_eyes_percent = 0.35 →
    fair_hair_green_eyes_percent = 0.65 →
    senior_position_blue_eyes_fair_hair_percent = 0.25 →
    let total_fair_hair_percent := (women_fair_hair_percent / women_fair_hair_ratio)
    in total_fair_hair_percent = 0.70 :=
by
    intros
    let total_fair_hair_percent := (women_fair_hair_percent / women_fair_hair_ratio)
    show total_fair_hair_percent = 0.70, from sorry

end fair_hair_percentage_l78_78680


namespace total_students_sum_is_90_l78_78294

theorem total_students_sum_is_90:
  ∃ (x y z : ℕ), 
  (80 * x - 100 = 92 * (x - 5)) ∧
  (75 * y - 150 = 85 * (y - 6)) ∧
  (70 * z - 120 = 78 * (z - 4)) ∧
  (x + y + z = 90) :=
by
  sorry

end total_students_sum_is_90_l78_78294


namespace find_width_of_lot_l78_78689

noncomputable def volume_of_rectangular_prism (l w h : ℝ) : ℝ := l * w * h

theorem find_width_of_lot
  (l h v : ℝ)
  (h_len : l = 40)
  (h_height : h = 2)
  (h_volume : v = 1600)
  : ∃ w : ℝ, volume_of_rectangular_prism l w h = v ∧ w = 20 := by
  use 20
  simp [volume_of_rectangular_prism, h_len, h_height, h_volume]
  sorry

end find_width_of_lot_l78_78689


namespace james_spends_on_pistachios_per_week_l78_78714

theorem james_spends_on_pistachios_per_week :
  let cost_per_can := 10
  let ounces_per_can := 5
  let total_ounces_per_5_days := 30
  let days_per_week := 7
  let cost_per_ounce := cost_per_can / ounces_per_can
  let daily_ounces := total_ounces_per_5_days / 5
  let daily_cost := daily_ounces * cost_per_ounce
  daily_cost * days_per_week = 84 :=
by
  sorry

end james_spends_on_pistachios_per_week_l78_78714


namespace find_m_l78_78240

theorem find_m (m : ℝ) : (∃ (x y : ℝ), (x^2 + y^2 - 4*x - m = 0) ∧ (π * (√1)^2 = π)) → (m = -3) :=
by
  intros h
  cases h with x hx
  cases hx with y hxy
  sorry

end find_m_l78_78240


namespace plant_supplier_money_left_correct_l78_78139

noncomputable def plant_supplier_total_earnings : ℕ :=
  35 * 52 + 30 * 32 + 20 * 77 + 25 * 22 + 40 * 15

noncomputable def plant_supplier_total_expenses : ℕ :=
  3 * 65 + 2 * 45 + 280 + 150 + 100 + 125 + 225 + 550

noncomputable def plant_supplier_money_left : ℕ :=
  plant_supplier_total_earnings - plant_supplier_total_expenses

theorem plant_supplier_money_left_correct :
  plant_supplier_money_left = 3755 :=
by
  sorry

end plant_supplier_money_left_correct_l78_78139


namespace jeff_ends_at_multiple_of_3_with_probability_1_over_8_l78_78314

-- Spinners results
inductive SpinnerResult
| right2
| right2
| left1
| stay

open SpinnerResult

-- Define the probability of each SpinnerResult
def prob (r : SpinnerResult) : ℚ :=
  match r with
  | right2 => 1 / 4
  | right2 => 1 / 4
  | left1 => 1 / 4
  | stay => 1 / 4

-- Define the effect on the number line for each result
def effect (r : SpinnerResult) (p : ℤ) : ℤ :=
  match r with
  | right2 => p + 2
  | left1 => p - 1
  | stay => p

-- Define starting positions
def starting_positions : Finset ℤ := Finset.range 13 \ {0}
def multiples_of_3 : Finset ℤ := starting_positions.filter (λ n => n % 3 = 0)
def one_more_than_multiple_of_3 : Finset ℤ := starting_positions.filter (λ n => n % 3 = 1)
def two_more_than_multiple_of_3 : Finset ℤ := starting_positions.filter (λ n => n % 3 = 2)

-- Define the final position after two spins
def final_position (init : ℤ) (r1 r2 : SpinnerResult) : ℤ :=
  effect r2 (effect r1 init)

-- Calculate probabilities
def probability_of_ending_at_multiple_of_3 : ℚ :=
  let outcomes := [right2, right2, left1, stay];
  let probabilities := outcomes.product outcomes;
  let total_prob := 1 / 12 * 
    (4 * (3 / 16) +
     3 * (1 / 16) +
     5 * (1 / 16));
  total_prob

-- Prove the probability is 1/8
theorem jeff_ends_at_multiple_of_3_with_probability_1_over_8 :
  probability_of_ending_at_multiple_of_3 = 1 / 8 :=
by
  -- This is a placeholder for proof, so we use 'sorry'
  sorry

end jeff_ends_at_multiple_of_3_with_probability_1_over_8_l78_78314


namespace nonneg_sol_eq_l78_78587

theorem nonneg_sol_eq {a b c : ℝ} (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c) 
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) : 
  a = b ∧ b = c := 
sorry

end nonneg_sol_eq_l78_78587


namespace fish_added_l78_78656

theorem fish_added (T C : ℕ) (h1 : T + C = 20) (h2 : C = T - 4) : C = 8 :=
by
  sorry

end fish_added_l78_78656


namespace intersecting_circles_area_l78_78387

open Real

theorem intersecting_circles_area 
  (r : ℝ) (θ : ℝ) (a b c : ℝ) :
  r = 5 ∧ θ = π / 2 ∧ 
  let total_area := 3 * (1/4 * π * r^2)
  let overlap := (π * r^2 - (5^2 * sqrt 3 / 4)) / 3 
  let final_area := total_area - overlap
  final_area = 25 * sqrt 3 / 12 + 12.5 * π → 
  a = 25 ∧ b = 3 ∧ c = 12.5 →
  a + b + c = 40.5 :=
by {
  intros h hr;
  sorry
}

end intersecting_circles_area_l78_78387


namespace condition1_condition2_condition3_condition4_l78_78897

-- Define the conditions 1 through 4 as individual theorems to be proved
theorem condition1 (n m : ℕ) : C n m = A n m / m! := sorry

theorem condition2 (n m : ℕ) : C n m = C n (n - m) := sorry

theorem condition3 (n r : ℕ) : C (n + 1) r = C n r + C n (r - 1) := sorry

theorem condition4 (n m : ℕ) : A (n + 2) (m + 2) = (n + 2) * (n + 1) * A n m := sorry

end condition1_condition2_condition3_condition4_l78_78897


namespace inverse_linear_intersection_l78_78980

theorem inverse_linear_intersection (m n : ℝ) 
  (h1 : n = 2 / m) 
  (h2 : n = m + 3) 
  : (1 / m) - (1 / n) = 3 / 2 := 
by sorry

end inverse_linear_intersection_l78_78980


namespace oliver_final_amount_l78_78356

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l78_78356


namespace fraction_comparison_l78_78051

theorem fraction_comparison :
  (2 : ℝ) * (4 : ℝ) > (7 : ℝ) → (4 / 7 : ℝ) > (1 / 2 : ℝ) :=
by
  sorry

end fraction_comparison_l78_78051


namespace find_N_l78_78945

def consecutive_product_sum_condition (a : ℕ) : Prop :=
  a*(a + 1)*(a + 2) = 8*(a + (a + 1) + (a + 2))

theorem find_N : ∃ (N : ℕ), N = 120 ∧ ∃ (a : ℕ), a > 0 ∧ consecutive_product_sum_condition a := by
  sorry

end find_N_l78_78945


namespace total_exercise_time_l78_78310

theorem total_exercise_time :
  let javier_minutes_per_day := 50
  let javier_days := 7
  let sanda_minutes_per_day := 90
  let sanda_days := 3
  (javier_minutes_per_day * javier_days + sanda_minutes_per_day * sanda_days) = 620 :=
by
  sorry

end total_exercise_time_l78_78310


namespace initial_rate_oranges_per_rupee_l78_78119

theorem initial_rate_oranges_per_rupee:
  ∃ C : ℝ, (1.44 * C = 1 / 11) → (0.99 * C = 1 / 16) :=
begin
  sorry,
end

end initial_rate_oranges_per_rupee_l78_78119


namespace maximum_snowmen_count_l78_78495

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78495


namespace alina_masha_difference_l78_78552

-- Definition of the problem's assumptions
def initially_proposed := 27
def remaining_problems := 10
def rejected_problems := initially_proposed - remaining_problems

variables (A M : ℕ)

-- Conditions provided in the problem
def total_problems_condition := A + M = initially_proposed
def rejection_condition := A / 2 + 2 * M / 3 = rejected_problems

-- Statement to be proved
theorem alina_masha_difference (h1 : total_problems_condition) 
                               (h2 : rejection_condition) : 
  M - A = 15 :=
sorry

end alina_masha_difference_l78_78552


namespace angle_DEF_measure_l78_78022

-- Definitions based on conditions in a)
def tangents_drawn_from_point (D E F O : Point) (radius : ℝ) : Prop :=
  circle O radius ∧ tangent D E O ∧ tangent D F O

def arcs_ratio (arcEF arcFE' : ℝ) : Prop :=
  arcEF / arcFE' = 3 / 5

-- Lean theorem based on translating the problem
theorem angle_DEF_measure (D E F O : Point) (radius arcEF arcFE' : ℝ)
  (h1 : tangents_drawn_from_point D E F O radius)
  (h2 : arcs_ratio arcEF arcFE') :
  angle_measure DEF = 67.5 :=
sorry

end angle_DEF_measure_l78_78022


namespace find_total_students_l78_78293

-- Let's define the conditions as given in the problem
variables (S : ℝ) -- Total number of students

-- Percentage conditions
def percent_math := 0.44
def percent_bio := 0.40
def percent_math_and_socio := 0.30

-- Given condition about number of students studying only biology
def only_bio := 180

-- We need to prove that the total number of students is 1800
theorem find_total_students 
  (h_math : 0.44 * S)
  (h_bio : 0.40 * S)
  (h_math_and_socio : 0.30 * S)
  (h_only_bio : 0.10 * S = only_bio) : S = 1800 :=
begin
  sorry
end

end find_total_students_l78_78293


namespace measure_angle_ADB_140_l78_78283

variable {A B C D : Type} -- Define the points A, B, C, D as types

-- Given conditions
variables {triangle_A : A} {triangle_B : B} {triangle_C : C} {point_D : D}
variable [has_angle : Triangle.angle measure (triangle_A, triangle_B, triangle_C)]
variables (BD_eq_DC : dist(B, D) = dist(D, C))
variable (angle_BCD_70 : measure (angle B C D) = 70)

-- Prove that the measure of angle ADB equals 140 degrees
theorem measure_angle_ADB_140 (h1 : BD_eq_DC) (h2 : angle_BCD_70) :
  measure (angle A D B) = 140 :=
sorry

end measure_angle_ADB_140_l78_78283


namespace N_subset_M_values_l78_78643

def M : Set ℝ := { x | 2 * x^2 - 3 * x - 2 = 0 }
def N (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem N_subset_M_values (a : ℝ) (h : N a ⊆ M) : a = 0 ∨ a = -2 ∨ a = 1/2 := 
by
  sorry

end N_subset_M_values_l78_78643


namespace find_number_l78_78038

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l78_78038


namespace largest_25_supporting_X_l78_78120

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i : ℝ) ∈ Set.Ico (- (1 / 2) : ℝ) ((25 : ℝ) / 2 + (1 / 2)) → 
  ∃ i, |a i - 1 / 2| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ 
  (∀ Y : ℝ, Y > X → ¬is_25_supporting Y) ∧ X = 0.02 :=
sorry

end largest_25_supporting_X_l78_78120


namespace largest_supporting_25_X_l78_78125

def is_supporting_25 (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ X

theorem largest_supporting_25_X :
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ 0.02 :=
by {
  sorry
}

end largest_supporting_25_X_l78_78125


namespace first_digit_base8_of_891_is_1_l78_78029

theorem first_digit_base8_of_891_is_1 : ∀ (n : ℕ), n = 891 → (∃ d : ℕ, d = (891 / 8^3) ∧ d = 1) :=
by
  intros n H
  have H1 : 8^3 = 512 := by norm_num
  have H2 : 891 / 512 = 1 := by norm_num
  use 1
  split
  { exact H2 }
  { norm_num }
  sorry

end first_digit_base8_of_891_is_1_l78_78029


namespace largest_25_supporting_X_l78_78134

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i) ∈ Int → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ ∀ Y : ℝ, (is_25_supporting Y → Y ≤ 0.02) :=
sorry

end largest_25_supporting_X_l78_78134


namespace velma_vs_daphne_l78_78421

def veronica_visibility : ℝ := 1000

def freddie_visibility (V : ℝ) : ℝ := 3 * V

def velma_visibility (F : ℝ) : ℝ := 5 * F - 2000

def daphne_visibility (V F Ve : ℝ) : ℝ := (V + F + Ve) / 3

theorem velma_vs_daphne (V : ℝ) (F : ℝ) (Ve : ℝ) (D : ℝ) :
  V = 1000 →
  F = freddie_visibility V →
  Ve = velma_visibility F →
  D = daphne_visibility V F Ve →
  Ve - D = 7666.67 :=
by
  sorry

end velma_vs_daphne_l78_78421


namespace james_weekly_pistachio_cost_l78_78716

def cost_per_can : ℕ := 10
def ounces_per_can : ℕ := 5
def consumption_per_5_days : ℕ := 30
def days_per_week : ℕ := 7

theorem james_weekly_pistachio_cost : (days_per_week / 5 * consumption_per_5_days) / ounces_per_can * cost_per_can = 90 := 
by
  sorry

end james_weekly_pistachio_cost_l78_78716


namespace solve_eq_l78_78765

theorem solve_eq : ∃ x : ℚ, 3 * x + 5 * x = 600 - (4 * x + 6 * x) ∧ x = 100 / 3 := by
  use 100 / 3
  split
  · linarith
  · rfl

end solve_eq_l78_78765


namespace proposition_form_l78_78783

-- Definitions based on the conditions
def p : Prop := (12 % 4 = 0)
def q : Prop := (12 % 3 = 0)

-- Problem statement to prove
theorem proposition_form : p ∧ q :=
by
  sorry

end proposition_form_l78_78783


namespace max_distinct_fans_l78_78076

-- Define the problem conditions and the main statement
theorem max_distinct_fans : 
  let n := 6  -- Number of sectors per fan
  let configurations := 2^n  -- Total configurations without considering symmetry
  let unchanged_flips := 8  -- Number of configurations unchanged by flipping
  let distinct_configurations := (configurations - unchanged_flips) / 2 + unchanged_flips 
  in 
  distinct_configurations = 36 := by sorry  # We state the answer based on the provided steps


end max_distinct_fans_l78_78076


namespace equal_areas_of_shaded_quadrilaterals_l78_78021

noncomputable def quadrilateral := sorry

theorem equal_areas_of_shaded_quadrilaterals
  (square1 square2 : quadrilateral)
  (parallelogram1 parallelogram2 : quadrilateral)
  (K D L B M P : quadrilateral_Point) :
  is_square square1 → is_square square2 → 
  parallelograms_are_congruent parallelogram1 parallelogram2 →
  quadrilateral1_Area K D L B = quadrilateral2_Area B_1 M D_1 P :=
  sorry

end equal_areas_of_shaded_quadrilaterals_l78_78021


namespace distinct_fan_count_l78_78090

def max_distinct_fans : Nat :=
  36

theorem distinct_fan_count (n : Nat) (r b : S) (paint_scheme : Fin n → bool) :
  (∀i, r ≠ b → (paint_scheme i = b ∨ paint_scheme i = r)) ∧ 
  (∀i, paint_scheme i ≠ paint_scheme (i + n / 2 % n)) →
  n = 6 →
  max_distinct_fans = 36 :=
by
  sorry

end distinct_fan_count_l78_78090


namespace sugar_cone_count_l78_78160

theorem sugar_cone_count (ratio_sugar_waffle : ℕ → ℕ → Prop) (sugar_waffle_ratio : ratio_sugar_waffle 5 4) 
(w : ℕ) (h_w : w = 36) : ∃ s : ℕ, ratio_sugar_waffle s w ∧ s = 45 :=
by
  sorry

end sugar_cone_count_l78_78160


namespace max_min_values_l78_78182

-- Define the function f(x) = x^2 - 2ax + 1
def f (x a : ℝ) : ℝ := x ^ 2 - 2 * a * x + 1

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem max_min_values (a : ℝ) : 
  (a > 2 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = 5 - 4 * a))
  ∧ (1 ≤ a ∧ a ≤ 2 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (0 ≤ a ∧ a < 1 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (a < 0 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = 1)) := by
  sorry

end max_min_values_l78_78182


namespace count_triples_l78_78652

open Set

theorem count_triples 
  (A B C : Set ℕ) 
  (h_union : A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (h_inter : A ∩ B ∩ C = ∅) :
  (∃ n : ℕ, n = 60466176) :=
by
  -- Proof can be filled in here
  sorry

end count_triples_l78_78652


namespace max_number_of_snowmen_l78_78517

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78517


namespace M_inter_N_l78_78281

def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def N : Set ℝ := { x | x ≤ 1/2 }

theorem M_inter_N :
  (∃ y : ℝ, y ∈ M ∧ y ∈ N) ↔ (0, 1/2] := 
sorry

end M_inter_N_l78_78281


namespace degrees_for_basic_astrophysics_correct_l78_78434

-- Definitions for conditions
def percentage_allocations : List ℚ := [13, 24, 15, 29, 8]
def total_percentage : ℚ := percentage_allocations.sum
def remaining_percentage : ℚ := 100 - total_percentage

-- The question to answer
def total_degrees : ℚ := 360
def degrees_for_basic_astrophysics : ℚ := remaining_percentage / 100 * total_degrees

-- Prove that the degrees for basic astrophysics is 39.6
theorem degrees_for_basic_astrophysics_correct :
  degrees_for_basic_astrophysics = 39.6 :=
by
  sorry

end degrees_for_basic_astrophysics_correct_l78_78434


namespace valid_quadratic_polynomials_l78_78197

theorem valid_quadratic_polynomials (b c : ℤ)
  (h₁ : ∃ x₁ x₂ : ℤ, b = -(x₁ + x₂) ∧ c = x₁ * x₂)
  (h₂ : 1 + b + c = 10) :
  (b = -13 ∧ c = 22) ∨ (b = -9 ∧ c = 18) ∨ (b = 9 ∧ c = 0) ∨ (b = 5 ∧ c = 4) := sorry

end valid_quadratic_polynomials_l78_78197


namespace polygon_sides_sum_l78_78003

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l78_78003


namespace boat_breadth_l78_78869

noncomputable theory

open Classical

variables (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (ρ : ℝ)

/-- Given the length of the boat is 6 meters, the boat sinks by 1 cm when a man with a mass of 180 kg 
gets on it, the density of water is approximately 1000 kg/m³, and the acceleration due to gravity 
is approximately 9.81 m/s², prove that the breadth of the boat is 3 meters. -/
theorem boat_breadth
    (hl : L = 6)
    (hh : h = 0.01)
    (hm : m = 180)
    (hg : g = 9.81)
    (hρ : ρ = 1000) :
    let W := m * g in
    let V := W / (ρ * g) in
    let B := V / (L * h) in
    B = 3 :=
by
  rw [hl, hh, hm, hg, hρ]
  simp [W, V, B]
  sorry -- proof skipping

end boat_breadth_l78_78869


namespace add_neg_eq_sub_l78_78444

theorem add_neg_eq_sub (a b : ℤ) : a + (-b) = a - b := by sorry

example : -3 + 5 = 2 := by
  rw add_comm (-3) 5
  rw add_neg_eq_sub
  norm_num

end add_neg_eq_sub_l78_78444


namespace nth_equation_pattern_l78_78351

theorem nth_equation_pattern (n : ℕ) : 
  let lhs := (list.range (2 * n - 1)).map (λ x, n + x) in 
  (lhs.sum = (2 * n - 1)^2) :=
by
  sorry

end nth_equation_pattern_l78_78351


namespace find_t_l78_78206

variable (a b : ℂ)

-- Given conditions
axiom ha : |a| = 2
axiom hb : |b| = sqrt 26
axiom hab : a * b = t - 2 * complex.i

-- Proof statement
theorem find_t (h_ab : a * b = t - 2 * complex.i) (h_a : |a| = 2) (h_b : |b| = sqrt 26) : t = 10 := by
sorry

end find_t_l78_78206


namespace f_monotonic_interval_range_of_a_l78_78248

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := ln x - a * x + 1

-- Define the function g
def g (x : ℝ) : ℝ := ln x - x / 4 + 3 / (4 * x)

-- First part: Monotonic interval of f(x)
theorem f_monotonic_interval (a : ℝ) : 
  (a ≤ 0 → ∀ x > 0, f x a > f 0 a) ∧ 
  (a > 0 → ∀ x₁ x₂ > 0, x₁ < x₂ → (x > 1/a → f x₁ a < f x₂ a) ∧ 
                                         (x < 1/a → f x₁ a > f x₂ a)) :=
sorry

-- Second part: Range of a such that given condition holds
theorem range_of_a : (∀ x₁ ∈ Ioi (0:ℝ), ∃ x₂ ∈ Ioi 1, f x₁ a < g x₂) ↔ a > 1/3 * exp(1/2) :=
sorry

end f_monotonic_interval_range_of_a_l78_78248


namespace tina_mother_age_l78_78361

variable {x : ℕ}

theorem tina_mother_age (h1 : 10 + x = 2 * x - 20) : 2010 + x = 2040 :=
by 
  sorry

end tina_mother_age_l78_78361


namespace exists_prime_divisor_exclusive_l78_78948

theorem exists_prime_divisor_exclusive (n : ℕ) (hn : n > 1) (l : ℕ) :
  ∀ k ∈ finset.range (n + 1), ∃ p : ℕ, nat.prime p ∧ p ∣ (n * l + k) ∧ ∀ j ∈ finset.range (n + 1).filter (≠ k), ¬p ∣ (n * l + j) :=
by
  sorry

end exists_prime_divisor_exclusive_l78_78948


namespace find_C_coordinates_l78_78540

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -3, y := 5 }
def B : Point := { x := 9, y := -1 }
def C : Point := { x := 15, y := -4 }

noncomputable def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

theorem find_C_coordinates :
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  C = C_actual :=
by
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  show C = C_actual
  rfl

end find_C_coordinates_l78_78540


namespace oliver_money_left_l78_78359

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l78_78359


namespace calculate_days_l78_78382

variable (a d g : ℝ)

-- Define original conditions
def original_rate_per_worker := d / (a * g)

-- Define the increased rate due to technology
def increased_rate_per_worker := 1.25 * original_rate_per_worker a d g

-- Define the equation to compute the number of days y
def new_time (a d g : ℝ) : Prop :=
  ∃ y : ℝ, a = (increased_rate_per_worker a d g) * (g * y)

-- Result to prove the number of days y is as expected
theorem calculate_days (a d g : ℝ) (h : new_time a d g) : 
  ∃ y : ℝ, y = a^2 / (1.25 * d) :=
sorry

end calculate_days_l78_78382


namespace total_cost_expression_minimum_cost_at_100_l78_78017

noncomputable def total_cost (x : ℝ) : ℝ :=
  if 50 ≤ x ∧ x ≤ 100 then
    (1820 * x ^ 2 + 520 * x + 260) / x ^ 2
  else
    0

theorem total_cost_expression :
  ∀ x : ℝ, 50 ≤ x ∧ x ≤ 100 → total_cost(x) = (1820 * x^2 + 520 * x + 260) / x^2 :=
by
  intros x hx
  rw total_cost
  simp [hx]

theorem minimum_cost_at_100 :
  total_cost 100 = 1872.26 :=
by
  rw total_cost
  norm_num
  sorry

end total_cost_expression_minimum_cost_at_100_l78_78017


namespace initial_problems_eq_60_l78_78372

-- Define the conditions as variables/constants
def finished_problems : Nat := 20
def pages_remaining : Nat := 5
def problems_per_page : Nat := 8

-- Define the initial number of homework problems and its proof
theorem initial_problems_eq_60 : 
  let total_problems_remaining := pages_remaining * problems_per_page
  let total_problems := finished_problems + total_problems_remaining
  total_problems = 60 :=
by
  let total_problems_remaining := pages_remaining * problems_per_page
  let total_problems := finished_problems + total_problems_remaining
  exact calc
    total_problems_remaining = 5 * 8 := by rfl
    _ = 40 := by norm_num
    total_problems = 20 + 40 := by rfl
    _ = 60 := by norm_num

end initial_problems_eq_60_l78_78372


namespace graph_f_plus_3_is_E_l78_78786

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Default case, though the given intervals suffice for our use

def f_plus_3 (x : ℝ) : ℝ := f x + 3

theorem graph_f_plus_3_is_E :
  -- Here we need a formal way to represent that the graph corresponds to "E"
  sorry

end graph_f_plus_3_is_E_l78_78786


namespace mother_age_when_harry_born_l78_78267

theorem mother_age_when_harry_born:
  (harry_age father's_age mother's_age : ℚ) 
  (h1 : harry_age = 50)
  (h2 : father's_age = harry_age * (1 + 0.33))
  (h3 : mother's_age = father's_age * (1 - 0.04))
  : mother's_age - harry_age = 13.84 :=
by
  simp [h1, h2, h3]
  sorry

end mother_age_when_harry_born_l78_78267


namespace sum_of_5_and_8_l78_78426

theorem sum_of_5_and_8 : 5 + 8 = 13 := by
  rfl

end sum_of_5_and_8_l78_78426


namespace circle_through_M_A_B_tangent_line_AB_passes_fixed_point_l78_78255

theorem circle_through_M_A_B_tangent (m : ℝ) (hm : m > 0) :
  let M : ℝ × ℝ := (0, -1),
      A : ℝ × ℝ := (2, 1),
      B : ℝ × ℝ := (-2, 1),
      circle_eq := x^2 + (y - 1)^2 = 4 in
  (circle_eq.substitute { x := M.1, y := M.2 }) = true ∧
  (circle_eq.substitute { x := A.1, y := A.2 }) = true ∧
  (circle_eq.substitute { x := B.1, y := B.2 }) = true ∧
  ∀ y, y = -1 → circle_center := (0, 1) → distance M circle_center = 2 ∧
  circle_radius := 2 ∧
  circle_eq.same as l_eq :=
  sorry

theorem line_AB_passes_fixed_point (x1 y1 x2 y2 x0 y0 m : ℝ) (hm : m > 0) (hx1 : x1^2 = 4 * y1) (hx2 : x2^2 = 4 * y2) :
  let M : ℝ × ℝ := (x0, y0),
      A : ℝ × ℝ := (x1, y1),
      B : ℝ × ℝ := (x2, y2) in
  y0 = -m →
  ∀ (x : ℝ), M ∈ l_eq
  (A ≠ B → line_through A B passes (0, m)) :=
  sorry

end circle_through_M_A_B_tangent_line_AB_passes_fixed_point_l78_78255


namespace min_mnp_l78_78672

open Nat

-- We first declare the primes and the specified conditions as theorem

theorem min_mnp (m n p : ℕ) (hm : Prime m) (hn : Prime n) (hp : Prime p) (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p) (h_add : m + n = p) :
  m * n * p ≥ 30 :=
by
  sorry

end min_mnp_l78_78672


namespace max_snowmen_l78_78511

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78511


namespace min_mnp_l78_78675

theorem min_mnp (m n p : ℕ) (hm : m.prime) (hn : n.prime) (hp : p.prime) (hdiff : m ≠ n ∧ m ≠ p ∧ n ≠ p) (h_sum : m + n = p) :
  m * n * p = 30 :=
sorry

end min_mnp_l78_78675


namespace find_number_l78_78042

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l78_78042


namespace max_snowmen_l78_78498

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78498


namespace range_of_a_value_of_a_l78_78227

-- Definitions based on conditions
def circle (x y : ℝ) := (x - 1)^2 + y^2 = 25
def line (a x y : ℝ) := ax - y + 5 = 0
def intersects_at_two_points (a : ℝ) : Prop :=
  let discriminant := ((a + 5)^2 - 25 * (a^2 + 1)) in
  discriminant > 0

-- Lean statement for part 1
theorem range_of_a (a : ℝ) : intersects_at_two_points a -> 
  a < 0 ∨ a > 5/12 :=
by
  sorry

-- Additional condition for part 2
def perpendicular_bisector_passes_through (a : ℝ) (P : ℝ × ℝ) : Prop :=
  P = (-2, 4)

-- Lean statement for part 2
theorem value_of_a (a : ℝ) : intersects_at_two_points a → 
  perpendicular_bisector_passes_through a (-2, 4) -> 
  a = 3/4 :=
by
  sorry

end range_of_a_value_of_a_l78_78227


namespace max_distinct_fans_l78_78070

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l78_78070


namespace find_number_l78_78039

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l78_78039


namespace percentage_for_overnight_stays_l78_78719

noncomputable def total_bill : ℝ := 5000
noncomputable def medication_percentage : ℝ := 0.50
noncomputable def food_cost : ℝ := 175
noncomputable def ambulance_cost : ℝ := 1700

theorem percentage_for_overnight_stays :
  let medication_cost := medication_percentage * total_bill
  let remaining_bill := total_bill - medication_cost
  let cost_for_overnight_stays := remaining_bill - food_cost - ambulance_cost
  (cost_for_overnight_stays / remaining_bill) * 100 = 25 :=
by
  sorry

end percentage_for_overnight_stays_l78_78719


namespace count_random_events_l78_78554

theorem count_random_events :
  let events := [{name := "Event A", is_random := true},
                 {name := "Event B", is_random := true},
                 {name := "Event C", is_random := true}] in
    (events.filter (λ e, e.is_random)).length = 3 := 
by 
  sorry

end count_random_events_l78_78554


namespace temperature_difference_l78_78751

def highest_temperature : ℝ := 8
def lowest_temperature : ℝ := -1

theorem temperature_difference : highest_temperature - lowest_temperature = 9 := by
  sorry

end temperature_difference_l78_78751


namespace sides_of_polygon_l78_78002

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l78_78002


namespace range_of_a_l78_78968

-- Defining the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 5) → x^2 - (a + 1) * x + a ≤ 0
def q (a : ℝ) : Prop := 3 < a ∧ a < 6

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : ¬(p a) ∧ q a) : 3 < a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l78_78968


namespace ivan_income_tax_l78_78710

-- Define the salary schedule
def first_two_months_salary: ℕ := 20000
def post_probation_salary: ℕ := 25000
def bonus_in_december: ℕ := 10000
def income_tax_rate: ℝ := 0.13

-- Define the total taxable income
def total_taxable_income: ℕ :=
  (first_two_months_salary * 2) + (post_probation_salary * 8) + bonus_in_december

-- Define the expected tax amount
def expected_tax: ℕ := 32500

-- Define the personal income tax calculation function
def calculate_tax (income: ℕ) (rate: ℝ): ℕ :=
  (income * rate).toInt

-- The statement which shows that the calculated tax is equal to the expected tax
theorem ivan_income_tax: calculate_tax total_taxable_income income_tax_rate = expected_tax := by
  -- Skip the actual proof
  sorry

end ivan_income_tax_l78_78710


namespace final_share_approx_l78_78812

-- Define the given conditions
def total_bill := 211.00
def number_of_people := 7
def tip_percentage := 0.15

-- Define calculation for tip
def tip := tip_percentage * total_bill

-- Define calculation for total amount with tip
def total_amount_with_tip := total_bill + tip

-- Define calculation for each person's share
def each_person_share := total_amount_with_tip / number_of_people

-- State the theorem
theorem final_share_approx (approx_val : ℝ) : abs (each_person_share - approx_val) < 0.01 :=
sorry

-- In this case, the user is supposed to prove that the approximated value (34.66) is indeed the final share of each person within a small tolerance (0.01).

end final_share_approx_l78_78812


namespace game_ends_after_six_rounds_l78_78464

/--  We define the initial number of tokens for each player. -/
def initial_tokens : ℕ × ℕ × ℕ := (16, 14, 12)

/-- A function to simulate the game for a given number of rounds
    and return the final number of tokens for each player. -/
def play_game : ℕ → ℕ × ℕ × ℕ → ℕ × ℕ × ℕ
| 0, tokens := tokens
| (n + 1), (a, b, c) :=
  if a > b ∧ a > c then 
    if a > 10 then play_game n (a - 4, b + 1, c + 1)
    else play_game n (a - 3, b + 1, c + 1)
  else if b > a ∧ b > c then
    if b > 10 then play_game n (a + 1, b - 4, c + 1)
    else play_game n (a + 1, b - 3, c + 1)
  else 
    if c > 10 then play_game n (a + 1, b + 1, c - 4)
    else play_game n (a + 1, b + 1, c - 3)

/-- Defining the proof for the number of completed rounds until the game ends. -/
theorem game_ends_after_six_rounds :
  ∃ (n : ℕ), play_game n initial_tokens = (0, _, _) ∨ play_game n initial_tokens = (_, 0, _) ∨ play_game n initial_tokens = (_, _, 0) :=
sorry

end game_ends_after_six_rounds_l78_78464


namespace maximum_snowmen_count_l78_78489

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78489


namespace perimeter_of_regular_polygon_l78_78537

def regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 360 / n ∧ n * side_length > 0

theorem perimeter_of_regular_polygon :
  ∀ (n : ℕ) (side_length : ℝ), regular_polygon n side_length 45 → side_length = 7 → 8 = n → n * side_length = 56 :=
by
  intros n side_length h1 h2 h3
  rw [h2, h3]
  sorry

end perimeter_of_regular_polygon_l78_78537


namespace log_expression_evaluation_l78_78094

theorem log_expression_evaluation :
  (1 / 4)^(-2) + (1 / 2) * log 3 6 - log 3 (sqrt 2) = 33 / 2 := 
by {
  sorry
}

end log_expression_evaluation_l78_78094


namespace last_triangle_perimeter_l78_78734

theorem last_triangle_perimeter :
  ∃ (n : ℕ), ∀ (sides : ℕ) (T : Triangle),
    (T.side_lengths = [1015, 1016, 1017]) →
    (∀ n ≥ 1, ∃ (D E F : Point), 
      TangentPointIncircle T (D E F)) →
    (∀ n, ∃ (T_next : Triangle), 
      T_next.side_lengths = [AD, BE, CF] where (D E F) are tangent points) →
    T.perimeter = 762 / 128 :=
begin
  sorry
end

end last_triangle_perimeter_l78_78734


namespace dmitry_black_socks_l78_78858

theorem dmitry_black_socks :
  let blue_socks := 10
  let initial_black_socks := 22
  let white_socks := 12
  let total_initial_socks := blue_socks + initial_black_socks + white_socks
  ∀ x : ℕ,
    let total_socks := total_initial_socks + x
    let black_socks := initial_black_socks + x
    (black_socks : ℚ) / (total_socks : ℚ) = 2 / 3 → x = 22 :=
by
  sorry

end dmitry_black_socks_l78_78858


namespace dmitriev_older_by_10_l78_78015

-- Define the ages of each of the elders
variables (A B C D E F : ℕ)

-- The conditions provided in the problem
axiom hAlyosha : A > (A - 1)
axiom hBorya : B > (B - 2)
axiom hVasya : C > (C - 3)
axiom hGrisha : D > (D - 4)

-- Establishing an equation for the age differences leading to the proof
axiom age_sum_relation : A + B + C + D + E = (A - 1) + (B - 2) + (C - 3) + (D - 4) + F

-- We state that Dmitriev is older than Dima by 10 years
theorem dmitriev_older_by_10 : F = E + 10 :=
by
  -- sorry replaces the proof
  sorry

end dmitriev_older_by_10_l78_78015


namespace trapezoid_ratio_l78_78724

structure Trapezoid (α : Type) [LinearOrderedField α] :=
  (AB CD : α)
  (areas : List α)
  (AB_gt_CD : AB > CD)
  (areas_eq : areas = [3, 5, 6, 8])

open Trapezoid

theorem trapezoid_ratio (α : Type) [LinearOrderedField α] (T : Trapezoid α) :
  ∃ ρ : α, T.AB / T.CD = ρ ∧ ρ = 8 / 3 :=
by
  sorry

end trapezoid_ratio_l78_78724


namespace triangle_inequality_l78_78324

noncomputable def midpoint (A B : ℝ×ℝ) : ℝ×ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem triangle_inequality
  (A B C D M : ℝ×ℝ)
  (hM : M = midpoint B C)
  (hAD_AB : dist A D = dist A B)
  (hAD_perp_AB : inner (A - D) (A - B) = 0)
  (hC_D_side : ¬(same_side_of_line A B C D)) :
  sqrt (dist A B * dist A C + dist B C * dist A M) ≥ (sqrt 2 / 2) * dist C D :=
by
  sorry

end triangle_inequality_l78_78324


namespace lights_on_bottom_layer_l78_78696

theorem lights_on_bottom_layer
  (a₁ : ℕ)
  (q : ℕ := 3)
  (S₅ : ℕ := 242)
  (n : ℕ := 5)
  (sum_formula : S₅ = (a₁ * (q^n - 1)) / (q - 1)) :
  (a₁ * q^(n-1) = 162) :=
by
  sorry

end lights_on_bottom_layer_l78_78696


namespace value_of_f_sec_squared_l78_78598

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 ∧ x ≠ 1 then 1 / (x / (x - 1)) else 0

theorem value_of_f_sec_squared (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π / 2) : f (Real.sec t ^ 2) = Real.sin t ^ 2 := 
by
  sorry

end value_of_f_sec_squared_l78_78598


namespace Erik_money_left_l78_78191

theorem Erik_money_left 
  (init_money : ℝ)
  (loaf_of_bread : ℝ) (n_loaves_of_bread : ℝ)
  (carton_of_orange_juice : ℝ) (n_cartons_of_orange_juice : ℝ)
  (dozen_eggs : ℝ) (n_dozens_of_eggs : ℝ)
  (chocolate_bar : ℝ) (n_chocolate_bars : ℝ)
  (pound_apples : ℝ) (n_pounds_apples : ℝ)
  (pound_grapes : ℝ) (n_pounds_grapes : ℝ)
  (discount_bread_and_eggs : ℝ) (discount_other_items : ℝ)
  (sales_tax : ℝ) :
  n_loaves_of_bread = 3 →
  loaf_of_bread = 3 →
  n_cartons_of_orange_juice = 3 →
  carton_of_orange_juice = 6 →
  n_dozens_of_eggs = 2 →
  dozen_eggs = 4 →
  n_chocolate_bars = 5 →
  chocolate_bar = 2 →
  n_pounds_apples = 4 →
  pound_apples = 1.25 →
  n_pounds_grapes = 1.5 →
  pound_grapes = 2.5 →
  discount_bread_and_eggs = 0.1 →
  discount_other_items = 0.05 →
  sales_tax = 0.06 →
  init_money = 86 →
  (init_money - 
     (n_loaves_of_bread * loaf_of_bread * (1 - discount_bread_and_eggs) + 
      n_cartons_of_orange_juice * carton_of_orange_juice * (1 - discount_other_items) + 
      n_dozens_of_eggs * dozen_eggs * (1 - discount_bread_and_eggs) + 
      n_chocolate_bars * chocolate_bar * (1 - discount_other_items) + 
      n_pounds_apples * pound_apples * (1 - discount_other_items) + 
      n_pounds_grapes * pound_grapes * (1 - discount_other_items)) * (1 + sales_tax)) = 32.78 :=
by
  sorry

end Erik_money_left_l78_78191


namespace tangent_intersect_x_axis_l78_78457

-- Defining the conditions based on the given problem
def radius1 : ℝ := 3
def center1 : ℝ × ℝ := (0, 0)

def radius2 : ℝ := 5
def center2 : ℝ × ℝ := (12, 0)

-- Stating what needs to be proved
theorem tangent_intersect_x_axis : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (∀ (x1 x2 : ℝ), 
    (x1 = x) ∧ 
    (x2 = 12 - x) ∧ 
    (radius1 / (center2.1 - x) = radius2 / x2) → 
    (x = 9 / 2)) := 
sorry

end tangent_intersect_x_axis_l78_78457


namespace age_difference_l78_78009

variable (A B C D : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 16) : (A + B) - (B + C) = 16 :=
by
  sorry

end age_difference_l78_78009


namespace decrease_in_profit_due_to_looms_breakdown_l78_78148

theorem decrease_in_profit_due_to_looms_breakdown :
  let num_looms := 70
  let month_days := 30
  let total_sales := 1000000
  let total_expenses := 150000
  let daily_sales_per_loom := total_sales / (num_looms * month_days)
  let daily_expenses_per_loom := total_expenses / (num_looms * month_days)
  let loom1_days := 10
  let loom2_days := 5
  let loom3_days := 15
  let loom_repair_cost := 2000
  let loom1_loss := daily_sales_per_loom * loom1_days
  let loom2_loss := daily_sales_per_loom * loom2_days
  let loom3_loss := daily_sales_per_loom * loom3_days
  let total_loss_sales := loom1_loss + loom2_loss + loom3_loss
  let total_repair_cost := loom_repair_cost * 3
  let decrease_in_profit := total_loss_sales + total_repair_cost
  decrease_in_profit = 20285.70 := by
  sorry

end decrease_in_profit_due_to_looms_breakdown_l78_78148


namespace system_of_inequalities_l78_78766

theorem system_of_inequalities :
  ∃ (a b : ℤ), 
  (11 > 2 * a - b) ∧ 
  (25 > 2 * b - a) ∧ 
  (42 < 3 * b - a) ∧ 
  (46 < 2 * a + b) ∧ 
  (a = 14) ∧ 
  (b = 19) := 
sorry

end system_of_inequalities_l78_78766


namespace largest_sum_is_5_over_6_l78_78917

def sum_1 := (1/3) + (1/7)
def sum_2 := (1/3) + (1/8)
def sum_3 := (1/3) + (1/2)
def sum_4 := (1/3) + (1/9)
def sum_5 := (1/3) + (1/4)

theorem largest_sum_is_5_over_6 : (sum_3 = 5/6) ∧ ((sum_3 > sum_1) ∧ (sum_3 > sum_2) ∧ (sum_3 > sum_4) ∧ (sum_3 > sum_5)) :=
by
  sorry

end largest_sum_is_5_over_6_l78_78917


namespace problem_1_problem_2_problem_3_l78_78230

section sequence_problems

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ)
variable (a₀ : ℝ)
variable (h₀ : a₀ = 1)
variable (hrec : ∀ n : ℕ, a n.succ - 2 * a n = 2^n)

-- Problem (1): Prove the sequence {aₙ / 2ⁿ} is arithmetic
theorem problem_1 : ∃ d : ℝ, ∀ n : ℕ, (a n) / (2^n) = a₀ / (2^0) + n * d :=
begin
  sorry
end

-- Problem (2): Find the general formula for {aₙ}
theorem problem_2 : ∀ n : ℕ, a n = n * 2^(n-1) :=
begin
  sorry
end

-- Problem (3): Let bₙ = (n+2)*2^(n-1) / (aₙ * aₙ₊₁), prove that Sₙ < 1
def bₙ (n : ℕ) := (n + 2) * 2^(n-1) / ((a n) * (a n.succ))
def Sₙ (n : ℕ) := ∑ i in finset.range n, bₙ a i

theorem problem_3 {n : ℕ} : Sₙ a n < 1 :=
begin
  sorry
end

end sequence_problems

end problem_1_problem_2_problem_3_l78_78230


namespace behavior_of_g_l78_78573

noncomputable def g (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x - 9

theorem behavior_of_g :
  (∀ x, x > 0 → g x → ∞) ∧ (∀ x, x < 0 → g x → ∞) :=
by 
  sorry

end behavior_of_g_l78_78573


namespace inequality_solution_set_nonempty_range_l78_78999

theorem inequality_solution_set_nonempty_range (a : ℝ) :
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ (a ≤ -2 ∨ a ≥ 6 / 5) :=
by
  -- Proof is omitted
  sorry

end inequality_solution_set_nonempty_range_l78_78999


namespace ivan_income_tax_l78_78707

theorem ivan_income_tax :
  let salary_probation := 20000
  let probation_months := 2
  let salary_after_probation := 25000
  let after_probation_months := 8
  let bonus := 10000
  let tax_rate := 0.13
  let total_income := salary_probation * probation_months +
                      salary_after_probation * after_probation_months + bonus
  total_income * tax_rate = 32500 := sorry

end ivan_income_tax_l78_78707


namespace football_team_total_players_l78_78411

theorem football_team_total_players (P : ℕ) (throwers : ℕ) (left_handed : ℕ) (right_handed : ℕ) :
  throwers = 49 →
  right_handed = 63 →
  left_handed = (1/3) * (P - 49) →
  (P - 49) - left_handed = (2/3) * (P - 49) →
  70 = P :=
by
  intros h_throwers h_right_handed h_left_handed h_remaining
  sorry

end football_team_total_players_l78_78411


namespace eriks_dog_speed_l78_78936

theorem eriks_dog_speed (rabbit_speed : ℝ) (head_start : ℝ) (time_minutes : ℝ) (time_hours : ℝ) 
    (distance_rabbit : ℝ) (total_distance_rabbit : ℝ) (distance_dog : ℝ) :
    rabbit_speed = 15 →
    head_start = 0.6 →
    time_minutes = 4 →
    time_hours = time_minutes / 60 →
    distance_rabbit = rabbit_speed * time_hours →
    total_distance_rabbit = distance_rabbit + head_start →
    distance_dog = total_distance_rabbit →
    (distance_dog / time_hours) = 24 :=
by
  intros
  rw [←h, ←h_1, ←h_2, ←h_3, ←h_4, ←h_5, ←h_6]
  sorry

end eriks_dog_speed_l78_78936


namespace parabola_equation_area_triangle_minimum_line_PE_through_fixed_point_l78_78640

-- Given conditions: Parabola, points, and mutual distances
theorem parabola_equation (p : ℝ) (hp : p > 0) (x_P : ℝ) (hxP : x_P = 3) 
  (C : ℝ → ℝ → Prop) (hC : ∀ x y, C x y ↔ y^2 = 2 * p * x) :
  ∃ p, y^2 = 4 * x :=
sorry

theorem area_triangle_minimum (p : ℝ) (hp : p > 0) (x_P y_P : ℝ) (hxP : x_P = 3) (y0 : ℝ)
  (C : ℝ → ℝ → Prop) (hC : ∀ x y, C x y ↔ y^2 = 2 * p * x)
  (S : ℝ × ℝ) (hS : |S.1 - (p / 2)| = x_P + p / 2) (P : ℝ × ℝ) (FP : ℝ) (FS : ℝ) :
  ∃ p, p = 2 ∧ P = (3, y_P) ∧ |FP| = |FS| ∧ y0^2 = 4 ∧ (|y_P| + 4 / |y_P| > 2) :=
sorry

theorem line_PE_through_fixed_point (p : ℝ) (hp : p > 0) (x_P y_P : ℝ) (hxP : x_P = 3) (y0 : ℝ)
  (C : ℝ → ℝ → Prop) (hC : ∀ x y, C x y ↔ y^2 = 2 * p * x)
  (S : ℝ × ℝ) (hS : |S.1 - (p / 2)| = x_P + p / 2) (P : ℝ × ℝ) (PE : ℝ × ℝ)
  (hPE : y_P^2 ∈ ℝ) :
  (P = (3, y_P)) ∧ ∃ (fixed_point : ℝ × ℝ), fixed_point = (1,0) :=
sorry

end parabola_equation_area_triangle_minimum_line_PE_through_fixed_point_l78_78640


namespace range_k_l78_78262

variable (x ω : ℝ)
variable (k : ℝ)
variable (f h : ℝ → ℝ)
variable (a b : ℝ × ℝ)

noncomputable def vector_a (ω x : ℝ) : ℝ × ℝ := (Real.cos (ω * x)^2 - Real.sin (ω * x)^2, Real.sin (ω * x))
noncomputable def vector_b (ω : ℝ) : ℝ × ℝ := (Real.sqrt 3, 2 * Real.cos (ω * x))

noncomputable def f (x : ℝ) (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
noncomputable def h (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 3))

theorem range_k :
  let ω := (0 : ℝ) in
  (∀ x, f x (vector_a ω x) (vector_b ω x) = 2 * Real.sin ((1 / 3) * x + (Real.pi / 3)))
  → (∀ x, h (x / 6 + (Real.pi / 3)) = 2 * Real.sin (2 * x - (Real.pi / 3)))
  → (∀ x, (h x + k = 0) ↔ x ∈ Icc 0 (Real.pi / 2))
  → k ∈ Icc (-Real.sqrt 3) (Real.sqrt 3) ∨ k = -2 :=
begin
  sorry
end

end range_k_l78_78262


namespace maximum_snowmen_count_l78_78491

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78491


namespace total_selling_price_l78_78458

def original_price : ℝ := 120
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.15

def sale_price (original_price discount_percent : ℝ) : ℝ :=
  original_price * (1 - discount_percent)

def final_price (sale_price tax_percent : ℝ) : ℝ :=
  sale_price * (1 + tax_percent)

theorem total_selling_price :
  final_price (sale_price original_price discount_percent) tax_percent = 96.6 :=
sorry

end total_selling_price_l78_78458


namespace isosceles_triangle_base_range_perimeter_20_l78_78796

theorem isosceles_triangle_base_range_perimeter_20 (x : ℝ) :
  (5 < x) ∧ (x < 10) ↔
  let side := (20 - x) / 2 
  in (side > 0) ∧ (2 * side > x) ∧ (x > 0) := 
by 
  sorry

end isosceles_triangle_base_range_perimeter_20_l78_78796


namespace num_sides_polygon_l78_78007

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l78_78007


namespace trapezium_angle_equality_l78_78686

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Module ℝ α]

noncomputable def midpoint (A B : α) : α := (A + B) / 2

theorem trapezium_angle_equality
  {A B C D P Q : α}
  (htrapezium: ∃ M: α, A + B = C + D) -- Trapezium condition
  (hP: midpoint A C = P) -- P is the midpoint of diagonal AC
  (hQ: midpoint B D = Q) -- Q is the midpoint of diagonal BD
  (hangle: ∠DAQ = ∠CAB) -- Given angle condition
  : ∠PBA = ∠DBC := 
sorry

end trapezium_angle_equality_l78_78686


namespace cos_pi_div_12_value_l78_78210

noncomputable def cos_pi_div_12 : ℝ :=
  cos (π / 12)

theorem cos_pi_div_12_value : cos_pi_div_12 = (√6 + √2) / 4 :=
by
  sorry

end cos_pi_div_12_value_l78_78210


namespace craftsman_jars_l78_78460

theorem craftsman_jars (J P : ℕ) 
  (h1 : J = 2 * P)
  (h2 : 5 * J + 15 * P = 200) : 
  J = 16 := by
  sorry

end craftsman_jars_l78_78460


namespace slope_inequality_l78_78252

-- Definitions of the given functions and points
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + ln x + 2

def g (a : ℝ) (x : ℝ) : ℝ := f a x - a * x^2

def slope (k : ℝ) (x1 x2 : ℝ) : Prop :=
  k = (ln x2 - ln x1) / (x2 - x1)

-- The mathematical proof problem
theorem slope_inequality {a k x1 x2 : ℝ} (h1 : 0 < x1) (h2 : x1 < x2) 
  (h3 : slope k x1 x2) : x1 < 1/k ∧ 1/k < x2 :=
sorry

end slope_inequality_l78_78252


namespace shopping_festival_profit_margin_l78_78453

theorem shopping_festival_profit_margin (x : ℝ) (hx : 0 < x) :
  let y := 1.47 * x,
      original_price := y / 0.7,
      festival_price := original_price,
      cost_for_two := 2 * x
  in (festival_price - cost_for_two) / cost_for_two = 0.05 :=
by
  let y := 1.47 * x
  let original_price := y / 0.7
  let festival_price := original_price
  let cost_for_two := 2 * x
  have h_profit_margin : (festival_price - cost_for_two) / cost_for_two = 0.05
  . sorry
  exact h_profit_margin

end shopping_festival_profit_margin_l78_78453


namespace nicky_cristina_catch_up_time_l78_78350

theorem nicky_cristina_catch_up_time :
  ∃ t : ℕ, (let head_start := 36, 
                cristina_pace := 6, 
                nicky_pace := 3 in
            cristina_pace * t = head_start + nicky_pace * t) → 
  t = 12 :=
by
  sorry

end nicky_cristina_catch_up_time_l78_78350


namespace at_least_two_black_balls_probability_l78_78102

theorem at_least_two_black_balls_probability :
  let total_balls := 19
  let black_balls := 10
  let white_balls := 9
  let drawn_balls := 3
  let favorable_outcomes := (Nat.choose black_balls 2 * Nat.choose white_balls 1) + Nat.choose black_balls 3
  let total_outcomes := Nat.choose total_balls drawn_balls
  in (favorable_outcomes / total_outcomes : ℚ) = 175 / 323 := sorry

end at_least_two_black_balls_probability_l78_78102


namespace curve_touches_x_axis_at_most_three_times_l78_78959

theorem curve_touches_x_axis_at_most_three_times
  (a b c d : ℝ) :
  ∃ (x : ℝ), (x^4 - x^5 + a * x^3 + b * x^2 + c * x + d = 0) → ∃ (y : ℝ), (y = 0) → 
  ∃(n : ℕ), (n ≤ 3) :=
by sorry

end curve_touches_x_axis_at_most_three_times_l78_78959


namespace claudia_filled_5oz_glasses_l78_78921

theorem claudia_filled_5oz_glasses :
  ∃ (n : ℕ), n = 6 ∧ 4 * 8 + 15 * 4 + n * 5 = 122 :=
by
  sorry

end claudia_filled_5oz_glasses_l78_78921


namespace negation_equivalence_l78_78396

variable (U : Type) (S R : U → Prop)

-- Original statement: All students of this university are non-residents, i.e., ∀ x, S(x) → ¬ R(x)
def original_statement : Prop := ∀ x, S x → ¬ R x

-- Negation of the original statement: ∃ x, S(x) ∧ R(x)
def negated_statement : Prop := ∃ x, S x ∧ R x

-- Lean statement to prove that the negation of the original statement is equivalent to some students are residents
theorem negation_equivalence : ¬ original_statement U S R = negated_statement U S R :=
sorry

end negation_equivalence_l78_78396


namespace smallest_n_candy_price_l78_78684

theorem smallest_n_candy_price :
  ∃ n : ℕ, 25 * n = Nat.lcm (Nat.lcm 20 18) 24 ∧ ∀ k : ℕ, k > 0 ∧ 25 * k = Nat.lcm (Nat.lcm 20 18) 24 → n ≤ k :=
sorry

end smallest_n_candy_price_l78_78684


namespace rooks_attack_after_knight_move_l78_78362

theorem rooks_attack_after_knight_move (r : ℕ) (c : ℕ) (rooks : Fin 15 → Fin 15 × Fin 15) :
  (∀ i j, i ≠ j → rooks i ≠ rooks j ∧ rooks i.1 ≠ rooks j.1 ∧ rooks i.2 ≠ rooks j.2) →
  (∃ i j, i ≠ j ∧ ((rooks' i = rooks' j) ∨ rooks' i.1 = rooks' j.1 ∨ rooks' i.2 = rooks' j.2))
  :=
begin
  sorry -- proof goes here
end

end rooks_attack_after_knight_move_l78_78362


namespace intersection_with_xz_plane_l78_78592

-- Initial points on the line
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def point1 : Point3D := ⟨2, -1, 3⟩
def point2 : Point3D := ⟨6, -4, 7⟩

-- Definition of the line parametrization
def param_line (t : ℝ) : Point3D :=
  ⟨ point1.x + t * (point2.x - point1.x)
  , point1.y + t * (point2.y - point1.y)
  , point1.z + t * (point2.z - point1.z) ⟩

-- Prove that the line intersects the xz-plane at the expected point
theorem intersection_with_xz_plane :
  ∃ t : ℝ, param_line t = ⟨ 2/3, 0, 5/3 ⟩ :=
sorry

end intersection_with_xz_plane_l78_78592


namespace remainder_at_minus_2_l78_78141

noncomputable def polynomial_remainder (p : Polynomial ℝ) : Polynomial ℝ :=
  p % ((X - 3) * (X + 1) * (X - 4))

theorem remainder_at_minus_2 (p : Polynomial ℝ) (h1 : p.eval 3 = 5) (h2 : p.eval (-1) = -2) (h3 : p.eval 4 = 7) :
  (polynomial_remainder p).eval (-2) = -7.2 := 
sorry

end remainder_at_minus_2_l78_78141


namespace supermarket_spent_more_than_collected_l78_78016

-- Given conditions
def initial_amount : ℕ := 53
def collected_amount : ℕ := 91
def amount_left : ℕ := 14

-- Finding the total amount before shopping and amount spent in supermarket
def total_amount : ℕ := initial_amount + collected_amount
def spent_amount : ℕ := total_amount - amount_left

-- Prove that the difference between spent amount and collected amount is 39
theorem supermarket_spent_more_than_collected : (spent_amount - collected_amount) = 39 := by
  -- The proof will go here
  sorry

end supermarket_spent_more_than_collected_l78_78016


namespace orange_juice_production_correct_l78_78407

noncomputable def orangeJuiceProduction (total_oranges : Float) (export_percent : Float) (juice_percent : Float) : Float :=
  let remaining_oranges := total_oranges * (1 - export_percent / 100)
  let juice_oranges := remaining_oranges * (juice_percent / 100)
  Float.round (juice_oranges * 10) / 10

theorem orange_juice_production_correct :
  orangeJuiceProduction 8.2 30 40 = 2.3 := by
  sorry

end orange_juice_production_correct_l78_78407


namespace AK_equals_BM_l78_78054

open EuclideanGeometry

variables {C D A B K M O P : Point}

noncomputable def problem_statement (circle : Circle) (hAB : Diameter A B circle) (hCD : Chord C D circle)
  (hK : PerpIntersect C (line_through A B) K) (hM : PerpIntersect D (line_through A B) M) : Prop :=
  distance A K = distance B M

-- Not providing the proof here, just the statement
theorem AK_equals_BM (circle : Circle) (hAB : Diameter A B circle) (hCD : Chord C D circle)
  (hK : PerpIntersect C (line_through A B) K) (hM : PerpIntersect D (line_through A B) M) : 
  problem_statement circle hAB hCD hK hM :=
sorry

end AK_equals_BM_l78_78054


namespace diane_harvest_increase_l78_78581

-- Define the conditions
def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

-- Definition of the increase in honey harvest
def increase_in_harvest : ℕ := this_year_harvest - last_year_harvest

-- The theorem statement we need to prove
theorem diane_harvest_increase : increase_in_harvest = 6085 := 
by
  -- skip the proof for now
  sorry

end diane_harvest_increase_l78_78581


namespace fractional_equation_solution_l78_78800

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ -3) (h : 1/x = 2/(x+3)) : x = 3 :=
sorry

end fractional_equation_solution_l78_78800


namespace min_handshakes_l78_78448

theorem min_handshakes (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  ∃ (m : ℕ), m = 45 ∧ ∀ (graph : n → list ℕ), (∀ v : ℕ, v < n → length (graph v) = k)
  → (2 * (card { e : ℕ × ℕ // e.fst < n ∧ e.snd < n ∧ e.fst ≠ e.snd }) ≤ n * k) :=
sorry

end min_handshakes_l78_78448


namespace max_length_cos_theta_l78_78571

def domain (x y : ℝ) : Prop := (x^2 + (y - 1)^2 ≤ 1 ∧ x ≥ (Real.sqrt 2 / 3))

theorem max_length_cos_theta :
  (∃ x y : ℝ, domain x y ∧ ∀ θ : ℝ, (0 < θ ∧ θ < (Real.pi / 2)) → θ = Real.arctan (Real.sqrt 2) → 
  (Real.cos θ = Real.sqrt 3 / 3)) := sorry

end max_length_cos_theta_l78_78571


namespace solve_fractional_equation_l78_78808

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l78_78808


namespace shortest_path_correct_l78_78778

noncomputable def shortest_path_length (length width height : ℕ) : ℝ :=
  let diagonal := Real.sqrt ((length + height)^2 + width^2)
  Real.sqrt 145

theorem shortest_path_correct :
  ∀ (length width height : ℕ),
    length = 4 → width = 5 → height = 4 →
    shortest_path_length length width height = Real.sqrt 145 :=
by
  intros length width height h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shortest_path_correct_l78_78778


namespace total_movies_seen_l78_78178

theorem total_movies_seen (d h a c : ℕ) (hd : d = 7) (hh : h = 12) (ha : a = 15) (hc : c = 2) :
  (c + (d - c) + (h - c) + (a - c)) = 30 :=
by
  sorry

end total_movies_seen_l78_78178


namespace tangent_cotangent_inequality_l78_78226

variable (n : ℕ) (θ : Fin n → ℝ) 
  (h : ∀ i, 0 < θ i ∧ θ i < Real.pi / 2)

theorem tangent_cotangent_inequality :
  (∑ i, Real.tan (θ i) * ∑ i, Real.cot (θ i)) ≥
  (∑ i, Real.sin (θ i)) ^ 2 + (∑ i, Real.cos (θ i)) ^ 2 := by
  sorry

end tangent_cotangent_inequality_l78_78226


namespace circumscribed_inscribed_coincide_l78_78295

theorem circumscribed_inscribed_coincide
  {A B C D : Point} (h1 : dist A B = dist C D)
                      (h2 : dist B C = dist A D)
                      (h3 : dist A C = dist B D) :
  ∃ O : Point, is_circumsphere_center O A B C D ∧ is_insphere_center O A B C D :=
sorry

end circumscribed_inscribed_coincide_l78_78295


namespace arrangement_correctness_l78_78012

def num_arrangements (total_people : ℕ) 
  (A B C D : ℕ)
  (arrangements with_no_adj_ABC no_adj_AD : total_people → Prop) 
: ℕ := sorry

theorem arrangement_correctness : 
  num_arrangements 7 1 2 3 4 
  (arrangements 7) 
  (λ s, ¬(s 1 = s 2 ∧ s 2 = s 3)) 
  (λ s, ¬(s 1 = s 4)) = 720 
:=
sorry

end arrangement_correctness_l78_78012


namespace number_of_true_statements_is_two_l78_78958

def line_plane_geometry : Type :=
  -- Types representing lines and planes
  sorry

def l : line_plane_geometry := sorry
def alpha : line_plane_geometry := sorry
def m : line_plane_geometry := sorry
def beta : line_plane_geometry := sorry

def is_perpendicular (x y : line_plane_geometry) : Prop := sorry
def is_parallel (x y : line_plane_geometry) : Prop := sorry
def is_contained_in (x y : line_plane_geometry) : Prop := sorry

axiom l_perpendicular_alpha : is_perpendicular l alpha
axiom m_contained_in_beta : is_contained_in m beta

def statement_1 : Prop := is_parallel alpha beta → is_perpendicular l m
def statement_2 : Prop := is_perpendicular alpha beta → is_parallel l m
def statement_3 : Prop := is_parallel l m → is_perpendicular alpha beta

theorem number_of_true_statements_is_two : 
  (statement_1 ↔ true) ∧ (statement_2 ↔ false) ∧ (statement_3 ↔ true) := 
sorry

end number_of_true_statements_is_two_l78_78958


namespace range_of_a_l78_78258

def p (x : ℝ) : Prop := 1 / 2 ≤ x ∧ x ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (a x : ℝ) 
  (hp : ∀ x, ¬ (1 / 2 ≤ x ∧ x ≤ 1) → (x < 1 / 2 ∨ x > 1))
  (hq : ∀ x, ¬ ((x - a) * (x - a - 1) ≤ 0) → (x < a ∨ x > a + 1))
  (h : ∀ x, (q x a) → (p x)) :
  0 ≤ a ∧ a ≤ 1 / 2 := 
sorry

end range_of_a_l78_78258


namespace determine_n_l78_78025

-- Constants and variables
variables {a : ℕ → ℝ} {n : ℕ}

-- Definition for the condition at each vertex
def vertex_condition (a : ℕ → ℝ) (i : ℕ) : Prop :=
  a i = a (i - 1) * a (i + 1)

-- Mathematical problem statement
theorem determine_n (h : ∀ i, vertex_condition a i) (distinct_a : ∀ i j, a i ≠ a j) : n = 6 :=
sorry

end determine_n_l78_78025


namespace minimize_base_side_length_l78_78470

theorem minimize_base_side_length (V : ℝ) (a h : ℝ) 
  (volume_eq : V = a ^ 2 * h) (V_given : V = 256) (h_eq : h = 256 / (a ^ 2)) :
  a = 8 :=
by
  -- Recognize that for a given volume, making it a cube minimizes the surface area.
  -- As the volume of the cube a^3 = 256, solving for a gives 8.
  -- a := (256:ℝ) ^ (1/3:ℝ)
  sorry

end minimize_base_side_length_l78_78470


namespace perpendiculars_form_square_l78_78882

theorem perpendiculars_form_square
  (A B C D A1 B1 C1 D1 : Type)
  (ABCD_square : IsSquare A B C D)
  (A1B1C1D1_parallelogram : IsParallelogram A1 B1 C1 D1)
  (A_on_A1B1 B_on_B1C1 C_on_C1D1 D_on_D1A1 : Moreover)
  (l1_perpendicular : Perpendicular A1 (AB : Line))
  (l2_perpendicular : Perpendicular B1 (BC : Line))
  (l3_perpendicular : Perpendicular C1 (CD : Line))
  (l4_perpendicular : Perpendicular D1 (DA : Line)) :
  IsSquare l1 l2 l3 l4 := 
sorry

end perpendiculars_form_square_l78_78882


namespace led_information_l78_78815

-- Define the main problem with conditions
def Problem : Prop :=
  ∀ (LEDs : Fin 7 → Bool) (colors : Fin 7 → Bool),
    (∑ i : Fin 7, if LEDs i then 1 else 0 = 3) ∧
    (∀ i : Fin 6, LEDs i → ¬ LEDs (i + 1)) →
    (∃ positions : Finset (Fin 7), (positions.card = 17 ∧ ∀ p ∈ positions, ¬ (positions = positions ∪ (Finset.mk {i | colors i}))) →
    positions.card * 2^3 = 136)

-- Main theorem to prove
theorem led_information : Problem := sorry

end led_information_l78_78815


namespace division_problem_l78_78044

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l78_78044


namespace max_min_lcm_l78_78327

theorem max_min_lcm (n : ℕ) (hn : n > 1) (S : set (finset ℕ)) (hS : S = { A | A.card = n ∧ A ⊆ finset.range (2*n + 1) }):
  (∀ A ∈ S, true) → -- dummy condition to use S in the context
  (max (λ A -> min (λ xy -> let ⟨x, y⟩ := xy in if x ≠ y then nat.lcm x y else ⊤) (A.val.pair_finset A.val)) (S.to_finset) = 
  if n % 2 = 1 then 3*(n+1) else if n = 2 then 12 else if n = 4 then 24 else if n ≥ 6 then 3*(n+2) else 0) :=
begin
  intro h,
  sorry,
end

end max_min_lcm_l78_78327


namespace max_min_values_of_x_l78_78740

theorem max_min_values_of_x (x y z : ℝ) (h1 : x + y + z = 0) (h2 : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  -2/3 ≤ x ∧ x ≤ 2/3 :=
sorry

end max_min_values_of_x_l78_78740


namespace no_four_points_with_equal_tangents_l78_78309

theorem no_four_points_with_equal_tangents :
  ∀ (A B C D : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    A ≠ C ∧ B ≠ D →
    ¬ (∀ (P Q : ℝ × ℝ), (P = A ∧ Q = B) ∨ (P = C ∧ Q = D) →
      ∃ (M : ℝ × ℝ) (r : ℝ), M ≠ P ∧ M ≠ Q ∧
      (dist A M = dist C M ∧ dist B M = dist D M ∧
       dist P M > r ∧ dist Q M > r)) :=
by sorry

end no_four_points_with_equal_tangents_l78_78309


namespace bela_wins_if_and_only_if_n_even_l78_78902

theorem bela_wins_if_and_only_if_n_even (n : ℕ) (hn : n > 10) :
  (∃ b : Ω, b = 1) ↔ n % 2 = 0 := 
sorry

end bela_wins_if_and_only_if_n_even_l78_78902


namespace sum_of_squares_l78_78190

noncomputable def PQR_side_length := Real.sqrt 44
noncomputable def QT_length := Real.sqrt 33

theorem sum_of_squares (P Q R T₁ U₁ T₂ U₂ : Point)
  (h₁ : equilateral_triangle P Q R)
  (h₂ : congruent_triangle P T₁ U₁ P Q R)
  (h₃ : congruent_triangle P T₂ U₂ P Q R)
  (hQT₁ : dist Q T₁ = QT_length)
  (hQT₂ : dist Q T₂ = QT_length)
  (hPQ : dist P Q = PQR_side_length)
  (hQR : dist Q R = PQR_side_length)
  (hRP : dist R P = PQR_side_length) :
  ∑ k in Finset.range 2, (λ k, dist P (if k = 0 then U₁ else U₂)^2) k = 176 :=
sorry

end sum_of_squares_l78_78190


namespace max_snowmen_l78_78508

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78508


namespace proof_ellipse_fixed_point_l78_78971

theorem proof_ellipse_fixed_point 
  (a b : ℝ)
  (ha : a = 2)
  (hab : a^2 + b^2 = 7)
  (h_ab : a > b ∧ b > 0) :
  let E := (-sqrt(a^2 - b^2), 0)
  let F := (sqrt(a^2 - b^2), 0)
  let x := (-4, 0)
  let C : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / a^2) + (y^2 / b^2) = 1}
  let M := (mx, my) -- with the condition that M is on ellipse
  let M' := (-mx, -my) -- reflection of M about the x-axis
  ∃ N : ℝ × ℝ, (N ∈ C) ∧
    (let line_MN : set (ℝ × ℝ) := {p | ∃ t,  p = (mx + t * (nx - mx), my + t * (ny - my))} in
     ∀ p : ℝ × ℝ, p ∈ line_MN → p ∈ (line_TM N M')) ∧ line_TM x y passes through (4, 0) :=
  sorry

end proof_ellipse_fixed_point_l78_78971


namespace max_snowmen_l78_78503

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78503


namespace number_of_members_l78_78435

theorem number_of_members (n : ℕ) (h : n^2 = 9801) : n = 99 :=
sorry

end number_of_members_l78_78435


namespace common_ratio_of_gp_l78_78056

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem common_ratio_of_gp (a : ℝ) (r : ℝ) (h : geometric_sum a r 6 / geometric_sum a r 3 = 28) : r = 3 :=
by
  sorry

end common_ratio_of_gp_l78_78056


namespace real_root_range_l78_78669

theorem real_root_range (a : ℝ) : 
  (∃ x : ℝ, 2^(2*x) + 2^x * a + a + 1 = 0) ↔ 
  (a ∈ Set.Icc (-∞) (2 - 2 * Real.sqrt 2) ∨ 
   a ∈ Set.Icc (2 + 2 * Real.sqrt 2) ∞) :=
sorry

end real_root_range_l78_78669


namespace solve_for_a_l78_78238

-- Define the complex numbers z1 and z2
def z1 (a : ℝ) : ℂ := a + 2 * complex.I
def z2 : ℂ := 2 - complex.I

-- Condition that the moduli of z1 and z2 are equal
def moduli_equal (a : ℝ) : Prop := complex.abs (z1 a) = complex.abs z2

-- The value of the real number a
theorem solve_for_a (a : ℝ) (h : moduli_equal a) : a = 1 ∨ a = -1 :=
by sorry

end solve_for_a_l78_78238


namespace probability_meeting_proof_l78_78757

noncomputable def probability_meeting (arrival_time_paul arrival_time_caroline : ℝ) : Prop :=
  arrival_time_paul ≤ arrival_time_caroline + 1 / 4 ∧ arrival_time_paul ≥ arrival_time_caroline - 1 / 4

theorem probability_meeting_proof :
  ∀ (arrival_time_paul arrival_time_caroline : ℝ)
    (h_paul_range : 0 ≤ arrival_time_paul ∧ arrival_time_paul ≤ 1)
    (h_caroline_range: 0 ≤ arrival_time_caroline ∧ arrival_time_caroline ≤ 1),
  (probability_meeting arrival_time_paul arrival_time_caroline) → 
  ∃ p, p = 7/16 :=
by
  sorry

end probability_meeting_proof_l78_78757


namespace find_point_C_l78_78543

theorem find_point_C :
  ∃ C : ℝ × ℝ, let A : ℝ × ℝ := (-3, 5) in
                 let B : ℝ × ℝ := (9, -1) in
                 let AB := (B.1 - A.1, B.2 - A.2) in
                 C = (B.1 + 0.5 * AB.1, B.2 + 0.5 * AB.2) ∧ 
                 C = (15, -4) :=
by
  sorry

end find_point_C_l78_78543


namespace collinear_O1AP_l78_78415

open EuclideanGeometry

-- Definitions of all terms involved in the conditions
variables {O1 O2 A B P : Point}
variables {circle1 circle2 : Circle}
variables {circle3 : Circle}

-- Conditions
axiom circle1_center_O1 : circle1.center = O1
axiom circle2_center_O2 : circle2.center = O2
axiom intersection_AB : (A ∈ circle1) ∧ (A ∈ circle2) ∧ (B ∈ circle1) ∧ (B ∈ circle2)
axiom circle3_through_OB1B2 : (O1 ∈ circle3) ∧ (O2 ∈ circle3) ∧ (B ∈ circle3)
axiom P_on_circle2 : P ∈ circle2
axiom P_on_circle3 : P ∈ circle3

-- Question (Proof to show that O1, A, and P are collinear)
theorem collinear_O1AP : Collinear {O1, A, P} :=
by
  sorry

end collinear_O1AP_l78_78415


namespace new_class_mean_score_l78_78678

theorem new_class_mean_score : 
  let s1 := 68
  let n1 := 50
  let s2 := 75
  let n2 := 8
  let s3 := 82
  let n3 := 2
  (n1 * s1 + n2 * s2 + n3 * s3) / (n1 + n2 + n3) = 69.4 := by
  sorry

end new_class_mean_score_l78_78678


namespace stratified_sampling_second_grade_l78_78797

theorem stratified_sampling_second_grade (r1 r2 r3 : ℕ) (total_sample : ℕ) (total_ratio : ℕ):
  r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ total_sample = 50 ∧ total_ratio = r1 + r2 + r3 →
  (r2 * total_sample) / total_ratio = 15 :=
by
  sorry

end stratified_sampling_second_grade_l78_78797


namespace surface_area_tetrahedron_is_16pi_l78_78892

-- Definitions based on the given conditions
def edge1 : ℝ := 1
def edge2 : ℝ := Real.sqrt 6
def edge3 : ℝ := 3

-- Calculate the diagonal (diameter of the circumscribing sphere)
def diameter : ℝ := Real.sqrt (edge1^2 + edge2^2 + edge3^2)

-- Calculate the radius of the sphere
def radius : ℝ := diameter / 2

-- The expected surface area of the sphere
def expected_surface_area : ℝ := 4 * Real.pi * radius^2

theorem surface_area_tetrahedron_is_16pi :
  expected_surface_area = 16 * Real.pi :=
by
  sorry

end surface_area_tetrahedron_is_16pi_l78_78892


namespace phone_plan_cost_equal_at_2500_l78_78530

-- We define the costs C1 and C2 as described in the problem conditions.
def C1 (x : ℕ) : ℝ :=
  if x <= 500 then 50 else 50 + 0.35 * (x - 500)

def C2 (x : ℕ) : ℝ :=
  if x <= 1000 then 75 else 75 + 0.45 * (x - 1000)

-- We need to prove that the costs are equal when x = 2500.
theorem phone_plan_cost_equal_at_2500 : C1 2500 = C2 2500 := by
  sorry

end phone_plan_cost_equal_at_2500_l78_78530


namespace pairs_m_n_l78_78941

theorem pairs_m_n (m n : ℤ) : n ^ 2 - 3 * m * n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) :=
by sorry

end pairs_m_n_l78_78941


namespace joy_can_choose_17_rods_for_quadrilateral_l78_78720

theorem joy_can_choose_17_rods_for_quadrilateral :
  ∃ (possible_rods : Finset ℕ), 
    possible_rods.card = 17 ∧
    ∀ rod ∈ possible_rods, 
      rod > 0 ∧ rod <= 30 ∧
      (rod ≠ 3 ∧ rod ≠ 7 ∧ rod ≠ 15) ∧
      (rod > 15 - (3 + 7)) ∧
      (rod < 3 + 7 + 15) :=
by
  sorry

end joy_can_choose_17_rods_for_quadrilateral_l78_78720


namespace relationship_between_b_and_k_equation_of_line_range_of_area_of_triangle_AOB_l78_78725

open Real
-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the foci
def F1 := (-sqrt 2, 0 : ℝ × ℝ)
def F2 := (sqrt 2, 0 : ℝ × ℝ)

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2
def diameter := dist (prod.fst F1, prod.snd F1) (prod.fst F2, prod.snd F2)

-- Define line l: y = kx + b
def line_l (k b x y : ℝ) : Prop := y = k * x + b

-- Define vector projection
def projection (u v : ℝ × ℝ) : ℝ := (u.1 * v.1 + u.2 * v.2) / (v.1^2 + v.2^2)

-- Problem statements
theorem relationship_between_b_and_k (k b : ℝ) (h1 : ∀ x y : ℝ, line_l k b x y → circle_O x y) :
  b^2 = 2 * (k^2 + 1) := sorry

theorem equation_of_line
  (A B : ℝ × ℝ) (k b : ℝ)
  (h1 : hyperbola A.1 A.2) (h2 : hyperbola B.1 B.2)
  (h3 : line_l k b A.1 A.2) (h4 : line_l k b B.1 B.2)
  (h5 : A ≠ B) (h6 : projection (A.1 - B.1, A.2 - B.2) (F2.1 - F1.1, F2.2 - F1.2) = 1 / sqrt (k^2 + 1)) :
  (A.1 * B.1 + A.2 * B.2) * 1 / (1 + k^2) = 1 →
  (b = sqrt 6 ∨ b = -sqrt 6) ∧ (k = sqrt 2 ∨ k = -sqrt 2) := sorry

theorem range_of_area_of_triangle_AOB (m : ℝ) (h1 : 2 ≤ m) (h2 : m ≤ 4) :
  3 * sqrt 10 ≤ sqrt (16 * m^2 + 12 * m + 2) ∧ sqrt (16 * m^2 + 12 * m + 2) ≤ 3 * sqrt 34 := sorry

end relationship_between_b_and_k_equation_of_line_range_of_area_of_triangle_AOB_l78_78725


namespace find_number_l78_78036

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l78_78036


namespace dividend_rate_correct_l78_78106

-- Define the stock's yield and market value
def stock_yield : ℝ := 0.08
def market_value : ℝ := 175

-- Dividend rate definition based on given yield and market value
def dividend_rate (yield market_value : ℝ) : ℝ :=
  (yield * market_value)

-- The problem statement to be proven in Lean
theorem dividend_rate_correct :
  dividend_rate stock_yield market_value = 14 := by
  sorry

end dividend_rate_correct_l78_78106


namespace set_union_intersection_l78_78969

def A := {0, 1, 2, 4, 5, 7, 8}
def B := {1, 3, 6, 7, 9}
def C := {3, 4, 7, 8}

theorem set_union_intersection : ((A ∩ B) ∪ C) = {1, 3, 4, 7, 8} :=
by
  sorry

end set_union_intersection_l78_78969


namespace max_snowmen_l78_78522

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78522


namespace floor_sum_eq_self_l78_78326

theorem floor_sum_eq_self (n : ℕ) : 
  (∑ i in Finset.range n, ⌊(n + 2^i) / (2^(i+1))⌋) = n := 
sorry

end floor_sum_eq_self_l78_78326


namespace find_number_l78_78043

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l78_78043


namespace proof_problem_l78_78736

noncomputable def w : ℂ := complex.exp (complex.I * 2 * real.pi * (2 / 3))

theorem proof_problem : 
  (w / (1 + w ^ 3) + w ^ 2 / (1 + w ^ 6) + w ^ 3 / (1 + w ^ 9) = 0) :=
by
  have h : w ^ 9 = 1, 
    sorry,
  have h_nonzero : w ≠ 1, 
    sorry,
  have h_w_sum_to_zero : w + w ^ 2 + w ^ 3 = 0,
    sorry,
  have h_1_w3_2 : 1 + w ^ 3 = 2,
    sorry,
  have h_1_w6_2 : 1 + w ^ 6 = 2,
    sorry,
  calc
    (w / (1 + w ^ 3) + w ^ 2 / (1 + w ^ 6) + w ^ 3 / (1 + w ^ 9))
      = w / 2 + w ^ 2 / 2 + w ^ 3 / 2 : by sorry
  ... = (w + w ^ 2 + w ^ 3) / 2      : by sorry
  ... = 0                           : by sorry

end proof_problem_l78_78736


namespace remainder_sum_l78_78339

theorem remainder_sum (n : ℤ) (h : n ≡ 10 [MOD 24]) : 
  (n % 4 + n % 6) = 6 :=
by sorry

end remainder_sum_l78_78339


namespace largest_25_supporting_X_l78_78123

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i : ℝ) ∈ Set.Ico (- (1 / 2) : ℝ) ((25 : ℝ) / 2 + (1 / 2)) → 
  ∃ i, |a i - 1 / 2| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ 
  (∀ Y : ℝ, Y > X → ¬is_25_supporting Y) ∧ X = 0.02 :=
sorry

end largest_25_supporting_X_l78_78123


namespace problem1_problem2_l78_78247

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

-- Problem (1): Prove the range of values for a such that f(x) ≤ a on [-3, 1]
theorem problem1 (a : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → f(x) ≤ a) ↔ (4 ≤ a) :=
by
  sorry

-- Problem (2): Solve f(x) > 3x
theorem problem2 : ∀ x : ℝ, f(x) > 3 * x ↔ (x > 4 ∨ (-4 < x ∧ x < 1)) :=
by
  sorry

end problem1_problem2_l78_78247


namespace nadia_probability_condition_l78_78348

noncomputable def probability_no_favorite_track :: (tracks : List ℕ) (duration : ℕ) (favorite_track_length : ℕ) : ℚ :=
  let n := tracks.length
  let total_permutations := nat.factorial n
  let valid_permutations := -- Calculate the number of valid permutations where the favorite track is in the first 360 seconds
     -- Implementation of valid permutation calculation goes here
  1 - (valid_permutations / total_permutations)

theorem nadia_probability_condition (tracks : List ℕ) (duration : ℕ) (favorite_track_length : ℕ) :
  tracks.length = 12 →
  ∀ i, (i < 12 → tracks.nth i = some (20 + 20 * i)) →
  favorite_track_length = 280 →
  duration = 360 →
  let prob := probability_no_favorite_track tracks duration favorite_track_length 
  prob = 1 - -- Computed value
 := by 
  -- skip proof
  sorry

end nadia_probability_condition_l78_78348


namespace order_count_of_lecturers_l78_78878

theorem order_count_of_lecturers (total_lecturers : ℕ) (condition : total_lecturers = 6) :
  let total_permutations := nat.factorial total_lecturers in
  let valid_orders := total_permutations / 2 in
  valid_orders = 360 :=
by {
  sorry
}

end order_count_of_lecturers_l78_78878


namespace sum_of_fractions_l78_78563

-- Definition of the fractions given as conditions
def frac1 := 2 / 10
def frac2 := 4 / 40
def frac3 := 6 / 60
def frac4 := 8 / 30

-- Statement of the theorem to prove
theorem sum_of_fractions : frac1 + frac2 + frac3 + frac4 = 2 / 3 := by
  sorry

end sum_of_fractions_l78_78563


namespace oliver_total_money_l78_78353

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l78_78353


namespace total_surface_area_l78_78405

theorem total_surface_area (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 :=
sorry

end total_surface_area_l78_78405


namespace max_number_of_snowmen_l78_78515

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78515


namespace length_de_l78_78853

theorem length_de (a b c d e : ℝ) (ab bc cd de ac ae : ℝ)
  (H1 : ab = 5)
  (H2 : bc = 2 * cd)
  (H3 : ac = ab + bc)
  (H4 : ac = 11)
  (H5 : ae = ab + bc + cd + de)
  (H6 : ae = 18) :
  de = 4 :=
by {
  sorry
}

-- Explanation:
-- a, b, c, d, e are points on a straight line
-- ab, bc, cd, de, ac, ae are lengths of segments between these points
-- H1: ab = 5
-- H2: bc = 2 * cd
-- H3: ac = ab + bc
-- H4: ac = 11
-- H5: ae = ab + bc + cd + de
-- H6: ae = 18
-- Prove that de = 4

end length_de_l78_78853


namespace determine_OP_l78_78752

variable (a b c d : ℝ)
variable (O A B C D P : ℝ)
variable (p : ℝ)

def OnLine (O A B C D P : ℝ) : Prop := O < A ∧ A < B ∧ B < C ∧ C < D ∧ B < P ∧ P < C

theorem determine_OP (h : OnLine O A B C D P) 
(hAP : P - A = p - a) 
(hPD : D - P = d - p) 
(hBP : P - B = p - b) 
(hPC : C - P = c - p) 
(hAP_PD_BP_PC : (p - a) / (d - p) = (p - b) / (c - p)) :
  p = (a * c - b * d) / (a - b + c - d) :=
sorry

end determine_OP_l78_78752


namespace max_snowmen_constructed_l78_78477

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78477


namespace halfway_distance_is_semi_major_axis_l78_78459

-- Define the major axis and the semi-major axis
noncomputable def major_axis_length (perihelion aphelion : ℝ) : ℝ := perihelion + aphelion
noncomputable def semi_major_axis (perihelion aphelion : ℝ) : ℝ := major_axis_length perihelion aphelion / 2

-- Define the distance calculation at the halfway point of the orbit
def halfway_distance (perihelion aphelion : ℝ) : ℝ := semi_major_axis perihelion aphelion

-- The theorem to be proved
theorem halfway_distance_is_semi_major_axis (perihelion aphelion : ℝ) 
  (h_perihelion : perihelion = 3)
  (h_aphelion : aphelion = 15) :
  halfway_distance perihelion aphelion = 9 :=
by
  sorry

end halfway_distance_is_semi_major_axis_l78_78459


namespace watch_cost_l78_78549

variables (w s : ℝ)

theorem watch_cost (h1 : w + s = 120) (h2 : w = 100 + s) : w = 110 :=
by
  sorry

end watch_cost_l78_78549


namespace fractional_equation_solution_l78_78803

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l78_78803


namespace solution_set_of_inequality_l78_78624

variable {x : ℝ} (f : ℝ → ℝ)
-- Conditions
axiom f_deriv : ∀ x : ℝ, HasDerivAt f (f' x)
axiom f_even_diff : ∀ x : ℝ, f(x) - f(-x) = 2 * x^3
axiom f_prime_pos : ∀ x : ℝ, 0 ≤ x → f'(x) > 3 * x^2

-- Theorem to be proved
theorem solution_set_of_inequality : { x | f(x) - f(x-1) > 3 * x^2 - 3 * x + 1 } = { x | x > 1 / 2 } :=
by
  -- This proof is left as an exercise
  sorry

end solution_set_of_inequality_l78_78624


namespace airplane_altitude_l78_78155

-- We introduce variables and conditions according to part a)
variable (h : ℝ) -- altitude of the airplane in miles
variable (AB : ℝ) -- distance between Alice and Bob
variable (θ_Alice θ_Bob : ℝ) -- angles of elevation for Alice and Bob

-- Set the known values
def alice_bob_distance := (AB = 12)
def alice_angle := (θ_Alice = 45)
def bob_angle := (θ_Bob = 30)

-- The expression for law of cosines to solve for h
def law_of_cosines := (12^2 = h^2 + (h * real.sqrt 3)^2 - 2 * h * h * real.sqrt 3 * real.cos (real.pi / 4))

-- The final statement to prove
theorem airplane_altitude (h:ℝ) (AB:ℝ) (θ_Alice:ℝ) (θ_Bob:ℝ)
  (h8 : h ≈ 8) -- asserting that we found h ≈ 8
  (alice_bob_distance : AB = 12)
  (alice_angle : θ_Alice = 45)
  (bob_angle : θ_Bob = 30) :
  law_of_cosines := 
by sorry

end airplane_altitude_l78_78155


namespace graph_inverse_shifted_l78_78741

-- Defining the conditions
variable {α β : Type*}
variable (f : α → β) (f_inv : β → α)
variable [function.inv_fun f_inv f = function.id]
variable [function.inv_fun f f_inv = function.id]

-- The given point condition for function f
variable (condition_f : f 1 = 0)

-- The goal to be proven
theorem graph_inverse_shifted :
  f_inv 0 + 1 = 2 :=
by
  -- using the condition that f(1) = 0 implies f_inv(0) = 1
  have h_inv_eq : f_inv 0 = 1, from function.left_inverse_inv_fun.eq_inv_fun f_inv function.id _ 0,
  -- substituting f_inv(0) = 1 into the equation
  rw h_inv_eq,
  -- simplify the equation
  simp,
  exact rfl

end graph_inverse_shifted_l78_78741


namespace max_snowmen_constructed_l78_78479

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78479


namespace original_price_united_flight_l78_78567

theorem original_price_united_flight : 
  ∀ (delta_original united_original : ℝ) 
    (delta_discount united_discount : ℝ)
    (savings : ℝ), 
  delta_original = 850 → 
  delta_discount = 0.20 → 
  united_discount = 0.30 → 
  savings = 90 →
  let delta_discounted := delta_original * (1 - delta_discount),
      united_discounted := united_original * (1 - united_discount)
  in
  delta_discounted + savings = united_discounted →
  united_original = 1100 := 
by
  intros _ _ _ _ hd_orig hu_orig hd_disc hu_disc hsavings h_eq;
  sorry

end original_price_united_flight_l78_78567


namespace average_repair_rate_l78_78110

variable (total_length : ℝ) (total_time : ℝ)

theorem average_repair_rate (h : total_length = 200) (h2 : total_time = 20) : 
  (total_length / total_time = 10) ∧ (total_time / total_length = 0.1) := 
by {
  sorry, -- Proof goes here
}

end average_repair_rate_l78_78110


namespace king_total_payment_l78_78468

theorem king_total_payment
  (crown_cost : ℕ)
  (architect_cost : ℕ)
  (chef_cost : ℕ)
  (crown_tip_percent : ℕ)
  (architect_tip_percent : ℕ)
  (chef_tip_percent : ℕ)
  (crown_tip : ℕ)
  (architect_tip : ℕ)
  (chef_tip : ℕ)
  (total_crown_cost : ℕ)
  (total_architect_cost : ℕ)
  (total_chef_cost : ℕ)
  (total_paid : ℕ) :
  crown_cost = 20000 →
  architect_cost = 50000 →
  chef_cost = 10000 →
  crown_tip_percent = 10 →
  architect_tip_percent = 5 →
  chef_tip_percent = 15 →
  crown_tip = crown_cost * crown_tip_percent / 100 →
  architect_tip = architect_cost * architect_tip_percent / 100 →
  chef_tip = chef_cost * chef_tip_percent / 100 →
  total_crown_cost = crown_cost + crown_tip →
  total_architect_cost = architect_cost + architect_tip →
  total_chef_cost = chef_cost + chef_tip →
  total_paid = total_crown_cost + total_architect_cost + total_chef_cost →
  total_paid = 86000 := by
  sorry

end king_total_payment_l78_78468


namespace max_snowmen_l78_78502

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78502


namespace cos_540_eq_neg1_l78_78445

theorem cos_540_eq_neg1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg1_l78_78445


namespace rectangles_may_not_be_similar_l78_78846

def is_similar (shape1 shape2 : Type) : Prop := sorry -- Define 'similar figures'

-- Define each shape type
structure EquilateralTriangle := (side : ℝ)
structure IsoscelesRightTriangle := (leg : ℝ)
structure Rectangle := (length : ℝ) (width : ℝ)
structure Square := (side : ℝ)

-- Propositions for similarity conditions
def are_similar_equilateral_triangles (T1 T2 : EquilateralTriangle) : Prop :=
  is_similar T1 T2

def are_similar_isosceles_right_triangles (T1 T2 : IsoscelesRightTriangle) : Prop :=
  is_similar T1 T2

def are_similar_rectangles (R1 R2 : Rectangle) : Prop :=
  is_similar R1 R2

def are_similar_squares (S1 S2 : Square) : Prop :=
  is_similar S1 S2

theorem rectangles_may_not_be_similar :
  ¬ ∀ (R1 R2 : Rectangle), is_similar R1 R2 :=
sorry

end rectangles_may_not_be_similar_l78_78846


namespace annie_weeks_off_sick_l78_78159

-- Define the conditions and the question
def weekly_hours_chess : ℕ := 2
def weekly_hours_drama : ℕ := 8
def weekly_hours_glee : ℕ := 3
def semester_weeks : ℕ := 12
def total_hours_before_midterms : ℕ := 52

-- Define the proof problem
theorem annie_weeks_off_sick :
  let total_weekly_hours := weekly_hours_chess + weekly_hours_drama + weekly_hours_glee
  let attended_weeks := total_hours_before_midterms / total_weekly_hours
  semester_weeks - attended_weeks = 8 :=
by
  -- Automatically prove by computation of above assumptions.
  sorry

end annie_weeks_off_sick_l78_78159


namespace conic_section_is_ellipse_l78_78052

noncomputable def is_ellipse (x y : ℝ) : Prop :=
  sqrt (x^2 + (y + 2)^2) + sqrt ((x - 6)^2 + (y - 4)^2) = 14

theorem conic_section_is_ellipse : ∀ x y : ℝ, is_ellipse x y → true := sorry

end conic_section_is_ellipse_l78_78052


namespace BD_eq_DC_l78_78813

theorem BD_eq_DC (A B C X D Z Y : Type)
  (h1 : AB ≠ AC)
  (h2 : X ∈ (perp_bisector B C) ∩ (angle_bisector A B C))
  (h3 : foot_perp X AB = Z)
  (h4 : foot_perp X AC = Y)
  (h5 : line_through Z Y ∩ BC = D)
  : BD / DC = 1 := 
sorry

end BD_eq_DC_l78_78813


namespace min_mnp_l78_78673

open Nat

-- We first declare the primes and the specified conditions as theorem

theorem min_mnp (m n p : ℕ) (hm : Prime m) (hn : Prime n) (hp : Prime p) (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p) (h_add : m + n = p) :
  m * n * p ≥ 30 :=
by
  sorry

end min_mnp_l78_78673


namespace player_b_frozen_probability_l78_78433

/-- 
In a board game played by rolling a pair of fair 6-sided dice and moving forward the number of spaces indicated by the sum on the dice, if player A is currently 8 spaces behind player B, 
prove that the probability that player B will be frozen after player A rolls is 1/4.
-/
theorem player_b_frozen_probability :
  let outcomes := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
                   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                   (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)],
      total_outcomes := List.length outcomes,
      favorable_outcomes := List.filter (λ (d : Nat × Nat), 
                                          d.1 + d.2 = 8) outcomes,
      probability := List.length favorable_outcomes / total_outcomes in
  probability = 1 / 4 :=
by sorry

end player_b_frozen_probability_l78_78433


namespace polynomial_product_is_square_l78_78368

theorem polynomial_product_is_square (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 :=
by
  sorry

end polynomial_product_is_square_l78_78368


namespace min_value_f_l78_78942

def f (a x : ℝ) : ℝ := 3 - 2 * a * Real.sin x - Real.cos x ^ 2

theorem min_value_f (a : ℝ) : 
  ∃ (min_val : ℝ), 
    (∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), f a x ≥ min_val) ∧
    (
      (a < -1/2 ∧ min_val = a + 9/4) ∨ 
      (-1/2 ≤ a ∧ a ≤ 1 ∧ min_val = -a^2 + 2) ∨ 
      (a > 1 ∧ min_val = -2 * a + 3)
    ) :=
by
  sorry

end min_value_f_l78_78942


namespace part1_part2_l78_78172

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |2 * x + a|

theorem part1 (x : ℝ) : f x 1 + |x - 1| ≥ 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∃ x : ℝ, f x a = 2) : a = 2 ∨ a = -6 :=
  sorry

end part1_part2_l78_78172


namespace probability_average_five_l78_78285

noncomputable def choose (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Define the conditions as hypotheses
def labels : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_selected := 7
def target_sum := 35

-- Define the problem statement as a theorem
theorem probability_average_five (h : ∀ (s : Finset ℕ), s.card = num_selected → s.sum id = target_sum → 
  ∃ (l : list (ℕ × ℕ)), l = [(1, 9), (2, 8), (3, 7), (4, 6)] ∧ (length l) = 4) :
  let total_ways := choose 9 7 in
  let favorable_ways := 4 in
  (favorable_ways / total_ways : ℚ) = (1 / 9 : ℚ) := 
sorry

end probability_average_five_l78_78285


namespace find_decreasing_intervals_l78_78931

noncomputable def is_decreasing_interval (k : ℤ) : Set ℝ :=
{ x : ℝ | k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 }

theorem find_decreasing_intervals :
  ∀ {k : ℤ}, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) ↔ 
    is_decreasing_interval k x := sorry

end find_decreasing_intervals_l78_78931


namespace possible_degrees_of_remainder_l78_78048

theorem possible_degrees_of_remainder (p : Polynomial ℝ) :
  ∃ r q : Polynomial ℝ, p = q * (3 * X^3 - 4 * X^2 + 5 * X - 6) + r ∧ r.degree < 3 :=
sorry

end possible_degrees_of_remainder_l78_78048


namespace intersection_height_pole_lines_l78_78418

theorem intersection_height_pole_lines 
  (height1 height2 distance : ℝ) 
  (line1 : ℝ → ℝ) 
  (line2 : ℝ → ℝ) :
  height1 = 30 ∧ height2 = 90 ∧ distance = 120 ∧ 
  (∀ x, line1 x = -x / 4 + 30) ∧ 
  (∀ x, line2 x = 3 * x / 4) → 
  ∃ x y, line1 x = y ∧ line2 x = y ∧ y = 18 :=
by
  intros
  rcases this with ⟨h1, h2, d, l1_eq, l2_eq⟩
  sorry

end intersection_height_pole_lines_l78_78418


namespace general_term_inequality_l78_78404

open Real

noncomputable def S (n : ℕ) : ℝ := b^n + r

variables (b r : ℝ) (Hb : b > 0) (Hb_ne_1 : b ≠ 1)
variables (a : ℕ → ℝ)

-- Define the sequence a_n with the given conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → S n = (finset.range n).sum (λ k, a (k + 1))

-- Define b_n for b = 2
def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ := 2 * (log 2 (a n) + 1)

-- The two proofs to be established
theorem general_term (H : geometric_sequence a) :
  ∀ n : ℕ, n > 0 → a n = (b - 1) * b ^ (n - 1) :=
sorry

theorem inequality
  (H : geometric_sequence a) (Hb_2 : b = 2) :
  ∀ n : ℕ, n > 0 →
    (finset.range n).prod (λ i, (b_n a (i + 1) + 1) / b_n a (i + 1)) > sqrt (↑n + 1) :=
sorry

end general_term_inequality_l78_78404


namespace range_of_a_l78_78989

noncomputable def f' (a x : ℝ) := a * (x + 1) * (x - a)

theorem range_of_a (a : ℝ) (h : ∀ x, ∀ f : ℝ → ℝ, (∀ x, f' a x = deriv f x) → local_maximum f a) :
  -1 < a ∧ a < 0 := 
sorry

end range_of_a_l78_78989


namespace problem1_problem2_l78_78913

theorem problem1 :
  0.064 ^ (-1 / 3) - (-7 / 8) ^ 0 + 16 ^ 0.75 + 0.01 ^ (1 / 2) = 48 / 5 :=
by sorry

theorem problem2 :
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 
  - 25 ^ (Real.log 3 / Real.log 5) = -7 :=
by sorry

end problem1_problem2_l78_78913


namespace logger_forest_problem_l78_78365

/--
One logger can cut down 6 trees per day.
The forest is a rectangle measuring 4 miles by 6 miles, and each square mile has 600 trees.
If it takes 8 loggers 10 months to cut down all the trees, prove that there are 30 days in each month.
-/
theorem logger_forest_problem :
  let trees_per_logger_per_day := 6
  let forest_area := 4 * 6
  let trees_per_square_mile := 600
  let total_trees := forest_area * trees_per_square_mile
  let loggers := 8
  let days_per_month := 30
  let months := 10
  let total_days := months * days_per_month
  let trees_per_day_all_loggers := loggers * trees_per_logger_per_day
  let total_trees_cut := trees_per_day_all_loggers * total_days
  total_trees_cut = total_trees → 
  days_per_month = 30 := 
by {
  intros,
  sorry
}

end logger_forest_problem_l78_78365


namespace distribute_volunteers_ways_l78_78933

-- Define a function to count the number of ways to distribute 5 volunteers to 3 venues with given conditions.
noncomputable def countWays : ℕ :=
  let c (n k : ℕ) := Nat.choose n k
  let a (n : ℕ) := Nat.perm n n
  c 5 3 * a 3 + (c 5 2 * c 3 2) / a 2 * a 3

-- The theorem that states that our countWays function returns 150
theorem distribute_volunteers_ways : countWays = 150 := by
  sorry

end distribute_volunteers_ways_l78_78933


namespace smallest_positive_period_of_f_is_pi_l78_78184

def f (x : Real) : Real := 1 - 2 * Real.sin x ^ 2

theorem smallest_positive_period_of_f_is_pi : ∃ T > 0, (∀ x : Real, f (x + T) = f x) ∧ ∀ T' > 0, (∀ x : Real, f (x + T') = f x) → T' ≥ T :=
  sorry

end smallest_positive_period_of_f_is_pi_l78_78184


namespace geometric_seq_condition_l78_78983

-- Defining a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Defining an increasing sequence
def is_increasing_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The condition to be proved
theorem geometric_seq_condition (a : ℕ → ℝ) (h_geo : is_geometric_seq a) :
  (a 0 < a 1 → is_increasing_seq a) ∧ (is_increasing_seq a → a 0 < a 1) :=
by 
  sorry

end geometric_seq_condition_l78_78983


namespace katie_remaining_problems_l78_78316

variable (total_problems : Nat) (finished_problems : Nat)

theorem katie_remaining_problems (h1 : total_problems = 9) (h2 : finished_problems = 5) :
  total_problems - finished_problems = 4 :=
  by
    rw [h1, h2]
    exact rfl

end katie_remaining_problems_l78_78316


namespace max_fans_theorem_l78_78073

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l78_78073


namespace largest_convex_ngon_in_2004_grid_l78_78064

noncomputable def largest_convex_ngon_vertices (grid_size : ℕ) : ℕ :=
  if grid_size = 2004 then 561 else 0

theorem largest_convex_ngon_in_2004_grid :
  ∀ grid_size : ℕ, grid_size = 2004 → largest_convex_ngon_vertices grid_size = 561 :=
by
  intro grid_size h
  rw [largest_convex_ngon_vertices]
  split_ifs
  exact h
  contradiction

end largest_convex_ngon_in_2004_grid_l78_78064


namespace max_fans_theorem_l78_78075

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l78_78075


namespace truck_driver_earns_90_dollars_l78_78548

def cost_per_gallon : ℝ := 2
def efficiency : ℝ := 10
def rate : ℝ := 30
def payment_per_mile : ℝ := 0.5
def time : ℝ := 10

theorem truck_driver_earns_90_dollars (cpg eq rate ppm time : ℝ) :
  cpg = 2 → eq = 10 → rate = 30 → ppm = 0.5 → time = 10 → 
  (rate * time * ppm) - ((rate * time / eq) * cpg) = 90 := 
by {
  intros,
  sorry
}

end truck_driver_earns_90_dollars_l78_78548


namespace inequality_geq_l78_78336

theorem inequality_geq (t : ℝ) (n : ℕ) (ht : t ≥ 1/2) : 
  t^(2*n) ≥ (t-1)^(2*n) + (2*t-1)^n := 
sorry

end inequality_geq_l78_78336


namespace complex_pow_eight_l78_78321

theorem complex_pow_eight (z : ℂ) (h : z = (Complex.mk (Real.sqrt 3) 1) / 2) :
  z ^ 8 = -1 := by
  sorry

end complex_pow_eight_l78_78321


namespace fraction_result_l78_78026

theorem fraction_result (x : ℚ) (h₁ : x * (3/4) = (1/6)) : (x - (1/12)) = (5/36) := 
sorry

end fraction_result_l78_78026


namespace neg_of_univ_false_proof_of_neg1_false_neg_of_exist_false_proof_of_neg2_false_l78_78380

section ProofProblem1
variables {x : ℝ}

theorem neg_of_univ_false : ¬(∀ x : ℝ, x ^ 2 + x + 1 > 0) ↔ ∃ x_0 : ℝ, x_0 ^ 2 + x_0 + 1 ≤ 0 := sorry

theorem proof_of_neg1_false : ¬(¬(∃ x_0 : ℝ, x_0 ^ 2 + x_0 + 1 ≤ 0)) := by
  intro h
  let contradiction := λ x_0 : ℝ, x_0 ^ 2 + x_0 + 1 ≥ 3 / 4
  sorry
end ProofProblem1

section ProofProblem2
variables {x y : ℝ}

theorem neg_of_exist_false : ¬(∃ x_0 y_0 : ℝ, sqrt (x_0 - 1) + (y_0 + 1) ^ 2 = 0) ↔ ∀ x y : ℝ, sqrt ((x - 1) ^ 2) + (y + 1) ^ 2 ≠ 0 := sorry

theorem proof_of_neg2_false : ¬(¬(∀ x y : ℝ, sqrt ((x - 1) ^ 2) + (y + 1) ^ 2 ≠ 0)) := by
  intro h
  have : (1 : ℝ) = 1, from rfl
  have : (-1 : ℝ) = -1, from rfl
  let contradiction := sqrt ((1 - 1) ^ 2) + (-1 + 1) ^ 2 = 0
  sorry
end ProofProblem2

end neg_of_univ_false_proof_of_neg1_false_neg_of_exist_false_proof_of_neg2_false_l78_78380


namespace stratified_sampling_correct_l78_78452

/-- Given the total number of students in a school and the number in each grade,
verify the number of students to be drawn from each year using stratified sampling. -/
theorem stratified_sampling_correct :
  ∀ (total_students year1_students year2_students year3_students sample_size : ℕ), 
  total_students = 900 →
  year1_students = 300 →
  year2_students = 200 →
  year3_students = 400 →
  sample_size = 45 →
  let year1_sample := Nat.div (300 * 45) 900,
      year2_sample := Nat.div (200 * 45) 900,
      year3_sample := Nat.div (400 * 45) 900
  in 
  year1_sample = 15 ∧ year2_sample = 10 ∧ year3_sample = 20 :=
by 
  intros total_students year1_students year2_students year3_students sample_size h_tot h_y1 h_y2 h_y3 h_sample
  let year1_sample := Nat.div (300 * 45) 900
  let year2_sample := Nat.div (200 * 45) 900
  let year3_sample := Nat.div (400 * 45) 900
  split
  exact rfl
  split
  exact rfl
  exact rfl
  sorry

end stratified_sampling_correct_l78_78452


namespace mike_average_rate_l78_78346

noncomputable def average_rate (D : ℝ) (d1 v1 d2 v2 d3 v3 d4 v4 : ℝ) (rest_time : ℝ) : ℝ :=
  let t1 := d1 / v1
  let t2 := d2 / v2
  let t3 := d3 / v3
  let t4 := d4 / v4
  let total_time := t1 + t2 + t3 + t4 + rest_time
  D / total_time

theorem mike_average_rate :
  average_rate 800 200 80 300 60 150 70 150 65 0.5 ≈ 64.24 :=
by
  sorry

end mike_average_rate_l78_78346


namespace remainder_division_lemma_l78_78602

theorem remainder_division_lemma (j : ℕ) (hj : 0 < j) (hmod : 132 % (j^2) = 12) : 250 % j = 0 :=
sorry

end remainder_division_lemma_l78_78602


namespace alpha_beta_sum_l78_78662

theorem alpha_beta_sum (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hroots : ∃ (x y : ℝ), (x^2 - 3 * real.sqrt 3 * x + 4 = 0) ∧ 
              x = real.tan α ∧ y = real.tan β) :
  α + β = 2 * π / 3 :=
by
  sorry

end alpha_beta_sum_l78_78662


namespace centroid_quadrilateral_area_ratio_l78_78723

open Function

/-- Definition of a centroid of a triangle in ℝ^3 --/
def centroid (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (((p1.1 + p2.1 + p3.1) / 3), ((p1.2 + p2.2 + p3.2) / 3), ((p1.3 + p2.3 + p3.3) / 3))

/-- The main theorem: Calculate the area ratio of the quadrilateral formed by the centroids
    of the triangles inside a parallelogram to the parallelogram --/
theorem centroid_quadrilateral_area_ratio
  (A B C D : ℝ × ℝ × ℝ)
  (parallelogram : A - B + C - D = 0) :
  let G_A := centroid B C D,
      G_B := centroid A C D,
      G_C := centroid A B D,
      G_D := centroid A B C
  in [G_A, G_B, G_C, G_D] / [A, B, C, D] = 1 / 9 :=
by
  sorry

end centroid_quadrilateral_area_ratio_l78_78723


namespace sum_of_distinct_GH_product_l78_78792

theorem sum_of_distinct_GH_product :
  ∀ (G H : ℕ), G < 10 ∧ H < 10 ∧
  (ite (2 * H + 6) % 8 = 0 true false) ∧
  (let d_sum := 5 + 4 + 1 + 5 + 0 + 7 + 2 + 6 in
    (d_sum + G + H) % 9 = 0) →
  (GH = G * H ∧ (GH = 5 ∨ GH = 54)) →
  (GH = 5 ∨ GH = 54) →
  59 := by sorry

end sum_of_distinct_GH_product_l78_78792


namespace maximum_distance_and_line_equation_l78_78967

-- Define the point P
def P : ℝ × ℝ := (-2, -1)

-- Define the line family l parametrized by λ
def line_family (λ : ℝ) (x y : ℝ) : Prop := 
  (1 + 3 * λ) * x + (1 + λ) * y - 2 - 4 * λ = 0

-- Define the conditions based on the problem
def condition1 (x y : ℝ) : Prop := x + y - 2 = 0
def condition2 (x y : ℝ) : Prop := 3 * x + y - 4 = 0

-- Prove that the maximum distance from P to any line in the family 
-- is sqrt(13), and it occurs when the line's equation is 2x - 3y + 1 = 0
theorem maximum_distance_and_line_equation :
  ∃ Q : ℝ × ℝ, 
    -- Conditions: Q satisfies both derived equations
    condition1 Q.1 Q.2 ∧ condition2 Q.1 Q.2 ∧
    -- Maximum distance from P to Q is sqrt(13)
    (dist P Q = sqrt 13) ∧
    -- Equation of the line passing through P and Q is 2x - 3y + 1 = 0
    (∀ (x y : ℝ), (x - Q.1) * (Q.2 + 1) = (y + 1) * (Q.1 + 2) →
                 2 * x - 3 * y + 1 = 0) :=
sorry

end maximum_distance_and_line_equation_l78_78967


namespace fractional_equation_solution_l78_78804

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l78_78804


namespace n_queens_problem_l78_78024

noncomputable def placeQueens (N : ℕ) : Option (Array ℕ) :=
sorry

theorem n_queens_problem (N : ℕ) : 
  ∃ (V : Array ℕ), 
    (∀ {i j : ℕ}, i < N → j < N → i ≠ j → V[i] ≠ V[j] 
        ∧ abs (V[i] - V[j]) ≠ abs (i - j)) 
    ∨ placeQueens N = none := sorry

end n_queens_problem_l78_78024


namespace find_FV_sum_l78_78768

-- Define the conditions
variable {A V F: Type} [metric_space V]

-- Function definitions representing the given conditions
def AF : ℝ := 25
def AV : ℝ := 28
variable {d : ℝ} -- Length FV

-- Given ratio determining equation for d
def ratio_condition (d : ℝ) : Prop := 4 * (25 - d / 4) ^ 2 + d ^ 2 = 10000

-- Definition of the sum of all possible values of FV
def sum_of_all_possible_FV : ℝ := 200 / 3

-- Proof goal statement
theorem find_FV_sum (h: ratio_condition d) : d = sum_of_all_possible_FV :=
sorry

end find_FV_sum_l78_78768


namespace probability_of_four_or_more_same_l78_78946

noncomputable def at_least_four_same_value_probability : ℚ :=
  let p_all_five_same := (1 / 6) ^ 4
  let p_exactly_four_same := (5 * (1 / 6) ^ 3 * 5 / 6) in
  p_all_five_same + p_exactly_four_same

theorem probability_of_four_or_more_same :
  at_least_four_same_value_probability = 13 / 648 :=
by 
  sorry

end probability_of_four_or_more_same_l78_78946


namespace relationship_among_abc_l78_78731

noncomputable def a : ℝ := 6^0.7
noncomputable def b : ℝ := 0.7^6
noncomputable def c : ℝ := Real.log 6 / Real.log 0.7

theorem relationship_among_abc : c < b ∧ b < a := by
  sorry

end relationship_among_abc_l78_78731


namespace eggs_per_basket_l78_78375

theorem eggs_per_basket
  (kids : ℕ)
  (friends : ℕ)
  (adults : ℕ)
  (baskets : ℕ)
  (eggs_per_person : ℕ)
  (htotal : kids + friends + adults + 1 = 20)
  (eggs_total : (kids + friends + adults + 1) * eggs_per_person = 180)
  (baskets_count : baskets = 15)
  : (180 / 15) = 12 :=
by
  sorry

end eggs_per_basket_l78_78375


namespace simplify_proof_l78_78378

def simplify_fractions_product : ℚ :=
  (27 / 25) * (20 / 33) * (55 / 54)

theorem simplify_proof :
  simplify_fractions_product = 25 / 3 :=
by
  sorry

end simplify_proof_l78_78378


namespace number_in_289th_position_divisible_by_17_or_20_l78_78848

/-- The sequence number in the 289th position among all non-zero natural numbers divisible by 17 or 20 is 2737. -/
theorem number_in_289th_position_divisible_by_17_or_20 : 
  ∃ n : ℕ, (∀ m, (m ≤ 289 → (17 ∣ n ∨ 20 ∣ n)) ∧ (17 ∣ m ∨ 20 ∣ m)) ∧ (n = 2737) :=
begin
  sorry
end

end number_in_289th_position_divisible_by_17_or_20_l78_78848


namespace all_functions_zero_l78_78940

theorem all_functions_zero (f : ℕ → ℤ) (h : ∀ (m n : ℕ), n ∣ f(m) ↔ m ∣ ∑ d : ℕ in divisors n, f(d)) : 
    ∀ n : ℕ, f(n) = 0 :=
by
  sorry

end all_functions_zero_l78_78940


namespace max_number_of_snowmen_l78_78514

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78514


namespace customers_left_is_31_l78_78151

-- Define the initial number of customers
def initial_customers : ℕ := 33

-- Define the number of additional customers
def additional_customers : ℕ := 26

-- Define the final number of customers after some left and new ones came
def final_customers : ℕ := 28

-- Define the number of customers who left 
def customers_left (x : ℕ) : Prop :=
  (initial_customers - x) + additional_customers = final_customers

-- The proof statement that we aim to prove
theorem customers_left_is_31 : ∃ x : ℕ, customers_left x ∧ x = 31 :=
by
  use 31
  unfold customers_left
  sorry

end customers_left_is_31_l78_78151


namespace recycling_money_l78_78601

theorem recycling_money (cans_per_unit : ℕ) (payment_per_unit_cans : ℝ) 
  (newspapers_per_unit : ℕ) (payment_per_unit_newspapers : ℝ) 
  (total_cans : ℕ) (total_newspapers : ℕ) : 
  cans_per_unit = 12 → payment_per_unit_cans = 0.50 → 
  newspapers_per_unit = 5 → payment_per_unit_newspapers = 1.50 → 
  total_cans = 144 → total_newspapers = 20 → 
  (total_cans / cans_per_unit) * payment_per_unit_cans + 
  (total_newspapers / newspapers_per_unit) * payment_per_unit_newspapers = 12 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end recycling_money_l78_78601


namespace total_length_is_23_l78_78152

-- Define the dimensions of the rectangle and the square
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 6
def square_side : ℕ := 3

-- Define the remaining segment lengths in Figure 2
def bottom_segment : ℕ := rectangle_length
def left_segment : ℕ := square_side
def top_segment : ℕ := rectangle_length - square_side
def right_segment : ℕ := rectangle_width - square_side

-- Define the total length of segments in Figure 2
def total_length_of_segments : ℕ :=
  bottom_segment + left_segment + top_segment + right_segment

-- The theorem that we need to prove
theorem total_length_is_23 :
  total_length_of_segments = 23 :=
by
  simp [total_length_of_segments, bottom_segment, left_segment, top_segment, right_segment, rectangle_length, rectangle_width, square_side]
  sorry

end total_length_is_23_l78_78152


namespace number_of_functions_l78_78179

noncomputable def norm (A B : ℝ × ℝ) : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2

def S : set (ℝ × ℝ) := { p | p.1 ∈ finset.range 101 ∧ p.2 ∈ finset.range 101 }

theorem number_of_functions (f : (ℝ × ℝ) → (ℝ × ℝ)) :
  (∀ A B ∈ S, (norm A B) % 101 = (norm (f A) (f B)) % 101) →
  (finset.card (finset.image f (finset.filter S finset.univ)) = 2040200) :=
sorry

end number_of_functions_l78_78179


namespace sum_of_coordinates_l78_78814

theorem sum_of_coordinates :
  let points := {p : ℝ × ℝ | (abs (p.2 - 15) = 6 ∧ (p.1 - 6)^2 + (p.2 - 15)^2 = 14^2)} in
  (∑ p in points, p.1 + p.2) = 84 :=
sorry

end sum_of_coordinates_l78_78814


namespace solve_integral_eqn_l78_78379

noncomputable def K (x t : ℝ) : ℝ :=
if 0 <= x ∧ x <= t then (x + 1) * t
else if t <= x ∧ x <= 1 then (t + 1) * x
else 0

theorem solve_integral_eqn (φ : ℝ → ℝ) (λ : ℝ) :
  (∀ x, φ(x) - λ * ∫ t in 0..1, K x t * φ(t) = cos (π * x)) →
  (∀ x, φ(x) = cos (π * x) + λ * ((1 + exp(1)) / (1 + π^2) * exp(x) / (λ - 1) - π / (2 * (λ + π^2)) * (sin (π * x) + π * cos (π * x))) 
   + C * (sin (n * π * x) + n * π * cos (n * π * x))) :=
sorry

end solve_integral_eqn_l78_78379


namespace problem1_problem2_solution_l78_78861

noncomputable def trig_expr : ℝ :=
  3 * Real.tan (30 * Real.pi / 180) - (Real.tan (45 * Real.pi / 180))^2 + 2 * Real.sin (60 * Real.pi / 180)

theorem problem1 : trig_expr = 2 * Real.sqrt 3 - 1 :=
by
  -- Proof omitted
  sorry

noncomputable def quad_eq (x : ℝ) : Prop := 
  (3*x - 1) * (x + 2) = 11*x - 4

theorem problem2_solution (x : ℝ) : quad_eq x ↔ (x = (3 + Real.sqrt 3) / 3 ∨ x = (3 - Real.sqrt 3) / 3) :=
by
  -- Proof omitted
  sorry

end problem1_problem2_solution_l78_78861


namespace factor_polynomial_l78_78930

theorem factor_polynomial 
(a b c d : ℝ) :
  a^3 * (b^2 - d^2) + b^3 * (c^2 - a^2) + c^3 * (d^2 - b^2) + d^3 * (a^2 - c^2)
  = (a - b) * (b - c) * (c - d) * (d - a) * (a^2 + ab + ac + ad + b^2 + bc + bd + c^2 + cd + d^2) :=
sorry

end factor_polynomial_l78_78930


namespace shaded_area_is_14_percent_l78_78417

def side_length : ℕ := 20
def rectangle_width : ℕ := 35
def rectangle_height : ℕ := side_length
def rectangle_area : ℕ := rectangle_width * rectangle_height
def overlap_length : ℕ := 2 * side_length - rectangle_width
def shaded_area : ℕ := overlap_length * side_length
def shaded_percentage : ℚ := (shaded_area : ℚ) / rectangle_area * 100

theorem shaded_area_is_14_percent : shaded_percentage = 14 := by
  sorry

end shaded_area_is_14_percent_l78_78417


namespace complex_power_equality_l78_78161

theorem complex_power_equality :
  (3 * (Complex.cos (Real.pi / 6) + Complex.i * Complex.sin (Real.pi / 6))) ^ 8 = 
  -3280.5 - 3280.5 * Complex.i * Real.sqrt 3 :=
by
  sorry

end complex_power_equality_l78_78161


namespace perfect_square_digits_l78_78793

theorem perfect_square_digits (x y : ℕ) (h_ne_zero : x ≠ 0) (h_perfect_square : ∀ n: ℕ, n ≥ 1 → ∃ k: ℕ, (10^(n + 2) * x + 10^(n + 1) * 6 + 10 * y + 4) = k^2) :
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0) :=
sorry

end perfect_square_digits_l78_78793


namespace ratio_areas_triangle_ABR_ACS_eq_one_l78_78753

open EuclideanGeometry

/-- Proof problem: Given a triangle ABC with an obtuse angle at C, a point M on BC, a point D such that
  BCD is acute and A and D are on opposite sides of BC, the circumcircle ω_B of triangle BMD, the circumcircle ω_C of triangle CMD,
  and additional points P, Q, R, and S as given, prove that the ratio of the areas of triangles ABR and ACS is 1. -/
theorem ratio_areas_triangle_ABR_ACS_eq_one
  (A B C M D P Q R S : Point)
  (hABC_obtuse : ∠A C B > 90°)
  (hM_on_BC : M ∈ Line B C)
  (hBCD_acute: ∠B C D < 90° ∧ ∠B D C < 90° ∧ ∠D B C < 90°)
  (hA_D_opposite_sides : (Line B C).same_side A D)
  (hωB_circumcircle : circle_circum B M D)
  (hωC_circumcircle : circle_circum C M D)
  (hP_on_AB : P ≠ B ∧ P ∈ circle_circum B M D ∧ P ∈ Line A B)
  (hQ_on_AC : Q ≠ C ∧ Q ∈ circle_circum C M D ∧ Q ∈ Line A C)
  (hR_on_PD : R ≠ D ∧ R ∈ circle_circum C M D ∧ R ∈ Line P D)
  (hS_on_QD : S ≠ D ∧ S ∈ circle_circum B M D ∧ S ∈ Line Q D) :
  area (triangle A B R) / area (triangle A C S) = 1 := by
  sorry

end ratio_areas_triangle_ABR_ACS_eq_one_l78_78753


namespace smallest_a_for_non_prime_l78_78595

theorem smallest_a_for_non_prime (a : ℕ) (x : ℤ) : 
  (1 ≤ a → ∃ x, Nat.prime (x^4 + a^3) = false) → a = 6 :=
sorry

end smallest_a_for_non_prime_l78_78595


namespace oranges_weigh_4_ounces_each_l78_78746

def apple_weight : ℕ := 4
def max_bag_capacity : ℕ := 49
def num_bags : ℕ := 3
def total_weight : ℕ := num_bags * max_bag_capacity
def total_apple_weight : ℕ := 84
def num_apples : ℕ := total_apple_weight / apple_weight
def num_oranges : ℕ := num_apples
def total_orange_weight : ℕ := total_apple_weight
def weight_per_orange : ℕ := total_orange_weight / num_oranges

theorem oranges_weigh_4_ounces_each :
  weight_per_orange = 4 := by
  sorry

end oranges_weigh_4_ounces_each_l78_78746


namespace at_least_four_boxes_same_item_count_l78_78299

theorem at_least_four_boxes_same_item_count (n_boxes : ℕ) (max_items_per_box : ℕ)
  (h1 : n_boxes = 376) (h2 : max_items_per_box = 125) :
  ∃ k, k ≤ max_items_per_box ∧ 4 ≤ (finset.filter (λ n, n = k) (finset.range n_boxes)).card :=
by
  sorry

end at_least_four_boxes_same_item_count_l78_78299


namespace rational_function_sum_l78_78572

noncomputable def s (x : ℝ) : ℝ := -x^3 + 4 * x
noncomputable def r (x : ℝ) : ℝ := -x

theorem rational_function_sum:
  (s -1 = 1) →
  (s 1 = 3) →
  (r -1 = 1) →
  ∃ x, r x + s x = -x^3 + 3 * x :=
by {
  intro h1 h2 h3,
  use x,
  rw [s x, r x],
  sorry -- Proof steps would go here.
}

end rational_function_sum_l78_78572


namespace max_distinct_fans_l78_78082

theorem max_distinct_fans : 
  let sectors := 6,
      initial_configs := 2^sectors,
      unchanged_configs := 8,
      unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  in unique_configs = 36 :=
by
  let sectors := 6
  let initial_configs := 2^sectors
  let unchanged_configs := 8
  let unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  have h : unique_configs = 36 := by sorry
  exact h

end max_distinct_fans_l78_78082


namespace largest_square_area_l78_78953

theorem largest_square_area (side_len : ℕ) (corner_cut : ℕ) 
  (h1 : side_len = 5) (h2 : corner_cut = 1) : 
  ∃ area : ℕ, area = 9 :=
by 
  have h3 : side_len - 2 * corner_cut = 3, from sorry,
  have h4 : (side_len - 2 * corner_cut) ^ 2 = 9, from sorry,
  exact ⟨9, h4⟩

end largest_square_area_l78_78953


namespace average_speed_l78_78437

theorem average_speed (D : ℝ) (hD : D > 0) :
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 15
  let t3 := (D / 3) / 48
  let total_time := t1 + t2 + t3
  let avg_speed := D / total_time
  avg_speed = 30 :=
by
  sorry

end average_speed_l78_78437


namespace solve_fractional_equation_l78_78806

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l78_78806


namespace gcd_3570_4840_l78_78423

-- Define the numbers
def num1 : Nat := 3570
def num2 : Nat := 4840

-- Define the problem statement
theorem gcd_3570_4840 : Nat.gcd num1 num2 = 10 := by
  sorry

end gcd_3570_4840_l78_78423


namespace fish_added_l78_78654

theorem fish_added (x : ℕ) (hx : x + (x - 4) = 20) : x - 4 = 8 := by
  sorry

end fish_added_l78_78654


namespace minimum_distance_from_point_to_line_l78_78628

-- Define the function
def f (x : ℝ) : ℝ := Real.exp (2 * x)

-- Define the line
def line (p : ℝ × ℝ) : Prop := p.2 = 2 * p.1

-- Define the distance formula from a point to a line
def distance_to_line (p : ℝ × ℝ) : ℝ := abs (-2 * p.1 + p.2) / Real.sqrt (4 + 1)

-- The main statement
theorem minimum_distance_from_point_to_line :
  ∃ P : ℝ × ℝ, P = (0, 1) ∧ distance_to_line P = Real.sqrt 5 / 5 :=
by
  use (0, 1)
  split
  · rfl
  · sorry

end minimum_distance_from_point_to_line_l78_78628


namespace inradius_length_l78_78730

-- Definitions based on the given conditions
def isosceles_triangle (A B C : ℝ) (triangle : A ≠ B ∧ B = C) := true
def distance (P Q : ℝ) (d: ℝ) := d
def incenter (J : ℝ) := true
def inradius (distance : ℝ) : ℝ := distance
def angle_bisector (J K : ℝ) := true

-- Given conditions
variable {DE EF DF : ℝ}
variable {J D F K : ℝ}

-- Isosceles triangle with given side lengths
axiom h1 : isosceles_triangle DE EF
axiom h2 : distance DF 40
axiom h3 : incenter J
axiom h4 : distance J DF = 25

-- The proof problem statement
theorem inradius_length : inradius (distance J DF) = 15 :=
sorry

end inradius_length_l78_78730


namespace largest_25_supporting_X_l78_78132

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i) ∈ Int → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ ∀ Y : ℝ, (is_25_supporting Y → Y ≤ 0.02) :=
sorry

end largest_25_supporting_X_l78_78132


namespace percentage_difference_l78_78439

variable (p : ℝ) (j : ℝ) (t : ℝ)

def condition_1 := j = 0.75 * p
def condition_2 := t = 0.9375 * p

theorem percentage_difference : (j = 0.75 * p) → (t = 0.9375 * p) → ((t - j) / t * 100 = 20) :=
by
  intros h1 h2
  rw [h1, h2]
  -- This will use the derived steps from the solution, and ultimately show 20
  sorry

end percentage_difference_l78_78439


namespace orthogonal_then_value_acute_angle_then_range_l78_78263

-- Define the vectors a and b
def a (m : ℝ) : ℝ × ℝ := (4, m)
def b : ℝ × ℝ := (2, -1)

-- First proof: If a ⊥ b, then m = 8
theorem orthogonal_then_value (m : ℝ) (h : a m.1 * b.1 + a m.2 * b.2 = 0) : m = 8 := by
  sorry

-- Second proof: If the angle between a and b is acute, then m in (-∞, -2) ∪ (-2, 8)
theorem acute_angle_then_range (m : ℝ) (h : a m.1 * b.1 + a m.2 * b.2 > 0) (h2 : m ≠ -2) : m < 8 := by
  sorry

end orthogonal_then_value_acute_angle_then_range_l78_78263


namespace vector_parallel_S21_l78_78545

-- Define a structure for vectors with real number coordinates
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define addition for Vector2D
instance : Add Vector2D where
  add v₁ v₂ := ⟨v₁.x + v₂.x, v₁.y + v₂.y⟩

-- Define scalar multiplication for Vector2D
instance : HasSmul ℝ Vector2D where
  smul c v := ⟨c * v.x, c * v.y⟩

-- Define vector sequence
def vectorSequence (a₁ d : Vector2D) (n : ℕ) : Vector2D :=
  a₁ + (↑n * d)

-- Define sum of the first n terms of the vector sequence
def Sn (a₁ d : Vector2D) (n : ℕ) : Vector2D :=
  (n + 1) • a₁ + (n * (n + 1) / 2) • d

-- Given conditions
variable (a₁ d : Vector2D)

-- Theorem to prove
theorem vector_parallel_S21 :
  let S21 := Sn a₁ d 20
  let a11 := vectorSequence a₁ d 10
  ∃ c : ℝ, S21 = c • a11 := by
sorry

end vector_parallel_S21_l78_78545


namespace ellipse_a_eq_sqrt_two_b_locus_of_point_D_l78_78925

-- Proof of a = sqrt(2) b
theorem ellipse_a_eq_sqrt_two_b (a b : ℝ) (h : a > b) (e : a > 0) (f : b > 0) :
  (∀ A : ℝ × ℝ, let F₁ := (-a, 0), F₂ := (a, 0), c := sqrt (a^2 - b^2), O := (0, 0) in
    A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
    (A.1 - a)^2 + A.2^2 = c^2 ∧
    dist O (line_through A F₁) = (1 / 3) * dist O F₁) →
  a = sqrt 2 * b := sorry

-- Proof of equation of locus of point D
theorem locus_of_point_D (a b : ℝ) (h : a > b) (e : a > 0) (f : b > 0) :
  (∀ Q₁ Q₂ : ℝ × ℝ, O := (0, 0),
    ∃ D : ℝ × ℝ, Q₁.1^2 / a^2 + Q₂.1^2 / a^2 = 1 ∧
    Q₁.2^2 / b^2 + Q₂.2^2 / b^2 = 1 ∧
    dist O D = (dist Q₁ Q₂) / 2) →
  ∀ D : ℝ × ℝ, D.1^2 + D.2^2 = (2 / 3) * b^2 := sorry

end ellipse_a_eq_sqrt_two_b_locus_of_point_D_l78_78925


namespace part1_part2_1_part2_2_l78_78692

noncomputable def Pascal_distribution (r k : ℕ) (p : ℝ) :=
  (Nat.choose (k-1) (r-1)) * (p^r) * ((1 - p)^(k - r))

theorem part1 (r k : ℕ) (p : ℝ) (h_r : r = 3) (h_p : p = 1/3) (h_k : k = 5) :
  Pascal_distribution r k p = 8 / 81 := 
sorry

theorem part2_1 (r : ℕ) (p : ℝ) (h_r : r = 2) (h_p : p = 1/2) (n : ℕ) (h_n : n ≥ 2) :
  ∑ i in Finset.range (n-1), Pascal_distribution r (i+2) p = 1 - (n+1)/2^n :=
sorry

theorem part2_2 (n : ℕ) (h_n : n ≥ 2) :
  (1 - (n+1)/2^n) ≥ 3/4 ↔ n = 5 :=
sorry

end part1_part2_1_part2_2_l78_78692


namespace base_conversion_and_operations_l78_78027

-- Definitions to convert numbers from bases 7, 5, and 6 to base 10
def base7_to_nat (n : ℕ) : ℕ := 
  8 * 7^0 + 6 * 7^1 + 4 * 7^2 + 2 * 7^3

def base5_to_nat (n : ℕ) : ℕ := 
  1 * 5^0 + 2 * 5^1 + 1 * 5^2

def base6_to_nat (n : ℕ) : ℕ := 
  1 * 6^0 + 5 * 6^1 + 4 * 6^2 + 3 * 6^3

def base7_to_nat2 (n : ℕ) : ℕ := 
  1 * 7^0 + 9 * 7^1 + 8 * 7^2 + 7 * 7^3

-- Problem statement: Perform the arithmetical operations
theorem base_conversion_and_operations : 
  (base7_to_nat 2468 / base5_to_nat 121) - base6_to_nat 3451 + base7_to_nat2 7891 = 2059 := 
by
  sorry

end base_conversion_and_operations_l78_78027


namespace cleaner_used_after_30_minutes_l78_78531

-- Define function to calculate the total amount of cleaner used
def total_cleaner_used (time: ℕ) (rate1: ℕ) (time1: ℕ) (rate2: ℕ) (time2: ℕ) (rate3: ℕ) (time3: ℕ) : ℕ :=
  (rate1 * time1) + (rate2 * time2) + (rate3 * time3)

-- The main theorem statement
theorem cleaner_used_after_30_minutes : total_cleaner_used 30 2 15 3 10 4 5 = 80 := by
  -- insert proof here
  sorry

end cleaner_used_after_30_minutes_l78_78531


namespace area_triangle_equality_l78_78825

theorem area_triangle_equality {ABCD A1B1C1D1 : square} 
  (F : point) (G : point) (O : point)
  (hF : F ∈ line_through (ABCD.A) (A1B1C1D1.D))
  (hG : G ∈ line_through (ABCD.C) (A1B1C1D1.D))
  (hO : O = center ABCD):
  area (triangle_black F G O) = area (triangle_gray1 F G O) + area (triangle_gray2 F G O) := 
sorry

end area_triangle_equality_l78_78825


namespace distance_covered_by_car_Y_during_acceleration_l78_78165

variable (d : ℝ)

def speed_car_X := d / 3

def speed_car_Y (d : ℝ) := 2 * speed_car_X d

-- representing the distance covered by Car Y during the acceleration phase
theorem distance_covered_by_car_Y_during_acceleration 
  (d : ℝ) 
  (H1 : speed_car_X d = d / 3) 
  (H2 : speed_car_Y d = 2 * speed_car_X d) 
  (H3 : d > 0) :
  (d / 3) = (2 * (d / 3)) :=
sorry

end distance_covered_by_car_Y_during_acceleration_l78_78165


namespace area_OPQ_eq_a_sqrt_ab_l78_78726

variable (O F P Q : Point)
variable (a b : ℝ)

def OF := dist O F
def PQ := dist P Q

axiom OF_eq_a : OF = a
axiom PQ_eq_b : PQ = b

theorem area_OPQ_eq_a_sqrt_ab 
  (h : ∀ (P Q : Point), PQ = b → Area(O, P, Q) = a * sqrt(a * b)) : 
  Area O P Q = a * sqrt(a * b) :=
  by sorry

end area_OPQ_eq_a_sqrt_ab_l78_78726


namespace max_distinct_fans_l78_78081

theorem max_distinct_fans : 
  let sectors := 6,
      initial_configs := 2^sectors,
      unchanged_configs := 8,
      unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  in unique_configs = 36 :=
by
  let sectors := 6
  let initial_configs := 2^sectors
  let unchanged_configs := 8
  let unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  have h : unique_configs = 36 := by sorry
  exact h

end max_distinct_fans_l78_78081


namespace alternating_sum_sequence_l78_78911

theorem alternating_sum_sequence :
  let seq := λ (n : ℕ), if even n then 1985 - 10 * n else 1985 - 10 * n
  (finset.sum (finset.range 98) seq) + 20 = 1000 :=
by
  sorry

end alternating_sum_sequence_l78_78911


namespace probability_of_lilii_l78_78412

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutations_with_repetition (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem probability_of_lilii : 
  let total_cards := 5
  let l_count := 2
  let i_count := 3
  let total_permutations := permutations_with_repetition total_cards [l_count, i_count]
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_permutations 
  probability = 0.1 :=
by
  sorry

end probability_of_lilii_l78_78412


namespace find_n_in_range_l78_78200

theorem find_n_in_range (n : ℤ) (h1 : 10 ≤ n ∧ n ≤ 20) (h2 : n ≡ 7882 [MOD 7]) : n = 14 :=
by {
  have h3 : 7882 ≡ 0 [MOD 7], {
    calc 7882 = 7 * 1126 : by norm_num
      ... ≡ 0 [MOD 7] : by rw [Int.mod_eq_zero_of_dvd]; use 1126,
  },
  have h4 : n ≡ 0 [MOD 7], {
    rw h3 at h2,
    exact h2
  },
  have possible_n := [14, 21], -- potential candidates for n modulo 7 in the range
  have h5 : n ∈ possible_n := sorry, -- we need to show n is in the possible_n list
  -- Here we ensure that n is within the list of possibilities
  have h6 : n = 14, {
    cases possible_n with a rest,
    { rw List.mem_singleton at h5, exact h5 },
    { cases rest with b rest', 
      { rw List.mem_cons_iff at h5, cases h5,
        { assumption },
        { exfalso, linarith [h1] } },
      { exfalso, exact List.not_mem_nil n rest' },
    } },
  exact h6
}

end find_n_in_range_l78_78200


namespace extended_fishing_rod_length_l78_78568

def original_length : ℝ := 48
def increase_factor : ℝ := 1.33
def extended_length (orig_len : ℝ) (factor : ℝ) : ℝ := orig_len * factor

theorem extended_fishing_rod_length : extended_length original_length increase_factor = 63.84 :=
  by
    -- proof goes here
    sorry

end extended_fishing_rod_length_l78_78568


namespace cos_alpha_minus_beta_sin_alpha_l78_78647

variable (α β : ℝ)
variable (vector_a : ℝ × ℝ := (Real.cos α, Real.sin α))
variable (vector_b : ℝ × ℝ := (Real.cos β, Real.sin β))
variable h1 : |(Prod.fst vector_a - Prod.fst vector_b, Prod.snd vector_a - Prod.snd vector_b)| = 2 * Real.sqrt 5 / 5
variable h2 : 0 < α ∧ α < Real.pi / 2
variable h3 : -Real.pi / 2 < β ∧ β < 0
variable h4 : Real.sin β = -5 / 13

theorem cos_alpha_minus_beta : Real.cos (α - β) = 3 / 5 := by
  sorry

theorem sin_alpha : Real.sin α = 33 / 65 := by
  sorry

end cos_alpha_minus_beta_sin_alpha_l78_78647


namespace distance_approached_theorem_l78_78422

-- Conditions
def skyscraper_height := 120
def additional_distance := 300
def angle_increase := 45
def initial_elevation_angle := α
def new_elevation_angle := initial_elevation_angle + angle_increase

-- Problem statement
def distance_approached (x : ℝ) : Prop :=
  let initial_tan := (skyscraper_height : ℝ) / (additional_distance + x) in
  let new_tan := (skyscraper_height : ℝ) / x in
  new_tan = (initial_tan + 1) / (1 - initial_tan)

-- Theorem to prove
theorem distance_approached_theorem : ∃ x : ℝ, distance_approached x ∧ x = 60 :=
by
  sorry

end distance_approached_theorem_l78_78422


namespace perfect_square_for_x_l78_78856

def expr (x : ℝ) : ℝ := 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02

theorem perfect_square_for_x : expr 0.04 = (11.98 + 0.02) ^ 2 :=
by
  sorry

end perfect_square_for_x_l78_78856


namespace surface_area_circumscribed_sphere_l78_78966

-- Definitions of the conditions
def SA_perp_plane_ABC : Prop := sorry
def AB_perp_AC : Prop := sorry
def SA : ℝ := 3
def AB : ℝ := 2
def AC : ℝ := 2

-- Theorem statement
theorem surface_area_circumscribed_sphere 
  (h1 : SA_perp_plane_ABC) 
  (h2 : AB_perp_AC) 
  (h3 : SA = 3) 
  (h4 : AB = 2) 
  (h5 : AC = 2) : 
  4 * real.pi * (1 / 4 * (AB^2 + AC^2 + SA^2)) = 17 * real.pi := 
by sorry

end surface_area_circumscribed_sphere_l78_78966


namespace sum_log_is_eq_l78_78168

noncomputable def log_base (b x : ℝ) := log x / log b

noncomputable def summand (k : ℕ) : ℝ := log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3

theorem sum_log_is_eq :
  (∑ k in finset.range 48 \ finset.range 2, summand (k + 3)) = 1 - (1 / log_base 3 51) :=
sorry

end sum_log_is_eq_l78_78168


namespace max_snowmen_l78_78487

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78487


namespace circumcenters_on_perpendiculars_l78_78873

-- Definitions based on given conditions
variables {A B C D O M O₁ O₂ : Type}

-- Assume trapezoid properties and intersection conditions
variables (cyclic_trapezoid: cyclic_trapezoid A B C D)
          (circumcenter_O: circumscribed_circle O A B C D)
          (intersection_M: intersection M A C B D)
          (circumcenter_O₁: circumcenter O₁ A M D)
          (circumcenter_O₂: circumcenter O₂ B M C)

-- The theorem to be proved
theorem circumcenters_on_perpendiculars:
  ∃ O₁ O₂ M O,
    cyclic_trapezoid A B C D ∧
    circumscribed_circle O A B C D ∧
    intersection M A C B D ∧
    circumcenter O₁ A M D ∧
    circumcenter O₂ B M C →
    perpendicular M O₁ A D ∧ perpendicular M O₂ B C :=
sorry

end circumcenters_on_perpendiculars_l78_78873


namespace smallest_n_inequality_l78_78943

theorem smallest_n_inequality :
  ∃ (n : ℕ), ∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧ n = 4 :=
by
  -- Proof steps would go here
  sorry

end smallest_n_inequality_l78_78943


namespace sum_of_number_and_square_is_306_l78_78438

theorem sum_of_number_and_square_is_306 : ∃ x : ℤ, x + x^2 = 306 ∧ x = 17 :=
by
  sorry

end sum_of_number_and_square_is_306_l78_78438


namespace avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l78_78374

variable (c d : ℤ)
variable (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7 :
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7) / 7 = c + 7 :=
by
  sorry

end avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l78_78374


namespace find_triples_l78_78947

theorem find_triples (m n x y : ℕ) (hm_pos : 0 < m) (hcoprime : Nat.coprime m n) :
  ((x^2 + y^2)^m = (xy)^n) ↔
  (∃ k : ℕ, m = 2 * k ∧ n = m + 1 ∧ x = 2^k ∧ y = 2^k) :=
begin
  sorry
end

end find_triples_l78_78947


namespace solve_for_r_l78_78343

variable (a r : ℝ) 
variable (FA OG AG QC OQ OA OC : ℝ)

-- Define the conditions from the problem
noncomputable def problem_conditions := 
  (FA = a) ∧ 
  (OG = r) ∧ 
  (AG = a) ∧ 
  (QC = 4 - r) ∧ 
  (OQ = 4 + r) ∧ 
  (OA = 8 - r) ∧ 
  (OC = a)

-- Define the system of equations from the problem
noncomputable def system_of_equations :=
  ((4 - r)^2 + a^2 = (4 + r)^2) ∧ 
  (r^2 + a^2 = (8 - r)^2)

-- The theorem to be proved
theorem solve_for_r (h₁ : problem_conditions) (h₂ : system_of_equations) : r = 2 :=
  sorry

end solve_for_r_l78_78343


namespace no_square_in_arithmetic_progression_l78_78432

theorem no_square_in_arithmetic_progression 
: ∀ (n : ℕ), ¬ ∃ (k : ℕ), 3 * k - 1 = n^2  := 
by
  intro n
  intro h
  obtain ⟨k, hk⟩ := h
  have h_div := congr_fun (congr_arg (λ x, x % 3) hk) 0
  have h3k := congr_fun (congr_arg (λ x, x % 3) (id rfl : 3 * k % 3 = 0 % 3)) 0
  rw [hk, Nat.add_mod, Nat.mul_mod, Nat.mod_mod] at h_div
  norm_num at h_div
  have hsq : n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2 := by exact Nat.mod_lt n (by norm_num)
  cases n % 3 with h_mod; { cases h_mod } <;> norm_num at h_div h_mod

end no_square_in_arithmetic_progression_l78_78432


namespace product_of_roots_l78_78578

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 2)
noncomputable def f_inv_condition (b : ℝ) : ℝ := f b 3 = f b⁻¹ (b + 2)
noncomputable def equation_roots (b : ℝ) : Prop := 3 * b^2 - 15 * b - 28 = 0

theorem product_of_roots 
  (b : ℝ) 
  (h1 : f_inv_condition b) 
  (h2 : equation_roots b) : (∃ b1 b2 : ℝ, (b1 * b2 = -28 / 3 ∧ equation_roots b1 ∧ equation_roots b2)) := sorry

end product_of_roots_l78_78578


namespace max_snowmen_l78_78505

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78505


namespace max_snowmen_l78_78499

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78499


namespace max_distinct_fans_l78_78083

theorem max_distinct_fans : 
  let sectors := 6,
      initial_configs := 2^sectors,
      unchanged_configs := 8,
      unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  in unique_configs = 36 :=
by
  let sectors := 6
  let initial_configs := 2^sectors
  let unchanged_configs := 8
  let unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  have h : unique_configs = 36 := by sorry
  exact h

end max_distinct_fans_l78_78083


namespace even_function_behavior_l78_78975

variable (f : ℝ → ℝ)

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)

theorem even_function_behavior 
  (h_even : is_even_function f)
  (h_pos : ∀ x : ℝ, 0 < x → f x = 10^x)
  (x : ℝ) 
  (h_neg : x < 0) :
  f x = (1 / 10) ^ x := 
sorry

end even_function_behavior_l78_78975


namespace juju_juice_bar_l78_78117

theorem juju_juice_bar (M P : ℕ) 
  (h₁ : 6 * P = 54)
  (h₂ : 5 * M + 6 * P = 94) : 
  M + P = 17 := 
sorry

end juju_juice_bar_l78_78117


namespace gun_fan_image_equivalence_l78_78465

def gunPiercingImage : String := "point moving to form a line"
def foldingFanImage : String := "line moving to form a surface"

theorem gun_fan_image_equivalence :
  (gunPiercingImage = "point moving to form a line") ∧ 
  (foldingFanImage = "line moving to form a surface") := by
  -- Proof goes here
  sorry

end gun_fan_image_equivalence_l78_78465


namespace solve_infinite_power_tower_eq4_l78_78776

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  x ^ infinite_power_tower x

theorem solve_infinite_power_tower_eq4 :
  ∀ x : ℝ, infinite_power_tower x = 4 → x = real.sqrt 2 := sorry

end solve_infinite_power_tower_eq4_l78_78776


namespace find_radius_of_ball_l78_78451

noncomputable def radius_of_ball : ℝ :=
  let r : ℝ := 16.25 in r

theorem find_radius_of_ball :
  ∀ (r₀ d₀ : ℝ), 
  r₀ = 15 → d₀ = 10 →
  radius_of_ball = 16.25 :=
by
  intros r₀ d₀ hr₀ hd₀
  unfold radius_of_ball
  sorry

end find_radius_of_ball_l78_78451


namespace complex_numbers_with_imaginary_part_l78_78373

theorem complex_numbers_with_imaginary_part : 
  let s := {0, 1, 2, 3, 4, 5, 6}
  in (∑ b in s \ {0}, ∑ a in s \ {b}, 1) = 36 := by
  sorry

end complex_numbers_with_imaginary_part_l78_78373


namespace sum_a1_to_a19_and_b1_to_b88_l78_78442

variable {n : Nat} {m : Nat} {k : Nat}
variable {a : Nat → Nat} {b : Nat → Nat}

-- Conditions
axiom a_seq_monotonic : ∀ n, a n ≤ a (n+1)
axiom a_positive : ∀ n, 1 ≤ a n
axiom a_19_eq_88 : a 19 = 88
axiom b_def : ∀ m, b m = Nat.find (λ n, a n ≥ m)

-- Proof statement
theorem sum_a1_to_a19_and_b1_to_b88 :
  (∑ i in Finset.range 20, a i)  + (∑ i in Finset.range 89, b i) = 1760 :=
sorry

end sum_a1_to_a19_and_b1_to_b88_l78_78442


namespace correct_option_D_l78_78845

theorem correct_option_D (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by sorry

end correct_option_D_l78_78845


namespace triangle_integer_sides_l78_78408

theorem triangle_integer_sides (A B C : ℝ) (a b c r : ℝ) 
    (h : cot (A / 2) ^ 2 + 4 * cot (B / 2) ^ 2 + 9 * cot (C / 2) ^ 2 = (9 * (a + b + c) ^ 2) / (49 * r ^ 2)) 
    (ha : a = |BC|) (hb : b = |CA|) (hc : c = |AB|):
  ∃ (k : ℝ), (a, b, c) = (13 * k, 40 * k, 45 * k) :=
by
  sorry

end triangle_integer_sides_l78_78408


namespace edge_ratio_of_cubes_l78_78855

theorem edge_ratio_of_cubes (a b : ℝ) (h : (a^3) / (b^3) = 64) : a / b = 4 :=
sorry

end edge_ratio_of_cubes_l78_78855


namespace harvested_tomatoes_l78_78347

-- Problem definitions based on the conditions
variables (kg_sold_maxwell : ℝ) (kg_sold_wilson : ℝ) (kg_not_sold : ℝ)
variable (total_kg_harvested : ℝ)

-- Given conditions
def sold_to_maxwell := kg_sold_maxwell = 125.5
def sold_to_wilson := kg_sold_wilson = 78
def not_sold := kg_not_sold = 42

-- Statement to prove
theorem harvested_tomatoes (h1 : sold_to_maxwell) (h2 : sold_to_wilson) (h3 : not_sold) :
  total_kg_harvested = 245.5 :=
by
  sorry

end harvested_tomatoes_l78_78347


namespace average_visitors_per_day_l78_78436

theorem average_visitors_per_day 
  (avg_sunday : ℕ) (avg_other_days : ℕ) 
  (days_in_month : ℕ) (sundays : ℕ) (other_days : ℕ) 
  (month_begins_sunday : Bool)
  (num_sundays : days_in_month // 7 * (days_in_month % 7 ≠ 0) = sundays)
  (num_other_days : days_in_month - num_sundays = other_days) :
  days_in_month = 30 → month_begins_sunday = true → avg_sunday = 150 → avg_other_days = 120 → 
  (avg_sunday * sundays + avg_other_days * other_days) / days_in_month = 124 :=
by
  intros
  sorry

end average_visitors_per_day_l78_78436


namespace trig_expression_l78_78944

open Real

noncomputable def tan (x : ℝ) : ℝ := sin x / cos x

theorem trig_expression :
  (tan (7.5 * π / 180) * tan (15 * π / 180) / (tan (15 * π / 180) - tan (7.5 * π / 180))
  + sqrt 3 * (sin (7.5 * π / 180) ^ 2 - cos (7.5 * π / 180) ^ 2)) = -sqrt 2 := by
  sorry

end trig_expression_l78_78944


namespace product_of_two_special_numbers_is_perfect_square_l78_78965

-- Define the structure of the required natural numbers
structure SpecialNumber where
  m : ℕ
  n : ℕ
  value : ℕ := 2^m * 3^n

-- The main theorem to be proved
theorem product_of_two_special_numbers_is_perfect_square :
  ∀ (a b c d e : SpecialNumber),
  ∃ x y : SpecialNumber, ∃ k : ℕ, (x.value * y.value) = k * k :=
by
  sorry

end product_of_two_special_numbers_is_perfect_square_l78_78965


namespace shorter_piece_length_l78_78867

theorem shorter_piece_length (total_len : ℝ) (ratio : ℝ) (shorter_len : ℝ) (longer_len : ℝ) 
  (h1 : total_len = 49) (h2 : ratio = 2/5) (h3 : shorter_len = x) 
  (h4 : longer_len = (5/2) * x) (h5 : shorter_len + longer_len = total_len) : 
  shorter_len = 14 := 
by
  sorry

end shorter_piece_length_l78_78867


namespace father_age_is_40_l78_78851

def man_age (F : ℝ) : ℝ := (2/5) * F
def man_age_in_future (M F : ℝ) : ℝ := M + 8
def father_age_in_future (F : ℝ) : ℝ := F + 8

theorem father_age_is_40 (F : ℝ) (M : ℝ) 
    (h1 : M = man_age F) 
    (h2 : man_age_in_future M F = (1/2) * father_age_in_future F) : 
    F = 40 :=
by
    sorry

end father_age_is_40_l78_78851


namespace banana_bread_ratio_l78_78935

theorem banana_bread_ratio (bananas_per_loaf : ℕ) (loaves_monday : ℕ) (total_bananas : ℕ) (loaves_made_tuesday : ℕ) : 
    bananas_per_loaf = 4 → 
    loaves_monday = 3 → 
    total_bananas = 36 → 
    loaves_made_tuesday = (total_bananas - loaves_monday * bananas_per_loaf) / bananas_per_loaf → 
    (loaves_made_tuesday : loaves_monday = 2 : 1) :=
by
  intros
  sorry

end banana_bread_ratio_l78_78935


namespace value_of_k_l78_78574

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k * x - 8

theorem value_of_k:
  (f 5) - (g 5 k) = 20 → k = -10.8 :=
by
  sorry

end value_of_k_l78_78574


namespace David_subtracts_79_l78_78820

theorem David_subtracts_79 (y : ℤ) : ∀ (x : ℤ), x = y - 1 → x^2 = y^2 - 79 :=
by
  intro x
  intro h
  rw [h]
  calc (y - 1)^2 = y^2 - 2 * y * 1 + 1^2 : by ring
  ... = y^2 - 2 * y + 1 : by ring
  ... = y^2 - 79 : by sorry

end David_subtracts_79_l78_78820


namespace find_x_value_l78_78211

theorem find_x_value (x : ℚ) (h : 5 * (x - 10) = 3 * (3 - 3 * x) + 9) : x = 34 / 7 := by
  sorry

end find_x_value_l78_78211


namespace oliver_money_left_l78_78360

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l78_78360


namespace vector_properties_l78_78642

noncomputable theory

open_locale real_inner_product_space

variables (a b c : ℝ × ℝ) (t : ℝ)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2)
variables (hab : ‖a - b‖ = real.sqrt 7)
variables (h_perp : inner_product_space.has_inner.smul_right_inner_product a c = 0)

-- Define the main theorem
theorem vector_properties :
  ∃ θ : ℝ, ∃ t : ℝ, (inner_product_space.cosine a b = -1 / 2) ∧ 
  (θ = 2 * real.pi / 3) ∧ (t = 1) ∧ (‖t • a + b‖ = real.sqrt 3) :=
by {
  sorry -- proof is omitted
}

end vector_properties_l78_78642


namespace parallel_to_a_perpendicular_to_a_l78_78218

variable (x y : ℝ)

def vector_a := (3 : ℝ, 4 : ℝ)
def vector_b := (x, y)

def is_unit_vector (v : ℝ × ℝ) := (v.1 ^ 2 + v.2 ^ 2 = 1)

-- Part 1
theorem parallel_to_a (h1 : is_unit_vector vector_b) (h2 : 3 * y - 4 * x = 0) : 
  vector_b = (3/5, 4/5) ∨ vector_b = (-3/5, -4/5) := 
sorry

-- Part 2
theorem perpendicular_to_a (h1 : is_unit_vector vector_b) (h3 : 3 * x + 4 * y = 0) : 
  vector_b = (-4/5, 3/5) ∨ vector_b = (4/5, -3/5) := 
sorry

end parallel_to_a_perpendicular_to_a_l78_78218


namespace part_1_simplify_part_2_extrema_l78_78993

noncomputable def f (x : ℝ) : ℝ :=
  sin (2 * x + π / 3) + sin (2 * x - π / 3) + 2 * cos x ^ 2 - 1

theorem part_1_simplify :
  ∀ x : ℝ, f x = sqrt 2 * sin (2 * x + π / 4) :=
sorry

theorem part_2_extrema :
  ∀ x : ℝ, x ∈ Icc (-π / 4) (π / 4) → 
    (∀ {b}, f b ≤ sqrt 2) ∧ (∀ {a}, -1 ≤ f a) :=
sorry

end part_1_simplify_part_2_extrema_l78_78993


namespace find_C_coordinates_l78_78541

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -3, y := 5 }
def B : Point := { x := 9, y := -1 }
def C : Point := { x := 15, y := -4 }

noncomputable def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

theorem find_C_coordinates :
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  C = C_actual :=
by
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  show C = C_actual
  rfl

end find_C_coordinates_l78_78541


namespace club_joining_ways_l78_78213

-- Definitions based on the conditions
variable (students : Fin 5) -- Represents 5 students
variable (clubs : Fin 4) -- Represents 4 clubs
def joined_club : students → clubs -- Function representing which club each student joins

-- Conditions
axiom at_least_one_participant_each_club :
  ∀ c : clubs, ∃ s : students, joined_club s = c

axiom each_student_one_club :
  ∀ s1 s2 : students, s1 ≠ s2 → joined_club s1 ≠ joined_club s2

axiom student_a_not_street_dance :
  joined_club 0 ≠ 1 -- Assuming student A is represented by 0 and "Street Dance Club" is represented by 1

-- Theorem to prove
theorem club_joining_ways : 
  ∃ ways : ℕ, ways = 180 := sorry

end club_joining_ways_l78_78213


namespace cat_food_per_day_l78_78749

theorem cat_food_per_day
  (bowl_empty_weight : ℕ)
  (bowl_weight_after_eating : ℕ)
  (food_eaten : ℕ)
  (days_per_fill : ℕ)
  (daily_food : ℕ) :
  (bowl_empty_weight = 420) →
  (bowl_weight_after_eating = 586) →
  (food_eaten = 14) →
  (days_per_fill = 3) →
  (bowl_weight_after_eating - bowl_empty_weight + food_eaten = days_per_fill * daily_food) →
  daily_food = 60 :=
by
  sorry

end cat_food_per_day_l78_78749


namespace domain_of_function_l78_78590

noncomputable def domain (f : ℝ → ℝ) : set ℝ := {x | ∃ y, f x = y}

def g : ℝ → ℝ := λ x, x^3 - 4*x^2 + 4*x - 4

theorem domain_of_function :
  domain (λ x, (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / g x) = {x | x ≠ 2} :=
by
  sorry

end domain_of_function_l78_78590


namespace max_snowmen_l78_78506

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78506


namespace find_a_l78_78277

theorem find_a (a : ℝ) (h : Complex.pure_imaginary (Complex.mk (a^2 - 3*a + 2) (a - 1))) : a = 2 :=
sorry

end find_a_l78_78277


namespace max_fans_theorem_l78_78074

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l78_78074


namespace largest_25_supporting_X_l78_78135

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i) ∈ Int → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ ∀ Y : ℝ, (is_25_supporting Y → Y ≤ 0.02) :=
sorry

end largest_25_supporting_X_l78_78135


namespace area_of_equilateral_triangle_with_inscribed_circle_radius_l78_78688

noncomputable def radius := 3 -- given radius of the inscribed circle
noncomputable def equilateral_triangle_area (r : ℝ) : ℝ :=
  let base := 6 * real.sqrt 3 * r in       -- BC length
  let height := 3 * real.sqrt 3 in             -- AX length
  (1 / 2) * base * height                     -- Area of triangle ABC

theorem area_of_equilateral_triangle_with_inscribed_circle_radius :
  equilateral_triangle_area radius = 27 * real.sqrt 3 :=
by
  sorry

end area_of_equilateral_triangle_with_inscribed_circle_radius_l78_78688


namespace point_in_fourth_quadrant_l78_78695

theorem point_in_fourth_quadrant (x : ℝ) (hx : P (2 * x - 6) (x - 5)) : 3 < x ∧ x < 5 :=
by {
  sorry
}

end point_in_fourth_quadrant_l78_78695


namespace max_blue_cells_n2_max_blue_cells_n15_l78_78772

theorem max_blue_cells_n2 :
  ∀ (row col : Fin 30), 
    (∀ (i j : Fin 30), ∃ (color : Fin 2), 
      (∀ (k : Fin 30), ∃ (c : Fin 2), c ∈ row ∧ c ∈ col)) →
    (∃ (blue_count : Nat), blue_count ≤ 450) :=
sorry

theorem max_blue_cells_n15 :
  ∀ (row col : Fin 30), 
    (∀ (i j : Fin 30), ∃ (color : Fin 15), 
      (∀ (k : Fin 30), ∃ (c : Fin 15), c ∈ row ∧ c ∈ col)) →
    (∃ (blue_count : Nat), blue_count ≤ 400) :=
sorry

end max_blue_cells_n2_max_blue_cells_n15_l78_78772


namespace cost_of_used_cd_l78_78319

theorem cost_of_used_cd (N U : ℝ) 
    (h1 : 6 * N + 2 * U = 127.92) 
    (h2 : 3 * N + 8 * U = 133.89) :
    U = 9.99 :=
by 
  sorry

end cost_of_used_cd_l78_78319


namespace Donovan_percentage_correct_l78_78187

-- Definitions based on conditions from part a)
def fullyCorrectAnswers : ℕ := 35
def incorrectAnswers : ℕ := 13
def partiallyCorrectAnswers : ℕ := 7
def pointPerFullAnswer : ℝ := 1
def pointPerPartialAnswer : ℝ := 0.5

-- Lean 4 statement to prove the problem mathematically
theorem Donovan_percentage_correct : 
  (fullyCorrectAnswers * pointPerFullAnswer + partiallyCorrectAnswers * pointPerPartialAnswer) / 
  (fullyCorrectAnswers + incorrectAnswers + partiallyCorrectAnswers) * 100 = 70.00 :=
by
  sorry

end Donovan_percentage_correct_l78_78187


namespace least_number_subtracted_l78_78063

theorem least_number_subtracted (x : ℕ) (y : ℕ) (h : 2590 - x = y) : 
  y % 9 = 6 ∧ y % 11 = 6 ∧ y % 13 = 6 → x = 10 := 
by
  sorry

end least_number_subtracted_l78_78063


namespace common_points_intervals_l78_78391

noncomputable def h (x : ℝ) : ℝ := (2 * Real.log x) / x

theorem common_points_intervals (a : ℝ) (h₀ : 1 < a) : 
  (∀ f g : ℝ → ℝ, (f x = a ^ x) → (g x = x ^ 2) → 
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃) → 
  a < Real.exp (2 / Real.exp 1) :=
by
  sorry

end common_points_intervals_l78_78391


namespace T_zero_implies_A_zero_rank_preserved_l78_78859

namespace MatrixDetRank

open Matrix

variable (n : ℕ) [fact (0 < n)] [decidable_eq n] [fintype n]

def MnC := matrix (fin n) (fin n) ℂ

-- Given conditions
variable (T : MnC n → MnC n)
variable (hT_det : ∀ A : MnC n, det A = det (T A))

-- Part (1)
theorem T_zero_implies_A_zero (A : MnC n) (hT : T A = 0) : A = 0 := by
  sorry

-- Part (2)
theorem rank_preserved (A : MnC n) : rank A = rank (T A) := by
  sorry

end MatrixDetRank

end T_zero_implies_A_zero_rank_preserved_l78_78859


namespace simplify_sqrt_7_6_simplify_sqrt_n_sum_series_l78_78841

-- Proof for the first equivalence
theorem simplify_sqrt_7_6 : (2 : ℝ) / (Real.sqrt 7 + Real.sqrt 6) = 2 * Real.sqrt 7 - 2 * Real.sqrt 6 :=
by
  sorry

-- Proof for the second equivalence, with n being a positive integer
theorem simplify_sqrt_n (n : ℕ) (h : 0 < n) : 
  (1 : ℝ) / (Real.sqrt (n + 1) + Real.sqrt n) = Real.sqrt (n + 1) - Real.sqrt n :=
by
  sorry

-- Proof for the sum of the series
theorem sum_series : 
  ∑ k in Finset.range 2022, (1 : ℝ) / (Real.sqrt (k + 1) + Real.sqrt (k + 2)) = 17 * Real.sqrt 7 - 1 :=
by
  sorry

end simplify_sqrt_7_6_simplify_sqrt_n_sum_series_l78_78841


namespace correct_equation_l78_78843

theorem correct_equation (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by
  sorry

end correct_equation_l78_78843


namespace minimum_rubles_cost_l78_78420

open Nat

def rubles : Set Nat := {1, 2, 5, 10}

def is_valid_representation (expr : Nat) (cost : Nat) : Prop :=
  ∃ (rep : List Nat), (∀ r ∈ rep, r ∈ rubles) ∧ (rep.sum = cost) ∧ (expr = 2009)

theorem minimum_rubles_cost : Exists (λ cost, is_valid_representation 2009 cost)  ∧ ∀ cost' < 23, ¬ is_valid_representation 2009 cost' :=
by
  sorry

end minimum_rubles_cost_l78_78420


namespace rowing_upstream_distance_l78_78528

theorem rowing_upstream_distance 
  (b s t d1 d2 : ℝ)
  (h1 : s = 7)
  (h2 : d1 = 72)
  (h3 : t = 3)
  (h4 : d1 = (b + s) * t) :
  d2 = (b - s) * t → d2 = 30 :=
by 
  intros h5
  sorry

end rowing_upstream_distance_l78_78528


namespace cyclist_waits_15_minutes_l78_78467

-- Definitions
def hiker_rate := 7 -- miles per hour
def cyclist_rate := 28 -- miles per hour
def wait_time := 15 / 60 -- hours, as the cyclist waits 15 minutes, converted to hours

-- The statement to be proven
theorem cyclist_waits_15_minutes :
  ∃ t : ℝ, t = 15 / 60 ∧
  (∀ d : ℝ, d = (hiker_rate * wait_time) →
            d = (cyclist_rate * t - hiker_rate * t)) :=
by
  sorry

end cyclist_waits_15_minutes_l78_78467


namespace parallelogram_not_axially_symmetric_l78_78798

def is_axially_symmetric (shape : Type) : Prop :=
  ∃ (axis : shape → shape → Prop), ∀ (s1 s2 : shape), axis s1 s2 → s1 = s2

inductive Shape
  | Rectangle
  | IsoscelesTrapezoid
  | Parallelogram
  | EquilateralTriangle

open Shape

theorem parallelogram_not_axially_symmetric :
  ¬ is_axially_symmetric Parallelogram :=
sorry

end parallelogram_not_axially_symmetric_l78_78798


namespace largest_positive_integer_l78_78665

def binary_operation (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer (n : Int) : (∀ m : Int, m > 0 → n - (n * 5) < -19 → m ≤ n) 
  ↔ n = 5 := 
by
  sorry

end largest_positive_integer_l78_78665


namespace max_snowmen_l78_78480

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78480


namespace division_problem_l78_78046

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l78_78046


namespace three_digit_reverse_squares_l78_78398

def reverse_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d => acc * 10 + d) 0

theorem three_digit_reverse_squares :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧
          10000 ≤ n^2 ∧ n^2 < 100000 ∧
          10000 ≤ (reverse_number n)^2 ∧ (reverse_number n)^2 < 100000 ∧
          reverse_number (n^2) = (reverse_number n)^2}.

end three_digit_reverse_squares_l78_78398


namespace find_f_neg5_l78_78997

-- Define the function f
def f (x : ℝ) (b : ℝ) : ℝ := Real.asin (3 * x) + b * Real.tan x + 1

-- Given conditions
variables (b : ℝ) (h1 : f 5 b = 7)

-- Goal
theorem find_f_neg5 : f (-5) b = -5 := by
  -- Sorry to skip the proof
  sorry

end find_f_neg5_l78_78997


namespace incorrect_option_C_l78_78960

variable {a : ℕ → ℝ}  -- Define a sequence a_n
variable h : a 6 + a 10 = 20  -- Given condition

theorem incorrect_option_C : ¬ (a 16 = 20) :=
sorry

end incorrect_option_C_l78_78960


namespace hyperbola_standard_equation_l78_78809

noncomputable def hyperbola_equation (a b : ℝ) := (∃ foci_left_right_length : ℝ, foci_left_right_length = 8 ∧ a = 2 ∧ b = 2 ∧ (x, y : ℝ)) →
  ∃ (c : ℝ), (c = 4) →
  ∃ (b_val : ℝ), b_val = real.sqrt((c^2) - (a^2)) →

theorem hyperbola_standard_equation (a b : ℝ) (c : ℝ) (b_val : ℝ) :
  (
    ∃ (a_ : ℝ) (b_ : ℝ) (hyper_eq : ℝ → ℝ → Prop), 
    -a_ = a ∧ c = 4 ∧ b_ = b ∧ b_val = real.sqrt (c^2 - a^2) ∧ hyper_eq = λ x y, (x^2 / 4) - (y^2 / 12) = 1 ->
    hyper_eq x y = (x^2 / 4) - (y^2 / 12) = 1
  ) :=
sorry

end hyperbola_standard_equation_l78_78809


namespace rationalize_denominator_l78_78371

theorem rationalize_denominator (h : Real.sqrt 200 = 10 * Real.sqrt 2) : 
  (7 / Real.sqrt 200) = (7 * Real.sqrt 2 / 20) :=
by
  sorry

end rationalize_denominator_l78_78371


namespace smallest_positive_period_and_intervals_of_decrease_max_and_min_values_on_interval_l78_78636

noncomputable def f (x : ℝ) := cos (2 * x - π / 6) * sin (2 * x) - 1 / 4

theorem smallest_positive_period_and_intervals_of_decrease :
  (∀ x, f (x + π / 2) = f x) ∧
  (∀ k : ℤ, ∀ x, x ∈ Icc (π / 6 + k * (π / 2)) (5 * π / 12 + k * (π / 2)) → 
    strictMonoDecOn (f) (Icc (π / 6 + k * (π / 2)) (5 * π / 12 + k * (π / 2)))) :=
sorry

theorem max_and_min_values_on_interval :
  ( ∃ x ∈ Icc (-π / 4) (0 : ℝ), f x = 1 / 4) ∧ 
  ( ∃ y ∈ Icc (-π / 4) (0 : ℝ), f y = -1 / 2) :=
sorry

end smallest_positive_period_and_intervals_of_decrease_max_and_min_values_on_interval_l78_78636


namespace ratio_of_areas_in_rectangle_l78_78323

variables {A B C D K L M : Type}
variables [has_measurable_area A] [has_measurable_area B] [has_measurable_area C] [has_measurable_area D]
variables [has_measurable_area K] [has_measurable_area L] [has_measurable_area M] 

theorem ratio_of_areas_in_rectangle (AB CD AK CL ABKM ABCL : Set (A × A))  [rectangle AB CD]
  (hAB : measure_of AB = a) (hBC : measure_of BC = b)
  (h_midpoints : midpoint BC = K) (h_midpoints : midpoint DA = L)
  (h_perpendicular : ∀ x ∈ A, perpendicular (B, AK) (x, CL) → x = M)
  (hABKM : measure_of ABKM = (measure_of AB / 2))
  (hABCL : measure_of ABCL = (3 * measure_of AB / 4)) : 
  (measure_of ABKM / measure_of ABCL = 2 / 3) :=
begin
  sorry
end

end ratio_of_areas_in_rectangle_l78_78323


namespace calculate_expression_l78_78163

open Real

theorem calculate_expression :
  (cbrt 27) * (sqrt 81 ^ (1/4)) * (sqrt 9) = 27 := by
  sorry

end calculate_expression_l78_78163


namespace relationship_a_b_l78_78228

noncomputable def f : ℝ → ℝ := sorry  -- f is a differentiable function
variable {x : ℝ}

-- f'(x) + f(x) < 0
axiom diff_function (x : ℝ) : deriv f x + f x < 0

-- Definition of a and b
def a (m : ℝ) : ℝ := f (m - m^2)
def b (m : ℝ) : ℝ := exp (m^2 - m + 1) * f 1

theorem relationship_a_b (m : ℝ) : a m > b m :=
by
  sorry

end relationship_a_b_l78_78228


namespace find_prob_normal_distribution_l78_78330

noncomputable def normal_prob (μ σ x₁ x₂ : ℝ) := 
  1 / (σ * Math.sqrt (2 * Math.pi)) * ∫ t in x₁..x₂, Math.exp (-((t - μ) ^ 2) / (2 * σ ^ 2))

theorem find_prob_normal_distribution : 
  normal_prob 5 1 6 7 = 0.1359 := 
by
  sorry

end find_prob_normal_distribution_l78_78330


namespace students_helped_on_third_day_l78_78887

theorem students_helped_on_third_day (books_total : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day4 : ℕ) (books_day3 : ℕ) :
  books_total = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day4 = 9 →
  books_day3 = books_total - ((students_day1 + students_day2 + students_day4) * books_per_student) →
  books_day3 / books_per_student = 6 :=
by
  sorry

end students_helped_on_third_day_l78_78887


namespace probability_at_least_7_consecutive_heads_l78_78113

theorem probability_at_least_7_consecutive_heads :
  ∑ (i = 1 to 4), probability (exactly_7_heads i) + ∑ (i = 1 to 3), probability (exactly_8_heads i) + ∑ (i = 1 to 2), probability (exactly_9_heads i) + probability (exactly_10_heads) = 10 / 1024 :=
sorry

end probability_at_least_7_consecutive_heads_l78_78113


namespace cubical_tank_water_volume_l78_78461

theorem cubical_tank_water_volume 
    (s : ℝ) -- side length of the cube in feet
    (h_fill : 1 / 4 * s = 1) -- tank is filled to 0.25 of its capacity, water level is 1 foot
    (h_volume_water : 0.25 * (s ^ 3) = 16) -- 0.25 of the tank's total volume is the volume of water
    : s ^ 3 = 64 := 
by
  sorry

end cubical_tank_water_volume_l78_78461


namespace area_of_square_abcd_l78_78901

theorem area_of_square_abcd :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 2)
  let D := (0, 2)
  let F := (2, 1)
  (B.1 - F.1)^2 + (B.2 - F.2)^2 = 1^2 →   -- BF = 1
  F.1 = B.1 →                              -- F on AB (F.x = B.x)
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - F.1)^2 + (C.2 - F.2)^2 + (B.1 - F.1)^2 + (B.2 - F.2)^2 → 
  (C.2 - B.2 = C.1 - F.1) →                 -- right triangle
  let side_length := 2
  side_length^2 = 4 :=
by {
  intros,
  sorry
}

end area_of_square_abcd_l78_78901


namespace angle_OAB_is_18_degrees_l78_78143

/-- Given a regular 10-sided polygon inscribed in a circle with center O,
and A and B as adjacent vertices of the polygon, ∠OAB is equal to 18°. -/
theorem angle_OAB_is_18_degrees
  (O A B : Point)
  (h : is_center_of_circle O)
  (h1 : is_vertex_of_regular_polygon A 10)
  (h2 : is_vertex_of_regular_polygon B 10)
  (h3 : adjacent_vertices A B)
  (h4 : inscribed_in_circle_of_center O A B):
  ∠ O A B = 18 :=
sorry

end angle_OAB_is_18_degrees_l78_78143


namespace part1_solution_part2_solution_part3_solution_l78_78916

-- Define the basic conditions
variables (x y m : ℕ)

-- Part 1: Number of pieces of each type purchased (Proof for 10 pieces of A, 20 pieces of B)
theorem part1_solution (h1 : x + y = 30) (h2 : 28 * x + 22 * y = 720) :
  (x = 10) ∧ (y = 20) :=
sorry

-- Part 2: Maximize sales profit for the second purchase
theorem part2_solution (h1 : 28 * m + 22 * (80 - m) ≤ 2000) :
  m = 40 ∧ (max_profit = 1040) :=
sorry

-- Variables for Part 3
variables (a : ℕ)
-- Profit equation for type B apples with adjusted selling price
theorem part3_solution (h : (4 + 2 * a) * (34 - a - 22) = 90) :
  (a = 7) ∧ (selling_price = 27) :=
sorry

end part1_solution_part2_solution_part3_solution_l78_78916


namespace evaluate_expression_l78_78938

-- Define the different parts of the expression
def log3_half_pow_half : ℝ := (Real.logb 3 (3 ^ (1 / 2))) ^ 2
def log_quarter : ℝ := Real.logb 0.25 (1 / 4)
def nine_log5_sqrt5 : ℝ := 9 * Real.logb 5 (sqrt 5)
def log_sqrt3_one : ℝ := Real.logb (sqrt 3) 1

-- Combine them into the whole expression
def expression := log3_half_pow_half + log_quarter + nine_log5_sqrt5 - log_sqrt3_one

-- The proof statement
theorem evaluate_expression : expression = 5.75 := by
  sorry

end evaluate_expression_l78_78938


namespace normal_distribution_probability_l78_78671

variables (ξ : ℝ → ℝ) (μ σ : ℝ)

-- Definitions for given conditions
def is_normal_distribution (ξ : ℝ → ℝ) (μ σ : ℝ) :=
  ∀ a b : ℝ, P(a < ξ ∧ ξ ≤ b) = ∫ x in a..b, exp (-(x - μ)^2 / (2 * σ^2)) / (σ * sqrt (2 * π))

-- Problem statement
theorem normal_distribution_probability :
  (E ξ = 3) → (variance ξ = 1) → (is_normal_distribution ξ 3 1) → (P (2 < ξ ∧ ξ ≤ 4) = 0.683) :=
sorry

end normal_distribution_probability_l78_78671


namespace max_snowmen_constructed_l78_78472

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78472


namespace length_segment_AB_l78_78302

theorem length_segment_AB : 
  let polar_line := fun (ρ θ : ℝ) => ρ * cos θ = 4,
      parametric_curve := fun (t : ℝ) => (t^2, t^3) in
  (∀ t : ℝ, (t = 2 ∨ t = -2) →
    let A := parametric_curve 2,
        B := parametric_curve (-2) in
    dist A B = 16) :=
sorry

end length_segment_AB_l78_78302


namespace exists_reals_b_i_l78_78217

theorem exists_reals_b_i (n : ℕ) (a : Fin n → ℝ) (h : n ≥ 1) :
  ∃ b : Fin n → ℝ, (∀ i, a i - b i ∈ ℤ ∧ a i - b i > 0) ∧ 
    ∑ i j, (i < j : Prop) * ((b i - b j) ^ 2) ≤ (n ^ 2 - 1) / 12 := 
by
  sorry

end exists_reals_b_i_l78_78217


namespace remaining_integers_count_l78_78742

def T := { n : ℕ | n > 0 ∧ n ≤ 100 }

def is_multiple (x y : ℕ) : Prop := ∃ k : ℕ, y = k * x

def filter_multiples (S : set ℕ) (p : ℕ) : set ℕ :=
  { n ∈ S | ¬ (is_multiple p n) }

def T_filtered := (filter_multiples (filter_multiples (filter_multiples T 2) 3) 5)

theorem remaining_integers_count : T_filtered.card = 29 := by
  sorry

end remaining_integers_count_l78_78742


namespace max_number_of_snowmen_l78_78519

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78519


namespace volume_of_region_l78_78014

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

theorem volume_of_region (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 7) :
  volume_of_sphere r_large - volume_of_sphere r_small = 372 * Real.pi := by
  rw [h_small, h_large]
  sorry

end volume_of_region_l78_78014


namespace find_ab_y_lt_zero_l78_78955

theorem find_ab (a b: ℝ) (h1: 2 * a + 4 * b = 10) (h2: 3 * a + b = 10) : 
  a = 3 ∧ b = 1 :=
by
  -- Given that 2a + 4b = 10 and 3a + b = 10, we need to prove that a = 3 and b = 1
  sorry

theorem y_lt_zero (x: ℝ) (h: x > 10 / 3) : 
  let y := 10 - 3 * x in y < 0 :=
by
  -- Given x > 10 / 3, we need to prove that y = 10 - 3x < 0
  sorry

end find_ab_y_lt_zero_l78_78955


namespace DE_perp_AC_l78_78676

-- Definition of the conditions of the problem in Lean 4
variables {A B C D E M : Type}
variables [EuclideanGeometry A B C D E M]
variables (triangle_ABC : is_triangle A B C)
variables (angles_acute : acute_angle (\angle B A C) ∧ acute_angle (\angle C A B))
variables (M_midpoint : midpoint M A B)
variables (D_halfline : point_on_halfline D C B)
variables (angle_condition : ∠ D A B = ∠ B C M)
variables (E_intersection : E = intersection (perpendicular_from B (line_through C D)) (perpendicular_bisector A B))

-- Statement of the proof problem
theorem DE_perp_AC :
  perpendicular (line_through D E) (line_through A C) :=
sorry

end DE_perp_AC_l78_78676


namespace probability_AB_cannot_participate_together_l78_78606

theorem probability_AB_cannot_participate_together : 
  let total_ways := nat.choose 6 4,
      ways_AB_together := nat.choose 4 2 in
  (1 - (ways_AB_together / total_ways : ℚ)) = 3 / 5 :=
by 
  have h_total_ways : total_ways = 15 := by norm_num,
  have h_ways_AB_together : ways_AB_together = 6 := by norm_num,
  rw [h_total_ways, h_ways_AB_together],
  norm_num,
  sorry

end probability_AB_cannot_participate_together_l78_78606


namespace train_cross_time_l78_78307

-- Definitions from conditions
def train_length : ℝ := 100 -- in meters
def train_speed_kmph : ℝ := 360 -- in km/hr

-- Conversion factor and converted speed (not derived from solution steps, but necessary for Lean)
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600
def train_speed_mps : ℝ := (train_speed_kmph * km_to_m) / hr_to_s 

-- Problem statement and proof placeholder
theorem train_cross_time : train_length / train_speed_mps = 1 := 
by 
  have h_conversion : train_speed_mps = 100 := sorry -- Verify the conversion
  rw h_conversion
  norm_num
  sorry

end train_cross_time_l78_78307


namespace probability_one_tail_given_one_head_l78_78575

-- Define what it means to toss a fair coin three times
def outcomes : finset (fin 3 → bool) := 
  finset.univ

-- Define the event of getting at least one head
def at_least_one_head (s : fin 3 → bool) : Prop :=
  ∃ i, s i = tt

-- Define the event of getting exactly one tail
def exactly_one_tail (s : fin 3 → bool) : Prop :=
  finset.card (finset.filter (λ i, s i = ff) finset.univ) = 1

-- Define the set of outcomes with at least one head
def outcomes_with_head := 
  (finset.filter at_least_one_head outcomes)

-- Define the set of outcomes with exactly one tail given at least one head
def favorable_outcomes := 
  (finset.filter exactly_one_tail outcomes_with_head)

-- Proof statement: Probability calculation
theorem probability_one_tail_given_one_head :
  (favorable_outcomes.card : ℚ) / (outcomes_with_head.card : ℚ) = 3 / 7 :=
sorry

end probability_one_tail_given_one_head_l78_78575


namespace circle_area_from_polar_eq_l78_78180

-- Given conditions
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- Statement to be proved
theorem circle_area_from_polar_eq :
  (∃ r θ : ℝ, polar_eq r θ) →
  ∃ (A : ℝ), A = Real.pi * ((5 / 2) ^ 2) :=
by
  intro h
  use Real.pi * ((5 / 2) ^ 2)
  sorry

end circle_area_from_polar_eq_l78_78180


namespace smallest_leading_coefficient_l78_78032

theorem smallest_leading_coefficient :
  ∀ (P : ℤ → ℤ), (∃ (a b c : ℚ), ∀ (x : ℤ), P x = a * (x^2 : ℚ) + b * (x : ℚ) + c) →
  (∀ x : ℤ, ∃ k : ℤ, P x = k) →
  (∃ a : ℚ, (∀ x : ℤ, ∃ k : ℤ, a * (x^2 : ℚ) + b * (x : ℚ) + c = k) ∧ a > 0 ∧ (∀ a' : ℚ, (∀ x : ℤ, ∃ k : ℤ, a' * (x^2 : ℚ) + b * (x : ℚ) + c = k) → a' ≥ a) ∧ a = 1 / 2) := 
sorry

end smallest_leading_coefficient_l78_78032


namespace f_plus_f_inv_sum_f_terms_l78_78637

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem f_plus_f_inv (x : ℝ) (hx : x ≠ 0) : f(x) + f(1/x) = 1 := 
by
  sorry

theorem sum_f_terms : 
  (Finset.range 2010).sum (λ n, f (n + 1)) + (Finset.range 2010).sum (λ n, f (1 / (n + 2))) = 4019 / 2 :=
by
  sorry

end f_plus_f_inv_sum_f_terms_l78_78637


namespace decreasing_function_range_l78_78579

theorem decreasing_function_range (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) ^ x < (a^2 - 1) ^ (x + 1)) → 1 < |a| ∧ |a| < sqrt 2 :=
by 
  sorry

end decreasing_function_range_l78_78579


namespace Susan_apples_correct_l78_78903

def Phillip_apples : ℝ := 38.25
def Ben_apples : ℝ := Phillip_apples + 8.5
def Tom_apples : ℝ := (3 / 8) * Ben_apples
def Susan_apples : ℝ := (1 / 2) * Tom_apples + 7

theorem Susan_apples_correct :
  Susan_apples = 15.765625 := 
by
  unfold Phillip_apples Ben_apples Tom_apples Susan_apples
  sorry

end Susan_apples_correct_l78_78903


namespace n_eq_14_l78_78962

variable {a : ℕ → ℕ}  -- the arithmetic sequence
variable {S : ℕ → ℕ}  -- the sum function of the first n terms
variable {d : ℕ}      -- the common difference of the arithmetic sequence

-- Given Conditions
axiom Sn_eq_4 : S 4 = 40
axiom Sn_eq_210 : ∃ (n : ℕ), S n = 210
axiom Sn_minus_4_eq_130 : ∃ (n : ℕ), S (n - 4) = 130

-- Main theorem to prove
theorem n_eq_14 : ∃ (n : ℕ),  S n = 210 ∧ S (n - 4) = 130 ∧ n = 14 :=
by
  sorry

end n_eq_14_l78_78962


namespace max_snowmen_l78_78496

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78496


namespace radius_omega2_proof_angle_BDC_proof_l78_78884

-- Define the given conditions
variables {A B C D O O1 O2 F P : Type*}
variables (circle1 circle2 : A)
variables (r : ℝ) (alpha : ℝ)
variable (B F : ℝ)
variable (D P : ℝ)

-- Given conditions:
-- 1. Quadrilateral ABCD is inscribed in a circle with center O.
-- 2. Two circles Omega1 and Omega2 have equal radii and are inscribed
--    in angles ∠ABC and ∠ADC, respectively.
-- 3. Circle Omega1 touches side BC at point F.
-- 4. Circle Omega2 touches side AD at point P.
-- 5. BF = 3√2.
-- 6. DP = √2.
-- 7. O1 is the center of the circle circumscribed around triangle BOC.

noncomputable def radius_omega2 := r
axiom r_squared_eq_six {r : ℝ} : r ^ 2 = 6

noncomputable def angle_BDC := alpha
axiom BDC_equals_30 (alpha : ℝ) : alpha = 30

-- Proof problems corresponding to the given conditions and results:
theorem radius_omega2_proof : radius_omega2 = sqrt 6 :=
by sorry

theorem angle_BDC_proof : angle_BDC = 30 :=
by sorry

end radius_omega2_proof_angle_BDC_proof_l78_78884


namespace smallest_nat_num_with_props_l78_78207

-- Define a function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

-- Main statement
theorem smallest_nat_num_with_props :
  ∃ (N : ℕ), 
    (N % 100 = 56) ∧ 
    (56 ∣ N) ∧ 
    (sum_of_digits N = 56) ∧ 
    ∀ (M : ℕ), 
      (M % 100 = 56) ∧ (56 ∣ M) ∧ (sum_of_digits M = 56) → N ≤ M :=
  ⟨29899856, by sorry⟩

end smallest_nat_num_with_props_l78_78207


namespace not_possible_to_sum_2020_with_same_digit_sum_99_l78_78308

-- Define digit sum function
def digit_sum (n : ℕ) : ℕ :=
  (nat.digits 10 n).sum

-- Define the main proposition
theorem not_possible_to_sum_2020_with_same_digit_sum_99 : 
  ¬(∃ (N d : ℕ), (∀ i ∈ finset.range 99, digit_sum (N + d * i) = d) ∧ 
                  finset.sum (finset.range 99) (λ i, N + d * i) = 2020) :=
by
  sorry

end not_possible_to_sum_2020_with_same_digit_sum_99_l78_78308


namespace geric_initial_bills_l78_78607

theorem geric_initial_bills :
  ∀ (bills_jessa : ℕ) (bills_kylan: ℕ) (bills_geric : ℕ),
  bills_jessa = 10 →
  bills_kylan = bills_jessa - 2 →
  bills_geric = 2 * bills_kylan →
  bills_geric = 16 :=
by
  intros bills_jessa bills_kylan bills_geric h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  rw [h3]
  sorry

end geric_initial_bills_l78_78607


namespace abs_value_equation_l78_78738

-- Define the main proof problem
theorem abs_value_equation (a b c d : ℝ)
  (h : ∀ x : ℝ, |2 * x + 4| + |a * x + b| = |c * x + d|) :
  d = 2 * c :=
sorry -- Proof skipped for this exercise

end abs_value_equation_l78_78738


namespace AC_plus_EC_gt_AD_l78_78364

-- Define the basic geometric constructs and relationships
variables {A B C D E : Type} [EuclideanGeometry A B C]
variables {angle_ABC : Angle A B C} (angle_ACB_eq_40 : angle_ABC = 40)
variables {D_on_AB : Point D ∈ LineSegment A B}
variables {E_on_BC : Point E ∈ LineSegment B C}
variables {angle_BED : Angle B E D} (angle_BED_eq_20 : angle_BED = 20)

-- State the theorem to be proven
theorem AC_plus_EC_gt_AD (h1 : angle_ACB_eq_40) (h2 : angle_BED_eq_20) : 
  distance A C + distance E C > distance A D := 
by 
  sorry

end AC_plus_EC_gt_AD_l78_78364


namespace customer_can_receive_exact_change_l78_78872

theorem customer_can_receive_exact_change (k : ℕ) (hk : k ≤ 1000) :
  ∃ change : ℕ, change + k = 1000 ∧ change ≤ 1999 :=
by
  sorry

end customer_can_receive_exact_change_l78_78872


namespace phantom_needs_additional_money_l78_78367

noncomputable def phantom_additional_money_needed
  (black_qty red_qty yellow_qty blue_qty magenta_qty cyan_qty : ℕ)
  (black_cost red_cost yellow_cost blue_cost magenta_cost cyan_cost : ℝ)
  (phantom_money sales_tax_rate : ℝ) : ℝ :=
  let subtotal := 
    black_qty * black_cost + red_qty * red_cost + yellow_qty * yellow_cost + 
    blue_qty * blue_cost + magenta_qty * magenta_cost + cyan_qty * cyan_cost in
  let total_cost := subtotal + subtotal * sales_tax_rate in
  total_cost - phantom_money

theorem phantom_needs_additional_money :
  phantom_additional_money_needed 3 4 3 2 2 1 12 16 14 17 15 18 50 0.05 = 185.20 :=
by
  unfold phantom_additional_money_needed
  simp
  norm_num
  sorry

end phantom_needs_additional_money_l78_78367


namespace find_point_C_l78_78544

theorem find_point_C :
  ∃ C : ℝ × ℝ, let A : ℝ × ℝ := (-3, 5) in
                 let B : ℝ × ℝ := (9, -1) in
                 let AB := (B.1 - A.1, B.2 - A.2) in
                 C = (B.1 + 0.5 * AB.1, B.2 + 0.5 * AB.2) ∧ 
                 C = (15, -4) :=
by
  sorry

end find_point_C_l78_78544


namespace profit_percent_l78_78441

variable (C S : ℝ)
variable (h : (1 / 3) * S = 0.8 * C)

theorem profit_percent (h : (1 / 3) * S = 0.8 * C) : 
  ((S - C) / C) * 100 = 140 := 
by
  sorry

end profit_percent_l78_78441


namespace negation_of_proposition_l78_78790

theorem negation_of_proposition :
  (¬∃ x₀ ∈ Set.Ioo 0 (π/2), Real.cos x₀ > Real.sin x₀) ↔ ∀ x ∈ Set.Ioo 0 (π / 2), Real.cos x ≤ Real.sin x :=
by
  sorry

end negation_of_proposition_l78_78790


namespace johns_latest_race_time_l78_78718

-- Definitions for the conditions
def initial_times : List ℕ := [100, 108, 112, 104, 110]

-- Prove that John's time for the latest race is 104 seconds
theorem johns_latest_race_time (additional_time : ℕ) (new_list : List ℕ)
  (h1 : new_list = (initial_times ++ [additional_time]).sort)
  (h2 : new_list.nth_le 2 sorry + new_list.nth_le 3 sorry = 212) : additional_time = 104 :=
sorry

end johns_latest_race_time_l78_78718


namespace product_of_solutions_abs_eq_four_l78_78697

theorem product_of_solutions_abs_eq_four :
  (∀ x : ℝ, (|x - 5| - 4 = 0) → (x = 9 ∨ x = 1)) →
  (9 * 1 = 9) :=
by
  intros h
  sorry

end product_of_solutions_abs_eq_four_l78_78697


namespace equation_of_ellipse_equation_of_line_l78_78979

noncomputable def FocalLength : ℝ := 4
noncomputable def Eccentricity : ℝ := Real.sqrt 2 / 2
noncomputable def PointP : ℝ × ℝ := (0, -1)
noncomputable def RatioPAtoPB : ℝ := 2

theorem equation_of_ellipse (c e : ℝ) (c_pos : 0 < c) (e_pos : 0 < e) (e_def : e = Real.sqrt 2 / 2) (c_def : c = 2) :
  ∃ a b : ℝ, (a > b ∧ b > 0 ∧ (c = e * a) ∧ (a = 2 * Real.sqrt 2) ∧ (b^2 = a^2 - c^2) ∧ (b = 2) ∧ ( ( ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1 ) ) ) :=
sorry

theorem equation_of_line (c e : ℝ) (P : ℝ × ℝ) (ratio : ℝ) (a b : ℝ) 
  (c_pos : 0 < c) (e_pos : 0 < e) (P_def : P = (0, -1)) (ratio_def : ratio = 2) 
  (ellipse_eq : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1) 
  (line_intersects_ellipse : ∃ (l : ℝ → ℝ), ∀ x : ℝ, ∃ y : ℝ, y = l x ∧ (x^2) / (a^2) + (y^2) / (b^2) = 1) :
  ∃ k : ℝ, (l : ℝ → ℝ) (l_eq : l = λ x, k * x - 1) ∧ (k = (± (3 * Real.sqrt 10 / 10))) :=
sorry

end equation_of_ellipse_equation_of_line_l78_78979


namespace area_parallelogram_l78_78558

variables (ABCD EF GH EBH DGF ECG : Type)
variables [Parallelogram ABCD] [Parallel EF AD] [Parallel GH AB] 

namespace Geometry

def area (T : Type) [HasArea T ℝ] : ℝ := sorry

theorem area_parallelogram (hEF_AD : EF ∥ AD) 
                           (hGH_AB : GH ∥ AB) 
                           (hEBH_area : area EBH = 6)
                           (hDGF_area : area DGF = 8)
                           (hECG_area : area ECG = 18) :
                           area ABCD = 60 := 
sorry

end Geometry

end area_parallelogram_l78_78558


namespace sum_of_perimeters_l78_78761

theorem sum_of_perimeters (p q : ℕ) (h_coprime : Nat.coprime p q) (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_lt : p < q) :
  let triangles_count := p + q - 1 in
  let triangle_perimeter := 2 + Real.sqrt ((p^2 + q^2) / p^2) in
  (triangles_count * triangle_perimeter) = (p + q - 1) * (2 + Real.sqrt ((p^2 + q^2) / p^2)) := by
  sorry

end sum_of_perimeters_l78_78761


namespace find_y_l78_78212

theorem find_y (y : ℕ) : y = (12 ^ 3 * 6 ^ 4) / 432 → y = 5184 :=
by
  intro h
  rw [h]
  sorry

end find_y_l78_78212


namespace reflex_angle_at_T_is_correct_l78_78952

-- Define the vertices P, Q, R, S, T lying on a line and T not on the line
variables {P Q R S T : Type}

-- Define the angles given in the conditions
variables (α β : ℝ)
hypothesis hα : α = 100
hypothesis hβ : β = 110

-- Define the angles around point Q and R
noncomputable def angle_TQP : ℝ := 180 - α
noncomputable def angle_TRS : ℝ := 180 - β

-- Define the angle TQR using the sum of angles in triangle TQR
noncomputable def angle_TQR : ℝ := 180 - angle_TQP α - angle_TRS β

-- Define the reflex angle at T
noncomputable def reflex_angle_at_T : ℝ := 360 - angle_TQR α β

-- The theorem to prove
theorem reflex_angle_at_T_is_correct : reflex_angle_at_T α β = 330 :=
by {
  sorry
}

end reflex_angle_at_T_is_correct_l78_78952


namespace quadratic_root_a_value_l78_78667

theorem quadratic_root_a_value (a : ℝ) (h : 2^2 - 2 * a + 6 = 0) : a = 5 :=
sorry

end quadratic_root_a_value_l78_78667


namespace find_angle_A_l78_78296

def triangle_ABC_angle_A (a b : ℝ) (B A : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute) : Prop :=
  A = Real.pi / 3

theorem find_angle_A 
  (a b A B : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute)
  (h_conditions : triangle_ABC_angle_A a b B A acute ha hb hB hacute) : 
  A = Real.pi / 3 := 
sorry

end find_angle_A_l78_78296


namespace superhero_movies_l78_78176

theorem superhero_movies (d h a together : ℕ) (H1: d = 7) (H2: h = 12) (H3: a = 15) (H4: together = 2) :
  (d + h + a - together) = 32 :=
by
  rw [H1, H2, H3, H4]
  norm_num

end superhero_movies_l78_78176


namespace exists_positive_integer_sequence_l78_78604

theorem exists_positive_integer_sequence (n : ℕ) (h : n ≥ 4) :
  ∃ (a : ℕ → ℕ), (∀ i, 1 ≤ i → i ≤ n → a i > 0) ∧
  (∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) ∧
  a n = (∑ i in finset.range (n - 1).succ, a (i + 1)) ∧
  1 / a 1 = (∑ i in finset.range (n - 1), 1 / a (i + 2)) :=
sorry

end exists_positive_integer_sequence_l78_78604


namespace max_unique_triangles_l78_78888

def triangle (a b c : ℕ) : Prop :=
  a >= b ∧ b >= c ∧ b + c > a ∧ a < 7 ∧ b < 7 ∧ c < 7

def is_similar (a b c x y z : ℕ) : Prop :=
  a * y = b * x ∧ b * z = c * y ∧ c * x = a * z

def count_triangles : ℕ :=
  ∑ i in finset.range 7, ∑ j in finset.range (i + 1), ∑ k in finset.range (j + 1), if triangle i j k then 1 else 0

theorem max_unique_triangles : count_triangles = 31 :=
by
  sorry

end max_unique_triangles_l78_78888


namespace convert_degrees_to_radians_convert_radians_to_degrees_l78_78174

noncomputable def degrees_to_radians : ℝ → ℝ :=
λ deg, deg * (Real.pi / 180)

noncomputable def radians_to_degrees : ℝ → ℝ :=
λ rad, rad * (180 / Real.pi)

theorem convert_degrees_to_radians (deg : ℝ) (result : ℝ) : degrees_to_radians deg = result := 
  by
  sorry

theorem convert_radians_to_degrees (rad : ℝ) (result : ℝ) : radians_to_degrees rad = result := 
  by
  sorry

example : convert_degrees_to_radians (-135) (-(3 / 4) * Real.pi) := 
  by
  sorry

example : convert_radians_to_degrees ((11 / 3) * Real.pi) 660 := 
  by
  sorry

end convert_degrees_to_radians_convert_radians_to_degrees_l78_78174


namespace max_number_of_snowmen_l78_78518

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78518


namespace find_number_l78_78037

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l78_78037


namespace proof_b_lt_a_lt_c_l78_78611

noncomputable def a : ℝ := Real.log 0.6 / Real.log 0.5
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := 1.1 ^ 0.9

theorem proof_b_lt_a_lt_c : b < a ∧ a < c :=
by
  have ha : a = Real.log 0.6 / Real.log 0.5 := rfl
  have hb : b = Real.log 0.9 / Real.log 1.1 := rfl
  have hc : c = 1.1 ^ 0.9 := rfl
  sorry

end proof_b_lt_a_lt_c_l78_78611


namespace tetrahedron_circumsphere_surface_area_l78_78694

theorem tetrahedron_circumsphere_surface_area
  (SA CB : ℝ) (SB AC : ℝ) (SC AB : ℝ)
  (hSA_CB : SA = CB) (hSA_val : SA = √5) 
  (hSB_AC : SB = AC) (hSB_val : SB = √10)
  (hSC_AB : SC = AB) (hSC_val : SC = √13) :
  let D := √(5 + 10 + 13) in
  let r := D / 2 in
  let A := 4 * π * r^2 in
  A = 14 * π := 
by
  -- Definitions for lengths
  have h1 : CB = √5 := by rw [← hSA_CB, hSA_val],
  have h2 : AC = √10 := by rw [← hSB_AC, hSB_val],
  have h3 : AB = √13 := by rw [← hSC_AB, hSC_val],
  
  -- Calculation based on the conditions (proof steps not included)
  sorry

end tetrahedron_circumsphere_surface_area_l78_78694


namespace max_distinct_fans_l78_78080

-- Define the problem conditions and the main statement
theorem max_distinct_fans : 
  let n := 6  -- Number of sectors per fan
  let configurations := 2^n  -- Total configurations without considering symmetry
  let unchanged_flips := 8  -- Number of configurations unchanged by flipping
  let distinct_configurations := (configurations - unchanged_flips) / 2 + unchanged_flips 
  in 
  distinct_configurations = 36 := by sorry  # We state the answer based on the provided steps


end max_distinct_fans_l78_78080


namespace hectares_used_for_soybeans_and_corn_l78_78750

theorem hectares_used_for_soybeans_and_corn :
  ∀ (total_land wheat_fraction soybeans_ratio corn_ratio soybeans_hectares corn_hectares : ℕ), 
  total_land = 20 →
  wheat_fraction = 1 / 5 →
  soybeans_ratio = 3 →
  corn_ratio = 5 →
  soybeans_hectares = 6 →
  corn_hectares = 10 →
  (total_land * (1 - wheat_fraction) = 16) →
  (soybeans_ratio + corn_ratio = 8) →
  (16 * (soybeans_ratio / (soybeans_ratio + corn_ratio)) = soybeans_hectares) →
  (16 - soybeans_hectares = corn_hectares) :=
by {
  intros,
  sorry
}

end hectares_used_for_soybeans_and_corn_l78_78750


namespace max_snowmen_constructed_l78_78478

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78478


namespace problem2_l78_78223

theorem problem2 (x y : ℝ) (h1 : x^2 + x * y = 3) (h2 : x * y + y^2 = -2) : 
  2 * x^2 - x * y - 3 * y^2 = 12 := 
by 
  sorry

end problem2_l78_78223


namespace log_base_5_of_reciprocal_of_125_l78_78582

theorem log_base_5_of_reciprocal_of_125 : log 5 (1 / 125) = -3 := by
  sorry

end log_base_5_of_reciprocal_of_125_l78_78582


namespace star_246_135_l78_78722

axiom star : ℕ → ℕ → ℕ
variables (x y : ℕ)

axiom star_axiom1 : ∀ x : ℕ, (x + 1) ∗ 0 = (0 ∗ x) + 1
axiom star_axiom2 : ∀ y : ℕ, 0 ∗ (y + 1) = (y ∗ 0) + 1
axiom star_axiom3 : ∀ x y : ℕ, (x + 1) ∗ (y + 1) = (x ∗ y) + 1

axiom star_given : star 123 456 = 789

theorem star_246_135 : star 246 135 = 579 := sorry

end star_246_135_l78_78722


namespace sequence_converges_limit_is_integer_limit_irrational_if_odd_l78_78599

-- Define the sequence and convergence
theorem sequence_converges (k : ℕ) (hk : 1 ≤ k) : 
  ∃ L, ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((a_n k n) - L) < ε :=
sorry

-- Define the condition for k such that the limit is an integer
theorem limit_is_integer (k : ℕ) (L : ℝ) (hk : 1 ≤ k) :
  (∃ m : ℕ, odd m ∧ L = (1 + real.sqrt (4 * k + 1)) / 2) → 
  ∃ k : ℕ, k = (m^2 - 1) / 4 :=
sorry

-- Prove that if k is odd, then the limit is irrational
theorem limit_irrational_if_odd (k : ℕ) (L : ℝ) (hk : 1 ≤ k) (odd_k : odd k) :
  irrational ((1 + real.sqrt (4 * k + 1)) / 2) :=
sorry

end sequence_converges_limit_is_integer_limit_irrational_if_odd_l78_78599


namespace min_workers_for_profit_l78_78109

def revenue (n : ℕ) : ℕ := 240 * n
def cost (n : ℕ) : ℕ := 600 + 200 * n

theorem min_workers_for_profit (n : ℕ) (h : 240 * n > 600 + 200 * n) : n >= 16 :=
by {
  -- Placeholder for the proof steps (which are not required per instructions)
  sorry
}

end min_workers_for_profit_l78_78109


namespace problem1_problem2_l78_78633

-- Definition of the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * log x + 1 / x - 1

-- Problem 1: Monotonicity of f(x)
theorem problem1 (a : ℝ) :
  (∀ x > 0, f a x ≤ f a x) ∨
  (∀ x > 0, x < 1/a → f a x < f a x) ∨
  (∀ x > 0, x > 1/a → f a x > f a x) :=
sorry

-- Problem 2: Prove the inequality
theorem problem2 (n : ℕ) (h : 2 ≤ n) :
  (∑ i in range (n + 1), log i ^ 2) > (n - 1)^4 / (4 * n) :=
sorry

end problem1_problem2_l78_78633


namespace max_snowmen_l78_78482

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78482


namespace spherical_coords_eq_l78_78928

theorem spherical_coords_eq :
  let x := 4
  let y := -4 * Real.sqrt 3
  let z := 4
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := Real.atan2 y x
  let φ := Real.acos (z / ρ)
  (ρ, θ, φ) = (4 * Real.sqrt 5, 5 * Real.pi / 3, Real.acos (1 / Real.sqrt 5)) :=
by
  let x := 4
  let y := -4 * Real.sqrt 3
  let z := 4
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := Real.atan2 y x
  let φ := Real.acos (z / ρ)
  have ρ_pos : 0 < ρ := by sorry
  have θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi := by sorry
  have φ_range : 0 ≤ φ ∧ φ ≤ Real.pi := by sorry
  have h1 : ρ = 4 * Real.sqrt 5 := by sorry
  have h2 : θ = 5 * Real.pi / 3 := by sorry
  have h3 : φ = Real.acos (1 / Real.sqrt 5) := by sorry
  exact ⟨h1, h2, h3⟩

end spherical_coords_eq_l78_78928


namespace set_intersection_complement_l78_78645

open Set

variable {U : Type} [TopologicalSpace U]

theorem set_intersection_complement (A B : Set ℝ) :
  A = {x : ℝ | x > 1} →
  B = {x : ℝ | x > 2} →
  A ∩ (univ \ B) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  assume hA hB
  sorry

end set_intersection_complement_l78_78645


namespace non_zero_digits_of_fraction_l78_78937

theorem non_zero_digits_of_fraction :
  ∀ (n d : ℕ), n = 120 → d = 2^4 * 5^9 → 
  let frac := (n : ℚ) / d in
  let decimal_form := (frac : ℝ).to_digits 10 in
  (count_non_zero_digits decimal_form.digits decimal_form.exponent 10) = 3 := 
by
  intros n d hn hd frac decimal_form
  sorry

end non_zero_digits_of_fraction_l78_78937


namespace sides_of_polygon_l78_78000

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l78_78000


namespace least_positive_integer_greater_than_100_l78_78834

theorem least_positive_integer_greater_than_100 : ∃ n : ℕ, n > 100 ∧ (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_greater_than_100_l78_78834


namespace train_cross_time_l78_78868

theorem train_cross_time
  (length_train : ℕ) (speed_train_kph : ℕ) (speed_train_mps := speed_train_kph * 1000 / 3600) :
  length_train = 75 → speed_train_kph = 54 → (length_train / speed_train_mps = 5 : ℕ) :=
by
  intro h1 h2
  have h3 : speed_train_mps = 15 := by
  {
    sorry
  }
  rw [h1, h3]
  norm_num
  sorry

end train_cross_time_l78_78868


namespace lemons_for_500_ml_l78_78449

def lemons_needed_per_gallon (lemons gallons: ℝ) : ℝ := lemons / gallons

def milliliters_to_gallons (milliliters: ℝ) : ℝ := milliliters / 3785

theorem lemons_for_500_ml (
  lemon_ratio: ℝ := lemons_needed_per_gallon 50 80,
  gallons_in_500_ml: ℝ := milliliters_to_gallons 500
) : lemons_needed_per_gallon 50 80 * (500 / 3785) = 0.083 :=
by
  sorry

end lemons_for_500_ml_l78_78449


namespace molecular_weight_proof_l78_78835

noncomputable def molecular_weight_C7H6O2 := 
  (7 * 12.01) + (6 * 1.008) + (2 * 16.00) -- molecular weight of one mole of C7H6O2

noncomputable def total_molecular_weight_9_moles := 
  9 * molecular_weight_C7H6O2 -- total molecular weight of 9 moles of C7H6O2

theorem molecular_weight_proof : 
  total_molecular_weight_9_moles = 1099.062 := 
by
  sorry

end molecular_weight_proof_l78_78835


namespace lucie_cannot_continue_indefinitely_l78_78463

def finite_list {α : Type*} (l : list α) := l.length < cardinal.omega

theorem lucie_cannot_continue_indefinitely (a : list ℕ) (h : finite_list a) :
  ¬(∀ (x y : ℕ) (i : ℕ), i < a.length - 1 → x = a.nth_le i _ → y = a.nth_le (i+1) _ → x > y →
       (∀ (new_a : list ℕ), (new_a = a.take i ++ [x-1, x] ++ a.drop (i+2) ∨ new_a = a.take i ++ [y+1, x] ++ a.drop (i+2)) →
         ∃ (b : list ℕ), finite_list b ∧ (new_a = b) → lucie_cannot_continue_indefinitely b))
:= sorry

end lucie_cannot_continue_indefinitely_l78_78463


namespace triangle_AC_range_l78_78702

noncomputable def length_AB : ℝ := 12
noncomputable def length_CD : ℝ := 6

def is_valid_AC (AC : ℝ) : Prop :=
  AC > 6 ∧ AC < 24

theorem triangle_AC_range :
  ∃ m n : ℝ, 
    (6 < m ∧ m < 24) ∧ (6 < n ∧ n < 24) ∧
    m + n = 30 ∧
    ∀ AC : ℝ, is_valid_AC AC →
      6 < AC ∧ AC < 24 :=
by
  use 6
  use 24
  simp
  sorry

end triangle_AC_range_l78_78702


namespace trig_identity_on_line_l78_78282

theorem trig_identity_on_line (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 :=
sorry

end trig_identity_on_line_l78_78282


namespace time_for_A_and_D_together_l78_78847

theorem time_for_A_and_D_together (A_rate D_rate combined_rate : ℝ)
  (hA : A_rate = 1 / 10) (hD : D_rate = 1 / 10) 
  (h_combined : combined_rate = A_rate + D_rate) :
  1 / combined_rate = 5 :=
by
  sorry

end time_for_A_and_D_together_l78_78847


namespace ali_peter_fish_ratio_l78_78154

theorem ali_peter_fish_ratio (P J A : ℕ) (h1 : J = P + 1) (h2 : A = 12) (h3 : A + P + J = 25) : A / P = 2 :=
by
  -- Step-by-step simplifications will follow here in the actual proof.
  sorry

end ali_peter_fish_ratio_l78_78154


namespace jenna_hike_distance_l78_78315

noncomputable def distance_walked (x y : ℝ) : ℝ :=
  (x^2 + y^2).sqrt

theorem jenna_hike_distance : distance_walked (5 + 4 * real.sqrt 2) (4 * real.sqrt 2) = 3 * real.sqrt 17 := sorry

end jenna_hike_distance_l78_78315


namespace triangle_ABC_area_l78_78400

-- Definition of the points via reflections
def point_A : ℝ × ℝ := (3, 4)
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

-- Definitions of the points after reflections
def point_B : ℝ × ℝ := reflect_over_y_axis point_A
def point_C : ℝ × ℝ := reflect_over_y_eq_neg_x point_B

-- Distance function for two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Definition of the area of triangle given 3 points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let base := distance A B in
  let height := real.abs (A.2 - C.2) in
  (base * height) / 2

-- The theorem we need to prove
theorem triangle_ABC_area : triangle_area point_A point_B point_C = 21 := by
  sorry

end triangle_ABC_area_l78_78400


namespace miles_for_15_dollars_l78_78265

noncomputable def initial_fare := 3.50
noncomputable def first_mile := 0.75
noncomputable def additional_fare := 0.30
noncomputable def additional_unit := 0.1
noncomputable def total_amount := 15
noncomputable def tip := 3
noncomputable def available_fare := total_amount - tip

theorem miles_for_15_dollars : 
  ∃ x : ℝ, initial_fare + 3 * (x - first_mile) = available_fare → x = 3.6 :=
begin
  sorry
end

end miles_for_15_dollars_l78_78265


namespace find_m_l78_78389

-- Definitions based on the conditions
def ellipse (m : ℝ) := (x^2 / (10 - m)) + (y^2 / (m - 2)) = 1
def eccentricity (m : ℝ) := 4

-- Statement of the theorem
theorem find_m (m : ℝ) (h_ellipse : ellipse m) (h_ecc : eccentricity m) : m = 4 ∨ m = 8 := 
sorry

end find_m_l78_78389


namespace unit_prices_max_colored_tiles_l78_78345

-- Define the given conditions
def condition1 (x y : ℝ) := 40 * x + 60 * y = 5600
def condition2 (x y : ℝ) := 50 * x + 50 * y = 6000

-- Prove the solution for part 1
theorem unit_prices (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 40 := 
sorry

-- Define the condition for the kitchen tiles
def condition3 (a : ℝ) := 80 * a + 40 * (60 - a) ≤ 3400

-- Prove the maximum number of colored tiles for the kitchen
theorem max_colored_tiles (a : ℝ) (h3 : condition3 a) :
  a ≤ 25 := 
sorry

end unit_prices_max_colored_tiles_l78_78345


namespace regular_polygon_perimeter_l78_78535

theorem regular_polygon_perimeter (s : ℝ) (exterior_angle : ℝ) 
  (h1 : s = 7) (h2 : exterior_angle = 45) : 
  8 * s = 56 :=
by
  sorry

end regular_polygon_perimeter_l78_78535


namespace num_sides_polygon_l78_78008

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l78_78008


namespace largest_supporting_25_X_l78_78127

def is_supporting_25 (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ X

theorem largest_supporting_25_X :
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ 0.02 :=
by {
  sorry
}

end largest_supporting_25_X_l78_78127


namespace rectangle_ratio_l78_78850

theorem rectangle_ratio 
  (length width : ℕ)
  (h_length : length = 10)
  (h_width : width = 6)
  (half_length : length / 2 = 5)
  (half_width : width / 2 = 3) :
  let P_small := 2 * (half_length + half_width),
      P_large := 2 * (length + half_width) in
  P_small / P_large = 8 / 13 := by
  sorry

end rectangle_ratio_l78_78850


namespace smallest_integer_with_ten_factors_l78_78836

/-- The function to calculate the number of distinct positive factors of a number n -/
def num_factors (n : ℕ) : ℕ :=
  let factors := n.factors
  factors.foldr (λ (p : ℕ) (m : ℕ), (factors.count p + 1) * m) 1

/-- The statement to be proved in Lean -/
theorem smallest_integer_with_ten_factors : ∃ (n : ℕ), num_factors(n) = 10 ∧ ∀ (m : ℕ), num_factors(m) = 10 → n ≤ m :=
by
  sorry

end smallest_integer_with_ten_factors_l78_78836


namespace function_translation_l78_78821

def translateLeft (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x + a)
def translateUp (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => (f x) + b

theorem function_translation :
  (translateUp (translateLeft (λ x => 2 * x^2) 1) 3) = λ x => 2 * (x + 1)^2 + 3 :=
by
  sorry

end function_translation_l78_78821


namespace solve_fractional_equation_l78_78807

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l78_78807


namespace unique_combination_of_segments_l78_78690

theorem unique_combination_of_segments :
  ∃! (x y : ℤ), 7 * x + 12 * y = 100 := sorry

end unique_combination_of_segments_l78_78690


namespace general_formula_for_a_arithmetic_sum_l78_78961

section ArithmeticSequence

variables {a S : ℕ → ℤ}

/-- Assuming the sum of the first n terms in an arithmetic sequence S satisfies -/
axiom S3_eq_0 : S 3 = 0
axiom S5_eq_neg5 : S 5 = -5

/-- Given the conditions, the general formula for the sequence a_n is proved to be 2 - n -/
theorem general_formula_for_a (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : S 3 = 3 * a 1 + 3 * (a 2 - a 1) = 0)
  (h₂ : S 5 = 5 * a 1 + 10 * (a 2 - a 1) = -5) :
  ∀ n, a n = 2 - n :=
sorry

/-- The sum for the sequence a_{1} + a_{4} + a_{7} + ... + a_{3n+1} -/
theorem arithmetic_sum (a : ℕ → ℤ) (h : ∀ n, a n = 2 - n) :
  ∀ n, (finset.sum (finset.range (n + 1)) (λ k, a (3 * k + 1))) = (n + 1) * (2 - 3 * n) / 2 :=
sorry

end ArithmeticSequence

end general_formula_for_a_arithmetic_sum_l78_78961


namespace no_solution_lines_parallel_l78_78183

theorem no_solution_lines_parallel (m : ℝ) :
  (∀ t s : ℝ, (1 + 5 * t = 4 - 2 * s) ∧ (-3 + 2 * t = 1 + m * s) → false) ↔ m = -4 / 5 :=
by
  sorry

end no_solution_lines_parallel_l78_78183


namespace relationship_between_a_b_c_l78_78992

noncomputable def f (x : ℝ) : ℝ := (4 ^ x - 1) / (2 ^ x)
def a : ℝ := f (2 ^ 0.3)
def b : ℝ := f (0.2 ^ 0.3)
def c : ℝ := f (Real.logBase 0.3 2)

theorem relationship_between_a_b_c : c < b ∧ b < a := 
by {
  -- Placeholder for proof
  sorry
}

end relationship_between_a_b_c_l78_78992


namespace max_snowmen_l78_78520

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78520


namespace problem_l78_78982

noncomputable def f : ℝ → ℝ := sorry

theorem problem :
  (∀ x : ℝ, f (x) + f (x + 2) = 0) →
  (f (1) = -2) →
  (f (2019) + f (2018) = 2) :=
by
  intro h1 h2
  sorry

end problem_l78_78982


namespace discriminant_eq_perfect_square_l78_78273

variables (a b c t : ℝ)

-- Conditions
axiom a_nonzero : a ≠ 0
axiom t_root : a * t^2 + b * t + c = 0

-- Goal
theorem discriminant_eq_perfect_square :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 :=
by
  -- Conditions and goal are stated, proof to be filled.
  sorry

end discriminant_eq_perfect_square_l78_78273


namespace angle_between_vectors_l78_78290

theorem angle_between_vectors (A_polar : ℝ × ℝ) (B_polar : ℝ × ℝ) 
(hA : A_polar = (2, π / 6))
(hB : B_polar = (6, - π / 6)) : 
∃ θ : ℝ, θ = π / 3 := 
begin
  sorry
end

end angle_between_vectors_l78_78290


namespace min_h_condition_l78_78246

open Real

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
{ p | let x := p.1, let y := p.2 in y^2 / a^2 + x^2 / b^2 = 1 }

def parabola (h : ℝ) : set (ℝ × ℝ) :=
{ p | let x := p.1, let y := p.2 in y = x^2 + h }

def midpoint_x (A B : ℝ × ℝ) : ℝ := (A.1 + B.1) / 2

theorem min_h_condition (a b : ℝ)  (h : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b = { p : ℝ × ℝ | p.2^2 / a^2 + p.1^2 / b^2 = 1 } ∧
  ellipse a b ⊆ {(-a, 0), (a, 0), (0, -b), (0, b)} ∧
  right_vertex (1, 0) (ellipse a b) ∧
  chord_perpendicular_to_major_axis a b = 1 ∧
  let P : ℝ × ℝ := (t, t^2 + h) in
  tangent_to_parabola P parabola intersects_ellipse M N ∧
  let midpoint_AP_x := midpoint_x A P in
  let midpoint_MN_x := midpoint_x M N in
  midpoint_AP_x = midpoint_MN_x
→ h = 1 :=
sorry

end min_h_condition_l78_78246


namespace final_purchase_price_correct_l78_78747

-- Definitions
def initial_house_value : ℝ := 100000
def profit_percentage_Mr_Brown : ℝ := 0.10
def renovation_percentage : ℝ := 0.05
def profit_percentage_Mr_Green : ℝ := 0.07
def loss_percentage_Mr_Brown : ℝ := 0.10

-- Calculations
def purchase_price_mr_brown : ℝ := initial_house_value * (1 + profit_percentage_Mr_Brown)
def total_cost_mr_brown : ℝ := purchase_price_mr_brown * (1 + renovation_percentage)
def purchase_price_mr_green : ℝ := total_cost_mr_brown * (1 + profit_percentage_Mr_Green)
def final_purchase_price_mr_brown : ℝ := purchase_price_mr_green * (1 - loss_percentage_Mr_Brown)

-- Statement to prove
theorem final_purchase_price_correct : 
  final_purchase_price_mr_brown = 111226.50 :=
by
  sorry -- Proof is omitted

end final_purchase_price_correct_l78_78747


namespace ivan_income_tax_l78_78708

theorem ivan_income_tax :
  let salary_probation := 20000
  let probation_months := 2
  let salary_after_probation := 25000
  let after_probation_months := 8
  let bonus := 10000
  let tax_rate := 0.13
  let total_income := salary_probation * probation_months +
                      salary_after_probation * after_probation_months + bonus
  total_income * tax_rate = 32500 := sorry

end ivan_income_tax_l78_78708


namespace right_triangle_pythagoras_l78_78301

theorem right_triangle_pythagoras
  (a b c : ℝ)
  (A B C : Triangle)
  (h1 : Triangle.is_right_angle A B C A)
  (h2 : A.sides = (a, b, c))
  (h3 : A.angle = 90) :
  b^2 + c^2 = a^2 :=
begin
  sorry
end

end right_triangle_pythagoras_l78_78301


namespace superhero_movies_l78_78175

theorem superhero_movies (d h a together : ℕ) (H1: d = 7) (H2: h = 12) (H3: a = 15) (H4: together = 2) :
  (d + h + a - together) = 32 :=
by
  rw [H1, H2, H3, H4]
  norm_num

end superhero_movies_l78_78175


namespace no_such_integers_exists_l78_78939

theorem no_such_integers_exists (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) :
  ¬ (a^2 + b^2 = 3 * c^2) := 
by {
  sorry
}

end no_such_integers_exists_l78_78939


namespace max_value_fraction_l78_78214

noncomputable def a_n (n : ℕ) : ℕ := 
  let factors := List.range (n - 1) in
  List.map (λ i, if n % i = 0 then i else 0) factors |>.foldr (+) 0

theorem max_value_fraction (n : ℕ) (hn : n > 1) : 
  (∃ (a_n : ℕ), a_n ≤ n ∧ a_n % n = 0) → a_n / n = 1 / 2 :=
by
  sorry

end max_value_fraction_l78_78214


namespace max_snowmen_l78_78500

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78500


namespace packs_of_chocolate_l78_78745

theorem packs_of_chocolate (t c k x : ℕ) (ht : t = 42) (hc : c = 4) (hk : k = 22) (hx : x = t - (c + k)) : x = 16 :=
by
  rw [ht, hc, hk] at hx
  simp at hx
  exact hx

end packs_of_chocolate_l78_78745


namespace fractional_equation_solution_l78_78801

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ -3) (h : 1/x = 2/(x+3)) : x = 3 :=
sorry

end fractional_equation_solution_l78_78801


namespace sequence_term_2019_l78_78280

/-- Given a sequence {a_n} where a₁ is 2 and aₙ₊₁ equals aₙ minus 2,
    prove that a₂₀₁₉ equals -4034. -/
theorem sequence_term_2019 : (a : ℕ → ℤ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = a n - 2) : a 2019 = -4034 :=
by {
  sorry
}

end sequence_term_2019_l78_78280


namespace smallest_integer_with_ten_factors_l78_78837

/-- The function to calculate the number of distinct positive factors of a number n -/
def num_factors (n : ℕ) : ℕ :=
  let factors := n.factors
  factors.foldr (λ (p : ℕ) (m : ℕ), (factors.count p + 1) * m) 1

/-- The statement to be proved in Lean -/
theorem smallest_integer_with_ten_factors : ∃ (n : ℕ), num_factors(n) = 10 ∧ ∀ (m : ℕ), num_factors(m) = 10 → n ≤ m :=
by
  sorry

end smallest_integer_with_ten_factors_l78_78837


namespace problem_statement_l78_78337

variables {a b c : ℝ}

theorem problem_statement 
  (h1 : a^2 + a * b + b^2 = 9)
  (h2 : b^2 + b * c + c^2 = 52)
  (h3 : c^2 + c * a + a^2 = 49) : 
  (49 * b^2 - 33 * b * c + 9 * c^2) / a^2 = 52 :=
by
  sorry

end problem_statement_l78_78337


namespace compute_f_sum_l78_78767

noncomputable def f : ℝ → ℝ := sorry -- placeholder for f(x)

variables (x : ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = x^2

-- Prove the main statement
theorem compute_f_sum : f (-3 / 2) + f 1 = 3 / 4 :=
by
  sorry

end compute_f_sum_l78_78767


namespace work_done_by_A_alone_l78_78275

theorem work_done_by_A_alone (Wb : ℝ) (Wa : ℝ) (D : ℝ) :
  Wa = 3 * Wb →
  (Wb + Wa) * 18 = D →
  D = 72 → 
  (D / Wa) = 24 := 
by
  intros h1 h2 h3
  sorry

end work_done_by_A_alone_l78_78275


namespace polynomial_operation_correct_l78_78158

theorem polynomial_operation_correct :
    ∀ (s t : ℝ), (s * t + 0.25 * s * t = 0) :=
by
  intros s t
  sorry

end polynomial_operation_correct_l78_78158


namespace total_cars_l78_78546

theorem total_cars (yesterday today : ℕ) (h_yesterday : yesterday = 60) (h_today : today = 2 * yesterday) : yesterday + today = 180 := 
sorry

end total_cars_l78_78546


namespace find_width_of_room_l78_78777

noncomputable def room_width : ℝ :=
let l := 15 in -- length of the room
let verandah_width := 2 in -- width of the verandah
let verandah_area := 124 in -- area of the verandah
let total_length := l + 2 * verandah_width in -- total length including verandah
let total_area (w : ℝ) := total_length * (w + 2 * verandah_width) - l * w in -- total area including verandah minus the room area
classroom_width : ℝ :=
if hh : ∃ w : ℝ, total_area w = verandah_area then Classical.choose hh else 0

theorem find_width_of_room : classroom_width = 12 :=
begin
  sorry
end

end find_width_of_room_l78_78777


namespace danica_planes_l78_78576

def smallestAdditionalPlanes (n k : ℕ) : ℕ :=
  let m := k * (n / k + 1)
  m - n

theorem danica_planes : smallestAdditionalPlanes 17 7 = 4 :=
by
  -- Proof would go here
  sorry

end danica_planes_l78_78576


namespace product_calculation_l78_78910

theorem product_calculation :
  (∏ n in Finset.range 15, (n + 1) * (n + 3) / (n + 4)^2) = 17 / 18 := by
  sorry

end product_calculation_l78_78910


namespace max_domains_count_l78_78638

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 1

-- Prove that the maximum number of domains for the given function and range is 9
theorem max_domains_count : 
  ∃ D : set (set ℝ), 
    (∀ d ∈ D, ∀ y ∈ {-1, 0, 1}, ∃ x ∈ d, f x = y) ∧ 
    (∀ d1 d2 ∈ D, d1 ≠ d2 → ¬ (d1 ⊆ d2 ∧ d2 ⊆ d1)) ∧ 
    D.card = 9 := 
by 
  sorry

end max_domains_count_l78_78638


namespace find_n_l78_78875

variables {x : ℝ}
def a (n : ℕ) : ℝ := if n = 1 then sin x else if n = 2 then cos x else if n = 3 then tan x
  else have r : ℝ := cos x / sin x, r^(n-3) * tan x

theorem find_n (x : ℝ) (n : ℕ) (h₁ : (a 1) = sin x) (h₂ : (a 2) = cos x) (h₃ : (a 3) = tan x) :
  a n = 1 + cos x ↔ n = 8 :=
sorry

end find_n_l78_78875


namespace largest_25_supporting_X_l78_78133

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i) ∈ Int → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ ∀ Y : ℝ, (is_25_supporting Y → Y ≤ 0.02) :=
sorry

end largest_25_supporting_X_l78_78133


namespace max_number_of_snowmen_l78_78512

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78512


namespace sum_log_is_eq_l78_78169

noncomputable def log_base (b x : ℝ) := log x / log b

noncomputable def summand (k : ℕ) : ℝ := log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3

theorem sum_log_is_eq :
  (∑ k in finset.range 48 \ finset.range 2, summand (k + 3)) = 1 - (1 / log_base 3 51) :=
sorry

end sum_log_is_eq_l78_78169


namespace smallest_whole_number_divisible_by_8_leaves_remainder_1_l78_78033

theorem smallest_whole_number_divisible_by_8_leaves_remainder_1 :
  ∃ (n : ℕ), n ≡ 1 [MOD 2] ∧ n ≡ 1 [MOD 3] ∧ n ≡ 1 [MOD 4] ∧ n ≡ 1 [MOD 5] ∧ n ≡ 1 [MOD 7] ∧ n % 8 = 0 ∧ n = 7141 :=
by
  sorry

end smallest_whole_number_divisible_by_8_leaves_remainder_1_l78_78033


namespace sum_of_arithmetic_sequence_l78_78333

theorem sum_of_arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) :
  (∀ n, a n = 4 + (n - 4) * d) →
  a 4 = 4 →
  S 5 = 15 →
  (∀ n, S n = n * (a 3)) →
  d = 1 →
  (∑ n in range (m + 1), (1 / (n * (n + 1)))) = 10 / 11 →
  m = 10 :=
by
  sorry

end sum_of_arithmetic_sequence_l78_78333


namespace width_of_Carols_rectangle_l78_78915

theorem width_of_Carols_rectangle 
  (w : ℝ) 
  (h1 : 15 * w = 6 * 50) : w = 20 := 
by 
  sorry

end width_of_Carols_rectangle_l78_78915


namespace quadratic_roots_irrational_l78_78927

theorem quadratic_roots_irrational (c : ℝ) (h : 3 * (3 : ℝ)^2 = 4 * 3 * c) :
  ∃ x : ℝ, (3 * x^2 - 6 * x * real.sqrt 3 + c = 0) ∧ real.sqrt 3 = x :=
by
  sorry

end quadratic_roots_irrational_l78_78927


namespace find_diameter_of_outer_boundary_l78_78107

/-- The conditions for the problem. -/
def running_track_width : ℝ := 7
def seating_ring_width : ℝ := 12
def major_axis_pitch : ℝ := 18
def minor_axis_pitch : ℝ := 14

/-- The goal is to prove the diameter of the circle that is the outer boundary of the running track. -/
theorem find_diameter_of_outer_boundary :
  let radius_minor_axis := minor_axis_pitch / 2 in
  let total_radius := radius_minor_axis + seating_ring_width + running_track_width in
  (2 * total_radius) = 52 :=
by
  sorry

end find_diameter_of_outer_boundary_l78_78107


namespace find_v_l78_78383

-- Given conditions translated to Lean
variables {p q r u v w : ℝ}

-- roots of the first polynomial
def poly1_roots : Prop := p + q + r = -5 ∧ pq + qr + rp = 2

-- roots of the second polynomial expressed in terms of p, q, r
def poly2_roots : Prop :=
  let p_plus_q := p + q in
  let q_plus_r := q + r in
  let r_plus_p := r + p in
  u = -(p_plus_q + q_plus_r + r_plus_p) ∧
  v = p_plus_q * q_plus_r + q_plus_r * r_plus_p + r_plus_p * p_plus_q ∧
  w = -(p_plus_q * q_plus_r * r_plus_p)

-- The theorem we want to prove
theorem find_v (h1 : poly1_roots) (h2 : poly2_roots) : v = 6 :=
by
  sorry

end find_v_l78_78383


namespace students_like_both_30_l78_78286

theorem students_like_both_30 (total_students : ℕ) (students_like_math : ℕ) 
                              (students_like_english : ℕ) (students_neither : ℕ) :
  total_students = 48 → students_like_math = 38 → students_like_english = 36 → 
  students_neither = 4 → 
  (students_like_math + students_like_english - (total_students - students_neither)) = 30 :=
by {
  intros,
  sorry -- proof not required
}

end students_like_both_30_l78_78286


namespace solve_triangle_ABC_l78_78300

-- Define the right triangle with the given conditions
def triangle_ABC (A B C : Type) [MetricSpace C] (a b c : C) :=
  Angle A c = 30 ∧ Angle B c = 60 ∧ dist b c = sqrt 3

-- State the theorem to solve the right triangle
theorem solve_triangle_ABC (A B C : Type) [MetricSpace C] 
  (a b c : C) (h1 : Angle A c = 30) (h2 : Angle B c = 60) (h3 : dist b c = sqrt 3) :
  dist a b = 2 * sqrt 3 ∧ dist a c = 3 :=
sorry

end solve_triangle_ABC_l78_78300


namespace radius_of_circle_l78_78593

namespace CircleRadius

variable (r : ℝ) 

-- Define the conditions
def chordLength (r : ℝ) : Prop := ∃ k : ℝ, k = sqrt (r^2 - 9) ∧ (2*k + 2)^2 + 1 = r^2

-- Prove the radius is sqrt(10)
theorem radius_of_circle : chordLength r → r = sqrt 10 :=
by
  intro h
  cases' h with k hk
  sorry

end CircleRadius

end radius_of_circle_l78_78593


namespace sum_of_first_8_terms_l78_78810

-- Define the geometric sequence properties
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), (a 0 = a₁) ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first n terms of a sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- State the conditions:
-- Considering sequence a is geometric, and sums S₄ and S₁₂ are given 
axiom S4 : ℝ := 5
axiom S12 : ℝ := 35

theorem sum_of_first_8_terms (a : ℕ → ℝ) (h1 : is_geometric_sequence a) 
  (h2 : sum_first_n_terms a 4 = S4) (h3 : sum_first_n_terms a 12 = S12) : 
  sum_first_n_terms a 8 = 15 :=
sorry

end sum_of_first_8_terms_l78_78810


namespace product_of_repeating_decimal_and_integer_l78_78909

theorem product_of_repeating_decimal_and_integer :
  (let x := 0.333...) → 
  x * 9 = 3 :=
by
  sorry

end product_of_repeating_decimal_and_integer_l78_78909


namespace min_value_PF_PA_l78_78224

open Classical

noncomputable section

def parabola_eq (x y : ℝ) : Prop := y^2 = 16 * x

def point_A : ℝ × ℝ := (1, 2)

def focus_F : ℝ × ℝ := (4, 0)  -- Focus of the given parabola y^2 = 16x

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def PF_PA (P : ℝ × ℝ) : ℝ :=
  distance P focus_F + distance P point_A

theorem min_value_PF_PA :
  ∃ P : ℝ × ℝ, parabola_eq P.1 P.2 ∧ PF_PA P = 5 :=
sorry

end min_value_PF_PA_l78_78224


namespace max_snowmen_constructed_l78_78474

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78474


namespace fraction_simplification_l78_78431

theorem fraction_simplification (a b c x y : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), (y ≠ 0 → (y^2 / x^2) ≠ (y / x))) ∧
  (∀ (a b c : ℝ), (a + c^2) / (b + c^2) ≠ a / b) ∧
  (∀ (a b m : ℝ), ¬(m ≠ -1 → (a + b) / (m * a + m * b) = 1 / 2)) ∧
  (∃ a b : ℝ, (a - b) / (b - a) = -1) :=
  by
  sorry

end fraction_simplification_l78_78431


namespace max_value_of_f_l78_78789

open Real

noncomputable def f (x : ℝ) : ℝ := -x - 9 / x + 18

theorem max_value_of_f : ∀ x > 0, f x ≤ 12 :=
by
  sorry

end max_value_of_f_l78_78789


namespace ivan_income_tax_l78_78709

theorem ivan_income_tax :
  let salary_probation := 20000
  let probation_months := 2
  let salary_after_probation := 25000
  let after_probation_months := 8
  let bonus := 10000
  let tax_rate := 0.13
  let total_income := salary_probation * probation_months +
                      salary_after_probation * after_probation_months + bonus
  total_income * tax_rate = 32500 := sorry

end ivan_income_tax_l78_78709


namespace find_x2_l78_78683

theorem find_x2
  (x2 : ℝ)
  (h : abs (x2 - 4) * 9 = 76) :
  x2 = 76 / 9 + 4 :=
begin
  sorry
end

end find_x2_l78_78683


namespace product_of_repeating_decimal_and_integer_l78_78908

theorem product_of_repeating_decimal_and_integer :
  (let x := 0.333...) → 
  x * 9 = 3 :=
by
  sorry

end product_of_repeating_decimal_and_integer_l78_78908


namespace smallest_integer_remainder_l78_78425

theorem smallest_integer_remainder :
  ∃ n : ℕ, n > 1 ∧
           (n % 3 = 2) ∧
           (n % 4 = 2) ∧
           (n % 5 = 2) ∧
           (n % 7 = 2) ∧
           n = 422 :=
by
  sorry

end smallest_integer_remainder_l78_78425


namespace distinct_fan_count_l78_78087

def max_distinct_fans : Nat :=
  36

theorem distinct_fan_count (n : Nat) (r b : S) (paint_scheme : Fin n → bool) :
  (∀i, r ≠ b → (paint_scheme i = b ∨ paint_scheme i = r)) ∧ 
  (∀i, paint_scheme i ≠ paint_scheme (i + n / 2 % n)) →
  n = 6 →
  max_distinct_fans = 36 :=
by
  sorry

end distinct_fan_count_l78_78087


namespace max_distinct_fans_l78_78069

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l78_78069


namespace angle_CBE_double_ABC_l78_78551

open EuclideanGeometry
open Real

variables (A B C M D E F : Point) (triangle_abc : Triangle A B C)
variables (h1 : angle A B C > 90)
variables (h2 : midpoint M A B)
variables (h3 : foot D C A B)
variables (h4 : ray_contains A C E)
variables (h5 : distance E M = distance B M)
variables (h6 : intersection D E B C F)
variables (h7 : distance B E = distance B F)

theorem angle_CBE_double_ABC :
  angle C B E = 2 * angle A B C :=
sorry

end angle_CBE_double_ABC_l78_78551


namespace sequence_terms_distinct_l78_78377

theorem sequence_terms_distinct (n m : ℕ) (hnm : n ≠ m) : 
  (n / (n + 1) : ℚ) ≠ (m / (m + 1) : ℚ) :=
sorry

end sequence_terms_distinct_l78_78377


namespace product_of_repeating_decimal_l78_78907

theorem product_of_repeating_decimal :
  let q := (1 : ℚ) / 3 in
  q * 9 = 3 :=
by
  sorry

end product_of_repeating_decimal_l78_78907


namespace min_value_of_p_l78_78996

noncomputable def p (x : ℝ) : ℝ := x^2 + 6 * x + 5

theorem min_value_of_p : ∃ x : ℝ, p(x) = -4 :=
by
  have h : ∃ x : ℝ, p x = -4 := ⟨-3, by norm_num [p]⟩
  exact h

end min_value_of_p_l78_78996


namespace maximum_snowmen_count_l78_78493

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78493


namespace max_distinct_fans_l78_78068

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l78_78068


namespace hire_year_1971_l78_78871

def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed = 70

def hired_year (first_eligible_year : ℕ) (years_employed : ℕ) : ℕ :=
  first_eligible_year - years_employed

theorem hire_year_1971 :
  ∀ (birth_year eligible_year : ℕ),
  let age_when_hired := eligible_year - (70 - (eligible_year - birth_year))
  (birth_year = 32) → (eligible_year = 2009) → age_when_hired = 1971 := by 
  intros birth_year eligible_year h_birth_year h_eligible_year
  unfold hired_year rule_of_70
  sorry

end hire_year_1971_l78_78871


namespace find_valid_pairs_l78_78585

def validPair (x y: ℕ) : Prop := (xy - 6)^2 ∣ (x^2 + y^2)

theorem find_valid_pairs :
  { (x, y) ∈ setOf (λ p : ℕ × ℕ, validPair p.1 p.2) | x > 0 ∧ y > 0 } = {(7, 1), (4, 2), (3, 3)} := sorry

end find_valid_pairs_l78_78585


namespace arrange_digits_2210_l78_78691

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements_ending_0 : ℕ :=
  factorial 3 / factorial 2

def valid_arrangements_ending_2 : ℕ :=
  factorial 3 - 2

def total_arrangements : ℕ :=
  arrangements_ending_0 + valid_arrangements_ending_2

theorem arrange_digits_2210 : total_arrangements = 7 :=
  sorry

end arrange_digits_2210_l78_78691


namespace max_snowmen_l78_78484

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78484


namespace sequence_product_correct_l78_78998

def sequence (n : ℕ) : ℤ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

theorem sequence_product_correct : sequence 2 * sequence 3 = 20 := by
  sorry

end sequence_product_correct_l78_78998


namespace probability_s_less_than_zero_l78_78668

open Real

theorem probability_s_less_than_zero (p : ℕ) (hp : 1 ≤ p ∧ p ≤ 10) (hprob : ∀ p, p ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → (p ^ 2 - 13 * p + 40 < 0) ↔ (p = 6 ∨ p = 7)) :
  (∃ s, p ^ 2 - 13 * p + 40 = s) ∧ (∑ p in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, if p ^ 2 - 13 * p + 40 < 0 then 1 else 0) / 10 = 0.2 :=
by
  sorry

end probability_s_less_than_zero_l78_78668


namespace perpendiculars_form_square_l78_78881

theorem perpendiculars_form_square
  (A B C D A1 B1 C1 D1 : Type)
  (ABCD_square : IsSquare A B C D)
  (A1B1C1D1_parallelogram : IsParallelogram A1 B1 C1 D1)
  (A_on_A1B1 B_on_B1C1 C_on_C1D1 D_on_D1A1 : Moreover)
  (l1_perpendicular : Perpendicular A1 (AB : Line))
  (l2_perpendicular : Perpendicular B1 (BC : Line))
  (l3_perpendicular : Perpendicular C1 (CD : Line))
  (l4_perpendicular : Perpendicular D1 (DA : Line)) :
  IsSquare l1 l2 l3 l4 := 
sorry

end perpendiculars_form_square_l78_78881


namespace number_of_solutions_l78_78650

theorem number_of_solutions (x y: ℕ) (hx : 0 < x) (hy : 0 < y) :
    (1 / (x + 1) + 1 / y + 1 / ((x + 1) * y) = 1 / 1991) →
    ∃! (n : ℕ), n = 64 :=
by
  sorry

end number_of_solutions_l78_78650


namespace range_of_absolute_difference_l78_78170

theorem range_of_absolute_difference : (∃ x : ℝ, y = |x + 4| - |x - 5|) → y ∈ [-9, 9] :=
sorry

end range_of_absolute_difference_l78_78170


namespace hexagon_perpendicular_dist_sum_l78_78095

-- Definitions as per conditions
def RegularHexagon (ABCDEF : Type) [Hexagon ABCDEF] : Prop := 
  ∀ (A B C D E F : Point), Equilateral (∠ ABC ∧ ∠ BCD ∧ ∠ CDE ∧ ∠ DEF ∧ ∠ EFA ∧ ∠ FAB)

def Perpendicular (A : Point) (l : Line) (P : Point) : Prop := 
  isPerpendicular (lineThrough A l) (P)

def CenterHexagon (O : Point) (ABCDEF : Type) [Hexagon ABCDEF] : Prop :=
  ∀ (A B C D E F : Point), dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D ∧ dist O D = dist O E ∧ dist O E = dist O F

variable [RegularHexagon ABCDEF]
variable {A B C D E F O P Q R : Point}
variable (hP : Perpendicular A lineCD P) (hQ : Perpendicular A lineEF Q) (hR : Perpendicular A lineBC R)
variable (hO : CenterHexagon O ABCDEF)
variable (hOP : dist O P = 1)

-- Lean statement of the proof problem
theorem hexagon_perpendicular_dist_sum :
  dist A O + dist A Q + dist A R = 3 :=
sorry

end hexagon_perpendicular_dist_sum_l78_78095


namespace extreme_values_a_neg2_monotone_f_a4_l78_78995

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 3

theorem extreme_values_a_neg2 :
  let a := -2 in
  let f := f x₀ a in
  interval := set.Icc (-4 : ℝ) 6 in
  is_min_on f interval 2 (-1) ∧ 
  is_max_on f interval (-4) 35 :=
by
  intros F a
  let interval := set.Icc (-4 : ℝ) 6
  show is_min_on F interval 2 (-1) ∧ 
isen not a statement
  (f : ℝ → ℝ) :=
-- define the interval first
let interval := set.Icc (-4 : ℝ) 6 in
-- then provide two parts of the proof
  is_min_on (f (-2) F) interval 2 (-1) ∧ -- minimum occurring at x = 2
  is_max_on (f (-2) F) interval (-4) 35 -- maximum occurring at x = -4
clear
sorry

theorem monotone_f_a4 :
  let a := 4 in
  let f := f x₀ a in
  ∀ x₁ x₂: ℝ, 
    -4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 6 → f x₁ < f x₂ :=
by
  intros F a
  show monotone_on F (set.Icc (-4 : ℝ) 6)
  sorry

end extreme_values_a_neg2_monotone_f_a4_l78_78995


namespace digit_count_first_1500_even_integers_l78_78912

theorem digit_count_first_1500_even_integers : 
  let one_digit_numbers : ℕ := 4, 
      two_digit_numbers : ℕ := (98 - 10) / 2 + 1,
      three_digit_numbers : ℕ := (998 - 100) / 2 + 1 in
  (one_digit_numbers * 1) + (two_digit_numbers * 2) + (three_digit_numbers * 3) = 1444 :=
  by
  let one_digit_numbers := 4
  let two_digit_numbers := (98 - 10) / 2 + 1
  let three_digit_numbers := (998 - 100) / 2 + 1
  sorry

end digit_count_first_1500_even_integers_l78_78912


namespace orthocenter_on_circumcircle_l78_78057

variables {A B C M N D E H : Type*}
variables [triangle A B C] [acute_triangle ABC] 
variables [segment M AC] [segment N AC] (MN_eq_AC : MN = AC)
variables [perpendicular_from M BC D] [perpendicular_from N AB E]
variables [orthocenter H ABC] [circumcircle BED]

theorem orthocenter_on_circumcircle :
  H ∈ circumcircle BED :=
sorry

end orthocenter_on_circumcircle_l78_78057


namespace maximum_snowmen_count_l78_78492

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78492


namespace ratio_CX_XF_l78_78414

theorem ratio_CX_XF (A B C E F X : Type) [NonComputable] [inhabited X]
  (h1: ∀ AF FB AE EC, AF / FB = (AE / EC) ^ 2)
  (h2: ∀ BE midpoint_X_BE, X = midpoint_BE) :
  CX / XF = sqrt 5 := sorry

end ratio_CX_XF_l78_78414


namespace trig_identity_proof_l78_78954

theorem trig_identity_proof
  (α : ℝ)
  (h : Real.sin (α - π / 6) = 3 / 5) :
  Real.cos (2 * π / 3 - α) = 3 / 5 :=
sorry

end trig_identity_proof_l78_78954


namespace novel_pages_l78_78344

theorem novel_pages (x : ℝ) : (1 / 4 * x - 55 = 50) → x = 420 := by
  intro h
  have := calc
    1 / 4 * x - 55 = 50    : by { exact h }
    1 / 4 * x = 105        : by { sorry }
    x = 420                : by { sorry }
  exact this

end novel_pages_l78_78344


namespace mass_of_region_l78_78162

noncomputable def density (x y : ℝ) : ℝ :=
  (5 * (x^2 + y^2)) / 6

noncomputable def inside_cylinder_cone_region : Set (ℝ × ℝ × ℝ) :=
  {p | (let x := p.1 in let y := p.2 in let z := p.3 in
        x^2 + y^2 ≤ 1 ∧ 36 * (x^2 + y^2) = z^2 ∧ x ≥ 0 ∧ z ≥ 0)}

theorem mass_of_region : ∫ (p : ℝ × ℝ × ℝ) in inside_cylinder_cone_region, density p.1 p.2 :=
  2 * Real.pi :=
sorry

end mass_of_region_l78_78162


namespace count_f100_equals_27_l78_78600

def d (n : ℕ) : ℕ := Nat.divisors n |>.length

def f1 (n : ℕ) : ℕ := 3 * d n

def fj : ℕ → ℕ → ℕ
| 1, n => f1 n
| j+1, n => f1 (fj j n)

theorem count_f100_equals_27 :
  (Finset.filter (λ n => fj 100 n = 27) (Finset.range 101)).card = 6 := sorry

end count_f100_equals_27_l78_78600


namespace ptolemy_zero_side_ptolemy_rectangle_ptolemy_trapezoid_l78_78830

-- Case 1: One side of the cyclic quadrilateral is reduced to zero
theorem ptolemy_zero_side (b c d e f : ℝ) (h1 : c ≠ 0) : b * d = e * f → b * d = e * f := by
  intro h
  exact h

-- Case 2: Cyclic quadrilateral is a rectangle
theorem ptolemy_rectangle (a b : ℝ) (h : a > 0 ∧ b > 0) (e f : ℝ) (he : e = f) (hf : e = sqrt (a^2 + b^2)) :
  2 * a * b = a^2 + b^2 :=
by
  have heq: e = sqrt (a^2 + b^2) := hf
  rw [he] at heq
  exact (by sorry)

-- Case 3: Cyclic quadrilateral is an isosceles trapezoid
theorem ptolemy_trapezoid (a b c e : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h : c = d) : 
  e^2 = c^2 + a * b := 
by
  assume h_eq
  rw [h_eq]
  exact (by sorry)

end ptolemy_zero_side_ptolemy_rectangle_ptolemy_trapezoid_l78_78830


namespace smallest_covering_radius_l78_78384

-- Define the basic geometric entities and conditions.
structure Circle := 
  (center : ℝ × ℝ)
  (radius : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the non-overlapping condition of two circles.
def non_overlapping (c1 c2 : Circle) : Prop := 
  distance c1.center c2.center > c1.radius + c2.radius

-- Define the covering radius of the union of two circles.
def covering_radius (c1 c2 : Circle) [h : non_overlapping c1 c2] : ℝ := 
  ∃ B1 C2, (B1 ∈ c1 ∧ C2 ∈ c2 ∧ distance B1 C2 = distance c1.center c2.center + c1.radius + c2.radius) 
  ∧ (by sorry)

-- The Lean statement for the smallest covering radius problem.
theorem smallest_covering_radius (c1 c2 : Circle) (non_overlap : non_overlapping c1 c2) : 
  covering_radius c1 c2 = (1 / 2) * (distance c1.center c2.center + c1.radius + c2.radius) :=
by sorry

end smallest_covering_radius_l78_78384


namespace angle_CAD_is_115_degrees_l78_78863

theorem angle_CAD_is_115_degrees (A B C D : Type) 
  [IsoscelesTriangle ABC (AB) (BC)] 
  (angle_C : ℝ) (h1 : angle_C = 50) : 
  (mangle CAD = 115) :=
sorry

end angle_CAD_is_115_degrees_l78_78863


namespace total_value_of_coins_is_correct_l78_78934

def rolls_dollars : ℕ := 6
def rolls_half_dollars : ℕ := 5
def rolls_quarters : ℕ := 7
def rolls_dimes : ℕ := 4
def rolls_nickels : ℕ := 3
def rolls_pennies : ℕ := 2

def coins_per_dollar_roll : ℕ := 20
def coins_per_half_dollar_roll : ℕ := 25
def coins_per_quarter_roll : ℕ := 40
def coins_per_dime_roll : ℕ := 50
def coins_per_nickel_roll : ℕ := 40
def coins_per_penny_roll : ℕ := 50

def value_per_dollar : ℚ := 1
def value_per_half_dollar : ℚ := 0.5
def value_per_quarter : ℚ := 0.25
def value_per_dime : ℚ := 0.10
def value_per_nickel : ℚ := 0.05
def value_per_penny : ℚ := 0.01

theorem total_value_of_coins_is_correct : 
  rolls_dollars * coins_per_dollar_roll * value_per_dollar +
  rolls_half_dollars * coins_per_half_dollar_roll * value_per_half_dollar +
  rolls_quarters * coins_per_quarter_roll * value_per_quarter +
  rolls_dimes * coins_per_dime_roll * value_per_dime +
  rolls_nickels * coins_per_nickel_roll * value_per_nickel +
  rolls_pennies * coins_per_penny_roll * value_per_penny = 279.50 := 
sorry

end total_value_of_coins_is_correct_l78_78934


namespace minimum_additional_squares_needed_to_achieve_symmetry_l78_78922

def initial_grid : List (ℕ × ℕ) := [(1, 4), (4, 1)] -- Initial shaded squares

def is_symmetric (grid : List (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ), x ∈ grid → y ∈ grid →
    ((x.1 = 2 * 2 - y.1 ∧ x.2 = y.2) ∨
     (x.1 = y.1 ∧ x.2 = 5 - y.2) ∨
     (x.1 = 2 * 2 - y.1 ∧ x.2 = 5 - y.2))

def additional_squares_needed : ℕ :=
  6 -- As derived in the solution steps, 6 additional squares are needed to achieve symmetry

theorem minimum_additional_squares_needed_to_achieve_symmetry :
  ∀ (initial_shades : List (ℕ × ℕ)),
    initial_shades = initial_grid →
    ∃ (additional : List (ℕ × ℕ)),
      initial_shades ++ additional = symmetric_grid ∧
      additional.length = additional_squares_needed :=
by 
-- skip the proof
sorry

end minimum_additional_squares_needed_to_achieve_symmetry_l78_78922


namespace intervals_of_monotonicity_max_and_min_values_l78_78249

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * (cos x)^2

theorem intervals_of_monotonicity : 
  ∃ k : ℤ, 
    (∀ x, x ∈ Set.Icc (k * π - 5 * π / 12) (k * π + π / 12) → ∀ y, y ∈ Set.Icc (k * π - 5 * π / 12) (k * π + π / 12) → f x ≤ f y) ∧ 
    (∀ x, x ∈ Set.Icc (k * π + π / 12) (k * π + 7 * π / 12) → ∀ y, y ∈ Set.Icc (k * π + π / 12) (k * π + 7 * π / 12) → f x ≥ f y) :=
sorry

theorem max_and_min_values : 
  ∃ (a b : ℝ), 
    (a = 2 + sqrt 3) ∧ (b = 0) ∧ 
    (∀ x, x ∈ Set.Icc (-π / 3) (π / 3) → f x ≤ a) ∧ 
    (∀ x, x ∈ Set.Icc (-π / 3) (π / 3) → f x ≥ b) :=
sorry

end intervals_of_monotonicity_max_and_min_values_l78_78249


namespace increasing_log_function_l78_78670

theorem increasing_log_function (a m : ℝ) (f : ℝ → ℝ)
    (h_f_def : ∀ x, f x = x^2 - 2 * a * x + 3)
    (h0 : 0 < a) :
    (∀ x ∈ Ioo 0 m, log (1 / 2) (f x) > log (1 / 2) (f x + m)) → 
    (if a <= sqrt 3 then m ≤ a else m ≤ a - sqrt (a^2 - 3)) :=
begin
  sorry
end

end increasing_log_function_l78_78670


namespace probability_union_l78_78794

theorem probability_union (P_A P_B P_AB : ℝ) (hA : P_A = 0.45) (hB : P_B = 0.4) (hAB : P_AB = 0.25) : 
  P_A + P_B - P_AB = 0.60 :=
by
  rw [hA, hB, hAB]
  norm_num

end probability_union_l78_78794


namespace percentage_increase_correct_l78_78773

def highest_price : ℕ := 24
def lowest_price : ℕ := 16

theorem percentage_increase_correct :
  ((highest_price - lowest_price) * 100 / lowest_price) = 50 :=
by
  sorry

end percentage_increase_correct_l78_78773


namespace tangent_line_eq_f_monotonicity_intervals_g_extremum_l78_78390

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / x - a * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x ^ 2 + f x a

-- (I) Tangent line problem
theorem tangent_line_eq {a : ℝ} (h : a = -1) : 
  ∃ m b, (∀ x, f x a = m * x + b) ∧ f 1 a = 2 ∧ m = -1 ∧ b = 3 :=
sorry

-- (II) Monotonicity intervals
theorem f_monotonicity_intervals {a : ℝ} : 
  if h : a ≥ 0 then (∀ x > 0, Deriv f x a < 0)
  else (∀ x, (0 < x ∧ x < -2 / a → Deriv f x a < 0) 
             ∧ (x > -2 / a → Deriv f x a > 0)) :=
sorry

-- (III) Extremum of g(x)
theorem g_extremum {a : ℝ} : 
  (∃ x, x ∈ Ioo 0 1 ∧ Deriv (g x a) = 0) ↔ (a < 0) :=
sorry

end tangent_line_eq_f_monotonicity_intervals_g_extremum_l78_78390


namespace maximum_size_of_wanting_set_l78_78739

def is_wanting_set {α : Type} (T : set α) [decidable_eq α] (S : set α) (n : ℕ) : Prop :=
  ∃ (c : ℕ), 0 < c ∧ c ≤ n / 2 ∧ ∀ (s1 s2 : α), s1 ∈ S → s2 ∈ S → s1 ≠ s2 → |s1 - s2| ≠ c

theorem maximum_size_of_wanting_set {n : ℕ} (h : n > 3) :
  ∃ (S : set ℕ), is_wanting_set (set.Icc 1 n) S n ∧ S.card = floor (2 * n / 3) :=
sorry

end maximum_size_of_wanting_set_l78_78739


namespace parallel_lines_implies_a_eq_one_l78_78981

theorem parallel_lines_implies_a_eq_one 
(h_parallel: ∀ (a : ℝ), ∀ (x y : ℝ), (x + a * y = 2 * a + 2) → (a * x + y = a + 1) → -1/a = -a) :
  ∀ (a : ℝ), a = 1 := by
  sorry

end parallel_lines_implies_a_eq_one_l78_78981


namespace example_theorem_l78_78963

noncomputable def ellipse_eccentricity_square_geq_two_thirds (a b : ℝ) (x0 y0 : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x0^2 / a^2 + y0^2 / b^2 = 1) ∧ (∃ P : ℝ × ℝ, 
    (angle ((-a, 0) : ℝ × ℝ) ((a, 0) : ℝ × ℝ) P = 120)) → 
  let e := Real.sqrt (1 - (b / a)^2) in
  e^2 ≥ 2 / 3

-- usage in a proof would look like this
theorem example_theorem {a b x0 y0 : ℝ} : 
  ellipse_eccentricity_square_geq_two_thirds a b x0 y0 :=
sorry

end example_theorem_l78_78963


namespace trigonometric_identity_proof_l78_78613

theorem trigonometric_identity_proof (α : ℝ) 
  (h1 : π < α)
  (h2 : α < 2 * π)
  (h3 : cos (α - 9 * π) = -3 / 5) : 
  cos (α - 11 * π / 2) = 4 / 5 :=
sorry

end trigonometric_identity_proof_l78_78613


namespace perimeter_of_regular_polygon_l78_78536

def regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 360 / n ∧ n * side_length > 0

theorem perimeter_of_regular_polygon :
  ∀ (n : ℕ) (side_length : ℝ), regular_polygon n side_length 45 → side_length = 7 → 8 = n → n * side_length = 56 :=
by
  intros n side_length h1 h2 h3
  rw [h2, h3]
  sorry

end perimeter_of_regular_polygon_l78_78536


namespace james_weekly_pistachio_cost_l78_78715

def cost_per_can : ℕ := 10
def ounces_per_can : ℕ := 5
def consumption_per_5_days : ℕ := 30
def days_per_week : ℕ := 7

theorem james_weekly_pistachio_cost : (days_per_week / 5 * consumption_per_5_days) / ounces_per_can * cost_per_can = 90 := 
by
  sorry

end james_weekly_pistachio_cost_l78_78715


namespace f_satisfies_conditions_l78_78874

def g (n : Int) : Int :=
  if n >= 1 then 1 else 0

def f (n m : Int) : Int :=
  if m = 0 then n
  else n % m

theorem f_satisfies_conditions (n m : Int) : 
  (f 0 m = 0) ∧ 
  (f (n + 1) m = (1 - g m + g m * g (m - 1 - f n m)) * (1 + f n m)) := by
  sorry

end f_satisfies_conditions_l78_78874


namespace polygon_sides_sum_l78_78005

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l78_78005


namespace enrique_commission_l78_78189

def original_prices := 
  (suits: Real := 700.00, shirts: Real := 50.00, loafers: Real := 150.00, ties: Real := 30.00, socks: Real := 10.00)

def quantities := 
  (suits: Nat := 2, shirts: Nat := 6, loafers: Nat := 2, ties: Nat := 4, socks: Nat := 5)

def discount :=
  (suits: Real := 0.10, shirts: Real := 0.0, loafers: Real := 0.0, ties: Real := 0.0, socks: Real := 0.0)

def tax_rate :=
  (suits: Real := 0.0, shirts: Real := 0.05, loafers: Real := 0.05, ties: Real := 0.05, socks: Real := 0.05)

def commission_rate :=
  (clothing: Real := 0.15, accessories: Real := 0.10)

def is_accessory (item: String) : Bool :=
  item = "loafers" ∨ item = "ties" ∨ item = "socks"

noncomputable def salesprice (original_price: Real) (discount: Real) :=
  original_price * (1 - discount)

noncomputable def salesprice_after_tax (original_price: Real) (discount: Real) (tax_rate: Real) :=
  salesprice original_price discount * (1 + tax_rate)

noncomputable def commission (quantity: Nat) (price: Real) (commission_rate: Real) : Real :=
  quantity * price * commission_rate

theorem enrique_commission :
  let suit_price_after_discount := salesprice original_prices.suits discount.suits
  let suits_commission := commission quantities.suits suit_price_after_discount commission_rate.clothing
  let shirts_price_after_tax := salesprice_after_tax original_prices.shirts discount.shirts tax_rate.shirts
  let shirts_commission := commission quantities.shirts shirts_price_after_tax commission_rate.clothing
  let loafers_price_after_tax := salesprice_after_tax original_prices.loafers discount.loafers tax_rate.loafers
  let loafers_commission := commission quantities.loafers loafers_price_after_tax commission_rate.accessories
  let ties_price_after_tax := salesprice_after_tax original_prices.ties discount.ties tax_rate.ties
  let ties_commission := commission quantities.ties ties_price_after_tax commission_rate.accessories
  let socks_price_after_tax := salesprice_after_tax original_prices.socks discount.socks tax_rate.socks
  let socks_commission := commission quantities.socks socks_price_after_tax commission_rate.accessories
  let total_commission := suits_commission + shirts_commission + loafers_commission + ties_commission + socks_commission
  total_commission = 285.60 := sorry

end enrique_commission_l78_78189


namespace sum_of_angles_eq_90_l78_78020

noncomputable theory

open_locale real geometry

variables {A B C D E F G H I J K L : Point}
-- Declare that $A, B, C$ are distinct points forming triangle ABC
-- Squares ABDE, BCFG, and CAHI are constructed on sides AB, BC, and CA respectively
-- Parallelograms DBGJ, FCIK, and HAEL are formed by extending triangles DBG, FCI, and HAE

def is_square (a b c d : Point) : Prop := 
  dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a ∧ 
  angle (a - b) (b - c) = real.pi / 2 ∧ angle (b - c) (c - d) = real.pi / 2 ∧ 
  angle (c - d) (d - a) = real.pi / 2 ∧ angle (d - a) (a - b) = real.pi / 2

def is_parallelogram (p q r s : Point) : Prop := 
  dist p q = dist r s ∧ dist q r = dist s p ∧ 
  vector.angle (p - q) (q - r) = vector.angle (r - s) (s - p)

theorem sum_of_angles_eq_90 :
  triangle A B C →
  is_square A B D E → is_square B C F G →
  is_square C A H I →
  is_parallelogram D B G J → is_parallelogram F C I K →
  is_parallelogram H A E L →
  ∠ A K B + ∠ B L C + ∠ C J A = 90 :=
sorry

end sum_of_angles_eq_90_l78_78020


namespace sum_base6_l78_78562

theorem sum_base6 (a b c : ℕ) 
  (ha : a = 1 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 1 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hc : c = 1 * 6^1 + 5 * 6^0) :
  a + b + c = 2 * 6^3 + 2 * 6^2 + 0 * 6^1 + 3 * 6^0 :=
by 
  sorry

end sum_base6_l78_78562


namespace max_colored_squares_l78_78828

theorem max_colored_squares (n : ℕ) (m : ℕ) (k : ℕ) (color_count : ℕ) (grid : Fin n → Fin m → Fin k) :
  n = 99 → m = 99 → k = 5 →
  (∀ (i : Fin n) (j1 j2 : Fin m), grid i j1 = grid i j2) →
  (∀ (j : Fin m) (i1 i2 : Fin n), grid i1 j = grid i2 j) →
  (∀ c : Fin k, (∑ i j, if grid i j = c then 1 else 0) = (n * m) / k) →
  ∃ (max_cells : ℕ), max_cells = 1900 :=
by
  sorry

end max_colored_squares_l78_78828


namespace value_of_frac_sum_l78_78664

theorem value_of_frac_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : (x + y) / 3 = 11 / 9 :=
by
  sorry

end value_of_frac_sum_l78_78664


namespace volleyball_team_starters_l78_78366

theorem volleyball_team_starters :
  nat.choose 16 7 = 11440 :=
by
  sorry

end volleyball_team_starters_l78_78366


namespace increasing_interval_l78_78031

open Real

-- Definition of the function y = sin(-2x + π/4)
def func (x : ℝ) : ℝ := sin (-2 * x + π / 4)

-- The increasing interval for the function given the conditions
theorem increasing_interval (k : ℤ) :
  ∃ (I : set ℝ), I = set.Icc (k * π + 3 * π / 8) (k * π + 7 * π / 8) ∧ 
  ∀ x ∈ I, func x < func (x + ε) := 
sorry

end increasing_interval_l78_78031


namespace sides_of_polygon_l78_78001

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l78_78001


namespace max_snowmen_l78_78527

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l78_78527


namespace ratio_rounded_to_nearest_tenth_l78_78288

-- Define the conditions
def votes_in_favor := 15
def total_students := 22
def ratio := votes_in_favor / total_students

-- Lean statement asserting the main question
theorem ratio_rounded_to_nearest_tenth (v : ℕ := votes_in_favor) (t : ℕ := total_students) :
  (Real.round (v / t * 10) / 10) = 0.7 :=
  by
    have h1 : ratio = (15 : ℝ) / 22 := sorry
    have h2 : ratio ≈ 0.6818 := sorry
    sorry

end ratio_rounded_to_nearest_tenth_l78_78288


namespace sequence_properties_l78_78253

-- Define the sequence formula
def a_n (n : ℤ) : ℤ := n^2 - 5 * n + 4

-- State the theorem about the sequence
theorem sequence_properties :
  -- Part 1: The number of negative terms in the sequence
  (∃ (S : Finset ℤ), ∀ n ∈ S, a_n n < 0 ∧ S.card = 2) ∧
  -- Part 2: The minimum value of the sequence and the value of n at minimum
  (∀ n : ℤ, (a_n n ≥ -9 / 4) ∧ (a_n (5 / 2) = -9 / 4)) :=
by {
  sorry
}

end sequence_properties_l78_78253


namespace dishonest_dealer_profit_l78_78557

def weight_actual := 575
def weight_claimed := 1000

def profit_percentage (actual claimed : ℕ) : ℕ :=
  ((claimed - actual) * 100) / claimed

theorem dishonest_dealer_profit :
  profit_percentage weight_actual weight_claimed = 42.5 := 
sorry

end dishonest_dealer_profit_l78_78557


namespace sufficient_but_not_necessary_decreasing_l78_78181

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f y ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6 * m * x + 6

theorem sufficient_but_not_necessary_decreasing (m : ℝ) :
  m = 1 → is_decreasing_on (f m) (Set.Iic 3) :=
by
  intros h
  rw [h]
  sorry

end sufficient_but_not_necessary_decreasing_l78_78181


namespace concyclic_B_C_I_I_A_center_of_circle_l78_78864

variables {A B C : Point}
variable {I : Point} -- incenter of triangle ABC
variable {I_A : Point} -- A-excenter of triangle ABC
variable {S : Point} -- supposed circumcenter of BIC

-- Conditions given:
def is_incenter := ∃ I, incenter I A B C
def is_a_excenter := ∃ I_A, excenter I_A B C 

theorem concyclic_B_C_I_I_A (h_incenter : is_incenter) (h_a_excenter : is_a_excenter) :
  concyclic B C I I_A := sorry

theorem center_of_circle (h_incenter : is_incenter) (h_a_excenter : is_a_excenter) :
  circle_center S B C I I_A := sorry

end concyclic_B_C_I_I_A_center_of_circle_l78_78864


namespace total_exercise_time_l78_78313

-- Definitions based on given conditions
def javier_daily : ℕ := 50
def javier_days : ℕ := 7
def sanda_daily : ℕ := 90
def sanda_days : ℕ := 3

-- Proof problem to verify the total exercise time for both Javier and Sanda
theorem total_exercise_time : javier_daily * javier_days + sanda_daily * sanda_days = 620 := by
  sorry

end total_exercise_time_l78_78313


namespace determine_a_l78_78986

theorem determine_a 
  (a : ℝ) 
  (intersects : ∃ P : ℝ × ℝ, P ∈ {P | P.1^2 + P.2^2 + 2 * P.1 - 2 * P.2 + 2 * a = 0} 
                         ∧ P.1 + P.2 + 2 = 0)
  (chord_length : ∃ P Q : ℝ × ℝ, P ∈ {P | P.1^2 + P.2^2 + 2 * P.1 - 2 * P.2 + 2 * a = 0} 
                         ∧ Q ∈ {Q | Q.1^2 + Q.2^2 + 2 * Q.1 - 2 * Q.2 + 2 * a = 0} 
                         ∧ (P - Q).norm = 4) :
  a = -2 := 
sorry

end determine_a_l78_78986


namespace problem_sol_1_problem_sol_2_problem_sol_3_l78_78156

def choose_classes_ways := 546
def allocate_students_ways := 90
def arrange_people_ways := 2400

theorem problem_sol_1 (liberal_arts: ℕ) (science: ℕ) (choose_1: ℕ) (choose_2: ℕ):
  liberal_arts = 6 → science = 14 → choose_1 = 1 → choose_2 = 2 →
  (liberal_arts.choose choose_1) * (science.choose choose_2) = choose_classes_ways :=
by sorry

theorem problem_sol_2 (students: ℕ) (classA: ℕ) (classB: ℕ) (classC: ℕ):
  students = 5 → classA = 1 → classB = 2 → classC = 2 →
  ((students.choose classA) * ((students - classA).choose classB) * ((students - classA - classB).choose classC)) / (2! * 1! * 1!) * 3! = allocate_students_ways :=
by sorry

theorem problem_sol_3 (total_people: ℕ) (students: ℕ) (supervisors: ℕ):
  total_people = 8 → students = 5 → supervisors = 3 →
  ((total_people - 2)! * students!) / supervisors! = arrange_people_ways :=
by sorry

end problem_sol_1_problem_sol_2_problem_sol_3_l78_78156


namespace meeting_probability_l78_78529

noncomputable def probability_meeting_occurs : ℝ :=
  let x y w z : ℝ := sorry
  if h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ w ∧ w ≤ 3 ∧ 0 ≤ z ∧ z ≤ 3 ∧ z > x ∧ z > y ∧ z > w ∧ |x - y| ≤ 0.5 ∧ |x - w| ≤ 0.5 ∧ |y - w| ≤ 0.5
  then 1/6
  else 0

theorem meeting_probability : probability_meeting_occurs = 1/6 := 
  sorry

end meeting_probability_l78_78529


namespace probability_sum_less_than_product_l78_78824

def set_of_even_integers : Set ℕ := {2, 4, 6, 8, 10}

def sum_less_than_product (a b : ℕ) : Prop :=
  a + b < a * b

theorem probability_sum_less_than_product :
  let total_combinations := 25
  let valid_combinations := 16
  (valid_combinations / total_combinations : ℚ) = 16 / 25 :=
by
  sorry

end probability_sum_less_than_product_l78_78824


namespace eval_expression_l78_78446

theorem eval_expression :
  (∛(-8) + |(-6)| - 2^2) = 0 :=
by 
  sorry  -- proof would go here

end eval_expression_l78_78446


namespace value_of_f_log3_54_l78_78234

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem value_of_f_log3_54
  (h1 : is_odd f)
  (h2 : ∀ x, f (x + 2) = -1 / f x)
  (h3 : ∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) :
  f (Real.log 54 / Real.log 3) = -3 / 2 := sorry

end value_of_f_log3_54_l78_78234


namespace min_a_plus_b_l78_78325

theorem min_a_plus_b (a b : ℕ) (h : a * b - 7 * a - 11 * b + 13 = 0) : a + b = 34 :=
by sorry

end min_a_plus_b_l78_78325


namespace composite_two_powers_l78_78759

theorem composite_two_powers (n : ℕ) (h : n > 2) : (¬ nat.prime (2^n - 1)) ∨ (¬ nat.prime (2^n + 1)) :=
sorry

end composite_two_powers_l78_78759


namespace triangle_circle_tangent_ratio_l78_78456

theorem triangle_circle_tangent_ratio 
  (A B C M L K : Point)
  (tangent_circle : Circle)
  (h_tangent : tangent_circle.passes_through A ∧ tangent_circle.tangent_to_side_at BC M)
  (h_intersect_AC : tangent_circle.intersects_AC_at L)
  (h_intersect_AB : tangent_circle.intersects_AB_at K)
  (h_ratio_CM_BM : CM / BM = 3 / 2)
  (h_len_LC_KB : LC = 2 * KB) 
  : AC / AB = 9 / 8 := 
by
  sorry

end triangle_circle_tangent_ratio_l78_78456


namespace problem_statement_l78_78703

variables {A B C H O : Type*} [InnerProductSpace ℝ Type*]
variables (R : ℝ) (angle_A angle_B angle_C : ℝ)
variables (AH BH : ℝ)

-- Assumptions
axiom triangle_ABC : Triangle A B C 
axiom altitudes_intersect_at_H : Altitude A B C H
axiom circumscribed_circle_radius : circumscribed_circle_radius A B C R
axiom angle_ordering : angle_A ≤ angle_B ∧ angle_B ≤ angle_C

-- Statement to prove
theorem problem_statement : AH + BH ≥ 2 * R := sorry

end problem_statement_l78_78703


namespace distinct_fan_count_l78_78089

def max_distinct_fans : Nat :=
  36

theorem distinct_fan_count (n : Nat) (r b : S) (paint_scheme : Fin n → bool) :
  (∀i, r ≠ b → (paint_scheme i = b ∨ paint_scheme i = r)) ∧ 
  (∀i, paint_scheme i ≠ paint_scheme (i + n / 2 % n)) →
  n = 6 →
  max_distinct_fans = 36 :=
by
  sorry

end distinct_fan_count_l78_78089


namespace shaded_area_l78_78564

theorem shaded_area : 
  let base_triangle := 6 -- cm
  let height_triangle := 8 -- cm
  let width_rectangle := 5 -- cm
  let height_rectangle := height_triangle -- sharing the same height
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle -- (1 / 2) * base * height
  let area_rectangle := width_rectangle * height_rectangle -- width * height
  let half_area_rectangle := area_rectangle / 2 -- half of the rectangle area
  let total_shaded_area := area_triangle + half_area_rectangle -- sum of areas
  in total_shaded_area = 44 := 
by 
  sorry

end shaded_area_l78_78564


namespace range_of_a_l78_78239

theorem range_of_a (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ ⦃a b⦄, 0 ≤ a → a ≤ b → f a ≤ f b)
  (h_cond : ∀ a, f a < f (2 * a - 1) → a > 1) :
  ∀ a, f a < f (2 * a - 1) → 1 < a := 
sorry

end range_of_a_l78_78239


namespace original_dining_bill_l78_78115

theorem original_dining_bill (B : ℝ) (h1 : B * 1.15 / 5 = 48.53) : B = 211 := 
sorry

end original_dining_bill_l78_78115


namespace find_sin_minus_cos_find_sin_cube_minus_cos_cube_l78_78220

variable (x : ℝ)

-- Define the conditions
def sin_cos_sum_condition : Prop := Mathlib.sin x + Mathlib.cos x = 1 / 3
def angle_in_second_quadrant : Prop := x > (π / 2) ∧ x < π

-- Statement for part (1)
theorem find_sin_minus_cos (h1 : sin_cos_sum_condition x) (h2 : angle_in_second_quadrant x) :
  Mathlib.sin x - Mathlib.cos x = (sqrt 17) / 3 := sorry

-- Statement for part (2)
theorem find_sin_cube_minus_cos_cube (h1 : sin_cos_sum_condition x) (h2 : angle_in_second_quadrant x) :
  Mathlib.sin x ^ 3 - Mathlib.cos x ^ 3 = (5 * sqrt 17) / 27 := sorry

end find_sin_minus_cos_find_sin_cube_minus_cos_cube_l78_78220


namespace determine_genuine_coins_in_two_weighings_l78_78013

-- Define the context for the 8 coins
structure Coin :=
  (is_genuine : Bool)

def problem_statement (coins : List Coin) (test_coins : List Coin) : Prop :=
  (coins.length = 8) → 
  (coins.countp (λ c, ¬c.is_genuine) = 3) →
  (∀ c, c ∈ coins.filter (λ c, c.is_genuine) → c.is_genuine = true) →
  (∀ c, c ∈ coins.filter (λ c, ¬c.is_genuine) → c.is_genuine = false) →
  (test_coins.length = 3) →
  (test_coins.all (λ c, c ∈ coins)) →
  (∃ w1 w2 : Coin × Coin × Coin × Coin × Coin × Coin → Bool,
      (w1 ((test_coins.head!)::(test_coins.tail).head!, (coins.drop 3).head!, (coins.drop 4).head!, (coins.drop 5).head!, (coins.drop 6).head!, (coins.drop 7).head!)) ≠ 
      (w2 ((test_coins.head!)::(test_coins.tail).head!, (coins.take 2).head!, (coins.take 3).head!, (coins.take 4).head!, (coins.take 5).head!, (coins.take 6).last!)))

-- Problem statement: Given the conditions, can he verify the coins in 2 weighings?
theorem determine_genuine_coins_in_two_weighings (coins : List Coin) (test_coins : List Coin) : problem_statement coins test_coins :=
sorry

end determine_genuine_coins_in_two_weighings_l78_78013


namespace length_BD_correct_l78_78822

noncomputable def length_BD (A B C D : Point) (right_triangle : is_right_triangle A B C)
  (midpoint_D : is_midpoint D B C) (CE_length : length (C, E) = 14) : ℝ :=
  9.9

theorem length_BD_correct (A B C D : Point) (right_triangle : is_right_triangle A B C)
  (midpoint_D : is_midpoint D B C) (CE_length : length (C, E) = 14) :
  length_BD A B C D right_triangle midpoint_D CE_length = 9.9 :=
  sorry

end length_BD_correct_l78_78822


namespace magnitude_of_a_l78_78623

noncomputable def e1 : ℝ³ := sorry
noncomputable def e2 : ℝ³ := sorry

axiom unit_e1 : ‖e1‖ = 1
axiom unit_e2 : ‖e2‖ = 1
axiom dot_product_e1_e2 : inner e1 e2 = 1/2

def a : ℝ³ := 2 • e1 + e2

-- The problem statement to prove
theorem magnitude_of_a : ‖a‖ = Real.sqrt 7 := by
  sorry

end magnitude_of_a_l78_78623


namespace area_of_triangle_AOB_l78_78699

def S (OA OB : ℝ) (angleAOB : ℝ) : ℝ := (1 / 2) * OA * OB * Real.sin angleAOB

theorem area_of_triangle_AOB :
  let OA := 2
  let OB := 4
  let angleAOB := π / 6
  S OA OB angleAOB = 2 :=
by
  let OA := 2
  let OB := 4
  let angleAOB := π / 6
  have h1 : S OA OB angleAOB = 1/2 * OA * OB * Real.sin angleAOB :=
    by rfl
  have h2 : S OA OB angleAOB = 1/2 * 2 * 4 * Real.sin (π / 6) :=
    by rw [h1, Real.sin_pi_div_six]
  have h3 : 1/2 * 2 * 4 * 1/2 = 2 :=
    by norm_num
  exact eq.trans h2 h3

end area_of_triangle_AOB_l78_78699


namespace alan_spent_103_95_l78_78153

-- Define the prices and quantities
def egg_price := 2
def eggs_bought := 20
def chicken_price := 8
def chickens_bought := 6
def milk_price := 4
def milk_bought := 3
def bread_price := 3.5
def bread_bought := 2

-- Define the discount and tax rates
def chicken_discount_condition := 4
def free_chickens_per_discount := 1
def tax_rate := 0.05

-- Define the calculations
def total_egg_cost := eggs_bought * egg_price
def total_chicken_cost := (chickens_bought - (chickens_bought / chicken_discount_condition) * free_chickens_per_discount) * chicken_price
def total_milk_cost := milk_bought * milk_price
def total_bread_cost := bread_bought * bread_price

def total_cost_before_tax := total_egg_cost + total_chicken_cost + total_milk_cost + total_bread_cost
def tax_amount := total_cost_before_tax * tax_rate
def total_cost_including_tax := total_cost_before_tax + tax_amount

-- Proof statement
theorem alan_spent_103_95 : total_cost_including_tax = 103.95 :=
by
  sorry

end alan_spent_103_95_l78_78153


namespace max_snowmen_constructed_l78_78476

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78476


namespace tan_alpha_one_implies_expr_l78_78661

-- Define the given condition: tan(alpha) = 1
def tan_eq_one (α : ℝ) : Prop := Real.tan α = 1

-- Define the expression to be proven: sin(2 * α) - cos(α)^2
def expression (α : ℝ) : ℝ := Real.sin (2 * α) - Real.cos α ^ 2

-- State the theorem: If tan(α) = 1, then sin(2 * α) - cos(α)^2 = 1/2
theorem tan_alpha_one_implies_expr (α : ℝ) (h : tan_eq_one α) : expression α = 1 / 2 := 
sorry

end tan_alpha_one_implies_expr_l78_78661


namespace range_of_f_on_0_1_l78_78251

open Real

noncomputable def omega : ℝ := π / 2
noncomputable def phi : ℝ := π / 6

def f (x : ℝ) : ℝ := sin (omega * x + phi)

theorem range_of_f_on_0_1 :
  (set.range (λ x : ℝ, if 0 ≤ x ∧ x ≤ 1 then f x else 0)) = set.Icc (1 / 2) 1 :=
by
  sorry

end range_of_f_on_0_1_l78_78251


namespace keaton_annual_earnings_l78_78318

def annualEarnings (months: ℕ) (harvestFrequency: ℕ) (pricePerHarvest: ℕ) : ℕ :=
  (months / harvestFrequency) * pricePerHarvest

theorem keaton_annual_earnings :
  let oranges := annualEarnings 12 2 50 in
  let apples := annualEarnings 12 3 30 in
  let peaches := annualEarnings 12 4 45 in
  let blackberries := annualEarnings 12 6 70 in
  oranges + apples + peaches + blackberries = 695 :=
by
  sorry

end keaton_annual_earnings_l78_78318


namespace tan_of_B_in_right_triangle_l78_78194

theorem tan_of_B_in_right_triangle :
  ∀ (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C]
  (AB AC BC : ℝ) (h₁ : AC ≫= 9) (h₂ : AB ≫= 15)
  (h₃ : AC^2 + BC^2 = AB^2),
  let BC := sqrt (AB^2 - AC^2) in tan B = 4 / 3 :=
by
  sorry

end tan_of_B_in_right_triangle_l78_78194


namespace same_solution_m_l78_78949

theorem same_solution_m (m x : ℤ) : 
  (8 - m = 2 * (x + 1)) ∧ (2 * (2 * x - 3) - 1 = 1 - 2 * x) → m = 10 / 3 :=
by
  sorry

end same_solution_m_l78_78949


namespace divisible_2n_minus_3_l78_78758

theorem divisible_2n_minus_3 (n : ℕ) : (2^n - 1)^n - 3 ≡ 0 [MOD 2^n - 3] :=
by
  sorry

end divisible_2n_minus_3_l78_78758


namespace angle_range_l78_78977

variables {a b : ℝ} (ab : ℝ)

noncomputable def angle_between (a b : ℝ) : ℝ := 
cos⁻¹ ((ab) / (a * b))

theorem angle_range (a b ab : ℝ) (ha : a = 2 * b) (h_ab : ab = (2 * b) * b - ab) :
  ∃ θ : ℝ, angle_between a b ab ∈ set.Icc (π / 3) π :=
sorry

end angle_range_l78_78977


namespace inclination_angle_l78_78698

open Real

/-- Given a line l with inclination angle α through point M(-2, -4),
    and curve C with polar equation ρ sin²θ = 2 cosθ,
    if l intersects C at points A and B such that |MA| * |MB| = 40,
    then α = π / 4. -/
theorem inclination_angle (α : ℝ) (C : ℝ → ℝ → Prop)
  (M : ℝ × ℝ) (MA MB : ℝ)
  (line_l : ℝ → ℝ × ℝ)
  (hC : ∀ ρ θ, C ρ θ ↔ ρ * (sin θ)^2 = 2 * cos θ)
  (hM : M = (-2, -4))
  (hline_l : ∀ t, line_l t = (-2 + t * cos α, -4 + t * sin α))
  (h_intersections : ∃ A B : ℝ × ℝ, C A.fst A.snd ∧ C B.fst B.snd
                     ∧ ∃ t1 t2 : ℝ, line_l t1 = A ∧ line_l t2 = B ∧ abs (−2 - A.fst, −4 - A.snd) * abs (−2 - B.fst, −4 - B.snd) = 40) :
  α = π / 4 :=
by sorry

end inclination_angle_l78_78698


namespace max_snowmen_l78_78485

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78485


namespace max_number_of_snowmen_l78_78513

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l78_78513


namespace function_real_roots_probability_correct_l78_78885

noncomputable def function_has_real_roots_probability : Prop :=
  let a : ℝ := -3
  let b : ℝ := 4
  let interval_length := b - a
  let d := 2
  let sub_satisfy_length := 2 * d
  let p := sub_satisfy_length / interval_length
  p = 4 / 7

theorem function_real_roots_probability_correct:
  function_has_real_roots_probability := by
  let a := -3
  let b := 4
  let interval_length := b - a
  let d := 2
  let sub_satisfy_length := 2 * d
  let p := sub_satisfy_length / interval_length
  have result: p = 4 / 7 := sorry
  exact result

end function_real_roots_probability_correct_l78_78885


namespace max_snowmen_l78_78507

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78507


namespace oliver_final_amount_l78_78357

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l78_78357


namespace algebraic_expression_value_l78_78663

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1) * (m - 1) + 2022 = 2024 :=
by
  sorry

end algebraic_expression_value_l78_78663


namespace bus_driver_hours_l78_78104

theorem bus_driver_hours (h : ℕ) (regular_rate : ℕ) (extra_rate1 : ℕ) (extra_rate2 : ℕ) (total_earnings : ℕ)
  (h1 : regular_rate = 14)
  (h2 : extra_rate1 = (14 + (14 * 35 / 100)))
  (h3: extra_rate2 = (14 + (14 * 75 / 100)))
  (h4: total_earnings = 1230)
  (h5: total_earnings = 40 * regular_rate + 10 * extra_rate1 + (h - 50) * extra_rate2)
  (condition : 50 < h) :
  h = 69 :=
by
  sorry

end bus_driver_hours_l78_78104


namespace APEQ_is_square_l78_78685

-- Definitions of the geometric entities and conditions
variables {A B C D E P Q : Type}
variables [affine_space A] [affine_space B] [affine_space C] [affine_space D]
variables [affine_space E] [affine_space P] [affine_space Q]

-- Given conditions
axiom square_ABCD : is_square A B C D
axiom point_E_on_diagonal_BD : on_diagonal E B D
axiom circumcenter_P : is_circumcenter P A B E
axiom circumcenter_Q : is_circumcenter Q A D E

-- Proof statement to prove that APEQ is a square
theorem APEQ_is_square : is_square A P E Q :=
begin
  sorry
end

end APEQ_is_square_l78_78685


namespace g_function_expression_l78_78626

theorem g_function_expression (f g : ℝ → ℝ) (a : ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, g (-x) = g x) (h3 : ∀ x : ℝ, f x + g x = x^2 + a * x + 2 * a - 1) (h4 : f 1 = 2) :
  ∀ t : ℝ, g t = t^2 + 4 * t - 1 :=
by
  sorry

end g_function_expression_l78_78626


namespace max_snowmen_constructed_l78_78475

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78475


namespace problem_statement_l78_78625

variable {x y z : ℝ}

theorem problem_statement (h : x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0)
  (hne : ¬(x = y ∧ y = z)) (hpos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x + y + z = 3) ∧ (x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6) :=
sorry

end problem_statement_l78_78625


namespace convert_base10_to_base9_l78_78173

theorem convert_base10_to_base9 : 
  (2 * 9^3 + 6 * 9^2 + 7 * 9^1 + 7 * 9^0) = 2014 :=
by
  sorry

end convert_base10_to_base9_l78_78173


namespace largest_25_supporting_X_l78_78121

def is_25_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i : ℝ) ∈ Set.Ico (- (1 / 2) : ℝ) ((25 : ℝ) / 2 + (1 / 2)) → 
  ∃ i, |a i - 1 / 2| ≥ X

theorem largest_25_supporting_X : 
  ∃ X : ℝ, is_25_supporting X ∧ 
  (∀ Y : ℝ, Y > X → ¬is_25_supporting Y) ∧ X = 0.02 :=
sorry

end largest_25_supporting_X_l78_78121


namespace inscribed_quad_angle_sum_l78_78964

noncomputable def inscribed_quadrilateral_angle_sum (A B C D P Q K N : Type*) [metric_space A B C D P Q K N] :=
  (inscribed_quadrilateral A B C D) →
  (extension_points_intersection A B A C B D D C P) →
  (extension_points_intersection C D A B A D B C Q) →
  (midpoint K A C) →
  (midpoint N B D) →
  ∑ (angle PKQ) + (angle PNQ) = 180

theorem inscribed_quad_angle_sum
  {A B C D P Q K N : Type*}
  [metric_space A B C D P Q K N]
  (h1 : inscribed_quadrilateral A B C D)
  (h2 : extension_points_intersection A B A C B D D C P)
  (h3 : extension_points_intersection C D A B A D B C Q)
  (h4 : midpoint K A C)
  (h5 : midpoint N B D)
  : ∑ (angle PKQ) + (angle PNQ) = 180 :=
sorry

end inscribed_quad_angle_sum_l78_78964


namespace cupcakes_left_l78_78556

-- Definitions based on the conditions
def total_baked_cupcakes : ℕ := 120
def fraction_given_away : ℚ := 7 / 10
def fraction_eaten : ℚ := 11 / 25

-- Main theorem stating the question and the correct answer
theorem cupcakes_left (baked : ℕ) (given_away : ℚ) (eaten : ℚ) (h_baked : baked = total_baked_cupcakes) (h_given_away : given_away = fraction_given_away) (h_eaten : eaten = fraction_eaten) : ℕ :=
  let given := (baked : ℚ) * given_away in -- Cupcakes given away
  let remaining := (baked : ℚ) - given in -- Cupcakes remaining after giving away
  let eaten_cupcakes := remaining * eaten in -- Cupcakes eaten (note: fraction -> rounding might be required in proof)
  let left := remaining - eaten_cupcakes in -- Cupcakes left after eating
  left.toNat -- Convert to Nat and return (the proof will handle the rounding part)
sorry

end cupcakes_left_l78_78556


namespace factor_quadratic_l78_78583

theorem factor_quadratic (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := 
by 
  sorry

end factor_quadratic_l78_78583


namespace problem_statement_l78_78990

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 5 else (2 : ℝ) ^ x

theorem problem_statement : f (f (1 / 25)) = 1 / 4 :=
by
  sorry

end problem_statement_l78_78990


namespace square_binomial_unique_a_l78_78932

theorem square_binomial_unique_a (a : ℝ) : 
  (∃ r s : ℝ, (ax^2 - 8*x + 16) = (r*x + s)^2) ↔ a = 1 :=
by
  sorry

end square_binomial_unique_a_l78_78932


namespace isosceles_triangle_perimeter_l78_78298

theorem isosceles_triangle_perimeter (a b : ℕ) (h_iso : a = 2 ∨ a = 4 ∧ a = b) : 
  ∃ c : ℕ, (a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a) :=
by {
  -- Definitions from the problem conditions
  assume (a b : ℕ) (h_iso : a = 2 ∨ a = 4 ∧ a = b),
  -- State the solution
  sorry
}

end isosceles_triangle_perimeter_l78_78298


namespace solve_x_l78_78658

theorem solve_x : ∀ (x y : ℝ), (3 * x - y = 7) ∧ (x + 3 * y = 6) → x = 27 / 10 :=
by
  intros x y h
  sorry

end solve_x_l78_78658


namespace imaginary_part_of_complex_number_l78_78788

theorem imaginary_part_of_complex_number 
  (i_im : ℂ) 
  (h : i_im = complex.I * (1 + complex.I)) 
  : complex.im i_im = 1 :=
by
  sorry

end imaginary_part_of_complex_number_l78_78788


namespace max_snowmen_l78_78481

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l78_78481


namespace total_exercise_time_l78_78311

theorem total_exercise_time :
  let javier_minutes_per_day := 50
  let javier_days := 7
  let sanda_minutes_per_day := 90
  let sanda_days := 3
  (javier_minutes_per_day * javier_days + sanda_minutes_per_day * sanda_days) = 620 :=
by
  sorry

end total_exercise_time_l78_78311


namespace reflex_angle_at_T_l78_78216

-- Assume points P, Q, R, and S are aligned
def aligned (P Q R S : ℝ × ℝ) : Prop :=
  ∃ a b, ∀ x, x = 0 * a + b + (P.1, Q.1, R.1, S.1)

-- Angles given in the problem
def PQT_angle : ℝ := 150
def RTS_angle : ℝ := 70

-- definition of the reflex angle at T
def reflex_angle (angle : ℝ) : ℝ := 360 - angle

theorem reflex_angle_at_T (P Q R S T : ℝ × ℝ) :
  aligned P Q R S → PQT_angle = 150 → RTS_angle = 70 →
  reflex_angle 40 = 320 :=
by
  sorry

end reflex_angle_at_T_l78_78216


namespace range_a_is_nonnegative_l78_78795

noncomputable def f (a x : ℝ) : ℝ := real.log (real.sqrt (a * x^2 + 2 * x - 1))

theorem range_a_is_nonnegative (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y = f a x) → a ∈ set.Ici 0 :=
by sorry

end range_a_is_nonnegative_l78_78795


namespace largest_supporting_25_X_l78_78126

def is_supporting_25 (X : ℝ) : Prop :=
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ X

theorem largest_supporting_25_X :
  ∀ (a : Fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, abs (a i - 1 / 2) ≥ 0.02 :=
by {
  sorry
}

end largest_supporting_25_X_l78_78126


namespace smallest_d_squared_l78_78363

noncomputable def z : ℂ := sorry
def area := complex.abs (complex.sin (2 * complex.arg z))
def d : ℝ := complex.abs (z + 1 / z)

theorem smallest_d_squared (h : complex.re z > 0) (h_area : area = 45 / 47) : d ^ 2 = 78 / 47 := sorry

end smallest_d_squared_l78_78363


namespace min_area_quad_l78_78681

theorem min_area_quad (s : ℝ) (x : ℝ) (h : s = 1) (he : ∀ (E A A₁ : ℝ × ℝ × ℝ), E = (0, 0, x) ∧ A₁ = (0, 0, s))
  (hf : ∀ (F C₁ : ℝ × ℝ × ℝ), F = (1, 1, 1-x) ∧ C₁ = (1, 1, 1)) :
  ∃ xmin, ∀ (x : ℝ), s = 1 → x = 1/2 → area E B F D₁ = √6/2 := by
  sorry

noncomputable def area (E B F D₁ : ℝ) : ℝ :=
  let EB := (1, 0, 0 - x)
  let FD₁ := (0, 1, 1 - 1 + x)
  (cross_product EB FD₁).norm / 2
  
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

#eval min_area_quad 1 (1 / 2) (rfl) (by rw [he, hf]) sorry

end min_area_quad_l78_78681


namespace bees_second_day_l78_78748

-- Define the number of bees on the first day
def bees_on_first_day : ℕ := 144 

-- Define the multiplier for the second day
def multiplier : ℕ := 3

-- Define the number of bees on the second day
def bees_on_second_day : ℕ := bees_on_first_day * multiplier

-- Theorem stating the number of bees seen on the second day
theorem bees_second_day : bees_on_second_day = 432 := by
  -- Proof is pending.
  sorry

end bees_second_day_l78_78748


namespace middle_circle_radius_l78_78303

theorem middle_circle_radius 
  (r1 r3 : ℝ) 
  (geometric_sequence: ∃ r2 : ℝ, r2 ^ 2 = r1 * r3) 
  (r1_val : r1 = 5) 
  (r3_val : r3 = 20) 
  : ∃ r2 : ℝ, r2 = 10 := 
by
  sorry

end middle_circle_radius_l78_78303


namespace DustStormCoverage_l78_78112

variable (TotalArea : ℕ) (UntouchedArea : ℕ)

def PrairieTotalArea : TotalArea = 65057 := by rfl
def PrairieUntouchedArea : UntouchedArea = 522 := by rfl

theorem DustStormCoverage (hTotal : TotalArea = 65057) (hUntouched : UntouchedArea = 522) : 
  TotalArea - UntouchedArea = 64535 :=
by {
  rw [hTotal, hUntouched],
  exact rfl,
}

end DustStormCoverage_l78_78112


namespace marked_side_returns_to_original_position_l78_78538

theorem marked_side_returns_to_original_position (triangle : Type) (marked_side : ℚ) 
(reflect : triangle → triangle) 
(initial_position : triangle = reflect (reflect (reflect (reflect triangle)))) : 
  marked_side = marked_side :=
by 
  -- Assume the necessary conditions
  sorry

end marked_side_returns_to_original_position_l78_78538


namespace max_distinct_fans_l78_78085

theorem max_distinct_fans : 
  let sectors := 6,
      initial_configs := 2^sectors,
      unchanged_configs := 8,
      unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  in unique_configs = 36 :=
by
  let sectors := 6
  let initial_configs := 2^sectors
  let unchanged_configs := 8
  let unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  have h : unique_configs = 36 := by sorry
  exact h

end max_distinct_fans_l78_78085


namespace max_snowmen_l78_78504

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l78_78504


namespace sum_log_identity_l78_78166

noncomputable def summand (k : ℕ) : ℝ :=
  real.log (1 + (1 / k.to_real)) / real.log 3 * real.log 3 / real.log k * real.log 3 / real.log (k + 1)

theorem sum_log_identity : 
  ∑ k in finset.range(48) + 3, summand k = 1 - 1 / real.log 51 := 
begin
  sorry,
end

end sum_log_identity_l78_78166


namespace count_nonnegative_integers_l78_78268

theorem count_nonnegative_integers : 
  ∃ (count: ℕ), count = 1094 ∧ 
  ∀ n, 0 <= n ∧ n <= 1093 → 
    ∃ (b : fin 7 → {-1, 0, 1}), 
      n = (b 0) * 3^0 + 
          (b 1) * 3^1 + 
          (b 2) * 3^2 + 
          (b 3) * 3^3 + 
          (b 4) * 3^4 + 
          (b 5) * 3^5 + 
          (b 6) * 3^6 :=
begin
  sorry
end

end count_nonnegative_integers_l78_78268


namespace transform_sum_correctness_l78_78914

-- Define the transformation of digits when flipped upside down
def flip_digits (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => 0  -- treating other digits as invalid or zero when flipped

-- Function to transform a list of numbers by flipping their digits
def transform_list (l : List ℕ) : List ℕ :=
  l.map flip_digits

-- Define the original and the expected sum after transformation
def original_sequence : List ℕ := [1, 9, 1, 8]
def transformed_sum : ℕ := original_sequence.reduce (+)

-- The theorem to prove
theorem transform_sum_correctness :
  transformed_sum = 19 :=
by
  -- Proof would go here
  sorry

end transform_sum_correctness_l78_78914


namespace f_value_2015_l78_78976

noncomputable def f : ℝ → ℝ := sorry 
-- This is for defining the function which would encapsulate periodicity and the given sections

theorem f_value_2015 : f(2015) = -2 :=
begin
  -- conditions
  assume h_periodic : ∀ x, f(x + 4) = f(x),
  assume h_odd : ∀ x, f(-x) = -f(x),
  assume h_given : ∀ x, (0 < x ∧ x ≤ 2) → f(x) = 2^x + log x / log 2,
  -- proof parts will go here (omitted, includes use of given conditions to prove the result)
  sorry,
end

end f_value_2015_l78_78976


namespace function_A_is_periodic_and_even_l78_78157

-- Define the conditions and the question
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def func_A (x : ℝ) : ℝ := sin (2 * x - π / 2)

-- State the theorem
theorem function_A_is_periodic_and_even :
  is_periodic func_A π ∧ is_even func_A :=
sorry

end function_A_is_periodic_and_even_l78_78157


namespace distance_C_to_plane_MB1ND_equals_sqrt6_div3_l78_78900

noncomputable def distance_from_C_to_plane_MB1ND {α : Type*} [normed_field α] 
    (C M B1 N D D1 C1 A B : EuclideanSpace α) [is_midpoint M D1 C1] [is_midpoint N A B] 
    [cube A C1] : α :=
  sorry  -- Placeholder for the actual proof

-- Statement in Lean definition form
theorem distance_C_to_plane_MB1ND_equals_sqrt6_div3 (C M B1 N D D1 C1 A B : EuclideanSpace ℝ)
    [is_midpoint M D1 C1] [is_midpoint N A B] [cube A C1] :
  distance_from_C_to_plane_MB1ND C M B1 N D D1 C1 A B = real.sqrt 6 / 3 :=
sorry

end distance_C_to_plane_MB1ND_equals_sqrt6_div3_l78_78900


namespace max_distance_bicycle_l78_78101

theorem max_distance_bicycle (front_tire_last : ℕ) (rear_tire_last : ℕ) :
  front_tire_last = 5000 ∧ rear_tire_last = 3000 →
  ∃ (max_distance : ℕ), max_distance = 3750 :=
by
  sorry

end max_distance_bicycle_l78_78101


namespace options_proof_l78_78241

theorem options_proof (f : ℝ → ℝ) (h_deriv : ∀ x, deriv f x - f x = exp x) :
  e * f 1 < f 2 ∧ e^3 * f (-1) < f 2 ∧ e * f 0 < f 1 :=
by
  -- Proving each of the options should be handled here
  sorry

end options_proof_l78_78241


namespace words_memorized_on_fourth_day_l78_78266

-- Definitions for the conditions
def first_three_days_words (k : ℕ) : ℕ := 3 * k
def last_four_days_words (k : ℕ) : ℕ := 4 * k
def fourth_day_words (k : ℕ) (a : ℕ) : ℕ := a
def last_three_days_words (k : ℕ) (a : ℕ) : ℕ := last_four_days_words k - a

-- Problem Statement
theorem words_memorized_on_fourth_day {k a : ℕ} (h1 : first_three_days_words k + last_four_days_words k > 100)
    (h2 : first_three_days_words k * 6 = 5 * (4 * k - a))
    (h3 : 21 * (2 * k / 3) = 100) : 
    a = 10 :=
by 
  sorry

end words_memorized_on_fourth_day_l78_78266


namespace trapezoid_PQ_length_l78_78306

theorem trapezoid_PQ_length
  (A B C D P Q : Point)
  (BC AD : Line)
  (point_A_on_AD : A ∈ AD)
  (point_D_on_AD : D ∈ AD)
  (point_P_on_BC : P ∈ BC)
  (point_Q_on_AD : Q ∈ AD)
  (id_midpoint_P : ∀ (P Q midpoint), midpoint = (P + Q) / 2 → midpoint)
  (BC_parallel_AD : ∥ BC ∥ AD)
  (length_BC : dist B C = 800)
  (length_AD : dist A D = 1800)
  (angle_A_is_45 : ∠ A = 45 °)
  (angle_D_is_45 : ∠ D = 45 °)
  : dist P Q = 500 := sorry

end trapezoid_PQ_length_l78_78306


namespace hexagon_midpoints_equilateral_l78_78877

noncomputable def inscribed_hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : Prop :=
  ∀ (M N P : ℝ), 
    true

theorem hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : 
  inscribed_hexagon_midpoints_equilateral r h hex :=
sorry

end hexagon_midpoints_equilateral_l78_78877


namespace max_distinct_fans_l78_78084

theorem max_distinct_fans : 
  let sectors := 6,
      initial_configs := 2^sectors,
      unchanged_configs := 8,
      unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  in unique_configs = 36 :=
by
  let sectors := 6
  let initial_configs := 2^sectors
  let unchanged_configs := 8
  let unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  have h : unique_configs = 36 := by sorry
  exact h

end max_distinct_fans_l78_78084


namespace solution_statement_l78_78924

noncomputable def problem_statement (A B C D P : Point) : Prop :=
  Convex A B C D ∧
  Inside P A B C D ∧
  Angle A P D = 1 * Angle B P A ∧
  Angle B P A = 2 * Angle D P A ∧
  Angle C B P = 1 * Angle B A P ∧
  Angle B A P = 2 * Angle B P C ∧
  Concurrent (AngleBisector A D P) (AngleBisector P C B) (PerpendicularBisector A B)

theorem solution_statement (A B C D P : Point) (h : problem_statement A B C D P) : 
  Concurrent (AngleBisector A D P) (AngleBisector P C B) (PerpendicularBisector A B) :=
sorry

end solution_statement_l78_78924


namespace number_of_insects_l78_78561

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 48) (h2 : legs_per_insect = 6) : (total_legs / legs_per_insect) = 8 := by
  sorry

end number_of_insects_l78_78561


namespace correct_option_D_l78_78844

theorem correct_option_D (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by sorry

end correct_option_D_l78_78844


namespace division_problem_l78_78047

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l78_78047


namespace original_cost_price_l78_78532

theorem original_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1275) 
  (h2 : loss_percentage = 15) 
  (h3 : SP = (1 - loss_percentage / 100) * C) : 
  C = 1500 := 
by 
  sorry

end original_cost_price_l78_78532


namespace correct_number_of_valid_statements_is_zero_l78_78898

def Statement1 (L₁ L₂ : Type) (P : Type) [HasParallel L₁ P] [HasParallel L₂ P] : Prop := 
  Parallel L₁ P ∧ Parallel L₂ P → Parallel L₁ L₂

def Statement2 (L₁ L₂ : Type) : Prop :=
  (∀ x, ¬ (x ∈ L₁ ∧ x ∈ L₂)) → Parallel L₁ L₂

def Statement3 (L₁ L₂ L₃ : Type) [HasPerpendicular L₁ L₃] [HasPerpendicular L₂ L₃] : Prop :=
  Perpendicular L₁ L₃ ∧ Perpendicular L₂ L₃ → Parallel L₁ L₂

def Statement4 (L : Type) (P : Type) : Prop :=
  (∀ L₀, L₀ ∈ P → ¬ (∃ x, x ∈ L ∧ x ∈ L₀)) → Parallel L P

theorem correct_number_of_valid_statements_is_zero :
  ∀ (L₁ L₂ L₃ L P : Type) [HasParallel L₁ P] [HasParallel L₂ P] [HasPerpendicular L₁ L₃] [HasPerpendicular L₂ L₃],
  ¬ Statement1 L₁ L₂ P ∧ ¬ Statement2 L₁ L₂ ∧ ¬ Statement3 L₁ L₂ L₃ ∧ ¬ Statement4 L P → 0 = 0 := 
by
  intro L₁ L₂ L₃ L P h1 h2 h3 h4
  apply eq.refl 0
sorry

end correct_number_of_valid_statements_is_zero_l78_78898


namespace problem2_l78_78222

theorem problem2 (x y : ℝ) (h1 : x^2 + x * y = 3) (h2 : x * y + y^2 = -2) : 
  2 * x^2 - x * y - 3 * y^2 = 12 := 
by 
  sorry

end problem2_l78_78222


namespace news_spread_time_l78_78058

theorem news_spread_time (n : ℕ) (m : ℕ) :
  (2^m < n ∧ n < 2^(m+k+1) ∧ (n % 2 = 1) ∧ n % 2 = 1) →
  ∃ t : ℕ, t = (if n % 2 = 1 then m+2 else m+1) := 
sorry

end news_spread_time_l78_78058


namespace range_of_k_l78_78260

theorem range_of_k (k x y : ℝ) 
  (h₁ : 2 * x - y = k + 1) 
  (h₂ : x - y = -3) 
  (h₃ : x + y > 2) : k > -4.5 :=
sorry

end range_of_k_l78_78260


namespace correctNumberOfProps_l78_78782

def prop1 : Prop :=
  ∀ (L1 L2 : Line) (P : Point),
    L1 ≠ L2 ∧ L1.intersect L2 = P → (L1.angle P = L2.angle P) ↔ (L1.isPerpendicularTo L2)

def prop2 : Prop :=
  ∀ (L1 L2 : Line) (P : Point) (A B : Angle),
    L1 ≠ L2 ∧ L1.angle P = A ∧ L2.angle P = B ∧ A = B ↔ L1.isPerpendicularTo L2

def prop3 : Prop :=
  ∀ (L1 L2 : Line) (A B : Angle),
    L1.parallelTo L2 ∧ (A = L1.angle) ∧ (B = L2.angle) ∧ A = B ↔ (L1.angleBisector.isPerpendicularTo L2.angleBisector)

def prop4 : Prop :=
  ∀ (L1 L2 : Line) (A B : Angle),
    (A = L1.angle) ∧ (B = L2.angle) ∧ A + B = 90 ↔ L1.angleBisector.isPerpendicularTo L2.angleBisector

def numberOfCorrectProps : Nat :=
  if prop1 = false then 0 else 1
    + if prop2 = true then 1 else 0
    + if prop3 = false then 0 else 1
    + if prop4 = true then 1 else 0

theorem correctNumberOfProps : numberOfCorrectProps = 2 := by
  -- Placeholder for steps that verify each of the propositions
  sorry

end correctNumberOfProps_l78_78782


namespace sqrt_74_between_8_and_9_product_of_consecutive_integers_l78_78011

theorem sqrt_74_between_8_and_9 : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9 := sorry

theorem product_of_consecutive_integers (h : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9) : 8 * 9 = 72 := by
  have h1 : 8 < Real.sqrt 74 := And.left h
  have h2 : Real.sqrt 74 < 9 := And.right h
  calc
    8 * 9 = 72 := by norm_num

end sqrt_74_between_8_and_9_product_of_consecutive_integers_l78_78011


namespace correct_option_l78_78050

-- Define the options as propositions
def OptionA (a : ℕ) := a ^ 3 * a ^ 5 = a ^ 15
def OptionB (a : ℕ) := a ^ 8 / a ^ 2 = a ^ 4
def OptionC (a : ℕ) := a ^ 2 + a ^ 3 = a ^ 5
def OptionD (a : ℕ) := 3 * a - a = 2 * a

-- Prove that Option D is the only correct statement
theorem correct_option (a : ℕ) : OptionD a ∧ ¬OptionA a ∧ ¬OptionB a ∧ ¬OptionC a :=
by
  sorry

end correct_option_l78_78050


namespace solve_equation_l78_78862

variable (x : ℝ)

theorem solve_equation (h : x * (x - 4) = x - 6) : x = 2 ∨ x = 3 := 
sorry

end solve_equation_l78_78862


namespace oliver_total_money_l78_78352

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l78_78352


namespace max_distinct_fans_l78_78067

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l78_78067


namespace line_intersection_l78_78186

/-- Prove the intersection of the lines given by the equations
    8x - 5y = 10 and 3x + 2y = 1 is (25/31, -22/31) -/
theorem line_intersection :
  ∃ (x y : ℚ), 8 * x - 5 * y = 10 ∧ 3 * x + 2 * y = 1 ∧ x = 25 / 31 ∧ y = -22 / 31 :=
by
  sorry

end line_intersection_l78_78186


namespace compare_f_values_l78_78250

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1 / x

def a : ℝ := f (1 / 3)
def b : ℝ := f Real.pi
def c : ℝ := f 5

theorem compare_f_values : c < b ∧ b < a := by
  sorry

end compare_f_values_l78_78250


namespace rectangle_area_x_value_l78_78292

theorem rectangle_area_x_value :
  ∃ (x : ℝ), (let coords := [(-9, 1), (1, 1), (1, -8), (-9, -8)] in 
                 (90 = 9 * abs ((x - 1))) ∧ 
                 coords = [(x, 1), (1, 1), (1, -8), (x, -8)] ∧
                 x = -9 ) :=
sorry

end rectangle_area_x_value_l78_78292


namespace teresa_die_sides_l78_78769

-- Definitions based on the conditions
def Die := { dice : fin 8 // dice ≠ 0 }  -- Definitions for a finite set representing die faces

def smallest_positive_unrolled (rolled: finset (fin 8)) : fin 8 :=
  fin.succ_above (nat.find (λ k, (k : ℕ) + 1 ∉ rolled))

-- Function representing the probability
def probability_last_roll_is_7 (p : ℚ := 1/4) (n : ℕ := 7) : ℚ :=
  p

-- The statement we are proving:
theorem teresa_die_sides (p : ℚ := 1/4) (a : ℕ := 1) (b : ℕ := 4) : 100 * a + b = 104 → (probability_last_roll_is_7 p = a / b) :=
by
  sorry

end teresa_die_sides_l78_78769


namespace inequality_solution_set_l78_78402

theorem inequality_solution_set (a : ℝ) : (-16 < a ∧ a ≤ 0) ↔ (∀ x : ℝ, a * x^2 + a * x - 4 < 0) :=
by
  sorry

end inequality_solution_set_l78_78402


namespace decreasing_on_interval_l78_78635

variable {x m n : ℝ}

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := |x^2 - 2 * m * x + n|

theorem decreasing_on_interval
  (h : ∀ x, f x m n = |x^2 - 2 * m * x + n|)
  (h_cond : m^2 - n ≤ 0) :
  ∀ x y, x ≤ y → y ≤ m → f y m n ≤ f x m n :=
sorry

end decreasing_on_interval_l78_78635


namespace cos_sq_minus_sin_sq_l78_78956

variable (α β : ℝ)

theorem cos_sq_minus_sin_sq (h : Real.cos (α + β) * Real.cos (α - β) = 1 / 3) :
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1 / 3 :=
sorry

end cos_sq_minus_sin_sq_l78_78956


namespace QF_distance_l78_78256

noncomputable def parabola := {p : ℝ // p > 0}

def point_P (p : parabola) (m : ℝ) : ℝ × ℝ := (m, m^2 / (2 * p))
def point_Q (p : parabola) (m : ℝ) : ℝ × ℝ := (0, - m^2 / (2 * p))
def focus_F (p : parabola) : ℝ × ℝ := (0, p / 2)

def distance (A B : ℝ × ℝ) : ℝ :=
  ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2)

theorem QF_distance (p : parabola) (m : ℝ) (h : distance (point_P p.val m) (focus_F p.val) = 5) : distance (point_Q p.val m) (focus_F p.val) = 5 := 
by
  sorry -- Proof to be constructed

end QF_distance_l78_78256


namespace find_b_l78_78701

theorem find_b (b : ℝ) :
  (∃ x₁ x₂ : ℝ, y₁ = 2^x₁ ∧ y₂ = log 2 x₂ ∧ y₁ = -x₁ + b ∧ y₂ = -x₂ + b ∧ x₁ + x₂ = 6) → b = 6 :=
sorry

end find_b_l78_78701


namespace girl_speed_l78_78114

theorem girl_speed (distance time : ℝ) (h₁ : distance = 128) (h₂ : time = 32) : distance / time = 4 := 
by 
  rw [h₁, h₂]
  norm_num

end girl_speed_l78_78114


namespace evaluate_expression_zero_l78_78192

variable (b : ℚ)
variable h_b : b = 4 / 3

theorem evaluate_expression_zero : (6 * b^2 - 8 * b + 3) * (3 * b - 4) = 0 :=
by
  rw [h_b]
  sorry

end evaluate_expression_zero_l78_78192


namespace collinear_midpoints_implies_right_l78_78829

-- Definition of a triangle (as a structure with points A, B, C)
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Definition of collinear points
def collinear (P Q R : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), (Q.1 - P.1) = k * (R.1 - P.1) ∧ (Q.2 - P.2) = k * (R.2 - P.2)

-- Definition of midpoint
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Definition of altitude (omitting specific construction, focus on conceptual altitude)
def altitude_midpoint (T : Triangle) : ℝ × ℝ :=
  -- This would be the formal midpoint of an altitude from a vertex to the opposite side
  sorry

-- Theorem statement
theorem collinear_midpoints_implies_right (T : Triangle)
  (h : collinear (altitude_midpoint T) (altitude_midpoint T) (altitude_midpoint T)) : 
  -- In a right triangle, one of the angles is 90 degrees.
  ∃ α β γ : ℝ, α + β + γ = 180 ∧ (α = 90 ∨ β = 90 ∨ γ = 90) :=
sorry

end collinear_midpoints_implies_right_l78_78829


namespace cricket_matches_total_l78_78386

theorem cricket_matches_total 
  (N : ℕ)
  (avg_total : ℕ → ℕ)
  (avg_first_8 : ℕ)
  (avg_last_4 : ℕ) 
  (h1 : avg_total N = 48)
  (h2 : avg_first_8 = 40)
  (h3 : avg_last_4 = 64) 
  (h_sum : (avg_first_8 * 8 + avg_last_4 * 4 = avg_total N * N)) :
  N = 12 := 
  sorry

end cricket_matches_total_l78_78386


namespace monotonic_intervals_range_of_a_l78_78630

def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1/x) - |x - 1/x|

theorem monotonic_intervals (a : ℝ) (x : ℝ) : 
  a = 1/2 → 
  ((x ∈ (-∞, -1] ∪ (0, 1]) → monotone (f a)) ∧
   (x ∈ [-1, 0) ∪ [1, ∞)) → antitone (f a)) :=
by 
  sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f a x ≥ (1/2) * x) ↔ a ≥ 3/2 :=
by 
  sorry

end monotonic_intervals_range_of_a_l78_78630


namespace solve_for_x_l78_78605

theorem solve_for_x : ∃ x : ℝ, 42 - 3 * x = 18 ∧ x = 8 :=
by {
  use 8,
  split,
  { linarith, },
  { refl, }
}

end solve_for_x_l78_78605


namespace union_of_sets_eq_A_l78_78338

noncomputable def A : Set ℝ := {x | x / ((x + 1) * (x - 4)) < 0}
noncomputable def B : Set ℝ := {x | Real.log x < 1}

theorem union_of_sets_eq_A: A ∪ B = A := by
  sorry

end union_of_sets_eq_A_l78_78338


namespace driver_catch_train_l78_78098

theorem driver_catch_train :
  ∀ (distance_km : ℕ) (time_minutes : ℕ) (initial_speed_kmh : ℕ),
  distance_km = 2 →
  time_minutes = 2 →
  initial_speed_kmh = 30 →
  let initial_speed_mph := initial_speed_kmh * 1000 / 60 in
  let remaining_distance_m := (distance_km * 1000) - initial_speed_mph in
  (remaining_distance_m * 60 / 1000) = 90 :=
by
  intros distance_km time_minutes initial_speed_kmh.
  intros h_distance h_time h_speed.
  let initial_speed_mph := initial_speed_kmh * 1000 / 60
  let remaining_distance_m := (distance_km * 1000) - initial_speed_mph
  calc (remaining_distance_m * 60 / 1000) = 90 : sorry

end driver_catch_train_l78_78098


namespace problem_A_l78_78760

-- Define the problem
variable (A B C D F E : Point)
variable (a b c : ℝ)

-- Given conditions
-- 1. Quadrilateral ABCD is a trapezoid with bases AB = 8 units and CD = 5 units.
def is_trapezoid : Prop :=
  is_parallel (A, B) (C, D) ∧ AB = 8 ∧ CD = 5
  
-- 2. Non-parallel sides AD and BC are equal.
def non_parallel_sides_equal : Prop :=
  AD = BC

-- 3. Drop perpendiculars from D and C to AB, meeting at points F and E.
def perpendiculars_from_D_and_C : Prop :=
  is_perpendicular (D, F) (A, B) ∧ is_perpendicular (C, E) (A, B)

-- 4. Calculate the length of segment EF.
def length_EF : Prop :=
  segment_length (E, F) = 24/13

theorem problem_A : is_trapezoid A B C D ∧ non_parallel_sides_equal A D B C ∧ perpendiculars_from_D_and_C D C A B F E → length_EF E F :=
begin
  sorry
end

end problem_A_l78_78760


namespace propositions_correct_l78_78096

variable (x : ℝ)
variable (p q : Prop)

theorem propositions_correct :
  (∀ (x : ℝ), x^2 + 2*x + 7 > 0) ∧
  (∃ (x : ℝ), x + 1 > 0) ∧
  ((¬ p → ¬ q) → (q → p)) ∧
  (let p := (∀ s, ∅ ⊆ s) in
   let q := (0 ∈ ∅) in
   (p ∨ q) ∧ ¬ (p ∧ q)) :=
by
  exact (
    and.intro
      (by intro x; linarith)
      (by use 0; linarith)
      (by intros h q; classical; contrapose! h; assumption)
      (by
        let p := ∀ s, ∅ ⊆ s
        let q := 0 ∈ ∅
        have hp : p := by intros s; exact set.empty_subset s
        have hq : ¬ q := by finish
        exact and.intro (or.inl hp) (not_and_of_not_right q hq)))

end propositions_correct_l78_78096


namespace prove_perpendicular_AB_CD_l78_78413

structure Point :=
  (x : ℝ)
  (y : ℝ)

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

noncomputable def is_perpendicular_to_x_axis (p1 p2 : Point) : Prop :=
  slope p1 p2 = -1

def A : Point := { x := -2, y := 2 }
def B : Point := { x := 2, y := -2 }
def C : Point := { x := 2, y := 1 }
def D : Point := { x := 1, y := 2 }

theorem prove_perpendicular_AB_CD :
  is_perpendicular_to_x_axis A B ∧ is_perpendicular_to_x_axis C D :=
by
  sorry

end prove_perpendicular_AB_CD_l78_78413


namespace sum_possible_values_of_a_l78_78185

section
  def f (a : ℤ) := λ x : ℤ, x^2 - a * x + 3 * a

  theorem sum_possible_values_of_a
    (a : ℤ)
    (h : ∀ r s : ℤ, f a r = 0 ∧ f a s = 0 → r + s = a ∧ r * s = 3 * a) :
    a ∈ {25, 16, 12} → 25 + 16 + 12 = 53 :=
  by
    sorry
end

end sum_possible_values_of_a_l78_78185


namespace max_fans_theorem_l78_78071

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l78_78071


namespace old_friends_games_l78_78317

theorem old_friends_games (total_games : ℕ) (new_friends_games : ℕ)
  (h1 : total_games = 141)
  (h2 : new_friends_games = 88)
  (total_friends_games : ℕ)
  (h3 : total_friends_games = 141) :
  (total_friends_games - new_friends_games) = 53 :=
by
  have h4 : total_friends_games = total_games := by rw [h1, h3]
  rw [h4, h2]
  exact Nat.sub_eq_of_eq_add' rfl⟩

# Print old_friends_games

end old_friends_games_l78_78317


namespace files_per_folder_l78_78349

-- Define the conditions
def initial_files : ℕ := 43
def deleted_files : ℕ := 31
def num_folders : ℕ := 2

-- Define the final problem statement
theorem files_per_folder :
  (initial_files - deleted_files) / num_folders = 6 :=
by
  -- proof would go here
  sorry

end files_per_folder_l78_78349


namespace f_odd_f_increasing_on_2_to_inf_f_range_m4_on_neg8_neg2_l78_78221

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem f_odd (m : ℝ) : ∀ x : ℝ, x ≠ 0 → f (-x) m = -f x m := 
by
  intro x hx
  have h1 : f (-x) m = -x + m / (-x) := rfl
  have h2 : -f x m = -(x + m / x) := rfl
  rw [h1, h2]
  ring_nf
  rw div_neg
  ring

theorem f_increasing_on_2_to_inf :
  ∀ x : ℝ, x > 2 → f x 4 = x + 4 / x ∧ ∀ y : ℝ, y > x → f y 4 > f x 4 :=
by
  intro x hx
  have h_fx : f x 4 = x + 4 / x := rfl
  split
  · exact h_fx
  · intro y hy
    have h_diff : (f y 4 - f x 4) = (y - x) + 4 * (1 / y - 1 / x) := by ring_nf
    have h_pos : y - x > 0 := sub_pos.mpr hy
    have h_inequality : (4 * (1 / y - 1 / x)) > 0 := by
      have h1 : 1 / y < 1 / x := by
        apply one_div_lt_one_div_of_lt hx hy
        exact zero_lt_two.trans hx
      ring_nf
      linarith
    linarith

theorem f_range_m4_on_neg8_neg2 :
  ∀ x : ℝ, -8 ≤ x ∧ x ≤ -2 → -10 ≤ f x 4 ∧ f x 4 ≤ -4 :=
by
  intro x h_range
  have h1 : f x 4 = x + 4 / x := rfl
  split
  · have h_lower := calc
      f (-8) 4 = -10 := rfl
      f x 4 ≥ f (-8) 4 := by
        have h_inc : ∀ a b : ℝ, -8 ≤ a → a ≤ b → b ≤ -2 → a ≤ b → f a 4 ≤ f b 4 := 
          by 
            intro a b h_neg8 a_le_b b_le_neg2 le_ab
            apply f_increasing_on_2_to_inf 
            linarith only [le_ab]
        exact h_inc (-8) x (-by linarith) (-2) (-8) h_range
      -10
      
    exact h_lower
      
  · have h_upper := calc
      f (-2) 4 = - 4 := rfl
      f x 4 ≤ f (-2) 4 := by
        have h_inc : ∀ a b : ℝ, a ≤ b → b ≤ -2 → f a 4 ≤ f b 4 := 
          by 
            intro a b le_ab b_le_neg2
            apply f_increasing_on_2_to_inf 
            linarith only [le_ab]
        exact h_inc x (-2) h_range.1 h_range.2
    linarith only [h_upper]
  sorry   -- this part left for further proof steps


end f_odd_f_increasing_on_2_to_inf_f_range_m4_on_neg8_neg2_l78_78221


namespace determine_plane_correct_condition_l78_78429

-- Definitions of conditions for determining a plane
def three_points_in_space (p1 p2 p3 : point) : Prop := 
  ∃ (p1 p2 p3 : point), ¬ collinear p1 p2 p3

def line_and_point (l : line) (p : point) : Prop := 
  ∃ (l : line) (p : point), ¬ on_line p l

def two_perpendicular_lines (l1 l2 : line) : Prop := 
  ∃ (l1 l2 : line), (perpendicular l1 l2) ∧ (same_plane l1 l2)

def two_parallel_lines (l1 l2 : line) : Prop := 
  ∃ (l1 l2 : line), (parallel l1 l2)

-- Lean statement to prove the correct condition
theorem determine_plane_correct_condition :
  (three_points_in_space p1 p2 p3 → ¬determine_plane p1 p2 p3) ∧ 
  (line_and_point l p → ¬determine_plane l p) ∧ 
  (two_perpendicular_lines l1 l2 → ¬determine_plane l1 l2) ∧ 
  (two_parallel_lines l1 l2 → determine_plane l1 l2) :=
sorry

end determine_plane_correct_condition_l78_78429


namespace perpendiculars_form_square_l78_78880

-- Definitions of points and geometric entities
variables {Point : Type} [AffineGeometry Point]
variables (A B C D : Point) (A1 B1 C1 D1 : Point)

-- Conditions: A, B, C, D form a square; A1, B1, C1, D1 form a parallelogram circumscribed around the square
def is_square (A B C D : Point) : Prop := /* Definition for points forming a square */
def is_parallelogram (A1 B1 C1 D1 : Point) : Prop := /* Definition for points forming a parallelogram */
def circumscribed (A B C D A1 B1 C1 D1 : Point) : Prop := /* Definition for the parallelogram A1B1C1D1 circumscribed around square ABCD */

-- Question: The perpendiculars dropped from A1, B1, C1, D1 to sides of ABCD form a square
theorem perpendiculars_form_square
  (h_square : is_square A B C D)
  (h_parallelogram : is_parallelogram A1 B1 C1 D1)
  (h_circumscribed : circumscribed A B C D A1 B1 C1 D1) :
  ∃ l₁ l₂ l₃ l₄ : Point → Prop,
    (∀ p, l₁ p → ⟪p, _⟫) ∧  -- perpendicular from vertices of parallelogram
    (∀ p, l₂ p → ⟪p, _⟫) ∧
    (∀ p, l₃ p → ⟪p, _⟫) ∧
    (∀ p, l₄ p → ⟪p, _⟫) ∧
    -- Showing l1, l2, l3, l4 form a square
    /* formulation for l₁, l₂, l₃, and l₄ forming a square */
    sorry

end perpendiculars_form_square_l78_78880


namespace average_speed_is_correct_l78_78657

-- Define the speeds
def speed1 : ℝ := 110
def speed2 : ℝ := 88

-- Define the formula for average speed given two speeds
def average_speed (s1 s2 : ℝ) : ℝ :=
  (2 * s1 * s2) / (s1 + s2)

-- State the theorem we want to prove
theorem average_speed_is_correct :
  average_speed speed1 speed2 = 97.78 := by
  sorry

end average_speed_is_correct_l78_78657


namespace existence_of_tangent_circle_l78_78381

variables (A B C D : Point)
variables (AB AD BC CD : Line)
variables (w1 w2 w3 w4 : Circle)

-- Conditions
variables (parallelogram_ABCD : Parallelogram A B C D)
variables (tangent_w1_AB : Tangent w1 AB)
variables (tangent_w1_AD : Tangent w1 AD)
variables (tangent_w2_BC : Tangent w2 BC)
variables (tangent_w2_CD : Tangent w2 CD)
variables (tangent_w3_AD : Tangent w3 AD)
variables (tangent_w3_DC : Tangent w3 DC)
variables (externally_tangent_w3_w1 : ExternallyTangent w3 w1)
variables (externally_tangent_w3_w2 : ExternallyTangent w3 w2)

-- Proof statement
theorem existence_of_tangent_circle :
  ∃ w4, Tangent w4 AB ∧ Tangent w4 BC ∧ ExternallyTangent w4 w1 ∧ ExternallyTangent w4 w2 :=
sorry

end existence_of_tangent_circle_l78_78381


namespace cost_of_candies_l78_78103

-- Definitions based on given conditions
def candy_per_box : ℕ := 30
def cost_per_box : ℕ := 750 -- in cents to avoid dealing with floats
def total_candies_needed : ℕ := 750
def number_of_boxes_needed := total_candies_needed / candy_per_box
def discount_threshold := 20
def discount_rate := 0.10

-- Proof statement
theorem cost_of_candies :
  number_of_boxes_needed > discount_threshold →
  cost_per_box * number_of_boxes_needed * (1 - discount_rate) = 16875 :=
by
  intro h
  sorry

end cost_of_candies_l78_78103


namespace value_of_expression_l78_78010

theorem value_of_expression : (-0.125)^2021 * (8:^2022) = -8 :=
by
  sorry

end value_of_expression_l78_78010


namespace find_x_minus_y_l78_78985

variables {V : Type*} [inner_product_space ℝ V]
variables (OM ON OP : V)
variables (x y : ℝ)

-- Conditions:
def unit_vector (v : V) : Prop := ‖v‖ = 1
def angle_sixty_degrees (u v : V) : Prop := inner u v = 0.5
def op_decomposition (OM ON : V) (x y : ℝ) : V := x • OM + y • ON
def is_right_triangle (M N P : V) (v w : V) : Prop := inner v w = 0

-- Theorem and proof to find the value of x - y:
theorem find_x_minus_y
  (OM_unit : unit_vector OM)
  (ON_unit : unit_vector ON)
  (angle_60 : angle_sixty_degrees OM ON)
  (op_eq : OP = op_decomposition OM ON x y)
  (right_triangle : is_right_triangle OP OM ON (op_decomposition OM ON (x - 1) y) (ON - OM)) :
  (x - y = 1) :=
sorry

end find_x_minus_y_l78_78985


namespace max_fans_theorem_l78_78072

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l78_78072


namespace max_gcd_ab_eq_10_l78_78831

theorem max_gcd_ab_eq_10 (a b : ℕ) (h : a * b = 600) : nat.gcd a b = 10 :=
sorry

end max_gcd_ab_eq_10_l78_78831


namespace general_term_a_n_general_term_b_n_l78_78984

def S6 := 60
def a1 := 5
def d := 2

-- Arithmetic sequence
def a (n : ℕ) : ℕ := 2*n + 3

-- Conditions for S6 and geometric sequence
lemma arithmetic_conditions : 6*a1 + 15*d = S6 ∧ a1*(a1 + 20*d) = (a1 + 5*d)^2 :=
by {
  -- sorry is used to indicate the steps of the proof that we are not providing
  sorry
}

-- General term formula of sequence a_n
theorem general_term_a_n (n : ℕ) : a n = 2*n + 3 :=
begin
  exact rfl,
end

-- Given b_{n+1} - b_n = a_n for n ∈ ℕ_+ and b1 = 3
def b (n : ℕ) : ℕ := n*(n + 2)

-- General term formula of sequence b_n
theorem general_term_b_n (n : ℕ) : b n = n*(n + 2) :=
by {
  -- sorry is used to indicate the steps of the proof that we are not providing
  sorry
}

end general_term_a_n_general_term_b_n_l78_78984


namespace chord_length_l78_78394

theorem chord_length (x y : ℝ) :
  (x^2 + y^2 - 2 * x - 4 * y = 0) →
  (x + 2 * y - 5 + Real.sqrt 5 = 0) →
  ∃ l, l = 4 :=
by
  intros h_circle h_line
  sorry

end chord_length_l78_78394


namespace quadrilateral_area_l78_78028

noncomputable def area_of_quadrilateral (a b c d: ℝ) (alpha: ℝ) : ℝ :=
  let e_sq := a^2 + d^2 - 2 * a * d * real.cos alpha
  let e := real.sqrt e_sq

  let T1 := 0.5 * a * d * real.sin alpha

  let s := (b + c + e) / 2
  let T2 := real.sqrt (s * (s - b) * (s - c) * (s - e))

  T1 + T2

theorem quadrilateral_area :
  area_of_quadrilateral 52 56 33 39 (112 + 37/60 + 12/3600) = 1774 :=
sorry

end quadrilateral_area_l78_78028


namespace num_correct_propositions_eq_2_l78_78780

-- Declare the propositions as variables.
variables (p1 p2 p3 p4 : Prop)

-- Define each condition from part a)
def condition_1 : Prop := 
  ∀ (a b : ℝ), p1 ↔ (a + b = 180 ∧ a = b) → (a = 90)

def condition_2 : Prop := 
  ∀ (a : ℝ), p2 ↔ (a + a = 180) ∧ (a = 90)

def condition_3 : Prop := 
  ∀ (a b : ℝ), p3 ↔ (a = b) → (some intersection or alignment property, e.g. parallel) ∧ (error inferred)

def condition_4 : Prop := 
  ∀ (a b : ℝ), p4 ↔ (a + b = 90) → (45*2 = 90, so perpendicular)

-- Definition of the resulting proof problem
theorem num_correct_propositions_eq_2
  (h1 : condition_1 p1)
  (h2 : condition_2 p2)
  (h3 : condition_3 p3)
  (h4 : condition_4 p4) :
  (ite p1 1 0 + ite p2 1 0 + ite p3 1 0 + ite p4 1 0) = 2 :=
by
  -- Proof steps would go here, but are omitted by the guidelines
  sorry

end num_correct_propositions_eq_2_l78_78780


namespace option_C_is_quadratic_l78_78049

-- Define the conditions
def option_A (x : ℝ) : Prop := 2 * x = 3
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (4 * x - 3) * (3 * x + 1) = 0
def option_D (x : ℝ) : Prop := (x + 3) * (x - 2) = (x - 2) * (x + 1)

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x, f x = (a * x^2 + b * x + c = 0)) ∧ a ≠ 0

-- The main theorem statement
theorem option_C_is_quadratic : is_quadratic option_C :=
sorry

end option_C_is_quadratic_l78_78049


namespace number_of_three_cell_shapes_l78_78093

theorem number_of_three_cell_shapes (x y : ℕ) (h : 3 * x + 4 * y = 22) : x = 6 :=
sorry

end number_of_three_cell_shapes_l78_78093


namespace max_snowmen_constructed_l78_78473

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l78_78473


namespace number_of_digits_in_Q_l78_78728

-- Define the three numbers
def num1 : ℕ := 93_456_789_000_789
def num2 : ℕ := 37_497_123_456
def num3 : ℕ := 502

-- Define the product Q
def Q : ℕ := num1 * num2 * num3

-- Define a function to count the number of digits in a given number
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log10 n + 1

-- State the theorem to be proven
theorem number_of_digits_in_Q : num_digits Q = 30 := sorry

end number_of_digits_in_Q_l78_78728


namespace max_distinct_fans_l78_78077

-- Define the problem conditions and the main statement
theorem max_distinct_fans : 
  let n := 6  -- Number of sectors per fan
  let configurations := 2^n  -- Total configurations without considering symmetry
  let unchanged_flips := 8  -- Number of configurations unchanged by flipping
  let distinct_configurations := (configurations - unchanged_flips) / 2 + unchanged_flips 
  in 
  distinct_configurations = 36 := by sorry  # We state the answer based on the provided steps


end max_distinct_fans_l78_78077


namespace probability_of_F_l78_78895

theorem probability_of_F (P : String → ℚ) (hD : P "D" = 1/4) (hE : P "E" = 1/3) (hG : P "G" = 1/6) (total : P "D" + P "E" + P "F" + P "G" = 1) :
  P "F" = 1/4 :=
by
  sorry

end probability_of_F_l78_78895


namespace equal_foreign_liquid_l78_78409

variables (a b : ℝ)
hypothesis (h1 : a > b)

theorem equal_foreign_liquid : 
  let foreign_liquid_A := a * (b / (a + b))
  let foreign_liquid_B := (a - b) + (b^2 / (a + b))
  in foreign_liquid_A = foreign_liquid_B :=
by {
  sorry
}

end equal_foreign_liquid_l78_78409


namespace equilateral_triangle_constant_ratio_non_equilateral_variable_ratio_l78_78443

variable {α : Type*} [EuclideanGeometry α] (A B C P : Point α)
variable (A' B' C' : Point α)
variable (equilateral_triangle : Triangle α)

noncomputable def feet_of_perpendicular {α : Type*} [EuclideanGeometry α] (P : Point α) (ABC : Triangle α) 
  : Point α × Point α × Point α := sorry

theorem equilateral_triangle_constant_ratio:
  (is_equilateral (A, B, C)) → 
  (∀ P : Point α, 
    let (A', B', C') := feet_of_perpendicular P (A, B, C)
    in (PA'.dist + PB'.dist + PC'.dist) = (AC'.dist + BA'.dist + CB'.dist) * sqrt 3) := 
sorry

theorem non_equilateral_variable_ratio:
  (is_acute_angled (A, B, C)) → ¬ (is_equilateral (A, B, C)) →
  ¬ (∀ P : Point α, 
    let (A', B', C') := feet_of_perpendicular P (A, B, C)
    in ∃ k : ℝ, (PA'.dist + PB'.dist + PC'.dist) = k * (AC'.dist + BA'.dist + CB'.dist)) :=
sorry

end equilateral_triangle_constant_ratio_non_equilateral_variable_ratio_l78_78443


namespace cos_alpha_plus_beta_l78_78272

noncomputable def eulers_formula (x : ℝ) : ℂ := complex.exp (complex.I * x)

theorem cos_alpha_plus_beta (α β : ℝ) 
  (h1 : eulers_formula α = (4 / 5 : ℂ) - (3 / 5 : ℂ) * complex.I) 
  (h2 : eulers_formula β = (5 / 13 : ℂ) + (12 / 13 : ℂ) * complex.I) : 
  real.cos (α + β) = - (16 / 65 : ℝ) := 
by 
  sorry

end cos_alpha_plus_beta_l78_78272


namespace shape_covering_circle_l78_78369

theorem shape_covering_circle (F : set ℝ) (d : F.diameter = 1) 
  : ∃ (M : set ℝ) (O : ℝ) (r : ℝ), r = (√3 / 2) ∧ M = metric.ball O r ∧ F ⊆ M :=
by sorry

end shape_covering_circle_l78_78369


namespace collinear_A2_B2_C2_l78_78755

-- Points A, B, C are on line m
axiom points_on_line_m : (A B C : Point) (m : Line), 
  A ∈ m ∧ B ∈ m ∧ C ∈ m

-- Points A1, B1, C1 are on line n
axiom points_on_line_n : (A1 B1 C1 : Point) (n : Line), 
  A1 ∈ n ∧ B1 ∈ n ∧ C1 ∈ n

-- Lines AA1, BB1, CC1 are parallel
axiom parallel_lines : (A A1 B B1 C C1 : Point),
  parallel (line_through A A1) (line_through B B1) ∧ 
  parallel (line_through B B1) (line_through C C1)

-- Points A2, B2, C2 divide AA1, BB1, CC1 respectively in equal ratios
axiom divide_segments : (A A1 A2 B B1 B2 C C1 C2 : Point) (t : Real) (0 < t < 1),
  segment_division A A1 A2 t ∧ 
  segment_division B B1 B2 t ∧ 
  segment_division C C1 C2 t

-- The line containing A2, B2, C2 belongs to the same pencil as lines m and n
theorem collinear_A2_B2_C2 : 
  ∀ (A B C A1 B1 C1 A2 B2 C2 : Point) (m n : Line) (t : Real),
  (points_on_line_m A B C m) → 
  (points_on_line_n A1 B1 C1 n) → 
  (parallel_lines A A1 B B1 C C1) → 
  (divide_segments A A1 A2 B B1 B2 C C1 C2 t) → 
  (∃ l : Line, (l ∈ pencil m n) ∧ A2 ∈ l ∧ B2 ∈ l ∧ C2 ∈ l) :=
sorry

end collinear_A2_B2_C2_l78_78755


namespace sum_of_roots_of_equation_l78_78838

theorem sum_of_roots_of_equation : 
  (∑ x in {x : ℝ | 10 = (x^3 + 5 * x^2 - 8 * x) / (x + 2) ∧ x ≠ -2}, x) = 2 :=
sorry

end sum_of_roots_of_equation_l78_78838


namespace sampling_problem_l78_78118

theorem sampling_problem (students : ℕ)
  (students_first_grade : ℕ)
  (students_second_grade : ℕ)
  (students_third_grade : ℕ)
  (sample_1 sample_2 sample_3 sample_4 : list ℕ)
  (seq1 : sample_1 = [7, 34, 61, 88, 115, 142, 169, 196, 223, 250])
  (seq2 : sample_2 = [5, 9, 100, 107, 111, 121, 180, 195, 200, 265])
  (seq3 : sample_3 = [11, 38, 65, 92, 119, 146, 173, 200, 227, 254])
  (seq4 : sample_4 = [30, 57, 84, 111, 138, 165, 192, 219, 246, 270])
  (students = 270)
  (students_first_grade = 108)
  (students_second_grade = 81)
  (students_third_grade = 81) :
  ¬(is_stratified_sampling sample_2 students_first_grade students_second_grade students_third_grade) ∧
  ¬(is_stratified_sampling sample_4 students_first_grade students_second_grade students_third_grade) :=
sorry

def is_stratified_sampling (sample: list ℕ)
  (students_first_grade: ℕ) (students_second_grade: ℕ) (students_third_grade: ℕ) : Prop :=
-- Definition of stratified sampling goes here
sorry

end sampling_problem_l78_78118


namespace fractional_equation_solution_l78_78802

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ -3) (h : 1/x = 2/(x+3)) : x = 3 :=
sorry

end fractional_equation_solution_l78_78802


namespace area_difference_l78_78146

theorem area_difference (d_square : ℝ) (d_circle : ℝ) : d_square = 6 ∧ d_circle = 8 → 
  (let s := d_square / real.sqrt 2 in
   let r := d_circle / 2 in
   let area_square := s^2 in
   let area_circle := real.pi * r^2 in
   real.abs (area_circle - area_square) ≈ 32.3) :=
begin
  sorry
end

end area_difference_l78_78146


namespace cos_arcsin_l78_78570

theorem cos_arcsin (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arcsin x) = Real.sqrt (1 - x^2) :=
  sorry

example : Real.cos (Real.arcsin (2 / 3)) = (Real.sqrt 5) / 3 :=
by
  -- Use the general theorem about cos(arcsin(x)) to solve this specific case
  have h_cos_arcsin := cos_arcsin (2 / 3)
  -- Prove the bounds are satisfied for 2/3
  have bounds : 0 ≤ 2 / 3 ∧ 2 / 3 ≤ 1 := ⟨by norm_num, by norm_num⟩
  rw h_cos_arcsin bounds
  norm_num
  rw Real.sqrt_div
  norm_num
  rw Real.sqrt_mul_self
  norm_num
  norm_num

end cos_arcsin_l78_78570


namespace lattice_point_condition_l78_78469

noncomputable def max_value_a : ℚ :=
  50 / 99

theorem lattice_point_condition : 
    ∀ (m : ℚ) (a : ℚ), (1 / 2 < m ∧ m < a) ∧
    (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬ ∃ y : ℤ, y = m * x + 2) 
    → a = max_value_a := 
  by 
    intros m a ha h_condition
    sorry

end lattice_point_condition_l78_78469


namespace find_parabola_a_l78_78785

noncomputable def parabola_vertex_form (a x : ℝ) : ℝ := a * (x - 2)^2 + 5

theorem find_parabola_a :
  ∃ (a : ℝ), parabola_vertex_form a 1 = 3 ∧ parabola_vertex_form a 2 = 5 ∧ a = -2 :=
by
  use -2
  split
  case left =>
    calc
      parabola_vertex_form (-2) 1
          = (-2) * (1 - 2)^2 + 5 : by rw parabola_vertex_form
      ... = 3                     : by norm_num
  case right =>
    split
    case left =>
      calc
        parabola_vertex_form (-2) 2
            = (-2) * (2 - 2)^2 + 5 : by rw parabola_vertex_form
        ... = 5                     : by norm_num
    case right =>
      rfl

end find_parabola_a_l78_78785


namespace oliver_money_left_l78_78358

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l78_78358


namespace three_digit_integer_one_more_than_LCM_l78_78839

theorem three_digit_integer_one_more_than_LCM:
  ∃ (n : ℕ), (n > 99 ∧ n < 1000) ∧ (∃ (k : ℕ), n = k + 1 ∧ (∃ m, k = 3 * 4 * 5 * 7 * 2^m)) :=
  sorry

end three_digit_integer_one_more_than_LCM_l78_78839


namespace rectangle_area_l78_78854

variable (L B : ℕ)

theorem rectangle_area :
  (L - B = 23) ∧ (2 * L + 2 * B = 166) → (L * B = 1590) :=
by
  sorry

end rectangle_area_l78_78854


namespace subscription_amount_l78_78550

noncomputable def total_subscription (A B C : ℝ) (total_profit : ℝ) (A_profit : ℝ) : ℝ := A + B + C

theorem subscription_amount :
  ∀ (C_sub : ℝ),
  let B_sub := C_sub + 5000 in
  let A_sub := B_sub + 4000 in
  let total_profit := 70000 in
  let A_profit := 29400 in
  (A_profit / total_profit = A_sub / (A_sub + B_sub + C_sub)) →
  (total_subscription A_sub B_sub C_sub total_profit A_profit = 50000) :=
by
  intros C_sub B_sub A_sub total_profit A_profit h
  unfold total_subscription
  sorry

end subscription_amount_l78_78550


namespace find_y_of_angle_l78_78244

theorem find_y_of_angle 
  (θ : ℝ)
  (P : ℝ × ℝ)
  (hP : P.1 = 4)
  (hsinθ : sin θ = - (2 * real.sqrt 5) / 5) :
  P.2 = -8 :=
sorry

end find_y_of_angle_l78_78244


namespace equation_of_line_BC_l78_78978

theorem equation_of_line_BC (A B C : ℝ × ℝ) (x1 y1 x2 y2 : ℝ) :
  let A := (0, 4)
  ∧ (B = (x1, y1) ∨ B = (x2, y2))
  ∧ (C = (x2, y2) ∨ C = (x1, y1))
  ∧ ((x1^2 / 20) + (y1^2 / 16) = 1)
  ∧ ((x2^2 / 20) + (y2^2 / 16) = 1)
  ∧ (2 = (0 + x1 + x2) / 3)
  ∧ (0 = (4 + y1 + y2) / 3)
  → 6 * x - 5 * y - 28 = 0 :=
by
  sorry

end equation_of_line_BC_l78_78978


namespace part1_part2_l78_78341

-- Part 1: M ⊆ N implies m ∈ [3, +∞)
theorem part1 (M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5})
  (N : Set ℝ := {x | ∃ (m : ℝ), 1 - 2 * m ≤ x ∧ x ≤ 2 + m }) 
  (H : ∀ x ∈ M, x ∈ N) : ∀ (m : ℝ), m ≥ 3 :=
by
  sorry

-- Part 2: p necessary but not sufficient for q implies m ∈ (-∞, 3/2]
theorem part2 (M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5})
  (N : Set ℝ := {x | ∃ (m : ℝ), 1 - 2 * m ≤ x ∧ x ≤ 2 + m }) 
  (H : ∀ x ∈ M, x ∈ N) : ∀ (m : ℝ), m ∈ Set.Ioc (-(1 : ℝ / 3)) (3 / 2) :=
by
  sorry

end part1_part2_l78_78341


namespace total_movies_seen_l78_78177

theorem total_movies_seen (d h a c : ℕ) (hd : d = 7) (hh : h = 12) (ha : a = 15) (hc : c = 2) :
  (c + (d - c) + (h - c) + (a - c)) = 30 :=
by
  sorry

end total_movies_seen_l78_78177


namespace ivan_income_tax_l78_78706

noncomputable def personalIncomeTax (monthly_salary: ℕ → ℕ) (bonus: ℕ) (tax_rate: ℚ) : ℕ :=
  let taxable_income := (monthly_salary 3 + monthly_salary 4) +
                       (List.sum (List.map monthly_salary [5, 6, 7, 8, 9, 10, 11, 12])) +
                       bonus
  in taxable_income * tax_rate

theorem ivan_income_tax :
  personalIncomeTax
    (λ m, if m ∈ [3, 4] then 20000 else if m ∈ [5, 6, 7, 8, 9, 10, 11, 12] then 25000 else 0)
    10000 0.13 = 32500 :=
  sorry

end ivan_income_tax_l78_78706


namespace distinct_prime_factors_of_B_l78_78334

noncomputable def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

noncomputable def B : ℕ :=
  List.foldr (· * ·) 1 (divisors 30)

theorem distinct_prime_factors_of_B : 
  (List.toFinset (List.filter (Nat.prime) (Nat.factors B))).card = 3 := by
  sorry

end distinct_prime_factors_of_B_l78_78334


namespace count_multiples_4_or_5_not_20_l78_78649

-- We define the necessary ranges and conditions
def is_multiple_of (n k : ℕ) := n % k = 0

def count_multiples (n k : ℕ) := (n / k)

def not_multiple_of (n k : ℕ) := ¬ is_multiple_of n k

def count_multiples_excluding (n k l : ℕ) :=
  count_multiples n k + count_multiples n l - count_multiples n (Nat.lcm k l)

theorem count_multiples_4_or_5_not_20 : count_multiples_excluding 3010 4 5 = 1204 := 
by
  sorry

end count_multiples_4_or_5_not_20_l78_78649


namespace max_distinct_fans_l78_78078

-- Define the problem conditions and the main statement
theorem max_distinct_fans : 
  let n := 6  -- Number of sectors per fan
  let configurations := 2^n  -- Total configurations without considering symmetry
  let unchanged_flips := 8  -- Number of configurations unchanged by flipping
  let distinct_configurations := (configurations - unchanged_flips) / 2 + unchanged_flips 
  in 
  distinct_configurations = 36 := by sorry  # We state the answer based on the provided steps


end max_distinct_fans_l78_78078


namespace circle_properties_l78_78923

variable (a : ℝ)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * a * x - 2 * a * y = 0

theorem circle_properties :
  (∀ (x y : ℝ), circle_eq a x y → (x + y = 0)) ∧ -- The circle is symmetric about the line x + y = 0
  ¬ (∃ x : ℝ, circle_eq a x a) ∧ -- The center does not lie on the x-axis
  circle_eq a 0 0 ∧ -- The circle passes through the origin
  (∃ r : ℝ, r = sqrt (2) * |a| ∧ r ≠ sqrt (2) * a) -- The radius is sqrt(2) * |a|, not sqrt(2)a
:= by
  sorry

end circle_properties_l78_78923


namespace bill_amount_shared_l78_78406

open BigOperators
open Real

-- Definitions for conditions
variables (totalBill : ℝ) (tipRate : ℝ) (numPeople : ℕ)
def tip (totalBill tipRate : ℝ) : ℝ := totalBill * tipRate
def totalBillWithTip (totalBill tipRate : ℝ) : ℝ := totalBill + tip totalBill tipRate
def amountPerPerson (totalBill tipRate : ℝ) (numPeople : ℕ) : ℝ := totalBillWithTip totalBill tipRate / numPeople

-- Given conditions
axiom h1 : totalBill = 139.00
axiom h2 : tipRate = 0.10
axiom h3 : numPeople = 6

-- Goal
theorem bill_amount_shared (h1 : totalBill = 139.00) (h2 : tipRate = 0.10) (h3 : numPeople = 6) : 
  (Float.round (amountPerPerson totalBill tipRate numPeople) * 100) / 100 = 25.48 := 
sorry

end bill_amount_shared_l78_78406


namespace not_right_triangle_angle_ratio_l78_78430

theorem not_right_triangle_angle_ratio (A B C : ℝ) (h₁ : A / B = 3 / 4) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end not_right_triangle_angle_ratio_l78_78430


namespace sum_of_solutions_in_range_l78_78329

theorem sum_of_solutions_in_range :
  let T := ∑ y in {y : ℝ | y > 0 ∧ y ^ (3 ^ (sqrt 3)) = (sqrt 3) ^ (3 ^ y)}, y
  in 4 ≤ T ∧ T < 8 :=
by
  let T := ∑ y in {y : ℝ | y > 0 ∧ y ^ (3 ^ (sqrt 3)) = (sqrt 3) ^ (3 ^ y)}, y
  sorry

end sum_of_solutions_in_range_l78_78329


namespace hyperbola_eccentricity_is_correct_l78_78254

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  let t := λ c : ℝ, c
  let e := λ c : ℝ, c / ((sqrt 3 - 1) / 2 * c)
  e 1  -- setting c=1 to simply use t = 1 as given t = c in the solution.

theorem hyperbola_eccentricity_is_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ M : ℝ × ℝ, (∃ t : ℝ, (|M.1 - a| = t) ∧ |M.2 - sqrt 3 * t| = sqrt 3 * t) ∧ hyperbola_eccentricity a b h1 h2 = sqrt 3 + 1 :=
by
  sorry

end hyperbola_eccentricity_is_correct_l78_78254


namespace solution_set_of_inequality_l78_78615

variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}

axiom mono_f : Monotonic f
axiom passes_A : f (-3) = 2
axiom passes_B : f (2) = -2
axiom inverse_f : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x

theorem solution_set_of_inequality :
  { x : ℝ | |2 * f_inv x + 1| < 5 } = { x : ℝ | -2 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l78_78615


namespace volume_formula_l78_78138

noncomputable def volume_of_parallelepiped
  (a b : ℝ) (h : ℝ) (θ : ℝ) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ)
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2)) : ℝ :=
  a * b * h 

theorem volume_formula 
  (a b : ℝ) (h : ℝ) (θ : ℝ)
  (area_base : ℝ) 
  (area_of_base_eq : area_base = a * b) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ) 
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2))
  (height_eq : h = (base_diagonal / 2) * (Real.sqrt 3)): 
  volume_of_parallelepiped a b h θ θ_eq base_diagonal base_diagonal_eq 
  = (144 * Real.sqrt 3) / 5 :=
by {
  sorry
}

end volume_formula_l78_78138


namespace sum_log_identity_l78_78167

noncomputable def summand (k : ℕ) : ℝ :=
  real.log (1 + (1 / k.to_real)) / real.log 3 * real.log 3 / real.log k * real.log 3 / real.log (k + 1)

theorem sum_log_identity : 
  ∑ k in finset.range(48) + 3, summand k = 1 - 1 / real.log 51 := 
begin
  sorry,
end

end sum_log_identity_l78_78167


namespace largest_25_supporting_X_l78_78129

def is_25_supporting (X : ℝ) : Prop :=
∀ (a : fin 25 → ℝ), (∑ i, a i).den % 1 = 0 → ∃ i, |a i - 0.5| ≥ X

theorem largest_25_supporting_X : Sup {X : ℝ | is_25_supporting X} = 0.02 :=
sorry

end largest_25_supporting_X_l78_78129


namespace arrangement_plans_correct_l78_78466

-- Define that there are six classes and four students
constant classes : Fin 6 → Type
constant students : Fin 4

-- Define the arrangement function
noncomputable def arrangement_plans : ℕ := 90

-- Theorem statement asserting that the number of arrangement plans is 90
theorem arrangement_plans_correct : (∃ C1 C2 : Fin 6, C1 ≠ C2) ∧ (∃ S1 S2 S3 S4 : students, true) → arrangement_plans = 90 :=
by
  sorry

end arrangement_plans_correct_l78_78466


namespace parabola_coefficients_l78_78136

theorem parabola_coefficients (a b c : ℝ) (f : ℝ → ℝ) (vertex : f 2 = 3) (point : f 0 = 7) : 
  (f = λ x, a * x^2 + b * x + c) → a + b + c = 4 := 
by {
  sorry
}

end parabola_coefficients_l78_78136


namespace sum_mod_7_l78_78905

theorem sum_mod_7 (n : ℕ) (h : n = 127) : 
  (∑ i in Finset.range (n + 1), i) % 7 = 1 :=
by
  rw h
  sorry

end sum_mod_7_l78_78905


namespace angle_B_is_60_degrees_l78_78328

theorem angle_B_is_60_degrees
  {A B C G : Type}
  [has_vector_add A B C G]
  (centroid_G : is_centroid G A B C)
  (h : (sqrt 7 / (vector GA) * (sin A) + 3 * (vector GB) * (sin B) + 3 * sqrt 7 * (vector GC) * (sin C) = 0)) :
  angle B = 60 := 
sorry

end angle_B_is_60_degrees_l78_78328


namespace slope_of_intersection_line_l78_78215

theorem slope_of_intersection_line (u x y : ℝ) 
  (h1 : 2 * x + 3 * y = 8 * u + 4)
  (h2 : 3 * x - 2 * y = 5 * u - 3) :
  ∃ m : ℝ, m = 10 / 31 :=
begin
  use 10 / 31,
  sorry
end

end slope_of_intersection_line_l78_78215


namespace max_snowmen_l78_78497

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l78_78497


namespace max_min_values_l78_78202

def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

theorem max_min_values :
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  max (f a) (f b) = f a ∧ f a = 269 / 27 ∧ min (f a) (f b) = f b ∧ f b = -5 :=
by
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  have ha : f a = 269 / 27 := sorry
  have hb : f b = -5 := sorry
  have max_eq : max (f a) (f b) = f a := by sorry
  have min_eq : min (f a) (f b) = f b := by sorry
  exact ⟨max_eq, ha, min_eq, hb⟩

end max_min_values_l78_78202


namespace problem1_problem2_l78_78264

noncomputable def p := (1, Real.sqrt 3)
noncomputable def q (x : ℝ) := (Real.cos x, Real.sin x)

theorem problem1 (x : ℝ) (h : ∀ (a : ℝ) (a ≠ 0), q x = a • p) : 
  Real.sin (2 * x) - Real.cos x ^ 2 = (2 * Real.sqrt 3 - 1) / 4 := 
sorry

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := f (x / 2 + π / 3)

theorem problem2 (k : ℤ) (x : ℝ) (h : ∃ n : ℤ, x = (2 * n + k) * π / 3 + π / 6): 
  -2 * π / 3 + k * π ≤ x ∧ x ≤ -π / 6 + k * π :=
sorry

end problem1_problem2_l78_78264


namespace sequence_summation_l78_78297

variable {a : ℕ → ℕ}
variable {n : ℕ}

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n, n ≥ 1 → a (n+1) - a n ^ 2 + a (n-1) = 0

def non_zero (a : ℕ → ℕ) : Prop :=
∀ n, a n ≠ 0

-- Statement of the problem in Lean 4
theorem sequence_summation (h_seq : is_arithmetic_sequence a) (h_non_zero : non_zero a) : 
  (∑ i in finset.range (2*n-1), a i) - 4 * n = -2 :=
sorry

end sequence_summation_l78_78297


namespace cone_lateral_surface_area_l78_78771

theorem cone_lateral_surface_area (a : ℝ) (h : a > 0) :
  let r := a,
      l := 2 * a,
      S := π * r * l in 
  S = 2 * π * a^2 :=
by
  let r := a
  let l := 2 * a
  let S := π * r * l
  show S = 2 * π * a^2
  sorry

end cone_lateral_surface_area_l78_78771


namespace intersection_A_B_l78_78622

variable A : Set ℤ := {-2, 0, 2}
variable B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_A_B : A ∩ B = {2} := sorry

end intersection_A_B_l78_78622


namespace no_playful_two_digit_numbers_l78_78893

def is_playful (a b : ℕ) : Prop := 10 * a + b = a^3 + b^2

theorem no_playful_two_digit_numbers :
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → ¬ is_playful a b) :=
by {
  sorry
}

end no_playful_two_digit_numbers_l78_78893


namespace sum_of_roots_eq_two_l78_78596

theorem sum_of_roots_eq_two :
  let a := 25
  let b := -50
  let c := 35
  let d := 7
  let roots := (λ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0)
  symmetric_difference roots =
begin
  sorry -- Proof to be filled in
end

end sum_of_roots_eq_two_l78_78596


namespace maximum_snowmen_count_l78_78494

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78494


namespace friends_equal_payment_l78_78099

theorem friends_equal_payment (num_friends : ℕ) (total_bill : ℝ) (discount_percent : ℝ) (each_payment : ℝ) :
  num_friends = 6 →
  total_bill = 400 →
  discount_percent = 5 →
  each_payment = (total_bill * (1 - discount_percent / 100)) / num_friends →
  each_payment = 63.33 :=
by
  intros h1 h2 h3 h4
  have h5 : (total_bill * (1 - discount_percent / 100)) / num_friends = 63.33 := sorry
  rw [h1, h2, h3] at h4
  exact h5

end friends_equal_payment_l78_78099


namespace maximum_snowmen_count_l78_78488

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l78_78488


namespace geometric_sequence_a3_l78_78229

noncomputable def a_1 (S_4 : ℕ) (q : ℕ) : ℕ :=
  S_4 * (q - 1) / (1 - q^4)

noncomputable def a_3 (a_1 : ℕ) (q : ℕ) : ℕ :=
  a_1 * q^(3 - 1)

theorem geometric_sequence_a3 (a_n : ℕ → ℕ) (S_4 : ℕ) (q : ℕ) :
  (q = 2) →
  (S_4 = 60) →
  a_3 (a_1 S_4 q) q = 16 :=
by
  intro hq hS4
  rw [hq, hS4]
  sorry

end geometric_sequence_a3_l78_78229


namespace no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l78_78235

-- Part (i)
theorem no_solutions_for_a_ne_4 (a : ℕ) (h : a ≠ 4) :
  ¬∃ (u v : ℕ), (u > 0 ∧ v > 0 ∧ u^2 + v^2 - a * u * v + 2 = 0) :=
by sorry

-- Part (ii)
theorem solutions_for_a_eq_4_infinite :
  ∃ (a_seq : ℕ → ℕ),
    (a_seq 0 = 1 ∧ a_seq 1 = 3 ∧
     ∀ n, a_seq (n + 2) = 4 * a_seq (n + 1) - a_seq n ∧
    ∀ n, (a_seq n) > 0 ∧ (a_seq (n + 1)) > 0 ∧ (a_seq n)^2 + (a_seq (n + 1))^2 - 4 * (a_seq n) * (a_seq (n + 1)) + 2 = 0) :=
by sorry

end no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l78_78235


namespace ivan_income_tax_l78_78704

noncomputable def personalIncomeTax (monthly_salary: ℕ → ℕ) (bonus: ℕ) (tax_rate: ℚ) : ℕ :=
  let taxable_income := (monthly_salary 3 + monthly_salary 4) +
                       (List.sum (List.map monthly_salary [5, 6, 7, 8, 9, 10, 11, 12])) +
                       bonus
  in taxable_income * tax_rate

theorem ivan_income_tax :
  personalIncomeTax
    (λ m, if m ∈ [3, 4] then 20000 else if m ∈ [5, 6, 7, 8, 9, 10, 11, 12] then 25000 else 0)
    10000 0.13 = 32500 :=
  sorry

end ivan_income_tax_l78_78704


namespace total_money_received_by_A_l78_78896

-- Define the given conditions and required proof
open Real

theorem total_money_received_by_A :
  let total_profit := 16500
  let a_management_percentage := 0.12
  let remaining_profit_percentage_a := 0.35
  let a_management_share := a_management_percentage * total_profit
  let remaining_profit := total_profit - a_management_share
  let a_profit_share := remaining_profit_percentage_a * remaining_profit
  let total_money_received_a := a_management_share + a_profit_share
  total_money_received_a = 7062 :=
by
  -- Define the values
  let total_profit := 16500
  let a_management_percentage := 0.12
  let remaining_profit_percentage_a := 0.35
  -- Calculate A's management share
  let a_management_share := a_management_percentage * total_profit
  -- Calculate the remaining profit
  let remaining_profit := total_profit - a_management_share
  -- Calculate A's share from the remaining profit
  let a_profit_share := remaining_profit_percentage_a * remaining_profit
  -- Calculate the total money received by A
  let total_money_received_a := a_management_share + a_profit_share
  -- Prove that total_money_received_a is equal to Rs. 7062
  have h : total_money_received_a = 7062 := sorry
  exact h

end total_money_received_by_A_l78_78896


namespace spending_total_march_to_july_l78_78392

/-- Given the conditions:
  1. Total amount spent by the beginning of March is 1.2 million,
  2. Total amount spent by the end of July is 5.4 million,
  Prove that the total amount spent during March, April, May, June, and July is 4.2 million. -/
theorem spending_total_march_to_july
  (spent_by_end_of_feb : ℝ)
  (spent_by_end_of_july : ℝ)
  (h1 : spent_by_end_of_feb = 1.2)
  (h2 : spent_by_end_of_july = 5.4) :
  spent_by_end_of_july - spent_by_end_of_feb = 4.2 :=
by
  sorry

end spending_total_march_to_july_l78_78392


namespace n4_l78_78092

theorem n4 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∀ n : ℕ, n > 2020 ^ 2020 → ∃ m : ℕ, Nat.coprime m n ∧ a ^ n + b ^ n ∣ a ^ m + b ^ m) :
  a = b :=
sorry

end n4_l78_78092


namespace find_C_coordinates_l78_78539

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -3, y := 5 }
def B : Point := { x := 9, y := -1 }
def C : Point := { x := 15, y := -4 }

noncomputable def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

theorem find_C_coordinates :
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  C = C_actual :=
by
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  show C = C_actual
  rfl

end find_C_coordinates_l78_78539


namespace calculate_exponent_product_l78_78566

theorem calculate_exponent_product : (2^2021) * (-1/2)^2022 = (1/2) :=
by
  sorry

end calculate_exponent_product_l78_78566


namespace find_mean_proportional_l78_78203

noncomputable def mean_proportional (A B : ℝ) : ℝ :=
  Real.sqrt (A * B)

theorem find_mean_proportional (x : ℝ) (h₁ : 3*x + 5 > 0) (h₂ : 4*x - 1 > 0) 
  : mean_proportional (abs (sin x + Real.log (3 * x + 5))) (abs (cos (2 * x) - Real.log2 (4 * x - 1))) = 
    Real.sqrt (abs (sin x + Real.log (3 * x + 5)) * abs (cos (2 * x) - Real.log2 (4 * x - 1))) :=
  sorry

end find_mean_proportional_l78_78203


namespace part1_l78_78097

theorem part1 (a b c d : ℤ) (h : a * d - b * c = 1) : Int.gcd (a + b) (c + d) = 1 :=
sorry

end part1_l78_78097


namespace vector_dot_product_calculation_l78_78660

variables {V : Type*} [inner_product_space ℝ V]

variables (a b c : V)

theorem vector_dot_product_calculation 
  (h₁ : inner_product a b = 5)
  (h₂ : inner_product a c = -2)
  (h₃ : inner_product b c = 3) :
  inner_product a (4 • b - 3 • c) = 26 :=
by 
  -- proof would go here
  sorry

end vector_dot_product_calculation_l78_78660


namespace solution_set_inequality_l78_78799

theorem solution_set_inequality (x : ℝ) :
  (3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0) ↔ (-1 / 3 ≤ x ∧ x < 5) :=
by
  sorry

end solution_set_inequality_l78_78799


namespace volume_of_tetrahedron_is_4_l78_78035

-- Declare the conditions (edge lengths).
variables (AB AC AD BC BD CD : ℝ)
variables (h_AB : AB = 2) (h_AC : AC = 3)
          (h_AD : AD = 4) (h_BC : BC = √13)
          (h_BD : BD = 2 * √5) (h_CD : CD = 5)

-- Define the statement of the problem:
theorem volume_of_tetrahedron_is_4 :
  volume_of_tetrahedron AB AC AD BC BD CD = 4 :=
by sorry

end volume_of_tetrahedron_is_4_l78_78035


namespace probability_correct_l78_78140

noncomputable def probability_purple_greater_green_less_twice_green : ℝ :=
  let green := measure_Icc (0, 1) in
  let purple := measure_Icc (0, 1) in
  let total_area := 1 in
  let shaded_area := (1/2) * (1/2) + (1/2) * (1/8) in
  shaded_area / total_area

theorem probability_correct : 
  probability_purple_greater_green_less_twice_green = 1 / 4 :=
by
  sorry

end probability_correct_l78_78140


namespace prism_edges_l78_78682

theorem prism_edges (faces_prism : ℕ) (h_faces : faces_prism = 5) : ∃ edges_prism : ℕ, edges_prism = 9 :=
by {
  use 9,
  rw h_faces,
  sorry
}

end prism_edges_l78_78682


namespace standard_ellipse_form_existence_of_fixed_points_perpendicular_mn_mb_l78_78987

-- Constants and conditions for the ellipse problem
variable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (e : ℝ) (he : e = (sqrt 2) / 2)
variable (x y : ℝ)
variable (eq1 : (x * x) / (a * a) + (y * y) / (b * b) = 1)
variable (left_focus : ℝ) (lf : left_focus = - sqrt 2)
noncomputable def standard_form_ellipse := sorry

-- Point P condition
variable (x1 y1 x2 y2 : ℝ)
variable (x_p y_p : ℝ)
variable (eq_p : (x_p = x1 + 2 * x2) ∧ (y_p = y1 + 2 * y2))
variable (slope_prod : (y1 * y2) / (x1 * x2) = -1/2)
noncomputable def fixed_points : (ℝ × ℝ) × (ℝ × ℝ) := ((-2, 0), (2, 0))

-- Perpendicularity of MN and MB
variable (M N A B : point)
variable (in_first_quadrant : M.x > 0 ∧ M.y > 0)
variable (symmetric_about_origin : N = -M)
variable (projection_M_on_x : A = (M.x, 0))
variable (intersection : B ∈ ellipse)
noncomputable def perpendicular_MN_MB : Prop := sorry

-- Statement 1: Ellipse standard form
theorem standard_ellipse_form : (eq1) → standard_form_ellipse :=
begin
  sorry
end

-- Statement 2: Fixed points F_1 and F_2
theorem existence_of_fixed_points :
  eq_p ∧ slope_prod →
  ∃ F1 F2: ℝ × ℝ, F1 = (-2, 0) ∧ F2 = (2, 0) ∧ |P - F1| + |P - F2| = 4 * sqrt 2 :=
begin
  sorry
end

-- Statement 3: Perpendicular MN MB
theorem perpendicular_mn_mb :
  in_first_quadrant ∧ symmetric_about_origin ∧ projection_M_on_x ∧ intersection →
  perpendicular_MN_MB :=
begin
  sorry
end

end standard_ellipse_form_existence_of_fixed_points_perpendicular_mn_mb_l78_78987


namespace smallest_multiple_of_24_g_gt_24_l78_78331

-- Step 1: Define the function g(n)
def g (n : ℕ) : ℕ :=
  Inf { k : ℕ | k.factorial % n = 0 }

-- Step 2: State the conditions on n
def is_multiple_of_24 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 24 * k

-- Step 3: Translate the final proof goal into Lean statement
theorem smallest_multiple_of_24_g_gt_24 :
  ∃ n : ℕ, is_multiple_of_24 n ∧ g n > 24 ∧ n = 696 :=
by
  sorry

end smallest_multiple_of_24_g_gt_24_l78_78331
