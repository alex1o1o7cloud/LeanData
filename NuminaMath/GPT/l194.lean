import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Multiset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Probability.CondProb
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Time
import Mathlib.Geometry
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Logic.Basic
import Mathlib.Order.AbsoluteValue
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Set
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace part1_part2_l194_194555

-- Define the function f(x) = x * ln x - (a/2) * x^2
def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

-- Define the line l
def l (x k : ℝ) : ℝ := (k - 2) * x - k + 1

-- Assumption references
variable (e_val e_sq : ℝ)
#check Real.exp 1

-- Part 1: if x \in [e, e^2] and f(x) > 0, then a < 2/e
theorem part1 (x : ℝ) (h1 : x ∈ Set.Icc (Real.exp 1) (Real.exp 2)) (h2 : f x a > 0) : a < 2 / Real.exp 1 := sorry

-- Part 2: if a = 0 and f(x) > l for x > 1, then the max integer k is 4
theorem part2 (k : ℤ) (h1 : ∀ x > 1, f x 0 > l x k) : k ≤ 4 := sorry

end part1_part2_l194_194555


namespace squirrel_travel_time_3_miles_l194_194773

def squirrel_speed := 6    -- Speed in miles per hour
def distance := 3          -- Distance in miles

def travel_time_in_minutes (speed : ℝ) (dist : ℝ) : ℝ :=
  (dist / speed) * 60     -- Convert hours to minutes

theorem squirrel_travel_time_3_miles : 
  travel_time_in_minutes squirrel_speed distance = 30 := 
by
  simp [squirrel_speed, distance, travel_time_in_minutes]
  norm_num
  sorry

end squirrel_travel_time_3_miles_l194_194773


namespace boy_speed_on_second_day_l194_194004

theorem boy_speed_on_second_day 
  (d : ℝ) (s1 : ℝ) (t1_late : ℝ) (distance : d = 2.5) 
  (speed1 : s1 = 5) (time1 : t1_late = (1/12)) 
  (t2_early : ℝ) (time2 : t2_early = (1/6)) : 
  ∃ s2 : ℝ, s2 = 7.5 :=
begin
  use 7.5,
  have h1 : 0.5 * s1 - t2_early * s1 + 0.5 * s1 + t1_late * s1 = 0.5 * 7.5,
  { -- Proof of h1 follows from the arithmetic calculations in the solution
    sorry },
  -- Concluding the theorem using h1
  exact h1
end

end boy_speed_on_second_day_l194_194004


namespace distance_house_market_l194_194447

theorem distance_house_market : 
  ∀ (d_hs d_sp d_pf d_fs d_sh d_total d_hm : ℝ), 
  d_hs = 50 ∧ 
  d_sp = 25 ∧ 
  d_pf = d_sp / 2 ∧ 
  d_fs = d_pf ∧ 
  d_sh = d_hs ∧ 
  d_total = 345 ∧ 
  d_total = d_hs + d_sp + d_pf + d_fs + d_sh + d_hm → 
  d_hm = 195 :=
by
  intros d_hs d_sp d_pf d_fs d_sh d_total d_hm 
  intro h
  rw [and_assoc, and_assoc, and_assoc, and_assoc, and_assoc, and_assoc] at h
  obtain ⟨h_hs, h_sp, h_pf, h_fs, h_sh, h_total, h_eq_total⟩ := h
  sorry

end distance_house_market_l194_194447


namespace four_digit_palindrome_squares_count_l194_194210

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem four_digit_palindrome_squares_count : (Finset.filter (λ n, is_palindrome (n * n)) (Finset.range 100)).card = 2 := by
  sorry

end four_digit_palindrome_squares_count_l194_194210


namespace perpendicular_vectors_eq_l194_194173

theorem perpendicular_vectors_eq {x : ℝ} (h : (x - 5) * 2 + 3 * x = 0) : x = 2 :=
sorry

end perpendicular_vectors_eq_l194_194173


namespace problem1_problem2_l194_194314

theorem problem1 (x : ℝ) (h1 : x * (x + 4) = -5 * (x + 4)) : x = -4 ∨ x = -5 := 
by 
  sorry

theorem problem2 (x : ℝ) (h2 : (x + 2) ^ 2 = (2 * x - 1) ^ 2) : x = 3 ∨ x = -1 / 3 := 
by 
  sorry

end problem1_problem2_l194_194314


namespace intersection_A_B_l194_194885

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}
def intersection_of_A_and_B : Set ℕ := {0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = intersection_of_A_and_B :=
by
  sorry

end intersection_A_B_l194_194885


namespace quadrant_of_minus_reciprocal_square_l194_194864

theorem quadrant_of_minus_reciprocal_square (z : Complex) (h : Real.pi / 4 < Complex.arg(z) ∧ Complex.arg(z) < Real.pi / 2) : 
  ∃ (θ : Real), 0 < θ ∧ θ < Real.pi / 2 ∧ -Complex.inv (Complex.sq z) = Complex.polar (Complex.abs z ^ (-2)) θ := sorry

end quadrant_of_minus_reciprocal_square_l194_194864


namespace permutation_count_4_l194_194214

theorem permutation_count_4 : ∃ n : ℕ, n = 4! := by
  use 24
  sorry

end permutation_count_4_l194_194214


namespace total_amount_spent_l194_194256

variables (original_price_backpack : ℕ) (increase_backpack : ℕ) (original_price_binder : ℕ) (decrease_binder : ℕ) (num_binders : ℕ)

-- Given Conditions
def original_price_backpack := 50
def increase_backpack := 5
def original_price_binder := 20
def decrease_binder := 2
def num_binders := 3

-- Prove the total amount spent is $109
theorem total_amount_spent :
  (original_price_backpack + increase_backpack) + (num_binders * (original_price_binder - decrease_binder)) = 109 := by
  sorry

end total_amount_spent_l194_194256


namespace hyperbola_eq_passing_point_asymptote_l194_194759

theorem hyperbola_eq_passing_point_asymptote 
  (λ x y : ℝ)
  (asymptote1 : 2 * x + 3 * y = 0) 
  (asymptote2 : 2 * x - 3 * y = 0) 
  (point_on_hyperbola : (1, 2) ∈ set_of (λ p, 4 * p.1^2 - 9 * p.2^2 = λ)) : 
    4 * x^2 - 9 * y^2 = -32 :=
by
  sorry

end hyperbola_eq_passing_point_asymptote_l194_194759


namespace find_A_value_l194_194995

noncomputable def A := 15 * Real.tan (44 * Real.pi / 180) * Real.tan (45 * Real.pi / 180) * Real.tan (46 * Real.pi / 180)

theorem find_A_value : A = 15 := by
  have tan_45_eq_1 : Real.tan (45 * Real.pi / 180) = 1 := 
    by calc
      Real.tan (45 * Real.pi / 180) = 1 : sorry
  
  have tan_46_identity : Real.tan (46 * Real.pi / 180) = 1 / Real.tan (44 * Real.pi / 180) := 
    by calc
      Real.tan (46 * Real.pi / 180) 
        = Real.cot (44 * Real.pi / 180) : sorry
        ... = 1 / Real.tan (44 * Real.pi / 180) : sorry
  
  calc
    A = 15 * Real.tan (44 * Real.pi / 180) * Real.tan (45 * Real.pi / 180) * Real.tan (46 * Real.pi / 180) : rfl
    ... = 15 * Real.tan (44 * Real.pi / 180) * 1 * Real.tan (46 * Real.pi / 180) : by rw [tan_45_eq_1]
    ... = 15 * Real.tan (44 * Real.pi / 180) * 1 * (1 / Real.tan (44 * Real.pi / 180)) : by rw [tan_46_identity]
    ... = 15 * (Real.tan (44 * Real.pi / 180) * (1 / Real.tan (44 * Real.pi / 180))) : mul_assoc _ _ _
    ... = 15 * 1 : mul_one _
    ... = 15 : rfl

end find_A_value_l194_194995


namespace four_digit_palindrome_squares_count_l194_194211

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem four_digit_palindrome_squares_count : (Finset.filter (λ n, is_palindrome (n * n)) (Finset.range 100)).card = 2 := by
  sorry

end four_digit_palindrome_squares_count_l194_194211


namespace sum_a_b_neg1_l194_194907

-- Define the problem using the given condition
theorem sum_a_b_neg1 (a b : ℝ) (h : |a + 3| + (b - 2) ^ 2 = 0) : a + b = -1 := 
by
  sorry

end sum_a_b_neg1_l194_194907


namespace custom_operation_difference_correct_l194_194219

def custom_operation (x y : ℕ) : ℕ := x * y + 2 * x

theorem custom_operation_difference_correct :
  custom_operation 5 3 - custom_operation 3 5 = 4 :=
by
  sorry

end custom_operation_difference_correct_l194_194219


namespace salary_January_l194_194672

variable (J F M A May : ℝ)

axiom avg_salary_Jan_to_April (h1 : ((J + F + M + A) / 4 = 8000)) : (J + F + M + A = 32000)

axiom avg_salary_Feb_to_May (h2 : ((F + M + A + May) / 4 = 8800)) : (F + M + A + May = 35200)

axiom salary_May (h3 : May = 6500) : True

theorem salary_January : J = 3300 := by
  have h1 : (J + F + M + A) = 32000 := avg_salary_Jan_to_April (by linarith)
  have h2 : (F + M + A + 6500) = 35200 := avg_salary_Feb_to_May (by linarith)
  have h3 : (F + M + A) = 35200 - 6500 := by linarith
  linarith
  sorry

end salary_January_l194_194672


namespace height_of_square_pyramid_is_13_l194_194772

noncomputable def square_pyramid_height (base_edge : ℝ) (adjacent_face_angle : ℝ) : ℝ :=
  let half_diagonal := base_edge * (Real.sqrt 2) / 2
  let sin_angle := Real.sin (adjacent_face_angle / 2 : ℝ)
  let opp_side := half_diagonal * sin_angle
  let height := half_diagonal * sin_angle / (Real.sqrt 3)
  height

theorem height_of_square_pyramid_is_13 :
  ∀ (base_edge : ℝ) (adjacent_face_angle : ℝ), 
  base_edge = 26 → 
  adjacent_face_angle = 120 → 
  square_pyramid_height base_edge adjacent_face_angle = 13 :=
by
  intros base_edge adjacent_face_angle h_base_edge h_adj_face_angle
  rw [h_base_edge, h_adj_face_angle]
  have half_diagonal := 26 * (Real.sqrt 2) / 2
  have sin_angle := Real.sin (120 / 2 : ℝ) -- sin 60 degrees
  have sqrt_three := Real.sqrt 3
  have height := (half_diagonal * sin_angle) / sqrt_three
  sorry

end height_of_square_pyramid_is_13_l194_194772


namespace problem_l194_194977

noncomputable theory

open Real

theorem problem
  (x y : ℝ)
  (h1 : ∀ n : ℕ, n > 0 → ∃ k : ℤ, x * (k : ℝ) = n - 1 ∧ k = ⌊ny⌋) :
  (x * y = 1) ∧ irrational y ∧ (y > 1) :=
by
  sorry

end problem_l194_194977


namespace greatest_integer_for_prime_abs_expression_l194_194719

open Int

-- Define the quadratic expression and the prime condition
def quadratic_expression (x : ℤ) : ℤ := 6 * x^2 - 47 * x + 15

-- Statement that |quadratic_expression x| is prime
def is_prime_quadratic_expression (x : ℤ) : Prop :=
  Prime (abs (quadratic_expression x))

-- Prove that the greatest integer x such that |quadratic_expression x| is prime is 8
theorem greatest_integer_for_prime_abs_expression :
  ∃ (x : ℤ), is_prime_quadratic_expression x ∧ (∀ (y : ℤ), is_prime_quadratic_expression y → y ≤ x) → x = 8 :=
by
  sorry

end greatest_integer_for_prime_abs_expression_l194_194719


namespace base_conversion_problem_l194_194320

theorem base_conversion_problem (b : ℕ) (h : b^2 + b + 3 = 34) : b = 6 :=
sorry

end base_conversion_problem_l194_194320


namespace find_a_l194_194270

theorem find_a (a : ℝ) (h : (2 + a * complex.I) / (1 + complex.I) = 3 + complex.I) : a = 4 :=
by
  sorry

end find_a_l194_194270


namespace complement_domain_l194_194286

open Set

variable (x : ℝ)

def f (x : ℝ) : ℝ := Real.sqrt (1 - x)
def M : Set ℝ := {x | 1 - x ≥ 0}
def complement_M : Set ℝ := {x | x > 1}

theorem complement_domain : complement_M = {x | x > 1} :=
by 
  unfold complement_M
  unfold M 
  sorry

end complement_domain_l194_194286


namespace total_jumps_400_l194_194309

theorem total_jumps_400 (ronald_jumps rupert_extra_jumps : ℕ) (H1: ronald_jumps = 157) (H2: rupert_extra_jumps = 86) : 
  let rupert_jumps := ronald_jumps + rupert_extra_jumps in
  let average_jumps := (ronald_jumps + rupert_jumps) / 2 in
  ronald_jumps + rupert_jumps = 400 :=
by
  have H3 : rupert_jumps = 157 + 86 := by
    rw [H1, H2]
    norm_num
  have H4 : average_jumps = (157 + (157 + 86)) / 2 := by 
    rw [H1, H2, H3]
    norm_num
  have H5 : average_jumps = 200 := by 
    norm_num
  have H6 : ronald_jumps + rupert_jumps = 157 + (157 + 86) := by 
    rw [H1, H3]
    norm_num
  rw [H6]
  norm_num

end total_jumps_400_l194_194309


namespace coordinates_of_point_M_l194_194225

theorem coordinates_of_point_M :
    ∀ (M : ℝ × ℝ),
      (M.1 < 0 ∧ M.2 > 0) → -- M is in the second quadrant
      dist (M.1, M.2) (M.1, 0) = 1 → -- distance to x-axis is 1
      dist (M.1, M.2) (0, M.2) = 2 → -- distance to y-axis is 2
      M = (-2, 1) :=
by
  intros M in_second_quadrant dist_to_x_axis dist_to_y_axis
  sorry

end coordinates_of_point_M_l194_194225


namespace truncated_tetrahedron_volume_square_eq_l194_194062

/-- Define a truncated tetrahedron with the given conditions: 4 triangles, 4 regular hexagons,
    each triangle borders 3 hexagons, and all sides are of length 1.
    Prove that the square of its volume is equal to 529/72. -/
theorem truncated_tetrahedron_volume_square_eq :
  ∀ (a : ℝ), 
  (a = 1) →
  (let V := 9 * a ^ 3 * real.sqrt 2 / 4) in
  let V_small := real.sqrt 2 / 3 in
  let V_truncated := V - V_small in
  let V_square := V_truncated ^ 2 in 
  V_square = 529 / 72 := 
by {
  intros,
  sorry
}

end truncated_tetrahedron_volume_square_eq_l194_194062


namespace total_money_l194_194968

theorem total_money (John Alice Bob : ℝ) (hJohn : John = 5 / 8) (hAlice : Alice = 7 / 20) (hBob : Bob = 1 / 4) :
  John + Alice + Bob = 1.225 := 
by 
  sorry

end total_money_l194_194968


namespace shirt_cost_is_43_l194_194650

def pantsCost : ℕ := 140
def tieCost : ℕ := 15
def totalPaid : ℕ := 200
def changeReceived : ℕ := 2

def totalCostWithoutShirt := totalPaid - changeReceived
def totalCostWithPantsAndTie := pantsCost + tieCost
def shirtCost := totalCostWithoutShirt - totalCostWithPantsAndTie

theorem shirt_cost_is_43 : shirtCost = 43 := by
  have h1 : totalCostWithoutShirt = 198 := by rfl
  have h2 : totalCostWithPantsAndTie = 155 := by rfl
  have h3 : shirtCost = totalCostWithoutShirt - totalCostWithPantsAndTie := by rfl
  rw [h1, h2] at h3
  exact h3

end shirt_cost_is_43_l194_194650


namespace at_least_two_equal_l194_194260

theorem at_least_two_equal
  {a b c d : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h₁ : a + b + (1 / (a * b)) = c + d + (1 / (c * d)))
  (h₂ : (1 / a) + (1 / b) + (a * b) = (1 / c) + (1 / d) + (c * d)) :
  a = c ∨ a = d ∨ b = c ∨ b = d ∨ a = b ∨ c = d := by
  sorry

end at_least_two_equal_l194_194260


namespace problem1_problem2_l194_194791

-- Let's define the first problem statement in Lean
theorem problem1 : 2 - 7 * (-3) + 10 + (-2) = 31 := sorry

-- Let's define the second problem statement in Lean
theorem problem2 : -1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14 := sorry

end problem1_problem2_l194_194791


namespace triangle_FE_ratio_l194_194299

universe u
variable {α : Type u}

-- Definitions to handle points, lines and triangles.
structure Point (α : Type u) :=
(x : α) (y : α)

structure Line (α : Type u) :=
(p1 : Point α) (p2 : Point α)

structure Triangle (α : Type u) :=
(A : Point α) (B : Point α) (C : Point α)

structure Ratio (α : Type u) :=
(num : α) (den : α)

-- Main theorem
theorem triangle_FE_ratio
  (α : Type u) [LinearOrderedField α]
  (A B C H D E F : Point α)
  (BH : Line α) (FE : Line α)
  (condition1 : BH.p1 = B ∧ BH.p2 = H)
  (condition2 : FE.p1 = F ∧ FE.p2 = E)
  (condition3 : D ∈ BH ∧ D ∈ Triangle α A B C)
  (condition4 : E ∈ Line α A D)
  (condition5 : F ∈ Line α C D)
  (condition6 : ∀ K : Point α, K ∈ BH → ∃ r : Ratio α, r = ⟨1, 3⟩ → Segment α F H E → Segment α F K E) :
  Ratio α FH HE = ⟨1, 3⟩ := sorry

end triangle_FE_ratio_l194_194299


namespace count_three_digit_numbers_l194_194897

def count_decreasing_digit_numbers : ℕ :=
  ∑ h in Finset.range 10 \ {0, 1}, ∑ t in Finset.range h, t

theorem count_three_digit_numbers :
  count_decreasing_digit_numbers = 120 :=
sorry

end count_three_digit_numbers_l194_194897


namespace equivalent_expression_l194_194729

-- Define the main components of the original expression, ensuring that they are equivalent to the parts given.
def expr : (a b : ℝ) → ℝ := 
  λ a b, 1.22 * (((sqrt a + sqrt b)^2 - 4 * b) / 
  ((a - b) / (sqrt (1 / b) + 3 * sqrt (1 / a)))) / 
  ((a + 9 * b + 6 * sqrt (a * b)) / 
  (1 / sqrt a + 1 / sqrt b))

-- Main statement to prove
theorem equivalent_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (expr a b = 1 / (a * b)) :=
sorry

end equivalent_expression_l194_194729


namespace orthocenter_on_BD_l194_194604

variables (Point : Type) [EuclideanSpace Point] 
variables (A B C D X Y : Point)
variables (AC AD BD : Line Point)
variables (PBC PCD : Point → Line Point) -- Perpendiculars from Point to Line

-- Definitions based on the conditions
def bisects_angle (AC : Line Point) (BA : Ray Point B) (AD : Ray Point D) : Prop :=
  ∃ (α : Angle Point), BA α ∧ AD α ∧ AC α

def angle_equality (ADC ACB : Angle Point) : Prop := 
  ADC = ACB

def perpendicular_foot (Point Line : Type) [EuclideanSpace Point] 
  (P : Point) (L : Line Point) : Point := 
  feet P L 

-- Proposition: Given the conditions, prove the orthocenter of triangle AXY lies on line BD
theorem orthocenter_on_BD 
(h1 : bisects_angle AC (Ray.mk A B) (Ray.mk A D))
(h2 : angle_equality (∠ADC) (∠ACB))
(h3 : X = perpendicular_foot A (PBC B))
(h4 : Y = perpendicular_foot A (PCD D)) :
  orthocenter (triangle A X Y) ∈ BD :=
sorry

end orthocenter_on_BD_l194_194604


namespace domain_of_c_is_all_reals_l194_194472

theorem domain_of_c_is_all_reals (k : ℝ) : 
  (∀ x : ℝ, -3 * x^2 + 5 * x + k ≠ 0) ↔ k < -(25 / 12) :=
by
  sorry

end domain_of_c_is_all_reals_l194_194472


namespace total_cell_phones_correct_l194_194348

-- Definitions for the problem conditions
def total_population : ℕ := 974000
def age_group_A_percentage : ℤ := 20
def age_group_B_percentage : ℤ := 65
def age_group_C_percentage : ℤ := 15

def cell_phone_rate_A : ℤ := 890
def cell_phone_rate_B : ℤ := 760
def cell_phone_rate_C : ℤ := 420

-- The expected total number of cell phones
def expected_total_phones : ℤ := 715690

-- Statement of the theorem to prove
theorem total_cell_phones_correct : 
  let population_A := (age_group_A_percentage * total_population) / 100,
      population_B := (age_group_B_percentage * total_population) / 100,
      population_C := (age_group_C_percentage * total_population) / 100,
      phones_A := (cell_phone_rate_A * population_A) / 1000,
      phones_B := (cell_phone_rate_B * population_B) / 1000,
      phones_C := (cell_phone_rate_C * population_C) / 1000,
      total_phones := phones_A + phones_B + phones_C
  in total_phones = expected_total_phones :=
by
  -- proof goes here
  sorry

end total_cell_phones_correct_l194_194348


namespace four_digit_square_palindromes_are_zero_l194_194177

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

-- Define the main theorem statement
theorem four_digit_square_palindromes_are_zero : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) → 
             is_palindrome n → 
             (∃ m : ℕ, n = m * m) → 
             n = 0 :=
by
  sorry

end four_digit_square_palindromes_are_zero_l194_194177


namespace subset_A_l194_194575

variable A : Set ℝ
variable h : A = {x | x > -1}

theorem subset_A : {0} ⊆ A := by
  -- sorry to indicate that the actual proof is not provided
  sorry

end subset_A_l194_194575


namespace positive_integers_satisfy_inequality_l194_194691

theorem positive_integers_satisfy_inequality :
  ∀ (n : ℕ), 2 * n - 5 < 5 - 2 * n ↔ n = 1 ∨ n = 2 :=
by
  intro n
  sorry

end positive_integers_satisfy_inequality_l194_194691


namespace num_4_digit_palindromic_squares_l194_194201

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_n (n : ℕ) : Prop :=
  32 ≤ n ∧ n ≤ 99

theorem num_4_digit_palindromic_squares : ∃ (count : ℕ), count = 3 ∧ ∀ n, valid_n n → is_4_digit (n^2) → is_palindrome (n^2) :=
sorry

end num_4_digit_palindromic_squares_l194_194201


namespace percent_increase_bike_helmet_l194_194616

theorem percent_increase_bike_helmet :
  let old_bike_cost := 160
  let old_helmet_cost := 40
  let bike_increase_rate := 0.05
  let helmet_increase_rate := 0.10
  let new_bike_cost := old_bike_cost * (1 + bike_increase_rate)
  let new_helmet_cost := old_helmet_cost * (1 + helmet_increase_rate)
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let increase_amount := new_total_cost - old_total_cost
  let percent_increase := (increase_amount / old_total_cost) * 100
  percent_increase = 6 :=
by
  sorry

end percent_increase_bike_helmet_l194_194616


namespace evaluate_expression_l194_194810

theorem evaluate_expression : ( (5 : ℝ) / 6 ) ^ 4 * ( (5 : ℝ) / 6 ) ^ (-4) = 1 :=
by {
  sorry
}

end evaluate_expression_l194_194810


namespace value_of_k_l194_194537

theorem value_of_k
  (a : ℕ → ℕ )
  (h_increasing : ∀ m n, m < n → a m < a n)
  (h_nonneg : ∀ n, 0 ≤ a n)
  (h_sum : ∑ i in finset.Icc 1 k, 2^(a i) = (2^289 + 1) / (2^17 + 1)) :
  k = 137 :=
sorry

end value_of_k_l194_194537


namespace tangent_line_equation_l194_194681

-- Given function f and the point of tangency
def f (x : ℝ) : ℝ := x^3 - x + 1
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Derivative of f
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Statement asserting the equation of the tangent line at the given point
theorem tangent_line_equation : 
  let (x₀, y₀) := point_of_tangency in
  let k := f' x₀ in
  k = 2 ∧ ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -1 ∧ (∀ x y : ℝ, y - y₀ = k * (x - x₀) ↔ 2 * x - y - 1 = 0) :=
by
  let (x₀, y₀) := (1, 1)
  let k := 2
  have k_def : f' x₀ = k := by sorry
  use 2, -1, -1
  constructor
  exact k_def
  constructor
  rfl
  constructor
  rfl
  intro x y
  simp
  sorry

end tangent_line_equation_l194_194681


namespace ball_hits_ground_l194_194680

def height (t : ℚ) : ℚ := -490/100 * t^2 + 5 * t + 10

theorem ball_hits_ground : 
  ∃ t : ℚ, height t = 0 ∧ t = 20/7 :=
by
  sorry

end ball_hits_ground_l194_194680


namespace isosceles_triangle_base_length_l194_194784

theorem isosceles_triangle_base_length
  (perimeter : ℝ)
  (side1 side2 base : ℝ)
  (h_perimeter : perimeter = 18)
  (h_side1 : side1 = 4)
  (h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base)
  (h_triangle : side1 + side2 + base = 18) :
  base = 7 := 
sorry

end isosceles_triangle_base_length_l194_194784


namespace number_of_elements_in_T_l194_194634

def g (x : ℝ) : ℝ := (2 * x + 9) / x

def g_seq : ℕ → (ℝ → ℝ)
| 0       => g
| (n + 1) => g ∘ g_seq n

def T : Set ℝ := {x | ∃ n : ℕ, g_seq n x = x}

theorem number_of_elements_in_T : 
  Set.card T = 2 :=
sorry

end number_of_elements_in_T_l194_194634


namespace coeff_of_x3_in_expansion_l194_194867

def polynomial := (x + 1 / x) * (a * x + 1) ^ 5

def sum_of_coefficients_eq_64 (a : ℝ) : Prop :=
  2 * (a + 1) ^ 5 = 64

theorem coeff_of_x3_in_expansion (a : ℝ) (h : sum_of_coefficients_eq_64 a) : coeff (expand polynomial) x^3 = 15 :=
sorry

end coeff_of_x3_in_expansion_l194_194867


namespace probability_of_multiple_6_or_8_l194_194662

def is_probability_of_multiple_6_or_8 (n : ℕ) : Prop := 
  let num_multiples (k : ℕ) := n / k
  let multiples_6 := num_multiples 6
  let multiples_8 := num_multiples 8
  let multiples_24 := num_multiples 24
  let total_multiples := multiples_6 + multiples_8 - multiples_24
  total_multiples / n = 1 / 4

theorem probability_of_multiple_6_or_8 : is_probability_of_multiple_6_or_8 72 :=
  by sorry

end probability_of_multiple_6_or_8_l194_194662


namespace total_amount_after_ten_years_l194_194036

-- Define the initial investment details and calculate the total investment value after 10 years.
noncomputable def initial_amount : ℝ := 8000
noncomputable def annual_interest_rate : ℝ := 0.08
noncomputable def annual_payment : ℝ := 500
noncomputable def investment_period : ℕ := 10

noncomputable def compounded_initial_investment (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

noncomputable def future_value_annuity (P r : ℝ) (n : ℕ) : ℝ :=
  P * ((1 + r) ^ n - 1) / r

noncomputable def total_investment_value (initial P r : ℝ) (n : ℕ) : ℝ :=
  compounded_initial_investment initial r n + future_value_annuity P r n

theorem total_amount_after_ten_years :
  total_investment_value initial_amount annual_payment annual_interest_rate investment_period ≈ 24515 :=
by
  -- Replace the exact proof details because we're not concerned with the solution steps in the Lean statement.
  sorry

end total_amount_after_ten_years_l194_194036


namespace train_speed_is_60_l194_194776

noncomputable def train_speed_proof : Prop :=
  let train_length := 550 -- in meters
  let time_to_pass := 29.997600191984645 -- in seconds
  let man_speed_kmhr := 6 -- in km/hr
  let man_speed_ms := man_speed_kmhr * (1000 / 3600) -- converting km/hr to m/s
  let relative_speed_ms := train_length / time_to_pass -- relative speed in m/s
  let train_speed_ms := relative_speed_ms - man_speed_ms -- speed of the train in m/s
  let train_speed_kmhr := train_speed_ms * (3600 / 1000) -- converting m/s to km/hr
  train_speed_kmhr = 60 -- the speed of the train in km/hr

theorem train_speed_is_60 : train_speed_proof := by
  sorry

end train_speed_is_60_l194_194776


namespace five_coins_cannot_sum_to_40_l194_194100

def is_combination_possible (coins : List Nat) (n : Nat) : Prop :=
  ∃ (a b c d e : ℕ), a * coins.head + b * coins.get 1 + c * coins.get 2 + d * coins.get 3 + e * coins.get 4 = n

theorem five_coins_cannot_sum_to_40 (a b c d e : ℕ)
    (h₁ : a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 40) :
  a + b + c + d + e ≠ 5 :=
sorry

end five_coins_cannot_sum_to_40_l194_194100


namespace sphere_surface_area_l194_194767

def right_square_pyramid_height : ℝ := 4
def right_square_pyramid_volume : ℝ := 16

theorem sphere_surface_area :
  ∀ (h : ℝ) (V : ℝ), 
  h = right_square_pyramid_height →
  V = right_square_pyramid_volume →
  (surface_area : ℝ) (radius : ℝ),
  radius = (sqrt(6)) →
  surface_area = 4 * real.pi * radius^2 →
  surface_area = 24 * real.pi :=
by
  intros h V h_def V_def surface_area radius
  assume radius_def surface_area_def
  sorry

end sphere_surface_area_l194_194767


namespace correct_observation_l194_194339

-- Given constants and conditions
constant n : Nat := 50
constant original_mean : ℝ := 36
constant new_mean : ℝ := 36.5
constant wrong_observation : ℝ := 23

-- Summarizing derived total sums
constant original_total_sum : ℝ := n * original_mean
constant new_total_sum : ℝ := n * new_mean

-- The statement we want to prove
theorem correct_observation : 
  ∃ x : ℝ, original_total_sum - wrong_observation + x = new_total_sum := 
sorry

end correct_observation_l194_194339


namespace exists_cycle_not_multiple_of_3_l194_194931

variable {V : Type} [Fintype V]
variable {adj : V → V → Prop}

theorem exists_cycle_not_multiple_of_3 
  (H : ∀ v : V, (∃ (neighbors : Finset V), neighbors.card ≥ 3 ∧ (∀ w ∈ neighbors, adj v w)))
  : ∃ (cycle : List V), cycle.Nodup ∧ cycle.Head = (cycle.getLast sorry) ∧ ¬ 3 ∣ (cycle.length - 1) :=
sorry

end exists_cycle_not_multiple_of_3_l194_194931


namespace symmetric_difference_solution_l194_194499

noncomputable def A : Set ℝ := {x | x ≥ -9/4}
noncomputable def B : Set ℝ := {x | x < 0}

noncomputable def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
noncomputable def symmetric_difference (M N : Set ℝ) : Set ℝ := set_difference(M, N) ∪ set_difference(N, M)

theorem symmetric_difference_solution : symmetric_difference(A, B) = {x | x ≥ 0 ∨ x < -9/4} :=
by sorry

end symmetric_difference_solution_l194_194499


namespace total_amount_spent_l194_194255

variables (original_price_backpack : ℕ) (increase_backpack : ℕ) (original_price_binder : ℕ) (decrease_binder : ℕ) (num_binders : ℕ)

-- Given Conditions
def original_price_backpack := 50
def increase_backpack := 5
def original_price_binder := 20
def decrease_binder := 2
def num_binders := 3

-- Prove the total amount spent is $109
theorem total_amount_spent :
  (original_price_backpack + increase_backpack) + (num_binders * (original_price_binder - decrease_binder)) = 109 := by
  sorry

end total_amount_spent_l194_194255


namespace third_side_length_l194_194944

noncomputable def triangle_third_side_length (a b : ℝ) (α β : ℝ) : ℝ :=
  if h₀ : b ≠ 0 ∧ cos α ≥ 0 ∧ cos β ≥ 0 then
    have h₁ : α = 3 * β, from sorry,
    have h₂ : b = 6, from sorry,
    have h₃ : a = 18, from sorry,
    have h₄ : sin α = 3 * sin β - 4 * sin β ^ 3, from sorry,
    let cosC : ℝ := cos β in
    let sinC : ℝ := sin β in
    have h₅ : cosC = (a^2 + b^2 - 6^2) / (2 * a * 6), from sorry,
    let cos3C := \(\cos (3 * β) = 4 \cos β^3 - 3 \cos β\) in
    have h₆ : sin β = sqrt (1 / 2), from sorry,
    have h₇ : cos β = sqrt (1 / 2), from sorry,
    let lhs : ℝ := (a^2 + 288) / (36 * a) in
    let rhs : ℝ := sqrt(2) / 2 in
    have h₈ : lhs = rhs, from sorry,
    let equation : ℝ := a^2 - 18 * sqrt (2) * a + 288 = 0 in
    let sol1 := 24 * sqrt(2) in
    let sol2 := 12 * sqrt(2) in
    if sol1 in [24*sqrt(2)] then sol1 else sol2
  else 0

theorem third_side_length (a b : ℝ) (α β : ℝ) (h : triangle_third_side_length 18 6 (3 * β) β = 24 * sqrt 2 : Prop) :
  triangle_third_side_length 18 6 (3 * β) β = 24 * sqrt 2 :=
sorry

end third_side_length_l194_194944


namespace problem_inequality_l194_194998

variable {a b c : ℝ}

-- Assuming a, b, c are positive real numbers
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Assuming abc = 1
variable (h_abc : a * b * c = 1)

theorem problem_inequality :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by sorry

end problem_inequality_l194_194998


namespace sum_possible_student_numbers_l194_194429

theorem sum_possible_student_numbers : 
  let S := {s : ℕ | 150 ≤ s ∧ s ≤ 200 ∧ (s % 8 = 1)} in 
  ∑ s in S, s = 1038 :=
by
  sorry

end sum_possible_student_numbers_l194_194429


namespace length_PS_l194_194609

noncomputable def triangle_PQR : Type := 
{ P Q R S : Type // 
  (PQ : ℝ) (QR : ℝ) (PR : ℝ) 
  (PQ_eq_8 : PQ = 8) 
  (QR_eq_15 : QR = 15) 
  (PR_eq_17 : PR = 17)
  (PS : Q → P → S)
  (angle_bisector : ∀ Q P S, PS = (Q, P, S))
}

theorem length_PS (P Q R S : Type) [triangle_PQR P Q R S]: 
  (@triangle_PQR.PS P Q R S (triangle_PQR.angle_bisector Q P S)) = sqrt 799 / 4 :=
sorry

end length_PS_l194_194609


namespace imag_part_of_complex_l194_194490

open Complex

theorem imag_part_of_complex : (im ((5 + I) / (1 + I))) = -2 :=
by
  sorry

end imag_part_of_complex_l194_194490


namespace intersection_points_value_l194_194466

def f (x : ℝ) : ℝ := 2 * (x - 2) * (x + 3)
def g (x : ℝ) : ℝ := -f(x)
def h (x : ℝ) : ℝ := f(-x)

def a : ℕ := 2 -- manually assign based on solution steps
def b : ℕ := 1 -- manually assign based on solution steps

theorem intersection_points_value :
  10 * a + b = 21 :=
by
  sorry

end intersection_points_value_l194_194466


namespace find_x_l194_194780

theorem find_x (x : ℕ) : 
  (∃ (students : ℕ), students = 10) ∧ 
  (∃ (selected : ℕ), selected = 6) ∧ 
  (¬ (∃ (k : ℕ), k = 5 ∧ k = x) ) ∧ 
  (1 ≤ 10 - x) ∧
  (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

end find_x_l194_194780


namespace baron_munchausen_not_lying_l194_194045

def sum_of_digits (n : Nat) : Nat := sorry

theorem baron_munchausen_not_lying :
  ∃ a b : Nat, a ≠ b ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a < 10^10 ∧ 10^9 ≤ a) ∧ (b < 10^10 ∧ 10^9 ≤ b) ∧ 
  (a + sum_of_digits (a ^ 2) = b + sum_of_digits (b ^ 2)) :=
sorry

end baron_munchausen_not_lying_l194_194045


namespace duration_trip_for_cyclist1_l194_194714

-- Definitions
variable (s : ℝ) -- the speed of Cyclist 1 without wind in km/h
variable (t : ℝ) -- the time in hours it takes for Cyclist 1 to travel from A to B
variable (wind_speed : ℝ := 3) -- wind modifies speed by 3 km/h
variable (total_time : ℝ := 4) -- total time after which cyclists meet

-- Conditions
axiom consistent_speed_aid : ∀ (s t : ℝ), t > 0 → (s + wind_speed) * t + (s - wind_speed) * (total_time - t) / 2 = s - wind_speed * total_time

-- Goal (equivalent proof problem)
theorem duration_trip_for_cyclist1 : t = 2 := by
  sorry

end duration_trip_for_cyclist1_l194_194714


namespace distinct_cube_constructions_l194_194753

theorem distinct_cube_constructions : 
  let cubes := [3, 3, 2] in
  -- This statement expresses that there are 24 distinct ways
  -- to construct the cube given the initial conditions.
  number_of_distinct_constructions cubes = 24 := 
sorry

end distinct_cube_constructions_l194_194753


namespace calc_expr_l194_194046

theorem calc_expr : (3^5 * 6^3 + 3^3) = 52515 := by
  sorry

end calc_expr_l194_194046


namespace problem1_problem2_l194_194126

variables {a b c x1 x2 : ℝ}

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Provided conditions
axiom minimum_at_2 : ∀ a b c, (deriv (quadratic_function a b c) 2) = 0
axiom inter_x_axis_at_x1_x2 : ∀ a b c, quadratic_function a b c x1 = 0 ∧ quadratic_function a b c x2 = 0
axiom tan_diff_condition : ∀ a b c x1 x2, x1 < 0 ∧ x2 > 0 → (c / x1) - (-c / x2) = 1

-- Proving the first part
theorem problem1 : b + 4 * a = 0 :=
by sorry

-- Proving the second part
theorem problem2 : a = 1 / 4 ∧ b = -1 :=
by sorry

end problem1_problem2_l194_194126


namespace segment_length_l194_194125

noncomputable def line_through_P : ℝ → ℝ × ℝ :=
  λ t, (-3 + (sqrt 3 / 2) * t, (1 / 2) * t)

def curve (x y: ℝ) : Prop := x^2 - y^2 = 4

theorem segment_length :
  ∃ t1 t2 : ℝ, (line_through_P t1).fst ^ 2 - (line_through_P t1).snd ^ 2 = 4 ∧
               (line_through_P t2).fst ^ 2 - (line_through_P t2).snd ^ 2 = 4 ∧
               abs (t1 - t2) = 2 * sqrt 17 :=
sorry

end segment_length_l194_194125


namespace max_value_of_a_l194_194863

theorem max_value_of_a (a b c d : ℤ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 := by 
  sorry

end max_value_of_a_l194_194863


namespace valid_password_count_l194_194782

def digit := Fin 10

def is_valid_password (a b c d : digit) : Prop :=
  ¬ (c = 4 ∧ d = 5)

def count_valid_passwords : ℕ :=
  (Finset.univ.product (Finset.univ.product (Finset.univ.product Finset.univ))).filter (λ p, is_valid_password p.1.1.1 p.1.1.2 p.1.2 p.2)
  .card

theorem valid_password_count : count_valid_passwords = 9900 := by
  sorry

end valid_password_count_l194_194782


namespace count_four_digit_integers_l194_194569

theorem count_four_digit_integers :
    ∃! (a b c d : ℕ), 1 ≤ a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (10 * b + c)^2 = (10 * a + b) * (10 * c + d) := sorry

end count_four_digit_integers_l194_194569


namespace problem_statement_l194_194430

noncomputable def a : ℕ → ℝ
| 0       := a0
| 1       := a1
| (n+2) := 1 + a (n + 1) / a n

theorem problem_statement
  (a0 a1 : ℝ)
  (h0 : 0 < a0)
  (h1 : 0 < a1) :
  |a 2012 - 2| < 10 ^ (-200) :=
sorry

end problem_statement_l194_194430


namespace common_difference_of_arithmetic_sequence_l194_194845

theorem common_difference_of_arithmetic_sequence :
  ∀ (a : ℕ → ℤ) (n : ℕ) (S_odd S_even : ℤ) (d : ℤ),
    n = 20 →
    (∑ i in finset.filter (λ i, odd i) (finset.range n), a i) = S_odd →
    (∑ i in finset.filter (λ i, even i) (finset.range n), a i) = S_even →
    S_odd = 132 →
    S_even = 112 →
    d = -2 :=
by
  intros a n S_odd S_even d hn hSodd hSeven hSodd132 hSeven112
  sorry

end common_difference_of_arithmetic_sequence_l194_194845


namespace find_a_l194_194168

theorem find_a (a : ℝ) (A B : set ℝ) (hA : A = {a^2, a+1, -3}) (hB : B = {a-3, a^2+1, 2a-1}) (hAB : A ∩ B = {-3}) : a = -1 :=
sorry

end find_a_l194_194168


namespace set_intersection_l194_194169

variable {α : Type*} [LinearOrder α]

def P (x : α) : Prop := x - 1 ≤ 0

def Q (x : α) : Prop := (x - 2) / x ≤ 0

noncomputable def complement_P := {x : α | ¬ P x}

noncomputable def intersection := {x : α | complement_P x ∧ Q x}

theorem set_intersection :
  (complement_P ∩ Q : set ℝ) = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end set_intersection_l194_194169


namespace domain_w_l194_194717

noncomputable def w (x : ℝ) : ℝ := real.sqrt (x - 2) + real.cbrt (x - 3)

theorem domain_w : {x : ℝ | ∃ y : ℝ, w x = y} = set.Ici 2 := by
  sorry

end domain_w_l194_194717


namespace sum_of_integer_solutions_in_inequality_l194_194401

theorem sum_of_integer_solutions_in_inequality :
  (∀ x : ℝ,
    5 * x - 11 >= 0 ∧ 5 * x^2 - 21 * x + 21 >= 0 →
    sqrt (5 * x - 11) - sqrt (5 * x^2 - 21 * x + 21) >=
    5 * x^2 - 26 * x + 32) →
  ∑ x in {x : ℤ | 5 * (x : ℝ) - 11 >= 0 ∧ 
              5 * (x : ℝ)^2 - 21 * (x : ℝ) + 21 >= 0 ∧ 
              sqrt (5 * (x : ℝ) - 11) - sqrt (5 * (x : ℝ)^2 - 21 * (x : ℝ) + 21) >= 5 * (x : ℝ)^2 - 26 * (x : ℝ) + 32}, x =
  3 :=
  sorry

end sum_of_integer_solutions_in_inequality_l194_194401


namespace concurrency_of_lines_l194_194955

-- Define the geometrical setup, points, and conditions
variables (A B C A' B' C' P M N K L : Type)

-- Definitions of the lines a, b, and c in terms of midpoints of segments
def line_a := sorry -- Line a defined through midpoints (requires actual definitions based on midpoints logic)
def line_b := sorry -- Line b defined similarly
def line_c := sorry -- Line c passing through midpoints of MN and KL

-- The proof that the lines a, b, and c intersect at a single point
theorem concurrency_of_lines 
  (triangle_ABC : triangle A B C) 
  (cevians_at_P : cevians_intersect_at A' B' C' P)
  (circ_PA'B'_intersects_AC_BC : circumcircle (triangle P A' B') intersects AC at M intersects BC at N)
  (circ_PC'B'_intersects_AC_at_K : circumcircle (triangle P C' B') intersects AC at K)
  (circ_PA'C'_intersects_BC_at_L : circumcircle (triangle P A' C') intersects BC at L)
  (line_c_through_midpoints : line_passes_through_midpoints MN KL line_c)
  (line_a_similar_define : line_passes_through_midpoints (analogous_segments) line_a)
  (line_b_similar_define : line_passes_through_midpoints (analogous_segments) line_b) :
  lines_concurrent line_a line_b line_c :=
begin
  sorry -- Proof required
end

end concurrency_of_lines_l194_194955


namespace integer_k_values_l194_194920

noncomputable def is_integer_solution (k x : ℤ) : Prop :=
  ((k - 2013) * x = 2015 - 2014 * x)

theorem integer_k_values (k : ℤ) (h : ∃ x : ℤ, is_integer_solution k x) :
  ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_k_values_l194_194920


namespace range_g_l194_194827

def g (x : ℝ) : ℝ :=
if x ≠ -5 then (3 * (x - 4)) else 27

theorem range_g : {y : ℝ | ∃ x : ℝ, x ≠ -5 ∧ y = g x } = {y : ℝ | y ≠ -27} :=
by sorry

end range_g_l194_194827


namespace time_spent_answering_questions_l194_194892

theorem time_spent_answering_questions (total_questions answered_per_question_minutes unanswered_questions : ℕ) (minutes_per_hour : ℕ) :
  total_questions = 100 → unanswered_questions = 40 → answered_per_question_minutes = 2 → minutes_per_hour = 60 → 
  ((total_questions - unanswered_questions) * answered_per_question_minutes) / minutes_per_hour = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end time_spent_answering_questions_l194_194892


namespace no_parallel_no_perpendicular_l194_194287

variables {P : Type} [plane P]
variables {L : Type} [line L]
variables {α β l m n : L}
variables {α' β' : P}

-- Definitions of the conditions
axiom m_in_alpha : m ∈ α'
axiom n_in_beta : n ∈ β'
axiom alpha_perp_beta : ∀ (p : P) (l1 l2 : L), (l1 ∈ p ∧ l2 ∈ p) → l1 ⊥ l2
axiom alpha_cap_beta_l : α' ∩ β' = l
axiom m_not_parallel_l : ¬ (parallel m l)
axiom m_not_perp_l : ¬ (m ⊥ l)
axiom n_not_parallel_l : ¬ (parallel n l)
axiom n_not_perp_l : ¬ (n ⊥ l)

-- Proof goal
theorem no_parallel_no_perpendicular (m_in_alpha : m ∈ α') (n_in_beta : n ∈ β') (alpha_perp_beta : ∀ (p : P) (l1 l2 : L), (l1 ∈ p ∧ l2 ∈ p) → l1 ⊥ l2)
    (alpha_cap_beta_l : α' ∩ β' = l) (m_not_parallel_l : ¬ (parallel m l)) (m_not_perp_l : ¬ (m ⊥ l)) (n_not_parallel_l : ¬ (parallel n l)) (n_not_perp_l : ¬ (n ⊥ l)) : 
    ¬ (parallel m n) ∧ ¬ (m ⊥ n) :=
by 
  sorry

end no_parallel_no_perpendicular_l194_194287


namespace mark_and_james_need_to_buy_2_dice_to_play_their_game_l194_194640

variable (total_dice_needed : ℕ)
variable (mark_total_dice : ℕ)
variable (mark_percent_12_sided : ℝ)
variable (james_total_dice : ℕ)
variable (james_percent_12_sided : ℝ)

def number_of_dice_to_buy : ℕ :=
  let mark_12_sided := mark_percent_12_sided * mark_total_dice
  let james_12_sided := james_percent_12_sided * james_total_dice
  total_dice_needed - mark_12_sided.toNat - james_12_sided.toNat

theorem mark_and_james_need_to_buy_2_dice_to_play_their_game :
  total_dice_needed = 14 →
  mark_total_dice = 10 →
  mark_percent_12_sided = 0.6 →
  james_total_dice = 8 →
  james_percent_12_sided = 0.75 →
  number_of_dice_to_buy total_dice_needed mark_total_dice mark_percent_12_sided james_total_dice james_percent_12_sided = 2 :=
by
  sorry

end mark_and_james_need_to_buy_2_dice_to_play_their_game_l194_194640


namespace angle_FHG_is_45_l194_194951

theorem angle_FHG_is_45 (EF GH IJ : line) (E F G H I J : point)
  (h1 : parallel EF GH) 
  (h2 : intersects IJ EF F) 
  (h3 : intersects IJ GH H) 
  (h4 : angle E H G = 53) 
  (h5 : angle G H J = 82) :
  angle F H G = 45 := 
by 
  sorry

end angle_FHG_is_45_l194_194951


namespace find_a_l194_194508

def f (a : ℝ) : ℝ → ℝ := λ x, a * x + 4

theorem find_a (a : ℝ) (h1 : (deriv (f a) 1) = 2) : a = 2 := by
  sorry

end find_a_l194_194508


namespace line_perpendicular_passes_through_circumcenter_l194_194241

-- Definitions of points and properties as given in the problem.
variables {A B C D E F P X Y Q : Point}
-- Conditions
axiom h1 : convex_quadrilateral A B C D
axiom h2 : collinear {A, B, E}
axiom h3 : collinear {C, D, E}
axiom h4 : collinear {A, D, F}
axiom h5 : collinear {B, C, F}
axiom h6 : line_intersection (line_through A C) (line_through B D) = P
axiom h7 : circle_tangent_at (circle_through D) (line_through A C) P
axiom h8 : circle_tangent_at (circle_through C) (line_through B D) P
axiom h9 : line_intersection (line_through A D) (circle_through D) = X
axiom h10 : line_intersection (line_through B C) (circle_through C) = Y
axiom h11 : circle_intersection (circle_through D) (circle_through C) P Q

-- Statement to be proved
theorem line_perpendicular_passes_through_circumcenter :
  ∃ O : Point,
    is_circumcenter O X Q Y ∧
    (line_through P O).perpendicular (line_through E F) :=
sorry

end line_perpendicular_passes_through_circumcenter_l194_194241


namespace sum_of_digits_nine_ab_is_36410_l194_194635

-- Define a to be an integer consisting of a sequence of 2023 nines
def a : ℕ := (10^2023 - 1)

-- Define b to be an integer consisting of a sequence of 2023 sevens
def b : ℕ := (7 * (10^2023 - 1)) / 9

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum_nat

-- Define the function to calculate 9ab
def nine_ab (a b : ℕ) : ℕ :=
  9 * a * b

-- The theorem to be proven
theorem sum_of_digits_nine_ab_is_36410 :
  sum_of_digits (nine_ab a b) = 36410 := 
sorry

end sum_of_digits_nine_ab_is_36410_l194_194635


namespace main_theorem_l194_194218

def inequality_hold (x : ℝ) (hx : x > 0) : Prop :=
  (x / (1 + x)) < real.sqrt x

theorem main_theorem (x : ℝ) (hx : x > 0) : inequality_hold x hx :=
sorry

end main_theorem_l194_194218


namespace number_of_positive_expressions_l194_194881

-- Define the conditions
variable (a b c : ℝ)
variable (h_a : a < 0)
variable (h_b : b > 0)
variable (h_c : c < 0)

-- Define the expressions
def ab := a * b
def ac := a * c
def a_b_c := a + b + c
def a_minus_b_c := a - b + c
def two_a_plus_b := 2 * a + b
def two_a_minus_b := 2 * a - b

-- Problem statement
theorem number_of_positive_expressions :
  (ab < 0) → (ac > 0) → (a_b_c > 0) → (a_minus_b_c < 0) → (two_a_plus_b < 0) → (two_a_minus_b < 0)
  → (2 = 2) :=
by
  sorry

end number_of_positive_expressions_l194_194881


namespace maximum_a_l194_194105

noncomputable def validate_inequality : Prop :=
  ∀ x : ℝ, x ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) →
    (sqrt[3] (Real.tan x) - sqrt[3] (Real.cot x)) / (sqrt[3] (Real.sin x) + sqrt[3] (Real.cos x)) > (4 * Real.root (Real.sixthRoot 2)) / 2

theorem maximum_a : validate_inequality :=
begin
  sorry,
end

end maximum_a_l194_194105


namespace smallest_geometric_number_l194_194792

noncomputable def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def is_smallest_geometric_number (n : ℕ) : Prop :=
  n = 261

theorem smallest_geometric_number :
  ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10)) ∧
  (n / 100 = 2) ∧ (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  is_smallest_geometric_number n :=
by
  sorry

end smallest_geometric_number_l194_194792


namespace min_value_of_f_l194_194343

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ x ∈ Set.Icc (Real.exp 0) (Real.exp 3), f x = 6 - 2 * Real.log 2 :=
sorry

end min_value_of_f_l194_194343


namespace extremum_points_range_of_a_l194_194061

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log (x + 1) + a * (x^2 - x)

-- Statement 1: Extremum points
theorem extremum_points (a : ℝ) :
  (a < 0 → (∃ x, (Real.log_deriv (x + 1) + 2 * a * x - a) = 0)) ∧
  (0 ≤ a ∧ a ≤ 8/9 → ¬(∃ x, (Real.log_deriv (x + 1) + 2 * a * x - a) = 0)) ∧
  (a > 8/9 → ∃ x1 x2, x1 < x2 ∧ (x1 < 0 ∧ 0 < x2) ∧ (Real.log_deriv (x1 + 1) + 2 * a * x1 - a) = 0 ∧ (Real.log_deriv (x2 + 1) + 2 * a * x2 - a) = 0) := sorry

-- Statement 2: Range of a for non-negative f(x)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → f x a ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := sorry

end extremum_points_range_of_a_l194_194061


namespace coordinates_of_point_M_l194_194226

theorem coordinates_of_point_M :
    ∀ (M : ℝ × ℝ),
      (M.1 < 0 ∧ M.2 > 0) → -- M is in the second quadrant
      dist (M.1, M.2) (M.1, 0) = 1 → -- distance to x-axis is 1
      dist (M.1, M.2) (0, M.2) = 2 → -- distance to y-axis is 2
      M = (-2, 1) :=
by
  intros M in_second_quadrant dist_to_x_axis dist_to_y_axis
  sorry

end coordinates_of_point_M_l194_194226


namespace g_at_6_is_zero_l194_194629

def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

theorem g_at_6_is_zero : g 6 = 0 :=
by {
  sorry
}

end g_at_6_is_zero_l194_194629


namespace num_values_divisible_by_120_l194_194570

theorem num_values_divisible_by_120 (n : ℕ) (h_seq : ∀ n, ∃ k, n = k * (k + 1)) :
  ∃ k, k = 8 := sorry

end num_values_divisible_by_120_l194_194570


namespace symmetric_points_origin_l194_194149

theorem symmetric_points_origin (a b : ℝ) (h1 : 1 = -b) (h2 : a = 2) : a + b = 1 := by
  sorry

end symmetric_points_origin_l194_194149


namespace Mark_James_dice_problem_l194_194648

theorem Mark_James_dice_problem : 
  let mark_total_dice := 10
  let mark_percent_12_sided := 0.6
  let james_total_dice := 8
  let james_percent_12_sided := 0.75
  let total_needed := 14
  let mark_12_sided := mark_total_dice * mark_percent_12_sided
  let james_12_sided := james_total_dice * james_percent_12_sided
  let total_12_sided := mark_12_sided + james_12_sided
  let dice_to_buy := total_needed - total_12_sided
  ⟶ dice_to_buy = 2 := by sorry

end Mark_James_dice_problem_l194_194648


namespace math_problem_l194_194224

theorem math_problem (x y : ℝ) (h1 : x - 2 * y = 4) (h2 : x * y = 8) :
  x^2 + 4 * y^2 = 48 :=
sorry

end math_problem_l194_194224


namespace rectangle_y_value_l194_194936

theorem rectangle_y_value (y : ℝ) :
  let length := (1 - (-3)),
  let height := (y - (-2)),
  length * height = 12 →
  y = 1 :=
by
  assume h : (1 - (-3)) * (y - (-2)) = 12
  sorry

end rectangle_y_value_l194_194936


namespace pupils_count_l194_194594

-- Definitions based on given conditions
def number_of_girls : ℕ := 692
def girls_more_than_boys : ℕ := 458
def number_of_boys : ℕ := number_of_girls - girls_more_than_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

-- The statement that the total number of pupils is 926
theorem pupils_count : total_pupils = 926 := by
  sorry

end pupils_count_l194_194594


namespace total_tables_l194_194012

variables (F T : ℕ)

-- Define the given conditions
def condition1 := F = 16
def condition2 := 4 * F + 3 * T = 124

-- State the theorem given the conditions to prove the total number of tables.
theorem total_tables (h1 : condition1) (h2 : condition2) : F + T = 36 :=
by
  -- This is a placeholder as we are skipping the proof itself
  sorry

end total_tables_l194_194012


namespace magic_square_find_x_l194_194234

theorem magic_square_find_x :
  ∃ x, x = 208 ∧
    (∀ a b c d e f g h : ℤ,
      x + 5 + f = 102 + d + f ∧
      d = x - 97 ∧
      x + (x - 97) + h = 102 + e + h ∧
      e = 2x - 199 ∧
      x + 23 + 102 = 5 + (x - 97) + (2x - 199)) :=
by 
  sorry

end magic_square_find_x_l194_194234


namespace eval_sqrt_5_of_8_pow_15_l194_194479

theorem eval_sqrt_5_of_8_pow_15 : (8^(1/5:ℝ))^15 = 512 :=
by
  sorry

end eval_sqrt_5_of_8_pow_15_l194_194479


namespace ratio_distance_l194_194504

-- Definitions based on conditions
def speed_ferry_P : ℕ := 6 -- speed of ferry P in km/h
def time_ferry_P : ℕ := 3 -- travel time of ferry P in hours
def speed_ferry_Q : ℕ := speed_ferry_P + 3 -- speed of ferry Q in km/h
def time_ferry_Q : ℕ := time_ferry_P + 1 -- travel time of ferry Q in hours

-- Calculating the distances
def distance_ferry_P : ℕ := speed_ferry_P * time_ferry_P -- distance covered by ferry P
def distance_ferry_Q : ℕ := speed_ferry_Q * time_ferry_Q -- distance covered by ferry Q

-- Main theorem to prove
theorem ratio_distance (d_P d_Q : ℕ) (h_dP : d_P = distance_ferry_P) (h_dQ : d_Q = distance_ferry_Q) : d_Q / d_P = 2 :=
by
  sorry

end ratio_distance_l194_194504


namespace k_ge_1_l194_194361

theorem k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 2 * x + 9 > 6 * x + 1 ∧ x - k < 1 → x < 2) → k ≥ 1 :=
by 
  sorry

end k_ge_1_l194_194361


namespace train_crossing_platform_time_l194_194746

noncomputable def train_length : ℝ := 300
noncomputable def signal_pole_time : ℝ := 10
noncomputable def platform_length : ℝ := 870

theorem train_crossing_platform_time :
  let speed := train_length / signal_pole_time in
  let total_distance := train_length + platform_length in
  let crossing_time := total_distance / speed in
  crossing_time = 39 :=
by 
  sorry

end train_crossing_platform_time_l194_194746


namespace det_A_squared_sub_2A_eq_25_l194_194573

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 3], ![2, 1]]

-- The statement to prove
theorem det_A_squared_sub_2A_eq_25 : det (A ⬝ A - (2 : ℝ) • A) = 25 :=
by
  sorry

end det_A_squared_sub_2A_eq_25_l194_194573


namespace jacks_walking_speed_l194_194958

/-- Definitions of initial distances and speeds --/
def initial_distance : ℝ := 240
def christina_speed : ℝ := 3
def lindy_speed : ℝ := 9
def lindy_distance : ℝ := 270

/-- Theorem: Prove that Jack's walking speed v is 5 feet per second given the conditions --/
theorem jacks_walking_speed :
  ∃ (v : ℝ), v = 5 ∧ 
  (let time := lindy_distance / lindy_speed in
  let distance_christina_covers := christina_speed * time in
  let distance_jack_covers := initial_distance - distance_christina_covers in
  v = distance_jack_covers / time) :=
sorry

end jacks_walking_speed_l194_194958


namespace AC_tangent_to_circle_locus_of_pointP_l194_194536

variable (a b : ℝ) (h1 : a > b) (h2 : b > 0)

-- Define the circle and ellipse
def circle (x y : ℝ) : Prop := x^2 + y^2 = a^2
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the geometric points based on given conditions
def pointA (θ : ℝ) : ℝ × ℝ := (a * real.cos θ, a * real.sin θ)
def pointB (θ : ℝ) : ℝ × ℝ := (a * real.cos θ, b * real.sin θ)
def pointC (θ : ℝ) : ℝ × ℝ := (a / real.cos θ, 0)

-- Proof that AC is a tangent to the circle
theorem AC_tangent_to_circle (θ : ℝ) :
  ∀ x y : ℝ, circle a b x y →
  let A := pointA a θ in
  let C := pointC b θ in 
  ∀ (x y ∈ ℝ), (x - A.1) * (y - C.2) = (x - C.1) * (y - A.2) :=
sorry

-- Find the equation of the locus of point P
theorem locus_of_pointP (θ : ℝ) (x y : ℝ) :
  (a^2 + b)^2 =
  let P := (x, y) in
  ∀ P, x^2 + y^2 = (a + b)^2 :=
sorry

end AC_tangent_to_circle_locus_of_pointP_l194_194536


namespace problem1_problem2_l194_194741

-- Define condition p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

-- Define the negation of p
def neg_p (x a : ℝ) : Prop := ¬ p x a
-- Define the negation of q
def neg_q (x : ℝ) : Prop := ¬ q x

-- Question 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem problem1 (x : ℝ) (h1 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬ p is a sufficient but not necessary condition for ¬ q, then 1 < a ≤ 2
theorem problem2 (a : ℝ) (h2 : ∀ x : ℝ, neg_p x a → neg_q x) : 1 < a ∧ a ≤ 2 := 
by sorry

end problem1_problem2_l194_194741


namespace decreasing_function_range_a_l194_194858

open Real

noncomputable def y (a x : ℝ) : ℝ := (log a (3 - a * x))

-- Define the derivative of y with respect to x
noncomputable def y' (a x : ℝ) : ℝ := 
  -a / ((3 - a * x) * log a)

-- The final theorem ensures that for y = log_a (3 - ax) to be decreasing on [0, 1], a must be in (0, 3)
theorem decreasing_function_range_a (a : ℝ) :
  (∀ x ∈ Icc 0 1, y' a x < 0) ↔ (0 < a ∧ a < 3) :=
  sorry

end decreasing_function_range_a_l194_194858


namespace find_m_l194_194163

theorem find_m (m : ℝ) (h : ∀ x : ℝ, x > 0 → (m^2 - 2m - 2) * x^(m^2 - 2) > 0) : m = 3 := by
  sorry

end find_m_l194_194163


namespace cricket_player_increased_average_by_4_l194_194752

theorem cricket_player_increased_average_by_4
  (initial_innings : ℕ)
  (initial_avg_runs : ℕ)
  (next_inning_runs : ℕ)
  (desired_increase : ℕ)
  (initial_innings = 10)
  (initial_avg_runs = 32)
  (next_inning_runs = 76)
  (desired_increase = 4)
  (total_runs_initial : ℕ := initial_avg_runs * initial_innings)
  (total_runs_new : ℕ := total_runs_initial + next_inning_runs)
  (new_avg_runs : ℕ := (total_runs_new / (initial_innings + 1)))
  (expected_new_avg : ℕ := initial_avg_runs + desired_increase):
  new_avg_runs = expected_new_avg :=
by
  -- proof will be provided here
  sorry

end cricket_player_increased_average_by_4_l194_194752


namespace age_problem_l194_194451

theorem age_problem (S Sh K : ℕ) 
  (h1 : S / Sh = 4 / 3)
  (h2 : S / K = 4 / 2)
  (h3 : K + 10 = S)
  (h4 : S + 8 = 30) :
  S = 22 ∧ Sh = 17 ∧ K = 10 := 
sorry

end age_problem_l194_194451


namespace part1_part2_l194_194559

def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x - 3)
noncomputable def M := 3 / 2

theorem part1 (x : ℝ) (m : ℝ) : (∀ x, f x ≥ abs (m + 1)) → m ≤ M := sorry

theorem part2 (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b + c = M →  (b^2 / a + c^2 / b + a^2 / c) ≥ M := sorry

end part1_part2_l194_194559


namespace sum_p_until_2001_l194_194261

-- Definition of p(n), the product of the decimal digits of n
def p (n : ℕ) : ℕ := 
  (n.digits 10).prod

-- The main theorem to prove
theorem sum_p_until_2001 : (∑ n in Finset.range 2002, p n) = 184320 := by
  sorry

end sum_p_until_2001_l194_194261


namespace dot_product_l194_194849

-- Define the given vectors
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (3, -1)

-- Define c as a function of t where c is parallel to b when added to a
def c (t : ℝ) : ℝ × ℝ := (t, t)
def parallel (v₁ v₂: ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v₁ = (k • v₂).prod

-- Define the condition
def condition (t : ℝ) : Prop :=
  parallel (a.1 + c t.1, a.2 + c t.2) b

-- The goal to prove
theorem dot_product : ∃ t : ℝ, condition t ∧ (a.1 * c t.1 + a.2 * c t.2 = -5/4) :=
by
  sorry

end dot_product_l194_194849


namespace height_in_right_triangle_le_geometric_mean_l194_194942

theorem height_in_right_triangle_le_geometric_mean 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let m := (a * b) / Real.sqrt (a^2 + b^2) in
  m ≤ (Real.sqrt ((a^a * b^b)^(1 / (a + b)))) / Real.sqrt 2 := 
sorry

end height_in_right_triangle_le_geometric_mean_l194_194942


namespace min_value_PQ_l194_194538

-- Define the curve equation (x-1)^2 + y^2 = 1 where point P lies
def curveP (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the parametric equations for point Q
def curveQ (x y : ℝ) : Prop := ∃ t : ℝ, x = 1 + t ∧ y = 4 + 2 * t

-- Find the distance between a point and a line
noncomputable def dist_point_line (px py : ℝ) : ℝ :=
  let a := 2
  let b := -1
  let c := 2
  (abs (a * px + b * py + c) / real.sqrt (a^2 + b^2))

-- Define the minimum distance function
noncomputable def min_dist : ℝ :=
  dist_point_line 1 0 - 1

-- Prove that the minimum value of |PQ| is (4 * real.sqrt 5) / 5 - 1
theorem min_value_PQ :
  min_dist = (4 * real.sqrt 5) / 5 - 1 :=
sorry

end min_value_PQ_l194_194538


namespace expand_and_simplify_l194_194483

variable (y : ℝ)

theorem expand_and_simplify :
  -2 * (5 * y^3 - 4 * y^2 + 3 * y - 6) = -10 * y^3 + 8 * y^2 - 6 * y + 12 :=
  sorry

end expand_and_simplify_l194_194483


namespace tangent_through_fixed_point_segment_ratio_l194_194737

section parabola_tangent

variable {k : ℝ}

def parabola (x : ℝ) : ℝ := (1 / 2) * x ^ 2 

def line (x : ℝ) : ℝ := k * x - 1 

def fixed_point : ℝ × ℝ := (k, 1)

theorem tangent_through_fixed_point (x1 x2 : ℝ) :
  ∀ P : ℝ × ℝ, P ∈ {P : ℝ × ℝ | P.2 = line P.1} →
  ∃ A B : ℝ × ℝ, A ∉ {A : ℝ × ℝ | A.2 = line A.1} ∨ B ∉ {B : ℝ × ℝ | B.2 = line B.1} →
  let AB := (A.1 - B.1, A.2 - B.2) in
    ∃ Q : ℝ × ℝ, Q = fixed_point ∧ (Q.2 - A.2) / (Q.1 - A.1) = (Q.2 - B.2) / (Q.1 - B.1) :=
sorry

theorem segment_ratio (P : ℝ × ℝ) (M N : ℝ × ℝ) :
  P ∈ {P : ℝ × ℝ | P.2 = line P.1} → 
  M ∈ {M : ℝ × ℝ | M.2 = parabola M.1} → 
  N ∈ {N : ℝ × ℝ | N.2 = parabola N.1} → 
  let QM := (fixed_point.1 - M.1, fixed_point.2 - M.2),
      QN := (fixed_point.1 - N.1, fixed_point.2 - N.2) in
    |(QM.1 / QN.1)| = |(M.1 - P.1) / (N.1 - P.1)| :=
sorry

end parabola_tangent

end tangent_through_fixed_point_segment_ratio_l194_194737


namespace probability_ace_spades_then_king_spades_l194_194376

theorem probability_ace_spades_then_king_spades :
  ∃ (p : ℚ), (p = 1/52 * 1/51) := sorry

end probability_ace_spades_then_king_spades_l194_194376


namespace number_of_computer_literate_females_l194_194238

-- Definitions from the conditions
variables (total_employees : ℕ)
variables (female_percentage : ℝ)
variables (male_literate_percentage : ℝ)
variables (total_literate_percentage : ℝ)
variables (male_employees : ℕ)
variables (female_employees : ℕ)
variables (literate_male_employees : ℕ)
variables (total_literate_employees : ℕ)
variables (literate_female_employees : ℕ)

-- Given conditions
def conditions :=
  total_employees = 1100 ∧
  female_percentage = 0.60 ∧
  male_literate_percentage = 0.50 ∧
  total_literate_percentage = 0.62 ∧
  female_employees = (female_percentage * total_employees).to_nat ∧
  male_employees = total_employees - female_employees ∧
  literate_male_employees = (male_literate_percentage * male_employees).to_nat ∧
  total_literate_employees = (total_literate_percentage * total_employees).to_nat ∧
  literate_female_employees = total_literate_employees - literate_male_employees

-- Theorem to be proven
theorem number_of_computer_literate_females : conditions -> literate_female_employees = 462 := by
  intro h
  sorry

end number_of_computer_literate_females_l194_194238


namespace reach_empty_table_from_initial_config_reach_empty_table_from_any_config_reach_empty_table_8x8_l194_194611

noncomputable theory

def move1 (M : Matrix (Fin 2) (Fin 2) ℕ) (i : Fin 2) : Matrix (Fin 2) (Fin 2) ℕ :=
  λ r c, if r = i then M r c - 1 else M r c

def move2 (M : Matrix (Fin 2) (Fin 2) ℕ) (c : Fin 2) : Matrix (Fin 2) (Fin 2) ℕ :=
  λ r c', if c' = c then 2 * M r c' else M r c'

def is_empty (M : Matrix (Fin 2) (Fin 2) ℕ) : Prop :=
  ∀ r c, M r c = 0

theorem reach_empty_table_from_initial_config :
  ∃ seq : list (Matrix (Fin 2) (Fin 2) ℕ → Matrix (Fin 2) (Fin 2) ℕ),
    is_empty (seq.foldl (λ m f, f m) (λ _ _, 5 ! 5 ! 3)) :=
begin
  sorry
end

theorem reach_empty_table_from_any_config (M : Matrix (Fin 2) (Fin 2) ℕ) (h : ∀ r c, 0 < M r c) :
  ∃ seq : list (Matrix (Fin 2) (Fin 2) ℕ → Matrix (Fin 2) (Fin 2) ℕ),
    is_empty (seq.foldl (λ m f, f m) M) :=
begin
  sorry
end

theorem reach_empty_table_8x8 (M : Matrix (Fin 8) (Fin 8) ℕ) (h : ∀ r c, 0 < M r c) :
  ∃ seq : list (Matrix (Fin 8) (Fin 8) ℕ → Matrix (Fin 8) (Fin 8) ℕ),
    ∀ r c, (seq.foldl (λ m f, f m) M) r c = 0 :=
begin
  sorry
end

end reach_empty_table_from_initial_config_reach_empty_table_from_any_config_reach_empty_table_8x8_l194_194611


namespace substitute_and_simplify_l194_194243

theorem substitute_and_simplify : (∀ (x y : ℝ), 3 * x - y = 18 → y = x + 1 → 3 * x - x - 1 = 18) :=
by
  intros x y h₁ h₂
  rw [h₂] at h₁
  simp at h₁
  exact h₁

end substitute_and_simplify_l194_194243


namespace pascal_row_15_sum_l194_194927

theorem pascal_row_15_sum : (∑ i in Finset.range 16, Nat.CasesOn i 1 (λ n, Nat.choose 15 n)) = 32768 := 
by 
  sorry

end pascal_row_15_sum_l194_194927


namespace find_base_of_numeral_system_l194_194695

def base_of_numeral_system (x : ℕ) : Prop :=
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2

theorem find_base_of_numeral_system :
  ∃ x : ℕ, base_of_numeral_system x ∧ x = 7 := sorry

end find_base_of_numeral_system_l194_194695


namespace determine_quadruple_l194_194755

def function_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → ∃ y, f x = y

def function_range (f : ℝ → ℝ) (c d : ℝ) : Prop :=
  ∀ y, ∃ x, f x = y ∧ c ≤ y ∧ y ≤ d

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := 1 - f (x + 1)

theorem determine_quadruple (f : ℝ → ℝ)
  (hf_domain : function_domain f 0 2)
  (hf_range : function_range f 0 1) :
  (∃ a b c d, a = -1 ∧ b = 1 ∧ c = 0 ∧ d = 1 ∧
    function_domain (g f) a b ∧ function_range (g f) c d) :=
begin
  sorry
end

end determine_quadruple_l194_194755


namespace intersection_xz_plane_l194_194824

-- Define the two points through which the line passes
def point1 : ℝ × ℝ × ℝ := (3, 4, 1)
def point2 : ℝ × ℝ × ℝ := (5, 1, 6)

-- Direction vector from point1 to point2
def direction_vector : ℝ × ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2, point2.3 - point1.3)

-- Parametric equation of the line through point1 with direction vector
def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (point1.1 + t * direction_vector.1, point1.2 + t * direction_vector.2, point1.3 + t * direction_vector.3)

-- Proving the intersection of the line with the xz-plane
theorem intersection_xz_plane : ∃ t : ℝ, parametric_line t = (17/3, 0, 23/3) :=
by
  sorry

end intersection_xz_plane_l194_194824


namespace g_of_1986_l194_194993

-- Define the function g and its properties
noncomputable def g : ℕ → ℤ :=
sorry  -- Placeholder for the actual definition according to the conditions

axiom g_is_defined (x : ℕ) : x ≥ 0 → ∃ y : ℤ, g x = y
axiom g_at_1 : g 1 = 1
axiom g_add (a b : ℕ) (h_a : a ≥ 0) (h_b : b ≥ 0) : g (a + b) = g a + g b - 3 * g (a * b) + 1

-- Lean statement for the proof problem
theorem g_of_1986 : g 1986 = 0 :=
sorry

end g_of_1986_l194_194993


namespace cos_transform_to_cos_div_4_l194_194710

theorem cos_transform_to_cos_div_4 :
  ∀ (x : ℝ), ∃ (k : ℝ), (k = 4) ∧ (cos (x / 4) = cos (k * (x / k))) :=
by
  intros x
  use 4
  split
  . norm_num
  . sorry

end cos_transform_to_cos_div_4_l194_194710


namespace gcf_of_180_270_450_l194_194388

theorem gcf_of_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  have prime_factor_180 : ∃ (a b c : ℕ), 180 = 2^2 * 3^2 * 5 := ⟨2, 2, 1, rfl⟩
  have prime_factor_270 : ∃ (a b c : ℕ), 270 = 2 * 3^3 * 5 := ⟨1, 3, 1, rfl⟩
  have prime_factor_450 : ∃ (a b c : ℕ), 450 = 2 * 3^2 * 5^2 := ⟨1, 2, 2, rfl⟩
  sorry

end gcf_of_180_270_450_l194_194388


namespace speed_of_man_l194_194030

theorem speed_of_man (l : ℝ) (v_train_kmph : ℝ) (t : ℝ) (v_man_kmph : ℝ) :
  l = 350 ∧ v_train_kmph = 68 ∧ t = 20.99832013438925 →
  v_man_kmph = 7.9916 :=
by
  -- Define constants and conversions
  let v_train_mps := v_train_kmph * (5 / 18)
  let rel_speed_mps := l / t
  let v_man_mps := v_train_mps - rel_speed_mps
  let v_man_kmph_calc := v_man_mps * 3.6
   
  -- Use the conversions and definitions to establish v_man_kmph
  have h : v_man_kmph_calc = 7.9916 := sorry

  -- Conclude the proof with the calculation
  exact h

end speed_of_man_l194_194030


namespace count_4_digit_palindromic_squares_is_2_l194_194182

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def count_4_digit_palindromic_squares : ℕ :=
  (Finset.range 100).filter (λ n, 32 ≤ n ∧ is_4_digit_number (n * n) ∧ is_palindrome (n * n)).card

theorem count_4_digit_palindromic_squares_is_2 : count_4_digit_palindromic_squares = 2 :=
  sorry

end count_4_digit_palindromic_squares_is_2_l194_194182


namespace sum_of_roots_eq_l194_194721

theorem sum_of_roots_eq (x : ℝ) : (x - 4)^2 = 16 → (∃ r1 r2 : ℝ, (x - 4) = 4 ∨ (x - 4) = -4 ∧ r1 + r2 = 8) :=
by
  have h := (x - 4) ^ 2 = 16
  sorry  -- You would proceed with the proof here.

end sum_of_roots_eq_l194_194721


namespace Fred_last_week_l194_194974

-- Definitions from conditions
def Fred_now := 40
def Fred_earned := 21

-- The theorem we need to prove
theorem Fred_last_week :
  Fred_now - Fred_earned = 19 :=
by
  sorry

end Fred_last_week_l194_194974


namespace convert_500yah_to_bah_l194_194908

-- Definitions from conditions in the problem
def bahs_to_rahs (bahs: ℕ) : ℕ := (bahs * 30) / 20
def rahs_to_yahs (rahs: ℕ) : ℕ := (rahs * 25) / 10

-- Now we add the statement of the problem as a theorem 
theorem convert_500yah_to_bah : 
  let rahs := rahs_to_yahs 500
  let bahs := rahs_to_bahs rahs
  bahs = 133 :=
by {
  -- we will add the proof later
  sorry
}

end convert_500yah_to_bah_l194_194908


namespace total_pears_l194_194310

theorem total_pears (s t : ℕ) (hs : s = 6) (ht : t = 5) : s + t = 11 :=
by {
  rw [hs, ht],
  exact Nat.add_comm 6 5 ▸ by rfl,
  sorry
}

end total_pears_l194_194310


namespace irrational_solution_l194_194262

noncomputable def fractional_part (x: ℝ) : ℝ := x - floor x

theorem irrational_solution (x : ℝ) (h : fractional_part x + fractional_part (1 / x) = 1) : ¬ ∃ q : ℚ, x = q :=
by 
  sorry

end irrational_solution_l194_194262


namespace number_of_people_l194_194668

theorem number_of_people (n : ℕ) :
  (∑ k in Finset.range n, (3 + k)) = 100 * n ↔ n = 195 :=
by
  sorry

end number_of_people_l194_194668


namespace quadrilateral_S_is_parallelogram_l194_194890

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

-- Define points and Position Vectors
variables {O : V} {A B C : fin 4 → V}
-- Define Centroids
def S_i (i : fin 4) : V := (A i + B i + C i) / 3

-- Condition: Given that A, B, and C each form a parallelogram
variables (hA : ∀ i j : fin 2, A (2 * i) - A (2 * i + 1) = A (4 - 2 * i - 1) - A (4 - 2 * i - 2))
variables (hB : ∀ i j : fin 2, B (2 * i) - B (2 * i + 1) = B (4 - 2 * i - 1) - B (4 - 2 * i - 2))
variables (hC : ∀ i j : fin 2, C (2 * i) - C (2 * i + 1) = C (4 - 2 * i - 1) - C (4 - 2 * i - 2))

theorem quadrilateral_S_is_parallelogram : 
(∀ i j : fin 2, 
  S_i 0 - S_i 1 = S_i 3 - S_i 2) :=
begin
  sorry
end

end quadrilateral_S_is_parallelogram_l194_194890


namespace negation_of_universal_proposition_l194_194686

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 > 1) ↔ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 ≤ 1 := 
sorry

end negation_of_universal_proposition_l194_194686


namespace probability_rain_all_three_days_l194_194349

theorem probability_rain_all_three_days : 
  let P_fri := 2 / 5
  let P_sat := 1 / 2
  let P_sun := 1 / 5
  P_fri * P_sat * P_sun * 100 = 4 :=
by
  let P_fri := 2 / 5
  let P_sat := 1 / 2
  let P_sun := 1 / 5
  calc
    P_fri * P_sat * P_sun * 100 = ((2 / 5) * (1 / 2) * (1 / 5)) * 100 : by rfl
                          ... = (2 / (5 * 2 * 5)) * 100 : by rw [mul_assoc, mul_assoc, mul_comm (1 / 2)]
                          ... = (2 / 50) * 100 : by norm_num
                          ... = 1 / 25 * 100 : by rfl
                          ... = 4 : by norm_num
  sorry

end probability_rain_all_three_days_l194_194349


namespace smallest_n_property_l194_194071

noncomputable def smallest_n : ℕ := 13

theorem smallest_n_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (x * y * z ∣ (x + y + z) ^ smallest_n) :=
by
  intros x y z hx hy hz hxy hyz hzx
  use smallest_n
  sorry

end smallest_n_property_l194_194071


namespace salary_increase_l194_194292

theorem salary_increase (prev_income : ℝ) (prev_percentage : ℝ) (new_percentage : ℝ) (rent_utilities : ℝ) (new_income : ℝ) :
  prev_income = 1000 ∧ prev_percentage = 0.40 ∧ new_percentage = 0.25 ∧ rent_utilities = prev_percentage * prev_income ∧
  rent_utilities = new_percentage * new_income → new_income - prev_income = 600 :=
by 
  sorry

end salary_increase_l194_194292


namespace inequality_solution_l194_194313

theorem inequality_solution (x : ℝ) : 
  (x + 7) / (x^2 + 2*x + 8) ≥ 0 ↔ x ≥ -7 :=
begin
  sorry,
end

end inequality_solution_l194_194313


namespace arithmetic_seq_a6_l194_194948

variable (a : ℕ → ℝ)

-- Conditions
axiom a3 : a 3 = 16
axiom a9 : a 9 = 80

-- Theorem to prove
theorem arithmetic_seq_a6 : a 6 = 48 :=
by
  sorry

end arithmetic_seq_a6_l194_194948


namespace tile_arrangement_probability_l194_194101

theorem tile_arrangement_probability :
  let X := 5
  let O := 4
  let total_tiles := 9
  (1 : ℚ) / (Nat.choose total_tiles X) = 1 / 126 :=
by
  sorry

end tile_arrangement_probability_l194_194101


namespace relationship_between_l_and_m_l194_194841

noncomputable def circle_c (x y r : ℝ) : Prop := x^2 + y^2 = r^2

noncomputable def line_m (a b r : ℝ) : Prop := a * x + b * y = r^2

theorem relationship_between_l_and_m
    (a b r x y : ℝ)
    (h_ab : a ≠ 0 ∧ b ≠ 0)
    (h_point : circle_c x y r)
    (h_inside : a^2 + b^2 < r^2) :
    l_perpendicular_m ∧ ¬(line_m a b r ∩ circle_c x y r ≠ ∅) :=
sorry

end relationship_between_l_and_m_l194_194841


namespace circle_angle_theorem_l194_194318

theorem circle_angle_theorem (O A B : Point) (α β : ℝ) 
  (h1 : central_angle O A B = α) (h2 : circumference_angle A B = β) :
  β = α / 2 := sorry

end circle_angle_theorem_l194_194318


namespace angle_J_in_convex_pentagon_l194_194779

theorem angle_J_in_convex_pentagon :
  ∀ (FGHIJ : Type) [geometry FGHIJ] 
  (F G H I J : FGHIJ) 
  (convex_pentagon : convex FGHIJ [F, G, H, I, J])
  (equal_sides : eq_length [F, G, H, I, J])
  (angle_F : angle F = 100)
  (angle_G : angle G = 100), 
  angle J = 140 := 
by 
  sorry

end angle_J_in_convex_pentagon_l194_194779


namespace taxi_range_l194_194928

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 
    5
  else if x <= 10 then
    5 + (x - 3) * 2
  else
    5 + 7 * 2 + (x - 10) * 3

theorem taxi_range (x : ℝ) (h : fare x + 1 = 38) : 15 < x ∧ x ≤ 16 := 
  sorry

end taxi_range_l194_194928


namespace percentage_volume_removed_correct_l194_194431

-- Definitions based on the given conditions
def box_length : ℝ := 24
def box_width : ℝ := 16
def box_height : ℝ := 12
def cube_side : ℝ := 2
def number_of_cubes : ℝ := 8

-- Calculation of volumes
def volume_box : ℝ := box_length * box_width * box_height
def volume_cube : ℝ := cube_side ^ 3
def total_volume_cubes : ℝ := number_of_cubes * volume_cube

-- Calculation of the percentage volume removed
def percentage_volume_removed : ℝ := (total_volume_cubes / volume_box) * 100

-- Theorem statement
theorem percentage_volume_removed_correct :
  percentage_volume_removed = 1.3888888888888888 := by
  sorry

end percentage_volume_removed_correct_l194_194431


namespace fraction_change_l194_194716

theorem fraction_change (a b k : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_k : 0 < k) :
  (a < b → (a + k) / (b + k) > a / b) ∧ (a > b → (a + k) / (b + k) < a / b) ∧ 
  (∀ ε > 0, ∃ N, ∀ m > N, abs ((a + m) / (b + m) - 1) < ε) := 
by
  sorry

end fraction_change_l194_194716


namespace speed_at_t2_l194_194023

noncomputable def position (t : ℝ) : ℝ × ℝ :=
  (t^2 + 2 * t + 7, 3 * t^2 + 4 * t - 13)

noncomputable def speed (t : ℝ) : ℝ :=
  let (x₁, y₁) := position t
  let (x₂, y₂) := position (t + 1)
  real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem speed_at_t2 : speed 2 = real.sqrt 410 :=
  sorry

end speed_at_t2_l194_194023


namespace smallest_N_for_same_length_diagonals_in_2017gon_is_1008_l194_194937

theorem smallest_N_for_same_length_diagonals_in_2017gon_is_1008 :
  ∀ (N : ℕ), (N > 1007) → ∃ (d1 d2 : (fin 2017) × (fin 2017)), 
               d1 ≠ d2 ∧ length_of_diagonal d1 = length_of_diagonal d2 :=
sorry

/-- Helper function to calculate the length of a diagonal in a regular polygon -/
noncomputable def length_of_diagonal (d : (fin 2017) × (fin 2017)) : ℝ :=
  -- Since actual diagonal length calculation involves some trigonometry on fin 2017, we leave it undefined here.
  sorry

end smallest_N_for_same_length_diagonals_in_2017gon_is_1008_l194_194937


namespace quadrilateral_is_cyclic_and_tangential_l194_194326

-- Definitions for cyclic and tangential quadrilateral
def cyclic_quadrilateral (K L M N : Type) := ∀ α β γ δ : ℝ, α + γ = 180 ∧ β + δ = 180

def tangential_quadrilateral (K L M N : Type) := ∃ P : Type, 
  is_angle_bisector P L K ∧ is_angle_bisector P K M ∧ is_angle_bisector P M N ∧ is_angle_bisector P N L

-- Main statement to prove KLMN is both cyclic and tangential quadrilateral
theorem quadrilateral_is_cyclic_and_tangential
  (A B C D P K L M N : Type)
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_diags_perp : is_perpendicular (diag1 A C) (diag2 B D))
  (h_perp_drops : are_perpendicular_drops P A B C D K L M N) :
  cyclic_quadrilateral K L M N ∧ tangential_quadrilateral K L M N :=
by
  sorry

end quadrilateral_is_cyclic_and_tangential_l194_194326


namespace apples_per_basket_l194_194700

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) (h : total_apples = 629) (k : num_baskets = 37) :
  total_apples / num_baskets = 17 :=
by
  -- proof omitted
  sorry

end apples_per_basket_l194_194700


namespace number_of_valid_sets_l194_194578

-- Definitions for the sets involved in the problem
def a : Type := unit
def b : Type := unit
def c : Type := unit

-- The universal set
def U : set (unit ⊕ unit ⊕ unit) := {sum.inl (), sum.inr (sum.inl ()), sum.inr (sum.inr ())}

-- Define a predicate to express the condition {a} ⊆ A ⊆ {a, b, c}
def valid_set (A : set (unit ⊕ unit ⊕ unit)) : Prop :=
  (sum.inl () ∈ A) ∧ A ⊆ U

-- The statement of the theorem that needs to be proved
theorem number_of_valid_sets : (finset.filter valid_set (finset.powerset U)).card = 4 := 
sorry

end number_of_valid_sets_l194_194578


namespace Mark_James_dice_problem_l194_194647

theorem Mark_James_dice_problem : 
  let mark_total_dice := 10
  let mark_percent_12_sided := 0.6
  let james_total_dice := 8
  let james_percent_12_sided := 0.75
  let total_needed := 14
  let mark_12_sided := mark_total_dice * mark_percent_12_sided
  let james_12_sided := james_total_dice * james_percent_12_sided
  let total_12_sided := mark_12_sided + james_12_sided
  let dice_to_buy := total_needed - total_12_sided
  ⟶ dice_to_buy = 2 := by sorry

end Mark_James_dice_problem_l194_194647


namespace probability_top_red_second_black_l194_194413

def num_red_cards : ℕ := 39
def num_black_cards : ℕ := 39
def total_cards : ℕ := 78

theorem probability_top_red_second_black :
  (num_red_cards * num_black_cards) / (total_cards * (total_cards - 1)) = 507 / 2002 := 
sorry

end probability_top_red_second_black_l194_194413


namespace quadratic_single_root_a_l194_194357

theorem quadratic_single_root_a (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by
  sorry

end quadratic_single_root_a_l194_194357


namespace distance_covered_at_40_kmph_l194_194006

theorem distance_covered_at_40_kmph (x : ℝ) (h : 0 ≤ x ∧ x ≤ 250) 
  (total_distance : x + (250 - x) = 250) 
  (total_time : x / 40 + (250 - x) / 60 = 5.5) : 
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l194_194006


namespace AP_perpendicular_BQ_l194_194980

variables (A B C D P Q : Point)
variables (l : Line)
variables [trapezoid : AB_parallel_CD]
variables (h1 : Parallel AB CD) 
variables (h2 : Length AB = 2 * Length CD)
variables (h3 : Perpendicular_to_CD l C)
variables (circle : Circle)

noncomputable def hyp : Circle (center D) := circle_center_radius D (length DA)

theorem AP_perpendicular_BQ  
    (h4 : Circle.intersect_line hyp l = {P, Q}) :
   Perpendicular AP BQ :=
sorry

end AP_perpendicular_BQ_l194_194980


namespace part_a_part_b_l194_194332

def square_side_length : ℝ := 10
def square_area (side_length : ℝ) : ℝ := side_length * side_length
def triangle_area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height

-- Part (a)
theorem part_a :
  let side_length := square_side_length
  let square := square_area side_length
  let triangle := triangle_area side_length side_length
  square - triangle = 50 := by
  sorry

-- Part (b)
theorem part_b :
  let side_length := square_side_length
  let square := square_area side_length
  let small_triangle_area := square / 8
  2 * small_triangle_area = 25 := by
  sorry

end part_a_part_b_l194_194332


namespace frank_total_cost_l194_194112

-- Conditions from the problem
def cost_per_bun : ℝ := 0.1
def number_of_buns : ℕ := 10
def cost_per_bottle_of_milk : ℝ := 2
def number_of_bottles_of_milk : ℕ := 2
def cost_of_carton_of_eggs : ℝ := 3 * cost_per_bottle_of_milk

-- Question and Answer
theorem frank_total_cost : 
  let cost_of_buns := cost_per_bun * number_of_buns in
  let cost_of_milk := cost_per_bottle_of_milk * number_of_bottles_of_milk in
  let cost_of_eggs := cost_of_carton_of_eggs in
  cost_of_buns + cost_of_milk + cost_of_eggs = 11 :=
by
  sorry

end frank_total_cost_l194_194112


namespace equation_of_line_l194_194530

noncomputable def point := (ℝ × ℝ)

def A : point := (1, -2)
def B : point := (5, 6)

def midpoint (P Q : point) : point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def M : point := midpoint A B

theorem equation_of_line :
  ∃ (a b c : ℝ), (a = 2 ∧ b = -3 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -5) ∧ 
    ∀ (x y : ℝ), (a * x + b * y + c = 0) ↔ (M.1 * a + M.2 * b + c = 0) := 
sorry

end equation_of_line_l194_194530


namespace find_base_k_l194_194807

def repeating_base_k_representation (k : ℕ) : Prop :=
  (3 + 4 / k) / (k^2 - 1) = 8 / 65

theorem find_base_k : ∃ k : ℕ, k > 0 ∧ repeating_base_k_representation k :=
  ∃ k, k > 0 ∧ (3 * k + 4)/(k^2 - 1) = 8 / 65

end find_base_k_l194_194807


namespace find_x_l194_194891

variable x : ℝ
def a := (x, -1)
def b := (4, 2)
def parallel (u v : ℝ × ℝ) := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_x (h : parallel a b) : x = -2 :=
by
  sorry

end find_x_l194_194891


namespace tan_pi_div_a_of_point_on_cubed_function_l194_194866

theorem tan_pi_div_a_of_point_on_cubed_function (a : ℝ) (h : (a, 27) ∈ {p : ℝ × ℝ | p.snd = p.fst ^ 3}) : 
  Real.tan (Real.pi / a) = Real.sqrt 3 := sorry

end tan_pi_div_a_of_point_on_cubed_function_l194_194866


namespace equal_profits_at_20000_end_month_more_profit_50000_l194_194291

noncomputable section

-- Define the conditions
def profit_beginning_month (x : ℝ) : ℝ := 0.15 * x + 1.15 * x * 0.1
def profit_end_month (x : ℝ) : ℝ := 0.3 * x - 700

-- Proof Problem 1: Prove that at x = 20000, the profits are equal
theorem equal_profits_at_20000 : profit_beginning_month 20000 = profit_end_month 20000 :=
by
  sorry

-- Proof Problem 2: Prove that at x = 50000, selling at end of month yields more profit than selling at beginning of month
theorem end_month_more_profit_50000 : profit_end_month 50000 > profit_beginning_month 50000 :=
by
  sorry

end equal_profits_at_20000_end_month_more_profit_50000_l194_194291


namespace monotonic_intervals_f_x_bound_l194_194558

-- Given definitions
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a x : ℝ) : ℝ := (D f x) - a * x ^ 2

-- Statement to prove 1
theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, 0 < x → g a x > 0) ∧
  (a > 0 → ∀ x : ℝ, 0 < x → x < 1 / sqrt (2 * a) → g a x > 0) ∧
  (a > 0 → ∀ x : ℝ, x > 1 / sqrt (2 * a) → g a x < 0) := sorry

-- Statement to prove 2
noncomputable def k (x : ℝ) : ℝ := Real.log x / x
noncomputable def h (x : ℝ) : ℝ := (2 * Real.exp (x - 2)) / (x ^ 2)

theorem f_x_bound : ∀ x : ℝ, 0 < x → f x < 2 * Real.exp (x - 2) := sorry

end monotonic_intervals_f_x_bound_l194_194558


namespace find_n_in_arithmetic_sequence_l194_194844

noncomputable def arithmetic_sequence (n : ℕ) (a_n S_n d : ℕ) :=
  ∀ (a₁ : ℕ), 
    a₁ + d * (n - 1) = a_n →
    n * a₁ + d * n * (n - 1) / 2 = S_n

theorem find_n_in_arithmetic_sequence 
   (a_n S_n d n : ℕ) 
   (h_a_n : a_n = 44) 
   (h_S_n : S_n = 158) 
   (h_d : d = 3) :
   arithmetic_sequence n a_n S_n d → 
   n = 4 := 
by 
  sorry

end find_n_in_arithmetic_sequence_l194_194844


namespace smallest_integer_in_list_l194_194763

theorem smallest_integer_in_list 
  (L : List ℕ) 
  (h_len : L.length = 5)
  (h_pos : ∀ x ∈ L, 0 < x)
  (h_occurs : List.count L 6 = 2) 
  (h_median : List.nth_le (L.qsort (≤)) 2 (by linarith) = 12) 
  (h_mean : List.sum L / 5 = 14) :
  ∃ n ∈ L, n = 6 :=
by
  sorry

end smallest_integer_in_list_l194_194763


namespace smallest_positive_period_and_range_area_of_triangle_l194_194636

noncomputable def f (x : ℝ) : ℝ :=
  sin (x + π / 2) * (sqrt 3 * sin x + cos x)

theorem smallest_positive_period_and_range :
  (∃ T > 0, ∀ x, f(x + T) = f(x)) ∧ 
  (∀ y, ∃ x, y = f(x) → y ∈ (set.Icc (-1/2) (3/2))) :=
sorry

theorem area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : f A = 1)
  (h2 : a = sqrt 3)
  (h3 : b + c = 3)
  (h4 : 0 < A) (h5 : A < π)
  : (1/2) * b * c * sin A = sqrt 3 / 2 :=
sorry

end smallest_positive_period_and_range_area_of_triangle_l194_194636


namespace expression_simplifies_to_zero_l194_194935

theorem expression_simplifies_to_zero (x y : ℝ) (h : x = 2024) :
    5 * (x ^ 3 - 3 * x ^ 2 * y - 2 * x * y ^ 2) -
    3 * (x ^ 3 - 5 * x ^ 2 * y + 2 * y ^ 3) +
    2 * (-x ^ 3 + 5 * x * y ^ 2 + 3 * y ^ 3) = 0 :=
by {
    sorry
}

end expression_simplifies_to_zero_l194_194935


namespace cos_CBD_area_ABD_range_l194_194605

variables {A B C D : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 2))] 
(variable BD : ℝ) (variable CD : ℝ)
(variable ∠ABC : real_angle) (variable ∠C : real_angle)

axiom angle_ABC_90 : ∠ABC = 90 * real.pi / 180
axiom angle_C_135 : ∠C = 135 * real.pi / 180
axiom BD_val : BD = real.sqrt 5
axiom CD_val : CD = real.sqrt 2
axiom ∠CBD : real_angle

theorem cos_CBD :
  cos ∠CBD = 2 * real.sqrt 5 / 5 :=
sorry

theorem area_ABD_range (h : triangle.is_acute A B D) :
  1 < triangle.area A B D ∧ triangle.area A B D < 5 :=
sorry

end cos_CBD_area_ABD_range_l194_194605


namespace michael_matchstick_houses_l194_194289

theorem michael_matchstick_houses :
  ∃ n : ℕ, n = (600 / 2) / 10 ∧ n = 30 := 
sorry

end michael_matchstick_houses_l194_194289


namespace days_to_finish_job_l194_194398

def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 4 / 15
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

theorem days_to_finish_job (A B C : ℚ) (h1 : A + B = work_rate_a_b) (h2 : C = work_rate_c) :
  1 / (A + B + C) = 3 :=
by
  sorry

end days_to_finish_job_l194_194398


namespace mary_initial_money_l194_194288

theorem mary_initial_money (M : ℤ) : 
  (let marco_has := 24 in
  let marco_gives := marco_has / 2 in
  let mary_new := M + marco_gives in
  let marco_left := marco_has - marco_gives in
  let mary_spends := 5 in
  let mary_after_spending := mary_new - mary_spends in
  mary_after_spending = marco_left + 10) →
  M = 27 := 
by
  intros h
  sorry

end mary_initial_money_l194_194288


namespace garden_perimeter_l194_194336

noncomputable def perimeter_of_garden (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) : ℝ :=
  2 * l + 2 * w

theorem garden_perimeter (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) :
  perimeter_of_garden w l h1 h2 = 304.64 :=
sorry

end garden_perimeter_l194_194336


namespace expression_change_l194_194394

theorem expression_change (a b c : ℝ) : 
  a - (2 * b - 3 * c) = a + (-2 * b + 3 * c) := 
by sorry

end expression_change_l194_194394


namespace club_members_neither_subject_l194_194933

theorem club_members_neither_subject (total members_cs members_bio members_both : ℕ)
  (h_total : total = 150)
  (h_cs : members_cs = 80)
  (h_bio : members_bio = 50)
  (h_both : members_both = 15) :
  total - ((members_cs - members_both) + (members_bio - members_both) + members_both) = 35 := by
  sorry

end club_members_neither_subject_l194_194933


namespace find_d_l194_194859

variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {d : ℝ}

-- Conditions
axiom arithmetic_seq (n : ℕ) : ∃ a_1 n_1 n_2, a_n n = a_1 + (n - 1) * d
axiom sum_arith_seq (n : ℕ) : S_n n = n * (a_n 1 + (n - 1) / 2 * d)
axiom given_condition : (S_n 2018 / 2018) - (S_n 18 / 18) = 100

theorem find_d : d = 1 / 10 :=
by
  sorry

end find_d_l194_194859


namespace exp_to_rectangular_form_l194_194801

theorem exp_to_rectangular_form :
  (exp (15 * real.pi * complex.I / 2)) = -complex.I :=
by
  sorry

end exp_to_rectangular_form_l194_194801


namespace triangle_angles_equal_l194_194610

theorem triangle_angles_equal
  {A B C P1 P2 Q1 Q2 R S M : Point}
  (h_triangle_ABC : Triangle A B C)
  (h_P1_on_AB : OnLineSegment P1 A B)
  (h_P2_on_BP1 : OnLineSegment P2 B P1)
  (h_AP1_eq_BP2 : AP1 = BP2)
  (h_Q1_on_BC : OnLineSegment Q1 B C)
  (h_Q2_on_BQ1 : OnLineSegment Q2 B Q1)
  (h_BQ1_eq_CQ2 : BQ1 = CQ2)
  (h_R_intersection : Intersection P1 Q2 P2 Q1 R)
  (h_S_circumcircles : SecondIntersectionCircumcircles P1 P2 R Q1 Q2 R S)
  (h_S_inside_P1Q1R : InsideTriangle S P1 Q1 R)
  (h_M_mid_AC : Midpoint M A C) :
  ∠ P1 R S = ∠ Q1 R M := 
sorry

end triangle_angles_equal_l194_194610


namespace ratio_of_work_papers_l194_194615

-- Definitions based on the conditions
def T : ℝ := 8
def B : ℝ := T / 2
def laptop_weight : ℝ := T + 2
def B_full : ℝ := 2 * T
def W : ℝ := B_full - laptop_weight

-- Question statement
theorem ratio_of_work_papers :
  (W / B_full) = 3 / 8 :=
by sorry  -- proof not required

end ratio_of_work_papers_l194_194615


namespace minimum_questions_to_determine_color_l194_194303

theorem minimum_questions_to_determine_color : ∃ q : ℕ, q = 2 :=
by
  -- Define the grid
  let m := 8
  let n := 8
  -- Define the checkerboard pattern
  let checkerboard : ℕ × ℕ → Prop := λ i j, (i + j) % 2 = 0
  -- Define the point inside one of the squares
  let chosen_point : (ℕ × ℕ) := (i, j)
  -- Vasya can ask two questions to determine row and column parity
  let questions : ℕ := 2
  -- Demonstrate that 2 questions are enough
  exists questions
  exact (by trivial : questions = 2)

end minimum_questions_to_determine_color_l194_194303


namespace find_x_satisfying_g502_l194_194272

open Real

noncomputable def g₁ (x : ℝ) : ℝ := (1 / 2) - (4 / (4 * x + 2))

noncomputable def gₙ : ℕ → ℝ → ℝ
| 1, x := g₁ x
| (n + 1), x := g₁ (gₙ n x)

theorem find_x_satisfying_g502:
  ∀ x : ℝ, gₙ 502 x = x - 2 ↔ (x = 115 / 64 ∨ x = 51 / 64) :=
sorry

end find_x_satisfying_g502_l194_194272


namespace percentage_saved_on_hats_l194_194040

/-- Suppose the regular price of a hat is $60 and Maria buys four hats with progressive discounts: 
20% off the second hat, 40% off the third hat, and 50% off the fourth hat.
Prove that the percentage saved on the regular price for four hats is 27.5%. -/
theorem percentage_saved_on_hats :
  let regular_price := 60
  let discount_2 := 0.2 * regular_price
  let discount_3 := 0.4 * regular_price
  let discount_4 := 0.5 * regular_price
  let price_1 := regular_price
  let price_2 := regular_price - discount_2
  let price_3 := regular_price - discount_3
  let price_4 := regular_price - discount_4
  let total_regular := 4 * regular_price
  let total_discounted := price_1 + price_2 + price_3 + price_4
  let savings := total_regular - total_discounted
  let percentage_saved := (savings / total_regular) * 100
  percentage_saved = 27.5 :=
by
  sorry

end percentage_saved_on_hats_l194_194040


namespace value_of_a5_a7_a9_l194_194516

noncomputable def is_root (p : ℝ → ℝ) (a : ℝ) := p(a) = 0

def geometric_sequence (a₁ a₁₃ : ℝ) (n : ℕ) : ℝ :=
  a₁ * (rat.sqrt (a₁₃ / a₁) ^ (n - 1))

theorem value_of_a5_a7_a9
  (a₁ a₁₃ : ℝ) 
  (h1 : is_root (λ x : ℝ, x^2 - 8 * x + 1) a₁)
  (h2 : is_root (λ x : ℝ, x^2 - 8 * x + 1) a₁₃) 
  (h3 : 0 < a₁)
  (h4 : 0 < geometric_sequence a₁ a₁₃ 13)
  (h5 : ∀ n : ℕ, odd n → 0 < geometric_sequence a₁ a₁₃ n) :
  geometric_sequence a₁ a₁₃ 5 * geometric_sequence a₁ a₁₃ 7 * geometric_sequence a₁ a₁₃ 9 = 1 :=
sorry

end value_of_a5_a7_a9_l194_194516


namespace original_proposition_number_of_correct_propositions_l194_194563

variables {R : Type*} [LinearOrderedField R]

-- Definitions for odd and even functions
def is_odd_fn (f : R → R) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_even_fn (f : R → R) : Prop :=
  ∀ x, f (-x) = f (x)

-- Given conditions and proof goal
theorem original_proposition 
  (f g : R → R) 
  (hf : is_odd_fn f) 
  (hg : is_odd_fn g) 
  : is_even_fn (λ x, f x * g x) :=
sorry

-- The number of correct propositions is 2
theorem number_of_correct_propositions : 
  ∀ (f g : R → R) 
  (hf : is_odd_fn f) 
  (hg : is_odd_fn g), 
  let h := λ x, f x * g x in 
  is_even_fn h ∧ 
  ¬(∀ x, h (-x) = f (-x) * g (-x) → is_odd_fn f ∧ is_odd_fn g) ∧ 
  ¬(is_odd_fn f ∧ is_odd_fn g ∨ is_even_fn f ∧ is_even_fn g) ∧ 
  (∀ x, h (-x) = f (-x) * g (-x) → is_even_fn h) :=
sorry

end original_proposition_number_of_correct_propositions_l194_194563


namespace road_renovation_l194_194709

theorem road_renovation (x : ℕ) (h : 200 / (x + 20) = 150 / x) : 
  x = 60 ∧ (x + 20) = 80 :=
by {
  sorry
}

end road_renovation_l194_194709


namespace greatest_possible_median_of_nine_nonnegative_numbers_l194_194295

theorem greatest_possible_median_of_nine_nonnegative_numbers (x : fin 9 → ℝ) (h1 : ∀ i, 0 ≤ x i) 
  (h2 : (∑ i, x i) / 9 = 10) : (∀ y : fin 9 → ℝ, (∀ i, 0 ≤ y i) → (∑ i, y i) / 9 = 10 → 
  let sorted_y := sort (<=) y in (sorted_y 4 ≤ 18)) :=
begin
  sorry
end

end greatest_possible_median_of_nine_nonnegative_numbers_l194_194295


namespace triangle_area_is_correct_l194_194589

noncomputable def area_of_triangle_ABC (A C : ℝ) (b : ℝ := 2) (B : ℝ := real.pi / 3) :=
  if A = real.pi / 2 then
    (1 / 2) * b * (b / real.tan B)
  else if A = B ∧ C = B then
    (1 / 2) * b * b * real.sin B
  else
    0

theorem triangle_area_is_correct (A C : ℝ) (h1 : b = 2 := by rfl) (h2 : B = real.pi / 3 := by rfl) 
  (h3 : real.sin (2 * A) + real.sin (A - C) = real.sin B) :
  area_of_triangle_ABC A C = sqrt 3 ∨ area_of_triangle_ABC A C = (2 * sqrt 3) / 3 :=
begin
  sorry
end

end triangle_area_is_correct_l194_194589


namespace part_1_part_2_l194_194284

-- Given definitions and conditions
def f (x : ℝ) (b : ℝ) (a : ℝ) : ℝ := (b * x / (Real.log x)) - (a * x)
def tangent_line (x : ℝ) (b : ℝ) (a : ℝ) : ℝ := 3 * x + f x b a - 4 * Real.sqrt Real.exp 1

-- Proof problems
theorem part_1 (a b : ℝ) :
  tangent_line (Real.sqrt Real.exp 1) b a = 0 →
  f (Real.sqrt Real.exp 1) b a = Real.sqrt Real.exp 1 →
  f' (Real.sqrt Real.exp 1) b a = -3 →
  a = 1 ∧ b = 1 :=
by
  sorry

theorem part_2 (a : ℝ) :
  ∀ x1 x2 ∈ set.Icc (real.exp 1) (real.exp 1 ^ 2),
  b = 1 →
  f x1 1 a ≤ f' x2 1 a + a →
  a = 1/2 - 1/(4 * Real.exp 1 ^ 2) :=
by
  sorry

end part_1_part_2_l194_194284


namespace john_total_distance_l194_194613

variables (v_alone_flat : ℝ) (r_alone : ℝ) (v_dog_flat : ℝ) (r_dog : ℝ)
          (t_dog t_alone : ℝ) (e_dog e_alone : ℝ)

def distance_with_dog : ℝ :=
  let speed_with_dog := v_dog_flat - (e_dog / 500 * r_dog) in
  speed_with_dog * t_dog

def distance_alone : ℝ :=
  let speed_alone := v_alone_flat - (e_alone / 500 * r_alone) in
  speed_alone * t_alone

theorem john_total_distance : 
  v_alone_flat = 4 → r_alone = 0.5 → v_dog_flat = 6 → r_dog = 0.75 →
  t_dog = 0.5 → e_dog = 1000 → t_alone = 0.5 → e_alone = 500 →
  (distance_with_dog v_alone_flat r_alone v_dog_flat r_dog t_dog e_dog + 
   distance_alone v_alone_flat r_alone v_dog_flat r_dog t_alone e_alone) = 4 :=
by {
  intros,
  simp [distance_with_dog, distance_alone],
  sorry
}

end john_total_distance_l194_194613


namespace four_digit_square_palindromes_are_zero_l194_194176

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

-- Define the main theorem statement
theorem four_digit_square_palindromes_are_zero : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) → 
             is_palindrome n → 
             (∃ m : ℕ, n = m * m) → 
             n = 0 :=
by
  sorry

end four_digit_square_palindromes_are_zero_l194_194176


namespace part1_part2_l194_194161

open Real

noncomputable def Q (x : ℝ) : ℝ :=
  if x > 0 then ln x / x else 0

theorem part1 (a : ℝ) : 
  (∀ x > 0, ln x ≠ a * x) ↔ (a ∈ (1 / exp 1, +∞)) :=
by
  sorry

noncomputable def H (x : ℝ) : ℝ :=
  exp x - x * ln x

theorem part2 :
  ∃ m : ℝ, ∀ x ∈ Ioi (1 / 2 : ℝ), (ln x + m / x < H x) ∧ (⌊m⌋ = 1) :=
by
  sorry

end part1_part2_l194_194161


namespace Sandy_goal_water_l194_194661

-- Definitions based on the conditions in problem a)
def milliliters_per_interval := 500
def time_per_interval := 2
def total_time := 12
def milliliters_to_liters := 1000

-- The goal statement that proves the question == answer given conditions.
theorem Sandy_goal_water : (milliliters_per_interval * (total_time / time_per_interval)) / milliliters_to_liters = 3 := by
  sorry

end Sandy_goal_water_l194_194661


namespace even_function_properties_l194_194805

noncomputable def f : ℝ → ℝ := sorry

theorem even_function_properties
  (H1 : ∀ x : ℝ, f(-x) = f(x))
  (H2 : ∀ x : ℝ, f(x-3) = -f(x))
  (H3 : ∀ x1 x2 : ℝ, x1 ∈ set.Icc 0 3 ∧ x2 ∈ set.Icc 0 3 ∧ x1 ≠ x2 → (f(x1)-f(x2))/(x1-x2) > 0) :
  f(49) < f(64) ∧ f(64) < f(81) :=
sorry

end even_function_properties_l194_194805


namespace simplify_and_evaluate_expression_l194_194312

noncomputable def tan_45 : Real := Real.tan (π / 4)

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 - 3 * tan_45) :
  (1 / (3 - x) - ((x ^ 2 + 6 * x + 9) / (x ^ 2 + 3 * x)) / ((x ^ 2 - 9) / x)) = 2 / 5 := by
  sorry

end simplify_and_evaluate_expression_l194_194312


namespace determine_a1_a2_a3_l194_194912

theorem determine_a1_a2_a3 (a a1 a2 a3 : ℝ)
  (h : ∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3) :
  a1 + a2 + a3 = 19 :=
by
  sorry

end determine_a1_a2_a3_l194_194912


namespace no_real_x_satisfying_quadratic_inequality_l194_194919

theorem no_real_x_satisfying_quadratic_inequality (a : ℝ) :
  ¬(∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end no_real_x_satisfying_quadratic_inequality_l194_194919


namespace maximum_a_l194_194104

noncomputable def validate_inequality : Prop :=
  ∀ x : ℝ, x ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) →
    (sqrt[3] (Real.tan x) - sqrt[3] (Real.cot x)) / (sqrt[3] (Real.sin x) + sqrt[3] (Real.cos x)) > (4 * Real.root (Real.sixthRoot 2)) / 2

theorem maximum_a : validate_inequality :=
begin
  sorry,
end

end maximum_a_l194_194104


namespace geometric_series_sum_l194_194074

theorem geometric_series_sum : 
  let a := (1 : ℚ) / 4
  let r := (-1 : ℚ) / 4
  let sum := a * (1 - r^6) / (1 - r)
  sum = 273 / 1365 :=
by
  let a : ℚ := 1 / 4
  let r : ℚ := -1 / 4
  have h : sum = a * (1 - r^6) / (1 - r)
  show sum = 273 / 1365
  sorry

end geometric_series_sum_l194_194074


namespace estimate_students_l194_194600

noncomputable def mean : ℝ := 90
noncomputable def std_dev : ℝ := σ -- σ > 0
noncomputable def prob_range : ℝ := 0.8
noncomputable def total_students : ℕ := 780
noncomputable def prob_gt_120 : ℝ := (1 - prob_range) / 2
noncomputable def estimated_students_gt_120 : ℕ := (prob_gt_120 * total_students).to_nat

theorem estimate_students {σ : ℝ} (hσ : σ > 0) :
  estimated_students_gt_120 = 78 :=
by 
  sorry

end estimate_students_l194_194600


namespace complex_modulus_conjugate_l194_194220

-- Define the problem
theorem complex_modulus_conjugate (z : ℂ) (h : z = (1 + complex.i) / (1 - complex.i)) : complex.abs (conj z) = 1 :=
by
  sorry -- Proof of the theorem

end complex_modulus_conjugate_l194_194220


namespace projection_of_skew_lines_l194_194377

-- Assuming basic definitions related to projections, skew lines, and planes
noncomputable def skew_lines (l1 l2 : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ∉ l1 ∧ p2 ∉ l2 ∧ ¬parallel l1 l2 ∧ ¬intersect l1 l2

theorem projection_of_skew_lines (l1 l2 : Line) (proj_A proj_B proj_D : set (set Point)) :
  (skew_lines l1 l2) →
  orthogonal_projection l1 = proj_A →
  orthogonal_projection l2 = proj_B →
  orthogonal_projection l1 ≠ orthogonal_projection l2 →
  proj_A ≠ proj_D ∧ proj_B ≠ proj_D → 
  orthogonal_projection l1 ≠ orthogonal_projection l1 :=
sorry

end projection_of_skew_lines_l194_194377


namespace distance_from_point_to_plane_l194_194626

-- Define the given conditions
variables (a : ℝ) (α : ℝ)
variable (ABC : Type) -- the type representing the plane ABC
variables (A B C M : Type) -- points A, B, C, and M
variable [metric_space ABC] -- assume ABC has a metric space structure
variable (h_triangle : is_right_triangle A B C) -- ABC is a right triangle
variable (h_hyp : hypotenuse_length A B = a) -- AB = a
variable [has_inner_product ABC] -- assume the existence of an inner product space structure
variable (h_angles : ∀ (X ∈ {A, B, C}), angle_between M X ABC = α) -- MA, MB, and MC make angle α with plane ABC

-- Define the expected result
def distance_from_plane : ℝ := (1 / 2) * a * real.tan α

-- Lean statement to prove the distance
theorem distance_from_point_to_plane :
  ∃ (O : ABC), is_orthogonal_projection O M ABC ∧ 
              (dist O M = distance_from_plane a α) := 
sorry -- Proof is omitted

end distance_from_point_to_plane_l194_194626


namespace q_is_composite_l194_194744

open Nat

-- Define two consecutive odd prime numbers p1 and p2
variable (p1 p2 q : ℕ)
variable [prime_p1 : Prime p1] [prime_p2 : Prime p2]
variable [p1_consecutive_p2 : ∀ n, (n = p1 + 2) → n = p2]
variable (h : p1 + p2 = 2 * q)

theorem q_is_composite : ¬ Prime q :=
by
  -- Assume prime conditions
  assume prime_q : Prime q
  -- You would proceed to show contradiction here
  sorry

end q_is_composite_l194_194744


namespace soccer_games_l194_194355

theorem soccer_games (L : ℕ) (r_win r_loss r_tie : ℕ) 
  (losses : L = 9)
  (ratio : r_win = 4 ∧ r_loss = 3 ∧ r_tie = 1) :
  let parts_per_game := 3 in
  let total_parts := r_win + r_loss + r_tie in
  (total_parts * parts_per_game) = 24 :=
by
  sorry

end soccer_games_l194_194355


namespace find_z_area_of_triangle_l194_194513

def is_possible_z (z : ℂ) (x y : ℝ) : Prop :=
  z = x + y * complex.I ∧ complex.abs z = real.sqrt 2 ∧ complex.imag (z ^ 2) = 2

theorem find_z (z : ℂ) : 
  (∃ x y : ℝ, is_possible_z z x y) → 
  (z = 1 + complex.I ∨ z = -1 - complex.I) :=
by
  sorry

def area_triangle (A B C : ℂ) : ℝ :=
  real.abs ((A.re * (B.im - C.im) + B.re * (C.im - A.im) + C.re * (A.im - B.im)) / 2)

theorem area_of_triangle (z : ℂ) (A B C : ℂ) : 
  (A = z ∧ B = z^2 ∧ C = z - z^2) → 
  (area_triangle A B C = 1) :=
by
  sorry

end find_z_area_of_triangle_l194_194513


namespace modulus_of_complex_fraction_l194_194143

theorem modulus_of_complex_fraction (i : ℂ) (h_i : i^2 = -1) :
  abs ((1 - i) / (1 + i)) = 1 := by
  sorry

end modulus_of_complex_fraction_l194_194143


namespace not_product_of_consecutives_l194_194306

theorem not_product_of_consecutives (n k : ℕ) : 
  ¬ (∃ a b: ℕ, a + 1 = b ∧ (2 * n^(3 * k) + 4 * n^k + 10 = a * b)) :=
by sorry

end not_product_of_consecutives_l194_194306


namespace find_p8_l194_194631

noncomputable def p (x : ℝ) : ℝ := sorry

theorem find_p8 (h1 : p(1) = 1)
  (h2 : p(2) = 2)
  (h3 : p(3) = 3)
  (h4 : p(4) = 4)
  (h5 : p(5) = 5)
  (h6 : p(6) = 6)
  (h7 : p(7) = 7)
  (h8 : ∀ a b : ℝ, p(a) - p(b) = a - b) :
  p(8) = 5048 := 
by
  sorry

end find_p8_l194_194631


namespace num_four_digit_palindromic_squares_is_two_l194_194194

open Nat

-- Define the condition for a palindrome
def is_palindrome (n : ℕ) : Prop :=
  to_digits 10 n = (to_digits 10 n).reverse

-- Define the range of numbers to check
def range_32_to_99 := {x : ℕ | 32 ≤ x ∧ x ≤ 99}

-- Define the function to compute the square of a number
def square (n : ℕ) : ℕ := n * n

-- Define the set of 4-digit squares that are palindromes
def four_digit_palindromic_squares : Finset ℕ :=
  (Finset.filter (λ n => is_palindrome n) (Finset.image square (Finset.filter (λ n => 1000 ≤ square n ∧ square n < 10000) 
  (Finset.filter (λ n => n ∈ range_32_to_99) (Finset.range 100)))))

-- The main theorem stating the number of 4-digit palindromic squares
theorem num_four_digit_palindromic_squares_is_two :
  four_digit_palindromic_squares.card = 2 := sorry

end num_four_digit_palindromic_squares_is_two_l194_194194


namespace cost_of_two_pans_l194_194970

-- Given conditions
def pot_cost : ℝ := 20
def num_pots : ℝ := 3
def num_pans : ℝ := 4
def total_after_discount : ℝ := 100
def discount_rate : ℝ := 0.1

-- Total cost before discount
def total_before_discount : ℝ := total_after_discount / (1 - discount_rate)

-- The sum of cost of pots and pans before discount
def total_before_discount_sum : ℝ := num_pots * pot_cost + num_pans * cost_of_one_pan

-- Cost of one pan before discount
def cost_of_one_pan : ℝ := (total_before_discount - num_pots * pot_cost) / num_pans

-- Theorem to prove
theorem cost_of_two_pans : 2 * cost_of_one_pan = 25.56 := by
  sorry

end cost_of_two_pans_l194_194970


namespace prove_f_ff_neg4_eq_4_l194_194874

def f (x : ℝ) : ℝ :=
if x > 0 then x^(1/2) else (1/2)^x

theorem prove_f_ff_neg4_eq_4 : f (f (-4)) = 4 :=
by
  sorry

end prove_f_ff_neg4_eq_4_l194_194874


namespace cube_coordinates_are_integers_l194_194122

-- Define the cube and its properties
structure Cube :=
  (edge_length : ℤ)
  (face_vertices : List (ℤ × ℤ × ℤ))

-- Assumptions on the cube
axiom cube_has_integer_edges (c : Cube) : ∀ v ∈ c.face_vertices, ∃ e, v = (e, e, e) ∧ e ∈ ℤ

-- Parameters for the problem
variables (c : Cube) (edge : ℤ) (face : List (ℤ × ℤ × ℤ))

-- Function to get the vertices of the other face given the known face
noncomputable def other_face_vertices (known_face : List (ℤ × ℤ × ℤ)) : List (ℤ × ℤ × ℤ) :=
  sorry -- This function will be defined based on the problem's solution steps

-- The main theorem to prove
theorem cube_coordinates_are_integers (c : Cube) (h1 : c.edge_length ∈ ℤ)
  (h2 : ∀ v ∈ c.face_vertices, ∃ e, v = (e, e, e) ∧ e ∈ ℤ) :
  ∀ v ∈ other_face_vertices c.face_vertices, ∃ e, v = (e, e, e) ∧ e ∈ ℤ :=
by
  sorry

end cube_coordinates_are_integers_l194_194122


namespace coprime_ab_and_a_plus_b_l194_194275

theorem coprime_ab_and_a_plus_b (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a * b) (a + b) = 1 := by
  sorry

end coprime_ab_and_a_plus_b_l194_194275


namespace sum_of_distinct_integers_l194_194833

theorem sum_of_distinct_integers 
  (p q r s : ℕ) 
  (h1 : p * q = 6) 
  (h2 : r * s = 8) 
  (h3 : p * r = 4) 
  (h4 : q * s = 12) 
  (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) : 
  p + q + r + s = 13 :=
sorry

end sum_of_distinct_integers_l194_194833


namespace no_solution_in_natural_numbers_l194_194733

theorem no_solution_in_natural_numbers :
  ¬ ∃ (x y : ℕ), 2^x + 21^x = y^3 :=
sorry

end no_solution_in_natural_numbers_l194_194733


namespace average_people_added_each_year_l194_194441

-- a) Identifying questions and conditions
-- Question: What is the average number of people added each year?
-- Conditions: In 2000, about 450,000 people lived in Maryville. In 2005, about 467,000 people lived in Maryville.

-- c) Mathematically equivalent proof problem
-- Mathematically equivalent proof problem: Prove that the average number of people added each year is 3400 given the conditions.

-- d) Lean 4 statement
theorem average_people_added_each_year :
  let population_2000 := 450000
  let population_2005 := 467000
  let years_passed := 2005 - 2000
  let total_increase := population_2005 - population_2000
  total_increase / years_passed = 3400 := by
    sorry

end average_people_added_each_year_l194_194441


namespace circle_radius_l194_194097

theorem circle_radius (x y : ℝ) : x^2 - 10*x + y^2 + 4*y + 13 = 0 → ∃ r : ℝ, r = 4 :=
by
  -- sorry here to indicate that the proof is skipped
  sorry

end circle_radius_l194_194097


namespace pascal_triangle_contains_29_once_l194_194571

theorem pascal_triangle_contains_29_once (n : ℕ) :
  ( ∃! k : ℕ, k ∈ {i | ∃ j, i = binomial j (nat.succ i) ∧ j = 29}) ↔ n = 29 := 
begin
  sorry
end

end pascal_triangle_contains_29_once_l194_194571


namespace probability_expr_divisibility_l194_194656

/-- Define the set of positive integers {1, ..., 2015} -/
def positiveSet : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2015}

/-- Define the expression to check divisibility -/
def expr (a b c : ℕ) : ℕ := a * (b * c + b + 1)

/-- Define the main theorem statement -/
theorem probability_expr_divisibility (a b c : ℕ)
  (ha : a ∈ positiveSet) 
  (hb : b ∈ positiveSet) 
  (hc : c ∈ positiveSet) :
  (expr a b c) % 3 = 0 ∧ (expr a b c) % 5 = 0 → 
  ∃ P, P = (\textbf{C}) :=
sorry

end probability_expr_divisibility_l194_194656


namespace four_digit_palindrome_squares_count_l194_194208

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem four_digit_palindrome_squares_count : (Finset.filter (λ n, is_palindrome (n * n)) (Finset.range 100)).card = 2 := by
  sorry

end four_digit_palindrome_squares_count_l194_194208


namespace range_of_k_l194_194359

theorem range_of_k (x k : ℝ):
  (2 * x + 9 > 6 * x + 1) → (x - k < 1) → (x < 2) → k ≥ 1 :=
by 
  sorry

end range_of_k_l194_194359


namespace julieta_total_spent_l194_194254

theorem julieta_total_spent (original_backpack_price : ℕ)
                            (original_ringbinder_price : ℕ)
                            (backpack_price_increase : ℕ)
                            (ringbinder_price_decrease : ℕ)
                            (number_of_ringbinders : ℕ)
                            (new_backpack_price : ℕ)
                            (new_ringbinder_price : ℕ)
                            (total_ringbinder_cost : ℕ)
                            (total_spent : ℕ) :
  original_backpack_price = 50 →
  original_ringbinder_price = 20 →
  backpack_price_increase = 5 →
  ringbinder_price_decrease = 2 →
  number_of_ringbinders = 3 →
  new_backpack_price = original_backpack_price + backpack_price_increase →
  new_ringbinder_price = original_ringbinder_price - ringbinder_price_decrease →
  total_ringbinder_cost = new_ringbinder_price * number_of_ringbinders →
  total_spent = new_backpack_price + total_ringbinder_cost →
  total_spent = 109 := by
  intros
  sorry

end julieta_total_spent_l194_194254


namespace num_four_digit_square_palindromes_l194_194193

open Nat

-- Define what it means to be a 4-digit number
def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- The main theorem stating that there are exactly 2 four-digit squares that are palindromes
theorem num_four_digit_square_palindromes : 
  { n : ℕ | is_four_digit n ∧ is_palindrome n ∧ ∃ k : ℕ, k^2 = n ∧ k >= 32 ∧ k <= 99 }.to_finset.card = 2 :=
sorry

end num_four_digit_square_palindromes_l194_194193


namespace cody_initial_tickets_l194_194452

theorem cody_initial_tickets (T : ℕ) (h1 : T - 25 + 6 = 30) : T = 49 :=
sorry

end cody_initial_tickets_l194_194452


namespace longest_sticks_triangle_shortest_sticks_not_triangle_l194_194754

-- Define the lengths of the six sticks in descending order
variables {a1 a2 a3 a4 a5 a6 : ℝ}

-- Assuming the conditions
axiom h1 : a1 ≥ a2
axiom h2 : a2 ≥ a3
axiom h3 : a3 ≥ a4
axiom h4 : a4 ≥ a5
axiom h5 : a5 ≥ a6
axiom h6 : a1 + a2 > a3

-- Proof problem 1: It is always possible to form a triangle from the three longest sticks.
theorem longest_sticks_triangle : a1 < a2 + a3 := by sorry

-- Assuming an additional condition for proof problem 2
axiom two_triangles_formed : ∃ b1 b2 b3 b4 b5 b6: ℝ, 
  ((b1 + b2 > b3 ∧ b1 + b3 > b2 ∧ b2 + b3 > b1) ∧
   (b4 + b5 > b6 ∧ b4 + b6 > b5 ∧ b5 + b6 > b4 ∧ 
    a1 = b1 ∧ a2 = b2 ∧ a3 = b3 ∧ a4 = b4 ∧ a5 = b5 ∧ a6 = b6))

-- Proof problem 2: It is not always possible to form a triangle from the three shortest sticks.
theorem shortest_sticks_not_triangle : ¬(a4 < a5 + a6 ∧ a5 < a4 + a6 ∧ a6 < a4 + a5) := by sorry

end longest_sticks_triangle_shortest_sticks_not_triangle_l194_194754


namespace problem_proposition_l194_194136

-- Define propositions p and q
def p := ∀ x : ℝ, (0 < x) → (3^x > 2^x)
def q := ∃ x : ℝ, (x < 0) ∧ (3*x > 2*x)

-- Lean 4 statement to prove
theorem problem_proposition (hp: p) (hq: ¬ q) : p ∧ (¬ q) :=
by {
  exact ⟨hp, hq⟩
  sorry
}

end problem_proposition_l194_194136


namespace divisor_problem_l194_194024

theorem divisor_problem (n : ℕ) (hn_pos : 0 < n) (h72 : Nat.totient n = 72) (h5n : Nat.totient (5 * n) = 96) : ∃ k : ℕ, (n = 5^k * m ∧ Nat.gcd m 5 = 1) ∧ k = 2 :=
by
  sorry

end divisor_problem_l194_194024


namespace findEquationOfEllipse_maximumAreaOfTriangle_l194_194522

namespace EllipseProblem

def isEllipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ a^2 - b^2 = (a / (sqrt 3 / 3))^2

structure Ellipse where
  a : ℝ
  b : ℝ
  equation : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

structure Line where
  m : ℝ
  b : ℝ
  equation : ∀ x y : ℝ, y = m * x + b

def Circle (b : ℝ) : Prop := b > 0 ∧ ∀ x y : ℝ, x^2 + y^2 = b^2

def TriangleArea (O A B : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) (k : ℝ) : ℝ :=
  O.1 * A.2 - A.1 * O.2

theorem findEquationOfEllipse (h1 : isEllipse 3 (sqrt 2))
(h2 : ∀ x y : ℝ, (x^2) / 3 + (y^2) / 2 = 1)
(h3 : Circle (sqrt 2)) :
∀ x y : ℝ, (x^2) / 3 + (y^2) / 2 = 1 := 
sorry

theorem maximumAreaOfTriangle (h1 : ∀ x y : ℝ, (x^2) / 3 + (y^2) / 2 = 1)
(h2 : ∀ k > 0, ∀ x y : ℝ, |y| = k * x) :
∃ maxArea : ℝ, maxArea = sqrt 6 / 2 :=
sorry

end EllipseProblem

end findEquationOfEllipse_maximumAreaOfTriangle_l194_194522


namespace math_problem_l194_194990

theorem math_problem (n : ℕ) (a b : Fin n → ℝ) (c : Fin n → ℝ) (hc : ∀ i, 0 < c i) :
  ((∑ i j, (a i * a j) / (c i + c j)) * 
   (∑ i j, (b i * b j) / (c i + c j))) ≥ 
  ((∑ i j, (a i * b j) / (c i + c j)) ^ 2) :=
by 
  sorry

end math_problem_l194_194990


namespace hyperbola_chord_perimeter_l194_194009

theorem hyperbola_chord_perimeter
  (a b : ℝ)
  (a_pos : a = 4)
  (hyperbola_eq : ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1)
  (chord_length : ℝ)
  (chord_length_condition : chord_length = 6) :
  let F1 := (-sqrt (a^2 + b^2), 0),
      F2 := (sqrt (a^2 + b^2), 0),
      A := (x1, y1),
      B := (x2, y2),
      distance (a : ℝ × ℝ) (b : ℝ × ℝ) := sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) in
  (distance A F2 + distance B F2 + chord_length) = 28 :=
by sorry

end hyperbola_chord_perimeter_l194_194009


namespace find_number_l194_194922

theorem find_number (x y : ℝ) (h1 : x = y + 0.25 * y) (h2 : x = 110) : y = 88 := 
by
  sorry

end find_number_l194_194922


namespace team_savings_correct_l194_194566

noncomputable def total_cost_without_discount : ℝ :=
let price_shirt_A := 7.5 + 6 in
let price_pants_B := 20 in
let price_socks_A := 4.5 in
(12 * (price_shirt_A + price_pants_B + price_socks_A))

noncomputable def total_cost_with_discount : ℝ :=
let price_shirt_A_discount := 6.5 + 6 in
let price_pants_B_discount := 17 in
let price_socks_A_discount := 4 in
(12 * (price_shirt_A_discount + price_pants_B_discount + price_socks_A_discount) +
3 * 4.5 - 2 * 6 + 6)

noncomputable def savings : ℝ :=
total_cost_without_discount - total_cost_with_discount

theorem team_savings_correct : savings = 46.50 :=
by
  sorry

end team_savings_correct_l194_194566


namespace recurring_decimal_addition_l194_194484

noncomputable def recurring_decimal_sum : ℚ :=
  (23 / 99) + (14 / 999) + (6 / 9999)

theorem recurring_decimal_addition :
  recurring_decimal_sum = 2469 / 9999 :=
sorry

end recurring_decimal_addition_l194_194484


namespace y_six_power_eq_44_over_27_l194_194426

theorem y_six_power_eq_44_over_27
  (y : ℝ)
  (h_pos : 0 < y)
  (h_equation : ∛(2 - y^3) + ∛(2 + y^3) = 2)
  : y^6 = 44 / 27 :=
sorry

end y_six_power_eq_44_over_27_l194_194426


namespace average_shifted_variance_scaled_f_minimum_f_sum_lower_bound_l194_194843

noncomputable theory
open scoped BigOperators

variables {n : ℕ} {x : Fin n → ℝ} {s : ℝ}
variable [Fact (n > 0)]
def average (x : Fin n → ℝ) := (∑ i, x i) / n
def variance (x : Fin n → ℝ) (mean : ℝ) := (∑ i, (x i - mean) ^ 2) / n
def f (x : Fin n → ℝ) (a : ℝ) := ∑ i, (x i - a) ^ 2

-- The average of x1 + b, x2 + b, ..., xn + b is x̄ + b
theorem average_shifted (b : ℝ) : 
  average (λ i, x i + b) = average x + b :=
sorry

-- The variance of ax1, ax2, ..., axn is a^2 * s^2
theorem variance_scaled (a : ℝ) (hx : average x = 0) : 
  variance (λ i, a * x i) 0 = a^2 * variance x 0 :=
sorry

-- When x = x̄, the function f(x) has a minimum value of ns^2
theorem f_minimum (mean : ℝ) (hmean : average x = mean) : 
  f x mean = n * variance x mean :=
sorry

-- f(x1) + f(x2) + ... + f(xn) >= n^2 * s^2
theorem f_sum_lower_bound (hx : average x = 0) : 
  ∑ i, f x (x i) ≥ n^2 * variance x 0 :=
sorry

end average_shifted_variance_scaled_f_minimum_f_sum_lower_bound_l194_194843


namespace lcm_of_three_numbers_is_180_l194_194692

-- Define the three numbers based on the ratio and HCF condition
def a : ℕ := 2 * 6
def b : ℕ := 3 * 6
def c : ℕ := 5 * 6

-- State the theorem regarding the LCM
theorem lcm_of_three_numbers_is_180 : Nat.lcm (Nat.lcm a b) c = 180 :=
by
  sorry

end lcm_of_three_numbers_is_180_l194_194692


namespace find_a_values_l194_194497

theorem find_a_values (a : ℝ) :
  log 10 (a^2 - 17 * a) = 2 ↔ a = (17 + Real.sqrt 689) / 2 ∨ a = (17 - Real.sqrt 689) / 2 :=
by sorry

end find_a_values_l194_194497


namespace num_distinct_permutations_l194_194568

open Multiset

theorem num_distinct_permutations : 
  let s := {3, 3, 3, 5, 5, 7, 7, 9} in
  s.card = 8 ∧
  count 3 s = 3 ∧
  count 5 s = 2 ∧
  count 7 s = 2 ∧
  count 9 s = 1 →
  multiset.perm_card s = 1680 := 
by 
  intros s hs
  sorry

end num_distinct_permutations_l194_194568


namespace octal_67_equals_ternary_2001_l194_194065

def octalToDecimal (n : Nat) : Nat :=
  -- Definition of octal to decimal conversion omitted
  sorry

def decimalToTernary (n : Nat) : Nat :=
  -- Definition of decimal to ternary conversion omitted
  sorry

theorem octal_67_equals_ternary_2001 : 
  decimalToTernary (octalToDecimal 67) = 2001 :=
by
  -- Proof omitted
  sorry

end octal_67_equals_ternary_2001_l194_194065


namespace thomas_weekly_allowance_l194_194705

theorem thomas_weekly_allowance
  (total_savings_needed : ℕ := 13000)
  (second_year_savings : ℕ := 12220)
  (weeks_in_year : ℕ := 52)
  : (∃ A : ℕ, weeks_in_year * A + second_year_savings = total_savings_needed) → 
          ∃ A, A = 15 := 
by {
  intro h,
  cases h with A hA,
  use 15,
  linarith,
  sorry   -- Proof can be completed here
}

end thomas_weekly_allowance_l194_194705


namespace circle_radius_l194_194094

theorem circle_radius (x y : ℝ) : x^2 - 10 * x + y^2 + 4 * y + 13 = 0 → (x - 5)^2 + (y + 2)^2 = 4^2 :=
by
  sorry

end circle_radius_l194_194094


namespace num_four_digit_palindromic_squares_is_two_l194_194199

open Nat

-- Define the condition for a palindrome
def is_palindrome (n : ℕ) : Prop :=
  to_digits 10 n = (to_digits 10 n).reverse

-- Define the range of numbers to check
def range_32_to_99 := {x : ℕ | 32 ≤ x ∧ x ≤ 99}

-- Define the function to compute the square of a number
def square (n : ℕ) : ℕ := n * n

-- Define the set of 4-digit squares that are palindromes
def four_digit_palindromic_squares : Finset ℕ :=
  (Finset.filter (λ n => is_palindrome n) (Finset.image square (Finset.filter (λ n => 1000 ≤ square n ∧ square n < 10000) 
  (Finset.filter (λ n => n ∈ range_32_to_99) (Finset.range 100)))))

-- The main theorem stating the number of 4-digit palindromic squares
theorem num_four_digit_palindromic_squares_is_two :
  four_digit_palindromic_squares.card = 2 := sorry

end num_four_digit_palindromic_squares_is_two_l194_194199


namespace fraction_simplification_l194_194813

-- We define the given fractions
def a := 3 / 7
def b := 2 / 9
def c := 5 / 12
def d := 1 / 4

-- We state the main theorem
theorem fraction_simplification : (a - b) / (c + d) = 13 / 42 := by
  -- Skipping proof for the equivalence problem
  sorry

end fraction_simplification_l194_194813


namespace arrange_A_at_front_l194_194701

theorem arrange_A_at_front : ∃ n : ℕ, n = 24 ∧ n = nat.factorial 4 :=
by
  use nat.factorial 4
  split
  · rfl
  · sorry

end arrange_A_at_front_l194_194701


namespace largest_integer_coloring_l194_194138

noncomputable def largest_integer_with_property (α β : ℝ) (h1 : 1 < α) (h2 : α < β) : ℕ :=
⌈real.log β / real.log α⌉

theorem largest_integer_coloring (α β : ℝ) (h1 : 1 < α) (h2 : α < β) :
  ∃ r : ℕ, r = largest_integer_with_property α β h1 h2 ∧
    (∀ f : ℕ → ℕ, ∃ x y : ℕ, f x = f y ∧ α ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β) :=
sorry

end largest_integer_coloring_l194_194138


namespace cost_per_box_is_1_20_l194_194727

-- Define the dimensions of the box
def length : ℝ := 20
def width : ℝ := 20
def height : ℝ := 15

-- Define the total volume of the collection and the total cost
def total_volume : ℝ := 3060000
def total_cost : ℝ := 612

-- Calculate the volume of one box
def volume_of_one_box : ℝ := length * width * height

-- Calculate the number of boxes needed
def number_of_boxes : ℝ := total_volume / volume_of_one_box

-- Calculate the cost per box
noncomputable def cost_per_box : ℝ := total_cost / number_of_boxes

-- The theorem to prove
theorem cost_per_box_is_1_20 : cost_per_box = 1.20 := 
by 
  -- skip actual proof
  sorry

end cost_per_box_is_1_20_l194_194727


namespace baron_munchausen_not_lying_l194_194044

def sum_of_digits (n : Nat) : Nat := sorry

theorem baron_munchausen_not_lying :
  ∃ a b : Nat, a ≠ b ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a < 10^10 ∧ 10^9 ≤ a) ∧ (b < 10^10 ∧ 10^9 ≤ b) ∧ 
  (a + sum_of_digits (a ^ 2) = b + sum_of_digits (b ^ 2)) :=
sorry

end baron_munchausen_not_lying_l194_194044


namespace maximum_a_value_l194_194107

noncomputable def max_a_condition (x : ℝ) : Prop :=
  x ∈ Ioo (3 * Real.pi / 2) (2 * Real.pi)

def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  (Real.cbrt (Real.tan x) - Real.cbrt (Real.cot x))
    / (Real.cbrt (Real.sin x) + Real.cbrt (Real.cos x))
  > a / 2

theorem maximum_a_value :
  ∃ a : ℝ, (a = 4 * Real.root 6 2 ∧ ∀ x : ℝ, max_a_condition x → inequality_condition x a) :=
by
  sorry

end maximum_a_value_l194_194107


namespace infinite_solutions_system_l194_194467

theorem infinite_solutions_system :
  ∃ (x y : ℝ), (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = 10) :=
by
  existsi (λ y : ℝ, (3 * y + 5) / 2)
  intro y
  split
  · sorry
  · sorry

end infinite_solutions_system_l194_194467


namespace maize_total_l194_194444

constant monthly_storage : ℕ := 1
constant duration_years : ℕ := 2
constant theft : ℕ := 5
constant donation : ℕ := 8

theorem maize_total :
  (monthly_storage * 12 * duration_years - theft + donation) = 27 :=
by
  -- Proof would go here
  sorry

end maize_total_l194_194444


namespace hyperbola_eccentricity_l194_194560

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (h4 : c = 3 * b) 
  (h5 : c * c = a * a + b * b)
  (h6 : e = c / a) :
  e = 3 * Real.sqrt 2 / 4 :=
by
  sorry

end hyperbola_eccentricity_l194_194560


namespace cos_beta_third_quadrant_l194_194901

theorem cos_beta_third_quadrant (α β m : ℝ) (h1 : sin (α - β) * cos α - cos (α - β) * sin α = m)
  (h2 : (π < β) ∧ (β < 3*π/2)) :
  cos β = -sqrt(1 - m^2) :=
by
  -- Proof goes here
  sorry

end cos_beta_third_quadrant_l194_194901


namespace correct_value_is_48_l194_194342

-- Given conditions as Lean definitions
def num_observations : ℕ := 50
def initial_mean : ℚ := 36
def incorrect_value : ℚ := 23
def corrected_mean : ℚ := 36.5

-- The proof problem
theorem correct_value_is_48 :
  let initial_sum := num_observations * initial_mean
  let new_sum := num_observations * corrected_mean
  ∃ correct_value : ℚ, initial_sum - incorrect_value + correct_value = new_sum ∧ correct_value = 48 :=
by
  -- Adding the statement to be proven
  let initial_sum := num_observations * initial_mean
  let new_sum := num_observations * corrected_mean
  existsi (48 : ℚ)
  split
  -- Proof of initial_sum - incorrect_value + correct_value = new_sum
  { sorry }
  -- Proof of correct_value = 48
  { refl }

end correct_value_is_48_l194_194342


namespace seq_a_general_term_Tn_range_l194_194607

theorem seq_a_general_term (n : ℕ) (Sn : ℕ → ℕ) (an : ℕ → ℕ) :
  (∀ n, Sn n = n * (n + 1) / 2) →
  (∀ n, an n = Sn n - Sn (n - 1)) →
  an 1 = 1 ∧ (∀ n ≥ 2, an n = n) :=
by sorry

theorem Tn_range (n : ℕ) (an : ℕ → ℕ) (Tn : ℕ → ℝ) :
  (∀ n, an n = if n = 0 then 0 else n) →
  (∀ n, Tn n = ∑ k in Finset.range (n+1), (an k) / (2^k)) →
  (∀ n, 1 / 2 ≤ Tn n ∧ Tn n < 2) :=
by sorry

end seq_a_general_term_Tn_range_l194_194607


namespace find_PB_l194_194239

theorem find_PB 
  (C D B : Type) [linear_ordered_field C]
  (A P : B) 
  (CD_vertical_AB : CD ⊥ AB)
  (BC_vertical_AD : BC ⊥ AD)
  (CD_eq_39 : CD = 39)
  (BC_eq_50 : BC = 50)
  (AP_eq_15 : AP = 15) 
  : PB = Real.sqrt 2275 :=
sorry

end find_PB_l194_194239


namespace increasing_function_conditions_l194_194154

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 3
  else a * x + b

theorem increasing_function_conditions (a b : ℝ) (h1 : f.is_monotonic_increasing ℝ)
  : a > 0 ∧ b ≤ 3 :=
sorry

end increasing_function_conditions_l194_194154


namespace soccer_minimum_wins_l194_194027

/-
Given that a soccer team has won 60% of 45 matches played so far, 
prove that the minimum number of matches that the team still needs to win to reach a winning percentage of 75% is 27.
-/
theorem soccer_minimum_wins 
  (initial_matches : ℕ)                 -- the initial number of matches
  (initial_win_rate : ℚ)                -- the initial win rate (as a percentage)
  (desired_win_rate : ℚ)                -- the desired win rate (as a percentage)
  (initial_wins : ℕ)                    -- the initial number of wins

  -- Given conditions
  (h1 : initial_matches = 45)
  (h2 : initial_win_rate = 0.60)
  (h3 : desired_win_rate = 0.75)
  (h4 : initial_wins = 27):
  
  -- To prove: the minimum number of additional matches that need to be won is 27
  ∃ (n : ℕ), (initial_wins + n) / (initial_matches + n) = desired_win_rate ∧ 
                  n = 27 :=
by 
  sorry

end soccer_minimum_wins_l194_194027


namespace lassis_from_mangoes_l194_194459

theorem lassis_from_mangoes (lassis_per_three_mangoes : ℚ) (total_mangoes : ℚ) : ℚ :=
  have rate := lassis_per_three_mangoes / 3
  total_mangoes * rate

example : lassis_from_mangoes 7 15 = 35 :=
by 
  unfold lassis_from_mangoes
  simp
  norm_num
  sorry

end lassis_from_mangoes_l194_194459


namespace total_cost_correct_l194_194110

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end total_cost_correct_l194_194110


namespace elements_in_set_M_number_of_elements_in_set_M_l194_194150

variable (M : Set ℕ)

theorem elements_in_set_M : 
  {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4} → M = {1, 2} ∨ M = {1, 2, 3} ∨ M = {1, 2, 4} :=
sorry

theorem number_of_elements_in_set_M : 
  {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4} → M.card = 3 :=
sorry

end elements_in_set_M_number_of_elements_in_set_M_l194_194150


namespace evaluate_f_comp_f_l194_194553

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 + 1 else log x / log 3

theorem evaluate_f_comp_f :
  f (f (1 / 9)) = -7 :=
by
  sorry

end evaluate_f_comp_f_l194_194553


namespace coefficient_of_x3_term_l194_194118

theorem coefficient_of_x3_term :
  let m := ∫ x in 1..(Real.exp 2), 1/x
  (1 - m * x)^5 :=
  binom_expansion (1 - m * x) 5 :=
  let c := (binomial (5, 3)) * (-2)^3
  c = -80
:=
sorry

end coefficient_of_x3_term_l194_194118


namespace n_ice_cream_customers_l194_194469

theorem n_ice_cream_customers 
  (cone_cost : ℝ) 
  (total_revenue : ℝ) 
  (free_cones : ℕ) 
  (P F T n : ℕ) 
  (cost_eq : cone_cost = 2) 
  (revenue_eq : total_revenue = 100) 
  (F_eq : F = 10) 
  (P_calc : P = total_revenue / cone_cost) 
  (T_calc : T = P + F) 
  (n_calc : n = T / F) 
  : n = 6 :=
by
  have : cone_cost = 2 := cost_eq
  have : total_revenue = 100 := revenue_eq
  have : F = 10 := F_eq
  have hp : P = 50 := by calc
    P = total_revenue / cone_cost : P_calc
      _ = 100 / 2 : by rw [this, this]
      _ = 50 : by norm_num
  have ht : T = 60 := by
    rw [T_calc, hp, this]
    norm_num
  calc
  n = T / F : n_calc
    _ = 60 / 10 : by rw [ht, this]
    _ = 6 : by norm_num

end n_ice_cream_customers_l194_194469


namespace construct_symmetric_point_l194_194131

variables {Point : Type} [inhabited Point] [metric_space Point]
variables (A A' B l : Point)
variables (symmetric : ∀ (x y : Point), y = 2 * proj l x - x)

-- Definitions of symmetry and line projection:
def symmetric_point (p l : Point) := 2 * proj l p - p
def proj (l p : Point) := sorry  -- Assume projection logic here

/- Theorem stating that the symmetric point B' can be constructed given the conditions -/
theorem construct_symmetric_point (hA : A' = symmetric_point A l) (hB : True) :
  ∃ B', B' = symmetric_point B l :=
sorry

end construct_symmetric_point_l194_194131


namespace units_digit_of_result_is_2_l194_194684

-- Define the conditions and theorem
theorem units_digit_of_result_is_2 (a b c : ℕ) (h1 : a = c + 3) :
  let result := (101 * c + 10 * b + 405) - (101 * c + 10 * b + 3)
  in result % 10 = 2 :=
by
  sorry

end units_digit_of_result_is_2_l194_194684


namespace celine_collected_10_l194_194051

variable (G C J E D : ℕ)

/-- Assume Gabriel, Celine, Julian, Erica, and David collected G, 2G, 4G, 12G, and 60G erasers respectively --/
variable (h1 : C = 2 * G)
variable (h2 : J = 4 * C)
variable (h3 : E = 12 * J)
variable (h4 : D = 60 * E)
variable (h5 : G + C + J + E + D = 380)

theorem celine_collected_10 : C = 10 :=
by
  sorry

end celine_collected_10_l194_194051


namespace arccos_cos_eq_x_div_3_solutions_l194_194663

theorem arccos_cos_eq_x_div_3_solutions (x : ℝ) :
  (Real.arccos (Real.cos x) = x / 3) ∧ (-3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2) 
  ↔ x = -3 * Real.pi / 2 ∨ x = 0 ∨ x = 3 * Real.pi / 2 :=
by
  sorry

end arccos_cos_eq_x_div_3_solutions_l194_194663


namespace equation_of_plane_through_A_perpendicular_to_BC_l194_194395

section

variable (A : ℝ × ℝ × ℝ)
variable (B : ℝ × ℝ × ℝ)
variable (C : ℝ × ℝ × ℝ)

def plane_equation (A B C : ℝ × ℝ × ℝ) : Prop :=
  let n : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3) in
  let (a, b, c) := n in
  a * (x + 7) + b * (y - 1) + c * (z + 4) = 0

theorem equation_of_plane_through_A_perpendicular_to_BC :
  A = (-7, 1, -4) →
  B = (8, 11, -3) →
  C = (9, 9, -1) →
  plane_equation A B C = (x - 2y + 2z + 17 = 0)
  :=
by
  intros hA hB hC
  sorry

end

end equation_of_plane_through_A_perpendicular_to_BC_l194_194395


namespace f_eval_at_e_plus_1_f_range_l194_194514

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then x^2 + 4 * x + 3 else Real.log (x - 1) + 1

theorem f_eval_at_e_plus_1 : f (Real.exp 1 + 1) = 2 :=
by
  sorry

theorem f_range :
  set.range f = set.Ici (-1 : ℝ) :=
by
  sorry

end f_eval_at_e_plus_1_f_range_l194_194514


namespace gcf_of_180_270_450_l194_194387

theorem gcf_of_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  have prime_factor_180 : ∃ (a b c : ℕ), 180 = 2^2 * 3^2 * 5 := ⟨2, 2, 1, rfl⟩
  have prime_factor_270 : ∃ (a b c : ℕ), 270 = 2 * 3^3 * 5 := ⟨1, 3, 1, rfl⟩
  have prime_factor_450 : ∃ (a b c : ℕ), 450 = 2 * 3^2 * 5^2 := ⟨1, 2, 2, rfl⟩
  sorry

end gcf_of_180_270_450_l194_194387


namespace unique_integer_triplet_solution_l194_194486

theorem unique_integer_triplet_solution (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : 
    (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end unique_integer_triplet_solution_l194_194486


namespace sequence_value_l194_194883

theorem sequence_value (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 :=
by
  -- The proof is not required, so we add sorry to indicate that
  sorry

end sequence_value_l194_194883


namespace cube_difference_l194_194855

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : a^3 - b^3 = 124 :=
by sorry

end cube_difference_l194_194855


namespace sum_of_digits_9999_7777_equals_720_l194_194495

theorem sum_of_digits_9999_7777_equals_720 :
    ∀ (n : ℕ), n = 80 → (∑ d in (digits (10^80 - 1) * digits (7 * 10^80 - 7)), d) = 720 :=
by
  intro n
  intro hn
  rw hn
  sorry

end sum_of_digits_9999_7777_equals_720_l194_194495


namespace find_a_plus_b_l194_194146

-- Given points A and B, where A(1, a) and B(b, -2) are symmetric with respect to the origin.
variables (a b : ℤ)

-- Definition for symmetry conditions
def symmetric_wrt_origin (x1 y1 x2 y2 : ℤ) :=
  x2 = -x1 ∧ y2 = -y1

-- The main theorem
theorem find_a_plus_b :
  symmetric_wrt_origin 1 a b (-2) → a + b = 1 :=
by
  sorry

end find_a_plus_b_l194_194146


namespace distance_of_intersection_points_l194_194337

open Classical

noncomputable def distance_between_intersections : ℝ :=
  let Cx1 := (-1 + Real.sqrt 10) / 3
  let Cx2 := (-1 - Real.sqrt 10) / 3
  Real.abs (Cx1 - Cx2)

theorem distance_of_intersection_points :
  distance_between_intersections = 2 * Real.sqrt 10 / 3 :=
by
  sorry

end distance_of_intersection_points_l194_194337


namespace square_root_unique_l194_194363

theorem square_root_unique (x : ℝ) (h1 : x + 3 ≥ 0) (h2 : 2 * x - 6 ≥ 0)
  (h : (x + 3)^2 = (2 * x - 6)^2) :
  x = 1 ∧ (x + 3)^2 = 16 := 
by
  sorry

end square_root_unique_l194_194363


namespace C_share_of_profit_l194_194032

-- Given conditions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

-- Objective to prove that C's share of the profit is given by 36000
theorem C_share_of_profit : (total_profit / (investment_A / investment_C + investment_B / investment_C + 1)) = 36000 :=
by
  sorry

end C_share_of_profit_l194_194032


namespace even_factors_count_l194_194473

-- Define the number m
def m : ℕ := 2^3 * 5^1 * 11^2

-- Define the condition for an even factor
def isEvenFactor (n : ℕ) : Prop := n ∣ m ∧ ∃ k, 2 * k = n

-- Prove the number of even factors of m is 18
theorem even_factors_count : (finset.filter (λ n, isEvenFactor n) (finset.range (m + 1))).card = 18 := 
sorry

end even_factors_count_l194_194473


namespace range_of_sum_l194_194913

theorem range_of_sum {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : a * b = a + b + 3) : 6 ≤ a + b :=
begin
  sorry
end

end range_of_sum_l194_194913


namespace original_denominator_is_18_l194_194774

variable (d : ℕ)

theorem original_denominator_is_18
  (h1 : ∃ (d : ℕ), (3 + 7) / (d + 7) = 2 / 5) :
  d = 18 := 
sorry

end original_denominator_is_18_l194_194774


namespace algebra_inequality_l194_194498

theorem algebra_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : 
  (a / real.sqrt (a^2 + 8 * b * c) + b / real.sqrt (b^2 + 8 * c * a) + c / real.sqrt (c^2 + 8 * a * b)) >= 1 := 
sorry

end algebra_inequality_l194_194498


namespace probability_toner_never_displayed_l194_194711

theorem probability_toner_never_displayed:
  let total_votes := 129
  let toner_votes := 63
  let celery_votes := 66
  (toner_votes + celery_votes = total_votes) →
  let probability := (celery_votes - toner_votes) / (celery_votes + toner_votes)
  probability = 1 / 43 := 
by
  sorry

end probability_toner_never_displayed_l194_194711


namespace probability_of_highest_number_six_l194_194003

open ProbabilityTheory

-- Definitions of the condition
def box : set ℕ := {1, 2, 3, 4, 5, 6, 7}
def number_of_cards_selected := 4

-- Probability that the highest number selected is 6
def probability_highest_is_six : ℚ :=
  (3 / 7 : ℚ)

theorem probability_of_highest_number_six :
  ∃ p : ℚ, p = probability_highest_is_six ∧ 
    probability_space (finset.powerset_len number_of_cards_selected box) 
      (λ s, 6 ∈ s ∧ ∀ x ∈ s, x ≤ 6) = p := by
  sorry

end probability_of_highest_number_six_l194_194003


namespace sum_internal_angles_40deg_l194_194331

theorem sum_internal_angles_40deg {e : ℝ} (h : e = 40) : 
  let n := 360 / e in ((n - 2) * 180 = 1260) :=
by
  sorry

end sum_internal_angles_40deg_l194_194331


namespace find_f_l194_194217

-- Definitions of odd and even functions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x, g (-x) = g x

-- Main theorem
theorem find_f (f g : ℝ → ℝ) (h_odd_f : odd_function f) (h_even_g : even_function g) 
    (h_eq : ∀ x, f x + g x = 1 / (x - 1)) :
  ∀ x, f x = x / (x ^ 2 - 1) :=
by
  sorry

end find_f_l194_194217


namespace find_missing_digit_l194_194089

theorem find_missing_digit :
  ∃ d3 : ℕ, (3 + 5 + d3 + 7 + 2) % 9 = 0 ∧ d3 < 10 :=
begin
  use 1,
  split,
  { norm_num },
  { norm_num }
end

end find_missing_digit_l194_194089


namespace need_to_buy_more_dice_l194_194643

theorem need_to_buy_more_dice (mark_dice : ℕ) (mark_percent_12_sided : ℕ) (james_dice : ℕ) (james_percent_12_sided : ℕ) (total_needed_12_sided : ℕ) :
  mark_dice = 10 → mark_percent_12_sided = 60 →
  james_dice = 8 → james_percent_12_sided = 75 →
  total_needed_12_sided = 14 →
  (total_needed_12_sided - (mark_dice * mark_percent_12_sided / 100 + james_dice * james_percent_12_sided / 100)) = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end need_to_buy_more_dice_l194_194643


namespace conditional_probability_of_white_balls_l194_194232

theorem conditional_probability_of_white_balls
  (w r : ℕ)
  (A B : Set (Fin w + r))
  (ha : A.card = 5)
  (hb : B.card = 5 - 1)
  (draw_without_replacement : ∀ (x y : Fin w + r), x ≠ y → x ∈ A → y ∈ B)
  : P(B |A) = 4/7 
  := sorry

end conditional_probability_of_white_balls_l194_194232


namespace area_inequality_l194_194953

noncomputable theory

variables {A B C D K L M N : Point}
variables {ABC AKL : Triangle}
variables {S T : ℝ}

-- Conditions
axiom h₁ : is_right_triangle A B C
axiom h₂ : altitude_from A D B C
axiom h₃ : incenter_intersects ABD AB K
axiom h₄ : incenter_intersects ACD AC L
axiom h₅ : triangle_area ABC = S
axiom h₆ : triangle_area AKL = T

-- Question
theorem area_inequality : S ≥ 2 * T := sorry

end area_inequality_l194_194953


namespace population_function_time_to_reach_1_2_million_growth_rate_control_l194_194677

-- Define the initial conditions
def initial_population : ℝ := 100 -- in ten thousand
def growth_rate : ℝ := 1.012

-- (1) Function relationship for population over time
theorem population_function (x : ℕ) : 
  (initial_population * growth_rate^x) = 100 * (1 + 1.2/100)^x := by
  sorry

-- (2) Time to reach 1.2 million population
theorem time_to_reach_1_2_million : 
  ∃ n : ℕ, initial_population * growth_rate^n ≈ 120 := by
  sorry

-- (3) Growth rate control to not exceed 1.2 million in 20 years
theorem growth_rate_control : 
  ∀ a : ℝ, (∀ t : ℕ, t ≤ 20 → initial_population * (1 + a/100)^t ≤ 120) → a ≤ 0.9 := by
  sorry

end population_function_time_to_reach_1_2_million_growth_rate_control_l194_194677


namespace length_of_de_l194_194734

theorem length_of_de
  {a b c d e : ℝ} 
  (h1 : b - a = 5) 
  (h2 : c - a = 11) 
  (h3 : e - a = 22) 
  (h4 : c - b = 2 * (d - c)) :
  e - d = 8 :=
by 
  sorry

end length_of_de_l194_194734


namespace tetrahedron_has_triangle_vertex_l194_194305

structure Tetrahedron :=
  (A B C D : Type)
  (dist : A → B → ℝ)
  (dist_nonneg : ∀ x y, dist x y ≥ 0)
  (dist_eq_zero_iff : ∀ x y, dist x y = 0 ↔ x = y)
  (dist_symm : ∀ x y, dist x y = dist y x)
  (dist_triangle : ∀ x y z, dist x z ≤ dist x y + dist y z)

theorem tetrahedron_has_triangle_vertex (T : Tetrahedron) :
  ∃ v : T.A, ∃ x y z : T.A, x ≠ y ∧ T.dist x v > 0 ∧ T.dist y v > 0 ∧ T.dist z v > 0 ∧ T.dist v x + T.dist v y > T.dist v z ∧ T.dist v y + T.dist v z > T.dist v x ∧ T.dist v z + T.dist v x > T.dist v y :=
  sorry

end tetrahedron_has_triangle_vertex_l194_194305


namespace find_ab_of_cubic_polynomials_l194_194493

theorem find_ab_of_cubic_polynomials 
  (a b : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ r ≠ 0 ∧ s ≠ 0 ∧ 
    (r * s * (r + s) + a * (r + s) * r * s + 8 * (r + s) * r + 12 = 0) ∧ 
    (r * s * (r + s) + b * (r + s) * r * s + 20 * (r + s) * r + 16 = 0))
  (h2 : ∃ k : ℤ, (x : ℝ) = k ∧ k has a real root of both polynomial equations of the forms 
    x^3 + ax^2 + 8x + 12 = 0 and x^3 + bx^2 + 20x + 16 = 0) :
  (a, b) = (7, 6) := sorry

end find_ab_of_cubic_polynomials_l194_194493


namespace num_4_digit_palindromic_squares_l194_194203

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_n (n : ℕ) : Prop :=
  32 ≤ n ∧ n ≤ 99

theorem num_4_digit_palindromic_squares : ∃ (count : ℕ), count = 3 ∧ ∀ n, valid_n n → is_4_digit (n^2) → is_palindrome (n^2) :=
sorry

end num_4_digit_palindromic_squares_l194_194203


namespace determine_numbers_l194_194400

theorem determine_numbers (x : ℕ → ℕ) (n : ℕ) (S V : ℕ) 
  (h1 : S = ∑ i in Finset.range 10, x i)
  (h2 : V = ∑ i in Finset.range 10, x i * 10^(i * n))
  (h3 : ∀ i ∈ Finset.range 10, x i < 10^n) :
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℕ), 
  (x 0 = x1) ∧ (x 1 = x2) ∧ (x 2 = x3) ∧ (x 3 = x4) ∧ 
  (x 4 = x5) ∧ (x 5 = x6) ∧ (x 6 = x7) ∧ (x 7 = x8) ∧ 
  (x 8 = x9) ∧ (x 9 = x10) :=
sorry

end determine_numbers_l194_194400


namespace dart_scores_3_probability_l194_194653

/-- Definitions from conditions --/
def outer_radius := 8
def inner_radius := 4
def inner_scores := [3, 4, 4]
def outer_scores := [4, 3, 3]
def dart_prob_is_proportional_to_area := true

/-- Area calculations based on the conditions --/
def inner_area := real.pi * inner_radius^2
def outer_area := real.pi * (outer_radius^2 - inner_radius^2)
def total_board_area := real.pi * outer_radius^2

/-- Total areas of the regions with score 3 --/
def inner_area_score3 := inner_area / 3
def outer_area_score3 := (outer_area / 3) * 2
def total_area_score3 := inner_area_score3 + outer_area_score3

/-- The probability calculation --/
def probability_dart_scores_3 := total_area_score3 / total_board_area

/-- Proof statement: probability that a single dart throw scores 3 is 7/12 --/
theorem dart_scores_3_probability : probability_dart_scores_3 = 7 / 12 :=
by
  sorry

end dart_scores_3_probability_l194_194653


namespace tv_price_with_tax_l194_194572

-- Define the original price of the TV
def originalPrice : ℝ := 1700

-- Define the value-added tax rate
def taxRate : ℝ := 0.15

-- Calculate the total price including tax
theorem tv_price_with_tax : originalPrice * (1 + taxRate) = 1955 :=
by
  sorry

end tv_price_with_tax_l194_194572


namespace four_digit_palindrome_squares_count_l194_194209

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem four_digit_palindrome_squares_count : (Finset.filter (λ n, is_palindrome (n * n)) (Finset.range 100)).card = 2 := by
  sorry

end four_digit_palindrome_squares_count_l194_194209


namespace triangle_XYZ_perimeter_l194_194601

theorem triangle_XYZ_perimeter (XY YZ : ℝ) (XYZ_right : XY = 6 ∧ YZ = 8 ∧ ∃ Z : ℝ, 
  XY^2 + YZ^2 = Z^2) : XY + YZ + Real.sqrt (XY^2 + YZ^2) = 24 :=
by
  -- given conditions
  let XY := 6
  let YZ := 8
  have XZ2 : ℝ := XY^2 + YZ^2
  have XZ : ℝ := Real.sqrt XZ2
  have perimeter : ℝ := XY + YZ + XZ
  -- assertion
  have perimeter_correct : perimeter = 24
  -- proof of perimeter
  have : XY^2 + YZ^2 = 100 := by
    -- Pythagorean theorem
    have hXY : XY^2 = 36 := by sorry
    have hYZ : YZ^2 = 64 := by sorry
    show XY^2 + YZ^2 = 100 from by
      rw [hXY, hYZ]
  have hXZ : XZ = 10 := by
    rw [this]
    show Real.sqrt 100 = 10
  show XY + YZ + XZ = 24 from by 
    rw [hXZ]
    show 6 + 8 + 10 = 24

end triangle_XYZ_perimeter_l194_194601


namespace rectangle_area_in_semicircle_l194_194038

theorem rectangle_area_in_semicircle (DA FD AE : ℝ) (DA_eq : DA = 16) (FD_AE_eq : FD = 9) :
  let EF := FD + DA + AE in
  let OC := EF / 2 in
  let OD := DA / 2 in
  let OC_sq := OC * OC in
  let OD_sq := OD * OD in
  let CD := real.sqrt (OC_sq - OD_sq) in
  let area := DA * CD in
  EF = 34 → OC = 17 → OD = 8 → CD = 15 → area = 240 :=
by
  sorry

end rectangle_area_in_semicircle_l194_194038


namespace sylvia_time_to_complete_job_l194_194665

theorem sylvia_time_to_complete_job (S : ℝ) (h₁ : 18 ≠ 0) (h₂ : 30 ≠ 0)
  (together_rate : (1 / S) + (1 / 30) = 1 / 18) :
  S = 45 :=
by
  -- Proof will be provided here
  sorry

end sylvia_time_to_complete_job_l194_194665


namespace maximum_a_value_l194_194106

noncomputable def max_a_condition (x : ℝ) : Prop :=
  x ∈ Ioo (3 * Real.pi / 2) (2 * Real.pi)

def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  (Real.cbrt (Real.tan x) - Real.cbrt (Real.cot x))
    / (Real.cbrt (Real.sin x) + Real.cbrt (Real.cos x))
  > a / 2

theorem maximum_a_value :
  ∃ a : ℝ, (a = 4 * Real.root 6 2 ∧ ∀ x : ℝ, max_a_condition x → inequality_condition x a) :=
by
  sorry

end maximum_a_value_l194_194106


namespace find_a_l194_194323

theorem find_a (a : ℝ) :
  let expansion_coeff := (binom 6 3) - 2 * a * (binom 6 2) + a^2 * (binom 6 1) in
  expansion_coeff = -16 → a = 2 ∨ a = 3 :=
by
  intro h
  sorry

end find_a_l194_194323


namespace chloe_probability_l194_194052

theorem chloe_probability :
  let total_numbers := 60
  let multiples_of_4 := 15
  let non_multiples_of_4_prob := 3 / 4
  let neither_multiple_of_4_prob := (non_multiples_of_4_prob) ^ 2
  let at_least_one_multiple_of_4_prob := 1 - neither_multiple_of_4_prob
  at_least_one_multiple_of_4_prob = 7 / 16 := by
  sorry

end chloe_probability_l194_194052


namespace pedestrian_avg_waiting_time_at_traffic_light_l194_194435

theorem pedestrian_avg_waiting_time_at_traffic_light :
  ∀ cycle_time green_time red_time : ℕ,
  cycle_time = green_time + red_time →
  green_time = 1 →
  red_time = 2 →
  let prob_green := green_time / cycle_time in
  let prob_red := red_time / cycle_time in
  let E_T_given_green := 0 in
  let E_T_given_red := (0 + 2) / 2 in
  (E_T_given_green * prob_green + E_T_given_red * prob_red) * 60 = 40 := 
by
  -- Insert proof here
  sorry

end pedestrian_avg_waiting_time_at_traffic_light_l194_194435


namespace divides_rearrangement_of_digits_l194_194069

theorem divides_rearrangement_of_digits (d n : ℕ) (h_d_pos : 0 < d) (h_n_pos : 0 < n) :
  (∀ m, (has_same_digits n m) → d ∣ m) ↔ d ∈ {1, 3, 9} :=
by
  sorry

end divides_rearrangement_of_digits_l194_194069


namespace interval_inequality_solution_l194_194083

theorem interval_inequality_solution (x : ℝ) :
  (x ∈ Set.Ioo (-Real.sqrt(9/2)) (-1) ∨ x ∈ Set.Ioo (-1) 1 ∨ x ∈ Set.Ioo 1 (Real.sqrt(9/2))) ↔
  (x^2 - 4.5) / (x^2 - 1) ≤ 0 :=
sorry

end interval_inequality_solution_l194_194083


namespace xp_passes_through_midpoint_l194_194984

theorem xp_passes_through_midpoint (A B C P M N X T : Point)
  (h1 : is_triangle A B C)
  (h2 : is_foot_of_altitude_from A P)
  (h3 : midpoint M A B)
  (h4 : midpoint N A C)
  (h5 : is_circumcircle_intersection X B M P C N P (X ≠ P))
  (h6 : midpoint T M N) :
  lies_on_line X P T :=
sorry

end xp_passes_through_midpoint_l194_194984


namespace trig_eq_num_solutions_l194_194093

theorem trig_eq_num_solutions :
  (∃ (s : finset ℝ),
    (∀ x : ℝ, (cos (4 * x) + cos (3 * x) ^ 2 + cos (2 * x) ^ 3 + cos (x) ^ 4 + sin (x) ^ 2 = 0) ↔ x ∈ s)
    ∧ s.card = 10) ↔
  (-π ≤ x ∧ x ≤ π) :=
begin
  sorry
end

end trig_eq_num_solutions_l194_194093


namespace length_of_goods_train_l194_194418

theorem length_of_goods_train (speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) (length_of_train : ℝ) :
  speed_kmph = 96 → platform_length = 360 → crossing_time = 32 → length_of_train = (26.67 * 32 - 360) :=
by
  sorry

end length_of_goods_train_l194_194418


namespace evaluate_9x_plus_2_l194_194574

theorem evaluate_9x_plus_2 (x : ℝ) (h : 3^(2*x) = 17) : 9^(x + 2) = 1377 := 
by sorry

end evaluate_9x_plus_2_l194_194574


namespace sum_of_arithmetic_series_l194_194365

theorem sum_of_arithmetic_series (A B C : ℕ) (n : ℕ) 
  (hA : A = n * (2 * a₁ + (n - 1) * d) / 2)
  (hB : B = 2 * n * (2 * a₁ + (2 * n - 1) * d) / 2)
  (hC : C = 3 * n * (2 * a₁ + (3 * n - 1) * d) / 2) :
  C = 3 * (B - A) := sorry

end sum_of_arithmetic_series_l194_194365


namespace tan_alpha_in_second_quadrant_l194_194837

theorem tan_alpha_in_second_quadrant (α : ℝ) (h1 : sin α = 4 / 5) (h2 : π / 2 < α ∧ α < π) : tan α = -4 / 3 :=
by
  sorry

end tan_alpha_in_second_quadrant_l194_194837


namespace quadratic_bounds_l194_194882

variable (a b c: ℝ)

-- Conditions
def quadratic_function (x: ℝ) : ℝ := a * x^2 + b * x + c

def within_range_neg_1_to_1 (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7

-- Main statement
theorem quadratic_bounds
  (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7 := sorry

end quadratic_bounds_l194_194882


namespace three_digit_numerals_with_prime_ending_l194_194213

def prime_digits : Finset ℕ := {2, 3, 5, 7}

theorem three_digit_numerals_with_prime_ending :
  let count := (9 * 10 * 4) in  -- hundreds place: 9 options, tens place: 10 options, ones place: 4 options
  count = 360 :=
by
  sorry

end three_digit_numerals_with_prime_ending_l194_194213


namespace tan_alpha_one_sin_alpha_plus_pi_four_one_l194_194835

variables (α : ℝ) (h_cos : cos α = (√2) / 2) (h_alpha : 0 < α ∧ α < π / 2)

theorem tan_alpha_one : tan α = 1 :=
sorry

theorem sin_alpha_plus_pi_four_one : sin (α + π / 4) = 1 :=
sorry

end tan_alpha_one_sin_alpha_plus_pi_four_one_l194_194835


namespace q_composition_l194_194283

def q (x y : ℝ) : ℝ :=
  if x ≥ 0 ∧ y ≥ 0 then x^2 + y^2
  else if x < 0 ∧ y < 0 then x^2 - 3 * y
  else 2 * x + 2 * y

theorem q_composition : q (q 2 (-2)) (q (-3) (-1)) = 144 := by
  sorry

end q_composition_l194_194283


namespace coeff_x_15_in_poly_l194_194464

-- Definitions and conditions
def poly1 (x : ℕ → ℤ) : ℤ[X] := (polynomial.sum (polynomial.range 21) (λ n : ℕ, polynomial.monomial n 1))
def poly2 (x : ℕ → ℤ) : ℤ[X] := (polynomial.sum (polynomial.range 11) (λ n : ℕ, polynomial.monomial n 1)) ^ 2
def poly : ℤ[X] := poly1 x * poly2 x

-- Theorem statement
theorem coeff_x_15_in_poly (x : ℕ → ℤ) : polynomial.coeff poly 15 = 106 := 
sorry

end coeff_x_15_in_poly_l194_194464


namespace profit_without_discount_l194_194770

theorem profit_without_discount (CP SP_original SP_discount : ℝ) (h1 : CP > 0) (h2 : SP_discount = CP * 1.14) (h3 : SP_discount = SP_original * 0.95) :
  (SP_original - CP) / CP * 100 = 20 :=
by
  have h4 : SP_original = SP_discount / 0.95 := by sorry
  have h5 : SP_original = CP * 1.2 := by sorry
  have h6 : (SP_original - CP) / CP * 100 = (CP * 1.2 - CP) / CP * 100 := by sorry
  have h7 : (SP_original - CP) / CP * 100 = 20 := by sorry
  exact h7

end profit_without_discount_l194_194770


namespace proposition_p_or_q_l194_194851

theorem proposition_p_or_q (p q : Prop) 
  (h₁ : ∀ x : ℝ, 2^x > 0) 
  (h₂ : ¬ ∃ x : ℝ, sin x = 2) : p ∨ q :=
by
  have hp : p := sorry -- Proof that p is true
  have hq : ¬ q := sorry -- Proof that q is false
  exact Or.inl hp -- Hence, p ∨ q is true.

end proposition_p_or_q_l194_194851


namespace find_k_l194_194515

noncomputable def f : ℝ → ℝ := sorry
variable k : ℝ

axiom f_condition1 : f 1 = 1
axiom f_condition2 : f 7 = 163
axiom f_property : ∀ x y : ℝ, f (x + y) = f x + f y + k * x * y - 2

theorem find_k : k = 8 := 
by
  sorry

end find_k_l194_194515


namespace polynomial_sum_of_squares_l194_194996

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 < P.eval x) :
  ∃ (U V : Polynomial ℝ), P = U^2 + V^2 := 
by
  sorry

end polynomial_sum_of_squares_l194_194996


namespace simplify_expression_to_inverse_abc_l194_194063

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem simplify_expression_to_inverse_abc :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ca + 3)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (ca)⁻¹ + 3) = (1 : ℝ) / (abc) :=
by
  sorry

end simplify_expression_to_inverse_abc_l194_194063


namespace find_y_six_l194_194424

theorem find_y_six (y : ℝ) (h : y > 0) (h_eq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
    y^6 = 116 / 27 :=
by
  sorry

end find_y_six_l194_194424


namespace hyperbola_foci_distance_l194_194474

theorem hyperbola_foci_distance (a b : ℝ) (ha : a^2 = 45) (hb : b^2 = 5) :
  let c := Real.sqrt (a^2 + b^2) in 2 * c = 10 * Real.sqrt 2 :=
by
  sorry

end hyperbola_foci_distance_l194_194474


namespace comic_book_arrangement_l194_194651

theorem comic_book_arrangement :
  let spiderman_books := 7
  let archie_books := 6
  let garfield_books := 5
  let groups := 3
  Nat.factorial spiderman_books * Nat.factorial archie_books * Nat.factorial garfield_books * Nat.factorial groups = 248005440 :=
by
  sorry

end comic_book_arrangement_l194_194651


namespace find_lambda_l194_194850

def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (2, 6)
def EF (λ : ℝ) : ℝ × ℝ := (-1, λ)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_lambda (λ : ℝ) (h : dot_product ((Q.1 - P.1, Q.2 - P.2)) (EF λ) = 0) : λ = -1 / 2 :=
by sorry

end find_lambda_l194_194850


namespace olympiad_2002_largest_smallest_sum_l194_194246

theorem olympiad_2002_largest_smallest_sum:
  ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 ∧ 
  (∀ A' B' C' : ℕ, A' ≠ B' ∧ B' ≠ C' ∧ A' ≠ C' ∧ A' * B' * C' = 2310 →
   A + B + C ≥ A' + B' + C') ∧ 
  (∀ A'' B'' C'' : ℕ, A'' ≠ B'' ∧ B'' ≠ C'' ∧ A'' ≠ C'' ∧ A'' * B'' * C'' = 2310 →
   A + B + C ≤ A'' + B'' + C'') :=
by {
  let A_largest := 6,
  let B_largest := 35,
  let C_largest := 11,
  let A_smallest := 2,
  let B_smallest := 3,
  let C_smallest := 385,
  use [A_largest, B_largest, C_largest],
  split,
  { exact nat.succ_ne_self 5 }, split,
  { exact nat.succ_ne_self 34 }, split,
  { exact nat.succ_ne_self 10 }, split,
  { have product_largest : A_largest * B_largest * C_largest = 2310 := by norm_num,
    exact product_largest }, split,
  { intros A' B' C' h1 h2 h3 h4,
    have product_largest : A_largest + B_largest + C_largest = 52 := by norm_num,
    sorry }, 
  { intros A'' B'' C'' h1 h2 h3 h4,
    have product_smallest : A_smallest + B_smallest + C_smallest = 390 := by norm_num,
    sorry } }

end olympiad_2002_largest_smallest_sum_l194_194246


namespace yahs_to_bahs_1200_l194_194909

def conversion_equiv : Prop :=
  ∀ (bahs rahs yahs : ℕ),
    (20 * bahs = 30 * rahs) →
    (12 * rahs = 20 * yahs) →
    (1200 * yahs = 480 * bahs)

theorem yahs_to_bahs_1200 : conversion_equiv :=
by
  intros bahs rahs yahs bahs_to_rahs rahs_to_yahs
  have h1 : rahs = (2 / 3 : ℚ) * bahs := by sorry
  have h2 : yahs = (3 / 5 : ℚ) * rahs := by sorry
  have h3 : yahs = (2 / 5 : ℚ) * bahs := by sorry
  have h4 : (2 / 5 : ℚ) * bahs * 1200 = 480 * bahs := by sorry
  exact h4

end yahs_to_bahs_1200_l194_194909


namespace minimize_f_a_n_distance_l194_194552

noncomputable def f (x : ℝ) : ℝ :=
  2^x + Real.log x

noncomputable def a (n : ℕ) : ℝ :=
  0.1 * n

theorem minimize_f_a_n_distance :
  ∃ n : ℕ, n = 110 ∧ ∀ m : ℕ, (m > 0) -> |f (a 110) - 2012| ≤ |f (a m) - 2012| :=
by
  sorry

end minimize_f_a_n_distance_l194_194552


namespace number_of_functions_l194_194808

theorem number_of_functions :
  (∃ g : ℝ → ℝ, ∀ x y z : ℝ, g(xy)^2 + g(xz)^2 - 2 ≥ g(x) * g(yz))
  → ∃ (g1 g2 : ℝ → ℝ), (g1 = fun x => sqrt 2) ∧ (g2 = fun x => -sqrt 2) ∧
    (∃ g, g = g1 ∨ g = g2) ∧
    (∃ n : ℕ, n = 2) :=
by
  sorry

end number_of_functions_l194_194808


namespace num_four_digit_square_palindromes_l194_194190

open Nat

-- Define what it means to be a 4-digit number
def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- The main theorem stating that there are exactly 2 four-digit squares that are palindromes
theorem num_four_digit_square_palindromes : 
  { n : ℕ | is_four_digit n ∧ is_palindrome n ∧ ∃ k : ℕ, k^2 = n ∧ k >= 32 ∧ k <= 99 }.to_finset.card = 2 :=
sorry

end num_four_digit_square_palindromes_l194_194190


namespace team_total_points_l194_194732

theorem team_total_points 
  (n : ℕ)
  (best_score actual : ℕ)
  (desired_avg : ℕ)
  (hypothetical_score : ℕ)
  (current_best_score : ℕ)
  (team_size : ℕ)
  (h1 : team_size = 8)
  (h2 : current_best_score = 85)
  (h3 : hypothetical_score = 92)
  (h4 : desired_avg = 84)
  (h5 : hypothetical_score - current_best_score = 7)
  (h6 : team_size * desired_avg = 672) :
  (actual = 665) :=
sorry

end team_total_points_l194_194732


namespace limit_of_f_at_1_minus_2h_l194_194579

-- Definition of the function and its derivative at 1
def f : ℝ → ℝ := sorry
def f' (x : ℝ) : ℝ := if x = 1 then 1/2 else sorry

-- Hypothesis: the derivative of f at 1 is 1/2
lemma deriv_f_at_1 : f'(1) = 1/2 := by sorry

-- Main statement to be proved
theorem limit_of_f_at_1_minus_2h :
  (∀ f : ℝ → ℝ, (f'(1) = 1/2) → (∃ L : ℝ, (lim (λ h : ℝ, (f (1 - 2 * h) - f 1) / (3 * h)) (𝓝 0) = L) ∧ L = -1/3)) := by 
  sorry

end limit_of_f_at_1_minus_2h_l194_194579


namespace solution_set_inequalities_l194_194921

theorem solution_set_inequalities (a b x : ℝ) (h1 : ∃ x, x > a ∧ x < b) :
  (x < 1 - a ∧ x < 1 - b) ↔ x < 1 - b :=
by
  sorry

end solution_set_inequalities_l194_194921


namespace count_4_digit_palindromic_squares_is_2_l194_194185

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def count_4_digit_palindromic_squares : ℕ :=
  (Finset.range 100).filter (λ n, 32 ≤ n ∧ is_4_digit_number (n * n) ∧ is_palindrome (n * n)).card

theorem count_4_digit_palindromic_squares_is_2 : count_4_digit_palindromic_squares = 2 :=
  sorry

end count_4_digit_palindromic_squares_is_2_l194_194185


namespace largest_value_c_in_range_l194_194087

theorem largest_value_c_in_range :
  ∃ c, (∀ x : ℝ, (x^2 + 2*x * real.cos x + c = 1) → c ≤ 2) ∧
       (∀ c' > 2, ∀ x : ℝ, x^2 + 2*x * real.cos x + c' ≠ 1) :=
sorry

end largest_value_c_in_range_l194_194087


namespace parallelogram_area_l194_194488

theorem parallelogram_area (base height : ℕ) (h_base : base = 26) (h_height : height = 16) : base * height = 416 := 
by
  -- Introduce the base and height according to their given values
  rw [h_base, h_height]
  -- Simplify the multiplication
  exact rfl

end parallelogram_area_l194_194488


namespace four_digit_square_palindromes_are_zero_l194_194181

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

-- Define the main theorem statement
theorem four_digit_square_palindromes_are_zero : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) → 
             is_palindrome n → 
             (∃ m : ℕ, n = m * m) → 
             n = 0 :=
by
  sorry

end four_digit_square_palindromes_are_zero_l194_194181


namespace sum_evaluation_p_q_r_sum_l194_194481

theorem sum_evaluation :
  (∑ n in Finset.range 10000, 1 / Real.sqrt (n + 2 * Real.sqrt (n^2 - 1))) = 49 + 70 * Real.sqrt 2 := sorry

theorem p_q_r_sum :
  let p := 49
  let q := 70
  let r := 2
  p + q + r = 121 := by
    simp [p, q, r]
    norm_num

end sum_evaluation_p_q_r_sum_l194_194481


namespace strictly_decreasing_interval_l194_194086

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

theorem strictly_decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 3 → f'(x) < 0 :=
by
  sorry

end strictly_decreasing_interval_l194_194086


namespace basketball_free_throws_l194_194347

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : b = a - 2)
  (h3 : 2 * a + 3 * b + x = 68) : x = 44 :=
by
  sorry

end basketball_free_throws_l194_194347


namespace find_sin_2θ_plus_pi_over_2_l194_194543

noncomputable def angle_vertex_origin (θ : ℝ) : Prop :=
  -- Vertex of angle θ is at origin
  true

noncomputable def initial_side_positive_x_axis (θ : ℝ) : Prop :=
  -- Initial side coincides with positive x-axis
  true

noncomputable def point_on_terminal_side (θ : ℝ) : Prop :=
  -- Point A(3, -4) is on the terminal side of θ
  (3, -4) ∈ set_of (λ P:ℝ × ℝ, ∃ (r:ℝ), P = (r * Math.cos θ, r * Math.sin θ))

theorem find_sin_2θ_plus_pi_over_2 (θ : ℝ) 
  (h1 : angle_vertex_origin θ) 
  (h2 : initial_side_positive_x_axis θ) 
  (h3 : point_on_terminal_side θ) : 
  sin (2 * θ + π / 2) = -7 / 25 := 
sorry

end find_sin_2θ_plus_pi_over_2_l194_194543


namespace distance_home_to_school_l194_194396

def speed_walk := 5
def speed_car := 15
def time_difference := 2

variable (d : ℝ) -- distance from home to school
variable (T1 T2 : ℝ) -- T1: time to school, T2: time back home

-- Conditions
axiom h1 : T1 = d / speed_walk / 2 + d / speed_car / 2
axiom h2 : d = speed_car * T2 / 3 + speed_walk * 2 * T2 / 3
axiom h3 : T1 = T2 + time_difference

-- Theorem to prove
theorem distance_home_to_school : d = 150 :=
by
  sorry

end distance_home_to_school_l194_194396


namespace log2_of_denominator_eq_409_l194_194704

-- Definitions: total number of games, possible outcomes, and probability
noncomputable def num_teams : ℕ := 30
noncomputable def total_pairings : ℕ := num_teams * (num_teams - 1) / 2
noncomputable def total_outcomes : ℕ := 2 ^ total_pairings
noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def unique_wins_probability : ℚ :=
  (factorial num_teams : ℚ) / total_outcomes

-- The main theorem to prove
theorem log2_of_denominator_eq_409 :
  ∃ m n : ℕ, unique_wins_probability = m / n ∧ n = 2 ^ 409 ∧ Nat.gcd m n = 1 :=
by
  sorry

end log2_of_denominator_eq_409_l194_194704


namespace range_of_a_l194_194156

def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (2 - a) * x + 3 * a 
  else Real.log x / Real.log 2

theorem range_of_a {a : ℝ} :
  (-1 ≤ a ∧ a < 2) ↔ ∀ x y : ℝ, (x < 1 → y = (2 - a) * x + 3 * a) ∨ (x ≥ 1 → y = Real.log x / Real.log 2) ->
  (∀ y, ∃ x, y = f a x) :=
sorry

end range_of_a_l194_194156


namespace sum_of_digits_of_greatest_prime_divisor_of_16385_is_19_l194_194391

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_19 :
  let n := 16385
  let p := 3277
  let prime_p : Prime p := by sorry
  let greatest_prime_divisor := p
  let sum_digits := 3 + 2 + 7 + 7
  sum_digits = 19 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_16385_is_19_l194_194391


namespace compute_expr1_factorize_expr2_l194_194406

-- Definition for Condition 1: None explicitly stated.

-- Theorem for Question 1
theorem compute_expr1 (y : ℝ) : (y - 1) * (y + 5) = y^2 + 4*y - 5 :=
by sorry

-- Definition for Condition 2: None explicitly stated.

-- Theorem for Question 2
theorem factorize_expr2 (x y : ℝ) : -x^2 + 4*x*y - 4*y^2 = -((x - 2*y)^2) :=
by sorry

end compute_expr1_factorize_expr2_l194_194406


namespace num_4_digit_palindromic_squares_l194_194202

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_n (n : ℕ) : Prop :=
  32 ≤ n ∧ n ≤ 99

theorem num_4_digit_palindromic_squares : ∃ (count : ℕ), count = 3 ∧ ∀ n, valid_n n → is_4_digit (n^2) → is_palindrome (n^2) :=
sorry

end num_4_digit_palindromic_squares_l194_194202


namespace range_g_l194_194828

def g (x : ℝ) : ℝ :=
if x ≠ -5 then (3 * (x - 4)) else 27

theorem range_g : {y : ℝ | ∃ x : ℝ, x ≠ -5 ∧ y = g x } = {y : ℝ | y ≠ -27} :=
by sorry

end range_g_l194_194828


namespace cards_dealt_l194_194222

theorem cards_dealt (cards people : ℕ) (h_cards : cards = 72) (h_people : people = 10) :
  let quotient := cards / people in
  let remainder := cards % people in
  (people - remainder) = 8 :=
by
  sorry

end cards_dealt_l194_194222


namespace minimize_relative_waiting_time_l194_194010

-- Definitions of task times in seconds
def task_U : ℕ := 10
def task_V : ℕ := 120
def task_W : ℕ := 900

-- Definition of relative waiting time given a sequence of task execution times
def relative_waiting_time (times : List ℕ) : ℚ :=
  (times.head! : ℚ) / (times.head! : ℚ) + 
  (times.head! + times.tail.head! : ℚ) / (times.tail.head! : ℚ) + 
  (times.head! + times.tail.head! + times.tail.tail.head! : ℚ) / (times.tail.tail.head! : ℚ)

-- Sequences
def sequence_A : List ℕ := [task_U, task_V, task_W]
def sequence_B : List ℕ := [task_V, task_W, task_U]
def sequence_C : List ℕ := [task_W, task_U, task_V]
def sequence_D : List ℕ := [task_U, task_W, task_V]

-- Sum of relative waiting times for each sequence
def S_A := relative_waiting_time sequence_A
def S_B := relative_waiting_time sequence_B
def S_C := relative_waiting_time sequence_C
def S_D := relative_waiting_time sequence_D

-- Theorem to prove that sequence A has the minimum sum of relative waiting times
theorem minimize_relative_waiting_time : S_A < S_B ∧ S_A < S_C ∧ S_A < S_D := 
  by sorry

end minimize_relative_waiting_time_l194_194010


namespace length_KL_l194_194739

-- Define the conditions
variables (G H I J K L : Type) [has_length G H I J K L] (HI GH JK KL : ℝ)

-- Condition 1: Similar triangles
axiom similar_triangles (G H I J K L : Type) : Triangle_Similar G H I J K L
  
-- Condition 2: Lengths of the sides
axiom length_HI : HI = 10
axiom length_GH : GH = 7
axiom length_JK : JK = 4

-- Prove that the length of KL is approximately 5.7 cm
theorem length_KL : KL ≈ 5.7 :=
  sorry

end length_KL_l194_194739


namespace perp_lines_l194_194934

universe u
variables {A B C D O P : Type u}
variables [ConvexQuadrilateral A B C D]
variable (Midpoint B C O)
variable (Midpoint A P O)
variable [Perpendicular BD AB]
variable [Perpendicular AC CD]

theorem perp_lines (h1: ConvexQuadrilateral A B C D)
                   (h2: Midpoint B C O)
                   (h3: Midpoint A P O)
                   (h4: Perpendicular BD AB)
                   (h5: Perpendicular AC CD) : 
                   Perpendicular BC DP :=
sorry

end perp_lines_l194_194934


namespace min_diagonal_sum_l194_194446

-- Definitions based on conditions
def is_valid_8x8_array (A : array (fin 8) (array (fin 8) ℕ)) : Prop :=
  (∀ i j : fin 8, 1 ≤ A[i][j] ∧ A[i][j] ≤ 64) ∧
  (∀ i j : fin 8, (i < 7 ∨ j < 7) → abs (A[i][j] - A[i + 1][j]) = 1 ∨ abs (A[i][j] - A[i][j + 1]) = 1) ∧
  (A.map (λ row => row.filter (λ x => x)).flatten = list.range' 1 64)

-- Main proof statement
theorem min_diagonal_sum (A : array (fin 8) (array (fin 8) ℕ)) 
  (h : is_valid_8x8_array A) :
  (fin 8).sum (λ i => A[i][i]) = 64 :=
sorry

end min_diagonal_sum_l194_194446


namespace sum_of_real_solutions_eq_two_l194_194475

noncomputable def sum_of_real_solutions : ℝ :=
  let solutions := {x : ℝ | |x + 3| + |x - 3| = 3 * |x - 1|}
  ∑ x in solutions, x

theorem sum_of_real_solutions_eq_two : sum_of_real_solutions = 2 := by sorry

end sum_of_real_solutions_eq_two_l194_194475


namespace symmetry_axis_is_2_range_of_a_l194_194518

-- Definitions given in the conditions
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition 1: Constants a, b, c and a ≠ 0
variables (a b c : ℝ) (a_ne_zero : a ≠ 0)

-- Condition 2: Inequality constraint
axiom inequality_constraint : a^2 + 2 * a * c + c^2 < b^2

-- Condition 3: y-values are the same when x=t+2 and x=-t+2
axiom y_symmetry (t : ℝ) : quadratic_function a b c (t + 2) = quadratic_function a b c (-t + 2)

-- Question 1: Proving the symmetry axis is x=2
theorem symmetry_axis_is_2 : ∀ t : ℝ, (t + 2 + (-t + 2)) / 2 = 2 :=
by sorry

-- Question 2: Proving the range of a if y=2 when x=-2
theorem range_of_a (h : quadratic_function a b c (-2) = 2) (b_eq_neg4a : b = -4 * a) : 2 / 15 < a ∧ a < 2 / 7 :=
by sorry

end symmetry_axis_is_2_range_of_a_l194_194518


namespace cycles_and_tree_l194_194121

/-- This statement defines the problem conditions and the assertion that r + s = n -/
theorem cycles_and_tree {G : Graph} (n r s : ℕ)
  (connected : G.Connected)
  (edges_count : G.edges.card = n)
  (no_parallel_edges : ¬∃ (e1 e2 : G.edges), e1 ≠ e2 ∧ e1.verts = e2.verts)
  (outer_cycle : ∀ (C C' : set G.edges), is_cycle C → is_cycle C' → 
    G.outer_cycle C C' = {x | x ∈ (C \ C') ∪ (C' \ C)})
  (r_condition : ∃ (C1 C2 ... Cr : set G.edges), ∀ (1 ≤ k ≤ r) (1 ≤ i, j1, j2, ... jk ≤ r),
    ¬(C_i = C_{j1} * C_{j2} * ... * C_{jk} ∧ (j1 ≠ i ∨ k ≠ 1)))
  (s_condition : ∃ (E1 E2 ... Es : G.edges), ∀ (1 ≤ i ≤ s), ¬is_cycle ({E_i}))
  : r + s = n := sorry

end cycles_and_tree_l194_194121


namespace property_A_property_B_property_D_l194_194775

-- Define the function y = (x - 1)^2
def f (x : ℝ) : ℝ := (x - 1) ^ 2

-- Property A: For all x in ℝ, f(1 + x) = f(1 - x)
theorem property_A : ∀ x : ℝ, f(1 + x) = f(1 - x) :=
by sorry

-- Property B: The function is decreasing on (-∞, 0]
theorem property_B : ∀ x y : ℝ, x < y → y ≤ 0 → f x > f y :=
by sorry

-- Property C is not included since it is not supposed to be satisfied.

-- Property D: f(0) is not the minimum value of the function
theorem property_D : ∃ x : ℝ, f x < f 0 :=
by sorry

end property_A_property_B_property_D_l194_194775


namespace general_formula_an_range_k_l194_194519

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {b : ℕ → ℝ} {c : ℕ → ℝ} {T : ℕ → ℝ}

-- Condition
def condition1 (n : ℕ) : Prop := S n = 2 * a n - 2

-- General formula for the sequence {a_n}
theorem general_formula_an (ha : ∀ n, S n = 2 * a n - 2) : ∀ n, a n = 2^n :=
sorry

-- Definitions for bn and cn based on an
def b_n (n : ℕ) := Real.log (a n) / Real.log 2
def c_n (n : ℕ) := 1 / (b_n n * b_n (n + 1))

-- Sum of first n terms of the sequence {c_n}
def sum_c_n (n : ℕ) := ∑ i in Finset.range n, c_n i

-- Range of values for k
theorem range_k (ha : ∀ n, a n = 2^n) (hb : ∀ n, b n = Real.log (a n) / Real.log 2)
  (hc : ∀ n, c n = 1 / (b n * b (n + 1))) (hT : ∀ n : ℕ, sum_c_n n ≤ k * (n + 4)) :
  k ≥ 1 / 9 :=
sorry

end general_formula_an_range_k_l194_194519


namespace nonneg_int_solutions_l194_194091

theorem nonneg_int_solutions (a b : ℕ) (h : abs (a - b) + a * b = 1) :
  (a, b) = (1, 0) ∨ (a, b) = (0, 1) ∨ (a, b) = (1, 1) :=
by
  sorry

end nonneg_int_solutions_l194_194091


namespace arithmetic_sequence_problem_l194_194128

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 9 = 81)
  (h3 : a (k - 4) = 191)
  (h4 : S k = 10000) :
  k = 100 :=
by
  sorry

end arithmetic_sequence_problem_l194_194128


namespace total_books_correct_l194_194369

-- Definitions based on the conditions
def num_books_bottom_shelf (T : ℕ) := T / 3
def num_books_middle_shelf (T : ℕ) := T / 4
def num_books_top_shelf : ℕ := 30
def total_books (T : ℕ) := num_books_bottom_shelf T + num_books_middle_shelf T + num_books_top_shelf

theorem total_books_correct : total_books 72 = 72 :=
by
  sorry

end total_books_correct_l194_194369


namespace num_4_digit_palindromic_squares_l194_194204

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_n (n : ℕ) : Prop :=
  32 ≤ n ∧ n ≤ 99

theorem num_4_digit_palindromic_squares : ∃ (count : ℕ), count = 3 ∧ ∀ n, valid_n n → is_4_digit (n^2) → is_palindrome (n^2) :=
sorry

end num_4_digit_palindromic_squares_l194_194204


namespace max_a_l194_194860

theorem max_a {a b c d : ℤ} (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 :=
by {
  have h_b : b ≤ 3 * c - 1 := by linarith,
  have h_c : c ≤ 4 * d - 1 := by linarith,
  have h_d : d ≤ 99 := by linarith,
  have h_max_a := calc
    a ≤ 2 * b - 1 : by linarith
    ... ≤ 2 * (3 * c - 1) - 1 : by linarith
    ... ≤ 6 * c - 3 : by linarith
    ... ≤ 6 * (4 * d - 1) - 3 : by linarith
    ... ≤ 24 * d - 9 : by linarith
    ... ≤ 24 * 99 - 9 : by linarith
    ... = 2367 : by norm_num,
  exact h_max_a,
  sorry
}

end max_a_l194_194860


namespace decryption_proof_l194_194370

-- Definitions
def Original_Message := "МОСКВА"
def Encrypted_Text_1 := "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ"
def Encrypted_Text_2 := "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП"
def Encrypted_Text_3 := "РТПАИОМВСВТИЕОБПРОЕННИИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК"

noncomputable def Encrypted_Message_1 := "ЙМЫВОТСЬЛКЪГВЦАЯЯ"
noncomputable def Encrypted_Message_2 := "УКМАПОЧСРКЩВЗАХ"
noncomputable def Encrypted_Message_3 := "ШМФЭОГЧСЙЪКФЬВЫЕАКК"

def Decrypted_Message_1_and_3 := "ПОВТОРЕНИЕМАТЬУЧЕНИЯ"
def Decrypted_Message_2 := "СМОТРИВКОРЕНЬ"

-- Theorem statement
theorem decryption_proof :
  (Encrypted_Text_1 = Encrypted_Text_3 ∧ Original_Message = "МОСКВА" ∧ Encrypted_Message_1 = Encrypted_Message_3) →
  (Decrypted_Message_1_and_3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ Decrypted_Message_2 = "СМОТРИВКОРЕНЬ") :=
by 
  sorry

end decryption_proof_l194_194370


namespace count_4_digit_palindromic_squares_is_2_l194_194186

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def count_4_digit_palindromic_squares : ℕ :=
  (Finset.range 100).filter (λ n, 32 ≤ n ∧ is_4_digit_number (n * n) ∧ is_palindrome (n * n)).card

theorem count_4_digit_palindromic_squares_is_2 : count_4_digit_palindromic_squares = 2 :=
  sorry

end count_4_digit_palindromic_squares_is_2_l194_194186


namespace not_all_acute_angled_triangles_l194_194847

theorem not_all_acute_angled_triangles (A B C D : Point)
  (h1 : ¬(Collinear A B C))
  (h2 : ¬(Collinear A B D))
  (h3 : ¬(Collinear A C D))
  (h4 : ¬(Collinear B C D))
  : ∃ (P Q R : Point), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ Angle.is_not_acute (angle P Q R) := 
sorry

end not_all_acute_angled_triangles_l194_194847


namespace average_percent_score_is_65_point_25_l194_194758

theorem average_percent_score_is_65_point_25 :
  let percent_score : List (ℕ × ℕ) := [(95, 10), (85, 20), (75, 40), (65, 50), (55, 60), (45, 15), (35, 5)]
  let total_students : ℕ := 200
  let total_score : ℕ := percent_score.foldl (fun acc p => acc + p.1 * p.2) 0
  (total_score : ℚ) / (total_students : ℚ) = 65.25 := by
{
  sorry
}

end average_percent_score_is_65_point_25_l194_194758


namespace gcf_180_270_450_l194_194385

theorem gcf_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  sorry

end gcf_180_270_450_l194_194385


namespace darnel_jog_laps_l194_194470

theorem darnel_jog_laps (x : ℝ) (h1 : 0.88 = x + 0.13) : x = 0.75 := by
  sorry

end darnel_jog_laps_l194_194470


namespace perpendicular_vectors_k_equals_one_l194_194141

/-- Non-collinear unit vectors a and b,
    and a real number k, if a + b is perpendicular to k * a - b,
    then k = 1. -/
theorem perpendicular_vectors_k_equals_one
  (a b : ℝ^3)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (h_neq : a ≠ b)
  (k : ℝ)
  (h_perp : (a + b) ⬝ (k • a - b) = 0) : k = 1 :=
sorry

end perpendicular_vectors_k_equals_one_l194_194141


namespace monotonicity_of_even_function_l194_194902

-- Define the function and its properties
def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + 2*m*x + 3

-- A function is even if f(x) = f(-x) for all x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

-- The main theorem statement
theorem monotonicity_of_even_function :
  ∀ (m : ℝ), is_even (f m) → (f 0 = 3) ∧ (∀ x : ℝ, f 0 x = - x^2 + 3) →
  (∀ a b, -3 < a ∧ a < b ∧ b < 1 → f 0 a < f 0 b → f 0 b > f 0 a) :=
by
  intro m
  intro h
  intro H
  sorry

end monotonicity_of_even_function_l194_194902


namespace sum_of_even_powers_coefficients_l194_194628

theorem sum_of_even_powers_coefficients (n : ℕ) : 
  let f := (1 + (λ x, x^2) + (λ x, x^4)) in
  let expanded := f ^ n in
  let s := ∑ i in (finset.range (4 * n + 1)).filter (λ i, i % 4 = 0), polynomial.coeff expanded i in
  s = 3 ^ n :=
begin
  sorry
end

end sum_of_even_powers_coefficients_l194_194628


namespace complex_div_imaginary_unit_eq_l194_194623

theorem complex_div_imaginary_unit_eq :
  (∀ i : ℂ, i^2 = -1 → (1 / (1 + i)) = ((1 - i) / 2)) :=
by
  intro i
  intro hi
  /- The proof will be inserted here -/
  sorry

end complex_div_imaginary_unit_eq_l194_194623


namespace roots_and_angle_l194_194550

theorem roots_and_angle (α : ℝ) (x : ℂ) :
  (0 < α ∧ α < 2 * Real.pi) →
  (∃ a, x^4 + (1 / 3) * Real.sin α * x^2 + (1 / 200) * Real.cos (Real.pi / 3) = 0) →
  let roots := [(-3) * a, (-1) * a, a, 3 * a] in
  Let sin_vals := [Real.sin (Real.pi / 6), Real.sin (5 * Real.pi / 6), Real.sin (7 * Real.pi / 6), Real.sin (11 * Real.pi / 6)] in
  let cos_val := Real.cos (Real.pi /3) / 200 in
  roots.all (λ x, x^4 + (1 / 3)::((λ b : ℝ, x^2 :=b) + cos_val = 0) ↔ α = Real.pi / 6 ∨ α = 5 * Real.pi / 6 ∨ α = 7 * Real.pi / 6 ∨ α = 11π / 6 :=
sorry

end roots_and_angle_l194_194550


namespace isosceles_triangle_perimeter_l194_194237

-- Define the isosceles triangle with given side lengths
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

-- Conditions of the problem
axiom side1 : ℕ := 6
axiom side2 : ℕ := 14
axiom is_isosceles : is_isosceles_triangle side1 side2 14

-- Prove that the perimeter of the triangle is 34
theorem isosceles_triangle_perimeter : side1 + side2 + 14 = 34 :=
by {
  -- Here, proof steps would normally go, but they are omitted as per the instructions
  sorry
}

end isosceles_triangle_perimeter_l194_194237


namespace quadrilateral_area_correct_l194_194489

noncomputable def quadrilateral_area (d : ℝ) (h1 : ℝ) (h2 : ℝ) : ℝ :=
  let area1 := 1 / 2 * d * h1
  let area2 := 1 / 2 * d * h2
  area1 + area2

theorem quadrilateral_area_correct :
  quadrilateral_area 30 10 6 = 240 :=
by
  unfold quadrilateral_area
  norm_num
  sorry

end quadrilateral_area_correct_l194_194489


namespace trig_identity_l194_194699

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem trig_identity (θ : ℝ) : sin (θ + 75 * Real.pi / 180) + cos (θ + 45 * Real.pi / 180) - Real.sqrt 3 * cos (θ + 15 * Real.pi / 180) = 0 :=
by
  sorry

end trig_identity_l194_194699


namespace square_distance_from_B_to_center_l194_194019

-- Defining the conditions
structure Circle (α : Type _) :=
(center : α × α)
(radius2 : ℝ)

structure Point (α : Type _) :=
(x : α)
(y : α)

def is_right_angle (a b c : Point ℝ) : Prop :=
(b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) = 0

noncomputable def distance2 (p1 p2 : Point ℝ) : ℝ :=
(p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

theorem square_distance_from_B_to_center :
  ∀ (c : Circle ℝ) (A B C : Point ℝ), 
    c.radius2 = 65 →
    distance2 A B = 49 →
    distance2 B C = 9 →
    is_right_angle A B C →
    distance2 B {x:=0, y:=0} = 80 := 
by
  intros c A B C h_radius h_AB h_BC h_right_angle
  sorry

end square_distance_from_B_to_center_l194_194019


namespace conjugate_Z_l194_194545

noncomputable def Z : ℂ := complex.I * (1 - complex.I)
def Z_conj := 1 - complex.I

theorem conjugate_Z :
  complex.conj Z = Z_conj := sorry

end conjugate_Z_l194_194545


namespace max_sine_product_l194_194956

theorem max_sine_product (A B C : ℝ) (h_sum : A + B + C = Real.pi)
  (h_A_pos : 0 < sin (A / 2)) (h_B_pos : 0 < sin (B / 2)) (h_C_pos : 0 < sin (C / 2)) :
  (sin (A / 2) * sin (B / 2) * sin (C / 2)) ≤ (1 / 8) :=
sorry

end max_sine_product_l194_194956


namespace specificTriangle_perimeter_l194_194525

-- Assume a type to represent triangle sides
structure IsoscelesTriangle (a b : ℕ) : Prop :=
  (equal_sides : a = b ∨ a + b > max a b)

-- Define the condition where we have specific sides
def specificTriangle : Prop :=
  IsoscelesTriangle 5 2

-- Prove that given the specific sides, the perimeter is 12
theorem specificTriangle_perimeter : specificTriangle → 5 + 5 + 2 = 12 :=
by
  intro h
  cases h
  sorry

end specificTriangle_perimeter_l194_194525


namespace intersection_A_B_l194_194167

-- Define the sets A and B
def A : Set ℤ := {−2, −1, 0, 1}
def B : Set ℤ := {y | y ≠ 0}

-- The theorem we need to prove
theorem intersection_A_B : A ∩ B = {−2, −1, 1} :=
by sorry

end intersection_A_B_l194_194167


namespace ellipse_focus_coordinates_l194_194455

def is_ellipse_major_axis (P Q : ℝ × ℝ) : Prop :=
  P = (1, 0) ∧ Q = (5, 0)

def is_ellipse_minor_axis (P Q : ℝ × ℝ) : Prop :=
  P = (3, 3) ∧ Q = (3, -3)

def ellipse_center (P Q : ℝ × ℝ) : (ℝ × ℝ) :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def ellipse_focus_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_focus_coordinates {P Q R S : ℝ × ℝ} (c : ℝ × ℝ) (f_d : ℝ) :
  is_ellipse_major_axis P Q →
  is_ellipse_minor_axis R S →
  c = ellipse_center P Q →
  f_d = ellipse_focus_distance ((R.2 - S.2) / 2) ((Q.1 - P.1) / 2) →
  (c.1, c.2 + f_d) = (3, Real.sqrt 5) :=
by
  intros h_major h_minor hc hf_d
  rw [h_major, h_minor] at hc
  have h_center : c = (3, 0), by simp [ellipse_center, h_major]
  rw h_center at hf_d
  simp [ellipse_focus_distance, h_center] at hf_d
  exact sorry

end ellipse_focus_coordinates_l194_194455


namespace incenter_sum_equals_one_l194_194785

noncomputable def incenter (A B C : Point) : Point := sorry -- Definition goes here

def side_length (A B C : Point) (a b c : ℝ) : Prop :=
  -- Definitions relating to side lengths go here
  sorry

theorem incenter_sum_equals_one (A B C I : Point) (a b c IA IB IC : ℝ) (h_incenter : I = incenter A B C)
    (h_sides : side_length A B C a b c) :
    (IA ^ 2 / (b * c)) + (IB ^ 2 / (a * c)) + (IC ^ 2 / (a * b)) = 1 :=
  sorry

end incenter_sum_equals_one_l194_194785


namespace dominoes_per_player_l194_194962

-- Define the conditions
def total_dominoes : ℕ := 28
def number_of_players : ℕ := 4

-- The theorem
theorem dominoes_per_player : total_dominoes / number_of_players = 7 :=
by sorry

end dominoes_per_player_l194_194962


namespace baron_not_lying_l194_194043

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem baron_not_lying : 
  ∃ a b : ℕ, 
  (a ≠ b ∧ a ≥ 10^9 ∧ a < 10^10 ∧ b ≥ 10^9 ∧ b < 10^10 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a + sum_of_digits (a * a) = b + sum_of_digits (b * b))) :=
  sorry

end baron_not_lying_l194_194043


namespace law_of_sines_example_l194_194923

-- Let's define the given conditions and the problem to be proved in Lean.
theorem law_of_sines_example
  (A : ℝ) (a : ℝ) (b : ℝ) (B : ℝ)
  (hA : A = π / 6)
  (ha : a = sqrt 2) :
  b / sin B = 2 * sqrt 2 :=
by
  -- The proof would go here, but as per instructions, we use sorry.
  sorry

end law_of_sines_example_l194_194923


namespace find_x_l194_194745

theorem find_x (x : ℕ) (h1 : x ≥ 10) (h2 : x > 8) : x = 9 := by
  sorry

end find_x_l194_194745


namespace definite_integral_abs_x_squared_minus_4_eq_11_over_3_l194_194047

noncomputable def integral_abs_x_squared_minus_4 : ℝ :=
  ∫ x in 0..1, |x^2 - 4|

theorem definite_integral_abs_x_squared_minus_4_eq_11_over_3 :
  integral_abs_x_squared_minus_4 = 11/3 :=
by sorry

end definite_integral_abs_x_squared_minus_4_eq_11_over_3_l194_194047


namespace mass_percentage_O_in_CaO_l194_194821

theorem mass_percentage_O_in_CaO :
  let molar_mass_Ca := 40.08
  let molar_mass_O := 16.00
  let molar_mass_CaO := molar_mass_Ca + molar_mass_O
  let mass_percentage_O := (molar_mass_O / molar_mass_CaO) * 100
  mass_percentage_O = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l194_194821


namespace hyperbola_with_same_vertices_and_eccentricity_l194_194872

theorem hyperbola_with_same_vertices_and_eccentricity (e : ℝ) (a1 a2 b1 b2 : ℝ) :
  e = 2 ∧
  ((a1 = 4 ∧ b1 = sqrt (e^2 * a1^2 - a1^2)) ∨ (a2 = 3 ∧ b2 = sqrt (e^2 * a2^2 - a2^2))) →
  (∃ (a1 b1 : ℝ), ∀ x y : ℝ, (x^2 / a1^2 - y^2 / b1^2 = 1)) ∨
  (∃ (a2 b2 : ℝ), ∀ x y : ℝ, (y^2 / a2^2 - x^2 / b2^2 = 1)) :=
by
  sorry

end hyperbola_with_same_vertices_and_eccentricity_l194_194872


namespace usual_time_to_office_l194_194382

theorem usual_time_to_office (S T : ℝ) (h : T = 4 / 3 * (T + 8)) : T = 24 :=
by
  sorry

end usual_time_to_office_l194_194382


namespace domain_combination_l194_194875

variable {R : Type} [LinearOrder R] [OrderedSub R] [OrderedAdd R] [OrderedSemiring R] [OrderedCommRing R] [one_lt R]

def domain_f (x : R) : Prop := 0 ≤ x ∧ x ≤ 4
def domain_f_2x (x : R) : Prop := 0 ≤ 2 * x ∧ 2 * x ≤ 4
def domain_ln (x : R) : Prop := 1 < x

theorem domain_combination :
  ∀ (x : R), domain_f_2x x ∧ domain_ln x ↔ 1 < x ∧ x ≤ 2 := by
  sorry

end domain_combination_l194_194875


namespace example_theorem_l194_194278

-- Let p and q be real numbers. The roots of the polynomial x^3 - 6x^2 + px - q are distinct positive integers 1, 2, 3.
-- We need to prove that p + q = 17.
theorem example_theorem (p q : ℝ) (h: ∀ r : ℝ, r ∈ {1, 2, 3} ∧ (r^3 - 6*r^2 + p*r - q = 0)) :
    p + q = 17 := 
  sorry

end example_theorem_l194_194278


namespace tables_count_l194_194015

def total_tables (four_legged_tables three_legged_tables : Nat) : Nat :=
  four_legged_tables + three_legged_tables

theorem tables_count
  (four_legged_tables three_legged_tables : Nat)
  (total_legs : Nat)
  (h1 : four_legged_tables = 16)
  (h2 : total_legs = 124)
  (h3 : 4 * four_legged_tables + 3 * three_legged_tables = total_legs) :
  total_tables four_legged_tables three_legged_tables = 36 :=
by
  sorry

end tables_count_l194_194015


namespace minimum_overall_average_price_l194_194766

theorem minimum_overall_average_price 
  (shirts_sold : ℕ) (first_three_prices : list ℕ) 
  (remaining_average_price : ℕ → ℕ) :
  shirts_sold = 10 →
  first_three_prices = [20, 22, 25] →
  (∀ n, n = 7 → remaining_average_price n = 19) →
  ∃ X, X ≥ 20 :=
by
  sorry

end minimum_overall_average_price_l194_194766


namespace nonneg_integer_solutions_l194_194664

theorem nonneg_integer_solutions : 
  {x : ℤ // 0 ≤ x ∧ x < 3} = {0, 1, 2} :=
by
  sorry

end nonneg_integer_solutions_l194_194664


namespace min_w_for_factors_l194_194007

theorem min_w_for_factors (w : ℕ) (h_pos : w > 0)
  (h_product_factors : ∀ k, k > 0 → ∃ a b : ℕ, (1452 * w = k) → (a = 3^3) ∧ (b = 13^3) ∧ (k % a = 0) ∧ (k % b = 0)) : 
  w = 19773 :=
sorry

end min_w_for_factors_l194_194007


namespace quadrilateral_area_l194_194819

theorem quadrilateral_area :
  let line1 (P : ℝ × ℝ) := 3 * P.1 + 4 * P.2 - 12 = 0,
      line2 (P : ℝ × ℝ) := 6 * P.1 - 4 * P.2 - 12 = 0,
      line3 (P : ℝ × ℝ) := P.1 = 3,
      line4 (P : ℝ × ℝ) := P.2 = 1 in
  let points := [{ x := 8/3, y := 1 }, { x := 3, y := 3/4 }, { x := 3, y := 3/2 }, { x := 4/3, y := 2 }] in
  let area := |(8/3 * 3/4 + 3 * 3/2 + 3 * 2 + 4/3 * 1) - (1 * 3 + 3/4 * 3 + 3/2 * 4/3 + 2 * 8/3)| / 2 in
  area = 0.5 := 
sorry

end quadrilateral_area_l194_194819


namespace symmetry_axis_of_f_find_side_b_l194_194836

noncomputable def m (x : ℝ) : ℝ × ℝ := (1/2 * real.sin x, sqrt 3/2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (real.cos x, real.cos x ^ 2 - 1/2)
noncomputable def f (x : ℝ) := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem symmetry_axis_of_f (x : ℝ) (k : ℤ) :
  f x = (1/2) * real.sin (2 * x + (π / 3)) → x = 1/2 * k * π + π/12 := sorry

theorem find_side_b (A B : ℝ) (a b : ℝ) :
  f A = 0 ∧ real.sin B = 4/5 ∧ a = sqrt 3 →
  A = π/3 → b = 8/5 := sorry

end symmetry_axis_of_f_find_side_b_l194_194836


namespace suitable_for_systematic_sampling_l194_194033

def city_districts : ℕ := 2000
def student_ratio : List ℕ := [3, 2, 8, 2]
def sample_size_city : ℕ := 200
def total_components : ℕ := 2000

def condition_A : Prop := 
  city_districts = 2000 ∧ 
  student_ratio = [3, 2, 8, 2] ∧ 
  sample_size_city = 200

def condition_B : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 5

def condition_C : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 200

def condition_D : Prop := 
  ∃ (n : ℕ), n = 20 ∧ n = 5

theorem suitable_for_systematic_sampling : condition_C :=
by
  sorry

end suitable_for_systematic_sampling_l194_194033


namespace molecular_weight_of_N2O5_l194_194090

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_atoms_N : ℕ := 2
def num_atoms_O : ℕ := 5
def molecular_weight_N2O5 : ℝ := (num_atoms_N * atomic_weight_N) + (num_atoms_O * atomic_weight_O)

theorem molecular_weight_of_N2O5 : molecular_weight_N2O5 = 108.02 :=
by
  sorry

end molecular_weight_of_N2O5_l194_194090


namespace julieta_total_spent_l194_194253

theorem julieta_total_spent (original_backpack_price : ℕ)
                            (original_ringbinder_price : ℕ)
                            (backpack_price_increase : ℕ)
                            (ringbinder_price_decrease : ℕ)
                            (number_of_ringbinders : ℕ)
                            (new_backpack_price : ℕ)
                            (new_ringbinder_price : ℕ)
                            (total_ringbinder_cost : ℕ)
                            (total_spent : ℕ) :
  original_backpack_price = 50 →
  original_ringbinder_price = 20 →
  backpack_price_increase = 5 →
  ringbinder_price_decrease = 2 →
  number_of_ringbinders = 3 →
  new_backpack_price = original_backpack_price + backpack_price_increase →
  new_ringbinder_price = original_ringbinder_price - ringbinder_price_decrease →
  total_ringbinder_cost = new_ringbinder_price * number_of_ringbinders →
  total_spent = new_backpack_price + total_ringbinder_cost →
  total_spent = 109 := by
  intros
  sorry

end julieta_total_spent_l194_194253


namespace tomatoes_grown_l194_194448

theorem tomatoes_grown (x y : ℕ) 
    (h_last_year : x^2)
    (h_this_year : y^2) 
    (h_diff : y^2 = x^2 + 131) : 
    y^2 = 4356 := 
sorry

end tomatoes_grown_l194_194448


namespace average_speed_is_202_l194_194652

-- Define the initial and final odometer readings and times driven
def initial_odometer : ℤ := 12321
def final_odometer : ℤ := 14741
def time_day1 : ℕ := 5
def time_day2 : ℕ := 7

-- Define the total distance and total time
def total_distance : ℕ := final_odometer - initial_odometer
def total_time : ℕ := time_day1 + time_day2

-- Define the average speed (as a rational number to avoid fractional issues)
def average_speed : ℚ := (total_distance : ℚ) / (total_time : ℚ)

-- The theorem we need to prove
theorem average_speed_is_202 : average_speed ≈ 202 := sorry

end average_speed_is_202_l194_194652


namespace solve_for_y_l194_194549

theorem solve_for_y : (400 + 2 * 20 * 5 + 25 = y) → y = 625 :=
by
  intro h
  rw [mul_assoc, mul_assoc 2 20 5, add_assoc 400, add_assoc (400 + 2 * 20 * 5), ← add_assoc, add_comm 25, add_assoc 400]
  sorry

end solve_for_y_l194_194549


namespace train_speed_solution_l194_194031

def train_speed_problem (L v : ℝ) (man_time platform_time : ℝ) (platform_length : ℝ) :=
  man_time = 12 ∧
  platform_time = 30 ∧
  platform_length = 180 ∧
  L = v * man_time ∧
  (L + platform_length) = v * platform_time

theorem train_speed_solution (L v : ℝ) (h : train_speed_problem L v 12 30 180) :
  v * 3.6 = 36 :=
by
  sorry

end train_speed_solution_l194_194031


namespace max_ratio_is_half_l194_194463

-- Definitions based on given conditions
def hyperbola1 (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def hyperbola2 (a b x y : ℝ) : Prop := y^2 / b^2 - x^2 / a^2 = 1

-- Given that \( S_1 \) and \( S_2 \) are areas of particular quadrilaterals
def S1 (a b : ℝ) : ℝ := 4 * a * b
def S2 (a b : ℝ) : ℝ := 4 * (a^2 + b^2)^0.5 * (a^2 + b^2)^0.5

-- Ratio \( S_1 / S_2 \)
def ratio (a b : ℝ) : ℝ := (S1 a b) / (S2 a b)

-- The theorem to prove
theorem max_ratio_is_half (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ratio a b ≤ 1 / 2 :=
sorry

end max_ratio_is_half_l194_194463


namespace tangent_inclination_angle_at_x_1_l194_194453

noncomputable def g (x : ℝ) : ℝ := x^2 * Real.log x

theorem tangent_inclination_angle_at_x_1 : 
  let x₀ := 1 in 
  let g_deriv := deriv g x₀ in 
  let α := Real.arctan g_deriv in 
  α = Real.pi / 4 :=
by
  sorry

end tangent_inclination_angle_at_x_1_l194_194453


namespace total_outcomes_of_two_die_rolls_outcomes_with_sum_7_probability_sum_7_l194_194414

theorem total_outcomes_of_two_die_rolls (die_faces : Fin 6) :
  let outcomes := Finset.prod (Finset.univ : Finset (Fin 6)) (λ x, (Finset.univ : Finset (Fin 6))) in
  Finset.card outcomes = 36 := by
  sorry

theorem outcomes_with_sum_7 (die_faces : Fin 6) :
  let outcomes := Finset.prod (Finset.univ : Finset (Fin 6)) (λ x, (Finset.univ : Finset (Fin 6))) in
  let sum_7_outcomes := outcomes.filter (λ (xy : Fin 6 × Fin 6), xy.fst + xy.snd = 7) in
  Finset.card sum_7_outcomes = 6 := by
  sorry

theorem probability_sum_7 (die_faces : Fin 6) :
  let outcomes := Finset.prod (Finset.univ : Finset (Fin 6)) (λ x, (Finset.univ : Finset (Fin 6))) in
  let sum_7_outcomes := outcomes.filter (λ (xy : Fin 6 × Fin 6), xy.fst + xy.snd = 7) in
  (Finset.card sum_7_outcomes : ℚ) / (Finset.card outcomes : ℚ) = 1 / 6 := by
  sorry

end total_outcomes_of_two_die_rolls_outcomes_with_sum_7_probability_sum_7_l194_194414


namespace find_matrix_M_l194_194491

theorem find_matrix_M (a b c d : ℝ) :
  ∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
    M ⬝ Matrix.of ![![a, b], ![c, d]] = Matrix.of ![![2*a, 2*b], ![(1/2)*c, (1/2)*d]] → 
    M = Matrix.of ![![2, 0], ![0, 1/2]] :=
by
  -- proof is omitted
  sorry

end find_matrix_M_l194_194491


namespace expected_value_of_remainder_eq_1816_over_6561_l194_194276
-- Lean 4 statement for the math proof problem

noncomputable def expected_remainder_binomial_mod3 : ℚ :=
  1816 / 6561

theorem expected_value_of_remainder_eq_1816_over_6561 :
  ∀ (a b : ℕ), a ∈ Finset.range 81 → b ∈ Finset.range 81 → 
  (a ≠ b → ExpectedValue (λ (a b : ℕ) => (nat.choose a b) % 3) = expected_remainder_binomial_mod3) :=
begin
  intros a b ha hb h_not_eq,
  sorry -- proof is not required
end

end expected_value_of_remainder_eq_1816_over_6561_l194_194276


namespace number_of_birches_l194_194407

/-- Given 130 trees in a circle which are either birches or limes, with each tree having a sign "Two different trees grow next to me", and knowing that this statement is false for all limes and exactly one birch, prove that the number of birches must be 87. -/
theorem number_of_birches (total_trees : ℕ) (birches limes : ℕ) (sign_false_for_limes : ∀ l, l ∈ limes → "Two different trees grow next to me" is false)
  (sign_false_for_one_birch : ∃ b, b ∈ birches ∧ "Two different trees grow next to me" is false)
  (both_species_present : ∃ b l, b ∈ birches ∧ l ∈ limes) :
  total_trees = 130 -> birches = 87 :=
by
  intros ht total_trees_eq birches limes_eq
  sorry

end number_of_birches_l194_194407


namespace prove_relationship_l194_194144

variable {f : ℝ → ℝ}

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x ≠ 0 → f(x) ≠ 0) ∧
  (f(1) = 0) ∧
  (∀ a b : ℝ, (0 < a ∧ 0 < b) → f(a * b) ≥ a * f(b) + b * f(a))

theorem prove_relationship (f : ℝ → ℝ) (h : satisfies_conditions f) (a : ℝ) (ha : 0 < a) (n : ℕ) (hn : n > 0) :
  f(a^n) ≥ n * a^(n-1) * f(a) :=
sorry

end prove_relationship_l194_194144


namespace one_way_streets_possible_l194_194932

-- Definitions for the problem conditions
def corner : Type := ℕ -- A type to represent corners in the city.
def city : Type := set corner -- A type to represent the city as a set of corners.

variables (C : city) -- Given a city C.

-- Given condition that from any corner to any other corner, there are two different routes.
axiom two_routes (a b : corner) (ha : a ∈ C) (hb : b ∈ C) : ∃ p₁ p₂ : list corner, p₁.head = a ∧ p₁.last = b ∧ p₂.head = a ∧ p₂.last = b ∧ p₁ ≠ p₂

-- Main theorem: The city's streets can be made one-way while maintaining reachability between any pair of corners.
theorem one_way_streets_possible (hfin : C.finite) :
  ∃ (D : ∀ (a b : corner), a ∈ C → b ∈ C → Prop),
    (∀ a b, a ∈ C → b ∈ C → (D a b ∨ D b a)) ∧ 
    (∀ a b c, a ∈ C → b ∈ C → c ∈ C → (D a b ∧ D b c → D a c)) := sorry

end one_way_streets_possible_l194_194932


namespace distinct_triangles_of_prism_l194_194895

theorem distinct_triangles_of_prism (vertices : Finset ℕ) (h_vertices : vertices.card = 10) : 
  (vertices.choose 3).card = 120 := 
by
  sorry

end distinct_triangles_of_prism_l194_194895


namespace tan_C_eq_79_3_l194_194588

theorem tan_C_eq_79_3 (A B C : ℝ) 
(h1 : tan A = 3/4) 
(h2 : tan (A - B) = -1/3) 
(h3 : C = π - (A + B)) : 
  tan C = 79/3 := 
by 
  sorry

end tan_C_eq_79_3_l194_194588


namespace center_mass_intersection_at_single_point_l194_194840

theorem center_mass_intersection_at_single_point
  (n : ℕ) (h : n ≥ 3) (P : Fin n → ℝ × ℝ) (O : ℝ × ℝ)
  (h1: ∀ i : Fin n, (P i).dist O = (P 0).dist O) : -- all points lie on the circle centered at O
  ∃ Q : ℝ × ℝ, ∀ (i j : Fin n), i ≠ j →
  let M_ij := center_of_mass ({k : Fin n | k ≠ i ∧ k ≠ j}, (λ k : {k : Fin n | k ≠ i ∧ k ≠ j}, P k.1)) in
  let K_ij := midpoint (P i) (P j) in
  let L_ij := perpendicular_from_center_mass_to_chord M_ij (P i) (P j) in
  L_ij = (Q, (y : ℝ) -- line passes through Q
        | x : ℝ, ∃ k, x = k * (M_ij.1 - K_ij.1) + Q.1 ∧ y = k * (M_ij.2 - K_ij.2) + Q.2)
        := sorry

noncomputable def center_of_mass (s : Set.{u} (Fin n)) (f : ∀ a : s, ℝ × ℝ) : ℝ × ℝ :=
  (finset.centerMass s.toFinset (λ i, (1 : ℝ)) (λ i, f ⟨i, sorry⟩))

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def perpendicular_from_center_mass_to_chord (M : ℝ × ℝ) (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  { P : ℝ × ℝ | ∃ k : ℝ, (P.1 - M.1) = k * (B.2 - A.2) ∧ (P.2 - M.2) = -k * (B.1 - A.1) }

end center_mass_intersection_at_single_point_l194_194840


namespace find_f_at_seven_l194_194839

-- Definitions and conditions

def is_even_function_on (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def defined_on_interval (f : ℝ → ℝ) (a b : ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → f x = g x

-- The proof statement in Lean
theorem find_f_at_seven 
  (f : ℝ → ℝ)
  (h_even : is_even_function_on f)
  (h_periodic : periodic_function f 4)
  (h_defined_interval : defined_on_interval f 0 2 (λ x, 2 * x^2)) :
  f 7 = 2 :=
  sorry

end find_f_at_seven_l194_194839


namespace band_length_needed_l194_194674

constants (π : ℝ) (A : ℝ) (extra_length : ℝ)
noncomputable def radius (A π : ℝ) : ℝ := real.sqrt (A * (7 / 22))
noncomputable def circumference (r π : ℝ) : ℝ := 2 * π * r

-- Given conditions
axiom h1 : π = 22 / 7
axiom h2 : A = 616
axiom h3 : extra_length = 5

-- Proof statement
theorem band_length_needed : 
  let r := radius A (22 / 7) in
  let C := circumference r (22 / 7) in
  C + extra_length = 93 :=
by
  sorry

end band_length_needed_l194_194674


namespace right_triangle_trig_ratios_l194_194947

theorem right_triangle_trig_ratios
  (A B C : Type) [triangle A B C]
  (angle_BAC : angle A B C = 90)
  (AB : length A B = 40)
  (AC : length A C = 41)
  : tan (angle C B A) = (40 / 9) ∧ sin (angle C B A) = (40 / 41) := by
sorry

end right_triangle_trig_ratios_l194_194947


namespace length_CE_l194_194620

variables (C A E D B F : Type)

def perpendicular (x y : Type) : Prop := sorry -- Represents perpendicularity
def on_line (x y z : Type) : Prop := sorry     -- Represents being on the same line
def distinct (x y : Type) : Prop := sorry      -- Represents distinct points
def length (x y : Type) : ℝ := sorry           -- Represents the length between points

variables (h1 : on_line D A E)
variables (h2 : ¬ on_line C A E)
variables (h3 : perpendicular C D A E)
variables (h4 : on_line B C E)
variables (h5 : perpendicular A B C E)
variables (h6 : length A B = 6)
variables (h7 : length C D = 10)
variables (h8 : length A E = 7)
variables (h9 : on_line F A E)
variables (h10 : distinct D F)
variables (h11 : length D F = 3)

theorem length_CE :
  length C E = 35 / 3 :=
begin
  sorry
end

end length_CE_l194_194620


namespace king_gvidon_descendants_l194_194973

/-- King Gvidon had 5 sons. Among his descendants, 100 each had exactly 3 sons, 
    and the rest died childless. We aim to prove the total number of descendants 
    of King Gvidon is 305 including all generations. -/
theorem king_gvidon_descendants (sons: ℕ) (descendants_with_sons: ℕ) (sons_each: ℕ) : 
  sons = 5 → descendants_with_sons = 100 → sons_each = 3 → 
  5 + 100 * 3 = 305 :=
by
  intros hsons hdescendants_with_sons hsons_each
  rw [hsons, hdescendants_with_sons, hsons_each]
  norm_num
  sorry

end king_gvidon_descendants_l194_194973


namespace radius_of_cone_l194_194868

theorem radius_of_cone (S : ℝ) (h_S: S = 9 * Real.pi) (h_net: net_is_semi_circle) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_cone_l194_194868


namespace rotate180_of_point_A_l194_194162

-- Define the point A and the transformation
def point_A : ℝ × ℝ := (-3, 2)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement for the problem
theorem rotate180_of_point_A :
  rotate180 point_A = (3, -2) :=
sorry

end rotate180_of_point_A_l194_194162


namespace winning_percentage_l194_194029

theorem winning_percentage (total_games first_games remaining_games : ℕ) 
                           (first_win_percent remaining_win_percent : ℝ)
                           (total_games_eq : total_games = 60)
                           (first_games_eq : first_games = 30)
                           (remaining_games_eq : remaining_games = 30)
                           (first_win_percent_eq : first_win_percent = 0.40)
                           (remaining_win_percent_eq : remaining_win_percent = 0.80) :
                           (first_win_percent * (first_games : ℝ) +
                            remaining_win_percent * (remaining_games : ℝ)) /
                           (total_games : ℝ) * 100 = 60 := sorry

end winning_percentage_l194_194029


namespace max_number_of_pies_l194_194322

def total_apples := 250
def apples_given_to_students := 42
def apples_used_for_juice := 75
def apples_per_pie := 8

theorem max_number_of_pies (h1 : total_apples = 250)
                           (h2 : apples_given_to_students = 42)
                           (h3 : apples_used_for_juice = 75)
                           (h4 : apples_per_pie = 8) :
  ((total_apples - apples_given_to_students - apples_used_for_juice) / apples_per_pie) ≥ 16 :=
by
  sorry

end max_number_of_pies_l194_194322


namespace CD_CK_ratio_l194_194957

variables (A B C D K : Type) [Triangle A] [Triangle B] [Triangle C] [Triangle D] [Triangle K]

theorem CD_CK_ratio (α : ℝ) (AC BC CD CK : ℝ)
  (h1 : BC / AC = 3)
  (h2 : ∠ACB = α)
  (h3 : ∠ACD = α / 3)
  (h4 : ∠DCK = α / 3)
  (h5 : ∠KCB = α / 3) :
  CD / CK = (2 * cos (α / 3) + 3) / (1 + 6 * cos (α / 3)) :=
sorry

end CD_CK_ratio_l194_194957


namespace exists_no_interesting_pair_l194_194280

noncomputable def sequence (m : ℕ) : ℕ → ℕ
| 0     := m
| (n+1) := (sequence n) ^ 2 + 1

def interesting_pair (m : ℕ) (k l : ℕ) : Prop :=
0 < l - k ∧ l - k < 2016 ∧ (sequence m k) ∣ (sequence m l)

theorem exists_no_interesting_pair : ∃ (m : ℕ), ∀ (k l : ℕ), ¬ interesting_pair m k l :=
sorry

end exists_no_interesting_pair_l194_194280


namespace original_amount_charged_l194_194975

variables (P : ℝ) (interest_rate : ℝ) (total_owed : ℝ)

theorem original_amount_charged :
  interest_rate = 0.09 →
  total_owed = 38.15 →
  (P + P * interest_rate = total_owed) →
  P = 35 :=
by
  intros h_interest_rate h_total_owed h_equation
  sorry

end original_amount_charged_l194_194975


namespace intersection_S_T_l194_194170

def setS (x : ℝ) : Prop := (x - 1) * (x - 3) ≥ 0
def setT (x : ℝ) : Prop := x > 0

theorem intersection_S_T : {x : ℝ | setS x} ∩ {x : ℝ | setT x} = {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (3 ≤ x)} := 
sorry

end intersection_S_T_l194_194170


namespace average_waiting_time_l194_194434

-- Define the problem conditions
def light_period : ℕ := 3  -- Total cycle time in minutes
def green_time : ℕ := 1    -- Green light duration in minutes
def red_time : ℕ := 2      -- Red light duration in minutes

-- Define the probabilities of each light state
def P_G : ℚ := green_time / light_period
def P_R : ℚ := red_time / light_period

-- Define the expected waiting times given each state
def E_T_G : ℚ := 0
def E_T_R : ℚ := red_time / 2

-- Calculate the expected waiting time using the law of total expectation
def E_T : ℚ := E_T_G * P_G + E_T_R * P_R

-- Convert the expected waiting time to seconds
def E_T_seconds : ℚ := E_T * 60

-- Prove that the expected waiting time in seconds is 40 seconds
theorem average_waiting_time : E_T_seconds = 40 := by
  sorry

end average_waiting_time_l194_194434


namespace find_f1_find_range_f2_l194_194405

noncomputable def f1 (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 3

theorem find_f1 :
  (∀ x : ℝ, f1 (x + 1) = 4 * x^2 + 2 * x + 1) :=
by
  sorry

noncomputable def f2 (x : ℝ) : ℝ := -x^2 + x - 2

theorem find_range_f2 :
  (∀ x : ℝ, f2 (x + 2) - 2 * f2 (x) = x^2 - 5 * x) →
  (∀ y : ℝ, y ∈ set.range f2 → y ≤ -7 / 4) :=
by
  sorry

end find_f1_find_range_f2_l194_194405


namespace Triangle_tiling_Quad_tiling_Hex_tiling_l194_194307

-- Define the plane tiling problems for triangles, quadrilaterals, and centrally symmetric hexagons.
def canTilePlaneWithTriangles (T : Triangle) : Prop :=
  ∃ (f : ℝ × ℝ → Triangle), (∀ p, Congruent (f p) T) ∧ Tiling (f '' Univ)

def canTilePlaneWithQuadrilaterals (Q : Quadrilateral) : Prop :=
  ∃ (f : ℝ × ℝ → Quadrilateral), (∀ p, Congruent (f p) Q) ∧ Tiling (f '' Univ)

def canTilePlaneWithHexagons (H' : Hexagon) (h_symm : isCentrallySymmetric H') : Prop :=
  ∃ (f : ℝ × ℝ → Hexagon), (∀ p, Congruent (f p) H') ∧ Tiling (f '' Univ)

-- Now state the theorems indicating these properties formally
theorem Triangle_tiling (T : Triangle) : canTilePlaneWithTriangles T := sorry
theorem Quad_tiling (Q : Quadrilateral) : canTilePlaneWithQuadrilaterals Q := sorry
theorem Hex_tiling (H' : Hexagon) (h_symm : isCentrallySymmetric H') : canTilePlaneWithHexagons H' h_symm := sorry

end Triangle_tiling_Quad_tiling_Hex_tiling_l194_194307


namespace average_speeds_equal_l194_194251

def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

def jim_distance : ℝ := 16
def jim_time : ℝ := 2
def frank_distance : ℝ := 20
def frank_time : ℝ := 2.5
def susan_distance : ℝ := 12
def susan_time : ℝ := 1.5

theorem average_speeds_equal :
  let jim_speed := average_speed jim_distance jim_time
  let frank_speed := average_speed frank_distance frank_time
  let susan_speed := average_speed susan_distance susan_time
  (jim_speed = frank_speed) ∧ (jim_speed = susan_speed) ∧ (frank_speed - jim_speed = 0) ∧ (susan_speed - jim_speed = 0) :=
by
  sorry

end average_speeds_equal_l194_194251


namespace square_area_of_triangle_of_circles_is_275_l194_194067

noncomputable def square_area_of_triangle_of_circles : ℝ :=
  let r1 := 3
  let r2 := 5
  let r3 := 7
  let O1O2 := 2 * r1
  let O1O3 := 2 * r1
  let O2O3 := 2 * r2
  let s := (O1O2 + O1O3 + O2O3) / 2
  sqrt (s * (s - O1O2) * (s - O1O3) * (s - O2O3))

theorem square_area_of_triangle_of_circles_is_275 : square_area_of_triangle_of_circles ^ 2 = 275 :=
by
  sorry

end square_area_of_triangle_of_circles_is_275_l194_194067


namespace frank_total_cost_l194_194111

-- Conditions from the problem
def cost_per_bun : ℝ := 0.1
def number_of_buns : ℕ := 10
def cost_per_bottle_of_milk : ℝ := 2
def number_of_bottles_of_milk : ℕ := 2
def cost_of_carton_of_eggs : ℝ := 3 * cost_per_bottle_of_milk

-- Question and Answer
theorem frank_total_cost : 
  let cost_of_buns := cost_per_bun * number_of_buns in
  let cost_of_milk := cost_per_bottle_of_milk * number_of_bottles_of_milk in
  let cost_of_eggs := cost_of_carton_of_eggs in
  cost_of_buns + cost_of_milk + cost_of_eggs = 11 :=
by
  sorry

end frank_total_cost_l194_194111


namespace largest_third_altitude_l194_194707

theorem largest_third_altitude (DE DF EF : ℝ) (h1 h2 h3 : ℝ) :
  DE ≠ DF ∧ DF ≠ EF ∧ DE ≠ EF ∧  -- the triangle is scalene
  h1 = 6 ∧ h2 = 18 ∧              -- known altitudes
  h3 = 9 →                        -- proposed third altitude
  (∀ h h3 : ℝ,                  -- consider all possible third altitudes
    0 < h ∧ h ∈ ℤ →              -- altitude is a positive integer
    h3 ≤ h) →                     -- proposed third altitude is the maximum
  h3 = 9 :=
sorry

end largest_third_altitude_l194_194707


namespace correct_propositions_l194_194781

def isosceles_triangle_base_angles_half_vertex_is_right (A B C : ℝ) (h₁ : A = B) (h₂ : C = 2 * A) : Prop :=
  C = 90 ∧ A = 45 ∧ B = 45

def greater_area_circle (l : ℝ) : Prop :=
  let s_area := (l / 4)^2
  let r := l / (2 * Real.pi)
  let c_area := Real.pi * r^2
  c_area > s_area

def polyhedron_5_faces_not_necessarily_pyramid (P : ℕ) : Prop :=
  ∃ (faces : ℕ), faces = 5 ∧ ¬(P = faces)

def planes_divide_space (n : ℕ) : Prop :=
  n ∈ [4, 6, 7, 8]

theorem correct_propositions : 
  (isosceles_triangle_base_angles_half_vertex_is_right 45 45 90 true true) ∧ 
  (greater_area_circle 4) ∧ 
  ¬(polyhedron_5_faces_not_necessarily_pyramid 5) ∧ 
  (planes_divide_space 4) ∧ 
  (planes_divide_space 6) ∧ 
  (planes_divide_space 7) ∧ 
  (planes_divide_space 8) :=
sorry

end correct_propositions_l194_194781


namespace solution_set_of_inequality_l194_194358

theorem solution_set_of_inequality (x : ℝ) : 
  (0 < x ∧ x < 2) ↔ (| (2 - x) / x | > (x - 2) / x) :=
by
  sorry

end solution_set_of_inequality_l194_194358


namespace coordinates_of_point_M_in_second_quadrant_l194_194227

theorem coordinates_of_point_M_in_second_quadrant
  (x y : ℝ)
  (second_quadrant : x < 0 ∧ 0 < y)
  (dist_to_x_axis : abs(y) = 1)
  (dist_to_y_axis : abs(x) = 2) :
  (x, y) = (-2, 1) :=
sorry

end coordinates_of_point_M_in_second_quadrant_l194_194227


namespace sin_alpha_plus_beta_is_correct_cos_alpha_minus_beta_is_correct_l194_194534

variables (α β : ℝ)

-- Conditions
def cond1 : Prop := (π / 4) < α ∧ α < (3 * π / 4)
def cond2 : Prop := 0 < β ∧ β < (π / 4)
def cond3 : Prop := cos(π / 4 + α) = -4 / 5
def cond4 : Prop := sin(3 * π / 4 + β) = 12 / 13

-- Problems to prove
theorem sin_alpha_plus_beta_is_correct : cond1 α ∧ cond2 β ∧ cond3 α ∧ cond4 β → sin (α + β) = 63 / 65 :=
by sorry

theorem cos_alpha_minus_beta_is_correct : cond1 α ∧ cond2 β ∧ cond3 α ∧ cond4 β → cos (α - β) = -33 / 65 :=
by sorry

end sin_alpha_plus_beta_is_correct_cos_alpha_minus_beta_is_correct_l194_194534


namespace neither_sufficient_nor_necessary_l194_194988

-- Definitions based on given problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

-- The main theorem to prove
theorem neither_sufficient_nor_necessary (d a1 : ℝ) :
  let a : ℕ → ℝ := λ n, a1 + n * d
  let S : ℕ → ℝ := λ n, sum_of_first_n_terms a n
  is_arithmetic_sequence a d ∧ d > 0 -> (∃ n, S (n + 1) ≤ S n) ∧ (∃ n, S (n + 1) > S n) :=
sorry

end neither_sufficient_nor_necessary_l194_194988


namespace count_4_digit_palindromic_squares_is_2_l194_194183

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def count_4_digit_palindromic_squares : ℕ :=
  (Finset.range 100).filter (λ n, 32 ≤ n ∧ is_4_digit_number (n * n) ∧ is_palindrome (n * n)).card

theorem count_4_digit_palindromic_squares_is_2 : count_4_digit_palindromic_squares = 2 :=
  sorry

end count_4_digit_palindromic_squares_is_2_l194_194183


namespace num_4_digit_palindromic_squares_l194_194200

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_n (n : ℕ) : Prop :=
  32 ≤ n ∧ n ≤ 99

theorem num_4_digit_palindromic_squares : ∃ (count : ℕ), count = 3 ∧ ∀ n, valid_n n → is_4_digit (n^2) → is_palindrome (n^2) :=
sorry

end num_4_digit_palindromic_squares_l194_194200


namespace total_cost_correct_l194_194108

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end total_cost_correct_l194_194108


namespace domain_and_range_of_g_l194_194145

noncomputable def f : ℝ → ℝ := sorry

def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def g (x : ℝ) : ℝ :=
  Real.sqrt (f (x^2 - 3) + f (x + 1))

theorem domain_and_range_of_g :
  (strictly_increasing f) →
  (odd_function f) →
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f x ∧ f x ≤ 1) →
  (∀ x, x ∈ { -2 } ↔ f (x^2 - 3) + f (x + 1) ≥ 0) →
  ∀ x, x ∈ { -2 } ↔ g x = 0 :=
by
  intros h1 h2 h3 h4
  split
  { intro hx
    rw Set.mem_singleton_iff at hx
    rw hx
    have h5 : -2 ∈ [-1,1] := by norm_num
    have h6 : 1 ∈ [-1,1] := trivial
    rw Set.mem_singleton_iff
    simp only [g, Real.sqrt_eq_zero]
    rw [←h4 (-2), hx]
    ring
    sorry
  }
  { sorry }

end domain_and_range_of_g_l194_194145


namespace sqrt_comparison_l194_194622

theorem sqrt_comparison :
  let a := Real.sqrt 2
  let b := Real.sqrt 7 - Real.sqrt 3
  let c := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by
{
  sorry
}

end sqrt_comparison_l194_194622


namespace smallest_sum_l194_194689

-- Define the setup of the vertices and side sums
variables {a c e g i b d f h j : ℕ}

-- Define the condition for the common side sum
def common_sum (S : ℕ) :=
  a + b + c = S ∧ c + d + e = S ∧ e + f + g = S ∧ g + h + i = S ∧ i + j + a = S

-- Define the total sum of the numbers 1 through 10
def total_sum :=
  a + b + c + d + e + f + g + h + i + j = 55

-- Prove that the smallest possible value of the common side sum S is 14
theorem smallest_sum (h_sum : total_sum) :
  ∃ S, common_sum S ∧ S = 14 :=
sorry

end smallest_sum_l194_194689


namespace num_four_digit_square_palindromes_l194_194192

open Nat

-- Define what it means to be a 4-digit number
def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- The main theorem stating that there are exactly 2 four-digit squares that are palindromes
theorem num_four_digit_square_palindromes : 
  { n : ℕ | is_four_digit n ∧ is_palindrome n ∧ ∃ k : ℕ, k^2 = n ∧ k >= 32 ∧ k <= 99 }.to_finset.card = 2 :=
sorry

end num_four_digit_square_palindromes_l194_194192


namespace add_number_to_perfect_square_l194_194736

theorem add_number_to_perfect_square :
  ∃ n : ℕ, n + 13600 = 118 ^ 2 :=
by {
  use 324,
  sorry
}

end add_number_to_perfect_square_l194_194736


namespace quadratic_completing_square_l194_194352

theorem quadratic_completing_square:
  (∀ (b c : ℝ), (∀ x : ℝ, x^2 + 1800*x + 1800 = (x + b)^2 + c) → b = 900 ∧ c = -808200) →
  ∀ b c : ℝ, b = 900 → c = -808200 → c / b = -898 :=
by
  intros h b c hb hc
  rw [hb, hc]
  norm_num
  sorry

end quadratic_completing_square_l194_194352


namespace b_minus_c_eq_l194_194831

noncomputable def a (n : ℕ) (h : n > 1) : ℝ := 1 / Real.log_base (1024 : ℝ) n

noncomputable def b : ℝ :=
  a 3 (by decide) + a 4 (by decide) + a 5 (by decide) + a 6 (by decide)

noncomputable def c : ℝ :=
  a 15 (by decide) + a 16 (by decide) + a 17 (by decide) + a 18 (by decide) + a 19 (by decide)

theorem b_minus_c_eq : b - c = Real.log_base 1024 (1 / 3876) := by
  sorry

end b_minus_c_eq_l194_194831


namespace find_a_and_b_l194_194991

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := 3 * x - 8

theorem find_a_and_b (h : ∀ x, g (f x) = 4 * x + 5) : a + b = 17 / 3 := by
  sorry

end find_a_and_b_l194_194991


namespace wire_cut_example_l194_194730

theorem wire_cut_example (total_length piece_ratio : ℝ) (h1 : total_length = 28) (h2 : piece_ratio = 2.00001 / 5) :
  ∃ (shorter_piece : ℝ), shorter_piece + piece_ratio * shorter_piece = total_length ∧ shorter_piece = 20 :=
by
  sorry

end wire_cut_example_l194_194730


namespace smallest_positive_period_max_min_values_l194_194551

noncomputable def f (x : ℝ) := 2 * cos x * sin (x + π / 6) + cos x ^ 4 - sin x ^ 4

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem max_min_values :
  let a := -π / 12 in
  let b := π / 6 in
  ∃ x_min x_max,
  a ≤ x_min ∧ x_min ≤ b ∧
  a ≤ x_max ∧ x_max ≤ b ∧
  (∀ x, a ≤ x ∧ x ≤ b → f x ≥ f x_min) ∧
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
  f x_min = (√3 + 1)/2 ∧
  f x_max = √3 + 1/2 :=
sorry

end smallest_positive_period_max_min_values_l194_194551


namespace smallest_N_diagonals_l194_194940

theorem smallest_N_diagonals (n : ℕ) (h_n : n = 2017) : 
   ∃ (N : ℕ), N = 1008 ∧
    ∀ selected_diagonals : (Fin 2017 → Fin 2017) → list (Fin 2017 × Fin 2017), 
    (∀ i j, selected_diagonals i ≠ selected_diagonals j → (i ≠ j)) → 
    N > 1007 →
    ∃ i j, i ≠ j ∧ (∃ k l, selected_diagonals i = (⟨k, h_n⟩, ⟨l, h_n⟩) ∧ selected_diagonals j = (⟨k, h_n⟩, ⟨l, h_n⟩)) := by
  -- Proof skipped
  sorry

end smallest_N_diagonals_l194_194940


namespace daniela_vs_carlos_emilio_l194_194678

def miles_biked_Daniela := 75
def miles_biked_Carlos := 60
def miles_biked_Emilio := 45

theorem daniela_vs_carlos_emilio :
  miles_biked_Daniela - (miles_biked_Carlos + miles_biked_Emilio) = -30 :=
by
  sorry

end daniela_vs_carlos_emilio_l194_194678


namespace negation_of_universal_l194_194657

variable (f : ℝ → ℝ) (m : ℝ)

theorem negation_of_universal :
  (∀ x : ℝ, f x ≥ m) → ¬ (∀ x : ℝ, f x ≥ m) → ∃ x : ℝ, f x < m :=
by
  sorry

end negation_of_universal_l194_194657


namespace summation_problem_l194_194048

open BigOperators

theorem summation_problem : 
  (∑ i in Finset.range 50, ∑ j in Finset.range 75, 2 * (i + 1) + 3 * (j + 1) + (i + 1) * (j + 1)) = 4275000 :=
by
  sorry

end summation_problem_l194_194048


namespace range_of_k_l194_194360

theorem range_of_k (x k : ℝ):
  (2 * x + 9 > 6 * x + 1) → (x - k < 1) → (x < 2) → k ≥ 1 :=
by 
  sorry

end range_of_k_l194_194360


namespace circle_condition_chord_length_l194_194873

noncomputable def eq_circle (m : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 - 2 * m * x - 4 * y + 5 * m = 0 →
  (x - m)^2 + (y - 2)^2 = m^2 - 5 * m + 4

theorem circle_condition (m : ℝ) :
  (∃ (h : eq_circle m), m < 1 ∨ 4 < m) :=
sorry

theorem chord_length (m x y : ℝ) (h : eq_circle m) (line : 2 * x - y + 1 = 0) :
  m = -2 → ∃ r d : ℝ, r = 3 * real.sqrt 2 ∧ d = real.sqrt 5 ∧ 2 * real.sqrt (r^2 - d^2) = 2 * real.sqrt 13 :=
sorry

end circle_condition_chord_length_l194_194873


namespace envelopes_distributed_l194_194591

def WeChatGroup := {A, B, C, D}
def RedEnvelopes := {1, 2, 3}

/--
Given the WeChat group members and the red envelopes,
if A and B each grab one envelope, then there are exactly 
12 ways in which the envelopes can be distributed.
-/
theorem envelopes_distributed :
  (∃ s : Finset (WeChatGroup × RedEnvelopes), 
  s.card = 3 ∧  
  (∃ A_grabs B_grabs : (WeChatGroup × RedEnvelopes),
    A_grabs.fst = A ∧ B_grabs.fst = B ∧ 
    A_grabs ∈ s ∧ B_grabs ∈ s) ) → 
  {s : Finset (WeChatGroup × RedEnvelopes) | 
    s.card = 3 ∧ (A, envelope) ∈ s ∧ (B, envelope) ∈ s}.card = 12 :=
sorry

end envelopes_distributed_l194_194591


namespace minimize_PR_RQ_l194_194529

def calculate_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

def line_eq (x1 y1 slope x : ℝ) : ℝ := slope * (x - x1) + y1

theorem minimize_PR_RQ (x : ℝ) (m : ℝ) (P_x P_y Q_x Q_y : ℝ) :
  P_x = -1 → P_y = -2 → Q_x = 5 → Q_y = 3 →
  x = 2 →
  let slope := calculate_slope P_x P_y Q_x Q_y in
  let R_y := line_eq P_x P_y slope x in
  m = R_y →
  m = 1 / 2 :=
by
  intros hP_x hP_y hQ_x hQ_y hx hslope hR_y
  rw [hP_x, hP_y, hQ_x, hQ_y, hx] at *
  simp at *
  sorry

end minimize_PR_RQ_l194_194529


namespace class_C_payment_l194_194298

-- Definitions based on conditions
variables (x y z : ℤ) (total_C : ℤ)

-- Given conditions
def condition_A : Prop := 3 * x + 7 * y + z = 14
def condition_B : Prop := 4 * x + 10 * y + z = 16
def condition_C : Prop := 3 * (x + y + z) = total_C

-- The theorem to prove
theorem class_C_payment (hA : condition_A x y z) (hB : condition_B x y z) : total_C = 30 :=
sorry

end class_C_payment_l194_194298


namespace range_of_2alpha_minus_beta_l194_194910

def condition_range_alpha_beta (α β : ℝ) : Prop := 
  - (Real.pi / 2) < α ∧ α < β ∧ β < (Real.pi / 2)

theorem range_of_2alpha_minus_beta (α β : ℝ) (h : condition_range_alpha_beta α β) : 
  - Real.pi < 2 * α - β ∧ 2 * α - β < Real.pi / 2 :=
sorry

end range_of_2alpha_minus_beta_l194_194910


namespace dominoes_per_player_l194_194961

-- Define the conditions
def total_dominoes : ℕ := 28
def number_of_players : ℕ := 4

-- The theorem
theorem dominoes_per_player : total_dominoes / number_of_players = 7 :=
by sorry

end dominoes_per_player_l194_194961


namespace find_area_l194_194539

-- Lean definitions corresponding to the conditions
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry -- Placeholder for actual area calculation

variable A B C D E F : ℝ × ℝ

-- Given conditions as variables and hypotheses
variable (h₀ : triangle_area A B C = 10)
variable (h₁ : AD = 2)
variable (h₂ : DB = 3)
variable (h₃ : D ≠ A ∧ D ≠ B ∧ E ≠ B ∧ E ≠ C ∧ F ≠ A ∧ F ≠ C)
variable (h₄ : triangle_area A B E = (triangle_area D B E + triangle_area D E F))

-- Proof statement: area of triangle ABE is 6
theorem find_area (h₀ : triangle_area A B C = 10)
  (h₁ : AD = 2)
  (h₂ : DB = 3)
  (h₃ : D ≠ A ∧ D ≠ B ∧ E ≠ B ∧ E ≠ C ∧ F ≠ A ∧ F ≠ C)
  (h₄ : triangle_area A B E = (triangle_area D B E + triangle_area D E F))
  : triangle_area A B E = 6 :=
sorry

end find_area_l194_194539


namespace eval_complex_exponentiations_l194_194076

theorem eval_complex_exponentiations :
  (3 : ℂ) * (complex.I) ^ (23 : ℕ) + (2 : ℂ) * (complex.I) ^ (47 : ℕ) = - (5 : ℂ) * complex.I := 
by
  sorry

end eval_complex_exponentiations_l194_194076


namespace distinct_values_for_even_integers_lt_20_l194_194468

open Set

theorem distinct_values_for_even_integers_lt_20 :
  let evens := {2, 4, 6, 8, 10, 12, 14, 16, 18}
  let results := { (p + 1) * (q + 1) - 1 | p in evens, q in evens }
  results.finite :=
by sorry

end distinct_values_for_even_integers_lt_20_l194_194468


namespace cyclic_points_l194_194037

open EuclideanGeometry

variables {A B C D O O1 O2 E : Point}

def isosceles_trapezoid (A B C D : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  (∃ (a b c d : ℝ), a ≠ b ∧ d ≠ c ∧
  distance A B = a ∧ distance B C = b ∧ distance C D = c ∧ distance D A = d ∧
  a + c = b + d)

def inscribed_in_circle (A B C D O : Point) : Prop :=
  collinear [A, B, O] ∧
  collinear [B, C, O] ∧
  collinear [C, D, O] ∧
  collinear [D, A, O] ∧ 
  (distance O A = distance O B ∧
  distance O B = distance O C ∧ 
  distance O C = distance O D)

theorem cyclic_points {A B C D O O1 O2 E : Point}
  (h_trap : isosceles_trapezoid A B C D)
  (h_in_circle : inscribed_in_circle A B C D O)
  (h_BO_int_AD : collinear [B, O, E] ∧ collinear [A, D, E]) :
  cyclic O1 O2 O C := sorry

end cyclic_points_l194_194037


namespace brass_to_band_ratio_l194_194786

theorem brass_to_band_ratio
  (total_students : ℕ)
  (marching_band_fraction brass_saxophone_fraction saxophone_alto_fraction : ℚ)
  (alto_saxophone_students : ℕ)
  (h1 : total_students = 600)
  (h2 : marching_band_fraction = 1 / 5)
  (h3 : brass_saxophone_fraction = 1 / 5)
  (h4 : saxophone_alto_fraction = 1 / 3)
  (h5 : alto_saxophone_students = 4) :
  ((brass_saxophone_fraction * saxophone_alto_fraction) * total_students * marching_band_fraction = 4) →
  ((brass_saxophone_fraction * 3 * marching_band_fraction * total_students) / (marching_band_fraction * total_students) = 1 / 2) :=
by {
  -- Here we state the proof but leave it as a sorry placeholder.
  sorry
}

end brass_to_band_ratio_l194_194786


namespace sum_positive_integers_l194_194829

theorem sum_positive_integers (P : ℕ → Prop) :
  (∀ n : ℕ, P n ↔ 1.5 * n - 6.75 < 8.25) →
  ∑ n in Finset.filter P (Finset.range 10), n = 45 :=
begin
  assume h,
  sorry
end

end sum_positive_integers_l194_194829


namespace largest_abs_value_l194_194445

theorem largest_abs_value :
  let a := -real.pi
  let b := 0
  let c := 3
  let d := real.sqrt 3
  abs a > abs b ∧ abs a > abs c ∧ abs a > abs d := by
  sorry

end largest_abs_value_l194_194445


namespace part_a_part_b_part_c_l194_194259

def f (n d : ℕ) : ℕ := sorry

theorem part_a (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 ≤ n :=
sorry

theorem part_b (n d : ℕ) (h_even_n_minus_d : (n - d) % 2 = 0) : f n d ≤ (n + d) / (d + 1) :=
sorry

theorem part_c (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 = n :=
sorry

end part_a_part_b_part_c_l194_194259


namespace minimize_xy_l194_194223

theorem minimize_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_eq : 7 * x + 4 * y = 200) : (x * y = 172) :=
sorry

end minimize_xy_l194_194223


namespace exp_f_f_increasing_inequality_l194_194865

noncomputable def f (a b : ℝ) (x : ℝ) :=
  (a * x + b) / (x^2 + 1)

-- Conditions
variable (a b : ℝ)
axiom h_odd : ∀ x : ℝ, f a b (-x) = - f a b x
axiom h_value : f a b (1/2) = 2/5

-- Proof statements
theorem exp_f : f a b x = x / (x^2 + 1) := sorry

theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : 
  f a b x1 < f a b x2 := sorry

theorem inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  f a b (2 * x - 1) + f a b x < 0 := sorry

end exp_f_f_increasing_inequality_l194_194865


namespace common_roots_correct_l194_194329

noncomputable section
def common_roots_product (A B : ℝ) : ℝ :=
  let p := sorry
  let q := sorry
  p * q

theorem common_roots_correct (A B : ℝ) (h1 : ∀ x, x^3 + 2*A*x + 20 = 0 → x = p ∨ x = q ∨ x = r) 
    (h2 : ∀ x, x^3 + B*x^2 + 100 = 0 → x = p ∨ x = q ∨ x = s)
    (h_sum1 : p + q + r = 0) 
    (h_sum2 : p + q + s = -B)
    (h_prod1 : p * q * r = -20) 
    (h_prod2 : p * q * s = -100) : 
    common_roots_product A B = 10 * (2000)^(1/3) ∧ 15 = 10 + 3 + 2 :=
by
  sorry

end common_roots_correct_l194_194329


namespace trig_identity_from_point_l194_194152
-- Import necessary libraries

-- Define our problem conditions and exactly as outlined
theorem trig_identity_from_point (a : ℝ) (h : a < 0) :
  let x := -4 * a,
      y := 3 * a,
      r := Real.sqrt (x^2 + y^2),
      sin_alpha := y / r,
      cos_alpha := x / r
  in 2 * sin_alpha + cos_alpha = -2 / 5 :=
by {
  -- We are skipping the proof.
  sorry
}

end trig_identity_from_point_l194_194152


namespace tangent_value_correct_l194_194216

def tangent_sum_value (A B : ℝ) (tan_A tan_B : ℝ) : ℝ :=
  (1 + tan_A) * (1 + tan_B)

theorem tangent_value_correct :
  let A := 30 * (Real.pi / 180)
  let B := 40 * (Real.pi / 180)
  let tan_A := Real.tan A
  let tan_B := Real.tan B
  tangent_sum_value A B tan_A tan_B = 2.90 :=
by
  let A := 30 * (Real.pi / 180)
  let B := 40 * (Real.pi / 180)
  let tan_A := Real.tan A
  let tan_B := Real.tan B
  let result := (1 + tan_A) * (1 + tan_B)
  have h : Real.abs (result - 2.90) < 0.001 := sorry
  sorry

end tangent_value_correct_l194_194216


namespace four_digit_square_palindromes_are_zero_l194_194178

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

-- Define the main theorem statement
theorem four_digit_square_palindromes_are_zero : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) → 
             is_palindrome n → 
             (∃ m : ℕ, n = m * m) → 
             n = 0 :=
by
  sorry

end four_digit_square_palindromes_are_zero_l194_194178


namespace area_of_trapezoid_l194_194950

theorem area_of_trapezoid (ABC isosceles : Bool) (area_ABC : ℝ) (areas_smallest_triangles : list ℝ) 
  (num_smallest_triangles : ℕ) (area_trapezoid : ℝ) : 
  ABC.isosceles = true → area_ABC = 50 → 
  (∀ t in areas_smallest_triangles, t = 2) → num_smallest_triangles = 5 →  
  area_trapezoid = (50 - 3 * 2) :=
by 
  intros;
  sorry

end area_of_trapezoid_l194_194950


namespace number_of_terms_with_positive_integer_powers_l194_194346

theorem number_of_terms_with_positive_integer_powers 
  (x : ℝ) : 
  ∃ n : ℕ, n = 2 ∧ 
    (∀ r ∈ {0, 2}, ∃ T : ℝ, is_term_in_expansion T r x) :=
sorry

end number_of_terms_with_positive_integer_powers_l194_194346


namespace basketball_score_l194_194930

noncomputable theory

def remaining_score {T : ℝ} (A_score B_score : ℝ) (C_score : ℝ) (max_per_player : ℝ) (remaining_players_count : ℕ): ℝ :=
  T - (A_score + B_score + C_score)

theorem basketball_score (T : ℝ) (A_score B_score : ℝ) (C_score total_max_remaining : ℝ) (remaining_players_count : ℕ)
  (hA : A_score = 1/4 * T)
  (hB : B_score = 2/7 * T)
  (hC : C_score = 15)
  (h_max : total_max_remaining = remaining_players_count * 2) :
  remaining_score A_score B_score C_score (remaining_players_count * 2) remaining_players_count = 13 :=
sorry

end basketball_score_l194_194930


namespace construct_congruent_triangle_with_parallel_lines_l194_194712

noncomputable def triangle (α β γ : Type) := sorry

theorem construct_congruent_triangle_with_parallel_lines (ABC A1B1C1: Type)
  (area_ABC_eq_area_A1B1C1 : sorry)
  (parallel_lines: sorry) :
  ∃ (A2 B2 C2: Type), 
  triangle A2 B2 C2 ∧
  triangle A1B1C1 A2 B2 C2 ∧ 
  (line A A2 ∥ line B B2) ∧ 
  (line A A2 ∥ line C C2) ∧ 
  (line B B2 ∥ line C C2) :=
sorry

end construct_congruent_triangle_with_parallel_lines_l194_194712


namespace magnitude_of_2a_minus_b_l194_194544

noncomputable def vec_a : ℝ := sorry
noncomputable def vec_b : ℝ := sorry

def angle_between_vectors := 30
def mag_a := real.sqrt 3
def mag_b := 4

axiom dot_product_formula (a b : ℝ) (angle : ℝ) : a * b * real.cos (angle * real.pi / 180) = a * b * real.cos (angle / 180 * real.pi / 2)

theorem magnitude_of_2a_minus_b :
  let dot_ab := mag_a * mag_b * real.cos (angle_between_vectors * real.pi / 180) in
  let mag_2a_minus_b_squared := (2 * mag_a)^2 + mag_b^2 - 2 * 2 * mag_a * mag_b * real.cos (angle_between_vectors * real.pi / 180) in
  real.sqrt mag_2a_minus_b_squared = 2 :=
by
  sorry

end magnitude_of_2a_minus_b_l194_194544


namespace sine_avg_lt_sine_avg_l194_194528

variable {x1 x2 : ℝ}
variable (hx1 : 0 < x1 ∧ x1 < π) (hx2 : 0 < x2 ∧ x2 < π) (hx1_ne_x2 : x1 ≠ x2)

theorem sine_avg_lt_sine_avg (hx1 : 0 < x1 ∧ x1 < π) (hx2 : 0 < x2 ∧ x2 < π) (hx1_ne_x2 : x1 ≠ x2) :
  (sin x1 + sin x2) / 2 < sin ((x1 + x2) / 2) :=
sorry

end sine_avg_lt_sine_avg_l194_194528


namespace sum_geom_seq_nine_l194_194602

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sum_geom_seq_nine {a : ℕ → ℝ} {q : ℝ} (h_geom : geom_seq a q)
  (h1 : a 1 * (1 + q + q^2) = 30) 
  (h2 : a 4 * (1 + q + q^2) = 120) :
  a 7 + a 8 + a 9 = 480 :=
  sorry

end sum_geom_seq_nine_l194_194602


namespace equality_of_coefficients_l194_194633

open Real

theorem equality_of_coefficients (a b c x : ℝ)
  (h1 : a * x^2 - b * x - c = b * x^2 - c * x - a)
  (h2 : b * x^2 - c * x - a = c * x^2 - a * x - b)
  (h3 : c * x^2 - a * x - b = a * x^2 - b * x - c):
  a = b ∧ b = c :=
sorry

end equality_of_coefficients_l194_194633


namespace baron_not_lying_l194_194042

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem baron_not_lying : 
  ∃ a b : ℕ, 
  (a ≠ b ∧ a ≥ 10^9 ∧ a < 10^10 ∧ b ≥ 10^9 ∧ b < 10^10 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a + sum_of_digits (a * a) = b + sum_of_digits (b * b))) :=
  sorry

end baron_not_lying_l194_194042


namespace prove_conditions_and_find_range_l194_194123

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 1 / (4^x + 1)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonic_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem prove_conditions_and_find_range (a : ℝ) (k : ℝ) :
  is_odd_function (f a) →
  f 0 a = 0 →
  f a = λ x, -1/2 + 1 / (4^x + 1) →
  is_monotonic_decreasing (f (-1/2)) →
  (∀ t : ℝ, f (t^2 - 2 * t) (-1/2) + f (2 * t^2 - k) (-1/2) < 0) →
  a = -1/2 ∧ k < -1/3 :=
by
  sorry

end prove_conditions_and_find_range_l194_194123


namespace flower_beds_fraction_l194_194026

-- Define the main problem parameters
def leg_length := (30 - 18) / 2
def triangle_area := (1 / 2) * (leg_length ^ 2)
def total_flower_bed_area := 2 * triangle_area
def yard_area := 30 * 6
def fraction_of_yard_occupied := total_flower_bed_area / yard_area

-- The theorem to be proved
theorem flower_beds_fraction :
  fraction_of_yard_occupied = 1/5 := by
  sorry

end flower_beds_fraction_l194_194026


namespace polar_equation_C1_intersection_C2_C1_distance_l194_194952

noncomputable def parametric_to_cartesian (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + 2 * Real.cos α ∧ y = 4 + 2 * Real.sin α

noncomputable def cartesian_to_polar (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 4

noncomputable def polar_equation_of_C1 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 16 = 0

noncomputable def C2_line_polar (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem polar_equation_C1 (α : ℝ) (ρ θ : ℝ) :
  parametric_to_cartesian α →
  cartesian_to_polar (2 + 2 * Real.cos α) (4 + 2 * Real.sin α) →
  polar_equation_of_C1 ρ θ :=
by
  sorry

theorem intersection_C2_C1_distance (ρ θ : ℝ) (t1 t2 : ℝ) :
  C2_line_polar θ →
  polar_equation_of_C1 ρ θ →
  (t1 + t2 = 6 * Real.sqrt 2) ∧ (t1 * t2 = 16) →
  |t1 - t2| = 2 * Real.sqrt 2 :=
by
  sorry

end polar_equation_C1_intersection_C2_C1_distance_l194_194952


namespace number_of_rectangles_in_4x4_grid_l194_194798

theorem number_of_rectangles_in_4x4_grid : 
  let n := 5 in -- n is the number of lines (5 horizontal and 5 vertical lines)
  (∑ i in finset.range n, ∑ j in finset.range n, if 2 ≤ i ∧ 2 ≤ j then 1 else 0) = 100 :=
by
  let n := 5
  sorry

end number_of_rectangles_in_4x4_grid_l194_194798


namespace inequality_false_implies_range_of_a_l194_194351

theorem inequality_false_implies_range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - 2 * t - a ≥ 0) ↔ a ≤ -1 :=
by
  sorry

end inequality_false_implies_range_of_a_l194_194351


namespace fraction_sum_l194_194687

theorem fraction_sum (y : ℝ) (a b : ℤ) (h : y = 3.834834834) (h_frac : y = (a : ℝ) / b) (h_coprime : Int.gcd a b = 1) : a + b = 4830 :=
sorry

end fraction_sum_l194_194687


namespace PD_equals_R_l194_194632

open Real

-- Let A, B, C, D be points on a circle with radius R
variable {A B C D P : Point}
variable (R : ℝ)
variable (circum : Circle R)

-- Let P be a point on the segment CD
variable (P_on_CD : P ∈ [CD])

-- Assume the given conditions
axiom h1 : dist C B = dist B P
axiom h2 : dist B P = dist P A
axiom h3 : dist P A = dist A B

-- Define the goal
theorem PD_equals_R : dist P D = R := by
  sorry

end PD_equals_R_l194_194632


namespace square_area_proof_l194_194028

-- Define the conditions in Lean
variable (Radius : ℝ) (SideLength : ℝ) (Area : ℝ)

-- Assume radius of each circle is 7 inches
def radius_of_circle := Radius = 7

-- Define the side length of the square given the radius
def side_length_of_square := SideLength = 2 * 2 * Radius

-- Define the area of the square as the square of its side length
def area_of_square := Area = SideLength * SideLength

-- The theorem stating the area of the square under the given conditions
theorem square_area_proof : radius_of_circle Radius → side_length_of_square Radius SideLength → area_of_square Radius SideLength Area → Area = 784 := by
  intros hradius hside_length harea
  sorry

end square_area_proof_l194_194028


namespace chessboard_impossible_equal_black_white_1900x1900_l194_194079

theorem chessboard_impossible_equal_black_white_1900x1900 :
  let n := 1900 in
  let is_symmetric_diff (board : Fin n → Fin n → Bool) : Prop :=
        ∀ i j, board i j ≠ board (Fin.ofNat (n - 1 - i)) (Fin.ofNat (n - 1 - j)) in
  ∀ (board : Fin n → Fin n → Bool), is_symmetric_diff board →
    ¬ (∀ i, Fintype.card {j // board i j} = Fintype.card {j // ¬ board i j} ∧
             Fintype.card {i // board i j} = Fintype.card {i // ¬ board i j}) :=
by
  let n := 1900
  let is_symmetric_diff := λ (board : Fin n → Fin n → Bool) =>
        ∀ i j, board i j ≠ board (Fin.ofNat (n - 1 - i)) (Fin.ofNat (n - 1 - j))
  assume board h
  sorry

end chessboard_impossible_equal_black_white_1900x1900_l194_194079


namespace sin_cos_sum_l194_194697

theorem sin_cos_sum (α : ℝ) (h : ∃ (c : ℝ), Real.sin α = -1 / c ∧ Real.cos α = 2 / c ∧ c = Real.sqrt 5) :
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 :=
by sorry

end sin_cos_sum_l194_194697


namespace average_transformation_l194_194916

theorem average_transformation (a b c : ℝ) (h : (a + b + c) / 3 = 12) : ((2 * a + 1) + (2 * b + 2) + (2 * c + 3) + 2) / 4 = 20 :=
by
  sorry

end average_transformation_l194_194916


namespace g_1987_l194_194904

def g (x : ℕ) : ℚ := sorry

axiom g_defined_for_all (x : ℕ) : true

axiom g1 : g 1 = 1

axiom g_rec (a b : ℕ) : g (a + b) = g a + g b - 3 * g (a * b) + 1

theorem g_1987 : g 1987 = 2 := sorry

end g_1987_l194_194904


namespace binomial_problem_l194_194542

theorem binomial_problem {x : ℝ} (C : Fin n → ℕ) (h_sum_coef : ∑ k in Finset.range n.succ, C k = 256) :
  n = 8 ∧ (∃ T : ℕ, T = Nat.choose 8 2 ∧ T = 28) :=
by
  sorry

end binomial_problem_l194_194542


namespace problem_1_problem_2_l194_194564

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (AB CD : EuclideanSpace ℝ (Fin 3))

noncomputable def angle := real.angle a b

-- Definitions of lengths and vectors
def length_a : ℝ := 2
def length_b : ℝ := 1
def theta : ℝ := real.angle a b

-- Conditions
axiom length_a_condition : ∥a∥ = length_a
axiom length_b_condition : ∥b∥ = length_b
axiom AB_condition : AB = 2 • a - b
axiom CD_condition : CD = a + 3 • b

-- Problem (1): Given the angle between a and b is 60 degrees
noncomputable def angle_60_degrees : ℝ := real.pi / 3
axiom angle_condition : real.angle a b = angle_60_degrees

-- Problem (2): Vectors AB and CD are perpendicular
axiom perpendicular_condition : inner AB CD = 0

-- Proofs
theorem problem_1 : ∥a - b∥ = real.sqrt 3 :=
by sorry

theorem problem_2 : angle = 2 * real.pi / 3 :=
by sorry

end problem_1_problem_2_l194_194564


namespace circle_radius_l194_194095

theorem circle_radius (x y : ℝ) : x^2 - 10 * x + y^2 + 4 * y + 13 = 0 → (x - 5)^2 + (y + 2)^2 = 4^2 :=
by
  sorry

end circle_radius_l194_194095


namespace coefficient_condition_l194_194586

theorem coefficient_condition (m : ℝ) (h : m^3 * Nat.choose 6 3 = -160) : m = -2 := sorry

end coefficient_condition_l194_194586


namespace probability_of_rain_given_east_wind_l194_194777

variable (P : Prop → ℚ)
variable (A B : Prop)
variable hA : P A = 3 / 10
variable hAB : P (A ∧ B) = 4 / 15

theorem probability_of_rain_given_east_wind:
  P (B | A) = 8 / 9 :=
by
  sorry

end probability_of_rain_given_east_wind_l194_194777


namespace sheets_borrowed_l194_194894

-- Define the total number of sheets
def total_sheets : ℕ := 30

-- Define the condition on page number types 
def total_pages : ℕ := 60

-- Define the average of remaining pages that Hiram calculates
def average_remaining_pages : ℕ := 25

-- Function to compute the number of borrowed sheets, which needs to be proven to be 15
noncomputable def borrowed_sheets (n : ℕ) (c : ℕ) (avg : ℕ) : Prop :=
  (c = 15) ∧ (n = 30) ∧ (avg = 25)

-- Prove that borrowed_sheets condition holds for the given problem
theorem sheets_borrowed : borrowed_sheets total_sheets _ average_remaining_pages :=
by
  -- exists exactly when the number of borrowed sheets is 15
  use 15,
  split,
  { 
    sorry
  },
  {
    split,
    {
      exact rfl,  -- Referencing the definition
    },
    {
      exact rfl, -- Referencing the average of 25
    }
  }

end sheets_borrowed_l194_194894


namespace contradiction_proof_l194_194304

theorem contradiction_proof (x y : ℝ) (h1 : x + y ≤ 0) (h2 : x > 0) (h3 : y > 0) : false :=
by
  sorry

end contradiction_proof_l194_194304


namespace no_integers_a_c_l194_194630

noncomputable theory
open_locale classical

variables (p : ℕ) (hp : prime p) (hp_ge_7 : p ≥ 7)
          (A : finset ℕ) (hA : ∀ x ∈ A, ∃ k, k^2 ≡ x [MOD p] ∧ ¬ x ≡ 0 [MOD p])

-- Defining a condition for A to be the set of non-zero quadratic residues modulo p
def is_quadratic_residue (x : ℕ) : Prop := ∃ k, k^2 ≡ x [MOD p] ∧ ¬ x ≡ 0 [MOD p]

-- Definition of the main proof goal
theorem no_integers_a_c (a c : ℤ) (a_non_zero : ¬ a ≡ 0 [MOD p]) (c_non_zero : ¬ c ≡ 0 [MOD p]) :
    ∀ (b ∈ A), (a * b + c) ∉ A :=
begin
    sorry
end

end no_integers_a_c_l194_194630


namespace natural_numbers_condition_l194_194487

theorem natural_numbers_condition {n : ℕ} (x : Fin n → ℤ) (y : ℤ) :
  (∀ i, x i ≠ 0) ∧ y ≠ 0 ∧ (∑ i, x i) = 0 ∧ n * y^2 = ∑ i, (x i)^2 → n ≥ 2 :=
by
  sorry

end natural_numbers_condition_l194_194487


namespace sum_of_roots_eq_l194_194722

theorem sum_of_roots_eq (x : ℝ) : (x - 4)^2 = 16 → (∃ r1 r2 : ℝ, (x - 4) = 4 ∨ (x - 4) = -4 ∧ r1 + r2 = 8) :=
by
  have h := (x - 4) ^ 2 = 16
  sorry  -- You would proceed with the proof here.

end sum_of_roots_eq_l194_194722


namespace complex_ratio_max_min_diff_l194_194994

noncomputable def max_minus_min_complex_ratio (z w : ℂ) : ℝ :=
max (1 : ℝ) (0 : ℝ) - min (1 : ℝ) (0 : ℝ)

theorem complex_ratio_max_min_diff (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) : 
  max_minus_min_complex_ratio z w = 1 :=
by sorry

end complex_ratio_max_min_diff_l194_194994


namespace f_monotonic_range_a_max_value_f_diff_l194_194159

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 2 * Real.log x

-- Problem 1
theorem f_monotonic_range_a (a : ℝ) : 
  (∀ x, f x a = x^2 + a * x + 2 * Real.log x ∧ x > 0 → monotone_on (λ x, f x a) (set.Ioi 0)) ↔ a ≥ -4 :=
  by sorry

-- Problem 2
theorem max_value_f_diff (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, f x a = x^2 + a * x + 2 * Real.log x ∧ 0 < x₁ < 1 ∧ 1 < x₂) ∧
  (| x₁ - x₂ | ≤ (3 / 2)) ∧ (2 * x₂^2 + a * x₂ + 2 = 0) →
  | f x₁ a - f x₂ a | ≤ (15 / 4 - 2 * Real.log 4) :=
  by sorry

end f_monotonic_range_a_max_value_f_diff_l194_194159


namespace tim_total_sleep_correct_l194_194372

def total_sleep_hours (home_country_sleep_hours_per_day : ℕ) (days_in_home_country : ℕ) (weekend_trip_sleep_hours_per_day : ℕ) (time_zone_change_saturday : ℤ) (daylight_saving_adjustment_sunday : ℤ) : ℤ :=
  let mon_fri_sleep := days_in_home_country * home_country_sleep_hours_per_day
  let saturday_sleep := weekend_trip_sleep_hours_per_day - time_zone_change_saturday
  let sunday_sleep := weekend_trip_sleep_hours_per_day + daylight_saving_adjustment_sunday
  mon_fri_sleep + saturday_sleep + sunday_sleep

theorem tim_total_sleep_correct :
  total_sleep_hours 6 5 10 3 (-1) = 48 := by
  simp [total_sleep_hours]
  norm_num
  done
  sorry

end tim_total_sleep_correct_l194_194372


namespace number_of_correct_statements_l194_194368

theorem number_of_correct_statements:
  (¬∀ (a : ℝ), -a < 0) ∧
  (∀ (x : ℝ), |x| = -x → x < 0) ∧
  (∀ (a : ℚ), (∀ (b : ℚ), |b| ≥ |a|) → a = 0) ∧
  (∀ (x y : ℝ), 5 * x^2 * y ≠ 0 → 2 + 1 = 3) →
  2 = 2 := sorry

end number_of_correct_statements_l194_194368


namespace quadratic_coefficient_a_l194_194334

theorem quadratic_coefficient_a (a b c : ℝ) :
  (2 = 9 * a - 3 * b + c) ∧
  (2 = 9 * a + 3 * b + c) ∧
  (-6 = 4 * a + 2 * b + c) →
  a = 8 / 5 :=
by
  sorry

end quadratic_coefficient_a_l194_194334


namespace find_interesting_numbers_l194_194021

def is_interesting (A B : ℕ) : Prop :=
  A > B ∧ (∃ p : ℕ, Nat.Prime p ∧ A - B = p) ∧ ∃ n : ℕ, A * B = n ^ 2

theorem find_interesting_numbers :
  {A | (∃ B : ℕ, is_interesting A B) ∧ 200 < A ∧ A < 400} = {225, 256, 361} :=
by
  sorry

end find_interesting_numbers_l194_194021


namespace probability_of_adjacent_rs_is_two_fifth_l194_194390

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def countArrangementsWithAdjacentRs : ℕ :=
factorial 4

noncomputable def countTotalArrangements : ℕ :=
factorial 5 / factorial 2

noncomputable def probabilityOfAdjacentRs : ℚ :=
(countArrangementsWithAdjacentRs : ℚ) / (countTotalArrangements : ℚ)

theorem probability_of_adjacent_rs_is_two_fifth :
  probabilityOfAdjacentRs = 2 / 5 := by
  sorry

end probability_of_adjacent_rs_is_two_fifth_l194_194390


namespace volume_ratio_l194_194247

variable (r h : ℝ)

def cone_volume (r h : ℝ) : ℝ := (1/3) * π * r^2 * h
def prism_volume (r h : ℝ) : ℝ := (3 * r)^2 * h 

theorem volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (cone_volume r h) / (prism_volume r h) = (π / 27) :=
by
  sorry

end volume_ratio_l194_194247


namespace tony_running_distance_each_morning_l194_194373

def timeWalking (d_walk speed_walk : ℕ) := d_walk / speed_walk
def timeRunning (d_run speed_run : ℕ) := d_run / speed_run
def totalExerciseTime (daily_walk_time daily_run_time days_weekly : ℕ) :=
  daily_walk_time * days_weekly + daily_run_time * days_weekly

theorem tony_running_distance_each_morning 
    (d_walk : ℕ) (speed_walk : ℕ) (d_run : ℕ) (speed_run : ℕ) (weekly_exercise_hours : ℕ) (days_weekly : ℕ)
    (h_walk : d_walk = 3) (h_speed_walk : speed_walk = 3) 
    (h_speed_run : speed_run = 5) (h_exercise_time : weekly_exercise_hours = 21) 
    (h_days : days_weekly = 7) : 
    d_run = 10 :=
  by
  let time_walking := timeWalking d_walk speed_walk
  let time_running := timeRunning d_run speed_run
  have h1 : time_walking * days_weekly + time_running * days_weekly = weekly_exercise_hours := sorry
  have h2 : time_walking = 1 := by 
    rw [timeWalking, h_walk, h_speed_walk]
    norm_num

    sorry
  have h3 : time_running = d_run / speed_run := by 
    rw [timeRunning] 
    sorry
  sorry

end tony_running_distance_each_morning_l194_194373


namespace game_duration_l194_194966

theorem game_duration (G : ℝ) :
  let total_games := 2 * 8,
      total_practice_hours := total_games * 4,
      total_hours_spent := 96,
      total_game_hours := total_hours_spent - total_practice_hours
  in total_game_hours / total_games = G → G = 2 :=
by
  let total_games := 2 * 8
  let total_practice_hours := total_games * 4
  let total_hours_spent := 96
  let total_game_hours := total_hours_spent - total_practice_hours
  have h1 : total_game_hours = total_hours_spent - total_practice_hours := by sorry
  have h2 : total_game_hours / total_games = 2 := by sorry
  show G = 2, from sorry

end game_duration_l194_194966


namespace value_of_a_value_of_a_in_rupees_l194_194221

theorem value_of_a (a : ℝ) (h : 0.005 * a = 70) : a = 14000 :=
by
  sorry

-- Equivalence of units conversion
theorem value_of_a_in_rupees (a : ℝ) (h : 0.005 * a = 70) : a / 100 = 140 :=
by
  have ha : a = 14000 := value_of_a a h
  rw [ha]
  exact (div_eq_mul_one_div _ _).symm.trans (mul_eq_mul_right_iff.2 (or.inl rfl))

end value_of_a_value_of_a_in_rupees_l194_194221


namespace three_digit_numbers_divisible_by_15_l194_194896

theorem three_digit_numbers_divisible_by_15 :
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 15 ∣ n}.to_finset.card = 60 :=
by
  sorry

end three_digit_numbers_divisible_by_15_l194_194896


namespace double_root_conditions_l194_194103

theorem double_root_conditions (k : ℝ) :
  (∃ x, (k - 1)/(x^2 - 1) - 1/(x - 1) = k/(x + 1) ∧ (∀ ε > 0, (∃ δ > 0, (∀ y, |y - x| < δ → (k - 1)/(y^2 - 1) - 1/(y - 1) = k/(y + 1)))))
  → k = 3 ∨ k = 1/3 :=
sorry

end double_root_conditions_l194_194103


namespace find_a_symmetric_l194_194556

def f (x : ℝ) (a : ℝ) : ℝ := log x / log 2 + a

def g (x : ℝ) : ℝ := 2^(x - 3)

theorem find_a_symmetric (a : ℝ) 
  (h : ∀ x : ℝ, f (g x) a = x ∧ g (f x a) = x) :
  a = 3 := 
sorry

end find_a_symmetric_l194_194556


namespace find_t_of_quadratic_root_l194_194838

variable (a t : ℝ)

def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ t : ℝ, Complex.ofReal a + Complex.I * 3 = Complex.ofReal a - Complex.I * 3 ∧
           (Complex.ofReal a + Complex.I * 3).re * (Complex.ofReal a - Complex.I * 3).re = t

theorem find_t_of_quadratic_root (h : quadratic_root_condition a) : t = 13 :=
sorry

end find_t_of_quadratic_root_l194_194838


namespace solve_for_k_and_j_l194_194171

theorem solve_for_k_and_j (k j : ℕ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end solve_for_k_and_j_l194_194171


namespace max_value_of_a_l194_194862

theorem max_value_of_a (a b c d : ℤ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 := by 
  sorry

end max_value_of_a_l194_194862


namespace A_and_B_finish_together_in_11_25_days_l194_194748

theorem A_and_B_finish_together_in_11_25_days (A_rate B_rate : ℝ)
    (hA : A_rate = 1/18) (hB : B_rate = 1/30) :
    1 / (A_rate + B_rate) = 11.25 := by
  sorry

end A_and_B_finish_together_in_11_25_days_l194_194748


namespace gecko_insects_eaten_l194_194001

theorem gecko_insects_eaten
    (G : ℕ)  -- Number of insects each gecko eats
    (H1 : 5 * G + 3 * (2 * G) = 66) :  -- Total insects eaten condition
    G = 6 :=  -- Expected number of insects each gecko eats
by
  sorry

end gecko_insects_eaten_l194_194001


namespace tournament_total_players_l194_194236

theorem tournament_total_players (n : ℕ) (total_points : ℕ) (total_games : ℕ) (half_points : ℕ → ℕ) :
  (∀ k, half_points k * 2 = total_points) ∧ total_points = total_games ∧
  total_points = n * (n + 11) + 132 ∧
  total_games = (n + 12) * (n + 11) / 2 →
  n + 12 = 24 :=
by
  sorry

end tournament_total_players_l194_194236


namespace mushroom_pickers_l194_194740

theorem mushroom_pickers (n : ℕ) (h : n ≥ 21) (total_mushrooms = 200) :
  ∃ i j : ℕ, i < n ∧ j < n ∧ i ≠ j ∧ (number_of_mushrooms i + number_of_mushrooms j = total_mushrooms) :=
sorry

end mushroom_pickers_l194_194740


namespace sort_sequence_by_swaps_l194_194982

theorem sort_sequence_by_swaps (k n : ℕ) (hk : 1 ≤ k) (hn : 1 ≤ n) (hrel : Nat.gcd k n = 1) 
  (l : List ℕ) (hl : l = List.range (k + n + 1))
  (hperm : l ~ l.perm) :
  ∃ swaps : (ℕ × ℕ) → List ℕ → List ℕ, 
    (∀ (a b : ℕ), a - b = k ∨ b - a = k ∨ a - b = n ∨ b - a = n → 
      ∃ l', l' = swaps (a, b) l) →
    (swaps (permutations_needed l)).perm = List.range' 1 (k + n) := 
sorry

end sort_sequence_by_swaps_l194_194982


namespace maximal_sum_l194_194618

-- Define the conditions given in the problem
variables {n : ℕ} (h₁ : n ≥ 3)
variables (a : ℕ → ℝ) (h₂ : ∀ i, a i ≤ 1)
variables (h₃ : ∀ i, a i = |a (i-1) - a (i-2)|)

-- Define the statement to be proved
theorem maximal_sum (h_div3 : n % 3 = 0) : ∑ i in range n, a i = (2 / 3 : ℝ) * n ∨ ∑ i in range n, a i = 0 :=
sorry

end maximal_sum_l194_194618


namespace angle_HK_median_AHB_l194_194300

noncomputable theory

variables {A B C H K : Type}
variables [triangle A B C]
variables [equilateral_triangle A B C]
variables [right_triangle A H B]
variables (h₁ : angle H B A = 60)
variables (h₂ : on_ray K B C)
variables (h₃ : angle C A K = 15)

theorem angle_HK_median_AHB : angle_between HK (median A H B) = 15 :=
sorry

end angle_HK_median_AHB_l194_194300


namespace width_of_square_is_sqrt11_l194_194764

noncomputable def width_of_square : Real :=
  let area_rectangle := 3 * 6
  let diff_area := 7
  let area_square := area_rectangle - diff_area
  Real.sqrt area_square

theorem width_of_square_is_sqrt11 : width_of_square = Real.sqrt 11 := by
  have h1 : 3 * 6 = 18 := by norm_num
  have h2 : 18 - 7 = 11 := by norm_num
  show width_of_square = Real.sqrt 11
  unfold width_of_square
  rw [h1, h2]
  rfl

end width_of_square_is_sqrt11_l194_194764


namespace range_of_m_l194_194690

variable {f : ℝ → ℝ}

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem range_of_m 
  (h₁ : is_odd f) 
  (h₂ : ∀ x, -2 ≤ x ∧ x ≤ 2 → (x ∈ (Ioc (-2 : ℝ) 2)) → ∀ y, x < y → f y < f x) 
  (h₃ : ∀ m, -2 ≤ m ∧ m ≤ 2 → f (1 + m) + f m < 0) :
  ∀ m, \((-\frac{1}{2} < m ∧ m ≤ 1)) :=
begin
  sorry 
end

end range_of_m_l194_194690


namespace correct_statement_l194_194034

-- Definitions for the geometric characteristics of pyramids and frustums
def is_pyramid (body : Type) : Prop :=
  ∃ base triangle_faces vertex, 
    (∀ face ∈ triangle_faces, face.has_common_vertex vertex) ∧
    (body.consists_of base triangle_faces)

def is_frustum (body : Type) (pyramid : Type) (cutting_plane : Type) : Prop :=
  (cutting_plane.parallel_to (pyramid.base)) ∧
  (body.formed_between (pyramid.base) cutting_plane)

def is_tetrahedron (body : Type) : Prop :=
  body.is_triangular_pyramid

-- Theorem stating the correct statement
theorem correct_statement : (∀ body : Type, 
  (is_pyramid body → false) → 
  (is_frustum body → false) → 
  (is_tetrahedron body → true)) :=
by {
  -- Proof is skipped
  sorry
}

end correct_statement_l194_194034


namespace smallest_positive_period_of_f_l194_194117

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 3) - 1

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 2 * π :=
by sorry

end smallest_positive_period_of_f_l194_194117


namespace max_super_bishops_l194_194380

structure SuperBishop where
  position : (ℕ × ℕ) -- Board position as a pair (row, column)

def attacks (A B : SuperBishop) : Prop :=
  (A.position.1 + A.position.2 = B.position.1 + B.position.2 ∨
   A.position.1 - A.position.2 = B.position.1 - B.position.2) ∧
  ∀ P, ∃ C: SuperBishop, (C.position = P) → (C = A ∨ C = B ∨ P = (B.position.1 + 1, B.position.2 + 1) ∨ P = (B.position.1 - 1, B.position.2 - 1))

def valid_attack_config (S : Finset SuperBishop) : Prop :=
  ∀ A ∈ S, ∃ B ∈ S, A ≠ B ∧ attacks A B

theorem max_super_bishops :
  ∃ (S : Finset SuperBishop), valid_attack_config S ∧ S.card = 32 :=
sorry

end max_super_bishops_l194_194380


namespace complete_square_monomials_l194_194114

theorem complete_square_monomials (x : ℝ) :
  ∃ (m : ℝ), (m = 4 * x ^ 4 ∨ m = 4 * x ∨ m = -4 * x ∨ m = -1 ∨ m = -4 * x ^ 2) ∧
              (∃ (a b : ℝ), (4 * x ^ 2 + 1 + m = a ^ 2 + b ^ 2)) :=
sorry

-- Note: The exact formulation of the problem might vary based on the definition
-- of perfect squares and corresponding polynomials in the Lean environment.

end complete_square_monomials_l194_194114


namespace fn_periodic_l194_194509

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := λ x, 0
| 1     := λ x, sin x + cos x
| (n+2) := λ x, (deriv (f (n + 1))) x

theorem fn_periodic :
  ∀ n : ℕ, (f (n + 4)) = (f n) :=
by 
  sorry

example : f 2015 = λ x, -sin x - cos x := by
  have h_periodic := fn_periodic 2011
  simp [f, h_periodic]
  sorry

end fn_periodic_l194_194509


namespace city_population_l194_194412

theorem city_population (p : ℝ) (hp : 0.85 * (p + 2000) = p + 2050) : p = 2333 :=
by
  sorry

end city_population_l194_194412


namespace manu_wins_l194_194614

noncomputable def prob_head : ℚ := 1 / 2
noncomputable def prob_consecutive_heads : ℚ := prob_head * prob_head
noncomputable def prob_no_heads_turn : ℚ := prob_head

theorem manu_wins (P : ℚ := 1 / 30) :
  let prob_first_win := prob_no_heads_turn ^ 3 * prob_consecutive_heads,
      prob_nth_win := λ n: ℕ, prob_no_heads_turn ^ (4 * n) * prob_consecutive_heads,
      total_prob := ∑' n, prob_nth_win(n+1)
  in total_prob = P := sorry

end manu_wins_l194_194614


namespace pedestrian_avg_waiting_time_at_traffic_light_l194_194436

theorem pedestrian_avg_waiting_time_at_traffic_light :
  ∀ cycle_time green_time red_time : ℕ,
  cycle_time = green_time + red_time →
  green_time = 1 →
  red_time = 2 →
  let prob_green := green_time / cycle_time in
  let prob_red := red_time / cycle_time in
  let E_T_given_green := 0 in
  let E_T_given_red := (0 + 2) / 2 in
  (E_T_given_green * prob_green + E_T_given_red * prob_red) * 60 = 40 := 
by
  -- Insert proof here
  sorry

end pedestrian_avg_waiting_time_at_traffic_light_l194_194436


namespace probability_delegates_l194_194374

theorem probability_delegates (total_ways undesired_ways : ℕ)
    (h_total : total_ways = 79833600)
    (h_undesired : undesired_ways = 40652136)
    (h_fraction : ∀ m n : ℕ, m = 4897683 → n = 9979200 → m + n = 14876883) :
    m + n = 14876883 :=
by
  have total_ways - undesired_ways = 39181464,
    from calc
      total_ways - undesired_ways = 79833600 - 40652136 : by rw [h_total, h_undesired]

  have probability = 39181464 / 79833600,
    from calc
      (total_ways - undesired_ways) / total_ways
      = 39181464 / 79833600 : by rw [total_ways - undesired_ways, h_total]

  have h_eq : ∃ m n, probability = m / n ∧ m = 4897683 ∧ n = 9979200,
    from exists.intro 4897683 (exists.intro 9979200 (and.intro rfl (and.intro rfl rfl)))

  exact h_fraction 4897683 9979200 rfl rfl

end probability_delegates_l194_194374


namespace greatest_median_l194_194294

-- Define the conditions in the problem
variable (n : ℕ := 9)          -- Number of nonnegative numbers
variable (avg : ℝ := 10)       -- Average of the numbers
variable (sum : ℝ := n * avg)  -- Sum of the numbers

-- Define the median as the fifth smallest number when the numbers are sorted
def is_median (m : ℝ) (lst : list ℝ) : Prop :=
  list.length lst = n ∧ 
  (∀ i < 5, 0 ≤ list.nth_le lst i sorry) ∧ -- first four numbers are nonnegative
  (∀ i ≥ 5, list.nth_le lst i sorry ≥ m)    -- the last five numbers are at least m

-- Define the proof statement
theorem greatest_median (m : ℝ) (lst : list ℝ) (h_nonneg : ∀ (x : ℝ), x ∈ lst → 0 ≤ x)
  (h_avg : list.sum lst = sum) : m ≤ 18 :=
by {
  -- proof to be filled in
  sorry
}

end greatest_median_l194_194294


namespace max_elements_in_subset_l194_194266

-- Define the set of natural numbers from 1 to 2005
def M : Finset ℕ := Finset.range (2005 + 1) \ {0}

-- Define the condition that if x is in A, then 15x is not in A
def valid_subset (A : Finset ℕ) : Prop :=
  ∀ x ∈ A, 15 * x ∉ A

-- Lean statement expressing the main assertion
theorem max_elements_in_subset :
  ∃ (A : Finset ℕ), A ⊆ M ∧ valid_subset A ∧ A.card = 1880 :=
begin
  sorry
end

end max_elements_in_subset_l194_194266


namespace cos_product_zero_l194_194679

theorem cos_product_zero (x : ℝ) :
  (sin x)^2 + (sin (3 * x))^2 + (sin (4 * x))^2 + (sin (5 * x))^2 = 2 →
  ∃ (a b c : ℕ), cos (a * x) * cos (b * x) * cos (c * x) = 0 ∧ a + b + c = 17 :=
by {
  sorry
}

end cos_product_zero_l194_194679


namespace negation_proposition_l194_194880

-- Define the initial proposition p
def proposition (n : ℝ) : Prop := ∃ a : ℝ, a ≥ -1 ∧ log (exp n + 1) > 1 / 2 

-- The statement of the theorem
theorem negation_proposition (n : ℝ) : ¬ proposition n ↔ ∀ a : ℝ, a ≥ -1 → log (exp n + 1) ≤ 1 / 2 :=
by
  sorry

end negation_proposition_l194_194880


namespace area_triangle_ABC_l194_194949

noncomputable def area_trapezoid (AB CD height : ℝ) : ℝ :=
  (AB + CD) * height / 2

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  base * height / 2

variable (AB CD height area_ABCD : ℝ)
variables (h0 : CD = 3 * AB) (h1 : area_trapezoid AB CD height = 24)

theorem area_triangle_ABC : area_triangle AB height = 6 :=
by
  sorry

end area_triangle_ABC_l194_194949


namespace jessies_weight_loss_l194_194250

-- Definitions based on the given conditions
def initial_weight : ℝ := 74
def weight_loss_rate_even_days : ℝ := 0.2 + 0.15
def weight_loss_rate_odd_days : ℝ := 0.3
def total_exercise_days : ℕ := 25
def even_days : ℕ := (total_exercise_days - 1) / 2
def odd_days : ℕ := even_days + 1

-- The goal is to prove the total weight loss is 8.1 kg
theorem jessies_weight_loss : 
  (even_days * weight_loss_rate_even_days + odd_days * weight_loss_rate_odd_days) = 8.1 := 
by
  sorry

end jessies_weight_loss_l194_194250


namespace musicians_in_band_l194_194688

theorem musicians_in_band : ∃ n : ℤ, 80 < n ∧ n < 130 ∧
  (n ≡ -1 [MOD 4]) ∧
  (n ≡ -2 [MOD 5]) ∧
  (n ≡ -3 [MOD 6]) ∧
  (n = 123) :=
by
  sorry

end musicians_in_band_l194_194688


namespace eccentricity_of_ellipse_l194_194523

theorem eccentricity_of_ellipse {a b c : ℝ} (h1 : 0 < b ∧ b < a)
  (h2 : a^2 - b^2 = c^2) (r max_r: ℝ) (h4: max_r = c / 3) (h5: ∀ P : ℝ × ℝ, P ∈ ({⟨P.1, P.2⟩ : ℝ × ℝ | P.1^2 / a^2 + P.2^2 / b^2 = 1})) :
  let e := c / a in e = 4 / 5 :=
sorry

end eccentricity_of_ellipse_l194_194523


namespace angle_A_l194_194060

-- Define the properties and conditions of the problem setup.
variables {A B C A' B' C' : Type} [Inhabited A] [Inhabited B] [Inhabited C]
          (triangle : Triangle A B C) (AngleBisector : A → B → C → Type)
          (ACB_angle : ∀ (ABC : Triangle A B C), Angle triangle C B = 120)
          (A'_prop : AngleBisector A B C = A')
          (B'_prop : AngleBisector B A C = B')
          (C'_prop : AngleBisector C A B = C')

-- Point to show the right angle assertion
def angle_ACB_120_and_bisectors : Prop :=
  ∃ (A' B' C' : Point), 
    (ACB_angle A B C = 120) ∧ 
    (A'_prop A B C = A') ∧ 
    (B'_prop B A C = B') ∧ 
    (C'_prop C A B = C') ∧ 
    (Angle A' C' B' = 90)

-- The theorem statement, asserting the angle is 90 degrees.
theorem angle_A'C'B'_is_90 
  (A B C A' B' C' : Point) 
  (triangle : Triangle A B C) 
  (AngleBisector : A → B → C → Type) 
  (ACB_angle : ∀ (ABC : Triangle A B C), Angle triangle C B = 120) 
  (A'_prop : AngleBisector A B C = A') 
  (B'_prop : AngleBisector B A C = B') 
  (C'_prop : AngleBisector C A B = C') :
  Angle A' C' B' = 90 := 
sorry

end angle_A_l194_194060


namespace midpoints_are_collinear_l194_194762

variables {A B C D E F M N P : Type*}
variables (triangle_ABC : Triangle A B C)
variables (D : Point) [OnLine D (Line A B)]
variables (E : Point) [OnLine E (Line B C)]
variables (F : Point) [OnLine F (Line.extended A C)]

variables (M : Point) [Midpoint M D C]
variables (N : Point) [Midpoint N A E]
variables (P : Point) [Midpoint P B F]

theorem midpoints_are_collinear : Collinear {M, N, P} :=
sorry

end midpoints_are_collinear_l194_194762


namespace four_digit_square_palindromes_are_zero_l194_194180

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

-- Define the main theorem statement
theorem four_digit_square_palindromes_are_zero : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) → 
             is_palindrome n → 
             (∃ m : ℕ, n = m * m) → 
             n = 0 :=
by
  sorry

end four_digit_square_palindromes_are_zero_l194_194180


namespace each_player_gets_seven_l194_194964

-- Define the total number of dominoes and players
def total_dominoes : Nat := 28
def total_players : Nat := 4

-- Define the question for how many dominoes each player would receive
def dominoes_per_player (dominoes players : Nat) : Nat := dominoes / players

-- The theorem to prove each player gets 7 dominoes
theorem each_player_gets_seven : dominoes_per_player total_dominoes total_players = 7 :=
by
  sorry

end each_player_gets_seven_l194_194964


namespace determine_a_l194_194856

variable {α : Type*} [linear_ordered_field α]

def odd_function (f : α → α) := ∀ x, f (-x) = -f x

theorem determine_a (f : α → α) (a : α) 
  (h1 : odd_function f)
  (h2 : ∀ x, x < 0 → f x = -exp (a * x))
  (h3 : f (log 2) = 8) :
  a = -3 :=
sorry

end determine_a_l194_194856


namespace point_equidistant_from_axes_l194_194606

theorem point_equidistant_from_axes (x : ℝ) :
  ((|x - 6| = |-2x|) → (x = 2 ∨ x = -6)) :=
sorry

end point_equidistant_from_axes_l194_194606


namespace jessica_dice_problem_l194_194967

noncomputable def probability_of_third_six (p q : ℕ) (h : Nat.coprime p q) :
  p = 109 ∧ q = 148 := by
  sorry

theorem jessica_dice_problem :
  probability_of_third_six 109 148 (Nat.coprime_intro 1 109 148) :=
  by sorry

end jessica_dice_problem_l194_194967


namespace find_y_six_l194_194425

theorem find_y_six (y : ℝ) (h : y > 0) (h_eq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
    y^6 = 116 / 27 :=
by
  sorry

end find_y_six_l194_194425


namespace cost_of_plastering_tank_l194_194731

theorem cost_of_plastering_tank
  (l w d : ℝ) 
  (plaster_cost_per_sqm : ℝ)
  (hl : l = 25)
  (hw : w = 12)
  (hd : d = 6)
  (hp : plaster_cost_per_sqm = 0.75) :
  (l * w + 2 * l * d + 2 * w * d) * plaster_cost_per_sqm = 558 := 
by {
  rw[hl, hw, hd, hp],
  simp,
  norm_num,
  sorry
}

end cost_of_plastering_tank_l194_194731


namespace james_main_job_hours_eq_30_l194_194959

-- Define the conditions
def main_job_hourly_wage : ℝ := 20 -- James earns $20 per hour at his main job
def second_job_hourly_wage (m : ℝ) : ℝ := m - 0.20 * m -- James earns 20% less at his second job
def half_hours_at_second_job (H : ℝ) : ℝ := H / 2 -- Works half the time in second job
def total_earnings (H : ℝ) (w : ℝ) : ℝ := 20 * H + (w * (H / 2)) -- Total weekly earnings

-- Prove that the number of hours James works at his main job equals 30, given he earns $840 per week
theorem james_main_job_hours_eq_30 :
  ∃ H : ℝ, total_earnings H (second_job_hourly_wage main_job_hourly_wage) = 840 ∧ H = 30 :=
begin
  sorry
end

end james_main_job_hours_eq_30_l194_194959


namespace find_c_eq_neg_9_over_4_l194_194682

theorem find_c_eq_neg_9_over_4 (c x : ℚ) (h₁ : 3 * x + 5 = 1) (h₂ : c * x - 8 = -5) :
  c = -9 / 4 :=
sorry

end find_c_eq_neg_9_over_4_l194_194682


namespace product_of_functions_l194_194160

noncomputable def f (x : ℝ) : ℝ := x^2 / real.sqrt (x + 1)
noncomputable def g (x : ℝ) : ℝ := real.sqrt (x + 1) / x

theorem product_of_functions (x : ℝ) : 
  f x * g x = x :=
by 
  have h_dom : x ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioi (0 : ℝ), from sorry
  exact sorry

end product_of_functions_l194_194160


namespace stability_of_triangle_prevents_deformation_l194_194728

/-- A structure representing a rectangular door frame with a diagonal wooden stick. --/
structure DoorFrame :=
  (a b c d : ℝ)
  (rectangular : a = c ∧ b = d)
  (diagonal : ℝ → (a:ℝ) = (b:ℝ) → Prop)

/-- Stability of triangles makes an object resistant to deformation. --/
theorem stability_of_triangle_prevents_deformation (df : DoorFrame) :
  (df.diagonal (df.a)) = (df.diagonal (df.b)) →
  Triangle (df.a, df.b, diagonal) → stable := sorry

end stability_of_triangle_prevents_deformation_l194_194728


namespace sequence_sum_l194_194790

theorem sequence_sum : 
  let s : ℕ → ℤ := λ n, if even (n+1) then -↑(n+1) else ↑(n+1) in
  (∑ n in (Finset.range 1998), s n) = -333 := by
  sorry

end sequence_sum_l194_194790


namespace range_of_a_l194_194547

theorem range_of_a (a : ℝ) (h : a > 0) : (∀ x : ℝ, x > 0 → 9 * x + a^2 / x ≥ a^2 + 8) → 2 ≤ a ∧ a ≤ 4 :=
by
  intros h1
  sorry

end range_of_a_l194_194547


namespace max_a_l194_194861

theorem max_a {a b c d : ℤ} (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 :=
by {
  have h_b : b ≤ 3 * c - 1 := by linarith,
  have h_c : c ≤ 4 * d - 1 := by linarith,
  have h_d : d ≤ 99 := by linarith,
  have h_max_a := calc
    a ≤ 2 * b - 1 : by linarith
    ... ≤ 2 * (3 * c - 1) - 1 : by linarith
    ... ≤ 6 * c - 3 : by linarith
    ... ≤ 6 * (4 * d - 1) - 3 : by linarith
    ... ≤ 24 * d - 9 : by linarith
    ... ≤ 24 * 99 - 9 : by linarith
    ... = 2367 : by norm_num,
  exact h_max_a,
  sorry
}

end max_a_l194_194861


namespace num_four_digit_palindromic_squares_is_two_l194_194195

open Nat

-- Define the condition for a palindrome
def is_palindrome (n : ℕ) : Prop :=
  to_digits 10 n = (to_digits 10 n).reverse

-- Define the range of numbers to check
def range_32_to_99 := {x : ℕ | 32 ≤ x ∧ x ≤ 99}

-- Define the function to compute the square of a number
def square (n : ℕ) : ℕ := n * n

-- Define the set of 4-digit squares that are palindromes
def four_digit_palindromic_squares : Finset ℕ :=
  (Finset.filter (λ n => is_palindrome n) (Finset.image square (Finset.filter (λ n => 1000 ≤ square n ∧ square n < 10000) 
  (Finset.filter (λ n => n ∈ range_32_to_99) (Finset.range 100)))))

-- The main theorem stating the number of 4-digit palindromic squares
theorem num_four_digit_palindromic_squares_is_two :
  four_digit_palindromic_squares.card = 2 := sorry

end num_four_digit_palindromic_squares_is_two_l194_194195


namespace find_f_f_neg2_l194_194116

def f (x : ℤ) : ℤ := 
  if x >= 0 then x + 2 else x^2

theorem find_f_f_neg2 : f(f(-2)) = 6 := 
by
  sorry

end find_f_f_neg2_l194_194116


namespace radical_axis_midpoint_l194_194986

variables {C C' : Circle} {A B I : Point}
variable {Δ : Line}

-- Given conditions
def is_radical_axis (Δ : Line) (C C' : Circle) : Prop := 
  ∀ P : Point, power P C = power P C'

def is_tangent (L : Line) (C : Circle) (P : Point) : Prop := 
  L.is_tangent_at C P

def midpoint (P Q R : Point) : Prop :=
  dist P R = dist Q R

-- The statement to prove
theorem radical_axis_midpoint (h_radical_axis : is_radical_axis Δ C C')
  (h_tangent_A : is_tangent (Line_through A B) C A)
  (h_tangent_B : is_tangent (Line_through A B) C' B)
  (h_intersection : I ∈ Δ ∧ I ∈ Line_segment A B) :
  midpoint I A B := 
sorry

end radical_axis_midpoint_l194_194986


namespace train_passes_tree_in_given_time_l194_194437

noncomputable def train_passing_time : ℝ := 
  let train_length_m : ℝ := 850   -- Length of the train in meters
  let train_speed_kmh : ℝ := 85   -- Speed of the train in km/hr
  let wind_speed_kmh : ℝ := 5     -- Speed of the wind in km/hr
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)  -- Convert train's speed to m/s
  let wind_speed_ms : ℝ := wind_speed_kmh * (1000 / 3600)    -- Convert wind's speed to m/s
  let effective_speed_ms : ℝ := train_speed_ms - wind_speed_ms  -- Effective speed of the train considering the wind
  train_length_m / effective_speed_ms  -- Time for the train to pass the tree in seconds

theorem train_passes_tree_in_given_time : train_passing_time ≈ 38.25 := sorry

end train_passes_tree_in_given_time_l194_194437


namespace evaluate_expression_l194_194809

theorem evaluate_expression : ( (5 : ℝ) / 6 ) ^ 4 * ( (5 : ℝ) / 6 ) ^ (-4) = 1 :=
by {
  sorry
}

end evaluate_expression_l194_194809


namespace probability_of_median_5_eq_2_div_7_l194_194757

def count_combinations (n k : ℕ) : ℕ := nat.choose n k

def favorable_outcomes : ℕ := count_combinations 4 2 * count_combinations 4 2

def total_outcomes : ℕ := count_combinations 9 5

def probability_median_is_5 : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_median_5_eq_2_div_7 :
  probability_median_is_5 = 2 / 7 :=
by sorry

end probability_of_median_5_eq_2_div_7_l194_194757


namespace Mark_James_dice_problem_l194_194646

theorem Mark_James_dice_problem : 
  let mark_total_dice := 10
  let mark_percent_12_sided := 0.6
  let james_total_dice := 8
  let james_percent_12_sided := 0.75
  let total_needed := 14
  let mark_12_sided := mark_total_dice * mark_percent_12_sided
  let james_12_sided := james_total_dice * james_percent_12_sided
  let total_12_sided := mark_12_sided + james_12_sided
  let dice_to_buy := total_needed - total_12_sided
  ⟶ dice_to_buy = 2 := by sorry

end Mark_James_dice_problem_l194_194646


namespace standard_ellipse_eqn_length_of_PQ_area_triangle_OPQ_l194_194871

section EllipseProof

variables (a b : ℝ) (h_pos : a > b) (h_order : b > 0)

/-- The standard equation of the ellipse using the given parameters. -/
def ellipse_eqn := ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_ellipse_eqn (h₁ : b = sqrt 2) (h₂ : a = 2) :
  ellipse_eqn 2 (sqrt 2) = ∀ (x y : ℝ), (x^2 / 4) + (y^2 / 2) = 1 :=
sorry

theorem length_of_PQ (m : ℝ) (h : m = 1) (PQ : ℝ) :
  PQ = (4 * sqrt 5) / 3 :=
sorry

theorem area_triangle_OPQ (m : ℝ) :
  ∃ m, ∀ (x y : ℝ), ((1 / 2) * ((4 / 3) * sqrt (6 - m^2)) * (abs m / sqrt 2) = (4 / 3)) ∧
  (m = 2 ∨ m = -2 ∨ m = sqrt 2 ∨ m = -sqrt 2) :=
sorry

end EllipseProof

end standard_ellipse_eqn_length_of_PQ_area_triangle_OPQ_l194_194871


namespace trajectory_is_ellipse_range_of_DH_MN_ratio_l194_194527

-- Condition for trajectory of P
def condition_trajectory (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  √((x+2)^2 + y^2) + √((x-2)^2 + y^2) = 4*√2

-- Equation of trajectory C
def equation_C (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / 8) + (y^2 / 4) = 1

-- Proving equivalence of condition and equation
theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : condition_trajectory P) : equation_C P :=
  sorry

-- Definition and conditions for line l, points M, N, and properties of H, D
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x - 2)

def points (M N : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  -- Points M, N lie on the trajectory and line l
  equation_C M ∧ equation_C N ∧ M.2 = l M.1 ∧ N.2 = l N.1

def D_H_MN (H D M N : ℝ × ℝ) (k : ℝ) : Prop :=
  -- Definitions of H, D and |DH|, |MN|
  let x1 := M.1
  let y1 := M.2
  let x2 := N.1
  let y2 := N.2
  let l := λ x, k * (x - 2)
  let midpoint := (M.1 + N.1) / 2, (M.2 + N.2) / 2
  let perp_bis := λ x, - (1 / k) * (x - midpoint.1) + midpoint.2
  points M N l ∧
  H = ((x1 + x2) / 2, (y1 + y2) / 2) ∧
  D.2 = 0 ∧ 
  D = (2 * k^2 / (1+2*k^2), 0) ∧
  let dist_H_D := √((H.1 - D.1)^2 + (H.2 - D.2)^2) in
  let dist_M_N := √(1 + k^2) * √((x1 + x2)^2 - 4 * x1 * x2) in
  let ratio := dist_H_D / dist_M_N in
  ratio ∈ (0, √2 / 4)

-- Proving the range of the ratio dist_H_D / dist_M_N
theorem range_of_DH_MN_ratio (H D M N : ℝ × ℝ) (k : ℝ) (h : D_H_MN H D M N k) :
  let ratio := √((H.1 - D.1)^2 + (H.2 - D.2)^2) / (√(1 + k^2) * √((M.1 + N.1)^2 - 4 * M.1 * N.1)) in
  ratio ∈ (0, √2 / 4) :=
  sorry

end trajectory_is_ellipse_range_of_DH_MN_ratio_l194_194527


namespace six_divides_p_plus_one_l194_194660

theorem six_divides_p_plus_one 
  (p : ℕ) 
  (prime_p : Nat.Prime p) 
  (gt_three_p : p > 3) 
  (prime_p_plus_two : Nat.Prime (p + 2)) 
  (gt_three_p_plus_two : p + 2 > 3) : 
  6 ∣ (p + 1) := 
sorry

end six_divides_p_plus_one_l194_194660


namespace solve_series_eq_100_l194_194806

theorem solve_series_eq_100 (x : ℝ) (h_conv : -1 < x ∧ x < 1) :
  2 + 7*x + 12*x^2 + 17*x^3 + ∑' (n : ℕ), (5*n + 2) * x^n = 100 ↔ x = 14 / 25 :=
begin
  sorry
end

end solve_series_eq_100_l194_194806


namespace geometric_sequence_property_l194_194593

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom: ∀ n, a (n + 1) = a n * r) 
  (h_pos: ∀ n, a n > 0)
  (h_root1: a 3 * a 15 = 8)
  (h_root2: a 3 + a 15 = 6) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_property_l194_194593


namespace cuboid_surface_area_is_94_l194_194817

-- Define the dimensions of the cuboid
def width : ℝ := 3
def length : ℝ := 4
def height : ℝ := 5

-- Define the surface area calculation for the cuboid
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width) + 2 * (width * height) + 2 * (length * height)

-- Prove that the surface area of the given cuboid is 94 cm^2
theorem cuboid_surface_area_is_94 :
  surface_area length width height = 94 := by
  sorry

end cuboid_surface_area_is_94_l194_194817


namespace maryville_population_increase_l194_194442

def average_people_added_per_year (P2000 P2005 : ℕ) (period : ℕ) : ℕ :=
  (P2005 - P2000) / period
  
theorem maryville_population_increase :
  let P2000 := 450000
  let P2005 := 467000
  let period := 5
  average_people_added_per_year P2000 P2005 period = 3400 :=
by
  sorry

end maryville_population_increase_l194_194442


namespace operation_addition_l194_194654

theorem operation_addition (a b c : ℝ) (op : ℝ → ℝ → ℝ)
  (H : ∀ a b c : ℝ, op (op a b) c = a + b + c) :
  ∀ a b : ℝ, op a b = a + b :=
sorry

end operation_addition_l194_194654


namespace problem_Q_value_l194_194997

theorem problem_Q_value (n : ℕ) (h : n = 2010) :
  (∏ k in finset.range (n - 1), 1 - (1 : ℚ) / (k + 3)) = 2 / (2011 : ℚ) :=
  sorry

end problem_Q_value_l194_194997


namespace num_four_digit_palindromic_squares_is_two_l194_194197

open Nat

-- Define the condition for a palindrome
def is_palindrome (n : ℕ) : Prop :=
  to_digits 10 n = (to_digits 10 n).reverse

-- Define the range of numbers to check
def range_32_to_99 := {x : ℕ | 32 ≤ x ∧ x ≤ 99}

-- Define the function to compute the square of a number
def square (n : ℕ) : ℕ := n * n

-- Define the set of 4-digit squares that are palindromes
def four_digit_palindromic_squares : Finset ℕ :=
  (Finset.filter (λ n => is_palindrome n) (Finset.image square (Finset.filter (λ n => 1000 ≤ square n ∧ square n < 10000) 
  (Finset.filter (λ n => n ∈ range_32_to_99) (Finset.range 100)))))

-- The main theorem stating the number of 4-digit palindromic squares
theorem num_four_digit_palindromic_squares_is_two :
  four_digit_palindromic_squares.card = 2 := sorry

end num_four_digit_palindromic_squares_is_two_l194_194197


namespace eval_expression_l194_194482

theorem eval_expression : (503 * 503 - 502 * 504) = 1 :=
by
  sorry

end eval_expression_l194_194482


namespace erin_paths_in_tetrahedron_l194_194478

   theorem erin_paths_in_tetrahedron : ∀ (V : Type) (start : V) (vertices : set V) (edges : set (V × V)), 
     vertices.card = 4 → 
     (∀ v ∈ vertices, ∃! u ∈ vertices, (u, v) ∈ edges ∨ (v, u) ∈ edges) →
     {p : list V | p.head = start ∧ p.nodup ∧ p.length = 4 ∧ 
                     ∀ i, i < 3 → (p.nth_le i sorry, p.nth_le (i+1) sorry) ∈ edges ∨ (p.nth_le (i+1) sorry, p.nth_le i sorry) ∈ edges}.card = 6 :=
   by
     sorry
   
end erin_paths_in_tetrahedron_l194_194478


namespace find_f_of_7_6_l194_194992

-- Definitions from conditions
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x k : ℤ, f (x + T * (k : ℝ)) = f x

def f_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = x

-- The periodic function f with period 4
def f : ℝ → ℝ := sorry

-- Hypothesis
axiom f_periodic : periodic_function f 4
axiom f_on_interval : f_in_interval f

-- Theorem to prove
theorem find_f_of_7_6 : f 7.6 = 3.6 :=
by
  sorry

end find_f_of_7_6_l194_194992


namespace vector_calculation_l194_194900

-- Define the vectors a, b, and c
def a : ℝ × ℝ × ℝ := (2, 0, 1)
def b : ℝ × ℝ × ℝ := (-3, 1, -1)
def c : ℝ × ℝ × ℝ := (1, 1, 0)

-- Define the expression and the expected result
def expression := (a.1 + 2 * b.1 - 3 * c.1, a.2 + 2 * b.2 - 3 * c.2, a.3 + 2 * b.3 - 3 * c.3)
def expected_result := (-7, -1, -1)

-- The proof statement
theorem vector_calculation : expression = expected_result :=
by {
  -- Proof steps would go here
  sorry
}

end vector_calculation_l194_194900


namespace ferryP_speed_l194_194834

variable (v_P v_Q : ℝ)
variable (h1 : v_Q = v_P + 3)
variable (h2 : ∀ tP tQ, tP = 3 → tQ = tP + 1 → 
  v_Q * tQ = 2 * (v_P * tP))

theorem ferryP_speed : v_P = 6 :=
by
  have tP := 3
  have tQ := 4 
  have h1 : v_Q = v_P + 3 := h1
  have h2 := h2 tP tQ (by rfl) (by rfl)
  sorry

end ferryP_speed_l194_194834


namespace sum_divisible_by_100_l194_194848

/-- Given 100 positive integers, it is possible to select one or more of them 
    such that the selected number or the sum of the selected numbers is divisible by 100. -/
theorem sum_divisible_by_100 (a : Fin 100 → ℕ) (h_pos : ∀ i, 0 < a i) :
  ∃ (s : Finset (Fin 100)), (0 < s.card) ∧ (100 ∣ (s.sum (λ i, a i))) :=
by
  sorry

end sum_divisible_by_100_l194_194848


namespace scorpion_segments_daily_total_l194_194002

theorem scorpion_segments_daily_total (seg1 : ℕ) (seg2 : ℕ) (additional : ℕ) (total_daily : ℕ) :
  (seg1 = 60) →
  (seg2 = 2 * seg1 * 2) →
  (additional = 10 * 50) →
  (total_daily = seg1 + seg2 + additional) →
  total_daily = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end scorpion_segments_daily_total_l194_194002


namespace shannon_bracelets_l194_194311

theorem shannon_bracelets (total_stones : ℝ) (stones_per_bracelet : ℝ) (h1 : total_stones = 48) (h2 : stones_per_bracelet = 8) : total_stones / stones_per_bracelet = 6 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end shannon_bracelets_l194_194311


namespace cost_of_two_pans_is_20_l194_194972

variable (cost_of_pan : ℕ)

-- Conditions
def pots_cost := 3 * 20
def total_cost := 100
def pans_eq_cost := total_cost - pots_cost
def cost_of_pan_per_pans := pans_eq_cost / 4

-- Proof statement
theorem cost_of_two_pans_is_20 
  (h1 : pots_cost = 60)
  (h2 : total_cost = 100)
  (h3 : pans_eq_cost = total_cost - pots_cost)
  (h4 : cost_of_pan_per_pans = pans_eq_cost / 4)
  : 2 * cost_of_pan_per_pans = 20 :=
by sorry

end cost_of_two_pans_is_20_l194_194972


namespace John_age_l194_194075

theorem John_age (Drew Maya Peter John Jacob : ℕ)
  (h1 : Drew = Maya + 5)
  (h2 : Peter = Drew + 4)
  (h3 : John = 2 * Maya)
  (h4 : (Jacob + 2) * 2 = Peter + 2)
  (h5 : Jacob = 11) : John = 30 :=
by 
  sorry

end John_age_l194_194075


namespace benjamin_decade_expense_l194_194789

-- Define the constants
def yearly_expense : ℕ := 3000
def years : ℕ := 10

-- Formalize the statement
theorem benjamin_decade_expense : yearly_expense * years = 30000 := 
by
  sorry

end benjamin_decade_expense_l194_194789


namespace num_words_polarized_l194_194899

theorem num_words_polarized : 
    let letters := ["p", "o", "l", "a", "r", "i", "z", "e", "d"] in
    letters.length = 9 → 
    (Nat.factorial 9) = 362880 := 
by
    intros letters_len
    -- Proof steps would go here
    sorry

end num_words_polarized_l194_194899


namespace invertible_function_labels_product_is_60_l194_194870

theorem invertible_function_labels_product_is_60 :
  let domain1 := {-6, -5, -4, -3, -2, -1, 0, 1}
  let function1 := function (x : ℤ) ∈ domain1

  let domain2 := set.Icc (-3 : ℝ) 3
  let function2 := λ x : ℝ, x^2 - 3*x + 2

  let domain3 := { x : ℝ | (-8 ≤ x ∧ x ≤ -0.5) ∨ (0.5 ≤ x ∧ x ≤ 8) }
  let function3 := λ x : ℝ, 1 / x

  let domain4 := set.Icc 0 Real.pi
  let function4 := λ x : ℝ, Real.sin x

  let domain5 := { x : ℝ | x ≠ 1 }
  let function5 := λ x : ℝ, 2 * x + 1

  (∃ f1_inv : function1.invertible, 
   ∃ f3_inv : function3.invertible, 
   ∃ f4_inv : function4.invertible, 
   ∃ f5_inv : function5.invertible) →
  1 * 3 * 4 * 5 = 60 :=
by
  intros
  have f1_inv := -/4 sorry
  have f3_inv := -/4 sorry
  have f4_inv := -/4 sorry
  have f5_inv := -/4 sorry
  calc
    1 * 3 * 4 * 5 = 1 * 3 * (4 * 5) := by sorry
    ... = 1 * 3 * 20 := by sorry
    ... = 60 := by sorry

end invertible_function_labels_product_is_60_l194_194870


namespace minimum_value_f_l194_194088

noncomputable def f (x : ℝ) := x * Real.exp x

theorem minimum_value_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 / Real.exp 1 :=
by
  use -1
  simp
  -- given the conditions from the problem
  -- (1) Compute the derivative: y' = e^x + xe^x
  -- (2) Solve y' = 0 to find critical points
  -- (3) Determine the minimum value
  sorry

end minimum_value_f_l194_194088


namespace ratio_CD_BD_two_l194_194127

-- Define the points and their relationships
variables {A B C E D : Type} [RightTriangle ABC B 90]

-- Let E be the midpoint of AC
def midpoint_AC (E : Type) [Midpoint E AC] : Prop := true

-- D is on BC such that \angle ADB = \angle EDC
def angle_conditions (D E : Type) [OnLine D BC] [OnLine E AC] [AngleEqual (∠ ADB) (∠ EDC)] : Prop := true

-- Prove the ratio CD : BD
theorem ratio_CD_BD_two (h_triangle : RightTriangle ABC B 90)
                        (h_midpoint : midpoint_AC E)
                        (h_angles : angle_conditions D E) :
    (CD / BD) = 2 := 
sorry

end ratio_CD_BD_two_l194_194127


namespace find_angle_D_l194_194140

theorem find_angle_D
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A + angle_B = 180)
  (h2 : angle_C = 2 * angle_D)
  (h3 : angle_A = 100)
  (h4 : angle_B + angle_C + angle_D = 180) :
  angle_D = 100 / 3 :=
by
  sorry

end find_angle_D_l194_194140


namespace range_of_m_l194_194889

def M (x : ℝ) : Prop := x^2 + 5 * x - 14 < 0

def N (x m : ℝ) : Prop := m < x ∧ x < m + 3

theorem range_of_m (m : ℝ) : (∀ x, ¬ (M x ∧ N x m)) ↔ (m ∈ Iic (-10) ∪ Ici 2) :=
by sorry

end range_of_m_l194_194889


namespace car_dealer_bmw_sales_l194_194749

theorem car_dealer_bmw_sales (total_cars : ℕ)
  (vw_percentage : ℝ)
  (toyota_percentage : ℝ)
  (acura_percentage : ℝ)
  (bmw_count : ℕ) :
  total_cars = 300 →
  vw_percentage = 0.10 →
  toyota_percentage = 0.25 →
  acura_percentage = 0.20 →
  bmw_count = total_cars * (1 - (vw_percentage + toyota_percentage + acura_percentage)) →
  bmw_count = 135 :=
by
  intros
  sorry

end car_dealer_bmw_sales_l194_194749


namespace line_passing_through_midpoint_of_arc_l194_194713

open EuclideanGeometry

theorem line_passing_through_midpoint_of_arc
    (O : Point) (A B : Point) (hO_eq : O ≠ A ∧ O ≠ B) 
    (O1 O2 : Circle) (M N : Point) (C : Point)
    (h1 : O1.Inside (Line.segment A B))
    (h2 : O2.Inside (Line.segment A B))
    (h3 : (O1.IntersectionPoints (O2)).Is_some) 
    (h4 : (O1.IntersectionPoints (O2)).Some = (M, N))
    (hC : IsMidpointArc O A B C) :
    isCollinear M N C := sorry

end line_passing_through_midpoint_of_arc_l194_194713


namespace max_value_trig_l194_194822

theorem max_value_trig (x : ℝ) : ∃ y : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ y := by
  use 5
  sorry

end max_value_trig_l194_194822


namespace malia_berries_second_bush_l194_194639

theorem malia_berries_second_bush :
  ∀ (b2 : ℕ), ∃ (d1 d2 d3 d4 : ℕ),
  d1 = 3 → d2 = 7 → d3 = 12 → d4 = 19 →
  d2 - d1 = (d3 - d2) - 2 →
  d3 - d2 = (d4 - d3) - 2 →
  b2 = d1 + (d2 - d1 - 2) →
  b2 = 6 :=
by
  sorry

end malia_berries_second_bush_l194_194639


namespace tim_weekly_earnings_l194_194708

-- Definitions based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- The theorem that we need to prove
theorem tim_weekly_earnings :
  (tasks_per_day * pay_per_task * days_per_week : ℝ) = 720 :=
by
  sorry -- Skipping the proof

end tim_weekly_earnings_l194_194708


namespace open_cold_tap_at_7_minutes_l194_194673

-- Defining the conditions
variable (t_hot t_cold : ℕ) -- time in minutes to fill with hot and cold water
variable (ratio_hot_to_cold : ℝ) -- ratio of hot water to cold water

-- Values from the conditions
def hot_fill_time : ℕ := 23
def cold_fill_time : ℕ := 17
def required_ratio : ℝ := 1.5

-- Define the total volume
def total_volume : ℝ := 1

-- Define the volumes V_hot and V_cold based on the proportion
def V_cold (V_hot : ℝ) : ℝ := V_hot / ratio_hot_to_cold
def V_hot_time (V_hot : ℝ) : ℝ := V_hot * hot_fill_time
def V_cold_time (V_hot : ℝ) : ℝ := (V_hot / ratio_hot_to_cold) * cold_fill_time

noncomputable def delay_time : ℝ := V_hot_time total_volume - V_cold_time total_volume

theorem open_cold_tap_at_7_minutes (V_hot : ℝ) : 
  t_hot = hot_fill_time → 
  t_cold = cold_fill_time → 
  ratio_hot_to_cold = required_ratio → 
  delay_time = 7 := by 
  intros
  sorry

end open_cold_tap_at_7_minutes_l194_194673


namespace transformed_graph_vertically_compressed_and_shifted_up_l194_194876

def f : ℝ → ℝ
| x if -3 ≤ x ∧ x ≤ 0 := -2 - x
| x if 0 ≤ x ∧ x ≤ 2 := sqrt (4 - (x - 2)^2) - 2
| x if 2 ≤ x ∧ x ≤ 3 := 2 * (x - 2)
| _ := 0  -- default case to ensure total function

def g (x : ℝ) : ℝ := (1/3) * f x + 2

theorem transformed_graph_vertically_compressed_and_shifted_up :
  ∀ x, -3 ≤ x ∧ x ≤ 3 → (g x = (1/3) * f x + 2) := 
by 
  sorry

end transformed_graph_vertically_compressed_and_shifted_up_l194_194876


namespace gun_can_hit_l194_194316

-- Define the constants
variables (v g : ℝ)

-- Define the coordinates in the first quadrant
variables (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0)

-- Prove the condition for a point (x, y) to be in the region that can be hit by the gun
theorem gun_can_hit (hv : v > 0) (hg : g > 0) :
  y ≤ (v^2 / (2 * g)) - (g * x^2 / (2 * v^2)) :=
sorry

end gun_can_hit_l194_194316


namespace brother_total_spent_l194_194638

variables (x y : ℝ)

def total_cost_brother (x y : ℝ) : ℝ :=
  0.9 * ((8 / 3) * x + 30 * y)

theorem brother_total_spent (x y : ℝ) :
  total_cost_brother x y = 2.4 * x + 27 * y := 
by
  sorry

end brother_total_spent_l194_194638


namespace problem_statement_l194_194524

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for f

-- Theorem stating the axis of symmetry and increasing interval for the transformed function
theorem problem_statement (hf_even : ∀ x, f x = f (-x))
  (hf_increasing : ∀ x₁ x₂, 3 < x₁ → x₁ < x₂ → x₂ < 5 → f x₁ < f x₂) :
  -- For y = f(x - 1), the following holds:
  (∀ x, (f (x - 1)) = f (-(x - 1))) ∧
  (∀ x₁ x₂, 4 < x₁ → x₁ < x₂ → x₂ < 6 → f (x₁ - 1) < f (x₂ - 1)) :=
sorry

end problem_statement_l194_194524


namespace petya_always_wins_l194_194302

noncomputable def petya_wins (m n : ℕ) (hm : m = 1000) (hn : n = 2020) : Prop :=
  ∃ (a b m1 n1 : ℕ), 
    m = (2 ^ a) * m1 ∧ n = (2 ^ b) * n1 ∧
    odd m1 ∧ odd n1 ∧ a ≠ b

theorem petya_always_wins : petya_wins 1000 2020 :=
by 
  have hm : 1000 = 2 ^ 3 * 125 := by sorry
  have hn : 2020 = 2 ^ 2 * 505 := by sorry
  have m1_odd : odd 125 := by sorry
  have n1_odd : odd 505 := by sorry
  exact ⟨3, 2, 125, 505, hm, hn, m1_odd, n1_odd, by linarith⟩

end petya_always_wins_l194_194302


namespace arcsin_neg_half_eq_neg_pi_six_l194_194057

theorem arcsin_neg_half_eq_neg_pi_six : 
  Real.arcsin (-1 / 2) = -Real.pi / 6 := 
sorry

end arcsin_neg_half_eq_neg_pi_six_l194_194057


namespace distance_between_harper_and_jack_is_848_l194_194625

variables (H J t : ℝ)

-- Defining the conditions
def harper_distance_eq : Prop := 1000 = H * t
def jack_distance_eq : Prop := 152 = J * t

-- The theorem we need to prove
theorem distance_between_harper_and_jack_is_848 
  (h : harper_distance_eq H t) 
  (j : jack_distance_eq J t) : 
  1000 - (J * t) = 848 :=
by 
  rw jack_distance_eq at j
  exact sorry

end distance_between_harper_and_jack_is_848_l194_194625


namespace num_four_digit_square_palindromes_l194_194191

open Nat

-- Define what it means to be a 4-digit number
def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- The main theorem stating that there are exactly 2 four-digit squares that are palindromes
theorem num_four_digit_square_palindromes : 
  { n : ℕ | is_four_digit n ∧ is_palindrome n ∧ ∃ k : ℕ, k^2 = n ∧ k >= 32 ∧ k <= 99 }.to_finset.card = 2 :=
sorry

end num_four_digit_square_palindromes_l194_194191


namespace quadratic_roots_always_implies_l194_194879

variable {k x1 x2 : ℝ}

theorem quadratic_roots_always_implies (h1 : k^2 > 16) 
  (h2 : x1 + x2 = -k)
  (h3 : x1 * x2 = 4) : x1^2 + x2^2 > 8 :=
by
  sorry

end quadratic_roots_always_implies_l194_194879


namespace sin_double_angle_l194_194852

variable (α : ℝ)
variable (cos_neg_half : cos α = -1/2)
variable (alpha_second_quadrant : π / 2 < α ∧ α < π)

theorem sin_double_angle : sin (2 * α) = -√3 / 2 :=
by
  sorry

end sin_double_angle_l194_194852


namespace digit_combinations_even_l194_194898

theorem digit_combinations_even (A B C : ℕ) (C_is_even : C % 2 = 0) :
  (1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) →
  (∃ N : ℕ, N = 450) :=
by {
  intro h,
  use 450,
  sorry
}

end digit_combinations_even_l194_194898


namespace shaded_region_area_l194_194242

-- Definitions and assumptions
variables (AD DC : ℤ) (square_side rect_length rect_width : ℤ)
variables (A C H G : point)
variables (DEFG : shape)
variables (ABCD : shape)
variables [Square DEFG] [Rectangle ABCD]

-- Given conditions
def conditions (AD DC : ℤ) (square_side rect_length rect_width : ℤ) : Prop :=
  AD = 6 ∧
  rect_length * rect_width = 2 ∧
  square_side * square_side = 16 ∧
  DC = AD / 3 ∧ 
  H ∈ FG ∧
  straight_line A C H

-- Main theorem to prove
theorem shaded_region_area (AD DC : ℤ) (square_side rect_length rect_width : ℤ) :
  conditions AD DC square_side rect_length rect_width → 
  area (ABCD) + area (DEFG) - area (triangle A H G) = 9 :=
by
  sorry

end shaded_region_area_l194_194242


namespace find_phi_l194_194267

theorem find_phi (Q s : ℂ) (φ : ℝ) (hQ : (Q = s * complex.exp (complex.I * real.pi * φ / 180))) 
  (hs : 0 < s) (h0φ : 0 ≤ φ ∧ φ < 360)
  (hp : polynomial.eval (Q : ℂ) (polynomial.X^8 + polynomial.X^6 + polynomial.X^4 + polynomial.X^3 + polynomial.X + 1) = 0) :
  φ = 180 :=
sorry

end find_phi_l194_194267


namespace average_speed_including_stoppages_l194_194080

/--
If the average speed of a bus excluding stoppages is 50 km/hr, and
the bus stops for 12 minutes per hour, then the average speed of the
bus including stoppages is 40 km/hr.
-/
theorem average_speed_including_stoppages
  (u : ℝ) (Δt : ℝ) (h₁ : u = 50) (h₂ : Δt = 12) : 
  (u * (60 - Δt) / 60) = 40 :=
by
  sorry

end average_speed_including_stoppages_l194_194080


namespace circle_square_relation_l194_194120

theorem circle_square_relation (r s : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = r^2) →
  (∃ v : ℝ → ℝ, -- there exists a function v which represents the vertices
    v 0 = (-s / 2, -s / 2) ∧ 
    v 1 = (s / 2, -s / 2) ∧ 
    v 2 = (s / 2, s / 2) ∧ 
    v 3 = (-s / 2, s / 2) ∧ 
    ∀ i, (v i).1^2 + (v i).2^2 = r^2) →
  r = s * real.sqrt 2 / 2 :=
begin
  sorry
end

end circle_square_relation_l194_194120


namespace find_xyz_sum_cube_l194_194582

variable (x y z c d : ℝ) 

theorem find_xyz_sum_cube (h1 : x * y * z = c) (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d := 
by
  sorry

end find_xyz_sum_cube_l194_194582


namespace second_pipe_filling_time_l194_194423

theorem second_pipe_filling_time (T : ℝ) (h1 : (1 / 10) + (1 / T) = 1 / 3.75) : T = 6 := by
  sorry

end second_pipe_filling_time_l194_194423


namespace exercise_l194_194269

def clubsuit (a b : ℝ) : ℝ := (3 * a / b) * (b / a)

theorem exercise : (clubsuit 7 (clubsuit 4 8)) clubsuit 2 = 3 :=
by
  have h₁ : clubsuit 4 8 = 3 := by
    unfold clubsuit
    field_simp
  have h₂ : clubsuit 7 3 = 3 := by
    unfold clubsuit
    field_simp
  have h₃ : clubsuit 3 2 = 3 := by
    unfold clubsuit
    field_simp
  rw [h₁]  
  rw [h₂]
  exact h₃

end exercise_l194_194269


namespace range_of_function_l194_194277

def f (a x : ℝ) : ℝ := a^x / (1 + a^x)

def g (a x : ℝ) := f a x - 1/2

theorem range_of_function (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∀ x : ℝ, (⌊g a x⌋ + ⌊g a (-x)⌋ = -1) ∨ (⌊g a x⌋ + ⌊g a (-x)⌋ = 0) :=
by
  sorry

end range_of_function_l194_194277


namespace minimum_basketballs_sold_l194_194432

theorem minimum_basketballs_sold :
  ∃ (F B K : ℕ), F + B + K = 180 ∧ 3 * F + 5 * B + 10 * K = 800 ∧ F > B ∧ B > K ∧ K = 2 :=
by
  sorry

end minimum_basketballs_sold_l194_194432


namespace sum_of_three_numbers_l194_194399

theorem sum_of_three_numbers
  (a b c : ℕ) (h_prime : Prime c)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + a * c = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_l194_194399


namespace bricks_in_wall_l194_194375

theorem bricks_in_wall (x : ℕ) (r₁ r₂ combined_rate : ℕ) :
  (r₁ = x / 8) →
  (r₂ = x / 12) →
  (combined_rate = r₁ + r₂ - 15) →
  (6 * combined_rate = x) →
  x = 360 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end bricks_in_wall_l194_194375


namespace tables_count_l194_194014

def total_tables (four_legged_tables three_legged_tables : Nat) : Nat :=
  four_legged_tables + three_legged_tables

theorem tables_count
  (four_legged_tables three_legged_tables : Nat)
  (total_legs : Nat)
  (h1 : four_legged_tables = 16)
  (h2 : total_legs = 124)
  (h3 : 4 * four_legged_tables + 3 * three_legged_tables = total_legs) :
  total_tables four_legged_tables three_legged_tables = 36 :=
by
  sorry

end tables_count_l194_194014


namespace johns_daily_earnings_l194_194969

def daily_earnings (monthly_visits : Nat) (earnings_per_visit : Nat → Rat) (days_in_month : Nat) : Rat :=
  (monthly_visits * earnings_per_visit monthly_visits) / days_in_month

theorem johns_daily_earnings :
  (daily_earnings 30000 (λ _, 0.01) 30) = 10 := by
sorry

end johns_daily_earnings_l194_194969


namespace polynomial_solution_l194_194999

def polynomial (n : ℕ) (an anm1 a0 a1 : ℝ) :=
  a_n * x ^ n + a_{n-1} * x ^ (n-1) + ... + a_0

theorem polynomial_solution (n : ℕ) (P : polynomial) (an a0 a1 anm1 : ℝ) : 
  n ≥ 2 → 
  ∀ x, P(x) = an * (x+1) ^ (n-1) * (x + β) → 
  ∃ β, β ≥ 1 ∧ 
  (an ≠ 0) ∧ 
  (a0^2 + a1 * an = an^2 + a0 * anm1) :=
begin
  sorry
end

end polynomial_solution_l194_194999


namespace find_a3_l194_194139

-- Define the polynomial equality
def polynomial_equality (x : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) :=
  (1 + x) * (2 - x)^6 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7

-- State the main theorem
theorem find_a3 (a0 a1 a2 a4 a5 a6 a7 : ℝ) :
  (∃ (x : ℝ), polynomial_equality x a0 a1 a2 (-25) a4 a5 a6 a7) :=
sorry

end find_a3_l194_194139


namespace value_of_m_l194_194000

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2 * x + 1 else 2 * (-x) + 1

theorem value_of_m (m : ℝ) (heven : ∀ x : ℝ, f (-x) = f x)
  (hpos : ∀ x : ℝ, x ≥ 0 → f x = 2 * x + 1)
  (hfm : f m = 5) : m = 2 ∨ m = -2 :=
sorry

end value_of_m_l194_194000


namespace f_0_eq_0_l194_194581

-- Define a function f with the given condition
def f (x : ℤ) : ℤ := if x = 0 then 0
                     else (x-1)^2 + 2*(x-1) + 1

-- State the theorem
theorem f_0_eq_0 : f 0 = 0 :=
by sorry

end f_0_eq_0_l194_194581


namespace codes_cause_ambiguity_l194_194946

variable (k : char → ℕ)
variable (alphabet : List char)
variable (word_STO word_PYTСOT : List char)

axiom codes_in_range : ∀ x, k x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
axiom word_STO_letters : word_STO = ['С', 'Т', 'О']
axiom word_PYTСOT_letters : word_PYTСOT = ['П', 'Я', 'Т', 'Ь', 'С', 'О', 'Т']
axiom weight_condition : k 'С' + k 'Т' + k 'О' ≥ k 'П' + k 'Я' + k 'Т' + k 'Ь' + k 'С' + k 'О' + k 'Т'

theorem codes_cause_ambiguity : (k 'П' = 0 ∧ k 'Я' = 0 ∧ k 'Т' = 0 ∧ k 'Ь' = 0) → 
  ∃ k_S k_O : ℕ, k_S ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ k_O ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  ∃ word1 word2 : List char, word1 ≠ word2 ∧
  (word1.sum (λ x, k x) = word2.sum (λ x, k x)) := 
sorry

end codes_cause_ambiguity_l194_194946


namespace cylinder_volume_and_surface_area_l194_194492

noncomputable def cylinder_volume (side_length : ℝ) : ℝ :=
  let radius := side_length / 2
  let height := side_length
  π * radius^2 * height

noncomputable def cylinder_surface_area (side_length : ℝ) : ℝ :=
  let radius := side_length / 2
  let height := side_length
  let base_area := π * radius^2
  let lateral_surface_area := 2 * π * radius * height
  2 * base_area + lateral_surface_area

theorem cylinder_volume_and_surface_area (side_length : ℝ) (h : side_length = 20) :
  cylinder_volume side_length = 2000 * π ∧
  cylinder_surface_area side_length = 600 * π :=
by
  sorry

end cylinder_volume_and_surface_area_l194_194492


namespace b3_b8_product_l194_194281

-- Definitions based on conditions
def is_arithmetic_seq (b : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

-- The problem statement
theorem b3_b8_product (b : ℕ → ℤ) (h_seq : is_arithmetic_seq b) (h4_7 : b 4 * b 7 = 24) : 
  b 3 * b 8 = 200 / 9 :=
sorry

end b3_b8_product_l194_194281


namespace purely_imaginary_has_specific_a_l194_194229

theorem purely_imaginary_has_specific_a (a : ℝ) :
  (a^2 - 1 + (a - 1 : ℂ) * Complex.I) = (a - 1 : ℂ) * Complex.I → a = -1 := 
by
  sorry

end purely_imaginary_has_specific_a_l194_194229


namespace find_value_of_a_l194_194599

noncomputable def parametric_equation_curve1 : ℝ → ℝ → Prop :=
λ a t, a > 0 ∧ ∃ x y, x = a * Real.cos t ∧ y = 1 + a * Real.sin t

def polar_equation_curve2 (θ : ℝ) : ℝ := 4 * Real.cos θ

def polar_equation_curve3 (α₀ : ℝ) : Prop := Real.tan α₀ = 2

def all_common_points_on_line 
  (a : ℝ) (t θ α₀ : ℝ) 
  (C1 C2 C3 : ℝ → ℝ → Prop) 
  : Prop := 
  ∀ (x y : ℝ), (C1 a t ∧ y = 1 + a * Real.sin t) 
               ∧ (y = 2 * x)
               → x = 4 * Real.cos θ
               → α₀ = α₀

theorem find_value_of_a 
  (a : ℝ) 
  (t θ α₀ : ℝ)
  (C1 : ℝ → ℝ → Prop := parametric_equation_curve1)
  (C2 : ℝ → ℝ := polar_equation_curve2)
  (C3 : ℝ → Prop := polar_equation_curve3) 
  (common_points : Prop := all_common_points_on_line a t θ α₀ C1 C2 (λ x y, y = 2 * x))
  : a = 1 :=
sorry

end find_value_of_a_l194_194599


namespace product_of_terms_l194_194603

variable (a : ℕ → ℝ)

-- Conditions: the sequence is geometric, a_1 = 1, a_10 = 3.
axiom geometric_sequence : ∀ n m : ℕ, a n * a m = a 1 * a (n + m - 1)

axiom a_1_eq_one : a 1 = 1
axiom a_10_eq_three : a 10 = 3

-- We need to prove that the product a_2a_3a_4a_5a_6a_7a_8a_9 = 81.
theorem product_of_terms : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end product_of_terms_l194_194603


namespace fraction_50th_decimal_l194_194384

theorem fraction_50th_decimal (n : ℕ) :
  n = 50 → (decimal_expansion (367 / 369)).digit n = d :=
sorry

end fraction_50th_decimal_l194_194384


namespace flight_landing_time_in_gander_l194_194392

-- Definitions based on the conditions
def toronto_to_gander_time_difference : Time := Time.mk 1 30 0 -- 1 hour 30 minutes
def flight_departure_time : Time := Time.mk 15 0 0 -- 3:00 p.m. in 24-hour format
def flight_duration : Time := Time.mk 2 50 0 -- 2 hours 50 minutes

-- The main statement to be proved
theorem flight_landing_time_in_gander :
  let landing_time_in_toronto := flight_departure_time + flight_duration in
  let landing_time_in_gander := landing_time_in_toronto + toronto_to_gander_time_difference in
  landing_time_in_gander = Time.mk 19 20 0 -- 7:20 p.m. in 24-hour format
  :=
by
  sorry

end flight_landing_time_in_gander_l194_194392


namespace number_of_valid_functions_l194_194416

def is_geom_seq (a b c : ℤ) : Prop := b^2 = a * c

def valid_f (f : ℕ → ℤ) : Prop :=
  f 1 = 1 ∧ 
  (∀ x : ℕ, x ∈ {1, 2, 3, ..., 11} → |f (x + 1) - f x| = 1) ∧
  is_geom_seq (f 1) (f 6) (f 12)

theorem number_of_valid_functions : ∃ n : ℕ, n = 155 ∧
  ∃ f : (ℕ → ℤ), valid_f f :=
sorry

end number_of_valid_functions_l194_194416


namespace unique_function_l194_194092

theorem unique_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f (x * y) + f (x * z) - f x * f (y * z) ≥ 1) → (∃! g : ℝ → ℝ, ∀ x, g x = 1) :=
by {
  intros h,
  let g := λ x, 1,
  use g,
  split,
  { intro x,
    refl },
  { intros h' h'_def,
    funext x,
    have h_fx := h x x x,
    rw [h'_def] at h_fx,
    sorry }
}

end unique_function_l194_194092


namespace train_passes_jogger_in_40_76_seconds_l194_194420

-- Define the constants and conditions
def jogger_speed_kmh : ℝ := 7
def distance_ahead_m : ℝ := 350
def train_length_m : ℝ := 250
def train_speed_kmh : ℝ := 60

-- Define the unit conversion from km/hr to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

-- Calculate the relative speed (train speed - jogger speed)
def relative_speed_ms := kmh_to_ms train_speed_kmh - kmh_to_ms jogger_speed_kmh

-- Calculate the total distance to be covered
def total_distance_m := train_length_m + distance_ahead_m

-- Calculate the time it will take for the train to pass the jogger
noncomputable def time_to_pass : ℝ := total_distance_m / relative_speed_ms

-- State the theorem to be proved
theorem train_passes_jogger_in_40_76_seconds :
  abs (time_to_pass - 40.76) < 0.01 :=
by
  sorry

end train_passes_jogger_in_40_76_seconds_l194_194420


namespace prime_root_range_l194_194905

-- Let's define our conditions first
def is_prime (p : ℕ) : Prop := Nat.Prime p

def has_integer_roots (p : ℕ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧ x + y = p ∧ x * y = -156 * p

-- Now state the theorem
theorem prime_root_range (p : ℕ) (hp : is_prime p) (hr : has_integer_roots p) : 11 < p ∧ p ≤ 21 :=
by
  sorry

end prime_root_range_l194_194905


namespace geometric_progression_common_ratio_l194_194273

theorem geometric_progression_common_ratio (x y z r: ℝ) 
  (h_distinct: x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h_geometric: x^2 * (y - z), y^2 * (z - x), z^2 * (x - y) form_geo_prog_with_common_ratio r):
  r^2 + r + 1 = 0 := sorry

end geometric_progression_common_ratio_l194_194273


namespace only_n_is_4_l194_194084

noncomputable def is_correct_n (n : ℕ) : Prop :=
  n > 3 ∧ 
  ∃ (polygon : fin n → ℝ × ℝ), 
    (convex_hull ℝ (set.range polygon)).convex 
    ∧ (∀ (i j k l : fin n), i ≠ j ∧ k ≠ l 
      → (∃ D1 D2 : (ℝ × ℝ) × (ℝ × ℝ), 
          (D1 = (polygon i, polygon j) ∧ ((polygon k, polygon l) = reflection D1) ∧ 
          D1.1 ≠ D1.2 ∧ (D1.1 + D1.2) / 2 = (D2.1 + D2.2) / 2 ∧ 
          dist D1.1 D1.2 = dist D2.1 D2.2)))

theorem only_n_is_4 : ∀ n : ℕ, is_correct_n n → n = 4 :=
by
  sorry

end only_n_is_4_l194_194084


namespace smallest_N_diagonals_l194_194939

theorem smallest_N_diagonals (n : ℕ) (h_n : n = 2017) : 
   ∃ (N : ℕ), N = 1008 ∧
    ∀ selected_diagonals : (Fin 2017 → Fin 2017) → list (Fin 2017 × Fin 2017), 
    (∀ i j, selected_diagonals i ≠ selected_diagonals j → (i ≠ j)) → 
    N > 1007 →
    ∃ i j, i ≠ j ∧ (∃ k l, selected_diagonals i = (⟨k, h_n⟩, ⟨l, h_n⟩) ∧ selected_diagonals j = (⟨k, h_n⟩, ⟨l, h_n⟩)) := by
  -- Proof skipped
  sorry

end smallest_N_diagonals_l194_194939


namespace min_value_ratio_l194_194512

theorem min_value_ratio 
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h : x - 2 * y + 3 * z = 0) : 
  ∃ m : ℝ, m = 3 ∧ ∀ (y : ℝ), y = (x + 3 * z) / 2 → (y^2 / (x * z)) ≥ m :=
begin
  use 3,
  split,
  { refl, },
  { sorry },
end

end min_value_ratio_l194_194512


namespace compare_neg_fractions_l194_194797

theorem compare_neg_fractions : (-2 / 3 : ℚ) < -3 / 5 :=
by
  sorry

end compare_neg_fractions_l194_194797


namespace container_capacity_l194_194410

variable (C : ℝ)
variable (h1 : 0.30 * C + 27 = (3/4) * C)

theorem container_capacity : C = 60 := by
  sorry

end container_capacity_l194_194410


namespace combined_area_of_four_removed_triangles_l194_194771

noncomputable def combined_area_of_removed_triangles (s x y: ℝ) : Prop :=
  x + y = s ∧ s - 2 * x = 15 ∧ s - 2 * y = 9 ∧
  4 * (1 / 2 * x * y) = 67.5

-- Statement of the problem
theorem combined_area_of_four_removed_triangles (s x y: ℝ) :
  combined_area_of_removed_triangles s x y :=
  by
    sorry

end combined_area_of_four_removed_triangles_l194_194771


namespace sum_of_first_seven_terms_l194_194151

noncomputable def arithmetic_sequence_sum (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n = a 1 - a 0) : Prop :=
∃ d : ℝ, ∀ n, a (n + 1) = a n + d

theorem sum_of_first_seven_terms (a : ℕ → ℝ) (h_arith : arithmetic_sequence_sum a) (h_cond : a 2 + a 3 + a 4 = 12) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 28 := 
sorry

end sum_of_first_seven_terms_l194_194151


namespace y_six_power_eq_44_over_27_l194_194427

theorem y_six_power_eq_44_over_27
  (y : ℝ)
  (h_pos : 0 < y)
  (h_equation : ∛(2 - y^3) + ∛(2 + y^3) = 2)
  : y^6 = 44 / 27 :=
sorry

end y_six_power_eq_44_over_27_l194_194427


namespace mark_and_james_need_to_buy_2_dice_to_play_their_game_l194_194642

variable (total_dice_needed : ℕ)
variable (mark_total_dice : ℕ)
variable (mark_percent_12_sided : ℝ)
variable (james_total_dice : ℕ)
variable (james_percent_12_sided : ℝ)

def number_of_dice_to_buy : ℕ :=
  let mark_12_sided := mark_percent_12_sided * mark_total_dice
  let james_12_sided := james_percent_12_sided * james_total_dice
  total_dice_needed - mark_12_sided.toNat - james_12_sided.toNat

theorem mark_and_james_need_to_buy_2_dice_to_play_their_game :
  total_dice_needed = 14 →
  mark_total_dice = 10 →
  mark_percent_12_sided = 0.6 →
  james_total_dice = 8 →
  james_percent_12_sided = 0.75 →
  number_of_dice_to_buy total_dice_needed mark_total_dice mark_percent_12_sided james_total_dice james_percent_12_sided = 2 :=
by
  sorry

end mark_and_james_need_to_buy_2_dice_to_play_their_game_l194_194642


namespace length_of_chord_l194_194132

def ellipse_c (x y : ℝ) : Prop := (x^2 / 5) + (y^2 / 4) = 1

def line_l (x y : ℝ) : Prop := y = x + 1

def focus (F : ℝ × ℝ) : Prop := F = (-1, 0)

theorem length_of_chord :
  ∀ (x1 y1 x2 y2 : ℝ),
    focus (-1, 0) →
    line_l x1 y1 →
    line_l x2 y2 →
    ellipse_c x1 y1 →
    ellipse_c x2 y2 →
    (x1 + x2) = -10 / 9 →
    (x1 * x2) = -5 / 3 →
    ((√2) * (√((x1 + x2)^2 - 4 * x1 * x2))) = 16 * (√5) / 9 := by
  sorry

end length_of_chord_l194_194132


namespace cost_per_board_game_is_15_l194_194965

-- Definitions of the conditions
def number_of_board_games : ℕ := 6
def bill_paid : ℕ := 100
def bill_value : ℕ := 5
def bills_received : ℕ := 2

def total_change := bills_received * bill_value
def total_cost := bill_paid - total_change
def cost_per_board_game := total_cost / number_of_board_games

-- The theorem stating that the cost of each board game is $15
theorem cost_per_board_game_is_15 : cost_per_board_game = 15 := 
by
  -- Omitted proof steps
  sorry

end cost_per_board_game_is_15_l194_194965


namespace part1_part2_l194_194924

-- Definitions and conditions
variables (A B C a b c : ℝ)
variables (h1 : a - b = 2) (h2 : c = 4) (h3 : sin A = 2 * sin B)

-- First part: Prove a = 4, b = 2, and cos B = 7/8
theorem part1 (A B C a b c : ℝ) (h1 : a - b = 2) (h2 : c = 4) (h3 : sin A = 2 * sin B) : 
    a = 4 ∧ b = 2 ∧ cos B = 7 / 8 := 
    sorry

-- Second part: Prove sin(2B - π/6) = (21√5 - 17) / 64
theorem part2 (A B C a b c : ℝ) (h1 : a - b = 2) (h2 : c = 4) (h3 : sin A = 2 * sin B)
  (h4 : a = 4) (h5 : b = 2) (h6 : cos B = 7 / 8) : 
    sin (2 * B - π / 6) = (21 * √5 - 17) / 64 := 
    sorry

end part1_part2_l194_194924


namespace num_four_digit_square_palindromes_l194_194188

open Nat

-- Define what it means to be a 4-digit number
def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- The main theorem stating that there are exactly 2 four-digit squares that are palindromes
theorem num_four_digit_square_palindromes : 
  { n : ℕ | is_four_digit n ∧ is_palindrome n ∧ ∃ k : ℕ, k^2 = n ∧ k >= 32 ∧ k <= 99 }.to_finset.card = 2 :=
sorry

end num_four_digit_square_palindromes_l194_194188


namespace triangle_perimeter_l194_194494

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

def P : ℝ × ℝ := (-3, 4)
def Q : ℝ × ℝ := (4, 3)
def R : ℝ × ℝ := (1, -6)

theorem triangle_perimeter :
  distance P Q + distance Q R + distance R P = real.sqrt 50 + real.sqrt 90 + real.sqrt 116 :=
by
  sorry

end triangle_perimeter_l194_194494


namespace blue_balls_count_l194_194409

noncomputable def num_blue_balls (total_balls blue_draw_probability both_blue_probability : ℕ) : ℕ :=
  let b := (6 : ℕ) in -- This is derived from b(b-1) = 6, solution of b^2 - b - 6 = 0
  if (total_balls = 12) ∧ (blue_draw_probability = b * (b - 1)) ∧ (both_blue_probability = 6) then
    b
  else
    0  -- This indicates an invalid scenario based on the given condition.

theorem blue_balls_count :
  num_blue_balls 12 22 6 = 3 :=
by
  sorry

end blue_balls_count_l194_194409


namespace at_least_one_zero_l194_194531

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) : 
  a = 0 ∨ b = 0 ∨ c = 0 := 
sorry

end at_least_one_zero_l194_194531


namespace solve_system_l194_194315

theorem solve_system :
  (∃ k n : ℤ, (x y : ℝ),
    (2 * cos x ^ 2 + 2 * sqrt 2 * cos x * cos (4 * x) ^ 2 + cos (4 * x) ^ 2 = 0) ∧
    (sin x = cos y) ∧
    ((x = 3 * π / 4 + 2 * π * ↑k ∧ y = π / 4 + 2 * π * ↑n) ∨ 
     (x = 3 * π / 4 + 2 * π * ↑k ∧ y = -π / 4 + 2 * π * ↑n) ∨
     (x = -3 * π / 4 + 2 * π * ↑k ∧ y = 3 * π / 4 + 2 * π * ↑n) ∨
     (x = -3 * π / 4 + 2 * π * ↑k ∧ y = -3 * π / 4 + 2 * π * ↑n))) :=
begin
  sorry
end

end solve_system_l194_194315


namespace find_x_l194_194408

theorem find_x (x : ℚ) (h : 2 / 5 = (4 / 3) / x) : x = 10 / 3 :=
by
sorry

end find_x_l194_194408


namespace exist_sequences_l194_194477

theorem exist_sequences (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ i, (3 * Real.pi / 2) ≤ a i ∧ a i ≤ b i)
  (h2 : ∀ i x, 0 < x → x < 1 → cos (a i * x) + cos (b i * x) ≥ - (1 / i)) :
  ∃ (a b : ℕ → ℝ), 
    (∀ i, (3 * Real.pi / 2) ≤ a i ∧ a i ≤ b i) ∧ 
    (∀ i x, 0 < x → x < 1 → cos (a i * x) + cos (b i * x) ≥ - (1 / i)) :=
  sorry

end exist_sequences_l194_194477


namespace collinear_vectors_l194_194987

variables {V : Type*} [AddCommGroup V] [Module ℝ V ] {e1 e2 : V}

def non_collinear (v1 v2 : V) : Prop :=
¬ ∃ k : ℝ, k • v1 = v2

theorem collinear_vectors {k : ℝ} (h₀ : e1 ≠ 0) (h₁ : e2 ≠ 0) (h₂ : non_collinear e1 e2) :
  (∃ λ : ℝ, k • e1 + e2 = λ • (e1 + k • e2)) ↔ k = 1 ∨ k = -1 :=
sorry

end collinear_vectors_l194_194987


namespace evaluate_expression_l194_194078

theorem evaluate_expression :
  (3 / 2) * ((8 / 3) * ((15 / 8) - (5 / 6))) / (((7 / 8) + (11 / 6)) / (13 / 4)) = 5 :=
by
  sorry

end evaluate_expression_l194_194078


namespace band_first_set_songs_count_l194_194693

theorem band_first_set_songs_count 
  (total_repertoire : ℕ) (second_set : ℕ) (encore : ℕ) (avg_third_fourth : ℕ)
  (h_total_repertoire : total_repertoire = 30)
  (h_second_set : second_set = 7)
  (h_encore : encore = 2)
  (h_avg_third_fourth : avg_third_fourth = 8)
  : ∃ (x : ℕ), x + second_set + encore + avg_third_fourth * 2 = total_repertoire := 
  sorry

end band_first_set_songs_count_l194_194693


namespace walking_running_distance_ratio_l194_194422

variables (d_w d_r : ℝ)

theorem walking_running_distance_ratio :
  d_w + d_r = 12 ∧ (d_w / 4 + d_r / 8 = 2.25) → d_w / d_r = 1 :=
by
  assume h : d_w + d_r = 12 ∧ (d_w / 4 + d_r / 8 = 2.25)
  sorry

end walking_running_distance_ratio_l194_194422


namespace person_in_10th_place_not_given_l194_194230

variable (person : Type)
variable [decidable_eq person]

-- Define the positions of Simon, David, Hikmet, Jack, Marta, Rand, Todd, and unnamed racers
axiom Simon David Hikmet Jack Marta Rand Todd : person
variable (position : person → ℕ)

-- Given conditions
axiom cond1 : position Simon = position Jack + 3
axiom cond2 : position Rand = position Hikmet - 7
axiom cond3 : position Marta = position Jack + 3
axiom cond4 : position David = position Hikmet + 3
axiom cond5 : position Jack = position Todd - 5
axiom cond6 : position Todd = position Rand + 2
axiom cond7 : position Marta = 7

-- The statement to be proved
theorem person_in_10th_place_not_given : ∃ (x : person), position x = 10 ∧ x ≠ Simon ∧ x ≠ David ∧ x ≠ Hikmet ∧ x ≠ Jack ∧ x ≠ Marta ∧ x ≠ Rand ∧ x ≠ Todd :=
by
  sorry

end person_in_10th_place_not_given_l194_194230


namespace farmer_steven_total_days_l194_194584

theorem farmer_steven_total_days 
(plow_acres_per_day : ℕ)
(mow_acres_per_day : ℕ)
(farmland_acres : ℕ)
(grassland_acres : ℕ)
(h_plow : plow_acres_per_day = 10)
(h_mow : mow_acres_per_day = 12)
(h_farmland : farmland_acres = 55)
(h_grassland : grassland_acres = 30) :
((farmland_acres / plow_acres_per_day) + (grassland_acres / mow_acres_per_day) = 8) := by
  sorry

end farmer_steven_total_days_l194_194584


namespace inequality_solution_l194_194403

theorem inequality_solution :
  let f (x : ℝ) := sqrt (5 * x - 11) - sqrt (5 * x^2 - 21 * x + 21)
  let g (x : ℝ) := 5 * x^2 - 26 * x + 32
  let valid x := 5 * x - 11 ≥ 0 ∧ 5 * x^2 - 21 * x + 21 ≥ 0 ∧ f x ≥ g x
  (∑ k in (Finset.filter valid (Finset.Icc ⌈2.939⌉ ⌊2.939⌋)), k) = 3 :=
by
  sorry

end inequality_solution_l194_194403


namespace find_speed_of_second_train_l194_194438

noncomputable def speed_of_second_train : ℕ := 36

theorem find_speed_of_second_train
    (departure_time_first : ℕ) (departure_time_second : ℕ)
    (speed_first : ℕ) (meeting_distance : ℕ) :
  departure_time_first = 9 →
  departure_time_second = 14 →
  speed_first = 30 →
  meeting_distance = 1050 →
  let t := 25 in
  let speed_second := 36 in
  let distance_second := speed_second * t in
  distance_second = 900 →
  speed_second = 36 :=
by
  intros h1 h2 h3 h4 t speed_second distance_second heq
  sorry

end find_speed_of_second_train_l194_194438


namespace circle_equation_l194_194328

theorem circle_equation (M : Point) (h k r : ℝ) (M_center : M = (0, 2)) (tan_x_axis : r = 2) : 
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 → x^2 + (y - 2)^2 = 4 :=
by
  sorry

end circle_equation_l194_194328


namespace equal_distances_in_acute_angle_triangle_l194_194595

theorem equal_distances_in_acute_angle_triangle
  {A B C H P : Type*} 
  (triangle_ABC_acute : acute_triangle A B C)
  (H_on_altitude_AH : altitude A H)
  (P_on_AH : P ∈ H_on_altitude_AH)
  (midpoint_E : midpoint E A C)
  (midpoint_F : midpoint F A B)
  (perpendicular_E_CP : ∃ Q: Type*, orthogonal_projection E P Q ∧ Q = line(C, P))
  (perpendicular_F_BP : ∃ R: Type*, orthogonal_projection F P R ∧ R = line(B, P))
  (K_intersection : ∃ K : Type*, intersection_point K Q R)
  : distance K B = distance K C := 
sorry

end equal_distances_in_acute_angle_triangle_l194_194595


namespace shaded_region_area_l194_194460

-- Given definitions for the conditions
noncomputable def circle_radius : ℝ := 3
noncomputable def midpoint_distance : ℝ := 3 * real.sqrt 3
noncomputable def midpoint_O (A B : point) : point := midpoint A B
noncomputable def tangent_OC (A : point) (radius: ℝ) : line := tangent A radius
noncomputable def tangent_OD (B : point) (radius: ℝ) : line := tangent B radius
noncomputable def common_tangent (E F : point) (A B : point) (radius: ℝ) : line := common_tangent E F A B radius

variable (A B O C D E F : point)

-- The theorem statement
theorem shaded_region_area
  (midpoint : O = midpoint_O A B) 
  (radius : circle_radius = 3) 
  (distance_OA : dist O A = midpoint_distance)
  (tangent_OC : tangent_OC A circle_radius)
  (tangent_OD : tangent_OD B circle_radius)
  (common_tangent_EF : common_tangent E F A B circle_radius):

  area ECODF = 18 * real.sqrt 3 - 9 * real.sqrt 2 - 9 * π / 4 :=
sorry -- Proof omitted

end shaded_region_area_l194_194460


namespace find_r_l194_194102

open Polynomial

noncomputable def poly : Polynomial ℝ := 9 * X^3 - 6 * X^2 - 48 * X + 54

theorem find_r (r s : ℝ) (h1 : poly = 9 * (X - C r)^2 * (X - C s)) :
  r = 4 / 3 :=
by
  -- Using the factor form to find matching coefficients
  have hfactor : 9 * (X - C r)^2 * (X - C s) = 
    9 * (X^3 - (2 * r + s) * X^2 + (r^2 + 2 * r * s) * X - r^2 * s),
  { ring },
  rw h1 at hfactor,
  -- Polynomial equality implies coefficient equality
  have hcoeff := Polynomial.ext_iff.1 hfactor,
  -- Matching coefficients
  have eq1 : 2 * r + s = 2 / 3, from (congr_arg coeff (monomial 2 1) hcoeff).symm,
  have eq2 : r^2 + 2 * r * s = -16 / 3, from (congr_arg coeff (monomial 1 1) hcoeff).symm,
  have eq3 : r^2 * s = -6, from (congr_arg coeff (monomial 0 1) hcoeff).symm,
  sorry

end find_r_l194_194102


namespace num_four_digit_palindromic_squares_is_two_l194_194198

open Nat

-- Define the condition for a palindrome
def is_palindrome (n : ℕ) : Prop :=
  to_digits 10 n = (to_digits 10 n).reverse

-- Define the range of numbers to check
def range_32_to_99 := {x : ℕ | 32 ≤ x ∧ x ≤ 99}

-- Define the function to compute the square of a number
def square (n : ℕ) : ℕ := n * n

-- Define the set of 4-digit squares that are palindromes
def four_digit_palindromic_squares : Finset ℕ :=
  (Finset.filter (λ n => is_palindrome n) (Finset.image square (Finset.filter (λ n => 1000 ≤ square n ∧ square n < 10000) 
  (Finset.filter (λ n => n ∈ range_32_to_99) (Finset.range 100)))))

-- The main theorem stating the number of 4-digit palindromic squares
theorem num_four_digit_palindromic_squares_is_two :
  four_digit_palindromic_squares.card = 2 := sorry

end num_four_digit_palindromic_squares_is_two_l194_194198


namespace domino_arrangement_count_l194_194379

def valid_domino_arrangements : ℕ :=
  63

theorem domino_arrangement_count :
  ∀ (grid : list (list bool)),
    let valid_positions := [(2, 2)] in
    (∀ pos ∈ valid_positions, grid.nth pos.1 >>= (λ row, row.nth pos.2) = none) →
    list.length (filter (λ pos, grid.nth pos.1 >>= (λ row, row.nth pos.2) = some false) (list.product (list.range 7) (list.range 3))) = 63 :=
sorry

end domino_arrangement_count_l194_194379


namespace percentage_of_knives_after_trade_l194_194794
open scoped Rat

theorem percentage_of_knives_after_trade :
  ∀ (initial_knives : ℕ) (initial_forks : ℕ) (initial_spoons : ℕ) (trade_knives : ℕ) (trade_spoons : ℕ),
    initial_knives = 6 →
    initial_forks = 12 →
    initial_spoons = 3 * initial_knives →
    trade_knives = 10 →
    trade_spoons = 6 →
    (16 : ℚ) / (40 : ℚ) * (100 : ℚ) = 40 :=
begin
  sorry
end

end percentage_of_knives_after_trade_l194_194794


namespace convert_89_to_base4_l194_194802

theorem convert_89_to_base4 : base_repr 89 4 = "1121" :=
by sorry

end convert_89_to_base4_l194_194802


namespace sequence_general_formula_l194_194245

theorem sequence_general_formula (n : ℕ) (h : n > 0) :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1)) ∧ a n = 1 / (3 * n - 2) :=
by sorry

end sequence_general_formula_l194_194245


namespace area_of_circle_II_l194_194053

-- Defining the problem conditions
variable (r1 : ℝ) (r2 : ℝ) -- radii of circle I and II respectively
variable (A1 : ℝ) -- area of circle I

-- Conditions as defined
def condition1 : Prop := r1 * 2 = r2 -- Diameter of circle I equals radius of circle II
def condition2 : Prop := A1 = 4 -- Area of circle I is 4 square inches
def condition3 : Prop := A1 = π * r1^2 -- Standard area formula for circle

-- The statement to be proved
theorem area_of_circle_II (h1 : condition1 r1 r2) (h2 : condition2 A1) (h3 : condition3 r1 A1) : π * r2^2 = 16 :=
  sorry

end area_of_circle_II_l194_194053


namespace max_tunnel_construction_mileage_find_value_of_a_l194_194796

-- Define variables and conditions
variables (x a : ℝ)

def total_mileage := 56
def ordinary_road_mileage := 32
def elevated_road_mileage_in_q1 := total_mileage - ordinary_road_mileage - x
def minimum_elevated_road_mileage := 7 * x

-- Problem 1: Maximum tunnel construction mileage
theorem max_tunnel_construction_mileage :
  elevated_road_mileage_in_q1 >= minimum_elevated_road_mileage → x <= 3 :=
by intros h; sorry

-- Define cost variables and conditions for Q2
def ordinary_road_cost_per_km_q1 := 1
def elevated_road_cost_per_km_q1 := 2
def tunnel_road_cost_per_km_q1 := 4
def ordinary_road_mileage_q2 := ordinary_road_mileage - 9 * a
def elevated_road_mileage_q2 := elevated_road_mileage_in_q1 - 2 * a
def tunnel_road_mileage_q2 := x + a
def elevated_road_cost_per_km_q2 := elevated_road_cost_per_km_q1 + 0.5 * a

-- Problem 2: Value of a
theorem find_value_of_a (h1: elevated_road_mileage_in_q1 >= minimum_elevated_road_mileage) :
  (ordinary_road_cost_per_km_q1 * ordinary_road_mileage +
   elevated_road_cost_per_km_q1 * elevated_road_mileage_in_q1 +
   tunnel_road_cost_per_km_q1 * x = 
   (ordinary_road_cost_per_km_q1 * ordinary_road_mileage_q2 +
   elevated_road_cost_per_km_q2 * elevated_road_mileage_q2 +
   tunnel_road_cost_per_km_q1 * tunnel_road_mileage_q2)) → 
   a = 3 / 2 :=
by intros h2; sorry

end max_tunnel_construction_mileage_find_value_of_a_l194_194796


namespace function_sum_property_l194_194011

-- Defining the function f and the condition it satisfies
variable (f : ℝ × ℝ × ℝ → ℝ)
variable (cond : ∀ (a b c d e : ℝ), f(a, b, c) + f(b, c, d) + f(c, d, e) + f(d, e, a) + f(e, a, b) = a + b + c + d + e)

-- The main theorem to prove
theorem function_sum_property (n : ℕ) (h : 5 ≤ n) (x : ℕ → ℝ) :
  (∑ i in finset.range n, f (x i, x ((i + 1) % n), x ((i + 2) % n))) = (∑ i in finset.range n, x i) :=
sorry

end function_sum_property_l194_194011


namespace unique_zero_point_range_l194_194918

noncomputable def has_unique_zero_point (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, (ax₀^3 - 6*x₀^2 + 1 = 0) ∧ (x₀ > 0) ∧
    (∀ x ≠ x₀, ax^3 - 6x^2 + 1 ≠ 0)

theorem unique_zero_point_range (a : ℝ) :
  has_unique_zero_point a → a < -4 * Real.sqrt 2 := sorry

end unique_zero_point_range_l194_194918


namespace bug_starting_vertex_eighth_move_l194_194411

noncomputable def Q : ℕ → ℚ
| 0 := 1
| n+1 := 1/3 * (1 - Q n)

theorem bug_starting_vertex_eighth_move :
  Q 8 = 547 / 2187 :=
by
  sorry

end bug_starting_vertex_eighth_move_l194_194411


namespace find_m_l194_194576

-- Define the conditions
variables (m n : ℝ) (b : ℝ)
-- Assume the given equation
axiom h : log 10 (m^2) = b - log 10 (n^3)

-- Prove that m = sqrt(10^b / n^3)
theorem find_m (h : log 10 (m^2) = b - log 10 (n^3)) : 
  m = sqrt (10^b / n^3) :=
sorry

end find_m_l194_194576


namespace find_x2_times_x1_plus_x3_l194_194624

noncomputable def a := Real.sqrt 2023
noncomputable def x1 := -Real.sqrt 7
noncomputable def x2 := 1 / a
noncomputable def x3 := Real.sqrt 7

theorem find_x2_times_x1_plus_x3 :
  let x1 := -Real.sqrt 7
  let x2 := 1 / Real.sqrt 2023
  let x3 := Real.sqrt 7
  x2 * (x1 + x3) = 0 :=
by
  sorry

end find_x2_times_x1_plus_x3_l194_194624


namespace first_digit_base8_l194_194321

theorem first_digit_base8 (y : ℕ) (hy : y = 2 * 3^6 + 1 * 3^5 + 2 * 3^4 + 0 * 3^3 + 2 * 3^2 + 1 * 3^1 + 2 * 3^0) : 
  (Nat.digits 8 y).head = 3 :=
by
  sorry

end first_digit_base8_l194_194321


namespace max_norm_a_add_b_sub_c_l194_194853

variables {V : Type*} [inner_product_space ℝ V]

-- Let a, b, c be unit vectors
variables (a b c : V)
variables (h_a : ∥a∥ = 1) (h_b : ∥b∥ = 1) (h_c : ∥c∥ = 1)
variables (h_ab : ⟪a, b⟫ = 0)
variables (h_cond : ⟪a - c, b - c⟫ ≤ 0)

-- Prove that the maximum value of ∥a + b - c∥ is 1
theorem max_norm_a_add_b_sub_c : 
  ∥a + b - c∥ ≤ 1 :=
begin
  -- Proof goes here
  sorry
end

end max_norm_a_add_b_sub_c_l194_194853


namespace not_possible_to_color_l194_194612

-- Define the types and predicates
def color : ℕ → ℕ → Prop

-- Formalize the conditions
axiom color_exists : ∃ c : ℕ → ℕ, (∀ n > 1, c n < 3) ∧ ((c 2 > 1) ∨ (c 2 < 1))

axiom all_colors_used : ∀ c : ℕ → ℕ, (∃ a, c a = 0) ∧ (∃ b, c b = 1) ∧ (∃ d, c d = 2)

axiom coloring_condition : ∀ c : ℕ → ℕ, (∀ a b, a > 1 ∧ b > 1 ∧ c a ≠ c b → (c (a * b) ≠ c a ∧ c (a * b) ≠ c b))

-- The impossibility theorem
theorem not_possible_to_color : ¬ (∃ c : ℕ → ℕ, (∀ n > 1, c n < 3) ∧ (∃ a, c a = 0) ∧ (∃ b, c b = 1) ∧ (∃ d, c d = 2) ∧ 
 (∀ a b, a > 1 ∧ b > 1 ∧ c a ≠ c b → (c (a * b) ≠ c a ∧ c (a * b) ≠ c b))) := 
begin
  sorry
end

end not_possible_to_color_l194_194612


namespace find_parallel_lines_a_l194_194133

/--
Given two lines \(l_1\): \(x + 2y - 3 = 0\) and \(l_2\): \(2x - ay + 3 = 0\),
prove that if the lines are parallel, then \(a = -4\).
-/
theorem find_parallel_lines_a (a : ℝ) :
  (∀ (x y : ℝ), x + 2*y - 3 = 0) 
  → (∀ (x y : ℝ), 2*x - a*y + 3 = 0)
  → (-1 / 2 = 2 / -a) 
  → a = -4 :=
by
  intros
  sorry

end find_parallel_lines_a_l194_194133


namespace choose_3_computers_l194_194503

theorem choose_3_computers (A B : Type) (a : A) (b : B) :
  (∃ C : set (A ⊕ B), 
  (card C = 3) ∧ 
  (∃ a_count b_count : ℕ, 
  (a_count + b_count = 3) ∧ (0 < a_count) ∧ (0 < b_count) ∧
  (a_count = 2 ∨ a_count = 1) ∧ (b_count = 2 ∨ b_count = 1))) → 18 := sorry

end choose_3_computers_l194_194503


namespace real_part_z1_mul_z2_l194_194546

def z1 (α : ℝ) : ℂ := complex.ofReal (cos α) + complex.I * (complex.ofReal (sin α))
def z2 (β : ℝ) : ℂ := complex.ofReal (cos β) + complex.I * (complex.ofReal (sin β))

theorem real_part_z1_mul_z2 (α β : ℝ) : complex.re (z1 α * z2 β) = cos (α + β) := by
  sorry

end real_part_z1_mul_z2_l194_194546


namespace collinear_hxi_l194_194619

-- Definitions of points and circles as per the given problem
variable (A B C H M N I X : Type*)
variable [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty H] 
variable [Nonempty M] [Nonempty N] [Nonempty I] [Nonempty X]

-- Assumptions required to state the problem
variable (isTriangle : Triangle A B C)
variable (isFootAltitude : FootAltitude A B C H)
variable (isMidpointM : Midpoint M A B)
variable (isMidpointN : Midpoint N A C)
variable (isMidpointI : Midpoint I M N)
variable (circumcircle1 : Circumcircle B H M)
variable (circumcircle2 : Circumcircle C N H)
variable (isSecondIntersection : SecondIntersection circumcircle1 circumcircle2 X)

-- The property to prove: collinearity of H, X, and I
theorem collinear_hxi : Collinear H X I := sorry

end collinear_hxi_l194_194619


namespace variance_five_numbers_l194_194585

-- Define the conditions and problem statement
theorem variance_five_numbers (m : ℝ) (h_avg : (1 + 2 + 3 + 4 + m) / 5 = 3) : 
  let numbers := [1, 2, 3, 4, m] in
  let mean := 3 in
  (let variance := (1/5) * ((mean - 1)^2 + (mean - 2)^2 + (mean - 3)^2 + (mean - 4)^2 + (mean - m)^2) in variance = 2) :=
by
  sorry

end variance_five_numbers_l194_194585


namespace ratio_EG_GH_l194_194608

variables {A B C M E H G : Type} [AddGroup G] [AddCommGroup G]

-- Midpoint definition
def midpoint (x y : G) : G := (x + y) / 2

-- Given conditions
variables (AB AC : ℝ) (AE_AH_Ratio: ℝ := 3)
variables (H : G) (E : G) (M : G := midpoint B C)
variables (a b c : G) -- Vectors for points A, B, C

-- Further given conditions
hypothesis (h_AB : ∥a - b∥ = 15)
hypothesis (h_AC : ∥a - c∥ = 24)
hypothesis (h_AE_AH : ∥a - E∥ = AE_AH_Ratio * ∥a - H∥)
hypothesis (h_G_intersection : ∃t s, (G = t • (M - a) + a) ∧ (G = s • (H - E) + E))

-- Theorem statement
theorem ratio_EG_GH : (EG / GH) = 2 / 3 := by
  sorry

end ratio_EG_GH_l194_194608


namespace removal_impossible_l194_194878

def is_arithmetic (a : ℕ) (d : ℕ) (n : ℕ) (seq : List ℕ) : Prop :=
  seq = List.range (n + 1) |>.map (λ i => a + i * d)

def sum_divisible_by (seq : List ℕ) (k : ℕ) : Prop :=
  seq.sum % k = 0

theorem removal_impossible :
  let seq := List.range 11 |>.map (λ i => 4 + i * 10)
  ¬(∃ (s1 s2 s3 s4 : List ℕ) (r1 r2 r3 r4 : ℕ) (x : List ℕ),
    (seq = s1 ++ x)  ∧ (r1 ∈ s1) ∧ (s2 = s1.erase r1) ∧ sum_divisible_by s2 11 ∧
    (s3 ⊆ s2.erase r1) ∧ (List.length s3 = 2) ∧ sum_divisible_by (s2 \ s3) 11 ∧
    (s4 ⊆ (s2 \ s3).erase_all s3) ∧ (List.length s4 = 3) ∧
    sum_divisible_by ((s2 \ s3) \ s4) 11 ∧
    List.length ((s2 \ s3) \ s4) = 1 ∧
    sum_divisible_by (remove_all ((s2 \ s3) \ s4) s4) 11) :=
by
  sorry

end removal_impossible_l194_194878


namespace min_value_of_x_plus_y_l194_194119

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y = x * y) : x + y ≥ 9 :=
by
  sorry

end min_value_of_x_plus_y_l194_194119


namespace painting_combinations_l194_194419

-- Define the conditions and the problem statement
def top_row_paint_count := 2
def total_lockers_per_row := 4
def valid_paintings := Nat.choose total_lockers_per_row top_row_paint_count

theorem painting_combinations : valid_paintings = 6 := by
  -- Use the derived conditions to provide the proof
  sorry

end painting_combinations_l194_194419


namespace find_b12_l194_194268

noncomputable def seq (b : ℕ → ℤ) : Prop :=
  b 1 = 2 ∧ 
  ∀ m n : ℕ, m > 0 → n > 0 → b (m + n) = b m + b n + (m * n * n)

theorem find_b12 (b : ℕ → ℤ) (h : seq b) : b 12 = 98 := 
by
  sorry

end find_b12_l194_194268


namespace max_area_curved_figure_l194_194039

-- Define the problem with conditions and the maximum area to be proved.
theorem max_area_curved_figure (x y r : ℝ) (π : ℝ) (h1 : x + y = 1) (h2 : r = x / π) : 
  (∃x, ∃y, ∃r, x + y = 1 ∧ r = x / π ∧ 
    (let S := (x^2 / (2 * π)) + (x / π * (1 - x - (2 * x / π))) in 
      S = 1 / (2 * (π + 4)))) :=
sorry

end max_area_curved_figure_l194_194039


namespace min_value_l194_194129

variable (d : ℕ) (a_n S_n : ℕ → ℕ)
variable (a1 : ℕ) (H1 : d ≠ 0)
variable (H2 : a1 = 1)
variable (H3 : (a_n 3)^2 = a1 * (a_n 13))
variable (H4 : a_n n = a1 + (n - 1) * d)
variable (H5 : S_n n = (n * (a1 + a_n n)) / 2)

theorem min_value (n : ℕ) (Hn : 1 ≤ n) : 
  ∃ n, ∀ m, 1 ≤ m → (2 * S_n n + 16) / (a_n n + 3) ≥ (2 * S_n m + 16) / (a_n m + 3) ∧ (2 * S_n n + 16) / (a_n n + 3) = 4 :=
sorry

end min_value_l194_194129


namespace harmonic_zero_on_y_axis_l194_194258

noncomputable def D := { p : ℝ × ℝ | p.1 > 0 ∧ p.2 ≠ 0 }

theorem harmonic_zero_on_y_axis (u : ℝ × ℝ → ℝ) 
  (hd : ∀ p ∈ D, differentiable_at ℝ u p)
  (hb : bdd_above {c : ℝ | ∃ p ∈ D, u p = c})
  (hu : ∀ p ∈ D, laplacian 2 u p = 0)
  (hy : ∀ x > 0, u (x, 0) = 0) :
  ∀ p ∈ D, u p = 0 :=
sorry

end harmonic_zero_on_y_axis_l194_194258


namespace range_m_l194_194166

def A (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_m (m : ℝ) :
  (∀ x, B m x → A x) ↔ -3 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_m_l194_194166


namespace no_elements_divisible_by_10_in_T_l194_194271

def g (x : ℤ) : ℤ := x^2 + 5 * x + 3

def T : set ℤ := {n | n ≥ 0 ∧ n ≤ 30}

theorem no_elements_divisible_by_10_in_T : 
  ∀ t ∈ T, ¬ (g t % 10 = 0) := 
by 
  intros t ht 
  sorry

end no_elements_divisible_by_10_in_T_l194_194271


namespace sum_of_roots_l194_194723

theorem sum_of_roots (x : ℝ) : (x - 4)^2 = 16 → x = 8 ∨ x = 0 := by
  intro h
  have h1 : x - 4 = 4 ∨ x - 4 = -4 := by
    sorry
  cases h1
  case inl h2 =>
    rw [h2] at h
    exact Or.inl (by linarith)
  case inr h2 =>
    rw [h2] at h
    exact Or.inr (by linarith)

end sum_of_roots_l194_194723


namespace part1_part2_l194_194506

-- Definition of the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Magnitude of a vector
noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Problem Part 1: Prove that |a - 3b| = 2√29
theorem part1 : vector_magnitude (a.1 - 3 * b.1, a.2 - 3 * b.2) = 2 * real.sqrt 29 :=
by
  sorry

-- Problem Part 2: Prove that if k * a + b is parallel to a - 3 * b, then k = -1/3
theorem part2 (k : ℝ) (h : (k * a + b).fst * (a - 3 * b).snd = (k * a + b).snd * (a - 3 * b).fst) :
  k = -1/3 :=
by
  sorry

end part1_part2_l194_194506


namespace sin_arccos_and_identity_l194_194462

noncomputable theory

-- Definitions from the problem conditions
def θ : ℝ := Real.arccos (1 / 5)
def cos_θ : ℝ := Real.cos θ
def sin_θ : ℝ := Real.sin θ

-- Statement of the theorem
theorem sin_arccos_and_identity :
  sin θ = 2 * Real.sqrt 6 / 5 ∧ sin θ ^ 2 + cos θ ^ 2 = 1 :=
by
  sorry

end sin_arccos_and_identity_l194_194462


namespace laura_reps_8lb_dumbbells_l194_194976

theorem laura_reps_8lb_dumbbells (n : ℚ) :
  (2 * 10 * 30 = 2 * 8 * n) → n = 37.5 :=
by
  intro h₁
  have h₂ : 2 * 10 * 30 = 600 := rfl
  rw h₂ at h₁
  linarith

end laura_reps_8lb_dumbbells_l194_194976


namespace correct_value_is_48_l194_194341

-- Given conditions as Lean definitions
def num_observations : ℕ := 50
def initial_mean : ℚ := 36
def incorrect_value : ℚ := 23
def corrected_mean : ℚ := 36.5

-- The proof problem
theorem correct_value_is_48 :
  let initial_sum := num_observations * initial_mean
  let new_sum := num_observations * corrected_mean
  ∃ correct_value : ℚ, initial_sum - incorrect_value + correct_value = new_sum ∧ correct_value = 48 :=
by
  -- Adding the statement to be proven
  let initial_sum := num_observations * initial_mean
  let new_sum := num_observations * corrected_mean
  existsi (48 : ℚ)
  split
  -- Proof of initial_sum - incorrect_value + correct_value = new_sum
  { sorry }
  -- Proof of correct_value = 48
  { refl }

end correct_value_is_48_l194_194341


namespace Carol_width_eq_24_l194_194793

-- Given conditions
def Carol_length : ℕ := 5
def Jordan_length : ℕ := 2
def Jordan_width : ℕ := 60

-- Required proof: Carol's width is 24 considering equal areas of both rectangles
theorem Carol_width_eq_24 (w : ℕ) (h : Carol_length * w = Jordan_length * Jordan_width) : w = 24 := 
by sorry

end Carol_width_eq_24_l194_194793


namespace mark_and_james_need_to_buy_2_dice_to_play_their_game_l194_194641

variable (total_dice_needed : ℕ)
variable (mark_total_dice : ℕ)
variable (mark_percent_12_sided : ℝ)
variable (james_total_dice : ℕ)
variable (james_percent_12_sided : ℝ)

def number_of_dice_to_buy : ℕ :=
  let mark_12_sided := mark_percent_12_sided * mark_total_dice
  let james_12_sided := james_percent_12_sided * james_total_dice
  total_dice_needed - mark_12_sided.toNat - james_12_sided.toNat

theorem mark_and_james_need_to_buy_2_dice_to_play_their_game :
  total_dice_needed = 14 →
  mark_total_dice = 10 →
  mark_percent_12_sided = 0.6 →
  james_total_dice = 8 →
  james_percent_12_sided = 0.75 →
  number_of_dice_to_buy total_dice_needed mark_total_dice mark_percent_12_sided james_total_dice james_percent_12_sided = 2 :=
by
  sorry

end mark_and_james_need_to_buy_2_dice_to_play_their_game_l194_194641


namespace coordinates_of_point_M_in_second_quadrant_l194_194228

theorem coordinates_of_point_M_in_second_quadrant
  (x y : ℝ)
  (second_quadrant : x < 0 ∧ 0 < y)
  (dist_to_x_axis : abs(y) = 1)
  (dist_to_y_axis : abs(x) = 2) :
  (x, y) = (-2, 1) :=
sorry

end coordinates_of_point_M_in_second_quadrant_l194_194228


namespace max_S_min_S_under_diff_constraint_l194_194308

open Finset

def sum_of_products (xs : Fin₅ → ℕ) : ℕ :=
  ∑ ij in Fin₅.pairs, xs ij.1 * xs ij.2

-- Definitions and conditions
def x₁ := 401
def x₂ := 401
def x₃ := 401
def x₄ := 401
def x₅ := 402

def x₁' := 400
def x₂' := 400
def x₃' := 402
def x₄' := 402
def x₅' := 402

-- Maximum S
theorem max_S : 
  ∑ i j in antidiagonal 5 (\[401, 401, 401, 401, 402\].nth! : Fin₅ → ℕ) = 
  1609614 :=
sorry

-- Minimum S under |x_i - x_j| ≤ 2
theorem min_S_under_diff_constraint : 
  ∑ i j in antidiagonal 5 (\[400, 400, 402, 402, 402\].nth! : Fin₅ → ℕ) = 
  1609612 :=
sorry

end max_S_min_S_under_diff_constraint_l194_194308


namespace gracie_height_is_56_l194_194174

noncomputable def Gracie_height : Nat := 56

theorem gracie_height_is_56 : Gracie_height = 56 := by
  sorry

end gracie_height_is_56_l194_194174


namespace area_parallelogram_proof_l194_194457

variable (p q : ℝ^3)
variable (a b : ℝ^3)

noncomputable def area_parallelogram {p q : ℝ^3} (h₁ : |p| = 7) (h₂ : |q| = 2) 
  (h₃ : ⟪p, q⟫ = cos (π / 3)) : ℝ :=
  let a := p + 4 * q
  let b := 2 * p - q
  abs (a × b)

theorem area_parallelogram_proof (h₁ : |p| = 7) (h₂ : |q| = 2) (h₃ : ⟪p, q⟫ = cos (π / 3)) :
  area_parallelogram p q h₁ h₂ h₃ = 63 * sqrt 3 :=
sorry

end area_parallelogram_proof_l194_194457


namespace sum_of_roots_of_polynomial_l194_194073

theorem sum_of_roots_of_polynomial (a b c : ℝ) (h : 3*a^3 - 7*a^2 + 6*a = 0) : 
    (∀ x, 3*x^2 - 7*x + 6 = 0 → x = a ∨ x = b ∨ x = c) →
    (∀ (x : ℝ), (x = a ∨ x = b ∨ x = c → 3*x^3 - 7*x^2 + 6*x = 0)) → 
    a + b + c = 7 / 3 :=
sorry

end sum_of_roots_of_polynomial_l194_194073


namespace part_a_part_b_l194_194981

variable 
  {A B C D E P : Type}
  [ConvexPentagon ABCDE Set]
  (circ_abe : Circle ABCD)
  (intersects_AC : ∃ P, P ∈ circ_abe ∧ is_on_line P AC)
  (bisects_BAE : bisects AC ∠BAE)
  (bisects_DCB : bisects AC ∠DCB)
  (angle_AEB_90 : ∠AEB = 90)
  (angle_BDC_90 : ∠BDC = 90)

namespace Geometry

theorem part_a :
  P = circumcenter (Triangle BDE) :=
sorry

theorem part_b :
  cyclic_quadrilateral A C D E :=
sorry

end Geometry

end part_a_part_b_l194_194981


namespace find_first_image_l194_194778

-- Definitions based on the problem conditions
variables (g g' g'' d'' A' A'' : Type)

-- Assuming g and d are lines that intersect at a right angle
axiom perpendicular_lines : Prop 

-- The projected first image of line g
noncomputable def first_projection (g g' g'' d'' A' A'' : Type) : Type := sorry

/-- Given the elements g', g'', d'', and intersections A'' and A'. Prove that the first image of line g is the perpendicular projection from A' to the plane containing d, assuming g intersects d at a right angle. -/
theorem find_first_image 
    (g g' g'' d'' A' A'' : Type)
    (h : perpendicular_lines) : first_projection g g' g'' d'' A' A'' = ??? := sorry

end find_first_image_l194_194778


namespace part_one_part_two_l194_194887

variable (m : ℝ)

def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log x / Real.log 3 ∧ 1/27 < x ∧ x < 9}
def C : Set ℝ := {x | x^2 + m * x - 6 * m < 0}

theorem part_one : (A ∩ B) = {x | -2 < x ∧ x < 2} :=
by
  sorry

theorem part_two : (A ∪ B ⊆ C) → (m > 25) :=
by
  sorry

end part_one_part_two_l194_194887


namespace problem_statement_l194_194124

variables {Point : Type*} {Line Plane : Type*} [HasSubset Line Plane]
variables {m : Line} {α β : Plane}

-- Assuming parallelism and subset relations
variables (is_parallel : Plane -> Plane -> Prop)
variables (is_subline : Line -> Plane -> Prop)

-- Axiom relating subset and parallel planes
axiom parallel_planes_implies_parallel_lines (m : Line) (α β : Plane) :
  is_subline m α → is_parallel α β → is_parallel m β

-- The main theorem to be proved
theorem problem_statement :
  ∀ (m : Line) (α β : Plane), is_subline m α → is_parallel α β → is_parallel m β :=
begin
  intros,
  exact parallel_planes_implies_parallel_lines m α β a b,
end

end problem_statement_l194_194124


namespace find_z2_range_m_l194_194526

/-- Given conditions for part 1 --/
def z1 : ℂ := -2 + complex.I
def condition1 : Prop := z1 * z2 = -5 + 5 * complex.I

/-- Finding z2 --/
def z2 : ℂ := 3 - complex.I
theorem find_z2 : z2 = 3 - complex.I := by
  sorry

/-- Given conditions for part 2 --/
def z3 (m : ℝ) : ℂ := (3 - z2) * ((m^2 - 2 * m - 3) + (m - 1) * complex.I)
def condition2 (m : ℝ) : Prop := 
  z3(m).re > 0 ∧ z3(m).im < 0

/-- Proving the range of m --/
theorem range_m : ∀ m : ℝ, condition2 m → -1 < m ∧ m < 1 := by
  sorry

end find_z2_range_m_l194_194526


namespace tangent_line_equation_l194_194535

noncomputable def sqrt_tangent_line_perpendicular (x y : ℝ) : Prop :=
  y = Real.sqrt x ∧
  ∃ (m : ℝ), (y = m * x + c) ∧ (perpendicular (line (y = m * x + c)) (line (y = -2 * x - 4)))

theorem tangent_line_equation (x_0 y_0 : ℝ) (h1 : y_0 = Real.sqrt x_0) (h_slope : (1 / (2 * Real.sqrt x_0)) = (1 / 2))
  (h_perp : ∀ (x y : ℝ), y = -2 * x - 4 → is_perpendicular (line (y = 1 / 2 * x)) (line (y = -2 * x - 4))) :
  ∃ (C : ℝ), (x - 2 * y + C = 0) :=
begin
  sorry
end

end tangent_line_equation_l194_194535


namespace train_pass_jogger_time_l194_194016

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 60
noncomputable def initial_distance_m : ℝ := 350
noncomputable def train_length_m : ℝ := 250

noncomputable def relative_speed_m_per_s : ℝ := 
  ((train_speed_km_per_hr - jogger_speed_km_per_hr) * 1000) / 3600

noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m

noncomputable def time_to_pass_s : ℝ := total_distance_m / relative_speed_m_per_s

theorem train_pass_jogger_time :
  abs (time_to_pass_s - 42.35) < 0.01 :=
by 
  sorry

end train_pass_jogger_time_l194_194016


namespace points_same_color_separed_by_two_l194_194367

theorem points_same_color_separed_by_two (circle : Fin 239 → Bool) : 
  ∃ i j : Fin 239, i ≠ j ∧ (i + 2) % 239 = j ∧ circle i = circle j :=
by
  sorry

end points_same_color_separed_by_two_l194_194367


namespace cube_surface_area_equals_353_l194_194765

noncomputable def volume_of_prism : ℝ := 5 * 3 * 30
noncomputable def edge_length_of_cube (volume : ℝ) : ℝ := (volume)^(1/3)
noncomputable def surface_area_of_cube (edge_length : ℝ) : ℝ := 6 * edge_length^2

theorem cube_surface_area_equals_353 :
  surface_area_of_cube (edge_length_of_cube volume_of_prism) = 353 := by
sorry

end cube_surface_area_equals_353_l194_194765


namespace total_cost_correct_l194_194109

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end total_cost_correct_l194_194109


namespace solve_for_x_l194_194698

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = 1 / 3 * (6 * x + 45)) : x = 0 :=
sorry

end solve_for_x_l194_194698


namespace complex_modulus_l194_194587

open Complex

theorem complex_modulus (z : ℂ) (hz : (3 - I) / z = 1 + I) : complex.abs z = Real.sqrt 5 :=
sorry

end complex_modulus_l194_194587


namespace complex_conjugate_of_z_l194_194461

theorem complex_conjugate_of_z (z : ℂ) (h : (1 + complex.i) * z = complex.abs (√3 - complex.i)) :
  complex.conj z = 1 + complex.i :=
sorry

end complex_conjugate_of_z_l194_194461


namespace four_digit_square_palindromes_are_zero_l194_194179

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

-- Define the main theorem statement
theorem four_digit_square_palindromes_are_zero : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) → 
             is_palindrome n → 
             (∃ m : ℕ, n = m * m) → 
             n = 0 :=
by
  sorry

end four_digit_square_palindromes_are_zero_l194_194179


namespace square_line_product_l194_194338

theorem square_line_product (b : ℝ) 
  (h1 : ∃ y1 y2, y1 = -1 ∧ y2 = 4) 
  (h2 : ∃ x1, x1 = 3) 
  (h3 : (4 - (-1)) = (5 : ℝ)) 
  (h4 : ((∃ b1, b1 = 3 + 5 ∨ b1 = 3 - 5) → b = b1)) :
  b = -2 ∨ b = 8 → b * 8 = -16 :=
by sorry

end square_line_product_l194_194338


namespace triangle_area_ratio_l194_194925

theorem triangle_area_ratio (P Q R T : Type) (h_co_planar : IsCoPlanar P Q R T)
  (h_p_not_collinear : ¬Collinear P Q R) (QT TR : ℝ) (h_QT : QT = 8) (h_TR : TR = 12)
  (h_altitude : ∃ h : ℝ, altitude P Q R h T) :
  let area_PQT := (1 / 2) * QT * h
  let area_PTR := (1 / 2) * TR * h
  let ratio := area_PQT / area_PTR
  ratio = 2 / 3 :=
by
  sorry

end triangle_area_ratio_l194_194925


namespace infinitely_many_multiples_of_7_l194_194356

noncomputable def a : ℕ → ℕ
| 1     := 1
| (n+1) := a n + a (n / 2)

theorem infinitely_many_multiples_of_7 :
  ∃ᶠ n in at_top, 7 ∣ a n := 
sorry

end infinitely_many_multiples_of_7_l194_194356


namespace sweetest_sugar_water_l194_194371

def initial_sugar_mass := 25
def initial_water_mass := 100
def initial_solution_mass := initial_sugar_mass + initial_water_mass
def initial_sugar_concentration := (initial_sugar_mass / initial_solution_mass) * 100

def sugar_mass_jia := initial_sugar_mass + (0.2 * 50)
def total_mass_jia := initial_solution_mass + 50
def concentration_jia := (sugar_mass_jia / total_mass_jia) * 100

def sugar_mass_yi := initial_sugar_mass + 20
def total_mass_yi := initial_solution_mass + 20
def concentration_yi := (sugar_mass_yi / total_mass_yi) * 100

def sugar_mass_bing := initial_sugar_mass + (0.5 * 40)
def total_mass_bing := initial_solution_mass + 40
def concentration_bing := (sugar_mass_bing / total_mass_bing) * 100

theorem sweetest_sugar_water : concentration_yi > concentration_jia ∧ concentration_yi > concentration_bing :=
by
  sorry

end sweetest_sugar_water_l194_194371


namespace find_other_number_l194_194335

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 9240) (h_gcd : Nat.gcd a b = 33) (h_a : a = 231) : b = 1320 :=
sorry

end find_other_number_l194_194335


namespace area_union_original_and_reflected_l194_194233

-- Define the original vertices
def A : (ℝ × ℝ) := (3, 2)
def B : (ℝ × ℝ) := (5, 5)
def C : (ℝ × ℝ) := (6, 1)

-- Define the reflected vertices
def A' : (ℝ × ℝ) := (11, 2)
def B' : (ℝ × ℝ) := (9, 5)
def C' : (ℝ × ℝ) := (8, 1)

-- Function to calculate the area of a triangle given three vertices
def triangleArea (P Q R : ℝ × ℝ) : ℝ :=
  abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)

-- The goal is to prove the area of the union of the original and the reflected triangle
theorem area_union_original_and_reflected :
  triangleArea A B C + triangleArea A' B' C' = 11 := by
  sorry

end area_union_original_and_reflected_l194_194233


namespace incenter_on_altitude_l194_194257

universe u

variables {Point : Type u} [HasMetric Point] [AddGroup Point] [Module ℝ Point]
variables (A B C P Q A_0 B_1 C_1 H : Point)

-- Definitions for conditions
def isAcuteTriangle (A B C : Point) : Prop := sorry
def isMidpoint (A_0 B C : Point) : Prop := sorry
def isAltitude (H A B C B_1 C_1 : Point) : Prop := sorry
def isParallel (A B C : Point) : Prop := sorry
def isIncenter (I : Point) (P A_0 Q : Point) : Prop := sorry
def liesOn (x : Point) (line_pts : Point × Point) : Prop := sorry

-- Problem statement
theorem incenter_on_altitude 
  (h1 : isAcuteTriangle A B C)
  (h2 : isAltitude H A B C B_1 C_1)
  (h3 : isMidpoint A_0 B C)
  (h4 : isParallel A_0 B_1 A_0 C_1 A) :
  ∃ I : Point, isIncenter I P A_0 Q ∧ liesOn I (A, H) :=
sorry

end incenter_on_altitude_l194_194257


namespace arun_takes_less_time_l194_194945

variable (distance : ℝ)
variable (V_Arun V_Anil V'_Arun : ℝ)
variable (T_Anil T_Arun T'_Arun : ℝ)

noncomputable def T_Anil := distance / V_Anil
noncomputable def T_Arun := distance / V_Arun
noncomputable def T'_Arun := distance / V'_Arun

-- Conditions
axiom V_Arun_eq : V_Arun = 5
axiom distance_eq : distance = 30
axiom time_relation : T_Arun = T_Anil + 2
axiom double_speed : V'_Arun = 2 * V_Arun

-- Proof Problem
theorem arun_takes_less_time : (T_Anil - T'_Arun) = 1 := by
  -- Proving steps would go here, but we use sorry for now.
  sorry

end arun_takes_less_time_l194_194945


namespace time_after_2500_minutes_l194_194726

/-- 
To prove that adding 2500 minutes to midnight on January 1, 2011 results in 
January 2 at 5:40 PM.
-/
theorem time_after_2500_minutes :
  let minutes_in_a_day := 1440 -- 24 hours * 60 minutes
  let minutes_in_an_hour := 60
  let start_time_minutes := 0 -- Midnight January 1, 2011 as zero minutes
  let total_minutes := 2500
  let resulting_minutes := start_time_minutes + total_minutes
  let days_passed := resulting_minutes / minutes_in_a_day
  let remaining_minutes := resulting_minutes % minutes_in_a_day
  let hours := remaining_minutes / minutes_in_an_hour
  let minutes := remaining_minutes % minutes_in_an_hour
  days_passed = 1 ∧ hours = 17 ∧ minutes = 40 :=
by
  -- Proof to be filled in
  sorry

end time_after_2500_minutes_l194_194726


namespace cyclist_is_jean_l194_194960

theorem cyclist_is_jean (x x' y y' : ℝ) (hx : x' = 4 * x) (hy : y = 4 * y') : x < y :=
by
  sorry

end cyclist_is_jean_l194_194960


namespace probability_of_two_tails_given_conditions_l194_194471

-- Define the problem conditions and question
def fair_coin : Type := 🛑    -- represents the fair coin flip, stuck for brevity

-- Define the conditions for Debra stopping
def sequence_stops (seq : list fair_coin) : Prop :=
  (list.consecutive seq tt tt) ∨ (list.consecutive seq hh hh) -- tt for two tails, hh for two heads

-- Define the probability of getting two tails in a row, seeing a second head before a second tail
def probability_two_tails_after_second_head : ℚ :=
  1/24
  
theorem probability_of_two_tails_given_conditions : ℚ 
  probability_two_tails_after_second_head :=
by
  sorry

end probability_of_two_tails_given_conditions_l194_194471


namespace total_surface_area_of_T_l194_194265

-- Define the cube properties
def cube_edge_length : ℝ := 10
def point_A : ℝ × ℝ × ℝ := (0, 0, 0)
def point_B : ℝ × ℝ × ℝ := (cube_edge_length, 0, 0)
def point_D : ℝ × ℝ × ℝ := (0, cube_edge_length, 0)
def point_E : ℝ × ℝ × ℝ := (0, 0, cube_edge_length)
def point_G : ℝ × ℝ × ℝ := (cube_edge_length, cube_edge_length, cube_edge_length)
def point_P : ℝ × ℝ × ℝ := (3, 0, 0)
def point_Q : ℝ × ℝ × ℝ := (0, 3, 0)
def point_R : ℝ × ℝ × ℝ := (0, 0, 3)

def AP_length : ℝ := 3
def AQ_length : ℝ := 3
def AR_length : ℝ := 3

-- Proof problem to show the total surface area of the solid T
theorem total_surface_area_of_T : 
  cube_edge_length = 10 ∧
  ∃ (P Q R : ℝ × ℝ × ℝ), (P = point_P ∧ Q = point_Q ∧ R = point_R ∧
  (dist point_A P = AP_length ∧ dist point_A Q = AQ_length ∧ dist point_A R = AR_length))
  → total_surface_area T = 660 := 
sorry

end total_surface_area_of_T_l194_194265


namespace min_positive_period_of_f_intervals_f_increasing_max_min_values_f_l194_194158

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 3 + x / 2)

theorem min_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 4 * Real.pi := sorry

theorem intervals_f_increasing : ∀ k : ℤ, ∀ x, 
  4 * ↑k * Real.pi - 8 * Real.pi / 3 ≤ x ∧ x ≤ 4 * ↑k * Real.pi - 2 * Real.pi / 3 → 
  f.derivative x > 0 := sorry

theorem max_min_values_f : ∃ (max min : ℝ), 
  max = 2 ∧ min = -Real.sqrt 3 ∧ 
  ∀ x ∈ Set.Icc (-Real.pi) Real.pi, min ≤ f x ∧ f x ≤ max := sorry

end min_positive_period_of_f_intervals_f_increasing_max_min_values_f_l194_194158


namespace cost_of_two_pans_is_20_l194_194971

variable (cost_of_pan : ℕ)

-- Conditions
def pots_cost := 3 * 20
def total_cost := 100
def pans_eq_cost := total_cost - pots_cost
def cost_of_pan_per_pans := pans_eq_cost / 4

-- Proof statement
theorem cost_of_two_pans_is_20 
  (h1 : pots_cost = 60)
  (h2 : total_cost = 100)
  (h3 : pans_eq_cost = total_cost - pots_cost)
  (h4 : cost_of_pan_per_pans = pans_eq_cost / 4)
  : 2 * cost_of_pan_per_pans = 20 :=
by sorry

end cost_of_two_pans_is_20_l194_194971


namespace prob_point_in_triangle_AMN_l194_194954

section
variables (A B C P M N : Type) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space P] [metric_space M] [metric_space N]
variables [tri : triangle A B C]
variables [AB : dist A B] [BC : dist B C] [CA : dist C A]
variables (P ∈ triangle A B C)

noncomputable def point_probability :=
  ∀ (P : point) (P ∈ triangle A (midpoint A B) (midpoint A C)), 
  probability P = 1 / 4 

theorem prob_point_in_triangle_AMN : 
  let A_B := 6 in
  let B_C := 8 in
  let C_A := 10 in
  point_probability A B C P A_B B_C C_A :=
sorry
end

end prob_point_in_triangle_AMN_l194_194954


namespace find_center_and_radius_l194_194676

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

theorem find_center_and_radius :
  (∀ x y : ℝ, circle_equation x y ↔ (x + 2)^2 + y^2 = 4) →
  (∃ center radius : ℝ, center = (-2, 0) ∧ radius = 2) :=
begin
  sorry,
end

end find_center_and_radius_l194_194676


namespace reflection_of_a_over_b_l194_194098

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 2)

-- Define the projection of vector a onto vector b
def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_ab := a.1 * b.1 + a.2 * b.2
  let dot_bb := b.1 * b.1 + b.2 * b.2
  let scalar := dot_ab / dot_bb
  (scalar * b.1, scalar * b.2)

-- Define the reflection of vector a over vector b using the projection
def reflect (a b : ℝ × ℝ) : ℝ × ℝ :=
  let p := proj a b
  (2 * p.1 - a.1, 2 * p.2 - a.2)

-- Prove the reflection equals to (-2/13, 29/13)
theorem reflection_of_a_over_b :
  reflect a b = (-2/13 : ℝ, 29/13 : ℝ) :=
by
  sorry

end reflection_of_a_over_b_l194_194098


namespace fourth_term_of_sequence_l194_194417

-- Given conditions
def first_term : ℕ := 5
def fifth_term : ℕ := 1280

-- Definition of the common ratio
def common_ratio (a : ℕ) (b : ℕ) : ℕ := (b / a)^(1 / 4)

-- Function to calculate the nth term of a geometric sequence
def nth_term (a r n : ℕ) : ℕ := a * r^(n - 1)

-- Prove the fourth term of the geometric sequence is 320
theorem fourth_term_of_sequence 
    (a : ℕ) (b : ℕ) (a_pos : a = first_term) (b_eq : nth_term a (common_ratio a b) 5 = b) : 
    nth_term a (common_ratio a b) 4 = 320 := by
  sorry

end fourth_term_of_sequence_l194_194417


namespace calculate_parallel_segment_length_l194_194926

theorem calculate_parallel_segment_length :
  ∀ (d : ℝ), 
    ∃ (X Y Z P : Type) 
    (XY YZ XZ : ℝ), 
    XY = 490 ∧ 
    YZ = 520 ∧ 
    XZ = 560 ∧ 
    ∃ (D D' E E' F F' : Type),
      (D ≠ E ∧ E ≠ F ∧ F ≠ D') ∧  
      (XZ - (d * (520/490) + d * (520/560))) = d → d = 268.148148 :=
by
  sorry

end calculate_parallel_segment_length_l194_194926


namespace systematic_sampling_method_l194_194768

-- Define the problem conditions
def total_rows : Nat := 40
def seats_per_row : Nat := 25
def attendees_left (row : Nat) : Nat := if row < total_rows then 18 else 0

-- Problem statement to be proved: The method used is systematic sampling.
theorem systematic_sampling_method :
  (∀ r : Nat, r < total_rows → attendees_left r = 18) →
  (seats_per_row = 25) →
  (∃ k, k > 0 ∧ ∀ r, r < total_rows → attendees_left r = 18 + k * r) →
  True :=
by
  intro h1 h2 h3
  sorry

end systematic_sampling_method_l194_194768


namespace count_4_digit_palindromic_squares_is_2_l194_194184

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def count_4_digit_palindromic_squares : ℕ :=
  (Finset.range 100).filter (λ n, 32 ≤ n ∧ is_4_digit_number (n * n) ∧ is_palindrome (n * n)).card

theorem count_4_digit_palindromic_squares_is_2 : count_4_digit_palindromic_squares = 2 :=
  sorry

end count_4_digit_palindromic_squares_is_2_l194_194184


namespace volume_of_tetrahedron_l194_194815

theorem volume_of_tetrahedron 
  (A B C D : Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  (angle_ABC_BCD : real)
  (area_ABC area_BCD : real)
  (length_BC : real)
  (condition1 : angle_ABC_BCD = π/4)
  (condition2 : area_ABC = 150)
  (condition3 : area_BCD = 90)
  (condition4 : length_BC = 12) :
  ∃ volume : real, volume = 375 * real.sqrt 2 :=
begin
  sorry
end

end volume_of_tetrahedron_l194_194815


namespace solve_problem1_solve_problem2_l194_194142

noncomputable def problem1 (a b c A B : ℝ) : Prop :=
  ∀ (cos_B sin_B : ℝ), a * (cos_B + sqrt 3 * sin_B) = b + c →
   sin A * cos B + sqrt 3 * (sin A) * (sin B) = sin B + sin (A + B) →
   sqrt 3 * sin A - cos A = 1 →
   (A = π / 3)

noncomputable def problem2 (b c a : ℝ) : Prop :=
  b + c = 7 →
  a = sqrt 7 →
  (area : ℝ) :=
  (area = 1 / 2 * b * c * (sqrt 3 / 2)) →
  (area = 7 * sqrt 3 / 2)

axiom cos_B_sin_B (B : ℝ) : ∃ cos_B sin_B, cos_B = cos B ∧ sin_B = sin B

theorem solve_problem1 (a b c A B : ℝ) :
  (∀ cos_B sin_B, a * (cos_B + sqrt 3 * sin_B) = b + c →
  sin A * cos B + sqrt 3 * (sin A) * (sin B) = sin B + sin (A + B) →
  sqrt 3 * sin A - cos A = 1 → A = π / 3) :=
by
  sorry

theorem solve_problem2 (a b c : ℝ) (cos_A : ℝ) :
  b + c = 7 →
  a = sqrt 7 →
  cos_A = 1 / 2 →
  let area := 1 / 2 * b * c * (sqrt 3 / 2) in
  area = 7 * sqrt 3 / 2 :=
by
  sorry

end solve_problem1_solve_problem2_l194_194142


namespace latus_rectum_of_parabola_l194_194820

theorem latus_rectum_of_parabola (x : ℝ) :
  (∀ x, y = (-1 / 4 : ℝ) * x^2) → y = (-1 / 2 : ℝ) :=
sorry

end latus_rectum_of_parabola_l194_194820


namespace sum_of_sequence_l194_194165

noncomputable def sequence_term (n : ℕ) : ℚ :=
  1 / ((3 * n - 2) * (3 * n + 1))

def partial_sum (n : ℕ) : ℚ :=
  ∑ i in finset.range n, sequence_term (i + 1)

theorem sum_of_sequence (n : ℕ) : partial_sum n = n / (3 * n + 1) :=
by
  induction n with k hk
  case zero =>
    simp [partial_sum, sequence_term]
  case succ k =>
    sorry

end sum_of_sequence_l194_194165


namespace projection_coords_l194_194115

variable (a b : ℝ × ℝ)
variable (dot_product : ℝ)

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_prod := u.1 * v.1 + u.2 * v.2
  let mag_sq := u.1 * u.1 + u.2 * u.2
  (dot_prod / mag_sq) • u

theorem projection_coords :
  let a := (0, 5)
  let b := (2, -1)
  proj a b = (0, -1) := by
    sorry

end projection_coords_l194_194115


namespace necessary_but_not_sufficient_condition_l194_194696

variables {a b : ℤ}

theorem necessary_but_not_sufficient_condition : (¬(a = 1) ∨ ¬(b = 2)) ↔ ¬(a + b = 3) :=
by
  sorry

end necessary_but_not_sufficient_condition_l194_194696


namespace probability_of_mutual_generation_l194_194598

theorem probability_of_mutual_generation :
  let elements := ["gold", "wood", "water", "fire", "earth"],
      mutual_generation := [(gold, water), (water, wood), (wood, fire), (fire, earth), (earth, gold)],
      total_pairs := Nat.choose 5 2
  in (mutual_generation.length : ℚ) / (total_pairs : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_mutual_generation_l194_194598


namespace rate_of_drawing_barbed_wire_l194_194669

-- Define the conditions
variable (A : ℕ) (gate_width cost_rate total_cost perimeter : ℕ)
variable (s P total_wire_length : ℕ)

-- Define the given problem context
def area_square_field (A : ℕ) : Prop := A = 3136
def gates : Prop := 2 * gate_width = 2
def total_cost_wire (total_cost : ℕ) : Prop := total_cost = 999
def perimeter_square (P : ℕ) : Prop := P = 4 * s
def field_side_length (s : ℕ) (A : ℕ) : Prop := s * s = A
def wire_length_without_gates (total_wire_length perimeter gate_width : ℕ) : Prop := 
  total_wire_length = perimeter - 2 * gate_width
def rate_per_meter (rate_per_meter total_cost total_wire_length : ℕ) : Prop := 
  rate_per_meter = total_cost / total_wire_length

-- Prove the rate of drawing barbed wire per meter
theorem rate_of_drawing_barbed_wire (rate_per_meter : ℕ) :
  ∃ A s P total_wire_length gate_width total_cost, 
    area_square_field A ∧ 
    field_side_length s A ∧ 
    gates ∧ 
    perimeter_square P ∧ 
    wire_length_without_gates total_wire_length P gate_width ∧ 
    total_cost_wire total_cost ∧ 
    rate_per_meter rate_per_meter total_cost total_wire_length → 
    rate_per_meter = 4.5 :=
by
  sorry

end rate_of_drawing_barbed_wire_l194_194669


namespace ratio_of_new_time_to_previous_time_l194_194439

-- Given conditions
def distance : ℕ := 288
def initial_time : ℕ := 6
def new_speed : ℕ := 32

-- Question: Prove the ratio of the new time to the previous time is 3:2
theorem ratio_of_new_time_to_previous_time :
  (distance / new_speed) / initial_time = 3 / 2 :=
by
  sorry

end ratio_of_new_time_to_previous_time_l194_194439


namespace sum_of_third_largest_and_third_smallest_l194_194397

theorem sum_of_third_largest_and_third_smallest :
  let digits := {2, 5, 6, 9}
  let all_numbers := List.join (digits.toFinset.powerset
    .filter (fun s => s.card = 3)
    .image (λ s, s.toList.permutations.map (λ l, l.foldl (λ r x => r * 10 + x) 0)))
  all_numbers.nth_le 2 (by simp [all_numbers]) + all_numbers.nth_le (all_numbers.length - 3) (by simp [all_numbers]) = 1221 := sorry

end sum_of_third_largest_and_third_smallest_l194_194397


namespace ratio_of_living_room_to_bedroom_is_4_l194_194297

noncomputable def bedroom_light_usage_per_hour : ℕ := 6
noncomputable def office_light_usage_per_hour : ℕ := 3 * bedroom_light_usage_per_hour
noncomputable def total_energy_used : ℕ := 96
noncomputable def hours_left_on : ℕ := 2

def energy_usage_bedroom_light : ℕ := bedroom_light_usage_per_hour * hours_left_on
def energy_usage_office_light : ℕ := office_light_usage_per_hour * hours_left_on

def energy_usage_living_room_light : ℕ := total_energy_used - (energy_usage_bedroom_light + energy_usage_office_light)

def ratio_living_room_to_bedroom : ℕ := energy_usage_living_room_light / energy_usage_bedroom_light

theorem ratio_of_living_room_to_bedroom_is_4 :
  ratio_living_room_to_bedroom = 4 :=
sorry

end ratio_of_living_room_to_bedroom_is_4_l194_194297


namespace min_value_xy_inv_xy_l194_194511

theorem min_value_xy_inv_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_sum : x + y = 2) :
  ∃ m : ℝ, m = xy + 4 / xy ∧ m ≥ 5 :=
by
  sorry

end min_value_xy_inv_xy_l194_194511


namespace wall_length_l194_194567

theorem wall_length
    (brick_length brick_width brick_height : ℝ)
    (wall_height wall_width : ℝ)
    (num_bricks : ℕ)
    (wall_length_cm : ℝ)
    (h_brick_volume : brick_length * brick_width * brick_height = 1687.5)
    (h_wall_volume :
        wall_length_cm * wall_height * wall_width
        = (brick_length * brick_width * brick_height) * num_bricks)
    (h_wall_height : wall_height = 600)
    (h_wall_width : wall_width = 22.5)
    (h_num_bricks : num_bricks = 7200) :
    wall_length_cm / 100 = 9 := 
by
  sorry

end wall_length_l194_194567


namespace tower_height_l194_194715

def tan_sum_formula (α β : ℝ) : Prop :=
  tan (α + β) = (tan α + tan β) / (1 - tan α * tan β)

theorem tower_height
  (α β x : ℝ)
  (h_tan_alpha : tan α = x / 50)
  (h_tan_beta : tan β = x / 100)
  (h_sum_angles : α + β = real.pi / 4)
  (h_tan_sum : tan (real.pi / 4) = 1) :
  x = 28.08 :=
by
  sorry

end tower_height_l194_194715


namespace city_visited_by_B_l194_194743

variable (visits : String → String → Bool)
variable (studentA studentB studentC : String)

-- Conditions
def A_statement := visits studentA "A" ∧ ¬ visits studentA "B" ∧ visits studentA "A" ∧ ¬ (visits studentB "A" ∨ visits studentB "B")

def B_statement := ¬ visits studentB "C"

def C_statement := visits studentA "A" = visits studentB "A" = visits studentC "A"

-- Theorem to prove
theorem city_visited_by_B : visits studentB "A" :=
by
  sorry

end city_visited_by_B_l194_194743


namespace smallest_n_with_372_sequence_l194_194325

theorem smallest_n_with_372_sequence : ∀ (m n : ℕ), 
  Nat.gcd m n = 1 ∧ m < n ∧ (∃ k : ℕ, ((m * 10^k) % n).natAbs / (10^k : ℝ) = 0.372) → n = 377 := by
  sorry

end smallest_n_with_372_sequence_l194_194325


namespace exist_fewer_than_2_pow_m_distinct_sums_l194_194617

theorem exist_fewer_than_2_pow_m_distinct_sums
  (m : ℕ) (hm : m > 0) (a : fin m → ℕ) (ha : ∀ i, 0 < a i) :
  ∃ (n : ℕ) (b : fin n → ℕ), (n < 2^m) ∧ 
    (∀ S T : finset (fin n), S ≠ T → S.sum (λ i, b i) ≠ T.sum (λ i, b i)) ∧
    (∀ i : fin m, ∃ (S : finset (fin n)), S.sum (λ j, b j) = a i) :=
by sorry

end exist_fewer_than_2_pow_m_distinct_sums_l194_194617


namespace num_quadruples_satisfying_product_98_l194_194070

theorem num_quadruples_satisfying_product_98 :
  ∃ (a b c d : ℕ), a * b * c * d = 98 ∧ (number_of_solutions (λ (a b c d : ℕ), a * b * c * d = 98) = 40) := sorry

end num_quadruples_satisfying_product_98_l194_194070


namespace min_vert_segment_length_is_7_l194_194465

open Real

noncomputable def min_vertical_segment_length : ℝ :=
  let f := λ x : ℝ, 2 * abs x
  let g := λ x : ℝ, x^2 - 2*x - 3
  let h := λ x : ℝ, abs (f x - g x)
  infi h

theorem min_vert_segment_length_is_7 : min_vertical_segment_length = 7 :=
  sorry

end min_vert_segment_length_is_7_l194_194465


namespace terminal_side_on_y_axis_l194_194541

theorem terminal_side_on_y_axis (α : ℝ) (h : sin α = 1 ∨ sin α = -1) : exists y, (0, y) ≠ (y, y) :=
by
  sorry

end terminal_side_on_y_axis_l194_194541


namespace parallel_planes_l194_194857

-- Definitions and conditions
variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions
axiom parallel_lines : m ∥ n
axiom m_perp_alpha : m ⟂ α
axiom n_perp_beta : n ⟂ β

-- Theorem to prove
theorem parallel_planes : α ∥ β :=
by {
  -- Placeholder for the proof
  sorry
}

end parallel_planes_l194_194857


namespace alternating_sum_2_to_100_l194_194055

/-- A Lean statement describing the alternation series from 2 to 100 --/
theorem alternating_sum_2_to_100 : 
  (Finset.range 50).sum (λ n, (2 * (n + 1) - 1)) + 100 = 51 := 
by
  sorry

end alternating_sum_2_to_100_l194_194055


namespace midpoint_of_segment_l194_194389

def midpoint (p₁ p₂ : ℝ × ℝ) : ℝ × ℝ :=
  ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)

theorem midpoint_of_segment : midpoint (5, -3) (-7, 9) = (-1, 3) :=
by
  -- Automatically generate the midpoint using the provided formula
  have m := midpoint (5, -3) (-7, 9)
  -- Directly verify the result
  have h1 : m.1 = (-1) := rfl
  have h2 : m.2 = 3 := rfl
  rw [h1, h2]
  -- Conclusion that both coordinates match the desired answer
  sorry

end midpoint_of_segment_l194_194389


namespace area_of_triangle_l194_194085

theorem area_of_triangle (x y : ℝ) :
  let line := 2 * x - 5 * y - 10 = 0 in
  ∃ A B C : ℝ × ℝ,
    A = (0, 0) ∧ B = (0, -2) ∧ C = (5, 0) ∧
    (let base := dist (0, 0) (5, 0)
     let height := dist (0, 0) (0, -2)
     (1/2) * base * height = 5) := sorry

end area_of_triangle_l194_194085


namespace inequality_proof_l194_194989

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) >= (2 / 3) * (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l194_194989


namespace max_tunnel_construction_mileage_find_value_of_a_l194_194795

-- Define variables and conditions
variables (x a : ℝ)

def total_mileage := 56
def ordinary_road_mileage := 32
def elevated_road_mileage_in_q1 := total_mileage - ordinary_road_mileage - x
def minimum_elevated_road_mileage := 7 * x

-- Problem 1: Maximum tunnel construction mileage
theorem max_tunnel_construction_mileage :
  elevated_road_mileage_in_q1 >= minimum_elevated_road_mileage → x <= 3 :=
by intros h; sorry

-- Define cost variables and conditions for Q2
def ordinary_road_cost_per_km_q1 := 1
def elevated_road_cost_per_km_q1 := 2
def tunnel_road_cost_per_km_q1 := 4
def ordinary_road_mileage_q2 := ordinary_road_mileage - 9 * a
def elevated_road_mileage_q2 := elevated_road_mileage_in_q1 - 2 * a
def tunnel_road_mileage_q2 := x + a
def elevated_road_cost_per_km_q2 := elevated_road_cost_per_km_q1 + 0.5 * a

-- Problem 2: Value of a
theorem find_value_of_a (h1: elevated_road_mileage_in_q1 >= minimum_elevated_road_mileage) :
  (ordinary_road_cost_per_km_q1 * ordinary_road_mileage +
   elevated_road_cost_per_km_q1 * elevated_road_mileage_in_q1 +
   tunnel_road_cost_per_km_q1 * x = 
   (ordinary_road_cost_per_km_q1 * ordinary_road_mileage_q2 +
   elevated_road_cost_per_km_q2 * elevated_road_mileage_q2 +
   tunnel_road_cost_per_km_q1 * tunnel_road_mileage_q2)) → 
   a = 3 / 2 :=
by intros h2; sorry

end max_tunnel_construction_mileage_find_value_of_a_l194_194795


namespace cos_780_eq_half_sin_minus_45_eq_minus_sqrt2_div2_l194_194056

-- Statement for Part 1: Proving that cos 780° = 1/2
theorem cos_780_eq_half: cos (780 : ℝ) = (1 / 2 : ℝ) :=
by sorry

-- Statement for Part 2: Proving that sin (-45°) = -sqrt(2)/2
theorem sin_minus_45_eq_minus_sqrt2_div2: sin (-45 : ℝ) = (- (Real.sqrt 2) / 2 : ℝ) :=
by sorry

end cos_780_eq_half_sin_minus_45_eq_minus_sqrt2_div2_l194_194056


namespace trigonometric_values_ratio_of_trig_functions_l194_194505

variables {α : ℝ}
variables (cos_alpha : ℝ) (sin_alpha : ℝ) (tan_alpha : ℝ)

def in_second_quadrant (α : ℝ) : Prop :=
  α > π / 2 ∧ α < π

theorem trigonometric_values
  (h1 : cos_alpha = -4/5)
  (h2 : in_second_quadrant α) :
  sin_alpha = 3/5 ∧ tan_alpha = -3/4 :=
begin
  sorry
end

theorem ratio_of_trig_functions
  (h1 : cos_alpha = -4/5)
  (h2 : sin_alpha = 3/5) :
  (2 * sin_alpha + 3 * cos_alpha) / (cos_alpha - sin_alpha) = 6/7 :=
begin
  sorry
end

end trigonometric_values_ratio_of_trig_functions_l194_194505


namespace max_k_value_l194_194554

def f (x : ℝ) : ℝ := x + x * Real.log x

noncomputable def F (x : ℝ) : ℝ := (x + x * Real.log x) / (x - 2)

theorem max_k_value (k : ℤ) (h : ∀ x > 2, k * (x - 2) < f x) : k ≤ 4 :=
by
  sorry

end max_k_value_l194_194554


namespace complex_imag_part_of_z_l194_194324

theorem complex_imag_part_of_z (z : ℂ) (h : z * (2 + ⅈ) = 3 - 6 * ⅈ) : z.im = -3 := by
  sorry

end complex_imag_part_of_z_l194_194324


namespace gcf_180_270_450_l194_194386

theorem gcf_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  sorry

end gcf_180_270_450_l194_194386


namespace max_subsets_l194_194637

def set_M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Definition of a condition where any two elements appear together in at most 2 subsets
def valid_subsets (M : Set ℕ) (A : Finset (Finset ℕ)) : Prop :=
  ∀ x y ∈ M, x ≠ y → (A.filter (λ S, x ∈ S ∧ y ∈ S)).card ≤ 2

-- Main statement
theorem max_subsets (A : Finset (Finset ℕ)) (hA : valid_subsets set_M A) :
  A.card ≤ 8 :=
sorry

end max_subsets_l194_194637


namespace collinear_midpoints_and_points_l194_194596

theorem collinear_midpoints_and_points
  {A B C D F G B1 C1 : Type*}
  (triangle_ABC : nonisomorphic_acutetriangle A B C)
  (midpoint_B1 : midpoint B1 A C)
  (midpoint_C1 : midpoint C1 A B)
  (D_on_BC : between B C D)
  (right_angle_AFC : ∠ A F C = 90)
  (angle_DCF_eq_FCA : ∠ D C F = ∠ F C A)
  (right_angle_AGB : ∠ A G B = 90)
  (angle_CBG_eq_GBA : ∠ C B G = ∠ G B A) :
  collinear [B1, C1, F, G] :=
sorry

end collinear_midpoints_and_points_l194_194596


namespace find_f1_l194_194799

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x, x ≠ 1 / 2 → f x + f ((x + 2) / (1 - 2 * x)) = x) :
  f 1 = 7 / 6 :=
sorry

end find_f1_l194_194799


namespace symmetric_points_origin_l194_194148

theorem symmetric_points_origin (a b : ℝ) (h1 : 1 = -b) (h2 : a = 2) : a + b = 1 := by
  sorry

end symmetric_points_origin_l194_194148


namespace log_evaluation_l194_194480

theorem log_evaluation : log 3 243 - log 3 (1 / 27) + log 3 9 = 10 := 
by sorry

end log_evaluation_l194_194480


namespace number_of_integer_solutions_l194_194823

theorem number_of_integer_solutions : 
  { (m n r : ℕ) // m > 0 ∧ n > 0 ∧ r > 0 ∧ m * n + n * r + m * r = 2 * (m + n + r) }.card 
  = 7 :=
sorry

end number_of_integer_solutions_l194_194823


namespace num_4_digit_palindromic_squares_l194_194205

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_n (n : ℕ) : Prop :=
  32 ≤ n ∧ n ≤ 99

theorem num_4_digit_palindromic_squares : ∃ (count : ℕ), count = 3 ∧ ∀ n, valid_n n → is_4_digit (n^2) → is_palindrome (n^2) :=
sorry

end num_4_digit_palindromic_squares_l194_194205


namespace is_min_value_of_x_plus_3y_l194_194282

open Real

noncomputable def min_value (x y : ℝ) : ℝ :=
  x + 3 * y

theorem is_min_value_of_x_plus_3y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
    (h3 : (1 / Real.cbrt (x + 3)) + (1 / Real.cbrt (y + 3)) = 1 / 2) :
  min_value x y ≥ 11.86 :=
sorry

end is_min_value_of_x_plus_3y_l194_194282


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l194_194421

-- Define the ink length of a figure
def ink_length (n : ℕ) : ℕ := 5 * n

-- Part (a): Determine the ink length of Figure 4.
theorem ink_length_figure_4 : ink_length 4 = 20 := by
  sorry

-- Part (b): Determine the difference between the ink length of Figure 9 and the ink length of Figure 8.
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 5 := by
  sorry

-- Part (c): Determine the ink length of Figure 100.
theorem ink_length_figure_100 : ink_length 100 = 500 := by
  sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l194_194421


namespace expand_polynomial_l194_194811

theorem expand_polynomial : 
  ∀ (x : ℝ), (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 :=
by
  intro x
  sorry

end expand_polynomial_l194_194811


namespace modulus_of_z_is_two_l194_194577

open Complex

noncomputable def z : ℂ := sorry
def z_conjugate : ℂ := conj z
def condition1 : Prop := z_conjugate = conj z
def condition2 : Prop := z * z_conjugate = 4

theorem modulus_of_z_is_two (h1 : condition1) (h2 : condition2) : abs z = 2 :=
by
  sorry

end modulus_of_z_is_two_l194_194577


namespace framing_feet_required_l194_194747

noncomputable def original_width := 5
noncomputable def original_height := 7
noncomputable def enlargement_factor := 4
noncomputable def border_width := 3
noncomputable def inches_per_foot := 12

theorem framing_feet_required :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (final_width + final_height)
  let framing_feet := perimeter / inches_per_foot
  framing_feet = 10 :=
by
  sorry

end framing_feet_required_l194_194747


namespace perimeter_tetrahedron_inequality_l194_194983

theorem perimeter_tetrahedron_inequality {A B C D K L M N : Point} 
  (face1 : Face ABC)
  (face2 : Face ABD)
  (face3 : Face ACD)
  (face4 : Face BCD) 
  (hK : K ∈ face1)
  (hL : L ∈ face2)
  (hM : M ∈ face3)
  (hN : N ∈ face4) :
  perimeter K L M N ≤ (4 / 3) * perimeter A B C D :=
sorry

end perimeter_tetrahedron_inequality_l194_194983


namespace set_union_intersection_l194_194562

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {2, 3, 4}

theorem set_union_intersection :
  (A ∩ B) ∪ C = {1, 2, 3, 4} := 
by
  sorry

end set_union_intersection_l194_194562


namespace cos_double_angle_nec_but_not_suff_l194_194675

theorem cos_double_angle_nec_but_not_suff (α : ℝ) :
  (cos (2 * α) = 1 / 2 ↔ (sin α = 1 / 2 ∨ sin α = -1 / 2)) ∧ (sin α = 1 / 2 → cos (2 * α) = 1 / 2) :=
by
  sorry

end cos_double_angle_nec_but_not_suff_l194_194675


namespace f_diff_l194_194153

def f (n : ℕ) : ℚ :=
  ∑ i in finset.range (2*n + 2), 1 / (n + 1 + i)

theorem f_diff (n : ℕ) : f (n + 1) - f n = 1 / (2 * n + 3) :=
by {
  sorry
}

end f_diff_l194_194153


namespace selection_ways_864_l194_194020

theorem selection_ways_864 : 
  let multiple_choice := 12
  let fill_in_blank := 4
  let open_ended := 6
  (∑ (x : Fin multiple_choice.card), (∑ (y : Fin fill_in_blank.card), (∑ (z : Fin open_ended.card), 1)) +
   (∑ (x₁ x₂ : Fin multiple_choice.card), (∑ (z : Fin open_ended.card), 1)) +
   (∑ (x : Fin multiple_choice.card), (∑ (z₁ z₂ : Fin open_ended.card), 1)))  = 864 :=
sorry

end selection_ways_864_l194_194020


namespace sale_price_for_profit_l194_194350

theorem sale_price_for_profit 
    (C : ℝ) 
    (h1 : ∃ S, S - C = C - 448) 
    (h2 : 975 = 1.5 * C) : 
    ∃ S, S = 852 :=
by
  have hC : C = 650 :=
  by sorry
  use (2 * C - 448)
  simp [hC]

end sale_price_for_profit_l194_194350


namespace exponential_decreasing_iff_frac_inequality_l194_194507

theorem exponential_decreasing_iff_frac_inequality (a : ℝ) :
  (0 < a ∧ a < 1) ↔ (a ≠ 1 ∧ a * (a - 1) ≤ 0) :=
by
  sorry

end exponential_decreasing_iff_frac_inequality_l194_194507


namespace mean_home_runs_correct_l194_194788

-- Definitions for the conditions
def number_of_home_runs : List ℕ := [5, 6, 7, 8, 9]
def number_of_players : List ℕ := [4, 5, 3, 2, 2]
def total_home_runs : ℕ := List.sum (List.zipWith (*) number_of_home_runs number_of_players)
def total_players : ℕ := List.sum number_of_players
def mean_home_runs : ℚ := total_home_runs / total_players 

-- The proof goal
theorem mean_home_runs_correct : mean_home_runs = 6.5625 := by
  -- The proof would go here
  sorry

end mean_home_runs_correct_l194_194788


namespace total_revenue_correct_l194_194803

-- Define the initial inventory, prices, and conditions
def initial_inventory := [
  ("Samsung Galaxy S20", 14, 800),
  ("iPhone 12", 8, 1000),
  ("Google Pixel 5", 7, 700),
  ("OnePlus 8T", 6, 600)
]

def discounts := [
  ("Samsung Galaxy S20", 0.10),
  ("iPhone 12", 0.15),
  ("Google Pixel 5", 0.05),
  ("OnePlus 8T", 0.20)
]

def tax_rates := [
  ("Samsung Galaxy S20", 0.12),
  ("iPhone 12", 0.10),
  ("Google Pixel 5", 0.08),
  ("OnePlus 8T", 0.15)
]

def adjustments := [
  ("Samsung Galaxy S20", -2),
  ("iPhone 12", -1),
  ("Google Pixel 5", 1),
  ("OnePlus 8T", -1)
]

def ending_inventory := [
  ("Samsung Galaxy S20", 10),
  ("iPhone 12", 5),
  ("Google Pixel 5", 8),
  ("OnePlus 8T", 3)
]

noncomputable def calculate_revenue : Float :=
  let samsung_revenue := 2 * 806.40
      iphone_revenue := 2 * 935
      google_revenue := 0 * 718.20
      oneplus_revenue := 2 * 552
  samsung_revenue + iphone_revenue + google_revenue + oneplus_revenue

theorem total_revenue_correct :
  calculate_revenue = 4586.80 := by
  sorry

end total_revenue_correct_l194_194803


namespace greatest_median_l194_194293

-- Define the conditions in the problem
variable (n : ℕ := 9)          -- Number of nonnegative numbers
variable (avg : ℝ := 10)       -- Average of the numbers
variable (sum : ℝ := n * avg)  -- Sum of the numbers

-- Define the median as the fifth smallest number when the numbers are sorted
def is_median (m : ℝ) (lst : list ℝ) : Prop :=
  list.length lst = n ∧ 
  (∀ i < 5, 0 ≤ list.nth_le lst i sorry) ∧ -- first four numbers are nonnegative
  (∀ i ≥ 5, list.nth_le lst i sorry ≥ m)    -- the last five numbers are at least m

-- Define the proof statement
theorem greatest_median (m : ℝ) (lst : list ℝ) (h_nonneg : ∀ (x : ℝ), x ∈ lst → 0 ≤ x)
  (h_avg : list.sum lst = sum) : m ≤ 18 :=
by {
  -- proof to be filled in
  sorry
}

end greatest_median_l194_194293


namespace area_ratio_l194_194985

attribute [instance] classical.prop_decidable

structure EquilateralTriangle (V : Type) [add_group V] :=
(A B C : V)
(equilateral : dist A B = dist B C ∧ dist B C = dist C A)

def P := Type

variable (V : Type) [inner_product_space ℝ V]

def extend_point (a b : V) (k : ℝ) : V := b + k • (b - a)

noncomputable def extended_triangle (T : EquilateralTriangle V) : EquilateralTriangle V :=
{ A := extend_point T.A T.C 4,
  B := extend_point T.A T.B 2,
  C := extend_point T.B T.C 3,
  equilateral := 
  begin
    sorry -- proof of new triangle equilateral condition
  end }

noncomputable def area (T : EquilateralTriangle V) : ℝ := 
let a := dist T.A T.B in (a^2 * sqrt 3) / 4

theorem area_ratio :
  ∀ (T : EquilateralTriangle V), 
  area (extended_triangle T) / area T = 25 :=
begin
  intros T,
  have h1 : dist (extend_point T.A T.B 2) (T.B) = 3 * dist T.A T.B, from sorry,
  have h2 : dist (extend_point T.B T.C 3) (T.C) = 4 * dist T.B T.C, from sorry,
  have h3 : dist (extend_point T.C T.A 4) (T.A) = 5 * dist T.C T.A, from sorry,
  let area_original := area T,
  let area_extended := area (extended_triangle T),
  calc
    area_extended / area_original = ((5 * dist T.A T.B)^2 * sqrt 3 / 4) / (dist T.A T.B ^ 2 * sqrt 3 / 4) : by sorry
    ... = 25 : by ring,
end

end area_ratio_l194_194985


namespace unit_price_of_mascots_max_type_A_mascots_l194_194666

theorem unit_price_of_mascots (x y : ℕ) (h₁ : 1.2 * x = y) (h₂ : 3000 * (1 / y + 1 / x) = 110) : x = 50 ∧ y = 60 :=
sorry

theorem max_type_A_mascots (b : ℕ := 300) (B_price : ℕ := 50) (A_price : ℕ := 60) (budget : ℕ := 16800)
  (h₁ : 1.2 * B_price = A_price) 
  (h₂ : 60 * (300 - b) + 50 * b ≤ budget) : b ≤ 180 :=
sorry

end unit_price_of_mascots_max_type_A_mascots_l194_194666


namespace girls_ran_miles_l194_194450

theorem girls_ran_miles (laps_boys : ℕ) (extra_laps : ℕ) (distance_per_lap : ℚ)
  (h1 : laps_boys = 124) (h2 : extra_laps = 48) (h3 : distance_per_lap = 5 / 13) :
  (laps_boys + extra_laps) * distance_per_lap = 860 / 13 :=
by
  -- Ensure that the input conditions are verified
  simp [h1, h2, h3]
  -- Calculate the total laps run by the girls
  have laps_girls := laps_boys + extra_laps
  simp [laps_girls]
  -- Calculate the total distance run by girls in miles
  have distance_girls := laps_girls * distance_per_lap
  simp [distance_girls]
  -- State the final equality
  sorry

end girls_ran_miles_l194_194450


namespace find_a_plus_b_l194_194627

theorem find_a_plus_b (a b : ℕ) 
  (h1 : 2^(2 * a) + 2^b + 5 = k^2) : a + b = 4 ∨ a + b = 5 :=
sorry

end find_a_plus_b_l194_194627


namespace no_three_times_age_ago_l194_194415

theorem no_three_times_age_ago (F D : ℕ) (h₁ : F = 40) (h₂ : D = 40) (h₃ : F = 2 * D) :
  ¬ ∃ x, F - x = 3 * (D - x) :=
by
  sorry

end no_three_times_age_ago_l194_194415


namespace approx_one_over_one_plus_alpha_approx_one_over_one_minus_alpha_approx_N_over_N_plus_alpha_l194_194683

theorem approx_one_over_one_plus_alpha (α : ℝ) (h : |α| < 1) :
  (1 / (1 + α)) ≈ (1 - α) :=
sorry

theorem approx_one_over_one_minus_alpha (α : ℝ) (h : |α| < 1) :
  (1 / (1 - α)) ≈ (1 + α) :=
sorry

theorem approx_N_over_N_plus_alpha (N α : ℝ) (h : |α / N| < 1) :
  (N / (N + α)) ≈ (1 - α / N) :=
sorry

end approx_one_over_one_plus_alpha_approx_one_over_one_minus_alpha_approx_N_over_N_plus_alpha_l194_194683


namespace range_of_g_l194_194826

-- Define the function g(x)
def g (x : ℝ) : ℝ := if x ≠ -5 then 3 * (x - 4) else 0 -- The value at x = -5 is irrelevant as g is undefined there

-- The main theorem
theorem range_of_g : set.range g = set.univ \ {-27} :=
by
  sorry -- proof omitted

end range_of_g_l194_194826


namespace valid_sequences_count_l194_194212

def g (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n < 3 then 0
  else g (n - 4) + 3 * g (n - 5) + 3 * g (n - 6)

theorem valid_sequences_count : g 17 = 37 :=
  sorry

end valid_sequences_count_l194_194212


namespace base7_sum_remainder_l194_194720

theorem base7_sum_remainder :
  let n1 := 2 * 7 + 4 
  let n2 := 3 * 49 + 6 * 7 + 4
  let n3 := 4 * 7 + 3
  let n4 := 1 * 7 + 2 
  let n5 := 3
  let n6 := 1 in
  (n1 + n2 + n3 + n4 + n5 + n6) % 6 = 3 := 
by
  sorry

end base7_sum_remainder_l194_194720


namespace lakeside_fitness_center_ratio_l194_194041

theorem lakeside_fitness_center_ratio (f m c : ℕ)
  (h_avg_age : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  f = 3 * (m / 6) ∧ f = 3 * (c / 2) :=
by
  sorry

end lakeside_fitness_center_ratio_l194_194041


namespace total_area_of_field_l194_194017

theorem total_area_of_field 
  (A_s : ℕ) 
  (h₁ : A_s = 315)
  (A_l : ℕ) 
  (h₂ : A_l - A_s = (1/5) * ((A_s + A_l) / 2)) : 
  A_s + A_l = 700 := 
  by 
    sorry

end total_area_of_field_l194_194017


namespace no_intersections_root_of_quadratic_l194_194354

theorem no_intersections_root_of_quadratic (x : ℝ) :
  ¬(∃ x, (y = x) ∧ (y = x - 3)) ↔ (x^2 - 3 * x = 0) := by
  sorry

end no_intersections_root_of_quadratic_l194_194354


namespace total_tables_l194_194013

variables (F T : ℕ)

-- Define the given conditions
def condition1 := F = 16
def condition2 := 4 * F + 3 * T = 124

-- State the theorem given the conditions to prove the total number of tables.
theorem total_tables (h1 : condition1) (h2 : condition2) : F + T = 36 :=
by
  -- This is a placeholder as we are skipping the proof itself
  sorry

end total_tables_l194_194013


namespace five_of_six_pairwise_coprime_l194_194659

theorem five_of_six_pairwise_coprime {a b c d e f : ℕ} (ha : 1000 ≤ a ∧ a < 10000)
  (hb : 1000 ≤ b ∧ b < 10000) (hc : 1000 ≤ c ∧ c < 10000)
  (hd : 1000 ≤ d ∧ d < 10000) (he : 1000 ≤ e ∧ e < 10000)
  (hf : 1000 ≤ f ∧ f < 10000) (h : ∀ (x y : ℕ), x ≠ y → x ∈ {a, b, c, d, e, f} → y ∈ {a, b, c, d, e, f} → Nat.gcd x y = 1) :
  ∃ (s : Finset ℕ), s.card = 5 ∧ (∀ (x y : ℕ), x ≠ y → x ∈ s → y ∈ s → Nat.gcd x y = 1) :=
by
  sorry

end five_of_six_pairwise_coprime_l194_194659


namespace exists_triad_l194_194703

open Classical

variables (A B C : Type) [Fintype A] [Fintype B] [Fintype C]
variables (n : ℕ)
variable (friend : ∀ x y : A ∪ B ∪ C, Prop)
variable (symm_friend : ∀ x y, friend x y → friend y x)

-- Each city has exactly n citizens
variable (ha : Fintype.card A = n)
variable (hb : Fintype.card B = n)
variable (hc : Fintype.card C = n)

-- Each citizen has exactly (n + 1) friends from the other two cities
variable (hf : ∀ x, Fintype.card { y // friend x y ∧ (¬(x ∈ A) → y ∈ A) ∧ (¬(x ∈ B) → y ∈ B) ∧ (¬(x ∈ C) → y ∈ C) } = n + 1)

theorem exists_triad :
  ∃ (a : A) (b : B) (c : C), friend a b ∧ friend a c ∧ friend b c :=
sorry

end exists_triad_l194_194703


namespace sum_of_possible_N_l194_194830

theorem sum_of_possible_N (N : ℕ) 
  (lines : Finset (Set (ℝ × ℝ))) 
  (h_num_lines : lines.card = 5) 
  (distinct_points : Finset (ℝ × ℝ))
  (N_values : Finset ℕ := {x | ∃ lines' ⊆ lines, lines'.card = 2 ∧ ∃ p, lines' = {line | p ∈ line}}) :
  (N_values.sum id = 53) :=
sorry

end sum_of_possible_N_l194_194830


namespace circle_radius_l194_194096

theorem circle_radius (x y : ℝ) : x^2 - 10*x + y^2 + 4*y + 13 = 0 → ∃ r : ℝ, r = 4 :=
by
  -- sorry here to indicate that the proof is skipped
  sorry

end circle_radius_l194_194096


namespace field_area_difference_l194_194018

theorem field_area_difference :
  ∃ V : ℝ, 
    let A := 500 - 225 in
    let B := 225 in
    (A - B) = (1 / 5) * V ∧ V = 250 :=
begin
  sorry
end

end field_area_difference_l194_194018


namespace intersection_of_hyperbola_and_circle_is_line_l194_194099

theorem intersection_of_hyperbola_and_circle_is_line :
  ∃ (points : set (ℝ × ℝ)), 
    (∀ point ∈ points, (point.1 * point.2 = 18) ∧ (point.1 ^ 2 + point.2 ^ 2 = 36)) ∧
    (is_line points) :=
by
  -- Since the proof is not required, we use sorry here
  sorry

def is_line (points : set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ (∀ point ∈ points, a * point.1 + b * point.2 = 0)

end intersection_of_hyperbola_and_circle_is_line_l194_194099


namespace magnitude_diff_of_perpendicular_vectors_l194_194134

theorem magnitude_diff_of_perpendicular_vectors 
  (k : ℝ) 
  (a : ℝ × ℝ := (-2, k)) 
  (b : ℝ × ℝ := (2, 4)) 
  (h_perpendicular : (a.1 * b.1 + a.2 * b.2) = 0) : 
  (real.sqrt (((a.1 - b.1) ^ 2) + ((a.2 - b.2) ^ 2)) = 5) := 
by sorry

end magnitude_diff_of_perpendicular_vectors_l194_194134


namespace maryville_population_increase_l194_194443

def average_people_added_per_year (P2000 P2005 : ℕ) (period : ℕ) : ℕ :=
  (P2005 - P2000) / period
  
theorem maryville_population_increase :
  let P2000 := 450000
  let P2005 := 467000
  let period := 5
  average_people_added_per_year P2000 P2005 period = 3400 :=
by
  sorry

end maryville_population_increase_l194_194443


namespace gcd_180_150_210_l194_194718

def prime_factors (n : ℕ) := multiset ℕ -- A multiset to store prime factors

noncomputable def factor180 : prime_factors 180 := {2, 2, 3, 3, 5}
noncomputable def factor150 : prime_factors 150 := {2, 3, 5, 5}
noncomputable def factor210 : prime_factors 210 := {2, 3, 5, 7}

theorem gcd_180_150_210 : Nat.gcd (Nat.gcd 180 150) 210 = 30 :=
  by
    -- This space will contain the proof steps
    sorry

end gcd_180_150_210_l194_194718


namespace parallelogram_AGHI_has_perimeter_60_l194_194590

noncomputable def parallelogram_perimeter (A B C G H I: ℝ) (AB AC BC GH HI AI BG IC AG AI: ℝ) (h1: AB = AC) (h2: AB = 30) (h3: BC = 28) 
(h4: GH = AC) (h5: HI = AB) (h6: AG + GH + HI + AI = AG + BG + IC + AI) : ℝ := 
AB + AC

theorem parallelogram_AGHI_has_perimeter_60 (A B C G H I : Type) 
(h1: AB = AC) (h2: AB = 30) (h3: BC = 28) (h4: GH = AC) (h5: HI = AB) 
(h6: AG = 1/2 * AB) (h7: AI = 1/2 * AC): parallelogram_perimeter A B C G H I 30 30 28 GH HI AI 30 30 0.0 0.0 0.0 AI = 60 :=
by
  -- Since AG = 1/2 * AB and AI = 1/2 * AC
  have p1 : AG + GH + HI + AI = AG + 30 + 30 + AI, from sorry,
  -- And we know AB + AC = 60
  have p2: 30 + 30 = 60, from sorry,
  -- Therefore, the perimeter of parallelogram AGHI is 60
  exact sorry

end parallelogram_AGHI_has_perimeter_60_l194_194590


namespace ice_cream_cone_volume_eq_radius_sphere_l194_194783

noncomputable def radius_of_sphere_equal_volume (r_cone : ℝ) (h_cone : ℝ) (V_sphere_eq : ℝ) : ℝ := 
  let V_cone := (1 / 3) * real.pi * (r_cone ^ 2) * h_cone
  let r_sphere := real.cbrt ((3 / 4) * V_cone / real.pi)
  r_sphere

theorem ice_cream_cone_volume_eq_radius_sphere :
  radius_of_sphere_equal_volume 2 8 (2 * real.cbrt 2) = 2 * real.cbrt 2 := by
  sorry

end ice_cream_cone_volume_eq_radius_sphere_l194_194783


namespace number_divided_by_189_l194_194082

noncomputable def target_number : ℝ := 3486

theorem number_divided_by_189 :
  target_number / 189 = 18.444444444444443 :=
by
  sorry

end number_divided_by_189_l194_194082


namespace need_to_buy_more_dice_l194_194644

theorem need_to_buy_more_dice (mark_dice : ℕ) (mark_percent_12_sided : ℕ) (james_dice : ℕ) (james_percent_12_sided : ℕ) (total_needed_12_sided : ℕ) :
  mark_dice = 10 → mark_percent_12_sided = 60 →
  james_dice = 8 → james_percent_12_sided = 75 →
  total_needed_12_sided = 14 →
  (total_needed_12_sided - (mark_dice * mark_percent_12_sided / 100 + james_dice * james_percent_12_sided / 100)) = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end need_to_buy_more_dice_l194_194644


namespace evaluate_expression_l194_194049

theorem evaluate_expression :
  ( (1 / 2) ^ (-2) - abs (sqrt 3 - 2) - 8 * (cos (real.pi / 3))^2 = sqrt 3) :=
by
  sorry

end evaluate_expression_l194_194049


namespace four_digit_palindrome_squares_count_l194_194206

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem four_digit_palindrome_squares_count : (Finset.filter (λ n, is_palindrome (n * n)) (Finset.range 100)).card = 2 := by
  sorry

end four_digit_palindrome_squares_count_l194_194206


namespace probability_green_slope_l194_194366

-- Conditions as definitions
variable (α β : ℝ)

theorem probability_green_slope (hα : 0 ≤ α ∧ α ≤ π/2) (hβ : 0 ≤ β ∧ β ≤ π/2) :
  ∃ γ : ℝ, (γ = real.arccos (√(1 - (real.cos α) ^ 2 - (real.cos β) ^ 2))) ∧ 
    ((real.cos γ) ^ 2 = 1 - (real.cos α) ^ 2 - (real.cos β) ^ 2) :=
begin
  -- Skipping the proof to focus on the statement equivalency
  sorry
end

end probability_green_slope_l194_194366


namespace sin_tan_identity_l194_194533

theorem sin_tan_identity (x : Real) (hx1 : cos x = -sqrt 2 / 10) (hx2 : x ∈ set.Ioo (π / 2) π) :
  sin x = 7 * sqrt 2 / 10 ∧ tan (2 * x + π / 4) = 31 / 17 :=
by
  sorry

end sin_tan_identity_l194_194533


namespace alex_received_12_cookies_l194_194175

theorem alex_received_12_cookies :
  ∃ y: ℕ, (∀ s: ℕ, y = s + 8 ∧ s = y / 3) → y = 12 := by
  sorry

end alex_received_12_cookies_l194_194175


namespace vasya_max_points_l194_194655

theorem vasya_max_points (cards : Finset (Fin 36)) 
  (petya_hand vasya_hand : Finset (Fin 36)) 
  (h_disjoint : Disjoint petya_hand vasya_hand)
  (h_union : petya_hand ∪ vasya_hand = cards)
  (h_card : cards.card = 36)
  (h_half : petya_hand.card = 18 ∧ vasya_hand.card = 18) : 
  ∃ max_points : ℕ, max_points = 15 := 
sorry

end vasya_max_points_l194_194655


namespace range_of_a_l194_194903

noncomputable def increasing_function (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x ≤ f y

def f (a x : ℝ) : ℝ := Real.cos (2 * x) + a * Real.cos (π / 2 + x)

theorem range_of_a (a : ℝ) :
  increasing_function (f a) (set.Ioo (π / 6) (π / 2)) →
  a ∈ set.Iic (-4) := 
sorry

end range_of_a_l194_194903


namespace isosceles_triangle_area_l194_194597

theorem isosceles_triangle_area
  (A B C D : ℝ)
  (h_isosceles : (A = B ∧ A = C))
  (h_base : B + C = 8)
  (h_bisect : D * 2 = B) :
  1 / 2 * 8 * real.sqrt (15^2 - (B / 2)^2) = 4 * real.sqrt 209 :=
by
  sorry

end isosceles_triangle_area_l194_194597


namespace solve_unit_prices_solve_purchasing_schemes_l194_194319

noncomputable def unit_prices (x y : ℕ) : Prop :=
  15 * x + 20 * y = 520 ∧ 20 * x + 17 * y = 616

theorem solve_unit_prices : ∃ x y, unit_prices x y ∧ x = 24 ∧ y = 8 :=
begin
  sorry
end

def purchasing_schemes (m n : ℕ) : Prop :=
  14 ≤ m ∧ m ≤ 16 ∧ 28 - m ≤ m ∧ 24 * m + 8 * (28 - m) ≤ 480

theorem solve_purchasing_schemes :
  ∃ m n, purchasing_schemes m n ∧ 
  ((m = 14 ∧ n = 14) ∨ 
   (m = 15 ∧ n = 13) ∨ 
   (m = 16 ∧ n = 12)) :=
begin
  sorry
end

end solve_unit_prices_solve_purchasing_schemes_l194_194319


namespace inv_of_15_mod_1003_l194_194054

theorem inv_of_15_mod_1003 : ∃ x : ℕ, x ≤ 1002 ∧ 15 * x ≡ 1 [MOD 1003] ∧ x = 937 :=
by sorry

end inv_of_15_mod_1003_l194_194054


namespace sum_n_bn_equals_a1_l194_194978

theorem sum_n_bn_equals_a1 {a b : ℕ → ℝ}
  (h_dec : ∀ n, a (n+1) ≤ a n)
  (h_pos : ∀ n, 0 < a n)
  (h_lim : tendsto a at_top (𝓝 0))
  (h_bn : ∀ n, b n = a n - 2 * a (n+1) + a (n+2) ∧ 0 ≤ b n) :
  ∑' n, n * b n = a 1 :=
by
  sorry

end sum_n_bn_equals_a1_l194_194978


namespace frank_total_cost_l194_194113

-- Conditions from the problem
def cost_per_bun : ℝ := 0.1
def number_of_buns : ℕ := 10
def cost_per_bottle_of_milk : ℝ := 2
def number_of_bottles_of_milk : ℕ := 2
def cost_of_carton_of_eggs : ℝ := 3 * cost_per_bottle_of_milk

-- Question and Answer
theorem frank_total_cost : 
  let cost_of_buns := cost_per_bun * number_of_buns in
  let cost_of_milk := cost_per_bottle_of_milk * number_of_bottles_of_milk in
  let cost_of_eggs := cost_of_carton_of_eggs in
  cost_of_buns + cost_of_milk + cost_of_eggs = 11 :=
by
  sorry

end frank_total_cost_l194_194113


namespace area_of_rectangle_l194_194263

noncomputable def rectangle_area {a : ℝ} (h : a > 0) (h_quad : 2 * (a * a / 3) + 50 = a * a / 6 * 2 * (3 / 2 + 1)) : ℝ :=
  2 * a * a

theorem area_of_rectangle (a : ℝ) (h : a > 0) (h_quad : 50 + (a * a / 3) = 100) : 
  rectangle_area h h_quad = 300 :=
sorry

end area_of_rectangle_l194_194263


namespace nineteenth_position_9357_l194_194548

theorem nineteenth_position_9357 :
  let digits := [3, 5, 7, 9]
  let permutations := list.permutations digits
  let sorted_permutations := list.sort (λ x y, (nat.digit x[0]*1000 + nat.digit x[1]*100 + nat.digit x[2]*10 + nat.digit x[3]) ≤ (nat.digit y[0]*1000 + nat.digit y[1]*100 + nat.digit y[2]*10 + nat.digit y[3])) permutations
  ((sorted_permutations.nth 18).getD [0,0,0,0] = [9, 3, 5, 7]) :=
by
  sorry

end nineteenth_position_9357_l194_194548


namespace time_to_reach_6400ft_is_200min_l194_194769

noncomputable def time_to_reach_ship (depth : ℕ) (rate : ℕ) : ℕ :=
  depth / rate

theorem time_to_reach_6400ft_is_200min :
  time_to_reach_ship 6400 32 = 200 := by
  sorry

end time_to_reach_6400ft_is_200min_l194_194769


namespace value_of_m_l194_194917

theorem value_of_m (m : ℝ) (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 :=
by 
suffices : m - 5 = 0, from sorry -- This is where the proof would go

end value_of_m_l194_194917


namespace value_of_S5_l194_194130

-- Define the arithmetic sequence and conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a n > 0

noncomputable def common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ a 5 / a 3 = q ^ 2 ∧ a 2 * a 6 = 64

def sum_a3_a5 (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 = 20

noncomputable def sum_of_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = Σ i in (Finset.range n), a i

-- Prove the value of S_5
theorem value_of_S5 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : arithmetic_seq a) (h2 : common_ratio a q) (h3 : sum_a3_a5 a) (h4 : sum_of_first_n a S) :
  S 5 = 31 :=
by
  sorry

end value_of_S5_l194_194130


namespace prime_solution_l194_194164

theorem prime_solution (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 :=
sorry

end prime_solution_l194_194164


namespace correct_observation_l194_194340

-- Given constants and conditions
constant n : Nat := 50
constant original_mean : ℝ := 36
constant new_mean : ℝ := 36.5
constant wrong_observation : ℝ := 23

-- Summarizing derived total sums
constant original_total_sum : ℝ := n * original_mean
constant new_total_sum : ℝ := n * new_mean

-- The statement we want to prove
theorem correct_observation : 
  ∃ x : ℝ, original_total_sum - wrong_observation + x = new_total_sum := 
sorry

end correct_observation_l194_194340


namespace num_four_digit_square_palindromes_l194_194189

open Nat

-- Define what it means to be a 4-digit number
def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n <= 9999

-- Define what it means to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- The main theorem stating that there are exactly 2 four-digit squares that are palindromes
theorem num_four_digit_square_palindromes : 
  { n : ℕ | is_four_digit n ∧ is_palindrome n ∧ ∃ k : ℕ, k^2 = n ∧ k >= 32 ∧ k <= 99 }.to_finset.card = 2 :=
sorry

end num_four_digit_square_palindromes_l194_194189


namespace f_zero_f_expression_A_intersect_not_B_l194_194157

noncomputable def f (x : ℝ) := x^2 + x - 2

-- Given conditions
axiom f_condition : ∀ x y : ℝ, f (x + y) - f y = x * (x + 2 * y + 1)
axiom f_one : f 1 = 0

-- Prove the required values
theorem f_zero : f 0 = -2 :=
sorry

theorem f_expression : ∀ x : ℝ, f x = x^2 + x - 2 :=
sorry

-- Define properties P and Q
def A := { a : ℝ | ∀ x : ℝ, 0 < x ∧ x < 1/2 → f x + 3 < 2 * x + a }
def B := { a : ℝ | ∀ x ∈ Icc (-2 : ℝ) 2, ∀ y ∈ Icc (-2 : ℝ) 2, monotone (fun x => x^2 + x - 2 - a * x) }

-- Required set operation
theorem A_intersect_not_B : {a : ℝ | 1 ≤ a ∧ a < 5} = A ∩ (Bᶜ) :=
sorry

end f_zero_f_expression_A_intersect_not_B_l194_194157


namespace find_a_plus_b_l194_194147

-- Given points A and B, where A(1, a) and B(b, -2) are symmetric with respect to the origin.
variables (a b : ℤ)

-- Definition for symmetry conditions
def symmetric_wrt_origin (x1 y1 x2 y2 : ℤ) :=
  x2 = -x1 ∧ y2 = -y1

-- The main theorem
theorem find_a_plus_b :
  symmetric_wrt_origin 1 a b (-2) → a + b = 1 :=
by
  sorry

end find_a_plus_b_l194_194147


namespace total_seeds_planted_l194_194301

def number_of_flowerbeds : ℕ := 9
def seeds_per_flowerbed : ℕ := 5

theorem total_seeds_planted : number_of_flowerbeds * seeds_per_flowerbed = 45 :=
by
  sorry

end total_seeds_planted_l194_194301


namespace hexagon_opposite_sides_equal_l194_194592

theorem hexagon_opposite_sides_equal (a b c d e f : ℝ) 
  (hexagon_angles_eq : ∀ i, 1 ≤ i ∧ i ≤ 6 → ∠(PolygonInternalAngle i) = 120) :
  a - e = b - f ∧ b - f = c - d :=
sorry

end hexagon_opposite_sides_equal_l194_194592


namespace sequence_term_general_formula_l194_194520

theorem sequence_term_general_formula (S : ℕ → ℚ) (a : ℕ → ℚ) :
  (∀ n, S n = n^2 + (1/2)*n + 5) →
  (∀ n, (n ≥ 2) → a n = S n - S (n - 1)) →
  a 1 = 13/2 →
  (∀ n, a n = if n = 1 then 13/2 else 2*n - 1/2) :=
by
  intros hS ha h1
  sorry

end sequence_term_general_formula_l194_194520


namespace find_angle_A_find_sinB_sinC_l194_194854

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = sqrt (b^2 + c^2 - 2 * b * c * cos A) ∧
  sin A = sin (A) ∧
  sin B = sin (B) ∧
  sin C = sin (C) ∧
  3 * cos B * cos C + 2 = 3 * sin B * sin C + 2 * cos (2 * A)

theorem find_angle_A (a b c A B C : ℝ) (h1 : b = 5) (h2 : 3 * cos B * cos C + 2 = 3 * sin B * sin C + 2 * cos (2 * A)) (h3 : S = 5 * sqrt (3)) 
  : A = π / 3 :=
begin
  sorry
end

theorem find_sinB_sinC (a b c A B C : ℝ) (h1 : b = 5) (h2 : 3 * cos B * cos C + 2 = 3 * sin B * sin C + 2 * cos (2 * A)) (h3 : S = 5 * sqrt (3)) (h4 : A = π / 3)
  : sin B * sin C = 5 / 7 :=
begin
  sorry
end

end find_angle_A_find_sinB_sinC_l194_194854


namespace sum_of_areas_of_alternately_colored_triangles_l194_194456

/-
Given a regular decagon and an interior point P, 
prove that the sum of the areas of the blue triangles 
formed by connecting P to the vertices 
of the decagon is equal to the sum of the areas of the red triangles.
-/

open EuclideanGeometry

theorem sum_of_areas_of_alternately_colored_triangles (P : Point) :
  ∀ (decagon : Polygon 10), 
  let triangles := (form_triangles decagon P),
  let red_triangles := alternately_colored triangles Red,
  let blue_triangles := alternately_colored triangles Blue 
  in sum_of_areas red_triangles = sum_of_areas blue_triangles :=
begin
  sorry
end

end sum_of_areas_of_alternately_colored_triangles_l194_194456


namespace num_both_books_500_l194_194345

-- Definitions based on conditions
def num_purchase_book_A (n_B : ℕ) : ℕ := 2 * n_B
def num_purchase_both_books (n_only_B : ℕ) : ℕ := 2 * n_only_B
def num_only_A : ℕ := 1000

-- Definition to assert the verification
def verify_num_purchase_both_books (num_B num_only_B : ℕ) : Prop :=
  num_purchase_book_A(num_B) = num_only_A + num_purchase_both_books(num_only_B) →
  num_only_B = 250 →

  -- Prove that the number of people who purchased both books is 500.
  num_purchase_both_books(num_only_B) = 500

-- The statement to prove
theorem num_both_books_500 (num_B num_only_B : ℕ) :
  verify_num_purchase_both_books num_B num_only_B :=
by 
  sorry

end num_both_books_500_l194_194345


namespace taehyung_candies_l194_194702

theorem taehyung_candies (total_candies seokjin_eats taehyung_eats : ℕ) (h1 : total_candies = 6) (h2 : seokjin_eats = 4) : taehyung_eats = total_candies - seokjin_eats := 
by
  rw [h1, h2]
  exact rfl

end taehyung_candies_l194_194702


namespace solve_quadratic_inequality_l194_194353

theorem solve_quadratic_inequality (a b c x : ℝ) (h : a ≠ 0) :
  let Δ := b^2 - 4 * a * c
  let x1 := (-b - real.sqrt Δ) / (2 * a)
  let x2 := (-b + real.sqrt Δ) / (2 * a)
  (if a > 0 then x ∈ set.Ioo x1 x2 else x ∈ set.Iio x1 ∪ set.Ioi x2) ↔ a * x^2 + b * x + c < 0 :=
by sorry

end solve_quadratic_inequality_l194_194353


namespace trig_identity_cos_sin_l194_194496

theorem trig_identity_cos_sin : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
sorry

end trig_identity_cos_sin_l194_194496


namespace eccentricity_of_ellipse_l194_194583

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem eccentricity_of_ellipse :
  let P := (2, 3)
  let F1 := (-2, 0)
  let F2 := (2, 0)
  let d1 := distance P F1
  let d2 := distance P F2
  let a := (d1 + d2) / 2
  let c := distance F1 F2 / 2
  let e := c / a
  e = 1 / 2 := 
by 
  sorry

end eccentricity_of_ellipse_l194_194583


namespace EH_length_l194_194264

structure Rectangle :=
(AB BC CD DA : ℝ)
(horiz: AB=CD)
(verti: BC=DA)
(diag_eq: (AB^2 + BC^2) = (CD^2 + DA^2))

structure Point :=
(x y : ℝ)

noncomputable def H_distance (E D : Point)
    (AB BC : ℝ) : ℝ :=
    (E.y - D.y) -- if we consider D at origin (0,0)

theorem EH_length
    (AB BC : ℝ)
    (H_dist : ℝ)
    (E : Point)
    (rectangle : Rectangle) :
    AB = 50 →
    BC = 60 →
    E.x^2 + BC^2 = 30^2 + 60^2 →
    E.y = 40 →
    H_dist = E.y - CD →
    H_dist = 7.08 :=
by
    sorry

end EH_length_l194_194264


namespace S_infinite_least_in_S_limit_ratio_of_S_l194_194804

noncomputable def f : ℕ → ℕ
| 1 => 1
| (n + 1) => if f n > n + 1 then f n - (n + 1) else f n + (n + 1)

def S : Set ℕ := { n | f n = 1993 }

theorem S_infinite : Infinite S :=
sorry

theorem least_in_S : ∃ n, n ∈ S ∧ ∀ m, m ∈ S → n ≤ m :=
begin
  use 12417,
  split,
  sorry,
  intros m hm,
  sorry
end

theorem limit_ratio_of_S : tendsto (λ i, (nth (filter (λ n, f n = 1993) finset.univ) (i + 1) / nth (filter (λ n, f n = 1993) finset.univ) i)) at_top (𝓝 3) :=
sorry

end S_infinite_least_in_S_limit_ratio_of_S_l194_194804


namespace find_median_room_number_l194_194787

def median_room_number (rooms : List ℕ) : ℕ :=
  rooms.sorted[(rooms.length / 2) := by linarith [rooms.length]]

theorem find_median_room_number :
  let rooms := List.range 21 |>.map (fun x => x + 1) -- Room numbers 1 through 21
  let remaining_rooms := rooms.erase 12 |>.erase 13 -- Excluding rooms 12 and 13
  median_room_number remaining_rooms = 10 :=
by
  let rooms := List.range 21 |>.map (fun x => x + 1)
  let remaining_rooms := rooms.erase 12 |>.erase 13
  have h : remaining_rooms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21] := by sorry
  rw [h]
  exact rfl

end find_median_room_number_l194_194787


namespace average_people_added_each_year_l194_194440

-- a) Identifying questions and conditions
-- Question: What is the average number of people added each year?
-- Conditions: In 2000, about 450,000 people lived in Maryville. In 2005, about 467,000 people lived in Maryville.

-- c) Mathematically equivalent proof problem
-- Mathematically equivalent proof problem: Prove that the average number of people added each year is 3400 given the conditions.

-- d) Lean 4 statement
theorem average_people_added_each_year :
  let population_2000 := 450000
  let population_2005 := 467000
  let years_passed := 2005 - 2000
  let total_increase := population_2005 - population_2000
  total_increase / years_passed = 3400 := by
    sorry

end average_people_added_each_year_l194_194440


namespace k_ge_1_l194_194362

theorem k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 2 * x + 9 > 6 * x + 1 ∧ x - k < 1 → x < 2) → k ≥ 1 :=
by 
  sorry

end k_ge_1_l194_194362


namespace probability_subinterval_l194_194911

theorem probability_subinterval (a : ℝ) (h₀ : 0 ≤ a ∧ a ≤ 1) : 
  (∃ p : ℝ, p = (1 - 2/3) / (1 - 0) ∧ p = 1/3) := 
begin
  use (1 - 2/3) / (1 - 0),
  split,
  { simp, },
  { simp, },
end

end probability_subinterval_l194_194911


namespace three_pow_gt_pow_three_for_n_ne_3_l194_194658

theorem three_pow_gt_pow_three_for_n_ne_3 (n : ℕ) (h : n ≠ 3) : 3^n > n^3 :=
sorry

end three_pow_gt_pow_three_for_n_ne_3_l194_194658


namespace find_b_squared_l194_194756

noncomputable def complex_f (a b : ℝ) (z : ℂ) : ℂ := (a + complex.I * b) * z

theorem find_b_squared :
  ∀ (a b : ℝ), 
  (∀ z : ℂ, abs (complex_f a b z - z) = abs (complex_f a b z - 3 * z)) →
  abs (a + complex.I * b) = 5 →
  b^2 = 21 :=
by
  intros a b h_eqdist h_magnitude
  sorry

end find_b_squared_l194_194756


namespace sequence_sum_a_b_l194_194521

theorem sequence_sum_a_b (a b : ℕ) (a_seq : ℕ → ℕ) 
  (h1 : a_seq 1 = a)
  (h2 : a_seq 2 = b)
  (h3 : ∀ n ≥ 1, a_seq (n+2) = (a_seq n + 2018) / (a_seq (n+1) + 1)) :
  a + b = 1011 ∨ a + b = 2019 :=
sorry

end sequence_sum_a_b_l194_194521


namespace volume_of_pyramid_l194_194059

-- Define the properties of the pyramid P-ABC
variables (A B C P : Type) [EuclideanGeometry A B C P]
variable (theta : Real)
variable (AB : Real)
variable (h_base : Real)
variable (h_total : Real)
variable (V : Real)

-- Given conditions
axiom base_equilateral : EquilateralTriangle A B C
axiom P_vertex_equidistant : Equidistant P [A, B, C]
axiom AB_eq_two : AB = 2
axiom angle_APB : ∠(A, P, B) = theta

-- Calculate h_base using given AB
definition calculate_base_height : Real := by
  sorry

-- Define centroid G and its properties
variable (G : Type)
axiom centroid_G : G = centroid A B C
-- Calculate distances involving centroid
definition distance_AG : Real := by 
  sorry
definition position_P_relative_to_G : Real := by 
  sorry

-- Define height PG in terms of theta
definition height_PG_in_terms_of_theta : Real := 1 / tan (theta / 2)

-- Total height from base to P
definition total_height : Real := distance_AG + height_PG_in_terms_of_theta

-- Calculate the volume of the pyramid
theorem volume_of_pyramid :
  V = (sqrt 3 / 9) * ((2 / 3) * sqrt 3 + 1 / tan (theta / 2)) := by
  sorry

end volume_of_pyramid_l194_194059


namespace find_BC_l194_194670

-- Given the parameters of the trapezoid
variables (AB CD h area BC : ℝ)

-- Conditions
def conditions : Prop :=
  AB = 13 ∧ CD = 20 ∧ h = 8 ∧ area = 208

-- The proof statement we aim to prove
theorem find_BC (h : conditions) : BC = 11.71 := 
sorry

end find_BC_l194_194670


namespace sum_of_integer_solutions_in_inequality_l194_194402

theorem sum_of_integer_solutions_in_inequality :
  (∀ x : ℝ,
    5 * x - 11 >= 0 ∧ 5 * x^2 - 21 * x + 21 >= 0 →
    sqrt (5 * x - 11) - sqrt (5 * x^2 - 21 * x + 21) >=
    5 * x^2 - 26 * x + 32) →
  ∑ x in {x : ℤ | 5 * (x : ℝ) - 11 >= 0 ∧ 
              5 * (x : ℝ)^2 - 21 * (x : ℝ) + 21 >= 0 ∧ 
              sqrt (5 * (x : ℝ) - 11) - sqrt (5 * (x : ℝ)^2 - 21 * (x : ℝ) + 21) >= 5 * (x : ℝ)^2 - 26 * (x : ℝ) + 32}, x =
  3 :=
  sorry

end sum_of_integer_solutions_in_inequality_l194_194402


namespace rhombus_area_l194_194818

-- Define the side length of the rhombus
def side_length : ℝ := 13

-- Define the length of one diagonal of the rhombus
def d1 : ℝ := 24

-- Define the other diagonal using Pythagorean theorem
noncomputable def d2 := 2 * Real.sqrt ((side_length ^ 2) - (d1 / 2) ^ 2)

-- Statement to prove the area of the rhombus
theorem rhombus_area : ∀ (a d1: ℝ), a = side_length → d1 = 24 → let d2 := 2 * Real.sqrt((a ^ 2) - (d1 / 2) ^ 2) in (d1 * d2) / 2 = 120 :=
by
  intros a d1 h1 h2
  rw [h1, h2]
  let d2 := 2 * Real.sqrt((side_length ^ 2) - (d1 / 2)^2)
  exact sorry

end rhombus_area_l194_194818


namespace ab_value_l194_194155

def f (x : ℝ) := 2^(x - 1)

def inverse_f (x : ℝ) := Real.log x / Real.log 2 + 1 -- Using natural logarithm for flexibility

variable {a b : ℝ}

theorem ab_value (h : inverse_f a + inverse_f b = 4) : a * b = 4 := by
  sorry

end ab_value_l194_194155


namespace initial_volume_is_72_l194_194008

noncomputable def initial_volume (V : ℝ) : Prop :=
  let salt_initial : ℝ := 0.10 * V
  let total_volume_new : ℝ := V + 18
  let salt_percentage_new : ℝ := 0.08 * total_volume_new
  salt_initial = salt_percentage_new

theorem initial_volume_is_72 :
  ∃ V : ℝ, initial_volume V ∧ V = 72 :=
by
  sorry

end initial_volume_is_72_l194_194008


namespace find_m_l194_194888

open Set

def A (m: ℝ) := {x : ℝ | x^2 - m * x + m^2 - 19 = 0}

def B := {x : ℝ | x^2 - 5 * x + 6 = 0}

def C := ({2, -4} : Set ℝ)

theorem find_m (m : ℝ) (ha : A m ∩ B ≠ ∅) (hb : A m ∩ C = ∅) : m = -2 :=
  sorry

end find_m_l194_194888


namespace complex_division_example_l194_194738

open Complex

theorem complex_division_example :
  (i / (Complex.sqrt 7 + 3 * i)) = (3 / 16 + Complex.sqrt 7 / 16 * i) :=
by sorry

end complex_division_example_l194_194738


namespace work_completion_days_l194_194735

theorem work_completion_days (john_days : ℕ) (rose_days : ℕ) (combined_days : ℕ) : 
  john_days = 10 → rose_days = 40 → combined_days = 8 → 
  (1 / john_days + 1 / rose_days = 1 / combined_days) := 
by
  intros hj hr hc
  rw [hj, hr, hc]
  sorry

end work_completion_days_l194_194735


namespace one_shooter_hits_l194_194914

/-- Probability calculations for three shooters --/
def prob_three_shooters (pA pB pC : ℚ) (¬pA ¬pB ¬pC : ℚ) : ℚ :=
  pA * ¬pB * ¬pC + ¬pA * pB * ¬pC + ¬pA * ¬pB * pC

theorem one_shooter_hits (hA : 8 / 10 = 4 / 5) (hB : 6 / 10 = 3 / 5) (hC : 7 / 10 = 7 / 10) :
  prob_three_shooters (4 / 5) (3 / 5) (7 / 10) (1 - 4 / 5) (1 - 3 / 5) (1 - 7 / 10) = 47 / 250 :=
by
  sorry

end one_shooter_hits_l194_194914


namespace green_ball_probability_l194_194064

def prob_green_ball : ℚ :=
  let prob_container := (1 : ℚ) / 3
  let prob_green_I := (4 : ℚ) / 12
  let prob_green_II := (5 : ℚ) / 8
  let prob_green_III := (4 : ℚ) / 8
  prob_container * prob_green_I + prob_container * prob_green_II + prob_container * prob_green_III

theorem green_ball_probability :
  prob_green_ball = 35 / 72 :=
by
  -- Proof steps are omitted as "sorry" is used to skip the proof.
  sorry

end green_ball_probability_l194_194064


namespace problem1_problem2_problem3_problem4_l194_194050

-- statement for problem 1
theorem problem1 : -5 + 8 - 2 = 1 := by
  sorry

-- statement for problem 2
theorem problem2 : (-3) * (5/6) / (-1/4) = 10 := by
  sorry

-- statement for problem 3
theorem problem3 : -3/17 + (-3.75) + (-14/17) + (15/4) = -1 := by
  sorry

-- statement for problem 4
theorem problem4 : -(1^10) - ((13/14) - (11/12)) * (4 - (-2)^2) + (1/2) / 3 = -(5/6) := by
  sorry

end problem1_problem2_problem3_problem4_l194_194050


namespace triangle_proportion_l194_194979

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def angle_bisector (A B C : Point) : Line := sorry
noncomputable def perpendicular (l1 l2 : Line) : Prop := sorry
noncomputable def on_line (p : Point) (l : Line) : Prop := sorry

theorem triangle_proportion
  (A B C D M : Point) 
  (h₁ : midpoint A B = M)
  (h₂ : on_line D (angle_bisector B A C))
  (h₃ : perpendicular (line_through M D) (angle_bisector B A C)) :
  distance A B = 3 * distance B C := 
sorry

end triangle_proportion_l194_194979


namespace median_in_interval_65_69_l194_194800

-- Definitions for student counts in each interval
def count_50_54 := 5
def count_55_59 := 7
def count_60_64 := 22
def count_65_69 := 19
def count_70_74 := 15
def count_75_79 := 10
def count_80_84 := 18
def count_85_89 := 5

-- Total number of students
def total_students := 101

-- Calculation of the position of the median
def median_position := (total_students + 1) / 2

-- Cumulative counts
def cumulative_up_to_59 := count_50_54 + count_55_59
def cumulative_up_to_64 := cumulative_up_to_59 + count_60_64
def cumulative_up_to_69 := cumulative_up_to_64 + count_65_69

-- Proof statement
theorem median_in_interval_65_69 :
  34 < median_position ∧ median_position ≤ cumulative_up_to_69 :=
by
  sorry

end median_in_interval_65_69_l194_194800


namespace factorized_expression_l194_194081

variable {a b c : ℝ}

theorem factorized_expression :
  ( ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) / 
    ((a - b)^3 + (b - c)^3 + (c - a)^3) ) 
  = (a + b) * (a + c) * (b + c) := 
  sorry

end factorized_expression_l194_194081


namespace need_to_buy_more_dice_l194_194645

theorem need_to_buy_more_dice (mark_dice : ℕ) (mark_percent_12_sided : ℕ) (james_dice : ℕ) (james_percent_12_sided : ℕ) (total_needed_12_sided : ℕ) :
  mark_dice = 10 → mark_percent_12_sided = 60 →
  james_dice = 8 → james_percent_12_sided = 75 →
  total_needed_12_sided = 14 →
  (total_needed_12_sided - (mark_dice * mark_percent_12_sided / 100 + james_dice * james_percent_12_sided / 100)) = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end need_to_buy_more_dice_l194_194645


namespace smallest_N_for_same_length_diagonals_in_2017gon_is_1008_l194_194938

theorem smallest_N_for_same_length_diagonals_in_2017gon_is_1008 :
  ∀ (N : ℕ), (N > 1007) → ∃ (d1 d2 : (fin 2017) × (fin 2017)), 
               d1 ≠ d2 ∧ length_of_diagonal d1 = length_of_diagonal d2 :=
sorry

/-- Helper function to calculate the length of a diagonal in a regular polygon -/
noncomputable def length_of_diagonal (d : (fin 2017) × (fin 2017)) : ℝ :=
  -- Since actual diagonal length calculation involves some trigonometry on fin 2017, we leave it undefined here.
  sorry

end smallest_N_for_same_length_diagonals_in_2017gon_is_1008_l194_194938


namespace intersection_M_N_l194_194561

noncomputable def M : Set ℝ := { x | x^2 - x ≤ 0 }
noncomputable def N : Set ℝ := { x | 1 - abs x > 0 }
noncomputable def intersection : Set ℝ := { x | x ≥ 0 ∧ x < 1 }

theorem intersection_M_N : M ∩ N = intersection :=
by
  sorry

end intersection_M_N_l194_194561


namespace negation_correct_not_necessary_and_sufficient_equation_not_always_curve_binary_representation_multiple_choice_problem_l194_194393

-- Define the problem in terms of Lean statements.
theorem negation_correct : (∀ (x : ℝ), ¬(sin (x + cos x) = sqrt 3)) ↔ ¬(∃ (x : ℝ), sin (x + cos x) = sqrt 3) :=
sorry

theorem not_necessary_and_sufficient : ¬(∀ (a b : ℝ), (a ≠ 1 ∨ b ≠ 2) ↔ (a + b ≠ 3)) :=
sorry

theorem equation_not_always_curve : ¬(∀ (f : ℝ → ℝ → ℝ) (C : set (ℝ × ℝ)), (∀ p ∈ C, f p.1 p.2 = 0) → (∀ (x y : ℝ), f x y = 0 → (x, y) ∈ C)) :=
sorry

theorem binary_representation : nat.bin_of (nat.of_digits 2 [1,0,0,0,0,1,0]) = 66 :=
sorry

-- The main theorem combining the results
theorem multiple_choice_problem : (negation_correct ∧ binary_representation) ∧ 
                                  (not not_necessary_and_sufficient) ∧ 
                                  (not equation_not_always_curve) → 
                                   true :=
sorry

end negation_correct_not_necessary_and_sufficient_equation_not_always_curve_binary_representation_multiple_choice_problem_l194_194393


namespace each_player_gets_seven_l194_194963

-- Define the total number of dominoes and players
def total_dominoes : Nat := 28
def total_players : Nat := 4

-- Define the question for how many dominoes each player would receive
def dominoes_per_player (dominoes players : Nat) : Nat := dominoes / players

-- The theorem to prove each player gets 7 dominoes
theorem each_player_gets_seven : dominoes_per_player total_dominoes total_players = 7 :=
by
  sorry

end each_player_gets_seven_l194_194963


namespace num_four_digit_palindromic_squares_is_two_l194_194196

open Nat

-- Define the condition for a palindrome
def is_palindrome (n : ℕ) : Prop :=
  to_digits 10 n = (to_digits 10 n).reverse

-- Define the range of numbers to check
def range_32_to_99 := {x : ℕ | 32 ≤ x ∧ x ≤ 99}

-- Define the function to compute the square of a number
def square (n : ℕ) : ℕ := n * n

-- Define the set of 4-digit squares that are palindromes
def four_digit_palindromic_squares : Finset ℕ :=
  (Finset.filter (λ n => is_palindrome n) (Finset.image square (Finset.filter (λ n => 1000 ≤ square n ∧ square n < 10000) 
  (Finset.filter (λ n => n ∈ range_32_to_99) (Finset.range 100)))))

-- The main theorem stating the number of 4-digit palindromic squares
theorem num_four_digit_palindromic_squares_is_two :
  four_digit_palindromic_squares.card = 2 := sorry

end num_four_digit_palindromic_squares_is_two_l194_194196


namespace trig_function_identity_l194_194580

theorem trig_function_identity :
  (∀ x, f (Real.cos x) = Real.sin (3 * x)) →
  f (Real.sin (Real.pi / 6)) = 0 :=
by
  intros h
  have h1 : Real.sin (Real.pi / 6) = Real.cos (Real.pi / 3) := by sorry
  rw h1
  sorry

end trig_function_identity_l194_194580


namespace smallest_n_for_geometric_sequence_divisible_by_power_of_two_l194_194285

theorem smallest_n_for_geometric_sequence_divisible_by_power_of_two (a r : ℚ) 
  (h₀ : a = 5 / 3) (h₁ : a * r = 10) : 
  ∃ n : ℕ, (n ≥ 1) ∧ (r^(n - 1) * a) ≠ 0 ∧ (∀ m < n, ¬ (256 ∣ a * r^(m - 1))) ∧ (256 ∣ a * r^(n - 1)) :=
begin
  sorry
end

end smallest_n_for_geometric_sequence_divisible_by_power_of_two_l194_194285


namespace positive_integral_solution_exists_l194_194485

theorem positive_integral_solution_exists :
  ∃ n : ℕ, n > 0 ∧
  ( (n * (n + 1) * (2 * n + 1)) * 100 = 27 * 6 * (n * (n + 1))^2 ) ∧ n = 5 :=
by {
  sorry
}

end positive_integral_solution_exists_l194_194485


namespace arrow_estimate_closest_to_9_l194_194327

theorem arrow_estimate_closest_to_9 
  (a b : ℝ) (h₁ : a = 8.75) (h₂ : b = 9.0)
  (h : 8.75 < 9.0) :
  ∃ x ∈ Set.Icc a b, x = 9.0 :=
by
  sorry

end arrow_estimate_closest_to_9_l194_194327


namespace total_price_is_26_l194_194378

-- Define the prices of the items
def price_tea : ℝ := 10
def price_cheese : ℝ := price_tea / 2
def price_butter : ℝ := 0.8 * price_cheese
def price_bread : ℝ := price_butter / 2
def price_eggs : ℝ := price_bread / 2
def price_honey : ℝ := price_eggs + 3

-- The total price of items
def total_price : ℝ :=
  price_butter + price_bread + price_cheese + price_tea + price_eggs + price_honey

-- The theorem to prove
theorem total_price_is_26 :
  total_price = 26 := by
  sorry

end total_price_is_26_l194_194378


namespace initial_mean_calculated_l194_194685

theorem initial_mean_calculated (M : ℝ) (h1 : 25 * M - 35 = 25 * 191.4 - 35) : M = 191.4 := 
  sorry

end initial_mean_calculated_l194_194685


namespace arithmetic_sequence_general_formula_and_sum_l194_194240

/-- Arithmetic sequence problem -/
theorem arithmetic_sequence_general_formula_and_sum 
  (a: ℕ → ℕ)
  (h1: a 2 = 4) 
  (h2: a 4 + a 7 = 15) 
  (b: ℕ → ℕ)
  (h_b: ∀ n, b n = 2 ^ (a n - 2)) : 
  (∀ n, a n = n + 2) ∧ (b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 + b 10 = 2046) :=
by 
  have h3 : a 1 + (4 - 1) * (a 2 - a 1) = 4 := by rw [h1]
  have h4 : (a 1 + 3 * (a 2 - a 1)) + (a 1 + 6 * (a 2 - a 1)) = 15 := by rw [h1, h2]
  sorry

end arithmetic_sequence_general_formula_and_sum_l194_194240


namespace inequality_solution_l194_194404

theorem inequality_solution :
  let f (x : ℝ) := sqrt (5 * x - 11) - sqrt (5 * x^2 - 21 * x + 21)
  let g (x : ℝ) := 5 * x^2 - 26 * x + 32
  let valid x := 5 * x - 11 ≥ 0 ∧ 5 * x^2 - 21 * x + 21 ≥ 0 ∧ f x ≥ g x
  (∑ k in (Finset.filter valid (Finset.Icc ⌈2.939⌉ ⌊2.939⌋)), k) = 3 :=
by
  sorry

end inequality_solution_l194_194404


namespace vector_parallel_l194_194172

noncomputable def a := (1, 2)
noncomputable def b (x : ℝ) := (x, 1)
noncomputable def c (x : ℝ) : ℝ × ℝ := (1 + 2 * x, 4)
noncomputable def d (x : ℝ) : ℝ × ℝ := (2 - x, 3)

theorem vector_parallel (x : ℝ) : 
  (c x).1 * (d x).2 - (c x).2 * (d x).1 = 0 ↔ x = 1 / 2 :=
begin
  sorry
end

end vector_parallel_l194_194172


namespace pieces_present_l194_194454

def total_pieces : ℕ := 32
def missing_pieces : ℕ := 10

theorem pieces_present : total_pieces - missing_pieces = 22 :=
by {
  sorry
}

end pieces_present_l194_194454


namespace intersection_complement_l194_194532

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def complement_U (A : Set ℝ) : Set ℝ := {x | ¬ (A x)}

theorem intersection_complement (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  B ∩ (complement_U A) = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l194_194532


namespace range_of_g_l194_194825

-- Define the function g(x)
def g (x : ℝ) : ℝ := if x ≠ -5 then 3 * (x - 4) else 0 -- The value at x = -5 is irrelevant as g is undefined there

-- The main theorem
theorem range_of_g : set.range g = set.univ \ {-27} :=
by
  sorry -- proof omitted

end range_of_g_l194_194825


namespace a7_a8_a9_sum_l194_194846

variable (a_n : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (d : ℕ)
variable (a1 : ℕ)

-- Define the arithmetic sequence terms and sums
axiom S_def : ∀ n, S n = n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
axiom S3_eq_9 : S 3 = 9
axiom S5_eq_30 : S 5 = 30

-- Define what we need to prove
theorem a7_a8_a9_sum : a1 = 0 → d = 3 → (a_n 7 + a_n 8 + a_n 9 = 63) :=
by
  intro h1 h2
  have hS9 : S 9 = 108 := by
    -- Proof omitted (sorry used here as placeholder)
    sorry
  have hS6 : S 6 = 45 := by
    -- Proof omitted (sorry used here as placeholder)
    sorry
  rw [hS9, hS6]
  have ha7a8a9 : a_n 7 + a_n 8 + a_n 9 = S 9 - S 6 := by
    -- Proof omitted (sorry used here as placeholder)
    sorry
  rw [ha7a8a9]
  exact rfl

end a7_a8_a9_sum_l194_194846


namespace regular_polygon_sides_l194_194941

-- Define the main theorem statement
theorem regular_polygon_sides (n : ℕ) : 
  (n > 2) ∧ 
  ((n - 2) * 180 / n - 360 / n = 90) → 
  n = 8 := by
  sorry

end regular_polygon_sides_l194_194941


namespace englishman_land_earnings_l194_194035

noncomputable def acres_to_square_yards (acres : ℝ) : ℝ := acres * 4840
noncomputable def square_yards_to_square_meters (sq_yards : ℝ) : ℝ := sq_yards * (0.9144 ^ 2)
noncomputable def square_meters_to_hectares (sq_meters : ℝ) : ℝ := sq_meters / 10000
noncomputable def cost_of_land (hectares : ℝ) (price_per_hectare : ℝ) : ℝ := hectares * price_per_hectare

theorem englishman_land_earnings
  (acres_owned : ℝ)
  (price_per_hectare : ℝ)
  (acre_to_yard : ℝ)
  (yard_to_meter : ℝ)
  (hectare_to_meter : ℝ)
  (h1 : acres_owned = 2)
  (h2 : price_per_hectare = 500000)
  (h3 : acre_to_yard = 4840)
  (h4 : yard_to_meter = 0.9144)
  (h5 : hectare_to_meter = 10000)
  : cost_of_land (square_meters_to_hectares (square_yards_to_square_meters (acres_to_square_yards acres_owned))) price_per_hectare = 404685.6 := sorry

end englishman_land_earnings_l194_194035


namespace vectors_coplanar_l194_194449

theorem vectors_coplanar :
  let a := ![4, -1, -6]
  let b := ![1, -3, -7]
  let c := ![2, -1, -4]
  in (a ⬝ (b × c)) = 0 :=
by
  -- Define the vectors a, b, and c
  let a := ![4, -1, -6]
  let b := ![1, -3, -7]
  let c := ![2, -1, -4]
  -- Calculate the dot product of a with the cross product of b and c
  have habc : (a ⬝ (b × c)) = 0 :=
    by sorry
  -- Conclude that the vectors are coplanar because the scalar triple product is zero
  exact habc

end vectors_coplanar_l194_194449


namespace sum_series_denominator_lcm_l194_194066

-- Define the double factorials
def even_double_factorial (n : ℕ) : ℕ := 2^n * n.factorial
def odd_double_factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 
  else (2*n + 1) * odd_double_factorial (n - 1)

-- Define the sum to be proven 
def sum_series (n : ℕ) : ℕ := 
  (Finset.sum (Finset.range (n + 1)) (λ i, 1 / (2 * i + 1)))

theorem sum_series_denominator_lcm :
  let S := ∑ i in Finset.range 1005, 1 / (2 * i + 1)
  let d := lcm' (Finset.range 1 2009) 
  let c := 0
  in 
    d.mod 2 = 1 ∧ c = 0 ∧ c * d / 10 = 0 := 
  by {
    -- Sorry is a placeholder for the proof.
    sorry
  }

end sum_series_denominator_lcm_l194_194066


namespace cos_2alpha_val_beta_val_l194_194565

variables {α β : ℝ}

-- Conditions for the first problem
def m (α : ℝ) : ℝ × ℝ := (Real.cos α, -1)
def n (α : ℝ) : ℝ × ℝ := (2, Real.sin α)
def perp (α : ℝ) := (m α).1 * (n α).1 + (m α).2 * (n α).2 = 0
def alpha_in_range := α ∈ set.Ioo 0 (Real.pi / 2)

-- Theorem for the value of cos 2α
theorem cos_2alpha_val (h1 : perp α) (h2 : alpha_in_range) : 2 * (Real.cos α)^2 - 1 = -3/5 :=
by sorry

-- Conditions for the second problem
def sin_alpha_minus_beta := Real.sin (α - β) = Real.sqrt 10 / 10
def beta_in_range := β ∈ set.Ioo 0 (Real.pi / 2)
def cos_2alpha := 2 * (Real.cos α)^2 - 1 = -3/5

-- Theorem for the value of β
theorem beta_val (h1 : cos_2alpha) (h2 : sin_alpha_minus_beta) (h3 : beta_in_range) : β = Real.pi / 4 :=
by sorry

end cos_2alpha_val_beta_val_l194_194565


namespace triple_divisor_sum_6_l194_194500

-- Summarize the definition of the divisor sum function excluding the number itself
def divisorSumExcluding (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ≠ n) (Finset.range (n + 1))).sum id

-- This is the main statement that we need to prove
theorem triple_divisor_sum_6 : divisorSumExcluding (divisorSumExcluding (divisorSumExcluding 6)) = 6 := 
by sorry

end triple_divisor_sum_6_l194_194500


namespace nontrivial_power_of_nat_l194_194501

theorem nontrivial_power_of_nat (n : ℕ) :
  (∃ A p : ℕ, 2^n + 1 = A^p ∧ p > 1) → n = 3 :=
by
  sorry

end nontrivial_power_of_nat_l194_194501


namespace root_quad_eqn_l194_194344

theorem root_quad_eqn (a : ℝ) (h : a^2 - a - 50 = 0) : a^3 - 51 * a = 50 :=
sorry

end root_quad_eqn_l194_194344


namespace marble_problem_l194_194761

theorem marble_problem
  (M : ℕ)
  (X : ℕ)
  (h1 : M = 18 * X)
  (h2 : M = 20 * (X - 1)) :
  M = 180 :=
by
  sorry

end marble_problem_l194_194761


namespace minimum_y_value_l194_194279

noncomputable def minimum_y (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x - 15) + abs (x - a - 15)

theorem minimum_y_value (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  minimum_y x a = 15 :=
by
  sorry

end minimum_y_value_l194_194279


namespace ranges_are_same_l194_194364

-- Given conditions
variable {α β : Type} (f : α → β) 

-- Defining the two functions
def g1 (x : α) : β := f x
def g2 (x : α) : β := f (x + 1)

-- Proof statement
theorem ranges_are_same (f : ℝ → ℝ) : 
  set.range (λ x, f x) = set.range (λ x, f (x + 1)) :=
sorry

end ranges_are_same_l194_194364


namespace ratio_nearest_integer_l194_194915

noncomputable def ratio_approx_eq (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a + b) / 2 = 2 * Real.sqrt (a * b)) : ℝ :=
  a / b

theorem ratio_nearest_integer (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a + b) / 2 = 2 * Real.sqrt (a * b)) :
  Int.nearest (ratio_approx_eq a b h1 h2 h3) = 14 :=
sorry

end ratio_nearest_integer_l194_194915


namespace find_number_l194_194502

theorem find_number (x : ℤ) (h : 42 + 3 * x - 10 = 65) : x = 11 := 
by 
  sorry 

end find_number_l194_194502


namespace crocodiles_count_l194_194235

theorem crocodiles_count (C : ℕ) (frogs : ℕ) (total_eyes : ℕ)
  (frog_eyes : ℕ) (croc_eyes : ℕ) (eye_eq : frogs * frog_eyes + C * croc_eyes = total_eyes) :
  F := 20 ∧ frog_eyes = 2 ∧ croc_eyes = 2 ∧ total_eyes = 60 → C = 10 :=
by
  assume h : frogs = 20 ∧ frog_eyes = 2 ∧ croc_eyes = 2 ∧ total_eyes = 60
  cases h with h₁ hrest
  cases hrest with h₂ hrest
  cases hrest with h₃ h₄
  rw [h₁, h₂, h₃, h₄, mul_comm, mul_comm] at eye_eq
  sorry

end crocodiles_count_l194_194235


namespace problem_1_problem_2_problem_3_l194_194510

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*a*x + 1

theorem problem_1 : (∀ x ∈ set.Icc (1:ℝ) 3, 0 ≤ g x 1 ∧ g x 1 ≤ 4) → (a = 1) :=
by
  intro h
  sorry

theorem problem_2 (k : ℝ) : 
  (∀ x ∈ set.Ici (1:ℝ), g (2^x) 1 - k * 4^x ≥ 0) → (k ≤ 1 / 4) :=
by
  intro h
  sorry

theorem problem_3 (k : ℝ) : 
  (∀ x ∈ set.univ, let t := |2^x - 1| in 
   y t k = 0 ∧ ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ t₁ ≠ y t k ∧ (0 < t₁ ∧ t₁ < 1) ∧ (t₂ > 1)) →
  (0 < k) :=
by
  intro h
  sorry

where 
  y (t : ℝ) (k : ℝ) : ℝ := (t^2 - (3*k + 2)*t + 1 + 2*k) - 3*k*t

end problem_1_problem_2_problem_3_l194_194510


namespace sufficiency_of_p_for_q_not_necessity_of_p_for_q_l194_194137

noncomputable def p (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m
noncomputable def q (m : ℝ) := ∀ x : ℝ, (- (5 - 2 * m)) ^ x < 0

theorem sufficiency_of_p_for_q : ∀ m : ℝ, (m < 1 → m < 2) :=
by sorry

theorem not_necessity_of_p_for_q : ∀ m : ℝ, ¬ (m < 2 → m < 1) :=
by sorry

end sufficiency_of_p_for_q_not_necessity_of_p_for_q_l194_194137


namespace car_mpg_city_l194_194750

theorem car_mpg_city
  (h c T : ℝ)
  (h1 : h * T = 480)
  (h2 : c * T = 336)
  (h3 : c = h - 6) :
  c = 14 :=
by
  sorry

end car_mpg_city_l194_194750


namespace find_k_l194_194557

def f (k x : ℝ) : ℝ := ((k+1) * x^2 + (k+3) * x + (2 * k - 8)) / ((2 * k - 1) * x^2 + (k + 1) * x + (k - 4))

theorem find_k (k : ℝ) (x : ℝ) (D : set ℝ) : 
  (∀ x ∈ D, f k x > 0) ↔ 
  k = 1 ∨ k > (15 + 16 * real.sqrt 2) / 7 ∨ k < (15 - 16 * real.sqrt 2) / 7 :=
sorry

end find_k_l194_194557


namespace transformed_mean_stddev_l194_194540

variables (n : ℕ) (x : Fin n → ℝ)

-- Given conditions
def mean_is_4 (mean : ℝ) : Prop :=
  mean = 4

def stddev_is_7 (stddev : ℝ) : Prop :=
  stddev = 7

-- Definitions for transformations and the results
def transformed_mean (mean : ℝ) : ℝ :=
  3 * mean + 2

def transformed_stddev (stddev : ℝ) : ℝ :=
  3 * stddev

-- The proof problem
theorem transformed_mean_stddev (mean stddev : ℝ) 
  (h_mean : mean_is_4 mean) 
  (h_stddev : stddev_is_7 stddev) :
  transformed_mean mean = 14 ∧ transformed_stddev stddev = 21 :=
by
  rw [h_mean, h_stddev]
  unfold transformed_mean transformed_stddev
  rw [← h_mean, ← h_stddev]
  sorry

end transformed_mean_stddev_l194_194540


namespace victor_boxes_l194_194381

theorem victor_boxes (total_books : ℕ) (books_per_box : ℕ) (boxes_bought : ℕ) 
  (h1 : total_books = 24) (h2 : books_per_box = 3) : boxes_bought = 8 :=
by {
  have h : boxes_bought = total_books / books_per_box, {
    exact nat.div_eq_of_eq_mul_left (by simp) rfl
  },
  rw [h1, h2] at h,
  exact h,
  sorry
}

end victor_boxes_l194_194381


namespace problem1_problem2_l194_194742

-- Definitions based on the first problem conditions
def expr1 := (2 + 1 / 4 : ℝ) ^ (1 / 2) - (-0.96 : ℝ) ^ 0 - (3 + 3 / 8 : ℝ) ^ (-2 / 3) + (1.5 : ℝ) ^ (-2)

-- Theorem based on the first problem result
theorem problem1 : expr1 = (1 / 2 : ℝ) :=
  by sorry

-- Definitions based on the second problem conditions
variables (a b : ℝ)

def lhs_expr2 := (2 * a^(2 / 3) * b^(1 / 2)) * (-6 * a^(1 / 2) * b^(1 / 3)) / (-3 * a^(1 / 6) * b^(5 / 6))

-- Theorem based on the second problem result
theorem problem2 : lhs_expr2 a b = 4 * a :=
  by sorry

end problem1_problem2_l194_194742


namespace sum_a_b_neg1_l194_194906

-- Define the problem using the given condition
theorem sum_a_b_neg1 (a b : ℝ) (h : |a + 3| + (b - 2) ^ 2 = 0) : a + b = -1 := 
by
  sorry

end sum_a_b_neg1_l194_194906


namespace bn_arithmetic_lambda_range_sum_inequality_l194_194842

-- Define the sequence a_n
def a : ℕ+ → ℝ
| ⟨1, _⟩ := 1
| ⟨n+1, hn⟩ := 1 - (1 / (4 * (a ⟨n, hn⟩)))

-- Define the sequence b_n
def b (n : ℕ+) : ℝ := 2 / (2 * a(n) - 1)

-- Problem (1): Proving b_{n+1} - b_n = 2
theorem bn_arithmetic (n : ℕ+) : b ⟨n+1, sorry⟩ - b n = 2 :=
sorry

-- Define the sequence c_n
def c (n : ℕ+) (λ : ℝ) : ℝ := 6^n + (-1)^(n-1) * λ * 2^(b n)

-- Problem (2): Determining the range of λ for c_{n+1} > c_n
theorem lambda_range (n : ℕ+) (λ : ℝ) : (n % 2 = 0 → λ > - ((3 / 2) ^ n)) ∧ (n % 2 = 1 → λ < (3 / 2) ^ n) :=
sorry

-- Problem (3): Proving the sum inequality
theorem sum_inequality (n : ℕ) : (∑ i in range n, 1 / (b ⟨i+1, sorry⟩ * (b ⟨i+1, sorry⟩ + 1))) < 13 / 42 :=
sorry

end bn_arithmetic_lambda_range_sum_inequality_l194_194842


namespace find_special_number_l194_194022

theorem find_special_number : 
  ∃ n, 
  (n % 12 = 11) ∧ 
  (n % 11 = 10) ∧ 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 27719) :=
  sorry

end find_special_number_l194_194022


namespace pattern_four_five_pattern_general_sum_pattern_equation_solution_l194_194812

-- Given the pattern and the specific instance for n = 4 and general n:
theorem pattern_four_five : (1 : ℝ) / (4 * 5) = (1 : ℝ) / 4 - (1 : ℝ) / 5 := sorry

theorem pattern_general (n : ℕ) : (1 : ℝ) / (n * (n + 1)) = (1 : ℝ) / n - (1 : ℝ) / (n + 1) := sorry

-- Given the summation pattern and its result:
theorem sum_pattern (n : ℕ) : (∑ k in Finset.range(n+1), (1 : ℝ) / (k * (k + 1))) = (n : ℝ) / (n + 1) := sorry

-- Given the equation and the solution x = 50:
theorem equation_solution (x : ℝ) (h : x ≠ -100): 
  (∑ k in Finset.range(50), (1 : ℝ) / ((x + 2 * k) * (x + 2 * k + 2))) = (1 : ℝ) / (x + 100) → x = 50 :=
sorry

end pattern_four_five_pattern_general_sum_pattern_equation_solution_l194_194812


namespace binomial_expansion_problem_l194_194330

noncomputable def binomial_expansion_sum_coefficients (n : ℕ) : ℤ :=
  (1 - 3) ^ n

def general_term_coefficient (n r : ℕ) : ℤ :=
  (-3) ^ r * (Nat.choose n r)

theorem binomial_expansion_problem :
  ∃ (n : ℕ), binomial_expansion_sum_coefficients n = 64 ∧ general_term_coefficient 6 2 = 135 :=
by
  sorry

end binomial_expansion_problem_l194_194330


namespace sum_of_roots_l194_194724

theorem sum_of_roots (x : ℝ) : (x - 4)^2 = 16 → x = 8 ∨ x = 0 := by
  intro h
  have h1 : x - 4 = 4 ∨ x - 4 = -4 := by
    sorry
  cases h1
  case inl h2 =>
    rw [h2] at h
    exact Or.inl (by linarith)
  case inr h2 =>
    rw [h2] at h
    exact Or.inr (by linarith)

end sum_of_roots_l194_194724


namespace broken_line_length_ge_1248_l194_194005

theorem broken_line_length_ge_1248 :
  ∀ (broken_line : ℕ → ℝ×ℝ) (side_length : ℝ) (P Q : ℝ×ℝ),
    (∀ (t : ℝ), 0 ≤ t → t ≤ 1 → ∃ (i : ℕ), (broken_line i).fst = t ∨ (broken_line i).snd = t) →
    (∀ (x y : ℝ×ℝ), |x - y| ≤ 1 → ∃ (i : ℕ), (broken_line i).fst = x.fst ∨ (broken_line i).snd = x.fst) →
    side_length = 50 → 
    (∀ (P : ℝ×ℝ), (0 ≤ P.1 ∧ P.1 ≤ side_length) ∧ (0 ≤ P.2 ∧ P.2 ≤ side_length) → 
    ∃ (Q : ℝ×ℝ), Q ∈ (set.range broken_line) ∧ dist P Q ≤ 1) →
    ∃ (n : ℕ ), 
      let L := ∑ i in finset.range n, dist (broken_line i) (broken_line (i + 1)) 
      in L ≥ 1248 :=
by sorry

end broken_line_length_ge_1248_l194_194005


namespace prime_factor_of_difference_l194_194215

-- Definition of distinct digits A and B from 1 to 9
def distinct_digits (A B : ℕ) : Prop :=
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ A ≠ B

-- Statement to prove that 3 is a prime factor of the difference
theorem prime_factor_of_difference {A B : ℕ} (h : distinct_digits A B) :
  ∃ k : ℕ, (909 * (A - B)) = 3 * k := 
sorry

end prime_factor_of_difference_l194_194215


namespace count_4_digit_palindromic_squares_is_2_l194_194187

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def count_4_digit_palindromic_squares : ℕ :=
  (Finset.range 100).filter (λ n, 32 ≤ n ∧ is_4_digit_number (n * n) ∧ is_palindrome (n * n)).card

theorem count_4_digit_palindromic_squares_is_2 : count_4_digit_palindromic_squares = 2 :=
  sorry

end count_4_digit_palindromic_squares_is_2_l194_194187


namespace value_of_a_2016_l194_194244

noncomputable def a : ℕ → ℤ
| 0     := 0  -- dummy value for index 0
| 1     := -1
| 2     := 2
| (n+3) := a (n+2) - a (n+1)

theorem value_of_a_2016 : a 2016 = -3 := 
by sorry

end value_of_a_2016_l194_194244


namespace george_faster_than_greta_l194_194893

theorem george_faster_than_greta :
    ∀ (greta_time george_time gloria_time : ℕ),
    greta_time = 6 →
    gloria_time = 8 →
    gloria_time = 2 * george_time →
    george_time = 4 →
    6 - george_time = 2 :=
by
  intros greta_time george_time gloria_time h1 h2 h3 h4
  rw [h1, h4]
  exact rfl

end george_faster_than_greta_l194_194893


namespace interest_earned_l194_194667

theorem interest_earned :
  let P : ℝ := 1500
  let r : ℝ := 0.02
  let n : ℕ := 3
  let A : ℝ := P * (1 + r) ^ n
  let interest : ℝ := A - P
  interest = 92 := 
by
  sorry

end interest_earned_l194_194667


namespace smallest_period_tan_l194_194694

theorem smallest_period_tan (h : ∀ x, tan (x + π) = tan x) :
  ∀ x, ∃ P > (0 : ℝ), (∀ x, tan (2 * x + π / 6 + P) = tan (2 * x + π / 6)) :=
by
  sorry

end smallest_period_tan_l194_194694


namespace negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l194_194135

open Real

theorem negation_of_exists_sin_gt_one_equiv_forall_sin_le_one :
  (¬ (∃ x : ℝ, sin x > 1)) ↔ (∀ x : ℝ, sin x ≤ 1) :=
sorry

end negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l194_194135


namespace volume_of_cone_l194_194751

variables (S L : ℝ)
-- Area of the triangle
axiom h_area : (∃ r h : ℝ, 0 < r ∧ 0 < h ∧ (1/2) * r * h = S)

-- Length of the circumference described by the intersection point of the medians
axiom h_circumference : (∃ r : ℝ, 2 * π * (r / 3) = L)

theorem volume_of_cone : ∃ V : ℝ, V = S * L :=
by
  use S * L
  sorry

end volume_of_cone_l194_194751


namespace pythagorean_theorem_l194_194943

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l194_194943


namespace three_digit_integer_equal_sum_factorials_l194_194725

open Nat

theorem three_digit_integer_equal_sum_factorials :
  ∃ (a b c : ℕ), a = 1 ∧ b = 4 ∧ c = 5 ∧ 100 * a + 10 * b + c = a.factorial + b.factorial + c.factorial :=
by
  use 1, 4, 5
  simp
  sorry

end three_digit_integer_equal_sum_factorials_l194_194725


namespace sum_of_all_areas_is_correct_l194_194248

noncomputable def sum_of_areas_of_squares : ℕ → ℝ
| 0     := 1
| (n+1) := (sqrt 3 / 2) ^ (2 * (n + 1)) * sum_of_areas_of_squares n

theorem sum_of_all_areas_is_correct :
  ∀ n, (∑ i in range (n + 1), sum_of_areas_of_squares i) = (1 + sqrt 3) / 2 :=
by
  sorry

end sum_of_all_areas_is_correct_l194_194248


namespace sum_of_100th_group_is_1010100_l194_194884

theorem sum_of_100th_group_is_1010100 : (100 + 100^2 + 100^3) = 1010100 :=
by
  sorry

end sum_of_100th_group_is_1010100_l194_194884


namespace exists_no_x_y_u_v_l194_194249

theorem exists_no_x_y_u_v (n : ℕ) (h : n ≥ 2 ^ 2018) :
  ∃∞ (n : ℕ), n ≥ (2 ^ 2018) ∧ (¬∃ x y u v : ℕ, u > 1 ∧ v > 1 ∧ n = x ^ u + y ^ v) :=
sorry

end exists_no_x_y_u_v_l194_194249


namespace sum_in_range_l194_194458

def a : ℚ := 4 + 1/4
def b : ℚ := 2 + 3/4
def c : ℚ := 7 + 1/8

theorem sum_in_range : 14 < a + b + c ∧ a + b + c < 15 := by
  sorry

end sum_in_range_l194_194458


namespace find_C_l194_194760

theorem find_C (A B C : ℕ)
  (hA : A = 348)
  (hB : B = A + 173)
  (hC : C = B + 299) :
  C = 820 :=
sorry

end find_C_l194_194760


namespace white_line_length_l194_194814

theorem white_line_length :
  ∀ (blue_line white_line : ℝ), 
    blue_line = 3.3333333333333335 ∧ 
    white_line = blue_line + 4.333333333333333 →
    white_line = 7.666666666666667 :=
by
  intros blue_line white_line h
  cases h with h_blue h_white
  rw [h_blue, h_white]
  sorry

end white_line_length_l194_194814


namespace average_income_A_B_l194_194671

def monthly_incomes (A B C : ℝ) : Prop :=
  (A = 4000) ∧
  ((B + C) / 2 = 6250) ∧
  ((A + C) / 2 = 5200)

theorem average_income_A_B (A B C X : ℝ) (h : monthly_incomes A B C) : X = 5050 :=
by
  have hA : A = 4000 := h.1
  have hBC : (B + C) / 2 = 6250 := h.2.1
  have hAC : (A + C) / 2 = 5200 := h.2.2
  sorry

end average_income_A_B_l194_194671


namespace volume_at_10_l194_194832

noncomputable def gas_volume (T : ℝ) : ℝ :=
  if T = 30 then 40 else 40 - (30 - T) / 5 * 5

theorem volume_at_10 :
  gas_volume 10 = 20 :=
by
  simp [gas_volume]
  sorry

end volume_at_10_l194_194832


namespace altitude_difference_l194_194317

theorem altitude_difference 
  (alt_A : ℤ) (alt_B : ℤ) (alt_C : ℤ)
  (hA : alt_A = -102) (hB : alt_B = -80) (hC : alt_C = -25) :
  (max (max alt_A alt_B) alt_C) - (min (min alt_A alt_B) alt_C) = 77 := 
by 
  sorry

end altitude_difference_l194_194317


namespace stock_percent_change_l194_194252

variable (x : ℝ)

theorem stock_percent_change (h1 : ∀ x, 0.75 * x = x * 0.75)
                             (h2 : ∀ x, 1.05 * x = 0.75 * x + 0.3 * 0.75 * x):
    ((1.05 * x - x) / x) * 100 = 5 :=
by
  sorry

end stock_percent_change_l194_194252


namespace evaluate_y_correct_l194_194077

noncomputable def evaluate_y (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - 4 * x + 4) + Real.sqrt (x^2 + 6 * x + 9) - 2

theorem evaluate_y_correct (x : ℝ) : 
  evaluate_y x = |x - 2| + |x + 3| - 2 :=
by 
  sorry

end evaluate_y_correct_l194_194077


namespace planting_flowers_cost_l194_194333

theorem planting_flowers_cost 
  (flower_cost : ℕ) (clay_cost : ℕ) (soil_cost : ℕ)
  (h₁ : flower_cost = 9)
  (h₂ : clay_cost = flower_cost + 20)
  (h₃ : soil_cost = flower_cost - 2) :
  flower_cost + clay_cost + soil_cost = 45 :=
sorry

end planting_flowers_cost_l194_194333


namespace determine_d_minus_r_l194_194068

theorem determine_d_minus_r :
  ∃ d r: ℕ, (∀ n ∈ [2023, 2459, 3571], n % d = r) ∧ (1 < d) ∧ (d - r = 1) :=
sorry

end determine_d_minus_r_l194_194068


namespace four_digit_palindrome_squares_count_l194_194207

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem four_digit_palindrome_squares_count : (Finset.filter (λ n, is_palindrome (n * n)) (Finset.range 100)).card = 2 := by
  sorry

end four_digit_palindrome_squares_count_l194_194207


namespace problem_statement_l194_194517

noncomputable def ellipse_satisfied_conditions (a b c : ℝ) (h : a > b ∧ b > 0 ∧ c^2 = a^2 - b^2)
  (hx : a^2 / c = 3) (focus_condition : ((0, -2 * sqrt(3)) ∈ line_passing_focus (l1 : ℝ × ℝ → ℝ)))
  : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ ellipse_equation a b ∧ ellipse_center_symmetric l1 (0, 0)

noncomputable def line_eq_l2 (k : ℝ) : Prop :=
  k = -2 ∨ k = sqrt(3) / 3 ∨ k = -sqrt(3) / 3

theorem problem_statement : 
  ∃ (a b c : ℝ), (a > b ∧ b > 0 ∧ c^2 = a^2 - b^2 ∧ a^2 / c = 3 ∧ ((0, -2 * sqrt(3)) ∈ line_passing_focus)
  (focus_cond : ellipse_satisfied_conditions a b c (a > b ∧ b > 0 ∧ c^2 = a^2 - b^2) (a^2 / c = 3) ((0, -2 * sqrt(3)) ∈ line_passing_focus))
  (l2_eq : line_eq_l2 (B : ℝ × ℝ → ℝ → ℝ) (intersect_ellipse (B (-2, 0)) ellipse (∠MON ≠ π/2) ((OM · ON) * sin ∠MON = (4 * sqrt(6)) / 3))) : 
  ∀ (x y : ℝ), (x, y) ∈ ellipse_equation 6 2 ↔ l2_eq x ∨ l2_eq y := 
sorry

end problem_statement_l194_194517


namespace sum_min_max_m_eq_10_l194_194869

theorem sum_min_max_m_eq_10 :
  let C := (3, 4)
  let r := 1
  ∃ m : ℝ, ∃ P : ℝ × ℝ, ((P.1 - 3)^2 + (P.2 - 4)^2 = 1) ∧ 
    (let A := (-m, 0) in let B := (m, 0) in ∃ P : ℝ × ℝ, 
    ((P.1 - 3)^2 + (P.2 - 4)^2 = 1) ∧ (∠ P A B = 90°)) →
    let OC := 5 
    let max_m := OC + r
    let min_m := OC - r
    min_m + max_m = 10 := 
by
  sorry

end sum_min_max_m_eq_10_l194_194869


namespace Meghan_scored_20_marks_less_than_Jose_l194_194231

theorem Meghan_scored_20_marks_less_than_Jose
  (M J A : ℕ)
  (h1 : J = A + 40)
  (h2 : M + J + A = 210)
  (h3 : J = 100 - 10) :
  J - M = 20 :=
by
  -- Skipping the proof
  sorry

end Meghan_scored_20_marks_less_than_Jose_l194_194231


namespace max_min_distance_l194_194929

def grid_size : ℕ := 100
def num_cells : ℕ := grid_size * grid_size
def max_distance : ℝ := 50 * real.sqrt 2

def grid_position (n : ℕ) : (ℕ × ℕ) :=
  let row := (n - 1) / grid_size + 1
  let col := (n - 1) % grid_size + 1
  (row, col)

def cell_distance (pos1 pos2 : ℕ × ℕ) : ℝ :=
  real.sqrt ((pos1.1 - pos2.1) ^ 2 + (pos1.2 - pos2.2) ^ 2)

theorem max_min_distance :
  ∃ (S : ℝ), 
    (∀ i j, 1 ≤ i ∧ i ≤ num_cells ∧ 1 ≤ j ∧ j ≤ num_cells ∧ (i - j = 5000 ∨ j - i = 5000) →
      S ≤ cell_distance (grid_position i) (grid_position j)) ∧
    (∀ T, (∀ i j, 1 ≤ i ∧ i ≤ num_cells ∧ 1 ≤ j ∧ j ≤ num_cells ∧ (i - j = 5000 ∨ j - i = 5000) →
      T ≤ cell_distance (grid_position i) (grid_position j)) → S ≥ T) :=
begin
  use max_distance,
  sorry
end

end max_min_distance_l194_194929


namespace rectangle_area_l194_194428

theorem rectangle_area (p : ℝ) (l : ℝ) (h1 : 2 * (l + 2 * l) = p) :
  l * 2 * l = p^2 / 18 :=
by
  sorry

end rectangle_area_l194_194428


namespace car_tire_circumferences_l194_194025

noncomputable def speed_in_m_per_min (speed_kmh : ℕ) : ℚ :=
  (speed_kmh * 1000 : ℚ) / 60

def carA_speed := 100
def carA_rpm := 450
def carB_speed := 120
def carB_rpm := 400

noncomputable def circumference (speed : ℚ) (rpm : ℕ) : ℚ :=
  speed / rpm

def carA_speed_m_per_min := speed_in_m_per_min carA_speed
def carA_circumference := circumference carA_speed_m_per_min carA_rpm

def carB_speed_m_per_min := speed_in_m_per_min carB_speed
def carB_circumference := circumference carB_speed_m_per_min carB_rpm

theorem car_tire_circumferences :
  carA_circumference = 3.70 ∧ carB_circumference = 5 := by
  sorry

end car_tire_circumferences_l194_194025


namespace equation_of_hyperbola_range_of_k_l194_194877

theorem equation_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ x y : ℝ, x = sqrt 6 ∧ y = sqrt 3 ∧ (x^2 / a^2) - (y^2 / b^2) = 1)
  (h4 : ∃ x : ℝ, x = - sqrt 6 ∧ (x^2 / a^2) - 0 / b^2 = 1) :
  (a^2 = 3 ∧ b^2 = 3) ↔ ∀ x y : ℝ, (x^2 - y^2 = 3) :=
sorry

theorem range_of_k (k : ℝ) :
  (∀ x y : ℝ, (x^2 - y^2 = 3 ∧ y = k*x + 2) → (x^2 * (1 - k^2) - 4*k*x - 7 = 0))
  ↔ k ∈ Ioo (- sqrt 21 / 3) -1 :=
sorry

end equation_of_hyperbola_range_of_k_l194_194877


namespace sum_of_coefficients_l194_194072

theorem sum_of_coefficients : 
  (x : ℕ) (y : ℕ),
  let p := (x^2 - 3*x*y + 2*y^2)^7 
  in sum_of_coefficients p = 0 := sorry

end sum_of_coefficients_l194_194072


namespace isogonal_conjugate_law_l194_194274

open Lean
open Mathlib.Geometry.EuclideanTriangle

-- Given conditions: P inside the triangle ABC and Q as its isogonal conjugate
variable (A B C P Q : Point ℝ)
variable [IsogonalConjugate A B C P Q]

-- Proof statement (theorem) in Lean
theorem isogonal_conjugate_law (hPQ : IsogonalConjugate A B C P Q) :
  (dist A P) * (dist A Q) * (dist B C) + 
  (dist B P) * (dist B Q) * (dist A C) + 
  (dist C P) * (dist C Q) * (dist A B) = 
  (dist A B) * (dist B C) * (dist C A) :=
by
  -- Placeholders for the proof
  sorry

end isogonal_conjugate_law_l194_194274


namespace number_of_subsets_sum_of_elements_all_subsets_l194_194886

open Set

namespace Proof

noncomputable def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem number_of_subsets : ∃ n, n = 2^(10) ∧ n = card (powerset M) :=
by
  sorry

theorem sum_of_elements_all_subsets : 
  let sum_all_subsets := (list.sum [1,2,3,4,5,6,7,8,9,10]) * 2^(9)
  in ∃ s, s = sum_all_subsets :=
by
  sorry

end Proof

end number_of_subsets_sum_of_elements_all_subsets_l194_194886


namespace problem_statement_l194_194816

theorem problem_statement (x : ℝ) : 
  ( (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) > (1 / 4) ) ↔ x ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 0 2 :=
sorry

end problem_statement_l194_194816


namespace greatest_possible_median_of_nine_nonnegative_numbers_l194_194296

theorem greatest_possible_median_of_nine_nonnegative_numbers (x : fin 9 → ℝ) (h1 : ∀ i, 0 ≤ x i) 
  (h2 : (∑ i, x i) / 9 = 10) : (∀ y : fin 9 → ℝ, (∀ i, 0 ≤ y i) → (∑ i, y i) / 9 = 10 → 
  let sorted_y := sort (<=) y in (sorted_y 4 ≤ 18)) :=
begin
  sorry
end

end greatest_possible_median_of_nine_nonnegative_numbers_l194_194296


namespace floor_of_factorial_expression_l194_194058

theorem floor_of_factorial_expression :
  ∀ (a b c d e : ℕ),
  a = 2008 → b = 2007 → c = 2006 → d = 2005 → e = 2004 →
  (⌊(a! + d!) / (b! + c!)⌋ = 2007) :=
by sorry

end floor_of_factorial_expression_l194_194058


namespace average_waiting_time_l194_194433

-- Define the problem conditions
def light_period : ℕ := 3  -- Total cycle time in minutes
def green_time : ℕ := 1    -- Green light duration in minutes
def red_time : ℕ := 2      -- Red light duration in minutes

-- Define the probabilities of each light state
def P_G : ℚ := green_time / light_period
def P_R : ℚ := red_time / light_period

-- Define the expected waiting times given each state
def E_T_G : ℚ := 0
def E_T_R : ℚ := red_time / 2

-- Calculate the expected waiting time using the law of total expectation
def E_T : ℚ := E_T_G * P_G + E_T_R * P_R

-- Convert the expected waiting time to seconds
def E_T_seconds : ℚ := E_T * 60

-- Prove that the expected waiting time in seconds is 40 seconds
theorem average_waiting_time : E_T_seconds = 40 := by
  sorry

end average_waiting_time_l194_194433


namespace combined_weight_l194_194290

theorem combined_weight (a b c : ℕ) (h1 : a + b = 122) (h2 : b + c = 125) (h3 : c + a = 127) : 
  a + b + c = 187 :=
by
  sorry

end combined_weight_l194_194290


namespace cos_negative_19pi_over_6_eq_neg_sqrt_3_over_2_l194_194476

theorem cos_negative_19pi_over_6_eq_neg_sqrt_3_over_2 :
  cos (-19 * π / 6) = -real.sqrt 3 / 2 :=
by
  have h1 : cos (- (-4 * π + 5 * π / 6)) = cos (-4 * π + 5 * π / 6),
  -- cosine is an even function
  { exact real.cos_neg _ },
  rw [h1],
  have h2 : cos (4 * π + 5 * π / 6) = cos (5 * π / 6),
  -- 4π is a multiple of 2π, which does not change the value of cosine
  { exact real.cos_add_int_mul_two_pi _ _ },
  rw [h2],
  have h3 : 5 * π / 6 = π - π / 6,
  -- simplification
  { refl },
  rw [h3, real.cos_pi_sub],
  -- applying the cofunction identity
  have h4 : cos (π / 6) = real.sqrt 3 / 2,
  { sorry },
  rw [h4],
  norm_num,
  -- final simplification to get the result

end cos_negative_19pi_over_6_eq_neg_sqrt_3_over_2_l194_194476


namespace books_club_per_month_l194_194649

def books_start : ℕ := 72
def books_end : ℕ := 81
def books_purchased_bookstore : ℕ := 5
def books_purchased_yardsale : ℕ := 2
def books_received_daughter : ℕ := 1
def books_received_mother : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

def net_books_received (books_club : ℕ) : Prop :=
  books_end = books_start + books_purchased_bookstore + books_purchased_yardsale + books_received_daughter + books_received_mother 
  - (books_donated + books_sold) + books_club

theorem books_club_per_month (books_club : ℕ) : books_club = 12 → books_club / 12 = 1 :=
by intro h; rw [h]; norm_num; sorry

end books_club_per_month_l194_194649


namespace yellow_marbles_in_basket_A_l194_194706

theorem yellow_marbles_in_basket_A : 
  ∀ (Y : ℕ), 
  (∀ (dif_A dif_B dif_C : ℕ), 
    dif_A = | 4 - Y | ∧ dif_B = | 6 - 1 | ∧ dif_C = | 9 - 3 | ∧ 
    dif_C = 6 ∧ (dif_A ≤ 6)) → Y = 10 :=
by
  intros Y h
  sorry

end yellow_marbles_in_basket_A_l194_194706


namespace six_digit_perfect_square_l194_194383

theorem six_digit_perfect_square :
  ∃ n : ℕ, ∃ x : ℕ, (n ^ 2 = 763876) ∧ (n ^ 2 >= 100000) ∧ (n ^ 2 < 1000000) ∧ (5 ≤ x) ∧ (x < 50) ∧ (76 * 10000 + 38 * 100 + 76 = 763876) ∧ (38 = 76 / 2) :=
by
  sorry

end six_digit_perfect_square_l194_194383


namespace circle_center_radius_sum_l194_194621

theorem circle_center_radius_sum :
  let c := 3
  let d := -1
  let s := 3 * Real.sqrt 2
  (x y : ℝ) in
    (x^2 + 2 * y - 8 = -(y^2) + 6 * x) →
    ((x - c)^2 + (y - d)^2 = 18) →
    c + d + s = 2 + 3 * Real.sqrt 2 :=
by
  intros
  sorry

end circle_center_radius_sum_l194_194621
