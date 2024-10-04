import Mathlib
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.CharZero
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.ArithmeticMean
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialLemmas
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Range
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vec.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Init.Data.Int.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.LinearAlgebra.Projection
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction.Fin
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Statistics.Variance
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Real

namespace all_nonnegative_integers_can_be_covered_with_disjoint_historic_sets_l257_257897

def is_historic_set (x y z : ℕ) : Prop := 
  x < y ∧ y < z ∧ ({z - y, y - x} = {1776, 2001})

def covers_all_nonnegative_integers (historic_sets : set (set ℕ)) : Prop := 
  (∀ n : ℕ, ∃ s ∈ historic_sets, n ∈ s)

def are_disjoint_sets (historic_sets : set (set ℕ)) : Prop :=
  ∀ s1 s2 ∈ historic_sets, s1 ≠ s2 → s1 ∩ s2 = ∅

theorem all_nonnegative_integers_can_be_covered_with_disjoint_historic_sets :
  ∃ (historic_sets : set (set ℕ)), 
    (∀ s ∈ historic_sets, ∃ x y z : ℕ, s = {x, y, z} ∧ is_historic_set x y z) ∧
    covers_all_nonnegative_integers historic_sets ∧
    are_disjoint_sets historic_sets :=
sorry

end all_nonnegative_integers_can_be_covered_with_disjoint_historic_sets_l257_257897


namespace pow_mod_equality_l257_257773

theorem pow_mod_equality : 
  ∃ m : ℤ, 0 ≤ m ∧ m < 11 ∧ 13^6 ≡ m [MOD 11] ∧ m = 9 :=
by {
  use 9,
  split, 
  norm_num,
  split,
  norm_num,
  split,
  exact_mod_cast nat.gcd_pow(13, 6, 11),
  refl
}

end pow_mod_equality_l257_257773


namespace at_least_three_points_in_circle_l257_257750

theorem at_least_three_points_in_circle (points : Fin 51 → (ℝ × ℝ)) :
  (∀ i, 0 ≤ points i.1 ∧ points i.1.1 ≤ 1 ∧ 0 ≤ points i.2 ∧ points i.2 ≤ 1) →
  ∃ (c : ℝ × ℝ) (r : ℝ), r = 1 / 7 ∧ (∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
    dist (points i) c ≤ r ∧ dist (points j) c ≤ r ∧ dist (points k) c ≤ r) :=
begin
  sorry
end

end at_least_three_points_in_circle_l257_257750


namespace hypotenuse_right_triangle_l257_257828

theorem hypotenuse_right_triangle (a b : ℕ) (h₁ : a = 45) (h₂ : b = 60) : ∃ (c : ℕ), c = 75 :=
by 
  have h₃ : a * a + b * b = 45 * 45 + 60 * 60 := by rw [h₁, h₂]
  have h₄ : 45 * 45 + 60 * 60 = c * c := by sorry
  use 75
  have : 75 * 75 = c * c := by sorry
  exact this

end hypotenuse_right_triangle_l257_257828


namespace minimum_value_of_d_l257_257320

noncomputable def min_length_chord (α : ℝ) : ℝ :=
let x1 := Real.arcsin α in
let x2 := Real.arccos α in
let y1 := Real.arcsin α in
let y2 := -Real.arccos α in
Real.sqrt (2 * ((x1 * x1) + (x2 * x2)))

theorem minimum_value_of_d (α : ℝ) (h1 : -1 ≤ α) (h2 : α ≤ 1) :
  ∃ d ≥ 0.5 * π, min_length_chord α = d := sorry

end minimum_value_of_d_l257_257320


namespace compute_y_l257_257674

theorem compute_y (y : ℝ) (h : log 3 (y^3) + log (1/3) y = 6) : y = 27 := by
  sorry

end compute_y_l257_257674


namespace new_students_admitted_l257_257128

theorem new_students_admitted :
  let S_original := 15 in
  let students_per_original_section := 23 in
  let new_sections := 5 in
  let total_sections := 20 in
  let students_per_new_section := 19 in
  23 * 15 + 35 = 19 * 20 :=
by
  let S_original := 20 - 5
  let total_students_before := 23 * S_original
  let total_students_after := 19 * total_sections
  have : total_students_after - total_students_before = 35 := sorry
  exact this

end new_students_admitted_l257_257128


namespace percent_non_condiments_l257_257895

def sandwich_weight : ℕ := 150
def condiment_weight : ℕ := 45
def non_condiment_weight (total: ℕ) (condiments: ℕ) : ℕ := total - condiments
def percentage (num denom: ℕ) : ℕ := (num * 100) / denom

theorem percent_non_condiments : 
  percentage (non_condiment_weight sandwich_weight condiment_weight) sandwich_weight = 70 :=
by
  sorry

end percent_non_condiments_l257_257895


namespace sandy_age_l257_257126

-- Definitions corresponding to the problem's conditions
def ageOfSandyAndMolly (M S : ℕ) : Prop :=
  S = M - 16 ∧ S * 9 = 7 * M

-- The main problem statement to prove that Sandy's age is 56 given the conditions
theorem sandy_age (M S : ℕ) (h : ageOfSandyAndMolly M S) : S = 56 :=
begin
  sorry
end

end sandy_age_l257_257126


namespace whole_number_closest_to_area_l257_257889

noncomputable def area_of_shaded_region (rect_length rect_width diam_semi : ℝ) : ℝ :=
  let radius := diam_semi / 2 in
  let area_rect := rect_length * rect_width in
  let area_semi := (real.pi * radius * radius) / 2 in
  area_rect - area_semi

theorem whole_number_closest_to_area :
  let shaded_area := area_of_shaded_region 4 5 2 in
  int.ceil (shaded_area - 0.5) = 18 :=
by
  let shaded_area := area_of_shaded_region 4 5 2
  sorry

end whole_number_closest_to_area_l257_257889


namespace max_rectangle_area_l257_257638

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l257_257638


namespace problem_1_problem_2_l257_257185

-- Problem 1
def prob1_expr : ℝ :=
  -36 * ((3 / 4) - (1 / 6) + (2 / 9) - (5 / 12)) + abs (-21 / 5 / (7 / 25))
def prob1_answer : ℝ := 61
theorem problem_1 : prob1_expr = prob1_answer := by sorry

-- Problem 2
def prob2_expr : ℝ :=
  (-3)^3 - (((-1)^3 / (5 / 2)) + (9 / 4 * -4)) / (362 / 15 - 407 / 15)
def prob2_answer : ℝ := -143 / 6
theorem problem_2 : prob2_expr = prob2_answer := by sorry

end problem_1_problem_2_l257_257185


namespace total_blocks_minimum_fare_maximum_fare_l257_257863

theorem total_blocks : 
  let horizontal_streets := 7; 
  let vertical_streets := 13; 
  (horizontal_streets - 1) * (vertical_streets - 1) = 72 := 
  sorry

theorem minimum_fare (pickup_i pickup_j dest_i dest_j : ℕ) (side_length fare_per_100m: ℕ) : 
  let pickup_i := 4; 
  let pickup_j := 2;
  let dest_i := 1; 
  let dest_j := 9;
  let side_length := 100;
  let fare_per_100m := 1; 
  (abs (dest_j - pickup_j) * side_length + abs (pickup_i - dest_i) * side_length) / 100 * fare_per_100m = 10 :=
  sorry

theorem maximum_fare (pickup_i pickup_j dest_i dest_j : ℕ) (side_length fare_per_100m: ℕ) :
  let pickup_i := 4; 
  let pickup_j := 2;
  let dest_i := 1; 
  let dest_j := 9;
  let side_length := 100;
  let fare_per_100m := 1; 
  (abs (dest_j - pickup_j) * side_length + abs (pickup_i - dest_i) * side_length) / 100 * fare_per_100m ≤ 12 :=
  sorry

end total_blocks_minimum_fare_maximum_fare_l257_257863


namespace twice_brother_age_l257_257326

theorem twice_brother_age (current_my_age : ℕ) (current_brother_age : ℕ) (years : ℕ) :
  current_my_age = 20 →
  (current_my_age + years) + (current_brother_age + years) = 45 →
  current_my_age + years = 2 * (current_brother_age + years) →
  years = 10 :=
by 
  intros h1 h2 h3
  sorry

end twice_brother_age_l257_257326


namespace more_perfect_squares_with_7_digit_17th_l257_257839

noncomputable def seventeenth_digit (n : ℕ) : ℕ :=
  (n / 10^16) % 10

theorem more_perfect_squares_with_7_digit_17th
  (h_bound : ∀ n, n < 10^10 → (n * n) < 10^20)
  (h_representation : ∀ m, m < 10^20 → ∃ n, n < 10^10 ∧ m = n * n) :
  (∃ majority_digit_7 : ℕ,
    (∃ majority_digit_8 : ℕ,
      ∀ n, seventeenth_digit (n * n) = 7 → majority_digit_7 > majority_digit_8)
  ) :=
sorry

end more_perfect_squares_with_7_digit_17th_l257_257839


namespace problem_statement_l257_257581

theorem problem_statement 
  {α β : ℝ}
  (h1 : sin α + cos β = 0)
  (h2 : sin^2 α + sin^2 β = 1) : 
  true := 
sorry

end problem_statement_l257_257581


namespace concentration_of_fourth_cup_l257_257812

theorem concentration_of_fourth_cup (a : ℝ) :
  let concentration_cup1 := 10 / 100
  let concentration_cup2 := 20 / 100
  let concentration_cup3 := 45 / 100
  let fraction_cup1 := 1 / 2
  let fraction_cup2 := 1 / 4
  let fraction_cup3 := 1 / 5
  let dissolved_mass_cup1 := fraction_cup1 * concentration_cup1 * a
  let dissolved_mass_cup2 := fraction_cup2 * concentration_cup2 * a
  let dissolved_mass_cup3 := fraction_cup3 * concentration_cup3 * a
  let total_transferred_mass := fraction_cup1 * a + fraction_cup2 * a + fraction_cup3 * a
  let total_dissolved_mass := dissolved_mass_cup1 + dissolved_mass_cup2 + dissolved_mass_cup3
  let concentration := (total_dissolved_mass / total_transferred_mass) * 100
  in concentration = 20 :=
by {
  -- Conditions explicitly stated
  let concentration_cup1 := 10 / 100
  let concentration_cup2 := 20 / 100
  let concentration_cup3 := 45 / 100
  let fraction_cup1 := 1 / 2
  let fraction_cup2 := 1 / 4
  let fraction_cup3 := 1 / 5
  
  -- Correct answer was given
  sorry
}

end concentration_of_fourth_cup_l257_257812


namespace max_area_of_rectangle_with_perimeter_60_l257_257618

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l257_257618


namespace trig_identity_l257_257932

theorem trig_identity : 
  sin (4/3 * Real.pi) * cos (11/6 * Real.pi) * tan (3/4 * Real.pi) = 3/4 := 
by 
  sorry

end trig_identity_l257_257932


namespace joe_can_determine_faulty_chair_l257_257938

-- Definitions from the conditions
def chair := {c : Type} (chair1 : c) (chair2 : c)

structure Chairs (c : Type) :=
(faulty_chair : c)
(other_chair  : c)

axiom one_faulty (c : Type) (ch : Chairs c) :
  ∃ (x : c), x = ch.faulty_chair

-- Guard alternates between truth and lies
inductive TruthDay : Type
| truth_day : TruthDay
| lie_day : TruthDay

-- Guard's response depends on the day condition
def guard_response (ch : Chairs Type) (day : TruthDay) (question : String) : String :=
match day with
| TruthDay.truth_day => if question = "Which chair is faulty?" then ch.faulty_chair.toString else ch.other_chair.toString
| TruthDay.lie_day   => if question = "Which chair is faulty?" then ch.other_chair.toString else ch.faulty_chair.toString

-- The question Joe should ask to determine the faulty chair
theorem joe_can_determine_faulty_chair (ch : Chairs Type) (today : TruthDay) :
  let answer :=
    match today with
    | TruthDay.truth_day => guard_response ch TruthDay.lie_day "Which chair is faulty?"
    | TruthDay.lie_day   => guard_response ch TruthDay.truth_day "Which chair is faulty?"
  in answer = ch.faulty_chair.toString :=
by
  sorry

end joe_can_determine_faulty_chair_l257_257938


namespace surface_area_of_sphere_l257_257990

-- Define conditions
def is_right_prism (P : Prism) : Prop := sorry
def all_vertices_on_sphere (P : Prism) (S : Sphere) : Prop := sorry
def prism_height (P : Prism) : ℝ := 4
def prism_volume (P : Prism) : ℝ := 32

-- Assume existence of Prism and Sphere
variables (P : Prism) (S : Sphere)

-- Correct answer
def sphere_surface_area (S : Sphere) : ℝ := 32 * π

-- Main theorem
theorem surface_area_of_sphere :
  is_right_prism P ∧ all_vertices_on_sphere P S ∧ prism_height P = 4 ∧ prism_volume P = 32 →
  sphere_surface_area S = 32 * π :=
begin
  sorry
end

end surface_area_of_sphere_l257_257990


namespace factor_expression_l257_257935

variable {a : ℝ}

theorem factor_expression :
  ((10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32)) = 4 * (3 * a^3 * (a - 12) - 16) :=
by
  sorry

end factor_expression_l257_257935


namespace odd_function_equiv_l257_257318

noncomputable def odd_function (f : ℝ → ℝ) :=
∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_equiv (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f (x)) ↔ (∀ x : ℝ, f (-(-x)) = -f (-x)) :=
by
  sorry

end odd_function_equiv_l257_257318


namespace incorrect_min_value_of_sqrt_sum_l257_257583

theorem incorrect_min_value_of_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) :
  ∀ x, (x = √a + √b) → x ≠ √2 :=
sorry

end incorrect_min_value_of_sqrt_sum_l257_257583


namespace sin_C_in_right_triangle_l257_257698

theorem sin_C_in_right_triangle (A B C : ℝ) 
  (hA : sin A = 1) 
  (hB : cos B = 3/5) 
  (right_angle_A : A = π / 2) 
  (angle_sum : A + B + C = π) : 
  sin C = 3/5 :=
by 
  sorry

end sin_C_in_right_triangle_l257_257698


namespace certain_number_eq_1000_l257_257470

theorem certain_number_eq_1000 (x : ℝ) (h : 3500 - x / 20.50 = 3451.2195121951218) : x = 1000 := 
by
  sorry

end certain_number_eq_1000_l257_257470


namespace cos_angle_between_planes_l257_257730

def n1 : ℝ × ℝ × ℝ := (3, -2, 1)
def n2 : ℝ × ℝ × ℝ := (9, -6, -4)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem cos_angle_between_planes :
  let cos_theta := dot_product n1 n2 / (magnitude n1 * magnitude n2)
  cos_theta = 35 / Real.sqrt 1862 :=
by
  sorry

end cos_angle_between_planes_l257_257730


namespace base_nine_to_ten_l257_257138

theorem base_nine_to_ten (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 7) (h₃ : c = 3) :
  a * 9^0 + b * 9^1 + c * 9^2 = 312 :=
by 
  rw [h₁, h₂, h₃]
  norm_num
  exact sorry

end base_nine_to_ten_l257_257138


namespace congruent_triangles_in_triangle_l257_257710

/-- In triangle ABC, with centroid P and D, E, F being the midpoints of BC, CA, and AB respectively,
proving that the triangles formed are congruent in opposite pairs. -/
theorem congruent_triangles_in_triangle 
  (A B C P D E F : Type) 
  [IsTriangle ABC P]
  [Centroid ABC P]
  [Midpoint D (BC)]
  [Midpoint E (CA)]
  [Midpoint F (AB)] :
  congruent (triangle A P D) (triangle E P D) ∧ 
  congruent (triangle B P E) (triangle D P F) ∧
  congruent (triangle C P F) (triangle F P E) :=
sorry

end congruent_triangles_in_triangle_l257_257710


namespace smallest_prime_with_digit_sum_19_l257_257102

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_with_digit_sum_19 : ∃ p : ℕ, is_prime p ∧ digit_sum p = 19 ∧ ∀ q : ℕ, is_prime q ∧ digit_sum q = 19 → p ≤ q :=
  ∃ p, p = 199 ∧ is_prime p ∧ digit_sum p = 19 ∧ ∀ q, is_prime q ∧ digit_sum q = 19 → p ≤ q :=
begin
  sorry
end

end smallest_prime_with_digit_sum_19_l257_257102


namespace popularity_of_blender_and_toaster_l257_257533

variable (p_b : ℕ) (c_b : ℕ) (k_b : ℕ := 8000)
variable (p_t : ℕ) (c_t : ℕ) (k_t : ℕ := 6000)

theorem popularity_of_blender_and_toaster :
  (20 * 400 = k_b) → (p_b * 800 = k_b) →
  (p_t * 600 = k_t) → (p_b = 10 ∧ p_t = 10) :=
by
  intros h1 h2 h3
  have hp1 : k_b = 8000 := by simp [h1]
  have hp2 : p_b = 8000 / 800 := by simp [h2, hp1]
  have hp3 : p_b = 10 := by norm_num [hp2]
  have ht1 : k_t = 6000 := by simp [h3]
  have ht2 : p_t = 6000 / 600 := by simp [h3, ht1]
  have ht3 : p_t = 10 := by norm_num [ht2]
  exact ⟨hp3, ht3⟩

end popularity_of_blender_and_toaster_l257_257533


namespace complex_in_fourth_quadrant_l257_257575

theorem complex_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8*m + 15 > 0) ∧ (m^2 - 5*m - 14 < 0) →
  (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

end complex_in_fourth_quadrant_l257_257575


namespace total_amount_spent_is_198_l257_257485

/-
A couple spent some amount while dining out and paid this amount using a credit card. 
The amount included a 20 percent tip which was paid on top of the price which 
already included a sales tax of 10 percent on top of the price of the food. 
Prove that the total amount they spent is $198 given that the actual price of the food before tax and tip was $150.
-/

def actual_price_of_food : ℝ := 150
def sales_tax_rate : ℝ := 0.10
def tip_rate : ℝ := 0.20

theorem total_amount_spent_is_198 
  (actual_price_of_food = 150) 
  (sales_tax_rate = 0.10) 
  (tip_rate = 0.20) : 
  ((actual_price_of_food * (1 + sales_tax_rate)) * (1 + tip_rate) = 198) :=
by
  sorry

end total_amount_spent_is_198_l257_257485


namespace sum_of_first_n_terms_l257_257705

theorem sum_of_first_n_terms (d q : ℕ) (h1 : d = q + 1) (h2 : q > 0) 
  (h3 : 1 + d + q^2 = 8) (h4 : 1 + 2*d + q = 9) :
  (∀ n : ℕ, (n % 2 = 0 → nat.sum (λ i, 2^((i * 3) - 1)) ⟨n/2, sorry⟩ = (6 * (8^(n/2) - 1)) / 7 ))
  ∧ (∀ n : ℕ, (n % 2 = 1 → 2 + nat.sum (λ i, 2^((i * 3) - 1)) ⟨(n - 1)/2, sorry⟩ = (20 * (8^((n-1)/2) - 1)) / 7 + 2 )) :=
begin
  sorry,
end

end sum_of_first_n_terms_l257_257705


namespace find_angle_B_find_max_length_AD_l257_257343

/-- Part (1): Lean 4 Statement -/
theorem find_angle_B
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : a = 2 * √3)
    (h2 : b = 2)
    (h3 : ∀ {x}, x ≠ A → sin x = x)
    (h4 : (sin A + sin B) / sin C = (c + b) / (a - b))
    : B = π / 6 := by
  sorry

/-- Part (2): Lean 4 Statement -/
theorem find_max_length_AD
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : a = 2 * √3)
    (h2 : b = 2)
    (area_ABC : ℝ)
    (AD : ℝ)
    (h3 : area_ABC = √3)
    : AD = 1 := by
  sorry

end find_angle_B_find_max_length_AD_l257_257343


namespace Beto_goal_achievable_l257_257065

theorem Beto_goal_achievable :
  ∀ (x y z : Fin 2019 → ℤ),
    (∀ i, x i = i + 1) →
    (∃ (σ : Equiv.Perm (Fin 2019)), ∀ i, y i = x (σ i)) →
    (∀ i, z i = abs (x i - y i)) →
    (∃ A B : Finset (Fin 2019), disjoint A B ∧ A ∪ B = Finset.univ ∧
      (∑ i in A, z i) = (∑ i in B, z i)) :=
by
  sorry

end Beto_goal_achievable_l257_257065


namespace domain_of_function_l257_257708

noncomputable def is_defined (x : ℝ) : Prop :=
  (x + 4 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_function :
  ∀ x : ℝ, is_defined x ↔ x ≥ -4 ∧ x ≠ 0 :=
by
  sorry

end domain_of_function_l257_257708


namespace projection_onto_plane_l257_257727

open Matrix Vec

def n : Vec ℝ 3 := ⟨2, 1, -2⟩

def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![5/9, -2/9, 4/9],
    ![-2/9, 8/9, 2/9],
    ![4/9, 2/9, 5/9]
  ]

theorem projection_onto_plane (u : Vec ℝ 3) :
  ((Q ⬝ u) : Three) = proj u :=
sorry

end projection_onto_plane_l257_257727


namespace min_value_of_F_on_neg_infinity_l257_257135

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions provided in the problem
axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom g_odd : ∀ x : ℝ, g (-x) = - g x
noncomputable def F (x : ℝ) := a * f x + b * g x + 2
axiom F_max_on_pos : ∃ x ∈ (Set.Ioi 0), F x = 5

-- Prove the conclusion of the problem
theorem min_value_of_F_on_neg_infinity : ∃ y ∈ (Set.Iio 0), F y = -1 :=
sorry

end min_value_of_F_on_neg_infinity_l257_257135


namespace max_rectangle_area_l257_257632

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l257_257632


namespace closest_vector_t_l257_257967

noncomputable def vector_v (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 6 * t, 1 - 4 * t, -4 + 2 * t)

def vector_a : ℝ × ℝ × ℝ := (-1, 6, 3)

theorem closest_vector_t : 
  let t := (-15 : ℝ) / 28 in
  let v_minus_a := (4 + 6 * t, -5 - 4 * t, -7 + 2 * t) in
  let direction := (6, -4, 2) in
  (v_minus_a.1 * direction.1 + v_minus_a.2 * direction.2 + v_minus_a.3 * direction.3 = 0) := 
begin
  sorry
end

end closest_vector_t_l257_257967


namespace driver_schedule_l257_257041

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l257_257041


namespace exact_four_red_probability_l257_257359

noncomputable def probability_exactly_four_red (total_draws red_marbles total_marbles : ℕ) (p_red p_blue : ℚ) : ℚ :=
  let p_red_four_times := p_red ^ 4
  let p_blue_four_times := p_blue ^ 4
  let specific_sequence_prob := p_red_four_times * p_blue_four_times
  let ways_to_choose_four_red := nat.choose total_draws 4
  let total_probability := ways_to_choose_four_red * specific_sequence_prob
  total_probability

theorem exact_four_red_probability :
  probability_exactly_four_red 8 8 12 (2/3) (1/3) = (1120/6561) :=
by
  sorry

end exact_four_red_probability_l257_257359


namespace relationship_between_m_n_p_l257_257227

-- Define the conditions
def m : ℝ := (1/2)⁻²
def n : ℝ := (-2) ^ 3
def p : ℝ := -((-1/2) ^ 0)

-- State the theorem
theorem relationship_between_m_n_p : n < p ∧ p < m := by
  sorry

end relationship_between_m_n_p_l257_257227


namespace loss_per_metre_l257_257898

theorem loss_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (cost_price_per_m: ℕ)
  (selling_price_total : selling_price = 18000)
  (cost_price_per_m_def : cost_price_per_m = 95)
  (total_metres_def : total_metres = 200) :
  ((cost_price_per_m * total_metres - selling_price) / total_metres) = 5 :=
by
  sorry

end loss_per_metre_l257_257898


namespace eight_sided_dice_probability_l257_257542

noncomputable def probability_even_eq_odd (n : ℕ) : ℚ :=
  if h : n % 2 = 0 then
    let k := n / 2 in
    (nat.choose n k : ℚ) * (1/2)^n
  else 0

theorem eight_sided_dice_probability :
  probability_even_eq_odd 8 = 35/128 :=
by trivial

end eight_sided_dice_probability_l257_257542


namespace original_price_of_a_mango_l257_257540

theorem original_price_of_a_mango
  (price_increase : ∀ (x : ℝ), 1.15 * x)  
  (orange_price : ℝ := 40)
  (total_cost : 10 * (1.15 * orange_price) + 10 * (price_increase ?mango_price) = 1035) :
  price_increase ?mango_price = 57.5 :=
by
  sorry

end original_price_of_a_mango_l257_257540


namespace probability_of_two_red_shoes_l257_257129

theorem probability_of_two_red_shoes (total_shoes red_shoes green_shoes : ℕ) 
  (h_total : total_shoes = 10)
  (h_red : red_shoes = 7)
  (h_green : green_shoes = 3) :
  let C := λ n k : ℕ, nat.choose n k in
  (C red_shoes 2 : ℚ) / (C total_shoes 2) = 7 / 15 := 
by
  let C := λ n k : ℕ, nat.choose n k
  rw [h_total, h_red, h_green]
  sorry

end probability_of_two_red_shoes_l257_257129


namespace max_area_of_rectangular_pen_l257_257608

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l257_257608


namespace ratio_of_areas_is_one_fourth_l257_257149

def side_length_large_square : ℝ := 4

def side_length_inscribed_square : ℝ :=
  let point1 := side_length_large_square / 4
  let point2 := 3 * (side_length_large_square / 4)
  point2 - point1

def area_ratio : ℝ :=
  let area_large_square := side_length_large_square ^ 2
  let area_inscribed_square := side_length_inscribed_square ^ 2
  area_inscribed_square / area_large_square

theorem ratio_of_areas_is_one_fourth :
  area_ratio = 1 / 4 := by
  sorry

end ratio_of_areas_is_one_fourth_l257_257149


namespace problem1_problem2_l257_257998

-- Define the sets A, B, and C
def A := {x : ℝ | x^2 - 3 * x + 2 ≤ 0 }
def B (a : ℝ) := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x + a }
def C (a : ℝ) := {x : ℝ | x^2 - a * x - 4 ≤ 0 }

-- Define the propositions p and q
def p (a : ℝ) := ¬ (A ∩ B a).nonempty
def q (a : ℝ) := A ⊆ C a

-- Problem 1: Prove that if p a is false, then a > 3
theorem problem1 (a : ℝ) (hp_false : ¬ p a) : a > 3 := sorry

-- Problem 2: Prove that if p a and q a are true, then 0 ≤ a ≤ 3
theorem problem2 (a : ℝ) (hp : ¬ p a) (hq : q a) : 0 ≤ a ∧ a ≤ 3 := sorry

end problem1_problem2_l257_257998


namespace rectangular_box_length_l257_257171

theorem rectangular_box_length (W L H : ℕ) (max_boxes : ℕ) (volumes_equal : W * L * H = 336) (max_equal : max_boxes = 1000000) : 
  real := 
let box_length := (336 / 1000000) ^ (1/3) in
box_length ≈ 6.93

end rectangular_box_length_l257_257171


namespace real_solutions_l257_257209

theorem real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6)) = 1 / 12) ↔ (x = 12 ∨ x = -4) :=
by
  sorry

end real_solutions_l257_257209


namespace negation_of_existential_l257_257426

theorem negation_of_existential (P : Prop) :
  (¬ (∃ x : ℝ, x ^ 3 > 0)) ↔ (∀ x : ℝ, x ^ 3 ≤ 0) :=
by
  sorry

end negation_of_existential_l257_257426


namespace solve_for_x_l257_257105

theorem solve_for_x : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end solve_for_x_l257_257105


namespace countUltimateMountainNumbers_l257_257823

-- Define the conditions for being an ultimate mountain number.
def isUltimateMountain (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  (1000 ≤ n ∧ n < 10000) ∧ (d3 > d2 ∧ d3 > d4)

-- State the theorem to be proved.
theorem countUltimateMountainNumbers : 
  (Finset.filter isUltimateMountain (Finset.range 10000)).card = 204 :=
by
  sorry

end countUltimateMountainNumbers_l257_257823


namespace find_x_l257_257781

theorem find_x :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 (x : ℕ) := (10 + 60 + x) / 3
  ∃ x : ℕ, avg2 x = avg1 - 5 ∧ x = 35 :=
by
  let avg1 := (20 + 40 + 60) / 3
  let avg2 (x : ℕ) := (10 + 60 + x) / 3
  use 35
  have H1 : avg1 = 40 := by
    calc (20 + 40 + 60) / 3 = 120 / 3 : by norm_num
      ... = 40 : by norm_num
  have H2 : avg2 35 = 35 := by
    calc (10 + 60 + 35) / 3 = 105 / 3 : by norm_num
      ... = 35 : by norm_num
  refine ⟨H2, rfl⟩
  sorry

end find_x_l257_257781


namespace young_fish_per_pregnant_fish_l257_257743

-- Definitions based on conditions
def tanks := 3
def fish_per_tank := 4
def total_young_fish := 240

-- Calculations based on conditions
def total_pregnant_fish := tanks * fish_per_tank

-- The proof statement
theorem young_fish_per_pregnant_fish : total_young_fish / total_pregnant_fish = 20 := by
  sorry

end young_fish_per_pregnant_fish_l257_257743


namespace max_product_l257_257100

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l257_257100


namespace range_of_x_l257_257551

theorem range_of_x (x : ℝ) (h1 : 2 ≤ |x - 5|) (h2 : |x - 5| ≤ 10) (h3 : 0 < x) : 
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := 
sorry

end range_of_x_l257_257551


namespace paid_amount_divisible_by_11_l257_257699

-- Define the original bill amount and the increased bill amount
def original_bill (x : ℕ) : ℕ := x
def paid_amount (x : ℕ) : ℕ := (11 * x) / 10

-- Theorem: The paid amount is divisible by 11
theorem paid_amount_divisible_by_11 (x : ℕ) (h : x % 10 = 0) : paid_amount x % 11 = 0 :=
by
  sorry

end paid_amount_divisible_by_11_l257_257699


namespace initial_ratio_proof_l257_257137

variable (p q : ℕ) -- Define p and q as non-negative integers

-- Condition: The initial total volume of the mixture is 30 liters
def initial_volume (p q : ℕ) : Prop := p + q = 30

-- Condition: Adding 12 liters of q changes the ratio to 3:4
def new_ratio (p q : ℕ) : Prop := p * 4 = (q + 12) * 3

-- The final goal: prove the initial ratio is 3:2
def initial_ratio (p q : ℕ) : Prop := p * 2 = q * 3

-- The main proof problem statement
theorem initial_ratio_proof (p q : ℕ) 
  (h1 : initial_volume p q) 
  (h2 : new_ratio p q) : initial_ratio p q :=
  sorry

end initial_ratio_proof_l257_257137


namespace sufficient_drivers_and_ivan_petrovich_departure_l257_257035

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l257_257035


namespace drivers_sufficiency_and_ivan_petrovich_departure_l257_257031

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l257_257031


namespace general_formula_l257_257664

open Nat

def a : ℕ → ℚ
| 0     := 0
| (n+1) := if n = 0 then -1 else a n + 1 / ((n) * (n+1))

theorem general_formula (n : ℕ) (h : n > 0) :
  a n = - (1 / n) :=
sorry

end general_formula_l257_257664


namespace tray_height_l257_257502

/--
A square piece of paper has sides of length 120 units. From each corner, a wedge is cut such that
each of the two cuts starts √20 units from the corner and meets on the diagonal at an angle of 45°.
The paper is then folded up along the lines joining the vertices of adjacent cuts and taped together
to form a tray. Prove that the height of the tray is 
h = ( (sqrt 20) + (sqrt 40) ) * ( sqrt (2 - sqrt 2) ) / 2.
-/
theorem tray_height
  (side_length : ℕ) (cut_distance : ℕ) (cut_angle : ℕ)
  (H1 : side_length = 120) 
  (H2 : cut_distance = (Real.sqrt 20 : ℝ))
  (H3 : cut_angle = 45) :
  let h := ( (Real.sqrt 20) + (Real.sqrt 40) ) * ( Real.sqrt (2 - Real.sqrt 2) ) / 2 in
  true :=
sorry

end tray_height_l257_257502


namespace work_rate_b_l257_257873

theorem work_rate_b (A C B : ℝ) (hA : A = 1 / 8) (hC : C = 1 / 24) (h_combined : A + B + C = 1 / 4) : B = 1 / 12 :=
by
  -- Proof goes here
  sorry

end work_rate_b_l257_257873


namespace regression_equation_example_l257_257652

noncomputable def regression_equation (x y : ℝ) (neg_corr : Prop) (mean_x : ℝ) (mean_y : ℝ) : Prop :=
  (neg_corr ∧ mean_x = 3 ∧ mean_y = 2.7) → ∃ b₀ b₁, (∀ x, (b₀, b₁) = (-0.2, 3.3)) ∧ ∀ y, y = b₀ + b₁ * x

theorem regression_equation_example : regression_equation x y (x ∧ y < 0) 3 2.7 :=
by
  sorry

end regression_equation_example_l257_257652


namespace find_number_l257_257871

theorem find_number (x : ℝ) (h : 5020 - (1004 / x) = 4970) : x = 20.08 := 
by
  sorry

end find_number_l257_257871


namespace complement_of_domain_l257_257651

open Set Real

def f (x : ℝ) : ℝ := sqrt (1 - x^2)

theorem complement_of_domain : (compl (Icc (-1 : ℝ) 1)) = Iio (-1) ∪ Ioi 1 := by
  sorry

end complement_of_domain_l257_257651


namespace relationship_between_m_n_p_l257_257228

-- Define the conditions
def m : ℝ := (1/2)⁻²
def n : ℝ := (-2) ^ 3
def p : ℝ := -((-1/2) ^ 0)

-- State the theorem
theorem relationship_between_m_n_p : n < p ∧ p < m := by
  sorry

end relationship_between_m_n_p_l257_257228


namespace trig_identity_l257_257189

open Real

theorem trig_identity :
  (1 - 1 / cos (23 * π / 180)) *
  (1 + 1 / sin (67 * π / 180)) *
  (1 - 1 / sin (23 * π / 180)) * 
  (1 + 1 / cos (67 * π / 180)) = 1 :=
by
  sorry

end trig_identity_l257_257189


namespace parabola_min_value_l257_257264

theorem parabola_min_value (x : ℝ) : (∃ x, x^2 + 10 * x + 21 = -4) := sorry

end parabola_min_value_l257_257264


namespace solid_of_revolution_minimum_volume_l257_257322

noncomputable def minimum_volume (a : ℝ) : ℝ :=
  let b := -a - 2
  let α := (-a - real.sqrt (a ^ 2 + 4 * a + 8)) / 2
  let β := (-a + real.sqrt (a ^ 2 + 4 * a + 8)) / 2
  let Δ := β - α
  π * (Δ ^ 5) / 30

theorem solid_of_revolution_minimum_volume
  (a : ℝ)
  (h : (1:ℝ) ^ 2 + a * (1:ℝ) - (a + 2) = -1) :
  minimum_volume a = (16 * π) / 15 :=
  sorry

end solid_of_revolution_minimum_volume_l257_257322


namespace minimum_value_of_f_l257_257796

def f (x : ℝ) : ℝ := x + 4 / x

theorem minimum_value_of_f (x : ℝ) (hx : x > 0) : f x ≥ 4 := by
  sorry

end minimum_value_of_f_l257_257796


namespace two_results_l257_257024

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l257_257024


namespace common_internal_tangents_single_line_l257_257751

noncomputable theory

-- Define points P, Q, and circles Γ₁ and Γ₂ corresponding to the inscribed circles of triangles PAB and PCD
variables (P Q : Point)
variables (Γ₁ Γ₂ : Circle)
variables (triangle_PAB triangle_PCD : Triangle)
variables [Inscribed Γ₁ triangle_PAB] [Inscribed Γ₂ triangle_PCD]

-- The circles Γ₁ and Γ₂ intersect at point Q
axiom circles_intersect_at_Q : Γ₁ ∩ Γ₂ = {Q}

-- Prove the common internal tangents to Γ₁ and Γ₂ are a single straight line
theorem common_internal_tangents_single_line :
  ∃ T : Point, CommonInternalTangent Γ₁ Γ₂ T ∧
    Collinear {P, Q, T} := 
sorry

end common_internal_tangents_single_line_l257_257751


namespace units_digit_6_pow_6_l257_257830

theorem units_digit_6_pow_6 : (6 ^ 6) % 10 = 6 := 
by {
  sorry
}

end units_digit_6_pow_6_l257_257830


namespace transportation_cost_function_minimum_cost_speed_l257_257418

-- Definitions based on the conditions
def distance : ℝ := 500
def max_speed : ℝ := 100
def variable_coefficient : ℝ := 0.01
def fixed_cost : ℝ := 100
noncomputable def time (v : ℝ) := distance / v
noncomputable def total_transportation_cost (v : ℝ) := fixed_cost * time(v) + variable_coefficient * v^2 * time(v)

-- Statement for the first part
theorem transportation_cost_function (v : ℝ) (h : 0 < v ∧ v ≤ max_speed) : 
    total_transportation_cost v = (50000 / v) + 5 * v := 
sorry

-- Statement for the second part
theorem minimum_cost_speed : 
    argmin (λ v : ℝ, total_transportation_cost v) (Icc 0 max_speed) = 100 := 
sorry

end transportation_cost_function_minimum_cost_speed_l257_257418


namespace rita_needs_9_months_l257_257696

def total_required_hours : ℕ := 4000
def backstroke_hours : ℕ := 100
def breaststroke_hours : ℕ := 40
def butterfly_hours : ℕ := 320
def monthly_practice_hours : ℕ := 400

def hours_already_completed : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_required_hours - hours_already_completed
def months_needed : ℕ := (remaining_hours + monthly_practice_hours - 1) / monthly_practice_hours -- Ceiling division

theorem rita_needs_9_months :
  months_needed = 9 := by
  sorry

end rita_needs_9_months_l257_257696


namespace exists_100_digit_number_divisible_by_2_pow_100_l257_257760

theorem exists_100_digit_number_divisible_by_2_pow_100 :
  ∃ (n : ℕ), nat.digits 10 n = list.replicate 100 1 ∨ list.replicate 100 2 →
  n % 2^100 = 0 :=
sorry

end exists_100_digit_number_divisible_by_2_pow_100_l257_257760


namespace coefficient_x2_in_binomial_expansion_l257_257785

-- Definition to calculate the binomial coefficient
def binom (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

-- Proposition: Coefficient of x^2 in (2 - x)^5 is 80
theorem coefficient_x2_in_binomial_expansion :
  (∃ (c : ℤ), ∀ x : ℤ, (2 - x)^5 = (2:ℤ)^5 * (1 - x/2)^5 → c = 80) :=
sorry

end coefficient_x2_in_binomial_expansion_l257_257785


namespace max_area_of_rectangular_pen_l257_257607

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l257_257607


namespace relationship_m_n_p_l257_257225

noncomputable def m : ℝ := (1 / 2) ^ (-2)
noncomputable def n : ℝ := (-2) ^ 3
noncomputable def p : ℝ := -(-(1 / 2) ^ 0)

theorem relationship_m_n_p : n < p ∧ p < m :=
by {
  have hm : m = 4 := by sorry,
  have hn : n = -8 := by sorry,
  have hp : p = 1 := by sorry,
  rw [hm, hn, hp],
  exact ⟨by norm_num, by norm_num⟩
}

end relationship_m_n_p_l257_257225


namespace carolyn_correct_sum_l257_257774

-- Define the initial conditions
def n : ℕ := 10

def initial_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Constraint: Carolyn cannot remove a prime number on her first turn
def is_prime (n : ℕ) : Bool := n > 1 ∧ (n __pairwise_coprime [2..n-1])
def can_remove (is_prime_on_first_turn : Bool) (turn : ℕ) (num : ℕ) : Prop :=
if turn = 1 then ¬is_prime num else True

-- Carolyn's first move
def initial_carolyn_removal : ℕ := 4

-- List after first removal
def list_after_first_removal : List ℕ := [1, 2, 3, 5, 6, 7, 8, 9, 10]

-- Define the final sum of numbers Carolyn removes
def carolyn_sum : ℕ := 4 + 8 + 6

-- Proof problem statement
theorem carolyn_correct_sum : carolyn_sum = 18 := by
  sorry

end carolyn_correct_sum_l257_257774


namespace sequence_decreasing_range_of_an_a_10_not_in_interval_a_50_in_interval_l257_257992

noncomputable def a_seq : ℕ → ℝ
| 0       := 1
| (n+1) := a_seq n / (a_seq n^2 + 1)

theorem sequence_decreasing : ∀ n : ℕ, a_seq (n + 1) < a_seq n :=
by sorry

theorem range_of_an : ∀ n : ℕ, 0 < a_seq n ∧ a_seq n ≤ 1 :=
by sorry

theorem a_10_not_in_interval : ¬ (1 / 8 < a_seq 9 ∧ a_seq 9 < 1 / 7) :=
by sorry

theorem a_50_in_interval : 1 / 11 < a_seq 49 ∧ a_seq 49 < 1 / 10 :=
by sorry

end sequence_decreasing_range_of_an_a_10_not_in_interval_a_50_in_interval_l257_257992


namespace sum_of_divisors_of_6_l257_257966

def is_integer (x : ℚ) : Prop := ∃ z : ℤ, x = z

theorem sum_of_divisors_of_6 :
  let S := {n : ℕ | n > 0 ∧ is_integer ((n + 6) / n)} in 
  ∑ n in S, n = 12 :=
by 
  sorry

end sum_of_divisors_of_6_l257_257966


namespace prove_an_prove_bn_l257_257241

noncomputable def geometric_seq (a₁ a₄ : ℕ) (a : ℕ → ℕ) :=
  ∃ q : ℕ, a₁ = 3 ∧ a 4 = 24 ∧ ∀ n > 0, a n = a₁ * q^(n-1)

noncomputable def sum_seq (a b : ℕ → ℕ) :=
  b 1 = 0 ∧ ∀ n > 0, b n + b (n + 1) = a n

theorem prove_an (a : ℕ → ℕ) 
  (h1 : geometric_seq 3 24 a) :
  ∀ n, a n = 3 * 2^(n - 1) :=
sorry

theorem prove_bn (a b : ℕ → ℕ)
  (h_a : ∀ n, a n = 3 * 2^(n - 1))
  (h_b : sum_seq a b) :
  ∀ n, b n = 2^(n-1) + (-1)^n :=
sorry

end prove_an_prove_bn_l257_257241


namespace carla_receives_correct_amount_l257_257366

theorem carla_receives_correct_amount (L B C X : ℝ) : 
  (L + B + C + X) / 3 - (C + X) = (L + B - 2 * C - 2 * X) / 3 :=
by
  sorry

end carla_receives_correct_amount_l257_257366


namespace songs_per_album_l257_257456

theorem songs_per_album (C P : ℕ) (h1 : 4 * C + 5 * P = 72) (h2 : C = P) : C = 8 :=
by
  sorry

end songs_per_album_l257_257456


namespace exists_divisible_number_l257_257400

theorem exists_divisible_number : ∃ n : ℕ, 
  (∃ k m : ℕ, n = (foldr (λ _ x, x * 10 + 2021) 0 (range k)) * 10 ^ m) ∧
  2022 ∣ n := by
  sorry

end exists_divisible_number_l257_257400


namespace axis_of_symmetry_of_translated_cosine_l257_257681

theorem axis_of_symmetry_of_translated_cosine :
  ∀ k : ℤ, ∀ x : ℝ, y = cos (2 * x) → x + π / 12 → x = (k * π) / 2 - π / 12 where k ∈ ℤ :=
by 
  sorry

end axis_of_symmetry_of_translated_cosine_l257_257681


namespace trapezoid_area_sum_l257_257503

theorem trapezoid_area_sum :
  ∃ (r_1 r_2 r_3 : ℚ) (n_1 n_2 : ℕ),
  (∀ p : ℕ, ¬ (p^2 ∣ n_1 ∧ p^2 ∣ n_2)) ∧ 
  let S := r_1 * real.sqrt n_1 + r_2 * real.sqrt n_2 + r_3 in
  (S = 24 * real.sqrt 2) ∧ 
  ⌊r_1 + r_2 + r_3 + n_1 + n_2⌋ = 26 :=
sorry

end trapezoid_area_sum_l257_257503


namespace max_volume_right_tetrahedron_l257_257991

theorem max_volume_right_tetrahedron (PA PB PC : ℝ) (h1 : 0 ≤ PA) (h2 : 0 ≤ PB) (h3 : 0 ≤ PC) (h_angle : (PA^2 + PB^2 = PC^2)) (S : ℝ) (hS : S = PA + PB + PC + real.sqrt (PA^2 + PB^2) + real.sqrt (PB^2 + PC^2) + real.sqrt (PC^2 + PA^2)) :
  ∃ (V : ℝ), V ≤ (1/162) * (5 * real.sqrt 2 - 7) * S^3 :=
begin
  sorry
end

end max_volume_right_tetrahedron_l257_257991


namespace combined_value_is_correct_l257_257310

noncomputable def a : ℕ :=
  let p := 0.95
  in (p / 0.005).toNat

noncomputable def b : ℕ := 3 * a - 50

noncomputable def c : ℕ := (a - b) * (a - b)

theorem combined_value_is_correct : a + b + c = 109610 :=
by
  have h_a : a = 190 := by
    norm_num [a]
  have h_b : b = 520 := by
    rw [h_a]
    norm_num [b]
  have h_c : c = 108900 := by
    rw [h_a, h_b]
    norm_num [c]
  rw [h_a, h_b, h_c]
  norm_num

end combined_value_is_correct_l257_257310


namespace sum_of_squares_l257_257779

theorem sum_of_squares (x y z : ℝ)
  (h1 : (x + y + z) / 3 = 10)
  (h2 : (xyz)^(1/3) = 6)
  (h3 : 3 / ((1/x) + (1/y) + (1/z)) = 4) : 
  x^2 + y^2 + z^2 = 576 := 
by
  sorry

end sum_of_squares_l257_257779


namespace drivers_sufficiency_and_ivan_petrovich_departure_l257_257032

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l257_257032


namespace probability_R_S_l257_257753

/-- Given points P, Q, R, and S on a line segment PQ such that PQ = 4PR = 8QR,
    this theorem proves that the probability of selecting a point on PQ
    that lies between R and S is 5/8. -/
theorem probability_R_S (P Q R S : ℝ)
  (h1 : P < Q)
  (h2 : P < R ∧ R < Q)
  (h3 : P < S ∧ S < Q)
  (h4 : PQ = 4 * PR)
  (h5 : PQ = 8 * QR) :
  let RS := Q - (P + R + Q) in
  let prob := RS / (Q - P) in
  prob = 5 / 8 := 
sorry

end probability_R_S_l257_257753


namespace range_m_l257_257574

variable {x m : ℝ}

theorem range_m (h1 : m / (1 - x) - 2 / (x - 1) = 1) (h2 : x ≥ 0) (h3 : x ≠ 1) : m ≤ -1 ∧ m ≠ -2 := 
sorry

end range_m_l257_257574


namespace exists_symmetric_points_l257_257265

theorem exists_symmetric_points (m : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ e^(-x) - real.log (x + m) - 2 = 0) ↔ m < 1 / real.exp 1 := 
by
  sorry

end exists_symmetric_points_l257_257265


namespace second_class_tickets_needed_l257_257061

theorem second_class_tickets_needed :
  let total_stations := 17 in
  ∑ i in Finset.range total_stations, ∑ j in Finset.range i.succ, 1 = 68 :=
by
  sorry

end second_class_tickets_needed_l257_257061


namespace max_product_of_two_integers_l257_257089

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l257_257089


namespace mary_age_l257_257124

theorem mary_age :
  ∃ M R : ℕ, (R = M + 30) ∧ (R + 20 = 2 * (M + 20)) ∧ (M = 10) :=
by
  sorry

end mary_age_l257_257124


namespace natasha_quarters_l257_257392

theorem natasha_quarters (n : ℕ) :
  20 < n ∧ n < 200 ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 6 = 1) → 
  n = 61 ∨ n = 121 ∨ n = 181 := 
by 
  intro h, 
  sorry

end natasha_quarters_l257_257392


namespace complex_expression_evaluation_l257_257192

-- Conditions
def i : ℂ := Complex.I -- Representing the imaginary unit i

-- Defining the inverse of a complex number
noncomputable def complex_inv (z : ℂ) := 1 / z

-- Proof statement
theorem complex_expression_evaluation :
  (i - complex_inv i + 3)⁻¹ = (3 - 2 * i) / 13 := by
sorry

end complex_expression_evaluation_l257_257192


namespace cistern_wet_surface_area_l257_257484

-- Function that describes the bottom of the cistern
def f (x y : ℝ) : ℝ := 1 + 0.1 * Math.sin x * Math.cos y

-- Dimensions of the rectangular region
def length := 9
def width := 6

-- Water depth in the cistern
def depth := 2.25

-- Proof that the total area of the wet surface is 121.5 m²
theorem cistern_wet_surface_area : 
  let A_bottom := (∫ x in 0..length, ∫ y in 0..width, (1:ℝ)) in
  let A_long_sides := 2 * length * depth in
  let A_short_sides := 2 * width * depth in
  A_bottom + A_long_sides + A_short_sides = 121.5 := by
  sorry

end cistern_wet_surface_area_l257_257484


namespace solution_l257_257071

-- Define the discount conditions
def discount (price : ℕ) : ℕ :=
  if price > 22 then price * 7 / 10 else
  if price < 20 then price * 8 / 10 else
  price

-- Define the given book prices
def book_prices : List ℕ := [25, 18, 21, 35, 12, 10]

-- Calculate total cost using the discount function
def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (λ acc price => acc + discount price) 0

def problem_statement : Prop :=
  total_cost book_prices = 95

theorem solution : problem_statement :=
  by
  unfold problem_statement
  unfold total_cost
  simp [book_prices, discount]
  sorry

end solution_l257_257071


namespace g_at_neg_1001_l257_257775

-- Defining the function g and the conditions
def g (x : ℝ) : ℝ := 2.5 * x - 0.5

-- Defining the main theorem to be proved
theorem g_at_neg_1001 : g (-1001) = -2503 := by
  sorry

end g_at_neg_1001_l257_257775


namespace three_distinct_divisors_l257_257453

theorem three_distinct_divisors (M : ℕ) : (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ M ∧ b ∣ M ∧ c ∣ M ∧ (∀ d, d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬ d ∣ M)) ↔ (∃ p : ℕ, Prime p ∧ M = p^2) := 
by sorry

end three_distinct_divisors_l257_257453


namespace count_integer_values_of_x_l257_257300

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l257_257300


namespace max_days_for_process_C_l257_257496

/- 
  A project consists of four processes: A, B, C, and D, which require 2, 5, x, and 4 days to complete, respectively.
  The following conditions are given:
  - A and B can start at the same time.
  - C can start after A is completed.
  - D can start after both B and C are completed.
  - The total duration of the project is 9 days.
  We need to prove that the maximum number of days required to complete process C is 3.
-/
theorem max_days_for_process_C
  (A B C D : ℕ)
  (hA : A = 2)
  (hB : B = 5)
  (hD : D = 4)
  (total_duration : ℕ)
  (h_total : total_duration = 9)
  (h_condition1 : A + C + D = total_duration) : 
  C = 3 :=
by
  rw [hA, hD, h_total] at h_condition1
  linarith

#check max_days_for_process_C

end max_days_for_process_C_l257_257496


namespace sin_half_angle_l257_257677

variable {θ : ℝ}
variable h1 : abs (Real.cos θ) = 1 / 5
variable h2 : (5 / 2) * Real.pi < θ ∧ θ < 3 * Real.pi

theorem sin_half_angle :
  Real.sin (θ / 2) = - (Real.sqrt 15 / 5) :=
by
  sorry

end sin_half_angle_l257_257677


namespace no_such_function_exists_l257_257119

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = f (n + 1) - f n :=
by
  sorry

end no_such_function_exists_l257_257119


namespace original_number_of_people_l257_257435

-- Defining the conditions
variable (n : ℕ) -- number of people originally
variable (total_cost : ℕ := 375)
variable (equal_cost_split : n > 0 ∧ total_cost = 375) -- total cost is $375 and n > 0
variable (cost_condition : 375 / n + 50 = 375 / 5)

-- The proof statement
theorem original_number_of_people (h1 : total_cost = 375) (h2 : 375 / n + 50 = 375 / 5) : n = 15 :=
by
  sorry

end original_number_of_people_l257_257435


namespace calc_num_int_values_l257_257295

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l257_257295


namespace two_results_l257_257023

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l257_257023


namespace find_complex_number_l257_257196

open Complex

theorem find_complex_number (z : ℂ) 
  (h : 3 * z - 4 * I * conj(z) = -4 + 5 * I) :
  z = (-32 / 25) - (1 / 25) * I := 
sorry

end find_complex_number_l257_257196


namespace min_value_achieved_at_x_equal_4_l257_257104

noncomputable def f (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem min_value_achieved_at_x_equal_4 : ∃ x : ℝ, x = 4 ∧ ∀ y : ℝ, f(4) ≤ f(y) := by
  sorry

end min_value_achieved_at_x_equal_4_l257_257104


namespace necessary_not_sufficient_l257_257239

def A (x : ℝ) : Prop := (x^2 + x) / (x - 1) ≥ 0
def B (x : ℝ) : Prop := Real.log 3 (2 * x + 1) ≤ 0

theorem necessary_not_sufficient : 
    (∀ x, B x → A x) ∧ (∃ x, B x ∧ ¬ A x) :=
by
  sorry


end necessary_not_sufficient_l257_257239


namespace max_product_l257_257098

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l257_257098


namespace circle_diameter_l257_257417

def area := 132.73228961416876
def pi := Real.pi

theorem circle_diameter (A : ℝ) (h : A = area) : ∃ d : ℝ, d = 2 * Real.sqrt (A / pi) ∧ d ≈ 12.998044 :=
by {
  sorry
}

end circle_diameter_l257_257417


namespace lazy_worker_days_worked_l257_257884

theorem lazy_worker_days_worked :
  ∃ x : ℕ, 24 * x - 6 * (30 - x) = 0 ∧ x = 6 :=
by
  existsi 6
  sorry

end lazy_worker_days_worked_l257_257884


namespace complex_quadrant_l257_257588

theorem complex_quadrant
  (a b : ℝ)
  (h_eq : a + complex.I = (b + complex.I) * (2 - complex.I)) :
  b > 0 ∧ a > 0 := 
by {
  -- Applying given conditions and solving for a and b
  let H_real := complex.ext_iff.1 h_eq).1,
  have H_b : b = 1,
  { linarith, },
  subst H_b,

  have H_a : a = 3,
  { linarith, },
  
  -- Values imply coordinates (3, 1) which are in the first quadrant
  exact ⟨by linarith, by linarith⟩,
}

end complex_quadrant_l257_257588


namespace number_of_elements_in_A_that_satisfy_condition_l257_257665

def A : set (fin 5 → ℤ) :=
  { v | (∀ i, v i ∈ { -1, 0, 1 }) }

def condition (v : fin 5 → ℤ) : Prop :=
  1 ≤ ∑ i, abs (v i) ∧ ∑ i, abs (v i) ≤ 3

theorem number_of_elements_in_A_that_satisfy_condition :
  (finset.card (finset.filter condition (A.to_finset))) = 130 := 
sorry

end number_of_elements_in_A_that_satisfy_condition_l257_257665


namespace rectangle_width_l257_257498

theorem rectangle_width
  (l w : ℕ)
  (h1 : l * w = 1638)
  (h2 : 10 * l = 390) :
  w = 42 :=
by
  sorry

end rectangle_width_l257_257498


namespace frog_reaches_edge_probability_l257_257972

-- Definition of the grid and the frog's movement

noncomputable def probability_reaching_edge_within_six_hops : ℚ := 211/256

theorem frog_reaches_edge_probability :
  let
    -- Define the size of the grid
    grid_size := 4,
    
    -- Define the initial starting positions (one of the central positions)
    central_positions := {(2, 2), (2, 3), (3, 2), (3, 3)},
    
    -- Define edge positions
    edge_positions := { (x, y) | x = 1 ∨ x = grid_size ∨ y = 1 ∨ y = grid_size },
    
    -- Define probability function
    p : ℕ → (ℕ × ℕ) → ℚ
  in
    -- Probability of reaching an edge within 6 hops from any central position
    (∑ pos in central_positions, p 6 pos) / 4 = probability_reaching_edge_within_six_hops :=
sorry

end frog_reaches_edge_probability_l257_257972


namespace smallest_N_gt_100_l257_257880

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k+1 => if (∃ m, k = m * (m + 1) / 2 - 1) then 
             2^((nat.find (nat.find_spec (λ m, k = m * (m + 1) / 2 - 1)))) 
           else 1

def sum_of_first_n_terms (N : ℕ) : ℕ :=
  (list.sum (list.map (sequence) (list.range N)))

theorem smallest_N_gt_100 : ∃ N, N > 100 ∧ sum_of_first_n_terms N = 2^k for some k :=
  ∃ N, N = 440 sorry

end smallest_N_gt_100_l257_257880


namespace max_rectangle_area_l257_257625

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l257_257625


namespace trapezoid_area_l257_257826

/-- Define the equations of the lines that bound the trapezoid -/
def line1 (x : ℝ) : ℝ := 2 * x
def line2 : ℝ := 12
def line3 : ℝ := 6

/-- Define the vertices of the trapezoid -/
def vertex1 : ℝ × ℝ := (3, 6)
def vertex2 : ℝ × ℝ := (6, 12)
def vertex3 : ℝ × ℝ := (0, 12)
def vertex4 : ℝ × ℝ := (0, 6)

/-- Define the lengths of the bases and the height of the trapezoid -/
def base1_length : ℝ := 3
def base2_length : ℝ := 6
def height : ℝ := 6

/-- Calculate the area of the trapezoid -/
def area_of_trapezoid : ℝ := 0.5 * (base1_length + base2_length) * height

/-- State the theorem to be proved -/
theorem trapezoid_area :
  area_of_trapezoid = 27.0 :=
  by
    /- Proof placeholder -/
    sorry

end trapezoid_area_l257_257826


namespace part1_part2_l257_257701

noncomputable def point_on_y_axis (m : ℝ) : Prop :=
  m - 1 = 0

noncomputable def line_parallel_to_y_axis (m : ℝ) : Prop :=
  m - 1 = -3

noncomputable def coordinates (m : ℝ) : (ℝ × ℝ) :=
  (m - 1, 2 * m + 3)

theorem part1 (m : ℝ) (h : point_on_y_axis m) : m = 1 :=
by {
  exact h
}

theorem part2 (m : ℝ) (h : line_parallel_to_y_axis m) : 
  let M := coordinates m,
      N := (-3, 2 : ℝ),
      distance := abs (N.2 - (2 * m + 3))
  in distance = 3 :=
by {
  cases M,
  simp only [coordinates] at h,
  have hm : m = -2,
  { exact h, },
  simp only [hm, coordinates, abs_sub] at *,
  norm_num
}

end part1_part2_l257_257701


namespace count_four_digit_numbers_with_digit_sum_eq_27_l257_257282

theorem count_four_digit_numbers_with_digit_sum_eq_27 : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ (n.digits 10).sum = 27}.card = 20 :=
sorry

end count_four_digit_numbers_with_digit_sum_eq_27_l257_257282


namespace gcf_45_135_90_l257_257082

theorem gcf_45_135_90 : Nat.gcd (Nat.gcd 45 135) 90 = 45 := 
by
  sorry

end gcf_45_135_90_l257_257082


namespace hyperbola_A_asymptote_l257_257511

-- Define the hyperbola and asymptote conditions
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def asymptote_eq (y x : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Statement of the proof problem in Lean 4
theorem hyperbola_A_asymptote :
  ∀ (x y : ℝ), hyperbola_A x y → asymptote_eq y x :=
sorry

end hyperbola_A_asymptote_l257_257511


namespace eval_f_neg_5_l257_257738

def f (x : ℝ) : ℝ :=
if x < 3 then 3 * x - 4 else x + 6

theorem eval_f_neg_5 : f (-5) = -19 :=
by
  sorry

end eval_f_neg_5_l257_257738


namespace find_x_l257_257477

theorem find_x (x : ℝ) (h : 15 * x + 16 * x + 19 * x + 11 = 161) : x = 3 :=
sorry

end find_x_l257_257477


namespace largest_stamps_per_page_max_largest_stamps_per_page_l257_257357

theorem largest_stamps_per_page (n : ℕ) :
  (840 % n = 0) ∧ (1008 % n = 0) ∧ (672 % n = 0) → n ≤ 168 :=
by sorry

theorem max_largest_stamps_per_page :
  ∃ n, (840 % n = 0) ∧ (1008 % n = 0) ∧ (672 % n = 0) ∧ n = 168 :=
by {
  use 168,
  split,
  { calc 840 % 168 = 0 : by sorry },
  split,
  { calc 1008 % 168 = 0 : by sorry },
  { calc 672 % 168 = 0 : by sorry },
  exact eq.refl 168
}

end largest_stamps_per_page_max_largest_stamps_per_page_l257_257357


namespace leftover_value_is_zero_l257_257500

theorem leftover_value_is_zero (michael_quarters sarah_quarters michael_nickels sarah_nickels roll_quarters roll_nickels: ℕ) :
  michael_quarters = 75 →
  sarah_quarters = 85 →
  michael_nickels = 123 →
  sarah_nickels = 157 →
  roll_quarters = 40 →
  roll_nickels = 40 →
  let total_quarters := michael_quarters + sarah_quarters in
  let total_nickels := michael_nickels + sarah_nickels in
  let leftover_quarters := total_quarters % roll_quarters in
  let leftover_nickels := total_nickels % roll_nickels in
  let value_leftover_quarters := leftover_quarters * 0.25 in
  let value_leftover_nickels := leftover_nickels * 0.05 in
  value_leftover_quarters + value_leftover_nickels = 0.00 :=
by {
  intro h_mq h_sq h_mn h_sn h_rq h_rn,
  let total_quarters := 75 + 85,
  let total_nickels := 123 + 157,
  have h_tq: total_quarters = 160 := rfl,
  have h_tn: total_nickels = 280 := rfl,
  let leftover_quarters := 160 % 40,
  let leftover_nickels := 280 % 40,
  have h_lq: leftover_quarters = 0 := by norm_num,
  have h_ln: leftover_nickels = 0 := by norm_num,
  let value_leftover_quarters := 0 * 0.25,
  let value_leftover_nickels := 0 * 0.05,
  have h_vlq: value_leftover_quarters = 0.00 := by norm_num,
  have h_vln: value_leftover_nickels = 0.00 := by norm_num,
  show 0.00 + 0.00 = 0.00,
  norm_num
} sorry

end leftover_value_is_zero_l257_257500


namespace sum_of_fourth_powers_l257_257587

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := 
by 
  sorry

end sum_of_fourth_powers_l257_257587


namespace quadratic_roots_quadratic_roots_one_quadratic_roots_two_l257_257747

open scoped Classical

variables {p : Type*} [Field p] {a b c x : p}

theorem quadratic_roots (h_a : a ≠ 0) :
  (¬ ∃ y : p, y^2 = b^2 - 4 * a * c) → ∀ x : p, ¬ a * x^2 + b * x + c = 0 :=
by sorry

theorem quadratic_roots_one (h_a : a ≠ 0) :
  (b^2 - 4 * a * c = 0) → ∃ x : p, a * x^2 + b * x + c = 0 ∧ ∀ y : p, a * y^2 + b * y + c = 0 → y = x :=
by sorry

theorem quadratic_roots_two (h_a : a ≠ 0) :
  (∃ y : p, y^2 = b^2 - 4 * a * c) ∧ (b^2 - 4 * a * c ≠ 0) → ∃ x1 x2 : p, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by sorry

end quadratic_roots_quadratic_roots_one_quadratic_roots_two_l257_257747


namespace beads_per_necklace_l257_257718

-- Definitions based on conditions
def total_beads_used (N : ℕ) : ℕ :=
  10 * N + 2 * N + 50 + 35

-- Main theorem to prove the number of beads needed for one beaded necklace
theorem beads_per_necklace (N : ℕ) (h : total_beads_used N = 325) : N = 20 :=
by
  sorry

end beads_per_necklace_l257_257718


namespace problem1_l257_257232

open Real

theorem problem1
  (a b : ℝ)
  (h₀ : ∀ x, (differentiable_on ℝ (λ x, 1 / 3 * x^3 + a * x + b) (set.of (λ x, x = 2))))
  (h₁ : (1 / 3 * (2:ℝ)^3 + a * (2) + b) = -4 / 3)
  (h₂ : ∀ x ∈ set.Icc (-4:ℝ) (3), (1 / 3 * x^3 + a * x + b) ≤ (λ m, m^2 + m + 10 / 3) m) :
  (a = -4 ∧ b = 4 ∧ (m ∈ set.Icc (-∞) (-3) ∪ set.Icc (2) (∞))) :=
by
  sorry

end problem1_l257_257232


namespace find_A_in_terms_of_B_and_C_l257_257381

noncomputable def f (A B : ℝ) (x : ℝ) := A * x - 3 * B^2
noncomputable def g (B C : ℝ) (x : ℝ) := B * x + C

theorem find_A_in_terms_of_B_and_C (A B C : ℝ) (h : B ≠ 0) (h1 : f A B (g B C 1) = 0) : A = 3 * B^2 / (B + C) :=
by sorry

end find_A_in_terms_of_B_and_C_l257_257381


namespace winning_margin_l257_257066

theorem winning_margin (total_votes : ℝ) (winning_votes : ℝ) (winning_percent : ℝ) (losing_percent : ℝ) 
  (win_votes_eq: winning_votes = winning_percent * total_votes)
  (perc_eq: winning_percent + losing_percent = 1)
  (win_votes_given: winning_votes = 550)
  (winning_percent_given: winning_percent = 0.55)
  (losing_percent_given: losing_percent = 0.45) :
  winning_votes - (losing_percent * total_votes) = 100 := 
by
  sorry

end winning_margin_l257_257066


namespace solution_set_inequality_l257_257641

noncomputable theory
open Real

variable {f : ℝ → ℝ}

-- Defining the conditions of the problem
axiom f_derivative_condition : ∀ x : ℝ, deriv f x > -1
axiom f_initial_condition : f 0 = -2

-- The Lean 4 statement (no proof provided, just the statement)
theorem solution_set_inequality : {x : ℝ | f x + 2 * exp x + x < 0} = Iio 0 := by
  sorry

end solution_set_inequality_l257_257641


namespace count_integer_values_of_x_l257_257291

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : {n : ℕ | 121 < n ∧ n ≤ 144}.card = 23 := 
by
  sorry

end count_integer_values_of_x_l257_257291


namespace path_length_l257_257925

noncomputable def EF : ℝ := 3 / real.pi

def circumference (r : ℝ) : ℝ := 2 * real.pi * r

theorem path_length (EF_eq : EF = 3 / real.pi) :
  let r := EF in
  let C := circumference r in
  C = 6 :=
by
  sorry

end path_length_l257_257925


namespace number_of_proper_subsets_of_A_cap_B_l257_257999

open real set

noncomputable def A : set ℤ := { x | x^2 - 3 * x - 4 ≤ 0 }
noncomputable def B : set ℝ := { x | 0 < log x ∧ log x < 2 }

noncomputable def intersection_AB : set ℤ := { x | x ∈ A ∧ (x : ℝ) ∈ B }

theorem number_of_proper_subsets_of_A_cap_B : 
  (finset.powerset (finset.filter (λ x, x ∈ intersection_AB) (finset.range 5)).erase ∅).card = 7 :=
sorry

end number_of_proper_subsets_of_A_cap_B_l257_257999


namespace max_area_of_fenced_rectangle_l257_257617

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l257_257617


namespace south_walk_correct_representation_l257_257687

theorem south_walk_correct_representation {north south : ℤ} (h_north : north = 3) (h_representation : south = -north) : south = -5 :=
by
  have h1 : -north = -3 := by rw [h_north]
  have h2 : -3 = -5 := by sorry
  rw [h_representation, h1]
  exact h2

end south_walk_correct_representation_l257_257687


namespace max_rectangle_area_l257_257631

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l257_257631


namespace locus_of_points_A_B_C_l257_257395

-- Definitions of points and the geometric relationship
variables {A B C M P Q P' Q' : Point}
variables [lie_on : Line]
variable [homothety : Center]

-- All points A, B, and C lie on line with B between A and C
def points_on_line (A B C : Point) : Prop :=
  (A ∈ lie_on) ∧ (B ∈ lie_on) ∧ (C ∈ lie_on) ∧ is_between B A C

-- Defining the condition for M
def locus_point (M : Point) :=
  ∃ P Q,
    circumradius A M B = circumradius C M B ∧
    homothety_centered_at B 2 (midpoint P Q) = M ∧
    projection_lies_on_line A C P' Q'

/-- Theorem statement: Given points A, B, and C lie on a line with B between A and C,
    find the locus of points M such that the circumradii of triangles AMB and CMB are equal. -/
theorem locus_of_points_A_B_C (A B C M : Point) (h : points_on_line A B C) :
  locus_point M ∧ ¬(M ∈ line A C) :=
sorry

end locus_of_points_A_B_C_l257_257395


namespace betty_afternoon_catch_l257_257215

def flies_eaten_per_day := 2
def days_in_week := 7
def flies_needed_for_week := days_in_week * flies_eaten_per_day
def flies_caught_morning := 5
def additional_flies_needed := 4
def flies_currently_have := flies_needed_for_week - additional_flies_needed
def flies_caught_afternoon := flies_currently_have - flies_caught_morning
def flies_escaped := 1

theorem betty_afternoon_catch :
  flies_caught_afternoon + flies_escaped = 6 :=
by
  sorry

end betty_afternoon_catch_l257_257215


namespace jaco_budget_for_parents_l257_257352

/-- Assume Jaco has 8 friends, each friend's gift costs $9, and Jaco has a total budget of $100.
    Prove that Jaco's budget for each of his mother and father's gift is $14. -/
theorem jaco_budget_for_parents :
  ∀ (friends_count cost_per_friend total_budget : ℕ), 
  friends_count = 8 → 
  cost_per_friend = 9 → 
  total_budget = 100 → 
  (total_budget - friends_count * cost_per_friend) / 2 = 14 :=
by
  intros friends_count cost_per_friend total_budget h1 h2 h3
  rw [h1, h2, h3]
  have friend_total_cost : friends_count * cost_per_friend = 72 := by norm_num
  have remaining_budget : total_budget - friends_count * cost_per_friend = 28 := by norm_num [friend_total_cost]
  have divided_budget : remaining_budget / 2 = 14 := by norm_num [remaining_budget]
  exact divided_budget

end jaco_budget_for_parents_l257_257352


namespace Winnie_keeps_balloons_l257_257457

-- Definitions based on conditions
def red_balloons := 24
def white_balloons := 50
def green_balloons := 72
def chartreuse_balloons := 96
def friends := 12

-- The proof problem statement
theorem Winnie_keeps_balloons : 
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons
  in total_balloons % friends = 2 :=
by
  sorry

end Winnie_keeps_balloons_l257_257457


namespace tangent_line_at_0_1_is_correct_l257_257959

-- Define the function f(x) = x + exp(x)
def f (x : ℝ) : ℝ := x + Real.exp x

-- The tangent line at point (0, 1) has the equation 2x - y + 1 = 0
theorem tangent_line_at_0_1_is_correct : ∃ k b, (∀ x y, y = 2*x + b -> y - 1 = f x - f 0) ∧ b = -1 := 
by
  -- We need to construct the slope k and the y-intercept b
  use 2, -1
  split
  -- Prove the general form of the tangent line equation with f
  {
      intros x y h
      rw [h, f]
      simp
      exact sorry
  }
  -- Prove the y-intercept
  { exact sorry }

end tangent_line_at_0_1_is_correct_l257_257959


namespace max_tiles_on_floor_l257_257462

-- Define the dimensions of the tile and the floor
def tile_length_1 : ℕ := 60
def tile_width_1 : ℕ := 56
def tile_length_2 : ℕ := 56
def tile_width_2 : ℕ := 60
def floor_length : ℕ := 560
def floor_width : ℕ := 240

-- Prove the maximum number of tiles that can be accommodated on the floor
theorem max_tiles_on_floor :
  -- Number of tiles along the length and width for both tile orientations
  let tiles_lengthwise_1 := floor_length / tile_length_1 in
  let tiles_widthwise_1 := floor_width / tile_width_1 in
  let tiles_lengthwise_2 := floor_length / tile_length_2 in
  let tiles_widthwise_2 := floor_width / tile_width_2 in
  -- Total tiles for both orientations
  let total_tiles_orientation_1 := tiles_lengthwise_1 * tiles_widthwise_1 in
  let total_tiles_orientation_2 := tiles_lengthwise_2 * tiles_width_2 in
  -- The maximum number of tiles
  total_tiles_orientation_2 = 40 :=
-- Assuming the conditions above, we want to show that:
by
  sorry

end max_tiles_on_floor_l257_257462


namespace problem_part1_problem_part2_l257_257273

theorem problem_part1 (a b : ℝ × ℝ) (k : ℝ) 
  (h_a_b_not_collinear : ¬ (∃ λ : ℝ, a = λ • b))
  (h_c : (fun x y => (k * x, k * y)) a + b = (fun x y => (x, y)) c)
  (h_d : (fun x y => (x, y)) d = a - b)
  (h_c_parallel_to_d : ∃ λ : ℝ, c = λ • d)
  : k = -1 ∧ c = (-d) :=
sorry

theorem problem_part2 (a b : ℝ × ℝ) (k : ℝ) 
  (h_a_b_not_collinear : ¬ (∃ λ : ℝ, a = λ • b))
  (h_a_b_equal : ∥a∥ = ∥b∥)
  (h_angle_a_b : angle a b = π / 3)
  (h_c : (fun x y => (k * x, k * y)) a + b = (fun x y => (x, y)) c)
  (h_d : (fun x y => (x, y)) d = a - b)
  (h_c_perpendicular_to_d : c ⬝ d = 0)
  : k = 1 :=
sorry

end problem_part1_problem_part2_l257_257273


namespace semicircle_perimeter_approx_l257_257050

noncomputable def semicircle_perimeter (r : ℕ) : ℝ :=
  Real.pi * r + 2 * r

theorem semicircle_perimeter_approx (r : ℕ) (h : r = 7) :
  semicircle_perimeter r ≈ 35.99 :=
by
  rw [←h]
  have pi_approx : Real.pi ≈ 3.14159 := by sorry
  show Real.pi * 7 + 2 * 7 ≈ 35.99
  apply sorry

end semicircle_perimeter_approx_l257_257050


namespace quadrants_containing_points_l257_257802

theorem quadrants_containing_points (x y : ℝ) :
  (y > x + 1) → (y > 3 - 2 * x) → 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end quadrants_containing_points_l257_257802


namespace gratuity_percentage_is_15_l257_257048

noncomputable def total_cost_with_gratuity := 207.00
noncomputable def number_of_people := 15
noncomputable def average_cost_per_person_without_gratuity := 12.00

noncomputable def total_cost_without_gratuity := average_cost_per_person_without_gratuity * number_of_people
noncomputable def gratuity := total_cost_with_gratuity - total_cost_without_gratuity
noncomputable def gratuity_percentage := (gratuity / total_cost_without_gratuity) * 100

theorem gratuity_percentage_is_15 :
  gratuity_percentage = 15 := by
  sorry

end gratuity_percentage_is_15_l257_257048


namespace find_a_value_l257_257791

-- Define the problem conditions
theorem find_a_value (a : ℝ) :
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  (mean_y = 0.95 * mean_x + 2.6) → a = 2.2 :=
by
  -- Let bindings are for convenience to follow the problem statement
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  intro h
  sorry

end find_a_value_l257_257791


namespace magicians_can_perform_trick_l257_257076

-- Conditions: A deck of 100 cards labeled from 1 to 100
-- Three cards selected by spectators
-- Second magician adds one more card
-- The first magician determines the original cards and their order

noncomputable def magician_trick : Prop :=
∀ (deck : Finset ℕ) (a b c d : ℕ),
  deck = Finset.range 100 ∧
  a ∈ deck ∧ b ∈ deck ∧ c ∈ deck ∧ d ∈ deck ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  d ∉ {a, b, c} ∧
  (∃ f : (Fin 4) → Fin 4, bijective f ∧ 
    (∃ h : ℕ → ℕ, 
      h {a, b, c, d} = {f 0, f 1, f 2, f 3}))
  → 
  (∃ g : ℕ → Finset ℕ, 
    g (a, b, c) = d ∧ 
    ∀ σ : equiv.perm (Fin 3), 
      ∀ (a' b' c' d' : ℕ), {a', b', c', d'} = {a, b, c, d} → 
      σ • (a, b, c) = (a', b', c'))

-- Proof that the magicians can perform this trick
theorem magicians_can_perform_trick : magician_trick :=
begin
  sorry -- Proof is skipped
end

end magicians_can_perform_trick_l257_257076


namespace median_unchanged_after_removal_l257_257508

open Set

noncomputable def median (s : Set ℝ) : ℝ := sorry

-- Assume a sufficiently large set of distinct real numbers
variable {S : Set ℝ} (h_distinct : ∀ a b ∈ S, a ≠ b → a ≠ b)
variable (h_large : ∃ a b c d ∈ S, True) -- simplification for sufficiently large set

theorem median_unchanged_after_removal :
  ∀ S' ⊆ S, S'.card = S.card - 2 → median S' = median S :=
begin
  sorry -- proof omitted as per instructions
end

end median_unchanged_after_removal_l257_257508


namespace cover_equilateral_triangles_l257_257446

theorem cover_equilateral_triangles (a b : ℕ) (h₁ : a = 15) (h₂ : b = 1) :
  let A₁ := (Real.sqrt 3) / 4 * b^2 in
  let A₂ := (Real.sqrt 3) / 4 * a^2 in
  A₂ / A₁ = 225 :=
by
  sorry

end cover_equilateral_triangles_l257_257446


namespace rectangle_transformed_to_cylinder_l257_257072

-- Definitions based on the conditions
def length_of_rectangle : ℝ := 6
def width_of_rectangle : ℝ := 3

-- The first possible volume when the length of the rectangle becomes the circumference of the base
def volume_case_1 : ℝ := (π * ( (length_of_rectangle / (2 * π)) ^ 2) * width_of_rectangle)

-- The second possible volume when the width of the rectangle becomes the circumference of the base
def volume_case_2 : ℝ := (π * ( (width_of_rectangle / (2 * π)) ^ 2) * length_of_rectangle)

-- The final theorem statement
theorem rectangle_transformed_to_cylinder : 
  volume_case_1 = 27 / π ∧ volume_case_2 = 27 / (4 * π) := 
by sorry

end rectangle_transformed_to_cylinder_l257_257072


namespace period_of_function_l257_257582

theorem period_of_function {f : ℝ → ℝ} (a : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x : ℝ, f(x + a) = (1 + f(x)) / (1 - f(x))) :
  ∃ T : ℝ, T = 4 * |a| ∧ ∀ x : ℝ, f(x + T) = f(x) :=
by
  have h₂ : ∀ x : ℝ, f(x + 2 * a) = -1 / f(x)
  have h₃ : ∀ x : ℝ, f(x + 4 * a) = f(x)
  use 4 * |a|
  sorry

end period_of_function_l257_257582


namespace find_f_neg2_l257_257260

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * log (real.sqrt (x^2 + 1) + x) - 4

theorem find_f_neg2 (a b : ℝ) (h : f a b 2 = 2) : f a b (-2) = -10 :=
by
  sorry

end find_f_neg2_l257_257260


namespace range_of_inverse_dist_sum_l257_257703

theorem range_of_inverse_dist_sum 
  (t α : ℝ) 
  (P Q A : ℝ × ℝ)
  (C1 : ℝ × ℝ → Prop := λ point, ∃ (θ : ℝ), point = ⟨2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ⟩)
  (C2 : ℝ × ℝ → Prop := λ point, ∃ (t : ℝ), point = ⟨t * Real.cos α, 1 + t * Real.sin α⟩)
  (A_def : A = (0, 1))
  (intersections : C1 P ∧ C2 P ∧ C1 Q ∧ C2 Q) :
  2 < 1 / (Real.dist P A) + 1 / (Real.dist Q A) ∧ 
  1 / (Real.dist P A) + 1 / (Real.dist Q A) ≤ 2 * Real.sqrt 2 :=
sorry

end range_of_inverse_dist_sum_l257_257703


namespace value_of_f2009_f2010_l257_257534

def f (x : ℝ) : ℝ := sorry  -- The specific definition of f is not given.

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom periodic_six : ∀ x : ℝ, f(x + 6) = f(x)
axiom f_one : f(1) = 2010

theorem value_of_f2009_f2010 : f(2009) + f(2010) = -2010 :=
by
  sorry

end value_of_f2009_f2010_l257_257534


namespace incorrect_min_value_of_sqrt_sum_l257_257584

theorem incorrect_min_value_of_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) :
  ∀ x, (x = √a + √b) → x ≠ √2 :=
sorry

end incorrect_min_value_of_sqrt_sum_l257_257584


namespace largest_int_lt_100_remainder_3_div_by_8_l257_257213

theorem largest_int_lt_100_remainder_3_div_by_8 : 
  ∃ n, n < 100 ∧ n % 8 = 3 ∧ ∀ m, m < 100 ∧ m % 8 = 3 → m ≤ 99 := by
  sorry

end largest_int_lt_100_remainder_3_div_by_8_l257_257213


namespace parallel_lines_l257_257682

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = a - 7) → a = 3 :=
by sorry

end parallel_lines_l257_257682


namespace sum_of_first_seven_terms_l257_257234

theorem sum_of_first_seven_terms :
  ∃ (a_1 : ℚ), 
    let a := λ n : ℕ => a_1 * (2 ^ (n - 1))
    in a 7 = 127 * (a 4) ^ 2 → 
       let S_7 := (∑ i in finset.range 7, a (i + 1)) 
       in S_7 = 1 :=
begin
  sorry
end

end sum_of_first_seven_terms_l257_257234


namespace a_alone_time_to_complete_work_l257_257116

theorem a_alone_time_to_complete_work :
  (W : ℝ) →
  (A : ℝ) →
  (B : ℝ) →
  (h1 : A + B = W / 6) →
  (h2 : B = W / 12) →
  A = W / 12 :=
by
  -- Given conditions
  intros W A B h1 h2
  -- Proof is not needed as per instructions
  sorry

end a_alone_time_to_complete_work_l257_257116


namespace correct_conclusions_l257_257790

theorem correct_conclusions (a b c : ℝ) :
  (a + b > 0 ∧ ab > 0 → a > 0 ∧ b > 0) ∧
  (a / b = -1 → a + b = 0) ∧
  (a < b ∧ b < c → |a - b| + |b - c| = |a - c|) ∧
  (-1 < a ∧ a < 0 → a^2 > a ∧ a > a^3 ∧ a^3 > 1 / a) → 
  (True := true) := 
  sorry

end correct_conclusions_l257_257790


namespace f_prime_at_zero_problem_solution_l257_257693

variable {R : Type*} [Field R] [CharZero R]

def geometric_seq (a : ℕ → R) :=
  ∃ (a₁ r : R), ∀ n : ℕ, a n = a₁ * r ^ (n - 1)

theorem f_prime_at_zero (a : ℕ → ℝ) 
  (h2 : a 2 = 1) 
  (h9 : a 9 = 9) 
  (geom_seq : geometric_seq a) : 
  (x : ℝ → ℝ) :=
sorry

theorem problem_solution : f_prime_at_zero = 3 ^ 10 :=
by sorry

end f_prime_at_zero_problem_solution_l257_257693


namespace non_attacking_rooks_8x8_removed_corners_l257_257079

theorem non_attacking_rooks_8x8_removed_corners :
  let rows := Finset.range 8
  let columns := Finset.range 8
  let corners := {(0, 0), (0, 7), (7, 0), (7, 7)}
  let remaining_squares := (rows.product columns).filter (λ rc, ¬ rc ∈ corners)
  let rook_placement := {f // Function.Injective f ∧ ∀ r, (r, f r) ∈ remaining_squares}
  Finset.card rook_placement = 21600 :=
by sorry

end non_attacking_rooks_8x8_removed_corners_l257_257079


namespace distribute_places_l257_257312

open Nat

theorem distribute_places (places schools : ℕ) (h_places : places = 7) (h_schools : schools = 3) : 
  ∃ n : ℕ, n = (Nat.choose (places - 1) (schools - 1)) ∧ n = 15 :=
by
  rw [h_places, h_schools]
  use 15
  , sorry

end distribute_places_l257_257312


namespace both_inequalities_equiv_l257_257174

theorem both_inequalities_equiv (x : ℝ) : (x - 3)/(2 - x) ≥ 0 ↔ (3 - x)/(x - 2) ≥ 0 := by
  sorry

end both_inequalities_equiv_l257_257174


namespace train_speed_l257_257902

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l257_257902


namespace sum_f_1_to_2017_l257_257144

noncomputable def f (x : ℝ) : ℝ :=
  if x % 6 < -1 then -(x % 6 + 2) ^ 2 else x % 6

theorem sum_f_1_to_2017 : (List.sum (List.map f (List.range' 1 2017))) = 337 :=
  sorry

end sum_f_1_to_2017_l257_257144


namespace factorial_floor_computation_l257_257190

theorem factorial_floor_computation :
  ⌊(Nat.factorial 2017 + Nat.factorial 2014) / (Nat.factorial 2016 + Nat.factorial 2015)⌋ = 2016 :=
by 
  sorry

end factorial_floor_computation_l257_257190


namespace problem_value_of_m_l257_257263

theorem problem_value_of_m (m : ℝ)
  (h1 : (m + 1) * x ^ (m ^ 2 - 3) = y)
  (h2 : m ^ 2 - 3 = 1)
  (h3 : m + 1 < 0) : 
  m = -2 := 
  sorry

end problem_value_of_m_l257_257263


namespace yuri_total_puppies_l257_257851

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l257_257851


namespace unique_real_c_value_l257_257573

noncomputable def satisfies_conditions (c : ℝ) : Prop :=
  (complex.abs (2 / 3 - complex.I * c) = 5 / 6) ∧ (c > 0)

theorem unique_real_c_value :
  ∃! (c : ℝ), satisfies_conditions c :=
sorry

end unique_real_c_value_l257_257573


namespace probability_laurent_greater_chloe_l257_257934

open ProbabilityTheory

noncomputable def probability_greater (x y : ℝ) (hx : x ∈ Set.Icc 0 2017) 
  (hy : y ∈ Set.Icc 0 4034) : ℝ := 
OfReal (MeasureTheory.Measure.prod
  (MeasureTheory.Measure.restrict) x hx)
  (MeasureTheory.Measure.restrict (ofReal y hy))
  {p : ℝ × ℝ | p.2 > p.1}

theorem probability_laurent_greater_chloe :
  (probability_greater x y hx hy = (¾ : ℝ)) := 
sorry

end probability_laurent_greater_chloe_l257_257934


namespace distinguishable_colorings_of_cube_l257_257204

def colors := {red, white, blue, green}

def cube_faces : ℕ := 6

noncomputable def num_distinguishable_colorings (c : Finset colors) (f : ℕ) : ℕ :=
  if h : c.card = 4 ∧ f = cube_faces then 52 else 0

theorem distinguishable_colorings_of_cube :
  num_distinguishable_colorings colors cube_faces = 52 :=
 by 
  -- proof goes here
  sorry

end distinguishable_colorings_of_cube_l257_257204


namespace number_of_valid_n_l257_257219

noncomputable def f (n : ℕ) : ℤ := 7 + 4 * n + 3 * n^2 + 5 * n^3 + 4 * n^4 + 3 * n^5

theorem number_of_valid_n : 
  (finset.filter (λ n, f n % 9 = 0) (finset.range 101)).card - (finset.filter (λ n, n < 2) (finset.range 101)).card = 22 :=
by 
  -- Proof not required.
  sorry

end number_of_valid_n_l257_257219


namespace volume_of_solid_P_t_l257_257988

-- Definitions for the conditions
variable {P : Type} -- Parallelepiped type
variable (V_P S_P L_P : ℝ) -- Volume, surface area, and edge lengths sum of P
variable (t : ℝ) (ht : t ≥ 0) -- Non-negative real number

-- The theorem to prove
theorem volume_of_solid_P_t (V_P S_P L_P : ℝ) (t : ℝ) (ht : t ≥ 0) :
  let V_Pt := V_P + S_P * t + (ℚ.pi / 4) * L_P * t ^ 2 + (4 * ℚ.pi / 3) * t ^ 3 in 
  V_Pt = V_P + S_P * t + ( ℚ.pi / 4) * L_P * t ^ 2 + (4 * ℚ.pi / 3) * t ^ 3 :=
sorry

end volume_of_solid_P_t_l257_257988


namespace driver_schedule_l257_257044

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l257_257044


namespace necessary_and_sufficient_conditions_l257_257387

-- Definitions for sets A and B
def U : Set (ℝ × ℝ) := {p | true}

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

-- Given point P(2, 3)
def P : ℝ × ℝ := (2, 3)

-- Complement of B
def B_complement (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n > 0}

-- Intersection of A and complement of B
def A_inter_B_complement (m n : ℝ) : Set (ℝ × ℝ) := A m ∩ B_complement n

-- Theorem stating the necessary and sufficient conditions for P to belong to A ∩ (complement of B)
theorem necessary_and_sufficient_conditions (m n : ℝ) : 
  P ∈ A_inter_B_complement m n ↔ m > -1 ∧ n < 5 :=
sorry

end necessary_and_sufficient_conditions_l257_257387


namespace sqrt_sum_eq_constant_l257_257201

theorem sqrt_sum_eq_constant :
  sqrt (16 - 8 * sqrt 3) + 2 * sqrt (16 + 8 * sqrt 3) = 10 + 6 * sqrt 3 :=
by
  -- Proof omitted.
  sorry

end sqrt_sum_eq_constant_l257_257201


namespace units_digit_lucas_L10_is_4_l257_257778

def lucas : ℕ → ℕ 
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_lucas_L10_is_4 : units_digit (lucas (lucas 10)) = 4 := 
  sorry

end units_digit_lucas_L10_is_4_l257_257778


namespace probability_at_least_one_seafood_mass_less_than_265_l257_257479

noncomputable def normal_pdf (μ σ x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - μ)^2 / (2 * σ^2))

def P_within_range (μ σ : ℝ) : ℝ := 0.9974

def P_below_point (μ σ point : ℝ) : ℝ :=
  let prob_within_3sigma := P_within_range μ σ
  let total_prob := 1
  (total_prob - prob_within_3sigma) / 2

def binomial_p (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  Nat.choose n k * p ^ k * (1 - p) ^ (n - k)

def P_at_least_one (n : ℕ) (p : ℝ) : ℝ := 1 - binomial_p n p 0

theorem probability_at_least_one_seafood_mass_less_than_265 :
  let μ : ℝ := 280
  let σ : ℝ := 5
  let point : ℝ := 265
  let single_prob := P_below_point μ σ point
  let n : ℕ := 10
  single_prob = 0.0013 →
  P_at_least_one n single_prob = 1 - (1 - 0.0013) ^ 10 :=
by
  sorry

end probability_at_least_one_seafood_mass_less_than_265_l257_257479


namespace distances_sum_leq_3r_l257_257722

variables {ABC : Triangle} {H : Point} {BC CA AB : Line}
variables [AcuteAngledTriangle ABC]
variable (orthocenter : H = Orthocenter ABC)
variable (inradius : r = inradius ABC)
variables (d_A d_B d_C : ℝ)
variables [dist_H_BC : Distance H BC = d_A]
          [dist_H_CA : Distance H CA = d_B]
          [dist_H_AB : Distance H AB = d_C]

theorem distances_sum_leq_3r :
  d_A + d_B + d_C ≤ 3 * r :=
by
  sorry

end distances_sum_leq_3r_l257_257722


namespace number_of_colorings_l257_257809

-- Definition of the problem
def pentagonal_pyramid := { sides : Finset (Fin 5) // ∀ (i j : Fin 5), i ≠ j → ((i, j) ∈ sides)}

def valid_coloring (colors: Finset (Fin 5)) (paint: Fin 5 → Fin 5) : Prop :=
  ∀ (i j : Fin 5), (i ≠ j) → ((i, j) ∈ pentagonal_pyramid.sides) → (paint i ≠ paint j)

-- Theorem statement
theorem number_of_colorings : 
  ∀ (colors: Finset (Fin 5)), colors.card = 5 →
  ∃ (count: ℕ), count = 1020 ∧ 
  (∃ (paintings: Fin (1020)), ∀ (paint: Fin 5 → Fin 5), (paint ∈ paintings) → valid_coloring colors paint) :=
sorry

end number_of_colorings_l257_257809


namespace more_numbers_not_expressible_as_cube_plus_even_square_l257_257530

-- Definitions based on conditions
def within_bounds (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 1000000
def is_cube_plus_even_square (n : ℕ) : Prop :=
  ∃ (a b m : ℕ), m % 2 = 0 ∧ n = a^3 + m * b^2

-- Main theorem statement
theorem more_numbers_not_expressible_as_cube_plus_even_square :
  ∃ (N : ℕ), N = 1000000 ∧ ∃ more_than : ℕ, more_than > (N - 400100) ∧ 
  (∃ n, within_bounds n ∧ (¬ is_cube_plus_even_square n)) :=
begin
  sorry
end

end more_numbers_not_expressible_as_cube_plus_even_square_l257_257530


namespace ingrid_income_l257_257716

variable (I : ℝ) -- Ingrid's income
def john_income : ℝ := 58000
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def combined_tax_rate : ℝ := 0.3554

-- John's tax
def john_tax : ℝ := john_tax_rate * john_income

-- Ingrid's tax
def ingrid_tax : ℝ := ingrid_tax_rate * I

-- Combined income
def combined_income : ℝ := john_income + I

-- Combined tax
def combined_tax : ℝ := john_tax + ingrid_tax

theorem ingrid_income : I = 72000 :=
by
  have h1 : 0.3554 * combined_income = combined_tax := by
    calc
      0.3554 * combined_income = 0.3554 * (john_income + I)   : by rw [combined_income]
                           ... = 0.3554 * 58000 + 0.3554 * I  : by ring
                           ... = 20612.8 + 0.3554 * I        : by norm_num
      have h2 : john_tax + ingrid_tax = 17400 + 0.4 * I := by
    calc
      john_tax + ingrid_tax = 0.30 * john_income + 0.40 * I : by rw [john_tax, ingrid_tax]
                     ... = 17400 + 0.40 * I                  : by norm_num
  have h3 : 0.3554 * 58000 + 0.3554 * I = 17400 + 0.4 * I := h1.trans h2.symm
  solve_by_elim
-- Proof is left as an exercise for the reader.

end ingrid_income_l257_257716


namespace difference_between_relations_l257_257827

-- Definitions based on conditions
def functional_relationship 
  (f : α → β) (x : α) (y : β) : Prop :=
  f x = y

def correlation_relationship (X Y : Type) : Prop :=
  ∃ (X_rand : X → ℝ) (Y_rand : Y → ℝ), 
    ∀ (x : X), ∃ (y : Y), X_rand x ≠ Y_rand y

-- Theorem stating the problem
theorem difference_between_relations :
  (∀ (f : α → β) (x : α) (y : β), functional_relationship f x y) ∧ 
  (∀ (X Y : Type), correlation_relationship X Y) :=
sorry

end difference_between_relations_l257_257827


namespace milk_after_operations_l257_257141

theorem milk_after_operations :
  let initial : ℝ := 40
  let take_out : ℝ := 4
  let t1_milk : ℝ := initial - take_out
  let t1_total : ℝ := initial -- after adding water, total volume remains the same
  let t2_proportion : ℝ := t1_milk / t1_total
  let t2_milk_taken : ℝ := take_out * t2_proportion
  let t2_milk : ℝ := t1_milk - t2_milk_taken
  let t2_total : ℝ := initial -- after adding water, total volume remains the same
  let t3_proportion : ℝ := t2_milk / t2_total
  let t3_milk_taken : ℝ := take_out * t3_proportion
  let final_milk : ℝ := t2_milk - t3_milk_taken
  final_milk = 29.16 :=
by
  have t1_milk_eq : t1_milk = 36 := rfl
  have t2_proportion_eq : t2_proportion = 0.9 := rfl
  have t2_milk_taken_eq : t2_milk_taken = 3.6 := rfl
  have t2_milk_eq : t2_milk = 32.4 := rfl
  have t3_proportion_eq : t3_proportion = 0.81 := rfl
  have t3_milk_taken_eq : t3_milk_taken = 3.24 := rfl
  have final_milk_eq : final_milk = 29.16 := rfl
  exact final_milk_eq

end milk_after_operations_l257_257141


namespace tan_beta_value_l257_257287

variable {α β : Real}

theorem tan_beta_value (h1 : (sin α * cos α) / (cos (2 * α) + 1) = 1)
                       (h2 : tan (α - β) = 3) :
  tan β = -1 / 7 :=
by
  sorry

end tan_beta_value_l257_257287


namespace max_area_of_rectangular_pen_l257_257599

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l257_257599


namespace find_number_l257_257454

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 10) : x = 5 :=
by
  sorry

end find_number_l257_257454


namespace train_speed_60_kmph_l257_257912

theorem train_speed_60_kmph (length_train : ℕ) (time_to_cross : ℕ) 
  (h_length : length_train = 200) 
  (h_time : time_to_cross = 12) : 
  let distance_km := (length_train : ℝ) / 1000
  let time_hr := (time_to_cross : ℝ) / 3600
  let speed_kmph := distance_km / time_hr
  speed_kmph = 60 := 
by 
  rw [h_length, h_time]
  simp [distance_km, time_hr, speed_kmph]
  norm_num
  sorry

end train_speed_60_kmph_l257_257912


namespace hyperbola_eccentricity_l257_257008

theorem hyperbola_eccentricity 
  (p1 p2 : ℝ × ℝ)
  (asymptote_passes_through_p1 : p1 = (1, 2))
  (hyperbola_passes_through_p2 : p2 = (2 * Real.sqrt 2, 4)) :
  ∃ e : ℝ, e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l257_257008


namespace batsman_average_19th_inning_l257_257855

theorem batsman_average_19th_inning (initial_avg : ℝ) 
    (scored_19th_inning : ℝ) 
    (new_avg : ℝ) 
    (h1 : scored_19th_inning = 100) 
    (h2 : new_avg = initial_avg + 2)
    (h3 : new_avg = (18 * initial_avg + 100) / 19) :
    new_avg = 64 :=
by
  have h4 : initial_avg = 62 := by
    sorry
  sorry

end batsman_average_19th_inning_l257_257855


namespace polar_to_cartesian_parabola_l257_257011

theorem polar_to_cartesian_parabola (ρ θ : ℝ) :
  (ρ * sin (θ)^2 - 2 * cos (θ) = 0) →
  ∃ x y : ℝ, (y^2 = 2 * x) ∧ x = ρ * cos (θ) ∧ y = ρ * sin (θ) :=
by 
  sorry

end polar_to_cartesian_parabola_l257_257011


namespace train_speed_60_kmph_l257_257913

theorem train_speed_60_kmph (length_train : ℕ) (time_to_cross : ℕ) 
  (h_length : length_train = 200) 
  (h_time : time_to_cross = 12) : 
  let distance_km := (length_train : ℝ) / 1000
  let time_hr := (time_to_cross : ℝ) / 3600
  let speed_kmph := distance_km / time_hr
  speed_kmph = 60 := 
by 
  rw [h_length, h_time]
  simp [distance_km, time_hr, speed_kmph]
  norm_num
  sorry

end train_speed_60_kmph_l257_257913


namespace magnitude_of_sum_is_root_10_l257_257595

open Real

noncomputable def magnitude_add {x y : ℝ} (a b c : ℝ × ℝ) (h_a : a = (x, 1)) (h_b : b = (1, y)) (h_c : c = (2, -4))
  (h_dot : x * 2 + 1 * -4 = 0) (h_parallel : ∃ k : ℝ, b = k • c) : ℝ :=
  (a + b).norm

theorem magnitude_of_sum_is_root_10 (x y : ℝ)
  (a b c : ℝ × ℝ)
  (h_a : a = (x, 1))
  (h_b : b = (1, y))
  (h_c : c = (2, -4))
  (h_dot : x * 2 + 1 * -4 = 0)
  (h_parallel : ∃ k : ℝ, b = k • c) :
  magnitude_add a b c h_a h_b h_c h_dot h_parallel = sqrt 10 :=
begin
  sorry
end

end magnitude_of_sum_is_root_10_l257_257595


namespace problem_1_problem_2_l257_257668

-- Definitions of sets and intervals
def U : Set ℝ := Set.univ
def A (m : ℝ) : Set ℝ := { x : ℝ | m - 1 < x ∧ x < 2 * m + 1 }
def B : Set ℝ := { x : ℝ | (x - 7) / (x - 2) < 0 }
def U_compl_B : Set ℝ := { x : ℝ | x ≤ 2 ∨ x ≥ 7 }

-- Problem 1 Statement
theorem problem_1 (m : ℝ) (h : m = 2) : 
  A m ∩ U_compl_B = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
by
    rw h
    sorry

-- Problem 2 Statement
theorem problem_2 : 
  A m ∪ B = B → ( m ≤ -2 ∨ m = 3 ) :=
by
    sorry

end problem_1_problem_2_l257_257668


namespace correct_geometry_problems_l257_257900

-- Let A_c be the number of correct algebra problems.
-- Let A_i be the number of incorrect algebra problems.
-- Let G_c be the number of correct geometry problems.
-- Let G_i be the number of incorrect geometry problems.

def algebra_correct_incorrect_ratio (A_c A_i : ℕ) : Prop :=
  A_c * 2 = A_i * 3

def geometry_correct_incorrect_ratio (G_c G_i : ℕ) : Prop :=
  G_c * 1 = G_i * 4

def total_algebra_problems (A_c A_i : ℕ) : Prop :=
  A_c + A_i = 25

def total_geometry_problems (G_c G_i : ℕ) : Prop :=
  G_c + G_i = 35

def total_problems (A_c A_i G_c G_i : ℕ) : Prop :=
  A_c + A_i + G_c + G_i = 60

theorem correct_geometry_problems (A_c A_i G_c G_i : ℕ) :
  algebra_correct_incorrect_ratio A_c A_i →
  geometry_correct_incorrect_ratio G_c G_i →
  total_algebra_problems A_c A_i →
  total_geometry_problems G_c G_i →
  total_problems A_c A_i G_c G_i →
  G_c = 28 :=
sorry

end correct_geometry_problems_l257_257900


namespace binomial_coefficient_middle_term_l257_257415

theorem binomial_coefficient_middle_term (n : ℕ) (x : ℝ) : 
  n = 10 → 
  (1 - x)^n = ∑ i in range (n + 1), (binom n i) * (-x)^i → 
  ∃ k, k = (n + 1) / 2 ∧ binom n k = binom 10 5 :=
by sorry

end binomial_coefficient_middle_term_l257_257415


namespace sequence_nth_term_less_than_5n_l257_257080

def a : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then 10 * (n / 2) + (n % 5) else 10 * ((n - 1) / 2) + (n % 5) + 5

theorem sequence_nth_term_less_than_5n (n : ℕ) : a n < 5 * n :=
begin
  sorry
end

end sequence_nth_term_less_than_5n_l257_257080


namespace pyramid_water_volume_ratio_l257_257163

noncomputable def volume_of_square_pyramid (s h : ℝ) : ℝ := (1 / 3) * s^2 * h
noncomputable def volume_of_similar_pyramid (s h : ℝ) : ℝ := 
  ((1 / 3) * (1 / 2) * s^2 * (2 / 3) * h)

theorem pyramid_water_volume_ratio (s h : ℝ) (H : s > 0 ∧ h > 0) :
  let V := volume_of_square_pyramid s h
  let V_w := volume_of_similar_pyramid s h
  V_w / V = 1 / 3 ∧ (V_w / V : ℝ).toDecimalString = "0.3333" :=
by {
  intros V V_w,
  unfold volume_of_square_pyramid volume_of_similar_pyramid,
  sorry
}

end pyramid_water_volume_ratio_l257_257163


namespace part1_part2_l257_257271

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def set_B : Set ℝ := {x : ℝ | x < -1 ∨ x > 1}

theorem part1 (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a > 3) :=
by sorry

theorem part2 (a : ℝ) : (set_A a ∪ set_B = Set.univ) ↔ (-2 ≤ a ∧ a ≤ -1 / 2) :=
by sorry

end part1_part2_l257_257271


namespace hyperbola_proof_l257_257660

noncomputable def hyperbola_equation (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : Prop :=
  ∃ c : ℝ, c = 5 ∧ a^2 + b^2 = 25 ∧ b = 4 / 3 * a

theorem hyperbola_proof : hyperbola_equation 3 4 (by norm_num) (by norm_num) :=
  by {
    use (5 : ℝ),
    constructor,
    { norm_num },
    constructor,
    { norm_num },
    { linarith }
  }

end hyperbola_proof_l257_257660


namespace sum_of_first_twelve_multiples_of_nine_l257_257450

theorem sum_of_first_twelve_multiples_of_nine : 
  (∑ i in finset.range 12, 9 * (i + 1)) = 702 := by
sorry

end sum_of_first_twelve_multiples_of_nine_l257_257450


namespace hyperbola_equation_k_range_l257_257987
open Real

theorem hyperbola_equation :
  -- Conditions
  (center : ℝ × ℝ) (F : ℝ × ℝ) (d : ℝ) 
  (center = (0, 0)) (F = (2, 0)) (d = 1) :
  -- Conclusion
  equation (C : ℝ × ℝ → Prop) (C = λ p, p.1 ^ 2 / 3 - p.2 ^ 2 = 1) :=
sorry

theorem k_range :
  -- Conditions
  (C : ℝ × ℝ → Prop) (C = λ p, p.1 ^ 2 / 3 - p.2 ^ 2 = 1)
  (line : ℝ → ℝ) (line = λ k x, k * x + 2) 
  (OA OB : ℝ × ℝ)
  (dot_product : ℝ)
  (dot_product = OA.1 * OB.1 + OA.2 * OB.2)
  (dot_product > 2) :
  -- Conclusion
  set_like.has_mem ℝ {k | k^2 > 1/3 ∧ k^2 < 5/3} k :=
sorry

end hyperbola_equation_k_range_l257_257987


namespace AZ_perp_BC_l257_257344

open EuclideanGeometry

-- Define the problem with given conditions
variables {A B C M N K X Y Z : Point}
variables (Γ_B Γ_C : Semicircle)

-- Midpoints of the sides
hypothesis hM : midpoint B C M
hypothesis hN : midpoint C A N
hypothesis hK : midpoint A B K

-- Semicircles with diameters AC and AB
hypothesis hΓ_B : diameter Γ_B = AC ∧ lies_outside_triangle Γ_B ABC
hypothesis hΓ_C : diameter Γ_C = AB ∧ lies_outside_triangle Γ_C ABC

-- Intersections with the semicircles
hypothesis h_intersect_X : ∃ X, segment_intersect_with_semicircle MK Γ_C X
hypothesis h_intersect_Y : ∃ Y, segment_intersect_with_semicircle MN Γ_B Y

-- Tangents intersect at point Z
hypothesis h_tangent_Z : ∃ Z, tangent_at_point Γ_C X ∧ tangent_at_point Γ_B Y ∧ intersect_at Z

-- Proving the perpendicularity AZ ⊥ BC
theorem AZ_perp_BC : ∃ Z, perp (line_through A Z) (line_through B C) :=
sorry

end AZ_perp_BC_l257_257344


namespace count_distinct_triangles_up_to_similarity_l257_257279

theorem count_distinct_triangles_up_to_similarity :
  ∃ (A B C : ℕ) (k : ℕ), (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ 
  (kC ≤ 360) ∧ 
  (cos A * cos B + sin A * sin B * sin (k * C) = 1) ∧ 
  (number_of_even_divisors 90 = 6) :=
sorry

end count_distinct_triangles_up_to_similarity_l257_257279


namespace ninth_graders_science_only_l257_257018

theorem ninth_graders_science_only 
    (total_students : ℕ := 120)
    (science_students : ℕ := 80)
    (programming_students : ℕ := 75) 
    : (science_students - (science_students + programming_students - total_students)) = 45 :=
by
  sorry

end ninth_graders_science_only_l257_257018


namespace root_product_of_two_real_roots_is_a_root_of_sextic_l257_257397

theorem root_product_of_two_real_roots_is_a_root_of_sextic :
  ∀ (a b c d : ℝ), 
  (polynomial.eval a (polynomial.X^4 + polynomial.X^3 - 1) = 0) →
  (polynomial.eval b (polynomial.X^4 + polynomial.X^3 - 1) = 0) →
  (polynomial.eval c (polynomial.X^4 + polynomial.X^3 - 1) = 0) →
  (polynomial.eval d (polynomial.X^4 + polynomial.X^3 - 1) = 0) →
  (a + b + c + d = -1) →
  (a * b + a * c + a * d + b * c + b * d + c * d = 0) →
  (a * b * c + a * b * d + a * c * d + b * c * d = 0) →
  (a * b * c * d = -1) →
  (polynomial.eval (a * b) (polynomial.X^6 + polynomial.X^4 + polynomial.X^3 - polynomial.X^2 - 1) = 0) :=
by
  intros a b c d root_a root_b root_c root_d sum_0 prod_0 prod_1 prod_2
  sorry

end root_product_of_two_real_roots_is_a_root_of_sextic_l257_257397


namespace part1_part2_l257_257373

noncomputable def increasing_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d > 0, ∀ n : ℕ, a (n + 1) = a n + d

theorem part1 (a : ℕ → ℝ) (h : increasing_arithmetic_seq a) (k l : ℕ)
  (hk : k ≥ 2) (hl : l > k) :
  (a (l+1) / a (k+1)) < (a l / a k) ∧ (a l / a k) < (a (l-1) / a (k-1)) :=
sorry

theorem part2 (a : ℕ → ℝ) (h : increasing_arithmetic_seq a) (k : ℕ) (hk : k ≥ 2) :
  Real.sqrt (a (2013 * k + 1) / a (k + 1)^(1/k)) <
    (List.product $ List.range 2012 |>.map (λ i, a ((i + 1) * k + 1 + 1) / a ((i + 1) * k + 1)))
  ∧ (List.product $ List.range 2012 |>.map (λ i, a ((i + 1) * k + 1 + 1) / a ((i + 1) * k + 1))) <
    Real.sqrt (a (2012 * k + 2) / a 2) :=
sorry

end part1_part2_l257_257373


namespace last_student_initial_position_196_l257_257437

theorem last_student_initial_position_196 :
  let students : List ℕ := List.range (196 + 1) in
  (students.filter (fun n => n % 2 = 0)).length == 1 →
  students.head! = 128 :=
by
  sorry

end last_student_initial_position_196_l257_257437


namespace hunting_dog_catches_fox_l257_257147

theorem hunting_dog_catches_fox :
  ∀ (V_1 V_2 : ℝ) (t : ℝ),
  V_1 / V_2 = 10 ∧ 
  t * V_2 = (10 / (V_2) + t) →
  (V_1 * t) = 100 / 9 :=
by
  intros V_1 V_2 t h
  sorry

end hunting_dog_catches_fox_l257_257147


namespace sum_g_h_l257_257741

-- Let a, b, c, d, e, f, g, and h be real numbers.
variables {a b c d e f g h : ℝ}

-- Conditions given in the problem
def avg_abc : Prop := (a + b + c) / 3 = 103 / 3
def avg_def : Prop := (d + e + f) / 6 = 375 / 6
def overall_avg : Prop := (a + b + c + d + e + f + g + h) / 8 = 23 / 2

-- Theorem to prove
theorem sum_g_h (h_abc : avg_abc) (h_def : avg_def) (h_overall : overall_avg) :
  g + h = -198.5 :=
sorry

end sum_g_h_l257_257741


namespace calculate_value_l257_257183

theorem calculate_value : (245^2 - 225^2) / 20 = 470 :=
by
  sorry

end calculate_value_l257_257183


namespace range_of_a_l257_257658

noncomputable def has_two_zero_points_in_interval (f : ℝ → ℝ) (a : ℝ) :=
  0 < a ∧ a < 2 + Real.log 2 ∧
  ∃ x1 x2 : ℝ, x1 ∈ Icc (-2 : ℝ) (0 : ℝ) ∧ x2 ∈ Icc (0 : ℝ) 2 ∧ f x1 = 0 ∧ f x2 = 0

theorem range_of_a {a : ℝ} :
  has_two_zero_points_in_interval
    (λ x, if x ≤ 0 then x^2 - a else x - a + Real.log x) a ↔ 0 < a ∧ a < 2 + Real.log 2 :=
by
  sorry

end range_of_a_l257_257658


namespace black_cards_taken_out_l257_257409

theorem black_cards_taken_out (total_black_cards remaining_black_cards : ℕ)
  (h1 : total_black_cards = 26) (h2 : remaining_black_cards = 21) :
  total_black_cards - remaining_black_cards = 5 :=
by
  sorry

end black_cards_taken_out_l257_257409


namespace shorter_piece_length_l257_257920

theorem shorter_piece_length : ∃ (x : ℕ), (x + (x + 2) = 30) ∧ x = 14 :=
by {
  sorry
}

end shorter_piece_length_l257_257920


namespace M_inter_N_l257_257667

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {-1, 0}

theorem M_inter_N :
  M ∩ N = {0} :=
by
  sorry

end M_inter_N_l257_257667


namespace missing_angles_sum_l257_257328

theorem missing_angles_sum 
  (calculated_sum : ℕ) 
  (missed_angles_sum : ℕ)
  (total_corrections : ℕ)
  (polygon_angles : ℕ) 
  (h1 : calculated_sum = 2797) 
  (h2 : total_corrections = 2880) 
  (h3 : polygon_angles = total_corrections - calculated_sum) : 
  polygon_angles = 83 := by
  sorry

end missing_angles_sum_l257_257328


namespace problem_statement_l257_257109

def has_arithmetic_square_root (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x

theorem problem_statement :
  (¬ has_arithmetic_square_root (-abs 9)) ∧
  (has_arithmetic_square_root ((-1/4)^2)) ∧
  (has_arithmetic_square_root 0) ∧
  (has_arithmetic_square_root (10^2)) := 
sorry

end problem_statement_l257_257109


namespace no_function_satisfies_condition_l257_257954

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℤ → ℤ, ∀ x y : ℤ, f (x + f y) = f x - y :=
sorry

end no_function_satisfies_condition_l257_257954


namespace max_product_of_sum_2024_l257_257096

theorem max_product_of_sum_2024 : 
  ∃ (x y : ℤ), x + y = 2024 ∧ x * y = 1024144 :=
by
  use 1012, 1012
  split
  · sorry
  · sorry

end max_product_of_sum_2024_l257_257096


namespace solve_quadratic_eqn_l257_257431

theorem solve_quadratic_eqn :
  ∃ x₁ x₂ : ℝ, (x - 6) * (x + 2) = 0 ↔ (x = x₁ ∨ x = x₂) ∧ x₁ = 6 ∧ x₂ = -2 :=
by
  sorry

end solve_quadratic_eqn_l257_257431


namespace max_area_of_rectangular_pen_l257_257602

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l257_257602


namespace price_of_cheaper_book_l257_257063

theorem price_of_cheaper_book
    (total_cost : ℕ)
    (sets : ℕ)
    (price_more_expensive_book_increase : ℕ)
    (h1 : total_cost = 21000)
    (h2 : sets = 3)
    (h3 : price_more_expensive_book_increase = 300) :
  ∃ x : ℕ, 3 * ((x + (x + price_more_expensive_book_increase))) = total_cost ∧ x = 3350 :=
by
  sorry

end price_of_cheaper_book_l257_257063


namespace isosceles_triangle_angles_l257_257795

theorem isosceles_triangle_angles (α β γ : ℝ) 
  (h1 : α = 50)
  (h2 : α + β + γ = 180)
  (isosceles : (α = β ∨ α = γ ∨ β = γ)) :
  (β = 50 ∧ γ = 80) ∨ (γ = 50 ∧ β = 80) :=
by
  sorry

end isosceles_triangle_angles_l257_257795


namespace chord_length_l257_257188

-- Define radii of the circles
def r1 : ℝ := 5
def r2 : ℝ := 12
def r3 : ℝ := r1 + r2

-- Define the centers of the circles
variable (O1 O2 O3 : ℝ)

-- Define the points of tangency and foot of the perpendicular
def T1 : ℝ := O1 + r1
def T2 : ℝ := O2 + r2
def T : ℝ := O3 - r3

-- Given the conditions
theorem chord_length (m n p : ℤ) : 
  (∃ (C1 C2 C3 : ℝ) (tangent1 tangent2 : ℝ),
    C1 = r1 ∧ C2 = r2 ∧ C3 = r3 ∧
    -- Externally tangent: distance between centers of C1 and C2 is r1 + r2
    dist O1 O2 = r1 + r2 ∧
    -- Internally tangent: both C1 and C2 are tangent to C3
    dist O1 O3 = r3 - r1 ∧
    dist O2 O3 = r3 - r2 ∧
    -- The chord in C3 is a common external tangent to C1 and C2
    tangent1 = O3 + ((O1 * O2) - (O1 * O3)) / r1 ∧
    tangent2 = O3 + ((O2 * O1) - (O2 * O3)) / r2 ∧
    m = 10 ∧ n = 546 ∧ p = 7 ∧
    m + n + p = 563)
  := sorry

end chord_length_l257_257188


namespace exists_regular_tetrahedron_on_parallel_planes_l257_257648

theorem exists_regular_tetrahedron_on_parallel_planes (d : ℝ) (h_d : d > 0) :
  ∃ (A B C D : EuclideanSpace ℝ (Fin 3)),
    A.3 = 0 ∧ B.3 = 3 * d ∧ C.3 = d ∧ D.3 = 2 * d ∧
    dist A B = dist A C ∧ dist A C = dist A D ∧ 
    dist A D = dist B C ∧ dist B C = dist B D ∧ 
    dist B D = dist C D :=
by
  sorry

end exists_regular_tetrahedron_on_parallel_planes_l257_257648


namespace exists_nat_numbers_l257_257754

theorem exists_nat_numbers (h1 : ∀ (m n : ℕ), coprime m n → (|a - m| + |b - n| > 1000)) : 
  ∃ (a b : ℕ), ∀ (m n : ℕ), coprime m n → |a - m| + |b - n| > 1000 := 
by {
  sorry
}

end exists_nat_numbers_l257_257754


namespace total_questions_to_review_is_1750_l257_257505

-- Define the relevant conditions
def num_classes := 5
def students_per_class := 35
def questions_per_exam := 10

-- The total number of questions to be reviewed by Professor Oscar
def total_questions : Nat := num_classes * students_per_class * questions_per_exam

-- The theorem stating the equivalent proof problem
theorem total_questions_to_review_is_1750 : total_questions = 1750 := by
  -- proof steps are skipped here 
  sorry

end total_questions_to_review_is_1750_l257_257505


namespace option_B_is_equal_to_a_8_l257_257919

-- Statement: (a^2)^4 equals a^8
theorem option_B_is_equal_to_a_8 (a : ℝ) : (a^2)^4 = a^8 :=
by { sorry }

end option_B_is_equal_to_a_8_l257_257919


namespace sum_first_10_terms_eq_10_l257_257249

-- Definitions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q > 0, ∀ n, a (n + 1) = a n * q
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, a i

-- Conditions
variables (a : ℕ → ℝ)
variable (h_geom : is_geometric_sequence a)
variable (h_a1 : a 1 = 1)
variable (h_a5 : a 5 = 1)

-- Proof statement
theorem sum_first_10_terms_eq_10 : sum_first_n_terms a 10 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end sum_first_10_terms_eq_10_l257_257249


namespace concert_ticket_revenue_l257_257070

theorem concert_ticket_revenue :
  let price_student : ℕ := 9
  let price_non_student : ℕ := 11
  let total_tickets : ℕ := 2000
  let student_tickets : ℕ := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  revenue_student + revenue_non_student = 20960 :=
by
  -- Definitions
  let price_student := 9
  let price_non_student := 11
  let total_tickets := 2000
  let student_tickets := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  -- Proof
  sorry  -- Placeholder for the proof

end concert_ticket_revenue_l257_257070


namespace first_person_work_days_l257_257771

theorem first_person_work_days (x : ℝ) (h1 : 0 < x) :
  (1/x + 1/40 = 1/15) → x = 24 :=
by
  intro h
  sorry

end first_person_work_days_l257_257771


namespace square_side_length_of_cross_reassembly_l257_257538

theorem square_side_length_of_cross_reassembly (a b : ℝ)
  (h_cross_area : a * b = 5) :
  a = b → b = sqrt 5 := 
sorry

end square_side_length_of_cross_reassembly_l257_257538


namespace trigonometric_identity_l257_257759

variables (α β γ a b c : ℝ)
variables (h1 : 0 < α) (h2 : α < π)
variables (h3 : 0 < β) (h4 : β < π)
variables (h5 : 0 < γ) (h6 : γ < π)
variables (h7 : 0 < a)
variables (h8 : 0 < b)
variables (h9 : 0 < c)
variables (h10 : b = c * (cos α + cos β * cos γ) / (sin γ)^2)
variables (h11 : a = c * (cos β + cos α * cos γ) / (sin γ)^2)

theorem trigonometric_identity :
  1 - cos α ^ 2 - cos β ^ 2 - cos γ ^ 2 - 2 * cos α * cos β * cos γ = 0 :=
sorry

end trigonometric_identity_l257_257759


namespace Claire_photos_is_5_l257_257742

variable (Claire_photos : ℕ)
variable (Lisa_photos : ℕ := 3 * Claire_photos)
variable (Robert_photos : ℕ := Claire_photos + 10)

theorem Claire_photos_is_5
  (h1 : Lisa_photos = Robert_photos) :
  Claire_photos = 5 :=
by
  sorry

end Claire_photos_is_5_l257_257742


namespace convex_quadrilateral_count_l257_257074

theorem convex_quadrilateral_count (n k : ℕ) (hn : n = 12) (hk : k = 4) : nat.choose n k = 3960 := by
  rw [hn, hk]
  exact nat.choose_succ_succ_succ_succ 11 9

end convex_quadrilateral_count_l257_257074


namespace solve_for_m_l257_257834

theorem solve_for_m :
  (∀ (m : ℕ), 
   ((1:ℚ)^(m+1) / 5^(m+1) * 1^18 / 4^18 = 1 / (2 * 10^35)) → m = 34) := 
by apply sorry

end solve_for_m_l257_257834


namespace int_solutions_eq_count_int_values_b_l257_257557

theorem int_solutions_eq (b : ℤ) : 
  ∃! x : ℤ, ∃! y : ℤ, (x + y = -b) ∧ (x * y = 12 * b) \/
  (x + y = -b) ∧ (x * y = 12 * b) :=
begin
  -- Assume roots p, q exist
  -- Use Vieta's formulas: p + q = -b, pq = 12b
  -- Transform the equation using SFFT
  sorry
end

theorem count_int_values_b :
  set_finite {b : ℤ | ∃! x : ℤ, ∃! y : ℤ, 
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} ∧
  fintype.card {b : ℤ | ∃! x : ℤ, ∃! y : ℤ,
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} = 16 :=
begin
  sorry
end

end int_solutions_eq_count_int_values_b_l257_257557


namespace time_for_all_students_l257_257896

/-- Conditions of the problem -/
def num_students_per_classroom := 54
def num_classrooms := 32
def students_passing_one_main_entrance_per_minute := 120
def students_passing_one_side_entrance_per_minute := 80

/-- Proof statement -/
theorem time_for_all_students (x_main y_side : ℕ) 
  (h1 : x_main = students_passing_one_main_entrance_per_minute)
  (h2 : y_side  = students_passing_one_side_entrance_per_minute) 
  (h3 : 1 * x_main + 2 * y_side = 560 / 2)
  (h4 : 1 * x_main + 1 * y_side = 800 / 4)
  (h5 : num_classrooms * num_students_per_classroom = 32 * 54):

  ( num_classrooms * num_students_per_classroom / (2 * x_main + y_side) = 5.4 ) :=
by 
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end time_for_all_students_l257_257896


namespace problem_part_I_problem_part_II_l257_257276

noncomputable def vector_a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3 / 2 * x), Real.sin (3 / 2 * x))

noncomputable def vector_b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (1 / 2 * x), -Real.sin (1 / 2 * x))

def vector_dot (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def vector_norm (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

noncomputable def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def f (x m : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  vector_dot a b - 4 * m * vector_norm (vector_sum a b) + 1

theorem problem_part_I (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ Real.pi / 2) :
  vector_dot (vector_a x) (vector_b x) = Real.cos (2 * x) ∧
  vector_norm (vector_sum (vector_a x) (vector_b x)) = 2 * Real.cos x := sorry

theorem problem_part_II (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f(x, m) ≥ -1 / 2) →
  m = 1 / 4 := sorry

end problem_part_I_problem_part_II_l257_257276


namespace min_two_loadings_needed_to_prove_first_ingot_weight_is_1_l257_257363

structure IngotWeights (α : Type) :=
  (weights : Fin 11 → α)
  (distinct_weights : Function.Injective weights)
  (range_weights : ∀ i : Fin 11, weights i ∈ (Finset.range 12))

def bag_capacity : ℕ := 11

def total_weight {α : Type} [Add α] [Zero α] (ws : Finset α) : α :=
  ws.sum id

theorem min_two_loadings_needed_to_prove_first_ingot_weight_is_1
  (ws : IngotWeights ℕ) (f1 f2 f5 : Fin 11) (f3 f4 f6 : Fin 11) : Prop :=
  (ws.weights f1 = 1) ∧
  total_weight (Finset.ofList [ws.weights f1, ws.weights f2, ws.weights f3, ws.weights f5]) ≤ bag_capacity ∧
  total_weight (Finset.ofList [ws.weights f1, ws.weights f4, ws.weights f6]) ≤ bag_capacity

#check min_two_loadings_needed_to_prove_first_ingot_weight_is_1

end min_two_loadings_needed_to_prove_first_ingot_weight_is_1_l257_257363


namespace find_a_l257_257657

noncomputable def f (x a : ℝ) : ℝ := x + a / x 

theorem find_a (a : ℝ) (h1 : ∀ x, 0 < x → f x a = x + a / x)
  (h2 : ∀ x, 0 < x ∧ x < 2 → (f x a)' < 0)
  (h3 : ∀ x, 2 < x → (f x a)' > 0) :
  a = 4 :=
sorry

end find_a_l257_257657


namespace exponent_product_l257_257590

variables {a m n : ℝ}

theorem exponent_product (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 :=
by
  sorry

end exponent_product_l257_257590


namespace proposition_D_is_true_l257_257512

-- Define the propositions
def proposition_A : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def proposition_B : Prop := ∀ x : ℝ, 2^x > x^2
def proposition_C : Prop := ∀ a b : ℝ, (a + b = 0 ↔ a / b = -1)
def proposition_D : Prop := ∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1

-- Problem statement: Proposition D is true
theorem proposition_D_is_true : proposition_D := 
by sorry

end proposition_D_is_true_l257_257512


namespace axis_of_symmetry_r_minus_2s_zero_l257_257944

/-- 
Prove that if y = x is an axis of symmetry for the curve 
y = (2 * p * x + q) / (r * x - 2 * s) with p, q, r, s nonzero, 
then r - 2s = 0. 
-/
theorem axis_of_symmetry_r_minus_2s_zero
  (p q r s : ℝ) (h_p : p ≠ 0) (h_q : q ≠ 0) (h_r : r ≠ 0) (h_s : s ≠ 0) 
  (h_sym : ∀ (a b : ℝ), (b = (2 * p * a + q) / (r * a - 2 * s)) ↔ (a = (2 * p * b + q) / (r * b - 2 * s))) :
  r - 2 * s = 0 :=
sorry

end axis_of_symmetry_r_minus_2s_zero_l257_257944


namespace solve_quadratic_inequality_l257_257803

theorem solve_quadratic_inequality (x : ℝ) : (2 * x^2 - x - 1 > 0) ↔ x ∈ set.Ioo (-∞) (-1/2) ∪ set.Ioo 1 ∞ :=
sorry

end solve_quadratic_inequality_l257_257803


namespace combined_value_l257_257309

theorem combined_value (a b c : ℕ) (h1 : 0.005 * a = 0.95)
  (h2 : b = 3 * a - 50)
  (h3 : c = (a - b)^2)
  (h_pos_a : 0 < a) (h_pos_c : 0 < c) : 
  a + b + c = 109610 :=
sorry

end combined_value_l257_257309


namespace math_problem_l257_257270

noncomputable def point (x y z : ℝ) := (x, y, z)

def A : ℝ × ℝ × ℝ := point 0 1 0
def B : ℝ × ℝ × ℝ := point 2 2 0
def C : ℝ × ℝ × ℝ := point -1 3 1

def vector (p1 p2 : ℝ × ℝ × ℝ) := 
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def AB := vector A B
def AC := vector A C

def unit_vector (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  let mag := real.sqrt (v.1^2 + v.2^2 + v.3^2)
  in (v.1 / mag, v.2 / mag, v.3 / mag)

def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ := 
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
  let mag_v1 := real.sqrt (v1.1^2 + v1.2^2 + v1.3^2)
  let mag_v2 := real.sqrt (v2.1^2 + v2.2^2 + v2.3^2)
  in dot_product / (mag_v1 * mag_v2)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

theorem math_problem :
  (AB ≠ (2 : ℝ) • AC) ∧
  (unit_vector AB = (2 * real.sqrt 5 / 5, real.sqrt 5 / 5, 0)) ∧
  (cos_angle AB (vector B C) ≠ real.sqrt 55 / 11) ∧
  (cross_product AB AC = (1, -2, 5)) :=
sorry

end math_problem_l257_257270


namespace jaco_budget_for_parents_l257_257351

/-- Assume Jaco has 8 friends, each friend's gift costs $9, and Jaco has a total budget of $100.
    Prove that Jaco's budget for each of his mother and father's gift is $14. -/
theorem jaco_budget_for_parents :
  ∀ (friends_count cost_per_friend total_budget : ℕ), 
  friends_count = 8 → 
  cost_per_friend = 9 → 
  total_budget = 100 → 
  (total_budget - friends_count * cost_per_friend) / 2 = 14 :=
by
  intros friends_count cost_per_friend total_budget h1 h2 h3
  rw [h1, h2, h3]
  have friend_total_cost : friends_count * cost_per_friend = 72 := by norm_num
  have remaining_budget : total_budget - friends_count * cost_per_friend = 28 := by norm_num [friend_total_cost]
  have divided_budget : remaining_budget / 2 = 14 := by norm_num [remaining_budget]
  exact divided_budget

end jaco_budget_for_parents_l257_257351


namespace seating_arrangements_l257_257069

-- Define the conditions and the proof problem
theorem seating_arrangements (children : Finset (Fin 6)) 
  (is_sibling_pair : (Fin 6) -> (Fin 6) -> Prop)
  (no_siblings_next_to_each_other : (Fin 6) -> (Fin 6) -> Bool)
  (no_sibling_directly_in_front : (Fin 6) -> (Fin 6) -> Bool) :
  -- Statement: There are 96 valid seating arrangements
  ∃ (arrangements : Finset (Fin 6 -> Fin (2 * 3))),
  arrangements.card = 96 :=
by
  -- Proof omitted
  sorry

end seating_arrangements_l257_257069


namespace smallest_a_and_b_l257_257965

theorem smallest_a_and_b :
  ∃ a b : ℕ, (b > 1) ∧ (a = 256) ∧ (b = 128) ∧ (sqrt (a * sqrt (a * sqrt a)) = b) :=
by
  use 256, 128
  split
  · exact Nat.lt_of_succ_lt_succ (by norm_num : (1 + 1 : ℕ) < 128)
  split
  · rfl
  split
  · rfl
  sorry

end smallest_a_and_b_l257_257965


namespace probability_third_quadrant_l257_257974

-- Define the sets for a and b
def a_set : Set ℝ := {1/2, 1/3, 2, 3}
def b_set : Set ℤ := {-2, -1, 1, 2}

-- Total number of basic events
def total_events : ℕ := Set.size a_set * Set.size b_set

-- Favorable outcomes where y = a^x + b passes through the third quadrant
def favorable_cases: ℕ := 6

-- Lean statement to prove the probability is 3/8
theorem probability_third_quadrant :
  (favorable_cases : ℚ) / total_events = 3 / 8 :=
sorry

end probability_third_quadrant_l257_257974


namespace range_of_k_l257_257969

theorem range_of_k :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) :=
by
  sorry

end range_of_k_l257_257969


namespace calc_num_int_values_l257_257292

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l257_257292


namespace count_integers_l257_257572

def isPerfectSquare (x : ℚ) : Prop :=
  ∃ n : ℤ, n^2 = x

theorem count_integers (count : ℕ) : 
  (count = 2) ↔ (count = {n : ℤ | isPerfectSquare (n / (24 - n))}.toFinset.card) :=
by
  sorry

end count_integers_l257_257572


namespace hexagonal_tile_total_l257_257492

-- Define the number of tiles along each side of the hexagonal region
def numTilesAlongSide : ℕ := 10

-- Hypothesize the total number of hexagonal tiles covering the entire region
theorem hexagonal_tile_total :
  let n := numTilesAlongSide in
  (∑ i in finset.range (2 * n - 1), if i < n then n + i else 3 * n - 2 - i) = 310 :=
by
  sorry

end hexagonal_tile_total_l257_257492


namespace max_product_of_sum_2024_l257_257094

theorem max_product_of_sum_2024 : 
  ∃ (x y : ℤ), x + y = 2024 ∧ x * y = 1024144 :=
by
  use 1012, 1012
  split
  · sorry
  · sorry

end max_product_of_sum_2024_l257_257094


namespace integer_values_of_b_for_quadratic_eqn_l257_257561

noncomputable def number_of_integer_values_of_b : ℕ := 16

theorem integer_values_of_b_for_quadratic_eqn :
  ∃(b : ℤ) (k ≥ 0), ∀m n : ℤ, (m + n = -b ∧ m * n = 12 * b) → (m + 12) * (n + 12) = 144 → k = number_of_integer_values_of_b := sorry

end integer_values_of_b_for_quadratic_eqn_l257_257561


namespace max_intersections_circle_cosine_l257_257482

theorem max_intersections_circle_cosine (h k r : ℝ) :
  ∃ x y : ℝ, (x - h)^2 + (cos x - k)^2 = r^2 → y = cos x → 
  ∃ n : ℕ, n ≤ 8 :=
sorry

end max_intersections_circle_cosine_l257_257482


namespace train_speed_60_kmph_l257_257910

theorem train_speed_60_kmph (length_train : ℕ) (time_to_cross : ℕ) 
  (h_length : length_train = 200) 
  (h_time : time_to_cross = 12) : 
  let distance_km := (length_train : ℝ) / 1000
  let time_hr := (time_to_cross : ℝ) / 3600
  let speed_kmph := distance_km / time_hr
  speed_kmph = 60 := 
by 
  rw [h_length, h_time]
  simp [distance_km, time_hr, speed_kmph]
  norm_num
  sorry

end train_speed_60_kmph_l257_257910


namespace angle_between_a_and_b_l257_257275

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Definitions of given conditions
def a_norm : real := ‖a‖ = 1
def b_norm : real := ‖b‖ = 3 * real.sqrt 2
def combined_norm : real := ‖2 • a + b‖ = real.sqrt 10

-- Statement to prove the angle between a and b is 135 degrees
theorem angle_between_a_and_b (ha : a_norm) (hb : b_norm) (hab : combined_norm) :
  real.arccos (inner a b / (‖a‖ * ‖b‖)) = (135:ℝ) * (real.pi / 180) :=
sorry

end angle_between_a_and_b_l257_257275


namespace find_B_share_l257_257117

-- Definitions for the conditions
def proportion (a b c d : ℕ) := 6 * a = 3 * b ∧ 3 * b = 5 * c ∧ 5 * c = 4 * d

def condition (c d : ℕ) := c = d + 1000

-- Statement of the problem
theorem find_B_share (A B C D : ℕ) (x : ℕ) 
  (h1 : proportion (6*x) (3*x) (5*x) (4*x)) 
  (h2 : condition (5*x) (4*x)) : 
  B = 3000 :=
by 
  sorry

end find_B_share_l257_257117


namespace max_area_of_fenced_rectangle_l257_257612

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l257_257612


namespace count_valid_d_l257_257002

def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def valid_d_values : Finset ℕ := (Finset.range 10).filter (λ d, (0.1 * d : ℚ) > 0.063)

theorem count_valid_d : valid_d_values.card = 3 :=
by 
  sorry

end count_valid_d_l257_257002


namespace hyperbola_A_angle_bisector_l257_257882

open Real

def hyperbola_equation (a b x y : ℝ) : Prop :=
  (x^2) / (a^2) - (y^2) / (b^2) = 1

noncomputable def solve_hyperbola_equation (x y : ℝ) (ecc : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ecc = a / b ∧ hyperbola_equation a b x y

noncomputable def angle_bisector_line (F1 F2 A M : Point) : Prop :=
  ∃ m : ℝ, ∃ (x : ℝ), x = m ∧ 0 < m ∧ 
    (let line_F1A := 3 * x - 4 * y + 12 = 0 in
    let line_F2A := x = 4 in
    distance_point_line M line_F1A = distance_point_line M line_F2A)

theorem hyperbola_A (A : Point) : 
  solve_hyperbola_equation 4 6 2 :=
sorry

theorem angle_bisector (F1 F2 A : Point) (M : Point) : 
  -4 < fst M ∧ fst M < 4 ∧ 
  angle_bisector_line F1 F2 A M :=
sorry

end hyperbola_A_angle_bisector_l257_257882


namespace count_integer_values_of_x_l257_257288

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : {n : ℕ | 121 < n ∧ n ≤ 144}.card = 23 := 
by
  sorry

end count_integer_values_of_x_l257_257288


namespace sum_a_d_g_l257_257177

-- Definitions based on given conditions:
variables {a b c d e f g h i : ℕ}

-- a, b, c, d, e, f, g, h, i each must be distinct and range from 1 to 9.
axiom distinct_numbers : ∀ x ∈ {a, b, c, d, e, f, g, h, i}, 1 ≤ x ∧ x ≤ 9
axiom all_different : list.nodup [a, b, c, d, e, f, g, h, i]

-- Sum of these 9 sums is equal
axiom nine_sums_equal : 
  ∀ s, 
    s = a + b + c ∧
    s = d + e + f ∧
    s = g + h + i ∧
    s = a + d + g ∧
    s = b + e + h ∧
    s = c + f + i ∧
    s = c + d + e ∧
    s = h + i + a ∧
    s = f + g + b

-- The proof statement: prove that a + d + g = 18
theorem sum_a_d_g :
  a + d + g = 18 := 
sorry

end sum_a_d_g_l257_257177


namespace find_y_l257_257952

def vectors_orthogonal_condition (y : ℝ) : Prop :=
  (1 * -2) + (-3 * y) + (-4 * -1) = 0

theorem find_y : vectors_orthogonal_condition (2 / 3) :=
by
  sorry

end find_y_l257_257952


namespace found_nails_l257_257277

theorem found_nails (N : ℕ) (h1 : 247 + N + 109 = 500) : N = 144 := 
by {
  have h2 : 356 + N = 500 := by linarith,
  have h3 : N = 500 - 356 := by linarith,
  exact h3,
}

end found_nails_l257_257277


namespace method_1_saves_more_money_l257_257164

-- Definitions for conditions
def price_racket := 20
def price_shuttlecock := 5
def quantity_rackets := 4
def quantity_shuttlecocks := 30

-- Definitions for discount methods
def cost_method_1 := price_racket * quantity_rackets + price_shuttlecock * (quantity_shuttlecocks - quantity_rackets)
def cost_method_2 := (price_racket * quantity_rackets + price_shuttlecock * quantity_shuttlecocks) * 0.92

theorem method_1_saves_more_money : cost_method_1 < cost_method_2 :=
by {
  -- The proof is omitted
  sorry
}

end method_1_saves_more_money_l257_257164


namespace max_avg_play_time_l257_257389

theorem max_avg_play_time
  (wednesday thursday : ℕ) 
  (tom_friday fred_friday extra_friday : ℕ) 
  (total_days : ℕ) :
  wednesday = 2 →
  thursday = 2 →
  tom_friday = 4 →
  fred_friday = 6 →
  extra_friday = 0.5 →
  total_days = 3 →
  (wednesday + thursday + fred_friday + extra_friday) / total_days = 3.5 :=
by
  sorry

end max_avg_play_time_l257_257389


namespace total_puppies_is_74_l257_257846

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l257_257846


namespace max_area_of_fenced_rectangle_l257_257616

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l257_257616


namespace sufficient_drivers_and_ivan_petrovich_departure_l257_257038

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l257_257038


namespace concert_revenue_l257_257424

-- Defining the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := ticket_price_adult / 2
def attendees_adults : ℕ := 183
def attendees_children : ℕ := 28

-- Defining the total revenue calculation based on the conditions
def total_revenue : ℕ :=
  attendees_adults * ticket_price_adult +
  attendees_children * ticket_price_child

-- The theorem to prove the total revenue
theorem concert_revenue : total_revenue = 5122 := by
  sorry

end concert_revenue_l257_257424


namespace max_odd_integers_l257_257918

theorem max_odd_integers (a b c d e f : ℕ) (h1 : (a * b * c * d * e * f).even) (h2 : (a + b + c + d + e + f).odd) : 
  ∃ n : ℕ, n ≤ 5 ∧ (a.odd + b.odd + c.odd + d.odd + e.odd + f.odd) = n ∧ (a * b * c * d * e * f).even ∧ (a + b + c + d + e + f).odd :=
by {
  sorry
}

end max_odd_integers_l257_257918


namespace vectors_relationship_l257_257274

open RealEuclideanSpace

noncomputable def a : ℝ³ := ⟨-2, -3, 1⟩
noncomputable def b : ℝ³ := ⟨2, 0, 4⟩
noncomputable def c : ℝ³ := ⟨-4, -6, 2⟩

theorem vectors_relationship :
  (a ∥ c) ∧ (a ⊥ b) :=
by
  -- Proof omitted
  sorry

end vectors_relationship_l257_257274


namespace prove_trig_inequality_l257_257800

noncomputable def trig_inequality : Prop :=
  (0 < 1 / 2) ∧ (1 / 2 < Real.pi / 6) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.sin x < Real.sin y) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.cos x > Real.cos y) →
  (Real.cos (1 / 2) > Real.tan (1 / 2) ∧ Real.tan (1 / 2) > Real.sin (1 / 2))

theorem prove_trig_inequality : trig_inequality :=
by
  sorry

end prove_trig_inequality_l257_257800


namespace total_area_of_sheet_l257_257893

theorem total_area_of_sheet (x : ℕ) (h1 : 4 * x - x = 2208) : x + 4 * x = 3680 := 
sorry

end total_area_of_sheet_l257_257893


namespace pq_vector_magnitude_range_l257_257231

noncomputable def vector_magnitude {A P Q : ℝ × ℝ} (A P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem pq_vector_magnitude_range {x y : ℝ} (h1 : x^2 + y^2 = 1) :
  ∃ t : ℝ, ∀ (P : ℝ × ℝ), P = (x, y) ∧
  ∀ (A Q : ℝ × ℝ), A = (1, 1) ∧ Q = (y + 1, x + 1) →
  sqrt(2 * (x - y)^2 + 2) ∈ set.Icc (sqrt 2) (sqrt 6) :=
begin
  sorry
end

end pq_vector_magnitude_range_l257_257231


namespace hikers_speed_l257_257145

theorem hikers_speed (v : ℝ) :
  (∀ v : ℝ, let distance := 30 * (12/60)
  ∧ hiker_time := 48/60
  ∧ 6 = v * hiker_time
  in v = 7.5) := sorry

end hikers_speed_l257_257145


namespace max_area_of_rectangular_pen_l257_257600

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l257_257600


namespace monotonic_intervals_range_of_a_for_two_zeros_inequality_l257_257656

open Real

-- Definitions and axioms
def f (a : ℝ) (x : ℝ) : ℝ := a * ln x - x

-- Problem 1: Monotonic intervals
theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, (∀ x₁ > x₂, f a x₁ < f a x₂)) ∧
  (a > 0 → (∀ x ∈ Ioo 0 a, (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂)) ∧
          (∀ x ∈ Ioi a, (∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂))) :=
sorry

-- Problem 2: Range of a
theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) → a > exp 1 :=
sorry

-- Problem 3: Prove the inequality
theorem inequality (a x₁ x₂ : ℝ) (h₀ : 0 < x₁) (h₁ : x₁ < x₂) (h₂ : a = x₁ / ln x₁) (h₃ : a = x₂ / ln x₂) :
  x₁ / ln x₁ < 2 * x₂ - x₁ :=
sorry

end monotonic_intervals_range_of_a_for_two_zeros_inequality_l257_257656


namespace problem_solution_l257_257639

theorem problem_solution (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
  (∑ i : Fin n, (x i)^2 / x (Fin.cycle n i)) ≥ 
  (∑ i : Fin n, x i) + 
  (4 * (x 0 - x 1)^2 / ∑ i : Fin n, x i) :=
by
  sorry

end problem_solution_l257_257639


namespace base_7_3516_is_1287_l257_257441

-- Definitions based on conditions
def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 3516 => 3 * 7^3 + 5 * 7^2 + 1 * 7^1 + 6 * 7^0
  | _ => 0

-- Proving the main question
theorem base_7_3516_is_1287 : base7_to_base10 3516 = 1287 := by
  sorry

end base_7_3516_is_1287_l257_257441


namespace general_formulas_and_sum_l257_257248

noncomputable def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := (-2) ^ (n - 1)

def sequence_c (n : ℕ) : ℕ := sequence_a n * 2 ^ (n - 1)

def sum_c (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ n + 3

theorem general_formulas_and_sum 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = 2 * n - 1)
  (h2 : ∀ n, b n = (-2) ^ (n - 1))
  (h3 : ∀ n, c n = a n * 2 ^ (n - 1))
  (h4 : ∀ n, T n = (2 * n - 3) * 2 ^ n + 3) :
  (a 3 = 5) → (a 5 = 9) →
  (∀ n, S_n = (2 / 3) * b n + 1 / 3) →
  (sum_c = T) :=
sorry

end general_formulas_and_sum_l257_257248


namespace larry_wins_first_l257_257364

noncomputable def p_hit : ℝ := 1/3
noncomputable def p_miss : ℝ := 2/3

noncomputable def larry_first_win_prob : ℝ :=
  let p_first_hit := p_hit * p_hit in
  let p_repeat := (p_miss * p_miss * p_miss * p_miss * p_miss * p_miss) in
  let geom_sum := p_first_hit * (1 / (1 - p_repeat)) in
  geom_sum

theorem larry_wins_first : larry_first_win_prob = 729 / 5985 := by
  sorry  -- Proof here

end larry_wins_first_l257_257364


namespace percent_absent_of_students_l257_257757

theorem percent_absent_of_students
  (boys girls : ℕ)
  (total_students := boys + girls)
  (boys_absent_fraction girls_absent_fraction : ℚ)
  (boys_absent_fraction_eq : boys_absent_fraction = 1 / 8)
  (girls_absent_fraction_eq : girls_absent_fraction = 1 / 4)
  (total_students_eq : total_students = 160)
  (boys_eq : boys = 80)
  (girls_eq : girls = 80) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 18.75 :=
by
  sorry

end percent_absent_of_students_l257_257757


namespace two_results_l257_257022

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l257_257022


namespace max_area_of_rectangular_pen_l257_257610

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l257_257610


namespace arithmetic_mean_l257_257993

theorem arithmetic_mean (n : ℕ) (h : n > 1) (a1 a2 : ℝ) (an : Fin (n-2) → ℝ)
  (h1 : a1 = 1 - 1 / n) (h2 : a2 = 1 + 1 / (n^2)) (h_an : ∀ i, an i = 1) :
  (a1 + a2 + ∑ i, an i) / n = 1 - 1 / (n^2) + 1 / (n^3) :=
by
  sorry

end arithmetic_mean_l257_257993


namespace find_value_l257_257107

theorem find_value (a b : ℝ) (h : a + b + 1 = -2) : (a + b - 1) * (1 - a - b) = -16 := by
  sorry

end find_value_l257_257107


namespace problem1_problem2_problem3_l257_257520

-- Problem (1)
theorem problem1 : -36 * (5 / 4 - 5 / 6 - 11 / 12) = 18 := by
  sorry

-- Problem (2)
theorem problem2 : (-2) ^ 2 - 3 * (-1) ^ 3 + 0 * (-2) ^ 3 = 7 := by
  sorry

-- Problem (3)
theorem problem3 (x : ℚ) (y : ℚ) (h1 : x = -2) (h2 : y = 1 / 2) : 
    (3 / 2) * x^2 * y + x * y^2 = 5 / 2 := by
  sorry

end problem1_problem2_problem3_l257_257520


namespace max_students_l257_257158

def P : ℕ := 4860
def C : ℕ := 3645

def twoP : ℕ := 2 * P
def threeC : ℕ := 3 * C

theorem max_students : gcd twoP threeC / nat.lcm 2 3 = 202 := by
  sorry

end max_students_l257_257158


namespace band_to_orchestra_ratio_is_two_l257_257806

noncomputable def ratio_of_band_to_orchestra : ℤ :=
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  let band_students := (total_students - orchestra_students - choir_students)
  band_students / orchestra_students

theorem band_to_orchestra_ratio_is_two :
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  ratio_of_band_to_orchestra = 2 := by
  sorry

end band_to_orchestra_ratio_is_two_l257_257806


namespace cos_C_in_triangle_abc_l257_257341

noncomputable def cosine_C {A B C : ℝ}
  (h : (sin A) * 3 * 4 = (sin B) * 2 * 4 ∧ (sin B) * 4 * 2 = (sin C) * 2 * 3) : ℝ :=
-cos C

theorem cos_C_in_triangle_abc
  {A B C : ℝ} (h : (sin A) * 3 * 4 = (sin B) * 2 * 4 ∧ (sin B) * 4 * 2 = (sin C) * 2 * 3) :
  cosine_C h = -1 / 4 :=
by
  sorry

end cos_C_in_triangle_abc_l257_257341


namespace projection_of_a_onto_b_l257_257978

-- Define the vectors a and b.
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-2, 4)

-- Helper function to calculate the dot product of two vectors.
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Helper function to calculate the squared magnitude of a vector.
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

-- Define the projection function.
def projection (u v : ℝ × ℝ) : ℝ × ℝ := 
  let coeff := (dot_product u v) / (magnitude_squared v)
  in (coeff * v.1, coeff * v.2)

-- Proposition stating the desired result.
theorem projection_of_a_onto_b :
  projection a b = (-4 / 5, 8 / 5) :=
sorry

end projection_of_a_onto_b_l257_257978


namespace median_unchanged_after_removal_l257_257507

theorem median_unchanged_after_removal (s : Finset ℝ) (h_distinct: s.card ≥ 3) :
  let s' : Finset ℝ := s.erase (Finset.max' s (by sorry)).erase (Finset.min' s (by sorry)) in
  (Finset.card s' % 2 = 1) →
  (∃ m ∈ s', (Finset.median s') = m) :=
by
  sorry

end median_unchanged_after_removal_l257_257507


namespace first_player_wins_game_l257_257515

/-- 
Proof that the first player in a game where each player can add a number between 1 and 10 to the board,
starting at 0, can always guarantee to be the first to reach a number 1000 or greater.
-/
theorem first_player_wins_game : 
  ∃ strategy : (ℕ → ℕ) → ℕ, ∀ moves : ℕ → ℕ, 
    (∀ i, 1 ≤ moves i ∧ moves i ≤ 10) → 
    (10 + (λ i, (moves i + (11 - moves i))).sum (range 90) ≥ 1000) :=
by
  sorry

end first_player_wins_game_l257_257515


namespace fg_of_2_l257_257982

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 10 :=
by
  sorry

end fg_of_2_l257_257982


namespace train_speed_l257_257907

/-- Given that a train crosses a pole in 12 seconds and the length of the train is 200 meters,
prove that the speed of the train in km/hr is 60. -/
theorem train_speed 
  (cross_time : ℝ) (train_length : ℝ)
  (H1 : cross_time = 12) (H2 : train_length = 200) : 
  let distance_km := train_length / 1000
      time_hr := cross_time / 3600
      speed := distance_km / time_hr in
  speed = 60 :=
by
  sorry

end train_speed_l257_257907


namespace number_of_intersections_actual_number_of_intersections_l257_257198

-- Define the conditions as Lean definitions
variables {L : Type} (lines : fin 10 → L)
variable (parallel : ∃ i j : fin 10, i ≠ j ∧ lines i = lines j)
variable (concurrent : ∃ i j k : fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ true)

-- Define the statement to be proven
theorem number_of_intersections (lines_unique : ∀ (i j : fin 10), i ≠ j → lines i ≠ lines j) :
  ¬ parallel ∧ ¬ concurrent → 45 :=
begin
  sorry
end

-- Define the conditions for the actual problem
theorem actual_number_of_intersections 
  (no_more_parallel : ∀ (i j : fin 10), (i ≠ j ∧ lines i = lines j) → parallel)
  (intersect_at_single_point : ∀ (i j k : fin 10), (i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ true) → concurrent) :
  (parallel ∧ concurrent) → 42 :=
begin
  sorry
end

end number_of_intersections_actual_number_of_intersections_l257_257198


namespace quadratic_with_root_l257_257953

theorem quadratic_with_root :
  ∃ (a b c: ℚ), a ≠ 0 ∧ (∀ x: ℝ, a * x^2 + b * x + c = 0 ↔ x = 2 * real.sqrt 2 - 3 ∨ x = -2 * real.sqrt 2 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = 1 := 
by {
  use [1, 6, 1],
  split,
  { norm_num },
  split,
  { intros x,
    split,
    { intro h,
      rw [quadratic_eq_iff, mul_one] at h,
      cases h with h1 h2,
      simp only [add_left_eq_self, mes_sqrt2_eq_zero_iff] at *,
      exact or.intro_left _ h1,
      exact or.intro_right _ h2},
    { intros h,
      rw [quadratic_eq_iff ,mul_one, finsupp.zero_eq_map_iff] at h,
      tauto }
  },
  {norm_num },
  {norm_num},
}

end quadratic_with_root_l257_257953


namespace speed_of_train_l257_257460

variable (train_length : ℝ) (bridge_length : ℝ) (time : ℝ)
variable (train_cross_bridge : train_length = 300 ∧ bridge_length = 300 ∧ time = 45)

theorem speed_of_train (h : train_cross_bridge) : (train_length + bridge_length) / time = 13.33 :=
by
  cases h with ht hb_t
  cases hb_t with hb ht
  rw [ht, hb, ht]
  exact (600 / 45 : ℝ) = 13.33

end speed_of_train_l257_257460


namespace all_integers_occur_in_sequence_l257_257792

theorem all_integers_occur_in_sequence (a : ℕ → ℕ)
    (H1 : ∀ i, 0 ≤ a i ∧ a i ≤ i)
    (H2 : ∀ k, (∑ i in Finset.range (k + 1), Nat.choose k (a i)) = 2 ^ k) :
    ∀ N, ∃ i, a i = N :=
sorry

end all_integers_occur_in_sequence_l257_257792


namespace probability_sum_less_than_8_l257_257068

open_locale classical

-- Definitions based on conditions
def draws : list ℕ := [1, 2, 3]

-- All possible outcomes of drawing from the bag three times
def all_possible_outcomes := list.product (list.product draws draws) draws

-- Define function to calculate the sum of three draws (as tuples).
def sum_of_outcome (t : (ℕ × ℕ) × ℕ) : ℕ := t.fst.fst + t.fst.snd + t.snd

-- Count favorable outcomes
def count_favorable_outcomes : ℕ :=
  (all_possible_outcomes.countp (λ t, sum_of_outcome t < 8))

-- Define total possible outcomes (3^3)
def total_possible_outcomes : ℕ := 3 * 3 * 3

-- Define probability as a ratio of favorable outcomes to total outcomes
def probability : ℚ := count_favorable_outcomes / total_possible_outcomes

-- The theorem to be proven
theorem probability_sum_less_than_8 : probability = 23 / 27 := 
by sorry

end probability_sum_less_than_8_l257_257068


namespace straw_hat_value_l257_257816

theorem straw_hat_value:
  ∃ x y z : ℕ,
  (x^2 - 20 * y = 10 + z) ∧ (y ≥ 1) ∧ (1 ≤ z < 10) ∧ (10 - z = 4) := by
  sorry

end straw_hat_value_l257_257816


namespace general_term_formula_sum_first_n_terms_l257_257659

noncomputable def geometric_sequence (a_3 a_6 : ℕ) (n : ℕ) : ℕ :=
  if h : (a_3 = 4) ∧ (a_6 = 32) then 2^(n-1) else 0

theorem general_term_formula (a_3 a_6 n : ℕ) (h : (a_3 = 4) ∧ (a_6 = 32)) :
  geometric_sequence a_3 a_6 n = 2^(n-1) :=
by
  sorry

noncomputable def b_n (a_n : ℕ → ℕ) (n : ℕ) : ℕ :=
  a_n n - 3 * n

noncomputable def sum_b_n (a_n : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, b_n a_n (i + 1)

theorem sum_first_n_terms (a_3 a_6 : ℕ) :
  ∀ n, sum_b_n (geometric_sequence a_3 a_6) n = 2^n - 1 - (3 * n^2 + 3 * n) / 2 :=
by
  intros
  sorry

end general_term_formula_sum_first_n_terms_l257_257659


namespace trajectory_of_point_l257_257995

/-- 
  Given points A and B on the coordinate plane, with |AB|=2, 
  and a moving point P such that the sum of the distances from P
  to points A and B is constantly 2, the trajectory of point P 
  is the line segment AB. 
-/
theorem trajectory_of_point (A B P : ℝ × ℝ) 
  (h_AB : dist A B = 2) 
  (h_sum : dist P A + dist P B = 2) :
  P ∈ segment ℝ A B :=
sorry

end trajectory_of_point_l257_257995


namespace max_sum_sequence_2016_l257_257269

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n, |a (n + 1)| = |a n - 2|

theorem max_sum_sequence_2016
  (a : ℕ → ℤ)
  (h : sequence a) :
  (∑ n in finset.range 2016, a (n + 1)) ≤ 2016 :=
sorry

end max_sum_sequence_2016_l257_257269


namespace greatest_product_l257_257091

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l257_257091


namespace max_product_of_sum_2024_l257_257095

theorem max_product_of_sum_2024 : 
  ∃ (x y : ℤ), x + y = 2024 ∧ x * y = 1024144 :=
by
  use 1012, 1012
  split
  · sorry
  · sorry

end max_product_of_sum_2024_l257_257095


namespace distance_inequality_l257_257393

-- Given definitions for the proof context
variables {A B C P R S T : Point} -- Points in the Euclidean plane
variables {d D : ℝ} -- distances
variables (triangle_ABC : IsAcuteTriangle A B C) -- acute-angled triangle condition
variables (P_in_ABC : PointInTriangle P A B C) -- P is inside the triangle

-- Define maximum and minimum distances
def max_distance (P : Point) (A B C : Point) : ℝ :=
max (dist P A) (max (dist P B) (dist P C))

def min_distance (P : Point) (R S T : Point) : ℝ :=
min (dist P R) (min (dist P S) (dist P T))

-- Main theorem statement: D ≥ 2d, and D = 2d if and only if the triangle is equilateral
theorem distance_inequality (triangle_ABC : IsAcuteTriangle A B C) (P_in_ABC : PointInTriangle P A B C) :
  let D := max_distance P A B C,
      d := min_distance P R S T in
  D ≥ 2 * d ∧ (D = 2 * d ↔ IsEquilateralTriangle A B C) := 
sorry

end distance_inequality_l257_257393


namespace two_power_condition_ln_l257_257866

theorem two_power_condition_ln {a b : ℝ} (h₁ : 2^a > 2^b) (h₂ : ln a > ln b) : 
  (∃ (a b > 0), 2^a > 2^b ∧ ¬( ln a > ln b)) ∧ 
  (a > b → ln a > ln b) :=
sorry

end two_power_condition_ln_l257_257866


namespace scientific_notation_of_78922_l257_257331

theorem scientific_notation_of_78922 : ∃ (c : ℝ) (n : ℤ), 78922 = c * 10 ^ n ∧ c = 7.8922 ∧ n = 4 :=
by 
  use 7.8922, 4
  split
  · sorry
  split
  · sorry
  · sorry

end scientific_notation_of_78922_l257_257331


namespace max_rectangle_area_l257_257629

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l257_257629


namespace cannot_form_set_l257_257838

-- Declare group of objects as sets with the given constraints
def non_neg_reals_not_exceeding_20 := {x : ℝ | 0 ≤ x ∧ x ≤ 20}
def solutions_x_squared_minus_9 := {x : ℝ | x^2 = 9}
def all_approx_values_sqrt3 := {x : ℝ | true} -- Note: Approximations lack determinacy
def students_linchuan_2016_170cm := {x : Type | true} -- Placeholder; exact representation of the group isn't relevant for proof

-- State the theorem for proving the group which cannot form a set
theorem cannot_form_set :
  ¬(∃ (S : set ℝ), S = all_approx_values_sqrt3) :=
by
  sorry

end cannot_form_set_l257_257838


namespace train_speed_l257_257906

/-- Given that a train crosses a pole in 12 seconds and the length of the train is 200 meters,
prove that the speed of the train in km/hr is 60. -/
theorem train_speed 
  (cross_time : ℝ) (train_length : ℝ)
  (H1 : cross_time = 12) (H2 : train_length = 200) : 
  let distance_km := train_length / 1000
      time_hr := cross_time / 3600
      speed := distance_km / time_hr in
  speed = 60 :=
by
  sorry

end train_speed_l257_257906


namespace trajectory_A_eq_ellipse_l257_257644

-- Definition of the problem conditions
def triangle (A B C : ℝ × ℝ) := A ≠ B ∧ B ≠ C ∧ C ≠ A
def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
def perimeter (A B C : ℝ × ℝ) : ℝ := distance A B + distance B C + distance C A

-- Main statement to prove
theorem trajectory_A_eq_ellipse :
  ∃ (A : ℝ × ℝ), triangle A (-2, 0) (2, 0) ∧ perimeter A (-2, 0) (2, 0) = 10 ∧
    ((A.1 ^ 2) / 9 + (A.2 ^ 2) / 5 = 1 ∧ A.2 ≠ 0) :=
sorry

end trajectory_A_eq_ellipse_l257_257644


namespace least_number_of_palindromes_least_number_of_nice_palindromes_l257_257822

-- Definitions based on conditions
def code (n : ℕ) := list (fin 2)  -- represents a sequence of n digits (0 or 1)
def is_palindrome {α : Type} (l : list α) : Prop := l = l.reverse
def is_nice_palindrome {α : Type} (l : list α) : Prop := is_palindrome l

-- Theorems to be proven
theorem least_number_of_palindromes (n : ℕ) : ∃ c : code n, ∀ l : list (fin 2), l.length ≤ n → is_palindrome l → ((∃ k : ℕ, k = 2^(math.ceil(n / 2) + 1) - 2)) :=
sorry

theorem least_number_of_nice_palindromes (n : ℕ) : ∃ c : code n, ∀ l : list (fin 2), l.length ≤ n → is_nice_palindrome l → ((∃ k : ℕ, k = 2 * n - 2)) :=
sorry

end least_number_of_palindromes_least_number_of_nice_palindromes_l257_257822


namespace find_measure_of_angle_C_find_value_of_ab_l257_257694

variables {α : Type*}
variables (a b c : ℝ) (A B C : ℝ)
variables (S : ℝ)

-- Conditions
def triangle_ABC_is_acute_and_sides {a b c : ℝ} (h1 : a^2 + b^2 - c^2 = ab) : Prop :=
  ∃ (A B C : ℝ), A + B > C ∧ B + C > A ∧ C + A > B ∧ C < π / 2 

def area_of_triangle (S : ℝ) (a b : ℝ) (C : ℝ) (h2 : S = (1/2) * a * b * sin C) : Prop := True

-- Questions translated into proof forms
theorem find_measure_of_angle_C 
  (h1 : triangle_ABC_is_acute_and_sides a b c (by sorry)) :
  ∃ C : ℝ, a^2 + b^2 - c^2 = ab → cos C = 1/2 := sorry
  
theorem find_value_of_ab 
  (h1 : triangle_ABC_is_acute_and_sides a b c (by sorry)) 
  (h2 : c = sqrt 7)
  (h3 : area_of_triangle (3 * sqrt 3 / 2) a b (π / 3) (by sorry)) :
  a + b = 5 := sorry

end find_measure_of_angle_C_find_value_of_ab_l257_257694


namespace find_fx_at_pi_over_4_l257_257787

-- Given function and conditions
def f (x : Real) : Real := sin (2 * x + Real.pi / 6)

lemma distance_between_symmetry_axes : ∀ (ω > 0) (0 < varphi < Real.pi), 
    (Real.pi / 2) = Real.pi / ω :=
by 
  intro ω hω hvarphi
  sorry

lemma terminal_side_angle (0 < varphi < Real.pi) : 
    Real.tan varphi = sqrt 3 / 3 := 
by 
  intro hvarphi 
  sorry

theorem find_fx_at_pi_over_4 : 
  f (Real.pi / 4) = sqrt 3 / 2 :=
by 
  rw [f, Real.sin_add, Real.sin_pi_div_six, Real.cos_pi_div_six]
  sorry

end find_fx_at_pi_over_4_l257_257787


namespace triangle_sum_of_squares_l257_257723

variable {A B C G : Type}
variable [MetricSpace A B C G]

-- Let G be the centroid of triangle ABC
def is_centroid (G : [A B C]) : Prop := 
  sorry -- Definition of centroid is omitted for simplicity

-- Given GA^2 + GB^2 + GC^2 = 90
variable (GA2 GB2 GC2: ℝ)
def given_condition : Prop :=
  GA2 + GB2 + GC2 = 90

-- Proving AB^2 + AC^2 + BC^2 = 180
theorem triangle_sum_of_squares
  (G_cent : is_centroid G)
  (G_sq_sum : given_condition GA2 GB2 GC2) :
  let AB2 := dist A B ^ 2
  let AC2 := dist A C ^ 2
  let BC2 := dist B C ^ 2
  AB2 + AC2 + BC2 = 180 :=
sorry

end triangle_sum_of_squares_l257_257723


namespace parabola_focus_of_y_eq_ax2_l257_257416

variable (a : ℝ) (ha : a < 0)

def parabola_focus_coordinates : Prop :=
  -- Prove that the focus of the parabola y = ax^2 has coordinates (0, 1 / (4 * a))
  focus = (0, 1 / (4 * a))

theorem parabola_focus_of_y_eq_ax2 (h : parabola_focus_coordinates a ha) :
    ∀ (focus : ℝ × ℝ), focus = (0, 1 / (4 * a)) := 
  sorry

end parabola_focus_of_y_eq_ax2_l257_257416


namespace sufficient_drivers_and_ivan_petrovich_departure_l257_257037

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l257_257037


namespace find_second_largest_element_l257_257886

open List

theorem find_second_largest_element 
(a1 a2 a3 a4 a5 : ℕ) 
(h_pos : 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4 ∧ 0 < a5) 
(h_sorted : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) 
(h_mean : (a1 + a2 + a3 + a4 + a5) / 5 = 15) 
(h_range : a5 - a1 = 24) 
(h_mode : a2 = 10 ∧ a3 = 10) 
(h_median : a3 = 10) 
(h_three_diff : (a1 ≠ a2 ∨ a1 ≠ a3 ∨ a1 ≠ a4 ∨ a1 ≠ a5) ∧ (a4 ≠ a5)) :
a4 = 11 :=
sorry

end find_second_largest_element_l257_257886


namespace complex_sum_equals_exp_form_l257_257521

theorem complex_sum_equals_exp_form (cos_pos: cos (4 * Real.pi / 13) > 0):
  12 * Complex.exp (3 * Real.pi * Complex.I / 13) + 12 * Complex.exp (20 * Real.pi * Complex.I / 26)
  = 24 * abs (cos (4 * Real.pi / 13)) * Complex.exp (Complex.I * Real.pi / 2) :=
by
  sorry

end complex_sum_equals_exp_form_l257_257521


namespace initial_money_l257_257494

theorem initial_money (B S G M : ℕ) 
  (hB : B = 8) 
  (hS : S = 2 * B) 
  (hG : G = 3 * S) 
  (change : ℕ) 
  (h_change : change = 28)
  (h_total : B + S + G + change = M) : 
  M = 100 := 
by 
  sorry

end initial_money_l257_257494


namespace concert_total_revenue_l257_257423

def adult_ticket_price : ℕ := 26
def child_ticket_price : ℕ := adult_ticket_price / 2
def num_adults : ℕ := 183
def num_children : ℕ := 28

def revenue_from_adults : ℕ := num_adults * adult_ticket_price
def revenue_from_children : ℕ := num_children * child_ticket_price
def total_revenue : ℕ := revenue_from_adults + revenue_from_children

theorem concert_total_revenue :
  total_revenue = 5122 :=
by
  -- proof can be filled in here
  sorry

end concert_total_revenue_l257_257423


namespace sum_of_angles_eq_1080_l257_257801

noncomputable def cis (θ : ℝ) : ℂ := complex.exp (θ * complex.I)
noncomputable def roots_of_unity (n : ℕ) (k : ℕ) : ℂ := cis ((2 * π * k) / n)
def z : ℂ := -1 / 2 - complex.I * (real.sqrt 3 / 2)
def θs : List ℝ := [40, 100, 160, 220, 280, 340]

theorem sum_of_angles_eq_1080 :
  (0 ≤ θs.all fun x => x ∧ θs.all fun x => x < 360 ∧ θs.sum = 1080) := 
sorry

end sum_of_angles_eq_1080_l257_257801


namespace find_area_of_DEFG_l257_257528

-- Definitions of points and their properties
def is_midpoint (P A B : Point) : Prop :=
  2 * vector_distance P A = vector_distance A B ∧ 2 * vector_distance P B = vector_distance A B

noncomputable def area_of_rectangle (P Q R S : Point) : ℝ :=
  vector_distance P Q * vector_distance Q R

noncomputable def area_of_quadrilateral (P Q R S : Point) : ℝ :=
  sorry

-- Given conditions
variables (A B C D E F G H : Point)

-- Conditions from the problem statement
hypothesis h1 : area_of_rectangle A B C D = 100
hypothesis h2 : is_midpoint E A B
hypothesis h3 : is_midpoint H C D
hypothesis h4 : is_midpoint F B C
hypothesis h5 : vector_distance D F = vector_distance F G ∧ vector_distance F G = vector_distance G C

-- Question formulated as a proof problem
theorem find_area_of_DEFG :
  area_of_quadrilateral D E F G = 25 :=
sorry

end find_area_of_DEFG_l257_257528


namespace min_volume_for_cone_l257_257877

noncomputable def min_cone_volume (V1 : ℝ) : Prop :=
  ∀ V2 : ℝ, (V1 = 1) → 
    V2 ≥ (4 / 3)

-- The statement without proof
theorem min_volume_for_cone : 
  min_cone_volume 1 :=
sorry

end min_volume_for_cone_l257_257877


namespace stock_percentage_change_l257_257931

theorem stock_percentage_change :
  let initial_value := 100
  let value_after_first_day := initial_value * (1 - 0.25)
  let value_after_second_day := value_after_first_day * (1 + 0.35)
  let final_value := value_after_second_day * (1 - 0.15)
  let overall_percentage_change := ((final_value - initial_value) / initial_value) * 100
  overall_percentage_change = -13.9375 := 
by
  sorry

end stock_percentage_change_l257_257931


namespace suff_not_nec_condition_l257_257867

theorem suff_not_nec_condition (λ : ℝ) (hλ : λ < 1) :
    (∀ n : ℕ, 1 ≤ n → (n+1)^2 - 2*λ*(n+1) - (n^2 - 2*λ*n) > 0) ∧ ¬(∀ n : ℕ, 1 ≤ n → (n+1)^2 - 2*λ*(n+1) - (n^2 - 2*λ*n) > 0 → λ < 1) :=
by {
  have suff : ∀ n : ℕ, 1 ≤ n → (n+1)^2 - 2*λ*(n+1) - (n^2 - 2*λ*n) > 0,
  { intro n,
    intro hn,
    calc
      (n+1)^2 - 2*λ*(n+1) - (n^2 - 2*λ*n)
        = 2*n - 2*λ + 1 : by ring
        ... > 0 : by linarith [hn, hλ]
  },
  have not_nec : ¬(∀ n : ℕ, 1 ≤ n → (n+1)^2 - 2*λ*(n+1) - (n^2 - 2*λ*n) > 0 → λ < 1),
  { intro h,
    specialize h 1 (by linarith),
    linarith },
  exact ⟨suff, not_nec⟩
}

end suff_not_nec_condition_l257_257867


namespace translated_function_is_correct_l257_257815

-- Define the original function
def f (x : ℝ) : ℝ := (x - 2) ^ 2 + 2

-- Define the translated function after moving 1 unit to the left
def g (x : ℝ) : ℝ := f (x + 1)

-- Define the final function after moving 1 unit upward
def h (x : ℝ) : ℝ := g x + 1

-- The statement to be proved
theorem translated_function_is_correct :
  ∀ x : ℝ, h x = (x - 1) ^ 2 + 3 :=
by
  -- Proof goes here
  sorry

end translated_function_is_correct_l257_257815


namespace sum_ef_l257_257740

variables (a b c d e f : ℝ)

-- Definitions based on conditions
def avg_ab : Prop := (a + b) / 2 = 5.2
def avg_cd : Prop := (c + d) / 2 = 5.8
def overall_avg : Prop := (a + b + c + d + e + f) / 6 = 5.4

-- Main theorem to prove
theorem sum_ef (h1 : avg_ab a b) (h2 : avg_cd c d) (h3 : overall_avg a b c d e f) : e + f = 10.4 :=
sorry

end sum_ef_l257_257740


namespace grocery_store_spending_l257_257719

/-- Lenny has $84 initially. He spent $24 on video games and has $39 left.
We need to prove that he spent $21 at the grocery store. --/
theorem grocery_store_spending (initial_amount spent_on_video_games amount_left after_games_left : ℕ) 
    (h1 : initial_amount = 84)
    (h2 : spent_on_video_games = 24)
    (h3 : amount_left = 39)
    (h4 : after_games_left = initial_amount - spent_on_video_games) 
    : after_games_left - amount_left = 21 := 
sorry

end grocery_store_spending_l257_257719


namespace sufficient_drivers_and_ivan_petrovich_departure_l257_257036

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l257_257036


namespace int_solutions_eq_count_int_values_b_l257_257556

theorem int_solutions_eq (b : ℤ) : 
  ∃! x : ℤ, ∃! y : ℤ, (x + y = -b) ∧ (x * y = 12 * b) \/
  (x + y = -b) ∧ (x * y = 12 * b) :=
begin
  -- Assume roots p, q exist
  -- Use Vieta's formulas: p + q = -b, pq = 12b
  -- Transform the equation using SFFT
  sorry
end

theorem count_int_values_b :
  set_finite {b : ℤ | ∃! x : ℤ, ∃! y : ℤ, 
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} ∧
  fintype.card {b : ℤ | ∃! x : ℤ, ∃! y : ℤ,
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} = 16 :=
begin
  sorry
end

end int_solutions_eq_count_int_values_b_l257_257556


namespace exists_infinitely_many_composite_l257_257553

noncomputable def tau (a : ℕ) : ℕ := sorry -- Placeholder definition for tau

def f (n : ℕ) : ℕ := tau n! - tau (n-1)!

theorem exists_infinitely_many_composite (n : ℕ) : ∃ n : ℕ, Nat.Prime n ∧ 
  (∀ m < n, f(m) < f(n)) := sorry

end exists_infinitely_many_composite_l257_257553


namespace max_rectangle_area_l257_257627

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l257_257627


namespace initial_price_correct_l257_257049

-- Definitions based on the conditions
def initial_price : ℝ := 3  -- Rs. 3 per kg
def new_price : ℝ := 5      -- Rs. 5 per kg
def reduction_in_consumption : ℝ := 0.4  -- 40%

-- The main theorem we need to prove
theorem initial_price_correct :
  initial_price = 3 :=
sorry

end initial_price_correct_l257_257049


namespace parameter_range_for_point_in_ellipse_l257_257046

theorem parameter_range_for_point_in_ellipse {x y k : ℝ} 
    (h_eq : k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0) 
    (h_point : x = y = 0) : 0 < |k| ∧ |k| < 1 :=
sorry

end parameter_range_for_point_in_ellipse_l257_257046


namespace restaurant_total_cost_l257_257927

def total_cost
  (adults kids : ℕ)
  (adult_meal_cost adult_drink_cost adult_dessert_cost kid_drink_cost kid_dessert_cost : ℝ) : ℝ :=
  let num_adults := adults
  let num_kids := kids
  let adult_total := num_adults * (adult_meal_cost + adult_drink_cost + adult_dessert_cost)
  let kid_total := num_kids * (kid_drink_cost + kid_dessert_cost)
  adult_total + kid_total

theorem restaurant_total_cost :
  total_cost 4 9 7 4 3 2 1.5 = 87.5 :=
by
  sorry

end restaurant_total_cost_l257_257927


namespace math_club_team_selection_l257_257391

noncomputable def choose (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.descFactorial n k / Nat.factorial k else 0

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_selected := 4
  let girls_selected := 4
  choose boys boys_selected * choose girls girls_selected = 103950 := 
by simp [choose]; sorry

end math_club_team_selection_l257_257391


namespace jacos_budget_l257_257354

theorem jacos_budget :
  (friends : Nat) (friend_gift_cost total_budget : Nat)
  (jaco_remainder_budget : Nat)
  : friends = 8 →
  friend_gift_cost = 9 →
  total_budget = 100 →
  jaco_remainder_budget = total_budget - (friends * friend_gift_cost) →
  (jaco_remainder_budget / 2) = 14 := by
  intros friends friend_gift_cost total_budget jaco_remainder_budget friends_eq friend_gift_cost_eq total_budget_eq jaco_remainder_budget_eq
  rw [friends_eq, friend_gift_cost_eq, total_budget_eq, jaco_remainder_budget_eq]
  simp
  sorry

end jacos_budget_l257_257354


namespace f_at_5_l257_257259

def f : ℝ → ℝ
| x ≤ 0 := 2^x
| x > 0 := f (x - 3)

theorem f_at_5 : f 5 = 1/2 :=
by sorry

end f_at_5_l257_257259


namespace train_speed_60_kmph_l257_257911

theorem train_speed_60_kmph (length_train : ℕ) (time_to_cross : ℕ) 
  (h_length : length_train = 200) 
  (h_time : time_to_cross = 12) : 
  let distance_km := (length_train : ℝ) / 1000
  let time_hr := (time_to_cross : ℝ) / 3600
  let speed_kmph := distance_km / time_hr
  speed_kmph = 60 := 
by 
  rw [h_length, h_time]
  simp [distance_km, time_hr, speed_kmph]
  norm_num
  sorry

end train_speed_60_kmph_l257_257911


namespace append_five_new_number_l257_257316

theorem append_five_new_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) : 
  10 * (10 * t + u) + 5 = 100 * t + 10 * u + 5 :=
by sorry

end append_five_new_number_l257_257316


namespace estimate_pi_l257_257203

theorem estimate_pi :
  ∀ (r : ℝ) (side_length : ℝ) (total_beans : ℕ) (beans_in_circle : ℕ),
  r = 1 →
  side_length = 2 →
  total_beans = 80 →
  beans_in_circle = 64 →
  (π = 3.2) :=
by
  intros r side_length total_beans beans_in_circle hr hside htotal hin_circle
  sorry

end estimate_pi_l257_257203


namespace ferry_speed_difference_l257_257576

open Nat

-- Define the time and speed of ferry P
def timeP := 3 -- hours
def speedP := 8 -- kilometers per hour

-- Define the distance of ferry P
def distanceP := speedP * timeP -- kilometers

-- Define the distance of ferry Q
def distanceQ := 3 * distanceP -- kilometers

-- Define the time of ferry Q
def timeQ := timeP + 5 -- hours

-- Define the speed of ferry Q
def speedQ := distanceQ / timeQ -- kilometers per hour

-- Define the speed difference
def speedDifference := speedQ - speedP -- kilometers per hour

-- The target theorem to prove
theorem ferry_speed_difference : speedDifference = 1 := by
  sorry

end ferry_speed_difference_l257_257576


namespace circle_area_less_than_circumference_probability_l257_257495

theorem circle_area_less_than_circumference_probability :
  let die : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  ∃ (x y : ℕ), x ∈ die ∧ y ∈ die ∧ 
    let d := x + y in
    d < 4 ∧ 
    (P(d = 2) + P(d = 3)) = 3/64 :=
begin
  sorry
end

end circle_area_less_than_circumference_probability_l257_257495


namespace range_of_a_l257_257321

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, a * x^2 + 2 * x + a ≤ 0) → a > 1 :=
begin
  sorry
end

end range_of_a_l257_257321


namespace drivers_schedule_l257_257029

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l257_257029


namespace solution_set_of_inequalities_l257_257407

theorem solution_set_of_inequalities
  (a m : ℝ) :
    (if a = 0 then {x : ℝ | 2 < x}
    else if a > 1 then {x : ℝ | x < 2 / a} ∪ {x : ℝ | 2 < x}
    else if a < 0 then {x : ℝ | 2 / a < x ∧ x < 2}
    else if 0 < a ∧ a < 1 then {x : ℝ | x < 2} ∪ {x : ℝ | 2 / a < x}
    else {x : ℝ | x < 2} ∪ {x : ℝ | 2 < x}) ∧
    (if m ≥ 1 / 4 then set.univ
    else {x : ℝ | x = (-1 + real.sqrt (1 - 4 * m)) / 2 ∨ x = (-1 - real.sqrt (1 - 4 * m)) / 2}) :=
sorry

end solution_set_of_inequalities_l257_257407


namespace least_integer_value_satisfying_inequality_l257_257961

theorem least_integer_value_satisfying_inequality : ∃ x : ℤ, 3 * |x| + 6 < 24 ∧ (∀ y : ℤ, 3 * |y| + 6 < 24 → x ≤ y) :=
  sorry

end least_integer_value_satisfying_inequality_l257_257961


namespace evaluate_power_l257_257673

theorem evaluate_power (n : ℕ) (h : 3^(2 * n) = 81) : 9^(n + 1) = 729 :=
by sorry

end evaluate_power_l257_257673


namespace limit_qn_limit_integral_l257_257550

noncomputable def point_of_intersection (n : ℕ) (hn : n > 0) : ℝ × ℝ :=
  let pn := 1 / (n : ℝ) * Real.exp (Int.log n)
  let qn := log (n * pn)
  (pn, qn)

theorem limit_qn :
  ∀ (n : ℕ) (hn : n > 0), 
  let (pn, qn) := point_of_intersection n hn in
  ∀ ε > 0, ∃ N, ∀ m > N, |qn - 1| < ε := 
by 
  sorry

theorem limit_integral :
  ∀ (n : ℕ) (hn : n > 0),
  let (pn, _) := point_of_intersection n hn in
  ∀ ε > 0, ∃ N, ∀ m > N, |m * ∫ x in (1 / (m : ℝ))..pn, log (m * x) - 1| < ε :=
by 
  sorry

end limit_qn_limit_integral_l257_257550


namespace find_missing_number_l257_257680

theorem find_missing_number 
  (x : ℕ) 
  (avg : (744 + 745 + 747 + 748 + 749 + some_num + 753 + 755 + x) / 9 = 750)
  (hx : x = 755) : 
  some_num = 804 := 
  sorry

end find_missing_number_l257_257680


namespace function_upper_bound_constant_not_improvable_function_l257_257737

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (hdom : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f(x)) 
  (hf1 : f 1 = 1) 
  (hconvex : ∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2) :
  ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x :=
sorry

theorem constant_not_improvable_function 
  (f : ℝ → ℝ) 
  (hdom : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f(x)) 
  (hf1 : f 1 = 1) 
  (hconvex : ∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2) :
  ¬(∃ c > 0, c < 2 ∧ (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ c * x)) :=
sorry

end function_upper_bound_constant_not_improvable_function_l257_257737


namespace souvenir_purchase_and_profit_l257_257776

variable (x y : ℕ)

/-- The distributor purchased a total of 100 items -/
def total_items := x + y = 100

/-- The one-time cost was 6200 yuan -/
def total_cost := 50 * x + 70 * y = 6200

/-- Number of type A souvenirs -/
def type_A_count := x = 40

/-- Number of type B souvenirs -/
def type_B_count := y = 60

/-- Profit calculations for type A -/
def profit_A := 50 * x

/-- Profit calculations for type B -/
def profit_B := 20 * y

/-- Total profit made if all items are sold -/
def total_profit := profit_A + profit_B = 3200

theorem souvenir_purchase_and_profit :
  total_items x y → 
  total_cost x y → 
  type_A_count x → 
  type_B_count y → 
  total_profit x y :=
by {
  intros h1 h2 h3 h4,
  rw [h3, h4],
  sorry
}

end souvenir_purchase_and_profit_l257_257776


namespace find_range_of_x_l257_257245

-- Conditions
variable (f : ℝ → ℝ)
variable (even_f : ∀ x : ℝ, f x = f (-x))
variable (mono_incr_f : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Equivalent proof statement
theorem find_range_of_x (x : ℝ) :
  f (Real.log (abs (x + 1)) / Real.log (1 / 2)) < f (-1) ↔ x ∈ Set.Ioo (-3 : ℝ) (-3 / 2) ∪ Set.Ioo (-1 / 2) 1 := by
  sorry

end find_range_of_x_l257_257245


namespace number_of_square_pentomino_tilings_l257_257284

theorem number_of_square_pentomino_tilings : 
  ∀ (rectangle_length rectangle_width pentomino_area : ℕ), 
    rectangle_length = 12 → 
    rectangle_width = 12 → 
    pentomino_area = 5 → 
    ∃ (num_tilings : ℕ), 
    (rectangle_length * rectangle_width) ≠ pentomino_area * num_tilings := 
begin
  intros l w p h1 h2 h3,
  use 144 / 5,
  rw [h1, h2, h3],
  exact nat.dvd_not_mem_divisors 144 5 28,
end

end number_of_square_pentomino_tilings_l257_257284


namespace minimum_special_city_count_l257_257064

noncomputable def special_city_minimum (V: Type) [Fintype V] [Nonempty V] (E: Type) [Graph V E] 
  (h_connected: ∀ v1 v2 : V, Path V E v1 v2) 
  (h_vertices_count: Fintype.card V = 2017) : ℕ :=
1344

theorem minimum_special_city_count (V: Type) [Fintype V] [Nonempty V] (E: Type) [Graph V E]
  (h_connected: ∀ v1 v2 : V, Path V E v1 v2)
  (h_vertices_count: Fintype.card V = 2017) :
  ∃ k: ℕ, (k = special_city_minimum V E h_connected h_vertices_count) :=
begin
  use 1344,
  reflexivity,
end

end minimum_special_city_count_l257_257064


namespace intersection_points_circle_parabola_l257_257798

theorem intersection_points_circle_parabola :
  let circle (x y : ℝ) := x^2 + y^2 = 25
  let parabola (y : ℝ) := y^2 = 16
  ∃ (s : Finset (ℝ × ℝ)), (∀ p ∈ s, circle p.1 p.2 ∧ parabola p.2) ∧ Finset.card s = 4 :=
by
  sorry

end intersection_points_circle_parabola_l257_257798


namespace net_deflection_l257_257175

theorem net_deflection 
  (m1 m2 : ℝ) (x1 x2 h1 h2 : ℝ)
  (g : ℝ) 
  (h_m1 : m1 = 78.75)
  (h_x1 : x1 = 1)
  (h_h1 : h1 = 15)
  (h_m2 : m2 = 45)
  (h_h2 : h2 = 29)
  (h_g : g = 9.8) -- assuming the usual gravitational constant for completeness
  : let k := (2 * m1 * g * (h1 + x1)) / (x1 ^ 2) in
    let eq := m2 * g * (h2 + x2) = (k * (x2 ^ 2)) / 2 in
    (28 * x2 ^ 2 - x2 - 29 = 0) → x2 ≈ 1.04 :=
by 
  intros k eq h_eq 
  -- proof goes here
  sorry

end net_deflection_l257_257175


namespace max_rectangle_area_l257_257634

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l257_257634


namespace max_value_of_f_on_interval_l257_257224

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x ^ 3 - 6 * x ^ 2 + a

theorem max_value_of_f_on_interval : 
  ∃ (a : ℝ), (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x a ≥ 3) → 
  (f 0 a = 43 ∧ (∀ y ∈ set.Icc (-2 : ℝ) (2 : ℝ), f y a ≤ 43)) :=
by
  sorry

end max_value_of_f_on_interval_l257_257224


namespace king_luis_courtiers_are_odd_l257_257717

theorem king_luis_courtiers_are_odd (n : ℕ) 
  (h : ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ i ≠ j) : 
  ¬ Even n := 
sorry

end king_luis_courtiers_are_odd_l257_257717


namespace new_ratio_of_mixture_l257_257121

theorem new_ratio_of_mixture (initial_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) (added_water : ℕ) (final_milk_volume final_water_volume : ℕ) :
  initial_volume = 45 → 
  milk_ratio = 4 → 
  water_ratio = 1 → 
  added_water = 3 → 
  final_milk_volume = (milk_ratio * initial_volume) / (milk_ratio + water_ratio) → 
  final_water_volume = ((water_ratio * initial_volume) / (milk_ratio + water_ratio)) + added_water → 
  (final_milk_volume : final_water_volume) = 3 : 1 :=
by
  sorry

end new_ratio_of_mixture_l257_257121


namespace hiker_displacement_l257_257146

theorem hiker_displacement :
  let start_point := (0, 0)
  let move_east := (24, 0)
  let move_north := (0, 20)
  let move_west := (-7, 0)
  let move_south := (0, -9)
  let final_position := (start_point.1 + move_east.1 + move_west.1, start_point.2 + move_north.2 + move_south.2)
  let distance_from_start := Real.sqrt (final_position.1^2 + final_position.2^2)
  distance_from_start = Real.sqrt 410
:= by 
  sorry

end hiker_displacement_l257_257146


namespace sum_of_first_10_terms_log2_a_l257_257643

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2 * sequence_a (n - 1)

def sequence_log2_a (n : ℕ) : ℕ :=
  n

theorem sum_of_first_10_terms_log2_a :
  (Finset.range 10).sum (λ n, sequence_log2_a (n + 1)) = 55 :=
by
  sorry

end sum_of_first_10_terms_log2_a_l257_257643


namespace count_integer_values_of_x_l257_257302

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l257_257302


namespace divisible_by_other_l257_257888

theorem divisible_by_other (y : ℕ) 
  (h1 : y = 20)
  (h2 : y % 4 = 0)
  (h3 : y % 8 ≠ 0) : (∃ n, n ≠ 4 ∧ y % n = 0 ∧ n = 5) :=
by 
  sorry

end divisible_by_other_l257_257888


namespace units_digit_of_41_cubed_plus_23_cubed_l257_257831

theorem units_digit_of_41_cubed_plus_23_cubed : 
  (41^3 + 23^3) % 10 = 8 := 
by
  -- Define the units digits of 41^3 and 23^3
  let u41 : ℕ := 1
  let u23 : ℕ := 7
  -- Define the calculation of the units digit
  have h : (u41 + u23) % 10 = 8 := by sorry
  -- Adding the units digits gives the required units digit
  exact h

end units_digit_of_41_cubed_plus_23_cubed_l257_257831


namespace solve_equation_solutions_count_l257_257767

open Real

theorem solve_equation_solutions_count :
  (∃ (x_plural : List ℝ), (∀ x ∈ x_plural, 2 * sqrt 2 * (sin (π * x / 4)) ^ 3 = cos (π / 4 * (1 - x)) ∧ 0 ≤ x ∧ x ≤ 2020) ∧ x_plural.length = 505) :=
sorry

end solve_equation_solutions_count_l257_257767


namespace total_puppies_adopted_l257_257849

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l257_257849


namespace storage_temperature_overlap_l257_257252

theorem storage_temperature_overlap (T_A_min T_A_max T_B_min T_B_max : ℝ) 
  (hA : T_A_min = 0)
  (hA' : T_A_max = 5)
  (hB : T_B_min = 2)
  (hB' : T_B_max = 7) : 
  (max T_A_min T_B_min, min T_A_max T_B_max) = (2, 5) := by 
{
  sorry -- The proof is omitted as per instructions.
}

end storage_temperature_overlap_l257_257252


namespace chess_game_participation_l257_257810

theorem chess_game_participation :
  ∃ k : ℕ, (k = 2) ∧ (∏ (i : ℕ) in finset.range 8, i.choose k) = 28 := by
  sorry

end chess_game_participation_l257_257810


namespace complement_union_l257_257976

def M : set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 2}
def N : set (ℝ × ℝ) := {p | p.2 ≠ -p.1}
def I : set (ℝ × ℝ) := {p | true}

theorem complement_union :
  (I \ (M ∪ N)) = {(-1 : ℝ, 1 : ℝ)} :=
by sorry

end complement_union_l257_257976


namespace variance_daily_reading_time_l257_257875

theorem variance_daily_reading_time :
  let mean10 := 2.7
  let var10 := 1
  let num10 := 800

  let mean11 := 3.1
  let var11 := 2
  let num11 := 600

  let mean12 := 3.3
  let var12 := 3
  let num12 := 600

  let num_total := num10 + num11 + num12

  let total_mean := (2.7 * 800 + 3.1 * 600 + 3.3 * 600) / 2000

  let var_total := (800 / 2000) * (1 + (2.7 - total_mean)^2) +
                   (600 / 2000) * (2 + (3.1 - total_mean)^2) +
                   (600 / 2000) * (3 + (3.3 - total_mean)^2)

  var_total = 1.966 :=
by
  sorry

end variance_daily_reading_time_l257_257875


namespace red_ball_probability_l257_257324

theorem red_ball_probability :
  let total_balls := 7
  let red_balls_initial := 4
  let white_balls_initial := 3
  let first_ball_red := True
  let total_balls_after_first := total_balls - 1
  let red_balls_after_first := red_balls_initial - 1
  let probability_first_red := red_balls_initial / total_balls
  let probability_second_red_given_first_red := red_balls_after_first / total_balls_after_first
  probability_second_red_given_first_red = 1/2 :=
by
  sorry

end red_ball_probability_l257_257324


namespace handshake_count_l257_257928

theorem handshake_count (team_members: ℕ) (referees: ℕ) :
  team_members = 6 → referees = 3 → (team_members * team_members + 2 * team_members * referees) = 72 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end handshake_count_l257_257928


namespace number_of_possible_values_of_x_l257_257298

theorem number_of_possible_values_of_x : 
  (∃ x : ℕ, ⌈Real.sqrt x⌉ = 12) → (set.Ico 144 169).card = 25 := 
by
  intros h
  sorry

end number_of_possible_values_of_x_l257_257298


namespace thirty_percent_less_than_80_equals_one_fourth_more_l257_257067

theorem thirty_percent_less_than_80_equals_one_fourth_more (n : ℝ) :
  80 * 0.30 = 24 → 80 - 24 = 56 → n + n / 4 = 56 → n = 224 / 5 :=
by
  intros h1 h2 h3
  sorry

end thirty_percent_less_than_80_equals_one_fourth_more_l257_257067


namespace drivers_schedule_l257_257025

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l257_257025


namespace median_equals_mean_sum_is_neg3_l257_257449

theorem median_equals_mean_sum_is_neg3 (y : ℝ) :
  (∀ (y : ℝ), (median([3, 7, 9, 19, y]) = (38 + y) / 5) → y = -3) →
  ∑ (y : ℝ) in {y | median ([3, 7, 9, 19, y]) = (38 + y) / 5}, y = -3 :=
sorry

end median_equals_mean_sum_is_neg3_l257_257449


namespace count_integer_values_of_x_l257_257303

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l257_257303


namespace decreasing_implies_a_leq_3_max_min_value_implies_a_gt_2_sqrt_2_l257_257257

def f (x : ℝ) (a : ℝ) := -x^2 + a*x + 1 - Real.log x

def f_prime (x : ℝ) (a : ℝ) := -2*x + a - 1/x

def g (x : ℝ) := 2*x + 1/x

-- Proof Problem 1
theorem decreasing_implies_a_leq_3 (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → f_prime x a ≤ 0) → a ≤ 3 :=
sorry

-- Proof Problem 2
theorem max_min_value_implies_a_gt_2_sqrt_2 (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0) → a > 2 * Real.sqrt 2 :=
sorry

end decreasing_implies_a_leq_3_max_min_value_implies_a_gt_2_sqrt_2_l257_257257


namespace locus_of_points_for_Brocard_angle_l257_257947

variables {A B C M A1 B1 C1 : Type*}
variables [Point A] [Point B] [Point C] [Point M] [Point A1] [Point B1] [Point C1]
variables (triangle_ABC : triangle A B C)
variables (foot_A1 : perpendicular M (line B C))
variables (foot_B1 : perpendicular M (line C A))
variables (foot_C1 : perpendicular M (line A B))
variables (BrocardAngle : ℝ)

theorem locus_of_points_for_Brocard_angle
  (h_perpendiculare_A1 : A1 = foot_A1)
  (h_perpendiculare_B1 : B1 = foot_B1)
  (h_perpendiculare_C1 : C1 = foot_C1)
  (h_Brocard_angle : BrocardAngle φ (triangle A1 B1 C1)) :
  ∃ (circle1 circle2 : ℝ² → Prop),
    circle_interior circle1 ∧
    circle_exterior circle2 ∧
    ∀ P, (circle1 P ∨ circle2 P) ↔ BrocardAngle φ (triangle A1 B1 C1) :=
sorry

end locus_of_points_for_Brocard_angle_l257_257947


namespace kayden_total_processed_l257_257362

-- Definition of the given conditions and final proof problem statement in Lean 4
variable (x : ℕ)  -- x is the number of cartons delivered to each customer

theorem kayden_total_processed (h : 4 * (x - 60) = 160) : 4 * x = 400 :=
by
  sorry

end kayden_total_processed_l257_257362


namespace sufficient_drivers_and_ivan_petrovich_departure_l257_257039

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l257_257039


namespace Jaco_budget_for_parents_gifts_l257_257350

theorem Jaco_budget_for_parents_gifts :
  ∃ (m n : ℕ), (m = 14 ∧ n = 14) ∧ 
  (∀ (friends gifts_friends budget : ℕ), 
   friends = 8 → gifts_friends = 9 → budget = 100 → 
   (budget - (friends * gifts_friends)) / 2 = m ∧ 
   (budget - (friends * gifts_friends)) / 2 = n) := 
sorry

end Jaco_budget_for_parents_gifts_l257_257350


namespace balance_scale_with_blue_balls_l257_257943

variables (G Y W B : ℝ)

-- Conditions
def green_to_blue := 4 * G = 8 * B
def yellow_to_blue := 3 * Y = 8 * B
def white_to_blue := 5 * B = 3 * W

-- Proof problem statement
theorem balance_scale_with_blue_balls (h1 : green_to_blue G B) (h2 : yellow_to_blue Y B) (h3 : white_to_blue W B) : 
  3 * G + 3 * Y + 3 * W = 19 * B :=
by sorry

end balance_scale_with_blue_balls_l257_257943


namespace square_diag_proof_l257_257000

open Real

theorem square_diag_proof (AE EC BF : ℝ) 
  (AB_side : ℝ) 
  (h1 : AE = 1) (h2 : EC = 2) (h3 : BF = 2) (h4 : AB_side = 4) 
  : let EF := AB_side - AE
    let EG := AB_side - EC
    let FG := sqrt (EF ^ 2 + EG ^ 2)
    let DG := AB_side - BF - FG
    in DG = 2 - sqrt 13 :=
by
  let EF := AB_side - AE
  let EG := AB_side - EC
  let FG := sqrt (EF ^ 2 + EG ^ 2)
  let DG := AB_side - BF - FG
  linarith [h1, h2, h3, h4, EF, EG, FG]
  sorry

end square_diag_proof_l257_257000


namespace correct_calculation_l257_257837

theorem correct_calculation :
    (1 + Real.sqrt 2)^2 = 3 + 2 * Real.sqrt 2 :=
sorry

end correct_calculation_l257_257837


namespace speed_second_boy_l257_257443

theorem speed_second_boy (v : ℝ) (t : ℝ) (d : ℝ) (s₁ : ℝ) :
  s₁ = 4.5 ∧ t = 9.5 ∧ d = 9.5 ∧ (d = (v - s₁) * t) → v = 5.5 :=
by
  intros h
  obtain ⟨hs₁, ht, hd, hev⟩ := h
  sorry

end speed_second_boy_l257_257443


namespace triangle_area_determinant_l257_257168

theorem triangle_area_determinant : 
  let A := (1, 3)
  let B := (4, -2)
  let C := (-2, 2)
  let area := (1 / 2) * (abs (1 * (-2 - 2) + 4 * (2 - 3) + (-2) * (3 - (-2)))) 
  area = 9.0 :=
by
  let A := (1, 3)
  let B := (4, -2)
  let C := (-2, 2)
  let area := (1 / 2) * (abs (1 * (-2 - 2) + 4 * (2 - 3) + (-2) * (3 - (-2)))) 
  calc
    area = (1 / 2) * (abs (1 * (-2 - 2) + 4 * (2 - 3) + (-2) * (3 - (-2)))) : rfl
    ... = (1 / 2) * (abs (-4 + (-4) + (-10))) : by simp
    ... = (1 / 2) * (abs (-18)) : by simp
    ... = (1 / 2) * 18 : by simp
    ... = 9.0 : by norm_num

end triangle_area_determinant_l257_257168


namespace balcony_more_than_orchestra_l257_257118

theorem balcony_more_than_orchestra (O B : ℕ) 
  (h1 : O + B = 355) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 115 :=
by 
  -- Sorry, this will skip the proof.
  sorry

end balcony_more_than_orchestra_l257_257118


namespace yuri_total_puppies_l257_257850

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l257_257850


namespace domain_function_l257_257014

theorem domain_function (x : ℝ) :
  (x + 1 ≥ 0 ∧ x - 1 ≠ 0 ∧ 2 - x > 0 ∧ log (2 - x) ≠ 0) ↔ (x ∈ [-1, 1) ∨ x ∈ (1, 2)) := by
  sorry

end domain_function_l257_257014


namespace net_amount_spent_is_correct_l257_257404

noncomputable def total_spent_before_discount : ℝ := 13.99 + 12.14 + 18.25 + 7.89
def discount_rate : ℝ := 0.10
def refund : ℝ := 7.43
def tax_rate : ℝ := 0.05

noncomputable def discounted_total : ℝ := total_spent_before_discount * (1 - discount_rate)
noncomputable def total_after_refund : ℝ := discounted_total + refund
noncomputable def sales_tax : ℝ := total_after_refund * tax_rate
noncomputable def net_amount_spent : ℝ := total_after_refund + sales_tax

theorem net_amount_spent_is_correct : Real.round (net_amount_spent * 100) / 100 = 57.20 :=
by
  sorry

end net_amount_spent_is_correct_l257_257404


namespace sum_geometric_arithmetic_sequences_l257_257235

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + 1

def geometric_sequence (b : ℕ → ℕ) : Prop :=
  b 1 = 1 ∧ ∀ n, b (n + 1) = b n * 4

theorem sum_geometric_arithmetic_sequences :
  ∀ (a b : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℝ),
  arithmetic_sequence a → geometric_sequence b →
  b 2 + (a 1 + a 2) = 7 →
  b 3 + (a 1 + a 2 + a 3) = 22 →
  (∀ n, c n = (2 ^ (n - 1) * a n) / b n) →
  (∀ n, T n = ∑ k in finset.range n, c k) →
  (∀ n, (-1) ^ n * m - T n < (n : ℝ) / 2^(n-1)) →
  -2 < m ∧ m < 3 :=
begin
  intros,
  sorry
end

end sum_geometric_arithmetic_sequences_l257_257235


namespace latest_temperature_85_is_9_temperature_at_9_is_85_l257_257691

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

-- Define the condition, t value for which temperature is 85 degrees
def is_temperature_85 (t : ℝ) : Prop := temperature t = 85

-- Prove the main statement, that the latest t with temperature 85 is 9
theorem latest_temperature_85_is_9 :
  ∀ t : ℝ, is_temperature_85 t → t ≤ 9 :=
begin
  assume t,
  intro ht,
  have hquadratic : -t^2 + 14 * t + 40 = 85 := ht,
  sorry -- proof steps are omitted
end

-- Prove that t=9 actually makes temperature 85
theorem temperature_at_9_is_85 : temperature 9 = 85 :=
begin
  unfold temperature,
  norm_num,
end

end latest_temperature_85_is_9_temperature_at_9_is_85_l257_257691


namespace drivers_sufficiency_and_ivan_petrovich_departure_l257_257033

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l257_257033


namespace A_investment_l257_257883

theorem A_investment (x : ℝ) (hx : 0 < x) :
  (∃ a b c d e : ℝ,
    a = x ∧ b = 12 ∧ c = 200 ∧ d = 6 ∧ e = 60 ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    ((a * b) / (a * b + c * d)) * 100 = e)
  → x = 150 :=
by
  sorry

end A_investment_l257_257883


namespace max_pairs_distinct_sums_l257_257221

theorem max_pairs_distinct_sums : 
  ∃ (k : ℕ), 
  k ≤ 3000 / 2 ∧ 
  (∀ i j, i ≠ j → a_i + b_i ≠ a_j + b_j) ∧ 
  (∀ i, a_i + b_i ≤ 3000) ∧ 
  (∀ i, 1 ≤ a_i < b_i ≤ 3000) ∧ 
  (∀ i j, {a_i, b_i} ∩ {a_j, b_j} = ∅) → 
  k ≤ 1199 := 
sorry

end max_pairs_distinct_sums_l257_257221


namespace number_of_possible_values_of_x_l257_257299

theorem number_of_possible_values_of_x : 
  (∃ x : ℕ, ⌈Real.sqrt x⌉ = 12) → (set.Ico 144 169).card = 25 := 
by
  intros h
  sorry

end number_of_possible_values_of_x_l257_257299


namespace determine_parallel_planes_l257_257728

variables {Plane : Type} {Line : Type} [IncidencePlane Plane Line]
variables {α β : Plane} {l m : Line}

-- Define the required notions and relations
def is_subset_of (l : Line) (α : Plane) : Prop := l ∈ α
def is_parallel_to (α β : Plane) : Prop := ∀ (l : Line), l ∈ β → l ∈ α ∨ (∃ m, m ∈ α ∧ m ∥ l)
def is_perpendicular_to (l : Line) (α : Plane) : Prop := l ∉ α

-- Problem Statement in Lean
theorem determine_parallel_planes
    (h1 : is_perpendicular_to l α)
    (h2 : is_perpendicular_to m β)
    (h3 : l ∥ m) :
  is_parallel_to α β :=
sorry

end determine_parallel_planes_l257_257728


namespace max_area_of_rectangular_pen_l257_257606

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l257_257606


namespace sum_of_first_50_digits_is_216_l257_257106

noncomputable def sum_first_50_digits_of_fraction : Nat :=
  let repeating_block := [0, 0, 0, 9, 9, 9]
  let full_cycles := 8
  let remaining_digits := [0, 0]
  let sum_full_cycles := full_cycles * (repeating_block.sum)
  let sum_remaining_digits := remaining_digits.sum
  sum_full_cycles + sum_remaining_digits

theorem sum_of_first_50_digits_is_216 :
  sum_first_50_digits_of_fraction = 216 := by
  sorry

end sum_of_first_50_digits_is_216_l257_257106


namespace minimum_product_sum_l257_257735

noncomputable def g (x : ℂ) : ℂ := x^4 + 18*x^3 + 97*x^2 + 18*x + 1

theorem minimum_product_sum :
  let roots := {z : ℂ | (∃ w : ℂ, w ∈ {x | g x = 0} ∧ z = w)} in
  ∃ (z1 z2 z3 z4 : ℂ) (h1 : z1 ∈ roots) (h2 : z2 ∈ roots) (h3 : z3 ∈ roots) (h4 : z4 ∈ roots),
  z1 ≠ z2 ∧ z1 ≠ z3 ∧ z1 ≠ z4 ∧ z2 ≠ z3 ∧ z2 ≠ z4 ∧ z3 ≠ z4 ∧
  ∀ (a b c d : ℂ),
  a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧ d ∈ roots ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  |a * b + c * d| = 2 :=
begin
  sorry
end

end minimum_product_sum_l257_257735


namespace no_such_natural_number_exists_l257_257539

theorem no_such_natural_number_exists :
  ¬ ∃ (n : ℕ), (∃ (m k : ℤ), 2 * n - 5 = 9 * m ∧ n - 2 = 15 * k) :=
by
  sorry

end no_such_natural_number_exists_l257_257539


namespace projection_of_a_onto_b_l257_257980

open Real

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-2, 4)

theorem projection_of_a_onto_b :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b_squared := vector_b.1 ^ 2 + vector_b.2 ^ 2
  let scalar_projection := dot_product / magnitude_b_squared
  let proj_vector := (scalar_projection * vector_b.1, scalar_projection * vector_b.2)
  proj_vector = (-4/5, 8/5) :=
by
  sorry

end projection_of_a_onto_b_l257_257980


namespace cost_of_insulation_l257_257499

def rectangular_tank_dimension_l : ℕ := 6
def rectangular_tank_dimension_w : ℕ := 3
def rectangular_tank_dimension_h : ℕ := 2
def total_cost : ℕ := 1440

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def cost_per_square_foot (total_cost surface_area : ℕ) : ℕ := total_cost / surface_area

theorem cost_of_insulation : 
  cost_per_square_foot total_cost (surface_area rectangular_tank_dimension_l rectangular_tank_dimension_w rectangular_tank_dimension_h) = 20 :=
by
  sorry

end cost_of_insulation_l257_257499


namespace find_minimum_a_l257_257262

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x

theorem find_minimum_a (a : ℝ) :
  (∀ x, 1 ≤ x → 0 ≤ 3 * x^2 + a) ↔ a ≥ -3 :=
by
  sorry

end find_minimum_a_l257_257262


namespace parallel_sides_trapezoid_l257_257006

noncomputable def trapezoid_parallel_sides (t m n : ℝ) (E : ℝ) : ℝ × ℝ :=
  let a := 24
  let c := 10
  if t = 204 ∧ m = 14 ∧ n = 2 ∧ E = 59 + 29/60 + 23/3600
  then (a, c)
  else (0, 0)

theorem parallel_sides_trapezoid
  (t m n E : ℝ)
  (ht : t = 204)
  (hm : m = 14)
  (hn : n = 2)
  (hE : E = 59 + 29/60 + 23/3600) :
  trapezoid_parallel_sides t m n E = (24, 10) :=
by
  unfold trapezoid_parallel_sides
  rw [ht, hm, hn, hE]
  rfl

end parallel_sides_trapezoid_l257_257006


namespace p_is_necessary_not_sufficient_l257_257676

-- Definitions based on the conditions
def p (x : ℝ) : Prop := (x^2 + 6*x + 8) * real.sqrt (x + 3) ≥ 0
def q (x : ℝ) : Prop := x = -3

-- Theorem statement
theorem p_is_necessary_not_sufficient : ∀ (x : ℝ), p (x) → q (x) :=
by
  -- Placeholder for the proof
  intro x hx
  sorry

end p_is_necessary_not_sufficient_l257_257676


namespace circular_garden_area_l257_257140

theorem circular_garden_area
  (r : ℝ) (h_r : r = 16)
  (C A : ℝ) (h_C : C = 2 * Real.pi * r) (h_A : A = Real.pi * r^2)
  (fence_cond : C = 1 / 8 * A) :
  A = 256 * Real.pi := by
  sorry

end circular_garden_area_l257_257140


namespace slower_train_speed_l257_257820

theorem slower_train_speed : ∀ (V : ℝ) (time : ℝ) (length1 length2 : ℝ) (speed_faster : ℝ),
  time = 19.6347928529354 →
  length1 = 500 →
  length2 = 700 →
  speed_faster = 120 →
  length1 + length2 = (V + speed_faster) * (1000 / 3600) * time → 
  V ≈ 100 :=
begin
  intros V time length1 length2 speed_faster h_time h_length1 h_length2 h_speed_faster h_eq,
  sorry
end

end slower_train_speed_l257_257820


namespace measure_of_AB_l257_257338

theorem measure_of_AB (α β a b : ℝ) (A B C D E : Type)
  [normed_group A] [normed_group B] [normed_group C] [normed_group D] [normed_group E]
  (AB CD AD DE : ℝ)
  (h1 : AB ∥ CD)
  (h2 : ∠D = 3 * ∠B)
  (h3 : AD = a)
  (h4 : CD = b) :
  AB = a + b := 
sorry

end measure_of_AB_l257_257338


namespace find_ellipse_equation_max_triangle_area_diff_l257_257994

-- Definition of the ellipse with given conditions
def ellipse_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h : a > b) :=
  ∀ x y : ℝ, x = 1 → y = - (Real.sqrt 3) / 2 → (x^2 / a^2 + y^2 / b^2 = 1)

-- Eccentricity condition
def eccentricity_condition (a b c : ℝ) :=
  c = a * (Real.sqrt 3) / 2 ∧ a^2 - b^2 = c^2

-- Equation of the ellipse given the contextual conditions
def ellipse_Γ_equation (x y : ℝ) :=
  (x^2 / 4) + y^2 = 1

-- Proof that the given conditions lead to the equation of the ellipse
theorem find_ellipse_equation (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h : a > b)
  (passes_through_P : ellipse_equation a b a_pos b_pos h)
  (eccentricity_given : eccentricity_condition a b c) :
  (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

-- Definition of areas of triangles ΔABC and ΔABD
def triangle_area_diff (xa ya xb yb xc yc xd yd : ℝ) (A B C D : ℝ) :=
  let S1 := abs ((xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb)) / 2) in
  let S2 := abs ((xa * (yb - yd) + xb * (yd - ya) + xd * (ya - yb)) / 2) in
  abs (S1 - S2)

-- Maximum value of |S1 - S2|
theorem max_triangle_area_diff :
  ∃ ε : ℝ, ε = Real.sqrt 3 ∧
  ∀ k : ℝ, k ≠ 0 →
  ∃ F A B C D : ℝ,
  triangle_area_diff A B C D (- (Real.sqrt 3)) 0 (k * (- (Real.sqrt 3))) 0 ε = ε :=
sorry

end find_ellipse_equation_max_triangle_area_diff_l257_257994


namespace batsman_average_after_17th_inning_l257_257473

-- Definitions for the conditions
def runs_scored_in_17th_inning : ℝ := 95
def increase_in_average : ℝ := 2.5

-- Lean statement encapsulating the problem
theorem batsman_average_after_17th_inning (A : ℝ) (h : 16 * A + runs_scored_in_17th_inning = 17 * (A + increase_in_average)) :
  A + increase_in_average = 55 := 
sorry

end batsman_average_after_17th_inning_l257_257473


namespace count_integer_values_of_x_l257_257290

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : {n : ℕ | 121 < n ∧ n ≤ 144}.card = 23 := 
by
  sorry

end count_integer_values_of_x_l257_257290


namespace distinct_integer_values_b_for_quadratic_l257_257570

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end distinct_integer_values_b_for_quadratic_l257_257570


namespace sum_of_squares_l257_257755

theorem sum_of_squares (n : ℕ) :
  (∃ a b : ℕ, n = a^2 + b^2) ↔
  (∀ p ∈ factorization n, (p % 4 = 3 → even (factorization n p))) ∧
  (∃ a b : ℕ, n = a^2 + b^2 ∧ gcd a b = 1 ↔ (∀ p ∈ factorization n, p % 4 ≠ 3) ∧ ¬(4 ∣ n)) :=
  sorry

end sum_of_squares_l257_257755


namespace number_of_solutions_l257_257970

theorem number_of_solutions : 
  {n : ℤ // 2 ≤ n ∧ n < 7.5 ∧ (real.sqrt n ≤ real.sqrt (5 * n - 8)) ∧ (real.sqrt (5 * n - 8) < real.sqrt (3 * n + 7))}.card = 6 :=
by sorry

end number_of_solutions_l257_257970


namespace maximize_pasture_area_l257_257892

theorem maximize_pasture_area :
  ∃ (length_parallel_500ft_side : ℝ),
    length_parallel_500ft_side = 100 ∧
    ∀ (y : ℝ) (fence_length : ℝ) (fence_cost : ℝ) (total_cost : ℝ),
       fence_length = 200 ∧ fence_cost = 10 ∧ total_cost = 2000 →
       (fence_length = (2000 / 10)) ∧ (fence_length = 200) →
       let side_parallel := 200 - 2 * y in
       let area := y * side_parallel in
       area ≤ (50 * (200 - 2 * 50)) :=
begin
  sorry,
end

end maximize_pasture_area_l257_257892


namespace correct_system_l257_257688

theorem correct_system (x y : ℕ) (h1 : y + 1 = 2 * (x - 1)) (h2 : y - 1 = x + 1) :
  y + 1 = 2 * (x - 1) ∧ y - 1 = x + 1 :=
by {
  exact ⟨h1, h2⟩,
  sorry  -- Proof is omitted as per instructions
}

end correct_system_l257_257688


namespace exp_addition_property_l257_257015

theorem exp_addition_property (x y : ℝ) : (Real.exp (x + y)) = (Real.exp x) * (Real.exp y) := 
sorry

end exp_addition_property_l257_257015


namespace correct_propositions_l257_257654

-- Definitions according to the given conditions
def generatrix_cylinder (p1 p2 : Point) (c : Cylinder) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def generatrix_cone (v : Point) (p : Point) (c : Cone) : Prop :=
  -- Check if the line from the vertex to a base point is a generatrix
  sorry

def generatrix_frustum (p1 p2 : Point) (f : Frustum) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def parallel_generatrices_cylinder (gen1 gen2 : Line) (c : Cylinder) : Prop :=
  -- Check if two generatrices of the cylinder are parallel
  sorry

-- The theorem stating propositions ② and ④ are correct
theorem correct_propositions :
  generatrix_cone vertex point cone ∧
  parallel_generatrices_cylinder gen1 gen2 cylinder :=
by
  sorry

end correct_propositions_l257_257654


namespace sequences_same_period_length_l257_257864

theorem sequences_same_period_length
  (v : ℕ → ℝ)
  (v' : ℕ → ℝ)
  (t : ℕ → ℝ := λ n, if n = 0 then v 1 / v 0 else v (n + 1) / v n)
  (t' : ℕ → ℝ := λ n, if n = 0 then v' 1 / v' 0 else v' (n + 1) / v' n)
  (h1 : ¬(t 0)^2 - (t 0) - 1 = 0)
  (h2 : ¬(t' 0)^2 - (t' 0) - 1 = 0) :
  ∃ r, ∀ n, r > 0 → t (n + r) = t n ∧ t' (n + r) = t' n := 
sorry

end sequences_same_period_length_l257_257864


namespace construct_triangle_if_l257_257532

variable (α m ρ : ℝ) -- Given parameters: alpha (angle), m (altitude), and ρ (radius of the inscribed circle)

-- Hypotheses or conditions
axiom (h1 : α > 0)          -- Condition that alpha is a positive angle
axiom (h2 : m > 0)          -- Condition that m is a positive segment
axiom (h3 : ρ > 0)          -- Condition that ρ is a positive radius

-- Theorem statement
theorem construct_triangle_if (h4 : m > 2 * ρ) : 
  ∃ (A B C : Point), ∃ (O : Point), InscribedCircle A B C O ρ ∧ VertexAngle A B C α ∧ Altitude A B C m :=
sorry

end construct_triangle_if_l257_257532


namespace largest_inscribed_square_length_l257_257712

noncomputable def inscribed_square_length (s : ℝ) (n : ℕ) : ℝ :=
  let t := s / n
  let h := (Real.sqrt 3 / 2) * t
  s - 2 * h

theorem largest_inscribed_square_length :
  inscribed_square_length 12 3 = 12 - 4 * Real.sqrt 3 :=
by
  sorry

end largest_inscribed_square_length_l257_257712


namespace ratio_llamas_to_goats_l257_257440

def cost_per_goat : ℕ := 400
def num_goats : ℕ := 3
def total_spent : ℕ := 4800

noncomputable def cost_per_llama : ℕ := cost_per_goat + (cost_per_goat / 2)
noncomputable def num_llamas : ℕ := (total_spent - (num_goats * cost_per_goat)) / cost_per_llama

theorem ratio_llamas_to_goats : (num_llamas : ℕ) / num_goats = 2 := by
  have h1 : num_llamas = 6 := by
    rw [num_llamas, cost_per_llama, cost_per_goat]
    norm_num
  have h2 : num_goats = 3 := by
    norm_num
  rw [h1, h2]
  norm_num

end ratio_llamas_to_goats_l257_257440


namespace gcf_45_135_90_l257_257084

def gcd (a b : Nat) : Nat := Nat.gcd a b

noncomputable def gcd_of_three (a b c : Nat) : Nat :=
  gcd (gcd a b) c

theorem gcf_45_135_90 : gcd_of_three 45 135 90 = 45 := by
  sorry

end gcf_45_135_90_l257_257084


namespace not_parallel_to_both_skew_lines_l257_257133

variables {a b c : Line}

-- Definition: Skew lines
def skew (a b : Line) : Prop :=
  ¬(a ∥ b) ∧ ¬∃ p, p ∈ a ∧ p ∈ b  -- Lines a and b are not parallel and do not intersect.

-- Definition: Parallel lines
def parallel (a b : Line) : Prop := a ∥ b

-- Theorem: Line c cannot be parallel to both skew lines a and b
theorem not_parallel_to_both_skew_lines (h_skew : skew a b) : ¬(parallel c a ∧ parallel c b) :=
sorry

end not_parallel_to_both_skew_lines_l257_257133


namespace trapezoid_area_correct_l257_257825

noncomputable def trapezoid_area : ℝ :=
  let base1 := 4 in  -- Length of base 1
  let base2 := 1 in  -- Length of base 2
  let height := 6 in -- Height between y = 8 and y = 2
  0.5 * (base1 + base2) * height

theorem trapezoid_area_correct : trapezoid_area = 15 :=
  sorry

end trapezoid_area_correct_l257_257825


namespace intersection_of_lines_l257_257942

theorem intersection_of_lines :
  ∃ (x y : ℚ), y = -3 * x + 1 ∧ y + 4 = 15 * x - 2 ∧ x = 7 / 18 ∧ y = -1 / 6 :=
begin
  sorry
end

end intersection_of_lines_l257_257942


namespace rectangular_box_inscribed_in_sphere_l257_257890

theorem rectangular_box_inscribed_in_sphere (
  (a b c : ℝ)
  (h₁ : a + b + c = 40)
  (h₂ : 2 * a * b + 2 * b * c + 2 * c * a = 512)
  (h₃ : ∀ r : ℝ, (2 * r) ^ 2 = a^2 + b^2 + c^2)
  (r : ℝ)
  (h₄ : (a^2 + b^2 + c^2) = 1088)
) : r = sqrt 130 :=
by {
  sorry
}

end rectangular_box_inscribed_in_sphere_l257_257890


namespace gray_area_l257_257819

theorem gray_area (A B : ℕ) (C : ℕ) (black_area : ℕ) 
  (hA : A = 8 * 10)
  (hB : B = 12 * 9)
  (hBlack : black_area = 37) :
  C = 65 :=
by
  have A_white : A = 80, from hA
  have B_gray : B = 108, from hB
  have remaining_white := A_white - black_area
  have C_area := B_gray - remaining_white
  exact Eq.trans C_area (by rw [remaining_white, hBlack])
  sorry

end gray_area_l257_257819


namespace f_direct_prop_g_inverse_prop_f_g_odd_l257_257468

noncomputable def f (x : ℝ) := x
noncomputable def g (x : ℝ) := 2 / x

theorem f_direct_prop : (∀ k : ℝ, f 1 = 1 → f x = k * x) :=
by sorry

theorem g_inverse_prop : (∀ k : ℝ, g 1 = 2 → g x = k / x) :=
by sorry

theorem f_g_odd : (∀ x : ℝ, f x + g x) = -(f (-x) + g (-x)) :=
by sorry

end f_direct_prop_g_inverse_prop_f_g_odd_l257_257468


namespace find_triples_l257_257504

def is_solution (x y : ℕ) (z : ℤ) : Prop :=
  x! + y! = 24 * z + 2017

def is_odd (z : ℤ) : Prop := z % 2 ≠ 0

theorem find_triples :
  (∀ x y : ℕ, ∀ z : ℤ, is_solution x y z → is_odd z) →
  (
  (is_solution 1 4 (-83) ∧ is_solution 4 1 (-83) ∧
   is_solution 1 5 (-79) ∧ is_solution 5 1 (-79))
  ) :=
by sorry

end find_triples_l257_257504


namespace evaluate_expression_l257_257950

theorem evaluate_expression : 
  2 * log 5 10 + log 5 0.25 + (8 ^ 0.25) * (2 ^ (1/4)) = 4 := 
by
  -- Proof is omitted
  sorry

end evaluate_expression_l257_257950


namespace max_load_truck_l257_257143

theorem max_load_truck (bag_weight : ℕ) (num_bags : ℕ) (remaining_load : ℕ) 
  (h1 : bag_weight = 8) (h2 : num_bags = 100) (h3 : remaining_load = 100) : 
  bag_weight * num_bags + remaining_load = 900 :=
by
  -- We leave the proof step intentionally, as per instructions.
  sorry

end max_load_truck_l257_257143


namespace angle_of_inclination_of_line_l257_257005

theorem angle_of_inclination_of_line (x y : ℝ) (h : x - y - 1 = 0) : 
  ∃ α : ℝ, α = π / 4 := 
sorry

end angle_of_inclination_of_line_l257_257005


namespace tan_cos_mul_expr_l257_257527

theorem tan_cos_mul_expr :
  tan 70 * cos 10 * (sqrt 3 * tan 20 - 1) = -1 :=
by
  sorry

end tan_cos_mul_expr_l257_257527


namespace count_even_4digit_between_5000_and_8000_l257_257671

theorem count_even_4digit_between_5000_and_8000 : 
  ∃ (n : ℕ), 98 = (finset.range (8000 - 5000)).filter
    (λ x, let n := x + 5000 in
           (n % 5 = 0) ∧ 
           (n % 2 = 0) ∧ 
           (n / 1000 ≠ (n / 100 % 10)) ∧ 
           (n / 1000 ≠ (n / 10 % 10)) ∧ 
           (n / 1000 ≠ (n % 10)) ∧ 
           ((n / 100 % 10) ≠ (n / 10 % 10)) ∧ 
           ((n / 100 % 10) ≠ (n % 10)) ∧ 
           ((n / 10 % 10) ≠ (n % 10))).card :=
sorry

end count_even_4digit_between_5000_and_8000_l257_257671


namespace combined_value_l257_257308

theorem combined_value (a b c : ℕ) (h1 : 0.005 * a = 0.95)
  (h2 : b = 3 * a - 50)
  (h3 : c = (a - b)^2)
  (h_pos_a : 0 < a) (h_pos_c : 0 < c) : 
  a + b + c = 109610 :=
sorry

end combined_value_l257_257308


namespace relationship_m_n_p_l257_257226

noncomputable def m : ℝ := (1 / 2) ^ (-2)
noncomputable def n : ℝ := (-2) ^ 3
noncomputable def p : ℝ := -(-(1 / 2) ^ 0)

theorem relationship_m_n_p : n < p ∧ p < m :=
by {
  have hm : m = 4 := by sorry,
  have hn : n = -8 := by sorry,
  have hp : p = 1 := by sorry,
  rw [hm, hn, hp],
  exact ⟨by norm_num, by norm_num⟩
}

end relationship_m_n_p_l257_257226


namespace part1_monotonicity_n2_part2_unique_solution_f_eq_0_l257_257593

open Real

noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  exp x - (Finset.range (n + 1)).sum (λ k, x^k / k.factorial)

theorem part1_monotonicity_n2 :
  (∀ x : ℝ, f x 2 = exp x - (1 + x + x^2 / 2)) ∧ (∀ x : ℝ, deriv (λ x, f x 2) x > 0) :=
by sorry

theorem part2_unique_solution_f_eq_0 (n : ℕ) :
  ∃! x : ℝ, f x n = 0 :=
by sorry

end part1_monotonicity_n2_part2_unique_solution_f_eq_0_l257_257593


namespace transformed_function_correct_l257_257001

-- Define the original function
def original_function (x : ℝ) : ℝ := sin (2 * x)

-- Define the transformation: stretching the abscissa by a factor of 2
def stretched_function (x : ℝ) : ℝ := sin (x)

-- Define the final transformation: shifting to the right by π/4
def transformed_function (x : ℝ) : ℝ := sin (x - π / 4)

-- The statement to prove in Lean
theorem transformed_function_correct :
  ∀ x : ℝ, transformed_function x = sin (x - π / 4) :=
by
  intros x
  -- Proof skipping with sorry
  sorry

end transformed_function_correct_l257_257001


namespace probability_same_color_socks_l257_257360

-- Define the total number of socks and the groups
def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

-- Define combinatorial functions to calculate combinations
def comb (n m : ℕ) : ℕ := n.choose m

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  comb blue_socks 2 +
  comb green_socks 2 +
  comb red_socks 2

-- Calculate the total number of possible outcomes
def total_outcomes : ℕ := comb total_socks 2

-- Calculate the probability as a ratio of favorable outcomes to total outcomes
def probability := favorable_outcomes / total_outcomes

-- Prove the probability is 19/45
theorem probability_same_color_socks : probability = 19 / 45 := by
  sorry

end probability_same_color_socks_l257_257360


namespace problem_statement_l257_257585

noncomputable def incorrect_statement_D (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : Prop :=
  ∀ c : ℝ, (sqrt a + sqrt b = c) → (c >= sqrt 2 → false)

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : incorrect_statement_D a b h1 h2 h3 :=
sorry

end problem_statement_l257_257585


namespace integer_values_of_b_l257_257564

theorem integer_values_of_b (b : ℤ) : 
  (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0 ∧ x ≠ y) → 
  ∃ S : finset ℤ, S.card = 8 ∧ ∀ c ∈ S, ∃ x : ℤ, x^2 + c * x + 12 * c = 0 :=
sorry

end integer_values_of_b_l257_257564


namespace max_area_of_rectangle_with_perimeter_60_l257_257623

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l257_257623


namespace first_player_wins_l257_257464

theorem first_player_wins :
  ∀ moves: ℕ → ℕ, ∀ strip: Fin (2005 + 1), 
  (∀ k: ℕ, moves k = 2^k) →
  (∀ pos: Fin (2005 + 1), pos = 1003 → 
    (∃ k: ℕ, pos.val + moves k ≤ 2005 ∧ pos.val - moves k ≥ 1)) →
  ∃ (win_strategy: Fin (2005 + 1) → ℕ), 
    win_strategy 1003 = 1 :=
begin
  sorry
end

end first_player_wins_l257_257464


namespace angle_equality_l257_257379

open EuclideanGeometry

noncomputable def common_tangent_condition := sorry -- Define the conditions mathematically in Lean

theorem angle_equality
  (O_1 O_2 A P_1 P_2 Q_1 Q_2 M_1 M_2 : Point)
  (h1 : Circle O_1 ≠ Circle O_2)
  (h2 : A ∈ (Circle O_1 ∩ Circle O_2))
  (h3 : common_tangent_condition O_1 O_2 P_1 P_2 Q_1 Q_2)
  (h4 : midpoint P_1 Q_1 = M_1)
  (h5 : midpoint P_2 Q_2 = M_2) :
  ∠ O_1 A O_2 = ∠ M_1 A M_2 :=
sorry

end angle_equality_l257_257379


namespace max_rectangle_area_l257_257628

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l257_257628


namespace implies_neg_p_and_q_count_l257_257937

-- Definitions of the logical conditions
variables (p q : Prop)

def cond1 : Prop := p ∧ q
def cond2 : Prop := p ∧ ¬ q
def cond3 : Prop := ¬ p ∧ q
def cond4 : Prop := ¬ p ∧ ¬ q

-- Negative of the statement "p and q are both true"
def neg_p_and_q := ¬ (p ∧ q)

-- The Lean 4 statement to prove
theorem implies_neg_p_and_q_count :
  (cond2 p q → neg_p_and_q p q) ∧ 
  (cond3 p q → neg_p_and_q p q) ∧ 
  (cond4 p q → neg_p_and_q p q) ∧ 
  ¬ (cond1 p q → neg_p_and_q p q) :=
sorry

end implies_neg_p_and_q_count_l257_257937


namespace sum_seq_lt_one_l257_257052

noncomputable def seq : ℕ → ℚ
| 0       := 1/2
| (n + 1) := seq n ^ 2 / (seq n ^ 2 - seq n + 1)

theorem sum_seq_lt_one (n : ℕ) : (finset.range (n + 1)).sum seq < 1 :=
sorry

end sum_seq_lt_one_l257_257052


namespace rationalize_denominator_l257_257756

theorem rationalize_denominator
  (A B C D E : ℤ)
  (h1 : 4\sqrt{6} + 3\sqrt{7} ≠ 0)
  (h2 : \(\frac{3}{4\sqrt{6} + 3\sqrt{7}} = \frac{A\sqrt{B} + C\sqrt{D}}{E}\))
  (h3 : B < D) : 
  (A + B + C + D + E = 25) :=
sorry

end rationalize_denominator_l257_257756


namespace solve_for_x_l257_257764

theorem solve_for_x (x : ℕ) : 100^3 = 10^x → x = 6 := by
  sorry

end solve_for_x_l257_257764


namespace exponential_quotient_l257_257596

variable {x a b : ℝ}

theorem exponential_quotient (h1 : x^a = 3) (h2 : x^b = 5) : x^(a-b) = 3 / 5 :=
sorry

end exponential_quotient_l257_257596


namespace exterior_angle_bisectors_of_quadrilateral_l257_257697

theorem exterior_angle_bisectors_of_quadrilateral
  (A B C D : ℝ)
  (hA : A > 0) (hB : B > 0) (hC : C > 0) (hD : D > 0)
  (sum_of_angles : A + B + C + D = 360) :
  let E := (360 - (B + D)) / 2
  in E = (360 - (B + D)) / 2 :=
by
  sorry

end exterior_angle_bisectors_of_quadrilateral_l257_257697


namespace max_C_eq_pi_div_3_l257_257243

noncomputable def max_angle_C (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if (cos A * sin B * sin C + cos B * sin A * sin C = 2 * cos C * sin A * sin B) then
    max C
  else
    0 -- default value, the actual equation ensures non-zero

theorem max_C_eq_pi_div_3 
  (a b c A B C : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : A + B + C = π)
  (h5 : cos A * sin B * sin C + cos B * sin A * sin C = 2 * cos C * sin A * sin B):
  max_angle_C a b c A B C = π / 3 := sorry

end max_C_eq_pi_div_3_l257_257243


namespace brenda_trays_l257_257180

-- Define main conditions
def cookies_per_tray : ℕ := 80
def cookies_per_box : ℕ := 60
def cost_per_box : ℕ := 350
def total_cost : ℕ := 1400  -- Using cents for calculation to avoid float numbers

-- State the problem
theorem brenda_trays :
  (total_cost / cost_per_box) * cookies_per_box / cookies_per_tray = 3 := 
by
  sorry

end brenda_trays_l257_257180


namespace total_puppies_adopted_l257_257847

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l257_257847


namespace f_conjecture_l257_257592

def f (x : ℝ) : ℝ := 1 / (3^x + real.sqrt 3)

theorem f_conjecture (x : ℝ) : f(x) + f(1 - x) = real.sqrt 3 / 3 := by
  sorry

end f_conjecture_l257_257592


namespace complex_number_value_l257_257229

open Complex

theorem complex_number_value (a : ℝ) 
  (h1 : z = (2 + a * I) / (1 + I)) 
  (h2 : (z.re, z.im) ∈ { p : ℝ × ℝ | p.2 = -p.1 }) : 
  a = 0 :=
by
  sorry

end complex_number_value_l257_257229


namespace minimize_PA_PB_l257_257010

theorem minimize_PA_PB 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (5, 1)) : 
  ∃ P : ℝ × ℝ, P = (4, 0) ∧ 
  ∀ P' : ℝ × ℝ, P'.snd = 0 → (dist P A + dist P B) ≤ (dist P' A + dist P' B) :=
sorry

end minimize_PA_PB_l257_257010


namespace sum_g_equals_1000_l257_257382

def g (x : ℝ) : ℝ := 4 / (8^x + 4) -- Define the function g(x)

theorem sum_g_equals_1000 : 
  ∑ k in Finset.range 2000, g (k.succ / 2001 : ℝ) = 1000 :=
begin
  sorry -- Proof placeholder
end

end sum_g_equals_1000_l257_257382


namespace max_product_l257_257101

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l257_257101


namespace definite_integral_result_l257_257130

theorem definite_integral_result : 
  (∫ x in real.pi / 2..2 * real.arctan 2, 1 / (real.sin x * (1 + real.sin x))) = real.log 2 - (1 / 3) :=
by 
  sorry

end definite_integral_result_l257_257130


namespace different_graphs_l257_257108

def eq1 (x : ℝ) : ℝ := x - 3
def eq2 (x : ℝ) : ℝ := if x ≠ -3 then (x^2 - 9) / (x + 3) else 0
def eq3 (x : ℝ) : ℝ := if x ≠ -3 then (x^2 - 9) / (x + 3) else arbitrary ℝ

theorem different_graphs :
  ¬ ( ∀ x, eq1 x = eq2 x ) ∧
  ¬ ( ∀ x, eq1 x = eq3 x ) ∧
  ¬ ( ∀ x, eq2 x = eq3 x ) :=
by
  sorry

end different_graphs_l257_257108


namespace fixed_point_of_line_l257_257554

theorem fixed_point_of_line (m : ℝ) : 
  (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  sorry

end fixed_point_of_line_l257_257554


namespace sum_of_coordinates_of_reflected_midpoint_is_one_l257_257752

noncomputable def P : (ℝ × ℝ) := (2, 1)
noncomputable def R : (ℝ × ℝ) := (12, 15)
noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def reflect_over_y (A : (ℝ × ℝ)) : (ℝ × ℝ) := 
  (-A.1, A.2)

noncomputable def P' : (ℝ × ℝ) := reflect_over_y P
noncomputable def R' : (ℝ × ℝ) := reflect_over_y R
noncomputable def M' : (ℝ × ℝ) := midpoint P' R'

theorem sum_of_coordinates_of_reflected_midpoint_is_one :
  M'.1 + M'.2 = 1 :=
by
  sorry

end sum_of_coordinates_of_reflected_midpoint_is_one_l257_257752


namespace number_of_correct_statements_l257_257936

theorem number_of_correct_statements :
  (statement_1 : x^2 + y^2 - 2 * x + 4 * y + 6 = 0 → (¬∃ r : ℝ, r > 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2)) →
  (statement_2 : ∀ m n : ℝ, m > n ∧ n > 0 → mx^2 + ny^2 = 1 → ∃ a b : ℝ, a ≠ b ∧ 1/m = a^2 ∧ 1/n = b^2) →
  (statement_3 : ∀ P : ℝ × ℝ, |(P.1 + 1) + P.2| - |(P.1 - 1) + P.2| = 2 → ¬(P.1^2 - P.2^2 = 1)) →
  (statement_4 : (∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, ((x - x0)^2 + (y - y0)^2 = c^2) implies tangent (directrix parabola))) →
  2 :=
by
  sorry

end number_of_correct_statements_l257_257936


namespace count_multiples_200_to_400_l257_257283

def count_multiples_in_range (a b n : ℕ) : ℕ :=
  (b / n) - ((a + n - 1) / n) + 1

theorem count_multiples_200_to_400 :
  count_multiples_in_range 200 400 78 = 3 :=
by
  sorry

end count_multiples_200_to_400_l257_257283


namespace conversion_rate_l257_257917

variable (Wi : ℝ) (Wf : ℝ) (h : ℝ) (r : ℝ) (d : ℕ)

def initial_weight_kg := 80
def final_weight_lb := 134
def hours_per_day := 2
def weight_loss_rate := 1.5
def exercise_days := 14

theorem conversion_rate (h_eq : h = hours_per_day) (r_eq : r = weight_loss_rate) (d_eq : d = exercise_days) (Wi_eq : Wi = initial_weight_kg) (Wf_eq : Wf = final_weight_lb): 
  (Wi * 2.2 = 176) := 
by 
  have total_loss := h * r * d
  have initial_weight_lb := Wf + total_loss
  have conv_rate := initial_weight_lb / Wi
  sorry

end conversion_rate_l257_257917


namespace solve_system_of_equations_l257_257408

theorem solve_system_of_equations :
  ∀ x y z : ℝ,
  (3 * x * y - 5 * y * z - x * z = 3 * y) →
  (x * y + y * z = -y) →
  (-5 * x * y + 4 * y * z + x * z = -4 * y) →
  (x = 2 ∧ y = -1 / 3 ∧ z = -3) ∨ 
  (y = 0 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l257_257408


namespace find_d_q_l257_257246

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def b_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  b1 * q^(n - 1)

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) / 2) * d

-- Sum of the first n terms of a geometric sequence
noncomputable def T_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  if q = 1 then n * b1
  else b1 * (1 - q^n) / (1 - q)

theorem find_d_q (a1 b1 d q : ℕ) (h1 : ∀ n : ℕ, n > 0 →
  n^2 * (T_n b1 q n + 1) = 2^n * S_n a1 d n) : d = 2 ∧ q = 2 :=
by
  sorry

end find_d_q_l257_257246


namespace length_of_diagonal_l257_257210

theorem length_of_diagonal (area : ℝ) (h1 h2 : ℝ) (d : ℝ) 
  (h_area : area = 75)
  (h_offsets : h1 = 6 ∧ h2 = 4) :
  d = 15 :=
by
  -- Given the conditions and formula, we can conclude
  sorry

end length_of_diagonal_l257_257210


namespace area_of_folded_shape_is_two_units_squared_l257_257162

/-- 
A square piece of paper with each side of length 2 units is divided into 
four equal squares along both its length and width. From the top left corner to 
bottom right corner, a line is drawn through the center dividing the square diagonally.
The paper is folded along this line to form a new shape.
We prove that the area of the folded shape is 2 units².
-/
theorem area_of_folded_shape_is_two_units_squared
  (side_len : ℝ)
  (area_original : ℝ)
  (area_folded : ℝ)
  (h1 : side_len = 2)
  (h2 : area_original = side_len * side_len)
  (h3 : area_folded = area_original / 2) :
  area_folded = 2 := by
  -- Place proof here
  sorry

end area_of_folded_shape_is_two_units_squared_l257_257162


namespace range_f_prime_one_l257_257256

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ :=
  (sin θ / 3) * x^3 + (sqrt 3 * cos θ / 2) * x^2 + tan θ

def f_prime (θ : ℝ) (x : ℝ) : ℝ :=
  (sin θ) * x^2 + (sqrt 3 * cos θ) * x

theorem range_f_prime_one (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ (5 * π / 12)) :
  sqrt 2 ≤ f_prime θ 1 ∧ f_prime θ 1 ≤ 2 :=
sorry

end range_f_prime_one_l257_257256


namespace equality_of_floor_squares_l257_257131

theorem equality_of_floor_squares (n : ℕ) (hn : 0 < n) :
  (⌊Real.sqrt n + Real.sqrt (n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  (⌊Real.sqrt (4 * n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  (⌊Real.sqrt (4 * n + 2)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 3)⌋ :=
by
  sorry

end equality_of_floor_squares_l257_257131


namespace Jaco_budget_for_parents_gifts_l257_257349

theorem Jaco_budget_for_parents_gifts :
  ∃ (m n : ℕ), (m = 14 ∧ n = 14) ∧ 
  (∀ (friends gifts_friends budget : ℕ), 
   friends = 8 → gifts_friends = 9 → budget = 100 → 
   (budget - (friends * gifts_friends)) / 2 = m ∧ 
   (budget - (friends * gifts_friends)) / 2 = n) := 
sorry

end Jaco_budget_for_parents_gifts_l257_257349


namespace int_solutions_eq_count_int_values_b_l257_257558

theorem int_solutions_eq (b : ℤ) : 
  ∃! x : ℤ, ∃! y : ℤ, (x + y = -b) ∧ (x * y = 12 * b) \/
  (x + y = -b) ∧ (x * y = 12 * b) :=
begin
  -- Assume roots p, q exist
  -- Use Vieta's formulas: p + q = -b, pq = 12b
  -- Transform the equation using SFFT
  sorry
end

theorem count_int_values_b :
  set_finite {b : ℤ | ∃! x : ℤ, ∃! y : ℤ, 
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} ∧
  fintype.card {b : ℤ | ∃! x : ℤ, ∃! y : ℤ,
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} = 16 :=
begin
  sorry
end

end int_solutions_eq_count_int_values_b_l257_257558


namespace least_integer_value_satisfying_inequality_l257_257962

theorem least_integer_value_satisfying_inequality : ∃ x : ℤ, 3 * |x| + 6 < 24 ∧ (∀ y : ℤ, 3 * |y| + 6 < 24 → x ≤ y) :=
  sorry

end least_integer_value_satisfying_inequality_l257_257962


namespace trip_total_time_l257_257874

variable (T : ℝ) -- Total time of the trip
variable (x : ℝ) -- Additional hours after the first 4 hours

-- Conditions
def speed_for_first_4_hours := 70
def time_for_first_4_hours := 4
def distance_first_part := speed_for_first_4_hours * time_for_first_4_hours

def speed_for_additional_hours := 60
def distance_additional_part := speed_for_additional_hours * x

def total_distance := distance_first_part + distance_additional_part

def average_speed := 65
def total_time := 4 + x

-- Proof statement
theorem trip_total_time : total_time = T → total_distance = average_speed * total_time → T = 8 :=
by
  intros h1 h2
  sorry

end trip_total_time_l257_257874


namespace n_divisible_by_6_l257_257411

open Int -- Open integer namespace for convenience

theorem n_divisible_by_6 (m n : ℤ)
    (h1 : ∃ (a b : ℤ), a + b = -m ∧ a * b = -n)
    (h2 : ∃ (c d : ℤ), c + d = m ∧ c * d = n) :
    6 ∣ n := 
sorry

end n_divisible_by_6_l257_257411


namespace area_triangle_QPO_l257_257695

-- Given conditions definitions
variables (A B C D P Q N S M R O : Type)
variables [parallelogram ABCD : Prop]
variables [trisects DP BC N S : Prop]
variables [meets_point DP AB_extension P : Prop]
variables [trisects CQ AD M R : Prop]
variables [meets_point CQ AB_extension Q : Prop]
variables [intersection DP CQ O : Prop]
variables (k : ℝ)

-- Statement to be proven
theorem area_triangle_QPO (A B C D P Q N S M R O : Type) 
  [parallelogram ABCD] 
  [trisects DP BC N S] 
  [meets_point DP AB_extension P] 
  [trisects CQ AD M R] 
  [meets_point CQ AB_extension Q] 
  [intersection DP CQ O] 
  (area_ABCD : ℝ) : 
  area_triangle QPO = (8 / 9) * area_ABCD := 
sorry

end area_triangle_QPO_l257_257695


namespace domain_of_f_l257_257197

noncomputable def f (x : ℝ) : ℝ := (2*x + 3) / Real.sqrt (3*x - 9)

theorem domain_of_f : ∀ x : ℝ, (3 < x) ↔ (∃ y : ℝ, f y ≠ y) :=
by
  sorry

end domain_of_f_l257_257197


namespace sufficient_money_l257_257166

-- Given conditions
def S : ℝ -- Sum of money
def x_days : ℝ := 36
def y_days : ℝ := 45
def z_days : ℝ := 60

-- Daily wages
def x_daily_wage := S / x_days
def y_daily_wage := S / y_days
def z_daily_wage := S / z_days

-- Combined daily wage of all three workers
def combined_daily_wage := x_daily_wage + y_daily_wage + z_daily_wage

-- Prove the money can pay for the combined daily wage for 15 days
theorem sufficient_money {S : ℝ} :
  (S / x_days) + (S / y_days) + (S / z_days) = S / 15 := 
by 
sorry

end sufficient_money_l257_257166


namespace find_m_l257_257736

/-
Define the ellipse equation
-/
def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2) = 1

/-
Define the region R
-/
def region_R (x y : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (2*y = x) ∧ ellipse_eqn x y

/-
Define the region R'
-/
def region_R' (x y m : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (y = m*x) ∧ ellipse_eqn x y

/-
The statement we want to prove
-/
theorem find_m (m : ℝ) : (∃ (x y : ℝ), region_R x y) ∧ (∃ (x y : ℝ), region_R' x y m) →
(m = (2 : ℝ) / 9) := 
sorry

end find_m_l257_257736


namespace billy_can_finish_3_books_l257_257517

-- Definitions based on given conditions
def reading_speed (n: Nat) : ℚ :=
  60 * (0.9 ^ n)

def time_to_read_book (n: Nat) : ℚ :=
  80 / reading_speed n

def total_time_to_read_books (k: Nat) : ℚ :=
  (List.range k).sum (λ n, time_to_read_book n)

-- The main theorem stating the proof problem
theorem billy_can_finish_3_books :
  ∀ (k: Nat), total_time_to_read_books k ∈ (0:ℚ, 5.6:ℚ) → k ≤ 3 :=
by
  intros k hk
  sorry -- Proof is omitted

end billy_can_finish_3_books_l257_257517


namespace exist_midpoints_l257_257865
open Classical

noncomputable def h (a b c : ℝ) := (a + b + c) / 3

theorem exist_midpoints (a b c : ℝ) (X Y Z : ℝ) (AX BY CZ : ℝ) :
  (0 < X) ∧ (X < a) ∧
  (0 < Y) ∧ (Y < b) ∧
  (0 < Z) ∧ (Z < c) ∧
  (X + (a - X) = (h a b c)) ∧
  (Y + (b - Y) = (h a b c)) ∧
  (Z + (c - Z) = (h a b c)) ∧
  (AX * BY * CZ = (a - X) * (b - Y) * (c - Z))
  → ∃ (X Y Z : ℝ), X = (a / 2) ∧ Y = (b / 2) ∧ Z = (c / 2) :=
by
  sorry

end exist_midpoints_l257_257865


namespace max_rectangle_area_l257_257636

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l257_257636


namespace fifth_term_is_19_l257_257335

-- Define the first term and the common difference
def a₁ : Int := 3
def d : Int := 4

-- Define the formula for the nth term in the arithmetic sequence
def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

-- Define the Lean 4 statement proving that the 5th term is 19
theorem fifth_term_is_19 : arithmetic_sequence 5 = 19 :=
by
  sorry -- Proof to be filled in

end fifth_term_is_19_l257_257335


namespace part1_A_inter_B_part1_complement_B_union_A_part2_C_subset_A_implies_a_range_l257_257272

def A : Set ℝ := {x | 3 ≤ 3^x ∧ 3^x ≤ 27}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 1}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}
def complement_R (B : Set ℝ) : Set ℝ := {x | ¬ (x ∈ B)}

theorem part1_A_inter_B : 
  A ∩ B = {x | 2 < x ∧ x ≤ 3} := 
by
  sorry

theorem part1_complement_B_union_A : 
  (complement_R B) ∪ A = {x | x ≤ 3} := 
by
  sorry

theorem part2_C_subset_A_implies_a_range (a : ℝ) : 
  (C a ⊆ A) → (a ≤ 3) := 
by
  sorry

end part1_A_inter_B_part1_complement_B_union_A_part2_C_subset_A_implies_a_range_l257_257272


namespace train_speed_l257_257904

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l257_257904


namespace interesting_quadruples_count_eq_200_l257_257536

open Nat

def is_interesting_quadruple (a b c d : ℕ) : Prop := 
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 12 ∧ a + b > c + d

theorem interesting_quadruples_count_eq_200 : 
  (∃ (a b c d : ℕ), is_interesting_quadruple a b c d) = 200 := 
sorry

end interesting_quadruples_count_eq_200_l257_257536


namespace product_value_l257_257529

-- Define the sequence of fractions
def fraction (n : ℕ) : ℚ := (n + 1) / n

-- Define the product of the sequence from 2/1 to 1010/1009
noncomputable def product_from_2_to_1010 : ℚ :=
  ∏ n in Finset.range 1009, fraction (n + 1)

-- The theorem statement we want to prove
theorem product_value : product_from_2_to_1010 = 1010 :=
by
  sorry

end product_value_l257_257529


namespace minute_hand_length_l257_257794

theorem minute_hand_length 
  (A : ℝ)
  (r θ : ℝ)
  (h1 : A = 14.163809523809524)
  (h2 : θ = π / 3) 
  (h3 : A = (1 / 2) * r^2 * θ) : 
  r ≈ 3.678 :=
by
  sorry

end minute_hand_length_l257_257794


namespace eccentricity_of_ellipse_l257_257347

-- Define the conditions and given facts.
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0)
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the shared right focus F2 and point P as a common point.
-- Point P (x_p, y_p) lies on both ellipse and hyperbola
variables {x_p y_p c : ℝ} (h3 : ellipse x_p y_p) (h4 : hyperbola x_p y_p)
(h5 : |Complex.abs (Complex.mk x_p y_p - Complex.mk (2 * sqrt 2) 0)| = 2)

-- Define the eccentricity of the ellipse.
def eccentricity (a c : ℝ) : ℝ := c / a

-- Now state the theorem
theorem eccentricity_of_ellipse : eccentricity a (2 * sqrt 2) = (sqrt 2) / 2 := sorry

end eccentricity_of_ellipse_l257_257347


namespace max_rectangle_area_l257_257637

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l257_257637


namespace task_probability_l257_257113

theorem task_probability :
  let P1 := (3 : ℚ) / 8
  let P2_not := 2 / 5
  let P3 := 5 / 9
  let P4_not := 5 / 12
  P1 * P2_not * P3 * P4_not = 5 / 72 :=
by
  sorry

end task_probability_l257_257113


namespace line_always_passes_fixed_point_l257_257267

theorem line_always_passes_fixed_point:
  ∀ a x y, x = 5 → y = -3 → (a * x + (2 * a - 1) * y + a - 3 = 0) :=
by
  intros a x y h1 h2
  rw [h1, h2]
  sorry

end line_always_passes_fixed_point_l257_257267


namespace skittles_taken_away_l257_257523

def initial_skittles : ℕ := 25
def remaining_skittles : ℕ := 18

theorem skittles_taken_away (C_initial C_remaining : ℕ) (h1 : C_initial = 25) (h2 : C_remaining = 18) :
  (C_initial - C_remaining = 7) :=
by
  sorry

end skittles_taken_away_l257_257523


namespace points_are_concyclic_l257_257369

theorem points_are_concyclic
  (A B C H E F E' F' H' : Point)
  (h1 : acute_triangle A B C)
  (h2 : orthocenter H A B C)
  (Γ : Circle)
  (h3 : Γ.center = H ∧ Γ.radius = dist A H)
  (h4 : circle_intersect_line Γ (line_through A B) E)
  (h5 : circle_intersect_line Γ (line_through A C) F)
  (h6 : reflection_over_line E (line_through B C) E')
  (h7 : reflection_over_line F (line_through B C) F')
  (h8 : reflection_over_line H (line_through B C) H') :
  concyclic {A, E', F', H'} :=
by sorry

end points_are_concyclic_l257_257369


namespace max_area_of_fenced_rectangle_l257_257613

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l257_257613


namespace length_of_BC_l257_257689

theorem length_of_BC 
  (A B C X : Type) 
  (AB AC : ℕ) 
  (radius : ℕ) 
  (BC BX CX : ℕ) 
  (H1 : AB = 80) 
  (H2 : AC = 100) 
  (H3 : radius = AB)
  (H4 : BC = BX + CX) 
  (H5 : CX * BC = 3600)
  (H6 : BX + CX > AC) 
  (H7 : CX = 40 ∧ BX = 50) :
  BC = 90 :=
by
  have : CX * BC = 3600 := H5
  have h₁ := congr_arg (λ x, x + BX) (Eq.symm H7.2)
  have h₂ := add_comm 50 40
  have h₃ := congr_arg (λ x, x + 40) h₂
  have h₄ : 90 + 0 = 90 := rfl
  sorry

end length_of_BC_l257_257689


namespace range_of_varphi_l257_257655

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ) + 1

theorem range_of_varphi (ω ϕ : ℝ) (h_ω_pos : ω > 0) (h_ϕ_bound : |ϕ| ≤ (Real.pi) / 2)
  (h_intersection : (∀ x, f x ω ϕ = -1 → (∃ k : ℤ, x = (k * Real.pi) / ω)))
  (h_f_gt_1 : (∀ x, -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ω ϕ > 1)) :
  ω = 2 → (Real.pi / 6 ≤ ϕ) ∧ (ϕ ≤ Real.pi / 3) :=
by
  sorry

end range_of_varphi_l257_257655


namespace find_function_f_l257_257207

-- The function f maps positive integers to positive integers
def f : ℕ+ → ℕ+ := sorry

-- The statement to be proved
theorem find_function_f (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, (f m)^2 + f n ∣ (m^2 + n)^2) : ∀ n : ℕ+, f n = n :=
sorry

end find_function_f_l257_257207


namespace intersection_of_AB_CD_l257_257336

def point (α : Type*) := (α × α × α)

def A : point ℚ := (5, -8, 9)
def B : point ℚ := (15, -18, 14)
def C : point ℚ := (1, 4, -7)
def D : point ℚ := (3, -4, 11)

def parametric_AB (t : ℚ) : point ℚ :=
  (5 + 10 * t, -8 - 10 * t, 9 + 5 * t)

def parametric_CD (s : ℚ) : point ℚ :=
  (1 + 2 * s, 4 - 8 * s, -7 + 18 * s)

def intersection_point (pi : point ℚ) :=
  ∃ t s : ℚ, parametric_AB t = pi ∧ parametric_CD s = pi

theorem intersection_of_AB_CD : intersection_point (76/15, -118/15, 170/15) :=
  sorry

end intersection_of_AB_CD_l257_257336


namespace probability_three_of_a_kind_l257_257973

theorem probability_three_of_a_kind :
  let n_total_ways := Nat.choose 52 5,
  let n_successful_outcomes := 13 * 4 * 12 * 4 * 11 * 4,
  let probability := (109824 : ℚ) / (2598960 : ℚ) in
  probability = (1719 : ℚ) / (40921 : ℚ) :=
by
  let n_total_ways := Nat.choose 52 5
  let n_successful_outcomes := 13 * 4 * 12 * 4 * 11 * 4
  let probability := (109824 : ℚ) / (2598960 : ℚ)
  show probability = (1719 : ℚ) / (40921 : ℚ)
  sorry

end probability_three_of_a_kind_l257_257973


namespace contribution_per_person_correct_l257_257843

-- Definitions from conditions
def total_fundraising_goal : ℕ := 2400
def number_of_participants : ℕ := 8
def administrative_fee_per_person : ℕ := 20

-- Desired answer
def total_contribution_per_person : ℕ := total_fundraising_goal / number_of_participants + administrative_fee_per_person

-- Proof statement
theorem contribution_per_person_correct :
  total_contribution_per_person = 320 :=
by
  sorry  -- Proof to be provided

end contribution_per_person_correct_l257_257843


namespace frank_completes_book_in_three_days_l257_257971

-- Define the total number of pages in a book
def total_pages : ℕ := 249

-- Define the number of pages Frank reads per day
def pages_per_day : ℕ := 83

-- Define the number of days Frank needs to finish a book
def days_to_finish_book (total_pages pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

-- Theorem statement to prove that Frank finishes a book in 3 days
theorem frank_completes_book_in_three_days : days_to_finish_book total_pages pages_per_day = 3 := 
by {
  -- Proof goes here
  sorry
}

end frank_completes_book_in_three_days_l257_257971


namespace solve_equation_l257_257768

theorem solve_equation (x : ℝ) (h : x ≠ 2) : -x^2 = (4 * x + 2) / (x - 2) ↔ x = -2 :=
by sorry

end solve_equation_l257_257768


namespace find_circle_center_l257_257483

-- Definitions and conditions
def is_tangent (c : ℝ × ℝ) (p1 p2 : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  let (a, b) := c in let (x1, y1) := p1 in
  let slope := (y1 - f x1) / (x1 - x1^2) -- Calculate the slope for tangent
  (y1 - b) = slope * (x1 - a)

def is_on_circle (center : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : Prop :=
  let (a, b) := center in
  let (x, y) := p in
  (x - a)^2 + (y - b)^2 = r^2

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  ((x1 + x2) / 2, (y1 + y2) / 2)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  (y2 - y1) / (x2 - x1)

-- Proof statement
theorem find_circle_center :
  ∃ a b : ℝ, 
  is_tangent (a, b) (1, 1) (λ x, x^2) ∧ 
  is_on_circle (a, b) (0, 3) ((1 - 0)^2 + (1 - 3)^2) ∧ 
  let mp := midpoint (1,1) (0,3) 
  in slope (mp, (a,b)) = -2 ∧ 
  (a, b) = (0, 3/2) := sorry

end find_circle_center_l257_257483


namespace mayor_vice_mayor_happy_people_l257_257916

theorem mayor_vice_mayor_happy_people :
  (∃ (institutions_per_institution : ℕ) (num_institutions : ℕ),
    institutions_per_institution = 80 ∧
    num_institutions = 6 ∧
    num_institutions * institutions_per_institution = 480) :=
by
  sorry

end mayor_vice_mayor_happy_people_l257_257916


namespace calc_num_int_values_l257_257294

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l257_257294


namespace shape_of_rho_eq_2c_l257_257552

theorem shape_of_rho_eq_2c (c : ℝ) (h : 0 < c) : 
  ∃ (shape : Type), shape = "sphere" :=
begin
  use "sphere",
  sorry
end

end shape_of_rho_eq_2c_l257_257552


namespace zoey_preparation_months_l257_257853
open Nat

-- Define months as integers assuming 1 = January, 5 = May, 9 = September, etc.
def month_start : ℕ := 5 -- May
def month_exam : ℕ := 9 -- September

-- The function to calculate the number of months of preparation excluding the exam month.
def months_of_preparation (start : ℕ) (exam : ℕ) : ℕ := (exam - start)

theorem zoey_preparation_months :
  months_of_preparation month_start month_exam = 4 := by
  sorry

end zoey_preparation_months_l257_257853


namespace infinitely_many_unattainable_l257_257371

-- Define the necessary variables and conditions
variables (n : ℕ) (Hn : n ≥ 9)

-- Define what it means for a number to be attainable
def is_attainable (a : ℕ) : Prop :=
a = 1 ∨ (∃ (s : List (ℕ → ℕ)) (H_first : ∀ f, s.head = some f → (f = (λ x, x + 2) ∨ f = (λ x, x * 2))),
  (∀ i, s.nth i = some (λ x, x + 2) → ((s.nth (i + 1) = some (λ x, x * 2)) ∨ (s.nth (i + 1) = some (λ x, x * n))) ∧ 
       (s.nth i = some (λ x, x * 2) → ((s.nth (i + 1) = some (λ x, x + 2)) ∨ (s.nth (i + 1) = some (λ x, x + n)))),
   a = s.foldl (λ acc f, f acc) 1)

-- Define what it means for a number to be unattainable
def is_unattainable (a : ℕ) : Prop := ¬is_attainable n a

-- The main theorem stating that there are infinitely many unattainable numbers for n >= 9
theorem infinitely_many_unattainable (n : ℕ) (Hn : n ≥ 9) : ∃ (A : set ℕ), set.infinite A ∧ ∀ a ∈ A, is_unattainable n a :=
sorry

end infinitely_many_unattainable_l257_257371


namespace smallest_M_l257_257548

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - (Real.floor x)

theorem smallest_M (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2024) :
  fractional_part a + fractional_part b + fractional_part c ≤ 2 + 2024 / 2025 :=
sorry

end smallest_M_l257_257548


namespace number_of_correct_conclusions_l257_257233

-- Given conditions
variables {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : c > 3)
           (h₂ : a * 25 + b * 5 + c = 0)
           (h₃ : -b / (2 * a) = 2)
           (h₄ : a < 0)

-- Proof should show:
theorem number_of_correct_conclusions 
  (h₀ : a ≠ 0)
  (h₁ : c > 3)
  (h₂ : 25 * a + 5 * b + c = 0)
  (h₃ : - b / (2 * a) = 2)
  (h₄ : a < 0) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ (a * x₁^2 + b * x₁ + c = 2) ∧ (a * x₂^2 + b * x₂ + c = 2)) ∧ 
  (a < -3 / 5) := 
by
  sorry

end number_of_correct_conclusions_l257_257233


namespace b_5_correct_l257_257940

noncomputable def b : ℕ → ℚ 
| 1       := 2
| 2       := 1
| (n + 3) := (b n * b (n + 1)) / (2 * b n + b (n + 1))

theorem b_5_correct (p q : ℕ) (h_coprime : Nat.gcd p q = 1) :
  ∃ p q : ℕ, p = 2 ∧ q = 29 ∧ p + q = 31 ∧ (b 5 = p / q) := 
begin
  -- The actual proof would go here.
  sorry
end

end b_5_correct_l257_257940


namespace age_of_replaced_person_is_46_l257_257783

variable (age_of_replaced_person : ℕ)
variable (new_person_age : ℕ := 16)
variable (decrease_in_age_per_person : ℕ := 3)
variable (number_of_people : ℕ := 10)

theorem age_of_replaced_person_is_46 :
  age_of_replaced_person - new_person_age = decrease_in_age_per_person * number_of_people → 
  age_of_replaced_person = 46 :=
by
  sorry

end age_of_replaced_person_is_46_l257_257783


namespace total_surface_area_of_cube_with_holes_l257_257899

-- Definitions based on conditions
def edge_length := 4
def hole_side := 2

-- Theorem statement
theorem total_surface_area_of_cube_with_holes (edge_length hole_side : ℕ) (h_cube : edge_length = 4) (h_hole : hole_side = 2) : 
  let original_surface_area := 6 * edge_length^2 in
  let hole_area := hole_side^2 in
  let removed_area := 6 * hole_area in
  let exposed_internal_area := 6 * 4 * hole_area in
  original_surface_area - removed_area + exposed_internal_area = 168 := 
  by  sorry

end total_surface_area_of_cube_with_holes_l257_257899


namespace ball_flight_time_l257_257930

-- Conditions
def ball_speed : ℝ := 20 -- in feet per second
def collie_speed : ℝ := 5 -- in feet per second
def catch_up_time : ℝ := 32 -- in seconds

-- Question to be proved: The time the ball flies before hitting the ground
theorem ball_flight_time : ∃ t : ℝ, t = 8 ∧ 20 * t = collie_speed * catch_up_time := by
  use 8
  split
  {
    refl
  }
  {
    calc
      20 * 8 = 20 * 8 : by refl
      ... = 160 : by norm_num
      ... = 5 * 32 : by norm_num
  }

end ball_flight_time_l257_257930


namespace max_area_of_rectangular_pen_l257_257603

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l257_257603


namespace number_of_ordered_pairs_eq_one_l257_257963

theorem number_of_ordered_pairs_eq_one :
  ∃! (x y : ℝ), 32^(x^2 + y) + 32^(x + y^2) = 1 :=
by {
  sorry
}

end number_of_ordered_pairs_eq_one_l257_257963


namespace total_puppies_adopted_l257_257848

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l257_257848


namespace minimum_matches_for_top_two_l257_257132

/-- In a tournament with 25 chess players where each player has different skill levels and the stronger player always wins,
    the minimum number of matches required to determine the two strongest players is 28. -/
theorem minimum_matches_for_top_two (P : Type) [LinearOrder P] 
  (players : Finset P) (h : players.card = 25) 
  (wins : ∀ (p1 p2 : P), p1 ∈ players → p2 ∈ players → p1 ≠ p2 → p1 < p2 ∨ p2 < p1) : 
  ∃ m, m = 28 ∧ 
  (∀ (matches : Finset (P × P)), (∀ (p1 p2 : P), (p1, p2) ∈ matches → p1 ∈ players ∧ p2 ∈ players ∧ p1 ≠ p2) → 
  (∃ topTwo : Finset P, topTwo.card = 2 ∧ 
  (∀ p ∈ topTwo, ∃ (M : Finset (P × P)), (∀ p1 p2, (p1, p2) ∈ M → p1 < p2) ∧ M.card ≤ m))
  → matches.card = m) :=
sorry

end minimum_matches_for_top_two_l257_257132


namespace trig_identity_proof_l257_257191

theorem trig_identity_proof :
  (sin 110 * sin 20 / (cos 155 ^ 2 - sin 155 ^ 2)) = 1 / 2 :=
by
  sorry

end trig_identity_proof_l257_257191


namespace max_product_of_two_integers_l257_257086

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l257_257086


namespace incorrect_log_statement_l257_257307

theorem incorrect_log_statement (a x : ℝ) (h₀ : a ≠ 1) (h₁ : a ≠ 0) :
  ¬(∀ x, 0 < x ∧ x < 1 → log a x < 0) :=
sorry

end incorrect_log_statement_l257_257307


namespace cos_double_angle_l257_257304

theorem cos_double_angle (α : ℝ) (h : Real.sin α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = 1 / 3 :=
by
  sorry

end cos_double_angle_l257_257304


namespace value_added_to_numbers_l257_257782

theorem value_added_to_numbers :
  ∀ (S X : ℝ), (S / 8 = 8) → ((S + 5 * X) / 8 = 10.5) → X = 4 :=
by
  intros S X h1 h2
  have hS : S = 64 := by
    rw div_eq_iff (by norm_num : (8:ℝ) ≠ 0) at h1
    exact h1
  rw hS at h2
  rw [add_div, mul_div_cancel_left] at h2
  swap
  norm_num
  rw eq_div_iff (by norm_num : (8:ℝ) ≠ 0) at h2
  linarith

end value_added_to_numbers_l257_257782


namespace city_mpg_l257_257403

-- Definitions of the conditions
variables (C x C_mpg highway_miles:int)
variable (total_miles: 365)
variable (total_gallons: 11)
variable (highway_mpg: 37)
variable (highway_extra: 5)

-- Lean 4 statement 
theorem city_mpg : 
  (x + highway_extra = highway_miles) ∧  -- Highway miles = city miles + 5
  (highway_miles / highway_mpg + x / C = total_gallons) ∧  -- Total gallons used
  (x + highway_miles = total_miles)  -- Total miles driven
  → C = 30 :=
sorry

end city_mpg_l257_257403


namespace area_of_bounded_figure_l257_257519

noncomputable def area_under_parametric_curves : ℝ :=
  let x t := 8 * (t - sin t)
  let y t := 8 * (1 - cos t)
  let x_bound := (16:ℝ) * Real.pi
  let y_line := (12:ℝ)
  let parametric_area := ∫ t in (2 * Real.pi / 3)..(4 * Real.pi / 3), 8 * (1 - cos t) * (deriv (λ t, 8 * (t - sin t))) t
  let rectangle_area := y_line * ((x (4 * Real.pi / 3)) - (x (2 * Real.pi / 3)))
  parametric_area - rectangle_area

theorem area_of_bounded_figure : area_under_parametric_curves = 48 * Real.sqrt 3 :=
sorry

end area_of_bounded_figure_l257_257519


namespace blue_pill_cost_proof_l257_257179

-- Define variables for the costs of blue and orange pills
variable (blue_pill_cost orange_pill_cost : ℚ)

-- Given conditions
def conditions_1 : Prop := ∀ days, days = 21
def conditions_2 : Prop := blue_pill_cost = orange_pill_cost + 2
def conditions_3 : Prop := (21 * (2 * blue_pill_cost + (blue_pill_cost - 2))) = 756

-- Theorem statement: the cost of one blue pill
theorem blue_pill_cost_proof (days : ℚ) (h1 : conditions_1) (h2 : conditions_2) (h3 : conditions_3) : blue_pill_cost = 38/3 :=
by
  sorry

end blue_pill_cost_proof_l257_257179


namespace median_unchanged_after_removal_l257_257509

open Set

noncomputable def median (s : Set ℝ) : ℝ := sorry

-- Assume a sufficiently large set of distinct real numbers
variable {S : Set ℝ} (h_distinct : ∀ a b ∈ S, a ≠ b → a ≠ b)
variable (h_large : ∃ a b c d ∈ S, True) -- simplification for sufficiently large set

theorem median_unchanged_after_removal :
  ∀ S' ⊆ S, S'.card = S.card - 2 → median S' = median S :=
begin
  sorry -- proof omitted as per instructions
end

end median_unchanged_after_removal_l257_257509


namespace solve_inequality_l257_257956

theorem solve_inequality (x : ℝ) : ( (x-1)/(x-3) ≥ 2 ) → x ∈ Ioo 3 5 ∨ x = 5 :=
by
sorry

end solve_inequality_l257_257956


namespace driver_schedule_l257_257040

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l257_257040


namespace minimum_PA_PB_l257_257333

section 

variable (α : ℝ)
variable {t : ℝ}

-- Parametric equations of the line l
def x (t : ℝ) (α : ℝ) := 2 + t * Real.cos α
def y (t : ℝ) (α : ℝ) := 1 + t * Real.sin α

-- Cartesian coordinate equation of circle C
def circle_eq (x y : ℝ) := (x - 3)^2 + y^2 = 9

-- Circle equation in terms of Cartesian coordinates
example : circle_eq (x 0 α) (y 0 α) :=
by sorry

-- Minimum value of |PA| + |PB| 
theorem minimum_PA_PB : 
  let t1 := (2 * Real.cos α - (2 * Real.sin α) + (Real.sqrt ((2 * Real.cos α - 2 * Real.sin α)^2 + 28))) / 2,
      t2 := (2 * Real.cos α - (2 * Real.sin α) - (Real.sqrt ((2 * Real.cos α - 2 * Real.sin α)^2 + 28))) / 2 in
  (|t1| + |t2|) = 2 * Real.sqrt 7 :=
by sorry

end

end minimum_PA_PB_l257_257333


namespace num_integers_achievable_le_2014_l257_257824

def floor_div (x : ℤ) : ℤ := x / 2

def button1 (x : ℤ) : ℤ := floor_div x

def button2 (x : ℤ) : ℤ := 4 * x + 1

def num_valid_sequences (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 2
  else num_valid_sequences (n - 1) + num_valid_sequences (n - 2)

theorem num_integers_achievable_le_2014 :
  num_valid_sequences 11 = 233 :=
  by
    -- Proof starts here
    sorry

end num_integers_achievable_le_2014_l257_257824


namespace necessary_but_not_sufficient_l257_257594

def p (x : ℝ) : Prop := (1 / 2) ^ x > 1
def q (x : ℝ) : Prop := -2 < x ∧ x < -1

theorem necessary_but_not_sufficient (x : ℝ) : (q x → p x) ∧ ¬ (p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_l257_257594


namespace toy_poodle_height_cm_l257_257432

-- Define the conditions given in the problem statement
def std_poodle_height_in_inches : ℝ := 28
def std_poodle_height_in_cm : ℝ := 71.12
def miniature_poodle_height_in_inches (S : ℝ) : ℝ := S - 8.5
def toy_poodle_height_in_inches (M : ℝ) : ℝ := M - 6.25
def moyen_poodle_height_in_inches (S : ℝ) : ℝ := S - 3.75
def moyen_poodle_toy_poodle_relationship (Mo T : ℝ) : Prop := Mo = T + 4.75
def klein_poodle_height (T Mo : ℝ) : ℝ := (T + Mo) / 2
def inch_to_cm (x : ℝ) : ℝ := x * 2.54

-- Prove that the height of the toy poodle is 33.655 cm
theorem toy_poodle_height_cm :
  let S := std_poodle_height_in_inches in
  let M := miniature_poodle_height_in_inches S in
  let T := toy_poodle_height_in_inches M in
  inch_to_cm T = 33.655 :=
by
  sorry

end toy_poodle_height_cm_l257_257432


namespace ratio_first_to_second_l257_257804

theorem ratio_first_to_second (S F T : ℕ) 
  (hS : S = 60)
  (hT : T = F / 3)
  (hSum : F + S + T = 220) :
  F / S = 2 :=
by
  sorry

end ratio_first_to_second_l257_257804


namespace sides_of_length_five_l257_257195

theorem sides_of_length_five (GH HI : ℝ) (L : ℝ) (total_perimeter : ℝ) :
  GH = 7 → HI = 5 → total_perimeter = 38 → (∃ n m : ℕ, n + m = 6 ∧ n * 7 + m * 5 = 38 ∧ m = 2) := by
  intros hGH hHI hPerimeter
  sorry

end sides_of_length_five_l257_257195


namespace number_of_pages_l257_257841

-- Define the conditions
def rate_of_printer_A (P : ℕ) : ℕ := P / 60
def rate_of_printer_B (P : ℕ) : ℕ := (P / 60) + 6

-- Define the combined rate condition
def combined_rate (P : ℕ) (R_A R_B : ℕ) : Prop := (R_A + R_B) = P / 24

-- The main theorem to prove
theorem number_of_pages :
  ∃ (P : ℕ), combined_rate P (rate_of_printer_A P) (rate_of_printer_B P) ∧ P = 720 := by
  sorry

end number_of_pages_l257_257841


namespace ellipse_eccentricity_l257_257254

-- Definition of points and ellipse properties
noncomputable def foci (F1 F2 P : Type) : Prop :=
-- Definitions representing the geometric conditions of the problem
PerpendicularToMajorAxis (F2 : Type) : Prop := sorry
IsoscelesTriangle (F1 P F2 : Type) : Prop := sorry

-- Definition of the eccentricity of the ellipse
noncomputable def eccentricity (a b c : ℝ) : ℝ := c / a

-- Problem statement
theorem ellipse_eccentricity (a b c : ℝ) (F1 F2 P : Type) 
  (h1 : PerpendicularToMajorAxis(F2))
  (h2 : foci(F1, F2, P))
  (h3 : IsoscelesTriangle(F1, P, F2))
  (ha_ne_zero : a ≠ 0) :
  eccentricity(a, b, c) = (ℝ.sqrt 2 - 1) := 
sorry

end ellipse_eccentricity_l257_257254


namespace necessary_condition_for_inequality_l257_257589

theorem necessary_condition_for_inequality (a b : ℝ) 
  (h : a * real.sqrt a + b * real.sqrt b > a * real.sqrt b + b * real.sqrt a) :
  a ≥ 0 ∧ b ≥ 0 ∧ a ≠ b :=
sorry

end necessary_condition_for_inequality_l257_257589


namespace cuboid_diagonal_l257_257055

theorem cuboid_diagonal (a b c : ℝ) (h₁ : 2 * (a * b + b * c + a * c) = 20) (h₂ : 4 * (a + b + c) = 24) :
  real.sqrt (a^2 + b^2 + c^2) = 4 :=
by
  sorry

end cuboid_diagonal_l257_257055


namespace projection_length_ratio_l257_257729

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def projection (u v : V) : V := (⟪u, v⟫ / ⟪v, v⟫) • v

theorem projection_length_ratio
  (v w : V)
  (h1 : ∥projection v w∥ / ∥v∥ = 3 / 4) :
  let p := projection v w,
      u := projection w v,
      q := projection p u 
  in ∥q∥ / ∥v∥ = 9 / 16 := 
by 
  -- Proof input placeholder
  sorry

end projection_length_ratio_l257_257729


namespace range_of_a_l257_257685

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1))
  ↔ (-1 < a ∧ a < 3) :=
by sorry

end range_of_a_l257_257685


namespace max_area_of_rectangle_with_perimeter_60_l257_257621

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l257_257621


namespace machine_production_l257_257125

theorem machine_production
  (rate_per_minute : ℕ)
  (machines_total : ℕ)
  (production_minute : ℕ)
  (machines_sub : ℕ)
  (time_minutes : ℕ)
  (total_production : ℕ) :
  machines_total * rate_per_minute = production_minute →
  rate_per_minute = production_minute / machines_total →
  machines_sub * rate_per_minute = total_production / time_minutes →
  time_minutes * total_production / time_minutes = 900 :=
by
  sorry

end machine_production_l257_257125


namespace value_of_y_l257_257315

theorem value_of_y (y : ℝ) (α : ℝ) (h₁ : (-3, y) = (x, y)) (h₂ : Real.sin α = -3 / 4) : 
  y = -9 * Real.sqrt 7 / 7 := 
  sorry

end value_of_y_l257_257315


namespace brick_in_box_probability_l257_257438

noncomputable def c1_c2_c3_d1_d2_d3 : finset ℕ := finset.range 1 501

def q := 1 / 4

theorem brick_in_box_probability :
  let sum_numerator_denominator (frac : ℚ) := frac.num.nat_abs + frac.denom in
  sum_numerator_denominator q = 5 :=
by
  -- Definitions for c1, c2, c3, d1, d2, d3 are implicit here by the setup
  let c1 := 0; let c2 := 0; let c3 := 0
  let d1 := 0; let d2 := 0; let d3 := 0
  have q_frac : q = 1 / 4 := by rfl
  have sum_q : (1 : ℚ).nat_abs + (4 : ℚ).nat_abs = 5 := by norm_num
  exact sum_q

end brick_in_box_probability_l257_257438


namespace tangent_line_at_zero_l257_257958

noncomputable def curve (x : ℝ) : ℝ := Real.exp (2 * x)

theorem tangent_line_at_zero :
  ∃ m b, (∀ x, (curve x) = m * x + b) ∧
    m = 2 ∧ b = 1 :=
by 
  sorry

end tangent_line_at_zero_l257_257958


namespace david_bicycles_65_km_l257_257016

noncomputable def distance_david_travels : ℝ :=
  let distance_ac_centerville := 60
  let distance_centerville_midland := 60
  let distance_centerville_drake := 60
  let speed_aaron := 17
  let speed_michael := 7
  let total_distance_ac_midland := distance_ac_centerville + distance_centerville_midland
  let combined_speed := speed_aaron + speed_michael
  let meeting_time := total_distance_ac_midland / combined_speed
  let aaron_distance := speed_aaron * meeting_time
  let michael_distance := speed_michael * meeting_time
  let meeting_point := distance_midland - michael_distance
  let distance_x := meeting_point
  let distance_y := distance_centerville_drake
  real.sqrt (distance_x^2 + distance_y^2)

theorem david_bicycles_65_km :
  let distance := distance_david_travels
  distance = 65 := sorry

end david_bicycles_65_km_l257_257016


namespace second_largest_values_l257_257493

def list_of_integers (a : list ℕ) : Prop :=
  a.length = 5 ∧ -- list has five integers
  (a.sum : ℚ) / 5 = 15 ∧ -- mean is 15
  (list.maximum a - list.minimum a) = 20 ∧ -- range is 20
  list.median a = 10 ∧ -- median is 10
  list.mode a = 10 -- mode is 10

theorem second_largest_values :
  ∀ a : list ℕ, list_of_integers a → (second_largest_values_count a = 6) :=
by
  sorry

end second_largest_values_l257_257493


namespace tickets_left_l257_257178

theorem tickets_left 
    (initial_tickets : ℕ) 
    (lost_tickets : ℕ) 
    (used_tickets : ℕ) 
    (remaining_tickets : ℕ) :
    initial_tickets = 14 → 
    lost_tickets = 2 → 
    used_tickets = 10 → 
    remaining_tickets = initial_tickets - lost_tickets - used_tickets → 
    remaining_tickets = 2 := 
by
    intros h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    norm_num
    sorry

end tickets_left_l257_257178


namespace polynomial_arithmetic_progression_roots_l257_257955

-- Define the problem as a Lean statement
theorem polynomial_arithmetic_progression_roots (p q : ℝ) :
  (∃ a b : ℝ, is_root (polynomial.C q + polynomial.C p * X ^ 2 + X^4) a ∧
    is_root (polynomial.C q + polynomial.C p * X ^ 2 + X^4) b ∧
    a + b = 0 ∧ a ≠ b ∧ a ≠ 0 ∧ b ≠ 0) ↔ (p ≤ 0 ∧ q = 0.09 * p^2) :=
begin
  sorry
end

end polynomial_arithmetic_progression_roots_l257_257955


namespace max_rectangle_area_l257_257630

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l257_257630


namespace find_k_values_l257_257885

theorem find_k_values (k : ℚ) 
  (h1 : ∀ k, ∃ m, m = (3 * k + 9) / (7 - k))
  (h2 : ∀ k, m = 2 * k) : 
  (k = 9 / 2 ∨ k = 1) :=
by
  sorry

end find_k_values_l257_257885


namespace regular_tetrahedron_volume_l257_257433

theorem regular_tetrahedron_volume (total_edge_length : ℝ) (h1 : total_edge_length = 72) : 
  ∃ (V : ℝ), V = 144 * real.sqrt 2 :=
begin
  let edge_length := total_edge_length / 6,
  have h2 : edge_length = 12, { rw h1, norm_num },
  let volume := (real.sqrt 2 / 12) * (edge_length^3),
  have h3 : volume = 144 * real.sqrt 2,
  { rw h2, norm_num, rw [pow_three, real.sqrt_sq (by norm_num : (2:ℝ) ≥ 0)] },
  use volume,
  exact h3,
end

end regular_tetrahedron_volume_l257_257433


namespace bisection_method_requires_all_structures_l257_257836

-- Define the conditions as hypotheses
def every_algorithm_uses_sequential_structure : Prop :=
  ∀ (algorithm : Type), algorithm → sequential_structure algorithm

def loop_structure_includes_conditional_structure : Prop :=
  ∀ (algorithm : Type), loop_structure algorithm → conditional_structure algorithm

def bisection_method_uses_loop_structure : Prop :=
  ∀ (algorithm : Type), algorithm = bisection_method → loop_structure algorithm

-- Now state the theorem which corresponds to our proof problem
theorem bisection_method_requires_all_structures
    (h1 : every_algorithm_uses_sequential_structure)
    (h2 : loop_structure_includes_conditional_structure)
    (h3 : bisection_method_uses_loop_structure) :
    ∃ (algorithm : Type), algorithm = bisection_method ∧ sequential_structure algorithm ∧ conditional_structure algorithm ∧ loop_structure algorithm :=
by
  sorry -- Proof is omitted, as instructed

end bisection_method_requires_all_structures_l257_257836


namespace incorrect_statement_b_l257_257455

/- 
A quadrilateral with perpendicular diagonals is not necessarily a rhombus. 
For example, a kite also has perpendicular diagonals but is not a rhombus.
-/

theorem incorrect_statement_b : ¬ (∀ (Q : Type) [quadrilateral Q], (perpendicular_diagonals Q) → rhombus Q) :=
by
  sorry

end incorrect_statement_b_l257_257455


namespace simplified_expression_evaluation_l257_257762

def x := 3 * Real.sqrt 3 + 2 * Real.sqrt 2
def y := 3 * Real.sqrt 3 - 2 * Real.sqrt 2

def expr := (x * (x + y) + 2 * y * (x + y)) / (x * y * (x + 2 * y)) / (x * y / (x + 2 * y))

theorem simplified_expression_evaluation : expr = 108 := by
  sorry

end simplified_expression_evaluation_l257_257762


namespace average_rate_of_interest_l257_257915

/-- Given:
    1. A woman has a total of $7500 invested,
    2. Part of the investment is at 5% interest,
    3. The remainder of the investment is at 7% interest,
    4. The annual returns from both investments are equal,
    Prove:
    The average rate of interest realized on her total investment is 5.8%.
-/
theorem average_rate_of_interest
  (total_investment : ℝ) (interest_5_percent : ℝ) (interest_7_percent : ℝ)
  (annual_return_equal : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent)
  (total_investment_eq : total_investment = 7500) : 
  (interest_5_percent / total_investment) = 0.058 :=
by
  -- conditions given
  have h1 : total_investment = 7500 := total_investment_eq
  have h2 : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent := annual_return_equal

  -- final step, sorry is used to skip the proof
  sorry

end average_rate_of_interest_l257_257915


namespace binomial_coeff_identity_l257_257653

theorem binomial_coeff_identity :
  let a₀ := (ℝ) 
  let a₁ := (ℝ) 
  let a₂ := (ℝ) 
  let a₃ := (ℝ) in
    ( (sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 ) →
    ((a₀ + a₂)^2 - (a₁ + a₃)^2 = -64) :=
by
  sorry

end binomial_coeff_identity_l257_257653


namespace drivers_sufficiency_and_ivan_petrovich_departure_l257_257030

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l257_257030


namespace sqrt_expression_correct_l257_257675

noncomputable def alpha := Real

theorem sqrt_expression_correct (α : alpha) (hα : Real.pi < α ∧ α < 3 * Real.pi / 2) :
  Real.sqrt (1 / 2 + 1 / 2 * Real.sqrt (1 / 2 + 1 / 2 * Real.cos (2 * α))) =
  Real.sin (α / 2) := 
sorry

end sqrt_expression_correct_l257_257675


namespace gcf_45_135_90_l257_257083

theorem gcf_45_135_90 : Nat.gcd (Nat.gcd 45 135) 90 = 45 := 
by
  sorry

end gcf_45_135_90_l257_257083


namespace yuko_in_front_of_yuri_l257_257115

theorem yuko_in_front_of_yuri (X : ℕ) (hYuri : 2 + 4 + 5 = 11) (hYuko : 1 + 5 + X > 11) : X = 6 := 
by
  sorry

end yuko_in_front_of_yuri_l257_257115


namespace problem_statement_l257_257466

theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) = Real.sqrt m - Real.sqrt n) →
  m + n = 2011 :=
sorry

end problem_statement_l257_257466


namespace george_max_pencils_l257_257578

-- Define the conditions for the problem
def total_money : ℝ := 9.30
def pencil_cost : ℝ := 1.05
def discount_rate : ℝ := 0.10

-- Define the final statement to prove
theorem george_max_pencils (n : ℕ) :
  (n ≤ 8 ∧ pencil_cost * n ≤ total_money) ∨ 
  (n > 8 ∧ pencil_cost * (1 - discount_rate) * n ≤ total_money) →
  n ≤ 9 :=
by
  sorry

end george_max_pencils_l257_257578


namespace batsman_highest_score_l257_257784

noncomputable def find_highest_score
  (total_runs : ℕ)
  (total_runs_excluding_H_L : ℕ)
  (sum_H_L : ℕ)
  (diff_H_L : ℕ) : ℕ :=
  let H := (sum_H_L + diff_H_L) / 2 in H

theorem batsman_highest_score
  (total_runs : ℕ := 2760)
  (total_runs_excluding_H_L : ℕ := 2552)
  (sum_H_L : ℕ := 208)
  (diff_H_L : ℕ := 180) :
  find_highest_score total_runs total_runs_excluding_H_L sum_H_L diff_H_L = 194 :=
by
  sorry

end batsman_highest_score_l257_257784


namespace wheat_field_problem_l257_257745

def equations (x F : ℕ) :=
  (6 * x - 300 = F) ∧ (5 * x + 200 = F)

theorem wheat_field_problem :
  ∃ (x F : ℕ), equations x F ∧ x = 500 ∧ F = 2700 :=
by
  sorry

end wheat_field_problem_l257_257745


namespace integer_values_of_b_l257_257567

theorem integer_values_of_b (b : ℤ) : 
  (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0 ∧ x ≠ y) → 
  ∃ S : finset ℤ, S.card = 8 ∧ ∀ c ∈ S, ∃ x : ℤ, x^2 + c * x + 12 * c = 0 :=
sorry

end integer_values_of_b_l257_257567


namespace area_of_rhombus_l257_257127

-- Defining conditions for the problem
def d1 : ℝ := 40   -- Length of the first diagonal in meters
def d2 : ℝ := 30   -- Length of the second diagonal in meters

-- Calculating the area of the rhombus
noncomputable def area : ℝ := (d1 * d2) / 2

-- Statement of the theorem
theorem area_of_rhombus : area = 600 := by
  sorry

end area_of_rhombus_l257_257127


namespace yuko_in_front_of_yuri_l257_257114

theorem yuko_in_front_of_yuri (X : ℕ) (hYuri : 2 + 4 + 5 = 11) (hYuko : 1 + 5 + X > 11) : X = 6 := 
by
  sorry

end yuko_in_front_of_yuri_l257_257114


namespace set_intersections_l257_257669

open Set Nat

def I : Set ℕ := univ

def A : Set ℕ := { x | ∃ n, x = 3 * n ∧ ∃ k, n = 2 * k }

def B : Set ℕ := { y | ∃ m, y = m ∧ 24 % m = 0 }

theorem set_intersections :
  A ∩ B = {6, 12, 24} ∧ (I \ A) ∩ B = {1, 2, 3, 4, 8} :=
by
  sorry

end set_intersections_l257_257669


namespace tiffany_optimal_area_l257_257439

def optimal_area (A : ℕ) : Prop :=
  ∃ l w : ℕ, l + w = 160 ∧ l ≥ 85 ∧ w ≥ 45 ∧ A = l * w

theorem tiffany_optimal_area : optimal_area 6375 :=
  sorry

end tiffany_optimal_area_l257_257439


namespace max_maple_trees_l257_257491

-- Define the mathematical problem given the conditions and the required proof.
theorem max_maple_trees (total_positions : ℕ) (cannot_be_3_between_maples : ℕ → Prop) :
  (total_positions = 20) →
  (∀ n, cannot_be_3_between_maples n ↔ n ≠ 3) →
  ∃ max_maples : ℕ, max_maples = 12 :=
begin
  intros h_total_positions h_cannot_be_3,
  use 12,
  sorry, -- Proof is omitted as per instruction
end

end max_maple_trees_l257_257491


namespace product_closest_to_106_l257_257799

theorem product_closest_to_106 :
  let product := (2.1 : ℝ) * (50.8 - 0.45)
  abs (product - 106) < abs (product - 105) ∧
  abs (product - 106) < abs (product - 107) ∧
  abs (product - 106) < abs (product - 108) ∧
  abs (product - 106) < abs (product - 110) :=
by
  sorry

end product_closest_to_106_l257_257799


namespace zero_sum_exists_l257_257238

theorem zero_sum_exists
  (n : ℕ)
  (a : Fin n → ℕ)
  (h1 : ∀ k : Fin n, a k ≤ k + 1)
  (h2 : (∑ k, a k) % 2 = 0) :
  ∃ (s : Fin n → Bool), (∑ k : Fin n, if s k then a k else -a k) = 0 := sorry

end zero_sum_exists_l257_257238


namespace abs_ineq_solution_l257_257244

noncomputable theory
open Set Real

theorem abs_ineq_solution (a b x : ℝ) (h_ab : |a - b| > 2) : |x - a| + |x - b| > 2 :=
sorry

end abs_ineq_solution_l257_257244


namespace initial_oranges_per_rupee_l257_257151

theorem initial_oranges_per_rupee (loss_rate_gain_rate cost_rate : ℝ) (initial_oranges : ℤ) : 
  loss_rate_gain_rate = 0.92 ∧ cost_rate = 18.4 ∧ 1.25 * cost_rate = 1.25 * 0.92 * (initial_oranges : ℝ) →
  initial_oranges = 14 := by
  sorry

end initial_oranges_per_rupee_l257_257151


namespace area_between_equal_parallel_chords_l257_257818

noncomputable def area_between_chords (r : ℝ) (d : ℝ) : ℝ :=
  let θ := real.acos (d/(2*r)) in
  2 * (r^2 * θ - (1/2) * d * real.sqrt(r^2 - (d/2)^2))

theorem area_between_equal_parallel_chords :
  area_between_chords 10 6 = 126.6 - 6 * real.sqrt 91 :=
sorry

end area_between_equal_parallel_chords_l257_257818


namespace number_of_teachers_l257_257159

theorem number_of_teachers (total_population sample_size teachers_within_sample students_within_sample : ℕ) 
    (h_total_population : total_population = 3000) 
    (h_sample_size : sample_size = 150) 
    (h_students_within_sample : students_within_sample = 140) 
    (h_teachers_within_sample : teachers_within_sample = sample_size - students_within_sample) 
    (h_ratio : (total_population - students_within_sample) * sample_size = total_population * teachers_within_sample) : 
    total_population - students_within_sample = 200 :=
by {
  sorry
}

end number_of_teachers_l257_257159


namespace distinct_integer_values_b_for_quadratic_l257_257568

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end distinct_integer_values_b_for_quadratic_l257_257568


namespace prove_eccentricity_of_conic_l257_257386

-- Defining the focal points and the ratio condition
variables (Γ : Type) [conic_section Γ] (F1 F2 P : point Γ)
variable (ratio_condition : |distance P F1| : |distance F1 F2| : |distance P F2| = 4 : 3 : 2)

-- Defining the eccentricity values we aim to prove are correct
def eccentricity_of_conic : ℝ :=
  if is_on_conic_section P Γ ∧ focus_pair F1 F2 Γ
  then (eccentricity Γ = 1/2 ∨ eccentricity Γ = 2/3)
  else false

theorem prove_eccentricity_of_conic :
  eccentricity_of_conic Γ F1 F2 P ratio_condition :=
sorrry

end prove_eccentricity_of_conic_l257_257386


namespace fraction_of_paint_used_l257_257715

theorem fraction_of_paint_used 
  (total_paint : ℕ)
  (paint_used_first_week : ℚ)
  (total_paint_used : ℕ)
  (paint_fraction_first_week : ℚ)
  (remaining_paint : ℚ)
  (paint_used_second_week : ℚ)
  (paint_fraction_second_week : ℚ)
  (h1 : total_paint = 360)
  (h2 : paint_fraction_first_week = 2/3)
  (h3 : paint_used_first_week = paint_fraction_first_week * total_paint)
  (h4 : remaining_paint = total_paint - paint_used_first_week)
  (h5 : remaining_paint = 120)
  (h6 : total_paint_used = 264)
  (h7 : paint_used_second_week = total_paint_used - paint_used_first_week)
  (h8 : paint_fraction_second_week = paint_used_second_week / remaining_paint):
  paint_fraction_second_week = 1/5 := 
by 
  sorry

end fraction_of_paint_used_l257_257715


namespace domain_of_function_l257_257013

noncomputable def function_domain_statement : set ℝ :=
  {x | 1 ≤ x ∧ x < 2 ∨ 2 < x ∧ x < 3}

theorem domain_of_function :
  {x : ℝ | x - 1 ≥ 0 ∧ x - 2 ≠ 0 ∧ -x^2 + 2*x + 3 > 0} = function_domain_statement := by
  sorry

end domain_of_function_l257_257013


namespace tan_half_angle_sin_cos_expression_l257_257134

-- Proof Problem 1: If α is an angle in the third quadrant and sin α = -5/13, then tan (α / 2) = -5.
theorem tan_half_angle (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -5 := 
by 
  sorry

-- Proof Problem 2: If tan α = 2, then sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5.
theorem sin_cos_expression (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 :=
by 
  sorry

end tan_half_angle_sin_cos_expression_l257_257134


namespace area_ratio_lim_l257_257339

noncomputable def area_ratio_tends_to (r : ℝ) : Prop :=
∀ ε > 0, ∃ k > 0, abs ((1/2 + (sqrt 3) / 2) - 
  (let PV := sqrt ((2 * r) ^ 2 - (2 * r - 2 * k) ^ 2),
       MR := sqrt ((2 * r) ^ 2 - (2 * r - 3 * k) ^ 2),
       U := PV * k,
       T1 := (k / 2) * (PV + MR)
    in T1 / U)) < ε

theorem area_ratio_lim {r : ℝ} (h : r > 0) : area_ratio_tends_to r :=
sorry

end area_ratio_lim_l257_257339


namespace integer_values_of_b_l257_257565

theorem integer_values_of_b (b : ℤ) : 
  (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0 ∧ x ≠ y) → 
  ∃ S : finset ℤ, S.card = 8 ∧ ∀ c ∈ S, ∃ x : ℤ, x^2 + c * x + 12 * c = 0 :=
sorry

end integer_values_of_b_l257_257565


namespace remainder_seven_times_quotient_l257_257103

theorem remainder_seven_times_quotient (n : ℕ) : 
  (∃ q r : ℕ, n = 23 * q + r ∧ r = 7 * q ∧ 0 ≤ r ∧ r < 23) ↔ (n = 30 ∨ n = 60 ∨ n = 90) :=
by 
  sorry

end remainder_seven_times_quotient_l257_257103


namespace total_number_of_soccer_games_l257_257062

theorem total_number_of_soccer_games (teams : ℕ)
  (regular_games_per_team : ℕ)
  (promotional_games_per_team : ℕ)
  (h1 : teams = 15)
  (h2 : regular_games_per_team = 14)
  (h3 : promotional_games_per_team = 2) :
  ((teams * regular_games_per_team) / 2 + (teams * promotional_games_per_team) / 2) = 120 :=
by
  sorry

end total_number_of_soccer_games_l257_257062


namespace find_x0_l257_257261

noncomputable def f (x : ℝ) : ℝ := exp (2 * x) - 2 * exp (x) + 2 * x

theorem find_x0 : 
  ∃ x0 : ℝ, 
  (∀ x : ℝ, (x - x0) * (f x - (2 * exp (2 * x0) - 2 * exp (x0) + 2) * (x - x0) + f x0) ≥ 0) 
  → x0 = -real.log 2 := 
sorry

end find_x0_l257_257261


namespace erased_number_is_one_or_twenty_l257_257136

theorem erased_number_is_one_or_twenty (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 20)
  (h₂ : (210 - x) % 19 = 0) : x = 1 ∨ x = 20 :=
  by sorry

end erased_number_is_one_or_twenty_l257_257136


namespace problem_statement_l257_257110

def has_arithmetic_square_root (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x

theorem problem_statement :
  (¬ has_arithmetic_square_root (-abs 9)) ∧
  (has_arithmetic_square_root ((-1/4)^2)) ∧
  (has_arithmetic_square_root 0) ∧
  (has_arithmetic_square_root (10^2)) := 
sorry

end problem_statement_l257_257110


namespace machines_initially_working_l257_257139

theorem machines_initially_working (N x : ℕ) (h1 : N * 4 * R = x)
  (h2 : 20 * 6 * R = 3 * x) : N = 10 :=
by
  sorry

end machines_initially_working_l257_257139


namespace buns_cost_eq_1_50_l257_257365

noncomputable def meat_cost : ℝ := 2 * 3.50
noncomputable def tomato_cost : ℝ := 1.5 * 2.00
noncomputable def pickles_cost : ℝ := 2.50 - 1.00
noncomputable def lettuce_cost : ℝ := 1.00
noncomputable def total_other_items_cost : ℝ := meat_cost + tomato_cost + pickles_cost + lettuce_cost
noncomputable def total_amount_spent : ℝ := 20.00 - 6.00
noncomputable def buns_cost : ℝ := total_amount_spent - total_other_items_cost

theorem buns_cost_eq_1_50 : buns_cost = 1.50 := by
  sorry

end buns_cost_eq_1_50_l257_257365


namespace asymptotes_and_eccentricity_of_hyperbola_l257_257788

noncomputable def hyperbola_asymptotes_and_eccentricity : Prop :=
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt 3
  ∀ (x y : ℝ), x^2 - (y^2 / 2) = 1 →
    ((y = 2 * x ∨ y = -2 * x) ∧ Real.sqrt (1 + (b^2 / a^2)) = c)

theorem asymptotes_and_eccentricity_of_hyperbola :
  hyperbola_asymptotes_and_eccentricity :=
by
  sorry

end asymptotes_and_eccentricity_of_hyperbola_l257_257788


namespace find_k_range_l257_257332

noncomputable def point := ℝ × ℝ

def A : point := (-2, 0)

def circle_C (B : point) : Prop := (B.1 - 2)^2 + (B.2)^2 = 4

def midpoint (A B M : point) : Prop := 
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_line_l (P : point) (k : ℝ) : Prop := 
  P.2 = k * P.1 - (sqrt 5) * k

def angle_OPM (O P M : point) : Prop :=
  ∃ θ : ℝ, θ = 30 ∧ 
    angle O P M = θ

theorem find_k_range : 
  ∀ (B M P : point) (k : ℝ), 
  circle_C B → midpoint A B M → on_line_l P k → angle_OPM (0, 0) P M →
  -2 ≤ k ∧ k ≤ 2 :=
by
  intros
  sorry

end find_k_range_l257_257332


namespace medians_inequality_l257_257345

  variable {a b c : ℝ} (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)

  noncomputable def median_length (a b c : ℝ) : ℝ :=
    1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2)

  noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
    (a + b + c) / 2

  theorem medians_inequality (m_a m_b m_c s: ℝ)
    (h_ma : m_a = median_length a b c)
    (h_mb : m_b = median_length b c a)
    (h_mc : m_c = median_length c a b)
    (h_s : s = semiperimeter a b c) :
    m_a^2 + m_b^2 + m_c^2 ≥ s^2 := by
  sorry
  
end medians_inequality_l257_257345


namespace trig_identity_example_l257_257200

theorem trig_identity_example :
  sin (80 * π / 180) * cos (20 * π / 180) - cos (80 * π / 180) * sin (20 * π / 180) = (√3 / 2) :=
by
  sorry

end trig_identity_example_l257_257200


namespace min_distance_circle_ellipse_l257_257402

-- Define the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := ((x + 2)^2 / 9) + ((y - 2)^2 / 9) = 1

-- Statement: Proving the minimum distance between the circle and the ellipse is 0
theorem min_distance_circle_ellipse : ∃ (x₁ y₁ x₂ y₂ : ℝ), circle x₁ y₁ ∧ ellipse x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 0 :=
sorry

end min_distance_circle_ellipse_l257_257402


namespace Jaco_budget_for_parents_gifts_l257_257348

theorem Jaco_budget_for_parents_gifts :
  ∃ (m n : ℕ), (m = 14 ∧ n = 14) ∧ 
  (∀ (friends gifts_friends budget : ℕ), 
   friends = 8 → gifts_friends = 9 → budget = 100 → 
   (budget - (friends * gifts_friends)) / 2 = m ∧ 
   (budget - (friends * gifts_friends)) / 2 = n) := 
sorry

end Jaco_budget_for_parents_gifts_l257_257348


namespace divisors_of_3960_multiple_of_5_l257_257672

def is_divisor (n d : ℕ) : Prop :=
  d ∣ n

noncomputable def prime_factorization : ℕ → list (ℕ × ℕ)
| 3960 := [(2, 3), (3, 2), (5, 1), (11, 1)]
| _    := []

def valid_exponent (base exponent : ℕ) : Prop :=
  match base with
  | 2    => 0 ≤ exponent ∧ exponent ≤ 3
  | 3    => 0 ≤ exponent ∧ exponent ≤ 2
  | 5    => exponent = 1
  | 11   => 0 ≤ exponent ∧ exponent ≤ 1
  | _    => false
  end

def divisor_count_factors_5_multiple : ℕ :=
  4 * 3 * 1 * 2

theorem divisors_of_3960_multiple_of_5 : 
  (∑ (d : ℕ) in (finset.range 3961).filter (λ d, is_divisor 3960 d ∧ is_divisor d 5), 1) = 24 := 
sorry

end divisors_of_3960_multiple_of_5_l257_257672


namespace max_area_of_rectangle_with_perimeter_60_l257_257622

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l257_257622


namespace part_one_part_two_l257_257385

noncomputable def f (x a : ℝ) : ℝ := |2 * x + a| + |x - 1 / a|

theorem part_one (x : ℝ) (h : f x 1 < x + 3) : -3/4 < x ∧ x < 3/2 :=
sorry

theorem part_two (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a ≥ sqrt 2 :=
sorry

end part_one_part_two_l257_257385


namespace tim_drinks_amount_l257_257278

theorem tim_drinks_amount (H : ℚ := 2/7) (T : ℚ := 5/8) : 
  (T * H) = 5/28 :=
by sorry

end tim_drinks_amount_l257_257278


namespace concert_total_revenue_l257_257422

def adult_ticket_price : ℕ := 26
def child_ticket_price : ℕ := adult_ticket_price / 2
def num_adults : ℕ := 183
def num_children : ℕ := 28

def revenue_from_adults : ℕ := num_adults * adult_ticket_price
def revenue_from_children : ℕ := num_children * child_ticket_price
def total_revenue : ℕ := revenue_from_adults + revenue_from_children

theorem concert_total_revenue :
  total_revenue = 5122 :=
by
  -- proof can be filled in here
  sorry

end concert_total_revenue_l257_257422


namespace sham_work_rate_l257_257123

theorem sham_work_rate (rahul_sham_together rahul_alone : ℝ) : 
  rahul_sham_together = 35 → rahul_alone = 60 → 
  let sham_alone := 84 
    in 1 / sham_alone = 1 / 35 - 1 / 60 :=
by
  intros h_r_s_together h_r_alone
  let sham_alone := 84
  have h_combined_rate : 1 / rahul_sham_together = 1 / rahul_alone + 1 / sham_alone := by sorry
  rw [h_r_s_together, h_r_alone] at h_combined_rate
  exact h_combined_rate

end sham_work_rate_l257_257123


namespace commute_times_difference_l257_257153

variable (x y : ℝ)
variable (commute_times : List ℝ := [x, y, 10, 11, 9])

def mean (xs : List ℝ) : ℝ := (xs.sum) / (xs.length)

def variance (xs : List ℝ) (μ : ℝ) : ℝ :=
  (xs.map (λ x => (x - μ) ^ 2)).sum / (xs.length)

theorem commute_times_difference :
  mean commute_times = 10 →
  variance commute_times 10 = 2 →
  |x - y| = 4 :=
by
  sorry

end commute_times_difference_l257_257153


namespace no_five_distinct_natural_numbers_feasible_l257_257946

theorem no_five_distinct_natural_numbers_feasible :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
  sorry

end no_five_distinct_natural_numbers_feasible_l257_257946


namespace multiple_of_q_digit_sum_l257_257720

theorem multiple_of_q_digit_sum (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_eq : q = 2 * p + 1) (h_gt : p > 0) : 
  ∃ (n : ℕ), (n % q = 0) ∧ (nat.digits 10 n).sum ≤ 3 ∧ (nat.digits 10 n).sum > 0 := 
sorry

end multiple_of_q_digit_sum_l257_257720


namespace range_of_a_l257_257579

def discrim (a b c : ℝ) : ℝ := b * b - 4 * a * c

noncomputable def A : Set ℝ := { x | x^2 + 4 * x = 0 }
noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 + 2 * (a + 1) * x + (a^2 - 1) = 0 }

theorem range_of_a (a : ℝ) (h : B(a) ⊆ A) : a ≤ -1 ∨ a = 1 :=
sorry

end range_of_a_l257_257579


namespace problem_statement_l257_257586

noncomputable def incorrect_statement_D (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : Prop :=
  ∀ c : ℝ, (sqrt a + sqrt b = c) → (c >= sqrt 2 → false)

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : incorrect_statement_D a b h1 h2 h3 :=
sorry

end problem_statement_l257_257586


namespace luther_latest_line_count_l257_257388

theorem luther_latest_line_count :
  let silk := 10
  let cashmere := silk / 2
  let blended := 2
  silk + cashmere + blended = 17 :=
by
  sorry

end luther_latest_line_count_l257_257388


namespace minimum_buses_needed_l257_257814

/-- Variables describing the problem conditions -/
variables
  (distanceAB : ℕ) -- Distance between A and B
  (equal_interval : ℕ) -- The distance between two adjacent stops
  (interval_time : ℕ) -- Interval time in minutes before and after each stop
  (bus_speed : ℕ) -- Speed of the bus in kilometers per hour

/-- The main theorem that encapsulates the problem -/
theorem minimum_buses_needed
  (h1 : distanceAB = 15)
  (h2 : interval_time = 3)
  (h3 : bus_speed = 60)
  (h4 : equal_interval = (interval_time * bus_speed / 60)) : 
  (distanceAB / equal_interval * 2) = 10 :=
by sorry

end minimum_buses_needed_l257_257814


namespace solve_xyz_l257_257957

theorem solve_xyz (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
(h : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 2 * (sqrt (x + 2) + sqrt (y + 2) + sqrt (z + 2))):
  x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 := 
sorry

end solve_xyz_l257_257957


namespace final_result_after_subtracting_15_l257_257152

theorem final_result_after_subtracting_15 :
  ∀ (n : ℕ) (r : ℕ) (f : ℕ),
  n = 120 → 
  r = n / 6 → 
  f = r - 15 → 
  f = 5 :=
by
  intros n r f hn hr hf
  have h1 : n = 120 := hn
  have h2 : r = n / 6 := hr
  have h3 : f = r - 15 := hf
  sorry

end final_result_after_subtracting_15_l257_257152


namespace train_travel_time_through_tunnel_l257_257461

/-- Define the length of the train (in meters) -/
def train_length : ℝ := 100

/-- Define the speed of the train (in km/hr) -/
def train_speed_km_hr : ℝ := 72

/-- Define the length of the tunnel (in km) -/
def tunnel_length_km : ℝ := 1.1

/-- Convert the speed of the train from km/hr to m/min -/
def train_speed_m_min : ℝ := (train_speed_km_hr * 1000) / 60

/-- Convert the length of the tunnel from km to meters -/
def tunnel_length_m : ℝ := tunnel_length_km * 1000

/-- Calculate the total distance to travel -/
def total_distance_to_travel : ℝ := tunnel_length_m + train_length

/-- Theorem stating the travel time to pass through the tunnel -/
theorem train_travel_time_through_tunnel : 
  (total_distance_to_travel / train_speed_m_min) = 1 :=
by
  -- placeholder for the proof
  sorry

end train_travel_time_through_tunnel_l257_257461


namespace ratio_is_76_over_95_l257_257181

-- Define the numerator sequence
def seq_num (n : ℕ) := 4 + 4 * (n - 1)

-- Define the denominator sequence
def seq_denom (n : ℕ) := 5 + 5 * (n - 1)

noncomputable def sum_arith_seq (a d n : ℕ) := n * (2 * a + (n - 1) * d) / 2

-- Define the number of terms in the numerator and the denominator sequences
def terms_num : ℕ := 18 -- Derived from solving 4 + (n-1) * 4 = 72
def terms_denom : ℕ := 18 -- Derived from solving 5 + (n-1) * 5 = 90

-- Define the sums of the sequences
noncomputable def S_num := sum_arith_seq 4 4 terms_num
noncomputable def S_denom := sum_arith_seq 5 5 terms_denom

-- Define the ratio of the sums
noncomputable def ratio := S_num / S_denom

-- The proof problem
theorem ratio_is_76_over_95 : ratio = 76 / 95 := by
  unfold ratio S_num S_denom sum_arith_seq terms_num terms_denom seq_num seq_denom
  sorry

end ratio_is_76_over_95_l257_257181


namespace perfect_square_sum_l257_257019

-- Define the numbers based on the given conditions
def A (n : ℕ) : ℕ := 4 * (10^(2 * n) - 1) / 9
def B (n : ℕ) : ℕ := 2 * (10^(n + 1) - 1) / 9
def C (n : ℕ) : ℕ := 8 * (10^n - 1) / 9

-- Define the main theorem to be proved
theorem perfect_square_sum (n : ℕ) : 
  ∃ k, A n + B n + C n + 7 = k * k :=
sorry

end perfect_square_sum_l257_257019


namespace ellipse_major_axis_length_l257_257921

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem ellipse_major_axis_length :
  let F1 := (5, 10)
  let F2 := (35, 40)
  let F1' := (-5, 10)
  in
  distance F1' F2 = 50 :=
by
  -- the detailed proof steps would be filled in here
  sorry

end ellipse_major_axis_length_l257_257921


namespace matrix_vector_multiplication_correct_l257_257526

noncomputable def mat : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![1, 5]]
noncomputable def vec : Fin 2 → ℤ := ![-1, 2]
noncomputable def result : Fin 2 → ℤ := ![-7, 9]

theorem matrix_vector_multiplication_correct :
  (Matrix.mulVec mat vec) = result :=
by
  sorry

end matrix_vector_multiplication_correct_l257_257526


namespace Dave_spent_on_books_l257_257939

theorem Dave_spent_on_books :
  let cost_animals := 8 * 10
  let cost_outer_space := 6 * 12
  let cost_trains := 9 * 8
  let cost_history := 4 * 15
  let cost_science := 5 * 18
  cost_animals + cost_outer_space + cost_trains + cost_history + cost_science = 374 :=
by
  let cost_animals := 8 * 10
  let cost_outer_space := 6 * 12
  let cost_trains := 9 * 8
  let cost_history := 4 * 15
  let cost_science := 5 * 18
  have h_animals : cost_animals = 80 := rfl
  have h_outer_space : cost_outer_space = 72 := rfl
  have h_trains : cost_trains = 72 := rfl
  have h_history : cost_history = 60 := rfl
  have h_science : cost_science = 90 := rfl
  calc
    80 + 72 + 72 + 60 + 90 = 374 : sorry

end Dave_spent_on_books_l257_257939


namespace Davey_Barbeck_ratio_is_1_l257_257929

-- Assume the following given conditions as definitions in Lean
variables (guitars Davey Barbeck : ℕ)

-- Condition 1: Davey has 18 guitars
def Davey_has_18 : Prop := Davey = 18

-- Condition 2: Barbeck has the same number of guitars as Davey
def Davey_eq_Barbeck : Prop := Davey = Barbeck

-- The problem statement: Prove the ratio of the number of guitars Davey has to the number of guitars Barbeck has is 1:1
theorem Davey_Barbeck_ratio_is_1 (h1 : Davey_has_18 Davey) (h2 : Davey_eq_Barbeck Davey Barbeck) :
  Davey / Barbeck = 1 :=
by
  sorry

end Davey_Barbeck_ratio_is_1_l257_257929


namespace sum_of_roots_of_Q_l257_257662

-- The problem involves a quadratic polynomial Q(x) with real coefficients 
-- satisfying the condition Q(x^3 + 2x) ≥ Q(x^2 + 3) for all x ∈ ℝ.
-- We need to prove that the sum of the roots of Q(x) = ax^2 + bx + c is 4/5.

theorem sum_of_roots_of_Q (a b c : ℝ) 
  (Q : ℝ → ℝ) (hQ : ∀ x, Q (x^3 + 2 * x) ≥ Q (x^2 + 3))
  (hQ_def : ∀ x, Q x = a * x^2 + b * x + c) :
  let sum_of_roots := -b/a in
  sum_of_roots = 4 / 5 :=
sorry

end sum_of_roots_of_Q_l257_257662


namespace max_y_over_x_l257_257984

noncomputable def complex_modulus_condition (x y : ℝ) (z : ℂ) : Prop :=
  z = x + y * complex.I ∧ abs (z - 2) = √3

theorem max_y_over_x (x y : ℝ) (z : ℂ) (h : complex_modulus_condition x y z) :
  y / x ≤ √3 :=
begin
  sorry
end

end max_y_over_x_l257_257984


namespace solve_for_y_l257_257410

theorem solve_for_y (y : ℝ) (h : sqrt(1 + sqrt(4 * y - 5)) = sqrt 8) : y = 13.5 :=
by
  sorry

end solve_for_y_l257_257410


namespace count_valid_bases_l257_257555

-- Given conditions and definitions
def valid_base (b : ℕ) : Prop :=
  2 ≤ b ∧ b ≤ 15 ∧ 575 % b = 0

-- Problem statement
theorem count_valid_bases : 
  (finset.filter valid_base (finset.range 16)).card = 1 :=
by sorry

end count_valid_bases_l257_257555


namespace max_area_of_rectangular_pen_l257_257598

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l257_257598


namespace decreasing_intervals_range_g_on_interval_l257_257258

noncomputable section

def f (x : ℝ) : ℝ := 2 * Cos x ^ 2 + 2 * Real.sqrt 3 * Sin x * Cos x

def g (x : ℝ) : ℝ := 2 * Sin (4 * x + π / 3) + 1

theorem decreasing_intervals (k : ℤ) : 
  monotone_decreasing_on f (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) := sorry

theorem range_g_on_interval : 
  Set.range (g ∘ (Set.Icc 0 (π / 4))) = Set.Icc (1 - Real.sqrt 3) 3 := sorry

end decreasing_intervals_range_g_on_interval_l257_257258


namespace max_rectangle_area_l257_257635

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l257_257635


namespace roundness_720_l257_257003

noncomputable def prime_factors_exponents (n : ℕ) : List ℕ :=
if h : n > 1 then
  let factors := Nat.factors n
  factors.foldl (λ acc x, acc x) 0
else 
  []

theorem roundness_720 : List.sum (prime_factors_exponents 720) = 7 :=
by
  sorry

end roundness_720_l257_257003


namespace driver_schedule_l257_257042

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l257_257042


namespace smallest_distinct_even_geometric_4digit_l257_257448

theorem smallest_distinct_even_geometric_4digit : ∃ (n : ℕ), 
  1000 ≤ n ∧ n < 10000 ∧ 
  (∀ i j, i ≠ j → n.digit i ≠ n.digit j) ∧ 
  (∃ (a r : ℕ), n = 1000 * a + 100 * (a * r) + 10 * (a * r^2) + (a * r^3)) ∧ 
  n % 10 % 2 = 0 ∧ 
  ∀ (m : ℕ), 1000 ≤ m ∧ m < 10000 ∧ 
  (∀ i j, i ≠ j → m.digit i ≠ m.digit j) ∧ 
  (∃ (a' r' : ℕ), m = 1000 * a' + 100 * (a' * r') + 10 * (a' * r'^2) + (a' * r'^3)) ∧ 
  m % 10 % 2 = 0 → n ≤ m :=
begin
  -- Proof goes here
  sorry
end

end smallest_distinct_even_geometric_4digit_l257_257448


namespace smallest_possible_area_of_triangle_is_zero_l257_257372

variable (s : ℝ)

def point_A := (0 : ℝ, 2, 1)
def point_B := (2 : ℝ, 3, 4)
def point_C := (s, 2, 1)

def vector_sub (x y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (x.1 - y.1, x.2 - y.2, x.3 - y.3)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def norm (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2 + u.3^2)

def area_of_triangle (A B C : ℝ × ℝ × ℝ) : ℝ :=
  1 / 2 * norm (cross_product (vector_sub B A) (vector_sub C A))

theorem smallest_possible_area_of_triangle_is_zero : ∃ (s : ℝ), area_of_triangle point_A point_B (point_C s) = 0 :=
by
  sorry

end smallest_possible_area_of_triangle_is_zero_l257_257372


namespace problem_statement_l257_257580

-- Define the vectors a and b
def a := (3 : ℝ, 1 : ℝ)
def b (α : ℝ) := (Real.sin α, Real.cos α)

-- Define the condition for parallel vectors
def vectors_parallel (α : ℝ) : Prop := a.1 / (b α).1 = a.2 / (b α).2

-- Define the main theorem to be proved
theorem problem_statement (α : ℝ) (h_parallel : vectors_parallel α) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := 
sorry

end problem_statement_l257_257580


namespace projection_matrix_correct_l257_257724

noncomputable def Q_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1/9, 5/9, 8/9], 
    ![7/9, 8/9, 2/9], 
    ![4/9, 2/9, 5/9]]

theorem projection_matrix_correct : 
  ∀ u : Fin 3 → ℝ,
  let n := ![2, 1, -2] in
  let Q : Matrix (Fin 3) (Fin 3) ℝ := Q_matrix in
  (Q.mulVec u) = u - ((((u ⬝ n) / (n ⬝ n)) • n)) :=
sorry

end projection_matrix_correct_l257_257724


namespace gcf_45_135_90_l257_257085

def gcd (a b : Nat) : Nat := Nat.gcd a b

noncomputable def gcd_of_three (a b c : Nat) : Nat :=
  gcd (gcd a b) c

theorem gcf_45_135_90 : gcd_of_three 45 135 90 = 45 := by
  sorry

end gcf_45_135_90_l257_257085


namespace amusement_park_admission_fees_l257_257413

theorem amusement_park_admission_fees
  (num_children : ℕ) (num_adults : ℕ)
  (fee_child : ℝ) (fee_adult : ℝ)
  (total_people : ℕ) (expected_total_fees : ℝ) :
  num_children = 180 →
  fee_child = 1.5 →
  fee_adult = 4.0 →
  total_people = 315 →
  expected_total_fees = 810 →
  num_children + num_adults = total_people →
  (num_children : ℝ) * fee_child + (num_adults : ℝ) * fee_adult = expected_total_fees := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amusement_park_admission_fees_l257_257413


namespace area_QMN_eq_10_l257_257709

-- Define the structures and conditions
variables (P Q R M N : Point)
variables (triangle_PQR : Triangle P Q R)
variables (midpoint_M : is_midpoint M P Q)
variables (midpoint_N : is_midpoint N P R)
variables (area_PQR : area triangle_PQR = 40)

-- Prove the required statement
theorem area_QMN_eq_10 : area (Triangle Q M N) = 10 :=
by sorry

end area_QMN_eq_10_l257_257709


namespace min_ab_l257_257684

theorem min_ab (a b : ℝ) (h_cond1 : a > 0) (h_cond2 : b > 0)
  (h_eq : a * b = a + b + 3) : a * b = 9 :=
sorry

end min_ab_l257_257684


namespace total_puppies_is_74_l257_257844

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l257_257844


namespace common_difference_of_arithmetic_sequence_l257_257983

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : a 5 = 10) (h3 : a 10 = -5) : d = -3 := 
by 
  sorry

end common_difference_of_arithmetic_sequence_l257_257983


namespace side_c_length_l257_257374

namespace TriangleProblem

variables {a b c : ℝ} {A B C : ℝ}
variables (h1 : sin A * cos B + sin B * cos A = sin (2 * C))
variables (h2 : a + c = 2 * b)
variables (h3 : (a * b * real.cos C) = 18)

theorem side_c_length : c = 6 :=
sorry

end TriangleProblem

end side_c_length_l257_257374


namespace arrangement_count_l257_257059

-- Definitions
def people : Type := {A B C D E : ℕ // A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E}

def arrangements (p : people) : set (list ℕ) := 
  {l | l.length = 5 ∧
        p.A ∈ l ∧ p.B ∈ l ∧ p.C ∈ l ∧ p.D ∈ l ∧ p.E ∈ l ∧ 
        (∀ i, l !! i = some p.A → l !! (i+1) = some p.B ∨ l !! (i-1) = some p.B) ∧
        (∀ i, l !! i = some p.B → l !! (i+1) = some p.A ∨ l !! (i-1) = some p.A) ∧
        (∀ i, l !! i = some p.D → ((l !! (i+1)) ≠ some p.A ∧ (l !! (i-1)) ≠ some p.A ∧ (l !! (i+1)) ≠ some p.B ∧ (l !! (i-1)) ≠ some p.B)) }

-- Theorem Statement
theorem arrangement_count (p : people) : 
  fincard arrangements p = 36 :=
by
  sorry

end arrangement_count_l257_257059


namespace diamond_example_l257_257220

def diamond (a b : ℝ) : ℝ := Real.sqrt (a*a + b*b)

theorem diamond_example :
  diamond (diamond 5 12) (diamond (-12) (-5)) = 13 * Real.sqrt 2 := by
  sorry

end diamond_example_l257_257220


namespace drivers_schedule_l257_257027

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l257_257027


namespace toothbrushes_given_l257_257487

theorem toothbrushes_given (hours_per_visit : ℝ) (hours_per_day : ℝ) (days_per_week : ℕ) (toothbrushes_per_visit : ℕ) 
  (h1 : hours_per_visit = 0.5) 
  (h2 : hours_per_day = 8) 
  (h3 : days_per_week = 5) 
  (h4 : toothbrushes_per_visit = 2) : 
  (toothbrushes_per_visit * (nat.floor (hours_per_day / hours_per_visit) * days_per_week)) = 160 :=
by
  sorry

end toothbrushes_given_l257_257487


namespace rabbit_weight_l257_257157

theorem rabbit_weight (a b c : ℕ) (h1 : a + b + c = 30) (h2 : a + c = 2 * b) (h3 : a + b = c) :
  a = 5 := by
  sorry

end rabbit_weight_l257_257157


namespace distinct_integer_values_b_for_quadratic_l257_257569

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end distinct_integer_values_b_for_quadratic_l257_257569


namespace max_area_of_rectangular_pen_l257_257609

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l257_257609


namespace number_of_candidates_l257_257458

theorem number_of_candidates
  (n : ℕ)
  (h : n * (n - 1) = 132) : 
  n = 12 :=
sorry

end number_of_candidates_l257_257458


namespace log_expression_equality_l257_257545

theorem log_expression_equality : 
  ∀ (a : ℝ), 
    (log 2 3 = a ∧ log 8 3 = a / 3 ∧ log 4 9 = a) → 
    (a + a / 3) / a = 4 / 3 :=
by
  intros a h
  cases h with h1 h_temp
  cases h_temp with h2 h3
  sorry

end log_expression_equality_l257_257545


namespace euler_characteristic_split_l257_257380

theorem euler_characteristic_split (S : Surface) (C : S.Circle) (S' S'' : Surface) :
  (cut_and_cap S C S' S'') → (χ(S) = χ(S') + χ(S'')) :=
by
  intros
  sorry -- Proof goes here

end euler_characteristic_split_l257_257380


namespace total_opponents_runs_l257_257872

-- Define the runs scored by the team in 7 games
def team_runs : List ℕ := [1, 3, 5, 6, 7, 8, 10]

-- Define a predicate stating the team lost by 2 runs in exactly 3 games
def lost_by_two (game_runs : ℕ) : Prop :=
  ∃ opponent_runs, opponent_runs = game_runs + 2

-- Define a predicate stating the team scored 3 times as many runs as opponent in a game
def three_times_as_many (game_runs : ℕ) : Prop :=
  ∃ opponent_runs, game_runs = 3 * opponent_runs

-- Define a proof problem asserting the total runs of opponents
theorem total_opponents_runs :
  let opponent_runs : List ℕ := [3, 5, 7, 2, 2, 2, 3] in
  opponent_runs.sum = 24 := 
by
  sorry

end total_opponents_runs_l257_257872


namespace greatest_number_of_factors_l257_257056

theorem greatest_number_of_factors (b n : ℕ) (hb : 1 ≤ b ∧ b ≤ 15) (hn : 1 ≤ n ∧ n ≤ 15) : 
  nat_factors_count(b^n) <= 496 :=
  sorry

-- Helper definition for the number of factors of a number
def nat_factors_count (m : ℕ) : ℕ :=
  if h : m > 0 then
    (finset.range m).filter m.dvd.length + 1
  else
    0

end greatest_number_of_factors_l257_257056


namespace find_m_value_l257_257150

theorem find_m_value
  (n : ℕ)
  (m : ℕ)
  (sum_n : ℕ)
  (list : list ℕ)
  (mode35 : ∀ k ∈ list, k = 35 → list.count k ≥ 2)
  (mean30 : sum_n = 30 * n)
  (smallest20 : list.min = some 20)
  (median_m : ∃ l r, list.sorted (<) ∧ list = l ++ [m] ++ r ∧ l.length = r.length)
  (replace_m_plus_8 :
    ∀ new_list : list ℕ, new_list = (list.erase m).insert (m+8) →
    (∃ l' r', new_list = l' ++ [m+8] ++ r' ∧ l'.length = r'.length) ∧
    (let sum_new := new_list.sum in sum_new = 32 * n))
  (replace_m_minus_10 :
    ∀ new_list2 : list ℕ, new_list2 = (list.erase m).insert (m-10) →
    (∃ l'' r'', new_list2 = l'' ++ [m-5] ++ r'' ∧ l''.length = r''.length)) :
  m = 35 :=
sorry

end find_m_value_l257_257150


namespace toothbrushes_given_in_week_l257_257489

theorem toothbrushes_given_in_week :
  ∀ (toothbrushes_per_patient : ℕ) (visit_duration : ℝ) (hours_per_day : ℝ) (days_per_week : ℕ),
  toothbrushes_per_patient = 2 →
  visit_duration = 0.5 →
  hours_per_day = 8 →
  days_per_week = 5 →
  (toothbrushes_per_patient * (hours_per_day / visit_duration).to_nat * days_per_week) = 160 :=
by
  intros toothbrushes_per_patient visit_duration hours_per_day days_per_week
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end toothbrushes_given_in_week_l257_257489


namespace sin_C_of_arithmetic_sequence_l257_257711

theorem sin_C_of_arithmetic_sequence 
  (A B C : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = Real.pi) 
  (h3 : Real.cos A = 2 / 3) 
  : Real.sin C = (Real.sqrt 5 + 2 * Real.sqrt 3) / 6 :=
sorry

end sin_C_of_arithmetic_sequence_l257_257711


namespace rhombus_area_l257_257894

theorem rhombus_area (s : ℝ) (theta : ℝ) (h_s : s = 13) (h_theta : theta = 60) :
  let A := s^2 * Real.sin (theta * Real.pi / 180) in A = 169 * Real.sqrt 3 / 2 :=
by
  sorry

end rhombus_area_l257_257894


namespace arithmetic_geometric_sequence_l257_257780

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
    (h1 : a 1 = 3)
    (h2 : a 1 + a 3 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 * a 4 = 36 := 
sorry

end arithmetic_geometric_sequence_l257_257780


namespace projection_onto_plane_l257_257726

open Matrix Vec

def n : Vec ℝ 3 := ⟨2, 1, -2⟩

def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![5/9, -2/9, 4/9],
    ![-2/9, 8/9, 2/9],
    ![4/9, 2/9, 5/9]
  ]

theorem projection_onto_plane (u : Vec ℝ 3) :
  ((Q ⬝ u) : Three) = proj u :=
sorry

end projection_onto_plane_l257_257726


namespace sin_cos_value_l257_257305

theorem sin_cos_value (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_value_l257_257305


namespace transformed_variance_l257_257057

variables {n : ℕ} {a : ℕ → ℝ}
variable (var_a : variance (list.of_fn a) = 1)

theorem transformed_variance : variance (list.of_fn (λ i, 2 * a i - 1)) = 4 :=
by sorry

end transformed_variance_l257_257057


namespace saved_percent_l257_257860

-- Definitions for conditions:
def last_year_saved (S : ℝ) : ℝ := 0.10 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_saved (S : ℝ) : ℝ := 0.06 * (1.10 * S)

-- Given conditions and proof goal:
theorem saved_percent (S : ℝ) (hl_last_year_saved : last_year_saved S = 0.10 * S)
  (hl_this_year_salary : this_year_salary S = 1.10 * S)
  (hl_this_year_saved : this_year_saved S = 0.066 * S) :
  (this_year_saved S / last_year_saved S) * 100 = 66 :=
by
  sorry

end saved_percent_l257_257860


namespace plot_length_l257_257122

noncomputable def length_of_plot (breadth cost_fencing total_cost : ℝ) : ℝ :=
  let perimeter := 2 * (breadth + 50) + 2 * breadth
  in if cost_fencing * perimeter = total_cost
     then breadth + 50
     else 0 -- or some placeholder value indicating invalid conditions

theorem plot_length (cost_fencing total_cost : ℝ) (breadth : ℝ) :
  cost_fencing = 26.50 ∧ total_cost = 5300 ∧ 
  (2 * (breadth + 50) + 2 * breadth) * cost_fencing = total_cost →
  length_of_plot breadth cost_fencing total_cost = 75 :=
begin
  sorry,
end

end plot_length_l257_257122


namespace jacos_budget_l257_257355

theorem jacos_budget :
  (friends : Nat) (friend_gift_cost total_budget : Nat)
  (jaco_remainder_budget : Nat)
  : friends = 8 →
  friend_gift_cost = 9 →
  total_budget = 100 →
  jaco_remainder_budget = total_budget - (friends * friend_gift_cost) →
  (jaco_remainder_budget / 2) = 14 := by
  intros friends friend_gift_cost total_budget jaco_remainder_budget friends_eq friend_gift_cost_eq total_budget_eq jaco_remainder_budget_eq
  rw [friends_eq, friend_gift_cost_eq, total_budget_eq, jaco_remainder_budget_eq]
  simp
  sorry

end jacos_budget_l257_257355


namespace find_an_find_bn_find_Tn_l257_257253

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Definitions based on the conditions
def an_arithmetic (n : ℕ) : Prop := ∀ k : ℕ, k > 0 → a (k + 1) = a k + 2
def a3_a4 := a 3 + a 4 = 12
def sum_sn := ∀ n : ℕ, S n = ∑ i in range (n + 1), a i

def bn_geometric (n : ℕ) : Prop := b 1 = a 2 ∧ b 2 = S 3 ∧ ∀ k : ℕ, k > 0 → b (k + 1) = b k * 3
def cn_definition (n : ℕ) : Prop := c n = (-1)^n * a n * b n

-- Proof statements based on the questions and answers
theorem find_an : an_arithmetic n → a3_a4 → (∀ n, a n = 2 * n - 1) := sorry

theorem find_bn : sum_sn → bn_geometric n → (∀ n, b n = 3 ^ n) := sorry

theorem find_Tn (T : ℕ → ℕ) : 
  (∀ n, a n = 2 * n - 1) → 
  (∀ n, b n = 3 ^ n) → 
  (∀ n, T n = (∑ i in range (n + 1), c i)) → 
  (∀ n, T n = (3 / 8 - 9 * (4 * n - 1) / 8 * (-3)^(n-1))) := sorry

end find_an_find_bn_find_Tn_l257_257253


namespace tangent_line_value_of_a_l257_257683

theorem tangent_line_value_of_a (a : ℝ) :
  (∃ (m : ℝ), (2 * m - 1 = a * m + Real.log m) ∧ (a + 1 / m = 2)) → a = 1 :=
by 
sorry

end tangent_line_value_of_a_l257_257683


namespace playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l257_257236

def hasWinningStrategyA (n : ℕ) : Prop :=
  n ≥ 8

def hasWinningStrategyB (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5

def draw (n : ℕ) : Prop :=
  n = 6 ∨ n = 7

theorem playerA_winning_strategy (n : ℕ) : n ≥ 8 → hasWinningStrategyA n :=
by
  sorry

theorem playerB_winning_strategy (n : ℕ) : (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) → hasWinningStrategyB n :=
by
  sorry

theorem no_winning_strategy (n : ℕ) : n = 6 ∨ n = 7 → draw n :=
by
  sorry

end playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l257_257236


namespace complex_power_identity_l257_257968

theorem complex_power_identity : ( (1 - Complex.i) / (1 + Complex.i) ) ^ 10 = -1 := 
by 
  sorry

end complex_power_identity_l257_257968


namespace max_area_of_rectangular_pen_l257_257605

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l257_257605


namespace part_a_l257_257922

structure is_interesting_equation (P Q : Polynomial ℤ) : Prop :=
  (degree_pos : P.degree ≥ 1 ∧ Q.degree ≥ 1)
  (inf_solutions : ∃∞ n : ℕ, P.eval n = Q.eval n)

structure yields_in (P Q F G : Polynomial ℤ) : Prop :=
  (exists_polynomial : ∃ R : Polynomial ℚ, ∀ x, F.eval x = R.eval (P.eval x) ∧ G.eval x = R.eval (Q.eval x))

def infinite_subset (S : set (ℕ × ℕ)) : Prop := 
  set.infinite S

theorem part_a {S : set (ℕ × ℕ)} (hS : infinite_subset S) :
  ∃ (P₀ Q₀ : Polynomial ℤ), 
    is_interesting_equation P₀ Q₀ ∧ 
    (∀ {P Q : Polynomial ℤ}, (is_interesting_equation P Q) → (∀ (x y : ℕ × ℕ), (x, y) ∈ S → P.eval x = Q.eval y) → yields_in P₀ Q₀ P Q) := sorry

end part_a_l257_257922


namespace problem_l257_257862

noncomputable def f (x : ℝ) : Set ℝ := sorry

theorem problem 
  (f : ℝ → Set ℝ)
  (A : Set (Set ℝ)) (hA : ∀ a b : ℝ, [a, b] ∈ A)
  (cond1 : ∀ x y : ℝ, x ∈ f y ↔ y ∈ f x)
  (cond2 : ∀ x y : ℝ, |x - y| > 2 ↔ f x ∩ f y = ∅)
  (cond3 : ∀ r : ℝ, 0 ≤ r ∧ r ≤ 1 → f r = (set.Icc (r^2 - 1) (r^2 + 1))) :
  ∃ g h : ℝ → ℝ, ∀ x : ℝ, f x = set.Icc (g x) (h x) :=
begin
  sorry
end

end problem_l257_257862


namespace det_equal_l257_257370

theorem det_equal (n : ℕ) (A B M : Matrix (Fin n) (Fin n) ℝ) (h1 : A * M = M * B) 
    (h2 : A.charPoly = B.charPoly) (X : Matrix (Fin n) (Fin n) ℝ) : 
    det (A - M * X) = det (B - X * M) :=
  sorry

end det_equal_l257_257370


namespace jaco_budget_for_parents_l257_257353

/-- Assume Jaco has 8 friends, each friend's gift costs $9, and Jaco has a total budget of $100.
    Prove that Jaco's budget for each of his mother and father's gift is $14. -/
theorem jaco_budget_for_parents :
  ∀ (friends_count cost_per_friend total_budget : ℕ), 
  friends_count = 8 → 
  cost_per_friend = 9 → 
  total_budget = 100 → 
  (total_budget - friends_count * cost_per_friend) / 2 = 14 :=
by
  intros friends_count cost_per_friend total_budget h1 h2 h3
  rw [h1, h2, h3]
  have friend_total_cost : friends_count * cost_per_friend = 72 := by norm_num
  have remaining_budget : total_budget - friends_count * cost_per_friend = 28 := by norm_num [friend_total_cost]
  have divided_budget : remaining_budget / 2 = 14 := by norm_num [remaining_budget]
  exact divided_budget

end jaco_budget_for_parents_l257_257353


namespace eval_fraction_l257_257951

theorem eval_fraction :
  (10 ^ (-2) * 5 ^ 0) / 10 ^ (-3) = 10 :=
by
  have h1 : 10 ^ (-2) = 1 / 10 ^ 2 := by sorry
  have h2 : 5 ^ 0 = 1 := by sorry
  have h3 : 1 / 10 ^ (-3) = 10 ^ 3 := by sorry
  -- Utilize the above conditions to prove the main statement
  sorry

end eval_fraction_l257_257951


namespace median_unchanged_after_removal_l257_257506

theorem median_unchanged_after_removal (s : Finset ℝ) (h_distinct: s.card ≥ 3) :
  let s' : Finset ℝ := s.erase (Finset.max' s (by sorry)).erase (Finset.min' s (by sorry)) in
  (Finset.card s' % 2 = 1) →
  (∃ m ∈ s', (Finset.median s') = m) :=
by
  sorry

end median_unchanged_after_removal_l257_257506


namespace problem_equivalent_l257_257285

theorem problem_equivalent :
  2^1998 - 2^1997 - 2^1996 + 2^1995 = 3 * 2^1995 :=
by
  sorry

end problem_equivalent_l257_257285


namespace integer_values_of_b_for_quadratic_eqn_l257_257560

noncomputable def number_of_integer_values_of_b : ℕ := 16

theorem integer_values_of_b_for_quadratic_eqn :
  ∃(b : ℤ) (k ≥ 0), ∀m n : ℤ, (m + n = -b ∧ m * n = 12 * b) → (m + 12) * (n + 12) = 144 → k = number_of_integer_values_of_b := sorry

end integer_values_of_b_for_quadratic_eqn_l257_257560


namespace magnitude_of_complex_fraction_l257_257230

def complex_magnitude (z : ℂ) : ℝ :=
Complex.abs z

theorem magnitude_of_complex_fraction : 
  complex_magnitude (5 * Complex.I / (2 + Complex.I)) = Real.sqrt 5 :=
by
  sorry

end magnitude_of_complex_fraction_l257_257230


namespace min_log_expression_is_zero_l257_257214

open Real

noncomputable def min_log_expression (a b c : ℝ) (h_cond : a ≥ b ∧ b ≥ c ∧ c > 1) : ℝ :=
  (log a / log (a^3 / b)) + (log b / log (b^3 / c))

theorem min_log_expression_is_zero (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c > 1) :
  min_log_expression a b c h = 0 :=
sorry

end min_log_expression_is_zero_l257_257214


namespace Eliza_height_l257_257949

theorem Eliza_height (H : ℕ) :
  let S := H + 4 in
  let total_height := 435 in
  let height_2_siblings := 2 * 66 in
  let height_4th_sibling := 60 in
  let height_5th_sibling := 75 in
  let known_siblings_height := height_2_siblings + height_4th_sibling + height_5th_sibling in
  let combined_height := total_height - known_siblings_height in
  H + S = combined_height → H = 82 :=
by
  sorry

end Eliza_height_l257_257949


namespace max_product_of_two_integers_l257_257088

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l257_257088


namespace coloring_sectors_l257_257945

noncomputable def ways_to_color_sectors (n m : ℕ) : ℕ :=
  if h : n ≥ 2 ∧ m ≥ 2 then (m - 1) ^ n + (-1) ^ n * (m - 1) else 0

theorem coloring_sectors (n m : ℕ) (h1 : n ≥ 2) (h2 : m ≥ 2) :
    ways_to_color_sectors n m = (m - 1) ^ n + (-1) ^ n * (m - 1) :=
by 
    simp [ways_to_color_sectors, h1, h2]
    sorry

end coloring_sectors_l257_257945


namespace dishonest_dealer_profit_l257_257490

theorem dishonest_dealer_profit (weight_claimed weight_actual : ℕ) (h₁ : weight_claimed = 1000) (h₂ : weight_actual = 723) :
  let profit := weight_claimed - weight_actual in
  let profit_percent := (profit * 100) / weight_claimed in
  profit_percent = 27.7 :=
by
  -- Definitions for clarity
  let profit := weight_claimed - weight_actual
  let profit_percent := (profit * 100) / weight_claimed
  -- We will calculate this in the proof (omitted for now)
  sorry

end dishonest_dealer_profit_l257_257490


namespace sugar_amount_l257_257859

theorem sugar_amount (S F B : ℝ) 
    (h_ratio1 : S = F) 
    (h_ratio2 : F = 10 * B) 
    (h_ratio3 : F / (B + 60) = 8) : S = 2400 := 
by
  sorry

end sugar_amount_l257_257859


namespace max_value_of_g_l257_257218

def g (x : ℝ) : ℝ := min (3 * x + 3) (min ((2 / 3) * x + 2) (-x + 9))

theorem max_value_of_g : ∃ x : ℝ, g(x) = 24 / 5 :=
by
  use 21 / 5
  have h : g(21 / 5) = min (3 * (21 / 5) + 3) (min ((2 / 3) * (21 / 5) + 2) (-21 / 5 + 9))
    by
      sorry
  rw h
  exact rfl

end max_value_of_g_l257_257218


namespace no_valid_n_l257_257217

def is_greatest_prime_factor_sqrt (n : ℕ) : Prop :=
  let p := Nat.sqrt n in
  p * p = n ∧ p.Prime ∧ ∀ q : ℕ, q.Prime → q ∣ n → q ≤ p

theorem no_valid_n : ∀ (n : ℕ), is_greatest_prime_factor_sqrt n ∧ is_greatest_prime_factor_sqrt (n + 60) → false :=
by
  intro n
  sorry

end no_valid_n_l257_257217


namespace sqrt_inequality_l257_257077

theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  sqrt (a + 1) + sqrt (a - 1) < 2 * sqrt a :=
sorry

end sqrt_inequality_l257_257077


namespace exist_triplets_l257_257078

open Finset

variable {α : Type}

theorem exist_triplets 
  (n : ℕ)
  (triples : Finset (Finset α))
  (h_triples_size : triples.card = 10 * n^2) 
  (h_elem_size : (univ : Finset α).card = n) : 
  ∃ a b c d e f : α, 
    ({a, b, d} ∈ triples) ∧ 
    ({b, c, e} ∈ triples) ∧ 
    ({c, a, f} ∈ triples) := 
sorry

end exist_triplets_l257_257078


namespace calc_num_int_values_l257_257293

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l257_257293


namespace tile_selection_probability_l257_257541

theorem tile_selection_probability :
  let M := {'M', 'A', 'T', 'H', 'C', 'O', 'U', 'N', 'T', 'S'}
  let T := {'T', 'E', 'A', 'C', 'H'}
  (|M ∩ T| / |M| = 1 / 2) :=
by
  let M := {'M', 'A', 'T', 'H', 'C', 'O', 'U', 'N', 'T', 'S'}
  let T := {'T', 'E', 'A', 'C', 'H'}
  let favorable := {'A', 'T', 'H', 'C', 'T'}
  let total := 10
  have h_favorable : |favorable| = 5 := sorry
  have h_total : |M| = total := sorry
  have p := (|favorable| / total)
  show p = 1 / 2, sorry

end tile_selection_probability_l257_257541


namespace rectangles_same_area_l257_257434

theorem rectangles_same_area (x y : ℕ) 
  (h1 : x * y = (x + 4) * (y - 3)) 
  (h2 : x * y = (x + 8) * (y - 4)) : x + y = 10 := 
by
  sorry

end rectangles_same_area_l257_257434


namespace problem_proof_l257_257452

-- Define I, J, and K respectively to be 9^20, 3^41, 3
def I : ℕ := 9^20
def J : ℕ := 3^41
def K : ℕ := 3

theorem problem_proof : I + I + I = J := by
  -- Lean structure placeholder
  sorry

end problem_proof_l257_257452


namespace wider_can_radius_l257_257817

theorem wider_can_radius (h : ℝ) : 
  (∃ r : ℝ, ∀ V : ℝ, V = π * 8^2 * 2 * h → V = π * r^2 * h → r = 8 * Real.sqrt 2) :=
by 
  sorry

end wider_can_radius_l257_257817


namespace measure_of_angle_A_l257_257342

noncomputable def triangle_angle_measure (a b c : ℝ) (h : a * Real.cos C = (2 * b - c) * Real.cos A) : Prop :=
  A = π / 3

noncomputable def triangle_area_range (a b c : ℝ) (h_a : a = 3) (h_bc : 0 < b * c ∧ b * c ≤ 9) : Prop :=
  let S := 1 / 2 * b * c * Real.sin (π / 3)
  0 < S ∧ S ≤ 9 * Real.sqrt 3 / 4

theorem measure_of_angle_A
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a * Real.cos C = (2 * b - c) * Real.cos A)
  (ha : a = 3)
  (h_bc : 0 < b * c ∧ b * c ≤ 9) :
  triangle_angle_measure a b c h1 ∧
  triangle_area_range a b c ha h_bc := by
  sorry

end measure_of_angle_A_l257_257342


namespace profit_percentage_is_22_percent_l257_257748

-- Define the given conditions
def scooter_cost (C : ℝ) := C
def repair_cost (C : ℝ) := 0.10 * C
def repair_cost_value := 500
def profit := 1100

-- Let's state the main theorem
theorem profit_percentage_is_22_percent (C : ℝ) 
  (h1 : repair_cost C = repair_cost_value)
  (h2 : profit = 1100) : 
  (profit / C) * 100 = 22 :=
by
  sorry

end profit_percentage_is_22_percent_l257_257748


namespace divide_sqrt_of_16_is_two_l257_257833

theorem divide_sqrt_of_16_is_two:
  sqrt 16 / x = 2 → x = 2 :=
by
  sorry

end divide_sqrt_of_16_is_two_l257_257833


namespace pencils_shared_l257_257879

theorem pencils_shared (T N : ℕ) (dozen : ℕ) (H1 : dozen = 12) (H2 : 2 * dozen = 24) (H3 : N = T + 4) (H4 : T + N = 24) :
  N = 14 :=
by
  unfold dozen at *
  have H2 : 24 = 24 := rfl
  have twoT_plus_4 := calc
    2 * T + 4 = 24 : by sorry
  have T := calc
    2 * T = 20 : by sorry
    T = 10 : by sorry
  exact calc
    N = 10 + 4 : by sorry
    N = 14 : by rfl

end pencils_shared_l257_257879


namespace arrange_plants_l257_257924

-- Define the problem conditions
def number_of_basil_plants : Nat := 5
def number_of_tomato_plants : Nat := 4

-- Define the condition that two specific basil plants and all tomato plants form two clusters
def specific_basil_plants_clustering := 2
def tomato_plants_clustering := 4

-- Prove the total number of ways to arrange the plants
theorem arrange_plants : 
  let entities := 3 in -- Group of tomato plants, Group of specific basil plants, Remaining basil plants
  let arrange_entities := Nat.factorial entities in
  let arrange_tomatoes := Nat.factorial number_of_tomato_plants in
  let arrange_specific_basils := Nat.factorial specific_basil_plants_clustering in
  let arrange_remaining_basils := Nat.factorial (number_of_basil_plants - specific_basil_plants_clustering) in
  (arrange_entities * arrange_tomatoes * arrange_specific_basils * arrange_remaining_basils) = 1728 := 
by 
  sorry

end arrange_plants_l257_257924


namespace sum_of_possible_cows_l257_257881

theorem sum_of_possible_cows : 
  ∑ n in {n | ∃ (a b : ℕ), n = 100 * a + 10 * b + a ∧ 
                          (n ≥ 100 ∧ n < 1000) ∧ 
                          (b % 4 = 0) ∧ 
                          (n % 11 = 0)}, 
      n = 726 := 
sorry

end sum_of_possible_cows_l257_257881


namespace area_ratio_cos_squared_l257_257704

theorem area_ratio_cos_squared (A B C D F : Type) [geometry A B C D] (β : ℝ) :
  (is_diameter A B) → -- AB is a diameter
  (is_parallel C D B A) → -- CD is parallel to AB
  (intersects A C B D F) → -- AC intersects BD at F
  (angle A F D = β) → -- angle AFD = β
  (area_ratio (triangle C D F) (triangle A B F) = cos β ^ 2) :=
by
  sorry

end area_ratio_cos_squared_l257_257704


namespace evaluate_expression_l257_257184

theorem evaluate_expression : 
  2 * Real.sin (30 * Real.pi / 180) - (Real.abs (1 - Real.sqrt 2)) + (Real.pi - 2022) ^ 0 = 3 - Real.sqrt 2 := by
  sorry

end evaluate_expression_l257_257184


namespace noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l257_257281

-- Problem 1: Four-digit numbers with no repeated digits
theorem noRepeatedDigitsFourDigit :
  ∃ (n : ℕ), (n = 120) := sorry

-- Problem 2: Five-digit numbers with no repeated digits and divisible by 5
theorem noRepeatedDigitsFiveDigitDiv5 :
  ∃ (n : ℕ), (n = 216) := sorry

-- Problem 3: Four-digit numbers with no repeated digits and greater than 1325
theorem noRepeatedDigitsFourDigitGreaterThan1325 :
  ∃ (n : ℕ), (n = 181) := sorry

end noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l257_257281


namespace min_buses_needed_l257_257160

theorem min_buses_needed (total_students : ℕ) (bus45_capacity : ℕ) (bus40_capacity : ℕ) : 
  total_students = 530 ∧ bus45_capacity = 45 ∧ bus40_capacity = 40 → 
  ∃ (n : ℕ), n = 12 :=
by 
  intro h
  obtain ⟨htotal, hbus45, hbus40⟩ := h
  -- Proof would go here...
  sorry

end min_buses_needed_l257_257160


namespace jacos_budget_l257_257356

theorem jacos_budget :
  (friends : Nat) (friend_gift_cost total_budget : Nat)
  (jaco_remainder_budget : Nat)
  : friends = 8 →
  friend_gift_cost = 9 →
  total_budget = 100 →
  jaco_remainder_budget = total_budget - (friends * friend_gift_cost) →
  (jaco_remainder_budget / 2) = 14 := by
  intros friends friend_gift_cost total_budget jaco_remainder_budget friends_eq friend_gift_cost_eq total_budget_eq jaco_remainder_budget_eq
  rw [friends_eq, friend_gift_cost_eq, total_budget_eq, jaco_remainder_budget_eq]
  simp
  sorry

end jacos_budget_l257_257356


namespace unique_prime_digit_l257_257501

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem unique_prime_digit :
  ∃! B : ℕ, B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ is_prime (303200 + B) :=
begin
  have h1 : 303209 = 303200 + 9 := rfl,
  -- Using computational tools or verified results:
  -- Proof omitted for brevity; assumed verified by computational method
  sorry
end

end unique_prime_digit_l257_257501


namespace max_area_of_fenced_rectangle_l257_257611

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l257_257611


namespace pages_left_to_be_read_l257_257807

def total_pages : ℕ := 381
def pages_read : ℕ := 149
def pages_per_day : ℕ := 20
def days_in_week : ℕ := 7

theorem pages_left_to_be_read :
  total_pages - pages_read - (pages_per_day * days_in_week) = 92 := by
  sorry

end pages_left_to_be_read_l257_257807


namespace complex_inequality_l257_257989

open Complex

-- Declare noncomputable theory
noncomputable theory 

-- Define the main theorem
theorem complex_inequality (r : ℝ) (n : ℕ) (z : ℕ → ℂ)
  (hr : 0 < r ∧ r < 1)
  (hz : ∀ k : ℕ, k < n → abs (z k - 1) ≤ r) :
  abs (finset.univ.sum (λ k, z k)) * abs (finset.univ.sum (λ k, 1 / (z k))) ≥ n^2 * (1 - r^2) :=
sorry

end complex_inequality_l257_257989


namespace sum_neg_128_l257_257933

noncomputable def compute_sum : ℤ := 
  ∑ n in finset.range (17), -- summing from -8 to 8 (range 0 to 16 in Lean's terms)
    if (n - 8) % 2 = 0 then
      if n - 8 < 0 then (2 : ℤ) ^ (n - 8) else 2 ^ (n - 8)
    else
      if n - 8 < 0 then -(2 : ℤ) ^ (-(n - 8)) else -(2 ^ (n - 8))

theorem sum_neg_128 : compute_sum = -128 :=
  sorry

end sum_neg_128_l257_257933


namespace smaller_base_length_trapezoid_l257_257793

variable (p q a b : ℝ)
variable (h : p < q)
variable (angle_ratio : ∃ α, ((2 * α) : ℝ) = α + (α : ℝ))

theorem smaller_base_length_trapezoid :
  b = (p^2 + a * p - q^2) / p :=
sorry

end smaller_base_length_trapezoid_l257_257793


namespace distance_between_stripes_l257_257165

theorem distance_between_stripes (d₁ d₂ L W : ℝ) (h : ℝ)
  (h₁ : d₁ = 60)  -- distance between parallel curbs
  (h₂ : L = 30)  -- length of the curb between stripes
  (h₃ : d₂ = 80)  -- length of each stripe
  (area_eq : W * L = 1800) -- area of the parallelogram with base L
: h = 22.5 :=
by
  -- This is to assume the equation derived from area calculation
  have area_eq' : d₂ * h = 1800 := by sorry
  -- Solving for h using the derived area equation
  have h_calc : h = 1800 / 80 := by sorry
  -- Simplifying the result
  have h_simplified : h = 22.5 := by sorry
  exact h_simplified

end distance_between_stripes_l257_257165


namespace number_of_distinct_paths_l257_257706

theorem number_of_distinct_paths (segments steps_per_segment : ℕ) (h_segments : segments = 15) (h_steps : steps_per_segment = 6) :
    ∑ i in Finset.range (segments + 1), Nat.fib (steps_per_segment + 1) = 195 :=
by
  have h : segments * Nat.fib (steps_per_segment + 1) = 15 * Nat.fib 7 := by
    rw [h_segments, h_steps]

  have fib_value : Nat.fib 7 = 13 := by
    rw [Nat.fib_succ_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_one, Nat.fib_zero]
    norm_num

  rw [fib_value] at h
  exact h
  sorry

end number_of_distinct_paths_l257_257706


namespace maximize_grazing_area_tying_to_B_l257_257513

-- Define the geometric conditions of the problem
def side_length := 12
def rope_length := 4
def distance_between_stakes := 3
def circle_area (r : ℝ) := π * r^2

-- Define the subsets of the grazing areas
def grazing_area_center := circle_area rope_length / 2
def grazing_area_corner := 3 * circle_area rope_length / 4

-- Define the statement that needs to be proved
theorem maximize_grazing_area_tying_to_B :
  grazing_area_corner > grazing_area_center :=
sorry

end maximize_grazing_area_tying_to_B_l257_257513


namespace milkshake_cost_is_five_l257_257923

def initial_amount : ℝ := 132
def hamburger_cost : ℝ := 4
def num_hamburgers : ℕ := 8
def num_milkshakes : ℕ := 6
def amount_left : ℝ := 70

theorem milkshake_cost_is_five (M : ℝ) (h : initial_amount - (num_hamburgers * hamburger_cost + num_milkshakes * M) = amount_left) : 
  M = 5 :=
by
  sorry

end milkshake_cost_is_five_l257_257923


namespace y_value_proof_l257_257731

noncomputable def bowtie (a b : ℝ) : ℝ := a + (Real.sqrt (b + (Real.sqrt (b + (Real.sqrt (b + ...))))))
    
theorem y_value_proof : ∃ y : ℝ, bowtie 7 y = 14 ∧ y = 42 := by
  sorry

end y_value_proof_l257_257731


namespace quadrilateral_is_rectangle_l257_257208

theorem quadrilateral_is_rectangle (ABCD : Type) 
  (DAB CDA BCD ABC : Type) (hDAB : DAB ≈ CDA)
  (hCDA : CDA ≈ BCD) (hBCD : BCD ≈ ABC) (hABC : ABC ≈ DAB) :
  is_rectangle ABCD := 
sorry

end quadrilateral_is_rectangle_l257_257208


namespace functional_equation_solution_l257_257642

theorem functional_equation_solution (α : ℝ) (hα : α = -1) : 
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y) → (∀ x : ℝ, f x = x) :=
by
  assume α hα
  assume f h
  sorry

end functional_equation_solution_l257_257642


namespace part1_part2_part3a_part3b_l257_257255

open Real

variable (θ : ℝ) (m : ℝ)

-- Conditions
axiom theta_domain : 0 < θ ∧ θ < 2 * π
axiom quadratic_eq : ∀ x : ℝ, 2 * x^2 - (sqrt 3 + 1) * x + m = 0
axiom roots_eq_theta : ∀ x : ℝ, (x = sin θ ∨ x = cos θ)

-- Proof statements
theorem part1 : 1 - cos θ ≠ 0 → 1 - tan θ ≠ 0 → 
  (sin θ / (1 - cos θ) + cos θ / (1 - tan θ)) = (3 + 5 * sqrt 3) / 4 := sorry

theorem part2 : sin θ * cos θ = m / 2 → m = sqrt 3 / 4 := sorry

theorem part3a : sin θ = sqrt 3 / 2 ∧ cos θ = 1 / 2 → θ = π / 3 := sorry

theorem part3b : sin θ = 1 / 2 ∧ cos θ = sqrt 3 / 2 → θ = π / 6 := sorry

end part1_part2_part3a_part3b_l257_257255


namespace toothbrushes_given_in_week_l257_257488

theorem toothbrushes_given_in_week :
  ∀ (toothbrushes_per_patient : ℕ) (visit_duration : ℝ) (hours_per_day : ℝ) (days_per_week : ℕ),
  toothbrushes_per_patient = 2 →
  visit_duration = 0.5 →
  hours_per_day = 8 →
  days_per_week = 5 →
  (toothbrushes_per_patient * (hours_per_day / visit_duration).to_nat * days_per_week) = 160 :=
by
  intros toothbrushes_per_patient visit_duration hours_per_day days_per_week
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end toothbrushes_given_in_week_l257_257488


namespace count_integer_values_of_x_l257_257301

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l257_257301


namespace maximize_T_l257_257758

def T (n : ℕ) : set (ℕ × ℕ × ℕ) :=
  {triplet | ∃ a b c, triplet = (a, b, c) ∧ 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n }

def satisfies_condition (s : set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (t₁ t₂ : ℕ × ℕ × ℕ), t₁ ∈ s ∧ t₂ ∈ s ∧ t₁ ≠ t₂ → 
    (((t₁.1 = t₂.1 → ¬(t₁.2 = t₂.2) ∨ (t₁.3 ≠ t₂.3)) ∧ 
    ((t₁.2 = t₂.2 → ¬(t₁.1 = t₂.1) ∨ (t₁.3 ≠ t₂.3)) ∧ 
    (t₁.3 = t₂.3 → ¬(t₁.1 = t₂.1) ∨ (t₁.2 ≠ t₂.2)))))

noncomputable def max_size (n : ℕ) : ℕ :=
  nat.choose n 3

theorem maximize_T (n : ℕ) : 
  ∃ (s : set (ℕ × ℕ × ℕ)), s ⊆ T n ∧ satisfies_condition s ∧ s.to_finset.card = max_size n :=
begin
  sorry
end

end maximize_T_l257_257758


namespace christine_personal_needs_allocation_l257_257187

variables (commission_rate : ℝ) (items_sold : ℝ) (amount_saved : ℝ)

def earnings_from_commission (commission_rate items_sold : ℝ) : ℝ :=
  commission_rate * items_sold

def amount_allocated_to_personal_needs (earnings amount_saved : ℝ) : ℝ :=
  earnings - amount_saved

def percentage_allocated_to_personal_needs (earnings personal_needs : ℝ) : ℝ :=
  (personal_needs / earnings) * 100

theorem christine_personal_needs_allocation :
  commission_rate = 0.12 →
  items_sold = 24000 →
  amount_saved = 1152 →
  percentage_allocated_to_personal_needs 
    (earnings_from_commission commission_rate items_sold)
    (amount_allocated_to_personal_needs 
       (earnings_from_commission commission_rate items_sold) 
       amount_saved) = 60 :=
by
  intros h_commission h_items h_saved
  rw [h_commission, h_items, h_saved]
  unfold earnings_from_commission
  unfold amount_allocated_to_personal_needs
  unfold percentage_allocated_to_personal_needs
  norm_num
  sorry

end christine_personal_needs_allocation_l257_257187


namespace expression_without_arithmetic_square_root_l257_257111

theorem expression_without_arithmetic_square_root : 
  ¬ (∃ x, x^2 = (-|-9|)) :=
by { intro h, cases h with y hy, 
     have hy_nonneg : y^2 ≥ 0 
       := sq_nonneg y,
     let expr := y^2,
     show false,
     calc 
       expr = -|-9| : hy
       ... = -9    : by norm_num
       ... < 0     : by linarith,
}

end expression_without_arithmetic_square_root_l257_257111


namespace dog_bones_remaining_l257_257142

noncomputable def initial_bones : ℕ := 350
noncomputable def factor : ℕ := 9
noncomputable def found_bones : ℕ := factor * initial_bones
noncomputable def total_bones : ℕ := initial_bones + found_bones
noncomputable def bones_given_away : ℕ := 120
noncomputable def bones_remaining : ℕ := total_bones - bones_given_away

theorem dog_bones_remaining : bones_remaining = 3380 :=
by
  sorry

end dog_bones_remaining_l257_257142


namespace projection_correct_l257_257497

theorem projection_correct :
  let u1 := ⟨1, 4⟩ : ℝ × ℝ
  let u2 := ⟨3, -1⟩ : ℝ × ℝ
  let proj_u2 := ⟨24/5, -8/5⟩ : ℝ × ℝ
  let v := ⟨6, -2⟩ : ℝ × ℝ
  let proj_formula (u v : ℝ × ℝ) : ℝ × ℝ := ((u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)) • v
  proj_formula u1 v = ⟨-3/10, 1/10⟩
:= by 
  sorry

end projection_correct_l257_257497


namespace compare_y_values_l257_257996

-- Define the quadratic function y = x^2 + 2x + c
def quadratic (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * x + c

-- Points A, B, and C on the quadratic function
variables 
  (c : ℝ) 
  (y1 y2 y3 : ℝ) 
  (hA : y1 = quadratic (-3) c) 
  (hB : y2 = quadratic (-2) c) 
  (hC : y3 = quadratic 2 c)

theorem compare_y_values :
  y3 > y1 ∧ y1 > y2 :=
by sorry

end compare_y_values_l257_257996


namespace books_from_second_shop_l257_257401

open Real

theorem books_from_second_shop 
  (total_spent_1 : ℝ) (total_spent_2 : ℝ) (books_1 : ℝ) (average_price : ℝ) :
  total_spent_1 = 1150 ∧ total_spent_2 = 920 ∧ books_1 = 65 ∧ average_price = 18 →
  let total_spent := total_spent_1 + total_spent_2 in
  let total_books := total_spent / average_price in
  let books_2 := total_books - books_1 in
  books_2 = 50 :=
by
  intro h
  let total_spent := total_spent_1 + total_spent_2 
  let total_books := total_spent / average_price
  let books_2 := total_books - books_1
  sorry

end books_from_second_shop_l257_257401


namespace int_solutions_eq_count_int_values_b_l257_257559

theorem int_solutions_eq (b : ℤ) : 
  ∃! x : ℤ, ∃! y : ℤ, (x + y = -b) ∧ (x * y = 12 * b) \/
  (x + y = -b) ∧ (x * y = 12 * b) :=
begin
  -- Assume roots p, q exist
  -- Use Vieta's formulas: p + q = -b, pq = 12b
  -- Transform the equation using SFFT
  sorry
end

theorem count_int_values_b :
  set_finite {b : ℤ | ∃! x : ℤ, ∃! y : ℤ, 
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} ∧
  fintype.card {b : ℤ | ∃! x : ℤ, ∃! y : ℤ,
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} = 16 :=
begin
  sorry
end

end int_solutions_eq_count_int_values_b_l257_257559


namespace three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l257_257396

theorem three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one (n : ℕ) (h : n > 1) : ¬(2^n - 1) ∣ (3^n - 1) :=
sorry

end three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l257_257396


namespace train_speed_l257_257909

/-- Given that a train crosses a pole in 12 seconds and the length of the train is 200 meters,
prove that the speed of the train in km/hr is 60. -/
theorem train_speed 
  (cross_time : ℝ) (train_length : ℝ)
  (H1 : cross_time = 12) (H2 : train_length = 200) : 
  let distance_km := train_length / 1000
      time_hr := cross_time / 3600
      speed := distance_km / time_hr in
  speed = 60 :=
by
  sorry

end train_speed_l257_257909


namespace two_results_l257_257021

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l257_257021


namespace sum_of_adjacent_to_21_l257_257047

-- Define 294 and its relevant divisors in Lean.
def num : ℕ := 294
def divisors : set ℕ := {7, 14, 21, 42, 49, 98, 147, 294}

-- Define the property of adjacency i.e., having a common factor greater than 1.
def common_factor_gt_one (a b : ℕ) : Prop :=
  ∃ x, x > 1 ∧ x ∣ a ∧ x ∣ b

-- Define the problem as a Lean statement.
theorem sum_of_adjacent_to_21 :
  (∃ a b, a ∈ divisors ∧ b ∈ divisors ∧ common_factor_gt_one 21 a ∧ common_factor_gt_one 21 b ∧ a + b = 49) :=
sorry

end sum_of_adjacent_to_21_l257_257047


namespace angle_ADB_eq_angle_CAM_l257_257640

theorem angle_ADB_eq_angle_CAM 
    {A B C D M : Point} 
    (quadrilateral : ConvexQuadrilateral A B C D)
    (angle_DAB_right : ∠ D A B = 90°)
    (midpoint_M : midpoint M B C)
    (angle_ADC_eq_angle_BAM : ∠ A D C = ∠ B A M) :
    ∠ A D B = ∠ C A M :=
begin
  sorry
end

end angle_ADB_eq_angle_CAM_l257_257640


namespace max_area_of_fenced_rectangle_l257_257615

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l257_257615


namespace Shelley_birthday_l257_257777

theorem Shelley_birthday (birth_year : ℕ) (anniversary_year : ℕ) (anniversary_day : DayOfWeek) (leap_year_condition : ℕ → Prop) (days_in_week : ℕ) :
  birth_year = 1792 →
  anniversary_year = 2042 →
  anniversary_day = DayOfWeek.monday →
  (leap_year_condition = λ y, (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)) →
  days_in_week = 7 →
  DayOfWeek.corresponding_day birth_year anniversary_year anniversary_day days_in_week leap_year_condition = DayOfWeek.thursday :=
by
  sorry

end Shelley_birthday_l257_257777


namespace polar_center_coordinates_l257_257337

theorem polar_center_coordinates (ρ θ : ℝ) :
  (∀ (ρ θ : ℝ), ρ = 2 * sqrt 2 * cos (θ + π / 4)) →
  (∃ r t : ℝ, r = sqrt 2 ∧ t = -π / 4) :=
by
  intro h
  use sqrt 2, -π / 4
  split
  · sorry
  · sorry

end polar_center_coordinates_l257_257337


namespace alpha_combination_nonzero_l257_257384

open_locale affine

variables {n : ℕ} (O : point ℝ)
variables (A : fin n → point ℝ) (α : fin n → ℝ)

noncomputable theory

def regular_ngon : Prop :=
  ∃ θ : ℝ, ∀ i : fin n, ∃ j : fin n, j = (i : ℕ + 1) % n ∧ dist (A i) (A j) = 1 ∧ angle (O -ᵥ A i) (O -ᵥ A j) = 2 * π / n

def alpha_sorted : Prop :=
  ∀ i j : fin n, i < j → α i > α j

def alpha_positive : Prop :=
  ∀ i : fin n, α i > 0

theorem alpha_combination_nonzero (h_ngon : regular_ngon A) (h_sorted : alpha_sorted α) (h_positive : alpha_positive α) :
  (finset.univ.sum (λ i, (α i) • (O -ᵥ (A i)))) ≠ 0 :=
sorry

end alpha_combination_nonzero_l257_257384


namespace line_slope_angle_l257_257053

theorem line_slope_angle (x y : ℝ) (h : x + √3 * y - 2 = 0) :
  ∃ α : ℝ, α = 5 * Real.pi / 6 := 
sorry

end line_slope_angle_l257_257053


namespace find_d_squared_l257_257739

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * complex.I) * z

theorem find_d_squared (c d : ℝ) (z : ℂ) (h : ∀ z : ℂ, complex.abs (g z c d - z) = complex.abs (g z c d))
  (hcabs : complex.abs (c + d * complex.I) = 5) : d^2 = 99 / 4 :=
by
  sorry

end find_d_squared_l257_257739


namespace ratio_of_areas_l257_257428

theorem ratio_of_areas (a b : ℝ) (ha : a = 2 * b) :
  ( √3 / 4 * a^2 ) / ( 3 * √3 / 2 * b^2 ) = 2 / 3 :=
by
  sorry

end ratio_of_areas_l257_257428


namespace max_area_of_fenced_rectangle_l257_257614

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l257_257614


namespace geometric_seq_common_ratio_l257_257985

theorem geometric_seq_common_ratio 
  (a : ℝ) (q : ℝ)
  (h1 : a * q^2 = 4)
  (h2 : a * q^5 = 1 / 2) : 
  q = 1 / 2 := 
by
  sorry

end geometric_seq_common_ratio_l257_257985


namespace remainder_of_division_l257_257060

theorem remainder_of_division (a b : ℕ) (digits : set ℕ) (h_digits : digits = {5, 9, 2, 7, 3}) (h_a : a = 975) (h_b : b = 23) : 
  a % b = 9 :=
by
  sorry

end remainder_of_division_l257_257060


namespace total_payment_per_month_l257_257476

-- Define the dimensions of the box
def box_length := 15
def box_width := 12
def box_height := 10

-- Calculate the volume of one box
def volume_of_one_box := box_length * box_width * box_height

-- Define the total volume occupied by all boxes
def total_volume := 1080000

-- Define the price per box per month
def price_per_box_per_month := 0.4

-- Calculate the number of boxes
def number_of_boxes := total_volume / volume_of_one_box

-- Calculate the total amount paid per month
def total_amount_paid_per_month := number_of_boxes * price_per_box_per_month

theorem total_payment_per_month : total_amount_paid_per_month = 240 :=
sorry

end total_payment_per_month_l257_257476


namespace impossible_equal_distribution_l257_257445

theorem impossible_equal_distribution :
  ∀ (initial full_capacity: ℝ), 
  (∃ moves : list (ℝ × ℝ),
    (∀ glass1 glass2 ∈ set_of moves,
      glass1 + glass2 = full_capacity ∧ 
      glass1 = glass2 / 2 ∧ glass2 = glass1 / 2)
     ∧ ((full_capacity/3, full_capacity/3, full_capacity/3) ∉ set_of moves)) := sorry

end impossible_equal_distribution_l257_257445


namespace ratio_result_l257_257222

theorem ratio_result (p q r s : ℚ) 
(h1 : p / q = 2) 
(h2 : q / r = 4 / 5) 
(h3 : r / s = 3) : 
  s / p = 5 / 24 :=
sorry

end ratio_result_l257_257222


namespace area_of_KLMN_constant_l257_257746

theorem area_of_KLMN_constant (A B C D E F K L M N : Point)
    (Hconvex : ConvexQuadrilateral A B C D)
    (HEF_sides : E ∈ Segment A B ∧ F ∈ Segment C D)
    (Hmidpoints : K = midpoint D E ∧ L = midpoint B F ∧ M = midpoint C E ∧ N = midpoint A F) :
    ConvexQuadrilateral K L M N ∧
    ∃ α, Area K L M N = (1 / 8) * Distance A B * Distance C D * Real.sin α :=
sorry

end area_of_KLMN_constant_l257_257746


namespace driver_schedule_l257_257043

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l257_257043


namespace derivative_of_x_squared_ln_x_derivative_of_sin_x_minus_x_div_x_squared_derivative_of_e_negx_sqrt_2x_minus_1_l257_257211

open Real

-- Problem 1
theorem derivative_of_x_squared_ln_x (x : ℝ) :
  has_deriv_at (λ x, x^2 * log x) (2 * x * log x + x) x :=
sorry

-- Problem 2
theorem derivative_of_sin_x_minus_x_div_x_squared (x : ℝ) :
  has_deriv_at (λ x, (sin x - x) / x^2) ((x * cos x - 2 * sin x + x) / x^3) x :=
sorry

-- Problem 3
theorem derivative_of_e_negx_sqrt_2x_minus_1 (x : ℝ) :
  has_deriv_at (λ x, exp (-x) * sqrt (2 * x - 1)) (exp (-x) * ((2 - 2 * x) * sqrt (2 * x - 1) / (2 * x - 1))) x :=
sorry

end derivative_of_x_squared_ln_x_derivative_of_sin_x_minus_x_div_x_squared_derivative_of_e_negx_sqrt_2x_minus_1_l257_257211


namespace max_words_in_antarctican_language_l257_257004

theorem max_words_in_antarctican_language : ∃ (a b : ℕ), a + b = 16 ∧ 8 * 16 * 8 = 1024 := 
by {
  let a := 8,
  let b := 8,
  have h1 : a + b = 16 := by linarith,
  have h2 : a * 16 * b = 1024 := by norm_num,
  exact ⟨a, b, h1, h2⟩,
  sorry
}

end max_words_in_antarctican_language_l257_257004


namespace cylinder_volume_l257_257202

-- Define the radius based on the diameter condition
def radius (d : ℝ) := d / 2

-- Define the height of the cylinder
def height : ℝ := 10

-- Define the volume formula for a cylinder
def volume (r h : ℝ) := Real.pi * r^2 * h

-- State the theorem based on the given conditions and proof the equivalence
theorem cylinder_volume :
  volume (radius 20) height = 1000 * Real.pi :=
by
  sorry

end cylinder_volume_l257_257202


namespace integral_abs_function_l257_257313

def f (x : ℝ) : ℝ := |x + 2|

theorem integral_abs_function : ∫ x in -4..3, f x = 29 / 2 :=
by sorry

end integral_abs_function_l257_257313


namespace sum_first_10_terms_eq_10_l257_257250

-- Definitions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q > 0, ∀ n, a (n + 1) = a n * q
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, a i

-- Conditions
variables (a : ℕ → ℝ)
variable (h_geom : is_geometric_sequence a)
variable (h_a1 : a 1 = 1)
variable (h_a5 : a 5 = 1)

-- Proof statement
theorem sum_first_10_terms_eq_10 : sum_first_n_terms a 10 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end sum_first_10_terms_eq_10_l257_257250


namespace minimum_cuts_for_polygons_l257_257444

theorem minimum_cuts_for_polygons (initial_pieces desired_pieces : ℕ) (sides : ℕ)
    (h_initial_pieces : initial_pieces = 1) (h_desired_pieces : desired_pieces = 100)
    (h_sides : sides = 20) :
    ∃ (cuts : ℕ), cuts = 1699 ∧
    (∀ current_pieces, current_pieces < desired_pieces → current_pieces + cuts ≥ desired_pieces) :=
by
    sorry

end minimum_cuts_for_polygons_l257_257444


namespace fifth_arithmetic_term_l257_257789

theorem fifth_arithmetic_term (x y : ℝ) :
  let a₁ := x + 2 * y,
      a₂ := x - y,
      a₃ := 2 * x * y,
      a₄ := x / (2 * y)
  in (a₂ - a₁ = -3 * y ∧ a₃ - a₂ = -3 * y ∧ a₄ - a₃ = -3 * y) →
     (a₄ - 3 * y = -1) :=
by
  simp [a₁, a₂, a₃, a₄]
  sorry

end fifth_arithmetic_term_l257_257789


namespace length_second_train_is_125_l257_257463

noncomputable def length_second_train (speed_faster speed_slower distance1 : ℕ) (time_minutes : ℝ) : ℝ :=
  let relative_speed_m_per_minute := (speed_faster - speed_slower) * 1000 / 60
  let total_distance_covered := relative_speed_m_per_minute * time_minutes
  total_distance_covered - distance1

theorem length_second_train_is_125 :
  length_second_train 50 40 125 1.5 = 125 :=
  by sorry

end length_second_train_is_125_l257_257463


namespace probability_multiple_of_50_l257_257686

theorem probability_multiple_of_50 : 
  let S := {2, 4, 10, 12, 15, 20, 25, 50}
  let count_pairs := (S.to_finset.powerset.filter (λ s, s.card = 2)).card
  let successful_pairs := 
    (S.to_finset.powerset.filter (λ s, s.card = 2 ∧ (50 ∣ (s.erase (s.min').get ∗ s.min').get))).card
  count_pairs = 28 ∧ successful_pairs = 13 → 
  (successful_pairs / count_pairs = (13 / 28 : ℚ)) :=
by
  sorry

end probability_multiple_of_50_l257_257686


namespace share_price_increase_l257_257516

theorem share_price_increase (P : ℝ) : 
  let P1 := 1.25 * P in
  let P2 := P1 * 1.24 in
  (P2 - P) / P * 100 = 55 :=
by
  let P1 := 1.25 * P
  let P2 := P1 * 1.24
  show (P2 - P) / P * 100 = 55
  sorry

end share_price_increase_l257_257516


namespace sin_300_eq_neg_sqrt3_div_2_l257_257199

theorem sin_300_eq_neg_sqrt3_div_2 : sin (300 * real.pi / 180) = -real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l257_257199


namespace tg_cos_identity_l257_257854

theorem tg_cos_identity (α : ℝ) :
  (Real.tan (4 * α) - Real.acos (4 * α)) = 
  (Real.sin (2 * α) - Real.cos (2 * α)) / 
  (Real.sin (2 * α) + Real.cos (2 * α)) :=
sorry

end tg_cos_identity_l257_257854


namespace firework_max_height_time_l257_257051

noncomputable def height (t : ℝ) : ℝ :=
  - (3 / 4) * t^2 + 12 * t - 21

theorem firework_max_height_time : ∃ t : ℝ, height t = 27 ∧ t = 8 :=
by
  sorry

end firework_max_height_time_l257_257051


namespace solve_inequality_l257_257769

open Real

noncomputable def inequality_left (x : ℝ) : ℝ := sqrt (2 * x + 7 / x^2) + sqrt (2 * x - 7 / x^2)

noncomputable def inequality_right (x : ℝ) : ℝ := 6 / x

theorem solve_inequality (x : ℝ) (hx : sqrt (2 * x + 7 / x^2) + sqrt (2 * x - 7 / x^2) < 6 / x)
  (h_cond1 : 2 * x + 7 / x^2 ≥ 0) (h_cond2 : 2 * x - 7 / x^2 ≥ 0) :
  x ∈ set.Icc (real.cbrt (7 / 2)) (real.cbrt (373 / 72)) :=
begin
  sorry
end

end solve_inequality_l257_257769


namespace period_of_function_satisfying_equation_l257_257194

theorem period_of_function_satisfying_equation :
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f(x + 6) + f(x - 6) = f(x)) →
  ∃ p : ℕ, p = 36 ∧ ∀ f : ℝ → ℝ, (∀ x : ℝ, f(x + 6) + f(x - 6) = f(x)) → (∀ x : ℝ, f(x) = f(x + p)) :=
begin
  sorry -- Proof goes here
end

end period_of_function_satisfying_equation_l257_257194


namespace subtraction_from_double_result_l257_257469

theorem subtraction_from_double_result (x : ℕ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end subtraction_from_double_result_l257_257469


namespace fraction_of_single_men_l257_257514

theorem fraction_of_single_men (E : ℝ) (h_pos : E > 0) : 
  let women := 0.76 * E,
      married_employees := 0.60 * E,
      married_women := 0.6842 * women,
      married_men := married_employees - married_women,
      men := 0.24 * E,
      single_men := men - married_men,
      fraction_single_men := single_men / men
  in fraction_single_men = 0.6683 := 
by
  sorry

end fraction_of_single_men_l257_257514


namespace integer_values_of_b_for_quadratic_eqn_l257_257562

noncomputable def number_of_integer_values_of_b : ℕ := 16

theorem integer_values_of_b_for_quadratic_eqn :
  ∃(b : ℤ) (k ≥ 0), ∀m n : ℤ, (m + n = -b ∧ m * n = 12 * b) → (m + 12) * (n + 12) = 144 → k = number_of_integer_values_of_b := sorry

end integer_values_of_b_for_quadratic_eqn_l257_257562


namespace taxi_fare_l257_257172

theorem taxi_fare (fare : ℕ → ℝ) (distance : ℕ) :
  (∀ d, d > 10 → fare d = 20 + (d - 10) * (140 / 70)) →
  fare 80 = 160 →
  fare 100 = 200 :=
by
  intros h_fare h_fare_80
  show fare 100 = 200
  sorry

end taxi_fare_l257_257172


namespace A_time_for_task_l257_257878

theorem A_time_for_task :
  ∀ (W : ℝ) (B_work_rate : ℝ), 
  (A_work_rate = 0.8 * B_work_rate) → 
  (B_work_rate * 6 = W) → 
  (A_work_rate * A_time = W) → 
  A_time = 7.5 := 
by {
  intros W B_work_rate A_work_rate_eq B_work_rate_eq A_work_rate_task_eq,
  sorry
}

end A_time_for_task_l257_257878


namespace longest_piece_length_l257_257358

-- Define the lengths of the ropes
def rope1 : ℕ := 45
def rope2 : ℕ := 75
def rope3 : ℕ := 90

-- Define the greatest common divisor we need to prove
def gcd_of_ropes : ℕ := Nat.gcd rope1 (Nat.gcd rope2 rope3)

-- Goal theorem stating the problem
theorem longest_piece_length : gcd_of_ropes = 15 := by
  sorry

end longest_piece_length_l257_257358


namespace find_cows_l257_257120

variable (D C : ℕ)

theorem find_cows (h1 : 2 * D + 4 * C = 2 * (D + C) + 36) : C = 18 :=
by
  -- Proof goes here
  sorry

end find_cows_l257_257120


namespace largest_constant_inequality_l257_257537

theorem largest_constant_inequality :
  ∃ C : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧ C = 2 / Real.sqrt 3 :=
begin
  sorry
end

end largest_constant_inequality_l257_257537


namespace min_value_of_expression_l257_257383

open Real

theorem min_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) :
  (3 * x^2 + 12 * x * y + 9 * y^2 + 15 * y * z + 3 * z^2) ≥ (243 / (real_root 4 9)) :=
by
  sorry

end min_value_of_expression_l257_257383


namespace part1_part2_part3_l257_257821

-- Part (1)
theorem part1 (m n : ℤ) (h1 : m - n = -1) : 2 * (m - n)^2 + 18 = 20 := 
sorry

-- Part (2)
theorem part2 (m n : ℤ) (h2 : m^2 + 2 * m * n = 10) (h3 : n^2 + 3 * m * n = 6) : 2 * m^2 + n^2 + 7 * m * n = 26 :=
sorry

-- Part (3)
theorem part3 (a b c m x : ℤ) (h4: ax^5 + bx^3 + cx - 5 = m) (h5: x = -1) : ax^5 + bx^3 + cx - 5 = -m - 10 :=
sorry

end part1_part2_part3_l257_257821


namespace volume_surface_area_polyhedron_l257_257813

-- Sphere with radius sqrt(2)
def sphere_radius : ℝ := real.sqrt 2

-- Volume of the polyhedron formed by intersection points of the lines with the sphere
def volume_of_polyhedron : ℝ := 14 / 3

-- Surface area of the polyhedron formed by intersection points of the lines with the sphere
def surface_area_of_polyhedron : ℝ := 12 + 4 * real.sqrt 3

theorem volume_surface_area_polyhedron :
  (∃ Poly : Type, ∀ S : Poly,
    (S.radius = sphere_radius) →
    (S.volume = volume_of_polyhedron) ∧ (S.surface_area = surface_area_of_polyhedron)) :=
begin
  sorry
end

end volume_surface_area_polyhedron_l257_257813


namespace response_rate_approximation_l257_257478

def response_rate (responses : ℕ) (questionnaires_sent : ℕ) : ℚ := (responses / questionnaires_sent) * 100

theorem response_rate_approximation :
  response_rate 222 370 ≈ 60 := 
by 
  sorry

end response_rate_approximation_l257_257478


namespace quadratic_inequality_solution_l257_257054

theorem quadratic_inequality_solution :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end quadratic_inequality_solution_l257_257054


namespace radical_axes_are_coincident_or_concurrent_or_parallel_l257_257869

noncomputable def radical_axis_of_three_circles 
  (Γ₁ Γ₂ Γ₃ : Type) [circle Γ₁] [circle Γ₂] [circle Γ₃] 
  (Δ₁ : radical_axis Γ₁ Γ₂)
  (Δ₂ : radical_axis Γ₂ Γ₃)
  (Δ₃ : radical_axis Γ₃ Γ₁) : Prop :=
(Δ₁ = Δ₂ ∧ Δ₂ = Δ₃) ∨ (concurrent Δ₁ Δ₂ Δ₃) ∨ (parallel Δ₁ Δ₂ Δ₃)

theorem radical_axes_are_coincident_or_concurrent_or_parallel 
  (Γ₁ Γ₂ Γ₃ : Type) [circle Γ₁] [circle Γ₂] [circle Γ₃] 
  (Δ₁ : radical_axis Γ₁ Γ₂)
  (Δ₂ : radical_axis Γ₂ Γ₃)
  (Δ₃ : radical_axis Γ₃ Γ₁) : 
  radical_axis_of_three_circles Γ₁ Γ₂ Γ₃ Δ₁ Δ₂ Δ₃ :=
sorry

end radical_axes_are_coincident_or_concurrent_or_parallel_l257_257869


namespace min_dot_product_l257_257314

-- Definitions and conditions from part (a)
noncomputable def hyperbola_eq (P : Point) : Prop :=
  P.1 ^ 2 / 4 - P.2 ^ 2 / 5 = 1

noncomputable def on_right_branch (P : Point) : Prop :=
  P.1 ≥ 2

def O : Point := (0, 0)
def F : Point := (-3, 0)

-- Statement of the proof problem
theorem min_dot_product : 
  ∀ P : Point, hyperbola_eq P → on_right_branch P → 
  (P.1 ^ 2 + 3 * P.1 + P.2 ^ 2) = 10 :=
by
  sorry

end min_dot_product_l257_257314


namespace triangle_construction_l257_257531

noncomputable def construct_triangle (c s r : ℝ) : Prop :=
∃ (A B C : ℝ × ℝ),
  let d := dist,
      circ_center := (0, 0),
      circ_radius := r in
  d A circ_center = r ∧
  d B circ_center = r ∧
  d A B = c ∧
  d A C + d C B = s

theorem triangle_construction (c s r : ℝ) :
  c <= 2 * r ∧ s >= c → construct_triangle c s r :=
begin
  sorry
end

end triangle_construction_l257_257531


namespace percentage_by_running_correct_strike_rate_correct_l257_257475

-- Definitions based on given conditions
variables (total_runs balls : ℕ) 
variables (boundaries sixes singles doubles no_balls : ℕ)

def total_runs_scored := 180
def total_balls_faced := 120
def boundaries := 4
def sixes := 10
def singles := 35
def doubles := 15
def no_balls := 2

-- Problem 1: Percentage of total score by running between the wickets
def runs_by_running := singles + 2 * doubles
def perc_runs_by_running := (runs_by_running * 100) / total_runs_scored

theorem percentage_by_running_correct :
  perc_runs_by_running = 36.11 :=
sorry

-- Problem 2: Strike Rate
def strike_rate := (total_runs_scored * 100) / total_balls_faced

theorem strike_rate_correct :
  strike_rate = 150 :=
sorry

end percentage_by_running_correct_strike_rate_correct_l257_257475


namespace train_speed_l257_257459

theorem train_speed :
  ∀ (L T : ℝ), L = 120 → T = 16 → (L / T) = 7.5 :=
by
  intros L T hL hT
  rw [hL, hT]
  norm_num
  exact eq.refl 7.5


end train_speed_l257_257459


namespace cross_country_team_scores_l257_257327

theorem cross_country_team_scores : 
  ∃ (winning_scores : Finset ℕ), 
  (winning_scores.card = 17) ∧ 
  (∀ n ∈ winning_scores, ∃ (team_scores : Finset ℕ), 
    team_scores.card = 3 ∧ 
    (∃ i ∈ team_scores, ∑ j in (insert i team_scores).erase i, id j = n)) ∧ 
  (∀ (teams : Finset ℕ), teams.card = 3 → 
    ∑ score in teams, score = 78) := 
sorry

end cross_country_team_scores_l257_257327


namespace projection_matrix_correct_l257_257725

noncomputable def Q_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1/9, 5/9, 8/9], 
    ![7/9, 8/9, 2/9], 
    ![4/9, 2/9, 5/9]]

theorem projection_matrix_correct : 
  ∀ u : Fin 3 → ℝ,
  let n := ![2, 1, -2] in
  let Q : Matrix (Fin 3) (Fin 3) ℝ := Q_matrix in
  (Q.mulVec u) = u - ((((u ⬝ n) / (n ⬝ n)) • n)) :=
sorry

end projection_matrix_correct_l257_257725


namespace betty_age_l257_257510

-- Define the constants and conditions
variables (A M B : ℕ)
variables (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 8)

-- Define the theorem to prove Betty's age
theorem betty_age : B = 4 :=
by sorry

end betty_age_l257_257510


namespace circle_inscribed_in_square_area_l257_257009

theorem circle_inscribed_in_square_area :
  ∀ (x y : ℝ) (h : 2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0),
  ∃ side : ℝ, 4 * (side^2) = 16 :=
by
  sorry

end circle_inscribed_in_square_area_l257_257009


namespace combined_value_is_correct_l257_257311

noncomputable def a : ℕ :=
  let p := 0.95
  in (p / 0.005).toNat

noncomputable def b : ℕ := 3 * a - 50

noncomputable def c : ℕ := (a - b) * (a - b)

theorem combined_value_is_correct : a + b + c = 109610 :=
by
  have h_a : a = 190 := by
    norm_num [a]
  have h_b : b = 520 := by
    rw [h_a]
    norm_num [b]
  have h_c : c = 108900 := by
    rw [h_a, h_b]
    norm_num [c]
  rw [h_a, h_b, h_c]
  norm_num

end combined_value_is_correct_l257_257311


namespace smallest_even_five_digit_hundreds_digit_l257_257786

theorem smallest_even_five_digit_hundreds_digit :
  ∃ (n : ℕ), 
    (∀ (d : ℕ), 
      d ∈ [0, 1, 3, 7, 8] → 
      (∀ m, m ∉ [0, 1, 3, 7, 8] → false)) ∧ 
    (n % 2 = 0) ∧ 
    (10000 ≤ n) ∧ (n < 100000) ∧ 
    n = 1 * 10000 + 0 * 1000 + 3 * 100 + 7 * 10 + 8 :=
begin
  sorry
end

end smallest_even_five_digit_hundreds_digit_l257_257786


namespace PR_plus_PS_eq_AF_l257_257868

open EuclideanGeometry

variables {A B C D P S R Q F : Point}
variables {AB CD : LineSegment}

-- Conditions
def is_trapezoid (A B C D : Point) (AB CD : LineSegment) : Prop :=
  is_parallel AB CD ∧ length AB > length CD

def perp (P Q : Point) (L : LineSegment) : Prop :=
  line_segment P Q ⊥ L

-- Required conditions for the proof
axiom trapezoid_ABCD : is_trapezoid A B C D AB CD
axiom point_P_on_AB : LiesOn P (line_segment A B)
axiom PS_perp_CD : perp P S CD
axiom PR_perp_AC : perp P R (line_segment A C)
axiom AF_perp_CD : perp A F CD
axiom F_on_CD : LiesOn F (line_segment C D)
axiom PQ_perp_AF : perp P Q (line_segment A F)

-- To be proven
theorem PR_plus_PS_eq_AF : (length (line_segment P R) + length (line_segment P S)) = length (line_segment A F) :=
sorry

end PR_plus_PS_eq_AF_l257_257868


namespace min_time_to_pass_l257_257169

noncomputable def tunnel_length : ℝ := 2150
noncomputable def num_vehicles : ℝ := 55
noncomputable def vehicle_length : ℝ := 10
noncomputable def speed_limit : ℝ := 20
noncomputable def max_speed : ℝ := 40

noncomputable def distance_between_vehicles (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then 20 else
if 10 < x ∧ x ≤ 20 then (1/6) * x ^ 2 + (1/3) * x else
0

noncomputable def time_to_pass_through_tunnel (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then (2150 + 10 * 55 + 20 * (55 - 1)) / x else
if 10 < x ∧ x ≤ 20 then (2150 + 10 * 55 + ((1/6) * x^2 + (1/3) * x) * (55 - 1)) / x + 9 * x + 18 else
0

theorem min_time_to_pass : ∃ x : ℝ, (10 < x ∧ x ≤ 20) ∧ x = 17.3 ∧ time_to_pass_through_tunnel x = 329.4 :=
sorry

end min_time_to_pass_l257_257169


namespace ship_sails_distance_interval_l257_257161

theorem ship_sails_distance_interval (AB BC AC: ℝ) (θ: ℝ) (h1: AB = 15) 
  (h2: BC = 30 * Real.cos θ) (h3: 30 <= θ <= 45) : 200 <= AC^2 ∧ AC^2 <= 300 :=
by
  -- The conditions here ensure the correct setup for the problem.
  have h4: AC^2 = AB^2 + BC^2 - 2 * AB * BC * Real.cos θ, by sorry
  calc
    AC^2 = 15^2 : by sorry
    ...  = 225   : by sorry
  have h5: 225 <= 300, by linarith
  have h6: 225 >= 200, by linarith
  exact ⟨h6, h5⟩

end ship_sails_distance_interval_l257_257161


namespace sum_a_n_lt_3000_l257_257216

def a_n (n : ℕ) : ℕ :=
  if n % 60 = 0 then 15
  else if n % 180 = 0 then 20
  else if n % 90 = 0 then 18
  else 0

theorem sum_a_n_lt_3000 : ∑ n in Finset.range 3000, a_n n = 1649 := by
  sorry

end sum_a_n_lt_3000_l257_257216


namespace drivers_schedule_l257_257026

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l257_257026


namespace max_area_of_rectangle_with_perimeter_60_l257_257624

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l257_257624


namespace product_of_roots_is_four_thirds_l257_257732

theorem product_of_roots_is_four_thirds :
  (∀ p q r s : ℚ, (∃ a b c: ℚ, (3 * a^3 - 9 * a^2 + 5 * a - 4 = 0 ∧
                                   3 * b^3 - 9 * b^2 + 5 * b - 4 = 0 ∧
                                   3 * c^3 - 9 * c^2 + 5 * c - 4 = 0)) → 
  - s / p = (4 : ℚ) / 3) := sorry

end product_of_roots_is_four_thirds_l257_257732


namespace circle_polar_coords_and_min_length_segment_AB_l257_257268

noncomputable def parametric_circle_eq (θ : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos θ, 3 * Real.sin θ)

def polar_line_eq (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem circle_polar_coords_and_min_length_segment_AB :
  (∀ θ : ℝ, let (x, y) := parametric_circle_eq θ in (x - 1)^2 + y^2 = 9) ∧
  ∃ ρ : ℝ, let θ := 0 in parametric_circle_eq θ = (1, 0) ∧
  (∃ θ : ℝ, ρ^2 - 2 * ρ * Real.cos θ - 8 = 0) ∧
  (∃ A B : ℝ × ℝ, (A ∈ (λ θ, parametric_circle_eq θ) ∧ B ∈ (λ ρ, (ρ * Real.cos (Real.pi / 4), ρ * Real.sin (Real.pi / 4))) ∧
  Real.Angle.angle (parametric_circle_eq 0) A B = Real.pi / 3 ∧ 
  ∀ BC : ℝ, BC >= 1 / Real.sqrt 2 ∧ BC = 3 / 2 -> 
  let AB := Real.sqrt ((BC - 3 / 2)^2 + 27 / 4) in AB = 3 * Real.sqrt 3 / 2)) := sorry

end circle_polar_coords_and_min_length_segment_AB_l257_257268


namespace infinitely_many_sum_of_squares_exceptions_l257_257399

-- Define the predicate for a number being expressible as a sum of two squares
def is_sum_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

-- Define the main theorem
theorem infinitely_many_sum_of_squares_exceptions : 
  ∃ f : ℕ → ℕ, (∀ k : ℕ, is_sum_of_squares (f k)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k - 1)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k + 1)) ∧ (∀ k1 k2 : ℕ, k1 ≠ k2 → f k1 ≠ f k2) :=
sorry

end infinitely_many_sum_of_squares_exceptions_l257_257399


namespace locus_of_point_parabola_l257_257317

/-- If the distance from point P to the point F (4, 0) is one unit less than its distance to the line x + 5 = 0, then the equation of the locus of point P is y^2 = 16x. -/
theorem locus_of_point_parabola :
  ∀ P : ℝ × ℝ, dist P (4, 0) + 1 = abs (P.1 + 5) → P.2^2 = 16 * P.1 :=
by
  sorry

end locus_of_point_parabola_l257_257317


namespace expression_largest_l257_257378

def x : ℝ := 10^(-1997)

theorem expression_largest :
  (3 + x < 3 - x) → (3 + x < 3 * x) → (3 + x < x / 3) → (3 + x < 3 / x) :=
by sorry

end expression_largest_l257_257378


namespace max_area_of_rectangular_pen_l257_257597

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l257_257597


namespace no_n_equal_sum_l257_257375

theorem no_n_equal_sum (n : ℕ) (h : n ≠ 0) :
  let r1 := 2 * n * (n + 3),
      r2 := 2 * n * (n + 1)
  in r1 ≠ r2 :=
by
  simp [*, not_and.mp h]
  sorry

end no_n_equal_sum_l257_257375


namespace black_ink_cost_l257_257749

theorem black_ink_cost (B : ℕ) 
  (h1 : 2 * B + 3 * 15 + 2 * 13 = 50 + 43) : B = 11 :=
by
  sorry

end black_ink_cost_l257_257749


namespace n_not_2_7_l257_257666

open Set

variable (M N : Set ℕ)

-- Define the given set M
def M_def : Prop := M = {1, 4, 7}

-- Define the condition M ∪ N = M
def union_condition : Prop := M ∪ N = M

-- The main statement to be proved
theorem n_not_2_7 (M_def : M = {1, 4, 7}) (union_condition : M ∪ N = M) : N ≠ {2, 7} :=
  sorry

end n_not_2_7_l257_257666


namespace maximum_cos_squared_sum_le_l257_257323

noncomputable def maximum_cos_squared_sum (A B C : ℝ) (h1 : sin C = 2 * cos A * cos B) (h2 : A + B + C = π) : ℝ :=
  max ((cos A)^2 + (cos B)^2)

theorem maximum_cos_squared_sum_le (A B C : ℝ) (h1 : sin C = 2 * cos A * cos B) (h2 : A + B + C = π) :
  maximum_cos_squared_sum A B C h1 h2 ≤ (√2 + 1) / 2 :=
sorry

end maximum_cos_squared_sum_le_l257_257323


namespace train_speed_l257_257905

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l257_257905


namespace nested_square_root_value_l257_257679

theorem nested_square_root_value :
  (∃ y : ℝ, y = √(3 + (λ y : ℝ, √(3 + (λ y : ℝ, √(3 + (λ y : ℝ, ...)))))) → y = (1 + √13) / 2) :=
begin
  sorry
end

end nested_square_root_value_l257_257679


namespace product_and_divisibility_l257_257205

theorem product_and_divisibility (n : ℕ) (h : n = 3) :
  (n-1) * n * (n+1) * (n+2) * (n+3) = 720 ∧ ¬ (720 % 11 = 0) :=
by
  sorry

end product_and_divisibility_l257_257205


namespace sum_of_roots_l257_257829

theorem sum_of_roots : 
  (∑ x in Finset.filter (λ x, x^2 - 16 * x + 9 = 0) {1, -1, 9, -9}, x) = 16 := 
by
  sorry

end sum_of_roots_l257_257829


namespace simplify_expression_l257_257761

theorem simplify_expression (p : ℝ) : ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end simplify_expression_l257_257761


namespace customer_paid_amount_l257_257045

theorem customer_paid_amount 
  (cost_price : ℝ) 
  (markup_percent : ℝ) 
  (customer_payment : ℝ)
  (h1 : cost_price = 1250) 
  (h2 : markup_percent = 0.60)
  (h3 : customer_payment = cost_price + (markup_percent * cost_price)) :
  customer_payment = 2000 :=
sorry

end customer_paid_amount_l257_257045


namespace max_value_b_l257_257266

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x
noncomputable def g (x a b : ℝ) : ℝ := 3 * a^2 * Real.log x + b

theorem max_value_b (a : ℝ) (ha : 0 < a) :
  ∃ (b : ℝ), (∀ x : ℝ, (f x a = g x a b → (1 / 2) * x^2 + 2 * a * x = 3 * a^2 * Real.log x + b ∧
    f' x a = g' x a b → x + 2 * a = 3 * a^2 / x)) ∧ 
    b = (5 / 2) * a^2 - 3 * a^2 * Real.log a ∧ 
    ∀ t : ℝ, t > 0 → let h := (5 / 2) * t^2 - 3 * t^2 * Real.log t in 
    h ≤ (3 / 2) * Real.exp (2 / 3) :=
begin
  sorry
end

end max_value_b_l257_257266


namespace initial_ratio_l257_257481

variables {P Q : ℕ}

theorem initial_ratio (h1 : P + Q = 25) (h2 : P * 4 = 5 * (Q + 2)) : P * 2 = 3 * Q :=
begin
  sorry,
end

end initial_ratio_l257_257481


namespace students_in_section_B_l257_257805

variable (x : ℕ)

/-- There are 30 students in section A and the number of students in section B is x. The 
    average weight of section A is 40 kg, and the average weight of section B is 35 kg. 
    The average weight of the whole class is 38 kg. Prove that the number of students in
    section B is 20. -/
theorem students_in_section_B (h : 30 * 40 + x * 35 = 38 * (30 + x)) : x = 20 :=
  sorry

end students_in_section_B_l257_257805


namespace ten_digit_numbers_with_parity_groups_l257_257398

def even_ones (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  (digits.filter (λ d => d = 1)).length % 2 = 0

def odd_ones (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  (digits.filter (λ d => d = 1)).length % 2 = 1

theorem ten_digit_numbers_with_parity_groups :
  ∀ (n m : ℕ), 
    10^9 ≤ n ∧ n < 10^10 ∧ 10^9 ≤ m ∧ m < 10^10 ∧ 
    (∀ i, (Nat.digits 10 n).nth i = 1 ∨ (Nat.digits 10 n).nth i = 2) ∧
    (∀ i, (Nat.digits 10 m).nth i = 1 ∨ (Nat.digits 10 m).nth i = 2) ∧
    (even_ones n ∧ even_ones m) ∨ (odd_ones n ∧ odd_ones m) →
    (Nat.digits 10 (n + m)).filter (λ d => d = 3).length ≥ 2 :=
by
  sorry

end ten_digit_numbers_with_parity_groups_l257_257398


namespace solution_set_of_inequality_l257_257430

noncomputable def solve_inequality : set ℝ :=
  {x : ℝ | x < -6 ∨ x > 1}

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 + 5 * x - 6 > 0) ↔ (x < -6 ∨ x > 1) := by
sorry

end solution_set_of_inequality_l257_257430


namespace modular_inverse_17_392_l257_257447

theorem modular_inverse_17_392 : ∃ x, 0 ≤ x ∧ x ≤ 391 ∧ (17 * x) % 392 = 1 :=
by { use 369, split, { norm_num }, split, { norm_num }, { norm_num } }

end modular_inverse_17_392_l257_257447


namespace four_digit_at_least_one_6_or_8_l257_257280

def four_digit_count_with_6_or_8 : ℕ :=
  let total_4digit := 9999 - 1000 + 1 in
  let without_6_8 := 7 * 8 * 8 * 8 in
  total_4digit - without_6_8

theorem four_digit_at_least_one_6_or_8 : four_digit_count_with_6_or_8 = 5416 := by
  sorry

end four_digit_at_least_one_6_or_8_l257_257280


namespace chinese_addition_result_l257_257334

def chinese_digit (贺 十 新 春 : ℕ) : Prop :=
  (贺 ≠ 十 ∧ 贺 ≠ 新 ∧ 贺 ≠ 春 ∧ 十 ≠ 新 ∧ 十 ≠ 春 ∧ 新 ≠ 春 ∧
   0 < 贺 ∧ 贺 < 10 ∧ 0 < 十 ∧ 十 < 10 ∧ 0 < 新 ∧ 新 < 10 ∧ 0 < 春 ∧ 春 < 10)

theorem chinese_addition_result (贺 十 新 春 : ℕ) (h : chinese_digit 贺 十 新 春) :
  (贺 * 100 + 十 * 10 + 新) + (新 * 100 + 十 * 10 + 春) = 18 :=
  sorry

end chinese_addition_result_l257_257334


namespace unique_pizzas_l257_257154

theorem unique_pizzas (n k: ℕ) (h_n: n = 8) (h_k: k = 5) : (nat.choose n k) = 56 := by
  sorry

end unique_pizzas_l257_257154


namespace proof_problem_l257_257240

def p : Prop := ∃ k : ℕ, 0 = 2 * k
def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem proof_problem : p ∨ q :=
by
  sorry

end proof_problem_l257_257240


namespace space_talent_recognition_l257_257840

theorem space_talent_recognition:
  ∀ (x : ℕ), -- x is the number of correctly answered questions
  25 = total_questions →
  (correct_points = 4) →
  (incorrect_penalty = -1) →
  (score_threshold = 90) →
  (score x = (correct_points * x + incorrect_penalty * (total_questions - x))) →
  score_threshold ≤ score x →
  23 ≤ x :=
begin
  intros x total_questions correct_points incorrect_penalty score_threshold score,
  sorry
end

end space_talent_recognition_l257_257840


namespace jimmy_cards_l257_257714

theorem jimmy_cards (B : ℕ) (h1 : 18 - (B + 2 * B) = 9) : B = 3 :=
by {
  sorry,
}

end jimmy_cards_l257_257714


namespace proof_problem_l257_257646

noncomputable def problem (S : Set ℕ) :=
  ∃ (F G : Finset ℕ), F ≠ G ∧ (F ⊆ S ∧ G ⊆ S) ∧ (F.sum (λ x, 1 / (x:ℚ)) = G.sum (λ x, 1 / (x:ℚ))) ∨ 
  ∃ (r : ℚ), 0 < r ∧ r < 1 ∧ ∀ (F : Finset ℕ), F ⊆ S → F.sum (λ x, 1 / (x:ℚ)) ≠ r

theorem proof_problem (S : Set ℕ) : problem S :=
sorry

end proof_problem_l257_257646


namespace PQH_collinear_l257_257721

noncomputable def point := Type
noncomputable def line := point → point → Prop
noncomputable def collinear (P Q H : point) : Prop := ∃ l : line, l P Q ∧ l P H ∧ l Q H

variables (A B C H M N P Q : point)
variables (orthocenter : point → point → point → point)
variables (segment : point → point → set point)
variables (M_on_AB : M ∈ segment A B)
variables (N_on_AC : N ∈ segment A C)
variables (circle_with_diameter : point → point → set point)
variables (P_Q_on_circles : P ∈ circle_with_diameter B N ∧ P ∈ circle_with_diameter C M ∧ Q ∈ circle_with_diameter B N ∧ Q ∈ circle_with_diameter C M)
variables (H_is_orthocenter : orthocenter A B C = H)

theorem PQH_collinear : collinear P Q H :=
sorry

end PQH_collinear_l257_257721


namespace cheaper_to_buy_more_l257_257670

def C (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 20 then 15 * n
  else if 21 ≤ n ∧ n ≤ 40 then 13 * n
  else if 41 ≤ n then 11 * n
  else 0

theorem cheaper_to_buy_more (n : ℕ) : ∃ (n_values : Finset ℕ), n_values.card = 5 ∧ 
  (∀ n ∈ n_values, C(n+1) < C(n)) :=
by
  sorry

end cheaper_to_buy_more_l257_257670


namespace simplify_fraction_l257_257763

theorem simplify_fraction :
  (75 : ℚ) / (225 : ℚ) = 1 / 3 := by
  sorry

end simplify_fraction_l257_257763


namespace sqrt_inequality_l257_257981

theorem sqrt_inequality (c : ℝ) (hc : 0 < c) : sqrt (c - 1) + sqrt (c + 1) < 2 * sqrt c := 
sorry

end sqrt_inequality_l257_257981


namespace projection_of_a_onto_b_l257_257979

open Real

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-2, 4)

theorem projection_of_a_onto_b :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b_squared := vector_b.1 ^ 2 + vector_b.2 ^ 2
  let scalar_projection := dot_product / magnitude_b_squared
  let proj_vector := (scalar_projection * vector_b.1, scalar_projection * vector_b.2)
  proj_vector = (-4/5, 8/5) :=
by
  sorry

end projection_of_a_onto_b_l257_257979


namespace circles_intersect_l257_257647

-- Define conditions
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 3

-- Define the centers of the circles
def center_M : ℝ × ℝ := (0, 0)
def center_N : ℝ × ℝ := (1, 2)

-- Define the radii of the circles
def radius_M : ℝ := Real.sqrt 2
def radius_N : ℝ := Real.sqrt 3

-- Define the distance between the centers
def distance_MN : ℝ := Real.sqrt (1^2 + 2^2)

-- Theorem: The positional relationship between the two circles is intersecting
theorem circles_intersect : Real.sqrt 3 - Real.sqrt 2 < distance_MN ∧ distance_MN < Real.sqrt 2 + Real.sqrt 3 :=
sorry

end circles_intersect_l257_257647


namespace yoongi_more_points_l257_257842

def yoongiPoints : ℕ := 4
def jungkookPoints : ℕ := 6 - 3

theorem yoongi_more_points : yoongiPoints > jungkookPoints := by
  sorry

end yoongi_more_points_l257_257842


namespace total_amount_l257_257861

-- Define the values given in the problem
def r_amount : ℝ := 3600.0000000000005
def fraction_r : ℝ := 2 / 3

-- State the theorem
theorem total_amount (T : ℝ) : fraction_r * T = r_amount → T = 5400.000000000001 :=
by
  intro h
  sorry

end total_amount_l257_257861


namespace coefficient_of_x3_in_expansion_l257_257941

-- Definitions based on the conditions provided
def binomial_expansion_term (n r: ℕ) (a x: ℝ) : ℝ := (Nat.choose n r) * (a^(n-r)) * (x^r)

-- The main theorem to prove
theorem coefficient_of_x3_in_expansion : 
  (polynomial.coeff ((1 - X) * polynomial.expand 5 2) 3 = -40) :=
by sorry

end coefficient_of_x3_in_expansion_l257_257941


namespace pages_left_to_be_read_l257_257808

def total_pages : ℕ := 381
def pages_read : ℕ := 149
def pages_per_day : ℕ := 20
def days_in_week : ℕ := 7

theorem pages_left_to_be_read :
  total_pages - pages_read - (pages_per_day * days_in_week) = 92 := by
  sorry

end pages_left_to_be_read_l257_257808


namespace distinct_integer_values_b_for_quadratic_l257_257571

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end distinct_integer_values_b_for_quadratic_l257_257571


namespace part1_l257_257467

theorem part1 (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x+1) = x^2 + 4x + 1) :
  ∀ x : ℝ, f x = x^2 + 2x - 2 :=
sorry

end part1_l257_257467


namespace centroid_quad_area_correct_l257_257772

noncomputable def centroid_quadrilateral_area (E F G H Q : ℝ × ℝ) (side_length : ℝ) (EQ FQ : ℝ) : ℝ :=
  if h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35 then
    12800 / 9
  else
    sorry

theorem centroid_quad_area_correct (E F G H Q : ℝ × ℝ) (side_length EQ FQ : ℝ) 
  (h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35) :
  centroid_quadrilateral_area E F G H Q side_length EQ FQ = 12800 / 9 :=
sorry

end centroid_quad_area_correct_l257_257772


namespace train_speed_l257_257903

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l257_257903


namespace problem1_problem2_l257_257522

open Real

theorem problem1 : (-2016)^0 + 32 * 2^(2/3) + (1/4)^(-1/2) = 5 := by
  sorry

theorem problem2 : log 3 81 + log 10 20 + log 10 5 + 4^(log 4 2) + log 5 1 = 8 := by
  sorry

end problem1_problem2_l257_257522


namespace find_number_l257_257471

theorem find_number : ∃ x : ℝ, 64 + 5 * x / (180 / 3) = 65 ∧ x = 12 :=
by
  exists 12
  split
  { 
    -- Simplify the original equation 
    have h : (180 / 3) = 60 := by norm_num
    rw [h]
    norm_num
  }
  sorry

end find_number_l257_257471


namespace katie_earnings_l257_257361

theorem katie_earnings :
  4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 = 53 := 
by 
  sorry

end katie_earnings_l257_257361


namespace projection_of_a_onto_b_l257_257977

-- Define the vectors a and b.
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-2, 4)

-- Helper function to calculate the dot product of two vectors.
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Helper function to calculate the squared magnitude of a vector.
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

-- Define the projection function.
def projection (u v : ℝ × ℝ) : ℝ × ℝ := 
  let coeff := (dot_product u v) / (magnitude_squared v)
  in (coeff * v.1, coeff * v.2)

-- Proposition stating the desired result.
theorem projection_of_a_onto_b :
  projection a b = (-4 / 5, 8 / 5) :=
sorry

end projection_of_a_onto_b_l257_257977


namespace acceleration_inverse_square_distance_l257_257156

noncomputable def s (t : ℝ) : ℝ := t^(2/3)

noncomputable def v (t : ℝ) : ℝ := (deriv s t : ℝ)

noncomputable def a (t : ℝ) : ℝ := (deriv v t : ℝ)

theorem acceleration_inverse_square_distance
  (t : ℝ) (h : t ≠ 0) :
  ∃ k : ℝ, k = -2/9 ∧ a t = k / (s t)^2 :=
sorry

end acceleration_inverse_square_distance_l257_257156


namespace smallest_value_w3_z3_l257_257237

theorem smallest_value_w3_z3 (w z : ℂ) 
  (h1 : |w + z| = 3) 
  (h2 : |w^2 + z^2| = 18) : 
  |w^3 + z^3| = 81 / 2 := 
sorry

end smallest_value_w3_z3_l257_257237


namespace negation_of_statement_l257_257017

theorem negation_of_statement (h: ∀ x : ℝ, |x| + x^2 ≥ 0) :
  ¬ (∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_statement_l257_257017


namespace two_results_l257_257020

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l257_257020


namespace number_is_three_l257_257472

theorem number_is_three (n : ℝ) (h : 4 * n - 7 = 5) : n = 3 :=
by sorry

end number_is_three_l257_257472


namespace find_passing_marks_l257_257856

-- Defining the conditions as Lean statements
def condition1 (T P : ℝ) : Prop := 0.30 * T = P - 50
def condition2 (T P : ℝ) : Prop := 0.45 * T = P + 25

-- The theorem to prove
theorem find_passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 200 :=
by
  -- Placeholder proof
  sorry

end find_passing_marks_l257_257856


namespace greatest_product_l257_257090

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l257_257090


namespace small_square_side_length_l257_257707

noncomputable def coordinate := (ℝ × ℝ)

structure Square :=
(A : coordinate)
(B : coordinate)
(C : coordinate)
(D : coordinate)
(side_length : ℝ)
(square_property : (B.1 - A.1) = side_length ∧ (B.2 = A.2) ∧ 
                  (C.2 - D.2) = side_length ∧ (C.1 = D.1) ∧ 
                  (B.1 = C.1) ∧ (A.1 = D.1) ∧ 
                  (A.2 - D.2) = side_length ∧ (C.2 = B.2))

structure EmbeddedSquare :=
(P : coordinate)
(Q : coordinate)
(R : coordinate)
(S : coordinate)
(side_length : ℝ)
(touched_side : coordinate)
(touched_constraint : S.1 = touched_side.1 ∧
                      touched_side.2 ≤ P.2 ∧ 
                      touched_side.2 ≥ C.2)

def middleX (A B : coordinate) : coordinate :=
((A.1 + B.1) / 2, A.2)

theorem small_square_side_length (sq : Square) (emb_sq : EmbeddedSquare)
  (H_sq : sq.side_length = 2)
  (H_P_mid : emb_sq.P = middleX sq.A sq.B)
  (H_S_bc : emb_sq.touched_side = sq.C) : 
  emb_sq.side_length = 1 := 
sorry

end small_square_side_length_l257_257707


namespace domain_of_function_l257_257012

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ x + 2 ≠ 0} = {x : ℝ | x ≥ -3 ∧ x ≠ -2} :=
by
  sorry

end domain_of_function_l257_257012


namespace count_integer_values_of_x_l257_257289

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : {n : ℕ | 121 < n ∧ n ≤ 144}.card = 23 := 
by
  sorry

end count_integer_values_of_x_l257_257289


namespace determine_m_plus_n_l257_257368

noncomputable def P : ℝ → ℝ := λ x, x^2 - 1

noncomputable def a : ℝ := sorry

theorem determine_m_plus_n :
  (P (P (P a)) = 99) →
  a > 0 →
  ∃ (m n : ℕ), a^2 = m + Real.sqrt n ∧ n ≠ 0 ∧ (∀ p : ℕ, Prime p → p^2 ∣ n → False) ∧ (m + n = 12) :=
sorry

end determine_m_plus_n_l257_257368


namespace division_quotient_correct_l257_257964

noncomputable def quotient_of_division (p q : Polynomial ℕ) : Polynomial ℕ :=
  Classical.some (Polynomial.divModUnique p q).1

theorem division_quotient_correct :
  quotient_of_division (Polynomial.mk [5, -8, 9, -7, 3, -4])
                       (Polynomial.mk [1, 2])
  = Polynomial.mk [2, -3, 5, -2.5, 2] :=
sorry

end division_quotient_correct_l257_257964


namespace max_product_of_sum_2024_l257_257097

theorem max_product_of_sum_2024 : 
  ∃ (x y : ℤ), x + y = 2024 ∧ x * y = 1024144 :=
by
  use 1012, 1012
  split
  · sorry
  · sorry

end max_product_of_sum_2024_l257_257097


namespace total_puppies_is_74_l257_257845

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l257_257845


namespace expression_without_arithmetic_square_root_l257_257112

theorem expression_without_arithmetic_square_root : 
  ¬ (∃ x, x^2 = (-|-9|)) :=
by { intro h, cases h with y hy, 
     have hy_nonneg : y^2 ≥ 0 
       := sq_nonneg y,
     let expr := y^2,
     show false,
     calc 
       expr = -|-9| : hy
       ... = -9    : by norm_num
       ... < 0     : by linarith,
}

end expression_without_arithmetic_square_root_l257_257112


namespace points_on_line_l257_257429

-- Define the points involved
def point1 : ℝ × ℝ := (4, 10)
def point2 : ℝ × ℝ := (-2, -8)
def candidate_points : List (ℝ × ℝ) := [(1, 1), (0, -1), (2, 3), (-1, -5), (3, 7)]
def correct_points : List (ℝ × ℝ) := [(1, 1), (-1, -5), (3, 7)]

-- Define a function to check if a point lies on the line defined by point1 and point2
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (10 - (-8)) / (4 - (-2))
  let b := 10 - m * 4
  p.2 = m * p.1 + b

-- Main theorem statement
theorem points_on_line :
  ∀ p ∈ candidate_points, p ∈ correct_points ↔ lies_on_line p :=
sorry

end points_on_line_l257_257429


namespace returns_to_start_total_crawling_time_l257_257148

-- defining the distances the ladybug crawls
def distances : List Int := [+5, -6, +10, -5, -6, +12, -10]

-- defining the speed of the ladybug
def speed := 0.5

-- Prove the ladybug returns to point P
theorem returns_to_start (dists : List Int) : List.sum dists = 0 :=
by
  unfold distances
  sorry

-- Prove the total time of crawling
theorem total_crawling_time (dists : List Int) (sp : Float) : 
  (List.sum (dists.map Int.natAbs)).toFloat / sp = 108 :=
by
  unfold distances speed
  sorry

end returns_to_start_total_crawling_time_l257_257148


namespace tan_angle_addition_l257_257678

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 2) : Real.tan (x + Real.pi / 3) = (5 * Real.sqrt 3 + 8) / -11 := by
  sorry

end tan_angle_addition_l257_257678


namespace probability_interval_l257_257155

open ProbabilityTheory

noncomputable def probability_red_greater_green (x y : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 then if x < y ∧ y < 3 * x then 1 else 0 else 0

theorem probability_interval (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (∫ (x : ℝ) in 0..1, ∫ (y : ℝ) in (x : ℝ)..(min (3 * x) 1), 1) = 5 / 18 := sorry

end probability_interval_l257_257155


namespace factor_expression_l257_257524

theorem factor_expression (x : ℝ) :
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by
  sorry

end factor_expression_l257_257524


namespace side_length_equilateral_base_l257_257414

noncomputable def side_length_of_base (area : ℝ) (slant_height : ℝ) :=
    (2 * area) / slant_height

theorem side_length_equilateral_base :
  (area : ℝ) (slant_height : ℝ) (side_length : ℝ) 
  (h1 : area = 120) (h2 : slant_height = 40) 
  (h3 : side_length = side_length_of_base area slant_height) :
  side_length = 6 :=
by
  sorry

end side_length_equilateral_base_l257_257414


namespace cot_difference_in_triangle_l257_257340

theorem cot_difference_in_triangle
  (A B C D P : Type)
  [triangle A B C] 
  [Foot_of_altitude A B C P]
  (x BD CD BP : ℝ) 
  (h_AD: BD = CD)
  (h_BP: B P == BP)
  (h_median_angle: angle (A-to-D-median) (B-to-C-side) = 30) :
  |cot (angle B A P) - cot (angle C A B)| = 2 * sqrt 3 :=
sorry

end cot_difference_in_triangle_l257_257340


namespace variance_of_sample_l257_257663

theorem variance_of_sample
  (x : ℝ)
  (h : (2 + 3 + x + 6 + 8) / 5 = 5) : 
  (1 / 5) * ((2 - 5) ^ 2 + (3 - 5) ^ 2 + (x - 5) ^ 2 + (6 - 5) ^ 2 + (8 - 5) ^ 2) = 24 / 5 :=
by
  sorry

end variance_of_sample_l257_257663


namespace files_deleted_l257_257390

-- Definitions based on the conditions
def initial_files : ℕ := 93
def files_per_folder : ℕ := 8
def num_folders : ℕ := 9

-- The proof problem
theorem files_deleted : initial_files - (files_per_folder * num_folders) = 21 :=
by
  sorry

end files_deleted_l257_257390


namespace min_points_of_convex_polygon_l257_257465

theorem min_points_of_convex_polygon (M : Finset Point) (h1 : ∃ V : Finset Point, V.card = 7 ∧ (convex_hull ℝ (V : Set Point)).is_polygon 7)
(h2 : ∀ (S : Finset Point), S.card = 5 → ∃ p ∈ M, p ∉ S ∧ p ∈ interior (convex_hull ℝ (S : Set Point))) :
  M.card ≥ 11 :=
sorry

end min_points_of_convex_polygon_l257_257465


namespace range_of_s_l257_257193

-- Define the function s(x)
def s (x : ℝ) : ℝ := 1 / ((2 - x)^2 + 1)

-- State the proof problem
theorem range_of_s : set.Ioo 0 1 ⊆ set.range s ∧ ∀ y, y ∈ set.range s → y ≤ 1 :=
sorry

end range_of_s_l257_257193


namespace find_f_2010_l257_257734

noncomputable def f (a b α β : ℝ) (x : ℝ) :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem find_f_2010 {a b α β : ℝ} (h : f a b α β 2009 = 5) : f a b α β 2010 = 3 :=
sorry

end find_f_2010_l257_257734


namespace no_tiling_with_t_tetrominoes_l257_257330

theorem no_tiling_with_t_tetrominoes :
  (∀ (n b : ℕ), 3 * n + b = 50 → n + 3 * b = 50 → n ∈ ℕ → b ∈ ℕ → (n + b) ≠ 25) :=
sorry

end no_tiling_with_t_tetrominoes_l257_257330


namespace log_alpha_zero_l257_257247

theorem log_alpha_zero (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = x^α) (h2 : f(2) = 2) : 
  Real.logb 2011 α = 0 :=
by
  sorry

end log_alpha_zero_l257_257247


namespace difference_sum_even_odd_l257_257212

/--
The difference between the sum of all even numbers and the sum of all odd numbers from 1 to 100 is 50.
-/
theorem difference_sum_even_odd : 
  let even_sum := (list.range 50).sum * 2 + 50 in
  let odd_sum := (list.range 50).sum * 2 + 0 in
  even_sum - odd_sum = 50 :=
by
  sorry

end difference_sum_even_odd_l257_257212


namespace drivers_schedule_l257_257028

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l257_257028


namespace domain_of_g_l257_257547

noncomputable def g (x : ℝ) : ℝ := Real.cot (Real.arcsin (x ^ 3))

theorem domain_of_g :
  {x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0} = (Set.Icc (-1 : ℝ) 0).erase 0 ∪ (Set.Icc 0 1).erase 0 :=
by
  sorry

end domain_of_g_l257_257547


namespace willam_tax_paid_l257_257419

-- Define our conditions
variables (T : ℝ) (tax_collected : ℝ) (willam_percent : ℝ)

-- Initialize the conditions according to the problem statement
def is_tax_collected (tax_collected : ℝ) : Prop := tax_collected = 3840
def is_farm_tax_levied_on_cultivated_land : Prop := true -- Essentially means we acknowledge it is 50%
def is_willam_taxable_land_percentage (willam_percent : ℝ) : Prop := willam_percent = 0.25

-- The final theorem that states Mr. Willam's tax payment is $960 given the conditions
theorem willam_tax_paid  : 
  ∀ (T : ℝ),
  is_tax_collected 3840 → 
  is_farm_tax_levied_on_cultivated_land →
  is_willam_taxable_land_percentage 0.25 →
  0.25 * 3840 = 960 :=
sorry

end willam_tax_paid_l257_257419


namespace initial_number_is_2008_l257_257346

theorem initial_number_is_2008 (N : ℕ) (h_initial : N > 0)
  (h_reach : ∀ x : ℕ, (x = 2008 → ∃ n ≥ 0, ↑n = 2008 :={z | ∃ x y : ℕ, x = 2*x + 1 ∨ x = x/(x+2)})
  (h_derive : N = 2008) :=
begin
sorry
end

end initial_number_is_2008_l257_257346


namespace max_area_of_rectangle_with_perimeter_60_l257_257620

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l257_257620


namespace integer_values_of_b_for_quadratic_eqn_l257_257563

noncomputable def number_of_integer_values_of_b : ℕ := 16

theorem integer_values_of_b_for_quadratic_eqn :
  ∃(b : ℤ) (k ≥ 0), ∀m n : ℤ, (m + n = -b ∧ m * n = 12 * b) → (m + 12) * (n + 12) = 144 → k = number_of_integer_values_of_b := sorry

end integer_values_of_b_for_quadratic_eqn_l257_257563


namespace number_of_possible_values_of_x_l257_257296

theorem number_of_possible_values_of_x : 
  (∃ x : ℕ, ⌈Real.sqrt x⌉ = 12) → (set.Ico 144 169).card = 25 := 
by
  intros h
  sorry

end number_of_possible_values_of_x_l257_257296


namespace unpainted_region_area_is_correct_l257_257442

-- Define the widths of the boards and the angle.
def width_board1 : ℝ := 5
def width_board2 : ℝ := 7
def crossing_angle : ℝ := 45 * Real.pi / 180 -- convert degrees to radians

-- Calculate the intersection lengths
def length_intersection_board1 : ℝ := width_board1 * Real.sqrt 2
def height_intersection_board2 : ℝ := width_board2 * Real.sqrt 2

-- The area of parallelogram
def area_unpainted_region : ℝ := length_intersection_board1 * width_board2

theorem unpainted_region_area_is_correct : area_unpainted_region = 35 * Real.sqrt 2 := by
  -- Placeholder for the actual proof
  sorry

end unpainted_region_area_is_correct_l257_257442


namespace find_CD_right_triangle_l257_257329

theorem find_CD_right_triangle 
  (AC BC AD CD : ℝ) 
  (hAC : AC = 15) 
  (hBC : BC = 20) 
  (hAD : AD = 4) 
  (hABC : AC ^ 2 + BC ^ 2 = (AC ^ 2 + BC ^ 2).sqrt ^ 2)
  (hcosA : cos (real.arccos (AC / (AC ^ 2 + BC ^ 2).sqrt)) = 3 / 5) :
  CD = 13 := 
by
  sorry

end find_CD_right_triangle_l257_257329


namespace monotonic_decreasing_interval_l257_257797

noncomputable def f (x : ℝ) : ℝ := log 2 (x^2 - x - 2)

def domain (x : ℝ) : Prop := (x^2 - x - 2 > 0)

def is_monotonic_decreasing (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f x > f y

theorem monotonic_decreasing_interval :
  is_monotonic_decreasing f {x | x < -1} ↔ True :=
sorry

#eval monotonic_decreasing_interval

end monotonic_decreasing_interval_l257_257797


namespace sum_multiple_of_five_l257_257306

noncomputable def sum_formula (n : ℕ) : ℂ := 
  ∑ k in Finset.range (n + 1), (k + 1 : ℂ) * (complex.I ^ k)

theorem sum_multiple_of_five (n : ℕ) (h : n % 5 = 0) :
  sum_formula n = (7 * n + 5 - 2 * n * complex.I) / 5 :=
by
  sorry

end sum_multiple_of_five_l257_257306


namespace train_speed_is_correct_l257_257167

   -- Define the conditions
   def train_length : ℝ := 280
   def time_to_pass_tree : ℝ := 14

   -- Define conversion factors
   def meters_per_second_to_km_per_hour : ℝ := 1000 / 3600

   -- Define the expected speed result
   def expected_speed_km_per_hour : ℝ := 5.56

   -- Theorem stating the given conditions imply the correct speed
   theorem train_speed_is_correct :
     (train_length / time_to_pass_tree) * meters_per_second_to_km_per_hour = expected_speed_km_per_hour := by
   sorry
   
end train_speed_is_correct_l257_257167


namespace valid_permutations_count_l257_257733

def is_inversion {α : Type*} [DecidableEq α] (l : List α) (x y : α) : Prop :=
  x ∈ l ∧ y ∈ l ∧ List.indexOf x l < List.indexOf y l

def inversion_number {α : Type*} [DecidableEq α] (l : List α) (x : α) : Nat :=
  l.countp (λ y, is_inversion l y x)

def valid_permutations (l : List Nat) : Prop :=
  inversion_number l 8 = 2 ∧
  inversion_number l 6 = 3 ∧
  inversion_number l 2 = 0

def count_valid_permutations : Nat :=
  (List.permutations [1, 2, 3, 4, 5, 6, 7, 8]).countp valid_permutations

theorem valid_permutations_count :
  count_valid_permutations = 420 :=
by
  sorry

end valid_permutations_count_l257_257733


namespace sqrt_21_between_4_and_5_l257_257544

theorem sqrt_21_between_4_and_5 : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := 
by 
  sorry

end sqrt_21_between_4_and_5_l257_257544


namespace collinearity_of_I_I_l257_257692

noncomputable def Gergonne_point (A B C : Point) : Point := sorry

noncomputable def Circle_passing_through_Ge (Ge : Point) (A B C : Point) : Circle := sorry

noncomputable def tangent_circles (Γ₁ Γ₂ Γ₃ : Circle) : Point × Point × Point := sorry

noncomputable def incenter_triangle (A B C : Point) : Point := sorry

noncomputable def incenter_PQR (P Q R : Point) : Point := sorry

theorem collinearity_of_I_I'_Ge 
  (A B C : Point)
  (Γ₁ : Circle)
  (Γ₂ : Circle)
  (Γ₃ : Circle)
  (Ge := Gergonne_point A B C)
  (I := incenter_triangle A B C)
  (PQR := tangent_circles Γ₁ Γ₂ Γ₃)
  (P := PQR.1) (Q := PQR.2.1) (R := PQR.2.2)
  (I_prime := incenter_PQR P Q R) :
  collinear I I_prime Ge := 
sorry

end collinearity_of_I_I_l257_257692


namespace yuri_total_puppies_l257_257852

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l257_257852


namespace graph_passes_fixed_point_l257_257420

-- Begin the proof
theorem graph_passes_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ x y, (x = 1) ∧ (y = -4) ∧ (y = a^(x - 1) - 5) :=
by {
  -- Provide an explicit fixed point that will be checked
  use (1, -4),
  split,
  -- Show x = 1
  exact rfl,
  split,
  -- Show y = -4
  exact rfl,
  -- Show y satisfies the function
  have h : (1 : ℝ) - 1 = 0 := by linarith,
  rw [h, pow_zero, sub_eq_add_neg],
  exact rfl,
}

end graph_passes_fixed_point_l257_257420


namespace pencils_in_total_l257_257206

theorem pencils_in_total
  (rows : ℕ) (pencils_per_row : ℕ) (total_pencils : ℕ)
  (h1 : rows = 14)
  (h2 : pencils_per_row = 11)
  (h3 : total_pencils = rows * pencils_per_row) :
  total_pencils = 154 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end pencils_in_total_l257_257206


namespace max_value_m_l257_257251

theorem max_value_m {m : ℝ} (h : ∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → m ≤ Real.tan x + 1) : m = 2 :=
sorry

end max_value_m_l257_257251


namespace sum_prime_factors_1320_l257_257451

noncomputable def sum_of_prime_factors (n : ℕ) : ℕ :=
  let factors := nat.factors n
  let distinctFactors := factors.toFinset
  distinctFactors.sum

theorem sum_prime_factors_1320 : sum_of_prime_factors 1320 = 21 :=
  by
  sorry

end sum_prime_factors_1320_l257_257451


namespace least_number_to_add_1054_23_l257_257832

def least_number_to_add (n k : ℕ) : ℕ :=
  let remainder := n % k
  if remainder = 0 then 0 else k - remainder

theorem least_number_to_add_1054_23 : least_number_to_add 1054 23 = 4 :=
by
  -- This is a placeholder for the actual proof
  sorry

end least_number_to_add_1054_23_l257_257832


namespace sum_pattern_l257_257770

theorem sum_pattern (a b : ℕ) : (6 + 7 = 13) ∧ (8 + 9 = 17) ∧ (5 + 6 = 11) ∧ (7 + 8 = 15) ∧ (3 + 3 = 6) → (6 + 7 = 12) :=
by
  sorry

end sum_pattern_l257_257770


namespace number_of_possible_values_of_x_l257_257297

theorem number_of_possible_values_of_x : 
  (∃ x : ℕ, ⌈Real.sqrt x⌉ = 12) → (set.Ico 144 169).card = 25 := 
by
  intros h
  sorry

end number_of_possible_values_of_x_l257_257297


namespace bus_stoppage_time_per_hour_l257_257858

theorem bus_stoppage_time_per_hour
  (speed_excluding_stoppages : ℕ) 
  (speed_including_stoppages : ℕ)
  (h1 : speed_excluding_stoppages = 54) 
  (h2 : speed_including_stoppages = 45) 
  : (60 * (speed_excluding_stoppages - speed_including_stoppages) / speed_excluding_stoppages) = 10 :=
by sorry

end bus_stoppage_time_per_hour_l257_257858


namespace max_rectangle_area_l257_257626

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l257_257626


namespace Freddy_journey_time_l257_257948

/-- Eddy and Freddy start simultaneously from city A. Eddy travels to city B, Freddy travels to city C.
    Eddy takes 3 hours from city A to city B, which is 900 km. The distance between city A and city C is
    300 km. The ratio of average speed of Eddy to Freddy is 4:1. Prove that Freddy takes 4 hours to travel. -/
theorem Freddy_journey_time (t_E : ℕ) (d_AB : ℕ) (d_AC : ℕ) (r : ℕ) (V_E V_F t_F : ℕ)
    (h1 : t_E = 3)
    (h2 : d_AB = 900)
    (h3 : d_AC = 300)
    (h4 : r = 4)
    (h5 : V_E = d_AB / t_E)
    (h6 : V_E = r * V_F)
    (h7 : t_F = d_AC / V_F)
  : t_F = 4 := 
  sorry

end Freddy_journey_time_l257_257948


namespace bacteria_division_l257_257480

theorem bacteria_division (initial_bacteria : ℕ) (division_time : ℕ) (hours : ℕ) :
  initial_bacteria = 1 → division_time = 20 → hours = 3 →
  let minutes := hours * 60 in
  let divisions := minutes / division_time in
  let final_bacteria := initial_bacteria * 2 ^ divisions in
  final_bacteria = 512 :=
by
  intros h1 h2 h3
  let minutes := hours * 60
  let divisions := minutes / division_time
  let final_bacteria := initial_bacteria * 2 ^ divisions
  have h_minutes: minutes = 180 := by sorry
  have h_divisions: divisions = 9 := by sorry
  have h_final_bacteria: final_bacteria = 512 := by sorry
  exact h_final_bacteria

end bacteria_division_l257_257480


namespace area_of_circular_segment_l257_257421

-- The main theorem stating that the area of the circular segment
theorem area_of_circular_segment (i h : ℝ) (i_eq : i = 10) (h_eq : h = 8) : 
  ∃ t, t = 14.6 :=
by
  -- Conditions
  have i_val : i = 10 := i_eq
  have h_val : h = 8 := h_eq

  -- Let t denote the area of the circular segment
  let t := 14.6

  -- Conclude that such a t exists
  use t
  -- Verification
  exact rfl

end area_of_circular_segment_l257_257421


namespace remainder_of_sum_division_l257_257744

theorem remainder_of_sum_division (x y : ℕ) (k m : ℕ) 
  (hx : x = 90 * k + 75) (hy : y = 120 * m + 115) :
  (x + y) % 30 = 10 :=
by sorry

end remainder_of_sum_division_l257_257744


namespace toothbrushes_given_l257_257486

theorem toothbrushes_given (hours_per_visit : ℝ) (hours_per_day : ℝ) (days_per_week : ℕ) (toothbrushes_per_visit : ℕ) 
  (h1 : hours_per_visit = 0.5) 
  (h2 : hours_per_day = 8) 
  (h3 : days_per_week = 5) 
  (h4 : toothbrushes_per_visit = 2) : 
  (toothbrushes_per_visit * (nat.floor (hours_per_day / hours_per_visit) * days_per_week)) = 160 :=
by
  sorry

end toothbrushes_given_l257_257486


namespace mode_eighth_grade_median_ninth_grade_excellent_health_ninth_grade_l257_257325

def eighth_grade_scores : List ℝ := [78, 86, 74, 81, 75, 76, 87, 70, 75, 90, 75, 79, 81, 76, 74, 80, 86, 69, 83, 77]
def ninth_grade_scores : List ℝ := [93, 73, 88, 81, 72, 81, 94, 83, 77, 83, 80, 81, 70, 81, 73, 78, 82, 80, 70, 40]
def total_ninth_grade_students := 180

theorem mode_eighth_grade : List.mode eighth_grade_scores = 75 := 
by 
    sorry

theorem median_ninth_grade : List.median ninth_grade_scores = 80.5 := 
by 
    sorry

theorem excellent_health_ninth_grade : 
    let count := List.count (λ x, 80 ≤ x ∧ x ≤ 100) ninth_grade_scores in 
    total_ninth_grade_students * count / List.length ninth_grade_scores = 108 := 
by 
    sorry

end mode_eighth_grade_median_ninth_grade_excellent_health_ninth_grade_l257_257325


namespace ratio_c_b_l257_257650

theorem ratio_c_b (x y a b c : ℝ) (h1 : x ≥ 1) (h2 : x + y ≤ 4) (h3 : a * x + b * y + c ≤ 0) 
    (h_max : ∀ x y, (x,y) = (2, 2) → 2 * x + y = 6) (h_min : ∀ x y, (x,y) = (1, -1) → 2 * x + y = 1) (h_b : b ≠ 0) :
    c / b = 4 := sorry

end ratio_c_b_l257_257650


namespace max_area_of_rectangle_with_perimeter_60_l257_257619

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l257_257619


namespace find_area_OCD_l257_257926

-- Given areas of trapezoid ABCD and parallelogram CDEF
variable (area_trap : ℝ) (area_para : ℝ)

-- Conditions from the problem
def cond1 : Prop := area_trap = 320
def cond2 : Prop := area_para = 240

-- Goal: Prove the area of triangle OCD
def goal (area_triangle : ℝ) : Prop := area_triangle = 45

-- Proposition combining all given conditions and proving the goal
theorem find_area_OCD (area_triangle : ℝ) (h1 : cond1) (h2 : cond2) : goal area_triangle :=
sorry

end find_area_OCD_l257_257926


namespace angle_APF_eq_angle_BPD_l257_257367

-- Statement of the problem in Lean
theorem angle_APF_eq_angle_BPD 
  {A B C D E F P : Point}
  (h_eq_triangle : equilateral_triangle A B C)
  (h_midpoint_D : midpoint D B C)
  (h_point_E : on_line E A C)
  (h_point_F : on_line F A B)
  (h_AF_eq_CE : dist A F = dist C E)
  (h_P_intersection : intersection P (line B E) (line C F)):
  angle A P F = angle B P D :=
sorry

end angle_APF_eq_angle_BPD_l257_257367


namespace total_length_of_fence_l257_257857

theorem total_length_of_fence
  (x : ℝ)
  (h1 : (2 : ℝ) * x ^ 2 = 200) :
  (2 * x + 2 * x) = 40 :=
by
sorry

end total_length_of_fence_l257_257857


namespace max_4x3_y3_l257_257376

theorem max_4x3_y3 (x y : ℝ) (h1 : x ≤ 2) (h2 : y ≤ 3) (h3 : x + y = 3) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : 
  4 * x^3 + y^3 ≤ 33 :=
sorry

end max_4x3_y3_l257_257376


namespace equal_division_of_balls_l257_257870

def total_balls : ℕ := 10
def num_boxes : ℕ := 5
def balls_per_box : ℕ := total_balls / num_boxes

theorem equal_division_of_balls :
  balls_per_box = 2 :=
by
  sorry

end equal_division_of_balls_l257_257870


namespace greatest_product_l257_257093

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l257_257093


namespace arithmetic_square_root_of_neg_4_squared_l257_257007

theorem arithmetic_square_root_of_neg_4_squared : 
  let x := (-4)^2
  in Real.sqrt x = 4 :=
by 
  let x := (-4:ℝ)^2
  show Real.sqrt x = 4
  sorry

end arithmetic_square_root_of_neg_4_squared_l257_257007


namespace train_speed_l257_257901

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor : ℝ) (speed : ℝ) : 
  length = 100 → 
  time = 9.99920006399488 → 
  conversion_factor = 3.6 → 
  speed = 36.002879976562 :=
by
  assume h1 h2 h3
  sorry

end train_speed_l257_257901


namespace expected_value_eq_5_5_l257_257577

def probability_roll (n : ℕ) : ℚ :=
  if n = 8 then 3 / 8
  else if 1 ≤ n ∧ n ≤ 7 then 5 / 56
  else 0

noncomputable def expected_value : ℚ :=
  ∑ n in {1, 2, 3, 4, 5, 6, 7, 8}, n * probability_roll n

theorem expected_value_eq_5_5 : expected_value = 5.5 := by
  sorry

end expected_value_eq_5_5_l257_257577


namespace find_a_l257_257242

noncomputable def f : ℝ → ℝ → ℝ :=
  λ a x, if x < 1 then 2 * x + a else - x - 2 * a

theorem find_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) :
  a = -3/4 :=
sorry

end find_a_l257_257242


namespace translate_parabola_l257_257073

-- Translating the parabola y = (x-2)^2 - 8 three units left and five units up
theorem translate_parabola (x y : ℝ) :
  y = (x - 2) ^ 2 - 8 →
  y = ((x + 3) - 2) ^ 2 - 8 + 5 →
  y = (x + 1) ^ 2 - 3 := by
sorry

end translate_parabola_l257_257073


namespace max_product_of_two_integers_l257_257087

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l257_257087


namespace greatest_product_l257_257092

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l257_257092


namespace positive_diff_mean_median_eq_14_l257_257436

-- Definitions for the given conditions
def vertical_drops : List ℕ := [190, 130, 155, 320, 220, 180]

-- Mean of the vertical drops
def mean (lst : List ℕ) : ℕ := lst.sum / lst.length

-- Median of the vertical drops, assuming the list is sorted and has an even number of elements
def median (lst : List ℕ) : ℕ := (lst[2] + lst[3]) / 2

-- Positive difference between mean and median
def positive_diff_mean_median (lst : List ℕ) : ℕ := 
  abs (mean lst - median lst)

-- Main theorem statement
theorem positive_diff_mean_median_eq_14 : positive_diff_mean_median vertical_drops = 14 :=
by
  sorry

end positive_diff_mean_median_eq_14_l257_257436


namespace min_value_fraction_l257_257661

theorem min_value_fraction (b c: ℝ) (hb: 0 < b) (hc: 0 < c) (h: b + c = 1) : 
  (4 / b) + (1 / c) = 9 :=
begin
  -- proof omitted
  sorry
end

end min_value_fraction_l257_257661


namespace problem1_l257_257186

theorem problem1 : Real.sqrt 36 + Real.cbrt (-1/64) - 1 / Real.sqrt 16 = 11 / 2 := 
  sorry

end problem1_l257_257186


namespace ali_red_caps_l257_257173

-- Problem conditions
def total_caps : ℕ := 125
def percent_green : ℝ := 0.60

-- Correct answer derived from the conditions
def expected_red_caps (total_caps : ℕ) (percent_green : ℝ) : ℕ :=
  let percent_red := 1.0 - percent_green
  (percent_red * total_caps).toInt

-- The theorem we need to prove
theorem ali_red_caps : expected_red_caps total_caps percent_green = 50 := by
  sorry

end ali_red_caps_l257_257173


namespace line_through_point_and_parallel_l257_257525

def point_A : ℝ × ℝ × ℝ := (-2, 3, 1)

def plane1 (x y z : ℝ) := x - 2*y - z - 2 = 0
def plane2 (x y z : ℝ) := 2*x + 3*y - z + 1 = 0

theorem line_through_point_and_parallel (x y z t : ℝ) :
  ∃ t, 
    x = 5 * t - 2 ∧
    y = -t + 3 ∧
    z = 7 * t + 1 :=
sorry

end line_through_point_and_parallel_l257_257525


namespace option_C_option_D_l257_257702

section
variables {k x y x₀ y₀ : ℝ}

-- Option C
theorem option_C (x₀ y₀ : ℝ) : 
  ∀ (l : ℝ → ℝ) (h₁ : l x₀ = y₀) (h₂ : ∀ x, l x = y₀), 
  ∀ (m : ℝ), m = real.pi/2 → (∀ x, l x = y₀) → ∀ (x : ℝ), x = x₀ :=
sorry

-- Option D
theorem option_D (k : ℝ) : 
  ∀ (x y : ℝ), x = -1 ∧ y = 3 → y - 3 = k * (x + 1) :=
sorry

end

end option_C_option_D_l257_257702


namespace volume_of_circumscribed_sphere_l257_257058

theorem volume_of_circumscribed_sphere (A B C D : ℝ × ℝ × ℝ)
  (hA : A = (1, 0, 1)) (hB : B = (1, 1, 0)) (hC : C = (0, 1, 0)) (hD : D = (1, 1, 1)) :
  ∃ (V : ℝ), V = (sqrt 3 / 2) * π :=
sorry

end volume_of_circumscribed_sphere_l257_257058


namespace last_three_digits_of_7_pow_83_l257_257960

theorem last_three_digits_of_7_pow_83 :
  (7 ^ 83) % 1000 = 886 := sorry

end last_three_digits_of_7_pow_83_l257_257960


namespace seven_people_arrangement_l257_257405

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def perm (n k : Nat) : Nat :=
  factorial n / factorial (n - k)

theorem seven_people_arrangement : 
  (perm 5 5) * (perm 6 2) = 3600 := by
sorry

end seven_people_arrangement_l257_257405


namespace merchant_loss_l257_257887

-- Definitions
def cost_A_usd : ℤ := 300
def exchange_rate_usd_inr : ℤ := 100
def sales_tax_A : ℤ := 5

def cost_B_eur : ℤ := 400
def exchange_rate_eur_inr : ℤ := 120
def sales_tax_B : ℤ := 10

def cost_C_gbp : ℤ := 500
def exchange_rate_gbp_inr : ℤ := 150
def sales_tax_C : ℤ := 7

def selling_price_A : ℤ := 33000
def selling_price_B : ℤ := 45000
def selling_price_C : ℤ := 56000

-- Total cost prices in INR including sales tax
def cost_A_inr : ℤ := cost_A_usd * exchange_rate_usd_inr
def total_cost_A_inr : ℤ := cost_A_inr + cost_A_inr * sales_tax_A / 100

def cost_B_inr : ℤ := cost_B_eur * exchange_rate_eur_inr
def total_cost_B_inr : ℤ := cost_B_inr + cost_B_inr * sales_tax_B / 100

def cost_C_inr : ℤ := cost_C_gbp * exchange_rate_gbp_inr
def total_cost_C_inr : ℤ := cost_C_inr + cost_C_inr * sales_tax_C / 100

-- Total cost price (TCP)
def TCP : ℤ := total_cost_A_inr + total_cost_B_inr + total_cost_C_inr

-- Total selling price (TSP)
def TSP : ℤ := selling_price_A + selling_price_B + selling_price_C

-- Loss
def loss : ℤ := TCP - TSP

-- Loss percentage
def loss_percentage : ℚ := loss.to_rat / TCP.to_rat * 100

-- Proof statement
theorem merchant_loss : loss_percentage ≈ 18.57 := by
  sorry

end merchant_loss_l257_257887


namespace find_BD_l257_257690

noncomputable def isosceles_triangle (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] :=
  (dist A B = 6) ∧ (dist A C = 5) ∧ (dist B C = 5) ∧ (dist C D = 10) ∧ (B ∈ segment ℝ A D)

theorem find_BD (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (h : isosceles_triangle A B C D) : dist B D = -3 + 2 * real.sqrt 21 :=
sorry

end find_BD_l257_257690


namespace ratio_of_areas_is_16_l257_257427

-- Definitions and conditions
variables (a b : ℝ)

-- Given condition: Perimeter of the larger square is 4 times the perimeter of the smaller square
def perimeter_relation (ha : a = 4 * b) : Prop := a = 4 * b

-- Theorem to prove: Ratio of the area of the larger square to the area of the smaller square is 16
theorem ratio_of_areas_is_16 (ha : a = 4 * b) : (a^2 / b^2) = 16 :=
by
  sorry

end ratio_of_areas_is_16_l257_257427


namespace possible_values_y_l257_257377

theorem possible_values_y (x : ℝ) (h : x^2 + 4 * (x / (x - 2))^2 = 45) : 
  ∃ y : ℝ, y = 2 ∨ y = 16 :=
sorry

end possible_values_y_l257_257377


namespace sum_of_products_le_zero_l257_257649

variable {n : ℕ} {a : Fin n → ℝ}

theorem sum_of_products_le_zero (h : (∑ i, a i) = 0) : (∑ i in Finset.range n, ∑ j in Finset.range n, if i < j then a i * a j else 0) ≤ 0 := 
by
  sorry

end sum_of_products_le_zero_l257_257649


namespace max_area_of_rectangular_pen_l257_257601

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l257_257601


namespace max_product_l257_257099

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l257_257099


namespace series_sum_eq_l257_257182

-- Definitions from conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 8

-- Theorem statement
theorem series_sum_eq :
  (∑ i in Finset.range n, a * r^i) = 255 / 256 :=
sorry

end series_sum_eq_l257_257182


namespace batsman_sixes_l257_257474

theorem batsman_sixes 
(scorer_runs : ℕ)
(boundaries : ℕ)
(run_contrib : ℕ → ℚ)
(score_by_boundary : ℕ)
(score : ℕ)
(h1 : scorer_runs = 125)
(h2 : boundaries = 5)
(h3 : ∀ (x : ℕ), run_contrib x = (0.60 * scorer_runs : ℚ))
(h4 : score_by_boundary = boundaries * 4)
(h5 : score = scorer_runs - score_by_boundary) : 
∃ (x : ℕ), x = 5 ∧ (scorer_runs = score + (x * 6)) :=
by
  sorry

end batsman_sixes_l257_257474


namespace alloy_cut_weight_l257_257412

variable (a b x : ℝ)
variable (ha : 0 ≤ a ∧ a ≤ 1) -- assuming copper content is a fraction between 0 and 1
variable (hb : 0 ≤ b ∧ b ≤ 1)
variable (h : a ≠ b)
variable (hx : 0 < x ∧ x < 40) -- x is strictly between 0 and 40 (since 0 ≤ x ≤ 40)

theorem alloy_cut_weight (A B : ℝ) (hA : A = 40) (hB : B = 60) (h1 : (a * x + b * (A - x)) / 40 = (b * x + a * (B - x)) / 60) : x = 24 :=
by
  sorry

end alloy_cut_weight_l257_257412


namespace max_area_of_rectangular_pen_l257_257604

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l257_257604


namespace integer_values_of_b_l257_257566

theorem integer_values_of_b (b : ℤ) : 
  (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0 ∧ x ≠ y) → 
  ∃ S : finset ℤ, S.card = 8 ∧ ∀ c ∈ S, ∃ x : ℤ, x^2 + c * x + 12 * c = 0 :=
sorry

end integer_values_of_b_l257_257566


namespace real_numbers_satisfy_relation_l257_257546

theorem real_numbers_satisfy_relation (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) → 
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end real_numbers_satisfy_relation_l257_257546


namespace epsilon_regular_partition_exists_l257_257223

theorem epsilon_regular_partition_exists
  (ε : ℝ) (hε : ε > 0)
  (V : Type*) (C D : set V) (hCD : disjoint C D)
  (h_eps_reg : ¬ ε_regular C D ε) :
  ∃ (C1 C2 : set V) (D1 D2 : set V),
    C = C1 ∪ C2 ∧ C1 ∩ C2 = ∅ ∧
    D = D1 ∪ D2 ∧ D1 ∩ D2 = ∅ ∧
    q (C1 ∪ C2, D1 ∪ D2) ≥ q (C, D) + ε^4 * (|C| * |D|) / n^2 :=
by sorry -- proof omitted

end epsilon_regular_partition_exists_l257_257223


namespace tips_fraction_to_salary_l257_257170

theorem tips_fraction_to_salary (S T I : ℝ)
  (h1 : I = S + T)
  (h2 : T / I = 0.6923076923076923) :
  T / S = 2.25 := by
  sorry

end tips_fraction_to_salary_l257_257170


namespace sum_of_coefficients_is_37_l257_257549

open Polynomial

noncomputable def polynomial := 
  -3 * (X ^ 8 - 2 * X ^ 5 + 4 * X ^ 3 - 6) +
  5 * (2 * X ^ 4 + 3 * X ^ 2 - X) -
  2 * (3 * X ^ 6 - 7)

theorem sum_of_coefficients_is_37 : (polynomial.eval 1 polynomial) = 37 :=
by
  sorry

end sum_of_coefficients_is_37_l257_257549


namespace det_3A_B_l257_257286

variable (A B : Matrix (Fin n) (Fin n) ℝ)
variable (hA : det A = -3)
variable (hB : det B = 5)

theorem det_3A_B (n : ℕ) : det (3 • A ⬝ B) = -15 * 3^n := 
by sorry

end det_3A_B_l257_257286


namespace jack_pages_l257_257713

theorem jack_pages (pages_per_booklet : ℕ) (num_booklets : ℕ) (h1 : pages_per_booklet = 9) (h2 : num_booklets = 49) : num_booklets * pages_per_booklet = 441 :=
by {
  sorry
}

end jack_pages_l257_257713


namespace find_x_l257_257811

theorem find_x (x : ℕ) (h_odd : x % 2 = 1) (h_pos : 0 < x) :
  (∃ l : List ℕ, l.length = 8 ∧ (∀ n ∈ l, n < 80 ∧ n % 2 = 1) ∧ l.Nodup = true ∧
  (∀ k m, k > 0 → m % 2 = 1 → k * x * m ∈ l)) → x = 5 := by
  sorry

end find_x_l257_257811


namespace sum_f_1_to_2013_l257_257535

def f (x : ℝ) : ℝ :=
  if h : (-3 ≤ x) ∧ (x < -1) then -(x + 2) ^ 2
  else if h : (-1 ≤ x) ∧ (x < 3) then x
  else f (x % 6)

theorem sum_f_1_to_2013 : (∑ n in Finset.range 2014, f n) = 337 := 
sorry

end sum_f_1_to_2013_l257_257535


namespace f_at_2_l257_257591

noncomputable def f : ℝ → ℝ
| x if x < 0  := Math.sin (Real.pi / 2 * x)
| x           := f (x - 1) + 2

theorem f_at_2 : f 2 = 5 :=
sorry

end f_at_2_l257_257591


namespace expected_return_correct_l257_257914

-- Define the probabilities
def p1 := 1/4
def p2 := 1/4
def p3 := 1/6
def p4 := 1/3

-- Define the payouts
def payout (n : ℕ) (previous_odd : Bool) : ℝ :=
  match n with
  | 1 => 2
  | 2 => if previous_odd then -3 else 0
  | 3 => 0
  | 4 => 5
  | _ => 0

-- Define the expected values of one throw
def E1 : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

def E2_odd : ℝ :=
  p1 * payout 1 true + p2 * payout 2 true + p3 * payout 3 true + p4 * payout 4 true

def E2_even : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

-- Define the probability of throwing an odd number first
def p_odd : ℝ := p1 + p3

-- Define the probability of not throwing an odd number first
def p_even : ℝ := 1 - p_odd

-- Define the total expected return
def total_expected_return : ℝ :=
  E1 + (p_odd * E2_odd + p_even * E2_even)


theorem expected_return_correct :
  total_expected_return = 4.18 :=
  by
    -- The proof is omitted
    sorry

end expected_return_correct_l257_257914


namespace largest_root_of_quadratic_l257_257406

theorem largest_root_of_quadratic :
  ∃ (x : ℝ), (3 * (8 * x ^ 2 + 10 * x + 8) = x * (8 * x - 34)) ∧
             (∀ y, 3 * (8 * y ^ 2 + 10 * y + 8) = y * (8 * y - 34) → y ≤ x) ∧
             x = -2 + sqrt 10 / 2 :=
begin
  sorry
end

end largest_root_of_quadratic_l257_257406


namespace tap_turn_off_time_l257_257075

theorem tap_turn_off_time (C : ℝ) (x : ℝ) (x_fill : ℕ) :
  (C / 45) + (C / 40) = (45 * 40) / 360 ∧ 
  (x * ((C / 45) + (C / 40)) + 23 * (C / 40) = C) :=
begin
    sorry
end

end tap_turn_off_time_l257_257075


namespace solve_equation_solutions_count_l257_257766

open Real

theorem solve_equation_solutions_count :
  (∃ (x_plural : List ℝ), (∀ x ∈ x_plural, 2 * sqrt 2 * (sin (π * x / 4)) ^ 3 = cos (π / 4 * (1 - x)) ∧ 0 ≤ x ∧ x ≤ 2020) ∧ x_plural.length = 505) :=
sorry

end solve_equation_solutions_count_l257_257766


namespace xiaohong_total_score_l257_257876

theorem xiaohong_total_score :
  ∀ (midterm_score final_score : ℕ) (midterm_weight final_weight : ℝ),
    midterm_score = 80 →
    final_score = 90 →
    midterm_weight = 0.4 →
    final_weight = 0.6 →
    (midterm_score * midterm_weight + final_score * final_weight) = 86 :=
by
  intros midterm_score final_score midterm_weight final_weight
  intros h1 h2 h3 h4
  sorry

end xiaohong_total_score_l257_257876


namespace mapping_count_l257_257997

open Finset

theorem mapping_count (A B : Finset ℕ) (hA : A = {1, 2, 3, 4}) (hB : B = {3, 4, 5}) :
  let M := A ∩ B,
      N := A ∪ B
  in M.card = 2 ∧ N.card = 5 → (N.card ^ M.card) = 25 :=
by
  intro h
  let M := A ∩ B
  let N := A ∪ B
  cases h with hM hN
  rw [hM, hN]
  exact (Nat.pow 5 2)

end mapping_count_l257_257997


namespace find_positive_a_l257_257975

noncomputable def find_a (a : ℝ) : Prop :=
  ((x - a/x)^7).coeff x^3 = 84 ∧ a > 0

theorem find_positive_a (a : ℝ) : find_a a → a = 2 :=
by
  sorry

end find_positive_a_l257_257975


namespace ball_distribution_count_l257_257394

theorem ball_distribution_count : 
  ∃ (x y z : ℕ), x + y + z = 9 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
  ∀ (x' y' z' : ℕ), (x' + y' + z' = 6) → (x' ≠ y' ∧ y' ≠ z' ∧ z' ≠ x') →
    18 :=
sorry

end ball_distribution_count_l257_257394


namespace decimal_417th_digit_of_8_over_19_l257_257081

theorem decimal_417th_digit_of_8_over_19 :
  let frac := 8 / 19 in
  let digits := "421052631578947368" in
  let cycle_length := 18 in
  let remainder := 417 % cycle_length in
  digits.nth (remainder - 1) = some '1' :=
by
  sorry

end decimal_417th_digit_of_8_over_19_l257_257081


namespace cabd_13th_permutation_l257_257835

theorem cabd_13th_permutation : 
  ∃ (n : ℕ), 
  (∀ (s : List Char), s ∈ ("A" :: "B" :: "C" :: "D" :: []).permutations) →
  (List.lexPermutations ("A" :: "B" :: "C" :: "D" :: []) !! n = some "CABD".toList) ∧ 
  n = 12 := sorry

end cabd_13th_permutation_l257_257835


namespace area_of_triangle_AOB_l257_257645

noncomputable def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1

theorem area_of_triangle_AOB {k m : ℝ} 
  (a : ℝ) (b : ℝ) 
  (h1 : 0 < b)
  (h2 : b < a)
  (h3 : ellipse_eq a b)
  (h4 : a = 2)
  (h5 : b = real.sqrt 3)
  (h6 : ∃ x y : ℝ, x = 1 ∧ y = 3 / 2 ∧ ellipse_eq a b)
  (intersect_points : ∀ A B : ℝ × ℝ, (∃ x y, A = (x, y) ∧ ellipse_eq a b ∧ B = (x, y) ∧ ellipse_eq a b))
  (ellipse_pts : ∀ x1 x2 y1 y2, 
    let P := (x1 / a, y1 / b),
        Q := (x2 / a, y2 / b) in 
    ∀ x' y', ((x' = x1 ∧ y' = y1) ∨ (x' = x2 ∧ y' = y2)) → ellipse_eq a b)
  (h7 : ∀ x1 x2 y1 y2, 
    let P := (x1 / 2, y1 / real.sqrt 3),
        Q := (x2 / 2, y2 / real.sqrt 3) in
    (P.1 * Q.1 / 4 + P.2 * Q.2 / 3 = 0) ∧ real.sqrt (P.1^2 + P.2^2) = 1) :
  ∃ A B : ℝ × ℝ, area_of_triangle A B = real.sqrt 3 :=
sorry

end area_of_triangle_AOB_l257_257645


namespace number_of_meals_per_day_l257_257518

-- Define the conditions
def horses := 4
def oat_per_meal := 4
def total_oats := 96
def days := 3

-- Define the question, implementing the conditions
def meals_per_day (x : ℕ) : Prop := (days * (horses * oat_per_meal * x) = total_oats)

-- State the theorem
theorem number_of_meals_per_day : ∃ x : ℕ, meals_per_day x ∧ x = 2 :=
begin
  use 2,
  simp [meals_per_day, horses, oat_per_meal, total_oats, days],
  norm_num,
end

end number_of_meals_per_day_l257_257518


namespace total_number_of_numbers_l257_257700

-- Definitions using the conditions from the problem
def sum_of_first_4_numbers : ℕ := 4 * 4
def sum_of_last_4_numbers : ℕ := 4 * 4
def average_of_all_numbers (n : ℕ) : ℕ := 3 * n
def fourth_number : ℕ := 11
def total_sum_of_numbers : ℕ := sum_of_first_4_numbers + sum_of_last_4_numbers - fourth_number

-- Theorem stating the problem
theorem total_number_of_numbers (n : ℕ) : total_sum_of_numbers = average_of_all_numbers n → n = 7 :=
by {
  sorry
}

end total_number_of_numbers_l257_257700


namespace max_rectangle_area_l257_257633

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l257_257633


namespace eight_sided_dice_probability_l257_257543

noncomputable def probability_even_eq_odd (n : ℕ) : ℚ :=
  if h : n % 2 = 0 then
    let k := n / 2 in
    (nat.choose n k : ℚ) * (1/2)^n
  else 0

theorem eight_sided_dice_probability :
  probability_even_eq_odd 8 = 35/128 :=
by trivial

end eight_sided_dice_probability_l257_257543


namespace drivers_sufficiency_and_ivan_petrovich_departure_l257_257034

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l257_257034


namespace train_speed_l257_257908

/-- Given that a train crosses a pole in 12 seconds and the length of the train is 200 meters,
prove that the speed of the train in km/hr is 60. -/
theorem train_speed 
  (cross_time : ℝ) (train_length : ℝ)
  (H1 : cross_time = 12) (H2 : train_length = 200) : 
  let distance_km := train_length / 1000
      time_hr := cross_time / 3600
      speed := distance_km / time_hr in
  speed = 60 :=
by
  sorry

end train_speed_l257_257908


namespace minimum_sum_dimensions_l257_257891

def is_product (a b c : ℕ) (v : ℕ) : Prop :=
  a * b * c = v

def sum (a b c : ℕ) : ℕ :=
  a + b + c

theorem minimum_sum_dimensions : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ is_product a b c 3003 ∧ sum a b c = 45 :=
by
  sorry

end minimum_sum_dimensions_l257_257891


namespace sum_S16_over_S4_l257_257986

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a q : α) (n : ℕ) := a * q^n

def sum_of_first_n_terms (a q : α) (n : ℕ) : α :=
if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem sum_S16_over_S4
  (a q : α)
  (hq : q ≠ 1)
  (h8_over_4 : sum_of_first_n_terms a q 8 / sum_of_first_n_terms a q 4 = 3) :
  sum_of_first_n_terms a q 16 / sum_of_first_n_terms a q 4 = 15 :=
sorry

end sum_S16_over_S4_l257_257986


namespace no_solution_equation_l257_257319

theorem no_solution_equation (m : ℝ) : 
  ¬∃ x : ℝ, x ≠ 2 ∧ (x - 3) / (x - 2) = m / (2 - x) → m = 1 := 
by 
  sorry

end no_solution_equation_l257_257319


namespace concert_revenue_l257_257425

-- Defining the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := ticket_price_adult / 2
def attendees_adults : ℕ := 183
def attendees_children : ℕ := 28

-- Defining the total revenue calculation based on the conditions
def total_revenue : ℕ :=
  attendees_adults * ticket_price_adult +
  attendees_children * ticket_price_child

-- The theorem to prove the total revenue
theorem concert_revenue : total_revenue = 5122 := by
  sorry

end concert_revenue_l257_257425


namespace number_of_remainders_l257_257176

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem number_of_remainders (p : ℕ) (hp : p > 7 ∧ is_prime p) : 
  (∃ r : ℕ, r < 210 ∧ ∃ s : ℕ, p^3 ≡ r [MOD 210] ∧ ∀ t, t ≠ r → ¬ (∃ u : ℕ, p^3 ≡ t [MOD 210])) := 
sorry

example (H : Nat.prime > 7  → p : p > 7): 
  ∃ (r : ℕ), r < 210 ∧ 
  ∃ s : ℕ, p^3 ≡ r [MOD 210] :=
begin
  -- Given a prime p greater than 7,
  -- we assume there exist r remainders and converting the question to the finding of s numbers.
  sorry

statement helper := ∀ (p: ℕ(p: >7), number_of_remainders p := 
-- attach the concluded values to state.
    sorry  

end number_of_remainders_l257_257176


namespace solve_for_x_l257_257765

theorem solve_for_x : ∀ (x : ℝ), (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by
  intros x h
  sorry

end solve_for_x_l257_257765
